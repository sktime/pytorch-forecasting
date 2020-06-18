from typing import Union, List, Dict

import math

import torch.nn as nn
import torch


class GLU(nn.Module):
    """Gated Linear Unit"""

    def __init__(self, input_size, hidden_size=None, dropout=None):
        super().__init__()

        if dropout is not None:
            self.dropout = nn.Dropout(dropout)
        else:
            self.dropout = dropout

        if hidden_size is None:
            self.hidden_size = input_size
        else:
            self.hidden_size = hidden_size
        self.fc1 = nn.Linear(input_size, self.hidden_size)
        self.fc2 = nn.Linear(input_size, self.hidden_size)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        if self.dropout is not None:
            x = self.dropout(x)
        sig = self.sigmoid(self.fc1(x))
        x = self.fc2(x)
        return torch.mul(sig, x)


class GateAddNorm(nn.Module):
    def __init__(self, input_size, hidden_size=None, dropout: float = 0.1):
        super().__init__()

        self.input_size = input_size
        if hidden_size is None:
            self.hidden_size = input_size
        else:
            self.hidden_size = hidden_size
        self.dropout = dropout

        self.glu = GLU(self.input_size, hidden_size=self.hidden_size, dropout=self.dropout)

        self.norm = nn.LayerNorm(self.hidden_size)

    def forward(self, x, skip):
        output = self.norm(self.glu(x) + skip)
        return output


class GatedResidualNetwork(nn.Module):
    def __init__(
        self, input_size, hidden_size, output_size, dropout=0.1, context=None,
    ):
        super().__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.context = context
        self.hidden_size = hidden_size
        self.dropout = dropout

        if self.input_size != self.output_size:
            self.skip_layer = nn.Linear(self.input_size, self.output_size)

        self.fc1 = nn.Linear(self.input_size, self.hidden_size)
        self.elu1 = nn.ELU()

        if self.context is not None:
            self.context = nn.Linear(self.context, self.hidden_size)

        self.fc2 = nn.Linear(self.hidden_size, self.output_size)
        self.elu2 = nn.ELU()
        self.gate_norm = GateAddNorm(self.output_size, dropout=self.dropout)

    def forward(self, x, context=None):
        if self.input_size != self.output_size:
            residual = self.skip_layer(x)
        else:
            residual = x

        x = self.fc1(x)
        if context is not None:
            context = self.context(context)
            x = x + context
        x = self.elu1(x)
        x = self.fc2(x)
        x = self.gate_norm(x, residual)
        return x


class VariableSelectionNetwork(nn.Module):
    def __init__(self, input_size, num_inputs, hidden_size, dropout=0.1, context=None):
        """
        Calcualte weights for ``num_inputs`` variables  which are each of size ``input_size``
        """
        super().__init__()

        self.hidden_size = hidden_size
        self.input_size = input_size
        self.num_inputs = num_inputs
        self.dropout = dropout
        self.context = context

        if self.context is not None:
            self.flattened_grn = GatedResidualNetwork(
                self.num_inputs * self.input_size, self.hidden_size, self.num_inputs, self.dropout, self.context,
            )
        else:
            self.flattened_grn = GatedResidualNetwork(
                self.num_inputs * self.input_size, self.hidden_size, self.num_inputs, self.dropout,
            )

        self.single_variable_grns = nn.ModuleList()
        for _ in range(self.num_inputs):
            self.single_variable_grns.append(
                GatedResidualNetwork(self.input_size, self.hidden_size, self.hidden_size, self.dropout,)
            )

        self.softmax = nn.Softmax(dim=-1)

    def forward(self, embedding: torch.Tensor, context: torch.Tensor = None):
        sparse_weights = self.flattened_grn(embedding, context)
        sparse_weights = self.softmax(sparse_weights).unsqueeze(-2)

        var_outputs = []
        for i in range(self.num_inputs):
            # select slice of embedding belonging to a single input
            var_outputs.append(
                self.single_variable_grns[i](embedding[..., (i * self.input_size) : (i + 1) * self.input_size])
            )
        var_outputs = torch.stack(var_outputs, axis=-1)
        outputs = var_outputs * sparse_weights
        outputs = outputs.sum(axis=-1)
        return outputs, sparse_weights


class PositionalEncoder(torch.nn.Module):
    def __init__(self, d_model, max_seq_len=160):
        super().__init__()
        assert d_model % 2 == 0, "model dimension has to be multiple of 2 (encode sin(pos) and cos(pos))"
        self.d_model = d_model
        pe = torch.zeros(max_seq_len, d_model)
        for pos in range(max_seq_len):
            for i in range(0, d_model, 2):
                pe[pos, i] = math.sin(pos / (10000 ** ((2 * i) / d_model)))
                pe[pos, i + 1] = math.cos(pos / (10000 ** ((2 * (i + 1)) / d_model)))
        pe = pe.unsqueeze(0)
        self.register_buffer("pe", pe)

    def forward(self, x):
        with torch.no_grad():
            x = x * math.sqrt(self.d_model)
            seq_len = x.size(0)
            pe = self.pe[:, :seq_len].view(seq_len, 1, self.d_model)
            x = x + pe
            return x


class ScaledDotProductAttention(nn.Module):
    def __init__(self, dropout=0.0, scale=True):
        super(ScaledDotProductAttention, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        self.softmax = nn.Softmax(dim=2)
        self.scale = scale

    def forward(self, q, k, v, mask=None):
        attn = torch.bmm(q, k.permute(0, 2, 1))

        if self.scale:
            dimention = torch.sqrt(torch.tensor(k.shape[-1]).to(torch.float32))
            attn = attn / dimention

        if mask is not None:
            attn = attn.masked_fill(mask == 0, -1e9)

        attn = self.softmax(attn)
        attn = self.dropout(attn)

        output = torch.bmm(attn, v)

        return output, attn


class InterpretableMultiHeadAttention(nn.Module):
    def __init__(self, n_head, d_model, dropout=0.0):
        super(InterpretableMultiHeadAttention, self).__init__()

        self.n_head = n_head
        self.d_model = d_model
        self.d_k = self.d_q = self.d_v = d_model // n_head
        self.dropout = nn.Dropout(p=dropout)

        self.v_layer = nn.Linear(self.d_model, self.d_v, bias=False)
        self.q_layers = nn.ModuleList([nn.Linear(self.d_model, self.d_q, bias=False) for _ in range(self.n_head)])
        self.k_layers = nn.ModuleList([nn.Linear(self.d_model, self.d_k, bias=False) for _ in range(self.n_head)])
        self.attention = ScaledDotProductAttention()
        self.w_h = nn.Linear(self.d_v, self.d_model, bias=False)

        self.init_weights()

    def init_weights(self):
        for name, p in self.named_parameters():
            if "bias" not in name:
                torch.nn.init.xavier_uniform_(p)
            else:
                torch.nn.init.zeros_(p)

    def forward(self, q, k, v, mask=None):

        heads = []
        attns = []
        for i in range(self.n_head):
            qs = self.q_layers[i](q)
            ks = self.k_layers[i](k)
            vs = self.v_layer(v)
            head, attn = self.attention(qs, ks, vs, mask)
            head_dropout = self.dropout(head)
            heads.append(head_dropout)
            attns.append(attn)

        head = torch.stack(heads, dim=2) if self.n_head > 1 else heads[0]
        attn = torch.stack(attns, dim=2)

        outputs = torch.mean(head, dim=2) if self.n_head > 1 else head
        outputs = self.w_h(outputs)
        outputs = self.dropout(outputs)

        return outputs, attn
