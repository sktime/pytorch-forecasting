from typing import Union, List, Dict

import math

import torch.nn as nn
import torch
import torch.nn.functional as F


class TimeDistributedInterpolation(nn.Module):
    def __init__(self, output_size: int, batch_first: bool = False, trainable: bool = False):
        super().__init__()
        self.output_size = output_size
        self.batch_first = batch_first
        self.trainable = trainable
        if self.trainable:
            self.mask = nn.Parameter(torch.zeros(self.output_size, dtype=torch.float32))
            self.gate = nn.Sigmoid()

    def interpolate(self, x):
        upsampled = F.interpolate(x.unsqueeze(1), self.output_size, mode="linear", align_corners=True).squeeze(1)
        if self.trainable:
            upsampled = upsampled * self.gate(self.mask.unsqueeze(0)) * 2.0
        return upsampled

    def forward(self, x):

        if len(x.size()) <= 2:
            return self.interpolate(x)

        # Squash samples and timesteps into a single axis
        x_reshape = x.contiguous().view(-1, x.size(-1))  # (samples * timesteps, input_size)

        y = self.interpolate(x_reshape)

        # We have to reshape Y
        if self.batch_first:
            y = y.contiguous().view(x.size(0), -1, y.size(-1))  # (samples, timesteps, output_size)
        else:
            y = y.view(-1, x.size(1), y.size(-1))  # (timesteps, samples, output_size)

        return y


class GLU(nn.Module):
    """Gated Linear Unit"""

    def __init__(self, input_size: int, hidden_size: int = None, dropout: float = None):
        super().__init__()

        if dropout is not None:
            self.dropout = nn.Dropout(dropout)
        else:
            self.dropout = dropout
        self.hidden_size = hidden_size or input_size
        self.fc1 = nn.Linear(input_size, self.hidden_size)
        self.fc2 = nn.Linear(input_size, self.hidden_size)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        if self.dropout is not None:
            x = self.dropout(x)
        sig = self.sigmoid(self.fc1(x))
        x = self.fc2(x)
        return torch.mul(sig, x)


class AddNorm(nn.Module):
    def __init__(self, input_size: int = None, trainable_add: bool = True):
        super().__init__()

        self.input_size = input_size
        self.trainable_add = trainable_add

        if self.trainable_add:
            self.mask = nn.Parameter(torch.zeros(self.input_size, dtype=torch.float))
            self.gate = nn.Sigmoid()
        self.norm = nn.LayerNorm(self.input_size)

    def forward(self, x, skip):
        if self.trainable_add:
            skip = skip * self.gate(self.mask) * 2.0
        output = self.norm(x + skip)
        return output


class GateAddNorm(nn.Module):
    def __init__(self, input_size: int, hidden_size: int = None, trainable_add: bool = True, dropout: float = 0.1):
        super().__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size or input_size
        self.dropout = dropout

        self.glu = GLU(self.input_size, hidden_size=self.hidden_size, dropout=self.dropout)
        self.add_norm = AddNorm(self.hidden_size, trainable_add=trainable_add)

    def forward(self, x, skip):
        output = self.glu(x)
        output = self.add_norm(output, skip)
        return output


class GatedResidualNetwork(nn.Module):
    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        output_size: int,
        dropout: float = 0.1,
        context_size: int = None,
        residual: bool = False,
    ):
        super().__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.context_size = context_size
        self.hidden_size = hidden_size
        self.dropout = dropout
        self.residual = residual

        if self.input_size != self.output_size and not self.residual:
            self.skip_layer = TimeDistributedInterpolation(self.output_size, batch_first=True, trainable=True)

        if self.input_size > 1 or self.hidden_size > 1:
            self.fc1 = nn.Linear(self.input_size, self.hidden_size)
            self.elu1 = nn.ELU()

            if self.context_size is not None:
                self.context = nn.Linear(self.context_size, self.hidden_size, bias=False)

            self.fc2 = nn.Linear(self.hidden_size, self.hidden_size)
            self.elu2 = nn.ELU()
            self.gate_norm = GateAddNorm(
                input_size=self.hidden_size, hidden_size=self.output_size, dropout=self.dropout
            )

    def forward(self, x, context=None, residual=None):
        if self.input_size != self.output_size and residual is None:
            residual = self.skip_layer(x)
        if residual is None:
            residual = x

        if self.input_size == 1 and self.hidden_size == 1:
            return residual

        x = self.fc1(x)
        if context is not None:
            context = self.context(context)
            x = x + context
        x = self.elu1(x)
        x = self.fc2(x)
        x = self.gate_norm(x, residual)
        return x


class VariableSelectionNetwork(nn.Module):
    def __init__(
        self,
        input_sizes: Dict[str, int],
        hidden_size: int,
        dropout: float = 0.1,
        context_size: int = None,
        single_variable_grns: Dict[str, GatedResidualNetwork] = {},
    ):
        """
        Calcualte weights for ``num_inputs`` variables  which are each of size ``input_size``
        """
        super().__init__()

        self.hidden_size = hidden_size
        self.input_sizes = input_sizes
        self.dropout = dropout
        self.context_size = context_size

        if self.num_inputs > 1:
            if self.context_size is not None:
                self.flattened_grn = GatedResidualNetwork(
                    self.input_size_total,
                    min(self.hidden_size, self.num_inputs),
                    self.num_inputs,
                    self.dropout,
                    self.context_size,
                    residual=True,
                )
            else:
                self.flattened_grn = GatedResidualNetwork(
                    self.input_size_total,
                    min(self.hidden_size, self.num_inputs),
                    self.num_inputs,
                    self.dropout,
                    residual=True,
                )

            # transform residual for flattened_grn
            self.residual_norm = nn.LayerNorm(self.num_inputs)

        self.single_variable_grns = nn.ModuleDict()
        for name, input_size in self.input_sizes.items():
            if name in single_variable_grns:
                self.single_variable_grns[name] = single_variable_grns[name]
            else:
                self.single_variable_grns[name] = GatedResidualNetwork(
                    input_size, min(input_size, self.hidden_size), self.hidden_size, self.dropout
                )

        self.softmax = nn.Softmax(dim=-1)

    @property
    def input_size_total(self):
        return sum(self.input_sizes.values())

    @property
    def num_inputs(self):
        return len(self.input_sizes)

    def forward(self, embedding: torch.Tensor, context: torch.Tensor = None):
        if self.num_inputs > 1:
            # transform single variables
            var_outputs = []
            variable_embedding_mean = []
            start = 0
            for name, input_size in self.input_sizes.items():
                # select slice of embedding belonging to a single input
                variable_embedding = embedding[..., start : (start + input_size)]
                variable_embedding_mean.append(variable_embedding.abs().mean(-1))
                var_outputs.append(self.single_variable_grns[name](variable_embedding))
                start += input_size
            var_outputs = torch.stack(var_outputs, axis=-1)

            # calculate variable weights
            # calculate residual/skip connection based on mean for each embedding and norm it element wise
            residual = self.residual_norm(torch.stack(variable_embedding_mean, dim=-1))
            sparse_weights = self.flattened_grn(embedding, context, residual=residual)
            sparse_weights = self.softmax(sparse_weights).unsqueeze(-2)

            outputs = var_outputs * sparse_weights
            outputs = outputs.mean(axis=-1)
        else:  # for one input, do not perform variable selection but just encoding
            outputs = self.single_variable_grns[0](embedding)  # fast forward if only one variable
            sparse_weights = torch.ones(outputs.size(0), outputs.size(1), 1, 1, device=outputs.device)
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
    def __init__(self, dropout: float = 0.0, scale: bool = True):
        super(ScaledDotProductAttention, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        self.softmax = nn.Softmax(dim=2)
        self.scale = scale

    def forward(self, q, k, v, mask=None):
        attn = torch.bmm(q, k.permute(0, 2, 1))  # query-key overlap

        if self.scale:
            dimension = torch.sqrt(torch.tensor(k.shape[-1]).to(torch.float32))
            attn = attn / dimension
        if mask is not None:
            attn = attn.masked_fill(mask, -1e9)
        attn = self.softmax(attn)
        attn = self.dropout(attn)
        output = torch.bmm(attn, v)
        return output, attn


class InterpretableMultiHeadAttention(nn.Module):
    def __init__(self, n_head: int, d_model: int, dropout: float = 0.0):
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

    def forward(self, q, k, v, mask=None) -> Tuple[torch.Tensor, torch.Tensor]:

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
