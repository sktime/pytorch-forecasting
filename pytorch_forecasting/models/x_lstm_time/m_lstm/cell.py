import torch
import torch.nn as nn
import math


class mLSTMCell(nn.Module):
    def __init__(self, input_size, hidden_size, dropout=0.2, layer_norm=True, device=None):
        super(mLSTMCell, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.layer_norm = layer_norm

        self.device = device if device is not None else torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.Wq = nn.Linear(input_size, hidden_size)
        self.Wk = nn.Linear(input_size, hidden_size)
        self.Wv = nn.Linear(input_size, hidden_size)

        self.Wi = nn.Linear(input_size, hidden_size)
        self.Wf = nn.Linear(input_size, hidden_size)
        self.Wo = nn.Linear(input_size, hidden_size)

        self.Wq.to(self.device)
        self.Wk.to(self.device)
        self.Wv.to(self.device)
        self.Wi.to(self.device)
        self.Wf.to(self.device)
        self.Wo.to(self.device)

        self.dropout = nn.Dropout(dropout)
        self.dropout.to(self.device)

        if layer_norm:
            self.ln_q = nn.LayerNorm(hidden_size)
            self.ln_k = nn.LayerNorm(hidden_size)
            self.ln_v = nn.LayerNorm(hidden_size)
            self.ln_i = nn.LayerNorm(hidden_size)
            self.ln_f = nn.LayerNorm(hidden_size)
            self.ln_o = nn.LayerNorm(hidden_size)

            self.ln_q.to(self.device)
            self.ln_k.to(self.device)
            self.ln_v.to(self.device)
            self.ln_i.to(self.device)
            self.ln_f.to(self.device)
            self.ln_o.to(self.device)

        self.sigmoid = nn.Sigmoid()
        self.tanh = nn.Tanh()

    def forward(self, x, h_prev, c_prev, n_prev):

        x = x.to(self.device)
        h_prev = h_prev.to(self.device)
        c_prev = c_prev.to(self.device)
        n_prev = n_prev.to(self.device)

        batch_size = x.size(0)
        assert x.dim() == 2, f"Input should be 2D (batch_size, input_size), got {x.dim()}D"
        assert h_prev.size() == (batch_size, self.hidden_size), f"h_prev shape mismatch: {h_prev.size()}"
        assert c_prev.size() == (batch_size, self.hidden_size), f"c_prev shape mismatch: {c_prev.size()}"
        assert n_prev.size() == (batch_size, self.hidden_size), f"n_prev shape mismatch: {n_prev.size()}"

        x = self.dropout(x)
        h_prev = self.dropout(h_prev)

        q = self.Wq(x)
        k = self.Wk(x) / math.sqrt(self.hidden_size)
        v = self.Wv(x)

        if self.layer_norm:
            q = self.ln_q(q)
            k = self.ln_k(k)
            v = self.ln_v(v)

        i = self.sigmoid(self.ln_i(self.Wi(x)) if self.layer_norm else self.Wi(x))
        f = self.sigmoid(self.ln_f(self.Wf(x)) if self.layer_norm else self.Wf(x))
        o = self.sigmoid(self.ln_o(self.Wo(x)) if self.layer_norm else self.Wo(x))

        k_expanded = k.unsqueeze(-1)
        v_expanded = v.unsqueeze(-2)

        kv_interaction = k_expanded @ v_expanded

        kv_sum = kv_interaction.sum(dim=1)

        c = f * c_prev + i * kv_sum
        n = f * n_prev + i * k

        epsilon = 1e-8
        normalized_n = n / (torch.norm(n, dim=-1, keepdim=True) + epsilon)
        h = o * self.tanh(c * normalized_n)

        return h, c, n

    def init_hidden(self, batch_size):
        """
        Initialize hidden, cell, and normalization states.
        """
        shape = (batch_size, self.hidden_size)
        return (
            torch.zeros(shape, device=self.device),
            torch.zeros(shape, device=self.device),
            torch.zeros(shape, device=self.device),
        )
