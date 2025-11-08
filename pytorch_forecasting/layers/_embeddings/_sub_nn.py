from typing import Union

import torch
import torch.nn as nn


class embedding_cat_variables(nn.Module):
    # at the moment cat_past and cat_fut together
    def __init__(self, seq_len: int, lag: int, d_model: int, emb_dims: list, device):
        """Class for embedding categorical variables, adding 3 positional variables during forward

        Parameters
        ----------
        seq_len: int
            length of the sequence (sum of past and future steps)
        lag: (int):
            number of future step to be predicted
        hidden_size: int
            dimension of all variables after they are embedded
        emb_dims: list
            size of the dictionary for embedding. One dimension for each categorical variable
        device : torch.device
        """  # noqa: E501
        super().__init__()
        self.seq_len = seq_len
        self.lag = lag
        self.device = device
        self.cat_embeds = emb_dims + [seq_len, lag + 1, 2]  #
        self.cat_n_embd = nn.ModuleList(
            [nn.Embedding(emb_dim, d_model) for emb_dim in self.cat_embeds]
        )

    def forward(
        self, x: Union[torch.Tensor, int], device: torch.device
    ) -> torch.Tensor:
        """All components of x are concatenated with 3 new variables for data augmentation, in the order:

        - pos_seq: assign at each step its time-position
        - pos_fut: assign at each step its future position. 0 if it is a past step
        - is_fut: explicit for each step if it is a future(1) or past one(0)

        Parameters
        ----------
            x: torch.Tensor
                `[bs, seq_len, num_vars]`

        Returns
        ------
            torch.Tensor:
                `[bs, seq_len, num_vars+3, n_embd]`
        """  # noqa: E501
        if isinstance(x, int):
            no_emb = True
            B = x
        else:
            no_emb = False
            B, _, _ = x.shape

        pos_seq = self.get_pos_seq(bs=B).to(device)
        pos_fut = self.get_pos_fut(bs=B).to(device)
        is_fut = self.get_is_fut(bs=B).to(device)

        if no_emb:
            cat_vars = torch.cat((pos_seq, pos_fut, is_fut), dim=2)
        else:
            cat_vars = torch.cat((x, pos_seq, pos_fut, is_fut), dim=2)
        cat_vars = cat_vars.long()
        cat_n_embd = self.get_cat_n_embd(cat_vars)
        return cat_n_embd

    def get_pos_seq(self, bs):
        pos_seq = torch.arange(0, self.seq_len)
        pos_seq = pos_seq.repeat(bs, 1).unsqueeze(2).to(self.device)
        return pos_seq

    def get_pos_fut(self, bs):
        pos_fut = torch.cat(
            (
                torch.zeros((self.seq_len - self.lag), dtype=torch.long),
                torch.arange(1, self.lag + 1),
            )
        )
        pos_fut = pos_fut.repeat(bs, 1).unsqueeze(2).to(self.device)
        return pos_fut

    def get_is_fut(self, bs):
        is_fut = torch.cat(
            (
                torch.zeros((self.seq_len - self.lag), dtype=torch.long),
                torch.ones((self.lag), dtype=torch.long),
            )
        )
        is_fut = is_fut.repeat(bs, 1).unsqueeze(2).to(self.device)
        return is_fut

    def get_cat_n_embd(self, cat_vars):
        cat_n_embd = torch.Tensor().to(cat_vars.device)
        for index, layer in enumerate(self.cat_n_embd):
            emb = layer(cat_vars[:, :, index])
            cat_n_embd = torch.cat((cat_n_embd, emb.unsqueeze(2)), dim=2)
        return cat_n_embd
