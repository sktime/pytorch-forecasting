from typing import Optional

import torch
from torch import nn
import torch.nn.functional as F


class TemporalLinear(nn.Module):
    def __init__(
        self,
        input_len: int,
        output_len: int,
        activation: Optional[str] = None,
        dropout: Optional[float] = 0,
    ):
        super().__init__()
        self.linear = nn.Linear(in_features=input_len, out_features=output_len)
        self.activation = None if activation is None else getattr(F, activation)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.linear(x.permute(0, 2, 1)).permute(0, 2, 1)
        x = x if self.activation is None else self.activation(x)
        x = self.dropout(x)
        return x


class TemporalResBlock(nn.Module):
    def __init__(
        self,
        input_len: int,
        input_size: int,
        activation: Optional[str] = None,
        dropout: Optional[float] = 0,
    ):
        super().__init__()
        self.temporal_linear = TemporalLinear(input_len, input_len, activation, dropout)
        self.norm = nn.LayerNorm(normalized_shape=(input_len, input_size))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        res = x
        x = self.temporal_linear(x)
        return self.norm(res + x)


class FeaturalResBlock(nn.Module):
    def __init__(
        self,
        input_len: int,
        input_size: int,
        hidden_size: int,
        output_size: int,
        activation: Optional[str] = "relu",
        dropout: Optional[float] = 0,
    ):
        super().__init__()
        self.linear1 = nn.Linear(in_features=input_size, out_features=hidden_size)
        self.linear2 = nn.Linear(in_features=hidden_size, out_features=output_size)
        self.res_linear = None
        if input_size != output_size:
            self.res_linear = nn.Linear(in_features=input_size, out_features=output_size)
        self.activation = getattr(F, activation)
        self.dropout = nn.Dropout(p=dropout)
        self.norm = nn.LayerNorm(normalized_shape=(input_len, output_size))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        res = x if self.res_linear is None else self.res_linear(x)
        x = self.linear1(x)
        x = self.activation(x)
        x = self.dropout(x)
        x = self.linear2(x)
        x = self.dropout(x)
        return self.norm(res + x)


class ConditionalFeaturalResBlock(nn.Module):
    def __init__(
        self,
        input_len: int,
        input_size: int,
        hidden_size: int,
        output_size: int,
        static_size: int,
        activation: Optional[str] = "relu",
        dropout: Optional[float] = 0,
    ):
        super().__init__()
        self.input_len = input_len
        self.static_block = FeaturalResBlock(1, static_size, hidden_size, hidden_size, activation, dropout)
        self.block = FeaturalResBlock(
            input_len,
            input_size + hidden_size,
            hidden_size,
            output_size,
            activation,
            dropout,
        )

    def forward(self, x: torch.Tensor, static: torch.Tensor) -> torch.Tensor:
        static = self.static_block(static.unsqueeze(1))
        static = torch.repeat_interleave(static, self.input_len, dim=1)
        x = torch.concat([x, static], dim=2)
        x = self.block(x)
        return x


class MixerBlock(nn.Module):
    def __init__(
        self,
        input_len: int,
        input_size: int,
        hidden_size: int,
        output_size: int,
        activation: str,
        dropout: float,
    ):
        super().__init__()
        self.temporal_res_block = TemporalResBlock(input_len, input_size, activation, dropout)
        self.ffwd_res_block = FeaturalResBlock(
            input_len,
            input_size,
            hidden_size,
            output_size,
            activation,
            dropout,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.temporal_res_block(x)
        x = self.ffwd_res_block(x)
        return x


class ConditionalMixerBlock(nn.Module):
    def __init__(
        self,
        input_len: int,
        input_size: int,
        hidden_size: int,
        output_size: int,
        static_size: int,
        activation: str,
        dropout: float,
    ):
        super().__init__()
        self.temporal_res_block = TemporalResBlock(input_len, input_size, activation, dropout)
        self.ffwd_res_block = ConditionalFeaturalResBlock(
            input_len,
            input_size,
            hidden_size,
            output_size,
            static_size,
            activation,
            dropout,
        )

    def forward(self, x: torch.Tensor, static: torch.Tensor) -> torch.Tensor:
        x = self.temporal_res_block(x)
        x = self.ffwd_res_block(x, static)
        return x


class TSMixerEncoder(nn.Module):
    def __init__(
        self,
        input_len: int,
        output_len: int,
        past_feat_size: int,
        future_feat_size: int,
        static_feat_size: int,
        hidden_size: int,
        activation: str,
        dropout: float,
        n_block: Optional[int] = 1,
    ):
        super().__init__()
        self.past_temporal_linear = TemporalLinear(input_len, output_len)
        self.past_featural_block = ConditionalFeaturalResBlock(
            input_len=output_len,
            input_size=past_feat_size,
            hidden_size=hidden_size,
            output_size=hidden_size,
            static_size=static_feat_size,
            activation=activation,
            dropout=dropout,
        )
        self.future_featural_block = ConditionalFeaturalResBlock(
            input_len=output_len,
            input_size=future_feat_size,
            hidden_size=hidden_size,
            output_size=hidden_size,
            static_size=static_feat_size,
            activation=activation,
            dropout=dropout,
        )
        self.blocks = nn.ModuleList(
            [
                ConditionalMixerBlock(
                    input_len=output_len,
                    input_size=(2 * hidden_size) if i == 0 else hidden_size,
                    hidden_size=hidden_size,
                    output_size=hidden_size,
                    static_size=static_feat_size,
                    activation=activation,
                    dropout=dropout,
                )
                for i in range(n_block)
            ]
        )

    def forward(
        self,
        past_feature: torch.Tensor,
        future_feature: torch.Tensor,
        static_feature: torch.Tensor,
    ) -> torch.Tensor:
        past_feature = self.past_temporal_linear(past_feature)
        past_feature = self.past_featural_block(past_feature, static_feature)
        future_feature = self.future_featural_block(future_feature, static_feature)
        x = torch.cat([past_feature, future_feature], dim=2)
        for block in self.blocks:
            x = block(x, static_feature)
        return x
