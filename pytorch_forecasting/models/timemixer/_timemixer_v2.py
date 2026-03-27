"""
Timemixer: 
----------------------------------------------------------------
"""

from typing import Any, Optional, Union
import warnings as warn

import torch
import torch.nn as nn
from torch.optim import Optimizer

from pytorch_forecasting.models.base._tslib_base_model_v2 import TslibBaseModel
from pytorch_forecasting.layers import (
    series_decomp,
    DataEmbedding_wo_pos,
    Normalize,
    PastDecomposableMixing
)

class TimeMixer(TslibBaseModel):
    """
    An implementation of the TimeMixer model for pytorch-forecasting-v2.
    """

    @classmethod
    def _pkg(cls):
        
        from pytorch_forecasting.models.timemixer._timemixer_pkg_v2 import TimeMixer_pkg_v2
        return TimeMixer_pkg_v2


    def __init__ (
        self,
        loss: nn.Module,
        enc_in: int = None,
        d_model: int = 32,
        embed: str = "timeF",
        dropout: float = 0.1,
        e_layers: int = 2,
        d_ff: int = 2048,
        down_sampling_layers: int = 3,
        down_sampling_window: int = 2,
        channel_independence: bool = False,
        moving_avg: int = 25,
        top_k: int = 5,
        decomp_method: str = "moving_avg", # can be dft_decomp too
        down_sampling_method: str = "avg",
        c_out: int = None,
        use_norm: int = 1,
        task_name: str = "long_term_forecast",
        logging_metrics: list[nn.Module] | None = None,
        optimizer: Optimizer | str | None = "adam",
        optimizer_params: dict | None = None,
        lr_scheduler: str | None = None,
        lr_scheduler_params: dict | None = None,
        metadata: dict | None = None,
        **kwargs: Any,
    ):
        
        super().__init__(
            loss=loss,
            logging_metrics=logging_metrics,
            optimizer=optimizer,
            optimizer_params=optimizer_params,
            lr_scheduler=lr_scheduler,
            lr_scheduler_params=lr_scheduler_params,
            metadata=metadata,
        )

        self.enc_in = enc_in or self.cont_dim
        self.d_model = d_model
        self.embed = embed
        self.dropout = dropout
        self.e_layers = e_layers
        self.d_ff = d_ff
        self.down_sampling_layers = down_sampling_layers
        self.down_sampling_window = down_sampling_window
        self.down_sampling_method = down_sampling_method
        self.channel_independence = channel_independence
        self.moving_avg = moving_avg
        self.top_k= top_k
        self.decomp_method = decomp_method
        self.c_out = c_out or self.target_dim
        self.use_norm = use_norm
        self.task_name = task_name
        self.freq = metadata["freq"]

        self.save_hyperparameters(ignore=["loss", "logging_metrics", "metadata"])
        self._init_network()

    def _init_network(self):
        
        self.preprocess = series_decomp(self.moving_avg)

        self.pdm_blocks = nn.ModuleList([PastDecomposableMixing(
            self.context_length,
            self.prediction_length,
            self.down_sampling_window,
            self.down_sampling_layers,
            self.d_model,
            self.d_ff,
            self.dropout,
            self.channel_independence,
            self.decomp_method,
            self.moving_avg,
            self.top_k,
        )
        for _ in range(self.e_layers)])


        if self.channel_independence:
            self.enc_embedding = DataEmbedding_wo_pos(1, self.d_model, self.embed, self.freq,
                                                      self.dropout)
        else:
            self.enc_embedding = DataEmbedding_wo_pos(self.enc_in, self.d_model, self.embed, self.freq,
                                                      self.dropout)
            
        self.normalize_layers = nn.ModuleList(
            [
                Normalize(self.enc_in, affine=True, non_norm=True if self.use_norm == 0 else False)
                for i in range(self.down_sampling_layers + 1)
            ]
        )

        self.predict_layers = nn.ModuleList(
                [
                    nn.Linear(
                        self.context_length // (self.down_sampling_window ** i),
                        self.prediction_length,
                    )
                    for i in range(self.down_sampling_layers + 1)
                ]
            )
        
        if self.channel_independence:
            self.projection_layer = nn.Linear(
                self.d_model, 1, bias=True)
        else:
            self.projection_layer = nn.Linear(
                self.d_model, self.c_out, bias=True)

            self.out_res_layers = torch.nn.ModuleList([
                torch.nn.Linear(
                    self.context_length // (self.down_sampling_window ** i),
                    self.context_length // (self.down_sampling_window ** i),
                )
                for i in range(self.down_sampling_layers + 1)
            ])

            self.regression_layers = torch.nn.ModuleList(
                [
                    torch.nn.Linear(
                        self.context_length // (self.down_sampling_window ** i),
                        self.prediction_length,
                    )
                    for i in range(self.down_sampling_layers + 1)
                ]
            )
        
    def _out_projection(self, 
                        dec_out, 
                        i, 
                        out_res):
        
        dec_out = self.projection_layer(dec_out)
        out_res = out_res.permute(0, 2, 1)
        out_res = self.out_res_layers[i](out_res)
        out_res = self.regression_layers[i](out_res).permute(0, 2, 1)
        dec_out = dec_out + out_res
        return dec_out

    def _pre_enc(self, 
                 x_list):
        if self.channel_independence:
            return (x_list, None)
        else:
            out1_list = []
            out2_list = []
            for x in x_list:
                x_1, x_2 = self.preprocess(x)
                out1_list.append(x_1)
                out2_list.append(x_2)
            return (out1_list, out2_list)

    def _multi_scale_process_inputs(self, 
                                    x_enc, 
                                    x_mark_enc):
        if self.down_sampling_method == 'max':
            down_pool = torch.nn.MaxPool1d(self.down_sampling_window, return_indices=False)
        elif self.down_sampling_method == 'avg':
            down_pool = torch.nn.AvgPool1d(self.down_sampling_window)
        elif self.down_sampling_method == 'conv':
            padding = 1 if torch.__version__ >= '1.5.0' else 2
            down_pool = nn.Conv1d(in_channels=self.enc_in, out_channels=self.enc_in,
                                  kernel_size=3, padding=padding,
                                  stride=self.down_sampling_window,
                                  padding_mode='circular',
                                  bias=False)
        else:
            return x_enc, x_mark_enc
        # B,T,C -> B,C,T
        x_enc = x_enc.permute(0, 2, 1)

        x_enc_ori = x_enc
        x_mark_enc_mark_ori = x_mark_enc

        x_enc_sampling_list = []
        x_mark_sampling_list = []
        x_enc_sampling_list.append(x_enc.permute(0, 2, 1))
        x_mark_sampling_list.append(x_mark_enc)

        for i in range(self.down_sampling_layers):
            x_enc_sampling = down_pool(x_enc_ori)

            x_enc_sampling_list.append(x_enc_sampling.permute(0, 2, 1))
            x_enc_ori = x_enc_sampling

            if x_mark_enc is not None:
                x_mark_sampling_list.append(x_mark_enc_mark_ori[:, ::self.down_sampling_window, :])
                x_mark_enc_mark_ori = x_mark_enc_mark_ori[:, ::self.down_sampling_window, :]

        x_enc = x_enc_sampling_list
        x_mark_enc = x_mark_sampling_list if x_mark_enc is not None else None

        return x_enc, x_mark_enc
    
    def _future_multi_mixing(self, 
                             B, 
                             enc_out_list, 
                             x_list):
        dec_out_list = []
        if self.channel_independence:
            x_list = x_list[0]
            for i, enc_out in zip(range(len(x_list)), enc_out_list):
                dec_out = self.predict_layers[i](enc_out.permute(0, 2, 1)).permute(
                    0, 2, 1)  # align temporal dimension
                dec_out = self.projection_layer(dec_out)
                dec_out = dec_out.reshape(B, self.c_out, self.prediction_length).permute(0, 2, 1).contiguous()
                dec_out_list.append(dec_out)

        else:
            for i, enc_out, out_res in zip(range(len(x_list[0])), enc_out_list, x_list[1]):
                dec_out = self.predict_layers[i](enc_out.permute(0, 2, 1)).permute(
                    0, 2, 1)  # align temporal dimension
                dec_out = self._out_projection(dec_out, i, out_res)
                dec_out_list.append(dec_out)

        return dec_out_list
        
    def _forecast(self, x: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:

        x_enc = x["history_cont"]
        x_dec = x["future_cont"]

        if x_enc.shape[-1] != x_dec.shape[-1]:
            diff = x_enc.shape[-1] - x_dec.shape[-1]
            if diff > 0:
                x_dec = torch.nn.functional.pad(x_dec, (0, diff))
            else:
                x_dec = x_dec[..., : x_enc.shape[-1]]

        if self.embed == "timeF":
            x_mark_enc = x["history_time_idx"].unsqueeze(-1).float()
            x_mark_dec = x["future_time_idx"].unsqueeze(-1).float()
        else:
            x_mark_enc = None
            x_mark_dec = None
        
        x_enc, x_mark_enc = self._multi_scale_process_inputs(x_enc, x_mark_enc)
        # for i, x, x_mark in zip(range(len(x_list[0])), x_list[0], x_mark_list):
        #         print(f"x shape : {x.shape} and x_mark shape : {x_mark.shape}")
        x_list = []
        x_mark_list = []
        if x_mark_enc is not None:
            for i, x, x_mark in zip(range(len(x_enc)), x_enc, x_mark_enc):
                B, T, N = x.size()
                x = self.normalize_layers[i](x, 'norm')
                if self.channel_independence:
                    x = x.permute(0, 2, 1).contiguous().reshape(B * N, T, 1)
                    x_list.append(x)
                    x_mark = x_mark.repeat(N, 1, 1)
                    x_mark_list.append(x_mark)
                else:
                    x_list.append(x)
                    x_mark_list.append(x_mark)
        
        else:
            for i, x in zip(range(len(x_enc)), x_enc, ):
                B, T, N = x.size()
                x = self.normalize_layers[i](x, 'norm')
                if self.channel_independence:
                    x = x.permute(0, 2, 1).contiguous().reshape(B * N, T, 1)
                x_list.append(x)
        print("Iter-1")
        for i, x, x_mark in zip(range(len(x_list[0])), x_list[0], x_mark_list):
            print(f"x shape : {x.shape} and x_mark shape : {x_mark.shape}")
        enc_out_list = []
        x_list = self._pre_enc(x_list)
        print("Iter-2")
        for i, x, x_mark in zip(range(len(x_list[0])), x_list[0], x_mark_list):
            print(f"x shape : {x.shape} and x_mark shape : {x_mark.shape}")
        if x_mark_enc is not None:
            for i, x, x_mark in zip(range(len(x_list[0])), x_list[0], x_mark_list):
                print(f"x shape : {x.shape} and x_mark shape : {x_mark.shape}")
                if x.shape[-2] != x_mark.shape[-2]:
                    diff = x_mark.shape[-2] - x.shape[-2]
                    if diff > 0:
                        x_mark = x_mark [: , :-diff, :]
                    else:
                        x_enc = x_enc [:, :diff, :]
                enc_out = self.enc_embedding(x, x_mark)  # [B,T,C]
                enc_out_list.append(enc_out)
        else:
            for i, x in zip(range(len(x_list[0])), x_list[0]):
                enc_out = self.enc_embedding(x, None)  # [B,T,C]
                enc_out_list.append(enc_out)
        
        for i in range(self.e_layers):
            enc_out_list = self.pdm_blocks[i](enc_out_list)

        # Future Multipredictor Mixing as decoder for future
        dec_out_list = self._future_multi_mixing(B, enc_out_list, x_list)

        dec_out = torch.stack(dec_out_list, dim=-1).sum(-1)
        dec_out = self.normalize_layers[0](dec_out, 'denorm')
        return dec_out

    def forward(self, x: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:

        out = self._forecast(x)
        prediction = out[:, : self.prediction_length, :]

        if "target_scale" in x:
            prediction = self.transform_output(prediction, x["target_scale"])

        return {"prediction": prediction}
        
