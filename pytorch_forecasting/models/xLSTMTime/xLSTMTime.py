import torch
import torch.nn as nn
from typing import Optional, Tuple, Union, Literal
from mLSTM.network import mLSTMNetwork
from sLSTM.network import sLSTMNetwork


class SeriesDecomposition(nn.Module):
    """Implements series decomposition using learnable moving averages."""

    def __init__(self, kernel_size: int):
        super(SeriesDecomposition, self).__init__()
        self.kernel_size = kernel_size
        self.padding = kernel_size // 2
        self.avg_pool = nn.AvgPool1d(kernel_size=kernel_size, stride=1, padding=self.padding)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Decomposes input series into trend and seasonal components.

        Args:
            x: Input tensor of shape (batch_size, seq_len, n_features)

        Returns:
            Tuple of (trend_component, seasonal_component)
        """

        batch_size, seq_len, n_features = x.shape
        x_reshaped = x.reshape(batch_size * n_features, 1, seq_len)

        trend = self.avg_pool(x_reshaped)

        trend = trend.reshape(batch_size, seq_len, n_features)
        seasonal = x - trend

        return trend, seasonal


class xLSTMTime(nn.Module):
    """
    Implementation of xLSTMTime architecture for time series forecasting.
    """

    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        output_size: int,
        xlstm_type: Literal["slstm", "mlstm"],
        num_layers: int = 1,
        decomposition_kernel: int = 25,
        dropout: float = 0.1,
        device: Optional[torch.device] = None,
    ):
        """
        Initialize xLSTMTime model.

        Args:
            input_size: Number of input features
            hidden_size: Size of hidden layers
            output_size: Number of output features
            xlstm_type: Type of LSTM to use ('slstm' or 'mlstm')
            num_layers: Number of LSTM layers
            decomposition_kernel: Kernel size for series decomposition
            dropout: Dropout rate
            device: Torch device to use
        """
        super(xLSTMTime, self).__init__()

        if xlstm_type not in ["slstm", "mlstm"]:
            raise ValueError("xlstm_type must be either 'slstm' or 'mlstm'")

        self.xlstm_type = xlstm_type
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.decomposition = SeriesDecomposition(decomposition_kernel)
        self.input_linear = nn.Linear(input_size * 2, hidden_size)

        self.batch_norm = nn.BatchNorm1d(hidden_size)

        if xlstm_type == "mlstm":
            self.lstm = mLSTMNetwork(
                input_size=hidden_size,
                hidden_size=hidden_size,
                num_layers=num_layers,
                output_size=hidden_size,
                dropout=dropout,
                device=self.device,
            )
        else:  # slstm
            self.lstm = sLSTMNetwork(
                input_size=hidden_size,
                hidden_size=hidden_size,
                num_layers=num_layers,
                output_size=hidden_size,
                dropout=dropout,
                device=self.device,
            )
        self.output_linear = nn.Linear(hidden_size, output_size)

        self.instance_norm = nn.InstanceNorm1d(output_size)

    def forward(
        self,
        x: torch.Tensor,
        hidden_states: Optional[
            Union[Tuple[torch.Tensor, torch.Tensor], Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]
        ] = None,
    ) -> Tuple[torch.Tensor, Union[Tuple[torch.Tensor, torch.Tensor], Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]]:
        """
        Forward pass through the network.

        Args:
            x: Input tensor of shape (batch_size, seq_len, input_size)
            hidden_states: Initial hidden states for LSTM

        Returns:
            Tuple of (output, hidden_states)
        """
        batch_size, seq_len, _ = x.shape

        trend, seasonal = self.decomposition(x)

        x = torch.cat([trend, seasonal], dim=-1)

        x = self.input_linear(x)

        # Reshape for batch norm
        x = x.transpose(1, 2)
        x = self.batch_norm(x)
        x = x.transpose(1, 2)

        if hidden_states is None:
            hidden_states = self.lstm.init_hidden(batch_size)

        x = x.transpose(0, 1)
        output, hidden_states = self.lstm(x, *hidden_states)

        if isinstance(output, tuple):
            output = output[0]

        if output.dim() == 2:
            output = output.unsqueeze(0)

        output = self.output_linear(output)

        output = output.transpose(1, 2)
        output = self.instance_norm(output)
        output = output.transpose(1, 2)

        return output, hidden_states

    def predict(
        self,
        x: torch.Tensor,
        hidden_states: Optional[
            Union[Tuple[torch.Tensor, torch.Tensor], Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]
        ] = None,
    ) -> torch.Tensor:
        """
        Make predictions using the model.

        Args:
            x: Input tensor
            hidden_states: Optional initial hidden states

        Returns:
            Predictions tensor
        """
        output, _ = self.forward(x, hidden_states)
        return output
