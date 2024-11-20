import pytest
import torch
import torch.nn as nn
from typing import Tuple, List

from sympy.stats.sampling.sample_numpy import numpy

from pytorch_forecasting.models.xLSTMTime.mLSTM.cell import mLSTMCell
from pytorch_forecasting.models.xLSTMTime.mLSTM.layer import mLSTMLayer
from pytorch_forecasting.models.xLSTMTime.mLSTM.network import mLSTMNetwork
from pytorch_forecasting.models.xLSTMTime.sLSTM.cell import sLSTMCell
from pytorch_forecasting.models.xLSTMTime.sLSTM.layer import sLSTMLayer
from pytorch_forecasting.models.xLSTMTime.sLSTM.network import sLSTMNetwork
from pytorch_forecasting.models.xLSTMTime.xLSTMTime import xLSTMTime, SeriesDecomposition


# Fixtures for common test parameters
@pytest.fixture
def device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


@pytest.fixture
def batch_size():
    return 32


@pytest.fixture
def seq_length():
    return 24


@pytest.fixture
def input_size():
    return 10


@pytest.fixture
def hidden_size():
    return 64


@pytest.fixture
def output_size():
    return 5


@pytest.fixture
def sample_input(batch_size, seq_length, input_size, device):
    return torch.randn(batch_size, seq_length, input_size).to(device)


# Test Series Decomposition
class TestSeriesDecomposition:
    def test_initialization(self):
        kernel_size = 25
        decomp = SeriesDecomposition(kernel_size)
        assert decomp.kernel_size == kernel_size
        assert decomp.padding == kernel_size // 2

    def test_forward_shape(self, sample_input):
        kernel_size = 25
        decomp = SeriesDecomposition(kernel_size)
        trend, seasonal = decomp(sample_input)

        assert trend.shape == sample_input.shape
        assert seasonal.shape == sample_input.shape

    def test_decomposition_sum(self, sample_input):
        kernel_size = 25
        decomp = SeriesDecomposition(kernel_size)
        trend, seasonal = decomp(sample_input)

        # Check if trend + seasonal approximately equals input
        torch.testing.assert_close(trend + seasonal, sample_input, rtol=1e-4, atol=1e-4)


# Test mLSTM Components
class TestMLSTM:
    def test_cell_initialization(self, input_size, hidden_size, device):
        cell = mLSTMCell(input_size, hidden_size, device=device)
        assert cell.input_size == input_size
        assert cell.hidden_size == hidden_size

    def test_cell_forward(self, batch_size, input_size, hidden_size, device):
        cell = mLSTMCell(input_size, hidden_size, device=device)
        x = torch.randn(1, batch_size, input_size).to(device)  # Add sequence dimension
        h_prev = torch.randn(batch_size, hidden_size).to(device)
        c_prev = torch.randn(batch_size, hidden_size).to(device)
        n_prev = torch.randn(batch_size, hidden_size).to(device)

        h, c, n = cell(x[0], h_prev, c_prev, n_prev)  # Use first timestep
        assert h.shape == (batch_size, hidden_size)
        assert c.shape == (batch_size, hidden_size)
        assert n.shape == (batch_size, hidden_size)

    def test_layer_initialization(self, input_size, hidden_size, device):
        layer = mLSTMLayer(input_size, hidden_size, num_layers=2, device=device)
        assert layer.input_size == input_size
        assert layer.hidden_size == hidden_size
        assert len(layer.cells) == 2

    def test_network_forward(self, sample_input, input_size, hidden_size, output_size, device):
        network = mLSTMNetwork(input_size, hidden_size, num_layers=2, output_size=output_size, device=device)
        # Transpose input to seq_len, batch_size, input_size
        x = sample_input.transpose(0, 1)
        output, (h, c, n) = network(x)
        # Transpose output back to batch_size, seq_len, output_size
        output = output.transpose(0, 1)
        assert output.shape == (output_size, sample_input.shape[0])


# Test sLSTM Components
class TestSLSTM:
    def test_cell_initialization(self, input_size, hidden_size, device):
        cell = sLSTMCell(input_size, hidden_size, device=device)
        assert cell.input_size == input_size
        assert cell.hidden_size == hidden_size

    def test_cell_forward(self, batch_size, input_size, hidden_size, device):
        cell = sLSTMCell(input_size, hidden_size, device=device)
        x = torch.randn(1, batch_size, input_size).to(device)  # Add sequence dimension
        h_prev = torch.randn(batch_size, hidden_size).to(device)
        c_prev = torch.randn(batch_size, hidden_size).to(device)

        h, c = cell(x[0], h_prev, c_prev)  # Use first timestep
        assert h.shape == (batch_size, hidden_size)
        assert c.shape == (batch_size, hidden_size)

    def test_layer_initialization(self, input_size, hidden_size, device):
        layer = sLSTMLayer(input_size, hidden_size, num_layers=2, device=device)
        assert layer.input_size == input_size
        assert layer.hidden_size == hidden_size
        assert len(layer.cells) == 2

    def test_network_forward(self, sample_input, input_size, hidden_size, output_size, device):
        network = sLSTMNetwork(input_size, hidden_size, num_layers=2, output_size=output_size, device=device)
        # Transpose input to seq_len, batch_size, input_size
        x = sample_input.transpose(0, 1)
        output, (h, c) = network(x)
        # Transpose output back to batch_size, seq_len, output_size
        output = output.transpose(0, 1)
        assert output.shape == (output_size, sample_input.shape[0])


# Test xLSTMTime
class TestXLSTMTime:
    @pytest.mark.parametrize("xlstm_type", ["mlstm", "slstm"])
    def test_initialization(self, input_size, hidden_size, output_size, xlstm_type, device):
        model = xLSTMTime(
            input_size=input_size,
            hidden_size=hidden_size,
            output_size=output_size,
            xlstm_type=xlstm_type,
            device=device,
        )
        assert isinstance(model.decomposition, SeriesDecomposition)
        assert isinstance(model.input_linear, nn.Linear)
        assert isinstance(model.output_linear, nn.Linear)

    @pytest.mark.parametrize("xlstm_type", ["mlstm", "slstm"])
    def test_forward(self, sample_input, input_size, hidden_size, output_size, xlstm_type, device):
        model = xLSTMTime(
            input_size=input_size,
            hidden_size=hidden_size,
            output_size=output_size,
            xlstm_type=xlstm_type,
            device=device,
        )
        output, hidden_states = model(sample_input)
        # Check output shape is batch_size, seq_len, output_size
        assert output.shape == (1, sample_input.shape[0], output_size)

        if xlstm_type == "mlstm":
            assert len(hidden_states) == 3  # h, c, n for mLSTM
            h, c, n = hidden_states
            assert h.shape == (1, sample_input.shape[0], hidden_size)
            assert c.shape == (1, sample_input.shape[0], hidden_size)
            assert n.shape == (1, sample_input.shape[0], hidden_size)
        else:
            assert len(hidden_states) == 2  # h, c for sLSTM
            h, c = hidden_states
            assert torch.stack(h).shape == (1, sample_input.shape[0], hidden_size)
            assert torch.stack(c).shape == (1, sample_input.shape[0], hidden_size)

    @pytest.mark.parametrize("xlstm_type", ["mlstm", "slstm"])
    def test_predict(self, sample_input, input_size, hidden_size, output_size, xlstm_type, device):
        model = xLSTMTime(
            input_size=input_size,
            hidden_size=hidden_size,
            output_size=output_size,
            xlstm_type=xlstm_type,
            device=device,
        )
        predictions = model.predict(sample_input)
        assert predictions.shape == (1, sample_input.shape[0], output_size)

    def test_invalid_xlstm_type(self, input_size, hidden_size, output_size, device):
        with pytest.raises(ValueError, match="xlstm_type must be either 'slstm' or 'mlstm'"):
            xLSTMTime(
                input_size=input_size,
                hidden_size=hidden_size,
                output_size=output_size,
                xlstm_type="invalid_type",
                device=device,
            )


# Test edge cases and error handling
class TestEdgeCases:
    @pytest.mark.parametrize("xlstm_type", ["mlstm", "slstm"])
    def test_single_sequence_length(self, batch_size, input_size, hidden_size, output_size, xlstm_type, device):
        model = xLSTMTime(
            input_size=input_size,
            hidden_size=hidden_size,
            output_size=output_size,
            xlstm_type=xlstm_type,
            device=device,
        )
        single_step = torch.randn(batch_size, 1, input_size).to(device)
        output, hidden_states = model(single_step)
        assert output.shape == (1, batch_size, output_size)

        if xlstm_type == "mlstm":
            h, c, n = hidden_states
            assert h.shape == (1, batch_size, hidden_size)
            assert c.shape == (1, batch_size, hidden_size)
            assert n.shape == (1, batch_size, hidden_size)
        else:  # slstm
            h, c = hidden_states
            assert torch.stack(h).shape == (1, batch_size, hidden_size)
            assert torch.stack(c).shape == (1, batch_size, hidden_size)

    @pytest.mark.parametrize("xlstm_type", ["mlstm", "slstm"])
    def test_input_nan_handling(self, batch_size, input_size, hidden_size, output_size, xlstm_type, device):
        """Test model behavior with NaN inputs"""
        model = xLSTMTime(
            input_size=input_size,
            hidden_size=hidden_size,
            output_size=output_size,
            xlstm_type=xlstm_type,
            device=device,
        )

        # Create input with some NaN values
        nan_input = torch.randn(batch_size, 24, input_size).to(device)
        nan_input[0, 0, 0] = float("nan")  # Insert a NaN value

        try:
            output, _ = model(nan_input)
            # If we reach here, check if output contains NaN
            assert torch.isnan(output).any(), "Expected NaN in output with NaN input"
        except Exception as e:
            # Model should either propagate NaN or raise an exception
            assert isinstance(e, (RuntimeError, ValueError)), "Expected RuntimeError or ValueError with NaN input"

    @pytest.mark.parametrize("xlstm_type", ["mlstm", "slstm"])
    def test_numerical_stability(self, batch_size, input_size, hidden_size, output_size, xlstm_type, device):
        """Test model behavior with extreme input values"""
        model = xLSTMTime(
            input_size=input_size,
            hidden_size=hidden_size,
            output_size=output_size,
            xlstm_type=xlstm_type,
            device=device,
        )

        # Test with very large values
        large_input = torch.full((batch_size, 24, input_size), 1e10).to(device)
        output_large, _ = model(large_input)
        assert not torch.isnan(output_large).any(), "NaN in output with large input values"
        assert not torch.isinf(output_large).any(), "Inf in output with large input values"

        # Test with very small values
        small_input = torch.full((batch_size, 24, input_size), 1e-10).to(device)
        output_small, _ = model(small_input)
        assert not torch.isnan(output_small).any(), "NaN in output with small input values"
        assert not torch.isinf(output_small).any(), "Inf in output with small input values"
