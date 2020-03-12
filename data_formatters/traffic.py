import pandas as pd
from utils.utils import get_single_col_by_input_type
from utils.utils import extract_cols_from_data_type
from .base import GenericDataFormatter
from .base import VolatilityFormatter
from .base import DataTypes, InputTypes
import sklearn.preprocessing

class TrafficFormatter(VolatilityFormatter):
    """Defines and formats data for the traffic dataset.
    This also performs z-score normalization across the entire dataset, hence
    re-uses most of the same functions as volatility.
    Attributes:
    column_definition: Defines input and data type of column used in the
      experiment.
    identifiers: Entity identifiers used in experiments.
    """

    _column_definition = [
      ('id', DataTypes.REAL_VALUED, InputTypes.ID),
      ('hours_from_start', DataTypes.REAL_VALUED, InputTypes.TIME),
      ('values', DataTypes.REAL_VALUED, InputTypes.TARGET),
      ('time_on_day', DataTypes.REAL_VALUED, InputTypes.KNOWN_INPUT),
      ('day_of_week', DataTypes.REAL_VALUED, InputTypes.KNOWN_INPUT),
      ('hours_from_start', DataTypes.REAL_VALUED, InputTypes.KNOWN_INPUT),
      ('categorical_id', DataTypes.CATEGORICAL, InputTypes.STATIC_INPUT),
    ]

    def split_data(self, df, valid_boundary=151, test_boundary=166):
        """Splits data frame into training-validation-test data frames.
        This also calibrates scaling object, and transforms data for each split.
        Args:
          df: Source data frame to split.
          valid_boundary: Starting year for validation data
          test_boundary: Starting year for test data
        Returns:
          Tuple of transformed (train, valid, test) data.
        """

        print('Formatting train-valid-test splits.')

        index = df['sensor_day']
        train = df.loc[index < valid_boundary]
        valid = df.loc[(index >= valid_boundary - 7) & (index < test_boundary)]
        test = df.loc[index >= test_boundary - 7]

        self.set_scalers(train)

        return (self.transform_inputs(data) for data in [train, valid, test])

    # Default params
    def get_fixed_params(self):
        """Returns fixed model parameters for experiments."""

        fixed_params = {
            'total_time_steps': 8 * 24,
            'num_encoder_steps': 7 * 24,
            'num_epochs': 100,
            'early_stopping_patience': 5,
            'multiprocessing_workers': 5
        }

        return fixed_params

    def get_default_model_params(self):
        """Returns default optimised model parameters."""

        model_params = {
            'dropout_rate': 0.3,
            'hidden_layer_size': 320,
            'learning_rate': 0.001,
            'minibatch_size': 128,
            'max_gradient_norm': 100.,
            'num_heads': 4,
            'stack_size': 1
        }

        return model_params

    def get_num_samples_for_calibration(self):
        """Gets the default number of training and validation samples.
        Use to sub-sample the data for network calibration and a value of -1 uses
        all available samples.
        Returns:
          Tuple of (training samples, validation samples)
        """
        return 450000, 50000