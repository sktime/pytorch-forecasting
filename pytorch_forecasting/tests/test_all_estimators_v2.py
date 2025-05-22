"""Automated tests based on the skbase test suite template."""

from inspect import isclass
import shutil

import lightning.pytorch as pl
from lightning.pytorch.callbacks import EarlyStopping
from lightning.pytorch.loggers import TensorBoardLogger

from pytorch_forecasting.tests._conftest import make_dataloaders
from pytorch_forecasting.tests.test_all_estimators import (
    BaseFixtureGenerator,
    PackageConfig,
)

# whether to test only estimators from modules that are changed w.r.t. main
# default is False, can be set to True by pytest --only_changed_modules True flag
ONLY_CHANGED_MODULES = False


def _integration(
    estimator_cls,
    data_with_covariates,
    tmp_path,
    cell_type="LSTM",
    data_loader_kwargs={},
    clip_target: bool = False,
    trainer_kwargs=None,
    **kwargs,
):
    pass


class TestAllPtForecastersV2(PackageConfig, BaseFixtureGenerator):
    """Generic tests for all objects in the mini package."""

    def test_doctest_examples(self, object_class):
        """Runs doctests for estimator class."""
        import doctest

        doctest.run_docstring_examples(object_class, globals())

    def test_integration(
        self,
        object_metadata,
        trainer_kwargs,
        tmp_path,
    ):
        pass
