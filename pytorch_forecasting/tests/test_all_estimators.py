"""Automated tests based on the skbase test suite template."""

from inspect import isclass
import shutil

import lightning.pytorch as pl
from lightning.pytorch.callbacks import EarlyStopping
from lightning.pytorch.loggers import TensorBoardLogger
from skbase.testing import BaseFixtureGenerator as _BaseFixtureGenerator

from pytorch_forecasting._registry import all_objects
from pytorch_forecasting.tests._config import EXCLUDE_ESTIMATORS, EXCLUDED_TESTS
from pytorch_forecasting.tests._conftest import make_dataloaders

# whether to test only estimators from modules that are changed w.r.t. main
# default is False, can be set to True by pytest --only_changed_modules True flag
ONLY_CHANGED_MODULES = False


class PackageConfig:
    """Contains package config variables for test classes."""

    # class variables which can be overridden by descendants
    # ------------------------------------------------------

    # package to search for objects
    # expected type: str, package/module name, relative to python environment root
    package_name = "pytorch_forecasting"

    # list of object types (class names) to exclude
    # expected type: list of str, str are class names
    exclude_objects = EXCLUDE_ESTIMATORS

    # list of tests to exclude
    # expected type: dict of lists, key:str, value: List[str]
    # keys are class names of estimators, values are lists of test names to exclude
    excluded_tests = EXCLUDED_TESTS


class BaseFixtureGenerator(_BaseFixtureGenerator):
    """Fixture generator for base testing functionality in sktime.

    Test classes inheriting from this and not overriding pytest_generate_tests
        will have estimator and scenario fixtures parametrized out of the box.

    Descendants can override:
        estimator_type_filter: str, class variable; None or scitype string
            e.g., "forecaster", "transformer", "classifier", see BASE_CLASS_SCITYPE_LIST
            which estimators are being retrieved and tested
        fixture_sequence: list of str
            sequence of fixture variable names in conditional fixture generation
        _generate_[variable]: object methods, all (test_name: str, **kwargs) -> list
            generating list of fixtures for fixture variable with name [variable]
                to be used in test with name test_name
            can optionally use values for fixtures earlier in fixture_sequence,
                these must be input as kwargs in a call
        is_excluded: static method (test_name: str, est: class) -> bool
            whether test with name test_name should be excluded for estimator est
                should be used only for encoding general rules, not individual skips
                individual skips should go on the EXCLUDED_TESTS list in _config
            requires _generate_object_class and _generate_object_instance as is
        _excluded_scenario: static method (test_name: str, scenario) -> bool
            whether scenario should be skipped in test with test_name test_name
            requires _generate_estimator_scenario as is

    Fixtures parametrized
    ---------------------
    object_class: estimator inheriting from BaseObject
        ranges over estimator classes not excluded by EXCLUDE_ESTIMATORS, EXCLUDED_TESTS
    object_instance: instance of estimator inheriting from BaseObject
        ranges over estimator classes not excluded by EXCLUDE_ESTIMATORS, EXCLUDED_TESTS
        instances are generated by create_test_instance class method of object_class
    trainer_kwargs: list of dict
        ranges over dictionaries of kwargs for the trainer
    """

    # overrides object retrieval in scikit-base
    def _all_objects(self):
        """Retrieve list of all object classes of type self.object_type_filter.

        If self.object_type_filter is None, retrieve all objects.
        If class, retrieve all classes inheriting from self.object_type_filter.
        Otherwise (assumed str or list of str), retrieve all classes with tags
        object_type in self.object_type_filter.
        """
        filter = getattr(self, "object_type_filter", None)

        if isclass(filter):
            object_types = filter.get_class_tag("object_type", None)
        else:
            object_types = filter

        obj_list = all_objects(
            object_types=object_types,
            return_names=False,
            exclude_objects=self.exclude_objects,
        )

        if isclass(filter):
            obj_list = [obj for obj in obj_list if issubclass(obj, filter)]

        # run_test_for_class selects the estimators to run
        # based on whether they have changed, and whether they have all dependencies
        # internally, uses the ONLY_CHANGED_MODULES flag,
        # and checks the python env against python_dependencies tag
        # obj_list = [obj for obj in obj_list if run_test_for_class(obj)]

        return obj_list

    # which sequence the conditional fixtures are generated in
    fixture_sequence = [
        "object_metadata",
        "object_class",
        "object_instance",
        "trainer_kwargs",
    ]

    def _generate_object_metadata(self, test_name, **kwargs):
        """Return object class fixtures.

        Fixtures parametrized
        ---------------------
        object_class: object inheriting from BaseObject
            ranges over all object classes not excluded by self.excluded_tests
        """
        object_classes_to_test = [
            est for est in self._all_objects() if not self.is_excluded(test_name, est)
        ]
        object_names = [est.name() for est in object_classes_to_test]

        return object_classes_to_test, object_names

    def _generate_object_class(self, test_name, **kwargs):
        """Return object class fixtures.

        Fixtures parametrized
        ---------------------
        object_class: object inheriting from BaseObject
            ranges over all object classes not excluded by self.excluded_tests
        """
        all_metadata = self._all_objects()
        all_cls = [est.get_model_cls() for est in all_metadata]
        object_classes_to_test = [
            est for est in all_cls if not self.is_excluded(test_name, est)
        ]
        object_names = [est.__name__ for est in object_classes_to_test]

        return object_classes_to_test, object_names

    def _generate_trainer_kwargs(self, test_name, **kwargs):
        """Return kwargs for the trainer.

        Fixtures parametrized
        ---------------------
        trainer_kwargs: dict
            ranges over all kwargs for the trainer
        """
        if "object_metadata" in kwargs.keys():
            obj_meta = kwargs["object_metadata"]
        else:
            return []

        all_train_kwargs = obj_meta.get_test_train_params()
        rg = range(len(all_train_kwargs))
        train_kwargs_names = [str(i) for i in rg]

        return all_train_kwargs, train_kwargs_names


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
    data_with_covariates = data_with_covariates.copy()
    if clip_target:
        data_with_covariates["target"] = data_with_covariates["volume"].clip(1e-3, 1.0)
    else:
        data_with_covariates["target"] = data_with_covariates["volume"]
    data_loader_default_kwargs = dict(
        target="target",
        time_varying_known_reals=["price_actual"],
        time_varying_unknown_reals=["target"],
        static_categoricals=["agency"],
        add_relative_time_idx=True,
    )
    data_loader_default_kwargs.update(data_loader_kwargs)
    dataloaders_with_covariates = make_dataloaders(
        data_with_covariates, **data_loader_default_kwargs
    )

    train_dataloader = dataloaders_with_covariates["train"]
    val_dataloader = dataloaders_with_covariates["val"]
    test_dataloader = dataloaders_with_covariates["test"]

    early_stop_callback = EarlyStopping(
        monitor="val_loss", min_delta=1e-4, patience=1, verbose=False, mode="min"
    )

    logger = TensorBoardLogger(tmp_path)
    if trainer_kwargs is None:
        trainer_kwargs = {}
    trainer = pl.Trainer(
        max_epochs=3,
        gradient_clip_val=0.1,
        callbacks=[early_stop_callback],
        enable_checkpointing=True,
        default_root_dir=tmp_path,
        limit_train_batches=2,
        limit_val_batches=2,
        limit_test_batches=2,
        logger=logger,
        **trainer_kwargs,
    )

    net = estimator_cls.from_dataset(
        train_dataloader.dataset,
        hidden_size=5,
        cell_type=cell_type,
        learning_rate=0.01,
        log_gradient_flow=True,
        log_interval=1000,
        n_plotting_samples=100,
        **kwargs,
    )
    net.size()
    try:
        trainer.fit(
            net,
            train_dataloaders=train_dataloader,
            val_dataloaders=val_dataloader,
        )
        test_outputs = trainer.test(net, dataloaders=test_dataloader)
        assert len(test_outputs) > 0
        # check loading
        net = estimator_cls.load_from_checkpoint(
            trainer.checkpoint_callback.best_model_path
        )

        # check prediction
        net.predict(
            val_dataloader,
            fast_dev_run=True,
            return_index=True,
            return_decoder_lengths=True,
            trainer_kwargs=trainer_kwargs,
        )
    finally:
        shutil.rmtree(tmp_path, ignore_errors=True)

    net.predict(
        val_dataloader,
        fast_dev_run=True,
        return_index=True,
        return_decoder_lengths=True,
        trainer_kwargs=trainer_kwargs,
    )


class TestAllPtForecasters(PackageConfig, BaseFixtureGenerator):
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
        """Fails for certain, for testing."""
        from pytorch_forecasting.metrics import NegativeBinomialDistributionLoss
        from pytorch_forecasting.tests._data_scenarios import data_with_covariates

        data_with_covariates = data_with_covariates()

        object_class = object_metadata.get_model_cls()

        if "loss" in trainer_kwargs and isinstance(
            trainer_kwargs["loss"], NegativeBinomialDistributionLoss
        ):
            data_with_covariates = data_with_covariates.assign(
                volume=lambda x: x.volume.round()
            )
        _integration(object_class, data_with_covariates, tmp_path, **trainer_kwargs)
