from copy import deepcopy
from inspect import getfullargspec, isclass
from inspect import isclass

from skbase.testing import BaseFixtureGenerator as _BaseFixtureGenerator, QuickTester
from skbase.testing.utils._conditional_fixtures import (
    create_conditional_fixtures_and_names,
)

from pytorch_forecasting._registry import all_objects

# whether to test only estimators from modules that are changed w.r.t. main
# default is False, can be set to True by pytest --only_changed_modules True flag
ONLY_CHANGED_MODULES = False


class QuickTesterWithPkg(QuickTester):
    """Mixin class which adds the run_tests method to run tests on one object.

    Modification for pytorch-forecasting to make it work with the pkg structure.
    """

    def run_tests(
        self,
        obj,
        raise_exceptions=False,
        tests_to_run=None,
        fixtures_to_run=None,
        tests_to_exclude=None,
        fixtures_to_exclude=None,
    ):
        """Run all tests on one single object.

        All tests in self are run on the following object type fixtures:
            if est is a class, then object_class = est, and
                object_instance loops over est.create_test_instance()
            if est is an object, then object_class = est.__class__, and
                object_instance = est

        This is compatible with pytest.mark.parametrize decoration,
            but currently only with multiple *single variable* annotations.

        Parameters
        ----------
        obj : object class or object instance
        raise_exceptions : bool, optional, default=False
            whether to return exceptions/failures in the results dict, or raise them
                if False: returns exceptions in returned `results` dict
                if True: raises exceptions as they occur
        tests_to_run : str or list of str, names of tests to run. default = all tests
            sub-sets tests that are run to the tests given here.
        fixtures_to_run : str or list of str, pytest test-fixture combination codes.
            which test-fixture combinations to run. Default = run all of them.
            sub-sets tests and fixtures to run to the list given here.
            If both tests_to_run and fixtures_to_run are provided, runs the *union*,
            i.e., all test-fixture combinations for tests in tests_to_run,
                plus all test-fixture combinations in fixtures_to_run.
        tests_to_exclude : str or list of str, names of tests to exclude. default = None
            removes tests that should not be run, after subsetting via tests_to_run.
        fixtures_to_exclude : str or list of str, fixtures to exclude. default = None
            removes test-fixture combinations that should not be run.
            This is done after subsetting via fixtures_to_run.

        Returns
        -------
        results : dict of results of the tests in self
            keys are test/fixture strings, identical as in pytest, e.g., test[fixture]
            entries are the string "PASSED" if the test passed,
                or the exception raised if the test did not pass
            returned only if all tests pass, or raise_exceptions=False

        Raises
        ------
        if raise_exception=True, raises any exception produced by the tests directly

        Examples
        --------
        >>> from skbase.tests.mock_package.test_mock_package import CompositionDummy
        >>> from skbase.testing.test_all_objects import TestAllObjects
        >>> TestAllObjects().run_tests(
        ...     CompositionDummy,
        ...     tests_to_run="test_constructor"
        ... )
        {'test_constructor[CompositionDummy]': 'PASSED'}
        >>> TestAllObjects().run_tests(
        ...     CompositionDummy, fixtures_to_run="test_repr[CompositionDummy-1]"
        ... )
        {'test_repr[CompositionDummy-1]': 'PASSED'}
        """
        tests_to_run = self._check_none_str_or_list_of_str(
            tests_to_run, var_name="tests_to_run"
        )
        fixtures_to_run = self._check_none_str_or_list_of_str(
            fixtures_to_run, var_name="fixtures_to_run"
        )
        tests_to_exclude = self._check_none_str_or_list_of_str(
            tests_to_exclude, var_name="tests_to_exclude"
        )
        fixtures_to_exclude = self._check_none_str_or_list_of_str(
            fixtures_to_exclude, var_name="fixtures_to_exclude"
        )

        # retrieve tests from self
        test_names = [attr for attr in dir(self) if attr.startswith("test")]

        # we override the generator_dict, by replacing it with temp_generator_dict:
        #  the only object (class or instance) is est, this is overridden
        #  the remaining fixtures are generated conditionally, without change
        temp_generator_dict = deepcopy(self.generator_dict())

        if isclass(obj):
            if hasattr(obj, "pkg"):
                object_pkg = obj.pkg
                object_class = obj
            else:
                object_pkg = obj
                object_class = obj.get_cls()
        else:
            if hasattr(obj, "pkg"):
                object_class = type(obj)
                object_pkg = obj.pkg
            else:
                object_pkg = type(obj)
                object_class = obj.get_cls()

        def _generate_object_pkg(test_name, **kwargs):
            return [object_pkg], [object_class.__name__]

        def _generate_object_class(test_name, **kwargs):
            return [object_class], [object_class.__name__]

        def _generate_object_instance(test_name, **kwargs):
            return [obj.clone()], [object_class.__name__]

        def _generate_object_instance_cls(test_name, **kwargs):
            return object_class.create_test_instances_and_names()

        temp_generator_dict["object_pkg"] = _generate_object_pkg
        temp_generator_dict["object_class"] = _generate_object_class

        if not isclass(obj) and hasattr(obj, "pkg"):
            temp_generator_dict["object_instance"] = _generate_object_instance
        else:
            temp_generator_dict["object_instance"] = _generate_object_instance_cls
        # override of generator_dict end, temp_generator_dict is now prepared

        # sub-setting to specific tests to run, if tests or fixtures were speified
        if tests_to_run is None and fixtures_to_run is None:
            test_names_subset = test_names
        else:
            test_names_subset = []
            if tests_to_run is not None:
                test_names_subset += list(set(test_names).intersection(tests_to_run))
            if fixtures_to_run is not None:
                # fixture codes contain the test as substring until the first "["
                tests_from_fixt = [fixt.split("[")[0] for fixt in fixtures_to_run]
                test_names_subset += list(set(test_names).intersection(tests_from_fixt))
            test_names_subset = list(set(test_names_subset))

        # sub-setting by removing all tests from tests_to_exclude
        if tests_to_exclude is not None:
            test_names_subset = list(
                set(test_names_subset).difference(tests_to_exclude)
            )

        # the below loops run all the tests and collect the results here:
        results = {}
        # loop A: we loop over all the tests
        for test_name in test_names_subset:
            test_fun = getattr(self, test_name)
            fixture_sequence = self.fixture_sequence

            # all arguments except the first one (self)
            fixture_vars = getfullargspec(test_fun)[0][1:]
            fixture_vars = [var for var in fixture_sequence if var in fixture_vars]

            # this call retrieves the conditional fixtures
            #  for the test test_name, and the object
            _, fixture_prod, fixture_names = create_conditional_fixtures_and_names(
                test_name=test_name,
                fixture_vars=fixture_vars,
                generator_dict=temp_generator_dict,
                fixture_sequence=fixture_sequence,
                raise_exceptions=raise_exceptions,
            )

            # if function is decorated with mark.parametrize, add variable settings
            # NOTE: currently this works only with single-variable mark.parametrize
            if hasattr(test_fun, "pytestmark"):
                if len([x for x in test_fun.pytestmark if x.name == "parametrize"]) > 0:
                    # get the three lists from pytest
                    (
                        pytest_fixture_vars,
                        pytest_fixture_prod,
                        pytest_fixture_names,
                    ) = self._get_pytest_mark_args(test_fun)
                    # add them to the three lists from conditional fixtures
                    fixture_vars, fixture_prod, fixture_names = self._product_fixtures(
                        fixture_vars,
                        fixture_prod,
                        fixture_names,
                        pytest_fixture_vars,
                        pytest_fixture_prod,
                        pytest_fixture_names,
                    )

            # loop B: for each test, we loop over all fixtures
            for params, fixt_name in zip(fixture_prod, fixture_names):
                # this is needed because pytest unwraps 1-tuples automatically
                # but subsequent code assumes params is k-tuple, no matter what k is
                if len(fixture_vars) == 1:
                    params = (params,)
                key = f"{test_name}[{fixt_name}]"
                args = dict(zip(fixture_vars, params))

                # we subset to test-fixtures to run by this, if given
                #  key is identical to the pytest test-fixture string identifier
                if fixtures_to_run is not None and key not in fixtures_to_run:
                    continue
                if fixtures_to_exclude is not None and key in fixtures_to_exclude:
                    continue

                if not raise_exceptions:
                    try:
                        test_fun(**deepcopy(args))
                        results[key] = "PASSED"
                    except Exception as err:
                        results[key] = err
                else:
                    test_fun(**deepcopy(args))
                    results[key] = "PASSED"

        return results


class BaseFixtureGenerator(_BaseFixtureGenerator, QuickTesterWithPkg):
    """Fixture generator for base testing functionality in sktime.

    Test classes inheriting from this and not overriding pytest_generate_tests
        will have estimator and scenario fixtures parametrized out of the box.

    Descendants can override:
        object_type_filter: str, class variable;
            Controls which objects are retrieved and tested:

            - If None, retrieves all objects.
            - If class, retrieves all classes inheriting from this class.
            - If str/list of str: retrieve objects with matching object_type tag.
            (e.g., "forecaster_pytorch_v1", "forecaster_pytorch_v2, "metric")
        fixture_sequence: list of str
            sequence of fixture variable names in conditional fixture generation
        _generate_[variable]: object methods, all (test_name: str, **kwargs) -> list
            generating list of fixtures for fixture variable with name [variable]
                to be used in test with name test_name
            can optionally use values for fixtures earlier in fixture_sequence,
                these must be input as kwargs in a call
        is_excluded: static method (test_name: str, est: class) -> bool
            whether test with name test_name should be excluded for object obj
            should be used only for encoding general rules, not individual skips
            individual skips should go on the EXCLUDED_TESTS list in _config
            requires _generate_object_class and _generate_object_instance as is
        _excluded_scenario: static method (test_name: str, scenario) -> bool
            whether scenario should be skipped in test with test_name test_name
            requires _generate_object_scenario as is.

    Fixtures parametrized
    ---------------------
    object_class: estimator inheriting from BaseObject
        ranges over estimator classes not excluded by EXCLUDE_ESTIMATORS, EXCLUDED_TESTS
    object_instance: instance of estimator inheriting from BaseObject
        ranges over estimator classes not excluded by EXCLUDE_ESTIMATORS, EXCLUDED_TESTS
        instances are generated by create_test_instance class method of object_class
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
        "object_pkg",
        "object_class",
        "object_instance",
    ]

    def _generate_object_pkg(self, test_name, **kwargs):
        """Return object package fixtures.

        Fixtures parametrized
        ---------------------
        object_pkg: object package inheriting from BaseObject
            ranges over all object packages not excluded by self.excluded_tests
        """
        object_classes_to_test = [
            obj for obj in self._all_objects() if not self.is_excluded(test_name, obj)
        ]
        object_names = [obj.name() for obj in object_classes_to_test]

        return object_classes_to_test, object_names

    def _generate_object_class(self, test_name, **kwargs):
        """Return object class fixtures.

        Fixtures parametrized
        ---------------------
        object_class: object inheriting from BaseObject
            ranges over all object classes not excluded by self.excluded_tests
        """

        if "object_pkg" in kwargs.keys():
            all_pkgs = [kwargs["object_pkg"]]
        else:
            # call _generate_object_pkg to get all the packages
            all_pkgs, _ = self._generate_object_pkg(test_name=test_name)

        all_cls = [obj.get_cls() for obj in all_pkgs]
        object_classes_to_test = [
            obj for obj in all_cls if not self.is_excluded(test_name, obj)
        ]
        object_names = [obj.__name__ for obj in object_classes_to_test]

        return object_classes_to_test, object_names
