"""Test configs."""

# list of str, names of estimators to exclude from testing
# WARNING: tests for these estimators will be skipped
EXCLUDE_ESTIMATORS = [
    "DummySkipped",
    "ClassName",  # exclude classes from extension templates
]

# dictionary of lists of str, names of tests to exclude from testing
# keys are class names of estimators, values are lists of test names to exclude
# WARNING: tests with these names will be skipped
EXCLUDED_TESTS = {}
