"""Test configs."""

# list of str, names of estimators to exclude from testing
# WARNING: tests for these estimators will be skipped
EXCLUDE_ESTIMATORS = [
    "DummySkipped",
    "ClassName",  # exclude classes from extension templates
    "NBeatsKAN_pkg",
    "NBeats_pkg",
    "TimeXer_pkg",
    "xLSTMTime_pkg",
]

# dictionary of lists of str, names of tests to exclude from testing
# keys are class names of estimators, values are lists of test names to exclude
# WARNING: tests with these names will be skipped
EXCLUDED_TESTS = {}
