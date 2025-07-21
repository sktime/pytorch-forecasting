"""Test configs."""

# list of str, names of metrics to exclude from testing
# WARNING: tests for these metrics will be skipped
EXCLUDE_METRICS = [
    "DummySkipped",
    "ClassName",  # exclude classes from extension templates
]

# dictionary of lists of str, names of tests to exclude from testing
# keys are class names of estimators, values are lists of test names to exclude
# WARNING: tests with these names will be skipped
EXCLUDED_TESTS = {
    "MQF2DistributionLoss": ["test_metric_functionality"],
    "PoissonLoss": [
        "test_metric_functionality",
    ],
}
