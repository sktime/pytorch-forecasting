"""Test configs."""

# list of str, names of metrics to exclude from testing
# WARNING: tests for these metrics will be skipped
EXCLUDE_METRICS = [
    "DummySkipped",
    "ClassName",  # exclude classes from extension templates
    "MASE_pkg",
]

# dictionary of lists of str, names of tests to exclude from testing
# keys are class names of estimators, values are lists of test names to exclude
# WARNING: tests with these names will be skipped
EXCLUDED_TESTS = {
    # MQF2DistributionLoss: Skipped because its to_quantiles and to_prediction
    # implementations do not conform to the base contract and are central to its
    # probabilistic forecasting design (uses distributional outputs).
    "MQF2DistributionLoss": ["test_metric_functionality"],
    # PoissonLoss: Skipped because its output and quantile logic are not compatible
    # with the standard metric API contract, due to its discrete distribution nature.
    "PoissonLoss": [
        "test_metric_functionality",
    ],
}
