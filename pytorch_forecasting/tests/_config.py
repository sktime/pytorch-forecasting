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


# groups dataset item key names across data modules that can be tested in
#  similar way. Test framework will test all keys in the list for the given role.
# if a key is present in the batch, but absent from the list,
# that key wont be tested by test_all_data_modules.py.
DATAMODULE_DATASET_KEYS_MAP: dict[str, list[str]] = {
    "history_cat": ["history_cat", "encoder_cat"],
    "history_cont": ["history_cont", "encoder_cont"],
    "future_cat": ["future_cat", "decoder_cat"],
    "future_cont": ["future_cont", "decoder_cont"],
    "history_length": ["history_length", "encoder_lengths"],
    "future_length": ["future_length", "decoder_lengths"],
    "future_target_len": ["future_target_len", "decoder_target_lengths"],
    "history_mask": ["history_mask", "encoder_mask"],
    "future_mask": ["future_mask", "decoder_mask"],
    "history_time_idx": ["history_time_idx", "encoder_time_idx"],
    "future_time_idx": ["future_time_idx", "decoder_time_idx"],
    "history_target": ["history_target", "target_past"],
    "groups": ["groups"],
    "target_scale": ["target_scale"],
}


def resolve_batch_key(batch: dict, role: str) -> str | None:
    """Return the first batch key in *batch* that matches *role*, or ``None``."""
    try:
        candidates = DATAMODULE_DATASET_KEYS_MAP[role]
    except KeyError as exc:
        raise KeyError(
            f"Unknown datamodule batch role '{role}'. "
            f"Known roles: {sorted(DATAMODULE_DATASET_KEYS_MAP)}"
        ) from exc

    for key in candidates:
        if key in batch:
            return key
    return None
