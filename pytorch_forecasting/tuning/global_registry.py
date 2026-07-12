"""
Global Hyperparameter Registry.

This is the "common ground" — standard ranges for parameters that appear
across multiple models. When BaseModel inspects a subclass's __init__
and finds 'hidden_size', it looks up the range HERE.

HOW TO EXTEND: If a new model introduces a new common parameter,
just add one line here. All models using that param name are instantly tuneable.
"""

from pytorch_forecasting.tuning.search_range import _SearchRange

_UNIVERSAL_PARAMS = {
    "optimizer": _SearchRange(
        param_type="categorical",
        choices=["adam", "adamw"],
    ),
    "optimizer_params.lr": _SearchRange(
        param_type="float",
        low=1e-5,
        high=1e-1,
        log=True,
    ),
}

_MODEL_PARAMS = {
    "hidden_size": _SearchRange(param_type="int", low=16, high=512, log=True),
    "dropout": _SearchRange(param_type="float", low=0.05, high=0.5),
    "dropout_rate": _SearchRange(param_type="float", low=0.05, high=0.5),
    "activation": _SearchRange(param_type="categorical", choices=["relu", "gelu"]),
    "n_heads": _SearchRange(param_type="categorical", choices=[1, 2, 4, 8]),
    "attention_head_size": _SearchRange(param_type="int", low=1, high=4),
    "e_layers": _SearchRange(param_type="int", low=1, high=4),
    "num_layers": _SearchRange(param_type="int", low=1, high=4),
    "d_ff": _SearchRange(param_type="int", low=64, high=2048, log=True),
    "d_model": _SearchRange(param_type="int", low=16, high=512, log=True),
    "patch_length": _SearchRange(
        param_type="categorical", choices=[1, 2, 4, 8, 12, 16, 24]
    ),
    "moving_avg": _SearchRange(
        param_type="categorical", choices=[3, 5, 7, 11, 15, 21, 25]
    ),
    "persistence_weight": _SearchRange(param_type="float", low=0.0, high=1.0),
    "factor": _SearchRange(param_type="int", low=1, high=10),
}


_TRAINER_PARAMS = {
    "gradient_clip_val": _SearchRange(
        param_type="float", low=0.01, high=100.0, log=True
    ),
}

_GLOBAL_SEARCH_SPACE = {**_UNIVERSAL_PARAMS, **_MODEL_PARAMS, **_TRAINER_PARAMS}
