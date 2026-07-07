"""
Global Hyperparameter Registry.

This is the "common ground" — standard ranges for parameters that appear
across multiple models. When BaseModel inspects a subclass's __init__
and finds 'hidden_size', it looks up the range HERE.

HOW TO EXTEND: If a new model introduces a new common parameter,
just add one line here. All models using that param name are instantly tuneable.
"""

from pytorch_forecasting.tuning.search_range import SearchRange

UNIVERSAL_PARAMS = {
    "optimizer": SearchRange(
        param_type="categorical",
        choices=["adam", "adamw"],
    ),
    "optimizer_params.lr": SearchRange(
        param_type="float",
        low=1e-5,
        high=1e-1,
        log=True,
    ),
}

MODEL_PARAMS = {
    "hidden_size": SearchRange(param_type="int", low=16, high=512, log=True),
    "dropout": SearchRange(param_type="float", low=0.05, high=0.5),
    "dropout_rate": SearchRange(param_type="float", low=0.05, high=0.5),
    "activation": SearchRange(param_type="categorical", choices=["relu", "gelu"]),
    "n_heads": SearchRange(param_type="categorical", choices=[1, 2, 4, 8]),
    "attention_head_size": SearchRange(param_type="int", low=1, high=4),
    "e_layers": SearchRange(param_type="int", low=1, high=4),
    "num_layers": SearchRange(param_type="int", low=1, high=4),
    "d_ff": SearchRange(param_type="int", low=64, high=2048, log=True),
    "d_model": SearchRange(param_type="int", low=16, high=512, log=True),
    "patch_length": SearchRange(
        param_type="categorical", choices=[1, 2, 4, 8, 12, 16, 24]
    ),
    "moving_avg": SearchRange(
        param_type="categorical", choices=[3, 5, 7, 11, 15, 21, 25]
    ),
    "persistence_weight": SearchRange(param_type="float", low=0.0, high=1.0),
    "factor": SearchRange(param_type="int", low=1, high=10),
}


TRAINER_PARAMS = {
    "gradient_clip_val": SearchRange(
        param_type="float", low=0.01, high=100.0, log=True
    ),
}

GLOBAL_SEARCH_SPACE = {**UNIVERSAL_PARAMS, **MODEL_PARAMS, **TRAINER_PARAMS}
