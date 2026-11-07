import sys


def test_ptf_import_redirection():
    # Clear any previous ptf imports from sys.modules
    for module_name in list(sys.modules.keys()):
        if module_name.startswith("ptf"):
            del sys.modules[module_name]

    # Import from ptf
    import ptf.models as ptf_models
    import pytorch_forecasting.models as pf_models

    # Verify redirection and content equality
    assert hasattr(ptf_models, "TemporalFusionTransformer")
    assert ptf_models.TemporalFusionTransformer is pf_models.TemporalFusionTransformer
