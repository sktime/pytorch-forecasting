import warnings
import pytest

def test_legacy_import_redirection():
    # Clear any previous imports from sys.modules to ensure we test the clean import hook path
    import sys
    for module_name in list(sys.modules.keys()):
        if module_name.startswith("pytorch_forecasting"):
            del sys.modules[module_name]

    # Catch warnings during import
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        
        # Try importing sub-module from legacy path
        import pytorch_forecasting.models as pf_models
        
        # Verify deprecation warning was emitted
        deprecation_warnings = [
            warning for warning in w 
            if issubclass(warning.category, DeprecationWarning) and "renamed to ptf" in str(warning.message)
        ]
        assert len(deprecation_warnings) > 0, "No DeprecationWarning was raised for legacy import"
        
        # Verify the modules and objects are accessible and correct
        assert hasattr(pf_models, "TemporalFusionTransformer"), "Failed to import legacy sub-module contents"
        
        # Verify the class is the same as the one in the new ptf namespace
        import ptf.models as ptf_models
        assert pf_models.TemporalFusionTransformer is ptf_models.TemporalFusionTransformer
