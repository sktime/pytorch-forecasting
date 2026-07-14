import sys
import warnings
from importlib.abc import MetaPathFinder

# Raise deprecation warning on import
warnings.warn(
    "pytorch_forecasting has been renamed to ptf. Please import from ptf instead.",
    DeprecationWarning,
    stacklevel=2,
)

# Custom import interceptor to redirect all sub-imports (e.g., pytorch_forecasting.models)
class LegacyRedirectFinder(MetaPathFinder):
    def find_spec(self, fullname, path, target=None):
        if fullname.startswith("pytorch_forecasting"):
            real_name = fullname.replace("pytorch_forecasting", "ptf", 1)
            try:
                __import__(real_name)
                sys.modules[fullname] = sys.modules[real_name]
                return sys.modules[real_name].__spec__
            except ImportError:
                return None
        return None

# Register the finder at the start of sys.meta_path
sys.meta_path.insert(0, LegacyRedirectFinder())

# Map the root package to the renamed ptf package
import ptf
sys.modules["pytorch_forecasting"] = ptf
