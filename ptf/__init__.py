from importlib.abc import MetaPathFinder
import sys


# Custom import interceptor to redirect all sub-imports
# (e.g., ptf.models -> pytorch_forecasting.models)
class LegacyRedirectFinder(MetaPathFinder):
    def find_spec(self, fullname, path, target=None):
        if fullname.startswith("ptf"):
            real_name = fullname.replace("ptf", "pytorch_forecasting", 1)
            try:
                __import__(real_name)
                sys.modules[fullname] = sys.modules[real_name]
                return sys.modules[real_name].__spec__
            except ImportError:
                return None
        return None


sys.meta_path.insert(0, LegacyRedirectFinder())

# Map the root package to pytorch_forecasting
import pytorch_forecasting

sys.modules["ptf"] = pytorch_forecasting
