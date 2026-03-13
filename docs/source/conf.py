# Configuration file for the Sphinx documentation builder.
#
# This file only contains a selection of the most common options. For a full
# list see the documentation:
# https://www.sphinx-doc.org/en/main/usage/configuration.html

# -- Path setup --------------------------------------------------------------

# If extensions (or modules to document with autodoc) are in another directory,
# add these directories to sys.path here. If the directory is relative to the
# documentation root, use os.path.abspath to make it absolute, like shown here.
#
import os
from pathlib import Path
import shutil
import sys

from sphinx.application import Sphinx
from sphinx.ext.autosummary import Autosummary
from sphinx.pycode import ModuleAnalyzer

SOURCE_PATH = Path(os.path.dirname(__file__))  # noqa # docs source
PROJECT_PATH = SOURCE_PATH.joinpath("../..")  # noqa # project root

sys.path.insert(0, str(PROJECT_PATH))  # noqa

import pytorch_forecasting  # isort:skip

# -- Project information -----------------------------------------------------

project = "pytorch-forecasting"
copyright = "2020, Jan Beitner"
author = "Jan Beitner"


# -- General configuration ---------------------------------------------------

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = [
    "nbsphinx",
    "recommonmark",
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.doctest",
    "sphinx.ext.intersphinx",
    "sphinx.ext.mathjax",
    "sphinx.ext.viewcode",
    "sphinx.ext.githubpages",
    "sphinx.ext.napoleon",
]

# Add any paths that contain templates here, relative to this directory.
templates_path = ["_templates"]

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = ["**/.ipynb_checkpoints"]


# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
#
html_theme = "pydata_sphinx_theme"
html_logo = "_static/logo.svg"
html_favicon = "_static/favicon.png"

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = ["_static"]


# setup configuration
def skip(app, what, name, obj, skip, options):
    """
    Document __init__ methods
    """
    if name == "__init__":
        return True
    return skip


apidoc_output_folder = SOURCE_PATH.joinpath("api")

PACKAGES = [pytorch_forecasting.__name__]


def get_by_name(string: str):
    """
    Import by name and return imported module/function/class

    Parameters
    ----------
    string (str):
        module/function/class to import, e.g. 'pandas.read_csv'
        will return read_csv function as defined by pandas

    Returns
    -------
        imported object
    """
    class_name = string.split(".")[-1]
    module_name = ".".join(string.split(".")[:-1])

    if module_name == "":
        return getattr(sys.modules[__name__], class_name)

    mod = __import__(module_name, fromlist=[class_name])
    return getattr(mod, class_name)


class ModuleAutoSummary(Autosummary):
    def get_items(self, names):
        new_names = []
        for name in names:
            mod = sys.modules[name]
            mod_items = getattr(mod, "__all__", mod.__dict__)
            for t in mod_items:
                if "." not in t and not t.startswith("_"):
                    obj = get_by_name(f"{name}.{t}")
                    if hasattr(obj, "__module__"):
                        mod_name = obj.__module__
                        t = f"{mod_name}.{t}"
                    if t.startswith("pytorch_forecasting"):
                        new_names.append(t)
        new_items = super().get_items(sorted(new_names))
        return new_items


def _make_estimator_overview(app):
    """Make estimator/model overview table.
    
    This function generates a dynamic table of all models in pytorch-forecasting
    by querying the registry system. The table is written as HTML and JSON files
    for inclusion in the documentation.
    """
    try:
        import pandas as pd
        from pytorch_forecasting._registry import all_objects
        
        # Base classes to exclude from the overview
        BASE_CLASSES = {
            "BaseModel",
            "BaseModelWithCovariates",
            "AutoRegressiveBaseModel",
            "AutoRegressiveBaseModelWithCovariates",
            "_BaseObject",
            "_BasePtForecaster",
            "_BasePtForecasterV2",
            "_BasePtForecaster_Common",
        }
        
        # Get all objects from registry
        all_objs = all_objects(return_names=True, suppress_import_stdout=True)
        
        records = []
        
        for obj_name, obj_class in all_objs:
            # Skip base classes
            if obj_name in BASE_CLASSES:
                continue
            
            # Skip if it's not a model class (check if it has get_class_tag method)
            if not hasattr(obj_class, "get_class_tag"):
                continue
            
            try:
                # Get model name from tags or use class name
                model_name = obj_class.get_class_tag("info:name", obj_name)
                
                # Get authors
                authors = obj_class.get_class_tag("authors", None)
                if authors is None:
                    authors = "pytorch-forecasting developers"
                elif isinstance(authors, list):
                    authors = ", ".join(authors)
                
                # Get object type
                object_type = obj_class.get_class_tag("object_type", "model")
                if isinstance(object_type, list):
                    object_type = ", ".join(object_type)
                
                # Get capabilities
                has_exogenous = obj_class.get_class_tag("capability:exogenous", False)
                has_multivariate = obj_class.get_class_tag("capability:multivariate", False)
                has_pred_int = obj_class.get_class_tag("capability:pred_int", False)
                has_flexible_history = obj_class.get_class_tag("capability:flexible_history_length", False)
                has_cold_start = obj_class.get_class_tag("capability:cold_start", False)
                
                # Get compute requirement
                compute = obj_class.get_class_tag("info:compute", None)
                
                # Get module path for documentation link
                module_path = obj_class.__module__
                class_name = obj_class.__name__
                
                # Construct documentation link
                # Convert module path to API documentation path
                api_path = module_path.replace(".", "/")
                doc_link = f"api/{api_path}.html#{module_path}.{class_name}"
                
                # Create model name with link
                model_name_link = f'<a href="{doc_link}">{model_name}</a>'
                
                # Build capabilities string
                capabilities = []
                if has_exogenous:
                    capabilities.append("Covariates")
                if has_multivariate:
                    capabilities.append("Multiple targets")
                if has_pred_int:
                    capabilities.append("Uncertainty")
                if has_flexible_history:
                    capabilities.append("Flexible history")
                if has_cold_start:
                    capabilities.append("Cold-start")
                
                capabilities_str = ", ".join(capabilities) if capabilities else ""
                
                records.append({
                    "Model Name": model_name_link,
                    "Type": object_type,
                    "Authors": authors,
                    "Covariates": "✓" if has_exogenous else "",
                    "Multiple targets": "✓" if has_multivariate else "",
                    "Uncertainty": "✓" if has_pred_int else "",
                    "Flexible history": "✓" if has_flexible_history else "",
                    "Cold-start": "✓" if has_cold_start else "",
                    "Compute": str(compute) if compute is not None else "",
                    "Capabilities": capabilities_str,
                    "Module": module_path,
                })
            except Exception as e:
                # Skip objects that can't be processed
                print(f"Warning: Could not process {obj_name}: {e}")
                continue
        
        if not records:
            print("Warning: No models found in registry")
            return
        
        # Create DataFrame
        df = pd.DataFrame(records)
        
        # Ensure _static directory exists
        static_dir = SOURCE_PATH.joinpath("_static")
        static_dir.mkdir(exist_ok=True)
        
        # Write HTML table
        html_file = static_dir.joinpath("model_overview_table.html")
        html_content = df[["Model Name", "Type", "Covariates", "Multiple targets", 
                           "Uncertainty", "Flexible history", "Cold-start", "Compute"]].to_html(
            classes="model-overview-table", index=False, border=0, escape=False
        )
        html_file.write_text(html_content, encoding="utf-8")
        print(f"Generated model overview table: {html_file}")
        
        # Write JSON database for interactive filtering (optional)
        json_file = static_dir.joinpath("model_overview_db.json")
        df.to_json(json_file, orient="records", indent=2)
        print(f"Generated model overview JSON: {json_file}")
        
    except ImportError as e:
        print(f"Warning: Could not generate model overview (missing dependency): {e}")
    except Exception as e:
        print(f"Warning: Error generating model overview: {e}")


def setup(app: Sphinx):
    app.add_css_file("custom.css")
    app.connect("autodoc-skip-member", skip)
    app.add_directive("moduleautosummary", ModuleAutoSummary)
    app.add_js_file("https://buttons.github.io/buttons.js", **{"async": "async"})
    # Connect model overview generator to builder-inited event
    app.connect("builder-inited", _make_estimator_overview)


# extension configuration
mathjax_path = "https://cdnjs.cloudflare.com/ajax/libs/mathjax/2.7.5/MathJax.js?config=TeX-MML-AM_CHTML"

# theme options
html_theme_options = {
    "github_url": "https://github.com/sktime/pytorch-forecasting",
    "navbar_end": ["navbar-icon-links.html", "search-field.html"],
    "show_nav_level": 2,
    "header_links_before_dropdown": 10,
    "external_links": [
        {"name": "GitHub", "url": "https://github.com/sktime/pytorch-forecasting"}
    ],
}

html_sidebars = {
    "index": [],
    # "getting-started": [],
    # "data": [],
    # "models": [],
    # "metrics": [],
    "faq": [],
    "contribute": [],
    "CHANGELOG": [],
}


autodoc_member_order = "groupwise"
autoclass_content = "both"

# autosummary
autosummary_generate = True
shutil.rmtree(SOURCE_PATH.joinpath("api"), ignore_errors=True)

# copy changelog
shutil.copy(
    "../../CHANGELOG.md",
    "CHANGELOG.md",
)

intersphinx_mapping = {
    "sklearn": ("https://scikit-learn.org/stable/", None),
}

suppress_warnings = [
    "autosummary.import_cycle",
]

# -----------nbsphinx extension ----------
nbsphinx_execute = "never"  # always
nbsphinx_allow_errors = False  # False
nbsphinx_timeout = 600  # seconds
