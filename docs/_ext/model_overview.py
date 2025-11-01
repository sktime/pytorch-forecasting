"""
Sphinx extension: Auto-generate pytorch_forecasting model overview.

Writes docs/source/models.rst during the build listing registry models with tags
and doc links. Behaves safely on hosted builders via PF_SKIP_MODEL_OVERVIEW.
"""
from __future__ import annotations

import os
import importlib
import inspect
import pkgutil
import json
from pathlib import Path
from typing import Callable, List, Dict, Any, Optional

import pandas as pd


def _is_safe_mode() -> bool:
    if os.environ.get("PF_SKIP_MODEL_OVERVIEW", "").lower() in {"1", "true", "yes"}:
        return True
    return False


def _try_get_registry_caller() -> Optional[Callable[[], List[Dict[str, Any]]]]:
    """
    If a public registry exists, return a callable that queries it and
    returns a list of normalized dict rows. If no registry available, return None.
    """
    candidates = [
        "pytorch_forecasting._registry",
        "pytorch_forecasting.registry",
    ]
    for modname in candidates:
        try:
            mod = importlib.import_module(modname)
        except Exception:
            continue
        # find plausible all_objects in module
        all_objects = getattr(mod, "all_objects", None)
        if callable(all_objects):
            def fetch_from_registry() -> List[Dict[str, Any]]:
                # try a couple of common signatures
                try:
                    df = all_objects(
                        object_types=["forecaster_pytorch_v1", "forecaster_pytorch_v2"],
                        as_dataframe=True,
                        return_tags=[
                            "object_type",
                            "info:name",
                            "authors",
                            "python_dependencies",
                        ],
                        return_names=True,
                    )
                    if df is None:
                        return []
                    records = df.to_dict(orient="records")
                except TypeError:
                    # fallback: call with no args and filter results
                    items = all_objects()
                    # items might be list of (name, cls) pairs
                    records = []
                    for it in items:
                        try:
                            name, cls = it
                        except Exception:
                            continue
                        records.append({"info:name": name, "class": cls})
                # normalize
                normalized = []
                for r in records:
                    # different registry shapes - try to extract sensible values
                    name = r.get("info:name") or r.get("name") or r.get("names") or r.get("info.name")
                    cls = r.get("class") or r.get("objects") or r.get("objects") or r.get("object")
                    if cls is None and name and isinstance(name, tuple):
                        # guard: sometimes df rows are weird
                        continue
                    object_type = r.get("object_type") or r.get("tags:object_type") or ""
                    authors = r.get("authors") or r.get("info:authors") or r.get("tags", {}).get("authors", [])
                    python_dependencies = r.get("python_dependencies") or r.get("dependencies") or r.get("python_dependencies", [])
                    normalized.append({
                        "name": name,
                        "class_obj": cls,
                        "object_type": object_type,
                        "authors": authors,
                        "python_dependencies": python_dependencies,
                        "raw": r,
                    })
                return normalized
            return fetch_from_registry
    return None


def _manual_model_discovery() -> List[Dict[str, Any]]:
    """
    Best-effort manual discovery of known model subpackages.
    Returns normalized rows (same shape as registry fetcher).
    """
    # Known model packages to attempt import from
    model_packages = [
        "deepar", "dlinear", "mlp", "nbeats", "nhits",
        "rnn", "temporal_fusion_transformer", "tide", "timexer", "xlstm"
    ]
    prefix = "pytorch_forecasting.models"
    normalized = []

    for pkg in model_packages:
        module_name = f"{prefix}.{pkg}"
        try:
            mod = importlib.import_module(module_name)
        except Exception:
            # skip modules that import heavy optional deps or fail
            continue

        for name, obj in inspect.getmembers(mod, inspect.isclass):
            if getattr(obj, "__module__", "").startswith(module_name) and not name.startswith("_"):
                # determine v1 vs v2 if helpful
                if pkg in {"deepar", "dlinear", "mlp", "nbeats", "nhits", "rnn"}:
                    object_type = "forecaster_pytorch_v1"
                else:
                    object_type = "forecaster_pytorch_v2"
                authors = getattr(obj, "authors", ["pytorch-forecasting developers"])
                python_dependencies = getattr(obj, "python_dependencies", [])
                normalized.append({
                    "name": name,
                    "class_obj": obj,
                    "object_type": object_type,
                    "authors": authors,
                    "python_dependencies": python_dependencies,
                    "raw": {"module": module_name},
                })
    return normalized


def _get_models_fetcher() -> Callable[[], List[Dict[str, Any]]]:
    """
    Return a callable that returns normalized model rows. Prefers registry,
    falls back to manual discovery.
    """
    registry_caller = _try_get_registry_caller()
    if registry_caller is not None:
        return registry_caller
    # otherwise return the manual discovery callable
    return _manual_model_discovery


def _make_doc_url_for_class(cls) -> str:
    """
    Best-effort doc URL. If class has _generate_doc_link, try it.
    """
    try:
        if hasattr(cls, "_generate_doc_link"):
            return cls._generate_doc_link()
    except Exception:
        pass
    # fallback to RTD style URL
    mod = getattr(cls, "__module__", "")
    name = getattr(cls, "__name__", "")
    if mod and name:
        return f"https://pytorch-forecasting.readthedocs.io/en/stable/api/{mod}.{name}.html"
    return ""


def _render_lines() -> List[str]:
    fetcher = _get_models_fetcher()

    lines: List[str] = []
    lines.append("Models")
    lines.append("======")
    lines.append("")
    lines.append("(This page is auto-generated from the registry at build time.)")
    lines.append("Do not edit manually.")
    lines.append("")

    if _is_safe_mode():
        lines.extend([
            ".. note::",
            "   Model overview generation disabled in this environment.",
            "",
        ])
        return lines

    try:
        models_data = fetcher()
    except Exception as e:
        lines.extend([
            ".. note::",
            f"   Failed to discover models: ``{e}``",
            "",
        ])
        return lines

    if not models_data:
        lines.extend([".. note::", "   No models found in registry.", ""])
        return lines

    # prepare table
    lines.append(".. list-table:: Available forecasting models")
    lines.append("   :header-rows: 1")
    lines.append("   :widths: 30 15 20 20 15")
    lines.append("")
    header_cols = [
        "Class Name",
        "Estimator Type",
        "Authors",
        "Maintainers",
        "Dependencies",
    ]
    lines.append("   * - " + "\n     - ".join(header_cols))

    # sort by name
    sorted_models = sorted(models_data, key=lambda x: (str(x.get("name") or "").lower()))

    for row in sorted_models:
        cls = row.get("class_obj")
        name = row.get("name") or getattr(cls, "__name__", "")
        # qualname for py:class role
        try:
            # If model is a wrapper that can return the real model class
            model_cls = getattr(row.get("class_obj"), "get_model_cls", lambda: cls)()
            qualname = f"{model_cls.__module__}.{model_cls.__name__}"
        except Exception:
            qualname = f"{getattr(cls, '__module__', '')}.{getattr(cls, '__name__', '')}"

        # canonical object_type
        object_type = row.get("object_type") or ""
        estimator_type = object_type

        authors = row.get("authors") or ["pytorch-forecasting developers"]
        if isinstance(authors, (list, tuple)):
            authors_str = ", ".join(authors)
        else:
            authors_str = str(authors)

        maintainers_str = authors_str

        dependencies = row.get("python_dependencies", [])
        if isinstance(dependencies, (list, tuple)) and dependencies:
            dependencies_str = ", ".join(dependencies)
        elif dependencies:
            dependencies_str = str(dependencies)
        else:
            dependencies_str = "None"

        # build py:class reference
        class_ref = f":py:class:`~{qualname}`"

        row_cells = [
            class_ref,
            estimator_type,
            authors_str,
            maintainers_str,
            dependencies_str,
        ]
        lines.append("   * - " + "\n     - ".join(row_cells))

    lines.append("")
    return lines


def _write_models_rst(app, config) -> None:
    out_file = os.path.join(app.confdir, "models.rst")
    try:
        lines = _render_lines()
    except Exception as exc:
        lines = [
            "Models",
            "======",
            "",
            "(Model overview could not be generated due to a build-time error.)",
            f"Error: ``{exc}``",
            "",
        ]
    os.makedirs(os.path.dirname(out_file), exist_ok=True)
    with open(out_file, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


def setup(app):
    # Generate early so Sphinx sees the written file during source discovery
    app.connect("config-inited", _write_models_rst)
    return {
        "version": "1.0",
        "parallel_read_safe": True,
        "parallel_write_safe": True,
    }
