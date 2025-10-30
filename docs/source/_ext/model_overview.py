"""
Sphinx extension: Auto-generate pytorch_forecasting model overview.

This writes/overwrites docs/source/models.rst during the build,
listing all registry models with tags and links to API docs.
"""

from __future__ import annotations

import os


def _safe_import_all_objects():
    try:
        # prefer public registry interface
        from pytorch_forecasting._registry import all_objects  # type: ignore

        return all_objects, None
    except Exception as e:  # pragma: no cover - defensive
        # fallback to manual discovery if registry fails (e.g., skbase not available)
        return _manual_model_discovery(), e


def _manual_model_discovery():
    """Fallback model discovery when registry is not available."""
    import importlib
    import inspect
    from pathlib import Path
    
    models = []
    models_dir = Path(__file__).parent.parent.parent.parent / "pytorch_forecasting" / "models"
    
    # Known model packages
    model_packages = [
        "deepar", "dlinear", "mlp", "nbeats", "nhits", 
        "rnn", "temporal_fusion_transformer", "tide", "timexer", "xlstm"
    ]
    
    for pkg_name in model_packages:
        try:
            module_name = f"pytorch_forecasting.models.{pkg_name}"
            module = importlib.import_module(module_name)
            
            # Look for model classes in the module
            for name, obj in inspect.getmembers(module):
                if (inspect.isclass(obj) and 
                    hasattr(obj, '__module__') and 
                    obj.__module__.startswith(module_name) and
                    not name.startswith('_')):
                    
                    # Determine estimator type based on package name
                    if pkg_name in ["deepar", "dlinear", "mlp", "nbeats", "nhits", "rnn"]:
                        estimator_type = "forecaster_v1"
                    else:
                        estimator_type = "forecaster_v2"
                    
                    models.append({
                        'names': name,
                        'objects': obj,
                        'object_type': estimator_type,
                        'authors': getattr(obj, 'authors', ['pytorch-forecasting developers']),
                        'python_dependencies': getattr(obj, 'python_dependencies', [])
                    })
        except Exception:
            continue
    
    return models


def _render_lines() -> list[str]:
    all_objects, err = _safe_import_all_objects()

    lines: list[str] = []
    lines.append("Models")
    lines.append("======")
    lines.append("")
    lines.append("(This page is auto-generated from the registry at build time.)")
    lines.append("Do not edit manually.")
    lines.append("")

    if all_objects is None:
        lines.extend(
            [
                ".. note::",
                "   Failed to import registry for model overview.",
                f"   Build-time error: ``{err}``",
                "",
            ]
        )
        return lines

    # Handle both registry DataFrame and manual discovery list
    if isinstance(all_objects, list):
        # Manual discovery fallback
        models_data = all_objects
    else:
        # Registry DataFrame
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
            models_data = df.to_dict('records') if df is not None else []
        except Exception as e:  # pragma: no cover - defensive
            lines.extend(
                [
                    ".. note::",
                    f"   Registry query failed: ``{e}``",
                    "",
                ]
            )
            return lines

    if not models_data:
        lines.extend([".. note::", "   No models found in registry.", ""])
        return lines

    # header
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

    # rows
    # Sort models by name
    sorted_models = sorted(models_data, key=lambda x: x.get("names", ""))
    
    for row in sorted_models:
        pkg_cls = row["objects"]
        try:
            model_cls = pkg_cls.get_model_cls()
            qualname = f"{model_cls.__module__}.{model_cls.__name__}"
        except Exception:
            qualname = f"{pkg_cls.__module__}.{pkg_cls.__name__}"

        # Get object type (forecaster_pytorch_v1 or forecaster_pytorch_v2)
        object_type = row.get("object_type", "")
        if object_type == "forecaster_pytorch_v1":
            estimator_type = "forecaster_v1"
        elif object_type == "forecaster_pytorch_v2":
            estimator_type = "forecaster_v2"
        else:
            estimator_type = object_type

        # Get authors from tags
        authors = row.get("authors", [])
        if isinstance(authors, list) and authors:
            authors_str = ", ".join(authors)
        else:
            authors_str = "pytorch-forecasting developers"

        # No maintainers tag exists, so use authors as maintainers
        maintainers_str = authors_str

        # Get dependencies from tags
        dependencies = row.get("python_dependencies", [])
        if isinstance(dependencies, list) and dependencies:
            dependencies_str = ", ".join(dependencies)
        else:
            dependencies_str = "None"

        row_cells = [
            f":py:class:`~{qualname}`",
            estimator_type,
            authors_str,
            maintainers_str,
            dependencies_str,
        ]
        lines.append("   * - " + "\n     - ".join(row_cells))

    lines.append("")
    return lines


def _is_safe_mode() -> bool:
    """Return True if model overview generation is explicitly disabled.

    By default, generation runs in all environments.
    Set PF_SKIP_MODEL_OVERVIEW=1 to skip.
    """
    if os.environ.get("PF_SKIP_MODEL_OVERVIEW", "").lower() in {"1", "true", "yes"}:
        return True
    return False


def _write_models_rst(app, config=None) -> None:
    """Write models.rst file.
    
    This can be called either from:
    - config-inited event (app, config)
    - builder-inited event (app)
    """
    # confdir is docs/source
    out_file = os.path.join(app.confdir, "models.rst")
    try:
        if _is_safe_mode():
            # minimal page on hosted builders to avoid heavy optional deps
            lines = [
                "Models",
                "======",
                "",
                "(Model overview generation is disabled in this build environment.)",
                "Use a local build to view the full, registry-driven table.",
                "",
            ]
        else:
            lines = _render_lines()
    except Exception as exc:  # pragma: no cover - defensive
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
    
    # Print confirmation for debugging
    print(f"[model_overview] Wrote {len(lines)} lines to {out_file}")


def setup(app):
    """Setup the Sphinx extension."""
    # Use builder-inited instead of config-inited
    # This ensures the extension is fully loaded
    app.connect("builder-inited", lambda app: _write_models_rst(app))
    
    return {
        "version": "1.0",
        "parallel_read_safe": True,
        "parallel_write_safe": True,
    }