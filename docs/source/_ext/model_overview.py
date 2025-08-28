"""
Sphinx extension: Auto-generate pytorch_forecasting model overview.

This writes/overwrites docs/source/models.rst during the build,
listing all registry models with tags and links to API docs.
"""
from __future__ import annotations

import os
from typing import List


def _safe_import_all_objects():
    try:
        # prefer public registry interface
        from pytorch_forecasting._registry import all_objects  # type: ignore

        return all_objects, None
    except Exception as e:  # pragma: no cover - defensive
        return None, e


def _render_lines() -> List[str]:
    all_objects, err = _safe_import_all_objects()

    lines: List[str] = []
    lines.append("Models")
    lines.append("======")
    lines.append("")
    lines.append(
        "(This page is auto-generated from the registry at build time. Do not edit manually.)"
    )
    lines.append("")

    if all_objects is None:
        lines.extend(
            [
                ".. note::",
                f"   Failed to import registry for model overview.",
                f"   Build-time error: ``{err}``",
                "",
            ]
        )
        return lines

    try:
        df = all_objects(
            object_types=["forecaster_pytorch_v1", "forecaster_pytorch_v2"],
            as_dataframe=True,
            return_tags=[
                "object_type",
                "info:name",
                "capability:exogenous",
                "capability:multivariate",
                "capability:pred_int",
                "capability:flexible_history_length",
                "capability:cold_start",
            ],
            return_names=True,
        )
    except Exception as e:  # pragma: no cover - defensive
        lines.extend(
            [
                ".. note::",
                f"   Registry query failed: ``{e}``",
                "",
            ]
        )
        return lines

    if df is None or len(df) == 0:
        lines.extend([".. note::", "   No models found in registry.", ""]) 
        return lines

    # header
    lines.append(".. list-table:: Available forecasting models")
    lines.append("   :header-rows: 1")
    lines.append("   :widths: 28 10 10 12 14 16 12")
    lines.append("")
    lines.append(
        "   * - Model\n     - Version\n     - Covariates\n     - Multivariate\n     - Pred. intervals\n     - Flexible history\n     - Cold-start"
    )

    # rows
    for _, row in df.sort_values("names").iterrows():
        pkg_cls = row["objects"]
        try:
            model_cls = pkg_cls.get_model_cls()
            qualname = f"{model_cls.__module__}.{model_cls.__name__}"
        except Exception:
            qualname = f"{pkg_cls.__module__}.{pkg_cls.__name__}"

        def _mark(v):
            if v is True:
                return "x"
            if v is False:
                return ""
            return "?"

        lines.append(
            "   * - "
            + f":py:class:`~{qualname}`\n     - {row.get('object_type', '')}\n     - {_mark(row.get('capability:exogenous'))}\n     - {_mark(row.get('capability:multivariate'))}\n     - {_mark(row.get('capability:pred_int'))}\n     - {_mark(row.get('capability:flexible_history_length'))}\n     - {_mark(row.get('capability:cold_start'))}"
        )

    lines.append("")
    return lines


def _write_models_rst(app) -> None:
    # confdir is docs/source
    out_file = os.path.join(app.confdir, "models.rst")
    lines = _render_lines()
    os.makedirs(os.path.dirname(out_file), exist_ok=True)
    with open(out_file, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


def setup(app):
    app.connect("builder-inited", _write_models_rst)
    return {
        "version": "1.0",
        "parallel_read_safe": True,
        "parallel_write_safe": True,
    }


