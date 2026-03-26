"""Sphinx extension to auto-generate model overview table from registry.

Registers the ``.. model-overview::`` directive which queries the
``pytorch_forecasting._registry.all_objects`` registry, extracts model
tags, and renders an RST table comparing model capabilities.

This replaces the manually maintained CSV table in models.rst, ensuring
the documentation always reflects the actual registered models.
"""

from docutils import nodes
from docutils.parsers.rst import Directive
from docutils.statemachine import StringList
from sphinx.util.docutils import SphinxDirective
from sphinx.util.nodes import nested_parse_with_titles


# Tag keys used to build the comparison table columns
_CAPABILITY_TAGS = [
    ("info:name", "Name"),
    ("capability:exogenous", "Covariates"),
    ("capability:multivariate", "Multiple targets"),
    ("info:y_type", "Regression"),
    ("info:y_type", "Classification"),
    ("info:pred_type", "Probabilistic"),
    ("capability:pred_int", "Prediction intervals"),
    ("info:compute", "Compute (1-5)"),
]


def _get_model_rows():
    """Query the registry and return a list of row dicts for each model.

    Each row maps column header to display value. Only includes v1 models
    (the ones with full tag metadata). v2 models are listed separately.
    """
    from pytorch_forecasting._registry import all_objects

    tag_keys = [
        "info:name",
        "info:compute",
        "info:pred_type",
        "info:y_type",
        "capability:exogenous",
        "capability:multivariate",
        "capability:pred_int",
    ]

    results = all_objects(
        return_names=True,
        return_tags=tag_keys,
        suppress_import_stdout=True,
    )

    rows = []
    for entry in results:
        name, klass, *tag_values = entry
        tags = dict(zip(tag_keys, tag_values))

        # Skip models without info:name (base classes, internal)
        model_name = tags.get("info:name")
        if not model_name:
            continue

        # Build the module path for cross-reference
        module = klass.__module__
        qualname = klass.__qualname__
        ref = f":py:class:`~{module}.{qualname}`"

        pred_types = tags.get("info:pred_type") or []
        if isinstance(pred_types, str):
            pred_types = [pred_types]

        y_types = tags.get("info:y_type") or []
        if isinstance(y_types, str):
            y_types = [y_types]

        row = {
            "Name": ref,
            "Covariates": "x" if tags.get("capability:exogenous") else "",
            "Multiple targets": "x" if tags.get("capability:multivariate") else "",
            "Regression": "x" if "numeric" in y_types else "",
            "Classification": "x" if "category" in y_types else "",
            "Probabilistic": "x" if "distr" in pred_types else "",
            "Prediction intervals": "x" if tags.get("capability:pred_int") else "",
            "Compute (1-5)": str(tags.get("info:compute", "")),
        }
        rows.append(row)

    return rows


def _build_rst_table(rows):
    """Build an RST grid table from a list of row dicts."""
    if not rows:
        return ["*No models found in registry.*", ""]

    headers = [
        "Name",
        "Covariates",
        "Multiple targets",
        "Regression",
        "Classification",
        "Probabilistic",
        "Prediction intervals",
        "Compute (1-5)",
    ]

    # Compute column widths
    col_widths = [len(h) for h in headers]
    for row in rows:
        for i, h in enumerate(headers):
            col_widths[i] = max(col_widths[i], len(row.get(h, "")))

    def _sep(char="-"):
        return "+" + "+".join(char * (w + 2) for w in col_widths) + "+"

    def _row(values):
        cells = []
        for i, v in enumerate(values):
            cells.append(f" {v:<{col_widths[i]}} ")
        return "|" + "|".join(cells) + "|"

    lines = []
    lines.append(_sep("-"))
    lines.append(_row(headers))
    lines.append(_sep("="))
    for row in rows:
        values = [row.get(h, "") for h in headers]
        lines.append(_row(values))
        lines.append(_sep("-"))
    lines.append("")

    return lines


class ModelOverviewDirective(SphinxDirective):
    """Directive that auto-generates a model comparison table.

    Usage in RST::

        .. model-overview::
    """

    has_content = False
    required_arguments = 0
    optional_arguments = 0

    def run(self):
        rows = _get_model_rows()
        rst_lines = _build_rst_table(rows)

        # Parse the generated RST back into docutils nodes
        source = self.state_machine.get_source_and_line(self.lineno)
        vl = StringList(rst_lines, source=source[0])

        node = nodes.section()
        node.document = self.state.document
        nested_parse_with_titles(self.state, vl, node)

        return node.children


def setup(app):
    app.add_directive("model-overview", ModelOverviewDirective)
    return {
        "version": "0.1",
        "parallel_read_safe": True,
        "parallel_write_safe": True,
    }
