from pytorch_forecasting._registry import all_objects

# Remove legacy lists and mappings for losses by pred/y type and tensor shape checks.
# Use tags and _get_test_dataloaders_from for all compatibility and test setup.

METRIC_PKGS = all_objects(object_types="metric", return_names=False)


def get_compatible_losses(pred_types, y_types):
    """
    Get compatible losses based on prediction types and target types.
    Returns a list of (pkg, loss_instance) tuples.
    """
    compatible_losses = []
    for pkg in METRIC_PKGS:
        pkg_pred_types = pkg.get_class_tag("info:pred_type", [])
        pkg_y_types = pkg.get_class_tag("info:y_type", [])
        if any(pt in pred_types for pt in pkg_pred_types) and any(
            yt in y_types for yt in pkg_y_types
        ):
            compatible_losses.append((pkg, pkg.get_cls()()))
    return compatible_losses
