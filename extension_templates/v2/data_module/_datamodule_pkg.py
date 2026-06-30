"""Extension template for custom D2 datamodule package containers.

Purpose of this implementation template:
    quick implementation of a new D2 datamodule package following the template
    NOT a concrete class to import! This is NOT a base class or concrete class!
    This is to be used as a "fill-in" coding template.

How to use this implementation template:
- make a copy of the template in a suitable location, give it a descriptive name.
    - e.g. ``my_datamodule_pkg.py`` for a class ``MyDataModule_pkg``.
- work through all the "todo" comments below
- fill in code for mandatory methods
- change docstrings for functions and the file
- once complete: use as a local library, or contribute to pytorch-forecasting via PR

Mandatory methods to implement:
    get_cls - lazy import of the Lightning datamodule class
    get_datamodule_test_params - constructor kwargs for parametrized tests
    get_expected_metadata_keys - keys checked in ``datamodule.metadata`` after setup
    get_batch_keys - keys checked in collated batch ``x`` after ``collate_fn``
    get_sample_item_keys - keys checked in single dataset item (optional override)
"""

# todo: write an informative docstring for the file or module, remove the above

from pytorch_forecasting.data.data_module.base._base_datamodule_pkg import (
    _BasePtDataModule,
)

# todo: add any necessary imports here
# do not import the datamodule class at module level — use lazy import in get_cls()


# todo: change class name and write docstring
class MyDataModule_pkg(_BasePtDataModule):
    """Package container for MyDataModule — registers it with the unified test suite.

    The test harness discovers this class via ``object_type="datamodule_v2"`` and uses
    the methods below to instantiate datamodules, run ``setup()``, and assert batch
    contracts without importing your datamodule at collection time.
    """

    # todo: fill out datamodule tags here
    # tags are inherited from parent class if they are not set
    _tags = {
        # Required: identifies this as a v2 datamodule package (do not change).
        "object_type": "datamodule_v2",
        # Labels the batch layout for format-specific test assertions.
        # Use "encoder_decoder" or "tslib" when your keys match an existing format.
        "batch_format": "encoder_decoder",
        # Human-readable name; should match the Lightning datamodule class name.
        "info:name": "MyDataModule",
    }

    # implement this — mandatory
    @classmethod
    def get_cls(cls):
        """Lazy-import and return the Lightning datamodule class.

        Returns
        -------
        type
            Your ``MyDataModule`` class (subclass of ``BaseTimeSeriesDataModule``).
        """
        from extension_templates.v2.data_module.data_module import MyDataModule

        return MyDataModule

    # implement this — mandatory
    @classmethod
    def get_datamodule_test_params(cls):
        """Return constructor kwargs for parametrized datamodule tests.

        The test suite calls ``get_cls()(**params, time_series_dataset=dataset)`` for
        each dict. The dataset is injected by the harness — do not include it here.

        Returns
        -------
        list of dict
            Parameter sets for ``pytest.mark.parametrize``.

        Notes
        -----
        - Keys should match your datamodule ``__init__`` parameters (e.g.
          ``max_encoder_length`` for encoder-decoder format,
          ``context_length`` for tslib format).
        """
        return [
            {},
            {
                "max_encoder_length": 8,
                "max_prediction_length": 4,
                "batch_size": 2,
            },
        ]

    # implement this — mandatory
    @classmethod
    def get_expected_metadata_keys(cls):
        """Return top-level keys that must exist in ``datamodule.metadata`` after setup.

        Called after ``object_instance.setup()``; asserts each key is present in the
        dict returned by ``_prepare_metadata()`` (via the ``metadata`` property).

        Returns
        -------
        list of str
            todo: list keys your ``_prepare_metadata`` always sets, e.g.
            ``["encoder_cat", "target", "max_encoder_length"]``
        """
        return ["target"]  # todo: match your ``_prepare_metadata`` output

    # implement this — mandatory
    @classmethod
    def get_batch_keys(cls):
        """Return keys required in the collated batch after ``collate_fn``.

        Returns
        -------
        list of str
            todo: keys your ``collate_fn`` stacks, e.g. ``encoder_cat``, ``groups``, …
            Omit optional keys that only appear when static features exist unless you
            always include them.
        """
        return []  # todo: list keys your collate_fn produces

    # optional override — defaults to get_batch_keys()
    @classmethod
    def get_sample_item_keys(cls):
        """Return keys required in a single dataset item.

        Override when per-item keys differ from collated batch keys.

        Returns
        -------
        list of str
            Defaults to ``get_batch_keys()`` if not overridden.
        """
        return cls.get_batch_keys()
