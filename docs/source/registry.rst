Registry
========

.. currentmodule:: pytorch_forecasting._registry

The registry is the single authority for the tags that objects in
PyTorch Forecasting carry. A tag is an entry of an object's ``_tags``
dictionary; tags record what an object is, what it can do, and how the test
suite should treat it.

Looking tags up
---------------

.. autofunction:: all_tags

.. autofunction:: check_tag_is_valid

Finding objects by tag
----------------------

.. autofunction:: all_objects

Tag reference
-------------

Each tag below is documented with its possible values, its default, and what
in the package reads it.

.. automodule:: pytorch_forecasting._registry._tags
   :members:
   :undoc-members:
   :exclude-members: check_tag_is_valid
