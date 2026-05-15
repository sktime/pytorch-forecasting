Utilities
=========

.. currentmodule:: pytorch_forecasting.utils

This module provides utility functions for data manipulation, preprocessing, and common operations.

Core Utilities
---------------

.. autosummary::
   :toctree: ../api

   create_mask
   apply_to_list
   to_list
   concat_sequences
   unpack_sequence
   unsqueeze_like
   get_embedding_size
   detach
   masked_op
   move_to_device
   padded_stack
   groupby_apply
   integer_histogram
   autocorrelation

network Components:
-------------------

.. autosummary::
   :toctree: ../api

   RecurrentNetwork
   DecoderMLP

Mixins
------

.. autosummary::
   :toctree: ../api

   InitialParameterRepresenterMixIn
   OutputMixIn
   TupleOutputMixIn

Development / Profiling
-----------------------

.. autosummary::
   :toctree: ../api

   profile
   redirect_stdout
   repr_class

Estimator Checks
----------------

.. autosummary::
   :toctree: ../api

   check_estimator
   parametrize_with_checks
