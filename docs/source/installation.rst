Contribute
==========

Contributions to PyTorch Forecasting are very welcome! You do not have to be an expert in deep learning
to contribute. If you find a bug - fix it! If you miss a feature - propose it!

Contribution guidelines
------------------------

* Open issues to discuss your proposed changes before starting pull requests.
  This ensures that your contribution will be swiftly integrated into the code base.

* Mark your PR with ``ready for review`` to indicate that you are done with it and
  request the maintainers to have a look.

* To contribute, fork and clone the repository, install depdencies with ``poetry install``,
  create a new branch from master such as ``feature/my_new_awesome_model``, write your code
  and create the PR on GitHub.


Design principles
------------------

* Backward compatible API if possible to prevent breaking code.
* Powerful abstractions to enable quick experimentation. At the same time, the abstractions should
  allow the user to still take full control.
* Intuitive default values that do not need changing in most cases.
* Focus on forecasting time-related data - specificially timeseries regression and classificiation.
  Contributions not directly related to this topic might not be merged. We want to keep the library as
  crisp as possible.
* Always add tests and documentation to new features.
