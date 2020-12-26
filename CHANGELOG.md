# Changelog

## v0.8.0 Unreleased

### Added

- Adding support for multiple targets in the TimeSeriesDataSet (#199) and amended tutorials.
- Temporal fusion transformer with support for multiple tagets (#199)

### Changed

- TimeSeriesDataSet's `y` of the dataloader is a tuple of (target(s), weight) - potentially breaking for model or metrics implementation
  Most implementations will not be affected as hooks in BaseModel and MultiHorizonMetric were modified.

### Contributors

- jdb78
- JustinNeumann

## v0.7.1 Tutorial on how to implement a new architecture (07/12/2020)

### Added

- Tutorial on how to implement a new architecture covering basic and advanced use cases (#188)
- Additional and improved documentation - particularly of implementation details (#188)

### Changed (breaking for new model implementations)

- Moved multiple private methods to public methods (particularly logging) (#188)
- Moved `get_mask` method from BaseModel into utils module (#188)
- Instead of using label to communicate if model is training or validating, using `self.training` attribute (#188)
- Using `sample((n,))` of pytorch distributions instead of deprecated `sample_n(n)` method (#188)

---

## v0.7.0 New API for transforming inputs and outputs with encoders (03/12/2020)

### Added

- Beta distribution loss for probabilistic models such as DeepAR (#160)

### Changed

- BREAKING: Simplifying how to apply transforms (such as logit or log) before and after applying encoder. Some transformations are included by default but a tuple of a forward and reverse transform function can be passed for arbitrary transformations. This requires to use a `transformation` keyword in target normalizers instead of, e.g. `log_scale` (#185)

### Fixed

- Incorrect target position if `len(static_reals) > 0` leading to leakage (#184)
- Fixing predicting completely unseen series (#172)

### Contributors

- jdb78
- JakeForsey

---

## v0.6.1 Bugfixes and DeepAR improvements (24/11/2020)

### Added

- Using GRU cells with DeepAR (#153)

### Fixed

- GPU fix for variable sequence length (#169)
- Fix incorrect syntax for warning when removing series (#167)
- Fix issue when using unknown group ids in validation or test dataset (#172)
- Run non-failing CI on PRs from forks (#166, #156)

### Docs

- Improved model selection guidance and explanations on how TimeSeriesDataSet works (#148)
- Clarify how to use with conda (#168)

### Contributors

- jdb78
- JakeForsey

---

## v0.6.0 Adding DeepAR (10/11/2020)

### Added

- DeepAR by Amazon (#115)
  - First autoregressive model in PyTorch Forecasting
  - Distribution loss: normal, negative binomial and log-normal distributions
  - Currently missing: handling lag variables and tutorial (planned for 0.6.1)
- Improved documentation on TimeSeriesDataSet and how to implement a new network (#145)

### Changed

- Internals of encoders and how they store center and scale (#115)

### Fixed

- Update to PyTorch 1.7 and PyTorch Lightning 1.0.5 which came with breaking changes for CUDA handling and with optimizers (PyTorch Forecasting Ranger version) (#143, #137, #115)

### Contributors

- jdb78
- JakeForesey

---

## v0.5.3 Bug fixes (31/10/2020)

### Fixes

- Fix issue where hyperparameter verbosity controlled only part of output (#118)
- Fix occasional error when `.get_parameters()` from `TimeSeriesDataSet` failed (#117)
- Remove redundant double pass through LSTM for temporal fusion transformer (#125)
- Prevent installation of pytorch-lightning 1.0.4 as it breaks the code (#127)
- Prevent modification of model defaults in-place (#112)

---

## v0.5.2 Fixes to interpretation and more control over hyperparameter verbosity (18/10/2020)

### Added

- Hyperparameter tuning with optuna to tutorial
- Control over verbosity of hyper parameter tuning

### Fixes

- Interpretation error when different batches had different maximum decoder lengths
- Fix some typos (no changes to user API)

---

## v0.5.1 PyTorch Lightning 1.0 compatibility (14/10/2020)

This release has only one purpose: Allow usage of PyTorch Lightning 1.0 - all tests have passed.

---

## v0.5.0 PyTorch Lightning 0.10 compatibility and classification (12/10/2020)

### Added

- Additional checks for `TimeSeriesDataSet` inputs - now flagging if series are lost due to high `min_encoder_length` and ensure parameters are integers
- Enable classification - simply change the target in the `TimeSeriesDataSet` to a non-float variable, use the `CrossEntropy` metric to optimize and output as many classes as you want to predict

### Changed

- Ensured PyTorch Lightning 0.10 compatibility
  - Using `LearningRateMonitor` instead of `LearningRateLogger`
  - Use `EarlyStopping` callback in trainer `callbacks` instead of `early_stopping` argument
  - Update metric system `update()` and `compute()` methods
  - Use `trainer.tuner.lr_find()` instead of `trainer.lr_find()` in tutorials and examples
- Update poetry to 1.1.0

---

## v0.4.1 Various fixes models and data (01/10/2020)

### Fixes

#### Model

- Removed attention to current datapoint in TFT decoder to generalise better over various sequence lengths
- Allow resuming optuna hyperparamter tuning study

#### Data

- Fixed inconsistent naming and calculation of `encoder_length`in TimeSeriesDataSet when added as feature

### Contributors

- jdb78

---

## v0.4.0 Metrics, performance, and subsequence detection (28/09/2020)

### Added

#### Models

- Backcast loss for N-BEATS network for better regularisation
- logging_metrics as explicit arguments to models

#### Metrics

- MASE (Mean absolute scaled error) metric for training and reporting
- Metrics can be composed, e.g. `0.3* metric1 + 0.7 * metric2`
- Aggregation metric that is computed on mean prediction over all samples to reduce mean-bias

#### Data

- Increased speed of parsing data with missing datapoints. About 2s for 1M data points. If `numba` is installed, 0.2s for 1M data points
- Time-synchronize samples in batches: ensure that all samples in each batch have with same time index in decoder

### Breaking changes

- Improved subsequence detection in TimeSeriesDataSet ensures that there exists a subsequence starting and ending on each point in time.
- Fix `min_encoder_length = 0` being ignored and processed as `min_encoder_length = max_encoder_length`

### Contributors

- jdb78
- dehoyosb

---

## v0.3.1 More tests and better docs (13/09/2020)

- More tests driving coverage to ~90%
- Performance tweaks for temporal fusion transformer
- Reformatting with sort
- Improve documentation - particularly expand on hyper parameter tuning

### Fixed

- Fix PoissonLoss quantiles calculation
- Fix N-Beats visualisations

---

## v0.3.0 More testing and interpretation features (02/09/2020)

### Added

- Calculating partial dependency for a variable
- Improved documentation - in particular added FAQ section and improved tutorial
- Data for examples and tutorials can now be downloaded. Cloning the repo is not a requirement anymore
- Added Ranger Optimizer from `pytorch_ranger` package and fixed its warnings (part of preparations for conda package release)
- Use GPU for tests if available as part of preparation for GPU tests in CI

### Changes

- **BREAKING**: Fix typo “add_decoder_length” to “add_encoder_length” in TimeSeriesDataSet

### Bugfixes

- Fixing plotting predictions vs actuals by slicing variables

---

## v0.2.4 Fix edge case in prediction logging (26/08/2020)

### Fixed

Fix bug where predictions were not correctly logged in case of `decoder_length == 1`.

### Added

- Add favicon to docs page

---

## v0.2.3 Make pip installable from master branch (23/08/2020)

Update build system requirements to be parsed correctly when installing with `pip install https://github.com/jdb78/pytorch-forecasting/`

---

## v0.2.2 Improving tests (23/08/2020)

- Add tests for MacOS
- Automatic releases
- Coverage reporting

---

## v0.2.1 Patch release (23/08/2020)

This release improves robustness of the code.

Fixing bug across code, in particularly

- Ensuring that code works on GPUs
- Adding tests for models, dataset and normalisers
- Test using GitHub Actions (tests on GPU are still missing)

Extend documentation by improving docstrings and adding two tutorials.

## Improving default arguments for TimeSeriesDataSet to avoid surprises

---

## v0.2.0 Minor release (16/08/2020)

### Added

- Basic tests for data and model (mostly integration tests)
- Automatic target normalization
- Improved visualization and logging of temporal fusion transformer
- Model bugfixes and performance improvements for temporal fusion transformer

### Modified

- Metrics are reduced to calculating loss. Target transformations are done by new target transformer
