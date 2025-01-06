# Release Notes

## v1.2.0

Maintenance update, minor feature additions and bugfixes.

* support for `numpy 2.X`
* end of life for `python 3.8`
* fixed documentation build
* bugfixes

### Dependency changes

* `pytorch-forecasting` is now compatible with `numpy 2.X` (core dependency)
* `optuna` (tuning soft dependency) bounds have been update to `>=3.1.0,<5.0.0`

### Fixes

* [BUG] fix `AttributeError: 'ExperimentWriter' object has no attribute 'add_figure'` by @ewth in https://github.com/sktime/pytorch-forecasting/pull/1694

### Documentation

* [DOC] typo fixes in changelog by @fkiraly in https://github.com/sktime/pytorch-forecasting/pull/1660
* [DOC] update URLs to `sktime` org by @fkiraly in https://github.com/sktime/pytorch-forecasting/pull/1674

### Maintenance

* [MNT] handle `mps backend` for lower versions of pytorch and fix `mps` failure on `macOS-latest` runner by @fnhirwa in https://github.com/sktime/pytorch-forecasting/pull/1648
* [MNT] updates the actions in the doc build CI by @fkiraly in https://github.com/sktime/pytorch-forecasting/pull/1673
* [MNT] fixes to `readthedocs.yml` by @fkiraly in https://github.com/sktime/pytorch-forecasting/pull/1676
* [MNT] updates references in CI and doc locations to `main` by @fkiraly in https://github.com/sktime/pytorch-forecasting/pull/1677
* [MNT] `show_versions` utility by @fkiraly in https://github.com/sktime/pytorch-forecasting/pull/1688
* [MNT] Relax `numpy` bound to `numpy<3.0.0` by @XinyuWuu in https://github.com/sktime/pytorch-forecasting/pull/1624
* [MNT] fix `pre-commit` failures on `main` by @ewth in https://github.com/sktime/pytorch-forecasting/pull/1696
* [MNT] Move linting to ruff by @airookie17 in https://github.com/sktime/pytorch-forecasting/pull/1692
1693
* [MNT] `ruff` linting - allow use of assert (S101) by @fkiraly in https://github.com/sktime/pytorch-forecasting/pull/1701
* [MNT] `ruff` - fix list related linting failures C416 and C419 by @fkiraly in https://github.com/sktime/pytorch-forecasting/pull/1702
* [MNT] Delete poetry.lock by @benHeid in https://github.com/sktime/pytorch-forecasting/pull/1704
* [MNT] fix `black` doesn't have `extras` dependency by @fnhirwa in https://github.com/sktime/pytorch-forecasting/pull/1697
* [MNT] Remove mutable objects from defaults by @eugenio-mercuriali in https://github.com/sktime/pytorch-forecasting/pull/1699
* [MNT] remove docs build in ci for all pr by @yarnabrina in https://github.com/sktime/pytorch-forecasting/pull/1712
* [MNT] EOL for python 3.8 by @fkiraly in https://github.com/sktime/pytorch-forecasting/pull/1661
* [MNT] remove `poetry.lock` by @fkiraly in https://github.com/sktime/pytorch-forecasting/pull/1651
* [MNT] update `pre-commit` requirement from `<4.0.0,>=3.2.0` to `>=3.2.0,<5.0.0` by @dependabot in https://github.com/sktime/pytorch-forecasting/pull/
* [MNT] update optuna requirement from `<4.0.0,>=3.1.0` to `>=3.1.0,<5.0.0` by @dependabot in https://github.com/sktime/pytorch-forecasting/pull/1715
* [MNT] CODEOWNERS file by @fkiraly in https://github.com/sktime/pytorch-forecasting/pull/1710

### All Contributors

@airookie17,
@benHeid,
@eugenio-mercuriali,
@ewth,
@fkiraly,
@fnhirwa,
@XinyuWuu,
@yarnabrina

## v1.1.1

Hotfix for accidental package name change in `pyproject.toml`.

The package name is now corrected to `pytorch-forecasting`.


## v1.1.0

Maintenance update widening compatibility ranges and consolidating dependencies:

* support for python 3.11 and 3.12, added CI testing
* support for MacOS, added CI testing
* core dependencies have been minimized to `numpy`, `torch`, `lightning`, `scipy`, `pandas`, and `scikit-learn`.
* soft dependencies are available in soft dependency sets: `all_extras` for all soft dependencies, and `tuning` for `optuna` based optimization.

### Dependency changes

* the following are no longer core dependencies and have been changed to optional dependencies : `optuna`, `statsmodels`, `pytorch-optimize`, `matplotlib`. Environments relying on functionality requiring these dependencies need to be updated to install these explicitly.
* `optuna` bounds have been updated to `optuna >=3.1.0,<4.0.0`
* `optuna-integrate` is now an additional soft dependency, in case of `optuna >=3.3.0`

### Deprecations and removals

* from 1.2.0, the default optimizer will be changed from `"ranger"` to `"adam"` to avoid non-`torch` dependencies in defaults. `pytorch-optimize` optimizers can still be used. Users should set the optimizer explicitly to continue using `"ranger"`.
*  from 1.1.0, the loggers do not log figures if soft dependency `matplotlib` is not present, but will raise no exceptions in this case. To log figures, ensure that `matplotlib` is installed.

## v1.0.0 Update to pytorch 2.0 (10/04/2023)


### Breaking Changes

- Upgraded to pytorch 2.0 and lightning 2.0. This brings a couple of changes, such as configuration of trainers. See the [lightning upgrade guide](https://lightning.ai/docs/pytorch/latest/upgrade/migration_guide.html). For PyTorch Forecasting, this particularly means if you are developing own models, the class method `epoch_end` has been renamed to `on_epoch_end` and replacing `model.summarize()` with `ModelSummary(model, max_depth=-1)` and `Tuner(trainer)` is its own class, so `trainer.tuner` needs replacing. (#1280)
- Changed the `predict()` interface returning named tuple - see tutorials.

### Changes

- The predict method is now using the lightning predict functionality and allows writing results to disk (#1280).

### Fixed

- Fixed robust scaler when quantiles are 0.0, and 1.0, i.e. minimum and maximum (#1142)

## v0.10.3 Poetry update (07/09/2022)

### Fixed

- Removed pandoc from dependencies as issue with poetry install (#1126)
- Added metric attributes for torchmetric resulting in better multi-GPU performance (#1126)

### Added

- "robust" encoder method can be customized by setting "center", "lower" and "upper" quantiles (#1126)

## v0.10.2 Multivariate networks (23/05/2022)

### Added

- DeepVar network (#923)
- Enable quantile loss for N-HiTS (#926)
- MQF2 loss (multivariate quantile loss) (#949)
- Non-causal attention for TFT (#949)
- Tweedie loss (#949)
- ImplicitQuantileNetworkDistributionLoss (#995)

### Fixed

- Fix learning scale schedule (#912)
- Fix TFT list/tuple issue at interpretation (#924)
- Allowed encoder length down to zero for EncoderNormalizer if transformation is not needed (#949)
- Fix Aggregation and CompositeMetric resets (#949)

### Changed

- Dropping Python 3.6 suppport, adding 3.10 support (#479)
- Refactored dataloader sampling - moved samplers to pytorch_forecasting.data.samplers module (#479)
- Changed transformation format for Encoders to dict from tuple (#949)

### Contributors

- jdb78

## v0.10.1 Bugfixes (24/03/2022)

### Fixed

- Fix with creating tensors on correct devices (#908)
- Fix with MultiLoss when calculating gradient (#908)

### Contributors

- jdb78

## v0.10.0 Adding N-HiTS network (N-BEATS successor) (23/03/2022)

### Added

- Added new `N-HiTS` network that has consistently beaten `N-BEATS` (#890)
- Allow using [torchmetrics](https://torchmetrics.readthedocs.io/) as loss metrics (#776)
- Enable fitting `EncoderNormalizer()` with limited data history using `max_length` argument (#782)
- More flexible `MultiEmbedding()` with convenience `output_size` and `input_size` properties (#829)
- Fix concatentation of attention (#902)

### Fixed

- Fix pip install via github (#798)

### Contributors

- jdb78
- christy
- lukemerrick
- Seon82

## v0.9.2 Maintenance Release (30/11/2021)

### Added

- Added support for running `lightning.trainer.test` (#759)

### Fixed

- Fix inattention mutation to `x_cont` (#732).
- Compatability with pytorch-lightning 1.5 (#758)

### Contributors

- eavae
- danielgafni
- jdb78

## v0.9.1 Maintenance Release (26/09/2021)

### Added

- Use target name instead of target number for logging metrics (#588)
- Optimizer can be initialized by passing string, class or function (#602)
- Add support for multiple outputs in Baseline model (#603)
- Added Optuna pruner as optional parameter in `TemporalFusionTransformer.optimize_hyperparameters` (#619)
- Dropping support for Python 3.6 and starting support for Python 3.9 (#639)

### Fixed

- Initialization of TemporalFusionTransformer with multiple targets but loss for only one target (#550)
- Added missing transformation of prediction for MLP (#602)
- Fixed logging hyperparameters (#688)
- Ensure MultiNormalizer fit state is detected (#681)
- Fix infinite loop in TimeDistributedEmbeddingBag (#672)

### Contributors

- jdb78
- TKlerx
- chefPony
- eavae
- L0Z1K

## v0.9.0 Simplified API (04/06/2021)

### Breaking changes

- Removed `dropout_categoricals` parameter from `TimeSeriesDataSet`.
  Use `categorical_encoders=dict(<variable_name>=NaNLabelEncoder(add_nan=True)`) instead (#518)
- Rename parameter `allow_missings` for `TimeSeriesDataSet` to `allow_missing_timesteps` (#518)
- Transparent handling of transformations. Forward methods should now call two new methods (#518):

  - `transform_output` to explicitly rescale the network outputs into the de-normalized space
  - `to_network_output` to create a dict-like named tuple. This allows tracing the modules with PyTorch's JIT. Only `prediction` is still required which is the main network output.

  Example:

  ```python
  def forward(self, x):
      normalized_prediction = self.module(x)
      prediction = self.transform_output(prediction=normalized_prediction, target_scale=x["target_scale"])
      return self.to_network_output(prediction=prediction)
  ```

### Fixed

- Fix quantile prediction for tensors on GPUs for distribution losses (#491)
- Fix hyperparameter update for RecurrentNetwork.from_dataset method (#497)

### Added

- Improved validation of input parameters of TimeSeriesDataSet (#518)

## v0.8.5 Generic distribution loss(es) (27/04/2021)

### Added

- Allow lists for multiple losses and normalizers (#405)
- Warn if normalization is with scale `< 1e-7` (#429)
- Allow usage of distribution losses in all settings (#434)

### Fixed

- Fix issue when predicting and data is on different devices (#402)
- Fix non-iterable output (#404)
- Fix problem with moving data to CPU for multiple targets (#434)

### Contributors

- jdb78
- domplexity

## v0.8.4 Simple models (07/03/2021)

### Added

- Adding a filter functionality to the timeseries datasset (#329)
- Add simple models such as LSTM, GRU and a MLP on the decoder (#380)
- Allow usage of any torch optimizer such as SGD (#380)

### Fixed

- Moving predictions to CPU to avoid running out of memory (#329)
- Correct determination of `output_size` for multi-target forecasting with the TemporalFusionTransformer (#328)
- Tqdm autonotebook fix to work outside of Jupyter (#338)
- Fix issue with yaml serialization for TensorboardLogger (#379)

### Contributors

- jdb78
- JakeForsey
- vakker

## v0.8.3 Bugfix release (31/01/2021)

### Added

- Make tuning trainer kwargs overwritable (#300)
- Allow adding categories to NaNEncoder (#303)

### Fixed

- Underlying data is copied if modified. Original data is not modified inplace (#263)
- Allow plotting of interpretation on passed figure for NBEATS (#280)
- Fix memory leak for plotting and logging interpretation (#311)
- Correct shape of `predict()` method output for multi-targets (#268)
- Remove cloudpickle to allow GPU trained models to be loaded on CPU devices from checkpoints (#314)

### Contributors

- jdb78
- kigawas
- snumumrik

## v0.8.2 Fix for output transformer (12/01/2021)

- Added missing output transformation which was switched off by default (#260)

## v0.8.1 Adding support for lag variables (10/01/2021)

### Added

- Add "Release Notes" section to docs (#237)
- Enable usage of lag variables for any model (#252)

### Changed

- Require PyTorch>=1.7 (#245)

### Fixed

- Fix issue for multi-target forecasting when decoder length varies in single batch (#249)
- Enable longer subsequences for min_prediction_idx that were previously wrongfully excluded (#250)

### Contributors

- jdb78

---

## v0.8.0 Adding multi-target support (03/01/2021)

### Added

- Adding support for multiple targets in the TimeSeriesDataSet (#199) and amended tutorials.
- Temporal fusion transformer and DeepAR with support for multiple tagets (#199)
- Check for non-finite values in TimeSeriesDataSet and better validate scaler argument (#220)
- LSTM and GRU implementations that can handle zero-length sequences (#235)
- Helpers for implementing auto-regressive models (#236)

### Changed

- TimeSeriesDataSet's `y` of the dataloader is a tuple of (target(s), weight) - potentially breaking for model or metrics implementation
  Most implementations will not be affected as hooks in BaseModel and MultiHorizonMetric were modified. (#199)

### Fixed

- Fixed autocorrelation for pytorch 1.7 (#220)
- Ensure reproducibility by replacing python `set()` with `dict.fromkeys()` (mostly TimeSeriesDataSet) (#221)
- Ensures BetaDistributionLoss does not lead to infinite loss if actuals are 0 or 1 (#233)
- Fix for GroupNormalizer if scaling by group (#223)
- Fix for TimeSeriesDataSet when using `min_prediction_idx` (#226)

### Contributors

- jdb78
- JustinNeumann
- reumar
- rustyconover

---

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
  - Use `Tuner(trainer).lr_find()` instead of `trainer.lr_find()` in tutorials and examples
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

- **BREAKING**: Fix typo "add_decoder_length" to "add_encoder_length" in TimeSeriesDataSet

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

Update build system requirements to be parsed correctly when installing with `pip install git+https://github.com/jdb78/pytorch-forecasting`

---

## v0.2.2 Improving tests (23/08/2020)

- Add tests for MacOS
- Automatic releases
- Coverage reporting

---

## v0.2.1 Patch release (23/08/2020)

This release improves robustness of the code.

- Fixing bug across code, in particularly

  - Ensuring that code works on GPUs
  - Adding tests for models, dataset and normalisers
  - Test using GitHub Actions (tests on GPU are still missing)

- Extend documentation by improving docstrings and adding two tutorials.
- Improving default arguments for TimeSeriesDataSet to avoid surprises

---

## v0.2.0 Minor release (16/08/2020)

### Added

- Basic tests for data and model (mostly integration tests)
- Automatic target normalization
- Improved visualization and logging of temporal fusion transformer
- Model bugfixes and performance improvements for temporal fusion transformer

### Modified

- Metrics are reduced to calculating loss. Target transformations are done by new target transformer
