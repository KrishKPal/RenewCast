# RenewCast — Renewable Energy Forecasting and Model Drift Monitoring

RenewCast forecasts hourly **German solar generation (MW)** one hour ahead using public historical observations. It compares a 24-hour seasonal-naive baseline with scikit-learn histogram gradient boosting, holds out the latest chronological period for testing, and checks whether test forecast errors have persistently increased relative to validation error. The Streamlit dashboard reads outputs produced by the scripts; it contains no example performance numbers.

## Requirements and setup

Run all commands **from the repository root** (`renewcast/`), on macOS or Linux with Python 3.11 or 3.12:

```bash
python3 --version
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

Download and clean the original data (about 124 MB):

```bash
python scripts/download_data.py
```

If the original source is down or you have already downloaded it, get the original **60-minute singleindex CSV** using the package link below, then run:

```bash
python scripts/download_data.py --raw-csv /absolute/path/to/time_series_60min_singleindex.csv
```

A different source for the *same CSV schema* may be passed with `--url`:

```bash
python scripts/download_data.py --url 'https://data.open-power-system-data.org/time_series/2020-10-06/time_series_60min_singleindex.csv'
```

The script reports download/validation failures and does **not** generate replacement observations. It saves `data/raw/time_series_60min_singleindex.csv` (unless you use `--raw-csv`) and `data/processed/germany_solar_hourly.csv`. Clean data contain UTC `timestamp` and `power_mw`. It also prints the exact observed date range and missing-hour count.

Then train, evaluate on the held-out test period, start the dashboard, and run tests:

```bash
python scripts/train.py
python scripts/evaluate.py
streamlit run app.py
PYTHONPATH=src python -m pytest -q
```

To stop Streamlit, press Ctrl+C. To leave the environment, use `deactivate`.

## Data source and licensing

- Source: **Open Power System Data (OPSD), Time series, version 2020-10-06**. Download: https://data.open-power-system-data.org/time_series/2020-10-06/ ; CSV: https://data.open-power-system-data.org/time_series/2020-10-06/time_series_60min_singleindex.csv
- Attribution: *Open Power System Data. 2020. Data Package Time series. Version 2020-10-06.* https://doi.org/10.25832/time_series/2020-10-06
- Selected original columns: `utc_timestamp` and `DE_solar_generation_actual` (MW), derived from ENTSO-E Transparency data. The publisher describes coverage as **2015 to mid-2020** for this release. The actual non-null German series' first and last timestamps are determined at download time and printed by the script, rather than presumed.
- **Rights:** This repository does not redistribute OPSD/ENTSO-E data. OPSD describes CC BY as a recommended licence, but that does not establish a blanket CC BY licence for all underlying time-series observations. The German solar input is attributed to **ENTSO-E Transparency**, and its upstream usage terms may apply; verify both the [OPSD legal notes](https://open-power-system-data.org/legal/) and [ENTSO-E terms](https://transparency.entsoe.eu/) before publishing or commercially redistributing observations. Source data are obtained by each user directly.
- Limitations: historical, national aggregate—not live generation, regional rooftop capacity, or a weather-conditioned power forecast. Time stamps are in UTC (not German local time). Historical measurements can contain outages, revisions, or missing values; earlier local dates do not guarantee continuous coverage for the German series.

## Methodology

**Forecast origin and horizon.** For target hour **t**, issue a prediction at the **end of hour t−1**, when the actual generation for t−1 is available. This is a **one-hour-ahead, continuously updated** forecast—not a 24-hour-ahead forecast. On historical backtests, observations from hours earlier than t are permitted as inputs even when those hours fall within the test period. The model never consumes actual generation from t or later to predict t.

**Validation and missing values.** Reject invalid/duplicate/unrounded UTC timestamps, non-numeric, negative and infinite outputs. Sort valid timestamps, trim leading/trailing missing targets, and reindex to an hourly UTC grid. Missing original values or omitted hours remain missing; **no interpolation, future-value fill, or synthetic replacements**. Rows with missing targets or incomplete feature windows are excluded from supervised fitting, prediction and metrics. During drift monitoring, a gap resets the full-hour rolling window.

**Features.** Prior values at t−1, t−24, and t−168; mean of preceding 24 and 168 hours, standard deviation of preceding 24 hours (all based on `shift(1)`); UTC hour and annual cyclical encodings plus weekday. Lag and window features at t use actual generation strictly before t. The seasonal-naive forecast is the observation at t−24. Calendar features use the forecast hour itself (known in advance). For an API-style single forecast, `renewcast.models.predict_next_hour(history, forecast_time, model, model_name)` checks that the history ends at t−1 and has a complete 168-hour lookback. It can use a saved boosting model loaded with `renewcast.pipeline.load_artifacts()` or the naive baseline.

**Models and time splits.** After feature construction and filtering, split usable rows by ascending time: 70% training, 15% validation, 15% untouched test (at least 24 rows per split). Train `HistGradientBoostingRegressor` only on the training segment, without scaling (tree model); its settings are fixed in `models.py`. Compare the gradient-boosting model and the seasonal-naive baseline on validation MAE; choose the lower-MAE model for monitoring. No test metrics inform that selection. Evaluate **both** models exactly once per invocation of `evaluate.py`, without refitting. Artifacts include SHA-256 of the clean input CSV, split timestamps, selected model, feature order and validation scores. Evaluation rejects modified datasets. Running `train.py` invalidates old evaluation CSVs.

**Metrics.** The comparison CSV reports MAE (MW), RMSE (MW), and **nMAE (%) = 100 × MAE / mean actual generation (MW) of the training subset**. Both models use the same positive training-only normalization denominator, not a test-derived or per-hour solar denominator. RMSE penalizes larger errors. Nighttime zero generation and seasonal imbalance can strongly influence aggregate scores.

**Error deterioration alert.** For the validation-selected model, compute absolute test residuals and a trailing **24-hour rolling MAE** (requiring 24 consecutive hourly errors). Trigger `drift_warning` only after that rolling MAE **exceeds 1.5 × selected-model validation MAE for 3 consecutive hours**. A missing hour makes the 24-hour window incomplete and resets consecutive breaches. The threshold is fixed before test evaluation. The zero-validation-MAE case uses a threshold of zero (positive rolling errors trigger). This is a heuristic *performance degradation* check, not a statistical concept-drift detector or proof of causation. Warnings are not guaranteed; the dashboard displays the actual result.

## Artifacts and layout

- `data/raw/`: downloaded original CSV (ignored by Git).
- `data/processed/germany_solar_hourly.csv`: timestamp-aligned observed series; missing target rows are retained (ignored by Git).
- `artifacts/model.joblib`: training-fitted scikit-learn model. Only load locally generated joblib files you trust.
- `artifacts/metadata.json`: feature configuration, forecast origin, dataset hash, split boundaries, validation scores, selection and drift rule.
- `artifacts/metrics.csv`: **test-only** model comparison, created by `evaluate.py`.
- `artifacts/predictions.csv`: test actuals, both model forecasts, selected-model residual errors, rolling errors and warning flags, created by `evaluate.py`.

All default paths are derived from the repository location, not the terminal's current directory. The commands above assume the repository root for clarity.

## Reproducibility and limitations

- Python 3.11/3.12, dependencies restricted to compatible ranges in `requirements.txt`, fixed random seed 42. Train/evaluate on the same unmodified CSV and versions for comparable results. Dependency version upgrades may change numerical results.
- No internet is required for tests: pytest uses small deterministic synthetic fixtures **only**, never substitutes synthetic data for the main dataset.
- Test estimates represent a single later historical period rather than repeated walk-forward seasons; reported results can depend on that period. Training only on the first 70% also means no test-period refitting. Training and test rows can legitimately share *past observations* as features at the split boundary; no later information is used.
- No irradiance/weather covariates, uncertainty intervals, night-specific metrics, installed-capacity normalization, or live deployment. Reanalysis, missing observations and delayed actual publication may limit operational performance; forecasting assumes t−1 has been published at origin.
- A one-hour-ahead result should not be presented as day-ahead forecasting. Performance numbers are intentionally absent from this README until the pipeline has been run on real observations.
