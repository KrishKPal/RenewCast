# RenewCast ☀️
<img width="1197" height="702" alt="Screenshot 2026-10-14 at 4 41 52 PM" src="https://github.com/user-attachments/assets/f8665afd-1e29-488b-bdaa-41d1634ab361" />


**Solar power forecasting with model evaluation and performance monitoring.**

RenewCast is an open-source tool for forecasting hourly German solar power generation one hour ahead. It compares a machine learning model against a simple historical baseline and tracks how prediction errors change over time.

The project combines a forecasting pipeline with an interactive dashboard, making it easier to explore historical generation, compare models, and identify periods when forecasts become less reliable.

## Dashboard

<!-- Add screenshots here after uploading them to the repository. -->

### Features

- **One-hour-ahead forecasting** using historical solar generation data.
- **Model comparison** between gradient boosting and a 24-hour seasonal baseline.
- **Time-based evaluation** using MAE, RMSE, and normalized MAE.
- **Performance monitoring** that flags sustained increases in forecast error.
- **Interactive dashboard** for exploring generation history, predictions, model performance, and error trends.
- **Reproducible pipeline** with saved model artifacts and dataset validation.

## How it works

1. Download and prepare historical solar generation data.
2. Build time-based features from past observations and calendar information.
3. Train a gradient boosting model and compare it with the seasonal baseline.
4. Select the model with lower validation MAE.
5. Evaluate both models on a later, held-out test period.
6. Monitor the selected model's rolling forecast errors for sustained deterioration.

The evaluation uses chronological data splits to avoid training on future observations. The forecast horizon is **one hour**, not day-ahead.

## Getting started

### Requirements

- Python 3.11 or 3.12
- Git
- Internet access for the initial data download

Clone the repository and install the dependencies:

```bash
git clone <YOUR_REPOSITORY_URL>
cd renewcast

python3 -m venv .venv
source .venv/bin/activate

python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

### Download the data

RenewCast uses historical German solar generation observations from [Open Power System Data (OPSD)](https://data.open-power-system-data.org/time_series/2020-10-06/).

```bash
python scripts/download_data.py
```

The script validates the source data and prepares the hourly dataset. It does not replace missing observations with synthetic data.

### Train and evaluate

```bash
python scripts/train.py
python scripts/evaluate.py
```

### Launch the dashboard

```bash
streamlit run app.py
```

Open the local URL printed by Streamlit to explore the dashboard.

### Run the tests

```bash
PYTHONPATH=src python -m pytest -q
```

The tests use small, deterministic fixtures and do not require downloading the full dataset.

## Data

- **Source:** [OPSD Time Series — 2020-10-06 release](https://data.open-power-system-data.org/time_series/2020-10-06/)
- **Original data:** German solar generation from the `DE_solar_generation_actual` column, measured in MW.
- **Time standard:** UTC.
- **Coverage:** The source release covers the period from 2015 to mid-2020. The actual usable range depends on the available observations.
- **Attribution:** Open Power System Data, *Data Package Time series*, version 2020-10-06. [DOI](https://doi.org/10.25832/time_series/2020-10-06).

Review the [OPSD legal notes](https://open-power-system-data.org/legal/) and applicable [ENTSO-E terms](https://transparency.entsoe.eu/) before redistributing the underlying observations.

## Understanding the results

- **MAE:** Average absolute prediction error, in MW.
- **RMSE:** Prediction error metric that penalizes larger errors more heavily.
- **Normalized MAE:** MAE expressed as a percentage of mean training-period generation.
- **Performance warning:** Flags sustained increases in rolling forecast error relative to validation performance. It is a practical alert, not proof of statistical concept drift.

All displayed performance figures are calculated from the pipeline's saved evaluation outputs. Results depend on the dataset and test period.

## Limitations

RenewCast is a historical forecasting and analysis tool, not a live electricity forecasting service. It does not use weather forecasts or solar irradiance data, and it does not provide day-ahead predictions. Its monitoring rule identifies potential performance deterioration but does not explain its cause.

## Built with

Python · pandas · NumPy · scikit-learn · Streamlit · Matplotlib · pytest

## Contributing

Contributions, bug reports, and suggestions are welcome. Please open an issue to discuss substantial changes before submitting a pull request.

## License

The project's software license should be specified in the repository's `LICENSE` file. The license for the software does not automatically determine the redistribution rights for the underlying energy data.
