import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402
import streamlit as st  # noqa: E402

from renewcast.config import ARTIFACT_DIR, CLEAN_CSV  # noqa: E402
from renewcast.data import load_clean_csv  # noqa: E402
from renewcast.pipeline import load_artifacts, verify_data_integrity  # noqa: E402

st.set_page_config(page_title="RenewCast", layout="wide")
st.title("RenewCast — German Solar Power Forecasting")
st.caption("One-hour-ahead historical backtest • MW • UTC")

metrics_file = ARTIFACT_DIR / "metrics.csv"
predictions_file = ARTIFACT_DIR / "predictions.csv"
if not all(path.is_file() for path in (CLEAN_CSV, metrics_file, predictions_file)):
    st.info(
        "Missing data or evaluation outputs. From the repository root, run "
        "`python scripts/download_data.py`, `python scripts/train.py`, "
        "then `python scripts/evaluate.py`."
    )
    st.stop()

try:
    history = load_clean_csv(CLEAN_CSV)
    _, metadata = load_artifacts()
    verify_data_integrity(CLEAN_CSV, metadata)
    metrics = pd.read_csv(metrics_file)
    predictions = pd.read_csv(predictions_file, parse_dates=["timestamp"])
    predictions = predictions.set_index("timestamp").sort_index()
    if predictions.empty or metrics.empty:
        raise ValueError("Evaluation CSVs contain no results.")
    if not predictions.index.is_unique or predictions.index.tz is None:
        raise ValueError("Prediction timestamps must be unique and timezone-aware.")
except (OSError, ValueError, KeyError, ImportError) as exc:
    st.error(f"Could not load pipeline outputs: {exc}. Re-run the train and evaluate commands.")
    st.stop()

st.markdown(
    "**Forecast protocol:** At the end of UTC hour **t−1**, after that hour's actual "
    "generation becomes available, predict generation for **hour t**. Backtesting "
    "updates lag features with observed values after each hour; it is not a "
    "24-hour-ahead or multi-step forecast."
)
st.write(
    f"Selected by validation MAE: **{metadata['selected_model']}**. "
    f"Test period: **{metadata['test_start']}** through **{metadata['test_end']}**."
)

st.subheader("Historical solar generation")
history_days = st.slider("Show most recent historical days", 7, 180, 30)
st.line_chart(history["power_mw"].tail(history_days * 24), y_label="MW")

st.subheader("Test-period model comparison")
st.dataframe(metrics.set_index("model"), use_container_width=True)
st.caption(
    "MAE and RMSE are measured in MW. nMAE (%) = 100 × test MAE / "
    "mean observed generation in the training period (same denominator for both models)."
)

test_days = st.slider("Show most recent test days", 2, 90, 14)
recent = predictions.tail(test_days * 24)
st.subheader("Actual versus predictions (unseen test period)")
st.line_chart(
    recent[["actual_mw", "seasonal_naive_24h_mw", "gradient_boosting_mw"]],
    y_label="MW",
)

st.subheader("Errors and residuals")
st.line_chart(recent[["absolute_error_mw", "rolling_mae_mw"]], y_label="MW")
residuals = predictions["actual_mw"] - predictions["selected_prediction_mw"]
figure, axis = plt.subplots(figsize=(7, 3))
axis.hist(residuals.dropna(), bins=40)
axis.axvline(0, color="black", linewidth=1)
axis.set_xlabel("Actual − predicted (MW)")
axis.set_ylabel("Test hours")
st.pyplot(figure)
plt.close(figure)

st.subheader("Rolling-error drift monitor")
threshold = metadata["drift_reference_mae_mw"] * metadata["drift_multiplier"]
st.write(
    f"Rule: a **{metadata['drift_window_hours']}-hour** rolling mean absolute "
    f"error must exceed **{metadata['drift_multiplier']}×** validation MAE "
    f"(**{threshold:.2f} MW**) for **{metadata['drift_consecutive_hours']} "
    "consecutive hours**. Missing-hour windows cannot trigger an alert. "
    "This measures error deterioration, not input-distribution drift."
)
warning_count = int(predictions["drift_warning"].sum())
if warning_count:
    st.warning(f"Drift rule triggered at {warning_count} test timestamps.")
    st.dataframe(
        predictions.loc[predictions["drift_warning"],
                        ["rolling_mae_mw", "absolute_error_mw"]].tail(20)
    )
else:
    st.success("No drift warnings were triggered in this test period.")

st.caption(
    "Limitations: historical OPSD observations end around mid-2020; no live data or "
    "weather inputs are used. Hourly forecasts assume the preceding hour's actual "
    "generation is already known; missing history prevents a prediction. Alerts are "
    "heuristic and do not establish a statistical cause."
)
