"""RenewCast dashboard for historical solar generation forecasts."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st

from renewcast.config import ARTIFACT_DIR, CLEAN_CSV
from renewcast.data import load_clean_csv
from renewcast.evaluation import score_forecast
from renewcast.pipeline import load_artifacts, verify_data_integrity

st.set_page_config(
    page_title="RenewCast | Solar Generation Forecasts",
    page_icon="☀️",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
    :root { --accent: #43dfbe; }
    .stApp { background: #0b1220; color: #e8f0f7; }
    [data-testid="stSidebar"] { background: #111c2c; }
    .block-container { max-width: 1440px; padding-top: 2rem; }
    h1, h2, h3 { letter-spacing: -0.035em; }
    h1 { font-weight: 800; }
    div[data-testid="stMetric"] {
        border: 1px solid #28384d;
        border-radius: 15px;
        padding: 1.05rem 1.2rem;
        background: linear-gradient(135deg, #152439, #101a2a);
    }
    div[data-testid="stMetricLabel"] { color: #a7bacd; }
    div[data-testid="stMetricValue"] { color: #f0f8fc; }
    .intro { color: #aebed0; margin-bottom: 1.2rem; }
    .eyebrow { color: #43dfbe; font-size: .8rem; font-weight: 700;
               letter-spacing: .13em; text-transform: uppercase; }
    .section-note { color: #a6b9cb; font-size: .94rem; }
    div[data-testid="stTabs"] button { font-weight: 650; }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_data(show_spinner="Loading data…")
def read_outputs(data_path: str, metrics_path: str, predictions_path: str,
                 signatures: tuple) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Cache CSV parsing; modified file timestamps invalidate the cache."""
    del signatures
    history = load_clean_csv(Path(data_path))
    metrics = pd.read_csv(metrics_path)
    predictions = pd.read_csv(predictions_path)
    if "timestamp" not in predictions.columns:
        raise ValueError("Predictions file must contain a timestamp column.")
    predictions["timestamp"] = pd.to_datetime(
        predictions["timestamp"], utc=True, errors="raise"
    )
    predictions = predictions.set_index("timestamp").sort_index()
    return history, metrics, predictions


def check_outputs(metrics: pd.DataFrame, predictions: pd.DataFrame) -> None:
    required_metrics = {"model", "mae_mw", "rmse_mw", "nmae_pct"}
    required_predictions = {
        "actual_mw", "seasonal_naive_24h_mw", "gradient_boosting_mw",
        "selected_prediction_mw", "absolute_error_mw", "rolling_mae_mw",
        "drift_warning",
    }
    if metrics.empty or predictions.empty:
        raise ValueError("Metrics and predictions must be nonempty. Re-run evaluation.")
    if not required_metrics.issubset(metrics.columns):
        raise ValueError(f"Missing metrics columns: {required_metrics - set(metrics.columns)}")
    if not required_predictions.issubset(predictions.columns):
        raise ValueError(
            f"Missing prediction columns: {required_predictions - set(predictions.columns)}"
        )
    if predictions.index.has_duplicates or predictions.index.tz is None:
        raise ValueError("Prediction timestamps must be unique and UTC-aware.")
    numeric_columns = ["actual_mw", "seasonal_naive_24h_mw",
                       "gradient_boosting_mw", "selected_prediction_mw",
                       "absolute_error_mw", "rolling_mae_mw"]
    for column in numeric_columns:
        predictions[column] = pd.to_numeric(predictions[column], errors="raise")
    if not np.isfinite(predictions[["actual_mw", "selected_prediction_mw"]]).all().all():
        raise ValueError("Actuals and selected predictions must be finite.")
    if not predictions["drift_warning"].isin([True, False, "True", "False"]).all():
        raise ValueError("Unexpected drift_warning values.")
    predictions["drift_warning"] = predictions["drift_warning"].map(
        {True: True, False: False, "True": True, "False": False}
    ).astype(bool)


def slice_dates(frame: pd.DataFrame, start, end) -> pd.DataFrame:
    dates = frame.index.date
    return frame.loc[(dates >= start) & (dates <= end)]


def warning_episodes(frame: pd.DataFrame) -> pd.DataFrame:
    """Group adjacent flagged timestamps, splitting at missing hourly intervals."""
    flagged = frame.loc[frame["drift_warning"]].copy()
    columns = ["start_utc", "end_utc", "flagged_hours", "peak_rolling_mae_mw"]
    if flagged.empty:
        return pd.DataFrame(columns=columns)
    boundaries = flagged.index.to_series().diff().ne(pd.Timedelta(hours=1))
    group_ids = boundaries.cumsum()
    grouped = flagged.groupby(group_ids)
    episodes = pd.DataFrame({
        "start_utc": grouped.apply(lambda part: part.index[0]),
        "end_utc": grouped.apply(lambda part: part.index[-1]),
        "flagged_hours": grouped.size(),
        "peak_rolling_mae_mw": grouped["rolling_mae_mw"].max(),
    }).reset_index(drop=True)
    return episodes.sort_values("start_utc", ascending=False)


def line_plot(frame: pd.DataFrame, fields: dict[str, str],
              y_label: str = "Power (MW)") -> None:
    st.line_chart(frame[list(fields)].rename(columns=fields), y_label=y_label,
                  use_container_width=True)


def export_csv(frame: pd.DataFrame, filename: str) -> None:
    st.download_button(
        "Download selected data (CSV)", frame.to_csv(index=True).encode("utf-8"),
        file_name=filename, mime="text/csv",
    )


metrics_path = ARTIFACT_DIR / "metrics.csv"
predictions_path = ARTIFACT_DIR / "predictions.csv"
required_files = [CLEAN_CSV, ARTIFACT_DIR / "model.joblib",
                  ARTIFACT_DIR / "metadata.json", metrics_path, predictions_path]
missing = [str(path.relative_to(Path(__file__).resolve().parent))
           for path in required_files if not path.is_file()]
if missing:
    st.title("☀️ RenewCast")
    st.info("Forecast results are not available yet. Run the setup commands below first.")
    st.code("python scripts/download_data.py\npython scripts/train.py\n"
            "python scripts/evaluate.py\nstreamlit run app.py", language="bash")
    st.caption("Missing files: " + ", ".join(missing))
    st.stop()

try:
    _, metadata = load_artifacts()
    verify_data_integrity(CLEAN_CSV, metadata)
    stamps = tuple((path.stat().st_mtime_ns, path.stat().st_size)
                   for path in [CLEAN_CSV, metrics_path, predictions_path])
    history, metrics, predictions = read_outputs(
        str(CLEAN_CSV), str(metrics_path), str(predictions_path), stamps
    )
    check_outputs(metrics, predictions)
    if predictions.index[0].isoformat() != metadata["test_start"] or \
            predictions.index[-1].isoformat() != metadata["test_end"]:
        raise ValueError("Prediction dates do not match trained test-period metadata.")
    selected_model = metadata["selected_model"]
    if selected_model not in set(metrics["model"]):
        raise ValueError("Selected model missing from metrics.csv.")
except (OSError, ValueError, KeyError, TypeError, ImportError) as exc:
    st.error(f"Could not load forecast results: {exc}")
    st.info("Run `python scripts/train.py` and `python scripts/evaluate.py` again.")
    st.stop()

baseline = metrics.set_index("model").loc["seasonal_naive_24h"]
selected = metrics.set_index("model").loc[selected_model]
improvement = 100 * (1 - selected["mae_mw"] / baseline["mae_mw"])
threshold = metadata["drift_reference_mae_mw"] * metadata["drift_multiplier"]
all_episodes = warning_episodes(predictions)

st.sidebar.markdown("## ☀️ RenewCast")
st.sidebar.caption("German solar generation · UTC")
st.sidebar.divider()
st.sidebar.markdown("**Forecast horizon**")
st.sidebar.write("One hour ahead")
st.sidebar.markdown("**Test period**")
st.sidebar.caption(
    f"{predictions.index.min():%d %b %Y} — {predictions.index.max():%d %b %Y}"
)
st.sidebar.divider()
st.sidebar.caption("Historical data · Not a live forecast")

st.title("German Solar Generation Forecasts")
st.markdown(
    '<div class="intro">Forecast solar generation one hour ahead, compare predictions with actual output, '
    'and track forecast errors over time.</div>',
    unsafe_allow_html=True,
)

col1, col2, col3, col4 = st.columns(4)
col1.metric("Test MAE", f"{selected['mae_mw']:,.1f} MW")
col2.metric("MAE vs. previous day", f"{improvement:.1f}% lower" if improvement >= 0 else f"{abs(improvement):.1f}% higher")
col3.metric("Test hours", f"{len(predictions):,}")
col4.metric("Hours with high error", f"{int(predictions['drift_warning'].sum()):,}")

st.caption(
    "Metrics are calculated on the test period. Error flags mark hours that meet the rolling-error rule."
)

overview, explorer, evaluation, drift_tab = st.tabs([
    "Overview", "Forecasts", "Model comparison", "Error monitoring"
])

with overview:
    st.subheader("Solar generation history")
    history_days = st.slider("Latest historical days to display", 7, 120, 30,
                             key="history_days")
    line_plot(history.tail(history_days * 24), {"power_mw": "Observed generation"})
    st.subheader("Recent forecasts")
    recent_days = st.slider("Latest test days to display", 3, 45, 14,
                            key="overview_days")
    recent = predictions.tail(recent_days * 24)
    line_plot(recent, {
        "actual_mw": "Actual",
        "selected_prediction_mw": "Forecast",
    })
    st.info(
        "Forecasts use the previous hour’s observed generation to predict the next hour. This is a historical backtest, not a day-ahead forecast."
    )
    st.caption(f"Test period: {metadata['test_start']} through {metadata['test_end']}.")

with explorer:
    st.subheader("Forecasts")
    st.markdown('<p class="section-note">Choose a date range to compare forecasts with actual generation.'
                '</p>', unsafe_allow_html=True)
    earliest, latest = predictions.index.min().date(), predictions.index.max().date()
    default_start = max(earliest, latest - pd.Timedelta(days=13))
    picked = st.date_input("Choose dates (UTC)",
                           value=(default_start, latest),
                           min_value=earliest, max_value=latest,
                           key="forecast_dates")
    if not isinstance(picked, tuple) or len(picked) != 2:
        st.info("Select a start and end date.")
    else:
        start, end = picked
        if start > end:
            st.warning("Start date must be on or before end date.")
        else:
            subset = slice_dates(predictions, start, end)
            if subset.empty:
                st.info("No observations available for these dates.")
            else:
                options = {
                    "Gradient boosting": "gradient_boosting_mw",
                    "Previous day (24-hour lag)": "seasonal_naive_24h_mw",
                }
                chosen = st.multiselect("Display forecasts", list(options),
                                        default=list(options))
                graph_columns = {"actual_mw": "Actual"}
                graph_columns.update({options[label]: label for label in chosen})
                line_plot(subset, graph_columns)
                active_column = f"{selected_model}_mw"
                local_mae = (subset["actual_mw"] - subset[active_column]).abs().mean()
                daytime = subset.loc[subset["actual_mw"] > 0]
                daytime_mae = (
                    (daytime["actual_mw"] - daytime[active_column]).abs().mean()
                    if not daytime.empty else float("nan")
                )
                first, second, third = st.columns(3)
                first.metric("MAE for selected dates", f"{local_mae:,.1f} MW")
                second.metric("MAE when generation is above zero",
                              f"{daytime_mae:,.1f} MW" if pd.notna(daytime_mae) else "N/A")
                third.metric("Hours shown", f"{len(subset):,}")
                st.caption("Generation-positive hours use actual > 0 MW; this is a "
                           "post-hoc diagnostic, not a feature available at forecast time.")
                with st.expander("Largest forecast errors"):
                    worst = subset.assign(
                        error_mw=(subset["actual_mw"] - subset[active_column]).abs()
                    ).nlargest(10, "error_mw")
                    st.dataframe(worst[["actual_mw", active_column, "error_mw"]]
                                 .rename(columns={active_column: "forecast_mw"}),
                                 use_container_width=True)
                export_csv(subset, "renewcast_selected_forecasts.csv")

with evaluation:
    st.subheader("Model comparison on the test period")
    comparison = metrics.copy()
    comparison["model"] = comparison["model"].replace({
        "seasonal_naive_24h": "Previous day (24-hour lag)",
        "gradient_boosting": "Gradient boosting",
    })
    st.dataframe(comparison.rename(columns={
        "model": "Model", "mae_mw": "MAE (MW)",
        "rmse_mw": "RMSE (MW)", "nmae_pct": "nMAE (%)",
    }).style.format({"MAE (MW)": "{:,.2f}", "RMSE (MW)": "{:,.2f}",
                     "nMAE (%)": "{:.2f}"}), use_container_width=True,
                 hide_index=True)
    metric_choice = st.radio("Compare metric", ["mae_mw", "rmse_mw", "nmae_pct"],
                             format_func=lambda x: {
                                 "mae_mw": "MAE", "rmse_mw": "RMSE", "nmae_pct": "nMAE"
                             }[x], horizontal=True)
    chart = metrics.set_index("model")[[metric_choice]].rename(index={
        "seasonal_naive_24h": "Seasonal naive", "gradient_boosting": "Gradient boosting"
    })
    st.bar_chart(chart, y_label="Percent" if metric_choice == "nmae_pct" else "MW")
    st.caption("nMAE = 100 × test MAE / mean observed training-period generation. "
               "It is a normalized error, not a prediction accuracy percentage.")
    st.subheader("Forecast error distribution")
    residuals = predictions["actual_mw"] - predictions["selected_prediction_mw"]
    fig, ax = plt.subplots(figsize=(10, 3.3))
    fig.patch.set_facecolor("#111c2c")
    ax.set_facecolor("#111c2c")
    ax.hist(residuals, bins=48, color="#43cdbb", alpha=0.85)
    ax.axvline(0, color="#f7c66c", linestyle="--", linewidth=1.5)
    ax.set_xlabel("Actual − forecast (MW)", color="#e5eff5")
    ax.set_ylabel("Number of hours", color="#e5eff5")
    ax.tick_params(colors="#b6c7d6")
    for edge in ax.spines.values():
        edge.set_color("#35465a")
    fig.tight_layout()
    st.pyplot(fig, use_container_width=True)
    plt.close(fig)
    st.markdown("**About the models**")
    st.write(
        "The previous-day baseline uses generation from the same hour on the previous day. Gradient boosting uses recent generation and calendar information. Data is split by time, so later observations are not used to train the model."
    )
    st.caption(f"Training samples: {metadata['train_samples']:,} · "
               f"Validation: {metadata['validation_samples']:,} · "
               f"Test: {metadata['test_samples']:,}")

with drift_tab:
    st.subheader("Forecast error over time")
    st.write(
        f"**Rule:** a {metadata['drift_window_hours']}-hour rolling MAE must exceed "
        f"{metadata['drift_multiplier']}× validation MAE "
        f"(**{threshold:,.1f} MW**) for "
        f"{metadata['drift_consecutive_hours']} consecutive hours. "
        "A missing hour breaks the rolling window. This rule flags rising forecast errors; it does not measure changes in input data."
    )
    flagged_count = int(predictions["drift_warning"].sum())
    a, b, c = st.columns(3)
    a.metric("Hours flagged", f"{flagged_count:,}")
    b.metric("Periods flagged", f"{len(all_episodes):,}")
    c.metric("Share of test hours flagged", f"{flagged_count / len(predictions):.1%}")
    if flagged_count:
        st.warning("Forecast errors crossed the alert threshold during this test period.")
    else:
        st.success("Forecast errors did not cross the alert threshold during this test period.")
    graph = predictions[["rolling_mae_mw"]].copy()
    graph["Alert threshold"] = threshold
    line_plot(graph, {"rolling_mae_mw": "24h rolling MAE",
                      "Alert threshold": "Alert threshold"}, "Error (MW)")
    st.caption("An episode groups contiguous warning timestamps; the first warning "
               "occurs only after the required consecutive breaches. Episodes are "
               "descriptive summaries, not independent statistical change points.")
    st.subheader("Periods with high forecast error")
    if all_episodes.empty:
        st.info("No periods were flagged.")
    else:
        shown = st.slider("Episodes to display", 5,
                          max(5, min(100, len(all_episodes))),
                          min(15, max(5, min(100, len(all_episodes)))),
                          key="episode_count") if len(all_episodes) > 5 else len(all_episodes)
        st.dataframe(all_episodes.head(shown), use_container_width=True,
                     hide_index=True)
        export_csv(all_episodes, "renewcast_drift_episodes.csv")

st.divider()
st.caption(
    "Historical German solar data ending in 2020. This app does not provide live or weather-based forecasts. Predictions are one hour ahead, using the previous hour’s observed generation. Error alerts indicate elevated forecast error, not its cause."
)
