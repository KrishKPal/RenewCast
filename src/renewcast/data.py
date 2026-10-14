from pathlib import Path
import urllib.error
import urllib.request

import numpy as np
import pandas as pd

from .config import SOURCE_COLUMN


def download_original_csv(url: str, destination: Path) -> None:
    """Stream the original CSV to disk; never retain a partial download."""
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".part")
    try:
        with urllib.request.urlopen(url, timeout=90) as response, temporary.open("wb") as out:
            while chunk := response.read(1024 * 1024):
                out.write(chunk)
        temporary.replace(destination)
    except (urllib.error.URLError, OSError, TimeoutError) as exc:
        temporary.unlink(missing_ok=True)
        raise RuntimeError(
            f"Download failed from {url}: {exc}. Download the original OPSD "
            "time_series_60min_singleindex.csv manually and run "
            "'python scripts/download_data.py --raw-csv /path/to/file.csv'."
        ) from exc


def _validate_hourly_values(frame: pd.DataFrame) -> pd.DataFrame:
    if frame.empty:
        raise ValueError("The input CSV contains no rows.")

    timestamps = pd.to_datetime(frame["timestamp"], utc=True, errors="coerce")
    if timestamps.isna().any():
        raise ValueError("Invalid or missing UTC timestamps in input CSV.")
    if timestamps.duplicated().any():
        raise ValueError("Duplicate UTC timestamps in input CSV.")
    if ((timestamps.dt.minute != 0) | (timestamps.dt.second != 0)
            | (timestamps.dt.microsecond != 0)).any():
        raise ValueError("Expected timestamps aligned to whole UTC hours.")

    raw_values = frame["power_mw"]
    values = pd.to_numeric(raw_values, errors="coerce")
    invalid = raw_values.notna() & raw_values.astype("string").str.strip().ne("") & values.isna()
    if invalid.any():
        raise ValueError("Non-numeric generation values in input CSV.")
    if np.isinf(values.to_numpy(dtype=float)).any():
        raise ValueError("Infinite generation values in input CSV.")
    if (values.dropna() < 0).any():
        raise ValueError("Negative generation values in input CSV.")

    series = pd.Series(values.to_numpy(dtype=float), index=pd.DatetimeIndex(timestamps), name="power_mw")
    series = series.sort_index().dropna().asfreq("h") if series.notna().any() else series.iloc[:0]
    if series.empty:
        raise ValueError("No observed solar generation values were found.")
    # Rebuild on a complete hourly grid without filling missing observations.
    # Missing targets and any feature windows touching them are skipped in training.
    result = series.to_frame()
    result.index.name = "timestamp"
    return result


def clean_original_csv(path: Path) -> pd.DataFrame:
    if not path.is_file():
        raise FileNotFoundError(f"Original CSV not found: {path}")
    try:
        frame = pd.read_csv(path, usecols=["utc_timestamp", SOURCE_COLUMN])
    except ValueError as exc:
        raise ValueError(
            f"Expected columns 'utc_timestamp' and '{SOURCE_COLUMN}' in {path}. "
            "Use the OPSD 60-minute singleindex CSV."
        ) from exc
    return _validate_hourly_values(frame.rename(columns={
        "utc_timestamp": "timestamp", SOURCE_COLUMN: "power_mw"
    }))


def save_clean_csv(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index_label="timestamp", date_format="%Y-%m-%dT%H:%M:%SZ")


def load_clean_csv(path: Path) -> pd.DataFrame:
    if not path.is_file():
        raise FileNotFoundError(
            f"Clean CSV not found: {path}. Run 'python scripts/download_data.py' first."
        )
    try:
        frame = pd.read_csv(path, usecols=["timestamp", "power_mw"])
    except ValueError as exc:
        raise ValueError(f"Clean CSV must have timestamp and power_mw columns: {path}") from exc
    return _validate_hourly_values(frame)
