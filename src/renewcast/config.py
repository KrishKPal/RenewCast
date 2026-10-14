from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
RAW_CSV = ROOT / "data" / "raw" / "time_series_60min_singleindex.csv"
CLEAN_CSV = ROOT / "data" / "processed" / "germany_solar_hourly.csv"
ARTIFACT_DIR = ROOT / "artifacts"

SOURCE_URL = (
    "https://data.open-power-system-data.org/time_series/2020-10-06/"
    "time_series_60min_singleindex.csv"
)
SOURCE_COLUMN = "DE_solar_generation_actual"

FORECAST_HOURS = 1
SEASONAL_LAG_HOURS = 24
LONGEST_LOOKBACK_HOURS = 168
TRAIN_FRACTION = 0.70
VALIDATION_FRACTION = 0.15
MIN_SAMPLES_PER_SPLIT = 24

DRIFT_WINDOW_HOURS = 24
DRIFT_MULTIPLIER = 1.5
DRIFT_CONSECUTIVE_HOURS = 3
