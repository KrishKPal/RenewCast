import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from renewcast.config import CLEAN_CSV, RAW_CSV, SOURCE_URL  # noqa: E402
from renewcast.data import (  # noqa: E402
    clean_original_csv,
    download_original_csv,
    save_clean_csv,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Download and clean German hourly solar data")
    parser.add_argument("--url", default=SOURCE_URL, help="Original OPSD hourly singleindex CSV URL")
    parser.add_argument("--raw-csv", type=Path, help="Use an already downloaded original OPSD CSV")
    parser.add_argument("--output", type=Path, default=CLEAN_CSV)
    args = parser.parse_args()

    try:
        source = args.raw_csv if args.raw_csv is not None else RAW_CSV
        if args.raw_csv is None:
            download_original_csv(args.url, source)
        cleaned = clean_original_csv(source)
        save_clean_csv(cleaned, args.output)
    except (RuntimeError, ValueError, FileNotFoundError, OSError) as exc:
        parser.exit(1, f"Data preparation failed: {exc}\n")
    print(f"Saved {len(cleaned):,} hourly timestamps to {args.output}")
    print(f"Observed: {cleaned['power_mw'].notna().sum():,}; missing: {cleaned['power_mw'].isna().sum():,}")
    print(f"Coverage (UTC): {cleaned.index.min()} to {cleaned.index.max()}")


if __name__ == "__main__":
    main()
