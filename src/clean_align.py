"""
Clean + align Hydro-Québec hourly demand data into a canonical hourly time series.

Input (raw):
- data/raw/hq_demand/historique-demande-electricite-quebec.csv
  columns: date;moyenne_mw
  example: 2023-01-01T03:00:00-05:00;20010.46

Output (processed):
- data/processed/demand_hourly.parquet
  columns: timestamp, demand_mw
  timestamp is canonical START-of-hour in America/Toronto timezone.

Run:
  python src/clean_align.py
"""

from __future__ import annotations

from pathlib import Path
import pandas as pd
import yaml 

cfg = yaml.safe_load(Path("configs/default.yaml").read_text(encoding="utf-8"))



RAW_CSV = Path("data/raw/hq_demand/historique-demande-electricite-quebec.csv")
OUT_DIR = Path("data/processed")
OUT_PARQUET = OUT_DIR / "demand_hourly.parquet"

TZ = cfg['project']['timezone']
END_TO_START_SHIFT_HOURS = cfg['hq']['end_to_start_shift_hours']  # HQ labels end-of-hour -> convert to start-of-hour


def main() -> None:
    if not RAW_CSV.exists():
        raise SystemExit(f"Missing raw file: {RAW_CSV}")

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(RAW_CSV, sep=";", usecols=["date", "moyenne_mw"])
    raw_rows = len(df)

    # ✅ Key fix: parse in UTC to handle mixed offsets (-04:00 / -05:00), then convert
    df["date"] = pd.to_datetime(df["date"], utc=True).dt.tz_convert(TZ)

    # Convert HQ end-of-hour label -> canonical start-of-hour
    df["timestamp"] = df["date"] - pd.Timedelta(hours=END_TO_START_SHIFT_HOURS)

    df = df.rename(columns={"moyenne_mw": "demand_mw"})[["timestamp", "demand_mw"]]

    # Sort (raw export may be unsorted)
    df = df.sort_values("timestamp")

    # Handle duplicates (can happen in some edge cases)
    dup_count = int(df.duplicated(subset=["timestamp"]).sum())
    if dup_count > 0:
        df = df.groupby("timestamp", as_index=False)["demand_mw"].mean()

    # Complete hourly index (missing hours become NaN explicitly)
    start = df["timestamp"].min()
    end = df["timestamp"].max()
    full_index = pd.date_range(start=start, end=end, freq="h", tz=TZ)

    df = df.set_index("timestamp").reindex(full_index)
    df.index.name = "timestamp"
    df = df.reset_index()

    missing_hours = int(df["demand_mw"].isna().sum())

    df.to_parquet(OUT_PARQUET, index=False)

    print("✅ Clean + align complete")
    print(f"  Raw rows read:         {raw_rows}")
    print(f"  Duplicate timestamps:  {dup_count} (resolved by averaging)")
    print(f"  Missing hours:         {missing_hours} (left as NaN)")
    print(f"  Output:                {OUT_PARQUET}")
    print(f"  Range:                 {df['timestamp'].min()}  →  {df['timestamp'].max()}")


if __name__ == "__main__":
    main()