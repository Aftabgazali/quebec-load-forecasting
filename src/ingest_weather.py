# fetch weather data (historic for backtests; later forecast-vintage for live) into data/raw/weather_*

"""
Download hourly observed temperature (2m) for a single location using Open-Meteo Historical Weather API.

Inputs:
- data/processed/demand_hourly.parquet (used only to determine date range)

Outputs:
- data/raw/weather_historic/open_meteo/weather_hourly_<start>_<end>.csv
- data/processed/weather_hourly.parquet   (columns: timestamp, temp_obs_c)

Run:
  python src/ingest_weather.py
"""

from __future__ import annotations

import time
from pathlib import Path
from datetime import timedelta

import pandas as pd
import requests
import yaml

cfg = yaml.safe_load(Path("configs/default.yaml").read_text(encoding="utf-8"))

TZ_LOCAL = cfg['project']['timezone']

# Default location: Montréal (can change later)
LAT = cfg['weather']['lat']
LON = cfg['weather']['lon']

API_URL = "https://archive-api.open-meteo.com/v1/archive"
MAX_TRIES = 3
SLEEP_SECONDS = 10

DEMAND_PARQUET = Path("data/processed/demand_hourly.parquet")
RAW_DIR = Path("data/raw/weather_historic/open_meteo")
PROCESSED_DIR = Path("data/processed")
PROCESSED_OUT = PROCESSED_DIR / "weather_hourly.parquet"


def fetch_chunk(start_date: str, end_date: str) -> pd.DataFrame:
    params = {
        "latitude": LAT,
        "longitude": LON,
        "start_date": start_date,
        "end_date": end_date,
        "hourly": cfg['weather']['variable'],
        "timezone": cfg['weather']['global_timezone'],
        "temperature_unit": "celsius",
    }
    
    last_err = None
    for attempt in range(1, MAX_TRIES + 1):
        try:
            r = requests.get(API_URL, params=params, timeout=60)
            r.raise_for_status()
            data = r.json()

            times = data["hourly"]["time"]
            temps = data["hourly"]["temperature_2m"]

            # Parse as UTC, then convert to local TZ
            ts_utc = pd.to_datetime(times, utc=True)
            ts_local = ts_utc.tz_convert(TZ_LOCAL)

            return pd.DataFrame({"timestamp": ts_local, "temp_obs_c": temps})

        except Exception as e:
            last_err = e
            print(f"[Attempt {attempt}/{MAX_TRIES}] Failed {start_date}..{end_date}: {e}")
            if attempt < MAX_TRIES:
                time.sleep(SLEEP_SECONDS)

    raise SystemExit(f"Failed after {MAX_TRIES} attempts for {start_date}..{end_date}: {last_err}")


def main() -> None:
    if not DEMAND_PARQUET.exists():
        raise SystemExit(f"Missing demand file: {DEMAND_PARQUET}. Run clean_align.py first.")

    RAW_DIR.mkdir(parents=True, exist_ok=True)
    Path("data/processed").mkdir(parents=True, exist_ok=True)

    demand = pd.read_parquet(DEMAND_PARQUET)
    ts = pd.to_datetime(demand["timestamp"])
    if ts.dt.tz is None:
        ts = ts.dt.tz_localize(TZ_LOCAL)
    else:
        ts = ts.dt.tz_convert(TZ_LOCAL)

    start = ts.min().date()
    end = ts.max().date()

    print(f"Demand range: {ts.min()} → {ts.max()}")
    print(f"Weather fetch: {start} → {end}  (LAT={LAT}, LON={LON}, fetch_tz=UTC, convert_to={TZ_LOCAL})")

    # Chunk by year
    all_parts: list[pd.DataFrame] = []
    year = start.year
    while year <= end.year:
        chunk_start = max(start, pd.Timestamp(f"{year}-01-01").date())
        chunk_end = min(end, pd.Timestamp(f"{year}-12-31").date())
        print(f"Fetching year chunk: {chunk_start} → {chunk_end}")
        all_parts.append(fetch_chunk(str(chunk_start), str(chunk_end)))
        year += 1

    weather = pd.concat(all_parts, ignore_index=True).sort_values("timestamp")

    raw_csv = RAW_DIR / f"weather_hourly_{start}_{end}.csv"
    weather.to_csv(raw_csv, index=False)
    weather.to_parquet(PROCESSED_OUT, index=False)

    print("✅ Weather ingest complete")
    print(f"  Raw:       {raw_csv}")
    print(f"  Processed: {PROCESSED_OUT}")
    print(f"  Rows:      {len(weather)}")
    print(f"  Range:     {weather['timestamp'].min()} → {weather['timestamp'].max()}")


if __name__ == "__main__":
    main()