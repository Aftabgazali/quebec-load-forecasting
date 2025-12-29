# create calendar + lag features + join weather → write data/processed/modeling_table.parquet
"""
Build modeling features from demand + weather.

Inputs:
- data/processed/demand_hourly.parquet
- data/processed/weather_hourly.parquet

Output:
- data/processed/modeling_table.parquet

Notes:
- demand_mw is the true target (can be NaN for missing hours).
- demand_mw_filled is used ONLY to compute lags/rollings smoothly.
- temp_fcst_proxy_7d_same_hour uses ONLY past temperatures (t-24..t-168), so no leakage.
"""

from __future__ import annotations

from pathlib import Path
import pandas as pd
import yaml 
import numpy as np

cfg = yaml.safe_load(Path("configs/default.yaml").read_text(encoding="utf-8"))

DEMAND_IN = Path("data/processed/demand_hourly.parquet")
WEATHER_IN = Path("data/processed/weather_hourly.parquet")
OUT = Path("data/processed/modeling_table.parquet")

TZ = cfg['project']['timezone']
LAGS = [1, 24, 168]
ROLLING_WINDOWS = [24, 168]

def main() -> None:
    if not DEMAND_IN.exists():
        raise SystemExit(f"Missing: {DEMAND_IN}")
    if not WEATHER_IN.exists():
        raise SystemExit(f"Missing: {WEATHER_IN} (run ingest_weather.py first)")

    demand = pd.read_parquet(DEMAND_IN)
    weather = pd.read_parquet(WEATHER_IN)

    # Parse + enforce TZ
    for df in (demand, weather):
        df["timestamp"] = pd.to_datetime(df["timestamp"])
        if df["timestamp"].dt.tz is None:
            df["timestamp"] = df["timestamp"].dt.tz_localize(TZ)
        else:
            df["timestamp"] = df["timestamp"].dt.tz_convert(TZ)

    demand = demand.sort_values("timestamp").reset_index(drop=True)
    weather = weather.sort_values("timestamp").reset_index(drop=True)

    # Join (left join keeps demand as the backbone)
    df = demand.merge(weather[["timestamp", "temp_obs_c"]], on="timestamp", how="left")

    # Target (truth)
    df["y"] = df["demand_mw"]

    # Fill missing demand ONLY for feature construction
    df["is_imputed_demand"] = df["demand_mw"].isna()
    filled = df["demand_mw"].copy()
    filled = filled.fillna(filled.shift(168))
    filled = filled.fillna(filled.shift(24))
    filled = filled.ffill().bfill()
    df["demand_mw_filled"] = filled

    # Calendar features
    df["hour"] = df["timestamp"].dt.hour
    df["dow"] = df["timestamp"].dt.dayofweek  # Mon=0 ... Sun=6
    df["is_weekend"] = (df["dow"] >= 5).astype(int)
    df["month"] = df["timestamp"].dt.month

    # Demand lag features
    for lag in LAGS:
        df[f"lag_{lag}"] = df["demand_mw_filled"].shift(lag)

    # Demand rolling mean features (past-only)
    for w in ROLLING_WINDOWS:
        df[f"roll_mean_{w}h"] = df["demand_mw_filled"].shift(1).rolling(window=w, min_periods=w).mean()

    # === Weather features ===
    # 7-day same-hour average using ONLY past temps: t-24..t-168 (7 values)
    temp = df["temp_obs_c"]
    past_same_hour = [temp.shift(24 * k) for k in range(1, 8)]
    df["temp_fcst_proxy_7d_same_hour"] = np.nanmean(np.vstack([s.to_numpy() for s in past_same_hour]), axis=0)

    # Optional: include a couple of temp lags too (still past-only, cheap signal)
    df["temp_lag_24"] = temp.shift(24)
    df["temp_lag_168"] = temp.shift(168)

    keep_cols = [
        "timestamp",
        "y",
        "is_imputed_demand",
        "temp_obs_c",
        "temp_fcst_proxy_7d_same_hour",
        "temp_lag_24",
        "temp_lag_168",
        "hour",
        "dow",
        "is_weekend",
        "month",
    ] + [f"lag_{l}" for l in LAGS] + [f"roll_mean_{w}h" for w in ROLLING_WINDOWS]

    out = df[keep_cols].copy()
    out.to_parquet(OUT, index=False)

    missing_temp = int(out["temp_obs_c"].isna().sum())
    print("✅ Features rebuilt (with weather)")
    print(f"  Rows total:                 {len(out)}")
    print(f"  Missing target (y):         {int(out['y'].isna().sum())}")
    print(f"  Missing temp_obs_c:         {missing_temp}")
    print(f"  Output:                     {OUT}")


if __name__ == "__main__":
    main()


#$ python src/build_features.py
# ✅ Features rebuilt (with weather)
# Rows total:                 43824
# Missing target (y):         51
# Missing temp_obs_c:         5
# Output:                     data\processed\modeling_table.parquet
# Why is it exactly 168 and not bigger (like 24 too)?
# Because your summary counts rows missing any feature. The early rows are already missing 
# the “week” features, so they’re flagged anyway. The “day” features (lag 24, roll 24) also cause missingness early on, but they’re “inside” that first-week block — so the total stays dominated by the week requirement and shows up as 168.
# So what does this mean for training? It means your effective training start is roughly:
# 2019-01-08 00:00 (one week after 2019-01-01 00:00) And that’s completely fine — you still have 43,605 usable rows, which is plenty.