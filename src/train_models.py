"""
Train final Direct (24 specialists) LightGBM models on all available history.

- Daily origin timestamps are those where timestamp.hour == ORIGIN_HOUR (23:00).
- For each horizon h (1..24), target is y(t+h).
- Origin features are taken at time t.
- Target-time features (calendar + temp proxy) are shifted so they align to time t.

Inputs:
- data/processed/modeling_table.parquet

Outputs:
- models/model_h01_<cutoff>.pkl ... models/model_h24_<cutoff>.pkl
- models/metadata_<cutoff>.json

Run:
  python src/train_models.py
"""

from __future__ import annotations

import json
import pickle
from pathlib import Path
from datetime import datetime, timezone

import numpy as np
import pandas as pd
import yaml

try:
    import lightgbm as lgb
except ImportError as e:
    raise SystemExit("LightGBM not installed. Run: pip install lightgbm") from e

cfg = yaml.safe_load(Path("configs/default.yaml").read_text(encoding="utf-8"))


IN_PARQUET = Path("data/processed/modeling_table.parquet")
OUT_DIR = Path("models")

HORIZON_HOURS = cfg['backtest']['horizon_hours']
ORIGIN_HOUR = cfg['backtest']['origin_hour']

# These MUST match the backtest to stay consistent
ORIGIN_FEATURES = [
    "lag_1",
    "lag_24",
    "lag_168",
    "roll_mean_24h",
    "roll_mean_168h",
    "temp_obs_c",
    "temp_lag_24",
    "temp_lag_168",
]

TARGET_FEATURES = [
    "hour",
    "dow",
    "is_weekend",
    "month",
    "temp_fcst_proxy_7d_same_hour",
]

MODEL_PARAMS = cfg["model"]["lgbm_params"]

def make_horizon_frame(df: pd.DataFrame, h: int) -> pd.DataFrame:
    out = pd.DataFrame()
    out["origin_ts"] = df["timestamp"]
    out["y_target"] = df["y"].shift(-h)

    # Origin-time features
    for c in ORIGIN_FEATURES:
        out[c] = df[c]

    # Target-time features aligned to origin time
    for c in TARGET_FEATURES:
        out[f"target_{c}"] = df[c].shift(-h)

    # Only keep daily origins at ORIGIN_HOUR
    out = out[df["timestamp"].dt.hour == ORIGIN_HOUR].copy()

    # Drop rows with missing target/features
    feature_cols = ORIGIN_FEATURES + [f"target_{c}" for c in TARGET_FEATURES]
    out = out.dropna(subset=["y_target"] + feature_cols)

    return out


def main() -> None:
    if not IN_PARQUET.exists():
        raise SystemExit(f"Missing input: {IN_PARQUET}. Run build_features.py first.")

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    df = pd.read_parquet(IN_PARQUET).sort_values("timestamp").reset_index(drop=True)

    # Build horizon frames
    horizon_data = {h: make_horizon_frame(df, h) for h in range(1, HORIZON_HOURS + 1)}

    # Define a single global training cutoff origin timestamp:
    # Use the latest origin_ts that exists for horizon 24 (most restrictive).
    h24 = horizon_data[24]
    if h24.empty:
        raise SystemExit("Horizon-24 training frame is empty. Check feature/target alignment.")
    cutoff_origin_ts = pd.Timestamp(h24["origin_ts"].max())

    # Restrict all horizons to <= cutoff_origin_ts to ensure consistent training horizon availability
    for h in range(1, HORIZON_HOURS + 1):
        horizon_data[h] = horizon_data[h][horizon_data[h]["origin_ts"] <= cutoff_origin_ts].copy()

    cutoff_label = cutoff_origin_ts.strftime("%Y-%m-%dT%H%M")
    feature_cols = ORIGIN_FEATURES + [f"target_{c}" for c in TARGET_FEATURES]

    model_files = []
    train_rows = {}

    print(f"Training cutoff origin_ts: {cutoff_origin_ts} (label={cutoff_label})")
    print(f"Using origin hour: {ORIGIN_HOUR}:00")
    print(f"Features: {len(feature_cols)} columns")

    for h in range(1, HORIZON_HOURS + 1):
        hd = horizon_data[h]
        n = len(hd)
        train_rows[h] = n

        if n < 500:
            raise SystemExit(f"Not enough training rows for horizon {h}: {n}")

        X = hd[feature_cols]
        y = hd["y_target"].to_numpy()

        model = lgb.LGBMRegressor(**MODEL_PARAMS)
        model.fit(X, y)

        out_path = OUT_DIR / f"model_h{h:02d}_{cutoff_label}.pkl"
        with open(out_path, "wb") as f:
            pickle.dump(model, f)

        model_files.append(str(out_path))
        print(f"✅ Saved h={h:02d}  rows={n}  -> {out_path.name}")

    meta = {
        "trained_at_utc": datetime.now(timezone.utc).isoformat(),
        "origin_hour": ORIGIN_HOUR,
        "horizons": list(range(1, HORIZON_HOURS + 1)),
        "cutoff_origin_ts": str(cutoff_origin_ts),
        "cutoff_label": cutoff_label,
        "feature_cols": feature_cols,
        "origin_features": ORIGIN_FEATURES,
        "target_features_shifted": TARGET_FEATURES,
        "model_params": MODEL_PARAMS,
        "train_rows_by_horizon": train_rows,
        "model_files": model_files,
        "input_table": str(IN_PARQUET),
    }

    meta_path = OUT_DIR / f"metadata_{cutoff_label}.json"
    meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")

    print("\n✅ Training complete")
    print(f"Metadata: {meta_path}")


if __name__ == "__main__":
    main()
