"""
Generate a next-24h forecast run using the latest trained 24 specialist models.

Inputs:
- data/processed/modeling_table.parquet
- models/metadata_*.json + referenced model_hXX_*.pkl files

Outputs:
- data/forecasts/runs/<run_id>/
    inputs.json
    yhat.parquet
    weather_used.parquet
- reports/runs/<run_id>/summary.md

Run:
  python src/predict_next_day.py
"""

from __future__ import annotations

import json
import pickle
from pathlib import Path
from datetime import datetime
import yaml

import pandas as pd

MODEL_DIR = Path("models")
TABLE_PATH = Path("data/processed/modeling_table.parquet")

RUNS_DIR = Path("data/forecasts/runs")
REPORTS_DIR = Path("reports/runs")

cfg = yaml.safe_load(Path("configs/default.yaml").read_text(encoding="utf-8"))


TZ = cfg['project']['timezone']


def latest_metadata_file() -> Path:
    metas = sorted(MODEL_DIR.glob("metadata_*.json"))
    if not metas:
        raise SystemExit("No metadata_*.json found in models/. Run train_models.py first.")
    return metas[-1]


def make_run_id(issue_time: pd.Timestamp) -> str:
    # Example: 2023-12-31T2300_America-Toronto
    return issue_time.strftime("%Y-%m-%dT%H%M") + "_America-Toronto"


def main() -> None:
    if not TABLE_PATH.exists():
        raise SystemExit(f"Missing: {TABLE_PATH}. Run build_features.py first.")

    meta_path = latest_metadata_file()
    meta = json.loads(meta_path.read_text(encoding="utf-8"))

    origin_hour = int(meta["origin_hour"])
    feature_cols = meta["feature_cols"]
    model_files = meta["model_files"]

    # Load models
    models = {}
    for mf in model_files:
        p = Path(mf)
        if not p.exists():
            raise SystemExit(f"Model file missing: {p}")
        # extract horizon from filename model_hXX_...
        name = p.name
        h = int(name.split("_")[1].replace("h", ""))  # "h01"
        with open(p, "rb") as f:
            models[h] = pickle.load(f)

    # Load table
    df = pd.read_parquet(TABLE_PATH).sort_values("timestamp").reset_index(drop=True)
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    if df["timestamp"].dt.tz is None:
        df["timestamp"] = df["timestamp"].dt.tz_localize(TZ)
    else:
        df["timestamp"] = df["timestamp"].dt.tz_convert(TZ)

    # Choose latest available origin timestamp at origin_hour
    origins = df[df["timestamp"].dt.hour == origin_hour]["timestamp"].sort_values()
    if origins.empty:
        raise SystemExit(f"No rows found at origin_hour={origin_hour}:00 in modeling table.")
    last_ts = df["timestamp"].max()

    # Need a full 24h ahead window
    valid_origins = origins[origins + pd.Timedelta(hours=24) <= last_ts]
    if valid_origins.empty:
        raise SystemExit("No valid origin has a full 24h ahead window in the dataset.")

    issue_time = pd.Timestamp(valid_origins.max())


    # Create run folder
    run_id = make_run_id(issue_time)
    run_dir = RUNS_DIR / run_id
    run_dir.mkdir(parents=True, exist_ok=False)  # fail if already exists

    # Build single-row feature vector at issue_time for each horizon:
    # We trained horizon models on rows where target-time features were already shifted into columns
    # (e.g., target_hour, target_temp_fcst_proxy_7d_same_hour).
    # Our modeling_table already contains those "base" columns unshifted, so we must replicate
    # the SAME shifting logic used in training/backtest for this single origin.
    #
    # Easiest approach: rebuild the horizon frames on-the-fly for the last origin only.

    # Find row index of issue_time in df
    idx = df.index[df["timestamp"] == issue_time]
    if len(idx) != 1:
        raise SystemExit("Could not uniquely locate issue_time row in table.")
    idx = int(idx[0])

    # Helper: get value at df[col] for row idx + offset (future)
    def at_offset(col: str, offset_hours: int):
        j = idx + offset_hours
        if j < 0 or j >= len(df):
            return None
        return df.loc[j, col]

    yhat_rows = []
    weather_rows = []

    # We'll also store the weather proxy value we used for each target hour
    for h in range(1, 25):
        model = models.get(h)
        if model is None:
            raise SystemExit(f"Missing model for horizon {h}")

        # Build feature dict matching meta["feature_cols"]
        feat = {}

        # Origin-time features: use the issue_time row directly
        # These are already in df as columns.
        for col in meta["origin_features"]:
            feat[col] = df.loc[idx, col]

        # Target-time features: need values at issue_time + h
        for col in meta["target_features_shifted"]:
            feat[f"target_{col}"] = at_offset(col, h)

        # Check missing
        if any(pd.isna(v) for v in feat.values()):
            raise SystemExit(
                f"Feature missing for horizon {h} at issue_time {issue_time}. "
                "This usually means you are too close to dataset end to have full 24h ahead."
            )

        X = pd.DataFrame([feat], columns=feature_cols)
        y_pred = float(model.predict(X)[0])

        target_ts = issue_time + pd.Timedelta(hours=h)

        yhat_rows.append(
            {
                "origin_ts": issue_time,
                "target_ts": target_ts,
                "horizon": h,
                "yhat": y_pred,
            }
        )

        # store the proxy weather used for that target time
        weather_rows.append(
            {
                "origin_ts": issue_time,
                "target_ts": target_ts,
                "temp_fcst_proxy_7d_same_hour": feat["target_temp_fcst_proxy_7d_same_hour"],
                "temp_obs_c_at_issue": feat["temp_obs_c"],
            }
        )

    yhat_df = pd.DataFrame(yhat_rows)
    weather_df = pd.DataFrame(weather_rows)

    # Save run artifacts
    yhat_path = run_dir / "yhat.parquet"
    weather_path = run_dir / "weather_used.parquet"
    inputs_path = run_dir / "inputs.json"

    yhat_df.to_parquet(yhat_path, index=False)
    weather_df.to_parquet(weather_path, index=False)

    inputs = {
        "run_id": run_id,
        "issue_time": str(issue_time),
        "origin_hour": origin_hour,
        "model_metadata": str(meta_path),
        "cutoff_origin_ts": meta.get("cutoff_origin_ts"),
        "feature_cols": feature_cols,
        "generated_at_local": str(pd.Timestamp.now(tz=TZ)),
    }
    inputs_path.write_text(json.dumps(inputs, indent=2), encoding="utf-8")

    # Write a tiny report
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    report_dir = REPORTS_DIR / run_id
    report_dir.mkdir(parents=True, exist_ok=True)
    report_path = report_dir / "summary.md"

    lines = []
    lines.append("# Next-day forecast run")
    lines.append("")
    lines.append(f"- Run id: `{run_id}`")
    lines.append(f"- Issue time: `{issue_time}`")
    lines.append(f"- Origin hour: `{origin_hour}:00`")
    lines.append(f"- Model metadata: `{meta_path}`")
    lines.append("")
    lines.append("## Forecast (first 10 rows)")
    lines.append("")
    lines.append(yhat_df.head(10).to_markdown(index=False))
    lines.append("")
    report_path.write_text("\n".join(lines), encoding="utf-8")

    print("✅ Forecast run created")
    print(f"  Run folder: {run_dir}")
    print(f"  Report:     {report_path}")
    print(f"  Issue time: {issue_time}")
    print(f"  Targets:    {yhat_df['target_ts'].min()} → {yhat_df['target_ts'].max()}")


if __name__ == "__main__":
    main()