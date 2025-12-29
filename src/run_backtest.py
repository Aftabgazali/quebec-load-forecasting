"""
Walk-forward backtest for Direct forecasting (24 specialists) using LightGBM.

We define daily forecast origins at 23:00 (end of day D) and predict next 24 hours (D+1 00:00..23:00).

Input:
- data/processed/modeling_table.parquet

Outputs:
- reports/backtest/backtest_summary.md
- reports/backtest/metrics_by_horizon.csv

Run:
  python src/run_backtest.py
"""

from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd
import yaml

try:
    import lightgbm as lgb
except ImportError as e:
    raise SystemExit("LightGBM not installed. Run: pip install lightgbm") from e

cfg = yaml.safe_load(Path("configs/default.yaml").read_text(encoding="utf-8"))

IN_PARQUET = Path("data/processed/modeling_table.parquet")
OUT_DIR = Path("reports/backtest")
OUT_MD = OUT_DIR / "backtest_summary.md"
OUT_CSV = OUT_DIR / "metrics_by_horizon.csv"
OUT_PEAK_CSV = OUT_DIR / "daily_peak_diagnostics.csv"

# Keep this small first (smoke test). Change to 90 after it runs once.
EVAL_DAYS = cfg['backtest']['eval_days']

HORIZON_HOURS = cfg['backtest']['horizon_hours']
ORIGIN_HOUR = cfg['backtest']['origin_hour'] # daily forecast origin time = 23:00

# IMPORTANT: We will NOT use future observed temperature.
# We use:
# - past-known weather: temp_obs_c (at origin), temp_lag_24, temp_lag_168
# - "forecast proxy" for target hour: temp_fcst_proxy_7d_same_hour (shifted to target time)
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


def mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.mean(np.abs(y_true - y_pred)))


def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def make_horizon_frame(df: pd.DataFrame, h: int) -> pd.DataFrame:
    """
    Build a horizon-specific dataset where each row corresponds to a DAILY ORIGIN time t (23:00),
    target is y(t+h), and target-time features are shifted so they align with origin time.
    """
    out = pd.DataFrame()
    out["origin_ts"] = df["timestamp"]
    out["y_target"] = df["y"].shift(-h)

    # Origin-time features (computed at origin hour)
    for c in ORIGIN_FEATURES:
        out[c] = df[c]

    # Target-time features (calendar + proxy) aligned back to origin
    for c in TARGET_FEATURES:
        out[f"target_{c}"] = df[c].shift(-h)

    # Filter: only origins at the chosen daily origin hour
    out = out[df["timestamp"].dt.hour == ORIGIN_HOUR].copy()

    # Drop rows where target is missing or any feature is missing
    feature_cols = ORIGIN_FEATURES + [f"target_{c}" for c in TARGET_FEATURES]
    out = out.dropna(subset=["y_target"] + feature_cols)

    return out


def main() -> None:
    if not IN_PARQUET.exists():
        raise SystemExit(f"Missing input: {IN_PARQUET}. Run build_features.py first.")

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    df = pd.read_parquet(IN_PARQUET).sort_values("timestamp").reset_index(drop=True)

    # Build horizon datasets
    horizon_data = {h: make_horizon_frame(df, h) for h in range(1, HORIZON_HOURS + 1)}

    # Use horizon 1 to define the origin timeline (they should all align)
    origins = horizon_data[1]["origin_ts"].sort_values().unique()
    if len(origins) <= EVAL_DAYS + 10:
        raise SystemExit(f"Not enough daily origins for EVAL_DAYS={EVAL_DAYS}.")

    eval_origins = origins[-EVAL_DAYS:]

    preds_records = []

    # LightGBM model
    base_model = cfg["model"]["lgbm_params"]

    for fold_i, origin_ts in enumerate(eval_origins, start=1):
        # For each fold (one day), train models using all origins strictly before origin_ts
        for h in range(1, HORIZON_HOURS + 1):
            hd = horizon_data[h]

            train = hd[hd["origin_ts"] < origin_ts]
            test = hd[hd["origin_ts"] == origin_ts]

            if len(test) != 1:
                # Some horizons can drop due to missing targets/features; just skip if not present
                continue
            if len(train) < 100:
                # Too little training data -> skip
                continue

            X_cols = ORIGIN_FEATURES + [f"target_{c}" for c in TARGET_FEATURES]
            X_train = train[X_cols]
            y_train = train["y_target"].to_numpy()

            X_test = test[X_cols]
            y_true = float(test["y_target"].iloc[0])

            model = lgb.LGBMRegressor(**base_model)
            model.fit(X_train, y_train)

            y_pred = float(model.predict(X_test)[0])

            target_ts = pd.Timestamp(origin_ts) + pd.Timedelta(hours=h)

            preds_records.append(
                {
                    "origin_ts": origin_ts,
                    "target_ts": target_ts,
                    "horizon": h,
                    "y_true": y_true,
                    "y_pred": y_pred,
                    "abs_err": abs(y_true - y_pred),
                    "sq_err": (y_true - y_pred) ** 2,
                    "err": (y_pred - y_true),
                }
            )

        if fold_i % 5 == 0:
            print(f"Fold {fold_i}/{len(eval_origins)} done...")

    pred_df = pd.DataFrame(preds_records)
    if pred_df.empty:
        raise SystemExit("No predictions were produced. Something went wrong with data filtering.")
    # Daily peak metrics (ops-focused; computed per origin day)
    daily = pred_df.copy()
    daily["date"] = pd.to_datetime(daily["target_ts"]).dt.date

    daily_peak = (
        daily.groupby("date")
        .apply(
            lambda g: pd.Series(
                {
                    "peak_error_mw": float(g["y_pred"].max() - g["y_true"].max()),
                    "peak_timing_error_h": float(
                        (
                            g.loc[g["y_pred"].idxmax(), "target_ts"]
                            - g.loc[g["y_true"].idxmax(), "target_ts"]
                        )
                        / pd.Timedelta(hours=1)
                    ),
                }
            )
        )
        .reset_index()
    )

    peak_error_mean = float(daily_peak["peak_error_mw"].mean()) if not daily_peak.empty else np.nan
    peak_timing_mean = float(daily_peak["peak_timing_error_h"].mean()) if not daily_peak.empty else np.nan

    # Overall metrics (computed from sums: correct + stable)
    abs_err_sum = float(pred_df["abs_err"].sum())
    sq_err_sum = float(pred_df["sq_err"].sum())
    err_sum = float(pred_df["err"].sum())
    y_sum = float(pred_df["y_true"].sum())
    n_points = int(len(pred_df))

    overall_mae = abs_err_sum / n_points
    overall_rmse = float(np.sqrt(sq_err_sum / n_points))
    overall_wape = (abs_err_sum / y_sum) if y_sum != 0 else np.nan
    overall_bias = err_sum / n_points

    # By-horizon metrics (from sums, not mean-of-means)
    by_h = (
        pred_df.groupby("horizon")
        .agg(
            n=("y_true", "size"),
            abs_err_sum=("abs_err", "sum"),
            sq_err_sum=("sq_err", "sum"),
            err_sum=("err", "sum"),
            y_sum=("y_true", "sum"),
        )
        .reset_index()
        .sort_values("horizon")
    )

    by_h["mae"] = by_h["abs_err_sum"] / by_h["n"]
    by_h["rmse"] = np.sqrt(by_h["sq_err_sum"] / by_h["n"])
    by_h["wape"] = by_h["abs_err_sum"] / by_h["y_sum"]
    by_h["bias"] = by_h["err_sum"] / by_h["n"]

    # Keep the table clean
    by_h = by_h[["horizon", "n", "mae", "rmse", "wape", "bias"]]
    by_h.to_csv(OUT_CSV, index=False)
    
    # Save the daily peak into csv so the pages could read them
    daily_peak.to_csv(OUT_PEAK_CSV, index=False)
    # Write markdown summary
    lines = []
    lines.append("# Model backtest summary (LightGBM, Direct 24 specialists)")
    lines.append("")
    lines.append(f"- Daily origin hour: {ORIGIN_HOUR}:00")
    lines.append(f"- Evaluation days: {EVAL_DAYS}")
    lines.append(f"- Scored points: {len(pred_df)}")
    lines.append("")
    lines.append("## Overall")
    lines.append(f"- MAE:   {overall_mae:.3f} MW")
    lines.append(f"- RMSE:  {overall_rmse:.3f} MW")
    lines.append(f"- WAPE:  {overall_wape:.5f} (fraction of total load)")
    lines.append(f"- Bias:  {overall_bias:.3f} MW (mean y_pred - y_true)")
    lines.append(f"- Peak error (mean): {peak_error_mean:.3f} MW")
    lines.append(f"- Peak timing error (mean): {peak_timing_mean:.3f} hours")
    lines.append("")
    lines.append("## Daily peak diagnostics (first 10 days)")
    lines.append(daily_peak.head(10).to_markdown(index=False))
    lines.append("")
    lines.append("## By horizon")
    lines.append(by_h.to_markdown(index=False))
    lines.append("")
    lines.append("## Features used")
    lines.append(f"- Origin features: {', '.join(ORIGIN_FEATURES)}")
    lines.append(f"- Target features: {', '.join('target_'+c for c in TARGET_FEATURES)}")
    lines.append("")
    OUT_MD.write_text("\n".join(lines), encoding="utf-8")

    print("✅ Backtest complete")
    print(f"  Overall MAE:  {overall_mae:.3f} MW")
    print(f"  Overall RMSE: {overall_rmse:.3f} MW")
    print(f"  Output: {OUT_MD}")
    print(f"  Output: {OUT_CSV}")
    print(f"  Output: {OUT_PEAK_CSV}")


if __name__ == "__main__":
    main()
