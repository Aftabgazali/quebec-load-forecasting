"""
Run baseline backtests on the modeling table.

Baselines:
- weekly_naive: yhat(t) = y(t-168)
- weekly_avg_k: mean of same hour across last k weeks

Output:
- reports/backtest/baselines_metrics.csv
- reports/backtest/baselines_summary.md
"""

from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd
import yaml
import tabulate

cfg = yaml.safe_load(Path("configs/default.yaml").read_text(encoding="utf-8"))

IN_PARQUET = Path("data/processed/modeling_table.parquet")
OUT_DIR = Path("reports/backtest")
OUT_CSV = OUT_DIR / "baselines_metrics.csv"
OUT_MD = OUT_DIR / "baselines_summary.md"

ORIGIN_HOUR = cfg["baselines"]["origin_hour"]
HORIZON_HOURS = cfg["baselines"]["horizon_hours"]
EVAL_DAYS = cfg["baselines"]["eval_days"]
K_WEEKS = cfg["baselines"]["weekly_average_k"]


def mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.mean(np.abs(y_true - y_pred)))


def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def main() -> None:
    if not IN_PARQUET.exists():
        raise SystemExit(f"Missing input: {IN_PARQUET}")

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    df = pd.read_parquet(IN_PARQUET).sort_values("timestamp").reset_index(drop=True)

    # We'll evaluate on timestamps where y is known
    # For baselines, we can compute predictions directly from lag features.
    # weekly naive uses lag_168; weekly avg uses lag_168, lag_336, ...
    # We'll reconstruct those extra lags from the filled series approach:
    # easiest: use y itself with shifts.
    # Build a list of unique dates available (based on timestamp local date)
    # Choose fold cutoffs: each cutoff is a timestamp where we "issue" a day-ahead forecast.
    # We'll use midnight cutoffs for simplicity: predict the next day (00:00..23:00).
    # So cutoff is end of day D at 23:00 -> next day starts at 00:00.
    # We'll implement by stepping over days via timestamps.
    series = df.set_index("timestamp")["y"]
    origins = df.loc[df["timestamp"].dt.hour == ORIGIN_HOUR, "timestamp"].sort_values().unique()
    eval_origins = origins[-EVAL_DAYS:]
    results = []

    # Backtest over all possible next-day forecasts (expanding window notion doesn’t matter for baselines)
    # For each day i, forecast day i (as D+1) using history up to day i-1.
    for origin_ts in eval_origins:
        target_index = pd.date_range(
            # target window: this day from 00:00 to 23:00
            start=pd.Timestamp(origin_ts) + pd.Timedelta(hours=1),
            periods=HORIZON_HOURS,
            freq="h",
            tz=series.index.tz
        )
        forecast_day = (pd.Timestamp(origin_ts) + pd.Timedelta(hours=1)).date()
        y_true = series.reindex(target_index).to_numpy()

        # weekly naive:
        yhat_naive = series.shift(168).reindex(target_index).to_numpy()

        # weekly avg k:
        preds = [series.shift(168*w).reindex(target_index).to_numpy() for w in range(1, K_WEEKS+1)]
        yhat_avg = np.nanmean(np.vstack(preds), axis=0)
        # Score: only where both pred and true exist
        def score_one(name: str, yhat: np.ndarray):
            mask = ~np.isnan(y_true) & ~np.isnan(yhat)
            if mask.sum() == 0:
                return

            yt = y_true[mask]
            yp = yhat[mask]

            abs_err = np.abs(yt - yp)
            err = (yp - yt)
            sq_err = (yt - yp) ** 2

            # WAPE (safe): sum(|e|)/sum(y)
            y_sum = float(np.sum(yt))
            wape = float(abs_err.sum() / y_sum) if y_sum != 0 else np.nan

            # Bias: mean(yp - yt)
            bias = float(err.mean())

            # Peak error + peak timing error (hours)
            idx = target_index[mask]
            s_true = pd.Series(yt, index=idx)
            s_pred = pd.Series(yp, index=idx)

            true_peak_ts = s_true.idxmax()
            pred_peak_ts = s_pred.idxmax()
            peak_error_mw = float(s_pred.max() - s_true.max())
            peak_timing_error_h = float((pred_peak_ts - true_peak_ts) / pd.Timedelta(hours=1))
            results.append(
                {
                    "date": str(forecast_day),
                    "baseline": name,
                    "n": int(mask.sum()),

                    # keep these (handy)
                    "mae": mae(yt, yp),
                    "rmse": rmse(yt, yp),

                    # NEW metrics (day-level)
                    "wape": wape,
                    "bias": bias,
                    "peak_error_mw": peak_error_mw,
                    "peak_timing_error_h": peak_timing_error_h,

                    # store sums so aggregate metrics are correct (not “mean of means”)
                    "abs_err_sum": float(abs_err.sum()),
                    "sq_err_sum": float(sq_err.sum()),
                    "err_sum": float(err.sum()),
                    "y_sum": y_sum,

                    # context (fix: use that day, not whole dataset)
                    "average_demand": float(np.mean(yt)),
                }
            )
        score_one("weekly_naive_lag168", yhat_naive) 
        score_one(f"weekly_avg_k{K_WEEKS}", yhat_avg)

    res = pd.DataFrame(results)
    if res.empty:
        raise SystemExit("No baseline results computed (unexpected).")

    # Aggregate across all folds
    summary = (
    res.groupby("baseline")
    .agg(
        days=("date", "nunique"),
        n_points=("n", "sum"),

        # sums for correct overall metrics
        abs_err_sum=("abs_err_sum", "sum"),
        sq_err_sum=("sq_err_sum", "sum"),
        err_sum=("err_sum", "sum"),
        y_sum=("y_sum", "sum"),

        # average across days for ops-style metrics
        peak_error_mw=("peak_error_mw", "mean"),
        peak_timing_error_h=("peak_timing_error_h", "mean"),

        average_demand=("average_demand", "mean"),
    )
    .reset_index()
)

    # Derive overall point metrics from sums (more correct than averaging daily MAE/RMSE)
    summary["mae"] = summary["abs_err_sum"] / summary["n_points"]
    summary["rmse"] = np.sqrt(summary["sq_err_sum"] / summary["n_points"])
    summary["wape"] = summary["abs_err_sum"] / summary["y_sum"]
    summary["bias"] = summary["err_sum"] / summary["n_points"]

    # Nice presentation order
    summary = summary[
        [
            "baseline", "days", "n_points",
            "mae", "rmse", "wape", "bias",
            "peak_error_mw", "peak_timing_error_h",
            "average_demand",
        ]
    ].sort_values("mae")

    summary.to_csv(OUT_CSV, index=False)

    # Simple markdown report
    lines = []
    lines.append("# Baselines backtest summary")
    lines.append("")
    lines.append(f"- Horizon: {HORIZON_HOURS} hours (next day)")
    lines.append(f"- Weekly average k: {K_WEEKS}")
    lines.append("")
    lines.append("## Results (lower is better)")
    lines.append("")
    lines.append(summary.to_markdown(index=False))
    OUT_MD.write_text("\n".join(lines), encoding="utf-8")

    print("✅ Baselines backtest complete")
    print(f"  Output: {OUT_CSV}")
    print(f"  Output: {OUT_MD}")
    print("")
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
