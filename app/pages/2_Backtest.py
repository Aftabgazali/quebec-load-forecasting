import streamlit as st
import pandas as pd
import altair as alt
from pathlib import Path

from lib.io import reports_dir

def fmt_metrics(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()

    # WAPE as percent
    if "wape" in out.columns:
        out["wape_%"] = (out["wape"].astype(float) * 100).round(2)
        out = out.drop(columns=["wape"])

    # round MW-ish columns
    for c in ["mae", "rmse", "bias", "peak_error_mw"]:
        if c in out.columns:
            out[c] = out[c].astype(float).round(2)

    # round timing columns
    for c in ["peak_timing_error_h"]:
        if c in out.columns:
            out[c] = out[c].astype(float).round(3)

    return out


st.title("Backtest")

base = reports_dir() / "backtest"
bm = base / "baselines_metrics.csv"
mh = base / "metrics_by_horizon.csv"

if not base.exists():
    st.warning("`reports/backtest/` not found.")
    st.stop()

st.subheader("Baselines metrics")
if bm.exists():
    df_bm = pd.read_csv(bm)
    st.dataframe(fmt_metrics(df_bm), use_container_width=True)
else:
    st.info("Missing baselines_metrics.csv")

st.subheader("Metrics by horizon (1..24)")
if mh.exists():
    df = pd.read_csv(mh)
    # Baseline comparison callout (MAE skill vs weekly naive)
    if bm.exists() and "mae" in df.columns and "n" in df.columns:
        # model overall MAE from horizon table (weighted by n)
        model_mae = float((df["mae"] * df["n"]).sum() / df["n"].sum())

        bn = df_bm[df_bm["baseline"] == "weekly_naive_lag168"] if "baseline" in df_bm.columns else pd.DataFrame()
        if not bn.empty and "mae" in bn.columns:
            baseline_mae = float(bn["mae"].iloc[0])
            improvement = (baseline_mae - model_mae) / baseline_mae * 100
            st.info(f"Model MAE improves over weekly_naive by **{improvement:.1f}%**")
    st.dataframe(fmt_metrics(df).head(30), use_container_width=True)
    # Let user pick a metric column to plot
    numeric_cols = [c for c in df.columns if c.lower() not in ["horizon", "h"]]
    horizon_col = "horizon" if "horizon" in df.columns else ("h" if "h" in df.columns else None)

    if horizon_col is None:
        st.warning("Could not find a horizon column (expected 'horizon' or 'h').")
        st.stop()
    st.subheader("Daily peak diagnostics")
    peak_csv = base / "daily_peak_diagnostics.csv"

    if peak_csv.exists():
        df_peak = pd.read_csv(peak_csv)
        st.dataframe(fmt_metrics(df_peak).head(30), use_container_width=True)

        # quick KPIs (optional but nice)
        c1, c2 = st.columns(2)
        c1.metric("Mean peak error (MW)", float(df_peak["peak_error_mw"].mean()))
        c2.metric("Mean peak timing error (h)", float(df_peak["peak_timing_error_h"].mean()))
    else:
        st.info("Missing daily_peak_diagnostics.csv (peak metrics are currently only in the markdown summary).")

    st.caption("Bias is mean(y_pred − y_true).  bias < 0 = under-forecasting, bias > 0 = over-forecasting.")
    default_metric = "mae" if "mae" in numeric_cols else numeric_cols[0]
    metric = st.selectbox("Choose metric to plot", numeric_cols, index=numeric_cols.index(default_metric))

    plot_df = df[[horizon_col, metric]].copy()
    plot_df.columns = ["horizon", "value"]
    y_title = metric
    if metric in ["mae", "rmse", "bias"]:
        y_title = f"{metric} (MW)"
    chart = alt.Chart(plot_df).mark_line().encode(
        x=alt.X("horizon:Q", title="Horizon (hours)"),
        y=alt.Y("value:Q", title=y_title),
    ).properties(height=420).interactive()
    
    st.altair_chart(chart, use_container_width=True)
else:
    st.info("Missing metrics_by_horizon.csv")
