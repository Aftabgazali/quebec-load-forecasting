import altair as alt
import streamlit as st
import pandas as pd

from lib.io import list_runs, load_parquet

TZ = "America/Toronto"

st.title("Forecast")

runs = list_runs()
if not runs:
    st.warning("No runs found.")
    st.stop()

run_ids = [r.run_id for r in runs]
run_id = st.selectbox("Select run", run_ids, index=len(run_ids) - 1)
run = next(r for r in runs if r.run_id == run_id)

left, right = st.columns([1, 2])

# ---------- LEFT: keep only the summary (remove Inputs section) ----------
with left:
    st.subheader("Run summary (reports)")
    if run.reports_run_dir is not None:
        summary_md = run.reports_run_dir / "summary.md"
        if summary_md.exists():
            st.markdown(summary_md.read_text(encoding="utf-8"))
        else:
            st.info("No summary.md found.")
    else:
        st.info("No reports run folder for this run_id.")


# ---------- RIGHT: load forecast + actuals and plot ----------
with right:
    st.subheader("Forecast vs Actual (next-day)")

    if run.data_run_dir is None:
        st.error("No data run folder found for this run_id.")
        st.stop()

    yhat_path = run.data_run_dir / "yhat.parquet"
    if not yhat_path.exists():
        st.error("Missing yhat.parquet in data run folder.")
        st.stop()

    yhat_df = load_parquet(yhat_path).copy()

    # Your schema (from screenshot): origin_ts, target_ts, horizon, yhat
    if "target_ts" not in yhat_df.columns or "yhat" not in yhat_df.columns:
        st.error(f"Unexpected yhat schema. Found columns: {list(yhat_df.columns)}")
        st.stop()

    # Ensure target_ts is tz-aware datetime
    yhat_df["target_ts"] = pd.to_datetime(yhat_df["target_ts"])
    if yhat_df["target_ts"].dt.tz is None:
        # If it's naive, assume it's already Toronto time
        yhat_df["target_ts"] = yhat_df["target_ts"].dt.tz_localize(TZ)

    # ---- Load actuals from modeling_table.parquet ----
    modeling_path = run.data_run_dir.parents[2] / "processed" / "modeling_table.parquet"
    # run.data_run_dir = repo/data/forecasts/runs/<id>
    # parents[2] = repo/data
    # so data/processed/modeling_table.parquet is correct

    if not modeling_path.exists():
        st.warning("modeling_table.parquet not found at data/processed/. Can't plot actuals.")
        st.dataframe(yhat_df[["target_ts", "yhat", "horizon"]], use_container_width=True)
        st.stop()

    # ---- Load actuals from modeling_table.parquet (DST-safe) ----
    actual_df = pd.read_parquet(modeling_path, columns=["timestamp", "y"]).copy()

    # Define the forecast window in UTC *before* converting/flooring
    tmin = yhat_df["target_ts"].min().floor("h")
    tmax = yhat_df["target_ts"].max().floor("h")

    tmin_utc = tmin.tz_convert("UTC")
    tmax_utc = (tmax + pd.Timedelta(hours=1)).tz_convert("UTC")  # exclusive upper bound

    # Build a UTC timestamp column robustly (works whether "timestamp" is int-ms OR datetime)
    ts_raw = actual_df["timestamp"]

    if pd.api.types.is_datetime64_any_dtype(ts_raw):
        ts = pd.to_datetime(ts_raw)

        # If it's timezone-naive (unlikely here), assume it's Toronto time
        if ts.dt.tz is None:
            ts = ts.dt.tz_localize(TZ, ambiguous="infer", nonexistent="shift_forward")

        ts_utc = ts.dt.tz_convert("UTC")
    else:
        # epoch milliseconds -> UTC
        ts_utc = pd.to_datetime(ts_raw, unit="ms", utc=True)

    # Filter in UTC (no DST ambiguity)
    actual_df = actual_df[(ts_utc >= tmin_utc) & (ts_utc < tmax_utc)].copy()

    # Create Toronto-time hourly timestamps for merging
    actual_df["ts"] = ts_utc.dt.floor("h").dt.tz_convert(TZ)

    # Merge actuals onto forecast
    merged = yhat_df.merge(actual_df[["ts", "y"]], left_on="target_ts", right_on="ts", how="left")
    merged = merged.sort_values("target_ts")

    # Quick sanity table
    st.caption(f"Forecast rows: {len(yhat_df)} • Actual rows in window: {len(actual_df)}")
    st.dataframe(merged[["target_ts", "horizon", "yhat", "y"]], use_container_width=True)

    # Metrics (only where actual exists)
    scored = merged.dropna(subset=["y"])
    if len(scored) > 0:
        mae = (scored["yhat"] - scored["y"]).abs().mean()
        st.metric("MAE (available hours)", float(mae))
    else:
        st.info("No actuals available for these target hours yet (y is NaN).")

    # Build long-form for plotting two lines
    plot_df = merged[["target_ts", "yhat", "y"]].copy()
    plot_df["target_ts_str"] = plot_df["target_ts"].astype(str)

    long = plot_df.melt(
        id_vars=["target_ts_str"],
        value_vars=["yhat", "y"],
        var_name="series",
        value_name="value",
    ).dropna(subset=["value"])

    chart = (
        alt.Chart(long)
        .mark_line()
        .encode(
            x=alt.X("target_ts_str:N", title="Target hour"),
            y=alt.Y("value:Q", title="MW"),
            color=alt.Color("series:N", title=""),
        )
        .properties(height=420)
        .interactive()
    )

    st.altair_chart(chart, use_container_width=True)