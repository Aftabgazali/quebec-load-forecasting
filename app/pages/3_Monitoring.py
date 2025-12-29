import streamlit as st
import pandas as pd

from lib.io import list_runs, pick_forecast_csv, load_csv, detect_columns

st.title("Monitoring / Anomalies")

runs = list_runs()
if len(runs) < 2:
    st.info("Need at least 2 runs to use lookback. Also requires p10/p90 + actuals for anomalies.")
    st.stop()

if not runs:
    st.warning("No runs found.")
    st.stop()

max_lb = min(200, len(runs))
lookback = st.slider(
    "Lookback (most recent runs)",
    min_value=2,
    max_value=max_lb,
    value=min(30, max_lb),
)

recent = runs[-lookback:]

rows = []
for run in recent:
    csv_path = pick_forecast_csv(run.csv_files)
    if csv_path is None:
        continue
    df = load_csv(csv_path)
    cols = detect_columns(df)
    if not (cols["time"] and cols["p10"] and cols["p90"] and cols["actual"]):
        continue

    tmp = df[[cols["time"], cols["p10"], cols["p90"], cols["actual"]]].copy()
    tmp.columns = ["time", "p10", "p90", "actual"]
    tmp["run_id"] = run.run_id
    tmp["anomaly"] = (tmp["actual"] < tmp["p10"]) | (tmp["actual"] > tmp["p90"])
    tmp["severity_mw"] = 0.0
    tmp.loc[tmp["actual"] < tmp["p10"], "severity_mw"] = (tmp["p10"] - tmp["actual"])
    tmp.loc[tmp["actual"] > tmp["p90"], "severity_mw"] = (tmp["actual"] - tmp["p90"])
    rows.append(tmp[tmp["anomaly"]])

if not rows:
    st.info(
        "No anomalies found (or missing columns). "
        "This page needs runs that include p10/p90 AND actuals."
    )
    st.stop()

anoms = pd.concat(rows, ignore_index=True).sort_values("severity_mw", ascending=False)

col1, col2, col3 = st.columns(3)
col1.metric("Anomalies (rows)", int(anoms.shape[0]))
col2.metric("Max severity (MW)", float(anoms["severity_mw"].max()))
col3.metric("Runs scanned", lookback)

st.subheader("Anomalies table")
st.dataframe(anoms, use_container_width=True)
