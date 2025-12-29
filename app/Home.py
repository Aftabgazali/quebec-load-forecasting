import re
from pathlib import Path

import pandas as pd
import streamlit as st

from lib.io import list_runs, reports_dir

st.set_page_config(page_title="Québec Load Forecasting Dashboard", layout="wide")

st.title("Québec Day-Ahead Load Forecasting — Dashboard (V1.1)")
st.caption("Read-only view of pipeline outputs written into `reports/`.")

# ---------- Helpers ----------
def safe_read_csv(path: Path) -> pd.DataFrame | None:
    try:
        if path.exists():
            return pd.read_csv(path)
    except Exception:
        return None
    return None

def parse_overall_mae_from_md(md_path: Path) -> float | None:
    """
    Tries to extract something like:
      - "Overall MAE:  734.002 MW"
      - or "- MAE:  734.002 MW"
    """
    if not md_path.exists():
        return None
    txt = md_path.read_text(encoding="utf-8", errors="ignore")
    m = re.search(r"MAE:\s*([0-9]+(?:\.[0-9]+)?)", txt)
    if not m:
        return None
    return float(m.group(1))

def switch_page(page_path: str) -> None:
    # Minimal compatibility across Streamlit versions
    if hasattr(st, "switch_page"):
        st.switch_page(page_path)
    elif hasattr(st, "experimental_switch_page"):
        st.experimental_switch_page(page_path)
    else:
        st.info("Use the left sidebar to navigate.")

# ---------- Locate report artifacts ----------
rep_dir = Path(reports_dir())

runs = list_runs()
latest_run_id = runs[-1].run_id if runs else None

baseline_csv = rep_dir / "backtest" / "baselines_metrics.csv"
by_horizon_csv = rep_dir / "backtest" / "metrics_by_horizon.csv"
backtest_md = rep_dir / "backtest" / "backtest_summary.md"

baseline_df = safe_read_csv(baseline_csv)
by_horizon_df = safe_read_csv(by_horizon_csv)
overall_mae = parse_overall_mae_from_md(backtest_md)

best_baseline_name = None
best_baseline_mae = None
if baseline_df is not None and "mae" in baseline_df.columns and len(baseline_df) > 0:
    # pick the lowest MAE baseline
    i = baseline_df["mae"].astype(float).idxmin()
    best_baseline_name = str(baseline_df.loc[i, "baseline"]) if "baseline" in baseline_df.columns else "baseline"
    best_baseline_mae = float(baseline_df.loc[i, "mae"])

# ---------- KPI Cards ----------
c1, c2, c3, c4 = st.columns(4)

c1.metric("Latest run", latest_run_id or "—")
c2.metric("Runs found", len(runs))
c3.metric("Backtest MAE (model)", f"{overall_mae:.0f} MW" if overall_mae is not None else "—")
if best_baseline_mae is not None:
    c4.metric("Best baseline MAE", f"{best_baseline_mae:.0f} MW", help=f"Best baseline: {best_baseline_name}")
else:
    c4.metric("Best baseline MAE", "—")

# ---------- Quick actions ----------
st.subheader("Quick actions")
b1, b2, b3 = st.columns(3)

with b1:
    if st.button("View latest forecast", use_container_width=True, disabled=latest_run_id is None):
        switch_page("pages/1_Forecast.py")

with b2:
    if st.button("View backtest", use_container_width=True):
        switch_page("pages/2_Backtest.py")

with b3:
    if st.button("View monitoring", use_container_width=True):
        switch_page("pages/3_Monitoring.py")

# ---------- Pipeline status ----------
st.subheader("Pipeline status")

status_rows = []

status_rows.append(("reports/ folder found", rep_dir.exists()))
status_rows.append(("runs present", len(runs) > 0))
status_rows.append(("baseline metrics present", baseline_csv.exists()))
status_rows.append(("backtest by-horizon metrics present", by_horizon_csv.exists()))
status_rows.append(("backtest summary present", backtest_md.exists()))

# You can add “actuals present” later (when you implement scored runs)
status_df = pd.DataFrame(status_rows, columns=["check", "ok"])
status_df["status"] = status_df["ok"].apply(lambda x: "✅" if x else "❌")
st.dataframe(status_df[["status", "check"]], hide_index=True, use_container_width=True)

# ---------- Recent runs preview ----------
st.subheader("Recent runs")
if runs:
    last = runs[-5:]
    st.write([r.run_id for r in last][::-1])
else:
    st.warning("No run folders found under `reports/runs/`. Run your pipeline once to generate a run artifact.")

# ---------- About (collapsed) ----------
with st.expander("About / How to use"):
    st.markdown(
        """
This dashboard is **read-only**: it only displays what your pipeline wrote into `reports/`.

Use the pages on the left:
- **Forecast**: inspect a single run (24h forecast, bands if available, metadata).
- **Backtest**: show baselines + horizon metrics from `reports/backtest/`.
- **Monitoring**: anomalies when actual is outside [P10, P90] (only works once quantiles + actuals exist).
"""
    )

with st.expander("Paths (debug)"):
    st.write("reports dir:", str(rep_dir))
    st.write("baseline metrics:", str(baseline_csv))
    st.write("horizon metrics:", str(by_horizon_csv))
    st.write("backtest summary:", str(backtest_md))