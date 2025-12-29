from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple, List, Dict

import pandas as pd


@dataclass(frozen=True)
class RunPaths:
    run_id: str
    reports_run_dir: Optional[Path]
    data_run_dir: Optional[Path]


def load_csv(path: Path) -> pd.DataFrame:
    return pd.read_csv(path)

def repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def reports_dir() -> Path:
    return repo_root() / "reports"


def data_runs_dir() -> Path:
    return repo_root() / "data" / "forecasts" / "runs"


def list_runs() -> List[RunPaths]:
    # union run_ids from reports/runs and data/forecasts/runs
    rep_base = reports_dir() / "runs"
    dat_base = data_runs_dir()

    rep_ids = set()
    dat_ids = set()

    if rep_base.exists():
        rep_ids = {p.name for p in rep_base.iterdir() if p.is_dir()}
    if dat_base.exists():
        dat_ids = {p.name for p in dat_base.iterdir() if p.is_dir()}

    all_ids = sorted(rep_ids | dat_ids)

    runs: List[RunPaths] = []
    for rid in all_ids:
        rdir = (rep_base / rid) if (rep_base / rid).exists() else None
        ddir = (dat_base / rid) if (dat_base / rid).exists() else None
        runs.append(RunPaths(run_id=rid, reports_run_dir=rdir, data_run_dir=ddir))
    return runs

def pick_forecast_csv(csv_files: List[Path]) -> Optional[Path]:
    if not csv_files:
        return None

    # Prefer filenames that look like forecasts
    preferred = []
    for p in csv_files:
        name = p.name.lower()
        if "forecast" in name or "pred" in name or "yhat" in name:
            preferred.append(p)

def read_json(path: Path) -> dict:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def load_parquet(path: Path) -> pd.DataFrame:
    return pd.read_parquet(path)


def detect_columns(df: pd.DataFrame) -> dict:
    cols = {c.lower(): c for c in df.columns}

    time_candidates = ["target_ts", "target_time", "timestamp", "time", "ds", "datetime", "origin_ts"]
    time_col = next((cols[c] for c in time_candidates if c in cols), None)

    p10_col = next((cols[c] for c in ["p10", "q10", "yhat_p10"] if c in cols), None)
    p50_col = next((cols[c] for c in ["p50", "q50", "yhat", "yhat_p50", "pred", "prediction"] if c in cols), None)
    p90_col = next((cols[c] for c in ["p90", "q90", "yhat_p90"] if c in cols), None)

    actual_col = next((cols[c] for c in ["actual", "y", "load_mw"] if c in cols), None)

    return {"time": time_col, "p10": p10_col, "p50": p50_col, "p90": p90_col, "actual": actual_col}
