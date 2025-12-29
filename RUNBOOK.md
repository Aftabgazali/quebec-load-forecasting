# RUNBOOK — Québec Day-Ahead Load Forecasting

Source of truth:
- SPEC.md (time conventions, issue time, weather vintage rule)
- project_blueprint.md (project scope & deliverables)

All commands use plain scripts:
***- python src/<scripts>.py***

---

## Phase 1 — MVP (offline, reproducible)

### 1) Ingest demand (HQ historical)
Command:
- python src/ingest_hq_demand.py

Expected outputs:
- data/raw/hq_demand/...

### 2) Ingest weather (historic for backtests)
Command:
- python src/ingest_weather.py

Expected outputs:
- data/raw/weather_historic/...

### 3) Clean + align (timezone + hourly index + HQ timestamp alignment)
Command:
- python src/clean_align.py

Expected outputs:
- data/processed/demand_hourly.parquet
- data/processed/weather_hourly.parquet

### 4) Build modeling table (calendar + lags + join weather)
Command:
- python src/build_features.py

Expected outputs:
- data/processed/modeling_table.parquet

### 5) Run baselines backtest
Command:
- python src/run_baselines.py

Expected outputs:
- reports/backtest/baselines_summary.md
- reports/backtest/baselines_metrics.csv

### 6) Run model backtest (Direct method: 24 specialists, expanding window)
Command:
- python src/run_backtest.py

Expected outputs:
- reports/backtest/backtest_summary.md
- reports/backtest/metrics_by_horizon.csv

### 7) Train final models (24 specialists)
Command:
- python src/train_models.py

Expected outputs:
- models/...(one file per horizon)
- models/metadata_*.json

### 8) Predict next day (demo run archive)
Command:
- python src/predict_next_day.py

Expected outputs:
- data/forecasts/runs/<run_id>/
  - inputs.json
  - yhat.parquet
  - weather_used.parquet
- reports/runs/<run_id>/summary.md

---

## Phase 2 — V1.2 (daily live loop) [later]

### Daily batch run: fetch → eval → retrain (weekly/trigger) → forecast → archive report
Command:
- python src/live_daily_run.py

Expected outputs:
- data/forecasts/runs/<run_id>/... (new run folder)
- reports/runs/<run_id>/summary.md
- metrics history updates (optional)
