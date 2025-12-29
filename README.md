# Québec Day-Ahead Electricity Demand Forecasting (V1.0)

Forecast **Hydro-Québec system load** for the **next 24 hours** (day-ahead) using demand history + calendar signals + temperature features (non-leaky proxy).

This repo is intentionally “simple pipeline, serious backtesting”:
- Strong baselines
- Leakage-safe feature building
- Walk-forward evaluation
- **Direct multi-step** forecasting (24 specialist models)

---

## Data

### Hydro-Québec demand (hourly historical)
Dataset: `historique-demande-electricite-quebec` (Hydro-Québec open data).

**Important timestamp detail:** Hydro-Québec labels each hourly value at the **end of the hour** (e.g., the value at `02:00` corresponds to roughly `01:00–02:00`).  
In `clean_align.py`, timestamps are shifted by **-1 hour** to make our canonical index **start-of-hour** (e.g., `02:00 → 01:00`) and then reindexed to a continuous hourly timeline.

### Hydro-Québec demand (live / V1.2 later)
Dataset: `demande-electricite-quebec` (15‑minute updates). This will be used later for the live loop.

### Weather
We ingest hourly `temperature_2m` and align it to the same hourly timestamp index.

---

## Problem framing

### Task
At a daily forecast origin, predict the next **24 hourly** demand values (day-ahead).

### Method summary
1) Build two baselines (no ML):
   - **weekly_naive_lag168**: tomorrow hour ≈ same hour last week
   - **weekly_avg_k4**: tomorrow hour ≈ average of same hour over last 4 weeks
2) Train the main ML model:
   - **Direct multi-step**: 24 separate “specialist” regressors (horizons 1…24)
   - Learner: **LightGBM**
3) Evaluate using walk-forward backtesting.

---

## Weather handling (day-ahead, no leakage)

We *do not* use tomorrow’s observed temperature when predicting tomorrow (that would be leakage).

Instead, we create a **forecast temperature proxy** using only past observed temperatures:

**Proxy (hourly, for target hour on D+1):**  
`temp_fcst_proxy(D+1, h) = mean( temp_obs(D, h), temp_obs(D-1, h), … , temp_obs(D-6, h) )`

In other words: **7‑day average of the same hour**, ending at the forecast origin day.

This provides a realistic “known-at-forecast-time” temperature input without requiring archived forecast vintages.

---

## Backtesting protocol (V1.0)

- Forecasting method: **Direct multi-step** (24 specialists, horizons 1…24)
- Backtesting: **walk-forward / rolling-origin**, expanding window
- Forecast origin: **daily at 23:00** (end of day D)
- Target window per fold: **next 24 hours** (D+1 00:00 → 23:00)
- Evaluation span: **90 daily folds** (configurable)

Metrics:
- **MAE (MW)**: average absolute error (typical miss)
- **RMSE (MW)**: penalizes big misses more

---

## Results (V1.0)

All numbers below use the **same origin (23:00)** and the **same 90‑day evaluation window**, so they are apples-to-apples.

### Baselines (no ML)
- Weekly naive (lag 168): **MAE = 2065.00 MW**, **RMSE = 2237.56 MW**
- Weekly average (k=4): **MAE = 2363.88 MW**, **RMSE = 2476.57 MW**

### LightGBM (Direct 24 specialists)
- Overall: **MAE = 706.868 MW**, **RMSE = 1023.794 MW** (90 folds)

Notes:
- Error increases with horizon; worst horizons are typically **18–24**.
- Tuning note: `num_leaves=128` improved MAE to ~**706 MW**.

See `RESULTS.md` for the full write-up and the horizon table.

---

## Repo structure (key files)

- `src/ingest_hq_demand.py` — download HQ hourly historical demand (raw)
- `src/clean_align.py` — timezone + end-of-hour → start-of-hour alignment + hourly reindex
- `src/ingest_weather.py` — ingest hourly temperature and align timestamps
- `src/build_features.py` — calendar + lags + rolling means + weather proxy features
- `src/run_baselines.py` — seasonal naive baselines backtest
- `src/run_backtest.py` — ML walk-forward backtest (LightGBM, 24 specialists)
- `src/train_models.py` — train final 24 specialist models and save to `models/`
- `src/predict_next_day.py` — generate a “forecast run” artifact folder (demo)

Artifacts:
- `reports/backtest/` — baseline + model backtest summaries
- `data/forecasts/runs/` — forecast-run folders (inputs + predictions)

---

## How to reproduce (minimal)

1) Ingest + clean demand  
- `python src/ingest_hq_demand.py`  
- `python src/clean_align.py`

2) Ingest weather + build features  
- `python src/ingest_weather.py`  
- `python src/build_features.py`

3) Backtests  
- `python src/run_baselines.py`  
- `python src/run_backtest.py`

4) Train final models  
- `python src/train_models.py`

---

## Roadmap

### V1.1 (small, portfolio-friendly)
- Add 1–2 plots to `reports/backtest/`:
  - MAE by horizon
  - MAE by hour-of-day
- Add a clean “baseline vs model” comparison table in the backtest report

### V1.2 (live daily loop)
- Daily ingestion from HQ 15‑minute dataset (aggregate to hourly)
- Daily evaluation of prior forecasts once actuals arrive
- Weekly retraining (or retrain when performance degrades)
- Automatically archive daily runs in `data/forecasts/runs/<run_id>/` and generate a report

---

## License
MIT