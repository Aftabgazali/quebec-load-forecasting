# Baseline Results (Day-ahead: next 24 hours)

These are intentionally simple “no-ML” reference forecasts. They use only *past demand* (no weather),
so they set a clean minimum performance level that our real model must beat.

### 1) weekly_naive_lag168 (Seasonal Naive)
**Rule:** For each target hour tomorrow, predict the demand from the **same hour last week**.  
Example: “Tomorrow 18:00 ≈ Last week 18:00” (lag = 168 hours).

### 2) weekly_avg_k4 (Seasonal Average, k=4)
**Rule:** For each target hour tomorrow, predict the **average of the same hour** from the last **4 weeks**.  
Example: “Tomorrow 18:00 ≈ average(18:00 from 1, 2, 3, 4 weeks ago)”.

---

## What the columns mean

- **days**: number of backtest days (folds). Each fold forecasts one full next-day (24 hours).
- **n_points**: number of hourly predictions that were actually scored.
  This is slightly less than `days × 24` due to missing hours in the dataset and time quirks (e.g., DST).
- **MAE (MW)**: *typical hourly miss* in megawatts.
  MAE = 1751 means: across all evaluated hours, the forecast is off by about **1751 MW per hour** on average.
- **RMSE (MW)**: like MAE, but penalizes big misses more (so it’s usually higher).

Lower is better for both MAE and RMSE.

---

## Backtest Results (Direct 24 specialists + LightGBM, origin = 23:00)

**Setup**

* Forecasting method: **Direct multi-step** (24 separate models, horizons 1…24).
* Learner: **LightGBM regressor** per horizon.
* Backtesting: **walk-forward / rolling-origin**, expanding window.
* Forecast origin: **daily at 23:00** (end of day D).
* Target window per fold: **next 24 hours** (D+1 00:00 → 23:00).
* Weather input: **observed hourly temperature joined to timestamps**, plus a **non-leaky “forecast proxy”** based only on past temperatures (7-day same-hour average using previous days), plus weather lags.
* Evaluation span: **90 days** (90 daily folds).

**Baselines (same origin and evaluation window)**

* Weekly naive (lag 168): **MAE = 2064.35 MW**, **RMSE = 2723.13 MW**, **WAPE = 8.99%**, **Bias = −547.87 MW** (n_points = 2134).
* Weekly average (k=4): **MAE = 2363.54 MW**, **RMSE = 3068.08 MW**, **WAPE = 10.29%**, **Bias = −1662.83 MW** (n_points = 2135).

**Model performance (LightGBM, Direct 24 specialists)**

* Overall: **MAE = 706.868 MW**, **RMSE = 1023.794 MW**, **WAPE = 3.09%**, **Bias = −21.671 MW** (90 folds, 2160 scored points).
* Under the same protocol, the model improves over the strongest baseline (weekly naive) by **~65.8% MAE** (and **~65.6% WAPE**).

**Horizon behavior**

* Error increases with horizon (farther into tomorrow is harder). In this backtest, **MAE rises from ~183 MW at h=1 to ~1125 MW at h=24**, with the hardest horizons typically **18–24**.

**Interpretation**

* The baselines show this is a non-trivial forecasting task (MAE ≈ **2.1–2.4 GW**).
* Using demand lags/rolling stats, calendar signals, and weather features, the LightGBM horizon specialists capture daily/weekly structure and weather sensitivity, reducing error to **~0.71 GW MAE** with near-zero overall bias.
* Peak diagnostics show the model is generally close on daily peak magnitude (mean peak error **−94.7 MW**) and typically within **~1 hour** on peak timing.
