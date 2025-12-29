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

* Weekly naive (lag 168): **MAE = 2065.00 MW**, **RMSE = 2237.56 MW** (n_points = 2134).
* Weekly average (k=4): **MAE = 2363.88 MW**, **RMSE = 2476.57 MW** (n_points = 2135).

**Model performance (LightGBM, Direct 24)**

* Overall: **MAE = 706.868 MW**, **RMSE = 1023.794 MW** (90 folds).
* The model **substantially outperforms** the strongest baseline under the same evaluation protocol.

**Horizon behavior**

* As expected, error increases with horizon; the worst performance occurs at **horizons 18–24** (farthest ahead and typically overlapping more volatile late-day hours).
* Example horizon metrics (n=90 each):

  * h=18: MAE 999.52, RMSE 1282.49
  * h=19: MAE 1022.12, RMSE 1382.36
  * h=20: MAE 1103.44, RMSE 1460.56
  * h=21: MAE 1107.86, RMSE 1490.13
  * h=22: MAE 1124.81, RMSE 1501.06
  * h=23: MAE 1127.05, RMSE 1511.86
  * h=24: MAE 1139.13, RMSE 1509.94

**Interpretation**

* Baselines confirm the task is non-trivial (MAE ~2.1–2.4 GW).
* Adding demand lags/rolling means, calendar signals, and weather features allows LightGBM specialists to capture daily/weekly structure and weather sensitivity, yielding a large reduction in error (MAE ~0.73 GW).
