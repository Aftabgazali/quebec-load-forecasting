# Baselines backtest summary

- Horizon: 24 hours (next day)
- Weekly average k: 4

## Results (lower is better)

| baseline            |   days |   n_points |     mae |    rmse |      wape |      bias |   peak_error_mw |   peak_timing_error_h |   average_demand |
|:--------------------|-------:|-----------:|--------:|--------:|----------:|----------:|----------------:|----------------------:|-----------------:|
| weekly_naive_lag168 |     89 |       2134 | 2064.35 | 2723.13 | 0.0898953 |  -547.872 |        -563.109 |             -0.011236 |          22963.4 |
| weekly_avg_k4       |     89 |       2135 | 2363.54 | 3068.08 | 0.102925  | -1662.83  |       -1926.56  |              1.40449  |          22962.7 |