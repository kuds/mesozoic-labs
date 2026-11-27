# Running averages for r_heat_peak compared (steer_heat_average.py)

72 episodes, 0 falls. Four D22 128 mm policies x 3 seeds per group. Cost at weight 1, from 2 s; 'warm' is adopted (trike_env.heat_peak_update).

## Cost over 2-20 s as a share of F2's own price

| command, noise | F2 (12 episodes) | F2 price per step | bc | floor50 | floor100 | floor200 | warm | cold | bc_price4 |
|---|---|---|---|---|---|---|---|---|---|
| 0.15 m/s, std 0.0 | 0.72-0.89 | 0.015 | 2.35 | 1.75 | 1.40 | 1.14 | 1.26 | 0.43 | 1.34 |
| 0.15 m/s, std 0.1 | 0.94-1.08 | 0.214 | 1.09 | 1.01 | 0.95 | 0.86 | 0.82 | 0.59 | 0.91 |
| 0.15 m/s, std 0.2 | 1.14-1.30 | 0.415 | 1.04 | 0.99 | 0.95 | 0.88 | 0.82 | 0.69 | 0.90 |
| 0.25 m/s, std 0.0 | 0.81-1.00 | 0.119 | 1.04 | 0.96 | 0.91 | 0.83 | 0.80 | 0.51 | 0.89 |
| 0.25 m/s, std 0.1 | 0.97-1.08 | 0.230 | 1.07 | 1.00 | 0.95 | 0.87 | 0.82 | 0.61 | 0.91 |
| 0.25 m/s, std 0.2 | 1.14-1.28 | 0.403 | 1.04 | 0.99 | 0.95 | 0.89 | 0.82 | 0.69 | 0.91 |

## Share of the cost in 2-4 s (11 % of the window's time)

| noise | bc | floor50 | floor100 | floor200 | warm | cold | bc_price4 |
|---|---|---|---|---|---|---|---|
| std 0.0 | 0.21 | 0.10 | 0.03 | 0.00 | 0.05 | 0.00 | 0.00 |
| std 0.1 | 0.16 | 0.09 | 0.04 | 0.00 | 0.04 | 0.00 | 0.00 |
| std 0.2 | 0.13 | 0.08 | 0.05 | 0.01 | 0.04 | 0.00 | 0.00 |

## Episodes whose F2 is under the margin (8 episodes): cost per step over all 1000 steps (ideally 0), and the share of it in 2-4 s

| measure | bc | floor50 | floor100 | floor200 | warm | cold | bc_price4 |
|---|---|---|---|---|---|---|---|
| cost per step | 0.0179 | 0.0093 | 0.0043 | 0.0024 | 0.0049 | 0.0003 | 0.0037 |
| share in 2-4 s | 0.79 | 0.60 | 0.13 | 0.00 | 0.48 | 0.00 | 0.00 |

## Extra cost of a 0.5 s burst at 2x rating (25 steps at (torque / rating)^2 = 4), mean over the episodes, at 3 s and at 10 s

| average | drive | noise | at 3 s | at 10 s | ratio |
|---|---|---|---|---|---|
| bc | hottest | 0.0 | 61.9 | 34.3 | 1.80 |
| bc | hottest | 0.1 | 59.7 | 32.0 | 1.87 |
| bc | hottest | 0.2 | 39.3 | 21.8 | 1.80 |
| bc | 2nd hottest | 0.0 | 48.8 | 20.7 | 2.36 |
| bc | 2nd hottest | 0.1 | 32.3 | 17.6 | 1.83 |
| bc | 2nd hottest | 0.2 | 25.4 | 10.7 | 2.37 |
| floor50 | hottest | 0.0 | 61.9 | 34.3 | 1.80 |
| floor50 | hottest | 0.1 | 59.7 | 32.0 | 1.87 |
| floor50 | hottest | 0.2 | 39.3 | 21.8 | 1.80 |
| floor50 | 2nd hottest | 0.0 | 48.8 | 20.7 | 2.36 |
| floor50 | 2nd hottest | 0.1 | 32.3 | 17.6 | 1.83 |
| floor50 | 2nd hottest | 0.2 | 25.4 | 10.7 | 2.37 |
| floor100 | hottest | 0.0 | 58.6 | 34.3 | 1.71 |
| floor100 | hottest | 0.1 | 58.1 | 32.0 | 1.81 |
| floor100 | hottest | 0.2 | 38.2 | 21.8 | 1.76 |
| floor100 | 2nd hottest | 0.0 | 45.7 | 20.7 | 2.21 |
| floor100 | 2nd hottest | 0.1 | 31.1 | 17.6 | 1.76 |
| floor100 | 2nd hottest | 0.2 | 24.6 | 10.7 | 2.30 |
| floor200 | hottest | 0.0 | 43.5 | 34.3 | 1.27 |
| floor200 | hottest | 0.1 | 49.9 | 32.0 | 1.56 |
| floor200 | hottest | 0.2 | 34.1 | 21.8 | 1.57 |
| floor200 | 2nd hottest | 0.0 | 31.0 | 20.7 | 1.50 |
| floor200 | 2nd hottest | 0.1 | 25.0 | 17.6 | 1.42 |
| floor200 | 2nd hottest | 0.2 | 21.3 | 10.7 | 1.99 |
| warm | hottest | 0.0 | 36.4 | 32.1 | 1.13 |
| warm | hottest | 0.1 | 37.4 | 30.2 | 1.24 |
| warm | hottest | 0.2 | 25.1 | 20.6 | 1.22 |
| warm | 2nd hottest | 0.0 | 26.3 | 19.2 | 1.37 |
| warm | 2nd hottest | 0.1 | 18.2 | 16.6 | 1.10 |
| warm | 2nd hottest | 0.2 | 14.9 | 10.1 | 1.48 |
| cold | hottest | 0.0 | 13.1 | 30.0 | 0.44 |
| cold | hottest | 0.1 | 30.9 | 30.9 | 1.00 |
| cold | hottest | 0.2 | 24.3 | 21.0 | 1.16 |
| cold | 2nd hottest | 0.0 | 4.9 | 17.2 | 0.29 |
| cold | 2nd hottest | 0.1 | 10.9 | 16.9 | 0.64 |
| cold | 2nd hottest | 0.2 | 13.5 | 10.3 | 1.31 |
| bc_price4 | hottest | 0.0 | 43.8 | 34.3 | 1.27 |
| bc_price4 | hottest | 0.1 | 43.0 | 32.0 | 1.34 |
| bc_price4 | hottest | 0.2 | 28.0 | 21.8 | 1.29 |
| bc_price4 | 2nd hottest | 0.0 | 31.5 | 20.7 | 1.52 |
| bc_price4 | 2nd hottest | 0.1 | 20.0 | 17.6 | 1.13 |
| bc_price4 | 2nd hottest | 0.2 | 16.0 | 10.7 | 1.49 |

## Discounted weight (gamma 0.99) of one step's heat in the average, by time (hinge and maximum over drives left out)

| average | 2.02 s | 2.5 s | 3 s | 4 s | 6 s | 10 s | 15 s |
|---|---|---|---|---|---|---|---|
| bc | 4.43 | 1.28 | 0.91 | 0.64 | 0.47 | 0.37 | 0.34 |
| floor50 | 1.22 | 1.10 | 0.91 | 0.64 | 0.47 | 0.37 | 0.34 |
| floor100 | 0.80 | 0.78 | 0.75 | 0.64 | 0.47 | 0.37 | 0.34 |
| floor200 | 0.53 | 0.52 | 0.52 | 0.51 | 0.47 | 0.37 | 0.34 |
| warm | 0.33 | 0.33 | 0.33 | 0.33 | 0.33 | 0.33 | 0.33 |
| cold | 0.33 | 0.33 | 0.33 | 0.33 | 0.33 | 0.33 | 0.33 |
| bc_price4 | 0.14 | 0.20 | 0.30 | 0.63 | 0.47 | 0.37 | 0.34 |
