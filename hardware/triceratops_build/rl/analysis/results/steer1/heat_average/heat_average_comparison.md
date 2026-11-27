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

## Cost per step on episodes whose F2 is under the margin (8 episodes; ideally 0)

| bc | floor50 | floor100 | floor200 | warm | cold | bc_price4 |
|---|---|---|---|---|---|---|
| 0.0179 | 0.0093 | 0.0043 | 0.0024 | 0.0049 | 0.0003 | 0.0037 |

## Extra cost of a 0.5 s burst at 2x rating on the hottest drive (deterministic episodes, mean)

| average | at 3 s | at 10 s | ratio |
|---|---|---|---|
| bc | 61.9 | 34.3 | 1.80 |
| floor50 | 61.9 | 34.3 | 1.80 |
| floor100 | 58.6 | 34.3 | 1.71 |
| floor200 | 43.5 | 34.3 | 1.27 |
| warm | 36.4 | 32.1 | 1.13 |
| cold | 13.1 | 30.0 | 0.44 |
| bc_price4 | 43.8 | 34.3 | 1.27 |

## Discounted weight (gamma 0.99) of one step's heat in the estimate, by time (hinge ignored)

| average | 2.02 s | 2.5 s | 3 s | 4 s | 6 s | 10 s | 15 s |
|---|---|---|---|---|---|---|---|
| bc | 4.43 | 1.28 | 0.91 | 0.64 | 0.47 | 0.37 | 0.34 |
| floor50 | 1.22 | 1.10 | 0.91 | 0.64 | 0.47 | 0.37 | 0.34 |
| floor100 | 0.80 | 0.78 | 0.75 | 0.64 | 0.47 | 0.37 | 0.34 |
| floor200 | 0.53 | 0.52 | 0.52 | 0.51 | 0.47 | 0.37 | 0.34 |
| warm | 0.33 | 0.33 | 0.33 | 0.33 | 0.33 | 0.33 | 0.33 |
| cold | 0.33 | 0.33 | 0.33 | 0.33 | 0.33 | 0.33 | 0.33 |
| bc_price4 | 0.14 | 0.20 | 0.30 | 0.63 | 0.47 | 0.37 | 0.34 |
