# Crawl on the 60 cm models (simulation)

Rules: `PREREG_CRAWL60.md` with amendment 1 (C1-C6 of `PREREG_CRAWL.md`, thresholds unchanged; gait parameters and the planner's grid scaled, no tuning; model revision 2). Judged seeds 3 and 4; seeds 1 and 2 reported. Margin on the ankle-pin basis.

Parameters: {"stride_m": 0.096, "lift_m": 0.032, "shift_s": 0.8944, "swing_s": 0.4472, "margin_plan_m": 0.052, "body_y_max_m": 0.08, "body_dx_max_m": 0.064, "cost_y_per_m": 3.0, "lift_frac": 0.2, "stance_press_m": 0.0016, "cycles": 8, "stand_s": 1.0, "control_dt": 0.01, "grid_dx_m": 0.008, "grid_dy_m": 0.0032}

| Model | Contact | Seed | μ | C1 | C2 | C3 | C4 | C5 | C6 | Min margin mm | Max pin motion mm | Travel m | Max tilt ° | Speed m/s |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| trike60_C_closed | as built (judged) | 3 | 0.989 | pass | pass | pass | **FAIL** | **FAIL** | pass | 50.4 | 19.2 | 0.839 | 0.53 | 0.0180 |
| trike60_C_closed | as built (judged) | 4 | 0.977 | pass | pass | pass | **FAIL** | **FAIL** | pass | 50.4 | 19.2 | 0.839 | 0.53 | 0.0180 |
| trike60_C_closed | as built (reported) | 1 | 0.712 | pass | pass | pass | **FAIL** | **FAIL** | pass | 50.5 | 19.8 | 0.831 | 0.52 | 0.0179 |
| trike60_C_closed | as built (reported) | 2 | 0.804 | pass | pass | pass | **FAIL** | **FAIL** | pass | 50.5 | 19.7 | 0.835 | 0.52 | 0.0180 |
| trike60_C_closed_ankleroll | as built (judged) | 3 | 0.834 | pass | pass | pass | pass | pass | pass | 51.5 | 6.2 | 0.834 | 0.97 | 0.0179 |
| trike60_C_closed_ankleroll | as built (judged) | 4 | 0.797 | pass | pass | pass | pass | pass | pass | 51.5 | 5.9 | 0.833 | 0.96 | 0.0179 |
| trike60_C_closed_ankleroll | as built (reported) | 1 | 0.890 | pass | pass | pass | pass | pass | pass | 51.5 | 6.5 | 0.837 | 1.00 | 0.0180 |
| trike60_C_closed_ankleroll | as built (reported) | 2 | 0.970 | pass | pass | pass | **FAIL** | pass | pass | 51.5 | 6.8 | 0.841 | 1.01 | 0.0181 |
| trike60_C_closed | impratio 10 (reported) | 3 | 0.989 | pass | pass | pass | **FAIL** | **FAIL** | pass | 49.2 | 20.1 | 0.848 | 0.96 | 0.0182 |
| trike60_C_closed | impratio 10 (reported) | 4 | 0.977 | pass | pass | pass | **FAIL** | **FAIL** | pass | 48.9 | 20.1 | 0.848 | 0.94 | 0.0182 |
| trike60_C_closed | impratio 10 (reported) | 1 | 0.712 | pass | pass | pass | **FAIL** | **FAIL** | pass | 49.8 | 19.7 | 0.830 | 0.73 | 0.0178 |
| trike60_C_closed | impratio 10 (reported) | 2 | 0.804 | pass | pass | pass | **FAIL** | **FAIL** | pass | 49.8 | 19.9 | 0.839 | 0.82 | 0.0180 |
| trike60_C_closed_ankleroll | impratio 10 (reported) | 3 | 0.834 | pass | pass | pass | **FAIL** | pass | pass | 49.9 | 7.7 | 0.836 | 1.06 | 0.0180 |
| trike60_C_closed_ankleroll | impratio 10 (reported) | 4 | 0.797 | pass | pass | pass | **FAIL** | pass | pass | 49.7 | 7.1 | 0.833 | 1.05 | 0.0179 |
| trike60_C_closed_ankleroll | impratio 10 (reported) | 1 | 0.890 | pass | pass | pass | **FAIL** | pass | pass | 49.5 | 9.7 | 0.840 | 1.09 | 0.0181 |
| trike60_C_closed_ankleroll | impratio 10 (reported) | 2 | 0.970 | pass | pass | pass | **FAIL** | **FAIL** | pass | 49.9 | 10.3 | 0.853 | 1.19 | 0.0183 |

Verdicts (judged seeds 3 and 4, as built):

- trike60_C_closed: **FAIL**
- trike60_C_closed_ankleroll: **PASS**

Sensitivity, impratio 10, seeds 3 and 4 (reported, not judged):

- trike60_C_closed: rules fail
- trike60_C_closed_ankleroll: rules fail
