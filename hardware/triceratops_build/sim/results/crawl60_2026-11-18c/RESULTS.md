# Crawl on the 60 cm models (simulation)

Rules: `PREREG_CRAWL60.md` with amendments 1 and 2 (C1-C6 of `PREREG_CRAWL.md`, thresholds unchanged; gait parameters and the planner's grid scaled, no tuning; model revision 3). Judged seeds 5 and 6; seeds 1-4 reported. Margin on the ankle-pin basis.

Parameters: {"stride_m": 0.096, "lift_m": 0.032, "shift_s": 0.8944, "swing_s": 0.4472, "margin_plan_m": 0.052, "body_y_max_m": 0.08, "body_dx_max_m": 0.064, "cost_y_per_m": 3.0, "lift_frac": 0.2, "stance_press_m": 0.0016, "cycles": 8, "stand_s": 1.0, "control_dt": 0.01, "grid_dx_m": 0.008, "grid_dy_m": 0.0032}

| Model | Contact | Seed | μ | C1 | C2 | C3 | C4 | C5 | C6 | Min margin mm | Max pin motion mm | Travel m | Max tilt ° | Speed m/s |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| trike60_C_closed | as built (judged) | 5 | 0.709 | pass | pass | pass | **FAIL** | **FAIL** | pass | 50.4 | 19.6 | 0.834 | 0.66 | 0.0179 |
| trike60_C_closed | as built (judged) | 6 | 0.649 | pass | pass | pass | **FAIL** | **FAIL** | pass | 50.4 | 19.8 | 0.831 | 0.66 | 0.0179 |
| trike60_C_closed | as built (reported) | 1 | 0.712 | pass | pass | pass | **FAIL** | **FAIL** | pass | 50.4 | 19.6 | 0.834 | 0.66 | 0.0179 |
| trike60_C_closed | as built (reported) | 2 | 0.804 | pass | pass | pass | **FAIL** | **FAIL** | pass | 50.4 | 19.3 | 0.838 | 0.66 | 0.0180 |
| trike60_C_closed | as built (reported) | 3 | 0.989 | pass | pass | pass | **FAIL** | **FAIL** | pass | 50.4 | 19.3 | 0.839 | 0.67 | 0.0180 |
| trike60_C_closed | as built (reported) | 4 | 0.977 | pass | pass | pass | **FAIL** | **FAIL** | pass | 50.4 | 19.3 | 0.839 | 0.67 | 0.0180 |
| trike60_C_closed_ankleroll | as built (judged) | 5 | 0.948 | pass | pass | pass | pass | pass | pass | 51.0 | 7.4 | 0.839 | 1.10 | 0.0180 |
| trike60_C_closed_ankleroll | as built (judged) | 6 | 0.816 | pass | pass | pass | pass | pass | pass | 51.0 | 6.9 | 0.834 | 1.08 | 0.0179 |
| trike60_C_closed_ankleroll | as built (reported) | 1 | 0.890 | pass | pass | pass | pass | pass | pass | 51.0 | 7.2 | 0.838 | 1.09 | 0.0180 |
| trike60_C_closed_ankleroll | as built (reported) | 2 | 0.970 | pass | pass | pass | pass | pass | pass | 51.0 | 7.4 | 0.839 | 1.10 | 0.0180 |
| trike60_C_closed_ankleroll | as built (reported) | 3 | 0.834 | pass | pass | pass | pass | pass | pass | 51.0 | 7.0 | 0.835 | 1.08 | 0.0180 |
| trike60_C_closed_ankleroll | as built (reported) | 4 | 0.797 | pass | pass | pass | pass | pass | pass | 51.0 | 6.8 | 0.834 | 1.07 | 0.0179 |
| trike60_C_closed | impratio 10 (reported) | 5 | 0.709 | pass | pass | pass | **FAIL** | **FAIL** | pass | 49.6 | 19.8 | 0.837 | 0.90 | 0.0180 |
| trike60_C_closed | impratio 10 (reported) | 6 | 0.649 | pass | pass | pass | **FAIL** | **FAIL** | pass | 49.9 | 19.7 | 0.833 | 0.90 | 0.0179 |
| trike60_C_closed_ankleroll | impratio 10 (reported) | 5 | 0.948 | pass | pass | pass | **FAIL** | **FAIL** | pass | 48.9 | 10.8 | 0.853 | 1.35 | 0.0183 |
| trike60_C_closed_ankleroll | impratio 10 (reported) | 6 | 0.816 | pass | pass | pass | **FAIL** | pass | pass | 49.7 | 8.2 | 0.836 | 1.19 | 0.0180 |

Verdicts (judged seeds 5 and 6, as built):

- trike60_C_closed: **FAIL**
- trike60_C_closed_ankleroll: **PASS**

Sensitivity, impratio 10, seeds 5 and 6 (reported, not judged):

- trike60_C_closed: rules fail
- trike60_C_closed_ankleroll: rules fail
