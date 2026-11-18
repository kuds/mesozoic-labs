# Crawl on the 60 cm models (simulation)

Rules: `PREREG_CRAWL60.md` (C1-C6 of `PREREG_CRAWL.md`, thresholds unchanged; gait parameters scaled, no tuning). Seeds 1 and 2. Margin on the ankle-pin basis.

Parameters: {"stride_m": 0.096, "lift_m": 0.032, "shift_s": 0.8944, "swing_s": 0.4472, "margin_plan_m": 0.052, "body_y_max_m": 0.08, "body_dx_max_m": 0.064, "cost_y_per_m": 3.0, "lift_frac": 0.2, "stance_press_m": 0.0016, "cycles": 8, "stand_s": 1.0, "control_dt": 0.01}

| Model | Contact | Seed | μ | C1 | C2 | C3 | C4 | C5 | C6 | Min margin mm | Max pin motion mm | Travel m | Max tilt ° | Speed m/s |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| trike60_C_closed | as built (judged) | 1 | 0.712 | pass | pass | pass | **FAIL** | **FAIL** | pass | 50.5 | 19.6 | 0.718 | 0.99 | 0.0154 |
| trike60_C_closed | as built (judged) | 2 | 0.804 | pass | pass | pass | **FAIL** | **FAIL** | pass | 50.5 | 19.7 | 0.718 | 0.98 | 0.0154 |
| trike60_C_closed_ankleroll | as built (judged) | 1 | 0.890 | pass | pass | pass | **FAIL** | pass | pass | 51.0 | 8.4 | 0.717 | 1.13 | 0.0154 |
| trike60_C_closed_ankleroll | as built (judged) | 2 | 0.970 | pass | pass | pass | **FAIL** | pass | pass | 51.0 | 9.3 | 0.717 | 1.14 | 0.0154 |
| trike60_C_closed | impratio 10 (reported) | 1 | 0.712 | pass | pass | pass | **FAIL** | **FAIL** | pass | 49.4 | 20.7 | 0.725 | 1.14 | 0.0156 |
| trike60_C_closed | impratio 10 (reported) | 2 | 0.804 | pass | pass | pass | **FAIL** | **FAIL** | pass | 48.9 | 20.9 | 0.729 | 1.09 | 0.0157 |
| trike60_C_closed_ankleroll | impratio 10 (reported) | 1 | 0.890 | pass | pass | pass | **FAIL** | **FAIL** | pass | 51.2 | 10.5 | 0.730 | 1.41 | 0.0157 |
| trike60_C_closed_ankleroll | impratio 10 (reported) | 2 | 0.970 | pass | pass | pass | **FAIL** | **FAIL** | pass | 51.3 | 10.5 | 0.733 | 1.54 | 0.0158 |

Verdicts (judged, as built):

- trike60_C_closed: **FAIL**
- trike60_C_closed_ankleroll: **FAIL**

Sensitivity, impratio 10 (reported, not judged):

- trike60_C_closed: rules fail
- trike60_C_closed_ankleroll: rules fail
