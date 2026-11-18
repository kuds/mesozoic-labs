# Pre-registration: four-foot standing check (task 3)

Written 18 Nov 2026, **before** `check_stand.py` was run on any build model. Committed with the models and the script
in the same commit. Any rule changed after a result will be labelled "changed after a result" here and in the report.

**Disclosure.** Before writing these rules I had seen the generator's own printout (`build_model.py`): the total and
category masses, the centre of mass, and a lever estimate of the front share, (x_com + 325 mm) / 650 mm: T20 55.6 %
(listed) and 55.3 % (closed); T15 54.2 % and 54.1 %. I had not run a simulation of these models. Rule R4b's band was set
from the parts list's own sensitivity (moving the battery 100 mm shifts the share by 0.8-1.1 points), not from those
numbers.

## What is run

- Models: `generated/trike_{T20,T15}_{listed,closed}.xml`, built by `build_model.py` at this commit, mujoco 3.10.0.
- Seeds: 0 (exact neutral pose; reported, not judged), 1 and 2 (judged). For seeds 1 and 2 every hinge joint starts
  with a uniform random offset in ±0.02 rad (`numpy.random.default_rng(seed)`), and the base starts 5 mm higher.
- Command: every position servo held at its neutral target (zero), for 10 s at the model's 1 ms step.
- Settled window: the last 1 s (9-10 s). Values in the window are means.
- Budget: 12 runs of 10 s simulated time, about a minute of CPU. Stop rule: none needed; if the script errors, fix the
  script and rerun all runs.

## Rules

| Rule | What | Pass if |
|---|---|---|
| R1 | Build | Model compiles; total mass equals the parts list's total (listed T20 18.83 kg, T15 14.61 kg; closed 20.00 / 15.00 kg) within ±0.01 kg; for the listed variant each section 6.2 category within ±0.01 kg |
| R2 | Stays standing | Over all 10 s: root height ≥ 0.90 × start, trunk tilt ≤ 5°, no NaN, no MuJoCo warning; in the window, root speed RMS < 5 mm/s |
| R3 | Four loaded feet | In the window, every foot carries ≥ 10 % of the weight, and the four sum to the weight within ±2 % |
| R4a | Front share in the design range | Front feet carry 50-60 % of the weight (the parts list's design point 55 % with its 50 % and 60 % sensitivity cases) |
| R4b | Agreement with the parts list's as-built estimate (flag, not a fail) | Within ±1.0 point of the parts list's estimate: T20 55.2 %, T15 53.6 % (`massbudget.py`, battery at -200 mm) |
| R4c | Left-right balance | Left feet carry 50 ± 2 % of the weight |
| R5a | Continuous load (fail rule) | Every leg joint's settled torque ≤ its drive's thermal-equilibrium torque at the joint (X4 7.5 N·m; X6 12 N·m, the X6-50-L table as stand-in; X8 34 N·m; flexion values × ratio × 0.90) |
| R5b | Against the parts list's standing figures (flag, not a fail) | Closed variant: front flexion ≤ 1.15 × the parts list's four-foot standing figure (T20 8.1 N·m, T15 6.1 N·m) and roll ≤ 1.15 × (4.3, 3.2 N·m). Listed variant: the same figures scaled by mass (18.83 / 20, 14.61 / 15). Hind flexion and pitch have no parts-list standing figure: reported only |
| R6 | No saturation | In the window no actuator is at ≥ 99 % of its force limit |
| R7 | Head and tail servos | In the window the neck pitch torque ≤ 3.11 N·m and the tail pitch torque (T20) ≤ 1.20 N·m (the servos' rated torques) |

**Verdict per build and variant:** PASS if R1, R2, R3, R4a, R4c, R5a, R6 and R7 hold for both judged seeds (1 and 2).
R4b and R5b are flags: a flag is reported and explained, it does not fail the build. Two seeds give the verdict; seed 0
is reported for reference.

## What this check does not cover

Walking, trotting, the held paw pose and the parts list's peak torques (tasks 4, 5, 6 and 7); speed limits; heat over
time; delay and noise. Standing torques are simulation results of a model whose part masses and positions are
estimates; nothing here is measured.
