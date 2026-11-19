# Analysis plan: the trike60-run2b-ankle policy (written before the analysis is run)

19 Nov 2026. One trained policy, so nothing here is a verdict; every number is reported and compared with the first
policy (trike60-pipe1). Any measure changed after a result will be labelled "changed after a result".

**Disclosure.** Before writing this I had seen the run's own training evaluations (`eval_during_training.jsonl`, one
deterministic episode per command every 250k steps) and its final `evaluate.py --sweep` on Colab: at the 0.25 m/s
command 0.244 m/s, no falls; worst joints' RMS torque / continuous rating 0.76 (left hind knee), 0.73 (right fore
pitch), 0.64 (left fore elbow); slip ratio 0.05-0.09; 22 W of joint mechanical power; about 1.2-1.6 debounced
touchdowns per foot per second.

## What is analysed

- **Policy:** the run's final checkpoint, `trike60_8003584_steps.zip` and `trike60_vecnormalize_8003584_steps.pkl`
  from the owner's Drive (run commit e92fd9ef; model `trike60_C_closed_ankleroll.xml`; 8,003,584 steps), loaded only
  through `rl/analysis/gait_pipe1.py`'s safe loaders. `rl/trike_env.py`, `rl/gait_metrics.py` and the model are
  unchanged since e92fd9ef.
- **Tools, unchanged in their definitions:** `rl/analytics/gait_report.py` (reviewed at ec178063; the only change is
  that the main cases and the speed sweep now use the run's own model, here the roll ankle, with one case on the
  other foot, PR #669's, and the cross-check with gait_pipe1's results applies only to that policy),
  `rl/robustness/sweep.py` (reviewed at d9c94c2d; it already uses the run's model) and
  `rl/analytics/training_dashboard.py`.
- **Settings:** as for pipe1: deterministic actions, servo delay 10 ms, seeds 100000-100004 for the report (200000+ for
  the sweep, 8 per point), randomisation on except the noise-off nominal case.

## Questions, and how each is read

1. **Drive heat:** does every joint stay under its continuous rating at 0.25 m/s (RMS torque / rating < 1.0, mean and
   worst episode), and how much margin does the hind knee have? This bears on F12/F13 (X4 hind knees).
2. **Torque limits:** time at the nominal limit per joint.
3. **Gait:** class, stride frequency and length, duty factors, symmetry (the first policy was asymmetric).
4. **Weight and balance:** per-foot peak loads; CoM outside the support polygon on step contact.
5. **Feet:** slip ratio, swing clearance, touchdown speed and load rate.
6. **Power and run time:** both electrical estimates (shared bus, per drive) and run time with a 30 W hotel load.
7. **Head steadiness:** bob (<= 5 Hz) and shake (5-20 Hz).
8. **Robustness:** where it breaks in each one-factor sweep, against the first policy's edges.
9. **The other foot:** how it walks on PR #669's foot, which it never saw.

Flags (reported against the next step, not pass or fail): any joint's RMS / rating above 1.0; any joint at its
nominal limit in more than 1 % of steps; any foot's slip ratio above 0.25; CoM outside the step-contact polygon more
than 5 % of the time; any fall in the requested robustness ranges.

## Outputs

`rl/analytics/results/run2b/`, `rl/robustness/results/run2b/`, the training dashboard comparing both runs, and
`rl/analysis/REPORT_RUN2B.md` with the comparison; then an independent check.
