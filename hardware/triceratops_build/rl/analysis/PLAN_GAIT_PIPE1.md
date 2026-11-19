# Analysis plan: the gait of the trike60-pipe1 policy (written before the analysis is run)

Written 19 Nov 2026, before `gait_pipe1.py` was run on the policy. One trained policy (one seed), so nothing here is a
verdict ("a verdict needs two seeds or two controllers"); every number is reported, and the flags below say what
would count as a problem for the next design or training step. Any measure or flag changed after a result will be
labelled "changed after a result".

**Disclosure.** Before writing this I had seen the run's own evaluation: forward speeds 0.003 / 0.099 / 0.191 / 0.239 /
0.257 m/s at commands 0 / 0.10 / 0.20 / 0.25 / 0.27 m/s, no falls (`RUNS.md`). I had not measured slip, contacts,
torques or power for this policy. An earlier local 300k-step probe of the same setup showed heavy sole slip.

## What is analysed

- **Policy:** the run's final checkpoint, `trike60_2007040_steps.zip` (SHA-256 dc06d301...5b69) with
  `trike60_vecnormalize_2007040_steps.pkl` (21b44428...3984), downloaded from the owner's Drive. Loaded without running
  stored code: the policy weights with `torch.load(weights_only=True)` into a freshly built policy of the run's shape
  (ELU, 256-256); the normalisation statistics through an unpickler restricted to numpy, gymnasium's Box and SB3's
  RunningMeanStd / VecNormalize names (the last mapped to a plain container).
- **Environment:** `rl/trike_env.py` and `sim/generated/trike60_C_closed.xml`, both unchanged since the run's commit
  6aa3a679; the script checks the environment configuration equals the run record's and the observation width equals
  the policy's input width.
- **Episodes:** deterministic actions; seeds 100000-100004 (as `evaluate.py`); randomisation and noise on, as in
  training; servo delay fixed at 10 ms; 20 s each. Commands 0.25 m/s (the target) and 0.15 m/s.
- **Sensitivity (reported):** the same 0.25 m/s episodes (a) with `impratio="10"`; (b) on the roll-ankle model
  `trike60_C_closed_ankleroll.xml` (the policy never saw it; same observation layout); (c) with randomisation and noise
  off (nominal model).
- **Window:** measures use t >= 2 s of each episode, to leave out the start.

## Measures

1. **Speed:** mean forward speed; sideways drift and heading change over the episode.
2. **Footfalls:** each foot's duty factor (fraction of time loaded above the environment's 2 N contact threshold);
   the fraction of time with 4, 3, 2 (split into diagonal pairs, same-side pairs, front pair, hind pair), 1 and 0 feet
   down; stride frequency from each foot's touchdowns.
3. **Slip:** while a foot is loaded, its sole's horizontal speed: mean and 95th percentile; the **slip ratio**, total
   sole travel while loaded divided by the body's forward travel, per foot.
4. **Drive loads, by axis group** (roll, pitch, front elbow X6, hind knee X4), at every 1 ms physics step: RMS over the
   window against the thermal-equilibrium torque at the joint (7.5 / 7.5 / 10.8 / 6.75 N·m); peak against the model
   limit (21 / 21 / 37.8 / 18.9 N·m); fraction of steps at >= 99 % of the limit.
5. **Joint speeds:** peak rpm per group against the budget's allowance (X4 83 rpm; X6 108 rpm).
6. **Power:** mean of sum |torque × joint speed| over the 12 leg drives (W); cost of transport = power / (mass × g ×
   speed).
7. **Body motion:** roll and pitch RMS and peak (degrees); chassis height range.

## Flags (reported against the next step, not a pass or fail)

- **Heat:** any group's RMS torque above its thermal-equilibrium torque (and the hind knee's ratio, the budget's F12
  estimate being 1.00).
- **Saturation:** any group at >= 99 % of its limit in more than 1 % of steps.
- **Speed:** any group's peak above its allowance.
- **Slip:** any foot's slip ratio above 0.25 (its loaded sole travels more than a quarter of the body's distance).
- **Contact sensitivity:** forward speed at 0.25 m/s with impratio 10 more than 20 % below the default contact's, or
  any fall.

## Outputs

`gait_pipe1.py` writes `results/gait_pipe1_<date>.json` and `.md` in this folder, plus two labelled stills (under
500 KB each) of one 0.25 m/s episode; a clip, if made, goes on the clips data branch (D11).

## Note after the independent review (19 Nov 2026; changed after a result)

The first analysis measured the time at the torque limit against each episode's randomised limit, not the model's
nominal limit this plan names; both are now reported and the flag uses the nominal one. Also added after the first
results: per-joint heat, heading change and sideways drift (planned; drift had been computed but not reported), the
Markdown output, and a stance-and-frequency diagnostic (`gait_freq_pipe1.py`). Details: `REPORT_GAIT_PIPE1.md`, last
section.
