# Quality skeptic: is the PHASE stance a quality stance, and a usable walking parent?

**Bottom line**
- What holds in the framing: all three PHASE finals are clean stances at about statue level. Gate 40/40 at every heading, no habit, no actuator near its limit, and more robust to command noise than the do-nothing statue.
- What is overstated: "meets our criteria" is a floor that the zero-action statue also clears, with similar margins. The tests that would show quality beyond the statue (M4 K1-K4, M2 posture) were not run. Nothing is certified, and no parent is named.
- No finding blocks "the stance behaviour works". Three things block naming a parent until they are closed: PHASE s60 falls about 1% of the time, PHASE s80's tail_2 command is still rising at the final checkpoint, and the maintainer has not ruled on head height (the heads sit at the reward's 0.90 m edge).
- Low action noise is a design item for the walking pilot, not a stance defect. Closing everything above needs CPU only.

OUT = `ST/final/stance_solved/skeptic`; my scripts (`gate_margins`, `m1_actuators`, `ledger_probe`, `.py`/`.out`) read only existing outputs.

## Concerns, rated

**1. Head posture sits at the reward's edge. MINOR for "solved"; needs a ruling.**
- Mean head tip over the last 8 evaluations: 0.913 / 0.904 / 0.902 m (s60/s42/s80) [measured: W/head_neck.out].
- Episodes below 0.90 m: 17/80, 26/80 and 40/80. Lowest episode mean: 0.890 m [measured: OUT/ledger_probe.out].
- The heads drifted down during training and stopped near 0.90: from 0.987 / 0.978 / 0.975 at 131,072 steps to 0.905 / 0.902 / 0.908 [measured: same].
- The statue's head is at 0.958 m [measured: BR `stance.toml` l.316-317 comment]. The reward gives no credit at 0.75 m and full credit from 0.90 m, with nothing extra above that. The curve is flat at 0.90, so dipping slightly below costs almost nothing; the policies collect 349.53-349.89 of 350 [measured: BR `stance.toml` l.315-318; `reward_functions.py` l.648-668; M3 JSON `reward_head_clearance`].
- Neck pitch is held 2.7-4.0° and head pitch 2.6-5.9° below home [measured: OUT/gate_margins.out, `dc_deg`].
- So the policy settles where the reward stops paying [inferred]. No gate reads head posture, and main's own notes say to "check a stance's head by hand before using it" [measured: MAIN `docs/KNOWN_ISSUES.md` l.1324-1347].
- Walking risk: main's locomotion config sets no head-clearance weight, so the default of 0 applies [measured: MAIN `locomotion.toml`; `base_env.py` l.254]. In walking, nothing would hold the head up [inferred].
- To settle: M2 posture at the finals (CPU), then a maintainer ruling.

**2. PHASE s60 falls. BLOCKS "parent for walking" for s60 only; MINOR for "solved".**
- 9 of its 910 training-evaluation episodes fell; every one is recorded with cause `fallen`. Training evaluations use the deterministic policy [measured: OUT/ledger_probe.out; BR `train_base.py` l.872].
- One fall is in the last 80 episodes (at 11,665,408 steps).
- Head minimums: 0.44-0.61 m in four falls, 1.17-1.61 m in three; both directions, perhaps [inferred].
- Under the in-run joint-noise test (K3 panel), s60 fell 7/240, against 2/240 for s42 and 1/240 for s80. At the final checkpoint s60 fell 1/20 [measured: same, from `k3_panel.csv`].
- The final passed 40/40 (M3) and showed no fall flags on 20 seeds (M1) [measured: `ST/f2/measure/stance7_r10LTSBpc_s60/m3_final.out`, `m1_final.out`].
- The 40-seed panel cannot see a rate this low. At 9/910, zero falls in 40 still has a chance of about 0.67 [inferred: my arithmetic].
- To settle: M4 at the finals, plus a larger panel such as the 280-seed panel the statue was given [RUNLOG l.594]. CPU only.

**3. PHASE s80's tail_2 command is still rising. MINOR; counts against s80 as a parent.**
- In the ledger it rose from +0.014 to +0.055 [measured: SCORE l.332].
- Deterministic replays (M1) at the judged checkpoints confirm the rise is real and speeding up: +0.0104, +0.0149, +0.0130, +0.0190, +0.0354, +0.0658 (final) [measured: `ST/f2/measure/stance7_r10LTSBpc_s80/m1_*.out`, `cmd_a` "13"].
- Still small: maximum |command| 0.085, force use 0.011, far below the 0.60 park bar [measured: OUT/m1_actuators.out]. Where it ends needs GPU; not worth it: choose another parent.

**4. Action noise is low at the finals. MINOR for stance; a design item for the walking pilot.**
- Mean action noise across the 15 actuators (std) is 0.0597 / 0.0569 / 0.0585. Training started at 0.2231 [measured: OUT/ledger_probe.out, `policy_probe.npz` `std_per_actuator`].
- The leg pitch, knee and ankle joints are near 0.007-0.013, and the entropy bonus has decayed to 0 (`ent_coef` 0) [measured: same].
- Walking pilot 1's parent had 0.095, and that pilot read STILL. Its own noise rose to 0.108-0.111 during the pilot [measured: MAIN `BEHAVIOR_RECIPES_PLAN.md` l.1319].
- So low noise did not stop it rising; why pilot 1 stood still is unknown [inferred].
- Mitigation: pilot 2's noise reset (log-std set to −1.5 at the warm start). It was designed but never launched [measured: same; RUNLOG l.607]. Code only.

**5. Gate margins match the statue's. BLOCKS "solved" as a quality claim until M4 reads; the floor itself is met.**
- The six headings give the same values on every check to 4 significant figures, and the same rewards to 0.1. The whole-scene turn makes them one 40-seed panel run six times [measured: OUT/gate_margins.out].
  The report says so too [measured: `stance_heading_report.txt` l.11].
- Worst value as a share of each check's allowance (s60 / s42 / s80, then the zero-action statue on the same seeds):
  - settle peak floor force: 0.80 / 0.80 / 0.80; statue 0.80
  - settle stance-width change: 0.76 / 0.65 / 0.56; statue 0.68
  - yaw change: 0.66 / 0.46 / 0.53; statue 0.56
  - sole tilt: 0.56 / 0.53 / 0.62; statue 0.80
  - corner lift: 0.46 / 0.44 / 0.52; statue 0.66
  - CoP fore-aft: 0.20 / 0.42 / 0.27; statue 0.40

  [measured: OUT/gate_margins.out]
- The statue passes the same six-heading certificate 40/40 [measured: RUNLOG l.601, l.620]. The plan says the statue passing "is by design" [measured: GQP l.516].
- So "meets our criteria" means clearing a floor the statue also clears. It is a screen measurement, not certification [measured: SCORE l.357].
- To settle: M4 K1-K4 at the finals (CPU).

**6. Other actuators: NOT A CONCERN for saturation; MINOR for hip-roll dither.**
- At the finals no actuator command reaches |0.9|. The largest is hip roll at 0.257 [measured: OUT/m1_actuators.out].
- Force use is at most 0.270, in the neck and head, which carry the head's weight; OFF maxima there are 0.28-0.31 [measured: same].
- Hip roll is the most active joint, with AC rms 0.076 / 0.076 / 0.062 against 0.039-0.067 in OFF. Commands change at an effective 33.6-34.4 Hz [measured: OUT/gate_margins.out].
- Jerk and smoothness charge about −0.01 to −0.02 per episode: the dither is unpriced [measured: same]. Settle: nothing for saturation; the dither belongs to the actor-input decision for walking (code; #491 is not found in MAIN's docs).

**7. Returns, slip, stance width, CoP: MINOR.**
- At the final, mean return beats the statue by +17.2 / +17.0 / +8.9, or at most 0.4% [measured: M3 reports, 4487.6 / 4487.4 / 4479.3 against 4470.4].
- s60's mean over 910 training evaluations, 4442.7, is below the statue's 4470.37 [measured: SCORE l.324; `progress.json`]. "Higher than OFF in all three pairs" mostly measures what the habits cost OFF, all below the statue [inferred: SCORE l.337, 341, 347].
- Worst foot slide (report only) 0.019-0.022 m against the statue's 0.010 m [measured: OUT/gate_margins.out]; width and CoP within bars (item 5). Nothing to settle.

**8. Owed work: BLOCKS "parent for walking" formally.**
- No M4 record, so no parent is named [measured: SCORE l.361]; M2 and the critic readouts not run [measured: REPORT "Open items"]. GQ-17 wants a certified stance before any T. rex walk [measured: GQP l.700]. M4/M2: CPU; certification on main: GPU (Option E).

## Evidence the stance IS good

- **Gate:** 40/40 at all six headings in all three, 0 hop-or-fall, full horizon; OFF s42 failed [measured: SCORE l.321, 325, 329, 338].
- **No tail habit:** tail drive (TD) 0.008-0.017 against 0.15-0.64 for OFF; no habit elsewhere; leg pitch, knee and ankle commands at most 0.017 [measured: SCORE l.362; OUT/m1_actuators.out].
- **Clean training:** unclean evaluation episodes were 18/910, 11/910 and 7/910, against 46, 265 and 132 for OFF. In the last 80: 1, 0 and 0 [measured: OUT/ledger_probe.out].
- **Closed-loop robustness beyond the statue:**
  - With the policy's own noise: 0/240 falls in every run.
  - Under joint noise (mean sigma 0.161): 0-1/20 at the finals.
  - The statue under open-loop noise of SD 0.10 reached the full horizon in only 16/40 [measured: `k3_panel.json` spec; OUT/ledger_probe.out; RUNLOG l.585].
  - Different tests, so compare loosely; still, the policy adds balance the statue lacks [inferred].
- **Feet better than the statue:** worst sole tilt 1.07-1.25° against 1.60°; worst window drift about 0.01 m against 0.019 m [measured: OUT/gate_margins.out].
- **Head held:** no head drop by main's 0.75 m standard. The OFF runs raised it 15-31 cm [measured: W/head_neck.out].
- **Best parent on existing data: s42** (no falls, joint-noise 2/240, tail_2 steady at −0.037) [measured: OUT/ledger_probe.out, m1_actuators.out]; M4 must confirm [inferred].
