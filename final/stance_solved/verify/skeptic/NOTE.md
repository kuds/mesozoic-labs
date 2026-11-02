# Verifier note: the "quality skeptic" evidence note

**Bottom line**
- The skeptic note's numbers are accurate. I opened every cited source I could reach and found no wrong figure.
- Three claims are wrong. (a) "The maintainer has not ruled on head height": the maintainer ruled on 2026-10-25 (D-D46) that head posture is report-only, with no gate change. (b) "Issue #491 is not found in MAIN's docs": four MAIN docs cite it. (c) "In walking, nothing would hold the head up": main's walking config ends an episode on head contact and penalises nose-diving.
- The bottom line overstates which items block a parent. The only formal block is the missing M4 record. s60's fall rate is within the gate's own tolerance, and the head item is report-only by the maintainer's decision.
- It overstates "more robust to noise than the statue". That comes from two unmatched tests. The knee and ankle noise in the policy's test is below the statue test's 0.10, and the own-noise test is almost noise-free.
- The note misses the most concrete "not yet a walking parent" fact: as built, the code refuses to warm-start any stage from a PHASE checkpoint when that stage declares no phase input, and walking declares none. The fix is code only (Option B).

OUT = `/home/user/work/pc_study/final/stance_solved/verify/skeptic`. I read existing outputs only. I wrote two small read-only probes, `fall_seeds.py` and a threshold read, each with its `.out`. I read GQP and BRP with `git show` and kept no copies.

## What I confirmed (numbers match their sources)

- **Gate and habits:**
  - The three PHASE finals pass M3 40/40 at every heading, with full horizon 1.0000 [measured: `round/panels/*/final/m3/stance_heading_report.txt` l.15-22; SCORE l.321, 325, 329].
  - Tail class CLEAR, no habit on any other motor. TD 0.0079 / 0.0173 / 0.0122 against OFF's 0.388 / 0.635 / 0.154 [measured: SCORE l.321-329, l.362].
- **Head (last 8 evaluations):**
  - Mean head tip 0.913 / 0.904 / 0.902 m [measured: W/head_neck.out].
  - Episodes below 0.90 m: 17/80, 26/80 and 40/80. Lowest episode mean 0.890 m [measured: skeptic `ledger_probe.out`, recomputed from the F2 ledgers].
- **Reward shape:**
  - Head credit is a smoothstep from 0.75 to 0.90 m, flat above 0.90 [measured: BR `reward_functions.py` l.648-668; BR `stance.toml` l.315-318, where the statue's head is 0.958 m].
  - Head-clearance reward is 349.73 / 349.89 / 349.53 per episode [measured: skeptic `gate_margins.out`, heading 0].
- **PHASE s60 falls:**
  - Training evaluations are deterministic [measured: BR `train_base.py` l.872 `deterministic=True`].
  - 9 short episodes in 910, all with cause `fallen`; one of them in the last 80 [measured: W/short_eps.out; my `fall_seeds.out`].
  - K3 joint-noise falls 7/240, against 2/240 (s42) and 1/240 (s80). At the final: 1/20. The final also passed M3 40/40 and M1 with 0 fall flags [measured: skeptic `ledger_probe.out`; `f2/measure/stance7_r10LTSBpc_s60/m1_final.out`, `m3_final.out`].
  - The chance of 0 falls in 40 at 9/910 is 0.672 [my arithmetic].
- **PHASE s80 tail_2 command:**
  - Rises in the ledger from +0.0142 (4,194,304 steps) to +0.0552 (11,927,552) [measured: SCORE l.332, guard field].
  - M1 replays: +0.0104, +0.0149, +0.0130, +0.0190, +0.0354, +0.0658 [measured: `f2/measure/stance7_r10LTSBpc_s80/m1_*.out`, `cmd_a` "13"].
- **Action noise:**
  - Mean std 0.0597 / 0.0569 / 0.0585, down from 0.2231. Leg joints 0.007-0.013. ent_coef 0 at the end [measured: skeptic `ledger_probe.out`].
  - Pilot 1: parent 0.095, read STILL, rose to 0.108-0.111 [measured: MAIN BRP l.1319].
  - Pilot 2's reset of log-std to −1.5 was retired, never launched [measured: RUNLOG l.607; BRP l.1320].
- **Margin table:** every value matches `gate_margins.out` to two decimals. The statue passes the same certificate 40/40 [measured: RUNLOG l.601, l.620]. Its pass is "by design" [measured: GQP l.516].
- **Actuators:**
  - No command at or above 0.9. The largest PHASE command is hip roll at 0.257. The largest force use is 0.270, against 0.276-0.313 for OFF [measured: skeptic `m1_actuators.out`].
  - Hip-roll AC rms 0.076 / 0.076 / 0.062, against 0.039-0.067 for OFF. Jerk and smoothness cost −0.01 and −0.02 per episode [measured: `gate_margins.out`].
- **Returns:**
  - M3: 4487.6 / 4487.4 / 4479.3 against the statue's 4470.4.
  - s60's training-evaluation mean is 4442.7. The OFF means are 4412.6 / 4377.7 / 4415.8 [measured: SCORE l.324, 337, 341, 347].
- **Foot slide:** 0.019-0.022 m against the statue's 0.010 m [measured: `gate_margins.out`].
- **Unclean evaluation episodes:** 18 / 11 / 7 of 910, against 46 / 265 / 132 for OFF [measured: SCORE Q4 `clean` field].
- **Owed work:** M4, M2 and the critic readouts were not run. No parent is named. Nothing is certified [measured: SCORE l.357, l.361; REPORT l.19, l.362-373].

## What is wrong or overstated

1. **"The maintainer has not ruled on head height" (bottom line, item 1): WRONG as written.**
   - RUNLOG l.607, maintainer decision (2): "head posture: report-only telemetry ..., no gate change".
   - MAIN `docs/KNOWN_ISSUES.md` l.1336-1338 records the same as D-D46.
   - Under that ruling, head height cannot formally block a parent. Main's own drop standard is a window-mean head tip below 0.75 m [measured: KNOWN_ISSUES l.1342-1343], and the PHASE heads stay at or above 0.879 m.
   - What is still open is a judgement call on whether 0.90 m is acceptable for a parent. That is not a missing ruling [inferred].
2. **"Issue #491 is not found in MAIN's docs": WRONG.**
   - `docs/hardware/REFERENCE_MICRODUCK_2026_10.md` l.57 calls it "a candidate cause of #491's chatter".
   - `docs/investigations/TREX_STAGE1_BOUNCE_2026_08.md` cites it at l.5, l.182 and l.194.
   - `docs/hardware/SIM_TO_REAL_PLAN.md` cites it at l.186 [measured: git grep on origin/main].
3. **"In walking, nothing would hold the head up": OVERCLAIM.**
   - It is true that no head-clearance weight is set. The default is 0 [measured: MAIN `environments/tyrannosaurus_rex/base_env.py` l.254].
   - But `locomotion.toml` keeps `nosedive_weight = 0.5` "since head_contact is major termination cause", and `fall_penalty = -150` [measured: MAIN `locomotion.toml` l.10, l.21-22].
   - So walking keeps the head off the floor, though not at 0.90 m.
4. **"Three things block naming a parent" (bottom line): OVERCLAIM, and inconsistent with the body.**
   - The body rates s80's tail_2 MINOR and the head MINOR.
   - s60's falls and s80's creep each affect one seed and leave s42 untouched.
   - The only formal block is the absent M4 input [measured: SCORE l.361].
   - s60's late fall rate, 3 of 280 after 8,388,608 steps [measured: my `fall_seeds.out`], is within the gate's own tolerance: `max_hop_or_fall_episodes` 1 and `min_full_horizon_fraction` 0.95 [measured: M3 JSON thresholds, my `thresholds.out`].
   - Its joint-noise falls, 7/240, are close to OFF s60 (6/240) and OFF s42 (6/240), and below OFF s80 (17/240) [measured: skeptic `ledger_probe.out`].
   - "s60 is the weakest candidate" fits the evidence. "Blocked" does not [inferred].
5. **"More robust to command noise than the do-nothing statue" (bottom line): OVERCLAIM.** The body says to compare loosely; the bottom line states it as fact.
   - The joint-noise test's mean sigma is 0.161. But its knee and ankle sigmas are 0.078, 0.067, 0.082 and 0.066 [measured: `k3_panel.json.copy1` `joint_sigma.vector`]. They are below the statue test's SD of 0.10 [measured: RUNLOG l.585].
   - The own-noise test (0/240) uses std about 0.06, and 0.007-0.013 on the legs, so it is close to deterministic. The OFF runs also score 0/240 on it [measured: `ledger_probe.out`].
   - Robustness beyond the statue is likely but unmeasured on a matched test [inferred].
6. **Smaller points:**
   - "Closing everything above needs CPU only" does not hold for the tail_2 endpoint. The note's own body says that needs GPU.
   - The 33.6-34.4 Hz figure is the effective frequency of the whole action vector, not of hip roll alone.
   - The script found "distinct heading signatures: 6 of 6" at full precision. The headings agree only to 4 significant figures, a nuance the note leaves out.
   - The neck and head angles "below home" are commanded DC offsets, and their direction is inferred from the sign. They are not measured joint angles.
   - s42's report-only `max_stance_width_change_m` is 0.08106, above the gated settle bar's 0.08 [measured: `gate_margins.out`]. s42 also has the worst PHASE centre of pressure (0.42 of its allowance, against the statue's 0.40).

## What the note missed

- **No warm start, as built.** BR refuses any load whose critic reads a different phase, or none, than the stage declares. A PHASE stance cannot warm-start walking or recovery, and main cannot load it [measured: BR `train_base.py` l.667-676; docstring l.600-610; NEXT_STEPS l.295-305]. This blocks "usable walking parent" today. The fix is code only (Option B: drop the critic's phase columns at the warm start).
- **Screen recipe, not main's.** "Solved" is on the never-merged screen branch's LTSB recipe. Main's stance has "no r10 seed yet (bar 2)" [measured: MAIN `stages.toml` l.33].
- **Precedent for low noise is worse than stated.** PHASE noise (0.057-0.060) is below the OFF runs' (0.065-0.070) and well below pilot 1's parent (0.095), which read STILL [measured: `ledger_probe.out`; BRP l.1319]. The design item is stronger than "minor".
- **Main's own warning.** The HIGH known issue says stance gates admit stances that chatter, march or stand on one foot [measured: KNOWN_ISSUES l.1264]. This backs the "floor, not quality" point.
- **s80's tail_1 is also rising.** Its command went from +0.006 to +0.0415 at the final [measured: `m1_*.out` `cmd_a` "11"].
- **Head lowering has a price.** The neck-posture term pays 194.65-196.71 of 200 [measured: `gate_margins.out`], so lowering the head costs reward that some other term must pay back. The "settles where the reward stops paying" story is incomplete [inferred].
