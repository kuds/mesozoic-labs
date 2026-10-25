# r10 tail chain: probe brief for the maintainer

**Scope.** I tested CPU only, with no training, on scratch MJCF copies. The repo was not changed. The two probes behind this brief:
- **chainsizer** (the tail-chain sizing probe): `scratchpad2/tailround/r10probe/chainsizer/`, written **CS** below.
- **cost** (the repo-cost probe): `scratchpad2/tailround/r10probe/cost/`, written **COST** below.

The r9 MJCF in the scratch copy is byte-identical to main at 9f2355a3, which includes #636 to #640 [measured: cmp, `git diff --stat HEAD origin/main -- environments/tyrannosaurus_rex` is empty; `CS/r9.xml`].

**Labels.** [m] = measured, with its source. [i] = inferred. Unless a row says otherwise, values come from the zero-action statue after a 20 s settle [m: `CS/results_v1.json`; the statue reward and survival columns come from `CS/panels/*.json`].

**In short:** three springref values (SR3) give a level tail that the springs hold at rest, with the servos near zero command and no joint on a limit. The cost is a physics revision: every r9 checkpoint stops loading and must be retrained. Policy interface r15 can stay.

---

## 1. Candidate tail chains

r9 is the baseline. Only spring settings were changed; masses and geometry are as in r9.

| | **r9 (today)** | **SR3, recommended** | **SR2** | **Stiff servos** (alternative not taken) |
|---|---|---|---|---|
| **Changed parameters** | none | springref t1/t2/t3 = 52.5 / 25.9 / 10.4 deg (3 values). Compiled model differs from r9 only in `qpos_spring` [m: MjModel array diff; `CS/xml/SR3_t123.xml`] | springref t1/t2 = 52.5 / 25.9 deg (2 values) | kp x10 and forcerange x2 on the three tail servos (6 values) |
| **Tail joint angles q1/q2/q3 at rest** | -12.07 / -11.62 / -5.30 deg [m] | 0.01 / 0.01 / 0.01 deg, the keyframe pose [m] | 0 / 0 / -5.92 deg [m] | -3.86 / -2.33 / -1.22 deg [m] |
| **Segment angles to horizontal, base to tip** | -16.2, -26.0, -27.5, -25.2, -21.7 deg [m] | -4.7, -2.8, +1.0, +3.3, +6.7 deg [m] | -4.7, -2.8, -4.9, -2.6, +0.8 deg [m] | -8.5, -9.0, -6.4, -4.1, -0.7 deg [m] |
| **Base-to-tip chord / tip height** | -23.4 deg / 0.397 m [m] | +0.1 deg / 0.933 m [m] | -3.1 deg / 0.856 m [m] | -6.2 deg / 0.785 m [m] |
| **Whole-body CoM minus mean foot site** | -16.1 mm [m] | -26.8 mm, i.e. 10.7 mm further back [m] | -26.7 mm [m] | -25.6 mm [m] |
| **Pelvis pitch, nose-down** | 1.54 deg [m] | 0.96 deg [m] | 0.96 deg [m] | 1.03 deg [m] |
| **Statue, zero action, seeds 3042-3081** | 39/40; seed 3081 nosedives; reward 3755.7 ± 19.9 [m: `CS/panels/C0_r9.json`] | 40/40; reward 3768.8 ± 16.1 with the tail target re-centred, 3584.4 with r9's target [m: `CS/panels/SR3_recentred.json`, `CS/panels/SR3_r9target.json`] | 40/40; reward 3767.4 with the target re-centred [m] | 40/40; reward 3766.4 [m] |
| **Statue, seeds 7042-7081** | not run | 40/40 [m] | not run | not run |
| **Statue under open-loop command noise, SD 0.10** | 21/40 full horizon; falls mostly nosedives [m: `CS/panels/*_noise10.json`, `CS/panels/falldir_*`] | 16/40; falls mostly backward [m] | not run | 17/40 [m] |
| **Servo force at rest, share of forcerange (t1/t2/t3)** | 26% / 25% / 11% [m] | about 0 (at most 0.03 N.m) [m] | 0% / 0% / 12% [m] | 42% / 25% / 13% of the doubled range; tail_1 is about 67 N.m continuous [m] |
| **Command needed to hold a level tail (q = 0)** | +3.50 / +2.16 / +1.15. Out of reach: commands are capped at ±1, and tail_1 never leaves its stop [m] | about 0 (-0.003 to -0.001) [m] | 0 / 0 / +1.15. tail_3 is still past the rail, which is the +1 pin geometry [m] | 0.35 / 0.22 / 0.12 [m] |
| **Distal-segment range, all three servos within ±0.8** | -31.8 to -19.5 deg [m] | -12.4 to +16.2 deg, about ±4-5 deg per joint [m] | -17.9 to +9.7 deg [m] | -30.8 to +20.0 deg [m] |
| **Joint-limit contacts** | tail_1 on its ventral stop, 30.3 N.m [m] | none at rest or anywhere in the command sweep; servo force at most 15% of forcerange even at a command of ±1 [m] | none [m] | none at rest; tail_1 reaches its stop at a command of -0.8 [m] |

**Rejected options:**
- **springref on tail_1 alone (SR1):** distal segment still -16.6 deg [m].
- **Raising the ventral stops to -3 deg:** all three joints rest on their stops and the servos barely move them, which is r9's stuck-on-stop failure moved up [m: `RNG_m3` entry in `results_v1.json`].
- **Stiffness 40 or 160 together with springref:** stiffness only trades reach against the size of springref; it adds no benefit [m].

**Caveats:**
- **The 52.5 deg springref is a preload device, not an anatomical angle.** It applies 73 N.m at q = 0 and stands for passive support from ligaments or muscle tone [i]. The settled pose matches the soft-literature level-tail reconstruction, but no source fixes a tail angle [i: `posture/brief_corrected_summary.md`].
- **The 21 versus 16 of 40 noise result is within chance at n = 40** [i]. The direction it points to, a more rearward CoM and backward tipping, is real [m]. No trained policy was rolled on any candidate.
- **The noise result does not depend on the exact target.** Every raised-tail variant scores 15-17/40, including SR3 aimed at -3 deg per joint (distal -2.3 deg) [m]. A target with a slight droop (springref 45.3 / 19.6 / 5.0 deg gives distal -8 deg) recovers only 3.7 mm of the CoM shift [m: `CS/results_tsweep.json`].

## 2. What r10 costs in the repo

**r15 can stay, but only for a physics-only change.**
- springref, stiffness, damping, kp, forcerange, joint range, mass and the keyframe qpos move only the physics digest [m: `COST/out/layer_probe.out`].
- A change to the tail ctrlrange or to the home keyframe ctrl moves the policy-interface digest too, and forces r16 [m: same file, V6 and V7].
- SR3 changes neither, so r15 stays [i].
- Re-measuring `_TAIL_SETTLED_QPOS` moves no plant digest [m: `COST/out/const_probe.out`].

**Regenerated by tools.** The cost probe ran these on a placeholder r10 tree [m: `COST/om_r10` `git diff --numstat`]:
- **Plant manifests:** 2 x 5 lines. The writer refuses the change until physics_revision is raised.
- **Catalog:** 1 + 4 lines.
- **Mirror map:** 1 line.
- **Digest snapshot:** 30 of 649 lines, all T. rex, no policy lines.
- **Reset golden:** 117 lines, plus a hand edit of RECAPTURED 9 -> 10 in the test.

**Hand edits and re-measurement on CPU:**
- **Files:** the MJCF, `plant_versions.toml` (9 -> 10 plus a note), and the records (CHANGELOG, a decision row, KNOWN_ISSUES, NEXT_STEPS, README, website page).
- **Constants to re-measure:**
  - `_TAIL_SETTLED_QPOS`
  - `natural_pitch`, 0.027 today, about 0.016 on SR3 [m]
  - `_LEG_SETTLED_QPOS`, which moves by up to 5 mrad [m]
  - the statue constants in the four stage TOMLs, with `statue_constants_physics_revision`
  - the CoM and pitch comments in the XML header

  `target_z` stays at 0.9260, and the neck constants move by 1 mrad or less [m].
- **Statue panels:** re-panel the statue on the 280-seed and six-heading panels, about 13 min per 6+6 panels [m: COST].

**Tests.** 11 targeted T. rex tests failed on the placeholder r10, and all 11 pass on unmodified r9 [m: `COST/out/tests_r10_targeted.log`].
- **9 are expected:** they pin r9 values that must be re-measured.
- **2 need investigating before r10 is claimed safe:**
  - The heading-invariance rollout: joints differed by 7.2e-8 against a 1e-9 bound.
  - The pad centre-of-pressure fixture: 0.817 against a bar of > 0.82.

  The placeholder left tail_1 on its stop and SR3 does not, so the heading failure may not recur on SR3. That is untested [i].

**Effort and compute:**
- **Engineering:** about one agent-day of CPU work after the design is approved, about 12-14 hand-edited and about 8 regenerated files [i: from the r9 precedent #611 (33 files, merged the same day it was committed) [m: `git show --stat be9ce12e`]].
- **Retraining dominates:** one 12M stance run took 17 h 23 min [m: the stance symmetry and foot investigation doc]. Five r10 control seeds would take about 85 GPU-h, and a 5+5 treatment and control round about 170 GPU-h [i].

**Numbering clash:** "physics r10" already names the parked #623 foot screen in the records [m: `docs/investigations/stance_2026_10/r10_screen/criteria.txt`, CHANGELOG].

## 3. What it means for stance and walking

**Statue.**
- **Pose:** the tail goes from a drooped chain resting on its stop to a straight, level chain at the keyframe pose [m].
- **Reset transient on the tail:** falls from 1.34 to 0.47 rad/s peak [m].
- **Statue reliability:** 40/40 instead of 39/40, with no failure on the 11 gate thresholds re-checked [m]. That check used my copy of the thresholds, not the gate code.
- **Balance:** the CoM moves 10.7 mm rearward and the failure mode shifts from nosedive to backward fall [m].

**Reward tail target.**
- **The current target is the r9 droop.** `tail_home_pose` targets `_TAIL_SETTLED_QPOS` = (-0.2107, 0, -0.2029, -0.0926) rad. On SR3 the statue scores 0.258 on that term [m].
- **Re-centring it on SR3's settled pose (about the keyframe zeros) restores 0.999** [m]. The keyframe could then replace the constant [i].
- **Decision needed on the tolerance:** it is 0.05 rad, while the tail now has about ±0.07-0.10 rad of useful travel at a command of ±0.8. Keep the tolerance tight (reward only a level tail) or widen it [m/i].

**Gate reference.**
- **To re-measure:** `collapse_peak_floor_reference` (3756.8 today; SR3 statue 3768.8 [m]) and `min_avg_reward`, both with `statue_constants_physics_revision`; `natural_pitch`; and the gate-note statue values such as sole CoP, which need the gate's own code [i].
- **Live statue reference:** the gate rolls its statue reference live in each report [m: COST].

**Walking and lineage.**
- **Every r9 checkpoint is refused** for training, resume, warm start, evaluation and widening. That includes stance3 Runs 1-3 (Run 2 is walking pilot 2's parent), stance4 x3, stance5 x3, stance5r s70/s80 and r9 arm L s60 [m: `identity.py` compatibility_errors, `train_base.py` 579-600, the widen tool needs an equal physics_sha256]. The only bypasses are inspection-only flags [m].
- **Not tested:** actually loading a checkpoint.
- **Walking pilot 2** can still run on its own r9 branch, but its results would not carry over to an r10 main [i].
- **The tail_3 +1 habit is open:** whether a policy drifts tail_3 outward again on r10 can only be learned by training. At +1 the tail would now point above level, not merely less drooped [i].

## 4. Recommendation

**Do r10 (SR3) before more r9 stance training, after three cheap CPU checks. Run walking pilot 2 on its r9 branch first if its result is wanted soon.**

**For r10 first:**
- **r9's tail is fixed by the plant, not the policy.** tail_1 cannot leave its stop at any command, and a level tail needs commands of +3.5 / +2.2 / +1.15 [m]. Every further r9 stance run therefore learns around a drooped tail stuck on its stop, and the +1 pin habit feeds on that.
- **SR3 is the smallest fix:** three parameters, physics only, r15 kept.
- **The statue does not get worse** [m].
- **The retraining bill grows with time:** every r9 checkpoint has to be retrained eventually, and each new r9 run adds to that [i].

**Against r10 first:**
1. **It throws away all r9 lineage** and costs about 85-170 GPU-h before any r10 result can be compared with r9 [i].
2. **The balance cost is untested in closed loop.** The CoM moves 10.7 mm rearward and open-loop failures shift to backward falls (16 vs 21 of 40, not significant) [m]. A policy might handle this worse, and only training can show that.
3. **The 52.5 deg springref is a modelling device, not anatomy** [i]. The literature only softly prefers a level tail. A drooped tail is a weak reason to change physics on its own, if the stance goals do not depend on the tail.
4. **Two test failures are unexplained** (heading invariance, pad CoP), and the r10 number clashes with #623.
5. **A new rail habit could appear.** A policy could rail tail_3 on r10 as well, so r10 does not guarantee a fix.

**Before deciding, run these CPU checks** on the actual SR3 tree, not the placeholder:
- the heading-invariance and pad-CoP tests;
- the 280-seed and six-heading statue panels through the real gate code;
- a checkpoint-load attempt, to confirm what is refused.

**Your decisions:**
- the tail target: level (SR3) or about -8 deg distal;
- the number: r10, or fold the change into #623 or a later revision;
- whether `tail_home_pose` is re-centred, and with what tolerance.

All physics changes remain your decision.
