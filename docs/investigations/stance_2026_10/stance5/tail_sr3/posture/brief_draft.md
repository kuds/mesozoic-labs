# Tail posture decision brief (after the tail_3 bound screen, outcome C)

Date: 2026-10-25. Nothing was trained for this brief. The only simulations were noise-free statue settles and forward kinematics (FK) on byte copies of the r9 MJCF (sha256 4c3068f3, identical to the band clone at cf80ecb1). The four judge clones under fix/ have an empty `git status --porcelain --ignored` [measured, checked by the research agents at the end of their tasks].

**Labels.** [measured] means the value was read or computed from the named file. [inferred] means it was reasoned from measured values, or comes from general literature knowledge that was not checked in this session.

**Source keys.** Paths are under `scratchpad2/tailround/` unless stated otherwise.
- SCORE = `screen/scores/score.txt`
- CRIT = `screen/criteria.txt`
- PANELS = `screen/panels.csv`
- VERIFY = `screen/verify/REPORT.md`
- PT = `posture/posture-translator/` (groups.txt, per_panel.csv, extra.out, run2_fk.out)
- AN = `posture/anatomy/probe/tailpose.out`
- OPT = `posture/options/` (fk_auth.out, reward_arith.out, com_shift.out, traj.out)
- PROBES = `probes/` (noisevalue/analyze.out, unclip/out/)
- BRIEF2 = `brief_v2.md`
- SAT = `checkout/docs/investigations/stance_2026_10/stance4/satdiag/report.md`
- RUNLOG = `scratchpad/foot/stance4/live/run_log.txt`

**Angle conventions.** "Distal" means the tail_3 segment's angle to horizontal. Tail_3 is fused with segments 4 and 5. "Base-to-tip" means the line from the tail base to the tail tip. The world angles and tip heights assume the statue's root pose, because the screen traces have no root or CoM columns [measured, PT, CSV header]. The two FK measures used (capsule axis in PT/OPT, segment frame in AN) differ by up to about 0.4 deg. Each comparison below uses a single measure.

---

## Bottom line

- **The screen's posture difference is small.** BOUND60's parked tail sits about 1.5 deg below the rail at the tail_3 joint. It sits about 2.5 deg above the one self-cured control (CURED, below the line) at the joint, and about 3 deg above it at the distal segment. That is a few centimetres at the tip [measured, PT].
- **Within r9, the "cured" posture is the least plausible one.** CURED here means the drooped statue tail. Every posture r9 can hold carries the whole tail about 17 to 26 deg below horizontal. A plausible T. rex tail is roughly level [inferred, AN plus literature].
- **What the PARKED line actually measures is actuator headroom, not anatomy** [inferred].
- **Recommendation: revisit the parked rule (option 1).** Judge the tail on the rail and on persistence, and report posture in degrees. This would apply only to a new pre-registered round. BOUND60's NOT ADOPTED stands.
- **Separately, record now that a level tail needs an r10 tail-chain revision.** Decide that before robotics, not before walking.

---

## 1. What the screen says, in posture terms

**Verdicts.** Outcome C stands, and it was re-scored independently [measured, SCORE, VERIFY].
- BOUND90: 6/6 replicates on the rail.
- BOUND60: 6/6 PARKED, with 0 tail-up flags. The R1/R2 mean CMD was 0.600 to 0.632.
- Both s42 BOUND60 margins over the 0.60 line are thin: +0.0002 and +0.007. The s80 margins are +0.015 to +0.032 [measured, SCORE, VERIFY].

**Where BOUND60 holds the tail (R1/R2, 6 replicates).**
- tail_3 window-mean command: +0.616 [measured, PT groups.txt].
- Joint angle q3: -0.0398 rad, which is -2.28 deg (range -2.15 to -2.45) [measured, PT groups.txt].
- Distal segment: -24.55 deg [inferred FK, PT].
- Tip height: 0.429 m [inferred FK, PT].
- The tail is held still within an episode: the q3 standard deviation is at most 0.0011 rad [measured, PT].
- The bound sets the posture: the command tracks b (0.57 to 0.69 at all 54 BOUND60 panels) [measured, OPT traj.out].
- The plant map q3 = -0.0924 + 0.0863a predicts BOUND60's q3 to within 0.0006 rad [inferred, BRIEF2 map against PT].

**BOUND60 against the reference postures.**

| Posture | q3 (deg) | Distal (deg) | Tip (m) | Source |
|---|---|---|---|---|
| Statue droop (zero command) | -5.29 | -27.68 | 0.391 | measured q, AN; FK OPT fk_auth.out |
| Self-cured control s42 r202 at R2 | -4.78 | -27.50 | 0.391 | measured q, PT; FK inferred |
| **BOUND60, R1/R2** | **-2.28** | **-24.55** | **0.429** | measured q, PT; FK inferred |
| s80 start state (4.19M) | -2.20 | -24.67 | 0.427 | measured q, PT; FK inferred |
| stance3 Run 2 final (walking-pilot parent, CMD +0.841, 0/40 flagged) | -1.09 | about -23.5 | about 0.442 | q measured, SAT §5; FK inferred, assumes statue tail_1 and tail_2 |
| Rail (controls and BOUND90, R1/R2) | about -0.8 | -22.62 | 0.455 | measured q, PT; FK inferred |
| tail_3 at its +12 deg dorsal stop | +12.0 | about -10.4 | about 0.62 | OPT fk_auth.out, AN [measured, unreachable at r9 servo reach] |

**What the table shows.**
- **Against the cured control:** BOUND60 is 2.5 deg higher at the joint, 2.9 deg higher at the distal segment and about 37 mm higher at the tip at R2 [joint measured; distal and tip inferred, PT].
- **The cured replicate is a single run that was still descending.** It passed through BOUND60's posture at 4,980,736 (command 0.584, distal -24.9 deg) on its way to the statue droop [measured, PT].
- **Against the walking-pilot parent:** BOUND60 is about 1.2 deg *lower* (more drooped) at the joint [measured, SAT and PT].
- **Against the s80 start:** BOUND60 is the s80 start's posture to within 0.1 deg [measured, PT].
- **The whole r9 range is narrow.** From statue droop to rail, the distal segment moves only about 5 deg. tail_1 sits on its ventral stop (-12.07 deg) in every panel of every arm [measured, PT per_panel.csv].
- **The PARKED line separates 0.3 command units.** CMD 0.60 against 0.90 is about 1.5 deg at the tail_3 joint [inferred, plant map].

**What BOUND60 cost (1M-step window only).**
- **Falls:** none at R1 or R2. There was 1 fall in 1,080 BOUND60 episodes, at 4,980,736, in an episode with the tail *down* (command -0.226). Each of the other arms had 0 in 1,080 [measured, PT extra.out].
- **Saturation and the other 14 actuators:** no window step at |cmd| >= 0.9 in any arm. Force fractions are alike across arms [measured, PT].
- **tail_3 force fraction:** 0.203, against 0.247 on the rail and 0.111 on the statue [measured, PT].
- **head_pitch command:** higher in BOUND60 (+0.178 against +0.134 for the rail control). This is report-only; satdiag found head carriage not specific to the habit [measured, PT; SAT §3].
- **Returns:**
  - Against the rail control: +20 to +21 per episode [measured, PT summarize.out].
  - Against BOUND90: +21 to +36 [measured, PT summarize.out].
  - Against the one cured control at R2: -18 to -23, with paired SE 4 to 7 [measured, PT extra.out].
  - The deficit against the cured control most likely reflects what tail_home_pose charges for a raised tail, not a cost of the bound. At the same posture (the s80 start), holding tail_3 at 0 gained +39 per window [inferred, PROBES noisevalue, n = 1 cured].
- **Not yet known:** a K3 cost needs 6.29M and final, so 1M steps cannot show one [measured, CRIT KNOWN LIMITS].
- **The policy still prefers the tail higher than b:** 52 to 59% of training rows had tail_3 mu above 0.60 [measured, VERIFY].
- **The reward does not pay for tail-up.** At both start states, holding tail_3 at 0 beat the free policy in 40/40 paired stochastic episodes, by +36 to +55 per window [measured, PROBES noisevalue]. The drift is still unexplained [inferred].

---

## 2. What a plausible standing T. rex tail is, and where this model sits

**Literature.** These points come from the research agent's own knowledge. None was checked against the sources in this session.
- **Established:** large theropods, Tyrannosaurus included, are reconstructed with the trunk roughly horizontal and balanced over the hips, and the tail held clear of the ground and extended roughly horizontally (Newman 1970, Biol. J. Linn. Soc. 2: 119-123) [inferred].
- **Established:** M. caudofemoralis longus runs from the proximal tail to the femur's fourth trochanter. It is the main femoral retractor. It uses the tail as an anchor and does not hold the tail up (Gatesy 1990, Paleobiology 16: 170-186; Persons & Currie 2011, Anat. Rec. 294: 119-131) [inferred].
- **Established in principle:** tail mass behind the hip helps balance the trunk and head (Henderson 1999, Paleobiology 25: 88-106; Allen et al. 2013, Nature 497: 104-107) [inferred].
- **Debated:** the exact resting angle and curvature, tail mass, muscle size, and caudal range of motion. No source fixes a degree value. "Roughly level, within about ±10 deg base-to-tip" is defensible as a target; any particular angle is not [inferred; the ±10 deg tolerance is the agent's own judgement].
- **No tyrannosaur source known to the agent:**
  - The MJCF's reason for fusing tail_4 and tail_5 ("ossified tendons"; lines 204-206 of the copy) looks borrowed from other clades [inferred].
  - The ±12 deg per lumped joint is a modelling choice, not a measured value [inferred].

**Where the model sits.**
- **The authored zero pose is roughly level.** With all tail joints at 0, base-to-tip is -0.8 deg and the tip is at 0.907 m, against a hip at 0.93 m [measured, AN §B].
- **The settled statue droops.**
  - q1 -0.2107 sits on tail_1's ventral stop, q2 -0.2025 sits near tail_2's stop, and q3 is -0.0924 [measured, AN §C].
  - Base-to-tip is -23.2 deg and the tip is at 0.401 m, about 43% of hip height [measured, AN §C].
  - tail_1 and tail_2 account for about 24 of the roughly 29 deg of cumulative sag, so tail_3 cannot fix it [measured, AN].

Base-to-tip and tip height for the postures r9 can reach [measured, AN §C-D; screen postures use PANELS window means]:

| Posture | Base-to-tip (deg) | Tip (m) |
|---|---|---|
| Statue | -23.2 | 0.40 |
| BOUND60 park | -21.4 | 0.44 |
| Rail | -20.3 to -20.6 | 0.46 |
| tail_3 ctrlrange widened to ±0.30, command +1 | -19.5 | 0.48 |
| Best reachable in r9 (tail_2 and tail_3 both at +1) | -16.6 | 0.55 |
| Plausible target, q about (0, 0 to -0.05, 0 to -0.05) | -3 to +1 | 0.86 to 0.95 |

- **Neither tail_1 nor the policies lift the tail base.** Adding tail_1 at +1 changes nothing: q1 stays on its stop [measured, AN]. The screen's policies leave tail_2 near its stop too (q2 -0.18 to -0.21) [measured, PANELS].
- **Holding a level tail needs more servo reach than r9 has** [measured, AN §E; OPT fk_auth.out]:

| Joint | Static torque at q = 0 | Servo reach (kp × ctrlrange) | Forcerange |
|---|---|---|---|
| tail_1 | 73.3 N.m | 20.9 N.m | 80 |
| tail_2 | 36.2 N.m | 16.8 N.m | 65 |
| tail_3 | 14.5 N.m | 12.6 N.m | 50 |

- The binding limit is servo reach, not forcerange [measured, AN §E].
- At tail_1, 73 N.m is 92% of the forcerange [inferred, arithmetic].
- **Balance gives little reason to prefer one posture.** Going from the statue to the rail moves the whole-body CoM back about 1 mm; a level tail moves it about 6 mm back and 28 mm up [measured, AN; OPT com_shift.out].
- **Verdict.**
  - The raised tail points the right way anatomically, but only marginally: the rail closes about 3 of a 23 deg gap.
  - No posture reachable in r9 is anatomically plausible.
  - The screen's CURED class (statue droop) is the *least* plausible posture available [inferred].

---

## 3. Options

These options can be combined. Option 5 uses GPU while option 2 or option 4 screens on CPU. Option 4 is meaningful only with option 1's ruling. Option 3 ends the r9 lineage that all the others build on [inferred, OPT].

### Option 1: Revisit the PARKED rule for future rounds (Recommended)

**Approval class.** The first step is a maintainer ruling, governing a new pre-registered round only. The screen's NOT ADOPTED stands: fails stay fails, and the 17:46 ruling changes no verdict [measured, CRIT line 228, RUNLOG 17:46]. A production bound would then need an opt-in `[ppo]` training-loss key, off by default with coef-0 bit-identity. hyperparameters_sha256 moves, but checkpoints still load [measured, OPT; scratch bound diff cf80ecb1..4c0c8da7].

**What it changes.** The criterion becomes the rail plus persistence. Concretely: no tail-up flag at any judged checkpoint, with posture reported as command, q3 and distal degrees. The CMD >= 0.60 line no longer drives a verdict.

**Evidence for.**
- The line separates postures about 1.5 deg apart that are anatomically equivalent [inferred, §1-2].
- In r9, the "better" side of the line is the more drooped, less plausible tail [inferred, §2].
- The literal line already classes clean holding states as PARKED, including the walking-pilot parent (+0.841, 0/40) and the s80 start (+0.6285) [measured, CRIT REFERENCE TABLE, RUNLOG 17:37/17:46].
- The original 16:00 ruling described "parked" as command about 0.6-0.9. BOUND60 sits at that range's bottom edge [measured, CRIT line 33].
- BOUND60 showed no measurable harm in its window [measured, §1].

**Evidence against.**
- No single CMD or angle line separates stable from relapsing states. P1 at 0.794 and Run 2 at 10.49M (0.866) were clean and then relapsed [measured, CRIT REFERENCE TABLE]. So the rule must carry a persistence clause, not just a higher line [inferred].
- BOUND60's evidence is 1M steps of continuation from two states. Its coef (2.4) was calibrated on continuation states and has not been tested from scratch [measured, RUNLOG 19:05, CRIT].
- Changing a rule after seeing the data invites the charge of moving goalposts. This is mitigated only by pre-registering before the next round [inferred].

**Cost.** The ruling costs nothing. The settling round is 5 from-scratch runs (seeds 42/52/60/70/80) of 12M steps on L4, about 68-72 GPU-h, plus judging on CPU [inferred, throughput from BRIEF2 §4].

**Control.** The 5 recorded band runs [measured]:
- H is 2/5 and saturation alarms are 2/5.
- They were judged under two judge schemas.
- No deterministic tail_3 command was recorded for band s52, s60 or s70, so posture readings for the control need Drive fetches [measured, CRIT KNOWN LIMITS].

**Risks.**
- The round cannot show the rail effect statistically: 0/5 against 2/5 gives Fisher p 0.22 [measured, BRIEF2 §3-4].
- A K3 cost is unknown.
- The policy may push against the bound for 12M steps (52-59% of rows sat above b) [measured, VERIFY].

**What would settle it.** In a pre-registered from-scratch round: no tail-up flags at any judged checkpoint through final on most seeds, K2-K4 no worse than the band control under the agreed harm rule, and a stable posture readout (q3 about -0.04, distal about -24.5 deg).

### Option 2: Re-centre the tail_3 reward target

**Approval class.** A reward change. It is built as a kwarg declared in OPT_IN_TASK_KWARGS, following the leg_home_pose_reference pattern. task_sha256 moves when the switch is on. Once on, resume is refused; only an initialize_next_stage warm start loads [measured, OPT; task_fingerprint.py; base_env.py 193-210].
- Editing the `_TAIL_SETTLED_QPOS` constant directly would change the reward silently, without moving the hash [measured, OPT].

**What it changes.** The tail_3 target moves from the statue (q3 -0.0926) to about -0.0406 (the a ≈ 0.6 posture).

**Evidence for.**
- The reward's optimum moves to a 0.6, and its pull at the rail is 4.9x today's [measured, OPT reward_arith.out §C].
- It makes the reward agree with the posture you want, instead of rewarding the droop [inferred].

**Evidence against.**
- The habit already survived about 47 per window of reward cost. The reward already prefers a = 0 by 36-55 per window, so the drift is not limited by the reward's slope [measured, OPT, PROBES, SAT §4/§6].
- A broad component of the kind used for the leg term weakens the pull at the rail (0.75-0.85x). The best single-Gaussian retune reaches only 2.4x [measured, OPT §B].
- A target at 0 (straight) would push *toward* the rail [measured, OPT §C].
- The statue's tail_3 term falls from 0.0625 to 0.0214 per step [measured, OPT §C].

**Cost.** About 3 h CPU for a continuation screen [inferred]. recovery.toml inherits stance's `[env]`, so its task moves too. collapse_peak_floor_reference and min_avg_reward must be re-measured [measured, stance.toml, recovery.toml].

**Risks.** It addresses what posture means more than it addresses the drift. It does not lift the tail base [inferred].

**What would settle it.** A CPU continuation screen with the rail cleared and posture at about 0.6, at no extra harm, against a coef-0 control.

### Option 3: r10 tail-chain plant (the only route to a plausible tail)

**Approval class.** A physics revision (r10). Policy interface r16 is also needed if any ctrlrange changes. Every r9/r15 checkpoint stops loading, including the walking-pilot parent and all band runs [measured, OPT; BRIEF2 §2B; plant_versions.toml physics_revision 9 / policy_interface_revision 15].

**What it changes.** It gives tail_1 and tail_2 (and tail_3) the reach to hold about q = 0. There are two ways [inferred, arithmetic on AN §E]:
- **Passive support:** move the spring rest angles (springref about 0.92, 0.45 and 0.18 rad) so that a zero command holds a level tail.
- **More servo reach:** raise kp to about 350, 173 and 69, or widen ctrlrange to about ±0.73, ±0.45 and ±0.24.
- Passive support suits a later robot better, because it needs less continuous servo torque at the tail base.

**Evidence for.**
- It is the only option that reaches the plausible zone (base-to-tip -3 to +4 deg) [measured, AN §B/E].
- The authored zero pose already shows a level tail was the design intent [measured, AN §B].

**Evidence against.**
- A level tail does little for static balance (6 mm of CoM) [measured, AN].
- It does not by itself say whether the policy keeps or drifts the tail [inferred].
- It ends the r9 lineage [measured].

**Cost.** A maintainer decision on the intended posture first. Then a statue validation on CPU (hours). Then a manifest regeneration and new statue constants (`_TAIL_SETTLED_QPOS`, home ctrl, statue-referenced bars). Then about 140 GPU-h for 5 + 5 seeds [measured classes; inferred hours].

**Risks.**
- tail_1 would be at 92% of its forcerange [inferred].
- It adds new unknowns to every statue-referenced bar [inferred].
- It could delay walking [inferred].

**What would settle it.** An r10 statue on CPU that settles with base-to-tip within ±10 deg, stable stance and sane force margins. Then a stance round with no rail and K1-K4 at least as good as r9's.

### Option 4: Deterministic-state bound

**Approval class.** An opt-in training-loss key. Checkpoints still load [measured, OPT].

**What it changes.** The bound is applied to states from deterministic rollouts, as well as training rows.

**Evidence for.** It targets the quantity the gate reads.

**Evidence against.**
- At b 0.90, its best outcome is a command near 0.9, which reads PARKED under the literal rule.
- At b <= 0.60, the existing bound already holds deterministic play.
- So it has a use only after option 1's ruling, as a rail guard [inferred, CRIT line 222, OPT traj.out].

**Cost.** About 2-3 h CPU if a coef-0 identity proof lets the screen's CONTROL replicates be reused; otherwise about 4.6 h [inferred from RUNLOG 19:11-23:46 timings].

**Risks.** More code: a deterministic-state collector and an extra loss term [measured, OPT].

**What would settle it.** A b 0.90 CPU screen with 0 RAIL replicates, judged under a revised rule.

### Option 5: Pause tail work and start walking pilot 2

**Approval class.** Already approved under decision 7a(i), and on HOLD on `claude/pilot-walk-r9-b` at 8c5478a6. A criteria amendment is needed for a tail readout [measured, RUNLOG lines 204-206, 250-272].

**What it changes.** No change to the tail. It adds walking evidence from an r9 stance parent.

**Evidence for.**
- It is the cheapest GPU option and already approved.
- The habit has not been shown to harm anything measurable (OTHER 0 everywhere) [measured, SCORE].

**Evidence against.**
- The walk stage prices neither tail pose nor saturation (both default to 0) [measured, OPT; locomotion.toml; base_env.py 277-280].
- The parent goes in and out of the habit (28/40 judged saturation failures at 9.96M) [measured, SAT §5].
- The WALK rule never reads the tail [measured].

**Cost.** 3 seeds × 4,194,304 steps, about 15 GPU-h [inferred; walk-stage throughput not measured].

**Risks.** Without a readout it cannot inform the tail question [inferred].

**What would settle it.** Pilot 2's own WALK rule. For the tail question, a report-only tail_3 readout (window-mean command, q3, flags at 2M, 3M and 4.19M), added before any step.

### Why option 1 is recommended

- **The rule, not the policy, is what blocks progress.** BOUND60 did what a posture bound should: it cleared the rail 6/6 with no measurable harm [measured]. It failed only on a line that separates postures about 1.5 deg apart, and that line ranks the more drooped tail as better [inferred].
- **This is a recommendation to revisit the parked rule.** It does not change the screen's verdict.
- **It costs nothing to decide.** It also opens option 4 as a rail guard, and it does not block option 5 (run in parallel, with a tail readout added by amendment).
- **The anatomy question is a separate decision.** If an anatomically plausible tail is a goal, only option 3 reaches it [inferred]. Record that intent now, so the r10 statue probe can be planned before robotics.
- **Honest caveat.** The evidence for option 1's round is 1M continuation steps, one cured comparison and an untested from-scratch dose. The GPU round is where it is settled, not the ruling.

---

## 4. Still unknown, and cheap to learn

All of these are CPU or no-cost unless noted.
1. **Root pitch and CoM per arm.** These are not in the traces [measured, PT]. A read-only re-trace of the screen's final checkpoints, logging root and CoM, would turn the [inferred] world angles into measured ones. CPU, under an hour per arm [inferred].
2. **r10 statue probe.** On a copy of the MJCF, set the tail springref (passive support) or raise tail_1/tail_2 kp, settle the statue, and read stance stability, force margins and tail angle. CPU, minutes. Needed before option 3 can be priced honestly [inferred].
3. **Run 2's tail_1 and tail_2 angles.** These are not in SAT's tables. They fix the parent's distal angle, now assumed at about -23.5 deg. Needs a re-trace of the final checkpoint, which may need Drive [measured gap, PT].
4. **The parked posture's value in closed loop under full 15-channel training noise.** Not yet measured (BRIEF2 §5 item 1). A noise-value probe like the existing ones, at BOUND60's final [inferred, CPU hours].
5. **What drives the drift at s42's onset (3.1-3.5M).** It is untested there. The existing hold probes, applied at those checkpoints, would say whether the reward or the objective pushes outward [measured gap, OPT open questions].
6. **The coef-0 identity proof for option 4.** CPU minutes. It decides whether the screen's controls can be reused [inferred].
7. **The pilot 2 tail readout amendment.** A text change only, before any pilot step [inferred].
8. **The intended posture and tolerance.** A maintainer statement, for example "base-to-tip within ±10 deg of level" or "droop acceptable". This anchors options 1-3 [inferred].
9. **Literature checks before anything relies on them.** Verify the citations above, the multibody models' tail segmentation (Sellers & Manning 2007; Sellers et al. 2017), and the fusion rationale, which should be re-sourced or reworded [inferred, AN open questions].
