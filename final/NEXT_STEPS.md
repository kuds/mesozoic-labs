# After the phase-critic round: next steps

2026-11-02 (UTC), drafted 19:46-20:03 UTC, checked by an adversarial verifier (`ST/final/verify_next/NOTES.txt`),
and finalized from 20:14 UTC with each finding checked against its source (`ST/final/FINALIZE_NOTES.txt`; the draft
is kept as `ST/final/drafts/NEXT_STEPS.md.v1`). **Proposals only.** Nothing here is started, approved or scheduled.
No training, simulation, Drive access, git write or network call was made for this text. Three small read-only
scripts read the round's F2 copies; they and their outputs are in `ST/final/next_steps_work/` (W below).

**In short.** The pinned reading is line 4, "CONSISTENT WITH THE PHASE CRITIC REMOVING THE HABIT, NOT CONFIRMATORY";
the pre-registered test failed (NOT SEPARATED, p = 20/20). The recommendation is to close out the three PHASE finals on
CPU first (Option A: the owed M4 judge, head posture, the critic readouts, PHASE s60's falls), land the phase critic on
main default-off (Option B), and only then decide whether a walking pilot from an uncertified PHASE parent (Option D)
is worth an exception to the plan's rule of a certified stance before walking. More stance GPU hours (Option C) buy a
confirmatory claim, not a better stance.

Labels: **[measured: source]** = read or computed from the named file, line or field. **[inferred]** = reasoning; its
basis is given when it is not obvious. "My arithmetic" means I computed it from the named file.

## Short names

| Name | Meaning |
|---|---|
| ST | `/home/user/work/pc_study` |
| RUNLOG | the run log, `S1/foot/stance4/live/run_log.txt`, read at 971 lines (sha256 89425ea9...) at 19:46 UTC |
| S1, S2 | the two session scratch folders, as `ST/round/criteria.txt` lines 38-39 define them (full paths left out: they contain a product name) |
| PS | the phase screen, `/home/user/work/phase_screen` (its `REPORT.md`) |
| CHAT | the session's own conversation record (a JSONL file whose path contains a product name, so it is not written out); "record N" is its line number |
| MAIN | `origin/main` at 4c9d5e9b (the merge of #657), read with `git show`; nothing changed |
| BR | the phase-critic branch `screen-trex-r10-phasecritic` at 1dfe2327 (worktree `/home/user/work/pc_branch`, read only) |
| RB | the run branch `screen-trex-r10-armltsb` at 8737cd3d, BR's base (the "LTSB arm" stance recipe) |
| PHASE, OFF | PC's three runs with the critic phase input, and its three switch-off runs |
| STANCE6 | the first r10 round (`stance6_r10LTSB_s60/_s42/_s80`, trained 2026-10-27/28) |
| SCORE | `ST/f2/score/score_pc.txt` (the pinned scorer's output); DEV = `ST/f2/score/score_pc_dev.txt` |
| CRIT | `ST/round/criteria.txt` (pinned) |
| REPORT | `ST/final/PC_REPORT.md` (this review's round report) |
| NS6 | `S2/r10round/f2/NEXT_STEPS.txt` (the previous round's next steps); REVIEW6 = `S2/r10round/f2/REVIEW.txt` |
| REC | `/home/user/work/wt_records3/docs/investigations/TREX_TAIL_SATURATION_HABIT_2026_10.md` (unpushed records draft) |
| BRP, GQP | `MAIN:docs/BEHAVIOR_RECIPES_PLAN.md`, `MAIN:docs/GAIT_QUALITY_PLAN_2026_09.md` |
| W | `ST/final/next_steps_work/`: `head_neck.py`/`.out`, `short_eps.py`/`.out`, `env_fields.py`/`.out` |

Branch names are written without their agent prefix.

## Words used (one line each)

- **Actor:** the network that picks the motor commands. It is what runs on the robot.
- **Critic:** the network that scores how good a state is. It only shapes training; it is not used at run time.
- **Phase input ("phase critic"):** three extra inputs, for the critic only, saying where the episode is relative to
  step 200 (a ramp, a "settled" flag, a discount). The actor's 63 inputs are unchanged.
- **Hidden clock:** a reward term that switches on at a fixed episode step that no input shows. The working
  explanation of the tail habit [inferred] is that the critic then guesses the time from something that drifts, such
  as the tail angle, and training pays the actor to move it.
- **Rail / park:** the tail motor held at its limit / held high (mean command 0.60 or more) without a rail.
- **TD (tail drive):** the mean size of the tail_3 command over a run's 91 training evaluations. 0 = idle; 1 = full.
- **Gate certificate:** the six-heading stance test on the final checkpoint, 40 seeded episodes per heading. A screen
  measurement, not certification.
- **M4 judge (K1-K4):** four extra stance tests: a clean test panel, a centre-of-pressure rail, falls under noise,
  and a roll rail.
- **Warm start:** a new stage (for example walking) begins from an earlier stage's trained networks.
- **Default-off:** the code is on main, but nothing uses it unless a config turns it on.
- **Sham input:** extra critic inputs that carry no time information. They separate "the phase" from "any extra input".
- **Deterministic:** the same seed, code and machine image give the same run, byte for byte.
- **L4-hour:** one hour of one Colab L4 GPU session.
- **LTSB:** the run branch's stance reward recipe (the stance3 recipe plus a left-right stagger penalty and a per-foot
  centre-of-pressure band). **F2:** the end-of-run copy of each run's files from Drive, checked against its manifest.
- **M1:** deterministic replays of a saved checkpoint. **M3:** the gate certificate. **C8:** the manipulation check
  (is the phase in force; is the actor blind to it). **A6:** the pre-launch proof that switch-off code equals the old
  code. **K3 panel:** the in-run copy of the K3 noise-falls test.
- **V − G:** the critic's value minus the real discounted return. **Loop gain:** how much the tail_3 command rises per
  unit rise of the observed tail angle. **XR:** the cross-reference file of the #655 review
  (`/home/user/work/pr655_review/xref/XREF.txt`).
- **Log-std:** the policy's action-noise setting (log of its standard deviation). **screen_v2_reasons:** the ledger
  column that records why an evaluation episode was not clean.
- **Sign-flip test:** a paired test that flips the sign of each pair's difference in every possible way and asks how
  often the total is as large as observed. **Bootstraps the value:** at a time-limit cut, training uses the critic's
  estimate in place of the missing future reward.

## 0. The question, and the answer given then

- **The question** [measured: CHAT record 14791, 2026-11-01 03:39:10 UTC]: "Gaming planning if all six runs
  essential show that the phase critic works, what next? Some more fine tuning on the stance behavior or moving onto
  walking behavior?" (quoted as typed).
- **It is not in RUNLOG** [measured: grep of RUNLOG for "walking", "what next", "fine tun" at 19:46 UTC]. RUNLOG's
  last lines on walking are 606-607 (walking pilot 2 retired) and 867 (the critic-only input chosen as "valid for
  later walking stages").
- **The answer then** [measured: CHAT, the assistant entry of 03:39:37 UTC]: move toward walking rather than keep
  tuning stance, "but through a few short consolidation steps first":
  1. records and handoff;
  2. put the fix on main as a reviewed PR, and decide which stance reward setup main should carry (the run branch
     has four step-200 reward terms, main one, #655);
  3. pick and certify a stance trunk (walking starts from a stance checkpoint); it added: "The plan's GQ-3 rule
     already requires a certified stance before walking pilots.";
  4. audit the next stages for hidden time before training them;
  5. decide the actor's inputs now, for example a previous-action input for the chatter in #491 (labelled an
     inference then);
  6. then follow the chain; it named recovery as next.

  It named four cases for staying on stance instead. PC against them:

| Stay-on-stance case named then | What PC found | Source |
|---|---|---|
| The gate fails | **No.** PHASE gate certificate PASS in 3 of 3, 40 of 40 clean seeds each | [measured: SCORE lines 321, 325, 329; REPORT per-run table] |
| The habit moves to another joint (s80's neck) | **No.** "Habit elsewhere" FREE in 3 of 3 PHASE runs. OFF s80's saturating neck habit (alarms 5.5-7.1M steps) is absent in PHASE s80 | [measured: SCORE lines 329, 342, 345] |
| The tail parks high | **No.** Tail class CLEAR in 3 of 3. TD 0.008-0.017, against 0.15-0.64 for OFF | [measured: SCORE lines 321, 325, 329 (CLEAR), 362 (PHASE TDs), 334, 338, 342 (OFF TDs)] |
| Phase and off runs look the same | **Split.** In the numbers, no: every PHASE TD is below every OFF TD. **But** the pre-registered test reads NOT SEPARATED (p = 20/20) because of the 17:44 UTC opening; only the flagged deviation reads LOWER (p = 1/20); and the OFF runs are byte-for-byte reproductions of STANCE6, so only three control seeds exist. For this case the answer then said more seeds were needed "before relying on it" | [measured: SCORE lines 334, 338, 342, 362; DEV line 393; RUNLOG lines 902, 970-971; CHAT 03:39:37 UTC] |

- PC met none of the first three cases. The fourth is split: the numbers separate, the pre-registered test does not,
  and only three control seeds exist [measured: the table].
- [inferred] By the rule stated then, PC leans toward moving on, but the fourth case's own remedy (more seeds) is
  still open. The weakness is formal, not in the numbers. The pinned reading says "CONSISTENT WITH THE PHASE CRITIC
  REMOVING THE HABIT, NOT CONFIRMATORY" [measured: SCORE line 357]. That fail stays a fail.
- **One correction to the answer then.** On main, walking warm-starts from stance, not recovery. The walk recipe "flips
  to recovery once a recovery run passes (plan A2)" [measured: MAIN `configs/tyrannosaurus_rex/anatomical/stages.toml`
  lines 47-48]. So recovery is a sibling of walking today, not a required step before it.

## 1. Recommendation and decision tree

**Recommendation: move toward walking. Do not spend GPU hours on more stance tuning now.** In order:

1. **Option A** (CPU only, about a day): close out the three PHASE finals: run the owed M4 judge, check head posture
   (M2 at the judged checkpoints), run the skipped critic readouts and D-1 to D-15, and look at PHASE s60's falls and
   PHASE s80's tail_2 rise. This decides whether a PHASE final can be a parent.
2. **Option B** (code only, in parallel, 1-2 days): land the phase critic on main **default-off**, with a rule for
   warm starts across stages. As built, the code would refuse a warm start from a PHASE stance checkpoint into
   recovery or walking [inferred from the code text; not run].
3. **Option D** (about 30 L4-hours): a short walking pilot from the PHASE final that A names, as a declared exception
   like D-D36. This departs from the earlier answer's own step 3 ("The plan's GQ-3 rule already requires a certified
   stance before walking pilots") and from the plan itself: GQ-3 (a) holds "no walker session on today's rewards", and
   GQ-17 asks for a stance retrain of two seeds "before any trex walk" [measured: CHAT 03:39:37 UTC; GQP lines 185,
   700]. It needs the maintainer's explicit exception; without one, go to Option E instead.

Spend GPU hours on more stance seeds (**Option C**) only if you want a confirmatory claim, or want the phase critic
**default-on** in main's stance. **Option E** (main's stance recipe and certification) is the planned route to a
certified walking lineage; D tells you whether it is worth doing with the phase critic.

Why not more stance tuning [inferred]: no stance problem was found by the tests that ran. Gate PASS 3 of 3; tail CLEAR
3 of 3; no saturating habit elsewhere; mean return higher than the same-seed OFF run in all three pairs (+30.1, +103.4,
+66.5) [measured: SCORE lines 321, 325, 329 and 324, 328, 332, 337, 341, 347; REPORT "Returns"]. But K1-K4 (M4) and
the critic readouts were not run, PHASE s60 fell in 9 of 910 training-evaluation episodes against 0-1 in the other
five runs, and PHASE s80's tail_2 command crept up to +0.055 [measured: RUNLOG line 967; W/short_eps.out; SCORE line
332]. The open stance questions are about evidence (seeds, a sham control, main's recipe) and these loose ends, not
about a behaviour known to need tuning. The loose ends go into A.

```
Q1  Is "consistent, not confirmatory" (3 seeds; the pre-registered test failed) enough to act on?
    (Report only, beside it: if the phase critic changed nothing, "no rail in 3 runs" still has a 0.296 chance,
    SCORE line 359; Statistic A, phase 0 of 3 against switch-off 2 of 3 runs with an alarm, one-sided Fisher p 0.2,
    not significant, SCORE line 363. Post hoc, counting rails or parks at the OFF runs' 2 of 3, 0 of 3 has a plug-in
    chance of 0.037; a rate from 3 runs is very rough.)
    NO  -> Option C first (C1 canary, about 1.5 L4-h; then C2 sham arm, about 45 L4-h,
           or C3 new seeds, about 150 L4-h). Then come back to Q2.
    YES -> Q2.
Q2  Option A (CPU, no training): the pinned parent rule names the best PHASE final from M4's record. Does the
    named final also meet the bar the maintainer sets (a bar to add: for example K1-K4 pass at that final), with
    acceptable head posture and falls?
    NO  -> stay on stance for a parent: Option C3 (new seeds) or Option E (main's recipe).
    YES -> Q3. (Option B runs in parallel; D, E and any later use on main need it.)
Q3  Accept an uncertified, run-branch parent for a first walking test (an exception like D-D36)?
    YES -> Option D: walking pilot from the named PHASE final.
    NO  -> Option E: decide main's stance recipe, certify two seeds on main, then walk on main.
Q4  After D: did the pilot walk without a tail or neck habit?
    YES             -> Option E is the route to a certified walking lineage.
    NO, it stood    -> the cause could be the parent's low action noise (a stance-stage property) or the
                       walking reward; the pilot alone does not tell which.
    NO, habit back  -> audit locomotion for hidden time again; consider a locomotion-specific phase.
```

**Cost summary** (estimates [inferred] unless marked; basis below the table):

| Option | GPU (L4-hours) | CPU-hours | Build and review | Wall clock | Unblocks |
|---|---|---|---|---|---|
| A Close out the PHASE finals | 0 | about 4-8 | about half a day | about 1 day | naming a parent; a head-posture ruling |
| B Phase critic on main, default-off | 0 | about 1 | about 1 day | 1-2 days with review | loading PHASE checkpoints on main; cross-stage warm starts |
| C1 Canary replication of a PHASE run | about 1.5 | under 1 | about 1 hour | half a day | whether past runs can be reused as controls now |
| C2 Sham-input arm, seeds 60/42/80 | about 45 | about 4 | about half a day | about 1.5 days | "the phase" against "any extra input" |
| C3 New-seed paired round (5 seeds x 2 arms) | about 150 | about 8 | about 1 day | about 3.1 days | a confirmatory claim; default-on |
| D Walking pilot from a PHASE parent | about 30 (8M steps) or 15 (4.2M) | about 3 | about 1 day | about 1.5 days | whether a habit-free stance trunk walks |
| E Main's stance recipe and certification | 30-60 | about 4 | 1-3 days | about 1 week | a certified r10 stance on main (GQ-17) |

Basis: an L4 trains about 238-240 steps/s [measured: REVIEW6 lines 821-822]; 12,001,280 steps took 14.75-14.88 h from
session start to manifest in STANCE6 [inferred, as recorded: REVIEW6 line 823] and 14.7-15.0 h in OFF [measured:
REPORT "Consequences" 1]. Locomotion's pace on r10 is not measured; I assume the same steps per second, so 8,000,000
steps take about 9.5 h [inferred]. STANCE6's measurements took about 5.1 CPU-hours for three runs [measured: REVIEW6
lines 826-830]. BR's build, review and fix took about 4.3 h (20:57-01:12 UTC) [measured: RUNLOG lines 869, 873].
#643's full-depth test-sb3 CI job ran about 24 minutes [measured: RUNLOG line 605, 13:41:32-14:05:08]; the span of the
whole CI run is not recorded there.

## 2. The options

### Option A: close out the three PHASE finals (CPU only, no training)

**Goal.** Decide whether one PHASE final can be a parent, and close the loose ends PC left.

**What it would establish.**
- **A parent, or not.** The parent rule printed "not named": no M4 judge record exists at any PHASE final [measured:
  SCORE line 361]. Under the criteria, M4 at the finals was F2 work once line 4 read [measured: CRIT lines 269-272],
  and F2 did not do it [measured: RUNLOG line 967 lists C8, M1 and M3 only]. That is an F2 omission, and it is why no
  parent is named. How it happened: the pinned F2 step list (COLAB_STEPS lines 264-269) names the manipulation check,
  M1, M3, the completeness tool and the scorer, but not M4 or the critic readouts, and the process check
  (ST/f2/check_process) had no M4 item [measured]. [inferred] The slip sits in a gap between two pinned texts, and the
  independent process check did not catch it. The omission stays a fail.
- **What the pinned parent rule does** [measured: ST/tools/pc/score_r10round.py lines 505-540, sha256 3de2330a...,
  the pinned version]: among the POSITIVE finals, it names the one with the most clean panel episodes in its M4
  record, then the fewest noise falls (ties named). It never reads a K1-K4 pass or fail; it prints "not named" only
  when a record is missing. So running M4 will name a parent whatever K1-K4 read.
- **Head and neck posture.** No gate reads head posture [measured as recorded: NS6 section 0.1, citing KNOWN_ISSUES
  l1236-l1252]. A first look, report only, from the F2 ledgers (mean of the last 8 training evaluations)
  [measured: W/head_neck.out, my arithmetic on each run's `01_stance/eval_episodes.csv.copy1`]:

  | Seed | PHASE head tip | OFF head tip | PHASE neck_pitch command | OFF neck_pitch command |
  |---|---|---|---|---|
  | s60 | 0.913 m | 1.124 m | +0.074 | -0.353 |
  | s42 | 0.904 m | 1.111 m | +0.101 | -0.400 |
  | s80 | 0.902 m | 1.268 m | +0.076 | -0.405 |

  The r10 statue's head tip is 0.959 m [measured as recorded: NS6 line 76, citing REVIEW6 D-4; a different
  measuring window, so compare loosely]. So all three OFF runs held the head 15-31 cm above the statue's, and the
  PHASE heads sit about 5 cm below it [inferred from the table]. The stance reward pays full head credit from 0.90 m
  up, nothing at 0.75 m, and nothing extra above 0.90 m, in both RB's and main's recipe [measured: RB `stance.toml`
  line 316, MAIN line 292, head_clearance_target = 0.90; MAIN `environments/shared/reward_functions.py`
  reward_head_clearance]. The PHASE heads (0.902-0.913 m) sit just at that bar; they may simply settle where the
  reward stops paying [inferred]. Only OFF s80 crossed the "saturating" bar [measured: SCORE line 342]. This look is
  post hoc and decides nothing. In these three pairs the PHASE runs did not show the OFF runs' raised head; that would
  fit the why-probe's one-checkpoint finding of a neck "clock" in s80 [measured: REC §10.2 "Post hoc"; the link is
  inferred], but it is not a measured effect of the phase input.
- **The mechanism in these runs.** The pre-registered, report-only critic readouts (tail-angle slope, V − G by
  episode-step bucket, loop gain; CRIT lines 332-337) were not run in F2 [measured: RUNLOG line 967 lists C8, M1 and
  M3 only]. Run them, and add the same slope for `neck_pitch`.
- **PHASE s60's falls.** 9 of its 910 training-evaluation episodes ended early, against 0, 0, 1, 1 and 1 in the other
  five runs. They are spread from 3.4M to 11.7M steps and lasted 150-378 steps; 7 of the 9 ended between steps 211
  and 378 [measured: W/short_eps.out]. That is soon after step 200, where the gated terms switch on; whether the two
  are linked is not tested [inferred]. Read their recorded reasons (`screen_v2_reasons`) and the in-run K3 panel.
- **The heading gate over time.** The gate reads the final checkpoint only [measured: CRIT line 515]. OFF s42 failed it
  with 2 hop-or-fall episodes at several headings [measured: SCORE line 338]. Run the six-heading certificate (M3,
  about 33 min per checkpoint [measured as recorded: CRIT line 530]) at two earlier PHASE checkpoints, for example
  6,291,456 and 10,485,760 steps.

- **Other owed report-only work.** M2 posture at the judged checkpoints and D-1 to D-15 "as computable" (CRIT lines
  329-331) were not run either [measured: grep of SCORE]. PHASE s80's tail_2 ledger command rose from +0.014 to
  +0.055 over 4.19M-11.93M steps, the drift PS's guard warned of [measured: SCORE line 332]; M1's CMD at J shows
  whether it is real at the checkpoints.

**Design sketch.** CPU only, on the F2 copies already held: every 1,048,576-step checkpoint pair and the final pair
[measured: listing of `ST/round/fetch/F2/stance7_r10LTSBpc_s42/copies/01_stance/models/`]. At most 2 simulation
processes. The outcomes are known, so everything is report only, except that M4 feeds the parent rule that is already
pinned (stance3's rule) [measured: CRIT lines 270-272].

**Cost.** 0 GPU. About 4-8 CPU-hours [inferred: STANCE6's 84 measurement jobs took about 5.1 CPU-hours]. About half a
day to build and check, then a review.

**Risks.** The pinned parent rule names the best PHASE final whatever K1-K4 read (most clean panel episodes, then
fewest falls). If K1-K4 fail at that final, the maintainer decides whether it can still be a parent. For scale:
STANCE6 s80 failed K1 at 6.29M steps, not at its final (40 of 40 clean there), and K3 with 5 joint-noise falls at
6.29M and 2 at the final [measured: REVIEW6 lines 395-402; NS6 line 418]. M4 at the finals only would not have caught
its 6.29M K1 failure. My head-tip figures come from the ledger's telemetry columns, not the M2 posture tool, so they
may shift a little [inferred].

**Unblocks.** Q2 (a parent for Option D) and the maintainer's ruling on head posture (about 0.90 m, the reward's own
full-credit bar, against the statue's 0.96 m).

### Option B: land the phase critic on main, default-off

**Goal.** Put the critic-only phase input on main as a reviewed option. Then PHASE checkpoints load with main's code,
and any stage can turn the option on.

**What it would establish.** That the option is safe on main: with the key absent, main trains exactly as before; with
it on, the actor is untouched.

**Facts it rests on.**
- **Size.** BR adds 6 commits on RB: 10 files, +1,353 / −5 lines. Two new modules (`critic_phase.py`,
  `critic_phase_ppo.py`), `train_base.py` +69, small edits to diagnostics, `mirror_kl_ppo` and `policy_probe`, the
  notebook switch, one CI line, and 25 test functions (8 + 17) [measured: `git diff --stat 8737cd3d..1dfe2327`;
  count of `def test_` in the two test files].
- **Distance from main.** RB is 7 commits on merge base 06b1a8e1; main is 38 commits past that base [measured:
  `git rev-list --count`]. RB's own 7 commits (+3,425 / −776 lines over 22 files: the LTSB recipe's stagger and
  centre-of-pressure-band reward code and tests, the checkpoint cadence, test re-calibration and the round's pins) are
  **not** part of this landing [measured: `git diff --stat 06b1a8e1..8737cd3d`]. Main has no stagger or CoP-band reward code [measured: `git grep` of MAIN's
  `environments/tyrannosaurus_rex/base_env.py` finds "stagger" only in two comments, lines 723 and 1135].
- **Off path.** On RB, switch-off equals the old code: on CPU by test (A6), and on GPU by result, OFF reproduces
  STANCE6 byte for byte [measured: SCORE line 348; RUNLOG lines 970-971]. The port must prove this again against main.
- **Actor untouched.** The manipulation check found the actor's output changed by exactly 0.0 when the phase was
  perturbed, at all 6 checkpoints checked per PHASE run [measured as reported: REPORT "All six runs are COMPLETE",
  citing `score_pc.json` C8].
- **Does the clock problem exist on main? In the reward, yes but smaller. In behaviour, unmeasured.**
  - Stance: main pays **one** term only after step 200: `stance_width` with reference "settled" [measured: MAIN
    `stance.toml` lines 135, 165, 188; `tyrannosaurus_rex/base_env.py` lines 1149-1158]. The lateral-lean term has the
    same gate in code but weight 0 on main [measured: `base_env.py` lines 277, 1193-1195; `stance.toml` sets no
    `lateral_lean_weight`]. On the statue the reward rises +0.099 per step across step 200 on main, against +0.345 on
    RB: about 3.5 times smaller [measured as recorded: REC §10.3, citing XR l69-l76].
  - Recovery: `stance_width` is gated at step 140, and pushes land at 200 ± 50 steps [measured: MAIN `recovery.toml`
    lines 80-81]. Two hidden timings.
  - Locomotion: no step-gated term (Option D).
  - Behaviour: whether main's stance recipe forms a tail habit on r10 is unknown. Main has "no r10 seed yet (bar 2)"
    for stance, that is, no certified r10 seed [measured: MAIN `stages.toml` line 34]. No main-recipe r10 run appears
    in RUNLOG, so none is known to probe [inferred].
- **No cross-stage rule.** As built, any load (resume or warm start) whose critic reads a different phase, or none,
  than the stage declares is refused, both ways [measured: BR `train_base.py` line 609; `critic_phase_ppo.py` lines
  295-319]. The phase is keyed by the settle step S and the stage's discount gamma [measured: BR `critic_phase.py`
  lines 52, 81]. Stance is (200, 0.98). Recovery reads S = 140. Locomotion keeps S = 200 by default but has gamma
  0.995 [measured: MAIN `stance.toml` line 396, `recovery.toml` line 80, `locomotion.toml` line 44, `base_env.py` line
  271]. So, as built, a PHASE stance checkpoint cannot warm-start recovery or walking [inferred from the code text; not
  run]. For walking, the first refusal is that walking declares no phase at all: a checkpoint whose critic reads a
  phase is refused by a stage without one [measured: BR `train_base.py` lines 667-676, the code; line 609 is the
  docstring]. The gamma clash would matter only if walking turned the option on. Note: recovery inherits stance's `[ppo]` table [measured: `recovery.toml` line 49], so turning the option on
  in main's stance would turn it on in recovery too, with S = 140, and the stance-to-recovery warm start would then
  be refused [inferred].

**Design sketch.**
1. Port BR's 6 commits onto current main as one PR, without RB's recipe commits. Fix the module text, which says
   "four reward terms are paid only after the settle" [measured: BR `critic_phase.py` line 3]; on main it is one.
2. Tests: keep the 25. Add (a) a key-absent equivalence test against main (rollout-buffer digests, as A6 did); (b) a
   test that main's stance hyperparameter digest does not move while the key is absent; (c) tests for the chosen
   cross-stage rule.
3. Choose the cross-stage rule: keep the refusal; or drop the critic's three phase columns at a warm start into a stage
   without the option (the exact reverse of PC's zero-append; the actor is untouched); or start a fresh critic. The
   drop is the simplest rule that lets walking start from a PHASE stance [inferred].
4. Log the torch, CUDA runtime, cuDNN and GPU driver versions in `provenance.json` (section 3) [inferred, cheap].
5. **Default-off.** Default-on in main's stance is a separate, later decision: it changes every future stance run on
   main, moves its hyperparameter digest, drags recovery along (point above), and is untested on main's recipe
   [inferred]. It should wait for Option C3 or E evidence.
6. Records: the records draft holds a placeholder for PC [measured: RUNLOG lines 871-872]; the PR can cite it.

**Cost.** 0 GPU. About 1 CPU-hour of tests plus about 24 minutes of full CI. About 1 day of port, independent review and
fixes [inferred from BR's own 4.3 h build-review-fix].

**Risks.** Conflicts in `train_base.py`, which RB also changed (+34 lines) [measured: diff stat]. A subtle change on the
off path would end byte identity with past runs, and with it the reuse of past controls (section 3) [inferred].

**Unblocks.** Loading PHASE checkpoints with main's code; Option D's warm start (if the rule allows it); Option E with
the phase critic; a later default-on decision.

### Option C: more stance evidence, in three sizes

**Goal.** Turn "consistent with" into a tested claim, or answer "is it the phase?", for the fewest GPU hours.

**Facts.** Only three control seeds exist for RB's recipe on r10: STANCE6's s60/s42/s80, which OFF reproduced byte for
byte [measured: RUNLOG lines 970-971]. Also report only: "no rail in 3 runs" would have a 0.296 chance with no effect
(SCORE line 359), and Statistic A reads Fisher p 0.2 (SCORE line 363) [measured]. They were chosen with their earlier outcomes known [measured: CRIT lines
507-508]. A paired sign test on 3 pairs cannot go below p = 1/8 [measured: CRIT line 316]. PHASE's own determinism
was not measured [measured as stated: REPORT "Consequences" 3].

**C1. Canary: replicate a PHASE run's first 1,048,576 steps.**
- *Establishes:* whether PHASE training is deterministic too, and whether today's Colab image still reproduces.
- *Design:* relaunch PHASE s42's exact settings under a new RUN_ID; stop once its 1,048,576-step checkpoint lands;
  fetch that checkpoint and the evaluation rows up to it; compare with PHASE s42's own (policy-weight CRC32, row
  sha256). RUN_IDs do not enter training: OFF matched STANCE6 under different RUN_IDs [measured: REPORT "Really new
  runs"]. One L4.
- *Cost:* about 1.5 L4-hours (1,048,576 steps at about 240 steps/s is about 1.2 h, plus setup) [inferred]; under 1
  CPU-hour.
- *Risks:* a 1M-step match does not prove a 12M-step match, though OFF matched over 12M [inferred]. Stopping by hand
  must not trigger a resume.
- *Unblocks:* whether C2 can reuse STANCE6, OFF and PHASE as controls without rerunning them.

**C2. Sham-input arm on seeds 60/42/80.**
- *Establishes:* whether the phase, or any extra critic input, removes the habit. Neither PS nor PC had a sham arm
  [measured: CRIT line 513].
- *Design:* 3 runs, identical to PHASE except that the three critic columns carry no time information (for example
  held constant, or a phase drawn from another episode). Controls: STANCE6 (= OFF) and PHASE, already trained, if C1
  matches [inferred]. A short pre-registration fixes the reading (per-seed TD of sham against PHASE and OFF) before
  any run. Colab L4, three at once, 12,001,280 steps.
- *Cost:* about 45 L4-hours (3 × 14.7-15.0 h) [measured pace]; about 4 CPU-hours of measurement; about half a day to
  build the sham switch.
- *Risks:* a deterministic system is still sensitive: any input change, even a sham, can move a run [inferred]. So C2
  speaks clearly mainly if the sham runs look like OFF (habit back). The seeds are the same three known seeds.
- *Unblocks:* whether the hidden-clock story is right, which shapes the recovery and walking designs.

**C3. New-seed paired round.**
- *Establishes:* a confirmatory test on seeds whose outcomes nobody has seen.
- *Design:* 5 seeds never trained on r9 or r10 (checked against every Drive run name first), each run PHASE and OFF,
  12,001,280 steps, on BR (or main after B), Colab L4, three at once. Statistic: a paired sign-flip test on the
  per-seed TD difference (OFF minus PHASE). With 5 pairs the smallest p is 1/32 = 0.031 and needs all 5 pairs in the
  predicted direction. To allow one reversal you need 6 pairs, and then only if the reversed pair has the smallest
  difference (p = 2/64 = 0.031) [inferred: exact arithmetic]. (A sign test, which counts signs only, would need 8
  pairs for one reversal: p = 9/256 = 0.035.) Launch both runs of a seed in the same wave, and open no run's output
  until the last launch, so that no early look can void the test as the 17:44 UTC opening did [measured: RUNLOG line
  902; inferred remedy].
- *Cost:* about 150 L4-hours for 5 pairs (10 runs; about 180 for 6 pairs). With three Colab slots and both runs of a
  seed in one wave, each wave holds one pair, so 5 pairs take 5 waves of about 15 h, about 3.1 days of wall clock (6
  pairs: about 3.8 days); about 8 CPU-hours of measurement [inferred].
- *Risks:* cost; an image change between waves (a C1-style canary per wave guards it); OFF's habit rate on new seeds
  is unknown (2 of 3 so far [measured: SCORE line 358]), so a test on the habit class may be weak; a test on TD is
  safer [inferred].
- *Unblocks:* default-on for this recipe; a citable claim in the records.

The other stance checks asked about, the saturating neck habit of OFF s80 and the heading gate, need no training. They
sit in Option A.

### Option D: a walking pilot from a PHASE parent

**Goal.** Learn whether a stance trunk without the tail habit can start to walk, and whether a tail or neck habit
forms again during walking training.

**What it would establish.** For 3 seeds, at one budget: WALK, HOP, STILL or FALL by pilot 1's frozen rules, and
whether any tail or neck habit appears in locomotion. Three seeds give existence, not a rate [inferred].

**Does the locomotion reward recreate the clock problem?**
- **No step-gated term.** Main's locomotion `[env]` sets no `stance_width` or `lateral_lean` key, and their defaults
  are weight 0 [measured: MAIN `locomotion.toml`, 0 matches; `tyrannosaurus_rex/base_env.py` lines 268, 277]. So the
  stance clock is not recreated there [inferred].
- **Other time in locomotion.** The 1,000-step limit [measured: `locomotion.toml` line 33] is reported as
  "truncated" [measured: MAIN `shared/base_env.py` lines 1361, 1402-1404]; the training library then bootstraps the
  value at the cut [inferred: standard library behaviour, not checked in this tree]. The forward-velocity weight ramps
  from 0.2 to the stage's 2.0 over the first 2,000,000 training steps; a 300,000-step warm-up (clip range 0.02,
  entropy 0.02) runs over the same first steps [measured: `locomotion.toml` lines 6, 158-162; MAIN
  `environments/shared/curriculum/advancement.py` RewardRampCallback, progress = num_timesteps / ramp_timesteps;
  `train_base.py` lines 1086-1097]: a change across training, not a clock inside an episode [inferred].
- **Nothing prices a parked tail in locomotion.** `action_saturation_weight` and `tail_home_pose_weight` are not set
  and default to 0 [measured: `locomotion.toml`, 0 matches; `base_env.py` lines 260, 262]; `tail_stability_weight`
  is 0.05 [measured: line 18].
- **Later stages do have hidden timings.** Recovery: step 140 and pushes at 200 ± 50 steps [measured: `recovery.toml`
  lines 80-81]. The follow-direction behaviour changes its command every 6.0 ± 1.0 s [measured:
  `behaviors/follow_direction.toml` lines 18-19]; the command is an input, its next change time is not [inferred].
- **The phase critic as built is a stance tool.** Its features mark the step-200 settle [measured: BR
  `critic_phase.py` lines 1-30]. In locomotion they would mark a step where no reward changes [inferred]. The warm start
  would be refused: first because walking declares no phase, so a critic that reads one is refused [measured: BR
  `train_base.py` lines 667-676]; and, if walking turned the option on, because its gamma (0.995) differs from
  stance's (0.98) [inferred from code; Option B].

**History.**
- Walking pilot 1 (D-D36) **read STILL on 3 of 3 seeds** at 4,194,304 steps: the policy did not move. Its
  forward-velocity reward was 0.5-3.2 per episode, against 39-54 for September's walkers at 2.1M [measured: BRP
  D-D36, amendment of 2026-10-18]. That failed pilot stands.
- Pilot 2 (a new parent and a log-std reset) was retired on 2026-10-25, never launched [measured: RUNLOG line 607;
  BRP D-D36 amendments]. D-D45: a new pilot is designed once r10 has a stance parent [measured: RUNLOG line 607].
- GQ-3 (a), "no walker session on today's rewards", and GQ-17 / §12, a certified stance before any trex walk, are still
  open [measured: GQP lines 185, 199, 700]. A PHASE parent is uncertified and trained on RB's recipe, so a pilot would
  be an exception like D-D36 [measured as precedent: BRP D-D36].

**Design sketch.**
- Parent: the PHASE final that Option A names.
- Critic during walking: the stock critic. The phase columns are dropped at the warm start (Option B's rule, on a
  never-merged pilot branch if B is not merged yet). Record V − G by episode-step bucket at checkpoints, report only,
  to catch any hidden clock [inferred design].
- Before training a walking trunk, decide the actor's inputs (the earlier answer's item 5) [inferred there].
- Runs: 3 seeds (42, 52, 60, as in pilot 1), main's locomotion stage, 8,000,000 steps, the production budget. Pilot
  1's 4,194,304 steps were "too short to say anything about the gait" [measured: BRP D-D36 amendment]. A cheaper
  variant: 4,194,304 steps with pilot 2's log-std reset to −1.5.
- Reading: pilot 1's frozen gait probe (WALK, HOP, STILL, FALL) and its pre-registered outcome rules; tail and neck
  readouts (TD, alarms, head tip) report only. Colab L4, three at once.

**Cost.** About 30 L4-hours at 8,000,000 steps (3 × about 9.5 h), or about 15 at 4,194,304 [inferred; locomotion pace
on r10 is unmeasured]. About 3 CPU-hours of judging. About 1 day for the pre-registration and the pilot branch.

**Risks.**
- **STILL again.** The stance entropy bonus decays to 0 [measured: MAIN `stance.toml` line 400], so the parent's
  action noise is small; pilot 1's parent had 0.095 [measured: BRP D-D36 amendment].
- A habit may form during walking, where nothing prices a parked tail.
- The lineage is disposable: an uncertified parent on a recipe main does not have.
- GQ-3 (a) exists because "children inherit gaits" [measured: GQP line 185].

**Unblocks.** Q4: whether to certify a phase-critic stance on main (Option E), or fix locomotion first.

### Option E: decide main's stance recipe, then certify two seeds on main

**Goal.** The planned route to walking: a certified r10 T. rex stance on main.

**What it would establish.** A deliverable stance node on main; its bar is 2 seeds [measured: `stages.toml` line 34].
GQ-17 / §12 require it before any trex walk [measured: GQP line 700].

**Facts.** Main's stance recipe is not RB's: no stagger or CoP-band code, lean weight 0, no settled leg reference
[measured: Option B "Distance from main"; REC §10.3]. PC tested only RB's recipe. With RB's recipe and no phase input,
a rail or park formed in 2 of 3 runs [measured: SCORE line 358]. Main's recipe has no certified r10 seed [measured:
`stages.toml` line 34, "no r10 seed yet (bar 2)"], and no main-recipe r10 run appears in RUNLOG [inferred].

**Design sketch.** Two paths for the maintainer:
- **E1:** land RB's LTSB terms on main (RB's own diff is +3,425 / −776 lines over 22 files, and it also holds the
  round's pins, the checkpoint cadence and test re-calibration, so the reward PR would be a subset [measured: `git diff
  --stat 06b1a8e1..8737cd3d`; `git log` of its 7 commits]), plus Option B with the phase critic on in stance; then
  train 2 certification seeds.
- **E2:** keep main's recipe, turn the phase critic on, and train 2 seeds, each with a same-seed OFF run (4 runs),
  because no controls exist for main's recipe [inferred].

**Cost.** E1: 2-3 days of PR work and review, about 30 L4-hours. E2: about 1 day of build, about 60 L4-hours
[inferred].

**Risks.** E1 brings a large reward change to main. E2 tests a recipe whose r10 habit rate is unknown, so a clean
result may not tell an effect from no effect [inferred]. Either way, the recipe is the maintainer's decision.

**Unblocks.** A certified walking lineage under the plan's own rules, with no exception.

## 3. What determinism now allows, and its limits

**What was measured.**
- The three OFF runs equal STANCE6's same-seed runs byte for byte: ledgers, training rows, final weights and optimizer
  state, probe arrays and the M3 report text [measured: RUNLOG line 970; spot check line 971; REPORT "MEASURED: the
  three OFF runs ..."]. They ran 5 days apart, under different RUN_IDs and commits [measured: REPORT "Really new runs"].
- All nine runs (STANCE6's three, PHASE's three, OFF's three) record the same software and hardware: training
  library 2.9.0, gymnasium 1.3.0, MuJoCo 3.10.0, NumPy 2.1.3, Python 3.13.16, Linux kernel 6.6.159+, NVIDIA L4, CUDA
  13.0 [measured: W/env_fields.out, from each run's `provenance.json` and `01_stance/stage_config.json` F2 copy 1].
  **None records the torch, cuDNN or GPU driver version**, and the Colab image itself is not measured [measured:
  same; CRIT lines 497-498].
- The repository sets no determinism switch [measured: `git grep` on BR 1dfe2327]. The training library, at the
  version all nine runs record (2.9.0), sets the cuDNN flags (deterministic on, benchmark off) whenever it seeds on a
  GPU [measured: S2's copy of that version, `common/utils.py` lines 42-45, called from `base_class.py`
  set_random_seed]. Neither the repository nor the training library sets a deterministic-algorithms flag or a cuBLAS
  workspace setting [measured: grep of both]. So
  this determinism is a property of this setup, not a guarantee [inferred].
- A pinned premise was wrong: "GPU training is not bit-reproducible" [measured: CRIT line 149; REPORT "Consequences" 5].

**What it allows** [inferred unless marked].
1. **Paired designs.** Two runs with the same seed, code and image differ in their settings only by the change, and
   the per-seed difference carries no run-to-run noise. But a deterministic run is still sensitive: any change, even a
   sham input, sends it down another path, so one pair does not isolate what the change did. Any test is still across
   seeds: seed-to-seed spread remains.
2. **Cheaper controls.** For seeds 60/42/80 on RB's recipe, STANCE6 already is the OFF control, and PC's PHASE runs
   are a phase control. A new arm on that recipe needs 3 runs (about 45 L4-hours), not 6. PC's OFF arm cost 44.6
   L4-hours and bought this knowledge rather than a new sample [measured: REPORT "Consequences" 1].
3. **Exact replication checks.** A canary of about 1.5 L4-hours (C1) can show, before a round, whether the image still
   reproduces. If it does not, the controls must be rerun.
4. **Re-creating states.** A run can be replayed to a step it did not save, for probing, provided it never resumed;
   whether a resumed run reproduces an unbroken one is untested here.
5. **Clear blinding rules.** Reused controls are already known. A new pre-registration should say so in advance; PC's
   did not (the OFF TDs were printed in CRIT at the pin) [measured: REPORT "Consequences" 2].

**Limits.**
- Measured only for the switch-off configuration, on Colab L4 sessions of 2026-10-27/28 and 2026-11-01/02 whose
  recorded GPU and software fields are equal; the image itself is not measured [measured: REPORT "What it does not
  show"]. PHASE determinism is unmeasured (C1).
- A library, CUDA or driver upgrade, another GPU type, another `N_ENVS`, another thread count, or any change on the code
  path can break identity, possibly silently [inferred]. Thread count already changed results once: in the heading
  test, "which headings fail depends on the torch thread count" [measured: RUNLOG line 595].
- Colab can change its image without notice, and the torch and driver versions are not logged [measured that they are
  not logged; inferred that it can change]. Option B's PR can log them.
- A disconnect and resume would leave the byte-identical path [inferred]. PC's rule was "no resume" [measured: RUNLOG
  line 874].
- Determinism does not remove sensitivity: a tiny change can send a run somewhere else. One pair proves little; the
  same direction across many seeds is what counts [inferred; REPORT "What it means"].
- Reused runs carry seeds whose outcomes are known [measured: CRIT lines 507-508]. A confirmatory test still needs new
  seeds (C3).

## 4. Whatever is chosen

- The pinned reading stays the reading of record: line 4, "CONSISTENT WITH THE PHASE CRITIC REMOVING THE HABIT, NOT
  CONFIRMATORY", and THE ONE TEST NOT SEPARATED (p = 20/20), a failed pre-registered test. Any later text should quote
  it first and flag the deviation reading (LOWER, p = 1/20) as a deviation [measured: SCORE lines 357, 362; DEV line
  393].
- The owed F2 work (M4 at the three PHASE finals; the critic readouts; M2 posture at J; D-1 to D-15) is listed as open
  in REPORT "Open items"; Option A does it.
- The stance6 review's "[measured]" label on "GPU training is not bit-reproducible" (S2/r10round/f2/REVIEW.txt line
  511) is wrong: its source labels it inferred, and PC now contradicts it [measured: REPORT "Consequences" 5]. The
  records should correct it.
- PC has no records PR yet; the draft branch holds a placeholder [measured: RUNLOG lines 871-872].
- Every new option above needs the maintainer's word, and each training option its own pre-registration first.

**Points only the maintainer can decide.**
1. Whether "consistent, not confirmatory" is enough to act on (Q1), or a confirmatory round (C3) comes first.
2. What bar a parent must meet beyond the pinned rule (Q2): for example K1-K4 pass at the named final.
3. The head-posture ruling: PHASE heads at about 0.90 m, the reward's own full-credit bar, against the statue's 0.96 m.
4. Whether a walking pilot from an uncertified PHASE parent (Option D) is allowed as an exception to GQ-3 (a) and
   GQ-17 (a certified stance before any trex walk).
5. Whether the phase critic lands on main default-off (Option B), and, later, whether it goes default-on.

## 5. Sources read (read only)

- RUNLOG lines 1-971 (searched in full; lines 860-971 read closely; lines 595, 605-607 for history).
- CHAT records 14789-14800 (the question and the answer), found by text search.
- ST: `round/criteria.txt`; `f2/score/score_pc.txt`, `score_pc_dev.txt`; the F2 copies under `round/fetch/F2/`
  (eval ledgers, `provenance.json`, `stage_config.json`); `design/DESIGN_NOTES.txt`; `f2/measure/*/complete_post.txt`;
  `final/PC_REPORT.md`, `final/ALL_EXPERIMENTS_SUMMARY.md`.
- STANCE6 F2 copies under `S1/foot/stance6/fetch/F2/` (the three runs' `provenance.json` and `stage_config.json`).
- `/home/user/work/phase_screen/REPORT.md`; NS6; REVIEW6 §7.3.
- REC §10 (records draft, branch `records-phase-screen`).
- MAIN: `configs/tyrannosaurus_rex/anatomical/{stages,stance,recovery,locomotion}.toml`,
  `behaviors/follow_direction.toml`; `environments/tyrannosaurus_rex/base_env.py`; `environments/shared/base_env.py`;
  BRP (D-D36); GQP (GQ-3, GQ-17, §12).
- BR and RB with `git show`, `git diff --stat`, `git grep`, `git rev-list`: `critic_phase.py`, `critic_phase_ppo.py`,
  `train_base.py`, the two test files.
- Added at finalizing: ST/tools/pc/score_r10round.py lines 505-540 (the parent rule); COLAB_STEPS lines 264-269;
  REVIEW6 lines 395-402 and 818-830; MAIN `reward_functions.py` reward_head_clearance, `curriculum/advancement.py`
  RewardRampCallback, `train_base.py` lines 1086-1097; RB and MAIN `stance.toml` head_clearance lines; the training
  library's `common/utils.py` and `base_class.py` (S2's copy, version 2.9.0).
- W: the three scripts and outputs written for this text (sha256 of the outputs: `head_neck.out` 390d891d...,
  `short_eps.out` 1f7484f0..., `env_fields.out` 13338ae2...).
