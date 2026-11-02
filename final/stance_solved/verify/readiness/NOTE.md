# Verification of the "readiness to move on" note

**Bottom line.**
- The note mostly holds up. Its numbers, quotes, code lines and plan text check out against MAIN, BR, RB, RUNLOG, SCORE and the F2 copies, and none of its conclusions is wrong.
- It does overclaim in one place. "'Stance solved' holds only for the screen recipe" is too strong even for that recipe. What passed there were screen measurements: a gate certificate "measured on a screen task; not certification". The K1-K4 judge (M4) has not run, the pinned reading says "NOT CONFIRMATORY", and THE ONE TEST reads "NOT SEPARATED".
- About six citations are misplaced or loose. The cost of PR-G5 to PR-G7 is not "not found": GQP §6 gives their sizes. One label is inconsistent: the BR recovery count is tagged [measured] in the bottom line and [inferred] in §5.
- It leaves out counter-evidence on action noise. Pilot 1's own action std rose to 0.108–0.111, and it still stood still. Pilot 2's rebuilt parent had 0.064.
- I confirmed the note's load claim more strongly than it could: the PHASE checkpoints do record the critic-phase classes and spec [measured].

Labels: [measured: source] or [inferred]. My scripts and outputs are in this folder (`std_check.*`, `zip_classes.*`). Copies of MAIN files are in `m/`.

## Confirmed claims

**Plan state**
- `stages.toml` l.34 reads "no r10 seed yet (bar 2)".
- SCORE l.361 shows `"named": null` because "an input is absent", the M4 judge record. No parent was named because M4 never ran, not because a candidate failed.
- GQP l.1090 says "GQ-3 (a) and GQ-17 stay open". It is the plan's last paragraph.
- These GQP quotes are correct: GQ-3 at l.185, GQ-17 at l.199, §12 at l.700, and the September hops at l.86 and l.89.
- The walking gate is still `reward_and_length/v1` (`locomotion.toml` l.88).
- [All measured.]

**Behavior chain**
- Recovery and locomotion both warm-start from stance (`stages.toml` l.39, 47). BRP A2 is at l.1068-1069.
- `follow_direction.toml` sets `parent = "locomotion"` at l.8 and the 6.0 / 1.0 s command timing at l.18-19.
- These files are byte-identical on MAIN and BR: `stages`, `recovery`, `locomotion`, `behavior` and `follow_direction`. `stance.toml` differs. BR's `stance.toml` equals RB's.
- [All measured: cmp, diff.]

**Stance recipes**
- MAIN has one step-gated term: width (`stance.toml` l.135, 165, 188). Its lean term is gated too, but its weight is 0. MAIN has no stagger or CoP-band reward code.
- RB has four gated terms: width, lean 0.25 (l.418), stagger 0.30 (l.425) and band 0.30 (l.433). RB also sets the settled leg target (l.420) and `mirror_kl` (l.511). BR enforces the gates at `base_env.py` l.1278, 1315, 1339 and 1421.
- The settled leg target is a constant pose, not a step gate (BR `base_env.py` l.159-180), so "four" is the right count.
- [All measured.]

**Loading**
- MAIN `train_base.py` l.535-537 says warm starts record the boundary as lineage rather than refusing it.
- `plant_versions.toml` is identical on both: T. rex physics 10 and interface 15 (l.608-609).
- MAIN has no `critic_phase` code (0 lines).
- SB3 `save_util.py` l.166 catches only `(RuntimeError, TypeError, AttributeError)`. MAIN's `schedule_custom_objects` never replaces `policy_class` (`policy_loading.py` l.454-499).
- **New measurement.** I read the PHASE s60 checkpoint's zip `data` member at 11,534,336 steps without unpickling it [measured: `zip_classes.out`].
  - It names `environments.shared.critic_phase_ppo` `CriticPhaseActorCriticPolicy` and `CriticPhaseRolloutBuffer`.
  - It records `mesozoic_critic_phase` with `settle_steps` 200 and `gamma` 0.98.
  - OFF s60 stores the stock `ActorCriticPolicy`.
- So "the load fails on MAIN" is sound, from the checkpoint data as well as the code [inferred; not run].

**BR refuses both ways**
- The rule is in BR `train_base.py` l.606-610 and is enforced at l.671-676. The checks are `critic_phase_mismatch` l.299-304 and l.310-311, and the notebook sets the switch at l.426-430.
- The phase spec is (settle step, gamma) (`critic_phase.py` l.81-85). Stance's spec is (200, 0.98).
  - Recovery's is (140, 0.98): the 140 is at l.80, and the gamma is inherited through `extends`, whose only `[ppo]` change is at l.95-98.
  - Locomotion's is (200, 0.995): the 200 is the env default (BR `base_env.py` l.292) and the gamma is at l.44.
- Both specs differ from stance's, so both warm starts are refused.
- [All measured: code text.]

**Hidden clock and history**
- MAIN's T. rex env reads `_step_count` only at l.850, 1149, 1157, 1194 and 1251. BR's stagger and band weights default to 0.0 (l.301, 303).
- Pilot 1 read STILL on 40 of 40 episodes for each of its 3 seeds, from a parent with action std 0.095. Pilot 2 was retired on 2026-10-25, never launched (BRP l.1319 and l.1328; RUNLOG l.606-607).
- RUNLOG never names GQ-3 or GQ-17. On MAIN, BRP mentions them only at l.1216 and l.1319, so D-D58 (l.1341) does not touch them.
- PHASE action std at 12,001,280 steps is 0.0597, 0.0569 and 0.0585, with `ent_coef` 0 (`std_check.out`).
- The note's step costs match NEXT_STEPS' table. Its "about 4 days" is its own sum [inferred].
- [All measured except the sum.]

## Corrections

1. **Overclaim: "'stance solved' holds only for the screen recipe."**
   - What holds is that screen measurements passed in 3 of 3 runs: tail CLEAR, gate certificate PASS, no habit elsewhere (SCORE l.357, 364).
   - M4's K1-K4 has not run at the finals, and head posture (M2) has not been read (SCORE l.361; NEXT_STEPS §1).
   - The reading is "NOT CONFIRMATORY", and the pre-registered test reads NOT SEPARATED, p = 20/20 (SCORE l.357, 362).
   - "Solved" is earned on neither recipe.
   - [Measured.]
2. **Label.** The bottom line tags "four on BR, all switching on at step 140" as [measured]; §5 tags it [inferred]. The text supports the claim: BR `stance.toml` l.25 says recovery inherits that table "all but the keys it writes out". It has still not been run.
3. **Citations.**
   - "Ramp … l.161-162" sits under `base_env.py`, where those lines are a leg-pose comment. The ramp is at `locomotion.toml` l.161-162.
   - "Stance's (200, 0.98) … `stance.toml` l.396": that line is only the gamma. The 200 is at l.188 on MAIN and l.212 on BR.
   - "Parked (D-D37) [RUNLOG l.250-256, 270]": those lines are the rebuild and the push. The parking is in BRP l.1320 and RUNLOG l.173.
   - "No `warm_start_log_std`" and "`locomotion_gait/v1` no match": git grep finds 30 and 13 lines, all in docs. Both conclusions hold for code only.
   - [Measured.]
4. **"Each pilot so far ran" as an exception.** Only pilot 1 ran. Pilot 2 was declared an exception but never launched [measured: BRP l.1319, 1328].
5. **"PR cost: not found" is partly wrong.**
   - GQP §6 l.489-492 gives sizes: PR-G6 "about +600, tests +500", PR-G5 "TOML + tests", PR-G7 one TOML PR per species×stage.
   - It also gives prerequisites: PR-G3, PR-G4, CU-11, CU-12, PR-9 and "pilots" [measured].
   - It gives no hours. The planned route is longer than the note lists [inferred].
6. **"Each of six steps needs approval" is soft for step 2.** CRIT l.270-272 already makes M4 at the finals F2 work once line 4 reads, and line 4 read [measured].

## Missing from the note

- **Pilot 1's noise rose and it still stood still.**
  - Its std reached 0.108–0.111 at 4,194,304 steps, up from the parent's 0.095. Its forward-velocity term earned 0.5–3.2 per episode [measured: BRP l.1319].
  - So low parent noise is not shown to be the cause. NEXT_STEPS Q4 leaves the cause open.
  - "A noise reset looks needed" is pilot 2's untested hypothesis, not a finding.
- **Walking has its own entropy.** `warmup_ent_coef` is 0.02 for 300,000 steps, then `ent_coef` runs 0.005 → 0.001 [measured: `locomotion.toml` l.47-48, 158-160]. The parent's `ent_coef` 0 does not carry over.
- **A closer noise comparator exists.**
  - Pilot 2's rebuilt parent had mean std 0.064 [measured: RUNLOG l.256].
  - The PHASE finals are at 0.057–0.060.
  - The OFF finals are at 0.065–0.070 [measured: the note's `action_std.out`].
- **Option E carries a recipe and code decision.**
  - Stagger and CoP-band reward code is not on MAIN [measured].
  - Certifying RB's recipe therefore means a reward change on main.
  - Certifying main's one-term recipe means using a recipe the phase critic was never trained on at r10 [inferred].
- **A parent that needs no new code exists, but it carries a habit.**
  - Stock-critic finals would need no cross-stage rule. Only OFF s60's zip was read [inferred].
  - The cleanest, OFF s80, has a saturating neck habit [measured: SCORE l.342].
  - GQ-3's "Children inherit gaits" argues against using it.
- **Step 2 is not strictly required.** D-D36's parent had "no gate verdict" [measured: BRP l.1319]. A maintainer-picked PHASE final without M4 is shorter, at more risk [inferred].
- **Pushes recur every 2.0 ± 0.5 s** (`recovery.toml` l.91-92). There is not just one push at "200 ± 50 steps" [measured].
