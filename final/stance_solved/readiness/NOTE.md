# Readiness to move on from stance (slice: readiness)

**Bottom line.**
- Moving toward walking is reasonable, but "stance solved" is true only for the screen recipe. No certified r10 stance exists on main (`stages.toml` l.34: "no r10 seed yet (bar 2)"). The round named no parent (SCORE l.361). GQP still keeps GQ-3 (a) and GQ-17 open (GQP l.1090) [measured].
- As the code stands, nothing can warm-start walking or recovery from a PHASE checkpoint. Main has no critic-phase code, and BR refuses the load both ways [measured: code text; not run].
- Main's walking reward has no term gated on episode step. Recovery has one on main and four on BR (from step 140), plus pushes at 200 ± 50 steps [measured].
- The shortest path to a first walk is a D-D36-style exception: about 15-30 L4-hours and about 4 days. Each of its six steps needs the maintainer's approval [inferred: NEXT_STEPS costs].

Labels: [measured: source] or [inferred]. MAIN = origin/main 4c9d5e9b. Config paths are under `configs/tyrannosaurus_rex/anatomical/`. Copies of MAIN's code and configs are in `main_src/`; the plan docs were read with `git show`. My script is `action_std.py/.out`.

## 1. The chain after stance

| Node | Warm start from | Source [measured] |
|---|---|---|
| stance | root; "no r10 seed yet (bar 2)" | MAIN `stages.toml` l.29-34 |
| recovery | stance; `recovery_quality/v1` | l.36-41; `recovery.toml` l.118 |
| locomotion | stance; "flips to recovery once a recovery run passes (plan A2)" | l.43-49; BRP l.1068-1069 |
| hunt / follow_direction | locomotion | l.51-57; `behaviors/follow_direction.toml` l.8 |

- Recovery is not a step before walking; both start from stance. These TOMLs and `stages.toml` are byte-identical on MAIN and BR [measured: `cmp`].
- **Plan preconditions.** §12: "The trex now needs a stance retrain (two seeds at today's `certification_seeds`) before any trex walk" [measured: GQP l.700]. The planned `gait-r1` walk also needs PR-G5, PR-G6 and PR-G7 [measured: GQP §6]. Neither `gait_rewards.py` nor `locomotion_gait/v1` is on MAIN [measured: `git ls-tree`/`git grep`, no match].
- MAIN's locomotion gate is still `reward_and_length/v1` [measured: `locomotion.toml` l.88]. That gate passed both September T. rex walkers, and both were two-footed hops [measured: GQP l.86, 89]. So a first walk must be judged by the frozen gait probe plus video [inferred].

## 2. MAIN's stance recipe against RB's

- **MAIN pays one gated term**, `stance_width` (weight 0.1, "settled", step 200) [measured: MAIN `stance.toml` l.135, 165, 188; `base_env.py` l.1149, 1157]. The lean term has the same gate but weight 0.0 [measured: l.277, 1194; no key in `stance.toml`]. MAIN has no stagger or CoP-band code [measured: grep].
- **RB pays four**: width, lean 0.25, stagger 0.30 and band 0.30, all gated on `stance_width_settle_steps` [measured: RB `stance.toml` l.418, 425, 433; BR `base_env.py` l.1278, 1315, 1339, 1421]. RB also sets `symmetry = "mirror_kl"` and the settled leg target; MAIN sets neither [measured: RB l.420, 511]. #655's "four against one" is confirmed.
- **What this means for a PHASE parent on MAIN.** The recipe gap does not block a load. `initialize_next_stage` "records the boundary as lineage" rather than refusing it [measured: MAIN `train_base.py` l.535-537], and the plant files are identical (T. rex physics 10, interface 15) [measured: `cmp` of `plant_versions.toml`]. But no PHASE final can be MAIN's certified stance: that needs a retrain on MAIN (Option E) [inferred; GQP: screens "certify nothing"].

## 3. Loading a PHASE checkpoint

- **On MAIN: it fails** [inferred; not run]. MAIN has no `critic_phase` code [measured: `git grep`, 0 matches]. The checkpoints store `CriticPhaseActorCriticPolicy` and `CriticPhaseRolloutBuffer` [measured: BR `critic_phase_ppo.py` l.323-330]. SB3 2.9.0's member unpickling catches only `RuntimeError, TypeError, AttributeError` [measured: venv `save_util.py` l.166], so the missing module raises at load (MAIN `train_base.py` l.596). The critic also reads 66 inputs, not 63.
- **On BR: refused both ways.** NEXT_STEPS is correct. "Any load -- a resume or a warm start -- whose critic reads another phase, or none, than the stage declares is refused" [measured: BR `train_base.py` l.606-610, enforced l.671-676].
  - Switch off: a phase critic is refused [measured: `critic_phase_ppo.py` l.299-304].
  - Switch on (the notebook sets it "on every node of the chain", l.426-430): recovery reads (140, 0.98) and locomotion (200, 0.995), against stance's (200, 0.98). Both are refused [measured: `recovery.toml` l.80, `locomotion.toml` l.44, `stance.toml` l.396; l.310-311].

## 4. GQ-3 (a) and GQ-17

- **GQ-3 (a)**: "...no trex or robot hunt/follow session; no robot recovery session; ... no walker session on today's rewards (amends G3)." Reason given: "Children inherit gaits" [measured: GQP l.185].
- **GQ-17**: order "Robot, trex seed-42 walk, dibothrosuchus, brachiosaurus"; recommendation "As listed" [measured: l.199]. §12 adds the certified stance [measured: l.700].
- **Nothing amends or retires them.** The last GQP note says "GQ-3 (a) and GQ-17 stay open, and a new pilot is the maintainer's to decide" [measured: l.1090]. BRP's last row, D-D58, does not touch them, and RUNLOG never names them [measured: grep]. They are open recommendations, and each pilot so far was run as a declared "deliberate exception" to them [measured: GQP l.906-915; BRP l.1319]. In that sense they still bind [inferred].
- GQ-3 (a) bars robot recovery, not trex recovery [measured: l.185]. D-D45: "a new pilot is designed once r10 has a stance parent" [measured: BRP l.1328]. None is named yet [measured: SCORE l.361].

## 5. Hidden-clock audit (text only)

- **Walking: none, on MAIN or BR.** `locomotion.toml` sets no width, lean, stagger or band key, and all default to 0.0 [measured: MAIN `base_env.py` l.268, 277; BR l.289, 298, 301, 303]. MAIN's T. rex env reads `_step_count` only at l.850, 1149, 1157, 1194 and 1251; the last is action-penalty bookkeeping [measured]. The forward-velocity ramp runs over training steps [measured: l.161-162]. The 1,000-step limit is a truncation [measured: shared `base_env.py` l.1404].
- **Recovery: yes.** On MAIN, width switches on after step 140 [measured: `recovery.toml` l.80]. On BR, recovery inherits stance's `[env]` (`extends`, l.49), so all four terms switch on at 140 [inferred]. Pushes land at 200 ± 50 steps, and the observation is "deliberately untouched" [measured: l.81; shared `base_env.py` l.1257-1271].
- **Follow**: the command changes every 6.0 ± 1.0 s [measured: `follow_direction.toml` l.18-19]. The policy sees the command, but not when it will next change [inferred].
- The phase critic marks step S only [measured: `critic_phase.py` l.1-30]. In walking, nothing changes at that step [inferred].

## 6. Past walking attempts

- **Pilot 1** (D-D36, r9 parent) read **STILL** on 3 of 3 seeds, 40 of 40 episodes each, at 4,194,304 steps. The parent's action std was 0.095 [measured: BRP l.1319, amendment of 2026-10-18].
- **Pilot 2** was the same with log_std reset to −1.5. It was parked (D-D37), rebuilt on 2026-10-21 [measured: RUNLOG l.250-256, 270], then **retired** on 2026-10-25, never launched, because its r9 parent is refused on MAIN [measured: RUNLOG l.606-607; BRP l.1328].
- **PHASE parents are quieter still.** Mean action std at 12,001,280 steps is 0.0597, 0.0569 and 0.0585 (s60, s42, s80), with `ent_coef` 0 [measured: `action_std.out`, from F2 `policy_probe.npz.copy1`]. A noise reset looks needed [inferred]. MAIN has no `warm_start_log_std` [measured: `git grep`].

## Minimal path to a first walking run

| # | Step | Maintainer approval? | Cost |
|---|---|---|---|
| 1 | Accept "consistent, not confirmatory" (Q1); grant a D-D36-style exception to GQ-3 (a), GQ-17 and §12 for a pilot from an uncertified PHASE parent | **Yes: plan exception** | 0 |
| 2 | Option A: M4 at the three PHASE finals so the pinned rule names a parent (plus M2, critic readouts, s60's falls); maintainer sets the parent bar (Q2) | Yes: go-ahead (owed under CRIT) | 4-8 CPU-h, about 1 day |
| 3 | Never-merged pilot branch on BR: cross-stage rule (drop the critic's 3 phase columns, or a fresh critic), pilot 2's log_std reset ported, V−G-by-step readouts | **Yes: training code** (Option B instead = **code on main**, 1-2 days) | about 1 day with step 4 |
| 4 | Amend pilot 2's criteria for the r10 parent (frozen probe; ≥30 WALK and ≤4 FALL of 40; 2 of 3 seeds; video review) | **Yes: before any training step** | in step 3 |
| 5 | 3 locomotion runs on Colab L4 | **Yes: GPU training** | about 30 L4-h at 8M or 15 at 4.2M; about 1.5 days |
| 6 | Probe, then the maintainer's video review | Yes: review | about 3 CPU-h |

- Costs are from NEXT_STEPS' table [inferred there]. The total comes to about 4 days.
- Any change to the actor's inputs must come before the parent that is kept. GQ-15 says a policy-interface revision that widening cannot carry means "fresh stances everywhere" [measured: GQP l.197].
- **Planned route, with no exception**: Option E (30-60 L4-h, about 1 week), then GQ-17's order (robot first), then PR-G5, PR-G6 and PR-G7. Cost of those PRs: not found.
