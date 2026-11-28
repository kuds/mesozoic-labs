# Known Issues & Open Recommendations

The single living list of verified-but-unfixed findings and standing
recommendations, consolidated from the dated code reviews in
[`docs/reviews/`](reviews/). When an item here gets fixed, delete it (the
full context stays in the archived review it came from). When a new review
lands, fold its open items in here and archive the review document.

**Review history:**

| Review | Scope | Outcome |
|---|---|---|
| [reviews/CODE_REVIEW.md](reviews/CODE_REVIEW.md) (2026-03) | Duplication + code quality | Consolidation done in v0.3.0; bugs fixed except thread-unsafe CSV writes (below) |
| [reviews/REPO_REVIEW_2026_06.md](reviews/REPO_REVIEW_2026_06.md) | Full repo: SB3 + JAX RL correctness, sweeps, configs, docs | ~25 verified bugs fixed in PRs #423–#425 |
| [reviews/REPO_REVIEW_2026_07_RL_GCP.md](reviews/REPO_REVIEW_2026_07_RL_GCP.md) | GCP/Vertex integration, SB3/JAX/sweep delta pass, notebooks | ~30 verified bugs fixed in PR #426 (incl. the JAX eval/CLI follow-up pass) |
| [reviews/VELOCIRAPTOR_PLANT_REVIEW.md](reviews/VELOCIRAPTOR_PLANT_REVIEW.md) (2026-07-27) | Raptor plant: anatomy vs published *Velociraptor* material, and mechanics | 11 findings; finding 7 (MJX termination) retired with the JAX/MJX runtime (D-D17, cleanup PR-B); execution was deferred until the T-Rex clears stages 1–3, until the maintainer's D-D25 (2026-10-06): velociraptor physics r3 fixes findings 2, 3 and 5 and replaces finding 1's reward gate with `stance_quality/v2`, and 4, 6 and 8–11 stayed open until velociraptor physics r5 (the maintainer's decision of 2026-11-08, `configs/plant_versions.toml` note 22) closed them all (the review's dated status note); see below |
| [reviews/RL_PIPELINE_GAP_REVIEW_2026_08.md](reviews/RL_PIPELINE_GAP_REVIEW_2026_08.md) (2026-08-28) | SB3 training core, notebooks, env/physics, evaluation, JAX/MJX, configs and sweeps, CI, scripts; cleanup opportunities | 120 ids: 78 fixed (#514–#517, #519, #530, #534, #535), 25 retired by D-D17 (16 of them after a fix), 3 fixed with a residue, 6 duplicates; the 8 open ones and the three residues (SM7, DU4, CI8) are below (its appendix B, 2026-09-30) |
| [reviews/SB3_NOTEBOOK_REVIEW_2026_11.md](reviews/SB3_NOTEBOOK_REVIEW_2026_11.md) (2026-11-22) | The SB3 training notebook, the pipeline behind it, every post-training artifact, the Drive summary, `results/` and the website, across all seven models | 141 verified findings in 30 themes, written before the hunt's removal (D-D59) and the command-driven walk (D-D62). Its status of 2026-11-25 (decision D-D63 (7)) finds 11 addressed, 20 moot and 110 open, each open one with what remains. Re-checked on 2026-11-26 (decision D-D65 (4)), no status changed, and the 110 open ones are folded in here as 52 new entries and 8 amendments to existing ones, each closing with "(2026-11 notebook review: ...)": by the entry that carries it, 1 is HIGH, 31 MEDIUM, 76 LOW and 2 unrated (the Post-training artifacts list has no severities), and 38 sit under Notebooks, 33 Infrastructure, 21 Training / RL, 15 Configs, docs & website, 2 Post-training artifacts and 1 Testing / CI. The review's addendum "Folded on 2026-11-26" records the counts, the amendments and what the re-check corrected, and a search of this file for an id finds the entry that carries it |

Severity: **HIGH** = wrong results in common cases, **MEDIUM** = edge cases /
robustness, **LOW** = cosmetic / QoL.

---

## Training / RL

- **HIGH** — **the walk's `command_tracking/v1` bars are provisional, and no
  controller in the repository has passed a committed walk panel (2026-11-24,
  decision D-D62 (16); updated 2026-11-26, decisions D-D64 to D-D66).**
  Every committed walk follows a live command and is judged by the gate's
  walk profile at `PROVISIONAL_WALK_THRESHOLDS` and the model's gait clause.
  On the real plants the zero-action statue fails every walk on the cruise
  family's success bound (0 of 20).
  - *Resolved by D-D64:* the per-step tracking reading, D-D62 (16) (a),
    which D-D63 (2) decided. At a moving command the tracking term and the
    gate's in-tolerance test now read a trailing one-stride mean
    (`[commands] tracking_window_s`, 0.70-1.50 s by model), under the panel
    protocol `command_panel/v2`; stops and the speed ratio stay per step.
    Under a command it matches, every reference walk now clears the
    in-tolerance fraction and the speed ratio: in tolerance after the settle,
    its worst episode reads 0.873-1.000, against 0.10-0.86 per step. The
    maintainer accepted its boxcar, its one window rule and its window-fill
    settle allowance as recorded on 2026-11-26 (D-D65 (3)); the
    triangular-window fallback stays on record.
    - The brachiosaurus (at 0.35 m/s) and the dibothrosuchus (at 0.157 m/s)
      pass the whole walk gate there. Both commands lie below their
      commanded ranges.
    - The T. rex and the Allosaurus fail `heading_settle` on 20 of 20, on
      their scripted starts. A centred stride mean of their speed reaches
      the speed tolerance at 1.95 s and 2.70 s, so the gate's trailing
      window first reads in tolerance at about 2.5 s and 3.5 s, after their
      2.12 s and 2.31 s allowances. A quicker T. rex start passes every
      episode.
    - On the velociraptor one episode, seed 3045, falls at step 373, and its
      truncated trace also fails the lead-swap bar; the other 19 pass every
      bar.
    - At the dibothrosuchus' and the velociraptor's matched commands, which
      lie inside their velocity tolerances, even the statue reads in
      tolerance; there the speed ratio and the gait clause are what the
      panels demonstrate.
  - *Still open:*
    - No controller passes a committed walk panel. On the brachiosaurus'
      committed panel its steered reference clears every cruise bar only on
      its 5 slowest commands; no controller walks the dibothrosuchus inside
      its commanded speeds; and the velociraptor reference runs at 0.12 x its
      cruise.
    - The Compsognathus pair's tracking sigma lies under an estimate of a
      home-width gait's lateral sway. The window only partly relieves it:
      the robot's home-width gait fits on 14% of plausible strides (D-D64
      (7)).
    - `gait_symmetry` pays a quadruped statue its whole weight.
    - The brachiosaurus reference walk's hind-pair alternation reads 0.786
      on its worst episodes, against the 0.75 bar.
    - A flightless quadruped bound whose feet swing their lead clears every
      quadruped bar (measured on synthetic floor truth).
    - The window's own risks (D-D64 (11)). A learned stride in the boxcar's
      sidelobe (window over stride about 1.2-1.6) keeps up to 0.22 of the
      sway. A stop-and-go whose cycle divides the window earns full
      tracking, and only the report-only `longest_stop_s` sees it, as a
      panel maximum (per episode its values overlap). Whether it becomes a
      bar is decided after the pilot (D-D65 (2)). A quadruped start
      sprint is charged little tracking under the expanding window and the
      +W allowance, and out-earns the steady reference by 0.4-2.1% a step,
      through `gait_symmetry`. Whether such a start overshoot should be
      charged is decided after the pilot (D-D66 (4)). The follow profile's
      mid-episode command changes are unmeasured.

    No weight, sigma or bar was changed for them; the bars are to be frozen
    from a pilot. The in-training evaluation keeps the declared end-state
    commands during the cruise ramp (D-D63 (3)). Candidates are ranked on
    the task that is certified, and the handoff is still chosen afterwards
    on the selection panel.
- **HIGH** — **Completed short stance reward pilots remain uncertified;
  the base T-Rex stance reward is CoP-indifferent, and the velociraptor
  fails by an exploration-driven hop plus an unloaded digit IV.** The
  2026-10-09 Colab batch completed all 15 fresh 1,048,576-step runs
  (three paired training seeds per arm). T-Rex clean episodes rise from
  55/120 to 96/120 with CoP weight 0.10 and safe fraction 0.65 (per seed
  30/5/20 → 31/31/34, n = 3), but its best 40-episode LCB is 0.7253
  against 0.80 and full-horizon counts do not improve. The base T-Rex
  reward does not price pad CoP: on 3042–3081 a full-budget policy on its
  pad's front edge (0/40 clean) and one that passes v2 (39/40) earn the
  same (3783.4 and 3782.2), both about 26 above the statue. Every
  velociraptor arm trained at action std 1.0, far above the statue's
  noise cliff (about 0.07); 312/360 velociraptor episodes end early, the
  policies hop and fall, and digit IV stays unloaded even with both feet
  down, so all 360 fail contact-site duty and coverage. Increasing alive
  support fraction from 0.20 to 0.50 or adding the three-term
  contact-quality bundle yields no clean episode. Both plants' home
  controllers pass 40/40 on calibration and matched evaluation panels.
  Recommendation: full-budget paired T-Rex CoP validation after the
  agreed heading-interface migration, with a settled `leg_home_pose`
  reference as a separate candidate; a paired velociraptor exploration
  sweep (`log_std_init` −1.5, about −2.3 and about −3.0), then
  airborne/impact pricing, isolated per-site load shaping and observation
  ablations. Audit toe preload/compliance/damping only after separating
  raw action saturation from actual force clipping. Preserve the gate and
  existing production training prerequisites. Summarized findings, the
  full-budget reference runs, scope limits and acceptance criteria:
  [STANCE_REWARD_PILOT_FINDINGS_2026_10.md](investigations/STANCE_REWARD_PILOT_FINDINGS_2026_10.md).
  *Update (2026-10-11):* velociraptor physics r4 (the elliptic friction
  cone, D-D30, `configs/plant_versions.toml` note 17) strands every
  velociraptor pilot and both 6M references, so the follow-up trains on r4.
  Its statue is still clean on 40/40 of 3042–3081, but its feet share their
  load a little less evenly and rest digit IV a little less steadily than
  on r3 (support-geom duty at least 0.964 against 1.000; out of sample one
  episode in 240 reads under the 0.35 windowed load-share bar), so the
  digit-IV deficit above has slightly less statue margin to aim at.
  *Update (2026-11-06, the maintainer's decision that day):* the velociraptor half of the recommendation
  is taken as one change, not as the paired sweep: the stance recipe starts
  at action std 0.07 (`log_std_init` −2.66), at the r4 statue's measured
  noise cliff (40/40 full horizon at σ 0.05 and 0.06, 37/40 and 40/40 on two
  seed blocks at 0.07, 32/40 at 0.08, 7/40 at 0.10, 0/40 at 0.135 and 0.22),
  and prices airborne substeps and floor impacts (capped) with the T. rex's
  terms, plus a tail home-pose term; the hop watch, a relative collapse floor
  and the checkpoint archive guard the run. Under the same σ panel the old
  std 1.0 is itself a hop (no foot down on 49% of steps). With the new terms
  the per-step reward, the fall penalty aside, stays positive at every σ
  measured (+2.81 at 0.07, +0.29 at the old 1.0, where it is +0.78 without
  them), so no term pays a noisy policy to fall; why the contact-quality arm
  shortened its episodes is not explained by them. The per-site load shaping,
  the per-site observation channels and the toe audit stay open, and so does
  this entry until a velociraptor stance certifies: nothing has trained on
  the 2026-11-06 recipe, so whether the quiet start reaches the standing basin,
  and whether digit IV loads there, is unmeasured.
  *Update (2026-11-09; the maintainer's decision of 2026-11-08):* velociraptor
  physics r5 (`configs/plant_versions.toml` note 22) rebuilds the foot this
  entry's digit-IV item is about. Each foot stands on an MTP pad and digits
  III and IV, the toes passive on sprung hinges with no actions (the T. rex's
  passive toes, note 9), so the toe audit has no toe actuator left to
  saturate, and the toe springs' preloads carry their static loads (within
  0.23% on digit III and 0.11% on digit IV,
  `test_anatomical_static_balance.py`); the observation carries one
  channel per touch sensor (policy interface r14), the per-site channels this
  entry kept open. The r5 statue stands on pad / digit III / digit IV at
  0.300 / 0.357 / 0.343 of each foot's load, every site at least 3.1 N after
  the settle (r4 by the review's protocol: metatarsus 0.285, digit III 0.547,
  digit IV 0.168). The stance starts at action std 0.10 (`log_std_init`
  −2.30), at the r5 statue's cliff (40/40 to σ 0.09, 38/40 at 0.10, 19/40 at
  0.135, 0/40 from 0.20), and every statue constant was re-measured on it.
  Still open: the per-site load shaping (with the T. rex's LTSB recipe, a
  later change; `docs/NEXT_STEPS.md`) and this entry until a velociraptor
  stance certifies on r5: nothing has trained on it. The settle-peak bar,
  which on r5 the statue under N(0, 0.05) command noise breaks on 30 of 40
  seeds while the impact term prices only from 3.0 body weights, stays at
  2.0 by the maintainer's decision of 2026-11-09, as a settle-window chatter
  screen (it admits σ 0.02–0.03; `stance.toml`).

<!-- The two items below come from the 2026-08-05 stage-1 bounce
     investigation and its addenda; full evidence in
     investigations/TREX_STAGE1_BOUNCE_2026_08.md. Three sibling items
     (actuator saturation unopposed; a filter cannot be retrofitted; the pose
     needs continuous feedback) were deleted 2026-08-15 after the r7/r11
     campaign fixed or falsified each — see the CHANGELOG entry of that date
     for where their evidence now lives. -->

- **HIGH** — **T-Rex stage 1 passes or bounces depending on the run, and we
  cannot yet say which is typical.** Three 10M runs, all seed 42: one passed
  with duty 0.0000, two converged to a phase-locked vertical bounce at exact
  integer subharmonics of the 100 Hz control rate (duty 1/6 = 16.7 Hz, and
  1/5 = 20.0 Hz), both with single-support ≈ 0. **The bounce is not
  reward-preferred** — scored under the same reward it is 450 points *worse*
  than the policy that passed, and loses on every term. Every candidate reward
  tweak (`foot_load_balance_airborne_penalty`, `support_conditioned_alive_fraction`,
  `action_jerk_weight`) is already firing, already correct, and already losing,
  so this is an optimisation failure rather than a shaping one and further
  reward changes are not indicated. **The next experiment should be seed
  replicates of the passing configuration**, which is the only thing that
  distinguishes "solved" from "lucky".
  **Update (2026-08-15):** every number above is r6-era; the plant, interface,
  and reward have all moved since (r7 passive toes, r11 10 Hz command filter,
  the #504 shaping pack), and the campaign ran three more 10M runs — two gate
  FAILs by different mechanisms (18.7 Hz foot chatter at duty 0.41; a
  knee-locked crouch at 0.40) and then the first certified PASS (duty 0.0048,
  UCB 0.0080;
  [investigations/TREX_STAGE1_GATE_PASS_RUN_2026_08.md](investigations/TREX_STAGE1_GATE_PASS_RUN_2026_08.md)).
  The ask stands unchanged with the target moved: seed replicates of the
  **r11 gate-pass configuration**, which is n = 1.
  **Update 2 (2026-08-15, evening):** the first replicate is in — seed 43,
  run `20260815_014118`, identical configuration: gate **FAIL** at duty
  0.0597 / UCB 0.0747, full-horizon 0.925, 0 of 200 panels passing, despite
  a healthy 3093 panel reward. Same three-act trajectory and the **same
  stance** (unsaturated, near-home, tail-live) — the act-3 re-descent
  stalled at a broadband-noise floor (AC 0.329 vs seed 42's 0.135) instead
  of quieting. **n = 2: 1 pass / 1 fail — the configuration is
  seed-sensitive.** A third seed is queued; full record in
  [investigations/TREX_STAGE1_SEED43_REPLICATE_2026_08.md](investigations/TREX_STAGE1_SEED43_REPLICATE_2026_08.md).
  **Update 3 (2026-08-16):** seed 44 (run `20260815_205206`) — **PASS**,
  the strongest yet: full-horizon 40/40, duty 0.0069 / UCB 0.0117, reward
  3408.3 ± 88.5 (97.5% of the statue), AC 0.132, same unsaturated stance.
  **n = 3: 2 pass / 1 fail**, and the split tracks the post-anneal noise
  floor exactly — both passes quieted to AC ≈ 0.13, the one fail stalled
  at 0.33. The open question is no longer whether the configuration can
  pass (it usually does) but what decides the anneal's endpoint; the
  seed-43 postmortem's candidate responses stand.
  **Update 4 (2026-10-06):** the passes were not what they looked like, and
  the configuration they passed is gone. The stance-hack audit
  ([investigations/STANCE_HACK_AUDIT_2026_10.md](investigations/STANCE_HACK_AUDIT_2026_10.md)
  §3.1) found every current T-Rex PASS (seed 42 `20260914_123816`, seed 44
  `20260920_010912` and `20260930_024929`) hopping at reset and standing on
  a propped, rolled pad, which v1 admits. Decision D-D24 then changed the
  plant (physics r7 → r8: hip-roll servos kp 150 → 600), the stance task (a
  settled neck target, pad flatness, stance width) and the gate
  (`stance_quality/v2`, on which those three stances score 0/40), so every
  number above, and every T-Rex checkpoint, belongs to a plant and a task
  that no longer exist. No r8 stance has trained yet: the ask is the same
  one, now for r8 under v2 — how often a seed certifies, at
  `certification_seeds = 2`.
  **Update 5 (2026-10-07):** the first two r8 stances trained (D-D24's
  config byte for byte, seed 42 `20261006_185343` and seed 44
  `20261006_185704`) and **both fail `stance_quality/v2`**, as recorded and
  as re-derived: seed 42 is clean on 0/40, a 12.5 Hz two-foot micro-hop at
  0.80 of the statue's reward; seed 44 on 13/40 (bound 0.204) and on 8/40
  of the fresh seeds 7042–7081, a statue-like stance after a settle stomp
  that re-seats its feet, with a whole-episode hop on about 4% of resets.
  The r7 hacks are gone. The split repeats the pattern above on the new
  plant, one config and two basins: both runs entered a hop regime under
  std-1.0 exploration, seed 44 escaped it between 5.0M and 6.0M and seed 42
  locked in at 6.5M as the entropy bonus ran out at 7M; why is not
  established (n = 2). Decision D-D27 changed what the stance reward pays
  for (the settle re-seat, settle impacts and flight, raw-command chatter,
  unloaded feet), starts exploration at std 0.22 (`log_std_init = -1.5`),
  adds a hop watch that stops a seed still hopping at 7M, and tightens the
  gate ([investigations/STANCE_HACK_AUDIT_2026_10.md](investigations/STANCE_HACK_AUDIT_2026_10.md)
  §9). No T-Rex stance is certified on r8, and the D-D27 task has trained no
  policy past 0.45M steps. The ask stands, on the D-D27 task: at least two
  seeds (plan three), each judged on 3042–3081 and re-judged on 7042–7081;
  whether the quieter start changes the basin split is the first thing those
  runs measure.
  **Update 6 (2026-10-10):** three D-D27 runs have since trained (seed 42
  `20261008_163256`, seed 48 `20261008_163410`, seed 52 `20261009_155723`).
  Seeds 42 and 48 fail v2 (0/40 and 36/40); seed 52 passed on its own
  3042–3081 panel (39/40) but is clean on 32/40 of the re-judge seeds
  7042–7081 (re-rolled from its checkpoint in the 2026-10-10 review), so
  the two-seed ask stands. On 2026-10-11 the maintainer ruled seed 52
  research-only, not a trunk for recovery or walk (the stance-gate HIGH
  below;
  [investigations/STANCE_REWARD_PILOT_FINDINGS_2026_10.md](investigations/STANCE_REWARD_PILOT_FINDINGS_2026_10.md)
  §1.2–§1.3).
  **Update 7 (2026-10-17):** the physics-r9 stance run `20261011_165957`
  (seed 56, 12M steps) fails v2 with one pad on its toe edge (1/40 clean),
  and every screen since has failed its predeclared criteria: lower
  exploration noise, a CoP term, capped leg authority, mirror-symmetric PPO
  with the lateral lean term and the settled leg target (symmetric policies
  that still creep forward; the options are in the library, off by default),
  and the same on the draft physics-r10 foot, where two of three seeds stood
  on their toes. The two-seed ask stands; the next step is a short walking
  pilot (decision D-D36 of 2026-10-17,
  [BEHAVIOR_RECIPES_PLAN.md](BEHAVIOR_RECIPES_PLAN.md) §6.2; launched that
  day, not yet judged). Record, numbers and evidence:
  [investigations/TREX_STANCE_SYMMETRY_AND_FOOT_2026_10.md](investigations/TREX_STANCE_SYMMETRY_AND_FOOT_2026_10.md).
  **Update 8 (2026-10-18):** the walking pilot read STILL on all three seeds
  (each STILL on 40 of 40 panel episodes at 4,194,304 steps; judged
  2026-10-17, final), which by its criteria is too short to tell, and the
  maintainer chose a second pilot with wider exploration (D-D36's amendment of
  2026-10-18; not yet trained, no result). The stage's own six-heading
  certificate, rolled offline on the r9 arm L symmetry-screen checkpoints
  (report only: they were trained in a scratch harness on another task, so the
  judge would refuse them), passes seed 60 (40/40) and seed 52 (38/40) at 6M
  and fails seed 42 (29/40). The passes replicate on fresh seeds, but thinly:
  seed 52 is at the minimum passing count (37/40) on one fresh panel, both 6M
  checkpoints fall on audit seed 7047 where the statue stands, and the
  reward-selected `robust_best` checkpoints are clean on 0 of 40 seeds under
  their own action noise. The fore-aft bar has two blind spots. It reads the
  pad's own CoP, which sits 0.10–0.25 pad half-lengths ahead of the COM
  projection because part of the load rides on the digits at the MTP joints;
  and it is unsigned, bounding |CoP| without telling a toe-ward stance from a
  heel-ward one (seed 60 passes it with the smallest forward push margin of
  the policies tested, and a track-like stance over the MTP joints would fail
  it). Changing either is a gate decision for the maintainer. A foot and ankle
  audit (measurement only) found nothing in the anatomy that pushes the stance
  forward and lists seven candidate physics changes, none decided. The
  two-seed ask stands: the only T-Rex stance trained on a committed physics-r9
  config (run `20261011_165957`, Update 7) was refused, and no seed is
  certified. Record, numbers and evidence: §15 of
  [investigations/TREX_STANCE_SYMMETRY_AND_FOOT_2026_10.md](investigations/TREX_STANCE_SYMMETRY_AND_FOOT_2026_10.md).
  **Update 9 (2026-10-18):** an offline diagnosis (no training) found that
  the committed stance reward pays for a forward lean of about 23 mm through
  its keyframe-centred leg pose term; arm L's settled leg target removes that
  pay but prices a 14 mm lean at only about 0.3% of the reward, and the lean
  lives in the ankles (about 10.7 mm of the centre of mass per degree, so the
  pad bar is about 4° away). On 2026-10-18 the maintainer chose three full
  12M stance screen runs on physics r9 instead of the second walking pilot,
  which is parked (decision D-D37,
  [BEHAVIOR_RECIPES_PLAN.md](BEHAVIOR_RECIPES_PLAN.md) §6.2): arm L on seed
  60, and arm L with the settled leg pose term at weight 1.0 and tolerance
  0.05 rad on seeds 60 and 52, judged by criteria fixed before training that
  add a signed per-foot whole-foot centre-of-pressure bar and a no-drift bar,
  both screen only. They were launched on Colab L4 runtimes that day and are
  not yet judged; as screens they certify nothing. The two-seed ask stands.
  Record, numbers and evidence: §16 of
  [investigations/TREX_STANCE_SYMMETRY_AND_FOOT_2026_10.md](investigations/TREX_STANCE_SYMMETRY_AND_FOOT_2026_10.md).
  **Update 10 (2026-10-19):** the three screen runs were judged by their
  fixed criteria, final, and **all three fail**: arm L on seed 60 on the
  no-drift bar (its panel median ranged over 33.9 mm against 30, an
  ankle-carried whole-body drift) and on joint-space noise (7 and 11 of 20
  falls at 6M and 12M); the heavier, narrower leg term on seed 60 on the gate
  and fore-aft early (one tail actuator pinned at full command, and a
  left-right fore-aft split of the feet left from the spawn stagger); and on
  seed 52 on fore-aft (6 of 40 episodes over the pad bar at 8.4M, 4 outside
  the signed per-foot bar at 12M, a no-drift range of 30.6 mm). Their
  decision table's line is 3b: a joint-space price does not hold the stance.
  What the next round tests is the maintainer's decision, not taken. The
  two-seed ask stands: no T-Rex stance is certified. Record, numbers and
  evidence: §17 of
  [investigations/TREX_STANCE_SYMMETRY_AND_FOOT_2026_10.md](investigations/TREX_STANCE_SYMMETRY_AND_FOOT_2026_10.md).
  **Update 11 (2026-10-19):** the maintainer chose the next round
  (decision D-D39, [BEHAVIOR_RECIPES_PLAN.md](BEHAVIOR_RECIPES_PLAN.md)
  §6.2): three more 12M stance screen runs on physics r9, the heavier leg
  term plus a penalty on a fore-aft stagger of the feet, from scratch on
  seeds 52, 60 and 42, judged against two of the runs of Update 10 by the
  same four criteria, fixed before training. The penalty is proposed for
  committing only if at least two of the three runs hold and the arm does
  no worse than the control on any criterion. The ankle anchor and arm L
  without the heavier leg term are dropped from further screens. As
  screens they certify nothing. The two-seed ask stands. Record and
  criteria: §18 of
  [investigations/TREX_STANCE_SYMMETRY_AND_FOOT_2026_10.md](investigations/TREX_STANCE_SYMMETRY_AND_FOOT_2026_10.md).
  **Update 12 (2026-10-20):** the three stance screen runs of Update 11 were
  judged by their fixed criteria, final, and **all three fail**: on seeds 52
  and 60 on the gate and on fore-aft (21–30 of 40 episodes clean, and 10–19 of
  40 over the pad bar, at every judged checkpoint from 4.2M to 10.5M steps;
  both 39 of 40 clean at 12M), and on seed 42 on all four criteria (the
  actuator-saturation rail failed in most episodes up to 12M, which actuator
  is not recorded, and at 12M 8 of 20 joint-space noise falls and 4 of 40
  roll-rail failures). The penalty had lined the feet up by 4.2M, but on seeds
  52 and 60 the left-right fore-aft split stayed, now as one foot heel-loaded
  and the other toe-loaded. Their decision table's line is 5: the penalty is
  not adopted, and the next round tests a per-foot centre-of-pressure band
  term or the v6b ankle spring. As screens they certify nothing. The two-seed
  ask stands: no T-Rex stance is certified. Record, numbers and
  evidence: §19 of
  [investigations/TREX_STANCE_SYMMETRY_AND_FOOT_2026_10.md](investigations/TREX_STANCE_SYMMETRY_AND_FOOT_2026_10.md).
  **Update 13 (2026-10-20):** the maintainer chose the next round (decision
  D-D40, [BEHAVIOR_RECIPES_PLAN.md](BEHAVIOR_RECIPES_PLAN.md) §6.2): three
  more 12M stance screen runs, from scratch on seeds 52, 60 and 42, of Update
  12's recipe plus a new term that charges a foot whose centre of pressure
  leaves a band of its sole (opt-in, off by default), judged against the runs
  of Update 12 by the same four criteria, fixed before training. The v6b ankle
  spring is held in reserve: if the band fixes the split but runs fail K2 on a
  common fore-aft drift, the next round tests it. Mirrored-leg spawns and the
  prey straight ahead stay diagnostic panels. The term's code, criteria, judge
  and Colab steps are being built separately and are not yet committed
  anywhere public. As screens they certify nothing. The two-seed ask stands.
  Record: §20 of
  [investigations/TREX_STANCE_SYMMETRY_AND_FOOT_2026_10.md](investigations/TREX_STANCE_SYMMETRY_AND_FOOT_2026_10.md).
  **Update 14 (2026-10-22):** the three stance screen runs of Update 13 were
  judged by their fixed criteria, final: seed 52 holds on all four criteria;
  seed 60 fails only the noise test (3 of 20 joint-space noise falls at 12M,
  against 2 allowed); seed 42 fails the gate (the actuator-saturation rail: the
  tail-tip actuator `tail_3_pitch_act` pinned at +1, as on Update 12's seed 42,
  whose actuator an offline diagnosis has since named) and the noise test (13 of
  20). All three pass the fore-aft test, but the arm passes the noise test less
  often than its control. Their decision table's line is 3: the band is not
  adopted, and is to be replicated on two more seeds before any commit. The
  band's code is on a pushed screen branch, which supersedes Update 13's "not
  yet committed anywhere public". As screens they certify nothing. The two-seed
  ask stands: no T-Rex stance is certified. Record, numbers and evidence: §21 of
  [investigations/TREX_STANCE_SYMMETRY_AND_FOOT_2026_10.md](investigations/TREX_STANCE_SYMMETRY_AND_FOOT_2026_10.md).
  **Update 15 (2026-10-22):** the maintainer chose the next step (decision
  D-D41, [BEHAVIOR_RECIPES_PLAN.md](BEHAVIOR_RECIPES_PLAN.md) §6.2): the band
  recipe of Update 14, unchanged, on two more seeds (70 and 80), judged against
  the runs of Update 12 on the same four criteria, written anew and fixed before
  training; the band is adopted only if both new seeds hold (3 of the 5 band
  runs). Not yet trained. As screens they certify nothing. The two-seed ask
  stands. Record: §22 of
  [investigations/TREX_STANCE_SYMMETRY_AND_FOOT_2026_10.md](investigations/TREX_STANCE_SYMMETRY_AND_FOOT_2026_10.md).
  **Update 16 (2026-10-24):** the two replication runs of Update 15 were
  judged by their fixed criteria, final: seed 70 holds on all four criteria;
  seed 80 fails only the gate (the actuator-saturation rail, first flagged at
  4,325,376 steps, its alarm at 4,456,448, recovered by 10.5M; rollouts of its
  4,718,592 checkpoint named the tail-tip actuator `tail_3_pitch_act` at +1, as
  on Update 14's seed 42) and passes the noise test (2 of 20 joint-space falls
  at 12M). Two of the five band runs hold, so the decision table's line is 2:
  the band is not adopted. As screens they certify nothing. The two-seed ask
  stands: no T-Rex stance is certified. Record, numbers and evidence: §23 of
  [investigations/TREX_STANCE_SYMMETRY_AND_FOOT_2026_10.md](investigations/TREX_STANCE_SYMMETRY_AND_FOOT_2026_10.md).
  **Update 17 (2026-10-24):** the maintainer chose the next step (decision
  D-D42, [BEHAVIOR_RECIPES_PLAN.md](BEHAVIOR_RECIPES_PLAN.md) §6.2): a round on
  the tail saturation habit, designed with criteria fixed and approved before
  any training; not yet designed when the copied run log ends (15:27 UTC). As
  screens they certify nothing. The two-seed ask stands. Record: §23.7 of
  [investigations/TREX_STANCE_SYMMETRY_AND_FOOT_2026_10.md](investigations/TREX_STANCE_SYMMETRY_AND_FOOT_2026_10.md).
  **Update 18 (2026-10-24):** the round of Update 17 was designed and
  approved (decision D-D43, [BEHAVIOR_RECIPES_PLAN.md](BEHAVIOR_RECIPES_PLAN.md)
  §6.2): a scratch CPU continuation screen of a training-loss bound on the
  tail-tip actuator's mean, upper side only (no bound, a bound at 0.90 and one at
  0.60), 18 one-million-step continuations from two band checkpoints, judged by
  criteria fixed before any training step; a tail parked below the rail (a
  command of 0.60 or more without saturating) counts as not adopted.
  No-training probes on the same two checkpoints found no reward incentive
  there for the tail-up posture (report only; untested at s42's onset and for
  the discounted objective). Record:
  §24.2–§24.4 of
  [investigations/TREX_STANCE_SYMMETRY_AND_FOOT_2026_10.md](investigations/TREX_STANCE_SYMMETRY_AND_FOOT_2026_10.md).
  **Update 19 (2026-10-25):** the screen of Update 18 was scored, final, and
  independently verified: neither bound is adopted. Under the bound at 0.90
  every replicate had tail-up saturation episodes (s42's from a railed start,
  s80's from a clean one); the bound at 0.60 cleared the rail but left R1/R2
  mean tail-tip commands of 0.60 to 0.63 (s42's above 0.60 by as little as
  0.00017), which the criteria count as parked, not cured. The habit stays
  open; the maintainer takes up the posture options next. As a screen it
  certifies nothing. The two-seed ask stands: no T-Rex stance is certified. Record, numbers and evidence: §24 of
  [investigations/TREX_STANCE_SYMMETRY_AND_FOOT_2026_10.md](investigations/TREX_STANCE_SYMMETRY_AND_FOOT_2026_10.md).
  **Update 20 (2026-10-25):** the maintainer took decision D-D44
  ([BEHAVIOR_RECIPES_PLAN.md](BEHAVIOR_RECIPES_PLAN.md) §6.2): T-Rex physics r10
  (the plant counter's next integer, not the foot that the parked draft pull
  request #623 screened), the three tail pitch springs preloaded (`springref`
  52.5, 25.9 and 10.4°), so the zero-action statue rests its tail level on its
  springs instead of on `tail_1`'s ventral stop, with `tail_home_pose`
  re-centred on that tail and every constant pinned to r9 re-measured; nothing
  trained. The plant contract refuses every physics-r9 T-Rex checkpoint once r10
  is on `main`, so every stance screen run above and walking pilot 2's parent
  stop loading there (none was certified; walking pilot 2 can run only on its r9
  branch). Whether a policy trained on r10 pins the tail-tip actuator again is
  untested: r9 policies rolled zero-shot on r10 gave no sign that r10 cures it
  (their actuator-saturation failures rose from 27 to 80 of 320 deterministic
  episodes; two of eight newly pinned a tail actuator at +1, one lost its rail),
  and zero-shot cannot predict a policy trained on r10. The first r10 round
  needs its own criteria, approved before any training. The two-seed ask stands,
  on physics r10, where no T-Rex stance has trained. Record, numbers and
  evidence: §25 of
  [investigations/TREX_STANCE_SYMMETRY_AND_FOOT_2026_10.md](investigations/TREX_STANCE_SYMMETRY_AND_FOOT_2026_10.md).
  **Update 21 (2026-10-27):** the first r10 round is set up and training
  (decisions D-D45 to D-D51,
  [BEHAVIOR_RECIPES_PLAN.md](BEHAVIOR_RECIPES_PLAN.md) §6.2). Walking pilot
  2 is retired, never launched. The round trains arm LTSB of the stance5
  screens, ported to r10, on seeds 60, 42 and 80 from scratch, 12M steps each
  on Colab L4s, with report-only telemetry (head posture, hip roll, commands,
  body posture, energy, drift and slip, an in-run K3 panel); its criteria,
  approved on 2026-10-27 before any training step, read each run all or
  nothing and test one statistic, the runs with an alarm against LTSB's r9 2
  of 5, a parked tail counting against success. Its three runs were launched
  that day: results pending, so whether a policy trained on r10 pins the
  tail-tip actuator again is still open. The two-seed ask stands. Record:
  §26 of
  [investigations/TREX_STANCE_SYMMETRY_AND_FOOT_2026_10.md](investigations/TREX_STANCE_SYMMETRY_AND_FOOT_2026_10.md).
  **Update 22 (2026-10-28):** the first r10 round was judged by its approved
  criteria, final: line 1, a tail rail formed on r10 in training (decision
  D-D51's amendment, [BEHAVIOR_RECIPES_PLAN.md](BEHAVIOR_RECIPES_PLAN.md)
  §6.2). Of its three runs, all complete, seed 42 pinned the tail-tip
  actuator at +1 (22 training-ledger alarms from 3.9M steps; its final failed
  the gate), seed 60 parked it (a mean command of +0.64 against the 0.60
  line; gate passed) and seed 80 formed a non-tail habit instead,
  `neck_pitch_act` saturated at its negative end with the head raised (gate
  passed: the round's one positive run). So a policy trained on r10 does pin
  the tail-tip actuator again: existence only, 1 of 3, not a rate. The runs
  with an alarm, 2 of 3 against LTSB's r9 2 of 5 (one-sided Fisher p 0.93),
  do not show a lower alarm rate (three runs measure no rate), and on r10
  the habit is not only a tail habit (inferred, from one run). Report only,
  the review's inferred answer to the entropy question: the entropy bonus is
  no direct driver of the rail, its role in when an onset happens untested.
  Next (D-D52): a CPU continuation screen of the entropy coefficient from
  the round's own checkpoints, its criteria to be approved before any
  training. As a screen the round certifies nothing. The two-seed ask
  stands. Record: §27 of
  [investigations/TREX_STANCE_SYMMETRY_AND_FOOT_2026_10.md](investigations/TREX_STANCE_SYMMETRY_AND_FOOT_2026_10.md).
  **Update 23 (2026-10-29):** the CPU continuation screen of the entropy
  coefficient (decision D-D52's amendment,
  [BEHAVIOR_RECIPES_PLAN.md](BEHAVIOR_RECIPES_PLAN.md) §6.2) read, final: line
  3, no difference detected. From seed 42's r10 checkpoint at 3.1M steps the
  tail-tip rail formed again in all 12 continuations, with the entropy bonus
  as recorded or at 0 (6 of 6 each); from seed 80's at 4.2M, 1 of 6 in each
  arm, on the neck. So removing the remaining bonus from mid-rise states does
  not stop the rail; its role in the first 3.1M steps is untested. Report
  only, with no training (D-D53 to D-D55): the rail is a saturated command,
  not a saturated joint or motor (+1 holds the tail tip about 5° up); it is no
  balance aid (on the plant alone it moves the centre of mass 0.2 mm forward
  and 1.6 mm up; held at 0 it gains 63.6 reward per episode and leaves
  the stance otherwise unchanged, the head about 3.5° lower); it
  persists through a learned positive feedback from the observed tail
  angle to the tail command, whose growth the observation normaliser's
  drift does not carry; and no nose-down or forward centre-of-mass shift
  came before the tail rose (the verifier did not re-check the r9 runs),
  though from mid-rise on seed 42's networks read their nose-down
  posture as a tail offset. Why the loop forms in training is open. Everything
  tried on the habit since 2026-07-09 is gathered in
  [investigations/TREX_TAIL_SATURATION_HABIT_2026_10.md](investigations/TREX_TAIL_SATURATION_HABIT_2026_10.md).
  The two-seed ask stands. Record: §28 of
  [investigations/TREX_STANCE_SYMMETRY_AND_FOOT_2026_10.md](investigations/TREX_STANCE_SYMMETRY_AND_FOOT_2026_10.md).
  **Update 24 (2026-10-30):** report-only work on why the loop forms
  (decisions D-D56 and D-D57,
  [BEHAVIOR_RECIPES_PLAN.md](BEHAVIOR_RECIPES_PLAN.md) §6.2). A no-training
  probe found that only the policy-gradient term moves the tail command's
  mean and that, from seed 42's mid-rise checkpoint at 2.1M steps on, the
  critic's bootstrapped value is the source of its upward push; just after
  the onset (at 1.31M, about 0.1M after the tail began to rise) a push
  carried by the tail's own exploration noise is already there, its source
  unresolved. No loop-only early warning separates the runs that rail from
  those that park late. Two scratch CPU screens then hid the `tail_3` angle
  from the critic (the actor still sees it): from 1.05M the first read not
  informative (its controls did not rise enough); from 2.1M the second read
  line 2, reduced: every control continuation ended above every blinded one
  (p 1/70, though a spread-only effect would give such a separation about 3%
  of the time), and two of the four controls had tail over-bar episodes
  where no blinded run had any. So, from that mid-rise state, removing the
  critic's view of the tail angle stopped the climb toward the rail
  (measured on one start over 1.44M steps; the critic as the climb's driver
  is inferred). Next, as the maintainer chose at 16:23 UTC (D-D58): a long
  critic-blind study to 12M steps, its criteria reviewed and pinned before
  any training; pending as of 16:32 UTC, where the run-log copy this update
  draws on ends (what followed goes to the next records change). As screens
  they certify nothing. The two-seed ask stands. Record: §9 of
  [investigations/TREX_TAIL_SATURATION_HABIT_2026_10.md](investigations/TREX_TAIL_SATURATION_HABIT_2026_10.md)
  and §29 of
  [investigations/TREX_STANCE_SYMMETRY_AND_FOOT_2026_10.md](investigations/TREX_STANCE_SYMMETRY_AND_FOOT_2026_10.md).
  **Update 25 (2026-11-06):** the maintainer's stance-landing decision of
  that day (cited by date: the next records change assigns its id) puts on
  `main` the stance recipe whose three r10 phase-critic screen runs passed the
  six-heading certificate: the stance stagger penalty and the per-foot
  centre-of-pressure band (opt-in env terms, the screens' code), the
  lateral lean, the settled leg reference with `leg_home_pose` 1.0 / 0.05
  and mirror-KL PPO, task `1e5964b6`; the collapse reference re-measured
  to 4470.4 and recovery's to 1609.1; the gate and its bars unchanged.
  The nine r10 screen runs never count as certified, as replicates or as
  reusable ancestors (`environments/shared/run_exclusions.py`). No T-Rex
  stance is certified on `main` yet: the next session is two new seeds
  with `CRITIC_PHASE = True` (NEXT_STEPS §3). Record: §30 of the stance
  record above.
  **Update 26 (2026-11-08):** two new seeds certified that stance on
  `main` (`CRITIC_PHASE = True`, commit `820dbda`):
  `trex_stance_r10_ltsb_s44` and `trex_stance_r10_ltsb_s43` each pass the
  six-heading certificate clean on 40/40 seeds at every heading (bound
  0.928) and the fresh-seed audit on seeds 7042–7081 the same way, with no
  hop-watch stop and robust-best handoffs at 10.85M and 11.35M of 12M steps.
  The second's bundle records the stance as replication 2, not provisional,
  so the two-seed ask is met; the first's own bundle, published before the
  second was judged, keeps replication 1 (NEXT_STEPS §2). The stance-landing
  decision stays cited by date in this records change; its number is still
  to be assigned. Record: §31 of the stance record above.
- **LOW** — **stage 1a (stance) contains no in-episode disturbance, so a
  stance-gate PASS certifies stance quality, not active balance control.** The
  only perturbation is joint-angle noise at reset (`reset_noise_scale 0.05`).
  That is by design since the 1a/1b split: the disturbance lives in the
  recovery stage (`configs/tyrannosaurus_rex/anatomical/recovery.toml` — stance's `[env]` plus the
  scheduled pushes, warm-started from the certified stance checkpoint), whose
  `recovery_quality/v1` gate was frozen 2026-08-28 with measured thresholds.
  The pre-split `--impulse-probe` envelope (2026-08-06: 0.50 m/s one way,
  0.00 the other) is superseded by the recovery records —
  [STAGE1B_IMPLEMENTATION_PLAN.md](STAGE1B_IMPLEMENTATION_PLAN.md) and
  [investigations/TREX_RECOVERY_STAGE_FIRST_RUNS_2026_08.md](investigations/TREX_RECOVERY_STAGE_FIRST_RUNS_2026_08.md).
  **Update (2026-09-28):** nor does a PASS always certify a clean stance: the
  certified seed-44 stance `20260920_010912` hops to rebalance in 6 of 40
  episodes (the stance-gate HIGH at the end of this section).
- **MEDIUM (operational)** — **every checkpoint trained before the Phase C
  interface revision is refused as a trunk, every `gate_resolution.json`
  frozen before it is stale, and the two recipes below (the D-A22 re-judge
  and the D-B16 republish) are DEAD for pre-Phase-C runs — the only path is
  the command-line `widen_checkpoint` into a NEW run id that the notebook then
  judges (decision D-D14; the notebook's widen cell and knobs, which the two
  widen sessions used, left with consolidation PR-14a); the one certified
  stance parent that needed widening (`20260815_205206`, seed 44, an r11
  archive two revisions behind r13, hence a revision gap of 2, decision
  D-C17) was widened on 2026-09-20 as `20260920_010912`; seed 42 is already
  certified at r13 by `20260914_123816`** (BEHAVIOR_RECIPES_PLAN §4.6,
  decisions D-C8–D-C14 and D-C17;
  [PLANT_CONTRACT.md](PLANT_CONTRACT.md#widening-a-checkpoint-across-a-policy-interface-bump)
  holds the widen command, the three Colab steps, the identity gate and what
  the tool writes). Phase C appended a 3-dim command segment to every species'
  observation (`policy_interface_revision` T-Rex 12 → 13, velociraptor 9 → 10,
  brachiosaurus 7 → 8, dibothrosuchus 6 → 7, both Compsognathus models
  1 → 2). The task payload carries the plant's
  `policy_interface_sha256` (`task_fingerprint.py`), so every stage's
  `task_sha256` moved with it: reuse rule 3 (the verdict's `task_sha256` must
  equal the fingerprint of the CURRENT stage config) refuses every
  pre-Phase-C verdict, and rule 6 (the checkpoint's recorded plant identity
  must validate against the current plant, no legacy allowance) would refuse
  the archive too (rules run 1, 2, 3, 7, 4, 5, 6). As with rule 7 the refusal
  is a log line, and the node then TRAINS in the new run. Every recovery
  `gate_resolution.json` frozen before Phase C, the 2026-08-28 T-Rex freeze
  included, records a pre-Phase-C `task_sha256`, which
  `require_gate_resolution` refuses ("Recalibrate — re-measure the null
  panels under the current task"), so each is re-frozen from a widened
  handoff. The two recipes die for these runs: the re-judge path in the
  storage cell, where `initialize_result_bundle` raises `run directory already
  belongs to a different run: {'plant_identity': …}` against the session's r13
  identity (the missing `certification_panel` role refuses these two runs the
  same way, D-B17), and, with `provenance.json` removed, in
  `validate_model_plant` before any panel rolls (`backfill_gate_verdict.py
  --force` could at best mint a verdict under the old task hash, which rule 3
  refuses); the republish path in the audit (`stage <N> config
  plant_identity does not match provenance.json`). The remedy: widen on the
  command line before the notebook's storage cell has opened the new run id,
  then run the notebook with `RUN_ID` set to that id, `SEED` to the parent's
  recorded `run.seed` (D-C14) and `TRUNK_FROM = ""`; the chain loop judges the
  widened root before it consults any trunk (D-C13 as amended by ROW-4/6's
  decision 6 (b)) on a fresh 40-episode panel (seeds 3042–3081) under the
  current gate. Never by pointing `RUN_ID` at the old run: for a pre-Phase-C
  run the storage cell refuses the directory, and for a same-plant run whose
  verdict rule 3 or 7 refuses the chain loop raises "mint a fresh RUN_ID"
  (D-C13).
  Inventory: the two certified stance parents are `20260810_145546` (seed
  42) and `20260815_205206` (seed 44) — and BOTH are policy-interface **r11**
  archives (`sha256:96ef13…`;
  [investigations/TREX_STAGE1_GATE_PASS_RUN_2026_08.md](investigations/TREX_STAGE1_GATE_PASS_RUN_2026_08.md),
  [investigations/TREX_STAGE1_SEED43_REPLICATE_2026_08.md](investigations/TREX_STAGE1_SEED43_REPLICATE_2026_08.md)),
  not r12: T-Rex went r11 → r12 on 2026-08-16 (commit `8795e28`, the
  perturbation engine entering the interface fingerprint; `observation_dim`
  stayed 61, physics r7 and `action_dim` 15 unchanged, no tensor moved) after
  both had trained, and an archive's identity is stamped at training time.
  The 2026-09-17 Drive survey
  ([investigations/DRIVE_RUN_SURVEY_2026_09.md](investigations/DRIVE_RUN_SURVEY_2026_09.md))
  found no `gate_verdict.json` in either parent's `stage1/` (the 2026-08
  verdicts live only in the run-level records), and widening needs none (a
  parent verdict, if present, must record `passed = true`). Under the default
  bound `widen_checkpoint` refuses both parents with
  `policy_interface_revision: parent=11, current=13 (gap 2 exceeds
  max_revision_gap=1; pass --max-revision-gap 2 / max_revision_gap=2 …)`, and
  `--allow-legacy-plant` does not apply (it covers only an archive with NO
  recorded identity). **The remedy is decision D-C17**: `--max-revision-gap 2`,
  which for T-Rex r11 → r13 crosses `configs/plant_versions.toml`'s note 11
  (the perturbation engine) and note 12 (the command segment); the widened
  stage's `widen_report.json` then records `revision_gap: 2` /
  `max_revision_gap: 2` and its run block
  `widened_from_policy_interface_revision: 11`; a hand re-stamp of the archive
  is never the path. The three r12 stance PASSes on the log tree —
  `20260816_180603`, `20260817_165515`, `20260818_134249`
  ([investigations/TREX_RECOVERY_STAGE_FIRST_RUNS_2026_08.md](investigations/TREX_RECOVERY_STAGE_FIRST_RUNS_2026_08.md)
  §2.1) — carry r12 identities and widen under the default bound, but they
  are not the certified, published parents; widening one is a first
  certification under the new hash with its own `run.seed`. Replicates count
  only among siblings judged under the SAME `task_sha256`: session 1 of
  [NEXT_STEPS.md](NEXT_STEPS.md) §3 widened the seed-44 parent on 2026-09-20 as
  `20260920_010912` (re-panel identical to the r11 certificate, recovery
  certified 2026-09-21, bundle `complete` with T-Rex stance at replication 2,
  beside `20260914_123816`). The outcome table is
  [investigations/TREX_STANCE_WIDENED_INTERFACE_2026_09.md](investigations/TREX_STANCE_WIDENED_INTERFACE_2026_09.md)
  (its §1–§3 seed-44 columns filled, §7 and §8 appended; its §6, appended
  2026-09-19, marks Session 1 (the seed-42 widen) and Session 2 superseded by
  `20260914_123816`). One neighbour: the two compsognathus recovery
  calibrations were restamped, not re-measured, which moved their
  `profile_sha256`, so every recovery of either Compsognathus model
  freeze made before Phase C is refused and must be re-frozen from the
  restamped profile (`environments/compsognathus_longipes/RECOVERY_CALIBRATION.md`);
  the anatomical profile has since been re-measured on physics r2 (D-D26,
  2026-10-07), which refuses every anatomical freeze made before it.
  **Update (2026-10-06):** moot for the T. rex since its physics revision
  r7 → r8 (D-D24): the plant contract refuses every T-Rex checkpoint, the r13
  `20260914_123816` and the widened `20260920_010912` included, and no widen
  crosses a physics revision, so every T-Rex node retrains from a fresh r8
  stance and its recovery re-freezes on that stance.
- **MEDIUM (operational)** — **SB3 archives are bound to the interpreter that
  saved them; only `policy_loading.load_sb3_model` opens one safely, and the
  Colab image moves without notice.** SB3 stores a model's `learning_rate`,
  `lr_schedule` and `clip_range` members through cloudpickle. A closure — the
  `linear_schedule` every stage TOML with `learning_rate_end` produced before
  2026-09-19, and the nested schedule functions older SB3 releases built for a
  float learning rate — is pickled by value with its code object, and
  `PPO.load` / `SAC.load` execute it while rebuilding the optimizer
  (`_setup_model` → `lr_schedule(1)`). Bytecode compiled by Python 3.12 run by
  3.13, or the reverse, segfaults the process with no Python traceback:
  reproduced both ways in the review container with torch held constant at
  2.14 (`faulthandler` places the crash in the closure body, called from
  SB3's `FloatSchedule.__call__` inside `policies._build`), so torch is not the
  cause the 2026-08-28 note
  ([investigations/TREX_RECOVERY_STAGE_FIRST_RUNS_2026_08.md](investigations/TREX_RECOVERY_STAGE_FIRST_RUNS_2026_08.md)
  §9, "Python 3.13 / torch 2.11") took it for; merely unpickling the data does
  not crash, calling the schedule does. **The incident:** Colab's L4 image
  moved from Python 3.12.13 / numpy 2.0.2 / jax 0.7.2 (the r11 parent
  `20260815_205206`, 2026-08-15) to Python 3.13.15 / numpy 2.1.3 / jax 0.11.1
  (already the image of the runs of 2026-09-14/15, `20260914_123816` and
  `20260915_160239`). Both first attempts at NEXT_STEPS.md session 1
  (`20260919_170528` at `22c1fc8` and `20260919_190251` at `ab35dbd`) died
  with `AsyncIOLoopKernelRestarter: restarting kernel` in the widen tool's
  self-verification, a bare `alg_cls.load` of the r11 parent; two T-Rex
  seed-44 widen sessions of 2026-09-18 at `22c1fc8` (`20260918_230155`,
  `20260918_230335`, on Drive, recorded 2026-09-23) had been the first to
  load an archive saved under the older image and left the same four-file
  stray. **Fixed on
  2026-09-19 (the loader change; its CHANGELOG `[0.3.8]` "Fixed" entry has the
  details):** every archive load in the repository and both notebooks goes
  through `load_sb3_model`, which supplies the schedule members through SB3's
  `custom_objects` instead of unpickling them and refuses an archive whose
  other members carry another interpreter's bytecode; archives saved since
  carry no bytecode; the widen tool re-states a parent's schedules (the
  2026-09-20 widen report of the r11 parent reads `parent_stage_config`); and
  the notebook's load preflight loads a real archive, the trunk run's root
  handoff when there is one, through
  `policy_loading.sb3_archive_load_preflight` since cleanup CU-6
  (`test_policy_loading.py`, with fixture archives saved under 3.12 and
  3.13). **What stays true and is why this entry stands:** every archive on
  Drive trained before that date — every T-Rex and compsognathus stage
  checkpoint, both r11 stance parents and the compsognathus r1 parent
  included — embeds its saving interpreter's bytecode in those three members
  for good, so any path outside the loader (an ad-hoc `PPO.load` in a notebook
  cell, the SB3 CLI, a third-party tool, an older checkout) still dies on a
  foreign image with no traceback; a run's `provenance.json` `python_version`
  names the interpreter its archives belong to, and the image will move
  again. The four stray run directories `20260918_230155`, `20260918_230335`,
  `20260919_170528` and `20260919_190251` hold only `provenance.json` and four
  unverified model files (`robust_best_model.zip`, its `_vecnorm.pkl`,
  `stage1_final.zip` and its `_vecnorm.pkl`, re-pickling the parent's 3.12
  bytecode under a 3.13 `system_info.txt`), with no report, identity or
  fingerprint; nothing can reuse them (`select_trunk` lists them as refused);
  session 1 re-ran beside them on 2026-09-20 without harm, and they are still
  to be deleted (housekeeping in [NEXT_STEPS.md](NEXT_STEPS.md) §3).
- **MEDIUM (operational)** — **run-level records go stale when a run is
  continued in a later session.** The certified r13 T-Rex walker
  `20260914_123816` shows it: its run-level `summary.json`,
  `provenance.json` and `artifact_manifest.json` were written 2026-09-15
  02:08 UTC, after the stance verdict (judged 01:57 UTC), and the
  locomotion verdict judged 11:39 UTC the same day was never folded in
  (only `training_summary.txt` was refreshed), so the run-level records
  under-report the run as stance-only. Nothing canonical breaks: reuse
  reads the per-node files (`gate_verdict.json`, `task_fingerprint.json`,
  `plant_identity.json`, `stage_config.json`), which are written the moment
  a node is judged, so `TRUNK_FROM = "auto"` still selects the run and
  reuses both nodes for a hunt session (before decision D-D59 removed the
  hunt), whose chain runs through both
  (2026-09-17 Drive survey,
  [investigations/DRIVE_RUN_SURVEY_2026_09.md](investigations/DRIVE_RUN_SURVEY_2026_09.md));
  a recover or walk session (a stand session before D-D35) considers
  stance only, which the newer `20260920_010912` also covers since
  2026-09-20, so the tie goes to it.
  Remedy for that run: none in place (corrected 2026-09-23). Its bundle is
  `complete` under a stance target (`target_deliverable "1"`), not
  `partial`; a re-entry that reuses every node never reaches the chain
  loop's `save_run_bundle`, and a direct save is refused because
  `03_locomotion/` appeared after the publication (a complete bundle is
  immutable, [RESULT_BUNDLES.md](RESULT_BUNDLES.md); since consolidation
  PR-14a the notebook refuses, before anything is trained or written, a
  session that would judge or train a node into a complete run). The
  records stay stance-only, with stance at replication 1, although the
  seed-44 sibling `20260920_010912` counts this run at 2; nothing reads them
  for reuse. Remedy in general: a `partial` bundle is rebuilt by the next
  session that trains or judges a node in the run, or, since 2026-11-26, by
  the notebook's publish cell (`PUBLISH_RUN = True`) under the target the
  run records; PR-14a's complete-run refusal does not change this
  ([CONSOLIDATION_PLAN_2026_09.md](CONSOLIDATION_PLAN_2026_09.md) §8).
  *Update (2026-11-26):* a replicate that certifies after a run's last node
  cannot be counted by hand either. `save_run_bundle`'s docstring in the SB3
  notebook's helpers cell said "Re-run this cell once a replicate finishes to
  count it" (`sb3_training.ipynb:939`, before the publish cell), but that cell
  only defines the function: the chain loop (:1683) and the manual cell
  (:2099) call it, and only after a node trains or judges, and a re-entry that
  reuses every node `continue`s before the call (:1539). Fix: a "republish"
  cell after the chain loop that calls `save_run_bundle(chain_results())` (a
  `partial` bundle is rebuilt; a `complete` one regenerates its derived
  artifacts when the replication record is its only change), or a docstring
  that names the cell to run. (2026-11 notebook review: no-republish-cell)
  *Update (2026-11-26, later):* fixed for a `partial` run by the publish
  cell (`PUBLISH_RUN = True`), which rebuilds its bundle and so counts the
  replicate, and `save_run_bundle`'s docstring now names that cell. A
  `complete` run's late replicate still takes a hand-typed
  `save_run_bundle(chain_results(), model_id=MODEL_ID)` once the chain loop
  has re-entered the run, as the cell's refusal of a complete run says.
- **LOW (operational)** — **a re-entry that only reuses a `partial` or
  `failed` run's nodes stops at the bundle-verification cell (verified
  2026-09-24).** In a new runtime the storage cell's
  `initialize_result_bundle` appends the session to the run's
  `provenance.json` (`sessions`), and the zero-action cell refreshes the
  run's `zero_action_baseline.json`; the run's `artifact_manifest.json`
  declares `provenance.json`. With every chain node reused in place nothing
  rebuilds the bundle (only a node trained or judged in the run does), so the
  verification cell's (§8, "Verify the result bundle"; §10 "Cleanup" before
  #558) `validate_result_bundle` raises
  (`artifact size mismatch: provenance.json; summary provenance does not
  match provenance.json`) and "Run all" stops before the auto-disconnect.
  Verified against the library: `initialize_result_bundle` from a new
  process on a run sealed `partial` makes `validate_result_bundle` refuse
  it, while on a run sealed `complete` it records no session and the bundle
  still verifies (and since consolidation PR-14a the zero-action cell keeps
  a complete run's copy). Nothing certified is touched, and the next node
  trained or judged in the run rebuilds the bundle over both files. The
  same holds, since ROW-4/6 (decision 4 (a)), for the run's
  `trunk_run.json` in such a run without ancestor records: the resolve cell
  writes it where the run has none yet and rewrites it when this session
  resolved another trunk, and the next node trained or judged rebuilds the
  bundle over it too. `verify_artifact_manifest` reports only the first
  declared file that differs, before any undeclared one, so in a new runtime
  the cell names `provenance.json` as above (always so for a first write,
  which only a run opened before ROW-4/6 gets). The file's own messages show
  after a rewrite in the runtime that last rebuilt the bundle:
  `artifact hash mismatch: trunk_run.json` when the two values have the same
  length, as two timestamp run ids do, else
  `artifact size mismatch: trunk_run.json`, and a first write alone would be
  `undeclared bundle artifacts: ['trunk_run.json']` (all three verified
  against the library 2026-10-03; the masking executed the same day).
  Remedy: run the auto-disconnect cell (§9) by hand.
  *Update (2026-11-24):* the zero-action cell no longer refreshes an
  unchanged copy: `preflight` (`scripts/zero_action_baseline.py`) rewrites
  `zero_action_baseline.json` only when the run has none or the baseline
  differs in something other than `captured_at` (a new plant, env kwargs or
  score), so on an unchanged plant it is not one of the files that differ;
  `provenance.json` still is.
  *Update (2026-11-26):* the publish cell after the chain loop
  (`PUBLISH_RUN = True`; the maintainer's decision of 2026-11-26) does not
  rebuild the bundle in such a re-entry. In a `partial` run, a session whose
  chain loop reuses every node has a shallower `BEHAVIOR` than the run's
  target (under the recorded one the loop raises on the failed node or
  trains the missing one), and the cell refuses a `BEHAVIOR` whose target is
  not the one the run's `provenance.json` records, before it writes
  anything, since publishing would retarget the run. So the remedy above
  stands for it. Run with the recorded `BEHAVIOR`, the cells above section 5
  and then the publish cell rebuild a `partial` run's bundle over
  `provenance.json` and `trunk_run.json`, and the bundle verifies again
  (`environments/shared/tests/test_publish_cell.py`). A `complete` run is
  refused there, which this entry does not affect.
- **MEDIUM (operational)** — **a node's best-effort artifacts can go missing
  with nothing recording it, and a re-entry in place never regenerates them
  (read from the code 2026-11-26).** `generate_stage_artifacts` writes
  `gate_verdict.json` right after the gate's own report (`_apply_stage_gate`,
  `reporting/stage_artifacts.py:2221`), then the stage summary, the figures
  and replays (rendered into local staging and published together on exit,
  `:2250`) and last the probes (`:2362`); the SB3 notebook's chain loop writes
  `training_summary.txt` and the bundle after it
  (`notebooks/sb3_training.ipynb:1682-1683`). Each artifact step only logs its
  failure: one `try` covers the four older figure calls
  (`stage_artifacts.py:2256-2302`) and one both replays, so a failed selected
  replay skips the final one (`:2437-2541`; the renderer entry under
  Infrastructure). Nothing lists what a node should hold:
  `artifact_manifest.json` hashes whatever files are present
  (`result_bundle/manifest.py:140-176`), so a bundle without a replay or a
  figure verifies. And a runtime lost between the verdict and the bundle
  leaves a passed node that the reuse rule accepts on its verdict, digests and
  handoff pair (`find_certified_ancestor`, `ancestors.py`): a re-entry in the
  same `RUN_DIR` takes its results from the verdict (the `same_run` branch,
  `sb3_training.ipynb:1511-1519`) and regenerates no figure, replay, probe or
  bundle. A later node's bundle then seals the incomplete set, which verifies;
  with no later node, the chain loop rebuilds no bundle and the verification
  cell stops "Run all" (the entry "a re-entry that only reuses a `partial` or
  `failed` run's nodes ..."), and the opt-in publish cell
  (`PUBLISH_RUN = True`, since 2026-11-26) seals the same incomplete set.
  Fix: write a per-node `artifact_status.json` of the files expected for its
  gate kind, model and role (the table the artifact-guide entry under
  Configs, docs & website asks for) against those present, state it in the
  stage summary, and have the `same_run` branch regenerate what is missing
  and then the bundle. (2026-11 notebook review:
  verdict-without-bundle-unrecoverable, silent-artifact-failures)
- **LOW (operational)** — **a re-entry that is not a resume, under another
  trunk than the run's `trunk_run.json` records, is only warned (ROW-4/6;
  executed against the notebook cells 2026-10-03).** Since ROW-4/6's
  decision 4 (a) the resolve cell records the trunk a session resolved in
  `RUN_DIR/trunk_run.json`, as the `TRUNK_FROM` value that reproduces it,
  and keeps that record once the run holds an `ancestors/` record, the trunk
  those records came through (when the cells ran in order; below). The
  RESUME cell refuses, before it trains, a resume under another trunk and a
  resume of a node the run holds as an `ancestors/` record. A session that
  does not resume gets a WARNING from the resolve cell naming the recorded
  `TRUNK_FROM`, and then, as before: under another trunk that certifies a
  recorded node the chain loop refuses a second parent for it
  (`record_ancestor`: "a run cannot reuse two parents for one node",
  `ancestors.py:777-780`) before it trains anything there, and under
  `TRUNK_FROM = ""` it trains the nodes this run holds only as records again
  here (found by the review of #559's first round, `ee91f51`; review 2 in
  CLEANUP_PLAN_2026_09.md §5.4), as it does a recorded node that another
  trunk does not certify. A run opened before ROW-4/6 that holds records has
  no `trunk_run.json` (the resolve cell never records a guess there and says
  so), so a resume there is still pinned by section 5's manual route: the
  trunk the interrupted session printed, or the run the
  `ancestors/<id>/ancestor.json` record nearest the interrupted node names
  (a shallower record names the run that certified that node, not
  necessarily the trunk). A wrong pin there still fails late
  (`record_ancestor` refuses only after the RESUME cell trained) or, under
  `TRUNK_FROM = ""`, trains the recorded ancestors again. The record is only
  as good as the order the cells ran in (found designing ROW-4/6; executed
  against the notebook cells 2026-10-03; the command-line route by reading
  `train_base.py`): only the resolve cell writes `trunk_run.json`, and only
  while the run holds no record, and the chain loop re-derives `TRUNK_DIR`
  only under `"auto"`, and then from the kernel's last selection. So
  ancestors a session recorded without running the resolve cell under that
  trunk came through a trunk the file does not name: the storage cell re-run
  with another pinned `TRUNK_FROM` and the resolve cell skipped (a stale
  kernel), an older notebook copy on this library, or the command-line
  curriculum given the run as its output directory. The resolve cell's
  WARNING and the RESUME cell's refusal then name that stale value: a resume
  under the trunk the records came through is refused, one under the
  recorded trunk fails late (`record_ancestor` refuses a second parent after
  the resume trained), and under a stale `""` following them trains the
  recorded ancestors again here, with no warning. Remedy: set `TRUNK_FROM`
  as the WARNING says and re-run sections 2-3 together, never the chain loop
  alone after a `TRUNK_FROM` change. If the records came through another
  trunk than the file names, in a run that is not `complete`, remove the
  run's `trunk_run.json`: the resolve cell then records nothing (it never
  guesses), and a resume is pinned by section 5's manual route from the
  `ancestors/<id>/ancestor.json` record nearest the interrupted node, not
  from the trunk the resolve cell printed. The run's manifest still declares
  the removed file (`manifest artifact is missing: trunk_run.json`) until
  the next node trained or judged rebuilds the bundle. (#558 reviews)
- **LOW (operational)** — **the chain loop's JUDGE branch applies no chain
  check (reuse rule 4) to the target, or to any node when no trunk is set
  (found reading the code for ROW-4/6; executed against the notebook cells
  2026-10-03).** A node trained here is judged on whatever parent its
  `stage_config.json` records, even when this session resolved that parent
  elsewhere: the target, for example, after its parent's own verdict was
  refused (a gate edit, say) and a trunk's copy of the parent was reused and
  recorded. The bundle's lineage check passes it, because it hashes the
  in-run parent that `load_path` names (`reporting/bundles.py:694-718`).
  ROW-4/6's `result_bundle.refuse_judging_off_the_resolved_parent` covers
  only a node the loop judges ahead of a trunk (decision 6 (b)); calling it
  in the JUDGE branch would close the rest, and is a separate change.
- **LOW (operational)** — **`train --output-dir` or `curriculum
  --output-dir` into a run whose bundle is `complete` is not refused when the
  stage directory holds no verdict.** Neither `train_base.train` nor
  `train_base.train_curriculum` has a complete-bundle refusal
  (`train_base.py` reads no bundle status): a new stage directory, or an
  unjudged one, written into a `complete` run adds files its manifest never
  listed, so the bundle stops verifying. `curriculum` also writes the run's
  `plant_identity.json` first, and then every node the run does not hold as
  its own stage directory (a trunked run whose stance is only an
  `ancestors/` record trains the next node there). The notebook refuses this
  before it writes
  (`result_bundle.refuse_complete_run_session` in the resolve cell,
  `refuse_write_into_complete_run` in the chain, RESUME and manual cells).
  Workaround: never point `--output-dir` into a run whose bundle is
  complete; a new attempt is a new run directory.
  **Fixed on 2026-11-24 (the D-A20 amendment; the cleanup plan's §2 row 5):**
  the judged case this entry was opened for (guard executed 2026-09-26: `train
  --load <checkpoint>`, default `--load-mode resume_same_stage`, into a
  directory holding `gate_verdict.json` re-saved `stage_config.json`, always
  rewrote the final pair, and rewrote the handoff pair the verdict hashes
  whenever a post-resume evaluation beat the seeded best). The D-A20 guard
  `config.refuse_occupied_stage_dir` now refuses a directory that holds
  `gate_verdict.json` under every load mode, the same-stage resume included,
  so the CLI and the library refuse it as the RESUME cell has since D-D16;
  the notebook's manual-node cell, which let `MANUAL_LOAD_MODE =
  "resume_same_stage"` through to that guard and then rewrote the verdict,
  refuses a judged target before it writes anything. The website's recipes and
  quick-start pages state the judged case. `test_write_guards.py` pins all
  three. (#558 reviews)
- **MEDIUM (operational)** — **every `gate_verdict.json` written before the
  gate-configuration digest (decision D-A22, Phase B WS-B3) is refused as a
  trunk until it is re-judged.** Reuse rule 7 compares the verdict's
  `gate_sha256` (the digest of the gate it was judged under) with the gate
  the reusing run declares; a verdict without the field certifies an unknown
  gate and `--trunk-from` / `TRUNK_FROM` refuses it naming the re-judge
  paths — and then TRAINS the node in the new run (the refusal is only in
  the log: `Not reusing 'stance' from --trunk-from ...: ... Training it in
  this run instead.`). **Superseded for every pre-Phase-C run (2026-09-14):**
  the two re-judge paths (the notebook JUDGE branch for a directory holding no
  verdict, `backfill_gate_verdict.py --force` for a pre-D-A22 verdict) and the
  digest measurement owed before the first Phase-B trunked run are dead for a
  checkpoint minted under the previous policy interface, which covers the
  certified T-Rex stance run `20260810_145546` and the seed-44 replicate
  `20260815_205206` (the 2026-09-17 Drive survey found no `gate_verdict.json`
  in either `stage1/`); see the Phase C entry above. The widened copy is
  re-paneled under the current gate in a new run, which settles the digest
  question for it. The recipes, kept as history until cleanup CU-17, are in
  this file at commit `20ab100` (`0.3.9`). What stays true for a same-plant
  directory: a moved rail or a statue re-measure is a RE-JUDGE of every
  certified trunk, never a retrain, except in a run whose bundle is
  `complete`, which is never re-judged in place (a trunked run then trains
  the node; D-B8 as amended 2026-11-24) (decisions D-B7/D-B8; stance's
  `min_avg_reward` rail moved 1940.0 → 2100.0 on 2026-08-10, the day
  `20260810_145546` ran); `--gate current` re-aggregates a
  `reward_and_length/v1` directory's evidence rows under the checkout's block,
  but a `stance_quality/v1` verdict is read off `stance_gate_report.json`,
  which certifies only the thresholds it scored, so the tool refuses a report
  scored under an old rail under either `--gate` (a pass at 1950 under the
  1940 rail must never be minted as a pass under 2100) and a stance directory
  is re-judged through the notebook JUDGE branch; recovery verdicts cannot be
  backfilled and need a notebook re-roll.
- **MEDIUM (operational)** — **pre-Phase-B stance bundles need republishing
  with the `certification_panel` seed role, and pre-D-A22 siblings join a
  deliverable's replication count only after re-backfill** (decisions
  D-B16/D-B17, Phase B WS-B4; plan §4.5). The publication audit binds
  `stance_panel_selected.csv` row `i` to `seed_roles.certification_panel + i`
  and a recovery `gate_resolution.json`'s `decision_procedure.panel_seed_start`
  to the role, and REFUSES a recorded `stance_quality/v1` pass whose
  provenance declares no such role. No committed bundle is affected, but
  every Drive bundle published before this change — the certified T-Rex
  stance run `20260810_145546` (the bundle
  `TREX_STAGE1_GATE_PASS_RUN_2026_08.md` audits) and the seed-44 replicate
  `20260815_205206` — carries a `seed_roles` without it and fails its next
  audit or republish; `initialize_result_bundle` also refuses to resume such
  a run directory from the new notebook (the seed roles are identity, and
  the mismatch reads as "already belongs to a different run"). **Superseded
  for every pre-Phase-C run (2026-09-14):** the same-plant republish recipe is
  dead for these two runs (the storage cell now mints an r13 provenance the
  audit rejects against their r11 stage configs; it is in this file at commit
  `20ab100`, `0.3.9`, kept as history until cleanup CU-17); the seed-44 run is
  widened into a new run instead and the seed-42 widen is superseded by the
  fresh r13 run `20260914_123816` (the Phase C entry above), whose bundles
  carry the `certification_panel` role from the start. Replication is
  writer-recorded from `LOG_BASE/<species>/<embodiment>/<algo>/` siblings whose verdict
  carries `gate_sha256`. For the r13 pair (2026-09-21): the
  widened seed-44 run `20260920_010912` and the fresh seed-42 run
  `20260914_123816` share the r13 stance `task_sha256`, the seed-44 bundle
  records replication 2, and the seed-42 run's own records stay at
  replication 1: its `complete` bundle cannot be rewritten in place (the
  MEDIUM entry "run-level records go stale" above; a complete bundle is
  immutable, [RESULT_BUNDLES.md](RESULT_BUNDLES.md)).
- **LOW** — **the recipe digest leaves out `n_envs`, so runs on different
  environment counts count as one recipe (read from the code 2026-11-26).**
  `hyperparameters_sha256` hashes the algorithm, its `[ppo]` or `[sac]` table
  and the shaping keys (`environments/shared/config.py:950-969`); `n_envs` is
  recorded beside the digest, never in it (the stage config's run block,
  `train_base.py:2440`, :3597; the command line's `curriculum_results.csv`
  row, :3895; `training_summary.txt`'s "Parallel envs" line,
  `reporting/text_summaries.py:213`). It changes how the recipe trains: PPO
  collects `n_steps` × `n_envs` transitions per update (8,192 at the
  notebook's `N_ENVS = 4` on a 2048-step stance, 16,384 at 8), and SAC counts
  `train_freq` in vector steps. Seed-replicate discovery matches siblings on
  that digest (`replication.py:224-229`), the trunk scan counts a node's
  replicates through it (`ancestors.py:1057-1077`), and a same-stage resume
  refuses only a change of digest (`config.refuse_resume_recipe_change`,
  `config.py:1074`). So a replicate on another `N_ENVS` counts toward a
  stance's `certification_seeds`, and a resume continues under another batch
  with no warning. The SAC entry under Notebooks (the notebook's 4 SAC
  environments against the command line's 8) is one case of it. Fix: put
  `n_envs` in the digest's view (that moves every recorded digest, so it needs
  its own decision), or refuse or flag a replicate, a trunk count or a resume
  whose recorded `n_envs` differs. (2026-11 notebook review:
  n-envs-outside-recipe-digest)

<!-- The items below come from the 2026-07-31 plant validation pass; full
     evidence in PLANT_VALIDATION_AND_STAGE1_OBJECTIVE.md. The reset and
     self-collision defects that pass also found are FIXED in PR #479 and so
     are deliberately not listed here. Two further entries — the stage-1
     statue-optimum-cannot-be-reward-gated finding (§9) and "stage 1's real
     gate machinery does not exist yet" (§12/§14) — were deleted 2026-09-05:
     their ask shipped as the stance_quality/v1 gate kind
     (environments/shared/curriculum/gate_schema.py, stance_gate.py), which
     certified the first GATE: PASS on 2026-08-11
     (investigations/TREX_STAGE1_GATE_PASS_RUN_2026_08.md). -->

- **HIGH** — **`foot_load_balance_min_support_force = 0.0` makes the §7.1
  airborne repair a no-op.** `derive_stance_info` and `reward_foot_load_balance`
  differ only in the near-zero branch, yet produced bit-identical output over all
  709 logged rollouts (`max |diff| = 0.000000`, correlation `1.000000`) spanning
  30–67% unsupported duty. The sum of two touch-sensor readings is essentially
  never exactly `0.0`, so the airborne branch never fires. The duty metrics use
  `> 0.1 N` per foot, so a foot at 0.001 N is *unsupported* in the diagnostic and
  *supported* in the reward. Needs a real threshold and a **monotone** ordering
  (airborne strictly worse than single support, not equal). `derive_stance_info`
  still scores true-airborne as perfect balance for the same reason.
  (PLANT_VALIDATION §11.1)

- **MEDIUM** — **`smoothness_weight` penalises action-delta magnitude, not
  frequency, and cannot see a high-frequency limit cycle.** From the 7/31 run's
  best to final checkpoint, `action_delta` *fell* 12.0 → 10.5 and the smoothness
  penalty *improved* −0.286 → −0.250, while toe-motion power above 4 Hz
  **doubled** 35% → 71%. The policy got smoother by the metric while getting
  buzzier in fact. Needs a contact-switch-rate cost or smoothness on the second
  difference of actions. (PLANT_VALIDATION §11.2)

- **MEDIUM** — **`collapse_peak_floor` is still an absolute reward value on
  the brachiosaurus and dibothrosuchus stance configs and cannot survive a
  reward-function edit.**
  The T-Rex is done: `stance.toml` and `recovery.toml` use the relative
  `collapse_peak_floor_fraction` (PLANT_VALIDATION §14 item 3) after the
  absolute floor failed to arm twice. The anatomical `velociraptor_mongoliensis`,
  `brachiosaurus_altithorax` and `dibothrosuchus_elaphros` `stance.toml` still carry absolute floors
  (1300 / 1300 / 1950, each commented "Absolute pending §14 item 3") derived
  from their statues' standing reward; re-derive them as fractions the same
  way. (PLANT_VALIDATION §11.4) **Update 2026-09-23:** an absolute floor at
  0.75 x the statue also sits below the UNTRAINED policy (action 0 is the
  nominal stance under home-keyframe-residual/v1), so without a peak
  warm-up the backstop arms on initialisation: dibothrosuchus run
  `20260923_020654` stopped both nodes at 1.45M (stance armed on the 2592.9
  statue plateau; locomotion on the 2246.9 standing level, 22x its absolute
  floor of 100). `collapse_peak_warmup_timesteps` now separates the two on
  dibothrosuchus and brachiosaurus stages 1-2 (1.0M on stance; on
  locomotion the D-B5 bound `warmup_timesteps + ramp_timesteps`, 3.3M and
  4.0M, conservative since the two shaping schedules run concurrently),
  replayed on that run's series in
  `test_curriculum_early_stopping.py`. Still open: the four floors stay
  absolute (convert them to the relative pair, with the
  `statue_constants_physics_revision` pin, once the planned dibothrosuchus
  and brachiosaurus plant updates have settled; the locomotion statues
  re-measured 2026-09-23 are 2196.9 +/- 217.7 and 2242.7 +/- 7.8,
  `zero_action_baseline.py --model-id <species> <embodiment> --stage 2 --episodes 40 --seed 3042`), and
  velociraptor stage 1 has the same shape with no warm-up (its session-3 run
  `20260922_125248` trained the full 6M by its trajectory, not by protection).
  **Update 2026-10-06:** velociraptor physics r3 (D-D25) re-derived its floor
  as 0.75× the r3 statue's 2842.76, 1300 → 2130, still absolute.
  **Update 2026-10-11:** the elliptic friction cone (D-D30, note 17) moved the
  statues inside every rounding: velociraptor r4 2843.00, brachiosaurus r5
  1738.38, dibothrosuchus r2 2598.53, so their stage-1 rails and floors
  (1710 / 2130, 1040 / 1300, 1560 / 1950) keep their values; the
  locomotion statues read 2197.0 ± 217.8
  (dibothrosuchus) and 2201.3 ± 200.6 (brachiosaurus, median 2244.9: two of
  40 episodes lose part of `gait_symmetry` to a briefly unloaded foot).
  **Update 2026-11-06** (the maintainer's decision that day): the
  velociraptor is done too. Its stance uses the relative pair (0.45 of the
  re-measured 3091.3 statue, pin 4) behind a 1.0M peak warm-up, and its
  locomotion the pair (0.45 of its r4 statue's 796.00 ± 3.58) behind the
  D-B5 warm-up of 2.3M, replayed on the dibothrosuchus series scaled to that
  statue in `test_curriculum_early_stopping.py`. Still open: the
  brachiosaurus and dibothrosuchus floors stay absolute. On Brachiosaurus physics
  r6 the stage-1 statue returned 1749.61 ± 0.15 over seeds 3042–3081. The
  recalibrated minimum reward is 1050 and the absolute peak floor is 1310.
  **Update 2026-11-24 (decision D-D62):** the walks follow a live command, and
  the forward-velocity ramp keys left them, so the locomotion D-B5 bound is
  now `warmup_timesteps + cruise_ramp_timesteps`: 2.8M on dibothrosuchus
  (800k + 2M, from 3.3M) and 3.0M on brachiosaurus (1M + 2M, from 4.0M),
  with the velociraptor's 2.3M unchanged and a new 2.3M on the T-Rex walk
  (D-D62 (12), (15)). The velociraptor walk's floor is 2.4 x its live-walk
  statue's 891.3 (D-D62 (15)); the two quadruped walks keep the absolute 100.
  **Update 2026-11-26 (decision D-D65 (1)):** the quadruped walks' floors
  are relative too. On the matched basis (the reference walk under a command
  matched to its own speed) each walker is expected to evaluate above its
  stander's 10-stand level, so the brachiosaurus walk's floor is 1.20 x
  2359.7 = 2831.6 and the dibothrosuchus walk's 1.31 x 2286.0 = 2994.7, each
  between its stander's 9- and 10-stand evaluations, with the
  `statue_constants_physics_revision` pin (7, 4). A standing plateau no
  longer arms them, so a collapse from standing is not caught on these
  walks. On the dibothrosuchus, neither, on most seeds, is the collapse of a
  learned walk that tracks none of the committed commands; on the
  brachiosaurus such a walk evaluates within a few units of the floor, so
  ordinary fluctuation between evaluations usually arms it.
  **Update 2026-11-26 (decision D-D66):** the maintainer kept the
  dibothrosuchus walk's fraction at 1.31, the D-D62 (15) midpoint, rather
  than 1.32, so the rule stays as written and the same on every model. A
  stander arms it on 1.13% of seeds with a run's episodes fixed (0.82% at
  1.32), which is harmless: a steady level never stops a run. A learned walk
  that tracks none of the committed commands, whose collapse the
  dibothrosuchus floor mostly misses, is accepted as within the trade.
  `collapse_drop_fraction` stays 0.5 on the four walks with a backstop, so
  an armed run stops once 10 evaluations in a row fall below half its peak,
  and a return to standing after walking can do that: from the matched
  reference walker's level it stops the run on the T. rex at 2 or 3 stands
  of the 30 evaluation episodes (27.4% of evaluation seeds), on the
  velociraptor at 6 or fewer (84.7%, the default seed 42 included), on the
  dibothrosuchus at 2 or fewer on the family means (15.1%; 17.2% with each
  run's own episodes fixed, some of them at 3 or 4 stands) and on the
  brachiosaurus at none (computed on the committed means and replayed
  through `EvalCollapseEarlyStopCallback`). A walk that scores higher stops
  on at least as many seeds. Whether to change the drop fraction is decided
  after the pilot, once learned walks show how high they score. Still open:
  the brachiosaurus and dibothrosuchus stance floors (1310, 1870) stay
  absolute.
- **LOW** — **an early stop by the collapse backstop is invisible in the run
  records (verified 2026-09-23).** Run `20260923_020654` stopped both nodes
  at 1,450,000 steps; `stage_config.json` keeps the budget in `run.timesteps`
  (6,000,000 / 12,000,000), `gate_verdict.json` records
  `stage_result.timesteps = 1450000`, and no field names a stop or its
  reason (`collected_results.csv`'s `gate_reason` / `gate_evaluable` are
  empty). The callback logs the arming and the stop, but only to the
  session log; the cause was established by replaying `evaluations.npz`
  through `EvalCollapseEarlyStopCallback`. Fix: record `early_stopped`,
  the stop step and the armed peak in the stage result.
- **LOW** — **the zero-action baseline watch compares the recovery nodes with
  the stance statue, and calls a statue-level stance a failure (read from the
  code 2026-11-26).** A run keeps one statue reward: the SB3 notebook's
  preflight (cell 13) measures stage 1 only (`BASELINE_STAGE = 1`,
  `notebooks/sb3_training.ipynb:763`) and copies the record, with its `stage`,
  into `RUN_DIR/zero_action_baseline.json`
  (`scripts/zero_action_baseline.py:385`, :417-432), and
  `read_zero_action_baseline` (`curriculum/baseline_watch.py:54-80`) takes no
  stage and never reads that field. Since decision D-D62 a live-command stage,
  the walk, skips all three readers (the watch,
  `build_baseline_progress_callback`, :206-217; `progress.json`'s
  `zero_action_reward`, `curriculum/progress_report.py:467-480`; the
  gate-progress "zero action" line, `visualization.py:909-919`), but the
  recovery nodes of the T. rex and both Compsognathus models run no command
  and still read the stance statue, so all three compare a pushed recovery
  task with standing still. On a stance whose reward the statue all but maxes
  (the brachiosaurus' and dibothrosuchus' `reward_and_length/v1` stances,
  whose rail it clears by design, and the anatomical Compsognathus', where no
  near-statue wins its reward back,
  `configs/compsognathus_longipes/anatomical/stance.toml:41-53`) the watch's
  tolerance-free `latest >= self.baseline_reward` test
  (`baseline_watch.py:137`), against a statue scored on the preflight's seeds
  3042-3081 rather than the in-training evaluation's own seed stream
  (`train_base.eval_env_seed`, `train_base.py:1644`), cannot be met by a
  statue-level policy, and past 35% of the budget it warns once, also into
  `progress.json`'s warnings, that the run is "in a worse-than-trivial local
  optimum" like one that "failed its gate" (:153-178), beside a gate the
  stance passes. On the Allosaurus, whose statue falls within a second (86-91
  steps of 1000 on seeds 3042-3044, measured 2026-11-26), the comparison is
  vacuous. Advisory only: no gate, verdict or checkpoint choice reads it. Fix:
  have the reader take the stage and refuse a record of another stage, so
  every node but the one measured skips the watch and the line; compare on the
  same seed stream with a tolerance, and on a stance the statue passes by
  design report "statue-level (expected)" instead of a failure.
  (2026-11 notebook review:
  baseline-watch-uses-stance-statue-for-every-node,
  stage1-baseline-on-all-nodes,
  brachiosaurus-stance-reward-ceiling-no-headroom,
  compsognathus-baseline-warning-on-statue-optimal-stance,
  allosaurus-statue-baselines-vacuous-for-torque-plant)
- **MEDIUM** — **the quadruped gait-symmetry reward pays a motionless
  statue its full weight on every step (verified 2026-09-23).**
  `BaseDinoEnv._compute_quadruped_gait_symmetry` (`base_env.py`) scores the
  alternation ratio of a touchdown history that never decays: a statue logs
  one touchdown per diagonal pair at reset, the ratio is 1.0 from then on,
  and `weight x 1.0` is paid every step. On the locomotion stages it is
  most of a typical do-nothing episode: dibothrosuchus 1994.0 of 2244.2
  (`gait_symmetry_weight` 2.0 x 997 steps, 89 percent), brachiosaurus
  2200.0 of 2244.0 (2.2 x 1000, 98 percent), measured with
  `zero_action_baseline.py --model-id <species> <embodiment> --stage 2`. Standing still is therefore a
  strong local optimum of the walk stage, and dibothrosuchus run
  `20260923_020654` judged a stand-still locomotion checkpoint
  (2249.86, 0.0012 m/s; gate FAIL on the forward rail). In two of 40
  dibothrosuchus statue episodes (seeds 3045, 3056) one diagonal pair touches
  down twice while the statue settles after reset (by step 3-4), the history
  freezes at AAB / BBA, the ratio is 0.5 for the whole episode and it scores
  about 1248. The biped `_compute_gait_symmetry` shares the
  history-ratio shape (not measured here). A reward change moves the
  task fingerprint, so it belongs with the planned plant and reward work on
  these species, not with a backstop setting.
  **Update (2026-09-28, gait audit):** the biped `_compute_gait_symmetry` has
  been measured: a synchronous two-foot landing appends `"R"` then `"L"`
  (`base_env.py:768-771`), so a bounce scores 1.000, as do a true alternating
  walk and the statue ([STAGE1_SPLIT_PLAN.md](STAGE1_SPLIT_PLAN.md) §6
  item 7). Its weight is 0.0 in every biped stage, so only its
  `alternation_ratio` diagnostic misleads. The quadruped version likewise pays
  a pronk, bound or pace in full, since a simultaneous landing appends both
  pairs (:868-871; [gait audit](investigations/GAIT_AUDIT_2026_09.md) §3). The
  gait plan keeps the code byte-identical and sets the weight to 0 in the
  quadruped `gait-r1` revisions ([gait plan](GAIT_QUALITY_PLAN_2026_09.md)
  §5.3, PR-G7); this entry stays while any TOML uses the term.

- **LOW** — **contact-switch rate conflates bilateral↔single with
  bilateral↔airborne.** The PR #479 plant repair moved T-Rex's raw switch count
  *up* (0.86 → 1.00 /s) while unsupported duty went to **zero** — the extra
  switches are ordinary weight-shifting. Do not gate on it until decomposed;
  gate on unsupported duty instead. (PLANT_VALIDATION §11.3)

- **LOW** — **four stage-1 reward terms are saturated and contribute no
  gradient**: `head_clearance` pinned at exactly its full 0.350 weight in every
  measured window, `height` 0.578 of 0.6, `neck_posture` 0.173 of 0.2,
  `leg_home_pose` 0.312 of 0.5. (PLANT_VALIDATION §14)

- **LOW** — **collidable necks are deferred until terrain lands.** Velociraptor
  is the reference: its neck geom collides *and* sits in `_body_ground_geoms`,
  so hitting the ground with it terminates the episode. The other three carry
  `contype=0` necks (plus cosmetic `brow_ridge` / `crest` / `sagittal_crest`
  and dibothrosuchus' twelve `scute`s), and brachiosaurus documents the choice
  explicitly, using the collidable head as the termination proxy. On a flat
  floor this is unobservable — an animal whose neck reaches the ground has
  already tripped tilt, height or head-contact termination — so the decision
  was to leave physics alone and revisit when heightfield terrain arrives, at
  which point the raptor's pattern is the template. Newly-colliding long neck
  capsules must be checked for home-pose self-collision (the defect class
  fixed twice in the PR #480 series). The
  cosmetic geoms should stay non-collidable permanently; they are already
  excluded from the ground-settle probe. On behavior terrain, T. rex's
  non-colliding neck is probed through `_terrain_contact_probe_geoms`
  (#556, consolidation PR-7) and ends the episode as `neck_ground_contact`;
  brachiosaurus and dibothrosuchus can opt in by declaring their neck geoms
  there, without changing canonical physics.
  *Update (2026-11-09; the maintainer's decision of 2026-11-08):* the
  velociraptor is no longer the colliding-neck reference. Physics r5
  (`configs/plant_versions.toml` note 22) gives it a separate neck and skull
  on the T. rex's pattern: its two neck capsules never collide (contype 0),
  `skull_upper` and `snout` do and sit in `_body_ground_geoms`, and the neck
  capsules are its `_terrain_contact_probe_geoms` on behavior terrain, as the
  T. rex's neck is; the head-tip and skull height terminations (0.07 m and
  0.28 m, every physics substep) end a head dip on the flat floor before any
  head geom touches it. The T. rex's probe pattern is now the template for
  terrain.

- **MEDIUM** — **the policy saturates its action bound, and the
  `diagnostics/action_*` family mixes pre-clip and post-clip quantities.**
  Measured on T-Rex stage-1 run `20260727_130726` (PPO, 6.0M steps), from its
  own tensorboard:

  | `diagnostics/` scalar | first | last | mean | max | measured on |
  |---|---|---|---|---|---|
  | `action_abs_max` | 4.55 | 6.71 | **6.11** | **7.77** | pre-clip |
  | `action_std` | 1.00 | 1.93 | 1.42 | 1.93 | pre-clip |
  | `action_saturation` | 0.32 | **0.68** | 0.49 | 0.68 | pre-clip |
  | `action_delta` | 21.27 | 6.24 | 15.04 | 21.43 | post-clip |

  `action_saturation` (fraction of components at or beyond 0.99) rose
  **monotonically** across all 6M steps — 0.32 → 0.68. The policy's raw output
  peaks at 7.77 against a `Box(-1, 1)` action space. Note `train/std` *fell*
  over the run (1.00 → 0.72) while empirical `action_std` nearly doubled: the
  policy is pushing its **mean** out of bounds, not widening its exploration.
  The table predates the 2026-08-15 rename (CHANGELOG `[0.3.8]`): its
  `action_saturation` row is the pre-clip quantity now named
  `diagnostics/raw_action_saturation`, and `diagnostics/action_saturation` is
  the env's filtered-command ramp fraction (the policy's own clipped command
  on the T. rex stance and recovery since D-D27, `action_penalty_source =
  "raw"`).

  **What this is not: the reward is not inflated.** SB3 clips before
  stepping the environment, so `_get_reward_info` receives an in-bound
  action and both penalties are computed on it (SB3
  `on_policy_algorithm.py:214-218` clips immediately before `env.step`, and
  `policies.py:379` inside `predict()`, which both eval loops use). So no
  `min_avg_reward` gate needs re-deriving, no historical reward comparison is
  invalidated, and the `r ∝ w^-0.16` smoothness fit in
  [TREX_LEG_FLEXING_PLAN.md](TREX_LEG_FLEXING_PLAN.md) stands (an earlier
  version of this entry claimed the energy term was inflated ~3.8×
  (~209/episode, ~7.9% of return) and the fit unsound; retracted by the
  2026-07 T-Rex telemetry review, whose full retraction is in this file at
  commit `20ab100`).

  **The metric hazard.** `action_mean`, `action_std`, `action_abs_max`,
  `action_abs_mean` and `raw_action_saturation` come from
  `DiagnosticsCallback._on_step`'s read of `self.locals["actions"]` — SB3's
  **pre-clip** Gaussian sample. `action_delta` is returned by
  `reward_action_smoothness` from *inside* the env, so it is computed on the
  **post-clip** action. Reading the group as one space is an easy and
  consequential mistake.
  *Update (2026-11-26):* the offline `scripts/action_bound_report.py --model`
  makes the mistake the other way: it steps `model.predict(obs_input,
  deterministic=True)` (:140), which SB3 has already clipped to the box
  (`policies.py:379`, as above), and prints it under "Raw (pre-clip) action
  statistics" (:165), so its |a| > 1, abs max and E[a²] lines read in bound
  for every policy, however far the policy's mean sits outside the box. Only
  its untrained-Gaussian mode (no `--model`) sees pre-clip values. Fix: take
  the pre-clip lines from the policy's distribution mean and keep `predict()`
  for the trajectory. (2026-11 notebook review:
  fu-offline-analysis-tools-species-coverage-action-bound-model-mode-measures-clipped-actions)

  **What is real.** PPO stores the raw action and its `log_prob`, while the
  environment responds to the clipped one. With 68% of components saturated,
  most of the policy's output distribution sits where moving the mean further
  changes the plant not at all — the standard bias from sampling an unbounded
  Gaussian into a bounded action space, and a plausible contributor to the
  bang-bang envelope measured on the same run (`used` = 100% of range on 20 of
  21 actuators). Worth watching `raw_action_saturation` as a first-class health
  metric rather than as evidence about the reward.

  **Latent trap — now scoped to filter-free species.** `BaseDinoEnv.step`
  passes the raw `action` to `_get_reward_info` while `_scale_action` clips
  separately on the way to `ctrl`. Harmless under SB3 today, but any direct
  caller — a notebook, a custom rollout loop, a diagnostic script — is
  silently charged energy and smoothness penalties for magnitude the plant
  never sees. Since r11 this applies only to species with
  `action_filter_cutoff_hz = 0` — today, every species except the T-Rex:
  `_filter_action` clips before the reward terms read the action, so the
  T-Rex's two paths already agree. Clipping once at the top of `step` for the
  filter-free species would close it everywhere and moves no fingerprint.
  (2026-07 T-Rex telemetry review)

  **Update 2026-10-29:** on the T-Rex since physics r9 the habit is one
  actuator at a time, almost always `tail_3_pitch_act` commanded to +1 in
  deterministic play, and it fails the stance gate's saturation key without
  falls; on physics r10 it persists through a learned positive feedback from
  the observed tail angle to the tail command. Everything tried on it since
  2026-07-09, and what is open, is gathered in
  [investigations/TREX_TAIL_SATURATION_HABIT_2026_10.md](investigations/TREX_TAIL_SATURATION_HABIT_2026_10.md)
  (the T-Rex stance entry's Update 23 has the latest).

  **Update 2026-10-30:** on the T-Rex the policy-gradient term alone moves
  the saturating tail command, and from mid-rise on the critic's
  bootstrapped value pays for raising it: in a scratch CPU screen from seed
  42's checkpoint at 2.1M steps, hiding the tail angle from the critic (at
  its input only) kept the tail command far below the bound while every
  control climbed toward it. A long critic-blind study was chosen next
  (pending as of 16:32 UTC that day); see §9 of
  [investigations/TREX_TAIL_SATURATION_HABIT_2026_10.md](investigations/TREX_TAIL_SATURATION_HABIT_2026_10.md)
  (the T-Rex stance entry's Update 24 has the latest).

- **LOW** — `BaseDinoEnv.reset` still applies one `reset_noise_scale` scalar to
  the whole of `qvel`, which mixes root linear velocity (m/s), root angular
  velocity (rad/s) and joint velocities (rad/s). This is the same
  units-conflation that made the root-height jitter wrong, but far less severe:
  a velocity kick has to actually defeat the controller, whereas the height
  jitter could spawn an episode already outside `healthy_z_range`. Worth
  separating if a species much smaller than Dibothrosuchus is ever added.
  (2026-07 Dibothrosuchus review)

- **LOW** — the T-Rex and Dibothrosuchus SB3 envs accept
  `foot_contact_weight` / `foot_contact_gate` (`tyrannosaurus_rex/base_env.py:195-196`,
  `dibothrosuchus_elaphros/base_env.py:123-124`) and no SB3 reward reads them: they were
  knobs of the MJX reward, which left with the JAX/MJX runtime (D-D17, cleanup
  PR-B). They stay, with their six `[env]` keys (`configs/tyrannosaurus_rex/anatomical/stance.toml:13-14`,
  `configs/tyrannosaurus_rex/anatomical/recovery.toml:77-78`, `configs/dibothrosuchus_elaphros/anatomical/stance.toml:13-14`),
  because `task_sha256` hashes the constructor defaults overlaid with `[env]`:
  removing either moves the T-Rex and dibothrosuchus task digests (stripping the
  dibothrosuchus keys moved its certified stance task from `083e2966` to
  `abfb339f`, measured 2026-09-26). Do not delete them
  ([CLEANUP_PLAN_2026_09.md](CLEANUP_PLAN_2026_09.md) §7). A typo'd weight
  still does nothing; a loud rejection of unknown env kwargs must keep these two
  names. (June §1.6)
- **MEDIUM (perf)** — **in-training evaluation runs serially, at up to 0.8
  evaluation steps per training step on a stance, and no stage can set its
  cadence (re-read from the code 2026-11-26).** Every `eval_freq` steps
  `EvalCallback` rolls max(30, `min_eval_episodes`) episodes
  (`train_base.py:152`, `:239-253`) one after another on one evaluation env
  (`:1788-1798`): 40 on five of the seven stances (Allosaurus, both
  Compsognathus models, T. rex, velociraptor) and on the three recoveries, 30
  on the brachiosaurus and dibothrosuchus stances and on every walk. The SB3
  notebook passes `train_stage`'s default `eval_freq=50_000`
  (`notebooks/sb3_training.ipynb:856`, a line of the `.ipynb` file), with no
  knob, and no stage TOML has a key for it. On a 1,000-step stance that is up
  to 40,000 evaluation steps per 50,000 training steps, up to 8.8M over an
  11M-step stance at 40 episodes (6.6M at 30); the Allosaurus walk's
  3,000-step episodes make it up to 90,000 per 50,000. After training
  `evaluate_stage_checkpoints` rolls up to 60 more episodes, and the command
  line's `curriculum` adds a 10-episode supplementary evaluation at every
  evaluation (`CurriculumCallback`, `curriculum/advancement.py:231-270`),
  which the notebook and `train` do not run. The evaluation-count keys assume
  50,000 steps an evaluation (`collapse_min_evals = 20` is "~1M steps" in the
  T. rex, velociraptor and quadruped walks' `locomotion.toml`, "20 evals x
  50k" on the quadrupeds; `required_consecutive`), so another cadence must
  convert them. Fix: a per-stage cadence key that `train_base.train` reads and
  `stage_config.json` records, with an `EVAL_FREQ` knob in the notebook, or a
  subprocess-vectorized evaluation env; the gate panels keep their size. (June
  §6.1; 2026-11 notebook review: eval-cadence-dominates-wallclock)
  *Update (2026-11-28, decision D-D79 (4)):* the cadence is 100,000 steps
  (`constants.DEFAULT_EVAL_FREQ`) in the notebook, the command line's
  `--eval-freq` default and `train_base`, which halves the evaluation steps
  (0.4 per training step on a 40-episode stance, about 4.4M over 11M), and
  `stage_config.json` records it as `run.eval_freq`. `collapse_min_evals`
  (20 → 10) and `collapse_patience` (10 → 5) moved with it on the eight
  stance and walk configs that declare them, so the backstop keeps its 1M and
  500k step spans; `required_consecutive` (a gate key) and the advisory
  posture-watch counts did not. Still open: a per-stage cadence key, and the
  serial evaluation env.
- **MEDIUM** — **under SAC, `diagnostics.npz` gains a row every few
  transitions and is rewritten whole every minute, and SAC's exploration
  reaches no report (read from the code 2026-11-26).** `DiagnosticsCallback`
  appends one row to every series at each `_on_rollout_end`
  (`diagnostics.py:310-337`). SB3 calls it once per PPO rollout (`n_steps` ×
  `n_envs`: on 4 envs, 4,096-8,192 transitions on a stance) but under SAC
  after every `train_freq` vector steps: every 64 transitions on the
  velociraptor, T. rex and dibothrosuchus stances, every 16 on the
  brachiosaurus and both Compsognathus models, every 4 on the Allosaurus
  (SB3's default `train_freq` of 1). Over a stance, 11M steps on every model,
  that is about 172,000 rows (the velociraptor, T. rex and dibothrosuchus) to
  2,750,000 (the Allosaurus), against about 1,300 under PPO (about 2,700 on
  the two Compsognathus stances, whose `n_steps` is 1024). Only the rewrite
  is throttled: the whole history is saved uncompressed
  (`file_io.atomic_savez`, `np.savez`, `file_io.py:171-182`) at most every
  60 s (`diagnostics.py:230,459`), on the Drive mount in the notebook. Rows
  that dense cannot be compared with a PPO run's per-rollout means either. And
  nothing a report reads shows SAC's exploration:
  - the per-actuator std pages read a `log_std` tensor
    (`reporting/training_history.py:964-978`) or `policy_probe.npz`, which
    builds no bank and writes no row without a rollout buffer
    (`policy_probe.py:518-523`); under SAC's state-dependent std they are
    empty and say "policy.pth has no log_std (not a Gaussian PPO policy)"
    (:1065) or "no leg actuator std: no policy_probe.npz std_per_actuator and
    no checkpoint zip's log_std read" (:5573), not that SAC has no global std;
  - SAC's entropy coefficient is captured as an `algo_` series and charted
    nowhere (`visualization.py` reads no `algo_` key;
    `scripts/compare_run_diagnostics.py:41-46` lists PPO's);
  - `progress.json` gets no `policy` block, which only a probe row fills
    (`policy_probe.py:836`).

  The review's fourth point, `K3_PANEL = True` switched off under SAC with an
  INFO log alone while the configuration cell printed "In-run K3 panel: on",
  is gone with the in-run K3 panel, retired on 2026-11-26 (decision D-D48's
  retirement note). Fix: record the diagnostics on a fixed step cadence (for
  example one row per 8,192 transitions) for both algorithms and write them
  compressed; read SAC's std from the actor's `log_std` head on a fixed state
  bank, or mark those pages "not applicable to SAC"; chart the entropy
  coefficient. (2026-11 notebook review: sac-diagnostics-density,
  fu-sac-artifact-path-sac-exploration-unobservable)
- **Watch item** — the biped home-residual action mapping has a
  slope discontinuity at action 0 (the two piecewise segments span
  home→min and home→max, which are unequal), so zero-mean Gaussian
  exploration produces a physically biased mean command away from home
  wherever the home control is off-midpoint. The listed Velociraptor effect
  is hip pitch ≈ −0.29 rad toward flexion, knee ≈ −0.16, ankle ≈ +0.13 at
  σ=1; the T-Rex lower-body home controls are midpoint-aligned, so its
  asymmetry is limited to neck/head controls. If a fresh run's early training
  looks persistently off-home while `algo_std` is still ≈1.0, suspect this
  bias before suspecting rewards; possible mitigations (smaller
  `log_std_init`, a smoothed mapping) are interface experiments and must be
  run in isolation. (Stage-1 basin investigation follow-up)
- **MEDIUM** — **the direction/terrain pilots are a second pipeline
  (#540/#541).** The 2026-09-17 review found that the pilots delivered new
  content (a direction controller, a tracking reward, a heightfield terrain
  generator, a replay recorder with terrain maps) beside every canonical
  concept instead of through it: a behavior env class that bypasses the
  reserved `BaseDinoEnv._draw_episode_command` hook and writes
  `self._command` directly (consolidation PR-7 deleted the second,
  `TRexBehaviorEnv`); a second checkpoint preparer
  (`environments/shared/behavior_checkpoint.py`); a second PPO trainer
  (`environments/shared/train_behaviors.py`, one CPU env) with its own
  recipe dialect; 66 behavior TOMLs under `configs/<species>/<embodiment>/behaviors/`
  (11 templates × 6 models); a second gate outside `GATE_KINDS`
  (`configs/behavior_certification.toml`, judged by `judge_behavior_panel` and
  reached only from tests; never a `gate_verdict.json`); a checkpoint identity
  keyed on source-file hashes (any edit to `behavior_env.py` strands exact
  resume) — about 7,000 lines of modules at 22c1fc8, tests excluded, with the
  notebook `COMMAND_TERRAIN_BEHAVIOR` mode switch that 19 of the notebook's 22
  code cells referenced. Pilot outputs
  (`logs/<species>/ppo/behaviors/<behavior>/<run-id>/`) are evaluation-only
  (decision D-D9): none is reusable through `find_certified_ancestor` or
  carried forward as a training parent. The fold-back — manifest nodes under
  `locomotion`, the controller as the body of the reserved hook, one generic
  opt-in terrain env subclass, a registered gate kind, the second trainer
  deleted — is
  [CONSOLIDATION_PLAN_2026_09.md](CONSOLIDATION_PLAN_2026_09.md)
  PR-3..PR-15, released 2026-09-20 in the notebook-first order of D-D13; its
  status table says which of them have landed (the certified library, the
  `[pilot]` dialect, the notebook mode switch with its knobs and
  `behavior_notebook.py` are gone). Until PR-11 adds the follow and terrain
  manifest nodes, the pilots run from the command line only
  (`python -m environments.shared.train_behaviors`, D-D13), with
  [TRAIN_DIRECTION_AND_TERRAIN.md](TRAIN_DIRECTION_AND_TERRAIN.md) as the
  operator guide and an explicit `--checkpoint` / `--vecnormalize` pair in
  every load mode and for evaluation.
  **Update (2026-09-28, gait audit):** the certificate has no contact
  criteria. Its rules (`configs/behavior_certification.toml`) are survival,
  success, commanded speed and yaw-rate tracking, settling and course
  progress, so the per-episode statistic PR-13 registers as
  `terrain_command/v1` (D-D6) would certify a hop, scoot or slide that tracks
  the command ([gait audit](investigations/GAIT_AUDIT_2026_09.md) §3). The
  gait plan recommends a per-episode gait clause in that kind
  ([gait plan](GAIT_QUALITY_PLAN_2026_09.md) PR-G10; GQ-18, open).
  **Update (2026-10-05, consolidation PR-9):** the behavior env's commands
  now come through the reserved hook and `step()`'s `_update_command()`
  (`BaseDinoEnv` owns the controller), and a behavior checkpoint's identity
  is its task fingerprint, whose versioned stage (`command-terrain/v2`)
  replaces the source-file hashes, so an edit that changes no meaning no
  longer strands exact resume; the rest of the second pipeline stays until
  PR-10 .. PR-13. **Update (2026-10-05, consolidation PR-10):** the
  preparation's command-column zeroing and its probe are now the canonical
  warm start's (`policy_loading.neutralize_command_columns` and
  `assert_command_blind`); the rest stays until PR-11 .. PR-13.
  **Update (2026-11-23, decision D-D60):** the tracking reward, its metrics,
  `info` keys and dwell now live in `BaseDinoEnv`, active only under a live
  command (the behavior env inherits them; the nine pilot captures re-run
  exactly), and a stage TOML can declare the command task in a `[commands]`
  table, so a manifest node can train what a pilot trains. No committed stage
  declared one then; the second trainer, the recipe dialect and the second gate
  stay. `prepare_behavior_checkpoint` refuses a walker trained on live
  commands. **Update (2026-11-24, decision D-D62):** every committed walk now
  declares `[commands]`, so no recipe can start from a walker trained since;
  the pilot `follow_direction` recipes are superseded by the planned follow
  nodes, and the terrain recipes need a command-free walker on the current
  plant interface. No such walker exists: that interface is the revision of
  `configs/plant_versions.toml` note 24 (2026-11-23), in this same unreleased
  series, every committed walk declares `[commands]`, and
  `--override commands.*` is refused, so no committed path produces one. Until a follow or terrain node
  with a command exists, or a command-free walk config is added (a
  `locomotion.toml` without `[commands]` and with a gate the in-training
  manager runs), the 63 terrain recipes have no parent a user can produce.
  The refusal is right: a command-trained walker under a command-free recipe
  sees a zero command, which its walk was trained to stand still on.
- **HIGH (terrain blocker)** — **every certified walker survives the plane and
  falls on a flat heightfield (measured 2026-09-25).** An eval-only run (seed
  1, nothing trained or written under `logs/`) of the `robust_best_model` pair
  each node's `gate_verdict.json` names — trex `20260914_123816/03_locomotion`,
  velociraptor `20260922_125248/02_locomotion`, compsognathus
  `20260921_203149/03_locomotion` — on four direction/terrain recipes kept all
  23 plane episodes per walker at full horizon. On the `terrain_contact`
  family, an all-zero heightfield (`mode = "flat"`, `terrain.py:391`) with the
  plane's geometry, full horizon fell to 1/13 for T-Rex (fallen 11,
  head_contact 1), 0/13 for velociraptor (tail_contact 3, fallen 1; the other
  9 left the map) and 0/13 for compsognathus (excessive_tilt 12, body_contact
  1); the sloped, bumps, depressions and mixed families gave 3/24, 0/24 and
  0/24. Cells are the committed 200 mm (T-Rex), 137.5 mm (velociraptor) and
  20 mm (compsognathus); T-Rex at 100 mm fell 7/7, although its zero-action
  statue survives 3/4 there, so statue results do not predict the walker. The
  cause is open. The two precompiled scenes
  (`SpeciesBehaviorMixin._select_contact_model`, `behavior_env.py:248`) give
  the `floor` geom the same `solref`, `solimp`, friction, margin, gap and
  condim and differ only in its type (plane vs hfield; checked 2026-09-26).
  That leaves the hfield collision and the terrain spawn settle
  (`behavior_env.py:397-415`, which reproduces velociraptor's authored −44.6 mm
  toe penetration) as the suspects. Compsognathus-pair foot contact also
  flickers on a heightfield (a statue measurement from the 2026-09-25 readiness
  review). Velociraptor's map exits are a separate mismatch: its recipes cruise
  at 2.0 m/s, the walker runs at 3.64–3.66 m/s, and the maps fit 25 s at
  cruise. No terrain pilot or terrain node should start from these parents,
  and consolidation PR-11 copies the terrain keys verbatim. Plan: a dated
  heightfield-contact investigation before PR-11, decided together with the
  recipe speeds and map sizes
  ([CLEANUP_PLAN_2026_09.md](CLEANUP_PLAN_2026_09.md) §2 and §5.1). The T-Rex
  walker no longer loads since T-Rex physics r8 (D-D24, 2026-10-06), nor the
  velociraptor walker since velociraptor physics r3 (D-D25, the same day),
  whose keyframe puts the toes at −0.5 mm of contact instead of −44.6 mm; the
  finding stands for the walkers that replace them.

<!-- The items below come from the 2026-09-28 gait audit: CPU replays of
     the certified nodes and zero-action rollouts of every stage gate, with
     contact read two ways, "touch" (the env's own foot sensors) and "floor
     contact" (the floor's normal force on a leg above 0.1 N on at least
     half of a control step's substeps). Full evidence in
     investigations/GAIT_AUDIT_2026_09.md; the plan that acts on them is
     GAIT_QUALITY_PLAN_2026_09.md, whose decisions (GQ-1..GQ-18) are all
     open and whose code PRs build on 0.3.9 (2026-10-06: GQ-6 (a) and
     GQ-7 (a) taken for stance only, as D-D23). The lunge-then-fall and
     falling-step hunt items were executed through the repository's own
     code before they were entered. -->

- **HIGH** — **no locomotion gate reads a foot contact, and three of the five
  certified walkers hop (replayed 2026-09-28).** All six locomotion stages
  judge `reward_and_length/v1` (`reporting/gates.py:1560-1625`) on mean reward,
  length and forward velocity, with reward floors 3–22× below the zero-action
  statue, so beyond mean length the only test is mean root speed. On floor
  contact, T-Rex seed 42 `20260914_123816` and seed 44 `20260925_033501` hop on
  both feet at 7.7 and 9.2 Hz, airborne 35% and 43% of steps (both retired
  by T-Rex physics r8, D-D24, which refuses every r7 T-Rex checkpoint), and
  compsognathus_robot `20260924_031815` micro-hops at 9.2 Hz, airborne 48%;
  only compsognathus `20260921_203149` walks (retired by anatomical
  compsognathus physics r2, D-D26, which refuses every r1 compsognathus
  checkpoint) and velociraptor `20260922_125248` runs (retired by
  velociraptor physics r3, D-D25, which refuses every r2 velociraptor
  checkpoint). The in-training dibothrosuchus re-run
  `20260928_012318` skids on three legs at 4.3M and clears its gate as written
  ([gait audit](investigations/GAIT_AUDIT_2026_09.md) §3–§4). Plan:
  `locomotion_gait/v1` in the six locomotion TOMLs, then retrains
  ([gait plan](GAIT_QUALITY_PLAN_2026_09.md) PR-G5 and PR-G7; GQ-8, open).
  *Update (2026-11-24, decisions D-D61 and D-D62):* this describes the walk
  gate before the command-driven walk. `reward_and_length/v1` now judges no
  locomotion stage: all seven `locomotion.toml` declare `command_tracking/v1`,
  whose walk profile judges every panel episode and whose gait clause reads
  each cruise episode's floor-truth footfalls (GAIT_QUALITY_PLAN GQ-8 (a),
  GQ-9 (a), GQ-18; D-D61 (7)). `locomotion_gait/v1` was not built; the walk
  gate's gait clause takes its place. Its bars are provisional, and no
  controller has passed a committed walk panel (D-D62 (16) and D-D64, the
  first item of this section). The item stays open until a walk certifies under that gate;
  the hopping certificates named above are stranded in any case by plant note
  24, which refuses every earlier checkpoint.

- **HIGH (hardware target)** — **the Compsognathus robot stands and walks on
  stacked feet, and its touch sensors count sole-on-sole force as support
  (replayed 2026-09-28).** Its 10 × 9 cm soles collide with each other
  (`compsognathus_longipes_robot.xml:181,254`), and both certified nodes of run
  `20260924_031815` rest the right sole on the left foot: the stance carries
  81% of its floor load on the left foot, and its reported two-foot support of
  0.9988 is 0.62 on floor contact. The walker's feet touch on 96% of substeps,
  pushing 1.12 body weights into each touch sensor, so the support flag that
  scales its alive, posture and height rewards (touch sum above 4% of body
  weight, `compsognathus_longipes/base_env.py:188,258`) is on 99.6% of steps by touch and
  34.3% by floor: about 838 of its about 2627 reward pays for support the
  floor never gives ([gait audit](investigations/GAIT_AUDIT_2026_09.md) §5).
  Plan: the robot's `gait-r1` revisions, with floor-contact support and a
  foot-collision penalty ([gait plan](GAIT_QUALITY_PLAN_2026_09.md) PR-G7;
  GQ-14, open).
  *Update (2026-11-23, decision D-D59, `configs/plant_versions.toml` note
  24):* the observation half is fixed. Since the robot's policy interface r4
  its two foot channels read each sole's normal force against the floor
  (`BaseDinoEnv._foot_floor_contact_forces` over the robot class's
  `FOOT_OBSERVATION_FLOOR_GEOMS`, `right_sole` and `left_sole`), so a sole
  resting on the other foot reads 0 N; the soles still collide with each
  other, as GQ-14 (a) keeps them. On the zero-action statue (seeds
  3042–3061, 1000 steps) the floor and touch readings agree to 0.0 N on
  every step; posed sole on sole, the upper foot's channel reads 0 N where
  its touch sensor reads 7.8 N. The reward half stays open: the support flag
  above still reads the touch sensors on both Compsognathus models until
  PR-G7's floor support.

- **HIGH** — **the stance gates admit stances that chatter, march, stand on
  one foot, stack their feet, prop a leg on a rolled pad or hop (replayed
  2026-09-28; re-measured 2026-10-06).** `stance_quality/v1` counts only steps
  with neither foot's touch sensors above 0.1 N, averaged over the panel after
  an unscored 200-step settle window; on 2026-09-28 the other three stances
  gated on a reward rail at 0.60× the statue, and the zero-action statue
  passed all six by design (`stance_gate.py:13-20`). The Allosaurus stance,
  the seventh, refuses its statue. No current stance node stands as its statue
  does ([stance-hack audit](investigations/STANCE_HACK_AUDIT_2026_10.md) §3):
  all three passing T-Rex stances (seed 42 `20260914_123816`, which the
  2026-09-28 audit called clean, and seed 44 `20260920_010912` and
  `20260930_024929`) hop at reset inside the settle window, replant their feet
  5–7 cm wider and stand with one leg splayed on a pad rolled 3–4.4° onto its
  front-inner corner, and `20260920_010912` also hops to rebalance in 6 of 40
  episodes under a panel mean of 0.0069; compsognathus `20260921_203149` and
  `20261001_225856` march in place on the tip of one toe at about 3 Hz;
  velociraptor `20260922_125248` holds a saturated crouch, chatters at about
  20 Hz and slides 0.6 m; the robot `20260924_031815` stands on one foot with
  the other stacked on it; dibothrosuchus `20260923_020654` and
  `20261004_025237` certified the untrained statue, three-legged in some
  episodes ([gait audit](investigations/GAIT_AUDIT_2026_09.md) §3–§4). The
  floor-truth gate kind `stance_quality/v2` exists (D-D23, 2026-10-06): on
  40-episode panels it fails every one of these checkpoints while every statue
  is clean on 40/40 (the stance-hack audit's §5). The T-Rex stance adopted it
  with T-Rex physics r8 (D-D24, 2026-10-06), on bars set on the r8 statue
  (clean on 40/40; the three T-Rex stances above and the failed seed-45 node on
  0/40 each, the audit's §7), and the velociraptor stance with velociraptor
  physics r3 (D-D25, the same day), on bars set on the r3 statue (clean on
  40/40; both audited checkpoints of `20260922_125248` on 0/40, the audit's
  §8), and the compsognathus stance with compsognathus physics r2 and its
  soft-cubic leg interface (D-D26, 2026-10-07), on bars set on the r2 statue
  (clean on 40/40; both certified r1 marches, `20260921_203149` and
  `20261001_225856`, rolled on the r1 plant, on 0/20 each, the audit's §11).
  On that light plant a two-foot hop started after the settle flies 1–3 of a
  step's ten substeps, so it reads both legs down on every step and a statue
  on every step-level bar; the kind's two optional window hop criteria
  (`max_window_airborne_substeps`, `max_window_peak_floor_force_bw`, D-D26),
  declared on the compsognathus stance, refuse all 410 such square-wave hops
  measured, so that gap is narrowed there, not closed: a softer two-foot
  bounce, a one-foot flutter and a fast weight shuttle still certify (the next
  entry; the T-Rex and velociraptor blocks do not declare the pair). The other
  four stance stages still certify under v1 (the Compsognathus robot, and the
  Allosaurus, added 2026-11-21, whose statue falls in every episode) or their
  reward rail (the dibothrosuchus and the brachiosaurus,
  `reward_and_length/v1`). Plan: each remaining stance stage adopts v2, a gate
  revision with bars re-measured on that species' own statue panel
  ([gait plan](GAIT_QUALITY_PLAN_2026_09.md) §11–§16 and PR-G8; GQ-10, open);
  each species stays open until its stance certifies under v2, the T-Rex,
  velociraptor and compsognathus included: no r2 compsognathus stance has
  trained yet; every r3 velociraptor stance trained so far is clean on 0/40
  (two 6M runs and nine 1M-step reward pilots,
  [STANCE_REWARD_PILOT_FINDINGS_2026_10.md](investigations/STANCE_REWARD_PILOT_FINDINGS_2026_10.md)
  §1.3–§1.4; every one is stranded by velociraptor physics r4, D-D30,
  whose statue fails the 0.35 windowed load-share bar on 1 of 240
  out-of-sample episodes, every bar kept); and the first two r8 T-Rex
  stances fail v2 (a two-foot micro-hop, and a settle stomp with a
  few-percent whole-episode hop mode;
  the audit's §9), after which D-D27 revised the T-Rex stance task and added
  settle-touchdown, yaw and hop-or-fall keys to its block. Of the D-D27
  runs, seeds 42 and 48 fail (0/40 and 36/40), and seed 52
  (`20261009_155723`) passed v2 on its own 3042–3081 panel (39/40) but is
  clean on 32/40 of the re-judge seeds 7042–7081 (re-rolled from its
  checkpoint in the 2026-10-10 review; the findings doc's §1.2).
  *Update (2026-10-25): no stance gate reads head posture.* Rolled zero-shot
  on T-Rex physics r10 (D-D44), the physics-r9 stance that is walking pilot
  2's parent (`stance3_armLT_s60`'s final) drops its head tip from 1.24 to
  0.61 m (neck pitch about +33°) and still passes `stance_quality/v2` on
  37/40, and the band's seed-52 final drops it from 1.08 to 0.67 m and passes
  39/40 (measured, deterministic, seeds 3042–3081; the stance record's §25.3,
  evidence in `stance5/tail_sr3/cpucheck/`). The reward's `head_clearance`
  term prices it (about 330 per episode on both), the gate does not. A
  head-height criterion is a gate change and the maintainer's to decide, for
  the first r10 round's criteria at the latest; until then, check a stance's
  head tip by hand before using it. *Update (2026-10-27):* the maintainer kept
  head posture out of the gate (2026-10-25, D-D46: report-only telemetry, no
  gate change). `eval_episodes.csv` now records each evaluation episode's
  head-tip height and neck and head pitch, the training-history report draws
  them, and the first r10 round's criteria read them as a report-only
  diagnostic readout, never in a class or its reading; no gate reads them, so
  check a stance's head before using it still. *Update (2026-10-28):* in the
  first r10 round no stance dropped its head (no window-mean head tip below
  0.75 m at any judged checkpoint or in any evaluation; one evaluation episode
  of seed 60 that fell, at 11,534,336 steps, read 0.749 m), but seed 80 raised
  it during its neck saturation habit (its head tip averaging 1.31 m from 4.2M
  to 8.4M steps against the statue's 0.96 m) and later held its head-pitch
  actuator parked negative, and seed 60's final stands with its head tip at
  1.12 m; no gate reads head posture, so check a stance's head by hand before
  using it still (§27 of the stance record).
  *Update (2026-11-26):* the two reward-rail stances roll no stance report, so
  no report screens their per-foot load: the brachiosaurus' evaluations run no
  recorder, and the per-foot columns of the dibothrosuchus' evaluation ledger
  and replay `_stance.csv` reach no summary or verdict. The dibothrosuchus r3
  quick-test handoff stood on three legs in its replay (2026-11-22), and no
  summary said so (its checkpoint is stranded by plant note 24). Until they
  adopt v2 (GQ-10; the six-heading certificate since D-D74, the next
  update), give the reward-rail stances a report-only per-foot
  floor-load screen (each foot's share of the window's floor load, and the
  share of steps with a foot unloaded), which needs the four-foot
  `derive_stance_info` of the forefeet entry below (PR-G2). (2026-11 notebook
  review: dibothrosuchus-three-legged-stance-certified-unflagged-r3)
  *Update (2026-11-27):* D-D74 (the maintainer's decision of 2026-11-26)
  changes the plan above in its target and sets its timing: each model still
  on v1 or a reward rail adopts `stance_quality_heading/v1`, the six-heading
  certificate over v2's episode classification, not v2 at one heading,
  before its next stance run, with bars measured on its own statue panel.
  That is the allosaurus (added 2026-11-21) and the Compsognathus robot from
  `stance_quality/v1`, and the brachiosaurus and the dibothrosuchus from
  `reward_and_length/v1`. GQ-10 and PR-G8 name v2, which predates the
  certificate; D-D74 amends both for these four
  ([gait plan](GAIT_QUALITY_PLAN_2026_09.md) §18). No config has changed
  for D-D74 yet: #703 kept each of the four on its gate kind, though it
  re-measured the dibothrosuchus' rail (1560 → 1500). Since its revision
  of all seven plants (`configs/plant_versions.toml` note 24) every model's
  observation is heading-free, the robot's and the two quadrupeds'
  included, so each upgrade certifies a heading-free observation at six
  headings, and the choice of v2 at one heading, open when D-D74 was
  taken, is moot.

- **MEDIUM** — **`stance_quality/v2` on the anatomical compsognathus certifies
  a soft two-foot bounce, a one-foot flutter and a fast weight shuttle
  (measured 2026-10-08).** A leg is down on a step when it is loaded on at
  least half of the step's ten substeps, and the window hop pair counts only
  substeps with both feet off the floor and the window's peak floor force, so
  on this plant a foot that lifts for less than half a control step, and a
  two-foot hop that lands under 2.0 body weights and flies no longer than the
  admitted noise, read a statue on every bar. Scripted from step 200 on the
  committed stance and rolled through the report path (seeds 3042–3081, the
  report's own statue panel), each certifies: the post-settle hop's pattern
  (knees +a, ankles and hip pitch −a) driven by a sine at 5 Hz (a = 0.09;
  40/40, 0.989 of the statue's reward; both feet off the floor on 39 of 40
  episodes for 0–32 window substeps, each foot lifting off 16–54 times an
  episode, landing at 1.70–1.80 BW) and at 2.5 Hz (a = 0.12; 40/40, 0.990;
  0–27 substeps, 1.33–1.54 BW); one 10 Hz square cycle of it every 0.4 s (a =
  0.04; 39/40, 0.974; 0–41 substeps, 1.65–1.66 BW); the right leg alone at
  6.25 Hz (a = 0.07; 40/40, 0.972), its foot lifting for one substep 130–190
  times an episode and never both feet; and the legs in antiphase at 6.25 Hz
  (a sine, a = 0.10; 40/40, 0.905), which unloads a foot at most twice an
  episode but puts the lighter one under 20% of the floor load on 44–48% of
  window steps (down to 14%), below what the foot-share bars see when they
  average over the window or over 1 s blocks. The right toe alone at ±0.25 and
  10 Hz (146–175 lift-offs) certified too (40/40, 0.984) until the stance
  declared D-D28's pad centre-of-pressure bar, which refuses it on 40/40: each
  tap rocks the pad from edge to edge (0.826–0.848 against the bar's 0.70; the
  other five read 0.14–0.48 and still certify). No count bar separates these
  from the noise the block admits by design: the statue under σ 0.03 command
  noise flies up to 23 two-foot window substeps, lifts each foot off 53–99
  times and moves its root through 0.6–0.9 mm (the 5 Hz bounce 1.0–1.1 mm),
  and under σ 0.05 lifts a foot off 109–156 times; the refused square-wave
  hops at a = 0.05–0.07 lift a foot off 133–224 times. The reward keeps
  0.905–0.990 of the statue on all of them, because the legacy substep-minimum
  support gate prices only a substep with both feet unloaded. Plan: until a
  gate revision lands, count each trained r2 stance's per-foot lift-offs and
  view its window root height for a steady rhythm before it is used ([recipe
  review](../environments/compsognathus_longipes/TRAINING_RECIPE_REVIEW.md) amendment);
  a follow-up revision of `stance_quality/v2` needs a periodicity criterion
  (for example a spectral peak of the window's floor force), measured on the
  statue, the jittered statue and these scripts ([stance-hack
  audit](investigations/STANCE_HACK_AUDIT_2026_10.md) §11, its hack table and
  Limits).
  *Update (2026-11-26):* the check is still by hand. Nothing computes a
  stance's per-foot lift-offs or its window root-height rhythm: the stance
  report's touchdown bar (`max_touchdown_rate`,
  `configs/compsognathus_longipes/anatomical/stance.toml:383`) counts
  debounced step-level touchdowns, which the sub-step lifts above never reach;
  the SB3 notebook does not prompt for the check; and a walk session
  (`BEHAVIOR = "walk"`; `"auto"` left with decision D-D79) warm-starts the
  walk from a certified stance without it. Fix: compute both
  as report-only columns of the stance certificate (lift-offs per foot per
  episode, and a spectral peak of the window root height) and print them at
  the handoff. (2026-11 notebook review:
  compsognathus-manual-liftoff-check-not-automated)

- **MEDIUM** — **σ 0.20 of exploration noise topples the anatomical
  compsognathus statue, and σ 0.135 bounces it (measured 2026-10-07).**
  Physics r2, its toe armature and the soft-cubic leg residual (D-D26) moved
  the noise cliff past the stance recipe's initial σ, but by less than 1.5×:
  zero-mean action noise on the stance task (seeds 3042 + i, paired streams)
  leaves the statue standing to the horizon on 40/40 episodes at σ 0.10, on
  151/160 at the recipe's σ 0.135 (`log_std_init` −2.0; four seed blocks of
  40, one with an independent noise stream, each 37–39 of 40; all nine falls
  end on the tail tip, at steps 174–938) and on 0/40 at σ 0.20 (median 130–180
  steps over two noise streams); r1 with the linear residual stood on 0/20 at
  σ 0.135, falling in a median of 12 steps. A larger `log_std_init`, or an
  entropy schedule that lets the learned σ grow, trains on a plant that falls.
  Below the cliff the noise still bounces this light plant: two 200k-step PPO
  runs on the stance task reach the horizon stochastically on 20/20 but are
  clean under `stance_quality/v2` on 0/20 (touchdowns 1.2–2.4 per foot per
  second, window peaks 3.0–4.9 body weights), and only their deterministic
  means are clean (40/40), which shows only that the mean does not drift to a
  march or a hop early.
  `environments/compsognathus_longipes/tests/test_anatomical_noise_tolerance.py` pins σ 0.10 on
  the first 5 episodes and σ 0.135 on at least 10 of the first 12 (the
  review's 0.8). Plan: keep `log_std_init` −2.0 and the decaying entropy
  schedule ([recipe
  review](../environments/compsognathus_longipes/TRAINING_RECIPE_REVIEW.md)), watch the
  learned standard deviation in the first r2 stance runs (at least two seeds),
  and re-measure the cliff with any plant or interface change.

- **MEDIUM** — **stance exploration noise is set against the statue's noise
  cliff on two plants only (read from the configs 2026-11-26).** The
  velociraptor (std 0.10, `log_std_init = -2.30`,
  `configs/velociraptor_mongoliensis/anatomical/stance.toml:240`) and the
  anatomical Compsognathus (0.135, `-2.0`,
  `configs/compsognathus_longipes/anatomical/stance.toml:207`) start at a σ
  their statue was measured to survive (the entry "σ 0.20 of exploration noise
  topples the anatomical compsognathus statue" has its numbers), and only they
  have a noise-tolerance test (`test_anatomical_noise_tolerance.py` in
  `environments/velociraptor_mongoliensis/tests/` and
  `environments/compsognathus_longipes/tests/`). The T. rex's 0.22 (`-1.5`,
  `configs/tyrannosaurus_rex/anatomical/stance.toml:518`) was set from reward
  cost and short PPO runs and sits above its statue's cliff by decision
  (D-D27). The rest have no measurement:
  - The dibothrosuchus, brachiosaurus and Allosaurus `stance.toml`s set no
    `log_std_init`, so they train at SB3's std 1.0, with no recorded reason.
    In the 2026-11 review's quick tests, 0 of 873 dibothrosuchus training
    rollouts reached the horizon (excessive tilt 501, tail contact 295) and 0
    of 821 Allosaurus; the brachiosaurus kept 49 of 54.
  - The Compsognathus robot copies its sibling's `-2.0`
    (`configs/compsognathus_longipes/robot/stance.toml:56`) with no
    measurement of its own, although in that review its statue fell on every
    episode at σ 0.02 (and stood at 0.01) and all 1,109 of its quick-test
    training rollouts fell. The [stance-hack
    audit](investigations/STANCE_HACK_AUDIT_2026_10.md) §3.4 notes that it
    falls under any PPO-scale noise; no test or plan item followed.
  - Those readings predate note 24 of `configs/plant_versions.toml`, which
    moved all seven plants. The SB3 notebook prints no training fall fraction
    or policy std, though `progress.json` holds the one
    (`training_rollouts.fall_fraction`) and the training history draws the
    other.

  Fix: measure each plant's cliff with the existing noise test, extended to
  the robot, the two quadrupeds and the Allosaurus, then set `log_std_init` or
  record in the stage TOML why std 1.0 is intended; print the fall fraction
  and policy std in the notebook's per-evaluation readout (the progress entry
  under Notebooks). (2026-11 notebook review:
  cmp-exploration-start-uncalibrated-quadrupeds-allosaurus,
  compsognathus-robot-falls-under-any-exploration-noise)

- **MEDIUM** — **locomotion gates average per-episode means, so a policy that
  lunges and falls passes (executed 2026-09-28).** An episode's speed is the
  mean of its per-step `info["forward_vel"]` (`evaluation.py:105-123`), the
  panel's the unweighted mean of those (`reporting/stage_artifacts.py:1714`,
  `curriculum/manager.py:259`), and length a panel mean, so a 250-step episode
  weighs as much as a 1,000-step one. Through the repository's own evaluation
  and gate code on T-Rex locomotion (length 750, speed 1.0 m/s), three scripted
  episodes at 0.7 m/s for 1,000 steps and one at 2.0 m/s that falls at step
  250 give mean length 812.5 and speed 1.025 m/s (recorded 1.02) and pass,
  though distance over time is 0.80 m/s
  ([gait audit](investigations/GAIT_AUDIT_2026_09.md) §3); on a 30-episode
  panel, 23 walks and 7 lunges pass both the post-training judge and the
  in-training `CurriculumManager`, while 24 and 6 fail. The post-training
  judge also compares the speed rounded to two decimals (`reporting/stage_artifacts.py:1714`): a 0.996 m/s
  panel passed the 1.0 bar. Plan: per-episode qualification under
  `locomotion_gait/v1` ([gait plan](GAIT_QUALITY_PLAN_2026_09.md) §4.2, PR-G5;
  GQ-7, open for locomotion; taken for stance only, as D-D23, on 2026-10-06).
  *Update (2026-11-24, decisions D-D61 and D-D62):* this no longer applies to
  any committed stage. Every walk is judged per episode by
  `command_tracking/v1`'s walk profile: at 20 cruise episodes
  `min_cruise_survival_lcb` 0.80 is cleared only by 20 of 20 reaching the
  horizon, so a lunge that falls fails, and the gait clause judges each
  cruise episode's footfalls. The panel-mean reading above is `reward_and_length/v1`'s,
  which judges no locomotion stage now. The walk profile's bars are
  provisional (D-D62 (16)).

- **MEDIUM** — **the recovery safe set has no support clause, and the
  certified T-Rex recovery answers forward pushes with two-footed hops
  (replayed 2026-09-28).** The calibrated judge is posture-only on purpose,
  `min_foot_force_n = 0.0` (`recovery_evaluation.py:74-79`; the compsognathus
  pair's calibrations the same), because quiet certified stance reads 0.0 N on
  a foot during weight shifts. On a CPU replay of the seed-44 recovery in
  `20260920_010912` (certified 28/40), none of 41 forward pushes is absorbed
  in place: 27 mix two-footed hops with single-foot touches, 13 are pure
  two-footed hops and 1 falls; 37% of its touchdowns fall outside push
  windows, and it drifts 0.67 m per episode
  ([gait audit](investigations/GAIT_AUDIT_2026_09.md) §4). Plan:
  `recovery_quality/v2`, with windowed support clauses, registered and then
  adopted per species ([gait plan](GAIT_QUALITY_PLAN_2026_09.md) PR-G9; GQ-13,
  open); allowing no two-footed touchdown in a push window, seed 44 scores
  5/40.

- **MEDIUM** — **the T. rex recovery is frozen and judged on its physics-r7
  judge, with no warning and no mark in the verdict (read from the code
  2026-11-26).** `_model_calibration` returns no calibration for
  `LEGACY_RECOVERY_JUDGE_MODEL_ID`, the anatomical T. rex
  (`harnesses/freeze_recovery_gate.py:397-399`;
  `curriculum/gate_resolver.py:54`), so its null panels and its policy panel
  use `recovery_evaluation.py`'s `CALIBRATED_POSTURE_ONLY` safe set and
  `CALIBRATED_HEIGHT_REFERENCE_M` 0.9267 m (:79-100), measured on the r7
  certified stance. The comment above them says they are stale since physics
  r8 and are to be re-measured on the first certified r10 stance before the
  gate is re-frozen (D-D24); the plant is r12 now
  (`configs/plant_versions.toml` note 24), and the comment still names r10.
  The SB3 notebook freezes the T. rex recovery on it whenever the node trains
  (cell 20, `sb3_training.ipynb:1619`; the manual cell, :2034); the freeze
  summary's warnings (`freeze_recovery_gate.py:839-862`) say nothing of the
  judge, the quiet safe-set check that qualifies the parent on the other two
  models is skipped for it (:742-743), and neither `gate_resolution.json` nor
  `gate_verdict.json` marks the verdict as judged by an unmeasured judge. Fix:
  refuse to freeze the T. rex recovery until the judge is re-measured on a
  certified stance of the current plant (or warn at freeze and record the
  legacy judge in the resolution and the verdict), and update the comment.
  (2026-11 notebook review:
  tyrannosaurus-trex-recovery-legacy-judge-auto-frozen-stale)

- **LOW** — **a Compsognathus stance is checked against its recovery judge
  only when the recovery node starts (read from the code 2026-11-26).** The
  95% quiet safe-set occupancy a stance checkpoint must reach before its
  recovery can freeze is measured by `_quiet_brace_controller`
  (`harnesses/freeze_recovery_gate.py:411-470`, the bar at :425), which runs
  only inside `freeze_recovery_gate` when the chain loop starts the recovery
  node (cell 20, `sb3_training.ipynb:1619`). Nothing measures it when the
  stance is judged or handed off, so a stance can certify, and feed the walk,
  that the recovery freeze then refuses ("recalibrate the judge from qualified
  stance evidence before recovery training"), as the review's manual run of
  the robot's recovery was refused at 68% (2026-11-22). Recalibration is
  `scripts/calibrate_recovery.py`, from the command line only. T. rex's legacy
  judge skips the check (the T. rex recovery entry in this section). Fix: at
  the end of a stance whose manifest has a recovery child, measure the quiet
  occupancy under that child's judge, record it in the stance's report, and
  warn below 95%. (2026-11 notebook review:
  cmp-recovery-parent-qualification-asymmetric)

- **MEDIUM** — **knee, shin and proximal-tail floor contact never ends an
  episode on T-Rex, velociraptor, brachiosaurus or dibothrosuchus (read from
  the code, clearances measured 2026-09-28).** Their `_body_ground_geoms` hold
  only the torso (and belly), head and distal tail (`tyrannosaurus_rex/base_env.py:471-478`,
  `velociraptor_mongoliensis/base_env.py:314-321`, `brachiosaurus_altithorax/base_env.py:238-244`,
  `dibothrosuchus_elaphros/base_env.py:254-261`; velociraptor adds its neck), so a policy
  may kneel or crawl while the root stays above the height floor. Settled
  clearance of the lowest such geom, then the root drop allowed: T-Rex tibia
  0.157 m, 0.226 m; velociraptor metatarsus already touching, 0.194 m;
  brachiosaurus shins 0.067 m, 0.186 m (0.336 m on locomotion); dibothrosuchus
  shins 0.047 m, 0.133–0.153 m. The compsognathus pair ends on any non-foot
  geom ([gait audit](investigations/GAIT_AUDIT_2026_09.md) §3). Plan: the
  reward kit's `terminate_on_leg_contact`, set in the dibothrosuchus and
  brachiosaurus `gait-r1` revisions ([gait plan](GAIT_QUALITY_PLAN_2026_09.md)
  PR-G6 and PR-G7); none is planned for T-Rex or velociraptor.
  *Update (2026-11-09; the maintainer's decision of 2026-11-08):* on
  velociraptor physics r5 `_body_ground_geoms` holds the torso, belly,
  ribcage, `skull_upper`, `snout` and `tail_3` to `tail_5` (the neck no
  longer collides). The settled noise-free stance statue's lowest
  non-terminating geom is the metatarsus shaft, 14.4 mm clear (then the claw
  19.5 mm and the shank 109 mm), and the height floor of 0.33 m allows a
  root drop of 0.219 m.

- **LOW** — **the shared stance diagnostic reads only the forefeet on
  quadrupeds (read from the code 2026-09-28).** `derive_stance_info`
  (`stance_diagnostics.py:77`) takes `r_foot_contact` and `l_foot_contact`,
  which the brachiosaurus and dibothrosuchus envs set to their front feet
  (`brachiosaurus_altithorax/base_env.py:417-418`, `dibothrosuchus_elaphros/base_env.py:456-457`), so its
  unsupported duty and balance ignore the hind feet, which carry 79% of the
  certified dibothrosuchus stance's floor load
  ([gait audit](investigations/GAIT_AUDIT_2026_09.md) §3). No quadruped gate
  reads it today. Plan: a four-foot `derive_stance_info`
  ([gait plan](GAIT_QUALITY_PLAN_2026_09.md) PR-G2).

<!-- The items below come from the 2026-08-28 RL pipeline gap review
     (reviews/RL_PIPELINE_GAP_REVIEW_2026_08.md, whose appendix B records
     what became of every finding). Each was re-checked against the code on
     2026-09-30; EP1 and EP4 were executed. -->

- **MEDIUM** — **a same-stage resume inside a stage's entry window drops the
  rest of its reward ramp and warm-up (read from the code 2026-09-30).** A
  node entered from its parent (`initialize_next_stage`) trains under
  `StageWarmupCallback` (default 100k steps) and, when its
  `forward_vel_weight` is positive, `RewardRampCallback` (from 0.1 to the
  stage's weight over 500k steps by default). A `resume_same_stage` load
  attaches neither: `_stage_entry_shaping_callbacks` returns nothing for that
  mode (`train_base.py:916-917`, called at :1321-1328), so the env trains at
  the stage's full `forward_vel_weight` from the first resumed step. Since the
  gap review's TC1 fix a continuation keeps the checkpoint's step counter
  (`train_base.py:1218-1232`, `reset_num_timesteps=not resuming` at :1349),
  and the ramp reads that counter (`curriculum/advancement.py:640-659`), so
  its position is recoverable; nothing re-applies it. The warm-up marker is
  cleared with a warning that the rest of the warm-up is not re-applied
  (`train_base.py:611-626`); the ramp's remainder goes without a log line. It bites when a
  session dies early in a locomotion or behavior stage and the RESUME cell
  continues it: 200k steps into a 0.1 → 1.0 ramp, the policy meets the other
  0.54 of the weight in one step. Fix: re-attach the shaping on a
  continuation, offset by the counter (not on a curated `--load`, which
  `_is_resume_continuation` tells apart), or at least warn. No cleanup PR
  owns it. (2026-08 gap review TC8) Since decision D-D62 no committed stage
  builds `RewardRampCallback`: every walk follows a live command and ramps
  its commanded speeds instead, and that cruise ramp does resume, from the
  position every checkpoint records (`curriculum/cruise_ramp.py`). The
  forward-velocity ramp and the warm-up remainder still behave as above for
  a command-free stage, and the warm-up remainder for the walk too.

- **MEDIUM** — **certification is judged on one fixed 40-seed block, and the
  held-out confirmation panel the stance design calls for does not exist
  (read from the code 2026-09-30).** `stance_quality/v1` implements items 1
  and 2 of its adopted rule; item 3, one predeclared held-out panel at
  n ≈ 100–180 for the full-horizon event, "is offline and deliberately lives
  outside this module" (`curriculum/stance_gate.py:49-51`), and no code runs
  it ([STAGE1_SPLIT_PLAN.md](STAGE1_SPLIT_PLAN.md) §2.3: "neither is item 3's
  held-out panel"). Every certifying panel runs on the registered block
  3042–3081 (`constants.py:34`, `PUBLICATION_SEED_START`): publication
  refuses a `certification_panel` role that starts anywhere else
  (`result_bundle/evidence.py:1095-1099`) and binds each stance panel row to it
  (:456-463). The panel is deterministic given the policy, so a marginal
  policy that happens to fit the block passes every re-check; by the review's
  arithmetic, one with a true full-horizon rate near 0.93 fits about 46% of
  blocks and would fail a fresh one about 54% of the time. The blocks do
  differ: on physics r7 the statue scored 119/120 over three of them
  (`configs/tyrannosaurus_rex/anatomical/stance.toml:667`), and on r8 it is clean on 40/40, 80/80
  and 39/40 of three disjoint blocks under `stance_quality/v2` (seed 3174
  nosedives on reset noise alone). Seed replication (`certification_seeds`,
  the review's SS1 fix) counts training seeds, not fresh evaluation seeds.
  Fix: run the confirmation panel on a disjoint seed block before a stance is
  certified and record both blocks (`stance_gate_report.py --episodes`
  already sizes such a panel). No cleanup PR owns it. (2026-08 gap review SS3
  and SS4)
  *Update (2026-10-07):* the r8 T-Rex seed-44 stance `20261006_185704` is the
  measured case. On the certification block 3042–3081 it fails 1 of 40
  episodes in the window (seed 3080, after a floating-point divergence), on
  the fresh block 7042–7081 4 of 40 (three whole-episode hops and a fall);
  over about 207 resets it hops for the whole episode on 4.3% (95% CI
  2.0–8.1%) and rolls over on one more, a hop-or-fall rate of 4.8%. A bound
  of 0.80 at n = 40 admits up to 3 unclean episodes, so a panel whose only
  unclean episodes are such a mode's passes with probability about 0.87. The
  T-Rex stance now declares the `max_hop_or_fall_episodes = 1` rail (D-D27),
  which counts the falls too and cuts that to about 0.42, and refuses a
  statue-quality policy on about 1 panel in 23; the fresh-block confirmation
  stays manual ([NEXT_STEPS.md](NEXT_STEPS.md) asks for a re-judgement on
  7042–7081 before a T-Rex stance is handed on), and of the other stances only
  the velociraptor's declares the rail (2026-11-06,
  `configs/velociraptor_mongoliensis/anatomical/stance.toml:520`; the
  anatomical Compsognathus is the next entry).

- **MEDIUM** — **the anatomical Compsognathus stance certifies on the
  six-heading certificate without the guards the T. rex and velociraptor
  stances carry, and records no reason (read from the code 2026-11-26).** The
  three stances on `stance_quality_heading/v1` share the certificate, and the
  Compsognathus anneals its entropy bonus to 0 at 7M as the other two do
  (`configs/compsognathus_longipes/anatomical/stance.toml:199-201`). Only the
  T. rex and the velociraptor declare `certification_seeds = 2`
  (`configs/tyrannosaurus_rex/anatomical/stance.toml:1030`,
  `configs/velociraptor_mongoliensis/anatomical/stance.toml:633`), the
  500,000-step checkpoint archive (:1049 and :625), the hop watch that stops a
  seed still hopping when the entropy bonus runs out (:1152-1167 and :644-657)
  and the `max_hop_or_fall_episodes = 1` rail (:887 and :520). The
  Compsognathus stance declares none of them and gives a reason for the rail
  alone ("adding one is a separate threshold decision", :354-355). So its
  default of one seed (`gate_schema.py:594-601`) publishes a single certified
  seed as certified, a seed that locks into a hop at 7M trains on to the 11M
  budget, and no archived checkpoint is left to re-judge. No stance has
  trained on the current Compsognathus plant (physics r4), so whether the
  recipe is seed-sensitive is unmeasured. Fix: give the Compsognathus stance
  `certification_seeds = 2`, the archive and a hop watch with its bar measured
  on its statue, or record in the TOML why each is left out. (2026-11 notebook
  review: cmp-certificate-stances-uneven-guards)

- **MEDIUM** — **only three stances certify heading (the T. rex's and the
  anatomical compsognathus's since D-D32, heading-free since policy
  interface r14 and r4; the velociraptor's since 2026-11-06, heading-free
  since r13), and both r8 T-Rex stances stand only near the heading they
  trained at (measured 2026-10-07).** The stance stage spawns every episode facing +x (yaw 0, the
  prey within about ±11°), and up to policy interface r13 the T-Rex
  observation carried heading only in world-frame quantities (the pelvis
  quaternion, the pelvis linear velocity, the prey vector), so a policy could
  learn a stance that reads its absolute heading, and no v2 bar sees it.
  Spawned turned about vertical, the
  zero-action statue stands 8/8 at ±45° and ±90°, so the plant is
  heading-neutral (exactly so since physics r9, D-D29: on r8 the pyramidal
  friction cone made the box pads' friction depend on world heading, and
  the statue's ±45° rewards on seeds 3042–3049 differed from 0° by up to
  0.03; under the elliptic cone every heading agrees to 3 × 10⁻¹¹); the
  seed-44 stance `20261006_185704` stands 0/8 at ±90° (the animal turned
  alone) and 0/8 at ±45° with the prey turned too (only
  the observation differs), and the seed-42 stance `20261006_185343` falls
  4/4 at ±90° and, when it stands, steers to an absolute heading of about
  −45° ([stance-hack audit](investigations/STANCE_HACK_AUDIT_2026_10.md) §9).
  D-D27 added a report-only probe (`stance_probe_spawn_yaw_deg`, written to
  `stance_heading_probe.txt`; `stance_gate_report.py --spawn-yaw-offsets`)
  and a 25° bar on episode yaw change, which refuses a stance that turns but
  not one that only falls when turned. Fix, decided by the maintainer on
  2026-10-07 (its observation half carried out by D-D31, the update below):
  heading-free observations for the T. rex and the
  compsognathus (body orientation without world yaw, linear velocity in the
  yaw frame, the prey direction relative to the body, and every other
  world-heading channel), in one cross-species migration after the
  compsognathus revision (D-D26) lands and before either species trains,
  each species' policy-interface revision moving so their checkpoints stop
  loading; and a gate that sees heading. The gate is carried out
  (2026-10-11, D-D32): both stances declare the six-heading certificate
  `stance_quality_heading/v1`, which rolls the 40 certification seeds with
  the whole scene turned −90, −45, 0, 45, 90 and 180 degrees and passes only
  when every heading passes the v2 decision and the bound on seeds clean at
  every heading clears 0.80, so a stance like seed 52's, 40/40 at 0° and
  0/8 at −90° and −45° in the probe, fails, and handoff, reuse and
  publication refuse it. It certifies a policy, not the observation: any
  stance that stands at every heading passes it, whatever its observation
  reads. The
  velociraptor follows only after the maintainer's two current velociraptor
  runs are reviewed (physics r4, D-D30, strands them; its plant, like every
  species', is on the elliptic cone, so its headings are physically
  identical too). Up to policy interface r3 the anatomical compsognathus
  observation carried heading the same way (the pelvis quaternion, linear
  velocity and target direction in the world frame; every spawn faces +x):
  its r2 statue is clean spawned
  yawed ±45° and ±90°, and two 200k-step PPO means are clean with seed 11 at
  +90° and seed 12 at −90° (20/20 each; the stance-hack audit's §11), but no
  trained r2 stance has been rolled turned. No other species' trained stance
  has been probed. On the stances without the certificate (the velociraptor
  and the compsognathus robot) the probe stays report only, and its text
  (`stance_heading_probe.txt`, from `reporting/stance_report.py`) now names
  the certificate as the gate where it is declared. By the maintainer's decision of
  2026-10-11 the handoff tooling warns, without blocking, when a whole-scene
  row's full-horizon fraction is below 0.95, the v2 per-panel rail: a
  handoff-warning block in that text, which the
  `stance_gate_report.py --spawn-yaw-offsets` output prints too, and a
  logged warning where the probe is written, where `train` or the
  command-line curriculum starts a node on the stance's handoff, and where
  `record_ancestor` reuses it. The maintainer's approval of the same day
  turns the probe on for the velociraptor, compsognathus and compsognathus
  robot stances too, as the T. rex stance had it (the two quadrupeds'
  reward-and-length stances roll no stance report), and each one's statue
  stands 8/8 on every row of it; on the T. rex and the anatomical
  compsognathus the certificate replaced the probe the same day (D-D32:
  config load refuses the key beside the certificate, and the handoff
  warns instead when the certificate is missing or does not re-derive to a
  pass, which a manual `train` load, reading no verdict, would otherwise
  hand on silently). A stage that declares the probe and
  has none warns that its heading robustness is unmeasured; the
  command-line curriculum rolls no probe, so every stance it trains of the
  velociraptor and the compsognathus robot warns so at its handoff. No
  verdict changes there.
  *Update (2026-10-11, D-D31):* the observation half of the fix is carried
  out too. The T. rex (policy interface r14) and the anatomical compsognathus
  (r4) observe projected gravity, the root velocity in the gravity-aligned
  heading frame and the prey direction in pelvis coordinates
  (`bipedal-target-heading-free/v1`, `configs/plant_versions.toml` note 18),
  so a whole-scene turn leaves their observation unchanged and one seed's
  six headings repeat the same episode to solver tolerance; every T. rex r13
  and compsognathus r3 checkpoint is refused, the probed stances above
  included. The certificate above judges each heading's outcome rather
  than expecting identical traces: a heading-free policy can still behave
  differently turned than at the training heading, though only through
  floating-point rounding that an unstable closed loop amplifies (an
  untrained policy behind near-degenerate normalizer statistics diverged
  by 0.03 in action over 1000 steps; see
  `environments/shared/tests/test_heading_free_rollouts.py`). The entry
  stays open for the velociraptor and the compsognathus robot, which
  still read world-frame channels and certify no heading. The probe's
  note and the `SpawnYaw` docstring in `reporting/stance_report.py` say
  so: a whole-scene turn changes nothing physical on any plant (every
  plant declares the elliptic cone) and nothing in the T. rex r14 or
  anatomical compsognathus r4 observation, and it changes only the
  world-frame channels of the velociraptor's and the compsognathus
  robot's.
  *Update (2026-11-06, the maintainer's decision that day):* the velociraptor is heading-free too (policy
  interface r13, `configs/plant_versions.toml` note 21; one seed's six
  headings repeat its episode, joint state to 9.1e-14), and its stance
  declares the six-heading certificate in place of the report-only probe
  (its statue clean on 40/40 seeds at every heading on 3042–3081 and on
  7042–7081). The entry stays open for the compsognathus robot alone.
  *Update (2026-11-23, decision D-D59, `configs/plant_versions.toml` note
  24):* every model's observation is heading-free now, the robot's
  included: the prey, food and course target left every scene, and the
  robot (policy interface r4), the brachiosaurus (r12) and the
  dibothrosuchus (r11) moved from the world frame to the same layout
  (`bipedal-heading-free/v1`, and `quadrupedal-heading-free/v1` on the
  quadrupeds' torso), so a whole-scene turn changes nothing any policy
  observes, and the episode's forward reference is the spawn's own heading.
  What stays open is the gate: the robot stance still declares only the
  report-only probe, not the six-heading certificate, and the two
  quadrupeds' stances roll no stance report, so no heading of those three
  is certified.
  *Update (2026-11-26):* the Allosaurus stance (added 2026-11-21) declares the
  report-only probe too, `stance_probe_spawn_yaw_deg = [-90.0, -45.0, 45.0,
  90.0]` beside `stance_quality/v1`
  (`configs/allosaurus_fragilis/anatomical/stance.toml:22,33`), and records no
  reason for either. Its observation is heading-free
  (`bipedal-heading-free/v1`, `configs/plant_versions.toml:246-248`), so every
  whole-scene row repeats the unturned episode, and its zero-action statue
  falls in every episode (88.5 of 1,000 steps on seeds 3042-3043,
  `scripts/zero_action_baseline.py --stage 1`, measured 2026-11-26), so every
  row carries the warning that "the control did not stand, so those rows do
  not isolate heading" (`reporting/stance_report.py:3491-3495`): on this plant
  the probe measures nothing. Fix: choose and record the Allosaurus stance's
  gate kind; the six-heading certificate, as on the other heading-free bipeds,
  needs a statue that stands, so until one does, drop the probe and say why.
  (2026-11 notebook review:
  cmp-allosaurus-heading-probe-on-heading-free-plant)

- **LOW** — **the pad centre-of-pressure bar leaves partial-edge stances
  open, and only the T. rex and the compsognathus declare it (measured
  2026-10-07 and 2026-10-08).** D-D28's `max_sole_cop_fore_aft`, declared by
  the T. rex stance at 0.80, refuses the r8 seed-44 left foot on its pad's
  front edge; the fix and its calibration are the
  [stance-hack audit](investigations/STANCE_HACK_AUDIT_2026_10.md)'s §10.
  What it leaves open: (1) it is a window mean of the per-step reading, so
  a pad on its front edge for part of the window beside a centred s passes
  up to (0.80 − s)/(1 − s) of it (71% at the statue's 0.30). Measured: the
  zero-action statue switched to the seed-44 policy at step 460 (blended
  over 50 steps) passes a whole 40-episode panel (37/40 clean, bound
  0.817) with the left pad on its front edge for 0.56–0.66 of the window in
  every clean episode; switched at step 400, it is 19/40 and fails. So a
  policy whose pads drift onto their front edge after about the first third
  of the window certifies. A per-foot fraction of window steps with the
  CoP past 0.9 of the half-length would separate that construction (0.56–
  0.66; the statue at most 0.18, the 369k-step D-D27 study checkpoint at
  most 0.024), but command jitter on every step at σ 0.05 already reads
  0.24–0.55 on it, so it needs a tremor calibration before it can be a
  bar. (2) A pad rolled about 1° that also carries seed 44's toe-down pitch
  loads three corners, the two toe corners and one heel corner, and reads
  about 0.76 (0.70–0.77 on scripted probes, with the digits at 5–7% of the
  foot's load against the statue's at least 11%), so it passes; a
  toe-down sole-pitch bar would separate it, on a thin margin (the statue
  −0.21°, the study checkpoint −0.35 to −0.43°, these probes −0.54 to
  −0.73°, seed 44 −0.5 to −0.9°). (3) A pad loaded on its heel and toe in
  turn inside one control step reads centred (physically implausible behind
  the 10 Hz command filter). Lifting the pad, or grazing the floor with
  it, hides no edge stance: a step with no loaded sole contact counts as on
  the edge. The bar exists only for the T. rex and the compsognathus: the
  velociraptor's foot has no sole and the brachiosaurus's soles are
  ellipsoids, so the metric is unmeasured on both (on the velociraptor the
  support-geom duty and coverage bars read where its foot is loaded), and
  the box-soled robot and dibothrosuchus are not on `stance_quality/v2` (the
  robot on v1, dibothrosuchus on `reward_and_length/v1`), so each gets a bar
  when it adopts v2, measured on its own statue (their statues read
  0.30–0.69 here). The compsognathus adopts v2 in its revision (D-D26),
  which comes after D-D28 and declares 0.70, measured on its r2 statue
  (the audit's §11; the r1 readings in these records do not carry
  over: the r2 statue reads 0.143–0.155 over 200 episodes on five seed
  blocks, heel side, against r1's 0.454–0.473). Its pad and digits are one
  plate, and the digits reach past the pad's front edge, so the reading
  follows the centre of mass along the foot and nothing else sees it: MTP
  servo targets or ankles that lean both pads onto their front or heel
  edges, level, on about two loaded corners with every digit loaded, certify
  whole 40-episode panels without the bar, and 33 of the swept-back crouch's
  40 episodes are clean without it, all at 0.84–1.00; the bar refuses each.
  The same gap (1) is open there at (0.70 − s)/(1 − s) of the window, 65% at
  the statue's 0.15: the MTP servo targets ramped linearly onto the front
  edges over steps 450–550 pass 10/10 (0.62–0.63, the pads past 0.9 of the
  half-length on 54% of the window), over steps 350–450 fail 10/10
  (0.72–0.73, 67%). That split is the ramp's, not a margin: ramping the raw
  action linearly instead, which the soft-cubic map turns into a lagging
  target, the same 350–450 posture reads 0.69–0.71 and passes 11 of 20
  (seeds 17042–17061).

- **MEDIUM** — **a step that diverges in MuJoCo returns as an ordinary step
  (executed 2026-09-30).** `BaseDinoEnv.step` (`base_env.py:1244-1384`, its
  frame-skip loop at :1302) never reads `data.warning`, and on a bad `qvel`
  or `qacc` MuJoCo resets the state to the model's default pose (`qpos0`) and
  carries on. On the T-Rex stance config, setting `qvel` to 1e12 and taking
  one zero-action step printed MuJoCo's "The simulation is unstable" warning
  and returned `terminated=False`, reward 1.101 and a finite observation, with the pelvis
  at 0.977 m (`qpos0`, not the settled stance) and
  `warning[mjWARN_BADQVEL].number` at 1, but no `termination_reason`. In
  training, a mid-episode divergence silently moves the animal back to that
  pose (clearing its drift and velocity) while the push clock, VecNormalize's
  statistics and the gate's length and reward metrics take in the impossible
  trajectory. Fix: compare the warning counters around the frame-skip loop
  and end the episode with its own `termination_reason`, logged. No cleanup PR
  owns it. (2026-08 gap review EP1)

- **MEDIUM (design gap)** — **every push of a recovery stage has the same
  magnitude (read from the code 2026-09-30).** `push_schedule` draws each
  push's start and heading from hash lanes `k*2` and `k*2+1`
  (`perturbation.py:74-98`), and `external_push_force` applies one scalar
  `force_newtons` to every push (:101-119), while the recovery design
  pre-generated push "times, directions, and magnitudes" per episode
  ([STAGE1B_IMPLEMENTATION_PLAN.md](STAGE1B_IMPLEMENTATION_PLAN.md) W1), so a
  certified recovery policy has met one force. Adding magnitudes needs a hash
  lane disjoint from {`k*2`, `k*2+1`} (renumbering the lanes would silently
  change every existing schedule and its null-controller pairing), and it
  changes the transition, so `SCHEDULE_IMPLEMENTATION`
  (`task_fingerprint.py:77`) and every recovery task digest move. No cleanup
  PR owns it. (2026-08 gap review EP2)

- **LOW** — **after a noisy reset one foot spawns just above the floor and
  reads 0 N for the first few steps (executed 2026-09-30).**
  `_settle_root_on_ground` (`base_env.py:1556-1587`) shifts the root so the
  lowest geom sits at the home clearance, which grounds one foot and can
  leave the other millimetres up. On the T-Rex stance config
  (`reset_noise_scale` 0.05, zero action), seeds 0–3 read 0.0 N on one foot
  for the first 5, 3, 3 and 4 control steps, with a step reward of 1.86–1.99
  on those steps against 3.19–3.38 within two steps of the second foot's
  touchdown. The stance duty gate is unaffected
  (`settle_steps = 200` excludes the transient); the reward lost is small and
  the same for every policy, but any measurement window that opens at a
  reset reads the same zeros. Fix: settle each foot, or leave the first
  post-reset steps out of the support-conditioned terms; either one moves the
  digest-snapshot golden's `reward` lines, which capture noisy resets. No
  cleanup PR owns it. (2026-08 gap review EP4)

- **LOW** — **a `stance_quality/v1` stage that leaves out `min_eval_episodes`
  gets a 10-episode panel floor in training and in the recorded-gate reader,
  and a 40-episode one in the stance report and publication (read from the
  code 2026-09-30).** The SB3 curriculum manager's `StageThreshold` falls
  back to `DEFAULT_MIN_EVAL_EPISODES` = 10 (`curriculum/manager.py:32-39,113`)
  and hands it to the stance gate as a field copy (`stance_thresholds`,
  :116-132), the in-training diagnostics callback sizes the gate's duty count
  from the same 10 (`eval_diagnostics.py:636-639`), the recorded-gate reader
  the Drive summary uses falls back to it too (`reporting/gates.py:147`), and
  the species catalog publishes it (`species_catalog.py:545`), while the
  stance report, its probes and publication fall back to
  `DEFAULT_MIN_EVAL_EPISODES_STANCE` = 40 (`curriculum/stance_gate.py:142`,
  `reporting/stage_artifacts.py:239,629`, `result_bundle/evidence.py:511`).
  The gate schema requires `min_eval_episodes` of the floor-truth stance
  kinds (`stance_quality/v2` and the six-heading certificate) but not of
  `stance_quality/v1` (`task_success/v1`, which also required it, was
  removed with the hunt nodes, decision D-D59), so nothing stops a new
  `stance_quality/v1` TOML from leaving it out; the two that exist (the
  Compsognathus robot's and the Allosaurus' `stance.toml`) set 40, so it is
  latent. #519 named both values and kept them apart on purpose, because
  unifying them would change which panels the SB3 manager certifies (gap
  review DU4, part 2 not applied). Fix: require `min_eval_episodes` for
  `stance_quality/v1` in the gate schema, as the floor-truth kinds do (both
  `stance_quality/v1` TOMLs already set it), or unify the two fallbacks. No
  cleanup PR owns it. (2026-08 gap review DU4)

- **LOW** — **a hand-run T-Rex tool still assumes the physics r11 plant or
  older (found 2026-11-23).**
  `environments/tyrannosaurus_rex/scripts/replay_walk.py` builds the current
  environment from an archived Pilot4 stage's recorded config, which names
  the bite arguments and prey range the environment now refuses (the
  arguments since `e3cccca`, the range since note 24), and feeds a 63-wide policy
  the current 59-wide observation, so it runs only in a checkout of
  `3368abe` or earlier. It does not run in CI or training. Fix: port the
  replay to a re-trained walk, or retire it with its review's Pilot4 transfer.
  Updated 2026-11-23: on a later checkout the script now stops before it
  reads a stage and names `3368abe` (`require_a_runnable_checkout`, pinned by
  `environments/tyrannosaurus_rex/tests/test_replay_walk.py`). The other tool
  this item named is fixed: `harnesses/recovery_offdist_panel.py` sized its
  `mag120` and `mag210` rows as `1.5 × target / 165.501`, the push that
  capture multiple 1.5 derived on physics r7 through r11, so on physics r12
  (`configs/plant_versions.toml` note 24, 159.7 N at 1.5) they pushed about
  115.8 N and 203.0 N; it now converts 120 N and 210 N to multiples on the
  plant it builds (`multiple_for_force`, through `derive_push_parameters`).

## Infrastructure

- **LOW** — `metrics.py` `velocity_consistency` explodes when mean velocity
  ≈ 0; thread-unsafe CSV appends under concurrent local runs. (June §3.3;
  CODE_REVIEW §2.1#1)
- **LOW** — **a node re-entered from its own `gate_verdict.json` prints a
  0.01 s control step in `training_summary.txt` (reproduced 2026-09-28).**
  When the SB3 notebook's chain loop re-runs in the same `RUN_DIR`, it takes
  a certified node's results from its verdict
  (`NODE_RESULTS[NODE.id] = dict(stage_result)`, the `same_run` branch). The
  verdict's `stage_result` projection (`_PERSISTED_STAGE_RESULT_KEYS`,
  `result_bundle/gate_verdict.py:72`) omits `sim_dt`, so
  `write_training_summary` falls back to 0.01 s
  (`reporting/text_summaries.py:158`): a reused node of either
  Compsognathus model shows 1,000 steps as "10.00s sim time" instead of
  20 s, while a freshly trained one shows 20 s. No gate, verdict or digest
  reads it. CU-2 fixed the other 0.01 s fallback, in summaries built from
  `evaluations.npz`, but not this one. Fix: persist `sim_dt` in the
  projection (new verdicts only), or give `write_training_summary` the
  node's control step; no cleanup PR owns it yet. (2026-09 CU-2 review)
- **LOW** — **the final-checkpoint replay runs on raw observations, with no
  warning, when the final pair has lost its sidecar (reproduced
  2026-10-01).** `_record_stage_replays` records the selected checkpoint only
  from a pair whose `_vecnorm.pkl` exists and otherwise skips it with a
  warning (`reporting/stage_artifacts.py:2029-2037`). The final replay checks
  only the `.zip` (`:2066`) and passes `<stage>_final_vecnorm.pkl` (`:2007`,
  `:2091`), which `evaluation.record_stage_video` ignores when the file is
  missing (`evaluation.py:316`): the policy is replayed on unnormalised
  observations, which makes it a different policy, and its `_final.mp4` is
  written as usual. In a probe through the unchanged library code (the SB3
  classes, the loader, the plant checks and mediapy stubbed; the same at
  `02d98ca`), the selected replay's policy saw normalised observations, the
  final one's raw, and nothing was logged at WARNING or above. It is latent:
  every caller today hands the replays an intact final pair (the SB3
  notebook's chain loop and manual cell call `generate_stage_artifacts` right
  after training saved both files, and its JUDGE branch judges only an
  intact final pair, `checkpoint_pair_problem`). A direct call on a stage
  directory whose final `.pkl` is missing, for example one copied without
  it, would publish the misleading video. Fix: skip the final replay with a
  warning when its sidecar is missing, as the selected replay does, or have
  `record_stage_video` refuse a named `vecnorm_path` that does not exist; no
  cleanup PR owns it yet. (2026-10 CU-8b scouting)
- ~~**LOW** — **a same-stage resume does not continue `diagnostics.npz`
  (reproduced 2026-10-13).**~~ **FIXED** (2026-11-24): a PPO continuation
  seeds `DiagnosticsCallback` from the `diagnostics.npz` beside the
  `evaluations.npz` it seeds (`DiagnosticsCallback.seed_prior_diagnostics`,
  which `seed_resume_eval_state` calls first, so a missing or ragged
  evaluation record does not stop it). Each of the file's three clocks keeps
  its rows at or before the resumed step, and a series the prior file lacks
  reads NaN on them. A session's first rewrite of a `diagnostics.npz` already
  there copies it aside as `diagnostics.prior_<UTC>.npz`, so a resume that
  cannot seed (an unreadable file, or a SAC or curated `--load` into a
  directory that holds no training records yet) loses no row either, and a
  file a seeded session wrote, its back-filled NaN rows included, seeds
  again. The stage figures
  (`visualization.py`) and `scripts/compare_run_diagnostics.py` read
  `diagnostics.npz` by name and now cover the whole stage
  (`tests/test_resume_record_integrity.py`). On a live-command stage the seed
  also carries the four command-tracking series (`tracking_in_tolerance`,
  `tracking_error_v`, `tracking_error_yaw`, `reward_tracking`), so the
  command-tracking figure's tracking-fraction panel (2026-11-24), which marks
  the step its recorded history starts at, covers the whole stage of a
  continuation too. Its resume caveat (merged 2026-11-25) is drawn for a PPO
  continuation only when the series starts after the checkpoint the stage
  continued from, and for every recorded resume that cannot seed (a SAC or
  curated `--load`, which restarted the step counter) wherever the series
  starts, naming the checkpoint whose earlier steps the record does not
  hold. As it was: the callback started its rollout history empty in every
  session and nothing seeded it, so a resume into the stage directory it
  continued overwrote the prior session's rows at its first save. (2026-10
  review of the `gate_progress.npz` resume fix)
- **LOW** — **a result-bundle JSON file that is not UTF-8 escapes as a raw
  `UnicodeDecodeError` instead of the reader's own refusal (reproduced
  2026-10-01).** These readers read with `encoding="utf-8"` but catch only
  `OSError` and `json.JSONDecodeError` (and their own errors), so the
  decode error, a `ValueError`, passes through: `result_bundle/audit.py:208`
  (the summary), `:245` (the artifact manifest) and `:357` (the plant
  identity), `reporting/bundles.py:538` (the previous manifest in
  `save_result_bundle`), `result_bundle/evidence.py:943` (the gate
  resolution), `result_bundle/manifest.py:216` (`verify_artifact_manifest`),
  `result_bundle/provenance.py:371` (`load_provenance`) and
  `curriculum/baseline_watch.py:68` (`read_zero_action_baseline`, which
  training calls when it builds its callbacks, so there a bad file stops
  the stage before it trains instead of skipping the advisory watch).
  `initialize_result_bundle` reads an existing `provenance.json` with no
  handler at all (`result_bundle/provenance.py:252`), so a file there that
  is not UTF-8, or not JSON, escapes raw too, also from
  `save_result_bundle` given a `run_id`. In a probe, a run directory
  holding only a `summary.json` with a 0xff byte inside a string makes
  `audit_result_bundle` raise `UnicodeDecodeError` instead of returning an
  audit error, and `load_provenance`, `verify_artifact_manifest` and
  `read_zero_action_baseline` do the same on such a file of their own; the
  same at `08f7bdd`. It is latent: the project's writers write UTF-8, so
  only a corrupt or hand-edited file reaches it. CU-8c closed the leak for
  `stage_config.json` alone, by catching `ValueError` in
  `save_result_bundle`, `audit_result_bundle` and
  `validate_evaluation_evidence`. Fix: catch `ValueError` in these
  handlers too, and give `initialize_result_bundle`'s read its own
  refusal; no cleanup PR owns it yet. (2026-10 CU-8c review)
- **LOW** — **the stale-manifest error does not name the MuJoCo version
  (read from the code 2026-09-30).** The package pins `mujoco==3.10.0`
  (`pyproject.toml:25`), the version the plant digests are built on. An
  install that gets another MuJoCo stops training in
  `current_plant_identity` with "generated plant manifest is stale for
  <species>/<embodiment>; run the plant-contract check before training"
  (`plant_contract/manifest.py:484-486`); only the check it points to names
  the cause ("plant manifest generation requires MuJoCo 3.10.0", :169-171).
  In the review's reproduction (MuJoCo 3.12.0) training stopped before its
  first step, so the cost is a two-step diagnosis, not a wrong result. Fix:
  name the MuJoCo version in the stale-manifest error. No cleanup PR owns
  it. (2026-08 gap review OP9; its other half, the species
  `requirements.txt` files that took any MuJoCo from 3.0.0, was fixed by
  deleting them.)
- **MEDIUM** — **nothing promotes a run into `results/`, nothing can mark a
  published result current or stale, and `results/` cannot hold a chain built
  on a trunk (read from the code 2026-11-26).** No function, script or
  checklist copies a Drive bundle into `results/`: the Drive summary's
  strict-promotion cell only validates, and `results/README.md:75-76` says
  promotion is deliberate. `results/` holds no summary since the hunt's
  removal deleted the four historical ones (D-D59,
  `results/README.md:20-23,37-57`), so no model page shows a run. No code path
  marks a result current or verified: `result_bundle/provenance.py:416-417`
  and `reporting/summaries.py:48-49` write `model_revision_status =
  "historical"` and `verification_status = "unverified"` on every run, the
  catalog passes them on (`species_catalog.py:863-866`) and the website calls
  every published summary a historical record
  (`website/src/components/SpeciesCatalog/index.tsx:365-368`); nothing
  compares a published summary's plant, task or gate digests with the current
  manifest, so a result can neither become current nor be flagged stale. The
  layout holds one summary per model and algorithm: the catalog reads
  `results/*/*/*/summary.json` (`species_catalog.py:1234-1237`) and never
  aggregates across bundles (D-B10, :917-920), and a cross-run ancestor writes
  no CSV row or summary stage in the run that reused it
  (`sb3_training.ipynb:817-818`), so a chain built with `TRUNK_FROM`, or a
  deliverable's seed replicates, cannot be published whole. Fix: a promotion
  tool and checklist built on `validate_result_bundle(...,
  require_publishable=True)` that refuses screens and quick tests, compares
  the plant, task and gate digests with the current manifest and sets the two
  status fields; a run-id (or ancestor) level under
  `results/<species>/<embodiment>/<algo>/`. (2026-11 notebook review:
  no-promotion-path, no-staleness-or-current-status,
  results-layout-cannot-publish-trunked-chain)
- **LOW** — **the run summary and the CSV row leave out a node's verdict, its
  reasons and its certified statistic, and no page ties a run together (read
  from the code 2026-11-26).** `write_training_summary`
  (`reporting/text_summaries.py:188-256`) writes the `training_summary.txt`
  the SB3 notebook prints after every node
  (`notebooks/sb3_training.ipynb:980`, called at :1682, :1943 (the publish
  cell) and :2098): per node its timesteps, duration, final and best
  evaluation and a "Best model:" path (`text_summaries.py:238-244`), the last
  even for a node that failed, and no gate kind, verdict, failure reason,
  reuse origin, certified statistic or handoff.
  `stage_summary.txt` has the verdict and the failures (:166-185) and, on a
  walk, the family statistics and the handoff (`_command_tracking_lines`,
  :37-80), but no gate kind, and on a stance not the bound its verdict records
  (`stance_heading_clean_seed_lcb`, `result_bundle/gate_verdict.py:124`),
  which reaches only the stance reports and the CSV row
  (`reporting/csv_output.py:345`). The chain loop's own line is `<node>: gate
  PASSED (<kind>); handoff <stem>.zip` (cell 20, `sb3_training.ipynb:1690`),
  and the report paths go to `logger.info`, which the notebook never shows. In
  `collected_results.csv`, `CSV_METRIC_COLUMNS` (`csv_output.py:80-99`) still
  lists `gate_evaluable`, `gate_reason`, `quality_score` and `quality_rank`,
  left from the retired sweep and scoring, and nothing writes them, so
  `gate_reason` is empty on a FAIL (the early-stop entry under Training / RL
  reads the same empty column). The Drive summary's tables are the Drive
  tables entry under Notebooks. No page links a run's nodes: nothing writes a
  run-level index of each node's `gate_progress.png`, `training_history.html`,
  reports and replays. Fix: a per-node verdict block (gate kind, PASS/FAIL,
  reasons, headline statistic, reused-from, handoff) in `training_summary.txt`
  and the chain loop's print, with the gate kind and headline statistic in
  `stage_summary.txt` too; write `gate_reason` from `gate_failures` and drop
  the three dead columns; a run-level `index.html` linking every node's
  figures, reports and replays. (2026-11 notebook review:
  no-chain-level-human-report, summary-omits-verdicts,
  gate-reasons-not-propagated, and the CSV half of
  collected-results-dead-and-kind-dependent-columns)
- **LOW** — **the Allosaurus stance records a 0% success rate for an event it
  cannot raise (read from the code 2026-11-26).** `evaluate_stage_checkpoints`
  records `mean_success_rate` and `best_model_success_rate` for every node of
  a model that declares success keys
  (`reporting/stage_artifacts.py:1950-1954`, :2077 and :2102), by model, not
  by node. Six models declare none since the hunt's removal; the Allosaurus
  declares `course_complete`
  (`environments/allosaurus_fragilis/base_env.py:29`), and its stance can
  never complete the course: under `command_mode "none"` a course exists only
  while `forward_vel_weight` is positive (:191-201, :204-213), and its
  `stance.toml` sets 0.0 (:6). So its stance prints "Eval: success_rate=0%"
  and "Success rate: 0%" in `stage_summary.txt`
  (`reporting/text_summaries.py:154-155`), and its CSV row and summary record
  0.0, on a node no success event applies to. On its walk the rate is course
  completion, which the `command_tracking/v1` gate does not read. Fix: record
  a success rate only for a node whose task can raise the event (a stage with
  a course), and label it the task event, never the gate. (2026-11 notebook
  review: success-rate-on-non-task-nodes)
- **LOW** — **a passing recovery verdict records none of the statistics it was
  judged on (read from the code 2026-11-26).** `RecoveryGateResult` carries
  the success bound and the paired policy-minus-statue bound
  (`curriculum/recovery_gate.py:197-237`), but `_recovery_stage_gate` keeps
  only `result.passed` and `result.failures` (`reporting/gates.py:1811-1831`),
  so the two numbers appear only inside a failure string.
  `gate_verdict._PERSISTED_STAGE_RESULT_KEYS`
  (`result_bundle/gate_verdict.py:72-150`) has stance and command fields and
  no recovery one, the CSV row adds statistic columns for the two stance kinds
  and the walk only (`reporting/csv_output.py:328-350`), `stage_summary.txt`
  prints none, and `gate_progress.png` has no recovery panel. The chain loop
  prints the panel's count, `<k>/<n> episodes recovered`
  (`sb3_training.ipynb:1659`), and the per-episode evidence is on disk
  (`recovery_episodes_<controller>.csv`, `recovery_evaluation.py:346-359`),
  but no artifact states the bounds. T. rex and the two Compsognathus models
  have a recovery node. Fix: return the result's counts and bounds from the
  gate and persist them in the verdict's `stage_result`, the CSV row and
  `stage_summary.txt`, and draw a recovery panel in `gate_progress`. (2026-11
  notebook review: tyrannosaurus-recovery-verdict-records-no-statistic)
- **LOW** — **user-facing text names a node by its legacy stage number, which
  on three models is not its directory's prefix (read from the code
  2026-11-26).** `stage_summary.txt` is titled
  `Mesozoic Labs: Stage <N> Summary`
  (`environments/shared/reporting/text_summaries.py:110`; `Stage: <N> (...)`,
  :114), each node of `training_summary.txt` is headed `Stage <N>: <name>`
  (:224), the older per-node figures' legends read the same
  (`visualization.py:138`, :391, :495, :626), and the judge logs
  `Stage <N> curriculum gate: PASS` or `FAIL`
  (`reporting/stage_artifacts.py:1811-1813`), where `<N>` is the legacy number
  (1 the stance, 2 the locomotion) or the bare id. On the T. rex and both
  Compsognathus models the walk sits in `03_locomotion` and reads "Stage 2",
  and the recovery reads "Stage recovery"; the notebook explains the two
  numberings only for `RESUME_STAGE` (cell 17,
  `notebooks/sb3_training.ipynb:1101`). The legacy numbers and the file
  prefixes are deliberate (`stage_manifest.stage_dirname`,
  `stage_manifest.py:612-639`); the display is not. Fix: show the node id in
  this text, the legacy number only in parentheses where a file name needs it.
  (2026-11 notebook review: stage-number-wording, and the legend half of
  figure-labels-units)
- **LOW** — **every stance report but the T. rex's logs a spurious
  `stance_probe_filter_hz` warning (read from the code 2026-11-26).** The
  filtered-action probe passes
  `curriculum_kwargs.get("stance_probe_filter_hz")` to `_probe_cutoffs`
  (`environments/shared/reporting/stage_artifacts.py:1097`), which wraps a
  missing key's `None` in a list, fails `float(None)` and logs
  "stance_probe_filter_hz entry is not a number: None; ignoring it" at WARNING
  (:1050-1057) before it returns no cutoff. The probe is skipped, as it should
  be, under a warning that reads as a config error. Only the T. rex stance
  sets the key (`configs/tyrannosaurus_rex/anatomical/stance.toml:1172`), so
  the velociraptor, both Compsognathus models and the Allosaurus, whose
  stances write a stance report, log it on every artifact pass (the review saw
  it on the Allosaurus). The impulse and heading probes return early on a
  missing key (`stage_artifacts.py:1317`, :1399). Fix: the same early return
  for this key. (2026-11 notebook review: probe-cutoffs-none-warning,
  probe-filter-none-warning)
- **LOW** — **the older per-node figures mislabel, mis-scale or hide what they
  plot (read from the code 2026-11-26).** Report only: no gate or bundle check
  reads them. In `visualization.py`:
  - `behavioral_metrics.png`: "Distance Traveled (path length)" is the rollout
    mean of the per-step running total `distance_traveled`
    (`base_env.py:1703`, `:1724`, averaged by `diagnostics.py:319-324`), which
    depends on where episodes start within the rollout and is no episode's
    path length; the heading band is the mean plus or minus the within-rollout
    std, not clipped at cos θ = 1 (`visualization.py:503-513`).
  - `foot_contacts.png`: "Diag A (FR+RL)" and "Diag B (FL+RR)" are the
    element-wise maximum of two feet's rollout-mean forces (`:647-648`),
    neither a sum nor a contact pattern, under a title that promises the
    walk/trot pattern (`:689`), and neither axis gives a unit (`:681`,
    `:688`).
  - `locomotion_health.png`: each Termination Breakdown bar is a reason's
    share of a rollout's terminations, averaged over the rollouts that had any
    (`:399-401`, "Mean Fraction", `:452`; `diagnostics.py:348-360`), with no
    counts and no time axis: the review's velociraptor quick test, 5 falls in
    about 55 episodes, all nosedives, read nosedive 1.0. `diagnostics.npz`
    already holds the counts (`rollout_episodes`, `rollout_falls`,
    `rollout_term_count_<reason>`, `diagnostics.py:97-104`).
  - `training_curves.png` and `gate_progress.png` draw plain lines
    (`visualization.py:143`, `:156`, `:910`), so a stage with one evaluation,
    as every quick test has, shows empty panels; `command_gate_progress.png`
    uses markers.
  - The legends name a node by its legacy stage number ("Stage recovery:
    Recovery" on a recovery node, `S<number>` in the reward panel, `:416`; the
    stage-number entry above), and the locomotion-health and
    behavioral-metrics panel titles repeat the model and algorithm their
    suptitle gives.

  Fix: plot the episode-end distance (or say what is plotted), clip the
  heading band, draw contact per foot or drop the diagonal panel, draw
  termination counts over training, add markers, name nodes by id and give
  units; or retire these figures once `gate_progress`, the command figures and
  the training history cover them. (2026-11 notebook review:
  behavioral-metrics-semantics, locomotion-health-panels-misleading,
  single-point-series-invisible, and the titles and units half of
  figure-labels-units)
- **LOW** — **the Reward Decomposition panel picks its terms from the TOML's
  weights, so it hides terms paid at a constructor default and draws terms
  nothing recorded (read from the code 2026-11-26).**
  `plot_diagnostics_graphs` draws a reward key when the stage's `[env]` or its
  stand block names a non-zero weight for it, through
  `_REWARD_KEY_TO_CONFIG_WEIGHTS` (`visualization.py:247-280`, `:348-365`), or
  when the key has no entry there (`reward_speed`, `reward_gait_symmetry` and
  `reward_home_pose`, `:361-363`). `DiagnosticsCallback` saves every one of
  its reward keys, as NaN where the env never emitted it
  (`diagnostics.py:319-327`, `:682-685`). On the Allosaurus stance (checked on
  its env 2026-11-26) the panel draws `drift` and three all-NaN legend entries
  for the unmapped keys, and hides `alive` (1.0 a step at the constructor's
  `alive_bonus = 1.0`, `allosaurus_fragilis/base_env.py:60`, which
  the stance TOML does not restate), `posture` (paid at a fixed weight,
  `:159-162`; the env has no `posture_weight` for the map to read) and
  `energy`; its walk restates `alive_bonus` and `energy_penalty_weight` but
  still hides `posture`. Colours are tab10 by key index
  (`visualization.py:417`), so terms ten apart share a colour and line style.
  Fix: draw the terms whose recorded series is finite and not all zero, read
  from `diagnostics.npz` itself, each in its own colour. (2026-11 notebook
  review: reward-decomposition-selection)
- **LOW** — **two locomotion readouts look up keys the episode aggregate never
  holds, and print 0 for what was not measured (read from the code
  2026-11-26).** `LocomotionMetrics.aggregate_episodes` prefixes each
  per-episode key with `mean_` and `std_` (`metrics.py:297-301`), so an
  episode's `mean_forward_velocity` aggregates to
  `mean_mean_forward_velocity`. The supplementary evaluation's log line in the
  command line's `curriculum` (`CurriculumCallback._log_locomotion_metrics`)
  looks up `mean_forward_velocity` and `mean_pelvis_height`
  (`curriculum/advancement.py:288-290`) and so prints neither. `evaluate()`'s
  report (`evaluation.py:801-875`) tests `"mean_heading_alignment" in agg`
  (`:857`), which never holds, so it never prints the heading alignment
  either, and reads every other value with `agg.get(key, 0)` (`:814-855`), so
  a metric the episodes never measured prints as 0.000 instead of n/a. Nothing
  gates on these lines, and in the SB3 notebook they are hidden anyway, since
  no cell configures logging (the section 8 entry under Notebooks). Fix: read
  the `mean_mean_*` keys and print n/a for an absent one. (2026-11 notebook
  review: supplementary-log-key-mismatch)
- **LOW** — **a renderer failure costs a replay's physics-only `_stance.csv`
  with its videos, and the setup cell reports EGL rendering without trying it
  (read from the code 2026-11-26).** `record_stage_video` collects a stance or
  recovery replay's per-frame rows as it renders (`evaluation.py:383-412`) but
  writes `_stance.csv` only after every frame has rendered and every video is
  written (`:447-460`), so an error in `render()` or `mediapy.write_video`
  drops the CSV too, which needs no renderer. `_record_stage_replays` records
  the selected and the final replay under one `try`
  (`reporting/stage_artifacts.py:2437-2541`), so the first failure also skips
  the second. In Colab the SB3 notebook's setup cell (§1) sets `MUJOCO_GL=egl`
  and writes the NVIDIA EGL vendor file without checking for a GPU
  (`notebooks/sb3_training.ipynb:55-60`), then prints "Colab setup complete
  (EGL rendering enabled)." (`:134`) without rendering a frame: a runtime
  where EGL does not work first shows it as the replays' logged warning, after
  training. Fix: write the CSV in a `finally` (or before the videos), give
  each replay its own `try`, and have the setup cell render one frame and
  print the backend that worked. (2026-11 notebook review:
  fu-colab-runtime-specifics-egl-backend-unprobed-replay-and-stance-csv-lost-together)
- **LOW** — **the `stance_quality/v1` text report prints fixed statue support
  duties for every model, and its support lines read touch only (read from the
  code 2026-11-26).** `reporting/stance_report.py:1749-1750` prints "bilateral
  support ... (statue 0.998, not gated)" and "single support ... (statue
  0.002, not gated)", literals from a T. rex statue panel of an older plant,
  whatever model the report is for. The two v1 stances today are the
  Compsognathus robot's and the Allosaurus'
  (`configs/compsognathus_longipes/robot/stance.toml:74`,
  `configs/allosaurus_fragilis/anatomical/stance.toml:22`); the Allosaurus
  statue falls in every episode (88.5 of 1,000 steps on seeds 3042-3043,
  measured 2026-11-26) and never produces those duties, and a v1 report rolls
  no statue panel to measure them. Both lines come from `derive_stance_info`
  over the env's `r_foot_contact` and `l_foot_contact`, which on both
  Compsognathus models are the touch sensors
  (`compsognathus_longipes/base_env.py:258-262`), so on the robot they count
  sole-on-sole force as support (the robot entry in Training / RL: its
  observation reads floor contact since policy interface r4, the report does
  not), with no floor-truth cross-check until PR-G2 and PR-G8. Fix: print the
  model's own statue duties (rolled beside the panel, or read from the stage's
  zero-action baseline) or drop the literals, and add a floor-contact support
  line beside the touch one. (2026-11 notebook review:
  allosaurus-v1-report-hardcoded-statue-duty,
  compsognathus-v1-report-touch-bilateral-and-hardcoded-statue)
- **LOW** — **the training-history report draws T. rex's actuator list,
  exploration reference and first leg pair on every model (read from the code
  2026-11-26).** In `reporting/training_history.py`, `KEY_ACTUATORS =
  ("hip_pitch", "hip_roll", "knee", "ankle", "tail_3")` (:257) chooses the
  `10_policy_std` page's first panel and the checkpoint table's std columns
  (:5407-5472, :6604-6606): the anatomical Compsognathus' `r_toe_act` and
  `l_toe_act` and the Allosaurus' `r_mtp` and `l_mtp` are left out, and
  neither model has a `tail_3`. The exploration page draws
  `EXPLORATION_REFERENCE_STD` 0.08 (:352), labelled "stance3 line-2 floor 0.08
  (not applied)", on every model and node (:5664-5676); stance3 is a T. rex
  stance design, and the label does not say so. `lr_columns` keeps the first
  left-minus-right column per joint and the first hip-roll pair (:513-523), so
  on the dibothrosuchus, whose ledger lists the `fl_minus_fr` offsets before
  `rl_minus_rr`, the `09_asymmetry` page plots only the front pair, with no
  pair named in its titles (:3417-3436; the caption names "the first foot pair
  on a quadruped" for hip roll alone), while the rear pair carries about 73%
  of standing load (the review's measurement). Fix: derive the key actuators
  per model from its leg actuators, plot every left/right pair (or the
  load-bearing one) and name it, and draw the exploration reference only where
  it was set, labelled as the T. rex's. (2026-11 notebook review:
  compsognathus-history-key-actuators-miss-toe,
  cmp-exploration-page-trex-reference-everywhere,
  dibothrosuchus-asymmetry-page-forelimbs-only)
- **LOW** — **the Allosaurus runs no evaluation recorder, has no leg pairs or
  spin keys, and its policy probe records nothing and logs a traceback on
  every PPO node (read from the code and a state-bank build 2026-11-26).** Its
  pads are ellipsoids
  (`environments/allosaurus_fragilis/anatomical/assets/allosaurus_fragilis_anatomical.xml:211,241`),
  so, like the velociraptor and the brachiosaurus, it has no box sole and its
  evaluations run no floor-truth recorder; its training-history report is then
  mostly placeholder pages (about 14 of 20 in the review's quick-test run,
  2026-11-22). `sagittal_leg_dc_pairs` finds no pair on it (its leg joints are
  torque motors, and the function keeps only joints driven by a unit-gear
  hinge position servo, `eval_diagnostics.py:131-160`, :180-183), so it has no
  leg-DC columns, and its info
  (`environments/allosaurus_fragilis/base_env.py:166-178`) carries none of the
  `heading_alignment`, `pelvis_angular_vel`, `pelvis_yaw_vel` and
  `spin_instability` keys `diagnostics.py:199-209` records. The policy probe's
  state bank keeps the statue at steps 199 and 599 and three ankle leans at
  steps 400 and 599, on seeds 3042 and 3043 (`BANK_SPEC`,
  `policy_probe.py:77-88`); a row whose episode ends first is dropped, and
  when none is left `build_state_bank` raises "no bank state could be built"
  (:285-286). The Allosaurus statue falls within about 90 steps on both seeds,
  on the stance and on the walk (measured 2026-11-26; on the walk in every
  episode, `configs/allosaurus_fragilis/anatomical/locomotion.toml:124-127`),
  so `_on_training_start` logs the failure with its traceback and leaves the
  bank unset (`policy_probe.py:518-540`), and every later update returns at
  once (:729-730). `policy_probe.npz` then gets no row: not the bank values,
  and not the per-update std per actuator, normalizer drift or rollout clip
  share either, which read only the parameters and the rollout buffer, so the
  training-history pages that read it are empty on every Allosaurus node.
  Nothing names the gap: the artifact guide's recorder table leaves the
  Allosaurus out (the artifact-guide entry under Configs, docs & website).
  Fix: write the probe's bank-free columns without a bank, and when the statue
  falls build the bank from a reference stand or the policy's own rollouts, or
  skip it with one warning line; add the Allosaurus' leg pairs and spin keys;
  name it in the guide with the pages it cannot fill (or attach a recorder).
  (2026-11 notebook review: training-history-pages-not-model-or-role-aware,
  telemetry-ledger-box-sole-only, policy-probe-never-builds-allosaurus)
- **LOW** — **`provenance.json` gives every evaluation role 30 episodes,
  whatever ran, and a stance's checkpoint-selection evaluations are unpaired
  (read from the code 2026-11-26).** `initialize_result_bundle` writes one
  `evaluation_protocols` entry per evaluation seed role, each with the single
  `evaluation_episodes` count
  (`environments/shared/result_bundle/provenance.py:289-296`), and the SB3
  notebook passes 30 (`notebooks/sb3_training.ipynb:514`, cell 8, and :1013,
  cell 15). What runs differs. The in-training evaluation, the
  `checkpoint_selection_evaluation` role, rolls `max(30, min_eval_episodes)`
  episodes (`train_base.py:239-253`): 40 on the five stances and the three
  recoveries that declare 40 (the velociraptor, the T. rex, both Compsognathus
  models and the Allosaurus; the T. rex and both Compsognathus recoveries). A
  walk's selection and certification panels roll their own family counts
  (`min_cruise_episodes`, `min_stand_episodes`). And under
  `command_mode "none"`, every stance and recovery, the evaluation env is not
  reseeded between evaluations (`curriculum/command_evaluation.py:30-31`), so
  each evaluation draws new episodes and `best_model` and `robust_best_model`
  rank checkpoints scored on different draws; only a live-command stage rolls
  the same episodes at every evaluation (`_command_evaluation_plan`,
  `train_base.py:170-192`, decision D-D62). No gate reads the protocol block;
  a reader of the bundle's provenance does. Fix: record episodes and seeds per
  role from what each evaluation ran, and reseed the stance and recovery
  evaluation env before every evaluation, as the walk's is. (2026-11 notebook
  review: eval-protocol-provenance-mismatch)
- **LOW** — **the offline per-actuator scripts take a legacy stage number that
  is not the directory's prefix and cannot name the recovery, never check what
  they load, and no command line re-rolls a recovery panel (read from the code
  2026-11-26).** `scripts/action_bound_report.py` (:96),
  `scripts/joint_excursion_report.py` (:52) and
  `scripts/observation_ablation_report.py` (:187) take the stage as `type=int,
  choices=(1, 2)`, the legacy numbers of the stance and the locomotion
  (`stage_manifest.resolve_stage_key`, :595-609), and
  `scripts/episode_trace_report.py` takes an int (:86). On the T. rex and both
  Compsognathus models the walk is stage 2 but lives in `03_locomotion`, so
  the number a user reads off the run tree names the wrong stage or none, and
  the recovery, which has no number, cannot be named at all. The three build
  the env from the checkout's stage TOML (`config.build_env`, `config.py:598`)
  and load the checkpoint with `load_sb3_checkpoint` passing no plant identity
  (`action_bound_report.py:118`, `joint_excursion_report.py:67`,
  `observation_ablation_report.py:213`), and nothing reads the checkpoint's
  recorded task, so a checkpoint of another stage, task or plant revision is
  scored on the current stage with no warning. Nothing else reports on a
  recovery checkpoint either: `cli eval --stage recovery` is accepted
  (`cli.py:667-672`), but `evaluate()` prints velocity, balance and
  terminations only (`evaluation.py:823-875`), no push or recovery outcome,
  and `roll_policy_panel` (`harnesses/freeze_recovery_gate.py:580`), the panel
  roller the chain loop judges with, has no command line (the module's `main`,
  :872-918, freezes the null panels only). So a recovery checkpoint can be
  re-rolled or inspected only from Python or the notebook, and, as the
  Training / RL entry on verdicts written before the gate-configuration digest
  says, a recovery verdict cannot be backfilled. Fix: take a node id or a
  stage directory, resolve it through the stage manifest (the recovery
  included), and refuse a checkpoint whose recorded plant or task differs, as
  training's loads do; give `roll_policy_panel` a command line that prints the
  per-seed outcomes and both bounds. (2026-11 notebook review:
  fu-offline-analysis-tools-species-coverage-report-scripts-legacy-stage-int-unchecked,
  fu-offline-analysis-tools-species-coverage-recovery-deliverable-no-offline-analysis)
- **LOW** — **the offline actuator scripts assume position servos named `r_`
  and `l_` in a self-contained MJCF, so they misreport, misdrive or fail on
  the Allosaurus and the Compsognathus robot (reproduced 2026-11-26).**
  - `environments/shared/scripts/joint_excursion_report.py:102-117` converts
    every actuator's `ctrl` and `ctrlrange` to degrees; the Allosaurus'
    actuators are torque motors (`<motor ... gear="1" ctrlrange="-12500 12500"
    />` in its MJCF, `allosaurus_fragilis_anatomical.xml:265-274`), so its
    command band, full range, control span and per-step delta are
    newton-metres printed as degrees (the "used" share, a ratio of two of
    them, and the achieved joint excursion, are right).
  - `scripts/actuator_saturation_report.py:61` builds the model with
    `mujoco.MjModel.from_xml_string` from the MJCF's text (:78), which cannot
    resolve the file's relative assets: on the Allosaurus it fails with "Error
    opening file 'textures/hide.png'", on the robot with "Error opening file
    'meshes/roll_motor_bottom.stl'". Its default panel is four of the six
    position-servo models (the velociraptor, T. rex, brachiosaurus and
    dibothrosuchus, :52-57), so the failure appears only under `--model-id`;
    and once a model loads, its table lists position servos only (:86,
    :90-92), so on the Allosaurus, whose ten actuators are all torque motors
    with no `forcerange`, it prints no actuator row and a root-z divergence of
    0.0000 m between the bounded and unbounded plants (run with its assets
    resolved, 2026-11-26).
  - The scripted gait that report drives (`gait_commands`,
    `environments/shared/tests/actuator_bounds_helpers.py:64-92`) gives a leg
    actuator its phase by name prefix (`r_`, `l_` or a quadruped's `fr_`,
    `fl_`, `rr_`, `rl_`). The robot's twelve actuators are named `left_*_act`
    and `right_*_act`, so none matches: every one gets the 0.4× in-phase
    wiggle meant for non-leg actuators, and the report would measure no
    alternating gait.
  - `scripts/observation_ablation_report.py` holds a slice table for the T.
    rex alone (`SLICES`, :85-90) and exits "no observation slice table" on
    every other model (:208-209).

  [SIM_TO_REAL_PLAN.md](hardware/SIM_TO_REAL_PLAN.md):353-356 plans the
  robot's torque table on this tooling. Fix: read each actuator's type and
  report a torque motor in N·m against its `ctrlrange`; load the MJCF by path
  (`MjModel.from_xml_path`) so relative files resolve; phase `left_` and
  `right_` actuators in `_leg_phase`; add the robot's observation slices to
  the ablation. (2026-11 notebook review:
  fu-offline-analysis-tools-species-coverage-allosaurus-torque-plant-in-servo-reports,
  fu-offline-analysis-tools-species-coverage-robot-hardware-tools-misdrive-or-skip)
- **LOW** — **nothing launches the seed replicates a two-seed publication bar
  needs (read from the code 2026-11-26).** The velociraptor and T. rex stances
  declare `certification_seeds = 2`
  (`configs/velociraptor_mongoliensis/anatomical/stance.toml:633`,
  `configs/tyrannosaurus_rex/anatomical/stance.toml:1030`): the deliverable
  stays provisional until two distinct-seed runs of one recipe certify it.
  `replication.discover_replicates` (`environments/shared/replication.py:100`)
  counts such sibling runs, but nothing starts them: the notebook and the
  command line's `train` and `curriculum` each run one `SEED` (`cli.py:351`,
  :433), so the second seed is a second session set up by hand with the same
  knobs. Sweeps were retired on purpose (D-D17;
  [ROADMAP.md](ROADMAP.md):383-384 asks a new decision for a tuning backend),
  but a replicate is not a sweep. Fix: a small launcher, a command-line option
  or a notebook cell, that runs N seeds of one node under one recipe into
  sibling runs of the certification tree, for a model whose stance declares
  `certification_seeds` above 1. (2026-11 notebook review:
  no-replicate-or-sweep-orchestration)

## Post-training artifacts (recommended additions)

1. Persist the curriculum gate history (`CurriculumManager.summary()` →
   `curriculum_state.json`) and the achieved gate metrics in
   `curriculum_results.csv`; include CLI overrides in the saved effective
   config. (July §4; June §6.5)
2. CLI curriculum parity with the notebook: call `generate_stage_artifacts`
   per stage and run the quality eval so `metrics.json` exists on that path.
   (July §4)
   *Update (2026-11-26):* the command line's verdict differs too. A
   `curriculum` node of a `reward_and_length/v1` or `stance_quality/v1` stance
   (the brachiosaurus, the dibothrosuchus, the Compsognathus robot and the
   Allosaurus) records the in-training manager's decision, `ready_to_advance`
   (`environments/shared/train_base.py:3706`), bound to the pair
   `select_handoff_checkpoint` names (:3685;
   `curriculum/checkpoints.py:195-230`): `robust_best_model`, ranked on mean -
   std reward alone. So a command-line PASS can certify a checkpoint whose own
   evaluations never met the gate, where the notebook stops no node on a gate
   pass and judges that handoff once, after the stage
   (`generate_stage_artifacts`). Only the floor-truth stance kinds and the
   walk are re-judged after training on the command line (:3710-3731). The
   command line has no baseline watch either: the callback reads
   `zero_action_baseline.json` from the run directory
   (`curriculum/baseline_watch.py:218`), which only the notebook's preflight
   writes. Fix: the call above, with its verdict, or label these verdicts
   training-time. (2026-11 notebook review: cli-notebook-judging-parity)
3. Write the per-step replay CSV on every node, not only on stance and
   recovery (read from the code 2026-11-26). One flag, `replay_diagnostics =
   stage in (1, "stance", "recovery")` (`reporting/stage_artifacts.py:2451`),
   turns on both the per-frame `_stance.csv` of the selected checkpoint's
   deterministic replay, with each actuator's command and position per step
   (`collect_stance_diagnostics`, :2503 and :2536), and the extra front and
   side camera views (:2452). So a walk gets two default-view videos and no
   per-step actuator record of the checkpoint it hands on. Its
   `command_tracking_trace.csv` (D-D61) records commanded and achieved
   velocity, heading error and course progress on one cruise and one stand
   episode, and no actuator column
   (`reporting/command_tracking_trace.py:63-82`); the per-rollout actuator
   series in `policy_probe.npz` and the per-evaluation ones are not a replay
   of the selected checkpoint. Fix: decouple the CSV from the camera views and
   write it on every node, keeping the extra views on stance and recovery if
   their cost matters. (2026-11 notebook review:
   fu-offline-analysis-tools-species-coverage-walk-hunt-nodes-no-per-actuator-record)

## MuJoCo models (July 2026 model review)

Fixed on this branch: the brachiosaurus could not physically hold any torso
height in its alive region (home-keyframe-controlled settle z≈0.68,
straight-leg command z≈0.70, vs `healthy_z_range` floor 1.0 and height target
1.2) — leg springs now reference the stance angles and the leg servos are
stronger with bounded force, giving an actively servo-held home stand at
z≈1.13, level, all four feet grounded. The T-Rex's former step-77 neutral
nosedive is also fixed: a shallow plantar contact, stance-referenced springs,
mass-scaled gait servos, named-home residual actions, and live load-bearing
foot sensors give it a full-horizon neutral stand; see
[the home-equilibrium investigation](investigations/TREX_HOME_EQUILIBRIUM.md).
All three models now use
`integrator="implicitfast"` and bounded
`forcerange` on position actuators, sized to clip impact/reset spikes only,
with gait-critical actuators at 1.5×kp on every species (raptor hip
pitch/knee/ankle; T-Rex hip pitch/knee/ankle; brachiosaurus all four hip
pitches — the raptor knee joined the 1.5× set in July 2026 after measuring
0 % clip at the moderate 2.5 Hz/0.8-amplitude regime but 30–46 % at
sprint-like 3–4 Hz full-amplitude excitation while still capped at
0.8×kp). The original home-control-only sizing clipped 20–50 % of
gait-cycle torque per species and collapsed velociraptor stage-2 twice — see
[investigations/STAGE2_RECOMMENDATIONS.md](investigations/STAGE2_RECOMMENDATIONS.md)
§5. Post-fix clipping on the pinned gait-critical hip/knee/ankle actuators is
≤0.5 %, covered by each species' `tests/test_anatomical_actuator_bounds.py`; re-measure
any re-sizing with
`environments/shared/scripts/actuator_saturation_report.py`.

Neutral-action stability and truly actuator-disabled passive behavior are now
separate test contracts. Layered policy, physics, visual, and source identities
are documented in [PLANT_CONTRACT.md](PLANT_CONTRACT.md).

### Brachiosaurus mesh fingerprints on macOS ARM64

The anatomical Brachiosaurus uses the Linux x86-64 reference for visual revision 3.
The symmetric mesh revision compiled from identical MJCF bytes on both platforms.
With MuJoCo 3.10.0 the physics and policy digests matched. Native macOS ARM64 produced
a different visual digest. A mesh-normal component differed by at most `1.79e-7`, and a
mesh-centroid coordinate differed by at most `5.55e-17` model metres. The full plant verifier compares the visual digest and can reject
a native macOS run as stale. Use Linux x86-64 for verified training and reference-file
regeneration. The Gymnasium environment and reference walk also run on macOS ARM64.
See [the plant contract](PLANT_CONTRACT.md).

### Foot touch sensors under-report on two species — fixed (July 2026 sensor audit); a substep under-read is open

A MuJoCo touch sensor sums only contacts on geoms belonging to its site's **own body**, so a
site on a parent segment silently misses whatever the child geoms carry. Auditing all four
species against `mj_contactForce` found T-Rex and dibothrosuchus correct (ratio 1.000) and two
species wrong:

| species | sensor / measured contact | missing |
|---|---|---|
| velociraptor | ~~0.553~~ **FIXED** — now 1.000 | was `metatarsus` 17.54 N + `toe_d4` 12.03 N per foot, the site being on `toe_d3` only; repaired per plant_versions note 14 (metatarsus and digit-IV touch sites and sensors appended, sensordata 27–30, and summed per foot; physics r3, policy interface r11, D-D25, 2026-10-06) |
| brachiosaurus | ~~0.000~~ **FIXED** — now 1.000 | was everything; repaired per plant_versions note 8 (pad sites enlarged, meta sensors appended, pad + meta summed on both backends) |

Total floor reaction equals body weight on all four species, so the contacts were real and the
plants in equilibrium; these were sensor-scope defects, not physics ones. `aa3395c` fixed the
raptor site's *size* but not its *body scope*; note 14 is the `aa87445`-shaped repair, and since
the frozen MJX registration cannot sum it (D-D17), the velociraptor declared itself SB3-only in
the same policy-interface revision ([CLEANUP_PLAN_2026_09.md](CLEANUP_PLAN_2026_09.md) §4.1).
The raptor's observation and its `r_foot_contact` / `l_foot_contact` info keys now carry the
whole foot's load, which its stage-1 support terms read. Evidence, method and reproduction in
[investigations/FOOT_SENSOR_VERIFICATION.md](investigations/FOOT_SENSOR_VERIFICATION.md);
re-check any repair with `environments/shared/scripts/foot_sensor_report.py`.
*Update (2026-11-09; the maintainer's decision of 2026-11-08):* velociraptor physics r5
(`configs/plant_versions.toml` note 22) rebuilds the foot with one touch site and sensor on each support
body (the MTP pad, digit III and digit IV), resolved by name in `_cache_ids`, which fails closed on a missing
one. Each sensor equals the floor's normal force on its geom at every physics substep (the six sum to the floor
force to 5.8e-16), and since policy interface r14 the observation carries each as its own channel; the info keys
and rewards keep the per-foot sum.

**Update (2026-09-28, gait audit):** a second under-read, shared by every species, comes from how
substeps are combined rather than from sensor scope. The contact-shaped rewards and foot-contact
info keys read `_aggregated_foot_contact_forces()`, the per-foot **minimum** over a control step's
physics substeps (`base_env.py:979-992`), so a foot touching on only some substeps reads as
airborne. Against the floor's normal force on the leg (above 0.1 N on at least half the substeps),
the velociraptor run `20260922_125248` reads 49% flight by touch against 33% (its stance 25% against
15%), and the in-training dibothrosuchus re-run `20260928_012318` reads 0.39 body weights against
1.00 with all of its floor load inside the touch-site volumes, its 10–30 ms contacts lost to the
minimum ([gait audit](investigations/GAIT_AUDIT_2026_09.md) §2.2). The gait plan measures gait on
floor contact instead ([gait plan](GAIT_QUALITY_PLAN_2026_09.md) §3.2; GQ-6, open for gait and
rewards; taken for the stance gate, as D-D23, on 2026-10-06, whose `stance_quality/v2` reads floor
truth and never touch). `BaseDinoEnv.step` now also keeps every substep's per-foot force beside the
minimum (D-D27, 2026-10-07): the T-Rex stance's floor-impact and airborne-substep terms read its peak
and its airborne substeps, and the anatomical compsognathus stance's new support terms read its
mean (D-D26); its legacy alive, posture and height terms keep the minimum on purpose, since that is what
prices a two-foot hop (a one-foot unload leaves the other foot's minimum above the support bar, so
only the new bilateral and coverage terms price a one-foot chatter), and every other species, and
every legacy term, still reads the minimum alone. The opposite error, touch over-reporting through the Compsognathus robot's
stacked soles, is under Training / RL.

### Velociraptor plant — closed by physics r5 (July 2026 raptor review)

The review's spring, keyframe, touch-sensor, `natural_pitch` and stage-1
gate findings are closed by velociraptor physics r3 (D-D25, 2026-10-06;
`configs/plant_versions.toml` note 14 and the outcome appended to
[reviews/VELOCIRAPTOR_PLANT_REVIEW.md](reviews/VELOCIRAPTOR_PLANT_REVIEW.md));
its other findings are closed by velociraptor physics r5, the life-size
plant (the maintainer's decision of 2026-11-08; note 22 and the review's
dated status note), as each item below says. Measured on the r5 MJCF by the
review's protocol (home keyframe, `key_ctrl` held, 3000 MuJoCo steps).

- ~~**HIGH (fidelity) — the metatarsus is 78% too long**~~ **FIXED** (physics
  r5): femur 238.0, tibia 255.0 and MT III 99.1 mm joint to joint, IGM
  100/986's, so MT III / femur is 0.416 (r4 0.741) and tibia / femur 1.071;
  the metatarsus shaft sits 14.4 mm clear of the floor and carries nothing
  (r4 28.5% of each foot), and each foot stands on an MTP pad under the
  metatarsophalangeal joints and digits III and IV (0.299 / 0.365 / 0.336
  of its load). As reviewed: the metatarsus was 78% too long relative to the femur:
  model MT III/femur 0.741 against 0.416 (Persons & Currie 2016, *Sci Rep*
  6:19828, Table 1, IGM 100/986) and ~0.51 from a second specimen (Norell &
  Makovicky 1999, *AMNH Novitates* 3282). It also bears 26.2% of each foot's
  load (28% at the physics-r3 stance, which keeps its length) and forms the
  *rear* edge of the support polygon, so the "digitigrade"
  foot is functionally part-plantigrade. tibia:femur is within 3.6% and correct
  — leave it alone.
- ~~**MEDIUM — the two claw motors are the only unbounded actuators**~~
  **FIXED** (physics r5): each claw is a position servo (kp 5) bounded at
  ±4 N·m, 65 N at the farthest point of its geom from the claw joint, 0.44
  body weights (the r4 motor's ±50 N·m gave 609 N there, 4.6 body weights);
  no actuator of any committed plant is unbounded. As reviewed: the only
  unbounded actuators in any plant (`forcelimited=False`, `gear=50`): 693 N at the claw tip, 5.2× body
  weight, on the geom that scores stage 3. The July 2026 `forcerange` sweep
  missed them.
- **Ankle range and columnar stance (findings 8 and 9): FIXED** (physics r5).
  The stance is Bishop et al. 2018's paravian posture (the MJCF's leg
  comment): knee 125.0° and ankle 151.5° interior (r4 167.9° and 138.3°),
  the femur 35.1° from vertical (r4 13.7°), 7.8° of knee per centimetre of
  hip height by a finite difference on the settled pose (r4 11.3° by the
  same), and the hip pitch, knee and ankle each settle at 50.0% of their
  ranges (r4 ankle 30.6%).
- **Head, neck and forelimbs (findings 10 and 11): FIXED** (physics r5):
  `neck_pitch`, `neck_yaw` and `head_pitch` on separate neck and skull bodies
  (the T. rex's names), servoed with a gravity preload in the home control;
  the arms are welded and non-colliding, no joints or actuators, as the T.
  rex's are (r4: two shoulder servos each), humerus 160, forearm 120 and hand
  175 mm, 0.30 kg each, 77% of femur + tibia + MT III (r4 25%).
- **Not recommended:** porting the T-Rex stance correction here. That argument
  rests on a live stage-1 height term forcing knee travel through a
  near-singular joint, and the raptor env has **no height reward at all** —
  `height` appears in `velociraptor_mongoliensis/base_env.py` only in the `pelvis_height` diagnostic
  and the shared height/tilt termination, never in a reward term. *Update
  (2026-11-09):* still so on r5; the r5 knee flexion comes from the
  paravian posture evidence above, not from the T-Rex correction, and the
  env still has no height term.
- **LOW — the r5 plant's residuals** (the MJCF's comments; no reward or
  gate reads them): in Macaulay et al. 2023's standardised neutral posture
  the centre of mass sits 0.056 m below the hip against their 0.065–0.078 m
  scaled to 15 kg; the digit III and IV contact capsules are light (579 and
  464 kg/m³, geom mass over capsule volume, inside the 0.05 kg digit
  budget); and the welded, non-colliding hand passes visually through a
  straightened shank once the hip flexes 19.5° past the stance.
- **LOW — the review's R7 is not done** (its 2026-11-09 status note): the
  velociraptor's nosedive margin is still the shared 0.5 literal in
  `_is_terminated`, on the one termination path left since the MJX runtime
  retired (D-D17), not a constructor argument. The env keeps it a literal on
  purpose (`velociraptor_mongoliensis/base_env.py`: a constructor knob would
  add its default to the effective config that `task_sha256` hashes); with
  `natural_pitch` 0.002 rad it ends an episode 30.1° nose-down, where the
  T-Rex stance ends at 31.3°. Whether to promote it is open.

> **Note:** these changes alter the physics plant. Policies trained before the
> change are incompatible by contract, including when a change seems marginal;
> use an explicit legacy override only for deliberate historical evaluation.

Still open:

- **MEDIUM** — **the T-Rex's passive fore-aft balance is marginal on
  physics r7, r8 and r9: a 1.5° ankle offset topples the zero-action statue
  in most episodes (on r9 the −1.5° one in only 16/40), and reset noise alone
  topples it on some seeds (measured 2026-10-06; r9 2026-10-11).** Constant
  ankle commands of ±0.03 (±1.5° on both ankles, zero action otherwise; the
  stance env, reset noise 0.05, seeds 3042–3081): +0.03
  reaches the horizon on 14/40 episodes on r8 and 13/40 on r7, the rest
  ending `tail_contact`; −0.03 on 18/40 on both, the rest `nosedive`. The
  plain statue nosedives on seed 3174 on both plants (step 265 on r8, 228 on
  r7). Quasi-static push capacity (`configs/plant_versions.toml` note 13's
  protocol) is 3.2% of body weight forward and backward on both plants
  against 5.1% laterally on r8, and the pushed recovery statue's 40 ends on
  r8 are all fore-aft (nosedive 27, `tail_contact` 13). The limit is the leg
  chain's pitch compliance, not the hips, digits or tail: the centre of
  pressure saturates at a pad edge while the centre of mass creeps, and
  stiffer passive digits (300 / 600 N·m/rad) raise forward capacity only
  11% / 17% and backward not at all (note 13). Physics r8 changed the hip
  roll only, so every r8 stance and recovery policy has to hold fore-aft
  balance actively. A fix re-derives the ankle gains and their 5.5° gravity
  preload (`tyrannosaurus_rex_anatomical.xml`'s actuator comment), a physics revision that retires
  every T-Rex checkpoint again.
  *Update (2026-10-11):* physics r9 (the elliptic friction cone, D-D29,
  `configs/plant_versions.toml` note 16) leaves it as marginal, and the
  statue now falls on its certification block. The ankle probe: +0.03
  reaches the horizon on 14/40 (r8 14/40), −0.03 on 24/40 (r8 18/40). The
  plain statue nosedives on 4 of 280 stance-panel seeds (3081 at step 292,
  3174, 5048, 9062) against r8's 2 (3174, 9062), Fisher exact p = 0.69, so
  not a significant difference; but 3081 is on 3042–3081, where the r9
  statue is clean on 39/40 and uses the whole `max_hop_or_fall_episodes = 1`
  allowance, and a statue-quality policy falling at 4 in 280 passes that
  rail on about 0.89 of 40-episode panels (0.97 at r8's 2 in 280). The pads
  also sit less flat: the worst sole tilt is 2.061° on 1 of 240
  out-of-sample episodes (seed 3121) against the 2.0° bar, and the median
  loaded sole contacts drop from about 3.8 to about 2.6 (least 1.956
  against the 1.5 bar). The pushed recovery statue's 40 ends stay fore-aft
  (nosedive 26, `tail_contact` 14). Every stance bar was kept (the stance
  config's comments carry the r9 extremes); the first r9 stances are where
  the rail and the sole-contact margin need watching.
  *Update (2026-10-25):* physics r10 (the preloaded tail pitch springs, D-D44,
  `configs/plant_versions.toml` note 20) moves the statue's centre of mass 10.7
  mm further back (the probe) and its settled pitch from about 1.55° to about
  0.95° nose-down, and turns the failure direction: on the 280 stance-panel
  seeds the r10 statue falls once, seed 9047 backward (`fallen` at step 273,
  about 38° nose-up) where r9's four falls were nosedives (Fisher exact p 0.37,
  measured; not significant, inferred), the pushed recovery statue's 40 ends
  become nosedive 20, fallen 19 and `tail_contact` 1 (r9 26, 0 and 14), and
  under open-loop command noise of SD 0.10 its falls are mostly backward (16/40
  reach the horizon against r9's 21/40, within chance). Seed 3121's pad tilts
  2.087° against the 2.0° bar (r9 2.061°). The ±0.03 ankle probe above was not
  re-run on r10; re-running it is the first check of the backward margin, and
  the first r10 stances are where the backward falls, the hop-or-fall rail and
  the pad tilt need watching. Backward-fall risk under perturbation is untested
  at useful power (the CPU checks' skeptic: 0 falls in 40 episodes is compatible
  with a 2% rate). *Update (2026-10-27):* the first r10 stances, the round of
  D-D49 to D-D51, are training; results pending (§26 of the stance record).
  *Update (2026-10-28):* in that round's deterministic panels (40 episodes at
  each judged checkpoint) no episode fell, and the finals of seeds 60 and 80
  passed the six-heading certificate, its 2.0° sole-tilt bar included (a
  measurement of the gate's criteria on a screen task, not certification);
  seed 80's joint-space noise falls (5 of 20 at 6,291,456 steps, 2 at the
  final) are inferred backward from the centre of mass. The 280-seed panel,
  the out-of-sample certificate and a check of the head tip by hand were not
  run on the finals; they are among the proposals' report-only checks. The
  ±0.03 ankle probe of the r10 statue is still not re-run, and no proposal
  includes it (§27 of the stance record).
- ~~**LOW** — the raptor's toe-clipping margin now sits at the toes~~
  **FIXED** (physics r5, 2026-11-09): the toes are passive, with no
  actuators, and the velociraptor actuator-bound tests' clip protocols (home
  1000 steps, gait 2000, sprint 3.5 Hz at full amplitude) clip 0.0% on all
  17 actuators. As it was: at
  sprint-like excitation (3–4 Hz, full amplitude) the 0.8×kp toe caps clip
  10–16 % and the 1.5×kp hip pitch ~11–16 % (its physical envelope); the
  knee measures 0 % after its 1.5× bump. Re-measure with
  `environments/shared/scripts/actuator_saturation_report.py` before any
  faster-gait (stage-3 sprint) work.
- **LOW** — scene boilerplate (skybox/grid/floor/option) is copy-pasted
  across the three XMLs → extract a shared `scene.xml` include; limbs are
  hand-mirrored sign-flips → generate via script/PyMJCF or add a left/right
  symmetry test.
- **LOW** — ~~raptor claw `motor gear="50"` on a 0.05 kg claw (huge
  torque-to-inertia; slams its limits)~~ (physics r5: a kp-5 position servo
  bounded at ±4 N·m on a 0.011 kg claw); contype-0 neck geoms can visually
  clip the floor (the velociraptor's too since r5); no `<light>`/`<visual>` block for nicer renders. (The
  brachiosaurus food that had no collision partner left the scene with physics r7, decision D-D59.)
- ~~HIGH — the brachiosaurus cannot hold its home stance.~~ **FIXED**
  (plant_versions notes 7–8): the collapse decomposed into the midpoint action
  mapping never commanding the home pose (knees dragged up to 0.35 rad off)
  and leg servos that sagged 71.6 mm under static weight, leaving the planted
  stance's roll stiffness at parity with `m·g·h`. Brachiosaurus now uses the
  home-keyframe-residual mapping like the other species, leg kp is doubled
  (sag 11.1 mm), and the zero-action baseline is 40/40 full-horizon at
  1739.08 ± 1.17 (was 0/40 at 163.35 ± 81.40). The full-horizon neutral test
  this entry asked for now exists on the brachiosaurus
  `TestNeutralActionStability` subclass.
- **Experiment** — with `implicitfast`, a `timestep` 0.002→0.004 A/B is
  worth running (halves sim cost if stable).

### Compsognathus plant — open (2026-10 standing review)

The standing review of the anatomical compsognathus (2026-10-07) found its
tail denser than bone at the tip, its metatarsus 1.07 mm off the floor, its
right foot built as a left foot and its leg action map putting the whole
holding torque inside 1% of the action, and its implementation found the MTP
servo ringing in a numerical heel chatter; anatomical physics r2 and policy
interface r3 fix those (D-D26, 2026-10-07; `configs/plant_versions.toml`
note 15 and the stance-hack audit's §11). What it leaves open:

- **MEDIUM** — **the single-support tiptoe stays physically possible.** The
  MTP servo's 0.6 N·m cap holds one body weight on the digit-III tip, 61 mm
  ahead of the joint (0.594 N·m), and the rigid foot puts all its load on
  that tip at 0.05° of toe-down pitch; both certified r1 stances held the tip
  with the MTP servo at its force cap on 64–65% of loaded steps. The
  `20260921_203149` march, replayed on the r2 plant through its own linear
  interface, survives 20 of 20 episodes on that tip (seeds 3042–3061). The
  stance reward (the marches re-score at 0.61 of the statue) and
  `stance_quality/v2` (support-geom duty and coverage, sole tilt, corner
  lift, touchdowns) are what refuse it; `max_actuator_saturation_fraction` reads commands and is
  blind to the MTP servo's force saturation. In the review a 0.4 N·m cap
  flattened the loaded foot (at 0.3 N·m the march fell in 42–49 steps while
  the statue was unchanged in every test), but it also limits the digit
  plate's push-off. The maintainer deferred the cut (D-D26); it is one
  number, `toe_forcerange_nm` in `data/model_parameters.json`, and a
  physics revision. Plan: measure the MTP torque budget of the first r2
  walker, then decide the cap.
- **LOW (fidelity)** — **the proportions fidelity pass is deferred
  (D-D26).** The model's tail reaches 5.5 femur lengths behind the hip,
  against at least 6.4 for the Canjuers specimen (Bidar, Demay and Thomel
  1972), and an anatomically long tail at r2's uniform density and 178 g
  gives back about 43% of r2's centre-of-mass gain (0.38 → 0.32 of the
  foot), so the pass has to re-tune the mass layout with it. r2's tail
  density, 854–871 kg/m³ per segment, is below soft tissue only because the
  fixed 178 g fills oversized meshes; the distribution, not the absolute
  value, is the correction.

## Configs, docs & website

- Brachiosaurus stage-2 `natural_pitch = -0.15` while stages 1/3 use 0.0 —
  intentional? (June §4)
- `docs/investigations/REWARD_SCALE_REDESIGN.md` uses `*_bonus_weight` key
  names that don't exist. (June §5)
- **LOW** — `website/static/img/logo.svg` is 38,406 bytes because it is not a
  vector drawing: it is a 1000×1000 SVG wrapping one base64 PNG (500×500,
  28,552 bytes, with a 418-byte XMP text chunk) and has no paths and no
  SVG editor metadata, so SVGO or path simplification, the remedy
  [WEBSITE_PLAN.md](WEBSITE_PLAN.md) proposed, would not shrink it; the
  navbar's dark-mode `logo-dark.svg` (25,294 bytes) is built the same way
  (`website/docusaurus.config.ts:108-112`), its PNG (18,720 bytes) without
  the text chunk. Fix: recompress both PNGs, dropping `logo.svg`'s XMP chunk,
  or redraw the logo as a vector. (measured 2026-09-30, each file 1 byte
  longer since its final newline was added on 2026-10-12; moved from
  WEBSITE_PLAN.md's "Optimize Logo SVG")
- **LOW** — **the T. rex stance's four probe reports have no reader
  documentation, and their TOML derivations are stale (read from the code
  2026-11-26).** `configs/tyrannosaurus_rex/anatomical/stance.toml`, the only
  stage that declares them, turns on all four report-only probes the stage
  artifacts roll after the gate report (`stance_probe_filter_hz`, :1172;
  `stance_probe_hold_constant`, :1215; `stance_probe_release_ablation`, :1244;
  `stance_probe_impulse_speeds`, :1262): about 350 episodes by the TOML's own
  counts, in every session, `QUICK_TEST` included (the quick-test entry under
  Notebooks). Their `stance_gate_probe_*` reports are named in no artifact doc
  (`docs/TRAINING_ARTIFACT_PRESENTATION.md` and `website/docs/` do not mention
  them; only `docs/investigations/stance_2026_10/` does) and get no notebook
  readout. Their comments describe older plants: the release ablation counts
  "17 on the current r7 plant" (:1258-1260), and the hold and impulse probes
  quote the August checkpoint `20260805_011234` (:1236-1238, :1278-1281), on a
  plant now at physics r12 (`configs/plant_versions.toml:944-946`). The
  impulse grid `[0.5, 1.0, 2.0]` m/s is the one the October battery found
  "starts above every role's envelope", so every sagittal row falls
  (`docs/investigations/stance_2026_10/battery/battery_report.md:83`; its item
  2, :140, proposes forward and backward rows and a 0.05-0.5 m/s grid). Fix:
  document the four reports beside the other stance artifacts, re-derive or
  drop the stale counts, adopt the battery's grid or drop the impulse probe,
  and skip the battery under a smoke mode. (2026-11 notebook review:
  tyrannosaurus-trex-probe-battery-unscaled-undocumented)
- **LOW** — **the `[sac]` comments promise a replay buffer that survives the
  warm start, misstate the update-to-data ratio and "match" `[ppo]` values
  that have moved (read from the code 2026-11-26).** Three claims are wrong:
  - *The buffer.* The velociraptor, T. rex, brachiosaurus and dibothrosuchus
    walks size `buffer_size = 1_000_000` because it "retains Stage 1 balance
    transitions"
    (`configs/velociraptor_mongoliensis/anatomical/locomotion.toml:135`,
    `configs/tyrannosaurus_rex/anatomical/locomotion.toml:165`,
    `configs/brachiosaurus_altithorax/anatomical/locomotion.toml:132`,
    `configs/dibothrosuchus_elaphros/anatomical/locomotion.toml:137`), and
    [RL_TRAINING_PLAN.md](RL_TRAINING_PLAN.md):105-108 counts it against
    forgetting between stages. Nothing in `environments/shared/` saves or
    loads a replay buffer, and a warm start restarts the step counter
    (`reset_num_timesteps=not resuming`, `train_base.py:2027`), so a SAC walk
    starts with an empty buffer and `learning_starts` (10,000) uniformly
    random steps.
  - *The ratio.* SB3 counts `train_freq` in vector-env steps, each `n_envs`
    transitions. `train_freq = 16` with `gradient_steps = 8`, commented "2:1
    data-to-update ratio" (the velociraptor's `stance.toml:276-277` and
    `locomotion.toml:137-138`, the T. rex's `stance.toml:575-576` and
    `locomotion.toml:168-169`, the dibothrosuchus' `stance.toml:75-76`), is 8
    transitions per update on the notebook's 4 envs and 16 on the CLI's 8; the
    brachiosaurus' 4 and 4, "Match train_freq to maintain update-to-data
    ratio" (`stance.toml:59-60`, `locomotion.toml:135-136`), and the
    dibothrosuchus walk's
    (`configs/dibothrosuchus_elaphros/anatomical/locomotion.toml:139-140`) are
    4; both Compsognathus models' uncommented 4 and 1 is 16, and the
    Allosaurus' SB3 default 1 and 1 is 4. The spread is fourfold and nothing
    derives it.
  - *"Match PPO".* `learning_rate = 3e-4` is commented "Match PPO starting LR"
    on the velociraptor, brachiosaurus and dibothrosuchus stances
    (`configs/velociraptor_mongoliensis/anatomical/stance.toml:269`,
    `configs/brachiosaurus_altithorax/anatomical/stance.toml:50`,
    `configs/dibothrosuchus_elaphros/anatomical/stance.toml:68`) and `gamma =
    0.99` "Match PPO" on the first two (:271, :53), beside `[ppo]` tables at
    3e-5 and 0.98; the velociraptor's `batch_size = 256` "matches PPO" (:270)
    and the brachiosaurus' is "Larger than PPO's 64" (:51), where both `[ppo]`
    tables say 128. The T. rex's `train_freq` and `gradient_steps` cite the
    "proven raptor SAC config" (`stance.toml:575-576`,
    `locomotion.toml:168-169`), a historical run on a plant several revisions
    back. Only the T. rex stance's `[sac]` carries a staleness warning
    (`stance.toml:558-563`).

  Fix: correct the comments (no buffer crosses a warm start; ratios as
  transitions per gradient update at the run's `n_envs`), copy the T. rex
  staleness warning onto every `[sac]` block until a SAC run re-derives it,
  and correct RL_TRAINING_PLAN Trial 3's buffer rationale. (2026-11 notebook
  review: fu-sac-artifact-path-sac-warm-start-semantics-misdescribed,
  fu-sac-artifact-path-sac-update-to-data-ratio-misstated,
  fu-sac-artifact-path-sac-recipe-provenance-stale, and the `[sac]` half of
  brachiosaurus-stale-recipe-comments)
- **LOW** — **the artifact guide has no table of which artifacts a node gets,
  leaves out the Allosaurus, and misdescribes the velociraptor's head and the
  dibothrosuchus' feet (read from the code 2026-11-26).** What a stage
  directory holds depends on its gate kind, role and model: the stance report
  and the heading probe or six-heading certificate come only with a stance
  gate kind (`STANCE_GATE_KINDS`, `reporting/stage_artifacts.py:238-241`), so
  the two quadrupeds' reward-rail stances get neither;
  `handoff_in_training.json` is written only beside a stance gate kind's
  report and the walk's command report (:241, :295, :336, :677), so those two
  stances and every recovery get none; `stance_diagnostics.png` (:2291-2296)
  and the filtered, constant-hold, release and impulse probe reports
  (`stance_gate_probe_*`) come only on the T. rex stance, the one model the
  figure draws (`visualization.py:33-35`) and the one stance TOML that enables
  those probes (`configs/tyrannosaurus_rex/anatomical/stance.toml:1172-1262`);
  the per-frame `_stance.csv` and the extra camera views come only on stance
  and recovery (:2451-2452); the walk gets the command report, its panel CSV,
  `command_tracking_trace.csv`, `command_tracking.png` and
  `command_gate_progress.png`; and the policy probe needs a rollout buffer and
  a statue that stands past step 199, so it is empty under SAC and on every
  Allosaurus node.
  [TRAINING_ARTIFACT_PRESENTATION.md](TRAINING_ARTIFACT_PRESENTATION.md)
  documents each file but has no table of node against file, and:
  - it names the Allosaurus once, in the rollout-balance paragraph (:938-946),
    which still says its walk ends on `course_complete`: under the committed
    live command it does not
    (`environments/allosaurus_fragilis/base_env.py:191-192,219-235`), and the
    `DiagnosticsCallback` docstring repeats it
    (`environments/shared/diagnostics.py:97-100`). Its model lists stop at
    six: the sagittal-pair keys "exist on all six" (:314-315), though the
    Allosaurus' torque motors are not the "unit-gear hinge position servos"
    that sentence requires (:312-313); the recorder table and the paragraph
    after it (:754-770) cover the other six models; and the probe paragraph
    says a scalar reads NaN when none of its rows remain (:993-999), not that
    the whole file stays empty, as it does on the Allosaurus (the Allosaurus
    entry under Infrastructure);
  - the paragraph after the recorder table (:761-765) says the velociraptor
    has "no head-tip site, no neck hinge"; since physics r5 its MJCF has a
    `head_tip` site
    (`environments/velociraptor_mongoliensis/anatomical/assets/velociraptor_mongoliensis_anatomical.xml:181`)
    and `neck_pitch` and `neck_yaw` hinges (:164-165), so its posture columns
    are missing only because it has no box pad and so runs no recorder, the
    gap D-D46 kept, and a recorder would now fill them;
  - the pad paragraphs say no dibothrosuchus foot declares digits (:125-128)
    and that the digit share exists only on the T. rex and the anatomical
    Compsognathus (:267-268), but the dibothrosuchus' `FOOT_SUPPORT` lists
    four digit pads per foot beside the sole
    (`environments/dibothrosuchus_elaphros/anatomical/env.py:24-28`), so every
    foot records `<label>_digit_share`;
  - the stage-end report is "the nineteen pages" (:1678), and
    `training_history.page_names()` lists 20.

  Fix: a table of expected artifacts by gate kind, model and role, with the
  Allosaurus where its set differs (the per-node `artifact_status.json` of the
  artifacts entry under Training / RL is the same list checked in code);
  correct the passages above and the docstring; and put the velociraptor's
  recorder gap back to the maintainer now that its stated reason has gone.
  (2026-11 notebook review: artifact-coverage-matrix-uneven,
  allosaurus-missing-from-artifact-docs,
  velociraptor-recorder-gap-premise-stale,
  dibothrosuchus-artifact-doc-digit-share-stale)
- **LOW** — **`results/README.md` and the Drive summary's strict-promotion
  cell describe publication differently from the code (read from the code
  2026-11-26).** The README says the exporter writes one `summary.json` "after
  the run's last stage" (:68-69); it writes one whenever at least one
  deliverable is certified and rewrites it after every later node
  (`docs/RESULT_BUNDLES.md:38`), so a chain stopped above its target publishes
  a `partial` summary. It also says "summary schema version 2 made this
  backend and provenance metadata mandatory" (:32-35), while readers accept
  schema 5 alone since D-D63 (:51-52). The strict-promotion cell
  (`notebooks/google_drive_summary.ipynb` cell 22, :1429) calls
  `validate_result_bundle(..., require_complete=True)`, which accepts only
  `canonical-valid`, while `docs/RESULT_BUNDLES.md:12` names
  `require_publishable=True` as promotion validation and the species catalog
  publishes a `partial` summary (`species_catalog.py:728-741`): a run that
  certified its stance and failed its walk passes the catalog and is refused
  by the cell. Fix: correct both README sentences, and have the cell validate
  with `require_publishable=True` (or offer both and say which promotion
  needs). (2026-11 notebook review: results-readme-and-strict-cell-drift)
- **LOW** — **each model page embeds its media by hand, and no two do it alike
  (read from the code 2026-11-26).** The manifest declares stage videos for
  the velociraptor, T. rex and the brachiosaurus only
  (`configs/species_manifest.toml:112-235`). Their pages and the
  dibothrosuchus' render `SpeciesVideoSection`, headed "Historical training
  videos" on three and "Training Videos" on the dibothrosuchus, which declares
  none (`website/docs/models/dibothrosuchus_elaphros/anatomical.mdx:43-45`).
  The Allosaurus page embeds its reference walk by hand and has no video
  section (`website/docs/models/allosaurus_fragilis/anatomical.mdx:21`); the
  two Compsognathus pages
  (`website/docs/models/compsognathus_longipes/anatomical.mdx` and
  `robot.mdx`) show no media at all; the only GIF is the velociraptor's
  historical `results/velociraptor_mongoliensis/anatomical/ppo/stance.gif`.
  With no published summary, the CSV and section differences the review also
  listed are gone. Fix: generate each page's media section from the catalog
  (stage videos and reference walks) under one heading, and say on a page with
  none that none is published. (2026-11 notebook review:
  presentation-inconsistent-across-models)
- **LOW** — **the T. rex recovery config states its recovery window at 50 Hz
  (read from the code 2026-11-26).** In
  `configs/tyrannosaurus_rex/anatomical/recovery.toml`,
  `recovery_t_recover_steps = 100` is "2.0 s at the 50 Hz control rate" (:200)
  and `recovery_dwell_steps = 50` is "1.0 s" (:204). The T. rex steps at 100
  Hz (`timestep="0.002"` in its MJCF, `frame_skip` 5 at
  `environments/tyrannosaurus_rex/base_env.py:242`, which no stage overrides),
  so the window is 1.0 s and the dwell 0.5 s. Nothing reads the comment (the
  judge's own stale comment in `recovery_evaluation.py` is in the T. rex
  recovery entry under Training / RL). Fix: correct it. (2026-11 notebook
  review: tyrannosaurus-trex-recovery-timing-and-stale-comments)
- **LOW** — **the species catalog gives four stances the command line's
  consecutive-pass rule, which the notebook never applies (read from the code
  2026-11-26).** The "SB3 early-advancement gate" column of
  [SPECIES_CATALOG.md](SPECIES_CATALOG.md) ends the brachiosaurus' and the
  dibothrosuchus' `reward_and_length/v1` stance rows (:94, :123) and the
  Compsognathus robot's and the Allosaurus' `stance_quality/v1` rows (:183,
  :214) with "3 consecutive passes"
  (`environments/shared/species_catalog.py:1559`), and the two reward-gated
  definitions in `configs/species_manifest.toml` (:210, :262) read "over the
  required consecutive evaluations". That is the in-training manager's rule,
  which only the command-line `curriculum` applies. The SB3 notebook trains
  with `train_base.train`, which stops no node on a gate pass, and judges the
  handoff once, on one post-stage panel (`generate_stage_artifacts`); item 2
  under Post-training artifacts has the verdict that follows. The six-heading,
  recovery and walk rows already say "post-stage". Fix: render the criteria by
  gate kind and entry point (one post-stage panel in the notebook, consecutive
  passes on the command line) in `species_catalog.py` and the two definitions,
  and regenerate. (2026-11 notebook review:
  cmp-catalog-gate-semantics-by-kind)
- **LOW** — **the brachiosaurus walk's fall-floor comment gives a stance
  height the plant no longer settles at (measured 2026-11-26).** It explains
  `healthy_z_range = [0.85, 3.5]`
  (`configs/brachiosaurus_altithorax/anatomical/locomotion.toml:46`) by a
  "settled stance ~1.13 m" whose gait bob dips to ~1.10 m and by
  642-of-1000-step episodes under a 1.0 floor. On physics r7 the zero-action
  stance settles at 1.218-1.219 m of torso height on the stance and the walk
  env alike (seeds 3042-3044, steps 200-1000, every episode to the horizon),
  so the floor sits 0.37 m under it, not the 0.28 m the comment implies, and
  the episodes it cites come from an older plant. The stance's `[sac]` "Match
  PPO" lines are in the `[sac]` comments entry, and the statue values the
  review also found stale are re-measured on physics r7
  (`configs/brachiosaurus_altithorax/anatomical/stance.toml:69-87`). Fix:
  re-measure the gait's height dip on a current walker and restate the comment
  (or the floor). (2026-11 notebook review: the height half of
  brachiosaurus-stale-recipe-comments)
- **LOW** — **the Compsognathus recipe review, cited by every Compsognathus
  TOML as the next-run protocol, describes superseded gates and observations
  (read from the docs 2026-11-26).** The stance and walk TOMLs of both models
  open with "See environments/compsognathus_longipes/TRAINING_RECIPE_REVIEW.md
  for evidence, validation limits, and the next-run protocol" (line 2 of
  each), and the species README sends readers there for the lift-off count and
  the PPO settings (`environments/compsognathus_longipes/README.md:127`, :134,
  :234-235). The recipe review's only edit since the 2026-11 notebook review
  is the display-name change of `a7ecc79`, and:
  - its protocol's step 4 (:143-149) asks for "three consecutive qualifying
    screening evaluations", the command line's rule; the notebook judges a
    stance once, post-stage;
  - its amendment of 2026-10-07 (:253-295) has the anatomical stance certify
    under `stance_quality/v2` and names physics r2 and interface r3. The
    stance has been on the six-heading certificate since D-D32
    (`configs/compsognathus_longipes/anatomical/stance.toml:232`), on physics
    r4 and interface r7. The amendment calls the heading-free observation a
    coming revision and warns that a stance on "the current observation, which
    reads the world heading" would be stranded; the anatomical model has
    observed heading-free since policy interface r4 (D-D31), and the robot
    since its interface r4 (note 24); both read `bipedal-heading-free/v1` now
    (`configs/plant_versions.toml:1189`, :1195);
  - the protocol covers the stance alone, with nothing on the `recover` recipe
    (D-D35) or the command-driven walk (D-D62) that the TOMLs it heads now
    train.

  Fix: a dated amendment that points the protocol at the current gates and the
  walk, or point the TOMLs and the README at the current recipe pages instead.
  (2026-11 notebook review: compsognathus-recipe-review-protocol-stale)
- **LOW** — **ROADMAP still lists stranded nodes as certified (read from the
  docs 2026-11-26).** Its "Current focus (2026-10-05)" paragraph says the T.
  rex, the velociraptor and both Compsognathus models "have certified stance
  and walk nodes" ([ROADMAP.md](ROADMAP.md):28-31), and the Phase 3
  Compsognathus item says both embodiments do (:456-457). The dated update
  under the paragraph (2026-11-24, :54-58) says those nodes no longer load
  since `configs/plant_versions.toml` note 24 and that every stance and walk
  retrains; D-D24 (the T. rex), D-D25 (the velociraptor) and D-D26 (the
  anatomical Compsognathus) had stranded them already, and D-D33 every node
  (NEXT_STEPS §2's rows and its dated D-D33 note). Fix: restate both passages
  as history, what was certified on which plant, and say that no node is
  certified on the current plants. (2026-11 notebook review:
  compsognathus-robot-stranded-chain-still-listed-certified)

## Notebooks

- The SB3 notebook pins the plant compiler (`mujoco==3.10.0`) and
  `stable-baselines3[extra]==2.9.0`, but torch is unpinned and the
  `pyproject.toml` extras stay ranges; pin complete lockfiles for reproducible
  training. (July §5)
- **MEDIUM** — **`QUICK_TEST` shortens only the training budget, so it cannot
  smoke-test the SB3 notebook (read from the code 2026-11-26).** Cell 5 calls
  it "a pipeline check, 50,000 steps a node (it may stop at a gate)"
  (`notebooks/sb3_training.ipynb:195`), but besides the run's tree
  (`environments/shared/run_tree.py`) and the `Quick test:` line of
  `training_summary.txt` it changes only `node_budget` (cell 15,
  `sb3_training.ipynb:836-843`); no other library code reads it. Everything
  after training runs at full size: the evaluation cadence
  (`eval_freq=50_000`, :856), the stance certification panels (40 seeds; the
  six-heading certificate at each of six headings), the T. rex stance's probe
  battery (the probe-report entry under Configs, docs & website), and the
  replays with their extra camera views. Where a quick test stops depends on
  the species: the review's quick runs of 2026-11-22 (before
  `configs/plant_versions.toml` note 24 moved every plant) failed the
  Allosaurus and robot stances at 50k steps and passed the velociraptor and T.
  rex ones and the quadrupeds' (whose statue already clears the gate; the
  anatomical Compsognathus' very likely passes too, inferred), and a stance
  that passes now chains into the walk, where a 50k-step policy can be
  expected to fail `command_tracking/v1` on every model (the statue fails
  every committed walk panel and no controller has passed one; the walk entry
  under Training / RL). Under `BEHAVIOR = "walk"` (the default `"auto"` until
  decision D-D79 of 2026-11-28 removed it; the default is `"stance"` since, and
  the one-behavior guard lets a quick test train the whole chain) every quick test
  should therefore end at a gate failure, and `halt`
  (`environments/shared/notebook_runtime.py:51-57`, called at
  `sb3_training.ipynb:1689`) raises before section 8: cells 26-32 (the
  evaluation, the curves, the replays and the bundle check) never run, so the
  one end-to-end check open to a user never exercises them. In Colab with the
  defaults (`AUTO_DISCONNECT = True`, a run on Drive) the same halt releases
  the runtime (`disconnect_runtime`, `notebook_runtime.py:18-48`) and ends in
  an unqualified gate-failure `RuntimeError`, so a quick test costs its
  session; a run kept on the VM's disk is never released (`f3d3100`). Fix: a
  smoke mode (`QUICK_TEST = "smoke"`) that also shrinks the panels, skips the
  probe battery, shortens the replays and scales the cadence (the test helper
  `smoke_configs`,
  `environments/shared/tests/test_compsognathus_longipes_training.py:92`,
  shows how), continues past a gate failure with `pipeline OK through <node>`,
  and never disconnects. (2026-11 notebook review:
  quick-test-disconnects-colab, quick-test-cannot-smoke-the-notebook,
  quick-test-stops-at-first-gate, quick-test-does-not-exercise-report-path,
  compsognathus-quick-test-passes-stance-at-init)
- **MEDIUM** — **the SB3 notebook runs chains past Colab's ~24 h session cap
  with no time check, and resuming one is a manual second pass (read from the
  code 2026-11-26).** The default walk chain is 14M-27M steps (stance plus
  locomotion: the Compsognathus pair 11M + 3M, the T. rex and velociraptor
  11M + 8M, the brachiosaurus 11M + 16M), and the two 12M-step T. rex
  stances of 2026-11-07 took 23 h 07 min and 23 h 29 min on a Colab L4
  (`docs/NEXT_STEPS.md:411-412`). The chain loop (cell 20) starts each node
  with no time check. The run memo is the kernel global `_ACTIVE_RUN_ID` (cell
  8, `notebooks/sb3_training.ipynb:378-379,523`), so a fresh runtime mints a
  new run unless `RUN_ID` is set; the loop recognises the interrupted node but
  only raises with instructions (`:1593-1601`); and `RESUME_STAGE` takes a
  node, not `"auto"` (cell 18, `:1164`). Section 5's procedure (cell 17,
  `:1074-1105`) sets `RUN_ID`, the `TRUNK_FROM` that `trunk_run.json` records,
  `RETRAIN_FROM = ""` and `RESUME_STAGE`, then a second Run all; `76b27b2`
  made the RESUME cell refuse what would not continue the node and warn on
  `BEHAVIOR = "auto"` (removed with decision D-D79, and its warning with it)
  and another commit, but the pass stays manual, as
  decision D-D16 (c) chose. A session the cap cuts off before its final save
  records no duration: `train_base.train` writes `run.duration_seconds` only
  at the final save, summed over the sessions that reached one
  (`train_base.py:2418-2420`, :2453-2455, :2492; decision D-A15), so a capped
  segment drops out of `training_summary.txt`'s total. Fix: accept
  `RESUME_STAGE = "auto"` (adopt `trunk_run.json` and the interrupted node,
  keeping every refusal), persist `RUN_ID` to the run tree on Drive, record
  the duration at each periodic checkpoint, and add a session-budget knob that
  refuses to start a node projected past the session's end. The same-stage
  resume's lost reward ramp is the separate entry under Training / RL.
  (2026-11 notebook review: no-colab-cap-awareness-or-auto-resume,
  resume-is-manual-two-pass)
- **MEDIUM** — **only a failed gate and the last cell release the Colab
  runtime; any other error keeps it assigned (read from the code
  2026-11-26).** `AUTO_DISCONNECT` (`notebooks/sb3_training.ipynb:229`, lines
  of the `.ipynb` file) acts in two places: `halt` after a node fails its gate
  (`:1689`; `notebook_runtime.py:51-57`) and the auto-disconnect cell at the
  end (§9, `sb3_training.ipynb:2272`). Every other raise stops "Run all"
  before §9 and leaves the GPU runtime assigned until Colab reclaims it: the
  chain loop's refusals of a node that already holds a FAILED verdict, or a
  passed one the reuse rule refused (`:1549-1561`), and of an interrupted node
  (`:1594-1601`); any exception out of training, `generate_stage_artifacts`,
  the figure display or `save_run_bundle` (`:1638-1683`); the report cell's
  refusal of an uncertified target (§8, `:2129-2133`); and the verification
  cell's second `validate_result_bundle`, which nothing guards (`:2236`). The
  re-entry that stops at the verification cell (the Training / RL entry "a
  re-entry that only reuses a `partial` or `failed` run's nodes stops at the
  bundle-verification cell", whose remedy is to run §9 by hand) is one case of
  this. No cell installs an exception hook. A run kept on the VM's own disk
  keeps its runtime on purpose (`notebook_runtime.py:30-36`). Fix: in Colab,
  with `AUTO_DISCONNECT` and `RUNS_ON_DRIVE`, have the chain loop install an
  IPython custom exception handler (`set_custom_exc`) that writes the
  traceback to `RUN_DIR`, flushes Drive and releases the runtime, so that the
  report and verification cells are covered too. (2026-11 notebook review:
  disconnect-only-on-gate, exceptions-skip-auto-disconnect)
- **MEDIUM** — **no stage or model says whether its recipe is tuned, a pilot
  or held, so the SB3 notebook offers every chain alike (read from the code
  2026-11-26).** No `stages.toml` entry and no model in
  `configs/species_manifest.toml` has a status or hold field (the manifest's
  `status` keys describe project capabilities, :44-62, and
  `model_revision_status` its videos), and the chain table the resolve cell
  prints (cell 9, `sb3_training.ipynb:632-641`) shows each node's id, name,
  parent, recipe, deliverable flag and gate only. Until decision D-D79
  (2026-11-28) `BEHAVIOR = "auto"` resolved to the last deliverable, the walk on
  all seven models, so a plain Run all trained the stance and then the walk;
  the default is now the stance alone, and a walk session refuses to train a
  stance it cannot reuse. What the records say instead is
  prose the notebook never shows:
  - GQ-3 of [GAIT_QUALITY_PLAN_2026_09.md](GAIT_QUALITY_PLAN_2026_09.md)
    (:190, open) recommends three holds that bear on the notebook: the
    brachiosaurus' session 5 until its `gait-r1`, any robot recovery session,
    and any walker session on the rewards of the time; its hunt holds are
    moot since D-D59, its T. rex and robot follow hold concerns the
    `behaviors/` pilots, which no chain trains, and it was written before the
    command-driven walk;
  - every walk is judged on provisional bars, and no controller has passed a
    committed walk panel (the first Training / RL entry); the seven model
    pages say the bars are provisional (`website/docs/models/`), the notebook
    does not;
  - [NEXT_STEPS.md](NEXT_STEPS.md) asks, before any velociraptor walk, for a
    `BEHAVIOR="stance"` session on at least two seeds, each re-judged on the
    fresh seeds 7042-7081 (:136-142), and a scan of the stance parent's noise
    tolerance in the locomotion env (:152-159). The default chain does neither
    and warm-starts the walk from one seed's stance: `certification_seeds = 2`
    is a publication bar, not a gate threshold
    (`configs/velociraptor_mongoliensis/anatomical/stance.toml:633`);
  - the Allosaurus, added 2026-11-21 and the only torque-driven plant, trains
    on near-default tables, three `[ppo]` keys and two `[sac]` keys
    (`configs/allosaurus_fragilis/anatomical/stance.toml:11-18`, copied into
    the walk, `locomotion.toml:78-85`); neither its README nor its model page
    calls it untuned, and its policy probe never runs (the Allosaurus entry
    under Infrastructure).

  Fix: a per-node `status` in `stages.toml` (tuned, pilot or held, with a
  reason and a doc link), printed in the chain table, a warning or refusal
  before a held node unless an override is set, and the same mark on the
  catalog and the model pages; mark the Allosaurus a pilot; until a
  velociraptor stance certifies on two seeds, print the NEXT_STEPS protocol
  when `BEHAVIOR` resolves to its walk. (2026-11 notebook review:
  cmp-no-recipe-status-signal,
  brachiosaurus-chain-hold-not-surfaced-in-notebook,
  dibothrosuchus-recipe-on-hold-not-surfaced,
  velociraptor-pre-walk-protocol-not-surfaced, allosaurus-least-instrumented)
- **MEDIUM** — **the Drive summary hides a run's certified nodes while a later
  node trains, and after it is interrupted (read from the code 2026-11-26).**
  Once a run's first node certifies, `save_run_bundle` seals a `partial`
  bundle whose manifest lists the files then present. The next node's first
  file (its `stage_config.json`, a checkpoint, an event file) is undeclared,
  and `audit_result_bundle` rejects undeclared files by default
  (`reject_unlisted=True`, `result_bundle/audit.py:160`;
  `result_bundle/manifest.py:298-312`), so the run audits `canonical-conflict`
  (`audit.py:634-635`). The Drive summary's `scan_run`
  (`notebooks/google_drive_summary.ipynb` cell 9, :931-940) puts a conflict in
  `QUARANTINED_RUNS` and returns no row, so the certified node drops out of
  every table and of `runs_summary.csv` for as long as the next node trains or
  stays interrupted. Quarantine before the first node certifies is intended
  (CU-15), and it is the only case `test_drive_summary_notebook.py` pins
  (:306-310). Fix: audit a stage directory that holds no verdict, beside a
  `partial` manifest, as a node in progress rather than as undeclared files,
  and show the run's certified rows with an in-progress marker. (2026-11
  notebook review: drive-summary-quarantines-interrupted-runs)
- **LOW** — **the SB3 notebook offers SAC as PPO's equal, but trains it on
  half the command line's environments and still runs the documented PPO-only
  cases under it (read from the code 2026-09-30 and 2026-11-26).**
  `environments/shared/cli.py` raises `n_envs` from 4 to 8 for SAC unless
  `--n-envs` says otherwise (:526-529, applied at :550-552 for `train` and
  :610-612 for `curriculum`), while the configuration cell (cell 6) of
  [notebooks/sb3_training.ipynb](../notebooks/sb3_training.ipynb) sets `N_ENVS
  = 4` whatever `ALGORITHM` is (`sb3_training.ipynb:224`) and says nothing
  about SAC. The two entry points' default SAC runs differ twofold in
  collection against gradient updates; `n_envs` is recorded, so the difference
  can be found but is never flagged. `ALGORITHM` is free text (`"ppo"` or
  `"sac"`, :223), not a dropdown, and nothing marks SAC as untried: no SAC run
  is recorded on a current plant ([ROADMAP.md](ROADMAP.md):171 leaves "Run
  Stages 1-3 with SAC" open), and only the T. rex stance's `[sac]` says so
  (`configs/tyrannosaurus_rex/anatomical/stance.toml:558-563`). Of the cases
  the repository documents as PPO-only, only the resume is refused (cell 18,
  :1225-1231). Under SAC:
  - `CRITIC_PHASE = True` writes `[ppo] critic_phase = true` into every chain
    node and prints "Critic phase on" (cell 9, :606-609; cell 6 prints it
    too, :260), but SAC reads only its `[sac]` table
    (`train_base.py:552-553`), so the switch does nothing (the configuration
    cell refuses it under SAC since decision D-D79, 2026-11-28);
  - the T. rex stance's mirror-symmetric PPO (`[ppo] symmetry = "mirror_kl"`,
    `stance.toml:511`) is dropped with no log: `pop_symmetry_config` runs only
    when `algorithm == "ppo"` (`train_base.py:1826`);
  - the T. rex recovery, whose TOML says "this stage is PPO-only" and extends
    only the stance's `[stage]`, `[env]` and `[ppo]`
    (`configs/tyrannosaurus_rex/anatomical/recovery.toml:26-27,57`), still
    trains: it logs that SAC "will train on stable-baselines3's default
    hyperparameters" (`train_base.py:554-562`) while the warm start keeps the
    stance checkpoint's saved SAC settings (the load passes only the empty
    table's keys, `train_base.py:959`), and its hyperparameter digest hashes
    that empty `[sac]` (`config.py:950-967`), so replicate matching ignores a
    change to the stance's `[sac]`;
  - with no trunk, the section-4 load preflight proves the load path on a
    throwaway PPO (`policy_loading.py:826-828`), never on a SAC archive.

  Fix: a dropdown that labels SAC experimental; `N_ENVS` per algorithm, as the
  CLI does; refuse `CRITIC_PHASE`, a `[ppo]` symmetry and the T. rex recovery
  under SAC, or let the recovery extend the stance's `[sac]` too. (2026-08 gap
  review NB9; 2026-11 notebook review: sac-path-second-class,
  sac-path-second-class-unflagged,
  fu-sac-artifact-path-trex-recovery-sac-recipe-unrecorded)
- **LOW** — **the run tree marks only the screens trained since `RUN_KIND`
  existed (read from the code 2026-11-24).** A screen the SB3 notebook trains
  with `RUN_KIND = "screen"` lives under `<algo>_screen/`, outside trunk
  selection and replicate discovery, and a certification run refuses it as a
  pinned `TRUNK_FROM` (`environments/shared/run_tree.py`). Two gaps remain.
  The screens and pilots trained before then sit in `<algo>/`:
  `environments/shared/run_exclusions.py` names the nine r10 stance screens,
  and the Drive summary labels them and the runs built on them `screen`, but
  the walking pilots (`pilot*_walk`) and any other older screen read
  `run_kind` `certify` there, and both scans treat them as certification runs
  (the pilots' recorded PPO digest differs from `main`'s, D-D36, so today they
  match no landed recipe). And nothing refuses a certification run that
  loads, resumes or widens a later screen's checkpoint: the manual node's
  `MANUAL_LOAD_PATH`, `widen_checkpoint` into `<algo>/` and a command-line
  `--load` record the load in `stage_config.json`, but rule (d) of
  `run_exclusions.py` reads only the runs it names. Fix: have rule (d) count
  a load path through a `<algo>_screen/` directory for a run of `<algo>/`,
  and add an older screen to the ledger once its digests can match a landed
  recipe.
- **LOW** — **a quick test is marked only by its folder and one summary line
  (read from the code 2026-11-26).** Decision D-D16 (b) put quick tests under
  `<algo>_quick_test/` "with no digest or provenance change", and since
  `29cd504` the Drive summary reads that tree into its `run_kind` column and
  its strict promotion cell refuses a quick-test or screen run. Nothing else
  records it: `provenance.json`, `gate_verdict.json` and the CSV rows carry no
  `quick_test` field (`result_bundle/` and `reporting/csv_output.py` never
  mention it; only `write_training_summary` prints it,
  `reporting/text_summaries.py:195,212`), and `validate_result_bundle`
  verifies a 50k-step quick-test bundle as `canonical-valid` like a full run
  (`result_bundle/audit.py:639`; the review's brachiosaurus quick test of
  2026-11-22 did, and the T. rex's 50k-step stance passed under the same gate
  digest as its 12M-step certification). A quick-test run copied or moved out
  of its tree, or published to `results/` by hand (the promotion path is
  manual), carries nothing that says what it is. Fix: write `quick_test`, or
  the run's `run_kind`, into the `provenance.json`, `gate_verdict.json` and
  CSV rows of new runs, and have `validate_result_bundle` refuse such a run
  under `require_complete=True`; the entry above has the same gap for screens.
  (2026-11 notebook review: quick-test-indistinguishable)
- **LOW** — **the SB3 notebook's section 8 evaluation shows and keeps nothing,
  because no cell configures logging (read from the code 2026-11-26).** Cell
  26 (`notebooks/sb3_training.ipynb:2134-2143`) rolls `evaluate()` on the
  deliverable's handoff for 30 episodes, but `evaluate()`
  (`environments/shared/evaluation.py:618`) reports only through `logger.info`
  (`_log_eval_results`, :801) and returns `None` (:797-798), and no cell of
  the notebook installs a logging handler, so Python's last-resort handler,
  which prints WARNING and above, drops the INFO records: in the review's runs
  the cell printed only its own "Evaluating the ... policy" line. Nothing is
  saved either: the velocity consistency, posture, termination reasons and (on
  a model with a success event) success rate that the cell's comment promises
  reach neither the run directory nor the bundle. The same gap hides the
  library's other INFO lines in every session: the gate's PASS line
  (`reporting/stage_artifacts.py:1811`; a FAIL is a WARNING, :1813, and
  shows), the six-heading certificate's summary (:566), the command-tracking
  report's (:668) and the baseline watch's per-evaluation line
  (`curriculum/baseline_watch.py:141-147`). The chain loop prints its own
  "gate PASSED" (`sb3_training.ipynb:1690`) and the reports are written to the
  stage directory, so nothing is lost, only unseen. Cell 26's seed
  (`DEFAULT_EVAL_SEED` = 2000, `evaluation.py:48`) differs from the bundle's
  evaluation seed (`EVALUATION_SEED`, 3000 above `SEED`) on purpose, for
  paired command-line runs; the gait symmetry, stride frequency and cost of
  transport the cell once advertised were deleted with LocomotionMetrics' gait
  block. Fix: install a notebook logging handler at INFO for `environments.*`
  in the setup cell; have `evaluate()` return its aggregate and cell 26 print
  it as a table, saved as an artifact the bundle declares (an undeclared file
  stops cell 32's bundle check, as cell 28's comment says) or built from the
  node's `evaluation_selected.csv`. On the walk, the deliverable on every
  model, lead the table with the command-tracking report's statistics: the
  printout's "Consistency" line inherits the `velocity_consistency` blow-up at
  zero speed (Infrastructure, first entry) on the stand episodes. (2026-11
  notebook review: report-evaluate-silent,
  notebook-logging-hides-eval-and-certificates, cell24-eval-invisible-unsaved,
  deliverable-eval-not-persisted)
- **LOW** — **the zero-action preflight names no next action, and reads a
  falling statue on a stance-gated plant as a gate problem (read from the code
  2026-11-26).** `preflight`
  (`environments/shared/scripts/zero_action_baseline.py:267`, which the SB3
  notebook's cell 13 runs every session) judges a stance gate kind first
  (:295-306), so a statue that falls on every episode, as the Allosaurus' does
  under its `stance_quality/v1` gate, reads "CHECK GATE — the stance gate
  refuses the statue" and never reaches "CHECK PLANT — the statue never
  survives a full episode" (:311-315). On the brachiosaurus' and
  dibothrosuchus' `reward_and_length/v1` stances, whose rail is 0.6 x the
  statue by design, it reads "FAILS — a statue clears this gate" (:309-310) in
  every session. No verdict says what to do next, so each model shows its own
  alarm, and cell 12's markdown does not say which ones are expected. Fix:
  test `n_standing == 0` before the stance-gate branch, mark the by-design
  cases as expected, and give each verdict its next action. (2026-11 notebook
  review: zero-action-preflight-verdicts)
- **LOW** — **a day-long node shows no progress, ETA or notification in the
  SB3 notebook (read from the code 2026-11-26).** `VERBOSE` (cell 6, "0=quiet,
  1=progress bar, 2=debug", `notebooks/sb3_training.ipynb:226`) is passed as
  SB3's own verbosity (`train_base.py:563`) as well as the progress bar's
  switch (`progress_bar=verbose >= 1`, :2021), so `VERBOSE = 1` also prints
  SB3's logger table every rollout, and the default 0 shows nothing for a node
  of up to about 21.5 h (an 11M T. rex stance, scaled from the 12M runs).
  `progress.json` already holds a `projected_finish`
  (`curriculum/progress_report.py:449-466`) that no cell reads; cell 15 prints
  that the diagnostics "will log to TensorBoard"
  (`sb3_training.ipynb:1033-1035`), but no cell opens it; and nothing tells the
  user when an unattended node finishes, fails its gate or crashes: status is
  pull-only, by opening `progress.json` on Drive. No doc says what a ~24 h run
  needs of the Colab tab and runtime tier. Fix: drive the progress bar from
  `VERBOSE >= 1` with SB3's verbosity at 0 (2 keeps the table), print a
  one-line `progress.json` readout per evaluation with the projected finish
  against the session's end, add an optional `%tensorboard` cell and an
  optional notification hook (a webhook or ntfy topic) on node end, halt and
  exception (the exception handler of the runtime-release entry above is where
  the last belongs), and a short note on the Colab conditions. (2026-11
  notebook review: no-progress-eta-or-tensorboard,
  fu-colab-runtime-specifics-no-unattended-run-signal-or-colab-tier-guidance)
- **LOW** — **the SB3 notebook trains on whatever device torch finds, and
  neither chooses, prints nor records it (read from the code 2026-11-26).**
  The notebook's metadata opens Colab on an L4 GPU (`"accelerator": "GPU"`,
  `"gpuType": "L4"`, `notebooks/sb3_training.ipynb:2282-2284`), and
  `train_base` builds PPO and SAC at SB3's default `device="auto"`
  (`train_base.py:1058`, `:1061`), so a Colab run trains its small MLP policy
  on CUDA, with no stated reason. No knob chooses the device and the notebook
  never prints it (`VERBOSE = 0`, `sb3_training.ipynb:226`, hides SB3's
  "Using cuda device"); only SB3's GPU warning below names it, for a plain PPO
  policy alone. No record names it either: `stage_config.json` carries a `gpu`
  block whenever a GPU is present (`config.py:1398-1400`) and provenance the
  CUDA runtime and driver (`result_bundle/provenance.py:136-165`), which say
  that a GPU existed, not that the policy ran on it. SB3's warning that PPO
  with an MLP policy belongs on the CPU fires only for a policy class named
  `ActorCriticPolicy` (`_maybe_recommend_cpu` in SB3 2.9.0's
  `on_policy_algorithm.py`), so it stays silent for
  `CriticPhaseActorCriticPolicy` (`critic_phase_ppo.py:231`), which the
  notebook's `CRITIC_PHASE` switch builds. Whether the L4 trains these
  networks faster than Colab's CPU is unmeasured. Fix: measure PPO throughput
  on both, then add a `DEVICE` knob that `train_base.train` passes to the
  model, printed at start and recorded in `stage_config.json`'s run block.
  (2026-11 notebook review:
  fu-colab-runtime-specifics-ppo-device-unpinned-unrecorded-no-runtime-guidance)
- **LOW** — **the SB3 notebook never prints the handoff checkpoint's step, and
  it advances a stance the hop watch stopped (read from the code
  2026-11-26).** The robust-best rule has no warm-up
  (`RobustBestModelCallback`, `curriculum/checkpoints.py:36-80`), so it can
  hand off a shallow early checkpoint, and
  [NEXT_STEPS.md](NEXT_STEPS.md):955-956 names "the audit and the handoff's
  step count" as the guards. The step is written to `handoff_in_training.json`
  and logged at INFO (`reporting/stage_artifacts.py:492-515`), which no
  notebook cell shows, since none configures logging (the section 8 entry).
  What the notebook prints is the step of the best mean evaluation ("Best
  model eval: ... (at N steps)", `reporting/stage_artifacts.py:1970`),
  "Selected checkpoint: robust_best_model" (:2022) and, in the chain loop
  (cell 20), "gate PASSED (...); handoff robust_best_model.zip"
  (`sb3_training.ipynb:1690`). A stance the hop watch stopped is judged as
  usual, by design (`curriculum/hop_watch.py:20-23`), and the chain loop reads
  no `hop_watch.json`, so a stopped seed that certifies warm-starts the next
  node without a word. Fix: print the handoff's step and whether the hop watch
  tripped in the chain loop's verdict line, and stop the chain after a
  hop-watch stop unless an override is set. (2026-11 notebook review:
  velociraptor-stance-certificate-certifies-init-handoff)
- **LOW** — **the Drive summary's tables set the final checkpoint's numbers
  beside a verdict on the selected one, show no walk statistic, and stop at
  PASS/FAIL with no certification board across models (read from the code
  2026-11-26).** In `notebooks/google_drive_summary.ipynb`:
  - the detailed and per-model tables (cells 16 and 18, :1268-1273 and the
    per-model list after :1321) show `mean_forward_vel` and
    `mean_success_rate`, read from the final checkpoint's evaluation (the "Avg
    fwd vel" line of a stage's `stage_summary.txt` or the results CSV,
    :513-556; `stage_summary`'s `avg_forward_vel` and `mean_success_rate`,
    :878-881), beside `stage_passed`, which judges the selected handoff; the
    `selected_model_mean_forward_vel` and `selected_model_success_rate` the
    notebook does read (:885-886) are in neither table, and since the hunt's
    removal a success rate is recorded only for the Allosaurus, the one model
    with a success key;
  - the pivot (cell 14, its `s<N>_reward`, :1179) and the detailed table
    (:1267-1268) set the in-training best evaluation, `best_mean_reward`,
    beside `reward_threshold`, while the rail is judged on the selected
    checkpoint's panel (`best_model_reward` first,
    `environments/shared/reporting/gates.py:1720,1859`), the CSV's
    `selected_model_mean_reward` (`reporting/csv_output.py:304`);
  - a walk is judged by `command_tracking/v1` on its own panels, and its
    family statistics reach the results CSV (the `command_*` columns,
    `csv_output.py:25-73`) and the stage summary, but the notebook reads no
    `command_*` column, so a walk's row shows a final-model velocity and empty
    `reward_threshold` and `forward_vel_threshold` columns (`min_avg_reward`
    and `min_avg_forward_vel`, :598-600, which no walk declares);
  - neither table shows a gate kind, the statistic a node was certified on
    (the six-heading seed count and bound, or a walk's cruise and stand family
    numbers, all in `collected_results.csv` since D-D61), the replication
    count or provisional label, the lineage (reused-from, the trunk), the
    plant revision, or a link to a node's `training_history.html`,
    `gate_progress.png` or replays, and nothing gives one row per model and
    deliverable for the run `select_trunk` (`ancestors.py:1157`) would choose.

  Fix: show the selected checkpoint's statistics in both tables
  (`selected_model_mean_reward` beside the rail, and for a walk its cruise
  success bound and stand success fraction), and drop or relabel the
  final-model columns; add a "Certification status" section, one row per model
  and deliverable (the selected run, gate kind, headline statistic,
  replication n/N and provisional, reused-from, plant revision, artifact
  paths), saved beside the cross-run table as `logs/certification_status.csv`.
  (2026-11 notebook review: hunt-success-statistic,
  drive-summary-no-certification-board, and the Drive half of
  collected-results-dead-and-kind-dependent-columns)
- **LOW** — **the SB3 notebook embeds every replay twice and re-plots the
  older figures, but never shows the gate, command or history figures (read
  from the code 2026-11-26).** `notebook_runtime.display_stage_videos`
  (:60-73) base64-embeds every mp4 of a stage with `autoplay loop`. The chain
  loop calls it after each node (cell 20, `sb3_training.ipynb:1676`), and the
  replay cell embeds every certified node again, reused ancestors included
  (cell 30, :2196-2205). A stance or recovery node records three camera views
  of the final and the selected checkpoint
  (`reporting/stage_artifacts.py:2451-2452`), six clips; the review's executed
  Allosaurus notebook held about 4.5 MB of video for episodes of about 155
  steps, which scales to about 27 MB of base64 per 1000-step stance or
  recovery node (2026-11-22). Cell 28 (:2169-2174) re-plots
  `plot_training_curves` and `plot_diagnostics_graphs`, already drawn by the
  chain loop (:1677-1678), while the saved `gate_progress.png`,
  `foot_contacts.png`, a walk's `command_tracking.png` and
  `command_gate_progress.png`, and `training_history.html` are never shown
  inline (the last by design,
  `docs/TRAINING_ARTIFACT_PRESENTATION.md:1733-1734`). Fix: inline only the
  selected checkpoint's default-view replay, behind a `SHOW_VIDEOS` knob, and
  print the other paths; in cell 30 show only reused ancestors; display the
  saved gate and command figures and link `training_history.html` instead of
  re-plotting. (2026-11 notebook review: embedded-videos-bloat-notebook,
  inline-video-bloat, notebook-inline-figures)
- **LOW** — **the SB3 notebook's report names no offline analysis tool, and
  nothing compares runs or draws a whole chain (read from the code
  2026-11-26).** Section 8 (cells 25-32) points to no script; the notebook
  names only `zero_action_baseline.py` (cell 12,
  `notebooks/sb3_training.ipynb:736`) and `backfill_gate_verdict.py` (:1559).
  `scripts/action_bound_report.py`, `scripts/observation_ablation_report.py`
  and `scripts/joint_excursion_report.py` are described only in plans,
  reviews and investigations (among them STAGE1_SPLIT_PLAN.md,
  STAGE1B_IMPLEMENTATION_PLAN.md and investigations/TREX_REVIEW_2026_07.md);
  no website page lists any script,
  and [TRAINING_ARTIFACT_PRESENTATION.md](TRAINING_ARTIFACT_PRESENTATION.md)
  documents `plot_training_history.py` (whose several-stage-directory mode
  overlays runs) and `episode_trace_report.py`, which nothing in the notebook
  mentions. Each node's step counter restarts at its warm start
  (`reset_num_timesteps=not resuming`, `train_base.py:2027`) and no figure
  puts a chain on one axis or sets two runs or seeds side by side. PPO's and
  SAC's optimizer metrics reach `diagnostics.npz` (`diagnostics.py:467-506`),
  but `scripts/compare_run_diagnostics.py`, the one tool that compares them
  across runs, lists PPO's keys alone (`_ALGO_SIGNALS`, :40-47): SAC's
  `critic_loss`, `actor_loss` and `ent_coef` are never compared (the SAC
  diagnostics entry under Training / RL; the missing `%tensorboard` cell is in
  the entry on progress and ETA). Fix: list the offline tools with one line of
  usage each in section 8 and on a website training page; add SAC's keys to
  `compare_run_diagnostics.py`; optionally write a run-level chain figure on a
  cumulative step axis. (2026-11 notebook review:
  fu-offline-analysis-tools-species-coverage-offline-tools-not-surfaced-in-report-section,
  no-chain-or-optimizer-overview)
- **LOW** — **the SB3 notebook's configuration cell offers choices it later
  refuses, and refuses several only after the run directory is written (read
  from the code 2026-11-26).** Cell 6 sets 16 knobs, three of them Colab
  `@param` fields (`SPECIES`, `EMBODIMENT` and `BEHAVIOR`,
  `notebooks/sb3_training.ipynb:212-213`, :238); `ALGORITHM`, `QUICK_TEST`,
  `AUTO_DISCONNECT` and the rest are free text, some under one-line comments
  of several hundred characters of decision ids (`TRUNK_FROM`'s, :239). The
  two dropdowns make 12 pairs, 7 of them models; the other five are refused in
  that cell, before anything is written. `BEHAVIOR` offers `"recover"` on
  every model, and three have it. Cells 12 and 13 still say "all six" models
  (:734, :758). Three refusals come after the storage cell (cell 8) has
  created the run directory and written `provenance.json` (`_run_dir.mkdir`,
  :484; `initialize_result_bundle`, :498): a pinned `TRUNK_FROM` that is this
  run, not a run or another model's (:554-579; only a trunk in a screen or
  quick-test tree is refused first, :444); an unknown `BEHAVIOR`
  (`MANIFEST.resolve_behavior`, cell 9, :602); and a `RETRAIN_FROM` that is no
  node or off the chain (:612-618). A corrected Run all in the same runtime
  re-enters that run id, so only a restarted runtime leaves the empty run
  behind. Until decision D-D79 (2026-11-28) `CRITIC_PHASE = True` was refused
  later still, when the first node built its training env, on every model
  whose env defines no `stance_width_settle_steps` (the velociraptor, the
  notebook's default model, the brachiosaurus, the dibothrosuchus, the
  Compsognathus robot and the Allosaurus); it now trains on every model
  (`critic_phase.DEFAULT_SETTLE_STEPS`), and the configuration cell refuses
  `BEHAVIOR = "auto"` and `CRITIC_PHASE` under SAC before anything is written. Fix: one `MODEL` dropdown of the
  seven model ids; `@param` on the other knobs, with the decision prose in the
  markdown cell; one call before cell 8 writes anything that validates
  `BEHAVIOR`, `RETRAIN_FROM`, a pinned `TRUNK_FROM` and `CRITIC_PHASE` against
  the model's env; and "all six" corrected. (2026-11 notebook review:
  knob-density-and-dropdowns, knobs-validated-after-write,
  velociraptor-critic-phase-unsupported-on-raptor)

## Testing / CI

- TOML→env round-trip test: construct each env with each stage's
  `env_kwargs`, assert no unknown/unused keys. (June §6.8)
- **LOW** — **the site deploy waits on nothing but its own build (read from
  the code 2026-09-30).** `.github/workflows/deploy.yml` runs on a push to
  `main` that touches `website/**`, a `summary.json` or the workflow itself
  (:3-15), and its `deploy` job needs only the `build` job (`npm ci`, `tsc`,
  `npm test`, the site build) and the `main` ref (:76-82). A push with a hand-edited or
  stale `website/src/data/species.generated.json` therefore deploys while
  python-ci's `species_catalog --check` (`python-ci.yml:127-130`) fails on the
  same commit, and `main` has no required checks
  ([CLEANUP_PLAN_2026_09.md](CLEANUP_PLAN_2026_09.md) §4.8 item 2). The
  workflow's comment (:7-11) leaves the wait to a required-checks setting, but
  it can be built in the tree. Fix: a `species_catalog --check` step in the
  build job, or a `workflow_run` trigger on python-ci that deploys only when
  it succeeds, with python-ci's path filters widened to all of `website/`
  (today a push that touches only the site config never runs python-ci); turning on required checks (the cleanup plan's decision 18 (a))
  closes it for everyone who cannot bypass them. No cleanup PR owns it.
  (2026-08 gap review CI8)
- **LOW** — **the SB3 notebook's own training and report cells are tested on
  the Compsognathus pair only (read from the code 2026-11-26).**
  `test_actual_notebook_training_stance_and_recovery_reports`
  (`environments/shared/tests/test_compsognathus_longipes_training.py:418-420`)
  runs the notebook's cells through `train_stage` and
  `generate_stage_artifacts` for the two Compsognathus models alone
  (`MODEL_IDS`, :40), and a pull request runs only its robot PPO case
  (`.github/workflows/python-ci.yml:459-466`). Since decision D-D62 every
  model's committed walk trains through `train_base.train` and is judged by
  `generate_stage_artifacts` in
  `environments/shared/tests/test_walk_species_training.py` (:120-121), on all
  seven on the nightly, dispatch and `full-ci` runs and on the robot alone on
  a pull request (`python-ci.yml:488-494`), but its stance parent is a 64-step
  root whose gate is never consulted (its module docstring). So the
  velociraptor's, T. rex's, brachiosaurus', dibothrosuchus' and Allosaurus'
  stance gates, with their own keys (the T. rex probe battery and hop watch,
  the quadrupeds' `reward_and_length/v1`, the Allosaurus' heading probe), and
  the T. rex recovery node never meet the post-stage judge after real training
  in a test. Fix: parametrize the notebook smoke over all seven models on the
  nightly run, with `smoke_configs`-style overrides. (2026-11 notebook review:
  cmp-notebook-path-tested-on-compso-only)
