# SB3 training notebook, analytics and artifacts: cross-species review

*2026-11-22. Scope: `notebooks/sb3_training.ipynb`, the RL pipeline under `environments/shared/` (trainer, curriculum, callbacks, gates), every post-training artifact (figures, training-history report, replays, probes and certificates, summaries, result bundle), the Drive summary notebook, `results/` and the website presentation, across all 7 models.*

## 1. Bottom line

**Is it consistent across the species?** The plumbing is; the content and the recipes are not.

- **Consistent (checked by running the notebook on all 7 models; complete stance artifact sets were seen on four models and downstream figure sets on six walk and hunt nodes, see 2a).** Every model goes through the same chain loop, gate machinery and artifact generator. Every node writes the same core set: `gate_verdict.json`, `stage_summary.txt`, the evaluation CSVs, the five per-node figures, the 20-page training-history report (PNG, HTML, JSON), `progress.json`, the models and TensorBoard. Where the set differs, it differs mostly by gate kind or role, and mostly by design. Stance certificates and heading probes follow the stance gate kind, recovery evidence exists only where there is a recovery node, task-success evidence only for the T. rex hunt, and front/side views and per-frame CSVs only on stance and recovery nodes. Section 5 lists the deliberate differences.
- **Not consistent: what the artifacts say off the T. rex path.** The tooling grew up around T. rex, and its constants have become shared defaults:
  - `KEY_ACTUATORS`, the 0.08 exploration reference and the `statue 0.998` literal come from T. rex.
  - The success panel only knows `strike_success` and `bite_success`.
  - Dibothrosuchus's snap keys are missing from the diagnostics.

  So the same chart reads as intended mostly on T. rex (and even there the 'stride freq', 'cost of transport' and per-step 'Strike Success Rate' panels are mislabelled, as on every model; see gap 3), is empty or mislabelled where a species' success keys or actuators differ (Brachiosaurus, Dibothrosuchus, both Compsognathus models, Allosaurus), and is largely placeholder pages on Allosaurus, the raptor and Brachiosaurus (by design for the last two, D-D46).
- **Not consistent: recipes and reward design.** Several differences have no recorded reason:
  - exploration noise (Dibothrosuchus at std 1.0; the Compsognathus robot at 7x its statue's noise cliff)
  - hunt collapse backstops (three of the four hunts that arm one, raptor, Brachiosaurus and Dibothrosuchus, still have the floor-100, no-warm-up pattern that was fixed only for T. rex; the Compsognathus hunts set no floor, so theirs never arms)
  - stance certification guards (the anatomical Compsognathus lacks T. rex's and the raptor's hop watch, archive and two-seed bar)
  - what a walk "speed bar" means

  Most seriously, on **3 of the 7 final nodes succeeding pays less than not succeeding**: both Compsognathus target_reach hunts and the Allosaurus walk. On the Compsognathus hunts this is predicted to fail their 70% success gate. The Allosaurus walk gate does not require completion, so there it costs the `course_complete` metric and the replays rather than the verdict. None of the three has been trained yet.
- **Not consistent: maturity.** Allosaurus, the newest and only torque-driven model, trains on near-default PPO/SAC tables, and much of the shared tooling misreports it. No stage carries a status field: apart from the free-text "Pilot:" description on the two Compsognathus recovery nodes, nothing in the notebook tells a user which recipes are tuned, pilot or on hold.

**Are there gaps?** Yes. The ones with the most user impact:

1. **The notebook computes things it never shows.** Section 8's `evaluate()` cell rolls 30 episodes and prints only its own header line, because the notebook never configures logging (which also hides the passing gate line and the certificate summary). `gate_progress.png` and the training-history report are never displayed (for the training-history pages this is documented as intended, TRAINING_ARTIFACT_PRESENTATION.md:1612-1613, which tells users to open `training_history.html` themselves). The run summary leaves out the gate verdict, failure reasons and certified statistic.
2. **Wrong reference lines and alarms.** Every node after stance (recovery, walk, hunt) is compared with the *stance* statue in the baseline watch, `progress.json` and `gate_progress.png`. On statue-optimal stances the watch calls a by-design PASS a "worse-than-trivial local optimum".
3. **The metric the hunt gates judge is never charted.** Per-evaluation success is saved but no figure draws it against `min_success_rate`. The "Strike Success Rate" panel is empty on 5 of 7 models, the "cost of transport" is a scaled action penalty, and the gait symmetry and stride frequency maths is wrong.
4. **QUICK_TEST is not a smoke test.** Only the step budget shrinks. Certificates, probes and replays run at full size, where it stops depends on the species, and a 50k-step quick test verifies as `canonical-valid` with no quick_test field in its provenance, verdict or CSV rows (D-D16(b) deliberately left provenance unchanged; only the directory name and the 50,000-step budget give it away).
5. **Colab robustness.**
   - No awareness of the ~24 h session cap. Default chains need 9-34M steps (28M for T. rex, 32M for the raptor), and a single 12M-step T. rex stance took about 23 h on an L4.
   - Resume is a manual, usually two-pass procedure (a deliberate choice, D-D16(c), that fix 9 would extend with an `auto` mode).
   - Apart from the final cell, only a failed gate releases the runtime (any other error leaves it assigned until Colab reclaims it), and figure, replay or probe failures are only logged; a run re-entered in place after its verdict never regenerates them.
   - There is no inline progress, ETA or notification during day-long nodes (`progress.json` on Drive already holds a projected finish that nothing shows).
   - In-training evaluation runs serially at about 0.8 env steps per training step on stances, which helps push 12M-step stances to the cap. The notebook hard-codes `eval_freq`, and the per-stage cadence key (D-D36) is unmerged.
   - The training device is never chosen or shown.
6. **SAC is offered like PPO but behaves differently.** Its resume is not a continuation, warm starts begin with an empty replay buffer that the comments say carries over, and its diagnostics write a row every `train_freq` vector step, an estimated 750k rows (from the code, at N_ENVS=4 and SB3's default train_freq=1) for the 3M-step Allosaurus stance.
7. **Results are hard to see and publish.** While node 2 or later is training or interrupted, the Drive summary audits the run as a canonical conflict and hides its already-certified nodes. The summary also has no cross-model certification board. Promotion to `results/` and the website is manual, nothing marks a published result current or stale, and `results/` cannot hold a trunked chain or seed replicates (themes 3.20-3.21).

**What should be added?** Section 2 ranks ten fixes. Four are small (S) and most of the rest take a few days, because much of the data already exists and only needs to be surfaced or keyed correctly. Beyond those fixes, section 3 recommends these features and quality-of-life additions:
- **Notebook:** a printed and saved section 8 table; `gate_progress.png` shown inline and a link to `training_history.html`; a per-node verdict block; one MODEL dropdown, with every knob validated before anything is written; a `SHOW_VIDEOS` knob that embeds only the selected replay; node ids instead of "Stage N"; an optional `%tensorboard` cell; and a list of the offline analysis tools in section 8 (3.2, 3.19, 3.22, 3.26-3.29).
- **Pipeline:** a real smoke mode, run for all 7 models in CI; `RESUME_STAGE='auto'` with `RUN_ID` persisted to Drive, a session-budget check, an `EVAL_FREQ` knob (or the per-stage cadence key) and a device knob; notification and exception hooks; a per-node `status` (tuned, pilot, held); an `artifact_status.json` of expected versus present files; and a seed-replicate launcher for models with `certification_seeds > 1` (3.4-3.7, 3.14, 3.27).
- **Results:** a run-level `index.html`, a cross-model certification board in the Drive summary, and a promotion tool that refuses quick tests and marks results current or stale (3.19-3.21).

## 2. Fix first (ranked)

| # | Fix | Why | Models | Effort |
|---|---|---|---|---|
| 1 | Fix the inverted task rewards: both Compsognathus `behavior.toml`s (zero alive bonus, cut posture/height, raise `target_reach_bonus` well above per-step/(1-γ)) and the Allosaurus walk (completion bonus, or no success termination). Choose hunt handoffs by the gated success statistic, with reward only as a tie-break. | A success step is worth about -59 against hovering on Compsognathus, and arriving forfeits about 250 on Allosaurus. The deliverable is predicted to be unlearnable and the selector would prefer non-reaching checkpoints. Check this before anyone spends Colab time on these nodes. | compso-a, compso-r, allo; selection affects all 6 hunts | M |
| 2 | Add a notebook logging handler (INFO for `environments.*`), have `evaluate()` return its aggregate, and print and save a table in section 8, leaving out cost of transport, symmetry and stride frequency until their maths is fixed (theme 3.10). Show `gate_progress.png` and link `training_history.html` per node. Add a verdict block (gate kind, PASS/FAIL, reasons, statistic, handoff step) to `training_summary.txt` and the per-node print. | The notebook already computes all of this and then hides it. | all | S |
| 3 | Make the zero-action baseline stage-aware (key the file by stage, refuse a mismatch, or skip the line), and make `BaselineProgress` gate-kind aware with a tolerance. | Every later node's baseline watch, `progress.json` and `gate_progress` line currently use the stance statue (the gate verdict itself does not read it), and statue-optimal stances get false alarms. | all | S-M |
| 4 | Chart what the gates judge: per-evaluation success with a binomial band against the bar, and per-evaluation eval forward velocity, in `gate_progress`. Read success keys from each env's `SUCCESS_KEYS`, which also adds the snap keys. Count successes separately from falls. | Hunt gates fail on success rate, and nothing shows success over training. | all hunts; worst on brachio, dibotho, compso | M |
| 5 | Add a real smoke mode, `QUICK_TEST='smoke'`: small panels, no probe battery, short replays, continue past gate failures, never auto-disconnect. Write `quick_test` into provenance, the verdict and the CSV, and refuse quick tests at promotion. Run the notebook smoke for all 7 models in CI, not just the Compsognathus pair. | Under load, the raptor and T. rex quick stances were killed at the 40-minute harness limit inside certificate, replay and probe generation, and a quick-test bundle looks publishable. | all | M |
| 6 | Port the T. rex hunt backstop fix (relative floor 0.45 x measured hunt statue, peak warm-up ≥ warm-up + ramp, explicit 20/10/0.5) to strike, food_reach and snap. | The floor of 100 sits 2.8-19x below a do-nothing policy, and snap can stop at about 950k, inside its own 1.0M warm-up-plus-ramp window (plausible, not observed: no current-config hunt run exists). | raptor, brachio, dibotho | S |
| 7 | Calibrate stance exploration: measure the noise cliff and set `log_std_init`, or record why std 1.0 is intended. Add the robot and the quadrupeds to the noise test. | Dibothrosuchus: 0/873 training rollouts reach the horizon. Compsognathus robot: statue falls at σ 0.02, recipe starts at 0.135. | dibotho, compso-r (brachio, allo) | S |
| 8 | Add a per-node `status` (tuned, pilot or held, with a doc link) in `stages.toml`, print it in the chain table and warn before a held node. Mark Allosaurus as pilot. | GQ-3 recommends holding several sessions; the notebook offers them all alike. | all | S |
| 9 | Colab hardening: an exception hook that writes the traceback and disconnects on any error; a session-budget check before each node; `RESUME_STAGE='auto'` with `RUN_ID` persisted to Drive; a device knob that is printed and recorded; an optional `%tensorboard` cell and notification hook. | Lost compute and manual recovery on day-long nodes. | all | M-L |
| 10 | Give hunt replays a task camera (as Brachiosaurus has) and make the Allosaurus target visible. Write the per-step CSV for every node. | On 5 of 6 hunts the prey starts out of frame: 0-7 of 25 spawn points are visible, against 25 of 25 on Brachiosaurus. | raptor, trex, dibotho, compso-a, compso-r, allo | S-M |

Smaller but cheap wins, mostly effort S:
- Treat `stance_probe_filter_hz = None` as unset.
- Show node ids instead of "Stage N".
- Embed only the selected replay, once.
- Validate all knobs before the run directory is written.
- Use one MODEL dropdown of the 7 valid ids.
- Label SAC as experimental and refuse its documented PPO-only cases.
- Fix the stale TOML comments and getting-started docs.

## 2a. How this review was done, and its limits

- **Three review workflows (about 130 subagents).**
  - Six dimension reviewers: notebook, pipeline, artifacts, charts, publication, metrics.
  - Seven per-model profilers and a cross-model comparator.
  - A completeness critic with four follow-ups: offline tools, Colab runtime, the first executed hunt node, and SAC artifacts.
  - A synthesizer.

  Every finding was checked by an independent verifier told to refute it: 143 checked, 141 survived (62 confirmed, 79 partly confirmed with corrected wording). Each of the 14 headline claims in sections 1-2 was then re-checked by two more verifiers, one tracing code and one reproducing numerically. All 14 held, 7 with tighter wording (used here).
- **The real notebook was executed headlessly** on all 7 models with `QUICK_TEST=True` (50k steps per node), `N_ENVS=2`, PPO, on CPU with OSMesa rendering.
  - Stance ran through the chain loop.
  - Downstream nodes ran through the notebook's MANUAL cell, loading the parent's quick-test checkpoint: the Allosaurus, Brachiosaurus, Dibothrosuchus and Compsognathus-robot walks; the Brachiosaurus and Compsognathus-robot hunts; T. rex recovery, walk and hunt; and Compsognathus-robot recovery.
  - Section 6 summarizes what the runs showed.
- **Limits.**
  - The 4-CPU machine ran under heavy contention (load 7-12), so wall times are inflated. Several runs were cut off by the harness: the raptor, T. rex and Compsognathus-robot stances and the Compsognathus-robot walk during artifact generation, after their verdicts; the anatomical Compsognathus stance and the T. rex walk before their verdicts; and T. rex recovery during training. The Compsognathus-robot hunt reached its verdict and was then cut off during artifact generation. Downstream figure sets were seen on the Allosaurus, Brachiosaurus, Dibothrosuchus and Compsognathus-robot walks and on the Brachiosaurus food_reach and T. rex bite hunts. To save time the raptor and anatomical Compsognathus downstream nodes and the Dibothrosuchus hunt were not run; they go through the same code paths as the nodes that did.
  - Not executed end-to-end: Colab itself, full-budget training, and SAC through the notebook. SAC artifacts were generated from a test-produced SAC stance.
  - The Compsognathus and Allosaurus reward inversions are predicted from the reward arithmetic and zero-action rollouts. They have not been observed in a trained run, because no such run exists. The success-blind hunt selection is likewise a structural risk: no mis-selection has been observed, and the one full Brachiosaurus food_reach run ended at 100% success with a best evaluation (1368) well below the ~1875 hover level.
- **Already tracked.** Each theme says what `docs/KNOWN_ISSUES.md`, `NEXT_STEPS.md`, the plans or D-xx decisions already cover. Three of the ten ranked fixes (1-3) address problems no doc tracks; the other seven are tracked only in part.

## 3. Findings by theme (priority order)

Each theme merges the verified findings listed under it. Severity: **high** = wrong results or a deliverable that cannot be learned; **medium** = misleading output, wasted Colab compute, or notable friction; **low** = polish. Effort: S < 1 day, M = a few days, L = a week or more. "Tracked" says what the repo docs already cover.

### 3.1 Task rewards where succeeding pays less than not succeeding, plus checkpoint selection that ignores success

*High · reward/recipe design · effort M · models: compso-a, compso-r, allo, brachio (success-blind selection affects all six hunt nodes)*

Both Compsognathus target_reach hunts pay about 0.85 per supported standing step but only target_reach_bonus = 25 for a success that ends the episode, so at gamma 0.99 (with SB3 bootstrapping through truncation) a success transition has an advantage of about -59 and a 3M-step hunt is predicted to fail min_success_rate 0.7 (predicted from the arithmetic and an 850-per-1000-step statue rollout; no Compsognathus hunt has been trained). The Allosaurus walk ends on course completion with no bonus while paying about 2.0-2.5 per step (about 248 of discounted value forfeited), breaking the repo's own rule that a success bonus 'must greatly exceed discounted future per-step value' (T. rex behavior.toml:18, raptor strike.toml). On Brachiosaurus food_reach the discounted objective still prefers reaching, but undiscounted eval return ranks hovering near the food (about 1875) above every quick-test success (1002-1304). Every hunt node gates, replays and hands off robust_best_model, ranked on mean - 1.0 x std of return with no look at success, so these inversions decide which checkpoint the gate judges. Severity is high because two members are verified high.

**Fix.** Zero alive_bonus and cut posture/height in both Compsognathus behavior.toml files, raise target_reach_bonus well above per_step/(1-gamma) (the sibling hunts use about 10x), and add the arithmetic comment the other hunts carry, then re-measure the statue and a scripted reacher. For the Allosaurus walk, add a completion bonus or switch off success termination as the other walks do, and set gamma explicitly. On nodes that declare min_success_rate or task_success/v1, choose the handoff by success first and reward second. For food_reach, pay proximity only on approach or tighten head_proximity_max_dist.

**Already tracked.** no. D-B14 (BEHAVIOR_RECIPES_PLAN.md:1135) covers only the Compsognathus hunt gate kind. TRAINING_REVIEW.md Q3 diagnosed the same mechanism for the raptor only. KNOWN_ISSUES.md:1194-1197 proposes tightening head_proximity_max_dist for gradient focus, not because of the inversion. NEXT_STEPS.md:841 calls the robust-best early handoff a known limit, for stance only.

Findings: `compsognathus-hunt-success-terminal-unprofitable`, `compsognathus-target-reach-success-is-net-loss`, `allosaurus-walk-reward-penalizes-course-completion`, `fu-first-hunt-node-evidence-brachio-food-reach-hover-outearns-reach`, `fu-first-hunt-node-evidence-hunt-handoff-selection-penalizes-partial-success`

### 3.2 Section 8's evaluation cell shows and saves nothing, because the notebook never configures logging

*Medium · correctness (wrong or misleading output) · effort S · models: all*

Cell 24 runs evaluate() for 30 episodes (about 50 s in the Brachiosaurus and Dibothrosuchus quick runs), but evaluate() reports only through logger.info and returns None, and no notebook cell configures logging, so Jupyter drops the records and the cell printed only "Evaluating the 'stance' policy". It is the only place on the notebook path that computes gait symmetry, cost of transport and stride frequency, and the notebook passes report_metrics=False, so those advertised metrics never reach the user or the bundle. The same logging gap hides the passing gate line (PASS is INFO, FAIL is WARNING), the six-heading certificate summary, the handoff line and per-heading progress. Cell 24 also raises unless the target is certified, so it never runs after a halted gate; its default seed 2000 (against the bundle's SEED+3000) is intentional for paired CLI runs.

**Fix.** Install a notebook logging handler at INFO for environments.* in the infrastructure cell. Make evaluate() return its aggregate, and have cell 24 print a table and save it as a declared bundle artifact (or build the table from evaluation_selected.csv and NODE_RESULTS). Run eval_policy_quality (report_metrics=True) on the selected checkpoint, after fixing the gait and cost-of-transport math (theme gait-and-energy-metrics-wrong).

**Already tracked.** no. CLEANUP_PLAN_2026_09.md:1163-1164 and CONSOLIDATION_PLAN_2026_09.md:40 note report_metrics=False. KNOWN_ISSUES.md:2002-2004 covers only the CLI side. ROADMAP.md:81 plans to move print() to logging, which would hide even more.

Findings: `report-evaluate-silent`, `notebook-logging-hides-eval-and-certificates`, `cell24-eval-invisible-unsaved`, `deliverable-eval-not-persisted`, `notebook-path-drops-quality-metrics`

### 3.3 The zero-action ('statue') baseline is applied to the wrong node, and its alarms are wrong on statue-optimal or falling plants

*Medium · correctness (wrong or misleading output) · effort M · models: all (misleading alarms observed on brachio, dibotho, compso-a, allo)*

Notebook cell 13 hard-codes BASELINE_STAGE = 1 and read_zero_action_baseline takes no stage, so on every later node (recovery, walk, hunt) the BaselineProgress watch, progress.json zero_action_reward and the gate_progress.png 'zero action' line all use the stance statue. Allosaurus locomotion drew 'zero action 108' (the 1000-step stance statue) on a 3000-step node, and the Brachiosaurus food_reach watch compared hunt evaluations with the 1749.6 stance statue. On statue-optimal stances (Brachiosaurus returns capped at 1750 against a 1749.6 statue; Dibothrosuchus; the anatomical Compsognathus at 99.95%), the strict latest >= baseline comparison, made on a different seed panel, cannot be met by a statue-level policy, yet the watch prints 'worse-than-trivial local optimum ... failed its gate' next to a by-design PASS. The Allosaurus statue falls in about 0.9 s, but the watch calls it the reward optimum and the preflight labels it 'CHECK GATE' instead of CHECK PLANT.

**Fix.** Key the baseline file by stage and have the reader take the stage and refuse a mismatch, or skip the watch and chart line on nodes without a measured statue. Make BaselineProgress gate-kind aware: compare on the same eval seed stream with a tolerance, and say 'statue-level (expected)' on reward-rail stances. Reorder the preflight so n_standing == 0 reports CHECK PLANT, and give each verdict a next action.

**Already tracked.** no. The quadrupeds' statue-cleared stance is accepted by design (BEHAVIOR_RECIPES_PLAN.md:990-995). The only acknowledgement of the alarm is an informal 'ignore' instruction to operators.

Findings: `baseline-watch-uses-stance-statue-for-every-node`, `stage1-baseline-on-all-nodes`, `brachiosaurus-stance-reward-ceiling-no-headroom`, `compsognathus-baseline-warning-on-statue-optimal-stance`, `allosaurus-statue-baselines-vacuous-for-torque-plant`, `zero-action-preflight-verdicts`

### 3.4 QUICK_TEST only shortens training: it isn't a smoke test, it stops at different nodes per species, and its results look like real runs

*Medium · notebook UX / QoL · effort M · models: all*

QUICK_TEST changes only the per-node step budget (50,000) and the output root, so certificate panels (12 x 40 episodes), T. rex's ~350-episode probe battery, replays and the eval cadence run at full size: the raptor's certificate took about 14 minutes after training, and the raptor and T. rex quick runs were killed at 2400 s inside artifact generation. Where it stops depends on the species: Allosaurus and the Compsognathus robot fail stance, while the raptor (40/40), T. rex (38/40) and both quadrupeds (whose statue clears the gate) pass at 53,248 steps and continue downstream (the anatomical Compsognathus very likely passes too, inferred). In Colab with the defaults a gate failure calls halt(), which unassigns the runtime with a bare RuntimeError, and cells 24-32 usually never run. A quick test is recorded only in its directory name and training_summary.txt; the Brachiosaurus quick-test bundle verified as 'canonical-valid' and T. rex's 50k-step stance passed the same gate sha as its 12M-step certification, so promotion would accept it. CI runs the real train_stage plus generate_stage_artifacts path only on the Compsognathus pair (PR CI: robot PPO only).

**Fix.** Add QUICK_TEST='smoke' with a smoke_configs-style override (short horizon, small panels, no probe battery, short replays, scaled eval_freq) that continues past gate failures with 'pipeline OK through <node>' and never auto-disconnects. Write quick_test into provenance.json, gate_verdict.json and the CSV rows, and refuse quick tests at promotion and in the Drive strict cell. Run the notebook end-to-end smoke for all 7 models nightly.

**Already tracked.** partially. D-D16(b) (BEHAVIOR_RECIPES_PLAN.md:1302) only moves quick tests into their own directory. Notebook cell 5 says a quick test 'may stop at a gate'. The CI depth cut is documented (python-ci.yml:430-441), but the Compsognathus-only parametrization has no stated reason.

Findings: `quick-test-disconnects-colab`, `quick-test-cannot-smoke-the-notebook`, `quick-test-stops-at-first-gate`, `quick-test-does-not-exercise-report-path`, `compsognathus-quick-test-passes-stance-at-init`, `tyrannosaurus-trex-probe-battery-unscaled-undocumented`, `quick-test-indistinguishable`, `cmp-notebook-path-tested-on-compso-only`

### 3.5 Chains run past the ~24 h Colab cap with no time awareness, a manual two-pass resume, and no progress or notifications

*Medium · pipeline feature · effort L · models: all*

Default chains exceed one Colab session (T. rex 12M+8M+8M steps, raptor 12M+8M+12M, Brachiosaurus 6M+16M+12M, Dibothrosuchus 6M+12M+8M; a 12M-step T. rex stance took 23h07m-23h29m on an L4), yet the chain loop starts each node with no time check. The RUN_ID memo lives in kernel globals, so a fresh runtime mints a new run, and although the loop detects the interrupted node it only raises, leaving a ~50-line manual resume (RUN_ID, pinned TRUNK_FROM, RETRAIN_FROM, RESUME_STAGE) and a second Run all. Capped segments drop out of total_training_time. During a ~24 h node there is no usable inline progress or ETA (VERBOSE=1 also dumps SB3's logger table every iteration), no %tensorboard cell, nothing reads back progress.json's projected_finish, and nothing notifies the user when a node finishes, fails or crashes.

**Fix.** Accept RESUME_STAGE='auto' (adopt trunk_run.json and the interrupted node, keep the existing refusals), persist RUN_ID to Drive, and sum durations across resumed segments. Add a session-budget knob that refuses to start a node projected past the session end. Drive the progress bar from VERBOSE >= 1 with SB3 verbosity 0, and print a one-line progress.json readout per evaluation with projected finish against the session end. Add an optional %tensorboard cell and an optional notification hook (webhook/ntfy) on node end, halt or exception.

**Already tracked.** partially. D-D16(c) (BEHAVIOR_RECIPES_PLAN.md:1302) chose the manual RESUME knob. NEXT_STEPS.md:441,457 record the cap hits and lost durations as facts. KNOWN_ISSUES.md:1603 covers only the same-stage ramp loss.

Findings: `no-colab-cap-awareness-or-auto-resume`, `resume-is-manual-two-pass`, `no-progress-eta-or-tensorboard`, `fu-colab-runtime-specifics-no-unattended-run-signal-or-colab-tier-guidance`

### 3.6 In-training evaluation dominates wall-clock time, and the training device is neither chosen nor shown

*Medium · pipeline feature · effort M · models: all*

Every node evaluates serially on one eval env every 50k steps for max(30, min_eval_episodes) episodes (40 on five of the seven stances), about 0.8 serial env steps per training step on 1000-step stances, which pushes 12M-step stances to the ~24 h cap. The notebook hard-codes eval_freq=50_000, and the per-stage cadence key exists only on unmerged screen branches. Separately, the notebook metadata requests an L4 GPU, so PPO MlpPolicy trains on CUDA with no stated reason, no device knob and no printed device (VERBOSE=0 hides 'Using cuda device'), and SB3's train-on-CPU warning is skipped for the CriticPhase policy that the certified T. rex stances use.

**Fix.** Merge the per-stage eval cadence key (D-D36) and expose an EVAL_FREQ knob, or vectorize the eval env. Measure PPO throughput on Colab CPU against L4 CUDA, then pin the device with a knob that is printed at start and recorded in provenance.

**Already tracked.** partially. KNOWN_ISSUES.md:1191-1193 (MEDIUM perf) says to vectorize the eval env or trim episodes. The device is only inferred from the gpu block (TREX_STANCE_SYMMETRY_AND_FOOT_2026_10.md:2150-2156).

Findings: `eval-cadence-dominates-wallclock`, `fu-colab-runtime-specifics-ppo-device-unpinned-unrecorded-no-runtime-guidance`

### 3.7 Non-gate failures leave the Colab runtime assigned, and missing artifacts go silently unrepaired

*Medium · pipeline feature · effort M · models: all*

Only halt() on a failed gate and the last cell release the runtime; every other raise (a FAILED or refused verdict, an interrupted node, a bundle error, cell 24 on an uncertified target, cell 30's unguarded validate_result_bundle) leaves the GPU assigned until the idle timeout. The verdict is written before the figures, replays, probes and bundle, so if the runtime dies in between and the run is re-entered in place, the same-run reuse branch accepts the passed node without regenerating them, and cell 30 then stops Run all before the disconnect. Best-effort failures are only logged: one try covers all staged figures and one covers both replays (a failed selected replay skips the final one), and artifact_manifest.json has no expected list. The physics-only _stance.csv is written only after every frame renders, so a renderer error drops it too, and the setup cell writes the NVIDIA EGL file without checking for a GPU and prints 'EGL rendering enabled' without testing rendering.

**Fix.** When IN_COLAB and AUTO_DISCONNECT, arm an IPython custom exception handler in the chain loop that flushes Drive, writes the traceback to RUN_DIR and disconnects. Regenerate missing artifacts and the bundle in the reuse branch, and write an artifact_status.json of expected versus present files derived from gate kind, model and role. Give each replay its own try, write _stance.csv in a finally block, and do a one-frame renderer check in setup.

**Already tracked.** partially. KNOWN_ISSUES.md:680-713 covers only the re-entry case caught by the bundle check, with a manual remedy. NEXT_STEPS.md:435 records a past incident.

Findings: `disconnect-only-on-gate`, `exceptions-skip-auto-disconnect`, `verdict-without-bundle-unrecoverable`, `silent-artifact-failures`, `fu-colab-runtime-specifics-egl-backend-unprobed-replay-and-stance-csv-lost-together`

### 3.8 Hunt collapse backstops use a fixed floor below the do-nothing score and no warm-up; only the T. rex hunt was fixed

*Medium · reward/recipe design · effort S · models: raptor, brachio, dibotho (trex walk)*

Raptor strike, Brachiosaurus food_reach and Dibothrosuchus snap each set only collapse_peak_floor = 100 with no peak warm-up, so the defaults apply (12 evals, patience 8, drop 0.4, warm-up 0), although their zero-action statues score well above 100 (raptor strike about 265-283, Dibothrosuchus snap 496-1400). The backstop therefore arms at the 12th evaluation (600k steps), and the snap run can stop at about 950k, inside the stage's 1.0M warm-up plus ramp. Gap-review fix CF3 and D-B4/D-B5 fixed this pattern only for the T. rex hunt (relative floor 0.45 x 601.1, 1.0M warm-up, 20/10/0.5). T. rex locomotion is the only walk with a backstop and no peak warm-up, but its choice rests on a measured T. rex series, so it is an inconsistency to document rather than a clear bug.

**Fix.** Port the T. rex hunt pattern to the three hunts: a relative floor of 0.45 x the measured hunt statue, collapse_peak_warmup_timesteps = warm-up + ramp, and explicit min_evals, patience and drop. Extend the test that pins the D-B5 bound to every hunt, and document the T. rex walk exception.

**Already tracked.** partially. CF3 (RL_PIPELINE_GAP_REVIEW_2026_08.md:505-514) and D-B4/D-B5 fixed only the T. rex hunt. KNOWN_ISSUES.md:929-975 covers only the stance and walk floors.

Findings: `dibothrosuchus-snap-backstop-no-peak-warmup`, `velociraptor-strike-absolute-collapse-floor`, `cmp-hunt-backstop-brachio-same-pattern`, `tyrannosaurus-trex-walk-collapse-floor-without-peak-warmup`

### 3.9 Task success, what the hunt gates judge, is never charted against its bar and is mislabelled or missing on most species

*Medium · correctness (wrong or misleading output) · effort M · models: raptor, trex, brachio, dibotho, compso-a, compso-r, allo (walk)*

Per-evaluation hunt success is saved (evaluations.npz successes, gate_progress.npz termination_<cause>), but no chart draws it against min_success_rate, and per-evaluation forward velocity (the walk gate's criterion) is never saved or charted; the only speed chart is an unlabelled training-rollout 'Speed' panel with no bar. behavioral_metrics.png's 'Strike Success Rate' panel plots successes per training step, and only for strike_success and bite_success, so it is empty for Brachiosaurus, Dibothrosuchus, both Compsognathus models and Allosaurus; only locomotion_health.png's termination breakdown shows their successes, as a stage-mean share. Dibothrosuchus snap_success, reward_snap and reward_snout_proximity are missing from the shared REWARD_KEYS and INFO_KEYS, so they are not recorded at all. The training-history pages and progress.json count success terminations as falls and drop them from every per-episode statistic (a documented definition, but misleading on hunt nodes and the Allosaurus walk). Hunt evaluation CSVs have no termination-cause or start-distance column, hunt replays have no per-frame CSV or success marker, the raptor's claw and T. rex's head distances are emitted but never read, and 'jaw_distance # T-Rex' is emitted by no env.

**Fix.** Add a task panel to gate_progress (per-evaluation success against the bar, with a binomial LCB band), and save per-evaluation eval forward velocity. Read success keys from each env's SUCCESS_KEYS instead of hard-coded tuples, which adds the snap keys. Count successes separately from falls in training_history and progress.json, and add cause and start-distance columns to the evaluation CSVs.

**Already tracked.** partially. KNOWN_ISSUES.md:1178-1187 covers the hard-coded success tuples and the missing snap_success. TRAINING_ARTIFACT_PRESENTATION.md:855-862 and :1265 document the success-as-fall definition. KNOWN_ISSUES.md:1534-1550 covers a success on a fall step.

Findings: `hunt-success-rate-chart-wrong`, `fu-first-hunt-node-evidence-hunt-success-persisted-but-not-drawn`, `gate-criteria-not-charted`, `behavioral-metrics-chart`, `dibotho-keys-missing`, `success-terminations-counted-as-falls`, `fu-first-hunt-node-evidence-progress-json-counts-hunt-successes-as-falls`, `fu-first-hunt-node-evidence-hunt-success-evidence-no-step-or-cause`, `velociraptor-hunt-distance-pelvis-not-claw`

### 3.10 The gait and energy metrics are computed wrongly: cost of transport, symmetry, stride frequency and time-to-target

*Medium · correctness (wrong or misleading output) · effort M · models: all*

Both 'Cost of transport' readouts are built from |reward_energy| = energy_penalty_weight x mean(a^2), so they track a TOML weight (0.001 on the raptor, T. rex, Brachiosaurus and Dibothrosuchus final nodes, 0.02 on the Compsognathus hunts, the 0.03 default on the Allosaurus walk), not physical energy; identical motion scored 0.213 down to 0.0028 as the weight fell from 0.075 to 0.001. The evaluate() printout divides by a fixed 1 kg x g x distance and prints 0 when every episode is stationary; the locomotion_health.png panel divides by max(forward_vel, 0.01) instead and spikes for a stationary policy. The LocomotionMetrics gait block is also wrong: double-support walks read 0 Hz, onset counting counts both edges, symmetry ignores phase (a 0.7/0.3 limp scores 0.958) and quadrupeds read only the forefeet. 'Time to target / reached %' uses one 0.5 m threshold measured from a different reference point on each species and disagrees with task success for T. rex, Brachiosaurus and Compsognathus; the curriculum's supplementary log looks up mean_forward_velocity while aggregate_episodes emits mean_mean_*, and the eval report prints 0 for unmeasured metrics. Impact is limited because no gate or bundle reads these numbers and the logging gap hides them in the notebook; the verifier judged this not high.

**Fix.** Compute cost of transport from body_telemetry work_abs_j and the model's body weight (morphology body_weight_n), printing n/a when unmeasured. Compute stride frequency and symmetry per foot from contact onsets with phase, use each env's own success definition for 'reached', and fix the key names. Do this before surfacing the metrics in section 8.

**Already tracked.** partially. ROADMAP.md:155 notes that no phase difference is computed. KNOWN_ISSUES.md:984-1015 covers only the reward-side gait_symmetry. KNOWN_ISSUES.md:1894 covers only velocity_consistency.

Findings: `cot-not-physical`, `locomotionmetrics-gait-math`, `time-to-target-threshold`, `supplementary-log-key-mismatch`

### 3.11 The older per-node figures (behavioral metrics, locomotion health, reward decomposition, training curves) mislabel or hide data

*Medium · correctness (wrong or misleading output) · effort M · models: all (reward decomposition worst on allo)*

The Reward Decomposition panel picks terms from TOML weights alone, so it hides Allosaurus's dominant default-paid alive, posture and energy terms, and on every model it shows all-NaN ghost terms in an overflowing legend that reuses colours (T. rex: 30 entries). behavioral_metrics.png labels foot force 'stride freq' on the same axis as a 0-1 left/right balance line, draws the hunting row on every node because the key exists (empty Prey Distance and Strike panels with duplicated legends on stances), calls food 'Prey Distance', and lets its heading cos(theta) band exceed 1. locomotion_health.png's termination bars are unweighted mean fractions per rollout with no counts or time axis (the raptor's 5 falls in about 55 episodes read as nosedive 1.0), and its 'Cost of Transport' is not one. A one-evaluation series has no markers, so its panel looks empty in every quick run, and titles use legacy 'Stage N' numbers, repeat model and algorithm, and omit units.

**Fix.** Choose reward terms from the diagnostics data (finite and non-zero), fix labels, units and axes, and draw hunting panels only on task nodes. Plot termination counts over time, add markers, and title figures with node ids. Once gate_progress and training_history cover their content, consider retiring these figures.

**Already tracked.** partially. test_diagnostics.py:582 admits the five unmapped keys. TRAINING_ARTIFACT_PRESENTATION.md:864-870 says fractions alone mislead, but the plot still uses only fractions.

Findings: `reward-decomposition-selection`, `behavioral-metrics-semantics`, `behavioral-metrics-chart`, `locomotion-health-panels-misleading`, `single-point-series-invisible`, `figure-labels-units`

### 3.12 Stance exploration noise is calibrated on only some plants: Dibothrosuchus and the Compsognathus robot fall throughout training

*Medium · cross-species consistency · effort S · models: dibotho, compso-r (brachio and allo also run at the default std 1.0)*

Only the raptor (std 0.10) and the anatomical Compsognathus (0.135) start stance exploration at a measured noise level their statue survives; T. rex's 0.22 was tuned from reward cost and short PPO runs and sits above its statue's cliff (its falling rollouts are documented and accepted, D-D27). Dibothrosuchus trains at SB3's default std of about 1.0 with no recorded reason, and 0 of its 873 quick-test training rollouts reached the horizon (excessive_tilt 501, tail_contact 295). The Compsognathus robot copies its sibling's 0.135 with no measurement, although its statue already falls on every episode at sigma 0.02 (and stands at 0.01), and all 1109 quick rollouts fell; the audit note about this never became a test or issue. Brachiosaurus at std 1.0 kept 49 of 54 rollouts and is not materially affected, Allosaurus also trains at 1.0 (0/821 at full horizon), and the notebook shows none of the fall-fraction or std signals that training-history pages 01 and 10 and progress.json already hold.

**Fix.** Measure each plant's noise cliff with the existing noise test and set log_std_init, or record why std 1.0 is intended. Add the robot and the quadrupeds to the noise test, and print the training fall fraction and policy std in the notebook's per-evaluation readout.

**Already tracked.** partially. STANCE_HACK_AUDIT_2026_10.md:190 notes that the robot statue falls under any PPO-scale noise. Nothing covers Dibothrosuchus or Brachiosaurus, and GQ-10/PR-G8 plans their stance retrain without mentioning std.

Findings: `cmp-exploration-start-uncalibrated-quadrupeds-allosaurus`, `compsognathus-robot-falls-under-any-exploration-noise`

### 3.13 SAC is offered like PPO, but its resume, warm start, diagnostics and recipes behave differently and aren't flagged

*Medium · pipeline feature · effort M · models: all (when ALGORITHM=sac)*

ALGORITHM='sac' is free text with no caveat, yet a SAC resume from the RESUME cell is deliberately not a continuation (train_base.py:583), so post-resume checkpoints are pruned or misnamed and the best and robust-best handoffs and evaluation record are overwritten, without users being told. SAC warm starts begin with an empty replay buffer and a uniform-random learning_starts phase, although eight [sac] comments and RL_TRAINING_PLAN say the parent's transitions carry over. DiagnosticsCallback writes a row every train_freq vector step (every 64 transitions on raptor, T. rex and Dibothrosuchus, 16 on Brachiosaurus and Compsognathus, 4 on Allosaurus: about 750k rows over 3M steps) and rewrites the whole history to Drive every 60 s, while the std and K3 pages are empty with a wrong reason, alpha is never charted and progress.json has no policy block. Update-to-data comments are off by N_ENVS, several [sac] blocks wrongly claim to 'match PPO' (only the T. rex stance carries a staleness warning), T. rex recovery under SAC logs 'SB3 defaults' with an empty [sac] digest, CRITIC_PHASE prints 'on' but does nothing, and the notebook runs SAC on 4 envs where the CLI uses 8. In SAC's favour, a historical raptor SAC run exists, CI trains every chain path with SAC in tiny runs, and warm_start_log_std is refused under SAC.

**Fix.** Label SAC experimental in a cell-6 dropdown, refuse the documented PPO-only cases (CRITIC_PHASE, T. rex recovery), and document the resume and warm-start semantics in section 5. Aggregate diagnostics on a fixed step cadence with np.savez_compressed for both algorithms, and read SAC std from the actor's log_std head or mark those pages N/A. Correct the [sac] comments and set N_ENVS per algorithm.

**Already tracked.** partially. KNOWN_ISSUES.md:2332-2341 (NB9) covers 4 vs 8 envs. The SAC resume exclusion is deliberate and pinned (train_base.py:578-582; test_resume_continuation.py:251-253), but users are not told. T. rex stance.toml:548-553 carries the staleness warning, and ROADMAP.md:135 lists '[ ] Run Stages 1-3 with SAC'.

Findings: `sac-resume-not-a-continuation`, `sac-diagnostics-density`, `fu-sac-artifact-path-sac-warm-start-semantics-misdescribed`, `fu-sac-artifact-path-sac-exploration-unobservable`, `fu-sac-artifact-path-sac-update-to-data-ratio-misstated`, `fu-sac-artifact-path-sac-recipe-provenance-stale`, `fu-sac-artifact-path-trex-recovery-sac-recipe-unrecorded`, `sac-path-second-class`, `sac-path-second-class-unflagged`

### 3.14 No recipe-maturity or hold signal: every species' full chain is offered alike, including recipes the maintainers recommend holding

*Medium · cross-species consistency · effort S · models: all (brachio, dibotho, trex, compso-r, raptor, allo named)*

Neither stages.toml nor species_manifest.toml has a status or hold field, and neither the notebook's chain table nor the website's model pages surface a hold (the only hints are free text, such as the Compsognathus recovery stages' 'Pilot:' description and '# Experimental SB3 recipe' TOML comments), so BEHAVIOR='auto' trains every model's full chain the same way (for example Brachiosaurus's 6M+16M+12M hunt chain; its historical summary records 19h46m of training on an L4). Yet the open GAIT_QUALITY_PLAN GQ-3 recommends, and practice follows, holding walker sessions on today's rewards, the T. rex and robot hunts, the robot recovery and the Dibothrosuchus hunt. The raptor stance has never been trained on physics r5, and NEXT_STEPS prescribes a stance-only multi-seed protocol before any walk, which the default chain skips; Allosaurus trains on near-default tables. A user comparing species cannot tell which results to expect.

**Fix.** Add a per-node status field in stages.toml (tuned, pilot, or held with a reason and doc link), print it in the cell-9 chain table, and warn on or refuse held nodes unless an override is set. Mirror it on the catalog and model pages. Until the raptor's r5 stance is certified, default it to BEHAVIOR='stance' or print the NEXT_STEPS protocol.

**Already tracked.** partially. The holds exist only as prose: GAIT_QUALITY_PLAN_2026_09.md:185 (GQ-3, open), NEXT_STEPS.md section 3, :134-157 and :456. Showing them to users is not tracked.

Findings: `cmp-no-recipe-status-signal`, `brachiosaurus-chain-hold-not-surfaced-in-notebook`, `dibothrosuchus-recipe-on-hold-not-surfaced`, `velociraptor-pre-walk-protocol-not-surfaced`

### 3.15 Gate bars and success metrics mean different things across species, and some judge nothing

*Medium · cross-species consistency · effort M · models: all (allo, compso-a, compso-r, trex, brachio named)*

Walk velocity bars run from a Froude-matched target (raptor 2.1 m/s, Fr 0.82) down to floors that only rule out standing still (Compsognathus 0.08 and 0.04 m/s, Allosaurus 0.25 m/s, all at Fr of about 0.003 or below), and those three carry no rationale. T. rex's walk reward rail (100) was never re-derived as 0.6 x statue, so the 1092.7 statue clears it and speed is the only bar a statue fails. No gate reads Allosaurus's headline success metric, course_complete: its last node is reward_and_length/v1 with a 550-of-3000-step length floor and no min_success_rate, unlike every other model's last node, and nothing records why. Walk nodes record and publish incidental task success (Brachiosaurus's walk row reports 3.3% food_reached), and the Drive summary shows the final model's success rate and velocity next to bars that judge the selected checkpoint; the raw-mean success gate on non-T. rex hunts is the documented D-B14 decision.

**Fix.** Record each walk bar's dimensionless value and rationale in the TOML, the chain table and the catalog, labelling the three undocumented bars 'pilot bar' until derived. Re-derive the T. rex walk rail as 0.6 x statue, or state that it is intentionally vacuous until locomotion_gait/v1 (PR-G5). Gate Allosaurus course_complete or mark it report-only, blank success on nodes whose gate does not apply, and show selected-checkpoint statistics in the Drive table.

**Already tracked.** partially. KNOWN_ISSUES.md:1310-1327 (HIGH) says locomotion reward floors are 3-22x below the statue, with PR-G5 planned. GAIT_QUALITY_PLAN_2026_09.md:309-312 records the Froude numbers. D-B14 documents the raw-mean hunt gates.

Findings: `allosaurus-course-complete-ungated`, `cmp-walk-velocity-bars-not-comparable`, `tyrannosaurus-trex-walk-reward-rail-vacuous`, `brachiosaurus-walk-node-publishes-incidental-food-success`, `hunt-success-statistic`

### 3.16 Stance certification guards differ across species, and the documented manual checks aren't automated

*Medium · cross-species consistency · effort M · models: compso-a, raptor, trex, dibotho*

Of the three stances on the six-heading certificate, only T. rex and the raptor declare certification_seeds = 2, a 500k checkpoint archive and the hop watch; the anatomical Compsognathus has the same entropy schedule and foot-contact keys but none of these guards and no recorded reason. The robust-best handoff has no warm-up, and the notebook prints the mean-best step instead of the handoff step that NEXT_STEPS names as the guard for this known early-handoff limit; the quick raptor stance passed its full certificate at 50k steps with std still at its initial value, and a hop-watch-stopped stance still advances along the chain. The documented hand check of a Compsognathus stance (per-foot lift-offs, root-height rhythm) is never computed or prompted before locomotion warm-starts from it. The Dibothrosuchus stance still certifies on a reward rail (tracked HIGH) with no per-foot load screen, and its r3 quick-test handoff stood on three legs.

**Fix.** Add certification_seeds, the archive and the hop watch to the Compsognathus stance, or record why not. Print the handoff step and hop-watch status in the chain loop, and stop advancing after a hop-watch stop unless overridden. Compute per-foot lift-off counts and root-height rhythm as report-only certificate columns, and add a report-only per-foot load screen to the reward-rail stances until v2 is adopted (GQ-10).

**Already tracked.** partially. NEXT_STEPS.md:841 calls the early handoff a known limit with manual guards. hop_watch.py:20-23 judges a stopped run by design. KNOWN_ISSUES.md:1346-1390 (HIGH, GQ-10) covers Dibothrosuchus. KNOWN_ISSUES.md:1431-1470 and NEXT_STEPS.md:549-553 describe the Compsognathus check as a manual step only.

Findings: `cmp-certificate-stances-uneven-guards`, `velociraptor-stance-certificate-certifies-init-handoff`, `compsognathus-manual-liftoff-check-not-automated`, `dibothrosuchus-three-legged-stance-certified-unflagged-r3`

### 3.17 Allosaurus, the newest model and the only torque-driven one, is under-tuned, and much shared tooling breaks on it

*Medium · cross-species consistency · effort M · models: allo (v1 report literal also compso-r)*

Allosaurus (added 2026-11-21, the only torque-motor plant) trains on near-default PPO/SAC tables with the shortest stance budget, and the notebook does not mark it untuned. Tooling built for position servos and a standing statue misreports it: its statue falls by step 85-90, so the policy probe never builds a bank and records nothing (losing the bank-free std, normalizer and clip-share columns too), and the stance_quality/v1 report prints a hard-coded T. rex 'statue 0.998/0.002' support duty (shared with the Compsognathus robot, whose touch-only support lines also lack a floor-truth cross-check, tracked). It uses the v1 gate plus the report-only heading probe on a heading-free observation with no rationale, so the probe's 'statue falls' warning fires on every stance; its walk target is invisible (rgba 0 0 0 0) with no target-distance key; and joint_excursion_report prints its torques as degrees while actuator_saturation_report crashes on its MJCF. TRAINING_ARTIFACT_PRESENTATION.md never mentions it; its reward inversion, ungated course_complete, statue-baseline alarms, hidden reward terms and dense SAC diagnostics are covered in themes task-success-worth-less-than-failing, gate-bars-not-comparable, statue-baseline-misapplied, legacy-figures-misleading and sac-path-second-class.

**Fix.** Mark Allosaurus pilot/untuned (theme recipe-status-not-surfaced), and choose and document its stance gate kind. When the statue falls, build the probe bank from a reference stand (ReferenceWalk) or policy rollouts, always keep the bank-free columns, and replace hard-coded statue literals with measured per-model values. Add a visible target marker and a target-distance info key, make the offline scripts actuator-type aware, and add Allosaurus to the artifact docs.

**Already tracked.** partially. stance.toml:32 and locomotion.toml:20 document only the omitted collapse floor. KNOWN_ISSUES.md:1329-1344 and :1386-1390 cover the robot's touch and v1 parts. The Allosaurus-specific gaps are untracked.

Findings: `allosaurus-least-instrumented`, `policy-probe-never-builds-allosaurus`, `cmp-allosaurus-heading-probe-on-heading-free-plant`, `allosaurus-v1-report-hardcoded-statue-duty`, `compsognathus-v1-report-touch-bilateral-and-hardcoded-statue`, `allosaurus-walk-target-invisible-in-artifacts`, `fu-offline-analysis-tools-species-coverage-allosaurus-torque-plant-in-servo-reports`, `allosaurus-missing-from-artifact-docs`

### 3.18 Replays don't show the task, and walk and hunt nodes get thinner artifacts than stance nodes

*Medium · missing artifact or analytics · effort M · models: raptor, trex, dibotho, compso-a, compso-r, allo (coverage matrix: all)*

Hunt replays leave the prey or target off-screen: rendering through each env's replay camera over a 5 x 5 grid of spawn positions showed the prey at 0/25 points for the raptor, T. rex and Dibothrosuchus, 7/25 and 3/25 for the anatomical and robot Compsognathus, and 25/25 only for Brachiosaurus, the one hunt with a task camera; the Allosaurus walk target is invisible. One stance/recovery-only flag gates both the per-step _stance.csv and the front and side views, so walk and hunt nodes get two default-view mp4s and no deterministic per-step actuator record for the selected checkpoint (per-rollout actuator series do exist in policy_probe.npz). Which diagnostics a node gets (stance report, handoff sidecar, stance_diagnostics.png, probe battery, heading probe) depends on gate kind and species in ways the docs don't explain.

**Fix.** Give every task env a camera that frames both the body and the target (as Brachiosaurus does), and add a visible target marker. Write the per-step CSV for every node, decoupled from the extra camera views. Publish an expected-artifact matrix (gate kind x model x role) in the docs and in artifact_status.

**Already tracked.** partially. RECOMMENDATIONS.md:63-97 proposes a per-joint action envelope artifact. TRAINING_ARTIFACT_PRESENTATION.md:1615-1623 documents the handoff sidecar as stance-only. Three-view replays only on stance and recovery nodes is the same on every model.

Findings: `fu-first-hunt-node-evidence-hunt-replay-prey-out-of-frame`, `fu-offline-analysis-tools-species-coverage-walk-hunt-nodes-no-per-actuator-record`, `artifact-coverage-matrix-uneven`, `allosaurus-walk-target-invisible-in-artifacts`

### 3.19 Run and stage summaries leave out verdicts, failure reasons and certified statistics, and there is no chain-level report

*Medium · missing artifact or analytics · effort S · models: all*

write_training_summary prints per-node timesteps, duration, evaluations and a 'Best model:' path even for a FAIL, but no gate kind, verdict, failure reasons, reuse origin, certified statistic or selected-checkpoint evaluation; the certified '40/40 LCB95 0.9278' appears only in stance_heading_report.txt. gate_reason, gate_evaluable and six other retired CSV_METRIC_COLUMNS have no writer and stay empty (including gate_reason on the Allosaurus FAIL), and the T. rex hunt LCB reaches summary.json but not the CSV. Stance and walk nodes print 'Success rate: 0%' in stage_summary and the chain loop while the eval callback says N/A. There is no chain-level human report or index page, and key report paths are never printed.

**Fix.** Add a per-node verdict block (gate kind, PASS/FAIL, reasons, headline statistic, reused-from, handoff step) to training_summary.txt and to the notebook's per-node print. Write gate_reason, drop the dead columns, and print success only where the gate applies. Write a run-level index.html linking each node's gate_progress, training_history.html, replays and reports.

**Already tracked.** partially. D-B15 (BEHAVIOR_RECIPES_PLAN.md:1136) defers stance and recovery statistics in summary rows. KNOWN_ISSUES.md:979 mentions the empty gate_reason in passing.

Findings: `no-chain-level-human-report`, `summary-omits-verdicts`, `gate-reasons-not-propagated`, `collected-results-dead-and-kind-dependent-columns`, `success-rate-on-non-task-nodes`

### 3.20 The Drive summary hides runs that are mid-chain and has no certification board across models

*Medium · publication & docs · effort M · models: all*

Once node 1 certifies and writes its partial bundle, node 2's first file makes the run audit as canonical-conflict, and the Drive summary drops the whole run, certified node included, from every table and runs_summary.csv for as long as later nodes train or stay interrupted (quarantine before node 1 certifies is intended, CU-15). Beyond that, it stops at PASS/FAIL booleans: no cross-model certification board, no displayed gate statistics (though gate_kind and the six-heading columns are already in its data), and no replication n/N, provisional status, lineage, plant revision or links to training_history.html or the replays.

**Fix.** Treat 'canonical-partial plus a node in progress' as a valid state, and show the certified rows with an in-progress marker. Add a 'Certification status' section with one row per model and deliverable (select_trunk's choice, gate kind, headline statistic, replication, reused-from, plant revision, artifact paths), written to logs/certification_status.csv.

**Already tracked.** partially. CLEANUP_PLAN_2026_09.md:831 (CU-15) and test_drive_summary_notebook.py:299-303 pin only the first-node quarantine. D-A9/D-B15 defer the headline export. A leaderboard exists only as a ROADMAP idea.

Findings: `drive-summary-quarantines-interrupted-runs`, `drive-summary-no-certification-board`

### 3.21 Getting results onto results/ and the website is manual, and nothing marks a result current or stale

*Medium · publication & docs · effort L · models: all*

Promotion from Drive into results/ and onto the website is manual, with no tool or checklist, and results/ holds only pre-bundle (schema-2) summaries for 3 of the 7 models. No code path marks a result current or verified (provenance hard-codes historical/unverified), and nothing checks published results against the current plant, task or gate. results/ allows one summary per (model, algorithm) and the catalog drops ancestors (D-B10), so a TRUNK_FROM chain or a set of seed replicates cannot be published whole, and no committed cell re-publishes a bundle although save_run_bundle's docstring says 'Re-run this cell'. results/README.md still says summary.json is written 'after all three stages', the Drive strict-promotion cell requires complete bundles while the catalog publishes partial ones, and website presentation differs by model (stage videos only for the raptor, T. rex and Brachiosaurus; GIFs only for the raptor).

**Fix.** Build a promotion tool and checklist on promote_result_bundle that refuses quick tests, compares plant, task and gate digests with the current manifest, and sets the verification fields. Allow a run-id or ancestor directory under results/, and add a republish cell. Fix the README, align the strict cell with require_publishable, and generate each model page's media section from the catalog.

**Already tracked.** partially. NEXT_STEPS.md:404-413 and RESULT_BUNDLES.md:328-335 record hand-typed re-publication. results/README.md:74 says promotion is deliberate. D-B10 forbids aggregating across bundles.

Findings: `no-promotion-path`, `no-staleness-or-current-status`, `results-layout-cannot-publish-trunked-chain`, `no-republish-cell`, `results-readme-and-strict-cell-drift`, `presentation-inconsistent-across-models`

### 3.22 The notebook embeds every replay twice and re-plots the old figures, but never shows the newer gate and history reports

*Medium · notebook UX / QoL · effort S · models: all*

display_stage_videos base64-embeds every mp4 with autoplay; stance and recovery nodes have six clips each (3 views x final and selected), and cell 28 re-embeds every chain node already shown in cell 20. The Allosaurus executed notebook held about 4.5 MB of video for roughly 155-step episodes against a 110 KB source, which scales to about 27 MB of base64 per 1000-step stance or recovery node. Cell 26 re-plots the two older figures already drawn in cell 20, while the newer gate_progress.png, foot_contacts.png and training_history.html are never shown inline (documented as intended at TRAINING_ARTIFACT_PRESENTATION.md:1612-1613).

**Fix.** Inline only the selected default-view replay (behind a SHOW_VIDEOS knob) and print the other paths, and in cell 28 show only reused ancestors. Display the saved gate_progress.png and link or IFrame training_history.html instead of re-plotting.

**Already tracked.** no. Leaving the training-history pages out of the notebook is documented as intended.

Findings: `embedded-videos-bloat-notebook`, `inline-video-bloat`, `notebook-inline-figures`

### 3.23 The training-history report uses T. rex defaults on every species and blanks pages without saying why

*Low · cross-species consistency · effort M · models: all (worst on allo, dibotho, compso-a, compso-r)*

All 20 pages are drawn on every node using T. rex-derived constants: the policy-std page's 'Legs and tail_3' panel and the checkpoint std columns use KEY_ACTUATORS (hip, knee, ankle, tail_3), dropping the Compsognathus and Allosaurus toe servos, and the exploration page draws T. rex's 'stance3' 0.08 std reference on every model. On Dibothrosuchus the 09_asymmetry page plots only the front-leg pair for hip pitch, knee and ankle with no pair named, while the rear pair carries about 73% of standing load. Allosaurus falls into the documented no-recorder gap without being named (about 14 of 20 pages are placeholders, as on the raptor and Brachiosaurus by design under D-D46), and it also lacks heading and spin keys and leg-DC pairs. The docs are stale: the raptor has had a head-tip site and neck hinges since physics r5, which removes D-D46's stated reason for it, and Dibothrosuchus feet have recorded a digit share since r3.

**Fix.** Derive key actuators per model, name and plot every left/right pair (or the load-bearing one), and draw the exploration reference only where it applies. Collapse all-placeholder pages into one 'not recorded for this model' line. Update the doc tables (add Allosaurus) and revisit D-D46 for the raptor.

**Already tracked.** partially. D-D46 (BEHAVIOR_RECIPES_PLAN.md:1331; TRAINING_ARTIFACT_PRESENTATION.md:684-693) keeps the raptor and Brachiosaurus gap on purpose. The first-pair choice is documented for hip roll only (TRAINING_ARTIFACT_PRESENTATION.md:1309).

Findings: `compsognathus-history-key-actuators-miss-toe`, `cmp-exploration-page-trex-reference-everywhere`, `dibothrosuchus-asymmetry-page-forelimbs-only`, `training-history-pages-not-model-or-role-aware`, `telemetry-ledger-box-sole-only`, `velociraptor-recorder-gap-premise-stale`, `dibothrosuchus-artifact-doc-digit-share-stale`

### 3.24 Recovery nodes record no statistics, and T. rex silently uses a stale judge

*Low · missing artifact or analytics · effort M · models: trex, compso-a, compso-r*

Only T. rex and the Compsognathus pair have a recovery node (by design), and a passing recovery verdict records none of the statistics it was judged on (the success and paired-delta LCBs appear only in failure strings), unlike stance and hunt, with no figure of the recovery panel. The notebook freezes the T. rex recovery gate on the stale r7 legacy judge (no calibration profile; the r7 safe set on the r11 plant) with no warning or verdict flag, although code and design docs say it must be re-measured first. The Compsognathus pair's 95% quiet safe-set prerequisite is measured only when the recovery node starts, never at the stance handoff (the robot's manual run was refused at 68%), and recalibration is CLI-only. No CLI tool reports on a recovery checkpoint, and T. rex recovery.toml documents a 2.0 s / 1.0 s window that is 1.0 s / 0.5 s at 100 Hz.

**Fix.** Save the recovery statistics in gate_verdict, stage_results and the CSV, and add a recovery panel to gate_progress. Mark the T. rex verdict 'legacy judge, unmeasured on r11' and warn at freeze, or refuse until it is re-measured. Compute safe-set occupancy at the end of stance for recovery-parent models, and fix the comments.

**Already tracked.** partially. The re-measurement is planned in BEHAVIOR_RECIPES_PLAN.md:1310 (D-D24) and HEADING_INVARIANCE_DESIGN_2026_10.md:321-322. D-A9/D-B15 defer only the summary export. RECOVERY_CALIBRATION.md:268-273 documents the check.

Findings: `tyrannosaurus-recovery-verdict-records-no-statistic`, `tyrannosaurus-trex-recovery-legacy-judge-auto-frozen-stale`, `cmp-recovery-parent-qualification-asymmetric`, `fu-offline-analysis-tools-species-coverage-recovery-deliverable-no-offline-analysis`, `tyrannosaurus-trex-recovery-timing-and-stale-comments`

### 3.25 The CLI and the notebook certify differently, and provenance misdescribes the recipe and evaluations

*Low · correctness (wrong or misleading output) · effort M · models: all*

The CLI's in-training verdict is bound to robust_best_model, chosen on reward alone, so a CLI 'passed' can certify a checkpoint that never met the gate (documented only for the hunt), and the CLI lacks the baseline watch. For reward_and_length/v1 and stance_quality/v1 rows, the catalog's 'early-advancement gate' column and the manifest definitions describe the CLI's consecutive-pass semantics, while the notebook certifies once on a 30-episode post-stage panel. N_ENVS changes PPO's rollout batch but is outside the hyperparameters digest, so replicates, reuse and resume treat different N_ENVS as one recipe. provenance.json stamps every evaluation role with 30 episodes while in-training evaluation ran 40 on five stances, and checkpoint-selection evaluations are unpaired.

**Fix.** Re-judge the CLI handoff on the gate panel as the notebook does, or label CLI verdicts training-time. Make the catalog and manifest text depend on gate kind and path. Add n_envs to the digest or refuse a mismatched resume, and record episode counts and seeds per role in provenance.

**Already tracked.** partially. KNOWN_ISSUES.md:884 calls the CLI hunt verdict training-time, :2003 covers CLI artifact parity, and :2331-2340 notes that n_envs is recorded but never flagged.

Findings: `cli-notebook-judging-parity`, `cmp-catalog-gate-semantics-by-kind`, `n-envs-outside-recipe-digest`, `eval-protocol-provenance-mismatch`

### 3.26 The offline analysis scripts are hidden from users and break on several species and roles

*Low · missing artifact or analytics · effort M · models: all (worst on allo, compso-r; stage-number mismatch on trex, compso-a, compso-r)*

Section 8 points to no offline analysis tool, and action_bound_report, observation_ablation_report and joint_excursion_report are documented only in plans and investigations. The scripts accept only legacy stage numbers 1-3, which differ from the NN_ directory prefixes on T. rex and both Compsognathus models, cannot name recovery, and never check the checkpoint's recorded task. action_bound_report --model labels SB3's already-clipped predict() output 'Raw (pre-clip)', so its overshoot lines always read in-bound. joint_excursion_report prints Allosaurus torques as degrees, actuator_saturation_report crashes on the Allosaurus and Compsognathus robot MJCFs and hard-codes three models, observation ablation covers T. rex only, and the robot's scripted gait doesn't recognise its left_/right_ names.

**Fix.** Accept node ids and resolve them through the manifest and task fingerprint. Fix the pre-clip label by reading the distribution mean, load MJCFs from their path, and make the reports actuator-type aware. List the tools with one-line usage in section 8 and on a docs page.

**Already tracked.** partially. SIM_TO_REAL_PLAN.md:322-325 and GAIT_QUALITY_PLAN_2026_09.md:471 plan the robot torque table. The backfill refusal is D-B6, and the T. rex-only scope of recovery_offdist_panel is intentional.

Findings: `fu-offline-analysis-tools-species-coverage-offline-tools-not-surfaced-in-report-section`, `fu-offline-analysis-tools-species-coverage-report-scripts-legacy-stage-int-unchecked`, `fu-offline-analysis-tools-species-coverage-action-bound-model-mode-measures-clipped-actions`, `fu-offline-analysis-tools-species-coverage-allosaurus-torque-plant-in-servo-reports`, `fu-offline-analysis-tools-species-coverage-robot-hardware-tools-misdrive-or-skip`, `fu-offline-analysis-tools-species-coverage-recovery-deliverable-no-offline-analysis`

### 3.27 No whole-chain or cross-run view, no TensorBoard cell, and no replicate launcher

*Low · pipeline feature · effort M · models: all (replicates: trex, raptor)*

No chart covers a whole chain on a cumulative step axis (each node's counter restarts) or compares runs and seeds, and the Drive summary shows only tables. PPO and SAC optimizer metrics reach TensorBoard in every node directory, but the notebook never opens TensorBoard. No launcher runs seed replicates, although certification_seeds = 2 on T. rex and the raptor means two manual runs (sweeps were retired on purpose by D-D17), and compare_run_diagnostics.py exists but nothing points users to it and it lacks the SAC keys.

**Fix.** Add an optional %tensorboard cell, point section 8 to compare_run_diagnostics.py and the plot_training_history overlay, and add the SAC keys. Optionally write a run-level chain_overview.png, and add a small launcher that runs N seeds of one node for models with certification_seeds > 1.

**Already tracked.** partially. ROADMAP.md:333-336 says sweeps need a new decision under D-D17.

Findings: `no-chain-or-optimizer-overview`, `no-replicate-or-sweep-orchestration`

### 3.28 Knobs are dense, the dropdowns allow invalid choices, and validation runs after the run directory is written

*Low · notebook UX / QoL · effort S · models: all (CRITIC_PHASE: raptor, brachio, dibotho, compso-r, allo)*

Cell 6 has 18 knobs with long decision-ID comments; ALGORITHM, QUICK_TEST and AUTO_DISCONNECT have no @param, the SPECIES x EMBODIMENT dropdowns allow 12 pairs of which 7 are valid (the rest are refused safely), BEHAVIOR offers recover and hunt everywhere, and cells 12-13 still say 'all six'. An invalid BEHAVIOR, RETRAIN_FROM or pinned TRUNK_FROM is refused only after the storage cell has created the run directory and provenance.json (a corrected Run all in the same runtime reuses it, so a stray is left only after a restart). CRITIC_PHASE=True fails, after the run directory and baseline are written, on the raptor (the default model), Brachiosaurus, Dibothrosuchus, the Compsognathus robot and Allosaurus, because only the T. rex and anatomical Compsognathus envs expose stance_width_settle_steps.

**Fix.** Validate every knob (SPECIES, EMBODIMENT, BEHAVIOR, RETRAIN_FROM, TRUNK_FROM, CRITIC_PHASE) in one call before cell 8 writes anything. Use one MODEL dropdown of the 7 valid ids, add @param to the remaining knobs, move the decision prose into a collapsed advanced cell, and fix 'all six'.

**Already tracked.** no. D-D15 (BEHAVIOR_RECIPES_PLAN.md:1301) dropped the broader reorder.

Findings: `knob-density-and-dropdowns`, `knobs-validated-after-write`, `velociraptor-critic-phase-unsupported-on-raptor`

### 3.29 Spurious warnings and legacy 'Stage N' wording in console output

*Low · notebook UX / QoL · effort S · models: raptor, compso-a, compso-r, allo (warning); all (wording)*

Stance nodes that write a stance report, except T. rex's, log a spurious 'stance_probe_filter_hz entry is not a number: None' warning (raptor, both Compsognathus models and Allosaurus; observed on Allosaurus), although the probe is correctly skipped; the two duplicate findings disagree on the affected models, and the verifier's corrected list is the one given here. User-facing text uses legacy 'Stage N' numbers ('Stage 1 curriculum gate', 'Stage recovery: Recovery', 'not an active Stage 1 gate') while directories use NN_<id>, and on T. rex and Compsognathus (02_recovery, 03_locomotion) the two numberings disagree.

**Fix.** Treat None as unset in _probe_cutoffs. Show node ids in all user-facing text, with the legacy number only in parentheses.

**Already tracked.** partially. Legacy numbering and file prefixes are deliberate (stage_manifest.py:609-611). The display wording is untracked.

Findings: `probe-cutoffs-none-warning`, `probe-filter-none-warning`, `stage-number-wording`

### 3.30 Config comments and docs state values, protocols and statuses that no longer hold

*Low · publication & docs · effort S · models: brachio, compso-a, compso-r, trex, raptor, allo*

Brachiosaurus TOMLs cite old statue values, a settled stance height of about 1.13 m (r6 settles at about 1.22 m) and 'Match PPO' SAC settings that don't match PPO. The hunt ramp comments call ramp_start_value a fraction, but it is an absolute forward_vel_weight, so Brachiosaurus's walk-to-hunt drop from 4.4 to 0.2 is immediate and the 'gradual transition' ramp only moves 0.2 to 0.3. TRAINING_RECIPE_REVIEW.md, cited by every anatomical Compsognathus TOML as the next-run protocol, predates the six-heading certificate and the recover recipe, and ROADMAP says T. rex, the raptor and both Compsognathus models have certified stance and walk nodes although D-D24, D-D26 and D-D33 stranded them. The installation, quick-start and Compsognathus pages still say BEHAVIOR defaults to 'hunt', and quick-start omits allosaurus_fragilis.

**Fix.** Fix the ramp comments, or make ramp_start_value the fraction they describe. Update the stale statue and height comments, TRAINING_RECIPE_REVIEW.md, ROADMAP and the getting-started pages. Add a doc-lint test that generates model lists from species_manifest and flags 'BEHAVIOR = "hunt"'.

**Already tracked.** partially. NEXT_STEPS.md:323-325 carries a table-wide D-D33 stranding note. KNOWN_ISSUES.md:2310 covers only the Brachiosaurus natural_pitch comment.

Findings: `brachiosaurus-hunt-forward-ramp-misdescribed`, `brachiosaurus-stale-recipe-comments`, `compsognathus-recipe-review-protocol-stale`, `compsognathus-robot-stranded-chain-still-listed-certified`, `cmp-stale-hunt-default-docs`

## 4. Per-species verdicts

| Model | Verdict |
|---|---|
| **Velociraptor** (anatomical) | Since the 2026-11-06 port it is broadly consistent with T. rex: the same six-heading certificate and guards (certification_seeds 2, hop watch, 500k archive) and a measured std of 0.10. Its problems are downstream:<br>• the strike node keeps an absolute collapse floor of 100, below its ~265-283 statue, with no warm-up<br>• hunt replays start with the prey out of frame (0/25 spawn points) and show it only in the last metre or two of the approach<br>• hunt distances are pelvis-based<br>It is the default model, so a plain Run all chains its never-trained r5 stance straight into walk and strike, skipping the documented pre-walk protocol. `CRITIC_PHASE` fails on it. Its quick stance passed the full certificate at 50k steps, and the notebook does not flag that. |
| **T. rex** (anatomical) | The most tuned and most instrumented model. Its constants (`KEY_ACTUATORS`, the 0.08 exploration reference, the `statue 0.998` literal) have become shared defaults that other species inherit by accident. Its own gaps:<br>• **Recovery:** the gate freezes on the stale r7 legacy judge without warning, a passing verdict records no statistic, and the window comments are off by 2x. In the contended CPU quick run (an observation, not a verified finding), its training ran at ~9 env steps/s against 30-45 for the other nodes.<br>• **Walk:** the reward rail (100) is vacuous next to its 1092.7 statue, and the backstop has no peak warm-up (backed by a measured T. rex series, so an undocumented inconsistency with the other walks rather than a clear bug; section 5).<br>• **Stance probes:** the four-probe battery runs at full size even in QUICK_TEST and has no reader documentation.<br>It is the only hunt judged by `task_success/v1` with an LCB, which is by design. |
| **Brachiosaurus** (anatomical, quadruped) | By design it keeps a reward-rail stance that the statue clears (its quick test PASSED at 50k steps), and it has no evaluation recorder (D-D46). Its food_reach is the one hunt whose replay frames the target. It has the clearest selection inversion: standing near the food out-earns reaching it in undiscounted return, so reward-ranked handoffs and curves structurally favour slow reaches or none. PPO's discounted objective (γ 0.995) still prefers reaching, and no mis-selection has been observed yet (the one full run reached 100% success). Other problems:<br>• it shares the floor-100, no-warm-up hunt backstop<br>• the baseline watch calls its statue-level stance "worse-than-trivial"<br>• its walk row publishes an incidental 3.3% food success<br>• its TOML comments (ramp, statue, height, SAC) are stale<br>• its 34M-step chain is offered with no sign that GQ-3 recommends holding it |
| **Dibothrosuchus** (anatomical, quadruped) | Its gate design matches Brachiosaurus (reward-rail stance, by design), but its recipe and telemetry are the least consistent of the quadrupeds:<br>• its stance trains at SB3's default std ~1.0 with no recorded reason (0/873 quick rollouts reached the horizon)<br>• its snap backstop floor (100) sits 5-14x below a do-nothing policy, with no warm-up<br>• `snap_success` and the snap reward terms are missing from the shared diagnostics keys, so its training-rollout snap success is never saved to `diagnostics.npz`, and its evaluation success, though saved for the gate, is never charted<br>• its stance can certify three-legged (tracked HIGH)<br>• the asymmetry page plots only the front legs, although the rear pair carries ~73% of standing load<br>`results/` holds no run for it. |
| **Compsognathus** (anatomical) | It has the most heavily measured stance gate, and with the robot it is the only model whose real notebook training-and-artifact path runs in CI. Its hunt differs from the other four in a way that matters: target_reach pays ~0.85 per standing step against a 25-point success bonus, so the deliverable is predicted to be unlearnable. Other problems:<br>• it is on the same six-heading certificate as T. rex and the raptor but lacks their hop watch, archive and two-seed bar, with no recorded reason<br>• the documented manual lift-off check is never automated<br>• its cited recipe-review protocol predates the certificate<br>• the T. rex-centric std panel drops its toe servos |
| **Compsognathus** (robot) | It shares its sibling's inverted target_reach reward. In its quick-test hunt, all 30 selected-checkpoint evaluation episodes ran the full 1000 steps with 0% success and scored 822 ± 40, close to the ~845 standing reward. That is a 50k-step policy, so it is what any untrained hunt would show: consistent with the predicted inversion, not evidence of it. Its copied recipe does not fit the plant: its statue falls under σ 0.02 noise while the recipe starts at 0.135, and all 1109 quick-test training rollouts fell. It stays on the v1 touch-duty gate by design, so its report sits next to a hard-coded T. rex statue literal with no floor-truth cross-check (tracked). Other problems:<br>• the recovery parent check (68% against a 95% safe-set prerequisite) is only discovered when the recovery node starts<br>• the sim-to-real analysis scripts crash on its mesh paths<br>• ROADMAP still lists its stranded nodes as certified |
| **Allosaurus** (anatomical, torque-driven) | The least consistent model. It is the newest (2026-11-21) and the only torque-driven plant, and it trains on near-default PPO/SAC tables that the notebook does not mark as untuned. Its walk reward penalises finishing the course, and no gate reads its headline success metric, `course_complete`. Tooling built for position servos and a standing statue misreports it:<br>• the policy probe never builds<br>• the preflight and baseline watch misdiagnose its falling statue<br>• the v1 report prints T. rex statue literals<br>• the reward-decomposition panel hides its dominant terms<br>• offline reports print torques as degrees<br>Its walk target is invisible in replays, its v1-plus-heading-probe gate choice is undocumented, and the artifact docs omit it. |

## 5. Differences that are deliberate (and fine)

These came up during the review but are documented decisions. The issue in some of them is only that the user is never told. Three are deliberate, but this review still recommends changing them, so read them alongside the fixes: the manual RESUME knob (fix 9 adds an `auto` mode), success terminations counted as falls (fix 4 counts them separately), and the per-frame CSV written only on stance and recovery nodes (fix 10 writes it for every node). For the SAC resume, what users are not told is that it overwrites the best and robust-best handoffs (theme 3.13).

- **Stance gate kinds.**
  - The heading-free T. rex, the anatomical Compsognathus and the raptor certify on the six-heading certificate (D-D32).
  - Brachiosaurus and Dibothrosuchus keep `reward_and_length/v1`, labelled a "stance reward rail" because the statue clears it (BEHAVIOR_RECIPES_PLAN.md:990-995).
  - The Compsognathus robot stays on v1 with a report-only heading probe, because its observation reads world heading.
  - *The Allosaurus v1 choice is not documented.*
- **Hunt gates.** Only T. rex uses `task_success/v1` (LCB ≥ 0.5). The other hunts keep a raw-mean success rate (D-B14, D-B15). Success definitions differ by task, per `species_manifest.toml`.
- **Recovery nodes** exist only on T. rex and the Compsognathus pair (D-D35), and their thresholds differ by calibration. The robot's refusal below 95% quiet safe-set occupancy is the designed prerequisite.
- **Allosaurus ends at walk**, so `BEHAVIOR='auto'` resolves to walk.
- **Collapse floors.**
  - The Compsognathus pair and Allosaurus omit them on purpose.
  - The stance and walk floors on the quadrupeds are a known open item.
  - T. rex locomotion's lack of a peak warm-up rests on a measured series.
- **Stance guards.** The certification_seeds, hop watch, hop-or-fall rail and 500k archive on T. rex and the raptor are documented. *Their absence on the anatomical Compsognathus is not.*
- **T. rex-only recipe choices.** The probe battery, mirror-KL PPO, the 10 Hz action low-pass and `stance_diagnostics.png` are documented T. rex choices.
- **No evaluation recorder on the raptor and Brachiosaurus** (D-D46). *The raptor's stated reason is now stale.*
- **Front/side replay views and per-frame CSVs** on stance and recovery nodes only, the same on every model.
- **Evaluation size follows the gate:** 40 episodes on 40-seed stance panels, 30 elsewhere.
- **Stance and recovery statistics are not exported** to summary rows (D-A9, D-B15); the hunt LCB is the exception.
- **Legacy stage numbers and `NN_` prefixes** are deliberate; only the user-facing wording is at issue.
- **Success terminations count as "falls"** in the training-history outcome ledger. This is a documented definition, but misleading on task nodes.
- **SAC resume is not a continuation** because the replay buffer is not saved. This is deliberate and pinned by a test, but the notebook does not tell users.
- **Process choices:**
  - W&B removed (ROADMAP.md:147)
  - sweeps retired (D-D17)
  - manual RESUME knob (D-D16(c))
  - quarantine of runs interrupted before node 1 certifies (CU-15)
  - no cross-bundle aggregation (D-B10)
  - `evaluate()`'s fixed seed 2000 for paired CLI runs

## 6. What the notebook runs showed

The real notebook ran headlessly with `QUICK_TEST=True`, `N_ENVS=2`, PPO and CPU, under heavy contention. Downstream nodes ran through the MANUAL cell.

**Stance, through the chain loop**

| Model | Gate kind | Quick-test outcome |
|---|---|---|
| Allosaurus | stance_quality/v1 | FAIL in 195 s (no full-horizon episodes). Policy-probe state-bank traceback and spurious `stance_probe_filter_hz ... None` warning. Sections 8-9 never ran. |
| Brachiosaurus | reward_and_length/v1 | **PASS**: the statue clears the gate, and the preflight itself prints "FAILS — a statue clears this gate". The whole notebook ran in 693 s. Cell 24 ran 48 s and printed only its header. Bundle verified **canonical-valid**. Six replays embedded in the chain loop and again in cell 28. |
| Dibothrosuchus | reward_and_length/v1 | Same as Brachiosaurus (PASS by statue, 884 s, silent cell 24, canonical-valid). 0/873 training rollouts reached the horizon. |
| Velociraptor | stance_quality_heading/v1 | PASS (40/40). The certificate took ~14 min, and the run was killed at 2400 s while recording replays. |
| T. rex | stance_quality_heading/v1 | PASS (38/40, LCB 0.85, same `gate_sha256` as the 12M-step certification). Killed at 2400 s inside the probe battery. |
| Compsognathus (anatomical) | stance_quality_heading/v1 | Killed by the batch time limit before its gate verdict; models saved, no figures. |
| Compsognathus (robot) | stance_quality/v1 | FAIL (`unsupported_duty_ucb 0.023 > 0.02`), then killed at 5400 s in artifact generation. All 1109 training rollouts fell. |

Quick stances printed up to three warnings:
- "`BaselineProgress` ... worse-than-trivial local optimum" (all six stances that left a log, including the four PASSes)
- "`EvalCollapseEarlyStop` ... disabled for the whole run" (only the four whose stance declares a backstop: raptor, T. rex, Brachiosaurus, Dibothrosuchus)
- "trained 53,248 timesteps, short of its curriculum budget" (only Brachiosaurus and Dibothrosuchus, the two runs that reached the training summary)

**Downstream, through the MANUAL cell**

| Node | Outcome |
|---|---|
| Allosaurus walk | FAIL (expected). The result bundle was not rewritten ("stage_results_list must be a contiguous curriculum prefix"). Cell 24 raised "not certified in this session". It got the same 20 stance-oriented training-history pages, and replays of the default view only. |
| Brachiosaurus walk | FAIL (`forward vel 0.00 < 0.75`). Same bundle and cell-24 behaviour. The summary says "Stage 2 curriculum gate". |
| Brachiosaurus food_reach | FAIL (`success rate 17% < 50%`). See the chart observations below. |
| Dibothrosuchus walk | FAIL (`forward vel -0.00 < 0.90`). Same bundle and cell-24 behaviour. |
| T. rex bite (`task_success/v1`) | FAIL (`task_success_lcb 0.0000 < 0.5000`, 0/30 episodes). The baseline watch compared hunt evaluations (626) with the **stance** statue (4470.6) and called the run "never beaten the zero-action baseline" at 14%. Same bundle and cell-24 behaviour. |
| Compsognathus-robot target_reach | FAIL (0% < 70%). All 30 evaluation episodes ran 1000 steps and scored 822 ± 40, close to the ~845 standing reward. Summary titled "Stage 3" inside `04_behavior/`. |
| T. rex recovery | ~9 env steps/s (against 30-45 for the other nodes); did not finish 50k steps in 90 min. |
| Compsognathus-robot recovery | Refused at start, by design: "68% quiet safe occupancy, below the 95% prerequisite; recalibrate the judge". |

The Brachiosaurus food_reach charts showed these problems:
- **`behavioral_metrics.png`:**
  - an empty "Strike Success Rate" panel with a duplicated legend
  - a stride "frequency" of ~300 on the same axis as a symmetry of ~1
  - a "Prey Distance" label for food
  - a heading cos θ band above 1
- **`gate_progress.png`:**
  - reward only, against a "zero action 1750" line that is the stance statue
  - an invisible single evaluation point
  - no success series, although the gate failed on success

**Artifact presence across models** (from the run trees). Compared on the stance runs that got through replay generation (T. rex, later killed in its probe battery; Brachiosaurus; Dibothrosuchus; Allosaurus):
- **Identical:** the five figures, the 20 training-history pages plus HTML/JSON, the evaluation CSVs, `progress.json`, `gate_progress.npz`, `policy_probe.npz` (absent only on Allosaurus), the replays (three views each for final and selected, plus `_stance.csv`) and the gate files.
- **Different, by gate kind:**
  - `stance_heading_report.*` and `stance_heading_panel_selected.csv`: raptor and T. rex
  - `stance_gate_report.*` and `stance_panel_selected.csv`: Compsognathus robot and Allosaurus
  - `stance_heading_probe.*`: Allosaurus (the Compsognathus robot also declares it, but its run was killed before writing it)
  - `stance_gate_probe_{constant,filtered}.*` and `stance_diagnostics.png`: T. rex only
  - `handoff_in_training.json`: not written for the reward-rail quadrupeds
- **Walk and hunt nodes** that finished artifact generation (Allosaurus, Brachiosaurus and Dibothrosuchus walks, Brachiosaurus food_reach, T. rex bite; the Compsognathus-robot walk was killed after one replay): the same five figures and 20 pages, and two default-view replays with no CSV.

## Appendix. Every verified finding, by theme

| Theme | Finding | Sev | Verdict | Title |
|---|---|---|---|---|
| 1 | `W2/model/compsognathus-hunt-success-terminal-unprofitable` | high | confirmed | Compsognathus target_reach pays more for hovering near the target than for reaching it, against the rule the other four hunts document |
| 1 | `W2/model/compsognathus-target-reach-success-is-net-loss` | high | confirmed | Compsognathus target_reach reward pays more for standing near the target than for reaching it (alive/posture/height 0.85 per step vs a 25-point terminal bonus), so the hunt deliverable is unlearnable for both compso models |
| 1 | `W2/model/allosaurus-walk-reward-penalizes-course-completion` | medium | confirmed | Allosaurus walk rewards avoiding the course target: success ends the episode with no bonus, against the repo's documented strike-avoidance rule |
| 1 | `W2/followup-first-hunt-node-evidence/fu-first-hunt-node-evidence-brachio-food-reach-hover-outearns-reach` | medium | partially confirmed | Brachiosaurus food_reach: standing near the food out-earns reaching it, so reward-based checkpoint choice and reward curves favour slow reaches and non-reaches over fast reaches |
| 1 | `W2/followup-first-hunt-node-evidence/fu-first-hunt-node-evidence-hunt-handoff-selection-penalizes-partial-success` | medium | partially confirmed | The hunt checkpoint that is gated and replayed is chosen on return alone (mean - std), not on success. Where a success pays less than surviving (Brachiosaurus, both Compsognathus), the selector prefers checkpoints that do not succeed |
| 2 | `W1/notebook/report-evaluate-silent` | medium | confirmed | Section 8's evaluate() cell rolls 30 episodes but shows nothing, so the gait metrics its comment promises never appear anywhere |
| 2 | `W1/artifacts/notebook-logging-hides-eval-and-certificates` | medium | confirmed | The notebook never sets up logging, so section 8's evaluation prints nothing and certificate results and progress stay hidden |
| 2 | `W1/metrics/cell24-eval-invisible-unsaved` | medium | confirmed | Cell 24's evaluate() shows nothing in the notebook, saves nothing, and uses a different seed from the gate |
| 2 | `W1/publication/deliverable-eval-not-persisted` | low | partially confirmed | The deliverable evaluation in cell 24 (gait symmetry, cost of transport, stride frequency, heading alignment) is printed and then thrown away; website pages advertise these metrics |
| 2 | `W1/artifacts/notebook-path-drops-quality-metrics` | low | partially confirmed | No pipeline saves a deterministic gait-quality panel (LocomotionMetrics) for the deliverable's selected checkpoint; notebook cell 24 computes one and throws it away |
| 3 | `W1/pipeline/baseline-watch-uses-stance-statue-for-every-node` | medium | confirmed | The zero-action baseline watch and progress.json compare every later node (recovery, walk, hunt) with the stage-1 stance statue |
| 3 | `W1/charts/stage1-baseline-on-all-nodes` | medium | confirmed | Every node's gate-progress figure draws the stance statue's reward as its 'zero action' line |
| 3 | `W2/model/brachiosaurus-stance-reward-ceiling-no-headroom` | low | partially confirmed | On statue-optimal stances (Brachiosaurus, Dibothrosuchus) the zero-action baseline watch calls a statue-level policy a 'worse-than-trivial local optimum that failed its gate', and the preflight gives unattainable advice, beside a by-design gate PASS |
| 3 | `W2/model/compsognathus-baseline-warning-on-statue-optimal-stance` | low | partially confirmed | BaselineProgress cannot be satisfied by a statue-equivalent stance: strict comparison against a statue scored on a different seed panel, with 'worse-than-trivial ... failed its gate' wording |
| 3 | `W2/model/allosaurus-statue-baselines-vacuous-for-torque-plant` | low | partially confirmed | The Allosaurus statue falls in 0.9 s, yet BaselineProgress still calls it the reward optimum, and the validated ReferenceWalk stand cannot replace it as the model's baseline |
| 3 | `W1/notebook/zero-action-preflight-verdicts` | low | confirmed | The zero-action preflight prints a different alarm on each species, gives no action, and misdiagnoses Allosaurus |
| 4 | `W1/notebook/quick-test-disconnects-colab` | low | partially confirmed | A Colab QUICK_TEST always costs the runtime under the default AUTO_DISCONNECT=True and ends in an unqualified gate-failure RuntimeError |
| 4 | `W1/notebook/quick-test-cannot-smoke-the-notebook` | medium | confirmed | QUICK_TEST cannot smoke-test the notebook: where it stops depends on the species, and the certificate still runs at full size |
| 4 | `W1/pipeline/quick-test-stops-at-first-gate` | medium | confirmed | QUICK_TEST cannot exercise anything past the first node: 50k steps always fail the first gate |
| 4 | `W1/artifacts/quick-test-does-not-exercise-report-path` | medium | confirmed | QUICK_TEST shrinks only the training budget: certification panels run at full size, and the report and bundle-check cells usually never run |
| 4 | `W2/model/compsognathus-quick-test-passes-stance-at-init` | low | partially confirmed | At 50k steps the anatomical Compsognathus stance is statue-level (yaw-0 screen 40/40, LCB 0.928, 99.95% of statue reward), so its QUICK_TEST certificate very likely passes and a default BEHAVIOR='auto' run continues to locomotion, as on Brachiosaurus and Dibothrosuchus (inferred, not observed) |
| 4 | `W2/model/tyrannosaurus-trex-probe-battery-unscaled-undocumented` | low | partially confirmed | T. rex's four stance probe reports (stance_gate_probe_*) have no reader documentation or notebook readout; their TOML derivations are stale (r7 counts, an Aug-2026 checkpoint), and the impulse grid is one the project's own battery found uninformative |
| 4 | `W1/publication/quick-test-indistinguishable` | medium | confirmed | Whether a run is a quick test is recorded only in its directory name and in training_summary.txt; the Drive pivot hides run identity and promotion would accept a quick test |
| 4 | `W2/compare/cmp-notebook-path-tested-on-compso-only` | low | partially confirmed | The end-to-end notebook training-and-artifact smoke covers only the Compsognathus pair (PR CI: robot PPO only); the other five models' gate kinds never pass through train_stage plus generate_stage_artifacts in tests |
| 5 | `W1/pipeline/no-colab-cap-awareness-or-auto-resume` | medium | confirmed | No awareness of the Colab session limit: chains run far past 24 h, a node starts with minutes left, resume is a manual five-setting procedure, and capped segments drop out of the recorded durations |
| 5 | `W1/notebook/resume-is-manual-two-pass` | medium | confirmed | Resuming after the Colab cap takes two Run alls and several knobs, though the chain loop already knows which node was interrupted |
| 5 | `W1/notebook/no-progress-eta-or-tensorboard` | low | confirmed | No inline progress, ETA or session-cap warning during nodes that take about 24 h; VERBOSE=1 is unusable and there is no inline TensorBoard |
| 5 | `W2/followup-colab-runtime-specifics/fu-colab-runtime-specifics-no-unattended-run-signal-or-colab-tier-guidance` | low | partially confirmed | No push notification when an unattended ~24 h node finishes, fails its gate or crashes (status is pull-only via progress.json on Drive), and no doc on the Colab runtime/tab conditions such a run needs |
| 6 | `W1/pipeline/eval-cadence-dominates-wallclock` | medium | confirmed | In-training evaluation runs about 0.8 serial env steps per training step on stance nodes, pushing 12M-step stances to the 24 h cap; the per-stage cadence key exists only on unmerged screen branches |
| 6 | `W2/followup-colab-runtime-specifics/fu-colab-runtime-specifics-ppo-device-unpinned-unrecorded-no-runtime-guidance` | medium | partially confirmed | The notebook's metadata opens Colab on an L4 GPU, so PPO MlpPolicy trains on CUDA with no stated reason, no device knob and no printed device; the device is only implied by the gpu block and provenance's gpu_driver, and SB3's train-on-CPU warning is silent for the CRITIC_PHASE recipe |
| 7 | `W1/notebook/disconnect-only-on-gate` | medium | confirmed | Auto-disconnect fires only on a failed gate or on reaching the last cell; any other exception leaves the Colab runtime assigned |
| 7 | `W1/pipeline/exceptions-skip-auto-disconnect` | low | confirmed | Only a failed gate releases the runtime; any other exception in an unattended Run all leaves the Colab GPU runtime assigned |
| 7 | `W1/artifacts/verdict-without-bundle-unrecoverable` | medium | partially confirmed | A runtime lost or stopped after the gate verdict but before the bundle leaves a passed node without replays, probes or a bundle, and the chain loop never repairs it |
| 7 | `W1/artifacts/silent-artifact-failures` | medium | confirmed | Best-effort figure, replay and probe failures are only logged; no summary, manifest or bundle records what is missing |
| 7 | `W2/followup-colab-runtime-specifics/fu-colab-runtime-specifics-egl-backend-unprobed-replay-and-stance-csv-lost-together` | low | partially confirmed | The physics-only _stance.csv replay files are written only after every frame renders, so any renderer error drops them along with both videos; setup prints 'EGL rendering enabled' without probing the backend |
| 8 | `W2/model/dibothrosuchus-snap-backstop-no-peak-warmup` | medium | confirmed | Snap node's collapse backstop has an absolute floor of 100 and no peak warm-up, though the standing statue scores 470-1400 on snap |
| 8 | `W2/model/velociraptor-strike-absolute-collapse-floor` | low | partially confirmed | Raptor strike (and Brachiosaurus food_reach, Dibothrosuchus snap) still use the absolute collapse_peak_floor = 100, below the strike statue's measured ~265-283, with no peak warm-up and the implicit 12/8/0.4 thresholds; CF3 fixed only the T. rex hunt, and nothing tracks the other three |
| 8 | `W2/compare/cmp-hunt-backstop-brachio-same-pattern` | low | confirmed | Brachiosaurus food_reach is the third hunt with an absolute collapse floor of 100, no peak warm-up and the code-default 12/8/0.4 |
| 8 | `W2/model/tyrannosaurus-trex-walk-collapse-floor-without-peak-warmup` | low | partially confirmed | T. rex locomotion is the only walk with a collapse backstop and no peak warm-up, and its reason rests on one pre-D-B5, pre-r8 run |
| 9 | `W1/charts/hunt-success-rate-chart-wrong` | medium | partially confirmed | No chart shows the hunt success rate the gate judges; the 'Strike Success Rate' panel plots successes per training step and appears on every stage |
| 9 | `W2/followup-first-hunt-node-evidence/fu-first-hunt-node-evidence-hunt-success-persisted-but-not-drawn` | low | partially confirmed | Per-evaluation hunt success is persisted (evaluations.npz, gate_progress.npz) but no chart draws it against min_success_rate; only a stage-mean termination-share bar shows it |
| 9 | `W1/charts/gate-criteria-not-charted` | medium | partially confirmed | Per-evaluation eval velocity and success are never persisted or charted; the only speed chart is unlabelled training-rollout speed with no bar |
| 9 | `W1/metrics/behavioral-metrics-chart` | medium | confirmed | behavioral_metrics.png labels foot force as 'stride freq' on the same axis as a 0-1 L/R load-balance line, and its 'Strike Success Rate' panel is empty for every model but T. rex and the raptor (dibothrosuchus snap_success is never recorded) |
| 9 | `W1/metrics/dibotho-keys-missing` | low | partially confirmed | Dibothrosuchus's snap reward terms and snap_success are missing from the shared REWARD_KEYS/INFO_KEYS lists, so its hunt node's training diagnostics and the cell-24 eval printout omit them |
| 9 | `W1/charts/success-terminations-counted-as-falls` | medium | partially confirmed | On hunt nodes and the Allosaurus walk, training-history pages count successes as falls and leave them out of every per-episode statistic (documented definition, misleading on task stages) |
| 9 | `W2/followup-first-hunt-node-evidence/fu-first-hunt-node-evidence-progress-json-counts-hunt-successes-as-falls` | low | partially confirmed | progress.json's 'falls' counts include hunt success terminations (documented), so its fall-only posture means on a hunt node are averaged mostly over successes |
| 9 | `W2/followup-first-hunt-node-evidence/fu-first-hunt-node-evidence-hunt-success-evidence-no-step-or-cause` | low | partially confirmed | Hunt evaluation CSVs record no termination cause or start distance, and hunt replays have no per-frame CSV or success marker. The success step is already the episode length, since success ends the episode. |
| 9 | `W2/model/velociraptor-hunt-distance-pelvis-not-claw` | low | partially confirmed | Hunt effector distances are emitted but never recorded: the raptor's min_claw_prey_distance and T. rex's head_prey_distance have no reader, and DiagnosticsCallback's 'jaw_distance # T-Rex' key is emitted by no env, so the raptor's and T. rex's min_prey_distance stays pelvis-based while the quadrupeds' measures the effector |
| 10 | `W1/metrics/cot-not-physical` | medium | confirmed | Both 'Cost of transport' readouts (the evaluate() printout and the locomotion_health.png panel) scale with energy_penalty_weight, not physical energy; mass is fixed at 1 kg, values differ about 30x across the 7 final nodes, and a stationary policy prints 0 |
| 10 | `W1/metrics/locomotionmetrics-gait-math` | medium | partially confirmed | LocomotionMetrics gait block (Symmetry, Stride freq, quad_gait_symmetry) is wrong; in the notebook the whole evaluate() report is silently hidden |
| 10 | `W1/metrics/time-to-target-threshold` | low | partially confirmed | Eval readout's 'Time to target / reached %' uses a fixed 0.5 m threshold that disagrees with task success for T. rex, Brachiosaurus and Compsognathus |
| 10 | `W1/metrics/supplementary-log-key-mismatch` | medium | partially confirmed | The curriculum's supplementary locomotion log never prints forward velocity or pelvis height, and the eval report prints 0 for unmeasured metrics |
| 11 | `W1/charts/reward-decomposition-selection` | medium | partially confirmed | The Reward Decomposition panel picks terms from TOML weights only: it hides the Allosaurus's dominant default-paid terms and shows NaN ghost terms and an overflowing, colour-reused legend |
| 11 | `W1/charts/behavioral-metrics-semantics` | medium | partially confirmed | behavioral_metrics.png / foot_contacts.png plot mislabelled or mis-scaled quantities, show empty panels keyed on key presence, and never show task success for 4 of 7 models |
| 11 | `W1/metrics/behavioral-metrics-chart` | medium | confirmed | behavioral_metrics.png labels foot force as 'stride freq' on the same axis as a 0-1 L/R load-balance line, and its 'Strike Success Rate' panel is empty for every model but T. rex and the raptor (dibothrosuchus snap_success is never recorded) |
| 11 | `W1/charts/locomotion-health-panels-misleading` | low | confirmed | locomotion_health.png: the termination bars hide the counts and time trend, and 'Cost of Transport' is not a cost of transport |
| 11 | `W1/charts/single-point-series-invisible` | low | confirmed | A series with one evaluation is drawn without markers, so its panel looks empty |
| 11 | `W1/charts/figure-labels-units` | low | partially confirmed | The older figures use stage numbers instead of node ids, repeat titles and omit units |
| 12 | `W2/compare/cmp-exploration-start-uncalibrated-quadrupeds-allosaurus` | medium | partially confirmed | Dibothrosuchus stance trains at SB3's default action std 1.0 without a recorded reason, and all of its training rollouts fall; no other model has its noise cliff measured |
| 12 | `W2/model/compsognathus-robot-falls-under-any-exploration-noise` | medium | partially confirmed | The Compsognathus robot statue falls under σ 0.02 noise, yet its stance recipe starts at σ 0.135, a ceiling measured only for the anatomical sibling. The audit's 'falls under any PPO-scale noise' note never became a test, issue entry or plan item. |
| 13 | `W1/pipeline/sac-resume-not-a-continuation` | medium | partially confirmed | The PPO-only resume fix (TC1/TC2, #514) leaves a SAC resume from the notebook's RESUME cell in the defective state that fix removed: post-resume checkpoints are pruned or misnamed, and the best and robust-best handoffs and the evaluation record are overwritten. Nothing user-facing documents it. |
| 13 | `W1/charts/sac-diagnostics-density` | medium | partially confirmed | SAC runs write a diagnostics.npz row every train_freq vector step: 94k-750k rows per series, hundreds of MB rewritten to Drive every 60 s, and charts that cannot be compared with PPO's |
| 13 | `W2/followup-sac-artifact-path/fu-sac-artifact-path-sac-warm-start-semantics-misdescribed` | low | partially confirmed | SAC warm starts begin with an empty replay buffer and a uniform-random learning_starts phase, while eight [sac] comments and RL_TRAINING_PLAN say the buffer keeps the parent's transitions |
| 13 | `W2/followup-sac-artifact-path/fu-sac-artifact-path-sac-exploration-unobservable` | low | partially confirmed | SAC exploration is recorded but never shown: std pages empty with a wrong reason, alpha uncharted, no progress.json policy block, and K3_PANEL silently off |
| 13 | `W2/followup-sac-artifact-path/fu-sac-artifact-path-sac-update-to-data-ratio-misstated` | low | confirmed | SAC update-to-data ratios are written per vector step, so every comment is off by N_ENVS, and the effective ratio varies 4x across models without derivation |
| 13 | `W2/followup-sac-artifact-path/fu-sac-artifact-path-sac-recipe-provenance-stale` | low | confirmed | SAC blocks rest on stale 'match PPO' claims and an unverified March raptor run; only the T. rex stance carries a staleness warning, and no per-node SAC status exists |
| 13 | `W2/followup-sac-artifact-path/fu-sac-artifact-path-trex-recovery-sac-recipe-unrecorded` | low | partially confirmed | Under SAC the T. rex recovery node warm-starts with the stance checkpoint's saved SAC settings, but logs 'SB3 defaults' and records an empty [sac] block and recipe digest (so replicate matching ignores a stance [sac] change) |
| 13 | `W1/notebook/sac-path-second-class` | low | partially confirmed | The SAC path gets 4 envs (CLI: 8), a PPO-only load preflight, and a CRITIC_PHASE knob that prints 'on' but does nothing |
| 13 | `W1/pipeline/sac-path-second-class-unflagged` | low | partially confirmed | The notebook offers SAC with no caveat; two documented PPO-only cases (T. rex recovery, CRITIC_PHASE) still run under SAC, and T. rex's mirror-KL is dropped without a log |
| 14 | `W2/compare/cmp-no-recipe-status-signal` | medium | partially confirmed | No recipe-maturity or hold signal: the notebook offers every model's default chain alike, though the maintainers' session plan holds several |
| 14 | `W2/model/brachiosaurus-chain-hold-not-surfaced-in-notebook` | low | partially confirmed | The notebook shows no per-model training status, so the maintainers' open GQ-3 hold (all of Brachiosaurus session 5, plus walk/hunt/recovery sessions on other models) is invisible when a model is picked from the dropdown |
| 14 | `W2/model/dibothrosuchus-recipe-on-hold-not-surfaced` | low | partially confirmed | No per-node readiness status: the notebook, READMEs and model pages offer every walk node (and the Dibothrosuchus hunt) as an ordinary deliverable, though the gait plan's open GQ-3 recommends holding them on today's rewards |
| 14 | `W2/model/velociraptor-pre-walk-protocol-not-surfaced` | low | confirmed | The notebook defaults chain the raptor stance straight into walk and strike, skipping the documented pre-walk protocol |
| 15 | `W2/model/allosaurus-course-complete-ungated` | medium | partially confirmed | No gate reads the Allosaurus headline success metric (course_complete); its walk gate's 550-of-3000-step length floor certifies walks that never finish the course |
| 15 | `W2/compare/cmp-walk-velocity-bars-not-comparable` | low | partially confirmed | Walk speed bars mean different things across models (a Froude-matched speed target for the raptor, anti-statue floors at Froude 0.003 or below for both Compsognathus and the Allosaurus), and three bars are undocumented |
| 15 | `W2/model/tyrannosaurus-trex-walk-reward-rail-vacuous` | low | partially confirmed | T. rex's walk rail (min_avg_reward 100) was never re-derived as 0.6 x statue the way its stance and hunt (D-B4) rails were, and the walk deliverable definition still cites 'reward floors' that the statue clears; the walk gate's general reward-floor weakness is already tracked |
| 15 | `W2/model/brachiosaurus-walk-node-publishes-incidental-food-success` | low | confirmed | Walk nodes record and publish incidental 'task success' (ungated success flags plus eval_policy on every node); Brachiosaurus already publishes 3.3% on its walk row |
| 15 | `W1/metrics/hunt-success-statistic` | low | partially confirmed | Drive summary shows the final model's success rate (and forward velocity) next to gate bars that judge the selected checkpoint; the raw-mean gate on the four non-T. rex hunts is the documented D-B14 decision |
| 16 | `W2/compare/cmp-certificate-stances-uneven-guards` | medium | confirmed | Same six-heading certificate, different guards: the anatomical Compsognathus stance lacks the hop watch, checkpoint archive and two-seed bar that T. rex and the raptor carry |
| 16 | `W2/model/velociraptor-stance-certificate-certifies-init-handoff` | low | partially confirmed | The notebook never shows the handoff's step, the guard NEXT_STEPS gives for the known early-handoff limit, and it prints the mean-best step instead. A hop-watch-stopped stance is still advanced along the chain. |
| 16 | `W2/model/compsognathus-manual-liftoff-check-not-automated` | medium | confirmed | The documented by-hand check of a Compsognathus anatomical stance (per-foot lift-offs, root-height rhythm) is never computed, charted or prompted, and the default chain warm-starts locomotion from the stance without it |
| 16 | `W2/model/dibothrosuchus-three-legged-stance-certified-unflagged-r3` | low | partially confirmed | The Dibothrosuchus stance, certified on a reward rail (already tracked as HIGH), has no report-only per-foot load screen; the r3 quick-test handoff stood on three legs in its replay and no summary says so |
| 17 | `W1/pipeline/allosaurus-least-instrumented` | medium | partially confirmed | Allosaurus, the newest model and the only torque-controlled one, trains on a near-default PPO/SAC recipe that the notebook does not mark as untuned, and its policy probe can never run |
| 17 | `W1/artifacts/policy-probe-never-builds-allosaurus` | low | partially confirmed | The policy probe records nothing for the torque-driven Allosaurus: an empty state bank also discards the per-update std, normalizer and rollout clip-share columns, which never use the bank |
| 17 | `W2/compare/cmp-allosaurus-heading-probe-on-heading-free-plant` | low | confirmed | Allosaurus has a heading-free observation but uses the robot's v1 gate plus the report-only heading probe; its scene rows replay the 0° row and the statue-falls warning fires on every stance |
| 17 | `W2/model/allosaurus-v1-report-hardcoded-statue-duty` | low | confirmed | The stance_quality/v1 text report prints a hard-coded T. rex statue support duty ('statue 0.998 / 0.002') that the Allosaurus statue, which falls in about 0.9 s, never produces |
| 17 | `W2/model/compsognathus-v1-report-touch-bilateral-and-hardcoded-statue` | low | partially confirmed | The v1 stance report's support lines are touch-only and sit beside a stale, model-agnostic '(statue 0.998)' literal; the robot gets no floor-truth cross-check until PR-G2/PR-G8 land |
| 17 | `W2/model/allosaurus-walk-target-invisible-in-artifacts` | low | partially confirmed | The Allosaurus walk goal is invisible in replays (the only one of the 6 task models without a visible target), and with no target-distance info key the model's 'Forward walking' section of evaluate() never prints |
| 17 | `W2/followup-offline-analysis-tools-species-coverage/fu-offline-analysis-tools-species-coverage-allosaurus-torque-plant-in-servo-reports` | low | partially confirmed | On the torque-driven Allosaurus, joint_excursion_report prints torques (N·m) as degrees, and actuator_saturation_report crashes on its MJCF (and on the Compsognathus robot's) because from_xml_string cannot resolve their texture and mesh files |
| 17 | `W1/artifacts/allosaurus-missing-from-artifact-docs` | low | partially confirmed | The artifact docs leave out Allosaurus, whose artifact set is mostly empty placeholders |
| 18 | `W2/followup-first-hunt-node-evidence/fu-first-hunt-node-evidence-hunt-replay-prey-out-of-frame` | medium | confirmed | Hunt replays leave the prey or target off-screen for most of the clip on five of six hunt models; only Brachiosaurus has a task camera |
| 18 | `W2/followup-offline-analysis-tools-species-coverage/fu-offline-analysis-tools-species-coverage-walk-hunt-nodes-no-per-actuator-record` | low | partially confirmed | Walk and hunt nodes get no deterministic per-step actuator CSV or joint-position envelope for the selected checkpoint, because one stance/recovery-only flag gates both the CSV and the extra camera views (per-evaluation and per-rollout per-actuator series do exist) |
| 18 | `W1/artifacts/artifact-coverage-matrix-uneven` | low | confirmed | Which diagnostic artifacts a node gets depends on gate kind and species in ways the docs don't explain |
| 18 | `W2/model/allosaurus-walk-target-invisible-in-artifacts` | low | partially confirmed | The Allosaurus walk goal is invisible in replays (the only one of the 6 task models without a visible target), and with no target-distance info key the model's 'Forward walking' section of evaluate() never prints |
| 19 | `W1/artifacts/no-chain-level-human-report` | medium | confirmed | The run summary leaves out gate verdicts, certified statistics and reused nodes; there is no chain-level report or index page |
| 19 | `W1/notebook/summary-omits-verdicts` | low | confirmed | The run summary printed after every node omits gate verdicts, reuse and the handoff evaluation; key report paths are never printed |
| 19 | `W1/publication/gate-reasons-not-propagated` | medium | partially confirmed | Why a node failed, and which gate kind judged it, never reaches collected_results.csv, training_summary.txt or the Drive tables |
| 19 | `W1/artifacts/collected-results-dead-and-kind-dependent-columns` | low | partially confirmed | collected_results.csv keeps empty columns left over from the retired sweep/scoring, leaves gate_reason empty on a FAIL, and omits the hunt's measured LCB; the Drive table compares the wrong reward to the rail |
| 19 | `W1/metrics/success-rate-on-non-task-nodes` | low | partially confirmed | Stance and walk nodes print 'Success rate: 0%' while the eval callback says N/A |
| 20 | `W1/publication/drive-summary-quarantines-interrupted-runs` | medium | partially confirmed | While node 2 or later trains, or after it is interrupted, the Drive summary quarantines the run as a 'canonical conflict' and hides its already-certified nodes from the tables and runs_summary.csv |
| 20 | `W1/publication/drive-summary-no-certification-board` | medium | partially confirmed | The Drive summary stops at PASS/FAIL booleans: no cross-model status table, no gate statistics, replication counts, lineage or links to the newer artifacts |
| 21 | `W1/publication/no-promotion-path` | medium | confirmed | Moving a run from Drive into results/ and onto the website is manual, has no written checklist and no tool. results/ holds only pre-bundle runs for 3 of the 7 models |
| 21 | `W1/publication/no-staleness-or-current-status` | medium | confirmed | No code path ever marks a result current or verified, nothing checks published results against the current plant, task or gate, and the website text hard-codes 'historical' |
| 21 | `W1/publication/results-layout-cannot-publish-trunked-chain` | medium | confirmed | results/ allows one summary per (model, algorithm) and the catalog drops ancestors, so a chain built with TRUNK_FROM or with seed replicates cannot be published whole |
| 21 | `W1/notebook/no-republish-cell` | low | confirmed | No committed cell re-publishes a run's bundle, though the save_run_bundle docstring says 'Re-run this cell' |
| 21 | `W1/publication/results-readme-and-strict-cell-drift` | low | confirmed | results/README.md still says summary.json is written 'after all three stages', and the Drive strict-promotion cell accepts only complete bundles although the catalog publishes partial ones |
| 21 | `W1/publication/presentation-inconsistent-across-models` | low | partially confirmed | Website and results presentation differs by model: videos, GIFs, CSV columns, page sections and labels |
| 22 | `W1/artifacts/embedded-videos-bloat-notebook` | medium | confirmed | Each replay is embedded in the notebook as base64 twice (chain loop and cell 28), and stance and recovery nodes have six each |
| 22 | `W1/notebook/inline-video-bloat` | low | confirmed | Every replay is base64-embedded and autoplays inline, and the replay cell embeds them all again; figures are plotted twice |
| 22 | `W1/charts/notebook-inline-figures` | medium | partially confirmed | The notebook shows only the older figures inline, twice, and never the gate-progress figure or the training-history report |
| 23 | `W2/model/compsognathus-history-key-actuators-miss-toe` | low | confirmed | The training-history policy-std page's 'Legs and tail_3' panel and the checkpoint table's std columns use a T. rex-centric KEY_ACTUATORS list. On the anatomical Compsognathus (and the Allosaurus) the toe/MTP servo is left out of the key panel and the table, and no tail_3 exists |
| 23 | `W2/compare/cmp-exploration-page-trex-reference-everywhere` | low | partially confirmed | The training-history exploration page draws the T. rex stance3 0.08 std reference on every model and node, unlabelled as T. rex-specific (the K3 readout itself is conditional) |
| 23 | `W2/model/dibothrosuchus-asymmetry-page-forelimbs-only` | low | partially confirmed | On Dibothrosuchus the 09_asymmetry page plots only the fl-fr pair for hip pitch, knee and ankle too, with no pair named in the titles (the docs disclose this only for hip roll); the rl-rr pair, which carries about 73% of standing load, appears only on page 20 |
| 23 | `W1/charts/training-history-pages-not-model-or-role-aware` | low | partially confirmed | Allosaurus silently joins the documented no-recorder gap (14 of 20 training-history pages blank), and the hunt success chart covers only 2 of 5 hunt tasks |
| 23 | `W1/metrics/telemetry-ledger-box-sole-only` | low | partially confirmed | Allosaurus (added 2026-11-21) falls into the documented no-recorder telemetry gap without being named, and adds its own gaps: no heading or spin info keys, no leg-DC pairs, and a policy-probe traceback on every PPO node |
| 23 | `W2/model/velociraptor-recorder-gap-premise-stale` | low | partially confirmed | TRAINING_ARTIFACT_PRESENTATION.md:683 still says the raptor has no head-tip site or neck hinge; since physics r5 it has both, so the no-recorder gap D-D46 kept (for want of a box pad) now leaves out a posture ledger that would fully resolve |
| 23 | `W2/model/dibothrosuchus-artifact-doc-digit-share-stale` | low | confirmed | TRAINING_ARTIFACT_PRESENTATION says Dibothrosuchus feet have no digit share, but since r3 every foot records one |
| 24 | `W2/model/tyrannosaurus-recovery-verdict-records-no-statistic` | low | confirmed | A passing recovery verdict records none of the statistics it was judged on, unlike stance and hunt, and no figure shows the recovery panel |
| 24 | `W2/model/tyrannosaurus-trex-recovery-legacy-judge-auto-frozen-stale` | low | confirmed | The notebook freezes the T. rex recovery gate on the stale r7 judge without warning, and nothing in the verdict marks it as legacy or unmeasured on r11 |
| 24 | `W2/compare/cmp-recovery-parent-qualification-asymmetric` | low | partially confirmed | The Compsognathus pair's quiet safe-set occupancy (the recovery freeze's 95% prerequisite) is measured only when the recovery node starts, never at the stance handoff; T. rex's legacy judge skips it by design |
| 24 | `W2/followup-offline-analysis-tools-species-coverage/fu-offline-analysis-tools-species-coverage-recovery-deliverable-no-offline-analysis` | low | partially confirmed | No command-line tool reports on a recovery checkpoint: the per-actuator scripts reject 'recovery', episode_trace labels a recovery trace 'Stance', cli eval prints only walking metrics, and the model-agnostic policy-panel roller (roll_policy_panel) can only be called from Python or the notebook |
| 24 | `W2/model/tyrannosaurus-trex-recovery-timing-and-stale-comments` | low | confirmed | The T. rex recovery window is documented as 2.0 s / 1.0 s but is 1.0 s / 0.5 s at the 100 Hz control rate; other stale T. rex config comments |
| 25 | `W1/pipeline/cli-notebook-judging-parity` | low | partially confirmed | The CLI's in-training verdict is bound to robust_best_model (chosen on reward alone), so a CLI 'passed' can certify a checkpoint that never met the gate; this is undocumented outside the hunt, and the CLI also lacks the baseline watch and publication |
| 25 | `W2/compare/cmp-catalog-gate-semantics-by-kind` | low | partially confirmed | Catalog's 'early-advancement gate' column and the manifest definitions give CLI consecutive-pass semantics for reward_and_length/v1 and stance_quality/v1 nodes, while the notebook (which never early-advances) certifies on one post-stage panel; only task_success/recovery/v2/heading rows say so, and Allosaurus's walk label omits its gate kind |
| 25 | `W1/pipeline/n-envs-outside-recipe-digest` | low | partially confirmed | N_ENVS changes PPO's rollout batch but is outside the recipe digest, so replicates, reuse and resumes treat different N_ENVS as the same recipe |
| 25 | `W1/pipeline/eval-protocol-provenance-mismatch` | low | confirmed | Provenance's evaluation protocols do not describe the evaluations that run, and checkpoint-selection evaluations are unpaired |
| 26 | `W2/followup-offline-analysis-tools-species-coverage/fu-offline-analysis-tools-species-coverage-offline-tools-not-surfaced-in-report-section` | low | confirmed | Section 8 points to no offline analysis tool, and three report scripts are documented only in plans and investigations |
| 26 | `W2/followup-offline-analysis-tools-species-coverage/fu-offline-analysis-tools-species-coverage-report-scripts-legacy-stage-int-unchecked` | low | confirmed | Offline report scripts take only legacy stage numbers: on T. rex and both Compsognathus models these differ from the NN_ directory prefix and cannot name recovery, and no script checks the checkpoint's recorded task |
| 26 | `W2/followup-offline-analysis-tools-species-coverage/fu-offline-analysis-tools-species-coverage-action-bound-model-mode-measures-clipped-actions` | low | confirmed | action_bound_report --model labels deterministic predict() output, which SB3 has already clipped, as 'Raw (pre-clip)', so its overshoot lines (\|a\|>1, abs max, E[a^2]) are always in-bound |
| 26 | `W2/followup-offline-analysis-tools-species-coverage/fu-offline-analysis-tools-species-coverage-allosaurus-torque-plant-in-servo-reports` | low | partially confirmed | On the torque-driven Allosaurus, joint_excursion_report prints torques (N·m) as degrees, and actuator_saturation_report crashes on its MJCF (and on the Compsognathus robot's) because from_xml_string cannot resolve their texture and mesh files |
| 26 | `W2/followup-offline-analysis-tools-species-coverage/fu-offline-analysis-tools-species-coverage-robot-hardware-tools-misdrive-or-skip` | low | partially confirmed | Sim-to-real analysis scripts don't cover the Compsognathus robot: the saturation report crashes on its mesh paths and its scripted gait doesn't recognise left_/right_ names; observation ablation supports T. rex only |
| 26 | `W2/followup-offline-analysis-tools-species-coverage/fu-offline-analysis-tools-species-coverage-recovery-deliverable-no-offline-analysis` | low | partially confirmed | No command-line tool reports on a recovery checkpoint: the per-actuator scripts reject 'recovery', episode_trace labels a recovery trace 'Stance', cli eval prints only walking metrics, and the model-agnostic policy-panel roller (roll_policy_panel) can only be called from Python or the notebook |
| 27 | `W1/charts/no-chain-or-optimizer-overview` | low | partially confirmed | No chart covers the whole chain or compares runs, and the recorded PPO/SAC optimizer metrics are never charted |
| 27 | `W1/pipeline/no-replicate-or-sweep-orchestration` | low | partially confirmed | No orchestration for seed replicates or hyperparameter sweeps |
| 28 | `W1/notebook/knob-density-and-dropdowns` | low | confirmed | Too many knobs and too much decision-ID prose up front; dropdowns allow invalid pairs and ALGORITHM is free text |
| 28 | `W1/notebook/knobs-validated-after-write` | low | partially confirmed | Invalid BEHAVIOR, RETRAIN_FROM or a pinned TRUNK_FROM is refused only after the storage cell created a run directory and provenance.json |
| 28 | `W2/model/velociraptor-critic-phase-unsupported-on-raptor` | low | confirmed | CRITIC_PHASE=True fails on the raptor, the notebook's default model, only after the run directory and baseline are written |
| 29 | `W1/artifacts/probe-cutoffs-none-warning` | low | confirmed | Spurious 'stance_probe_filter_hz entry is not a number: None' warning on stance nodes that don't set the key |
| 29 | `W1/metrics/probe-filter-none-warning` | low | partially confirmed | Spurious 'stance_probe_filter_hz entry is not a number: None' warning on every stance node except T. rex |
| 29 | `W1/artifacts/stage-number-wording` | low | confirmed | User-facing text says 'Stage N' using legacy numbers, while directories use NN_<id> |
| 30 | `W2/model/brachiosaurus-hunt-forward-ramp-misdescribed` | low | confirmed | Hunt ramp comments misdescribe RewardRampCallback: ramp_start_value is an absolute forward_vel_weight, so the walk-to-hunt drop (Brachiosaurus 4.4 -> 0.2) is immediate and the food_reach 'gradual transition' ramp only moves 0.2 -> 0.3; raptor/T. rex comments call it a 'fraction' |
| 30 | `W2/model/brachiosaurus-stale-recipe-comments` | low | confirmed | Brachiosaurus stage-TOML comments state values that no longer hold (statue, stance height, SAC 'Match PPO') |
| 30 | `W2/model/compsognathus-recipe-review-protocol-stale` | low | confirmed | TRAINING_RECIPE_REVIEW.md, cited by every anatomical Compsognathus TOML and the README as the next-run protocol, predates the six-heading certificate (D-D31/D-D32) and the 'recover' recipe; its step 4 and amendment describe superseded gates |
| 30 | `W2/model/compsognathus-robot-stranded-chain-still-listed-certified` | low | partially confirmed | ROADMAP still says T-Rex, Velociraptor and both Compsognathus models have certified stance and walk nodes (ROADMAP.md:28-31, 403-405), but D-D24/D-D26/D-D33 stranded every one; NEXT_STEPS already flags this with a table-wide note |
| 30 | `W2/compare/cmp-stale-hunt-default-docs` | low | confirmed | Docs still say BEHAVIOR defaults to "hunt" and that "hunt" resolves for every model; on Allosaurus it is refused |

## Status on 2026-11-25

*Appended 2026-11-25 (decision D-D63 (7), `docs/BEHAVIOR_RECIPES_PLAN.md` §6.2). The review above is kept as it
was written on 2026-11-22, on the tree of that day (`main` at or just after `6c9982d`, the parent of the hunt
removal). This addendum records what the tree of 2026-11-25 does about each finding: HEAD `d1dc274`, the merge of
`main` (`4f5e0fa`) into the branch that removed the hunt and made the walk command-driven. Every status below was
checked against the current code, notebook and configs, not taken from a commit message.*

What changed in between:

- **The hunt is gone** (decision D-D59, 2026-11-23: `527aa8b`, `3368abe`, `e3cccca`, `3d3ba61`). The six hunt nodes,
  their stage TOMLs and their publication, the `task_success/v1` gate and the `min_success_rate` floor, the hunt
  reward code of four species, and the prey, food and course target are deleted. Every model is heading-free
  (`configs/plant_versions.toml` note 24) and every chain ends at walk. A finding about a hunt node, the prey or a
  hunt success key is moot.
- **The walk follows a live command** (decisions D-D60 to D-D62, 2026-11-23 and 2026-11-24: `1c22496`, `857b6dd`,
  `feeabbb`, `285611d`). It is judged after training by `command_tracking/v1` on its own panels. Its handoff is
  chosen by the gated statistic on a selection panel, and it has its own figures (`command_tracking.png`,
  `command_gate_progress.png`). Its family statistics reach the stage summary, the CSV row and the verdict. Its bars
  are provisional (D-D62 (16)).
- **`main`'s changes since the review**, merged by `d1dc274`:
  - `94916cd`: each species' `success_keys` are read instead of hard-coded tuples, and LocomotionMetrics' gait
    symmetry, stride frequency and cost of transport are deleted.
  - `0667b31`: the cost-of-transport and gait-symmetry panels are deleted, and an empty behavioral panel says so.
  - `76b27b2`: a same-stage resume continues its record, and a SAC resume in place is refused.
  - `f3d3100`: judged-stage and complete-run write guards, and a Colab run without Drive keeps its runtime.
  - `2fd1ad1`: run files are written atomically.
  - `29cd504`: `RUN_KIND`, so screens and pilots live in their own tree.
  - `fff098f` and `475606d`: website and quick-start command fixes.

The status words:

- **addressed**: a change in the tree does what the finding asked, or makes it false. The file and the decision or
  commit are named.
- **moot**: what the finding was about no longer exists.
- **open**: still true of the current tree, with what remains named. A finding that is partly addressed is marked
  open, and the addressed part is named too.

Finding ids are the last part of the appendix's ids. Notebook cells are numbered as the review numbered them, which
is still the current notebook's order. A module path that names no top-level directory (`curriculum/`, `reporting/`,
`result_bundle/`, `scripts/`, `harnesses/`) and a dotted name such as `train_base.train` or
`visualization.plot_gate_progress` are relative to `environments/shared/`; `models/handoff_choice.json` sits in a
walk's stage directory.

### The ranked fixes (section 2)

| # | Status | Where it stands |
|---|---|---|
| 1 | addressed | Both Compsognathus target_reach hunts are deleted (D-D59). Under its live command the Allosaurus walk records `course_complete` as an event, not a termination (D-D62 (4), `environments/allosaurus_fragilis/base_env.py`, `_live_course_complete`), so finishing no longer forfeits the rest of the episode. The walk's handoff is chosen by the gated statistic (D-D61, `environments/shared/reporting/command_tracking_report.py`, `choose_command_handoff`). |
| 2 | open | No cell configures logging, `evaluate()` still returns `None`, section 8 saves no table, and `training_summary.txt` has no verdict block. `gate_progress.png` is still not shown inline. The three metrics the fix said to leave out no longer exist (`94916cd`). |
| 3 | open | A live-command stage (the walk) skips the baseline watch, `progress.json`'s `zero_action_reward` and the gate-progress statue line (D-D62). The recovery nodes are still compared with the stance statue, and `BaselineProgress` is still not gate-kind aware. |
| 4 | addressed | The hunt gates are gone (D-D59). The walk's gated statistics are drawn against its bars at every handoff candidate (D-D61, `visualization.plot_command_gate_progress`), and success keys come from each species (`94916cd`). No committed stage ends an episode on success any more. |
| 5 | open | There is no smoke mode, and provenance, the verdict and the CSV hold no `quick_test` field. Every model's walk runs `train_base.train` and `generate_stage_artifacts` in CI (`test_walk_species_training.py`), but only at full depth; the stance gate kinds go through that path only on the Compsognathus pair. |
| 6 | moot | The hunts are deleted. The walks' backstops follow one rule with the D-B5 peak warm-up (D-D62 (15)). |
| 7 | open | The Dibothrosuchus, Brachiosaurus and Allosaurus stances still declare no `log_std_init`, and the robot's -2.0 is still unmeasured. |
| 8 | open | No stage has a status field. |
| 9 | open | None of the five parts exists: exception hook, session budget, `RESUME_STAGE = "auto"`, device knob and `%tensorboard` cell. |
| 10 | open | The task cameras and the Allosaurus target are moot, since the prey and the target are removed. The per-step CSV is still written for stance and recovery nodes only. |

Smaller wins:

- `stance_probe_filter_hz = None`: **open** (theme 3.29).
- Node ids instead of "Stage N": **open** (3.29).
- One embedded replay: **open** (3.22).
- Knob validation before the run directory is written: **open** (3.28).
- One `MODEL` dropdown: **open** (3.28).
- SAC labelled experimental: **open** (3.13). Only the resume case is refused (`76b27b2`).
- Stale getting-started docs: **addressed** (`website/docs/getting-started/quick-start.md` and `installation.md`:
  `fff098f`, `475606d`, `527aa8b`; theme 3.30).
- Stale TOML comments: **open** (3.30).

### Findings by theme (section 3)

#### 3.1 Task rewards where succeeding pays less than not succeeding

- `compsognathus-hunt-success-terminal-unprofitable`, `compsognathus-target-reach-success-is-net-loss`: **moot**. Both
  `behavior.toml` hunts and their target_reach reward are deleted (D-D59: `527aa8b`, `3d3ba61`).
- `allosaurus-walk-reward-penalizes-course-completion`: **addressed**. Under a live command the course runs along the
  commanded heading and completing it is an event that ends nothing. `success` holds from that step to the horizon
  (D-D62 (4), `environments/allosaurus_fragilis/base_env.py`, `_live_course_complete`). `configs/allosaurus_fragilis/anatomical/locomotion.toml`
  says so in its header.
- `fu-first-hunt-node-evidence-brachio-food-reach-hover-outearns-reach`: **moot** (the food_reach node and the food
  are deleted, D-D59).
- `fu-first-hunt-node-evidence-hunt-handoff-selection-penalizes-partial-success`: **addressed**. The hunt half is moot.
  The node that replaced it in the chain's last position, the walk, chooses its handoff by the gated statistic on a
  selection panel. That panel is disjoint from the certification panel (D-D61,
  `environments/shared/reporting/command_tracking_report.py`, `choose_command_handoff`, recorded in
  `models/handoff_choice.json`). The stances still hand off `robust_best_model`, ranked on reward (theme 3.25).

#### 3.2 Section 8's evaluation cell shows and saves nothing

- `report-evaluate-silent`, `notebook-logging-hides-eval-and-certificates`, `cell24-eval-invisible-unsaved`: **open**.
  - No notebook cell configures logging.
  - `environments/shared/evaluation.py`'s `evaluate()` still ends in `_log_eval_results` (logger only) and returns
    `None`.
  - Cell 24 still rolls 30 episodes at the default seed and saves nothing.
  - What changed: the cell's comment now promises velocity consistency, posture, termination reasons and a success
    rate instead of gait symmetry, cost of transport and stride frequency (`94916cd` deleted those).
- `deliverable-eval-not-persisted`: **open**. Nothing persists cell 24's numbers. Its advertised gait metrics no
  longer exist (`94916cd`).
- `notebook-path-drops-quality-metrics`: **moot**. The LocomotionMetrics gait block this asked to persist is deleted
  (`94916cd`). The remaining gap, that the cell's evaluation is thrown away, is `cell24-eval-invisible-unsaved`.

#### 3.3 The zero-action baseline is applied to the wrong node, and its alarms are wrong on statue-optimal or falling plants

- `baseline-watch-uses-stance-statue-for-every-node`, `stage1-baseline-on-all-nodes`: **open**. The hunt nodes are
  gone. A live-command stage skips all three readers:
  - the watch (`curriculum/baseline_watch.py`, `build_baseline_progress_callback`);
  - `progress.json`'s `zero_action_reward` (`curriculum/progress_report.py`, `_zero_action_reward`);
  - the gate-progress line (`visualization.plot_gate_progress`).

  These skips came with D-D62 (`feeabbb`, `285611d`). What remains: on the T. rex and both Compsognathus recovery
  nodes all three still read the run's one stance statue. Cell 13 still hard-codes `BASELINE_STAGE = 1`, and
  `read_zero_action_baseline` still takes no stage.
- `brachiosaurus-stance-reward-ceiling-no-headroom`, `compsognathus-baseline-warning-on-statue-optimal-stance`:
  **open**. The comparison is still strict, on another seed panel, with the "worse-than-trivial local optimum"
  wording (`baseline_watch.py`).
- `allosaurus-statue-baselines-vacuous-for-torque-plant`: **open**. The Allosaurus stance is command-free, so the
  watch still runs there against a statue that falls.
- `zero-action-preflight-verdicts`: **open**. `scripts/zero_action_baseline.py` still tests a stance gate kind
  before `n_standing == 0`, so a falling statue on a stance-gated plant reads "CHECK GATE", and no verdict names a
  next action.

#### 3.4 QUICK_TEST only shortens training

- `quick-test-disconnects-colab`: **open**. Under `QUICK_TEST` a gate failure still reaches `halt(...)` with
  `AUTO_DISCONNECT` (cell 20). Since `f3d3100` a run kept on the VM's disk is never released, but a Drive run still
  is.
- `quick-test-cannot-smoke-the-notebook`, `quick-test-does-not-exercise-report-path`: **open**. `node_budget` still
  changes only the step budget (cell 15). Certificate panels, probe batteries and replays run at full size.
- `quick-test-stops-at-first-gate`: **open**. Where a quick test stops now depends on the walk too: the walk's
  provisional `command_tracking/v1` gate fails a 50k-step policy on every model.
- `compsognathus-quick-test-passes-stance-at-init`: **open** (an inference about the anatomical stance's gate, which
  is unchanged).
- `tyrannosaurus-trex-probe-battery-unscaled-undocumented`: **open**. The T. rex stance still declares the full
  `stance_probe_filter_hz` sweep, and nothing scales it under `QUICK_TEST`.
- `quick-test-indistinguishable`: **open**. `29cd504` made the Drive summary read each run's tree into a `run_kind`
  column, and its strict promotion cell now refuses a quick-test or screen run. `provenance.json`,
  `gate_verdict.json` and the CSV rows still carry no `quick_test` field; the folder stays the only marker
  (D-D16 (b)).
- `cmp-notebook-path-tested-on-compso-only`: **open**. Every model's committed walk now goes through
  `train_base.train` and `generate_stage_artifacts` in `environments/shared/tests/test_walk_species_training.py`
  (D-D62 (19)). On pull requests only the robot pair runs; the nightly, dispatch and `full-ci` runs cover all seven
  (`.github/workflows/python-ci.yml`). The stance gate kinds of the velociraptor, the T. rex, the quadrupeds and
  the Allosaurus still go through that path only in `test_compsognathus_longipes_training.py`'s Compsognathus runs.

#### 3.5 No Colab time awareness, a manual two-pass resume, no progress or notifications

- `no-colab-cap-awareness-or-auto-resume`, `resume-is-manual-two-pass`: **open**.
  - The chain loop starts each node with no time check.
  - The run id memo is still `_ACTIVE_RUN_ID` in kernel globals (cell 8).
  - `RESUME_STAGE` takes a node, not `"auto"`.
  - A resumed stage still records the resumed session's duration only.

  What changed: `76b27b2` made the RESUME cell refuse a recipe, label or algorithm that would not continue the
  node, and warn on `BEHAVIOR = "auto"` and on another commit. It stays a manual second pass.
- `no-progress-eta-or-tensorboard`: **open**. `VERBOSE = 1` still sets SB3's own verbosity
  (`train_base.py`, `alg_kwargs["verbose"] = verbose`) beside the progress bar, and there is no `%tensorboard` cell.
- `fu-colab-runtime-specifics-no-unattended-run-signal-or-colab-tier-guidance`: **open** (no notification hook).

#### 3.6 In-training evaluation cost, and the training device

- `eval-cadence-dominates-wallclock`: **open**. Cell 15 still passes `eval_freq=50_000`, and no stage TOML declares
  a cadence key. `94916cd` fixed only a cadence of 0 below `--n-envs`.
- `fu-colab-runtime-specifics-ppo-device-unpinned-unrecorded-no-runtime-guidance`: **open** (no device knob or
  printed device).

#### 3.7 Non-gate failures keep the runtime, and missing artifacts stay unrepaired

- `disconnect-only-on-gate`, `exceptions-skip-auto-disconnect`: **open**. Only `halt` on a failed gate (cell 20) and
  the last cell release the runtime. `f3d3100` added a case that never releases it, a run on the VM's disk, but no
  exception hook.
- `verdict-without-bundle-unrecoverable`, `silent-artifact-failures`: **open**. The reuse branch still does not
  regenerate missing artifacts, and there is no expected-artifact list. `2fd1ad1` made the small run files atomic,
  which is a separate fix.
- `fu-colab-runtime-specifics-egl-backend-unprobed-replay-and-stance-csv-lost-together`: **open**. Setup (cell 3)
  still writes the NVIDIA EGL file without a GPU check and prints "EGL rendering enabled" untested.
  `evaluation.record_stage_video` still writes `_stance.csv` after every video.

#### 3.8 Hunt collapse backstops

- `dibothrosuchus-snap-backstop-no-peak-warmup`, `velociraptor-strike-absolute-collapse-floor`,
  `cmp-hunt-backstop-brachio-same-pattern`: **moot** (the strike, food_reach and snap nodes are deleted, D-D59).
- `tyrannosaurus-trex-walk-collapse-floor-without-peak-warmup`: **addressed**. Every walk that carries the backstop
  declares the D-B5 peak warm-up `warmup_timesteps + cruise_ramp_timesteps`. On the T. rex that is 2.3M, new
  (D-D62 (15), `configs/tyrannosaurus_rex/anatomical/locomotion.toml`, pinned by
  `test_curriculum_early_stopping.py`).

#### 3.9 Task success is never charted against its bar

- `hunt-success-rate-chart-wrong`, `fu-first-hunt-node-evidence-hunt-success-persisted-but-not-drawn`: **moot**.
  The hunt gates are deleted (`3368abe`), and the "Strike Success Rate" panel went with the hunting row (`e3cccca`,
  D-D59).
- `gate-criteria-not-charted`: **addressed**. The forward-velocity bar it named left the walk with
  `reward_and_length/v1` (D-D62 (12)). The walk's gated statistics are drawn at every handoff candidate against the
  bars (`command_gate_progress.png`, `visualization.plot_command_gate_progress`), with the tracking fraction over
  training against the per-episode bar (`command_tracking.png`) (D-D61, `857b6dd`). `training_curves.png`'s
  "Speed" panel is unchanged: still the training rollouts' speed with no bar, but no longer the only speed chart.
- `behavioral-metrics-chart`: **addressed** (`environments/shared/visualization.py`, `plot_diagnostics_graphs`'
  behavioral-metrics figure). Its "stride freq" panel is deleted (`0667b31`), and its Strike Success panel went
  with the hunt's row (`e3cccca`, D-D59). The Dibothrosuchus snap keys it named are moot (D-D59). The same finding
  is listed again under 3.11.
- `dibotho-keys-missing`: **moot**. The snap terms are deleted (`e3cccca`), and the success keys are read per species
  (`94916cd`).
- `success-terminations-counted-as-falls`, `fu-first-hunt-node-evidence-progress-json-counts-hunt-successes-as-falls`:
  **moot**. No committed stage ends an episode on success any more: the hunts are deleted, and the Allosaurus walk's
  course is an event under its live command (D-D62 (4)). One stale sentence remains:
  `docs/TRAINING_ARTIFACT_PRESENTATION.md` (the rollout-balance ledger paragraph) still says that the Allosaurus
  walk's `course_complete` ends its episode. That holds only for a command-free course.
- `fu-first-hunt-node-evidence-hunt-success-evidence-no-step-or-cause`, `velociraptor-hunt-distance-pelvis-not-claw`:
  **moot**. The hunt nodes, the prey and the effector distances are deleted, and `DiagnosticsCallback` no longer
  lists `jaw_distance`.

#### 3.10 The gait and energy metrics are computed wrongly

- `cot-not-physical`, `locomotionmetrics-gait-math`: **addressed**. Both cost-of-transport readouts and the
  LocomotionMetrics gait block are deleted: the gait symmetry, stride frequency and cost of transport of
  `LocomotionMetrics` (`environments/shared/metrics.py`, `94916cd`) and the cost-of-transport panel of
  `plot_diagnostics_graphs` (`environments/shared/visualization.py`, `0667b31`). No readout of them remains to be
  wrong.
- `time-to-target-threshold`: **moot** (`time_to_target` and the 0.5 m threshold went with the hunt metrics,
  `e3cccca`).
- `supplementary-log-key-mismatch`: **open**. `curriculum/advancement.py` still looks up `mean_forward_velocity` and
  `mean_pelvis_height`, which `aggregate_episodes` names `mean_mean_*`. `_log_eval_results` still prints 0 for an
  absent metric.

#### 3.11 The older per-node figures mislabel or hide data

- `reward-decomposition-selection`: **open**. `visualization.plot_diagnostics_graphs` still picks terms from the
  TOML's weights. D-D62 added the stand block's weights and the tracking term, but a term paid at a constructor
  default the TOML does not restate is still hidden.
- `behavioral-metrics-semantics`: **open**. `0667b31` deleted the stride and symmetry panel and made an empty panel
  read "No data recorded", and `e3cccca` (D-D59) deleted the hunting row. What remains:
  - "Distance Traveled" and "Drift Distance" are still per-step rollout means of running episode values.
  - `foot_contacts.png`'s "Diag A/B" are still `np.maximum` of per-rollout means.
  - The heading band is still mean ± std, unclipped above 1.
- `behavioral-metrics-chart`: see 3.9.
- `locomotion-health-panels-misleading`: **open**. The cost-of-transport panel is deleted (`0667b31`), but the
  termination bars are still unweighted mean fractions with no counts or time axis.
- `single-point-series-invisible`: **open**. `plot_training_curves` and `plot_gate_progress` still draw plain lines.
  The new command figures use markers.
- `figure-labels-units`: **open**. Legends still read "Stage N: …", and `foot_contacts.png` is still unitless.

#### 3.12 Stance exploration noise

- `cmp-exploration-start-uncalibrated-quadrupeds-allosaurus`, `compsognathus-robot-falls-under-any-exploration-noise`:
  **open**. The Dibothrosuchus, Brachiosaurus and Allosaurus `stance.toml` files still declare no `log_std_init`
  (SB3's std 1.0). `configs/compsognathus_longipes/robot/stance.toml` still copies -2.0 with no measurement. The
  noise-tolerance tests still cover only the velociraptor and the anatomical Compsognathus. Note 24 moved every
  plant, so the measured cliffs are those of the older revisions.

#### 3.13 SAC is offered like PPO but behaves differently

- `sac-resume-not-a-continuation`: **addressed**. The RESUME cell (cell 18) refuses `ALGORITHM` other than `"ppo"`.
  `train_base.train` refuses a same-stage load that does not continue the run into a directory holding the stage's
  records (`76b27b2`). Section 5's markdown says why.
- `sac-diagnostics-density`: **open**. `DiagnosticsCallback` still appends a row at every rollout end, every
  `train_freq` under SAC. Only its file rewrite is throttled.
- `fu-sac-artifact-path-sac-warm-start-semantics-misdescribed`: **open**. The velociraptor, T. rex, Brachiosaurus and
  Dibothrosuchus `locomotion.toml` `[sac]` comments still say the buffer keeps the stance's transitions.
- `fu-sac-artifact-path-sac-exploration-unobservable`: **open**. The std pages still read only a `log_std` tensor
  (`reporting/training_history.py`, `read_log_std`).
- `fu-sac-artifact-path-sac-update-to-data-ratio-misstated`, `fu-sac-artifact-path-sac-recipe-provenance-stale`:
  **open**. The "Match PPO" comments are unchanged in the velociraptor, Brachiosaurus and Dibothrosuchus stance
  TOMLs, and so are the update-to-data comments.
- `fu-sac-artifact-path-trex-recovery-sac-recipe-unrecorded`: **open**.
  `configs/tyrannosaurus_rex/anatomical/recovery.toml` still extends `["stage", "env", "ppo"]`, so it records no
  `[sac]`.
- `sac-path-second-class`, `sac-path-second-class-unflagged`: **open**. `ALGORITHM` is still free text. The
  notebook still runs `N_ENVS = 4`. `CRITIC_PHASE` still prints its state under SAC. Of the documented PPO-only
  cases, only the resume is refused.

#### 3.14 No recipe-maturity or hold signal

- `cmp-no-recipe-status-signal`, `brachiosaurus-chain-hold-not-surfaced-in-notebook`,
  `dibothrosuchus-recipe-on-hold-not-surfaced`: **open**. No stage in a `stages.toml`, and no model in
  `configs/species_manifest.toml`, has a status field, and the notebook's chain table prints none. The hunt sessions
  these named are moot (D-D59). The walk's provisional bars are now stated in prose on all seven model pages
  (`website/docs/models/`, D-D62 (17)). The notebook does not print them.
- `velociraptor-pre-walk-protocol-not-surfaced`: **open**. `BEHAVIOR = "auto"` still resolves to the walk, so a plain
  Run all chains the stance into it.

#### 3.15 Gate bars and success metrics mean different things across species

- `allosaurus-course-complete-ungated`: **addressed**. The Allosaurus walk is judged by `command_tracking/v1`, like
  every walk. `course_complete` is recorded as an event and is stated to be no criterion (D-D62 (4) and (17),
  `configs/species_manifest.toml`).
- `cmp-walk-velocity-bars-not-comparable`: **addressed**. Every walk is judged on one set of provisional bars, as
  fractions of each model's cruise speed (D-D62 (9) and (12)). Each `locomotion.toml` and the plan's §4.6 live-walk
  table (`docs/BEHAVIOR_RECIPES_PLAN.md`, "Froude at cruise") state the Froude number. The bars stay provisional
  (D-D62 (16)).
- `tyrannosaurus-trex-walk-reward-rail-vacuous`: **moot**. `min_avg_reward` left every walk with
  `reward_and_length/v1` (D-D62 (12)).
- `brachiosaurus-walk-node-publishes-incidental-food-success`: **moot**. The food is deleted, and the Brachiosaurus
  declares no success key (`e3cccca`, `3d3ba61`).
- `hunt-success-statistic`: **open**. The raw-mean hunt gates are deleted (`3368abe`). The Drive summary still shows
  the final model's `success_rate` and forward velocity columns. It reads no `command_*` column, so a walk's gated
  statistics never reach its tables (`notebooks/google_drive_summary.ipynb`).

#### 3.16 Stance certification guards differ across species

- `cmp-certificate-stances-uneven-guards`: **open**. `configs/compsognathus_longipes/anatomical/stance.toml` still
  declares no `certification_seeds`, checkpoint archive or hop watch, and records no reason.
- `velociraptor-stance-certificate-certifies-init-handoff`: **open**. The chain loop prints the handoff's file name,
  not its step, and a hop-watch-stopped stance still advances.
- `compsognathus-manual-liftoff-check-not-automated`: **open**.
- `dibothrosuchus-three-legged-stance-certified-unflagged-r3`: **open**. The stance is still `reward_and_length/v1`
  with no per-foot load screen. The r3 checkpoint itself is stranded (note 24).

#### 3.17 Allosaurus is under-tuned and much shared tooling breaks on it

- `allosaurus-least-instrumented`: **open** (near-default tables, no untuned mark).
- `policy-probe-never-builds-allosaurus`: **open**. `policy_probe.py` still raises "no bank state could be built"
  and drops the bank-free columns with it.
- `cmp-allosaurus-heading-probe-on-heading-free-plant`: **open**. The Allosaurus stance still declares
  `stance_quality/v1` with `stance_probe_spawn_yaw_deg`, and records no reason.
- `allosaurus-v1-report-hardcoded-statue-duty`, `compsognathus-v1-report-touch-bilateral-and-hardcoded-statue`:
  **open**. `reporting/stance_report.py` still prints "(statue 0.998, not gated)".
- `allosaurus-walk-target-invisible-in-artifacts`: **moot**. The course target is deleted (note 24), and the course
  runs along the commanded heading (D-D62 (4)). `command_tracking_trace.csv` records `course_progress_m`. The same
  finding is listed again under 3.18.
- `fu-offline-analysis-tools-species-coverage-allosaurus-torque-plant-in-servo-reports`: **open**.
  `scripts/joint_excursion_report.py` still converts every actuator to degrees, and
  `scripts/actuator_saturation_report.py` still loads through `from_xml_string`. The same finding is listed again
  under 3.26.
- `allosaurus-missing-from-artifact-docs`: **open**. `docs/TRAINING_ARTIFACT_PRESENTATION.md` names the Allosaurus
  only in the stale rollout-balance sentence noted under 3.9.

#### 3.18 Replays don't show the task; walk nodes get thinner artifacts

- `fu-first-hunt-node-evidence-hunt-replay-prey-out-of-frame`: **moot** (no prey or target, D-D59).
- `fu-offline-analysis-tools-species-coverage-walk-hunt-nodes-no-per-actuator-record`: **open**.
  `reporting/stage_artifacts.py` still sets `replay_diagnostics = stage in (1, "stance", "recovery")`. The walk
  gained `command_tracking_trace.csv` (D-D61): commanded and achieved velocity on one cruise and one stand episode,
  with no actuator columns.
- `artifact-coverage-matrix-uneven`: **open**. No expected-artifact matrix exists. The walk adds its command report,
  panel CSV, trace and two figures.
- `allosaurus-walk-target-invisible-in-artifacts`: see 3.17.

#### 3.19 Summaries leave out verdicts and reasons, and there is no chain-level report

- `no-chain-level-human-report`: **open**.
- `summary-omits-verdicts`: **open**. `reporting/text_summaries.py`'s `write_training_summary` still prints no gate
  kind, verdict or reason. Each `stage_summary.txt` carries the verdict, the failures and, on the walk, the family
  statistics and the chosen handoff (D-D61, `_command_tracking_lines`).
- `gate-reasons-not-propagated`: **open**. `gate_reason` is still in `reporting/csv_output.py`'s
  `CSV_METRIC_COLUMNS`, and nothing writes it.
- `collected-results-dead-and-kind-dependent-columns`: **open**. The hunt LCB is moot, and a walk row now carries its
  command statistics (D-D61, `csv_output.py`). `gate_evaluable`, `gate_reason`, `quality_score` and `quality_rank`
  are still empty columns.
- `success-rate-on-non-task-nodes`: **open**. A success rate is now recorded only for a model that declares success
  keys (`e3cccca`, `reporting/stage_artifacts.py` and `text_summaries.py`), and six models declare none. The
  Allosaurus declares `course_complete` on every node, so its command-free stance still records a 0% rate.

#### 3.20 The Drive summary hides mid-chain runs and has no certification board

- `drive-summary-quarantines-interrupted-runs`, `drive-summary-no-certification-board`: **open**. The routing in
  `notebooks/google_drive_summary.ipynb` is unchanged apart from the hunt's removal (`527aa8b`, `3368abe`) and
  `29cd504`'s `run_kind` column. A walk's command statistics are not shown (3.15).

#### 3.21 Publication to `results/` and the website is manual

- `no-promotion-path`: **open**. `results/` now holds no summary at all: the four historical ones were deleted with
  the hunt (`527aa8b`, D-D59).
- `no-staleness-or-current-status`: **open**. `result_bundle/provenance.py` and `reporting/summaries.py` still write
  `model_revision_status = "historical"`.
- `results-layout-cannot-publish-trunked-chain`, `no-republish-cell`: **open**. The notebook's `save_run_bundle`
  docstring still says "Re-run this cell".
- `results-readme-and-strict-cell-drift`: **open**. `results/README.md` now says `summary.json` is written after the
  run's last stage (`527aa8b`). The Drive strict-promotion cell still calls
  `validate_result_bundle(..., require_complete=True)`.
- `presentation-inconsistent-across-models`: **open**. With no published summary the CSV differences are gone, but
  each model page still embeds its media by hand (none on the two Compsognathus pages).

#### 3.22 Embedded replays and re-plotted figures

- `embedded-videos-bloat-notebook`, `inline-video-bloat`, `notebook-inline-figures`: **open**.
  `notebook_runtime.display_stage_videos` still embeds every mp4 with autoplay. Cell 28 re-embeds the chain, cell 26
  re-plots the older figures, and `gate_progress.png`, the command figures and `training_history.html` are not
  shown inline.

#### 3.23 The training-history report uses T. rex defaults

- `compsognathus-history-key-actuators-miss-toe`, `cmp-exploration-page-trex-reference-everywhere`,
  `dibothrosuchus-asymmetry-page-forelimbs-only`: **open**. `reporting/training_history.py` still defines
  `KEY_ACTUATORS = ("hip_pitch", "hip_roll", "knee", "ankle", "tail_3")` and still draws `EXPLORATION_REFERENCE_STD`
  0.08 on every model, and the pair choice is unchanged.
- `training-history-pages-not-model-or-role-aware`, `telemetry-ledger-box-sole-only`: **open** for the Allosaurus
  part. The hunt success chart they named is moot.
- `velociraptor-recorder-gap-premise-stale`: **open**. `docs/TRAINING_ARTIFACT_PRESENTATION.md`'s recorder table
  paragraph still says the velociraptor has no head-tip site or neck hinge. The velociraptor MJCF has a `head_tip`
  site.
- `dibothrosuchus-artifact-doc-digit-share-stale`: **open**. The same document's pad-columns paragraph still says no
  Dibothrosuchus foot declares digits.

#### 3.24 Recovery nodes record no statistics; the T. rex judge is stale

- `tyrannosaurus-recovery-verdict-records-no-statistic`: **open**. The verdict gained the walk's command fields
  (`result_bundle/gate_verdict.py`, D-D61). It gained no recovery statistic.
- `tyrannosaurus-trex-recovery-legacy-judge-auto-frozen-stale`: **open**. `harnesses/freeze_recovery_gate.py` still
  freezes the T. rex on `LEGACY_RECOVERY_JUDGE_MODEL_ID` without a warning. The plant has moved again since
  (physics r12, note 24).
- `cmp-recovery-parent-qualification-asymmetric`: **open**.
- `fu-offline-analysis-tools-species-coverage-recovery-deliverable-no-offline-analysis`: **open**. The per-actuator
  scripts' `stage` argument now takes `choices=(1, 2)`, so the hunt is gone from it, and it still cannot name
  recovery. The same finding is listed again under 3.26.
- `tyrannosaurus-trex-recovery-timing-and-stale-comments`: **open**.
  `configs/tyrannosaurus_rex/anatomical/recovery.toml` still calls `recovery_t_recover_steps = 100` "2.0 s at the
  50 Hz control rate". The plant steps at 100 Hz: `timestep="0.002"` and `frame_skip` 5.

#### 3.25 The CLI and the notebook certify differently; provenance misdescribes evaluations

- `cli-notebook-judging-parity`: **open**. The walk is judged after training on the CLI too, on the pair its gated
  statistic chose (D-D61, `train_base._post_training_command_verdict`). The stance kinds' CLI verdicts are still the
  in-training manager's, bound to `robust_best_model`.
- `cmp-catalog-gate-semantics-by-kind`: **open**. The walk rows now say "judged post-stage", and the Allosaurus walk
  is labelled by its kind (D-D62 (17)). The `reward_and_length/v1` stance rows of `docs/SPECIES_CATALOG.md` still
  read "3 consecutive passes".
- `n-envs-outside-recipe-digest`: **open** (`config.hyperparameters_sha256` reads no `n_envs`).
- `eval-protocol-provenance-mismatch`: **open**. `result_bundle/provenance.py` still stamps one episode count on
  every evaluation role.

#### 3.26 The offline analysis scripts are hidden and break on several species and roles

- `fu-offline-analysis-tools-species-coverage-offline-tools-not-surfaced-in-report-section`,
  `fu-offline-analysis-tools-species-coverage-report-scripts-legacy-stage-int-unchecked`,
  `fu-offline-analysis-tools-species-coverage-action-bound-model-mode-measures-clipped-actions`,
  `fu-offline-analysis-tools-species-coverage-robot-hardware-tools-misdrive-or-skip`: **open**.
  - The scripts take legacy numbers 1 and 2 only. A T. rex or Compsognathus walk lives in `03_locomotion`.
  - `scripts/action_bound_report.py` still prints "Raw (pre-clip)" for `predict()` output.
  - Section 8 points to no tool.
- `fu-offline-analysis-tools-species-coverage-allosaurus-torque-plant-in-servo-reports`: see 3.17.
- `fu-offline-analysis-tools-species-coverage-recovery-deliverable-no-offline-analysis`: see 3.24.

#### 3.27 No whole-chain view, TensorBoard cell or replicate launcher

- `no-chain-or-optimizer-overview`, `no-replicate-or-sweep-orchestration`: **open**.

#### 3.28 Knobs are dense, dropdowns allow invalid choices, and validation comes late

- `knob-density-and-dropdowns`: **open**. Cell 6 still offers the `SPECIES` × `EMBODIMENT` pairs, and `ALGORITHM`,
  `QUICK_TEST` and `AUTO_DISCONNECT` still have no `@param`. Cells 12 and 13 still say "all six". `BEHAVIOR` no
  longer offers `"hunt"` (D-D59).
- `knobs-validated-after-write`: **open**. `29cd504` refuses, before anything is written, a `RUN_ID` of another tree
  and a certification run's `TRUNK_FROM` into a screen or quick-test tree. An invalid `BEHAVIOR` or `RETRAIN_FROM` is
  still refused only in the resolve cell, after the storage cell wrote the run directory and `provenance.json`.
- `velociraptor-critic-phase-unsupported-on-raptor`: **open**. Only the T. rex and anatomical Compsognathus envs
  define `stance_width_settle_steps`.

#### 3.29 Spurious warnings and "Stage N" wording

- `probe-cutoffs-none-warning`, `probe-filter-none-warning`: **open**. `reporting/stage_artifacts.py` still passes an
  absent key's `None` to `_probe_cutoffs`, which warns.
- `stage-number-wording`: **open**. The stage summary is still titled "Stage N Summary", and the older figures'
  legends read "Stage N".

#### 3.30 Config comments and docs state values that no longer hold

- `brachiosaurus-hunt-forward-ramp-misdescribed`: **moot**. The hunt TOMLs are deleted. No committed stage sets
  `ramp_start_value` since the walk's forward-velocity ramp gave way to the cruise ramp (D-D62 (5)).
- `brachiosaurus-stale-recipe-comments`: **open**. `configs/brachiosaurus_altithorax/anatomical/locomotion.toml`'s
  `healthy_z_range` comment still says the settled stance is ~1.13 m, and the stance's `[sac]` lines still say
  "Match PPO".
- `compsognathus-recipe-review-protocol-stale`: **open**. `environments/compsognathus_longipes/TRAINING_RECIPE_REVIEW.md`
  has had only display-name edits since the review (`a7ecc79`), and both anatomical TOMLs and the README still
  cite it.
- `compsognathus-robot-stranded-chain-still-listed-certified`: **open**. `docs/ROADMAP.md` gained a dated update
  (`12500d7`): the certified nodes it names no longer load after note 24. Its "Current focus" paragraph and its
  Compsognathus row ("both have certified stance and walk nodes") still say otherwise.
- `cmp-stale-hunt-default-docs`: **addressed** (`website/docs/getting-started/quick-start.md` and
  `website/docs/getting-started/installation.md`). `475606d` corrected the website's `BEHAVIOR` default on both
  pages. `fff098f` added `allosaurus_fragilis` to the quick-start, and `527aa8b` removed the hunt from both.

### Sections 1, 4, 5 and 6

- Sections 1, 4 and 6 describe the tree and the quick-test runs of 2026-11-22. Note 24 strands every checkpoint
  they describe, and every hunt row describes a node that no longer exists.
- Section 5's list of deliberate differences has moved:
  - The hunt gate entries are moot (D-D59).
  - Every model now ends at walk.
  - The T. rex walk's backstop has its peak warm-up (D-D62 (15)).
  - No committed stage counts a success termination as a fall.
  - A SAC resume in place is refused (`76b27b2`).
  - The rest stands as the review describes it: the manual RESUME knob, the CU-15 quarantine, the per-frame CSV on
    stance and recovery only, the recorder gap of D-D46 and `evaluate()`'s seed 2000.

### Counts

Of the 141 findings:

- 11 are addressed;
- 20 are moot, most of them by the hunt's removal (D-D59) and the rest by `94916cd` and D-D62;
- 110 are open.

The open ones are mostly the notebook's own presentation and Colab handling (themes 3.2, 3.4-3.7, 3.22 and 3.28),
the older figures and the training-history defaults (3.11 and 3.23), SAC (3.13), and the per-species tooling and
docs (3.16, 3.17, 3.24, 3.26 and 3.30). They are not yet folded into `docs/KNOWN_ISSUES.md`, whose review table
points here.

## Folded on 2026-11-26

*Appended 2026-11-26 (decision D-D65 (4)). Every finding was re-checked on HEAD `9784a48`, which adds `d7d53cc`
(D-D63) and `9784a48` (D-D64) to the tree the status above was written on; D-D65's collapse floors (its item (1))
touch no finding. From here this document is a findings record, and
[`docs/KNOWN_ISSUES.md`](../KNOWN_ISSUES.md) tracks the open items: the PR that fixes one deletes or amends its
entry there.*

**Counts.** No status changed: of the 141 findings, 11 are addressed, 20 moot and 110 open. The 110 open ones are
carried by 52 new KNOWN_ISSUES entries and 8 amendments to existing entries, and every entry names the ids it
carries in its closing "(2026-11 notebook review: ...)", so a search of KNOWN_ISSUES for an id finds its entry. By
the entry that carries it, 1 id is HIGH, 31 MEDIUM, 76 LOW and 2 unrated (the Post-training artifacts list has no
severities); 38 sit under Notebooks, 33 under Infrastructure, 21 under Training / RL, 15 under Configs, docs &
website, 2 under Post-training artifacts and 1 under Testing / CI. Findings that share one fix share an entry; three
ids are split across two entries (`figure-labels-units`, `collected-results-dead-and-kind-dependent-columns` and
`brachiosaurus-stale-recipe-comments`); each is counted once, so
`collected-results-dead-and-kind-dependent-columns`, whose Drive half sits under Notebooks, counts under
Infrastructure. The amendments:

- the HIGH stance-gate entry takes `dibothrosuchus-three-legged-stance-certified-unflagged-r3`;
- the MEDIUM `stance_quality/v2` Compsognathus entry takes `compsognathus-manual-liftoff-check-not-automated`;
- the MEDIUM heading entry takes `cmp-allosaurus-heading-probe-on-heading-free-plant`;
- the MEDIUM (perf) `EvalCallback` entry is replaced by the `eval-cadence-dominates-wallclock` entry;
- the MEDIUM action-saturation entry's metric-hazard paragraph takes
  `fu-offline-analysis-tools-species-coverage-action-bound-model-mode-measures-clipped-actions`;
- the MEDIUM (operational) run-level-records entry takes `no-republish-cell`;
- Post-training artifacts item 2 takes `cli-notebook-judging-parity`;
- the LOW NB9 SAC entry (Notebooks) is replaced by the entry carrying `sac-path-second-class`,
  `sac-path-second-class-unflagged` and `fu-sac-artifact-path-trex-recovery-sac-recipe-unrecorded`.

**What the re-check corrected in the status above.** None of these moves a status:

- `no-colab-cap-awareness-or-auto-resume`: a resumed stage does not record "the resumed session's duration only".
  `train_base.train` sums the durations of the sessions that reach a final save (`train_base.py:2496-2498`,
  `:2531-2533`, `:2571`, D-A15); a session the Colab cap cuts off before its final save records none, so a capped
  segment still drops out.
- `knobs-validated-after-write`: besides `BEHAVIOR` and `RETRAIN_FROM`, a pinned `TRUNK_FROM` that is this run, not
  a run, or another model's is still refused only after the storage cell wrote the run directory (cell 8,
  `notebooks/sb3_training.ipynb:480-505`).
- `cmp-catalog-gate-semantics-by-kind`: four catalog rows read "3 consecutive passes", not two: the Compsognathus
  robot's and the Allosaurus' `stance_quality/v1` rows too (`docs/SPECIES_CATALOG.md:183`, `:214`).
- `compsognathus-recipe-review-protocol-stale`: all four Compsognathus stance and walk TOMLs cite the recipe review,
  the robot's included.
- `eval-protocol-provenance-mismatch`: the walk's in-training evaluations are paired since D-D62 (the evaluation env
  is reseeded each time); the stance and recovery ones are not.
- `brachiosaurus-stale-recipe-comments`: the statue values are re-measured on physics r7
  (`configs/brachiosaurus_altithorax/anatomical/stance.toml:68-86`); the height and `[sac]` "Match PPO" comments
  remain.
- `results-readme-and-strict-cell-drift`: since D-D63 retired result schema 2, `results/README.md:32-35` crediting
  schema 2 is stale as well.
- `supplementary-log-key-mismatch`: `evaluate()` also never prints the heading alignment, since it tests a key the
  aggregate never holds (`evaluation.py:857`).
- `success-terminations-counted-as-falls` stays moot, but two stale sentences say the Allosaurus walk ends on
  `course_complete` (`docs/TRAINING_ARTIFACT_PRESENTATION.md:938-946` and the `DiagnosticsCallback` docstring,
  `environments/shared/diagnostics.py:97-100`); the artifact-guide entry carries them.
- The fix of 3.21 names `promote_result_bundle`, which never existed; the promotion entry asks for a tool.

**Sections 1, 2, 4, 5 and 6.** The open part of each ranked fix (section 2) is carried by its theme's entries:
fix 2 by 3.2, 3.16, 3.19 and 3.22; fix 3 by 3.3; fix 5 by 3.4; fix 7 by 3.12; fix 8 by 3.14 and 3.17; fix 9 by
3.5, 3.6 and 3.7; fix 10's per-step CSV by 3.18; the smaller wins by 3.13, 3.22, 3.28, 3.29 and 3.30. Fixes 1, 4 and
6 and the getting-started docs are addressed or moot. Sections 1 and 4 restate the themes' findings. Section 6's two
observations without an id are not known issues: the T. rex recovery's ~9 env steps/s, which the review called
unverified, is not the env's (one T. rex env steps 697 times a second on the recovery, 637 on the stance and 675
on the walk, single process, 2026-11-26), and the manual cell's "contiguous curriculum prefix" refusal is by design
(`reporting/bundles.py:289-300`). Section 5's deliberate differences stay deliberate; what it asked to change is
carried by the entries of 3.5, 3.16, 3.17, 3.18 and 3.23.
