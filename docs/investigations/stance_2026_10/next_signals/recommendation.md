# T. rex stance: what to log and sense next

Code citations are relative to `main_ro` = `scratchpad/fix/main_ro` (origin/main 2e734e29, read-only). Evidence files are under `NS` = `scratchpad/foot/next_signals`. Nothing was trained. The certificates quoted below were run offline and are report-only.

## 1. How close we are

**In behaviour we are very close. In certification we are not.**

**What passes.** On main's physics-r9 plant, the stage's own six-heading certificate passes 2 of 3 r9 arm L seeds at 6.29M steps:

- s60: 40/40, lower bound 0.928.
- s52: 38/40, lower bound 0.851.
- s42: fails at 29/40 (0.586), on pad CoP, sole tilt and corner lift.

The reward-selected robust_best checkpoints also pass: s60 at 524k scores 40/40, s42 at 655k scores 40/40, and s52 at 918k scores 39/40 (NS/howclose/offline_certificates.txt).

**Where it is thin:**

- **Margin.** The forward pad CoP is the bar that binds (0.80, stance.toml:732). s60's worst episode is 0.780, against the statue's worst of 0.475 (NS/howclose/cert_margins.txt). s52 fails on seeds 3050 (0.997) and 3063 (0.823).
- **Fresh seeds.** On seeds 7042-7081, s60 scores 39/40 and s52 38/40. Both fall on seed 7047, where the statue stands, which uses up the whole one-fall allowance (stance.toml:780).
- **Noise.** Under joint-space noise there were 8/60 falls at 6M, against the screen bar of 7 (NS/howclose/evidence_table.txt:56).
- **Checkpoints.** s60 was clean on 40, 31, 1 and 40 seeds at 3, 4, 5 and 6M (NS/certify/screen_vs_panel.txt).
- **Seeds.** With 3 seeds, the 95% interval for a 2/3 per-seed pass rate is [0.09, 0.93] (NS/diagnose/notes_numbers.txt).
- **Statue-like passes.** The zero-action statue passes the certificate itself at 39/40 (stance.toml:645-648). The robust_best pairs earn 1.006-1.010 of the statue's reward.

**What "certified" requires (stance.toml):**

- **Gate** stance_quality_heading/v1 (513), on seeds 3042-3081 at six headings:
  - every heading passes v2 against the statue;
  - the seeds that are clean at all six headings reach a lower bound of at least 0.80, so 37/40 passes and 36/40 fails;
  - full horizon at least 0.95 (667), at most 1 hop-or-fall (780), and reward at least 0.60 of the statue's (888).
- **Seeds.** certification_seeds = 2 (902): two distinct-seed runs under one committed config.
- **Judge.** It binds the handoff pair, robust_best and then best (environments/shared/curriculum/checkpoints.py:108-122). It refuses a report rolled on another task (environments/shared/reporting/gates.py:701-711). required_consecutive = 3 is only scheduler hysteresis (895).

**Why none of this counts yet.**

- The passing runs used a scratch recipe, MirrorKLPPO plus experiment keys, recorded as task a2af3c5c. stance.toml does not enable that recipe.
- Their robust_best checkpoints were picked under the scratch reward, with 10-episode evaluations every 131k steps.
- The only production-config r9 run scored 1/40 (docs/KNOWN_ISSUES.md:154-156).

So we have **0 of the 2 certifying seeds** we need. Single checkpoints pass the panel. Robustness does not: the CoP margin is thin, the fall allowance is spent on fresh seeds, and noise causes falls. Reproducibility does not either, across seeds, across checkpoints, or under the committed config.

## 2. Recommended next signals

**None of this can be added to the walking pilot that is training now.** Callbacks are assembled once, before learn() (environments/shared/train_base.py:1408). The pilot branch also predates #626.

| # | Item | What it answers | Cost | Approval |
|---|---|---|---|---|
| **Report-only telemetry (no digest moves)** | | | | |
| 1 | **Per-episode eval ledger and in-training v2 clean count.** One row per eval episode: return, length, cause, signed per-foot CoP, tilt, corner lift and spawn splay. Also a per-eval clean count and lower bound, computed from the trace the recorder already returns (environments/shared/eval_diagnostics.py:596) with `apply_statue_criteria=False` (environments/shared/curriculum/stance_gate_v2.py:629-637). Today only full-horizon means are kept (eval_diagnostics.py:1077-1156). | Which of a run's checkpoints would pass on the gate's own bars. Today's only in-training signal is a veto: 0/7 cells with over-bar > 0 passed, against 5/17 with over-bar = 0 (NS/certify/screen_vs_panel.txt). | 1-2 days; no extra simulation | None. Keep these scalars out of the history a stance_quality/v2 stage reads (gates.py:118-121). |
| 2 | **Whole-body balance.** CoM fore-aft offset from the foot midpoint, trunk pitch, the extrapolated-CoM margin to the toe line, the mirror-even (L+R)/2 leg DC and achieved-minus-home joint angles, all read from mjData. Today the CoM offset is lateral only and computed only with the lean term on (environments/tyrannosaurus_rex/base_env.py:691-702, 1147), and the leg DC is L-R only (eval_diagnostics.py:155-186). | Separates a body lean from foot slide or ankle plantarflexion, and gives time to fall. In a scripted probe the extrapolated CoM crossed the toe line 1.60-1.68 s before each of 3 nosedives and in none of 9 stands (NS/sensing_robotics/probe/probe_summary.txt). | ~1 day | None |
| 3 | **Fixed-state-bank probe at every PPO update.** Per-actuator std, V(s) at the statue and at held leans, the deterministic action and its gains, per-dimension VecNormalize, and ent_coef. Today only the mean std is logged (environments/shared/mirror_kl_ppo.py:418-419), VecNormalize is one mean (environments/shared/diagnostics.py:349-352), and ent_coef is set but not logged (environments/shared/curriculum/schedules.py:215). | When the lean starts, at 8,192-step resolution; whether a change is a weight change or a normaliser shift; whether the critic values a lean. The mean std hides a 9-13x spread across actuators (NS/diagnose/lag_r9armL.json, from MirrorKLPPO runs). | ~2 days | None. It must touch no RNG; prove bit-identity. |
| 4 | **Training-rollout episode ledger (stochastic policy).** Cause, length, fore-aft CoM and the touch-share CoP proxy per episode, plus fall counts in diagnostics.npz, which today holds only shares (diagnostics.py:450-453; counts go to TensorBoard only, :285). | Whether noisy-policy falls push the deterministic policy back. On s60, 15-17% of episodes ended early at 4.72-5.24M with the CoM at +34 to +40 mm, falling to 0% at -2.4 mm (NS/questions/r9_armL_training_rollouts_by_quarterM.txt). | ~1 day | None. Read mjData in a callback rather than adding info keys, which the digest snapshot captures (docs/PLANT_CONTRACT.md:83-84). |
| 5 | **Force telemetry.** Per-actuator torque as a fraction of forcerange, ankle torque over load, fore-aft shear, and friction-cone use. The recorder decodes normal forces only (environments/shared/gait/recorder.py:36-47); actuator_force is read only by hand-run scripts (environments/shared/scripts/actuator_saturation_report.py:65-72). | Whether falls are torque-limited, and what a force sense would carry. In the probe, cone use was 0.042-0.056 standing against 0.235 in nosedives (probe_summary.txt). | 1-2 days | None |
| 6 | **Signed CoP trend advisory and a gate_progress figure.** PostureDrift fires only when the over-bar fraction is at least 0.5 twice (environments/shared/curriculum/posture_watch.py:13-21), and nothing plots gate_progress.npz. | Drift direction and timing. On s60, PostureDrift would first fire at 4.85M, although the panel had already failed at 4.19M (NS/questions/r9_armL_reward_selection_and_s60_eval_series.txt). | 0.5-1 day | None |
| 7 | **Small fixes.** Add reward_action_jerk to the training reward keys (diagnostics.py:94-136; the env emits it at base_env.py:1215). Split term_* into full-horizon episodes and falls (eval_diagnostics.py:686-694). Add the handoff's step and score to the certificate (step at environments/shared/curriculum/progress_report.py:452). | Bookkeeping | Hours | None |
| **Evaluation and selection** | | | | |
| 8 | **Keep the onset weights.** Only the newest 5 checkpoints are kept today (stance.toml:914). Add an archive rule in code, or set max_checkpoints = 0; that config edit moves the snapshot line at configs/digest_snapshot.generated.txt:386, so regenerate it. | It is the prerequisite for any offline study of when the lean starts. | ~0.5 GB per run | PR review only |
| 9 | **Offline battery on the existing r9 checkpoints:** 5042 and 9042 panels for s60/s52 at 6M and for the robust_best pairs; a replay of seed 7047; push-and-release; an own-sigma panel; what the policy senses on its own states. r10 checkpoints run only on the #623 tree, because physics validation refuses them on main (environments/shared/plant_contract/validation.py:56-62). | Whether today's passes are robust or sitting on the rail. | 2-4 CPU-h | The user's go-ahead for offline evaluation |
| 10 | **Report-only screened candidate.** The best checkpoint by mean minus std among evaluations whose screen passes, certified at stage end beside robust_best and stage1_final. | Whether reward-only selection is what blocks certification. | ~1 day, plus ~16 min per certificate | Report-only. Using it as the handoff needs the maintainer (gates.py:653-656). |
| 11 | **Robustness side panel** every 500k steps: own sigma plus joint noise, on a separate env with its own Generator. | Noise margin during training. | ~1 day | None |
| 12 | **Sensor-realism panel**, offline: IMU noise and bias, a 1-2 step delay, binary contact. | Which senses the stance relies on. | 2-4 CPU-h | Go-ahead |
| 13 | **Replicates of the committed config**, with items 1-11 in place. | Whether stock PPO certifies on 2 seeds. | 3 x 12M runs | The maintainer schedules them; fix the selection rule first. |
| **Sensing (policy-interface changes)** | | | | |
| 14 | **One r15 to r16 bump:** split the foot channel into pad and digit touch sensors, and add ankle force and the previous action. Every T. rex checkpoint is stranded, including the pilot's parent; widening covers only +3 command dims (environments/shared/scripts/widen_checkpoint.py:442-445). | Lets the policy sense its own CoP. | Retrain stand, recover, walk and hunt | Maintainer, after the pilot verdict and item 9 |
| 15 | **Privileged critic:** CoP and contacts to the value function only. | Whether training can tell centred stances from edge stances. | 200-400 lines, plus training runs | Maintainer (training algorithm) |
| 16 | **No MJCF sensors for logging.** Adding one moves the physics and policy digests (NS/sensing/probe_mjcf_cost.out). Bundle a sensor suite with the next physics bump instead. | | Nothing now | Maintainer (physics) |

**Changes made after the verifier:**

- max_checkpoints does move the digest snapshot, so it needs regenerating.
- The certificate already logs which checkpoint it judged (environments/shared/reporting/stage_artifacts.py:361); only its step and score are missing.
- r10 replays cannot run on main.
- A critic-only input is not shown to leave the policy digest unchanged (environments/shared/plant_contract/policy_layer.py:585-638).
- The std spread is 9-13x, not 5-10x.
- Only 3 of the 11 certificates were cross-checked against the scratch judge.
- For pad CoP, touch shares reach R^2 0.867 with a leave-one-posture-out median of 0.221 half-lengths (NS/sensing/cop_signal_probe.md:11). That is weaker than for whole-foot CoP.
- Main's 40-episode evaluations weaken the spawn-noise selection argument.

## 3. Other considerations

**While the pilot trains:**

- The pilot already keeps every checkpoint.
- Run item 9 if you approve offline evaluation. It is the only item that produces evidence now.
- Build items 1-7 on a branch, with #626's proofs (digest snapshot unchanged, an identical short run), so they are ready for the next run.
- When the pilot is judged, record each fall's step, cause, touchdowns, and the whole-foot CoP and toe margin over its last 50 steps.

**Wait for the pilot's verdict before:**

- items 13-16;
- the handoff-selection change;
- deciding whether stance should certify more than statue-quality standing (docs/investigations/TREX_STANCE_SYMMETRY_AND_FOOT_2026_10.md:919-921).

**Out of scope under the 2026-10-16 rule:** any reward change. That includes a CoP-centring term, re-weighting, the lateral-lean term, and committing arm L's options. Reset noise is a task decision for the maintainer.

**Process:**

- The gate's CoP statistic is unsigned (stance.toml:732), so heel-ward drift fails it too: s52 sat heel-ward in 30 of 48 evaluations (NS/diagnose/signed_pad_cop_r9armL.txt). Log the signed value.
- Never certify several candidates on 3042-3081 and hand off whichever one passes.
- **#623 note:** it is already there. The only comment opens with a correction block, updated 2026-10-17T18:54:09Z, and marks 1.06-1.10 as superseded (NS/diagnose/notes_numbers.txt). No GitHub write is needed.
