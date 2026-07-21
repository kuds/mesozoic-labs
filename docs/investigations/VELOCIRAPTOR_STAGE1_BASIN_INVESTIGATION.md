# Velociraptor Stage-1 Backward-Drift Basin

- **Date:** 2026-07-20
- **Scope:** Velociraptor PPO Stage 1, result-bundle integrity, home-keyframe
  action residuals, and natural-lean posture shaping
- **Status:** Commit A is implemented and evaluated. Commit B is implemented;
  its fresh A+B training run is still pending.

This is the append-only investigation record for the Stage-1 balance failure
that remained after the July model and optimizer work. It records the evidence,
the two isolated changes, the failed-but-informative run
`20260720_203454`, and the decision rules for the next experiment.

## Executive summary

The original failure was not one problem:

1. **Action zero did not mean “stand.”** It commanded actuator-range
   midpoints, even though the XML already defined a stable, non-midpoint
   `home` control pose. A 50-seed physics probe showed that this interface
   alone drove the model into a backward-drift/fall basin. Commit A changed
   Velociraptor actions to residuals around the XML home controls.
2. **After that correction, the Stage-1 reward still preferred the wrong
   posture.** The A-only PPO run moved forward instead of backward, but learned
   to straighten from the supported natural pitch of 0.35 rad (about 20°)
   toward world vertical. Its posture penalty improved while survival did not.
   Commit B therefore centres Velociraptor posture shaping on its natural
   forward lean without changing absolute-tilt safety limits.

The sequence matters. Commit A was evaluated before Commit B so its effect
could be identified. Reset noise, XML physics, termination thresholds, and
Stage-1 hyperparameters were held fixed. The next run must likewise be a fresh
PPO Stage-1 run with A+B and `reset_noise_scale = 0.05`; it must not resume the
A-only policy or advance to Stage 2 unless the survival gate passes.

## 1. Why evidence integrity came first

The schema, provenance, and exporter work was a prerequisite for interpreting
this experiment, not bookkeeping after it. The canonical bundle contract is
documented in [RESULT_BUNDLES.md](../RESULT_BUNDLES.md).

Two validation levels are intentionally different:

- **Drive validation** preserves partial and gate-failed runs as valid
  evidence.
- **Promotion validation** accepts only complete, internally consistent,
  promotable three-stage results.

Run `20260720_203454` demonstrates why that distinction is useful: it failed
Stage 1, was correctly labelled `failed`, and still retained enough evidence
to diagnose the next change. It must not be published into `results/`, the
generated species catalog, or the website.

Runtime provenance binds the evidence to:

- the exact repository commit and dirty/patch state;
- Python and dependency versions;
- training and deterministic evaluation seed roles;
- plant identity and resolved stage configuration;
- the selected checkpoint and matching SB3 VecNormalize hashes; and
- separate selected-policy and terminal-policy episode evidence.

Exporter tests protect relationships, not only JSON shape. They verify
summary/CSV/provenance/manifest identity agreement, frozen-gate recomputation,
selected and terminal evidence aggregation, mixed-plant rejection, tamper and
unlisted-artifact detection, write-free idempotence, failure preservation,
credential sanitization, and non-mutating legacy audits.

### Colab and Google Drive operating rules

For every causal experiment:

1. Start from a clean Colab runtime and install the exact intended commit.
2. Use a new run ID; never mix stages or artifacts from different code states.
3. Let the notebook export directly to its unique Drive run directory.
4. For a complete run, treat `summary.json` as canonical and generated
   CSV/documentation as derivatives. A partial or failed Stage-1 run correctly
   has no summary; audit its provenance, manifest, stage config, and raw
   evaluation evidence without synthesizing missing stages. Never manually
   copy metrics between formats.
5. Use `google_drive_summary.ipynb` to audit or migrate the bundle, and run
   strict promotion validation only after all three stages legitimately pass.

A failed experiment is valuable when it is attributable. A high-scoring run
without exact code, configuration, seeds, plant identity, and raw evaluation
episodes is not equivalent evidence.

## 2. Pre-Commit-A physics probe

Before spending another training budget, a 50-seed, 1,000-step physics probe
compared the old policy-neutral command with the XML standing command.

| Command | Reset noise | 1,000-step survivors | Observation |
|---|---:|---:|---|
| Old zero action (actuator midpoints) | 0.00 | 0/50 | Mean failure near step 167; backward drift |
| XML `home` controls | up to 0.03 | 50/50 | Stable in every seed |
| XML `home` controls | 0.05 | 48/50 | Stable in 96% of seeds |

This was a local exploratory probe against the pre-Commit-A mapping (the
parent of `c8af06d`), not a canonical Drive bundle. It used 50 seeded resets
and a 1,000-command horizon, but its script, exact seed list, per-seed output,
and explicit fall criterion were not retained. The aggregate is recorded as
decision evidence, not publication evidence. Any future reliance on the exact
rates should first reproduce the probe with a checked-in script that captures
the commit, seeds, command path, reset noise, horizon, and termination rule.

This isolated the action interface from reset noise and learning. Lowering
`reset_noise_scale` first would have hidden part of the symptom while leaving
zero action mechanically inconsistent with the standing pose.

The probe also supplied a useful priority rule: test the passive/nominal
control semantics before changing rewards, optimizer settings, or curriculum
noise. Cheap physics probes can reject an entire family of expensive training
experiments.

## 3. Commit A — home-keyframe residual actions

[Commit A (`c8af06d`)](https://github.com/kuds/mesozoic-labs/commit/c8af06deefbf878733dec915fd2051b85163bafe)
changed only the Velociraptor policy interface. For each actuator, the
piecewise-linear mapping preserves three anchors:

```text
action -1  -> actuator minimum
action  0  -> XML home control
action +1  -> actuator maximum
```

A single affine midpoint mapping cannot generally preserve those anchors
because the standing control is not the midpoint of the actuator range.
Piecewise residual control is standard in legged-robot learning because the
policy learns deviations from a viable nominal pose rather than having to
rediscover the pose while it is also learning balance.

Important compatibility details:

- Energy and smoothness remain defined in normalized residual-action space;
  zero residual has zero energy cost and no smoothness cost when unchanged.
- Reset, CPU evaluation, MJX training, JAX CPU evaluation, and video rollout
  all resolve the same named `home` keyframe.
- This is a breaking policy-interface change from Velociraptor revision 1 to
  revision 2. Pre-revision-2 PPO, SAC, and JAX checkpoints are historical and
  cannot be resumed as current policies even though the action dimension did
  not change.
- The XML, compiled physics revision, visual revision, actuator limits, and
  reset noise did not change.

## 4. A-only PPO run `20260720_203454`

The canonical Drive bundle records a clean run at repository merge commit
[`3695f4d`](https://github.com/kuds/mesozoic-labs/commit/3695f4d186446d57301c6150cb94c9ba103fb751),
Velociraptor policy-interface revision 2, result schema 2, and `failed` status.
It evaluated Commit A without Commit B.

### 4.1 Outcome

| Metric | A-only result | Stage-1 requirement |
|---|---:|---:|
| Training steps | 1.1M / 6M (early-stopped) | Up to 6M |
| Wall time | about 29m 09s | — |
| Selected eval reward | 149.22 ± 28.76 | mean ≥ 100 |
| Selected eval episode length | 128.4 ± 19.1 | mean ≥ 750 |
| Selected eval forward velocity | 0.18 ± 0.15 m/s | diagnostic |
| Selected eval distance | 0.774 m mean | diagnostic |
| Episodes reaching 750 steps | 0/30 | survival gate not met |
| Episode-length range | 88–174 | — |

The reward threshold passed but the balance threshold failed decisively, so
stopping before Stage 2 was correct.

### 4.2 Why the raw reward peak was misleading

The raw 50k checkpoint peak was `261.79 ± 261.72`, but its robust score
(`mean - standard deviation`) was only `0.07`. Around 650k and 700k, isolated
1-of-30 episodes reached 1,000 steps; the other 29 episodes still failed. Those
rare survivors created variance, not repeatable balance.

For balance tasks, mean reward alone can be dominated by a few long episodes.
The survival distribution, robust score, and termination mix must be reviewed
before declaring improvement.

### 4.3 What Commit A improved

Training forward velocity moved from roughly `-0.234 m/s` early to
`+0.026 m/s` late, and the selected evaluation averaged `+0.18 m/s`. This is
consistent with making action zero a viable standing command weakening the
catastrophic backward-drift basin; the within-run trend is not by itself a
controlled A/B comparison.

Commit A was therefore a partial success, not a failed hypothesis. It fixed
the action semantics but exposed a separate balance incentive conflict.

### 4.4 Failure anatomy

The raptor's natural pitch is `0.35 rad` (about 20.1°), with a natural
forward-axis vertical component of `-sin(0.35) ≈ -0.343`. During training:

- mean tilt moved from about 30.6° toward 15.8°, overshooting the supported
  lean toward world vertical;
- `forward_z` moved from about `-0.398` to `-0.174`, away from the natural
  target;
- posture reward improved from about `-0.457` to `-0.139`; and
- survival remained near 1.3 seconds.

The old posture term was functioning exactly as coded: it penalized absolute
tilt from vertical. PPO collected that reward by straightening the torso even
though the XML's tail-counterbalanced standing geometry was designed around a
forward lean.

Late termination proportions reinforce the balance diagnosis:

| Termination | Late-run share |
|---|---:|
| Fallen | about 39.2% |
| Tail contact | about 37.0% |
| Body contact | about 16.4% |
| Excessive tilt | about 7.1% |
| Nosedive | about 0.3% |

The failure was not predominantly a nosedive and was not an immediate reset
failure.

### 4.5 Hypotheses ruled down

**Optimizer instability was not the leading cause.** PPO's late diagnostics
were healthy: approximate KL was about 0.016, explained variance about 0.96,
and value loss about 0.063. The policy learned the provided objective; the
objective's posture geometry was wrong for this morphology.

**Reset noise was not the leading cause.** The XML home command survived
48/50 physics probes at the unchanged 0.05 noise, and learned-policy failures
were not concentrated at reset. Noise reduction remains a later controlled
experiment only if A+B is stable at zero/small noise but not at 0.05.

**More training was not justified.** The policy was becoming more upright
while the balance distribution remained catastrophic. Continuing the same
objective would spend compute reinforcing the conflict.

## 5. Commit B — natural-lean posture shaping

Commit B changes the Velociraptor posture reward from deviation from world
vertical to deviation from its direction-aware natural lean.

### 5.1 Reward geometry

For pelvis quaternion `[w, x, y, z]`, use the world-Z components of the body's
forward, lateral, and up axes:

```text
orientation = [2(xz - wy), 2(yz + wx), 1 - 2(x² + y²)]
target      = [-sin(natural_pitch), 0, cos(natural_pitch)]
alignment   = dot(orientation, target)

normalized_error = clip(
    (1 - alignment) / (1 - cos(max_tilt_angle)), 0, 1
)
posture_reward = -posture_weight * normalized_error
```

This construction is:

- **direction-aware** — correct forward pitch differs from backward pitch;
- **roll-aware** — lateral tilt reduces alignment;
- **yaw-invariant** — turning on the ground does not change the posture score;
  and
- **JAX-safe at the target** — normalized squared chord distance avoids the
  undefined `acos` gradient at exact alignment while retaining local
  quadratic behavior.

Penalizing `abs(tilt) - natural_pitch` would not meet the contract: it could
reward the wrong pitch direction or roll at the same absolute angle.

### 5.2 Safety semantics stay separate

The reward helper still returns absolute tilt relative to world up for
termination and historical diagnostics. Commit B does not change:

- `raptor.xml` or any physics/visual revision;
- `reset_noise_scale = 0.05`;
- the absolute-tilt termination threshold;
- the existing rounded MJX nosedive baseline (`-0.342`) or termination
  boundary; or
- Stage-1 TOML weights and PPO hyperparameters.

It also does not change the action/observation interface or the plant identity;
Velociraptor remains at policy-interface revision 2. Reward semantics are
identified by exact repository and resolved-config provenance rather than a
plant revision, and still require a fresh training run.

The exact reward-only target is kept separate from that rounded nosedive
baseline, so an isolated reward experiment cannot silently move a safety
boundary.

With the Stage-1 `posture_weight = 1.5` and `max_tilt_angle = 1.047`, the XML
home-pose penalty changes from `-0.166729` to approximately zero
(`-0.0000013` in the numeric probe). A world-upright pose, which formerly
incurred zero posture penalty, now costs `-0.181944`. This reverses the
incentive identified in the A-only run without changing the maximum weight or
gate.

### 5.3 Targeted backend and evaluation consistency

Reward correctness includes every consumer, not only the main training step.
The natural-lean target is applied consistently in:

- the SB3/Gymnasium environment;
- direct MJX environment stepping;
- JAX total-reward and detailed-component functions;
- JAX CPU evaluation and diagnostics; and
- custom `natural_pitch` overrides resolved at runtime.

Velociraptor alone opts into the lean-aware target. T-Rex and Brachiosaurus
retain their vertical posture reward. The JAX notebook binds reward functions
only after `create_env` has resolved the live MJX configuration; regression
tests pin this ordering. Binding earlier would capture stale registry defaults
and make notebook behavior disagree with direct environment creation.

This coverage pins the lean primitive and its routing through the relevant
paths. It is not yet the comprehensive, fixed-state, all-component SB3↔JAX
parity test retained in [KNOWN_ISSUES.md](../KNOWN_ISSUES.md).

## 6. Next experiment and decision rules

### Immediate run

After Commit B and the documentation/version follow-up are merged:

1. Start a fresh Colab runtime on the exact merged commit and reinstall the
   package.
2. Run `sb3_training.ipynb` with Velociraptor + PPO, Stage 1 from scratch, and
   a new run ID.
3. Keep the same Stage-1 TOML, seeds/evaluation protocol, 6M-step budget, and
   `reset_noise_scale = 0.05`.
4. Do not resume `20260720_203454`; it was optimized under the old reward and
   would weaken the A+B comparison.
5. Review the canonical bundle, not copied headline metrics.

The configured training gate remains mean reward at least 100 and mean episode
length at least 750 for three consecutive evaluations. A single high-return or
1,000-step episode is not a pass.

A-only and A+B total rewards are not directly comparable even though the
numeric weight and gate are unchanged: the home-pose posture baseline moves by
about 0.167 reward per step. Use episode-length distributions, survival rate,
and termination causes as the primary causal comparison; reward is the
configured gate, not an invariant cross-reward score.

### What happens after the run

- **If Stage 1 passes robustly:** continue PPO through Stages 2 and 3 using the
  new revision-2 Stage-1 checkpoint. Those stages need current checkpoints
  because revision-1 action semantics are incompatible.
- **If posture tracks the natural target but survival still fails:** inspect
  contact/support geometry and the termination distribution before changing
  optimizer settings.
- **If A+B is stable at zero or small reset noise but fails at 0.05:** then run
  an isolated reset-noise curriculum experiment (for example 0.02–0.03). Do
  not lower noise pre-emptively.
- **If survival passes but reward fails:** inspect component balance and the
  frozen gate; do not infer another physics defect from reward alone.
- **Do not run SAC merely to bypass this test.** Validate the environment and
  reward geometry with PPO first, then rerun SAC on the current policy
  interface if a current SAC result is desired.
- **The JAX natural-lean paths are already updated.** Defer expensive JAX
  retraining until the SB3 PPO experiment validates the morphology fix; any
  revision-1 JAX checkpoint remains historical.

## 7. General lessons

1. **Define neutral action semantically.** In a legged robot, zero should map
   to a viable nominal command, not an arbitrary actuator midpoint.
2. **Match rewards to morphology.** “Upright” is not synonymous with “world
   vertical” for a tail-counterbalanced biped.
3. **Probe physics before training.** A small seeded control test can identify
   action/reset defects more cheaply and clearly than millions of RL steps.
4. **Change one causal layer at a time.** Action mapping, reward target, reset
   noise, XML, termination, and optimizer settings should not move together.
5. **Use distributions, not seductive peaks.** Robust scores, episode-length
   distributions, and termination causes expose rare-survivor artifacts.
6. **Healthy PPO metrics do not prove a healthy objective.** An optimizer can
   efficiently learn the wrong reward geometry.
7. **Keep shaping separate from safety.** A reward target may change without
   moving absolute termination or nosedive boundaries.
8. **Parity includes evaluation and diagnostics.** A correct training reward
   is insufficient if checkpoint selection, CPU evaluation, or plots use
   different semantics.
9. **Version policy semantics, not just dimensions.** Identical tensor shapes
   do not make checkpoints compatible after action meaning changes.
10. **Instrument before experimenting.** A correctly failed, immutable bundle
    is more actionable than an unverifiable success claim.

## 8. Evidence and artifacts

- [Canonical run bundle](https://drive.google.com/drive/folders/1lH449rXldKQ0Gms0pOY06Map-nAw9a_V)
- [Stage-1 artifacts](https://drive.google.com/drive/folders/1giECOwZvN8V7F0mLwk3iyEHiemZxzpj4)
- [Training curves](https://drive.google.com/file/d/1KRF998INQb--RN678eS_c3dbUf0v8QDQ/view?usp=drivesdk)
- [Locomotion health](https://drive.google.com/file/d/1U3XaCQ_pJYZneiO1zVZ65zd_qbV_5FCF/view?usp=drivesdk)
- [Behavioral metrics](https://drive.google.com/file/d/1K_w-ZQ1ZfHAhwA3uJ066yPvVWNuR3fvm/view?usp=drivesdk)
- [Result-bundle contract](../RESULT_BUNDLES.md)
- [Plant/policy compatibility contract](../PLANT_CONTRACT.md)
- [Velociraptor Stage-1 config](../../configs/velociraptor/stage1_balance.toml)
- [Velociraptor environment](../../environments/velociraptor/envs/raptor_env.py)

## 9. Follow-up outcome (append only)

Pending a fresh A+B PPO Stage-1 run. Append its run ID, exact commit, selected
and terminal evaluation distributions, gate outcome, termination mix, and the
next decision here. Do not rewrite the A-only evidence above.
