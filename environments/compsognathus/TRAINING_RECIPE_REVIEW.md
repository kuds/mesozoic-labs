# Compsognathus Longipes training recipe review

September 8, 2026. Applies separately to the anatomical and robot variants.
Reviewed against repository revision `21dfa4521b0822f0bee60be927893c2099c139c7`
and Stable-Baselines3 2.9.0. This is an experimental recipe update, not a
claim of learned balance, walking, or recovery.

## Evidence and reference choice

Tyrannosaurus Rex is the project's successful reference for the updated
training workflow. Successful older Velociraptor Mongoliensis and
Brachiosaurus Altithorax runs are historical context; their recipes have not
been qualified on the updated code base. Package version alone is insufficient:
compare captured commit, plant identity, task, and gate configuration.

The August 21 Tyrannosaurus Rex run `20260821_142144` trained stance for
11,001,856 steps in 13h 7m 9s and passed its independent stance evaluation.
Its first numerical screening pass was at 6.75M steps, with three consecutive
qualifying evaluations by 6.85M. These are different milestones from the full
training duration. Sources: [stage summary](https://drive.google.com/file/d/1GyFVI-C0Gq90wBz1lszv3sDTu5BdcT4K/view),
[screening history](https://drive.google.com/file/d/1Diq_PTuitJAS8EeYyUDdfzRx7pPXYLw1/view),
and [independent stance report](https://drive.google.com/file/d/1rk90nXWhLwRwfmJVzOiC-y4HBdZ1a8co/view).

The new anatomical Compsognathus run `20260908_140150` was only at 151,552
steps in the September 8, 14:10 UTC snapshot. At 150k, its evaluation mean
length was 44.3 of 1,000 steps, with no full-horizon episodes. Approximate KL
was about 0.215 and clip fraction about 0.70 in the latest recorded update.
That supports investigating update size; it does not diagnose eventual
failure. Sources: [evaluations](https://drive.google.com/file/d/1dDjWD-hCFPlp6FuKOP8Rr8iFvmsN6NWI/view)
and [diagnostics](https://drive.google.com/file/d/1UeJXzemOddzhKv91m8wZ7RwYZoYwKt-v/view).
There is no corresponding full robot training result in this review.

The earlier proposed 6M budget was not established by compatible successful
Compsognathus runs. The revised **11M stance ceiling** gives the experiment
the same sample allowance as the current Tyrannosaurus Rex reference, including
time beyond its first stance pass. It is not a predicted time to learn.
The **notebook's stage cells normally train the full allowance**, then apply
the publication gate; a passing screening evaluation does not stop that
training call. The shared CLI's `curriculum` runner can advance early after
its consecutive gate passes. Budget notebook stance for approximately 11M
steps, allowing for a final partial-rollout overshoot.
Timesteps count aggregate environment interactions across workers; do not
multiply the budget by the number of environments. Wall-clock time depends on
the plant, runtime, evaluation cost, and worker configuration.

## Recipe changes

Both Compsognathus variants receive the same PPO update controls, while keeping
their distinct plants, task thresholds, and checkpoint identities.

| Setting | Original Compsognathus recipe | Revised stance | Revised later stages |
|---|---|---|---|
| Stage allowance | 1M stance; 3M each later stage | 11M | 3M each, still pilots |
| Learning rate | Constant `3e-4` | Linear `3e-5` to `1e-5` | Same |
| KL early stopping | Unset | `target_kl=0.03` | Same |
| Entropy coefficient | Constant `0.005` | `0.005` to zero over 7M | `0.005` to `0.001` over 2M |
| Stage-entry warm-up | Implicit shared defaults | None | Explicit 100k steps, clip `0.02`, entropy `0.005` |
| Network / initial log std | `[128,128]` / `-2` | Preserved | Preserved |
| Rollout / batch / epochs | 1024 / 64 / 5 | Preserved | Preserved |
| Discount / GAE | `0.99` / `0.95` | Preserved | Preserved |

The previous shared warm-up implicitly boosted entropy to `0.02`. Keeping it
at `0.005` avoids that fourfold bonus increase during stage transitions. The
existing PPO warm-up constrains clipping and entropy; its LR scaling option
is SAC-specific. The existing locomotion reward ramp remains in effect.

Budgets and stage-entry settings live in the shared curriculum table, so the
11M ceiling and `0.005` transition entropy also apply to SAC. SAC's optimizer
table is unchanged and its entropy remains automatically tuned; PPO's entropy
decay callback is not used for SAC. This review does not qualify SAC learning.

The PPO sweep ranges now include `3e-5`. Their existing 100k-step trials are
only wiring and early-stability probes: rankings at that budget cannot select
a converged controller or evaluate a 7M entropy schedule.

## What transfers, and what does not

- **KL control:** SB3 documents that clipping alone does not prevent large
  updates. In version 2.9, `target_kl=0.03` stops the remaining optimizer work
  when sampled approximate KL exceeds `1.5 * target_kl` (0.045). It is not a
  hard bound and does not undo earlier updates. Clip fraction measures policy
  probability ratios outside the clipping interval, not actuator saturation.
  See [PPO parameters](https://stable-baselines3.readthedocs.io/en/v2.9.0/modules/ppo.html)
  and [the implementation](https://stable-baselines3.readthedocs.io/en/v2.9.0/_modules/stable_baselines3/ppo/ppo.html).
- **Schedules:** SB3 accepts learning rates expressed in remaining training
  progress; the repository already constructs these schedules and its separate
  entropy callback. The endpoints and duration are project hypotheses informed
  by Tyrannosaurus Rex, not universal PPO prescriptions. See
  [learning-rate schedules](https://stable-baselines3.readthedocs.io/en/v2.9.0/guide/examples.html#learning-rate-schedule).
- **Exploration and capacity:** the PPO paper's MuJoCo experiments do not imply
  that every biped needs a large network or a positive entropy bonus. Preserve
  Compsognathus's smaller network and `log_std_init=-2`; copying the reference's
  default zero log std would increase initial standard deviation about 7.4-fold.
  Entropy decay does not force the learned action standard deviation to zero.
  See [PPO, sections 5–6 and Appendix A](https://arxiv.org/pdf/1707.06347).
- **Physical scale:** Compsognathus acts at 50 Hz while the reference Tyrannosaurus
  Rex environment acts at 100 Hz. Do not copy discount factors, force thresholds,
  noise, joint residual magnitudes, or reward coefficients without checking
  their meaning on this plant. Those settings and both MJCF models are preserved.
- **Stopping:** leave automatic reward-collapse stopping unarmed for this first
  qualification experiment. The shared resolver uses an infinite floor when
  no floor is specified. A statue reward alone cannot calibrate the timing of
  healthy exploration dips. Baseline reporting remains advisory and advancement
  gates remain enforced. Do not copy Tyrannosaurus Rex's measured statue constants.

This changes several update controls together. A better result would support
the combined recipe, not identify which individual change caused it.

## Next full-run protocol

1. Use an updated checkout and a fresh runtime/run directory. Select
   `Compsognathus Longipes`, `ALGORITHM="ppo"`, `N_ENVS=4`, `SEED=42`, and
   `QUICK_TEST=False` in the SB3 notebook. Start with the stance stage. Its
   existing loader reads the revised TOMLs; no additional notebook section is
   needed. Repeat separately for `Compsognathus Longipes (Robot)`.
2. Capture the zero-action baseline and preserve run provenance, stage config,
   evaluation/gate histories, diagnostics, and each policy's matching
   VecNormalize sidecar. Zero action already supports standing; approaching
   its reward is a useful reference, not proof of active recovery. A reward-only
   baseline warning is not the stance gate's verdict.
3. Review at roughly 1M, 3M, 6M, 9M, and the stage end. These are diagnostic
   checkpoints, not new pass/fail deadlines. Compare full-horizon fraction,
   support duty and its bound, reward, action variation, KL, clip fraction,
   and learned standard deviation. Repeated non-finite values or broken
   artifacts require investigation; an early reward dip alone does not.
4. Require the existing stance gate: at least 20 evaluation episodes, at least
   90% full horizon, mean unsupported duty at most 10%, its upper bound at most
   15%, reward at least 1,500, and three consecutive qualifying screening
   evaluations. Require the selected checkpoint's independent publication
   evaluation too; the last or highest-return checkpoint alone is insufficient.
5. Before calling the recipe reliable, repeat with additional training seeds
   (for example 43 and 44) and independent evaluation seeds, keeping settings
   and version identifiers fixed. Evaluate biological and robot variants
   independently. SB3's [experiment guidance](https://stable-baselines3.readthedocs.io/en/v2.9.0/guide/rl_tips.html)
   supports multiple seeds, separate evaluation, and environment-specific tuning.

Existing checkpoints remain plant/task compatible because optimizer settings
and budget do not change the environment identity. Resuming with these changes
is a **mixed-recipe continuation**, with inherited normalization, learned action
standard deviation, and optimizer state. It is useful for exploration but is
not a fresh test of this recipe. Do not relabel the already-running experiment
or overwrite its captured settings. A duration-only control, if needed, should
use the original optimizer settings with the same 11M allowance and seeds.

## Recovery and qualification limits

This recipe review originally covered stance → locomotion → target reaching.
The subsequent [recovery extension](RECOVERY_CALIBRATION.md) adds an opt-in
push-recovery pilot with a separate physical calibration. The stance
gate measures foot support and does not require both feet to carry load at
every instant. It does not certify disturbance recovery or a moving gait.

The recovery extension implements reproducible, physically scaled
perturbations; seeded evaluation panels; a measured zero-action/held-action
comparison; and a physical recovery judge measured separately on each variant.
It uses the shared task/provenance and frozen gate machinery, while learned
policy qualification remains outstanding and recovery remains non-advancing.
Copying Tyrannosaurus Rex's shove forces or declaring a recovery stage alone
would not supply that evidence. The robot's head and tail remain fixed and unpowered.

## Validation of this change

Validation used MuJoCo 3.10.0, SB3 2.9.0, Gymnasium 1.3.0, NumPy 2.3.5,
and PyTorch 2.9.0+cpu.

| Check | Result |
|---|---|
| Focused physical, environment, configuration, gate, schedule, and catalog regressions | 420 passed; the generated catalog was refreshed and its previously failing freshness check rerun successfully |
| Real shared-trainer and notebook integration | 8 passed across both variants and PPO/SAC |
| Live updates and checkpoint reloads | LR schedules, KL forwarding, entropy decay, and stage warm-up release verified |
| Model and task changes | Environment tables, gate thresholds, SAC optimizer tables, and policy architectures compared against the parent and preserved; plant manifest current |
| Generated interfaces | Species catalog and README regenerated and checked; notebook code parses |
| Python style and typing | Full `environments/` Ruff lint/format and CI mypy command pass (304 source files), using the lint job's dependency versions |
| Reusable probe CLI | Four-arm end-to-end smoke passed with `--updates 1`; configuration mapping explicitly matches the shared trainer's stage-key type |

The [bounded optimizer probe](scripts/probe_ppo_updates.py) compared the original
and revised stance settings on the same current implementation. Each of four
arms collected 32,768 steps with four environments and seed 42: eight complete
rollout/update cycles using the production network, minibatches, and horizon.
The `learn()` target remained the full configured stage allowance so this
short probe did **not** compress the 11M learning-rate or 7M entropy schedules.
All policy parameters remained finite. Exact configurations, versions, per-cycle
measurements, and raw illustrative evaluations are in
[ppo_recipe_probe_v1.json](data/ppo_recipe_probe_v1.json).

| Variant | Mean approximate KL: original → revised | Mean clip fraction: original → revised |
|---|---:|---:|
| Anatomical | 0.1442 → 0.0153 | 64.2% → 14.5% |
| Robot | 0.0949 → 0.0147 | 58.7% → 13.6% |

Each cycle reached its fifth attempted PPO epoch. The epoch counter and mean
KL log do not establish whether a late minibatch triggered KL early stopping.
These results support smaller early update sizes, not an isolated benefit from
the KL backstop or long entropy schedule. Ten held-out deterministic episodes per
arm were deliberately below the certification panel size. Anatomical mean
length increased from 156.1 to 272.7 steps, but full-horizon episodes fell from
1/10 to 0/10. Robot full-horizon episodes increased from 9/10 to 10/10 while
mean reward decreased. These mixed outcomes cannot qualify or rank learned
balance; the existing gates still need full, independent evaluation.

Reproduce the eight-cycle probe from a Git checkout containing the pinned
baseline commit, with the training dependencies installed:

```bash
python -m environments.compsognathus.scripts.probe_ppo_updates --output /tmp/compso-probe
pytest environments/shared/tests/test_compsognathus_training.py
```

The reusable CLI was refactored from the executed probe, preserving its training
and evaluation operations. Its configuration snapshots may differ in comments
and descriptive wording from the final TOMLs; environment and optimizer values
are captured in the evidence. Full 11M-step qualification runs, multi-seed
learning, locomotion, and recovery performance are not established by these checks.
