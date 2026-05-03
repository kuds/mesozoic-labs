# JEPA Variant Choice for Mesozoic Labs

> **Status: EXPLORATORY SCAFFOLDING — PAUSED PENDING REVIEW (2026-05-03).**
> The code in `environments/shared/jepa/` and `environments/shared/wrappers/`
> is a partial PR for a JEPA + RL pipeline. Before continuing, the
> premise itself was re-examined: see the "Is JEPA the right tool?"
> section at the bottom of this document. Net assessment: **JEPA is
> probably the wrong fit for the current state-based velociraptor
> task.** Better-fit alternatives (TD-MPC2, DreamerV3 on proprio,
> SPR-style auxiliary losses) are recommended in that section. This
> document is preserved on-branch so the analysis isn't lost.
>
> Scope (as originally written): velociraptor first; conclusions
> transfer to T-Rex / brachiosaurus with no architecture changes.

## TL;DR

We adopt **I-JEPA with a temporal predictor head** (a "V-JEPA-lite" variant
operating on short 4-frame stacks) as the primary configuration. Pure
I-JEPA on single frames is shipped as the smallest fallback, and a path
to **V-JEPA 2-AC** is reserved for a future stretch experiment once the
representation pipeline is validated and a larger GPU is available.

The SB3 integration is variant-agnostic: any encoder that emits a
`(B, D)` embedding from a `(B, C, H, W)` or `(B, T, C, H, W)` input
plugs into the same `JEPAFeatureExtractor`.

## Open questions, resolved

### 1. Is the velociraptor env pixel-based or state-only?

**State-only.** `RaptorEnv._get_obs()` returns a 67-dim proprioceptive
vector (joint qpos/qvel, pelvis quat/gyro/accel/linvel, foot contacts,
prey direction + distance). No image is ever produced during training.

`BaseDinoEnv` already supports `render_mode="rgb_array"` via
`mujoco.Renderer` (480×640) with a tracking camera on the pelvis, so
the rendering plumbing is in place. We add a Gymnasium observation
wrapper (`environments/shared/wrappers/pixel_observation.py`) that:

1. constructs a renderer at a configurable resolution (default 84×84,
   matching the canonical Atari/DMC pixel-RL setup);
2. on every `step` / `reset`, calls `mujoco.Renderer.update_scene` and
   returns either pixels-only or a `gym.spaces.Dict({"state": ...,
   "pixels": ...})` observation depending on a flag;
3. keeps the renderer per-process and closes it cleanly so SubprocVecEnv
   workers do not leak GL contexts.

The wrapper is **opt-in**: the existing MlpPolicy training pipeline is
untouched. JEPA-augmented training only flips it on.

### 2. Does the curriculum produce diverse-enough rollouts for JEPA pretraining?

**Mostly yes for stages 2–3, but stage 1 alone is too narrow.** Stage 1
(balance) is intentionally near-stationary — the agent should barely
move, so its rollouts contain very little visual diversity (the prey is
spawned 10–15 m away and the raptor mostly stands still). Pretraining on
stage-1 data would learn an embedding that ignores motion.

To keep things simple and reproducible we collect from three sources
*combined*:

- **stage-1 random policy** (uniform action sampling, ~20% of frames) —
  covers the "out of distribution" pose space the agent may briefly
  visit during early SAC exploration;
- **stage-2 partially-trained checkpoint** (e.g. 1M-step SAC, ~40% of
  frames) — locomotion phase, where most visual diversity lives;
- **stage-3 partially-trained checkpoint** (~40% of frames) — striking
  behavior, claw poses, prey approach geometry.

Reset noise is already 0.05 rad (stage 1) and prey distance is randomized,
which gives free-of-charge spatial diversity at episode boundaries.

This choice sidesteps a chicken-and-egg problem (you need an OK policy
to collect varied frames, but you want JEPA to *help* train that policy).
A single-pass pretraining is enough for an apples-to-apples sample-
efficiency comparison; iterative collection is documented as a follow-up.

### 3. Should JEPA pretraining happen once on combined-stage data, or per-stage in lockstep with the curriculum?

**Once, on combined-stage data**, for the proof-of-concept. Reasons:

- A single frozen encoder is the cleanest baseline against the
  from-scratch SB3 SAC. Per-stage encoders introduce a confound (which
  stage's frames helped at SAC time?).
- The curriculum reward changes, but the underlying pixel manifold —
  raptor body + ground + prey — is shared across stages. There is no
  obvious reason a stage-2 visual representation hurts stage-3.
- Compute. Three pretraining runs cost 3× as much as one and the gain
  on a Colab T4 is uncertain.

A `--per-stage` flag on `train_jepa.py` is implemented but documented as
*optional / experimental*.

## Why not full V-JEPA 2?

V-JEPA 2 ([arxiv:2506.09985](https://arxiv.org/abs/2506.09985)) is the
right long-term target, but the off-the-shelf models are 300 M to 1 B
parameters and expect 16-frame clips at 256² resolution. On a Colab T4
(16 GB VRAM) you cannot fit the standard model at the standard input
size, full stop.

Two viable degradations:

1. **Reimplement V-JEPA at small scale** — same masked tube prediction
   loss but a ViT-Tiny/16 backbone (~5 M params), 4-frame clips at 84².
   This is what the prompt's "V-JEPA-lite" stretch target asks for.
2. **Strip back to I-JEPA** — drop the temporal axis entirely, mask
   spatial blocks within a single frame, predict in embedding space.

The two degradations look the same when `T=1`. We start from the I-JEPA
side (stable, well-understood masked-block prediction) and add an
optional temporal axis via a 1-D temporal predictor over a length-`T`
embedding stack. This gives us a continuum from "frame I-JEPA" → "tubelet
V-JEPA-lite" without two separate codebases. The same encoder weights
can be evaluated as a frame extractor for SAC even if the temporal
predictor is trained on `T=4`.

## Architecture summary

| Component | Choice | Notes |
|-----------|--------|-------|
| Backbone | ViT-Tiny/16 (~5 M params) at 84×84 | Patch size 8 → 121 tokens; 4-block, 192-dim, 3-head |
| Target encoder | EMA copy of context encoder, momentum 0.998 | Standard JEPA recipe |
| Predictor | 2-block transformer, 96-dim, 3-head | Narrow per the I-JEPA paper |
| Masking | Multi-block, ~4 target blocks (~25 % of patches each), with overlap removed; for `T>1`, time-shared masks ("tubelet") | |
| Loss | Smooth-L1 in normalized embedding space | Layer-norm targets before the loss to match V-JEPA 2 |
| Optimizer | AdamW, lr 5e-4, wd 0.04, cosine schedule, 5 % warmup | |
| Compute target | ≤ 4 GB VRAM at batch 64, `T=4`, 84×84 on a T4 | Documented in `RESULTS.md` |

## Why this fits the existing repo conventions

- Lives under `environments/shared/jepa/` — same shape as
  `environments/shared/curriculum.py` etc. (The prompt referenced
  `mesozoiclabs/jepa/`, but the actual Python package is `environments/`;
  matching the existing layout was prioritized over the prompt's path.)
- Pretraining script logs to W&B project `mesozoic-labs-jepa` (sibling
  of the existing `mesozoic-labs` project), with config, run metadata,
  and final checkpoint uploaded as an artifact.
- `JEPAFeatureExtractor` is a standard SB3 `BaseFeaturesExtractor`
  subclass and slots into `policy_kwargs` — no fork of the SB3 training
  path, opt-in via a single CLI flag.
- A `[jepa]` extra in `pyproject.toml` adds `torch`, `einops`,
  `imageio`. The default install is unchanged.

## What is *not* in scope for this PR

- Latent MPC over the predictor (the "stretch" item in the brief). The
  predictor is trained but only the encoder is wired into RL.
- Domain randomization of textures, lighting, camera. Worthwhile but
  orthogonal — without it the JEPA encoder will overfit to the default
  MuJoCo textures, which is fine for an apples-to-apples comparison.
- Brachiosaurus / T-Rex training. The encoder is species-specific
  (different body geometries). Cross-species transfer is left to a
  separate experiment.

---

## Is JEPA the right tool? (added during review)

After scaffolding the pipeline above, the premise was re-examined.
**Recommendation: probably not, for the velociraptor task as it stands.**

### Why not

1. **Observation is already low-dimensional and informative.** The env
   exposes 67 hand-engineered proprioceptive scalars (joint qpos/qvel,
   pelvis quat/gyro/accel/linvel, foot contacts, prey direction +
   distance). JEPA's value proposition is compressing high-dimensional
   sensory streams (pixels, video) into useful embeddings. Going state →
   pixels → JEPA embedding → policy is, in the best case, lossy
   reconstruction of information already available cleanly. In the
   worst case it discards important details (exact joint angles, prey
   geometry) that the existing rewards depend on.
2. **The PPO/SAC sample-efficiency gap is not a representation gap.**
   The README shows velociraptor PPO actually passes all three stages
   at 22 M steps with 93.3 % strike success; SAC reaches similar
   quality faster. That gap is the textbook on-policy-vs-off-policy
   sample-efficiency story. No published result shows pixel-based
   self-supervision closing it on proprio inputs.
3. **Compute tax.** A pixel pipeline at 84×84 renders every step,
   roughly 5–10 × slower per env step on a Colab T4. Pretraining cost
   stacks on top. At 22 M+ steps per stage this is real money.
4. **Sim-to-real argument is weakened.** A JEPA encoder trained on
   default MuJoCo textures will not transfer to a real camera without
   aggressive domain randomization. State-based policies + a separate
   vision-to-state perception module is the more standard recipe.

### Where JEPA *would* be the right call here

- A future **pixel-based predator-perception task**: occluded prey,
  peripheral vision, terrain-ahead reasoning — anywhere the agent
  genuinely needs to see, not just be told.
- **Cross-species visual transfer** (shared scene geometry, different
  morphologies). Not relevant until the species share a vision sensor.
- **Sim-to-real with a real camera** stream and aggressive randomization.

### Better alternatives at the same compute budget

In rough order of expected ROI:

1. **TD-MPC2** (Hansen et al., 2024) — latent-dynamics MPC on the
   existing 67-dim state. Beats SAC by ~2–5× sample efficiency on
   DMC/MetaWorld locomotion at matched compute. *This is the right
   "world model + planning" target.*
2. **DreamerV3** on proprio — same world-model story without pixels.
3. **Auxiliary self-supervised heads** (SPR / BYOL-explore-style next-
   state prediction in a small MLP latent) bolted onto the existing
   SAC. ~10 lines of code; no rendering required.
4. **Reward / curriculum tuning** for the strike stage if the bottleneck
   is sparse reward, not representation.

The exploratory scaffold under `environments/shared/jepa/` and
`environments/shared/wrappers/` is preserved so the analysis (and the
encoder/predictor reference impl) is not lost — but no further code,
data collection, training, or evaluation is planned on this branch
until the direction is reconfirmed.
