# T-Rex Action-Space Rescaling

**Status:** Proposed — not implemented.
**Date:** 2026-07-27
**Motivating runs:** `20260725_194916` (w=0.1), `20260726_191730` (w=0.7),
`20260727_130726` (w=2.0), all T-Rex stage 1, PPO, seed 42.

Stage-1 policies visibly flex their legs in an unnatural way. Three
`smoothness_weight` escalations have reduced the effect without removing it.
This plan argues the residual cause is the action-to-joint-angle mapping, not
the reward, and scopes the change to fix it.

---

## The measurement

`smoothness_weight` works, but with returns that rule it out as a path to a
smooth gait. RMS action change per joint per step, measured with
`compare_run_diagnostics.py`:

| `smoothness_weight` | 0.1 | 0.7 | 2.0 |
|---|---|---|---|
| RMS action change / joint | 1.018 | 0.747 | 0.628 |
| final eval reward | 2699.8 | 2516.9 | 2654.7 |
| eval std | ±40.6 | ±87.2 | **±12.2** |

Both escalations fit `r ∝ w^-0.16` to within 4% (exponents −0.159 and −0.165).
Extrapolating, `r = 0.5` needs `w ≈ 8`, halving to `r = 0.31` needs `w ≈ 144`,
and the `r < 0.1` "smooth" threshold needs a weight in the six figures — a
regime where the smoothness term is the entire objective. The lever is real and
cheap (w=2.0 scored *better* than w=0.7 on every axis) but it cannot reach the
target.

### Why the returns are so poor

Measured on the settled home stance (zero action, 600 steps):

| joint | ctrl span | kp | static hold torque | force limit | load |
|---|---|---|---|---|---|
| hip_pitch | 130° | 1200 | 52.1 N·m | 1800 | 2.9% |
| knee | 100° | 1500 | 49.5 N·m | 2250 | 2.2% |
| ankle | 100° | 900 | 58.0 N·m | 1350 | 4.3% |

The legs are 20–45× over-powered for standing, and the action space spans the
full anatomical range of each joint. At the measured `r = 0.628` the policy
commands **31° of knee and 41° of hip every 10 ms** (control runs at 100 Hz:
`timestep 0.002` × `frame_skip 5`), producing 822–855 N·m of torque swing —
16–17× the static hold torque and ~40% of the actuators' force limits.

Holding the stance takes about 1.9° of equilibrium offset at the knee
(49.5 N·m ÷ kp 1500). The action space grants ±50°. The policy has roughly
**26× more authority than the task requires**, so each unit of action carries
enormous physical consequence — which is exactly why taxing action *deltas*
buys so little per unit of weight.

## What this is not

Ruled out by direct measurement of the plant, so these need no further work:

- **Center of mass.** Excluding the prey mocap body (65.4 kg, not part of the
  animal), the T-Rex masses 85.7 kg with its CoM at x = +0.0887, sitting 60.9%
  of the way from heel to toe within a 0.336 m support polygon — 0.131 m of
  margin to the toe edge, 0.205 m to the heel. Slightly forward of centre, as
  expected for a biped, and nowhere near tipping.
- **Joint conditioning.** Every leg joint parks at 46–50% of its range at the
  settled stance. Nothing pinned near a limit, no near-singular knee.
- **Mass distribution.** Pelvis 35 kg (41% of the animal), skull 8.7 kg, thighs
  5.0 kg each. Nothing anomalous.

## Prior art: the raptor

There is **no existing `action_scale` concept anywhere in the repo** — this
would be the first. The raptor's documented jitter fix was the entropy anchor,
not action scaling (`TREX_STAGE1_LEG_JITTER.md`: raptor S1 was the only stage
that annealed `algo_std`, and it did so at the *lowest* smoothness weight,
0.05).

The raptor plant is, however, better proportioned. Same probe, same settled
home stance, control at 100 Hz for both:

| | T-Rex | Raptor |
|---|---|---|
| animal mass | 85.7 kg | 13.5 kg |
| knee kp | 1500 | 180 |
| hip / knee / ankle hold vs force limit | 2.9% / 2.2% / 4.3% | 4.7% / 5.6% / **23.0%** |
| **mean leg authority ratio** | **22×** | **11×** |

"Authority ratio" is commanded degrees per unit of normalized action divided by
the degrees actually needed to hold the pose. The T-Rex has twice the raptor's
over-authority, and its ankle is the starkest case — loaded to 4.3% of its
force limit against the raptor's 23%.

**But the raptor is not a smoothness success story in absolute terms.** At its
documented `r = 0.925` and 57.5° per unit of action, the raptor commands ~53°
of knee travel per step — *more* than the T-Rex's current 31°. The normalized
metric flatters it. If the T-Rex's flexing is an authority problem, the raptor
likely has the same one, unexamined, and "match the raptor" is the wrong
target. Worth reviewing a raptor stage-1 video before assuming otherwise.

## Options

### A. Narrow the leg `ctrlrange` in `trex.xml`

Shrink the commanded envelope at the source.

- **Cost:** `actuator_ctrlrange` feeds *both* the policy-interface payload
  (`plant_contract.py:961`) and the physics payload (`:1107`), so this bumps
  **both** `policy_interface_revision` (6→7) and `physics_revision` (4→5).
- **Wrong granularity.** It applies to every stage. Stage 2 reaches 6.36 m/s
  and needs the swing; capping it would cripple locomotion.
- Rejected.

### B. Per-stage `action_scale` (recommended)

Add an env parameter that scales the normalized action about the ctrlrange
midpoint before the affine map, exposed per stage in the TOML `[env]` block.
Stage 1 runs a small scale; stages 2 and 3 keep 1.0 until measured.

- Pure function of the action, so it fits the existing action-mapping contract
  shape (`midpoint/v1`, `home-keyframe-residual/v1` — `plant_contract.py:56-58`)
  as a new mode rather than a structural change.
- Per-stage, which is the granularity the problem actually has.
- Leaves physics untouched: `policy_interface_revision` 6→7 only.
- **Zero-action behaviour is preserved** — action 0 still maps to the midpoint
  at any scale, so the zero-action baseline (1800.56 ± 1267.66) and therefore
  the 1900 stage gate stay valid. This is worth more than it sounds; option A
  moves the floor unless the narrowing is exactly symmetric.

### C. Action rate limit / low-pass filter

Cap per-step change while preserving reachable range.

- Attacks the measured quantity (Δa) most directly, and preserves stage 2/3
  reachability, so it is the better long-term answer for locomotion.
- **But** it makes the mapping history-dependent. That breaks the "pure
  function of action" assumption in the contract's callable fingerprinting
  (`_callable_semantics`), and the MJX path would have to carry filter state
  through the rollout (`mjx_env.py:508-556`, `jax_setup.py:487-509`).
- Deferred. Revisit if B fixes stage 1 but stages 2 and 3 still flex.

## Recommended change

Option B, stage 1 only to begin with.

**Sizing is the open question**, but the data already sets a floor. A policy
whose RMS action change per joint is `r` must sweep a commanded envelope of at
least `r` in normalized units — it cannot move further per step than the range
it visits. At `r = 0.628`, that is **at least 31° peak-to-peak of knee travel**
(±15.7°), and possibly much more.

So `action_scale = 0.25` — which caps the envelope at 25° peak-to-peak — sits
*below* what the current policy demonstrably uses, and would force a
qualitatively different policy rather than a smoother version of this one. That
may still be the right answer, but it is a bigger intervention than it looks.
**0.4–0.5 is the defensible starting range** pending measurement.

**Pre-work that settles it:** run
`environments/shared/scripts/joint_excursion_report.py trex 1 <best_model.zip>`
against `20260727_130726/stage1`. It reports commanded and achieved envelope
per joint in degrees, against the 31° analytic floor. If the policy sweeps 80°,
scaling to 0.5 leaves real headroom; if it sits at the floor, the envelope is
already minimal and `action_scale` will buy less than hoped. Needs SB3, so it
runs in Colab. This is the one thing to do before writing any code.

## Blast radius

| Area | Impact |
|---|---|
| `environments/shared/base_env.py:811` `_scale_action` | New scale term in the SB3 path |
| `environments/shared/mjx_utils.py:22` | New JAX twin alongside `scale_action_jax` |
| `environments/shared/mjx_env.py:554`, `jax_setup.py:501` | Dispatch for the new mode |
| `environments/shared/plant_contract.py:339` `_action_mapping_contract` | New mode + its JAX function name |
| `plant_manifest.generated.json` | Regenerate; `policy_interface.revision` 6→7 |
| `configs/trex/stage1_balance.toml` | New `action_scale` key |
| **Existing checkpoints** | **All invalidated.** A `policy_interface_revision` bump means no saved T-Rex policy loads against the new interface. |
| CI | `plant-contract` job verifies the manifest; `test-jax-cpu` verifies SB3/MJX parity |

The checkpoint break is the real cost. Stage 2 trains from stage 1's model, so
this forces a full 13h curriculum re-run — there is no partial migration.

## Validation

1. **Pre-work:** measure the trained policy's joint excursion envelope (above).
2. Re-run `zero_action_baseline.py trex --episodes 40`. Expected **unchanged**
   at 1800.56 ± 1267.66 — if it moves, the mapping is wrong.
3. Regenerate the plant manifest; confirm `physics.sha256` is unchanged and
   only `policy_interface` moves.
4. `plant-contract` and `test-jax-cpu` green — the SB3/MJX parity test is the
   one that matters and cannot be run locally.
5. Full stage-1 run. Compare with `compare_run_diagnostics.py` against
   `20260727_130726/stage1`. Success is RMS action change materially below
   0.628 **at equal or better reward** — 2654.7 ± 12.2 is the bar, and it is a
   high one.

## Risks

- **The policy may re-saturate.** `action_scale` caps amplitude, not normalized
  rate; PPO may relearn to sweep the smaller range at the same `r`. Absolute
  motion still falls by the scale factor, so the visual improvement is bounded
  below by that — but the headline `r` metric may barely move. Judge this
  change on commanded degrees per step, not on `r`.
- **Under-scaling costs balance.** Too small a scale and the policy cannot
  arrest a perturbation, which shows up as a stage-1 gate miss and halts the
  curriculum (`train_base.py:1282-1292`).
- **Stages 2 and 3 are untouched by this plan** and still run
  `smoothness_weight` 0.05 and 0.02. A smoother stage-1 policy will spend 16M
  subsequent steps under almost no action-rate penalty.

## Effort

Roughly a day of implementation across the SB3 path, the JAX twin, the contract
mode, and the manifest regeneration — the plumbing dominates, the config is one
line. Plus a 13h validation curriculum. The pre-work rollout is ~10 minutes in
Colab and should happen first, because it decides whether the whole change is
sized correctly.
