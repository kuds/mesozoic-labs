---
sidebar_position: 4
---

# Mirror Symmetry

A dinosaur's left and right halves are mirror images of each other, and a
level floor has no preferred side. So the right thing to do in the mirror
image of a situation is the mirror image of the right thing to do in the
original: if the animal leans left and should step its left foot out, then
leaning right it should step its right foot out. A policy that behaves this
way is called *mirror-symmetric*.

Stock PPO is never told this. Every joint, foot and sensor is just another
number to it, so nothing stops it from settling into a lopsided habit, such
as leaning onto the edge of one foot or holding one ankle more bent than the
other, and the T. rex stance runs of 2026 did exactly that. Mirror-symmetric
training adds the missing knowledge. It is an option, **off by default**: no
stage config turns it on, and with its keys absent training is stock PPO, bit
for bit.

## The mirror map

Everything starts from a *mirror map* for the model: for every observation
and every action, which entry is its mirror partner and whether the value
changes sign. The right hip's pitch angle swaps with the left hip's; a
sideways quantity (the body leaning left, a turn to the left, the lateral
command) flips sign; a forward or vertical quantity keeps it.

The map is derived from the model itself, never written by hand.
`environments/shared/symmetry.py` pairs every body, joint, actuator and
sensor with its partner by name (`r_` and `l_`) and checks the pairing against
the compiled geometry: positions, joint axes and ranges, masses, contact
settings and actuator gains must all be true mirror images, or the derivation
refuses the model. The checks cover what the current models use, so a new
species' map also needs a paired-rollout test like the T. rex's
(`environments/shared/tests/test_symmetry.py`), which runs a situation and its
mirror image side by side and checks that they stay mirror images. The result
is written to
`configs/<species>/<embodiment>/mirror_map.json`, which records the plant's
physics and policy-interface digests. A map only loads for the exact plant it
was derived from, so after any change to the model it must be regenerated:

```bash
# Derive and write a model's map
python -m environments.shared.symmetry --write --species tyrannosaurus_rex --embodiment anatomical

# Check every committed map against a fresh derivation (the test suite runs this too)
python -m environments.shared.symmetry --check
```

Today the anatomical T. rex has a committed map. The anatomical
Compsognathus, which shares its observation layout, derives one too. The
other anatomical models pass the model checks, but their older observation
layouts have no mirror rule yet. The Compsognathus robot fails the checks,
because its model is not mirror-symmetric as built.

## What mirror-symmetric training does

Turning the option on swaps PPO for `MirrorKLPPO`
(`environments/shared/mirror_kl_ppo.py`), which is the stock algorithm with
four additions:

1. **A mirror loss.** For every training sample, the policy is also asked
   what it would do in the mirrored situation, and it is penalised by how far
   that is from the mirror image of what it actually does. The penalty is
   measured in units of the policy's own exploration noise and grows smoothly
   with the asymmetry; its weight is `symmetry_kappa`.
2. **A mirrored critic.** The value network also learns that a situation and
   its mirror image are worth the same.
3. **Symmetric observation statistics.** The running mean and variance that
   normalise the observations are kept exactly symmetric, so mirroring a
   normalised observation is exact.
4. **Tied exploration noise.** Each left joint explores exactly as much as
   its right partner.

The usual alternative, adding mirrored copies of the experience to the PPO
batch, was measured on stored stance rollouts and rejected. Its corrective
push is measured relative to the exploration noise, and a standing policy
explores very little (a few hundredths of the action range). At that scale a
small asymmetry already pushes most mirrored samples outside PPO's clipping
range, where they stop correcting anything, while the few left unclipped make
the gradient erratic. The mirror loss above keeps pulling toward symmetry at
any noise level, and leaves the PPO update itself untouched.

Training changes nothing about the result's format: a checkpoint is a normal
PPO checkpoint with normal normalisation statistics, and evaluation, reports
and the next stage load it as usual. The checkpoint also records the settings
it was trained with (`mesozoic_symmetry`), and resuming it in the same stage
requires the same settings.

## Turning it on

Add the keys to a stage's `[ppo]` table:

```toml
[ppo]
symmetry = "mirror_kl"          # "none" (or no key) is stock PPO
symmetry_kappa = 0.3            # weight of the mirror loss
symmetry_sigma_floor = 0.01     # smallest noise scale the loss divides by
symmetry_mirror_critic = true   # the mirrored critic
```

The three settings shown are the defaults, so `symmetry = "mirror_kl"` alone
gives the same training. They are what the T. rex stance screen of October
2026 used. Still, write all four keys in a stage that turns the option on: the
stage's training recipe (its `hyperparameters_sha256`) hashes only the keys
its TOML writes, so a default left implicit could later change without moving
that digest. The keys work with `--override` too, for example
`--override ppo.symmetry=mirror_kl`. They are part of the training recipe, not
of the task: the task fingerprint does not move.

The stage's model must have a committed mirror map for its current plant;
otherwise training refuses to start.

## Two T. rex stance options

The same screen tried two reward changes on the T. rex stance, which ship as
`[env]` options, also off by default:

```toml
[env]
lateral_lean_weight = 0.25            # default 0.0: off
lateral_lean_tolerance_m = 0.015      # default 0.015
leg_home_pose_reference = "settled"   # default "keyframe"
```

- **The standing lean term** (`lateral_lean_weight`) rewards keeping the
  body's centre of mass over the middle of the feet, left to right. It is a
  bonus of up to `lateral_lean_weight` per step, which falls to about 37% of
  that at an offset of `lateral_lean_tolerance_m`. It is paid only after the
  stance settle (`stance_width_settle_steps`), only while the command is
  zero and only while both feet carry load (`foot_terms_min_support_force`).
  A bonus raises every standing episode's return by about
  `lateral_lean_weight` × 800, so the stance's collapse floor has to be
  re-measured before a stage enables it. The same change adds
  `reward_lateral_lean` to the logged reward components
  (`DiagnosticsCallback.REWARD_KEYS`) and to the reward-decomposition plot's
  weight map (`environments/shared/visualization.py`), which leave it out
  while no stage pays it.
- **The settled leg target** (`leg_home_pose_reference = "settled"`) centres
  the `leg_home_pose` term on the pose the legs actually settle into when the
  animal stands still (averaged left and right) instead of the authored
  keyframe. Under the animal's weight the legs settle a few degrees away from
  the keyframe at the hips and ankles, and a policy paid for pulling them
  back toward it tips the body forward. The settled pose is measured on the
  current plant and re-measured with any plant revision (a test checks it).

Off, the options change no reward, no logged value and no task fingerprint.
On, they are part of the task like any other `[env]` setting.

## What the screen found

In the October 2026 screen, three T. rex stance seeds trained to 6M steps
with mirror-symmetric training, the lean term and the settled leg target.
The policies came out symmetric (every joint group's median mirror asymmetry
under 0.1 degree), did not fall in any of the 120 final evaluation episodes
and passed the stance gate on two of the three seeds. They did not pass the
screen's stricter criteria for the foot's front-to-back pressure and for
robustness to action noise, so no stage enables the options yet. They are
kept as tested building blocks for later stance and walking experiments, and
for other species once they have a map.
