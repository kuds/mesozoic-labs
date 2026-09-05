#!/usr/bin/env python3
"""Report how far a policy's raw actions leave the declared action bound.

The action space is ``Box(-1, 1)``, but SB3's Gaussian policy is unbounded, so
the sampled action routinely lands outside it.  Both training paths clip before
stepping the environment -- SB3 at ``on_policy_algorithm.py:214-218`` and
``policies.py:379`` (inside ``predict``), the JAX trainer in
``collect_rollout`` inside ``_build_jit_fns``
(``environments/shared/jax_train_fn.py``) -- so the plant and the reward both
see an in-bound action.  What the clip hides is how hard the policy is pushing
against the bound, and that is what this script measures.

It matters because PPO stores the *raw* action and its ``log_prob`` while the
environment responds to the *clipped* one.  Once a large fraction of components
is saturated, most of the policy's output distribution sits where moving the
mean further does not move the plant at all.

Reading the ``diagnostics/`` scalars correctly
----------------------------------------------

Four of them are **pre-clip** and one is **post-clip**, which is easy to
misread:

* ``action_mean``, ``action_std``, ``action_abs_max``, ``action_saturation``
  come from ``diagnostics.py:210``, which reads ``self.locals["actions"]`` --
  SB3's raw Gaussian sample, before the clip.
* ``action_delta`` arrives by a different route: it is returned by
  ``reward_action_smoothness`` from *inside* the env, so it is computed on the
  post-clip action.

This script reports the pre-clip family at the same 0.99 threshold
``DiagnosticsCallback`` uses, so its output is directly comparable to the
``diagnostics/raw_action_saturation`` scalar already in tensorboard -- but
offline, per-actuator, and without waiting for a run to finish. (The env's
``diagnostics/action_saturation`` is a different quantity: the post-filter
ramp fraction behind ``reward_action_saturation``.)

Two modes:

* ``--model`` given: measures a trained checkpoint.  Answers "how saturated did
  this policy actually get, and which joints are pinned?"
* ``--model`` omitted: probes a Gaussian of configurable ``--std``, which is
  what an *untrained* policy looks like at initialisation.  Answers "would a
  start-of-training check have caught this?"

  It would, immediately.  SB3's default ``log_std_init = 0`` gives std 1.0, and
  this script at ``--std 1.0`` measures **32.0% of components outside the
  bound** -- matching the 0.3218 that run ``20260727_130726`` logged to the
  same scalar (named ``diagnostics/action_saturation`` at the time) at step
  8,192, its very first diagnostic write.  Nothing has to be learned for the condition to appear: it follows
  from pairing an unbounded Gaussian with a ``Box(-1, 1)`` action space, so a
  startup probe is enough to catch it.

The rollout mirrors SB3: the raw action is recorded for the statistics, and
``clip(action, -1, 1)`` is what steps the environment.

Usage::

    python environments/shared/scripts/action_bound_report.py trex 1 \\
        --model /path/to/run/stage1/models/robust_best_model.zip

    python environments/shared/scripts/action_bound_report.py trex 1 --std 1.0
"""

import argparse
import sys
from pathlib import Path

import numpy as np

_repo_root = str(Path(__file__).resolve().parents[3])
if _repo_root not in sys.path:
    sys.path.insert(0, _repo_root)

from environments.shared.config import SPECIES_NAMES, build_env
from environments.shared.constants import PUBLICATION_SEED_START
from environments.shared.policy_loading import UNNORMALIZED_BANNER, PolicyLoadError, load_sb3_checkpoint

# Matches DiagnosticsCallback.action_saturation_threshold, so the figures here
# are directly comparable to diagnostics/raw_action_saturation.
SATURATION_THRESHOLD = 0.99


def actuator_names(env, n_actuators: int) -> list[str]:
    """Actuator names from the compiled model, falling back to indices."""
    import mujoco

    names = [mujoco.mj_id2name(env.model, mujoco.mjtObj.mjOBJ_ACTUATOR, i) for i in range(n_actuators)]
    return [name or f"actuator[{i}]" for i, name in enumerate(names)]


def main(argv: list[str]) -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("species", choices=SPECIES_NAMES)
    parser.add_argument("stage", type=int, choices=(1, 2, 3))
    parser.add_argument("--model", default=None, help="SB3 .zip policy; omit to probe an untrained Gaussian")
    parser.add_argument("--vecnorm", default=None, help="VecNormalize .pkl (default: alongside the model)")
    parser.add_argument(
        "--allow-unnormalized",
        action="store_true",
        help="proceed without a VecNormalize sidecar (the report then scores a different policy)",
    )
    parser.add_argument("--std", type=float, default=1.0, help="Gaussian std when --model is omitted")
    parser.add_argument("--episodes", type=int, default=10)
    parser.add_argument("--seed", type=int, default=PUBLICATION_SEED_START)
    args = parser.parse_args(argv)

    env = build_env(args.species, args.stage)
    n_actuators = int(env.action_space.shape[0])
    names = actuator_names(env, n_actuators)

    model = None
    normalizer = None
    if args.model:
        try:
            model, normalizer, _ = load_sb3_checkpoint(
                args.model,
                args.vecnorm,
                lambda: build_env(args.species, args.stage),
                allow_unnormalized=args.allow_unnormalized,
            )
        except PolicyLoadError as exc:
            # The CLI boundary is where a loading failure becomes an exit status.
            raise SystemExit(str(exc)) from exc

    rng = np.random.default_rng(args.seed)
    returns: list[float] = []
    all_actions: list[np.ndarray] = []

    for index in range(args.episodes):
        obs, _ = env.reset(seed=args.seed + index)
        total = 0.0
        while True:
            if model is None:
                raw_action = rng.normal(0.0, args.std, size=n_actuators)
            else:
                obs_input = normalizer.normalize_obs(obs) if normalizer is not None else obs
                raw_action, _ = model.predict(obs_input, deterministic=True)
            action = np.asarray(raw_action, dtype=np.float64).reshape(-1)
            all_actions.append(action)
            # SB3 clips before stepping (on_policy_algorithm.py:214-218, and
            # policies.py:379 inside predict).  Mirror that here so the
            # trajectory is the one training would produce, while the
            # statistics are taken on the raw sample.
            obs, reward, terminated, truncated, _ = env.step(np.clip(action, -1.0, 1.0))
            total += float(reward)
            if terminated or truncated:
                break
        returns.append(total)

    env.close()

    acts = np.stack(all_actions)
    flat = acts.reshape(-1)
    source = f"policy {Path(args.model).name}" if args.model else f"untrained Gaussian, std={args.std}"

    print(f"\n{args.species} stage {args.stage}: {source}")
    if model is not None and normalizer is None:
        print(UNNORMALIZED_BANNER)
    print(f"{len(all_actions)} steps over {args.episodes} episodes, {n_actuators} actuators")
    print(f"mean return {np.mean(returns):.2f} +/- {np.std(returns):.2f}\n")

    print("Raw (pre-clip) action statistics, against an action space of Box(-1, 1):")
    print(f"  mean                {flat.mean():+.4f}     (diagnostics/action_mean)")
    print(f"  std                 {flat.std():.4f}      (diagnostics/action_std)")
    print(f"  abs max             {np.abs(flat).max():.4f}      (diagnostics/action_abs_max)")
    print(f"  E[a^2]              {np.mean(flat**2):.4f}      (<= 1.0 if the bound held)")
    print(f"  |a| > 1             {np.mean(np.abs(flat) > 1.0) * 100:.1f}% of components")
    print(
        f"  |a| >= {SATURATION_THRESHOLD}          {np.mean(np.abs(flat) >= SATURATION_THRESHOLD) * 100:.1f}%"
        "       (diagnostics/raw_action_saturation)\n"
    )

    per_actuator = np.mean(np.abs(acts) >= SATURATION_THRESHOLD, axis=0) * 100.0
    width = max(len(name) for name in names)
    print(f"Per-actuator saturation (share of steps with |a| >= {SATURATION_THRESHOLD}):")
    print(f"  {'actuator':<{width}}  {'saturated':>9}  {'abs max':>8}")
    for i in np.argsort(-per_actuator):
        print(f"  {names[i]:<{width}}  {per_actuator[i]:>8.1f}%  {np.abs(acts[:, i]).max():>8.2f}")

    print("\nThe reward is not affected by any of this: both training paths clip")
    print("before env.step, so reward_energy and reward_action_smoothness are")
    print("computed on in-bound actions.  What saturation costs is control")
    print("resolution, not reward -- a saturated component's gradient moves the")
    print("policy mean without moving the plant.")
    print("\nSeparately, base_env.py:783 passes the raw action to _get_reward_info")
    print("while line 765 clips for ctrl.  Harmless under SB3 and the JAX trainer,")
    print("but a direct caller that skips the clip is charged energy and")
    print("smoothness for magnitude the plant never sees.  This script clips.")


if __name__ == "__main__":
    main(sys.argv[1:])
