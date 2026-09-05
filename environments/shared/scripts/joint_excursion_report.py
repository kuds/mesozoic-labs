#!/usr/bin/env python3
"""Report the joint envelope a trained policy actually uses.

``compare_run_diagnostics.py`` reports RMS action change per joint per step,
which says how *fast* the commands move but not how *far* they range.  Sizing
an action-space change needs the second number: if a balance policy never
leaves +/-15 degrees of knee travel, the other 35 degrees of commanded range
are dead weight that only amplifies jitter.

Reports, per actuated joint, the commanded envelope (from ``data.ctrl``) and
the achieved envelope (from ``data.qpos``), both in degrees, alongside the
per-step command delta.  Deterministic rollouts, matching the eval videos.

Note the analytic floor: a policy whose RMS action change per joint is ``r``
must sweep a commanded envelope of at least ``r`` in normalised units, since it
cannot move further per step than the range it visits.  A measured envelope at
that floor means tight oscillation; well above it means genuine repositioning.

Usage::

    python environments/shared/scripts/joint_excursion_report.py \\
        trex 1 /path/to/run/stage1/models/robust_best_model.zip

    # The VecNormalize sidecar is found automatically under either naming
    # convention (curated ``<stem>_vecnorm.pkl`` or SB3's periodic
    # ``<prefix>_vecnormalize_<steps>_steps.pkl``); pass --vecnorm to
    # override.  A checkpoint with no sidecar is fatal unless
    # --allow-unnormalized is given.
"""

import argparse
import sys
from pathlib import Path

import mujoco
import numpy as np

_repo_root = str(Path(__file__).resolve().parents[3])
if _repo_root not in sys.path:
    sys.path.insert(0, _repo_root)

from environments.shared.config import SPECIES_NAMES, build_env
from environments.shared.constants import PUBLICATION_SEED_START

UNNORMALIZED_BANNER = "UNNORMALIZED EVAL — results are not comparable to training-time metrics"


def resolve_vecnorm_path(model_path: str, vecnorm_arg: str | None, allow_unnormalized: bool) -> str | None:
    """The VecNormalize sidecar to evaluate with, or ``None`` for a deliberately unnormalised run.

    An explicit ``--vecnorm`` wins.  Otherwise the trainer's own resolver
    probes both sidecar conventions -- the ``<stem>_vecnorm.pkl`` guess this
    script used to make can never match SB3's periodic
    ``<prefix>_vecnormalize_<steps>_steps.pkl``, so every periodic checkpoint
    was silently scored on raw observations.  No sidecar is fatal unless
    ``--allow-unnormalized``: a policy evaluated unnormalised is a different
    policy, and the report would blame the joints for a loading mistake.
    """
    if vecnorm_arg is not None:
        return vecnorm_arg
    from environments.shared.train_base import _resolve_vecnorm_sidecar

    candidate = _resolve_vecnorm_sidecar(model_path)
    if Path(candidate).exists():
        return candidate
    if allow_unnormalized:
        return None
    raise SystemExit(
        f"no VecNormalize sidecar found for {model_path} (probed {candidate}). A policy evaluated on "
        "unnormalised observations is a different policy; pass --vecnorm, or --allow-unnormalized "
        "to proceed deliberately."
    )


def main(argv: list[str]) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("species", choices=SPECIES_NAMES)
    parser.add_argument("stage", type=int, choices=(1, 2, 3))
    parser.add_argument("model", help="path to the SB3 .zip policy")
    parser.add_argument("--vecnorm", default=None, help="VecNormalize .pkl (default: alongside the model)")
    parser.add_argument(
        "--allow-unnormalized",
        action="store_true",
        help="proceed without a VecNormalize sidecar (the report then scores a different policy)",
    )
    parser.add_argument("--episodes", type=int, default=10)
    parser.add_argument("--seed", type=int, default=PUBLICATION_SEED_START)
    args = parser.parse_args(argv)

    from stable_baselines3 import PPO
    from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

    env = build_env(args.species, args.stage)
    model = PPO.load(args.model, device="cpu")

    vecnorm_path = resolve_vecnorm_path(args.model, args.vecnorm, args.allow_unnormalized)
    normalizer = None
    if vecnorm_path:
        normalizer = VecNormalize.load(vecnorm_path, DummyVecEnv([lambda: build_env(args.species, args.stage)]))
        normalizer.training = False
        normalizer.norm_reward = False

    mj = env.model
    ctrl_log: list[np.ndarray] = []
    qpos_log: list[np.ndarray] = []

    for i in range(args.episodes):
        obs, _ = env.reset(seed=args.seed + i)
        while True:
            obs_input = normalizer.normalize_obs(obs) if normalizer is not None else obs
            action, _ = model.predict(obs_input, deterministic=True)
            obs, _, terminated, truncated, _ = env.step(action)
            ctrl_log.append(env.data.ctrl.copy())
            qpos_log.append(env.data.qpos.copy())
            if terminated or truncated:
                break

    ctrl = np.asarray(ctrl_log)
    qpos = np.asarray(qpos_log)
    print(f"\n{args.species} stage {args.stage}: {len(ctrl)} steps over {args.episodes} episodes")
    if normalizer is None:
        print(f"  {UNNORMALIZED_BANNER}")
    print(
        f"  {'joint':<18}{'cmd p5-p95':>14}{'cmd full':>11}{'achieved':>11}"
        f"{'ctrl span':>11}{'used':>7}{'delta/step':>12}"
    )

    for a in range(mj.nu):
        name = mujoco.mj_id2name(mj, mujoco.mjtObj.mjOBJ_ACTUATOR, a)
        jid = int(mj.actuator_trnid[a, 0])
        qadr = mj.jnt_qposadr[jid]
        lo, hi = mj.actuator_ctrlrange[a]
        span = np.degrees(hi - lo)

        c = np.degrees(ctrl[:, a])
        q = np.degrees(qpos[:, qadr])
        band = np.percentile(c, 95) - np.percentile(c, 5)
        full = c.max() - c.min()
        achieved = q.max() - q.min()
        delta = float(np.sqrt(np.mean(np.diff(c, axis=0) ** 2)))
        print(
            f"  {name:<18}{band:11.1f} deg{full:8.1f} deg{achieved:8.1f} deg"
            f"{span:8.0f} deg{100 * full / span:6.0f}%{delta:9.1f} deg"
        )

    env.close()


if __name__ == "__main__":
    main(sys.argv[1:])
