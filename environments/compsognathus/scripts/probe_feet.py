"""Reproducible foot-isolation research, not training or gate certification.

Mechanical tests need only the core environment dependencies. Replay probes
also need SB3, a locally supplied checkpoint, its exact normalization sidecar,
and the recorded stage_config.json. No checkpoint is downloaded or published.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.metadata
import json
from pathlib import Path
from typing import cast

import mujoco
import numpy as np

from environments.compsognathus import MODEL_PATHS
from environments.compsognathus.envs.compsognathus_env import CompsognathusEnv
from environments.compsognathus.experiments.feet import (
    MECHANISMS,
    Mechanism,
    distal_clearance_xml,
    foot_xml,
    obstacle_xml,
    spectral_metrics,
)
from environments.compsognathus.experiments.foot_probe import FootProbe
from environments.shared.action_filter import low_pass_alpha
from environments.shared.plant_contract import (
    current_plant_identity,
    validate_environment_plant,
    validate_model_plant,
)

REPLAY_VARIANTS = {
    "baseline": ("feet", None, None),
    "feet_lp10": ("feet", 10.0, None),
    "feet_lp5": ("feet", 5.0, None),
    "feet_lp2": ("feet", 2.0, None),
    "ankles_feet_lp10": ("ankles_feet", 10.0, None),
    "all_lp10": ("all", 10.0, None),
    "all_lp5": ("all", 5.0, None),
    "feet_mean_after4s": ("feet", None, "mean"),
    "ankles_feet_mean_after4s": ("ankles_feet", None, "mean"),
    "feet_home_after4s": ("feet", None, "home"),
}


def _save(path: Path, value) -> None:
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def export_panel(path: Path, panel: dict) -> None:
    """Write reviewable trial-level CSV plus metadata; keep nested lists as JSON."""

    def flatten(value: dict, prefix: str = "") -> dict:
        row = {}
        for key, item in value.items():
            name = prefix + key
            if isinstance(item, dict):
                row.update(flatten(item, name + "."))
            else:
                row[name] = json.dumps(item) if isinstance(item, list) else item
        return row

    rows = [flatten(row) for row in panel["episodes"]]
    fields = sorted({key for row in rows for key in row})
    with path.with_suffix(".csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    _save(path.with_suffix(".metadata.json"), panel["metadata"])


def _load_model(xml: str) -> tuple[mujoco.MjModel, mujoco.MjData]:
    m = mujoco.MjModel.from_xml_string(xml)
    d = mujoco.MjData(m)
    mujoco.mj_resetDataKeyframe(m, d, m.key("home").id)
    mujoco.mj_forward(m, d)
    return m, d


def mechanical_episode(xml: str, seed: int, *, duration: float = 20, settle: float = 4) -> dict:
    if not np.isfinite(duration) or not np.isfinite(settle) or not 0 <= settle < duration:
        raise ValueError("Require finite 0 <= settle < duration")
    m, d = _load_model(xml)
    baseline, _ = _load_model(foot_xml("baseline"))
    rng = np.random.default_rng(seed)
    noise = {baseline.joint(j).name: rng.uniform(-0.01, 0.01) for j in range(1, baseline.njnt)}
    for j in range(1, m.njnt):
        d.qpos[m.jnt_qposadr[j]] += noise.get(m.joint(j).name, 0.0)
    mujoco.mj_forward(m, d)
    probe = FootProbe(m, d)
    initial = d.qpos[:2].copy()
    min_load = np.full(2, np.inf)
    unsupported = bilateral = windows = 0
    peak_tilt = 0.0
    reason = "horizon"
    nsteps = round(duration / m.opt.timestep)
    for i in range(nsteps):
        mujoco.mj_step(m, d)
        loads = probe.sample(record=i * m.opt.timestep >= settle)
        min_load = np.minimum(min_load, loads)
        if i % 10 == 9:
            if i * m.opt.timestep >= settle:
                windows += 1
                unsupported += int(np.all(min_load <= 0.1))
                bilateral += int(np.all(min_load > 0.1))
            min_load[:] = np.inf
        tilt = float(np.arccos(np.clip(d.xmat[m.body("pelvis").id].reshape(3, 3)[2, 2], -1, 1)))
        peak_tilt = max(peak_tilt, tilt)
        if not np.isfinite(d.qpos).all() or not np.isfinite(d.qvel).all():
            reason = "nonfinite"
            break
        if probe.body_hit or tilt > 0.7 or not 0.6 * 0.244391440454 < d.qpos[2] < 1.45 * 0.244391440454:
            reason = "body_contact" if probe.body_hit else "height_or_tilt"
            break
    return {
        "seed": seed,
        "duration_s": (i + 1) * m.opt.timestep,
        "full_horizon": reason == "horizon",
        "termination": reason,
        "peak_tilt_deg": float(np.rad2deg(peak_tilt)),
        "final_drift_m": float(np.linalg.norm(d.qpos[:2] - initial)),
        "unsupported_windows": unsupported / windows if windows else None,
        "bilateral_windows": bilateral / windows if windows else None,
        "digit_angles_deg": {
            m.joint(j).name: float(np.rad2deg(d.qpos[m.jnt_qposadr[j]]))
            for j in range(1, m.njnt)
            if "_digits" in m.joint(j).name
        },
        **probe.results(),
    }


class ActionIntervention:
    """Episode-local selected-joint filter or a causal mean/home handoff.

    Mean holds use only commands already observed in seconds 1–4, and ramp
    from the live command over seconds 4–5. No future trace is available.
    """

    def __init__(self, model: mujoco.MjModel, variant: str, dt: float):
        group, cutoff, self.hold = REPLAY_VARIANTS[variant]
        suffixes = ("_toe_act",) if group == "feet" else ("_toe_act", "_ankle_act")
        self.indices = [i for i in range(model.nu) if group == "all" or model.actuator(i).name.endswith(suffixes)]
        self.alpha = low_pass_alpha(cutoff, dt) if cutoff is not None else None
        self.dt = dt
        self.state: np.ndarray | None = None
        self.history: list[np.ndarray] = []
        self.target: np.ndarray | None = None

    def apply(self, raw: np.ndarray, step: int) -> np.ndarray:
        action = np.array(raw, dtype=np.float64, copy=True)
        selected = action[self.indices]
        if self.alpha is not None:
            self.state = selected.copy() if self.state is None else self.state + self.alpha * (selected - self.state)
            action[self.indices] = self.state
        if self.hold is not None:
            t = step * self.dt
            if 1 <= t < 4:
                self.history.append(selected.copy())
            if t >= 4:
                if self.target is None:
                    self.target = np.zeros_like(selected) if self.hold == "home" else np.mean(self.history, axis=0)
                blend = min(1.0, max(0.0, t - 4))
                action[self.indices] = (1 - blend) * selected + blend * self.target
        return action


def _load_policy(model_path: Path, normalization: Path, kwargs: dict):
    import torch
    from stable_baselines3 import PPO
    from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

    torch.set_num_threads(1)
    identity = current_plant_identity("compsognathus")
    # The supplied Colab artifact contains Python 3.13 schedule closures. Replace
    # only training schedule metadata for Python 3.12 diagnostic inference.
    model = PPO.load(
        model_path,
        device="cpu",
        custom_objects={
            "learning_rate": 1e-5,
            "lr_schedule": lambda _: 1e-5,
            "clip_range": lambda _: 0.2,
        },
    )
    validate_model_plant(model, identity, artifact=str(model_path))
    normalizer = VecNormalize.load(str(normalization), DummyVecEnv([lambda: CompsognathusEnv(**kwargs)]))
    normalizer.training = normalizer.norm_reward = False
    validate_model_plant(normalizer, identity, artifact=str(normalization))
    return model, normalizer, identity


def replay_episode(model, normalizer, identity, kwargs: dict, variant: str, seed: int) -> dict:
    env = CompsognathusEnv(**kwargs)
    validate_environment_plant(env, identity, artifact="unmodified foot-probe environment")
    obs, _ = env.reset(seed=seed)
    intervention = ActionIntervention(env.model, variant, env.dt)
    probe = FootProbe(env.model, env.data)

    def sample_substep() -> None:
        probe.sample(record=env._step_count * env.dt >= 4)

    env._substep_probe_hook = sample_substep
    reward_sum = 0.0
    commands: list[np.ndarray] = []
    unsupported = bilateral = windows = 0
    for step in range(env.max_episode_steps):
        raw, _ = model.predict(normalizer.normalize_obs(obs), deterministic=True)
        action = intervention.apply(raw, step)
        obs, reward, done, truncated, info = env.step(action)
        reward_sum += reward
        if step * env.dt >= 4:
            windows += 1
            unsupported += info["unsupported_duty"]
            bilateral += info["bilateral_support_duty"]
            ids = [env.model.actuator(s + "_toe_act").id for s in ("r", "l")]
            commands.append(env.data.ctrl[ids].copy())
        if done or truncated:
            break
    result = {
        "variant": variant,
        "seed": seed,
        "duration_s": (step + 1) * env.dt,
        "full_horizon": (step + 1 == env.max_episode_steps) and not done,
        "reward": reward_sum,
        "termination": info.get("termination_reason", "horizon"),
        "unsupported_windows": unsupported / windows if windows else None,
        "bilateral_windows": bilateral / windows if windows else None,
        "command_spectrum": spectral_metrics(np.asarray(commands), env.dt),
        "final_drift_m": info.get("drift_distance"),
        "diagnostic_intervention": variant != "baseline",
        **probe.results(),
    }
    env.close()
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=["build", "mechanical", "replay"])
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--seeds", type=int, nargs="+", default=[4010, 4011])
    parser.add_argument("--mechanisms", nargs="+", choices=MECHANISMS, default=list(MECHANISMS))
    parser.add_argument("--stiffness", type=float, default=0.15)
    parser.add_argument("--damping", type=float, default=0.002)
    parser.add_argument(
        "--distal-clearance", type=float, default=0.0, help="Separate capsule-envelope ablation, in meters (0 to 0.003)"
    )
    parser.add_argument("--model", type=Path)
    parser.add_argument("--normalization", type=Path)
    parser.add_argument("--stage-config", type=Path)
    parser.add_argument("--variants", nargs="+", choices=list(REPLAY_VARIANTS), default=list(REPLAY_VARIANTS))
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    source_hash = hashlib.sha256(MODEL_PATHS["biological"].read_bytes()).hexdigest()
    metadata = {
        "source_xml_sha256": source_hash,
        "mujoco": mujoco.__version__,
        "numpy": np.__version__,
        "mode": args.mode,
        "seeds": args.seeds,
        "settle_s": 4,
        "horizon_s": 20,
        "research_only": True,
    }
    rows = []
    if args.mode in ("build", "mechanical"):
        metadata.update(
            stiffness_per_digit=args.stiffness,
            damping_per_digit=args.damping,
            armature_per_digit=1e-6,
            distal_clearance_m=args.distal_clearance,
            mechanisms=args.mechanisms,
        )
        for name in args.mechanisms:
            xml = foot_xml(cast(Mechanism, name), stiffness=args.stiffness, damping=args.damping)
            xml = distal_clearance_xml(xml, args.distal_clearance)
            (args.output / (name + ".xml")).write_text(xml)
            if args.mode == "build":
                continue
            cases: list[tuple[str, str | None, str | None, float]] = [("flat", None, None, 0)]
            cases += [
                (s + "_" + loc + "_" + str(mm) + "mm", s, loc, mm / 1000)
                for s in ("r", "l")
                for loc, mm in (("middle", 3), ("middle", 6), ("outer", 3), ("pad", 3))
            ]
            for label, side, location, height in cases:
                scene = obstacle_xml(xml, side=side, location=location, height=height) if side and location else xml
                for seed in args.seeds:
                    r = {"mechanism": name, "case": label, **mechanical_episode(scene, seed)}
                    rows.append(r)
                    _save(args.output / "mechanical.json", {"metadata": metadata, "episodes": rows})
                    print(
                        json.dumps(
                            {
                                k: r[k]
                                for k in (
                                    "mechanism",
                                    "case",
                                    "seed",
                                    "duration_s",
                                    "full_horizon",
                                    "pad_load_fraction",
                                )
                            }
                        ),
                        flush=True,
                    )
    else:
        if not all((args.model, args.normalization, args.stage_config)):
            parser.error("replay requires --model, --normalization and --stage-config")
        config = json.loads(args.stage_config.read_text())
        kwargs = config["reward_weights"]
        if kwargs.get("frame_skip") != 10 or kwargs.get("max_episode_steps") != 1000:
            parser.error("this experiment requires the captured 50 Hz, 20-second stance configuration")
        metadata.update(
            {
                "variants": args.variants,
                "stage_config": config,
                "checkpoint_sha256": hashlib.sha256(args.model.read_bytes()).hexdigest(),
                "normalization_sha256": hashlib.sha256(args.normalization.read_bytes()).hexdigest(),
                "stable_baselines3": importlib.metadata.version("stable-baselines3"),
            }
        )
        model, normalizer, identity = _load_policy(args.model, args.normalization, kwargs)
        for variant in args.variants:
            for seed in args.seeds:
                r = replay_episode(model, normalizer, identity, kwargs, variant, seed)
                rows.append(r)
                _save(args.output / "replay.json", {"metadata": metadata, "episodes": rows})
                print(
                    json.dumps({k: r[k] for k in ("variant", "seed", "duration_s", "reward", "unsupported_windows")}),
                    flush=True,
                )
        normalizer.close()
    _save(args.output / "provenance.json", metadata)
    if rows:
        export_panel(args.output / args.mode, {"metadata": metadata, "episodes": rows})
    if hashlib.sha256(MODEL_PATHS["biological"].read_bytes()).hexdigest() != source_hash:
        raise RuntimeError("Production XML changed during the experiment")


if __name__ == "__main__":
    main()
