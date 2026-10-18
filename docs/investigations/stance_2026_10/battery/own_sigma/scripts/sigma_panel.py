"""Own-sigma (stochastic) heading-0 v2 panel on one checkpoint -- report-only OFFLINE EVALUATION.

Uncertified scratch-recipe checkpoints (r9 arm L symmetry screen); this certifies nothing.

Rolls main's own v2 panel rollout (``stance_report.run_panel(..., floor_truth=True)``, spawn_yaw None,
i.e. the v2 panel / the heading certificate's 0 deg cell) on main's stance stage config (options off), with
one of three policies:

  * ``deterministic``: main's ``_load_policy`` predict (SB3 ``deterministic=True``) -- a harness check that
    must reproduce the earlier certificate's 0 deg cell row for row;
  * ``stochastic``: the policy's own Gaussian, a = clip(mu(o) + sigma * z, low, high), mu and sigma read from
    ``policy.get_distribution`` (sigma = exp(log_std), state independent), z drawn from a numpy Generator
    fixed per episode: ``default_rng(SeedSequence(entropy=episode_seed, spawn_key=(1,)))`` -- child 1 of the
    episode seed's SeedSequence, so it never replays the env's own reset stream (gymnasium seeds
    ``np_random`` with ``SeedSequence(episode_seed)`` itself).  Same z stream for every checkpoint
    (common random numbers).  This mirrors SB3's ``predict(deterministic=False)`` (sample, then clip to
    the Box) with numpy in place of torch's RNG;
  * ``statue``: main's zero-action statue (the v2 statue-relative reference).

Every episode is judged with main's ``evaluate_stance_v2_gate`` / ``classify_stance_episode`` against the
statue reference rolled on the same seeds.  The floor-truth trace of every episode is also reduced to the
signed fore-aft pad CoP (main's recorder ``sole_cop[..., 0]``, pad half-lengths, +1 toe edge, -1 heel edge)
by wrapping ``stance_report.episode_stance_metrics`` (the metrics it returns are untouched).
"""
import argparse, hashlib, json, logging, math, sys, time
from pathlib import Path

import numpy as np

ap = argparse.ArgumentParser()
ap.add_argument("--label", required=True)
ap.add_argument("--mode", choices=["stochastic", "deterministic", "statue"], required=True)
ap.add_argument("--model")
ap.add_argument("--vecnorm")
ap.add_argument("--out", required=True)
ap.add_argument("--seed", type=int, default=3042)
ap.add_argument("--episodes", type=int, default=40)
ap.add_argument("--statue-ref", help="statue panel.json whose reference judges this panel")
args = ap.parse_args()

logging.basicConfig(level=logging.WARNING, format="%(asctime)s %(name)s %(message)s")
import torch  # noqa: E402

torch.set_num_threads(1)
import environments  # noqa: E402

assert "/scratchpad/fix/main_ro/" in environments.__file__, environments.__file__
from environments.shared.config import load_stage_config  # noqa: E402
from environments.shared.curriculum.gate_schema import gate_config_sha256, gate_config_view  # noqa: E402
from environments.shared.curriculum.stance_gate_v2 import (  # noqa: E402
    StanceV2Thresholds,
    StatueReference,
    evaluate_stance_v2_gate,
    statue_reference,
)
from environments.shared.model_id import ModelId  # noqa: E402
from environments.shared.plant_contract import current_plant_identity  # noqa: E402
from environments.shared.policy_loading import load_sb3_checkpoint  # noqa: E402
from environments.shared.reporting import stance_report as sr  # noqa: E402
from environments.shared.species_registry import env_class as model_env_class  # noqa: E402


def sha(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def fnum(x):
    if x is None:
        return None
    x = float(x)
    return x if math.isfinite(x) else None


mid = ModelId("tyrannosaurus_rex", "anatomical")
cfg = load_stage_config(mid, 1)
curriculum = cfg["curriculum_kwargs"]
assert curriculum["gate_kind"] == "stance_quality_heading/v1"
env_kwargs = dict(cfg["env_kwargs"])
horizon = int(env_kwargs.get("max_episode_steps", 1000))
control_dt = float(env_kwargs.get("timestep", 0.002)) * int(env_kwargs.get("frame_skip", 5))
thresholds = StanceV2Thresholds.from_curriculum(curriculum)
thresholds.validate_settle_window(control_dt)
settle = thresholds.settle_steps
plant_identity = current_plant_identity(mid)
env_class = model_env_class(mid)
probe = env_class(**env_kwargs)
try:
    actuator_names = sr._actuator_joint_names(probe)
finally:
    probe.close()

# ---- trace capture: wrap main's per-episode reduction, return its metrics unchanged ----
captured = []
_orig_metrics = sr.episode_stance_metrics


def _capturing_metrics(trace, *, settle_steps):
    m = _orig_metrics(trace, settle_steps=settle_steps)
    T = int(m.length)
    sc = np.asarray(trace.sole_cop, dtype=np.float64)[:T, :, 0]  # (T, F) signed fore-aft, half-lengths
    contacts = np.asarray(trace.sole_contacts_mean, dtype=np.float64)[:T]
    lift = np.asarray(trace.sole_corner_lift, dtype=np.float64)[:T]
    tilt = np.asarray(trace.sole_tilt_deg, dtype=np.float64)[:T]
    w = slice(settle_steps, T)
    loaded = np.isfinite(sc) & (contacts > 0)
    rec = {"length": T, "termination_reason": getattr(trace, "termination_reason", None)}
    if T > settle_steps:
        has_sole = np.isfinite(tilt[w]).all(axis=0) & np.isfinite(np.asarray(trace.home_sole_tilt_deg, dtype=np.float64))
        box = has_sole & np.isfinite(lift[w]).all(axis=0)
        lw, fw = loaded[w], sc[w]
        unsigned = np.where(lw, np.abs(fw), 1.0).mean(axis=0)  # the gate's per-foot reading
        signed = np.array([fw[lw[:, k], k].mean() if lw[:, k].any() else np.nan for k in range(sc.shape[1])])
        rec.update(
            box_feet=[bool(b) for b in box],
            unsigned_per_foot=[fnum(v) for v in unsigned],
            recomputed_max_sole_cop_fore_aft=fnum(unsigned[box].max()) if box.any() else None,
            signed_per_foot=[fnum(v) for v in signed],
            loaded_fraction_per_foot=[fnum(v) for v in lw.mean(axis=0)],
            toe_edge_fraction_per_foot=[fnum(v) for v in (lw & (fw >= 0.95)).mean(axis=0)],
            heel_edge_fraction_per_foot=[fnum(v) for v in (lw & (fw <= -0.95)).mean(axis=0)],
        )
        # the signed pad CoP of the episode: the gate foot (the box foot with the largest unsigned reading)
        gate_foot = int(np.flatnonzero(box)[np.argmax(unsigned[box])]) if box.any() else None
        rec["gate_foot"] = gate_foot
        rec["signed_pad_cop_gate_foot"] = None if gate_foot is None else fnum(signed[gate_foot])
        finite = [(abs(v), v) for v in signed[box] if np.isfinite(v)]
        rec["signed_pad_cop_extreme_foot"] = fnum(max(finite)[1]) if finite else None
    captured.append((rec, np.where(loaded, sc, np.nan).astype(np.float32)))
    return m


sr.episode_stance_metrics = _capturing_metrics

# ---- policy ----
meta = {
    "REPORT_ONLY": "offline evaluation of an uncertified scratch-recipe checkpoint; certifies nothing",
    "label": args.label, "mode": args.mode, "environments_file": environments.__file__,
    "seeds": [args.seed, args.seed + args.episodes - 1], "episodes": args.episodes, "heading_deg": 0.0,
    "spawn_yaw": None, "options": "off (main's stance stage config as loaded)",
    "horizon": horizon, "control_dt": control_dt, "settle_steps": settle,
    "gate_sha256": gate_config_sha256(gate_config_view(curriculum)),
    "thresholds": gate_config_view(curriculum)["thresholds"],
    "actuator_names": actuator_names,
}
policy_obj = None
if args.mode == "statue":
    predict = sr._zero_action_predict(env_class, env_kwargs)
else:
    meta.update(model=args.model, model_sha256=sha(args.model), vecnorm=args.vecnorm, vecnorm_sha256=sha(args.vecnorm))
    if args.mode == "deterministic":
        predict, note = sr._load_policy(args.model, args.vecnorm, lambda: env_class(**env_kwargs),
                                        plant_identity=plant_identity)
        meta["policy_note"] = note
        meta["normalization"] = getattr(predict, "normalization", None)
    model, normalizer, _ = load_sb3_checkpoint(args.model, args.vecnorm, lambda: env_class(**env_kwargs),
                                               guess_sidecar=False, allow_unnormalized=True,
                                               plant_identity=plant_identity, allow_legacy_plant=False)
    pol = model.policy
    assert not getattr(pol, "use_sde", False)
    log_std = pol.log_std.detach().cpu().numpy().astype(np.float64)
    meta["log_std"] = log_std.tolist()
    meta["std_per_actuator"] = {n: float(np.exp(v)) for n, v in zip(actuator_names, log_std)}
    meta["std_mean"] = float(np.exp(log_std).mean())
    meta["policy_class"] = f"{type(model).__name__}/{type(pol).__name__}/{type(pol.action_dist).__name__}"
    low = np.asarray(model.action_space.low, dtype=np.float32)
    high = np.asarray(model.action_space.high, dtype=np.float32)
    mean = np.asarray(normalizer.obs_rms.mean, dtype=np.float64)
    var = np.asarray(normalizer.obs_rms.var, dtype=np.float64)
    eps_n, clip_o = float(normalizer.epsilon), float(normalizer.clip_obs)

    class OwnSigma:
        """a = clip(mu + sigma * z): the policy's own Gaussian, z from a per-episode fixed numpy Generator."""

        def __init__(self):
            self.index = -1
            self.rng = None
            self.max_mode_vs_repo_predict = 0.0
            self.max_dist_std_vs_exp_log_std = 0.0
            self.steps = []
            self.clipped = []
            self.z_abs_sum = []
            self.normalization = {"mode": "frozen VecNormalize (as _load_policy)", "clip_obs": clip_o, "epsilon": eps_n}

        def reset(self):  # _roll_episodes calls this once per episode, right after env.reset(seed=seed+index)
            self.index += 1
            ss = np.random.SeedSequence(entropy=args.seed + self.index, spawn_key=(1,))
            self.rng = np.random.default_rng(ss)
            self.steps.append(0)
            self.clipped.append(np.zeros(len(low), dtype=np.int64))
            self.z_abs_sum.append(0.0)

        def __call__(self, obs):
            normalized = np.clip((obs - mean) / np.sqrt(var + eps_n), -clip_o, clip_o).astype(np.float32)
            with torch.no_grad():
                dist = pol.get_distribution(torch.as_tensor(normalized[None]))
                mu = dist.distribution.mean.cpu().numpy()[0]
                std = dist.distribution.stddev.cpu().numpy()[0]
            det = np.asarray(model.predict(normalized, deterministic=True)[0])
            self.max_mode_vs_repo_predict = max(self.max_mode_vs_repo_predict, float(np.max(np.abs(np.clip(mu, low, high) - det))))
            self.max_dist_std_vs_exp_log_std = max(self.max_dist_std_vs_exp_log_std, float(np.max(np.abs(std - np.exp(log_std)))))
            z = self.rng.standard_normal(mu.shape[0]).astype(np.float32)
            raw = mu + std * z
            self.steps[-1] += 1
            self.clipped[-1] += (raw < low) | (raw > high)
            self.z_abs_sum[-1] += float(np.abs(z).mean())
            return np.clip(raw, low, high)

    if args.mode == "stochastic":
        policy_obj = OwnSigma()
        predict = policy_obj
        meta["noise"] = ("a = clip(mu + sigma*z, -1, 1); z ~ N(0, I) from np.random.default_rng("
                         "SeedSequence(entropy=episode_seed, spawn_key=(1,))), episode_seed = seed + index")

t0 = time.time()
result = sr.run_panel(mid, predict=predict, episodes=args.episodes, seed=args.seed, settle_steps=settle, horizon=horizon,
                      env_kwargs=env_kwargs, plant_identity=plant_identity, control_dt=control_dt, floor_truth=True,
                      spawn_yaw=None)
meta["seconds"] = time.time() - t0
metrics = result["stance_metrics"]
assert len(captured) == len(metrics) == args.episodes
if policy_obj is not None:
    assert policy_obj.index == args.episodes - 1, policy_obj.index
    meta["check_max_abs_clip_mu_minus_repo_deterministic_predict"] = policy_obj.max_mode_vs_repo_predict
    meta["check_max_abs_dist_std_minus_exp_log_std"] = policy_obj.max_dist_std_vs_exp_log_std

out = Path(args.out)
out.mkdir(parents=True, exist_ok=True)
if args.mode == "statue":
    ref = statue_reference(metrics, horizon=horizon)
else:
    ref = StatueReference.from_dict(json.loads(Path(args.statue_ref).read_text())["statue_reference"])
v2 = evaluate_stance_v2_gate(metrics, thresholds, horizon=horizon, statue=ref, control_dt=control_dt)
reasons = v2.episode_reasons
rows = sr._stance_v2_episode_rows(metrics, reasons, result["episodes"], seed=args.seed, horizon=horizon)
for i, row in enumerate(rows):
    rec = captured[i][0]
    row["reward"] = float(result["rewards"][i])
    row["termination_reason"] = rec["termination_reason"]
    row["pad"] = {k: v for k, v in rec.items() if k not in ("length", "termination_reason")}
    if policy_obj is not None:
        row["noise"] = {"steps": policy_obj.steps[i], "clip_fraction_per_actuator": (policy_obj.clipped[i] / max(policy_obj.steps[i], 1)).tolist(),
                        "mean_abs_z": policy_obj.z_abs_sum[i] / max(policy_obj.steps[i], 1)}
    g = row.get("max_sole_cop_fore_aft")
    r = rec.get("recomputed_max_sole_cop_fore_aft")
    row["pad"]["recompute_matches_gate_metric"] = (g is None and r is None) or (g is not None and r is not None and abs(g - r) < 1e-12) or (
        g is not None and r is not None and not math.isfinite(g))
v2d = v2.as_dict()
payload = {"meta": meta, "v2_result": v2d, "statue_reference": ref.as_dict(), "terminations": result["terminations"],
           "reward_mean": float(result["rewards"].mean()), "action": result.get("action", {}), "episodes": rows}
(out / "panel.json").write_text(json.dumps(payload, indent=1, default=lambda o: o.tolist() if hasattr(o, "tolist") else str(o)))
L = max(c[1].shape[0] for c in captured)
F = captured[0][1].shape[1]
cop = np.full((len(captured), L, F), np.nan, dtype=np.float16)
for i, (_, a) in enumerate(captured):
    cop[i, : a.shape[0]] = a.astype(np.float16)
np.savez_compressed(out / "pad_cop_trace.npz", sole_cop_fore_aft_loaded=cop, lengths=np.array([c[1].shape[0] for c in captured]),
                    seeds=np.arange(args.seed, args.seed + args.episodes), settle_steps=settle)
falls = [r["seed"] for r in rows if not r["reached_horizon"]]
print(json.dumps({"label": args.label, "mode": args.mode, "seconds": round(meta["seconds"], 1), "n_clean": v2d.get("n_clean"),
                  "full_horizon_fraction": v2d.get("full_horizon_fraction"), "falls": falls, "passed": v2d.get("passed")}))
