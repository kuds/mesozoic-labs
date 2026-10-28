"""Zero-shot closed-loop panel of one r9 checkpoint (or the zero-action statue) on the r9 or SR3 plant.

Report only: never trains, never steps an optimizer, writes only under --out.

    PY -I -B zs_harness.py --code CODE_COPY --plant r9|sr3 [--sr3-xml X] --mode det|stoch|statue
        [--zip Z --pkl P] [--stoch-seed 3042] [--seed 3042 --episodes 40] [--statue-json S] --out DIR

What it runs is the arm's own evaluation path, imported from the code copy CODE_COPY (a git archive of the arm's
commit; nothing in it is modified):
  * the stage config: config.load_stage_config(tyrannosaurus_rex/anatomical, "stance") (env_kwargs, curriculum);
  * the policy: reporting.stance_report._load_policy (deterministic mode) -- load_sb3_checkpoint, frozen
    VecNormalize, obs = clip((obs - mean) / sqrt(var + eps), +-clip).astype(float32), model.predict(...,
    deterministic=True) (SB3 clips to the action bound) -- with the checkpoint and its statistics validated against
    the code copy's current (r9) plant identity as they load;
  * the rollout and the gate metrics: reporting.stance_report.run_panel(..., floor_truth=True), i.e. env.reset(seed=
    seed + i), the SubstepContactRecorder, gait.stance_metrics.episode_stance_metrics(settle_steps);
  * the gate: curriculum.stance_gate_v2.StanceV2Thresholds.from_curriculum, statue_reference,
    classify_stance_episode, evaluate_stance_v2_gate.

Scratch-only changes, all in THIS file (the bypass is recorded in harness_bypass.patch):
  1. stance_report.model_env_class is replaced in this process by a subclass of the arm's
     TyrannosaurusRexAnatomicalEnv whose only additions are (a) a step() override that calls the inherited step
     unchanged and then READS the state into a recorder (a private MjData copy is used for kinematics, the live
     MjData is never written), and (b) for --plant sr3, MODEL_PATH = the SR3 MJCF.  step() is not part of the
     plant contract's policy-interface digest (it hashes _get_obs, _scale_action, _cache_ids, reset), and on the
     r9 plant run_panel's validate_environment_plant runs on this very subclass and must pass.
  2. --plant sr3: run_panel is called with plant_identity=None, which skips its validate_environment_plant(env,
     current r9 identity) -- the one check that refuses the SR3 env.  Before that, this file calls
     validate_environment_plant on an SR3 env itself and proceeds only if the ONLY mismatch it reports is
     physics_sha256 (policy_interface_sha256, nq, nv, nu, observation_dim, action_dim all equal).  The checkpoint
     and its VecNormalize statistics are still validated against the r9 identity (they are r9 artifacts).
  3. --mode stoch: the command is clip(raw_mean + exp(log_std) * z, low, high), raw_mean the policy's Gaussian mean
     (policy.get_distribution(obs).mode() under no_grad), z = numpy.random.default_rng(stoch_seed + i)
     .standard_normal(nu) per step in episode i -- the screens' noise_matched_own / episode_trace --stochastic draw.
     The episode index advances through run_panel's predict.reset() hook.
"""
from __future__ import annotations

import argparse
import dataclasses
import hashlib
import json
import math
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

sys.dont_write_bytecode = True

ap = argparse.ArgumentParser()
ap.add_argument("--code", required=True)
ap.add_argument("--plant", choices=("r9", "sr3"), required=True)
ap.add_argument("--sr3-xml")
ap.add_argument("--mode", choices=("det", "stoch", "statue"), required=True)
ap.add_argument("--zip")
ap.add_argument("--pkl")
ap.add_argument("--stoch-seed", type=int, default=3042)
ap.add_argument("--seed", type=int, default=3042)
ap.add_argument("--episodes", type=int, default=40)
ap.add_argument("--statue-json")
ap.add_argument("--label", default="")
ap.add_argument("--out", required=True)
A = ap.parse_args()

CODE = str(Path(A.code).resolve())
sys.path.insert(0, CODE)

import mujoco  # noqa: E402
import numpy as np  # noqa: E402

from environments.shared.config import load_stage_config  # noqa: E402
from environments.shared.curriculum.stance_gate_v2 import (  # noqa: E402
    StanceV2Thresholds,
    StatueReference,
    classify_stance_episode,
    evaluate_stance_v2_gate,
    statue_reference,
)
from environments.shared.model_id import ModelId  # noqa: E402
from environments.shared.plant_contract import current_plant_identity  # noqa: E402
from environments.shared.plant_contract.errors import PlantCompatibilityError  # noqa: E402
from environments.shared.plant_contract.validation import validate_environment_plant  # noqa: E402
from environments.shared.policy_loading import load_sb3_checkpoint  # noqa: E402
from environments.shared.reporting import stance_report as SR  # noqa: E402
from environments.tyrannosaurus_rex.anatomical.env import TyrannosaurusRexAnatomicalEnv as Base  # noqa: E402

T0 = time.time()
OUTD = Path(A.out)
OUTD.mkdir(parents=True, exist_ok=True)
MID = ModelId("tyrannosaurus_rex", "anatomical")
CFG = load_stage_config(MID, "stance")
ENV_KW = dict(CFG["env_kwargs"])
CUR = CFG["curriculum_kwargs"]
HORIZON = int(ENV_KW.get("max_episode_steps", 1000))
CONTROL_DT = float(ENV_KW.get("timestep", 0.002)) * int(ENV_KW.get("frame_skip", 5))
TH = StanceV2Thresholds.from_curriculum(CUR)
SETTLE = int(TH.settle_steps)
PID = current_plant_identity(MID)
SAT = 0.99


def sha256_file(p):
    h = hashlib.sha256()
    with open(p, "rb") as fh:
        for c in iter(lambda: fh.read(1 << 20), b""):
            h.update(c)
    return h.hexdigest()


def utc():
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


# ------------------------------------------------------------------------------------------- the step recorder
class StepRecorder:
    """Reads the state after every env.step into per-episode buffers; never writes the live MjData."""

    def __init__(self):
        self.episodes = []
        self.series = []
        self.norm = None  # (mean, var, eps, clip) to measure the observation's distance from the training stats
        self._new()

    def _new(self):
        self.buf = {k: [] for k in ("cmd", "tq", "chord_base", "chord_dist", "seg_last", "tip_z", "com_fa", "com_lat",
                                    "pitch", "roll", "pz", "reward", "tailq", "objz", "obs_zmax", "obs_nclip",
                                    "head_z", "neck_q", "headq")}

    def bind(self, env):
        m = env.model
        self.scratch = mujoco.MjData(m)
        J = mujoco.mjtObj.mjOBJ_JOINT
        self.tail_qadr = [int(m.jnt_qposadr[mujoco.mj_name2id(m, J, n)])
                          for n in ("tail_1_pitch", "tail_1_yaw", "tail_2_pitch", "tail_3_pitch")]
        B = mujoco.mjtObj.mjOBJ_BODY
        S = mujoco.mjtObj.mjOBJ_SITE
        self.pel = mujoco.mj_name2id(m, B, "pelvis")
        self.t1 = mujoco.mj_name2id(m, B, "tail_1")
        self.t3 = mujoco.mj_name2id(m, B, "tail_3")
        self.t5 = mujoco.mj_name2id(m, B, "tail_5")
        self.tip = mujoco.mj_name2id(m, S, "tail_tip")
        self.feet = [mujoco.mj_name2id(m, S, s) for s in ("r_foot", "l_foot")]
        self.act_names = [mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_ACTUATOR, i) for i in range(m.nu)]
        self.qpos_spring = [float(m.qpos_spring[a]) for a in self.tail_qadr]
        self.obs_tail_idx = [a - 7 for a in self.tail_qadr]  # obs = qpos[7:], ...
        self.neck_qadr = [int(m.jnt_qposadr[mujoco.mj_name2id(m, J, n)]) for n in ("neck_pitch", "head_pitch")]
        self.head_tip = mujoco.mj_name2id(m, S, "head_tip")

    def record(self, env, action, obs, reward, terminated, truncated, info):
        m, d, s = env.model, env.data, self.scratch
        s.qpos[:] = d.qpos
        mujoco.mj_kinematics(m, s)
        mujoco.mj_comPos(m, s)
        b = self.buf
        b["cmd"].append(np.asarray(action, dtype=np.float64).ravel().copy())
        b["tq"].append(d.qpos[self.tail_qadr].copy())
        R = s.xmat[self.pel].reshape(3, 3)
        hx, hy = R[0, 0], R[1, 0]
        hn = math.hypot(hx, hy)
        hx, hy = hx / hn, hy / hn
        b["pitch"].append(math.degrees(math.atan2(-R[2, 0], math.hypot(R[0, 0], R[1, 0]))))  # nose-down positive
        b["roll"].append(math.degrees(math.atan2(R[2, 1], R[2, 2])))
        b["pz"].append(float(s.xpos[self.pel][2]))
        com = s.subtree_com[self.pel]
        fc = 0.5 * (s.site_xpos[self.feet[0]] + s.site_xpos[self.feet[1]])
        dx, dy = com[0] - fc[0], com[1] - fc[1]
        b["com_fa"].append(1000.0 * (dx * hx + dy * hy))  # mm, + = CoM ahead of the feet's centre
        b["com_lat"].append(1000.0 * (-dx * hy + dy * hx))  # mm, + = left
        tip, p1, p3, p5 = s.site_xpos[self.tip], s.xpos[self.t1], s.xpos[self.t3], s.xpos[self.t5]

        def ang(a, c):
            v = c - a
            return math.degrees(math.atan2(v[2], math.hypot(v[0], v[1])))

        b["chord_base"].append(ang(p1, tip))
        b["chord_dist"].append(ang(p3, tip))
        b["seg_last"].append(ang(p5, tip))
        b["tip_z"].append(float(tip[2]))
        b["reward"].append(float(reward))
        b["tailq"].append(float(info.get("tail_home_pose_quality", math.nan)))
        b["head_z"].append(float(s.site_xpos[self.head_tip][2]))  # v3 (output only)
        b["neck_q"].append(np.degrees(d.qpos[self.neck_qadr]).copy())
        b["headq"].append(float(info.get("head_clearance_quality", math.nan)))
        if self.norm is not None:
            mean, var, eps, clip = self.norm
            z = (np.asarray(obs, dtype=np.float64) - mean) / np.sqrt(var + eps)
            b["objz"].append(z[self.obs_tail_idx].copy())
            b["obs_zmax"].append(float(np.max(np.abs(z))))
            b["obs_nclip"].append(int(np.sum(np.abs(z) > clip)))
        if terminated or truncated:
            self._finish(env, terminated, truncated, info)

    def _finish(self, env, terminated, truncated, info):
        b = self.buf
        L = len(b["cmd"])
        cmd = np.asarray(b["cmd"])
        tq = np.degrees(np.asarray(b["tq"]))
        w = slice(SETTLE, L)
        nwin = max(L - SETTLE, 0)
        names = self.act_names
        t3 = names.index("tail_3_pitch_act")
        ep = {"length": L, "terminated": bool(terminated), "truncated": bool(truncated),
              "reached_horizon": L >= HORIZON,
              "termination_reason": str(info.get("termination_reason", "terminated" if terminated else "truncated")),
              "return_seq": float(sum(b["reward"])), "window_steps": nwin}
        pitch_end, roll_end = b["pitch"][-1], b["roll"][-1]
        ep.update(end_pitch_deg=pitch_end, end_roll_deg=roll_end, end_pelvis_z=b["pz"][-1],
                  end_com_fa_mm=b["com_fa"][-1], end_chord_dist_deg=b["chord_dist"][-1])
        if L >= HORIZON:
            ep["fall_dir"] = "none"
        elif abs(roll_end) > abs(pitch_end):
            ep["fall_dir"] = "side"
        else:
            ep["fall_dir"] = "forward" if pitch_end > 0 else "backward"
        first = np.flatnonzero(cmd[:, t3] >= SAT)
        ep["tail3_first_step_ge_0p99"] = int(first[0]) if first.size else None
        if nwin:
            cw = cmd[w]
            sat = (np.abs(cw) >= SAT).mean(axis=0)
            others = [i for i in range(len(names)) if i != t3]
            oi = int(others[int(np.argmax(sat[others]))])
            ep.update(
                tail3_cmd_mean=float(cw[:, t3].mean()),
                tail3_frac99=float((cw[:, t3] >= SAT).mean()),
                tail3_frac99_abs=float(sat[t3]),
                tail3_share_le_m0p99=float((cw[:, t3] <= -SAT).mean()),
                tail1_cmd_mean=float(cw[:, names.index("tail_1_pitch_act")].mean()),
                tail2_cmd_mean=float(cw[:, names.index("tail_2_pitch_act")].mean()),
                q1_deg=float(tq[w, 0].mean()), q1yaw_deg=float(tq[w, 1].mean()),
                q2_deg=float(tq[w, 2].mean()), q3_deg=float(tq[w, 3].mean()),
                q3_rad=float(np.radians(tq[w, 3]).mean()),
                chord_base_deg=float(np.mean(b["chord_base"][SETTLE:])),
                chord_dist_deg=float(np.mean(b["chord_dist"][SETTLE:])),
                seg_last_deg=float(np.mean(b["seg_last"][SETTLE:])),
                tip_z_m=float(np.mean(b["tip_z"][SETTLE:])),
                com_fa_mm=float(np.mean(b["com_fa"][SETTLE:])),
                com_lat_mm=float(np.mean(b["com_lat"][SETTLE:])),
                pitch_deg=float(np.mean(b["pitch"][SETTLE:])),
                roll_deg=float(np.mean(b["roll"][SETTLE:])),
                pelvis_z_m=float(np.mean(b["pz"][SETTLE:])),
                others_max_sat=float(sat[others].max()), others_max_sat_actuator=names[oi] if sat[oi] > 0 else "",
                all_max_sat=float(sat.max()),
                tail_home_pose_quality=float(np.nanmean(b["tailq"][SETTLE:])),
                head_tip_z_m=float(np.mean(b["head_z"][SETTLE:])),
                neck_pitch_deg=float(np.asarray(b["neck_q"])[w, 0].mean()),
                head_pitch_deg=float(np.asarray(b["neck_q"])[w, 1].mean()),
                head_clearance_quality=float(np.nanmean(b["headq"][SETTLE:])),
            )
            if b["objz"]:
                oz = np.asarray(b["objz"])[w]
                ep.update(obsz_tail1=float(oz[:, 0].mean()), obsz_tail2=float(oz[:, 2].mean()),
                          obsz_tail3=float(oz[:, 3].mean()),
                          obs_zmax_mean=float(np.mean(b["obs_zmax"][SETTLE:])),
                          obs_share_steps_any_clipped=float(np.mean(np.asarray(b["obs_nclip"][SETTLE:]) > 0)))
        self.episodes.append(ep)
        self.series.append({"cmd_t3": cmd[:, t3].astype(np.float32), "tq_deg": tq.astype(np.float32),
                            "com_fa_mm": np.asarray(b["com_fa"], np.float32),
                            "pitch_deg": np.asarray(b["pitch"], np.float32),
                            "chord_dist_deg": np.asarray(b["chord_dist"], np.float32)})
        self._new()


REC = StepRecorder()


def _step(self, action):
    out = Base.step(self, action)
    if getattr(self, "_zs_bound", False) is False:
        REC.bind(self)
        self._zs_bound = True
    obs, reward, terminated, truncated, info = out
    REC.record(self, action, obs, reward, terminated, truncated, info)
    return out


attrs = {"step": _step}
if A.plant == "sr3":
    attrs["MODEL_PATH"] = Path(A.sr3_xml).resolve()
PlantEnv = type(f"ZeroShotTRex_{A.plant}", (Base,), attrs)
SR.model_env_class = lambda model_id: PlantEnv  # run_panel builds its env through this name

# ------------------------------------------------------------------------------------------- plant check
probe = PlantEnv(**ENV_KW)
plant_check = {"model_path": str(probe.MODEL_PATH), "qpos_spring_tail_deg": None}
m0 = probe.model
plant_check["qpos_spring_tail_deg"] = [math.degrees(float(m0.qpos_spring[m0.jnt_qposadr[mujoco.mj_name2id(m0, mujoco.mjtObj.mjOBJ_JOINT, n)]]))
                                       for n in ("tail_1_pitch", "tail_2_pitch", "tail_3_pitch")]
try:
    validate_environment_plant(probe, PID, artifact=f"zeroshot {A.plant} env")
    plant_check["validate_environment_plant"] = "passed"
    plant_check["mismatches"] = []
except PlantCompatibilityError as exc:
    msg = str(exc)
    plant_check["validate_environment_plant"] = msg
    plant_check["mismatches"] = [ln[2:].split(":")[0] for ln in msg.splitlines() if ln.startswith("- ")]
probe.close()
if A.plant == "r9" and plant_check["mismatches"]:
    raise SystemExit(f"r9 env failed the plant check: {plant_check}")
if A.plant == "sr3" and plant_check["mismatches"] != ["physics_sha256"]:
    raise SystemExit(f"SR3 env differs from r9 in more than physics_sha256: {plant_check}")

# ------------------------------------------------------------------------------------------- the policy
policy_rec = {"mode": A.mode}
if A.mode == "statue":
    predict = SR._zero_action_predict(PlantEnv, ENV_KW)
    policy_rec["description"] = "zero action (stance_report._zero_action_predict)"
else:
    policy_rec.update(zip=A.zip, pkl=A.pkl, zip_sha256=sha256_file(A.zip), pkl_sha256=sha256_file(A.pkl))
    # The statistics, read through the same loader, for the observation-distance diagnostic and the stochastic mode.
    model, normalizer, _ = load_sb3_checkpoint(A.zip, A.pkl, lambda: Base(**ENV_KW), guess_sidecar=False,
                                               allow_unnormalized=True, plant_identity=PID)
    mean = np.asarray(normalizer.obs_rms.mean, dtype=np.float64)
    var = np.asarray(normalizer.obs_rms.var, dtype=np.float64)
    eps, clip = float(normalizer.epsilon), float(normalizer.clip_obs)
    REC.norm = (mean, var, eps, clip)
    policy_rec.update(algorithm=type(model).__name__, clip_obs=clip, epsilon=eps,
                      obs_rms_tail={"mean": mean[[3, 4, 5, 6]].tolist(), "std": np.sqrt(var[[3, 4, 5, 6]]).tolist()})
    if A.mode == "det":
        predict, note = SR._load_policy(A.zip, A.pkl, lambda: Base(**ENV_KW), plant_identity=PID)
        policy_rec["description"] = f"stance_report._load_policy: {note}; predict(deterministic=True)"
    else:
        import torch

        pol = model.policy
        pol.set_training_mode(False)
        sigma = np.exp(pol.log_std.detach().cpu().numpy().astype(np.float64).reshape(-1))
        probe = Base(**ENV_KW)
        low = np.asarray(probe.action_space.low, dtype=np.float64)
        high = np.asarray(probe.action_space.high, dtype=np.float64)
        probe.close()
        policy_rec.update(sigma=sigma.tolist(), stoch_seed=A.stoch_seed,
                          description="clip(raw_mean + exp(log_std) z), z ~ default_rng(stoch_seed + episode index)")

        class StochPredict:
            def __init__(self):
                self.index = -1
                self.rng = None

            def reset(self):  # run_panel calls this at the start of every episode
                self.index += 1
                self.rng = np.random.default_rng(A.stoch_seed + self.index)

            def __call__(self, obs):
                o = np.clip((obs - mean) / np.sqrt(var + eps), -clip, clip).astype(np.float32)
                with torch.no_grad():
                    t, _ = pol.obs_to_tensor(o)
                    raw = np.asarray(pol.get_distribution(t).mode().cpu().numpy(), dtype=np.float64).reshape(-1)
                return np.clip(raw + sigma * self.rng.standard_normal(raw.shape[0]), low, high)

        predict = StochPredict()

# ------------------------------------------------------------------------------------------- roll
res = SR.run_panel(MID, predict=predict, episodes=A.episodes, seed=A.seed, settle_steps=SETTLE, horizon=HORIZON,
                   env_kwargs=ENV_KW, plant_identity=(PID if A.plant == "r9" else None), control_dt=CONTROL_DT,
                   floor_truth=True)
metrics = res["stance_metrics"]
assert len(metrics) == len(REC.episodes) == A.episodes, (len(metrics), len(REC.episodes))

if A.mode == "statue":
    statue = statue_reference(metrics, horizon=HORIZON)
    statue_src = "this panel"
else:
    statue = StatueReference.from_dict(json.load(open(A.statue_json))["summary"]["statue_reference"])
    statue_src = A.statue_json
verdict = evaluate_stance_v2_gate(metrics, TH, horizon=HORIZON, statue=statue, control_dt=CONTROL_DT)

rows = []
for i, (ep, mt, pe) in enumerate(zip(REC.episodes, metrics, res["episodes"])):
    assert ep["length"] == int(mt.length) == int(pe["length"]), (i, ep["length"], mt.length, pe["length"])
    reasons = classify_stance_episode(mt, TH, horizon=HORIZON, statue=statue)
    assert tuple(reasons) == tuple(verdict.episode_reasons[i])
    row = {"seed": A.seed + i, "episode": i, **ep, "return_run_panel": float(pe["reward"]),
           "return_metrics": float(mt.reward), "gate_clean": not reasons,
           "gate_failed_rails": ";".join(sorted({r.split(":", 1)[0] for r in reasons})),
           "gate_reasons": " | ".join(reasons),
           "gate_max_actuator_saturation_fraction": float(mt.max_actuator_saturation_fraction)}
    if "all_max_sat" in ep:
        row["sat_consistency_abs_diff"] = abs(ep["all_max_sat"] - float(mt.max_actuator_saturation_fraction))
    rows.append(row)

np.savez_compressed(OUTD / "series.npz", **{f"{k}_{r['seed']}": v for r, s in zip(rows, REC.series) for k, v in s.items()})
summary = {
    "schema": "zeroshot-panel/1", "label": A.label, "utc_end": utc(), "seconds": round(time.time() - T0, 1),
    "code": CODE, "plant": A.plant, "sr3_xml": A.sr3_xml, "sr3_xml_sha256": sha256_file(A.sr3_xml) if A.sr3_xml else None,
    "plant_check": plant_check, "r9_identity": {"physics_sha256": PID.physics_sha256,
                                                "policy_interface_sha256": PID.policy_interface_sha256,
                                                "physics_revision": PID.physics_revision,
                                                "policy_interface_revision": PID.policy_interface_revision},
    "policy": policy_rec, "seeds": [A.seed, A.seed + A.episodes - 1], "horizon": HORIZON, "settle_steps": SETTLE,
    "control_dt": CONTROL_DT, "gate_kind": CUR.get("gate_kind"), "thresholds": dataclasses.asdict(TH),
    "statue_source": statue_src, "statue_reference": statue.as_dict(), "verdict": verdict.as_dict(),
    "terminations": res["terminations"], "stack": {"mujoco": mujoco.__version__, "numpy": np.__version__},
    # v2 (output only): run_panel's per-term reward totals divided by the episode count
    "reward_components_per_episode": res["components"],
}
open(OUTD / "panel.json", "w").write(json.dumps({"summary": summary, "episodes": rows,
                                                 "metrics": [mt.as_row() for mt in metrics]}, indent=1, default=float))
print(json.dumps({"label": A.label, "seconds": summary["seconds"], "n_full": verdict.n_full_horizon,
                  "n_clean": verdict.n_clean, "passed": verdict.passed, "mean_reward": verdict.mean_reward}))
