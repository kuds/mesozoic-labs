"""Posture of learned T. rex stances on main's physics-r9 plant (report-only, no repo write).

    python -P posture_roll.py --out OUT_DIR [--seeds 3042 ... ] [--targets statue s60 s52]

Rolls, per target, deterministic stance episodes on the UNMODIFIED stance stage config of
tyrannosaurus_rex/anatomical ("options off": no lateral lean term, keyframe leg target, stock env;
heading 0 = the panel's spawn, no spawn yaw), through main's own floor-truth recorder
(gait.recorder.SubstepContactRecorder) and pad telemetry, plus a boundary sample of the kinematics and
contact forces after every control step (read-only on env.data).

Targets
  statue   zero action (stance_report._zero_action_predict)
  s60/s52  r9 arm L (sym_lean_settled) checkpoints at 6,291,456 steps, policy mean (deterministic), frozen
           VecNormalize statistics (stance_report._load_policy, plant identity checked)

Per step it stores, per foot (r, l), world positions of the hip / knee / ankle / digit-III MTP joint
anchors and the digit-III tip, the pad centre and forward axis, the whole-animal COM (subtree of the
pelvis), the pelvis pitch, and the foot's total floor reaction (3D, mj_contactForce, every floor contact
of the limb) with its vertical-force-weighted contact point (the foot's CoP on the floor).

Writes OUT_DIR/steps_<target>_<seed>.npz (per-step arrays + the recorder's per-step CoP / loads) and
OUT_DIR/episodes.jsonl (per-episode window means, two windows: [40, T) and [200, T)).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import sys
import time

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

import numpy as np  # noqa: E402

import environments  # noqa: E402

assert "fix/main_ro" in environments.__file__, environments.__file__

import mujoco  # noqa: E402

from environments.shared.config import load_stage_config  # noqa: E402
from environments.shared.curriculum.stance_gate_v2 import StanceV2Thresholds, classify_stance_episode  # noqa: E402
from environments.shared.gait.morphology import Morphology  # noqa: E402
from environments.shared.gait.pad_telemetry import PadLayout, episode_pad_telemetry  # noqa: E402
from environments.shared.gait.recorder import SubstepContactRecorder  # noqa: E402
from environments.shared.gait.stance_metrics import episode_stance_metrics  # noqa: E402
from environments.shared.plant_contract import current_plant_identity  # noqa: E402
from environments.shared.reporting import stance_report as SR  # noqa: E402
from environments.shared.species_names import require_model_id  # noqa: E402

S = "scratchpad"
RUNS = f"{S}/foot/symmetry/screen/runs"
CKPT = {
    "s60": (f"{RUNS}/sym_lean_settled_s60/resume6m/models/stage1_6291456_steps.zip",
            f"{RUNS}/sym_lean_settled_s60/resume6m/models/stage1_vecnormalize_6291456_steps.pkl"),
    "s52": (f"{RUNS}/sym_lean_settled_s52/resume6m/models/stage1_6291456_steps.zip",
            f"{RUNS}/sym_lean_settled_s52/resume6m/models/stage1_vecnormalize_6291456_steps.pkl"),
}
WINDOWS = {"w40": 40, "w200": 200}
DIGIT_TIP_M = 0.19 + 0.028  # digit III capsule: fromto 0..0.19 m along its x axis, radius 0.028 (MJCF)
DIGIT_AXIS_M = 0.19
PAD_HALF_M = 0.07


def sha256(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


class Probe:
    """Boundary-sample kinematics and floor reactions, read-only on env.data."""

    def __init__(self, env):
        m = env.unwrapped.model
        self.m = m
        jid = lambda n: mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, n)  # noqa: E731
        bid = lambda n: mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, n)  # noqa: E731
        gid = lambda n: mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, n)  # noqa: E731
        self.floor = gid("floor")
        self.pelvis = bid("pelvis")
        self.j = {f"{s}_{n}": jid(f"{s}_{n}") for s in "rl" for n in ("hip_pitch", "knee", "ankle", "toe_d3_joint")}
        self.qadr = {k: int(m.jnt_qposadr[v]) for k, v in self.j.items()}
        self.toe3 = [bid(f"{s}_toe_d3") for s in "rl"]
        self.pad = [gid(f"{s}_plantar_geom") for s in "rl"]
        self.toes = [[gid(f"{s}_toe_d{k}_geom") for k in (2, 3, 4)] for s in "rl"]
        self.thigh = [bid(f"{s}_thigh") for s in "rl"]
        self.foot_geoms = []
        for tb in self.thigh:
            bodies = {b for b in range(m.nbody) if self._under(b, tb)}
            self.foot_geoms.append({g for g in range(m.ngeom) if m.geom_bodyid[g] in bodies})
        self.f6 = np.zeros(6)
        self.rows: dict[str, list] = {}

    def _under(self, body: int, anc: int) -> bool:
        while body != 0:
            if body == anc:
                return True
            body = int(self.m.body_parentid[body])
        return False

    def _put(self, k, v):
        self.rows.setdefault(k, []).append(np.asarray(v, dtype=np.float64).copy())

    def sample(self, env):
        d = env.unwrapped.data
        m = self.m
        R = d.xmat[self.pelvis].reshape(3, 3)
        fwd = R[:, 0].copy()
        h = np.array([fwd[0], fwd[1], 0.0])
        h /= np.linalg.norm(h)
        self._put("heading", h)
        self._put("pelvis_pitch_deg", math.degrees(math.asin(max(-1.0, min(1.0, -R[2, 0])))))  # + nose down
        self._put("pelvis_pos", d.xpos[self.pelvis])
        self._put("com", d.subtree_com[self.pelvis])
        for f, s in enumerate("rl"):
            self._put(f"{s}_hip", d.xanchor[self.j[f"{s}_hip_pitch"]])
            self._put(f"{s}_knee", d.xanchor[self.j[f"{s}_knee"]])
            self._put(f"{s}_ankle", d.xanchor[self.j[f"{s}_ankle"]])
            self._put(f"{s}_mtp", d.xanchor[self.j[f"{s}_toe_d3_joint"]])
            Rt = d.xmat[self.toe3[f]].reshape(3, 3)
            self._put(f"{s}_d3_axis_end", d.xpos[self.toe3[f]] + Rt[:, 0] * DIGIT_AXIS_M)
            self._put(f"{s}_d3_tip", d.xpos[self.toe3[f]] + Rt[:, 0] * DIGIT_TIP_M)
            Rp = d.geom_xmat[self.pad[f]].reshape(3, 3)
            self._put(f"{s}_pad_c", d.geom_xpos[self.pad[f]])
            self._put(f"{s}_pad_fwd", Rp[:, 0])
            for n in ("hip_pitch", "knee", "ankle", "toe_d3_joint"):
                self._put(f"{s}_q_{n}_deg", math.degrees(d.qpos[self.qadr[f"{s}_{n}"]]))
            F = np.zeros(3)
            wsum = 0.0
            wpos = np.zeros(3)
            fz_pad = fz_toes = fz_other = 0.0
            for k in range(d.ncon):
                c = d.contact[k]
                if c.geom1 == self.floor:
                    other = c.geom2
                elif c.geom2 == self.floor:
                    other = c.geom1
                else:
                    continue
                if other not in self.foot_geoms[f]:
                    continue
                mujoco.mj_contactForce(m, d, k, self.f6)
                fw = c.frame.reshape(3, 3).T @ self.f6[:3]
                if c.geom2 == self.floor:  # contact normal points geom1 -> geom2; force on the animal
                    fw = -fw
                F += fw
                fz = max(fw[2], 0.0)
                wsum += fz
                wpos += fz * c.pos
                if other == self.pad[f]:
                    fz_pad += fw[2]
                elif other in self.toes[f]:
                    fz_toes += fw[2]
                else:
                    fz_other += fw[2]
            self._put(f"{s}_F", F)
            self._put(f"{s}_cop_xyz", wpos / wsum if wsum > 1e-9 else np.full(3, np.nan))
            self._put(f"{s}_fz_pad", fz_pad)
            self._put(f"{s}_fz_toes", fz_toes)
            self._put(f"{s}_fz_other", fz_other)

    def arrays(self) -> dict[str, np.ndarray]:
        return {k: np.stack(v) for k, v in self.rows.items()}


def predictor(target: str, ctx: dict):
    if target == "statue":
        return SR._zero_action_predict(ctx["env_class"], ctx["env_kwargs"]), "zero action (statue)"
    zp, vp = CKPT[target]
    fn, note = SR._load_policy(zp, vp, lambda: ctx["env_class"](**ctx["env_kwargs"]),
                               plant_identity=ctx["plant_identity"], allow_legacy_plant=False)
    return fn, note


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--seeds", type=int, nargs="+", default=list(range(3042, 3052)))
    ap.add_argument("--targets", nargs="+", default=["statue", "s60", "s52"])
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    mid = require_model_id("tyrannosaurus_rex", "anatomical")
    cfg = load_stage_config(mid, "stance")
    env_kwargs = dict(cfg["env_kwargs"])
    assert not any(k.startswith("experiment_") for k in env_kwargs)
    assert float(env_kwargs.get("lateral_lean_weight", 0.0)) == 0.0, "options off: no lateral lean term"
    assert env_kwargs.get("leg_home_pose_reference", "keyframe") == "keyframe", "options off: keyframe leg target"
    cur = cfg["curriculum_kwargs"]
    thr = StanceV2Thresholds.from_curriculum(dict(cur))
    gate_settle = int(thr.settle_steps)
    horizon = int(env_kwargs.get("max_episode_steps", 1000))
    env_class = SR.model_env_class(mid)
    ctx = {"env_class": env_class, "env_kwargs": env_kwargs, "plant_identity": current_plant_identity(mid)}

    meta = {
        "repo": os.path.dirname(os.path.dirname(environments.__file__)),
        "model_id": str(mid), "stage": "stance", "env_kwargs_overrides": {}, "options": "off",
        "heading": "0 (panel spawn; no spawn yaw)", "deterministic": True, "horizon": horizon,
        "gate_settle_steps": gate_settle, "cop_bar": float(thr.max_sole_cop_fore_aft), "seeds": args.seeds,
        "windows": WINDOWS, "checkpoints": {k: {"zip": z, "zip_sha256": sha256(z), "vecnorm": v,
                                                "vecnorm_sha256": sha256(v)} for k, (z, v) in CKPT.items()},
        "mjcf": env_class.MODEL_PATH.as_posix() if hasattr(env_class, "MODEL_PATH") else None,
    }
    meta["mjcf_sha256"] = sha256(meta["mjcf"]) if meta["mjcf"] else None
    json.dump(meta, open(os.path.join(args.out, "meta.json"), "w"), indent=1)

    env = env_class(**env_kwargs)
    morph = Morphology.from_env(env, mid)
    layout = PadLayout.from_morphology(morph)
    out_jsonl = open(os.path.join(args.out, "episodes.jsonl"), "w")
    for target in args.targets:
        predict, note = predictor(target, ctx)
        t0 = time.monotonic()
        for seed in args.seeds:
            probe = Probe(env)
            rec = SubstepContactRecorder(env, morph).attach()
            reason = ""
            term = trunc = False
            try:
                obs, _ = env.reset(seed=seed)
                steps = 0
                while True:
                    a = np.asarray(predict(obs), dtype=np.float64).ravel()
                    obs, r, term, trunc, info = env.step(a)
                    rec.end_step(a, float(r))
                    probe.sample(env)
                    steps += 1
                    if term or trunc or steps >= horizon:
                        reason = str(info.get("termination_reason", "terminated" if term else "truncated"))
                        break
                trace = rec.end_episode(terminated=bool(term), truncated=bool(trunc), termination_reason=reason)
            finally:
                rec.detach()
            met = episode_stance_metrics(trace, settle_steps=gate_settle)
            reasons = classify_stance_episode(met, thr, horizon=horizon, apply_statue_criteria=False)
            pad = episode_pad_telemetry(trace, layout, settle_steps=gate_settle)
            arr = probe.arrays()
            names = list(trace.support_geom_names)
            for f, s in enumerate(trace.labels):
                arr[f"rec_{s}_sole_cop_fore"] = trace.sole_cop[:, f, 0]
                arr[f"rec_{s}_sole_contacts"] = trace.sole_contacts_mean[:, f]
                arr[f"rec_{s}_foot_cop_m"] = trace.foot_cop[:, f]
                arr[f"rec_{s}_leg_floor_N"] = trace.leg_floor_mean[:, f]
            for g, n in enumerate(names):
                arr[f"rec_support_{n}_N"] = trace.support_floor_mean[:, g]
            arr["rec_foot_support_span"] = trace.foot_support_span
            arr["rec_labels"] = np.array(trace.labels)
            np.savez_compressed(os.path.join(args.out, f"steps_{target}_{seed}.npz"), **arr)
            row = {"target": target, "seed": seed, "length": int(trace.length), "reason": reason,
                   "reached_horizon": int(trace.length) >= horizon,
                   "clean_nonstatue_criteria": not reasons, "reasons": [x.split(":")[0] for x in reasons],
                   "max_sole_cop_fore_aft": float(met.max_sole_cop_fore_aft),
                   "pad_telemetry_gate_window": {k: float(v) for k, v in pad.items()},
                   "policy_note": note}
            out_jsonl.write(json.dumps(row) + "\n")
            out_jsonl.flush()
            print(json.dumps({"target": target, "seed": seed, "length": row["length"],
                              "gate_cop": row["max_sole_cop_fore_aft"], "s": round(time.monotonic() - t0, 1)}),
                  flush=True)
    out_jsonl.close()
    env.close()


if __name__ == "__main__":
    sys.dont_write_bytecode = True
    main()
