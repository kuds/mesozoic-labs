"""MISSED-GAP CHECK (verifier's own addition): does the posture-to-tail pathway exist BEFORE the s42 tail rise?
Neither test examined a checkpoint before 3,145,728 (mid-rise).  The F2 copies hold r10 stance6 checkpoints at
1,048,576 (before the rise: training tail mu starts climbing ~1.2M) and 2,097,152 (early rise) for s42 and s80.
Read only (checkpoints opened, never written).  One process: deterministic rollouts (seeds 5001-5010, 1,000
steps) then network-only sweeps on the settled states, exactly as vp_sweep2.py (G = 0..6 deg least-squares slope x
5.16; c0 at obs[6] = 0; pitch slope per deg nose-up with gravity + accelerometer + prey rotated; transplant of the
other run's typical orientation + legs + feet at the SAME checkpoint step).  No training.
"""
from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import torch

from environments.shared.config import load_stage_config
from environments.shared.model_id import ModelId
from environments.shared.policy_loading import load_sb3_checkpoint
from environments.shared.species_registry import env_class

S1 = Path("/tmp/claude-0/-home-user-mesozoic-labs/0fdf74a4-2989-56c9-a773-92101004d7de/scratchpad")
SCR = Path("/tmp/claude-0/-home-user-mesozoic-labs/184ee67c-2802-5235-8e52-6dc7daa72cd3/scratchpad")
OUT = SCR / "com_revisit/notes/verify_posture"
MID = ModelId("tyrannosaurus_rex", "anatomical")
CTOR = dict(load_stage_config(MID, "stance")["env_kwargs"])
SEEDS = list(range(5001, 5011))
PLANT = 5.16
D2R = math.pi / 180
ORIENT = [42, 43, 44, 51, 52, 53, 56, 57, 58]
LEGS = list(range(7, 21))
FEET = [54, 55]
THETAS = np.arange(0.0, 6.01, 1.0)
CK = [(r, s) for s in (1048576, 2097152, 3145728) for r in ("s42", "s80")]


def factory():
    return env_class(MID)(**CTOR)


def paths(run, steps):
    m = S1 / f"foot/stance6/fetch/F2/stance6_r10LTSB_{run}/copies/01_stance/models"
    return m / f"stage1_{steps}_steps.zip.copy1", m / f"stage1_vecnormalize_{steps}_steps.pkl.copy1"


def rot_nose_up(obs, deg, chans=(42, 51, 56)):
    a = math.radians(-deg)
    c, s = math.cos(a), math.sin(a)
    RyT = np.array([[c, 0.0, -s], [0.0, 1.0, 0.0], [s, 0.0, c]])
    out = np.array(obs, dtype=np.float64, copy=True)
    for i0 in chans:
        out[..., i0:i0 + 3] = out[..., i0:i0 + 3] @ RyT.T
    return out


def pitch_up(q):
    w, x, y, z = q / np.linalg.norm(q)
    return math.degrees(math.asin(max(-1.0, min(1.0, 2.0 * (x * z - w * y)))))


class Net:
    def __init__(self, z, p):
        self.model, self.norm, _ = load_sb3_checkpoint(str(z), str(p), factory)
        self.p = self.model.policy

    def mu14(self, obs):
        nb = self.norm.normalize_obs(np.asarray(obs, dtype=np.float32))
        with torch.no_grad():
            x = torch.as_tensor(np.asarray(nb, dtype=np.float32))
            return self.p.action_net(self.p.mlp_extractor.forward_actor(self.p.extract_features(x, self.p.pi_features_extractor))).numpy()[:, 14].astype(np.float64)


def gfit(net, X):
    ys = []
    for th in THETAS:
        Y = np.array(X, copy=True)
        Y[:, 6] = th * D2R
        ys.append(net.mu14(Y))
    ys = np.array(ys)
    return float(np.median(np.polyfit(THETAS, ys, 1)[0])) * PLANT, float(np.median(ys[0]))


def slope(net, X, fn, h=0.5):
    return float(np.median((net.mu14(fn(X, h)) - net.mu14(fn(X, -h))) / (2 * h)))


def add(idx):
    def f(X, h):
        Y = np.array(X, copy=True)
        Y[:, idx] += h * D2R
        return Y
    return f


env = factory()
states, held, nets = {}, {}, {}
log = open(OUT / "vp_prerise.jsonl", "w")
for run, steps in CK:
    z, p = paths(run, steps)
    net = Net(z, p)
    nets[(run, steps)] = net
    X, pit, cmd = [], [], []
    for seed in SEEDS:
        obs, _ = env.reset(seed=seed)
        acts, pp = [], []
        ok = True
        for k in range(1, 1001):
            a = net.model.predict(net.norm.normalize_obs(obs), deterministic=True)[0]
            acts.append(np.clip(a, -1, 1))
            obs, r, term, trunc, _ = env.step(a)
            pp.append(pitch_up(env.unwrapped.data.qpos[3:7].copy()))
            if k >= 500 and k % 100 == 0:
                X.append(obs.copy())
            if term or trunc:
                ok = k == 1000
                break
        acts = np.array(acts)
        row = dict(run=run, steps=steps, seed=seed, n=len(acts), cmd14=float(acts[500:, 14].mean()) if len(acts) > 500 else None,
                   pitch=float(np.mean(pp[500:])) if len(pp) > 500 else None)
        log.write(json.dumps(row) + "\n")
        if len(acts) == 1000:
            pit.append(row["pitch"])
            cmd.append(row["cmd14"])
    states[(run, steps)] = np.array(X, dtype=np.float64)
    held[(run, steps)] = (float(np.median(pit)), float(np.median(cmd)), len(pit))
    print(run, steps, held[(run, steps)], flush=True)
log.close()

L = []
P = L.append
P("PRE-RISE CHECKPOINTS (F2 copies, r10 stance6), deterministic, seeds 5001-5010; network sweep on own settled states")
P("  run  step       n  held_pitch  own_cmd14     c0    Gfit   fixpt  pitch/deg  grav   prey   knees   hips")
for (run, steps), X in states.items():
    net = nets[(run, steps)]
    G, c0 = gfit(net, X)
    sl = slope(net, X, lambda Y, h: rot_nose_up(Y, h))
    sg = slope(net, X, lambda Y, h: rot_nose_up(Y, h, (42,)))
    sp = slope(net, X, lambda Y, h: rot_nose_up(Y, h, (56,)))
    sk = slope(net, X, add([9, 16]))
    sh = slope(net, X, add([7, 14]))
    hp, hc, n = held[(run, steps)]
    fpv = c0 / (1 - G) if G < 1 else float("inf")
    P(f"  {run} {steps/1e6:5.3f}M {n:3d}  {hp:+8.3f}  {hc:+8.3f}  {c0:+7.3f} {G:+7.3f} {fpv:+7.3f}  {sl:+8.3f}  {sg:+6.3f} {sp:+6.3f} {sk:+7.3f} {sh:+6.3f}")
P("")
P("  TRANSPLANT at the same step: s42 states with orientation + legs + feet set to the s80 run's typical (median)")
P("  values, and the control (s42's own typical); and the reverse")
for steps in (1048576, 2097152, 3145728):
    for tgt, src in (("s42", "s80"), ("s42", "s42"), ("s80", "s42"), ("s80", "s80")):
        X = states[(tgt, steps)]
        net = nets[(tgt, steps)]
        G0, c00 = gfit(net, X)
        Y = np.array(X, copy=True)
        g = ORIENT + LEGS + FEET
        Y[:, g] = np.median(states[(src, steps)], axis=0)[g]
        G1, c01 = gfit(net, Y)
        P(f"  {steps/1e6:5.3f}M {tgt} <- {src}: c0 {c00:+.3f} -> {c01:+.3f} ({c01-c00:+.3f}); G {G0:+.3f} -> {G1:+.3f}")
txt = "\n".join(L)
(OUT / "vp_prerise.txt").write_text(txt + "\n")
print(txt)
