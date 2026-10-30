"""VERIFY TEST 2: observation-only posture sensitivity sweep on MY settled states (seeds 5001-5010, steps 500..1000
by 100; vp_sim.py roll).  Network evaluation only (no stepping).  Written by the verifier.

Raw Gaussian mean of entry 14 computed by my own forward pass (features -> mlp_extractor.forward_actor ->
action_net) on normalize_obs(obs); checked against model.predict for entries inside [-1, 1].
Per state: central differences of +-0.5 (deg) unless stated; median over states per policy.
  G      = d mu14 / d obs[6] (per deg) x PLANT (deg per command unit, my measured 5.16)
  c0     = mu14 with obs[6] = 0
  pitch  = d mu14 / d(nose-up deg): gravity 42-44, accelerometer 51-53, prey direction 56-58 rotated together;
           and each channel alone
  knees  = obs 9 and 16 together (per deg); hips = obs 7 and 14; neck = obs 0; head = obs 2
Transplants: a policy's states with orientation (42-44, 51-53, 56-58), legs (7-20) and feet (54-55) replaced by
the other lineage's typical (median over that lineage's states from my rollouts); change in c0 and G; linear fixed
point c0 / (1 - G); control = own lineage's typical values.
"""
from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import torch

from environments.shared.config import load_stage_config
from environments.shared.model_id import ModelId
from environments.shared.policy_loading import load_sb3_checkpoint
from environments.shared.species_registry import env_class

SCR = Path("/tmp/claude-0/-home-user-mesozoic-labs/184ee67c-2802-5235-8e52-6dc7daa72cd3/scratchpad")
RUNS = SCR / "r10round/ecs/runs"
OUT = SCR / "com_revisit/notes/verify_posture"
MID = ModelId("tyrannosaurus_rex", "anatomical")
CTOR = dict(load_stage_config(MID, "stance")["env_kwargs"])
PLANT = 5.16
POL = {"start_s42": ("s42_A_1101", 3145728), "s42_A_1111": ("s42_A_1111", 4718592), "s42_B_1211": ("s42_B_1211", 4718592),
       "s42_A_1151": ("s42_A_1151", 4718592), "s42_B_1241": ("s42_B_1241", 4718592), "s80_A_1311": ("s80_A_1311", 5767168),
       "s80_B_1421": ("s80_B_1421", 5767168), "start_s80": ("s80_A_1301", 4194304)}

for _n in [f"s42_A_{n}" for n in range(1101, 1152, 10)] + [f"s42_B_{n}" for n in range(1201, 1252, 10)]:
    POL.setdefault(_n, (_n, 4718592))
for _n in [f"s80_A_{n}" for n in range(1301, 1352, 10)] + [f"s80_B_{n}" for n in range(1401, 1452, 10)]:
    POL.setdefault(_n, (_n, 5767168))
S42_ENDS = ["s42_A_1111", "s42_B_1211", "s42_A_1151", "s42_B_1241"]
S80_ENDS = ["s80_A_1311", "s80_B_1421"]
ORIENT = [42, 43, 44, 51, 52, 53, 56, 57, 58]
LEGS = list(range(7, 21))
FEET = [54, 55]
D2R = math.pi / 180


def factory():
    return env_class(MID)(**CTOR)


def load(who):
    run, steps = POL[who]
    m = RUNS / run / "01_stance" / "models"
    return load_sb3_checkpoint(str(m / f"stage1_{steps}_steps.zip"), str(m / f"stage1_vecnormalize_{steps}_steps.pkl"), factory)[:2]


def rot_nose_up(obs, deg, chans=(42, 51, 56)):
    a = math.radians(-deg)
    c, s = math.cos(a), math.sin(a)
    RyT = np.array([[c, 0.0, -s], [0.0, 1.0, 0.0], [s, 0.0, c]])
    out = np.array(obs, dtype=np.float64, copy=True)
    for i0 in chans:
        out[..., i0:i0 + 3] = out[..., i0:i0 + 3] @ RyT.T
    return out


class Net:
    def __init__(self, who):
        self.model, self.norm = load(who)
        self.p = self.model.policy

    def mu(self, obs):
        nb = self.norm.normalize_obs(np.asarray(obs, dtype=np.float32))
        with torch.no_grad():
            x = torch.as_tensor(np.asarray(nb, dtype=np.float32))
            f = self.p.extract_features(x, self.p.pi_features_extractor)
            lat = self.p.mlp_extractor.forward_actor(f)
            return self.p.action_net(lat).numpy().astype(np.float64)

    def mu14(self, obs):
        return self.mu(obs)[:, 14]


def per_state_slope(net, X, fn, h=0.5):
    return (net.mu14(fn(X, +h)) - net.mu14(fn(X, -h))) / (2 * h)


def add(idx, scale=D2R):
    def f(X, h):
        Y = np.array(X, dtype=np.float64, copy=True)
        Y[:, idx] += h * scale
        return Y
    return f


def G_c0(net, X):
    g = np.median(per_state_slope(net, X, add([6]))) * PLANT
    Y = np.array(X, dtype=np.float64, copy=True)
    Y[:, 6] = 0.0
    c0 = float(np.median(net.mu14(Y)))
    return float(g), c0


def fp(c0, G):
    return c0 / (1 - G) if G < 1 else float("inf")


states = {w: np.load(OUT / f"settled_{w}.npz")["obs"].astype(np.float64) for w in POL}
typ = {"s42": np.median(np.vstack([states[w] for w in S42_ENDS]), axis=0),
       "s80": np.median(np.vstack([states[w] for w in S80_ENDS]), axis=0),
       "start_s42": np.median(states["start_s42"], axis=0), "start_s80": np.median(states["start_s80"], axis=0)}
L = []
P = L.append
P("VERIFY TEST 2 SWEEP (vp_sweep.py): my settled states (10 seeds x 6 steps = 60 per policy), network only")
P(f"  typical held pelvis pitch from obs (asin(-g_x) of the median g): " + ", ".join(
    f"{k} {math.degrees(math.asin(-v[42])):+.3f}" for k, v in typ.items()))
P(f"  typical knees (deg): " + ", ".join(f"{k} {math.degrees(v[9]):+.2f}/{math.degrees(v[16]):+.2f}" for k, v in typ.items())
  + "; hips: " + ", ".join(f"{k} {math.degrees(v[7]):+.2f}/{math.degrees(v[14]):+.2f}" for k, v in typ.items()))
P("")
hdr = ("who", "n", "chk", "c0", "G", "fixpt", "pitch", "grav", "prey", "acc", "knees", "hips", "neck", "head", "1deg=tail")
P("  " + " ".join(f"{h:>9s}" for h in hdr))
rows = {}
for who in POL:
    X = states[who]
    net = Net(who)
    # check my forward pass against predict (clipped deterministic action)
    pred = np.array([net.model.predict(net.norm.normalize_obs(x.astype(np.float32)), deterministic=True)[0] for x in X[:10]])
    mine = np.clip(net.mu(X[:10]), -1, 1)
    chk = float(np.abs(pred - mine).max())
    G, c0 = G_c0(net, X)
    sl = {
        "pitch": np.median(per_state_slope(net, X, lambda Y, h: rot_nose_up(Y, h))),
        "grav": np.median(per_state_slope(net, X, lambda Y, h: rot_nose_up(Y, h, (42,)))),
        "prey": np.median(per_state_slope(net, X, lambda Y, h: rot_nose_up(Y, h, (56,)))),
        "acc": np.median(per_state_slope(net, X, lambda Y, h: rot_nose_up(Y, h, (51,)))),
        "knees": np.median(per_state_slope(net, X, add([9, 16]))),
        "hips": np.median(per_state_slope(net, X, add([7, 14]))),
        "neck": np.median(per_state_slope(net, X, add([0]))),
        "head": np.median(per_state_slope(net, X, add([2]))),
    }
    dtail = G / PLANT
    rows[who] = dict(c0=c0, G=G, **sl)
    P("  " + " ".join([f"{who:>9s}"[-10:], f"{len(X):9d}", f"{chk:9.1e}", f"{c0:+9.3f}", f"{G:+9.3f}", f"{fp(c0, G):+9.2f}"]
                      + [f"{sl[k]:+9.3f}" for k in ("pitch", "grav", "prey", "acc", "knees", "hips", "neck", "head")]
                      + [f"{-sl['pitch']/dtail:+9.2f}" if dtail != 0 else "      nan"]))
P("  (slopes per deg; pitch/grav/prey/acc per deg NOSE-UP; '1deg=tail' = deg of observed tail angle worth 1 deg nose-DOWN)")
P("")
P("TRANSPLANTS (orientation + legs + feet replaced by the named lineage's typical values)")
for who, src_list in (("s42_A_1111", ("s80", "s42")), ("s42_B_1211", ("s80", "s42")), ("s42_A_1151", ("s80", "s42")),
                      ("s42_B_1241", ("s80", "s42")), ("start_s42", ("start_s80", "s80", "start_s42")),
                      ("s80_A_1311", ("s42", "s80")), ("s80_B_1421", ("s42", "s80"))):
    X = states[who]
    net = Net(who)
    G0, c00 = G_c0(net, X)
    for src in src_list:
        for grp_name, grp in (("orient", ORIENT), ("legs", LEGS), ("orient+legs+feet", ORIENT + LEGS + FEET)):
            Y = np.array(X, copy=True)
            Y[:, grp] = typ[src][grp]
            G1, c01 = G_c0(net, Y)
            P(f"  {who:11s} <- {src:9s} {grp_name:17s}: c0 {c00:+.3f} -> {c01:+.3f} ({c01-c00:+.3f}); G {G0:+.3f} -> {G1:+.3f} "
              f"({G1-G0:+.3f}); fixed point {fp(c00, G0):+.2f} -> {fp(c01, G1):+.2f}")
txt = "\n".join(L)
(OUT / "vp_sweep.txt").write_text(txt + "\n")
print(txt)
