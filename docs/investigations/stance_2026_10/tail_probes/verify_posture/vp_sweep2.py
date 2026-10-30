"""VERIFY TEST 2 (v2: all 26 policies, both G definitions): observation-only posture sensitivity sweep on MY settled states (seeds 5001-5010, steps 500..1000
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



import json
from collections import defaultdict

S42 = [f"s42_A_{n}" for n in range(1101, 1152, 10)] + [f"s42_B_{n}" for n in range(1201, 1252, 10)]
S80 = [f"s80_A_{n}" for n in range(1301, 1352, 10)] + [f"s80_B_{n}" for n in range(1401, 1452, 10)]
THETAS = np.arange(0.0, 6.01, 1.0)


def G_fit(net, X):
    """Least-squares slope of mu14 over obs[6] = 0..6 deg per state (median), x PLANT; and c0 at 0."""
    ys = []
    for th in THETAS:
        Y = np.array(X, dtype=np.float64, copy=True)
        Y[:, 6] = th * D2R
        ys.append(net.mu14(Y))
    ys = np.array(ys)  # (7, n)
    sl = np.polyfit(THETAS, ys, 1)[0]
    return float(np.median(sl)) * PLANT, float(np.median(ys[0]))


def spearman(a, b):
    ra = np.argsort(np.argsort(a)).astype(float)
    rb = np.argsort(np.argsort(b)).astype(float)
    return float(np.corrcoef(ra, rb)[0, 1])


held = defaultdict(list)
for r in map(json.loads, open(OUT / "vp_roll.jsonl")):
    if r["steps"] == 1000:
        held[r["who"]].append(r)
states = {w: np.load(OUT / f"settled_{w}.npz")["obs"].astype(np.float64) for w in POL}
typ = {"s42": np.median(np.vstack([states[w] for w in S42]), axis=0),
       "s80": np.median(np.vstack([states[w] for w in S80]), axis=0),
       "start_s42": np.median(states["start_s42"], axis=0), "start_s80": np.median(states["start_s80"], axis=0)}
sd_all = np.std(np.vstack([states[w] for w in S42 + S80]), axis=0)
L = []
P = L.append
P("VERIFY TEST 2 SWEEP v2 (vp_sweep2.py): my settled states (seeds 5001-5010, steps 500..1000 by 100), network only")
P("  typical held pelvis pitch, asin(-g_x) of the median state: " + ", ".join(
    f"{k} {math.degrees(math.asin(-v[42])):+.3f}" for k, v in typ.items()))
P("  held pitch per policy (median over seeds of the window mean, deg): s42 ends median "
  f"{np.median([np.median([r['pitch'] for r in held[w]]) for w in S42]):+.3f}; s80 ends "
  f"{np.median([np.median([r['pitch'] for r in held[w]]) for w in S80]):+.3f}")
z = np.abs(typ["s42"] - typ["s80"]) / sd_all
grp = ORIENT + LEGS + FEET
P(f"  lineage gap in SD of the pooled states, orientation+legs+feet channels: max {z[grp].max():.2f}, median {np.median(z[grp]):.2f}")
P("")
cols = ("c0", "Gloc", "Gfit", "fixpt", "pitch", "grav", "prey", "acc", "knees", "hips", "neck", "head", "pitchHeld")
P("  " + f"{'who':>11s} " + " ".join(f"{c:>8s}" for c in cols))
rows = {}
nets = {}
for who in POL:
    X = states[who]
    net = Net(who)
    nets[who] = net
    Gl, _ = G_c0(net, X)
    Gf, c0 = G_fit(net, X)
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
    ph = float(np.median([r["pitch"] for r in held[who]]))
    rows[who] = dict(c0=c0, Gloc=Gl, Gfit=Gf, fixpt=fp(c0, Gf), pitchHeld=ph, **{k: float(v) for k, v in sl.items()})
    P("  " + f"{who:>11s} " + " ".join(f"{rows[who][c]:+8.3f}" for c in cols))


def summ(ws, key):
    v = np.array([rows[w][key] for w in ws])
    return f"{np.median(v):+.3f} [{v.min():+.3f},{v.max():+.3f}] +{(v>0).sum()}/-{(v<0).sum()}"


P("")
for lab, ws in (("s42 ends (12)", S42), ("s80 ends (12)", S80)):
    P(f"  {lab}: " + "; ".join(f"{k} {summ(ws, k)}" for k in ("c0", "Gloc", "Gfit", "fixpt", "pitch", "grav", "prey", "acc", "knees", "hips")))
    P(f"     predicted rail (c0/(1-Gfit) >= 1): {sum(rows[w]['fixpt'] >= 1 for w in ws)}/12; "
      f"1 deg nose-down = {np.median([-rows[w]['pitch'] / (rows[w]['Gfit'] / PLANT) for w in ws]):+.2f} deg of observed tail (median)")
P("")
P("  cross-policy Spearman (s42 ends, n 12; correlational): held pitch vs c0 "
  f"{spearman([rows[w]['pitchHeld'] for w in S42], [rows[w]['c0'] for w in S42]):+.2f}; vs pitch slope "
  f"{spearman([rows[w]['pitchHeld'] for w in S42], [rows[w]['pitch'] for w in S42]):+.2f}; vs Gfit "
  f"{spearman([rows[w]['pitchHeld'] for w in S42], [rows[w]['Gfit'] for w in S42]):+.2f}; c0 vs Gfit "
  f"{spearman([rows[w]['c0'] for w in S42], [rows[w]['Gfit'] for w in S42]):+.2f}")
P("  cross-policy Spearman (s80 ends): held pitch vs c0 "
  f"{spearman([rows[w]['pitchHeld'] for w in S80], [rows[w]['c0'] for w in S80]):+.2f}; vs pitch slope "
  f"{spearman([rows[w]['pitchHeld'] for w in S80], [rows[w]['pitch'] for w in S80]):+.2f}; vs Gfit "
  f"{spearman([rows[w]['pitchHeld'] for w in S80], [rows[w]['Gfit'] for w in S80]):+.2f}")
P("")
P("TRANSPLANTS (each policy's own states with the named channels set to the donor lineage's typical values;")
P("  G = Gfit; control = own lineage's typical values)")
tr = defaultdict(list)
for ws, other, own in ((S42, "s80", "s42"), (S80, "s42", "s80")):
    for who in ws:
        X = states[who]
        net = nets[who]
        G0, c00 = rows[who]["Gfit"], rows[who]["c0"]
        for donor in (other, own):
            for gname, g in (("orient", ORIENT), ("legs", LEGS), ("orient+legs+feet", ORIENT + LEGS + FEET)):
                Y = np.array(X, copy=True)
                Y[:, g] = typ[donor][g]
                G1, c01 = G_fit(net, Y)
                tr[(own, donor == own, gname)].append((c01 - c00, G1 - G0, fp(c00, G0) >= 1, fp(c01, G1) >= 1))
for (own, ctrl, gname), v in tr.items():
    v = np.array(v, dtype=float)
    P(f"  {own} ends <- {'OWN (control)' if ctrl else 'OTHER lineage'} {gname:17s}: dc0 {np.median(v[:,0]):+.3f} "
      f"[{v[:,0].min():+.3f},{v[:,0].max():+.3f}] +{(v[:,0]>0).sum()}/-{(v[:,0]<0).sum()}; dG {np.median(v[:,1]):+.3f} "
      f"[{v[:,1].min():+.3f},{v[:,1].max():+.3f}]; predicted rail {int(v[:,2].sum())}/12 -> {int(v[:,3].sum())}/12")
for who, donors in (("start_s42", ("start_s80", "s80", "start_s42")), ("start_s80", ("start_s42", "start_s80"))):
    X = states[who]
    net = nets[who]
    G0, c00 = rows[who]["Gfit"], rows[who]["c0"]
    for donor in donors:
        Y = np.array(X, copy=True)
        Y[:, ORIENT + LEGS + FEET] = typ[donor][ORIENT + LEGS + FEET]
        G1, c01 = G_fit(net, Y)
        P(f"  {who} <- {donor} orient+legs+feet: c0 {c00:+.3f} -> {c01:+.3f} ({c01-c00:+.3f}); Gfit {G0:+.3f} -> {G1:+.3f}; "
          f"fixed point {fp(c00, G0):+.2f} -> {fp(c01, G1):+.2f}")
txt = "\n".join(L)
(OUT / "vp_sweep2.txt").write_text(txt + "\n")
json.dump(rows, open(OUT / "vp_sweep2.json", "w"), indent=1)
print(txt)
