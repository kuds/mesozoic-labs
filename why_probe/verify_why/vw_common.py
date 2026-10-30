"""verify_why: shared analysis helpers (my own implementations; no code taken from the probes' analysis scripts).

Value network: rebuilt from a zip's policy.pth (torch.load weights_only) as SB3's ActorCriticPolicy computes it with
net_arch [512, 256] Tanh and separate pi/vf MLPs: V(z) = value_net(tanh(W2 tanh(W1 z + b1) + b2)).
"""
from __future__ import annotations

import io
import math
import os
import sys
import zipfile
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
S1 = "/tmp/claude-0/-home-user-mesozoic-labs/0fdf74a4-2989-56c9-a773-92101004d7de/scratchpad"
S2 = "/tmp/claude-0/-home-user-mesozoic-labs/184ee67c-2802-5235-8e52-6dc7daa72cd3/scratchpad"
CODE = S2 + "/r10round/bias_screen/code"
NOTES = S2 + "/com_revisit/notes"
PP = NOTES + "/phase_probe"
TP = NOTES + "/termination_probe"
F2 = S1 + "/foot/stance6/fetch/F2/stance6_r10LTSB_s42/copies/01_stance"
BS2R = S2 + "/r10round/bias_screen2/runs"
GAMMA = 0.98
LAM = 0.95
OBS_T3 = 6
SETTLE = 200
BINS = ((0, 50), (50, 200), (200, 400), (400, 1000))


def code_on_path():
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("MKL_NUM_THREADS", "1")
    os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
    code = os.path.realpath(CODE)
    if sys.path[:1] != [code]:
        sys.path.insert(0, code)


def load_pkl_stats(pkl_path):
    """mean, var, eps, clip_obs, clip_reward, sigma_r of a pickled VecNormalize (read only)."""
    import pickle

    code_on_path()
    with open(pkl_path, "rb") as fh:
        vn = pickle.load(fh)
    return {"mean": np.asarray(vn.obs_rms.mean, dtype=np.float64), "var": np.asarray(vn.obs_rms.var, dtype=np.float64),
            "eps": float(vn.epsilon), "clip_obs": float(vn.clip_obs), "clip_reward": float(vn.clip_reward),
            "sigma_r": math.sqrt(float(vn.ret_rms.var) + float(vn.epsilon)), "cls": type(vn).__name__,
            "obs_rms_cls": type(vn.obs_rms).__name__, "gamma": float(vn.gamma)}


class ValueNet:
    """The critic rebuilt from the zip (float32, as SB3 runs it)."""

    def __init__(self, zip_path):
        import torch

        torch.set_num_threads(1)
        with zipfile.ZipFile(zip_path) as f:
            sd = torch.load(io.BytesIO(f.read("policy.pth")), map_location="cpu", weights_only=True)
        self.W1 = sd["mlp_extractor.value_net.0.weight"].float()
        self.b1 = sd["mlp_extractor.value_net.0.bias"].float()
        self.W2 = sd["mlp_extractor.value_net.2.weight"].float()
        self.b2 = sd["mlp_extractor.value_net.2.bias"].float()
        self.W3 = sd["value_net.weight"].float()
        self.b3 = sd["value_net.bias"].float()
        self.P1 = sd["mlp_extractor.policy_net.0.weight"].float()
        self.pb1 = sd["mlp_extractor.policy_net.0.bias"].float()
        self.P2 = sd["mlp_extractor.policy_net.2.weight"].float()
        self.pb2 = sd["mlp_extractor.policy_net.2.bias"].float()
        self.A = sd["action_net.weight"].float()
        self.ab = sd["action_net.bias"].float()

    def _v(self, zt):
        import torch

        h = torch.tanh(zt @ self.W1.T + self.b1)
        h = torch.tanh(h @ self.W2.T + self.b2)
        return (h @ self.W3.T + self.b3)[:, 0]

    def value(self, Z, blind=False):
        import torch

        out = []
        for s in range(0, len(Z), 16384):
            zt = torch.as_tensor(np.asarray(Z[s:s + 16384], dtype=np.float32)).clone()
            if blind:
                zt[:, OBS_T3] = 0.0
            with torch.no_grad():
                out.append(self._v(zt).numpy().astype(np.float64))
        return np.concatenate(out)

    def grad(self, Z, blind=False):
        import torch

        out = []
        for s in range(0, len(Z), 16384):
            zt = torch.as_tensor(np.asarray(Z[s:s + 16384], dtype=np.float32)).clone().requires_grad_(True)
            zin = zt
            if blind:
                mask = torch.ones_like(zt)
                mask[:, OBS_T3] = 0.0
                zin = zt * mask
            g = torch.autograd.grad(self._v(zin).sum(), zt)[0]
            out.append(g.numpy().astype(np.float64))
        return np.concatenate(out)

    def mu(self, Z):
        import torch

        out = []
        for s in range(0, len(Z), 16384):
            zt = torch.as_tensor(np.asarray(Z[s:s + 16384], dtype=np.float32))
            with torch.no_grad():
                h = torch.tanh(zt @ self.P1.T + self.pb1)
                h = torch.tanh(h @ self.P2.T + self.pb2)
                out.append((h @ self.A.T + self.ab).numpy().astype(np.float64))
        return np.concatenate(out)


def normalise(raw, st):
    return np.clip((np.asarray(raw, dtype=np.float64) - st["mean"]) / np.sqrt(st["var"] + st["eps"]),
                   -st["clip_obs"], st["clip_obs"])


def fd_q3(vnet, raw, st, deg=0.25, blind=False, idx=OBS_T3):
    """Critic FD slope per degree: raw angle +-deg (radians added to the raw obs), renormalised and clipped."""
    Xp = np.array(raw, dtype=np.float64, copy=True)
    Xm = np.array(raw, dtype=np.float64, copy=True)
    Xp[:, idx] += math.radians(deg)
    Xm[:, idx] -= math.radians(deg)
    return (vnet.value(normalise(Xp, st), blind) - vnet.value(normalise(Xm, st), blind)) / (2 * deg)


def episodes(ep):
    """List of index arrays per episode (states contiguous per episode)."""
    ep = np.asarray(ep)
    cuts = np.flatnonzero(np.diff(ep)) + 1
    starts = np.r_[0, cuts]
    ends = np.r_[cuts, len(ep)]
    return [np.arange(s, e) for s, e in zip(starts, ends)]


def returns_to_go(r, ep_slices, boot):
    """G_k = sum gamma^(j-k) r_j + gamma^(T-k) boot_e (boot_e = V(s_T) at a truncation, 0 after a termination)."""
    G = np.empty(len(r), dtype=np.float64)
    for sl, b in zip(ep_slices, boot):
        acc = float(b)
        rr = r[sl]
        out = np.empty(len(sl))
        for t in range(len(sl) - 1, -1, -1):
            acc = rr[t] + GAMMA * acc
            out[t] = acc
        G[sl] = out
    return G


def phase_cols(k):
    k = np.asarray(k, dtype=np.int64)
    D0 = (k < 50).astype(float)
    D1 = ((k >= 50) & (k < 200)).astype(float)
    D2 = ((k >= 200) & (k < 400)).astype(float)
    R = GAMMA ** np.maximum(0, SETTLE - k)
    return np.column_stack([D0, D1, D2, R])


def bin_of(k):
    k = np.asarray(k)
    return np.select([k < 50, k < 200, k < 400], [0, 1, 2], 3)


def design(Z, extra=None, keep=None):
    """[1, Z (kept columns), extra]; keep = columns with SD >= 1e-6 (computed on the given Z unless passed)."""
    Z = np.asarray(Z, dtype=np.float64)
    if keep is None:
        keep = np.flatnonzero(Z.std(0) >= 1e-6)
    cols = [np.ones((len(Z), 1)), Z[:, keep]]
    if extra is not None:
        cols.append(np.asarray(extra, dtype=np.float64).reshape(len(Z), -1))
    return np.hstack(cols), keep


class ClusterOLS:
    """OLS with an episode-cluster bootstrap from per-episode Gram blocks (exact same estimator per replicate)."""

    def __init__(self, X, y, ep_slices):
        self.XtX = np.stack([X[sl].T @ X[sl] for sl in ep_slices])
        self.Xty = np.stack([X[sl].T @ y[sl] for sl in ep_slices])
        self.n = len(ep_slices)
        self.beta = np.linalg.solve(self.XtX.sum(0), self.Xty.sum(0))

    def boot(self, n_rep, rng):
        out = []
        for _ in range(n_rep):
            m = np.bincount(rng.integers(0, self.n, self.n), minlength=self.n).astype(float)
            out.append(np.linalg.solve(np.tensordot(m, self.XtX, 1), np.tensordot(m, self.Xty, 1)))
        return np.asarray(out)


def per_deg(coef_z6, var6, eps):
    return coef_z6 * (math.pi / 180.0) / math.sqrt(var6 + eps)


def auc(pos, neg):
    """P(pos > neg) + 0.5 P(tie), by ranks."""
    from scipy.stats import rankdata

    pos = np.asarray(pos, dtype=np.float64)
    neg = np.asarray(neg, dtype=np.float64)
    r = rankdata(np.r_[pos, neg])
    return (r[:len(pos)].sum() - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg))


def ci(a):
    return float(np.percentile(a, 2.5)), float(np.percentile(a, 97.5))


def lstd_blocks(Phi, r, ep_slices, terminal_reward_add, lam=LAM):
    """Per-episode LSTD(lambda) blocks A_e = sum e_t (phi_t - gamma phi_{t+1})', b_e = sum e_t r_t; the last
    transition is terminal (phi_{T} = 0) with reward r_{T-1} + terminal_reward_add_e (gamma V(s_T) at a truncation)."""
    d = Phi.shape[1]
    A = np.zeros((len(ep_slices), d, d))
    b = np.zeros((len(ep_slices), d))
    for i, (sl, add) in enumerate(zip(ep_slices, terminal_reward_add)):
        P = Phi[sl]
        rr = np.array(r[sl], dtype=np.float64, copy=True)
        rr[-1] += add
        T = len(sl)
        E = np.empty_like(P)
        e = np.zeros(d)
        gl = GAMMA * lam
        for t in range(T):
            e = gl * e + P[t]
            E[t] = e
        Pn = np.vstack([P[1:], np.zeros((1, d))])
        A[i] = E.T @ (P - GAMMA * Pn)
        b[i] = E.T @ rr
    return A, b


def lstd_solve(A, b, m=None):
    if m is None:
        return np.linalg.solve(A.sum(0), b.sum(0))
    return np.linalg.solve(np.tensordot(m, A, 1), np.tensordot(m, b, 1))
