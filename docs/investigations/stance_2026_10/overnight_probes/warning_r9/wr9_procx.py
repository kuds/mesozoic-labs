"""warning_r9 procedure X (PREREG.txt section 3): translate the r10 threshold 0.150 to the r9 probe bank.

Uses r10 networks (F2 copies, read only) and the r9 bank's raw STATES only; reads no r9 series.  Network evaluation
only (no rollouts, no training).

    PYTHONPATH=<r10round/ecs/code> OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 nice -n 15 venv/bin/python -B wr9_procx.py
"""
from __future__ import annotations

import glob
import hashlib
import re
import sys
from pathlib import Path

import numpy as np
import torch

torch.set_num_threads(1)
HERE = Path(__file__).resolve().parent
F2 = Path("/tmp/claude-0/-home-user-mesozoic-labs/0fdf74a4-2989-56c9-a773-92101004d7de/scratchpad/foot/stance6/fetch/F2")
TR = Path("/tmp/claude-0/-home-user-mesozoic-labs/184ee67c-2802-5235-8e52-6dc7daa72cd3/scratchpad/tailround/fetch")
A_T3 = 14


def r10_dir(run):
    return F2 / f"stance6_r10LTSB_{run}" / "copies" / "01_stance"


def checkpoints(run):
    m = r10_dir(run) / "models"
    out = []
    for z in glob.glob(str(m / "stage1_*_steps.zip.copy1")):
        n = int(re.search(r"stage1_(\d+)_steps", z).group(1))
        out.append((n, Path(z), m / f"stage1_vecnormalize_{n}_steps.pkl.copy1"))
    out.append((12_001_280, m / "stage1_final.zip.copy1", m / "stage1_final_vecnorm.pkl.copy1"))
    out.sort()
    for n, z, p in out:
        assert z.exists() and p.exists(), (z, p)
    return out


def bank(path):
    with np.load(path, allow_pickle=False) as d:
        X = np.asarray(d["bank_raw_obs"], dtype=np.float64)
        lab = np.asarray(d["bank_labels"]).astype(str)
        meta = {k: np.asarray(d[k]).tolist() for k in ("bank_seeds", "bank_steps") if k in d.files}
    return X, lab, hashlib.sha1(X.tobytes()).hexdigest()[:10], meta


def main() -> int:
    from environments.shared.config import load_stage_config
    from environments.shared.model_id import ModelId
    from environments.shared.policy_loading import load_sb3_checkpoint
    from environments.shared.species_registry import env_class

    mid = ModelId("tyrannosaurus_rex", "anatomical")
    ctor = dict(load_stage_config(mid, "stance")["env_kwargs"])
    factory = lambda: env_class(mid)(**ctor)  # noqa: E731

    X10, L10, s10, m10 = bank(r10_dir("s42") / "policy_probe.npz.copy1")
    banks9 = {r: bank(TR / f"stance5_LTSB_{r}" / "A" / "policy_probe.npz") for r in ("s42", "s52", "s60", "s70", "s80")}
    shas9 = {r: b[2] for r, b in banks9.items()}
    print(f"r10 bank sha1 {s10}; seeds {m10.get('bank_seeds')}; steps {m10.get('bank_steps')}")
    print(f"r9 bank sha1 per run {shas9}")
    X9, L9, s9, m9 = banks9["s42"]
    print(f"r9 bank seeds {m9.get('bank_seeds')}; steps {m9.get('bank_steps')}")
    assert len(set(shas9.values())) == 1, "r9 banks differ between runs"
    ss10, ss9 = L10 == "statue_settled", L9 == "statue_settled"
    print(f"statue_settled rows: r10 {np.flatnonzero(ss10).tolist()}, r9 {np.flatnonzero(ss9).tolist()}")
    d = X9[ss9] - X10[ss10]
    print(f"raw-state difference r9 - r10 on the statue_settled rows: max |diff| {np.abs(d).max():.4g}; "
          f"channels with |diff| > 1e-3: {np.flatnonzero(np.abs(d).max(axis=0) > 1e-3).tolist()}")
    print(f"  tail_3 angle obs[6] (deg): r10 {np.degrees(X10[ss10, 6]).round(3).tolist()}, r9 {np.degrees(X9[ss9, 6]).round(3).tolist()}")

    rows = []
    for run in ("s42", "s60", "s80"):
        with np.load(r10_dir(run) / "policy_probe.npz.copy1", allow_pickle=False) as pz:
            pt = np.asarray(pz["update_timesteps"]).astype(np.int64)
            pa = np.asarray(pz["bank_action_now"])[:, :, A_T3].astype(float)
            plab = np.asarray(pz["bank_labels"]).astype(str)
        p_ss = pa[:, plab == "statue_settled"].mean(axis=1)
        for n, z, p in checkpoints(run):
            model, norm, _ = load_sb3_checkpoint(str(z), str(p), factory)

            def mu(X):
                nb = norm.normalize_obs(np.asarray(X, dtype=np.float32))
                with torch.no_grad():
                    dist = model.policy.get_distribution(torch.as_tensor(np.asarray(nb, dtype=np.float32)))
                    return dist.distribution.mean.numpy().astype(np.float64)[:, A_T3]

            y10r = mu(X10[ss10])
            y9r = mu(X9[ss9])
            y10, y9 = float(y10r.mean()), float(y9r.mean())
            i = np.flatnonzero(pt == n)
            rec = float(p_ss[i[0]]) if len(i) else float("nan")
            rows.append((run, n, y10, y9, rec, y9r.tolist(), y10r.tolist()))
            print(f"  {run} {n:>9d}: y10 {y10:+.5f} (probe {rec:+.5f}, |diff| {abs(y10 - rec):.2e})  y9 {y9:+.5f}  "
                  f"y9 - y10 {y9 - y10:+.5f}   rows r10 {np.round(y10r, 4).tolist()} r9 {np.round(y9r, 4).tolist()}",
                  flush=True)
    y10 = np.array([r[2] for r in rows])
    y9 = np.array([r[3] for r in rows])
    rec = np.array([r[4] for r in rows])
    fin = np.array([r[1] == 12_001_280 for r in rows])
    print(f"\nVALIDATION y10 vs recorded probe (non-final, n {int((~fin).sum())}): max |diff| "
          f"{np.nanmax(np.abs(y10 - rec)[~fin]):.2e}; finals {np.abs(y10 - rec)[fin].round(4).tolist()}")
    b, a = np.polyfit(y10, y9, 1)
    res = y9 - (a + b * y10)
    r = float(np.corrcoef(y10, y9)[0, 1])
    sd = float(np.std(res, ddof=2))
    thr9 = a + b * 0.150
    print(f"FIT y9 = a + b*y10 over n {len(rows)} checkpoints: a {a:+.5f}, b {b:+.5f}, Pearson r {r:+.4f}, residual SD {sd:.5f}")
    print(f"thr9 = a + b*0.150 = {thr9:+.5f};  a + b*0.186 = {a + b * 0.186:+.5f}")
    print(f"transfer ADEQUATE (r >= 0.90 and residual SD <= 0.03): {r >= 0.90 and sd <= 0.03}")
    reg = (y10 >= 0.05) & (y10 <= 0.30)
    print(f"threshold region y10 in [0.05, 0.30]: n {int(reg.sum())}; y9 - y10 there: "
          f"{[(rows[k][0], rows[k][1], round(float(y9[k] - y10[k]), 4)) for k in np.flatnonzero(reg)]}")
    print(f"y9 - y10 over all: median {np.median(y9 - y10):+.4f}, range [{np.min(y9 - y10):+.4f}, {np.max(y9 - y10):+.4f}]")
    np.savez(HERE / "wr9_procx.npz", run=np.array([r[0] for r in rows]), steps=np.array([r[1] for r in rows]),
             y10=y10, y9=y9, rec=rec, a=a, b=b, r=r, sd=sd, thr9=thr9)
    return 0


if __name__ == "__main__":
    sys.exit(main())
