"""Verifier's own procedure X (warning_r9 PREREG.txt section 3), written independently of wr9_procx.py.
Network evaluation only: r10 checkpoints (F2 copies, read only) on the r10 and r9 probe banks' statue_settled raw
states.  The observation is normalised by hand from the pickled obs_rms (not normalize_obs) and the Gaussian mean is
taken through mlp_extractor.forward_actor + action_net (not get_distribution).  Also: does the loop gain G_ref
(early_warning ew_series.csv) explain the bank difference y9 - y10 (warning_r9 F8, an [inferred] hypothesis)?
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

torch.set_num_threads(1)
F2 = Path("/tmp/claude-0/-home-user-mesozoic-labs/0fdf74a4-2989-56c9-a773-92101004d7de/scratchpad/foot/stance6/fetch/F2")
TR = Path("/tmp/claude-0/-home-user-mesozoic-labs/184ee67c-2802-5235-8e52-6dc7daa72cd3/scratchpad/tailround/fetch")
NOTES = Path("/tmp/claude-0/-home-user-mesozoic-labs/184ee67c-2802-5235-8e52-6dc7daa72cd3/scratchpad/com_revisit/notes")


def ss_rows(npz):
    z = np.load(npz, allow_pickle=False)
    lab = z["bank_labels"].astype(str)
    return z["bank_raw_obs"].astype(np.float64)[lab == "statue_settled"]


def main():
    from environments.shared.config import load_stage_config
    from environments.shared.model_id import ModelId
    from environments.shared.policy_loading import load_sb3_checkpoint
    from environments.shared.species_registry import env_class

    mid = ModelId("tyrannosaurus_rex", "anatomical")
    kw = dict(load_stage_config(mid, "stance")["env_kwargs"])
    factory = lambda: env_class(mid)(**kw)  # noqa: E731
    X10 = ss_rows(F2 / "stance6_r10LTSB_s42/copies/01_stance/policy_probe.npz.copy1")
    X9s = {r: ss_rows(TR / f"stance5_LTSB_{r}/A/policy_probe.npz") for r in ("s42", "s52", "s60", "s70", "s80")}
    X9 = X9s["s42"]
    print(f"r9 statue_settled rows identical across the 5 runs: {all(np.array_equal(X9, v) for v in X9s.values())}")
    diff = np.abs(X9 - X10).max(axis=0)
    print(f"channels differing (> 1e-3): {int((diff > 1e-3).sum())} of {X10.shape[1]}; (> 1e-6): {int((diff > 1e-6).sum())}; "
          f"max |diff| {diff.max():.3f} at channel {int(diff.argmax())}")
    print(f"tail_3 obs[6] deg: r10 {np.degrees(X10[:, 6]).round(3).tolist()} r9 {np.degrees(X9[:, 6]).round(3).tolist()}")
    rows = []
    for run in ("s42", "s60", "s80"):
        m = F2 / f"stance6_r10LTSB_{run}/copies/01_stance/models"
        steps = sorted(int(p.name.split("_")[1]) for p in m.glob("stage1_*_steps.zip.copy1")) + [12_001_280]
        for n in steps:
            if n == 12_001_280:
                z, p = m / "stage1_final.zip.copy1", m / "stage1_final_vecnorm.pkl.copy1"
            else:
                z, p = m / f"stage1_{n}_steps.zip.copy1", m / f"stage1_vecnormalize_{n}_steps.pkl.copy1"
            model, norm, _ = load_sb3_checkpoint(str(z), str(p), factory)
            mu_, var_ = np.asarray(norm.obs_rms.mean, np.float64), np.asarray(norm.obs_rms.var, np.float64)
            eps, clip = float(norm.epsilon), float(norm.clip_obs)
            pol = model.policy

            def mean14(X):
                xn = np.clip((X - mu_) / np.sqrt(var_ + eps), -clip, clip).astype(np.float32)
                with torch.no_grad():
                    f = pol.extract_features(torch.as_tensor(xn), pol.pi_features_extractor) \
                        if hasattr(pol, "pi_features_extractor") else pol.extract_features(torch.as_tensor(xn))
                    lat = pol.mlp_extractor.forward_actor(f)
                    return pol.action_net(lat).numpy()[:, 14].astype(np.float64)

            rows.append({"run": run, "steps": n, "y10": float(mean14(X10).mean()), "y9": float(mean14(X9).mean())})
    d = pd.DataFrame(rows)
    old = np.load(NOTES / "warning_r9/wr9_procx.npz")
    od = pd.DataFrame({"run": old["run"].astype(str), "steps": old["steps"].astype(np.int64), "y10_o": old["y10"], "y9_o": old["y9"]})
    m_ = d.merge(od, on=["run", "steps"], how="outer")
    print(f"checkpoints mine {len(d)}, theirs {len(od)}, matched {int(m_[['y10', 'y10_o']].notna().all(axis=1).sum())}; "
          f"max |y10 - theirs| {np.nanmax(np.abs(m_['y10'] - m_['y10_o'])):.2e}; max |y9 - theirs| {np.nanmax(np.abs(m_['y9'] - m_['y9_o'])):.2e}")
    b, a = np.polyfit(d["y10"], d["y9"], 1)
    res = d["y9"] - (a + b * d["y10"])
    r = float(np.corrcoef(d["y10"], d["y9"])[0, 1])
    print(f"fit y9 = {a:+.4f} + {b:.4f} y10; Pearson r {r:.4f}; residual SD ddof2 {res.std(ddof=2):.4f} (ddof0 {res.std(ddof=0):.4f}); "
          f"thr9 = {a + b * 0.150:.4f}")
    ew = pd.read_csv(NOTES / "early_warning/ew_series.csv")
    ew = ew[ew["run"].isin(["s42", "s60", "s80"])][["run", "steps", "G_ref", "c0_ref", "G_own", "probe_bank_ss"]]
    j = d.merge(ew, on=["run", "steps"], how="left")
    print(f"joined to ew_series G_ref: {int(j['G_ref'].notna().sum())} of {len(j)}")
    jj = j.dropna(subset=["G_ref"])
    dd = jj["y9"] - jj["y10"]
    print(f"F8 check: corr(y9 - y10, G_ref) {np.corrcoef(dd, jj['G_ref'])[0, 1]:+.3f} (n {len(jj)}); "
          f"corr(y9, G_ref) {np.corrcoef(jj['y9'], jj['G_ref'])[0, 1]:+.3f}; corr(y9, y10) {np.corrcoef(jj['y9'], jj['y10'])[0, 1]:+.3f}")
    A = np.column_stack([np.ones(len(jj)), jj["y10"], jj["G_ref"]])
    coef, *_ = np.linalg.lstsq(A, jj["y9"].to_numpy(), rcond=None)
    pred = A @ coef
    ss_res = float(np.sum((jj["y9"] - pred) ** 2))
    ss_tot = float(np.sum((jj["y9"] - jj["y9"].mean()) ** 2))
    print(f"  y9 ~ 1 + y10 + G_ref: coef {np.round(coef, 4).tolist()}; R^2 {1 - ss_res / ss_tot:.3f} (y10 alone R^2 {r * r:.3f}); "
          f"residual SD {np.sqrt(ss_res / (len(jj) - 3)):.4f}")
    for run in ("s42", "s60", "s80"):
        q = jj[jj["run"] == run]
        print(f"  {run}: corr(y9 - y10, G_ref) {np.corrcoef(q['y9'] - q['y10'], q['G_ref'])[0, 1]:+.3f} (n {len(q)}); "
              + " ".join(f"{int(s / 1e5) / 10:.1f}M:G{g:+.2f}/y9{y:+.2f}" for s, g, y in zip(q["steps"], q["G_ref"], q["y9"])))
    j.to_csv(Path(__file__).resolve().parent / "vwf_procx.csv", index=False)
    return 0


if __name__ == "__main__":
    sys.exit(main())
