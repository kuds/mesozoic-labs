"""PHASE PROBE, POST HOC (not pre-registered): does the critic's value gradient on EVERY joint angle follow that joint's
drift with the episode phase?  Offline only: the value MLP is rebuilt from the checkpoint zip's policy.pth (read
only, torch.load weights_only) and checked against the V recorded in the rollout; gradients by autograd.  No policy is
trained.  Writes posthoc_joints.json / .txt in this folder.

Per checkpoint and joint angle j (observations 0..20):
  dS_j      episode-paired mean angle over k in [200,1000) minus over [50,200) (deg);
  crit_j    the critic's mean dV/d(angle_j) per degree over all states, over B1 = [50,200) and over B3 = [400,1000);
  bW3_j     the realised-return slope inside B3 (OLS G ~ Z on B3 states, the angle's coefficient per degree);
  bM0_j / bM2_j  the angle's coefficient for G in M0 (Z) and M2 (Z + bins + settle ramp), per degree.
Summary: across the 21 joints, corr(crit_j, dS_j) and corr(crit_j, bM0_j - bM2_j) (the phase-explained part of the
realised slope), and the count of joints where crit_B3_j and bW3_j differ in sign or where crit exceeds bW3.
"""
import os
import sys

os.environ["PYTHONDONTWRITEBYTECODE"] = "1"
sys.dont_write_bytecode = True
for _k in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ.setdefault(_k, "1")

import importlib.util  # noqa: E402
import io  # noqa: E402
import json  # noqa: E402
import math  # noqa: E402
import zipfile  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402

HERE = Path(__file__).resolve().parent
JOINTS = ["neck_pitch", "neck_yaw", "head_pitch", "tail_1_pitch", "tail_1_yaw", "tail_2_pitch", "tail_3_pitch",
          "r_hip_pitch", "r_hip_roll", "r_knee", "r_ankle", "r_toe_d2", "r_toe_d3", "r_toe_d4",
          "l_hip_pitch", "l_hip_roll", "l_knee", "l_ankle", "l_toe_d2", "l_toe_d3", "l_toe_d4"]


def load_analyze():
    spec = importlib.util.spec_from_file_location("phase_analyze", HERE / "phase_analyze.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules["phase_analyze"] = mod
    spec.loader.exec_module(mod)
    return mod


def value_mlp(zip_path):
    import torch

    with zipfile.ZipFile(zip_path, "r") as zf:
        sd = torch.load(io.BytesIO(zf.read("policy.pth")), map_location="cpu", weights_only=True)
    keys = sorted(k for k in sd if k.startswith("mlp_extractor.value_net.") or k.startswith("value_net."))
    lin = sorted({int(k.split(".")[2]) for k in keys if k.startswith("mlp_extractor.value_net.")})
    layers = [(sd[f"mlp_extractor.value_net.{i}.weight"].double(), sd[f"mlp_extractor.value_net.{i}.bias"].double())
              for i in lin]
    head = (sd["value_net.weight"].double(), sd["value_net.bias"].double())

    def f(z):
        h = z
        for w, b in layers:
            h = torch.tanh(h @ w.T + b)
        return (h @ head[0].T + head[1])[:, 0]
    return f, keys


def main():
    import torch

    torch.set_num_threads(1)
    A = load_analyze()
    labels = sys.argv[1].split(",") if len(sys.argv) > 1 else list(A.LABELS)
    out, lines = {}, []
    for lab in labels:
        f = HERE / "rollouts" / f"{lab}.npz"
        if not f.is_file():
            continue
        d = A.Data(lab)
        meta = d.meta
        V, keys = value_mlp(meta["zip"])
        Z = torch.as_tensor(d.Z, dtype=torch.float64)
        if d.blind:
            Z = Z.clone()
            Z[:, 6] = 0.0
        Zg = Z.clone().requires_grad_(True)
        v = V(Zg)
        g = torch.autograd.grad(v.sum(), Zg)[0].numpy()
        vmax = float(np.max(np.abs(v.detach().numpy() - d.V)))
        sd = np.sqrt(np.asarray(meta["obs_var"], dtype=np.float64) + float(meta["epsilon"]))
        gdeg = g * (math.pi / 180.0) / sd
        if d.blind:
            gdeg[:, 6] = 0.0
        k = d.k
        B1 = (k >= 50) & (k < 200)
        B3 = k >= 400
        X0 = A.design(d, "M0")
        X2 = A.design(d, "M2")
        b0 = A.EpOLS(X0, d.G, d.rows).solve()
        b2 = A.EpOLS(X2, d.G, d.rows).solve()
        rows3 = [r[d.k[r] >= 400] for r in d.rows]
        bw3 = A.EpOLS(X0, d.G, rows3).solve()
        bv3 = A.EpOLS(X0, d.V, rows3).solve()
        rec = []
        for j, name in enumerate(JOINTS):
            ang = np.degrees(d.raw[:, j])
            dS = np.full(d.n_ep, np.nan)
            for e, r in enumerate(d.rows):
                kk = d.k[r]
                a, b = kk >= 200, (kk >= 50) & (kk < 200)
                if a.any() and b.any():
                    dS[e] = ang[r][a].mean() - ang[r][b].mean()
            if j in d.zcols:
                c = 1 + int(np.flatnonzero(d.zcols == j)[0])
                conv = (math.pi / 180.0) / sd[j]
                bm0, bm2, bW3, bV3 = b0[c] * conv, b2[c] * conv, bw3[c] * conv, bv3[c] * conv
            else:
                bm0 = bm2 = bW3 = bV3 = float("nan")
            rec.append({"joint": name, "obs": j, "dS_deg": float(np.nanmean(dS)),
                        "angle_sd_deg": float(ang[k >= 50].std()),
                        "crit_all": float(gdeg[:, j].mean()), "crit_B1": float(gdeg[B1, j].mean()),
                        "crit_B3": float(gdeg[B3, j].mean()), "bM0": float(bm0), "bM2": float(bm2),
                        "bW3": float(bW3), "bV_W3": float(bV3)})
        cr = np.array([r["crit_all"] for r in rec])
        ds = np.array([r["dS_deg"] for r in rec])
        ph = np.array([r["bM0"] - r["bM2"] for r in rec])
        c3 = np.array([r["crit_B3"] for r in rec])
        w3 = np.array([r["bW3"] for r in rec])
        ok = np.isfinite(ph) & np.isfinite(w3)
        sdj = np.array([r["angle_sd_deg"] for r in rec])
        # standardised (per SD of the joint's angle over k >= 50): the toes' tiny SDs no longer dominate
        cr_s, ph_s, c3_s, w3_s, ds_s = cr * sdj, ph * sdj, c3 * sdj, w3 * sdj, ds / sdj
        summ_std = {"corr_crit_dS_std": float(np.corrcoef(cr_s, ds_s)[0, 1]),
                    "corr_crit_phase_part_std": float(np.corrcoef(cr_s[ok], ph_s[ok])[0, 1]),
                    "corr_critB3_bW3_std": float(np.corrcoef(c3_s[ok], w3_s[ok])[0, 1]),
                    "sign_agree_crit_vs_dS_times_Bbar": int(np.sum(np.sign(cr) == np.sign(ds) * np.sign(
                        float(d.gate_step[k >= 200].mean()))))}
        summ = {"max_abs_V_rebuilt_minus_recorded": vmax, "value_keys": keys, **summ_std,
                "corr_crit_dS": float(np.corrcoef(cr, ds)[0, 1]),
                "corr_crit_phase_part": float(np.corrcoef(cr[ok], ph[ok])[0, 1]),
                "corr_critB3_bW3": float(np.corrcoef(c3[ok], w3[ok])[0, 1]),
                "n_joints_critB3_gt_bW3": int(np.sum(c3[ok] > w3[ok])), "n_joints": int(ok.sum()),
                "B_bar": float(d.gate_step[k >= 200].mean())}
        out[lab] = {"summary": summ, "joints": rec}
        lines.append(f"=== {lab}: V rebuilt vs recorded max |diff| {vmax:.2e}; B_bar {summ['B_bar']:+.5f}; across 21 "
                     f"joints corr(critic slope, dS) {summ['corr_crit_dS']:+.3f}, corr(critic slope, phase part of the "
                     f"realised slope M0-M2) {summ['corr_crit_phase_part']:+.3f}, corr(critic B3 slope, realised B3 "
                     f"slope) {summ['corr_critB3_bW3']:+.3f}; critic B3 slope > realised B3 slope on "
                     f"{summ['n_joints_critB3_gt_bW3']}/{summ['n_joints']} joints; standardised per angle SD: "
                     f"corr(critic, dS) {summ['corr_crit_dS_std']:+.3f}, corr(critic, M0-M2) "
                     f"{summ['corr_crit_phase_part_std']:+.3f}, corr(critic B3, realised B3) "
                     f"{summ['corr_critB3_bW3_std']:+.3f}; sign(critic) = sign(dS x B_bar) on "
                     f"{summ['sign_agree_crit_vs_dS_times_Bbar']}/21")
        lines.append(f"  {'joint':<13} {'dS deg':>8} {'sd deg':>7} {'crit all':>9} {'crit B1':>8} {'crit B3':>8} "
                     f"{'G M0':>8} {'G M2':>8} {'G W3':>8} {'V W3':>8}   (slopes per degree, normalised return)")
        for r in rec:
            lines.append(f"  {r['joint']:<13} {r['dS_deg']:+8.3f} {r['angle_sd_deg']:7.3f} {r['crit_all']:+9.4f} "
                         f"{r['crit_B1']:+8.4f} {r['crit_B3']:+8.4f} {r['bM0']:+8.4f} {r['bM2']:+8.4f} "
                         f"{r['bW3']:+8.4f} {r['bV_W3']:+8.4f}")
        print("\n".join(lines[-23:]), flush=True)
    (HERE / "posthoc_joints.json").write_text(json.dumps(out, indent=1))
    (HERE / "posthoc_joints.txt").write_text("POST HOC (not pre-registered)\n" + "\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
