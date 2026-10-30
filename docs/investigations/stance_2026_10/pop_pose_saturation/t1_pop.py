"""Track 1: reset 'pop' transient vs static equilibrium under the zero-action statue."""
import json
import sys
import numpy as np

sys.path.insert(0, "scratchpad")
from common import make_env, hinge_info, actuator_info, rollout, zero_policy, custom_reset  # noqa: E402

OUT = "scratchpad/t1"
import os  # noqa: E402

os.makedirs(OUT, exist_ok=True)
env, u = make_env()
H = hinge_info(u)
A = actuator_info(u)
jn = [h["name"] for h in H]
qpos0 = u.model.key_qpos[u.home_keyframe_id][[h["qadr"] for h in H]]
WIN = [(0, 10), (10, 25), (25, 50), (50, 200), (200, 1000)]
NSEED = int(sys.argv[1]) if len(sys.argv) > 1 else 20


def summarize(recs, label):
    n = min(r["n"] for r in recs)
    ke = np.array([r["ke"][:n] for r in recs])
    vroot = np.array([np.linalg.norm(r["vroot"][:n], axis=1) for r in recs])
    wroot = np.array([np.linalg.norm(r["wroot"][:n], axis=1) for r in recs])
    qdmax = np.array([np.abs(r["qd"][:n]).max(axis=1) for r in recs])
    z = np.array([r["z"][:n] for r in recs])
    bw = np.array([r["peak_bw"][:n] for r in recs])
    tilt = np.array([r["tilt"][:n] for r in recs])
    out = {"label": label, "episodes": len(recs), "min_len": n,
           "ends": {e: sum(r["end"] == e for r in recs) for e in set(r["end"] for r in recs)}}
    out["reset_penetration_mm"] = float(np.mean([r["reset_pen"][0] for r in recs]) * 1e3)
    out["reset_penetration_mm_min"] = float(np.min([r["reset_pen"][0] for r in recs]) * 1e3)
    out["per_step_first"] = {
        "t": list(range(0, 60)),
        "ke_J": ke.mean(0)[:60].round(4).tolist(),
        "vroot": vroot.mean(0)[:60].round(4).tolist(),
        "qd_max": qdmax.mean(0)[:60].round(3).tolist(),
        "z": z.mean(0)[:60].round(5).tolist(),
        "peak_bw": bw.mean(0)[:60].round(3).tolist(),
    }
    win = {}
    for a, b in WIN:
        if b > n:
            continue
        w = {
            "ke_J_mean": float(ke[:, a:b].mean()), "ke_J_max": float(ke[:, a:b].max()),
            "vroot_mean": float(vroot[:, a:b].mean()), "vroot_max": float(vroot[:, a:b].max()),
            "wroot_mean": float(wroot[:, a:b].mean()), "qd_max_mean": float(qdmax[:, a:b].mean()),
            "qd_max_max": float(qdmax[:, a:b].max()),
            "z_min": float(z[:, a:b].min()), "z_max": float(z[:, a:b].max()), "z_mean": float(z[:, a:b].mean()),
            "peak_bw_mean": float(bw[:, a:b].mean()), "peak_bw_max": float(bw[:, a:b].max()),
            "tilt_deg_mean": float(np.degrees(tilt[:, a:b]).mean()),
        }
        terms = {}
        for k in recs[0]["terms"]:
            terms[k] = float(np.mean([r["terms"][k][a:b].mean() for r in recs]))
        w["terms_per_step"] = terms
        w["total_per_step"] = float(np.mean([r["total"][a:b].mean() for r in recs]))
        win[f"{a}-{b}"] = w
    out["windows"] = win
    out["episode_return_mean"] = float(np.mean([r["total"].sum() for r in recs]))
    out["episode_return_std"] = float(np.std([r["total"].sum() for r in recs]))
    # settle times: last step where KE above thresholds
    def last_above(x, thr):
        idx = np.where(x > thr)[0]
        return int(idx[-1]) + 1 if idx.size else 0
    out["settle"] = {
        "ke_peak_J_mean": float(ke.max(1).mean()),
        "ke_peak_step_median": float(np.median(ke.argmax(1))),
        "ke_ss_J_mean_200_1000": float(ke[:, 200:].mean()) if n >= 1000 else None,
        "last_step_ke_gt_10pct_peak_median": float(np.median([last_above(k, 0.1 * k.max()) for k in ke])),
        "last_step_ke_gt_1pct_peak_median": float(np.median([last_above(k, 0.01 * k.max()) for k in ke])),
        "last_step_vroot_gt_2cm_s_median": float(np.median([last_above(v, 0.02) for v in vroot])),
        "last_step_vroot_gt_5mm_s_median": float(np.median([last_above(v, 0.005) for v in vroot])),
        "last_step_qdmax_gt_0p1_median": float(np.median([last_above(v, 0.1) for v in qdmax])),
        "last_step_qdmax_gt_0p02_median": float(np.median([last_above(v, 0.02) for v in qdmax])),
    }
    return out


results = {}
# A: noise-free
recA = rollout(env, u, zero_policy, reset=lambda: custom_reset(env, u, 0))
results["A_noise_free"] = summarize([recA], "noise-free reset, action=0")
# per-joint settle analysis (noise-free)
qT = recA["q"][-1]
dq = recA["q"] - qT  # deviation from final
joint_settle = {}
for j, name in enumerate(jn):
    dev = np.abs(dq[:, j])
    tol = max(np.radians(0.1), 0.05 * abs(qT[j] - qpos0[j]))
    idx = np.where(dev > tol)[0]
    joint_settle[name] = {
        "qpos0_deg": float(np.degrees(qpos0[j])),
        "q_final_deg": float(np.degrees(qT[j])),
        "delta_deg": float(np.degrees(qT[j] - qpos0[j])),
        "q_at": {str(s): float(np.degrees(recA["q"][s, j] - qpos0[j])) for s in (0, 2, 5, 10, 20, 50, 100, 200, 500)},
        "settle_step(<max(0.1deg,5%))": int(idx[-1]) + 1 if idx.size else 0,
        "reset_qacc": float(recA["reset_qacc"][H[j]["dadr"]]),
    }
results["A_joint_settle"] = joint_settle
results["A_root_reset_qacc"] = recA["reset_qacc"][:6].round(4).tolist()
results["A_final_root"] = {"z": float(recA["z"][-1]), "forward_z": float(recA["fz"][-1]),
                           "tilt_deg": float(np.degrees(recA["tilt"][-1]))}
# Actuator tension at settled pose (noise-free)
act_rows = {}
for i, a in enumerate(A):
    j = jn.index(a["joint"])
    q = recA["q"][-1, j]
    ctrl = recA["ctrl"][-1, i]
    act_rows[a["name"]] = {
        "kp": a["kp"], "kv": a["kv"], "home_ctrl_deg": float(np.degrees(a["home"])),
        "q_settled_deg": float(np.degrees(q)), "pd_err_deg": float(np.degrees(ctrl - q)),
        "force_Nm": float(recA["afrc"][-1, i]), "forcerange": a["forcerange"].tolist(),
        "force_frac": float(abs(recA["afrc"][-1, i]) / max(abs(a["forcerange"][1]), 1e-9)),
        "ctrlrange_deg": np.degrees(a["ctrlrange"]).round(2).tolist(),
        "jrange_deg": np.degrees(a["jrange"]).round(2).tolist(),
    }
results["A_actuators_settled"] = act_rows
# passive spring torques at settled pose
results["A_spring_torque_settled"] = {
    h["name"]: float(-h["stiffness"] * (recA["q"][-1, k] - h["springref"])) for k, h in enumerate(H) if h["stiffness"] > 0
}

# B: native noisy reset
recB = [rollout(env, u, zero_policy, reset=lambda s=s: env.reset(seed=s)) for s in range(NSEED)]
results["B_native_noise_0.05"] = summarize(recB, "native reset noise 0.05 (qpos+qvel incl. root)")
# C: qpos noise only
recC = [rollout(env, u, zero_policy, reset=lambda s=s: custom_reset(env, u, s, qpos_noise=0.05)) for s in range(NSEED)]
results["C_qpos_noise_only"] = summarize(recC, "joint qpos noise 0.05 only")
# D: qvel noise only
recD = [rollout(env, u, zero_policy, reset=lambda s=s: custom_reset(env, u, s, qvel_noise=0.05)) for s in range(NSEED)]
results["D_qvel_noise_only"] = summarize(recD, "qvel noise 0.05 only (incl. root)")
# Native noise: settled pose spread vs qpos0
qfin = np.array([r["q"][-1] for r in recB if r["n"] == 1000])
results["B_settled_minus_qpos0_deg"] = {
    jn[j]: [float(np.degrees((qfin[:, j] - qpos0[j]).mean())), float(np.degrees((qfin[:, j] - qpos0[j]).std()))]
    for j in range(len(jn))
}
results["B_settled_minus_noisefree_deg_rms"] = float(np.degrees(np.sqrt(((qfin - qT) ** 2).mean())))
np.savez_compressed(f"{OUT}/t1_traces.npz",
                    A_q=recA["q"], A_ke=recA["ke"], A_z=recA["z"], A_bw=recA["peak_bw"],
                    B_ke=np.array([r["ke"] for r in recB if r["n"] == 1000]),
                    B_q=np.array([r["q"] for r in recB if r["n"] == 1000]),
                    B_bw=np.array([r["peak_bw"] for r in recB if r["n"] == 1000]),
                    B_z=np.array([r["z"] for r in recB if r["n"] == 1000]))
with open(f"{OUT}/t1_results.json", "w") as f:
    json.dump(results, f, indent=1, default=float)
print("done", {k: v.get("episode_return_mean") for k, v in results.items() if isinstance(v, dict) and "episode_return_mean" in v})
