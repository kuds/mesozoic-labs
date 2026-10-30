"""Does the Velociraptor's slide at reset noise 0.05 die out?  The same spawns run 60 s (max_episode_steps 6000 on the
test class's env; the 10 s episode is its first 1000 steps, checked against scan.jsonl), neutral action, base plant.
Per whole second: RMS horizontal CoM speed and the largest foot-site speed (control-rate differences); per-foot world
lateral (y) and fore-aft floor force (mj_contactForce at each control step, read-only); the per-foot load oscillation
over the last 2 s (scanlib's recorder).  Usage: python velo_ext.py out.jsonl"""

import json
import sys

import mujoco
import numpy as np

import scanlib as S
from environments.shared.gait.morphology import Morphology
from environments.shared.tests.com_stance_helpers import run_tracked_episode

CLS = S.CLASSES["velo"]
SEEDS_005 = [35, 69, 76, 95, 126, 145, 151, 4, 20, 50, 111, 199, 1, 3]  # 7 in-window CoM > 5 mm, 5 more travel-flagged, 2 unflagged
CASES = [(0.05, s) for s in SEEDS_005] + [(0.02, 20), (0.02, 274), (None, 0)]
STEPS = 6000
scan = {}
for line in open("scan.jsonl"):
    r = json.loads(line)
    if r["model"] == "velo":
        scan[(r["noise"], r["seed"])] = r
out = open(sys.argv[1], "w")
for noise, seed in CASES:
    inst = CLS()
    kw = {"max_episode_steps": STEPS}
    if noise is not None:
        kw["reset_noise_scale"] = noise
    env = inst.make_env(**kw)
    u = env.unwrapped
    m = u.model
    morph = Morphology.from_env(env, CLS.model_id)
    weight = float(m.body_subtreemass[morph.root_body]) * abs(float(m.opt.gravity[2]))
    rec = S.LoadRecorder(env, morph)
    u._substep_probe_hook = rec
    gf = np.asarray(morph.geom_foot)
    isf = np.asarray(morph.is_floor)
    f6 = np.zeros(6)
    tang = []

    def post(env_, step, action):
        d = u.data
        ft = np.zeros((len(morph.feet), 2))
        for i in range(int(d.ncon)):
            c = d.contact[i]
            g1, g2 = int(c.geom1), int(c.geom2)
            if isf[g1] == isf[g2]:
                continue
            g = g2 if isf[g1] else g1
            if gf[g] < 0:
                continue
            mujoco.mj_contactForce(m, d, i, f6)
            fw = c.frame.reshape(3, 3).T @ f6[:3]
            if fw[2] < 0:
                fw = -fw
            ft[gf[g]] += fw[:2]
        tang.append(ft)

    ep = run_tracked_episode(env, CLS.model_id, seed=seed, post_step=post, morphology=morph, steps=STEPS)
    env.close()
    tr = ep.track
    t = np.asarray(tr.t)
    com = np.asarray(tr.com)[:, :2]
    feet = np.asarray(tr.foot_pos)[:, :, :2]
    dt = float(ep.dt)
    vcom = np.linalg.norm(np.diff(com, axis=0), axis=1) / dt
    vfoot = np.linalg.norm(np.diff(feet, axis=0), axis=2).max(axis=1) / dt
    tm = t[1:]
    sec = np.floor(tm - 1e-9).astype(int)
    nsec = int(sec.max()) + 1
    com_sec = [1e3 * float(np.sqrt(np.mean(vcom[sec == s] ** 2))) for s in range(nsec)]
    foot_sec = [1e3 * float(vfoot[sec == s].max()) for s in range(nsec)]
    tang = np.array(tang)
    lat_sec = [[round(float(np.mean(tang[sec == s, k, 1])), 3) for k in range(tang.shape[1])] for s in range(nsec)]
    # consistency with the 10 s scan episode: the CoM at step 1000
    ref = scan.get((noise, seed))
    i10 = int(np.argmax(t >= 10.0 - 1e-9))
    i1 = int(np.argmax(t >= 1.0 - 1e-9))
    i3 = int(np.argmax(t >= 3.0 - 1e-9))
    sub = S.substep_summary(rec, weight, 3.0, float(m.opt.timestep))
    row = {
        "noise": noise, "seed": seed, "stood": ep.stood, "steps": ep.steps,
        "com_travel_3_10_mm": 1e3 * float(np.linalg.norm(com[i10] - com[i3])),
        "scan_com_travel_window_mm": ref["com_travel_window_mm"] if ref else None,
        "com_travel_10_60_mm": 1e3 * float(np.linalg.norm(com[-1] - com[i10])),
        "com_travel_30_60_mm": 1e3 * float(np.linalg.norm(com[-1] - com[int(np.argmax(t >= 30.0 - 1e-9))])),
        "foot_travel_after1s_to60_mm": 1e3 * float(np.linalg.norm(feet[i1:] - feet[i1], axis=2).max()),
        "foot_travel_10_60_mm": 1e3 * float(np.linalg.norm(feet[i10:] - feet[i10], axis=2).max()),
        "com_speed_rms_per_s_mmps": [round(v, 4) for v in com_sec],
        "foot_speed_max_per_s_mmps": [round(v, 4) for v in foot_sec],
        "lateral_force_per_s_n": lat_sec,
        "load_sd_tail_bw_max": sub["load_sd_tail_bw_max"],
        "root_vz_tail_rms_mmps": sub["root_vz_tail_rms_mmps"],
    }
    out.write(json.dumps(row) + "\n")
    out.flush()
    cs = row["com_speed_rms_per_s_mmps"]
    print(f"velo noise {noise} seed {seed}: stood {ep.stood}; CoM travel 3-10 s {row['com_travel_3_10_mm']:.3f} mm (scan "
          f"{row['scan_com_travel_window_mm']}), 10-60 s {row['com_travel_10_60_mm']:.3f}, 30-60 s {row['com_travel_30_60_mm']:.3f} mm; "
          f"CoM speed RMS (mm/s) s3 {cs[3]:.4f} s9 {cs[9]:.4f} s19 {cs[19]:.4f} s29 {cs[29]:.4f} s59 {cs[59]:.4f}; "
          f"foot max speed s9 {foot_sec[9]:.4f} s59 {foot_sec[59]:.4f} mm/s; foot travel 10-60 s {row['foot_travel_10_60_mm']:.3f} mm; "
          f"lateral force s1 {lat_sec[1]} s5 {lat_sec[5]} s9 {lat_sec[9]} s29 {lat_sec[29]} s59 {lat_sec[59]} N; "
          f"load sd tail {row['load_sd_tail_bw_max']:.2e} BW", flush=True)
