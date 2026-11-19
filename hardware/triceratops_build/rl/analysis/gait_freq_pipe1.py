"""Diagnostic added after the first results (not in PLAN_GAIT_PIPE1.md; "changed after a result"): the plan's touchdown
count (13-21 per foot per second) looked like contact chatter, not steps. This measures, for the first three seeds at
0.25 m/s (delay 10 ms, randomisation on, deterministic), each foot's stance and air segment lengths and the dominant
frequency (0.3-25 Hz) of every leg joint's speed and every foot's load. Simulation; reported, not judged.

Usage: python gait_freq_pipe1.py CKPT_DIR [OUT_JSON]
"""
import sys, json, math
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parent))
import gait_pipe1 as g
cdir = Path(sys.argv[1])
rec = json.loads((cdir / "run_record.json").read_text()); st = rec["settings"]
cfg = g.TrikeEnvConfig.from_dict({**st["env_config"], "model_path": str(g.GEN / "trike60_C_closed.xml")})
env = g.TrikeEnv(cfg)
stats = g.load_obs_stats(cdir / "trike60_vecnormalize_2007040_steps.pkl")
pol = g.build_policy(env, cdir / "trike60_2007040_steps.zip", st["net"], st["log_std_init"])
out = {}
for seed in g.SEEDS[:3]:
    a, _ = g.run_episode(env, pol, stats, seed, 0.25)
    t = a["t"]; w = t >= 2.0; dt = float(np.median(np.diff(t)))
    down = a["load"][w] > a["contact_n"]
    segs = {}
    for i, c in enumerate(g.CORNERS):
        x = down[:, i].astype(int); edges = np.flatnonzero(np.diff(x)); 
        runs = np.diff(np.r_[0, edges + 1, len(x)]); vals = x[np.r_[0, edges + 1]]
        st_ms = runs[vals == 1] * dt * 1000; sw_ms = runs[vals == 0] * dt * 1000
        segs[c] = {"stance_median_ms": float(np.median(st_ms)), "stance_p90_ms": float(np.percentile(st_ms, 90)),
                   "share_of_stance_time_in_segments_over_100ms": float(st_ms[st_ms > 100].sum() / st_ms.sum()),
                   "air_median_ms": float(np.median(sw_ms)) if len(sw_ms) else None,
                   "air_p90_ms": float(np.percentile(sw_ms, 90)) if len(sw_ms) else None}
    # dominant frequency of each joint's commanded motion (joint speed spectrum), 0.3-25 Hz
    om = a["omega"][w]; n = len(om); f = np.fft.rfftfreq(n, dt)
    band = (f > 0.3) & (f < 25)
    dom = {}
    for j, nm in enumerate(a["names"]):
        P = np.abs(np.fft.rfft(om[:, j] - om[:, j].mean())) ** 2
        dom[nm] = round(float(f[band][np.argmax(P[band])]), 2)
    # load spectrum of each foot
    L = a["load"][w]; domL = {}
    for i, c in enumerate(g.CORNERS):
        P = np.abs(np.fft.rfft(L[:, i] - L[:, i].mean())) ** 2
        domL[c] = round(float(f[band][np.argmax(P[band])]), 2)
    out[seed] = {"stance": segs, "joint_dominant_hz": dom, "foot_load_dominant_hz": domL}
    print(seed, json.dumps({c: {k: (round(v, 1) if isinstance(v, float) else v) for k, v in s.items()} for c, s in segs.items()}))
    print("   joint Hz", dom)
    print("   load Hz", domL)
(Path(sys.argv[2]) if len(sys.argv) > 2 else Path(__file__).resolve().parent / "results" / "gait_freq_pipe1_2026-11-19.json").write_text(json.dumps(out, indent=1) + "\n")
