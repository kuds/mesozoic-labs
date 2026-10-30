"""Verifier: VecNormalize observation statistics read with my own restricted unpickler (no SB3, no gymnasium import).

python -B -I v_stats.py > v_stats.txt     (writes v_stats.json beside this file)

Every class outside numpy's array reconstruction is mapped to an inert Stub that only stores its state, so the
pickles are read as data.  Reports, per lineage: eps / clip / flags, counts, the tail_3 angle mean / sd / scale per
step (median over 12 runs, per-run end/start ratio), every tail dim's end/start scale ratio, the 63-dim ranking by
|log scale ratio| and by mean shift, the start pair's identity across runs, and the implied mean of the frames added
per interval.  Read only.
"""
import hashlib
import io
import json
import math
import pickle
import sys
from pathlib import Path

import numpy as np

RUNS = Path("/tmp/claude-0/-home-user-mesozoic-labs/184ee67c-2802-5235-8e52-6dc7daa72cd3/scratchpad/r10round/ecs/runs")
HERE = Path(__file__).resolve().parent
SE = {"s42": (3_145_728, 4_718_592), "s80": (4_194_304, 5_767_168)}
STEP = 262_144
NAMES = {"s42": [f"s42_A_{n}" for n in range(1101, 1152, 10)] + [f"s42_B_{n}" for n in range(1201, 1252, 10)],
         "s80": [f"s80_A_{n}" for n in range(1301, 1352, 10)] + [f"s80_B_{n}" for n in range(1401, 1452, 10)]}
LAYOUT = (["neck_pitch", "neck_yaw", "head_pitch", "tail_1_pitch", "tail_1_yaw", "tail_2_pitch", "tail_3_pitch",
           "r_hip_pitch", "r_hip_roll", "r_knee", "r_ankle", "r_toe_d2", "r_toe_d3", "r_toe_d4",
           "l_hip_pitch", "l_hip_roll", "l_knee", "l_ankle", "l_toe_d2", "l_toe_d3", "l_toe_d4"])
DIMNAME = [f"{j}:q" for j in LAYOUT] + [f"{j}:v" for j in LAYOUT] + \
    [f"gravity[{i}]" for i in range(3)] + [f"gyro[{i}]" for i in range(3)] + [f"headvel[{i}]" for i in range(3)] + \
    [f"accel[{i}]" for i in range(3)] + ["contact[0]", "contact[1]"] + [f"preydir[{i}]" for i in range(3)] + \
    ["preydist"] + [f"command[{i}]" for i in range(3)]
assert len(DIMNAME) == 63


class Stub:
    def __init__(self, *a, **k):
        self._args = a

    def __setstate__(self, state):
        if isinstance(state, dict):
            self.__dict__.update(state)
        else:
            self._state = state


def _stub_factory(*a, **k):
    return Stub(*a)


class Reader(pickle.Unpickler):
    def find_class(self, module, name):
        if module in ("numpy._core.multiarray", "numpy.core.multiarray") and name in ("_reconstruct", "scalar"):
            return getattr(np._core.multiarray, name)
        if module == "numpy" and name in ("ndarray", "dtype"):
            return getattr(np, name)
        if module.startswith("numpy") and name.startswith("__"):
            return _stub_factory
        return Stub


def read(path: Path):
    raw = path.read_bytes()
    vn = Reader(io.BytesIO(raw)).load()
    rms = vn.obs_rms
    return {"mean": np.asarray(rms.mean, dtype=np.float64), "var": np.asarray(rms.var, dtype=np.float64),
            "count": float(rms.count), "eps": float(vn.epsilon), "clip": float(vn.clip_obs), "norm_obs": bool(vn.norm_obs),
            "norm_reward": bool(vn.norm_reward), "training": bool(vn.training), "sha": hashlib.sha256(raw).hexdigest()}


def med(x):
    return float(np.median(x))


def main():
    out = {}
    flags = set()
    for lin, runs in NAMES.items():
        s, e = SE[lin]
        steps = list(range(s, e + 1, STEP))
        D = {r: {t: read(RUNS / r / "01_stance" / "models" / f"stage1_vecnormalize_{t}_steps.pkl") for t in steps} for r in runs}
        for r in runs:
            for t in steps:
                x = D[r][t]
                flags.add((x["eps"], x["clip"], x["norm_obs"], x["norm_reward"]))
        out[lin] = {r: {str(t): {"mean": D[r][t]["mean"].tolist(), "var": D[r][t]["var"].tolist(), "count": D[r][t]["count"]}
                        for t in steps} for r in runs}
        print(f"\n===== {lin}  ({len(runs)} runs x {len(steps)} checkpoints)")
        shas = {D[r][s]["sha"] for r in runs}
        same = all(np.array_equal(D[r][s]["mean"], D[runs[0]][s]["mean"]) and np.array_equal(D[r][s]["var"], D[runs[0]][s]["var"]) for r in runs)
        print(f"start pickles: {len(shas)} distinct sha256 over runs; statistics identical: {same}")
        print(f"counts: start {D[runs[0]][s]['count']:.4f}; per step (run 0): " + ", ".join(f"{D[runs[0]][t]['count']:.0f}" for t in steps))
        dc = np.diff([D[runs[0]][t]["count"] for t in steps])
        print(f"  frames added per interval: {sorted(set(np.round(dc).astype(int).tolist()))}")
        # tail_3 angle
        print("tail_3 angle (obs 6): step, median mean deg, median sd deg, median scale /deg [min..max]")
        for t in steps:
            mu = [math.degrees(D[r][t]["mean"][6]) for r in runs]
            sd = [math.degrees(math.sqrt(D[r][t]["var"][6])) for r in runs]
            sc = [math.radians(1) / math.sqrt(D[r][t]["var"][6] + D[r][t]["eps"]) for r in runs]
            print(f"  {t:8d}  {med(mu):+.3f}  {med(sd):.3f}  {med(sc):.3f} [{min(sc):.3f}..{max(sc):.3f}]")
        ratio = {r: math.sqrt((D[r][s]["var"][6] + 1e-8) / (D[r][e]["var"][6] + 1e-8)) for r in runs}
        print(f"  end/start scale ratio per run: min {min(ratio.values()):.3f} max {max(ratio.values()):.3f} median {med(list(ratio.values())):.3f}")
        # all dims ratio
        R = np.array([[math.sqrt((D[r][s]["var"][i] + 1e-8) / (D[r][e]["var"][i] + 1e-8)) for i in range(63)] for r in runs])
        shift = np.array([[abs(D[r][e]["mean"][i] - D[r][s]["mean"][i]) / math.sqrt(D[r][s]["var"][i] + 1e-8) for i in range(63)] for r in runs])
        lr = np.median(np.abs(np.log(R)), axis=0)
        order = np.argsort(-lr)
        print("63 dims ranked by median |log(scale_end/scale_start)| (top 6), median ratio:")
        for k in order[:6]:
            print(f"   {k:2d} {DIMNAME[k]:18s} ratio {np.median(R[:, k]):.3f} [{R[:, k].min():.3f}..{R[:, k].max():.3f}]")
        print(f"  rank of obs 6: {int(np.where(order == 6)[0][0]) + 1} of 63; by mean shift rank "
              f"{int(np.where(np.argsort(-np.median(shift, axis=0)) == 6)[0][0]) + 1} (median shift {np.median(shift[:, 6]):.2f} start sd)")
        zero_var = [DIMNAME[i] for i in range(63) if D[runs[0]][s]["var"][i] < 1e-6]
        print(f"  dims with start var < 1e-6: {zero_var}")
        print("  tail dims, median end/start scale ratio [range over runs]:")
        for i in (3, 4, 5, 6, 24, 25, 26, 27):
            print(f"   obs {i:2d} {DIMNAME[i]:16s} {np.median(R[:, i]):.3f} [{R[:, i].min():.3f}..{R[:, i].max():.3f}]  "
                  f"mean shift {np.median(shift[:, i]):.2f} sd0")
        # implied mean of frames added per interval (tail_3 angle)
        imp = []
        for a, b in zip(steps[:-1], steps[1:]):
            vals = []
            for r in runs:
                ca, cb = D[r][a]["count"], D[r][b]["count"]
                vals.append(math.degrees((D[r][b]["mean"][6] * cb - D[r][a]["mean"][6] * ca) / (cb - ca)))
            imp.append(med(vals))
        print("  implied tail_3 mean of frames added per interval (deg, median over runs): " + ", ".join(f"{v:+.2f}" for v in imp))
    print(f"\n(eps, clip, norm_obs, norm_reward) over all pickles: {sorted(flags)}")
    json.dump(out, open(HERE / "v_stats.json", "w"))


if __name__ == "__main__":
    sys.exit(main())
