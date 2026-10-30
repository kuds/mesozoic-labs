"""VecNormalize observation statistics of every ECS stance checkpoint (start pair and every continuation checkpoint).

PYTHONPATH=<policies/code_copy> python -B read_stats.py OUT.json > stats.txt

Read only: each stage1_vecnormalize_<step>_steps.pkl is opened two ways and the two are checked equal:
  (a) pickle.load (as the verifier's verify_sim.Policy does), and
  (b) the harness's own path, environments.shared.policy_loading.load_sb3_checkpoint(zip, pkl, factory) (as
      run_policies.py / run_clamp.py do), whose VecNormalize then normalises the observation.
No training; nothing is written outside this folder.  Prints, per lineage and step, the tail dimensions' mean,
var and normalised scale 1/sqrt(var + eps), the clip value and the count; the per-run end values; and a ranking of
all 63 dimensions by how far their statistics moved from the start pair to the ends.
"""

from __future__ import annotations

import json
import math
import pickle
import sys
from pathlib import Path

import numpy as np

from environments.shared.config import load_stage_config
from environments.shared.model_id import ModelId
from environments.shared.policy_loading import load_sb3_checkpoint
from environments.shared.species_registry import env_class

RUNS = Path("/tmp/claude-0/-home-user-mesozoic-labs/184ee67c-2802-5235-8e52-6dc7daa72cd3/scratchpad/r10round/ecs/runs")
HERE = Path(__file__).resolve().parent
MID = ModelId("tyrannosaurus_rex", "anatomical")
NAMES = (
    [f"s42_A_{n}" for n in range(1101, 1152, 10)]
    + [f"s42_B_{n}" for n in range(1201, 1252, 10)]
    + [f"s80_A_{n}" for n in range(1301, 1352, 10)]
    + [f"s80_B_{n}" for n in range(1401, 1452, 10)]
)
SE = {"s42": (3_145_728, 4_718_592), "s80": (4_194_304, 5_767_168)}
STEP = 262_144
TAIL = {"t1p:q": 3, "t1y:q": 4, "t2p:q": 5, "t3p:q": 6, "t1p:v": 24, "t1y:v": 25, "t2p:v": 26, "t3p:v": 27}


def steps_of(lin: str) -> list[int]:
    s, e = SE[lin]
    return list(range(s, e + 1, STEP))


def main() -> int:
    out_path = Path(sys.argv[1])
    layout = json.load(open(HERE / "layout.json"))
    names = layout["names"]
    ctor = dict(load_stage_config(MID, "stance")["env_kwargs"])
    factory = lambda: env_class(MID)(**ctor)  # noqa: E731
    rec: dict = {"runs": {}, "obs_names": names}
    checked = 0
    for run in NAMES:
        lin = run[:3]
        models = RUNS / run / "01_stance" / "models"
        rec["runs"][run] = {}
        for st in steps_of(lin):
            pkl = models / f"stage1_vecnormalize_{st}_steps.pkl"
            with open(pkl, "rb") as fh:
                vn = pickle.load(fh)
            mean = np.asarray(vn.obs_rms.mean, dtype=np.float64)
            var = np.asarray(vn.obs_rms.var, dtype=np.float64)
            ent = {"mean": mean.tolist(), "var": var.tolist(), "count": float(vn.obs_rms.count),
                   "eps": float(vn.epsilon), "clip_obs": float(vn.clip_obs), "norm_obs": bool(vn.norm_obs),
                   "norm_reward": bool(vn.norm_reward), "training": bool(getattr(vn, "training", False))}
            # the harness's loader, on the end and start pickles of the first run of each lineage and every 3rd step
            if run in ("s42_A_1101", "s80_A_1301") or st == SE[lin][1]:
                zp = models / f"stage1_{st}_steps.zip"
                _, normalizer, _ = load_sb3_checkpoint(str(zp), str(pkl), factory)
                same = (np.array_equal(normalizer.obs_rms.mean, vn.obs_rms.mean)
                        and np.array_equal(normalizer.obs_rms.var, vn.obs_rms.var)
                        and normalizer.clip_obs == vn.clip_obs and normalizer.epsilon == vn.epsilon)
                rng = np.random.default_rng(st)
                o = (mean + np.sqrt(var) * rng.standard_normal(63)).astype(np.float32)
                hand = np.clip((o - mean) / np.sqrt(var + vn.epsilon), -vn.clip_obs, vn.clip_obs).astype(np.float32)
                via = np.asarray(normalizer.normalize_obs(o), dtype=np.float32)
                ent["harness_equal"] = bool(same)
                ent["harness_norm_maxdiff"] = float(np.max(np.abs(hand - via)))
                try:
                    normalizer.venv.close()
                except Exception:  # noqa: BLE001
                    pass
                checked += 1
            rec["runs"][run][str(st)] = ent
        print(f"# read {run}", file=sys.stderr, flush=True)
    json.dump(rec, open(out_path, "w"))

    # ---- report
    print("VecNormalize observation statistics, ECS stance checkpoints (read_stats.py)")
    flags = {(e["eps"], e["clip_obs"], e["norm_obs"], e["norm_reward"]) for r in rec["runs"].values() for e in r.values()}
    print(f"(eps, clip_obs, norm_obs, norm_reward) over all checkpoints: {sorted(flags)}")
    hc = [e for r in rec["runs"].values() for e in r.values() if "harness_equal" in e]
    print(f"harness loader check: {sum(e['harness_equal'] for e in hc)}/{len(hc)} pickles identical via load_sb3_checkpoint; "
          f"max |hand - normalize_obs| {max(e['harness_norm_maxdiff'] for e in hc):.3g}")
    for lin in ("s42", "s80"):
        runs = [r for r in NAMES if r.startswith(lin)]
        st0 = str(SE[lin][0])
        same_start = all(rec["runs"][r][st0]["mean"] == rec["runs"][runs[0]][st0]["mean"]
                         and rec["runs"][r][st0]["var"] == rec["runs"][runs[0]][st0]["var"] for r in runs)
        print(f"\n=== {lin}: start pair statistics identical in all 12 runs: {same_start}")
        print("count (frames) at start, end:", rec["runs"][runs[0]][st0]["count"],
              [rec["runs"][r][str(SE[lin][1])]["count"] for r in runs][:3], "...")
        print("  scale = 1/sqrt(var+eps); medians over the 12 continuations (range at the end)")
        hdr = "  step      " + " ".join(f"{k:>16s}" for k in TAIL)
        for what in ("mean", "sd", "scale"):
            print(f"  -- {what} (angles in deg / deg^-1 , velocities in deg/s / (deg/s)^-1)")
            print(hdr)
            for st in steps_of(lin):
                vals = []
                for k, i in TAIL.items():
                    xs = []
                    for r in runs:
                        e = rec["runs"][r][str(st)]
                        m, v = e["mean"][i], e["var"][i]
                        if what == "mean":
                            xs.append(math.degrees(m))
                        elif what == "sd":
                            xs.append(math.degrees(math.sqrt(v)))
                        else:
                            xs.append(math.radians(1.0) / math.sqrt(v + e["eps"]))  # per deg
                    vals.append(f"{np.median(xs):16.4g}")
                print(f"  {st:9d} " + " ".join(vals))
        print("  -- end / start ratio of scale, per run (tail_3 angle, tail_3 velocity), and end mean of tail_3 angle (deg)")
        for r in runs:
            e0, e1 = rec["runs"][r][st0], rec["runs"][r][str(SE[lin][1])]
            s = lambda e, i: 1 / math.sqrt(e["var"][i] + e["eps"])  # noqa: E731
            print(f"    {r}: scale ratio t3q {s(e1, 6) / s(e0, 6):.3f}  t3v {s(e1, 27) / s(e0, 27):.3f}  "
                  f"t2q {s(e1, 5) / s(e0, 5):.3f}  t1p {s(e1, 3) / s(e0, 3):.3f}  "
                  f"mean t3q {math.degrees(e0['mean'][6]):+.3f} -> {math.degrees(e1['mean'][6]):+.3f} deg  "
                  f"sd t3q {math.degrees(math.sqrt(e0['var'][6])):.3f} -> {math.degrees(math.sqrt(e1['var'][6])):.3f} deg")
        # drift ranking over all 63 dims
        print("  -- all 63 dims ranked by median over runs of |log(scale_end / scale_start)|; also the mean shift in start "
              "sd units |mu_end - mu_start| / sd_start")
        lr, ms = [], []
        for r in runs:
            e0, e1 = rec["runs"][r][st0], rec["runs"][r][str(SE[lin][1])]
            v0, v1 = np.array(e0["var"]), np.array(e1["var"])
            lr.append(0.5 * np.log((v0 + e0["eps"]) / (v1 + e1["eps"])))
            ms.append(np.abs(np.array(e1["mean"]) - np.array(e0["mean"])) / np.sqrt(v0 + e0["eps"]))
        lr, ms = np.array(lr), np.array(ms)
        med_lr = np.median(lr, axis=0)
        med_ms = np.median(ms, axis=0)
        order = np.argsort(-np.abs(med_lr))
        print(f"    {'rank':>4s} {'dim':>4s} {'name':24s} {'med log scale ratio':>20s} {'ratio':>7s} {'med |dmu|/sd0':>14s}")
        for rank, i in enumerate(order[:15]):
            print(f"    {rank + 1:4d} {i:4d} {names[i]:24s} {med_lr[i]:+20.3f} {math.exp(med_lr[i]):7.3f} {med_ms[i]:14.3f}")
        order2 = np.argsort(-med_ms)
        print("    by mean shift: " + ", ".join(f"{names[i]} {med_ms[i]:.2f}" for i in order2[:8]))
        rank_t3 = int(np.where(order == 6)[0][0]) + 1
        print(f"    tail_3 angle (dim 6) rank by |log scale ratio|: {rank_t3} of 63; by mean shift: "
              f"{int(np.where(order2 == 6)[0][0]) + 1} of 63")
    return 0


if __name__ == "__main__":
    sys.exit(main())
