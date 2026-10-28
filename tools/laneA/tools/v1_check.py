"""V1 (THIS12 lines 838-840, the criteria's V1): M2 on the branch's own tree, zero action, seeds 3042-3081: 40/40
clean; posture equal to DRAFT probe/out/statue_main_3042 within 1e-9 except the reward (LTSB terms; 4470.3686 +/-
0.05).

    PY -I -B v1_check.py M2_STATUE_DIR OUT.json

Compares the runner's M2 statue panel (panel.json, series.npz) with DRAFT's main-statue panel on r10 (08a4d189):
every per-episode field (panel.json "episodes" and "metrics") and every series array, numbers within 1e-9 and
everything else equal, except the fields that are the reward ("reward" or "return" in the name, and
reward_components_per_episode / statue_reference's reward figures, which LTSB's terms change by design).  The excluded
fields are listed in the output.  PASS = 40/40 clean and full horizon, every compared value within 1e-9, mean reward
within 0.05 of 4470.3686, and the runner's physics_revision 10.
"""
from __future__ import annotations

import json
import math
import sys

import numpy as np

sys.dont_write_bytecode = True
VERSION = "v1_check 1.0 (2026-10-26)"
DRAFT = ("/tmp/claude-0/-home-user-mesozoic-labs/184ee67c-2802-5235-8e52-6dc7daa72cd3/scratchpad/r10round/draft/probe/"
         "out/statue_main_3042")
TOL = 1e-9
WANT_REWARD, REWARD_TOL = 4470.3686, 0.05


def is_reward(name: str) -> bool:
    return "reward" in name or "return" in name


def cmp_value(a, b):
    """(equal within TOL, abs diff or None)."""
    if isinstance(a, bool) or isinstance(b, bool) or a is None or b is None or isinstance(a, str) or isinstance(b, str):
        return a == b, None
    if isinstance(a, (int, float)) and isinstance(b, (int, float)):
        if math.isnan(a) and math.isnan(b):
            return True, 0.0
        d = abs(float(a) - float(b))
        return d <= TOL, d
    if isinstance(a, list) and isinstance(b, list):
        if len(a) != len(b):
            return False, None
        worst, ok = 0.0, True
        for x, y in zip(a, b, strict=True):
            e, d = cmp_value(x, y)
            ok = ok and e
            worst = max(worst, d or 0.0)
        return ok, worst
    if isinstance(a, dict) and isinstance(b, dict):
        if set(a) != set(b):
            return False, None
        worst, ok = 0.0, True
        for k in a:
            e, d = cmp_value(a[k], b[k])
            ok = ok and e
            worst = max(worst, d or 0.0)
        return ok, worst
    return a == b, None


def main() -> int:
    print(VERSION, flush=True)
    m2_dir, out_path = sys.argv[1], sys.argv[2]
    got = json.load(open(f"{m2_dir}/panel.json"))
    ref = json.load(open(f"{DRAFT}/panel.json"))
    summ = json.load(open(f"{m2_dir}/m2_summary.json"))
    res = {"tool": VERSION, "m2_dir": m2_dir, "reference": DRAFT, "tolerance": TOL, "fields": {}, "excluded_reward": set(),
           "problems": []}
    for block in ("episodes", "metrics"):
        if len(got[block]) != len(ref[block]):
            res["problems"].append(f"{block}: {len(got[block])} rows vs {len(ref[block])}")
            continue
        for i, (g, r) in enumerate(zip(got[block], ref[block], strict=True)):
            if set(g) != set(r):
                res["problems"].append(f"{block}[{i}] keys differ: {sorted(set(g) ^ set(r))}")
            for k in sorted(set(g) & set(r)):
                if is_reward(k):
                    res["excluded_reward"].add(f"{block}.{k}")
                    continue
                ok, d = cmp_value(g[k], r[k])
                f = res["fields"].setdefault(f"{block}.{k}", {"equal": True, "max_abs_diff": 0.0, "n": 0})
                f["n"] += 1
                f["equal"] = f["equal"] and ok
                f["max_abs_diff"] = max(f["max_abs_diff"], d or 0.0)
    sg, sr = np.load(f"{m2_dir}/series.npz"), np.load(f"{DRAFT}/series.npz")
    if sorted(sg.files) != sorted(sr.files):
        res["problems"].append("series.npz arrays differ in name")
    for k in sorted(set(sg.files) & set(sr.files)):
        a, b = np.asarray(sg[k], dtype=np.float64), np.asarray(sr[k], dtype=np.float64)
        name = "series." + k.rsplit("_", 1)[0]
        f = res["fields"].setdefault(name, {"equal": True, "max_abs_diff": 0.0, "n": 0})
        f["n"] += 1
        if a.shape != b.shape:
            f["equal"] = False
            continue
        d = float(np.max(np.abs(a - b))) if a.size else 0.0
        f["equal"] = f["equal"] and d <= TOL
        f["max_abs_diff"] = max(f["max_abs_diff"], d)
    for k in ("n_clean", "n_full_horizon", "n_episodes", "passed"):
        ok, _ = cmp_value(got["summary"]["verdict"][k], ref["summary"]["verdict"][k])
        res["fields"][f"verdict.{k}"] = {"equal": ok, "max_abs_diff": 0.0, "n": 1}
    for k in sorted(got["summary"]["verdict"]):
        if is_reward(k):
            res["excluded_reward"].add(f"verdict.{k}")
    res["excluded_reward"].add("summary.reward_components_per_episode")
    res["excluded_reward"].add("summary.statue_reference (its reward figures; the panel is its own statue)")
    res["excluded_reward"] = sorted(res["excluded_reward"])
    v = got["summary"]["verdict"]
    mean_reward = v["mean_reward"]
    res["n_clean"], res["n_full_horizon"], res["mean_reward"] = v["n_clean"], v["n_full_horizon"], mean_reward
    res["reference_mean_reward"] = ref["summary"]["verdict"]["mean_reward"]
    res["mean_reward_within_0p05_of_4470p3686"] = abs(mean_reward - WANT_REWARD) <= REWARD_TOL
    res["all_compared_fields_within_1e-9"] = all(f["equal"] for f in res["fields"].values())
    res["worst_abs_diff"] = max(f["max_abs_diff"] for f in res["fields"].values())
    res["runner_physics_revision"] = summ.get("physics_revision")
    res["PASS"] = bool(v["n_clean"] == 40 and v["n_full_horizon"] == 40 and res["all_compared_fields_within_1e-9"]
                       and res["mean_reward_within_0p05_of_4470p3686"] and not res["problems"]
                       and summ.get("physics_revision") == 10)
    with open(out_path, "x") as fh:
        json.dump(res, fh, indent=1)
    print(json.dumps({k: res[k] for k in ("PASS", "n_clean", "n_full_horizon", "mean_reward", "reference_mean_reward",
                                          "all_compared_fields_within_1e-9", "worst_abs_diff", "problems",
                                          "runner_physics_revision")}))
    print("fields compared:", len(res["fields"]), "excluded as reward:", res["excluded_reward"])
    return 0 if res["PASS"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
