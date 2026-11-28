"""Checks of the independent review's E5 findings (E5-R1 to E5-R4), read from E5's stored files only (no episode is
run). Writes two new files in sim/results/candidates_2026-11-28/E5/:
  e5_review_checks.json      the numbers each finding rests on, recomputed here;
  e5_tuning_seeds_walltime.json  review E5-R4: each tuning row's seeds, conditions and summed episode wall time, and
                             each stage's pool wall time, beside e5_tuning_<size>.json (which records neither).
"""
import csv
import glob
import json
import os
import re
import sys
from pathlib import Path

import numpy as np

BUILD = Path("/home/user/mesozoic-labs/hardware/triceratops_build")
sys.path.insert(0, str(BUILD / "sim"))
import candidate_checks as C  # noqa: E402

E5S = C.CAND_SCRATCH / "E5"
EPI = E5S / "episodes"
OUT = C.RESULTS / "E5"
STAGE_DIR = C.CAND_SCRATCH / "E5_pitch0" / "review_staging"   # copied to OUT once checked
STAGE_DIR.mkdir(parents=True, exist_ok=True)
CHOSEN = {"0.60": "H0.60-s1-24", "0.50": "H0.50-s1-15"}
J = ("J1", "J2", "J3", "J4")


def eps(size, stage, prefix=None):
    out = []
    for f in sorted(glob.glob(str(EPI / size / stage / "*.json"))):
        if prefix is None or os.path.basename(f).startswith(prefix + "__"):
            out.append(json.loads(Path(f).read_text()))
    return out


res = {"experiment": "E5", "evidence": "checks of review findings E5-R1..R4 on E5's stored files (SIMULATION data; "
       "no episode run)", "date": C.DATE}

# ---------------------------------------------------------------- R1
dl = (BUILD / "DECISION_LOG.md").read_text()
hits = [ln[:160] for ln in dl.splitlines() if re.search(r"PitchedTrot|body[ _-]pitch|item 9", ln, re.I)]
design_hits = {}
for p in sorted((BUILD / "design").glob("*.md")):
    t = p.read_text()
    n = len(re.findall(r"PitchedTrot|body[ _-]pitch", t, re.I))
    if n:
        design_hits[p.name] = n
r1 = {"decision_log_lines_matching_PitchedTrot_or_body_pitch_or_item_9": hits,
      "design_md_files_mentioning_PitchedTrot_or_body_pitch": design_hits,
      "chosen": {}, "frozen_heldout_pitch_10_20_deg": {}, "stage1_near_zero_pitch_clean": {}}
for size, cid in CHOSEN.items():
    ch = json.loads((OUT / f"e5_choice_{size}.json").read_text())
    h = json.loads((OUT / f"e5_heldout_{size}.json").read_text())
    pr = [v["pitch_10_20_range_deg"] for ss in h["heldout"]["by_seed_set"].values()
          for v in ss["reported"].values()]
    r1["chosen"][size] = {"id": ch["chosen_id"], "body_pitch_deg": ch["chosen_values"]["body_pitch_deg"],
                          "heldout_pitch_10_20_deg_min": min(x[0] for x in pr),
                          "heldout_pitch_10_20_deg_max": max(x[1] for x in pr),
                          "heldout_condition_mean_pitch_10_20_deg_range": [
                              round(min(v["pitch_10_20_mean_deg"] for ss in h["heldout"]["by_seed_set"].values()
                                        for v in ss["reported"].values()), 3),
                              round(max(v["pitch_10_20_mean_deg"] for ss in h["heldout"]["by_seed_set"].values()
                                        for v in ss["reported"].values()), 3)]}
    sg, ok = [], []
    for e in eps(size, "heldout_frozen"):
        p = e["record"].get("pitch_10_20_deg")
        (sg if e["slow_gait"] else ok).append(p)
    r1["frozen_heldout_pitch_10_20_deg"][size] = {
        "slow_gait_episodes": len(sg), "slow_gait_mean": round(float(np.mean(sg)), 3) if sg else None,
        "slow_gait_range": [min(sg), max(sg)] if sg else None,
        "other_episodes": len(ok), "other_mean": round(float(np.mean(ok)), 3),
        "other_range": [min(ok), max(ok)]}
    rows = list(csv.DictReader((OUT / f"e5_tuning_{size}.csv").open()))
    s2 = {r["id"]: r for r in rows if r["stage"] == "s2"}
    r1["stage1_near_zero_pitch_clean"][size] = [
        {"id": r["id"], "body_pitch_deg": float(r["body_pitch_deg"]), "stage1_rank": int(r["stage1_rank"]),
         "screen_slow_gait_of_20": int(r["screen_slow_gait"]), "screen_worst_hind_knee": float(r["screen_worst_hind_knee"]),
         "stage2_meets_rule_on_101_110": (s2[r["id"]]["meets_rule_on_101_110"] if r["id"] in s2 else "not in stage 2")}
        for r in rows if r["stage"] == "s1" and abs(float(r["body_pitch_deg"])) <= 1.2
        and r["screen_slow_gait"] == "0" and r["screen_falls_J1_J3"] == "0"]
res["R1"] = r1

# ---------------------------------------------------------------- R2
r2 = {"rule": "context only, not judged: an episode whose 2-8 s speed is under 0.225 m/s or whose 2-8 s larger hind "
              "knee RMS is at or over 0.9 of 6.75 N m", "by_size": {}}
for size, cid in CHOSEN.items():
    d = {}
    for stage, pre in (("heldout", cid), ("s2", cid), ("s1", cid), ("heldout_frozen", None)):
        E = eps(size, stage, pre)
        fl = []
        for e in E:
            r = e["record"]
            s, k = r.get("speed_2_8_mps"), r.get("hind_knee_rms_over_rating_2_8")
            if (s is not None and s < 0.225) or (k is not None and k >= 0.9):
                fl.append({"condition": e["condition"], "seed": e["seed"], "speed_2_8_mps": s,
                           "hind_knee_2_8": k, "speed_10_20_mps": r.get("speed_10_20_mps"),
                           "hind_knee_10_20": r.get("hind_knee_rms_over_rating_10_20"),
                           "speed_2_20_mps": r.get("forward_speed_mps"),
                           "hind_knee_2_20": r.get("hind_knee_rms_over_rating_2_20"),
                           "slow_gait": e["slow_gait"]})
        s28 = [e["record"]["speed_2_8_mps"] for e in E if e["record"].get("speed_2_8_mps") is not None]
        k28 = [e["record"]["hind_knee_rms_over_rating_2_8"] for e in E
               if e["record"].get("hind_knee_rms_over_rating_2_8") is not None]
        d[stage] = {"setting": pre or f"H{size}-frozen", "episodes": len(E), "flagged": len(fl),
                    "min_speed_2_8_mps": min(s28), "max_hind_knee_2_8": max(k28),
                    "flagged_episodes": fl if stage != "heldout_frozen" else
                    [{k: x[k] for k in ("condition", "seed", "speed_2_8_mps", "hind_knee_2_8", "slow_gait")}
                     for x in fl]}
    r2["by_size"][size] = d
res["R2"] = r2

# ---------------------------------------------------------------- R3
r3 = {"frozen_50cm_heldout": {}}
h = json.loads((OUT / "e5_heldout_0.50.json").read_text())
for ss, v in h["heldout_frozen"]["by_seed_set"].items():
    rl = v["rule"]
    r3["frozen_50cm_heldout"][ss] = {"slow_gait_per_condition": rl["slow_gait"], "checks": rl["checks"],
                                     "would_meet_rule": rl["SUCCEEDS"],
                                     "worst_hind_knee_per_condition": {k: round(x, 4) for k, x in
                                                                       rl["worst_hind_knee"].items()}}
ch_k = [x for ss in h["heldout"]["by_seed_set"].values() for x in ss["rule"]["worst_hind_knee"].values()]
fz_k = [x for ss in h["heldout_frozen"]["by_seed_set"].values() for x in ss["rule"]["worst_hind_knee"].values()]
r3["per_condition_worst_hind_knee_range"] = {"chosen_H0.50-s1-15": [round(min(ch_k), 3), round(max(ch_k), 3)],
                                             "frozen": [round(min(fz_k), 3), round(max(fz_k), 3)]}
t50 = list(csv.DictReader((OUT / "e5_tuning_0.50.csv").open()))
s00 = [r for r in t50 if r["id"] == "H0.50-s1-00" and r["stage"] == "s1"][0]
r3["frozen_50cm_stage1_screen"] = {"slow_gait_of_20": int(s00["screen_slow_gait"]),
                                   "worst_hind_knee": float(s00["screen_worst_hind_knee"]),
                                   "J3_speed_mps": float(s00["screen_J3_speed_mps"])}
st = (BUILD / "design" / "FORM_FACTOR_STUDY_50_80CM.md").read_text().splitlines()
r3["study_lines"] = {str(i + 1): ln.strip()[:240] for i, ln in enumerate(st)
                     if ("no slow seed" in ln or "A retune does not remove" in ln or "3 of 10 seeds at 0.50 m" in ln
                         or "the adopted gains to 2 of 10" in ln or "The L1 geometry had none" in ln)}
res["R3"] = r3

# ---------------------------------------------------------------- R4
rr = json.loads((OUT / "e5_run_record.json").read_text())
r4 = {"e5_tuning_json_keys": {}, "row_keys": {}, "where_seeds_and_wall_time_live": {
    "seeds": ["e5_tuning_<size>.csv column 'seeds'", "e5_choice_<size>.json 'seeds_used'",
              "e5_run_record.json 'seeds'", "e5_tuning_seeds_walltime.json (this check, new)"],
    "wall_time": ["e5_run_record.json 'stages'[].pools[].wall_s (pool wall time per stage)",
                  "E5_SUMMARY.md 'Wall time on 2 workers'", "e5_heldout_<size>.json 'episode_wall_s_total'",
                  "e5_tuning_seeds_walltime.json (this check, new: summed episode wall time per tuning row)"]}}
addendum = {"experiment": "E5", "evidence": "provenance addendum (review E5-R4); SIMULATION bookkeeping",
            "date": C.DATE, "note": "e5_tuning_<size>.json records neither seeds nor wall time (prereg section 6 "
            "asks for both). This new file adds them beside it; the existing file is unchanged. Episode wall time is "
            "the sum of the setting's per-episode wall_s (one worker each, 2 in parallel); pool wall time is the "
            "stage's elapsed time on 2 workers.", "sizes": {}}
for size in ("0.60", "0.50"):
    t = json.loads((OUT / f"e5_tuning_{size}.json").read_text())
    r4["e5_tuning_json_keys"][size] = list(t.keys())
    r4["row_keys"][size] = list(t["rows"][0].keys())
    rows = []
    for r in t["rows"]:
        sd = list(C.E5_SCREEN_SEEDS) if r["stage"] == "s1" else list(C.E5_TUNING_SEEDS)
        cs = list(C.E5_SCREEN_CONDITIONS) if r["stage"] == "s1" else list(J + tuple(C.E5_PUSHES))
        w = 0.0
        for c in cs:
            for s in sd:
                w += json.loads((EPI / size / r["stage"] / f"{r['id']}__{c}__{s}.json").read_text())["wall_s"]
        rows.append({"stage": r["stage"], "id": r["id"], "seeds": sd, "conditions": cs,
                     "episodes": len(sd) * len(cs), "episode_wall_s_sum": round(w, 1),
                     "finished_utc": r["finished_utc"]})
    pools = [p for s in rr["stages"] if s["size"] == size for p in s["pools"]]
    addendum["sizes"][size] = {"tuning_file": f"e5_tuning_{size}.json", "rows": rows, "stage_pool_wall_s": pools,
                               "workers": rr["workers"], "seeds": rr["seeds"]}
res["R4"] = r4
res["provenance"] = C.provenance(Path(__file__))
addendum["provenance"] = res["provenance"]
(STAGE_DIR / "e5_review_checks.json").write_text(json.dumps(res, indent=1, default=float) + "\n")
(STAGE_DIR / "e5_tuning_seeds_walltime.json").write_text(json.dumps(addendum, indent=1) + "\n")
print(json.dumps({k: res[k] for k in ("R1", "R3")}, indent=1, default=float)[:6000])
print(json.dumps({s: {st: {k: v[k] for k in ("episodes", "flagged", "min_speed_2_8_mps", "max_hind_knee_2_8")}
                      for st, v in d.items()} for s, d in res["R2"]["by_size"].items()}, indent=1))
print(json.dumps(res["R2"]["by_size"]["0.60"]["heldout"]["flagged_episodes"]))
