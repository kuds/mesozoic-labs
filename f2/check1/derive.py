"""Independent re-derivation of the stance6 round's class inputs, classes and reading.

Written by the skeptic check (check1); reads files only, writes one JSON beside this script.
Rules: criteria.txt (lean v6) CLASSES PER RUN, TAIL CLASS, GATE CLASS, RUN OUTCOME, STATISTICAL ANALYSIS,
ROUND READING; THIS12 lines 297-364 for the ANY/TAIL inputs and missing inputs.
Nothing here imports a downloaded file or unpickles anything: CSV and JSON text only.
"""
import csv
import gzip
import hashlib
import io
import json
import math
import sys
from collections import Counter, defaultdict

S6 = "/tmp/claude-0/-home-user-mesozoic-labs/0fdf74a4-2989-56c9-a773-92101004d7de/scratchpad/foot/stance6"
OUT = "/tmp/claude-0/-home-user-mesozoic-labs/184ee67c-2802-5235-8e52-6dc7daa72cd3/scratchpad/r10round/f2/check1/derive_out.json"
RUNS = ["stance6_r10LTSB_s60", "stance6_r10LTSB_s42", "stance6_r10LTSB_s80"]
ACT = ["neck_pitch_act", "neck_yaw_act", "head_pitch_act", "r_hip_pitch_act", "r_hip_roll_act", "r_knee_act",
       "r_ankle_act", "l_hip_pitch_act", "l_hip_roll_act", "l_knee_act", "l_ankle_act", "tail_1_pitch_act",
       "tail_1_yaw_act", "tail_2_pitch_act", "tail_3_pitch_act"]  # MJCF actuator order on copy_8737cd3d
TAIL = {11, 12, 13, 14}
TAIL_PITCH = {11, 13, 14}
GRID = 131072
E_FINISHED = 12001280
J = ["4194304", "5242880", "6291456", "8388608", "10485760", "final"]
R1, R2 = "10485760", "final"
BAR = 0.10


def sha(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def missing(cell):
    if cell is None:
        return True
    s = cell.strip()
    if s == "":
        return True
    try:
        v = float(s)
    except ValueError:
        return True
    return math.isnan(v)


def fnum(cell):
    return float(cell.strip())


def act_index(name):
    s = (name or "").strip()
    if s in ACT:
        return ACT.index(s)
    if s.startswith("actuator[") and s.endswith("]"):
        try:
            return int(s[len("actuator["):-1])
        except ValueError:
            return None
    return None


def ledger(path):
    """M5 (a)-(c) and C6's row rules, per THIS12 lines 340-364 and criteria M5 / C6."""
    rows = list(csv.DictReader(open(path, newline="")))
    out = {"rows": len(rows), "c6_failures": [], "evals": {}}
    by_step = defaultdict(list)
    for i, r in enumerate(rows):
        t = r["timesteps"].strip()
        try:
            ts = int(float(t))
            ok = float(t) == ts and ts % GRID == 0 and GRID <= ts <= E_FINISHED
        except ValueError:
            ok = False
        if not ok:
            out["c6_failures"].append(("off-grid row", i, t))
            continue
        by_step[ts].append(r)
    grid = [k * GRID for k in range(1, E_FINISHED // GRID + 1)]
    out["grid_evaluations"] = len(grid)
    for ts in grid:
        rs = by_step.get(ts, [])
        eps = sorted(int(float(r["episode"])) for r in rs if not missing(r["episode"]))
        if len(rs) != 10 or eps != list(range(10)):
            out["c6_failures"].append(("evaluation rows", ts, len(rs), eps))
        any_over = 0
        tail_over = Counter()
        tail_rows = 0
        unresolved = 0
        any_by_act = Counter()
        row_missing = []
        for r in rs:
            clean = r["screen_v2_clean"]
            reasons = [x for x in (r["screen_v2_reasons"] or "").split(";") if x]
            frac_c = r["saturated_actuator_fraction"]
            frac_m = missing(frac_c)
            frac = None if frac_m else fnum(frac_c)
            length_c = r["length"]
            outlived = missing(length_c) or fnum(length_c) > 200
            name = r["saturated_actuator"]
            idx = act_index(name)
            screen_measured = not missing(clean)
            # ANY input
            if screen_measured:
                listed = "max_actuator_saturation_fraction" in reasons
                if not listed and frac is not None and frac > BAR:
                    row_missing.append(("contradiction: screen does not list saturation, fraction > bar", r["episode"]))
                    continue
                any_flag = listed
            else:
                if frac is None:
                    row_missing.append(("unmeasured screen and missing fraction", r["episode"]))
                    continue
                any_flag = frac > BAR
            # TAIL input
            tail_missing = False
            if outlived and frac is None and (not screen_measured or "max_actuator_saturation_fraction" in reasons):
                tail_missing = True
            if frac is not None and frac > BAR and idx is None:
                tail_missing = True
            if frac is not None and frac > BAR and idx is not None and idx not in TAIL:
                if missing(r["saturated_actuator_ties"]) or missing(r["actuators_saturated"]):
                    tail_missing = True
            if tail_missing:
                row_missing.append(("tail input missing", r["episode"]))
                continue
            if any_flag:
                any_over += 1
                any_by_act[name or "(unnamed)"] += 1
            if frac is not None and frac > BAR and idx in TAIL:
                ps = r["saturated_actuator_positive_share"]
                if missing(ps):
                    sign = "?"
                else:
                    p = fnum(ps)
                    sign = "+" if p > 0.5 else ("-" if p < 0.5 else "tie")
                tail_over[ACT[idx] + sign] += 1
                tail_rows += 1
            if (frac is not None and frac > BAR and idx is not None and idx not in TAIL
                    and (fnum(r["saturated_actuator_ties"]) >= 1 or fnum(r["actuators_saturated"]) >= 2)):
                unresolved += 1
        if row_missing:
            out["c6_failures"].append(("missing inputs", ts, row_missing))
        out["evals"][ts] = {"any": any_over, "tail": tail_rows, "tail_by": dict(tail_over), "unres": unresolved,
                            "any_by": dict(any_by_act), "rail_actuators": dict(Counter(r["rail_actuators"] for r in rs))}
    ev = out["evals"]
    any_flagged = [ts for ts in grid if ev[ts]["any"] >= 5]
    tail_flagged = [ts for ts in grid if ev[ts]["tail"] >= 5]
    any_alarm = [ts for ts in grid if ts - GRID in ev and ev[ts]["any"] >= 5 and ev[ts - GRID]["any"] >= 5]
    tail_alarm = [ts for ts in grid if ts - GRID in ev and ev[ts]["tail"] >= 5 and ev[ts - GRID]["tail"] >= 5]
    needs_res = []
    for ts in any_alarm:
        if ts in tail_alarm:
            continue
        a, b = ev[ts - GRID], ev[ts]
        if a["tail"] + a["unres"] >= 5 and b["tail"] + b["unres"] >= 5:
            needs_res.append(ts)
    out.update(any_flagged=any_flagged, tail_flagged=tail_flagged, any_alarm=any_alarm, tail_alarm=tail_alarm,
               needs_resolution=needs_res,
               any_alarm_detail=[{"step": ts, "any_by": ev[ts]["any_by"], "count": ev[ts]["any"]} for ts in any_alarm],
               tail_alarm_detail=[{"step": ts, "tail_by": ev[ts]["tail_by"]} for ts in tail_alarm])
    del out["evals"]
    out["per_eval_nonzero"] = {ts: v for ts, v in ev.items() if v["any"] or v["tail"] or v["unres"]}
    return out


def trace_label(run, label):
    """M1 reduction redone from the per-step trace CSVs (window steps 200..L-1, |command| >= 0.99 bar)."""
    d = f"{S6}/panels/{run}/{label}/m1"
    summ = json.load(open(f"{d}/m1_summary.json"))
    eps = []
    csv_sha_ok = True
    for seed in range(3042, 3062):
        raw = gzip.open(f"{d}/episode_trace_seed{seed}.csv.gz").read()
        if hashlib.sha256(raw).hexdigest() != summ["csv_sha256"][f"episode_trace_seed{seed}.csv"]:
            csv_sha_ok = False
        rows = list(csv.DictReader(io.StringIO(raw.decode())))
        L = len(rows)
        steps = [int(r["step"]) for r in rows]
        assert steps == list(range(L)), (run, label, seed)
        win = [r for r in rows if int(r["step"]) >= 200]
        assert all(r["in_window"] == "1" for r in win) and all(r["in_window"] == "0" for r in rows[:200]), (run, label, seed)
        e = {"seed": seed, "length": L, "full": L == 1000, "acts": {}}
        for i, a in enumerate(ACT):
            if not win:
                e["acts"][i] = None
                continue
            c = [float(r[f"{a}_command"]) for r in win]
            pos = sum(1 for x in c if x >= 0.99)
            neg = sum(1 for x in c if x <= -0.99)
            share = (pos + neg) / len(c)
            e["acts"][i] = {"mean": sum(c) / len(c), "share": share, "pos": pos, "neg": neg,
                            "over": share > BAR,
                            "signs": (["+"] if pos >= neg else []) + (["-"] if neg >= pos else []) if share > BAR else []}
        eps.append(e)
    cmd = {}
    n = {}
    for i in TAIL:
        vals = [e["acts"][i]["mean"] for e in eps if e["acts"][i] is not None]
        n[i] = len(vals)
        cmd[i] = sum(vals) / len(vals) if vals else None
    flags = []
    fall = []
    for e in eps:
        for i in TAIL:
            x = e["acts"][i]
            if x and x["over"]:
                for s in x["signs"]:
                    (flags if e["full"] else fall).append((e["seed"], i, s))
    any_over_eps = sorted({e["seed"] for e in eps for i in range(15) if e["acts"][i] and e["acts"][i]["over"]})
    # compare with lane A's summary
    lane_cmd = {int(k): v for k, v in summ["cmd_a"].items()}
    cmp_cmd = max(abs(lane_cmd[i] - cmd[i]) for i in TAIL)
    lane_flags = sorted((f["seed"], int(f["actuator"]), f["sign"]) for f in summ["tail_flags"]) if summ["tail_flags"] and isinstance(summ["tail_flags"][0], dict) else summ["tail_flags"]
    return {"cmd": cmd, "n": n, "flags": sorted(flags), "fall_flags": sorted(fall), "full_horizon": sum(e["full"] for e in eps),
            "any_over_eps": any_over_eps, "csv_sha_ok": csv_sha_ok, "lane_cmd_maxdiff": cmp_cmd,
            "lane_tail_flags": lane_flags, "lane_full_horizon": summ["full_horizon_episodes"],
            "physics_revision": summ["physics_revision"], "ckpt_sha": summ["checkpoint"]["sha256"],
            "vecnorm_sha": summ["checkpoint"]["vecnorm_sha256"], "ckpt_path": summ["checkpoint"]["path"],
            "vecnorm_path": summ["checkpoint"]["vecnorm"], "seeds": summ["seeds"], "copy_commit": summ["copy_seal"]["commit"],
            "tree_digest": summ["copy_seal"]["tree_digest"], "tree_after": summ["copy_tree_digest_after"],
            "num_timesteps": summ["checkpoint"]["num_timesteps"], "seed_run": summ["checkpoint"]["seed"],
            "trace_rc": summ["trace_run"]["returncode"], "trace_tb": summ["trace_run"]["traceback"],
            "tool_sha": summ["tool_sha256"]}


def fisher_one_sided_lower(a, n, b, m):
    """P(X <= a) for X hypergeometric: a successes of n in group 1, b of m in group 2."""
    K, N = a + b, n + m

    def h(x):
        return math.comb(K, x) * math.comb(N - K, n - x) / math.comb(N, n)
    return sum(h(x) for x in range(max(0, n - (N - K)), a + 1))


def main():
    res = {}
    for run in RUNS:
        base = f"{S6}/fetch/F2/{run}/copies"
        man = json.load(open(f"{base}/artifact_manifest.json.copy1"))
        ent = {e["path"]: e for e in man["files"]}
        r = {"manifest_status": man.get("status"), "manifest_run_id": man.get("run_id"),
             "manifest_schema": man.get("schema_version"), "manifest_entries": len(man["files"])}
        # C3 files: compare every copy on disk with the manifest
        c3 = ["provenance.json", "01_stance/stage_config.json", "01_stance/progress.json",
              "01_stance/eval_episodes.csv", "01_stance/gate_verdict.json"]
        if "01_stance/hop_watch.json" in ent:
            c3.append("01_stance/hop_watch.json")
        for lab in J[:-1]:
            c3 += [f"01_stance/models/stage1_{lab}_steps.zip", f"01_stance/models/stage1_vecnormalize_{lab}_steps.pkl"]
        c3 += ["01_stance/models/stage1_final.zip", "01_stance/models/stage1_final_vecnorm.pkl"]
        c3res = {}
        for p in c3:
            if p not in ent:
                c3res[p] = "NOT IN MANIFEST"
                continue
            want = ent[p]["sha256"].split(":", 1)[-1]
            import glob
            copies = sorted(glob.glob(f"{base}/{p}.copy*"))
            got = [(c.rsplit(".", 1)[-1], sha(c) == want, __import__("os").path.getsize(c) == ent[p]["size_bytes"]) for c in copies]
            c3res[p] = got
        r["c3"] = c3res
        r["hop_watch_listed"] = "01_stance/hop_watch.json" in ent
        # ledger
        r["ledger"] = ledger(f"{base}/01_stance/eval_episodes.csv.copy1")
        # traces
        tr = {lab: trace_label(run, lab) for lab in J}
        # check each trace read the manifest's pair
        for lab, t in tr.items():
            if lab == "final":
                zp, pp = "01_stance/models/stage1_final.zip", "01_stance/models/stage1_final_vecnorm.pkl"
            else:
                zp, pp = f"01_stance/models/stage1_{lab}_steps.zip", f"01_stance/models/stage1_vecnormalize_{lab}_steps.pkl"
            t["pair_matches_manifest"] = (t["ckpt_sha"] == ent[zp]["sha256"].split(":", 1)[-1]
                                          and t["vecnorm_sha"] == ent[pp]["sha256"].split(":", 1)[-1])
        r["traces"] = {lab: {k: v for k, v in t.items()} for lab, t in tr.items()}
        # MEAN_a
        mean = {}
        for i in TAIL:
            a, b = tr[R1]["cmd"][i], tr[R2]["cmd"][i]
            if a is not None and b is not None:
                mean[i] = (a + b) / 2
            elif a is not None or b is not None:
                mean[i] = a if a is not None else b
            else:
                mean[i] = None
        r["mean"] = mean
        # TRACE RAIL
        flags = tr[R1]["flags"] + tr[R2]["flags"]
        by = Counter((i, s) for (_, i, s) in flags)
        trace_rail = []
        single = []
        for (i, s), k in by.items():
            m = mean[i]
            mean_reach = m is not None and ((s == "+" and m >= 0.60) or (s == "-" and m <= -0.60))
            if k >= 2 or (k >= 1 and mean_reach):
                db = {}
                for lab in (R1, R2):
                    if any(f[1] == i and f[2] == s for f in tr[lab]["flags"]):
                        c = tr[lab]["cmd"][i]
                        db[lab] = c is not None and ((s == "+" and c >= 0.60) or (s == "-" and c <= -0.60))
                trace_rail.append({"act": ACT[i], "sign": s, "flags": k, "basis": "2+ flags" if k >= 2 else "1 flag, MEAN_a",
                                   "drive_backed": any(db.values()), "by_ckpt": db})
            else:
                single.append({"act": ACT[i], "sign": s})
        r["trace_rail"] = trace_rail
        r["single_tail_flags"] = single
        # PARKED
        parked = []
        for i in TAIL_PITCH:
            w = []
            if mean[i] is not None and mean[i] >= 0.60:
                w.append(("END", round(mean[i] - 0.60, 4)))
            jh = [(lab, round(tr[lab]["cmd"][i] - 0.60, 4)) for lab in J if tr[lab]["cmd"][i] is not None and tr[lab]["cmd"][i] >= 0.60]
            if len(jh) >= 2:
                w.append(("J", jh))
            if w:
                parked.append({"act": ACT[i], "windows": w})
        r["parked"] = parked
        parked_acts = {p["act"] for p in parked}
        driven = [ACT[i] for i in TAIL if mean[i] is not None and abs(mean[i]) >= 0.60
                  and not (ACT[i] in parked_acts and mean[i] >= 0.60)]
        r["driven"] = driven
        r["cmd_by_label"] = {lab: {ACT[i]: tr[lab]["cmd"][i] for i in TAIL} for lab in J}
        # class
        led = r["ledger"]
        tail_alarm = led["tail_alarm"]
        tail_flag_end = bool(tr[R1]["flags"] or tr[R2]["flags"])
        if tail_alarm or trace_rail:
            if tail_alarm and tail_flag_end:
                kind = "RAIL-LEDGER-PERSISTENT"
            elif tail_alarm:
                kind = "RAIL-LEDGER-RESOLVED"
            else:
                kind = "RAIL-END-ONLY"
            cls = "RAIL"
        elif parked:
            cls, kind = "PARKED", None
        else:
            cls, kind = "CLEAR", None
        r["class"], r["kind"] = cls, kind
        m3 = json.load(open(f"{S6}/panels/{run}/final/m3/m3_summary.json"))
        rep = json.load(open(f"{S6}/panels/{run}/final/m3/stance_heading_report.json"))
        r["m3"] = {"verdict": m3["verdict"], "passed": m3["passed"], "report_passed": rep["passed"],
                   "report_sha_ok": sha(f"{S6}/panels/{run}/final/m3/stance_heading_report.json") == m3["report_sha256"],
                   "ckpt_ok": m3["checkpoint"]["sha256"] == ent["01_stance/models/stage1_final.zip"]["sha256"].split(":", 1)[-1],
                   "vecnorm_ok": m3["checkpoint"]["vecnorm_sha256"] == ent["01_stance/models/stage1_final_vecnorm.pkl"]["sha256"].split(":", 1)[-1],
                   "failures": m3["failures"], "physics_revision": m3["physics_revision"],
                   "headings": [(h["yaw_deg"], h["n_clean"], h["n_episodes"]) for h in m3["headings"]]}
        gate = "PASS" if (m3["passed"] and rep["passed"]) else "FAIL"
        r["gate"] = gate
        reasons = []
        if cls == "RAIL":
            reasons.append(f"RAIL ({kind})")
        if cls == "PARKED":
            reasons.append("PARKED")
        if gate == "FAIL":
            reasons.append("GATE FAIL")
        r["outcome"] = "POSITIVE" if (cls == "CLEAR" and gate == "PASS") else "NEGATIVE"
        r["outcome_reasons"] = reasons
        r["any_alarm"] = bool(led["any_alarm"])
        res[run] = r
    A = sum(1 for r in res.values() if r["any_alarm"])
    nA = len(res)  # all three COMPLETE FINISHED (checked separately)
    p = fisher_one_sided_lower(A, nA, 2, 5)
    R = sum(1 for r in res.values() if r["class"] == "RAIL")
    P = sum(1 for r in res.values() if r["class"] == "PARKED")
    G = sum(1 for r in res.values() if r["outcome"] == "POSITIVE")
    U = 0
    if R >= 1:
        ledger_basis = any(r["class"] == "RAIL" and r["ledger"]["tail_alarm"] for r in res.values())
        line = "line 1: TAIL RAIL FORMED ON r10 IN TRAINING " + ("(ledger)" if ledger_basis else "(end-only ...)")
    elif U >= 1:
        line = "line 2"
    elif P >= 1:
        line = "line 3"
    elif G >= 2:
        line = "line 4"
    else:
        line = "line 5"
    summary = {"A": A, "n_A": nA, "fisher_p": p, "R": R, "U": U, "P": P, "G+": G, "reading": line}
    json.dump({"runs": res, "summary": summary}, open(OUT, "w"), indent=1, default=str)
    print(json.dumps(summary))
    for run, r in res.items():
        led = r["ledger"]
        print(run, r["class"], r["kind"], "gate", r["gate"], "outcome", r["outcome"], r["outcome_reasons"])
        print("  C6 failures", len(led["c6_failures"]), "grid", led["grid_evaluations"], "rows", led["rows"])
        print("  any_flagged", len(led["any_flagged"]), "tail_flagged", len(led["tail_flagged"]))
        print("  any_alarm", led["any_alarm"])
        print("  tail_alarm", led["tail_alarm"])
        print("  any_alarm detail", led["any_alarm_detail"])
        print("  needs_resolution", led["needs_resolution"])
        print("  MEAN", {ACT[i]: r["mean"][i] for i in TAIL})
        for lab in J:
            t = r["traces"][lab]
            print("  ", lab, {ACT[i]: round(t["cmd"][i], 4) for i in TAIL}, "n", t["n"], "full", t["full_horizon"],
                  "flags", Counter((ACT[i], s) for _, i, s in t["flags"]), "fall", t["fall_flags"],
                  "lane_diff", f"{t['lane_cmd_maxdiff']:.2e}", "csv_ok", t["csv_sha_ok"], "pair_ok", t["pair_matches_manifest"],
                  "phys", t["physics_revision"], "seeds", t["seeds"][0], t["seeds"][-1], len(t["seeds"]),
                  "copy", t["copy_commit"][:8], t["tree_digest"][:8], t["tree_after"][:8], "rc", t["trace_rc"], t["trace_tb"])
        print("  trace_rail", r["trace_rail"], "single", r["single_tail_flags"])
        print("  parked", r["parked"], "driven", r["driven"])
        print("  m3", r["m3"])
        print("  c3", {k: v for k, v in r["c3"].items()})


if __name__ == "__main__":
    sys.exit(main())
