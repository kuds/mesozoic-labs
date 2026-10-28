"""Independent re-application of the lean v6 class / reading rules (verify_rules lens).

Reads only JSON/CSV: F2 manifest-checked copies of eval_episodes.csv and artifact_manifest.json,
and the measurers' m1_summary.json / m3_summary.json. Never unpickles or imports a download.
"""
import csv
import hashlib
import json
import math
import sys
from math import comb

S = "/tmp/claude-0/-home-user-mesozoic-labs/0fdf74a4-2989-56c9-a773-92101004d7de/scratchpad/foot/stance6"
RUNS = {"s60": 60, "s42": 42, "s80": 80}
J = ["4194304", "5242880", "6291456", "8388608", "10485760", "final"]
TAIL = {11: "tail_1_pitch_act", 12: "tail_1_yaw_act", 13: "tail_2_pitch_act", 14: "tail_3_pitch_act"}
TAIL_PITCH = [11, 13, 14]
GRID = [131072 * k for k in range(1, 92)]
APPROVED_HEAD = "8737cd3d86431de1bdd0fcdb49a0d956a45e084d"


def sha(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def num(x):
    try:
        v = float(x)
    except (TypeError, ValueError):
        return None
    return None if math.isnan(v) else v


def actuator_index(name):
    if not name:
        return None
    name = name.strip()
    if name.startswith("actuator[") and name.endswith("]"):
        return int(name[9:-1])
    names = {"neck_pitch_act": 0, "tail_1_pitch_act": 11, "tail_1_yaw_act": 12, "tail_2_pitch_act": 13,
             "tail_3_pitch_act": 14}
    return names.get(name, name)


out = {}
for r, seed in RUNS.items():
    rid = f"stance6_r10LTSB_{r}"
    base = f"{S}/fetch/F2/{rid}/copies"
    man = json.load(open(f"{base}/artifact_manifest.json.copy1"))
    mf = {e["path"]: e["sha256"].split(":", 1)[-1] for e in man["files"]}
    led = f"{base}/01_stance/eval_episodes.csv.copy1"
    led_ok = sha(led) == mf["01_stance/eval_episodes.csv"]
    rows = list(csv.DictReader(open(led, newline="")))
    by_step = {}
    for row in rows:
        by_step.setdefault(int(float(row["timesteps"])), []).append(row)
    problems = []
    if sorted(by_step) != GRID:
        problems.append(("grid", sorted(set(by_step) ^ set(GRID))))
    evals = []
    for step in GRID:
        rs = by_step.get(step, [])
        if len(rs) != 10 or sorted(int(x["episode"]) for x in rs) != list(range(10)):
            problems.append(("rows", step, len(rs)))
        any_over = 0
        tail_over = {}
        unres = 0
        unres_any = 0
        missing = 0
        for x in rs:
            clean = x["screen_v2_clean"].strip()
            reasons = x["screen_v2_reasons"]
            frac = num(x["saturated_actuator_fraction"])
            measured = clean not in ("", "nan", "NaN")
            if measured:
                over = "max_actuator_saturation_fraction" in reasons
                if not over and frac is not None and frac > 0.10:
                    missing += 1
                    continue
            else:
                if frac is None:
                    missing += 1
                    continue
                over = frac > 0.10
            a = actuator_index(x["saturated_actuator"])
            tail_row = frac is not None and frac > 0.10 and isinstance(a, int) and a in TAIL
            if over:
                any_over += 1
            if tail_row:
                sp = num(x["saturated_actuator_positive_share"])
                sign = "+" if (sp is not None and sp > 0.5) else ("-" if (sp is not None and sp < 0.5) else "tie")
                tail_over[(a, sign)] = tail_over.get((a, sign), 0) + 1
            if frac is not None and frac > 0.10 and isinstance(a, int) and a not in TAIL:
                ties = num(x["saturated_actuator_ties"])
                nsat = num(x["actuators_saturated"])
                if ties is None or nsat is None:
                    missing += 1
                elif ties >= 1 or nsat >= 2:
                    unres += 1
        tail_n = sum(tail_over.values())
        evals.append({"step": step, "any": any_over, "tail": tail_n, "tail_by": {f"{k[0]}{k[1]}": v for k, v in tail_over.items()},
                      "unres": unres, "missing": missing})
    any_alarms, tail_alarms, needs_res = [], [], []
    for i in range(1, len(evals)):
        p, c = evals[i - 1], evals[i]
        if p["any"] >= 5 and c["any"] >= 5:
            any_alarms.append(c["step"])
        if p["tail"] >= 5 and c["tail"] >= 5:
            tail_alarms.append(c["step"])
        elif p["any"] >= 5 and c["any"] >= 5:
            if p["tail"] + p["unres"] >= 5 and c["tail"] + c["unres"] >= 5:
                needs_res.append(c["step"])
    tail_flagged = sum(1 for e in evals if e["tail"] >= 5)
    any_flagged = sum(1 for e in evals if e["any"] >= 5)
    # M1 at J / R1 / R2
    m1 = {}
    for lab in J:
        d = json.load(open(f"{S}/panels/{rid}/{lab}/m1/m1_summary.json"))
        ck = d["checkpoint"]
        zipname = "stage1_final.zip" if lab == "final" else f"stage1_{lab}_steps.zip"
        m1[lab] = {
            "cmd": {int(k): v for k, v in d["cmd_a"].items()},
            "cmd_n": {int(k): v for k, v in d["cmd_a_episodes"].items()},
            "flags": d["tail_flags_full_horizon_by_actuator_sign"],
            "phys": d["physics_revision"],
            "commit": d["copy_seal"]["commit"],
            "seeds_ok": d["seeds"] == list(range(3042, 3062)),
            "zip_sha_ok": ck["sha256"] == mf[f"01_stance/models/{zipname}"],
            "run_seed": d["filed"]["seed"],
        }
    R1, R2 = m1["10485760"], m1["final"]
    mean = {a: (R1["cmd"][a] + R2["cmd"][a]) / 2 for a in TAIL}
    # TRACE RAIL
    flag_counts = {}
    for lab in ("10485760", "final"):
        for k, v in m1[lab]["flags"].items():
            flag_counts[k] = flag_counts.get(k, 0) + v
    trace_rail = []
    for k, v in flag_counts.items():
        a, sg = int(k[:-1]), k[-1]
        mean_in_sign = mean[a] if sg == "+" else -mean[a]
        if v >= 2 or (v >= 1 and mean_in_sign >= 0.60):
            trace_rail.append((k, v))
    flag_at_end = any(v > 0 for v in flag_counts.values())
    # PARKED
    parked = []
    for a in TAIL_PITCH:
        jhits = [(lab, round(m1[lab]["cmd"][a] - 0.60, 4)) for lab in J if m1[lab]["cmd"][a] >= 0.60]
        if mean[a] >= 0.60 or len(jhits) >= 2:
            parked.append((TAIL[a], round(mean[a] - 0.60, 4), jhits))
    driven = [TAIL[a] for a in TAIL if abs(mean[a]) >= 0.60 and TAIL[a] not in [p[0] for p in parked]]
    if tail_alarms or trace_rail:
        cls = "RAIL"
        if tail_alarms and flag_at_end:
            kind = "RAIL-LEDGER-PERSISTENT"
        elif tail_alarms:
            kind = "RAIL-LEDGER-RESOLVED"
        else:
            kind = "RAIL-END-ONLY"
    elif parked:
        cls, kind = "PARKED", None
    else:
        cls, kind = "CLEAR", None
    m3 = json.load(open(f"{S}/panels/{rid}/final/m3/m3_summary.json"))
    gate = "PASS" if m3["verdict"] == "PASS" else "FAIL"
    reasons = []
    if cls != "CLEAR":
        reasons.append(cls)
    if gate == "FAIL":
        reasons.append("GATE FAIL")
    outcome = "POSITIVE" if (cls == "CLEAR" and gate == "PASS") else "NEGATIVE"
    out[r] = {
        "ledger_sha_equals_manifest": led_ok, "ledger_problems": problems,
        "missing_inputs": sum(e["missing"] for e in evals),
        "any_flagged": any_flagged, "tail_flagged": tail_flagged,
        "any_alarms": any_alarms, "tail_alarms": tail_alarms, "needs_resolution": needs_res,
        "unresolved_rows": sum(e["unres"] for e in evals),
        "m1_checks": {lab: {k: m1[lab][k] for k in ("phys", "commit", "seeds_ok", "zip_sha_ok", "run_seed")} for lab in J},
        "cmd_tail3_J": {lab: round(m1[lab]["cmd"][14], 4) for lab in J},
        "cmd_n_min": min(min(m1[lab]["cmd_n"].values()) for lab in J),
        "flags_R1R2": flag_counts, "mean": {TAIL[a]: round(mean[a], 4) for a in TAIL},
        "trace_rail": trace_rail, "parked": parked, "driven": driven,
        "class": cls, "kind": kind, "gate": gate, "m3_phys": m3["physics_revision"],
        "m3_commit": m3["copy_seal"]["commit"], "outcome": outcome, "reasons": reasons,
    }

# Statistic A
A = sum(1 for r in out if out[r]["any_alarms"])
nA = 3  # every run COMPLETE FINISHED (score C5 FINISHED; checked separately)
N, K, n = nA + 5, A + 2, nA
p = sum(comb(K, x) * comb(N - K, n - x) for x in range(0, A + 1)) / comb(N, n)
R = sum(1 for r in out if out[r]["class"] == "RAIL")
P = sum(1 for r in out if out[r]["class"] == "PARKED")
G = sum(1 for r in out if out[r]["outcome"] == "POSITIVE")
U = 0
if R >= 1:
    basis = "(ledger)" if any(out[r]["class"] == "RAIL" and out[r]["tail_alarms"] for r in out) else "(end-only ...)"
    line = f"line 1 {basis}"
elif U >= 1:
    line = "line 2"
elif P >= 1:
    line = "line 3"
elif G >= 2:
    line = "line 4"
else:
    line = "line 5"
res = {"runs": out, "A": A, "n_A": nA, "fisher_p_one_sided_r10_lower": round(p, 4), "R": R, "U": U, "P": P, "G+": G,
       "line": line}
json.dump(res, open(sys.argv[1], "w"), indent=1, default=str)
for r in out:
    o = out[r]
    print(r, o["class"], o["kind"], "GATE", o["gate"], o["outcome"], o["reasons"], "ANY", len(o["any_alarms"]),
          "TAIL", len(o["tail_alarms"]), "first", (o["tail_alarms"] or o["any_alarms"] or [None])[0],
          "tail_flagged", o["tail_flagged"], "any_flagged", o["any_flagged"], "needs_res", o["needs_resolution"],
          "unres_rows", o["unresolved_rows"], "missing", o["missing_inputs"], "ledger_ok", o["ledger_sha_equals_manifest"],
          o["ledger_problems"])
    print("   mean", o["mean"], "flags R1R2", o["flags_R1R2"], "trace_rail", o["trace_rail"])
    print("   parked", o["parked"], "driven", o["driven"], "cmd tail3 J", o["cmd_tail3_J"], "cmd_n_min", o["cmd_n_min"])
    print("   m1 checks all ok:", all(v["phys"] == 10 and v["commit"] == APPROVED_HEAD and v["seeds_ok"] and v["zip_sha_ok"]
                                       and v["run_seed"] == RUNS[r] for v in o["m1_checks"].values()),
          "m3", o["m3_phys"], o["m3_commit"] == APPROVED_HEAD)
print("A", A, "of", nA, "p", round(p, 4), "R", R, "U", U, "P", P, "G+", G, line)
