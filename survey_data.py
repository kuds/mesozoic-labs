"""Entropy continuation screen (ECS): design-data survey.  READ ONLY: no training, no simulation, no Drive, no GitHub.

    PY -I -B ECS/survey_data.py            (PY = S2/venv/bin/python; cwd /tmp)

Reads (never writes) the round's records:
  - F2 ledgers   ROUND/fetch/F2/<RUN_ID>/copies/01_stance/eval_episodes.csv.copy1  (M5 as the pinned lane B code reads it)
  - F2 probe     ROUND/fetch/F2/<RUN_ID>/copies/01_stance/policy_probe.npz.copy1  (allow_pickle False)
  - F2 models/   listing of the copies held; artifact_manifest.json.copy1; ROUND/fetch/manifest.csv
  - M1 panels    ROUND/panels/<RUN_ID>/<label>/m1/m1_summary.json
Writes only ECS/survey_data.json and ECS/survey_data.out (stdout copy).
The M5 rules are not re-implemented: r10round_common.py (lane B, sha256 ea2c5ea6...) is loaded by path and its own
read_ledger() and m5() are called; the file's sha256 is checked against lane B's SHA256SUMS.txt first.
"""
import sys

sys.dont_write_bytecode = True

import csv  # noqa: E402
import hashlib  # noqa: E402
import importlib.util  # noqa: E402
import json  # noqa: E402
import os  # noqa: E402
from pathlib import Path  # noqa: E402

S1 = Path("/tmp/claude-0/-home-user-mesozoic-labs/0fdf74a4-2989-56c9-a773-92101004d7de/scratchpad")
S2 = Path("/tmp/claude-0/-home-user-mesozoic-labs/184ee67c-2802-5235-8e52-6dc7daa72cd3/scratchpad")
ROUND = S1 / "foot" / "stance6"
ECS = S2 / "r10round" / "ecs"
LANEB = S2 / "r10round" / "lean" / "build" / "laneB"
COMMON = LANEB / "tools" / "r10round_common.py"
COMMON_PIN = "ea2c5ea6984b8dc00c1c066a4441a59526c515ceb111da17eaf1688a009226c5"

RUNS = {"s42": "stance6_r10LTSB_s42", "s80": "stance6_r10LTSB_s80", "s60": "stance6_r10LTSB_s60"}
GRID = 131072
FINAL = 12001280
ENT0, ENT_DECAY = 0.005, 7_000_000
SHOW = (0, 1, 2, 11, 12, 13, 14)          # neck_pitch, neck_yaw, head_pitch, tail_1_pitch, tail_1_yaw, tail_2_pitch, tail_3
HABIT = {"s42": (14, "+"), "s80": (0, "-"), "s60": (14, "+")}   # each run's habit actuator and sign (REVIEW D-1, D-6)

# candidate starts: (run, step, role)
STARTS = [
    ("s42", 3145728, "core (NEXT_STEPS Option 1)"),
    ("s80", 3145728, "core (NEXT_STEPS Option 1)"),
    ("s60", 6291456, "add-on C (NEXT_STEPS Option 1)"),
    ("s42", 3407872, "closer (RO l815-l819, REVIEW 4.5)"),
    ("s80", 3670016, "closer (RO l815-l819, REVIEW 4.5)"),
    ("s60", 6815744, "closer (RO l815-l819, REVIEW 4.5)"),
    ("s80", 4194304, "other F2-held (surveyor's addition)"),
    ("s42", 2097152, "other F2-held, earlier (NEXT_STEPS RISKS)"),
    ("s80", 2097152, "other F2-held, earlier (surveyor's addition)"),
    ("s42", 1048576, "other F2-held, earliest (NEXT_STEPS RISKS)"),
    ("s80", 1048576, "other F2-held, earliest (NEXT_STEPS RISKS)"),
    ("s60", 7340032, "other F2-held, later (surveyor's addition)"),
]
LENGTHS = (1048576, 1572864, 2097152, 2359296, 3145728)
DESIGN_LEN = {("s42", 3145728): 1572864, ("s80", 3145728): 2359296, ("s60", 6291456): 2097152,
              ("s42", 3407872): 1572864, ("s80", 3670016): 1572864, ("s60", 6815744): 1572864}

OUT_LINES = []


def out(*a):
    s = " ".join(str(x) for x in a)
    OUT_LINES.append(s)
    print(s)


def sha256_file(p):
    h = hashlib.sha256()
    with open(p, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def load_common():
    got = sha256_file(COMMON)
    if got != COMMON_PIN:
        raise SystemExit(f"r10round_common.py sha256 {got} != pin {COMMON_PIN}")
    sums = (LANEB / "SHA256SUMS.txt").read_text()
    if f"{COMMON_PIN}  tools/r10round_common.py" not in sums:
        raise SystemExit("pin not in lane B SHA256SUMS.txt")
    spec = importlib.util.spec_from_file_location("r10round_common", str(COMMON))
    mod = importlib.util.module_from_spec(spec)
    sys.modules["r10round_common"] = mod
    spec.loader.exec_module(mod)
    return mod


def ent_sched(t):
    return max(0.0, ENT0 * (1.0 - t / ENT_DECAY))


def ent_mean(a, b, n=20000):
    """Mean of the linear schedule over (a, b] (arm A as recorded)."""
    s = 0.0
    for i in range(n):
        s += ent_sched(a + (b - a) * (i + 0.5) / n)
    return s / n


def copies_dir(run):
    return ROUND / "fetch" / "F2" / RUNS[run] / "copies" / "01_stance"


def manifest_rows():
    with open(ROUND / "fetch" / "manifest.csv", newline="") as fh:
        return list(csv.DictReader(fh))


def models_listing(rows, run):
    for r in rows:
        if r["fetch"] == "F2" and r["run_id"] == RUNS[run] and r["path"] == "01_stance/models/" and r["status"] == "LISTING":
            note = r["note"]
            items = note.split("; ")[1:]
            d = {}
            for it in items:
                if ":" in it:
                    k, v = it.rsplit(":", 1)
                    d[k.strip()] = int(v)
            return d, r["utc"]
    return {}, None


def art_manifest(run):
    with open(copies_dir(run).parent / "artifact_manifest.json.copy1") as fh:
        m = json.load(fh)
    return {f["path"]: f for f in m["files"]}


def pair_names(step):
    return f"stage1_{step}_steps.zip", f"stage1_vecnormalize_{step}_steps.pkl"


def pair_status(rows, run, step):
    """Whether F2 holds the pair; the copies' sha256 against the fetch manifest's COPY rows and artifact_manifest."""
    am = art_manifest(run)
    listing, lutc = models_listing(rows, run)
    res = {"held": True, "files": []}
    for name in pair_names(step):
        path = f"01_stance/models/{name}"
        cp = copies_dir(run) / "models" / f"{name}.copy1"
        ent = am.get(path)
        entry_sha = ent["sha256"].split(":", 1)[1] if ent else None
        rec = {"path": path, "listed_size": listing.get(name), "artifact_manifest_sha256": entry_sha,
               "artifact_manifest_size": ent["size_bytes"] if ent else None}
        if cp.exists():
            got = sha256_file(cp)
            fm = [r for r in rows if r["fetch"] == "F2" and r["run_id"] == RUNS[run] and r["path"] == path
                  and r["status"] == "COPY"]
            rec.update(copy=str(cp), copy_sha256=got, copy_size=os.path.getsize(cp),
                       equals_artifact_manifest=(got == entry_sha),
                       fetch_manifest_copy_rows=len(fm),
                       equals_fetch_manifest=all(r["sha256"] == got for r in fm) and bool(fm))
        else:
            res["held"] = False
            rec.update(copy=None)
        res["files"].append(rec)
    return res


def m1_at(run, step):
    p = ROUND / "panels" / RUNS[run] / str(step) / "m1" / "m1_summary.json"
    if not p.exists():
        return None
    d = json.load(open(p))
    act = HABIT[run][0]
    eps = d.get("episodes", [])
    names = [None] * 15
    if eps:
        for nm, v in eps[0]["actuators"].items():
            names[v["index"]] = nm
    share99 = {}
    for i in SHOW:
        nm = names[i]
        vals = [e["actuators"][nm]["share_ge_0p99"] for e in eps if nm in e["actuators"]]
        share99[i] = (sum(vals) / len(vals)) if vals else None
    return dict(file=str(p), sha256=sha256_file(p), checkpoint_sha256=d["checkpoint"]["sha256"],
                vecnorm_sha256=d["checkpoint"]["vecnorm_sha256"], n=d["n_episodes"],
                full_horizon=d["full_horizon_episodes"],
                cmd={i: d["cmd_all_actuators"][str(i)] for i in SHOW},
                cmd_a_tail=d["cmd_a"], share99_mean=share99,
                tail_flags=len(d["tail_flags"]), any_over_bar_episodes=len(d["any_actuator_over_bar_episodes"]),
                any_over_bar_actuators=sorted({a for x in d["any_actuator_over_bar_episodes"] for a in x["actuators"]}),
                fall_flags=len(d["fall_flags"]), max_abs_command=d["max_abs_command_any_step"],
                habit_actuator=act)


def ledger_eval_commands(header, rows, step, C):
    cols = {i: f"{C.ACTUATOR_NAMES[i]}_command_mean" for i in SHOW}
    rs = [r for r in rows if C.grid_step(r.get("timesteps")) == step]
    out_ = {}
    for i, c in cols.items():
        v = [C.num(r.get(c)) for r in rs]
        v = [x for x in v if x is not None]
        out_[i] = (sum(v) / len(v)) if v else None
    return out_, len(rs)


def probe_at(run, step):
    import numpy as np
    p = copies_dir(run) / "policy_probe.npz.copy1"
    with np.load(p, allow_pickle=False) as z:
        ut = z["update_timesteps"]
        idx = int(np.searchsorted(ut, step))
        if idx >= len(ut) or int(ut[idx]) != step:
            return None
        std = z["std_per_actuator"][idx]
        mu = z["rollout_mu_mean"][idx]
        muhi = z["rollout_mu_beyond_high_share"][idx]
        mulo = z["rollout_mu_beyond_low_share"][idx]
        return dict(update_timesteps=int(ut[idx]), ent_coef=float(z["ent_coef"][idx]),
                    learning_rate=float(z["learning_rate"][idx]),
                    std={i: float(std[i]) for i in SHOW}, rollout_mu_mean={i: float(mu[i]) for i in SHOW},
                    mu_beyond_high={i: float(muhi[i]) for i in SHOW}, mu_beyond_low={i: float(mulo[i]) for i in SHOW})


def first_after(seq, step, end=None):
    for s in seq:
        if s > step and (end is None or s <= end):
            return s
    return None


def window_outcomes(res, run, start, L, C, header, rows):
    end = start + L
    ev = res["evals"]
    steps = [s for s in sorted(ev) if start < s <= end]
    anyf = [s for s in steps if ev[s]["any_flagged"]]
    tailf = [s for s in steps if ev[s]["tail_flagged"]]
    any_al = [s for s in steps if ev[s]["any_flagged"] and (s - GRID) in ev and (s - GRID) > start
              and ev[s - GRID]["any_flagged"]]
    tail_al = [s for s in steps if ev[s]["tail_flagged"] and (s - GRID) in ev and (s - GRID) > start
               and ev[s - GRID]["tail_flagged"]]
    over = sum(ev[s]["any"] for s in steps)
    first_over = next((s for s in steps if ev[s]["any"] > 0), None)
    act, sign = HABIT[run]
    endcmd, nrows = ledger_eval_commands(header, rows, end, C)
    m1e = m1_at(run, end)
    return dict(start=start, length=L, end=end, n_evals=len(steps), any_flagged=len(anyf),
                first_any_flagged=anyf[0] if anyf else None, tail_flagged=len(tailf),
                first_tail_flagged=tailf[0] if tailf else None, any_alarms=len(any_al),
                first_any_alarm=any_al[0] if any_al else None, tail_alarms=len(tail_al),
                first_tail_alarm=tail_al[0] if tail_al else None, over_bar_episodes=over,
                first_over_bar_episode=first_over,
                end_ledger_habit_cmd=endcmd.get(act), end_ledger_tail3_cmd=endcmd.get(14), end_ledger_rows=nrows,
                end_m1_habit_cmd=(m1e["cmd"][act] if m1e else None),
                end_m1_tail3_cmd=(m1e["cmd"][14] if m1e else None),
                end_m1_max_abs_cmd=(max(abs(v) for v in m1e["cmd"].values()) if m1e else None),
                end_m1_any_over_bar=(m1e["any_over_bar_episodes"] if m1e else None))


def main():
    C = load_common()
    rows_man = manifest_rows()
    data = {"utc_note": "times from date -u in the caller", "common_sha256": COMMON_PIN, "runs": {}, "starts": [],
            "windows": [], "events": {}}
    ledgers = {}
    for run in RUNS:
        lp = copies_dir(run) / "eval_episodes.csv.copy1"
        header, rows = C.read_ledger(lp)
        res = C.m5(header, rows)
        ledgers[run] = (header, rows, res)
        ev = res["evals"]
        overs = [s for s in sorted(ev) if ev[s]["any"] > 0]
        data["events"][run] = dict(
            ledger=str(lp), ledger_sha256=sha256_file(lp), n_evals=len(ev),
            over_bar_steps=overs, any_flagged=res["any_flagged"], tail_flagged=res["tail_flagged"],
            any_alarms=res["any_alarms"], tail_alarms=res["tail_alarms"],
            any_count_by_step={s: ev[s]["any"] for s in sorted(ev) if ev[s]["any"]})
        out(f"== {run}: ledger {lp.name} sha256 {data['events'][run]['ledger_sha256'][:16]}..., {len(ev)} evaluations; "
            f"over-bar steps {len(overs)}; ANY-flagged {len(res['any_flagged'])}; TAIL-flagged {len(res['tail_flagged'])}; "
            f"ANY ALARMs {len(res['any_alarms'])}; TAIL ALARMs {len(res['tail_alarms'])}")
    out("")
    out("PER START")
    for run, step, role in STARTS:
        header, rows, res = ledgers[run]
        ev = res["evals"]
        ps = pair_status(rows_man, run, step)
        m1 = m1_at(run, step)
        lc, nr = ledger_eval_commands(header, rows, step, C)
        pr = probe_at(run, step)
        overs = data["events"][run]["over_bar_steps"]
        nxt = dict(
            over_bar_episode=first_after(overs, step), any_flagged=first_after(res["any_flagged"], step),
            tail_flagged=first_after(res["tail_flagged"], step), any_alarm=first_after(res["any_alarms"], step),
            tail_alarm=first_after(res["tail_alarms"], step))
        before = dict(over_bar=[s for s in overs if s <= step], any_flagged=[s for s in res["any_flagged"] if s <= step],
                      any_alarms=[s for s in res["any_alarms"] if s <= step])
        st = dict(run=run, step=step, role=role, pair=ps, m1=m1, ledger_eval_cmd=lc, ledger_eval_rows=nr,
                  ledger_eval_any=ev.get(step, {}).get("any"), ledger_eval_tail=ev.get(step, {}).get("tail"),
                  probe=pr, ent_coef_schedule=ent_sched(step), next=nxt, before=before)
        data["starts"].append(st)
        out(f"-- {run} {step:,} [{role}]  F2 holds pair: {ps['held']}")
        for f in ps["files"]:
            if f.get("copy"):
                out(f"     {f['path']}: copy sha256 {f['copy_sha256'][:16]}... = artifact_manifest {f['equals_artifact_manifest']}"
                    f", = fetch-manifest COPY rows ({f['fetch_manifest_copy_rows']}) {f['equals_fetch_manifest']}; size {f['copy_size']}")
            else:
                out(f"     {f['path']}: NOT HELD; listed size {f['listed_size']}; artifact_manifest sha256 "
                    f"{(f['artifact_manifest_sha256'] or '')[:16]}... size {f['artifact_manifest_size']}")
        if m1:
            c = m1["cmd"]
            out("     M1 (20 det. episodes, seeds 3042-3061): CMD neck_pitch %+.3f neck_yaw %+.3f head_pitch %+.3f | tail_1p %+.3f "
                "tail_1y %+.3f tail_2p %+.3f tail_3 %+.3f; TAIL FLAGS %d; any-actuator over-bar episodes %d %s; falls %d; "
                "max |cmd| any step %.3f" % (c[0], c[1], c[2], c[11], c[12], c[13], c[14], m1["tail_flags"],
                                             m1["any_over_bar_episodes"], m1["any_over_bar_actuators"], m1["fall_flags"],
                                             m1["max_abs_command"]))
            out("        share>=0.99 (mean over episodes): neck %.3f head %.3f tail_3 %.3f" %
                (m1["share99_mean"][0], m1["share99_mean"][2], m1["share99_mean"][14]))
        else:
            out("     M1: no panel at this label")
        cells = [("%+.3f" % lc[i]) if lc[i] is not None else "-" for i in (0, 2, 11, 13, 14)]
        out("     ledger evaluation at this step (10 det. episodes; TEL *_command_mean): neck %s head %s tail_1p %s "
            "tail_2p %s tail_3 %s; over-bar %s, tail %s" % (*cells, st["ledger_eval_any"], st["ledger_eval_tail"]))
        if pr:
            out("     probe at update %d: ent_coef %.6f (schedule %.6f); lr %.3g; std neck %.4f head %.4f tail_3 %.4f; "
                "rollout mu neck %+.3f tail_3 %+.3f; mu-beyond tail_3 hi %.4f neck lo %.4f" % (
                    pr["update_timesteps"], pr["ent_coef"], ent_sched(step), pr["learning_rate"], pr["std"][0],
                    pr["std"][2], pr["std"][14], pr["rollout_mu_mean"][0], pr["rollout_mu_mean"][14],
                    pr["mu_beyond_high"][14], pr["mu_beyond_low"][0]))
        out("     next after start: over-bar episode %s; ANY-flagged %s; TAIL-flagged %s; ANY ALARM %s; TAIL ALARM %s" % tuple(
            (f"{v:,} (+{v - step:,})" if v else "none") for v in (nxt["over_bar_episode"], nxt["any_flagged"],
                                                                   nxt["tail_flagged"], nxt["any_alarm"],
                                                                   nxt["tail_alarm"])))
        out("     at or before start: over-bar episodes at %d evaluations (last %s); ANY-flagged %d; ANY ALARMs %d" % (
            len(before["over_bar"]), f"{before['over_bar'][-1]:,}" if before["over_bar"] else "none",
            len(before["any_flagged"]), len(before["any_alarms"])))
    out("")
    out("WINDOWS IN THE ORIGINAL RUN (evaluations at start + k x 131,072, k >= 1; an alarm needs both evaluations inside)")
    for run, step, role in STARTS:
        header, rows, res = ledgers[run]
        for L in LENGTHS:
            if step + L > FINAL:
                continue
            w = window_outcomes(res, run, step, L, C, header, rows)
            w.update(run=run, role=role, design=(DESIGN_LEN.get((run, step)) == L))
            data["windows"].append(w)
            act, sign = HABIT[run]
            endc = w["end_m1_habit_cmd"] if w["end_m1_habit_cmd"] is not None else w["end_ledger_habit_cmd"]
            src = "M1" if w["end_m1_habit_cmd"] is not None else "ledger"
            out("  %s %-10s L %-9s -> %-10s evals %2d | ANY-flagged %2d (first %s) | TAIL-flagged %2d | ANY ALARM %2d (first %s)"
                " | TAIL ALARM %2d | over-bar eps %3d (first %s) | end habit-actuator %s CMD %s (%s)%s" % (
                    run, f"{step:,}", f"{L:,}", f"{w['end']:,}", w["n_evals"], w["any_flagged"],
                    f"{w['first_any_flagged']:,}" if w["first_any_flagged"] else "-", w["tail_flagged"], w["any_alarms"],
                    f"{w['first_any_alarm']:,}" if w["first_any_alarm"] else "-", w["tail_alarms"],
                    w["over_bar_episodes"], f"{w['first_over_bar_episode']:,}" if w["first_over_bar_episode"] else "-",
                    f"{act}{sign}", ("%+.3f" % endc) if endc is not None else "-", src,
                    "  <- design length" if w["design"] else ""))
    # schedule arithmetic per design window
    out("")
    out("ARM A (RECORDED SCHEDULE) OVER EACH DESIGN WINDOW [computed: 0.005 x (1 - t / 7,000,000), floor 0]")
    for (run, step), L in DESIGN_LEN.items():
        a, b = step, step + L
        zero_frac = max(0, b - max(a, ENT_DECAY)) / L
        out("  %s %s -> %s: ent_coef at start %.6f, at end %.6f, mean over window %.6f; share of window at 0: %.3f" % (
            run, f"{a:,}", f"{b:,}", ent_sched(a), ent_sched(b), ent_mean(a, b), zero_frac))
        data.setdefault("schedule", []).append(dict(run=run, start=a, end=b, at_start=ent_sched(a), at_end=ent_sched(b),
                                                    mean=ent_mean(a, b), share_at_zero=zero_frac))
    with open(ECS / "survey_data.json", "w") as fh:
        json.dump(data, fh, indent=1, default=str)
    with open(ECS / "survey_data.out", "w") as fh:
        fh.write("\n".join(OUT_LINES) + "\n")


if __name__ == "__main__":
    main()
