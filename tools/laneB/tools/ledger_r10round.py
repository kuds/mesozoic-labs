"""T. rex r10 round: the ledger tool (criteria lean v5 M5, M6, M8; D-1).  Report only; never trains.

    PY -I -B ledger_r10round.py LEDGER.csv [--E STEPS] [--json OUT.json] [--probe policy_probe.npz] [--tel]
                                          [--r9-summary]

M5 on ONE eval_episodes.csv, rows as they stand (a COMPLETE run's whole ledger; a LOST run's as it stands for
PRESENCE and D-16):
  (a) the ANY count per evaluation: episodes whose ANY input (THIS12 lines 351-358) is over the bar;
  (b) the TAIL count: saturated_actuator_fraction > 0.10 with saturated_actuator a tail actuator (11-14), by actuator
      and sign (saturated_actuator_positive_share >= 0.5 = positive);
  (c) the TAIL-UNRESOLVED count: over the bar, a non-tail actuator named, ties >= 1 or actuators_saturated >= 2.
Rows are grouped by their timesteps on the grid 131,072 ... 11,927,552 (never by the evaluation column); a row off
the grid is printed and not read.  A row with a missing input flags nothing (a measured ANY input stays as
measured); every missing input is printed with its reason.  ANY-FLAGGED / TAIL-FLAGGED: at least 5 rows; ANY ALARM /
TAIL ALARM: the second of two consecutive flagged evaluations; NEEDS RESOLUTION as CLASSES defines it.  TEL's
rail_actuators column is printed beside (b) and (c), report only.
--E: the run's end step (C6 then also checks every evaluation of the grid up to E: 10 rows, episodes 0-9 once, every
     ANY and TAIL input measured, no row off the grid or beyond E).
--probe: M6, policy_probe.npz's per-rollout tail (11-14) and hip-roll (4, 8) shares and means (report only).
--tel: M8, TEL's columns per evaluation (mean over full-horizon episodes; a column missing is printed as absent).
--r9-summary: the per-ledger summary DRAFT verify/rail_check2.json records (V2's reproduction of the r9 ledgers).
Exit codes: 0 read; 2 input error.
"""
import os
import sys

os.environ["PYTHONDONTWRITEBYTECODE"] = "1"
sys.dont_write_bytecode = True

import argparse  # noqa: E402
import importlib.util  # noqa: E402
import json  # noqa: E402
from pathlib import Path  # noqa: E402

TOOLS = Path(__file__).resolve().parent


def _load(name):
    spec = importlib.util.spec_from_file_location(name, TOOLS / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


C = sys.modules.get("r10round_common") or _load("r10round_common")


# ------------------------------------------------------------------------------------------------- C6
def c6_check(header, rows, E):
    """C6: every evaluation of the grid up to E present with exactly 10 rows, episode indices 0-9 once; every row's ANY
    and TAIL inputs measured (THIS12 lines 351-364).  Rows off the grid or beyond E fail it (printed)."""
    res = C.m5(header, rows)
    grid = [s for s in C.GRID_STEPS if s <= E]
    fails = []
    for i, ts in res["offgrid"]:
        fails.append(f"row {i + 2} timesteps {ts!r} is not on the grid")
    for s in sorted(res["evals"]):
        if s > E:
            fails.append(f"evaluation {s} beyond E = {E}")
    for s in grid:
        e = res["evals"].get(s)
        if e is None:
            fails.append(f"evaluation {s} absent")
            continue
        if e["n"] != 10 or sorted(x for x in e["episodes"] if x is not None) != list(range(10)) or None in e["episodes"]:
            fails.append(f"evaluation {s}: {e['n']} rows, episodes {sorted(map(str, e['episodes']))}")
        for ep, why in e["any_missing"]:
            fails.append(f"evaluation {s} episode {ep}: ANY input missing ({why})")
        for ep, why in e["tail_missing"]:
            if (ep, why) not in e["any_missing"]:
                fails.append(f"evaluation {s} episode {ep}: TAIL input missing ({why})")
    return dict(passed=not fails, failures=fails, n_grid=len(grid), m5=res)


# ------------------------------------------------------------------------------------------------- D-1 and r9 summary
def d1_summary(res) -> dict:
    ev = res["evals"]
    steps = sorted(ev)
    first_tail_sat = next((s for s in steps if ev[s]["tail"] > 0), None)
    spells, cur = [], []
    for s in steps:
        if ev[s]["tail_flagged"] and (not cur or s - cur[-1] == C.GRID):
            cur.append(s)
        elif ev[s]["tail_flagged"]:
            spells.append((cur[0], cur[-1], len(cur)))
            cur = [s]
        else:
            if cur:
                spells.append((cur[0], cur[-1], len(cur)))
            cur = []
    if cur:
        spells.append((cur[0], cur[-1], len(cur)))
    windows = ((0, 4194304), (4194304, 8388608), (8388608, C.FINAL + 1))
    shares = {}
    for lo, hi in windows:
        n = sum(ev[s]["n"] for s in steps if lo <= s < hi)
        t = sum(ev[s]["tail"] for s in steps if lo <= s < hi)
        a = sum(ev[s]["any"] for s in steps if lo <= s < hi)
        shares[f"[{lo},{hi})"] = dict(rows=n, tail_over=t, any_over=a, tail_share=(t / n) if n else None,
                                      any_share=(a / n) if n else None)
    per_act = {}
    for s in steps:
        for k, v in ev[s]["any_by"].items():
            per_act[k] = per_act.get(k, 0) + v
    return dict(first_tail_saturated_episode=first_tail_sat,
                first_tail_flagged_evaluation=res["tail_flagged"][0] if res["tail_flagged"] else None,
                tail_alarm_steps=res["tail_alarms"], any_alarm_steps=res["any_alarms"], tail_flagged_spells=spells,
                window_shares=shares, any_over_by_actuator=per_act, any_alarm_count=len(res["any_alarms"]))


def r9_summary(res) -> dict:
    """The fields of DRAFT verify/rail_check2.json from M5 path (a) counts."""
    ev = res["evals"]
    per = [(s, ev[s]["n"], ev[s]["any"]) for s in sorted(ev)]
    flagged = [s for s, _n, a in per if a >= 5]
    alarm = res["any_alarms"][0] if res["any_alarms"] else None
    anysat = [s for s, _n, a in per if a >= 1]
    return dict(n_evals=len(per), n_eps=sum(n for _s, n, _a in per), sat_eps=sum(a for *_x, a in per),
                first_sat=anysat[0] if anysat else None, first_flag=flagged[0] if flagged else None, alarm=alarm,
                n_flagged=len(flagged), last_flag=flagged[-1] if flagged else None,
                last_eval=per[-1][0] if per else None, flagged_after_8M=sum(1 for s in flagged if s > 8388608),
                sat_eps_last10=sum(a for *_x, a in per[-10:]))


# ------------------------------------------------------------------------------------------------- M6, M8
M6_KEYS = ("rollout_mu_beyond_high_share", "rollout_mu_beyond_low_share", "rollout_action_clip_high_share",
           "rollout_action_clip_low_share", "rollout_mu_mean", "rollout_mu_q50", "rollout_mu_q95")
M6_ACTUATORS = (11, 12, 13, 14, 4, 8)


def m6(probe_path) -> dict:
    """M6: policy_probe.npz per rollout for actuators 11-14 and 4, 8 (allow_pickle False).  Report only."""
    import numpy as np
    out = {"file": str(probe_path), "sha256": C.sha256_file(probe_path), "keys": {}}
    with np.load(probe_path, allow_pickle=False) as z:
        names = list(z.files)
        out["arrays"] = names
        steps = z["rollout_timesteps"].tolist() if "rollout_timesteps" in names else None
        out["rollout_timesteps"] = steps
        for k in M6_KEYS:
            if k not in names:
                out["keys"][k] = "absent"
                continue
            a = z[k]
            if a.ndim != 2 or a.shape[1] < 15:
                out["keys"][k] = f"shape {a.shape}"
                continue
            out["keys"][k] = {C.ACTUATOR_NAMES[i]: [None if not np.isfinite(v) else float(v) for v in a[:, i]]
                              for i in M6_ACTUATORS}
    return out


def m8(header, rows, tel_columns) -> dict:
    """M8: TEL's columns per evaluation, the mean over full-horizon episodes (text columns: the episodes naming each
    actuator); a column missing from the ledger is printed as absent.  Report only."""
    res = {"absent": [c for c in tel_columns if c not in header], "per_evaluation": {}}
    present = [c for c in tel_columns if c in header]
    by = {}
    for r in rows:
        s = C.grid_step(r.get("timesteps"))
        if s is not None:
            by.setdefault(s, []).append(r)
    for s in sorted(by):
        full = [r for r in by[s] if (C.num(r.get("length")) or 0) >= C.FULL_HORIZON]
        d = {"full_horizon_episodes": len(full)}
        for c in present:
            if c in ("rail_actuators", "parked_actuators"):
                cnt = {}
                for r in full:
                    for x in (r.get(c) or "").split(";"):
                        x = x.strip()
                        if x:
                            cnt[x] = cnt.get(x, 0) + 1
                d[c] = cnt
            else:
                vals = [C.num(r.get(c)) for r in full]
                vals = [v for v in vals if v is not None]
                d[c] = (sum(vals) / len(vals)) if vals else None
        res["per_evaluation"][s] = d
    return res


# ------------------------------------------------------------------------------------------------- printing
def render(path, header, rows, res, c6=None) -> str:
    L = [C.VERSION, f"LEDGER {path}  rows {len(rows)}  columns {len(header)}",
         f"  #640 columns present: {[c for c in C.SAT640_COLUMNS if c in header]}; screen_v2_reasons "
         f"{'present' if 'screen_v2_reasons' in header else 'ABSENT'}"]
    if res["offgrid"]:
        L.append(f"  rows off the grid (printed, not read): {res['offgrid'][:10]}{' ...' if len(res['offgrid']) > 10 else ''}")
    L.append("  step        n  ANY  TAIL  UNRES  flags   tail by actuator/sign      ANY by actuator / reason   rail_actuators")
    for s in sorted(res["evals"]):
        e = res["evals"][s]
        fl = ("A" if e["any_flagged"] else "-") + ("T" if e["tail_flagged"] else "-")
        if e["any"] or e["tail"] or e["unres"] or e["any_missing"] or e["tail_missing"] or e["n"] != 10:
            L.append(f"  {s:>10} {e['n']:>3} {e['any']:>4} {e['tail']:>5} {e['unres']:>6}  {fl:5}  "
                     f"{json.dumps(e['tail_by']):26} {json.dumps(e['any_by'])} {json.dumps(e['reasons'])}  "
                     f"{e['rail'][:3]}")
            for ep, why in e["any_missing"]:
                L.append(f"      MISSING ANY input, episode {ep}: {why}")
            for ep, why in e["tail_missing"]:
                L.append(f"      MISSING TAIL input, episode {ep}: {why}")
    L.append(f"  evaluations read {len(res['evals'])} (rows of other evaluations: 0 over the bar, none missing)")
    L.append(f"  ANY-flagged evaluations: {res['any_flagged']}")
    L.append(f"  TAIL-flagged evaluations: {res['tail_flagged']}")
    L.append(f"  ANY ALARM steps: {res['any_alarms']}")
    L.append(f"  TAIL ALARM steps: {res['tail_alarms']}")
    L.append(f"  NEEDS RESOLUTION at: {res['needs_resolution']}")
    for s in res["tail_alarms"]:
        L.append(f"    TAIL ALARM {s}: {C.tail_alarm_detail(res, s)}")
    if c6 is not None:
        L.append(f"  C6 (E = {c6.get('E')}): {'PASS' if c6['passed'] else 'FAIL'}; grid evaluations up to E: {c6['n_grid']}")
        for f in c6["failures"][:50]:
            L.append(f"      {f}")
        if len(c6["failures"]) > 50:
            L.append(f"      ... {len(c6['failures']) - 50} more")
    return "\n".join(L)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("ledger")
    ap.add_argument("--E", type=int, default=None)
    ap.add_argument("--json")
    ap.add_argument("--probe")
    ap.add_argument("--tel", action="store_true")
    ap.add_argument("--pins", default=None)
    ap.add_argument("--r9-summary", action="store_true")
    a = ap.parse_args(argv)
    try:
        header, rows = C.read_ledger(a.ledger)
    except (OSError, C.InputError) as exc:
        print(f"{C.VERSION}\nINPUT ERROR: {exc}")
        return 2
    res = C.m5(header, rows)
    c6 = None
    if a.E is not None:
        c6 = c6_check(header, rows, a.E)
        c6["E"] = a.E
    print(render(a.ledger, header, rows, res, c6))
    out = dict(version=C.VERSION, ledger=a.ledger, sha256=C.sha256_file(a.ledger), any_alarms=res["any_alarms"],
               tail_alarms=res["tail_alarms"], needs_resolution=res["needs_resolution"],
               any_flagged=res["any_flagged"], tail_flagged=res["tail_flagged"], d1=d1_summary(res),
               per_evaluation={s: {k: e[k] for k in ("n", "any", "tail", "unres", "tail_by", "any_by", "reasons",
                                                     "any_missing", "tail_missing", "rail")}
                               for s, e in res["evals"].items()},
               offgrid=res["offgrid"])
    if c6 is not None:
        out["c6"] = dict(passed=c6["passed"], failures=c6["failures"], E=a.E)
    print("D-1:", json.dumps(out["d1"], default=str))
    if a.r9_summary:
        out["r9_summary"] = r9_summary(res)
        print("r9 summary:", json.dumps(out["r9_summary"]))
    if a.probe:
        try:
            out["m6"] = m6(a.probe)
            print("M6:", json.dumps({k: (v if isinstance(v, str) else "per actuator") for k, v in out["m6"]["keys"].items()}))
        except Exception as exc:  # noqa: BLE001 - a report-only readout: printed, nothing else
            out["m6"] = {"error": repr(exc)}
            print(f"M6: unreadable ({exc!r})")
    if a.tel:
        pins = C.load_pins(a.pins)
        cols = C.pin(pins, "tel_columns") or []
        out["m8"] = m8(header, rows, cols)
        print(f"M8: TEL columns {len(cols)}; absent {out['m8']['absent'][:10]}")
    if a.json:
        Path(a.json).write_text(json.dumps(out, indent=1, default=str) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main())
