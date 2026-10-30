"""CoM-P1 (PREREG.txt), verbatim: per checkpoint and panel, the LEAN and CREEP flags, their report-only companions,
and the replication of each flag on a held-out panel.  No simulation.

PYTHONPATH=<code_copy> python -B check.py OUT_PREFIX --m1 RUN_DIR [--heldout RUN_DIR] [--null RUN_DIR]
--null: a run of the statue only, whose seeds are split into consecutive panels of 20.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from analyze import load_tracks
from environments.shared.tests.stance_track_helpers import sign_test_p, stance_track_series

W = 5.0
ALPHA = 0.005
FLOORS = {"com_lateral_mean_m": 1.0e-3, "root_roll_mean_deg": 0.10}
REPORT_ONLY = {"left_load_excess": 0.01, "root_turn_deg": 0.10}
CREEP_SPEED = 1.0e-3  # m/s
CREEP_EPISODES = 2
REPLICATE_OF_10 = 9


def episode_values(run: Path) -> dict[str, dict[int, dict]]:
    out: dict[str, dict[int, dict]] = {}
    for line in open(run / "episodes.jsonl"):
        r = json.loads(line)
        m = r["metrics"].get(str(W))
        row = {"stood": bool(r["stood"])}
        if m:
            row.update({k: m[k] for k in FLOORS})
            row["left_load_excess"] = m["left_load_share_mean"] - 0.5
            row["root_turn_deg"] = m["root_turn_deg"]
        out.setdefault(r["who"], {})[int(r["seed"])] = row
    for who in out:
        for seed, tr, _ in load_tracks(run, who):
            if len(tr.t) < 1001:
                continue
            s = stance_track_series(tr)
            t = s["t"]
            w = t >= t[0] + W - 1e-9
            tw = t[w] - t[w].mean()
            feet = s["foot_xy"][w].mean(axis=1)
            slope = np.linalg.lstsq(np.vstack([tw, np.ones_like(tw)]).T, feet, rcond=None)[0][0]
            fwd = s["forward"][w].mean(axis=0)
            fwd /= np.linalg.norm(fwd)
            out[who][seed]["creep_mps"] = float(np.linalg.norm(slope))
            out[who][seed]["creep_fore_aft_mps"] = float(slope @ fwd)
    return out


def sign_flag(values: np.ndarray, floor: float) -> dict:
    pos, neg = int((values > 0).sum()), int((values < 0).sum())
    p = sign_test_p(pos, neg)
    med = float(np.median(values))
    return {"pos": pos, "neg": neg, "p": p, "median": med, "flag": bool(p <= ALPHA and abs(med) >= floor)}


def panel(rows: dict[int, dict], seeds: list[int]) -> dict:
    eps = [rows[s] for s in seeds if s in rows]
    if len(eps) != len(seeds) or not all(e["stood"] for e in eps):
        return {"read": False, "n": len(eps), "stood": sum(e["stood"] for e in eps)}
    out: dict = {"read": True, "n": len(eps)}
    for key, floor in {**FLOORS, **REPORT_ONLY}.items():
        out[key] = sign_flag(np.array([e[key] for e in eps]), floor)
    out["LEAN"] = bool(out["com_lateral_mean_m"]["flag"] or out["root_roll_mean_deg"]["flag"])
    creeping = [s for s in seeds if rows[s]["creep_mps"] >= CREEP_SPEED]
    out["creep_episodes"] = creeping
    out["creep_max_mps"] = max(rows[s]["creep_mps"] for s in seeds)
    out["CREEP"] = len(creeping) >= CREEP_EPISODES
    fa = np.array([rows[s]["creep_fore_aft_mps"] for s in seeds])
    out["creep_direction"] = {"forward": int((fa > 0).sum()), "backward": int((fa < 0).sum()),
                              "p": sign_test_p(int((fa > 0).sum()), int((fa < 0).sum())), "median_mps": float(np.median(fa))}
    return out


def replication(m1: dict, held: dict) -> dict:
    out = {}
    for key in FLOORS:
        sign = np.sign(m1[key]["median"])
        same = held[key]["pos"] if sign > 0 else held[key]["neg"]
        out[key] = {"m1_flag": m1[key]["flag"], "held_same_sign": int(same), "held_n": held["n"],
                    "replicates": bool(same >= REPLICATE_OF_10), "held_flag_n10": held[key]["flag"]}
    return out


def line(who: str, r: dict) -> str:
    if not r["read"]:
        return f"  {who:14s} NOT READ (stood {r['stood']}/{r['n']})"
    c, ro = r["com_lateral_mean_m"], r["root_roll_mean_deg"]
    ld, tn = r["left_load_excess"], r["root_turn_deg"]
    return (f"  {who:14s} LEAN {'FLAG' if r['LEAN'] else '-   '} CREEP {'FLAG' if r['CREEP'] else '-   '} | "
            f"com {c['pos']:2d}+{c['neg']:2d}- p{c['p']:.4f} med{c['median'] * 1000:+.2f}mm{'*' if c['flag'] else ' '} "
            f"roll {ro['pos']:2d}+{ro['neg']:2d}- p{ro['p']:.4f} med{ro['median']:+.3f}deg{'*' if ro['flag'] else ' '} | "
            f"load {ld['pos']:2d}+{ld['neg']:2d}- p{ld['p']:.3f} med{ld['median']:+.4f}{'*' if ld['flag'] else ' '} "
            f"turn {tn['pos']:2d}+{tn['neg']:2d}- p{tn['p']:.3f} med{tn['median']:+.3f}{'*' if tn['flag'] else ' '} | "
            f"creep eps {len(r['creep_episodes'])} max {r['creep_max_mps'] * 1000:.3f}mm/s dir {r['creep_direction']['forward']}f"
            f"{r['creep_direction']['backward']}b p{r['creep_direction']['p']:.3f}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("out", type=Path)
    ap.add_argument("--m1", type=Path, required=True)
    ap.add_argument("--heldout", type=Path)
    ap.add_argument("--null", type=Path)
    a = ap.parse_args()
    L, J = [f"CoM-P1 (PREREG.txt): window {W} s; alpha {ALPHA}; floors {FLOORS}; creep >= {CREEP_SPEED * 1000} mm/s on "
            f">= {CREEP_EPISODES} episodes; replication >= {REPLICATE_OF_10} of 10 held-out seeds with the M1 sign"], {}
    m1 = episode_values(a.m1)
    L.append(f"M1 panel ({a.m1}), seeds 3042-3061")
    J["m1"] = {}
    for who, rows in m1.items():
        r = panel(rows, list(range(3042, 3062)))
        J["m1"][who] = r
        L.append(line(who, r))
    pols = [w for w in J["m1"] if w != "statue"]
    nl = sum(J["m1"][w]["LEAN"] for w in pols)
    nc = sum(J["m1"][w]["CREEP"] for w in pols)
    st = J["m1"].get("statue")
    L.append(f"  policies flagged on the M1 panel: LEAN {nl}/{len(pols)}, CREEP {nc}/{len(pols)}"
             + (f"; statue LEAN {st['LEAN']} CREEP {st['CREEP']}" if st else ""))
    for g, pre in (("s42 ends", "s42_"), ("s80 ends", "s80_"), ("starts", "start_")):
        ws = [w for w in pols if w.startswith(pre)]
        L.append(f"    {g}: LEAN {sum(J['m1'][w]['LEAN'] for w in ws)}/{len(ws)} (COM {sum(J['m1'][w]['com_lateral_mean_m']['flag'] for w in ws)},"
                 f" ROLL {sum(J['m1'][w]['root_roll_mean_deg']['flag'] for w in ws)}); CREEP {sum(J['m1'][w]['CREEP'] for w in ws)}/{len(ws)}")
    if a.heldout:
        held = episode_values(a.heldout)
        seeds = sorted({s for rows in held.values() for s in rows})
        L.append("")
        L.append(f"HELD-OUT panel ({a.heldout}), seeds {seeds[0]}-{seeds[-1]} (n = {len(seeds)})")
        J["held"], J["replication"] = {}, {}
        for who, rows in held.items():
            r = panel(rows, seeds)
            J["held"][who] = r
            L.append(line(who, r))
        L.append("")
        L.append("REPLICATION of each M1 key on the held-out panel (same sign as the M1 median on >= 9 of 10)")
        flagged = replicated = 0
        for who in J["m1"]:
            if who not in J["held"] or not J["held"][who]["read"] or not J["m1"][who]["read"]:
                continue
            rep = replication(J["m1"][who], J["held"][who])
            J["replication"][who] = rep
            cells = []
            for key, d in rep.items():
                cells.append(f"{key[:14]} M1{'F' if d['m1_flag'] else '-'} held {d['held_same_sign']}/{d['held_n']} same"
                             f"{' REPL' if d['replicates'] else ''}{' n10F' if d['held_flag_n10'] else ''}")
                if d["m1_flag"]:
                    flagged += 1
                    replicated += d["replicates"]
            L.append(f"  {who:14s} " + " | ".join(cells))
        L.append(f"  flagged keys on M1: {flagged}; replicated: {replicated}; not replicated (false alarms of the check): "
                 f"{flagged - replicated}")
        J["replication_summary"] = {"flagged": flagged, "replicated": replicated}
        # unflagged keys: how often the held-out panel shows the same sign on >= 9 of 10 anyway
        un = [(w, k) for w in J["replication"] for k, d in J["replication"][w].items() if not d["m1_flag"]]
        L.append(f"  unflagged keys whose held-out panel still has the M1 median's sign on >= 9 of 10: "
                 f"{sum(J['replication'][w][k]['replicates'] for w, k in un)}/{len(un)}")
    if a.null:
        nul = episode_values(a.null)["statue"]
        seeds = sorted(nul)
        panels = [seeds[i:i + 20] for i in range(0, len(seeds) - 19, 20)]
        L.append("")
        L.append(f"NULL: the statue on {len(panels)} more panels of 20 ({seeds[0]}-{seeds[-1]}), plus the M1 panel above")
        J["null"] = []
        for ps in panels:
            r = panel(nul, ps)
            J["null"].append(r)
            L.append(line(f"statue {ps[0]}", r))
        allp = J["null"] + [J["m1"]["statue"]]
        L.append(f"  statue panels flagged: LEAN {sum(r['LEAN'] for r in allp)}/{len(allp)}, CREEP {sum(r['CREEP'] for r in allp)}/{len(allp)};"
                 f" keys at p <= 0.005 regardless of the floor: "
                 f"{sum(r[k]['p'] <= ALPHA for r in allp for k in FLOORS)}/{2 * len(allp)}")
        # all 2-key values pooled: 120 statue episodes, the per-episode sign balance
        pooled = {k: np.array([nul[s][k] for s in seeds] + [m1['statue'][s][k] for s in range(3042, 3062)]) for k in FLOORS}
        for k, v in pooled.items():
            L.append(f"  pooled statue {k}: {int((v > 0).sum())}+ {int((v < 0).sum())}- of {v.size}, p {sign_test_p(int((v > 0).sum()), int((v < 0).sum())):.3f}, "
                     f"median {np.median(v):+.5f}, |value| 95th pct {np.percentile(np.abs(v), 95):.5f}")
        J["null_pooled"] = {k: {"pos": int((v > 0).sum()), "neg": int((v < 0).sum())} for k, v in pooled.items()}
    a.out.with_suffix(".txt").write_text("\n".join(L) + "\n")
    a.out.with_suffix(".json").write_text(json.dumps(J, indent=1, default=float))
    print("\n".join(L))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
