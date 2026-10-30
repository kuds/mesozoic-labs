"""Compare the verifier's own rollouts (out/*.jsonl, verify_sim.py) with the clamp runner's raw episodes, and compute
the verifier's added comparisons.  Read only outside this folder.  python -B compare.py > compare.txt"""

import json
import math
from pathlib import Path

import numpy as np

V = Path(__file__).resolve().parent / "out"
C = Path("/tmp/claude-0/-home-user-mesozoic-labs/184ee67c-2802-5235-8e52-6dc7daa72cd3/scratchpad/com_revisit/notes/clamp")
D9 = Path("/tmp/claude-0/-home-user-mesozoic-labs/184ee67c-2802-5235-8e52-6dc7daa72cd3/scratchpad/r10round/f2/d9_d10/"
          "stance6_r10LTSB_s42/d9_3670016_hold_tail_3_pitch_act_0/d9_summary.json")
KEYS = ("com_fore_aft_mean_m", "root_pitch_mean_deg", "com_height_mean_m", "com_lateral_mean_m", "root_roll_mean_deg")


def load(p, key=("who", "seed")):
    return {tuple(r[k] for k in key): r for r in map(json.loads, open(p))}


def sign_p(pos, neg):
    n = pos + neg
    k = min(pos, neg)
    return 1.0 if n == 0 else min(1.0, 2 * sum(math.comb(n, i) for i in range(k + 1)) / 2 ** n)


def check_against_runner(mine, runner, label, ret_key="return"):
    n = exact_ret = exact_met = 0
    dmax = 0.0
    for key, r in mine.items():
        ref = runner.get(key)
        if ref is None:
            continue
        n += 1
        exact_ret += r[ret_key] == ref["return"]
        dmax = max(dmax, abs(r[ret_key] - ref["return"]))
        exact_met += all(r["metrics"]["5.0"][k] == ref["metrics"]["5.0"][k] for k in KEYS)
    print(f"  {label}: {n} episodes matched to the runner's; return identical {exact_ret}/{n} (max |diff| {dmax:.3e}); "
          f"5 s CoM/pitch/roll metrics identical {exact_met}/{n}")


def main():
    print("== 1. My rollouts against the runner's raw episodes (same policy, seed, condition)")
    pairs = [("s42_unforced", "run/s42_unforced", "return"), ("s42_held0", "run/s42_held0", "return"),
             ("s80_unforced_long", "run/s80_unforced", "return_1000"), ("s80_forced1_long", "run/s80_forced1", "return_1000"),
             ("start_s42_unforced", "start/s42_unforced", "return"), ("start_s42_held0", "start/s42_held0", "return")]
    mine_all = {}
    for mine_name, runner_dir, rk in pairs:
        mine = load(V / f"{mine_name}.jsonl")
        mine_all[mine_name] = mine
        check_against_runner(mine, load(C / runner_dir / "episodes.jsonl"), mine_name, rk)
    print("== 2. Override check inside the env (every step after the override): steps where the filter state or the")
    print("      applied ctrl[14] differ from the held value's mapping; max |deviation|")
    for name in ("s42_held0", "s42_hold05", "s80_forced1_long", "start_s42_held0", "start_s42_forced1", "d9_held0"):
        rows = list(load(V / f"{name}.jsonl").values())
        print(f"  {name:18s} episodes {len(rows)}; bad ctrl steps {sum(r['bad_ctrl_steps'] for r in rows)}; bad filter "
              f"steps {sum(r['bad_filter_steps'] for r in rows)}; max dev ctrl {max(r['max_abs_ctrl_dev'] for r in rows):.1e} "
              f"filter {max(r['max_abs_filter_dev'] for r in rows):.1e}; ctrl target {rows[0]['ctrl_expect']}; "
              f"falls {sum(not r['stood'] for r in rows)}")
    print("== 3. D-9 onset checkpoint, my loader, held 0: against d9_summary.json")
    ref = {e["seed"]: e for e in json.loads(D9.read_text())["window_return_vs_deterministic_m1"]["per_episode"]}
    for (who, seed), r in sorted(load(V / "d9_held0.jsonl").items()):
        e = ref[seed]
        print(f"  seed {seed}: return {r['return']:.6f} vs D-9 {e['return']:.6f}; window {r['window_return_200_1000']:.6f} vs "
              f"{e['window_return']:.6f}; held - D-9 baseline window {r['window_return_200_1000'] - e['baseline_window_return']:+.2f}; "
              f"own tail_3 raw window mean {r['own14_mean_5_10']:.4f} (D-9 window raw {e['held_actuator_raw_window_mean']:.4f}, 2-10 s)")
    print("== 4. My paired differences on the subset (forced - unforced), per policy median over 5 seeds")
    for g, un, fo in (("s42", "s42_unforced", "s42_held0"), ("s80", "s80_unforced_long", "s80_forced1_long")):
        u, f = mine_all[un], mine_all[fo]
        names = sorted({w for w, s in u})
        print(f"  {g} ({fo})")
        defs = {
            "fore_aft mm": lambda r: 1000 * r["metrics"]["5.0"]["com_fore_aft_mean_m"],
            "pitch deg (reduction)": lambda r: r["metrics"]["5.0"]["root_pitch_mean_deg"],
            "pitch deg (own proxy)": lambda r: r["pitch_proxy_deg_5_10"],
            "height mm (reduction)": lambda r: 1000 * r["metrics"]["5.0"]["com_height_mean_m"],
            "height mm (own subtree)": lambda r: r["comz_mm_5_10"],
            "|lateral| mm": lambda r: 1000 * abs(r["metrics"]["5.0"]["com_lateral_mean_m"]),
            "head_pitch joint deg": lambda r: r["head_deg_5_10"],
            "neck_pitch joint deg": lambda r: r["neck_deg_5_10"],
            "tail_3 joint deg": lambda r: r["tail3_deg_5_10"],
            "own tail_3 cmd": lambda r: r["own14_mean_5_10"],
            "own tail_3 raw mean": lambda r: r["raw14_mean_5_10"],
            "return 0-10 s": lambda r: r.get("return_1000", r["return"]),
        }
        for label, fn in defs.items():
            per = {w: float(np.median([fn(f[(w, s)]) - fn(u[(w, s)]) for (x, s) in u if x == w])) for w in names}
            lv = {w: (float(np.median([fn(u[(w, s)]) for (x, s) in u if x == w])),
                      float(np.median([fn(f[(w, s)]) for (x, s) in f if x == w]))) for w in names}
            print(f"    {label:24s} " + "  ".join(f"{w[4:]}: {per[w]:+8.3f} ({lv[w][0]:+.3f}->{lv[w][1]:+.3f})" for w in names))
    print("== 5. s80 long runs on three other policies (my addition), |lateral| and speed by window")
    u, f = mine_all["s80_unforced_long"], mine_all["s80_forced1_long"]
    names = sorted({w for w, s in u})
    for ws in ("5.0", "10.0", "15.0"):
        for label, fn in (("|lateral| mm", lambda r, ws=ws: 1000 * abs(r["metrics"][ws]["com_lateral_mean_m"])),
                          ("speed mm/s", lambda r, ws=ws: 1000 * r["metrics"][ws]["com_speed_rms_mps"]),
                          ("|roll| deg", lambda r, ws=ws: abs(r["metrics"][ws]["root_roll_mean_deg"])),
                          ("pitch deg", lambda r, ws=ws: r["metrics"][ws]["root_pitch_mean_deg"])):
            parts = []
            for w in names:
                d = [fn(f[(w, s)]) - fn(u[(w, s)]) for (x, s) in u if x == w]
                parts.append(f"{w[4:]}: {np.median(d):+.3f} ({sum(v > 0 for v in d)}/{sum(v < 0 for v in d)}) "
                             f"lvl {np.median([fn(u[(w, s)]) for (x, s) in u if x == w]):.3f}->{np.median([fn(f[(w, s)]) for (x, s) in f if x == w]):.3f}")
            print(f"  window from {ws:4s} s {label:12s} " + "  ".join(parts))
    print("== 6. Start checkpoint s42 forced to +1 (my addition): the 'both railed' start-to-ends comparison")
    sf = load(V / "start_s42_forced1.jsonl")
    ends_u = load(C / "run/s42_unforced/episodes.jsonl")
    ends_h = load(C / "run/s42_held0/episodes.jsonl")
    st_u = load(C / "start/s42_unforced/episodes.jsonl")
    st_h = load(C / "start/s42_held0/episodes.jsonl")
    names = sorted({w for w, s in ends_u})
    for k, sc in (("com_fore_aft_mean_m", 1000), ("root_pitch_mean_deg", 1), ("com_height_mean_m", 1000)):
        lv = lambda rows: float(np.median([r["metrics"]["5.0"][k] * sc for r in rows]))  # noqa: E731
        s_f = lv(sf.values())
        s_u = lv(st_u.values())
        s_h = lv(st_h.values())
        out = []
        for label, ends, start in (("unforced/unforced", ends_u, s_u), ("held0/held0", ends_h, s_h),
                                   ("railed: ends unforced / start forced+1", ends_u, s_f)):
            per = [float(np.median([r["metrics"]["5.0"][k] * sc for (w, s), r in ends.items() if w == n])) - start for n in names]
            pos, neg = sum(v > 0 for v in per), sum(v < 0 for v in per)
            out.append(f"{label}: {np.median(per):+.3f} ({pos}/{neg}, p {sign_p(pos, neg):.4f})")
        dd = [sf[key]["metrics"]["5.0"][k] * sc - st_u[key]["metrics"]["5.0"][k] * sc for key in st_u]
        print(f"  {k:20s} start unforced {s_u:+.3f}, held0 {s_h:+.3f}, forced+1 {s_f:+.3f} (forced - unforced seeds "
              f"+{sum(v > 0 for v in dd)}/-{sum(v < 0 for v in dd)})")
        print("      ends - start: " + "; ".join(out))
    tj = np.median([r["tail3_deg_5_10"] for r in sf.values()])
    own = np.median([r["own14_mean_5_10"] for r in sf.values()])
    print(f"  start_s42 forced+1: tail_3 joint {tj:+.3f} deg, own tail_3 cmd {own:+.3f}; falls {sum(not r['stood'] for r in sf.values())}")
    print("== 7. Intermediate hold (my addition): s42 policies with tail_3 held at +0.5, own command vs tail angle")
    h5 = load(V / "s42_hold05.jsonl")
    u, f = mine_all["s42_unforced"], mine_all["s42_held0"]
    for w in sorted({x for x, s in h5}):
        row = []
        for label, arm in (("held 0", f), ("held 0.5", h5), ("free", u)):
            tj = np.median([r["tail3_deg_5_10"] for (x, s), r in arm.items() if x == w])
            oc = np.median([r["own14_mean_5_10"] for (x, s), r in arm.items() if x == w])
            rw = np.median([r["raw14_mean_5_10"] for (x, s), r in arm.items() if x == w])
            row.append(f"{label}: tail {tj:+.2f} deg own {oc:+.3f} raw {rw:+.3f}")
        print(f"  {w}: " + " | ".join(row))
    print("== 8. Rail onset in my unforced s42 and forced s80 rollouts (0.25 s bins, 0-4 s; mean over episodes)")
    for name in ("s42_unforced", "s42_held0", "s80_unforced_long", "s80_forced1_long"):
        rows = list(mine_all[name].values())
        print(f"  {name:18s} own " + " ".join(f"{v:+.2f}" for v in np.mean([r['own14_bins_025s'] for r in rows], axis=0)))
        print(f"  {'':18s} deg " + " ".join(f"{v:+.2f}" for v in np.mean([r['tail3_deg_bins_025s'] for r in rows], axis=0)))


if __name__ == "__main__":
    main()
