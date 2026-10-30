"""Summarise sweep.json (sweep.py, test 2 observation-only).  Read only; plain numpy.

python -B summarize_sweep.py sweep.json > sweep.txt

Groups: start_s42, start_s80 (one checkpoint each), s42 ends and s80 ends (12 each: median [range] over policies,
count of policies with a positive / negative value).  Everything is the network's response to edited observations at
the policy's own settled states (120 per policy); nothing here is a closed-loop or training result.
"""

from __future__ import annotations

import json
import sys

import numpy as np

PLANT = 5.183
RAIL_DEG = 5.2


def grp(w):
    return w if w.startswith("start") else w[:3]


def cell(vals):
    v = np.asarray(vals, dtype=float)
    if len(v) == 1:
        return f"{v[0]:+.4f}"
    return f"{np.median(v):+.4f} [{v.min():+.4f},{v.max():+.4f}] +{int((v > 0).sum())}/-{int((v < 0).sum())}"


def main() -> None:
    r = json.load(open(sys.argv[1]))
    P = r["policies"]
    G = ("start_s42", "s42", "start_s80", "s80")
    whos = {g: [w for w in P if grp(w) == g] for g in G}
    G = tuple(g for g in G if whos[g])
    typ = {k: np.asarray(v) for k, v in r["typical"].items()}
    names = r["obs_names"]
    gap = typ["s42"] - typ["s80"]
    sp = r["seen_pitch_lineage_median_deg"]
    print("TEST 2, OBSERVATION-ONLY SWEEPS (sweep.py -> sweep.json).  Network responses to edited observations at each")
    print("policy's own settled states (120 per policy: steps 500..1000 by 100, seeds 3042-3061).  cmd14 = the raw Gaussian")
    print("mean of tail_3_pitch_act.  Cells: single checkpoint value, or median [range] +pos/-neg over 12 policies.")
    print(f"Lineage medians of the observed pelvis pitch: s42 ends {sp['s42']:+.3f} deg, s80 ends {sp['s80']:+.3f} deg"
          f" (gap {sp['s42'] - sp['s80']:+.3f} deg).")
    print()
    print("0. SETTLED STATES")
    for k in ("n_states", "seen_pitch_deg_median", "tail3_deg_median", "cmd14_at_states_median"):
        print(f"   {k:26s} " + " | ".join(f"{g}: {cell([P[w][k] for w in whos[g]])}" for g in G))
    print()
    print("1. THE TAIL LOOP ITSELF (verify_loop.py's sweep: tail_3 angle 0..6 deg, everything else fixed)")
    L = "tail_3_angle(per deg) [THE LOOP]"
    for k in ("cmd14_slope", "gain", "offset_c0", "slope_pos_share"):
        print(f"   {k:26s} " + " | ".join(f"{g}: {cell([P[w]['B_families'][L][k] for w in whos[g]])}" for g in G))
    print("   contribution of the loop at the rail = slope x 5.2 deg:  "
          + " | ".join(f"{g}: {cell([P[w]['B_families'][L]['cmd14_slope'] * RAIL_DEG for w in whos[g]])}" for g in G))
    print()
    print("2. POSTURE FAMILIES: d cmd14 per unit (unit in the name); also d head_pitch and d neck_pitch command")
    fams = [k for k in P[next(iter(P))]["B_families"] if k != L]
    for f in fams:
        print(f"   {f}")
        for out in ("cmd14_slope", "head_cmd_slope", "neck_cmd_slope"):
            print(f"      {out:16s} " + " | ".join(f"{g}: {cell([P[w]['B_families'][f][out] for w in whos[g]])}" for g in G))
    pf = "pitch_illusion_all(per deg nose-up)"
    print("   pitch pathway against the loop:")
    for g in G:
        ratio = [P[w]["B_families"][pf]["cmd14_slope"] / P[w]["B_families"][L]["cmd14_slope"] for w in whos[g]]
        contrib = [P[w]["B_families"][pf]["cmd14_slope"] * (sp["s80"] - sp["s42"]) for w in whos[g]]
        print(f"     {g:9s} d cmd14 / d pitch (nose-up) over d cmd14 / d tail deg = deg of tail angle equivalent to 1 deg"
              f" nose-up: {cell(ratio)};  cmd14 change for the lineage gap ({sp['s80'] - sp['s42']:+.2f} deg nose-up):"
              f" {cell(contrib)}")
    print()
    print("3. THE LOOP UNDER THE PITCH ILLUSION (offset at tail 0, gain, implied fixed point c0/(1-G); railed if >= 1)")
    for p in ("-2.2", "-1.1", "+0.0", "+1.1", "+2.2"):
        for k in ("offset_c0", "gain", "fixed_point_cmd"):
            print(f"   seen pitch {p} deg {k:16s} " + " | ".join(
                f"{g}: {cell([min(P[w]['C_loop_vs_pitch'][p][k], 9.99) for w in whos[g]])}" for g in G))
        print(f"   seen pitch {p} deg railed_pred      " + " | ".join(
            f"{g}: {sum(P[w]['C_loop_vs_pitch'][p]['railed_pred'] for w in whos[g])}/{len(whos[g])}" for g in G))
    for g in G:
        dg = [P[w]["C_loop_vs_pitch"]["-1.1"]["gain"] - P[w]["C_loop_vs_pitch"]["+1.1"]["gain"] for w in whos[g]]
        do = [P[w]["C_loop_vs_pitch"]["-1.1"]["offset_c0"] - P[w]["C_loop_vs_pitch"]["+1.1"]["offset_c0"] for w in whos[g]]
        print(f"   {g:9s} seen 1.1 deg nose-down minus 1.1 deg nose-up (2.2 deg): d gain {cell(dg)}; d offset {cell(do)}")
    print()
    print("4. TRANSPLANT (each policy's settled states with an input group replaced by the OTHER lineage's typical values;")
    print("   control: by its OWN lineage's typical values).  Change against no transplant.")
    ks = [k for k in P[next(iter(P))]["D_transplant"] if k.startswith("other:")]
    for k in ks:
        g2 = k.split(":", 1)[1]
        for meas in ("offset_c0", "gain", "cmd14_own_tail", "fixed_point_cmd"):
            line = []
            for g in G:
                d_o = [min(P[w]["D_transplant"][k][meas], 9.99) - min(P[w]["D_transplant"]["none"][meas], 9.99) for w in whos[g]]
                d_s = [min(P[w]["D_transplant"]["own:" + g2][meas], 9.99) - min(P[w]["D_transplant"]["none"][meas], 9.99) for w in whos[g]]
                line.append(f"{g}: other {cell(d_o)} / own {cell(d_s)}")
            print(f"   {g2:34s} d {meas:15s} " + " | ".join(line))
        line = []
        for g in G:
            zs = [P[w]["D_transplant"][k].get("shift_max_z", float("nan")) for w in whos[g]]
            zi = sorted({P[w]["D_transplant"][k].get("shift_max_z_input", "?") for w in whos[g]})
            line.append(f"{g}: {cell(zs)} ({','.join(zi)})")
        print(f"   {g2:34s} size of the edit, max |shift| in the recipient's normaliser units: " + " | ".join(line))
        line = []
        for g in G:
            n0 = sum(P[w]["D_transplant"]["none"]["railed_pred"] for w in whos[g])
            n1 = sum(P[w]["D_transplant"][k]["railed_pred"] for w in whos[g])
            n2 = sum(P[w]["D_transplant"]["own:" + g2]["railed_pred"] for w in whos[g])
            line.append(f"{g}: {n0}->{n1} (own {n2})/{len(whos[g])}")
        print(f"   {g2:34s} railed by the linear fixed point: " + " | ".join(line))
    print()
    print("5. EVERY INPUT (5-point sweep +-0.5 normaliser scale).  slope x (s42 typical - s80 typical) = the first-order cmd14")
    print("   change the lineage difference in that one input would make.  Rows sorted by |s42 ends' gap contribution|;")
    print("   inputs with |median| < 0.005 in every group are omitted.")
    rows = []
    for i, n in enumerate(names):
        c = {g: [P[w]["A_per_input"][n]["slope_raw"] * gap[i] for w in whos[g]] for g in G}
        rows.append((abs(np.median(c[G[-1] if "s42" not in G else "s42"])), n, i, c))
    rows.sort(reverse=True)
    tot = {g: np.zeros(len(whos[g])) for g in G}
    for _, n, i, c in rows:
        for g in G:
            tot[g] += np.asarray(c[g]) * (i != 6)
        if max(abs(np.median(c[g])) for g in G) < 0.005:
            continue
        print(f"   {n:18s} gap {gap[i]:+.4g}  " + " | ".join(f"{g}: {cell(c[g])}" for g in G))
    print("   SUM over all inputs except tail_3's angle (first order):  " + " | ".join(f"{g}: {cell(tot[g])}" for g in G))
    print()
    print("6. ACROSS EACH POLICY'S OWN 120 SETTLED STATES: Spearman of observed pitch against the cmd14 offset at tail 0")
    print("   (states that happen to be more nose-up -> higher offset if positive); correlational")
    for g in G:
        print(f"   {g:9s} rho {cell([P[w]['E_spearman_pitch_vs_offset'] for w in whos[g]])};"
              f" pitch SD over states {cell([P[w]['E_pitch_sd_deg'] for w in whos[g]])} deg")


if __name__ == "__main__":
    main()
