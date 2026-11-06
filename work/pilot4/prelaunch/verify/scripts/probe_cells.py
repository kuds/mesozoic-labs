"""The probe on the smoke's judged cells: pilot 3's labels re-derived from P3 L537-545's text (this file's own code),
the leg length L, the plant digests and the env drift, read from probe.json and binding.json.

Usage: python -I probe_cells.py CELL_DIR [CELL_DIR ...]
"""
import json
import math
import sys
from collections import Counter
from pathlib import Path


def num(r, k):
    try:
        return float(r.get(k))
    except (TypeError, ValueError):
        return math.nan


def label(r):
    if not (num(r, "length") >= 1000):
        return "FALL"
    if num(r, "floor_touchdowns_per_s_all_feet") >= 0.5 and num(r, "floor_simultaneous_fraction") >= 0.5:
        return "HOP"
    if not (num(r, "speed_displacement_mps") >= 0.10):
        return "STILL"
    ok = (num(r, "floor_alternation_index") >= 0.6 and num(r, "floor_lead_swaps_per_stride") >= 1.0
          and num(r, "floor_flight_fraction") <= 0.2 and 0.35 <= num(r, "jointlag_frac_hip_pitch") <= 0.65
          and num(r, "floor_stride_frequency_hz") <= 5.0 and num(r, "floor_duty_L") <= 0.80
          and num(r, "floor_duty_R") <= 0.80)
    return "WALK" if ok else "OTHER"


def find(d, pred, path=""):
    out = []
    if isinstance(d, dict):
        for k, v in d.items():
            q = f"{path}.{k}" if path else k
            if pred(k, v):
                out.append((q, v))
            out += find(v, pred, q)
    return out


for c in sys.argv[1:]:
    p = json.load(open(Path(c) / "probe.json"))
    b = json.load(open(Path(c) / "binding.json"))
    rows = p["episodes"]
    labs = Counter(label(r) for r in rows)
    walks = labs.get("WALK", 0) >= 30 and labs.get("FALL", 0) <= 4
    L = p["meta"]["robot"].get("leg_length_m")
    plant = find(p["meta"], lambda k, v: isinstance(v, (str, int)) and ("physics" in k or "interface" in k))
    drift = find(p["meta"], lambda k, v: "drift" in k)
    print(Path(c).parent.parent.name + "/" + Path(c).parent.name + "/" + Path(c).name, "n", len(rows), dict(labs),
          "WALKS" if walks else "not WALKS", "L", L, "| lengths", dict(Counter(int(num(r, "length")) for r in rows)))
    print("   plant (probe meta):", plant[:6])
    print("   drift:", drift[:3])
    print("   binding:", {k: b.get(k) for k in ("validation_only", "expected_budget", "cadence_checked",
                                               "text_anchors_checked", "shaping_inert", "resume_step", "plant")})
