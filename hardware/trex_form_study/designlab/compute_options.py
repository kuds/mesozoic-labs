"""Onboard compute alternatives: what each architecture does to power, runtime, mass and cost per size.

Inputs : designlab/data/research/compute_2026-11-07.json (sourced survey, verification and the
         synthesized architectures: whole-compute-set typical power, mass and price), and the Stage A
         records of the reference candidates (round1/stage_a/candidates/<h>-M-S-10.json).
Output : round1/compute_options.json and the tables quoted in COMPUTE.md.

Method (first order, stated): the reference candidate's drive power and the other auxiliaries are kept;
the Jetson budget (module 25 W + carrier/storage/cooling 6 W, through the 0.88 DC-DC) is replaced by
the architecture's typical whole-set power through the same converter; runtime is the selected
pack's usable energy (75 %) over the new walking power.  Mass and price change by the difference
from the Jetson set (0.294 kg incl. tray; $399 + $120 integration as costed).  The mass change
(tens to a few hundred grams) is not fed back into the joint loads; at these sizes it moves the
walking power by well under 1 %.  Camera-side inference power is counted in the architecture's
set when the architecture moves inference onto the camera.
Run from hardware/trex_form_study:  python -m designlab.compute_options
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from . import power

HEIGHTS = (60, 75, 100, 120)
JETSON_W = power.AUX_FIXED["jetson_module"] + power.AUX_FIXED["carrier_storage_cooling"]
JETSON_KG = 0.294
JETSON_USD = 399 + 120  # catalog price (NVIDIA list after the July 2026 rise) + integration, as costed


def effects(research: dict, stage_a: Path) -> dict:
    archs = research["synthesis"]["architectures"]
    out = {"method": __doc__.split("Method (first order, stated):")[1].split("Run from")[0].strip(), "sizes": {}}
    for h in HEIGHTS:
        rec = json.loads((stage_a / "candidates" / f"{h}-M-S-10.json").read_text())
        drives = rec["power"]["drives_w"]
        aux_load = rec["power"]["aux"]["load_w"]
        energy = rec["runtime"]["energy_wh"]
        base_total = rec["power"]["total_w"]
        rows = []
        for a in archs:
            p = a.get("power_w_typical")
            if p is None:
                continue
            load = aux_load - JETSON_W + p
            total = drives + load / power.ETA_DCDC
            minutes = 60.0 * energy * power.USABLE / total
            rows.append(dict(id=a["id"], name=a["name"], compute_w=p, walking_w=total,
                             runtime_min=minutes, runtime_change_min=minutes - rec["runtime"]["minutes"],
                             mass_change_kg=(a["mass_g"] / 1000.0 - JETSON_KG) if a.get("mass_g") else None,
                             price_change_usd=(a["price_usd"] + 120 - JETSON_USD) if a.get("price_usd") else None,
                             covers_minimum_spec=a.get("covers_minimum_spec"), best_for_sizes=a.get("best_for_sizes", [])))
        out["sizes"][str(h)] = dict(candidate=f"{h}-M-S-10", baseline_walking_w=base_total,
                                    baseline_runtime_min=rec["runtime"]["minutes"], battery=rec["selection"]["battery"],
                                    rows=rows)
    return out


def table_md(eff: dict) -> str:
    lines = []
    for h in HEIGHTS:
        s = eff["sizes"][str(h)]
        lines.append(f"\n**{h} cm** (`{s['candidate']}`, {s['battery']}, Jetson baseline "
                     f"{s['baseline_walking_w']:.0f} W walking, {s['baseline_runtime_min']:.0f} min)\n")
        lines.append("| Architecture | Compute set, W | Walking power | Runtime | Change | Mass change | Price change |")
        lines.append("|---|---|---|---|---|---|---|")
        for r in s["rows"]:
            mc = f"{1000 * r['mass_change_kg']:+.0f} g" if r["mass_change_kg"] is not None else "–"
            pc = f"{r['price_change_usd']:+,.0f} $" if r["price_change_usd"] is not None else "–"
            lines.append(f"| {r['name']} | {r['compute_w']:.1f} | {r['walking_w']:.0f} W | {r['runtime_min']:.1f} min | "
                         f"{r['runtime_change_min']:+.1f} min | {mc} | {pc} |")
    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--research", default="designlab/data/research/compute_2026-11-07.json")
    ap.add_argument("--stage-a", default="round1/stage_a")
    ap.add_argument("--out", default="round1/compute_options.json")
    a = ap.parse_args()
    research = json.loads(Path(a.research).read_text())
    eff = effects(research, Path(a.stage_a))
    Path(a.out).write_text(json.dumps(eff, indent=1))
    print(table_md(eff))


if __name__ == "__main__":
    main()
