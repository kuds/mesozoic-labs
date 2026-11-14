"""Command line entry points.

Run from hardware/trex_form_study (Python 3.11+, MuJoCo 3.10.0, numpy, scipy, Pillow):

  python -m designlab.cli build 60-M-S-10 --out models/            # MJCF + ledger, default parts
  python -m designlab.cli size 60-M-S-10 --out models/             # + load-based actuator/battery sizing
  python -m designlab.cli views 60-M-S-10 --out plates/            # quick 6-view sheet of a built model
  python -m designlab.cli catalog                                   # rebuild catalogs from research records

Candidate IDs: <height cm>-<width N|M|W>-<family T|S|H>-<posture 00|10|20>[-<proportion>][-<layout>][-belly]
e.g. 100-M-H-10, 60-M-S-10-anat, 120-M-S-10-serial, 75-M-S-10-belly.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import designlab  # noqa: F401  (selects the offscreen GL backend before mujoco is imported)

from .params import parse


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name in ("build", "size", "views"):
        p = sub.add_parser(name)
        p.add_argument("ids", nargs="+")
        p.add_argument("--out", default="models")
    sub.add_parser("catalog")
    a = ap.parse_args()
    if a.cmd == "catalog":
        from .data import build_catalog

        build_catalog.convert_actuators()
        build_catalog.convert_parts()
        return
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    for cid in a.ids:
        cand = parse(cid)
        if a.cmd == "build":
            from .pipeline import build_candidate

            r = build_candidate(cand, out)
            print(json.dumps({k: v for k, v in r["summary"].items()}, default=str))
        elif a.cmd == "size":
            from .pipeline import size_candidate

            r = size_candidate(cand, out, verbose=True)
            ev = r["evaluation"]
            print(cid, "mass %.2f kg" % r["summary"]["mass"], "walk %.0f W" % ev["power"]["total_w"],
                  "runtime %.1f min" % ev["runtime"]["minutes"], "cost nominal $%.0f" % ev["cost"]["nominal"])
        elif a.cmd == "views":
            import numpy as np
            from PIL import Image

            from . import render

            xml = out / f"{cid}.xml"
            if not xml.exists():
                from .pipeline import build_candidate

                build_candidate(cand, out)
            sc = render.Scene(str(xml), 1000, 640)
            tiles = [sc.render(v, "exterior") for v in ("side", "front", "top", "threequarter")]
            tiles += [sc.render("side", "cutaway"), sc.render("side", "silhouette")]
            sc.close()
            rows = [np.concatenate(tiles[i : i + 2], axis=1) for i in range(0, 6, 2)]
            Image.fromarray(np.concatenate(rows, axis=0)).save(out / f"{cid}_views.png")
            print(out / f"{cid}_views.png")


if __name__ == "__main__":
    main()
