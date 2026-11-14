"""Round-two what-if: bodies re-sized with the animal-like walk as a sizing case.

The behavior envelope (``envelope.py``) holds each sized body's drives against a prescribed walk at Froude 0.05
(``envelope.FAST``: stride 0.80 H, duty 0.56), the animal's preferred walking speed class.  The round-two
sizing walks are Froude 0.012 and 0.025 (BEHAVIOR_SPEC section 1), so a body can meet the spec and still lack
the drive speed or torque for that walk.  This re-sizes named round-two bodies with ``envelope.FAST`` added
to the peak and speed requirements (``sizing.EXTRA_WALKS``; the continuous requirement stays the nominal walk
and the holds, as for the brisk walk), then runs every round-two screen on the result, so a what-if body is
screened exactly like the round-two candidates.  Packaging overrides found by the round-two search are
searched again, since a larger drive can change the packaging.

Run from hardware/trex_form_study:
    python -m designlab.whatif --out round2/whatif_fr05 75-M-T-10 75-M-H-10
Outputs mirror ``round2``: <out>/candidates/<id>.json (with ``whatif``), <out>/review/mjcf (ledgers),
<out>/review/screening, <out>/review/plates (four views and a cutaway per body) and <out>/review/review_index.json.
"""

from __future__ import annotations

import argparse
import json
import multiprocessing as mp
from pathlib import Path

WHATIFS = ("fr05",)


def _init(name: str) -> None:
    from . import envelope, sizing

    sizing.EXTRA_WALKS = {"fr05": envelope.FAST} if name == "fr05" else {}


def _size(a):
    from . import round2

    name, cid, out, xml_dir = a
    _init(name)
    r = round2.size_one(cid, out, xml_dir)
    p = Path(out, "candidates", f"{cid}.json")
    d = json.loads(p.read_text())
    d["whatif"] = name
    p.write_text(json.dumps(d, indent=1))
    return r


def _screen(a):
    from . import round2

    name, cid, out = a
    _init(name)
    return round2._screen_star((cid, Path(out)))


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("ids", nargs="+")
    ap.add_argument("--whatif", default="fr05", choices=WHATIFS)
    ap.add_argument("--out", required=True)
    ap.add_argument("--workers", type=int, default=2)
    a = ap.parse_args()
    import designlab  # noqa: F401

    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    xml_dir = str(out / "models")
    ctx = mp.get_context("spawn")
    with ctx.Pool(a.workers) as pool:
        for r in pool.imap_unordered(_size, [(a.whatif, i, str(out), xml_dir) for i in a.ids]):
            print("sized", r, flush=True)
        results = list(pool.imap_unordered(_screen, [(a.whatif, i, str(out)) for i in a.ids]))
    idx = out / "review" / "review_index.json"
    index = json.loads(idx.read_text()) if idx.exists() else {}
    index.update({r["id"]: r for r in results})
    idx.write_text(json.dumps(index, indent=1, default=str))
    from . import plates
    from .round1 import compact_png

    P = out / "review" / "plates"
    P.mkdir(parents=True, exist_ok=True)
    for cid in a.ids:
        x = out / "review" / "mjcf" / f"{cid}.xml"
        if x.exists():
            for png in (plates.candidate_plate(str(x), str(P / f"{cid}_four_views.png"), cid,
                                               f"what-if: sized for the Froude 0.05 walk ({a.whatif})"),
                        plates.cutaway_plate(str(x), str(P / f"{cid}_cutaway.png"),
                                             f"{cid} sized for the Froude 0.05 walk: installed hardware")):
                if png:
                    compact_png(str(png))
    for r in results:
        print("screened", r["id"], {k: v for k, v in r.items() if k.endswith("_error") or k == "error"}, flush=True)


if __name__ == "__main__":
    main()
