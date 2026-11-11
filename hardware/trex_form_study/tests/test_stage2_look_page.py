"""Tests for the Stage 2 look review page (designlab/stage2/look_page.py; STAGE2_PLAN.md 4 A.5, look gate L1).

Run from hardware/trex_form_study:  python -m pytest tests/test_stage2_look_page.py -q
(with LOOK_MEDIA=<the render run's media directory> for part 3).

1. Number formatting: ratios (a third decimal from 0.95 up), spans and relative changes.
2. The page from the committed JSON files: its title, theme tokens and no external resource but Google Fonts; every
   clip, poster and plot staged is shown and nothing else is named; the over-limit walk and the tail-phase numbers as
   screen.json records them; the cost tables' ratios are screen.json's; the viewers' view names no rung, covers every
   blind clip's label band and turns its native controls off, and the key sits in the owner's view.
3. With the render run's media (skipped otherwise): the build copies every clip byte for byte (the blind set too),
   converts the posters and plots, names every file it writes, and stays under the size limit.
"""

from __future__ import annotations

import json
import os
import re
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import designlab  # noqa: E402,F401  (selects the GL backend before mujoco is imported)
from designlab.stage2 import look_page as LP  # noqa: E402

LOOK = ROOT / "stage2" / "look"
INPUTS = ("metrics.json", "screen.json", "clips.json", "look_refs.json")
needs_json = pytest.mark.skipif(not all((LOOK / n).exists() for n in INPUTS), reason="the look results are not here")
MEDIA = Path(os.environ["LOOK_MEDIA"]) if os.environ.get("LOOK_MEDIA") else None
needs_media = pytest.mark.skipif(MEDIA is None or not (MEDIA / "clips.json").exists(),
                                 reason="set LOOK_MEDIA to the render run's media directory")


# ------------------------------------------------------------------------------------------------
# 1. Formatting
# ------------------------------------------------------------------------------------------------
def test_number_formats():
    assert LP.ratio(0.8648) == "0.86"
    assert LP.ratio(0.9499) == "0.95"
    assert LP.ratio(0.95) == "0.950"
    assert LP.ratio(1.01229) == "1.012"
    assert LP.num(-0.5, 1) == "−0.5"
    assert LP.num(0.479, 2, True) == "+0.48"
    assert LP.span([0.42, 0.49]) == "0.42–0.49"
    assert LP.span([0.126, 0.1304], 2) == "0.13"
    assert LP.span([4.3, 7.9], 1, "mm") == "4.3–7.9 mm"
    assert LP.pct_span([-0.06, -0.19]) == "6–19 %"
    assert LP.pct_span([0.12]) == "12 %"


# ------------------------------------------------------------------------------------------------
# 2. The page from the JSON files
# ------------------------------------------------------------------------------------------------
@pytest.fixture(scope="module")
def built():
    d = LP.load(LOOK)
    return d, LP.page(d, commit="0000000")


def staged_paths(d) -> set[str]:
    out = set()
    for e in d.clips["clips"]:
        out |= {f"media/{e['file']}", f"media/{LP.poster_of(e['poster'])}"}
    for letter, _, _ in LP.BLIND:
        for view in ("visitor", "side"):
            out |= {f"media/{LP.blind_name(letter, view, 'mp4')}", f"media/{LP.blind_name(letter, view, 'jpg')}"}
    return out | {f"media/{p}" for p in d.metrics["plots"]}


@needs_json
def test_title_theme_and_external_resources(built):
    _, text = built
    assert text.startswith(f"<title>{LP.TITLE}</title>") and LP.TITLE == "T. rex look review"
    assert '<meta name="claude:viewport-fit-sides" content="cover">' in text
    hosts = set(re.findall(r'(?:src|href)="(https?://[^/"]+)', text))
    assert hosts == {"https://fonts.googleapis.com", "https://fonts.gstatic.com"}  # Google Fonts, preconnected
    assert "@media (prefers-color-scheme: dark)" in text and ':root[data-theme="dark"]' in text
    assert "<script src=" not in text and "<iframe" not in text


@needs_json
def test_every_staged_file_is_shown_and_nothing_else_is_named(built):
    d, text = built
    named = set(LP.referenced(text))
    assert named == staged_paths(d)
    assert len([p for p in named if p.endswith(".mp4") and "/blind/" not in p]) == len(d.clips["clips"])


@needs_json
def test_over_limit_walk_and_tail_finding_as_the_screen_records_them(built):
    d, text = built
    over = [o for b in LP.BODIES for s in LP.SPEEDS for v in LP.LADDER for o in d.s(b, s, v)["over"]]
    assert over, "the screen records the over-limit rung this page reports"
    for o in over:
        assert f"{o['limit']} {o['value']:.3f}" in text
    tp = d.screen["tail_phase"]
    for b in LP.BODIES:
        for s in LP.SPEEDS:
            c = tp[b][s]["changes"]
            for key in ("V0_no_tail_to_today", "V0_no_tail_to_F9_phase", "V0_phase_today_to_F9"):
                assert LP.num(100 * c[key]["mz_abs_max_nm"]["rel"], 1, True) + " %" in text
            assert tp[b][s]["today_adds_to_leg_yaw_moment"]


@needs_json
def test_cost_tables_read_the_screen():
    d = LP.load(LOOK)
    for v in LP.LADDER[1:]:
        base = "V5" if v == "V6" else LP.prev_rung(v)
        listed = LP.moves(d, v, base)
        for _, vals in listed:
            assert max(abs(c - a) for a, c in vals.values()) >= LP.MOVE - 1e-12
            assert set(vals) == set(LP.combos())
        n_moving = 0
        for key in LP.readings(d.s(LP.BODIES[0], LP.SPEEDS[0], v)):
            ch = [abs(LP.readings(d.s(b, s, v))[key][1] - LP.readings(d.s(b, s, base))[key][1]) for b, s in LP.combos()]
            n_moving += max(ch) >= LP.MOVE - 1e-12
        assert n_moving == len(listed)
    knee = dict(LP.moves(d, "V2"))["knee speed"]
    b, s = LP.BODIES[0], "nominal"
    assert knee[(b, s)] == (d.s(b, s, "V1")["drives"]["knee"]["speed_ratio"], d.s(b, s, "V2")["drives"]["knee"]["speed_ratio"])


@needs_json
def test_the_viewers_view_names_no_rung_and_covers_the_labels(built):
    _, text = built
    i = text.index('<section id="rate">')
    rate = text[i:text.index("</section>", i)]
    key = rate.index("The key: which clip is which")
    viewers = rate.index('<div class="prose" data-only="viewers">')
    assert rate.rfind('data-only="owner"', 0, key) != -1 and key < viewers
    blind = rate[viewers:]
    assert not re.search(r"\bV[0-6]\b", blind)
    boxes = re.findall(r'<div class="vbox">(<video [^>]*>)</video>(.*?)</div></div>', blind)
    assert len(boxes) == 2 * len(LP.BLIND)
    for tag, rest in boxes:
        assert " controls" not in tag and "/blind/clip_" in tag
        assert 'class="mask"' in rest and "kinematic reference, not a simulation" in rest
    assert 'html.viewers [data-only="owner"]{display:none!important}' in text
    assert "location.hash==='#viewers'" in text


@needs_json
def test_json_slots_name_clips_that_exist(built):
    d, text = built
    files = {f"media/{e['file']}" for e in d.clips["clips"]}
    for m in re.finditer(r'<script type="application/json" id="([a-z0-9-]+)">(.*?)</script>', text, flags=re.S):
        doc = json.loads(m.group(2).replace("<\\/", "</"))
        slots = ([x for xs in doc["slots"].values() for x in xs] if "slots" in doc
                 else [doc["v0"]] + [x for st in doc["steps"].values() for x in (st["prev"], st["after"])])
        assert slots and all(x["src"] in files for x in slots)


# ------------------------------------------------------------------------------------------------
# 3. The build, with the media
# ------------------------------------------------------------------------------------------------
@needs_json
@needs_media
def test_build_stages_the_media_exactly(tmp_path):
    from PIL import Image

    res = LP.build(MEDIA, tmp_path)
    d = LP.load(LOOK)
    assert res["unused"] == [] and res["total_bytes"] <= LP.MAX_PAGE_MB * 1e6
    by_path = {f["path"]: f for f in res["files"]}
    for e in d.clips["clips"]:
        assert by_path[f"media/{e['file']}"]["sha256"] == e["sha256"]
        with Image.open(tmp_path / "media" / LP.poster_of(e["poster"])) as im:
            assert im.format == "JPEG" and list(im.size) == e["size"]
    for letter, b, v in LP.BLIND:
        for view in ("visitor", "side"):
            assert by_path[f"media/{LP.blind_name(letter, view, 'mp4')}"]["sha256"] == d.clip(b, LP.BLIND_SPEED, v, view)["sha256"]
    for p in d.metrics["plots"]:
        with Image.open(tmp_path / "media" / p) as a, Image.open(MEDIA / p) as b:
            assert a.size == b.size and a.mode == "P"
    assert set(LP.referenced((tmp_path / "index.html").read_text())) == set(by_path)
