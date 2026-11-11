"""Tests for the Stage 2 look review page (designlab/stage2/look_page.py; STAGE2_PLAN.md 4 A.5, look gate L1).

Run from hardware/trex_form_study:  python -m pytest tests/test_stage2_look_page.py -q
(with LOOK_MEDIA=<the render run's media directory> for part 3).

1. Number formatting: ratios (a third decimal from 0.95 up), spans and relative changes.
2. The page from the committed JSON files: its title, theme tokens and no external resource but Google Fonts; every
   clip, poster and plot staged is shown and nothing else is named; the over-limit walk and the tail-phase numbers as
   screen.json records them; the cost tables' ratios are screen.json's; the viewers' view names no rung, covers every
   blind clip's label band and turns its native controls off, and the key sits in the owner's view; the blind band
   replaces the whole label band; the ladder's chips are the research's four tells (and the nod) as metrics.json reads
   them; the claw and toe-roll statements read every rung with a toe-off; the review's corrected texts; each view keeps
   its own score draft.
3. With the render run's media (skipped otherwise): the build copies every clip byte for byte, re-encodes the blind
   set with its label band replaced (and nothing else changed beyond one re-encode), converts the posters and plots,
   names every file it writes, and stays under the size limit.
"""

from __future__ import annotations

import json
import os
import re
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import designlab  # noqa: E402,F401  (selects the GL backend before mujoco is imported)
from designlab.stage2 import clips  # noqa: E402
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


def test_blind_frame_replaces_the_whole_band():
    rng = np.random.default_rng(0)
    a, b = (rng.integers(0, 256, (480, 854, 3), dtype=np.uint8) for _ in range(2))
    entry = dict(label_lines=[2, 1])
    rows = LP.blind_band_rows(entry)
    assert rows % 16 == 0 and rows >= clips.BANDS_PX[0]
    fa, fb = LP.blind_frame(a, "A", rows), LP.blind_frame(b, "A", rows)
    assert np.array_equal(fa[:rows], fb[:rows])  # nothing of the frame's own band survives
    assert np.array_equal(fa[rows:], a[rows:]) and np.array_equal(fb[rows:], b[rows:])
    assert not np.array_equal(fa[:rows], LP.blind_frame(a, "B", rows)[:rows])  # the letter is drawn
    assert LP.blind_lines("C")[0] == f"Clip C – {clips.LABEL}"


@needs_json
def test_ladder_tells_are_the_researchs_four(built):
    d, text = built
    assert [k for k, _ in LP.TELLS] == ["constant_hip_height", "flat_feet_at_lift_off", "identical_steps",
                                        "synchronous_extremes"]
    assert LP.NOD[0] == "metronome_nod" and "none of the four tells is left" not in text
    for v in LP.LADDER:
        m = d.m(*LP.TELLS_AT, v)
        i = text.index(f'<article class="rung" id="rung-{v}">')
        head = text[i:text.index("</div>", text.index('<p class="tnote">', i))]
        for _, lab in LP.TELLS + (LP.NOD,):
            assert lab in head
        pairs = m["mechanical_tells"]["synchronous_extremes"]
        assert all(LP.esc(LP.pair_text(p)) in head for p in pairs)
        flat = dict(LP.tell_states(m))["flat_feet_at_lift_off"]
        struck = "<s>flat feet at lift-off</s>" in head
        assert struck == (m["toe_off"]["at_lift_off_rad"] >= LP.TOE_OFF_LOW - LP.TOE_OFF_TOL)
        assert flat == ("on" if m["mechanical_tells"]["flat_feet_at_lift_off"] else "off" if struck else "part")
    # synchronous extremes remain on V6 at every walk, so the page says so
    assert all(d.m(b, s, "V6")["mechanical_tells"]["synchronous_extremes"] for b, s in LP.combos())
    assert "synchronous extremes remain" in text


@needs_json
def test_claw_and_toe_roll_read_every_rung(built):
    d, text = built
    rolls = LP.toe_rolls(d)
    for (b, s, v), (pk, cls) in rolls.items():
        info = d.refs["body_info"][b]
        touch = info["toe_clearance"]["clear_pitch_rad"]
        assert (cls == "claw") == (pk > touch)
    assert rolls[("75-M-H-10", "v040", "V3")][1] == "arc"  # 0.056 rad: on the arc, short of the 0.068 rad claw
    assert rolls[("75-M-H-10", "fr05", "V6")][1] == "claw" and rolls[("75-M-H-10", "fr05", "V5F13")][1] == "claw"
    i = text.index("<b>The claw (hardware).</b>")
    claw = text[i:text.index("</li>", i)]
    far, arc = claw.split("Rolls that far:")[1].split("From ")
    assert "75 cm at the animal pace (V6, V5F13 and V6S" in far and "75 cm at 0.40 m/s (V3–V6" not in far
    assert "75 cm at 0.40 m/s (V3–V6" in arc
    i = text.index("<b>How far the toes roll (F5).</b>")
    assert "0.020–0.084 rad" in text[i:text.index("</li>", i)]


@needs_json
def test_texts_the_checker_corrected(built):
    _, text = built
    assert "The foot pads clear the floor by" in text and "The toes clear the floor" not in text
    assert "only brakes" not in text and "mostly brakes it" in text
    assert "in cycles of about 5 and 10 s" in text and "over 5–20 s" not in text
    assert "the animal pace still above 0.9" in text
    assert "over its limit at 0.40 m/s (0.968 to 1.014)" in text  # V4 takes the 75 cm tail over (with friction)
    assert "No torque of its own" not in text and "Same motor costs as V4" not in text
    assert "Cheaper on the tail than V4" in text


@needs_json
def test_each_view_keeps_its_own_score_draft(built):
    _, text = built
    keys = re.search(r"var KEYS = \{owner: '([^']+)', viewers: '([^']+)'\}", text)
    assert keys and keys.group(1) != keys.group(2)
    assert "localStorage.setItem(KEY," in text and "addEventListener('hashchange', function(){ var k = keyNow()" in text


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
            e = d.clip(b, LP.BLIND_SPEED, v, view)
            path = tmp_path / "media" / LP.blind_name(letter, view, "mp4")
            # a re-encode, not a copy: the labelled clip's top band (which names the rung) is replaced
            assert by_path[f"media/{LP.blind_name(letter, view, 'mp4')}"]["sha256"] != e["sha256"]
            blind, labelled = LP.decode(path, tuple(e["size"])), LP.decode(MEDIA / e["file"], tuple(e["size"]))
            assert blind.shape == labelled.shape and len(blind) == e["frames"]
            rows = LP.blind_band_rows(e)
            for k in (0, len(blind) // 2):
                want = LP.blind_frame(labelled[k], letter, rows)
                assert np.abs(blind[k, :rows].astype(int) - want[:rows].astype(int)).mean() < 3.0
            mse = ((blind[:, rows:].astype(float) - labelled[:, rows:].astype(float)) ** 2).mean()
            assert 10 * np.log10(255 ** 2 / mse) > 35.0  # the walk itself: the labelled clip's, re-encoded once
            assert set(clips.probe(path).items()) >= {("frames", e["frames"]), ("width", e["size"][0]),
                                                      ("height", e["size"][1])}
            tags = json.loads(subprocess.run(["ffprobe", "-v", "error", "-show_entries", "format_tags", "-of", "json",
                                              str(path)], capture_output=True, text=True, check=True).stdout)
            assert not re.search(rf"\bV[0-6]\b|{b}|{LP.BLIND_SPEED}", json.dumps(tags))
            with Image.open(tmp_path / "media" / LP.blind_name(letter, view, "jpg")) as im:
                poster = np.asarray(im.convert("RGB"))
            with Image.open(MEDIA / e["poster"]) as im:
                want = LP.blind_frame(np.asarray(im.convert("RGB")), letter, rows)
            assert np.abs(poster[:rows].astype(int) - want[:rows].astype(int)).mean() < 3.0
    for p in d.metrics["plots"]:
        with Image.open(tmp_path / "media" / p) as a, Image.open(MEDIA / p) as b:
            assert a.size == b.size and a.mode == "P"
    assert set(LP.referenced((tmp_path / "index.html").read_text())) == set(by_path)
