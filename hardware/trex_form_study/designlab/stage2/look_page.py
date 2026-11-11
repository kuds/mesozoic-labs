"""Stage 2 look gate L1: the owner's look review page (``STAGE2_PLAN.md`` 4 A.5; ``research_look.md`` 4.1 items 3-6).

A static page, ``index.html`` beside a ``media/`` folder, for publishing as a claude.ai artifact: the look ladder's
clips (``clips.json``), the Stage 0 price of every rung (``screen.json``, the tail-phase check included), the machine
metrics and plots (``metrics.json``), the rubric with the Godspeed animacy items as a score sheet, and the decisions
the review asks of the owner.  Every number on the page is read from those files (and the rung specs in
``look_refs.json``); the prose around the numbers is this module's.  Simulation only: kinematic references with
perfect tracking, priced before any render; nothing is adopted (P24).

Media (``stage_media``), from the render run's directory (``--media``, outside git):

* every clip copied byte for byte, each checked against the sha256 ``clips.json`` records;
* each clip's poster as a JPEG (quality ``POSTER_QUALITY``; the PNG posters are 3-4 times larger);
* each plot (``metrics.json`` ``plots``) as a 256-colour PNG (line art on a flat ground: no visible change, a third of
  the bytes);
* the blind set (``BLIND``): V0 and V6 at 0.40 m/s from both views, copied again under neutral names
  (``media/blind/clip_<letter>_<view>.mp4``), for the ``#viewers`` view.

The page has two views.  The owner's (the default) shows everything.  ``#viewers`` (the page's address with that
hash) shows only the blind clips and the score sheet, for the 3-5 viewers new to the project (research 4.1 item 6):
each blind clip's top label band, which names the rung, is covered on the page and its native controls (and with
them full screen) are off.  The score sheet keeps a viewer's draft in the browser (``localStorage``) and copies the
scores as tab-separated text for the viewer to send; nothing leaves the page by itself.

Run from hardware/trex_form_study (the page and its media go outside git; ``--manifest`` writes a small record of
what the page shows, with every file's source and sha256, beside the look results):
    python -m designlab.stage2.look_page --media <look_media> --out <look_page> [--manifest stage2/look/look_page.json]
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import shutil
import time
from dataclasses import dataclass
from pathlib import Path

from PIL import Image

from .. import page_kit as K
from . import look

OUT = look.OUT
TITLE = "T. rex look review"
DATE = "11 Nov 2026"
BODIES = look.BODIES
SPEEDS = look.SPEEDS
LADDER = look.VARIANTS
LADDER_SPEED = "v040"
BODY_NAME = {"60-N-T-10": "60 cm Tapered", "75-M-H-10": "75 cm Hybrid"}
BODY_SHORT = {"60-N-T-10": "60 cm", "75-M-H-10": "75 cm"}
BODY_ID = {"60-N-T-10": "b60", "75-M-H-10": "b75"}
SPEED_TAB = {"nominal": "Nominal walk", "v040": "0.40 m/s", "fr05": "Animal pace"}
SPEED_SHORT = {"nominal": "nominal", "v040": "0.40 m/s", "fr05": "animal pace"}
SPEED_LONG = {"nominal": "the nominal walk (Froude 0.012)", "v040": "0.40 m/s, the agility bar's speed",
              "fr05": "the animal pace (Froude 0.05)"}
VIEW_NAME = {"visitor": "visitor's view", "side": "side view"}
GROUPS = ("hip_roll", "hip_pitch", "hip_yaw", "knee", "ankle_a", "ankle_b", "neck", "tail")
GROUP = {"hip_roll": "hip roll", "hip_pitch": "hip pitch", "hip_yaw": "thigh twist", "knee": "knee",
         "ankle_a": "ankle A", "ankle_b": "ankle B", "neck": "neck", "tail": "tail"}
KIND = {"cont": "sustained torque", "peak": "peak torque", "speed": "speed"}
KIND_SHORT = {"cont": "sustained", "peak": "peak", "speed": "speed"}
#: What a clip's label calls each rung (the V0-V6 comparison, the step viewer and the point-light clips).
SLOT_NAME = {"V0": "today's walk", "V1": "+ hip bob", "V2": "+ swing time", "V3": "+ swing shape, toe-off",
             "V4": "+ trunk roll", "V5": "+ tail timing, steady head", "V6": "every feature"}
#: The neck and tail readings with the plant's joint damping and friction (screen.json ``passive``).
PASSIVE = (("neck", "cont"), ("neck", "peak"), ("tail", "cont"), ("tail", "peak"))
#: A ratio "moves" when it changes by at least this much between rungs (the cost lines and their tables).
MOVE = 0.05
#: The screen's near band: within 0.02 of a limit (screen.json ``margins.near_band``).
NEAR = 0.98
#: The blind set for the viewers (letter, body, rung), at ``BLIND_SPEED``, both views; the order is counterbalanced
#: (the all-features walk first at 60 cm, second at 75 cm) and the same for every viewer.
BLIND = (("A", "60-N-T-10", "V6"), ("B", "60-N-T-10", "V0"), ("C", "75-M-H-10", "V0"), ("D", "75-M-H-10", "V6"))
BLIND_SPEED = "v040"
POSTER_QUALITY = 82
PLOT_COLOURS = 256
MAX_PAGE_MB = 60.0
RUNG_TITLE = {"V0": "Today's walk", "V1": "Hip bob and weight acceptance", "V2": "Swing time",
              "V3": "Swing shape and toe-off", "V4": "Trunk roll", "V5": "Tail timing and a steady head",
              "V6": "Variation"}
RUNG_TAG = {"V1": "bob", "V2": "swing time", "V3": "toe-off", "V4": "roll", "V5": "tail, head", "V6": "variation"}
RUNG_ADDS = {"V0": [], "V1": ["F1", "F2"], "V2": ["F3"], "V3": ["F4", "F5"], "V4": ["F6"], "V5": ["F9", "F10"],
             "V6": ["F14"]}
TELLS = (("constant_hip_height", "hips at one height"), ("flat_feet_at_lift_off", "flat feet at lift-off"),
         ("identical_steps", "identical steps"), ("metronome_nod", "metronome nod"))
RUBRIC = (
    ("1", "Foot placement and roll", "Feet lift and land flat; shuffling or slapping",
     "Feet peel off from the toes and land quietly and deliberately"),
    ("2", "Knee and hock motion", "Knees locked at one bend; stiff",
     "Legs fold and reach smoothly; the stance leg carries the body"),
    ("3", "Body bob and weight", "Body glides at a fixed height or slides sideways",
     "Body rises and falls with the steps; weight settles after each footfall"),
    ("4", "Head and tail coordination", "Head and tail swing like metronomes or stay frozen",
     "Head steady and purposeful; tail lags and settles"),
    ("5", "No shuffling or jitter", "Visible jitter, foot sliding, stutter", "Clean, continuous motion"),
    ("6", "Timing and anticipation", "Starts and stops abruptly; every stride identical",
     "Anticipates, settles, varies slightly"),
    ("7", "Overall", "A machine walking", "An animal walking"),
)
GODSPEED = (("G1", "Dead", "Alive"), ("G2", "Stagnant", "Lively"), ("G3", "Mechanical", "Organic"),
            ("G4", "Artificial", "Lifelike"), ("G5", "Inert", "Interactive"), ("G6", "Apathetic", "Responsive"))
CLIP_LETTERS = tuple(x[0] for x in BLIND)


# ------------------------------------------------------------------------------------------------
# Inputs
# ------------------------------------------------------------------------------------------------
@dataclass
class Docs:
    metrics: dict
    screen: dict
    clips: dict
    refs: dict

    def m(self, b: str, s: str, v: str) -> dict:
        return self.metrics["bodies"][b][s][v]

    def s(self, b: str, s: str, v: str) -> dict:
        return self.screen["rungs"][b][s][v]

    def spec(self, b: str, s: str, v: str) -> dict:
        return self.refs["bodies"][b][s][v]["spec"]

    def clip(self, b: str, s: str, v: str, view: str, mode: str = "covered") -> dict:
        for e in self.clips["clips"]:
            if (e["body"], e["speed"], e["variant"], e["view"], e["mode"]) == (b, s, v, view, mode):
                return e
        raise KeyError((b, s, v, view, mode))


def load(look_dir: Path = OUT) -> Docs:
    rd = lambda n: json.loads((Path(look_dir) / n).read_text())  # noqa: E731
    return Docs(rd("metrics.json"), rd("screen.json"), rd("clips.json"), rd("look_refs.json"))


# ------------------------------------------------------------------------------------------------
# Media
# ------------------------------------------------------------------------------------------------
def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def poster_of(path: str) -> str:
    return str(Path(path).with_suffix(".jpg"))


def blind_name(letter: str, view: str, ext: str) -> str:
    return f"blind/clip_{letter}_{view}.{ext}"


def _copy_checked(src: Path, dst: Path, want: str) -> None:
    got = sha256(src)
    if got != want:
        raise ValueError(f"{src}: sha256 {got[:12]} is not the {want[:12]} clips.json records")
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(src, dst)


def _jpeg(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    Image.open(src).convert("RGB").save(dst, "JPEG", quality=POSTER_QUALITY, optimize=True, progressive=True)


def _plot(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    im = Image.open(src).convert("RGB")
    im.quantize(colors=PLOT_COLOURS, method=Image.Quantize.MEDIANCUT, dither=Image.Dither.NONE).save(dst, optimize=True)


def stage_media(d: Docs, media: Path, out: Path) -> list[dict]:
    """Copy and convert the media into ``<out>/media`` (module docstring); one row per file written."""
    mdir = Path(out) / "media"
    rows = []

    def row(kind, rel, src, **kw):
        p = mdir / rel
        rows.append(dict(kind=kind, path=f"media/{rel}", source=str(Path(src).relative_to(media)), bytes=p.stat().st_size,
                         sha256=sha256(p), **kw))

    for e in d.clips["clips"]:
        _copy_checked(media / e["file"], mdir / e["file"], e["sha256"])
        row("clip", e["file"], media / e["file"], body=e["body"], speed=e["speed"], variant=e["variant"],
            view=e["view"], mode=e["mode"])
        _jpeg(media / e["poster"], mdir / poster_of(e["poster"]))
        row("poster", poster_of(e["poster"]), media / e["poster"])
    for letter, b, v in BLIND:
        for view in ("visitor", "side"):
            e = d.clip(b, BLIND_SPEED, v, view)
            _copy_checked(media / e["file"], mdir / blind_name(letter, view, "mp4"), e["sha256"])
            row("blind clip", blind_name(letter, view, "mp4"), media / e["file"], letter=letter)
            _jpeg(media / e["poster"], mdir / blind_name(letter, view, "jpg"))
            row("blind poster", blind_name(letter, view, "jpg"), media / e["poster"], letter=letter)
    for p in d.metrics["plots"]:
        _plot(media / p, mdir / p)
        row("plot", p, media / p)
    return rows


# ------------------------------------------------------------------------------------------------
# Numbers
# ------------------------------------------------------------------------------------------------
def num(x: float, nd: int = 2, signed: bool = False) -> str:
    s = f"{x:+.{nd}f}" if signed else f"{x:.{nd}f}"
    return s.replace("-", "−")


def ratio(x: float) -> str:
    """A load ratio: three decimals at or above 0.95 (where the third decides), else two."""
    return num(x, 3 if x >= 0.95 else 2)


def span(vals, nd: int = 2, unit: str = "") -> str:
    vals = [v for v in vals if v is not None]
    lo, hi = min(vals), max(vals)
    u = f" {unit}" if unit else ""
    if round(lo, nd) == round(hi, nd):
        return f"{num(lo, nd)}{u}"
    return f"{num(lo, nd)}–{num(hi, nd)}{u}"


def pct_span(vals, nd: int = 0) -> str:
    """Relative changes as a span of magnitudes ("6-19 %"); the caller says which way."""
    mags = [abs(100 * v) for v in vals]
    lo, hi = min(mags), max(mags)
    if round(lo, nd) == round(hi, nd):
        return f"{lo:.{nd}f} %"
    return f"{lo:.{nd}f}–{hi:.{nd}f} %"


def worst(drive: dict) -> tuple[str, float]:
    ks = {"cont": drive["cont_ratio"], "peak": drive["peak_ratio"], "speed": drive["speed_ratio"]}
    k = max(ks, key=ks.get)
    return k, ks[k]


def mu(r: dict) -> float:
    return r["friction"]["single_support"]["mu_req_max"]


def twist(r: dict) -> float:
    return r["friction"]["single_support"]["mz_abs_max_nm"]


def passive(r: dict, part: str, kind: str) -> float:
    return r["passive"][part][f"{kind}_ratio_total"]


def gov(r: dict) -> tuple[str, str, float]:
    g, k, v = r["governing"]
    return g, k, v


def readings(r: dict) -> dict[str, tuple[str, float]]:
    """Every ratio a cost line compares: each drive group's worst (Stage 0) and the neck and tail with the plant's
    joint damping and friction."""
    out = {g: worst(r["drives"][g]) for g in GROUPS}
    for part, kind in PASSIVE:
        out[f"{part}+{kind}"] = (kind, passive(r, part, kind))
    return out


def reading_label(key: str, kinds: set[str]) -> str:
    """A reading's name; a drive group's worst ratio is named by its kind when that kind is the same on both rungs at
    every body and speed."""
    if "+" in key:
        part, kind = key.split("+")
        return f"{part} servo, {KIND[kind]} with joint friction"
    if len(kinds) == 1:
        return f"{GROUP[key]} {KIND[next(iter(kinds))]}"
    return f"{GROUP[key]}, worst of " + " and ".join(KIND[k] for k in ("cont", "peak", "speed") if k in kinds)


def prev_rung(v: str) -> str:
    return LADDER[LADDER.index(v) - 1]


def combos():
    return [(b, s) for b in BODIES for s in SPEEDS]


def combo_label(b: str, s: str) -> str:
    return f"{BODY_SHORT[b]}, {SPEED_SHORT[s]}"


def moves(d: Docs, v: str, base: str | None = None) -> list[tuple[str, dict]]:
    """The ratios that change by ``MOVE`` or more from the rung before (or ``base``) at any body and speed:
    [(label, {(body, speed): (before, after)})], largest change first."""
    base = base or prev_rung(v)
    vals: dict[str, dict] = {}
    kinds: dict[str, set] = {}
    for b, s in combos():
        r0, r1 = readings(d.s(b, s, base)), readings(d.s(b, s, v))
        for key in r0:
            (k0, a), (k1, c) = r0[key], r1[key]
            vals.setdefault(key, {})[(b, s)] = (a, c)
            kinds.setdefault(key, set()).update((k0, k1))
    out = []
    for key, vv in vals.items():
        big = max(abs(c - a) for a, c in vv.values())
        if big >= MOVE - 1e-12:
            out.append((big, reading_label(key, kinds[key]), vv))
    out.sort(key=lambda x: -x[0])
    return [(lab, vv) for _, lab, vv in out]


def rel(a: float, b: float) -> float:
    return b / a - 1.0


# ------------------------------------------------------------------------------------------------
# HTML pieces
# ------------------------------------------------------------------------------------------------
esc = K.esc


def video(src: str, poster: str, aria: str, controls: bool = True) -> str:
    return (f'<video src="{esc(src)}" poster="{esc(poster)}" preload="none" playsinline muted loop '
            f'disablepictureinpicture{" controls" if controls else ""} width="854" height="480" '
            f'aria-label="{esc(aria)}"></video>')


def slot_data(d: Docs, b: str, s: str, v: str, view: str, mode: str = "covered") -> dict:
    e = d.clip(b, s, v, view, mode)
    feats = e["features"]
    name = f"{v} · {SLOT_NAME[v]}"
    sub = VIEW_NAME[view] + (", point-light" if mode == "pointlight" else "")
    flag = ""
    if e["stage0_over"]:
        flag = "over a limit: " + "; ".join(f"{o['limit']} {ratio(o['value'])}" for o in e["stage0_over"])
    aria = (f"{BODY_NAME[b]}, {SPEED_SHORT[s]}, {v} ({', '.join(feats) or 'today' + chr(39) + 's walk'}), "
            f"{sub}; kinematic reference, not a simulation")
    return dict(src=f"media/{e['file']}", poster=f"media/{poster_of(e['poster'])}", name=name, sub=sub, flag=flag,
                aria=aria)


def vslot(x: dict, controls: bool = True, mask: str | None = None) -> str:
    flag = f'<span class="vflag tag bad"{"" if x.get("flag") else " hidden"}>{esc(x.get("flag", ""))}</span>'
    m = ""
    if mask:
        m = (f'<div class="mask" aria-hidden="true"><b>{esc(mask)}</b>'
             '<span>kinematic reference, not a simulation</span></div>')
    return (f'<div class="vslot"><div class="vlab"><span class="vname">{esc(x["name"])}</span>'
            f'<span class="vsub">{esc(x["sub"])}</span>{flag}</div>'
            f'<div class="vbox">{video(x["src"], x["poster"], x["aria"], controls)}{m}</div></div>')


def controls(extra: str = "") -> str:
    return ('<div class="vctl"><button type="button" class="btn primary" data-act="play">Play together</button>'
            '<button type="button" class="btn" data-act="pause">Pause</button>'
            '<button type="button" class="btn" data-act="slow" aria-pressed="false">Half speed</button>'
            f'{extra}</div>')


def json_script(id_: str, obj) -> str:
    txt = json.dumps(obj, separators=(",", ":")).replace("</", "<\\/")
    return f'<script type="application/json" id="{id_}">{txt}</script>'


def plot_figure(d: Docs, rel_path: str, alt: str, plate: str, caption: str, attrs: str = "") -> str:
    w, h = PLOT_SIZE.get(rel_path, (1100, 1150))
    return (f'<figure><div class="frame"><img src="media/{esc(rel_path)}" alt="{esc(alt)}" loading="lazy" '
            f'width="{w}" height="{h}"{attrs}></div><figcaption><span class="plate">{esc(plate)}</span>'
            f'<span>{caption}</span></figcaption></figure>')


PLOT_SIZE: dict[str, tuple[int, int]] = {}


def tells_html(m: dict) -> str:
    t = m["mechanical_tells"]
    chips = []
    for key, lab in TELLS:
        if t[key]:
            chips.append(f'<span class="tell on">{esc(lab)}</span>')
        else:
            chips.append(f'<span class="tell off"><s>{esc(lab)}</s><span class="vh"> (gone)</span></span>')
    n = sum(1 for key, _ in TELLS if t[key])
    return f'<div class="tells" aria-label="{n} of {len(TELLS)} machine tells">{"".join(chips)}</div>'


# ------------------------------------------------------------------------------------------------
# The rung texts (what each rung changes, what it costs, what the numbers show)
# ------------------------------------------------------------------------------------------------
def _per_body(d: Docs, f, speeds=SPEEDS, nd: int = 1, unit: str = "") -> str:
    """"a / b / c mm at 60 cm and ... at 75 cm" for a reading f(body, speed)."""
    parts = []
    for b in BODIES:
        vals = " / ".join(num(f(b, s), nd) for s in speeds)
        parts.append(f"{vals}{(' ' + unit) if unit else ''} at {BODY_SHORT[b]}")
    return " and ".join(parts)


def rung_changes(d: Docs, v: str) -> str:
    m = d.m
    if v == "V0":
        return ("The walk Stage 1 trained on, built for sizing the motors. The hips stay at one height, so the stance "
                "knee bends and straightens in every step just to keep them level. The trunk stays level and rigid "
                "while the body slides sideways. The feet lift and land flat. The swing lasts a fixed share of the "
                "stride, so slow walks swing in slow motion. The head nods twice per stride and the tail swings once, "
                "both on the stride clock, and every stride is an exact copy of the last.")
    if v == "V1":
        tr = [m(b, s, "V1")["bob"]["trough_frac_stance"] for b, s in combos()]
        sp = d.spec("60-N-T-10", "v040", "V1")
        bob = _per_body(d, lambda b, s: m(b, s, "V1")["bob"]["pp_mm"], unit="mm")
        return (f"The hips now dip and rise twice per stride: at today's height at mid-stance and lowest just after "
                f"each touchdown ({span([100 * x for x in tr], 0)} % into the stance), reached by a quick "
                f"{sp['bob_drop']:.1f} s drop that reads as the body taking its weight, then a slow rise. The dip is "
                f"{100 * look.BOB_SHARE:.0f} % of the leg shortening that the flat walk forces on the knee: {bob} "
                "(nominal / 0.40 m/s / animal pace, peak to peak).")
    if v == "V2":
        st = {b: d.spec(b, "nominal", "V2")["swing_time"] for b in BODIES}
        a = {b: m(b, "nominal", "V1")["gait"]["swing_time_s"] for b in BODIES}
        c = {b: m(b, "nominal", "V2")["gait"]["swing_time_s"] for b in BODIES}
        d0 = [m(b, "nominal", "V1")["gait"]["duty"][0] for b in BODIES]
        d1 = [m(b, "nominal", "V2")["gait"]["duty"][0] for b in BODIES]
        dv = max(abs(m(b, "v040", "V2")["gait"]["swing_time_s"] - m(b, "v040", "V1")["gait"]["swing_time_s"])
                 for b in BODIES)
        return (f"Every swing now takes the same time at every speed, the animal pace's {st['60-N-T-10']:.2f} s at "
                f"60 cm and {st['75-M-H-10']:.2f} s at 75 cm, as birds do. A slow walk then spends longer with both "
                f"feet down instead of swinging in slow motion: at the nominal walk the swing shortens from "
                f"{a['60-N-T-10']:.2f} to {c['60-N-T-10']:.2f} s (60 cm) and from {a['75-M-H-10']:.2f} to "
                f"{c['75-M-H-10']:.2f} s (75 cm), and the duty, the share of the stride a foot is down, rises from "
                f"{span(d0, 3)} to {span(d1, 3)}. At 0.40 m/s the swing changes by {1000 * dv:.0f} ms or less; at the "
                "animal pace not at all.")
    if v == "V3":
        sp = d.spec("60-N-T-10", "nominal", "V3")
        lp = [m(b, s, "V3")["gait"]["lift_peak_frac_swing"] for b, s in combos()]
        fast = [(b, s) for b in BODIES for s in ("v040", "fr05")]
        to = [m(b, s, "V3")["toe_off"]["at_lift_off_rad"] for b, s in fast]
        ms = [1000 * m(b, s, "V3")["toe_off"]["ground_roll_s"] for b, s in fast]
        nom = [m(b, "nominal", "V3")["toe_off"]["at_lift_off_rad"] for b in BODIES]
        return (f"The foot now rises quickly after lift-off, highest at {100 * min(lp):.0f} % of the swing instead of "
                f"halfway, and comes down more slowly. Before lift-off the heel pad lifts first and the foot rolls up "
                f"on its toes; in the air it reaches slightly toes-up ({sp['toe_reach']:.3f} rad) before a flat "
                f"landing. The roll reaches {span(nom)} rad by lift-off at the nominal walk, but only {span(to, 3)} "
                f"rad at 0.40 m/s and the animal pace, where the foot is on the floor for just {min(ms):.0f}–"
                f"{max(ms):.0f} ms of its roll.")
    if v == "V4":
        sp = d.spec("60-N-T-10", "nominal", "V4")
        rms = [m(b, s, "V4")["trunk"]["roll_rms_deg"] for b, s in combos()]
        return (f"The body now banks {math.degrees(sp['roll_amp']):.1f}° toward the stance leg as the weight "
                f"shifts across ({span(rms)}° RMS), so the sideways sway reads as a weight shift instead of a "
                "cart sliding. The direction is your call: V4L, priced but not rendered, drops the swing side instead, "
                "as a human pelvis does.")
    if v == "V5":
        sp = d.spec("60-N-T-10", "nominal", "V5")
        cyc = [abs(m(b, s, "V5")["tail"]["phase_from_today_cycles"]) for b, s in combos()]
        share = [m(b, s, "V5")["tail"]["tip_toward_stance_in_swing"] for b, s in combos()]
        tip = "every" if min(share) >= 1.0 else f"{100 * min(share):.0f} % of"
        return (f"The tail swings {sp['tail_amp']:.3f} rad with its timing shifted by {span(cyc)} of a stride, so "
                f"its tip moves toward the stance side in {tip} swing and works against the legs' twist instead of "
                "adding to it (the tail finding, below). The head's metronome nod is gone: the neck cancels the "
                f"trunk's pitch and lets the head ride the bob, with a small {sp['head_settle']:.3f} rad dip "
                f"{sp['settle_lag']:.3f} s after each touchdown.")
    if v == "V6":
        sp = d.spec("60-N-T-10", "v040", "V6")
        b, s = "75-M-H-10", "fr05"
        s5, s6 = d.spec(b, s, "V5"), d.spec(b, s, "V6")
        c5, c6 = m(b, s, "V5")["gait"]["cadence_steps_s"], m(b, s, "V6")["gait"]["cadence_steps_s"]
        lo, hi = sp["var_band"]
        return (f"No two strides are twins: step times vary by about {100 * sp['phase_jitter']:.1f} % and the two "
                f"sides differ by {100 * sp['asym']:.1f} %, while the gaze ({sp['gaze_var']:.3f} rad) and the tail's "
                f"swing (±{100 * sp['tail_var']:.0f} %) drift slowly, over {1 / hi:.0f}–{1 / lo:.0f} s. "
                f"At 75 cm's animal pace V6 also takes a longer stride (F13: {s6['stride_rel']:.3f} H instead of "
                f"{s5['stride_rel']:.2f} H), which slows the cadence from {c5:.2f} to {c6:.2f} steps per second.")
    raise KeyError(v)


def rung_cost(d: Docs, v: str) -> str:
    s_ = d.s
    if v == "V0":
        g = {b: " / ".join(ratio(gov(s_(b, s, "V0"))[2]) for s in SPEEDS) for b in BODIES}
        over: dict[tuple, list] = {}
        for b, s in combos():
            for part, kind in PASSIVE:
                x = passive(s_(b, s, "V0"), part, kind)
                if x >= 1.0:
                    over.setdefault((b, s, part), []).append(f"{KIND_SHORT[kind]} {ratio(x)}")
        ov = "; ".join(f"the {BODY_SHORT[b]} {part} servo at the {SPEED_SHORT[s]} ({', '.join(xs)})"
                       for (b, s, part), xs in over.items())
        return (f"The baseline every other rung is priced against. The busiest motor runs at {g['60-N-T-10']} of its "
                f"limit at 60 cm and {g['75-M-H-10']} at 75 cm (nominal / 0.40 m/s / animal pace), all within. "
                + (f"Counting the joints' friction, which Stage 0 leaves out, today's walk already takes {ov} over its "
                   "limit." if over else ""))
    if v == "V1":
        dm = [rel(mu(s_(b, s, "V0")), mu(s_(b, s, "V1"))) for b, s in combos()]
        acc = {b: (s_(b, "v040", "V0")["accel"]["com_vertical_max"], s_(b, "v040", "V1")["accel"]["com_vertical_max"])
               for b in BODIES}
        n1 = {s: passive(s_("60-N-T-10", s, "V1"), "neck", "peak") for s in ("v040", "fr05")}
        n5 = {s: passive(s_("60-N-T-10", s, "V5"), "neck", "peak") for s in ("v040", "fr05")}
        return (f"No limit broken. The grip the standing foot needs falls {pct_span(dm)}. The body's upward "
                f"acceleration peaks rise (0.40 m/s: {num(acc['60-N-T-10'][0], 1)} to {num(acc['60-N-T-10'][1], 1)} "
                f"m/s² at 60 cm, {num(acc['75-M-H-10'][0], 1)} to {num(acc['75-M-H-10'][1], 1)} at 75 cm), "
                "carried by the stance legs. Counting the joints' friction, today's head nod riding on the bob takes "
                f"the 60 cm neck servo past its planning peak at 0.40 m/s and the animal pace ({ratio(n1['v040'])} and "
                f"{ratio(n1['fr05'])}); V5's steady head brings it back to {ratio(n5['v040'])} and {ratio(n5['fr05'])}.")
    if v == "V2":
        k = {b: (s_(b, "nominal", "V1")["drives"]["knee"]["speed_ratio"],
                 s_(b, "nominal", "V2")["drives"]["knee"]["speed_ratio"]) for b in BODIES}
        g = {b: (mu(s_(b, "nominal", "V1")), mu(s_(b, "nominal", "V2"))) for b in BODIES}
        kv = {b: s_(b, "v040", "V2")["drives"]["knee"]["speed_ratio"] - s_(b, "v040", "V1")["drives"]["knee"]["speed_ratio"]
              for b in BODIES}
        floor = min(d.screen["grips"])
        gs = "; ".join(f"{BODY_SHORT[b]}: {num(g[b][0])} to {num(g[b][1])}"
                       + (f", above the {floor} that most demo floors give" if g[b][1] > floor else "") for b in BODIES)
        return (f"No limit broken, but at the nominal walk the quicker swing moves the knee faster: "
                f"{ratio(k['60-N-T-10'][0])} to {ratio(k['60-N-T-10'][1])} of its speed limit at 60 cm and "
                f"{ratio(k['75-M-H-10'][0])} to {ratio(k['75-M-H-10'][1])} at 75 cm. It also accelerates the leg "
                f"harder at lift-off, so the grip the standing foot needs there rises "
                f"{100 * rel(*g['60-N-T-10']):.0f} % and {100 * rel(*g['75-M-H-10']):.0f} % ({gs}). At 0.40 m/s the "
                "knee moves "
                f"{num(kv['60-N-T-10'], 2, True)} / {num(kv['75-M-H-10'], 2, True)}; at the animal pace nothing "
                "changes. At the nominal walk this rung needs decision 3, the duty band.")
    if v == "V3":
        hp = (s_("60-N-T-10", "v040", "V2")["drives"]["hip_pitch"]["speed_ratio"],
              s_("60-N-T-10", "v040", "V3")["drives"]["hip_pitch"]["speed_ratio"])
        twd = max(worst(s_(b, s, "V3")["drives"]["hip_yaw"])[1] - worst(s_(b, s, "V2")["drives"]["hip_yaw"])[1]
                  for b, s in combos())
        z0 = [s_(b, s, "V2")["contact"]["zmp_margin_min_mm"] for b, s in combos()]
        z1 = [s_(b, s, "V3")["contact"]["zmp_margin_min_mm"] for b, s in combos()]
        gn = [rel(mu(s_(b, "nominal", "V2")), mu(s_(b, "nominal", "V3"))) for b in BODIES]
        gf = [rel(mu(s_(b, s, "V2")), mu(s_(b, s, "V3"))) for b in BODIES for s in ("v040", "fr05")]
        cl = [s_(b, s, "V3")["clearance"]["mid_swing_min_frac_H"] for b, s in combos()]
        t75 = (passive(s_("75-M-H-10", "v040", "V2"), "tail", "peak"), passive(s_("75-M-H-10", "v040", "V3"), "tail", "peak"))
        return (f"No limit broken. The hips swing more gently (hip pitch speed {ratio(hp[0])} to {ratio(hp[1])} at "
                f"60 cm, 0.40 m/s) while the thigh twist works harder (its ratio up by at most {num(twd)}). During "
                "the roll the "
                f"foot stands on its toe line alone: the balance point stays {span(z1, 1)} mm inside it, where it "
                f"stayed {span(z0, 0)} mm inside the pad before, so a controller will need to model the rolling "
                f"contact. The grip need falls {pct_span(gn)} at the nominal walk but rises {pct_span(gf)} at the "
                f"faster walks. The toes clear the floor by {span(cl, 3)} H in mid-swing (gate 0.04 H). Counting the "
                f"joints' friction, the 75 cm tail servo's peak reaches {ratio(t75[1])} at 0.40 m/s (from "
                f"{ratio(t75[0])}) until V5 fixes the tail's timing.")
    if v == "V4":
        mv = moves(d, "V4")
        dm = [abs(rel(mu(s_(b, s, "V3")), mu(s_(b, s, "V4")))) for b, s in combos()]
        tj = [s_(b, s, "V4")["clearance"]["tightest_joint"][1] for b, s in combos()]
        return ("Nothing measurable: " + ("no motor's ratio moves by 0.05 or more" if not mv else
                                          f"{len(mv)} ratios move by 0.05 or more")
                + f", and the grip need moves by under {100 * max(dm) + 0.05:.1f} %. The roll uses a little hip-roll "
                f"range; the tightest joint keeps {span(tj, 2)} rad inside its usable range (gate 0.10 rad).")
    if v == "V5":
        tc = [(passive(s_(b, s, "V4"), "tail", "cont"), passive(s_(b, s, "V5"), "tail", "cont")) for b, s in combos()]
        n = (passive(s_("60-N-T-10", "v040", "V4"), "neck", "peak"), passive(s_("60-N-T-10", "v040", "V5"), "neck", "peak"))
        dz = [rel(twist(s_(b, s, "V4")), twist(s_(b, s, "V5"))) for b, s in combos()]
        dm = [rel(mu(s_(b, s, "V4")), mu(s_(b, s, "V5"))) for b, s in combos()]
        return (f"It frees load. Counting the joints' friction, the tail servo's sustained load falls from "
                f"{span([a for a, _ in tc])} to {span([c for _, c in tc])} of its limit: the body's sway now swings "
                f"the tail and the servo only brakes it. The 60 cm neck servo's peak falls back under its limit "
                f"({ratio(n[0])} to {ratio(n[1])} at 0.40 m/s). The standing foot's peak twist falls {pct_span(dz)} "
                f"and its grip need {pct_span(dm)}.")
    if v == "V6":
        b, s = "75-M-H-10", "fr05"
        k60 = [s_(bb, ss, "V6")["drives"]["knee"]["speed_ratio"] - s_(bb, ss, "V5")["drives"]["knee"]["speed_ratio"]
               for bb in ("60-N-T-10",) for ss in SPEEDS]
        g60 = [(mu(s_("60-N-T-10", ss, "V5")), mu(s_("60-N-T-10", ss, "V6"))) for ss in SPEEDS]
        r6 = s_(b, s, "V6")
        cause = r6["cause"]["knee speed"]
        k5 = s_(b, s, "V5")["drives"]["knee"]["speed_ratio"]
        k13 = s_(b, s, "V5F13")["drives"]["knee"]["speed_ratio"]
        k6 = r6["drives"]["knee"]["speed_ratio"]
        floor = min(d.screen["grips"])
        nom = g60[0]
        ab = f" ({num(nom[1])} at the nominal walk, above the {floor} floor)" if nom[1] > floor else ""
        return (f"At 60 cm nothing new: the knee's speed rises by at most {num(max(k60))} and the grip need by up to "
                f"{max(100 * rel(a, c) for a, c in g60):.0f} %{ab}. At 75 cm's animal pace V6 is over a limit: the "
                f"knee reaches {ratio(k6)} of its speed limit on the shortest swing ({cause['fastest_swing_s']:.3f} s). "
                f"F13's longer stride alone takes it from {ratio(k5)} to {ratio(k13)} (V5F13, within); F14's "
                "shorter steps do the rest. Under the plan's rule V6 keeps its features and the 75 cm build loses the "
                "animal-pace rung unless you decide otherwise (decision 2).")
    raise KeyError(v)


def rung_numbers(d: Docs, v: str) -> str:
    m = d.m
    if v == "V0":
        b6 = {b: m(b, "fr05", "V0") for b in BODIES}
        hp = [m(b, s, "V0")["head"]["world_pitch_rms_deg"] for b, s in combos()]
        return (f"All four machine tells. The head pitches {span(hp)}° RMS in the world (the closed-loop band is "
                f"1.5°). At the animal pace the cadence is {b6['60-N-T-10']['gait']['cadence_steps_s']:.2f} and "
                f"{b6['75-M-H-10']['gait']['cadence_steps_s']:.2f} steps/s at 60 and 75 cm; a T. rex scaled to the same "
                f"size would take {b6['60-N-T-10']['bird_curves']['trex_cadence_steps_s']:.2f} and "
                f"{b6['75-M-H-10']['bird_curves']['trex_cadence_steps_s']:.2f}.")
    if v == "V1":
        ch = [m(b, s, x)["bob"]["crest_hold_share"] for b, s in combos() for x in LADDER[1:]]
        sinus = m("60-N-T-10", "v040", "V1")["bob"]["crest_hold_share_sinusoid"]
        return (f"Hips no longer at one height. The hips stay within 5 % of the crest for {span(ch)} of the time (a "
                f"smooth up-and-down would: {sinus:.2f}), so in the path plots the hip and snout run level for long "
                "stretches: straight segments, which the research reads as mechanical. A rounder bob is a small change "
                "if you want it.")
    if v == "V2":
        dn = [m(b, "nominal", "V2")["gait"]["duty"][0] for b in BODIES]
        dv = [m(b, "v040", "V2")["gait"]["duty"][0] for b in BODIES]
        bc = m("60-N-T-10", "nominal", "V2")["bird_curves"]
        return (f"Duty {span(dn, 3)} at the nominal walk, outside the behaviour spec's 0.55–0.65 band (tinamous: "
                f"{bc['tinamou_duty_band'][0]:.2f}–{bc['tinamou_duty_band'][1]:.2f}); {span(dv, 3)} at 0.40 m/s, "
                "inside it.")
    if v == "V3":
        fa = {s: [(d.s(b, s, "V2")["accel"]["foot_vertical_peak"], d.s(b, s, "V3")["accel"]["foot_vertical_peak"])
                  for b in BODIES] for s in SPEEDS}
        a0 = [a for s in SPEEDS for a, _ in fa[s]]
        a1 = [c for s in SPEEDS for _, c in fa[s]]
        return (f"Flat feet gone. The feet's peak vertical acceleration rises from {span(a0, 0)} to {span(a1, 0)} "
                "m/s², the cue the eye's “life detector” responds to. The roll reaches the research's "
                "0.3–0.5 rad only at the nominal walk (decision list: the claw question and the faster walks).")
    if v == "V4":
        sw = {s: [m(b, s, "V4")["sway"]["com_pp_per_stride_max_H"] for b in BODIES] for s in SPEEDS}
        return (f"Trunk roll {span([m(b, s, 'V4')['trunk']['roll_rms_deg'] for b, s in combos()])}° RMS (the "
                f"spec allows up to 4°). The sway itself is unchanged: {span(sw['nominal'], 3)} H at the nominal "
                f"walk and {span(sw['v040'] + sw['fr05'], 3)} H faster (the spec asks for 0.12 H).")
    if v == "V5":
        h4 = [m(b, s, "V4")["head"]["world_pitch_rms_deg"] for b, s in combos()]
        h5 = [m(b, s, "V5")["head"]["world_pitch_rms_deg"] for b, s in combos()]
        return (f"Metronome nod gone. Head pitch in the world {span(h4)} to {span(h5)}° RMS (band 1.5°). The "
                "trunk never pitches (F8), so the steady head shows only as the nod removed; it follows the trunk's "
                "roll, as there is no neck roll joint.")
    if v == "V6":
        cv = [100 * m(b, s, "V6")["step_times"]["cv"] for b, s in combos()]
        h6 = [m(b, s, "V6")["head"]["world_pitch_rms_deg"] for b, s in combos()]
        x = m("75-M-H-10", "fr05", "V6")
        return (f"Identical steps gone: none of the four tells is left. Step times vary {span(cv)} % (V5: 0); the "
                f"gaze drift puts the head at {span(h6)}° RMS. At 75 cm's animal pace: stride "
                f"{x['gait']['stride_H']:.3f} H, {x['gait']['cadence_steps_s']:.2f} steps/s (a T. rex at this size: "
                f"{x['bird_curves']['trex_cadence_steps_s']:.2f} steps/s with 1.25 H strides).")
    raise KeyError(v)


def moves_table(d: Docs, v: str, base: str | None = None) -> str:
    mv = moves(d, v, base)
    if not mv:
        return '<p class="cap">No ratio moves by 0.05 or more at any body or speed.</p>'
    head = "".join(f'<th class="num">{esc(combo_label(b, s))}</th>' for b, s in combos())
    rows = []
    for lab, vals in mv:
        cells = []
        for b, s in combos():
            a, c = vals[(b, s)]
            big = abs(c - a) >= MOVE - 1e-12
            txt = f"{ratio(a)} → {ratio(c)}"
            cls = "num" + (" mv" if big else " still")
            over = " ov" if c >= 1.0 else ""
            cells.append(f'<td class="{cls}{over}">{txt}</td>')
        rows.append(f"<tr><th scope=\"row\">{esc(lab)}</th>{''.join(cells)}</tr>")
    base = base or prev_rung(v)
    return (f'<div class="tablewrap"><table class="moves"><caption class="vh">Ratios that move from {base} to {v}'
            f'</caption><thead><tr><th>Ratio of its limit, {base} → {v}</th>{head}</tr></thead><tbody>'
            f'{"".join(rows)}</tbody></table></div>')


def rung_row(d: Docs, v: str) -> str:
    feats = RUNG_ADDS[v]
    adds = ", ".join(feats) if feats else "nothing; the reference"
    if v == "V6":
        adds += " (and F13 at 75 cm's animal pace)"
    tells = tells_html(d.m("60-N-T-10", LADDER_SPEED, v))
    base = "V5" if v == "V6" else None
    det = ""
    if v != "V0":
        det = (f'<details class="mini"><summary>Every ratio that moves by 0.05 or more ({prev_rung(v) if base is None else base}'
               f' → {v})</summary>{moves_table(d, v, base)}</details>')
    return (f'<article class="rung" id="rung-{v}"><div class="rhead"><span class="badge">{v}</span>'
            f'<h3>{esc(RUNG_TITLE[v])}</h3><p class="adds">Adds {esc(adds)}</p>{tells}</div>'
            f'<div><span class="lbl">What changes</span><p>{rung_changes(d, v)}</p></div>'
            f'<div><span class="lbl">What it costs</span><p>{rung_cost(d, v)}</p></div>'
            f'<div><span class="lbl">What the numbers show</span><p>{rung_numbers(d, v)}</p></div>'
            f'{det}</article>')


# ------------------------------------------------------------------------------------------------
# Sections
# ------------------------------------------------------------------------------------------------
def header(d: Docs) -> str:
    over = [(b, s, v) for b in BODIES for s in SPEEDS for v in LADDER if not d.s(b, s, v)["within_limits"]]
    n_walks = len(BODIES) * len(SPEEDS) * len(LADDER)
    tp = d.screen["tail_phase"]
    dz = [tp[b][s]["changes"]["V0_phase_today_to_F9"]["mz_abs_max_nm"]["rel"] for b, s in combos()]
    cad = {b: d.m(b, "fr05", "V6")["gait"]["cadence_steps_s"] for b in BODIES}
    trex = {b: d.m(b, "fr05", "V6")["bird_curves"]["trex_cadence_steps_s"] for b in BODIES}
    ov = "; ".join(f"{BODY_SHORT[b]} {v} at the {SPEED_SHORT[s]}: "
                   + ", ".join(f"{o['limit']} {ratio(o['value'])}" for o in d.s(b, s, v)["over"]) for b, s, v in over)
    return f"""
<header class="top" data-only="owner">
  <p class="kicker">Mesozoic Labs · robotic <i>T. rex</i> · Stage 2 look review, gate L1 · {DATE}</p>
  <h1>Pick the features that make it walk like an animal</h1>
  <p class="lede">Today's reference walk was built to size the motors, and it moves like a humanoid robot: hips at a
  fixed height, a rigid trunk sliding sideways, flat feet, and a head and tail on a metronome. The research note
  proposed fixes taken from animals and from animation. Here they are, added one group at a time to the 60 cm
  Tapered and the 75 cm Hybrid at three speeds, each priced against the motors' limits before it was rendered. You
  pick which to keep; the walking controller is trained on your pick afterwards.</p>
  <div class="facts">
    <div class="fact"><b>{n_walks} walks</b><span>V0 (today) to V6 (everything), on {len(BODIES)} bodies at
    {len(SPEEDS)} speeds; one feature group added at each step</span></div>
    <div class="fact"><b>{len(over)} over a limit</b><span>{esc(ov) if over else "none"}; under the plan's rule it
    keeps its features and loses that speed rung unless you decide otherwise</span></div>
    <div class="fact"><b>{pct_span(dz)} less twist</b><span>on the standing foot when the tail swings the other way
    (F9): today's tail adds to the legs' twist</span></div>
    <div class="fact"><b>{cad['60-N-T-10']:.2f} / {cad['75-M-H-10']:.2f} steps/s</b><span>V6 at the animal pace (60 /
    75 cm); a <i>T. rex</i> scaled to the same size: {trex['60-N-T-10']:.2f} / {trex['75-M-H-10']:.2f}</span></div>
    <div class="fact"><b>3 decisions</b><span>which features to keep; whether a feature may cost a speed rung; the
    duty band for F3</span></div>
  </div>
  <nav class="toc" aria-label="Sections">
    <a href="#what">What these are</a><a href="#ladder">The ladder</a><a href="#b60">60 cm Tapered</a>
    <a href="#b75">75 cm Hybrid</a><a href="#tail">The tail finding</a><a href="#decide">Your decisions</a>
    <a href="#rate">Rate the walks</a>
  </nav>
</header>
<header class="top" data-only="viewers">
  <p class="kicker">Robotic <i>T. rex</i> · walking clips</p>
  <h1>Which walk looks alive?</h1>
  <p class="lede">Four short clips of a small robot dinosaur walking, in two pairs. Watch each, score it on the
  sheet below, and say which clip of each pair looks more like a living animal. It takes about fifteen minutes, and
  there are no right answers.</p>
</header>"""


def section_what(d: Docs) -> str:
    sp = {b: {s: d.m(b, s, "V0")["gait"]["speed_m_s"] for s in SPEEDS} for b in BODIES}
    H = {b: d.m(b, "nominal", "V0")["H_m"] for b in BODIES}
    v4 = d.screen["v040"]
    cam = d.clips["cameras"]
    rows = []
    for b in BODIES:
        x = v4[b]
        rows.append([esc(BODY_NAME[b]), f"{x['brisk']['speed_m_s']:.2f} / {x['fr05']['speed_m_s']:.2f}",
                     f"{x['weight_w']:.3f}", f"{x['period_s']:.3f}", f"{x['stride_rel']:.3f}", f"{x['duty']:.3f}",
                     f"{x['duty_with_F3']:.3f}", f"{x['lift_frac']:.2f}"])
    v040_table = K.table(["Body", "Brisk / animal pace (m/s)", "Weight w", "Period (s)", "Stride (H)", "Duty",
                          "Duty with F3", "Lift (H)"], rows, num={1, 2, 3, 4, 5, 6, 7})
    r2 = d.screen["provenance"]["git"]["commit"][:7]
    return f"""
<section id="what" data-only="owner">
  <p class="kicker">Before you watch</p><h2>What these clips are, and what to look for</h2>
  <div class="grid2">
    <div class="prose">
      <p><b>These are the walks a controller would learn to follow, not the robot walking.</b> Each clip plays a
      reference walk back with perfect tracking: no physics, no balance controller, no servo delay. The real robot
      will fall short of them. How far is the next stage's question: a controller trained on the walks you pick here,
      then closed-loop clips for a second look (gate L1b). So judge the motion's character here, not whether it would
      stay up.</p>
      <p>Each rung of the ladder adds one group of features from the research note, so you can see what each buys.
      V0 is today's walk; V6 has everything. Every rung was priced by the Stage 0 screen before it was rendered: the
      torques and speeds the walk asks of each motor on the round-two model, against that motor's limits. Those
      prices are shown as recorded (screen at commit <code>{r2}</code>); none was revised after the renders. No rung
      uses the tuned walk of the earlier gait study (F25): every rung is built on today's walks.</p>
    </div>
    <div class="cards">
      <div class="card"><h3>Two cameras, real time</h3><p>The <b>visitor's view</b> sees the robot from
      {cam['visitor']['height_m']:g} m up, {cam['visitor']['distance_m']:g} m away, three-quarter front: how an adult
      standing nearby sees the back line, the sway and the tail. The <b>side view</b>, from
      {cam['side']['height_m']:g} m up, is a child's or a photographer's view of the feet and knees. Both follow the
      body's average forward motion, so the floor slides past. Clips are 10 s at real time; use Half speed to study
      a detail.</p></div>
      <div class="card"><h3>Point-light clips</h3><p>Only {len(d.clips['points'])} dots on black: snout, eye, neck
      base, hips, knees, the heel pads, toe tips, mid-tail and tail tip. People read life from moving dots alone, so
      these judge the motion without the covers.</p></div>
      <div class="card"><h3>Three speeds</h3><p>The <b>nominal walk</b> (Froude 0.012: {sp['60-N-T-10']['nominal']:.2f}
      m/s at 60 cm, {sp['75-M-H-10']['nominal']:.2f} m/s at 75 cm), <b>0.40 m/s</b> (the agility bar's walking speed)
      and the <b>animal pace</b> (Froude 0.05: {sp['60-N-T-10']['fr05']:.2f} and {sp['75-M-H-10']['fr05']:.2f} m/s,
      close to a <i>T. rex</i>'s estimated preferred speed). The Froude number scales speed to leg length, so both
      sizes walk alike.</p></div>
    </div>
  </div>
  <h3>What to look for</h3>
  <ul class="look">
    <li><b>Feet.</b> Do they peel off from the toes and land quietly, or lift and land flat? (F4, F5)</li>
    <li><b>Knees and hocks.</b> Do the legs fold and reach smoothly, and does the stance leg carry the body? (F1,
    F3)</li>
    <li><b>Body.</b> Does it rise and fall with the steps and settle after each footfall, or glide at a fixed height and
    slide sideways? (F1, F2, F6)</li>
    <li><b>Head and tail.</b> Is the head steady and the tail working against the legs, or do they swing like
    metronomes? (F9, F10)</li>
    <li><b>Smoothness.</b> Any jitter, foot sliding or stutter? A reference should have none.</li>
    <li><b>Timing.</b> Are the strides identical twins, or do they vary a little? (F14)</li>
    <li><b>Overall.</b> A machine walking, or an animal walking?</li>
  </ul>
  <details>
    <summary>How to read the numbers</summary>
    <dl class="gloss">
      <dt>H</dt><dd>Hip height: {H['60-N-T-10']:.3f} m on the 60 cm body, {H['75-M-H-10']:.3f} m on the 75 cm body.
      Lengths such as a 0.80 H stride are fractions of it.</dd>
      <dt>Ratio of its limit</dt><dd>A motor's load over its limit, with the design margins (1.15 on speed, 1.25 on
      peak torque). Under 1 is within the limit, over 1 is over it, and the screen calls 0.98 or more "near". The
      busiest motor's ratio is the walk's governing ratio. Ratios from 0.95 up carry a third decimal, where it
      decides.</dd>
      <dt>With joint friction</dt><dd>The simulation gives the neck and tail joints damping and friction that the
      Stage 0 screen leaves out. Those readings matter for the small neck and tail servos, but under the plan's rule
      they cost no speed rung.</dd>
      <dt>Sustained, peak</dt><dd>Sustained torque is the average load against the motor's planning limit (A1, 0.57
      of the vendor's rating); peak is the largest instant.</dd>
      <dt>Grip need</dt><dd>The friction coefficient the standing foot needs so it does not twist or slip while
      walking straight. The plan judges two floors: 0.9 (tile, the grippiest wood) and 0.54 (most demo floors).</dd>
      <dt>Twist</dt><dd>The turning moment about the vertical that the standing foot holds against the floor (N m).</dd>
      <dt>Duty</dt><dd>The share of the stride a foot is on the floor.</dd>
    </dl>
  </details>
  <details>
    <summary>How 0.40 m/s was defined</summary>
    <div class="prose"><p>0.40 m/s lies between the brisk walk and the animal pace on both bodies. Its walk is the
    one Stage 1's brisk-to-animal-pace speed change already passes through at 0.40 m/s: the period, duty and lift of
    the two walks mixed with the speed's weight w = (0.40 − brisk) / (animal pace − brisk), as
    <code>transitions.py</code> mixes them. It is a walk the study already uses, between two screened walks, not a
    new design.</p></div>
    {v040_table}
  </details>
</section>"""


def matrix(d: Docs) -> str:
    head1 = "".join(f'<th colspan="3" class="grp">{esc(BODY_NAME[b])}</th>' for b in BODIES)
    head2 = "".join(('<th class="bl">' if s == SPEEDS[0] else "<th>") + f"{esc(SPEED_TAB[s])}</th>"
                    for _ in BODIES for s in SPEEDS)
    rows = []
    for v in LADDER:
        cells = []
        for b in BODIES:
            for s in SPEEDS:
                r = d.s(b, s, v)
                g, k, x = gov(r)
                state = "over" if x >= 1.0 else ("near" if x >= NEAR else "")
                tag = K.tag(state, "bad" if state == "over" else "warn") if state else ""
                w = min(100.0, 100.0 * x / 1.1)
                tip = f"{ratio(x)} of its limit|{BODY_NAME[b]}, {SPEED_SHORT[s]}, {v}|busiest: {GROUP[g]} {KIND[k]}"
                cells.append(f'<td class="m{" bl" if s == SPEEDS[0] else ""}" data-tip="{esc(tip)}"><div class="v">'
                             f'<b>{ratio(x)}</b><span>{esc(GROUP[g])} {esc(KIND_SHORT[k])}</span>{tag}</div>'
                             f'<div class="meter" aria-hidden="true"><i class="{state}" style="width:{w:.1f}%"></i>'
                             '<b style="left:90.9%"></b></div></td>')
        rows.append(f'<tr><th scope="row"><span class="badge sm">{v}</span></th>{"".join(cells)}</tr>')
    return (f'<div class="tablewrap"><table class="matrix"><caption class="vh">The busiest motor on every walk, as a '
            f'ratio of its limit</caption><thead><tr><th rowspan="2">Rung</th>{head1}</tr><tr>{head2}</tr></thead>'
            f'<tbody>{"".join(rows)}</tbody></table></div>')


def extras_table(d: Docs) -> str:
    s_, m_ = d.s, d.m
    b6, b7 = "60-N-T-10", "75-M-H-10"
    rows = []
    n = {x: passive(s_(b6, "v040", x), "neck", "peak") for x in ("V1B50", "V1", "V1B100")}
    bob_mm = {x: m_(b6, "v040", x)["bob"]["pp_mm"] for x in ("V1B50", "V1", "V1B100")}
    rows.append(["V1B50, V1B100", "F1's bob at 50 % and 100 % of the shortening the flat walk forces (V1: 75 %)",
                 f"60 cm, 0.40 m/s: bob {num(bob_mm['V1B50'], 1)} / {num(bob_mm['V1'], 1)} / "
                 f"{num(bob_mm['V1B100'], 1)} mm; the neck servo's peak with joint friction {ratio(n['V1B50'])} / "
                 f"{ratio(n['V1'])} / {ratio(n['V1B100'])} (50 / 75 / 100 %). No Stage 0 limit at any size."])
    k3 = {b: s_(b, "nominal", "V3T")["drives"]["knee"]["speed_ratio"] for b in BODIES}
    rows.append(["V3T", "F5's toe-off at 0.40 rad by lift-off, the research's middle (V3: 0.30)",
                 f"The knee's speed at the nominal walk reaches {ratio(k3[b6])} (60 cm) and {ratio(k3[b7])} (75 cm) of "
                 "its limit: near. That is why the ladder uses 0.30 rad."])
    tj4 = s_(b6, "nominal", "V4")["clearance"]["tightest_joint"][1]
    tjl = s_(b6, "nominal", "V4L")["clearance"]["tightest_joint"][1]
    tc4 = passive(s_(b6, "nominal", "V4"), "tail", "cont")
    tcl = passive(s_(b6, "nominal", "V4L"), "tail", "cont")
    rows.append(["V4L", "F6's roll the other way: the swing side drops, as a human pelvis does",
                 f"Same motor costs as V4; the tightest joint (hip roll) keeps {tjl:.3f} rad of range against V4's "
                 f"{tj4:.3f} (gate 0.10); the 60 cm tail servo's sustained load with friction {ratio(tcl)} against "
                 f"{ratio(tc4)} at the nominal walk."])
    k5 = s_(b7, "fr05", "V5")["drives"]["knee"]["speed_ratio"]
    k13 = s_(b7, "fr05", "V5F13")["drives"]["knee"]["speed_ratio"]
    c13 = m_(b7, "fr05", "V5F13")["gait"]["cadence_steps_s"]
    rows.append(["V5F13", "75 cm, animal pace only: V5 with F13's longer stride, without F14",
                 f"Knee speed {ratio(k5)} to {ratio(k13)}, within; cadence {c13:.2f} steps/s. The stride alone fits; "
                 "F14 on top of it does not (V6)."])
    sw6 = {s: (m_(b6, s, "V6")["sway"]["com_pp_per_stride_max_H"], m_(b6, s, "V6S")["sway"]["com_pp_per_stride_max_H"])
           for s in SPEEDS}
    cop = {b: (s_(b, "nominal", "V6")["contact"]["single_support_cop_margin_min_mm"],
               s_(b, "nominal", "V6S")["contact"]["single_support_cop_margin_min_mm"]) for b in BODIES}
    o6s = s_(b7, "fr05", "V6S")
    zin = d.spec(b6, "nominal", "V6S")["zmp_inset_y"]
    rows.append(["V6S", f"F7, optional: V6 with the single-support balance point moved {1000 * zin:.0f} mm inward, for "
                 "less sway",
                 f"Sway {sw6['nominal'][0]:.3f} to {sw6['nominal'][1]:.3f} H at the nominal walk and "
                 f"{sw6['v040'][0]:.3f} to {sw6['v040'][1]:.3f} H at 0.40 m/s (60 cm; spec 0.12 H); the "
                 "single-support pressure centre's margin inside the pad falls from "
                 f"{cop[b6][0]:.0f} to {cop[b6][1]:.0f} mm (60 cm) and {cop[b7][0]:.0f} to {cop[b7][1]:.0f} mm (75 cm), "
                 f"the edge Stage 1 saw before every fall. Over at 75 cm's animal pace like V6 ("
                 + ", ".join(f"{o['limit']} {ratio(o['value'])}" for o in o6s["over"]) + ")."])
    return K.table(["Variant", "What it tests", "What the screen shows"], rows)


def section_ladder(d: Docs) -> str:
    rungs = "".join(rung_row(d, v) for v in LADDER)
    return f"""
<section id="ladder" data-only="owner">
  <p class="kicker">The ladder</p><h2>Seven walks, one feature group at a time</h2>
  <p class="prose-w">Each rung keeps everything before it and adds one group, so the difference between neighbours is
  what that group buys. The chips under each rung name the machine tells the research lists (a constant hip height,
  flat feet at lift-off, identical steps, a metronome nod) that are still present, and strike out the ones gone.
  "What it costs" compares each rung with the one before; open the table under it for every ratio that moves.</p>
  <div class="rungs">{rungs}</div>
  <h3>The busiest motor on every walk</h3>
  <p class="prose-w">The governing ratio: the busiest motor's load as a share of its limit (the tick marks 1.0).
  Under the plan's rule a walk over 1 keeps its features and loses that speed rung unless you decide otherwise.</p>
  {matrix(d)}
  <h3>Variants priced but not rendered</h3>
  <p class="prose-w">The screen also priced six side variants that answer smaller questions. They have no clips; say
  if you want any rendered.</p>
  {extras_table(d)}
</section>"""


def _cmp_slots(d: Docs, b: str, s: str, mode: str = "covered") -> list[dict]:
    return [slot_data(d, b, s, v, view, mode) for view in ("visitor", "side") for v in ("V0", "V6")]


def v0v6_table(d: Docs, b: str) -> str:
    m = d.m
    rows_def = [
        ("Hip bob, peak to peak (mm)", lambda x, r: num(x["bob"]["pp_mm"], 1)),
        ("Trunk roll (° RMS)", lambda x, r: num(x["trunk"]["roll_rms_deg"], 2)),
        ("Sideways sway (H per stride; spec 0.12)", lambda x, r: num(x["sway"]["com_pp_per_stride_max_H"], 3)),
        ("Head pitch in the world (° RMS; band 1.5)", lambda x, r: num(x["head"]["world_pitch_rms_deg"], 2)),
        ("Tail timing (stride from today's)", lambda x, r: num(x["tail"]["phase_from_today_cycles"], 2, True)
         if abs(x["tail"]["phase_from_today_cycles"]) > 5e-4 else "0"),
        ("Toe-off at lift-off (rad)", lambda x, r: num(max(0.0, x["toe_off"]["at_lift_off_rad"]), 3)),
        ("Swing time (s)", lambda x, r: num(x["gait"]["swing_time_s"], 3)),
        ("Duty", lambda x, r: num(x["gait"]["duty"][0], 3)),
        ("Stride (H)", lambda x, r: num(x["gait"]["stride_H"], 3)),
        ("Cadence (steps/s)", lambda x, r: num(x["gait"]["cadence_steps_s"], 2)),
        ("Step-time variation (%)", lambda x, r: num(100 * x["step_times"]["cv"], 2)),
        ("Standing foot's peak twist (N m)", lambda x, r: num(twist(r), 2)),
        ("Grip need", lambda x, r: num(mu(r), 3)),
        ("Busiest motor (ratio of its limit)", lambda x, r: f"{ratio(gov(r)[2])} {GROUP[gov(r)[0]]} {KIND_SHORT[gov(r)[1]]}"),
        ("Stage 0", lambda x, r: K.tag("within", "ok") if r["within_limits"] else K.tag("over", "bad")),
    ]
    head1 = "".join(f'<th colspan="2" class="grp" data-col="{s}">{esc(SPEED_TAB[s])}</th>' for s in SPEEDS)
    head2 = "".join(f'<th class="num{" bl" if v == "V0" else ""}" data-col="{s}">{v}</th>' for s in SPEEDS
                    for v in ("V0", "V6"))
    trs = []
    for lab, f in rows_def:
        tds = "".join(f'<td class="num{" bl" if v == "V0" else ""}" data-col="{s}">{f(m(b, s, v), d.s(b, s, v))}</td>'
                      for s in SPEEDS for v in ("V0", "V6"))
        trs.append(f'<tr><th scope="row">{esc(lab)}</th>{tds}</tr>')
    x = m(b, "fr05", "V0")["bird_curves"]
    note = (f'A <i>T. rex</i> scaled to this size would walk the animal pace at {x["trex_cadence_steps_s"]:.2f} steps/s '
            f'with {x["trex_stride_H"]:.2f} H strides. Tail timing: the shift of the tail\'s swing from today\'s, as a '
            'share of a stride (half a stride flips it).')
    return (f'<div class="tablewrap"><table class="v06"><thead><tr><th rowspan="2">Reading</th>{head1}</tr>'
            f'<tr>{head2}</tr></thead><tbody>{"".join(trs)}</tbody></table></div><p class="cap">{note}</p>')


def compare_block(d: Docs, b: str) -> str:
    bid = BODY_ID[b]
    data = dict(slots={s: _cmp_slots(d, b, s) for s in SPEEDS}, caps={})
    for s in SPEEDS:
        r6 = d.s(b, s, "V6")
        extra = ""
        if not r6["within_limits"]:
            extra = (" V6 here is over a limit (" + ", ".join(f"{o['limit']} {ratio(o['value'])}" for o in r6["over"])
                     + "); its clips say so in their top band.")
        at = SPEED_LONG[s] + ("" if s == "v040" else f", {d.m(b, s, 'V0')['gait']['speed_m_s']:.2f} m/s")
        data["caps"][s] = (f"{BODY_NAME[b]} at {at}: today's walk (left) against every feature (right), from both "
                           f"cameras.{extra}")

    def sub(s):
        return "agility bar" if s == "v040" else f"{d.m(b, s, 'V0')['gait']['speed_m_s']:.2f} m/s"

    seg = "".join(f'<button type="button" data-speed="{s}" aria-pressed="{str(s == LADDER_SPEED).lower()}">'
                  f'{esc(SPEED_TAB[s])}<span>{esc(sub(s))}</span></button>' for s in SPEEDS)
    slots = "".join(vslot(x) for x in data["slots"][LADDER_SPEED])
    return (f'<div class="vgroup" data-cmp="cmp-{bid}">'
            f'<div class="seg" role="group" aria-label="Speed">{seg}</div>'
            f'<p class="cap" data-cap aria-live="polite">{esc(data["caps"][LADDER_SPEED])}</p>'
            f'{controls()}<div class="vgrid">{slots}</div></div>{json_script(f"cmp-{bid}", data)}'
            f'{v0v6_table(d, b)}')


def step_caption(d: Docs, b: str, v: str) -> str:
    s = LADDER_SPEED
    m1, m0 = d.m(b, s, v), d.m(b, s, prev_rung(v))
    r1, r0 = d.s(b, s, v), d.s(b, s, prev_rung(v))
    a, c = ratio(gov(r0)[2]), ratio(gov(r1)[2])
    g = f" The busiest motor stays at {c} of its limit." if a == c else f" The busiest motor: {a} to {c} of its limit."
    if v == "V1":
        t = f"V1 adds the hip bob: {m1['bob']['pp_mm']:.1f} mm peak to peak, lowest just after each touchdown."
    elif v == "V2":
        t = (f"V2 fixes the swing time at {m1['gait']['swing_time_s']:.3f} s; at this speed it was "
             f"{m0['gait']['swing_time_s']:.3f} s, so the change is small (duty {m0['gait']['duty'][0]:.3f} to "
             f"{m1['gait']['duty'][0]:.3f}). It shows at the nominal walk.")
    elif v == "V3":
        t = (f"V3 adds the quick-rise swing and the toe-off; at this speed the roll reaches only "
             f"{m1['toe_off']['at_lift_off_rad']:.3f} rad by lift-off (0.30 at the nominal walk).")
    elif v == "V4":
        t = f"V4 banks the trunk toward the stance leg, {m1['trunk']['roll_rms_deg']:.2f}° RMS."
    elif v == "V5":
        t = (f"V5 shifts the tail's timing (its tip now moves toward the stance side) and steadies the head: head pitch "
             f"{m0['head']['world_pitch_rms_deg']:.2f} to {m1['head']['world_pitch_rms_deg']:.2f}° RMS.")
    else:
        t = (f"V6 adds small variations: step times vary by about {100 * m1['step_times']['cv']:.1f} % from step to "
             "step, and the gaze and the "
             "tail drift slowly.")
    return t + g


def stepper(d: Docs, b: str) -> str:
    bid = BODY_ID[b]
    s = LADDER_SPEED
    steps = {}
    for v in LADDER[1:]:
        plot = f"{b}/{b}_{s}_paths_V0_{v}.png"
        steps[v] = dict(prev=slot_data(d, b, s, prev_rung(v), "visitor"), after=slot_data(d, b, s, v, "visitor"),
                        cap=step_caption(d, b, v), plot=f"media/{plot}",
                        alt=f"Paths of the snout, left hip, left toe tip and tail tip, V0 against {v}, {BODY_NAME[b]} "
                            "at 0.40 m/s",
                        plate=f"Paths, V0 (grey) against {v} (blue)")
    data = dict(steps=steps, v0=slot_data(d, b, s, "V0", "visitor"), first="V1")
    st = steps["V1"]
    seg = "".join(f'<button type="button" data-step="{v}" aria-pressed="{str(v == "V1").lower()}">{v}'
                  f'<span>{esc(RUNG_TAG[v])}</span></button>' for v in LADDER[1:])
    extra = '<button type="button" class="btn" data-act="v0" aria-pressed="false">Left: compare with V0</button>'
    fig = (f'<figure><div class="frame"><img data-plot src="{esc(st["plot"])}" alt="{esc(st["alt"])}" loading="lazy" '
           f'width="1100" height="1150"></div><figcaption><span class="plate" data-plate>{esc(st["plate"])}</span>'
           '<span>Each panel traces one point in a frame moving with the walk\'s average speed, from the side and from '
           'above. Smooth loops read as alive; straight runs and corners read as a machine. The blue rung is the one '
           'selected above, always against V0.</span></figcaption></figure>')
    return (f'<div class="vgroup stepper" data-steps="steps-{bid}">'
            f'<div class="seg" role="group" aria-label="Rung">{seg}</div>'
            f'<p class="cap" data-cap aria-live="polite">{esc(st["cap"])}</p>'
            f'{controls(extra)}<div class="vgrid">{vslot(st["prev"])}{vslot(st["after"])}</div>'
            f'<div class="stepfig">{fig}</div></div>{json_script(f"steps-{bid}", data)}')


def pointlight(d: Docs, b: str) -> str:
    slots = "".join(vslot(slot_data(d, b, LADDER_SPEED, v, view, "pointlight"))
                    for view in ("visitor", "side") for v in ("V0", "V6"))
    return f'<div class="vgroup">{controls()}<div class="vgrid">{slots}</div></div>'


def plots(d: Docs, b: str) -> str:
    s = LADDER_SPEED
    main = plot_figure(d, f"{b}/{b}_{s}_paths_V0_V6.png", f"Paths, V0 against V6, {BODY_NAME[b]} at 0.40 m/s",
                       "Paths at 0.40 m/s, V0 (grey) against V6 (blue)",
                       "The snout, left hip, left toe tip and tail tip, each from the side and from above, in a frame "
                       "moving with the walk's average speed. Smooth loops read as alive; straight runs and corners as "
                       "a machine. V6 draws many loops because no two strides are alike.")
    ph = plot_figure(d, f"{b}/{b}_{s}_phase_V0-V1-V2-V3-V4-V5-V6.png",
                     f"When each part peaks in the stride, V0 to V6, {BODY_NAME[b]} at 0.40 m/s",
                     "Timing at 0.40 m/s, V0 to V6",
                     "Each row is one part's motion over a stride, every stride overlaid; filled dots mark highs, open "
                     "dots lows, grey bars each foot's swing. Staggered peaks read as overlap, alive; everything "
                     "peaking at once reads as a machine. Read across a row to see a rung change one part.")
    more = []
    for sp in ("nominal", "fr05"):
        more.append(plot_figure(d, f"{b}/{b}_{sp}_paths_V0_V6.png", f"Paths, V0 against V6, {BODY_NAME[b]} at "
                                f"{SPEED_SHORT[sp]}", f"Paths at {SPEED_SHORT[sp]}, V0 against V6",
                                f"As above, at {SPEED_LONG[sp]}."))
        more.append(plot_figure(d, f"{b}/{b}_{sp}_phase_V0_V6.png", f"Timing, V0 against V6, {BODY_NAME[b]} at "
                                f"{SPEED_SHORT[sp]}", f"Timing at {SPEED_SHORT[sp]}, V0 against V6",
                                f"As above, at {SPEED_LONG[sp]}."))
    v06 = plot_figure(d, f"{b}/{b}_{s}_phase_V0_V6.png", f"Timing, V0 against V6, {BODY_NAME[b]} at 0.40 m/s",
                      "Timing at 0.40 m/s, V0 against V6", "The ladder chart's two ends, larger.")
    return (f'<div class="figs">{main}{ph}</div><details><summary>The same plots at the nominal walk and the animal '
            f'pace, and V0 against V6 at 0.40 m/s</summary><div class="grid2">{v06}{"".join(more)}</div></details>')


def readings_tables(d: Docs, b: str) -> str:
    from .motion import SUMMARY_ROWS, _cell, _get

    out = []
    for s in SPEEDS:
        rungs = d.metrics["bodies"][b][s]
        cols = [v for v in LADDER + ("V5F13",) if v in rungs]
        head = "".join(f'<th class="num">{v}</th>' for v in cols)
        trs = []
        for lab, path, nd in SUMMARY_ROWS:
            cells = []
            for v in cols:
                c = _cell(_get(rungs[v], path), nd, path)
                c = "over" if c == "**over**" else c
                if c.startswith("-") and c != "-" and float(c) == 0.0:
                    c = c[1:]  # a negative zero, as "-0.000"
                cells.append(f'<td class="num">{esc(c.replace("-", chr(8722)) if c != "-" else "-")}</td>')
            trs.append(f'<tr><th scope="row">{esc(lab)}</th>{"".join(cells)}</tr>')
        out.append(f'<h4>{esc(SPEED_LONG[s][0].upper() + SPEED_LONG[s][1:])}</h4><div class="tablewrap"><table>'
                   f'<thead><tr><th>Reading</th>{head}</tr></thead><tbody>{"".join(trs)}</tbody></table></div>')
    return ('<details><summary>Every machine reading for this body (research 4.1 item 5)</summary>'
            '<p class="cap">SPARC is a smoothness score: less negative is smoother. Crest hold: the share of the time '
            'the hips stay within 5 % of the bob\'s crest. Definitions: <code>metrics.json</code>.</p>'
            + "".join(out) + "</details>")


def section_body(d: Docs, b: str) -> str:
    bid = BODY_ID[b]
    H = d.m(b, "nominal", "V0")["H_m"]
    if b == "60-N-T-10":
        g = {s: (mu(d.s(b, s, "V0")), mu(d.s(b, s, "V6"))) for s in ("v040", "fr05")}
        top = max(d.screen["grips"])
        intro = (f"Hip height {H:.3f} m. The 60 cm body chosen for the long training runs (plan S2-1). Its knee is "
                 "the busiest motor at the faster walks on every rung. At those walks today's walk needs more grip "
                 f"than even the grippiest floor the plan assumes ({top}): {g['v040'][0]:.2f} at 0.40 m/s and "
                 f"{g['fr05'][0]:.2f} at the animal pace. V6 needs less: {g['v040'][1]:.2f} and {g['fr05'][1]:.2f}.")
    else:
        r6 = d.s(b, "fr05", "V6")
        intro = (f"Hip height {H:.3f} m. The only body here with F13, the longer stride, at the animal pace, and the "
                 f"only one with a walk over a limit: V6 at the animal pace ("
                 + ", ".join(f"{o['limit']} {ratio(o['value'])}" for o in r6["over"]) + "), decision 2.")
    return f"""
<section id="{bid}" data-only="owner">
  <p class="kicker">{esc(BODY_NAME[b])} · {esc(b)}</p><h2>The {esc(BODY_NAME[b])}</h2>
  <p class="prose-w">{intro}</p>
  <h3>Today's walk against every feature</h3>
  {compare_block(d, b)}
  <h3>One step at a time, at 0.40 m/s</h3>
  <p class="prose-w">Pick a rung: the left clip is the rung before it, the right clip the rung itself (or compare with
  V0). Both play from the visitor's view.</p>
  {stepper(d, b)}
  <h3>Point-light, at 0.40 m/s</h3>
  <p class="prose-w">V0 (left) against V6 (right), the visitor's view above the side view. Without the covers, the
  bob, the roll and the tail's timing are all you see.</p>
  {pointlight(d, b)}
  <h3>Paths and timing</h3>
  {plots(d, b)}
  {readings_tables(d, b)}
</section>"""


# ------------------------------------------------------------------------------------------------
# The tail finding: two charts and their table
# ------------------------------------------------------------------------------------------------
def _rows_tail(d: Docs) -> list[dict]:
    tp = d.screen["tail_phase"]
    rows = []
    for b, s in combos():
        c = tp[b][s]["changes"]
        rows.append(dict(b=b, s=s, today=c["V0_no_tail_to_today"]["mz_abs_max_nm"]["rel"],
                         f9=c["V0_no_tail_to_F9_phase"]["mz_abs_max_nm"]["rel"],
                         f9_vs_today=c["V0_phase_today_to_F9"]["mz_abs_max_nm"]["rel"],
                         mu=c["V0_phase_today_to_F9"]["mu_req_max"]["rel"],
                         look=c["V4_today_to_F9_tail"]["mz_abs_max_nm"]["rel"],
                         cyc=tp[b][s]["f9_phase_cycles_from_today"],
                         load4=passive(d.s(b, s, "V4"), "tail", "cont"), load5=passive(d.s(b, s, "V5"), "tail", "cont"),
                         p4=d.s(b, s, "V4")["passive"]["tail"]["power_mean_id_w"],
                         p5=d.s(b, s, "V5")["passive"]["tail"]["power_mean_id_w"]))
    return rows


def _svg_layout(rows):
    """y positions: one band per row, a gap and a label between the bodies."""
    top, band, gap = 34, 38, 26
    ys, y, last = [], top, None
    for r in rows:
        if last is not None and r["b"] != last:
            y += gap
        ys.append(y)
        y += band
        last = r["b"]
    return ys, y + 34, band


def chart_twist(rows) -> str:
    W, left, right = 560, 118, 54
    lim = 5 * math.ceil(100 * max(max(abs(r["today"]), abs(r["f9"])) for r in rows) / 5)
    x0 = left + (W - left - right) / 2
    sx = lambda p: x0 + (W - left - right) / 2 * p / lim  # noqa: E731
    ys, H, band = _svg_layout(rows)
    out = [f'<svg viewBox="0 0 {W} {H}" role="img" aria-label="Change in the standing foot\'s peak twist against a '
           'walk with no tail, today\'s tail and F9\'s tail, per body and speed">']
    for t in range(-lim, lim + 1, 5):
        x = sx(t)
        out.append(f'<line x1="{x:.1f}" x2="{x:.1f}" y1="{ys[0] - 10}" y2="{H - 26}" stroke="var(--grid)" '
                   'stroke-width="1"/>')
        out.append(f'<text x="{x:.1f}" y="{H - 10}" text-anchor="middle">{num(t, 0, t != 0)} %</text>')
    out.append(f'<line x1="{x0:.1f}" x2="{x0:.1f}" y1="{ys[0] - 14}" y2="{H - 26}" stroke="var(--ink-3)" '
               'stroke-width="1"/>')
    out.append(f'<text x="{x0:.1f}" y="{ys[0] - 18}" text-anchor="middle">no tail</text>')
    last = None
    for r, y in zip(rows, ys, strict=True):
        if r["b"] != last:
            out.append(f'<text x="8" y="{y - 6}" class="val">{esc(BODY_NAME[r["b"]])}</text>')
            last = r["b"]
        out.append(f'<text x="{left - 12}" y="{y + 19}" text-anchor="end">{esc(SPEED_TAB[r["s"]])}</text>')
        for k, (key, col) in enumerate((("today", "var(--series-2)"), ("f9", "var(--series-1)"))):
            p = 100 * r[key]
            yy = y + 6 + 13 * k
            xa, xb = sorted((x0, sx(p)))
            w = xb - xa
            rr = min(4.0, w / 2)
            if p >= 0:  # rounded data end on the right
                path = (f"M{xa:.1f},{yy} h{w - rr:.1f} a{rr},{rr} 0 0 1 {rr},{rr} v{11 - 2 * rr:.1f} "
                        f"a{rr},{rr} 0 0 1 -{rr},{rr} h-{w - rr:.1f} z")
                lx, anchor = xb + 6, "start"
            else:
                path = (f"M{xb:.1f},{yy} h-{w - rr:.1f} a{rr},{rr} 0 0 0 -{rr},{rr} v{11 - 2 * rr:.1f} "
                        f"a{rr},{rr} 0 0 0 {rr},{rr} h{w - rr:.1f} z")
                lx, anchor = xa - 6, "end"
            name = "today's tail" if key == "today" else "F9's tail timing"
            tip = f"{num(p, 1, True)} %|{BODY_NAME[r['b']]}, {SPEED_SHORT[r['s']]}|{name}, against no tail"
            out.append(f'<g data-tip="{esc(tip)}"><rect x="{xa - 4:.1f}" y="{yy - 2}" width="{w + 8:.1f}" height="15" '
                       f'fill="transparent"/><path d="{path}" fill="{col}"/></g>')
            out.append(f'<text x="{lx:.1f}" y="{yy + 10}" text-anchor="{anchor}">{num(p, 1, True)} %</text>')
    out.append("</svg>")
    leg = ('<div class="legend"><span><i style="background:var(--series-2)"></i>Today\'s tail timing</span>'
           '<span><i style="background:var(--series-1)"></i>F9\'s timing (about half a stride later)</span></div>')
    return f'<div class="chart">{leg}{"".join(out)}</div>'


def chart_load(rows) -> str:
    W, left, right = 560, 118, 34
    hi = max(1.2, math.ceil(10 * max(max(r["load4"], r["load5"]) for r in rows)) / 10 + 0.1)
    sx = lambda v: left + (W - left - right) * v / hi  # noqa: E731
    ys, H, band = _svg_layout(rows)
    out = [f'<svg viewBox="0 0 {W} {H}" role="img" aria-label="Tail servo sustained load with joint friction, today\'s '
           'tail timing against F9\'s, per body and speed">']
    t = 0.0
    while t <= hi + 1e-9:
        x = sx(t)
        out.append(f'<line x1="{x:.1f}" x2="{x:.1f}" y1="{ys[0] - 10}" y2="{H - 26}" stroke="var(--grid)" '
                   'stroke-width="1"/>')
        out.append(f'<text x="{x:.1f}" y="{H - 10}" text-anchor="middle">{t:.1f}</text>')
        t += 0.2
    xl = sx(1.0)
    out.append(f'<line x1="{xl:.1f}" x2="{xl:.1f}" y1="{ys[0] - 14}" y2="{H - 26}" stroke="var(--bad)" '
               'stroke-width="1.5"/>')
    out.append(f'<text x="{xl:.1f}" y="{ys[0] - 18}" text-anchor="middle">limit</text>')
    last = None
    for r, y in zip(rows, ys, strict=True):
        if r["b"] != last:
            out.append(f'<text x="8" y="{y - 6}" class="val">{esc(BODY_NAME[r["b"]])}</text>')
            last = r["b"]
        yc = y + 13
        out.append(f'<text x="{left - 12}" y="{yc + 4}" text-anchor="end">{esc(SPEED_TAB[r["s"]])}</text>')
        a, c = sx(r["load5"]), sx(r["load4"])
        tip = (f"{ratio(r['load4'])} to {ratio(r['load5'])}|{BODY_NAME[r['b']]}, {SPEED_SHORT[r['s']]}|tail servo, "
               "sustained, with joint friction")
        out.append(f'<g data-tip="{esc(tip)}"><rect x="{a - 10:.1f}" y="{yc - 12}" width="{c - a + 20:.1f}" height="24" '
                   f'fill="transparent"/><line x1="{a:.1f}" x2="{c:.1f}" y1="{yc}" y2="{yc}" stroke="var(--ink-3)" '
                   f'stroke-width="2" stroke-linecap="round"/>'
                   f'<circle cx="{c:.1f}" cy="{yc}" r="5.5" fill="var(--series-2)" stroke="var(--surface)" '
                   f'stroke-width="2"/><circle cx="{a:.1f}" cy="{yc}" r="5.5" fill="var(--series-1)" '
                   'stroke="var(--surface)" stroke-width="2"/></g>')
        out.append(f'<text x="{a - 10:.1f}" y="{yc + 4}" text-anchor="end">{ratio(r["load5"])}</text>')
        out.append(f'<text x="{c + 10:.1f}" y="{yc + 4}">{ratio(r["load4"])}</text>')
    out.append("</svg>")
    leg = ('<div class="legend"><span><i style="background:var(--series-2);border-radius:50%"></i>Today\'s tail timing '
           '(V4)</span><span><i style="background:var(--series-1);border-radius:50%"></i>F9\'s timing (V5)</span></div>')
    return f'<div class="chart">{leg}{"".join(out)}</div>'


def section_tail(d: Docs) -> str:
    rows = _rows_tail(d)
    tbl = K.table(["Body, speed", "Today's tail vs no tail", "F9's vs no tail", "F9's vs today's", "Grip need, F9's vs "
                   "today's", "On the look walk (V4)", "F9's shift (stride)", "Tail servo with friction, V4 → V5"],
                  [[esc(combo_label(r["b"], r["s"])), f"{num(100 * r['today'], 1, True)} %", f"{num(100 * r['f9'], 1, True)} %",
                    f"{num(100 * r['f9_vs_today'], 1, True)} %", f"{num(100 * r['mu'], 1, True)} %",
                    f"{num(100 * r['look'], 1, True)} %", num(r["cyc"], 3, True),
                    f"{ratio(r['load4'])} → {ratio(r['load5'])}"] for r in rows],
                  num={1, 2, 3, 4, 5, 6, 7},
                  caption="Peak twist: the standing foot's largest turning moment in single support (N m), read on "
                          "today's walk (V0) with no tail, today's tail and F9's timing at today's 0.12 rad; “On "
                          "the look walk” swaps the tail on V4. The servo loads count the joint's damping and "
                          "friction, as a ratio of the servo's sustained limit.")
    up = pct_span([r["today"] for r in rows])
    down = pct_span([r["f9"] for r in rows])
    vs = pct_span([r["f9_vs_today"] for r in rows])
    gm = pct_span([r["mu"] for r in rows])
    lk = pct_span([r["look"] for r in rows])
    l4 = span([r["load4"] for r in rows])
    l5 = span([r["load5"] for r in rows])
    pw = all(r["p4"] > 0 > r["p5"] for r in rows)
    over4 = [r for r in rows if r["load4"] >= 1.0]
    rises = [(lab, b, s, a, c) for lab, vals in moves(d, "V5") for (b, s), (a, c) in vals.items() if c - a >= MOVE - 1e-12]
    parts = {lab.split(",")[0].split(" ")[0] for lab, *_ in rises}
    if not rises:
        cost = "Nothing rises against it: on V5 no motor's ratio goes up by 0.05 or more."
    elif parts == {"neck"}:
        by: dict[tuple, list] = {}
        for lab, b, s, a, c in sorted(rises, key=lambda x: "joint friction" in x[0]):
            by.setdefault((b, s), []).append(f"{ratio(a)} to {ratio(c)}"
                                             + (" counting joint friction" if "joint friction" in lab else ""))
        items = "; ".join(f"{combo_label(b, s)}: " + ", and ".join(xs) for (b, s), xs in by.items())
        cost = ("On V5, which adds F9 together with F10's steady head, only the neck's ratios rise by 0.05 or more "
                f"({items}). The neck carries the head, so the tail's new timing itself costs nothing.")
    else:
        cost = ("On V5, which adds F9 together with F10's steady head, the ratios that rise by 0.05 or more are "
                + "; ".join(f"{lab} at {combo_label(b, s)} ({ratio(a)} to {ratio(c)})" for lab, b, s, a, c in rises)
                + ".")
    return f"""
<section id="tail" data-only="owner">
  <p class="kicker">Finding</p><h2>Today's tail adds to the twist the feet must hold</h2>
  <div class="prose">
    <p>As each leg swings forward it twists the body about the vertical, and the standing foot has to hold that twist
    against the floor by friction. Reading the walk's code, the research note predicted that today's tail swings
    toward the swinging leg's side, and so adds to the twist. Round two's straight-walk reading confirms it on both
    bodies at all three speeds: with today's tail the standing foot's peak twist is {up} higher than with no tail at
    all.</p>
    <p>Shifting the tail's timing by about half a stride (F9) makes it cancel part of the legs' twist instead:
    {down} below no tail and {vs} below today's tail, with the grip need {gm} lower. On the look walk the same swap
    lowers the peak twist {lk}. The new timing is the one that minimised the twist in a scan of the tail's phase at
    each body and speed.</p>
    <p>It also unloads the tail servo.{" With today's timing the servo pushes the tail, and the joint's friction and damping add to its work; with F9's timing the body's sway swings the tail and the servo only brakes it." if pw else ""}
    Counting that friction, which Stage 0 leaves out, the tail servo's sustained load falls from {l4} to {l5} of its
    limit{(", and the " + " and ".join(sorted({BODY_SHORT[r["b"]] for r in over4})) + " tail is no longer over it at the faster walks") if over4 else ""}.
    {cost}</p>
  </div>
  <div class="grid2">
    <figure>{chart_twist(rows)}<figcaption><span class="plate">The standing foot's peak twist, against a walk with no tail</span>
    <span>Today's walk (V0) with today's tail and with F9's timing, same swing size. Hover a bar for its value.</span></figcaption></figure>
    <figure>{chart_load(rows)}<figcaption><span class="plate">The tail servo's sustained load, with joint friction</span>
    <span>Ratio of the servo's sustained limit on the look walk, before (V4) and after (V5) F9's timing.</span></figcaption></figure>
  </div>
  {tbl}
</section>"""


# ------------------------------------------------------------------------------------------------
# Decisions
# ------------------------------------------------------------------------------------------------
def section_decide(d: Docs) -> str:
    s_, m_ = d.s, d.m
    b6, b7 = "60-N-T-10", "75-M-H-10"
    dm1 = [rel(mu(s_(b, s, "V0")), mu(s_(b, s, "V1"))) for b, s in combos()]
    n1 = [passive(s_(b6, s, "V1"), "neck", "peak") for s in ("v040", "fr05")]
    k2 = {b: (s_(b, "nominal", "V1")["drives"]["knee"]["speed_ratio"], s_(b, "nominal", "V2")["drives"]["knee"]["speed_ratio"])
          for b in BODIES}
    g2 = [rel(mu(s_(b, "nominal", "V1")), mu(s_(b, "nominal", "V2"))) for b in BODIES]
    tw = max(worst(s_(b, s, "V3")["drives"]["hip_yaw"])[1] - worst(s_(b, s, "V2")["drives"]["hip_yaw"])[1]
             for b, s in combos())
    z = [s_(b, s, "V3")["contact"]["zmp_margin_min_mm"] for b, s in combos()]
    tc = [(passive(s_(b, s, "V4"), "tail", "cont"), passive(s_(b, s, "V5"), "tail", "cont")) for b, s in combos()]
    tcr = [rel(a, c) for a, c in tc]
    g5 = [rel(mu(s_(b, s, "V4")), mu(s_(b, s, "V5"))) for b, s in combos()]
    c5 = m_(b7, "fr05", "V5")["gait"]["cadence_steps_s"]
    c13 = m_(b7, "fr05", "V5F13")["gait"]["cadence_steps_s"]
    trex = m_(b7, "fr05", "V5")["bird_curves"]["trex_cadence_steps_s"]
    k5 = s_(b7, "fr05", "V5")["drives"]["knee"]["speed_ratio"]
    k13 = s_(b7, "fr05", "V5F13")["drives"]["knee"]["speed_ratio"]
    r6 = s_(b7, "fr05", "V6")
    k6 = r6["drives"]["knee"]["speed_ratio"]
    cause = r6["cause"]["knee speed"]
    sw = (m_(b6, "nominal", "V6")["sway"]["com_pp_per_stride_max_H"], m_(b6, "nominal", "V6S")["sway"]["com_pp_per_stride_max_H"])
    cop = (s_(b6, "nominal", "V6")["contact"]["single_support_cop_margin_min_mm"],
           s_(b6, "nominal", "V6S")["contact"]["single_support_cop_margin_min_mm"])
    st = {b: d.spec(b, "nominal", "V2")["swing_time"] for b in BODIES}
    dn = [m_(b, "nominal", "V2")["gait"]["duty"][0] for b in BODIES]
    dv = [m_(b, "v040", "V2")["gait"]["duty"][0] for b in BODIES]
    sw0 = {b: m_(b, "nominal", "V1")["gait"]["swing_time_s"] for b in BODIES}
    g6n = mu(s_(b6, "nominal", "V6"))
    floor = min(d.screen["grips"])
    brisk = {b: d.screen["v040"][b]["brisk"]["speed_m_s"] for b in BODIES}
    to_fast = [m_(b, s, "V3")["toe_off"]["at_lift_off_rad"] for b in BODIES for s in ("v040", "fr05")]
    k3t = [s_(b, "nominal", "V3T")["drives"]["knee"]["speed_ratio"] for b in BODIES]
    claw_at = {b: d.refs["body_info"][b]["toe_clearance"]["clear_pitch_rad"] for b in BODIES}
    on_claw = [combo_label(b, s) for b, s in combos()
               if d.refs["bodies"][b][s]["V3"]["measured"].get("claw_roll_frames")]
    claw_txt = (f"The middle claw reaches ahead of the toe pads and touches the floor once the foot has rolled "
                f"{claw_at[b6]:.3f} rad (60 cm) or {claw_at[b7]:.3f} rad (75 cm); from there the rigid foot pivots on "
                f"the claw's tip ({'; '.join(on_claw) if on_claw else 'no walk here'}), where the simulation has no "
                "contact.")
    feat = [
        ["F1, F2", "Hip bob, weight acceptance", "The body rises and falls with the steps and settles after each "
         "footfall.", f"No limit. Grip need {pct_span(dm1)} lower. With the joints' friction, today's head nod on the "
         f"bob takes the 60 cm neck servo past its planning peak at the faster walks ({span(n1, 3)}); F10 undoes it."],
        ["F3", "Swing time", "Slow walks keep a brisk swing and spend longer on two feet.",
         f"Nominal walk only: knee speed {ratio(k2[b6][0])} to {ratio(k2[b6][1])} of its limit, grip need "
         f"{pct_span(g2)} higher. Needs decision 3."],
        ["F4, F5", "Swing shape, toe-off", "Feet rise quickly and peel off from the toes (a full roll only at the "
         "nominal walk).", f"No limit. The thigh twist's ratio up by at most {num(tw)}. During the roll the support is "
         "the toe "
         f"line ({span(z, 1)} mm margin). The claw question below."],
        ["F6", "Trunk roll", "The body banks into each step.", "None measurable."],
        ["F9, F10", "Tail timing, steady head", "The tail counters the legs; the head stops nodding.",
         f"Frees load: tail servo {pct_span(tcr)} lower with friction; grip need {pct_span(g5)} lower."],
        ["F13", "Longer stride (75 cm, animal pace)", f"Fewer, longer steps read as a bigger animal: {c5:.2f} to "
         f"{c13:.2f} steps/s (a <i>T. rex</i> at this size: {trex:.2f}).",
         f"Knee speed {ratio(k5)} to {ratio(k13)}, within; with F14, over (decision 2)."],
        ["F14", "Variation", "No two strides alike; the gaze and the tail drift slowly.",
         f"No torque of its own, but its shorter steps push the 75 cm knee over its speed limit at the animal pace "
         f"({ratio(k6)})."],
        ["F7", "Less sway (optional, not rendered)", f"Narrower sway: {sw[0]:.3f} to {sw[1]:.3f} H at the 60 cm nominal "
         "walk (spec 0.12).", f"The single-support margin inside the pad falls {cop[0]:.0f} to {cop[1]:.0f} mm, the "
         "edge that came before every Stage 1 fall."],
    ]
    rows = [[f"<b>{x[0]}</b>", esc(x[1]), x[2], x[3], '<span class="box" aria-label="keep: yes or no"></span>']
            for x in feat]
    ftable = K.table(["Feature", "Name", "What you see", "What it costs", "Keep?"], rows)
    return f"""
<section id="decide" data-only="owner">
  <p class="kicker">Your decisions</p><h2>Three decisions, and a few smaller ones</h2>
  <div class="q">
    <h3>1. Which features to keep (plan decision 1)</h3>
    <p>Mark what you want in the walk the controller learns. A subset becomes a new reference (for example V6
    without F14), generated and priced the same way before any controller work; the reference you approve is then
    frozen by its hash. Not in this round: no trunk pitch (F8, by design); head-first turns (F11) and anticipation at
    starts and stops (F12) wait for turn and transition generators that understand the new walk, and so do the 90°
    turn and the two-step stop.</p>
    {ftable}
  </div>
  <div class="q">
    <h3>2. May a feature cost a speed rung? (plan decision 1, second half)</h3>
    <p>The plan's rule: over a motor limit, a walk keeps its features and loses that speed rung, character first,
    unless you decide otherwise. One walk triggers it: the 75 cm V6 at the animal pace. Its knee reaches
    {ratio(k6)} of its speed limit on the shortest swing ({cause['fastest_swing_s']:.3f} s; swings run
    {cause['swing_s_range'][0]:.3f}–{cause['swing_s_range'][1]:.3f} s). The longer stride (F13) takes the knee from
    {ratio(k5)} to {ratio(k13)}; F14's shorter steps do the rest.</p>
    <ol>
      <li><b>Yes, keep both</b> (the rule's default): the 75 cm build loses the animal-pace rung, so its speed tests
      stop below {m_(b7, 'fr05', 'V0')['gait']['speed_m_s']:.2f} m/s.</li>
      <li><b>No, keep the rung</b> and drop F14's step-time variation at that speed only: V5F13, priced at
      {ratio(k13)}, within. The gaze and tail drift could stay, but that combination is not priced yet.</li>
      <li><b>No, keep the rung</b> and drop F13 at 75 cm: V5 there is at {ratio(k5)} on the knee (F14 on top of it is
      not priced).</li>
    </ol>
    <p class="cap">Not used here: the earlier gait study's tuned animal-pace walk (F25, information only) lowered this
    knee to 0.75 of its limit, another way to make room later. Your answer also sets the rule for the turns and stops
    still to be priced.</p>
  </div>
  <div class="q">
    <h3>3. The duty band for F3 (plan decision 2)</h3>
    <p>F3 fixes every swing at the animal pace's {st[b6]:.2f} s (60 cm) or {st[b7]:.2f} s (75 cm). At the nominal walk
    the feet then stay down {span(dn, 3)} of the stride, outside the behaviour spec's 0.55–0.65 band. Birds walk
    like this (tinamous: 0.55–0.70). At 0.40 m/s the duty stays inside the band ({span(dv, 3)}); at the animal
    pace F3 changes nothing.</p>
    <ol>
      <li><b>Widen the band</b> to 0.72 below 0.30 m/s and keep F3 at every speed. The cost at the nominal walk: knee
      speed {ratio(k2[b6][0])} to {ratio(k2[b6][1])} of its limit, grip need {pct_span(g2)} higher (the 60 cm V6
      needs {g6n:.2f}{f", above the {floor} floor of most demo floors" if g6n > floor else ""}).</li>
      <li><b>Keep the band</b>: F3 then starts at the brisk walk ({brisk[b6]:.2f} / {brisk[b7]:.2f} m/s), and the
      nominal walk keeps today's {sw0[b6]:.2f} / {sw0[b7]:.2f} s swing, the slow-motion look.</li>
    </ol>
  </div>
  <div class="q soft">
    <h3>Smaller choices, if you have a view</h3>
    <ul>
      <li><b>Which way the body rolls (F6).</b> Banking toward the stance leg (V4, rendered) or dropping the swing
      side (V4L, priced at the same cost, not rendered).</li>
      <li><b>How big the bob is (F1).</b> 75 % of the shortening the flat walk forces (rendered); 50 % and 100 % are
      priced (V1B50, V1B100). A rounder bob, without F2's long level hold at the crest, is also possible.</li>
      <li><b>How far the toes roll (F5).</b> 0.30 rad at the nominal walk, the research's low end; 0.40 rad puts the
      knee at {span(k3t, 3)} of its speed limit. At 0.40 m/s and the animal pace the roll reaches only
      {span(to_fast, 3)} rad: a full toe-off there needs a new feature (the late single-support balance point moved
      toward the toe line), for you to approve or not.</li>
      <li><b>The claw (hardware).</b> {claw_txt} Options: a load-bearing claw tip (and claw contact in the model), a
      shorter claw, or a toe-off capped near that roll.</li>
      <li><b>Less sway (F7, V6S).</b> In or out; the cost is in the table above.</li>
    </ul>
  </div>
</section>"""


# ------------------------------------------------------------------------------------------------
# Rating: the blind clips and the score sheet
# ------------------------------------------------------------------------------------------------
def blind_pair(d: Docs, pair: list[tuple[str, str, str]]) -> str:
    b = pair[0][1]
    clips = []
    for letter, _, v in pair:
        slots = []
        for view in ("visitor", "side"):
            e = d.clip(b, BLIND_SPEED, v, view)
            x = dict(src=f"media/{blind_name(letter, view, 'mp4')}", poster=f"media/{blind_name(letter, view, 'jpg')}",
                     name=f"Clip {letter}", sub=VIEW_NAME[view], flag="",
                     aria=f"Clip {letter}, {VIEW_NAME[view]}; kinematic reference, not a simulation")
            del e
            slots.append(vslot(x, controls=False, mask=f"Clip {letter}"))
        clips.append(f'<div class="blind"><h4>Clip {letter}</h4><div class="vgroup">{controls()}'
                     f'<div class="vgrid">{"".join(slots)}</div></div></div>')
    return (f'<div class="pair"><h3>Pair {"1" if b == BODIES[0] else "2"}: the robot about {BODY_SHORT[b]} tall'
            f'</h3>{"".join(clips)}</div>')


def score_sheet() -> str:
    def inputs(item):
        return "".join(f'<td class="num"><input class="score" id="s-{item}-{c}" name="s-{item}-{c}" inputmode="numeric" '
                       f'maxlength="1" autocomplete="off" aria-label="Item {item}, clip {c}: score 1 to 5"></td>'
                       for c in CLIP_LETTERS)

    head = "".join(f'<th class="num">Clip {c}</th>' for c in CLIP_LETTERS)
    rows = ['<tr class="sep"><td colspan="3">The look (1 = machine, 3 = mixed, 5 = animal)</td>'
            f'<td colspan="{len(CLIP_LETTERS)}"></td></tr>']
    for k, name, lo, hi in RUBRIC:
        rows.append(f'<tr data-item="{k} {esc(name)}"><th scope="row">{k}. {esc(name)}</th><td class="pole">1: {esc(lo)}'
                    f'</td><td class="pole">5: {esc(hi)}</td>{inputs(k)}</tr>')
    rows.append('<tr class="sep"><td colspan="3">Impression (1 to 5 between the two words)</td>'
                f'<td colspan="{len(CLIP_LETTERS)}"></td></tr>')
    for k, lo, hi in GODSPEED:
        rows.append(f'<tr data-item="{k} {esc(lo)} to {esc(hi)}"><th scope="row">{k}.</th><td class="pole">1: {esc(lo)}'
                    f'</td><td class="pole">5: {esc(hi)}</td>{inputs(k)}</tr>')
    pairs = []
    for i, (a, b) in enumerate(((CLIP_LETTERS[0], CLIP_LETTERS[1]), (CLIP_LETTERS[2], CLIP_LETTERS[3])), start=1):
        opts = "".join(f'<label><input type="radio" name="pair{i}" id="pair{i}-{c}" value="{c}"> Clip {c}</label>'
                       for c in (a, b))
        pairs.append(f'<fieldset class="pairq" data-pair="Pair {i}"><legend>Pair {i}: which looks more like a living '
                     f'animal?</legend>{opts}</fieldset>')
    return f"""
<div class="sheet" id="sheet">
  <div class="tablewrap"><table class="scores"><thead><tr><th>Item</th><th>1 means</th><th>5 means</th>{head}</tr></thead>
  <tbody>{"".join(rows)}</tbody></table></div>
  <div class="pairs">{"".join(pairs)}</div>
  <div class="grid2 fields">
    <label class="field" for="viewer-name">Your name or initials<input id="viewer-name" autocomplete="off"></label>
    <label class="field" for="viewer-notes">Anything else you noticed<textarea id="viewer-notes" rows="3"></textarea></label>
  </div>
  <div class="vctl"><button type="button" class="btn primary" id="copy-scores">Copy my scores</button>
  <button type="button" class="btn" id="clear-scores">Clear the sheet</button></div>
  <p class="status" id="score-status" role="status"></p>
  <textarea class="out" id="score-text" readonly hidden aria-label="Your scores as text"></textarea>
</div>"""


def section_rate(d: Docs) -> str:
    pairs = [[x for x in BLIND if x[1] == b] for b in BODIES]
    key = "; ".join(f"Clip {letter}: {BODY_SHORT[b]} {v} ({'today' + chr(39) + 's walk' if v == 'V0' else 'every feature'})"
                    for letter, b, v in BLIND)
    return f"""
<section id="rate">
  <p class="kicker">Rating</p>
  <h2 data-only="owner">The score sheet, for you and 3–5 new viewers</h2>
  <h2 data-only="viewers">Watch, then score</h2>
  <div class="prose" data-only="owner">
    <p>The research's rubric (seven items: the behaviour spec's five plus timing and an overall impression) and the
    six Godspeed animacy items, each scored 1 to 5. Rate first yourself if you like. Then show the viewers' version
    of this page to 3–5 people new to the project: open it with <a href="#viewers">#viewers</a> at the end of the
    address. It shows only the four blind clips below and this sheet, with the labels that name the walk covered.
    Each viewer scores clips A–D and picks the more lifelike clip of each pair, then copies the scores to you.</p>
    <p>The lead checks that the items agree (Cronbach's α of at least 0.7, as the Godspeed authors recommend)
    before averaging. The plan's gate asks for scores on V0 and the variant you propose; if your pick differs from V6,
    the lead renders it and swaps it in. Item 6 can only judge the strides' variation here: starts and stops are not
    rendered yet.</p>
    <details><summary>The key: which clip is which (keep it from the viewers)</summary><p class="cap">{esc(key)}. All
    at 0.40 m/s, from both cameras; the order is the same for every viewer and puts the all-features walk first in
    one pair and second in the other.</p></details>
  </div>
  <div class="prose" data-only="viewers">
    <p>Each clip shows one walk from two cameras at once. In each pair the two clips are different walks. Press
    <b>Play together</b> and watch each clip at least twice; <b>Half speed</b> helps with the feet. Then score every
    item from 1 to 5 for each clip. Go with your impression, and leave a box empty if you cannot judge an item. The
    labels at the top of each clip are covered on purpose.</p>
    <p>When you are done, press <b>Copy my scores</b> and paste the text into a message to the person who sent you
    this page. Your scores stay in this browser until you send them.</p>
  </div>
  {"".join(blind_pair(d, p) for p in pairs)}
  <h3>Score sheet</h3>
  {score_sheet()}
</section>"""


def footer(d: Docs, commit: str | None = None) -> str:
    sc = d.screen["provenance"]["git"]["commit"][:7]
    mc = d.metrics["provenance"]["git"]["commit"][:7]
    cc = d.clips["provenance"]["git"]["commit"][:7]
    return f"""
<footer data-only="owner">
  <p>Simulation only: kinematic references with perfect tracking on the round-two models; nothing is adopted without
  you (P24). Prices: <code>stage2/look/SCREEN.md</code> and <code>screen.json</code> (commit {sc}). Clips:
  <code>clips.json</code> (commit {cc}), {len(d.clips['clips'])} clips of {d.clips['seconds']:g} s at
  {d.clips['fps']} fps, each frame labelled “{esc(d.clips['label'])}”. Readings and plots:
  <code>metrics.json</code> (commit {mc}). Research: <code>stage2/research/research_look.md</code>; plan:
  <code>STAGE2_PLAN.md</code> 4 A and 5. Page: <code>designlab/stage2/look_page.py</code>{f" (commit {esc(commit)})" if commit else ""}.</p>
</footer>
<footer data-only="viewers"><p>Thank you for taking part. The clips are computer playback of a planned walk, not a
recording of the robot.</p></footer>"""


# ------------------------------------------------------------------------------------------------
# The page
# ------------------------------------------------------------------------------------------------
CSS = r"""
:root{--mask-bg:#fcfcfb;--mask-ink:#0b0b0b;--mask-ink-2:#52514e}
[hidden]{display:none!important}
html{scroll-padding-top:12px}
.vh{position:absolute!important;width:1px;height:1px;overflow:hidden;clip:rect(0 0 0 0);white-space:nowrap}
html:not(.viewers) [data-only="viewers"]{display:none!important}
html.viewers [data-only="owner"]{display:none!important}
.prose-w{max-width:78ch;color:var(--ink-2)}
.prose{align-content:start}
h4{font:600 15px/1.3 "Source Sans 3",system-ui,sans-serif;margin:6px 0 0}
section h3{margin-top:10px}
.btn{font:600 14px/1.2 "Source Sans 3",system-ui,sans-serif;padding:8px 13px;min-height:38px;border-radius:8px;
  border:1px solid var(--rule);background:var(--surface);color:var(--ink);cursor:pointer}
.btn:hover{border-color:var(--accent)}
.btn.primary{background:var(--accent);border-color:var(--accent);color:var(--bg)}
.btn[aria-pressed="true"]{background:var(--wash);border-color:var(--accent);color:var(--ink)}
.btn:focus-visible,.seg button:focus-visible,input:focus-visible,textarea:focus-visible,summary:focus-visible,
a:focus-visible,[data-tip]:focus-visible{outline:2px solid var(--accent);outline-offset:2px}
.vctl{display:flex;flex-wrap:wrap;gap:8px;align-items:center}
.seg{display:flex;flex-wrap:wrap;gap:6px}
.seg button{font:600 14px/1.2 "Source Sans 3",system-ui,sans-serif;padding:8px 13px;min-height:38px;border-radius:999px;
  border:1px solid var(--rule);background:var(--surface);color:var(--ink-2);cursor:pointer}
.seg button span{font-weight:400;color:var(--ink-3);margin-left:6px}
.seg button[aria-pressed="true"]{background:var(--ink);border-color:var(--ink);color:var(--bg)}
.seg button[aria-pressed="true"] span{color:var(--bg)}
.vgroup{display:grid;gap:12px;min-width:0}
.vgrid{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:14px 16px}
@media (max-width:720px){.vgrid{grid-template-columns:minmax(0,1fr)}}
.vslot{display:grid;gap:6px;min-width:0;align-content:start}
.vlab{display:flex;flex-wrap:wrap;gap:2px 10px;align-items:baseline;font-size:14.5px;min-height:24px}
.vname{font-weight:600}
.vsub{color:var(--ink-3);font-size:13px}
.vbox{position:relative;aspect-ratio:854/480;width:100%;max-width:100%;border:1px solid var(--rule);border-radius:10px;
  overflow:hidden;background:var(--plate)}
.vbox video{display:block;width:100%;height:100%;object-fit:contain;background:var(--plate)}
.mask{position:absolute;left:0;right:0;top:0;height:10.4%;background:var(--mask-bg);color:var(--mask-ink);display:flex;
  align-items:center;gap:12px;padding:0 12px;font-size:clamp(10px,1.5vw,14px);pointer-events:none;white-space:nowrap;
  overflow:hidden}
.mask span{color:var(--mask-ink-2)}
.cap{font-size:14px;color:var(--ink-2);max-width:90ch}
.rungs{display:grid}
.rung{display:grid;grid-template-columns:170px repeat(3,minmax(0,1fr));gap:12px 24px;padding:20px 0;
  border-top:1px solid var(--rule)}
.rung:first-child{border-top:0;padding-top:4px}
.rung>*{min-width:0}
.rung details{grid-column:1/-1}
@media (max-width:960px){.rung{grid-template-columns:minmax(0,1fr)}}
.rhead{display:grid;gap:8px;align-content:start}
.rhead h3{font-size:19px;margin:0}
.adds{font:500 12.5px/1.45 "IBM Plex Mono",ui-monospace,monospace;color:var(--accent-2)}
.badge{font:600 13px/1 "IBM Plex Mono",ui-monospace,monospace;display:inline-block;justify-self:start;padding:5px 8px;
  border-radius:6px;background:var(--ink);color:var(--bg);letter-spacing:.04em}
.badge.sm{font-size:12px;padding:4px 6px}
.lbl{font:500 11.5px/1.4 "IBM Plex Mono",ui-monospace,monospace;letter-spacing:.06em;text-transform:uppercase;
  color:var(--ink-3);display:block;margin-bottom:5px}
.rung p{font-size:15px}
.tells{display:flex;flex-wrap:wrap;gap:5px}
.tell{font:500 11px/1.7 "IBM Plex Mono",ui-monospace,monospace;padding:0 6px;border-radius:4px;border:1px solid var(--rule)}
.tell.on{border-color:var(--warn);color:var(--warn)}
.tell.off{color:var(--ink-3)}
details{border:1px solid var(--rule);border-radius:10px;background:var(--surface);padding:0 14px;min-width:0}
details>summary{cursor:pointer;padding:11px 0;font-weight:600;color:var(--ink-2)}
details[open]>summary{border-bottom:1px solid var(--grid);margin-bottom:12px}
details[open]{padding-bottom:14px}
details .tablewrap{background:var(--bg)}
details.mini{background:transparent}
table.moves td.still{color:var(--ink-3)}
table.moves td.mv{font-weight:600}
table.moves td.ov,table.matrix td.ov{color:var(--bad)}
table th[scope="row"]{font-weight:600;color:var(--ink);background:transparent;white-space:normal;min-width:11em}
th.grp{text-align:center}
.bl{border-left:1px solid var(--rule)}
table.matrix td.m{min-width:126px}
table.matrix .v{display:flex;flex-wrap:wrap;gap:2px 6px;align-items:baseline}
table.matrix .v b{font-variant-numeric:tabular-nums;font-size:15px}
table.matrix .v span{color:var(--ink-3);font-size:12.5px}
.meter{position:relative;height:6px;border-radius:3px;background:var(--surface-2);margin-top:5px}
.meter i{position:absolute;left:0;top:0;bottom:0;border-radius:3px;background:var(--series-1)}
.meter i.near{background:var(--warn)}
.meter i.over{background:var(--bad)}
.meter b{position:absolute;top:-3px;bottom:-3px;width:1px;background:var(--ink-3)}
table.v06 [data-col].on{background:var(--wash)}
.figs{display:grid;gap:18px}
.figs figure:first-child .frame{max-width:860px}
.stepfig figure .frame{max-width:860px}
ul.look{margin:0;padding-left:1.2em;display:grid;gap:6px;max-width:80ch}
dl.gloss{display:grid;grid-template-columns:minmax(120px,190px) minmax(0,1fr);gap:8px 18px;margin:0;font-size:15px}
dl.gloss dt{font-weight:600}
dl.gloss dd{margin:0;color:var(--ink-2)}
@media (max-width:640px){dl.gloss{grid-template-columns:minmax(0,1fr)}dl.gloss dd{margin-bottom:6px}}
.q ol,.q ul{margin:0;padding-left:1.3em;display:grid;gap:6px}
.q.soft{border-left-color:var(--rule)}
.q .tablewrap{background:var(--bg)}
span.box{display:inline-block;width:22px;height:22px;border:1.5px solid var(--ink-3);border-radius:4px;vertical-align:middle}
.pair{display:grid;gap:12px;padding-top:6px}
.blind{display:grid;gap:8px}
.blind h4{font-family:Fraunces,Georgia,serif;font-size:18px}
.sheet{display:grid;gap:16px}
table.scores td.pole{color:var(--ink-2);font-size:13px;min-width:12em}
table.scores tr.sep td{background:var(--surface-2);font-weight:600;color:var(--ink-2)}
input.score{width:2.7em;height:2.3em;text-align:center;font:600 16px/1 "Source Sans 3",system-ui,sans-serif;
  border:1px solid var(--ink-3);border-radius:6px;background:var(--bg);color:var(--ink)}
input.score[aria-invalid="true"]{border:2px solid var(--bad)}
.pairs{display:grid;gap:10px}
fieldset.pairq{border:1px solid var(--rule);border-radius:10px;padding:10px 14px;display:flex;flex-wrap:wrap;gap:8px 22px;
  align-items:center;margin:0;min-width:0}
fieldset.pairq legend{font-weight:600;padding:0 4px}
fieldset.pairq label{display:flex;gap:6px;align-items:center;min-height:32px}
.fields{gap:14px}
.field{display:grid;gap:5px;font-weight:600;font-size:14.5px;min-width:0}
.field input,.field textarea{font:400 15px/1.4 "Source Sans 3",system-ui,sans-serif;padding:8px 10px;border:1px solid var(--rule);
  border-radius:8px;background:var(--bg);color:var(--ink);width:100%}
.status{font-size:14px;color:var(--ink-2);min-height:1.4em}
textarea.out{width:100%;min-height:10em;font:13px/1.45 "IBM Plex Mono",ui-monospace,monospace;padding:10px;border:1px solid var(--rule);
  border-radius:8px;background:var(--bg);color:var(--ink)}
@media print{
  nav.toc,.vctl,.seg,.vgrid,.stepfig,.figs,details,.chart,footer{display:none!important}
  section{break-inside:auto}
  input.score{border:1px solid #000}
}
"""

JS = r"""
(function(){
  'use strict';
  var docEl = document.documentElement;
  function each(list, f){ Array.prototype.forEach.call(list, f); }
  function setMode(){ var v = location.hash === '#viewers'; docEl.classList.toggle('viewers', v); if (v) { window.scrollTo(0, 0); } }
  window.addEventListener('hashchange', setMode);
  setMode();

  function vids(g){ return Array.prototype.slice.call(g.querySelectorAll('video')); }
  function slowOf(g){ return g.getAttribute('data-slow') === '1' ? 0.5 : 1; }
  function whenReady(vs, done){
    var left = vs.length, fired = false, timer = null;
    function finish(){ if (fired) { return; } fired = true; if (timer) { clearTimeout(timer); } done(); }
    function one(){ left -= 1; if (left <= 0) { finish(); } }
    timer = setTimeout(finish, 8000);
    vs.forEach(function(v){
      if (v.readyState >= 3) { one(); return; }
      var h = function(){ v.removeEventListener('canplay', h); v.removeEventListener('error', h); one(); };
      v.addEventListener('canplay', h); v.addEventListener('error', h);
      v.preload = 'auto';
      if (v.readyState === 0) { v.load(); }
    });
  }
  function playAll(g){
    var vs = vids(g), r = slowOf(g);
    vs.forEach(function(v){ v.pause(); });
    whenReady(vs, function(){
      vs.forEach(function(v){ try { v.currentTime = 0; } catch (e) {} v.defaultPlaybackRate = r; v.playbackRate = r; });
      vs.forEach(function(v){ var p = v.play(); if (p && p.catch) { p.catch(function(){}); } });
    });
  }
  function pauseAll(g){ vids(g).forEach(function(v){ v.pause(); }); }
  function toggleSlow(g, b){
    var on = g.getAttribute('data-slow') !== '1';
    g.setAttribute('data-slow', on ? '1' : '0'); b.setAttribute('aria-pressed', String(on));
    vids(g).forEach(function(v){ v.defaultPlaybackRate = on ? 0.5 : 1; v.playbackRate = on ? 0.5 : 1; });
  }
  document.addEventListener('click', function(e){
    var b = e.target.closest ? e.target.closest('[data-act]') : null; if (!b) { return; }
    var g = b.closest('.vgroup'); if (!g) { return; }
    var a = b.getAttribute('data-act');
    if (a === 'play') { playAll(g); } else if (a === 'pause') { pauseAll(g); } else if (a === 'slow') { toggleSlow(g, b); }
  });
  function setSlot(slot, d, g){
    var v = slot.querySelector('video');
    v.pause(); v.preload = 'none';
    v.setAttribute('poster', d.poster); v.setAttribute('aria-label', d.aria); v.setAttribute('src', d.src); v.load();
    var r = slowOf(g); v.defaultPlaybackRate = r; v.playbackRate = r;
    slot.querySelector('.vname').textContent = d.name;
    slot.querySelector('.vsub').textContent = d.sub;
    var f = slot.querySelector('.vflag'); if (f) { f.textContent = d.flag || ''; f.hidden = !d.flag; }
  }
  function data(el, attr){ return JSON.parse(document.getElementById(el.getAttribute(attr)).textContent); }
  each(document.querySelectorAll('[data-cmp]'), function(g){
    var D = data(g, 'data-cmp'), btns = g.querySelectorAll('[data-speed]');
    var table = g.nextElementSibling && g.nextElementSibling.nextElementSibling;
    function mark(sp){ if (!table) { return; } each(table.querySelectorAll('[data-col]'), function(c){ c.classList.toggle('on', c.getAttribute('data-col') === sp); }); }
    each(btns, function(b){ b.addEventListener('click', function(){
      var sp = b.getAttribute('data-speed');
      each(btns, function(u){ u.setAttribute('aria-pressed', String(u === b)); });
      each(g.querySelectorAll('.vslot'), function(s, i){ setSlot(s, D.slots[sp][i], g); });
      g.querySelector('[data-cap]').textContent = D.caps[sp];
      mark(sp);
    }); });
    mark('v040');
  });
  each(document.querySelectorAll('[data-steps]'), function(g){
    var D = data(g, 'data-steps'), cur = D.first, withV0 = false;
    var slots = g.querySelectorAll('.vslot'), btns = g.querySelectorAll('[data-step]');
    var img = g.querySelector('img[data-plot]'), plate = g.querySelector('[data-plate]');
    function show(){
      var st = D.steps[cur];
      setSlot(slots[0], withV0 ? D.v0 : st.prev, g); setSlot(slots[1], st.after, g);
      g.querySelector('[data-cap]').textContent = st.cap;
      img.src = st.plot; img.alt = st.alt; plate.textContent = st.plate;
    }
    each(btns, function(b){ b.addEventListener('click', function(){
      cur = b.getAttribute('data-step'); each(btns, function(u){ u.setAttribute('aria-pressed', String(u === b)); }); show();
    }); });
    var t = g.querySelector('[data-act="v0"]');
    if (t) { t.addEventListener('click', function(){ withV0 = !withV0; t.setAttribute('aria-pressed', String(withV0)); show(); }); }
  });

  var sheet = document.getElementById('sheet');
  if (sheet) {
    var KEY = 'trex-look-review-scores-v1';
    var fields = sheet.querySelectorAll('input, textarea');
    var status = document.getElementById('score-status'), out = document.getElementById('score-text');
    function save(){
      var o = {};
      each(fields, function(f){ if (f.id === 'score-text') { return; } if (f.type === 'radio') { if (f.checked) { o[f.name] = f.value; } } else { o[f.id] = f.value; } });
      try { localStorage.setItem(KEY, JSON.stringify(o)); } catch (e) {}
    }
    function restore(){
      var o = null;
      try { o = JSON.parse(localStorage.getItem(KEY) || 'null'); } catch (e) { o = null; }
      if (!o) { return; }
      each(fields, function(f){ if (f.id === 'score-text') { return; } if (f.type === 'radio') { f.checked = o[f.name] === f.value; } else if (typeof o[f.id] === 'string') { f.value = o[f.id]; } });
      each(sheet.querySelectorAll('input.score'), check);
    }
    function check(f){ var ok = /^[1-5]?$/.test(f.value.trim()); f.setAttribute('aria-invalid', String(!ok)); return ok; }
    restore();
    sheet.addEventListener('input', function(e){ if (e.target.classList.contains('score')) { check(e.target); } save(); });
    sheet.addEventListener('change', save);
    function val(id){ var el = document.getElementById(id); return el ? el.value.trim() : ''; }
    function text(){
      var letters = []; each(sheet.querySelectorAll('thead th.num'), function(th){ letters.push(th.textContent.trim()); });
      var lines = ['T. rex look review: scores', 'Viewer\t' + val('viewer-name'), '', ['Item'].concat(letters).join('\t')];
      each(sheet.querySelectorAll('tr[data-item]'), function(tr){
        var row = [tr.getAttribute('data-item')];
        each(tr.querySelectorAll('input.score'), function(i){ row.push(i.value.trim()); });
        lines.push(row.join('\t'));
      });
      each(sheet.querySelectorAll('fieldset[data-pair]'), function(fs){
        var c = fs.querySelector('input:checked'); lines.push(fs.getAttribute('data-pair') + ', more like a living animal\t' + (c ? 'Clip ' + c.value : ''));
      });
      lines.push('Notes\t' + val('viewer-notes').replace(/\s+/g, ' '));
      return lines.join('\n');
    }
    document.getElementById('copy-scores').addEventListener('click', function(){
      var bad = 0; each(sheet.querySelectorAll('input.score'), function(f){ if (!check(f)) { bad += 1; } });
      var t = text(); out.value = t;
      function done(ok){
        var note = bad ? ' ' + bad + ' box' + (bad > 1 ? 'es hold' : ' holds') + ' something other than 1 to 5 (outlined).' : '';
        if (ok) { status.textContent = 'Copied. Paste it into a message to the person who sent you this page.' + note; out.hidden = true; }
        else { status.textContent = 'This browser did not allow copying. Select the text below and copy it.' + note; out.hidden = false; out.focus(); out.select(); }
      }
      try { navigator.clipboard.writeText(t).then(function(){ done(true); }, function(){ done(false); }); } catch (e) { done(false); }
    });
    var clearBtn = document.getElementById('clear-scores'), armed = null;
    clearBtn.addEventListener('click', function(){
      if (!armed) { clearBtn.textContent = 'Press again to clear'; armed = setTimeout(function(){ armed = null; clearBtn.textContent = 'Clear the sheet'; }, 4000); return; }
      clearTimeout(armed); armed = null; clearBtn.textContent = 'Clear the sheet';
      each(fields, function(f){ if (f.type === 'radio') { f.checked = false; } else { f.value = ''; } if (f.classList.contains('score')) { f.setAttribute('aria-invalid', 'false'); } });
      out.hidden = true; status.textContent = 'Cleared.'; save();
    });
  }
})();
"""


def page(d: Docs, commit: str | None = None) -> str:
    body = "".join([header(d), section_what(d), section_ladder(d)] + [section_body(d, b) for b in BODIES]
                   + [section_tail(d), section_decide(d), section_rate(d), footer(d, commit)])
    early = "<script>if(location.hash==='#viewers'){document.documentElement.classList.add('viewers')}</script>"
    return (f'<title>{esc(TITLE)}</title>\n<meta name="claude:viewport-fit-sides" content="cover">\n'
            '<meta name="description" content="Kinematic clips, costs and a score sheet for picking the lifelike-walk '
            'features of the 60 and 75 cm robotic T. rex (Stage 2, look gate L1).">\n'
            '<link rel="preconnect" href="https://fonts.googleapis.com">\n'
            '<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>\n'
            f'<link rel="stylesheet" href="{K.FONTS}">\n<style>{K.CSS}{CSS}</style>\n{early}\n'
            f'<div class="wrap">{body}</div>\n<div id="tip" hidden></div>\n'
            f'<dialog id="zoom"><img alt=""></dialog>\n<script>{K.JS}</script>\n<script>{JS}</script>\n')


def _plot_sizes(d: Docs, out: Path) -> None:
    PLOT_SIZE.clear()
    for p in d.metrics["plots"]:
        f = Path(out) / "media" / p
        if f.exists():
            with Image.open(f) as im:
                PLOT_SIZE[p] = im.size


def referenced(text: str) -> list[str]:
    """Every media path the page names (src, poster and the JSON slot data), in order of first mention."""
    import re

    seen, out = set(), []
    for m in re.finditer(r'media/[A-Za-z0-9_./-]+\.(?:mp4|jpg|png)', text):
        p = m.group(0)
        if p not in seen:
            seen.add(p)
            out.append(p)
    return out


def build(media: Path, out: Path, look_dir: Path = OUT, commit: str | None = None) -> dict:
    """Stage the media and write ``<out>/index.html``; the manifest (module docstring), without provenance. ``commit``
    names the builder's code in the page's footer."""
    t0 = time.perf_counter()
    d = load(look_dir)
    out = Path(out)
    if (out / "media").exists():
        shutil.rmtree(out / "media")
    out.mkdir(parents=True, exist_ok=True)
    files = stage_media(d, Path(media), out)
    _plot_sizes(d, out)
    text = page(d, commit)
    (out / "index.html").write_text(text)
    refs = referenced(text)
    missing = [p for p in refs if not (out / p).exists()]
    if missing:
        raise FileNotFoundError(f"the page names media it does not have: {missing[:5]}")
    unused = sorted({f["path"] for f in files} - set(refs))
    total = (out / "index.html").stat().st_size + sum(f["bytes"] for f in files)
    if total > MAX_PAGE_MB * 1e6:
        raise ValueError(f"the page and its media are {total / 1e6:.1f} MB, over {MAX_PAGE_MB:g} MB")
    return dict(what="Stage 2 look gate L1: the owner's look review page (designlab/stage2/look_page.py)",
                title=TITLE, page=dict(path="index.html", bytes=(out / "index.html").stat().st_size,
                                       sha256=sha256(out / "index.html")),
                total_bytes=total, n_files=len(files), files=files, unused=unused,
                blind_key=[dict(letter=x[0], body=x[1], variant=x[2], speed=BLIND_SPEED) for x in BLIND],
                inputs={n: sha256(Path(look_dir) / n) for n in ("metrics.json", "screen.json", "clips.json",
                                                                 "look_refs.json")},
                media_dir="outside git (the --media directory; sources relative to it)",
                seconds=time.perf_counter() - t0)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="python -m designlab.stage2.look_page", description=__doc__.split("\n\n")[0])
    ap.add_argument("--media", required=True, help="the render run's media directory (clips, posters, plots)")
    ap.add_argument("--out", required=True, help="the page's directory (index.html and media/)")
    ap.add_argument("--look", default=str(OUT), help="stage2/look (metrics.json, screen.json, clips.json, look_refs.json)")
    ap.add_argument("--manifest", default=None, help="write the page's manifest here (JSON)")
    a = ap.parse_args(argv)
    from ..stage1 import plant

    prov = plant.provenance()
    git = prov.get("git") or {}
    commit = (git["commit"][:7] + (" with uncommitted changes" if git.get("dirty") else "")) if git.get("commit") else None
    res = build(Path(a.media), Path(a.out), Path(a.look), commit=commit)
    if a.manifest:
        res["provenance"] = prov
        Path(a.manifest).write_text(json.dumps(res, indent=1) + "\n")
    print(f"{Path(a.out) / 'index.html'}: page {res['page']['bytes'] / 1e3:.0f} kB, with media "
          f"{res['total_bytes'] / 1e6:.1f} MB; {res['n_files']} media files, {len(res['unused'])} unused")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
