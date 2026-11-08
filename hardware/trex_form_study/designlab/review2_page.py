"""Build the round-two review page (HTML + web-sized images) from the round-two records, screens and shortlist.

Run from hardware/trex_form_study after ``round2``, ``report2`` and ``shortlist``:
    python -m designlab.review2_page --round2 round2 --out <dir>
Every number on the page is read from round2/candidates, round2/review (review_index.json and the
screening files), round2/shortlist.json, round2/round1_recheck.json and round2/CROSSCHECK.json; every image
is a plate rendered from the compiled models.  The prose states what those numbers show.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from . import page_kit as K
from .params import Candidate
from .review_page import BATTERY_NAMES, part, web_image
from .round2 import FAMILIES, HEIGHTS, VARIANT_LABEL, VARIANTS, base_ids, candidate_ids, variant_key

FAMILY = {"T": "Tapered", "H": "Hybrid"}
GROUPS = ("hip_roll", "hip_pitch", "hip_yaw", "knee", "ankle_a", "ankle_b", "neck", "tail")
GROUP_LABEL = {"hip_roll": "hip roll", "hip_pitch": "hip pitch", "hip_yaw": "thigh twist", "knee": "knee",
               "ankle_a": "ankle A", "ankle_b": "ankle B", "neck": "neck", "tail": "tail"}
CATS = ("natural_walking", "turning", "carpet", "robustness", "following", "endurance", "head_and_tail", "packaging",
        "cost")
CAT_LABEL = {"natural_walking": "Natural walking", "turning": "Turning", "carpet": "Carpet", "robustness": "Robustness",
             "following": "Following", "endurance": "Endurance", "head_and_tail": "Head and tail",
             "packaging": "Packaging", "cost": "Cost"}
GATE_LABEL = {"clearance": "clearance", "drives_sizing_walks": "drives in the sizing walks",
              "turning_targets": "target turns", "person_in_view": "person in view", "runtime_20min": "20 min runtime",
              "reference_walk_cop": "reference-walk CoP", "carpet": "carpet"}


def kg(v):
    return f"{v:.1f} kg"


def cm(v):
    return f"{100 * v:.0f} cm"


def usd(v):
    return f"${v / 1000:.1f}k"


def mono(s):
    return f"<span class='mono'>{K.esc(s)}</span>"


def _ok(v, yes="pass", no="fail") -> str:
    if v is None:
        return K.tag("not run")
    return K.tag(yes, "ok") if v else K.tag(no, "bad")


def _rng(vals, f):
    vals = [v for v in vals if v is not None]
    if not vals:
        return "–"
    lo, hi = min(vals), max(vals)
    return f(lo) if abs(hi - lo) < 1e-9 or f(lo) == f(hi) else f"{f(lo)}–{f(hi)}"


def _worst(doc) -> tuple[str, float, str]:
    best = ("", 0.0, "")
    for g, m in doc["margins"].items():
        for kind in ("cont", "peak", "speed"):
            if m[f"{kind}_ratio"] > best[1]:
                best = (g, m[f"{kind}_ratio"], kind)
    return best


class Ctx:
    def __init__(self, r2: Path, r1: Path, out: Path):
        self.r2, self.r1, self.out = r2, r1, out
        self.D = {p.stem: json.loads(p.read_text()) for p in sorted((r2 / "candidates").glob("*.json"))}
        idx = r2 / "review" / "review_index.json"
        self.R = json.loads(idx.read_text()) if idx.exists() else {}
        sl = r2 / "shortlist.json"
        self.SL = json.loads(sl.read_text()) if sl.exists() else None
        rc = r2 / "round1_recheck.json"
        self.RC = json.loads(rc.read_text()) if rc.exists() else None
        xc = r2 / "CROSSCHECK.json"
        self.XC = json.loads(xc.read_text()) if xc.exists() else None
        self.D1 = {}
        for cid in base_ids():
            p = r1 / "stage_a" / "candidates" / f"{cid}.json"
            if p.exists():
                self.D1[cid] = json.loads(p.read_text())
        self.plates = r2 / "review" / "plates"
        self.screen = r2 / "review" / "screening"
        self.images, self.plate_no = [], 0
        self.ok = [c for c in candidate_ids() if self.D.get(c, {}).get("ok")]
        self.rows = {r["id"]: r for r in (self.SL or {}).get("rows", [])}

    def img(self, name: str, alt: str, title: str, caption: str, src_dir: Path | None = None) -> str:
        src = (src_dir or self.plates) / name
        if not src.exists():
            return f'<p class="note">Missing plate: <code>{K.esc(name)}</code></p>'
        self.plate_no += 1
        web = Path("img") / (Path(name).stem + ".webp")
        web_image(src, self.out / web)
        self.images.append(str(web))
        return K.figure(str(web), alt, f"Plate {self.plate_no} · {title}", caption)

    def gates(self, cid):
        return (self.rows.get(cid) or {}).get("gates") or {}

    def passes(self, cid):
        return bool(self.gates(cid).get("all"))

    def picks(self):
        return [p["id"] for p in (self.SL or {}).get("picks", [])]


# ------------------------------------------------------------------------------------------------
# Sections
# ------------------------------------------------------------------------------------------------
def header(x: Ctx) -> str:
    bases = [x.D[c] for c in base_ids() if c in x.D and x.D[c].get("ok")]
    n_pass = sum(1 for c in x.ok if x.passes(c))
    picks = x.picks()
    by_h = {h: [d for d in bases if d["candidate"]["height"] == h] for h in HEIGHTS}
    fact_mass = " / ".join(_rng([d["metrics"]["mass"] for d in by_h[h]], lambda v: f"{v:.1f}") for h in HEIGHTS)
    fact_rt = " / ".join(_rng([d["runtime"]["minutes"] for d in by_h[h]], lambda v: f"{v:.0f}") for h in HEIGHTS)
    fact_cost = " / ".join(_rng([d["cost"]["nominal"] for d in by_h[h]], usd) for h in HEIGHTS)
    return f"""
<header class="top">
  <p class="kicker">Mesozoic Labs · robotic <i>T. rex</i> · design review, round two · 8 Nov 2026</p>
  <h1>Two sizes, two bodies, three to verify</h1>
  <p class="lede">Round two built your round-one choices from scratch: 60 and 75 cm crown height, the Tapered and
  Hybrid bodies, a 10° chest-up stance and a Raspberry Pi 5 with the AI HAT+. Each of the four base bodies has
  four single-change variants: a narrower stance, a larger skull, compact proportions and the level posture.
  All {len(x.ok)} candidates were sized for loads, drives and battery, and screened for clearance, sensing,
  turning, carpet and a behavior envelope. A scorecard built from those results proposes three bodies for
  verification. Nothing here is a final design, and no controller has been trained.</p>
  <div class="facts">
    <div class="fact"><b>{fact_mass} kg</b><span>base bodies at 60 / 75 cm (Tapered and Hybrid)</span></div>
    <div class="fact"><b>{fact_rt} min</b><span>walking runtime at 60 / 75 cm on the selected pack (target 15–20 min)</span></div>
    <div class="fact"><b>{fact_cost}</b><span>nominal prototype cost at 60 / 75 cm</span></div>
    <div class="fact"><b>{n_pass} of {len(x.ok)}</b><span>candidates pass every round-two gate</span></div>
    <div class="fact"><b>{len(picks)}</b><span>bodies proposed for verification: {", ".join(K.esc(p) for p in picks) or "none"}</span></div>
  </div>
  <nav class="toc" aria-label="Sections">
    <a href="#choices">Your choices</a><a href="#bodies">The bodies</a><a href="#sizes">Sizes and drives</a>
    <a href="#behaviors">Behaviors</a><a href="#variants">Variants</a><a href="#inside">Inside</a>
    <a href="#scorecard">Scorecard and picks</a><a href="#corrections">Corrections to round one</a>
    <a href="#questions">Questions for you</a><a href="#next">Next</a>
  </nav>
</header>"""


def choices(x: Ctx) -> str:
    zr = ((x.RC or {}).get("summary") or {}).get("zmp_error_mm_r1")
    ztxt = f"{zr[0]:.0f}–{zr[1]:.0f} mm" if zr else "centimetres"
    rows = [
        ["Size", "60 cm and 75 cm, side by side", "Both classes sized from scratch with every variant (P10)."],
        ["Body", "Tapered and Hybrid, compared as separate bodies",
         "Two base bodies per size. Twelve shape changes from an independent visual critique (F10)."],
        ["Posture", "10° chest-up", "Every base and variant at 10°, except the level variant (P12)."],
        ["Level posture", "Also research 0°, same body",
         "A fifth variant per base: the same body pitched level, so its crown sits lower (P15)."],
        ["Proportions", "Larger skull, narrower stance, compact proportions, each separately",
         "Three single-change variants per base (P13). The narrow stance is the narrowest the hip hardware allows."],
        ["Compute", "Raspberry Pi 5 + AI HAT+",
         "The 31 W Jetson budget is replaced by 9.5 W (survey figures, A14). Batteries were re-sized."],
        ["Behavior checks", "Screen the behaviors now (Stage 0)",
         "Turning, carpet and a behavior envelope were added as screens on every candidate."],
    ]
    return f"""
<section id="choices">
  <p class="kicker">1 · Your choices</p><h2>What you chose, and what round two did with it</h2>
  {K.table(["Question", "Your answer", "Round two"], rows)}
  <p class="note">Two things changed under every number on this page. The reference walks now converge: round one's
  walk generator left {ztxt} of error in its zero-moment point, which overstated some loads. Section 8 gives the
  effect on round one. Second, the compute budget is the Pi 5's, not the Jetson's.</p>
</section>"""


def bodies(x: Ctx) -> str:
    fam = x.img("families_r2.png", "Side views of the four round-two base bodies", "Tapered and Hybrid at 60 and 75 cm",
                "The four base bodies at one physical scale (10° chest-up, balanced width). Hybrid shows its ribs, spine "
                "band and segmented tail; Tapered is the lean, fully covered body.")
    common = x.img("bases_common_scale.png", "The four base bodies side by side at one scale", "Bases at one scale",
                   "Front, side and top proportions of the four bases.")
    v60 = x.img("variants_60_side.png", "60 cm base and variants, side view", "60 cm: base and single-change variants",
                "Each column changes one thing against the base: stance width, skull size, compact proportions or the "
                "level posture.")
    v75 = x.img("variants_75_side.png", "75 cm base and variants, side view", "75 cm: base and single-change variants",
                "The same five bodies at 75 cm.")
    f60 = x.img("variants_60_front.png", "60 cm base and variants, front view", "60 cm, front view",
                "Front views show the stance widths. The narrow variant moves the legs in only as far as the hip "
                "hardware allows while walking.")
    f75 = x.img("variants_75_front.png", "75 cm base and variants, front view", "75 cm, front view",
                "The same at 75 cm.")
    return f"""
<section id="bodies">
  <p class="kicker">2 · The bodies</p><h2>Tapered and Hybrid at 60 and 75 cm</h2>
  <div class="prose">
    <p>Round two's shapes come from an independent visual critique of the round-one renders. Each change was
    prototyped and screened before it went in:</p>
    <ul>
      <li>An S-curved neck carries the head over the chest front without raising the crown, with a fuller throat.</li>
      <li>The skull is deeper and broader, with a longer snout, a heavier jaw and a blunt tip. Length and crown are unchanged.</li>
      <li>Calf fairings are sized around the ankle drives. The lower thigh is held wide enough to cover the knee's side plates.</li>
      <li>The tail tapers separately at top, bottom and width, with a full-depth root and a raised tip.</li>
      <li>The arms attach at the chest flank.</li>
      <li>Tapered gets a pubic keel and a small belly tuck.</li>
      <li>Hybrid gets six swept-back ribs under a dorsal spine band and a keeled chest front.</li>
    </ul>
  </div>
  {fam}
  {common}
  {v60}
  {v75}
  <div class="grid2">{f60}{f75}</div>
</section>"""


def sizes(x: Ctx) -> str:
    rows = []
    for cid in candidate_ids():
        d = x.D.get(cid)
        if not d or not d.get("ok"):
            rows.append([mono(cid), K.esc(VARIANT_LABEL.get(variant_key(cid), "")), "failed"] + ["–"] * 7)
            continue
        m = d["metrics"]
        g, r, kind = _worst(d)
        rows.append([mono(cid), K.esc(VARIANT_LABEL.get(variant_key(cid), "")), cm(m["crown"]), cm(m["length"]),
                     kg(m["mass"]), f"{d['selection']['bus']:.0f} V, {BATTERY_NAMES.get(d['selection']['battery'], d['selection']['battery'])}",
                     f"{d['power']['total_w']:.0f} W", f"{d['runtime']['minutes']:.0f} min",
                     f"{usd(d['cost']['nominal'])} <span class='mono'>({usd(d['cost']['high'])})</span>",
                     f"{GROUP_LABEL.get(g, g)} {kind} {r:.2f}"])
    tbl = K.table(["Candidate", "Variant", "Crown", "Length", "Mass", "Bus, battery", "Walk power", "Runtime",
                   "Cost nominal (high)", "Governing drive ratio"], rows, num={2, 3, 4, 6, 7})
    drows = []
    for cid in base_ids():
        d = x.D.get(cid)
        if not d or not d.get("ok"):
            continue
        cells = [mono(cid)]
        for g in GROUPS:
            mm = d["margins"][g]
            rr = max(mm["cont_ratio"], mm["peak_ratio"], mm["speed_ratio"])
            cells.append(f"{K.esc(part(mm['actuator']))} <span class='mono'>{rr:.2f}</span>")
        drows.append(cells)
    dtbl = K.table(["Base body"] + [GROUP_LABEL[g].capitalize() for g in GROUPS], drows)
    return f"""
<section id="sizes">
  <p class="kicker">3 · Sizes and drives</p><h2>Mass, drives, runtime and cost</h2>
  <div class="prose">
    <p>Every candidate was sized by the same loop as round one: loads from the single-support holds and the two
    reference walks, then the lightest-cheapest catalog drive meeting each joint group's continuous, peak and speed
    requirement, then the battery, repeated until the selection is stable. The governing drive ratio is the highest
    requirement-to-limit ratio on the body: 1.00 would be a drive at its planning limit.</p>
  </div>
  {tbl}
  <h3>Selected drives, base bodies</h3>
  {dtbl}
  <p class="note">Ratios are the requirement over the planning limit of the governing quantity: sustained torque at
  0.57 × rated, peak at 0.70 × vendor peak with a 1.25 margin, and speed with a 1.15 margin (assumptions A1, A2).
  Runtime is the nominal walk on the selected pack at 75 % usable energy, with no regeneration credit.</p>
</section>"""


def _turn_cell(tu):
    if not tu:
        return "–"
    w = tu.get("worst_targets") or {}
    r = tu.get("worst_ratio_targets")
    g = GROUP_LABEL.get(w.get("group"), w.get("group"))
    return f"{_ok(tu.get('targets_pass'))} {K.esc(g)} {r:.2f}" if r is not None else _ok(tu.get("targets_pass"))


def behaviors(x: Ctx) -> str:
    rows = []
    for cid in candidate_ids():
        d, rv = x.D.get(cid), x.R.get(cid) or {}
        if not d or not d.get("ok"):
            continue
        env = rv.get("envelope") or {}
        tu = rv.get("turning") or {}
        ca = rv.get("carpet") or {}
        se = rv.get("sensors") or {}
        fw = env.get("fast_walk_worst") or [None, None]
        fws = "–"
        if fw[0]:
            ok = env.get("fast_walk_drives_ok")
            fws = (K.tag("inconclusive", "warn") if ok is None else _ok(ok)) + f" {K.esc(GROUP_LABEL.get(fw[0], fw[0]))} {fw[1]:.2f}"
        car = (f"{_ok(ca.get('pass'))} {ca['toe_clearance_margin_mm']:.1f} mm, μ margin {ca['friction_margin']:.2f}"
               if ca and ca.get("toe_clearance_margin_mm") is not None else "–")
        stop = env.get("stop_max_one_foot_decel_ms2")
        push = env.get("push_max_dv_ms")
        tf = (se.get("torso_follow") or {}).get("home", {})
        rows.append([mono(cid), fws, _turn_cell(tu), car,
                     "–" if stop is None else f"{stop:.2f} m/s²", "–" if push is None else f"{push:.2f} m/s",
                     _ok(tf.get("ok"), "yes", "no") if se else "–",
                     f"{(env.get('session_worst') or ['', 0])[1]:.2f}" if env.get("session_worst") else "–"])
    tbl = K.table(["Candidate", "Animal-like walk, Froude 0.05 (governing drive)", "Target turns (governing drive)",
                   "Carpet: toe above pile top, friction margin", "One-foot stop (target 1.5)", "One-step push",
                   "Person in view", "Hottest drive, session"], rows, num={4, 5, 7})
    return f"""
<section id="behaviors">
  <p class="kicker">4 · Behaviors</p><h2>Can these bodies do what you want?</h2>
  <div class="prose">
    <p>These are necessary-condition checks on each compiled model, with prescribed motions and simple balance
    models. Failing one shows that no controller can produce the behavior with that body and those drives.
    Passing one does not show that a controller can: that is the verification step.</p>
    <ul>
      <li><b>Animal-like walking</b>: a prescribed walk at Froude 0.05 (the animal's preferred walking speed class,
      stride 0.80 H), solved like the sizing walks and held against the selected drives with the sizing margins.</li>
      <li><b>Turning</b>: a 0.35 rad/s walking turn and a 0.5 rad/s turn in place (targets), plus a 0.5 rad/s walking
      turn (stretch), with friction and yaw moments at each foot and the self-collision screen on every frame.</li>
      <li><b>Carpet</b>: low, medium and high pile from sourced compression data. The swing foot's lowest point
      (toes and claws included) must clear the pile top in mid-swing with the stance sole sunk to its 98 kPa depth,
      and the friction needed must stay below the carpet band's lower edge (0.54).</li>
      <li><b>Robustness</b>: the largest deceleration one foot can hold with the ankle alone, and the largest push
      recoverable in one step under ideal balance control (an upper bound for comparing bodies).</li>
      <li><b>Endurance</b>: runtime, and each drive's RMS torque over a following session against its sustained
      limit.</li>
    </ul>
  </div>
  {tbl}
  <p class="note">Every check here uses prescribed trajectories or simple balance models (no controller, no slips,
  impacts or touchdown dynamics). Full tables: <code>round2/TABLES.md</code> §8–§10. Method and limits:
  <code>SCREENING.md</code> §8.</p>
</section>"""


def variants(x: Ctx) -> str:
    rows = []
    for h in HEIGHTS:
        for f in FAMILIES:
            base = x.D.get(f"{h}-M-{f}-10")
            if not base or not base.get("ok"):
                continue
            for w, p, po in VARIANTS[1:]:
                cid = Candidate(h, w, f, po, proportion=p).id
                d = x.D.get(cid)
                if not d or not d.get("ok"):
                    continue
                bm, dm = base["metrics"], d["metrics"]
                tb = (x.rows.get(f"{h}-M-{f}-10") or {}).get("total")
                tv = (x.rows.get(cid) or {}).get("total")
                rows.append([mono(cid), K.esc(VARIANT_LABEL[(w, p, po)]), f"{dm['mass'] - bm['mass']:+.2f} kg",
                             f"{100 * (d['margins']['hip_pitch']['hold'] / base['margins']['hip_pitch']['hold'] - 1):+.0f} %",
                             f"{100 * (d['margins']['hip_roll']['hold'] / base['margins']['hip_roll']['hold'] - 1):+.0f} %",
                             f"{d['runtime']['minutes'] - base['runtime']['minutes']:+.1f} min",
                             f"{dm['foot_spacing'] / dm['hip_height']:.2f} H", cm(dm["crown"]),
                             "–" if (tb is None or tv is None) else f"{tv - tb:+.2f}",
                             _ok(x.passes(cid)) if x.SL else "–"])
    tbl = K.table(["Variant", "Change", "Mass vs base", "Hip-pitch hold", "Hip-roll hold", "Runtime", "Foot spacing",
                   "Crown", "Score vs base", "All gates"], rows, num={2, 3, 4, 5, 6, 7, 8})
    return f"""
<section id="variants">
  <p class="kicker">5 · Variants</p><h2>What each single change buys and costs</h2>
  <div class="prose"><p>Each variant changes one thing against its base body, so its row reads as the effect of that
  change alone. The score column is the scorecard total (section 7) against the base's.</p></div>
  {tbl}
</section>"""


def inside(x: Ctx) -> str:
    cut = x.img("families_r2_cutaway.png", "Cutaways of the four base bodies", "Installed hardware",
                "Raspberry Pi 5 compute, head camera, tail-base battery and every drive, at one physical scale.")
    r1r2 = x.img("r1_vs_r2_60_cutaway.png", "Round one against round two at 60 cm, cutaway", "60 cm: round one against round two",
                 "The same 60 cm ids built with round one's generator and Jetson (left) and round two's (right).")
    srows = []
    for cid in candidate_ids():
        d = x.D.get(cid)
        if not d or not d.get("ok"):
            continue
        ov = dict(d.get("overrides") or [])
        if not ov:
            continue
        txt = []
        if "hip_spacing_add_m" in ov:
            txt.append(f"legs moved out {1000 * ov['hip_spacing_add_m']:.0f} mm per side")
        if "ankle_frac" in ov:
            txt.append(f"ankle drives lowered to {ov['ankle_frac']:.2f} of the shank")
        srows.append([mono(cid), "; ".join(txt), str(len(d.get("packaging_search") or []))])
    stbl = K.table(["Candidate", "Packaging change found by the search", "Builds screened"], srows) if srows else \
        '<p class="note">No candidate needed a packaging change.</p>'
    crow = []
    for cid in base_ids():
        d = x.D.get(cid)
        if not d or not d.get("ok"):
            continue
        cf = d.get("geometry_checks", {}).get("camera_fit", {})
        se = (x.R.get(cid) or {}).get("sensors") or {}
        crow.append([mono(cid), f"{cf.get('station', 0):.3f}" if cf else "–",
                     f"{1000 * cf.get('drop_m', 0):.0f} mm" if cf else "–",
                     f"{100 * se['camera_blocked']['home']:.1f} %" if se.get("camera_blocked") else "–",
                     f"{se.get('centre_column_nearest_visible_floor_ahead_of_toes_m', float('nan')):.2f} m" if se else "–"])
    ctbl = K.table(["Base body", "Camera station (skull length)", "Below the skull roof", "View blocked by own body",
                    "Nearest floor seen ahead of the toes"], crow, num={1, 2, 3, 4})
    return f"""
<section id="inside">
  <p class="kicker">6 · Inside</p><h2>Packaging, camera and what the search changed</h2>
  {cut}
  {r1r2}
  <div class="prose">
    <p>When a sized body fails a walking clearance gate, the round-two runner reads the cause from the clearance
    screen and changes one packaging parameter, re-sizing each time. A pelvis plate against the thigh moves the legs
    out (the narrow variants: the narrowest stance this hip hardware walks with). A thigh against the calf or the
    upper ankle drive lowers the ankle drives on the shank.</p>
  </div>
  {stbl}
  <div class="prose"><p>The head camera is now placed by a search inside the skull cover: inset by half the wall
  plus 2 mm, clear of the eyes, as high as it fits. Round one's fixed station let the cover cut the camera box.</p></div>
  {ctbl}
</section>"""


def _bar(v):
    if v is None:
        return "–"
    w = max(0.0, min(1.0, v))
    return (f'<span class="mono" style="display:inline-flex;align-items:center;gap:6px">{v:.2f}'
            f'<span style="display:inline-block;width:46px;height:6px;border-radius:3px;background:var(--band)">'
            f'<span style="display:block;width:{100 * w:.0f}%;height:6px;border-radius:3px;background:var(--accent)">'
            f'</span></span></span>')


def scorecard(x: Ctx) -> str:
    if not x.SL:
        return ""
    sl = x.SL
    picks = sl.get("picks", [])
    pick_ids = [p["id"] for p in picks]
    rows = []
    for i, cid in enumerate(sl["ranking"], 1):
        r = x.rows[cid]
        name = mono(cid) + (" " + K.tag("pick", "ok") if cid in pick_ids else "")
        ws = r["stress"]["worst_stressed_ratio"]
        rows.append([str(i), name, K.esc(r["variant"]), f"<b>{r['total']:.2f}</b>"]
                    + [_bar(r["scores"].get(c)) for c in CATS] + ["–" if ws is None else f"{ws:.2f}"])
    tbl = K.table(["Rank", "Candidate", "Variant", "Total"] + [CAT_LABEL[c] for c in CATS] + ["Worst drive ratio, stressed"],
                  rows, num={0, 3, len(CATS) + 4})
    fails = "".join(f"<li>{mono(f['id'])}: {K.esc(', '.join(GATE_LABEL.get(k, k) for k in f['gates']))}</li>"
                    for f in sl.get("failing", []))
    cards = []
    for k, p in enumerate(picks):
        r = x.rows.get(p["id"]) or {}
        d = x.D.get(p["id"]) or {}
        sc = r.get("scores", {})
        top = sorted(((v, c) for c, v in sc.items() if v is not None), reverse=True)[:2]
        low = sorted(((v, c) for c, v in sc.items() if v is not None))[:2]
        cards.append(
            f'<div class="card"><p class="kicker">Pick {k + 1}</p><h3>{K.esc(p["id"])}</h3>'
            f'<p>{K.esc(FAMILY.get(d.get("candidate", {}).get("family"), ""))}, {K.esc(r.get("variant", ""))}; '
            f'{kg(d["metrics"]["mass"]) if d.get("metrics") else ""}, {d.get("runtime", {}).get("minutes", 0):.0f} min, '
            f'{usd(d.get("cost", {}).get("nominal", 0))} nominal.</p>'
            f'<p>Total {r.get("total", 0):.2f}. Strongest: {", ".join(CAT_LABEL[c].lower() for _, c in top)}. '
            f'Weakest: {", ".join(CAT_LABEL[c].lower() for _, c in low)}.</p>'
            f'<p class="mono" style="font-size:13px">{K.esc(p["why"])}</p></div>')
    xc = ""
    if x.XC:
        xc = f"""<h3>Independent cross-check</h3><div class="prose">{x.XC.get('html', '')}</div>"""
    weights = ", ".join(f"{CAT_LABEL[c].lower()} {v:.2f}" for c, v in sl["weights"].items())
    return f"""
<section id="scorecard">
  <p class="kicker">7 · Scorecard and picks</p><h2>The three bodies most likely to reach the behaviors</h2>
  <div class="prose">
    <p>The picks come from recorded screen results only, in five steps:</p>
    <ol>
      <li><b>Gates.</b> A body is out if it fails clearance, a drive limit in the sizing walks or the target turns,
      person-in-view, 20 min runtime, the reference-walk CoP or the carpet gate.</li>
      <li><b>Margins.</b> Each behavior scores 0–1 by how far its governing quantity sits from its limit.</li>
      <li><b>Weights</b> from your priorities: {K.esc(weights)}.</li>
      <li><b>Stress test.</b> Drive ratios are re-checked with the drives 15 % weaker and the robot 10 % heavier.</li>
      <li><b>Diversity.</b> The best passing body at each height, then the best remaining body of the other family,
      so verification compares real alternatives rather than near-twins.</li>
    </ol>
    <p>Looks are yours to judge and are not scored.</p>
  </div>
  <div class="cards">{"".join(cards)}</div>
  {tbl}
  {f'<div class="prose"><p>Failing a gate:</p><ul>{fails}</ul></div>' if fails else ''}
  {xc}
</section>"""


def corrections(x: Ctx) -> str:
    rc = (x.RC or {}).get("summary")
    if not rc:
        return ""
    rr = rc["requirement_ratio_r2_over_r1"]

    def r_(g, k):
        v = rr.get(g, {}).get(k)
        return "–" if not v else (f"{v[0]:.2f}" if abs(v[1] - v[0]) < 0.005 else f"{v[0]:.2f}–{v[1]:.2f}")

    rows = [[GROUP_LABEL[g].capitalize(), r_(g, "cont"), r_(g, "peak_req"), r_(g, "speed_req")] for g in GROUPS if g in rr]
    tbl = K.table(["Joint group", "Continuous requirement", "Peak requirement", "Speed requirement"], rows, num={1, 2, 3},
                  caption="Requirement at round two's walk accuracy over round one's, on the same round-one models "
                          f"({rc['n_checked']} candidates).")
    inf1 = ", ".join(rc["walk_infeasible_r1"]) or "none"
    tus = [(x.R.get(c) or {}).get("turning") or {} for c in x.ok]
    spd_rng = _rng([(t.get("hip_yaw_speed_required") or {}).get("in_place_0.50") for t in tus], lambda v: f"{v:.1f}")
    lim_rng = _rng([t.get("hip_yaw_speed_limit") for t in tus], lambda v: f"{v:.1f}")
    return f"""
<section id="corrections">
  <p class="kicker">8 · Corrections to round one</p><h2>What round one got wrong, and what it changes</h2>
  <div class="prose">
    <p><b>The reference walk did not converge.</b> Round one's walk generator stopped correcting the centre of mass at
    0.2 mm and estimated the stride boundary's derivatives, so its zero-moment-point correction never settled:
    {rc['zmp_error_mm_r1'][0]:.0f}–{rc['zmp_error_mm_r1'][1]:.0f} mm of error remained. Round two fixes both
    ({rc['zmp_error_mm_r2'][0]:.1f}–{rc['zmp_error_mm_r2'][1]:.1f} mm). Re-checking all {rc['n_checked']} round-one
    models at the new accuracy:</p>
    <ul>
      <li>The three walks flagged as leaving the support pad ({K.esc(inf1)}) keep their centre of pressure inside it.
      Every nominal walk now has {rc['nominal_cop_margin_mm_r2'][0]:.0f}–{rc['nominal_cop_margin_mm_r2'][1]:.0f} mm of margin
      (round one: {rc['nominal_cop_margin_mm_r1'][0]:.0f} to {rc['nominal_cop_margin_mm_r1'][1]:.0f} mm). The flags
      were generator artifacts, not a property of those bodies.</li>
      <li>Continuous requirements, which set most drive choices, barely move. Peak requirements fall, most at the
      ankles; thigh-twist speed requirements fall too, but they were small.</li>
      <li>Re-selecting drives from the corrected requirements changes at least one leg drive on
      {rc['reselection_changes']} of {rc['n_checked']} bodies ({K.esc(', '.join(f"{GROUP_LABEL.get(g, g)} {n}" for g, n in sorted(rc['reselection_changes_by_group'].items(), key=lambda t: -t[1])))}).</li>
    </ul>
    {tbl}
    <p><b>Turning in place needs more thigh twist than stated.</b> Round one estimated about 0.5 rad/s of thigh twist
    for a 0.5 rad/s turn in place. The swing foot must yaw through a whole stride's turn within its swing time, so the
    round-two bodies need {spd_rng} rad/s with the 1.15 speed margin, against {lim_rng} rad/s usable from their twist
    drives. On round one's 100-W-T-10, the CubeMars AK80-64 on 24 V (selected at 100 and 120 cm) would sit at 1.12 of
    its usable speed.</p>
  </div>
</section>"""


def questions(x: Ctx) -> str:
    picks = x.picks()
    q = [
        ("Which bodies should go to verification?",
         [f"The three picks: {', '.join(picks)}." if picks else "The scorecard's picks.",
          "Swap one pick for another body (name it), for example on looks.",
          "A different set: say what matters most."]),
        ("Do the priority weights match yours?",
         ["Yes: natural walking and turning first, then robustness, following, carpet, endurance, head and tail, "
          "packaging and cost.",
          "Reorder them: say which behaviors matter more or less."]),
        ("Calf fairings: keep or drop?",
         ["Keep: the lower leg reads as an animal's, at the cost of tighter knee packaging.",
          "Drop: exposed ankle drives and linkage (lighter, easier service, more robotic look)."]),
        ("Head camera position",
         ["Keep it in the skull roof: a person stays in view, but the snout hides the floor just ahead, so near-floor "
          "sensing relies on the two ToF sensors.",
          "Add a low forward camera for the near floor (more cost, power and wiring).",
          "Lower the camera in the skull (better floor view, less of a person's torso at close range)."]),
        ("May the closed-loop controller probe (Stage 1) start?",
         ["Yes: a bounded walking and turning probe on the three picks, equal budget each, before any longer training.",
          "Not yet: another design round first."]),
    ]
    items = "".join(f'<div class="q"><h3>{i + 1}. {K.esc(t)}</h3><ol type="a">'
                    + "".join(f"<li>{K.esc(o)}</li>" for o in opts) + "</ol></div>" for i, (t, opts) in enumerate(q))
    return f"""
<section id="questions">
  <p class="kicker">9 · Questions for you</p><h2>Five decisions before verification</h2>
  <div class="prose"><p>A letter per question is enough. Nothing is frozen until you answer, and no controller
  training starts without your go-ahead.</p></div>
  {items}
</section>"""


def nextsteps(x: Ctx) -> str:
    return """
<section id="next">
  <p class="kicker">10 · Next</p><h2>From screens to evidence</h2>
  <div class="prose">
    <ul>
      <li><b>Stage 1: closed-loop probe</b> (needs your go-ahead). A bounded controller study on the picked bodies,
      with the same budget each: walking at the target speeds, turning, start and stop, and a push test on hard floor
      and on the carpet floors from this round.</li>
      <li><b>Stage 2: single-leg bench.</b> One leg of the leading body on a bench: drive temperatures over a walking
      duty cycle, the knee and ankle clearances, and the measured compute draw (bench test 1).</li>
      <li><b>Verification training.</b> Equal-budget training on the finalists against <code>BEHAVIOR_SPEC.md</code>,
      then one design is chosen.</li>
      <li><b>Not yet shown by anything here:</b> walking under a controller, measured runtime, strength, thermal
      behavior, manufacturability and following performance. Every number is a model estimate or a sourced rating.</li>
    </ul>
  </div>
</section>"""


def footer(x: Ctx) -> str:
    return """
<footer>
  <p>Generated from <code>hardware/trex_form_study</code> on branch <code>claude/mesozoic-trex-design-iteration-azm2fs</code>:
  <code>python -m designlab.round2</code>, <code>designlab.report2</code>, <code>designlab.shortlist</code>,
  <code>designlab.recheck_r1</code> and <code>designlab.review2_page</code>. Every plate is a render of a generated
  MuJoCo model. No reference figure or photograph is reproduced.</p>
</footer>"""


SECTIONS = [header, choices, bodies, sizes, behaviors, variants, inside, scorecard, corrections, questions, nextsteps,
            footer]


def build(r2: Path, r1: Path, out: Path, sections=None) -> Path:
    out.mkdir(parents=True, exist_ok=True)
    ctx = Ctx(r2, r1, out)
    body = "".join(fn(ctx) for fn in (sections or SECTIONS))
    (out / "index.html").write_text(K.page("T. rex Round Two Review", body))
    (out / "files.json").write_text(json.dumps(ctx.images, indent=1))
    return out / "index.html"


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--round2", default="round2")
    ap.add_argument("--round1", default="round1")
    ap.add_argument("--out", default="round2/review_page")
    a = ap.parse_args()
    print(build(Path(a.round2), Path(a.round1), Path(a.out)))


if __name__ == "__main__":
    main()
