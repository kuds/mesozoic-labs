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
    a, b = f(lo).replace("-", "−"), f(hi).replace("-", "−")
    return a if abs(hi - lo) < 1e-9 or a == b else (f"{a} to {b}" if lo < 0 else f"{a}–{b}")


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
        wi = r2 / "whatif_fr05"
        self.W = {q.stem: json.loads(q.read_text()) for q in sorted((wi / "candidates").glob("*.json"))} if wi.exists() else {}
        wix = wi / "review" / "review_index.json"
        self.WR = json.loads(wix.read_text()) if wix.exists() else {}
        self.wplates = wi / "review" / "plates"
        self.D1 = {}
        for cid in base_ids():
            p = r1 / "stage_a" / "candidates" / f"{cid}.json"
            if p.exists():
                self.D1[cid] = json.loads(p.read_text())
        self.plates = r2 / "review" / "plates"
        self.screen = r2 / "review" / "screening"
        self.images, self.plate_no = [], 0
        self.ok = [c for c in candidate_ids() if self.D.get(c, {}).get("ok")]
        # scorecard rows by key: a round-two id, or "<id> [<source>]" for the what-if, size-study and refinement bodies
        self.rows = {r.get("key", r["id"]): r for r in (self.SL or {}).get("rows", [])}
        # overnight refinement, size study and materials (round2/refine, round2/size_study)
        rf = r2 / "refine"
        self.RF = json.loads((rf / "refine.json").read_text()) if (rf / "refine.json").exists() else None
        self.FX = json.loads((rf / "fixed_drives.json").read_text()) if (rf / "fixed_drives.json").exists() else []
        ss = r2 / "size_study" / "size_study.json"
        self.SS = json.loads(ss.read_text()) if ss.exists() else None
        self.LIFT = {p.stem: json.loads(p.read_text()) for p in sorted(rf.glob("carpet_lift*.json"))} if rf.exists() else {}

    def record(self, key: str) -> dict:
        """Sized record behind a scorecard key: round two, or an extra directory named in the key."""
        if "[" not in key:
            return self.D.get(key, {})
        cid, src = key.split(" [", 1)
        p = self.r2.parent / src.rstrip("]") / "candidates" / f"{cid}.json"
        if not p.exists():
            p = self.r2 / Path(src.rstrip("]")).name / "candidates" / f"{cid}.json"
        if not p.exists():
            p = self.r2 / "refine" / Path(src.rstrip("]")).name / "candidates" / f"{cid}.json"
        return json.loads(p.read_text()) if p.exists() else {}

    def pick_keys(self):
        return [p.get("key", p["id"]) for p in (self.SL or {}).get("picks", [])]

    def rf_row(self, case: str, cid: str) -> dict:
        return next((r for r in (self.RF or {}).get("rows", []) if r.get("case") == case and r.get("id") == cid), {})

    def fx_row(self, case: str, cid: str) -> dict:
        return next((r for r in self.FX if r.get("case") == case and r.get("id") == cid), {})

    def ss_row(self, case: str, cid: str) -> dict:
        return next((r for r in (self.SS or {}).get("rows", []) if r.get("case") == case and r.get("id") == cid), {})

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
  turning, carpet and a behavior envelope. Overnight the leading bodies were refined one lever at a time, your
  size, materials and power questions were simulated, and a corrected scorecard proposes three bodies for
  verification. Nothing here is a final design, and no controller has been trained.</p>
  <div class="facts">
    <div class="fact"><b>{fact_mass} kg</b><span>base bodies at 60 / 75 cm (Tapered and Hybrid)</span></div>
    <div class="fact"><b>{fact_rt} min</b><span>walking runtime at 60 / 75 cm on the selected pack (target 15–20 min)</span></div>
    <div class="fact"><b>{fact_cost}</b><span>nominal prototype cost at 60 / 75 cm</span></div>
    <div class="fact"><b>{n_pass} of {len(x.ok)}</b><span>round-two candidates pass every gate of scorecard version 2 (the animal-like walk is new)</span></div>
    <div class="fact"><b>{len(RECOMMENDED)}</b><span>bodies recommended for verification: {", ".join(K.esc(k) for k, _ in RECOMMENDED)}</span></div>
  </div>
  <nav class="toc" aria-label="Sections">
    <a href="#overnight">Refined picks</a><a href="#choices">Your choices</a><a href="#bodies">The bodies</a>
    <a href="#sizes">Sizes and drives</a><a href="#behaviors">Behaviors</a><a href="#variants">Variants</a>
    <a href="#whatif">75 cm drives</a><a href="#refine">Refinement</a><a href="#sizestudy">Size study</a>
    <a href="#materials">Materials and power</a><a href="#headtail">Head and tail</a><a href="#inside">Inside</a>
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
  <p class="kicker">Your choices</p><h2>What you chose, and what round two did with it</h2>
  {K.table(["Question", "Your answer", "Round two"], rows)}
  <p class="note">Two things changed under every number on this page. The reference walks now converge: round one's
  walk generator left {ztxt} of error in its zero-moment point, which overstated some loads. <a href="#corrections">Corrections to round one</a> gives the
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
  <p class="kicker">The bodies</p><h2>Tapered and Hybrid at 60 and 75 cm</h2>
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
    dtbl = K.table(["Base body"] + [GROUP_LABEL[g][0].upper() + GROUP_LABEL[g][1:] for g in GROUPS], drows)
    crows = [(cid, x.D[cid]["cost"]["low"], x.D[cid]["cost"]["nominal"], x.D[cid]["cost"]["high"]) for cid in base_ids()
             if x.D.get(cid, {}).get("ok")]
    chart = K.cost_chart(crows) if crows else ""
    return f"""
<section id="sizes">
  <p class="kicker">Sizes and drives</p><h2>Mass, drives, runtime and cost</h2>
  <div class="prose">
    <p>Every candidate was sized by the same loop as round one: loads from the single-support holds and the two
    reference walks, then the lightest-cheapest catalog drive meeting each joint group's continuous, peak and speed
    requirement, then the battery, repeated until the selection is stable. The governing drive ratio is the highest
    requirement-to-limit ratio on the body: 1.00 would be a drive at its planning limit.</p>
  </div>
  {tbl}
  <h3>Selected drives, base bodies</h3>
  {dtbl}
  <h3>Prototype cost, base bodies</h3>
  {chart}
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
    tus = [(x.R.get(c) or {}).get("turning") or {} for c in x.ok]
    cas = [(x.R.get(c) or {}).get("carpet") or {} for c in x.ok]
    envs = [(x.R.get(c) or {}).get("envelope") or {} for c in x.ok]
    spd = _rng([(t.get("hip_yaw_speed_required") or {}).get("in_place_0.50") for t in tus], lambda v: f"{v:.1f}")
    lim = _rng([t.get("hip_yaw_speed_limit") for t in tus], lambda v: f"{v:.2f}")
    mu60 = f"{max((t.get('mu_needed_max') or 0) for c, t in zip(x.ok, tus, strict=True) if c.startswith('60')):.2f}"
    lift = _rng([((c2.get("swing") or {}).get("nominal") or {}).get("lift_frac_needed", {}).get("high") for c2 in cas],
                lambda v: f"{v:.2f}")
    stop = _rng([e.get("stop_max_one_foot_decel_ms2") for e in envs], lambda v: f"{v:.2f}")
    tbl = K.table(["Candidate", "Animal-like walk, Froude 0.05 (governing drive)", "Target turns (governing drive)",
                   "Carpet: toe above pile top, friction margin", "One-foot stop (target 1.5)", "One-step push",
                   "Person in view", "Hottest drive, session"], rows, num={4, 5, 7})
    return f"""
<section id="behaviors">
  <p class="kicker">Behaviors</p><h2>Can these bodies do what you want?</h2>
  <div class="prose">
    <p>These checks run on each compiled model with prescribed motions and simple balance models, against the
    selected drives with the sizing margins. Failing the animal-like walk or the heat check shows that the prescribed
    motion exceeds those drives at those margins; failing the stop or push bound shows what a controller must add
    (a step, or both feet down). Passing one does not show that a controller can: that is the verification step.</p>
    <ul>
      <li><b>Animal-like walking</b>: a prescribed walk at Froude 0.05 (the animal's preferred walking speed class,
      stride 0.80 H), solved like the sizing walks and held against the selected drives with the sizing margins.</li>
      <li><b>Turning</b>: a 0.35 rad/s walking turn and a 0.5 rad/s turn in place (targets), plus a 0.5 rad/s walking
      turn (stretch), with friction and yaw moments at each foot and the self-collision screen on every frame.</li>
      <li><b>Carpet</b>: low, medium and high pile from sourced compression data. The swing foot's lowest point
      (toes and claws included) must clear the pile top in mid-swing with the stance sole sunk to its 98 kPa depth,
      and the friction needed, as a force ratio, must stay below the carpet band's lower edge (0.54).</li>
      <li><b>Robustness</b>: the largest deceleration one foot can hold with the ankle alone, and the largest push
      recoverable in one step under ideal balance control (an upper bound for comparing bodies).</li>
      <li><b>Endurance</b>: runtime, and each drive's RMS torque over a following session against its sustained
      limit.</li>
    </ul>
  </div>
  {tbl}
  <div class="prose">
    <p>What the table shows:</p>
    <ul>
      <li><b>Turning works at both sizes.</b> The swing foot must yaw through a stride's whole turn within its swing
      time, so turning in place needs {spd} rad/s of thigh twist with the margin, against {lim} rad/s from the drives.
      </li>
      <li><b>Grip is the 60 cm weak spot.</b> With the stance foot's yaw moment counted, the walking turn needs
      friction up to {mu60} at 60 cm, above the carpet band's lower edge (0.54), and the brisk straight walk already
      needs 0.73–0.88. The yaw moment on a small pad, more than the turn, sets this; pad compounds are in
      <a href="#materials">Materials and power</a>.</li>
      <li><b>Carpet: the bodies fit; the swing and the 60 cm knee needed work.</b> Toe clearance passes everywhere,
      and so does friction as a force ratio, but the carpet swing target needs a lift of about {lift} H against the
      reference walks' 0.10–0.11 H. At 60 cm the knee ran out of range first; <a href="#refine">Refinement</a> gives the
      fix. Standing on medium or high pile needs active ankle balance.</li>
      <li><b>The 75 cm hip-pitch drive caps the walking speed.</b> Both sizes meet the spec's walking speeds; at the
      animal's preferred Froude 0.05 every 75 cm body runs its RobStride 03 past its planning speed on 24 V (<a href="#whatif">75 cm drives</a> has the
      fix).</li>
      <li><b>Hard stops need a step or both feet.</b> One foot holds {stop} m/s² with the ankle alone, below the
      spec's 1.5 m/s² limit, so a stop at that limit needs double support or a step.</li>
    </ul>
  </div>
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
  <p class="kicker">Variants</p><h2>What each single change buys and costs</h2>
  <div class="prose"><p>Each variant changes one thing against its base body, so its row reads as the effect of that
  change alone. The score column is the scorecard total (<a href="#scorecard">Scorecard</a>) against the base's.</p></div>
  {tbl}
</section>"""


def whatif(x: Ctx) -> str:
    if not x.W:
        return ""
    rows = []
    for cid in ("75-M-T-10", "75-M-H-10"):
        b = x.D.get(cid)
        if not b:
            continue
        for label, w, rv in (("what-if (Froude 0.05 sizing)", x.W.get(cid), x.WR.get(cid) or {}),
                             ("targeted upgrade (+15 % on neck and hip roll)", x.record(f"{cid} [refine/targeted_75]"),
                              (x.rows.get(f"{cid} [refine/targeted_75]") or {}))):
            if not w or not w.get("ok"):
                continue
            ch = [f"{GROUP_LABEL[g]}: {K.esc(part(b['margins'][g]['actuator']))} → {K.esc(part(w['margins'][g]['actuator']))}"
                  for g in GROUPS if b["margins"][g]["actuator"] != w["margins"][g]["actuator"]]
            g, r, kind = _worst(w)
            sc = x.rows.get(f"{cid} [whatif_fr05]" if label.startswith("what") else f"{cid} [refine/targeted_75]") or {}
            rows.append([mono(cid), K.esc(label), "<br>".join(ch) or "none",
                         f"{kg(b['metrics']['mass'])} → {kg(w['metrics']['mass'])}",
                         f"{b['runtime']['minutes']:.1f} → {w['runtime']['minutes']:.1f} min",
                         f"{usd(b['cost']['nominal'])} → {usd(w['cost']['nominal'])}",
                         f"{K.esc(GROUP_LABEL.get(g, g))} {kind} {r:.2f}",
                         "–" if sc.get("total") is None else f"{sc['total']:.2f}"])
    tbl = K.table(["Body", "Sizing", "Drives that change", "Mass", "Runtime", "Cost nominal", "Governing drive",
                   "Score (v2)"], rows, num={7})
    plate = x.img("75-M-T-10_cutaway.png", "Cutaway of 75-M-T-10 sized for the Froude 0.05 walk",
                  "75-M-T-10 sized for the animal-like walk",
                  "The what-if body with its MyActuator X6-60 hip pitch and X4-36-L thigh twist.", src_dir=x.wplates)
    return f"""
<section id="whatif">
  <p class="kicker">75 cm drives</p><h2>75 cm, sized for the animal-like walk</h2>
  <div class="prose"><p>Every 75 cm body meets the spec's walking speeds but not the animal's preferred Froude 0.05: its
  RobStride 03 hip pitch runs past its planning speed on the 24 V bus. Re-sizing the two 75 cm bases with that walk
  as a sizing case (the what-if) fixes it, but the neck servo, the knee and the hip roll stay within a few per cent of
  their limits. The targeted upgrade, which every cross-check judge asked for, adds 15 % headroom on the neck servo
  and the hip roll: four drives change, the body gets lighter, and the 75 cm scorecard rises level with the 60 cm
  bodies.</p></div>
  {tbl}
  {plate}
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
  <p class="kicker">Inside</p><h2>Packaging, camera and what the search changed</h2>
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


def _md(text: str) -> str:
    """Minimal Markdown to HTML for the cross-check summary: paragraphs, `code` and **bold**."""
    import re

    out = []
    for para in [q.strip() for q in text.split("\n\n") if q.strip()]:
        h = K.esc(para)
        h = re.sub(r"`([^`]+)`", r"<code>\1</code>", h)
        h = re.sub(r"\*\*([^*]+)\*\*", r"<b>\1</b>", h)
        out.append(f"<p>{h}</p>")
    return "".join(out)


def _crosscheck(x: Ctx) -> str:
    c = (x.XC or {}).get("compare")
    if not c:
        return ""
    rows = [["Scorecard, version 1", ", ".join(mono(i) for i in c.get("scorecard_picks", []))]]
    rows += [[f"Judge: {K.esc(k)}", ", ".join(mono(i) for i in v)] for k, v in (c.get("judge_picks") or {}).items()]
    rows.append(["<b>Comparator's recommendation</b>", ", ".join(mono(i) for i in c.get("recommended_picks", []))])
    errs = "".join(f"<li>{K.esc(e.split('. ')[0].rstrip('.'))}.</li>" for e in c.get("errors_found", [])[:6])
    return f"""<h3>Independent cross-check</h3>
  <div class="prose"><p>Three judges, each with one lens (behavior risk, hardware, your stated priorities), chose three
  bodies blind to the scorecard; an adversarial comparator checked every disagreement against the records. Full
  record: <code>round2/CROSSCHECK.md</code>.</p></div>
  {K.table(["Who", "Picks"], rows)}
  <div class="prose">{_md(c.get("summary_md", ""))}
  <p>Errors it found (version 2 of the scorecard fixes the scorecard's own; the refined 75 cm body adds the stronger
  neck servo the judges asked for):</p><ul>{errs}</ul></div>"""


def scorecard(x: Ctx) -> str:
    if not x.SL:
        return ""
    sl = x.SL
    picks = sl.get("picks", [])
    pick_keys = [p.get("key", p["id"]) for p in picks]
    rows = []
    for i, key in enumerate(sl["ranking"], 1):
        r = x.rows[key]
        name = mono(key) + (" " + K.tag("pick", "ok") if key in pick_keys else "")
        ws = r["stress"]["worst_stressed_ratio"]
        rows.append([str(i), name, K.esc(r["variant"]), f"<b>{r['total']:.2f}</b>"]
                    + [_bar(r["scores"].get(c)) for c in CATS] + ["–" if ws is None else f"{ws:.2f}"])
    tbl = K.table(["Rank", "Body", "Variant", "Total"] + [CAT_LABEL[c] for c in CATS] + ["Worst leg-drive ratio, stressed"],
                  rows, num={0, 3, len(CATS) + 4})
    fails = "".join(f"<li>{mono(f.get('key', f.get('id')))}: {K.esc(', '.join(GATE_LABEL.get(k, k.replace('_', ' ')) for k in f['gates'] if k != 'all'))}</li>"
                    for f in sl.get("failing", []))
    cards = []
    for k, p in enumerate(picks):
        key = p.get("key", p["id"])
        r = x.rows.get(key) or {}
        d = x.record(key)
        sc = r.get("scores", {})
        top = sorted(((v, c) for c, v in sc.items() if v is not None), reverse=True)[:2]
        low = sorted(((v, c) for c, v in sc.items() if v is not None))[:2]
        cards.append(
            f'<div class="card"><p class="kicker">Pick {k + 1}</p><h3>{K.esc(key)}</h3>'
            f'<p>{K.esc(FAMILY.get(d.get("candidate", {}).get("family"), ""))}, {K.esc(r.get("variant", ""))}; '
            f'{kg(d["metrics"]["mass"]) if d.get("metrics") else ""}, {d.get("runtime", {}).get("minutes", 0):.0f} min, '
            f'{usd(d.get("cost", {}).get("nominal", 0))} nominal.</p>'
            f'<p>Total {r.get("total", 0):.2f}. Strongest: {", ".join(CAT_LABEL[c].lower() for _, c in top)}. '
            f'Weakest: {", ".join(CAT_LABEL[c].lower() for _, c in low)}.</p>'
            f'<p class="mono" style="font-size:13px">{K.esc(p["why"])}</p></div>')
    xc = _crosscheck(x)
    weights = ", ".join(f"{CAT_LABEL[c].lower()} {v:.2f}" for c, v in sl["weights"].items())
    return f"""
<section id="scorecard">
  <p class="kicker">Scorecard and picks</p><h2>The three bodies most likely to reach the behaviors</h2>
  <div class="prose">
    <p>Version {sl.get("version", 1)} of the scorecard, after the independent cross-check below. It ranks the 20
    candidates together with the what-if and refined bodies, from recorded screen results only:</p>
    <ol>
      <li><b>Gates.</b> A body is out if it fails clearance, a drive limit in the sizing walks or the target turns,
      the animal-like Froude 0.05 walk within its drives (new in version 2), person-in-view, 20 min runtime, the
      reference-walk CoP or the carpet gate.</li>
      <li><b>Margins.</b> Each behavior scores 0–1 by how far its governing quantity sits from its limit. Each drive
      counts once: the leg drives set walking, turning and endurance; the neck and tail servos set head and tail.</li>
      <li><b>Weights</b> from your priorities: {K.esc(weights)}.</li>
      <li><b>Stress test</b>, now part of the robustness score: torque ratios with the robot 10 % heavier and the
      drives 15 % weaker, speed ratios with the drives 15 % slower.</li>
      <li><b>Shortlist.</b> The best passing body at each height, then the other family's counterpart of the best
      overall, so verification compares size on one family and family at one size.</li>
    </ol>
    <p>Looks are yours to judge and are not scored. Totals within about 0.02 of each other are ties at this
    resolution.</p>
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
        if not v:
            return "–"
        a, b = f"{v[0]:.2f}", f"{v[1]:.2f}"
        return a if a == b else f"{a}–{b}"

    ch = rc.get("reselection_changed_ids") or {}
    kinds = (("an ankle drive", ("ankle_a", "ankle_b")), ("the tail drive", ("tail",)), ("the knee", ("knee",)),
             ("the hip pitch", ("hip_pitch",)), ("the hip roll", ("hip_roll",)), ("the thigh twist", ("hip_yaw",)),
             ("the neck servo", ("neck",)))
    parts = [f"{lab} on {n}" for lab, gs in kinds if (n := sum(1 for v in ch.values() if any(g in v for g in gs)))]
    reselect = ", ".join(parts)
    m1 = f"{rc['nominal_cop_margin_mm_r1'][0]:+.0f} to {rc['nominal_cop_margin_mm_r1'][1]:+.0f}".replace("-", "−")

    rows = [[GROUP_LABEL[g][0].upper() + GROUP_LABEL[g][1:], r_(g, "cont"), r_(g, "peak_req"), r_(g, "speed_req")]
            for g in GROUPS if g in rr]
    tbl = K.table(["Joint group", "Continuous requirement", "Peak requirement", "Speed requirement"], rows, num={1, 2, 3},
                  caption="Requirement at round two's walk accuracy over round one's, on the same round-one models "
                          f"({rc['n_checked']} candidates).")
    inf1 = ", ".join(rc["walk_infeasible_r1"]) or "none"
    tus = [(x.R.get(c) or {}).get("turning") or {} for c in x.ok]
    spd_rng = _rng([(t.get("hip_yaw_speed_required") or {}).get("in_place_0.50") for t in tus], lambda v: f"{v:.1f}")
    lim_rng = _rng([t.get("hip_yaw_speed_limit") for t in tus], lambda v: f"{v:.1f}")
    return f"""
<section id="corrections">
  <p class="kicker">Corrections to round one</p><h2>What round one got wrong, and what it changes</h2>
  <div class="prose">
    <p><b>The reference walk did not converge.</b> Round one's walk generator stopped correcting the centre of mass at
    0.2 mm and estimated the stride boundary's derivatives, so its zero-moment-point correction never settled:
    {rc['zmp_error_mm_r1'][0]:.0f}–{rc['zmp_error_mm_r1'][1]:.0f} mm of error remained. Round two fixes both
    ({rc['zmp_error_mm_r2'][0]:.1f}–{rc['zmp_error_mm_r2'][1]:.1f} mm). Re-checking all {rc['n_checked']} round-one
    models at the new accuracy:</p>
    <ul>
      <li>The three walks flagged as leaving the support pad ({K.esc(inf1)}) keep their centre of pressure inside it.
      Every nominal walk now has {rc['nominal_cop_margin_mm_r2'][0]:.0f}–{rc['nominal_cop_margin_mm_r2'][1]:.0f} mm of margin
      (round one: {m1} mm), and no brisk walk leaves
      the pad (round one: {rc['brisk_cop_off_pad_r1']} of {rc['n_checked']}). The flags were generator artifacts, not a
      property of those bodies.</li>
      <li>Continuous requirements, which set most drive choices, move by less than 2 %. Peak requirements fall, most
      at the ankles and the knee; thigh-twist speed requirements fall too, but they were small.</li>
      <li>Re-selecting parts from the corrected requirements changes at least one on {rc['reselection_changes']} of
      {rc['n_checked']} bodies: {K.esc(reselect)}. The holds and continuous loads behind round one's size and family
      comparisons are unchanged.</li>
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
    keys = x.pick_keys()
    sg = lambda v: "+" if v >= 0 else "−"  # noqa: E731

    def delta(key, base):
        d, b = x.record(key), x.D.get(base, {})
        if not d or not b:
            return ""
        dm, dc = d["metrics"]["mass"] - b["metrics"]["mass"], (d["cost"]["nominal"] - b["cost"]["nominal"]) / 1000
        dr = d["runtime"]["minutes"] - b["runtime"]["minutes"]
        return f" ({sg(dc)}${abs(dc):.1f}k nominal, {sg(dm)}{abs(dm):.2f} kg, {sg(dr)}{abs(dr):.1f} min against round two)"

    tgt = delta("75-M-T-10 [refine/targeted_75]", "75-M-T-10")
    kn = x.record("75-M-T-10 [refine/knee_75]")
    tg = x.record("75-M-T-10 [refine/targeted_75]")
    knee = ""
    if kn and tg:
        knee = (f" The knee's speed falls from {tg['margins']['knee']['speed_ratio']:.2f} to {kn['margins']['knee']['speed_ratio']:.2f} "
                f"of its limit, for {kn['metrics']['mass'] - tg['metrics']['mass']:+.2f} kg and "
                f"{kn['runtime']['minutes'] - tg['runtime']['minutes']:+.1f} min.").replace("-", "−")
    nc = x.rf_row("no_calf", "60-M-T-10")
    calf = ""
    if nc and nc.get("ok"):
        b = x.D.get("60-M-T-10", {})
        calf = f" (60 cm: {nc['mass_kg'] - b['metrics']['mass']:+.2f} kg)".replace("-", "−") if b else ""
    q = [
        ("Which bodies go to verification?",
         [f"The recommended three: {', '.join(k for k, _ in RECOMMENDED)}.",
          f"The scorecard's own picks: {', '.join(keys)}." if keys else "The scorecard's own picks.",
          "Swap one (name it), for example on looks or the narrow stance 60-N-T-10 to test stance width.",
          "A different set: say what matters most."]),
        ("Do the priority weights match yours?",
         ["Yes: natural walking and turning first, then robustness, following, carpet, endurance, head and tail, "
          "packaging and cost.",
          "Reorder them: say which behaviors matter more or less."]),
        ("If a 75 cm body goes forward, which drives?",
         [f"The targeted upgrade: Froude 0.05 sizing plus 15 % headroom on the neck servo and hip roll{tgt}.",
          "The targeted upgrade plus a faster RobStride 06 knee, for knee-speed headroom." + knee,
          "Keep the round-two sizing (the spec's speeds, not the animal's)."]),
        ("Calf fairings: keep or drop?",
         ["Keep: the lower leg reads as an animal's; at 60 cm the ankle drives move down to 0.60 of the shank so the "
          "knee can bend enough for the carpet swing.",
          "Drop: exposed ankle drives and push rods, lighter, easier to service, more robotic" + calf + "."]),
        ("Coverings material?",
         ["MJF PA11 with graded walls about a quarter thinner (recommended: tougher, lighter, about today's cost).",
          "Keep MJF PA12 at today's walls.",
          "Carbon skins on the head, neck and tail as well (lighter neck payload; moulds, weeks and $1.4–4.1k, estimate)."]),
        ("Power: battery, tether or both?",
         ["Battery plus an external DC input (recommended; as your power requirement already asks).",
          "Tether-only (not recommended: the pack is also the tail-base counterweight)."]),
        ("Head camera position",
         ["Keep it in the skull roof: a person stays in view; the near floor relies on the two ToF sensors.",
          "Add a low forward camera for the near floor (more cost, power and wiring).",
          "Move the camera forward and lower: a better floor view, less of a close person's torso."]),
        ("May the closed-loop probe (Stage 1) start?",
         ["Yes: a bounded walking and turning probe on the chosen bodies, with the same budget each.",
          "Not yet: another design round first."]),
    ]
    items = "".join(f'<div class="q"><h3>{i + 1}. {K.esc(t)}</h3><ol type="a">'
                    + "".join(f"<li>{K.esc(o)}</li>" for o in opts) + "</ol></div>" for i, (t, opts) in enumerate(q))
    return f"""
<section id="questions">
  <p class="kicker">Questions for you</p><h2>Eight decisions before verification</h2>
  <div class="prose"><p>A letter per question is enough. Nothing is frozen until you answer, and no controller
  training starts without your go-ahead.</p></div>
  {items}
</section>"""


def nextsteps(x: Ctx) -> str:
    return """
<section id="next">
  <p class="kicker">Next</p><h2>From screens to evidence</h2>
  <div class="prose">
    <ul>
      <li><b>Stage 1: closed-loop probe</b> (needs your go-ahead). A bounded controller study on the picked bodies,
      with the same budget each: walking at the target speeds, turning, start and stop, and a push test on hard floor
      and on the carpet floors from this round.</li>
      <li><b>Bench tests that need no robot.</b> Friction and twist tests of the chosen foot-pad compound at the
      robot's pad pressure on your floors (about a day and $100, estimate), and the Pi 5 set's draw at the converter
      input with the detector, camera and logging running.</li>
      <li><b>Stage 2: single-leg bench.</b> One leg of the leading body on a bench: drive temperatures over a walking
      duty cycle, the knee and ankle clearances with the ankle drives at their new station, and the knee's speed in
      the animal-like walk, the limit that now governs both refined bodies.</li>
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
  <code>python -m designlab.round2</code>, <code>designlab.report2</code>, <code>designlab.whatif</code>,
  <code>designlab.sizestudy</code>, <code>designlab.refine</code>, <code>designlab.shortlist</code>,
  <code>designlab.recheck_r1</code> and <code>designlab.review2_page</code>. Every plate is a render of a generated
  MuJoCo model. No reference figure or photograph is reproduced.</p>
</footer>"""


# ------------------------------------------------------------------------------------------------
# Overnight refinement (owner questions after the round-two package)
# ------------------------------------------------------------------------------------------------
def _gov(r: dict) -> str:
    g = r.get("governing") or [None, None, None]
    return f"{K.esc(GROUP_LABEL.get(g[0], g[0] or ''))} {K.esc(g[1] or '')} {g[2]:.2f}" if g[0] else "–"


def _fw(r: dict) -> str:
    fw = r.get("fast_walk") or {}
    w = fw.get("worst") or [None, None]
    if not w[0]:
        return "–"
    txt = f"{K.esc(GROUP_LABEL.get(w[0], w[0]))} {w[1]:.2f}"
    return (K.tag("within", "ok") if fw.get("ok") else K.tag("over", "bad")) + " " + txt


def _gates(r: dict) -> str:
    gt = r.get("gates") or {}
    if not gt:
        return "–"
    if gt.get("all"):
        return K.tag("all pass", "ok")
    bad = [GATE_LABEL.get(k, k.replace("_", " ")) for k, v in gt.items() if v is False and k != "all"]
    return K.tag("fails", "bad") + " " + K.esc(", ".join(bad))


def _cmp_table(rows: list[dict], label_of) -> str:
    out = []
    for r in rows:
        if not r or not r.get("ok"):
            continue
        out.append([mono(r["id"]), K.esc(label_of(r)), kg(r["mass_kg"]), _gov(r), _fw(r),
                    f"{r['runtime_min']:.0f} min", f"{usd(r['cost_nominal'])} <span class='mono'>({usd(r['cost_high'])})</span>",
                    "–" if r.get("tightest_joint_margin_rad") is None else f"{r['tightest_joint_margin_rad']:.2f} rad",
                    "–" if r.get("total") is None else f"{r['total']:.2f}", _gates(r)])
    return K.table(["Body", "Case", "Mass", "Governing drive", "Froude 0.05 walk", "Runtime", "Cost nominal (high)",
                    "Tightest joint margin", "Score", "Gates (version 2)"], out, num={2, 5, 7, 8})


def _lift_rows(name: str) -> list:
    out = []
    for r in x_lift(name):
        for L, v in r["lifts"].items():
            tj = v.get("tightest_joint") or [None, None]
            out.append([mono(r["id"]), f"{L} H",
                        f"{r['carpet_target_mm']:.1f} mm", f"{v['min_above_pile_top_mm']:.1f} mm",
                        _ok(v["meets_target_all_piles"], "met", "not met"),
                        f"{_ok(v['drives_within_limits'], 'within', 'over')} {K.esc(GROUP_LABEL.get(v['worst_group'], v['worst_group']))} {v['worst_ratio']:.2f}",
                        _ok(v["walking_gates"]),
                        "–" if tj[0] is None else f"{K.esc(tj[0].replace('_', ' '))} {tj[1]:.2f} rad"])
    return out


_LIFT_CACHE: dict = {}


def x_lift(name: str) -> list:
    return _LIFT_CACHE.get(name, [])


#: The review's recommended verification set (ROUND2_REVIEW.md §6), chosen from the scorecard's tied leaders for
#: the reasons given; the scorecard's own picks are shown in its section, and the owner decides (question 1).
RECOMMENDED = (
    ("60-M-T-10 [refine/carpet_60]", "The 60 cm anchor: the cleanest packaging of the leaders (4 mm in every turn, "
     "the widest joint-range margin, 0.20 rad), and with its ankle drives lowered it reaches the carpet swing target "
     "within its drives and knee range."),
    ("75-M-T-10 [refine/targeted_75]", "The 75 cm body after the targeted drive upgrade: every gate passes, it "
     "carries the carpet swing at 0.18 H, it needs the least grip of the leaders and it is the only one that stands "
     "on medium pile with its servos simply held. The faster-knee variant ties with it on the scorecard; question 3."),
    ("60-M-H-10 [refine/carpet_60]", "The family comparison at 60 cm, where the drive margins do not hide the "
     "difference between the bodies; the cross-check's choice for the third slot."),
)


def overnight(x: Ctx) -> str:
    cards = []
    for k, (key, why) in enumerate(RECOMMENDED):
        r, d = x.rows.get(key) or {}, x.record(key)
        if not d:
            continue
        src = key.split(" [", 1)[1].rstrip("]") if "[" in key else "round two"
        g, gr, kind = _worst(d)
        cards.append(
            f'<div class="card"><p class="kicker">Recommended {k + 1}</p><h3>{K.esc(d["id"])}</h3>'
            f'<p class="mono" style="font-size:13px">{K.esc(src)}</p>'
            f'<p>{K.esc(FAMILY.get(d["candidate"]["family"], ""))}, {cm(d["metrics"]["crown"])} crown; '
            f'{kg(d["metrics"]["mass"])}, {d["runtime"]["minutes"]:.0f} min, {usd(d["cost"]["nominal"])} nominal '
            f'({usd(d["cost"]["high"])} high).</p>'
            f'<p>Score {r.get("total", 0):.2f}; governing drive {K.esc(GROUP_LABEL.get(g, g))} {kind} {gr:.2f}.</p>'
            f'<p style="font-size:14px;color:var(--ink-2)">{K.esc(why)}</p></div>')
    if not cards:
        return ""
    sk = x.pick_keys()
    lead = [x.rows[k] for k in (x.SL or {}).get("ranking", [])[:4] if k in x.rows]
    spread = (max(r["total"] for r in lead) - min(r["total"] for r in lead)) if lead else None
    note = (f"The scorecard's own rule picks {', '.join(mono(k) for k in sk)}; its top four bodies are within "
            f"{spread:.3f} of each other, a tie at its resolution. The recommendation chooses among the tied leaders "
            f"for reasons the scorecard does not score, given on each card.") if sk and spread is not None else ""
    return f"""
<section id="overnight">
  <p class="kicker">Overnight · 7–8 Nov</p><h2>The recommended bodies</h2>
  <div class="prose"><p>Overnight the leading bodies were refined one lever at a time, each re-sized and screened
  exactly like the 20 candidates, and the scorecard was corrected after an independent cross-check (version 2). The
  sections after the round-two results give the refinements (<a href="#refine">Refinement</a>), your size questions
  (<a href="#sizestudy">Size study</a>), materials and power (<a href="#materials">Materials and power</a>) and head
  and tail motion (<a href="#headtail">Head and tail</a>).</p></div>
  <div class="cards">{"".join(cards)}</div>
  {f'<p class="note">{note}</p>' if note else ''}
</section>"""


REFINE_SHOW = ("carpet_60", "targeted_75", "knee_60", "knee_75", "no_calf", "pads_large", "toes_long", "foot_both",
               "narrow_3mm", "headroom_60")


def refinement(x: Ctx) -> str:
    if not x.RF:
        return ""
    labels = x.RF.get("cases", {})
    base = [r for r in x.RF["rows"] if r.get("case") == "round2" and r.get("id") in ("60-M-T-10", "60-M-H-10", "75-M-T-10", "75-M-H-10")]
    rows = base + [r for c in REFINE_SHOW for r in x.RF["rows"] if r.get("case") == c]
    tbl = _cmp_table(rows, lambda r: "round two" if r.get("case") == "round2" else labels.get(r["case"], r["case"]))
    for name, data in x.LIFT.items():
        _LIFT_CACHE[name] = data
    lifts = []
    for name in ("carpet_lift_carpet_60", "carpet_lift_carpet_60_0.195", "carpet_lift_targeted_75",
                 "carpet_lift_targeted_75_0.175_0.180"):
        lifts += _lift_rows(name)
    ltbl = K.table(["Body", "Swing lift", "Carpet target", "Lowest point above a pile top", "Target", "Drives (nominal and brisk walks lifted)",
                    "Walking clearance gates", "Tightest joint margin"], lifts, num={2, 3}) if lifts else ""
    rp = x.r2 / "refine" / "plates"
    side = x.img("refined_side.png", "Round two against the refined bodies, side view", "Round two and the refined bodies",
                 "Left: the round-two body. Right: the refined body (60 cm: ankle drives lowered to 0.60 of the shank; "
                 "75 cm: the targeted drive upgrade).", src_dir=rp) if (rp / "refined_side.png").exists() else ""
    cut = x.img("refined_cutaway.png", "Round two against the refined bodies, cutaway", "Installed hardware, round two and refined",
                "The same bodies with the covers cut away.", src_dir=rp) if (rp / "refined_cutaway.png").exists() else ""
    return f"""
<section id="refine">
  <p class="kicker">Overnight · Refinement</p><h2>One lever at a time on the leading bodies</h2>
  <div class="prose">
    <p>Each row changes one lever on a leading body, re-sizes it with the round-two rules and screens it like the
    candidates. Version 2 of the scorecard gates the animal-like Froude 0.05 walk, so the 75 cm bodies as sized
    fail it.</p>
    <ul>
      <li><b>Carpet-ready 60 cm.</b> The 60 cm knee ran out of range before the carpet swing (the thigh meets the
      calf fairing). Lowering the ankle drives to 0.60 of the shank opens the knee to about 1.0 rad with the
      fairing kept; sized with the carpet swing (0.19 H lift) and the Froude 0.05 walk, the drives do not change.</li>
      <li><b>Targeted 75 cm.</b> The Froude 0.05 walk as a sizing case plus 15 % headroom on the neck and hip roll:
      four drives change, the mass falls and every gate passes. The same body already carries a 0.18 H carpet swing.</li>
      <li><b>Knee headroom.</b> On both refined bodies the governing limit is the knee's speed (0.95). A faster
      RobStride 06 knee removes it but costs 0.7–0.9 kg and runtime.</li>
      <li><b>Feet, calf fairing, narrow stance, blanket headroom:</b> see the table.</li>
    </ul>
  </div>
  {tbl}
  {side}
  {cut}
  <h3>Carpet swing height</h3>
  <div class="prose"><p>The carpet target is 0.06 H above the pile top (at least 15 mm). For each lift, both reference
  walks are regenerated with that lift and checked against the selected drives and the walking clearance gates.</p></div>
  {ltbl}
  {_nominal_note(x)}
</section>"""


def _nominal_note(x: Ctx) -> str:
    p = x.r2 / "refine" / "carpet_lift_carpet_60_nominal_only.json"
    if not p.exists():
        return ""
    d = json.loads(p.read_text())
    v = [r["knee_speed"] for r in d.values()]
    return (f'<p class="note">The drives column covers both walks lifted. At 60 cm the nominal walk alone at 0.19–0.195 H '
            f'keeps the knee at {min(v):.2f}–{max(v):.2f} of its speed limit '
            f'(<code>round2/refine/carpet_lift_carpet_60_nominal_only.json</code>), so there the brisk walk is the one over, '
            f'and on carpet the 60 cm robot walks at the nominal speed with that swing.</p>')


def size_study_sec(x: Ctx) -> str:
    if not x.SS:
        return ""
    labels = x.SS.get("cases", {})
    order = ("round2", "small_resized", "small_60drives", "large_headroom", "80_resized", "80_headroom")
    rows = [r for c in order for r in x.SS["rows"] if r.get("case") == c]
    rows += [x.rf_row("targeted_75", c) for c in ("75-M-T-10", "75-M-H-10")]
    tbl = _cmp_table(rows, lambda r: "round two" if r.get("case") == "round2" else
                     labels.get(r["case"], (x.RF or {}).get("cases", {}).get(r["case"], r["case"])))
    s57 = [x.ss_row("small_resized", c) for c in ("57-M-T-10", "57-M-H-10")]
    s54 = [x.ss_row("small_resized", c) for c in ("54-M-T-10", "54-M-H-10")]
    b60 = [x.D.get(c, {}) for c in ("60-M-T-10", "60-M-H-10")]
    d57 = _rng([a["mass_kg"] - b["metrics"]["mass"] for a, b in zip(s57, b60, strict=True) if a and b], lambda v: f"{-v:.2f}")
    d54 = _rng([a["mass_kg"] - b["metrics"]["mass"] for a, b in zip(s54, b60, strict=True) if a and b], lambda v: f"{-v:.2f}")
    return f"""
<section id="sizestudy">
  <p class="kicker">Overnight · Size study</p><h2>A smaller 60 cm, larger 75 cm motors, or 80 cm?</h2>
  <div class="prose">
    <p><b>A 5–10 % smaller 60 cm robot does not gain margin.</b> It saves only {d57} kg at 57 cm and {d54} kg at
    54 cm, because about half the mass is catalog hardware that does not shrink. The hip-roll motor then overlaps
    the power board and the two ToF sensors overlap, and the knee runs faster: a smaller robot steps faster for the
    same Froude number, so at 54 cm the knee exceeds its speed limit in the animal-like walk.</p>
    <p><b>Larger motors at 75 cm: yes, targeted, and no 80 cm needed.</b> Four drive changes fix the 75 cm body
    (the targeted rows). Upsizing every drive by 20 % snowballs to about 20 kg and a 12S pack with no removal path;
    80 cm hits the same limits as 75 cm and gains nothing.</p>
  </div>
  {tbl}
</section>"""


def materials(x: Ctx) -> str:
    if not x.FX:
        return ""
    from .refine import MAT_LEVERS

    labels = {f"mat{h}_{k}": lab for h in (60, 75) for k, _, lab in MAT_LEVERS}
    order = ("cfqi", "cfqi4", "cf", "thinal", "thin", "pa11", "lightqi", "best", "tether")
    rows = []
    for h, cid in ((60, "60-M-T-10"), (75, "75-M-T-10")):
        for k in order:
            r = x.fx_row(f"mat{h}_{k}", cid)
            if not r:
                continue
            lw, rlw = r["leg_worst"], r["reference_leg_worst"]
            rt = "tethered" if r["tether"] else f"{r['runtime_min']:.1f} min"
            dm = r["mass_kg"] - r["reference_mass_kg"]
            dc = (r["cost"]["nominal"] - r["reference_cost"]["nominal"]) / 1000
            dcs = (f"{1000 * dc:+.0f}" if abs(dc) < 0.1 else f"{dc:+.1f}k").replace("+", "+$").replace("-", "−$")
            rows.append([f"{h} cm", K.esc(labels.get(r["case"], r["case"])),
                         f"{dm:+.2f} kg".replace("-", "−"), f"{r['walk_w']:.0f} W", rt, dcs,
                         f"{K.esc(GROUP_LABEL.get(lw[0], lw[0]))} {lw[2]:.2f}" + (" " + K.tag("over", "bad") if lw[2] > 1 else ""),
                         f"{r['ratios']['neck']['cont']:.2f}", f"{r['ratios']['tail']['peak']:.2f}",
                         f"{1e3 * r['com_ahead_of_hip_m']:.0f} mm"])
    th = [x.fx_row(f"mat{h}_thinal", c) for h, c in ((60, "60-M-T-10"), (75, "75-M-T-10"))]
    th = [r for r in th if r]
    thin = (_rng([r["reference_mass_kg"] - r["mass_kg"] for r in th], lambda v: f"{v:.2f}") + " kg") if th else "part of it"
    tbl = K.table(["Size", "Lever", "Mass change", "Walking power", "Runtime", "Cost change (nominal)",
                   "Governing leg drive", "Neck servo", "Tail servo peak", "CoM ahead of hip"], rows, num={2, 3, 4, 5, 7, 8, 9},
                  caption="Each lever on the carpet-ready 60 cm body (10.83 kg, 71 W, 46.5 min, $14.8k; knee speed 0.95, "
                          "CoM 34 mm ahead of the hip) or the targeted 75 cm body (14.58 kg, 88 W, 37.6 min, $18.4k; knee "
                          "speed 0.95, CoM 64 mm), keeping that body's drives and pack.")
    big = []
    for cid in ("100-M-T-10", "120-M-T-10", "120-M-H-10"):
        p = x.r1 / "stage_a" / "candidates" / f"{cid}.json"
        if p.exists():
            d = json.loads(p.read_text())
            w = d["power"]["total_w"] - d["power"]["aux"]["input_w"] + 9.5 / 0.88 + (4.0 + sum(
                v for k, v in d["power"]["aux"]["loads"].items() if k == "sensors")) / 0.88
            big.append((cid, d["metrics"]["mass"], w, d["selection"]["bus"]))
    brow = []
    packs = (("6S2P Li-ion, 200 Wh", 199.8, 0.861, 24.0), ("12S2P Li-ion, 400 Wh", 399.6, 1.721, 48.0))
    for cid, m, w, bus in big:
        for name, wh, pk, v in packs:
            if (bus > 30) != (v > 30):
                continue
            brow.append([mono(cid), kg(m), f"{w:.0f} W", K.esc(name), f"{pk:.2f} kg ({100 * pk / m:.1f} %)",
                         f"{60 * wh * 0.75 / w:.0f} min"])
    btbl = K.table(["Round-one body", "Mass", "Walking power with the Pi 5", "Pack", "Pack mass (share)", "Runtime"],
                   brow, num={1, 2, 4, 5}) if brow else ""
    return f"""
<section id="materials">
  <p class="kicker">Overnight · Materials and power</p><h2>Carbon fibre, coverings, pads, and battery or tether</h2>
  <div class="prose">
    <p><b>Lighter materials do not move the limit that binds these bodies.</b> The governing limit on both is the
    knee's speed in the animal-like walk, set by the walk's geometry and timing, not by mass. Lighter bodies lower
    torques, power and the head and tail loads: torque headroom and runtime, not speed headroom.</p>
    <ul>
      <li><b>Carbon-fibre frame plates: not for prototype 1.</b> With the sourced 3 mm quasi-isotropic plate they
      save about 0.4 kg for $1.6–2.1k, but that plate has only two-thirds of today's aluminium's rigidity; at equal
      rigidity the saving roughly halves, and thinner aluminium plate (2.54 mm) gets {thin} for nothing. Carbon
      cannot hold bearing fits or threads and hides fall damage.</li>
      <li><b>Coverings: the best value.</b> Walls about a quarter thinner in MJF PA11 (ductile, dents instead of
      cracking) save 0.3–0.5 kg at about today's cost, mostly off the neck and tail.</li>
      <li><b>Foot pads:</b> a cast polyether polyurethane tread (50–60 Shore A, small round studs) is expected to grip
      at 0.70–0.85 on carpet and 0.8–1.0 on wood and tile, but vinyl and laminate can be 0.42–0.50. Counting the
      stance foot's spin, the 60 cm robot needs about 0.55 just to walk and the 75 cm robot about 0.40, so the 60 cm
      robot may not walk on vinyl or laminate at all. Measure the pad on your floors.</li>
      <li><b>Tether-only: no.</b> The pack is also the counterweight at the tail base; without it the centre of mass
      moves forward and the knee goes past its speed limit with the same drives. Keep the battery and add an external
      DC input, as your power requirement already asks, with the cable hung from a boom (never trailed).</li>
      <li><b>3–4 ft robot: battery, yes.</b> About an hour of walking for 2–4 % of the robot's mass.</li>
    </ul>
  </div>
  {tbl}
  <h3>Battery on the 3–4 ft robot</h3>
  {btbl}
  <p class="note">Full note with sources: <code>round2/refine/MATERIALS.md</code>. Round one's 100 and 120 cm records
  with the Pi 5's compute budget; every figure is a model estimate.</p>
</section>"""


def head_tail(x: Ctx) -> str:
    keys = x.pick_keys()
    rows = []
    for key in keys:
        d = x.record(key)
        if not d:
            continue
        n, t = d["margins"]["neck"], d["margins"]["tail"]
        rows.append([mono(key), K.esc(part(n["actuator"])), f"{max(n['cont_ratio'], n['peak_ratio'], n['speed_ratio']):.2f}",
                     K.esc(part(t["actuator"])), f"{max(t['cont_ratio'], t['peak_ratio'], t['speed_ratio']):.2f}"])
    tbl = K.table(["Pick", "Neck servo", "Highest ratio", "Tail servo", "Highest ratio"], rows, num={2, 4}) if rows else ""
    return f"""
<section id="headtail">
  <p class="kicker">Overnight · Head and tail</p><h2>How the head and tail move</h2>
  <div class="prose">
    <ul>
      <li><b>Head:</b> one nodding joint at the neck base, ±0.35 rad (±20°): the S-curved neck and skull pitch together.
      The head does not turn side to side, the jaw is not powered yet (deferred) and the arms are fixed.</li>
      <li><b>Tail:</b> one side-to-side joint at the tail base, ±0.45 rad (±26°). The tail is one rigid spar; the Hybrid's
      segments are covers.</li>
      <li><b>In the reference walk:</b> the head nods ±0.05 rad (±3°) at the step frequency and the tail sways ±0.12 rad
      (±7°) at the stride frequency, inside the behavior spec's ±0.05–0.15 and ±0.1–0.2 rad.</li>
      <li><b>Following limits the head:</b> the camera is in the head, and a person stays in view only with the head
      near level.</li>
    </ul>
  </div>
  {tbl}
</section>"""


SECTIONS = [header, overnight, choices, bodies, sizes, behaviors, variants, whatif, refinement, size_study_sec, materials,
            head_tail, inside, scorecard, corrections, questions, nextsteps, footer]


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
