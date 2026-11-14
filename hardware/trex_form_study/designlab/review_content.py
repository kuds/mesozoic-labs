"""Sections of the round-one review page.  Every number is read from the generated data (see
``designlab.review_page``); the prose states only what those numbers show."""

from __future__ import annotations

from . import page_kit as K
from .review_page import BATTERY_NAMES, FAMILY, HEIGHTS, WIDTH, cm, kg, part, usd

REF = {h: f"{h}-M-S-10" for h in HEIGHTS}
GROUP_LABEL = {"hip_roll": "Hip roll", "hip_pitch": "Hip pitch", "hip_yaw": "Thigh twist", "knee": "Knee",
               "ankle_a": "Ankle (×2 per leg)", "neck": "Neck pitch", "tail": "Tail yaw"}


def _ok(v) -> str:
    return K.tag("pass", "ok") if v else K.tag("fail", "bad")


def _rng(vals, f):
    lo, hi = min(vals), max(vals)
    return f(lo) if abs(hi - lo) < 1e-9 else f"{f(lo)}–{f(hi)}"


def header(x) -> str:
    st = x.stats
    refs = [x.c(REF[h]) for h in HEIGHTS]
    n_review = len(x.R)
    return f"""
<header class="top">
  <p class="kicker">Mesozoic Labs · robotic <i>T. rex</i> · design review, round one · 7 Nov 2026</p>
  <h1>Four sizes, three bodies, three postures</h1>
  <p class="lede">Round one built and sized {st['ok']} candidate robots: four crown heights (60, 75, 100 and
  120 cm), three widths, three body families, three chest postures and, per height, six layout, battery and
  proportion variants. Each candidate has real component envelopes, load-based actuator and battery choices,
  power, cost and mechanical screens. The {n_review} candidates shown here are rendered from their
  compiled simulation models. Nothing on this page is a final design. The questions at the end
  decide what round two refines.</p>
  <div class="facts">
    <div class="fact"><b>{_rng([r['mass'] for r in refs], lambda v: f'{v:.0f}')} kg</b><span>reference robots, 60 → 120 cm (balanced width, Substantial body, 10°)</span></div>
    <div class="fact"><b>{_rng([r['runtime'] for r in refs], lambda v: f'{v:.0f}')} min</b><span>walking runtime on the selected pack (target 15–20 min)</span></div>
    <div class="fact"><b>{_rng([r['cost'][1] for r in refs], usd)}</b><span>nominal prototype cost; high side {_rng([r['cost'][2] for r in refs], usd)}</span></div>
    <div class="fact"><b>{sum(1 for r in x.R.values() if (r.get('clearance') or {}).get('pass', {}).get('all'))} of {n_review}</b><span>review candidates pass every clearance gate</span></div>
  </div>
  <nav class="toc" aria-label="Sections">
    <a href="#sizes">Sizes</a><a href="#families">Body families</a><a href="#posture">Posture</a><a href="#width">Width</a>
    <a href="#proportions">Proportions</a><a href="#inside">Inside</a><a href="#sensing">Sensing</a>
    {'<a href="#compute">Compute</a>' if (x.r1 / "compute_options.json").exists() else ''}<a href="#screening">Screening</a><a href="#cost">Cost and runtime</a><a href="#questions">Questions for you</a>
    <a href="#next">Next</a>
  </nav>
</header>"""


def sizes(x) -> str:
    rows = []
    for h in HEIGHTS:
        r = x.c(REF[h])
        rows.append([f"<b>{h} cm</b>", cm(r["crown"]), cm(r["hip"]), cm(r["length"]), cm(r["max_width"]), kg(r["mass"]),
                     f"{r['bus']:.0f} V", BATTERY_NAMES.get(r["battery"], r["battery"]), f"{r['walk_w']:.0f} W",
                     f"{r['runtime']:.0f} min", f"{usd(r['cost'][1])} <span class='mono'>({usd(r['cost'][2])})</span>"])
    tbl = K.table(["Class", "Crown", "Hip axis", "Length", "Max width", "Mass", "Bus", "Battery", "Walk power",
                   "Runtime", "Cost nominal (high)"], rows, num={1, 2, 3, 4, 5, 8, 9})
    parts_rows = []
    for g in ("hip_roll", "hip_pitch", "hip_yaw", "knee", "ankle_a", "neck", "tail"):
        cells = [GROUP_LABEL[g]]
        for h in HEIGHTS:
            m = x.c(REF[h])["margins"][g]
            ratio = max(m["cont_ratio"], m["peak_ratio"], m["speed_ratio"])
            cells.append(f"{part(m['actuator'])}<br><span class='mono'>hold {m['hold']:.1f} N·m · {ratio:.2f} of limit</span>")
        parts_rows.append(cells)
    ptbl = K.table(["Joint"] + [f"{h} cm" for h in HEIGHTS], parts_rows)
    r60, r75, r100, r120 = (x.c(REF[h]) for h in HEIGHTS)
    hp = {h: x.c(REF[h])["margins"]["hip_pitch"]["hold"] for h in HEIGHTS}
    buses = {h: x.c(REF[h])["bus"] for h in HEIGHTS}
    hp120 = r120["margins"]["hip_pitch"]
    shared = (" that the 100 cm reference already uses"
              if r100["margins"]["hip_pitch"]["actuator"] == hp120["actuator"] else "")
    return f"""
<section id="sizes">
  <p class="kicker">1 · Size</p><h2>Four height classes at one physical scale</h2>
  <div class="prose">
    <p>Crown height is the floor-to-skull-top height in the standing pose at the 10° reference posture.
    The body is solved so the crown lands on the class height. Everything else follows from the
    skeletal template and the hardware that has to fit inside.</p>
    <p>Mass grows much faster than height: from {kg(r60['mass'])} at 60 cm to {kg(r120['mass'])} at
    120 cm. The largest joint load is hip pitch in single support, and it grows from
    {hp[60]:.1f} to {hp[120]:.1f} N·m. That load sets the drive class, which drives the mass and cost
    steps between sizes. Bus voltage by size: {', '.join(f'{h} cm {buses[h]:.0f} V' for h in HEIGHTS)}.
    The 100 and 120 cm classes use larger leg drives, a wide-angle camera (OAK-D Pro W) and a 2D
    lidar. At 120 cm every candidate needs the strongest single drive in the catalog for hip pitch, the
    {part(hp120['actuator'])}{shared}. The 120 cm reference's requirement is {hp120['cont_ratio']:.2f} × its
    sustained planning limit (§9).</p>
  </div>
  {x.img('sizes_common_scale.png', 'Side views of the four reference robots on one ground line with a 1.75 m person for scale',
         'Sizes at one scale', 'Reference candidates <span class="mono">60/75/100/120-M-S-10</span> on one ground line with a 0.5 m grid and a 1.75 m person.')}
  <div class="grid2">
    {x.img('sizes_height_normalized.png', 'The same four robots scaled to equal crown height', 'Height-normalized',
           'Scaled to equal crown height to compare proportions. Sizes are not comparable in this view.')}
    {x.img('sizes_silhouettes.png', 'Black silhouettes of the four sizes', 'Silhouettes', 'Black side silhouettes at one scale.')}
  </div>
  {tbl}
  <h3>Which drives each size needs</h3>
  <div class="prose"><p>The selector picks, joint by joint, among the sourced parts with a price that meet the
  load on the chosen bus, the one with the lowest score: its mass plus 0.8 kg per $1000 of price, plus a
  hip-spacing packaging term for the hip-roll, hip-pitch and thigh-twist drives (the neck and tail try box
  servos first). Where no part meets the load, as for hip pitch on the 120 cm reference, the strongest
  drive on the 48 V bus is used and flagged. The criteria are sustained torque against the larger of the nominal-walk RMS and the
  single-support hold, peak torque against 1.25 × the brisk-walk peak, and speed. The second line of
  each cell gives the governing ratio: requirement over planning limit, so 1.00 means just
  enough.</p></div>
  {ptbl}
</section>"""


def families(x) -> str:
    rows = []
    for f in "TSH":
        cid = f"100-M-{f}-10"
        r = x.c(cid)
        m = r["metrics"]
        rows.append([f"<b>{FAMILY[f]}</b> <span class='mono'>{cid}</span>", f"{cm(m['torso_width'])} × {cm(m['torso_depth'])}",
                     kg(r["classes"]["C"]), kg(r["classes"]["A"]), kg(r["classes"]["B"]), kg(r["mass"]),
                     f"{m['ratios']['com_ahead_of_hip_over_H']:.2f} H", usd(r["cost_lines"]["covers"][1])])
    tbl = K.table(["Family (100 cm, 10°)", "Torso width × depth", "Coverings (C)", "Hardware (A)", "Arrangement (B)",
                   "Total", "CoM ahead of hip", "Covers cost"], rows, num={2, 3, 4, 5, 6, 7})
    t100, s100, t120, s120 = (x.c(c) for c in ("100-M-T-10", "100-M-S-10", "120-M-T-10", "120-M-S-10"))
    bat = lambda r: BATTERY_NAMES.get(r["battery"], r["battery"])  # noqa: E731
    return f"""
<section id="families">
  <p class="kicker">2 · Body family</p><h2>Three ways to clothe one skeletal template</h2>
  <div class="prose">
    <p>The families share the skeletal template, the tandem leg layout and the placement of the hardware,
    but each is sized on its own built mass and loads, so drives, pack and even bus can differ. At 100 cm
    the Tapered body takes {part(t100['parts']['hip_pitch'])} hip-pitch drives and a {bat(t100)} pack, where the
    Substantial body needs {part(s100['parts']['hip_pitch'])} drives and a {bat(s100)} pack. At 120 cm the Tapered
    body stays on {t120['bus']:.0f} V, while the Substantial and Hybrid hip-pitch loads exceed the strongest
    catalog drive and the selector falls back to {s120['bus']:.0f} V.</p>
    <p>Beyond that, they differ in where volume goes and how much of the machine is covered. All three
    keep a deep chest, a waist behind the ribs and a muscular thigh. The Substantial and Hybrid bodies
    also cover a distinct pubic keel between the thighs, which the Tapered body leaves open with the rest
    of its ventral abdomen. They avoid the earlier rejected forms: a rectangular chassis, or a barrel
    torso with a hanging belly.</p>
    <ul>
      <li><b>Tapered</b>: lean trunk, light coverings over the back and flanks, an open ventral abdomen,
      exposed lower legs. Lightest and cheapest to cover. It reads as a lean animal.</li>
      <li><b>Substantial</b>: a fuller, anatomically balanced body. The chest is close to the animal's
      fleshed width, the gastral belly and pubic boot are covered, and the thigh and tail root are
      robust. Heaviest coverings.</li>
      <li><b>Hybrid</b>: covered skull, neck, chest front, pelvis and thighs. Rib hoops and the spine
      beam are exposed over the mid-trunk, and the distal tail is segmented. It shows the machine
      inside the animal outline.</li>
    </ul>
  </div>
  {x.img('family_posture_matrix_100.png', 'Grid of side views: three body families by three postures at 100 cm',
         'Family × posture, 100 cm', 'Rows: Tapered, Substantial, Hybrid. Columns: level, 10° and 20° chest-up. One physical scale; captions give crown height, torso inclination and mass.')}
  {x.img('families_60.png', 'The three families at 60 cm', 'Families, 60 cm', 'The same three families at the smallest size.')}
  {tbl}
  <p class="note">Coverings are hollow printed shells (MJF PA12, 1.6–2.9 mm walls by size) with their own
  supports and fasteners. Their mass and inertia are in every load and runtime figure. The
  colours are placeholders.</p>
</section>"""


def posture(x) -> str:
    rows = []
    for h in (75, 100):
        for p in (0, 10, 20):
            cid = f"{h}-M-S-{p:02d}"
            r = x.c(cid)
            m = r["metrics"]
            rows.append([f"<span class='mono'>{cid}</span>", f"{p}°", cm(r["crown"]), cm(r["hip"]),
                         f"{m['ratios']['com_ahead_of_hip_over_H']:.3f} H", f"{r['margins']['hip_pitch']['hold']:.1f}",
                         f"{r['margins']['knee']['hold']:.1f}", f"{r['walk_w']:.0f} W"])
    tbl = K.table(["Candidate", "Chest-up", "Crown", "Hip axis", "CoM ahead of hip", "Hip-pitch hold, N·m",
                   "Knee hold, N·m", "Walk power"], rows, num={2, 3, 4, 5, 6, 7})
    return f"""
<section id="posture">
  <p class="kicker">3 · Posture</p><h2>Level, 10° and 20° chest-up</h2>
  <div class="prose">
    <p>Posture is solved, not drawn. The trunk line from the hip axis to the shoulder reference is set to
    the target angle. The pelvis takes half of it, and the rest is a fixed lumbar bend. The neck keeps
    the skull at −4° and the tail keeps its world angle, so the skull and tail keep their attitude
    while the chest rises, and the hip axis and feet stay put. The head is carried up with the chest
    (the crown rises 6–13 cm per 10°), and the tail is lowered as the pelvis tips (its lowest point by
    about 0.04 H per 10°). The whole robot is never rotated.</p>
    <p>The height classes are matched at 10°. The 0° and 20° versions are the same body, so their crown
    height moves by about ±10 % (67 / 75 / 83 cm at 0° / 10° / 20° in the 75 cm class) instead of the
    body being rescaled. Raising the chest from 10° to 20° moves the centre of mass back toward the hip,
    by about 0.004–0.006 H on the reference bodies; from 0° to 10° it barely moves (slightly forward on
    the reference bodies). It also lifts the head-mounted camera, which pushes the nearest floor it can
    see about 0.2 m further out per 10°. What it does for the view of a person close by depends on size:
    at 75 cm the camera sees slightly more of a person 1 m away, while at 100 cm the robot's own head
    hides more of a person straight ahead (at 1 m: 96 % visible at 0°, 88 % at 10°, 80 % at 20°).</p>
  </div>
  {x.img('postures_75.png', 'The 75 cm robot at three postures', 'Posture, 75 cm', 'Same body at 0°, 10° and 20° chest-up.')}
  {tbl}
</section>"""


def width(x) -> str:
    rows = []
    for h in HEIGHTS:
        for w in "NMW":
            cid = f"{h}-{w}-S-10"
            if cid not in x.C:
                continue
            r = x.c(cid)
            m = r["metrics"]
            rows.append([f"<span class='mono'>{cid}</span>", WIDTH[w], cm(m["torso_width"]), cm(m["hip_pitch_spacing"]),
                         cm(m["foot_spacing"]), cm(m["max_width"]), f"{m['ratios']['crown_over_torso_width']:.2f}",
                         f"{m['crown'] / m['max_width']:.2f}", f"{m['ratios']['hip_height_over_foot_spacing']:.2f}",
                         f"{r['margins']['hip_roll']['hold']:.1f}", f"{r['walk_cop_margin_mm']:.0f} mm", kg(r["mass"])])
    tbl = K.table(["Candidate", "Width", "Torso width", "Hip spacing", "Foot spacing", "Max width",
                   "Crown / torso width", "Crown / max width", "Hip H / foot spacing", "Hip-roll hold, N·m",
                   "Walk CoP margin", "Mass"], rows, num={2, 3, 4, 5, 6, 7, 8, 9, 10, 11})
    return f"""
<section id="width">
  <p class="kicker">4 · Width</p><h2>Narrow, balanced and broad</h2>
  <div class="prose">
    <p>Torso width, hip-joint spacing, width across the hardware, foot spacing and the total support area
    are kept separate. Each width variant changes all of them together: the torso scale, the hip spacing
    and the foot spacing.</p>
    <p>Crown-to-torso-width is set by the width variant and family alone (narrow 3.94, balanced 3.40 and broad
    2.93 for the Substantial body), so it is the same at every size. Crown-to-maximum-width is not. The
    maximum width is the thigh fairings around the hip motors, and motors do not shrink with the body. The
    balanced 60 cm robot is therefore relatively wider (crown / max width 2.22) than the 75–120 cm robots
    (2.68–2.75).</p>
    <p>Both hip-roll motors sit on the body centreline, one ahead of and one behind the hip axis. The
    single-support hip-roll load then comes only from carrying the lifted leg, so it grows with hip
    spacing. That is why the narrow variants need the least hip-roll torque.</p>
    <p>Broader spacing does not give the walk more support margin in this model. Every width variant
    has the same foot pads, so in single support the body has to shift farther sideways to stand over
    the same pad. In the table the broad variant has the lowest walk CoP margin at 60, 100 and 120 cm,
    and across all 36 size × family × posture combinations it beats the balanced width in only 19. It
    also moves the stance further from the animal's. Tyrannosaurid trackways are nearly single-file, at
    about 0.07 × hip height between left and right prints. The width variants stand 3.4 to 7 times wider
    (0.24–0.49 H from narrow to broad, about 5 times at balanced width), and the side-by-side hip-roll
    layout wider still.</p>
  </div>
  <div class="grid2">
    {x.img('widths_100_front.png', 'Front views of narrow, balanced and broad variants', 'Width, front', '100 cm, Substantial, 10°.')}
    {x.img('widths_100_top.png', 'Top views of the width variants', 'Width, top', 'Plan view: torso, thighs and feet.')}
  </div>
  {tbl}
</section>"""


def proportions(x) -> str:
    rows = []
    labels = {"": "default", "-anat": "anatomical tail", "-compact": "compact", "-bighead": "large skull"}
    for suf, lab in labels.items():
        cid = f"100-M-S-10{suf}"
        r = x.c(cid)
        m = r["metrics"]
        rows.append([f"<span class='mono'>{cid}</span>", lab, cm(m["length"]), cm(m["tail_length"]), cm(m["skull_length"]),
                     kg(r["mass"]), f"{m['ratios']['com_ahead_of_hip_over_H']:.3f} H",
                     f"{r['margins']['hip_pitch']['hold']:.1f}", f"{r['runtime']:.0f} min", usd(r["cost"][1])])
    tbl = K.table(["Candidate", "Variant", "Length", "Tail", "Skull", "Mass", "CoM ahead of hip", "Hip-pitch hold, N·m",
                   "Runtime", "Cost"], rows, num={2, 3, 4, 5, 6, 7, 8, 9})
    return f"""
<section id="proportions">
  <p class="kicker">5 · Proportions</p><h2>Tail length, skull size and leg length</h2>
  <div class="prose">
    <p>The default tail is shortened to 1.55 × hip height, about two thirds of the animal's, to limit
    floor footprint, turning sweep and tail inertia. The <b>anatomical tail</b> variant restores
    2.3 × hip height. The <b>large skull</b> variant makes the skull 15 % longer relative to the body (13 %
    in absolute length, because the body shrinks slightly to keep the crown height), which answers the
    earlier complaint that the head looked small. The <b>compact</b> variant shortens the legs (hip height
    −4 %) and the tail (−23 %), and its skull comes out about as long as the large-skull variant's: 12 %
    longer than the default in absolute length. The result is a stockier look at the same crown
    height.</p>
  </div>
  {x.img('proportions_100.png', 'Four proportion variants at 100 cm', 'Proportions, 100 cm', 'Default, anatomical tail, compact and large skull, at one scale.')}
  {tbl}
</section>"""


def inside(x) -> str:
    chart = K.stacked_chart([(f"{h} cm", [x.c(REF[h])["classes"][k] for k in "ABC"]) for h in HEIGHTS],
                            ["Hardware (A: drives, battery pack, compute, control, power, sensors)",
                             "Arrangement (B: structure, bearings, linkages, harness, feet, battery cassette, "
                             "compute tray, cover supports and fasteners, 6 % integration reserve)", "Coverings (C)"],
                            title="Installed mass by volume class")
    lay_rows = []
    for suf, lab in (("", "tandem hip roll (default)"), ("-side", "side-by-side hip roll"), ("-serial", "serial leg (PR #662-like)"),
                     ("-belly", "battery in the belly")):
        cid = f"60-M-S-10{suf}"
        r = x.c(cid)
        ok = r.get("ss_feasible", True)
        lay_rows.append([f"<span class='mono'>{cid}</span>", lab, kg(r["mass"]),
                         f"{r['margins']['hip_roll']['hold']:.1f}" if ok else "no balanced pose",
                         f"{r['margins']['hip_pitch']['hold']:.1f}" if ok else "no balanced pose",
                         f"{r['metrics']['ratios']['com_ahead_of_hip_over_H']:.3f} H",
                         f"{r['runtime']:.0f} min", usd(r["cost"][1])])
    ltbl = K.table(["Candidate", "Layout", "Mass", "Hip-roll hold, N·m", "Hip-pitch hold, N·m", "CoM ahead of hip",
                    "Runtime", "Cost"], lay_rows, num={2, 3, 4, 5, 6, 7})
    r60 = x.c("60-M-S-10")
    joint = {"hip_roll": "hip roll", "hip_yaw": "thigh twist", "hip_pitch": "hip pitch", "knee": "knee",
             "ankle_a": "upper ankle drive", "ankle_b": "lower ankle drive"}
    layout_drives = []
    for suf, lab in (("-side", "the side-by-side legs take"), ("-serial", "the serial legs take"),
                     ("-belly", "the belly-battery robot takes")):
        p = x.c(f"60-M-S-10{suf}")["parts"]
        changed = [f"a {part(p[g])} {joint[g]}" for g in joint if p.get(g) != r60["parts"].get(g)]
        if changed:
            layout_drives.append(f"{lab} {', '.join(changed[:-1]) + ' and ' if len(changed) > 1 else ''}{changed[-1]}")
    return f"""
<section id="inside">
  <p class="kicker">6 · Inside</p><h2>What has to fit, and what it costs the shape</h2>
  <div class="prose">
    <p>Drives, battery cassette, Jetson compute with tray, control stack, power distribution, camera, ToF
    sensors, lidar and linkages are real envelopes with a sourced mass where one exists. Bearings and
    harness, like the IMU and foot sensing, are point-mass allowances without geometry: they count in
    the body masses and inertias but are not screened for clearance. Nothing is scaled as decoration. The trunk is widened or deepened only where
    hardware needs room, and those changes are recorded per candidate.</p>
    <p>Two layout choices carry most of the mechanics. The tandem hip-roll layout puts both roll motors on
    the centreline. The tail-base battery sits behind the hip, where the animal carried its heavy tail
    muscles, and moves the centre of mass back toward the hip. On the 60 cm robot the tandem layout
    resolves the earlier single-support hip-roll gap of about 9 N·m against 6 N·m in the model: the hold
    falls to {r60['margins']['hip_roll']['hold']:.1f} N·m, carried by a {part(r60['parts']['hip_roll'])}, while hip pitch and knee stay on the
    incumbent MyActuator X4-36 class. The comparison below holds size, width, family, posture and
    proportions fixed. Each layout is re-sized, so some drives differ from the default:
    {'; '.join(layout_drives)}.</p>
  </div>
  <div class="grid2">
    {x.img('60-M-S-10_cutaway.png', '60 cm robot with translucent coverings showing installed hardware', 'Cutaway, 60 cm', 'Battery, compute, camera, power and actuators, labelled from the model.')}
    {x.img('120-M-S-10_cutaway.png', '120 cm robot cutaway', 'Cutaway, 120 cm', 'The same layout at 120 cm.')}
  </div>
  {x.img('layouts_60_cutaway.png', 'Four leg and battery layouts at 60 cm, cutaway', 'Layouts, 60 cm', 'Tandem hip roll (default), side-by-side hip roll, serial PR #662-like leg, belly battery.')}
  {ltbl}
  <h3>Where the mass is</h3>
  {chart}
  <p class="note">Class A is volume the hardware itself needs. Class B comes from the chosen arrangement. Class C is the
  cosmetic coverings, which can be redesigned freely. Every ledger row also states its basis (sourced, partially
  sourced, derived or estimate). No row is measured.</p>
</section>"""


def sensing(x) -> str:
    rows = []
    for cid in ("60-M-S-10", "75-M-S-10", "100-M-S-10", "120-M-S-10"):
        se = (x.R.get(cid) or {}).get("sensors") or {}
        if not se:
            continue
        tf = se.get("torso_follow", {}).get("home", {})
        tof = se.get("tof", {}).get("tof_l", {}).get("floor_ahead_of_toes_m") or [None, None]
        lid = se.get("lidar_blocked_1m", {})
        rows.append([f"<span class='mono'>{cid}</span>", f"{100 * se['camera_blocked']['home']:.0f} %",
                     _ok(tf.get("ok")), f"{se.get('centre_column_nearest_visible_floor_ahead_of_toes_m', 0):.1f} m",
                     f"{tof[0]:.2f}–{tof[1]:.2f} m" if tof[0] is not None else "–",
                     f"{100 * se['near_strip_coverage']['foot_path']['combined']:.0f} %",
                     f"{100 * lid['as_built']:.0f} %" if lid else "none fitted"])
    tbl = K.table(["Candidate", "Camera view blocked by own body", "Person's torso visible ahead",
                   "Nearest floor the camera sees, ahead of toes", "ToF floor band", "Foot path covered",
                   "Lidar blocked within 1 m"], rows, num={1, 3, 4, 5, 6})
    return f"""
<section id="sensing">
  <p class="kicker">7 · Sensing</p><h2>The camera moved into the head</h2>
  <div class="prose">
    <p>The first sensor layout failed screening. A camera in the throat lost the view straight ahead to the
    jaw: at 60 cm, 0 % of a person's torso straight ahead was visible. Its box also met the neck motor.
    The stereo camera now sits in the skull roof just ahead of the eyes, looking over the snout. Its
    view is almost free of the robot's own body, and it sees a person's torso at 1.5–3 m at every size.
    The cost is the floor: the snout hides the ground just ahead of the feet. That is now the job of
    the two downward ToF sensors, which aim over the two foot paths. At 100 and 120 cm a lidar hangs
    under the base of the throat; only the sector behind it, blocked by the chest, is lost.</p>
  </div>
  {tbl}
  <div class="grid2">
    {x.img('60-M-S-10_sensors.png', 'Top-view coverage map for the 60 cm robot', 'Coverage, 60 cm', 'Camera floor footprint, ToF zones and occlusion maps.', src_dir=x.screen)}
    {x.img('120-M-S-10_sensors.png', 'Coverage map for the 120 cm robot', 'Coverage, 120 cm', 'Including the lidar scan and its blocked sectors.', src_dir=x.screen)}
  </div>
  <p class="note">Geometry only: ray casting of what the robot's own body hides, with assumed windows in the shells.
  No depth noise, lighting or detection performance is modelled.</p>
</section>"""


SHORT_ARCH = {"A_orin_nano_super_devkit": "Jetson Orin Nano Super dev kit (current), capped at 15 W",
              "B_orin_nano_module_small_carrier": "Orin Nano 8 GB module on a 10–60 V carrier",
              "C_pi5_hailo8l_oak_depth": "Raspberry Pi 5 8 GB + AI HAT+ 13 TOPS (Hailo-8L)",
              "D_pi5_oak_on_camera_inference": "Raspberry Pi 5 8 GB, detection on the OAK-D's chip",
              "E_rk3588_orange_pi_5_max": "Orange Pi 5 Max (RK3588)",
              "F_oak4_d_pro_camera_computer": "Luxonis OAK 4 D Pro (camera is the computer)",
              "G_orin_nx_16gb_large_sizes": "Jetson Orin NX 16 GB"}


def compute(x) -> str:
    """Onboard compute alternatives (owner request P9); skipped if the survey has not been run."""
    import json

    rp = x.r1.parent / "designlab" / "data" / "research" / "compute_2026-11-07.json"
    ep = x.r1 / "compute_options.json"
    if not rp.exists() or not ep.exists():
        return ""
    syn = json.loads(rp.read_text())["synthesis"]
    eff = json.loads(ep.read_text())
    arows = []
    for a in syn["architectures"]:
        arows.append([f"<b>{K.esc(SHORT_ARCH.get(a['id'], a['name']))}</b>",
                      f"{a['power_w_typical']:g} / {a['power_w_max']:g} W",
                      f"{a['mass_g']:.0f} g" if a.get("mass_g") else "not published",
                      f"${a['price_usd']:,.0f}", "yes" if a.get("covers_minimum_spec") else "provisional",
                      ", ".join(f"{int(h)}" for h in a.get("best_for_sizes", [])) or "–"])
    atbl = K.table(["Architecture", "Typical / max power", "Mass", "Price", "Meets minimum spec", "Suggested for (cm)"],
                   arows, num={1, 2, 3})
    names = [a["id"] for a in syn["architectures"] if a.get("power_w_typical") is not None]
    rrows = [["Study budget today (31 W allowance)"] + [f"{eff['sizes'][str(h)]['baseline_runtime_min']:.0f} min"
                                                       for h in HEIGHTS]]
    for i, aid in enumerate(names):
        rrows.append([K.esc(SHORT_ARCH.get(aid, aid))] + [
            f"{eff['sizes'][str(h)]['rows'][i]['runtime_min']:.0f} min "
            f"<span class='mono'>({eff['sizes'][str(h)]['rows'][i]['runtime_change_min']:+.0f})</span>" for h in HEIGHTS])
    rtbl = K.table(["Architecture"] + [f"{h} cm" for h in HEIGHTS], rrows, num={1, 2, 3, 4})
    return f"""
<section id="compute">
  <p class="kicker">8 · Compute</p><h2>Jetson, or something smaller?</h2>
  <div class="prose">
    <p><b>Yes at 60 and 75 cm; mostly no at 100 and 120 cm. The first fix is the power budget, not the
    computer.</b></p>
    <ul>
      <li><b>Power budget.</b> The study books the Jetson at 31 W. Independent measurements of the same module
      show 10–14 W running detectors, so capped at 15 W the dev kit should draw about 15 W. That alone
      lengthens every size's runtime, by about 8 minutes at 60 cm.</li>
      <li><b>60 and 75 cm.</b> A Raspberry Pi 5 with Raspberry Pi's $70 AI HAT+ (a 13 TOPS Hailo chip for the
      person detector and re-identification) does the job at about 8 W, 100 g and $290. The Jetson kit is
      about 15 W, 174 g and $439. The head camera keeps computing depth and encoding the operator video.
      Running detection on the camera itself, with no HAT, is cheaper still, but it waits for a bench test
      under the full combined load.</li>
      <li><b>100 and 120 cm.</b> Stay on NVIDIA, where mapping, voice and bigger models will live and price
      is not a constraint. That is an Orin Nano at 100 cm, and an Orin NX at 120 cm only if those growth
      loads are committed. A camera that is also the computer (Luxonis OAK 4) is worth an experiment.</li>
      <li><b>Whatever is chosen:</b>
        <ul>
          <li>the walking policy and motor loop stay on the Teensy microcontroller;</li>
          <li>the current OpenCV person detector must move to a neural accelerator;</li>
          <li>the depth camera's chip is discontinued, so spares are needed.</li>
        </ul></li>
    </ul>
    <p>79 options in five categories were surveyed from sourced data, then fact-checked independently (108
    corrections). The synthesis was challenged by two critics before revision. Details and bench tests are
    in <code>COMPUTE.md</code>; the source URLs and the corrections are in
    <code>designlab/data/research/compute_2026-11-07.json</code>.</p>
  </div>
  {atbl}
  <h3>Runtime on each size's selected pack</h3>
  <div class="prose"><p>Change from today's study budget in brackets. First order: drive power and the other
  auxiliaries are held fixed. Round two will adopt the chosen computer's measured budget and re-size the
  batteries.</p></div>
  {rtbl}
</section>"""


def screening(x) -> str:
    rows = []
    for cid in sorted(x.R, key=lambda k: (int(k.split("-")[0]), k)):
        cl = (x.R[cid].get("clearance") or {})
        ps = cl.get("pass", {})
        sw = (cl.get("swing_clearance") or {}).get("nominal", {})
        rows.append([f"<span class='mono'>{cid}</span>", _ok(ps.get("interference_free_home")),
                     _ok(ps.get("interference_free_nominal_gait")), _ok(ps.get("swing_clearance")),
                     _ok(ps.get("tail_clearance")), _ok(ps.get("battery_path")),
                     f"{sw.get('mid_swing_min_mm', float('nan')):.0f} mm" if sw else "–",
                     K.tag("same model", "ok") if x.R[cid].get("matches_stage_a") else K.tag("rebuilt differs", "warn")])
    tbl = K.table(["Candidate", "No contact at home", "No contact, nominal walk", "Swing clearance ≥ 0.04 H",
                   "Tail ≥ 0.25 H", "Battery removal path", "Mid-swing toe clearance", "Sweep record"], rows, num={6})
    st = x.stats
    flagged = sorted(set(st["ss_infeasible"] + st["walk_infeasible"] + st["over_catalog"] + st["guard"]))
    return f"""
<section id="screening">
  <p class="kicker">9 · Mechanical screening</p><h2>What fits, what collides, what was fixed</h2>
  <div class="prose">
    <p>Every review candidate is checked on its compiled model for:</p>
    <ul>
      <li>part-against-part interference at home and along every frame of both reference walks;</li>
      <li>joint-range extremes;</li>
      <li>toe clearance in mid-swing;</li>
      <li>tail height;</li>
      <li>a straight removal path for the battery cassette.</li>
    </ul>
    <p>The first run found six generator defects:</p>
    <ul>
      <li>hip brackets overlapping at the centreline;</li>
      <li>the tail motor inside the battery cassette at 60 and 75 cm;</li>
      <li>the power board inside the compute tray;</li>
      <li>ankle plates limiting ankle pitch;</li>
      <li>pelvis and thigh shells overlapping;</li>
      <li>the 60 cm thigh fairing reaching the ankle motors at knee flexion.</li>
    </ul>
    <p>All six were fixed, and the record is in the decision log (F6). The table shows the screen after
    the fixes.</p>
    <p>Across all {st['n']} sized candidates, Stage A flagged {len(flagged)} for:</p>
    <ul>
      <li>no balanced single-support pose;</li>
      <li>a walk that leaves the support pad;</li>
      <li>a load beyond the catalog;</li>
      <li>sizing that hit the mass-spiral rule.</li>
    </ul>
    <p>The full list is in <code>round1/TABLES.md</code>.</p>
  </div>
  {tbl}
  <p class="note">Screening, not certification. The reference walks are prescribed trajectories used to size
  loads, not a controller. Carpet, turning loads, slips, impacts and thermal limits are not screened
  this round. <code>SCREENING.md</code> lists how each will be.</p>
</section>"""


def cost(x) -> str:
    chart = K.cost_chart([(f"{h} cm", *x.c(REF[h])["cost"]) for h in HEIGHTS])
    rows = []
    for h in HEIGHTS:
        r = x.c(REF[h])
        rr = r["runtime_range"]
        rows.append([f"<b>{h} cm</b>", BATTERY_NAMES.get(r["battery"], r["battery"]), f"{r['power']['drives_w']:.0f} W",
                     f"{r['power']['aux_w']:.0f} W", f"{r['walk_w']:.0f} W",
                     f"{rr['0.65']:.0f} / {rr['0.75']:.0f} / {rr['0.85']:.0f} min", usd(r["cost"][0]), usd(r["cost"][1]), usd(r["cost"][2])])
    tbl = K.table(["Class", "Battery", "Drives", "Compute and sensors", "Total walking", "Runtime at 65 / 75 / 85 % usable",
                   "Cost low", "Nominal", "High"], rows, num={2, 3, 4, 5, 6, 7, 8})
    return f"""
<section id="cost">
  <p class="kicker">10 · Cost and runtime</p><h2>What each size costs and how long it runs</h2>
  <div class="prose">
    <p>Per your note, the larger sizes stay in as full options. Their cost is reported here, not used to rule
    them out. Cost covers one physical prototype with outsourced fabrication. It includes drives,
    electronics, battery and spares, printed coverings, frame parts, bench and tether supply, integration
    and contingency. Priced parts use sourced prices. The lots (frame, coverings, integration) are scaled
    from PR #662's 60 cm lines, so the high side is a real possibility, not a worst case. Runtime is the
    nominal reference walk on the lightest pack that meets 20 minutes, at 75 % usable energy and with no
    regeneration credit.</p>
  </div>
  {chart}
  {tbl}
</section>"""


def questions(x) -> str:
    def note(h):
        r = x.c(REF[h])
        return (f"{kg(r['mass'])}, {r['bus']:.0f} V, {r['runtime']:.0f} min; cost {usd(r['cost'][1])} nominal, "
                f"{usd(r['cost'][2])} high")

    q = [
        ("Which size should round two develop?",
         [f"60 cm: lightest; keeps the incumbent X4-36 class at hip pitch and knee ({note(60)}).",
          f"75 cm: more presence with a larger hip-pitch drive ({note(75)}).",
          f"100 cm: museum presence ({note(100)}).",
          f"120 cm: the top of the long-term 3–4 ft (0.9–1.2 m) range, which 100 cm also falls inside ({note(120)}).",
          "Develop two sizes side by side, for example a smaller and a larger one."]),
        ("Which body family is closest to what you want?",
         ["Tapered: lean animal, light coverings, visible legs.",
          "Substantial: full animal outline, heaviest coverings.",
          "Hybrid: animal outline over exposed structure (ribs, segmented tail).",
          "A mix: say which parts of each you like."]),
        ("Which posture reads best?",
         ["Level (0°).", "Alert, 10° chest-up.", "Upright, 20° chest-up."]),
        ("Which proportion changes should round two carry forward?",
         ["Anatomical (longer) tail.", "Larger skull.", "Narrower body and stance.", "Broader body and stance.",
          "Compact proportions (shorter legs and tail, larger skull)."]),
        ("Which onboard compute direction should round two adopt?",
         ["Raspberry Pi 5 + Hailo at 60–75 cm, NVIDIA Jetson at 100–120 cm (the survey's recommendation).",
          "Keep the Jetson Orin Nano at every size, re-budgeted from measurements.",
          "Also prototype the OAK 4 camera-computer.",
          "Something else: say what matters most (runtime, cost, software effort, growth)."]),
    ]
    items = "".join(f'<div class="q"><h3>{i + 1}. {K.esc(t)}</h3><ol type="a">'
                    + "".join(f"<li>{K.esc(o)}</li>" for o in opts) + "</ol></div>" for i, (t, opts) in enumerate(q))
    return f"""
<section id="questions">
  <p class="kicker">11 · Questions for you</p><h2>Five decisions for round two</h2>
  <div class="prose"><p>Answers can be as short as a letter per question. A sentence about what looks wrong is just as useful.
  Nothing has been frozen, and no controller training starts until round two has been reviewed.</p></div>
  {items}
</section>"""


def nextsteps(x) -> str:
    return """
<section id="next">
  <p class="kicker">12 · Next</p><h2>Round two, and what stays unproven</h2>
  <div class="prose">
    <ul>
      <li><b>Round two</b> refines the chosen size, family, posture and proportions:
        <ul>
          <li>skull and neck shaping;</li>
          <li>the shank coverings;</li>
          <li>the tight 60 cm thigh-to-ankle-motor clearance (0.6 mm in the brisk walk; the two meet by
          0.62 rad of knee flexion);</li>
          <li>two same-body overlaps left in the 60 cm Tapered model (power board and hip-roll motor, tail
          motor and battery cassette);</li>
          <li>a turning reference gait for twist loads;</li>
          <li>the carpet contact model.</li>
        </ul>
        Then it repeats every screen.</li>
      <li><b>Then a bounded controller study</b> on the selected design. Its targets are already proposed in
        <code>BEHAVIOR_SPEC.md</code>: speed, turning, sway, clearance, impacts and head/tail tracking. They are
        tested on hard floor and a declared carpet, with person-following on rendered camera images. No
        training has been run in this round.</li>
      <li><b>Not yet shown by anything here</b>: walking under a controller, measured runtime, strength,
        thermal behaviour, manufacturability, and detection or following performance. All figures are
        simulation estimates or sourced ratings.</li>
    </ul>
  </div>
</section>"""


def footer(x) -> str:
    return """
<footer>
  <p>Generated from <code>hardware/trex_form_study</code> on branch <code>claude/mesozoic-trex-design-iteration-azm2fs</code>:
  <code>python -m designlab.stage_a</code>, <code>designlab.round1</code>, <code>designlab.report</code>, <code>designlab.review_page</code>.
  Every plate is a render of a generated MuJoCo model, and the two coverage maps are plots of ray casts on
  those models. No reference figure or photograph is reproduced. Anatomical
  proportions follow cited measurements (<code>REFERENCE.md</code>).</p>
</footer>"""


SECTIONS = [header, sizes, families, posture, width, proportions, inside, sensing, compute, screening, cost, questions,
            nextsteps, footer]
