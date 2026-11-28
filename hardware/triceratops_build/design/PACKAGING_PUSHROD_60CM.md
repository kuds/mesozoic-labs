# Packaging of all four legs, and the foreleg pushrod: desk checks on the adopted 60 cm model

27 Nov 2026 (D45: the readiness review's no-spend checks). **Desk work only.** The packaging figures are geometry: signed
distances between the shapes of the simulation model `sim/generated/trike60_C_closed_capsule_fore128.xml` (D25, D41;
12.751 kg). The pushrod figures are calculations on that model's geometry and on the drives' catalog torques
(`sim/data/drives.json`). The walking torques used for fatigue are **simulation** (the four 128 mm policies' gait
reports). Material and rod-end figures are **web-search summaries** read 27 Nov 2026 (pages not opened; the vendor
hosts were outside this session's network), typical values, not design allowables. **Nothing is measured**: nothing
is built. No pre-registration covers this work, so nothing here is a judged verdict; where a record already has a
rule (the bench plan's 40 deg transmission angle, the kickoff list's 2x rating), the figure is shown beside it.

Revised 28 Nov 2026 after the package's third review: the legs against each other are now searched over all eight
joints of each pair of legs (9 values per joint, then a local search between grid points), not over each joint's
range ends and 0, which had stepped over the poses where the legs meet (items 9 and 10, section 1.2); and the hind
rod's static case takes the X4's stall torque (section 2.2).

Tools: `sim/check_packaging60.py` (packaging, with single and combined trial fixes, the legs against each other with
candidate joint limits, and the room around the bay's tightest pairs; full output
`sim/results/packaging60_2026-11-27/packaging60_report.txt` and `.json`),
`design/pushrod60.py` (pushrod and crank pins; output `design/pushrod60_out.json`). Tests: `sim/tests/
test_check_packaging60.py`, `design/tests/test_parts60_pushrod60.py`.

## The answer

**1. Packaging.** The adopted model cannot be built as drawn, for reasons that are all chassis, linkage and joint-range
layout, not the drives' size. Eleven kinds of clash between solid parts, and one gap where two parts must meet; all
four legs checked over each leg's full roll x pitch x flexion range against the chassis, head and tail (9 values per
joint), and every pair of legs against each other over all eight of their joints (9 values per joint, then a local
search between grid points):

| # | Clash | Overlap, mm (neutral / worst in range) | Records before today | What would fix it |
|---|---|---|---|---|
| 1 | Hind X4 knee housing on its own upper strut, both hind legs | **38.70 / 38.70** (same body, every pose) | estimated "up to about 38.7" (readiness review, `fore128.xml:250-251`); now confirmed | A forked hind link (two side plates bolted to the housing's faces), or the drive moved at least 38.7 mm plus the wanted gap off the strut's centreline, as the 128 mm front leg does (its X6 axis sits 59.9 mm off its strut, my arithmetic from `fore128.xml:186-187`; an X4 there would clear by 21.2 mm). Either changes the model (rod length, masses) |
| 2 | Pelvic crossmember through both hips | hind roll drive **41.10**, pitch drive 40.10, yoke 24.80, upper strut 20.80 / the same | F21 (`DECISION_LOG.md`, F21), reproduced | No straight bar at x -260 mm: between today's rails, raised 45 mm, it clears the drives but meets the tail yaw servo by 13.6 mm. **On the shortened, inset rails' ends (x -212 mm) with the I/O carrier 20 mm forward (combined layout C, below) it clears every solid part, by 3.40 mm or more** (the packs below it, the moved I/O carrier ahead); the tail then needs a bracket to its pivot, 34.6 mm aft |
| 3 | Shoulder crossmember into the front roll and pitch drives | roll drives **1.10 / 1.10**; pitch drive 1.10 / **28.96** (roll +24 deg); yoke and strut - / 10.83 | F19 restated (`DECISION_LOG.md` (F19): 29.0, 10.8), reproduced | **With today's rails** (ending at x +-260 mm): end it at the rails and raise it 6-10 mm; it then sits on the rails' ends (1.2 mm overlap at 10 mm up) and clears every part by at least 0.61 mm (6 mm up) or 3.01 mm (10 mm up). It does not combine with fixes 4 and 6, which end the rails 45 mm short of it: with those, use layout C (it moves onto the rails' ends at x 212 mm, the DC-DC block 20 mm aft and 3 mm up; the two then clear by 3.40 mm) |
| 4 | Side rails into the roll drives | front **8.19**, hind **23.10** / the same | none (the 18 Nov layout check skips frame-to-drive pairs, `sim/check_layout60.py:26`) | End the rails short of the roll drives and carry each drive on a bracket at its inboard face, as PR #697's drop bracket does at the front (and a new one at each hind drive, which the rails no longer pass through): rails ending at x +-215 mm clear the drives by 4.9-5.0 mm. This leaves both crossmembers (x +-260 mm) 45 mm past the rails' ends: they move onto the rails' ends (layouts B, C) or onto brackets from the drives (layout A) |
| 5 | Side rails into the moving hind yokes | **8.80 / 11.61** (roll -18 deg) | none | Fixed by the same shortening (no hind yoke within 10 mm) |
| 6 | Hind cranks and pushrod against the rails | 0.45 mm clear at roll 0; overlap from roll -2 deg inward, **2.91** (cranks) and 2.41 (rod) at roll -18, pitch -38 deg | none | Not fixed by shortening the rails. Rails 8 mm further in (y +-75.2 mm) and shortened: 4.70 mm clear at the worst pose (the crossmembers as in fix 4). Or the hind linkage moved to the outboard side, as at the front (not tried) |
| 7 | Tail pitch servo into the pelvic crossmember | **3.90 / 11.65** (tail yaw -24 deg) | 3.9 mm (21 Nov 2026 review, R3) | Move the servo 16 mm aft (x -40 mm in the tail frame): 2.96 mm clear at the worst pose; 7 mm is not enough (5.26 mm overlap). Or move the crossmember 48 mm forward, as in layout C: no tail part then within 10 mm of it |
| 8 | Tail pitch servo into the tail yaw servo | - / **5.68** (yaw -24, pitch +15 deg) | none | The same 16 mm move: 9.78 mm clear |
| 9 | Front and hind legs on one side meet under the body: feet, ankle hubs and lower struts, both sides | - / soles **48.0** (the whole capsule; fore roll -2.4, pitch +20.8, elbow -47.0, ankle -21.7; hind roll -2.0, pitch -26.6, knee -45.9, ankle +21.4 deg); lower struts, and each ankle hub with the other leg's strut or hub, **19.2** (fully crossed; struts at fore -18.0 / +37.3 / -46.2, hind -13.0 / -37.9 / -45.0 deg); a lower strut or ankle hub into the other leg's sole 33.6 | none (this report's first version: struts 19.20 at range corners only, from 3 values per joint) | Not a corner but a region of the range: on the 9-value grid solid parts overlap at 0.42 % of the pair's poses (0.11 % with the soles left out), only where the fore leg is pitched +9.5 deg or more (foot aft) and the hind leg -19 deg or less (foot forward), the fore elbow at +4.9 deg or less (flexed, or at most slightly extended) and the hind knee at -5.5 deg or less (flexed); roll and ankle values across their whole ranges take part. These are each joint's bounds on the grid, not a region where the legs always meet: of the grid's poses inside them 7.7 % overlap, and only where the two legs' roll angles are within 21 deg of each other (10.5 deg with the soles left out). The real feet, half-capsule rockers (F17), overlap too (18.0 cm³ at the deepest pose). Candidate stops (each swept like the full ranges): a fore pitch aft stop at +5 deg (0.53 mm clear) or 0 deg (26.4 mm); a hind pitch forward stop at -10 deg (19.3 mm; -15 deg still overlaps by 11.4 mm); the same stop on both at 15 deg (1.46 mm) or 10 deg (44.2 mm); a flexion stop on both at -10 deg (36.9 mm; at -20 deg the soles still overlap by 24.0 mm). Each narrows a range the gaits use; or a coupled software limit on the combined pose, drawn on the overlapping poses themselves (`legs_to_each_other` in the check computes them; the JSON keeps their bounds, share and leasts), not on the joints' bounds. Whether a gait reaches the region is not known (no gait report records joint angles) |
| 10 | The two hind feet meet at the inward roll stop | - / soles **8.49** (both hind legs at roll -18, pitch +22.4, knee -47, ankle -22 deg) | none (this report's first version: 3.64 mm clear, from 3 values per joint) | Only next to the inward stop: on the grid only at roll -18 on both legs (pitch +9.5 to +28.5, knee -47 to -36.6 deg). The real rockers overlap too (2.05 cm³ at that pose). A hind inward roll stop at -17 deg clears by 4.79 mm, at -16 deg by 18.1 mm, at -14 deg by 44.8 mm. Nothing else of the two hind legs comes within 6.5 mm (an ankle hub to the other sole, 6.55 mm) |
| 11 | Neck cradle's end, at the head pivot, into the neck yaw servo | **0.44 / 0.44** (every neck pitch: the pivot is the cap's centre) | none | End the cradle in a clevis around the pivot, or move the servo 0.5 mm plus the wanted gap |
| 12 | Elbow and knee axles stop short of their output cranks, all four legs | a **0.75 mm** gap, not an overlap: the crank sits on nothing | 0.75 mm (21 Nov 2026 review, R3) | Lengthen each axle through the crank's 10 mm hub, to the crank's outer face (51.75 mm from the link's mid-plane at the front, 46.75 mm at the hind), and key or clamp it |

The two front legs stay 26.3 mm apart (their soles, both at roll -18, pitch -27.6, elbow -47, ankle +22 deg), and a
front leg never comes within 50 mm of the hind leg on the other side. All figures in items 9 and 10 are the model's
shapes (desk geometry); the soles are classed "foot shape" because the simulated capsule is a full shape, and the
rocker figures check its lower half, the shape F17 gives the real part.

**The fixes interact, and fixes 3, 4 and 6 cannot all be applied as listed.** Each trial moved one part against
today's layout (`TRIALS`, `sim/check_packaging60.py`). Fix 3 ends the shoulder crossmember on today's rails; fixes 4
and 6 end the rails at x +-215 mm, 45 mm short of both crossmembers, and take the rails out of the hind roll drives.
Three combined layouts, each on the shortened, inset rails (y +-75.2 mm) with every roll drive on a bracket from the
rail's end to its inboard face, each moved part swept over all four legs, the tail and the head and measured against
every chassis part, the other moved parts included (section 1.5). The clearances listed are the moved parts'; the
parts a layout leaves in place keep today's gaps (sections 1.2, 1.3 and the bay table below):

| Layout | Crossmembers | Solid parts that still overlap | Least clearance of the moved parts otherwise, mm | What the frame members miss |
|---|---|---|---|---|
| A | kept at x +-260 mm between the rails' lines; shoulder 10 mm up, pelvic 45 mm up | pelvic crossmember into the tail yaw servo, **13.60** | 0.60 (rail to the reserve block), 3.91 (pelvic crossmember above the hind roll drives) | both crossmembers end in the air: 26.9 mm (shoulder) and 49.4 mm (pelvic) from the rails, 8.9 and 3.9 mm above the roll drives (a saddle from each drive's mount would carry them), the tail's pivot 31.4 mm below the pelvic one |
| B | on the rails' ends, x +-212 mm (a ladder frame); shoulder at z 0, pelvic 4 mm up | shoulder into the DC-DC block, pelvic into the I/O carrier, **13.60** each | 0.60, then 3.40 (packs below the pelvic crossmember) | the ToF arms' roots, 32.0 mm from the shoulder crossmember; the tail's pivot, 34.6 mm aft of the pelvic one |
| **C** | **as B, with the DC-DC block 20 mm aft and 3 mm up, the I/O carrier 20 mm forward, and the four tray stays re-angled to end on the inset rails** | **none** for the moved parts over every leg's, the tail's and the head's range (items 1 and 8-12, inside the legs, head and tail and between the legs, are separate) | **0.60 (rail to the reserve block), 0.83 (front tray stays to the Pilz block), 2.00 (DC-DC block to the Pilz block), 3.40 (shoulder crossmember to the DC-DC block, pelvic crossmember to the I/O carrier, and to the packs), then 3.90 and more** | as B: re-root the ToF arms; a tail bracket to its pivot |

In C the crossmembers sit on the rails (11.2 and 7.2 mm overlap: joined) and the shoulder one still carries the neck
cradle (2.35 mm overlap); the moved DC-DC block and I/O carrier clear their crossmembers by 3.40 mm each (13.60 mm of
overlap in B); the X6s pass the shoulder crossmember at 6.11 mm (roll +24, pitch +28.5, flexion -47 deg), the hind
cranks pass the rails at 4.70 mm and the new hind brackets at 9.37 mm; the tail pitch servo clash (item 7) is gone.
The tray stays are drawn ending on the old rail line (y 83.2 mm, `fore128.xml:71-74`), so on the inset rails they would
stand 3.2 mm outboard of the rail's outer surface (88.0 against 84.8 mm), where the rear pair passes the hind cranks at
1.89 mm (section 1.2); in C they are re-angled to end on the inset rails' axis, their tray ends kept: they still meet
the rails and the tray, the rear pair passes the hind cranks at 9.50 mm, and the front pair comes 0.2 mm nearer the
Pilz block (0.83 mm, from 1.03). In A and B the stays are left as drawn. Shell and cables are not solid-part clashes
but must change: rib 4 meets the shoulder crossmember by 10.4 mm and the rails by 7.79 mm, and the looms meet the rails,
the brackets and the stays by 2.75-7.29 mm (rib relieved, looms re-routed in CAD). The
reserve block (a 0.138 kg mass placeholder, which the 21 Nov 2026 review asks to centre at y = 0 anyway, R7) is 0.60 mm
from the right rail. **C is a proposal for the chassis CAD (D5), not a design**; applying it changes the training model.

Also from the 21 Nov 2026 review, unchanged in the adopted model: the bay's tightest gaps (the review's item 5 and R3,
"expand chassis bay"; `git show 2ac57608:hardware/triceratops_build/REVIEW_60CM.md`, lines 118-122, 158). No record
sizes the gap each needs: it is set in the chassis CAD (D5) from the chosen parts' sheets. What each must hold, and
the room the model has to open it (free travel of each part before it meets another chassis part, the shell's ribs
included, cables left out; `bay_room` in the check, today's layout):

| Pair | Gap, mm | Must hold (the review) | Room in the model | Candidate move |
|---|---|---|---|---|
| NVMe tray / Pilz block | **0.50** | isolator deflection, cables | the Pilz block is boxed in on all six sides (0-2.5 mm: power board ahead, tray stays beside, Teensy carriers above; it rests on the tray below); the Jetson with its NVMe tray can move 8.0 mm aft (to the packs) | the Jetson and its tray up to 8 mm aft (gap up to 8.5 mm; the Jetson to the Pilz block, 1.50 mm today, opens by as much); then the packs, free 27 mm aft today (3.4 mm under the pelvic crossmember in C), aft with them for more |
| pack / pack | **1.00** | swelling, straps; leads (8 mm ahead, 17.4 mm aft today) | pack 1 outboard 8.5 mm (to the rear tray stay), pack 2 outboard 2.5 mm (to the reserve block) | both packs outboard: gap up to 12.0 mm; more once the reserve block is centred (R7) |
| Teensy carriers / Pilz block | **1.00** (vertical) | DIN terminals, spacing against electrical interference | the carriers are free upward for 30 mm or more | raise the carriers on standoffs, 10 mm or more |
| IMU / blowers | **1.00** (vertical) | vibration isolation (fans 1 mm from the only IMU) | the IMU can drop 3.5 mm (to the Jetson) or move sideways 30 mm or more | move the IMU out from under the blowers (it leaves their footprint after 22 mm in +y), or drop it 2.5 mm onto an isolator; either moves the model's IMU site, which the training model pins |

Next tightest: 1.03 mm (front tray stays to the Pilz block; 0.83 mm in C, its stays re-angled), 1.10 mm (pelvic crossmember to the tail yaw servo: gone in
C) and 1.50 mm (Jetson to the Pilz block). The drive cranks stand **3.00 mm** (front) and **6.25 mm** (hind) off their
drives' faces.

The front leg's own packaging (PR #697's repackaging) holds, as recorded (`rl/analysis/PLAN_FORELEG_D22.md:37-41`):
X6 to its upper strut 8.70 mm, X6 to the rail 7.68 mm, motor plate to the rail 4.68 mm, X6 to the output crank
7.16 mm, the moving rod past the X6 3.50 mm.

**2. The pushrod.** Recomputed from the model: the review's figures stand. Each linkage is a true parallelogram (28.00
mm cranks at both ends; pin to pin 81.50 mm front, 98.00 mm hind); its transmission angle is 90 deg at neutral and
falls to **43.0 deg at -47 deg** of flexion (54.0 deg at +36). Rod force there: **1,979 N** at the joint-side limit
(37.8 N m, the review's figure), **2,199 N** at the drive's 42 N m limit, **3,142 N** at the X6's 60 N m peak. The
hind rods carry half that (X4: 990 / 1,100 / 1,571 N), and **1,885 N** at the X4's 36 N m stall torque. The static
case is the largest torque `sim/data/drives.json` gives for each drive (a drive without a firmware limit, or a
fault): the X6-60's 60 N m peak (the file gives it no stall torque) and the X4-36-L's 36 N m stall torque (above its
30 N m peak). With safety factor 2.0 on static strength and buckling at 3,142 N (front) and 1,885 N (hind) and 1.5 on
fatigue:

- **Front: steel M8 rod ends** (DIN ISO 12240-4 class, steel on steel, static rating about 12.9 kN) on a **rolled-
  thread M8 steel rod of property class 8.8 or better**. It is the only candidate of 15 that passes every check,
  including fatigue at the drive's limit every cycle, fully reversed (n = 1.55), a bound no simulated gait reaches.
  A plain cut-thread 1018 rod at M8 passes the static checks (SF 4.3) and the walking and 0-to-limit fatigue cases
  (n 2.87, 1.63), but not that bound (0.92). M6 in 1018 passes static (SF 2.37 on the rod, 2.61 on the rod end) but
  not 0-to-limit fatigue (0.89); M6 in class 8.8 passes static and walking fatigue and fails only the bound (0.85). M5 fails static in every material (its rod ends' lowest listed static rating gives SF 1.08). Aluminium
  rods fail fatigue at every size here. The X6-60's stall torque is in no record; at 1.2 x its peak (72 N m, the
  X4's stall-to-peak ratio: an assumption) the front rod would carry 3,770 N and the M8 class 8.8 part would still
  pass (SF 5.63 tension, 4.69 compression, 3.42 on the rod end, 2.10 on the lowest M8 rating found; the 8 mm clevis
  pin 2.92-3.55).
- **Hind: M6 class 8.8 or M8 1018** pass everything at the stall torque's 1,885 N (M6 8.8: SF 6.18 tension, 3.46
  compression, 4.35 rod end; M8 1018: 7.18, 7.18, 6.84), as at the 1,571 N peak. One M8 steel part for all four legs
  keeps one spare.
- **Euler buckling is not what limits these rods**: at 81.5 mm they are short columns (slenderness 33-77; 39-92 at
  98 mm), so Johnson's parabola governs in most cases, and in steel the thread's tensile area limits first.
- **The crank pins matter more than the rod.** A pin cantilevered from the crank at the 3,142 N peak is stressed to
  375 MPa in bending at 8 mm diameter and 6 mm lever (SF 1.55 against an 8.8 proof stress of 580 MPa; fatigue n 1.17
  fully reversed at the limit force, below 1.5), 625 MPa at a 10 mm lever. **Use a clevis** at both cranks: an 8 mm
  pin with the rod end's 8 mm ball between 4 mm plates carries 31 MPa of shear and 125 MPa of bending across the
  clevis (156 MPa with a 1 mm washer each side), 136-165 MPa combined (SF 3.5-4.3) and fatigue n 2.8-3.5 at the limit
  force, fully reversed. Or a 10 mm cantilevered pin with a short lever (192 MPa at 6 mm, SF 3.02, fatigue n 2.28).
- **Open:** the X6's output bearing then carries up to 3.1 kN radially, offset from its face (3.00 mm gap plus half
  the crank), and its allowed radial load and moment are in no record; the rod ends' dimensions (two of them plus
  the rod within 81.5 mm, and no wider than the model's 9 mm rod if the 3.50 mm clearance to the X6 is to stay)
  need a maker's sheet.

## 1. Packaging (desk geometry)

### 1.1 Method

`sim/check_packaging60.py` on the adopted model, mujoco 3.10.0:
- every pair of parts of each leg (roll yoke, upper link, lower link, foot), and each leg part against every part of
  the chassis, head and tail, over 9 values per joint across roll x pitch x flexion (729 poses, ankle at 0), and the
  foot against its leg over flexion x ankle;
- a probe copy of each leg's pushrod placed where the parallelogram puts it at every pose (the model's rod is a
  fixed capsule on the upper link and never moves: `sim/build_foreleg60.py:288`);
- every pair of chassis parts at neutral, and the head and tail over their joint ranges;
- every pair of legs against each other over all eight of their joints (roll, pitch, flexion and ankle of each): 9
  values per joint, every pose of one leg against every pose of the other (43 million poses for two feet), the
  distance in closed form for capsules and with `mj_geomDistance` for the other shapes; then, from the 8 deepest grid
  local minima of every pair of parts that comes within 50 mm, a bounded local search (Nelder-Mead over the joints
  the two parts move with) between grid points. The moving rod stands in for the model's fixed one. A test places a
  clash between grid points and checks the search finds it while the grid alone does not; a 7-value grid gives the
  same least distances for every pair of legs (desk runs, 28 Nov 2026). Then candidate joint stops, each swept the
  same way, and the soles' lower halves (the real half-capsule rocker, F17) at the deepest pose of two soles;
- the room around the bay's tightest pairs: how far each part can move along each axis before it meets another
  chassis part (0.25 mm steps, up to 30 mm);
- trial fixes: candidate positions for the crossmembers, rails and tail servo, each moved part swept as above (a
  moved part also against the tail and the head over their ranges) and measured at neutral against the chassis parts
  left in place and the other moved parts, then three combined layouts, since the single trials interact; for each
  moved frame member, its distance to what it must end on or carry.

Distances are `mj_geomDistance` (negative = depth of overlap: the shortest translation that separates the pair).
Two exact zeros between the tail servos were failed solves; re-measured 0.06 deg away they read 0.68-1.10 mm (listed
at the end of the report file). The drive housings are the records' envelopes (X4 55 x 61 mm, X6 80 x 67.5 mm,
`sim/data/drives.json`; the check confirms the model matches them): housings only, without output flanges, connectors
or cable exits, which no record sizes.

Pairs are classed so that the joints and mounts the model draws as overlaps by design (a yoke on its drive's output,
a strut end in its axle, PR #697's motor plate on the X6, frame members joined to each other) are not counted as
clashes; the list with reasons is `JOINED` in the script. Shell parts are reported separately (1.4).

### 1.2 The legs (worst over the range; neutral in brackets)

Front legs (both sides alike):

| Pair | mm | Pose of the worst (roll, pitch, flexion, deg) |
|---|---|---|
| shoulder crossmember / X4 pitch drive | -28.96 (-1.10) | +24, -38, -47 |
| shoulder crossmember / shoulder yoke; / upper strut | -10.83 (+15.20; +15.40) | +24, -38, -47 |
| capsule sole / lower strut (the full simulated capsule, F17) | -21.60 (-21.60) | any |
| elbow axle / output crank (axial gap) | +0.75 | same body |
| rail / shoulder yoke | +1.85 (+7.20) | +24, -38, -47 |
| X6 / drive crank (axial gap) | +3.00 | same body |
| X6 / pushrod; moving rod / X6 | +3.50 | every pose |
| rail / offset motor plate | +4.68 (+27.93) | +24, +19, -47 |
| rail / X4 pitch drive | +5.93 (+16.70) | +24, -38, -47 |
| X4 roll drive / X4 pitch drive | +6.00 | same at every pose |
| X6 / output crank | +7.16 | over flexion |
| rail / X6 | +7.68 (+18.45) | +24, +9.5, -47 |
| upper strut / X6 | +8.70 | same body |
| moving rod / elbow axle | +9.44 (+17.85) | +13.5, +38, -47 |

Hind legs (both sides alike):

| Pair | mm | Pose of the worst |
|---|---|---|
| pelvic crossmember / X4 pitch drive | -40.10 (-40.10) | neutral |
| upper strut / X4 knee drive | **-38.70 (-38.70)** | same body |
| pelvic crossmember / shoulder yoke; / upper strut | -24.80; -20.80 (the same) | neutral |
| capsule sole / lower strut (F17) | -21.60 (-21.60) | any |
| rail / shoulder yoke | -11.61 (-8.80) | -18, -38, -47 |
| rail / drive crank; / output crank | -2.91 (+14.80; >50) | -18, -38, -47 / +25.6 |
| rail / pushrod (fixed); moving rod / rail | -2.41 (+28.02) | -18, -38 |
| elbow axle / output crank (axial gap) | +0.75 | same body |
| rear tray stay / drive crank | +1.89 (+15.27) | -18, -38, -47 |
| X4 roll drive / X4 pitch drive | +6.00 | |
| X4 knee drive / drive crank (axial gap) | +6.25 | same body |
| X4 knee drive / pushrod; moving rod / X4 | +6.76; +6.75 | every pose |
| moving rod / rear tray stay | +9.36 (+31.07) | -18, -9.5, -47 |

The hind linkage against the rails, stepped in roll 0.5 deg at a time over the pitch x flexion grid: clear (0 mm) from
roll -2.0 deg outward; at roll 0 the least gap is 0.45 mm (the crank discs' inner face, y 93.25 mm, passes 0.45 mm
beside the rail's outer surface, y 92.8 mm); 2.91 mm of overlap at the inward stop; never 3 mm clear anywhere in the
range (`hind_roll_limit_for_rails` in the JSON). Whether a gait uses inward hind roll with the leg
pitched forward is not in the gait reports (they record each foot's ankle angle percentiles, but not the roll,
pitch or flexion angles).

The legs against each other (section 6 of the report file; 9 values per joint over all eight joints of each pair of
legs, then the local search; deg are roll, pitch, flexion, ankle; both sides alike):

| Pair of legs | Parts | Least, mm (grid's least) | Pose of the least | Share of the grid's poses overlapping |
|---|---|---|---|---|
| fore / hind, one side | soles (foot shape) | **-48.00** (-46.65): the whole capsule; the lower halves (rocker) overlap by 17.97 cm³ | fore -2.4, +20.8, -47.0, -21.7; hind -2.0, -26.6, -45.9, +21.4 | 0.34 % |
| | a sole and the other leg's lower strut or ankle hub (foot shape) | -33.60 (-32.0 to -33.4) | fore pitch +28.5 to +38, hind pitch -27.9 to -38; each of the four pairs at the full depth (the sole's 24 mm radius plus the strut's or hub's 9.6 mm), so its pose is one of many | 0.12-0.16 % |
| | lower struts; an ankle hub and the other leg's lower strut or hub (two solid parts) | **-19.20** (-17.6 to -19.2): fully crossed | struts: fore -18.0, +37.3, -46.2; hind -13.0, -37.9, -45.0 | 0.03-0.06 % |
| | limb covers (shell), with each other or the other leg's foot or strut | -28.6 to -43.0 | | 0.10-0.26 % |
| hind / hind | soles (foot shape) | **-8.49** (-8.09); the lower halves overlap by 2.05 cm³ | both -18.0, +22.4, -47.0, -22.0 | 0.001 % |
| | an ankle hub and the other sole; a cover and the other sole | +6.55; +7.94 | both at roll -18, pitch +21 to +23, knee -46 to -47 | none |
| fore / fore | soles | +26.28 | both -18.0, -27.6, -47.0, +22.0 | none |
| fore / the other side's hind | | more than 50 | | none |

On the grid, the fore / hind overlap of solid parts (soles included; 0.42 % of the pair's poses) needs fore pitch +9.5
deg or more, hind pitch -19 deg or less, fore flexion +4.9 deg or less and hind flexion -5.5 deg or less, roll and
ankle values across their whole ranges taking part; with the soles left out (0.11 %), fore pitch +19 or more, hind
pitch -28.5 or less and both flexions -15.9 or less. These are each joint's bounds, not a region where the legs always
meet: of the grid's poses inside them 7.7 % overlap (7.7 % with the soles left out), and only where the two legs' roll
angles are within 21 deg of each other (10.5 deg). The hind / hind overlap appears on the grid only at roll -18 on both
legs. Grid values bound the region on the grid only; the candidate stops (items 9 and 10, last column; `leg_limits` in the check) are each swept like the
full ranges, grid and local search, over the narrowed range. This report's first version had sampled
each joint at its range ends and 0 only (3 values; struts 19.20 mm "at range corners only", hind soles 3.64 mm clear,
front legs 33.0 mm apart): it stepped over these poses.

### 1.3 Chassis bay, head and tail (neutral unless a pose is given)

| Pair | mm | Note |
|---|---|---|
| pelvic crossmember / hind roll drives | -41.10 | F21 |
| rails / hind roll drives | -23.10 | new |
| rails / front roll drives | -8.19 | new |
| shoulder crossmember / front roll drives | -1.10 | F19 |
| tail pitch servo / pelvic crossmember | -3.90 (-11.65 at tail yaw -24) | 21 Nov review: 3.9 |
| tail pitch servo / tail yaw servo | -5.68 at yaw -24, pitch +15 (+7.40 neutral) | new |
| neck yaw servo / neck cradle | -0.44 at every neck pitch | new |
| NVMe tray / Pilz block | +0.50 | 21 Nov review |
| pack / pack; IMU / blowers; Teensy carriers / Pilz block | +1.00 | 21 Nov review |
| front tray stays / Pilz block | +1.03 | |
| pelvic crossmember / tail yaw servo | +1.10 | |
| Jetson / Pilz block | +1.50 | |
| power board / Pilz block; tray / NVMe; Jetson / audio box; pack / reserve block | +2.50 | |
| packs / hind roll drives | +8.45 | |

The packs, the Pilz block and the Jetson rest on their trays (0.00 mm, classed as touching).

### 1.4 The shell and the limb covers (to be shaped in CAD)

The model's limb covers are capsules along the struts; the real covers would be shaped around the drives. As drawn
they overlap their own legs' drives: the front upper cover into the X6 by 4.1 mm, the hind upper cover through the X4
knee by 51.5 mm, the upper covers into the pitch drives by 51.0 mm. The covers also meet the shoulder crossmember
(23.6 mm at roll +24, F19) and the pelvic crossmember (33.6 mm). The shell's ribs and seams meet the hind cranks and
rods by up to 12.67 mm (rib 1 against the drive crank at roll +3, pitch -28.5, flexion -47) and the front X6 comes
within 9.82 mm of rib 4. The tail's fairing (its convex hull, which overstates a concave mesh) meets the hind roll
drives by up to 15.62 mm and the pelvic crossmember by up to 30.26 mm when the tail yaws 24 deg. The shell is cut
around the legs' swept volumes and the tail's root in CAD; the swept envelope of the hind linkage is the largest.

### 1.5 Trial fixes (each moved part swept against everything, the other moved parts included)

Each moved part is swept against all four legs over their ranges (with the moving rods), the tail and the head over
theirs, and measured at neutral against the chassis parts left in place and against the other moved parts. A row lists
the moved parts' pairs only: the parts a trial leaves in place keep the gaps of sections 1.2 and 1.3. "Meets" gives a
moved frame member's distance at neutral to what it must end on or carry (<= 0: the two meet and can be joined;
`MOUNTS` in the check). The single trials each move one part and keep the rest of
today's layout: **the crossmember rows assume today's rails** (x +-260 mm, y +-83.2 mm), and the rail rows keep
today's crossmembers at x +-260 mm.

| Trial | Least distance and what it meets | Meets | Reading |
|---|---|---|---|
| Shoulder crossmember between the rails only (y +-83.2 mm) | -1.32 pitch drive (roll +24); -1.10 roll drive | rails' ends -11.2 | not enough |
| ... and raised 6 mm | +0.61 pitch drive (roll +24); +4.90 roll drive | rails' ends -5.2 | clears every part |
| ... and raised 10 mm | +3.01 pitch drive; +8.90 roll drive; nothing else within 10 mm | rails' ends -1.2; neck cradle and ToF roots still on it | **clears by 3 mm, on today's rails** |
| Shoulder crossmember 48 mm aft (x 212 mm), between today's rails | -13.6 DC-DC block; -1.20 X6 (roll +24) | rails -11.2; ToF roots +32.0 | worse |
| Pelvic crossmember between the rails only | -41.10 roll drive; -24.80 yoke; -11.65 tail pitch servo (tail yaw -24) | rails' ends -11.2 | not enough |
| ... raised 45 mm | -13.6 tail yaw servo; +0.13 pitch drive (roll +24); +3.90 roll drive; +4.49 tail pitch servo | rails +33.8; tail pivot +31.4 | clears the drives only, and ends in the air |
| ... 48 mm forward (x -212 mm), raised 4 mm | -13.6 I/O carrier; -0.40 hind drive cranks (range corner); +3.40 packs | rails -7.2; tail pivot +34.6 | clears the drives only |
| Rails ending at x +-215 mm | -2.91 hind cranks, -2.41 hind rods (roll -18, pitch -38); +4.90-4.97 roll drives | (crossmembers 45 mm past the ends) | clears the drives and yokes |
| Rails ending at x +-215 mm and 8 mm further in (y +-75.2 mm) | +4.70 hind cranks; +4.90 roll drives; +0.60 reserve block; looms -6.35 and rib 4 -7.79 (to re-route and relieve) | (crossmembers 45 mm past the ends) | **clears every solid part by 4.7 mm** except the reserve block (0.6 mm); the tray stays, left as drawn, then stand 3.2 mm outboard of the rails (1.89 mm to the hind cranks, as today) |
| Tail pitch servo 7 mm aft | -5.26 crossmember (yaw -24) | | not enough |
| Tail pitch servo 16 mm aft | +2.96 crossmember; +9.78 tail yaw servo | | **clears by 3 mm** |
| Combined A: the last rail trial, drives on brackets, crossmembers at x +-260 mm between the rails' lines (shoulder 10 mm up, pelvic 45 mm up) | -13.60 tail yaw servo; +0.60 reserve block; +3.91 hind roll drives; +4.49 tail pitch servo; +4.70 hind cranks | shoulder: rails +26.9, roll drives +8.9; pelvic: rails +49.4, roll drives +3.9, tail pivot +31.4 | one clash, and no frame: both crossmembers end in the air |
| Combined B: as A, crossmembers on the rails' ends (x +-212 mm; pelvic 4 mm up) | -13.60 DC-DC block; -13.60 I/O carrier; +0.60 reserve block; +3.40 packs | rails -11.2 / -7.2; neck cradle -2.35; ToF roots +32.0; tail pivot +34.6 | a ladder frame; two blocks in the way |
| **Combined C: as B, DC-DC block 20 mm aft and 3 mm up, I/O carrier 20 mm forward, the four tray stays re-angled to end on the inset rails** | **+0.60 reserve block; +0.83 front tray stays to Pilz block; +2.00 DC-DC block to Pilz block; +3.40 shoulder crossmember to DC-DC block, pelvic crossmember to I/O carrier, and packs; +3.90 hind roll drives; +4.00 I/O carrier to packs; +4.40 power board; +4.70 hind cranks; +6.11 X6 (roll +24); +7.75 rear tray stays to packs; +9.50 rear tray stays to hind cranks (range corner)**; shell and looms as in the last rail trial, plus rib 4 -10.4 and the looms through the stays -7.29 | as B; the stays still meet the rails (-14.4) and the tray (-8.4) | **the moved parts clear every solid part**; ToF roots and the tail's mount to redesign |

These are proposals for the chassis CAD (D5), not designs: moving parts in the model changes the training model,
which the FINAL step sheets pin, so none of them is applied here.

### 1.6 What this check does not cover

- Flanges, connectors, cable exits and fasteners (unsized in any record); tolerances, deflection under load and wiring
  (PR #697's bench plan asks for worst-case manufactured, wired and loaded gaps; `git show 65a44fad:hardware/
  triceratops_build/bench/BENCH_PLAN_FORELEG.md`, line 11).
- Poses between grid points, for the legs against the chassis, head and tail (sections 1.2-1.4; the legs against each
  other are searched between grid points, above): rerun at 17 values per joint (desk run, 28 Nov 2026, scratch
  output), no solid pair deepens by more than 0.08 mm and no new pair comes within 10 mm; the shell's rib 1 against
  the hind drive crank deepens most, by 0.97 mm (12.67 to 13.64 mm).
- The walking envelope: the ranges checked are the joints' full ranges. Over the 128 mm model-based trot episode
  that F19 records (seed 101) the shoulder crossmember's overlap reached 5.3 mm (`DECISION_LOG.md`, F19); no gait
  report records the roll, pitch or flexion angles (only ankle percentiles, `rl/analytics/gait_report.py:572`).
- The chest camera's view (the 21 Nov review's "blinded" OAK-D Lite) and the tail yaw servo outside the shell: not
  packaging overlaps; unchanged in the adopted model.

## 2. The foreleg pushrod (desk calculation)

### 2.1 Geometry and forces

From the model, through the parallelogram (`design/pushrod60.py`, the same linkage code as the packaging check):

| | Front (X6-60) | Hind (X4-36-L) |
|---|---|---|
| Crank radius, both ends | 28.00 mm | 28.00 mm |
| Pin to pin (rod length) | 81.50 mm | 98.00 mm |
| Flexion range | -47 to +36 deg | -47 to +36 deg |
| Transmission angle: neutral / -47 deg / +36 deg | 90.0 / **43.0** / 54.0 deg | 90.0 / 43.0 / 54.0 deg |
| Drive torque: peak / model limit (0.70 x peak) / joint-side limit (x 0.90) | 60 / 42 / 37.8 N m | 30 / 21 / 18.9 N m |
| Stall torque (`stall_nm`) | none in `drives.json` | 36 N m |
| Static case: the largest of the two | 60 N m (peak) | **36 N m (stall)** |
| Rod force at 43 deg: peak / limit / joint-side limit | **3,142 / 2,199 / 1,979 N** | 1,571 / 1,100 / 990 N |
| Rod force at 43 deg, static case | **3,142 N** | **1,885 N** |
| Rod force at 90 deg: peak / limit | 2,143 / 1,500 N | 1,071 / 750 N |
| Worst walking RMS torque, joint side (simulation) | 8.52 N m (seed 2, left fore) | 6.38 N m (seed 3, right hind) |
| Walking fatigue amplitude: sqrt(2) x RMS / 0.90, at 43 deg | 701 N | 525 N |

Rod force = drive torque / (crank radius x sin(transmission angle)). Torques: `sim/data/drives.json` (X6-60
`peak_nm` 60 and `stall_nm` null; X4-36-L `peak_nm` 30 and `stall_nm` 36; `transmission_efficiency` 0.90); the 0.70 rule is the model's (`fore128.xml:313`,
`forcerange` 37.8). RMS: `rl/analytics/results/fore128-s{0..3}/gait_report.json`, target case, `rms_nm` worst episode.
The 43 deg minimum agrees with PR #697's bench plan ("calculated minimum 43 deg transmission angle"; `git show
65a44fad:.../bench/BENCH_PLAN_FORELEG.md`, line 10) and passes the build's bench rule of at least 40 deg
(`bench/BENCH_PLAN_FORELEG.md:64`, B6, written for the 75 cm T20 foreleg). The 21 Nov 2026 review's 1,979 N and 3,142 N
(`git show 2ac57608:hardware/triceratops_build/REVIEW_60CM.md`, line 117) are reproduced to the newton; the review
computed the limit case at the joint-side torque, while the rod carries the drive-side 42 N m (2,199 N). Simulated
walking reaches the limit: in all four policies the front elbows sit at their episode's randomised torque limit on
0.02-0.9 % of steps (`at_episode_limit`: mean of episodes 0.02-0.83 %, worst episode 0.07-0.89 %; never at the
nominal 37.8 N m, `at_nominal_limit` 0.0; target case, all four seeds' `gait_report.json`, simulation). The peaks
the reports give, 36.2-37.1 N m (`peak_nm`, worst episode; identical across the four seeds), are therefore those
episodes' clamps (`gait_report.py:454` takes the largest actuator force, which the episode's force range clips), not
what the gait would ask for unclipped, which can be more. The drives saturate; the rod sizing takes the limit and the
peak, not the walking peak.

The front rod's force exceeds the 75 cm T20 design's 1.59 kN at 40 deg (`hardware/triceratops_kickoff/
PARTS_PREMIUM.md:253`) because the cranks shrank from 35-36 mm to 28 mm. Longer cranks would cut it in proportion
(36 mm: 2,444 N at the peak), at the cost of a larger swept envelope next to the rails (section 1.2).

### 2.2 Sizing rule

- Static, on the static case (3,142 N front, 1,885 N hind): the largest torque `sim/data/drives.json` gives for the
  drive, which a drive without a firmware limit, or one in a fault, can reach stalled against a load. For the
  X4-36-L that is its `stall_nm`, 36 N m, above its `peak_nm` of 30 (the first version of this report took the
  peak, 1,571 N); for the X6-60 it is `peak_nm`, 60 N m, as the file gives it no `stall_nm` (null). Tension and
  compression at the thread's tensile stress area
  against yield (proof stress for class 8.8), buckling of the body pinned at both ends over the pin-to-pin length,
  and the rod end's catalog static rating: **safety factor 2.0** each. The kickoff list asked for rod ends "rated at
  least 2x" the design pull (`PARTS_PREMIUM.md:256`) and the review for at least 3.2 kN (REVIEW_60CM.md line 164);
  2 x 3,142 N = 6.3 kN covers both.
- Fatigue at the thread root, Goodman, **safety factor 1.5**, endurance = 0.5 x ultimate (steel) or the 5 x 10^8-cycle
  reversed strength (aluminium), x 0.70 for axial load, / Kf for the thread. Three cases: F1 fully reversed at the
  limit force every cycle (a bound: no simulated gait does it); F2 from 0 to the limit force every cycle; F3 fully
  reversed at the walking amplitude. Cycles: the gait reports' stride frequencies are 0.64-1.67 Hz per foot, so one
  hour of walking is 2,300-6,000 cycles per leg and 10^7 cycles about 1,700-4,300 hours: design for unlimited life.
- Rod ends: the ratio of the limit force to the dynamic rating C is shown; a plain rod end's life needs the maker's
  method (oscillation angle, frequency, liner).

### 2.3 Candidates, front rod (81.5 mm)

| Rod | Thread, body | SF tension | SF compression (buckling formula) | SF rod end C0 | Limit / C | n F1 | n F2 | n F3 | Static | Fatigue bound F1 | Walking F2, F3 |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 1018 CD steel, cut | M5, 5 mm | 1.67 | 1.67 (Johnson) | 1.08 | 0.88 | 0.36 | 0.63 | 1.11 | fail | fail | fail |
| 1018 CD steel, cut | M6, 6 mm | 2.37 | 2.37 (Johnson) | 2.61 | 0.61 | 0.50 | 0.89 | 1.58 | pass | fail | fail |
| 1018 CD steel, cut | M8, 8 mm | 4.31 | 4.31 (Johnson) | 4.11 | 0.38 | 0.92 | 1.63 | 2.87 | pass | fail | pass |
| 1018 CD steel, cut | M8, 10 mm | 4.31 | 4.31 (Johnson) | 4.11 | 0.38 | 0.92 | 1.63 | 2.87 | pass | fail | pass |
| class 8.8 stud, rolled | M5, 4.25 mm | 2.62 | 1.49 (Johnson) | 1.08 | 0.88 | 0.60 | 1.08 | 1.89 | fail | fail | fail |
| class 8.8 stud, rolled | M6, 5.06 mm | 3.71 | 2.58 (Johnson) | 2.61 | 0.61 | 0.85 | 1.53 | 2.67 | pass | fail | pass |
| **class 8.8 stud, rolled** | **M8, 6.83 mm** | **6.76** | **5.62 (Johnson)** | **4.11** | **0.38** | **1.55** | **2.78** | **4.87** | **pass** | **pass** | **pass** |
| 6061-T6, cut | M6, 6 mm | 1.77 | 1.74 (Johnson) | 2.61 | 0.61 | 0.22 | 0.41 | 0.69 | fail | fail | fail |
| 6061-T6, cut | M8, 8 / 10 mm | 3.21 | 3.21 (Johnson) | 4.11 | 0.38 | 0.40 | 0.74 | 1.26 | pass | fail | fail |
| 7075-T6, cut | M6, 6 mm | 3.22 | 2.16 (Euler) | 2.61 | 0.61 | 0.36 | 0.68 | 1.14 | pass | fail | fail |
| 7075-T6, cut | M8, 8 mm | 5.86 | 5.67 (Johnson) | 4.11 | 0.38 | 0.66 | 1.24 | 2.07 | pass | fail | fail |

(6061 M5 and 7075 M5 fail static; 7075 M8 on a 10 mm body is as on 8 mm. A stud's body is its stress-area diameter.)

**Hind rod (98.0 mm), in short, at the static case's 1,885 N:** 1018 M8 (n F1 1.83; SF 7.18 static), 8.8 M6 (1.71;
SF 3.46 in compression, 4.35 on the rod end) and 8.8 M8 (3.11) pass every check; 1018 M6 passes static (SF 3.94) and
the walking cases (F1 1.01); 8.8 M5 passes walking (F1 1.21) but now fails static (SF 1.75 in compression, Euler, and
1.80 on its rod end; 2.10 and 2.16 at the 1,571 N peak); 1018 M5 fails static on its rod end (1.80); aluminium fails F2
or F3 except 7075 M8 (F1 1.32, F2 2.47, F3 2.77: passes walking, not the bound). Full table:
`design/pushrod60_out.json`.

**Euler and Johnson.** Euler's load at 81.5 mm for a 6 mm steel rod is 18.9 kN, but its slenderness (54) is below the
transition (103 for 1018), so Johnson's inelastic load, 9.0 kN, governs, and at the thread the rod yields first
(7.4 kN). At the front only 7075-T6 at 5-6 mm, with its higher yield, reaches the Euler range (at the hind's 98 mm, also
6061-T6 M5 and the class 8.8 M5 stud; 1018 M5 stays in Johnson's range).

### 2.4 Crank pins and the drive's output

Static at the front's static case (3,142 N, the X6's peak); fatigue n at the limit force (2,199 N), F1 fully reversed / F2 0 to limit,
Goodman, endurance 0.5 x 800 MPa x 0.767 (the Marin surface factor for a machined surface, 4.51 x 800^-0.265) = 307
MPa, size factor taken as 1, no stress raiser counted. Class 8.8 pin, proof stress 580 MPa:

| Pin | Cantilever, 6 mm lever | Cantilever, 10 mm lever | Clevis, no gap: shear / bending / combined | Clevis, 1 mm gap each side: bending / combined |
|---|---|---|---|---|
| 6 mm | 889 MPa (SF 0.65; n 0.49 / 0.71) | 1,482 MPa (0.39; 0.30 / 0.43) | 56 / 259 / 277 MPa (SF 2.10; n 1.69 / 2.44) | 333 / 347 MPa (1.67; 1.31 / 1.90) |
| 8 mm | 375 MPa (1.55; 1.17 / 1.69) | 625 MPa (0.93; 0.70 / 1.01) | **31 / 125 / 136 MPa (4.26; 3.51 / 5.07)** | **156 / 165 MPa (3.51; 2.81 / 4.05)** |
| 10 mm | 192 MPa (3.02; 2.28 / 3.30) | 320 MPa (1.81; 1.37 / 1.98) | 20 / 72 / 80 MPa (7.26; 6.09 / 8.80) | 88 / 95 MPa (6.13; 4.98 / 7.20) |

The lever is the distance from the crank's face to the rod end's ball centre (6 and 10 mm are assumptions: half a
rod-end width plus a washer). The clevis pin bends across the clevis: M = F/2 x (t/2 + g + b/4), the load spread over
the ball's width b (taken as the pin's diameter: no maker's sheet), the reactions over plates of thickness t (4 mm,
an assumption), with a gap g each side; at 8 mm and no gap, M = 1,571 N x 4 mm = 6.3 N m. Bending, not shear,
governs: the shear-only figure the first version of this report gave (31 MPa) understated the pin's stress about
fourfold. The combined figure adds the bending peak (mid-span) and the shear peak (at the plates' faces) as if they
met at one section, which they do not: a bound. Plate bearing pressure at the peak: 49 MPa at 8 mm (65 at 6, 39 at
10). A cross-hole or a ring groove at the clevis pin's mid-span, or a press fit or thread at the cantilever's crank
face, adds a stress raiser where the moment peaks and lowers n further; keep them out of those sections. Hind pins
carry 60 % of the front's static force at the X4's stall torque (8 mm clevis: 82-99 MPa combined, SF 5.85-7.10;
fatigue at their limit force n 5.6-7.0). The drive crank also loads the X6's output bearing with
up to 3.1 kN, offset from the drive's face by the 3.00 mm gap plus half the crank's thickness (the model's crank disc
is 10 mm thick): about 25 N m of overturning moment at an 8 mm offset. Neither the X6's nor the X4's allowed radial
load or moment is in `sim/data/drives.json` or the kickoff list; the kickoff list wanted an angular-contact bearing
pair at the output rated at least 2x (`PARTS_PREMIUM.md:256`).

### 2.5 Fit in the model's space

- The rod ends' eyes sit at the crank pins, 28 mm from each axis, in the 9 mm-wide lane the model's rod occupies
  (rod diameter 9 mm, 3.50 mm from the X6's face). A rod end wider than 9 mm along the pin narrows that gap; a clevis
  widens the crank. Either way the lane is re-checked in CAD.
- Two rod ends, the rod between and the lock nuts fit in 81.5 mm pin to pin only if each rod end's centre-to-thread-
  end length leaves room: check the chosen part's sheet (the dimensions were not available from this session).
- The model's linkage mass is T15's 75 cm four-bar, 0.0825 kg per leg, not scaled (`sim/build_model60.py:60`); weigh
  the chosen parts against it.

## 3. For the records (the lead's to change; nothing here edits them)

- **DECISION_LOG.md**, findings: the hind knee overlap (38.70 mm, confirmed, every pose); the rails into the roll
  drives (8.19 / 23.10 mm) and the moving hind yokes (8.80-11.61 mm); the hind cranks and rods against the rails
  (0.45 mm clear at roll 0, overlapping inward of -2 deg, 2.91 mm at -18 deg); the tail pitch servo's clash growing
  to 11.65 mm with tail yaw and its 5.68 mm clash with the tail yaw servo; the front and hind legs on one side
  meeting under the body at some poses with the fore leg pitched aft, the hind leg forward and the two legs' rolls
  within 21 deg of each other (soles 48.0 mm, the real rockers too; lower
  struts 19.2 mm; item 9, with its candidate stops); the two hind soles meeting at the inward roll stop (8.49 mm; a
  -17 deg stop clears by 4.79 mm, -16 deg by 18.1 mm; item 10); the neck cradle's end into the neck yaw servo
  (0.44 mm). F21 says no tool checks the hind legs: `sim/check_packaging60.py` now does, for all four, and every pair
  of legs against each other. The pushrod sizing (section 2) as a finding with its sources, the hind static case at
  the X4's 36 N m stall torque (1,885 N); D47's row can point at the parts list.
- **README.md**: add `design/PACKAGING_PUSHROD_60CM.md` and `sim/check_packaging60.py`; the landed figures in "For
  the owner" ($20.1k-$20.9k) rise by $647 with the battery cut-off (`design/PARTS_60CM.md`); line 75's "0.5 kg over
  the budget's mass" is now 0.55 kg (0.85 kg with the cut-off's estimated 0.25 kg listed, 0.30 kg grown by the closed
  variant's x 1.2088 like the model's other estimates).
- **DECISION_LOG.md (F10) (F10)** gives 11.54 kg listed and 12.69 kg closed with `design/PARTS_60CM.md` as its source;
  the regenerated list shows the adopted model, 11.624 / 12.751 kg (13.053 kg closed with the cut-off). F10 describes
  the 188 mm model, so its source becomes the list as of aa150f89 (`git show aa150f89:hardware/triceratops_build/
  design/PARTS_60CM.md`, line 21), or the row gains the adopted figures.
- **The readiness review** (`READINESS_REVIEW_2026_11_27.md`) cites `design/PARTS_60CM.md` by line number as it stood
  at aa150f89; the regenerated list moves every one of those lines. By content, at aa150f89 then now: the
  "nothing here is quoted, bought, weighed or measured" line :3 then :3; the 12.694 kg total :21 then the 12.751 kg
  total at :21 (the review's sentence is about aa150f89 and stays right if it says so); the $90 blowers line :118 then
  :156; the frame line with the belly skid :120 then :158; the chosen-build cost paragraph :128 then :179 (now with the
  cut-off: $17,316, about $20,752 landed); the "Mass and the hind knees" gap (X6 knees +$318, +0.92 kg) :133 then
  :187; "weigh the first parts" :134 then :188; the TTL servos and RS-485 converter :136 then :190. Citing by section
  ("Known gaps", "Cost") would survive the next regeneration. These line numbers are for the list as written today.
- **The budget script** (`budget/size_budget.py`) does not carry the battery cut-off: its "Chosen build" (hardware
  $16,776, $20,105 landed, `budget/size_budget_out.txt:97`) is now below the parts list's ($17,316, $20,752,
  `design/PARTS_60CM.md:179`), which says so. Whether the cut-off's two lines move into `size_budget.py` (and its
  output and the README's figures with it) is the lead's and the owner's call; this package does not change the
  budget script, which the parts list reads but does not feed.
- **Bench plan** (to be rewritten for 128 mm anyway): in B6 a proof load of the linkage at the X6's peak torque at
  -47 deg (3.1 kN in the rod) and the crank-pin and output-bearing checks; a hind-linkage clearance check at inward
  roll; the 0.75 mm axle gap; in B8 a BMS trip input.
- **FINAL step sheets:** no change needed: nothing here changes the model. Applying any fix above, or adding the
  cut-off's mass to the model, does, and would need new sheets.

## 4. Left open

- Chassis CAD (D5): the one layout found whose moved parts clear every solid part over the legs', tail's and head's
  ranges is combined layout C (shortened, inset rails; a drive bracket at each roll drive; both crossmembers on the
  rails' ends; the DC-DC block and I/O carrier moved 20 mm; the tray stays re-angled to the rails), and it still needs the ToF arms re-rooted, a tail mount to its
  pivot 34.6 mm aft, rib 4 relieved and the looms re-routed. The single fixes (3, the tail servo move) hold only on
  today's rails. The bay's tightest gaps have room to open (section 1, table) but no record sizes them.
- The hind knee repackaging (fork or offset), then a regenerated model and this check rerun.
- The rod ends' and pins' maker sheets (dimensions, C, C0; these come from search summaries that disagree for M5), the
  X6 and X4 output-bearing ratings, and the linkage's price for the sizing (the budget's "4 pushrod linkages" $320 is
  unchanged).
- A walking torque spectrum (a rainflow count over stored torque traces) to replace the F1-F3 bounds: the gait
  reports keep RMS and peak only.
- Whether any gait uses the poses where the clashes sit: inward hind roll with the leg pitched forward (the rails);
  the fore leg pitched aft and the hind leg forward, the hind knee flexed and the two legs' rolls within 21 deg of each
  other (item 9); both hind legs at the inward roll stop
  (item 10). No gait report records the roll, pitch or flexion angles (only ankle percentiles). Any stop chosen from items 9 and 10 changes the training model's
  ranges, which the FINAL step sheets pin.

## Sources

Records: `sim/generated/trike60_C_closed_capsule_fore128.xml`; `sim/data/drives.json`; `DECISION_LOG.md (F19), :65`
(F19, F21); `rl/analysis/PLAN_FORELEG_D22.md:37-41`; `git show 2ac57608:hardware/triceratops_build/REVIEW_60CM.md`
(lines 29, 113, 116-123, 118-122 and 158 for the bay, 164); `git show 65a44fad:hardware/triceratops_build/bench/BENCH_PLAN_FORELEG.md` (lines
10-11); `hardware/triceratops_kickoff/PARTS_PREMIUM.md:253, :256`; `rl/analytics/results/fore128-s{0..3}/
gait_report.json`.

Web-search summaries, read 27 Nov 2026 (pages not opened; typical values, "not for design" on the aluminium sheets):
- AISI 1018 cold drawn, ultimate 440 MPa, yield 370 MPa, E 200 GPa: https://matweb.com/search/datasheettext.aspx?matid=6817
- Property class 8.8, nominal tensile 800 MPa, proof stress 580 MPa: https://www.trfastenings.com/knowledge-base/metric-screws/screws-and-bolts-mechanical-properties
- 6061-T6, ultimate 310, yield 276, fatigue 96.5 MPa at 5 x 10^8 reversed, E 68.9 GPa: https://www.aerospacemetals.com/wp-content/uploads/2023/06/Aluminum-6061-T6-6061-T651.pdf
- 7075-T6, ultimate 572, yield 503, fatigue 159 MPa at 5 x 10^8 reversed, E 71.7 GPa: https://quickparts.com/wp-content/uploads/2024/05/Aluminum-7075.pdf (Wikipedia's range is lower, 510-540 / 430-480 MPa: https://www.wikipedia.org/wiki/7075_aluminum)
- Thread tensile stress areas M5 14.2, M6 20.1, M8 36.6 mm^2: https://www.engineeringtoolbox.com/metric-bolts-minimum-ultimate-tensile-proof-loads-d_2026.html, https://roymech.org/Useful_Tables/Screws/Thread_Calcs.html
- Thread fatigue factors (rolled 2.2 / cut 2.8 up to class 5.8; 3.0 / 3.8 from 8.8): https://danotes.mech.uwa.edu.au/threads/fatigue/fatigue.html
- Steel endurance 0.5 x ultimate, axial load factor 0.70: https://www.engineering.iastate.edu/~gkstarns/me325/lecture24.pdf
- Marin surface factor a x Su^b (machined a 4.51 MPa, b -0.265; ground 1.58, -0.085): https://engineersedge.com/calculators/marin_endurance_limit_15637.htm, https://fchartsoftware.com/ees/mechanical_design/endurance/hs101.htm
- Steel/steel rod ends to DIN ISO 12240-4: M6 C 3.6 kN, C0 8.2 kN (https://www.marek.eu/data/katalogove-listy/kipp/K0722.pdf, https://uk.misumi-ec.com/vona2/detail/221303940723); M8 C 5.8 kN, C0 12.9 kN (norelem 27631-08, https://norelem.de/en/Product-overview/Systems-and-components-for-machine-and-plant-construction/27000/Rod-ends/Rod-ends-with-plain-bearing-external-thread-narrow-version-DIN-ISO-12240-4/Rod-End-with-plain-bearing-steel-galvanised/p/27631-08; the figures are from a web-search summary of 27 Nov 2026, and a second search that day could not read them off the page, which is outside this session's network: confirm on the maker's sheet. The lowest M8 steel static rating a second search found, 7.9 kN (RS PRO, https://uk.rs-online.com/web/p/rod-ends/1989083, bearing pairing not stated), still gives SF 2.51 against the 3,142 N peak); M5 dynamic 2.5-7.5 and static 3.4-6.2 kN across five listings (https://halder.com/PM/Standard-Parts/Machine-and-Fixture-Elements/Rod-Ends/Rod-Ends-DIN-12240-4-with-male-thread, https://uk.rs-online.com/web/p/rod-ends/1844466 and others), the lowest used. Ball-bearing rod ends (static about 1 kN) and polymer ones are excluded.

## Reproducing

```sh
python -m pip install mujoco==3.10.0 numpy scipy   # scipy: the check's local search (section 6)
python hardware/triceratops_build/sim/check_packaging60.py --json hardware/triceratops_build/sim/results/packaging60_2026-11-27/packaging60.json > hardware/triceratops_build/sim/results/packaging60_2026-11-27/packaging60_report.txt   # about 3 min
python hardware/triceratops_build/design/pushrod60.py
python -m pytest hardware/triceratops_build/sim/tests/test_check_packaging60.py hardware/triceratops_build/design/tests/test_parts60_pushrod60.py
```
