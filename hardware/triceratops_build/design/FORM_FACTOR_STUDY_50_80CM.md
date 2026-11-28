# Form-factor study, 50-80 cm: is there a Triceratops configuration that will likely work?

28 Nov 2026, branch `claude/pensive-pasteur-o3bqtv`; revised the same day after an independent review (findings
PH-1 to PH-8, CTRL-1 to CTRL-9, MECH-1 to MECH-8, F1 to F8) and again after a third review (R3-1 to R3-4).
**Desk and simulation work only.** Nothing is built,
bought, weighed or measured, and nothing here is adopted. Every option other than the drawn 60 cm design needs the
owner's decision. The only scores are desk scores on the criteria the lead fixed on 27 Nov 2026, before any
configuration was scored (section 1.1). They are desk results on simulated torques, not measurements.

- **Size** is the height to the top of the frill. Today's design is 60 cm to the frill crown, 1.2 m long, hips 0.32 m
  up, 12.75 kg in simulation, with 10 MyActuator X4 and 2 X6 drives on a 48 V (12S) bus
  (`sim/generated/trike60_C_closed_capsule_fore128.xml`).
- **"Work"** means it stands and walks on a flat indoor floor at 0.15-0.25 m/s, turns, survives a moderate shove,
  keeps its drives within their heat limits through the demo loop in a 35 C room, can be stopped safely, can be built
  and packaged, and stays near the $20k landed, all-in budget (decisions D2 and D12, `DECISION_LOG.md` (D2), `:108`).
- **The demo loop** is 90 s of walking at 0.25 m/s and 30 s of standing, repeated, with no turns, starts or stops
  (`rl/demo_course.py:3-20`).

## The answer

**No configuration in 50-80 cm passes the pre-registered desk criteria as written.** C4 (packaging) fails at every
size and with every drive set and layout: the criterion asks the layout to remove the 60 cm design's known clashes,
and no layout here removes clash #9, the front and hind legs of a side meeting under the body (soles overlapping by
40 mm at 50 cm and 48 mm at 60 cm in the study's geometry). That clash is a joint-range problem, not a size problem;
its fix (a range stop or a coupled software limit) narrows ranges the gaits use and has not been checked against any
gait.

**On a supporting reading that leaves #9 and the neck clash out (chosen after seeing that nothing removes #9, so it is
not the criterion), the strongest candidate is still the small end: 50 cm with today's drives in layout L1.** It is
also the only candidate whose torque scaling rule was checked in simulation for its drive set and layout. But that
check, and the candidate's margins, rest on several things that are not yet true:

| Size, drives, layout | Pre-registered result | Supporting reading (C4 without #9 and #11) | Torque rule checked in simulation? | Mass | Landed, 0 % duty, cut-off included |
|---|---|---|---|---|---|
| **50 cm, today's drives, L1** | Fails C4 as written (#9) | **Passes.** Tightest is C2, continuous torque: 0.705 of rating against the 0.80 line (12 % headroom); hottest winding 69.4 C (desk thermal model on simulated torques) | **Yes**, as built and in L1, but only with the leg servo gains rescaled to size (a change needing the owner's decision); with the adopted gains the check fails, even on its normal-gait seeds (sections 1.3, 2.2) | 11.8 kg | **$19.1k** (under $20k only at 0 % duty) |
| 55 cm, today's drives, L1 | Fails C4 as written | Fails: C2 0.815, and the two hind feet clear each other by only 2.4 mm at the inward roll stop | **Mixed**, as built (L1 not run): fails with the gains rescaled (the trot's slow gait on 2 of 3 judged seeds), holds with the adopted gains. Three seeds cannot tell 55 from 50 cm on this check (section 1.3) | 12.4 kg | $19.9k |
| **60 cm, as drawn** | **Fails C1** (98.3 C, desk thermal model on simulated torques), **C2** (0.914) and C4 (cannot be built as drawn) | - | Control case | 12.75 kg | $20.1k ($20.8k with the cut-off) |
| 60 cm, today's drives, L1 | Fails C1 (102.6 C), C2 (0.938), C4 | Fails: C1, C2, and the hind feet overlap 8.5 mm at the inward roll stop (record clash #10) | Yes | 13.1 kg | $20.8k |
| 50-65 cm, bigger drives (X6 or X8 sets) | Fail C4 as written | 2 pass, both with X6 knees in L1: 50 cm D1 and 55 cm D2 (and 50 cm with today's drives in L2). No set with X8 drives passes, nor any bigger set in L2: drawn at its SF 2.0 size, each knee axle is cleared by its rod by less than 5 mm (section 1.2) | **No.** The check fails for every drive set other than today's (the trot falls into its slow gait on 10 of 10 seeds), and in layout L2 for every set | 13.1-15.6 kg | $19.4-21.0k |
| 70-80 cm | Fail | Nothing passes | - | 14.9-30.7 kg | $24.1-32.7k |

**So will a 50-80 cm configuration work at some point? Probably, but nothing here shows it yet.** The reasons to
expect it:
- At the small end, today's drives have heat margin on today's policies (50 cm: 69.4 C, 0.705), and the cheapest
  design is under budget.
- At 60 cm the heat failure is not settled either. It comes from one policy's one-sided load: policy s3's left hind hip
  pitch runs at 0.914 of rating while its mirror runs at 0.509. Shared evenly, the worst of the four policies would be
  at 0.740 (73.4 C), under the line (desk arithmetic, not a pass). It also rests on the build's rating convention. On
  the X4's own 10-30 rpm maker rows the same profile gives 79.2-84.2 C and 0.806-0.857.
- Quadrupeds of 9-15 kg walk on learned policies on real floors (section 2.5).

The reasons for caution:
- Heat is not the only problem that size changes. Several problems get worse as the robot gets smaller (section 2.4):
  - joint speed: the 50 cm hip pitch needs 1.10 of the X4's 83 rpm cap on policy s3, and the simulation does not
    enforce the cap;
  - the bay: at 50 cm the two battery packs do not fit the scaled bay box;
  - delay: a 20 ms delay is relatively longer for a smaller robot;
  - shoves: the same push is a larger velocity change for a smaller robot;
  - the scripted trot's slow, hot gait becomes more common as the robot shrinks.
- Today's controllers have not shown turning or a moderate shove at any size.
- No precedent has this body's pitch and yaw inertia for its size: about twice the Unitree robots' (section 2.5).

**Most of what decides the answer now is controller and packaging work that costs nothing**, and it should come
before any size or drive choice:
1. **Train or select one 60 cm policy for heat** and score C1 and C2 again. Use the steering run's planned price on the
   hottest drive (D26, D29, `DECISION_LOG.md` (D26), `:125`) or a left-right symmetry term.
2. **Retrain one policy on a 50 cm scaled model** with the adopted servo gains unchanged and the 83 rpm cap enforced,
   then score it again.
3. **Sweep the record's stops for clash #9**, and an inward roll stop for #10, against the joint angles the gaits
   actually use.

A bench heat test of one X4 (section 3.2) is the cheapest purchase that would most change confidence, but buying it
is the owner's decision.

**How sure can we be? Not very, and only on paper.**
- **What holds up.**
  - Heat rises with size on today's policies and drives. The desk C2 for today's drives in L1 is 0.705 at 50 cm,
    0.938 at 60 cm and 1.525 at 80 cm. On the scaled trot (simulation, normal gait), the worst drive runs at 0.55 of
    rating at 50 cm, 0.70 at 60 cm and 1.12 at 80 cm.
  - The scaling rule predicted the trot's joint torques within ±25 % for today's drives as built at 50, 65, 70 and
    80 cm, and in layout L1 at 50 and 60 cm, whenever the servo gains were rescaled and the gait stayed in its normal
    mode.
- **What is thin.**
  - The 50 cm margin is 12 % on C2.
  - It shrinks or vanishes under each of these (desk arithmetic, section 4):
    - the rule under-predicting the hind hip pitch by 3-14 %: about 0.73-0.80;
    - the adopted servo gains (+14 to +33 % at that joint on the trot): about 0.80-0.94;
    - walking at 0.25 m/s instead of the Froude-matched 0.228 m/s: 0.733;
    - turns (not in the loop): illustratively about 0.78;
    - the knee axles' steel, which the mass model leaves out: about 0.72;
    - a 20 ms delay: at least 0.90.
  - The other way, on the X4's 10-30 rpm maker rows it is 0.62-0.66.
  - No heat figure is measured. The policies read 11 simulator-only inputs and run an ideal, frictionless servo, which
    is probably why they run cooler than a robot would.

## 1. The answer in numbers

### 1.1 The pre-registered criteria (verbatim, fixed by the lead on 27 Nov 2026)

> PRE-REGISTERED DESK CRITERIA (fixed by the lead on 27 Nov 2026, before any configuration was scored; do not change
> them; if one cannot be computed as written, say so and report what was computed instead, labelled "changed after a
> result" only if it was changed after seeing numbers):
> A configuration = a size (frill height 0.50, 0.55, 0.60, 0.65, 0.70, 0.75 or 0.80 m), a drive set (which drive at
> each of the 12 leg joints; head and tail as the build), a leg layout (where the knee drive sits: at the knee as now,
> or at the hip/shoulder driving the knee through a pushrod or belt), and a cooling assumption (passive as now, or a
> stated forced-air case with its source).
> - C1 Heat: on the demo loop's torque profile scaled to the configuration, every drive's hottest winding stays at or
>   below 90 C in a 35 C room (the bench's B3 rule, D48), using the build's thermal model (rl/telemetry/thermal.py and
>   the ambient option rl/analysis/thermal_ambient.py), for the worst of the four 128 mm policies' profiles. "With
>   margin" means at or below 80 C.
> - C2 Continuous torque: every drive's RMS torque over the demo loop is at most 0.8 of its continuous rating, worst
>   policy.
> - C3 Peak torque: every drive's peak torque, scaled from the simulated trot and push peaks, is at most 0.8 of its
>   peak rating.
> - C4 Packaging: each leg's drives and linkages fit the configuration's body and leg envelope with at least 5 mm
>   clearance over the full joint ranges, in a stated desk layout checked by geometry code (a cylinder-and-box check is
>   enough), and the layout removes the 60 cm design's known clashes.
> - C5 Battery: at least 30 minutes of the demo loop on one pack at 80 % depth of discharge, at the configuration's
>   mass.
> - C6 Cost: landed cost under the parts list's landed rule (budget/size_budget.py), reported against the $20k
>   ceiling; over-budget configurations are listed, not dropped.
> - C7 Existence-proof margin (supporting, not a pass/fail): the dimensionless continuous-torque margin of each leg
>   joint, tau_cont / (m g l) with l the joint's lever (hip to foot for hip pitch, knee to foot for the knee), compared
>   with the same figure for quadrupeds with published sim-to-real RL walking; report where the configuration sits in
>   that range.
> "Passes the desk criteria" = C1 to C5 all pass for the worst policy profile; C6 and C7 are reported beside it. The
> torque scaling rule (how joint torques at 60 cm map to another size and mass) must be stated, justified, and checked
> in simulation (the Simulate phase); report the prediction error.

**What was scored on them.** `design/form_factor_study.py` scored all 126 configurations (7 sizes × 9 drive sets ×
2 layouts, passive cooling) into `design/form_factor_out.json`: DESK on SIMULATED 60 cm torques. These are the only
results here scored on the criteria. The simulation checks S1 and S2 (section 2.2) test the scaling rule; they score
no configuration.

**Where a criterion could not be computed as written, and what was computed instead.**
- **C3.** Every stored and probed simulated peak sits at the model's force limit (0.70 × the maker's peak), so no run
  shows the peak a gait needs. The scored value is that clamp × k over the new drive's peak. For today's drives this is
  0.70 k at every joint, so C3 passes exactly when k ≤ 1.143: it adds nothing that k does not already say. A 60 cm
  probe (`sim/results/form_factor_2026-11-27/force_limit_probe60.json`) had no falls at 1.0, 0.7 and 0.5 × the limit,
  but with **0.5 m/s pushes only**, half the push at which the learned policies already fall (section 2.4). So the
  "× 0.5" reading bounds walking with a small nudge, not a moderate shove, and is used here to support no
  configuration. **The peak torque a moderate shove needs is unknown at every size.**
- **C4 could be computed as written, and it fails everywhere.** No layout removes known clash #9, and clash #11 (the
  neck) is not modelled. The first version of this study left #9 and the neck out of C4 and headlined passes on that
  narrower reading; that choice was made after seeing that no layout removes #9, so here it is labelled **"changed
  after a result"** and kept only as a supporting reading, out of the pass rule.
  - Other choices made after seeing geometry numbers (the criteria unchanged): the crossmember ends, the 28 mm crank
    floor, the tail-servo and knee-drive placement searches, adding set D8, ranking C4 on the next-tightest pair (the
    5.5 mm rod-to-drive gap is fixed by construction), and, after review, the legs swept against each other, the
    rods sized for their drives, and the 5 mm wiring allowance in the bay check; after the third review, each knee
    axle drawn at its own SF 2.0 diameter (R3-1).
  - The criterion's "body and leg envelope" was never defined in numbers. Widths are reported instead: L1 keeps the
    50 cm robot 0.35 m across the legs, and L2 makes it 0.53-0.60 m, against 0.31 m for the drawn 60 cm design scaled
    to 50 cm.
- **C1** is scored at the build's 8 min thermal time constant (4 and 15 min in the JSON). **Cooling** is passive in
  every row: no source found gives a forced-air thermal resistance.

**Drive sets.** D0 is today's: 10 X4-36-L plus 2 X6-60 at the fore knees. X6 sets: D1 (both knees), D2 (hip pitch
and knees), D5 (everywhere). X8-150 sets: D3 (hip pitch and knees), D4 (hip pitch, X6 knees), D6 (X6 roll, X8 hip
pitch and knees), D8 (everywhere). D7, RobStride RS03 at the hip pitch and knees, is a reference only: its ratings
assume a heat sink, so it is never counted as a pass.

**Layouts** (both change the drawn and training models, so both **need the owner's decision**). L1: the knee drive on
the upper link, offset from its strut, as the 128 mm front leg already does (`design/PACKAGING_PUSHROD_60CM.md:33`,
row 1's fix). L2: the knee drive at the hip, driving the knee through a 1:1 parallelogram. **L2 is not shown to be
buildable**: its rod and crank run 77-100 mm outboard of the link plane. On the model's 12.8 mm axle that would
stress the axle to 681-1,234 MPa at the static rod case for today's drives, against a 580 MPa proof stress, so the
layout draws each axle at its SF 2.0 diameter instead (15.5-20.8 mm for today's drives, 26-29 mm for X8 knees). Only
today's drives' rods clear those axles by 5 mm in L2 at 50-60 cm (section 1.2). The rod's offset also twists the
thigh with about 279 N m at the front knee's static case, twice L1's 137 N m (desk). It needs a forked or box thigh
and an outboard bearing that are neither drawn nor weighed.

### 1.2 Configurations that pass the supporting reading (C1-C3, C5, and C4 without #9 and #11), worst policy

**None of these passes the criteria as written.** Ranked by the smallest C1-C3 margin. Columns:
- C1: hottest winding (8 min).
- C2: loop RMS over the continuous rating.
- C3: scaled clamp over the maker's peak.
- C5: minutes per pack, with the pessimistic bound in brackets (section 4).
- Landed: $k at 0 % / 41.5 % duty, then 0 % with the D47 cut-off.
- Rule check: whether the torque rule held in simulation (S2) for this drive set, as built, and in this layout where it
  was run.
- Axles: each knee axle as drawn, fore / hind, at the larger of the model's 12.8 mm and its own SF 2.0 diameter, with
  the least rod-to-axle clearance over the joint ranges in brackets (desk geometry; scratch
  `r3/probe/axle_c4_out.json`).

The masses carry the structure sized for the load it holds (section 2.1), but not the axles' added steel (below).

| # | Size | Drives | Layout | kg | C1 C | C2 | C3 | C5 min | Landed $k | Rule check (S2) | Axles mm (rod clearance mm) | Also |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| **1** | **0.50** | **D0** | **L1** | **11.8** | **69.4** | **0.705** | **0.54** | **113 (79)** | **18.4 / 20.5 / 19.1** | **holds as built and in L1 (gains rescaled)** | **15.8 / 12.9 (10.3 / 8.1)** | **bay not shown; hip pitch 1.10 of the speed cap** |
| 2 | 0.50 | D0 | L2 | 11.9 | 69.8 | 0.709 | 0.54 | 112 (78) | 18.4 / 20.5 / 19.1 | fails in L2 | 20.7 / 17.0 (7.8 / 6.1) | L2; bay not shown |
| 3 | 0.50 | D1 | L1 | 13.1 | 78.2 | 0.779 | 0.60 | 104 (72) | 18.8 / 21.0 / 19.4 | fails for D1 | 15.8 / 15.8 (10.3 / 6.7) | bay not shown |
| 4 | 0.55 | D2 | L1 | 15.6 | 70.0 | 0.708 | 0.79 | 80 (56) | 20.3 / 22.8 / 21.0 | fails for D2 | 15.7 / 15.7 (10.3 / 6.8) | bay only with no wiring allowance |

**Reading the table.**
- **The second version of this table had 22 rows; 18 are gone (R3-1).** It drew every knee axle at the model's
  12.8 mm, which the study's own axle check puts at 1,234-3,304 MPa in the static rod case in those rows for every
  X8 knee and for every X6 knee in L2, against a 580 MPa proof stress. Drawn at its own SF 2.0 diameter, the axle is
  cleared by its rod by less than 5 mm at full knee flexion:
  - every X8 knee (D3, D6, D8) at 50-65 cm: 20.9-22.0 mm axles in L1, cleared by 2.6-4.7 mm; 27.9-28.8 mm in L2,
    -1.8 to 0.2 mm (an overlap at 50-60 cm);
  - X6 knees in L2 at 50-60 cm: 20.7-20.9 mm axles, cleared by 3.2-4.2 mm (by 5.0 mm at 65 cm, where those rows fail
    C1-C3).
  - Today's drives keep it in L2 (row 2) because the hind knee is an X4: its 17.0 mm axle is cleared by 6.1 mm.
  - An outboard bearing or fork, a larger crank or a stronger axle steel could bring some rows back. None is drawn,
    sized or sourced here, and each changes the drawn model (the owner's decision).
- **Only row 1 has its torque rule checked in simulation.** Every other row's C1 and C2 come from the rule on a drive
  set or layout where the simulation check failed.
  - Bigger drives: the trot fell into its slow gait on all 10 seeds for every X6 and X8 set, at 50-65 cm.
  - Layout L2: the roll torques ran 17-140 % over the rule at 50 cm and 5-52 % over it at 60 cm (section 2.2).
  - So rows 2-4 are **scaling rule unchecked; the simulation check fails with this trot**. They are listed, not
    recommended.
- **C3 depends on the X6's peak.** The stand-in X6-50-L's peak is 45-50 N m against the X6-60's 60 N m. At 45 N m,
  the fore-knee C3 for today's drives becomes 0.72 at 50 cm, 0.83 at 55 cm and 0.96 at 60 cm, and the two X6 rows
  reach 0.80 (row 3) and 1.05 (row 4).
- **The X8 knees would also need M10 rods** (M12 at the longer L2 hind rods). The packaging record's M8 rod fails the
  record's own rules against the X8's 150 N m stall torque, which gives 7,855 N: its rod end reaches SF 1.64 against
  2.0. The geometry draws the sized rod and moves its lane outboard; the X8 rows fail on the axle all the same.
- **The axles' steel is not in the mass model** (desk arithmetic, not rescored; scratch `r3/probe/axle_mass_out.json`).
  The ledger carries 4-8 g per knee axle, the model's short stub. Solid steel axles of the drawn diameter and length
  (about 50-100 mm, through the output crank's hub) would add 0.26 kg to row 1, 0.83 kg to row 2, 0.32 kg to row 3 and
  0.31 kg to row 4. With k scaled by that mass, C2 becomes about 0.72, 0.76, 0.80 and 0.72, and C1 about 70.9, 74.8,
  80.4 and 71.4 C. Row 3 then sits at the C2 line. About 0.20 kg of row 1's 0.26 kg is the 12.8 mm axle lengthened
  through the hub, which the earlier versions also left out; a hollow axle, or the strut's own bore, would take some
  of it back.
- **Under $20k:** rows 1-3 at 0 % duty with the cut-off ($19.1-19.4k); none at 41.5 % duty. Only row 1 has its rule
  checked.
- **The rating convention moves the margins both ways.**
  - With every drive rated at its rated speed, a pessimistic case for the X6 and X8 (X6 9 N m, X8 20 N m), rows 1-3
    still pass; row 4 does not (104.1 C, C2 0.944).
  - With the X4 rated on its own 10 or 30 rpm maker row (8 or 8.5 N m, refitted to the row's own temperature rise),
    row 1 moves to C2 0.661 / 0.622 and C1 62.3 / 59.7 C.
- **RobStride reference** (D7): not comparable (heat-sink ratings; a maker change needs the owner's decision, P3,
  `DECISION_LOG.md` (P3)).

### 1.3 The best at each size, and what binds

Today's drives in layout L1 (DESK on SIMULATED torques; C4 here is the supporting reading):

| Size | kg | C1 C | C2 | C3 | Hind feet left to right, mm | Bay fits? | Hip-pitch speed / 83 rpm cap | Landed $k 0 % (+cut-off) | Fails, supporting reading | Cheapest supporting pass at this size |
|---|---|---|---|---|---|---|---|---|---|---|
| 0.50 | 11.8 | 69.4 | 0.705 | 0.54 | +13.2 | no (the packs) | 1.10 | 18.4 (19.1) | none | D0 L1 |
| 0.55 | 12.4 | 83.0 | 0.815 | 0.62 | +2.4 | only with no allowance | 1.05 | 19.2 (19.9) | C2, C4 (#10) | D2 L1, $21.0k (C3 0.79; rule unchecked) |
| 0.60 | 13.1 | 102.6 | 0.938 | 0.72 | -8.5 | yes | 1.01 | 20.1 (20.8) | C1, C2, C4 (#10) | none |
| 0.65 | 13.9 | 127.3 | 1.059 | 0.82 | -9.2 | yes | 0.97 | 21.0 (21.7) | C1-C4 | none |
| 0.70 | 14.9 | 167.5 | 1.206 | 0.95 | -9.9 | yes | 0.93 | 23.6 (24.3) | C1-C4 | none |
| 0.75 | 15.9 | 224.2 | 1.357 | 1.09 | -10.6 | yes | 0.90 | 24.7 (25.4) | C1-C4 | none |
| 0.80 | 17.0 | 312.0 | 1.525 | 1.24 | -11.3 | yes | 0.87 | 25.9 (26.5) | C1-C4 | none |

**Why each criterion binds where it does.**
- **C2, for today's drives.** The deciding drive is a hip pitch X4 on policy s3 (left hind) or s2 (left fore). It
  scales with k = (m / 12.751 kg) × (S / 0.60 m), which is 0.77 at 50 cm and 0.89 at 55 cm.
  - **The deciding load is one-sided**, a quirk of today's policies (section 1.4). On the same policy, its mirror drive
    runs at 0.39-0.52 of rating across 50-60 cm.
  - So C2 varies between policies at one size about as much as across sizes. At 60 cm the best-to-worst policy ratio is
    0.82 (0.746 against 0.914), and going from 60 to 50 cm multiplies C2 by 0.75.
  - **"Smaller is cooler" is true for today's policies**; whether it is needed at all is a controller question.
- **C1 and C2 together from 60 cm up.** Copper heat goes as k², and the steady 90 C line in a 35 C room sits at 0.870
  of the rating.
- **C4.** Clash #9 fails everywhere. Beyond it:
  - the hind feet meet at the inward roll stop from 60 cm up, and clear by only 2.4 mm at 55 cm (record clash #10;
    its fix is an inward roll stop at -17 deg, record row 10);
  - in L1, the X6 sets D2, D4 and D5 at 50 cm and D4 at 55 cm clear by only -0.7 to 4.0 mm (the fore knee drive
    against the pitch drive, or the two fore cranks);
  - with each knee axle drawn at its SF 2.0 size, every X8 knee at 50-65 cm (both layouts) and every X6 knee in L2
    at 50-60 cm is cleared by its rod by less than 5 mm (section 1.2), so L2 rescues no bigger drive set.
- **C5 never binds:** at least 56 min per pack on the pessimistic bound in every supporting pass.

**The torque rule at 50 and 55 cm** (SIMULATION, S2; today's drives as built; L1 was run at 50 cm only;
`sim/results/form_factor_2026-11-27/scaled_trike_compare.json`).
- With the servo gains rescaled to size, S2 holds at 50 cm and fails at 55 cm. Three judged seeds cannot tell the two
  sizes apart: the trot's slow gait hit 3 of 10 seeds at 50 cm and 2 of 10 at 55 cm, and the verdict turns on which of
  seeds 101-103 it hit (103 at 50 cm; 101 and 102 at 55 cm).
- With the adopted A2 gains, the gains the robot keeps unless the owner decides otherwise, the order reverses. S2
  holds at 55 cm: the slow gait hit seed 101 only, and seeds 102 and 103 ran +12 to +21 % over the rule at the hip
  pitch and +4 to +12 % at the knees. It fails at 50 cm even on its normal-gait seeds 101 and 103: hind hip pitch +14
  to +28 %, hind knees +41 to +52 %.
- So this check does not favour 50 cm over 55 cm. Its one clean 50 cm result is in L1, with rescaled gains and no slow
  seed.

**Checks beside the criteria at 50 cm, today's drives, L1 (desk).** Each is a condition on the 50 cm result.
- **Speed.** Peak hip-pitch speed is 1.10 of the X4's 83 rpm cap on policy s3 (1.05 at 55 cm, 1.01 at 60 cm). The
  maker's V4.1 sheet caps the speed above 24 V (E1, `DECISION_LOG.md`). The simulation does not enforce the cap, so
  if the cap holds, the 50 cm gait and torques differ from what was scored.
- **Bay.** The bay box is 309 × 105 × 78 mm, and the listed items need 0.75 of its volume bare and 0.99 with a 5 mm
  wiring allowance per dimension. A packer places everything but the two battery packs (`bay_fit`; a greedy packer, so
  this reads "not shown to fit", not "cannot fit"). The packs would have to go outside the bay, or the bay would have
  to grow, which is not designed.
- **Rod and axle.** The M8 rods pass the record's rules at every size with today's drives. On the model's 12.8 mm
  front knee axle the static case gives 546 MPa, SF 1.06 against the 580 MPa proof stress. The layout now draws that
  axle at 15.8 mm (SF 2.0) and the hind one at 12.9 mm, and the rods clear them by 10.3 and 8.1 mm. At 60 cm the same
  axle is at 521 MPa, so it is not a size effect. The axles' steel, which the mass model leaves out, would add about
  0.26 kg and take C2 to about 0.72 (section 1.2, desk).
- **Width.** L1 keeps the robot 0.35 m across the legs.

### 1.4 How the current 60 cm design scores

As drawn: 12.751 kg, today's drives, no layout change, the stored demo loops.

| Criterion | Score | Result |
|---|---|---|
| C1 | Hottest winding 76.3 / 74.4 / 93.3 / **98.3 C** for policies s0-s3 (desk thermal model on simulated torques; `READINESS_REVIEW_2026_11_27.md:195`, reproduced) | **Fail** |
| C2 | 0.761 / 0.746 / 0.881 / **0.914** for s0-s3 | **Fail** |
| C3 | 0.70 as written (the clamp itself) | Pass |
| C4 | 11 solid-part clashes plus a 0.75 mm gap (record, `design/PACKAGING_PUSHROD_60CM.md:33-44`). The largest are 41.1 mm (pelvic crossmember through the hind roll drive) and 38.7 mm (hind knee drive inside its own strut). The study's geometry reproduces 38.70 / 41.10 / 8.19 / 23.10 mm, the hind feet's 8.49 mm (#10) and the soles' 48.0 mm (#9) | **Fail** |
| C5 | 84-96 min per pack | Pass |
| C6 | $20.1k / $22.1k at 0 % / 41.5 % duty; $20.8k / $22.8k with the cut-off | Over $20k |
| C7 | Hip 0.157, knee 0.279 (12.751 kg, levers to the sole); between the two reference points | Reported |

**Each policy's deciding drive and its mirror** (desk arithmetic on the stored simulated torques; loop RMS over the
continuous rating):

| Policy | Deciding drive | Its ratio | Mirror drive | Mirror's ratio | Standing RMS, deciding / mirror, N m | Even split (desk, not a pass) | C1 with the even split |
|---|---|---|---|---|---|---|---|
| s0 | left hind pitch | 0.761 | right hind pitch | 0.632 | 1.87 / 4.21 | 0.700 | 68.8 C |
| s1 | left fore pitch | 0.746 | right fore pitch | 0.585 | 2.05 / 2.97 | 0.682 | 66.8 C |
| s2 | left fore pitch | 0.881 | right fore pitch | 0.544 | 2.60 / 2.89 | 0.732 | 72.6 C |
| s3 | left hind pitch | 0.914 | right hind pitch | 0.509 | 4.12 / 1.05 | 0.740 | 73.4 C |

- **The deciding load is one-sided.**
  - Policy s3 stands with 4.12 N m RMS on its left hind pitch, against 1.05 N m on the right. At zero action the
    model's standing torque there is 0.45 N m (`probes/trike_levers_out.json`, scratch).
  - With each left/right pair given the same total copper loss split evenly, the worst of the four policies would be
    0.740 and 73.4 C, which passes both lines. This is **desk arithmetic, not a pass**: no policy has been trained to do
    it.
  - It does say that a controller change alone could move the 60 cm design's heat by about as much as 10 cm of size.
- **The rating convention.**
  - The build fits each drive so that its equilibrium torque reaches 100 C from 25 C in 90 min (75 K), and rates the
    X4 at its 83 rpm row (7.5 N m).
  - The maker's own X4 rows are 8 N m at 10 rpm (26 to 94 C) and 8.5 N m at 30 rpm (27 to 98 C). The scratch actuator
    fact sheet defines continuous as the lowest of the 10-30 rpm rows (`facts_actuators.md:99`).
  - Walking speeds sit in those rows: the policies' hind-pitch peaks are 56-60 rpm, and the trot's RMS speeds are
    13-18 rpm.
  - Refitted to those rows, the as-drawn design scores 84.2 C and 0.857 (10 rpm), or 79.2 C and 0.806 (30 rpm).
  - **So "today's 60 cm design fails heat" holds on the build's model and convention, as C1 requires.** On the maker's
    10-30 rpm rows it is a narrow C2 miss (0.81-0.86) with C1 passing.
- With the cut-off and the L1 layout (13.09 kg) the design scores 102.6 C and 0.938. C4 then still fails on #9 and on
  #10: the hind soles overlap 8.5 mm at roll -18, pitch +22.4, knee -47, ankle -22 deg, as the record found.

**The knee-drive clash is a layout problem, not a size problem.** The hind knee overlap is the drive's radius plus the
strut's (27.5 + 11.2 mm). It stays at 36.8-42.4 mm at every size from 50 to 80 cm (root-issue map, RI-04).

## 2. What size does and does not change

### 2.1 The torque scaling rule

- **The rule:**
  - k = (m / 12.751 kg) × (S / 0.60 m) on every leg-joint torque, RMS and peak;
  - walking speed min(0.25, 0.25 √(S / 0.60)) m/s (0.228 m/s at 50 cm);
  - joint speed × 1 / √s;
  - copper loss × k² × (Km60 / Km)².
  - **It has no layout term.** Moving the legs outboard (L2) raises the roll torques, which the rule does not see
    (section 2.2).
- **Why:** dynamic similarity at a Froude-matched gait gives torque ∝ m g L, joint speed ∝ √(g / L) and power
  ∝ m g v. It is the budget's own rule (`budget/size_budget.py:55`), normalised to the simulated model.
- **Desk check at 75 cm:** against the standing torques F4 (`DECISION_LOG.md`, F4) it predicts 5.91 / 4.73 N m where
  the simulation gives 5.7 / 4.7 N m (+3.7 % / +0.6 %).
- **Mass model.**
  - The 60 cm ledger is scaled part by part by the model's own rules (`sim/build_model60.py:43-67`):
    - structure × s^2.5;
    - shells and harness × s²;
    - head and tail × s³;
    - battery, electronics and safety fixed;
    - drives at catalog mass.
  - It reproduces the ledger (12.7504 against 12.7506 kg).
  - **Revised after review:** for drive sets heavier than today's, the structure is also scaled by (total / today's
    total at that size)^(2/3), so it carries the extra load at today's stress (DESK; solid sections, M ∝ m g L,
    Z ∝ d³). This adds 0.07-1.69 kg to the X6 and X8 sets (structure × 1.05-1.62) and raises their k by up to 6 %.
    None of the four supporting passes flips. 0.65 m D8 L2, which it dropped out on C3 (0.804), now also fails C4 on
    its sized axle.
  - The 75 cm comparison (16.65 kg against the T15 ledger's 15.00 kg, +11 %) is **not an independent test** of the
    s^2.5 exponent, because the 60 cm ledger was itself derived from the 75 cm parts with that exponent
    (`sim/build_model60.py:57`).
  - The scaled simulation models keep the model's own rule (s^2.5 only).

### 2.2 The simulation check (S1, S2) and its error

**How it was run.**
- `sim/build_scaled_trike.py` builds scaled copies of the adopted model:
  - geometry × s;
  - masses from the study's mass model;
  - drive limits from the drive set;
  - leg servo gains × (m / m60) s, so kp / (m g L) is kept;
  - 10 ms delay, sensor noise, friction 0.8.
- It walks them with the committed `walk/walk60.py` trot, Froude-scaled, not retuned.
- The pre-registered cases were written to `sim/results/form_factor_2026-11-27/scaled_trike_plan.json` before any
  scaled model ran. The cases added after review are in `scaled_trike_plan_addendum.json`, written before they ran.
  They are post hoc and supporting.
- The lead's phase rules, quoted there:
  - **S1**: "each variant stands for 10 s at zero action on a flat floor with every foot above 15 % of weight";
  - **S2**: "the rule holds if every leg joint's RMS is within +-25 % of the prediction on at least 2 of 3 seeds"
    (scripted trot, 3 seeds × 20 s).

**Results (SIMULATION, `scaled_trike_compare.json`, 36 cases).**
- **The control reproduces the source.** The 0.60 m copy matches the source model exactly: compiled arrays, plus
  standing and trot torques at 0.0 % on all 12 joints.
- **S1 holds in every case.** No case fell at zero action, and the lowest foot share is 0.214 of body weight
  against the 0.15 line. The trot fell 0 times in 360 episodes.
- **S2 holds for today's drives, as built, at all three pre-registered sizes** (0.50, 0.70 and 0.80 m) and at 0.65 m
  (supporting). It also holds in layout L1 at 0.50 and 0.60 m (post hoc).
- **S2 fails in 25 supporting cases:**

| Cases | k | S2 | Slow-gait seeds of 10 | Worst trot RMS / continuous | What it says |
|---|---|---|---|---|---|
| 0.55 m, today's drives (pre-planned) | 0.867 | fails | 2 | 0.929 | the slow gait hit 2 of the 3 judged seeds |
| Literal "stiffness per weight" gains, 0.50 / 0.80 m (pre-planned) | 0.749 / 1.737 | fails / fails | 8 / 0 | 0.779 / 1.239 | the gain rule matters |
| X8 everywhere at 0.80 m (pre-planned) | 2.963 | fails | 10 | 0.749 | standing error -24 / +43 % (fore / hind hip pitch) |
| **Every other drive set** (post hoc): D1, D2, D4, D5, D6, D8 at 0.50 m; D2, D6, D8 at 0.55 and 0.60 m; D6, D8 at 0.65 m | 0.82-2.15 | fails, all 14 | 10 in every case | 0.44-0.93 (0.60 m D2: 0.925) | **the rule is unchecked for every bigger drive set**: this trot falls into its slow gait on all of them |
| **Adopted A2 gains, unchanged** (post hoc), 0.50 / 0.55 m | 0.749 / 0.867 | **fails** / holds | 2 / 3 | 0.864 / 0.974 | at 50 cm the hind hip pitch runs 14-33 % over the rule (seeds 101-103) |
| **Layout L2** (post hoc), 0.60 m D0; 0.50 m D0, D2, D6; 0.55 m D6 | 0.75-1.56 | fails, all 5 | 0 | 0.44-0.78 | roll torques 5-52 % over the rule at 60 cm, 17-140 % at 50 cm: **the rule's missing layout term** |
| Layout L1 with X8 knees, 0.60 m D6 (post hoc) | 1.753 | fails | 10 | 0.601 | as the D6 case |

- The scaled models' masses for the drive-set cases use the model's own structure rule. With the scoring's heavier
  structure, k is up to 4 % higher.
- **Tests:** see the end of section 4.

**Prediction error, trot RMS per joint** (simulation ÷ (k × the 0.60 m run on the same seed)):

| Size | Range over the 12 joints | Pattern |
|---|---|---|
| 0.50 m, as built | -19 to +14 % on seeds 101-102; up to +117 % on seed 103, which fell into the slow gait | Predicts too little: **hind hip pitch +3 / +11 % (left), +5 / +14 % (right)** |
| 0.50 m, L1 (post hoc) | -13 to +16 %, no slow seed | hind hip pitch 0 to +14 % |
| 0.50 m, adopted gains (post hoc) | pitch +14 to +56 %, knees +41 to +67 % | **the 50 cm check needs the gains rescaled** |
| 0.70 m | -17 to +14 % | |
| 0.80 m | -21 to +16 % | Predicts too much at the knees (median -14 / -16 %) and elbows (-7 / -9 %); pitch within ±5 % |

**What else the runs show.**
- **Standing torques** are within -15 to +14 % at every size for today's drives as built. They are not within that
  range for X8 everywhere at 0.80 m (-24 / +43 % at the fore and hind hip pitch) or for layout L2 (the roll changes
  sign: -86 to +86 %).
- **Joint speeds** follow 1 / √s within -2 to +5 % at the pitch and knee joints. Not at the roll: +46 to +64 % at
  0.50 m and +124 to +173 % at 0.55 m (where the slow-gait seeds swing the roll), -14 to -15 % on two roll joints at
  0.80 m.
- **Heat on the scaled trot** (today's drives, normal gait, worst RMS over the continuous rating): 0.55 at 0.50 m (0.76
  in the slow gait), 0.70 at 0.60 m, 0.79 at 0.65 m, 0.92 at 0.70 m and 1.12 at 0.80 m.

**Where the rule breaks.**
1. **The trot's second, slow gait becomes more common as the robot shrinks.** Torques in it are 40-117 % higher.
   - As built, with rescaled gains, it hit 3 of 10 seeds at 0.50 m (103, 109, 110) and 2 at 0.55 m (101, 102). A third
     55 cm seed, 106, ran at 0.205 against 0.239 m/s with 3.8-3.9 hind touchdowns per second. It hit 1 seed at
     0.60 m and none at 0.65-0.80 m.
   - At 0.50 m, scaling the delay by √s cut it to 1 of 10. The literal gain rule (about 17 % softer) raised it to 8 of
     10, and the adopted gains to 2 of 10. The L1 geometry had none at 0.50 m.
   - So the slow gait depends on the parts that do not scale: the fixed 10 ms delay and 100 Hz control, and the servo
     stiffness. Its cause is **not established**; it is not "the controller and the capsule foot, not size".
   - **A 30 % rate at 50 cm is a risk to the favoured configuration**, not a controller-only issue. The worst trot
     RMS over the continuous rating at 0.50 m is 0.55 in the normal gait and 0.76 in the slow one.
2. **Peaks.** They sit at the force limit already at 60 cm, so the unclamped need cannot be seen. At 0.50 m the
   normal-gait pitch and hind-knee peaks are 19-29 % above k × the 60 cm peak.
3. **Servo gains.** The rule holds at 50 cm only with the servo stiffness rescaled with m g L. A real 50 cm robot with
   today's drives keeps the adopted A2 gains unless the owner decides otherwise (A2, `DECISION_LOG.md`). With them
   the trot runs 14-33 % over the rule at the deciding joint. At 55 cm the check holds with the adopted gains and
   fails with rescaled ones, the reverse of 50 cm (section 1.3).
4. **Layouts.** L2 moves each leg plane 44-75 mm outboard and raises the roll torques 5-140 % over the rule. Neither
   the rule nor the scores see it.
5. **Bigger drives.** No X6 or X8 set walked normally with this trot as built, so their rows are **not supported** by
   simulation. With L2's wider stance the D2 and D6 sets walked normally, but their roll torques broke the rule.
   Supporting them needs the trot retuned (or a policy trained) for each set's mass and leg inertia, which was not done
   here.

**What this does to the 50 cm result** (desk arithmetic on trot results, not a run of the policies):
- the 50 cm hind-pitch error applied to C2 = 0.705 gives about 0.73-0.80;
- with the adopted gains, about 0.80-0.94.

Only learned policies retrained at 50 cm, with the gains the robot would really have, can settle it.

### 2.3 Issues a configuration choice can remove or reduce (root-issue map, Group A)

From the scratch `root_issues.md`, a first-round map (the claims since withdrawn are listed in its closing note);
DESK or SIM, nothing measured. S is the frill height.

| Issue | How it moves with size | What fixes it |
|---|---|---|
| RI-01 Drive heat in a warm room | Strongly on today's policies: torque about S^1.75, copper heat S^3.5; smaller is better. But one-sided policy loads move it as much (section 1.4) | Policies trained for heat or symmetry (no spend); smaller size; bigger drives on hot axes; a hip-pitch reduction; standing breaks (all but the first need the owner's decision) |
| RI-02 Heat inside the shell | Weak; no model at any size | Ducting, an open exterior; cannot be scored yet |
| RI-03 Mass uncertainty | 47 % of the mass is estimated; nothing is weighed | Weighing parts |
| RI-04 Packaging | The knee-drive overlap is about constant; #9 at every size; the hind feet meet from 60 cm up (#10); the bay does not fit at 50 cm | Layout L1, the #9 stops or a coupled limit, an inward roll stop for #10, a bay re-pack |
| RI-05 Pushrods, axles, output bearings | Rod force at the drive's limit ∝ S^-1 with fixed cranks (held at 28 mm here); the model's 12.8 mm knee axle is under SF 2.0 at every size; X8 knees need M10 rods and 19-29 mm axles, which their rods do not clear by 5 mm at 50-65 cm | Rod ends, clevis pins, a 16 mm axle (today's drives, L1); for X8 knees an outboard bearing or a larger crank (neither drawn); a maker answer |
| RI-06 The X4's 83 rpm cap | Joint speed ∝ S^-0.5 at a Froude speed: 1.10 of the cap at 50 cm, 1.01 at 60 cm | Model the cap in training; bench |
| RI-07 Neck and head servo loads | ∝ S^4; a servo step above about 67.5 cm | Servo choice |
| RI-08 Structure strength | Unchecked; stress ∝ S^-1.25 | Sections and CAD |
| RI-09 The foot | Weak | Rocker design, measured grip |
| RI-10 Cost | About $12.2k fixed + $4.6k × (S / 60)² of hardware | About $0.8k landed per 5 cm |

### 2.4 Issues size does not fix (Group B), and those that get somewhat worse when smaller

**B1: no size fixes them.**

| Issue | Status | Kind of fix |
|---|---|---|
| RI-11 Simulator-only inputs | 11 of 68 policy inputs: true velocity, foot loads, ankle angles (D46, `DECISION_LOG.md`) | Train on own sensors (pre-registered, not run); an estimator; foot cells (+$254, D46) |
| RI-14 Turning | No controller turns yet; turning loads were 1.8-2.3 × the straight learned peaks (SIM; readiness-review claims) | The steering run (configured, untrained) |
| RI-16 Actuator-model fidelity | No friction or backlash in the model; kp 598.5 N·m/rad is an assumption (A2, `DECISION_LOG.md`) | Bench identification, then retrain |
| RI-17 The X4 on 48 V | The maker's V4.1 sheet (read 28 Nov 2026) gives 20-55 V and caps speed at 83 rpm above 24 V; never benched | Bench B1 and B4; a maker answer |
| RI-18 Stop chain, brakeless drives | No catalog brake on any leg drive read; the cut-off is adopted (D47) but not designed; the simulated sit tipped nothing | Sit-then-cut firmware, the BMS in the stop chain, a skid, bench B7 and B8 |
| RI-19 One consumer IMU | Latency and bias unmeasured | A better IMU (+$392) or isolation; an IMU log |
| RI-20 Firmware, test programs | Unwritten; the bench plan is still the 75 cm foreleg's | Write them. A size change adds a model, a packaging check, a parts list and retraining (4.5-5.1 h per judged run on Colab, `rl/PLAN_STEERING_RUN.md:75-76`) |

**B2: no size fixes them, and they get somewhat worse when the robot is smaller** (size factors from the root-issue
map, `root_issues.md:116`, `:119`, `:173`, and the runs above).

| Issue | Status | Size factor | Kind of fix |
|---|---|---|---|
| RI-12 Shove recovery | Learned policies fail ±1.0-1.25 m/s pushes on all four seeds; pushes are trained sideways only | The same velocity change is S^-0.5 harder; the same impulse S^-1.25 | Train pushes both ways; possibly the foot |
| RI-15 Delay, loop placement | Worst drive 0.87-1.02 of rating at 15 ms and 1.07-1.20 at 20 ms (SIM, 60 cm); compute (decision D3) is open | S^-0.5: a real 20 ms is 21.9 ms at 60 cm-equivalent at 50 cm | Loop on the microcontroller; bench B5; owner's decision D3 |
| RI-06 The X4's speed cap | As in Group A | S^-0.5 | Model the cap in training |
| RI-13 Trot's slow, hot mode | Hind knees at 1.02-1.07 of rating; on both capsule-foot models | 3 of 10 seeds at 50 cm, 2 at 55, 1 at 60, none at 65-80 (section 2.2) | Retune the trot for the capsule foot and the small sizes |

### 2.5 What other quadrupeds show (C7 and existence proofs)

Desk research, read 28 Nov 2026; details in the scratch `facts_existence.md`, a first-round sheet whose C7 peak and
heat claims this section replaces (its closing note lists them). MF is a model file opened from GitHub;
WS is a web-search summary only (arxiv.org, unitree.com, bostondynamics.com and ethz.ch were outside this session's
network).

- **Learned walking on real floors is shown at 9-15 kg**: the Mini Cheetah (9 kg; Margolis et al., RSS 2022, WS), the
  A1 and Go1 (12 kg), Barkour vB (about 14.5 kg, MF) and the Go2 (15 kg).
- **Turning is shown for the Mini Cheetah** (spins at 5.7 rad/s indoors and on ice, WS). The Go1 has yaw in its command
  space but no figures; Barkour's weave poles need sharp turns, no figures; A1 and Go2: not confirmed.
- **Pushes: no push size at this mass is sourced.** There is one kick of unknown size: a Go1 recovered from a kick while
  travelling at about 2 m/s, which is the robot's speed, not the kick's (WS; readiness-review claim X8). And one
  qualitative claim: the Mini Cheetah paper says it "responds robustly to disturbances". A1 and Barkour: not
  confirmed. No push, turning or terrain figures were found for the Go2.
- **Head-heavy character robots** of about 15 kg walk on learned policies too, but both are bipeds: BD-X (15.4 kg,
  0.66 m, 4-joint neck) and Olaf (14.9 kg). **No heavy-headed quadruped with learned walking was found.**
- **Body proportions** (SIM, models with meshes stripped, scratch `r2/probe/inertia_probe_out.json`). The whole robot's
  inertia about its centre of mass, over m l² with l the thigh plus shank:
  - pitch: 0.382 for the 60 cm Triceratops against 0.193 (Go1), 0.211 (A1), 0.197 (Go2) and 0.289 (ANYmal B, its
    in-plane thigh);
  - yaw: 0.417 against 0.175, 0.191, 0.185 and 0.268.
  - The moving head is only 0.854 kg (6.7 %) and the tail 0.258 kg (2.0 %), so the difference is the long body.
  - **No precedent has this pitch or yaw inertia for its size.** That bears on fore-aft shoves (RI-12) and turning
    (RI-14) at every size; the 55 % front load adds to it.
- **Heat.** An A1 carrying 3 kg tripped its overheat protection, and Olaf's neck overheated; both were fixed by a
  policy that observes drive temperatures (WS). The build's policies do not observe temperature.
- **C7 continuous** (levers to the sole's centre: knee 0.193 m and hip 0.381 m at the 60 cm hind leg).
  - There are **two reference points, with different definitions**, not a range: ANYmal B (0.090 hip / 0.162 knee, its
    drive's "nominal" 15 N·m, WS, in-plane thigh 0.25 m) and the Mini Cheetah (0.193 / 0.621, 6.9 N·m from a secondary
    source).
  - The 60 cm design is between them (0.157 / 0.279 at 12.751 kg). 50 cm with today's drives is above both at the hip
    (0.204) and between them at the knee (0.362).
  - At the operating rating (35 C room, 90 C limit: the X4 at 6.65 N m) the 50 cm hip is 0.181, between them.
  - C7 goes as 1 / (m l) by construction, so it shows the size trend rather than supporting it.
- **C7 peak** at the model's clamp, on the pre-registered levers:
  - hip pitch: 0.44-0.52 at 60 cm;
  - **hind knee: 0.78 at 60 cm and 1.01 at 50 cm, against 1.42-1.51 for the 9-15 kg comparators** (A1 1.42, Go1 1.42,
    Go2 1.45, Mini Cheetah 1.51; Barkour 0.67). At the maker's peak instead of the clamp the 60 cm hind knee is 1.12,
    still below them.
  - Knee peak torque is what catches a shove. **The design has about half the knee peak margin of the robots it is
    compared with**, which bears on RI-12 at every size.
- **Where the precedents stop.**
  - Most use 6-10:1 quasi-direct drives; the build uses 36:1 and 19.6:1. ANYmal, the geared precedent, needed a learned
    actuator model.
  - No published legged robot runs a learned gait on the X4 or X6.
  - The build's servo stiffness (598.5 N·m/rad) is 10-30 × the 20-60 N·m/rad of published deployments (MF: Walk These
    Ways, Isaac Lab). Whether that matters is untested, and changing it needs the owner's decision.

**So what robots of this mass have shown is limited:**
- shown: learned walking on real floors (RI-11 in part: the Mini Cheetah walks on a gyro and encoders), turning
  (RI-14, the Mini Cheetah), a learned actuator model for geared drives (RI-16, ANYmal) and stop chains (RI-18);
- weakly evidenced: shove recovery (RI-12);
- specific to these drives and this build, so no other robot has solved them: the X4 on 48 V (RI-17) and the firmware
  (RI-20).

Nothing shows the build impossible, and nothing yet shows it possible on these drives with this body.

## 3. A path of iterations

### 3.1 What to change first

| Order | Step | Cost | Owner's decision? |
|---|---|---|---|
| 1 | **Train or select one 60 cm 128 mm policy for heat**: the steering run's price on the hottest drive (D26, D29) or a left-right symmetry term. Score C1 and C2 again, worst policy. Today's policy spread at 60 cm (0.746-0.914; even split 0.740) is as large as 5-10 cm of size | 4.5-5 h of Colab per judged run | A study run only; adopting is the owner's |
| 2 | **Retrain one 128 mm policy on a 50 cm scaled model** with the adopted A2 gains unchanged and the 83 rpm cap in the model, then score C1 and C2 again on its demo loop. This tests the 12 % margin without the rescaled-gain assumption | 4.5-5 h of Colab per judged run | A study run only |
| 3 | **No-spend simulation work every size needs:** the own-sensor run (RI-11, `rl/analysis/PREREG_OWNSENSORS.md`), pushes trained both ways (RI-12), the steering run (RI-14), and a trot retune for the capsule foot and the small sizes (RI-13) | None | No; adopting results is the owner's |
| 4 | **Packaging sweeps:** the record's #9 candidate stops and its coupled limit, and an inward hind roll stop for #10, each swept and checked against the joint angles the gaits use (the gait reports record none today) and for their torque effect; and a bay re-pack at the chosen size | None | No; adopting a stop changes the model |
| 5 | **The single cheap test (3.2)** | One drive plus the bench | Yes: a purchase |
| 6 | **Choose the size and drive set** from steps 1-5 (below) | | Yes |
| 7 | **CAD of the chosen layout:** the hind-knee offset (L1), the #9 and #10 stops, a 16 mm knee axle, and the bay | | Yes: it changes the drawn and training models |

**Choosing in step 6** (thresholds proposed here as a desk judgement, not pre-registered):
- **Step 1 brings the worst 60 cm policy to C2 at or under 0.80 and C1 at or under 90 C:** keep 60 cm with today's
  drives. It then needs only the layout fixes (L1, #9, #10), $20.8k.
- **Otherwise, step 2 gives C2 at or under 0.75 and the bench confirms the model:** 50 cm with today's drives (D0, L1),
  $19.1k, once its bay is re-packed.
- **Neither:** no drive-set change here is supported by simulation yet. Before any is chosen, the trot (or a policy)
  has to walk that set's scaled model normally and S2 has to hold for it. Two X6 routes pass the supporting reading,
  both in L1: D2 at 55 cm ($21.0k; C3 0.79 on an uncertain X6 peak, 1.05 at 45 N m) and D1 at 50 cm ($19.4k; C2 0.779,
  about 0.80 with the axles' steel). D2 at 50 cm is out: its fore drives clash in L1 (2.7 mm), and in L2 its sized
  hind axle is cleared by only 4.2 mm. **Both X6 fallbacks are unsupported until then.**
- Levers not scored here could close part of the heat gap. Each needs the owner's decision and none has a sourced
  figure: a hip-pitch reduction, standing breaks, a temperature-aware policy input (the A1 and Olaf precedent), forced
  air.

### 3.2 The single cheap test that would most raise or lower confidence

**A heat bench for one X4-36-L at 35 C**, on the one-foreleg bench rig with the X4 in place of the knee drive. The
deciding drive is the X4 at the hip pitch, not the X6 elbow.

**Set-up.** Hold one X4-36-L in an enclosure at 35 C on a 42-49 V supply. Drive it through policy s3's left
hind-pitch demo-loop torque profile, scaled to 60 cm (k = 1.0) and then to 50 cm (k = 0.77), for at least 90 min
each, logging the drive's own winding temperature, current, speed and bus voltage. Run the mirror drive's profile
(0.509 of rating) as well, as a low reference.

**Cost:** $269 for the drive (myactuator.us, read 18 Nov 2026, F9, `DECISION_LOG.md` (F9)) plus the rig in
`bench/BENCH_PLAN_FORELEG.md`; buying is the owner's decision.

**Prediction** (simulated torques, desk thermal model, 8 min), as a range:
- on the 60 cm profile, about 98 C on the build's calibration (83 rpm row, 75 K rise;
  `READINESS_REVIEW_2026_11_27.md:195`), or 79-84 C on the X4's own 10-30 rpm maker rows;
- on the 50 cm profile, about 69 C (build), or 60-62 C on the maker's rows.

**What the result would mean.**
- **Runs cooler than the model** (60 cm profile at about 80-85 C, the maker's rows). The X4 has more margin than
  scored: 60 cm is then a narrow C2 miss on today's worst policy, and step 1's retrained policy decides it. 55 cm
  passes C1 and C2 on those rows (0.72-0.76), though not #10.
- **Matches the model** (60 cm profile within about 5 C of 98 C, the 50 cm profile at or under 75 C, the drive
  running normally at 42-49 V with its speed cap as E1 says). The scores here stand; the choice rests on steps 1-2.
- **Lowers confidence:** the 50 cm profile at 80-90 C. Today's drives then have no margin on today's policies anywhere
  in 50-80 cm. Only a policy that spreads the load (step 1) or a drive change could recover it. A drive change is
  unsupported until a controller walks that set's model (3.1), and the X6 would need its own test, since the V4 X6-60
  has no maker thermal table.
- **Means the plan is wrong:**
  - the 50 cm profile goes over 90 C (a thermal resistance about 1.5-1.6 × the fitted 5.56 K/W, or a lower real
    rating). The X4 is then out for this one-sided profile at the hip pitch. Whether it is out for a policy that
    shares the load evenly (0.74 of rating at 60 cm, desk) is a separate question, which the mirror run answers in
    part;
  - or the X4 derates, cuts out or will not run at 42-49 V (the maker's sheet says it will). Every set with an X4
    (D0-D4) is then out on 12S, and no row of the remaining sets (D5, D6, D8) passes the supporting reading: the
    cheapest, 0.50 m D5 at $20.6k, now fails C4 on its sized axle in L2 and on its fore drives in L1.

The same bench also answers RI-17 (the X4 at 48 V), and in part RI-06 (the speed cap), RI-18 (behaviour when hot) and
RI-16 (friction, gains).

### 3.3 Other results that would mean the plan is wrong

- **Step 1 cannot bring a 60 cm policy under the heat lines, and step 2's 50 cm policy fails heat** (worst demo loop C2
  above 0.80 or C1 above 90 C): today's drives are out on learned policies at both ends of the favoured range.
- **A retune does not remove the trot's slow gait at 50 cm:** the scripted fallback runs hot (0.76) at the size heat
  favours.
- **The own-sensor policy cannot walk, turn and take the moderate shove in simulation at 60 cm:** the blocker is the
  controller, which no configuration here fixes. The inertia and knee-peak gaps (2.5) make the shove the likeliest
  place for this.
- **No stop or coupled limit for #9 leaves the gaits their ranges:** the legs then need a layout change no
  configuration here has.
- **A shell thermal estimate adds 20 C or more at 50 cm** (69.4 → 90 C): the open-air pass is wrong.
- **The bench delay (B5) is near 20 ms:** C2 at least 0.90 at 50 cm by arithmetic; the loop placement (decision D3,
  the owner's) decides.

### 3.4 Options here that need the owner's decision

| Area | Options |
|---|---|
| Size | Any size other than the drawn 60 cm (decision D2) |
| Drives | Any drive set but D0. RobStride (D7) also changes the maker and the protocol shared with the T. rex (P3). |
| Layout | L1 or L2; the #9 and #10 stops; a larger knee axle, an outboard knee bearing or a larger crank |
| Other design changes | A hip-pitch reduction, a different servo stiffness (the 50 cm simulation check assumed one), forced-air cooling, a temperature-aware policy input |
| Existing decisions | Bus voltage (decision D4; 12S kept here), compute and loop placement (decision D3), foot load cells (D46) |
| Spending | Any purchase for the bench test |

## 4. Evidence types and limits

| Kind | What is in it |
|---|---|
| **Measured** | **Nothing.** No part is built, bought or weighed. |
| **Simulation** | The 60 cm demo-loop torques of four learned 128 mm policies (`rl/analysis/results/d22/demo/fore128-s*/demo_course.json`); the force-limit probe; the scaled stand and trot at 0.50-0.80 m (36 cases); the mass-sensitivity runs; the standing and inertia probes of published models |
| **Desk** | Every C1-C7 score; the scaling rule; the mass model; layouts L1 and L2; the rod, axle and bay checks; cost; thermal fits; maker ratings; the root-issue size factors; the even split |
| **Design** | The drawn 60 cm model and the adopted choices it carries |

**The assumptions that matter most, and how wrong they could be.**

| Assumption | How wrong it could be | Effect on the answer |
|---|---|---|
| The X4's continuous rating (7.5 N m, its 83 rpm row) and the build's thermal fit (75 K in 90 min) | The maker's 10-30 rpm rows give 8-8.5 N m at 68-71 K; the test mounting is unstated: ±20 % is plausible until measured | 60 cm as drawn: 98.3 C / 0.914 on the build's model, 79.2-84.2 C / 0.806-0.857 on the maker's rows. 50 cm: 0.705, or 0.622-0.661. The 50 cm result needs a real rating of at least 6.6 N m (0.705 × 7.5 / 0.8). This is the test in 3.2 |
| Today's policies' one-sided loads | A policy trained for heat or symmetry may spread them | At 60 cm the even split is 0.740 against 0.914 (desk arithmetic, not a pass) |
| The scaling rule on learned policies | Under-predicts the deciding joint at 50 cm by 3-14 % on the trot; untested on retrained policies; no layout term | C2 could be 0.73-0.80 at 50 cm; L2 rows unsupported |
| Servo gains rescaled with size in the 50 cm check | With the adopted gains the trot runs 14-33 % over the rule at the deciding joint | C2 about 0.80-0.94 at 50 cm (desk); the rescaling needs the owner's decision |
| Simulator-only inputs (11 of 68) | Policies on the robot's own sensors are likely less precise, so probably hotter | The 50 cm margin has to absorb it |
| The 83 rpm cap, not enforced in simulation | 1.10 of the cap at 50 cm on s3 | A capped gait is likely slower-stepping and higher-torque, so probably hotter; the 50 cm result is conditional on it |
| No turns, starts or stops in the loop | Turning peaks were 1.8-2.3 × the straight-walk peaks; no RMS figure exists | Illustrative only: 20 % of the walking at 1.5 × the RMS raises the walking mean square by 1.25 (0.8 + 0.2 × 1.5²) and the loop RMS by about 1.11, so 0.705 → 0.78 (desk; scratch `r3/probe/turn_mix_out.json`) |
| Walking at the Froude-matched 0.228 m/s at 50 cm | At 0.25 m/s (the 0.15-0.25 m/s line extended, desk) | C2 0.733, C1 72.6 C at 50 cm |
| A 10 ms delay | 15 ms at 50 cm is 16.4 ms at 60 cm-equivalent, where the 60 cm results jump; 20 ms is 21.9 ms | C2 about 0.76-0.90 at 15 ms and at least 0.90 at 20 ms at 50 cm (the stored 60 cm ratios 1.02 / 0.945 and 1.20 / 0.945); removes the 50 cm result at 20 ms |
| Open air, no shell | No shell model exists, so the added rise is unknown | 20.6 C of added rise reaches 90 C at 50 cm; less room at every larger size |
| The actuator model: no friction, stiff kp | Unknown. kp × 0.8 tips the trot into its slow gait; kp × 1.2 puts two seeds over rating (SIM) | Heat and shoves |
| C3 read from the clamp; the X6's peak | The real peak need is unknown; the X6's peak may be 45-50 N m, not 60 | At 45 N m the fore knee's C3 is 0.83 at 55 cm and 0.96 at 60 cm |
| The C4 desk layout, drawn to 5.5 mm | No CAD; #9 is not removed; the bay does not fit at 50 cm; L2 is not drawn as a buildable thigh; each knee axle is a solid cantilever at its SF 2.0 size, with no outboard bearing | Packaging is the most fragile result. A drawn bearing instead of a thicker axle could bring back some X8 or L2 rows, or not |
| Mass, 47 % estimated | The 75 cm check is +11 % against the T15 ledger, and is not independent of the s^2.5 exponent; the drawn knee axles' steel (0.26-0.83 kg in the supporting passes) is not in it | Each 10 % of mass moves k and C2 by about 10 %; the axles alone take the 50 cm C2 to about 0.72 |
| The C5 energy model | Braking energy recovered with no loss, no gearbox loss, copper at its 25 C resistance, a 30 W hotel load | The scored minutes are upper bounds. The pessimistic bound (copper × 1.29, 13 W more mechanical draw at 60 cm scaled as m g v, a 40 W hotel load) gives 79 min at 50 cm and at least 56 min in every supporting pass, so no verdict changes |
| Duty at 0 % or 41.5 %; prices of 18 Nov 2026 | $2.0k between the duty cases at 50 cm | Under $20k only with US-stock drives |
| The X6 rating taken from the X6-50-L's table | No V4 X6-60 table exists | Every X6 row (D1, D2, D5, D6) |

**Tests:** `python -m pytest design/tests/test_form_factor_study.py sim/tests/test_build_scaled_trike.py` (run with
`-p no:cacheprovider` and `PYTHONDONTWRITEBYTECODE=1`) printed `58 passed in 19.38s` (rerun 28 Nov 2026 after the third review's fixes).

## 5. Sources

Each price and specification carries its source and read date. "Repo catalog" is the T. rex study's catalog as restated
in `sim/data/drives.json`; the catalog itself was not opened.

| Item | Figure | Source (read date) |
|---|---|---|
| X4-36-L (RMD-X4-P36-36-E-L, V4.1) | Peak 30 N m, stall 36 N m, rated 10.5 N m, 36:1, 0.36 kg, 55 × 61 mm | `sim/data/drives.json:4-11` (repo catalog, maker sheets read 7 Nov 2026) |
| X4-36-L thermal | 90-min equilibrium 7.5 N m at 83 rpm (26 to 96 C); 8.5 N m at 0 rpm (27 to 98 C), 8 N m at 10 rpm (26 to 94 C), 8.5 N m at 30 rpm (27 to 98 C) | MyActuator V4.1 sheet, myactuator.com (28 Nov 2026); scratch `facts_actuators.md:107` |
| X4-36-L electrical | Kt 1.8 N m/A rms, 0.2 ohm per phase | `rl/analytics/results/fore128-s0/gait_report.json:84-85` (from the repo catalog) |
| X4-36-L speed | 83 rpm cap above 24 V | `DECISION_LOG.md` (E1) (E1); confirmed on the V4.1 sheet (28 Nov 2026) |
| X4-36-L price | $269 | myactuator.us store page (18 Nov 2026), `DECISION_LOG.md` (F9) (F9) |
| X6-60 (RMD-X6-P20-60-E, V4) | Peak 60 N m, rated 20 N m, 19.6:1, 0.82 kg; no stall figure | `sim/data/drives.json:12-20` |
| X6 thermal (stand-in) | 12 N m from the X6-50-L's 0 rpm table (13 at 10 rpm, 9 at 153 rpm); no V4 table | MyActuator V4.1 X6-50-L sheet (28 Nov 2026) |
| X6 peak uncertainty | X6-P20-50-E-L at 20 / 45 N m; the store page shows 50 N m | scratch `facts_actuators.md:54`; F9 (18 Nov 2026) |
| X6-60 electrical | Kt 2.1 N m/A, 0.205 ohm per phase | `gait_report.json:108-109` |
| X6-60 price | $428; the page shows the X6-50-L's 0.78 kg and 50 N m | F9 (18 Nov 2026) |
| X8-150 (RMD-X8-P20-150-E-03, V4.1) | Peak 120, stall 150, rated 43 N m; 1.33 kg; 96 × 71 mm; equilibrium 34 / 38 / 36 / 20 N m at 0 / 10 / 30 / 127 rpm | `sim/data/drives.json:21-29` |
| X8-150 electrical | Kt 2.97 N m/A rms, 0.09 ohm per phase | MyActuator V4.1 spec book (28 Nov 2026) |
| X8-150 price | $725 | F9 (18 Nov 2026) |
| RobStride 03 (reference) | 21 N m at 100 rpm, 60 N m peak, 900 g ± 20 g | Maker README, github.com/RobStride/Product_Information, `README_main.md:20` (28 Nov 2026) |
| RobStride 03, other figures | The aluminium heat-sink condition, 13 N m held still, Kt 2.36 N m/A rms, 880 g ± 20 g; resistance not found (0.293 ohm is a desk assumption) | Third-party extracts of the manual (github.com/Human-Pygmalion/HuphyChan-dev, `RS03.txt:151`, `:171`), scratch `facts_actuators.md:151`, `:231` (28 Nov 2026) |
| RobStride 03 price | $225, taken at 41.5 % duty (no US stock confirmed) | Seeed, `hardware/triceratops_kickoff/PARTS_PREMIUM.md:217` (page updated 21 Oct 2026) |
| Other drive makers | CubeMars, SteadyWin, Unitree and DAMIAO; not scored | Scratch `facts_actuators.md` (captures of 23 Jul 2026; search summaries of 28 Nov 2026) |
| Rods and rod ends | Class 8.8 rolled studs, proof 580 MPa, E 200 GPa; stress areas M8 36.6 mm² (record), M10 58.0 and M12 84.3 mm² (the record's formula on ISO pitch diameters, desk); steel rod ends C0 M8 12.9 kN (lowest found 7.9 kN), M10 17.6 kN, M12 20.8 kN (the lowest listed) | `design/pushrod60.py` SOURCES (web search 27 Nov 2026); M10 and M12: web-search summary 28 Nov 2026 (MISUMI uk.misumi-ec.com; INA via uk.rs-online.com 2600271; pages not opened; others list 23.5-25.5 kN) |
| Battery | 2 × Gens ace 6S 3300 mAh = 146.5 Wh, 1.012 kg, $107.19 per pack | genstattu.com via the repo catalog (7 Nov 2026); `PARTS_PREMIUM.md:82`, `:307-311`, `:391` |
| Demo-loop energy | 80 % usable; 30 W hotel load (25-40 W range); 84-96 min at 12.75 kg | `rl/demo_course.py:47-48`; `rl/analysis/REPORT_FORELEG_D22_APPENDIX.md:160-161` |
| Cut-off (D47) | Orion BMS Jr 2 at $440 list, plus about $100 (estimate) for a sensor: $540; mass 0.302 kg = a 0.25 kg estimate × the 1.2088 growth factor | Web search (18 Nov 2026), `PARTS_PREMIUM.md:392`; `DECISION_LOG.md` (D47); mass basis `design/PARTS_60CM.md:25-26`, `:164-177` (working tree, in flight) |
| Landed rule | (hardware + 5 % freight + duty × drives + brokerage + 7 % tax) × 1.07 | `budget/size_budget.py:160-164` |
| Duty | 0 % from US stock; 41.5 % + $150 brokerage on factory dispatch | `budget/size_budget.py:167-171`; owner's information (18 Nov 2026), E4, `DECISION_LOG.md` (E4) |
| 60 cm cost check | $16,776 hardware; $20,105 / $21,240 / $22,149 landed | `budget/size_budget_out.txt`, reproduced by the study's tests |
| Room and limit | 35 C room; 90 C winding rule (bench B3) | D48, `DECISION_LOG.md` (D48) |
| Thermal model | R_th 5.56 / 2.89 K/W at 8 min; fit 25 to 100 C in 90 min | `rl/telemetry/thermal.py:10-13`, `rl/analysis/thermal_ambient.py` (both in flight); `READINESS_REVIEW_2026_11_27.md:210-214` |
| Servo gains | kp 598.5 N·m/rad | A2, `DECISION_LOG.md` (A2) |
| Published robots | Masses, levers, torques, gains | Model files opened 28 Nov 2026 (sha256 in the scratch `facts_existence.json`): Cheetah-Software `MiniCheetah.h` and `rt_spi.cpp:36`; `unitree_ros` const.xacro (A1, Go1, Go2, Aliengo, B1, B2); Menagerie (ANYmal B and C, Barkour vB, Spot); legged_gym ANYmal B URDF (knee joint at xyz '0.0 0.109 -0.25'); Isaac Lab; Walk These Ways `go1_config.py`. Paper and maker figures are web-search summaries (28 Nov 2026). |

**Caveat.** The cited build files that were in flight while this study was written (the readiness review,
`DECISION_LOG.md`, `PARTS_60CM.md`, `thermal.py`, the heat35 results, the packaging and pushrod files, `sit60.py` and
the fore128 trot results) were committed by another job on 28 Nov 2026 (head `637c8170`). The decision-log lines cited
here were checked against that commit; the log's lines moved twice that day, so recheck them after any later edit.

## Files

Build tree:
- `design/form_factor_study.py`, `design/form_factor_out.json` (126 configurations) and
  `design/tests/test_form_factor_study.py` (36 tests);
- `sim/build_scaled_trike.py` and `sim/tests/test_build_scaled_trike.py` (22 tests);
- `sim/results/form_factor_2026-11-27/` (with `scaled_trike_plan_addendum.json` for the post-hoc cases).

Scratch fact sheets, the root-issue map, the reviews' probes and the 126-row table:
`/tmp/claude-0/-home-user-mesozoic-labs/ccd22965-ed7a-50aa-9e57-30e96c9784a3/scratchpad/form_factor/` (the second
review's round in `r2/`, the third's in `r3/`). The current 126-row table is `r3/form_factor_table_r3.md`; the
folder's `form_factor_table.md` (first version) and `r2/form_factor_table_r2.md` are superseded and say so, as do the
first-round `facts_existence.md` and `root_issues.md` in part.
