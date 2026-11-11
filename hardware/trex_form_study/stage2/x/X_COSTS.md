# What the expressive head costs (Stage 2 S2-2, the `-X` bodies)

11 Nov 2026. Simulation only. Nothing is adopted (P24), and P2's deferral of a jaw for the build stands (P29). Plan:
`STAGE2_PLAN.md` S2-2. Research: `stage2/research/research_look.md` section 5 ([look]). Bodies and their checks:
`HEAD_X.md`.

"a / b" figures are 60 cm / 75 cm: `60-N-T-10-X` against `60-N-T-10`, and `75-M-H-10-X` against `75-M-H-10`. "Today"
is today's head on the same body. A1 is the sustained planning torque (0.57 × the vendor's rating). "Peak" is the
planning peak (0.70 × the vendor's peak). For the STS3215 these are A1 0.491 N·m, peak 0.862 N·m and a rated speed of
2.83 rad/s (`designlab/data/actuators.json`, `designlab/catalog.py`).

**Sources.** Every number here is a field of `x_costs.json`, written by `x_costs.py` (`bash stage2/x/run_x.sh costs`,
about 5 min on 2 processes). Each block of the JSON names its rule and inputs. The bodies are rebuilt from their
records (XML sha256 checked: `993c5ac5`, `9718b3b5`, `0517e4d3`, `66f531bb`). The static numbers reproduce the case
records (`round2/refine/head_x_{60,75}/candidates/*.json`, block `head_x`) exactly: 53 checks per body, largest
difference 0. Figures quoted from other records name them. Tests: `tests/test_stage2_x_costs.py`.

## In brief, for the owner

- **The body.** The head adds 0.150 kg, the two servos with their brackets. The centre of mass moves 5.2 / 4.4 mm
  forward. The hip-pitch hold on one foot rises by 0.65 / 0.79 N·m, from 0.55 to 0.65 / 0.76 to 0.83 of its limit. That
  is inside, at the low end of the plan's expectation.
- **Walking, in the Stage 0 screen.** With the new joints held still, the nominal and animal-pace walks stay within
  every drive limit on both heads. No verdict changes. The tightest limit is still the knee's speed at the animal pace:
  0.93 → 0.97 / 0.93 → 0.96. The speed cost here is small; the closed-loop cost is Track B's to measure.
- **Looking at people works, calmly.** A 0.6 rad gaze shift fits within the yaw servo's limits and margins in 0.5 s at
  60 cm and in 0.6 s at 75 cm. In 0.8 s it uses about half the servo's peak. Holding a gaze while standing costs
  nothing.
- **The roar works, slowly.** Opening the jaw 0.6 rad in 0.4 s runs the servo at its rated speed. Half a second keeps
  both margins. Holding it open costs 4 / 9 % of the servo's sustained budget. A quick snap is out of reach.
- **Most of each move's load is the model's default joint damping and friction**, unsourced values (A16). A bench
  measurement could move these loads more than any part change ([look] 5.2).
- **The cost to watch is the 60 cm neck's heat budget.** The jaw servo sits in the head, so the neck-pitch hold rises
  from 0.75 to 0.84 of its limit. In Stage 1's closed loop, today's 60 cm neck already ran at 0.83–0.84 of it in the
  nominal walk (`STAGE1_PROBE.md`). The `-X` neck would scale to about 0.93 (an estimate), before any look-at or roar.

## 1. Mass, centre of mass and holds

Holds are computed as round one's screens compute them (`sizing.requirements`; `DEFINITIONS.md`, "Single-support
hold"). One foot is lifted 5 % of H and the centre of mass is held over the stance pad. The trunk's roll is searched
over ±6°. The hold is the larger of the two sides. Ratios are to A1.

| | 60 cm | 75 cm |
|---|---|---|
| Mass, kg | 10.81 → 10.96 (+0.150, +1.4 %) | 14.85 → 15.00 (+0.150, +1.0 %) |
| Added parts | yaw servo and bracket, 0.075 kg, on the trunk at the neck base; jaw servo and bracket, 0.075 kg, in the head | the same |
| Head and neck, kg | 0.531 → 0.606 | 0.792 → 0.867 |
| Centre of mass, added parts alone (the same pose) | 5.2 mm forward, 2.1 mm up | 4.4 mm forward, 1.8 mm up |
| Centre of mass ahead of the hips, each body at its own stance | +6.4 mm | +5.2 mm |
| Centre of mass with the head turned 0.6 rad | 3.6 mm sideways, 1.1 mm back | 4.8 mm sideways, 1.5 mm back |
| Hip-pitch hold, N·m (A1 5.99 / 11.40) | 3.27 → 3.92 (+0.65) | 8.69 → 9.48 (+0.79) |
| Hip-pitch hold, share of A1 | 0.55 → 0.65 | 0.76 → 0.83 |
| Neck-pitch hold, N·m (STS3250 A1 0.819 / XM540-W270 A1 1.770) | 0.617 → 0.690 (+0.073) | 1.160 → 1.247 (+0.088) |
| Neck-pitch hold, share of A1 | 0.75 → 0.84 | 0.66 → 0.70 |
| Neck pitch, Stage 0 continuous ratio (larger of the nominal walk's RMS and the hold) | 0.76 → 0.85 | 0.66 → 0.71 |
| Neck pitch standing at home, with the jaw closed / open 0.6 rad | 0.694 / 0.680 N·m (0.85 / 0.83) | 1.254 / 1.222 N·m (0.71 / 0.69) |
| Knee hold, N·m (falls) | 3.05 → 2.73 | 2.79 → 2.36 |
| New drives' holds on one foot: neck yaw / jaw, share of A1 | 0.15 / 0.07 | 0.27 / 0.15 |

- **Against expectations.** The plan expected +0.61–0.71 / +0.75–0.88 N·m on the hip pitch, taking it to 0.65–0.67 /
  0.83–0.84. The measured rise sits at the low end. [look] 5.2 estimated the neck at 0.75 → 0.84 of A1 at 60 cm and
  0.66 → 0.71 at 75 cm. The model gives 0.75 → 0.84 and 0.66 → 0.70 (Stage 0's continuous ratio: 0.66 → 0.71).
  [look] 5.2's centre-of-mass estimate, 5.7 / 5.1 mm, was slightly high.
- **Two stances.** The build re-solves the stance, so the feet move forward under the shifted centre of mass
  (`HEAD_X.md` §6). That move adds 1.2 / 0.8 mm to the centre of mass's lead over the hips. It changes no load: posed at
  today's home, the `-X` body's holds agree to within 1e-4 N·m.
- **Turning the head.** With the yaw axis vertical at home, turning the head does not change the neck-pitch hold. It
  moves the centre of mass 3.6 / 4.8 mm sideways.
- **The new drives on one foot.** In single support the trunk rolls, so gravity acts about the neck-yaw axis.
- **The records' other costs.** These come from the round-two cost and power models in the case records:
  - nominal cost: $14,809 → $14,890 / $18,521 → $18,596 (+$81 / +$75);
  - walking power: 69.3 → 73.4 W / 89.8 → 95.6 W (+4.1 / +5.9 W). The hip pitch adds +1.4 / +2.5 W, the two new
    servos holding still add +2.4 / +3.7 W, the neck adds +0.8 / +0.4 W, and the knee saves 0.7 / 0.8 W;
  - runtime on the same pack: 47.6 → 44.9 min / 36.7 → 34.5 min.

  The yaw stage's own bearing is in neither the mass nor the cost (`HEAD_X.md` §7).

## 2. The Stage 0 walking screen: today's nominal and animal-pace walks, the new joints held still

The walks are Stage 1's references: nominal is Froude 0.012, at 0.218 / 0.243 m/s, and the animal pace is Froude 0.05,
at 0.446 / 0.496 m/s. Each was regenerated on each body and checked equal to its saved arrays. On the `-X` bodies the
neck yaw and the jaw stay at 0 throughout: the generator moves only the legs, the neck pitch and the tail.

Each walk is priced the way the round-two envelope screen prices a walk (`envelope._ratios`, as Track A's look ladder
does), against the record's drives on the 24 V bus:

- cont: the walk's RMS torque over A1;
- peak: 1.25 × the peak torque over the planning peak;
- speed: 1.15 × the peak speed over the speed at the bus.

A ratio "moves" when the heads differ by 0.005 or more. Every other ratio moved less than that.

| Ratio, today → `-X` | 60 nominal | 60 animal pace | 75 nominal | 75 animal pace |
|---|---|---|---|---|
| Worst ratio (its drive) | 0.76 → 0.85 (neck) | 0.93 → 0.97 (knee speed) | 0.66 → 0.71 (neck) | 0.93 → 0.96 (knee speed) |
| Neck pitch cont | 0.76 → 0.85 | 0.77 → 0.86 | 0.66 → 0.71 | 0.67 → 0.72 |
| Neck pitch peak | 0.60 → 0.67 | 0.65 → 0.72 | 0.53 → 0.56 | 0.57 → 0.61 |
| Hip pitch cont | 0.44 → 0.51 | 0.64 → 0.69 | 0.57 → 0.61 | 0.70 → 0.75 |
| Hip pitch peak | 0.32 → 0.37 | 0.51 → 0.54 | 0.37 → 0.40 | 0.49 → 0.51 |
| Hip pitch speed | 0.53 → 0.52 | 0.90 → 0.88 | 0.49 → 0.48 | 0.83 → 0.82 |
| Knee speed | 0.56 → 0.58 | 0.93 → 0.97 | 0.56 → 0.57 | 0.93 → 0.96 |
| Knee cont | 0.46 → 0.43 | 0.49 → 0.46 | 0.52 → 0.48 | 0.56 → 0.53 |
| Knee peak | 0.28 → 0.26 | 0.34 → 0.32 | 0.33 → 0.31 | 0.43 → 0.41 |
| Other legs (each under 0.01) | ankles +0.006 | ankles, hip roll +0.007–0.008; hip yaw −0.009 | none | hip roll +0.008; hip yaw −0.007; ankles +0.005–0.006 |
| New: neck yaw held still, cont / peak (RMS, N·m) | 0.21 / 0.23 (0.10) | 0.27 / 0.31 (0.13) | 0.35 / 0.39 (0.17) | 0.46 / 0.52 (0.23) |
| New: jaw held closed, cont / peak | 0.07 / 0.06 | 0.07 / 0.07 | 0.15 / 0.13 | 0.16 / 0.15 |
| Neck pitch with the MJCF's damping and friction, cont (informational) | 0.82 → 0.90 | 0.84 → 0.93 | 0.68 → 0.72 | 0.69 → 0.74 |

- **Every walk passes on both heads.** No ratio reaches 1. The walk reading was regenerated equal to Stage 1's arrays,
  and the clearance screen's walking gates pass on both heads. Mid-swing toe clearance is unchanged at both sizes:
  0.050 H in the nominal walk, 0.054 H at the animal pace.
- **The knee's speed** at the animal pace is the governing limit on both bodies with either head. Its headroom
  shrinks: the knee's peak speed goes from 0.81 to 0.84 of its rated speed at both sizes (0.93 → 0.97 / 0.96 with the
  15 % margin). The rise comes from the added parts, not the re-solved stance: it is the same when the `-X` body
  starts from today's home (`HEAD_X.md` §6), since the walk is planned around the centre of mass either way.
- **Holding the head straight is work for the yaw servo.** With the trunk sliding side to side in every stride, the
  yaw servo carries 0.10–0.23 N·m RMS, or 0.21–0.46 of its A1, most at 75 cm's animal pace. The jaw servo holds the
  jaw closed against gravity.
- **Contact.**
  - The single-support centre-of-pressure margins do not move (30.3 → 30.5 and 29.7 → 29.7 mm at 60 cm; 45.3 → 45.3
    and 44.2 → 44.1 mm at 75 cm).
  - At 75 cm, the inverse dynamics' largest per-foot friction ratio rises: 0.285 → 0.309 (nominal) and 0.358 → 0.400
    (animal pace). Each rise sits at one double-support frame where one foot carries only 11–15 % of the weight.
  - The whole robot's friction need, which does not depend on how the load is split between the feet, does not move:
    0.214 → 0.213 and 0.290 → 0.288 at 60 cm; 0.208 → 0.208 and 0.282 → 0.281 at 75 cm (nominal; animal pace).
- **The passive terms.** Stage 0 leaves the MJCF's damping and friction out (informational row). Counting them, the
  60 cm `-X` neck reads 0.90 of A1 in the nominal walk and 0.93 at the animal pace. The tail reads the same on both
  heads: 0.68 and 0.85 at 60 cm, 0.80 and 1.03 at 75 cm (nominal, animal pace). The 75 cm tail's 1.03 at the animal
  pace is today's tail item (Stage 1b §6), not the head's.
- **The round-two envelope screen** has records for both heads
  (`round2/refine/{narrow_carpet_60,feet_75,head_x_60,head_x_75}/review/screening/*_envelope.json`). Its animal-pace
  walk is built differently (`turning.generate`) but gives the same knee reading to 12 digits: 0.929 → 0.966 /
  0.935 → 0.963. Elsewhere in that record:
  - the one-step push bound falls 2.2 / 1.6 %: 0.487 → 0.476 / 0.589 → 0.579 m/s;
  - the one-foot stop falls 0.6 / 0.4 %: 1.250 → 1.242 / 1.414 → 1.408 m/s²;
  - the session's hottest drive is still the neck: 0.76 → 0.85 / 0.66 → 0.71 of A1;
  - the session's hip pitch goes 0.41 → 0.48 / 0.53 → 0.57.

## 3. The character moves on the two STS3215s

**How the moves were measured.** Each move is a minimum-jerk profile from rest to rest, with the robot standing at
home. The moves are:

- the gaze: neck yaw from −0.3 to +0.3 rad. From the centre to one side (0 to 0.6 rad) the loads are the same to
  7 digits, since the yaw axis is vertical;
- the jaw: 0 to 0.6 rad and back.

Each joint keeps the MJCF's default damping (0.2 N·m·s/rad) and friction loss (0.15 N·m) (`designlab/mjcf.py`,
unsourced, A16). The armature is the catalog's. Each move is read two ways:

- **ID:** rigid-body inverse dynamics with the trunk held, plus damping × rate and friction × sign(rate);
- **Plant:** the Stage 1 plant standing on the hard floor. It is commanded every 2 ms with the rigid-body feedforward
  only (as Stage 1's references carry no damping or friction), with no delay. The reading is the servo's force, after
  the plant's torque-speed envelope.

"× margin" applies the sizing's margins: 1.25 × the peak torque, 1.15 × the speed.

| Move | 60 cm | 75 cm |
|---|---|---|
| **Gaze, 0.6 rad in 0.5 s**: peak torque, ID / plant | 0.635 / 0.657 N·m | 0.753 / 0.814 N·m |
| — share of the peak limit, ID / plant (× margin) | 0.74 / 0.76 (0.92 / 0.95) | 0.87 / 0.94 (1.09 / 1.18) |
| — the peak's parts: inertia, damping, friction | 0.07, 0.42, 0.15 | 0.25, 0.35, 0.15 |
| — peak speed 2.25 rad/s: share of rated (× margin) | 0.80 (0.92) | 0.80 (0.92) |
| — RMS over the move; one shift every 2 s (share of A1) | 0.432 (0.88); 0.216 (0.44) | 0.479 (0.98); 0.240 (0.49) |
| — reaction on the feet: yaw about the support centre, sideways force | 0.45 N·m, 1.0 N | 0.97 N·m, 1.8 N |
| **Gaze, 0.6 rad in 0.8 s**: peak, ID / plant (share of peak; × margin) | 0.440 / 0.446 N·m (0.51 / 0.52; 0.64 / 0.65) | 0.474 / 0.490 N·m (0.55 / 0.57; 0.69 / 0.71) |
| — speed 1.41 rad/s (× margin); feet: yaw | 0.50 (0.57); 0.17 N·m | 0.50 (0.57); 0.38 N·m |
| **Jaw opening, 0.6 rad in 0.4 s**: peak, ID / plant | 0.685 / 0.685 N·m | 0.650 / 0.651 N·m |
| — share of the peak limit (× margin) | 0.79 (0.99) | 0.75 (0.94) |
| — the peak's parts: inertia and gravity, damping, friction | −0.03, 0.56, 0.15 | −0.06, 0.56, 0.15 |
| — peak speed 2.81 rad/s: share of rated (× margin) | 0.995 (1.14, over) | 0.995 (1.14, over) |
| **Jaw closing, 0.6 rad in 0.4 s**: peak, ID / plant (× margin) | 0.740 / 0.739 N·m (1.07 / 1.07, over) | 0.776 / 0.775 N·m (1.12 / 1.12, over) |
| **Open jaw held at 0.6 rad**: ID / plant (share of A1) | 0.020 / 0.019 N·m (0.04) | 0.044 / 0.043 N·m (0.09) |
| Neck pitch during the jaw moves, plant | 0.59–0.81 N·m (A1 0.82, peak 1.44) | 1.11–1.38 N·m (A1 1.77, peak 3.11) |

**The fastest 0.6 rad move within the limits.** Read two ways:

- the ID by bisection;
- the plant swept from 0.40 to 0.80 s in 0.05 s steps. A move whose servo hits its force limit counts as over.

| Fastest 0.6 rad move, s: raw limits / with margins | 60 cm, ID | 60 cm, plant | 75 cm, ID | 75 cm, plant |
|---|---|---|---|---|
| Gaze | 0.40 / 0.46 (speed) | 0.40 / 0.50 | 0.45 / 0.54 (torque) | 0.50 / 0.60 |
| Jaw opening | 0.40 / 0.46 (speed) | 0.40 / 0.50 | 0.40 / 0.46 (speed) | 0.40 / 0.50 |
| Jaw closing | 0.40 / 0.46 (speed) | 0.40 / 0.50 | 0.40 / 0.47 (torque) | 0.40 / 0.50 |

- **The plant reads up to 8 % above the ID on the 75 cm gaze.** The feedforward leaves out the damping and friction,
  so the servo makes them up from its tracking error (about 0.01 rad). The plant's sweep is the one to plan by. At
  75 cm a gaze of 0.40–0.45 s saturates the servo.
- **The STS3215's rated speed caps every 0.6 rad move at about 0.40 s.** With the 15 % margin the cap is 0.46 s.
- **Standing, the feet carry the gaze's reaction easily.** In the plant the trunk twists less than 0.1°.

## 4. What it means for the look ([look] 5.1–5.3)

- **Looking at people: yes, at a calm pace.**
  - [look] 5.1's gaze is a 0.3–0.6 rad shift in 0.5–0.8 s, then a hold. The neck-base yaw servo does all of it at 60 cm
    within its margins, and at 75 cm from 0.6 s.
  - At 75 cm a 0.5 s shift still fits under the servo's limit, using 94 % of it in the plant, but not under the 25 %
    margin the sizing keeps for transients.
  - [look] estimated that a 0.5 s shift at 75 cm would exceed the peak, at 0.92 N·m. That figure equals the sum of the
    inertial, damping and friction peaks (0.32 + 0.45 + 0.15), which come at different moments. Measured together,
    the peak is 0.75 N·m (0.81 in the plant).
  - Holding a gaze while standing costs nothing, because the yaw axis is vertical at home. Even a busy rhythm, a full
    shift every 2 s, uses under half the servo's sustained budget.
- **Head-first turns** use the same shift, so the same timing applies. While walking, two things add up:
  - the yaw servo already works to keep the head straight against the trunk's sway (up to 0.46 of its A1 at 75 cm's
    animal pace);
  - a shift's reaction lands on the stance foot as spin (0.45 / 0.97 N·m for a 0.5 s shift; 0.17 / 0.38 N·m for
    0.8 s). Foot spin is the friction limit that binds (F21).

  Track B's gaze events measure both. Slower shifts, or shifts timed to double support, are the cheap answers.
- **The roar: yes, as a deliberate opening.**
  - Opening the mouth 0.5–0.6 rad in at least 0.4 s ([look] 5.1) runs the jaw servo at its rated speed. Half a second
    keeps both margins.
  - Closing is the harder direction, because gravity opens the jaw.
  - Holding the mouth open is nearly free.
  - A film-style snap (0.3–0.4 rad in 0.15 s) needs about 5 rad/s, out of the STS3215's reach, as [look] said. The
    look will be slow and deliberate, which suits a big animal's timing ([look] bottom line 5).
- **The model's damping sets most of these limits, not the head's weight.**
  - At the peak of each move the default joint damping is 0.56 of the jaw's 0.69 / 0.65 N·m and 0.42 / 0.35 of the
    gaze's. Inertia is under 0.08 N·m on the jaw.
  - The defaults are unsourced and probably pessimistic for a bearing-mounted pivot ([look] 5.2).
  - The tail bench test can measure the real joint friction and damping. Track D's lever 4 (damping and friction
    × 0.25 on the tail, neck yaw and jaw) shows what that would free up. [look] 5.2 expects that one measurement to
    move the expressive budgets more than any part change.
- **The cost to watch is the 60 cm neck pitch's heat, not speed.**
  - The jaw servo in the head takes the 60 cm neck's hold from 0.75 to 0.84 of its sustained limit.
  - In Stage 1's closed loop, today's 60 cm neck already ran at 0.83–0.84 of that limit in the nominal walk (Stage 0
    said 0.76; `STAGE1_PROBE.md`). Scaled the same way from Stage 0's 0.85, the `-X` neck would run at about
    0.93–0.94 before any look-at or roar. That is an estimate; Track B measures it.
  - The 60 cm `-X` may therefore show the bar only "but for A1" (plan §5, reading 3).
  - Mitigations, cheaper than a bigger servo ([look] 5.2): idling with the head slightly raised, a neck gravity spring
    (Track D lever 5), or the thinner PA11 covers already chosen for the build (P21). On a 60 cm round-two body the
    covers took the neck from 0.76 to 0.70 (F22); they are not yet on the picks' model.
  - At 75 cm the XM540-W270 neck goes from 0.66 to 0.71: comfortable.
- **On the owner's question** ("is the extra expressiveness worth a slight speed hit?"):
  - In Stage 0 the hit is small. No walk or limit changes its verdict. The knee's speed headroom at the animal pace
    shrinks by 3–4 points. The hip pitch keeps at least 0.17 of its sustained limit on one foot. The push bound falls
    about 2 %, and runtime falls 2.3–2.6 minutes.
  - [look] 5.3's view stands: the neck yaw is the most valuable joint per gram, and the jaw second. What it costs is
    calm timing and the 60 cm neck's heat budget.
  - The closed-loop answer, the bar with the head neutral and with gaze events, is plan §5's "head's cost" in
    Track B.

## 5. What this does not show

- **The closed loop.** Every load here is the reference's (Stage 0), the trunk-held inverse dynamics or a standing
  plant without delay. The walking loads under a controller, the 10 ms delay and gaze events while walking are
  Track B's.
- **The yaw stage's bearing,** and today's pitch servo riding on it (`HEAD_X.md` §7). Its mass, cost and friction are
  not modelled.
- **The joint damping and friction.** They are the MJCF's defaults (A16), here as in the plant and in training.
- **The jaw cover's overhang** behind the hinge, kept whole (`HEAD_X.md` §4). Trimming it would change every `-X`
  record.
- **No new model files.** The bodies are rebuilt in temporary directories, and the plant reads the cases' models
  already listed in `x_inputs.json`.
