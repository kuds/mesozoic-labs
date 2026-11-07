# Round-one screening

What was screened, how, what failed, and what was fixed. This is screening, not certification.
Every result comes from the compiled models and the prescribed reference walks, not from a
controller and not from hardware. Generated tables: `round1/TABLES.md`. Raw screen outputs:
`round1/review/screening/`.

## 1. Screens and where they run

| Screen | Candidates | Module | Gates |
|---|---|---|---|
| Packaging, joint loads, actuator and battery selection, power, runtime, cost | all 132 Stage A candidates | `stage_a.py` → `pipeline.size_candidate` (`statics`, `gait`, `sizing`, `power`, `cost`) | Balanced single-support pose exists within joint limits. Reference-walk CoP within 10 mm of the pad. Every load within a catalog part's planning limits. Sizing converged, or stopped on a recurring selection with the stronger parts kept (12 of 132); no mass spiral |
| Self-collision, swing and tail clearance, battery removal path | review set | `clearance.py` | No interference at home or along the nominal walk. Mid-swing toe clearance ≥ 0.04 H. Tail ≥ 0.25 H above the floor. At least one clear battery path |
| Camera, ToF and lidar coverage | review set | `sensors.py` | No gate: reported for comparison. A person's torso visible straight ahead at 1.5–3 m is the following requirement |

Load definitions are in `DEFINITIONS.md`. The quasi-static single-support hold lifts one foot by
5 % of H, holds the centre of mass over the stance pad, keeps the stance foot flat, and searches
trunk roll over ±6°. The reference walks are nominal (Froude 0.012, stride 0.55 H, duty 0.60) and
brisk (Froude 0.025, stride 0.62 H, duty 0.58). Each uses a linear-inverted-pendulum centre-of-mass
path corrected against the multibody ZMP, with full inverse dynamics.

## 2. Failed cases kept on record

Each failure was found by a screen, its cause identified, and the generator changed. The failing
versions were not committed as results. Their measurements are recorded here and in the decision log
(F3–F6).

| Found by | Failure | Cause | Class | Change |
|---|---|---|---|---|
| Stage A, first full sweep | Every 75 cm candidate spiralled to 21–23 kg on 48 V; no balanced single-support pose | Parts filtered by exact nominal voltage (the V4 X-series drives accept 20–55 V); score blind to motor diameter, which sets hip spacing; spiral guard compared every iteration with the first build | Sizing method | Voltage windows, packaging term, a per-iteration spiral test with only a 2× first-build backstop (F3) |
| Stage A, first full sweep | 100 cm sizing cycled between two hip-roll parts | Two selections each produced loads favouring the other | Sizing method | A recurring selection now keeps the stronger part (F3) |
| Sensor screen, layout v1 | Throat camera: 17–24 % of the view blocked by the jaw. At 60 cm, 0 % of a person's torso straight ahead visible | Camera below the head line | Sensor coverage | Camera moved to the skull roof (F4) |
| Sensor screen, layout v1 | Camera box intersecting the neck motor (0.5–11.9 mm) | Throat mount between neck motor and covers | Geometry | As above (F4). The neck-motor conflict is gone; the v2 mount check still flags the camera box against the skull cover (§5) |
| Sensor screen, layout v1 | ToF pair overlapping 82–90 %; foot-path coverage 15–56 % | Identical aim | Sensor coverage | 45° down, 12° outward each (F4) |
| Sensor screen, layout v1 | Lidar 93–100 % blocked within 1 m at 100/120 cm | Inside the hull, which had been deepened around it | Sensor coverage, geometry | Under the throat on a bracket (F4) |
| Visual check | Substantial trunk a flat-bottomed loaf; uncovered mid-trunk gap (Tapered); bone jaw read as a beak | Shape tables and cover regions | Appearance | Belly rises to a pubic keel, continuous Tapered back, skin-toned jaw (F5) |
| Clearance screen | Left and right hip brackets overlapping 6.2–6.5 mm in every pose | Both brackets centred on the hip axis across the centreline | Geometry | Each bracket on its own roll motor's side (F6) |
| Clearance screen | Tail motor inside the battery cassette (60, 75 cm); cassette reaching into the tail (60 cm); no battery removal path | Tail-base cassette placed at hip height and too long for the space ahead of the tail pivot | Geometry, service access | Tail motor on the pivot; cassette low with a ventral hatch; tail pivot moved back where the cassette needs room: 43 mm on the 60 cm reference `60-M-S-10`, 40–43 mm across the 60 cm tandem review models (F6) |
| Clearance screen | Power board inside the compute tray (60 cm, −18.6 mm) | Fixed x station | Geometry | Held behind the tray in the tail-base layout (F6); see §4 for overlaps that remain outside the gates |
| Clearance screen | Ankle pitch usable only ±0.14 rad (60 cm) | Metatarsus plates drawn to the ankle-joint centre | Geometry | Plates stop short of the joint block (F6) |
| Clearance screen | Pelvis and thigh shells overlapping at home, up to 21 mm (120 cm) | Trunk half-width at the hip wider than the thigh fairing's inner face | Shell layout | Pelvis covers narrowed inside the fairings (F6) |
| Clearance screen | Upper-thigh fairing meeting the upper ankle motor at knee flexion (60 cm) | Short 60 cm thigh against X4-size ankle motors | Geometry | Bulge tapers faster; motors 8 % lower on the shank. Both 60 cm walks are now contact-free, with near-contacts within 3 mm (§4) |

## 3. Stage A results (all 132 candidates)

All 132 candidates built and sized. No sizing run hit the mass-spiral rule. Every candidate reaches
at least 20 min on its selected pack (range 20–48 min). The full list is in `round1/TABLES.md` §11.
Flags:

| Flag | Candidates | Cause | Class |
|---|---|---|---|
| Hip-pitch load beyond the strongest single catalog drive (MyActuator X12-P20-320, 48.4 N·m sustained planning limit) | 120 cm with the Substantial or Hybrid body at narrow and balanced width (ratio 1.01–1.07), including the 120 cm reference `120-M-S-10` (1.056), `120-W-S-10` (1.03), and the 120 cm large-skull (1.11), compact (1.39), belly-battery (1.25) and serial-leg (1.03; `TABLES.md` §11 lists only its hip-roll flag) variants | The head and chest sit ahead of the hip: the narrow and balanced Substantial and Hybrid bodies (43–46 kg) hold 49–52 N·m in single support. The Tapered body fits (0.91–0.98; 44–48 N·m) | Actuation |
| Hip-roll load beyond the catalog | `100-M-S-10-serial` (1.04), `120-M-S-10-serial` (1.31), `120-M-S-10-side` (1.47) | Hip-roll axis offset from the centreline, as in PR #662 | Actuation, leg layout |
| No balanced single-support pose within joint limits | `60/75/100-M-S-10-side` | With the roll motors side by side, the roll axes sit off the centreline, and the trunk's own weight adds a roll moment the hip range cannot balance | Leg layout |
| Nominal reference walk leaves the support pad (CoP more than 10 mm outside) | `60-M-H-20` (−26 mm), `75-M-S-10-belly` (−12 mm), `60-W-S-00` (−11 mm). Nine more leave the pad by less than 10 mm: `60-N-S-00` (−8.3 mm), `60-M-H-00` (−7.9), `100-M-S-10-compact` (−6.5), `75-N-S-00` (−5.3), `120-W-S-10` (−3.5), `120-W-S-00` (−1.8), `60-W-S-20` and `60-W-T-10` (−1.1), `75-N-H-00` (−0.2) | The prescribed walk's centre-of-mass path does not fit these mass distributions at the fixed stride and timing. This limits the reference trajectory; it is not shown to be physically infeasible | Reference trajectory (re-test with a controller) |
| High-side cost above $20k, the top of the owner's $10–20k prototype budget (P4) | all 132 (high side $20.4–57.6k; 60 cm $20.4–23.3k). 68 are also above $20k at nominal: every 100 and 120 cm candidate, plus `75-M-S-10-serial` and `75-M-S-10-side` | The frame, cover, integration and contingency lots are scaled from PR #662's 60 cm lines (A8), whose own high case is already $20.6k. Per P8 the cost is reported, not used to rule candidates out | Cost |

In the tandem layout, hip-pitch load is the largest joint load at every size except at 60 cm, where thigh
twist is larger at broad width (4.1–4.4 vs 3.1–4.0 N·m) and in the anatomical-tail variant `60-M-S-10-anat`
(3.6 vs 3.3 N·m). In the serial and side-by-side layouts hip roll is larger at every size (for example
71.4 vs 38.4 N·m in `120-M-S-10-side`; on walk RMS for the side-by-side variants at 60–100 cm, which have
no single-support hold). Per height class:

| Class | Mass | Hip-pitch hold | Hip-pitch drive | Bus | Nominal cost | High cost |
|---|---|---|---|---|---|---|
| 60 cm | 10.7–12.7 kg | 3.1–8.0 N·m | MyActuator X4-36-L; RobStride 03 for the belly-battery variant | 24 V | $14.7–17.0k | $20.4–23.3k |
| 75 cm | 14.6–18.6 kg | 7.4–13.9 N·m | RobStride 03 (RobStride 04 or 06 in some variants) | 24 V | $17.3–21.7k | $24.2–29.4k |
| 100 cm | 28.2–37.6 kg | 20.2–33.9 N·m | MyActuator X8-150 or X12-320; RobStride 04 in some variants | 24 V | $26.2–33.0k | $36.4–44.9k |
| 120 cm | 42.8–56.4 kg | 38.4–67.4 N·m | MyActuator X12-320 (0.79–1.39 of its sustained planning limit; past it on 17 of 33) | 24 or 48 V | $35.3–42.0k | $48.3–57.6k |

Options for the 120 cm hip, for round two:
* two drives per hip-pitch joint;
* a belt or planetary reduction stage on an X10/X12-class drive;
* the Tapered body;
* the anatomical tail as a counterweight, which cuts the hold 24 % (51.2 → 39.0 N·m);
* bench thermal data showing the 0.57 sustained fraction (assumption A1) is too conservative for
  this drive.

## 4. Clearance (review set, after the fixes)

All 24 review models rebuilt byte-identical to their Stage A records (SHA-256 match). 20 of 24 pass every gate.

| Reference | Mid-swing toe clearance, nominal / brisk | Tail lowest point | Battery removal | Usable range before a new contact (hip roll in / out; hip flexion; knee flexion) | Near-contacts within 3 mm while walking (nominal / brisk) |
|---|---|---|---|---|---|
| 60-M-S-10 | 20 / 21 mm (0.050 H) | 0.80 H | ventral | −0.21 / +0.12; −0.71; +0.62 rad | 36 / 10 |
| 75-M-S-10 | 25 / 26 mm | 0.80 H | dorsal or ventral | −0.16 / +0.17; −0.52; +0.95 rad | 0 / 0 |
| 100-M-S-10 | 33 / 34 mm | 0.80 H | dorsal or ventral | −0.16 / +0.22; −0.61; +0.90 rad | 0 / 0 |
| 120-M-S-10 | 40 / 41 mm | 0.78 H | dorsal or ventral | −0.11 / +0.17; −0.76; full | 0 / 0 |

Ankle pitch and roll, thigh twist, neck and tail have their full ranges at every size. Inward hip roll is limited by
the two feet's toes meeting, and outward roll by the pelvis and upper-thigh shells. Both limits are wider than the
reference walks use.

Failures in the review set, kept with their cause:

| Candidate | Gate failed | Worst contact | Cause | Class |
|---|---|---|---|---|
| 60-M-S-10-serial | home, walk | Hip-roll motor inside the upper-thigh fairing (−51 mm); ankle motors against the toes (−24 mm) | The coverings and foot were designed around the tandem thigh pod. The serial leg (PR #662-like) puts motors where the fairing is | Shell layout for an alternative leg |
| 60-M-S-10-belly | walk, battery | Upper-thigh fairing vs upper ankle motor in the brisk walk (−18.7 mm) | The belly battery moves the centre of mass forward to 0.18 H (0.12 H on `60-M-S-10`) and raises the hip-pitch hold to 8.0 N·m (4.5 N·m), past the X4-36-L's limit, so the hip pitch gets a RobStride 03 (111 mm installed diameter, against 60 mm). The larger motor lengthens the upper-thigh fairing and moves the thigh-twist motor down the femur, so knee flexion is usable only to 0.29 rad (0.62 rad on `60-M-S-10`). The walks flex the knee no more than on the reference model (0.50 / 0.58 rad, nominal / brisk), but that is past the limit. The belly cassette overlaps the compute tray (−45.6 mm) and compute module (−30.6 mm) at home, which blocks both flank paths; the dorsal and ventral paths are blocked by the power board and spine beam and by the ToF sensors | Mass distribution and service access |
| 60-M-S-10-side | walk, battery | Pelvis plate vs upper-thigh shell (−3.6 mm) | Side-by-side roll motors widen the pelvis plates into the thigh's sweep, and reach 21 mm into the tail-base battery cassette at home, blocking the ventral removal path that is otherwise clear, so no battery removal direction is clear | Leg layout |
| 100-N-S-10 | home, walk | Pelvis plate vs upper-thigh shell (−5.0 mm at home, −8.3 mm in the nominal walk) | The narrow width adds no hip spacing beyond the hardware, so at 100 cm the thigh fairing sits against the pelvis plates | Width variant at this size |

These four are layout and width variants. All four reference sizes and all three families pass.

Overlaps between parts carried by the same body are reported by the screen but are not a gate (the
body is rigid, so they never change with the pose). Two review models still have them at home, and
both are round-two packaging items: in `60-M-T-10` the power board meets the left hip-roll motor
(−4.8 mm) and the tail motor the battery cassette (−0.2 mm); in `60-M-S-10-belly` the power board sits
inside the compute tray (−9.2 mm), next to the battery overlaps listed above.

## 5. Sensor coverage (review set, layout v2)

| Reference | Camera height | View blocked by own body | Person's torso ahead at 1.5–3 m (home / nose down / nose up) | Nearest floor the camera sees ahead of the toes | ToF floor band ahead of the toes | Foot path covered | Lidar blocked within 1 m |
|---|---|---|---|---|---|---|---|
| 60-M-S-10 | 0.58 m | 3.8 % | yes / no / yes | 1.86 m | 0.23–0.86 m | 71 % | none fitted |
| 75-M-S-10 | 0.72 m | 3.1 % | yes / no / no | 2.19 m | 0.28–1.06 m | 65 % | none fitted |
| 100-M-S-10 | 0.97 m | 3.2 % | yes / yes / yes | 2.82 m | 0.37–1.41 m | 52 % | 28 % (rear sector) |
| 120-M-S-10 | 1.17 m | 2.9 % | yes / yes / no | 3.30 m | 0.43–1.67 m | 45 % | 30 % (rear sector) |

* With the head level (home), every review candidate sees a person's torso straight ahead at 1.5–3 m.
* The camera moves with the head, so person-following needs the head held near level. The behaviour spec
  carries this as a constraint (`BEHAVIOR_SPEC.md` §3, "Head attitude while following").
* The camera mount check still flags the camera box. The skull cover cuts it at every size (2.3–13.7 mm at
  home, up to 15.5 mm at the neck limits), and at 60 cm the OAK-D Lite is wider than the skull at its
  station, so the cosmetic eye boxes fall inside it (5.3–9.6 mm). The cover needs a camera window and a
  local bulge in CAD; the coverage results assume that window.
* The floor straight ahead is hidden by the snout out to the distance in the table. Near-field floor sensing
  is the ToF pair's job, and it covers 45–71 % of the foot path. A third downward ToF sensor, or a wider one,
  is a round-two item.
* The lidar's blocked sector is behind it (the chest).

## 6. Head and tail drives: direct versus cable

Round one drives the neck-pitch and tail-yaw joints directly. Each motor sits on the trunk at the
joint, so its mass is already off the head and tail. The comparison with a remote cable (tendon)
drive comes from PR #662's tendon bench (`tendon_bench.json`, 60 cm nominal model, MuJoCo 3.10),
which regenerates to 1.8e-12 relative (`REPRODUCTION.md`). The cable cases use 3:1 drums (30 mm
output, 10 mm input radius). The nominal case has 12 kN/m cable and 18 N pretension per side, the
soft case 3 kN/m at 18 N, and the loose case 12 kN/m at 5 N with 1 mm of slack.

| Neck pitch | Direct | Cable, nominal | Cable, soft | Cable, loose |
|---|---|---|---|---|
| Tracking error RMS / max | 0.19° / 0.26° | 0.44° / 0.75° | 0.46° / 2.20° | 0.45° / 2.10° |
| Motor torque RMS / peak | 0.23 / 0.46 N·m | 0.08 / 0.15 N·m | 0.08 / 0.18 N·m | 0.08 / 0.21 N·m |
| Cable slack | – | 0 % | 0 % | 100 % (minimum tension 0 N) |

| Tail yaw | Direct | Cable, nominal | Cable, soft | Cable, loose |
|---|---|---|---|---|
| Tracking error RMS / max | 0.20° / 0.33° | 0.46° / 0.61° | 0.45° / 0.94° | 0.54° / 1.21° |
| Cable slack | – | 0 % | 0 % | 100 % |

The cable drive adds 0.152 kg for the two axes (drums, bearings, guides, tensioners). It also
roughly doubles the tracking error, and it needs pretension upkeep: the loose case runs slack
throughout. Its benefit is mass placement, which these joints already have. A cable or tendon
drive becomes worthwhile for a multi-segment tail with distributed bending, or for a jaw: in both,
it moves motor mass out of the moving segments. That is a round-two option if the owner wants a
more expressive tail.

## 7. Not screened in round one

| Item | Status | How it will be checked |
|---|---|---|
| Carpet | Not screened. No rollouts were run this round. A friction change alone is not a carpet model (`BEHAVIOR_SPEC.md` §1). | Pile compliance, toe catch and a friction band in the selected design's contact model, then controller trials |
| Turning loads | The reference walk is straight. A 0.5 rad/s turn in place needs about 0.5 rad/s of thigh twist; every selected twist drive offers at least 2.5 rad/s on its bus (CubeMars AK80-64 on 24 V). Turning torques were not computed. | A turning reference gait in `gait.py` before round-two sizing |
| Dynamic balance, slips, impacts, start/stop | Need a controller | Bounded training after design selection (`BEHAVIOR_SPEC.md`) |
| Thermal | Sustained planning fractions only (assumption A1) | Bench thermal tests per joint duty |
| Structure | Beam rules for plates and spars; no FEA, fasteners or bearing life | CAD with FEA after selection |
| Cable routing, connectors | Mass allowances only; the battery path is screened for box overlap, not for hand access or connectors | CAD |
