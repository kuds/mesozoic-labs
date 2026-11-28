# E3 per-policy heat at 60 cm: summary

28 Nov 2026. **DESK on SIMULATED torques.** The temperatures come from the build's one-node thermal model
(`rl/telemetry/thermal.py` via `rl/analysis/thermal_ambient.py`, method A: each drive fitted at the 25 C rating room,
run in a 35 C room; open air, no shell). The torques come from `rl/demo_course.py` segments (MuJoCo 3.10.0, flat floor,
randomisation off, sensor noise on, 10 ms delay). Nothing is measured. Rule and reading:
`design/PREREG_CANDIDATES_2026_11_28.md` (section 2 "E3", section 3 item 7, section 4 "E3"). Code: `sim/candidate_checks.py`
`policy_heat` and `demo_make` (sha256 `008aa308debdf1b6`, the frozen file), run by
`/tmp/claude-0/-home-user-mesozoic-labs/ccd22965-ed7a-50aa-9e57-30e96c9784a3/scratchpad/candidates/E3/code/run_e3.py`
(sha256 `abb0c99c9199a984`, recorded before the first demo course ran). Commit `fb1e80c8`.

**Rule (pre-registered).** A policy MEETS THE HEAT RULES AT 60 CM if C1 <= 90 C and C2 <= 0.80 on **both** of its demo
courses (seed0 700000 and 710000), at 35 C with today's drives (10 X4-36-L at 7.5 N m continuous, 2 X6-60 at 12 N m,
drive side; `sim/data/drives.json`). C1 is the hottest winding over the demo at the 8 min time constant. C2 is the
worst drive's loop RMS over its continuous rating; the loop is 90 s walking at 0.25 m/s plus 30 s standing, repeated
for one pack's runtime. This reads each policy on its own. It is not the study's worst-of-four criterion.

## Result

**5 of the 11 policies meet the rule:** fore128-s0 and fore128-s1 on the adopted model, run3-s0 and run3-s1 (188 mm
foreleg), and run2b (box foot with a passive roll ankle). The last three are on models other than the adopted one.
**6 do not meet it:** fore128-s2, fore128-s3 and all four fore160 policies. All 22 demo courses are valid (no
segment fell).

| Policy (model) | C1 at 700000 / 710000, C | C2 at 700000 / 710000 | Deciding drive (mirror's ratio at 700000 / 710000) | Even split C2 / C1 (desk, not a pass) | MEETS |
|---|---|---|---|---|---|
| fore128-s0 (adopted) | 76.31 / 74.91 | 0.7613 / 0.7500 | left hind pitch (0.6324 / 0.6205) | 0.6998 / 68.82 C; 0.6883 / 67.59 C | **yes** |
| fore128-s1 (adopted) | 74.38 / 76.41 | 0.7459 / 0.7620 | left fore pitch (0.5851 / 0.5971) | 0.6820 / 66.76 C; 0.6967 / 68.29 C | **yes** |
| fore128-s2 (adopted) | 93.27 / 91.04 | 0.8812 / 0.8671 | left fore pitch (0.5442 / 0.5387) | 0.7324 / 72.63 C; 0.7218 / 71.41 C | no (C1, C2 on both) |
| fore128-s3 (adopted) | 98.28 / 97.34 | 0.9137 / 0.9085 | left hind pitch (0.5090 / 0.5092) | 0.7396 / 73.43 C; 0.7365 / 73.04 C | no (C1, C2 on both) |
| fore160-s0 (160 mm) | 123.24 / 120.84 | 1.0326 / 1.0217 | right hind pitch (0.6354 / 0.6244) | 0.8573 / 89.40 C; 0.8467 / 87.81 C | no (C1, C2 on both) |
| fore160-s1 (160 mm) | 97.19 / 96.55 | 0.9066 / 0.9026 | right fore pitch (0.7271 / 0.7188) | 0.8217 / 84.15 C; 0.8159 / 83.34 C | no (C1, C2 on both) |
| fore160-s2 (160 mm) | 92.59 / 90.10 | 0.8814 / 0.8666 | right hind knee (0.8362 / 0.8242) | 0.8591 / 88.77 C; 0.8457 / 86.87 C | no (C1, C2 on both) |
| fore160-s3 (160 mm) | 80.63 / 83.26 | 0.7955 / **0.8198** | right fore pitch at 700000 (0.5376); left hind pitch at 710000 (0.6607) | 0.7285 / 71.64 C; 0.7445 / 73.31 C | no (C2 on 710000) |
| run3-s0 (188 mm) | 77.90 / 79.23 | 0.7775 / 0.7872 | right fore pitch (0.7001 / 0.6878) | 0.7449 / 73.50 C; 0.7525 / 74.51 C | **yes** |
| run3-s1 (188 mm) | 71.96 / 71.66 | 0.7248 / 0.7223 | left hind pitch (0.6838 / 0.6987) | 0.7046 / 69.67 C; 0.7106 / 70.34 C | **yes** |
| run2b (box foot, roll ankle) | 62.13 / 61.17 | 0.6373 / 0.6277 | left hind knee (0.5201 / 0.5120) | 0.5817 / 57.23 C; 0.5728 / 56.52 C | **yes** |

- The C1 drive is the C2 drive on every course. Over the 4-15 min time constants the meeting policies stay under
  90 C: fore128-s0 74.57-77.36 C, fore128-s1 73.98-77.48 C, run3-s0 77.71-79.98 C.
- **Margins of the meeting policies, the worse of their two courses:** fore128-s0 C2 0.7613 and C1 76.31 C;
  fore128-s1 0.7620 and 76.41 C; run3-s0 0.7872 and 79.23 C (1.6 % under the C2 line); run3-s1 0.7248 and 71.96 C;
  run2b 0.6373 and 62.13 C.
- **The two courses differ by -0.0148 to +0.0242 in C2 and -2.49 to +2.63 C in C1.** fore160-s3 is the only policy the
  second course changes. Its left hind pitch was already at 0.7875 at 700000 (standing RMS 6.18 N m), just under its
  right fore pitch's 0.7955. At 710000 it is at 0.8198 (standing RMS 7.08 N m).
- **On the adopted model the deciding load is still one-sided on the two policies that meet.** fore128-s0's left hind
  pitch is at 0.7613 / 0.7500 and its mirror at 0.6324 / 0.6205. fore128-s1's left fore pitch is at 0.7459 / 0.7620 and
  its mirror at 0.5851 / 0.5971.
- Each drive's loop RMS over continuous, its walking and standing RMS (N m) and its hottest winding at 35 C, for all
  22 courses: `e3_per_drive.csv`. Headline per course: `e3_policy_heat.csv`. Everything, with provenance:
  `e3_summary.json`.

## Checks of the harness (they test the code, not the rule)

- **The stored courses through `policy_heat` give the published figures.** C1 at 35 C: 76.3082 / 74.3768 / 93.2673 /
  98.2779 C (fore128 s0-s3) and 123.2381 / 97.1875 / 92.5905 / 80.6325 C (fore160 s0-s3), against
  `rl/analysis/REPORT_HEAT_35C.md`'s 76.3 / 74.4 / 93.3 / 98.3 and 123.2 / 97.2 / 92.6 / 80.6 C (all 8 within 0.05 C).
  The deciding drive, its mirror, the ratios and the even split agree with `design/FORM_FACTOR_STUDY_50_80CM.md`
  section 1.4 for all four fore128 policies (for example s3: 0.9137, mirror 0.5090, even split 0.7396 and 73.432 C,
  against 0.914, 0.509, 0.740 and 73.4 C).
- **The two thermal paths agree.** C1 at 25 C by `policy_heat`'s path equals each demo course's own 25 C hottest
  winding (`demo_course.run`) on all 22 courses (largest difference 0.0 C).
- **Determinism.** The four fore128 courses at seed0 710000 have the same `tau_drive_sq_mean` and minutes as E2's
  block-B courses (`candidates/E2/demo/`, run separately).
- **Supporting, not part of the rule: the stored courses rerun.** The 8 stored courses (seed0 700000), rerun through
  `demo_make` (the sweep's loaders), give the stored `tau_drive_sq_mean` exactly (largest relative difference 0.0) and
  the stored minutes. So the stored courses and the new ones come from the same pipeline. The reproduction was in the
  runner before any E3 course ran.
- All 11 checkpoints passed `verify_checkpoint` (sha256 against their stored sweeps) before the run. The build files'
  sha256 equal `plan.json`'s (17 of 17).

## Readings and changes

- Readings added in the runner before any E3 course ran (the pre-registration is silent on them; none loosens it):
  (i) a course with a fallen segment would not count as meeting the rule; it never applied, since all 22 are valid.
  (ii) The courses ran at `demo_course.run`'s default room, as E2's did; the room enters only the course's own 25 C
  field, never its torques or minutes, and C1 and C2 were computed at 35 C by `policy_heat`. (iii) "Each drive's RMS
  over continuous" is the loop RMS that `policy_heat` gives.
- **Nothing was changed after a result.** No course was rerun to improve a result.

## What it means

- **For A (60 cm, today's drives).** Two existing policies on the adopted model, fore128-s0 and fore128-s1, meet C1 and
  C2 at 35 C on both demo courses. Their worse C2 is 0.761-0.762, about 5 % under the line, and their worse C1 is
  76.3-76.4 C. So A has an existing policy that meets the heat rules on its own; choosing one is the owner's decision.
  This does not make the study's worst-of-four criterion pass: s2 and s3 still fail on both courses. It is a heat
  reading only. It says nothing about #9 and #10 (E1), the trot (E5), the policies' 11 simulator-only inputs (D46) or
  a shell.
- run3 x2 and run2b also meet, but on other models: the 188 mm foreleg found unpackageable (F18) and the box-foot roll
  ankle. They show that 60 cm policies with more heat margin exist, not that the adopted model has one. The 160 mm
  model does worse (0 of 4).
- **For B (50 cm).** No reading: E3 is a 60 cm experiment.
- **What it does not show.** The thermal model is open air with no shell, no housing node and no drive electronics, and
  a shell would raise every figure. The demo loop is straight, with no turns, starts or stops. The thermal time
  constant is a 4-15 min range. Nothing is measured: the bench heat test is the way to know.
