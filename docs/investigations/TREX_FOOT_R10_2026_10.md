# T. rex physics r10: the load-bearing tridactyl foot

**Date:** 2026-10-16

**Status:** carried out as T. rex physics r10 with the whole-foot stance rail
(decision D-D34; `configs/plant_versions.toml` note 20). The maintainer
approved it as a draft PR, held until a training screen shows the new foot at
least as good as the r9 foot.

On physics r9 each T. rex foot stood on a 14 x 9 cm plantar pad, with three
near-parallel digits on 40 N·m/rad springs that carried 18% of the statue's
load near their bases. The stance gate's fore-aft rail read the pad alone
(`max_sole_cop_fore_aft`, D-D28). r10 replaces the pad with a sole box under
the metatarsal pad and three splayed, load-bearing digits, and moves the
rail to the whole foot (`max_foot_cop_fore_aft`, #622's key) on a declared
span. This note summarises the evidence. The full scratch record (proposal,
two review rounds, the zero-shot and short-toe checks) was not committed.

## 1. What changed

| | r9 | r10 |
|---|---|---|
| Sole | pad 0.070-0.210 m ahead of the ankle, 9 cm wide | box 0.060-0.165 m, 11 cm wide, 6 mm lateral |
| Digits II/III/IV, MTP joint to claw tip | 0.157 / 0.218 / 0.176 m | 0.171 / 0.212 / 0.178 m |
| Divarication II-III / III-IV (chords) | 8.7 / 7.6° | 25 / 33° |
| MTP springs, N·m/rad | 40 / 40 / 40 | 3000 / 4500 / 3000 |
| Contact-line width / length (tip centres) | 0.39 | 0.78 (0.18 m / 0.232 m) |
| Digit II-II clearance at home | 0.140 m | 0.077 m |
| Statue load: sole / II / III / IV (median) | 0.83 / .065 / .047 / .073 | 0.67 / 0.15 / 0.09 / 0.12 |
| Stance CoP rail | pad CoP, 0.80 | whole-foot CoP on [0.060, 0.213] m, 0.80 |
| Pad CoP, `min_sole_contacts` | gated | report-only |

Unchanged: masses (animal 85.72 kg, foot 1.76 kg), every name and order,
the foot sites (stance width 0.280 m), the r15 toe touch sites, the
keyframe and the sibling exclusions, so the policy interface stays r15.
Physics 9 → 10 and visual 4 → 5. `MEASUREMENT_VERSION` moves to
`floor-truth/v3`, because the T. rex's registry entry (its span) changed
under an adopted stage, which `gait/constants.py` versions.

## 2. Why this foot

The proposal compared three variants on the scratch harnesses (V1 digits
only, V2 digitigrade, V3 the sole box with splayed digits) and chose V3 for
its mechanics, not for the literature-row count:

- **Robustness under noise.** Four r9-trained stance policies fall 0/80
  times under joint-noise vectors on V3 against 9/80 on r9 (Fisher p =
  0.003). The statue under the same vectors falls 6/60 against 30/60 (p =
  2e-6). Statue falls on five shared seed blocks are 0/200 against 4/200
  (p = 0.12, not significant).
- **No fore-aft rocking.** One policy's 23 mm CoP oscillation at 3.6 Hz on
  r9 becomes 5.3 mm at 0.13 Hz. V1 doubles it, and V2 turns it into a 64 mm,
  5 Hz limit cycle.
- **A modest toe-ward gain.** The forward push limit is 58.4 N against
  55.4 N (+5.4%; floor moment +4.6%), and the first nosedive under a held
  single-ankle offset comes 1.0-1.5° later. V1 (62.0 N) and V2 (60.8 N)
  resist more push; V3 leads only in CoP excursion.
- **Pitch compliance** is 1.3x r9's (V2 2.6x), and the mirror is exact.

Re-measured in this change: the statue stands on all 280 stance-panel
seeds and is clean on them; the KNOWN_ISSUES ankle probe (±0.03 on both
ankles) reaches the horizon on 25/40 and 40/40 (r9 14/40 and 24/40).

**What it does not show.** The proposal judged the r9-trained policies
zero-shot: on r10 two of three seeds still lean forward past the new rail
in some evaluations. Only training on r10 can show whether a policy trained
there avoids the lean, which is why the PR is held for the screen.

## 3. Anatomy, and where it comes from

Primary sources are marked (P), secondary (S).

- **Functionally tridactyl, digitigrade.** Digits II-IV bear weight. The
  hallux is small and set high, and MT V is a splint (Brochu 2003, P).
  Tyrannosaurid tracks show no metatarsus (McCrea et al. 2014, P). Neither
  the hallux nor MT V is modelled.
- **The sole box** stands for the metatarsal pad, "with a wide caudal
  margin", and for digits that are thick proximally, with a short free
  digit III (McCrea et al. 2014, P, Bellatoripes). "Metatarsodigital pad"
  is from Xing et al. 2019 (S, via Falkingham et al. 2026). The box is rigid
  from 0.060 to 0.165 m, across the MTP joints and the first IP joints,
  which flex in life: it is a stance approximation. The width and shape of
  the heel are not measured (Enriquez et al. 2021, read as an abstract
  only).
- **Digit lengths** apply Sue's digit / MT III ratios (FMNH PR 2081: Brochu
  2003 via Mortimer, Theropod Database, S; digit IV completed from
  Gorgosaurus) at this model's MT III. This is the top of the plausible
  range: at the leg scale they would be 0.157 / 0.194 / 0.163 m. They are
  kept for mechanics (§4). Sue's unguals (II 175, III 204 mm) come from
  Mortimer's compilation, not checked against Brochu's tables.
- **Divarication** of 25 / 33° is a chord angle, MTP joint to claw tip, and
  a compromise. The phalangeal axes of II and IV deviate progressively
  (Brochu 2003, P), so a straight capsule's chord sits below track angles.
  The Bellatoripes means are 29.7 / 35.7° (McCrea et al. 2014). The Hell
  Creek track attributed to T. rex, 35 / 40° (Falkingham et al. 2026, P), is
  not matched.
- **W/L.** At the contact line it is 0.78, inside Bellatoripes' 0.78-1.00
  and the T. rex-attributed 0.67-0.83. The Hell Creek R2 track is 0.83 (P).
  Tyrannosauripus at about 0.67 comes from secondary sources only. The
  claw-tip and outer-margin W/L (0.88, 0.99) are wider than every
  T. rex-attributed print.
- **Narrow trackways.** Tyrannosaurid prints overlap the midline, 40-50% in
  Falkingham et al. 2026, and turn slightly inward (McCrea et al. 2014).
  The digit II-II clearance of 0.077 m is therefore an anatomical
  constraint, not a defect (KNOWN_ISSUES).
- **Not modelled:** MTP stagger (MT II and IV are 13 and 7.5% shorter than
  MT III in Sue), a measured condyle spacing, interphalangeal joints and
  digit curvature.
- **Load is report-only.** Track depth is not load (Boeye et al. 2026: III
  and IV deeper than II, deepest around the proximal phalanges). The model's
  central digit carries the least.

## 4. Digit length and the springs: V3 against the leg-scale V3s

At the leg-scale lengths (V3s) the forward push limit is 49.8 N, below r9.
No spring rung recovers it: 51.8 / 54.0 / 55.2 N at 4x / 6x / 8x, the last
at ω_n·dt ≈ 0.69. V3s's held-offset first nosedive is -3.5 / -4.0°. The toe
margin scales at about 0.30 N of push, and 0.06° of held-offset tolerance,
per mm of digit III. So the toe-ward gain rests on the upper-bound lengths
and the chosen springs.

The springs (3000 / 4500 / 3000 N·m/rad) are the lowest rung of a ladder
(1x 43.8 N, 2x 54.0 N, 3x 58.4 N) whose push limit beats r9's by at least
5%. They are a mechanical choice, not a measurement, and they resist
toe-off (KNOWN_ISSUES).

## 5. The toe line

The span's heel line is the box's heel edge. Heel-ward held offsets end in
tail contact before the foot topples, so the heel side needs no measured
line. The toe line is measured on the statue by a fixed rule,
`environments/shared/scripts/foot_toe_line.py`:

1. Hold one ankle at an offset toward the toe, on a uniform 0.5° grid from
   -2.5°, ramped in over steps 0-199 and held.
2. Run seeds 3042-3046 at each level.
3. Read the other foot's window-mean whole-foot CoP, as the median over the
   seeds that do not fall.
4. The onset is the first level with a fall.
5. The line is the smaller onset of the two ankles, floored to 1 mm.

On r10 the onsets are -5.0° (214.84 mm) and -5.5° (213.44 mm), so the line
is 0.213 m. By the same rule r9's line is 0.196 m. Flooring moves both
plants less than 1 mm heel-ward of their onsets. Nearest-5 mm rounding
would have moved r10's 1.6 mm toe-ward and r9's 1.9 mm heel-ward. The
record (`environments/tyrannosaurus_rex/anatomical/data/foot_toe_line.json`)
pins the plant's physics digest, and `test_gait_morphology.py` pins
`FOOT_SUPPORT`'s toe line to it.

## 6. The rail and its bar

`max_foot_cop_fore_aft` = 0.80 on [0.060, 0.213] m, re-measured on r10 with
the repository's recorder (stance [env], reset noise 0.05):

| Condition | Reading |
|---|---|
| Statue 3042-3081 / out of sample (240) / 11042-11081 | at most 0.321 / 0.351 / 0.330 (median 0.153) |
| Statue + N(0, 0.05) command jitter, 60 episodes | at most 0.723, 0 above 0.80 |
| r9 run A's mean action held (forward lean), 20 episodes | 19 above 0.80, median 0.988 |
| Held single-ankle offsets (median of 5 seeds) | crosses 0.80 at -3.5 / -4.0°, first falls at -5.0 / -5.5° |

The bar trips 1.5° ahead of the fall on both ankles. The pad CoP reads the
10.5 cm box: up to 0.61 on the statue, up to 0.79 under jitter, and above
0.80 at every held-offset level. `min_sole_contacts` refuses front-edge box
stances while the digits carry the front of the foot. Both are report-only
for the T. rex; the compsognathus keeps its pad bar.

The tilt (2.0°) and corner-lift (4 mm) bars are kept. On the r10 box, 4 mm
is 2.18° of pure pitch and 2.08° of pure roll, so the tilt bar binds first
in pure pitch or pure roll. A diagonal tilt lifts a corner 4 mm at about
1.07° in each axis, so there the corner-lift bar binds first.

The whole-foot CoP is the vertical load's moment about the ankle axis over
the limb's floor force (PR #622). The ankle torque over that force is not
the same thing. `test_anatomical_foot.py` balances the torque against
three moments: the vertical load's, the floor shear times the contacts'
height below the ankle, and the foot's own weight. The residual is within
0.05 mm of CoP. Without the shear term, the torque misses by up to about
11 mm on the statue.

## 7. Open points

- **Settle touchdowns.** The r10 statue reaches `max_settle_touchdowns` = 2
  on 89 of 280 episodes (r9 at most 1). Its spawn leaves the floor on 17 of
  40 episodes, for up to 16 substeps, all inside the spawn grace (r9 none).
  The bar was kept, because D-D34 changed only the CoP rail. A policy that
  settles as the statue does and then re-plants once fails it on those
  seeds.
- **Digit loading** moves with the stance's roll. On some statue seeds one
  digit carries under 2% of its foot (digit IV on 47 of 160, digit II on
  35). The load split is report-only.
- **The recovery push** derives to 186.8 N on r10 (165.5 N on r7-r9),
  because the heel-ward support margin grows from 77.3 to 87.2 mm. The
  pushed statue recovers more of it (2/40 full horizon, 54 of 70 shoves).
  The frozen recovery gate was measured at 165.5 N and is re-frozen at the
  first r10 stance.
- **Toe-off, the rigid box and the touch site.** The stiff MTP springs
  resist toe-off. The rigid box spans the MTP and first IP joints, and the
  heel unloads at 26 N of forward push (r9 36 N). Digit II's r15 touch box
  leaves its tip cap uncovered past about 49° of tip-down pitch, which no
  measured episode reached. KNOWN_ISSUES has all three.
- **The window mean cannot see oscillation.** PR #622's CoP spread and
  dominant frequency are report-only.

## 8. Sources

- Brochu CA 2003. Osteology of *Tyrannosaurus rex*. SVP Memoir 7.
  doi:10.2307/3889334
- McCrea RT et al. 2014. A 'Terror of Tyrannosaurs'. PLoS ONE 9(7):
  e103613.
- Falkingham PL et al. 2026. A large tridactyl trackway from the Hell Creek
  Formation of North Dakota. J. Vert. Paleontol. e2706201 (citing Xing et
  al. 2019 for the Hekou track).
- Boeye AT et al. 2026. Evidence of bird-like foot function in
  *Tyrannosaurus*. R. Soc. Open Sci. 13: 252139.
- Mortimer M. The Theropod Database, Tyrannosauroidea (accessed
  2026-10-15).
- Hutchinson JR et al. 2011. PLoS ONE 6(10): e26037: foot mass 1.1-2.9% of
  body mass; the model's is 2.05%.
- Enriquez NJ et al. 2021. J. Vert. Paleontol. (abstract only): heel
  breadth grows with size.
