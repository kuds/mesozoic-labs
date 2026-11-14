# What makes a walking dinosaur look alive: research for Stage 2

10 Nov 2026. Research for the Stage 2 plan. This is simulation-side research only, and nothing here is adopted: P24
applies to every proposal. Sources are linked where they are used and listed at the end. Numbers marked **est.** are
my own arithmetic. They come from the round-two models (the `60-N-T-10` and `75-M-H-10` MJCF inertial blocks under
`round2/refine/narrow_carpet_60` and `round2/refine/feet_75`) and from the candidate records' drive margins, using rigid
bodies and simple motion profiles (script: `stage2/research/loads.py`, copied from the session scratchpad on 14 Nov 2026). Nothing was
simulated, fitted, trained or swept for this note.

`H` is hip height: 0.405 m on `60-N-T-10` and 0.502 m on `75-M-H-10`. Fr = v²/(gH) is the Froude number. "Nominal",
"brisk" and "animal pace" are the study's Froude 0.012, 0.025 and 0.05 walks. A1 is the study's sustained-torque
planning limit (0.57 × the vendor's rating).

## Bottom line

1. **The reference walk looks like a ZMP humanoid robot, and it was built that way.** `designlab/gait.py` is a sizing
   walk. It keeps the hips at a constant height, so in every step the stance knee has to bend and straighten 0.08–0.15
   rad just to hold the hips level (est.). The trunk stays level and rigid while it slides 0.164–0.169 H sideways in each
   stride. The feet lift and land parallel to the floor. At slow speed the swing stretches into slow motion, and the head
   nod and tail swing run on the stride clock with fixed phases, whatever the body is doing. Each of these traits has a
   documented animal counterpart and a cheap fix in the reference (section 1).
2. **The best value-for-cost changes** (section 3):
   - a pendular hip bob: the hips dip by up to 0.014 H (nominal) or 0.026 H (animal pace) in double support, which is
     what the flat walk removes today;
   - a fixed swing time of about 0.32 s at 60 cm and 0.36 s at 75 cm, so slow walks spend longer in double support. The
     nominal walk's duty factor rises from 0.60 to about 0.69;
   - a toe-off roll about the three toe pads' front edges, 0.3–0.5 rad by lift-off;
   - a head held still against trunk rotations instead of nodding;
   - a tail phase that works against the legs' yaw momentum, which also reads as follow-through.
3. **What not to add:** an imposed trunk pitch oscillation. ±1° at the step frequency costs 0.85–1.7 N·m of hip pitch
   at 60 cm and 1.5–3.0 N·m at 75 cm (est.). Also avoid large head and tail swings at the stride frequency. The plant
   gives every joint the MJCF's default damping (0.2 N·m·s/rad) and friction loss (0.15 N·m). Stage 0's inverse
   dynamics leave these out (`designlab/stage1/refs.py`). Those two terms alone give the tail joint about 0.24–0.29 N·m
   RMS at 0.12 rad (est.), and a 0.2 rad swing at the animal pace reaches 0.88–0.99 of A1. That is probably a large part
   of why the tail's closed-loop load is 2.7–2.8 times Stage 0's (Stage 1b §6).
4. **A finding to check (free to fix).** Reading `gait.py`'s conventions (left foot in swing for phase 0–0.4; a positive
   `tail_yaw` swings the tail tip to the robot's right), the reference tail moves toward the swinging leg's side in
   each swing. That adds to the legs' yaw momentum instead of cancelling it, by about 15 % at 60 cm and 20 % at 75 cm
   (est.). The stance foot's yaw moment is the friction limit that binds (F21). The turning screen's straight runs can
   confirm the sign in minutes.
5. **Big-animal timing.** At the animal pace the robot takes 2.75 / 2.47 steps/s with 0.80 H strides. A dynamically
   similar T. rex would take 1.83 / 1.64 steps/s with 1.25 H strides (van Bijlert et al. give 1.28 m/s at a 3.1 m hip,
   which is Fr 0.054). Short, quick steps read as small and busy. The knee's speed limits how far strides can grow (F25).
6. **Sway is a hardware-limited part of the look.** Theropod trackways are very narrow. The 60 cm feet are 0.36 H
   apart, which gives 0.17 H of sway at the nominal walk. Getting to 0.12 H by moving the ZMP inward would use up most of
   the inner centre-of-pressure margin. A better option is a 1–2° roll, so the sway reads as a weight shift rather than a
   slide.
7. **A cheap look check before any controller work** (section 4):
   - Render each reference variant kinematically with the existing offscreen renderer (`designlab/render.py`), at 1×
     and from a visitor's viewpoint. Show it beside the old walk, a time-scaled bird clip and a point-light version.
   - Price each variant with the Stage 0 inverse-dynamics screen.
   - The owner reviews with a seven-item rubric and the Godspeed animacy items.

   The whole check takes hours of CPU, not days.
8. **After control,** rate closed-loop clips blind, with the realistic servo delay: at least 12 viewers who are new to
   the project, randomised clips and anchor clips. The pass is a mean of at least 3.5/5 (the spec) and no more than
   about 0.5 below the same behaviour's reference render. Machine metrics show where the look was lost.
9. **At a slow pace, most of the character comes from the head:**
   - gaze: looking at people, a saccade-and-hold rhythm, and the head starting a turn 0.2–0.4 s before the body;
   - idle micro-motion;
   - a display: an open-mouth roar with sound, or a closed-mouth "boom", which has some scientific basis.

   Neck yaw is the most valuable joint to add. The jaw comes second (section 5).
10. **Loads:**
    - **Neck yaw:** an STS3215 at the neck base needs 0.05–0.14 N·m (60 cm) or 0.12–0.32 N·m (75 cm) of inertial
      torque for 0.6 rad gaze shifts in 0.5–0.8 s (est.). With the plant's default joint damping and friction, a 0.5 s
      shift at 75 cm goes over the STS3215's planning peak. At 75 cm, use shifts of 0.8 s or more, or a stronger servo.
    - **Jaw:** limited by speed, not torque. It needs at most 0.22 N·m, but 0.6 rad takes at least 0.4 s at the
      STS3215's planning speed of 2.83 rad/s.
    - **Neck pitch:** a jaw servo in the head raises the 60 cm neck-pitch hold from 0.75 to 0.84 of A1. If the yaw
      servo also sits in the head, it reaches 0.92. Put the yaw at the neck base, and consider a spring that carries the
      head's weight.

## 1. Why today's reference walk reads as mechanical

| What `gait.py` does | Why it reads as a machine | What animals and the sources show |
|---|---|---|
| Constant hip height: `q[2] = root0[2] - crouch·H` in every frame | As the hips pass over the stance foot, the leg must be shortest at mid-stance. The stance leg shortens 5.8 mm (0.014 H) in the nominal walk and 10.7 mm (0.026 H) at the animal pace (the same fractions at 75 cm). That is about 0.08–0.15 rad of knee bend and straightening in every step (est.). This is the bent-knee, flat-centre-of-mass look of ZMP humanoids. The linear inverted pendulum model assumes a constant height ([Kajita et al. 2003](https://people.csail.mit.edu/katiebyl/kb/DW2008/papers_of_tangential_interest/kajita03.pdf)), and humanoid groups add stretched knees with heel contact and toe-off to look human ([Ogura et al. 2006](https://researchr.org/publication/OguraSKMOMLT06/bibtex)). | Walking animals vault over the stance leg, with the centre of mass highest at mid-stance in slow walks ([Hancock et al. 2007](https://people.ohio.edu/stevensn/documents/Hancock%20et%20al%202007%20Tinamous%20early%20view.pdf)). Humans move it 3.3–3.7 % of leg length ([Smith et al.](https://katalog.lib.cas.cz/KNAV/EdsRecord/edsair,edsair.doi.dedup.....5711e2752a1f2d3be6fabfdd42692807)). In van Bijlert et al.'s simulation of T. rex walking pendularly, the base of the tail moves 0.08 m in amplitude at a 3.1 m hip, about 0.05 H peak to peak ([van Bijlert et al. 2021](https://pmc.ncbi.nlm.nih.gov/articles/PMC8059583)). Flattening the centre of mass costs more joint work ([Ortega & Farley 2005](https://scilit.com/publications/29280cbdf8609328df0f3ab2834e6f1b); [Gordon, Ferris & Kuo 2009](https://scilit.com/publications/8760e02900afeeb5520b48baad00b729); [Kuo 2007](https://scilit.com/publications/0bcb4397020cc31ca5a12a9cba7cb509)). |
| Level, rigid trunk: `q[3:7] = q_home[3:7]` | The body slides 0.164–0.169 H sideways in each stride with no roll, like a cart (STAGE1_PLAN decision 9). | Theropod trackways are "very narrow", with pace angulation up to 180° ([Botucatu trackways](https://igeo.ufrj.br/inc/isc/3/03-133-%20Farlowichnus%20Leonardi%20et%20al%202023.pdf)), so a theropod hardly swayed. The animals that waddle are wide-legged ducks ([Necker](https://reinhold-necker.de/seite10.html)). Animators drop the unsupported hip slightly ([community T. rex critique](https://blenderartists.org/t/t-rex-test-animation/1444558); secondary source). |
| Flat feet: foot targets keep the home orientation (`np.eye(3)`) | The feet lift and land parallel to the floor, the scripted-humanoid signature. | In birds the toes lie flat after contact, then roll up and forward from base to tip, and the middle toe pivots on its claw ([Falkingham & Gatesy 2014](https://cs.brown.edu/courses/cs137/2015/readings/Gatesy_XROMM_DEM_PNAS14.pdf)). OP3's scripted walk keeps the foot plates nearly parallel to the ground, while its learned walk pushes off the edges of the foot ([Haarnoja et al.](https://ar5iv.labs.arxiv.org/html/2304.13653)). Without heel-toe motion, Disney's Olaf looked "more robotic" ([Olaf](https://arxiv.org/html/2512.16705v1)). |
| Fixed duty factor (0.60 at the nominal walk) | Swing time grows with the stride period: 0.41 s in the nominal walk against 0.32 s at the animal pace on the 60 cm body. A slow walk therefore swings in slow motion; the swing foot's peak vertical acceleration is about half of g (est.). | In birds the swing time hardly changes with speed, and slower walks lengthen the stance ([mallard](https://peerj.com/articles/15362); [wild turkey](https://sicb.org/?p=25566)). Tinamous walking pendularly use a duty factor of 0.64 ± 0.05 (range 0.55–0.70). The visual "life detector" responds to the feet's vertical acceleration ([Chang & Troje 2009](https://www.yorku.ca/research/biomotionlab/?p=536); [Troje & Westhoff 2006](https://www.biomotionlab.ca/the-inversion-effect-in-biological-motion-maybe-there-are-two-of-them/)). |
| Swing lift `lift·sin²(πu/sw)` with its peak at mid-swing | A symmetric arch, where the rise and the fall mirror each other. | A natural foot rises quickly after toe-off and then descends more slowly. The forward motion is already right: the world-frame minimum-jerk path ends at zero ground speed, which gives a mild backward swing of the foot relative to the hip just before touchdown. That retraction is what makes walking models stable ([Wisse et al. 2005](https://research.tudelft.nl/en/publications/swing-leg-retraction-helps-biped-walking-stability/)). |
| Head nod `0.05·sin(2ωt)`; tail `0.12·sin(ωt + π/2)` | Both run on the stride clock at fixed phases, whatever the body does: a metronome. By my reading of the code, the tail phase adds to the legs' yaw momentum (F9 in section 3). | Pigeons on a moving perch keep the head "almost perfectly stabilized against rotations" while following translations ([Theunissen & Troje 2017](https://www.frontiersin.org/journals/neuroscience/articles/10.3389/fnins.2017.00551/pdf)). In ungulates the head moves out of phase with the withers (cited by van Bijlert et al.). Tails lag the body (follow-through), and in T. rex the tail heaves vertically at the step frequency (van Bijlert et al.). |
| Identical, mirror-image strides | Perfect repetition and left-right "twins" look lifeless. | The animation principles: no twins, overlapping action, arcs, slow in and slow out, anticipation ([Thomas & Johnston's twelve principles](https://en.wikipedia.org/wiki/Twelve_basic_principles_of_animation)). |

The look gaps in Stage 1's gap map, G7 (calm body motion: sway 0.176–0.180 H) and G8 (livelier head and tail: both
at the low end of their bands), were both traced to the reference walk, and so were the start and stop accelerations.
In steady walking the reference already peaks at 2.1–2.2 m/s² (STAGE1_PLAN decision 10), and F25's walks searched at
the animal pace stay near 3 m/s².

## 2. What the sources say, in brief

**Pace, stride, duty factor and cadence**
- T. rex's preferred walking speed from its tail's natural frequency is 1.28 m/s (range 0.80–1.64), from a step
  frequency of 0.66 Hz (0.41–0.84) and a 1.94 m step at a 3.1 m hip. That is Fr 0.054, so the study's "animal pace" is
  this estimate. Bipedal dinosaurs "seem to have preferred to walk at a relative stride length of 1.3"
  ([van Bijlert et al. 2021](https://pmc.ncbi.nlm.nih.gov/articles/PMC8059583)).
- At robot scale (time scaled by √(H ratio)), that step frequency becomes 1.83 Hz at 60 cm and 1.64 Hz at 75 cm (est.).
  The robot's animal-pace walk takes 2.75 and 2.47 steps/s.
- Birds' kinematics change continuously with speed. Their duty factors stay above 0.5 up to high relative speeds, and
  the whole-body centre of mass sits well ahead of the hips ([Bishop et al. 2018](https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0192172)).
- Tinamous vault (walk pendularly) below about Fr 0.4–0.6, with a duty factor of 0.64 ± 0.05. Their relative stride
  fits 2.14·Fr^0.26: 0.68 H at Fr 0.012 and 0.98 H at Fr 0.05 if extrapolated ([Hancock et al. 2007](https://people.ohio.edu/stevensn/documents/Hancock%20et%20al%202007%20Tinamous%20early%20view.pdf)).
- Alexander's relation, already in `gait.py`'s docstring, gives 0.61 H and 0.94 H. The robot uses 0.55 H and 0.80 H.
- Swing time is nearly constant across speed in birds; slower walks lengthen the stance
  ([mallard, PeerJ](https://peerj.com/articles/15362); [wild turkey](https://sicb.org/?p=25566)).

**Legs**
- T. rex models favour "more upright (although not completely columnar)" limbs ([Hutchinson et al. 2005](https://www.cambridge.org/core/journals/paleobiology/article/analysis-of-hindlimb-muscle-moment-arms-in-tyrannosaurus-rex-using-a-threedimensional-musculoskeletal-computer-model-implications-for-stance-gait-and-speed/30DD414511DEB56D79A5C6281A284320)).
- Chickens raised with artificial tails stood with a more upright femur (38° against 27° from horizontal) and swung it
  about three times as far in stance (44° against 15°). In other words they walked hip-driven, as non-avian theropods
  probably did ([Grossi et al. 2014](https://pmc.ncbi.nlm.nih.gov/articles/PMC3915051)).
- The caudofemoralis, a muscle anchored on the tail, was the main femur retractor ([Persons & Currie 2011](https://doaj.org/article/27a525d1b73b477daf599d43d68bac7d)).
- The robot's home pose fits this picture: the femur is 22° from vertical and the knee about 34° flexed, on both bodies
  (computed from the MJCF).

**Feet**
- Toes are flat after contact, then roll up from base to tip, and digit III pivots on its claw tip
  ([Falkingham & Gatesy 2014](https://cs.brown.edu/courses/cs137/2015/readings/Gatesy_XROMM_DEM_PNAS14.pdf)).
- The robot's feet are already a heel (MTP) pad plus three toe pads, so a roll about the toe pads' front edges will
  look right.

**Head**
- Head-bobbing is a hold-and-thrust cycle seen in pigeons, galliforms, cranes and herons, not in many other orders. Its
  main job is visual ([Necker](https://reinhold-necker.de/seite10.html)).
- What birds share is that head rotation is held still while translation is followed ([Theunissen & Troje 2017](https://www.frontiersin.org/journals/neuroscience/articles/10.3389/fnins.2017.00551/pdf)).
- A pigeon-style forward head thrust would look comic on a T. rex. Stabilising the head's rotation would not.
- When steering, humans turn the head toward the new direction about 200 ms before the body turns
  ([Grasso et al. 1996](https://vufind-test.katalog.k.utb.cz/SummonRecord/FETCH-LOGICAL-c3295-285d249ebde34c5be285545fc4b568b8ab5e3f3143c3f2302b544b80dc85c36b2);
  [Grasso et al. 1998](https://acnpsearch.tweb-dev.unibo.it/singlejournalindex/58514855)).

**Tail**
- The T. rex tail was held up passively by interspinous ligaments. Its natural mode is a vertical heave at the step
  frequency (van Bijlert et al.).
- Tyrannosaurids had low rotational inertia and could turn faster than other large theropods. One proposed turning
  mechanism, as in cheetahs, is to bend the tail sideways and counter-bend the body ([Snively et al. 2019](https://pmc.ncbi.nlm.nih.gov/articles/PMC6387760/)).
- Creature animators start tail motion at the base, let it travel toward the tip, and keep the head and tail roughly
  steady while the hips turn (community source, secondary: [Blender Artists](https://blenderartists.org/t/t-rex-test-animation/1444558)).

**Perception**
- People recognise living motion from a few moving points ([Johansson 1973, via a summary](https://en.wikipedia.org/wiki/Biological_motion)).
- The cue that carries it is the feet's local motion, especially their vertical acceleration ([Chang & Troje 2009](https://www.yorku.ca/research/biomotionlab/?p=536)).
- So the feet matter more than the robot's covers suggest.

**Character robots**
- **Disney's bipedal character (BD-X)** ([Grandia et al. 2025](https://arxiv.org/html/2501.05204v1)):
  - Artists author the walk cycles with "a procedural gait generation tool", and reinforcement-learning policies are
    trained to imitate them.
  - There are separate standing (perpetual), walking (periodic) and episodic policies. Each is conditioned on head and
    torso commands.
  - The animation engine has a looped idle "background" layer, and gaze is driven by the operator's stick. The torso
    follows when the neck reaches its limits.
  - Show functions blend in 0.1 s and the body in 0.35 s, "such that the facial expressions … naturally blend faster
    than the body animation".
  - Walking tracks with 0.123 rad mean absolute joint error, at up to 0.7 m/s and 1.8 rad/s.
  - Each policy trains for 100,000 iterations, "around 22 days" on one RTX 4090. "1500 iterations or 30 min are enough
    to preview the nominal behavior".
- **Disney's Olaf** ([arXiv 2512.16705](https://arxiv.org/html/2512.16705v1)):
  - The heel-toe walk cycles come from the same kind of gait tool.
  - An impact reward lowered the mean sound level by 13.5 dB.
  - The neck-pitch actuator is the one most prone to overheating: it reached 100 °C within 40 s without the thermal
    reward. Disney fixed this with a thermal model and a control-barrier reward, with the temperatures given to the
    policy as inputs.
- **Disney's gaze work** ([Pan et al. 2020](https://la.disneyresearch.com/publication/realistic-and-interactive-robot-gaze/);
  [IEEE Spectrum](https://spectrum.ieee.org/disney-research-robotic-gaze)) uses saccades, head motion from simulated
  breathing, blinks, and habituation to distracting sounds.
- **DeepMind's OP3 soccer** ([Haarnoja et al.](https://ar5iv.labs.arxiv.org/html/2304.13653)):
  - The learned walk reached 0.57 m/s against the scripted 0.20 m/s, and turned at 2.85 rad/s against 0.71 rad/s.
  - Training randomised observation delays of 10–50 ms.
  - Its skill teachers needed 2.4 × 10⁸ to 2 × 10⁹ steps, roughly 100–800 times Stage 1's PPO runs.
- **Imitating animal motion capture** gives robots natural gaits ([Peng et al. 2020](https://xbpeng.github.io/projects/Robotic_Imitation);
  [Escontrela et al. 2022, adversarial motion priors](https://arxiv.org/abs/2203.15103)).

## 3. Answer 1: reference-walk features to add or change

Each row names a proposed `GaitSpec` field (none exists yet), a typical value from the sources scaled to H, a starting
value for the screen, and the expected cost. Costs are estimates. The Stage 0 screen confirms them (section 4.1).

### 3.1 Features

| # | Feature | Parameter (proposed) | Typical value (source) → starting value | Expected cost (est.) | Controller note |
|---|---|---|---|---|---|
| F1 | **Pendular hip bob** (vaulting): hips at today's height at mid single support, dipping in double support; two cycles per stride | `bob` (dip as a fraction of H, peak to peak), `bob_phase` | Humans 3.3–3.7 % of leg length; T. rex model about 0.05 H. The leg shortening the flat walk enforces is 0.014 H (nominal) and 0.026 H (animal pace). Screen 50 % and 100 % of that: **0.007–0.014 H nominal, 0.013–0.026 H at the animal pace** (3–6 mm and 5–11 mm at 60 cm; 4–7 mm and 7–13 mm at 75 cm). | Should remove most of the 0.08–0.15 rad of forced stance knee bend. Mid-stance knee extension is no greater than today's, so the singularity guard is unchanged. The vertical acceleration is at most ±0.4 m/s² nominal and ±1.6 m/s² at the animal pace at the 100 % setting (sinusoidal equivalent; half that at 50 %). A pendular profile puts it mostly in double support, where both legs share it. Stance knee and hip torque should be neutral to slightly lower. Swing speeds are unchanged. | The generator's multibody ZMP loop absorbs the height change. The tracker's DCM uses a fixed ω0 = √(g/z_c), which is off by at most about 1–1.5 % (negligible). |
| F2 | **Weight acceptance ("down")**: the bob's lowest point just after touchdown, a quick dip then a slow rise | `bob_skew` (time of the trough, fraction of stance) | Animation's "down" position. **Trough at 10–15 % of stance, settling over 0.1–0.15 s.** | A small knee yield under load in early stance, giving a short rise in knee torque. | It reads as weight without a louder foot impact. |
| F3 | **Swing time instead of a fixed duty factor**: slow walks lengthen stance and double support | `swing_time` (s); duty = clip(1 − swing_time/T, 0.55, 0.72) | Birds keep the swing time nearly constant; tinamous walk at duty 0.55–0.70. Use **the animal-pace walk's swing time: 0.32 s at 60 cm, 0.36 s at 75 cm** (about 1.6·√(H/g)). Nominal duty becomes 0.69 (from 0.60); brisk 0.60 (from 0.58). | Swing joint speeds in the nominal walk rise ×1.28. The 60 cm knee goes from about 4.2 to 5.4 rad/s, still under the animal pace's 7.0. Brisk rises ×1.05; the animal pace is unchanged. Longer double support makes the weight transfer calmer and trims sway by about 5 %. | Needs the spec's duty band (0.55–0.65) widened at slow speed: an owner decision. The transitions (preview control) must be regenerated. |
| F4 | **Swing lift shape**: rise quickly, descend more slowly | `lift_peak` (fraction of swing at peak height) | **Peak at 35–40 % of the swing** (50 % today). Keep the world-frame minimum-jerk forward motion: it already gives zero ground speed at touchdown and a mild retraction relative to the hip (Wisse et al.). Keep near-zero touchdown speed for quiet feet (Olaf's −13.5 dB). | Early-swing hip and knee speeds rise about 10–20 %. At the animal pace the knee is the binding drive (0.93–0.95 of its speed with the 1.15 margin), so this needs F25's gait-tuned walk (knee 0.69–0.75) there. No cost at the nominal walk. | Helps the "life detector" cue. Retraction makes the walk more tolerant of touchdown timing errors (relevant to servo delay). |
| F5 | **Toe-off roll and toe motion**: the MTP ("heel") pad lifts first, the toe plate pivots about the toe pads' front edges, the foot levels by mid-swing, an optional slight toes-up reach before landing, and a flat landing | `toe_off` (rad), `toe_off_start` (fraction of stance), `toe_reach` (rad) | Toes roll up from base to tip, and digit III pivots on its claw (Falkingham & Gatesy). The learned OP3 walk pushes off the foot edges; Olaf's heel-toe. **0.3–0.5 rad by lift-off, starting at 80–85 % of stance (in double support); reach 0.05–0.10 rad.** | Ankle pitch speed rises by about 2–4 rad/s in late stance at the animal pace. That is fine on the 60 cm ankle drives (speed ratio 0.27) but tight on the 75 cm X4-36 ankles (0.71). Ankle torque is small because the foot is unloading. Toe clearance in early swing needs a check, since the toes leave last and could drag on carpet. | The support shrinks to the toe line. The tracker's ankle cap (pad polygon with a 5 mm inset) and the estimator's contact point both need a rolling-contact model during the roll, or the cap will hold a support that is not there. |
| F6 | **Trunk roll with the weight shift**: the body banks with the sway instead of sliding | `roll_amp` (rad, stride frequency), `roll_phase` | Human pelvic obliquity is 7–9° in range; the spec allows ≤ 4° RMS. **Roll 1–2° (0.017–0.035 rad).** The direction is for the owner's review: banking toward the stance leg (weight shift), or dropping the swing side (as in human pelvic list). | Trunk roll inertia is about 0.015 kg·m² at 60 cm, so under 0.05 N·m of hip roll. It uses hip-roll range; inward motion is limited by toe contact at −0.11 to −0.21 rad (F6), so check the walking clearance gates. | The trunk PD tracks the reference attitude, so it needs only a reference change. |
| F7 | **Less sway**: move the single-support ZMP toward the pad's inner edge, or narrow the steps | `zmp_inset_y` (m), `step_width` (fraction of H) | Narrow theropod trackways. Sway (peak to peak) ≈ 0.47 × the foot spacing at the 60 cm nominal walk, so each 10 mm of inward ZMP per side removes about 9 mm (0.023 H). Reaching 0.12 H needs about 20 mm per side. **Screen 8–12 mm** (to about 0.14–0.15 H). | The inner centre-of-pressure margin falls from about 30 mm to 18–22 mm. The pad-edge flag came before every Stage 1/1b fall, so this lever is risky. Narrower steps are limited by leg clearance (F16). | At the nominal walk, prefer F6 (make the sway read as a weight shift) and accept 0.14–0.17 H. Faster walks sway less (0.09–0.10 H at the animal pace). |
| F8 | **Trunk pitch: none imposed** (or at most ±0.5°); the body's life comes from F1 and F6, and from the head and tail | `pitch_amp` = 0 | Bird trunks pitch a few degrees, but the cost is high on this body. | ±1° at the step frequency costs 0.85 / 1.68 N·m of hip pitch (nominal / animal pace) at 60 cm, 16–20 % of the walk's hip-pitch peaks. At 75 cm it is 1.50 / 2.96 N·m, 12–18 % (upper-body pitch inertia about the hips: 0.32 and 0.70 kg·m²). | If any pitch is used, choose the phase in the screen that lowers hip-pitch torque. |
| F9 | **Tail against the legs' yaw momentum, with follow-through**: the tail tip moves toward the stance side during each swing, and settles at stops | `tail_amp`, `tail_phase`, `tail_settle` | Animators lag the tail behind the hips, and tails counter-swing in turns (Snively et al.). **0.10–0.15 rad at the stride frequency.** The phase is the one that minimises the stance foot's yaw moment. At a stop: 1–2 decaying swings of 0.1–0.2 rad at about 1–1.5 Hz. | The tail's yaw momentum at 0.12 rad is about 15 % (60 cm) or 20 % (75 cm) of the legs' (est.). The right phase lowers the stance-pad yaw moment that sets F21's friction need; today's phase raises it. Load: in the plant, 0.12 rad costs about 0.26–0.34 N·m RMS (0.52–0.69 of A1), mostly from the joint's default damping and friction rather than inertia (0.08–0.28 N·m peak). | Verify the sign with `turning.py`'s straight runs. Tail-yaw feedback that cancels stance-foot yaw is a listed F21 remedy. |
| F10 | **Head stabilised against trunk rotations**; remove the 0.05 rad metronome nod; add a small "settle" | `head_mode = "stabilise"`, `head_settle` (rad), `settle_lag` (s) | The bird pattern: rotations held, translations followed (Theunissen & Troje). Ungulate heads move out of phase with the withers. **The neck pitch cancels trunk pitch (gain 0.8–1.0) and the head follows the body bob. Settle: a 0.01–0.02 rad dip 0.05–0.1 s after each touchdown.** | Cheaper than today's nod. Cancelling ±0.03 rad of trunk pitch costs of order 0.05–0.1 N·m inertial at 60 cm and 0.1–0.2 at 75 cm (the trunk pitch the closed loop produces is not yet known), against 0.08–0.15 and 0.15–0.29 for today's 0.05 rad nod. The spec's 0.15 rad nod would cost 0.23–0.45 and 0.44–0.88 (all est., plus damping). | A small IMU feedback on the neck, outside the balance loop. It also steadies the head camera for the detector; the camera pose comes from the encoders and the IMU (BEHAVIOR_SPEC; A12). |
| F11 | **Head-first turns**: the head starts the turn and the body follows; the tail swings inward at the start and trails at the end | `head_lead_s`, `head_lead_frac` | Humans: about 200 ms. BD-X: the torso follows when the neck reaches its limit. **Lead 0.2–0.4 s, up to 0.3–0.5 rad of neck yaw.** Tail: a flick toward the inside to start, overshooting outward at the end. | Neck yaw as in section 5. Inward tail acceleration gives a reaction torque that helps the turn; the tail's momentum is about 10–15 % of the body's at 0.785 rad/s (est.). | Needs neck yaw. With pitch only, the head can lead only by a slight raise and look; the trunk must start the turn. |
| F12 | **Anticipation and settle** at starts, stops and speed changes | `anticipate_s`, `settle` | Animation principles (anticipation, follow-through). **0.15–0.25 s pre-shift (weight back, head dips 0.05 rad) before the first step. Head and tail overshoot 5–10 % on stopping and settle within 0.5–1 s.** | Small. Keep within the spec's 1.5 m/s² start/stop limit. | These belong in the transition generator (preview control) and a head/tail overlay. |
| F13 | **Longer stride at the animal pace**, where the knee allows | `stride_rel` | Alexander 0.94 H; tinamou fit 0.98 H; dinosaur trackways about 1.3 H. The robot uses 0.80 H. **Screen 0.85–0.90 H.** | Hip-pitch range and speed rise. In F25 the 75 cm body went to 0.86 H with lower drive ratios, while the 60 cm bodies went shorter (0.71–0.73 H) to relieve the knee. So the cost depends on the body; at 60 cm the knee rules it out unless the knee changes. | Lower cadence reads bigger. Keep the nominal stride at about 0.55–0.60 H: a lower cadence at that speed increases sway. |
| F14 | **Variation (no twins)**: small irregularities in timing, gaze and tail | `phase_jitter`, `asym`, `noise_seed` | Thomas & Johnston's "twins"; Disney gaze (saccades, habituation). **Step time ±1–2 %, left-right asymmetry ≤ 2–3 %, slow random variation of head gaze and tail amplitude at 0.05–0.2 Hz.** | None in torque. | The tracker's phase-rate PI already allows ±5 %. The gates and metrics must accept strides that are not identical. |

**Optional hardware ideas, for the owner and not the reference walk.**
- A passive vertical spring in the tail, at its root or mid-tail, tuned near the step frequency. It would give the
  T. rex heave of van Bijlert et al. for no servo load, driven by F1's bob. On the 60 cm tail the stiffness is about
  2.8 N·m/rad for 1.96 Hz, which sags 0.2 rad under gravity unless preloaded (est.).
- A neck gravity-compensation spring (section 5.2).

### 3.2 What to keep

- **Froude scaling and the Froude 0.05 animal pace.** It equals the T. rex preferred-speed estimate (Fr 0.054).
- **The leg posture: hip-driven and fairly upright** (femur 22° from vertical, knee about 34° flexed at home). This is
  consistent with "more upright but not columnar" T. rex reconstructions and with the tailed-chicken experiment.
- **The generator's multibody ZMP correction at the `r2` accuracy.** It keeps every new feature dynamically consistent,
  which is what makes a render worth judging.
- **The world-frame minimum-jerk forward swing.** It gives ground-speed matching and the mild retraction before
  touchdown (only the vertical profile changes, F4).
- **The nominal stride (about 0.55 H; Alexander gives 0.61 H) and its cadence of about 2 steps/s at 60 cm.**
- **The swing lift chosen by surface:** 0.10 H on hard floor, 0.19–0.195 H for carpet (F20, F23).
- **The narrow-stance bodies.**
- **The level-head home pose and the skull-roof camera** (P22).

### 3.3 What it means for the controller

- **R2 (the tracker):**
  - A new reference needs a refit of the 13 gains, because Stage 1's fits were made on the old walk.
  - The trunk PD, phase clock and step feedback take F1, F3, F6 and F14 as reference changes.
  - F5 needs a rolling-contact model in the ankle cap and in the estimator's contact point.
  - F10 and F11 are a head layer outside the balance loop (IMU and gaze target to neck command).
  - F9's phase can also become a feedback: tail yaw driven by the estimated stance-foot yaw moment.
- **Learning, on Colab (owner decision 1).** Follow the BD-X and Olaf recipe:
  - imitation rewards on the new reference (joint positions, torso pose, contact timing), plus action-rate and
    action-acceleration penalties for smoothness, an impact term (quiet feet), a head-stabilisation term, and a
    thermal-style penalty on the neck and tail servos (both near their limits here);
  - randomised servo delay (OP3 used 10–50 ms of observation delay; at 10 ms nothing walks under R2 today) and pushes
    while walking;
  - separate standing and walking policies, conditioned on head commands, so that gaze and idle stay authorable.

  Expect far more steps than Stage 1's 1–4 million: BD-X previews after about 30 GPU-minutes but trains about 22
  GPU-days, and OP3's skills used 10⁸–10⁹ steps.
- **The reference files are part of the controller.** New walks need new names in `refs` (e.g. `nominal_look`,
  `fr05_look`), because G2 compares each reference with the round-two records and the new ones will, rightly, not
  match them.

## 4. Answer 2: judging "looks alive"

### 4.1 Before any controller work (cheap: hours of CPU)

1. **A variant ladder.** For each body, at the nominal walk and the animal pace, build:
   - V0: today's walk.
   - V1: + F1/F2 (bob and down).
   - V2: + F3 (swing time).
   - V3: + F4/F5 (swing shape, toe-off).
   - V4: + F6 (roll).
   - V5: + F9/F10 (tail phase, stabilised head).
   - V6: everything, plus F14 (variation).

   Add a 90° walking turn and a start–stop for V0 and V6. Change one thing per step so the owner sees what each feature
   buys.
2. **Price each variant before anyone looks at it.** Use the Stage 0 screen (`gait.inverse_dynamics` + `summarize`, the
   walking clearance gates, and `turning.py`'s straight run for the stance-pad yaw moment and friction need). Record:
   - drive ratios with the 1.15 / 1.25 margins: the knee's speed at the animal pace, hip pitch, the ankles, neck and tail;
   - centre-of-pressure margins;
   - sway and peak acceleration.

   A variant that breaks a drive limit goes back before it is rendered. Optionally, the lead can run a two-stride R0
   open-loop playback as a physical sanity check; it takes seconds.
3. **Render kinematically at 1× real time.** Use `designlab/render.py` (MuJoCo offscreen, `exterior` mode, covers on),
   10 s clips at 30 fps, from two fixed cameras:
   - a visitor's view: 1.5 m high, 2.5 m away, three-quarter front. Adults see the robot from above, which shows the
     back line, sway and tail most;
   - a side view at 0.6 m height (a child's or photographer's view), which shows the feet and knees.

   Add three diagnostics:
   - a **point-light** version: sites at the snout, eye, neck base, hips, knees, MTPs, toe tips, mid-tail and tail tip,
     on black. It judges the motion without the covers (Johansson);
   - **path plots** of the snout, hip, toe and tail tip. Arcs should be smooth, without straight segments or corners;
   - a **phase chart** of each part's extremes. Staggered extremes (overlap) read as alive; everything peaking at the
     same instant reads as a machine.
4. **Anchors.**
   - Low anchor: V0.
   - High anchor: openly licensed footage of a long-legged walking bird (emu, rhea or heron) at a slow walk. Play it
     at √(H_bird/H_robot) speed to match dynamics (an emu with a hip of about 0.9 m: ×1.49 for the 60 cm body, ×1.34
     for the 75 cm body).
5. **Machine metrics** next to each clip:
   - hip bob (peak to peak, H);
   - trunk roll and pitch RMS; sway;
   - head world-pitch RMS (the stabilisation ratio: head angular rate over trunk angular rate);
   - tail phase relative to the legs;
   - toe-off angle; swing time and duty factor against the bird curves;
   - stance-pad yaw moment;
   - "mechanical tells": cycle-to-cycle step-time variation of 0, flat feet at lift-off, a constant hip height,
     synchronous extremes;
   - smoothness of foot and head velocity ([SPARC, Balasubramanian et al. 2015](https://pmc.ncbi.nlm.nih.gov/articles/PMC4674971)).
6. **Owner review first** (P24), then 3–5 people who are not on the project.
   - **Blind:** clips are unlabelled and randomised.
   - **Rubric:** the rubric below, plus the six Godspeed animacy items, each rated 1–5 between two poles: dead/alive,
     stagnant/lively, mechanical/organic, artificial/lifelike, inert/interactive, apathetic/responsive
     ([Bartneck et al. 2009](https://www.bartneck.de/publications/2009/measurementInstrumentsRobots/)). Compute
     Cronbach's α before averaging (they recommend at least 0.7).
   - **Paired comparisons:** for close variants, also ask "which looks more like a living animal?" Paired choices
     separate small differences better than a few Likert scores.
   - **One-page sheet per variant:** the owner gets clips, metrics and costs on one page and picks the features to
     carry into controller work.

**Rubric** (1 = machine, 3 = mixed, 5 = animal). Items 1–5 are the spec's; items 6–7 are added.

| # | Item | 1 | 5 |
|---|---|---|---|
| 1 | Foot placement and roll | Feet lift and land flat; shuffling or slapping | Feet peel off from the toes and land quietly and deliberately |
| 2 | Knee and hock motion | Knees locked at one bend; stiff | Legs fold and reach smoothly; the stance leg carries the body |
| 3 | Body bob and weight | Body glides at a fixed height or slides sideways | Body rises and falls with the steps; weight settles after each footfall |
| 4 | Head and tail coordination | Head and tail swing like metronomes or stay frozen | Head steady and purposeful; tail lags and settles |
| 5 | No shuffling or jitter | Visible jitter, foot sliding, stutter | Clean, continuous motion |
| 6 | Timing and anticipation | Starts and stops abruptly; every stride identical | Anticipates, settles, varies slightly |
| 7 | Overall | A machine walking | An animal walking |

### 4.2 After control (the spec's blind rating)

1. **Clips from the closed loop**, not the reference:
   - **Conditions:** the realistic servo delay (the bench value once A17's test exists; 10 ms until then, which is also
     the agility bar's condition) and the demo floors' grip (hardwood and vinyl, P26).
   - **Behaviours:**
     - the nominal walk and the fastest walk shown;
     - a 90° walking turn in about 2 s;
     - a start and a stop;
     - idle with look-at;
     - a nudge while walking.
   - **Format:** the same cameras, covers and 1× speed as in 4.1.
2. **Look-fidelity metrics from the traces**, compared with the reference:
   - bob amplitude within ±30 %;
   - head world-pitch RMS ≤ 1.5° in steady walking;
   - tail phase within ±0.1 cycle;
   - toe-off at least 70 % of the reference;
   - touchdown vertical speed;
   - step-time variation;
   - jitter: joint-velocity power above about 8 Hz;
   - foot slip.

   For scale, BD-X walks with a 0.123 rad mean joint-tracking error and Olaf with 4°, so style can be lost in tracking.
   These numbers show where.
3. **Blind rating.**
   - **Raters:** at least 12, ideally 20, independent and new to the project.
   - **Clips:** 8–12 per rater, in random order. They include the anchors and, for each behaviour, its reference render.
   - **Instrument:** the rubric plus the Godspeed animacy items.
   - **Analysis:** report the mean and 95 % confidence interval per clip, and rater agreement (Krippendorff's α or ICC).
   - **Pass:** a mean of at least 3.5/5 on the walking clips (BEHAVIOR_SPEC). A second criterion is suggested: each
     closed-loop clip within 0.5 of its own reference render (paired test). That separates "the reference is not alive
     enough" from "the controller lost the look".

   A private page with the randomised clips and a stored rating form would collect the answers. The lead can build it.
4. **On hardware later,** repeat with real footage and sound. Simulated clips are silent, but servo whine and foot slap
   are strong mechanical tells (Olaf's impact reward cut 13.5 dB).

## 5. Answer 3: head and tail behaviours that carry character at a slow pace

### 5.1 Behaviours, joints and loads

Inputs for these figures:
- **Head:** 0.531 kg with its centre of mass 0.119 m ahead of the neck-pitch axis on `60-N-T-10`; 0.792 kg at 0.150 m on
  `75-M-H-10`.
- **Inertia about a neck-base yaw axis,** with a jaw servo fitted: 0.0098 / 0.0229 kg·m². About a skull-only (atlas) yaw
  axis it would be about 0.0033 / 0.0076 kg·m².
- **Tail yaw inertia:** 0.0183 / 0.0383 kg·m².
- **Each added servo:** an STS3215 at 0.055 kg, plus about 0.02 kg of horn and bracket.
- **Neck-pitch servos:** STS3250 at 60 cm (A1 0.819 N·m, peak 1.437 N·m, 4.71 rad/s); XM540-W270 at 75 cm (A1
  1.770 N·m, peak 3.106 N·m, 1.885 rad/s).
- **Tail, neck-yaw and jaw servos:** STS3215 (A1 0.491 N·m, planning peak 0.862 N·m, 2.83 rad/s).
- **Passive terms:** "+ passive" figures add the plant's default joint damping (0.2 N·m·s/rad) and friction loss
  (0.15 N·m). These defaults are unsourced and probably pessimistic for a bearing-mounted pivot.

| Behaviour | Joints | Starting motion | Load, 60 cm (est.) | Load, 75 cm (est.) | Notes |
|---|---|---|---|---|---|
| **Look at a person** (standing or slow walk) | Neck yaw + neck pitch | A gaze shift (minimum-jerk) of 0.3–0.6 rad in 0.5–0.8 s, then hold with slight drift. Retarget every 2–6 s with a brief glance away (habituation, Disney gaze); tune by review. | Yaw: 0.05–0.14 N·m inertial; + passive, peak 0.48–0.74 (STS3215 peak 0.862). Pitch: hold 0.62–0.69 N·m. | Yaw: 0.12–0.32 N·m inertial; + passive, 0.56–0.92, so the 0.5 s shift exceeds the planning peak. Use ≥ 0.8 s or a stronger servo. Pitch hold: 1.16–1.25. | Neck pitch is limited to ±0.35 rad. From the eye height (0.58 m at 60 cm, 0.72 m at 75 cm), the head can point at an adult's face (about 1.6 m) only from about 3 m (60 cm) or 2.5 m (75 cm), and at a child's (about 1.1 m) from about 1.5 m or 1.1 m (est.). |
| **Head-first turn** | Neck yaw (pitch) | The head leads by 0.2–0.4 s, by 0.3–0.5 rad, and the body follows. | As above. | As above. | Without yaw, the trunk starts the turn and only a slight head raise can lead it. |
| **Gaze-stabilised walking head** (default) | Neck pitch (+ yaw if fitted) | Cancel trunk pitch and yaw rotation; follow the bob; small settle after each footfall. | 0.05–0.09 N·m inertial on top of the hold. | 0.09–0.18 N·m. | Cheaper than today's nod, and steadies the camera. |
| **Idle "breathing"** | Neck pitch; knees (body rise and fall) | ±0.01–0.02 rad at 0.2–0.3 Hz; body ±0.005 H. | Under 0.01 N·m inertial; friction dominates in the model. | Under 0.02 N·m. | Disney ties head motion to simulated breathing. Never fully still: the animators' "moving hold". |
| **Idle weight shift and glance-around** | Hips and ankles; neck | The centre of mass shifts between the feet by 0.03–0.05 H over 1–2 s every 4–8 s; the head and tail lag it. | Small. | Small. | A BD-X-style background layer. |
| **Inspect or sniff** | Neck pitch (+ yaw) | Head down to −0.3 rad; small sniffs of ±0.02 rad at about 3 Hz. | Hold rises to about 0.65 N·m (0.79 of A1), or 0.72 with a jaw servo (0.88). Keep it brief. | About 1.2–1.3 N·m (0.69–0.73). | Raising the head lowers the hold (−8 % at +0.2 rad): idle with the head slightly up. |
| **Roar (open mouth)** | Jaw + neck pitch (+ a small yaw shake) + tail; speaker | Anticipation: head back and down 0.1 rad for 0.3 s. Then raise 0.3 rad in 0.4 s while the jaw opens 0.5–0.6 rad in ≥ 0.4 s. Hold 1–1.5 s, with a shake of ±0.1 rad at 3 Hz (60 cm) or ±0.05 rad at 2.5 Hz (75 cm). Then close and settle. | Pitch raise: 0.11 N·m inertial plus the hold, ≤ 0.8 N·m against a 1.44 peak. Shake: 0.35 inertial, 0.50–0.66 with passive terms, above A1 but within peak, so bursts only. Jaw: ≤ 0.07 N·m. | Raise: 0.26 + hold ≤ 1.5 N·m (peak 3.1) at 1.41 rad/s (limit 1.885). Shake at ±0.1 rad: 0.82 N·m inertial and 0.97–1.05 with passive terms, over the STS3215's peak; use ±0.05 rad at 2.5 Hz (0.43–0.47). Jaw: ≤ 0.12 N·m. | Without sound it reads as a mouth opening. The tail lift is deferred; use a wide tail sweep instead. |
| **Closed-mouth "boom"** (a scientifically grounded alternative) | Neck pitch; speaker | Head dips 0.1–0.15 rad and holds, with a low-frequency rumble; jaw closed. | Small. | Small. | Closed-mouth vocalisation evolved at least 16 times in archosaurs ([Riede et al. 2016, via ScienceDaily](https://www.sciencedaily.com/releases/2016/07/160711121517.htm)). It needs no jaw, so it works on the current head. |
| **Snap** | Jaw | 0.3–0.4 rad clap. | Torque ≤ 0.1 N·m, but speed-limited: 0.4 rad takes ≥ 0.27 s at 2.83 rad/s, and a 0.15 s snap would need 5 rad/s. | ≤ 0.22 N·m; same speed limit. | Slower than film snaps. A sharp sound sells it. |
| **Tail mood** | Tail yaw | Relaxed: a wide slow sway of ±0.25 rad at 0.1–0.2 Hz. Alert: tail centred and still (small tremor ±0.02 rad). | Inertial < 0.01 N·m; passive about 0.16–0.19 N·m RMS (0.33–0.39 of A1). | Inertial < 0.02 N·m; passive the same. | Slow, large sways are cheap. Stride-frequency swings are not (F9). |
| **Tail follow-through** (stops, turns) | Tail yaw | 1–2 decaying swings after a stop; an inward flick to start a turn. Start at 0.15 rad and 1.2 Hz. | 0.15 rad at 1.2 Hz: about 0.42 N·m peak with passive terms; 0.2 rad at 1.5 Hz: about 0.65 (bursts only). | 0.15 rad at 1.2 Hz: about 0.55 N·m; 0.2 rad at 1.5 Hz: about 0.93, over the planning peak, so keep to 0.15 rad and 1.2 Hz or less. | Also helps the yaw balance in turns (F11). |
| **Startle** | Neck pitch, tail, legs | Head up 0.2 rad in 0.25 s; tail centres and stiffens; half a step back. | 1.5 rad/s, within the STS3250's speed. | 1.5 rad/s against the XM540-W270's 1.885 rad/s planning speed: tight. | |

### 5.2 Placement and servo notes

- **Neck yaw at the neck base (yaw, then pitch).** The pitch servo then carries no yaw-servo weight, and the whole neck
  turns, which reads as an animal rather than a skull swivelling on a fixed neck. The cost is about three times the
  yaw inertia of a skull-only joint.
  - The yaw stage needs its own bearing; the STS3215's output should not carry the head's bending moment (about
    0.6 / 1.2 N·m).
  - The camera pose then needs the yaw encoder too (A12).
- **Jaw servo in the skull, just above the jaw hinge** (about 0.10 m ahead of the neck axis at 60 cm, 0.12 m at 75 cm).
- **Neck-pitch thermal budget.** Over a session the neck servo is the hottest drive (F12). The jaw servo raises the
  60 cm static hold from 0.75 to 0.84 of A1, or 0.92 if the yaw servo also sits in the head. At 75 cm the figures are
  0.66, 0.71 and 0.75. Mitigations:
  - the adopted PA11 covers (walls ×0.75), which take the 60 cm neck from 0.76 to 0.70 (F22);
  - idle with the head slightly raised;
  - a neck gravity-compensation spring (a standard animatronics fix) to leave the servo's heat budget for motion;
  - an animation-engine duty budget, or Olaf's thermal-aware policy, which addressed exactly this failure on a neck
    pitch.
- **Whole-robot effect of the two servos** (0.15 kg, yaw at the base). The centre of mass moves 5.7 mm (60 cm) or
  5.1 mm (75 cm) forward. Hip-pitch hold rises by about 0.61 / 0.75 N·m (0.67 / 0.82 with the yaw in the head). Against
  the walking RMS, that is about 0.55 → 0.65 of the 60 cm hip-pitch drive's A1, and 0.76 → 0.83 at 75 cm (est., Stage 0
  basis).
- **The plant's default joint damping and friction dominate the modelled tail and neck-yaw loads.** When A1's tail
  bench test runs, also measure the tail joint's real friction and damping. For a bearing-mounted pivot they are likely
  far below 0.15 N·m and 0.2 N·m·s/rad, and that one measurement could move the expressive budgets more than any part
  change.

### 5.3 On the owner's question ("is the extra expressiveness worth a slight speed hit?")

**Research view, not a decision (P24).**

At a slow, character-first pace, people read life mostly from where the head is looking and how it moves before the
body:
- gaze, attention and head-first turns: Disney's gaze work, BD-X's gaze stick, and the human steering studies;
- a display: a roar or boom.

The neck yaw buys the first; it is the single most valuable joint per gram. The jaw buys the second, but is used rarely.

The speed cost is indirect: 0.15 kg at the head, +0.6–0.8 N·m of hip-pitch hold and 5–6 mm of centre-of-mass shift.
Stage 1b found the leg drives are not what limits speed today. So the hit is likely small, but it should be measured,
as the lead proposes, by modelling both heads.

The real costs to watch are:
- the 60 cm neck-pitch thermal budget;
- the yaw servo's torque for fast gaze shifts at 75 cm.

A cheaper middle step: the current head already supports gaze stabilisation (F10), the boom, the settle and the idle
layer. These test the character idea before any hardware change.

## 6. Caveats

- **No walking kinematics exist for any non-avian theropod.** Every value here comes from birds, humans, trackways or
  models (van Bijlert et al.'s tail model, Hutchinson's limb model). Viewers' idea of a "natural" T. rex also comes
  partly from film animation.
- **The loads are rigid-body estimates** on the round-two inertias with simple motion profiles. The Stage 0 screen and
  the closed loop decide.
- **The tail-phase finding rests on my reading of `gait.py`'s phase and sign conventions.** Verify it first.
- **The plant's joint damping and friction are unsourced defaults.** They dominate the tail and neck-yaw estimates.
- **Some animation advice is from community sources** (marked secondary).
- **The sway estimate (sway ≈ 0.47 × foot spacing) is a linear-pendulum approximation** fitted to the reference's
  reported value.

## Sources

Biology and palaeontology
- van Bijlert, van Soest & Schulp 2021, *Natural Frequency Method: estimating the preferred walking speed of Tyrannosaurus rex based on tail natural frequency*, R. Soc. Open Sci. 8: 201441. [PMC](https://pmc.ncbi.nlm.nih.gov/articles/PMC8059583); [PDF](https://repository.naturalis.nl/pub/802229/Bijlert-2021-Naturals-frequency-method.pdf)
- Bishop et al. 2018, *The influence of speed and size on avian terrestrial locomotor biomechanics*, PLOS ONE 13: e0192172. [Link](https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0192172)
- Hancock, Stevens & Biknevicius 2007, terrestrial locomotion in the Elegant-crested Tinamou, Ibis. [PDF](https://people.ohio.edu/stevensn/documents/Hancock%20et%20al%202007%20Tinamous%20early%20view.pdf)
- Grossi et al. 2014, *Walking like dinosaurs: chickens with artificial tails…*, PLOS ONE 9: e88458. [PMC](https://pmc.ncbi.nlm.nih.gov/articles/PMC3915051)
- Hutchinson, Anderson & Delp 2005, T. rex hindlimb moment arms, Paleobiology 31: 676–701. [Cambridge](https://www.cambridge.org/core/journals/paleobiology/article/analysis-of-hindlimb-muscle-moment-arms-in-tyrannosaurus-rex-using-a-threedimensional-musculoskeletal-computer-model-implications-for-stance-gait-and-speed/30DD414511DEB56D79A5C6281A284320)
- Persons & Currie 2011, caudofemoralis in theropods, PLOS ONE. [DOAJ](https://doaj.org/article/27a525d1b73b477daf599d43d68bac7d)
- Snively et al. 2019, *Lower rotational inertia and larger leg muscles indicate more rapid turns in tyrannosaurids…*, PeerJ 7: e6432. [PMC](https://pmc.ncbi.nlm.nih.gov/articles/PMC6387760/)
- Falkingham & Gatesy 2014, *The birth of a dinosaur footprint*, PNAS 111: 18279. [PDF](https://cs.brown.edu/courses/cs137/2015/readings/Gatesy_XROMM_DEM_PNAS14.pdf)
- Botucatu Formation theropod trackways (Leonardi et al. 2023). [PDF](https://igeo.ufrj.br/inc/isc/3/03-133-%20Farlowichnus%20Leonardi%20et%20al%202023.pdf)
- Necker, head-bobbing of walking birds (review). [Page](https://reinhold-necker.de/seite10.html)
- Theunissen & Troje 2017, *Head stabilization in the pigeon*, Front. Neurosci. 11: 551. [PDF](https://www.frontiersin.org/journals/neuroscience/articles/10.3389/fnins.2017.00551/pdf)
- Mallard swing and stance versus speed, PeerJ 2023. [Link](https://peerj.com/articles/15362); wild turkey limb swing (SICB abstract). [Link](https://sicb.org/?p=25566)
- Riede et al. 2016, closed-mouth vocal behaviour, Evolution. [ScienceDaily summary](https://www.sciencedaily.com/releases/2016/07/160711121517.htm)

Human gait, walking models and robots
- Smith, Lelas & Kerrigan, sex differences in pelvic motion and centre-of-mass displacement. [Record](https://katalog.lib.cas.cz/KNAV/EdsRecord/edsair,edsair.doi.dedup.....5711e2752a1f2d3be6fabfdd42692807)
- Ortega & Farley 2005, *Minimizing center of mass vertical movement increases metabolic cost in walking*. [Record](https://scilit.com/publications/29280cbdf8609328df0f3ab2834e6f1b)
- Gordon, Ferris & Kuo 2009, *Metabolic and mechanical energy costs of reducing vertical center of mass movement during gait*. [Record](https://scilit.com/publications/8760e02900afeeb5520b48baad00b729)
- Kuo 2007, *The six determinants of gait and the inverted pendulum analogy*. [Record](https://scilit.com/publications/0bcb4397020cc31ca5a12a9cba7cb509)
- Grasso et al. 1996, *The predictive brain: anticipatory control of head direction for the steering of locomotion*. [Record](https://vufind-test.katalog.k.utb.cz/SummonRecord/FETCH-LOGICAL-c3295-285d249ebde34c5be285545fc4b568b8ab5e3f3143c3f2302b544b80dc85c36b2); Grasso et al. 1998, eye–head coordination for steering. [Record](https://acnpsearch.tweb-dev.unibo.it/singlejournalindex/58514855)
- Wisse, Atkeson & Kloimwieder 2005, *Swing leg retraction helps biped walking stability*. [TU Delft](https://research.tudelft.nl/en/publications/swing-leg-retraction-helps-biped-walking-stability/)
- Kajita et al. 2003, biped pattern generation by preview control of the ZMP (constant CoM height). [PDF](https://people.csail.mit.edu/katiebyl/kb/DW2008/papers_of_tangential_interest/kajita03.pdf)
- Ogura et al. 2006, *Human-like walking with knee stretched, heel-contact and toe-off motion by a humanoid robot*, IROS. [Record](https://researchr.org/publication/OguraSKMOMLT06/bibtex)

Character robots and learning
- Grandia et al. 2025, *Design and Control of a Bipedal Robotic Character* (Disney Research). [arXiv 2501.05204](https://arxiv.org/html/2501.05204v1)
- *Olaf: Bringing an Animated Character to Life in the Physical World* (Disney Research). [arXiv 2512.16705](https://arxiv.org/html/2512.16705v1)
- Pan et al. 2020, *Realistic and Interactive Robot Gaze*, IROS. [Disney Research](https://la.disneyresearch.com/publication/realistic-and-interactive-robot-gaze/); [IEEE Spectrum](https://spectrum.ieee.org/disney-research-robotic-gaze)
- Haarnoja et al., *Learning agile soccer skills for a bipedal robot with deep reinforcement learning* (Science Robotics 2024). [arXiv 2304.13653](https://ar5iv.labs.arxiv.org/html/2304.13653)
- Peng et al. 2020, *Learning Agile Robotic Locomotion Skills by Imitating Animals*, RSS. [Project](https://xbpeng.github.io/projects/Robotic_Imitation)
- Escontrela et al. 2022, *Adversarial Motion Priors Make Good Substitutes for Complex Reward Functions*. [arXiv 2203.15103](https://arxiv.org/abs/2203.15103)

Perception and evaluation
- Biological motion and point-light displays (Johansson 1973). [Summary](https://en.wikipedia.org/wiki/Biological_motion)
- Troje & Westhoff 2006; Chang & Troje 2009, the feet's acceleration and the "life detector". [Biomotion Lab](https://www.biomotionlab.ca/the-inversion-effect-in-biological-motion-maybe-there-are-two-of-them/); [Biomotion Lab (2)](https://www.yorku.ca/research/biomotionlab/?p=536)
- Bartneck, Kulić, Croft & Zoghbi 2009, the Godspeed questionnaires. [Link](https://www.bartneck.de/publications/2009/measurementInstrumentsRobots/)
- Balasubramanian et al. 2015, *On the analysis of movement smoothness* (SPARC). [PMC](https://pmc.ncbi.nlm.nih.gov/articles/PMC4674971)

Animation
- Thomas & Johnston, the twelve principles of animation. [Summary](https://en.wikipedia.org/wiki/Twelve_basic_principles_of_animation)
- Community T. rex animation critique (secondary). [Blender Artists](https://blenderartists.org/t/t-rex-test-animation/1444558)

Project records used: `DECISION_LOG.md` (P22–P28, F6, F11–F12, F16, F20–F25, A1, A12, A17 draft);
`BEHAVIOR_SPEC.md` §1; `STAGE1_PLAN.md` decisions 9–10; `stage1/runs/2026-11-08/STAGE1_PROBE.md` §8 (G7, G8);
`stage1/runs/2026-11-10-speed/STAGE1B_SPEED.md` §5–§6; `round2/gait_tune/GAIT_TUNE.md`; `designlab/gait.py`;
`designlab/stage1/refs.py` (the feedforward leaves out the MJCF joint damping and friction); `designlab/render.py`;
the `60-N-T-10` and `75-M-H-10` records and MJCF models.
