# The foot study at 60 cm (decision D5): judged result (simulation)

22 Nov 2026. Rules `PREREG_FEET60.md`, committed before the judged run (00378469). Tool output
`results/feet60_2026-11-22/results.json` (`feet60_study.py` at 00378469, 5.8 min). results.json records no commit: the
commit and a clean tree come from the run's log, and the independent review re-ran 13 judged rows at a later commit,
with the runner and models unchanged, bit for bit. Models `feet60/` (`build_feet60.py`). Simulation of rigid feet on a
rigid floor, on a model whose masses are estimates. Revised after the independent review (see the end).

## Answer first

- **One foot passes every registered rule under both contact settings: the capsule sole.** It is a sole 88 mm long
  overall, rounded across its width (24 mm radius) and at both ends. Its straight contact line is 40 mm long. It keeps
  PR #669's pitch hinge and adds no joint.
  - It stands (R1-R7).
  - It crawls on all four seeds under both settings (C1-C4, C6, and C5c with 2.7-7.0 mm of net sliding).
  - The frozen trot passes W1-W4 on it in all four conditions.
  - Its crawl result hardly depends on the contact model's softness. A check made after the result, not
    pre-registered (`feet60_softness.py`), gave 4.4-7.1 mm at impratio 0.5, 1 and 2.
- **The other three feet each fail one rule:**
  - **The as-built foot** (PR #669's rigid sole) fails the crawl's C4 under both settings, on all four seeds. Its sole
    tips onto an edge and the third support foot unloads (F7, F11, reproduced).
  - **The roll hinge** (the existing ankle-roll variant, 2.032 N·m/rad) fails C4 under the stiffer setting, on all
    four seeds. Under the model's contact it passes, but narrowly: C5c is 9.85-9.87 mm on two seeds against the 10 mm
    limit. In development, softer and stiffer springs failed C4 under both settings.
  - **The 30 mm ball** fails C5c under the model's contact, narrowly and because of the contact model.
    - Its worst stance on each seed moves 10.9-11.0 mm (limit 10 mm); its other stances move 8.7-10.0 mm.
    - This is creep in MuJoCo's soft contact, not the foot breaking loose. The review found its stance forces well
      inside the friction cone: tangential over normal force has a median of 0.08 and a 95th percentile of 0.27,
      against a floor friction of 0.6-1.0.
    - A soft contact creeps in proportion to 1 / (impratio × contacts per sole). The ball has one contact, the
      capsule two and the flat box four.
    - In the same follow-up check the ball passes at impratio 2 (8.3 mm) and 4 (5.8 mm). It passes C4 under both
      settings.
    - Under the registered rule it fails. On the crawl it is a close second, not a clear fail.
- **Every foot passes the trot's rules, but the trot changes with the foot.**
  - Front touchdowns on grippy floors go from 1.5-1.6 per second on the as-built foot to 6.2-6.3 on the ball.
  - Joint power at friction 1.0 goes from 27-30 W to 37 W.
  - The foot decides the crawl: slow, statically stable stepping, where an edge-loaded sole unloads a support foot.
- **Recommendation for D5 (the owner decides): the capsule sole.**
  - It is the only foot that passes every rule, with margin under both settings.
  - It needs no extra joint: four fewer joints, springs and end stops than the roll hinge.
  - Its costs are listed below. Above all, it makes the trot touch down about three times per front stride, as the
    roll hinge does.

## Results (4 crawl seeds, 2 stand seeds, 5 trot seeds per condition; both contact settings)

| Foot | Stand R1-R7 (both) | Crawl, model's contact | Crawl, impratio 10 | Trot W1-W4, J1-J4 | Verdict |
|---|---|---|---|---|---|
| `pitch` (as built) | pass | **C4 fails** 4/4 | **C4 fails** 4/4 | pass | fails |
| `roll_k2.032` | pass | pass (C5c up to 9.87 mm) | **C4 fails** 4/4 | pass | fails |
| `capsule_r24` | pass | pass | pass | pass | **passes** |
| `ball_r30` | pass | **C5c fails** 4/4 (10.9-11.0 mm) | pass | pass | fails (C5c, narrowly) |

The crawl in detail. "Pin travel" is C5, reported only: a rolling foot moves its pin without sliding.

| Foot | Net sliding per stance, worst stance of each seed (C5c ≤ 10 mm): model / impratio 10 | Median stance: model / impratio 10 | Pin travel, worst: model / impratio 10 |
|---|---|---|---|
| `pitch` | 3.6-5.3 / 2.2-4.4 mm | 1.6-3.2 / 1.0-1.9 mm | 19.7 / 20.0 mm (tipping on an edge) |
| `roll_k2.032` | 8.0-9.9 / 7.8-8.6 mm | 3.8-4.8 / 3.2-4.5 mm | 7.8 / 10.9 mm |
| `capsule_r24` | 4.8-7.0 / 2.7-4.7 mm | 1.3 / 0.9-1.0 mm | 18.9 / 19.7 mm (rolling) |
| `ball_r30` | **10.9-11.0** / 3.0-5.3 mm | 8.7 / 2.4-2.5 mm | 27.3 / 24.5 mm (rolling) |

The capsule's worst stance on each seed is its first, which includes the start-up settling. Its later stances move
4.9 mm or less (from the review).

The trot in detail (5 episodes per condition; slip taken at the contacts). J1 and J3 are friction 1.0, J2 and J4
friction 0.4. J1 and J2 use the model's contact, J3 and J4 impratio 10.

| Foot | Speed, mean, m/s, J1 / J2 / J3 / J4 | Worst RMS / rating | Worst slip ratio | Front touchdowns per s, J1 / J3 (J2 / J4) | Joint power, mean, J1 / J3 | At the nominal torque limit, worst joint and episode, J1 / J3 |
|---|---|---|---|---|---|---|
| `pitch` | 0.247 / 0.252 / 0.250 / 0.252 | 0.98 (J3) | 0.18 | 1.5 / 1.6 (1.5 / 1.5) | 27 / 30 W | 0.5 / 0.4 % |
| `roll_k2.032` | 0.249 / 0.253 / 0.251 / 0.254 | 0.87 | 0.19 | 3.7 / 3.9 (1.5 / 1.6) | 30 / 31 W | 0.7 / 0.8 % |
| `capsule_r24` | 0.245 / 0.258 / 0.241 / 0.258 | 0.91 | 0.09 | 4.2 / 4.3 (2.1 / 2.0) | 31 / 35 W | 0.8 / 1.2 % |
| `ball_r30` | 0.255 / 0.275 / 0.252 / 0.276 | 0.85 | 0.10 | 6.2 / 6.3 (2.4 / 2.3) | 37 / 37 W | 1.9 / 1.8 % |

"Front touchdowns per s" is the mean of the two front feet over the five episodes. Joint power on the slippery floors
(J2, J4) is 24-25 W for every foot.

- **The trot** was developed on the roll-ankle foot. On the other feet it runs as a drop-in, untuned.
- **The capsule in J3:** two of its five episodes were slower, 0.228 and 0.224 m/s. The mean, 0.241, passes W2
  (at least 0.225, and no episode under 0.20).
- **Front touchdowns:** the as-built foot gives the fewest.
  - On grippy floors, every foot that can rock in roll touches down more often. Per episode, the as-built foot gives
    1.4-1.9 per second, against 3.4-4.1 for the roll hinge. The ranges do not overlap.
  - So the front-foot "bounce" of `../walk/REPORT_WALK60.md` (F15) comes largely from the roll hinge, not from the
    controller alone.
  - The capsule touches down about three times per front stride in all ten grippy episodes, and the ball more often
    still.
- **The ball** overruns the command on slippery floors (0.275 m/s).
- **The torque limit:** the ball (1.8-1.9 %) and the capsule in J3 (1.2 %) exceed the 1 % flag used for the learned
  policies (not a rule here).
- **Standing:** all four carry 55.3 % of the weight on the front feet. The roll drives hold 0.6-0.89 N·m on the
  as-built foot and 1.0-1.26 N·m on the others, well under their 7.5 N·m continuous rating.

## What the choice means for hardware (judgement, not simulation)

| | As built | Roll hinge | Capsule sole | Ball |
|---|---|---|---|---|
| Joints per foot | 1 (pitch) | 2 (pitch + roll) | 1 (pitch) | 1 (pitch) |
| Extra parts | - | a bearing pair, a centring spring and end stops per foot | a moulded rubber half-capsule rocker in place of the flat sole | a rubber hemisphere |
| Contact | flat 88 × 64 mm plate, rests on its edges | flat plate that follows the floor in roll | a 40 mm line along the foot with rounded ends, rocking across it | a point |
| Under a museum shell | a broad flat foot | a broad flat foot | a narrower (48 mm) rounded pad, elephant-like | a round knob |
| Wear | edges | edges and the hinge | along the centre line; replaceable pad | one spot; replaceable |
| What the result rests on | (fails C4) | the spring and the contact model | holds from impratio 0.5 to 10 here | (fails C5c narrowly; contact-model dependent) |

**What the simulated shapes are not:**
- **They are full shapes that pass through the lower leg.** At the home pose the capsule overlaps the lower strut by
  21.6 mm and the ball by 33.6 mm (from the review). The simulation hides this because a foot's contacts with its own
  lower leg are filtered. A real part is the lower half only: a rocker pad under the foot.
- **The mass sits too high.** The sole keeps the plate's 69 g at the full shape's centre, which raises the foot's
  centre of mass by 13-17 mm against the plate (from the review). A half-shape rocker would carry it lower.
- **Not modelled:** the capsule's width (48 mm against the plate's 64 mm), and its rubber beyond a rigid shape.
- **The `<corner>_toe` site** stays at the box's toe. No tool uses it.

A bench test of one moulded rocker belongs in the bench plan: its rolling, its wear and its grip on a museum floor.

## Consequences if D5 is the capsule

- **Run 3** (step sheets ready, not yet started) trains on the roll-ankle model. It needs the capsule model added to
  the training setup, the analysis tools made sole-shape aware, and new step sheets, all reviewed.
- **The model-based trot** passes on the capsule untuned. Its judged result (F15) is on the roll-ankle foot. On the
  capsule it touches down more often in front, uses more power in J3, and spends more time at the torque limit.
- **The 60 cm model** (`generated/`) would gain the capsule as its foot.

## What this does not show

- **Simulation only:** rigid feet on a rigid floor. The real pad's rubber compliance, wear and grip are not
  modelled; MuJoCo's soft contact is its own. The ball's verdict shows how much a soft contact model can decide.
- **Little variation between seeds:** a crawl seed changes only the joint offsets (±0.02 rad) and the friction
  (0.6-1.0). "All four seeds" is weak replication: the ball's C5c spans 10.92-10.99 mm across them.
- **Flat floor, walking straight:** no terrain or turning on any foot.
- **No learned policy** has been trained on the new feet.
- **Measures defined in development:** C5c and the contact-point slip were defined after the development results
  were seen. The judged seeds test the development finding; they were not blind to it. The redefinition of C5c
  changed which feet pass (see the pre-registration's addendum).
- **Lenient edges of the measures**, none changing a verdict:
  - The mean contact point hides a pure twist (yaw) of the box and the capsule.
  - C5c counts only while a foot touches the floor, so a foot that hops and lands elsewhere is not counted.

## Revised after the independent review (22 Nov; no verdict changed)

- **The ball:** "slides 11 mm per stance" was wrong. One stance per seed exceeds 10 mm, and the fail depends on the
  contact softness. The follow-up check `feet60_softness.py` (`results/feet60_2026-11-22/softness_check.json`) is
  added, labelled as not pre-registered.
- **The trot:** "does not depend on the foot" overreached. The capsule's trot costs are now listed: touchdowns,
  power and time at the torque limit. Two slow J3 episodes, not one.
- **The front-touchdown column** now traces to results.json: the mean of the two front feet. The first version gave
  ranges that matched neither foot's mean.
- **Also corrected:**
  - the roll hinge's narrow C5c pass;
  - the capsule's 40 mm contact line;
  - the overlap of the full shapes with the lower leg;
  - provenance;
  - two unsupported lines, "standing shifts" and "the ball's pitch joint could be removed".
