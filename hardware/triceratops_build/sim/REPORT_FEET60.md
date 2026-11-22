# The foot study at 60 cm (decision D5): judged result (simulation)

22 Nov 2026. Rules `PREREG_FEET60.md`, committed before the judged run (00378469). Tool output
`results/feet60_2026-11-22/results.json` (`feet60_study.py` at 00378469, clean, 5.8 min). Models `feet60/`
(`build_feet60.py`). Simulation of rigid feet on a rigid floor, on a model whose masses are estimates.

## Answer first

- **One foot passes every rule under both contact settings: the capsule sole.** It is an 88 mm-long sole rounded
  across its width (24 mm radius), with PR #669's pitch hinge kept and no extra joint.
  - It stands (R1-R7).
  - It crawls on all four seeds under both settings (C1-C4, C6, and C5c with 2.7-7.0 mm of net sliding).
  - The frozen trot walks on it in all four conditions (W1-W4).
- **Each other foot fails one test:**
  - **The as-built foot** (PR #669's rigid sole) fails the crawl's C4 under both settings, on all four seeds. Its sole
    tips onto an edge and the third support foot unloads (F7, F11, reproduced).
  - **The roll hinge** (the existing ankle-roll variant, 2.032 N·m/rad) passes under the model's contact and fails C4
    under the stiffer setting, on all four seeds. In development, softer and stiffer springs failed C4 under both.
  - **The 30 mm ball** passes C4 under both settings, but under the model's contact it slides 10.9-11.0 mm in a stance
    (C5c, limit 10 mm) on all four seeds.
- **Every foot stands and every foot trots.** The museum walk, the trot, does not depend on the foot. The foot
  matters for slow, statically stable stepping and standing shifts, where an edge-loaded sole unloads a support foot.
- **Recommendation for D5 (the owner decides):** the capsule sole. It is the only robust pass, and it needs no extra
  joint: four fewer joints, springs and end stops than the roll hinge. See below for its costs.

## Results (4 crawl seeds, 2 stand seeds, 5 trot seeds per condition; both contact settings)

| Foot | Stand R1-R7 (both) | Crawl, model's contact | Crawl, impratio 10 | Trot W1-W4, J1-J4 | Verdict |
|---|---|---|---|---|---|
| `pitch` (as built) | pass | **C4 fails** 4/4 | **C4 fails** 4/4 | pass | fails |
| `roll_k2.032` | pass | pass | **C4 fails** 4/4 | pass | fails |
| `capsule_r24` | pass | pass | pass | pass | **passes** |
| `ball_r30` | pass | **C5c fails** 4/4 | pass | pass | fails |

The crawl in detail. "Pin travel" is C5, reported only: a rolling foot moves its pin without sliding.

| Foot | Net sliding per stance, worst seed (C5c ≤ 10 mm): model / impratio 10 | Pin travel, worst: model / impratio 10 |
|---|---|---|
| `pitch` | 5.3 / 4.4 mm | 19.7 / 20.0 mm (tipping on an edge) |
| `roll_k2.032` | 9.9 / 8.6 mm | 7.8 / 10.9 mm |
| `capsule_r24` | 7.0 / 4.7 mm | 18.9 / 19.7 mm (rolling) |
| `ball_r30` | **11.0** / 5.3 mm | 27.3 / 24.5 mm (rolling) |

The trot in detail (mean of 5 episodes per condition; slip taken at the contacts). J1/J3 are friction 1.0 and J2/J4
friction 0.4; J1/J2 use the model's contact and J3/J4 impratio 10.

| Foot | Speed, m/s, J1 / J2 / J3 / J4 | Worst RMS / rating | Worst slip ratio | Front touchdowns per s, grippy (J1, J3) | Joint power |
|---|---|---|---|---|---|
| `pitch` | 0.247 / 0.252 / 0.250 / 0.252 | 0.98 (J3) | 0.18 | 1.6-1.8 | 24-30 W |
| `roll_k2.032` | 0.249 / 0.253 / 0.251 / 0.254 | 0.87 | 0.19 | 3.8-4.0 | 25-31 W |
| `capsule_r24` | 0.245 / 0.258 / 0.241 / 0.258 | 0.91 | 0.09 | 4.2-4.3 | 25-35 W |
| `ball_r30` | 0.255 / 0.275 / 0.252 / 0.276 | 0.85 | 0.10 | 6.3-6.6 | 25-37 W |

- **The trot** was developed on the roll-ankle foot and runs on the others untuned, as a drop-in. On the capsule,
  one J3 episode was slower (0.224 m/s; the mean 0.241 passes W2).
- **Front touchdowns:** the as-built foot gives the fewest, so the front-foot "bounce" in `../walk/REPORT_WALK60.md`
  comes partly from the roll hinge. The capsule and the ball touch down more often.
- **The ball** overruns the command on slippery floors (0.275 m/s).
- **Standing:** all four carry 55.3 % of the weight on the front feet. The roll drives hold 0.6-0.89 N·m on the
  as-built foot and 1.0-1.26 N·m on the others, well under their 7.5 N·m continuous rating.

## What the choice means for hardware (judgement, not simulation)

| | As built | Roll hinge | Capsule sole | Ball |
|---|---|---|---|---|
| Joints per foot | 1 (pitch) | 2 (pitch + roll) | 1 (pitch) | 1 (pitch; could be removed) |
| Extra parts | - | a bearing pair, a centring spring and end stops per foot | a moulded rubber rocker in place of the flat sole | a rubber ball or hemisphere |
| Contact | flat 88 × 64 mm plate, rests on its edges | flat plate that follows the floor in roll | a line along the foot, rocking across it | a point |
| Under a museum shell | a broad flat foot | a broad flat foot | a narrower (48 mm) rounded pad, elephant-like | a round knob |
| Wear | edges | edges and the hinge | along the centre line; replaceable pad | one spot; replaceable |
| What the pass rests on | (fails) | the spring and the contact model | robust in both settings here | (fails, slides) |

The capsule's width (48 mm against the plate's 64 mm) and its rubber are not modelled beyond a rigid shape. A bench
test of one moulded rocker (its rolling, wear and grip on a museum floor) belongs in the bench plan.

## Consequences if D5 is the capsule

- **Run 3** (step sheets ready, not yet started) trains on the roll-ankle model. Its policy would then be on a foot
  that was not chosen. Retargeting it is a small change: add the capsule model to the training models and new step
  sheets, both reviewed.
- **The model-based trot** passes on the capsule untuned. Its judged result (F15) is on the roll-ankle foot, and the
  walk report's touchdown and torque-limit figures would change with the foot.
- **The 60 cm model** (`generated/`) would gain the capsule as its foot.

## What this does not show

- **Simulation only:** rigid feet on a rigid floor. The real pad's rubber compliance, wear and grip are not
  modelled; MuJoCo's soft contact is its own.
- **Flat floor, walking straight:** no terrain or turning on any foot.
- **No learned policy** has been trained on the new feet.
- **Measures defined in development:** C5c and the contact-point slip were defined after the development results
  were seen (disclosed in the pre-registration). The judged seeds test the development finding; they were not blind
  to it.
