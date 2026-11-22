# Pre-registration: the foot study at 60 cm (decision D5), written before any judged run

22 Nov 2026. Owner's choice of next work (22 Nov). Simulation only. Candidates, tests, rules, seeds and stop rules are
fixed here. Any rule changed after a judged result will be labelled "changed after a result", with its first wording
kept.

## Why

The as-built foot (PR #669's: a passive pitch hinge and a rigid 88 × 64 mm sole) rests on its edges in roll.
- Standing: about 95 % of a forefoot's load sits on the sole's inner edge (F5).
- In the slow scripted crawl it unloads the third support foot (C4 fails at both sizes; F7, F11).
- A passive roll hinge fixed that under the model's own contact but not under the stiffer impratio-10 setting (F11).

The question for D5: which foot passes standing, the crawl and the walk under both contact settings?

## Development (disclosed; seeds below are not reused)

Development used crawl seeds 1-4, stand seeds 1-2 and trot seeds 1-2, on eight variants (`build_feet60.py`):
- the as-built foot;
- roll hinges with centring springs of 0.5, 1, 2.032 (the existing variant), 8 and 20 N·m/rad;
- capsule soles of radius 16 and 24 mm;
- balls of radius 20 and 30 mm.

Each of the following was found during development, before this file:
- **The env placed balls 28 mm into the floor.** The training env's reset placed the robot from box-sole corners;
  for a ball that sank it into the floor and the trot fell at once. The env now uses each sole's real lowest point,
  and the box path is unchanged (the regression digest holds).
- **The pin measure counts rolling.** C5 (the ankle pin's travel during a stance) counts a rolling foot as moving.
  C5c, the net sliding at the contacts, replaces it for every foot. C5c was defined after the development results
  were seen. A first version that integrated the sliding speed counted contact jitter and was dropped.
- **Slip is taken at the contact point.** The walk's slip ratio is measured at the contact point, the sole's
  velocity at its contacts (`gait_metrics.record_episode(slip_at_contact=True)`), for the same reason. This too was
  defined after development results were seen.

What development showed, with every rule here:
- **Standing:** every foot passes, under both settings.
- **The frozen trot:** every foot passes W1-W4, under both settings.
- **The crawl:**
  - The as-built foot fails C4 under both settings.
  - The roll hinge at 2.032 N·m/rad passes everything under the model's contact and fails C4 under impratio 10.
  - Softer or stiffer roll springs fail C4 under both.
  - Both capsules pass every rule under both settings.
  - Both balls pass C4 under both settings, but slide 11-13 mm (C5c) under the model's contact.

These rules are therefore not blind to which feet tend to pass; the judged seeds test whether that holds.

## What is judged

- **Feet:** four, frozen in `feet60/` by `build_feet60.py` at the commit that adds this file. Every one is the
  as-built 60 cm model (revision 3, 12.694 kg) with only the feet changed. Each keeps the sole's mass and its lowest
  point 36 mm under the ankle pin.

  | Foot | What it is |
  |---|---|
  | `pitch` | The as-built foot: a passive pitch hinge, a rigid 88 × 64 mm sole. The baseline |
  | `roll_k2.032` | The same, plus a passive roll hinge with a 2.032 N·m/rad centring spring. The existing ankle-roll variant; the walk and run 3 use it |
  | `capsule_r24` | The sole replaced by an 88 mm capsule of 24 mm radius along the foot: rounded across its width, so it rocks in roll without a joint. Pitch hinge kept |
  | `ball_r30` | The sole replaced by a 30 mm-radius ball. Pitch hinge kept |

- **Contact settings:** the model's own (impratio 1) and the stiffer impratio 10, for every test.
- **Stand:** the 60 cm standing check (`check_stand.run`, judged by `check_stand60.judge`), rules R1-R7 as for the
  as-built model, seeds 3 and 4.
- **Crawl:**
  - The 60 cm crawl (`crawl.run` with `crawl_judged60.PARAMS60`, no tuning), seeds 11-14.
  - Rules C1, C2, C3, C4 and C6 as in `PREREG_CRAWL.md`.
  - **C5c** in place of C5: each foot's net sliding at its contacts at most 10 mm from where it stood 0.10 s after
    touchdown, until lift-off. C5 (the pin) is reported, not judged.
- **Trot:**
  - The frozen model-based trot (`walk/walk60.py`) on the foot, in the walk's four conditions: friction 1.0 and
    0.4, each under both contact settings.
  - 10 ms delay, noise and randomisation on, 0.25 m/s, seeds 401-405 per condition.
  - Rules W1-W4 of `walk/PREREG_WALK60.md`, with the slip ratio taken at the contacts.
  - The trot was developed on the roll-ankle foot, so on the other feet it runs as a drop-in, untuned.
- **Tool:** `feet60_study.py` at the commit that adds this file. Run:
  `python feet60_study.py --models feet60 --out results/feet60_<date> --variants pitch,roll_k2.032,capsule_r24,ball_r30 --stand-seeds 3,4 --crawl-seeds 11,12,13,14 --walk-seeds 401,402,403,404,405`.

## Reading

- **A foot passes** when every rule above holds:
  - every stand seed passes R1-R7;
  - every crawl seed passes C1-C4, C5c and C6;
  - every trot condition passes W1-W4;
  - all of this under both contact settings.
- Otherwise the report names the tests, rules and settings it fails.
- The study recommends; the owner decides D5. Among feet that pass, the report also compares what each means for
  hardware: extra joints, parts, how it looks under a museum shell, and wear. Those comparisons are judgements,
  not simulation results.
- **Stop rules:**
  - One judged pass, with no reruns to improve a result.
  - If a fault in the harness (not a foot) is found after the run, the fix is reviewed and all four feet are run
    again on new seeds (stand 5-6, crawl 21-24, trot 501-505), labelled as a rerun. Both results are kept.

## Not shown

- Simulation of rigid feet on a rigid floor; the rubber of a real pad is not modelled.
- No terrain, no turning, no learned policy on the new feet (run 3 trains on the roll-ankle foot).
- The crawl's planner assumes contact under the ankle pin; for the rounded feet the contact rolls a little.

## Added after the judged result (22 Nov, after the independent review; no rule, seed or verdict changed)

Disclosures this file should have made before the run:
- **The C5c redefinition changed which feet pass.** On development seeds 1-4, the first C5c (the integrated sliding
  speed) failed every foot under the model's contact: as built 13.4 mm, roll hinges 16.4-17.2, capsules 14.6-14.8,
  balls 17.0-19.7. The net form (the same seeds: as built 4.9, roll 9.9, capsule r24 6.9, balls 11.0-13.1) is what
  lets any foot pass C5c under the model's contact, the recommended one included. The net form copies C5's own form
  (the largest distance from where the foot stood 0.10 s after touchdown), with the sliding in place of the pin.
- **Ten variants, not eight,** were developed: the ten listed above.
- **Why the 24 mm capsule and not the 16 mm one.** Both passed every rule in development. On the 16 mm capsule the
  trot came closer to its limits:
  - in the first development run (dev1, before the env's placement fix, which moves this capsule's start by only
    0.38 mm): one episode at 0.227 m/s and the worst joint at up to 0.97 of its rating (W3 needs under 1.0);
  - after the fix (dev2): its slowest episodes at 0.233-0.235 m/s and the worst joint at 0.92.
  The 24 mm capsule's development trots (dev2) ran at 0.245-0.260 m/s, with the worst joint at up to 0.87. The larger
  radius was taken for margin.
