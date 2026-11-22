# Dibothrosuchus elaphros anatomy and reference walk

**November 21, 2026 · physics r3 · moved from the species README**

Until 2026-11-21 this review was the "Overview", "Anatomical review", "Standing balance", "Walking mechanics" and
"Environment Details" sections of [the model's README](../../environments/dibothrosuchus_elaphros/README.md). It moved
here on that date. The summary paragraph and the reproduction section are new; the rest is unchanged apart from
headings, links and one command rewritten in the `python -m` form. It describes physics revision 3 of the anatomical
model.

*Dibothrosuchus elaphros* was a small terrestrial crocodylomorph from the Early Jurassic of Yunnan, China. The
anatomical model represents a quadruped with an upright stance and gracile limbs.
[Wu and Chatterjee (1993), description of IVPP V7907](https://doi.org/10.1080/02724634.1993.10011488)

At rest on level ground the model's center of mass projects 58.3 mm inside its loaded contact polygon. The reference
controller walks it at 0.161 m/s, placing the left hind foot, left forefoot, right hind foot and right forefoot in
sequence with at least two feet bearing weight throughout.

## Anatomy and design

The original stood on four rectangular soles. Its forelimb folded in the same direction as the hindlimb, and the visible skull extended beyond the snout landmark. Revision 3 places the elbow behind the shoulder and the hind knee ahead of the hip. A tapered trunk and tail, a distinct lower jaw, lateral eyes and nares, low frontal ridges and paired dorsal osteoderms give the surface recognizable landmarks. The two profiles below share a camera and scale. [Original MJCF at c096090](https://github.com/kuds/mesozoic-labs/blob/c096090ebe51cf8529f62eb595a6b6659a696d65/environments/dibothrosuchus_elaphros/anatomical/assets/dibothrosuchus_elaphros_anatomical.xml) · [Revised geometry](../../environments/dibothrosuchus_elaphros/anatomical/assets/dibothrosuchus_elaphros_anatomical.xml) · [Ruebenstahl et al. (2022), skull reconstructions and osteoderms](https://pmc.ncbi.nlm.nih.gov/articles/PMC9541040/)

![Original and revised side profiles at the same scale, with details of the revised skull, forefoot and hindfoot](../../environments/dibothrosuchus_elaphros/data/anatomical_review.jpg)

| Landmark | Original model | Revised model | Anatomical reference |
| --- | ---: | ---: | --- |
| Snout-tip to tail-tip sites, horizontal span at home | 1.250 m | 1.296 m | Estimated total length of 1.3 m. The tail is reconstructed. [Wu and Chatterjee (1993)](https://doi.org/10.1080/02724634.1993.10011488) |
| Skull, visible anteroposterior span | 200 mm | 164 mm | 164 mm cranial length. The front of the revised skin aligns with the snout site. [Zanno et al. (2015), Methods, Body size](https://www.nature.com/articles/srep09276) |
| Humerus, shoulder to elbow joint centers | 97.1 mm | 112.5 mm | 112.5 mm in IVPP V7907. [Bodenham et al. (2026), table 1](https://doi.org/10.1002/ar.70162) |
| Radius, elbow to wrist joint centers | 88.8 mm | 97.5 mm | 97.5 mm in IVPP V7907. The ulna, including its elbow projection, measures 110 mm. [Bodenham et al. (2026), table 1](https://doi.org/10.1002/ar.70162) |

**Forefoot.** Four compact rays share a toe hinge. The short first ray faces inward, and the third extends farthest forward. Each ray contacts the floor beside a small central sole. The fossil hand supports the absence of digit V. The lengths of the fleshed toes, amount of spread and ground-contact patches are reconstruction choices. The rig combines wrist and metacarpus into one link. [Ruebenstahl et al. (2022), fig. 33b](https://pmc.ncbi.nlm.nih.gov/articles/PMC9541040/) · [Bodenham et al. (2026), table 1, radiale and ulnare](https://doi.org/10.1002/ar.70162)

**Hindfoot.** The longer hindfeet have four supporting rays and a small, raised fifth ray on the outer side. The incomplete *Dibothrosuchus* hindlimb leaves the dimensions uncertain. The arrangement uses the described foot of *Terrestrisuchus gracilis* as a comparison. Colors, soft-tissue contours, the 8.65 kg simulated mass and its distribution are modeling choices. [Spiekman et al. (2024), fig. 30 and description of the pes](https://onlinelibrary.wiley.com/doi/full/10.1002/spp2.1577) · [Bodenham et al. (2026), table 1](https://doi.org/10.1002/ar.70162)

## Standing balance

With neutral joint commands on level ground for 6 s, the four feet carried 84.86 N. Each forefoot bore 11.34 N and each hindfoot 31.09 N. The center of mass projected 58.3 mm inside the nearest edge of the loaded contact polygon, down from 64.1 mm with the original block feet. The central cushions carried about 95% of the standing load. The revised trunk settled at 313 mm with 0.012° tilt, and the tail tip stood 238 mm above the floor. These are simulation measurements. [Home-pose checks](../../environments/dibothrosuchus_elaphros/tests/test_anatomical_static_balance.py) · [Whole-episode balance checks](../../environments/dibothrosuchus_elaphros/tests/test_com_stance_quality.py)

At the configured Stage 1 reset noise of 0.05, neutral control survived all 40 ten-second episodes (seeds 3082–3121). At the separate stance-calibration noise of 0.02, all 80 starts settled by 2 s. Four turned more than 1° from spawn (worst 1.22°). Eighteen exceeded the 2.63 mm post-landing foot-travel target (worst 3.74 mm). The corresponding revision 2 counts were seven and twenty-four. Both absolute targets remain in the suite. [Stage 1 settings](../../configs/dibothrosuchus_elaphros/anatomical/stance.toml) · [Standing calibration and target definitions](../../environments/shared/tests/com_stance_calibration.py)

## Walking mechanics

The level-ground reference places the left hind foot, left forefoot, right hind foot and right forefoot in sequence. Each foot rolls forward over its digits before lifting. Joint servos act on a freely moving body. The chosen sequence is one plausible quadrupedal walk, and living crocodylians exhibit several footfall patterns. Cadence, stride length, tail carriage and the timing of toe-off are assumptions for this reconstruction. [Hutchinson et al. (2019), comparative crocodylian gaits](https://www.nature.com/articles/s41598-019-55768-6) · [Reference controller and measurements](../../environments/dibothrosuchus_elaphros/scripts/reference_walk.py)

| Measured over six steady strides on level ground | Result |
| --- | --- |
| Forward speed and stride | 0.161 m/s, 0.128 m per stride, 0.80 s per cycle |
| Landings and support | 24 ordered touchdowns. At least two feet bore weight throughout, with zero airborne substeps and zero non-foot floor load |
| Foot placement | Minimum left–right spacing of 88 mm at the forefeet and 93 mm at the hindfeet |
| Forefoot motion | About 64% stance, 14 mm peak swing clearance and 7.4° loaded heel rise (95th percentile) |
| Hindfoot motion | About 73% stance, 9 mm peak swing clearance and 5.0–5.1° loaded heel rise (95th percentile) |
| Toe loading | The four digital rays carried 70–72% of forefoot impulse and 27–28% of hindfoot impulse. The central cushions carried the balance |
| Slip | 13 mm/s at the 95th percentile of tangential contact speed while loaded |
| Trunk, head and tail | Maximum absolute pitch 0.27° and roll 1.09°. Trunk bob 0.8 mm peak to peak, snout excursion 7.6 mm and tail-tip excursion 7.0 mm |

All eleven nine-stride checks completed in the expected order (one neutral start, five seeds at noise 0.01 and five at 0.05). Two 36-stride trials, a neutral start and seed 3 at noise 0.05, each recorded 132 correctly ordered touchdowns after the three startup strides. The small trunk excursions and repeatable cadence reflect the controller's steady-speed design. [Reproduce the reference walk](../../environments/dibothrosuchus_elaphros/scripts/reference_walk.py)

The reference uses floor force above 1% of body weight to identify a loaded foot and measures swing clearance at the lowest of its contact surfaces. The replay overlays four measured contact traces. It demonstrates one straight, slow walk on level ground. The model also exposes reinforcement-learning environments for stance, locomotion and prey contact. [Walk replay](../../website/static/videos/dibothrosuchus_elaphros_anatomical_reference_walk.mp4) · [Gait regression test](../../environments/dibothrosuchus_elaphros/tests/test_reference_walk.py)

## Environment interface

Observation and action totals are generated in the [catalog entry](../SPECIES_CATALOG.md#dibothrosuchus_elaphros-anatomical). The source of the observation layout
and action-to-actuator mapping is `environments/dibothrosuchus_elaphros/base_env.py`; actions are normalized to [-1, 1] residuals, where
zero commands the complete XML `home` control and the endpoints retain access to the full actuator ranges.

The Stage 3 “Snap” task detects contact between a fixed snout geom and a prey sphere with a 60 mm radius.
The neck and head are actuated. The jaws form a single rigid body.

### Termination conditions

- Trunk height outside the healthy range (0.18 m–0.55 m by default; the lower bound is the sprawl-collapse gate)
- Excessive tilt angle
- Nosedive (forward pitch exceeds the measured level neutral plus a threshold)
- Snout tip below 0.04 m (catches snout-propping that geom contact detection can miss)
- Skull, snout, trunk, belly, or distal tail contacts the ground
- Snap success (fixed `snout_snap` geom contacts prey; no jaw articulation)
- Episode length > max_episode_steps

## Actuator loads

The hip pitch, knee and ankle servos have a torque limit of 1.5 times their position gain. The other servos use a limit of 0.7 times their gain. The rear knee uses the opposite physical hinge axis to the fore elbow, so a flexion command folds each joint in its anatomical direction. The 4 Hz full-range stress test clipped the most constrained actuator on 1.65% of steps. To measure clipping under gait excitation, run `python -m environments.shared.scripts.actuator_saturation_report --model-id dibothrosuchus_elaphros anatomical`. [Actuator load checks](../../environments/dibothrosuchus_elaphros/tests/test_anatomical_actuator_bounds.py)

## Reproduce the trials

From the repository root, install the project with `pip install -e .` (add `".[test]"` for the tests).

```bash
# The six-stride walk in the table, printed as JSON
python -m environments.dibothrosuchus_elaphros.scripts.reference_walk
# The 36-stride walk from a perturbed start
python -m environments.dibothrosuchus_elaphros.scripts.reference_walk --strides 36 --seed 3 --reset-noise 0.05
# The standing panel at the configured Stage 1 reset noise (40 episodes from seed 3082)
python -m environments.shared.scripts.zero_action_baseline --model-id dibothrosuchus_elaphros anatomical --episodes 40 --seed 3082
# Actuator clipping under gait excitation
python -m environments.shared.scripts.actuator_saturation_report --model-id dibothrosuchus_elaphros anatomical
# The balance, walk and actuator checks linked above
pytest environments/dibothrosuchus_elaphros/tests
```

[Model and training setup](../../environments/dibothrosuchus_elaphros/README.md)
