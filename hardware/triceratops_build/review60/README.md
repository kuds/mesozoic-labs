# Reproduce the 60 cm capsule-foot simulations

These four summaries describe the Option C MuJoCo model with the scripted `walk60` controller. The standing trials use neutral controls. The simulations exercise the model without a learned checkpoint or a physical robot. The commands below run from this directory with the [simulation dependencies](../rl/requirements-colab.txt) and MuJoCo 3.10.0.

```sh
python sample.py --out scratch
python measures.py --traces scratch --out scratch/metrics.json
python terrain.py --out scratch/terrain.json
python expression_probe.py --out scratch/expression_results.json
```

`sample.py` writes `static.json` and four temporary compressed traces into the chosen output directory. `measures.py` reduces the traces to `metrics.json`. The committed JSONs are the reference outputs. The scripts accept `--build` for running against another checkout. The sample and terrain programs accept `--workers` to control parallelism.

| Result | Cases and measures |
|---|---|
| [static.json](static.json) | Level, 10-second stand on seeds 0, 3 and 4. The stand checker reports settled foot loads, body tilt, height and joint effort. |
| [metrics.json](metrics.json) | Each walking trial lasts 20 seconds. Measurements use 2–20 s. `clip` and `slow` use nominal mass and friction, observation noise and seed 101, with forward commands of 0.25 and 0.15 m/s. J1 and J4 each use seed 401 at 0.25 m/s, with the walk environment’s remaining randomization enabled. J1 uses friction 1.0 and model-default contact, J4 friction 0.4 and `impratio=10`. All four use 10 ms command delay. Foot arrays are left fore, right fore, left hind, right hind. |
| [terrain.json](terrain.json) | Fifty-six episodes at 0.25 m/s, eight seeds (200000–200007) per point, with the model’s nominal sweep settings. Compare level ground, 1/2/3 cm steps and 3°/5°/8° ramps. Speed and lateral drift use completed episodes, while fall counts include every episode. |
| [expression_results.json](expression_results.json) | Twelve 12-second runs at 0.25 m/s, seeds 101–103, nominal parameters with observation noise and 10 ms command delay. Each compares the neutral and cued neck/tail on the base model and with 0.40 kg added across the body, head, tail and lower limbs. Commands and the mass distribution are in the script. The data begin at 2 s. |

Foot load shares use the vertical-contact proxy obtained by summing sole normal force over the flat test. Stance uses a 2 N threshold with 10 ms debounce and merges short contact gaps or blips at 50 ms. A moving 1 s average is removed from the head position before reporting vertical and lateral 1st-to-99th-percentile ranges. Support margins use the projected simulated center of mass and the active sole geometry. Read the undefined fraction alongside the finite-margin fraction. Expression trials quantify a prescribed cue and additional inertial load. Shell stiffness, heat transfer and expression-servo communication require separate tests.
