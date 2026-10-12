# Species and model names

A **species ID** is the species' binomial in snake_case (genus, underscore,
epithet). An **embodiment** is `anatomical`, the animal-proportioned research
model, or `robot`, a buildable robot of that species. A robot's hardware
iterations are physics revisions of the same embodiment, not new embodiments.
A **model** is the pair (species, embodiment); in code it is a `ModelId`
(`environments/shared/model_id.py`), held in variables named `model_id`
(`model` means a policy or an MjModel). There are no aliases: machine
interfaces accept the canonical IDs only.

`configs/species_manifest.toml` owns the species, their embodiments and the
display names. Title-case labels name the animal inspiring each simplified
model; they do not certify specimen-level anatomical accuracy. A model's
display name is always the species display name followed by the embodiment
label, as in "Tyrannosaurus Rex (Anatomical)". It labels logs, reports, plots,
W&B titles, the model rows of the README and the species catalog, and the
catalog's results headings; the species display name labels the species itself
(the website's species category, the species READMEs, notebook prose).
Regenerate the public catalog with
`python -m environments.shared.species_catalog` after editing the manifest.

| Species ID | Embodiment | Display name | Gymnasium ID | Model page |
|---|---|---|---|---|
| `velociraptor_mongoliensis` | `anatomical` | Velociraptor Mongoliensis (Anatomical) | `MesozoicLabs/VelociraptorMongoliensis-Anatomical-v0` | [`/docs/models/velociraptor_mongoliensis/anatomical`](../website/docs/models/velociraptor_mongoliensis/anatomical.mdx) |
| `tyrannosaurus_rex` | `anatomical` | Tyrannosaurus Rex (Anatomical) | `MesozoicLabs/TyrannosaurusRex-Anatomical-v0` | [`/docs/models/tyrannosaurus_rex/anatomical`](../website/docs/models/tyrannosaurus_rex/anatomical.mdx) |
| `brachiosaurus_altithorax` | `anatomical` | Brachiosaurus Altithorax (Anatomical) | `MesozoicLabs/BrachiosaurusAltithorax-Anatomical-v0` | [`/docs/models/brachiosaurus_altithorax/anatomical`](../website/docs/models/brachiosaurus_altithorax/anatomical.mdx) |
| `dibothrosuchus_elaphros` | `anatomical` | Dibothrosuchus Elaphros (Anatomical) | `MesozoicLabs/DibothrosuchusElaphros-Anatomical-v0` | [`/docs/models/dibothrosuchus_elaphros/anatomical`](../website/docs/models/dibothrosuchus_elaphros/anatomical.mdx) |
| `compsognathus_longipes` | `anatomical` | Compsognathus Longipes (Anatomical) | `MesozoicLabs/CompsognathusLongipes-Anatomical-v0` | [`/docs/models/compsognathus_longipes/anatomical`](../website/docs/models/compsognathus_longipes/anatomical.mdx) |
| `compsognathus_longipes` | `robot` | Compsognathus Longipes (Robot) | `MesozoicLabs/CompsognathusLongipes-Robot-v0` | [`/docs/models/compsognathus_longipes/robot`](../website/docs/models/compsognathus_longipes/robot.mdx) |

Each model has its own environment class, configs, plant revision counters,
normalization statistics and checkpoint identity; the two Compsognathus models
share their species' base class, task, rewards and stage graph.

## One rule names everything

Every machine identifier is built from the pair by one rule. `<Species>` and
`<Embodiment>` are the PascalCase forms of the IDs (`TyrannosaurusRex`,
`Anatomical`).

| Identifier | Pattern | T. rex, anatomical | Compsognathus, robot |
|---|---|---|---|
| Species base module and class | `environments.<species>.base_env:<Species>Env` | `environments.tyrannosaurus_rex.base_env:TyrannosaurusRexEnv` | `environments.compsognathus_longipes.base_env:CompsognathusLongipesEnv` |
| Embodiment package | `environments.<species>.<embodiment>` | `environments.tyrannosaurus_rex.anatomical` | `environments.compsognathus_longipes.robot` |
| Embodiment module and class | `environments.<species>.<embodiment>.env:<Species><Embodiment>Env` | `environments.tyrannosaurus_rex.anatomical.env:TyrannosaurusRexAnatomicalEnv` | `environments.compsognathus_longipes.robot.env:CompsognathusLongipesRobotEnv` |
| Behavior environment class | `<Species><Embodiment>BehaviorEnv` | `TyrannosaurusRexAnatomicalBehaviorEnv` | `CompsognathusLongipesRobotBehaviorEnv` |
| Gymnasium ID (registered from the manifest) | `MesozoicLabs/<Species>-<Embodiment>-v0` | `MesozoicLabs/TyrannosaurusRex-Anatomical-v0` | `MesozoicLabs/CompsognathusLongipes-Robot-v0` |
| MJCF | `environments/<species>/<embodiment>/assets/<species>_<embodiment>.xml`, `<mujoco model="<species>_<embodiment>">` | `environments/tyrannosaurus_rex/anatomical/assets/tyrannosaurus_rex_anatomical.xml` | `environments/compsognathus_longipes/robot/assets/compsognathus_longipes_robot.xml` |
| Configs | `configs/<species>/<embodiment>/` (stage files, `stages.toml`, `behaviors/`) | `configs/tyrannosaurus_rex/anatomical/` | `configs/compsognathus_longipes/robot/` |
| Stage files | named by the stage, with no number | `stance.toml`, `recovery.toml`, `locomotion.toml`, `behavior.toml` | `stance.toml`, `recovery.toml`, `locomotion.toml`, `behavior.toml` |
| Manifest table | `[species.<species>.embodiments.<embodiment>]` | `[species.tyrannosaurus_rex.embodiments.anatomical]` | `[species.compsognathus_longipes.embodiments.robot]` |
| Plant revisions | `[plants.<species>.<embodiment>]` in `configs/plant_versions.toml` | `[plants.tyrannosaurus_rex.anatomical]` | `[plants.compsognathus_longipes.robot]` |
| Runs | `<LOG_BASE>/<species>/<embodiment>/<algo>/<YYYYMMDD_HHMMSS>/<NN>_<stage_id>/` | `<LOG_BASE>/tyrannosaurus_rex/anatomical/ppo/<YYYYMMDD_HHMMSS>/01_stance/` | `<LOG_BASE>/compsognathus_longipes/robot/ppo/<YYYYMMDD_HHMMSS>/01_stance/` |
| Default provenance run ID | `<species>-<embodiment>-<backend>-<algo>-<hex>` | `tyrannosaurus_rex-anatomical-stable-baselines3-ppo-<hex>` | `compsognathus_longipes-robot-stable-baselines3-ppo-<hex>` |
| Results | `results/<species>/<embodiment>/<algo>/summary.json` | `results/tyrannosaurus_rex/anatomical/ppo/summary.json` | `results/compsognathus_longipes/robot/ppo/summary.json` |
| Model page | `website/docs/models/<species>/<embodiment>.mdx`, served at `/docs/models/<species>/<embodiment>` | `/docs/models/tyrannosaurus_rex/anatomical` | `/docs/models/compsognathus_longipes/robot` |
| Catalog anchor | `<species>-<embodiment>` | `tyrannosaurus_rex-anatomical` | `compsognathus_longipes-robot` |
| Replay and stage videos | `<species>_<embodiment>_<algo>_stage<N>_<label>.mp4` | `tyrannosaurus_rex_anatomical_ppo_stage1_best.mp4` | `compsognathus_longipes_robot_ppo_stage1_best.mp4` |
| W&B run | ID `<species>-<embodiment>-s<stage>-<timestamp>`, tags `[species, embodiment, stage]` | `tyrannosaurus_rex-anatomical-s1-<timestamp>` | `compsognathus_longipes-robot-s1-<timestamp>` |
| Recovery calibration ID | `<species>-<embodiment>-fixed-command-recovery-<date>-v<N>` | `tyrannosaurus_rex-anatomical-fixed-command-recovery-<date>-v<N>` | `compsognathus_longipes-robot-fixed-command-recovery-2026-09-v1` |
| Notebook | `SPECIES` and `EMBODIMENT` dropdowns, `MODEL_ID = require_model_id(SPECIES, EMBODIMENT)` | `SPECIES = "tyrannosaurus_rex"`, `EMBODIMENT = "anatomical"` | `SPECIES = "compsognathus_longipes"`, `EMBODIMENT = "robot"` |
| Records | `"species"` and `"embodiment"`, two fields | `{"species": "tyrannosaurus_rex", "embodiment": "anatomical"}` | `{"species": "compsognathus_longipes", "embodiment": "robot"}` |

Two command-line forms exist. Tools that act on one model take
`--species <species> --embodiment <embodiment>`, both required:

```bash
python -m environments.shared.train --species tyrannosaurus_rex --embodiment anatomical train --stage 1
```

Tools that report on several models take `--model-id <species> <embodiment>`,
repeated, and keep their default panel when it is omitted. The `--model
PATH.zip` flags of the report tools still name a policy file.

Joined forms such as `tyrannosaurus_rex/anatomical` in log messages are built,
never parsed; records carry `species` and `embodiment` as two fields, and every
identity check compares both, so the two Compsognathus models never pass for
each other. Identity flows from the model to the path, never from a path to
the model.

## Where material lives

- **Species level, `environments/<species>/`:** the README and other species
  docs, `data/`, `scripts/`, `tests/`, and the species base class in
  `base_env.py`, which holds the observation, action and sensor code every
  embodiment of the species shares.
- **Embodiment level, `environments/<species>/<embodiment>/`:** the
  environment module `env.py` with the embodiment's body data (`MODEL_PATH`,
  `FOOT_SUPPORT`, `REWARD_PROBE`), `assets/`, and a robot's third-party
  `references/`; and `configs/<species>/<embodiment>/`, which also holds
  per-embodiment data such as `recovery_calibration.json`.
- **Shared robot infrastructure** (servo and IMU models, system
  identification, the ROS 2 bridge) belongs in one shared package, never per
  species. None of it is built yet.

## Adding a robot to a species

A robot is a new embodiment of an existing species: a folder under the
species, a config folder, a manifest table and a plant table, with no edit to
shared code. [CONTRIBUTING.md](../CONTRIBUTING.md#adding-an-embodiment-a-robot)
lists every step, and its "Adding a species" section covers a new species.

## History

The IDs changed on 2026-10-11 (decision D-D33). The CHANGELOG's D-D33
Migration entry maps every old identifier to its new name, and documents dated
before it use the old names ([docs map](README.md#investigations--run-analyses)).

Name references: [Velociraptor mongoliensis (AMNH)](https://www.amnh.org/explore/ology/ology-cards/018-velociraptor-mongoliensis),
[Brachiosaurus altithorax (Field Museum)](https://www.fieldmuseum.org/blog/why-we-dont-dress-sue-or-any-other-real-skeletons),
and [Compsognathus longipes (Natural History Museum)](https://www.nhm.ac.uk/discover/dino-directory/compsognathus.html).
Tyrannosaurus rex and Dibothrosuchus elaphros are already named in their
species READMEs.
