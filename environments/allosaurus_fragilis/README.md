# Allosaurus Fragilis

This reconstruction uses a free pelvis, ten torque-controlled hind-limb joints and two passive tail joints. An ordinary walking foot has three forward load-bearing digits and a raised inner toe. The limbs are proportioned against published Allosaurus material. The [anatomical review](../../docs/reviews/ALLOSAURUS_ANATOMICAL_REFERENCE.md) gives the measurements, floor-contact results and limits of the reconstruction.

## Models

| | Anatomical |
|---|---|
| Species / embodiment | `allosaurus_fragilis` / `anatomical` |
| Gymnasium ID | `MesozoicLabs/AllosaurusFragilis-Anatomical-v0` |
| Environment | [AllosaurusFragilisAnatomicalEnv](anatomical/env.py) |
| MuJoCo asset | [Allosaurus anatomical model](anatomical/assets/allosaurus_fragilis_anatomical.xml) |
| Stages | [Standing and walking](../../configs/allosaurus_fragilis/anatomical/stages.toml) |
| Catalog | [Allosaurus Fragilis (Anatomical)](../../docs/SPECIES_CATALOG.md#allosaurus_fragilis-anatomical) |

## Run the model

From the repository root, install with `python -m pip install -e .`. Run either reference trial from the same environment.

```bash
python -m environments.allosaurus_fragilis.scripts.audit_motion --mode stand --seconds 20
python -m environments.allosaurus_fragilis.scripts.audit_motion --mode walk --seconds 30
```

The audit reports the measured ground loads, touchdown positions, foot slip and trunk motion. Add `--trace walk.npz` to save the poses, contact loads and foot positions. For training, use the shared CLI or select `allosaurus_fragilis`, `anatomical` and `stance` or `walk` in the SB3 notebook.
