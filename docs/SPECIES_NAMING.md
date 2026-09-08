# Species names in training

`configs/species_manifest.toml` owns the full display names used by notebooks,
training plots, text summaries, the generated README and the website catalog.
Display labels use title case. They name the animal inspiring each simplified
model; they do not certify specimen-level anatomical accuracy.

| Display name | Stable ID | SB3 support |
|---|---|---|
| Velociraptor Mongoliensis | `velociraptor` | Registered environment and curriculum |
| Tyrannosaurus Rex | `trex` | Registered environment and curriculum |
| Brachiosaurus Altithorax | `brachiosaurus` | Registered environment and curriculum |
| Dibothrosuchus Elaphros | `dibothrosuchus` | Registered environment and curriculum |
| Compsognathus Longipes | `compsognathus` | Anatomical environment and PPO/SAC curriculum |
| Compsognathus Longipes (Robot) | `compsognathus_robot` | Robot environment and PPO/SAC curriculum |

Notebook selectors accept the full names. The shared SB3 registry also accepts
the stable IDs and existing aliases (`raptor`, `t-rex`, `brachio`, `dibo`),
ignoring case, spaces, underscores and hyphens. For example:

```bash
python -m environments.shared.train --species "Tyrannosaurus Rex" train --stage 1
```

Config directories, Python imports, Gymnasium IDs, saved checkpoint locations
and machine-readable result species fields keep their existing IDs. Historical
result artifacts are not renamed. Regenerate public catalog labels with
`python -m environments.shared.species_catalog` after editing the manifest.

Both Compsognathus variants are training selections in the SB3 notebook, with
separate environment classes, configs, normalization statistics and checkpoint
identities. The Ray Tune notebook includes both
variants with initial PPO/SAC search spaces. JAX/MJX support remains limited to
the original four species; backend-specific selectors follow the manifest.
See the [training guide](../environments/compsognathus/README.md#training).

Name references: [Velociraptor mongoliensis (AMNH)](https://www.amnh.org/explore/ology/ology-cards/018-velociraptor-mongoliensis),
[Brachiosaurus altithorax (Field Museum)](https://www.fieldmuseum.org/blog/why-we-dont-dress-sue-or-any-other-real-skeletons),
and [Compsognathus longipes (Natural History Museum)](https://www.nhm.ac.uk/discover/dino-directory/compsognathus.html).
Tyrannosaurus rex and Dibothrosuchus elaphros are already named in their
environment READMEs.
