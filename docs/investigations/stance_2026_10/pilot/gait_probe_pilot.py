"""Run the frozen 2026-09 gait-audit probe (docs/investigations/gait_2026_09/gait_probe.py) on today's API.

Scratch adapter for the walking pilot; never merged.  The probe was written at 7ae0a19, before decision D-D33 made a
model a (species, embodiment) pair: it calls ``SPECIES_FACTORIES[species]``, ``load_stage_config(species, ...)``,
``load_stage_manifest(species)`` and ``current_plant_identity(species)`` with a bare species string, which the current
library no longer accepts.  This adapter installs those four names on the current modules, each mapping the bare
species string to ``require_model_id(species, --embodiment)`` and calling the current function, then runs the probe's
own ``main()`` from the tree under test.  Not one line of the probe's measurement, event, metric or classification
code is copied or changed, so its numbers are the audit's definitions (floor truth through ``_substep_probe_hook``,
2-step debounce and merge, ``classify_biped``).

Run through iso.py with the tree as the working directory:
    ISO_WT=<tree> python3 iso.py gait_probe_pilot.py --embodiment anatomical -- <gait_probe.py arguments>
"""

from __future__ import annotations

import importlib.util
import os
import sys
from pathlib import Path


def main() -> int:
    argv = sys.argv[1:]
    embodiment = "anatomical"
    if argv[:1] == ["--embodiment"]:
        embodiment, argv = argv[1], argv[2:]
    if argv[:1] == ["--"]:
        argv = argv[1:]

    from environments.shared import config, plant_contract, species_registry, stage_manifest
    from environments.shared.model_id import ModelId
    from environments.shared.species_names import require_model_id

    def mid(species):
        return species if isinstance(species, ModelId) else require_model_id(str(species), embodiment)

    _load_stage_config = config.load_stage_config
    _load_stage_manifest = stage_manifest.load_stage_manifest
    _current_plant_identity = plant_contract.current_plant_identity

    class _Factories(dict):
        def __missing__(self, species):
            return lambda: species_registry.get_species_config(mid(species))

    species_registry.SPECIES_FACTORIES = _Factories()  # type: ignore[attr-defined]
    config.load_stage_config = lambda species, stage, *a, **k: _load_stage_config(mid(species), stage, *a, **k)
    stage_manifest.load_stage_manifest = lambda species, *a, **k: _load_stage_manifest(mid(species), *a, **k)
    plant_contract.current_plant_identity = lambda species, *a, **k: _current_plant_identity(mid(species), *a, **k)

    probe_path = Path.cwd() / "docs" / "investigations" / "gait_2026_09" / "gait_probe.py"
    spec = importlib.util.spec_from_file_location("gait_probe_frozen", probe_path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    print(f"[gait_probe_pilot] probe {probe_path} (embodiment {embodiment}); PYTHONPATH head "
          f"{os.environ.get('PYTHONPATH', '').split(os.pathsep)[0]}", flush=True)
    return int(module.main(argv) or 0)


if __name__ == "__main__":
    sys.exit(main())
