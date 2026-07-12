"""Report per-actuator force saturation on the raptor plant.

Thin wrapper kept at this documented path; the measurement now lives in
``environments/shared/scripts/actuator_saturation_report.py`` so every
species runs the identical excitation (docs/investigations/
STAGE2_RECOMMENDATIONS.md). Extra CLI args are forwarded, so this also
accepts a cadence override, e.g. ``... velociraptor:1.5`` style args of
the shared script.

Usage::

    python environments/velociraptor/scripts/actuator_saturation_report.py
"""

import sys
from pathlib import Path

# Add repo root to path
_repo_root = str(Path(__file__).resolve().parents[3])
if _repo_root not in sys.path:
    sys.path.insert(0, _repo_root)

from environments.shared.scripts.actuator_saturation_report import main

if __name__ == "__main__":
    main(sys.argv[1:] or ["velociraptor"])
