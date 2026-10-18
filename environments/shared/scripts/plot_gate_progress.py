"""Plot a stage's ``gate_progress.npz`` mid-run, from a copy of its stage directory (a Drive download, say).

``generate_stage_artifacts`` draws the same figure into ``figures/gate_progress.png`` when the stage ends
(:func:`environments.shared.visualization.plot_gate_progress`).  This draws it from a stage directory at
any point: the declared bars come from its ``stage_config.json`` (``config.read_recorded_stage_config``,
the shared lenient reader), the model from the same record.

The figure goes to the current directory by default (``gate_progress_<stage dir name>.png``), never into
the run: a result bundle's manifest hashes every file under the run directory
(``result_bundle/manifest.py``) and verification rejects a file it did not list, so a figure written
into a published run would make it fail verification.  An ``--out`` inside the run directory (the
nearest ancestor of STAGE_DIR holding ``provenance.json``) is refused.

Usage::

    python environments/shared/scripts/plot_gate_progress.py RUN/01_stance
    python environments/shared/scripts/plot_gate_progress.py RUN/01_stance --out /tmp/s60.png
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Sequence

_repo_root = str(Path(__file__).resolve().parents[3])
if _repo_root not in sys.path:
    sys.path.insert(0, _repo_root)

from environments.shared.model_id import ModelId  # noqa: E402


def run_directory(stage_dir: Path) -> Path | None:
    """The run directory a stage directory belongs to: its nearest ancestor (or itself) holding ``provenance.json``."""
    for candidate in (stage_dir, *stage_dir.parents):
        if (candidate / "provenance.json").is_file():
            return candidate
    return None


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n", 1)[0])
    parser.add_argument("stage_dir", type=Path, help="the stage directory holding gate_progress.npz")
    parser.add_argument("--out", type=Path, default=None, help="the PNG to write (default: the current directory)")
    args = parser.parse_args(argv)
    stage_dir = args.stage_dir.resolve()
    if not (stage_dir / "gate_progress.npz").is_file():
        print(f"no gate_progress.npz in {stage_dir}", file=sys.stderr)
        return 1
    out = (args.out if args.out is not None else Path.cwd() / f"gate_progress_{stage_dir.name}.png").resolve()
    run = run_directory(stage_dir)
    if run is not None and out.is_relative_to(run.resolve()):
        print(
            f"refusing to write {out} inside the run directory {run}: the result bundle's manifest "
            "(result_bundle/manifest.py) hashes every file there, and verification rejects a file it did not list. "
            "Pass an --out outside the run.",
            file=sys.stderr,
        )
        return 2
    from environments.shared.config import read_recorded_stage_config  # through file_io.read_json_object

    record = read_recorded_stage_config(stage_dir) or {}
    curriculum = record.get("curriculum")
    model_id = None
    if isinstance(record.get("species"), str) and isinstance(record.get("embodiment"), str):
        model_id = ModelId(record["species"], record["embodiment"])

    import matplotlib

    matplotlib.use("Agg")
    from environments.shared.visualization import plot_gate_progress

    figure = plot_gate_progress(
        stage_dir,
        curriculum_kwargs=curriculum if isinstance(curriculum, dict) else {},
        model_id=model_id,
        save_path=out,
        show=False,
    )
    if figure is None:
        print(f"gate_progress.npz in {stage_dir} holds no row to plot", file=sys.stderr)
        return 1
    print(out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
