"""Draw one or more stages' training history from their ledgers: PNG pages, an HTML page and a JSON summary.

Reads each stage directory's ``eval_episodes.csv``, ``train_episodes.csv``,
``stage_config.json``, ``progress.json``, ``plant_identity.json`` and
checkpoint zips (:mod:`environments.shared.reporting.training_history`) and
writes, into ``--out``: one PNG per page, ``training_history.html``
(self-contained: the pages embedded as base64) and ``training_history.json``
(``mesozoic.training-history/v1``).  One stage directory gives that run's page
set; several give the comparison set, one colour per seed (else per run) and
one line style per ``--groups`` value.  Labels must be distinct.  On a
comparison, a run that records no CoP band is counted against the edges the
band-paying runs share; ``--reference-band`` gives the edges explicitly (for a
single control run, say).  ``--judged`` names, per stage directory, a
directory of that run's judge files (``judge_<steps>.json``,
``judge_final.json``; ``-`` for none): the outcomes page then draws K3's
inputs at each judged checkpoint (the stance3, stance4, stance5 and stance5
replication judges' files, whose K3 is the same), as a readout of the stance5
replication judge's K3 criterion, never a verdict.  A run that is still
training can be drawn from a copy of its stage directory at any point.

Report only.  Nothing is written into a run: an ``--out`` inside any run
directory is refused, as ``plot_gate_progress.py`` refuses it (a result
bundle's manifest hashes every file there, and verification rejects a file it
did not list).  The run directory is the nearest ancestor of STAGE_DIR holding
``provenance.json``; for a copy without one, the stage directory's parent
(the ``<run>/<NN_stage>`` layout).  The refusal is for manual writes: the
copy in each stage's ``figures/training_history/`` is the library's own
artifact step (``generate_stage_artifacts`` calls the module directly, before
the bundle is saved), not this script.

Usage::

    python environments/shared/scripts/plot_training_history.py RUN/01_stance --out /tmp/history_s42 \\
        --mark-steps 4194304 5242880 6291456 8388608 10485760 final
    python environments/shared/scripts/plot_training_history.py A/01_stance B/01_stance --out /tmp/compare \\
        --labels a b --groups band control --title "Band against control"
    python environments/shared/scripts/plot_training_history.py RUN/01_stance --out /tmp/history_s70 \\
        --judged JUDGED/stance5_LTSB_s70 --mark-steps 6291456 final
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Sequence

_repo_root = str(Path(__file__).resolve().parents[3])
if _repo_root not in sys.path:
    sys.path.insert(0, _repo_root)

from environments.shared.scripts.plot_gate_progress import run_directory  # noqa: E402


def protected_directories(stage_dir: Path) -> list[Path]:
    """The directories an output must stay out of: the run directory (by ``provenance.json``, else the layout)."""
    stage = stage_dir.resolve()
    run = run_directory(stage)
    found = [run.resolve()] if run is not None else []
    found.append(stage.parent)
    return found


def refusal(out: Path, stage_dirs: Sequence[Path]) -> str | None:
    """Why *out* may not be written, or ``None``: it lies inside one of the stage directories' run directories."""
    target = out.resolve()
    for stage in stage_dirs:
        for protected in protected_directories(stage):
            if target == protected or target.is_relative_to(protected):
                return (
                    f"refusing to write {target} inside the run directory {protected}: the result bundle's manifest "
                    "(result_bundle/manifest.py) hashes every file there, and verification rejects a file it did not "
                    "list. Pass an --out outside the run."
                )
    return None


def _mark(text: str) -> int | str:
    if text.strip().lower() == "final":
        return "final"
    try:
        return int(text.replace(",", "").replace("_", ""))
    except ValueError:
        raise argparse.ArgumentTypeError(f"a mark is a step count or 'final', not {text!r}") from None


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n", 1)[0])
    parser.add_argument("stage_dirs", nargs="+", type=Path, metavar="STAGE_DIR", help="a stage directory (read only)")
    parser.add_argument("--out", type=Path, required=True, help="the directory to write (outside every run)")
    parser.add_argument("--labels", nargs="+", default=None, help="one label per STAGE_DIR (default: the run name)")
    parser.add_argument("--groups", nargs="+", default=None, help="one group per STAGE_DIR (one line style each)")
    parser.add_argument(
        "--mark-steps", nargs="+", type=_mark, default=[], help="steps to mark (J checkpoints); 'final' for the final"
    )
    parser.add_argument("--title", default=None, help="the pages' title")
    parser.add_argument(
        "--smooth",
        type=int,
        default=5,
        help="comparison pages: centred rolling mean over N evaluations of the continuous series (counts never)",
    )
    parser.add_argument(
        "--reference-band",
        nargs=4,
        type=float,
        default=None,
        metavar=("LOWER", "UPPER", "PAD_LOWER", "PAD_UPPER"),
        help="band edges to count a run that records no band against (default on a comparison: the band runs' own)",
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=None,
        help="PNG resolution (default training_history.DEFAULT_DPI, 96: point sizes as a browser draws them)",
    )
    parser.add_argument("--no-std", action="store_true", help="skip the per-actuator std (no torch import)")
    parser.add_argument(
        "--judged",
        nargs="+",
        default=None,
        metavar="JUDGED_DIR",
        help="one per STAGE_DIR ('-' for none): a directory of that run's judge files (judge_<steps>.json, "
        "judge_final.json), read for K3's inputs (the stance3 to stance5 judges' files: the same K3), drawn as a "
        "readout of the stance5 replication judge's K3 criterion, never a verdict",
    )
    args = parser.parse_args(argv)
    stage_dirs = [path.resolve() for path in args.stage_dirs]
    for name, values in (("--labels", args.labels), ("--groups", args.groups), ("--judged", args.judged)):
        if values is not None and len(values) != len(stage_dirs):
            parser.error(f"{name} needs one value per STAGE_DIR ({len(stage_dirs)}), got {len(values)}")
    if args.labels is not None and len(set(args.labels)) != len(args.labels):
        parser.error(f"--labels must be distinct, got {args.labels}")
    reason = refusal(args.out, stage_dirs)
    if reason is not None:
        print(reason, file=sys.stderr)
        return 2
    missing = [str(path) for path in stage_dirs if not path.is_dir()]
    if missing:
        print(f"no such stage directory: {', '.join(missing)}", file=sys.stderr)
        return 1

    from environments.shared.reporting import training_history

    if len(stage_dirs) > training_history.MAX_RUNS:
        print(f"at most {training_history.MAX_RUNS} stage directories per comparison", file=sys.stderr)
        return 1
    runs = [
        training_history.load_run_history(
            stage,
            label=args.labels[i] if args.labels else None,
            group=args.groups[i] if args.groups else None,
            read_std=not args.no_std,
            judged=None if not args.judged or args.judged[i] == "-" else Path(args.judged[i]),
        )
        for i, stage in enumerate(stage_dirs)
    ]
    labels = [run.label for run in runs]
    if len(set(labels)) != len(labels):
        print(f"two stage directories share a label ({labels}); pass distinct --labels", file=sys.stderr)
        return 1
    if args.reference_band is not None:
        edges = dict(zip(("lower", "upper", "pad_lower", "pad_upper"), args.reference_band))
        training_history.apply_reference_band(runs, edges)
    elif len(runs) > 1:
        training_history.apply_reference_band(runs)
    title = args.title
    if title is None:
        first = runs[0]
        model = first.model_display or "training history"
        title = f"{model}: {first.label}" if len(runs) == 1 else f"{model}: {', '.join(run.label for run in runs)}"
    written = training_history.write_report(
        runs,
        args.out,
        title=title,
        marks=args.mark_steps,
        smooth=args.smooth,
        dpi=args.dpi or training_history.DEFAULT_DPI,
    )
    for error in written["errors"]:
        print(f"warning: {error}", file=sys.stderr)
    print(written["html"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
