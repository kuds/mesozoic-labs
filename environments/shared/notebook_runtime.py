"""The SB3 notebook's Colab session helpers: release the runtime, halt "Run all", play a stage's videos.

``google.colab`` and ``IPython`` are imported only inside the calls that need them, so this module imports (and
is tested) without either. The notebook passes its values at call time: ``in_colab=IN_COLAB``,
``auto=AUTO_DISCONNECT`` and ``flush_drive=RUNS_ON_DRIVE``. ``RUNS_ON_DRIVE`` is the storage cell's resolved value,
never the ``USE_GOOGLE_DRIVE`` knob, which can change after that cell placed the run: it is True only while every
pass of the storage cell in this runtime put ``LOG_BASE`` on Drive. Otherwise a run lives on the VM's own disk,
which Colab deletes with the runtime, so the runtime is never released automatically.
"""

import time
from pathlib import Path
from typing import NoReturn

from environments.shared.reporting import stage_layout


def disconnect_runtime(reason: str, *, in_colab: bool, auto: bool, flush_drive: bool) -> None:
    """Print *reason*, flush Drive writes, and release the Colab runtime.

    Called whenever training halts (gate failure or completion) so an unattended "Run all" never leaves a GPU
    runtime idle. No-op outside Colab or when *auto* is False, and in Colab without Drive (*flush_drive*, the
    storage cell's ``RUNS_ON_DRIVE``, False): a run then lives only on the VM's disk, which releasing the runtime
    deletes seconds after the run ends, so the runtime is kept and the message says to download the results first.
    """
    print(f"\n{reason}")
    if not (in_colab and auto):
        print("Auto-disconnect skipped (not in Colab or AUTO_DISCONNECT is False).")
        return
    if not flush_drive:
        print(
            "Auto-disconnect skipped: the storage cell kept a run of this runtime on its own disk "
            "(USE_GOOGLE_DRIVE was False when it ran, so RUNS_ON_DRIVE is False), and releasing the runtime would "
            "delete it. Download the run directory, then disconnect the runtime yourself."
        )
        return
    # Flush buffered Drive writes before the runtime dies, otherwise
    # the tail of files written this session (e.g. evaluations.npz)
    # can be lost or truncated.
    from google.colab import drive

    print("Flushing Google Drive writes...")
    drive.flush_and_unmount()
    from google.colab import runtime

    print("Disconnecting runtime in 5 seconds...")
    time.sleep(5)
    runtime.unassign()


def halt(reason: str, *, in_colab: bool, auto: bool, flush_drive: bool) -> NoReturn:
    """Release the runtime as :func:`disconnect_runtime` does, then raise ``RuntimeError(reason)``.

    The raise stops "Run all" before the last cell; without Drive in Colab the runtime is kept, as there.
    """
    disconnect_runtime(reason, in_colab=in_colab, auto=auto, flush_drive=flush_drive)
    raise RuntimeError(reason)


def display_stage_videos(stage_dir) -> None:
    """Play *stage_dir*'s replay videos inline: embedded (Colab cannot fetch a local mp4), autoplaying and looping."""
    stage_dir = Path(stage_dir)
    # Resolved through stage_layout: replays live in `replays/` now, and this
    # also finds them in a legacy flat stage directory.
    mp4s = sorted(p for p in stage_layout.iter_replay_files(stage_dir) if p.suffix == ".mp4")
    if not mp4s:
        print(f"No videos found in {stage_dir}")
        return
    from IPython.display import Video, display

    for mp4 in mp4s:
        print(f"Playing: {mp4.name}")
        display(Video(str(mp4), embed=True, html_attributes="controls loop autoplay muted"))
