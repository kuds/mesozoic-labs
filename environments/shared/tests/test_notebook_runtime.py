"""Tests for ``environments.shared.notebook_runtime`` with fake ``google.colab`` and ``IPython`` modules.

No Colab, IPython, SB3 or torch is needed: the fakes are installed in ``sys.modules``, which the helpers' lazy
imports read first.
"""

import sys
import time
import types

import pytest

from environments.shared.notebook_runtime import disconnect_runtime, display_stage_videos, halt


@pytest.fixture
def colab(monkeypatch):
    """A fake ``google.colab`` whose ``drive`` and ``runtime`` record their calls; ``time.sleep`` only records."""
    calls = []
    module = types.ModuleType("google.colab")
    module.drive = types.SimpleNamespace(flush_and_unmount=lambda: calls.append("flush"))
    module.runtime = types.SimpleNamespace(unassign=lambda: calls.append("unassign"))
    monkeypatch.setitem(sys.modules, "google.colab", module)
    monkeypatch.setattr(time, "sleep", lambda seconds: calls.append(f"sleep {seconds}"))
    return calls


@pytest.mark.parametrize("in_colab,auto", [(False, True), (True, False), (False, False)])
def test_outside_colab_or_with_auto_off_it_only_prints(colab, capsys, in_colab, auto):
    disconnect_runtime("Training finished.", in_colab=in_colab, auto=auto, flush_drive=True)
    assert colab == []
    assert capsys.readouterr().out == (
        "\nTraining finished.\nAuto-disconnect skipped (not in Colab or AUTO_DISCONNECT is False).\n"
    )


@pytest.mark.parametrize(
    "flush_drive,expected", [(True, ["flush", "sleep 5", "unassign"]), (False, ["sleep 5", "unassign"])]
)
def test_in_colab_it_flushes_drive_when_asked_then_releases_the_runtime(colab, flush_drive, expected):
    disconnect_runtime("Training finished.", in_colab=True, auto=True, flush_drive=flush_drive)
    assert colab == expected


def test_the_knobs_are_keyword_only_and_required(colab):
    with pytest.raises(TypeError):
        disconnect_runtime("reason")
    with pytest.raises(TypeError):
        disconnect_runtime("reason", True, True, True)
    with pytest.raises(TypeError):
        halt("reason", in_colab=True)
    assert colab == []


@pytest.mark.parametrize("auto,released", [(True, ["flush", "sleep 5", "unassign"]), (False, [])])
def test_halt_releases_the_runtime_then_raises_the_reason(colab, auto, released):
    reason = "recovery failed its curriculum gate: success rate below the rail."
    with pytest.raises(RuntimeError) as excinfo:
        halt(reason, in_colab=True, auto=auto, flush_drive=True)
    assert str(excinfo.value) == reason
    assert colab == released, "the runtime is released before the raise, and only when AUTO_DISCONNECT is on"


@pytest.fixture
def ipython(monkeypatch):
    """A fake ``IPython.display`` whose ``display`` records the ``Video`` objects it is handed."""
    shown = []

    class Video:
        def __init__(self, data, *, embed, html_attributes):
            self.data, self.embed, self.html_attributes = data, embed, html_attributes

    module = types.ModuleType("IPython.display")
    module.Video, module.display = Video, shown.append
    monkeypatch.setitem(sys.modules, "IPython.display", module)
    return shown


def test_videos_play_embedded_in_path_order_and_stance_csvs_are_skipped(ipython, tmp_path, capsys):
    replays = tmp_path / "replays"
    replays.mkdir()
    for path in (replays / "b.mp4", replays / "a.mp4", replays / "a_stance.csv", tmp_path / "legacy.mp4"):
        path.write_bytes(b"")

    display_stage_videos(str(tmp_path))

    assert [video.data for video in ipython] == [
        str(tmp_path / "legacy.mp4"),
        str(replays / "a.mp4"),
        str(replays / "b.mp4"),
    ]
    # Embedded, and muted, autoplaying and looping as mediapy's player was.
    assert {(video.embed, video.html_attributes) for video in ipython} == {(True, "controls loop autoplay muted")}
    assert capsys.readouterr().out == "Playing: legacy.mp4\nPlaying: a.mp4\nPlaying: b.mp4\n"


def test_a_stage_without_videos_never_imports_ipython(monkeypatch, tmp_path, capsys):
    monkeypatch.setitem(sys.modules, "IPython.display", None)  # an import of it would raise
    display_stage_videos(tmp_path)
    assert capsys.readouterr().out == f"No videos found in {tmp_path}\n"
