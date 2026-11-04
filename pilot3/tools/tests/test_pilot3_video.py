"""Unit tests of pilot3_video.py's pure parts (pytest; walking pilot 3 scratch, never merged): the frame marks (time,
caption, L/R boxes), the floor channel from substep forces, the row comparison, the score JSON's episode list and the
mp4 writer (ffmpeg; frame count read back with ffprobe). The rollout and rendering are exercised end to end by
tests/e2e_parent_cell.py on a judged cell.
"""

from __future__ import annotations

import json
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

TOOLS = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(TOOLS))
import pilot3_video as pv  # noqa: E402


def blank(h=480, w=640, v=60):
    return np.full((h, w, 3), v, dtype=np.uint8)


def test_boxes_are_filled_only_while_the_foot_is_down():
    img = pv.annotate(blank(), 1.23, [True, False], ["L", "R"], "S42 8000000 EP 03 RESET 3045 WALK")
    size, gap = 46, 12
    x_l = 640 - 2 * (size + gap) - 4
    x_r = x_l + size + gap
    left = img[8 + 4: 8 + size - 4, x_l + 4: x_l + 14]   # inside the box, clear of the letter
    right = img[8 + 4: 8 + size - 4, x_r + 4: x_r + 14]
    assert (left == (40, 220, 90)).all(axis=-1).mean() > 0.95
    assert (right == (40, 220, 90)).all(axis=-1).mean() == 0.0
    assert (img[8, x_r: x_r + size] == (235, 235, 235)).all()  # the empty box's outline


def test_the_time_and_caption_are_drawn_and_the_input_frame_is_untouched():
    src = blank()
    img = pv.annotate(src, 9.99, [False, False], ["L", "R"], "EP 01")
    assert (src == 60).all()
    assert (img[10:31, 10:200] == 255).any() and (img[454:474, 10:120] == 255).any()
    a = pv.annotate(src, 1.00, [False, False], ["L", "R"], "")
    b = pv.annotate(src, 2.00, [False, False], ["L", "R"], "")
    assert not np.array_equal(a, b)  # the time mark changes with t


def test_unknown_characters_draw_as_blanks():
    img = blank()
    pv.draw_text(img, "~~~", 5, 5)
    assert not (img == 255).any()


def test_the_floor_channel_is_half_the_substeps_above_0_1_newton():
    sub = [np.array([0.2, 0.0]), np.array([0.2, 0.05]), np.array([0.0, 0.11]), np.array([0.05, 0.0]),
           np.array([0.3, 0.0])]
    assert pv.floor_down_from_substeps(sub, 0.1).tolist() == [True, False]  # 3/5 and 1/5
    sub = [np.array([0.11, 0.1])] * 2 + [np.array([0.0, 0.0])] * 2
    assert pv.floor_down_from_substeps(sub, 0.1).tolist() == [True, False]  # exactly half; 0.1 is not above 0.1


def test_compare_rows_ignores_the_index_keys_and_names_every_difference():
    rec = {"episode": 3, "reset_seed": 3045, "length": 1000.0, "floor_duty_L": 0.61, "note": None}
    assert pv.compare_rows({"length": 1000.0, "floor_duty_L": 0.61, "note": None}, rec) == []
    assert pv.compare_rows({"length": 1000.0, "floor_duty_L": 0.6100000001, "note": None}, rec) == ["floor_duty_L"]
    assert pv.compare_rows({"length": 1000.0, "floor_duty_L": 0.61}, rec) == ["note"]


def test_episodes_from_score():
    score = {"video_review_required": {"42": {"steps": 8000000, "episodes": [
        {"episode": 3, "label": "WALK"}, {"episode": 17, "label": "OTHER"}]}}}
    assert pv.episodes_from_score(score, 42, 8000000) == {3: "WALK", 17: "OTHER"}
    with pytest.raises(SystemExit, match="not this cell's"):
        pv.episodes_from_score(score, 42, 6000000)
    with pytest.raises(SystemExit, match="no video review for s52"):
        pv.episodes_from_score(score, 52, 8000000)


@pytest.mark.skipif(shutil.which("ffmpeg") is None or shutil.which("ffprobe") is None, reason="no ffmpeg/ffprobe")
@pytest.mark.parametrize("fps", [100.0, 25.0])
def test_the_mp4_writer_writes_every_frame_at_the_rate(tmp_path, fps):
    frames = [pv.annotate(blank(96, 160, 30 + 5 * k), k * 0.01, [k % 2 == 0, k % 3 == 0], ["L", "R"], "T")
              for k in range(12)]
    path = tmp_path / "x.mp4"
    pv.encode_mp4(frames, fps, path)
    probe = subprocess.run(["ffprobe", "-v", "error", "-count_frames", "-select_streams", "v:0", "-show_entries",
                            "stream=nb_read_frames,r_frame_rate,width,height,codec_name", "-of", "json", str(path)],
                           capture_output=True, text=True, check=True)
    s = json.loads(probe.stdout)["streams"][0]
    assert (int(s["nb_read_frames"]), s["codec_name"], s["width"], s["height"]) == (12, "h264", 160, 96)
    num, den = map(int, s["r_frame_rate"].split("/"))
    assert num / den == pytest.approx(fps)


def test_an_aborted_writer_leaves_no_file(tmp_path):
    w = pv.Mp4Writer(tmp_path / "y.mp4", 25.0, 160, 96)
    w.write(blank(96, 160))
    w.abort()
    assert not (tmp_path / "y.mp4").exists()
