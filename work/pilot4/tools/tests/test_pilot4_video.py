"""Tests of pilot4_video.py (pytest; walking pilot 4 scratch, never merged).

Walking pilot 3's tests/test_pilot3_video.py (sha256 51c7822f...) adapted to pilot 4's names: its pure-part tests are
kept (the frame marks: time, caption, L/R boxes; the floor channel from substep forces; the row comparison; the score
JSON's episode list at a pilot-4 cell's steps; the mp4 writer through ffmpeg, frame count read back with ffprobe).
New for pilot 4 (the TOOL PLAN: "the RUN_ID and paths; renders unchanged"): every rendering function and constant is
pilot 3's pinned file's, syntax tree for syntax tree (read as text, never imported from pilot 3's folder), and main()
differs only by the schema constant and the plant check's label; through iso.py on the pilot-4 tree (PILOT4_TREE,
default ../tree), the env the tool re-rolls in is the pilot-4 env (the live heading command, the prey on the spawn
line, the price), a pilot-3 cell's probe.json (pilot 3's env) is refused, and a cell judged at another head is
refused, each writing nothing. Rendering itself needs an OpenGL context; these tests run with MUJOCO_GL=disable, so
no frame is rendered from the simulator here.
"""

from __future__ import annotations

import ast
import hashlib
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

TOOLS = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(TOOLS))
import pilot4_video as pv  # noqa: E402

TREE = Path(os.environ.get("PILOT4_TREE", str(TOOLS / "tree"))).resolve()
SP = Path("/tmp/claude-0/-home-user-mesozoic-labs/aff94cce-91ba-5515-b81b-a9f018e73dc9/scratchpad")
PILOT3_VIDEO = SP / "work/pilot3/tools/pilot3_video.py"
PILOT3_VIDEO_SHA256 = "43f72bfa2e95dfac57bc96cf9eb61bdb882b36b18d3fbe7a6541ec03d4ffb87e"
PILOT3_CELL = SP / "work/pilot3/judging/OUT/s42/8000000"  # pilot 3's judged s42 8M cell (read only)


def blank(h=480, w=640, v=60):
    return np.full((h, w, 3), v, dtype=np.uint8)


def test_boxes_are_filled_only_while_the_foot_is_down():
    img = pv.annotate(blank(), 1.23, [True, False], ["L", "R"], "S42 2000000 EP 03 RESET 3045 WALK")
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
    score = {"video_review_required": {"42": {"steps": 2000000, "episodes": [
        {"episode": 3, "label": "WALK"}, {"episode": 17, "label": "OTHER"}]}}}
    assert pv.episodes_from_score(score, 42, 2000000) == {3: "WALK", 17: "OTHER"}
    with pytest.raises(SystemExit, match="not this cell's"):
        pv.episodes_from_score(score, 42, 1500000)
    with pytest.raises(SystemExit, match="no video review for s52"):
        pv.episodes_from_score(score, 52, 2000000)


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


# ---------------------------------------------------------------- pilot 4: renders unchanged


def _pilot3_source() -> str:
    data = PILOT3_VIDEO.read_bytes()
    assert hashlib.sha256(data).hexdigest() == PILOT3_VIDEO_SHA256, "pilot 3's pinned pilot3_video.py changed"
    return data.decode("utf-8")


def _strip_doc(body: list) -> list:
    if body and isinstance(body[0], ast.Expr) and isinstance(body[0].value, ast.Constant) \
            and isinstance(body[0].value.value, str):
        return body[1:]
    return body


def _defs(source: str) -> dict:
    """{name: ast dump without docstrings} of the module's top-level functions, classes and assignments."""
    out = {}
    for node in ast.parse(source).body:
        if isinstance(node, (ast.FunctionDef, ast.ClassDef)):
            clone = ast.parse(ast.unparse(node)).body[0]
            clone.body = _strip_doc(clone.body)
            for sub in ast.walk(clone):
                if isinstance(sub, ast.FunctionDef):
                    sub.body = _strip_doc(sub.body) or [ast.Pass()]
            out[node.name] = ast.dump(clone)
        elif isinstance(node, ast.Assign):
            out[",".join(n.id for t in node.targets for n in ast.walk(t) if isinstance(n, ast.Name))] = \
                ast.dump(node.value)
    return out


RENDER_PARTS = ("_FONT", "WIDTH,HEIGHT", "QUARTER", "SKIP_KEYS", "SPECIES,EMBODIMENT,STAGE", "draw_text", "draw_box",
                "annotate", "Mp4Writer", "encode_mp4", "floor_down_from_substeps", "compare_rows", "sha256_file",
                "episodes_from_score", "install_probe_shims", "load_probe", "annotate_speed")


def test_every_rendering_part_is_pilot_3_s():
    p3, p4 = _defs(_pilot3_source()), _defs(Path(pv.__file__).read_text(encoding="utf-8"))
    for name in RENDER_PARTS:
        assert p4[name] == p3[name], name
    assert sorted(n for n in set(p3) | set(p4) if p3.get(n) != p4.get(n)) == ["SCHEMA", "main"]
    assert pv.SCHEMA == "walking-pilot4-video/v1" and (pv.WIDTH, pv.HEIGHT, pv.QUARTER) == (640, 480, 4)


def test_main_differs_only_by_the_schema_constant_and_the_plant_check_s_label():
    p3 = _pilot3_source()
    node = next(n for n in ast.parse(p3).body if isinstance(n, ast.FunctionDef) and n.name == "main")
    src = ast.get_source_segment(p3, node)
    for old, new in (('"schema": "walking-pilot3-video/v1"', '"schema": SCHEMA'),
                     ('pilot3_video environment"', 'pilot4_video environment"')):
        assert src.count(old) == 1, old
        src = src.replace(old, new)
    assert _defs(src)["main"] == _defs(Path(pv.__file__).read_text(encoding="utf-8"))["main"]


def test_the_usage_names_pilot_4_s_tools_and_cells():
    doc = pv.__doc__
    assert "TOOLS/pilot4_video.py OUT/s42/2000000" in doc and "VIDEOS/s42_2000000" in doc
    assert "pilot4_judge.py's OUT/s<seed>/<steps>" in doc and "pilot4_score.py's --json" in doc
    assert "pilot4_walk_s<seed>[_b]" in doc


# ---------------------------------------------------------------- pilot 4: through iso.py on the pilot-4 tree

needs_tree = pytest.mark.skipif(not (TREE / "environments" / "__init__.py").is_file(), reason=f"no tree at {TREE}")


def iso_env() -> dict:
    env = dict(os.environ, ISO_WT=str(TREE), MUJOCO_GL="disable", PYTHONDONTWRITEBYTECODE="1", OMP_NUM_THREADS="1",
               GIT_OPTIONAL_LOCKS="0")
    env.pop("PYTHONPATH", None)
    return env


def iso_run(script: Path, *args: str) -> subprocess.CompletedProcess:
    return subprocess.run([sys.executable, "-I", str(TOOLS / "iso.py"), str(script), *args], cwd=TREE, env=iso_env(),
                          capture_output=True, text=True, timeout=600)


def tree_head() -> str:
    return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=TREE, text=True, env=iso_env()).strip()


ENV_SCRIPT = '''
import json, sys
sys.path.insert(0, sys.argv[1])
import pilot4_video as pv
pv.install_probe_shims()
probe, probe_path = pv.load_probe(__import__("pathlib").Path.cwd().resolve())
import environments
from environments.shared.config import load_stage_config
from environments.shared.curriculum.stance_gate import StanceGateThresholds
cfg = load_stage_config(pv.SPECIES, pv.STAGE)
kw = json.loads(json.dumps(probe._json_safe(dict(cfg["env_kwargs"]))))
gate = int(StanceGateThresholds.from_curriculum(cfg.get("curriculum_kwargs", {}), require_criteria=False).settle_steps)
print(json.dumps({"env_kwargs": kw, "settle_steps_gate": gate, "environments": environments.__file__,
                  "probe": str(probe_path)}))
'''


@needs_tree
def test_the_env_the_tool_re_rolls_in_is_the_pilot_4_env(tmp_path):
    """The tool's own path (its shims, the tree's frozen probe, load_stage_config(SPECIES, STAGE)) builds the pilot-4
    locomotion env: the live heading command with the committed table, the prey on the spawn line, the price."""
    script = tmp_path / "env_kwargs.py"
    script.write_text(ENV_SCRIPT)
    r = iso_run(script, str(TOOLS))
    assert r.returncode == 0, (r.stdout, r.stderr)
    got = json.loads(r.stdout.strip().splitlines()[-1])
    kw = got["env_kwargs"]
    assert Path(got["environments"]).resolve().is_relative_to(TREE)
    assert kw["command_mode"] == "heading" and kw["prey_lateral_range"] == [0.0, 0.0]
    assert kw["walk_heading_price_weight"] == 0.2 and kw["walk_heading_price_scale_rad"] == 0.5
    assert kw["command_config"]["straight_probability"] == 1.0 and kw["command_config"]["cruise_speed"] == 1.5
    assert (kw["walk_support_gate_bw"], kw["airborne_substep_weight"], kw["forward_vel_max"],
            kw["walk_touchdown_weight"]) == (0.04, 1.0, 1.5, 0.1)
    assert got["settle_steps_gate"] == 0


def _cell_from_pilot3(tmp_path, head: str) -> Path:
    """A cell holding pilot 3's judged s42 8M probe.json (pilot 3's env) and its binding, with the head set to `head`;
    the pair it names is pilot 3's verified run copy's (read only, sha256 unchanged)."""
    cell = tmp_path / "OUT" / "s42" / "8000000"
    cell.mkdir(parents=True)
    shutil.copyfile(PILOT3_CELL / "probe.json", cell / "probe.json")
    b = json.loads((PILOT3_CELL / "binding.json").read_text())
    b["head"] = head
    (cell / "binding.json").write_text(json.dumps(b))
    return cell


needs_pilot3_cell = pytest.mark.skipif(not (PILOT3_CELL / "probe.json").is_file(), reason="pilot 3's cell is absent")


@needs_tree
@needs_pilot3_cell
def test_a_pilot_3_cell_s_env_is_refused_and_nothing_is_written(tmp_path):
    """Pilot 3's judged probe.json records pilot 3's env (no command, the prey drawn, no price); in the pilot-4 tree
    the tool refuses it before loading the pair (exit 2) and writes nothing."""
    cell = _cell_from_pilot3(tmp_path, tree_head())
    meta = json.loads((cell / "probe.json").read_text())["meta"]
    assert meta["env_kwargs"].get("command_mode", "none") != "heading"  # pilot 3's env, as recorded
    out = tmp_path / "videos"
    r = iso_run(TOOLS / "pilot4_video.py", str(cell), "--episodes", "0", "--out", str(out))
    assert r.returncode == 2, (r.stdout, r.stderr)
    assert "refusing (nothing written): this tree's locomotion env kwargs or gate settle differ from probe.json's" \
        in r.stdout
    assert not out.exists()


@needs_tree
@needs_pilot3_cell
def test_a_cell_judged_at_another_head_is_refused_and_nothing_is_written(tmp_path):
    cell = _cell_from_pilot3(tmp_path, "0" * 40)
    out = tmp_path / "videos"
    r = iso_run(TOOLS / "pilot4_video.py", str(cell), "--episodes", "0", "--out", str(out))
    assert r.returncode == 2, (r.stdout, r.stderr)
    assert f"the cell was judged at {'0' * 40}, this tree is {tree_head()}" in r.stdout and not out.exists()
