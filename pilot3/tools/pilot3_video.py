"""Render a judged cell's panel episodes for walking pilot 3's video review (criteria_pilot3.txt "Video review" (b)).

Scratch, never merged. Pilot 1's tools/pilot_video.py (sha256 c7079c1d...) is lost; this is a rebuild to its contract
(pilot 1's criteria, "Video review" (b)): the tool re-rolls the panel through the frozen gait probe's own rollout
(gait_probe.rollout_episode, with the same env, loader and deterministic predict as the probe's main) and calls a
video the probe's episode only if every metric of the re-rolled episode (gait_probe.analyze_episode) equals
probe.json's; it writes a real-time and a quarter-speed mp4, each frame marked with the time and two boxes, L and R,
filled while that foot is on the floor (the probe's raw floor-truth channel: whole-leg floor normal force above 0.1 N
on at least half of the control step's substeps, before any debounce).

    cd TREE && ISO_WT=TREE MUJOCO_GL=glfw xvfb-run -a python3 -I TOOLS/iso.py TOOLS/pilot3_video.py OUT/s42/8000000
        (--from-score SCORE.json | --episodes 3,17) --out VIDEOS/s42_8000000 [--standalone]

* CELL is pilot3_judge.py's OUT/s<seed>/<steps>: binding.json names the checkpoint pair (sha256s re-checked) and the
  head the tree must be at; probe.json holds the rows to match and the probe's env_kwargs, settle steps and panel.
* --from-score reads pilot3_score.py's --json: the cell's seed's "video_review_required" episodes at this cell's steps
  (every WALK, and every episode that is not STILL with a foot's floor duty above 0.8). --episodes names them directly.
* The panel is re-rolled in order from episode 0 through the last named one, in one env, as the probe rolled it
  (every rolled episode is compared, rendered or not); --standalone rolls only the named episodes (faster; the
  comparison then shows whether an episode depends on the ones before it).
* A named episode whose re-rolled metrics differ from probe.json's gets no video (it is listed, with the keys that
  differ) and the exit status is 1.
* Frames: the env's own camera (the notebook replay's view: T. rex tracking the pelvis, distance 3.0, azimuth 135,
  elevation -20), 640x480, one frame per control step after it (t = (k+1) dt), real time at 1/dt frames per second,
  quarter speed at a quarter of that, streamed to ffmpeg (libx264, yuv420p) as the episode rolls. The boxes read a
  hook chained after the probe's substep hook that sums each foot's floor force as the probe's does; the episode's
  boxes must equal the probe's floor_contact_substep_frac >= 0.5 at every step, else its videos are deleted.
* Writes OUT_DIR/s<seed>_<steps>_ep<NN>_reset<seed>_{realtime,quarter}.mp4 and OUT_DIR/videos.json (per episode:
  label, match, differing keys, files and their sha256). Rendering needs an OpenGL context: MUJOCO_GL=glfw under
  xvfb-run here (egl on a GPU host). MESA_SHADER_CACHE_DISABLE defaults to true here, so Mesa writes no shader cache
  under the home folder; Xvfb itself keeps its socket folder under the system temp folder.
* The boxes are drawn L then R, whatever the probe's foot order.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np

SPECIES, EMBODIMENT, STAGE = "tyrannosaurus_rex", "anatomical", "locomotion"
WIDTH, HEIGHT = 640, 480
QUARTER = 4
SKIP_KEYS = ("episode", "reset_seed")

# ---------------------------------------------------------------- a 5x7 bitmap font (frame labels only)
_FONT = {
    "0": ["01110", "10001", "10011", "10101", "11001", "10001", "01110"],
    "1": ["00100", "01100", "00100", "00100", "00100", "00100", "01110"],
    "2": ["01110", "10001", "00001", "00010", "00100", "01000", "11111"],
    "3": ["11110", "00001", "00001", "01110", "00001", "00001", "11110"],
    "4": ["00010", "00110", "01010", "10010", "11111", "00010", "00010"],
    "5": ["11111", "10000", "11110", "00001", "00001", "10001", "01110"],
    "6": ["00110", "01000", "10000", "11110", "10001", "10001", "01110"],
    "7": ["11111", "00001", "00010", "00100", "01000", "01000", "01000"],
    "8": ["01110", "10001", "10001", "01110", "10001", "10001", "01110"],
    "9": ["01110", "10001", "10001", "01111", "00001", "00010", "01100"],
    "A": ["01110", "10001", "10001", "11111", "10001", "10001", "10001"],
    "B": ["11110", "10001", "10001", "11110", "10001", "10001", "11110"],
    "C": ["01110", "10001", "10000", "10000", "10000", "10001", "01110"],
    "D": ["11110", "10001", "10001", "10001", "10001", "10001", "11110"],
    "E": ["11111", "10000", "10000", "11110", "10000", "10000", "11111"],
    "F": ["11111", "10000", "10000", "11110", "10000", "10000", "10000"],
    "G": ["01110", "10001", "10000", "10111", "10001", "10001", "01111"],
    "H": ["10001", "10001", "10001", "11111", "10001", "10001", "10001"],
    "I": ["01110", "00100", "00100", "00100", "00100", "00100", "01110"],
    "K": ["10001", "10010", "10100", "11000", "10100", "10010", "10001"],
    "L": ["10000", "10000", "10000", "10000", "10000", "10000", "11111"],
    "M": ["10001", "11011", "10101", "10101", "10001", "10001", "10001"],
    "N": ["10001", "11001", "10101", "10011", "10001", "10001", "10001"],
    "O": ["01110", "10001", "10001", "10001", "10001", "10001", "01110"],
    "P": ["11110", "10001", "10001", "11110", "10000", "10000", "10000"],
    "R": ["11110", "10001", "10001", "11110", "10100", "10010", "10001"],
    "S": ["01111", "10000", "10000", "01110", "00001", "00001", "11110"],
    "T": ["11111", "00100", "00100", "00100", "00100", "00100", "00100"],
    "U": ["10001", "10001", "10001", "10001", "10001", "10001", "01110"],
    "W": ["10001", "10001", "10001", "10101", "10101", "10101", "01010"],
    "X": ["10001", "10001", "01010", "00100", "01010", "10001", "10001"],
    "Y": ["10001", "10001", "01010", "00100", "00100", "00100", "00100"],
    ".": ["00000", "00000", "00000", "00000", "00000", "01100", "01100"],
    "=": ["00000", "00000", "11111", "00000", "11111", "00000", "00000"],
    "-": ["00000", "00000", "00000", "11111", "00000", "00000", "00000"],
    "/": ["00001", "00010", "00010", "00100", "01000", "01000", "10000"],
    ":": ["00000", "01100", "01100", "00000", "01100", "01100", "00000"],
    " ": ["00000"] * 7,
}


def draw_text(img: np.ndarray, text: str, x: int, y: int, scale: int = 2, color=(255, 255, 255)) -> None:
    """Draw *text* (upper-cased; unknown characters as blanks) at (x, y), in place, with a dark backing box."""
    text = text.upper()
    w = len(text) * 6 * scale
    h = 7 * scale
    y0, y1 = max(y - scale, 0), min(y + h + scale, img.shape[0])
    x0, x1 = max(x - scale, 0), min(x + w + scale, img.shape[1])
    img[y0:y1, x0:x1] = (img[y0:y1, x0:x1] * 0.35).astype(img.dtype)
    for i, ch in enumerate(text):
        glyph = _FONT.get(ch, _FONT[" "])
        for r, line in enumerate(glyph):
            for c, bit in enumerate(line):
                if bit == "1":
                    yy, xx = y + r * scale, x + (i * 6 + c) * scale
                    img[yy: yy + scale, xx: xx + scale] = color


def draw_box(img: np.ndarray, x: int, y: int, size: int, filled: bool, label: str) -> None:
    """A size x size box at (x, y): solid when *filled* (the foot is on the floor), else a 3-pixel outline."""
    on, off = (40, 220, 90), (235, 235, 235)
    if filled:
        img[y: y + size, x: x + size] = on
    else:
        img[y: y + size, x: x + size] = (img[y: y + size, x: x + size] * 0.35).astype(img.dtype)
        for t in range(3):
            img[y + t, x: x + size] = off
            img[y + size - 1 - t, x: x + size] = off
            img[y: y + size, x + t] = off
            img[y: y + size, x + size - 1 - t] = off
    draw_text(img, label, x + size // 2 - 6, y + size // 2 - 7, scale=2, color=(0, 0, 0) if filled else off)


def annotate(frame: np.ndarray, t: float, down: "list[bool]", labels: "list[str]", caption: str) -> np.ndarray:
    """One review frame: the time, the caption and a box per foot (L, R), filled while that foot is on the floor."""
    img = frame.copy()
    draw_text(img, f"T={t:5.2f} S", 10, 10, scale=3)
    draw_text(img, caption, 10, img.shape[0] - 26, scale=2)
    size, gap = 46, 12
    x = img.shape[1] - len(labels) * (size + gap) - 4
    for lab, d in zip(labels, down):
        draw_box(img, x, 8, size, bool(d), lab)
        x += size + gap
    return img


class Mp4Writer:
    """Stream H x W x 3 uint8 frames into an H.264 mp4 at *fps* through ffmpeg's stdin (nothing held in memory)."""

    def __init__(self, path: Path, fps: float, width: int, height: int, ffmpeg: str = "ffmpeg"):
        self.path, self.frames = Path(path), 0
        cmd = [ffmpeg, "-hide_banner", "-loglevel", "error", "-y", "-f", "rawvideo", "-pix_fmt", "rgb24",
               "-s", f"{width}x{height}", "-r", f"{fps:.6g}", "-i", "-", "-an", "-c:v", "libx264", "-preset",
               "veryfast", "-crf", "20", "-pix_fmt", "yuv420p", str(self.path)]
        self.proc = subprocess.Popen(cmd, stdin=subprocess.PIPE)

    def write(self, frame: np.ndarray) -> None:
        assert self.proc.stdin is not None
        self.proc.stdin.write(np.ascontiguousarray(frame, dtype=np.uint8).tobytes())
        self.frames += 1

    def close(self) -> None:
        assert self.proc.stdin is not None
        self.proc.stdin.close()
        if self.proc.wait() != 0:
            raise RuntimeError(f"ffmpeg exited {self.proc.returncode} writing {self.path}")

    def abort(self) -> None:
        try:
            if self.proc.stdin is not None:
                self.proc.stdin.close()
        finally:
            self.proc.kill()
            self.proc.wait()
            self.path.unlink(missing_ok=True)


def encode_mp4(frames: "list[np.ndarray]", fps: float, path: Path, ffmpeg: str = "ffmpeg") -> None:
    """Write *frames* (equal shapes) to an mp4 at *fps*."""
    if not frames:
        raise ValueError("no frames")
    h, w = frames[0].shape[:2]
    writer = Mp4Writer(path, fps, w, h, ffmpeg)
    for f in frames:
        writer.write(f)
    writer.close()


def floor_down_from_substeps(substep_floor_newtons: "list[np.ndarray]", threshold: float) -> np.ndarray:
    """A foot is on the floor for a control step when its floor normal force exceeds *threshold* on at least half of
    the step's substeps (the probe's floor channel: floor_contact_substep_frac >= 0.5)."""
    fl = np.asarray(substep_floor_newtons, dtype=float)
    return (fl > threshold).mean(axis=0) >= 0.5


def compare_rows(rerolled: dict, recorded: dict) -> list[str]:
    """The keys whose values differ between a re-rolled row (already JSON-safe and round-tripped) and probe.json's."""
    keys = (set(rerolled) | set(recorded)) - set(SKIP_KEYS)
    return sorted(k for k in keys if rerolled.get(k, "<absent>") != recorded.get(k, "<absent>"))


def sha256_file(path: "str | Path") -> str:
    return "sha256:" + hashlib.sha256(Path(path).read_bytes()).hexdigest()


def episodes_from_score(score: dict, seed: int, steps: int) -> dict:
    """{episode: label} from pilot3_score.py's JSON for this seed at this cell's steps."""
    req = (score.get("video_review_required") or {}).get(str(seed))
    if req is None:
        raise SystemExit(f"the score JSON lists no video review for s{seed}")
    if int(req["steps"]) != int(steps):
        raise SystemExit(f"the score JSON reads s{seed} at {req['steps']}, not this cell's {steps}")
    return {int(e["episode"]): str(e.get("label", "")) for e in req["episodes"]}


def install_probe_shims(embodiment: str = EMBODIMENT) -> None:
    """gait_probe_pilot.py's four shims (bare species string -> the model id), so the frozen probe's functions run."""
    from environments.shared import (
        config,
        plant_contract,
        species_registry,
        stage_manifest,
    )
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


def load_probe(tree: Path):
    path = tree / "docs" / "investigations" / "gait_2026_09" / "gait_probe.py"
    spec = importlib.util.spec_from_file_location("gait_probe_frozen", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module, path


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("cell")
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--from-score", default=None)
    src.add_argument("--episodes", default=None, help="comma-separated panel episode indices (0-39)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--standalone", action="store_true", help="roll only the named episodes, not the panel prefix")
    ap.add_argument("--ffmpeg", default=shutil.which("ffmpeg") or "ffmpeg")
    a = ap.parse_args(argv)

    os.environ.setdefault("MESA_SHADER_CACHE_DISABLE", "true")
    cell = Path(a.cell).resolve()
    binding = json.loads((cell / "binding.json").read_text())
    doc = json.loads((cell / "probe.json").read_text())
    meta, rows = doc["meta"], doc["episodes"]
    seed, steps = int(binding["seed"]), int(binding["steps"])
    if a.from_score:
        wanted = episodes_from_score(json.loads(Path(a.from_score).read_text()), seed, steps)
    else:
        wanted = {int(x): "" for x in a.episodes.split(",") if x.strip()}
    if not wanted:
        print("nothing to render: no episode named")
        return 0
    if any(not 0 <= e < len(rows) for e in wanted):
        raise SystemExit(f"episodes {sorted(wanted)} are not all in the panel of {len(rows)}")

    tree = Path.cwd().resolve()
    import environments

    if not str(Path(environments.__file__).resolve()).startswith(str(tree) + os.sep):
        raise SystemExit(f"environments imported from {environments.__file__}, not from the tree {tree}")
    head = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=tree, text=True,
                                   env=dict(os.environ, GIT_OPTIONAL_LOCKS="0")).strip()
    problems = []
    if binding.get("head") != head:
        problems.append(f"the cell was judged at {binding.get('head')}, this tree is {head}")
    ckpt, vec = Path(binding["checkpoint"]), Path(binding["vecnorm"])
    for p, want in ((ckpt, binding.get("checkpoint_sha256")), (vec, binding.get("vecnorm_sha256"))):
        if not p.is_file() or sha256_file(p) != want:
            problems.append(f"{p} is missing or no longer the judged file ({want})")
    if meta.get("seed_scheme") != "panel":
        problems.append(f"probe.json's seed scheme is {meta.get('seed_scheme')!r}, not panel")
    if problems:
        print("refusing (nothing written): " + "; ".join(problems), flush=True)
        return 2

    install_probe_shims()
    probe, probe_path = load_probe(tree)
    import gymnasium as gym
    import mujoco
    from environments.shared.config import load_stage_config
    from environments.shared.curriculum.stance_gate import StanceGateThresholds
    from environments.shared.plant_contract import (
        current_plant_identity,
        validate_environment_plant,
    )
    from environments.shared.policy_loading import load_sb3_checkpoint
    from environments.shared.species_registry import SPECIES_FACTORIES

    stage_cfg = load_stage_config(SPECIES, STAGE)
    env_kwargs = dict(stage_cfg["env_kwargs"])
    gate_settle = int(StanceGateThresholds.from_curriculum(stage_cfg.get("curriculum_kwargs", {}),
                                                           require_criteria=False).settle_steps)
    settle = int(meta["settle_steps_gait_window"])
    same_kwargs = json.loads(json.dumps(probe._json_safe(env_kwargs))) == meta.get("env_kwargs")
    if not same_kwargs or gate_settle != meta.get("settle_steps_gate"):
        print("refusing (nothing written): this tree's locomotion env kwargs or gate settle differ from probe.json's")
        return 2
    env_class = SPECIES_FACTORIES[SPECIES]().env_class
    plant = current_plant_identity(SPECIES)
    model, normalizer, vec_path = load_sb3_checkpoint(str(ckpt), str(vec), lambda: env_class(**env_kwargs),
                                                      guess_sidecar=False, allow_unnormalized=False,
                                                      plant_identity=plant, allow_legacy_plant=False)
    obs_rms = normalizer.obs_rms
    mean = np.asarray(obs_rms.mean, dtype=np.float64)
    var = np.asarray(obs_rms.var, dtype=np.float64)
    eps, clip = float(normalizer.epsilon), float(normalizer.clip_obs)

    def predict(obs: np.ndarray) -> np.ndarray:  # the probe's own predict, verbatim in effect
        normalized = np.clip((obs - mean) / np.sqrt(var + eps), -clip, clip).astype(np.float32)
        return np.asarray(model.predict(normalized, deterministic=True)[0])

    base = env_class(**env_kwargs)
    validate_environment_plant(base, plant, artifact=f"{SPECIES} {STAGE} pilot3_video environment")
    renderer = mujoco.Renderer(base.unwrapped.model, height=HEIGHT, width=WIDTH)
    camera = base.unwrapped._make_camera()

    class Capture(gym.Wrapper):
        """Pass-through, but while `on`: after each control step, render the env's camera and stream the annotated
        frame to the real-time and quarter-speed writers. The boxes need each step's floor channel before the probe
        computes it, so a hook chained after the probe's own substep hook sums each foot's floor normal force as the
        probe's hook does (read only); after the episode the result must equal the probe's
        floor_contact_substep_frac >= 0.5 at every step, else the videos are deleted."""

        def __init__(self, env):
            super().__init__(env)
            self.on, self.writers, self.down, self.caption = False, None, [], ""

        def reset(self, **kw):
            self.down = []
            return self.env.reset(**kw)

        def step(self, action):
            if not self.on:
                return self.env.step(action)
            u = self.env.unwrapped
            m, d = u.model, u.data
            probe_hook = u._substep_probe_hook
            substeps: list = []
            f6 = np.zeros(6)

            def chained() -> None:
                if probe_hook is not None:
                    probe_hook()
                floor_n = np.zeros(rd.n_feet)
                for ci in range(d.ncon):
                    c = d.contact[ci]
                    g1, g2 = int(c.geom1), int(c.geom2)
                    f1, f2 = rd.geom_foot[g1], rd.geom_foot[g2]
                    if (f1 < 0 and f2 < 0) or not (rd.is_floor[g1] or rd.is_floor[g2]):
                        continue
                    mujoco.mj_contactForce(m, d, ci, f6)
                    floor_n[f1 if f1 >= 0 else f2] += float(f6[0])
                substeps.append(floor_n)

            u._substep_probe_hook = chained
            try:
                out = self.env.step(action)
            finally:
                u._substep_probe_hook = probe_hook
            down = floor_down_from_substeps(substeps, probe.REPO_CONTACT_THRESHOLD_N)
            self.down.append(down)
            renderer.update_scene(d, camera)
            frame = annotate(renderer.render(), len(self.down) * dt, [bool(down[f]) for f in box_order],
                             [labels[f] for f in box_order], self.caption)
            for kind, w in self.writers.items():
                w.write(annotate_speed(frame, kind))
            return out

    env = Capture(base)
    rd = probe.RobotDescription(base)
    dt = float(base.unwrapped.dt)
    labels = list(rd.display)
    box_order = sorted(range(len(labels)), key=lambda f: ({"L": 0, "R": 1}.get(labels[f], 2), f))
    out_dir = Path(a.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    order = sorted(wanted) if a.standalone else list(range(max(wanted) + 1))
    seed_base = int(meta["seed_base"])
    record = {"schema": "walking-pilot3-video/v1", "cell": str(cell), "tree": str(tree), "head": head,
              "probe": str(probe_path), "probe_sha256": sha256_file(probe_path), "tool_sha256": sha256_file(__file__),
              "checkpoint_sha256": binding["checkpoint_sha256"], "standalone": a.standalone, "dt": dt,
              "fps_realtime": 1.0 / dt, "fps_quarter": 1.0 / dt / QUARTER, "camera": {
                  "distance": float(camera.distance), "azimuth": float(camera.azimuth),
                  "elevation": float(camera.elevation), "type": int(camera.type)},
              "rolled": [], "episodes": {}}
    status = 0
    try:
        for i in order:
            env.on = i in wanted
            stem = f"s{seed}_{steps}_ep{i:02d}_reset{seed_base + i}"
            if env.on:
                env.caption = f"S{seed} {steps} EP {i:02d} RESET {seed_base + i} {wanted[i]}".strip()
                env.writers = {kind: Mp4Writer(out_dir / f"{stem}_{kind}.mp4", fps, WIDTH, HEIGHT, a.ffmpeg)
                               for kind, fps in (("realtime", 1.0 / dt), ("quarter", 1.0 / dt / QUARTER))}
            try:
                tr = probe.rollout_episode(env, predict, rd, seed_base + i)
            except BaseException:
                for w in (env.writers or {}).values():
                    w.abort()
                raise
            row = probe.analyze_episode(tr, rd, dt, settle, gate_settle)
            rerolled = json.loads(json.dumps(probe._json_safe(row)))
            diff = compare_rows(rerolled, rows[i])
            record["rolled"].append({"episode": i, "matches_probe": not diff, "differing_keys": diff[:20]})
            print(f"  ep {i:2d} (reset {seed_base + i}): {'matches probe.json' if not diff else f'DIFFERS in {len(diff)} keys {diff[:6]}'}"
                  + ("  [render]" if i in wanted else ""), flush=True)
            if not env.on:
                continue
            probe_down = np.asarray(tr["floor_contact_substep_frac"]) >= 0.5
            boxes_ok = np.array_equal(np.asarray(env.down, dtype=bool), probe_down)
            ent = {"reset_seed": seed_base + i, "label": wanted[i], "matches_probe": not diff, "differing_keys": diff,
                   "boxes_equal_probe_floor_channel": boxes_ok, "length": int(tr["length"]), "files": {}}
            if diff or not boxes_ok:
                status = 1
                for w in env.writers.values():
                    w.abort()
                ent["note"] = ("no video: the re-rolled episode is not the probe's" if diff
                               else "no video: the boxes' floor channel differs from the probe's")
            else:
                for kind, w in env.writers.items():
                    w.close()
                    ent["files"][kind] = {"path": str(w.path), "sha256": sha256_file(w.path), "frames": w.frames,
                                          "fps": 1.0 / dt if kind == "realtime" else 1.0 / dt / QUARTER}
                ent["floor_duty_window"] = {lab: float(probe_down[settle:, f].mean()) for f, lab in enumerate(labels)}
            env.writers = None
            record["episodes"][str(i)] = ent
    finally:
        renderer.close()
        env.close()
    record["all_rolled_match"] = all(r["matches_probe"] for r in record["rolled"])
    (out_dir / "videos.json").write_text(json.dumps(record, indent=1))
    print(f"wrote {out_dir / 'videos.json'}; named episodes matching probe.json: "
          f"{sum(e['matches_probe'] for e in record['episodes'].values())}/{len(record['episodes'])}", flush=True)
    return status


def annotate_speed(frame: np.ndarray, kind: str) -> np.ndarray:
    img = frame.copy()
    draw_text(img, "X1" if kind == "realtime" else "X0.25", img.shape[1] - 80, img.shape[0] - 26, scale=2)
    return img


if __name__ == "__main__":
    sys.exit(main())
