"""The notebook's replays of a stage, remade from a COPY of the smoke's run (walking pilot 4 render lane; scratch).

    cd TREE && ISO_WT=TREE withx python -I -B TOOLS/iso.py replay_render.py RUN_COPY/pilot4_walk_s42/03_locomotion OUT

The notebook's final-save cell (sb3_training.ipynb cell 22, the manual node) calls
reporting.generate_stage_artifacts(species_cfg=SPECIES_CFG, stage_config=STAGE_CONFIGS[stage], stage=stage,
algorithm=ALGORITHM, stage_dir=_manual_dir, seed=SEED, ...), whose replay step is
reporting.stage_artifacts._record_stage_replays(species_cfg, stage_config, stage, algorithm, stage_dir, replays_out,
model_dir=stage_dir / "models", seed, stage_results, allow_legacy_plant=False): the selected-checkpoint replay, then
the final replay, each through evaluation.record_stage_video. This script calls that same _record_stage_replays with
the arguments generate_stage_artifacts passes, built as the notebook builds them (MODEL_ID from SPECIES and
EMBODIMENT, SPECIES_CFG = get_species_config(MODEL_ID), STAGE_CONFIGS = load_all_stages(MODEL_ID) with CRITIC_PHASE
off, the stage from the manifest's "locomotion" node, ALGORITHM "ppo", SEED 42), with replays_out = OUT instead of the
staging folder the notebook publishes to stage_dir/replays. Nothing is trained: no learn(), no optimizer.

Two read-only additions: evaluation.replay_seed is wrapped to record what it returns (the seed record_stage_video passes
to the render env's reset; the value is returned unchanged), and a logging handler keeps every WARNING the writer logs
(it swallows its own failures as "Video recording failed."). OUT/replay_record.json lists the files, their sha256s, the
decoded frame count and per-frame md5s (ffmpeg's framemd5), and the ffprobe stream fields.
"""

from __future__ import annotations

import hashlib
import json
import logging
import subprocess
import sys
from pathlib import Path


def sha(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def frame_md5s(path: Path) -> list[str]:
    out = subprocess.run(["ffmpeg", "-hide_banner", "-loglevel", "error", "-i", str(path), "-f", "framemd5", "pipe:1"],
                         capture_output=True, check=True, text=True).stdout
    return [ln.split(",")[-1].strip() for ln in out.splitlines() if ln and not ln.startswith("#")]


def probe(path: Path) -> dict:
    out = subprocess.run(["ffprobe", "-v", "error", "-select_streams", "v:0", "-count_frames", "-show_entries",
                          "stream=codec_name,width,height,pix_fmt,r_frame_rate,avg_frame_rate,nb_frames,"
                          "nb_read_frames,duration", "-of", "json", str(path)],
                         capture_output=True, check=True, text=True).stdout
    return json.loads(out)["streams"][0]


class Keep(logging.Handler):
    def __init__(self):
        super().__init__(level=logging.WARNING)
        self.records: list[str] = []

    def emit(self, record):
        self.records.append(f"{record.levelname} {record.name}: {record.getMessage()}"
                            + (" [with traceback]" if record.exc_info else ""))


def main() -> int:
    stage_dir, out_dir = Path(sys.argv[1]).resolve(), Path(sys.argv[2]).resolve()
    out_dir.mkdir(parents=True, exist_ok=False)
    logging.basicConfig(level=logging.INFO, stream=sys.stderr, format="%(levelname)s %(name)s: %(message)s")
    keep = Keep()
    logging.getLogger().addHandler(keep)

    import environments
    from environments.shared import evaluation
    from environments.shared.config import load_all_stages
    from environments.shared.reporting import stage_artifacts
    from environments.shared.species_names import require_model_id
    from environments.shared.species_registry import get_species_config
    from environments.shared.stage_manifest import load_stage_manifest, stage_dirname, stage_label

    model_id = require_model_id("tyrannosaurus_rex", "anatomical")  # SPECIES, EMBODIMENT (the smoke's params)
    species_cfg = get_species_config(model_id)
    stage_configs = load_all_stages(model_id)
    manifest = load_stage_manifest(model_id)
    stage = manifest.resolve("locomotion").reference  # MANUAL_NODE = "locomotion"
    if stage_dirname(model_id, stage) != stage_dir.name:
        raise SystemExit(f"{stage_dir.name} is not the locomotion node's folder {stage_dirname(model_id, stage)}")

    calls: list = []
    real_replay_seed = evaluation.replay_seed

    def recording_replay_seed(seed, stage_ref):
        value = real_replay_seed(seed, stage_ref)
        calls.append({"seed_arg": seed, "stage_arg": stage_ref, "returned": value})
        return value

    evaluation.replay_seed = recording_replay_seed
    try:
        stage_artifacts._record_stage_replays(
            species_cfg=species_cfg,
            stage_config=stage_configs[stage],
            stage=stage,
            algorithm="ppo",  # ALGORITHM
            stage_dir=stage_dir,
            replays_out=out_dir,
            model_dir=stage_dir / "models",
            seed=42,  # SEED
            stage_results={},
            allow_legacy_plant=False,
        )
    finally:
        evaluation.replay_seed = real_replay_seed

    files = {}
    for p in sorted(out_dir.glob("*.mp4")):
        md5s = frame_md5s(p)
        files[p.name] = {"sha256": sha(p), "bytes": p.stat().st_size, "decoded_frames": len(md5s),
                         "frame_md5s": md5s, "ffprobe": probe(p)}
    rec = {"environments_imported_from": environments.__file__, "stage": stage, "stage_label": stage_label(stage),
           "stage_dir": str(stage_dir), "replay_seed_calls": calls, "replay_seed(42, stage)": real_replay_seed(42, stage),
           "warnings": keep.records, "files": files,
           "inputs_sha256": {n: sha(stage_dir / "models" / n) for n in (
               "stage2_final.zip", "stage2_final_vecnorm.pkl", "robust_best_model.zip", "robust_best_model_vecnorm.pkl")}}
    (out_dir / "replay_record.json").write_text(json.dumps(rec, indent=1, default=str))
    short = {k: v for k, v in rec.items() if k != "files"}
    short["files"] = {k: {kk: vv for kk, vv in v.items() if kk != "frame_md5s"} for k, v in files.items()}
    print(json.dumps(short, indent=1, default=str))
    return 0 if files and not keep.records else 1


if __name__ == "__main__":
    sys.exit(main())
