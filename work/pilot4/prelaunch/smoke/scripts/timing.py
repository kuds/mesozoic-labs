"""The smoke's CPU steps per second and wall times (smoke lane; reads only).

    python -I timing.py EVIDENCE_LAUNCH.json EVIDENCE_RESUME.json NB_DIR OUT.json

learn() spans are read from TensorBoard: an event file's name holds the epoch second its writer was created (SB3
creates it at learn()'s start), and the last train/ scalar's wall_time ends the last rollout's update. Steps per
second = the steps trained in that span / its seconds; it includes the in-training evaluations (10 episodes of 1,000
steps at each eval_freq), as SB3's own time/fps does (printed beside it). Wall times: the runner's start/end files
and the per-cell seconds of its record.
"""

from __future__ import annotations

import json
import re
import sys
from datetime import datetime, timezone
from pathlib import Path


def iso(t: float) -> str:
    return datetime.fromtimestamp(t, tz=timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%fZ")[:-4] + "Z"


def span(ev: dict, tb_file_index: int, steps_from: int, steps_to: int) -> dict:
    files = ev["tensorboard"]["files"]
    start = int(re.search(r"tfevents\.(\d+)\.", files[tb_file_index]).group(1))
    row = next(iter(ev["tensorboard"]["per_dir"].values()))
    ends = [w for s, _, w in row["train/std"] if s == steps_to]
    fps = [v for s, v, _ in row["time/fps"] if s == steps_to]
    end = ends[-1]
    secs = end - start
    return {"tb_writer_created": iso(start), "last_update_at_steps": steps_to, "last_update_wall": iso(end),
            "seconds": round(secs, 1), "steps": steps_to - steps_from,
            "steps_per_second": round((steps_to - steps_from) / secs, 1), "sb3_time_fps_at_end": fps[-1] if fps else None}


def main() -> int:
    ev_l, ev_r = (json.loads(Path(p).read_text()) for p in sys.argv[1:3])
    nb, out = Path(sys.argv[3]), Path(sys.argv[4])
    rec_l = json.loads((nb / "launch_s42" / "record.json").read_text())
    rec_r = json.loads((nb / "resume" / "record.json").read_text())
    t = {
        "launch_learn": span(ev_l, 0, 0, 65_536),
        "resume_learn": span(ev_r, 1, 32_768, 65_536),
        "launch_runner_wall": [(nb / "launch_s42.start").read_text().strip(), (nb / "launch_s42.end").read_text().strip()],
        "resume_runner_wall": [(nb / "resume.start").read_text().strip(), (nb / "resume.end").read_text().strip()],
        "launch_cells_seconds": {c["cell"]: c["seconds"] for c in rec_l["cells"]},
        "resume_cells_seconds": {c["cell"]: c["seconds"] for c in rec_r["cells"]},
        "stage_summary_duration_launch": ev_l.get("stage_summary.txt"),
        "resume_run_duration_seconds": ev_r["stage_config_other"]["run"].get("duration_seconds"),
        "launch_run_duration_seconds": ev_l["stage_config_other"]["run"].get("duration_seconds"),
    }
    for k in ("launch_runner_wall", "resume_runner_wall"):
        a, b = (datetime.fromisoformat(x.replace("Z", "+00:00")[:26] + "+00:00") for x in t[k])
        t[k].append(round((b - a).total_seconds(), 1))
    out.write_text(json.dumps(t, indent=1))
    print(json.dumps(t, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
