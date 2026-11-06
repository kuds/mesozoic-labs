"""Run the pinned nb_pilot_runner4.py unchanged, and copy the run directory once, right after its 32,768 pair is saved.

Smoke lane (pilot 4 pre-launch). Why: the RESUME test must start from the smoke's own 32,768 pair (the only training
permitted is the 65,536-step smoke and its RESUME from the 32,768 pair to 65,536), so no second launch is trained to
32,768 as pilot 3's rtest was. The copy is the run directory exactly as a runtime loss right after that pair would
leave it: it is taken inside SB3's CheckpointCallback._on_step, after the original method saved the pair and before
any later callback of that step runs -- the point where the runner's own --stop-after-pair calls os._exit(75). The
copy only reads the run directory; training continues unchanged (the copy's time is recorded).

    cd CLONE && python -I runner_snapshot.py RUNNER SNAP_DIR STEPS -- RUNNER_ARGS...

RUNNER is loaded from its pinned path (its __file__, so the record's tool_sha256 is the pinned file's) and its main()
is called with RUNNER_ARGS. SNAP_DIR/<run dir name>/ receives the copy; SNAP_DIR/snapshot.json records when, the
callback's num_timesteps, the pair's sha256s and every copied file with its size. With STEPS 0 no copy is made (the
RESUME phase). Exit status: the runner's.
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
import shutil
import sys
import time
from pathlib import Path


def sha256(path: Path) -> str:
    return "sha256:" + hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> int:
    runner_path, snap_dir, steps = sys.argv[1], Path(sys.argv[2]).resolve(), int(sys.argv[3])
    rest = sys.argv[4:]
    if rest[:1] == ["--"]:
        rest = rest[1:]
    spec = importlib.util.spec_from_file_location("nb_pilot_runner4", runner_path)
    runner = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(runner)
    if steps:
        from stable_baselines3.common.callbacks import CheckpointCallback

        original = CheckpointCallback._on_step
        done: dict = {}

        def _on_step(self):
            result = original(self)
            if done or getattr(self, "name_prefix", None) != "stage2":
                return result
            models = Path(self.save_path)
            zip_path = models / f"stage2_{steps}_steps.zip"
            pkl_path = models / f"stage2_vecnormalize_{steps}_steps.pkl"
            if zip_path.is_file() and pkl_path.is_file():
                t0 = time.time()
                run_dir = models.parent.parent
                dest = snap_dir / run_dir.name
                snap_dir.mkdir(parents=True, exist_ok=True)
                shutil.copytree(run_dir, dest, symlinks=True)
                files = sorted(str(p.relative_to(dest)) + f" {p.stat().st_size}" for p in dest.rglob("*") if p.is_file())
                done.update({"steps": steps, "num_timesteps": int(getattr(self, "num_timesteps", -1)),
                             "n_calls": int(getattr(self, "n_calls", -1)), "run_dir": str(run_dir), "copy": str(dest),
                             "at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                             "zip_sha256": sha256(zip_path), "vecnormalize_sha256": sha256(pkl_path),
                             "copy_zip_sha256": sha256(dest / "03_locomotion" / "models" / zip_path.name),
                             "copy_vecnormalize_sha256": sha256(dest / "03_locomotion" / "models" / pkl_path.name),
                             "copy_seconds": round(time.time() - t0, 3), "files": files})
                (snap_dir / "snapshot.json").write_text(json.dumps(done, indent=1))
                print(f"runner_snapshot: copied {run_dir} to {dest} after the {steps:,} pair "
                      f"(num_timesteps {done['num_timesteps']:,}, {done['copy_seconds']} s); training continues",
                      flush=True)
            return result

        CheckpointCallback._on_step = _on_step
    return int(runner.main(rest))


if __name__ == "__main__":
    sys.exit(main())
