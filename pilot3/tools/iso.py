"""Run one Python script against the tree under test and nothing else: walking pilot 3's isolation shim.

Scratch, never merged. Pilot 1's tools/iso.py (sha256 addbfc79...) is lost; this is a rebuild to the contract its
callers state (pilot 1's pilot_judge.py and gait_probe_pilot.py docstrings, kept by pilot3_judge.py):

    cd TREE && ISO_WT=TREE [PYTHONPATH=EXTRA[:EXTRA...]] python3 -I iso.py SCRIPT [ARGS...]

* ISO_WT (required) names the tree: a checkout holding environments/__init__.py. Relative paths resolve against the
  working directory.
* The import path becomes: the tree first, then each PYTHONPATH entry (python -I ignores PYTHONPATH, so this shim
  adds them; without -I they are not repeated), then the interpreter's own entries. The directory holding iso.py and
  the empty entry are dropped, so neither the tools folder nor the working directory shadows the tree.
* PYTHONPATH in os.environ is rewritten to the tree followed by the original entries, so a child process started by
  SCRIPT sees the tree first, and the frozen gait probe's own check (its first PYTHONPATH entry must be the tree it
  imported `environments` from) holds.
* Bytecode writing is turned off (sys.dont_write_bytecode), so nothing is written into the tree.
* SCRIPT runs as __main__ with sys.argv = [SCRIPT, *ARGS]; its exit status is this shim's.
* Refused (exit 2, before SCRIPT runs): ISO_WT unset, or not such a tree; no SCRIPT, or not a file; `environments`
  already imported. After SCRIPT returns, exit 2 when it imported `environments` from anywhere but the tree.
"""

from __future__ import annotations

import os
import runpy
import sys
from pathlib import Path


def _refuse(msg: str) -> int:
    print(f"iso.py: refusing: {msg}", file=sys.stderr, flush=True)
    return 2


def isolated_path(tree: str, extra: list[str], current: list[str], here: str) -> list[str]:
    """The import path: the tree, then the extra entries, then the current entries; iso.py's folder and '' dropped."""
    out: list[str] = []
    for entry in [tree, *extra, *current]:
        if entry in ("", here) or entry in out:
            continue
        out.append(entry)
    return out


def main(argv: list[str]) -> int:
    wt = os.environ.get("ISO_WT", "").strip()
    if not wt:
        return _refuse("ISO_WT is not set (it names the tree under test)")
    tree = Path(wt).resolve()
    if not (tree / "environments" / "__init__.py").is_file():
        return _refuse(f"ISO_WT={wt!r} is not a tree with environments/__init__.py")
    if len(argv) < 2:
        return _refuse("usage: ISO_WT=TREE python3 -I iso.py SCRIPT [ARGS...]")
    script = Path(argv[1])
    if not script.is_file():
        return _refuse(f"no script {argv[1]!r}")
    if "environments" in sys.modules:
        return _refuse("environments is already imported")
    here = str(Path(__file__).resolve().parent)
    extra = [str(Path(p).resolve()) for p in os.environ.get("PYTHONPATH", "").split(os.pathsep) if p.strip()]
    extra = [p for p in extra if p != str(tree)]
    sys.path[:] = isolated_path(str(tree), extra, list(sys.path), here)
    os.environ["PYTHONPATH"] = os.pathsep.join([str(tree), *extra])
    sys.dont_write_bytecode = True
    sys.argv = [str(script), *argv[2:]]
    code: object = 0
    try:
        runpy.run_path(str(script), run_name="__main__")
    except SystemExit as exc:
        code = exc.code
    mod = sys.modules.get("environments")
    mod_file = getattr(mod, "__file__", None) if mod is not None else None
    if mod_file and not str(Path(mod_file).resolve()).startswith(str(tree) + os.sep):
        return _refuse(f"{script} imported environments from {mod_file}, not from {tree}")
    if code is None:
        return 0
    if isinstance(code, int):
        return code
    print(code, file=sys.stderr, flush=True)
    return 1


if __name__ == "__main__":
    sys.exit(main(sys.argv))
