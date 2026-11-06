"""Run a module of the worktree as ``python -m`` would, under ``python -I`` (no cwd or PYTHONPATH on sys.path).

Usage (cwd = the worktree): python -I run_module.py <worktree> <module> [args...]
Refuses to run if ``environments`` would be imported from anywhere but the worktree.
"""

from __future__ import annotations

import os
import runpy
import sys

worktree, module, *args = sys.argv[1:]
if os.getcwd() != worktree:
    raise SystemExit(f"run_module: run from {worktree}, not {os.getcwd()}")
sys.path.insert(0, worktree)
import environments  # noqa: E402

if not os.path.realpath(environments.__file__).startswith(worktree + os.sep):
    raise SystemExit(f"run_module: environments imports from {environments.__file__}")
# postfix/fix copy: record the resolved package path in every log (stderr).
print(f"run_module: environments resolved to {os.path.realpath(environments.__file__)}", file=sys.stderr, flush=True)
sys.argv = [module, *args]
runpy.run_module(module, run_name="__main__", alter_sys=True)
