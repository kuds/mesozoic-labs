"""Track B's process environment (``STAGE2_PLAN.md`` section 4 B; the Track B rules, K9 and N): standard library only.

Three things every Track B process needs before it imports anything heavy, kept in one module so that no entry point
writes its own copy:

* ``assign_gl``: the value of ``MUJOCO_GL`` for this process.  MuJoCo picks its OpenGL backend when it is first
  imported, and ``designlab/__init__.py`` only sets a default (OSMesa or EGL on a machine with no display).  With
  libOSMesa loaded torch segfaults while it builds a stable-baselines3 policy (seen on 11 Nov 2026; ``tests/
  conftest.py``), and Stage 1's trainer never set the variable.  So every Track B entry point that may build a policy
  starts with ``tbenv.assign_gl("disable")`` before any other import, and the one that renders a clip with
  ``tbenv.assign_gl("osmesa")``.  The variable is assigned, never defaulted: an inherited value loses.
* ``child_env``: the whole environment of a child process (the backend, one BLAS thread, a fixed hash seed, the study
  directory on the path).  A child of a ``disable`` process that must render gets ``PYOPENGL_PLATFORM=osmesa`` with
  it: ``designlab/__init__.py`` defaults that variable from ``MUJOCO_GL``, and MuJoCo's OSMesa backend raises unless
  it is unset or ``osmesa``.
* ``repo_on_path``: the repository root at the END of ``sys.path``.  The loader imports the repository's
  ``environments.shared.policy_loading`` and ``environments.shared.curriculum.checkpoints``, and ``environments`` is
  not importable from the study directory with ``PYTHONPATH=.``.  Appended, never put first: nothing of the study may
  be shadowed by a name at the repository's top level.

This module is in no record key and imports nothing outside the standard library (a test imports it with numpy
blocked): the notebook's launcher (``nb.py``, standard library only) builds its children's environments with it.
"""

from __future__ import annotations

import os
import sys

#: The BLAS thread variables every Track B process runs with at "1" (``s2io.THREAD_VARS``: arrays depend on the
#: thread count through the BLAS, and 8 envs in W workers must not start W x cores threads).
THREAD_VARS = ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS")
#: The two backends Track B uses: no OpenGL at all (every process that may import torch) and OSMesa (the clip child).
GL_BACKENDS = ("disable", "osmesa")
#: The values of ``MUJOCO_GL`` that MuJoCo itself reads as "no OpenGL" (``mujoco/rendering/classic/gl_context.py``,
#: after ``lower().strip()``).
GL_DISABLED = ("disable", "disabled", "off", "false", "0")
#: What ``child_env`` sets besides the backend and the thread variables.
CHILD_FIXED = {"PYTHONHASHSEED": "0", "PYTHONPATH": "."}


def _same_backend(value: str | None, backend: str) -> bool:
    """Whether a process whose ``MUJOCO_GL`` was ``value`` when MuJoCo was imported runs ``backend``."""
    v = (value or "").lower().strip()
    return v in GL_DISABLED if backend == "disable" else v == backend


def _check_backend(backend: str) -> None:
    if backend not in GL_BACKENDS:
        raise ValueError(f"MUJOCO_GL backend {backend!r}: Track B uses {' or '.join(GL_BACKENDS)}")


def assign_gl(backend: str) -> None:
    """``os.environ["MUJOCO_GL"] = backend`` (``"disable"`` or ``"osmesa"``), assigned whatever was inherited.

    For ``"disable"`` ``PYOPENGL_PLATFORM`` is removed (nothing may load a GL library in this process, and a child
    must not inherit a platform its parent never used); for ``"osmesa"`` it is set to ``osmesa``.  If MuJoCo is
    already imported and the variable held another backend, the choice was made and cannot be undone: ``RuntimeError``
    (call this before any other import of the entry point)."""
    _check_backend(backend)
    before = os.environ.get("MUJOCO_GL")
    if "mujoco" in sys.modules and not _same_backend(before, backend):
        raise RuntimeError(
            f"mujoco is already imported with MUJOCO_GL={before!r}; this process needs {backend!r}: "
            f"call tbenv.assign_gl({backend!r}) before anything imports mujoco")
    os.environ["MUJOCO_GL"] = backend
    if backend == "disable":
        os.environ.pop("PYOPENGL_PLATFORM", None)
    else:
        os.environ["PYOPENGL_PLATFORM"] = "osmesa"


def child_env(gl: str = "disable", base=None) -> dict:
    """The environment of a Track B child process: a copy of ``base`` (``os.environ`` when None) with ``MUJOCO_GL``
    = ``gl``, ``PYOPENGL_PLATFORM`` = ``osmesa`` for ``"osmesa"`` and removed for ``"disable"``, the three thread
    variables = ``"1"``, ``PYTHONHASHSEED`` = ``"0"`` and ``PYTHONPATH`` = ``"."`` (start the child with the study
    directory as its working directory: ``study_dir()``).  ``base`` is not changed."""
    _check_backend(gl)
    env = {str(k): str(v) for k, v in (os.environ if base is None else base).items()}
    env["MUJOCO_GL"] = gl
    if gl == "disable":
        env.pop("PYOPENGL_PLATFORM", None)
    else:
        env["PYOPENGL_PLATFORM"] = "osmesa"
    for name in THREAD_VARS:
        env[name] = "1"
    env.update(CHILD_FIXED)
    return env


def study_dir() -> str:
    """The study directory (``hardware/trex_form_study``), from this file's location."""
    here = os.path.realpath(__file__)
    return os.path.dirname(os.path.dirname(os.path.dirname(here)))


def repo_on_path() -> str:
    """The repository root (the study directory's parent's parent), appended to ``sys.path`` unless it is there
    already; returned.  Never inserted at position 0."""
    root = os.path.dirname(os.path.dirname(study_dir()))
    if not any(p and os.path.realpath(p) == root for p in sys.path):
        sys.path.append(root)
    return root


__all__ = ["CHILD_FIXED", "GL_BACKENDS", "GL_DISABLED", "THREAD_VARS", "assign_gl", "child_env", "repo_on_path",
           "study_dir"]
