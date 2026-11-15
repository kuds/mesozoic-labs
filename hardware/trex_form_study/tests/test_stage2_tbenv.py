"""Track B's process environment (``designlab/stage2/tbenv.py``): the GL backend of a process and of its children,
one BLAS thread, the hash seed, and the repository root on ``sys.path``.

Every case that depends on what a process imported, or was started with, runs in a child process.  Nothing here needs
numpy, torch, arrays or models; the one case that imports MuJoCo with OSMesa skips where that backend does not load.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest
from designlab.stage2 import tbenv

STUDY = Path(__file__).resolve().parents[1]


def _child(code: str, env: dict) -> subprocess.CompletedProcess:
    return subprocess.run([sys.executable, "-c", code], cwd=STUDY, env=env, capture_output=True, text=True,
                          timeout=180)


def _last_json(done: subprocess.CompletedProcess):
    assert done.returncode == 0, done.stderr[-2000:]
    return json.loads(done.stdout.strip().splitlines()[-1])


# ------------------------------------------------------------------------------------------------
# child_env
# ------------------------------------------------------------------------------------------------
def test_a_clip_child_of_a_disable_process_gets_osmesa_twice(monkeypatch):
    # the parent's own environment holds what designlab/__init__.py leaves in a "disable" process
    monkeypatch.setenv("MUJOCO_GL", "disable")
    monkeypatch.setenv("PYOPENGL_PLATFORM", "disable")
    env = tbenv.child_env("osmesa")
    assert env["MUJOCO_GL"] == "osmesa" and env["PYOPENGL_PLATFORM"] == "osmesa"
    assert os.environ["MUJOCO_GL"] == "disable" and os.environ["PYOPENGL_PLATFORM"] == "disable"  # a copy
    # ... and a child started with it sees both (MuJoCo's OSMesa backend raises on any other PYOPENGL_PLATFORM)
    got = _last_json(_child("import json, os; import designlab; "
                            "print(json.dumps([os.environ['MUJOCO_GL'], os.environ['PYOPENGL_PLATFORM']]))", env))
    assert got == ["osmesa", "osmesa"]


def test_child_env_disable_removes_the_gl_platform():
    base = {"MUJOCO_GL": "osmesa", "PYOPENGL_PLATFORM": "osmesa", "PATH": "/usr/bin", "HOME": "/x"}
    env = tbenv.child_env("disable", base=base)
    assert env["MUJOCO_GL"] == "disable" and "PYOPENGL_PLATFORM" not in env
    assert env["PATH"] == "/usr/bin" and env["HOME"] == "/x"
    assert base == {"MUJOCO_GL": "osmesa", "PYOPENGL_PLATFORM": "osmesa", "PATH": "/usr/bin", "HOME": "/x"}
    assert tbenv.child_env(base=base) == env                                  # "disable" is the default


def test_child_env_sets_one_thread_the_hash_seed_and_the_path():
    base = {"OMP_NUM_THREADS": "4", "MKL_NUM_THREADS": "8", "PYTHONHASHSEED": "random", "PYTHONPATH": "/elsewhere"}
    for gl in tbenv.GL_BACKENDS:
        env = tbenv.child_env(gl, base=base)
        assert [env[v] for v in tbenv.THREAD_VARS] == ["1", "1", "1"]
        assert env["PYTHONHASHSEED"] == "0" and env["PYTHONPATH"] == "."
    assert tbenv.THREAD_VARS == ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS")
    assert all(isinstance(k, str) and isinstance(v, str) for k, v in tbenv.child_env().items())


def test_an_unknown_backend_is_refused():
    with pytest.raises(ValueError, match="egl"):
        tbenv.child_env("egl")
    with pytest.raises(ValueError, match="glfw"):
        tbenv.assign_gl("glfw")


# ------------------------------------------------------------------------------------------------
# assign_gl
# ------------------------------------------------------------------------------------------------
def _started_with(gl: str) -> dict:
    env = tbenv.child_env("disable")
    env["MUJOCO_GL"] = gl
    env["PYOPENGL_PLATFORM"] = gl
    return env


def test_assign_gl_leaves_disable_in_a_process_started_with_osmesa():
    code = ("from designlab.stage2 import tbenv\n"
            "tbenv.assign_gl('disable')\n"
            "import json, os, sys\n"
            "import mujoco\n"                       # after the assignment: MuJoCo reads 'disable' and loads no GL
            "print(json.dumps([os.environ.get('MUJOCO_GL'), os.environ.get('PYOPENGL_PLATFORM'),"
            " 'OpenGL' in sys.modules, hasattr(mujoco, 'GLContext')]))\n")
    assert _last_json(_child(code, _started_with("osmesa"))) == ["disable", None, False, False]


def test_assign_gl_osmesa_sets_both_variables():
    code = ("from designlab.stage2 import tbenv\n"
            "tbenv.assign_gl('osmesa')\n"
            "import json, os\n"
            "print(json.dumps([os.environ.get('MUJOCO_GL'), os.environ.get('PYOPENGL_PLATFORM')]))\n")
    assert _last_json(_child(code, _started_with("disable"))) == ["osmesa", "osmesa"]


def test_assign_gl_raises_when_mujoco_was_imported_first(monkeypatch):
    # a stand-in module: the rule reads sys.modules and the variable, whatever MuJoCo could load on this machine
    monkeypatch.setitem(sys.modules, "mujoco", type(sys)("mujoco"))
    monkeypatch.setenv("MUJOCO_GL", "osmesa")
    monkeypatch.setenv("PYOPENGL_PLATFORM", "osmesa")
    with pytest.raises(RuntimeError, match="already imported with MUJOCO_GL='osmesa'"):
        tbenv.assign_gl("disable")
    assert os.environ["MUJOCO_GL"] == "osmesa" and os.environ["PYOPENGL_PLATFORM"] == "osmesa"   # nothing was changed
    tbenv.assign_gl("osmesa")                                                 # the backend it has: no error
    monkeypatch.delenv("MUJOCO_GL")
    with pytest.raises(RuntimeError, match="already imported"):               # unset: MuJoCo chose its default
        tbenv.assign_gl("disable")
    for same in ("disable", "off", " Disabled ", "0", "false"):               # MuJoCo's own words for "no GL"
        monkeypatch.setenv("MUJOCO_GL", same)
        tbenv.assign_gl("disable")
        assert os.environ["MUJOCO_GL"] == "disable"
    with pytest.raises(RuntimeError, match="needs 'osmesa'"):
        tbenv.assign_gl("osmesa")


def test_assign_gl_raises_after_a_real_mujoco_import_with_osmesa():
    code = ("import json\n"
            "try:\n"
            "    import mujoco\n"
            "except Exception as exc:\n"
            "    print(json.dumps(['no osmesa', repr(exc)])); raise SystemExit(0)\n"
            "from designlab.stage2 import tbenv\n"
            "try:\n"
            "    tbenv.assign_gl('disable')\n"
            "except RuntimeError as exc:\n"
            "    print(json.dumps(['raised', str(exc)]))\n"
            "else:\n"
            "    print(json.dumps(['no error', '']))\n")
    what, text = _last_json(_child(code, _started_with("osmesa")))
    if what == "no osmesa":
        pytest.skip(f"MuJoCo does not import with MUJOCO_GL=osmesa here: {text}")
    assert what == "raised" and "before anything imports mujoco" in text


# ------------------------------------------------------------------------------------------------
# repo_on_path, study_dir, imports
# ------------------------------------------------------------------------------------------------
def test_study_dir_is_this_checkout():
    assert Path(tbenv.study_dir()) == STUDY
    assert (Path(tbenv.study_dir()) / "designlab" / "stage2" / "tbenv.py").is_file()


def test_repo_on_path_appends_once_and_environments_then_imports():
    code = ("import importlib.util, json, os, sys\n"
            "from designlab.stage2 import tbenv\n"
            "before = list(sys.path)\n"
            "found_before = importlib.util.find_spec('environments') is not None\n"
            "root = tbenv.repo_on_path()\n"
            "once = list(sys.path)\n"
            "again = tbenv.repo_on_path()\n"
            "import environments.shared.policy_loading as pl\n"
            "print(json.dumps(dict(root=root, again=again, found_before=found_before, appended=once == before + [root],"
            " first=once[0], twice=list(sys.path) == once, file=os.path.realpath(pl.__file__),"
            " torch='torch' in sys.modules, sb3='stable_baselines3' in sys.modules)))\n")
    got = _last_json(_child(code, tbenv.child_env("disable")))
    assert got["root"] == str(STUDY.parents[1]) == got["again"]
    assert got["found_before"] is False        # with PYTHONPATH=. the repository's package is not importable ...
    assert got["appended"] is True and got["twice"] is True and got["first"] != got["root"]   # ... appended, once
    assert got["file"] == str(STUDY.parents[1] / "environments" / "shared" / "policy_loading.py")
    assert got["torch"] is False and got["sb3"] is False                      # that import stays light


def test_tbenv_imports_with_numpy_blocked():
    code = ("import sys\n"
            "sys.modules['numpy'] = None\n"            # any 'import numpy' now raises ImportError
            "before = set(sys.modules)\n"
            "from designlab.stage2 import tbenv\n"
            "tbenv.assign_gl('disable'); tbenv.child_env('osmesa'); tbenv.repo_on_path(); tbenv.study_dir()\n"
            "new = sorted(m for m in set(sys.modules) - before if not m.startswith('designlab'))\n"
            "import json; print(json.dumps([m for m in new if m.split('.')[0] not in sys.stdlib_module_names]))\n")
    assert _last_json(_child(code, tbenv.child_env("disable"))) == []         # nothing outside the standard library
