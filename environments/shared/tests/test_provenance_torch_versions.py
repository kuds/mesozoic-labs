"""provenance.json's torch, CUDA runtime, cuDNN and GPU driver versions (result_bundle.provenance) with torch
installed: reading them changes nothing a run computes.  Needs torch (the SB3 job)."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")

REPOSITORY = Path(__file__).resolve().parents[3]

#: Runs in a fresh interpreter, so torch's state before the call is the state a training process has at the
#: notebook's storage cell: every random generator, the thread count and every determinism flag, before and after.
#: The module is imported before the first snapshot, as a training process has imported it long before that cell:
#: the import reaches stable_baselines3 (through ``result_bundle.audit`` and ``curriculum.sb3_compat``, as on main),
#: and with SB3's extras installed importing SB3 alone advances Python's global random state.  What is tested is
#: the call.
_PROBE = """
import json, random, sys
sys.path.insert(0, {root!r})
import numpy as np
import torch
from environments.shared.result_bundle.provenance import _dependency_versions

def state():
    return {{
        "torch_rng": torch.random.get_rng_state().tolist(),
        "numpy_rng": [np.random.get_state()[0], np.random.get_state()[1].tolist(), *np.random.get_state()[2:]],
        "python_rng": repr(random.getstate()),
        "threads": torch.get_num_threads(),
        "interop_threads": torch.get_num_interop_threads(),
        "deterministic": torch.are_deterministic_algorithms_enabled(),
        "cudnn": [torch.backends.cudnn.enabled, torch.backends.cudnn.benchmark, torch.backends.cudnn.deterministic],
        "cuda_initialized": torch.cuda.is_initialized(),
    }}

before = state()
versions = _dependency_versions()
after = state()
print(json.dumps({{"same": before == after, "versions": versions, "torch": torch.__version__,
                  "cuda": torch.version.cuda}}))
"""


def test_reading_the_versions_changes_no_random_state_thread_count_or_flag():
    completed = subprocess.run(
        [sys.executable, "-I", "-B", "-c", _PROBE.format(root=str(REPOSITORY))],
        capture_output=True,
        text=True,
        timeout=600,
        check=False,
    )
    assert completed.returncode == 0, completed.stderr[-3000:]
    result = json.loads(completed.stdout.strip().splitlines()[-1])
    assert result["same"] is True
    versions = result["versions"]
    # torch's own metadata version, e.g. "2.13.0+cpu" for the CPU wheel's "2.13.0+cpu" __version__.
    assert versions["torch"] == result["torch"] or versions["torch"].split("+")[0] == result["torch"].split("+")[0]
    if result["cuda"] is None:  # a CPU wheel: no CUDA runtime, no cuDNN
        assert versions["cuda_runtime"] is None and versions["cudnn"] is None
    else:
        assert versions["cuda_runtime"] == result["cuda"]
    assert versions["gpu_driver"] is None or isinstance(versions["gpu_driver"], str)
