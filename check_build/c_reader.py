"""PS build checker: the reader's on-policy step counters (read_bs.onpolicy_states, PS's file, unmodified) against the
phase each recorded state actually carried.  For this check only, the collection's envs are the phase envs
(bs_hooks.phase_env_class, swapped right after create_vec_env by an in-memory wrapper), so every recorded raw
observation carries the phase of its own counter in slots 60-62; the reader's K must reproduce it exactly.  Writes
nothing but TMPDIR scratch (PS/check_build/tmp/c_reader) and stdout.
    PY -I -B c_reader.py
"""
import importlib.util
import json
import os
import sys
from pathlib import Path

PS = Path("/home/user/work/phase_screen")
spec = importlib.util.spec_from_file_location("read_bs", str(PS / "tools" / "read_bs.py"))
RB = importlib.util.module_from_spec(spec)
sys.modules["read_bs"] = RB
spec.loader.exec_module(RB)
C, H = RB.C, RB.H
assert os.environ["TMPDIR"].startswith(str(PS / "check_build") + "/")
sys.path.insert(0, os.path.realpath(C.CODE))
import numpy as np  # noqa: E402
import torch  # noqa: E402

torch.set_num_threads(1)
import environments.shared.train_base as TB  # noqa: E402

orig = TB.create_vec_env


def swapped(*a, **k):
    v = orig(*a, **k)
    for e in v.venv.envs:
        u = e.unwrapped
        u.__class__ = H.phase_env_class(type(u))
    return v


TB.create_vec_env = swapped
zip_path, pkl_path = C.f2_file(C.START["zip_rel"]), C.f2_file(C.START["pkl_rel"])
raw, ks = RB.onpolicy_states(zip_path, pkl_path, RB.ONPOLICY_SEEDS[0])
got = np.asarray(raw[:, 60:63], dtype=np.float32)
want = H.phase_matrix(ks).astype(np.float32)
out = {"rows": int(len(ks)), "k_min": int(ks.min()), "k_max": int(ks.max()), "k0_rows": int((ks == 0).sum()),
       "presettle": int((ks < 200).sum()), "max_abs_err": float(np.max(np.abs(got.astype(np.float64) - want))),
       "exact": bool(np.array_equal(got, want))}
out["RESULT"] = "PASS" if out["exact"] else "FAIL"
print(json.dumps(out, indent=1))
sys.exit(0 if out["exact"] else 1)
