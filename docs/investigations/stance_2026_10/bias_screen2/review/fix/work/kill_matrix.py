"""The fixer's re-run of the scorer's mutation matrix: imports <BS2>/tools/tests/test_score_bs.py (its kill roots,
_kill_checks and MUTANTS, unchanged), builds the kill roots under OUT, and prints for the original and each mutant the
kill checks that fail.  Reads the tools; writes only under OUT.  Run: PY -I -B kill_matrix.py <BS2> <OUT>"""
import hashlib
import importlib.util
import sys
from pathlib import Path

BS2, OUT = Path(sys.argv[1]), Path(sys.argv[2])
sys.dont_write_bytecode = True
spec = importlib.util.spec_from_file_location("test_score_bs", str(BS2 / "tools" / "tests" / "test_score_bs.py"))
T = importlib.util.module_from_spec(spec)
spec.loader.exec_module(T)


class _Factory:
    def __init__(self, base):
        self.base = base

    def mktemp(self, name):
        p = self.base / name
        p.mkdir(parents=True, exist_ok=False)
        return p


OUT.mkdir(parents=True, exist_ok=False)
roots = T.kill_roots.__wrapped__(_Factory(OUT)) if hasattr(T.kill_roots, "__wrapped__") else None
if roots is None:                       # pytest fixtures hide the function; rebuild the roots as the fixture does
    import inspect
    src = inspect.getsource(T.kill_roots)
    body = src.split("def kill_roots(tmp_path_factory):", 1)[1]
    ns = dict(vars(T))
    exec("def _kr(tmp_path_factory):" + body, ns)  # noqa: S102 - the test module's own fixture body
    roots = ns["_kr"](_Factory(OUT))
print("score_bs.py sha256", hashlib.sha256((BS2 / "tools" / "score_bs.py").read_bytes()).hexdigest())
print("test_score_bs.py sha256", hashlib.sha256((BS2 / "tools" / "tests" / "test_score_bs.py").read_bytes()).hexdigest())
print("original fails:", T._kill_checks(T.SB, roots))
survivors = []
for tag, old, new in T.MUTANTS:
    assert T.SRC.count(old) == 1, tag
    failed = T._kill_checks(T._variant(T.SRC.replace(old, new), tag), roots)
    print(f"{tag:<32} caught by {failed}")
    if not failed:
        survivors.append(tag)
print(f"{len(T.MUTANTS) - len(survivors)} of {len(T.MUTANTS)} mutants caught; survivors {survivors}")
