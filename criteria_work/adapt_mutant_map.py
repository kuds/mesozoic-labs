"""Adapter lane: which kill checks catch each mutant of tools/score_bs.py (report for ADAPT_NOTES; read only except
the kill roots it builds under the folder given as argv[1], inside <PS>/criteria_work/adapt_tmp)."""
import importlib.util
import sys
from pathlib import Path

sys.dont_write_bytecode = True
T = Path("/home/user/work/phase_screen/tools/tests/test_score_bs.py")
spec = importlib.util.spec_from_file_location("test_score_bs_map", str(T))
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)
base = Path(sys.argv[1])
assert str(base.resolve()).startswith("/home/user/work/phase_screen/criteria_work/adapt_tmp/")
base.mkdir(parents=True, exist_ok=False)


class F:
    def mktemp(self, name):
        p = base / name
        p.mkdir()
        return p


roots = mod.kill_roots.__wrapped__(F())
mod.SB.code_seal_items = lambda d: []
print("original:", mod._kill_checks(mod.SB, roots))
for tag, old, new in mod.MUTANTS:
    assert mod.SRC.count(old) == 1, tag
    m = mod._variant(mod.SRC.replace(old, new), tag)
    print(f"{tag:<32} caught by {mod._kill_checks(m, roots)}")
