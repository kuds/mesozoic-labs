"""LB step 4: every return statement of every _on_step / _on_rollout_end in the code copy's callback modules (ast; reads
only).  A callback can end training only through an _on_step that returns something other than True."""
import ast
import json
import sys
from pathlib import Path

code = Path(sys.argv[1]) / "environments" / "shared"
mods = sorted(code.glob("*.py")) + sorted((code / "curriculum").glob("*.py"))
out = []
for f in mods:
    tree = ast.parse(f.read_text())
    for cls in [n for n in ast.walk(tree) if isinstance(n, ast.ClassDef)]:
        for fn in [n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == "_on_step"]:
            rets = [ast.unparse(r.value) if r.value is not None else "None" for r in ast.walk(fn)
                    if isinstance(r, ast.Return)]
            non_true = sorted({r for r in rets if r != "True"})
            out.append({"file": str(f.relative_to(code)), "class": cls.name, "line": fn.lineno,
                        "returns_other_than_True": non_true})
print(json.dumps([o for o in out if o["returns_other_than_True"]], indent=1))
print(f"# {len(out)} _on_step methods, {sum(1 for o in out if o['returns_other_than_True'])} can return non-True")
