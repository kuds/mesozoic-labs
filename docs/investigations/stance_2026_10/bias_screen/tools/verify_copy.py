"""Verify a code copy's seal with M1's own r10m_common.check_copy (loaded only at its pinned sha256)."""
import hashlib
import importlib.util
import json
import sys

M1_COMMON = ("/tmp/claude-0/-home-user-mesozoic-labs/184ee67c-2802-5235-8e52-6dc7daa72cd3/scratchpad/r10round/lean/"
             "build/laneA/tools/r10m_common.py")
PIN = "97431323a84a45026671d681dbd0dc5fe3914aa4068ab20a49e428620b26f5bf"
want = "331fe8fc3f6bdf73acc0d3cc324dd753d9d9d695d7ce914e4e011d5c6437ebf1"
h = hashlib.sha256(open(M1_COMMON, "rb").read()).hexdigest()
assert h == PIN, h
spec = importlib.util.spec_from_file_location("rc", M1_COMMON)
rc = importlib.util.module_from_spec(spec)
spec.loader.exec_module(rc)
out = {}
for path in sys.argv[1:]:
    seal = rc.check_copy(path, expect_commit="8737cd3d86431de1bdd0fcdb49a0d956a45e084d")
    digest, n = rc.tree_digest(path)
    out[path] = {"seal_tree_digest": seal["tree_digest"], "recomputed": digest, "files": n,
                 "equal_to_pinned": digest == want == seal["tree_digest"]}
print(json.dumps(out, indent=1))
sys.exit(0 if all(v["equal_to_pinned"] for v in out.values()) else 1)
