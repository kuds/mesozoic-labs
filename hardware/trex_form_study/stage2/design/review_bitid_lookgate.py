"""Reviewer's scratch check: the V0 set's look-format nominal and 0.40 m/s walks, built as the design says
(look.generate + look.arrays), equal the committed look ladder's V0 rungs byte for byte (a free gate)."""
import hashlib, io, json, sys
import numpy as np
from designlab import sizing
from designlab.stage2 import look
body = "75-M-H-10"
src = look.source(body)
rec = json.load(open("stage2/look/look_refs.json"))["bodies"][body]
for key, spec in (("nominal", sizing.NOMINAL), ("v040", look.speed_spec(src["H"]))):
    g = look.generate(src, spec)
    buf = io.BytesIO(); np.savez(buf, **look.arrays(src, g))
    got = hashlib.sha256(buf.getvalue()).hexdigest()
    print(key, got[:16], "equals the committed ladder's V0 rung:", got == rec[key]["V0"]["sha256"])
