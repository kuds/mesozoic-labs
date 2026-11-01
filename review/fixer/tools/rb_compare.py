"""D2(a)/A6: compare rollbuf.py dumps of HEAD (switch OFF) with ROUND V0 (h)'s pinned digests
(S2/r10round/runbranch/build/equiv/rollout_digest_pins.txt = ROUND l.780-793).  usage: PY -I -B rb_compare.py RBDIR"""
import hashlib, pickle, sys
from pathlib import Path
PINS = {60: ("3aab2bf74d8accd1b18374a5a1209a515da46b8253a0f9ff40c03c2a209c5428", "aeb166271485063f4b7f6d25328423019cfafe9b10ea930e133f69a3e464497f", "b25724e30cfe947ef7b1ca5710554a155d64183211e4155a7d3955b8570e2f41"),
        42: ("19b0024fb0f309139355597efc8773cd24971cf1692064ca855e1c385f8949ff", "688296d5d9f4ac1205d83cae3205299f1caec90dd2d127b7fd75512361e0d3c2", "b8870621262d31125c2ed84ee377d956a45c18fa3aa3a4995dbf4686882f2740"),
        80: ("8b675bb02c5fc7891242e54890364b1b022e56b0b44b8da5f9a1f1348f05698e", "d4b1dd4905b20a86255df46f35175c276726378e422f47b18e74916a98b44730", "3b79f2906de24f3815982b7245b253c705f88c76872eb7d5332841eef4674fde")}
R17, R33 = "08e778c79b9f8857f479647aec224321ba4918d114231b4a33b5e2657dc52a17", "94f7eb64eb52a9c83a9ddc4aa2c1d726784ee4043386f160e8050e41351645e9"
d = Path(sys.argv[1]); ok = True
def sorted_key(buf):
    return hashlib.sha256(b"".join(buf[k] for k in sorted(buf))).hexdigest()
for seed, (fo, sk, ip) in PINS.items():
    p = d / f"head_{seed}.pkl"
    if not p.is_file():
        print(f"seed {seed}: no dump"); ok = False; continue
    r = pickle.load(open(p, "rb"))
    got = (r["rollout_digests"][0], sorted_key(r["buffer"]), r["params_initial"])
    eq = [g == w for g, w in zip(got, (fo, sk, ip))]
    ok &= all(eq) and r["n_updates"] == 0
    print(f"seed {seed} first rollout: fixed-order {got[0][:16]} {'==' if eq[0] else '!='} pin; sorted-key {got[1][:16]} "
          f"{'==' if eq[1] else '!='} pin; init params {got[2][:16]} {'==' if eq[2] else '!='} pin; updates {r['n_updates']}; "
          f"class {r['class']}")
p = d / "head_42_33.pkl"
if p.is_file():
    r = pickle.load(open(p, "rb")); dg = r["rollout_digests"]
    e17, e33 = dg[16] == R17, dg[32] == R33
    ok &= e17 and e33 and r["n_updates"] == 0 and dg[0] == PINS[42][0]
    print(f"seed 42 x{len(dg)}: rollout 1 {dg[0][:16]} {'==' if dg[0] == PINS[42][0] else '!='} pin; rollout 17 {dg[16][:16]} "
          f"{'==' if e17 else '!='} pin; rollout 33 {dg[32][:16]} {'==' if e33 else '!='} pin; updates {r['n_updates']}; "
          f"steps {r['num_timesteps']}")
else:
    print("seed 42 x33: no dump yet"); ok = False
print("RESULT", "ALL EQUAL TO ROUND V0 (h)" if ok else "NOT ALL EQUAL (or incomplete)")
