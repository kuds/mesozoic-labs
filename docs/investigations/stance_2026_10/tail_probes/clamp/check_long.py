"""Are the 2,000-step episodes' first 1,000 steps the 1,000-step episodes, bit for bit?  (No simulation.)

python -B check_long.py > check_long.txt

Long unforced arms against the policies lens's tracks (com_revisit/notes/policies/run/tracks); long forced arms against
this folder's 1,000-step forced arms (run/s42_held0, run/s80_forced1).  Compared: CoM, root quaternion, feet, applied
action (float32 as saved), samples 0..1000 and actions 0..999.
"""

from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
LENS = HERE.parent / "policies" / "run" / "tracks"
PAIRS = {"long/s42_unforced": LENS, "long/s80_unforced": LENS,
         "long/s42_held0": HERE / "run/s42_held0/tracks", "long/s80_forced1": HERE / "run/s80_forced1/tracks"}

total = same = 0
for arm, ref_dir in PAIRS.items():
    for npz in sorted((HERE / arm / "tracks").glob("*.npz")):
        z, ref = np.load(npz), np.load(ref_dir / npz.name)
        rs = [int(s) for s in ref["seeds"]]
        for i, seed in enumerate(int(s) for s in z["seeds"]):
            j = rs.index(seed)
            ok = all(np.array_equal(z[f"{k}_{i}"][:1001], ref[f"{k}_{j}"]) for k in ("com", "quat", "feet")) and \
                np.array_equal(z[f"action_{i}"][:1000], ref[f"action_{j}"]) and len(z[f"t_{i}"]) == 2001
            total += 1
            same += int(ok)
            if not ok:
                print(f"DIFFERS {arm} {npz.stem} seed {seed}")
print(f"{same} of {total} long episodes: first 1,000 steps bit-identical to the 1,000-step episode (and 2,001 samples)")
