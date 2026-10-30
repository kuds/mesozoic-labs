"""Are the window rows fresh?  Window (S, T] returns of each continuation against the parent's rows at the same
steps, and pairwise between continuations of one start.  Reads only; prints; writes fresh_out.txt here."""
import csv
import io
from pathlib import Path

HERE = Path(__file__).resolve().parent
ECS = Path("/tmp/claude-0/-home-user-mesozoic-labs/184ee67c-2802-5235-8e52-6dc7daa72cd3/scratchpad/r10round/ecs")
F2 = Path("/tmp/claude-0/-home-user-mesozoic-labs/0fdf74a4-2989-56c9-a773-92101004d7de/scratchpad/foot/stance6/fetch/F2")
ST = {"s42": (3_145_728, 4_718_592, "stance6_r10LTSB_s42"), "s80": (4_194_304, 5_767_168, "stance6_r10LTSB_s80")}


def rows(p):
    return list(csv.DictReader(io.StringIO(Path(p).read_bytes().decode("utf-8"), newline="")))


def window(rs, S, T):
    return {(int(float(r["timesteps"])), int(float(r["episode"]))): r["return"] for r in rs
            if S < float(r["timesteps"]) <= T}


L = []
for st, (S, T, rid) in ST.items():
    par = window(rows(F2 / rid / "copies/01_stance/eval_episodes.csv.copy1"), S, T)
    runs = {d.name: window(rows(d / "01_stance/eval_episodes.csv"), S, T)
            for d in sorted((ECS / "runs").iterdir()) if d.name.startswith(st)}
    for nm, w in runs.items():
        same = sum(1 for k, v in w.items() if par.get(k) == v)
        L.append(f"{nm}: window rows {len(w)}; equal return to the parent's row at the same step/episode: {same}")
    names = sorted(runs)
    pairs = 0
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            a, b = runs[names[i]], runs[names[j]]
            same = sum(1 for k in a if b.get(k) == a[k])
            if same:
                L.append(f"  {names[i]} vs {names[j]}: {same} equal returns")
            pairs += 1
    L.append(f"{st}: {pairs} continuation pairs compared (equal-return pairs printed above, if any)")
(HERE / "fresh_out.txt").write_text("\n".join(L) + "\n")
print("\n".join(L))
