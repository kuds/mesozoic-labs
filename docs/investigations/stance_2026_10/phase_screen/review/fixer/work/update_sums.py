"""Fixer: SHA256SUMS.txt -- re-hash every existing line whose file changed, then append the fixer's block."""
import hashlib
from pathlib import Path

PS = Path("/home/user/work/phase_screen")
p = PS / "SHA256SUMS.txt"


def sha(path):
    q = Path(path) if str(path).startswith("/") else PS / path
    return hashlib.sha256(q.read_bytes()).hexdigest()


out, changed = [], []
for line in p.read_text().splitlines():
    if not line.strip() or line.startswith("#"):
        out.append(line)
        continue
    h, name = line.split(None, 1)
    name = name.strip()
    now = sha(name)
    if now != h:
        changed.append(f"{name}: {h} -> {now}")
    out.append(f"{now}  {name}")
names = ["review/FIXES.txt", "review/criteria_reviewed_v1.txt", "review/reviewed_sha256.txt",
         "tools/tests/test_bs_hooks.py"]
F = PS / "review" / "fixer"
for sub in ("bin", "work"):
    names += sorted(str(x.relative_to(PS)) for x in (F / sub).iterdir() if x.is_file())
names += sorted(str(x.relative_to(PS)) for x in (F / "orig").rglob("*") if x.is_file())
for log in ("fixer_compare_2.out", "outside_ps_record.txt", "m1_outside_find.out", "m1_outside_find_times.out",
            "m1_outside_find_tmp_root.out", "outside_val_window.out", "tb_buffer_before.txt", "verify_i1_i4.out",
            "verify_d2.out", "dry_runs.out", "mutant_map.out", "prelaunch_score_2.out", "preflight_all_2.out",
            "plan.out", "refill_pins_1.out", "verify_copy.out", "ruff_final.out", "pytest_final_run.out",
            "pytest_final_read.out", "pytest_final_waves.out", "pytest_final_score3.out", "pytest_final_hooks.out",
            "val/driver.log", "val/v1_independent.out", "val/v2_check.out", "val/v3_check.out"):
    names.append(f"review/fixer/logs/{log}")
have = {line.split(None, 1)[1].strip() for line in out if line.strip() and not line.startswith("#")}
add = [n for n in names if n not in have]
out.append("# fixer (2026-10-31): the review's fixes and their evidence (review/FIXES.txt); review/fixer/notes.txt is a "
           "live log, not summed")
out += [f"{sha(n)}  {n}" for n in add]
p.write_text("\n".join(out) + "\n")
print("re-hashed:", *changed, sep="\n  ")
print("added", len(add))
