"""Fixer probe (scratch, kept for the record): why the off _b root reads INCOMPLETE."""
import sys
from pathlib import Path

sys.path.insert(0, "/home/user/work/pc_study/tools/tests")
import test_score_pc as T  # noqa: E402

out, first, b, tb = T._off_b_round(Path(sys.argv[1]), 48)
print("incomplete:", out["incomplete"])
print("provisional:", out.get("provisional"))
