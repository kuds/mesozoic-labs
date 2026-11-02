"""Print the next transfer number for one path of this RUN_ID in F2 (manifest rows with a transfer number + 1),
read only from ROUND/fetch/manifest.csv.  usage: python -I -B nextk.py PATH"""
import csv
import sys

RUN = "stance7_r10LTSBpc_s42"
P = sys.argv[1]
n = 0
with open("/home/user/work/pc_study/round/fetch/manifest.csv", newline="", encoding="utf-8") as fh:
    for r in csv.DictReader(fh):
        if r["fetch"] == "F2" and r["run_id"] == RUN and r["path"] == P and (r["transfer"] or "").strip():
            n = max(n, int(r["transfer"]))
print(n + 1)
