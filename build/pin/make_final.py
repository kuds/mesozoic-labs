"""ECS pin step (build list v2 item 12): final/criteria_ecs_final.txt = v3/criteria_ecs_v3.txt with only its PINS
block filled.  Every value in the block is read here from its file (sha256 recomputed), and checked against the
recorded pin it must equal; any disagreement stops the script before anything is written.

    PY -I -B make_final.py <fill-utc>      (writes final/criteria_ecs_final.txt; refuses if it exists)

Nothing trains or simulates; it reads files and writes one text file.
"""
from __future__ import annotations

import csv
import hashlib
import sys
from pathlib import Path

S2 = "/tmp/claude-0/-home-user-mesozoic-labs/184ee67c-2802-5235-8e52-6dc7daa72cd3/scratchpad"
ECS = Path(S2 + "/r10round/ecs")
sys.path.insert(0, str(ECS / "tools"))
import ecs_common as C  # noqa: E402

V3 = ECS / "v3/criteria_ecs_v3.txt"
V3_SHA = "90a9f9aa6c3d87947817aceb93e23b1bb107214c22e0cdd294e7790056ad6d58"
OUT = ECS / "final/criteria_ecs_final.txt"
LANES = Path(S2 + "/r10round/lean/build")

# The v3 PINS block, exactly (lines 297-301); the block is replaced and nothing else.
OLD_BLOCK = """PINS (all filled before approval; nothing here is pinned yet)
  this text's sha256 (in <ECS>/APPROVED and RUNLOG); v1/table_e_ecs.py and .out; v2/check_v2.py and .out;
  <ECS>/code's tree digest (= 331fe8fc...); the sha256 of run_ecs.py, ecs_common.py, read_ecs.py, score_ecs.py,
  run_waves_ecs.sh and their tests; M1 and M5 as above; the parent records' sha256 from FMAN (BL item 2); V1-V5
  results.
"""

TOOLS = ("ecs_common.py", "run_ecs.py", "read_ecs.py", "score_ecs.py", "run_waves_ecs.sh",
         "tests/test_harness_run_ecs.py", "tests/test_harness_waves.py", "tests/test_read_score_fixtures.py",
         "tests/test_read_score_mutants.py", "tests/test_read_score_reader.py", "tests/test_read_score_scorer.py")


def sha(p) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def need(cond, msg):
    if not cond:
        sys.exit(f"make_final: REFUSED: {msg}")


def sums_file(path: Path) -> dict:
    out = {}
    for line in path.read_text().splitlines():
        parts = line.split(None, 1)
        if len(parts) == 2 and len(parts[0]) == 64:
            out[parts[1].lstrip("*").strip()] = parts[0]
    return out


def main(argv):
    need(len(argv) == 2, "usage: make_final.py <fill-utc>")
    utc = argv[1]
    need(not OUT.exists(), f"{OUT} exists")
    need(sha(V3) == V3_SHA, "v3 is not the text the fixer recorded")
    text = V3.read_text()
    need(text.count(OLD_BLOCK) == 1, "v3's PINS block not found exactly once")

    # cited outputs, against the v1/v2 manifests
    cited = {}
    for d, names in (("v1", ("table_e_ecs.py", "table_e_ecs.out")), ("v2", ("check_v2.py", "check_v2.out"))):
        man = sums_file(ECS / d / "MANIFEST.sha256")
        for n in names:
            got = sha(ECS / d / n)
            need(man.get(n) == got, f"{d}/{n} {got} != its MANIFEST.sha256")
            cited[f"{d}/{n}"] = got

    # the tools, against the fixer's final freeze
    frozen = sums_file(ECS / "build/fixes/logs_r2/tool_sha256_frozen_r2.txt")
    tools = {}
    for t in TOOLS:
        got = sha(ECS / "tools" / t)
        need(frozen.get(t) == got, f"tools/{t} {got} != the freeze {frozen.get(t)}")
        tools[t] = got
    on_disk = sorted(str(p.relative_to(ECS / "tools")) for p in (ECS / "tools").rglob("*") if p.is_file())
    need(on_disk == sorted(TOOLS), f"tools/ holds other files: {sorted(set(on_disk) ^ set(TOOLS))}")

    # M1 / M5, against ecs_common and the lane SHA256SUMS files
    lane_sums = {"laneA": sums_file(LANES / "laneA/SHA256SUMS.txt"), "laneB": sums_file(LANES / "laneB/SHA256SUMS.txt")}
    m = {}
    for path in (C.M1, C.M1_COMMON, C.M5, C.M5_COMMON):
        rel = str(Path(path).relative_to(LANES))
        lane, sub = rel.split("/", 1)
        got = sha(path)
        need(C.PIN_SHA256[path] == got == lane_sums[lane].get(sub), f"{rel}: {got} disagrees with its pins")
        m[rel] = got

    # start pairs and parent records: F2 copy1 recomputed, against FMAN and ecs_common
    fman = {}
    with open(C.FMAN, newline="") as fh:
        for r in csv.DictReader(fh):
            if r["fetch"] == "F2" and r["status"] == "COPY" and r["copy"] == "1":
                fman[(r["run_id"], r["path"])] = r["sha256"]
    f2 = []
    for start, st in C.STARTS.items():
        items = [(st["zip_rel"], st["zip_sha256"]), (st["pkl_rel"], st["pkl_sha256"])]
        items += [(f"{C.STAGE_DIR}/{n}", h) for n, h in st["records"].items()]
        for rel, pin in items:
            got = sha(C.f2_file(start, rel))
            need(got == pin == fman.get((st["run_id"], rel)), f"{start} {rel}: {got} disagrees with FMAN or the pin")
            f2.append((start, rel.split("/")[-1], got))

    # the code copy's measured digest (build/pin/check_copy.log, M1 check-copy exit 0)
    log = (ECS / "build/pin/check_copy.log").read_text()
    need(f'"tree_digest": "{C.TREE_DIGEST}"' in log and '"files": 1406' in log and "exit=0" in log
         and f'"commit": "{C.COMMIT}"' in log, "check_copy.log does not show the pinned digest, 1,406 files, exit 0")

    # V5's two digests, from the V5 log
    v5 = (ECS / "validate/fix2_logs/v5_check.out").read_text()
    hp_a = C.HYPERPARAMETERS_SHA256_A.split(":", 1)[1]
    hp_b = "be2e243b3fef90cdc4809ca9455c51e1fe07bb27fd30678c4e3e463fed9a0a76"
    need(f"sha256:{hp_a}" in v5 and f"sha256:{hp_b}" in v5 and "V5 OVERALL: PASS" in v5, "V5 log")
    for f, verdict in (("validate/fix_logs/v1_check.out", "V1 OVERALL: PASS"),
                       ("validate/fix2_logs/v1_check.out", "V1 OVERALL: PASS"),
                       ("validate/fix2_logs/v2v3_check.out", "V2/V3 OVERALL: PASS"),
                       ("validate/fix2_logs/v2_extra_check.out", "V2 EXTRA OVERALL: PASS"),
                       ("validate/fix2_logs/v4_reader.out", "V4 READER OVERALL: PASS"),
                       ("validate/fix2_logs/v4_scorer.out", "V4 SCORER OVERALL: PASS"),
                       ("validate/fix2_logs/v4_negatives.out", "V4 NEGATIVES OVERALL: PASS"),
                       ("build/fixes/logs_r2/tests_all.log", "167 passed"),
                       ("build/fixes/logs_r2/fix_mutants_harness.out", "26 mutants, 26 killed"),
                       ("build/fixes/logs_r2/fix_mutants_reading.out", "85 mutants, 82 killed, 3 survived"),
                       ("build/pin/v2_recheck.out", "ALL_OK")):
        need(verdict in (ECS / f).read_text(), f"{f} lacks {verdict!r}")

    w = 42
    L = [f"PINS (filled {utc} (date -u) by the build's pin step, BL item 12; nothing else differs from v3)",
         "  This text's sha256 cannot be held in the text itself: it is in final/APPROVAL_REQUEST.txt and",
         "  <ECS>/SHA256SUMS.txt, and goes into <ECS>/APPROVED and RUNLOG on approval. The text filled:",
         f"  v3/criteria_ecs_v3.txt, sha256 {V3_SHA}.",
         "  Paths are under <ECS> unless another root is named. [measured: each value below recomputed by",
         "  build/pin/make_final.py from its file and checked equal to the pin named beside it]",
         "  Cited outputs (equal to v1/ and v2/ MANIFEST.sha256):"]
    L += [f"    {k:<{w}}{v}" for k, v in cited.items()]
    L += ["  Code copy <ECS>/code (M1 check-copy, exit 0, 2026-10-29 01:32 UTC, build/pin/check_copy.log): commit",
          f"    {C.COMMIT}, {C.TREE_FILES:,} files, tree digest (= COPY's seal)",
          f"    {C.TREE_DIGEST}",
          "  ECS tools (equal to build/fixes/logs_r2/tool_sha256_frozen_r2.txt; listed in <ECS>/SHA256SUMS.txt):"]
    L += [f"    {'tools/' + k:<{w}}{v}" for k, v in tools.items()]
    L += ["  M1 and M5 (root S2/r10round/lean/build; equal to the lane SHA256SUMS.txt and ecs_common's pins):"]
    L += [f"    {k:<{w}}{v}" for k, v in m.items()]
    L += ["  Start pairs and parent records (F2 01_stance/ .copy1 files; equal to FMAN and ecs_common;",
          "  build/pin/f2_recheck.out, 01:32 UTC):"]
    L += [f"    {s + ' ' + n:<{w}}{v}" for s, n, v in f2]
    L += ["  V1-V5 (D2 (a): one process at a time, OMP 1, nice 15, cwd /tmp; no optimizer step; on the tools above):",
          "    V1 PASS. validate/fix_logs/v1_check.out (4 load-and-save runs, 2026-10-28 23:45-23:46 UTC, outputs",
          "       validate/fix_v1, fix_v1_m1; run_ecs.py and ecs_common.py as pinned), checked again on those",
          "       outputs in validate/fix2_logs/v1_check.out (2026-10-29 00:45 UTC).",
          "    V2 PASS. validate/fix2_logs/v2v3_check.out, v2_extra_check.out (4 no-update runs to S + 262,144 in",
          "       validate/fix2_v2, 00:45-01:26 UTC, driver log fix2_logs/v2_driver.log; train() reached 31 times",
          "       each, parameter and Adam digests unchanged; re-read: build/pin/v2_recheck.out).",
          "    V3 PASS. validate/fix2_logs/v2v3_check.out (M5's outputs in validate/fix2_v3_m5).",
          "    V4 PASS. Tests 167 passed (build/fixes/logs_r2/tests_all.log; the runner's 20 passed 3 more times,",
          "       tests_waves_x3.log); mutants killed 26 of 26 and 82 of 85, the 3 survivors equivalent",
          "       (fix_mutants_harness.out, fix_mutants_reading.out); ruff and bash -n (logs_r2/lint.log; again",
          "       build/pin/lint.log, 01:33 UTC); in validate/fix2_logs: v4_reader.out, v4_scorer.out,",
          "       v4_check_<name>.log, v4_read_<name>.log, v4_r3_trace.log, v4_negatives.out and v4_waves.out",
          "       (the runner's stub dry run, scenarios A-G).",
          "    V5 PASS. validate/fix2_logs/v5_check.out (8 stage_config.json files); hyperparameters_sha256",
          f"       A {hp_a}",
          f"       B {hp_b}",
          "  Superseded, not results: validate/v1, v1_m1, v2, v3_m5, v4*, fix_v2 (earlier tools; build/FIXES.txt)."]
    block = "\n".join(L) + "\n"
    long = [x for x in L if len(x) > 120 or not x.isascii()]
    need(not long, f"lines over 120 columns or not ASCII: {long}")
    new = text.replace(OLD_BLOCK, block, 1)
    OUT.write_text(new)
    print(f"wrote {OUT}: {len(new.splitlines())} lines, sha256 {sha(OUT)}; PINS block {len(L)} lines")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
