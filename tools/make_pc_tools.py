"""PC study (phase-critic Colab round): build the adapted tool set ST/tools/pc/ from the round's pinned originals
(ST/tools/orig/, byte copies whose sha256 equal the round's PINS, ROUND l.749-774).

Every change is an exact string replacement that must match exactly once in the original (else this script stops):
constants (paths, RUN_IDs, labels, branch, note kind, pins) and the listed additions (launch-check fact (8), C4 (p),
C8, the measuring copy's commit binding).  Files not listed here are copied byte for byte.  The unified diffs of every
adapted file against its original go to ST/tools/adapt_diffs/.  Run: PY -I -B make_pc_tools.py  (refuses an existing
pc/ file that differs from what it would write unless --force).
"""
from __future__ import annotations

import difflib
import hashlib
import shutil
import sys
from pathlib import Path

ST_TOOLS = Path(__file__).resolve().parent
ORIG = ST_TOOLS / "orig"
OUT = ST_TOOLS / "pc"
DIFFS = ST_TOOLS / "adapt_diffs"
ADD = ST_TOOLS / "additions"          # the longer added blocks, one file each (read here, pasted verbatim)

PC_ROUND = "/home/user/work/pc_study/round"
WT = "/home/user/work/pc_branch"
BASE = "8737cd3d86431de1bdd0fcdb49a0d956a45e084d"

BYTE_COPIES = {
    "laneB/ledger_r10round.py": "ledger_r10round.py",          # M5, reused unchanged (criteria MEASUREMENT)
    "laneB/score_r10round.py": "score_r10round.py",            # classes_of reused unchanged
    "laneB/k3panel_r10round.py": "k3panel_r10round.py",        # report only, unchanged
    "laneA/zs_harness_copy.py": "zs_harness_copy.py",          # M2 harness (pinned fbb71d22), unchanged
}


def add(name: str) -> str:
    return (ADD / name).read_text(encoding="utf-8")


EDITS: dict[str, tuple[str, list[tuple[str, str]]]] = {}

# ---------------------------------------------------------------------------------------------- r10round_common.py
EDITS["laneB/r10round_common.py"] = ("r10round_common.py", [
    ('"""T. rex r10 round (criteria lean v6): shared constants and readers for the lane B tools.  Report only; never trains.\n',
     '"""PC STUDY ADAPTATION (2026-10-31; ST/tools/make_pc_tools.py; constants only plus the PC block at the end of the\n'
     'constants: RUN_IDs stance7_r10LTSBpc_s60/42/80, RUNLOG kind STANCE7, <PCR> = /home/user/work/pc_study/round, the\n'
     'branch claude/screen-trex-r10-phasecritic, the PC pins file; every rule below is the round\'s, unchanged).\n\n'
     'T. rex r10 round (criteria lean v6): shared constants and readers for the lane B tools.  Report only; never trains.\n'),
    ('VERSION = ("r10round lane B tools 1.3 (criteria lean v6 sha256 e5bea5d314cb...; built 2026-10-26; fixer 2026-10-26; "\n'
     '           "v5 tool update 2026-10-26; v6 MEAN_a 2026-10-27)")',
     'VERSION = ("r10round lane B tools 1.3, PC adaptation pc-1 (phase-critic Colab round, ST/criteria.txt; adapted "\n'
     '           "2026-10-31 from the round\'s pinned lane B tools: constants, launch-check fact (8), C4 (p), C8)")'),
    ('ROUND = SP + "/foot/stance6"                      # <ROUND> (D16)',
     'ROUND = "' + PC_ROUND + '"                # <PCR> (PC criteria SHORT NAMES)'),
    ('BR = S2 + "/r10round/runbranch/clone"             # read only',
     'BR = "' + WT + '"                     # WT, read only (PC)'),
    ('CRITERIA = S2 + "/r10round/lean/v6/criteria_r10round_lean_v6.txt"',
     'CRITERIA = "/home/user/work/pc_study/criteria.txt"     # the PC text (its sha256 is APPROVED\'s; PC)\n'
     'ROUND_CRITERIA = S2 + "/r10round/final_lean/criteria_r10round_final.txt"   # ROUND (cited by line; PC)'),
    ('CRITERIA_SHA256 = "e5bea5d314cb3c6b6c1e2889452ad6dfa8ba1a90aec8dbee758262754bed2162"',
     'CRITERIA_SHA256 = None    # PC: the PC text\'s sha256 is read from <PCR>/APPROVED, never held in a tool\n'
     'ROUND_CRITERIA_SHA256 = "99a44469a387d01c55b98d76595d3633029681866ed14afcb185521eca0164db"   # PC'),
    ('RUN_IDS = {f"stance6_r10LTSB_s{s}": s for s in SEEDS}       # D16',
     'RUN_ID_PREFIX = "stance7_r10LTSBpc_s"                     # PC (CHOICE NAMES)\n'
     'RUN_IDS = {f"{RUN_ID_PREFIX}{s}": s for s in SEEDS}         # PC'),
    ('RUN_LABELS = {60: "r10 LTSB s60", 42: "r10 LTSB s42", 80: "r10 LTSB s80"}',
     'RUN_LABELS = {s: f"r10 LTSB phase-critic s{s}" for s in SEEDS}   # PC'),
    ('REPO_REF = "claude/screen-trex-r10-armltsb"',
     'REPO_REF = "claude/screen-trex-r10-phasecritic"           # PC'),
    ('NOTE_RE = re.compile(r"^(\\d{4}-\\d{2}-\\d{2} \\d{2}:\\d{2}(?::\\d{2})?) UTC\\s+STANCE6\\s+(\\S+)\\s+(\\S+)(?:\\s+(.*))?$")\n'
     'RUN_TOKEN_RE = re.compile(r"stance6_r10LTSB_s\\d+\\w*")',
     'NOTE_KIND = "STANCE7"                                      # PC: the RUNLOG note kind (SUBST)\n'
     'NOTE_RE = re.compile(r"^(\\d{4}-\\d{2}-\\d{2} \\d{2}:\\d{2}(?::\\d{2})?) UTC\\s+STANCE7\\s+(\\S+)\\s+(\\S+)(?:\\s+(.*))?$")\n'
     'RUN_TOKEN_RE = re.compile(r"stance7_r10LTSBpc_s\\d+\\w*")'),
    ('            "v0h_digests", "model_sha256", "model_out_sha256", "v10_result", "v7_failed_on_head")',
     '            "v0h_digests", "model_sha256", "model_out_sha256", "v10_result", "v7_failed_on_head",\n'
     '            "critic_phase_switch", "critic_phase_spec")                                   # PC'),
    ('    names = ("r10round_common.py", "ledger_r10round.py", "complete_r10round.py", "score_r10round.py",\n'
     '             "fetch_r10round.py", "k3panel_r10round.py", "v9_bundle.py")',
     '    names = ("r10round_common.py", "ledger_r10round.py", "complete_r10round.py", "score_r10round.py",\n'
     '             "fetch_r10round.py", "k3panel_r10round.py", "score_pc.py", "pc_common.py", "manip_check.py",\n'
     '             "r10m_common.py", "measure_r10round.py")                                        # PC'),
    ('PINS_FILE = TOOLS / "pins_r10round.json"',
     'PINS_FILE = TOOLS / "pins_pc.json"                          # PC: the PC pins file'),
    ('TRACE_SEEDS = tuple(range(3042, 3062))                      # M1: 20 episodes\n',
     'TRACE_SEEDS = tuple(range(3042, 3062))                      # M1: 20 episodes\n' + add("common_pc_block.py")),
])

# ---------------------------------------------------------------------------------------------- complete_r10round.py
EDITS["laneB/complete_r10round.py"] = ("complete_r10round.py", [
    ('"""T. rex r10 round: the completeness tool (criteria lean v5: launch check, C1-C7, PRESENCE p1-p3, D-16).  Report only.\n',
     '"""PC STUDY ADAPTATION (2026-10-31, ST/tools/make_pc_tools.py): RUN_IDs, RUNLOG kind STANCE7 and <PCR> from the\n'
     'adapted r10round_common; launch-check fact (8) the critic-phase switch ON in 01_stance/stage_config.json (a FAULT\n'
     'like (1)-(5) and (7)); C4 (p) the same record read with C4\'s facts (and PRESENCE\'s p2); C8 THE MANIPULATION CHECK\n'
     '(manip_check.py\'s record at <panels>/<label>/c8/ for R2 and every J pair, bound to the C3-verified pair like C7;\n'
     'absent = PENDING, never a failure; NOT IN FORCE = FAIL: the run is not of the study\'s configuration, LOST).\n'
     'COMPLETE = C1-C8.\n\n'
     'T. rex r10 round: the completeness tool (criteria lean v5: launch check, C1-C7, PRESENCE p1-p3, D-16).  Report only.\n'),
    ('# ------------------------------------------------------------------------------------------------- launch check\n'
     'def launch_check(run_id, store, notes, pins) -> dict:\n',
     add("complete_helpers.py") +
     '# ------------------------------------------------------------------------------------------------- launch check\n'
     'def launch_check(run_id, store, notes, pins) -> dict:\n'),
    ('    faults = sorted(k for k, f in F.items() if f["ok"] is False)\n'
     '    # when F0\'s copies were made against the launch',
     add("complete_fact8.py") +
     '    faults = sorted(k for k, f in F.items() if f["ok"] is False)\n'
     '    # when F0\'s copies were made against the launch'),
    ('        out.append(fact("C4 stage_config_view_sha256 = pin", None if want is None else vw == want, vw,\n'
     '                        None if want is not None else "pin unfilled"))\n',
     '        out.append(fact("C4 stage_config_view_sha256 = pin", None if want is None else vw == want, vw,\n'
     '                        None if want is not None else "pin unfilled"))\n'
     '        out.append(critic_phase_fact("C4 (p) stage_config.json records the critic-phase switch ON", sc, pins))   # PC\n'),
    ('    out["measured_on_another_file"] = c7_other_file\n',
     '    out["measured_on_another_file"] = c7_other_file\n' + add("complete_c8.py")),
    ('    cs = [out[k] for k in ("C1", "C2", "C3", "C4", "C5", "C6", "C7")]\n',
     '    cs = [out[k] for k in ("C1", "C2", "C3", "C4", "C5", "C6", "C7", "C8")]                 # PC: C8\n'),
    ('    else:\n        out["verdict"] = "PENDING (C7: a measurement absent)"\n',
     '    elif out["C7"]["ok"] is None:\n        out["verdict"] = "PENDING (C7: a measurement absent)"\n'
     '    else:\n        out["verdict"] = "PENDING (C8: a manipulation check absent)"                          # PC\n'),
    ('    def certificate(self, label="final"):\n',
     add("complete_panel_manip.py") + '    def certificate(self, label="final"):\n'),
    ('    for k in ("C1", "C2", "C3", "C4", "C5", "C6", "C7"):\n        if k not in d:\n',
     '    for k in ("C1", "C2", "C3", "C4", "C5", "C6", "C7", "C8"):                              # PC: C8\n'
     '        if k not in d:\n'),
    ('        for L in render_facts(f"C1-C7 {a.run} on {stores[-1].dir}", res):',
     '        for L in render_facts(f"C1-C8 {a.run} on {stores[-1].dir}", res):                    # PC: C8'),
    ('        print(f"RUNLOG line {i} names a RUN_ID but is not a STANCE6 note (printed, read by no rule): {line[:200]}")',
     '        print(f"RUNLOG line {i} names a RUN_ID but is not a {C.NOTE_KIND} note (printed, read by no rule): "\n'
     '              f"{line[:200]}")'),
])

# ---------------------------------------------------------------------------------------------- fetch_r10round.py
EDITS["laneB/fetch_r10round.py"] = ("fetch_r10round.py", [
    ('    (every subcommand takes --round ROUND; default SP/foot/stance6)',
     '    (every subcommand takes --round ROUND; default <PCR> = /home/user/work/pc_study/round: PC adaptation)'),
    ('print(f"<date -u \'+%Y-%m-%d %H:%M UTC\'> STANCE6 LISTING {a.run} folder=absent:{key} parent={parent} "',
     'print(f"<date -u \'+%Y-%m-%d %H:%M UTC\'> {C.NOTE_KIND} LISTING {a.run} folder=absent:{key} parent={parent} "'),
    ('print(f"<date -u \'+%Y-%m-%d %H:%M UTC\'> STANCE6 LISTING {a.run} folder={lst[\'folder_id\']} pages={lst[\'pages\']} "',
     'print(f"<date -u \'+%Y-%m-%d %H:%M UTC\'> {C.NOTE_KIND} LISTING {a.run} folder={lst[\'folder_id\']} pages={lst[\'pages\']} "'),
])

# ---------------------------------------------------------------------------------------------- fetch_agent_brief.txt
EDITS["laneB/fetch_agent_brief.txt"] = ("fetch_agent_brief.txt", [
    ("FETCH AGENT BRIEF -- T. rex r10 round (criteria lean v5 FETCHES), read-only Google Drive\n",
     "FETCH AGENT BRIEF -- PC (phase-critic Colab round; ST/criteria.txt FETCHES = the r10 round's, with SUBST), read-only\n"
     "Google Drive.  PC ADAPTATION (2026-10-31, ST/tools/make_pc_tools.py): ROUND, TOOL and the scorer's name only.\n"),
    ("            score_r10round.py names)", "            score_pc.py names)"),
    ("  ROUND   = /tmp/claude-0/-home-user-mesozoic-labs/0fdf74a4-2989-56c9-a773-92101004d7de/scratchpad/foot/stance6",
     "  ROUND   = /home/user/work/pc_study/round   (<PCR>)"),
    ("  TOOL    = PY -I -B <laneB>/tools/fetch_r10round.py --round ROUND",
     "  TOOL    = PY -I -B /home/user/work/pc_study/tools/pc/fetch_r10round.py --round ROUND"),
    ("             <laneB> = the pinned tools folder; --round may stand before or after the subcommand, fixer R-B3)",
     "             the PC tools folder, pinned; --round may stand before or after the subcommand, fixer R-B3)"),
])

# ---------------------------------------------------------------------------------------------- r10m_common.py (lane A)
EDITS["laneA/r10m_common.py"] = ("r10m_common.py", [
    ('"""Shared pins and helpers of the r10 round\'s measuring tools (lane A: M1-M4).',
     '"""PC STUDY ADAPTATION (2026-10-31, ST/tools/make_pc_tools.py): <PCR>, RUN_IDs, the branch, the CRITERIA pin and\n'
     'the measuring copy\'s commit binding (approved_branch_head: HEAD from <PCR>/APPROVED, a commit of WT descending from\n'
     'the base 8737cd3d with the M1/M3 pinned files unchanged), HP_SHA256 the switch-ON pin; --header-parent now names\n'
     'the base 8737cd3d (STANCE6\'s commit: validation inputs, VP1).  Every measurement rule is the round\'s, unchanged.\n\n'
     'Shared pins and helpers of the r10 round\'s measuring tools (lane A: M1-M4).'),
    ('COMMON_VERSION = "r10m_common 1.1 (2026-10-26; v5: the branch HEAD from the approval record)"',
     'COMMON_VERSION = "r10m_common 1.1, PC adaptation pc-1 (2026-10-31; HEAD from <PCR>/APPROVED, base 8737cd3d)"'),
    ('BR = S2 + "/r10round/runbranch/clone"                        # the text\'s BR (read only)',
     'BR = "' + WT + '"                                     # WT (read only; PC)'),
    ('ROUND = SP + "/foot/stance6"                                  # the text\'s <ROUND> (D16)',
     'ROUND = "' + PC_ROUND + '"                         # <PCR> (PC)'),
    ('CRITERIA = S2 + "/r10round/lean/v5/criteria_r10round_lean_v5.txt"\n'
     'CRITERIA_SHA256 = "16f288782d77e7e35c1b9f807094556050f793b729eaf26c040ce1583bb89794"',
     'CRITERIA = "/home/user/work/pc_study/criteria.txt"          # the PC text (PC)\n'
     'CRITERIA_SHA256 = None    # PC: read from <PCR>/APPROVED, never held in a tool'),
    ('BRANCH = "claude/screen-trex-r10-armltsb"',
     'BRANCH = "claude/screen-trex-r10-phasecritic"                # PC'),
    ('HEADER_PARENT = "cf9402bc57dab92c716c7e0ee4bc96f72f237efb"',
     'HEADER_PARENT = "' + BASE + '"   # PC: the BASE, STANCE6\'s commit 8737cd3d\n'
     'BASE_COMMIT = HEADER_PARENT                                    # PC'),
    ('HP_SHA256 = "sha256:054ab48303167eda1bd66c8894a64ef516bedd041a9e12f57dbd20e75c026b9c"     # V0 (b) "054ab483..."',
     'HP_SHA256 = "sha256:c64211e8f84782e97a38141d3d2b6a7cc8f0e455b1a0524a32afb6b4a137a954"     # PC: switch ON (PINS)\n'
     'HP_SHA256_ROUND = "sha256:054ab48303167eda1bd66c8894a64ef516bedd041a9e12f57dbd20e75c026b9c"   # STANCE6 (switch OFF)'),
    ('RUN_IDS = {60: "stance6_r10LTSB_s60", 42: "stance6_r10LTSB_s42", 80: "stance6_r10LTSB_s80"}   # D16',
     'RUN_IDS = {60: "stance7_r10LTSBpc_s60", 42: "stance7_r10LTSBpc_s42", 80: "stance7_r10LTSBpc_s80"}   # PC'),
    ('def approved_branch_head(approved: str | None = None, repo: str | None = None) -> dict:\n',
     add("r10m_approved_head.py") + '\n\ndef approved_branch_head_round(approved: str | None = None, repo: str | None = None) -> dict:\n'),
])

# ---------------------------------------------------------------------------------------------- measure_r10round.py
EDITS["laneA/measure_r10round.py"] = ("measure_r10round.py", [
    ('"""The r10 round\'s M1/M2/M3 runner (lane A).',
     '"""PC STUDY ADAPTATION (2026-10-31, ST/tools/make_pc_tools.py): the RUN_ID pattern (stance7_r10LTSBpc_s60/42/80 or\n'
     'its _b) and the messages naming it; <PCR>/panels and the measuring copy\'s commit come from the adapted r10m_common\n'
     '(HEAD from <PCR>/APPROVED; --header-parent = the base 8737cd3d).  The M1/M2/M3 reductions are the round\'s.\n\n'
     'The r10 round\'s M1/M2/M3 runner (lane A).'),
    ('VERSION = "measure_r10round 1.2 (2026-10-26; fixer: RA-1, RA-4, RA-5; v5: the approved HEAD, O-6, O-7, O-8)"',
     'VERSION = ("measure_r10round 1.2, PC adaptation pc-1 (2026-10-26; fixer: RA-1, RA-4, RA-5; v5: the approved HEAD, "\n'
     '           "O-6, O-7, O-8; PC 2026-10-31: RUN_IDs stance7_r10LTSBpc_s*)")'),
    ('ROUND_RUN_RE = re.compile(r"^stance6_r10LTSB_s(60|42|80)(_b)?$")',
     'ROUND_RUN_RE = re.compile(r"^stance7_r10LTSBpc_s(60|42|80)(_b)?$")                 # PC'),
    ('raise Refusal(f"--run-id {ns.run_id!r} is not a RUN_ID of the round (stance6_r10LTSB_s60/42/80 or its _b)")',
     'raise Refusal(f"--run-id {ns.run_id!r} is not a RUN_ID of the PC round (stance7_r10LTSBpc_s60/42/80 or its _b)")'),
])


def sha(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def main() -> int:
    force = "--force" in sys.argv[1:]
    OUT.mkdir(exist_ok=True)
    DIFFS.mkdir(exist_ok=True)
    report = []
    for src, dst in BYTE_COPIES.items():
        shutil.copyfile(ORIG / src, OUT / dst)
        report.append(f"COPY  {dst:24s} = orig/{src} sha256 {sha((OUT / dst).read_bytes())}")
    for src, (dst, edits) in EDITS.items():
        text = (ORIG / src).read_text(encoding="utf-8")
        new = text
        for i, (old, rep) in enumerate(edits, 1):
            n = new.count(old)
            if n != 1:
                print(f"STOP: {src} edit {i}: the old text occurs {n} times (must be exactly once):\n{old[:300]}")
                return 2
            new = new.replace(old, rep)
        target = OUT / dst
        if target.exists() and target.read_text(encoding="utf-8") != new and not force:
            print(f"STOP: {target} exists and differs (pass --force)")
            return 2
        target.write_text(new, encoding="utf-8")
        diff = "".join(difflib.unified_diff(text.splitlines(True), new.splitlines(True), f"orig/{src}", f"pc/{dst}"))
        (DIFFS / f"{dst}.diff").write_text(diff, encoding="utf-8")
        report.append(f"ADAPT {dst:24s} from orig/{src} ({len(edits)} edits) sha256 {sha(new.encode())}")
    (ST_TOOLS / "make_pc_tools.out").write_text("\n".join(report) + "\n", encoding="utf-8")
    print("\n".join(report))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
