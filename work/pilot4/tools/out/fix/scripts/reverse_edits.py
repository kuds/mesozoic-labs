"""Fix lane: rebuild the pre-fix bytes of pilot4_straight.py, pilot4_mech.py, tests/test_pilot4_straight.py and
tests/test_nb_pilot_runner4.py by undoing this stage's edits (each one an exact replacement that must occur once), so
that their diffs can be written; the result must have the sha256 the TOOLS-stage list pinned, or nothing is written.
Reads the current files only; writes OUTDIR/<name>.

usage: python -I -B reverse_edits.py TOOLS OUTDIR
"""

from __future__ import annotations

import hashlib
import re
import sys
from pathlib import Path

TOOLS, OUT = Path(sys.argv[1]), Path(sys.argv[2])
before = {}
for ln in (TOOLS / "out" / "fix" / "SHA256SUMS_before.txt").read_text().splitlines():
    m = re.match(r"([0-9a-f]{64})  (.+)$", ln)
    if m:
        before[m.group(2)] = m.group(1)


def block(text: str, start: str, end: str) -> str:
    """The exact text from the line holding start up to (not including) the line holding end."""
    i = text.index(start)
    i = text.rindex("\n", 0, i) + 1
    j = text.index(end, i)
    j = text.rindex("\n", 0, j) + 1
    return text[i:j]


def undo_straight(s: str) -> str:
    s = s.replace(
        "    drift = meta.get(\"env_kwargs_drift_vs_recorded\")\n    if drift:  # fix stage: the probe only warns when "
        "the run's recorded env kwargs differ from the tree's\n", "@@DRIFT@@", 1)
    i = s.index("@@DRIFT@@")
    j = s.index("    return problems\n", i)
    s = s[:i] + s[j:]
    s = s.replace(block(s, "    # fix stage, report only (v3 \"Course against heading\"", "    parts_a[\"n_full_of_panel\"]"), "", 1)
    s = s.replace(
        "             f\"{f(b.get('W0_minus_D_min_deg'))}..{f(b.get('W0_minus_D_max_deg'))}, crab median \"\n"
        "             f\"{f(b.get('report_crab_median_deg'))} deg, end cross-track median \"\n"
        "             f\"{f(b.get('report_end_cross_track_median_m'), 2)} m\",\n",
        "             f\"{f(b.get('W0_minus_D_min_deg'))}..{f(b.get('W0_minus_D_max_deg'))}\",\n", 1)
    s = s.replace(
        "def make_plot(path: Path, title: str, panel: list, recovery: list, panel_note: \"str | None\" = None) -> None:\n"
        "    \"\"\"The straightness look's figure: top-down pelvis paths with the commanded line (the world x axis), and the\n"
        "    heading error psi - psi_cmd against time, for every panel and recovery episode. panel_note (fix stage, report\n"
        "    only): the panel's median crab and end cross-track, added to the path panel's title.\"\"\"",
        "def make_plot(path: Path, title: str, panel: list, recovery: list) -> None:\n"
        "    \"\"\"The straightness look's figure: top-down pelvis paths with the commanded line (the world x axis), and the\n"
        "    heading error psi - psi_cmd against time, for every panel and recovery episode.\"\"\"", 1)
    s = s.replace(
        "    ax.set_title(f\"panel: {len(panel)} pelvis paths, top-down (x marks a fall)\"\n"
        "                 + (f\"\\n{panel_note}\" if panel_note else \"\"), color=ink, fontsize=11, loc=\"left\")",
        "    ax.set_title(f\"panel: {len(panel)} pelvis paths, top-down (x marks a fall)\", color=ink, fontsize=11, "
        "loc=\"left\")", 1)
    s = s.replace("            make_plot(Path(a.plot), title, plots[0], plots[1], panel_note=crab_note(record[\"parts\"]"
                  "[\"b\"]))", "            make_plot(Path(a.plot), title, plots[0], plots[1])", 1)
    s = s.replace(block(s, "def crab_note(b: dict)", "def write_outputs("), "", 1)
    s = s.replace(
        "            (panel, 3042, 40 episodes, settle 100) or names another checkpoint, or whose own env-kwargs drift record\n"
        "            (meta.env_kwargs_drift_vs_recorded: the run's recorded env kwargs against the tree's) is not empty (fix\n"
        "            stage); this tree's locomotion env kwargs or\n"
        "            gate settle differ from probe.json's;",
        "            (panel, 3042, 40 episodes, settle 100) or names another checkpoint; this tree's locomotion env kwargs or\n"
        "            gate settle differ from probe.json's;", 1)
    s = s.replace(
        "(a, b, c: verdict PASS / FAIL / NOT READ with its values and counts, b with R and the last mean's error as report,\n"
        "and, fix stage, report_crab_median_deg and report_end_cross_track_median_m, the medians over the same full-length\n"
        "episodes of crab_median_deg and end_cross_track_m, also printed and written on the path panel of the look's figure;",
        "(a, b, c: verdict PASS / FAIL / NOT READ with its values and counts, b with R and the last mean's error as report;",
        1)
    return s


def undo_mech(s: str) -> str:
    return s.replace("  # noqa: F841 - survey_analyze line; its use there (clearance, impulse) is not ported", "")


def undo_test_straight(s: str) -> str:
    s = s.replace("import subprocess\nimport sys\nimport textwrap\n", "import subprocess\nimport sys\n", 1)
    s = s.replace(block(s, "def test_r_reads_m950_states_900_to_1000_only():", "def test_the_share_removed_against"), "", 1)
    s = s.replace(block(s, "def test_a_probe_json_with_an_env_kwargs_drift_is_refused(", "def test_baseline_refuses_another"),
                  "", 1)
    # the three blocks above end with two blank lines before the next def: the originals kept them
    s = s.replace(block(s, "def test_tree_baseline_reports_the_crab_and_the_cross_track(",
                        "def test_tree_baseline_states_0_650"), "", 1)
    i = s.index("def val_record(name):")
    j = s.index("def first20(rec):", i)
    s = s[:i] + ("def val_record(name):\n    if VAL is None or not (VAL / name).is_file():\n"
                 "        pytest.skip(f\"no validation record {name} (STRAIGHT_VAL)\")\n"
                 "    return json.loads((VAL / name).read_text())\n\n\n") + s[j:]
    i = s.index("\n\n\n# v3 \"Report only\", course against heading")
    s = s[:i] + "\n"
    return s


def undo_test_runner(s: str) -> str:
    i = s.index("@pytest.mark.parametrize(\"phase\", [\"launch\", \"resume\"])\n@pytest.mark.parametrize(\"quick\"")
    j = s.index("@needs_runs\ndef test_an_implicit_sidecar_or_another_seeds_pair", i)
    return s[:i] + s[j:]


OUT.mkdir(parents=True, exist_ok=True)
ok = {}
for rel, fn in (("pilot4_straight.py", undo_straight), ("pilot4_mech.py", undo_mech),
                ("tests/test_pilot4_straight.py", undo_test_straight),
                ("tests/test_nb_pilot_runner4.py", undo_test_runner)):
    try:
        old = fn((TOOLS / rel).read_text())
    except ValueError as exc:
        ok[rel] = f"not rebuilt ({exc})"
        continue
    got = hashlib.sha256(old.encode()).hexdigest()
    if got == before.get(rel):
        (OUT / Path(rel).name).write_text(old)
        ok[rel] = f"rebuilt, sha256 {got} = the TOOLS-stage pin"
    else:
        (OUT / (Path(rel).name + ".mismatch")).write_text(old)
        ok[rel] = f"NOT equal to the pin ({got} != {before.get(rel)})"
for k, v in ok.items():
    print(k, v)
