"""Edit the pilot-4 frozen candidate with the render stage's record and decisions (37)-(39).

Usage: apply_render.py INPUT OUTPUT [--target-must-equal-input | --target-sha=<sha256>]

INPUT must be the candidate as the pre-launch fix stage left it (sha256 6a8b3340...). Every OLD text must occur
exactly once in INPUT and exactly once in the text at the moment it is replaced; otherwise nothing is written.
After the edits: every line <= 118 characters, ASCII only, and every protected rule string keeps its count.
With --target-must-equal-input, OUTPUT must already hold INPUT's bytes before it is overwritten; with
--target-sha, OUTPUT must already hold the named bytes (this script's own earlier output, rebuilt from INPUT).
"""
import hashlib
import sys

EXPECT_IN = "6a8b334060c236245ef26735677d9de7a6a744a58d7a6e137fd1bfad43cf6546"

EDITS = []


def edit(name, old, new):
    EDITS.append((name, old, new))


# E01 title: decisions (31)-(39) and the render stage's record.
edit("E01 title", """\
and its RESUME test (32,768 -> 65,536); built from draft v3 with the maintainer's decisions (31)-(36), the tools' fix
stage's 18 text corrections, the pre-launch checks' measurements and the pre-launch fix stage's edits, see "Changes
from v3" at the end, then "Changes in v3" and "Changes in v2"; not yet approved or frozen)
""", """\
and its RESUME test (32,768 -> 65,536); built from draft v3 with the maintainer's decisions (31)-(39), the tools' fix
stage's 18 text corrections, the pre-launch checks' measurements, the pre-launch fix stage's edits and the render
stage's record, see "Changes from v3" at the end, then "Changes in v3" and "Changes in v2"; not yet approved or
frozen)
""")

# E02 Status: decisions, what is done, what stays open.
edit("E02 status open list", """\
for a change. The maintainer took every item Q1-Q27 of draft v3 as proposed and decided (35) and (36) ("Decisions
taken"); no bar, seed, budget, cadence or outcome rule differs from v3's, and the one value that does is the
straightness tool's turned-twin tolerance, a check in its tests, 1e-5 -> 1e-4 deg (decision (35)). Every pre-launch
check is filled from measured evidence or written as not run yet; open, in this order ("Pre-launch checks"): the
maintainer's approval of this text (its sha256 recorded with the approval; it covers the smoke RESUME's departure from
pilot 3's procedure named there) and of the fourteen local commits (Q18), the OK to push, the push and the pushed head
checked on the GitHub remote by its URL, the durable copy of the pinned tools, the video tool's first render (or its
waiver), the Colab steps, and the launch approval. No seed is launched before all of them [P3 L15-16].
""", """\
for a change. The maintainer took every item Q1-Q27 of draft v3 as proposed and decided (35)-(39) ("Decisions
taken"); no bar, seed, budget, cadence or outcome rule differs from v3's, and the one value that does is the
straightness tool's turned-twin tolerance, a check in its tests, 1e-5 -> 1e-4 deg (decision (35)). Every pre-launch
check is filled from measured evidence or written as not run yet. Done since the pre-launch stages: the video tool's
first render and the notebook replay, rendered on this machine with MuJoCo's glfw backend under a private Xvfb
display, as pilot 3 rendered [P3 L1058-1059] (decision (37); "Pre-launch checks"); the smoke RESUME's departure from
pilot 3's procedure, accepted (decision (38)). Open, in this order ("Pre-launch checks"): the maintainer's approval of
this text (its sha256 recorded with the approval) and of the fourteen local commits (Q18); then, approved by
decision (39) for after that approval and not done yet, the push with an explicit refspec, the pushed head checked on
the GitHub remote by its URL, and the durable copy of the pinned tools and the evidence on an orphan branch; the
Colab steps; and the launch approval. No seed is launched before all of them [P3 L15-16].
""")

# E03 Status: pin the render stage's evidence.
edit("E03 status render pin", """\
  sha256sum -c 614 OK from SP/work/pilot4/prelaunch [measured: prelaunch/fix/pin/lanes_check.txt].
""", """\
  sha256sum -c 614 OK from SP/work/pilot4/prelaunch [measured: prelaunch/fix/pin/lanes_check.txt]. The render
  stage's evidence (every file of SP/work/pilot4/render/render/ but its CMDLOG.txt and its clone tree/; its tmp/ is
  empty) is pinned by its SHA256SUMS_render.txt, sha256
  22e87a5b588f4acc64e62df5400225b9ceab719cb151e65d697b072e94a5afc1 (324 entries), sha256sum -c 324 OK from that
  folder [measured: render/text/logs/render_manifest_check.txt, render_files_all.txt and
  render_files_manifest.txt].
""")

# E04 Status: where the render stage's files are cited from.
edit("E04 status citations", """\
SP/work/pilot4/prelaunch/fix/ (cited as prelaunch/fix/<file>, so as not to be read as the fix stage's fix/logs/
above).
""", """\
SP/work/pilot4/prelaunch/fix/ (cited as prelaunch/fix/<file>, so as not to be read as the fix stage's fix/logs/
above). The render stage's renders are in SP/work/pilot4/render/render/ and this text's checks of them in
SP/work/pilot4/render/text/ (cited as render/render/<file> and render/text/<file>).
""")

# E05 decisions behind this pilot: (31)-(39).
edit("E05 decisions pointer", """\
  (31)-(36), 2026-11-06, on draft v3 and before the freeze: "Decisions taken" below [measured: RL 1093,
  1096].
""", """\
  (31)-(39), 2026-11-06, on draft v3 and before the freeze: "Decisions taken" below [measured: RL 1093,
  1096, 1100].
""")

# E06 Video review: s42's replay reset seed is measured now.
edit("E06 video review reset seed", """\
  one deterministic 1,000-step episode at the notebook's reset seed, the training seed + 2000 + stage 2: 2044, 2054,
  2062 [inferred: the same code, P3 L595-600]. (b) The panel episodes the scorer lists at the deciding checkpoint,
""", """\
  one deterministic 1,000-step episode at the notebook's reset seed, the training seed + 2000 + stage 2: 2044 for s42
  [measured: the render stage's replay of the smoke's run, replay_seed(42, 2) = 2044 at both record_stage_video
  calls, render/render/replay/run1/replay_record.json; "Pre-launch checks"], 2054 for s52 and 2062 for s60
  [inferred: the same code, P3 L595-600]. (b) The panel episodes the scorer lists at the deciding checkpoint,
""")

# E07 Pre-launch checks heading: the render stage's folders.
edit("E07 checks heading", """\
smoke and its RESUME; each filled from measured evidence, or written as not run yet; the lanes' folders are cited as
setup/, tol/, smoke/, text/ and prelaunch/fix/ ("Status"), each with a CMDLOG.txt of its commands and their exit
statuses)
""", """\
smoke and its RESUME; each filled from measured evidence, or written as not run yet; the lanes' folders are cited as
setup/, tol/, smoke/, text/, prelaunch/fix/, render/render/ and render/text/ ("Status"), each with a CMDLOG.txt of its
commands and their exit statuses)
""")

# E08 push item: decision (39).
edit("E08 push item", """\
    Not run yet (required before launch, after the freeze): the pushed head equals the local head. The branch is not
    pushed (Q18); after the push, git ls-remote https://github.com/kuds/mesozoic-labs.git
""", """\
    Not run yet (required before launch, after the freeze): the pushed head equals the local head. The branch is not
    pushed (Q18); the push, with an explicit refspec, is approved for after the freeze approval (decision (39)) and
    not done yet; after the push, git ls-remote https://github.com/kuds/mesozoic-labs.git
""")

# E09 smoke item: what the smoke itself did not render, and where it was rendered.
edit("E09 smoke replay", """\
      driver values (null on CPU). The final replay was not rendered ("Video recording failed." under
      MUJOCO_GL=disable; neither EGL nor OSMesa works on this machine), so its reset seed is not checked here.
""", """\
      driver values (null on CPU). The smoke rendered no replay: under MUJOCO_GL=disable its first
      record_stage_video call (the selected checkpoint) raised, logged as "Video recording failed.", and the final
      replay, in the same try, was not attempted [measured: smoke/nb/launch_s42.console.txt line 126, the traceback
      at stage_artifacts.py line 2225; measured by code reading: environments/shared/reporting/stage_artifacts.py,
      one try over both calls, lines 2174-2278 at 0e6a57b]. Neither EGL nor OSMesa works on this machine; glfw under a
      private Xvfb display does, and both replays were rendered afterwards on a copy of this run, the reset seed
      checked (the next item's render record).
""")

# E10 RESUME departure: decision (38).
edit("E10 resume accepted", """\
    Done, the RESUME test, with one departure from pilot 3's procedure for the maintainer to confirm [measured:
    smoke/scripts/runner_snapshot.py, snap/snapshot.json, nb/resume/record.json and runner.stdout.log,
    nb/resume_stage_config_before.json and _after.json, out/compare_resume.json, runs/resume/pilot4_walk_s42/]:
""", """\
    Done, the RESUME test, with one departure from pilot 3's procedure, accepted by the maintainer on 2026-11-06
    (decision (38), RL 1100: "Accept the copy (Recommended)") [measured: smoke/scripts/runner_snapshot.py,
    snap/snapshot.json, nb/resume/record.json and runner.stdout.log, nb/resume_stage_config_before.json and
    _after.json, out/compare_resume.json, runs/resume/pilot4_walk_s42/]:
""")

# E11 the video item: the render record in place of "Not run".
edit("E11 render record", """\
    - Not run: pilot4_video.py (0c904fc4...) stopped at import under MUJOCO_GL=egl and osmesa (no libEGL and no
      libOSMesa on this machine), so the panel videos and the video tool's re-roll check wait for a machine that
      renders (pilot 3's smoke used a private Xvfb with glfw [P3 L1058-1059], which writes outside the lane's
      folders). The notebook's final replay was not rendered either (the smoke item above), so the replay's reset seed
      (2044 / 2054 / 2062, "Video review") stays inferred. Before launch either a machine that renders shows both, or
      the maintainer accepts in writing that the first renders are the Colab replays and the judging run's (the
      pending list). pilot4_mirror.py, pilot4_mech.py and pilot3_vg.py were not run on the smoke.
""", """\
    - Rendered (decision (37); the render stage, 2026-11-06, 14:11-15:12 UTC [measured: render/render/logs/
      video_run1.rc to replay_run2.rc]; the renders are the video tool's and the replay writer's, and nothing trained
      [measured as recorded: render/render/CMDLOG.txt]). In the smoke lane pilot4_video.py (0c904fc4...) had stopped
      at import under MUJOCO_GL=egl and osmesa (no libEGL and no libOSMesa on this machine). The render stage used
      MuJoCo's glfw backend under a private Xvfb display, as pilot 3's smoke rendered [P3 L1058-1059], through pilot
      3's wrapper SP/work/pilot3/judging/score/bin/withx (e342c9f2..., pilot 3's copy of SP/work/pilot3/r6_fix/bin/
      withx) copied to render/render/bin/withx (d1503ac7...) with one line changed, its S= folder, which also moves
      the Xvfb log: Xvfb :97 with no TCP or unix-socket listener, MUJOCO_GL=glfw, MESA_SHADER_CACHE_DISABLE=true
      [measured: the wrapper, render/render/logs/withx_diff_vs_p3_score.txt]; software GL, Mesa llvmpipe [measured as
      recorded: the render lane's gl_check.py run, its output not kept in a file].
    - The video tool [measured: render/render/logs/video_run1.log and video_run2.log, compare_videos.json,
      videos_run1_SHA256SUMS.txt, videos/run1/s42/videos.json; recomputed in render/text/data/render_facts.json]:
      pilot4_video.py through iso.py with the render line of the smoke's validation score JSON (f421c999...; written
      by pilot4_score.py line 1086: OUT/s42/65536 --from-score <that JSON>), on copies of that JSON and of the
      smoke's judged cell OUT/s42/65536 (its 6 files equal by sha256), from a clean clone at 0e6a57b
      (clean before and after, and tools/tree too); the tool read the judged pair f0ee303a... / aebded4b... where
      binding.json names it, in the smoke's run folder (unchanged after). Exit 0; 40 of 40 re-rolled panel episodes
      equal probe.json, and on each the L and R boxes equal the probe's floor-contact channel; 80 mp4s and
      videos.json: for each WALK episode (resets 3042-3081) a real-time file (100 fps, 10.0 s) and a quarter-speed
      file (25 fps, 40.0 s), each h264 640x480 with 1,000 frames, 49,527,598 bytes in all; camera tracking, distance
      3.0, azimuth 135, elevation -20. A second render, run at the same time in the same Xvfb, is byte-identical: 80
      of 80 sha256 equal, videos.json equal but for its output paths.
    - The notebook replay [measured: render/render/replay/run1/ and run2/ (replay_record.json and the mp4s),
      logs/replay_run1.log; recomputed in render/text/data/render_facts.json]: the replay writer of the manual cell's
      final save (generate_stage_artifacts -> reporting.stage_artifacts._record_stage_replays ->
      evaluation.record_stage_video, the selected checkpoint, then the final one) called with the notebook's
      arguments (tyrannosaurus_rex / anatomical, load_all_stages, stage 2, ppo, seed 42) on a copy of the smoke's
      launch run (56 files, equal by sha256), twice: tyrannosaurus_rex_anatomical_ppo_stage2_final.mp4, h264 640x480,
      100 fps, 1,000 frames, 10.0 s, 783,375 bytes, sha256
      00e67bb7d20205cc6266fea16a7c74279b5dafb109735c179780d010c147c528 (reward 3050.87 in the writer's log). Reset
      seed 2044: both record_stage_video calls computed replay_seed(42, 2) = 2044 (recorded by a wrapper that returns
      the value unchanged), the seed passed to the render env's reset [measured by code reading:
      environments/shared/evaluation.py line 359 at 0e6a57b]. The second run is byte-identical (0 of 1,000 frames
      differ), no warnings. The ..._stage2_selected.mp4 (robust_best_model, byte-equal to the 32,768 zip
      4ab2de8b...) is 085fc77c..., also byte-identical.
    - Looked at, not the maintainer's video review (a throwaway QUICK_TEST model): the T. rex walks upright in every
      panel episode and in both replays, the L and R boxes alternating through each stride; the prey sphere comes
      into view ahead, passes beside the feet (about 3.7 s in panel episode 0, about 5 s in the final replay) and
      drops behind; by 10 s the body has turned toward the fixed-angle camera [measured as recorded: the render
      lane's frames, render/render/frames/, two sheets looked at again here; inferred: the veer the (a) readout
      measures, median -27.8 deg].
    - Outside its folder the render stage's checks found one change, the /var/lib/xkb directory's modification time,
      at each Xvfb start, nothing left there but README.compiled [measured: render/render/logs/final_checks.txt;
      inferred: Xvfb's keymap compiler]; the pinned tools' list 1152 OK before and after; the smoke's files
      unchanged.
    - pilot4_mirror.py, pilot4_mech.py and pilot3_vg.py were not run on the smoke.
""")

# E12 tools item: the first render is done.
edit("E12 tools item", """\
    Done but for the durable copy and pilot4_video.py's first render (not run: the fourth item above; the tool is
    unit-tested, 15 cases, and differs from pilot 3's validated pilot3_video.py only in docstrings, its usage example,
""", """\
    Done but for the durable copy (pilot4_video.py's first render, made by the render stage on the smoke's judged
    cell with 40 of 40 re-rolled episodes equal to probe.json, is in the fourth item above; the tool is unit-tested,
    15 cases, and differs from pilot 3's validated pilot3_video.py only in docstrings, its usage example,
""")

# E13 durable copy item: decision (39).
edit("E13 durable copy", """\
    Not run yet (required before launch, after the freeze): the durable copy outside the session scratchpad [P3
    L1401-1403]: the pinned files, byte-identical, checked by sha256sum -c against SHA256SUMS.txt. The list names the
""", """\
    Not run yet (required before launch, after the freeze): the durable copy outside the session scratchpad [P3
    L1401-1403], which decision (39) approves for after the freeze approval as a copy of the tools and the evidence
    on an orphan branch (never merged, no pull request): the pinned files, byte-identical, checked by sha256sum -c
    against SHA256SUMS.txt. The list names the
""")

edit("E14 durable copy render check", """\
    1152 OK; where the copy also keeps the lanes' evidence, sha256sum -c fix/pin/SHA256SUMS_lanes.txt run from the
    copy's prelaunch folder must give 614 OK.
""", """\
    1152 OK; where the copy also keeps the lanes' evidence, sha256sum -c fix/pin/SHA256SUMS_lanes.txt run from the
    copy's prelaunch folder must give 614 OK, and, for the render stage's evidence, sha256sum -c
    SHA256SUMS_render.txt run from the copy's render/render folder must give 324 OK.
""")

# E15 pending list.
edit("E15 pending list", """\
  Pending, in this order: the maintainer's approval of this text (its sha256; with it the smoke RESUME's departure
  above) and of the fourteen local commits; the OK to push, then the push with an explicit refspec to the same-named
  remote branch, never a bare git push [P3 L1397-1400], after which git ls-remote
  https://github.com/kuds/mesozoic-labs.git refs/heads/<prefix>/pilot-walk-r10-p4 must print
  0e6a57bb4d6ba30aef36dd9aa30ff4d175786cc8 (by its URL, not origin: the first item), and nothing more is pushed to it
  until every seed is judged ("Code"); the durable copy of the pinned tools; the video tool's first render and the
  notebook replay's reset seed (2044 for s42) on a machine that renders, or the maintainer's written acceptance that
  the first renders are the Colab replays and the judging run's; the Colab steps (COLAB_STEPS_PILOT4.txt from pilot
  3's COLAB_STEPS_PILOT3.txt, written from this text once frozen), reviewed against this text; the maintainer's launch
  approval.
""", """\
  Pending, in this order: the maintainer's approval of this text (its sha256) and of the fourteen local commits (the
  smoke RESUME's departure above is accepted, decision (38)); then, approved by decision (39) for after that approval
  and not done yet, the push with an explicit refspec to the same-named remote branch, never a bare git push [P3
  L1397-1400], after which git ls-remote https://github.com/kuds/mesozoic-labs.git
  refs/heads/<prefix>/pilot-walk-r10-p4 must print 0e6a57bb4d6ba30aef36dd9aa30ff4d175786cc8 (by its URL, not origin:
  the first item), and nothing more is pushed to it until every seed is judged ("Code"), and the durable copy of the
  pinned tools and the evidence on an orphan branch (never merged, no pull request); the Colab steps
  (COLAB_STEPS_PILOT4.txt from pilot 3's COLAB_STEPS_PILOT3.txt, written from this text once frozen), reviewed
  against this text; the maintainer's launch approval. The video tool's first render and the notebook replay with
  its reset seed (2044 for s42), pending before, are done on this machine (decision (37); the fourth item above).
""")

# E16 Decisions taken: header sources.
edit("E16 decisions header", """\
(AskUserQuestion, on draft v3 Q1-Q27)"; RL 1096, 11:21 UTC, "MAINTAINER DECISIONS (AskUserQuestion)"]. The
alternatives v3 listed beside each proposal were not chosen and are not carried here (v3 keeps them).
""", """\
(AskUserQuestion, on draft v3 Q1-Q27)"; RL 1096, 11:21 UTC, and RL 1100, 14:05 UTC, "MAINTAINER DECISIONS
(AskUserQuestion)"]. The alternatives v3 listed beside each proposal were not chosen and are not carried here (v3
keeps them).
""")

# E17 Decisions taken: (37)-(39).
edit("E17 decisions 37-39", """\
       a throwaway model; approved), the judge, score and resume checks on its output, then the 18 text corrections
       and the tolerance applied to make this text; the push and the Colab launch still wait.
""", """\
       a throwaway model; approved), the judge, score and resume checks on its output, then the 18 text corrections
       and the tolerance applied to make this text; the push and the Colab launch still wait.
  (37) "Render here, then freeze (Recommended)": render pilot4_video.py and the notebook replay on the smoke's output
       with MuJoCo's glfw backend under a private Xvfb display (no training), add the evidence to this text, then
       the maintainer approves its final sha256 (rendered: "Pre-launch checks").
  (38) The RESUME test, "Accept the copy (Recommended)": the smoke's RESUME from a copy of its own 32,768 pair, the
       departure from pilot 3's procedure stated under "Pre-launch checks", is accepted.
  (39) The push, "Yes, after the freeze (Recommended)": after the freeze approval, push pilot-walk-r10-p4 with an
       explicit refspec, and a durable copy of the tools and the evidence to an orphan branch (never merged, no pull
       request); neither is done yet.
""")

# E18 Q18: the OK to push is given for after the freeze.
edit("E18 Q18", """\
  Q18. The approvals, (34): the order as proposed; the approvals themselves are still open: this text once frozen
      (its sha256), the fourteen local commits (seven built, four review fixes, three postfix fixes), the OK to push
      and the launch ("Status", "Pre-launch checks").
""", """\
  Q18. The approvals, (34): the order as proposed; the approvals themselves are still open: this text once frozen
      (its sha256), the fourteen local commits (seven built, four review fixes, three postfix fixes) and the launch;
      the OK to push is given for after the freeze approval (decision (39)) ("Status", "Pre-launch checks").
""")

# E19 Changes from v3: intro names this script.
edit("E19 changes intro", """\
every replaced text before replacing it, then edited by the pre-launch fix stage's
SP/work/pilot4/prelaunch/fix/scripts/apply_fix.py, which checks every replaced text the same way: the last bullet)
""", """\
every replaced text before replacing it, then edited by the pre-launch fix stage's
SP/work/pilot4/prelaunch/fix/scripts/apply_fix.py, which checks every replaced text the same way (the next-to-last
bullet), and by the render stage's SP/work/pilot4/render/text/scripts/apply_render.py, the same way (the last bullet))
""")

# E20 Changes from v3: the render stage's bullet.
edit("E20 changes bullet", """\
    straightness record; the test-target, end-to-end, smoke-runner and lineage sentences are made exact; the TOML's
    "PROPOSED" comments are noted under "What is trained".
""", """\
    straightness record; the test-target, end-to-end, smoke-runner and lineage sentences are made exact; the TOML's
    "PROPOSED" comments are noted under "What is trained".
  - The render stage (decisions (37)-(39); SP/work/pilot4/render/; the text before it kept as
    render/text/criteria_pilot4_before_render.txt, sha256 6a8b3340...; render/text/scripts/apply_render.py checks
    every replaced text), on the render lane's record; no bar, seed, budget, cadence, outcome rule or decided value
    and no tool changed (the tools' SHA256SUMS.txt still 0f42f29f..., sha256sum -c 1152 OK [measured:
    render/text/logs/sha_before.txt and sha_after.txt]): every passage that said the video tool or the notebook
    replay was not rendered, that no EGL or OSMesa meant no render, or that a first render or its waiver was still
    required now gives the render record or points to it (Status, the smoke item, the video item and the tools item
    of "Pre-launch checks", the pending list); s42's replay reset seed reads measured under "Video review" (2054 and
    2062 stay inferred); the RESUME departure is accepted (decision (38)); the push and the durable copy are
    approved for after the freeze approval and not done (decision (39): Status, the push and durable-copy items, the
    pending list, Q18); the title, Status, "Maintainer decisions behind this pilot" and "Decisions taken" name
    (37)-(39); Status pins the render stage's evidence and the lanes' folders name its two.
""")

# Rule strings whose counts must not change (bars, seeds, budget, cadence, outcome rules, decided values).
PROTECTED = [
    "|median of the signed values| <=\n      5.0 deg AND the 95th percentile of |yaw_change_deg| <= 15.0 deg",
    "Pass: the 95th percentile of D <= 11.5 deg AND the 95th percentile of W0 <= 18.5 deg",
    "Pass: median r <= 5.0 deg AND p95 r <= 11.5 deg",
    "With fewer than 30 counting episodes, (c) is NOT READ",
    "nudged once by +15 deg and once by -15 deg",
    "after state 650 (6.5 s)",
    "0.35 m",
    "Seeds 42, 52 and 60",
    "--seeds 42,52,60",
    "2,000,000",
    "2,015,232",
    "65,536",
    "250,000",
    "1064.0",
    "478.8",
    "TWIN_HEADING_TOL_DEG 1e-4",
    "STRAIGHT = (a) and (b) and (c)",
    "NOT WALKS (<label>)",
    "CURVED WALKS           WALKS and video-agreed",
    "STRAIGHT NO RECOVERY   WALKS and video-agreed",
    "STRAIGHT WALKS         WALKS and video-agreed",
    "(WALK count - 30)",
    "40 deterministic panel episodes on resets 3042-3081",
    "reset seeds 3042-3061",
    "walk_heading_price_weight = 0.2",
    "eval_freq 65536 (decision (33)), eval_episodes 10, checkpoint_freq 250000",
    "0e6a57bb4d6ba30aef36dd9aa30ff4d175786cc8",
    "0f42f29f45092ad87004519783af5175bcb1b7a7c61dcccd4985b682554cc448",
    "914a1caee2c086a09ef66e9e6d4ad234dd085b69c4da687cc29af82b110fcd7a",
    "fbc94bef69a91d5a65f1df99adb9c35e7897a259affaa3e7b7ba4477418eaa62",
]


def sha(b):
    return hashlib.sha256(b).hexdigest()


def main():
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    flags = [a for a in sys.argv[1:] if a.startswith("--")]
    target_sha = None
    for a in list(flags):
        if a.startswith("--target-sha="):
            target_sha = a.split("=", 1)[1]
            flags.remove(a)
    if len(args) != 2 or flags not in ([], ["--target-must-equal-input"]) or (flags and target_sha):
        sys.exit("usage: apply_render.py INPUT OUTPUT [--target-must-equal-input | --target-sha=<sha256>]")
    src, dst = args
    raw = open(src, "rb").read()
    if sha(raw) != EXPECT_IN:
        sys.exit(f"REFUSED: input sha256 {sha(raw)} is not {EXPECT_IN}")
    orig = raw.decode("ascii")
    text = orig
    for name, old, new in EDITS:
        n0 = orig.count(old)
        n1 = text.count(old)
        if n0 != 1 or n1 != 1:
            sys.exit(f"REFUSED: {name}: OLD occurs {n0} times in the input and {n1} times now (need 1 and 1)")
        text = text.replace(old, new, 1)
        print(f"applied {name}: -{old.count(chr(10))} +{new.count(chr(10))} lines")
    bad = [(i + 1, len(l)) for i, l in enumerate(text.split("\n")) if len(l) > 118]
    if bad:
        sys.exit(f"REFUSED: lines over 118 characters: {bad}")
    try:
        text.encode("ascii")
    except UnicodeEncodeError as e:
        sys.exit(f"REFUSED: non-ASCII character: {e}")
    for p in PROTECTED:
        a, b = orig.count(p), text.count(p)
        if a < 1 or a != b:
            sys.exit(f"REFUSED: protected string count {a} -> {b}: {p!r}")
    print(f"protected strings: {len(PROTECTED)} unchanged in count")
    out = text.encode("ascii")
    if "--target-must-equal-input" in flags:
        cur = open(dst, "rb").read()
        if sha(cur) != EXPECT_IN:
            sys.exit(f"REFUSED: target sha256 {sha(cur)} is not the input's {EXPECT_IN}")
    if target_sha:
        cur = open(dst, "rb").read()
        if sha(cur) != target_sha:
            sys.exit(f"REFUSED: target sha256 {sha(cur)} is not the named {target_sha}")
    with open(dst, "wb") as f:
        f.write(out)
    nl = out.count(b"\n")
    print(f"wrote {dst}: sha256 {sha(out)}, {nl} lines")


if __name__ == "__main__":
    main()
