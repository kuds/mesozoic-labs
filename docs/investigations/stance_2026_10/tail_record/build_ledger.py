"""Merge the three tail-habit sweeps into one ledger (ledger.csv), one row per merged entry.

Read only: the three sweeps' entries.json. Writes ledger.csv and ledger_index.json next to the draft.
Sweep ids: R = sweep_repo (list index), S = sweep_session, T = sweep_today, W = found by the writer.
Path prefixes are normalised to the repository's convention: 'scratchpad/' = the first session's scratch root
(the sweeps' SP or S1), 'scratchpad2/' = this session's (the sweeps' S2). Repo paths are relative to the repo root.
"""
import csv
import json
import os
import re

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))  # .../tail_record
OUT = os.path.dirname(HERE)  # .../tail_record/draft


def load():
    rep = json.load(open(os.path.join(ROOT, "sweep_repo", "entries.json")))
    ses = json.load(open(os.path.join(ROOT, "sweep_session", "entries.json")))["entries"]
    tod = json.load(open(os.path.join(ROOT, "sweep_today", "entries.json")))["entries"]
    E = {}
    for i, e in enumerate(rep):
        E["R%02d" % i] = e
    for i, e in enumerate(ses):
        E["S%02d" % i] = e
    for i, e in enumerate(tod):
        E["T%02d" % i] = e
    return E


W = {
    "W01": {
        "date": "2026-10-10",
        "kind": "observation",
        "title": "Stance reward pilots: the gate's saturation key reads raw normalised commands, not force clipping",
        "what": "The stance reward pilot findings (2026-10-10) list what the recorder measures and what any plant work must separate.",
        "result": "[measured, reading of code] 'The recorder's max_actuator_saturation_fraction measures raw normalized actions, not actual actuator-force clipping.' Its follow-up table asks, before geometry or strength changes, to 'separate actual actuator-force clipping from raw normalized command saturation', and its validation list asks to report 'raw command saturation and actual force clipping'. [inferred] The satdiag of 2026-10-21 found exactly this split on the tail: a command pinned at +1 with the joint at 0.26 of its force limit.",
        "status": "supported",
        "label": "measured",
        "sources": [
            "docs/investigations/STANCE_REWARD_PILOT_FINDINGS_2026_10.md §2 (lines 192, 209-210) and §3 (line 233)",
        ],
    },
    "W02": {
        "date": "2026-10-06",
        "kind": "observation",
        "title": "Another species: the velociraptor parked tail_1_pitch at its command limit as a counterweight",
        "what": "Stance-hack audit §3.2, the velociraptor's final checkpoint (a different model; noted only as a precedent).",
        "result": "[measured] '`tail_1_pitch` parked at its -15 deg command limit on 95% of steps as a counterweight, both knees saturated, 37% of all action components saturated.' [inferred] A parked tail on another species, read there as a counterweight; on the T. rex the counterweight reading was later ruled out (2026-10-29 static CoM check and clamp test). Not part of the T. rex record otherwise.",
        "status": "n/a",
        "label": "measured",
        "sources": ["docs/investigations/STANCE_HACK_AUDIT_2026_10.md §3.2 (line 170)"],
    },
}

# (sort key, [ids: primary first], overrides) -- overrides: date, title, status, sect, disagree
G = [
    # ---- 3.1 July-August 2026: stage 1, physics r6 and r7 ----
    ("2026-07-09", ["R00"], dict(sect="§3.1")),
    ("2026-07-27a", ["R01"], dict(sect="§3.1",
        disagree="KNOWN_ISSUES keeps this July MEDIUM open, while the 2026-08-15 item 'Half the actuators sit saturated and nothing opposes it' was deleted as fixed; the habit came back on r9 with the #504 terms in place (repo sweep note 8).")),
    ("2026-07-27b", ["R02", "S31"], dict(sect="§3.1; §4.5", date="2026-07-27 (narrowed 2026-10-24)")),
    ("2026-08-05", ["R03"], dict(sect="§3.1")),
    ("2026-08-06a", ["R04"], dict(sect="§3.1; §4.2",
        disagree="The bounce note said saturation is 'conspicuous, not causal' and that a tail- or saturation-targeted term would be the wrong reaction; days later #504 added exactly those terms, the next run had no saturated actuator, and from October the rail itself fails the gate (repo sweep note 7).")),
    ("2026-08-06b", ["R05"], dict(sect="§3.1")),
    ("2026-08-06c", ["R06"], dict(sect="§3.1; §4.6")),
    ("2026-08-06d", ["R07"], dict(sect="§3.1; §4.9")),
    ("2026-08-06e", ["R08"], dict(sect="§6.1")),
    ("2026-08-09", ["R09"], dict(sect="§3.1")),
    ("2026-08-10a", ["R10"], dict(sect="§3.1",
        disagree="Which actuators run 20260809_155206 parked: its §1 item 3 says five of 15 (knees, head_pitch, tail_2_pitch, tail_1_yaw); its §3 table lists tail_3_pitch at -8.6 deg +-8.0 as an 'active counterweight'; its §6 item 3 says six of 15 'and the entire tail yaw/pitch chain'; stance.toml lines 268-270 and 285-287 say tail_1_yaw, tail_2_pitch and tail_3_pitch were 'all three at stops by 10M' (6 of 15); CHANGELOG 0.3.8 line 6888 gives 5 of 15; the gate-pass doc's table 6 of 15. Whether tail_3 was parked is unresolved (repo sweep note 1).")),
    ("2026-08-10b", ["R11"], dict(sect="§3.1; §4.3; §4.4")),
    ("2026-08-10c", ["R12"], dict(sect="§3.1; §5",
        disagree="CHANGELOG #504 says 'the passive tail rests on its ventral stops'; the 2026-10-25 records (§25.5, plant_versions note 20) say tail_1 alone rested on its stop on r9 (repo sweep note 2).")),
    ("2026-08-11", ["R13"], dict(sect="§3.1")),
    ("2026-08-15a", ["R14"], dict(sect="§3.1")),
    ("2026-08-15b", ["R15"], dict(sect="§3.1")),
    ("2026-08-15c", ["R16"], dict(sect="§3.1")),
    # ---- 3.2 October, r8/r9, before the actuator was named ----
    ("2026-10-06a", ["R17"], dict(sect="§3.2; §4.10")),
    ("2026-10-06b", ["R18"], dict(sect="§1; §3.2")),
    ("2026-10-06c", ["W02"], dict(sect="§3.2; §4.2")),
    ("2026-10-07a", ["R19"], dict(sect="§3.2")),
    ("2026-10-07b", ["R20"], dict(sect="§3.2; §5")),
    ("2026-10-10", ["W01"], dict(sect="§3.2; §6.1")),
    ("2026-10-12", ["R21"], dict(sect="§3.2")),
    ("2026-10-15", ["R22"], dict(sect="§3.2; §4.2")),
    ("2026-10-17", ["R23"], dict(sect="§3.2; §4.10")),
    # ---- 3.3 r9: the rails named ----
    ("2026-10-19a", ["R24", "S00"], dict(sect="§3.3", date="2026-10-19 (runs of 2026-10-18)",
        disagree="The run log's 2026-10-20 line 82 says Run 2's habit 'recovered by 6.3M there'; later sources show it came back (28/40 judged saturation failures at 9.96M, 6/40 at 11.53M; last flagged evaluation 11,272,192; 44 of 91 flagged). The l82 wording fits only the 6.3M J checkpoint (session sweep note 1).")),
    ("2026-10-19b", ["R25"], dict(sect="§3.3", date="2026-10-19 to 2026-10-21",
        disagree="Run 2's final tail_3 command is +0.841 in satdiag's 20-episode table and +0.851 in the zero-shot 40-episode panel (stance6/evidence.txt lines 404-405 notes this) (session sweep note 2).")),
    ("2026-10-19c", ["R26"], dict(sect="§3.3; §4.3",
        disagree="This readout says 'tail_1 and tail_2 pitch rest on their ventral stops ... in every run and in the statue'; the corrected 2026-10-25 records say only tail_1 did, tail_2 at -0.2029 'just above its own' (repo sweep note 2; session sweep note 12).")),
    ("2026-10-19d", ["R27", "S02"], dict(sect="§1; §3.3")),
    ("2026-10-19e", ["R28"], dict(sect="§6.2")),
    ("2026-10-20a", ["R29", "S01"], dict(sect="§3.3",
        disagree="KNOWN_ISSUES Update 12 and §19.5 say the saturating actuator was 'not recorded' / 'not known'; satdiag named tail_3 the next day (Update 14) (repo sweep note 6). The run log (l220, l228) says 'ledger flags from 1,835,008'; Table A has 1,835,008 as the first saturated episode and 2,097,152 as the first flagged evaluation (wording, not data; session sweep note 3).")),
    ("2026-10-20b", ["S03"], dict(sect="§6.2")),
    ("2026-10-20c", ["R30", "S04"], dict(sect="§3.3")),
    ("2026-10-21a", ["R31", "S05"], dict(sect="§3.3")),
    ("2026-10-21b", ["R32"], dict(sect="§3.3; §4.9")),
    ("2026-10-21c", ["R33"], dict(sect="§3.3; §4.7")),
    ("2026-10-21d", ["R34"], dict(sect="§3.3; §4.2")),
    ("2026-10-21e", ["R35"], dict(sect="§3.3; §4.3")),
    ("2026-10-21f", ["R36"], dict(sect="§3.3; §4.1",
        disagree="Satdiag §6 and fix Y1 blame a passive spring stiffer than the servo and propose lowering the stiffness; brief_v2 finding F7 says at q3 = 0 the spring adds nothing, gravity is 13.48 N.m against a servo maximum of 12.57 N.m, and softening the spring would not help. The repository keeps both texts (repo sweep note 3).")),
    ("2026-10-21g", ["R37"], dict(sect="§3.3; §4.9")),
    ("2026-10-21h", ["R38", "S06"], dict(sect="§3.3; §5")),
    ("2026-10-21i", ["R39", "S07"], dict(sect="§6.1")),
    ("2026-10-21j", ["R40"], dict(sect="§1; §6.1")),
    ("2026-10-21k", ["R41", "S08", "S10", "S11"], dict(sect="§3.3", date="2026-10-21 to 2026-10-22",
        disagree="The run log's l335 prints 21 saturation-rail episodes at stance5 s42's final; l365-366 and §21.4 correct it to 20 (session sweep note 5; repo sweep note 6).")),
    ("2026-10-21l", ["S09"], dict(sect="§3.3")),
    ("2026-10-22a", ["S12"], dict(sect="§3.3")),
    ("2026-10-22b", ["R42", "S13"], dict(sect="§3.3")),
    ("2026-10-22c", ["R43", "S14", "S15"], dict(sect="§3.3; §5",
        disagree="The P1 report's 'Findings (as returned)' (line 140) says P1 kept the mean from parking beyond +1; its own skeptic check 1 refutes that, and the file keeps both (repo sweep note 6).")),
    ("2026-10-22d", ["R44"], dict(sect="§4.5; §6.1")),
    ("2026-10-22e", ["R45", "S18"], dict(sect="§3.3")),
    ("2026-10-22f", ["R46", "S53"], dict(sect="§6.1")),
    ("2026-10-22g", ["R47", "S17"], dict(sect="§6.2")),
    ("2026-10-23a", ["R48", "S19", "S22"], dict(sect="§3.3", date="2026-10-22 to 2026-10-23")),
    ("2026-10-23b", ["S20"], dict(sect="§6.2")),
    # ---- 3.4 the tail-habit round ----
    ("2026-10-24a", ["R49", "S21", "S23"], dict(sect="§3.4")),
    ("2026-10-24b", ["R53", "S24"], dict(sect="§1; §3.4", date="2026-10-23 to 2026-10-24")),
    ("2026-10-24c", ["R50", "S25"], dict(sect="§1; §3.4")),
    ("2026-10-24d", ["R51", "S26"], dict(sect="§1; §3.4")),
    ("2026-10-24e", ["R52", "S27"], dict(sect="§1; §3.4")),
    ("2026-10-24f", ["R55", "S28", "S29", "S30"], dict(sect="§3.4; §4.1; §4.3")),
    ("2026-10-24g", ["R56", "S32"], dict(sect="§3.4; §4.7",
        disagree="Loop-gain estimates differ by method and checkpoint and nothing reconciles them: about 0.72 (brief_v2, from satdiag's observation swap and the plant gain); about 0.66 (hold cells, decisions_v1, screen criteria); in-episode dmu/da 0.5-0.6 (unclipped probe); about 0.63 at the s42 ECS ends and 0.27 at their start (clamp verifier) (session sweep note 6).")),
    ("2026-10-24h", ["R57", "S16"], dict(sect="§3.4; §4.7", date="2026-10-22 (named 2026-10-24)")),
    ("2026-10-24i", ["R58", "S33"], dict(sect="§3.4; §4.9",
        disagree="The count of affected runs grew: satdiag 2 of 5 L+T runs (its verifier corrected an earlier 2 of 6), §21.5 3 of 8, §23.4 4 of 10; the draft options brief said 4 of 11, corrected to 4 of 10 by finding F5 (repo sweep note 5). brief_v2's 'every recorded onset came before 4.5M' holds for r9 only: on r10 s60's drive began at 7,340,032 and its first over-bar episode came at 8,126,464 (session sweep note 11).")),
    ("2026-10-24j", ["R59", "S34"], dict(sect="§3.4",
        disagree="The draft brief's case for 0.90 (P1 'overshot' its bound by up to 0.033) was labelled measured; finding F2: P1's window maximum 1.022-1.033 was the start state's own 1.028 (repo sweep note 6).")),
    ("2026-10-24k", ["R60", "S38"], dict(sect="§3.4; §5")),
    ("2026-10-24l", ["S36"], dict(sect="§5")),
    ("2026-10-24m", ["S37"], dict(sect="§5")),
    ("2026-10-24n", ["R63", "S39", "S44"], dict(sect="§3.4",
        disagree="Compute estimates for the screen: brief_v2 about 10-20 h, decisions_v1 about 4 h, the final criteria about 5.5 h; the D-D43 decision 'about 6 h'; the actual wall time about 4.6 h (session sweep note 10). s80's start command: +0.6285 (SE 0.048) in §24.3, 0.628516 in §24.4; NEXT_STEPS (Option 1 rationale) gives '+0.602, 0 flags [EVID l220]', unresolved (repo sweep note 4).")),
    ("2026-10-24o", ["R54", "S40"], dict(sect="§3.4",
        disagree="The coordinator's first reading (run log l516) had s42's bank above 1.0 at 3,530,752, just before the first flag, and s80 flagged '4,325,376-8.4M'; the checked probe (l528) shows the bank values are on ankle-minus rows and do not lead, and s80 was flagged to 10,223,616 (session sweep note 4).")),
    ("2026-10-24p", ["R61", "S41"], dict(sect="§3.4; §4.3")),
    ("2026-10-24q", ["R62", "S42"], dict(sect="§3.4; §4.7")),
    ("2026-10-24r", ["S43"], dict(sect="§3.4; §4.7")),
    ("2026-10-24s", ["R64", "S45"], dict(sect="§3.4")),
    ("2026-10-24t", ["S46"], dict(sect="§6.2")),
    ("2026-10-24u", ["S47"], dict(sect="§6.2")),
    ("2026-10-24v", ["R65", "S35"], dict(sect="§6.2")),
    ("2026-10-25a", ["R66", "S48"], dict(sect="§3.4; §5", date="2026-10-24 (scored 23:53 UTC)",
        disagree="The D-D43 criteria predicted BOUND90 would read PARKED at s80; it read RAIL (repo sweep note 6).")),
    ("2026-10-25b", ["S49"], dict(sect="§5")),
    ("2026-10-25c", ["S50"], dict(sect="§5")),
    ("2026-10-25d", ["S51"], dict(sect="§3.4")),
    ("2026-10-25e", ["R67", "S52"], dict(sect="§3.4")),
    ("2026-10-25f", ["R68", "S54"], dict(sect="§3.4")),
    ("2026-10-25g", ["R69", "S55"], dict(sect="§3.4; §4.10")),
    ("2026-10-25h", ["S56"], dict(sect="§3.4; §7")),
    ("2026-10-25i", ["S57"], dict(sect="§7")),
    ("2026-10-25j", ["R70"], dict(sect="§3.5")),
    ("2026-10-25k", ["R71", "S58"], dict(sect="§3.5; §4.10")),
    ("2026-10-25l", ["R72", "S59"], dict(sect="§3.5; §4.10")),
    ("2026-10-25m", ["R73", "S60"], dict(sect="§3.5; §5",
        disagree="Earlier records (CHANGELOG #504, the stance3 readout, satdiag §3) say tail_1 and tail_2 rest on their ventral stops on r9; the D-D44 measurement shows only tail_1 does (-0.21069 rad, 30.29 N.m), tail_2 at -0.2029 'just above its own' (repo sweep note 2; session sweep note 12).")),
    ("2026-10-25n", ["S61"], dict(sect="§4.10")),
    ("2026-10-25o", ["R74"], dict(sect="§3.5")),
    ("2026-10-25p", ["R75"], dict(sect="§6.1")),
    ("2026-10-25q", ["S83"], dict(sect="§3.5", date="2026-10-21 to 2026-10-25")),
    # ---- 3.5 r10 ----
    ("2026-10-26a", ["R76", "S63"], dict(sect="§1; §3.5", date="2026-10-25 to 2026-10-26")),
    ("2026-10-26b", ["R77", "S62"], dict(sect="§3.5")),
    ("2026-10-26c", ["R78"], dict(sect="§3.5")),
    ("2026-10-27a", ["R79", "S64"], dict(sect="§1; §3.5")),
    ("2026-10-27b", ["R80", "S65", "S66"], dict(sect="§3.5", date="2026-10-27 to 2026-10-28",
        disagree="Snapshot 1's s42 tail_3 command start: run log l681 says 'about 0.00 at 1.31M to +0.394'; READOUTS l835-836 says '(0.202 -> 0.394)'. The baseline evaluation differs and was not resolved (session sweep note 8).")),
    ("2026-10-28a", ["R81", "S67"], dict(sect="§3.5")),
    ("2026-10-28b", ["R82", "S68"], dict(sect="§3.5")),
    ("2026-10-28c", ["R83", "S69"], dict(sect="§3.5")),
    ("2026-10-28d", ["R84", "S70"], dict(sect="§3.5; §4.3",
        disagree="The value of holding tail_3 at 0 is positive at the r9 checkpoints (+73.1 to +85.9; +54.6 / +36.2 under noise), at r10 s42 3,670,016 (+78.29) and at the s42 ECS ends (+63.6), but negative at the r10 s42 final (-64.46, 2 of 20 positive), where there was no trace rail, only a parked command. Different checkpoints, not a direct contradiction; the sign flip is unexplained (session sweep note 7).")),
    ("2026-10-28e", ["R85", "S71", "T39"], dict(sect="§4.8")),
    ("2026-10-28f", ["R86", "S72"], dict(sect="§4.6",
        disagree="READOUTS.txt lines 802-805's entropy wording was narrowed by REVIEW §4.3 (repo sweep note 6).")),
    ("2026-10-28g", ["R87", "S73", "S74"], dict(sect="§3.5; §4.9")),
    ("2026-10-28h", ["R88"], dict(sect="§3.5")),
    ("2026-10-28i", ["S75"], dict(sect="§6.2")),
    ("2026-10-28j", ["S76"], dict(sect="§6.2")),
    ("2026-10-28k", ["S84"], dict(sect="§3.5")),
    ("2026-10-28l", ["R90"], dict(sect="§6.1")),
    ("2026-10-28m", ["R89", "S77", "T00"], dict(sect="§3.5; §7",
        disagree="NEXT_STEPS Option 1 (lines 169-178) proposed an s80 start at 3,145,728 with 4 continuations per arm; the approved screen used s80 at 4,194,304 with 6 per arm (a decision change, not an error; session sweep note 9).")),
    # ---- 3.6 2026-10-28/29: ECS, CoM revisit, clamp, normaliser probe ----
    ("2026-10-28n", ["T01"], dict(sect="§3.6; §4.6")),
    ("2026-10-28o", ["T02"], dict(sect="§3.6")),
    ("2026-10-28p", ["T03"], dict(sect="§3.6",
        disagree="FOR_THE_MAINTAINER.txt is draft v2 (not approved); the approved text is final/criteria_ecs_final.txt = criteria.txt (v3 with PINS). Its quoted power and cost figures match the final criteria (today sweep note l).")),
    ("2026-10-28q", ["T04"], dict(sect="§3.6; §7")),
    ("2026-10-29a", ["T05"], dict(sect="§3.6")),
    ("2026-10-29b", ["T06"], dict(sect="§3.6")),
    ("2026-10-29c", ["T07"], dict(sect="§3.6",
        disagree="Run log l738 says 'the last wave (6 continuations, 2 waves of 3) running'; 3 were running and 3 queued. Corrected at l742 (today sweep note h).")),
    ("2026-10-29d", ["T08", "S78"], dict(sect="§3.6",
        disagree="Run log l740 omits the reading line's pre-registered parenthesis, corrected at l742 (today sweep note h). The ANY-flag definition differs in wording: THIS12 l310-311 also lets a saturation count of at least 5 in the same fetch flag an evaluation; ECS R1 and the r10 criteria (line 169) use ledger rows only; no effect on the reading (today sweep note k).")),
    ("2026-10-29e", ["T09"], dict(sect="§3.6")),
    ("2026-10-29f", ["T10"], dict(sect="§3.6")),
    ("2026-10-29g", ["T11"], dict(sect="§3.6")),
    ("2026-10-29h", ["T12"], dict(sect="§3.6")),
    ("2026-10-29i", ["T13"], dict(sect="§3.6")),
    ("2026-10-29j", ["T14"], dict(sect="§3.6; §4.6")),
    ("2026-10-29k", ["T15"], dict(sect="§6.1",
        disagree="ECS arm-A rate: inferred 0.75-0.95 before the run (criteria l252-254, from r9's 3 of 3); observed 6 of 6 at s42 but 1 of 6 at s80, and that on the neck (today sweep note j).")),
    ("2026-10-29l", ["T16"], dict(sect="§3.6")),
    ("2026-10-29m", ["T19"], dict(sect="§3.6")),
    ("2026-10-29n", ["T17", "S79"], dict(sect="§3.6; §4.2",
        disagree="The static check is at the +-12 deg joint limits (+0.19/+0.24 mm fore-aft, +-2.03 mm vertical); the rail itself reaches only +5.2 deg (+0.092 mm, +1.561 mm kinematic; +0.2 mm, +1.6 mm in replay). Different angles, not a conflict, but the static figures overstate the rail's shift (today sweep note i).")),
    ("2026-10-29o", ["T23"], dict(sect="§3.6")),
    ("2026-10-29p", ["T20"], dict(sect="§3.6")),
    ("2026-10-29q", ["T21", "S80"], dict(sect="§3.6; §4.2",
        disagree="The policies lens said every difference was already in the start checkpoints (policies NOTES l137); verifier POL-4: the fore-aft gap halved (13.465 -> 6.710 mm) because s42 moved 5.0 mm back (today sweep note d).")),
    ("2026-10-29r", ["T22"], dict(sect="§3.6; §4.10",
        disagree="The policies lens explained r9's lean by r9 lacking the lean bonus and mirror PPO (policies NOTES l148-150); verifier POL-12 refuted it: r9 was trained with both and does not lean on its own plant (today sweep note e). This sweep did not check that the r9 -> r10 plant change is only the preload (today sweep unresolved item 6).")),
    ("2026-10-29s", ["T18"], dict(sect="§4.2")),
    ("2026-10-29t", ["T24", "S81"], dict(sect="§3.6")),
    ("2026-10-29u", ["T25"], dict(sect="§3.6")),
    ("2026-10-29v", ["T26"], dict(sect="§3.6; §4.1",
        disagree="Why +1 gives +5.2 deg: the runner says the servo lifts the joint 'against gravity and the spring' (clamp NOTES l76-78); the verifier says it is the servo-spring stiffness split with gravity cancelled by the preload (verify_clamp NOTES l71-78, l132); REPORT l100-101 uses the verifier's (today sweep note b).")),
    ("2026-10-29w", ["T27"], dict(sect="§3.6; §4.2")),
    ("2026-10-29x", ["T28"], dict(sect="§3.6; §4.3; §4.4")),
    ("2026-10-29y", ["T29"], dict(sect="§3.6; §4.2")),
    ("2026-10-29z", ["T30"], dict(sect="§3.6; §4.2",
        disagree="Head-clearance losers under the imposed rail: clamp NOTES l199 says two (s80_B_1411 -312, s80_B_1451 -332); verifier F8 says five; REPORT l110-111 uses five (today sweep note a).")),
    ("2026-10-29za", ["T31"], dict(sect="§3.6; §4.2")),
    ("2026-10-29zb", ["T32"], dict(sect="§3.6; §4.7",
        disagree="The policies' own tail_3 command with the tail held: 0.35 (rail_onset.txt bin means; REPORT l116), 0.33 [0.19, 0.53] (median of per-policy medians, clamp NOTES l182), 0.29-0.33 (verifier subset); forced up in s80: -0.27 (median of medians) vs -0.29 (episode mean). Aggregation differences, not conflicts (today sweep note f).")),
    ("2026-10-29zc", ["T33"], dict(sect="§3.6; §4.7")),
    ("2026-10-29zd", ["T34"], dict(sect="§4.7")),
    ("2026-10-29ze", ["T35"], dict(sect="§3.6")),
    ("2026-10-29zf", ["T36"], dict(sect="§3.6; §4.2",
        disagree="The runner reads the 5 mm drift as gone with the tail held at 0 in both (-0.72 mm; clamp NOTES l283-284); the verifier shows -7.65 mm with both railed, so the drift sits in the tail-up state (verify_clamp NOTES l27-32, F7); REPORT l120-122 uses the verifier's (today sweep note c).")),
    ("2026-10-29zg", ["T37"], dict(sect="§6.1")),
    ("2026-10-29zh", ["T38"], dict(sect="§6.2")),
    ("2026-10-29zi", ["T40", "S82"], dict(sect="§3.6; §7")),
    ("2026-10-29zj", ["T41"], dict(sect="§3.6; §4.8; §7", status="pending")),
    ("2026-10-29zk", ["T42"], dict(sect="§3.6")),
]


def norm_src(s):
    s = re.sub(r"\bS1/", "scratchpad/", s)
    s = re.sub(r"\bSP/", "scratchpad/", s)
    s = re.sub(r"\bS2/", "scratchpad2/", s)
    s = s.replace("/home/user/mesozoic-labs/", "")
    return s


def main():
    E = load()
    E.update(W)
    used = []
    rows = []
    for n, (key, ids, ov) in enumerate(sorted(G, key=lambda g: g[0]), start=1):
        p = E[ids[0]]
        used.extend(ids)
        srcs = []
        for i in ids:
            for s in E[i]["sources"]:
                s2 = norm_src(s)
                if s2 not in srcs:
                    srcs.append(s2)
        result = p["result"]
        for i in ids[1:]:
            result += " || [also, " + i + "] " + E[i]["result"]
        rows.append({
            "id": "L%03d" % n,
            "date": ov.get("date", p["date"]),
            "kind": p["kind"],
            "title": ov.get("title", p["title"]),
            "verdict": ov.get("status", p["status"]),
            "label": p["label"],
            "what": p["what"],
            "result": norm_src(result),
            "sources": " ; ".join(srcs),
            "merged_from": ";".join(ids),
            "where_in_record": ov.get("sect", ""),
            "sources_disagree": ov.get("disagree", ""),
        })
    missing = sorted(set(E) - set(used))
    dup = sorted({i for i in used if used.count(i) > 1})
    assert not missing, missing
    assert not dup, dup
    cols = ["id", "date", "kind", "title", "verdict", "label", "what", "result", "sources", "merged_from",
            "where_in_record", "sources_disagree"]
    with open(os.path.join(OUT, "ledger.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols, quoting=csv.QUOTE_ALL)
        w.writeheader()
        for r in rows:
            w.writerow(r)
    idx = {r["id"]: {"merged_from": r["merged_from"], "date": r["date"], "title": r["title"],
                     "where": r["where_in_record"]} for r in rows}
    json.dump(idx, open(os.path.join(HERE, "ledger_index.json"), "w"), indent=1)
    print(len(E), "entries in;", len(rows), "ledger rows out")
    for r in rows:
        print(r["id"], r["merged_from"], "|", r["date"], "|", r["kind"], "|", r["verdict"], "|", r["where_in_record"], "|", r["title"][:90])


if __name__ == "__main__":
    main()
