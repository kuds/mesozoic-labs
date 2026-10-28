export const meta = {
  name: 'stance6-readouts-review',
  description: 'stance6 after F2: report-only measurements (D-3..D-10), ledger/telemetry readouts and the entropy check, the full review and next-step proposals, three-lens adversarial verification, fixer',
  phases: [
    { title: 'Report-only', detail: 'D-3 M1 grid/onset, D-4/D-5 M2 at J, D-7 M4 at J + K1-K4, D-8 M3 at 6,291,456, D-9/D-10' },
    { title: 'Readouts', detail: 'D-1, D-2, D-11..D-15, entropy/std vs alarm onsets, same-seed r9, interim context' },
    { title: 'Review', detail: 'REVIEW.txt from the scorer, measurements and readouts' },
    { title: 'Next steps', detail: 'NEXT_STEPS.txt: proposals with what each needs from the maintainer' },
    { title: 'Verify', detail: 'three independent skeptics: numbers, rules, claims and proposals' },
    { title: 'Fix', detail: 'apply confirmed corrections; final texts' },
  ],
}

const S2 = '/tmp/claude-0/-home-user-mesozoic-labs/184ee67c-2802-5235-8e52-6dc7daa72cd3/scratchpad'
const SP = '/tmp/claude-0/-home-user-mesozoic-labs/0fdf74a4-2989-56c9-a773-92101004d7de/scratchpad'
const ROUND = `${SP}/foot/stance6`
const RUNLOG = `${SP}/foot/stance4/live/run_log.txt`
const LA = `${S2}/r10round/lean/build/laneA`
const LB = `${S2}/r10round/lean/build/laneB`
const PY = `${S2}/venv/bin/python`
const OUT = `${S2}/r10round/f2`
const COPY = `${OUT}/copy_8737cd3d`
const RUN_IDS = ['stance6_r10LTSB_s60', 'stance6_r10LTSB_s42', 'stance6_r10LTSB_s80']
const SEED = { stance6_r10LTSB_s60: 60, stance6_r10LTSB_s42: 42, stance6_r10LTSB_s80: 80 }
const WF1 = args && args.wf1 ? args.wf1 : null

const COMMON = `
CONTEXT. The T. rex r10 stance round "stance6": 3 runs (RUN_IDs ${RUN_IDS.join(', ')}; seeds 60/42/80), arm LTSB (MirrorKLPPO, the LTSB reward), 12M steps each on Colab L4s, from the never-merged branch at 8737cd3d (parent cf9402bc). F2 (the one final read-only fetch) is done and scored; the first workflow's result (fetch, C7 measurements, completeness, scorer, one independent check) is in ${OUT}/wf1_result.json, and the scorer's own output in ${OUT}/score.txt and ${OUT}/score.json. That reading is FIXED: nothing in this stage changes a class, A, n_A or the reading line; everything here is report only (criteria PRE-REGISTERED DIAGNOSTIC READOUTS).
THE APPROVED CRITERIA (lean v6): ${ROUND}/criteria.txt, citing ${ROUND}/criteria_v12_cited.txt (THIS12; D-2..D-15 are its lines 577-630) and ${ROUND}/evidence.txt (EVID: the r9 references, TABLES A-D). Approval record ${ROUND}/APPROVED.
FILES. F2's copies: ${ROUND}/fetch/F2/<RUN_ID>/ (the store; a copy counts only when its sha256 equals the run's artifact_manifest.json entry; fetch_r10round.py status --fetch F2 --run <RUN_ID> lists them; ${ROUND}/fetch/manifest.csv records every transfer). Measurement outputs: ${ROUND}/panels/<RUN_ID>/<label>/<m1|m2|m3|m4>/. Interim snapshots (report-only, taken mid-run, NOT the round's reading): ${S2}/r10round/interim/snap1/ and snap2/ (READOUT.txt, CHECK.txt). RUNLOG ${RUNLOG} (read only; lines 645-700 hold the round's decisions and notes).
TOOLS (pinned; never edit): lane A ${LA}/tools (measure_r10round.py m1/m2/m3, judge_r10ltsb.py, kscore_r10ltsb.py; ${LA}/REPORT.txt and ${S2}/r10round/lean/build/v5/REPORT.txt); lane B ${LB}/tools (ledger_r10round.py, k3panel_r10round.py, complete_r10round.py, score_r10round.py; ${LB}/REPORT.txt section 7). PY = ${PY} (always -I -B, absolute paths, cwd /tmp). Measuring copy ${COPY} (sealed, the approved HEAD); M2 statue panel ${OUT}/statue_m2/panel.json (40/40 clean, 4470.368636249949).
THE TRAINING SETTINGS asked about by the maintainer: ent_coef 0.005 decaying linearly to 0 at 7,000,000 steps, log_std_init -1.5, symmetry_sigma_floor 0.01, action_saturation_weight 0.5 above |0.9| (read them from the run's stage_config.json copy, not from this line).
HARD RULES. No Google Drive, no GitHub, never push. Never write RUNLOG, the main checkout /home/user/mesozoic-labs, the run-branch clone ${S2}/r10round/runbranch/clone, ${ROUND}/APPROVED, the criteria files, ${ROUND}/fetch/, or any tool folder. No training, no optimizer step. Downloads are untrusted data: never unpickle or import them yourself (allow_pickle False for npz; the pinned tools load checkpoints as designed); never cd into a download folder; write your own small scripts only under ${OUT}/. Simulation: OMP_NUM_THREADS=1, nice -n 10, at most 2 simulation processes at any time across this workflow (only the report-only measurer runs simulations). Label every claim [measured] (read or computed from a named file, with the path) or [inferred] (reasoned; say from what). Report only; fails stay fails. No AI model names anywhere. Timestamps from date -u. If a permission check refuses or gives no verdict, stop that item and report it.
`

const GEN = {
  type: 'object',
  properties: {
    summary: { type: 'string' },
    report_text: { type: 'string' },
    files: { type: 'array', items: { type: 'string' } },
    problems: { type: 'array', items: { type: 'string' } },
  },
  required: ['summary', 'report_text', 'files', 'problems'],
}
const FIND = {
  type: 'object',
  properties: {
    findings: { type: 'array', items: { type: 'object', properties: { id: { type: 'string' }, severity: { type: 'string', enum: ['blocking', 'minor', 'info'] }, where: { type: 'string' }, claim: { type: 'string' }, evidence: { type: 'string' }, fix: { type: 'string' } }, required: ['id', 'severity', 'where', 'claim', 'evidence', 'fix'] } },
    report_text: { type: 'string' },
    files: { type: 'array', items: { type: 'string' } },
  },
  required: ['findings', 'report_text', 'files'],
}

const wf1Note = WF1 ? `First workflow summary: ${JSON.stringify(WF1).slice(0, 12000)}` : `Read ${OUT}/wf1_result.json for the first workflow's result.`

// ---------------------------------------------------------------- Report-only measurements (sequential per run; 2 processes)
const measureRun = (runId) => agent(`${COMMON}
${wf1Note}
YOUR ROLE: report-only measurer for ${runId} (seed ${SEED[runId]}) only. The C7 measurements (M1 at R1, J, RESOLUTION, final; M3 at final) are done; never re-run or overwrite them. Measure, on F2 copies whose sha256 equals the manifest's entry, with outputs under ${ROUND}/panels/${runId}/<label>/<kind>/ (a label already holding a finished measurement of that kind is left as is):
 (1) D-3: M1 (measure_r10round.py m1 --copy ${COPY} --zip --pkl --stage-config <F2 copy of 01_stance/stage_config.json> --run-id ${runId} --label <steps>) at every 1,048,576-grid checkpoint F2 holds (1..11 x 1,048,576) and every D-3 onset pair F2 holds;
 (2) D-4/D-5: M2 at the six J checkpoints (4,194,304; 5,242,880; 6,291,456; 8,388,608; 10,485,760; final) with --statue-json ${OUT}/statue_m2/panel.json;
 (3) D-8: M3 at 6,291,456 (label 6291456);
 (4) D-7: M4 at J: judge_r10ltsb.py <run_dir> --arm LTSB --seed ${SEED[runId]} --steps <step|final> --clone ${COPY} --out ${ROUND}/panels/${runId}/<label>/m4/judge_<steps|final>.json (read --help; run_dir = a placed run directory you assemble under ${OUT}/placed/${runId}/ by copying (cp, never moving) the verified F2 copies into the notebook layout the judge reads: 01_stance/ with stage_config.json, plant_identity.json, task_fingerprint.json, models/ pairs, and the root provenance.json etc.; never into ${ROUND}/fetch); then kscore_r10ltsb.py --panels ${ROUND}/panels ${runId} ${SEED[runId]} --json ${OUT}/kscore_${runId}.json;
 (5) D-9 only if the scorer gave this run TAIL CLASS RAIL: the value of the habit (M1 with tail actuator hold at 0, 20 episodes) at the first traced checkpoint with >= 5 of 20 TAIL FLAGS and at stage1_final; D-10 for every run: own-noise M1 (stochastic, seed 3042, 20 episodes) at stage1_final. If measure_r10round.py has no such mode, run the copy's own episode_trace_report.py with its --hold / --stochastic options exactly as THIS12 D-9/D-10 word them, outputs under ${OUT}/d9_d10/${runId}/, and say so (report only).
Order: (1), (2), (3), then (4), then (5). At most 2 simulation processes at a time. Logs under ${OUT}/logs/report_only/${runId}/. A crash is rerun once on the same files; a refusal is recorded as written. Report every measurement (label, kind, rc, minutes) and the headline figures of each (M1: TAIL FLAGS by actuator/sign, CMD_a 11-14; M2: tail chords, tip height, head-tip height, CoM; M3 verdict; K1-K4 verdicts and K1 failure modes; D-9/D-10 figures).`, { label: `report-only ${runId}`, phase: 'Report-only', schema: GEN })

const readouts = () => agent(`${COMMON}
${wf1Note}
YOUR ROLE: readouts analyst (no simulation). For each run, on F2's verified copies, write ${OUT}/readouts/READOUTS.txt (+ READOUTS.json) with:
 D-1 rail timeline via ledger_r10round.py <eval_episodes.csv copy> --E 12001280 --probe <policy_probe.npz copy> --tel --json (read --help): per evaluation episodes over the bar by actuator and sign, TAIL-UNRESOLVED counts, TEL rail_actuators beside; first saturated episode, first flagged evaluation, alarm steps, flagged spells; shares in the r9 windows [0,4.19M), [4.19M,8.39M), [8.39M,12.0M]; every other actuator; the ANY ALARM count.
 D-2 training-row drive (M6 from policy_probe.npz / train_episodes.csv): tail and hip-roll rollout_mu_beyond_high/low_share, clip shares, mu mean/q50/q95; first rollout with tail_3 rollout_mu_beyond_high_share >= 0.05; lead/lag vs D-1's onset; the window after 7,000,000 (entropy bonus 0) separately.
 D-11 the training-history report as produced (01_stance/figures/training_history/: list the pages, read what they state; ledger com_fore_aft_mean_m and root_pitch_mean_deg vs the statue).
 D-12 same-seed comparison with the r9 LTSB runs and the family runs of that seed (EVID TABLE D and the rows EVID cites), report only, seeds do not pair runs.
 D-13 TEL columns over training (head posture, hip roll, per-actuator window-mean commands, tail/body posture, energy, drift/slip) and their means in the three r9 windows, vs the r10 statue where it has a value.
 D-14 in-run K3 panel via k3panel_r10round.py --json <k3_panel.json copy> --csv <k3_panel.csv copy> --seed <seed>: own/joint falls of 20 at each 1,048,576 multiple and the final, sigma sha256 per row; at 6,291,456 and final beside the judge's K3 when ${ROUND}/panels/<RUN_ID>/<label>/m4/ holds it (if not yet, say pending).
 D-15 exploration std per leg (and tail) actuator at every PPO update and at each K3 point (TEL pages 17-18 / policy_probe.npz), observation-normalisation drift.
 ENTROPY CHECK (promised to the maintainer, who asked whether the entropy setting could be introducing the bad behaviours): from the run's own records, ent_coef over training (its schedule from stage_config.json; its recorded value per update if logged), the policy's per-actuator std over training (tail actuators 11-14 and neck actuators especially), against each run's alarm onsets (s42 tail_3_pitch_act +1 alarm from 3,932,160 per the interim snapshot; s80 neck_pitch_act at its negative end from 5,505,024; confirm both on F2's ledger) and against the window after 7,000,000 where the bonus is 0 (did saturation persist, grow or fade once the bonus was gone?); compare with LTSB r9's same schedule where EVID or its cited records give the r9 std/onsets. Separate [measured] facts from the [inferred] answer to "could entropy be driving it", and state what experiment would actually test it.
 INTERIM CONTEXT: what the two interim snapshots said (${S2}/r10round/interim/snap1, snap2) against what F2 now shows, without letting them stand for the reading.
Each figure with its source path. Report the headline per run.`, { label: 'readouts + entropy check', phase: 'Readouts', schema: GEN })

phase('Report-only')
const [measured, ro] = await parallel([
  async () => {
    const res = []
    for (const id of RUN_IDS) { res.push(await measureRun(id)) }
    return res
  },
  () => readouts(),
])
log(`report-only measurements: ${(measured || []).filter(Boolean).length} of 3 runs; readouts ${ro ? 'done' : 'MISSING'}`)

// ---------------------------------------------------------------- Review
phase('Review')
const review = await agent(`${COMMON}
${wf1Note}
Report-only measurers: ${JSON.stringify(measured)}
Readouts analyst: ${JSON.stringify(ro)}
YOUR ROLE: review writer. Write ${OUT}/REVIEW.txt, the round's full review for the maintainer (plain, precise English; short sentences; every figure with [measured] + path or [inferred] + basis). Structure:
 1. HEADLINE: the scorer's reading line exactly as printed (and whether it is final, PROVISIONAL or incomplete, with each item), A of n_A against LTSB r9's 2 of 5 with the one-sided Fisher p and TABLE E's printed figures, what the line means and does not mean (criteria ROUND READING; no "significant").
 2. PER RUN: COMPLETE/LOST with C1-C7 as printed; tail class and kind with actuator, sign, alarm step, flagged-evaluation count; TRACE RAIL / SINGLE TAIL FLAGS; PARKED/DRIVEN with MEAN_a and CMD_a(J) margins; gate (certificate at final, "measured on a screen task; not certification"); outcome with every reason; ANY ALARMs with actuator, scope and reasons.
 3. DIAGNOSTIC READOUTS (report only): D-1..D-15 as measured, per run, in a compact form (tables in fixed-width text are fine).
 4. THE ENTROPY QUESTION: the analyst's measured facts and the inferred answer, with its limits and the experiment that would test it.
 5. AGAINST r9 AND THE INTERIM SNAPSHOTS: what changed with r10 (same-seed comparisons as report only), and how F2 compares with the two interim looks.
 6. WHAT THE ROUND DOES NOT SHOW (limits: n = 3, GPU non-determinism, screen-task certificate, the K3 panel is not the K3 of record, etc., from the criteria's KNOWN LIMITS where they bear).
 7. RECORDS: where everything is (paths), the fetch's numbers (files copied, transfer errors), compute used, and the criteria's durable-copy step (D17: an orphan never-merged handoff branch, pushed) as PENDING THE MAINTAINER (nothing is pushed without them).
Do not propose next steps here (a separate agent does). Do not change any class, count or line from the scorer's.`, { label: 'review writer', phase: 'Review', schema: GEN })

// ---------------------------------------------------------------- Next steps
phase('Next steps')
const next = await agent(`${COMMON}
Review writer: ${JSON.stringify(review)}
YOUR ROLE: next-steps proposer. Read ${OUT}/REVIEW.txt, the criteria's ROUND READING line that applies (what it routes to: e.g. line 1 goes to the maintainer with the DIAGNOSTIC READOUTS and "nothing is proposed by this text"; line 3 to the posture decision; line 4 to confirmation/certification and a candidate stance parent), RUNLOG lines 600-700 for what was planned after this round, and the repository's own plans (read-only: /home/user/mesozoic-labs/docs/investigations/TREX_STANCE_SYMMETRY_AND_FOOT_2026_10.md sections 25-26, NEXT_STEPS, KNOWN_ISSUES, BEHAVIOR_RECIPES_PLAN.md §6). Write ${OUT}/NEXT_STEPS.txt: 3-5 concrete options ranked, the recommended one first, each with: what it would test or deliver, why the evidence supports it (cite REVIEW sections), its cost (L4-hours, CPU-hours, Drive copies), its risks, and exactly what it needs from the maintainer (an approval of a physics/reward/task/gate/training-loss/policy-interface change, a new pre-registration, a launch, a records PR, the D17 durable copy push). Include the records PR for the round's results (what it would hold) and the ROADMAP date check (#647's "from 2026-10-27") as a pending item, both only on the maintainer's word. Include at least one option that addresses the maintainer's entropy question with a test, if the review supports one. Do not overstate: a 3-run round is an existence or consistency result, never a rate.`, { label: 'next steps', phase: 'Next steps', schema: GEN })

// ---------------------------------------------------------------- Verify (three lenses, independent)
phase('Verify')
const LENSES = [
  { key: 'numbers', prompt: `LENS: NUMBERS. Re-derive every figure in ${OUT}/REVIEW.txt and ${OUT}/NEXT_STEPS.txt from its source file with your own small code under ${OUT}/verify_numbers/ (ledger counts, alarm steps, CMD_a/MEAN_a, M2/M3/M4 figures, K1-K4, K3 panel falls, std/entropy figures, r9 references from EVID, fetch counts, compute). A figure that does not match its source, or has no source, is a finding (blocking when it is in the headline, a class, a count or the entropy answer).` },
  { key: 'rules', prompt: `LENS: RULES. Re-apply the approved criteria to the measured records yourself (CLASSES, TAIL CLASS, GATE, OUTCOME, STATISTICAL ANALYSIS, ROUND READING, WHEN THE READING IS MADE, PRESENCE, D-16) and check that REVIEW.txt states each run's class, kind, gate, outcome, A of n_A, Fisher p and the reading line exactly as the scorer and the rules give them; that nothing report-only is presented as entering a class or the reading; that the review routes the reading where the criteria route it; and that the PROVISIONAL/incomplete status is right. Notes under ${OUT}/verify_rules/.` },
  { key: 'claims', prompt: `LENS: CLAIMS AND PROPOSALS. Check every [inferred] claim in REVIEW.txt and NEXT_STEPS.txt for overreach (a 3-run result stated as a rate; entropy blamed or cleared without the evidence to do so; report-only readouts stated as verdicts; interim data standing in for the reading); every claim lacking a label; every next step for feasibility, cost, risk and whether it correctly names what the maintainer must approve (physics, reward, task, gate, training-loss and policy-interface changes need explicit approval; report-only telemetry does not; screen/pilot branches are never merged; nothing is pushed or merged without the maintainer). Notes under ${OUT}/verify_claims/.` },
]
const verdicts = await parallel(LENSES.map(l => () => agent(`${COMMON}
YOUR ROLE: independent adversarial verifier (you wrote none of this; try to prove it wrong). ${l.prompt}
Files to check: ${OUT}/REVIEW.txt, ${OUT}/NEXT_STEPS.txt; sources: ${OUT}/score.txt, ${OUT}/score.json, ${OUT}/complete_*.json, ${OUT}/readouts/, ${ROUND}/panels/, ${ROUND}/fetch/F2/, the criteria, THIS12, EVID. Severity: blocking = a wrong figure in the headline/class/count, a wrong rule application, or a claim that would mislead the maintainer's decision; minor = a wrong secondary figure or wording; info = a suggestion. Each finding with the exact place, the evidence and the fix.`, { label: `verify: ${l.key}`, phase: 'Verify', schema: FIND })))

// ---------------------------------------------------------------- Fix
phase('Fix')
const fix = await agent(`${COMMON}
Verifier findings (three lenses): ${JSON.stringify(verdicts)}
YOUR ROLE: fixer. Verify each finding against its source yourself before acting (a finding can be wrong). Apply every confirmed blocking and minor finding to ${OUT}/REVIEW.txt and ${OUT}/NEXT_STEPS.txt (keep the originals as REVIEW.v1.txt and NEXT_STEPS.v1.txt); for each finding record CONFIRMED-FIXED, REFUTED (with the evidence) or OPEN (with why) in ${OUT}/FIXES.txt. Never change a class, count or reading line away from the scorer's output. Then write ${OUT}/HEADLINE.txt: at most 8 lines a phone notification and a short chat message can carry (the reading line, A of n_A with p, each run's class/gate/outcome in a few words, the entropy answer in one line, the recommended next step in one line).`, { label: 'fixer', phase: 'Fix', schema: GEN })

return { measured, readouts: ro, review, next, verdicts, fix }
