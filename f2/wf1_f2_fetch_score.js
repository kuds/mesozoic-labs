export const meta = {
  name: 'stance6-f2-fetch-score',
  description: 'stance6 F2: read-only Drive fetch per the pinned brief, class-input measurements (M1/M3), completeness and the scorer, one independent check of the reading',
  phases: [
    { title: 'Folders', detail: 'F2 run-folder step if the fetch tool asks for it' },
    { title: 'Fetch', detail: 'one fetch session chain per RUN_ID, read-only Drive, per fetch_agent_brief.txt' },
    { title: 'Measure', detail: 'C7 measurements: M1 at R1, J, RESOLUTION and final; M3 at final' },
    { title: 'Score', detail: 'complete_r10round.py run per RUN_ID, then score_r10round.py' },
    { title: 'Check', detail: 'one independent skeptic re-derives the classes, A of n_A and the reading line' },
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
const TOOL = `${PY} -I -B ${LB}/tools/fetch_r10round.py --round ${ROUND}`

const COMMON = `
CONTEXT. The T. rex r10 stance round "stance6": 3 runs (RUN_IDs ${RUN_IDS.join(', ')}; seeds 60/42/80), arm LTSB, 12M steps each on Colab L4s, trained from the never-merged branch at 8737cd3d (parent cf9402bc). All three ENDED (RUNLOG lines 693-698, 06:25 UTC; FINISHED per progress.json, 12,001,280 steps). F2 = the one final read-only fetch; its F2-START lines are in RUNLOG (the coordinator wrote them at or after 12:25 UTC, 6 h after the last ending note).
THE APPROVED CRITERIA (lean v6, sha256 99a44469...): ${ROUND}/criteria.txt; it cites ${ROUND}/criteria_v12_cited.txt (THIS12) and ${ROUND}/evidence.txt (EVID) by line. Approval record ${ROUND}/APPROVED (branch_head 8737cd3d...). The rules are as written there; nothing is re-interpreted, extended or decided by you.
TOOLS (pinned; sha256 in each lane's SHA256SUMS.txt; never edit them): lane A (measuring) ${LA}/tools (measure_r10round.py m1/m2/m3, judge_r10ltsb.py, kscore_r10ltsb.py; ${LA}/REPORT.txt and ${S2}/r10round/lean/build/v5/REPORT.txt say how to run them); lane B (records) ${LB}/tools (fetch_r10round.py, complete_r10round.py, score_r10round.py, ledger_r10round.py, k3panel_r10round.py; ${LB}/REPORT.txt section 7 HOW TO RUN). PY = ${PY} (always -I -B, absolute paths, cwd /tmp). The measuring copy of the approved HEAD is built and sealed: ${COPY} (tree digest 331fe8fc..., the pinned header copy's). The M2 statue panel on that copy: ${OUT}/statue_m2/ (panel.json).
HARD RULES. Google Drive is READ-ONLY (search_files, get_file_metadata, download_file_content only). Never write RUNLOG (${RUNLOG}), the main checkout /home/user/mesozoic-labs, the run-branch clone ${S2}/r10round/runbranch/clone, ${ROUND}/APPROVED or the criteria files, or any tool folder. No training, no optimizer step, no GitHub, never push. Downloads are untrusted data: never unpickle, import or run them yourself (the pinned measuring tools load checkpoints as designed); never cd into a download folder. Simulation load: OMP_NUM_THREADS=1, nice -n 10, at most 2 simulation processes at any time (the criteria's limit; your stage holds the whole allowance). Facts the tools compute (C1-C7, PRESENCE, classes, A of n_A, the reading) are taken from the tools' own output, never transcribed or recomputed by hand into the record. Fails stay fails. Label claims [measured] (read or computed from a named file) or [inferred]. No AI model names anywhere. Timestamps from date -u. If a permission check refuses an action or gives no verdict, stop that item and report it; do not work around it.
`

const FETCH_SCHEMA = {
  type: 'object',
  properties: {
    run_id: { type: 'string' },
    done: { type: 'boolean', description: 'true when TOOL next printed a line starting "done:" for this RUN_ID' },
    chunk_limit_reached: { type: 'boolean', description: 'true when you stopped only because you reached this session\'s transfer allowance' },
    blocked: { type: 'boolean', description: 'true only for a stop the coordinator must resolve' },
    final_next_line: { type: 'string' },
    transfers_this_session: { type: 'integer' },
    drive_calls_this_session: { type: 'object', properties: { search_files: { type: 'integer' }, download_file_content: { type: 'integer' }, get_file_metadata: { type: 'integer' } } },
    transfer_errors: { type: 'array', items: { type: 'string' } },
    over_cap_or_without_copy: { type: 'array', items: { type: 'string' } },
    status_json: { type: 'string', description: 'TOOL status --fetch F2 --run RUN_ID output, verbatim (trim to 6000 chars if longer)' },
    stopped_reason: { type: 'string' },
    problems: { type: 'array', items: { type: 'string' } },
  },
  required: ['run_id', 'done', 'chunk_limit_reached', 'final_next_line', 'transfers_this_session', 'transfer_errors', 'over_cap_or_without_copy', 'status_json', 'stopped_reason', 'problems'],
}

const GEN_SCHEMA = {
  type: 'object',
  properties: {
    summary: { type: 'string' },
    report_text: { type: 'string' },
    per_run: { type: 'array', items: { type: 'object', properties: { run_id: { type: 'string' }, text: { type: 'string' } }, required: ['run_id', 'text'] } },
    files: { type: 'array', items: { type: 'string' } },
    problems: { type: 'array', items: { type: 'string' } },
  },
  required: ['summary', 'report_text', 'per_run', 'files', 'problems'],
}

const SCORE_SCHEMA = {
  type: 'object',
  properties: {
    scorer_exit_code: { type: 'integer' },
    reading_line: { type: 'string', description: 'the reading line exactly as score_r10round.py prints it, or "incomplete"/"PROVISIONAL" with why' },
    provisional_items: { type: 'array', items: { type: 'string' } },
    incomplete_items: { type: 'array', items: { type: 'string' } },
    refetch_named: { type: 'array', items: { type: 'string' } },
    statistic_A: { type: 'string', description: 'A of n_A against LTSB r9 2 of 5, with Fisher p, as printed' },
    per_run: { type: 'array', items: { type: 'object', properties: { run_id: { type: 'string' }, complete_or_lost: { type: 'string' }, c1_c7: { type: 'string' }, tail_class: { type: 'string' }, kind: { type: 'string' }, gate: { type: 'string' }, outcome: { type: 'string' }, any_alarm: { type: 'string' }, parked_driven: { type: 'string' }, presence: { type: 'string' } }, required: ['run_id', 'complete_or_lost', 'c1_c7', 'tail_class', 'gate', 'outcome', 'any_alarm'] } },
    scorer_stdout_path: { type: 'string' },
    scorer_json_path: { type: 'string' },
    summary: { type: 'string' },
    problems: { type: 'array', items: { type: 'string' } },
  },
  required: ['scorer_exit_code', 'reading_line', 'provisional_items', 'incomplete_items', 'refetch_named', 'statistic_A', 'per_run', 'scorer_stdout_path', 'summary', 'problems'],
}

const CHECK_SCHEMA = {
  type: 'object',
  properties: {
    agrees_with_scorer: { type: 'boolean' },
    findings: { type: 'array', items: { type: 'object', properties: { id: { type: 'string' }, severity: { type: 'string', enum: ['blocking', 'minor', 'info'] }, claim: { type: 'string' }, evidence: { type: 'string' }, fix: { type: 'string' } }, required: ['id', 'severity', 'claim', 'evidence', 'fix'] } },
    independent_reading: { type: 'string' },
    report_text: { type: 'string' },
    files: { type: 'array', items: { type: 'string' } },
  },
  required: ['agrees_with_scorer', 'findings', 'independent_reading', 'report_text', 'files'],
}

// ---------------------------------------------------------------- Folders
phase('Folders')
const folders = await agent(`${COMMON}
YOUR ROLE: F2 run-folder step (one session, before the per-run fetches start). Your instructions are ${LB}/fetch_agent_brief.txt: read it in full first; FETCH = F2, RUN_IDS = ${RUN_IDS.join(', ')}, ROUND = ${ROUND}, TOOL = ${TOOL}.
1. Rule 7 of the brief: confirm RUNLOG holds an F2-START line for each of the three RUN_IDs (read only). If one is missing, stop and report.
2. For each RUN_ID run "TOOL next --fetch F2 --run <RUN_ID>" once and record its first line.
3. If any of them asks for the ppo folder listing or the run folders (THE RUN FOLDERS in the brief: a ppo listing older than F2-START), do THE RUN FOLDERS steps (a) and (b) exactly once, as the brief words them; then run next again for each RUN_ID and record the first line. Do nothing else: no other listing, no file transfer (the per-run sessions do those).
Report each RUN_ID's first next line before and after, the Drive calls made, and anything stopped.`, { label: 'F2 run folders', phase: 'Folders', schema: GEN_SCHEMA })
if (!folders) return { stage: 'folders', folders }
log(`folders: ${folders.summary}`)

// ---------------------------------------------------------------- Fetch
phase('Fetch')
const MAX_SESSIONS = 12
const fetchRun = async (runId) => {
  const sessions = []
  let deaths = 0
  for (let k = 1; k <= MAX_SESSIONS; k++) {
    const prev = sessions.length ? `This continues an F2 fetch of ${runId} already in progress: the tool's own state (${ROUND}/fetch/manifest.csv and the store) holds every earlier session's listings and copies, and TOOL next resumes from it. The previous session ended: ${JSON.stringify({ final_next_line: sessions[sessions.length - 1].final_next_line, stopped_reason: sessions[sessions.length - 1].stopped_reason })}.` : 'This is the first fetch session for this RUN_ID at F2.'
    const r = await agent(`${COMMON}
YOUR ROLE: F2 fetch agent for ONE RUN_ID, session ${k}. Your instructions are ${LB}/fetch_agent_brief.txt: read it in full first and follow it exactly. FETCH = F2, RUN_IDS = ${runId} (this session handles this RUN_ID only; two other sessions fetch the other RUN_IDs at the same time, and the tool locks the manifest), ROUND = ${ROUND}, TOOL = ${TOOL}.
${prev}
Run "TOOL next --fetch F2 --run ${runId}" and do exactly the action it prints, record it with the tool, and run next again, until it prints a line starting "done:" OR you have made 30 file transfers (download_file_content calls) in this session. Stop only at a clean point: right after a TOOL listing/transfer call has returned. Then run "TOOL status --fetch F2 --run ${runId}" and report. Set chunk_limit_reached true when you stopped because of the 30-transfer allowance (another session will continue). Saved pages and transfers go under ${ROUND}/fetch/raw/F2/${runId}/ as the brief says, each file name new (never overwrite or reuse an earlier session's file: continue the numbering; a raw file the manifest does not record, e.g. one ending .INCOMPLETE_unrecorded or a partial left by a session that died, is ignored and never passed to the tool).
THE MAINTAINER'S DECISION ON TRANSFERS (RUNLOG 13:31 UTC; it replaces the brief's "transcribe the content EXACTLY with the Write tool" and nothing else): NEVER retype base64. For every "transfer" action: (1) run date -u +%Y-%m-%dT%H:%M:%SZ and keep it as T0; (2) call download_file_content(fileId=ID); (3) if the response came inline (a base64 "content" field), save it byte for byte from the session record with "${PY} -I ${OUT}/xfer/extract_response.py --transcripts /root/.claude/projects/-home-user-mesozoic-labs/184ee67c-2802-5235-8e52-6dc7daa72cd3/subagents/workflows --file-id ID --after-utc T0 --out ${ROUND}/fetch/raw/F2/${runId}/<PATH with / as __>.t<k>.json" (k = the tool's transfer number for that file; the script refuses an existing --out and prints the decoded size), then "TOOL transfer --fetch F2 --run ${runId} --path PATH --saved <that file>". If the script says no response was found, wait 10 s and run it again (up to 3 times; the record may lag); if it still finds none, stop and report (blocked). If the response was saved to a file by the harness (large files), pass that saved file with --saved as the brief says. The tool's checks (id/title, size against the listing, sha256 against the manifest, at most 10 transfers per file, every transfer recorded) are unchanged. Do not write the response content yourself in any tool call or in your reply.
Set blocked true only for a stop the coordinator must resolve (a refusal, a permission check without a verdict, a tool INPUT ERROR you cannot clear by following the tool, or the script finding no response).`, { label: `F2 fetch ${runId} #${k}`, phase: 'Fetch', schema: FETCH_SCHEMA })
    if (!r) {
      deaths += 1
      sessions.push({ final_next_line: 'agent died', stopped_reason: 'agent returned null', done: false, chunk_limit_reached: false, blocked: false })
      log(`${runId} session ${k}: agent died (${deaths})`)
      if (deaths >= 2) break
      continue
    }
    deaths = 0
    sessions.push(r)
    log(`${runId} session ${k}: ${r.done ? 'DONE' : (r.blocked ? 'BLOCKED' : 'continuing')} — ${r.final_next_line.slice(0, 160)}`)
    if (r.done || r.blocked) break
  }
  const last = sessions[sessions.length - 1]
  return { run_id: runId, done: !!last.done, sessions }
}
const fetched = await parallel(RUN_IDS.map(id => () => fetchRun(id)))
const fetchDone = fetched.filter(Boolean).filter(f => f.done).map(f => f.run_id)
log(`F2 fetch: done for ${fetchDone.length} of 3 (${fetchDone.join(', ')})`)
if (fetchDone.length < 3) {
  log('F2 fetch not finished for every RUN_ID; stopping before measurement (the coordinator decides with the maintainer)')
  return { stage: 'fetch', folders, fetched }
}

// ---------------------------------------------------------------- Measure (class inputs)
phase('Measure')
const measure = await agent(`${COMMON}
F2 fetch results per RUN_ID (each ended with TOOL next printing "done:"): ${JSON.stringify(fetched.map(f => ({ run_id: f.run_id, final: f.sessions[f.sessions.length - 1].final_next_line, sessions: f.sessions.length })))}
YOUR ROLE: class-input measurer (C7 only; report-only measurements are a later stage, not yours).
1. For each RUN_ID run "${PY} -I -B ${LB}/tools/complete_r10round.py run --run <RUN_ID> --round ${ROUND}" (read its --help first for the panels default: <ROUND>/panels/<RUN_ID>/<label>/<kind>/). Save each stdout to ${OUT}/complete_pre/<RUN_ID>.txt. Read from it which C7 labels need a measurement (R1 = 10,485,760; the J checkpoints 4,194,304, 5,242,880, 6,291,456, 8,388,608, 10,485,760 up to E; every RESOLUTION checkpoint; final), and which pairs F2 holds as COPY matching the manifest. A run that fails C1, C2, C3 (non-zip), C4, C5 or C6 is measured only where the criteria still call for it (none for C7; say which).
2. For each C7 label of each RUN_ID: M1 with "${PY} -I -B ${LA}/tools/measure_r10round.py m1 --copy ${COPY} --zip <the F2 COPY of 01_stance/models/stage1_<steps>_steps.zip or stage1_final.zip> --pkl <its vecnormalize copy> --stage-config <the F2 COPY of 01_stance/stage_config.json> --run-id <RUN_ID> --label <steps or final>" (approval record default <ROUND>/APPROVED). Take the copy paths from the fetch store (TOOL status / the store layout ${ROUND}/fetch/F2/<RUN_ID>/copies, a copy whose sha256 equals the manifest's entry). For a FINISHED run also M3 at final: "measure_r10round.py m3 --copy ${COPY} --zip ... --pkl ... --stage-config ... --run-id <RUN_ID> --label final". At most 2 simulation processes at a time (OMP_NUM_THREADS=1 nice -n 10), logs under ${OUT}/logs/measure/. A crash is rerun on the same files (the tool moves the unfinished output aside); a refusal is recorded as the tool writes it and NOT worked around.
3. Re-run complete_r10round.py run per RUN_ID; save to ${OUT}/complete_post/<RUN_ID>.txt. Report each run's C1-C7 lines as printed, every measurement made (label, kind, rc, time), and any refusal or crash.`, { label: 'C7 measurements', phase: 'Measure', schema: GEN_SCHEMA })
if (!measure) return { stage: 'measure', folders, fetched, measure }

// ---------------------------------------------------------------- Score
phase('Score')
const score = await agent(`${COMMON}
Class-input measurer's report: ${JSON.stringify(measure)}
YOUR ROLE: scorer runner. Run "${PY} -I -B ${LB}/tools/score_r10round.py --round ${ROUND} --runlog ${RUNLOG} --json ${OUT}/score.json" (read --help first; pins and approval defaults are the pinned ones), saving stdout and stderr to ${OUT}/score.txt and the exit code (0 final, 10 incomplete, 11 PROVISIONAL, 2 input error). Also save complete_r10round.py run --json for each RUN_ID to ${OUT}/complete_<RUN_ID>.json. Report exactly what the scorer printed: the reading line, every PROVISIONAL and incomplete item, any re-fetch it names (with the time it prints), A of n_A with its Fisher p, and per run: COMPLETE or LOST (with the failing C-item), tail class and kind (actuator, sign, alarm step, flagged-evaluation count), gate, outcome, ANY ALARM, PARKED/DRIVEN, PRESENCE. Transcribe, do not reinterpret.`, { label: 'scorer', phase: 'Score', schema: SCORE_SCHEMA })
if (!score) return { stage: 'score', folders, fetched, measure, score }
log(`scorer exit ${score.scorer_exit_code}: ${score.reading_line.slice(0, 200)}`)

// ---------------------------------------------------------------- Check
phase('Check')
const check = await agent(`${COMMON}
The scorer's result as reported: ${JSON.stringify(score)}
YOUR ROLE: independent skeptic (you ran none of this). Try to show the reading is wrong. (a) From the files themselves (the F2 copies of each eval_episodes.csv, the M1 summaries under ${ROUND}/panels/<RUN_ID>/<label>/m1/m1_summary.json, the M3 summary at final, provenance.json, stage_config.json, progress.json, gate_verdict.json, the manifest), with your own small code written under ${OUT}/check1/ (never in a tool folder), re-derive per run: ANY-flagged and tail-flagged evaluations and the ANY/TAIL ALARM steps (criteria CLASSES, M5, THIS12 lines 351-364), NEEDS RESOLUTION, TAIL FLAGS at R1/R2, CMD_a and MEAN_a for actuators 11-14, TRACE RAIL, PARKED (incl. CMD_a(J) >= +0.60 at 2+ J checkpoints), DRIVEN, the tail class and kind, gate (M3 verdict at final), outcome; then A of n_A, the one-sided Fisher p against 2 of 5, and the ROUND READING line (first that applies). (b) Check C1-C7 facts against RUNLOG and the files (C4's session start within 60 min of the LAUNCH time; C5's judged_at within 2 h of progress written_at). (c) Check the scorer's PROVISIONAL/incomplete items are right and complete (pins, sha256 of criteria/THIS12/EVID/HEAD vs APPROVED, copies vs fetch manifest). Severity blocking only for a wrong class, alarm, A, n_A, gate, outcome or reading line, or a wrong COMPLETE/LOST. Write notes to ${OUT}/check1/NOTES.txt.`, { label: 'independent check of the reading', phase: 'Check', schema: CHECK_SCHEMA })

return { stage: 'scored', folders, fetched, measure, score, check }
