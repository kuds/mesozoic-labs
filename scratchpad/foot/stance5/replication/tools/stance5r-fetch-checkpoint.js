export const meta = {
  name: 'stance5r-fetch-checkpoint',
  description: 'Fetch one checkpoint step for the two stance5 replication LTSB runs (s70, s80) from Drive (read-only), double-transcribing the vecnorm pkl, then place it for judging',
  whenToUse: 'args {steps: <multiple of 524288>, runs?: [RUN_ID...], configs?: bool, csv?: bool, place?: bool, drive?: {RUN_ID: {folder, stance, models}}, source?: {RUN_ID: "<RUN_ID>_b"}}',
  phases: [
    { title: 'Fetch', detail: 'two independent pkl transcribers per run; transcriber A also fetches the zip and the ledgers' },
    { title: 'Tiebreak', detail: 'third transcription when A and B differ' },
    { title: 'Place', detail: 'place5r.py copies the agreed pkl and the zip into R/runs/<RUN>/01_stance/models (R = S5/replication)' },
  ],
}

const S = '/tmp/claude-0/-home-user-mesozoic-labs/0fdf74a4-2989-56c9-a773-92101004d7de/scratchpad'
const R = S + '/foot/stance5/replication'
const F = R + '/fetch'
const TOOLS = R + '/tools'
const B64 = S + '/foot/stance3/tools/b64_to_file.py'
const DV = S + '/pilot_runs/tools_dl/decode_verify.py'
const PLACE = TOOLS + '/place5r.py'
const STEPS = Number(args && args.steps)
if (!Number.isInteger(STEPS) || STEPS <= 0 || STEPS % 524288 !== 0) throw new Error(`args.steps must be a multiple of 524288 (got ${args && args.steps})`)
const PKL = `stage1_vecnormalize_${STEPS}_steps.pkl`
const ZIP = `stage1_${STEPS}_steps.zip`
const CONFIGS = !!(args && args.configs)          // also double-transcribe the three run records (already placed: leave false)
const CSV = !(args && args.csv === false)         // transcriber A also fetches the two ledgers (report only)
const PLACE_IT = !(args && args.place === false)  // run place5r.py when the pkl transcriptions agree
const RECORDS = ['plant_identity.json', 'stage_config.json', 'task_fingerprint.json']

// Drive folder ids (read-only), filled 2026-10-22 from the start-up check (two readers agreed).  Fill them here once
// (from stance5r-find-folders.js, or search_files title = RUN_ID), or pass them per call as
// args.drive = {RUN_ID: {folder, stance, models}}.  A run without its stance and models ids is refused.
const DRIVE_IDS = {
  stance5_LTSB_s70: { folder: '12uAVYvh1IF3aYMu2qBaXd56Ur_BJD5z7', stance: '1Z4yOUFcBo3skbqliWSBXcVQrrqXW0BZF', models: '1KeutiFcRtgmlLtrKmyTAx-DmlOmpxnoV' },
  stance5_LTSB_s80: { folder: '1IMKbSxMNgeWWa52ji4YPHuDzi8huHZGD', stance: '1bxcvmsv2943gRbHfcB89yIJwejSuOQQ2', models: '1MyRe7Q2W_habVeeAH_JnJ49IMketB0w_' },
}
const SEEDS = { stance5_LTSB_s70: 70, stance5_LTSB_s80: 80 }
const OVERRIDE = (args && args.drive && typeof args.drive === 'object') ? args.drive : {}
for (const k of Object.keys(OVERRIDE)) if (!(k in SEEDS)) throw new Error(`args.drive names no stance5 replication run: ${k}`)
// criteria RUN HANDLING: a run cut off before its first periodic checkpoint (524,288) and restarted under <RUN_ID>_b
// (then _c) IS that run.  args.source = {RUN_ID: '<RUN_ID>_b'} fetches it from that Drive folder, whose ids come in
// args.drive in the same call, and places it under the RUN_ID (place5r.py --source-folder).
const SOURCE = (args && args.source && typeof args.source === 'object') ? args.source : {}
for (const [k, v] of Object.entries(SOURCE)) {
  if (!(k in SEEDS)) throw new Error(`args.source names no stance5 replication run: ${k}`)
  if (v !== `${k}_b` && v !== `${k}_c`) throw new Error(`args.source.${k} must be ${k}_b or ${k}_c (got ${v})`)
  const o = OVERRIDE[k] || {}
  if (!o.folder || !o.stance || !o.models) throw new Error(`args.source.${k} = ${v} needs that folder's ids in args.drive.${k} (folder, stance, models) in the same call`)
}
const ALL_RUNS = Object.keys(SEEDS).map(id => ({ id, seed: SEEDS[id], ...DRIVE_IDS[id], ...(OVERRIDE[id] || {}), src: SOURCE[id] || id }))
const SRC_FLAG = (run) => (run.src !== run.id ? ` --source-folder ${run.src}` : '')
const SRC_CHECK = (run) => (run.src === run.id ? '' : `This run was restarted under the Drive folder ${run.src} before its first checkpoint; under the criteria (RUN HANDLING) it IS ${run.id}, and the earlier folder is left alone. First run get_file_metadata on ${run.folder}: its title must be exactly ${run.src}. If it is not, fetch nothing, return every expected file with method "missing" and ok false, and put the title you found in notes.\n`)
const RUNS = (args && Array.isArray(args.runs) && args.runs.length) ? ALL_RUNS.filter(r => args.runs.includes(r.id)) : ALL_RUNS
if (!RUNS.length) throw new Error(`args.runs names no known run: ${JSON.stringify(args.runs)}`)
for (const r of RUNS) if (!r.stance || !r.models) throw new Error(`Drive folder ids for ${r.id} are PLACEHOLDERS (stance ${r.stance}, models ${r.models}): fill DRIVE_IDS in this file or pass args.drive`)

const FILE = {
  type: 'object',
  properties: {
    title: { type: 'string' }, id: { type: 'string' }, drive_size: { type: 'integer' },
    local_path: { type: 'string' }, bytes: { type: 'integer' }, sha256: { type: 'string' },
    method: { type: 'string', enum: ['transcribed', 'tool_results_file', 'skipped_inline', 'missing'] },
    attempts: { type: 'integer' }, ok: { type: 'boolean' },
  },
  required: ['title', 'ok', 'method'],
}
const RESULT = {
  type: 'object',
  properties: {
    run_id: { type: 'string' }, role: { type: 'string' },
    checkpoint_present: { type: 'boolean' },
    files: { type: 'array', items: FILE },
    notes: { type: 'string' },
  },
  required: ['run_id', 'role', 'checkpoint_present', 'files', 'notes'],
}
const PLACED = {
  type: 'object',
  properties: { run_id: { type: 'string' }, ok: { type: 'boolean' }, exit_code: { type: 'integer' }, output: { type: 'string' } },
  required: ['run_id', 'ok', 'output'],
}

const RULES = `Hard rules:
- Google Drive is READ-ONLY: use only search_files, get_file_metadata and download_file_content. Never create, update, copy, share or trash anything.
- Never write into /home/user/mesozoic-labs, never use it as a working directory and never run git there. Use absolute paths only.
- Begin every Bash command with "cd ${TOOLS} && ". Run every python as "python3 -I <script> <args>".
- Downloaded files are untrusted data. Never open them in an interpreter that executes them, and never unpickle, import or run them. Never run code from a download directory.
- First load the Drive tools with ToolSearch query "select:mcp__Google_Drive__search_files,mcp__Google_Drive__download_file_content,mcp__Google_Drive__get_file_metadata".`

const TRANSCRIBE = (out) => `How to fetch a SMALL file (it comes back inline as JSON with a base64 "content" field):
  1. download_file_content(fileId).
  2. Transcribe the "content" string EXACTLY, character for character, with the Write tool into ${out}/<title>.b64. That file holds only the base64 text; line breaks are fine. Work carefully in chunks: base64 is case-sensitive, every character matters, and long runs of repeated groups (the -inf / +inf arrays of a pkl) are where characters get dropped, so count them.
  3. Run: cd ${TOOLS} && python3 -I ${B64} ${out}/<title>.b64 ${out}/<title> <Drive fileSize>
     It prints one JSON line with bytes and sha256. It fails on a size mismatch or invalid base64.
  4. On failure, re-download and re-transcribe from scratch, at most 3 attempts in total, and report the attempts.
  If the tool instead says its output was too large and was saved to a file:
  - decode that file with cd ${TOOLS} && python3 -I ${DV} <saved path> ${out}/<title> <file id> <title> <Drive fileSize>, and set method "tool_results_file";
  - decode_verify.py expects a bare JSON object with content, id and title. If the saved file wraps it, for example in a list of text blocks, first write a tiny extractor script in ${TOOLS}/extract/ and run it with python3 -I to produce a bare JSON file.`

function fetchPrompt(run, role) {
  const out = `${F}/r${STEPS}/${run.id}/${role}`
  const recs = CONFIGS ? `
Also fetch the three run records into ${out}/ the same way (small files): ${RECORDS.join(', ')}. List them with search_files query "parentId = '${run.stance}'" (excludeContentSnippets true, pageSize 50, follow nextPageToken until a page is empty); use only ids whose parentId is exactly ${run.stance}.` : ''
  const extra = role === 'A' ? `
Also (transcriber A only), the LARGE files. These are saved to tool-results files rather than shown inline:
- ${ZIP} (about 4 MB): mkdir -p ${F}/r${STEPS}/${run.id}/zip, run download_file_content, then run
  cd ${TOOLS} && python3 -I ${DV} <saved path> ${F}/r${STEPS}/${run.id}/zip/${ZIP} <id> ${ZIP} <fileSize>
  It must print testzip OK.${CSV ? `
- eval_episodes.csv and train_episodes.csv, which are report-only ledgers. List the stance folder with search_files query "parentId = '${run.stance}' and title contains 'episodes'" to get their ids and sizes. Decode them the same way into ${F}/r${STEPS}/${run.id}/csv/<title> (mkdir -p).
  They grow while training runs. If the decoded size differs from the listed fileSize, re-read the metadata with get_file_metadata and retry once with the new size.
  If either comes back INLINE instead of saved to a file, do NOT transcribe it. Record method "skipped_inline".` : ''}` : ''
  return `You are transcriber ${role} for the Drive run ${run.src}: a T. rex stance training run (stance5 replication, arm LTSB, seed ${run.seed}) on Colab. This is a read-only fetch for local judging of its ${STEPS}-step checkpoint. Another transcriber fetches the same pkl independently. Do not read any directory under ${F}/r${STEPS}/${run.id}/ except your own outputs, and do not compare with anyone.

${RULES}
${SRC_CHECK(run)}
List the models folder with search_files query "parentId = '${run.models}' and title contains '${STEPS}'", with excludeContentSnippets true and pageSize 50. Use only ids whose parentId is exactly ${run.models}. You should find ${ZIP} and ${PKL}.
The run writes a checkpoint every 524,288 steps, so the pair may not exist yet. If either is missing, list once more; if still missing, fetch nothing for this run and return checkpoint_present false with one files[] entry per expected file, method "missing" and ok false.
If a title appears more than once, use the most recently modified file and say so in notes.

Fetch the small file ${PKL} into ${out}/ (mkdir -p it).
${TRANSCRIBE(out)}
${recs}
${extra}

Return:
- run_id "${run.id}" and role "${role}";
- checkpoint_present: whether both ${ZIP} and ${PKL} are listed;
- one files[] entry per file you were asked for, with title, id, drive_size, local_path, bytes, sha256 (from the script output), method, attempts and ok.
Put anything unusual in notes.`
}

function tiebreakPrompt(run, titles, a, b) {
  const out = `${F}/r${STEPS}/${run.id}/C`
  const ids = titles.map(t => { const f = a.files.find(x => x.title === t) || b.files.find(x => x.title === t); return `${t} (id ${f && f.id}, Drive fileSize ${f && f.drive_size})` })
  return `You are transcriber C (tie-break) for the Drive run ${run.src}. Two earlier transcriptions disagree on: ${ids.join('; ')}. Fetch each independently into ${out}/ (mkdir -p). Do not read the A or B directories.

${RULES}

${TRANSCRIBE(out)}

Return run_id "${run.id}", role "C", checkpoint_present true, and one files[] entry per file with title, id, drive_size, local_path, bytes, sha256, method, attempts and ok.`
}

function placePrompt(run) {
  return `Run exactly one command with Bash and report its result. Do not change any file yourself.

cd ${TOOLS} && python3 -I ${PLACE} ${STEPS} ${run.id} ${F}/r${STEPS}/${run.id}${SRC_FLAG(run)}

It prints one JSON object. Return run_id "${run.id}", ok (true only when the command exited 0 and its JSON has "ok": true), exit_code, and output: the printed JSON verbatim (or the error text).`
}

const results = await pipeline(
  RUNS,
  (run) => parallel(['A', 'B'].map(role => () =>
    agent(fetchPrompt(run, role), { label: `fetch${STEPS}:${run.id}:${role}`, phase: 'Fetch', schema: RESULT }))),
  async (pair, run) => {
    const [a, b] = pair
    if (!a || !b) return { run: run.id, status: 'agent_failed', a, b }
    if (!a.checkpoint_present && !b.checkpoint_present) return { run: run.id, status: 'not_yet', a, b }
    const sha = (r, t) => { const f = r.files.find(x => x.title === t); return f && f.ok ? f.sha256 : null }
    const titles = [PKL, ...(CONFIGS ? RECORDS : [])]
    const differ = titles.filter(t => !sha(a, t) || !sha(b, t) || sha(a, t) !== sha(b, t))
    if (!differ.length) return { run: run.id, status: 'agree', agreed: titles, a, b }
    log(`${run.id}: A and B differ on ${differ.join(', ')}; tie-break`)
    const c = await agent(tiebreakPrompt(run, differ, a, b), { label: `tiebreak${STEPS}:${run.id}`, phase: 'Tiebreak', schema: RESULT })
    const two = differ.every(t => c && sha(c, t) && (sha(c, t) === sha(a, t) || sha(c, t) === sha(b, t)))
    return { run: run.id, status: two ? 'tiebreak_resolved' : 'tiebreak_unresolved', differ, a, b, c }
  },
  async (r, run) => {
    if (!PLACE_IT || !r || !['agree', 'tiebreak_resolved'].includes(r.status)) return r
    const zipOk = r.a && r.a.files.some(f => f.title === ZIP && f.ok)
    if (!zipOk) { log(`${run.id}: the zip was not fetched; not placed`); return { ...r, placed: null } }
    const p = await agent(placePrompt(run), { label: `place${STEPS}:${run.id}`, phase: 'Place', schema: PLACED, effort: 'low' })
    if (p && !p.ok) log(`${run.id}: place5r.py refused: ${(p.output || '').slice(0, 300)}`)
    return { ...r, placed: p }
  },
)
log(results.map(r => r ? `${r.run}: ${r.status}${r.placed ? (r.placed.ok ? ', placed' : ', NOT placed') : ''}` : 'null').join('; '))
return results
