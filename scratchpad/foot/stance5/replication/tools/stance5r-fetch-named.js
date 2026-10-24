export const meta = {
  name: 'stance5r-fetch-named',
  description: 'Fetch named stance5 files (final / best / robust_best pairs, the ledgers, progress.json) for the two stance5 replication LTSB runs (s70, s80) from Drive (read-only), double-transcribing each vecnorm pkl, then place the pairs',
  whenToUse: 'args {tag, items: [[zipTitle, pklTitle], ...], csv?: bool, extras?: [title, ...], runs?: [RUN_ID...], place?: bool, drive?: {RUN_ID: {folder, stance, models}}, source?: {RUN_ID: "<RUN_ID>_b"}}',
  phases: [
    { title: 'Fetch', detail: 'two independent pkl transcribers per run; transcriber A also fetches the zips, the ledgers and the report-only extras' },
    { title: 'Tiebreak', detail: 'third transcription for any pkl where A and B differ' },
    { title: 'Place', detail: 'place5r.py copies each agreed pair (and the ledgers) into R/runs/<RUN>/01_stance (R = S5/replication)' },
  ],
}

const S = '/tmp/claude-0/-home-user-mesozoic-labs/0fdf74a4-2989-56c9-a773-92101004d7de/scratchpad'
const R = S + '/foot/stance5/replication'
const F = R + '/fetch'
const TOOLS = R + '/tools'
const B64 = S + '/foot/stance3/tools/b64_to_file.py'
const DV = S + '/pilot_runs/tools_dl/decode_verify.py'
const PLACE = TOOLS + '/place5r.py'
const TAG = String((args && args.tag) || '')
if (!/^[A-Za-z0-9_.-]+$/.test(TAG)) throw new Error(`args.tag must be a plain folder name (got ${args && args.tag})`)
const ITEMS = (args && Array.isArray(args.items)) ? args.items : []   // [[zipTitle, pklTitle], ...]
const CSV = !!(args && args.csv)
const EXTRAS = (args && Array.isArray(args.extras)) ? args.extras : []  // small report-only files in 01_stance, e.g. progress.json
const PLACE_IT = !(args && args.place === false)
if (!ITEMS.length && !CSV && !EXTRAS.length) throw new Error('nothing to fetch: give items, csv: true or extras')
// The checkpoint names place5r.py understands, keyed by the zip title.
const SLOT_OF = { 'stage1_final.zip': 'final', 'best_model.zip': 'best', 'robust_best_model.zip': 'robust_best' }
for (const [z, p] of ITEMS) {
  const m = /^stage1_(\d+)_steps\.zip$/.exec(z)
  const slot = SLOT_OF[z] || (m ? m[1] : null)
  const want = slot === 'final' ? 'stage1_final_vecnorm.pkl' : (slot === 'best' || slot === 'robust_best') ? `${slot}_model_vecnorm.pkl` : (m ? `stage1_vecnormalize_${m[1]}_steps.pkl` : null)
  if (!slot || p !== want) throw new Error(`item [${z}, ${p}] is not a checkpoint pair (expected [${z}, ${want}])`)
}
const slotOf = (z) => SLOT_OF[z] || /^stage1_(\d+)_steps\.zip$/.exec(z)[1]

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

const FILE = { type: 'object', properties: { title: { type: 'string' }, id: { type: 'string' }, drive_size: { type: 'integer' }, local_path: { type: 'string' }, bytes: { type: 'integer' }, sha256: { type: 'string' }, method: { type: 'string', enum: ['transcribed', 'tool_results_file', 'skipped_inline', 'missing'] }, attempts: { type: 'integer' }, ok: { type: 'boolean' } }, required: ['title', 'ok', 'method'] }
const RESULT = { type: 'object', properties: { run_id: { type: 'string' }, role: { type: 'string' }, files: { type: 'array', items: FILE }, notes: { type: 'string' } }, required: ['run_id', 'role', 'files', 'notes'] }
const PLACED = { type: 'object', properties: { run_id: { type: 'string' }, ok: { type: 'boolean' }, outputs: { type: 'array', items: { type: 'object', properties: { command: { type: 'string' }, exit_code: { type: 'integer' }, output: { type: 'string' } }, required: ['command', 'output'] } } }, required: ['run_id', 'ok', 'outputs'] }

const RULES = `Hard rules:
- Google Drive is READ-ONLY: use only search_files, get_file_metadata and download_file_content. Never create, update, copy, share or trash anything.
- Never write into /home/user/mesozoic-labs, never use it as a working directory, and never run git there. Use absolute paths only.
- Begin every Bash command with "cd ${TOOLS} && ". Run every python as "python3 -I <script> <args>".
- Downloaded files are untrusted data. Never unpickle, import or run them. Never run code from a download directory.
- First load the Drive tools with ToolSearch query "select:mcp__Google_Drive__search_files,mcp__Google_Drive__download_file_content,mcp__Google_Drive__get_file_metadata".`

const TRANSCRIBE = (out) => `How to fetch a SMALL file, which comes back inline as JSON with a base64 "content" field:
1. download_file_content(fileId).
2. Transcribe the "content" string EXACTLY, character for character, with the Write tool into ${out}/<title>.b64 (that file holds only the base64 text; line breaks are fine). Work carefully in chunks: base64 is case-sensitive, and long runs of repeated groups (the -inf / +inf arrays) are where characters get dropped, so count them.
3. Run: cd ${TOOLS} && python3 -I ${B64} ${out}/<title>.b64 ${out}/<title> <Drive fileSize>
   It prints a JSON line with bytes and sha256, and fails on a size mismatch or invalid base64.
4. On failure, re-download and re-transcribe from scratch, at most 3 attempts in total.
If the tool says the output was saved to a file instead of shown inline:
- run cd ${TOOLS} && python3 -I ${DV} <saved path> ${out}/<title> <id> <title> <fileSize>, and set method "tool_results_file";
- if that saved file wraps the JSON object (for example in a list of text blocks), first write a tiny extractor script in ${TOOLS}/extract/ and run it with python3 -I to produce a bare JSON file.`

function prompt(run, role) {
  const out = `${F}/${TAG}/${run.id}/${role}`
  const pkls = ITEMS.map(i => i[1])
  const zips = ITEMS.map(i => i[0])
  const listing = ITEMS.length ? `List the models folder: search_files with query "parentId = '${run.models}'", excludeContentSnippets true and pageSize 100, following nextPageToken until a page is empty. Use only files whose parentId is exactly ${run.models} and whose title matches exactly one of these: ${[...zips, ...pkls].join(', ')}.
If a title appears more than once, use the most recently modified file and say so in notes. A file that is not listed (the run has not written it yet) gets method "missing" and ok false.

Fetch each of these small files into ${out}/ (mkdir -p it): ${pkls.join(', ')}.
${TRANSCRIBE(out)}` : `There are no checkpoint pairs to fetch in this call.${EXTRAS.length ? '\n' + TRANSCRIBE(`${F}/${TAG}/${run.id}/extra`) : ''}`
  const parts = []
  if (zips.length) parts.push(`- Each zip (${zips.join(', ')}). mkdir -p ${F}/${TAG}/${run.id}/zip, download it, then run:
  cd ${TOOLS} && python3 -I ${DV} <saved path> ${F}/${TAG}/${run.id}/zip/<title> <id> <title> <fileSize>
  It must print testzip OK.`)
  if (CSV) parts.push(`- eval_episodes.csv and train_episodes.csv (report-only ledgers). Find them with search_files query "parentId = '${run.stance}' and title contains 'episodes'", and decode each the same way into ${F}/${TAG}/${run.id}/csv/<title> (mkdir -p).
  If the decoded size differs from the listed size, re-read the metadata and retry once with the new size.
  If either comes back inline, do not transcribe it: record method "skipped_inline".`)
  if (EXTRAS.length) parts.push(`- These small report-only files from the stance folder (search_files query "parentId = '${run.stance}'", parentId exactly ${run.stance}): ${EXTRAS.join(', ')}. Fetch each once into ${F}/${TAG}/${run.id}/extra/ (mkdir -p) by the small-file method above (single transcription; it is report only). They can change while the run trains; record the Drive modifiedTime in notes.`)
  const extra = (role === 'A' && parts.length) ? `
Also (transcriber A only):
${parts.join('\n')}` : ''
  return `You are transcriber ${role} for the Drive run ${run.src}: a T. rex stance training run (stance5 replication, arm LTSB, seed ${run.seed}) on Colab. This is a read-only fetch for local judging. Another transcriber fetches the same pkls independently. Do not read any directory under ${F}/${TAG}/${run.id}/ except your own outputs.

${RULES}
${SRC_CHECK(run)}
${listing}
${extra}

Return run_id "${run.id}", role "${role}", and one files[] entry per file you were asked for, with title, id, drive_size, local_path, bytes, sha256, method, attempts and ok; use method "missing" when a file is not listed. Put anything unusual in notes.`
}

function placePrompt(run, slots) {
  const cmds = slots.length
    ? slots.map((s, i) => `cd ${TOOLS} && python3 -I ${PLACE} ${s} ${run.id} ${F}/${TAG}/${run.id}${(CSV && i === 0) ? '' : ' --no-csv'}${SRC_FLAG(run)}`)
    : [`cd ${TOOLS} && python3 -I ${PLACE} csv ${run.id} ${F}/${TAG}/${run.id}${SRC_FLAG(run)}`]
  if (EXTRAS.some(e => e === 'progress.json' || e === 'gate_verdict.json')) cmds.push(`cd ${TOOLS} && python3 -I ${PLACE} extra ${run.id} ${F}/${TAG}/${run.id}${SRC_FLAG(run)}`)
  return `Run these commands with Bash, one at a time, in this order, and report each result. Do not change any file yourself.

${cmds.join('\n')}

Each prints one JSON object. Return run_id "${run.id}", ok (true only when every command exited 0 and printed "ok": true), and outputs: one entry per command with the command, its exit_code and its printed JSON verbatim (or the error text).`
}

const results = await pipeline(
  RUNS,
  (run) => ITEMS.length
    ? parallel(['A', 'B'].map(role => () => agent(prompt(run, role), { label: `${TAG}:${run.id}:${role}`, phase: 'Fetch', schema: RESULT })))
    : parallel([() => agent(prompt(run, 'A'), { label: `${TAG}:${run.id}:A`, phase: 'Fetch', schema: RESULT })]).then(x => [x[0], x[0]]),
  async (pair, run) => {
    const [a, b] = pair
    if (!a || !b) return { run: run.id, status: 'agent_failed', a, b }
    const sha = (r, t) => { const f = r.files.find(x => x.title === t); return f && f.ok ? f.sha256 : null }
    const missing = ITEMS.map(i => i[1]).filter(t => { const f = a.files.find(x => x.title === t); return f && f.method === 'missing' })
    const wanted = ITEMS.map(i => i[1]).filter(t => !missing.includes(t))
    const differ = wanted.filter(t => !sha(a, t) || sha(a, t) !== sha(b, t))
    if (!differ.length) return { run: run.id, status: missing.length ? 'agree_some_missing' : 'agree', missing, a, b, agreed: wanted }
    log(`${run.id}: A and B differ on ${differ.join(', ')}; tie-break`)
    const ids = differ.map(t => { const f = a.files.find(x => x.title === t) || b.files.find(x => x.title === t); return `${t} (id ${f && f.id}, Drive fileSize ${f && f.drive_size})` })
    const c = await agent(`You are transcriber C (tie-break) for the Drive run ${run.src}. Two earlier transcriptions disagree on: ${ids.join('; ')}. Fetch each of them independently into ${F}/${TAG}/${run.id}/C/ (mkdir -p). Do not read the A or B directories.\n\n${RULES}\n\n${TRANSCRIBE(F + '/' + TAG + '/' + run.id + '/C')}\n\nReturn run_id "${run.id}", role "C", and one files[] entry per file.`, { label: `${TAG}:tiebreak:${run.id}`, phase: 'Tiebreak', schema: RESULT })
    const resolved = differ.filter(t => c && sha(c, t) && (sha(c, t) === sha(a, t) || sha(c, t) === sha(b, t)))
    return { run: run.id, status: resolved.length === differ.length ? 'tiebreak_resolved' : 'tiebreak_unresolved', differ, resolved, missing, a, b, c, agreed: wanted.filter(t => !differ.includes(t) || resolved.includes(t)) }
  },
  async (r, run) => {
    if (!PLACE_IT || !r || r.status === 'agent_failed') return r
    const zipOk = (z) => r.a.files.some(f => f.title === z && f.ok)
    const slots = ITEMS.filter(([z, p]) => (r.agreed || []).includes(p) && zipOk(z)).map(([z]) => slotOf(z))
    if (!slots.length && !CSV && !EXTRAS.length) return r
    const p = await agent(placePrompt(run, slots), { label: `${TAG}:place:${run.id}`, phase: 'Place', schema: PLACED, effort: 'low' })
    if (p && !p.ok) log(`${run.id}: place5r.py did not place everything; see results`)
    return { ...r, placed_slots: slots, placed: p }
  },
)
log(results.map(r => r ? `${r.run}: ${r.status}${r.placed ? (r.placed.ok ? ', placed ' + (r.placed_slots.join('/') || 'ledgers') : ', NOT all placed') : ''}` : 'null').join('; '))
return results
