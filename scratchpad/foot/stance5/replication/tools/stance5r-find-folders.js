export const meta = {
  name: 'stance5r-find-folders',
  description: 'Find the Drive folder ids (run folder, 01_stance, models) of the two stance5 replication LTSB runs (s70, s80) by RUN_ID, read-only, to fill DRIVE_IDS in the stance5r fetch workflows',
  whenToUse: 'args {runs?: [RUN_ID...]}; run once after the runs have started (their folders exist on Drive)',
  phases: [
    { title: 'Find', detail: 'one read-only lookup per run: the RUN_ID folder, its 01_stance and models subfolders, and progress.json' },
  ],
}

const S = '/tmp/claude-0/-home-user-mesozoic-labs/0fdf74a4-2989-56c9-a773-92101004d7de/scratchpad'
const TOOLS = S + '/foot/stance5/replication/tools'
const SEEDS = { stance5_LTSB_s70: 70, stance5_LTSB_s80: 80 }
const LABELS = { stance5_LTSB_s70: 'stance screen LTSB s70', stance5_LTSB_s80: 'stance screen LTSB s80' }
const WANT = (args && Array.isArray(args.runs) && args.runs.length) ? args.runs : Object.keys(SEEDS)
for (const r of WANT) if (!(r in SEEDS)) throw new Error(`args.runs names no stance5 replication run: ${r}`)

const FOUND = {
  type: 'object',
  properties: {
    run_id: { type: 'string' }, found: { type: 'boolean' },
    folder: { type: 'string' }, stance: { type: 'string' }, models: { type: 'string' },
    path_checked: { type: 'string' },
    progress_json: { type: 'object', properties: { id: { type: 'string' }, modified_time: { type: 'string' }, timesteps: { type: 'integer' }, status: { type: 'string' } } },
    candidates: { type: 'integer' }, notes: { type: 'string' },
  },
  required: ['run_id', 'found', 'notes'],
}

const RULES = `Hard rules:
- Google Drive is READ-ONLY: use only search_files and get_file_metadata. Never create, update, copy, share, download or trash anything.
- Never write any file, and never run git. You only read Drive metadata and report ids.
- First load the Drive tools with ToolSearch query "select:mcp__Google_Drive__search_files,mcp__Google_Drive__get_file_metadata".`

function prompt(run) {
  return `Find the Google Drive folder ids of the Colab training run ${run} (a T. rex stance5 replication run, seed ${SEEDS[run]}), read-only.

${RULES}

The run's folder is MyDrive/mesozoic-labs/logs/tyrannosaurus_rex/anatomical/ppo/${run}/ and holds 01_stance/, which holds models/.
1. search_files with query "title = '${run}' and mimeType = 'application/vnd.google-apps.folder'" (excludeContentSnippets true). If more than one folder matches, use get_file_metadata on each and keep the one whose parent folder is titled "ppo" (check that parent's parent is "anatomical"); report the number of candidates. A folder titled "${run}_b" (or "${run}_c") is this run's restart: under the criteria (RUN HANDLING) it IS this run when the earlier folder was cut off before its first periodic checkpoint (524,288). Do not put its ids in folder, stance or models; report in notes its id, its 01_stance and models ids, and whether the earlier folder's models folder lists any stage1_*_steps.zip (search_files query "parentId = '<models id>' and title contains '_steps.zip'"). The fetch workflows take its ids as args.drive with args.source. Also search once for a folder titled '${LABELS[run]}' (the RUN_LABEL typed into RUN_ID by mistake, as happened once on stance5): if one exists, do not use it; report its id and modifiedTime in notes.
2. search_files with query "parentId = '<that folder id>' and title = '01_stance'" -> the stance folder id.
3. search_files with query "parentId = '<the stance folder id>' and title = 'models'" -> the models folder id.
4. search_files with query "parentId = '<the stance folder id>' and title = 'progress.json'"; if found, report its id and modifiedTime (get_file_metadata, snippetVerbosity MAX_ALLOWED) and, if the snippet shows them, its "timesteps" and "status".
If the run folder does not exist yet, return found false and say so in notes.

Return run_id "${run}", found, folder, stance, models (ids), path_checked (the titles of the parents you verified, e.g. "ppo/anatomical"), progress_json, candidates and notes.`
}

const results = await parallel(WANT.map(run => () => agent(prompt(run), { label: `find:${run}`, phase: 'Find', schema: FOUND, effort: 'low' })))
const table = {}
for (const r of results) if (r && r.found) table[r.run_id] = { folder: r.folder, stance: r.stance, models: r.models }
log(`DRIVE_IDS for ${TOOLS}/stance5r-fetch-*.js (or args.drive): ${JSON.stringify(table)}`)
return { results, drive: table }
