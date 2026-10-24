// Parse-check and dry-execute a Workflow script with stub hooks (no agents run).
import fs from 'node:fs'
const [file, argsJson, scenario] = process.argv.slice(2)
const src = fs.readFileSync(file, 'utf8')
if (!src.startsWith('export const meta = {')) throw new Error('script must begin with export const meta')
const body = src.replace(/^export const meta = /, 'const meta = ')
const prompts = []
const mk = (scen) => async (prompt, opts) => {
  prompts.push({ label: opts.label, phase: opts.phase, len: prompt.length, prompt })
  const run = (/stance5_LTSB_s\d\d/.exec(opts.label) || [''])[0]  // replication: s70, s80
  if (opts.phase === 'Find') return { run_id: run, found: scen !== 'notyet', folder: 'F' + run, stance: 'S' + run, models: 'M' + run, notes: '' }
  if (opts.phase === 'Place') return { run_id: run, ok: true, exit_code: 0, output: '{"ok": true}', outputs: [{ command: 'x', exit_code: 0, output: '{"ok": true}' }] }
  const role = opts.label.includes('tiebreak') ? 'C' : opts.label.slice(-1)
  const titles = [...prompt.matchAll(/(stage1_[A-Za-z0-9_]+\.(?:pkl|zip)|[a-z_]*model(?:_vecnorm)?\.(?:pkl|zip)|plant_identity\.json|stage_config\.json|task_fingerprint\.json|eval_episodes\.csv|progress\.json)/g)].map(m => m[1])
  const uniq = [...new Set(titles)]
  const notyet = scen === 'notyet' && run === 'stance5_LTSB_s80'
  const shaFor = (t) => (scen === 'differ' && role === 'A' && t.endsWith('.pkl') && run === 'stance5_LTSB_s70') ? 'bad' : 'good_' + t
  return { run_id: run, role, checkpoint_present: !notyet, notes: '',
    files: uniq.map(t => notyet ? { title: t, ok: false, method: 'missing' } : { title: t, ok: true, method: 'transcribed', sha256: shaFor(t), id: 'id', drive_size: 1 }) }
}
const fn = new Function('args', 'agent', 'parallel', 'pipeline', 'log', 'phase', 'budget', `return (async () => { ${body}\n })()`)
const parallel = async (thunks) => Promise.all(thunks.map(t => t().catch(e => { console.error('thunk error', e); return null })))
const pipeline = async (items, ...stages) => Promise.all(items.map(async (it, i) => { let v = it; for (const [k, s] of stages.entries()) { try { v = await s(k === 0 ? it : v, it, i) } catch (e) { console.error('stage error', e); return null } } return v }))
const logs = []
try {
  const res = await fn(JSON.parse(argsJson), mk(scenario), parallel, pipeline, (m) => logs.push(m), () => {}, { total: null })
  console.log(JSON.stringify({ ok: true, agents: prompts.map(p => `${p.phase}:${p.label}(${p.len})`), logs, statuses: Array.isArray(res) ? res.map(r => r && (r.status + (r.placed ? '+placed' : ''))) : res }, null, 1))
  if (process.env.SHOW) console.log(prompts.find(p => p.label.endsWith(process.env.SHOW)).prompt)
} catch (e) { console.log(JSON.stringify({ ok: false, error: String(e) })) }
