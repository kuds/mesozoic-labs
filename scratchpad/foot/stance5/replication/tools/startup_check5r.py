"""Start-up check of a stance5 replication run's records against stance5 s52's (report only, reads JSON as data).

usage: python3 -I startup_check5r.py RUN_ID DIR
DIR holds what was read from Drive: stage_config.json, task_fingerprint.json, plant_identity.json, progress.json
(01_stance) and provenance.json (the run folder), any of them possibly absent.  On stance5 the three runs'
stage_config.json differed only in run.seed and run.label, and their task_fingerprint.json and plant_identity.json
not at all (measured 2026-10-22), so the new run's must equal s52's except run.seed and run.label.
Prints one JSON object; exit 0 when every check present passes, 1 otherwise, 2 on usage errors.
"""
import json
import os
import sys

REF = '/tmp/claude-0/-home-user-mesozoic-labs/0fdf74a4-2989-56c9-a773-92101004d7de/scratchpad/foot/stance5/runs/stance5_LTSB_s52/01_stance'
RUNS = {'stance5_LTSB_s70': (70, 'stance screen LTSB s70'), 'stance5_LTSB_s80': (80, 'stance screen LTSB s80')}
COMMIT = 'cf80ecb14d1630ce3b4dcdea28a05835d143470b'
TASK = 'sha256:8c6998d260b850ae054c9c2d49a8e5bef2bf59d4770ad77d3674b911721918cf'
HP = 'sha256:054ab48303167eda1bd66c8894a64ef516bedd041a9e12f57dbd20e75c026b9c'


def flat(o, p=''):
    out = {}
    if isinstance(o, dict):
        for k, v in o.items():
            out.update(flat(v, f'{p}.{k}' if p else k))
    else:
        out[p] = o
    return out


def load(path):
    if not os.path.exists(path):
        return None, 'absent'
    try:
        with open(path) as f:
            return json.load(f), None
    except ValueError as e:
        return None, f'not JSON: {e}'


def main():
    if len(sys.argv) != 3 or sys.argv[1] not in RUNS:
        print(__doc__)
        return 2
    run, d = sys.argv[1], sys.argv[2]
    seed, label = RUNS[run]
    res = {'run_id': run, 'dir': d, 'checks': [], 'diffs': {}, 'readouts': {}}

    def check(name, ok, detail):
        res['checks'].append({'check': name, 'ok': bool(ok), 'detail': detail})

    sc, err = load(os.path.join(d, 'stage_config.json'))
    if sc is None:
        check('stage_config.json', False, err)
    else:
        ref = flat(load(os.path.join(REF, 'stage_config.json'))[0])
        new = flat(sc)
        allowed = {'run.seed': seed, 'run.label': label}
        diff = sorted(k for k in set(ref) | set(new) if ref.get(k) != new.get(k))
        unexpected = [k for k in diff if k not in allowed]
        res['diffs']['stage_config'] = {k: {'s52': ref.get(k), 'new': new.get(k)} for k in diff}
        check('stage_config equals s52 except run.seed/run.label', not unexpected, unexpected or 'only run.seed/run.label differ')
        check('git_commit', new.get('git_commit') == COMMIT, new.get('git_commit'))
        check('run.seed', new.get('run.seed') == seed, new.get('run.seed'))
        check('run.label (report only in place5r; checked here)', new.get('run.label') == label, new.get('run.label'))
        check('run.n_envs / run.timesteps', new.get('run.n_envs') == 4 and new.get('run.timesteps') == 12000000, [new.get('run.n_envs'), new.get('run.timesteps')])
        check('hyperparameters_sha256', new.get('run.hyperparameters_sha256') == HP, new.get('run.hyperparameters_sha256'))
        check('task_fingerprint.task_sha256 (in stage_config)', new.get('task_fingerprint.task_sha256') == TASK, new.get('task_fingerprint.task_sha256'))
        check('gpu L4', new.get('gpu.gpu_model') == 'L4', {k: v for k, v in new.items() if k.startswith('gpu.')})
    for f in ('task_fingerprint.json', 'plant_identity.json'):
        o, err = load(os.path.join(d, f))
        if o is None:
            check(f, False, err)
            continue
        ref = flat(load(os.path.join(REF, f))[0])
        new = flat(o)
        diff = sorted(k for k in set(ref) | set(new) if ref.get(k) != new.get(k))
        res['diffs'][f] = {k: {'s52': ref.get(k), 'new': new.get(k)} for k in diff}
        check(f'{f} equals s52', not diff, diff or 'identical')
    pg, err = load(os.path.join(d, 'progress.json'))
    if pg is None:
        check('progress.json', False, err)
    else:
        res['readouts']['progress'] = {k: pg.get(k) for k in ('status', 'written_at', 'run_id', 'run_dir', 'seed', 'timesteps', 'budget_timesteps', 'fraction_done', 'session_started_at', 'steps_per_s', 'projected_finish', 'zero_action_reward', 'warnings_total')}
        res['readouts']['progress']['warnings'] = pg.get('warnings')
        check('progress run_id/seed', pg.get('run_id') == run and pg.get('seed') == seed, [pg.get('run_id'), pg.get('seed')])
        check('progress budget 12,000,000', pg.get('budget_timesteps') == 12000000, pg.get('budget_timesteps'))
        zr = pg.get('zero_action_reward')
        check('zero_action_reward 4367.8', zr is not None and abs(zr - 4367.8) < 0.05, zr)
    pv, err = load(os.path.join(d, 'provenance.json'))
    if pv is None:
        check('provenance.json', False, err)
    else:
        fp = flat(pv)
        res['readouts']['provenance'] = fp
        hits = {k: v for k, v in fp.items() if any(s in k.lower() for s in ('commit', 'dirty', 'seed', 'run_id', 'n_envs', 'mujoco', 'physics', 'label', 'ref'))}
        res['readouts']['provenance_key_fields'] = hits
        commits = [v for k, v in fp.items() if 'commit' in k.lower() and isinstance(v, str) and len(v) >= 7]
        check('provenance commit cf80ecb1', bool(commits) and all(c.startswith(COMMIT[:len(c)]) or COMMIT.startswith(c) for c in commits), commits)
        dirty = [v for k, v in fp.items() if 'dirty' in k.lower()]
        check('provenance not dirty', bool(dirty) and not any(dirty), dirty)
        # training_seed and seed_roles.training are the run's seed; the evaluation seeds are the notebook's
        # SEED + 1000 (checkpoint selection) and SEED + 3000 (publication), and the certification panel is fixed.
        check('provenance training seed', fp.get('training_seed') == seed and fp.get('seed_roles.training') == seed,
              [fp.get('training_seed'), fp.get('seed_roles.training')])
        check('provenance evaluation seeds SEED+1000 / SEED+3000',
              fp.get('seed_roles.checkpoint_selection_evaluation') == seed + 1000
              and fp.get('seed_roles.publication_evaluation') == seed + 3000,
              [fp.get('seed_roles.checkpoint_selection_evaluation'), fp.get('seed_roles.publication_evaluation')])
        check('provenance plant r9 / interface r15',
              fp.get('plant_identity.physics_revision') == 9 and fp.get('plant_identity.policy_interface_revision') == 15,
              [fp.get('plant_identity.physics_revision'), fp.get('plant_identity.policy_interface_revision')])
        res['readouts']['provenance_versions'] = {k: v for k, v in fp.items() if k.startswith('dependency_versions.')
                                                   or k in ('python_version', 'platform', 'parallel_envs', 'captured_at')}
        rids = [v for k, v in fp.items() if k.lower().endswith('run_id')]
        check('provenance run_id', bool(rids) and all(r == run for r in rids), rids)
    res['ok'] = all(c['ok'] for c in res['checks'])
    print(json.dumps(res, indent=1, default=str))
    return 0 if res['ok'] else 1


if __name__ == '__main__':
    sys.exit(main())
