import json, csv, hashlib, os, sys
S = sys.argv[1]
mine = json.load(open(os.path.join(S, 'verify/rescore/rescore.json')))
sc = json.load(open(os.path.join(S, 'scores/score.json')))
P = {(r['name'], r['label']): r for r in csv.DictReader(open(os.path.join(S, 'panels.csv')))}
diffs = []
for n, r in mine['replicates'].items():
    s = sc['replicates'][n]
    m = json.load(open(os.path.join(S, 'runs', n, 'run_meta.json')))
    fz = m['final_sha256']['stage1_final.zip']; fp = m['final_sha256']['stage1_final_vecnorm.pkl']
    pz = hashlib.sha256(open(os.path.join(S, 'runs', n, 'stage1_final_vecnorm.pkl'), 'rb').read()).hexdigest()
    p1 = hashlib.sha256(open(os.path.join(S, 'runs', n, 'stage1_vecnormalize_5111808_steps.pkl'), 'rb').read()).hexdigest()
    ok = (P[(n, 'final')]['zip_sha256'] == fz and P[(n, 'final')]['pkl_sha256'] == fp == pz and P[(n, '5111808')]['pkl_sha256'] == p1)
    if not ok: diffs.append((n, 'sha'))
    for a, b, k in [(r['class'], s['class'], 'class'), (r['tailup_R1R2'], s['tailup_R1R2'], 'tu'), (r['other_R1R2'], s['other_R1R2'], 'oth'),
                    (r['onset_any'], s['onset_any_tailup'], 'onset'), (r['n_labels'], s['labels_measured'], 'nlab')]:
        if a != b: diffs.append((n, k, a, b))
    for a, b, k in [(r['my_5111808']['cmd'], s['cmd_R1'], 'cmdR1'), (r['my_final']['cmd'], s['cmd_R2'], 'cmdR2'), (r['cmd_mean'], s['cmd_mean'], 'mean')]:
        if abs(a - b) > 1e-9: diffs.append((n, k, a, b))
    # check run meta arm/coef/b/seed/state
    st, arm, rs = n.split('_')
    exp = {'control': (0.0, 0.9), 'b90': (2.4, 0.9), 'b60': (2.4, 0.6)}[arm]
    if (m['arm'], float(m['coef']), float(m['b']), int(m['rep_seed']), m['state']) != (arm, exp[0], exp[1], int(rs[1:]), st): diffs.append((n, 'meta', m['arm'], m['coef'], m['b']))
    for lab in ('5111808', 'final'):
        row = P[(n, lab)]
        if (float(row['coef']), float(row['b'])) != exp or row['seeds'] != '3042-3061' or row['n_episodes'] != '20' or row['measure_tree_digest'] != 'cf12a36b865a9172361c42372b31425d5c3545591447c67bb27e10ce8d5e3bc2': diffs.append((n, lab, 'rowmeta'))
for st in ('s42', 's80'):
    for B in ('BOUND90', 'BOUND60'):
        a = mine['states'][st][B]; b = sc['states'][st][B]
        if a['reading'] != b['reading'] or a['lines'] != b['lines_applying'] or a['other_b'] != b['other_bound'] or a['other_c'] != b['other_control']:
            diffs.append((st, B, a, b))
for B in ('BOUND90', 'BOUND60'):
    v = sc['verdicts'][B]; vs = v['verdict'] + (f" ({v['reason']})" if v.get('reason') else '')
    if vs != mine['verdicts'][B]: diffs.append((B, vs, mine['verdicts'][B]))
if sc['outcome']['line'] != mine['outcome']: diffs.append(('outcome', sc['outcome']['line'], mine['outcome']))
print('DIFFS', diffs)
print('score status', sc['status'], sc['provisional_reasons'], sc['incomplete'], sc['lost'])
# half point check
hp = sc['report_only']['half_point']
bad = [(n, l) for n in hp for l in hp[n] if abs(hp[n][l]['cmd'] - float(P[(n, l)]['cmd'])) > 1e-12 or hp[n][l]['tailup_flags'] != int(P[(n, l)]['tailup_flags'])]
print('halfpoint mismatches', bad)
# dose ordering check
for st in ('s42', 's80'):
    am = {a: sum(mine['replicates'][f'{st}_{a}_r{s}']['cmd_mean'] for s in (101, 202, 303)) / 3 for a in ('control', 'b90', 'b60')}
    print(st, am, am['b60'] <= am['b90'] <= am['control'], sc['report_only']['dose_ordering'][st]['ordered_b60_le_b90_le_control'])
# single-checkpoint class check across all rows
bad = []
for (n, l), r in P.items():
    c = 'RAIL' if int(r['tailup_flags']) >= 1 else ('PARKED' if float(r['cmd']) >= 0.60 else 'LOW')
    if c != r['class_single']: bad.append((n, l, c, r['class_single']))
print('class_single mismatches', bad)
# s80 control onset detail
for s in (101, 202, 303):
    n = f's80_control_r{s}'
    print(n, [P[(n, l)]['tailup_flags'] for l in ['4325376','4456448','4587520','4718592','4849664','4980736','5111808','5242880','final']])
