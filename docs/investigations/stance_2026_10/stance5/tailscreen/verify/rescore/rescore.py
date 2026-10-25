# Independent re-score from criteria.txt alone (does not read score_tailscreen.py).
import csv, gzip, hashlib, io, json, math, os, sys
import numpy as np
S = sys.argv[1]
OUT = os.path.join(S, 'verify', 'rescore')
R1, R2 = '5111808', 'final'
LABELS = ['4325376','4456448','4587520','4718592','4849664','4980736','5111808','5242880','final']
rows = list(csv.DictReader(open(os.path.join(S, 'panels.csv'))))
P = {}
dups = []
for r in rows:
    k = (r['name'], r['label'])
    if k in P: dups.append(k)
    P[k] = r
names = sorted({r['name'] for r in rows})

def sha(p):
    h = hashlib.sha256()
    with open(p, 'rb') as f:
        for b in iter(lambda: f.read(1 << 20), b''): h.update(b)
    return h.hexdigest()

def zpath(name, label):
    d = os.path.join(S, 'runs', name)
    return os.path.join(d, 'stage1_final.zip' if label == 'final' else f'stage1_{label}_steps.zip')

ACT = None
def episode_stats(csvtext):
    rd = csv.reader(io.StringIO(csvtext))
    hdr = next(rd)
    data = np.array([[float(x) if x != '' else np.nan for x in row] for row in rd])
    col = {h: i for i, h in enumerate(hdr)}
    L = data.shape[0]
    step = data[:, col['step']]
    win = (step >= 200) & (step <= L - 1)
    inw = data[:, col['in_window']] == 1
    acts = [h[:-len('_command')] for h in hdr if h.endswith('_command')]
    c3 = data[win, col['tail_3_pitch_act_command']]
    tailup_share = float(np.mean(c3 >= 0.99))
    harm = 0.0; harm_act = None
    for a in acts:
        c = data[win, col[a + '_command']]
        s = float(np.mean(c <= -0.99)) if a == 'tail_3_pitch_act' else float(np.mean(np.abs(c) >= 0.99))
        if s > harm: harm, harm_act = s, a
    return dict(L=L, win_ok=bool(np.array_equal(win, inw)), nwin=int(win.sum()),
                cmd=float(np.mean(c3)), tailup_share=tailup_share, tailup=tailup_share > 0.10,
                harm=harm, harm_act=harm_act, other=(harm > 0.10) or (L < 1000),
                cmd_max=float(np.nanmax(data[:, col['tail_3_pitch_act_command']])),
                cmd_min=float(np.nanmin(data[:, col['tail_3_pitch_act_command']])))

def panel_recompute(name, label):
    d = os.path.join(S, 'panels', name, label)
    shas = {}
    for line in open(os.path.join(d, 'csv_sha256.txt')):
        h, f = line.split(); shas[f] = h
    eps = []; sha_ok = True
    for seed in range(3042, 3062):
        raw = gzip.open(os.path.join(d, f'episode_trace_seed{seed}.csv.gz')).read()
        if hashlib.sha256(raw).hexdigest() != shas.get(f'episode_trace_seed{seed}.csv'): sha_ok = False
        e = episode_stats(raw.decode()); e['seed'] = seed; eps.append(e)
    return eps, sha_ok

res = {'dups': dups, 'n_rows': len(rows), 'names': names, 'replicates': {}, 'checks': []}
reps = {}
for n in names:
    st, arm, rs = n.split('_')
    rm = json.load(open(os.path.join(S, 'runs', n, 'run_meta.json')))
    rec = {'state': st, 'arm': arm, 'seed': int(rs[1:]), 'labels': sorted([l for (nn, l) in P if nn == n])}
    for lab in (R1, R2):
        r = P[(n, lab)]
        zs = sha(zpath(n, lab))
        rec[f'zip_ok_{lab}'] = (zs == r['zip_sha256'])
        if lab == 'final':
            rec['final_meta_ok'] = (rm.get('final_sha256') == zs) if 'final_sha256' in rm else ('final_sha256 missing', [k for k in rm if 'sha' in k])
        eps, sok = panel_recompute(n, lab)
        rec[f'csvsha_ok_{lab}'] = sok
        rec[f'my_{lab}'] = dict(cmd=float(np.mean([e['cmd'] for e in eps])),
                                tailup=sum(e['tailup'] for e in eps), other=sum(e['other'] for e in eps),
                                win_ok=all(e['win_ok'] for e in eps), Ls=sorted({e['L'] for e in eps}),
                                max_tailup_share_unflagged=max([e['tailup_share'] for e in eps if not e['tailup']] or [0]),
                                min_tailup_share_flagged=min([e['tailup_share'] for e in eps if e['tailup']] or [1]),
                                max_harm=max(e['harm'] for e in eps),
                                cmd_range=(min(e['cmd_min'] for e in eps), max(e['cmd_max'] for e in eps)))
        rec[f'ep_cmd_{lab}'] = [e['cmd'] for e in eps]
        rec[f'csv_{lab}'] = dict(cmd=float(r['cmd']), tailup=int(r['tailup_flags']), other=int(r['other_flags']),
                                 coef=r['coef'], b=r['b'], seeds=r['seeds'], n=r['n_episodes'], tree=r['measure_tree_digest'], stack=r['stack'])
    # onset: tail-up at any of 9 checkpoints (panels.csv)
    rec['onset_any'] = any(int(P[(n, l)]['tailup_flags']) >= 1 for l in LABELS if (n, l) in P)
    rec['n_labels'] = sum((n, l) in P for l in LABELS)
    tu = rec[f'my_{R1}']['tailup'] + rec[f'my_{R2}']['tailup']
    oth = rec[f'my_{R1}']['other'] + rec[f'my_{R2}']['other']
    m = (rec[f'my_{R1}']['cmd'] + rec[f'my_{R2}']['cmd']) / 2
    rec['tailup_R1R2'] = tu; rec['other_R1R2'] = oth; rec['cmd_mean'] = m
    rec['class'] = 'RAIL' if tu >= 1 else ('PARKED' if m >= 0.60 else 'CURED')
    # same from panels.csv numbers only
    tu2 = rec[f'csv_{R1}']['tailup'] + rec[f'csv_{R2}']['tailup']
    m2 = (rec[f'csv_{R1}']['cmd'] + rec[f'csv_{R2}']['cmd']) / 2
    rec['class_csv'] = 'RAIL' if tu2 >= 1 else ('PARKED' if m2 >= 0.60 else 'CURED')
    rec['cmd_mean_csv'] = m2
    # paired SE: per-episode mean of R1,R2 -> sd/sqrt(20)
    a1 = np.array(rec[f'ep_cmd_{R1}']); a2 = np.array(rec[f'ep_cmd_{R2}'])
    pe = (a1 + a2) / 2
    rec['se_paired'] = float(pe.std(ddof=1) / math.sqrt(20))
    rec['se_20_R1'] = float(a1.std(ddof=1) / math.sqrt(20)); rec['se_20_R2'] = float(a2.std(ddof=1) / math.sqrt(20))
    rec['se_40_indep'] = float(np.concatenate([a1, a2]).std(ddof=1) / math.sqrt(40))
    rec['corr_R1R2'] = float(np.corrcoef(a1, a2)[0, 1])
    rec['margin'] = m - 0.60
    rec['margin_in_paired_SE'] = (m - 0.60) / rec['se_paired']
    reps[n] = rec
res['replicates'] = reps

ARMB = {'BOUND90': 'b90', 'BOUND60': 'b60'}
states = {}
for st in ('s42', 's80'):
    ctrl = [reps[f'{st}_control_r{s}'] for s in (101, 202, 303)]
    oc = sum(r['other_R1R2'] for r in ctrl)
    for B, a in ARMB.items():
        br = [reps[f'{st}_{a}_r{s}'] for s in (101, 202, 303)]
        ob = sum(r['other_R1R2'] for r in br)
        lines = []
        if sum(r['class'] == 'RAIL' for r in br) >= 1: lines.append('RAIL')
        if sum(r['class'] == 'PARKED' for r in br) >= 2: lines.append('PARKED')
        if ob > oc: lines.append('OTHER')
        if st == 's42' and sum(r['class'] == 'CURED' for r in ctrl) >= 2: lines.append('NOT SEPARATED')
        lines.append('PASS')
        states.setdefault(st, {})[B] = dict(reading=lines[0], lines=lines[:-1] if len(lines) > 1 else [], other_b=ob, other_c=oc)
res['states'] = states
s80c = [reps[f's80_control_r{s}'] for s in (101, 202, 303)]
onset = 'INFORMATIVE' if sum(r['onset_any'] for r in s80c) >= 2 else 'onset not tested'
res['onset'] = (onset, [r['n_labels'] for r in s80c], sum(r['onset_any'] for r in s80c))
verd = {}
for B in ARMB:
    rd = [states[s][B]['reading'] for s in ('s42', 's80')]
    if 'RAIL' in rd: v = 'NOT ADOPTED (RAIL)'
    elif 'PARKED' in rd: v = 'NOT ADOPTED (PARKED)'
    elif 'OTHER' in rd: v = 'NOT ADOPTED (OTHER)'
    elif states['s42'][B]['reading'] == 'NOT SEPARATED': v = 'NOT ADOPTED (NOT SEPARATED)'
    else: v = 'PASSED'
    verd[B] = v
res['verdicts'] = verd
np_ = sum(v == 'PASSED' for v in verd.values())
res['outcome'] = 'A' if np_ == 2 else ('B' if np_ == 1 else 'C')
for n, r in reps.items():
    for k in list(r):
        if k.startswith('ep_cmd'): r[k] = [round(x, 6) for x in r[k]]
json.dump(res, open(os.path.join(OUT, 'rescore.json'), 'w'), indent=1, default=str)
for n in names:
    r = reps[n]
    print(f"{n:18s} {r['class']:7s} csv:{r['class_csv']:7s} tu {r['tailup_R1R2']:2d}({r[f'csv_{R1}']['tailup']+r[f'csv_{R2}']['tailup']:2d}) oth {r['other_R1R2']}({r[f'csv_{R1}']['other']+r[f'csv_{R2}']['other']}) "
          f"cmd {r[f'my_{R1}']['cmd']:+.4f}/{r[f'my_{R2}']['cmd']:+.4f} m {r['cmd_mean']:+.5f} (csv {r['cmd_mean_csv']:+.5f}) marg {r['margin']:+.4f} SEp {r['se_paired']:.4f} z {r['margin_in_paired_SE']:+.2f} "
          f"SE20 {r['se_20_R1']:.3f}/{r['se_20_R2']:.3f} SE40 {r['se_40_indep']:.3f} rho {r['corr_R1R2']:+.2f} onset {int(r['onset_any'])} nlab {r['n_labels']} "
          f"zip {r['zip_ok_'+R1]}/{r['zip_ok_final']} meta {r['final_meta_ok']} csvsha {r['csvsha_ok_'+R1]}/{r['csvsha_ok_final']} win {r['my_'+R1]['win_ok']}/{r['my_final']['win_ok']} L {r['my_'+R1]['Ls']}{r['my_final']['Ls']} "
          f"maxunfl {max(r['my_'+R1]['max_tailup_share_unflagged'], r['my_final']['max_tailup_share_unflagged']):.4f} minfl {min(r['my_'+R1]['min_tailup_share_flagged'], r['my_final']['min_tailup_share_flagged']):.4f} maxharm {max(r['my_'+R1]['max_harm'], r['my_final']['max_harm']):.4f} range {r['my_final']['cmd_range']}")
print(json.dumps(states, indent=0)); print(res['onset']); print(verd); print('OUTCOME', res['outcome']); print('dups', dups, 'rows', len(rows))
