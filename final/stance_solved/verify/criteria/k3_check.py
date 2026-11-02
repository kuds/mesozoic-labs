"""Read-only: independent recount of the in-run K3 panel falls at 6,291,456 and 12,001,280 (F2 copies), plus the
panel's column names, to check the criteria note's k3_gv.out."""
import csv
B = '/home/user/work/pc_study/round/fetch/F2/stance7_r10LTSBpc_%s/copies/01_stance/k3_panel.csv.copy1'
for s in ('s60', 's42', 's80'):
    rows = list(csv.DictReader(open(B % s, newline='')))
    if s == 's60':
        print('columns', list(rows[0].keys())[:20])
    for t in ('6291456', '12001280'):
        for test in ('joint', 'own'):
            sel = [r for r in rows if r['timesteps'] == t and r['test'] == test]
            print(s, t, test, 'falls', sum(int(r['fall']) for r in sel), 'of', len(sel))
