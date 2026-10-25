import gzip, csv, io, os, sys, glob
import numpy as np
S = sys.argv[1]
out = []
for d in sorted(glob.glob(os.path.join(S, 'panels', '*', '5111808')) + glob.glob(os.path.join(S, 'panels', '*', 'final'))):
    for f in sorted(glob.glob(os.path.join(d, '*.csv.gz'))):
        rd = csv.reader(io.StringIO(gzip.open(f).read().decode())); h = next(rd); i = h.index('tail_3_pitch_act_command'); j = h.index('in_window')
        c = np.array([float(r[i]) for r in rd if r[j] == '1'])
        sh = {t: float(np.mean(c >= t)) for t in (0.985, 0.99, 0.995)}
        strict = float(np.mean(c > 0.99)); at1 = float(np.mean(c >= 1.0)); between = int(np.sum((c >= 0.99) & (c < 1.0)))
        if 0.07 < sh[0.99] < 0.14 or any((v > 0.10) != (sh[0.99] > 0.10) for v in list(sh.values()) + [strict]):
            out.append((d.split('/panels/')[1], os.path.basename(f), sh, strict, at1, between))
for o in out: print(o)
