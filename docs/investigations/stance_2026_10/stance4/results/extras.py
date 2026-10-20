"""Report-only extras on each stance4 final: fresh panel (7042-7081), forward-start tail panel (32 seeds), common noise (40 seeds)."""
import json, os
S4 = 'scratchpad/foot/stance4'
def load(run, mode):
    f = f'{S4}/judged/{run}/judge_{mode}_final.json'
    return json.load(open(f)) if os.path.exists(f) else None
for run in ['stance4_LTS_s52', 'stance4_LTS_s60', 'stance4_LTS_s42']:
    out = [run]
    d = load(run, 'fresh')
    if d:
        p = d['panel']; falls = sum(v for k, v in p['terminations'].items() if k != 'truncated')
        out.append(f"fresh {p['verdict']} clean {p['n_clean']}/{p['episodes']} falls {falls} K2a {p['summary']['k2a_count']}")
    else:
        out.append('fresh pending')
    d = load(run, 'tail')
    out.append(f"tail falls {d['tail']['falls']}/{d['tail']['n']} (statue {d['tail']['statue_falls']})" if d else 'tail pending')
    d = load(run, 'common_noise')
    out.append(f"common-noise falls {d['common_noise']['falls']}/{d['common_noise']['n']}" if d else 'common-noise pending')
    print(' | '.join(out))
print('references: fresh stance3 Run 2 final 40/40, Run 3 final 39/40; tail statue 0, stance3 Run 2 final 0, Run 3 final 4 (of 32); common noise: no control reference measured')
