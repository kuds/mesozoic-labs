"""Hutchinson et al. 2011 Tables 4-5: do the segment masses sum to body mass with limbs counted once or twice?

    python -I vx_h2011_masses.py PMC3192160.xml OUT_TXT
"""
import sys
import xml.etree.ElementTree as ET
from collections import defaultdict

r = ET.parse(sys.argv[1]).getroot()
rows = []
for tw in r.iter('table-wrap'):
    lab = tw.find('label')
    if lab is None or lab.text not in ('Table 4', 'Table 5'):
        continue
    for tr in tw.iter('tr'):
        c = [' '.join(''.join(x.itertext()).split()) for x in tr]
        if len(c) >= 5 and c[0] != 'Specimen':
            rows.append((c[0], c[1], c[2], float(c[3]), float(c[4])))
seg = defaultdict(dict)
for sp, sg, mdl, mass, frac in rows:
    seg[(sp, mdl)][sg] = (mass, frac)
out = []
for sp in ('Carnegie', 'Sue', 'Stan', 'MOR', 'Jane'):
    for mdl in ('min', 'max'):
        s = dict(seg[(sp, 'all')]); s.update(seg[(sp, mdl)])
        body_mass_est = s['Body'][0] / s['Body'][1]  # body segment / its share
        single = sum(v[0] for v in s.values())
        limbs = sum(s[k][0] for k in ('Forelimb', 'Thigh', 'Shank', 'Foot'))
        double = single + limbs
        hl = sum(s[k][1] for k in ('Thigh', 'Shank', 'Foot'))
        out.append(f"{sp:8s} {mdl}: body mass from Body/share {body_mass_est:8.0f} kg | sum, limbs once {single:8.0f} "
                   f"({single / body_mass_est - 1:+.1%}) | limbs twice {double:8.0f} ({double / body_mass_est - 1:+.1%}) | "
                   f"thigh+shank+foot share {hl:.3f}, x2 {2 * hl:.3f}")
txt = "\n".join(out)
open(sys.argv[2], 'w').write(txt + "\n")
print(txt)
