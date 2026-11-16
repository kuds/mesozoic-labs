# ruff: noqa: E501, E402, E702, E731, E741
"""Fill the report's table placeholders from the run's digest and re-reading (verbatim rows; no number typed by hand)."""
import json
import re
import sys
from pathlib import Path

STUDY = Path(__file__).resolve().parents[4]
RUN = STUDY / 'stage2/runs/2026-11-15-75cm'
HERE = Path(__file__).resolve().parent
dig = (RUN / 'digest/DIGEST.md').read_text().split('\n')
aside = json.loads((RUN / 'rereads/joint_range_aside.json').read_text())

def section(title_start):
    i = next(k for k, l in enumerate(dig) if l.startswith(title_start))
    j = next((k for k in range(i + 1, len(dig)) if dig[k].startswith('## ')), len(dig))
    return [l for l in dig[i:j] if l.startswith('|')]

def table(lines, keep=lambda l: True):
    head, sep, body = lines[0], lines[1], lines[2:]
    return '\n'.join([head, sep] + [l for l in body if keep(l)])

walk09 = table(section('## 1. The walk at 0.40 m/s, grippy'))
walk054 = table(section('## 1. The walk at 0.40 m/s, slippery'))
med = table(section('## 2.'), lambda l: '`R2/' in l and re.search(r'\| (0|2|4|6) ms \|', l))
ladder = table(section('## 3.'), lambda l: l.startswith('| V0 ') or l.startswith('| F25F9cl '))
GATED = ('`turn/q90_left`', '`turn/q90_right`', '`nudge/forward`', '`nudge/backward`', '`nudge/left`', '`nudge/right`', '`stop/two_step`')
rest = table(section('## 4.'), lambda l: any(f'| {g} |' in l for g in GATED) and '`R2/' in l)
rest = rest.replace(' (falls ', ' (').replace(' b / ', ' b / ')

rows = {(x['set'], x['block'], x['behavior'], x['cond'], x['delay_ms'], x['row']): x for x in aside['records']}
L = ['| Walk (candidate) | Delay | Stayed up | Every limit, as registered | Every limit, joint range aside | Stage 1\'s reading, as recorded | Stage 1\'s reading, joint range aside |', '|---|---|---|---|---|---|---|']
names = {'nominal': '0.24 m/s (`nominal`)', 'v0.3': '0.30 m/s', 'v0.35': '0.35 m/s', 'v0.4': '0.40 m/s', 'v0.45': '0.45 m/s', 'v0.5': '0.50 m/s',
         'v0.4_mu0.54': '0.40 m/s, slippery floor', 'v0.45_mu0.54': '0.45 m/s, slippery floor', 'v0.5_mu0.54': '0.50 m/s, slippery floor'}
for cond, nm in names.items():
    for dl in (0, 2, 4, 6):
        a = rows.get(('F25F9cl', 'seeds3042', 'walk', cond, dl, 'R2/42')); b = rows.get(('F25F9cl', 'seeds3042', 'walk', cond, dl, 'R2/43'))
        if not a or not b:
            continue
        f = lambda k: f"{a[k]} · {b[k]}"
        L.append(f"| {nm} | {dl} ms | {f('up')} | {f('succeeded')} | {f('succeeded_term_aside')} | {f('stage1_succeeded')} | {f('stage1_succeeded_term_aside')} |")
aside_t = '\n'.join(L)

t = (HERE / 'template.md').read_text()
falls = table(section('## 7.'), lambda l: re.search(r'\| (6|8|10) ms \|', l))
dj = json.loads((RUN / 'digest/digest.json').read_text())['records']
S = ["| Delay | Today's walk | The pull request's walk | The candidate |", '|---|---|---|---|']
for d in (0, 2, 4, 6, 8, 10):
    cells = []
    for s_ in ('V0', 'F25F9', 'F25F9cl'):
        a_ = dj['|'.join((s_, 'seeds3042', 'R2/42', str(d), 'walk', 'v0.4'))]; b_ = dj['|'.join((s_, 'seeds3042', 'R2/43', str(d), 'walk', 'v0.4'))]
        cells.append(f"{a_['up']} · {b_['up']}")
    S.append(f"| {d} ms{' (judged)' if d == 10 else ''} | " + ' | '.join(cells) + ' |')
small = '\n'.join(S)
fill = {'WALK_09': walk09, 'WALK_054': walk054, 'MEDIANS': med, 'LADDER': ladder, 'REST': rest, 'ASIDE': aside_t, 'FALLS': falls, 'SMALL': small}
for k in ('TURN_ANSWER', 'TURN_SECTION'):
    p = HERE / f'{k}.md'
    fill[k] = p.read_text().strip() if p.exists() else f'[to be added when the reviewed views are merged: {k}]'
fill['STATUS'] = (HERE / 'STATUS.md').read_text().strip()
# the add-on's section (STAGE2_75_ADDON_PLAN.md A-10): its tables and sentences come verbatim from the add-on's own script
ADD = STUDY / 'stage2/runs/2026-11-16-75cm-addon'
add = (ADD / 'DELAYS_TABLE.md').read_text().split('\n')
addj = json.loads((ADD / 'delays_table.json').read_text())
def add_table(title_start, which=0):
    i = next(k for k, l in enumerate(add) if l.startswith(title_start))
    j = next((k for k in range(i + 1, len(add)) if add[k].startswith('## ')), len(add))
    blocks, cur = [], []
    for l in add[i:j]:
        if l.startswith('|'):
            cur.append(l)
        elif cur:
            blocks.append(cur); cur = []
    if cur:
        blocks.append(cur)
    return '\n'.join(blocks[which])
fill['ADDON_FAST'] = add_table('## 1.')
fill['ADDON_SLOW'] = add_table('## 2.')
fill['ADDON_PATTERN_FAST'] = addj['pattern_fast']['sentence']
fill['ADDON_PATTERN_SLOW'] = addj['pattern_slow']['sentence']
m_ = addj['mirror']
fill['ADDON_MIRROR'] = ('The five predictions are ' + str(m_['result']) + ' on the delays that were run'
                        + ('' if not m_['differ'] else ': ' + ', '.join(f'{d} ms' for d in m_['differ']) + ' differ') + '.')
for k, v in fill.items():
    assert '{{' + k + '}}' in t, k
    t = t.replace('{{' + k + '}}', v)
assert '{{' not in t
out = Path(sys.argv[1]) if len(sys.argv) > 1 else STUDY / 'STAGE2_75_REPORT.md'
out.write_text(t)
print(out, len(t.split('\n')), 'lines')
