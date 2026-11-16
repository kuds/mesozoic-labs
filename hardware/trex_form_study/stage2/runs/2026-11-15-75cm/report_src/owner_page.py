# ruff: noqa: E501, E402, E702, E731
"""Build the owner page for the 75 cm panels from the reviewed report.

The page's text is the report's own sections (the short answer, the turning room, what it does not show, the
questions), so the page cannot say anything the report does not.  The clips come from the committed manifest
stage2/runs/2026-11-15-75cm/views/clips.json; their media (git-ignored) are inlined as data URIs.

    python owner_page.py OUT.html
"""
import base64
import json
import re
import sys
from pathlib import Path

STUDY = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(STUDY))
sys.path.insert(0, str(STUDY / 'tools'))
import report_page  # noqa: E402

RUN = STUDY / 'stage2/runs/2026-11-15-75cm'
MEDIA = STUDY / 'logs/barviews'
HERE = Path(__file__).resolve().parent
OUT_DIR = Path(sys.argv[1]).resolve().parent  # the intermediates (the page's Markdown, the JPEG posters) go beside OUT
report = (STUDY / 'STAGE2_75_REPORT.md').read_text()


def section(title_start: str) -> str:
    i = report.index(title_start)
    m = re.search(r'\n## ', report[i + 4:])
    j = i + 4 + m.start() if m else len(report)
    body = report[i:j]
    return body[body.index('\n') + 1:].strip()


manifest = json.loads((RUN / 'views/clips.json').read_text())
digest = json.loads((RUN / 'digest/digest.json').read_text())['records']


def clip(set_, cond, row, view, kind='simulated'):
    for c in manifest['clips']:
        if (c['set'], c['condition'], c['view'], c['kind']) == (set_, cond, view, kind) and (kind == 'planned' or c['row'] == row):
            return c
    raise SystemExit(f'no clip for {set_} {cond} {row} {view} {kind}')


WALKS = {'V0': "today's walk", 'F25F9cl': 'the candidate walk'}


SEGMENT = {'start': 'in the start from standing', 'start_v0.4': 'in the start from standing',
           'nominal': 'in the first slow stride', 'nominal_to_v0.4': 'while speeding up to 0.40 m/s',
           'v0.4': 'in the 0.40 m/s walk'}
PASS_RULE = 'a pass needs all 20 up and at least 17 meeting every limit'


def caption(c) -> str:
    view = 'above' if c['view'] == 'top' else 'a visitor’s height'
    if c['kind'] == 'planned':
        return (f"<strong>Planned motion, no physics.</strong> {c['in_words'].capitalize()}, {WALKS[c['set']]}: the "
                f"reference played by itself, seen from {view}. {c['seconds']:.0f} s.")
    row, delay = c['row'].split('@')
    beh, cond = c['condition'].split('/')
    d = digest['|'.join((c['set'], c['block'], row, delay.replace('ms', ''), beh, cond))]
    rec = json.loads((RUN / c['record']).read_text()) if c.get('record') and (RUN / c['record']).exists() else \
        json.loads((RUN / d['record']).read_text())
    ep = next(e for e in rec['episodes'] if e['seed'] == c['seed'])
    fit = row.split('/')[1]
    if ep['fell']:
        shown = (f"Run shown: seed {c['seed']} ({c['selection']}): it fell {ep['fall_time_s']:.1f} s into the run, "
                 f"{SEGMENT.get(ep.get('segment_reached'), 'before its move')}. The picture stops at the instant the "
                 f"fall is counted and holds the last pose.")
    else:
        shown = f"Run shown: seed {c['seed']} ({c['selection']}): it stayed up."
    if d.get('a1_over'):
        counts = (f"In this record {d['up']} of 20 runs stayed up and {d['but_A1']} of 20 met every limit with the tail "
                  f"servo's limit left out (the only reading the rules allow for this turn)")
        passed = d['up'] == 20 and d['but_A1'] >= 17
    else:
        counts = (f"In this record {d['up']} of 20 runs stayed up and {d['every_limit']} of 20 met every limit "
                  f"({d['but_A1']} of 20 with the tail servo's limit left out)")
        passed = d['up'] == 20 and d['every_limit'] >= 17
    verdict = 'That is a pass at this delay.' if passed else f"That is not a pass ({PASS_RULE})."
    words = re.sub(r'nudge (\w+) \(0\.25 of the push bound\)', r'a light push to its \1 (about 22 N for 0.1 s)', c['in_words'])
    words = words[0].upper() + words[1:]
    return (f"<strong>Simulation.</strong> {words}, {WALKS[c['set']]}, controller fit {fit}, "
            f"{delay.replace('ms', ' ms')} of delay, seen from {view}. {shown} {counts}. {verdict} {c['seconds']:.0f} s.")


def data_uri(path: Path, mime: str) -> str:
    return f'data:{mime};base64,' + base64.b64encode(path.read_bytes()).decode()


def jpeg(png: Path) -> Path:
    """The poster as a JPEG (a tenth of the PNG's bytes), made once in scratch."""
    import subprocess
    out = OUT_DIR / 'posters' / (png.stem + '.jpg')
    out.parent.mkdir(exist_ok=True)
    if not out.exists() or out.stat().st_mtime < png.stat().st_mtime:
        subprocess.run(['ffmpeg', '-y', '-loglevel', 'error', '-i', str(png), '-q:v', '4', str(out)], check=True)
    return out


def video(c) -> str:
    return (f'<figure class="clip"><div class="frame"><video controls playsinline preload="metadata" '
            f'poster="{data_uri(jpeg(MEDIA / c["poster"]), "image/jpeg")}" src="{data_uri(MEDIA / c["file"], "video/mp4")}">'
            f'</video></div><figcaption><span>{caption(c)}</span></figcaption></figure>')


CLIPS = [
    ("Today's walk with no delay: walking, turning, a push, stopping",
     [('V0', 'walk/v0.4', 'R2/42@0ms', 'visitor'), ('V0', 'turn/q90_left', 'R2/42@0ms', 'visitor'),
      ('V0', 'turn/q90_left', 'R2/42@0ms', 'top'), ('V0', 'nudge/left', 'R2/42@0ms', 'visitor'),
      ('V0', 'stop/two_step', 'R2/42@0ms', 'visitor')], 'simulated'),
    ("At the judged 10 ms of delay: the fall before the walk begins",
     [('V0', 'walk/v0.4', 'R2/42@10ms', 'visitor'), ('F25F9cl', 'walk/v0.4', 'R2/42@10ms', 'visitor')], 'simulated'),
    ("The candidate walk's quick turn with no delay",
     [('F25F9cl', 'turn/q90_left', 'R2/42@0ms', 'visitor')], 'simulated'),
    ("The quick turn as planned",
     [('V0', 'turn/q90_left', None, 'visitor'), ('V0', 'turn/q90_left', None, 'top')], 'planned'),
]

clips_html = []
total = 0
for title, items, kind in CLIPS:
    clips_html.append(f'<h3>{title}</h3>')
    for s, cond, row, view in items:
        c = clip(s, cond, row, view, kind)
        total += c['bytes']
        clips_html.append(video(c))
clips_block = '\n'.join(clips_html)

status = (HERE / 'PAGE_STATUS.md').read_text().strip()
md = f"""# The 75 cm build in simulation: can it walk?

{status}

## The short answer

{section('## 1. The short answer')}

## Longer delays, added 16 Nov

{section('## 13. Added 16 Nov 2026')}

## Clips

Every clip is labelled on its frames as simulation or as planned motion. **Which records have a clip** was the
lead's choice, made from the tables: today's walk under the first fit of the controller on the grippy floor, at the
lowest delay run (walk, quick turn, a light push to its left, stop) and at the judged 10 ms (walk); the candidate's
walk at 10 ms and its quick turn with no delay; and the planned quick turn. **Which run of a record is shown**
follows a rule written before any record existed: the first seed, and if its outcome is not the record's majority
outcome, also the lowest seed with the majority outcome.

Ten of the report's 18 clips are shown here. The other eight are the same runs from a second camera and the
candidate's planned turn. **No clip is shown of a push at 4 or 6 ms of delay**, where a push fells today's walk in
up to 11 of 20 runs, so the push clip below is the kindest case. Watch full screen with the phone on its side: the
labels inside the picture are too small to read on an upright phone, and the line under each clip repeats what
they say.

CLIPS_PLACEHOLDER

## Turning room

{section('## 10. Turning room').split(chr(10) + '**Clips.**')[0].strip()}

## What this does not show

{section('## 2. What this does not show')}

## For you to decide (answered 16 Nov)

**You answered these five questions on 16 Nov** (`DECISION_LOG.md` P37): test 14 to 20 ms and then refit; the
counter-phased tail; no more runs on the candidate; close the pull request as superseded; carpet as a range with no
verdict. The text below is left as it was asked.

{section('## 3. For the owner')}

## Where the rest is

Section numbers above refer to the full report, `hardware/trex_form_study/STAGE2_75_REPORT.md` on the branch
`claude/mesozoic-trex-stage2-refine-4jffre`. Its section 10 is "Turning room" on this page, and its clips are under
"Clips". It holds every table, the candidate's recount, the pull request's
claims line by line, and the draft decision-log rows. The machine-made tables with every record path are
`stage2/runs/2026-11-15-75cm/BAR_TABLES.md`.
"""
src = OUT_DIR / 'OWNER_PAGE.md'
src.write_text(md)
html = report_page.build(src, kicker='Mesozoic Labs robotic T. rex · Stage 2 · simulation only',
                         lede='First panels of the agility bar on the 75 cm body: what stays up, what passes, how much room a turn takes, and five decisions for you.',
                         title='75 cm Walk Results',
                         footer='Built from STAGE2_75_REPORT.md and the committed records of stage2/runs/2026-11-15-75cm. Simulation only; nothing here was measured on hardware and nothing is adopted.')
html = html.replace('<p>CLIPS_PLACEHOLDER</p>', clips_block)
html, n_sec = re.subn(r'\s+'.join(map(re.escape, 'The figure, the table and the clips are in section 10.'.split())),
                      'The figure and the tables are under “Turning room” below, the clips under “Clips”.', html)
assert n_sec == 1, n_sec
# the report's figure and the italic line under it become one figure with a caption (so the tap-to-zoom applies)
html, n_fig = re.subn(r'<p><img\s+src="([^"]+)"\s+alt="([^"]*)"\s*/?></p>\s*<p><em>(.*?)</em></p>',
                      lambda m: (f'<figure><div class="frame"><img src="{m.group(1)}" alt="{m.group(2)}"></div>'
                                 f'<figcaption><span>{m.group(3)}</span></figcaption></figure>'), html, flags=re.S)
assert n_fig == 1, n_fig
# the first table (runs that stayed up, by delay) keeps its pairs on one line
html = html.replace('<div class="tablewrap"><table>', '<div class="tablewrap tight"><table>', 1)
html = html.replace('<div class="tablewrap"><table>', '<p class="hint">Slide a wide table sideways for its other columns.</p><div class="tablewrap"><table>', 1)
assert 'CLIPS_PLACEHOLDER' not in html
# inline the figure(s) the report links by path
def inline_img(m):
    p = STUDY / m.group(1)
    return f'src="{data_uri(p, "image/png")}"'
html = re.sub(r'src="(stage2/[^"]+\.png)"', inline_img, html)
extra = ('<style>figure.clip{margin:6px 0 14px}'
         'figure.clip video{display:block;width:auto;height:auto;max-width:100%;max-height:88vh;margin-inline:auto;background:#000}'
         '.report .tablewrap th{white-space:normal;min-width:6.5em;vertical-align:bottom}'
         '.report .tablewrap td{white-space:normal;min-width:5.5em}'
         '.report .tablewrap.tight th{white-space:normal;min-width:0}.report .tablewrap.tight td{white-space:nowrap;min-width:0}'
         '.report .hint{font-size:13px;color:var(--ink-2);margin:0}'
         '#tip[hidden]{display:none}footer p{color:var(--ink-2)}</style>')
html = html.replace('<article class="report">', extra + '<article class="report">', 1)
out = Path(sys.argv[1])
out.write_text(html)
print(out, f'{len(html) / 1e6:.2f} MB; clips {total / 1e6:.2f} MB before base64; videos {html.count("<video")}')
