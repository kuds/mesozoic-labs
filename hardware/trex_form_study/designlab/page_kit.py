"""Small HTML kit for the review page: theme, figures, tables and two SVG charts.

Used by ``designlab.review_page``.  No physics here.  Charts follow fixed mark specs (thin bars,
4 px rounded data ends, 2 px surface gaps, hover tooltips that never gate a value: every number
in a chart is also in a table on the page).
"""

from __future__ import annotations

import html

FONTS = ("https://fonts.googleapis.com/css2?family=Fraunces:opsz,wght@9..144,400;9..144,600&"
         "family=IBM+Plex+Mono:wght@400;500&family=Source+Sans+3:wght@400;600&display=swap")

CSS = r"""
:root{
  --bg:#f4f1e8;--surface:#fbf9f3;--surface-2:#eee9dc;--ink:#1c211d;--ink-2:#4a514a;--ink-3:#6b7068;
  --rule:#d8d1c0;--accent:#38553a;--accent-2:#8f6216;--wash:#e2e9d8;--ok:#2c6a37;--warn:#8f6216;--bad:#a2392a;
  --series-1:#2a78d6;--series-2:#eb6834;--series-3:#1baf7a;--band:#ddd6c5;--grid:#e3ddcf;--plate:#f6f5f1;
}
@media (prefers-color-scheme: dark){:root:not([data-theme="light"]){
  --bg:#121512;--surface:#1a1e1a;--surface-2:#232823;--ink:#ebe8df;--ink-2:#c4c3b8;--ink-3:#9b9e95;
  --rule:#343b34;--accent:#9dc596;--accent-2:#dcaa52;--wash:#233024;--ok:#8fc59a;--warn:#dcaa52;--bad:#e3806f;
  --series-1:#3987e5;--series-2:#d95926;--series-3:#199e70;--band:#2d332d;--grid:#2a302a;--plate:#e9e7e1;
  color-scheme:dark}}
:root[data-theme="dark"]{
  --bg:#121512;--surface:#1a1e1a;--surface-2:#232823;--ink:#ebe8df;--ink-2:#c4c3b8;--ink-3:#9b9e95;
  --rule:#343b34;--accent:#9dc596;--accent-2:#dcaa52;--wash:#233024;--ok:#8fc59a;--warn:#dcaa52;--bad:#e3806f;
  --series-1:#3987e5;--series-2:#d95926;--series-3:#199e70;--band:#2d332d;--grid:#2a302a;--plate:#e9e7e1;
  color-scheme:dark}
*{box-sizing:border-box}
body{background:var(--bg);color:var(--ink);font:16px/1.6 "Source Sans 3",system-ui,-apple-system,"Segoe UI",sans-serif;
  padding-left:max(16px,env(safe-area-inset-left,0px));padding-right:max(16px,env(safe-area-inset-right,0px));
  padding-block:0 64px}
.wrap{max-width:1120px;margin:0 auto}
header.top{padding-block:48px 28px;border-bottom:1px solid var(--rule)}
.kicker{font:500 12px/1.4 "IBM Plex Mono",ui-monospace,monospace;letter-spacing:.08em;text-transform:uppercase;color:var(--accent-2)}
h1,h2,h3{font-family:Fraunces,Georgia,"Times New Roman",serif;font-weight:600;text-wrap:balance;letter-spacing:-.01em;margin:0}
h1{font-size:clamp(30px,5vw,46px);line-height:1.1;margin-top:10px}
h2{font-size:clamp(23px,3.2vw,30px);line-height:1.2}
h3{font-size:19px;line-height:1.3}
p{margin:0}
.lede{font-size:18px;color:var(--ink-2);max-width:68ch;margin-top:14px}
.prose{display:grid;gap:12px;max-width:70ch}
.prose ul,.prose ol{margin:0;padding-left:1.2em;display:grid;gap:6px}
section{padding-block:40px 8px;display:grid;gap:18px;border-bottom:1px solid var(--rule)}
section:last-of-type{border-bottom:0}
.mono,code{font-family:"IBM Plex Mono",ui-monospace,monospace;font-size:.88em}
code{background:var(--surface-2);padding:1px 5px;border-radius:4px}
a{color:var(--accent)}
nav.toc{display:flex;flex-wrap:wrap;gap:6px 14px;margin-top:20px;font-size:14px}
nav.toc a{color:var(--ink-2);text-decoration:none;border-bottom:1px solid var(--rule)}
nav.toc a:hover{color:var(--accent);border-color:var(--accent)}
.facts{display:grid;grid-template-columns:repeat(auto-fit,minmax(200px,1fr));gap:12px;margin-top:22px}
.fact{background:var(--surface);border:1px solid var(--rule);border-radius:10px;padding:14px 16px;display:grid;gap:4px;min-width:0}
.fact b{font:600 22px/1.2 "Source Sans 3",system-ui,sans-serif}
.fact span{font-size:13.5px;color:var(--ink-2)}
figure{margin:0;display:grid;gap:8px;min-width:0}
figure .frame{background:var(--plate);border:1px solid var(--rule);border-radius:10px;overflow:hidden}
figure img{display:block;width:100%;height:auto;max-width:100%;cursor:zoom-in}
figcaption{font-size:14px;color:var(--ink-2);display:grid;gap:2px;max-width:90ch}
figcaption .plate{font:500 11.5px/1.4 "IBM Plex Mono",ui-monospace,monospace;letter-spacing:.06em;text-transform:uppercase;color:var(--ink-3)}
.grid2{display:grid;grid-template-columns:repeat(auto-fit,minmax(300px,1fr));gap:18px}
.tablewrap{overflow-x:auto;border:1px solid var(--rule);border-radius:10px;background:var(--surface)}
table{border-collapse:collapse;width:100%;font-size:14px}
th,td{padding:7px 10px;text-align:left;border-bottom:1px solid var(--grid);vertical-align:top}
th{font-weight:600;color:var(--ink-2);background:var(--surface-2);white-space:nowrap}
td.num,th.num{text-align:right;font-variant-numeric:tabular-nums;white-space:nowrap}
tr:last-child td{border-bottom:0}
.note{font-size:14px;color:var(--ink-2);background:var(--wash);border-radius:10px;padding:12px 14px;max-width:90ch}
.tag{display:inline-block;font:500 11px/1.6 "IBM Plex Mono",ui-monospace,monospace;padding:0 6px;border-radius:4px;border:1px solid var(--rule);color:var(--ink-2);white-space:nowrap}
.tag.ok{color:var(--ok);border-color:currentColor}.tag.warn{color:var(--warn);border-color:currentColor}.tag.bad{color:var(--bad);border-color:currentColor}
.cards{display:grid;grid-template-columns:repeat(auto-fit,minmax(280px,1fr));gap:14px}
.card{background:var(--surface);border:1px solid var(--rule);border-radius:12px;padding:16px 18px;display:grid;gap:8px;align-content:start;min-width:0}
.card h3{font-size:17px}
.q{background:var(--surface);border:1px solid var(--rule);border-left:4px solid var(--accent);border-radius:10px;padding:16px 18px;display:grid;gap:10px}
.q ol{margin:0;padding-left:1.2em;display:grid;gap:6px}
.chart{background:var(--surface);border:1px solid var(--rule);border-radius:10px;padding:12px;overflow-x:auto}
.chart svg{display:block;width:100%;height:auto;min-width:560px}
.chart text{fill:var(--ink-2);font:12px "Source Sans 3",system-ui,sans-serif}
.chart .val{fill:var(--ink);font-weight:600}
.legend{display:flex;flex-wrap:wrap;gap:6px 16px;font-size:13px;color:var(--ink-2);margin:2px 4px 8px}
.legend i{display:inline-block;width:12px;height:12px;border-radius:3px;margin-right:6px;vertical-align:-1px}
[data-tip]{cursor:default}
#tip{position:fixed;z-index:20;pointer-events:none;background:var(--surface);color:var(--ink);border:1px solid var(--rule);
  border-radius:8px;padding:8px 10px;font-size:13px;box-shadow:0 4px 18px rgba(0,0,0,.12);max-width:280px;display:grid;gap:2px}
#tip b{font-size:14px}
dialog{border:0;padding:0;background:transparent;max-width:min(96vw,1800px)}
dialog::backdrop{background:rgba(10,12,10,.82)}
dialog img{display:block;max-width:96vw;max-height:92vh;width:auto;height:auto;border-radius:8px;background:#f6f5f1}
footer{padding-block:28px;font-size:13.5px;color:var(--ink-3);display:grid;gap:6px}
@media (max-width:640px){body{font-size:15.5px}.lede{font-size:16.5px}th,td{padding:6px 8px}}
"""

JS = r"""
(function(){
  var tip=document.getElementById('tip');
  function show(e){var t=e.currentTarget.getAttribute('data-tip');if(!t)return;tip.textContent='';
    t.split('|').forEach(function(line,i){var el=document.createElement(i===0?'b':'span');el.textContent=line;tip.appendChild(el)});
    tip.hidden=false;move(e)}
  function move(e){var x=(e.clientX||0)+14,y=(e.clientY||0)+14;
    if(e.type==='focus'){var r=e.currentTarget.getBoundingClientRect();x=r.right+8;y=r.top}
    var w=tip.offsetWidth,h=tip.offsetHeight;if(x+w>innerWidth-8)x=Math.max(8,innerWidth-w-8);if(y+h>innerHeight-8)y=Math.max(8,y-h-28);
    tip.style.left=x+'px';tip.style.top=y+'px'}
  function hide(){tip.hidden=true}
  document.querySelectorAll('[data-tip]').forEach(function(el){el.setAttribute('tabindex','0');
    el.addEventListener('pointerenter',show);el.addEventListener('pointermove',move);el.addEventListener('pointerleave',hide);
    el.addEventListener('focus',show);el.addEventListener('blur',hide)});
  var dlg=document.getElementById('zoom'),zimg=dlg.querySelector('img');
  document.querySelectorAll('figure img').forEach(function(im){im.addEventListener('click',function(){
    zimg.src=im.getAttribute('data-full')||im.src;zimg.alt=im.alt;if(dlg.showModal)dlg.showModal()})});
  dlg.addEventListener('click',function(){dlg.close()});
})();
"""


def esc(s) -> str:
    return html.escape(str(s), quote=True)


def page(title: str, body: str) -> str:
    return (f'<title>{esc(title)}</title>\n<meta name="claude:viewport-fit-sides" content="cover">\n'
            f'<link rel="preconnect" href="https://fonts.googleapis.com">\n'
            f'<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>\n'
            f'<link rel="stylesheet" href="{FONTS}">\n<style>{CSS}</style>\n'
            f'<div class="wrap">{body}</div>\n<div id="tip" hidden></div>\n'
            f'<dialog id="zoom"><img alt=""></dialog>\n<script>{JS}</script>\n')


def figure(src: str, alt: str, plate: str, caption: str, full: str | None = None) -> str:
    full_attr = f' data-full="{esc(full)}"' if full else ""
    return (f'<figure><div class="frame"><img src="{esc(src)}" alt="{esc(alt)}" loading="lazy"{full_attr}></div>'
            f'<figcaption><span class="plate">{esc(plate)}</span><span>{caption}</span></figcaption></figure>')


def table(head: list[str], rows: list[list], num: set[int] = frozenset(), caption: str = "") -> str:
    th = "".join(f'<th class="num">{esc(h)}</th>' if i in num else f"<th>{esc(h)}</th>" for i, h in enumerate(head))
    trs = []
    for r in rows:
        tds = "".join(f'<td class="num">{c}</td>' if i in num else f"<td>{c}</td>" for i, c in enumerate(r))
        trs.append(f"<tr>{tds}</tr>")
    cap = f'<p class="note" style="margin-top:8px">{caption}</p>' if caption else ""
    return f'<div class="tablewrap"><table><thead><tr>{th}</tr></thead><tbody>{"".join(trs)}</tbody></table></div>{cap}'


def tag(text: str, kind: str = "") -> str:
    return f'<span class="tag {kind}">{esc(text)}</span>'


def _k(v: float) -> str:
    return f"${v / 1000:.1f}k"


def cost_chart(rows: list[tuple[str, float, float, float]], budget=(10000, 20000), xmax=None) -> str:
    """Cost range per class: thin range bar low->high, nominal marker, budget band. rows = (label, low, nom, high)."""
    W, left, right, top, rowh = 760, 92, 150, 30, 46
    xmax = xmax or max(r[3] for r in rows) * 1.08
    xmax = (int(xmax / 5000) + 1) * 5000
    H = top + rowh * len(rows) + 30
    sx = lambda v: left + (W - left - right) * v / xmax  # noqa: E731
    out = [f'<svg viewBox="0 0 {W} {H}" role="img" aria-label="Prototype cost range per height class">']
    out.append(f'<rect x="{sx(budget[0]):.1f}" y="{top - 14}" width="{sx(budget[1]) - sx(budget[0]):.1f}" '
               f'height="{rowh * len(rows) + 10}" fill="var(--band)" rx="4"/>')
    out.append(f'<text x="{(sx(budget[0]) + sx(budget[1])) / 2:.1f}" y="{top - 18}" text-anchor="middle">'
               f'original budget $10–20k</text>')
    for v in range(0, xmax + 1, 5000):
        x = sx(v)
        out.append(f'<line x1="{x:.1f}" x2="{x:.1f}" y1="{top - 8}" y2="{top + rowh * len(rows)}" stroke="var(--grid)" stroke-width="1"/>')
        out.append(f'<text x="{x:.1f}" y="{top + rowh * len(rows) + 18}" text-anchor="middle">{"$0" if v == 0 else _k(v)[:-3] + "k" if v % 1000 == 0 else _k(v)}</text>')
    for i, (lab, lo, nom, hi) in enumerate(rows):
        y = top + rowh * i + rowh / 2
        tipt = f"{_k(nom)} nominal|{lab}|low {_k(lo)} · high {_k(hi)}"
        out.append(f'<text x="{left - 12}" y="{y + 4:.1f}" text-anchor="end" class="val">{esc(lab)}</text>')
        out.append(f'<g data-tip="{esc(tipt)}"><rect x="{sx(lo) - 6:.1f}" y="{y - 14:.1f}" width="{sx(hi) - sx(lo) + 12:.1f}" height="28" fill="transparent"/>'
                   f'<rect x="{sx(lo):.1f}" y="{y - 4:.1f}" width="{sx(hi) - sx(lo):.1f}" height="8" rx="4" fill="var(--series-1)" opacity=".45"/>'
                   f'<circle cx="{sx(nom):.1f}" cy="{y:.1f}" r="7" fill="var(--series-1)" stroke="var(--surface)" stroke-width="2"/></g>')
        out.append(f'<text x="{sx(hi) + 10:.1f}" y="{y + 4:.1f}">{_k(nom)} · high {_k(hi)}</text>')
    out.append("</svg>")
    return '<div class="chart">' + "".join(out) + "</div>"


def stacked_chart(rows: list[tuple[str, list[float]]], series: list[str], unit="kg", title="") -> str:
    """Horizontal stacked bars (one per row), up to three series, 2 px surface gaps, total at the end."""
    W, left, right, top, rowh, bar = 760, 92, 90, 12, 40, 22
    tot = [sum(v) for _, v in rows]
    xmax = max(tot) * 1.05
    H = top + rowh * len(rows) + 26
    sx = lambda v: (W - left - right) * v / xmax  # noqa: E731
    cols = ["var(--series-1)", "var(--series-2)", "var(--series-3)"]
    out = [f'<svg viewBox="0 0 {W} {H}" role="img" aria-label="{esc(title)}">']
    step = 5 if xmax <= 40 else 10
    for v in range(0, int(xmax) + 1, step):
        x = left + sx(v)
        out.append(f'<line x1="{x:.1f}" x2="{x:.1f}" y1="{top}" y2="{top + rowh * len(rows)}" stroke="var(--grid)" stroke-width="1"/>')
        out.append(f'<text x="{x:.1f}" y="{top + rowh * len(rows) + 18}" text-anchor="middle">{v}</text>')
    for i, (lab, vals) in enumerate(rows):
        y = top + rowh * i + (rowh - bar) / 2
        x = left
        out.append(f'<text x="{left - 12}" y="{y + bar / 2 + 4:.1f}" text-anchor="end" class="val">{esc(lab)}</text>')
        n = len([v for v in vals if v > 0])
        k = 0
        for j, v in enumerate(vals):
            if v <= 0:
                continue
            w = sx(v)
            last = k == n - 1
            gap = 0 if last else 2
            tipt = f"{v:.1f} {unit}|{lab} · {series[j]}|{100 * v / tot[i]:.0f} % of {tot[i]:.1f} {unit}"
            if last:  # rounded data end only on the free end
                r = min(4, w / 2)
                d = (f"M{x:.1f},{y:.1f} h{w - r:.1f} a{r},{r} 0 0 1 {r},{r} v{bar - 2 * r:.1f} "
                     f"a{r},{r} 0 0 1 -{r},{r} h-{w - r:.1f} z")
                out.append(f'<path d="{d}" fill="{cols[j]}" data-tip="{esc(tipt)}"/>')
            else:
                out.append(f'<rect x="{x:.1f}" y="{y:.1f}" width="{max(0.5, w - gap):.1f}" height="{bar}" fill="{cols[j]}" data-tip="{esc(tipt)}"/>')
            x += w
            k += 1
        out.append(f'<text x="{x + 8:.1f}" y="{y + bar / 2 + 4:.1f}" class="val">{tot[i]:.1f} {unit}</text>')
    out.append("</svg>")
    leg = "".join(f'<span><i style="background:{cols[j]}"></i>{esc(s)}</span>' for j, s in enumerate(series))
    return f'<div class="chart"><div class="legend">{leg}</div>{"".join(out)}</div>'
