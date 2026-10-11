"""Helpers for building the HalluciGuard guide (HTML -> PDF via headless Chrome)."""
import html as _html

TOC = []  # (level, id, title)
_counter = {"n": 0}

CSS = """
@page { size: A4; margin: 22mm 18mm 22mm 18mm;
  @bottom-left { content: "HalluciGuard \\2014  Complete Project Guide"; font: 8pt 'Helvetica Neue', Arial, sans-serif; color: #6b7280; }
  @bottom-right { content: counter(page); font: 8pt 'Helvetica Neue', Arial, sans-serif; color: #6b7280; } }
@page cover { margin: 0; @bottom-left { content: none; } @bottom-right { content: none; } }
* { box-sizing: border-box; }
body { font-family: 'Helvetica Neue', Arial, sans-serif; font-size: 10pt; line-height: 1.5; color: #1f2937; }
h1 { font-size: 22pt; color: #1d4ed8; margin: 0 0 8pt; page-break-before: always; page-break-after: avoid; }
h1.nobreak { page-break-before: auto; margin-top: 18pt; }
h2 { font-size: 13pt; margin: 16pt 0 5pt; color: #111827; page-break-after: avoid; }
h3 { font-size: 10.5pt; margin: 11pt 0 3pt; color: #111827; page-break-after: avoid; }
p { margin: 0 0 7pt; } ul, ol { margin: 0 0 8pt; padding-left: 18pt; } li { margin-bottom: 2.5pt; }
code, pre { font-family: Menlo, Consolas, monospace; font-size: 8.6pt; }
code { background: #f3f4f6; padding: 0 2pt; border-radius: 2pt; }
pre { background: #f3f4f6; padding: 6pt 8pt; border-radius: 3pt; white-space: pre-wrap; margin: 0 0 8pt; page-break-inside: avoid; }
table { width: 100%; border-collapse: collapse; margin: 4pt 0 10pt; font-size: 9pt; page-break-inside: auto; }
tr { page-break-inside: avoid; }
th { background: #eef2ff; text-align: left; padding: 4pt 6pt; border: 0.5pt solid #d1d5db; }
td { padding: 4pt 6pt; border: 0.5pt solid #d1d5db; vertical-align: top; }
.box { border-left: 3pt solid; padding: 6pt 9pt; margin: 7pt 0 10pt; border-radius: 0 3pt 3pt 0; page-break-inside: avoid; }
.box .t { font-weight: 700; font-size: 8.8pt; margin-bottom: 2pt; }
.box p:last-child { margin-bottom: 0; }
.key { background: #eff6ff; border-color: #2563eb; } .key .t { color: #2563eb; }
.say { background: #ecfdf3; border-color: #16a34a; } .say .t { color: #15803d; }
.care { background: #fff4ec; border-color: #ea580c; } .care .t { color: #c2410c; }
.ex { background: #f8fafc; border-color: #475569; } .ex .t { color: #475569; }
.chk { background: #fffbe6; border-color: #b45309; } .chk .t { color: #92400e; }
.new { background: #f5f3ff; border-color: #7c3aed; } .new .t { color: #6d28d9; }
.fig { text-align: center; margin: 8pt 0 10pt; page-break-inside: avoid; }
.fig svg { max-width: 100%; height: auto; } .cap { font-size: 8.5pt; color: #6b7280; font-style: italic; margin-top: 2pt; }
.cover { page: cover; page-break-after: always; height: 290mm; padding: 90mm 24mm 0 24mm; overflow: hidden; }
.cover h1 { font-size: 40pt; margin: 0; page-break-before: auto; }
.cover .sub { font-size: 20pt; color: #374151; margin: 4pt 0 14pt; } .cover p { color: #4b5563; font-size: 11pt; max-width: 150mm; }
.toc { font-size: 9.5pt; } .toc a { color: inherit; text-decoration: none; display: block; }
.toc a { display: flex; } .toc a .d { flex: 1; border-bottom: 1px dotted #9ca3af; margin: 0 4px 3px; } .toc a .n { min-width: 14pt; text-align: right; }
.toc .l1 { font-weight: 700; margin-top: 5pt; } .toc .l2 { padding-left: 14pt; color: #4b5563; }
.part { page-break-before: always; page-break-after: always; padding-top: 80mm; } .part h1 { page-break-before: auto; font-size: 30pt; } .part p { font-size: 13pt; color: #374151; }
.red { background: #ff5c5c; color: #111; font-weight: 700; padding: 0 2pt; border-radius: 2pt; }
.g { background: #c8f7c5; padding: 1pt 3pt; border-radius: 3pt; } .a { background: #ffe9a8; padding: 1pt 3pt; border-radius: 3pt; }
.lr { background: #ffc2c2; padding: 1pt 3pt; border-radius: 3pt; } .gr { background: #e3e3e8; font-style: italic; color: #555; padding: 1pt 3pt; border-radius: 3pt; }
.pill { font-size: 8pt; background: #e5e7eb; padding: 0 4pt; border-radius: 6pt; }
"""


def esc(s):
    return _html.escape(s)


def _id(title):
    _counter["n"] += 1
    return f"s{_counter['n']}"


def h1(title, nobreak=False):
    i = _id(title)
    TOC.append((1, i, title))
    return f'<h1 id="{i}"{" class=nobreak" if nobreak else ""}>{title}</h1>'


def h2(title):
    i = _id(title)
    TOC.append((2, i, title))
    return f'<h2 id="{i}">{title}</h2>'


def h3(title):
    return f"<h3>{title}</h3>"


def p(*parts):
    return "<p>" + " ".join(parts) + "</p>"


def ul(items, ordered=False):
    tag = "ol" if ordered else "ul"
    return f"<{tag}>" + "".join(f"<li>{i}</li>" for i in items) + f"</{tag}>"


def box(kind, title, *body):
    inner = "".join(b if b.startswith("<") else f"<p>{b}</p>" for b in body)
    return f'<div class="box {kind}"><div class="t">{title}</div>{inner}</div>'


def table(headers, rows, widths=None):
    cols = ""
    if widths:
        cols = "<colgroup>" + "".join(f'<col style="width:{w}%">' for w in widths) + "</colgroup>"
    head = "<tr>" + "".join(f"<th>{h}</th>" for h in headers) + "</tr>"
    body = "".join("<tr>" + "".join(f"<td>{c}</td>" for c in r) + "</tr>" for r in rows)
    return f"<table>{cols}{head}{body}</table>"


def fig(svg, caption):
    return f'<div class="fig">{svg}<div class="cap">{caption}</div></div>'


def part_page(label, subtitle, extra=""):
    i = _id(label)
    TOC.append((1, i, f"{label}: {subtitle}"))
    return f'<div class="part" id="{i}"><h1>{label}</h1><p>{subtitle}</p>{extra}</div>'


# ---------- SVG helpers ----------
def svg_bars(items, title="", width=520, height=200, max_v=1.0, fmt="{:.2f}", colors=None, unit_pct=False, label_size=9):
    """items: list of (label, value[, color]). Vertical bar chart."""
    n = len(items)
    left, right, top, bottom = 34, 10, 26, 46
    cw = width - left - right
    ch = height - top - bottom
    bw = cw / n * 0.58
    out = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {width} {height}" width="{width}" height="{height}" font-family="Helvetica, Arial, sans-serif">']
    if title:
        out.append(f'<text x="{width/2}" y="14" text-anchor="middle" font-size="10" fill="#374151">{esc(title)}</text>')
    for g in range(0, 5):
        y = top + ch - ch * g / 4
        v = max_v * g / 4
        out.append(f'<line x1="{left}" y1="{y}" x2="{width-right}" y2="{y}" stroke="#e5e7eb"/>')
        lab = f"{int(round(v*100))}%" if unit_pct else f"{v:.2f}".rstrip("0").rstrip(".")
        out.append(f'<text x="{left-4}" y="{y+3}" text-anchor="end" font-size="8" fill="#6b7280">{lab}</text>')
    for i, it in enumerate(items):
        label, val = it[0], it[1]
        color = it[2] if len(it) > 2 else (colors[i] if colors else "#2563eb")
        x = left + cw / n * i + (cw / n - bw) / 2
        h = ch * min(val, max_v) / max_v
        y = top + ch - h
        out.append(f'<rect x="{x}" y="{y}" width="{bw}" height="{h}" fill="{color}" rx="2"/>')
        txt = f"{val*100:.0f}%" if unit_pct else fmt.format(val)
        out.append(f'<text x="{x+bw/2}" y="{y-3}" text-anchor="middle" font-size="9" font-weight="700" fill="#111827">{txt}</text>')
        for k, line in enumerate(label.split("\n")):
            out.append(f'<text x="{x+bw/2}" y="{top+ch+12+k*10}" text-anchor="middle" font-size="{label_size}" fill="#374151">{esc(line)}</text>')
    out.append("</svg>")
    return "".join(out)


def svg_boxes(boxes, arrows, width=560, height=150, note=None):
    """boxes: list of (x, y, w, h, title, sub, fill). arrows: list of (x1,y1,x2,y2)."""
    out = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {width} {height}" width="{width}" height="{height}" font-family="Helvetica, Arial, sans-serif">',
           '<defs><marker id="ar" markerWidth="8" markerHeight="8" refX="7" refY="4" orient="auto"><path d="M0,0 L8,4 L0,8 z" fill="#6b7280"/></marker></defs>']
    for (x1, y1, x2, y2) in arrows:
        out.append(f'<line x1="{x1}" y1="{y1}" x2="{x2}" y2="{y2}" stroke="#6b7280" stroke-width="1.2" marker-end="url(#ar)"/>')
    for (x, y, w, h, t, s, fill) in boxes:
        out.append(f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="6" fill="{fill}" stroke="#9ca3af"/>')
        out.append(f'<text x="{x+w/2}" y="{y+h/2-(2 if s else -3)}" text-anchor="middle" font-size="9.5" font-weight="700" fill="#111827">{esc(t)}</text>')
        for k, line in enumerate((s or "").split("\n")):
            if line:
                out.append(f'<text x="{x+w/2}" y="{y+h/2+10+k*9}" text-anchor="middle" font-size="7.6" fill="#4b5563">{esc(line)}</text>')
    if note:
        out.append(f'<text x="{width/2}" y="{height-4}" text-anchor="middle" font-size="8" fill="#6b7280">{esc(note)}</text>')
    out.append("</svg>")
    return "".join(out)


PAGES = {}  # section id -> page number (filled by a second build pass)


def build_toc():
    rows = []
    for lvl, i, t in TOC:
        rows.append(f'<a class="l{lvl}" href="#{i}"><span>{t}</span><span class="d"></span><span class="n">{PAGES.get(i, "")}</span></a>')
    return '<div class="toc">' + "".join(rows) + "</div>"


def page(body_parts, toc_html):
    return ("<!doctype html><html><head><meta charset='utf-8'><title>HalluciGuard - Complete Project Guide</title>"
            f"<style>{CSS}</style></head><body>" + body_parts[0] + toc_html + "".join(body_parts[1:]) + "</body></html>")
