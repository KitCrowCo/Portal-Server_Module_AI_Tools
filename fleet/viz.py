"""viz.py - Fleet's charts as inline SVG, drawn on the server (no chart library, no script): a timeline with lanes, bars, a weekday x hour heat map, a sparkline, an x-y chart and a matrix heat map.
Colours come from the theme (--accent, --text, --text_muted, --border) with fixed fallbacks; categories (models, nodes) get stable hues from their names. Every mark has a <title>, which the browser shows on hover."""
import math, zlib, html
from datetime import datetime

TEXT, MUTED, GRID, ACCENT = "var(--text, #ddd)", "var(--text_muted, #999)", "var(--border, #555)", "var(--accent, #4aa3df)"
OUTCOME = {"served": "hsl(140,45%,48%)", "failed": "hsl(0,65%,55%)", "cancelled": "hsl(40,80%,55%)"}

def esc(s) -> str: return html.escape(str(s), quote=True)
def hue(name) -> str: return f"hsl({zlib.crc32(str(name).encode()) % 360},55%,52%)"   # the same name always gets the same colour
def _svg(w: int, h: int, body: str, label: str = "") -> str: return f"""<svg viewBox="0 0 {w} {h:.0f}" width="{w}" preserveAspectRatio="xMinYMin meet" style="max-width:100%;height:auto;display:block;font-family:var(--font-main, sans-serif)" role="img" aria-label="{esc(label)}">{body}</svg>"""   # drawn at its own size, smaller only where the page is narrower

def _ticks(t0: float, t1: float, x, top: int, bottom: int) -> str:
    """Hour ticks (every 1, 2, 3, 6, 12 or 24 h - whichever gives at most about 12) with local time labels."""
    span_h = (t1 - t0) / 3600
    step = next((s for s in (1, 2, 3, 6, 12, 24, 48, 168) if span_h / s <= 12), 168) * 3600
    t, out = math.ceil(t0 / step) * step, ""
    while t <= t1:
        d = datetime.fromtimestamp(t)
        out += f"""<line x1="{x(t):.1f}" x2="{x(t):.1f}" y1="{top}" y2="{bottom}" stroke="{GRID}" stroke-width="0.5"/><text x="{x(t):.1f}" y="{bottom + 11}" font-size="9" fill="{MUTED}" text-anchor="middle">{d.strftime("%a %H:%M") if step >= 86400 or d.hour == 0 else d.strftime("%H:%M")}</text>"""
        t += step
    return out

def timeline(t0: float, t1: float, lanes: list, width: int = 1100) -> str:
    """lanes: [{"group", "label", "bands": [(start, end, colour, text, title)], "bars": [(start, end, height 0-1, colour, title)], "marks": [(t, symbol, colour, title)], "line": [(t, value 0-1)], "line_title"}] - one row each;
    a new group (a node) gets a heading row above its lanes."""
    left, row, gap, head = 90, 30, 6, 18
    heads = sum(1 for i, ln in enumerate(lanes) if ln.get("group") and (i == 0 or lanes[i - 1].get("group") != ln["group"]))
    h = len(lanes) * (row + gap) + heads * head + 18
    x = lambda t: left + (min(max(t, t0), t1) - t0) / max(t1 - t0, 1) * (width - left - 30)
    body, y = _ticks(t0, t1, x, 0, h - 18), 0
    for i, ln in enumerate(lanes):
        if ln.get("group") and (i == 0 or lanes[i - 1].get("group") != ln["group"]):
            body += f"""<text x="4" y="{y + 13}" font-size="12" font-weight="600" fill="{TEXT}">{esc(ln["group"])}</text>"""
            y += head
        body += f"""<rect x="{left}" y="{y}" width="{width - left - 30}" height="{row}" fill="none" stroke="{GRID}" stroke-width="0.5"/><text x="{left - 6}" y="{y + row / 2 + 4}" font-size="11" fill="{MUTED}" text-anchor="end">{esc(ln["label"][:14])}</text>"""
        for s, e, col, text, title in ln.get("bands", []):
            if e < t0 or s > t1: continue
            w = max(1.5, x(e) - x(s))
            name = f"""<text x="{x(s) + 3:.1f}" y="{y + row / 2 + 4}" font-size="10" fill="{TEXT}">{esc(text[:int(w / 6)])}</text>""" if w > 30 else ""
            body += f"""<g><title>{esc(title)}</title><rect x="{x(s):.1f}" y="{y + 3}" width="{w:.1f}" height="{row - 6}" rx="2" fill="{col}" fill-opacity="0.55"/>{name}</g>"""
        for s, e, frac, col, title in ln.get("bars", []):
            if e < t0 or s > t1: continue
            bh = max(2.0, (row - 4) * min(max(frac, 0.0), 1.0))
            body += f"""<rect x="{x(s):.1f}" y="{y + row - 2 - bh:.1f}" width="{max(1.2, x(e) - x(s)):.1f}" height="{bh:.1f}" fill="{col}"><title>{esc(title)}</title></rect>"""
        pts = [(x(t), y + row - 2 - (row - 4) * min(max(v, 0.0), 1.0)) for t, v in ln.get("line", []) if t0 <= t <= t1]
        if len(pts) > 1: body += f"""<polyline points="{" ".join(f"{a:.1f},{b:.1f}" for a, b in pts)}" fill="none" stroke="{ACCENT}" stroke-width="1.2"><title>{esc(ln.get("line_title", ""))}</title></polyline>"""
        for t, sym, col, title in ln.get("marks", []):
            if t0 <= t <= t1: body += f"""<text x="{x(t):.1f}" y="{y + 12}" font-size="11" fill="{col}" text-anchor="middle">{sym}<title>{esc(title)}</title></text>"""
        y += row + gap
    return _svg(width, h, body, "timeline")

def bars(items: list, width: int = 520, unit: str = "") -> str:
    """Horizontal bars: [(label, value, title)], scaled to the largest value."""
    if not items: return ""
    top, left, row = max((v or 0) for _, v, _ in items) or 1, 170, 18
    body = "".join(f"""<g><title>{esc(title)}</title><text x="{left - 6}" y="{i * row + 12}" font-size="11" fill="{TEXT}" text-anchor="end">{esc(str(label)[:28])}</text><rect x="{left}" y="{i * row + 3}" width="{max(1, (v or 0) / top * (width - left - 70)):.1f}" height="{row - 6}" rx="2" fill="{hue(label)}" fill-opacity="0.8"/><text x="{left + max(1, (v or 0) / top * (width - left - 70)) + 4:.1f}" y="{i * row + 12}" font-size="10" fill="{MUTED}">{"-" if v is None else format(v, ".3g")}{unit}</text></g>""" for i, (label, v, title) in enumerate(items))
    return _svg(width, len(items) * row + 4, body, "bars")

def heat_week(m: list, unit: str = "s", width: int = 640) -> str:
    """7 x 24 matrix (Monday first): cell opacity by value."""
    left, cell, top = 34, (width - 40) / 24, max(max(r) for r in m) or 1
    days = ("Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun")
    body = "".join(f"""<text x="{left + h * cell + cell / 2:.1f}" y="9" font-size="8" fill="{MUTED}" text-anchor="middle">{h}</text>""" for h in range(0, 24, 2))
    for d in range(7):
        body += f"""<text x="{left - 4}" y="{14 + d * 14 + 10}" font-size="9" fill="{MUTED}" text-anchor="end">{days[d]}</text>"""
        body += "".join(f"""<rect x="{left + h * cell:.1f}" y="{14 + d * 14}" width="{cell - 1:.1f}" height="12" rx="1" fill="{ACCENT}" fill-opacity="{0.06 + 0.94 * m[d][h] / top:.2f}"><title>{days[d]} {h:02d}:00-{(h + 1) % 24:02d}:00: {m[d][h]:.0f} {unit}</title></rect>""" for h in range(24))
    return _svg(width, 14 + 7 * 14 + 2, body, "activity by weekday and hour")

def spark(values: list, width: int = 220, height: int = 28, title: str = "") -> str:
    if not values: return ""
    top, n = max(values) or 1, len(values)
    w = width / n
    body = "".join(f"""<rect x="{i * w:.1f}" y="{height - max(1, v / top * height):.1f}" width="{max(1, w - 1):.1f}" height="{max(1, v / top * height):.1f}" fill="{ACCENT}" fill-opacity="{0.35 if v == 0 else 0.85}"><title>{v}</title></rect>""" for i, v in enumerate(values))
    return f"""<span title="{esc(title)}">{_svg(width, height, body, title)}</span>"""

def xy(series: list, xlabel: str = "", ylabel: str = "", width: int = 520, height: int = 200, logx: bool = False) -> str:
    """Lines with points: series = [{"name", "points": [(x, y, title)]}]; one colour per series name."""
    pts = [p for s in series for p in s["points"] if p[0] is not None and p[1] is not None]
    if not pts: return '<div class="status-label">nothing to plot</div>'
    fx = (lambda v: math.log10(max(v, 1e-9))) if logx else (lambda v: v)
    x0, x1, y1 = min(fx(p[0]) for p in pts), max(fx(p[0]) for p in pts), max(p[1] for p in pts) or 1
    L, B, T, R = 46, height - 26, 8, width - 8
    X = lambda v: L + (fx(v) - x0) / ((x1 - x0) or 1) * (R - L)
    Y = lambda v: B - v / y1 * (B - T)
    body = f"""<line x1="{L}" x2="{R}" y1="{B}" y2="{B}" stroke="{GRID}"/><line x1="{L}" x2="{L}" y1="{T}" y2="{B}" stroke="{GRID}"/>"""
    body += "".join(f"""<text x="{L - 4}" y="{Y(y1 * f) + 3:.1f}" font-size="9" fill="{MUTED}" text-anchor="end">{y1 * f:.3g}</text><line x1="{L}" x2="{R}" y1="{Y(y1 * f):.1f}" y2="{Y(y1 * f):.1f}" stroke="{GRID}" stroke-width="0.4"/>""" for f in (0.5, 1.0))
    xs = sorted({p[0] for p in pts})
    body += "".join(f"""<text x="{X(v):.1f}" y="{B + 12}" font-size="9" fill="{MUTED}" text-anchor="middle">{v:g}</text>""" for v in (xs if len(xs) <= 8 else xs[::math.ceil(len(xs) / 8)]))
    body += f"""<text x="{(L + R) / 2}" y="{height - 2}" font-size="9" fill="{MUTED}" text-anchor="middle">{esc(xlabel)}</text><text x="10" y="{(T + B) / 2}" font-size="9" fill="{MUTED}" transform="rotate(-90 10 {(T + B) / 2})" text-anchor="middle">{esc(ylabel)}</text>"""
    for i, s in enumerate(series):
        sp = sorted((p for p in s["points"] if p[0] is not None and p[1] is not None), key=lambda p: p[0])
        col = hue(s["name"])
        if len(sp) > 1: body += f"""<polyline points="{" ".join(f"{X(p[0]):.1f},{Y(p[1]):.1f}" for p in sp)}" fill="none" stroke="{col}" stroke-width="1.5"/>"""
        body += "".join(f"""<circle cx="{X(p[0]):.1f}" cy="{Y(p[1]):.1f}" r="3" fill="{col}"><title>{esc(s["name"])}: {esc(p[2])}</title></circle>""" for p in sp)
        body += f"""<rect x="{R - 120}" y="{T + i * 13}" width="9" height="9" fill="{col}"/><text x="{R - 107}" y="{T + i * 13 + 8}" font-size="9" fill="{TEXT}">{esc(s["name"][:20])}</text>"""
    return _svg(width, height, body, f"{ylabel} by {xlabel}")

def matrix(m: list, title: str = "", width: int = 900, row_label: str = "layer", col_label: str = "expert") -> str:
    """A rows x columns heat map (e.g. layers x experts): opacity by value relative to each row's largest, so every row's hot cells show."""
    rows, cols = len(m), max((len(r) for r in m), default=0)
    if not rows or not cols: return ""
    left, ch = 34, max(2.0, min(10.0, 360 / rows))
    cw = (width - left - 4) / cols
    body = ""
    for i, r in enumerate(m):
        top = max(r) or 1
        if i % max(1, rows // 12) == 0: body += f"""<text x="{left - 4}" y="{i * ch + ch:.1f}" font-size="8" fill="{MUTED}" text-anchor="end">{i}</text>"""
        body += "".join(f"""<rect x="{left + j * cw:.2f}" y="{i * ch:.2f}" width="{max(cw - 0.3, 0.5):.2f}" height="{max(ch - 0.4, 0.8):.2f}" fill="{ACCENT}" fill-opacity="{v / top:.2f}"><title>{row_label} {i}, {col_label} {j}: {v:g}</title></rect>""" for j, v in enumerate(r) if v)
    return _svg(width, rows * ch + 4, body, title)
