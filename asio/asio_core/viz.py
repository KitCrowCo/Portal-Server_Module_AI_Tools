"""asio/viz.py - server-rendered SVG charts of Asio results, no client libraries: an engine timeline (what each engine did and what held it back), limiter shares, the named cases against the Monte Carlo spread, sensitivity, a scatter for Pareto views (cost against capability, time against cost), and the pool's topology.
Text uses currentColor so the charts follow the page's theme; limiter and case colours are the only fixed ones."""
import math, html

LIMIT_COLORS = {"mem": "#4ea1ff", "compute": "#ffb347", "ssd": "#ff6b6b", "hop": "#b48cff", "overhead": "#9a9a9a"}
CASE_COLORS = {"worst": "#ff6b6b", "median": "#4ea1ff", "best": "#5ad68a", "theory": "#b48cff", "improved": "#ffd166", "plateau": "#9a9a9a"}
PALETTE = ["#4ea1ff", "#ffb347", "#5ad68a", "#ff6b6b", "#b48cff", "#ffd166", "#4dd4d4", "#e58bd0"]
FONT = "font-family:var(--font-mono, monospace);font-size:11px"

def esc(s) -> str: return html.escape(str(s), quote=True)

def fmt_t(s: float) -> str:
    if s is None: return "-"
    return f"{s:.2g}s" if s < 1 else f"{s:.0f}s" if s < 120 else f"{s / 60:.1f}m" if s < 7200 else f"{s / 3600:.1f}h" if s < 172800 else f"{s / 86400:.1f}d"

def fmt(v: float) -> str:
    if v is None: return "-"
    a = abs(v)
    return f"{v:.3g}" if a < 1000 else f"{v / 1e3:.3g}k" if a < 1e6 else f"{v / 1e6:.3g}M"

def ticks(lo: float, hi: float, n: int = 5) -> list:
    if hi <= lo: return [lo]
    raw = (hi - lo) / n
    step = 10 ** math.floor(math.log10(raw))
    step *= next(m for m in (1, 2, 2.5, 5, 10) if raw <= m * step)
    t = math.ceil(lo / step) * step
    out = []
    while t <= hi + 1e-9 * step:
        out.append(round(t, 12))
        t += step
    return out

def _svg(w: int, h: int, body: str, title: str = "") -> str: return f"""<svg viewBox="0 0 {w} {h}" width="100%" style="max-width:{w}px;{FONT};color:var(--text, #ddd)" xmlns="http://www.w3.org/2000/svg" role="img"><title>{esc(title)}</title>{body}</svg>"""

def legend(items: dict, x: int, y: int) -> str:
    out, cx = [], x
    for k, c in items.items():
        out.append(f"""<rect x="{cx}" y="{y - 9}" width="10" height="10" fill="{c}"/><text x="{cx + 14}" y="{y}" fill="currentColor">{esc(k)}</text>""")
        cx += 22 + 7 * len(str(k))
    return "".join(out)

# --- Charts ---

def timeline(trace: dict, span: float, w: int = 900) -> str:
    """One row per engine; each block coloured by what limited it, darker the more streams it served at once."""
    rows, lane, left = list(trace.items()), 26, 110
    h = 40 + lane * len(rows)
    sx = (w - left - 10) / max(span, 1e-9)
    body = []
    for i, (eid, tr) in enumerate(rows):
        y = 18 + i * lane
        body.append(f"""<text x="4" y="{y + 15}" fill="currentColor">{esc(eid)}</text><rect x="{left}" y="{y}" width="{w - left - 10}" height="{lane - 6}" fill="none" stroke="currentColor" stroke-opacity=".2"/>""")
        mx = max([b[3] for b in tr] + [1])
        for t0, t1, lim, n in tr:
            body.append(f"""<rect x="{left + t0 * sx:.2f}" y="{y}" width="{max((t1 - t0) * sx, 0.4):.2f}" height="{lane - 6}" fill="{LIMIT_COLORS.get(lim, '#888')}" fill-opacity="{0.35 + 0.65 * n / mx:.2f}"><title>{esc(lim)}, {n} at once, {fmt_t(t0)} to {fmt_t(t1)}</title></rect>""")
    for t in ticks(0, span, 6):
        x = left + t * sx
        body.append(f"""<line x1="{x:.1f}" x2="{x:.1f}" y1="14" y2="{h - 18}" stroke="currentColor" stroke-opacity=".12"/><text x="{x:.1f}" y="{h - 6}" text-anchor="middle" fill="currentColor">{fmt_t(t)}</text>""")
    return _svg(w, h + 16, "".join(body) + legend(LIMIT_COLORS, left, h + 12), "Engine timeline")

def limits(engines: dict, w: int = 900) -> str:
    """Per engine: busy share of the run, split by what limited it."""
    rows, left = list(engines.items()), 110
    h = 30 + 24 * len(rows)
    bw = w - left - 70
    body = []
    for i, (eid, e) in enumerate(rows):
        y, x = 14 + i * 24, left
        body.append(f"""<text x="4" y="{y + 13}" fill="currentColor">{esc(eid)}</text>""")
        for lim, share in sorted((e.get("limits") or {}).items(), key=lambda t: -t[1]):
            ww = bw * e["util"] * share
            body.append(f"""<rect x="{x:.1f}" y="{y}" width="{ww:.1f}" height="16" fill="{LIMIT_COLORS.get(lim, '#888')}"><title>{esc(lim)} {share:.0%} of busy time</title></rect>""")
            x += ww
        body.append(f"""<rect x="{left}" y="{y}" width="{bw}" height="16" fill="none" stroke="currentColor" stroke-opacity=".25"/><text x="{left + bw + 6}" y="{y + 13}" fill="currentColor">{e['util']:.0%} busy</text>""")
    return _svg(w, h + 14, "".join(body) + legend(LIMIT_COLORS, left, h + 8), "Engine limits")

def spread(metric: str, cases: dict, mc: dict = None, w: int = 900, label: str = None) -> str:
    """The named cases as markers on one axis, over the Monte Carlo 5-95% range (line) and 25-75% range (box)."""
    vals = {c: v["metrics"].get(metric) for c, v in cases.items() if not v.get("error") and isinstance(v["metrics"].get(metric), (int, float)) and math.isfinite(v["metrics"].get(metric))}
    s = (mc or {}).get(metric)
    pts = list(vals.values()) + ([s["p5"], s["p95"]] if s else [])
    if not pts: return ""
    lo, hi = min(pts), max(pts)
    pad = (hi - lo) * 0.08 or abs(hi) * 0.1 or 1
    lo, hi = lo - pad, hi + pad
    left, right, h, cy = 20, w - 20, 120, 58
    X = lambda v: left + (v - lo) / (hi - lo) * (right - left)
    body = [f"""<text x="{left}" y="14" fill="currentColor" font-weight="bold">{esc(label or metric)}</text>"""]
    if s:
        body.append(f"""<line x1="{X(s['p5']):.1f}" x2="{X(s['p95']):.1f}" y1="{cy}" y2="{cy}" stroke="currentColor" stroke-opacity=".5" stroke-width="2"/><rect x="{X(s['p25']):.1f}" y="{cy - 8}" width="{max(X(s['p75']) - X(s['p25']), 1):.1f}" height="16" fill="currentColor" fill-opacity=".18"/><line x1="{X(s['p50']):.1f}" x2="{X(s['p50']):.1f}" y1="{cy - 10}" y2="{cy + 10}" stroke="currentColor"/>""")
    rows_end, levels = [], (cy - 16, cy + 24, cy - 30, cy + 38)
    for c, v in sorted(vals.items(), key=lambda t: t[1]):
        text = f"{c} {fmt(v)}"
        x0 = X(v) - 3.5 * len(text)
        lv = next((i for i, end in enumerate(rows_end) if end < x0), len(rows_end))
        if lv == len(rows_end): rows_end.append(0)
        rows_end[lv] = X(v) + 3.5 * len(text) + 6
        y = levels[lv % 4] + (lv // 4) * 12 * (1 if lv % 2 else -1)
        body.append(f"""<circle cx="{X(v):.1f}" cy="{cy}" r="5" fill="{CASE_COLORS.get(c, '#888')}"><title>{esc(c)}: {fmt(v)}</title></circle><line x1="{X(v):.1f}" x2="{X(v):.1f}" y1="{cy if y > cy else y + 2}" y2="{y - 10 if y > cy else cy}" stroke="{CASE_COLORS.get(c, '#888')}" stroke-opacity=".4"/><text x="{X(v):.1f}" y="{y}" text-anchor="middle" fill="{CASE_COLORS.get(c, '#888')}">{esc(text)}</text>""")
    hh = h + 12 * (len(rows_end) // 4)
    for t in ticks(lo, hi, 6): body.append(f"""<text x="{X(t):.1f}" y="{hh - 4}" text-anchor="middle" fill="currentColor" fill-opacity=".7">{fmt(t)}</text>""")
    return _svg(w, hh, "".join(body), f"{metric} by case")

def tornado(rows: list, w: int = 900, title: str = "") -> str:
    """Rank correlation of each input with the output: right = the output rises with it. Faded rows are within what chance gives."""
    if not rows: return ""
    left, mid = 290, (290 + w - 10) / 2
    half = (w - 10 - left) / 2
    body = [f"""<text x="4" y="14" fill="currentColor" font-weight="bold">{esc(title)}</text><line x1="{mid}" x2="{mid}" y1="20" y2="{24 + 18 * len(rows)}" stroke="currentColor" stroke-opacity=".4"/>"""]
    for i, r in enumerate(rows):
        y, ww = 22 + i * 18, half * abs(r["rho"])
        x = mid if r["rho"] >= 0 else mid - ww
        body.append(f"""<text x="4" y="{y + 12}" fill="currentColor" fill-opacity="{1 if r.get('clear', True) else .5}">{esc(r['input'][:44])}</text><rect x="{x:.1f}" y="{y + 2}" width="{ww:.1f}" height="12" fill="{'#ffb347' if r['rho'] >= 0 else '#4ea1ff'}" fill-opacity="{.9 if r.get('clear', True) else .3}"/><text x="{w - 8}" y="{y + 12}" text-anchor="end" fill="currentColor">{r['rho']:+.2f}</text>""")
    return _svg(w, 30 + 18 * len(rows), "".join(body), title)

def scatter(points: list, xlab: str, ylab: str, w: int = 900, h: int = 380, logx: bool = False, logy: bool = False, title: str = "", lines: bool = False) -> str:
    """points: [{"x", "y", "label", "group", "size"}]. Groups get colours; the lower-left to upper edge shows the trade-off. lines joins each group's points in order of x."""
    pts = [p for p in points if p.get("x") is not None and p.get("y") is not None and (not logx or p["x"] > 0) and (not logy or p["y"] > 0)]
    if not pts: return ""
    tx, ty = (math.log10 if logx else float), (math.log10 if logy else float)
    xs, ys = [tx(p["x"]) for p in pts], [ty(p["y"]) for p in pts]
    x0, x1, y0, y1 = min(xs), max(xs), min(ys), max(ys)
    px, py = (x1 - x0) * 0.08 or 1, (y1 - y0) * 0.1 or 1
    x0, x1, y0, y1 = x0 - px, x1 + px, y0 - py, y1 + py
    groups = sorted({p.get("group", "") for p in pts})
    L, R, T, B = 60, w - 20, 26, h - (58 if len(groups) > 1 else 40)
    X, Y = (lambda v: L + (v - x0) / (x1 - x0) * (R - L)), (lambda v: B - (v - y0) / (y1 - y0) * (B - T))
    col = {g: PALETTE[i % len(PALETTE)] for i, g in enumerate(groups)}
    body = [f"""<text x="{L}" y="14" fill="currentColor" font-weight="bold">{esc(title)}</text><line x1="{L}" x2="{R}" y1="{B}" y2="{B}" stroke="currentColor" stroke-opacity=".4"/><line x1="{L}" x2="{L}" y1="{T}" y2="{B}" stroke="currentColor" stroke-opacity=".4"/>"""]
    for t in ticks(x0, x1, 6): body.append(f"""<text x="{X(t):.1f}" y="{B + 14}" text-anchor="middle" fill="currentColor" fill-opacity=".7">{fmt(10 ** t if logx else t)}</text>""")
    for t in ticks(y0, y1, 5): body.append(f"""<text x="{L - 6}" y="{Y(t) + 4:.1f}" text-anchor="end" fill="currentColor" fill-opacity=".7">{fmt(10 ** t if logy else t)}</text><line x1="{L}" x2="{R}" y1="{Y(t):.1f}" y2="{Y(t):.1f}" stroke="currentColor" stroke-opacity=".08"/>""")
    body.append(f"""<text x="{(L + R) / 2}" y="{h - 6}" text-anchor="middle" fill="currentColor">{esc(xlab)}{' (log)' if logx else ''}</text><text x="12" y="{(T + B) / 2}" transform="rotate(-90 12 {(T + B) / 2})" text-anchor="middle" fill="currentColor">{esc(ylab)}{' (log)' if logy else ''}</text>""")
    if lines:
        for g in groups:
            seq = sorted((xv, yv) for p, xv, yv in zip(pts, xs, ys) if p.get("group", "") == g)
            body.append(f"""<polyline fill="none" stroke="{col[g]}" stroke-width="1.6" points="{' '.join(f'{X(a):.1f},{Y(b):.1f}' for a, b in seq)}"/>""")
    for p, xv, yv in zip(pts, xs, ys):
        body.append(f"""<circle cx="{X(xv):.1f}" cy="{Y(yv):.1f}" r="{p.get('size', 5)}" fill="{col[p.get('group', '')]}" fill-opacity=".85"><title>{esc(p.get('label', ''))}: {fmt(p['x'])}, {fmt(p['y'])}</title></circle><text x="{X(xv) + (7 if X(xv) < R - 180 else -7):.1f}" y="{Y(yv) - 6:.1f}" text-anchor="{'start' if X(xv) < R - 180 else 'end'}" fill="currentColor" fill-opacity=".85">{esc(p.get('label', ''))[:28]}</text>""")
    return _svg(w, h, "".join(body) + (legend({g: col[g] for g in groups if g}, L, B + 34) if len(groups) > 1 else ""), title)

def hist(groups: dict, xlab: str, bins: int = 20, w: int = 900, h: int = 300, title: str = "") -> str:
    """groups: {name: [values]}: how often each range of values came up, one colour per group, bars side by side within a range."""
    vals = {g: [v for v in vs if v is not None and math.isfinite(v)] for g, vs in groups.items()}
    allv = [v for vs in vals.values() for v in vs]
    if not allv: return ""
    lo, hi = min(allv), max(allv)
    hi = hi if hi > lo else lo + 1
    edges = [lo + (hi - lo) * i / bins for i in range(bins + 1)]
    counts = {g: [sum(1 for v in vs if edges[i] <= v < edges[i + 1] or (i == bins - 1 and v == hi)) for i in range(bins)] for g, vs in vals.items()}
    top = max(max(c) for c in counts.values()) or 1
    L, R, T, B = 50, w - 20, 26, h - (58 if len(vals) > 1 else 40)
    bw = (R - L) / bins / max(len(vals), 1)
    col = {g: PALETTE[i % len(PALETTE)] for i, g in enumerate(vals)}
    body = [f"""<text x="{L}" y="14" fill="currentColor" font-weight="bold">{esc(title)}</text><line x1="{L}" x2="{R}" y1="{B}" y2="{B}" stroke="currentColor" stroke-opacity=".4"/>"""]
    for k, (g, cs) in enumerate(counts.items()):
        for i, c in enumerate(cs):
            if c: body.append(f"""<rect x="{L + (R - L) * i / bins + k * bw:.1f}" y="{B - (B - T) * c / top:.1f}" width="{max(bw - 1, 1):.1f}" height="{(B - T) * c / top:.1f}" fill="{col[g]}" fill-opacity=".85"><title>{esc(g)}: {c} between {fmt(edges[i])} and {fmt(edges[i + 1])}</title></rect>""")
    for t in ticks(lo, hi, 6): body.append(f"""<text x="{L + (t - lo) / (hi - lo) * (R - L):.1f}" y="{B + 14}" text-anchor="middle" fill="currentColor" fill-opacity=".7">{fmt(t)}</text>""")
    body.append(f"""<text x="{(L + R) / 2}" y="{h - 6}" text-anchor="middle" fill="currentColor">{esc(xlab)}</text><text x="{L - 6}" y="{T + 4}" text-anchor="end" fill="currentColor" fill-opacity=".7">{top}</text>""")
    return _svg(w, h, "".join(body) + (legend(col, L, B + 34) if len(vals) > 1 else ""), title)

def bars(rows: list, ylab: str, w: int = 900, h: int = 320, logy: bool = False, title: str = "") -> str:
    """rows: [(category, {group: value})]: one cluster of bars per category, one colour per group."""
    groups = list(dict.fromkeys(g for _, d in rows for g in d))
    vals = [v for _, d in rows for v in d.values() if v is not None and math.isfinite(v) and (not logy or v > 0)]
    if not vals: return ""
    f = math.log10 if logy else float
    y0, y1 = (min(f(v) for v in vals) - 0.3, max(f(v) for v in vals)) if logy else (min(0.0, min(vals)), max(vals))
    y1 = y1 if y1 > y0 else y0 + 1
    L, R, T, B = 60, w - 20, 26, h - (74 if len(groups) > 1 else 56)
    Y = lambda v: B - (f(v) - y0) / (y1 - y0) * (B - T)
    cw = (R - L) / max(len(rows), 1)
    bw = cw * 0.8 / max(len(groups), 1)
    col = {g: PALETTE[i % len(PALETTE)] for i, g in enumerate(groups)}
    body = [f"""<text x="{L}" y="14" fill="currentColor" font-weight="bold">{esc(title)}</text><line x1="{L}" x2="{R}" y1="{B}" y2="{B}" stroke="currentColor" stroke-opacity=".4"/>"""]
    for t in ticks(y0, y1, 5): body.append(f"""<text x="{L - 6}" y="{B - (t - y0) / (y1 - y0) * (B - T) + 4:.1f}" text-anchor="end" fill="currentColor" fill-opacity=".7">{fmt(10 ** t if logy else t)}</text><line x1="{L}" x2="{R}" y1="{B - (t - y0) / (y1 - y0) * (B - T):.1f}" y2="{B - (t - y0) / (y1 - y0) * (B - T):.1f}" stroke="currentColor" stroke-opacity=".08"/>""")
    for i, (cat, d) in enumerate(rows):
        for k, g in enumerate(groups):
            v = d.get(g)
            if v is None or not math.isfinite(v) or (logy and v <= 0): continue
            x = L + cw * i + cw * 0.1 + k * bw
            body.append(f"""<rect x="{x:.1f}" y="{min(Y(v), B):.1f}" width="{max(bw - 1, 1):.1f}" height="{abs(B - Y(v)):.1f}" fill="{col[g]}" fill-opacity=".85"><title>{esc(cat)} {esc(g)}: {fmt(v)}</title></rect>""")
        body.append(f"""<text x="{L + cw * (i + 0.5):.1f}" y="{B + 14}" text-anchor="middle" fill="currentColor" fill-opacity=".8">{esc(str(cat))[:int(cw / 7) + 2]}</text>""")
    body.append(f"""<text x="12" y="{(T + B) / 2}" transform="rotate(-90 12 {(T + B) / 2})" text-anchor="middle" fill="currentColor">{esc(ylab)}{' (log)' if logy else ''}</text>""")
    return _svg(w, h, "".join(body) + (legend(col, L, B + 40) if len(groups) > 1 else ""), title)

def topology(sc: dict, result: dict = None, w: int = 900, h: int = 300) -> str:
    """Nodes on a circle, links between them (switches and hubs as a centre point), each node with the engines it runs and how busy it was."""
    names = list(sc["nodes"])
    n, cx, cy, r = len(names), w / 2, h / 2 + 6, min(w, h) / 2 - 60
    pos = {nm: (cx + (r * math.cos(2 * math.pi * i / n - math.pi / 2) if n > 1 else 0), cy + (r * math.sin(2 * math.pi * i / n - math.pi / 2) if n > 1 else 0)) for i, nm in enumerate(names)}
    body = []
    for i, l in enumerate(sc.get("links", [])):
        c = PALETTE[i % len(PALETTE)]
        if "nodes" in l:
            hx, hy = sum(pos[m][0] for m in l["nodes"]) / len(l["nodes"]), sum(pos[m][1] for m in l["nodes"]) / len(l["nodes"]) + 8 * i
            body += [f"""<line x1="{pos[m][0]:.1f}" y1="{pos[m][1]:.1f}" x2="{hx:.1f}" y2="{hy:.1f}" stroke="{c}" stroke-dasharray="4 3"/>""" for m in l["nodes"]]
            body.append(f"""<circle cx="{hx:.1f}" cy="{hy:.1f}" r="4" fill="{c}"/><text x="{hx + 6:.1f}" y="{hy - 4:.1f}" fill="{c}">{esc(l['type'])}</text>""")
        else:
            (x1, y1), (x2, y2) = pos[l["a"]], pos[l["b"]]
            body.append(f"""<line x1="{x1:.1f}" y1="{y1:.1f}" x2="{x2:.1f}" y2="{y2:.1f}" stroke="{c}" stroke-width="2"/><text x="{(x1 + x2) / 2:.1f}" y="{(y1 + y2) / 2 - 4:.1f}" fill="{c}" text-anchor="middle">{esc(l['type'])}</text>""")
    engs = {}
    for e in sc["engines"]:
        for nm in e.get("nodes") or [e["node"]]: engs.setdefault(nm, []).append(e["id"])
    for nm, (x, y) in pos.items():
        util = ((result or {}).get("nodes") or {}).get(nm, {}).get("util")
        node = sc["nodes"][nm]
        tpl = node if isinstance(node, str) else node.get("template", "")
        body.append(f"""<circle cx="{x:.1f}" cy="{y:.1f}" r="22" fill="var(--surface, #222)" stroke="currentColor"/><text x="{x:.1f}" y="{y + 4:.1f}" text-anchor="middle" fill="currentColor" font-weight="bold">{esc(nm)}</text><text x="{x:.1f}" y="{y + 36:.1f}" text-anchor="middle" fill="currentColor" fill-opacity=".7">{esc(tpl)}</text><text x="{x:.1f}" y="{y + 49:.1f}" text-anchor="middle" fill="currentColor">{esc(', '.join(engs.get(nm, [])))}{f' ({util:.0%})' if util is not None else ''}</text>""")
    return _svg(w, h + 30, "".join(body), "Pool topology")
