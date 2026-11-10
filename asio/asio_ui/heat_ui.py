"""asio_ui/heat_ui.py - expert heat tables in the Catalog tab: how skewed each is, and bringing new ones in.
A heat entry is {"label", "model", "counts": [[count per expert] per layer] or "zipf": exponent, "src", "date", "tokens"}, kept in data/ai_tools/asio/catalog/heat_<id>.json (one file per entry, never replaced).
Sources: a pasted table (a JSON matrix, Magpie's usage JSON, colibri's usage lines "layer expert count", or Fleet trace lines), one of Fleet's routing traces, a Magpie node's /experts/<model>, or a simulated skew.
Reading a Magpie node contacts only that node, through its AI Tools connection."""
import json, time, uuid, collections
from pathlib import Path
import numpy as np
from asio_ui import base as B, fields as F

TRACES = B.DATA_DIR.parent / "fleet" / "traces"
PENDING: dict = {}

def moe_models(c) -> list: return [(i, c.get("models", i).get("label", i)) for i in c.ids("models") if c.get("models", i).get("arch") == "moe"]

def q_of(c, h: dict) -> np.ndarray:
    asio, _, _ = B.libs()
    from asio_core import heat as H
    m = c.get("models", h["model"])
    if h.get("counts") is None: return H.table({"zipf": h.get("zipf")}, m, c, asio.Valuer("median"))
    cc = np.asarray(h["counts"], dtype=float) + 1e-9
    cc = cc[None, :] if cc.ndim == 1 else cc
    return np.minimum(m["active_experts"] * cc / cc.sum(axis=1, keepdims=True), 1.0)

def curve_svg(curves: dict, w: int = 420, h: int = 180) -> str:
    """Cumulative share of choices against share of experts, hottest first; the diagonal is even heat."""
    colors = ["var(--accent)", "#f5a623", "#7ed321", "#bd10e0", "#ff6b6b"]
    pad, lines, keys = 28, [], ""
    for k, (name, ys) in enumerate(curves.items()):
        pts = " ".join(f"""{pad + (w - pad - 8) * (i + 1) / len(ys):.1f},{h - pad - (h - pad - 8) * y:.1f}""" for i, y in enumerate(ys))
        lines.append(f"""<polyline fill="none" stroke="{colors[k % 5]}" stroke-width="1.6" points="{pad},{h - pad} {pts}"/>""")
        keys += f"""<text x="{pad + 6}" y="{14 + 12 * k}" font-size="10" fill="{colors[k % 5]}">{B.esc(name)}</text>"""
    axes = f"""<line x1="{pad}" y1="{h - pad}" x2="{w - 8}" y2="{h - pad}" stroke="var(--border)"/><line x1="{pad}" y1="8" x2="{pad}" y2="{h - pad}" stroke="var(--border)"/><line x1="{pad}" y1="{h - pad}" x2="{w - 8}" y2="8" stroke="var(--text_muted)" stroke-dasharray="3 3"/>"""
    labels = f"""<text x="{w - 8}" y="{h - 10}" font-size="9" fill="var(--text_muted)" text-anchor="end">share of experts, hottest first</text><text x="4" y="{h / 2}" font-size="9" fill="var(--text_muted)" transform="rotate(-90 10 {h / 2})">share of choices</text>"""
    return f"""<svg viewBox="0 0 {w} {h}" width="{w}" height="{h}">{axes}{''.join(lines)}{keys}{labels}</svg>"""

def summary_html(c, h: dict, name: str = "") -> str:
    from asio_core import heat as H
    s = H.summary(q_of(c, h), c.get("models", h["model"])["active_experts"])
    return f"""<div class="asio-line">{s['layers']} layers x {s['experts']} experts{f", {h['tokens']:,} tokens" if h.get('tokens') else ''}: the hottest 10% of experts take {s['top10']:.0%} of choices, the hottest 25% take {s['top25']:.0%} (even heat: 10% and 25%).</div><div class="asio-fig">{curve_svg({name or h.get('label', 'heat'): s['curve']})}</div>"""

def line(c, h: dict) -> str:
    from asio_core import heat as H
    s = H.summary(q_of(c, h), c.get("models", h["model"])["active_experts"])
    return f"""{B.esc(c.get('models', h['model']).get('label', h['model']))}, {s['layers']}x{s['experts']}, hottest 10% take {s['top10']:.0%}"""

def card(c, i: str) -> str:
    h = c.get("heat", i)
    f = B.USER_CAT / f"heat_{i}.json"
    rm = B.btn("Move to trash", "asio_heat", cls="asio-btn asio-danger", op="trash", id=i, confirm=f"Move heat table {i} to the workspace trash? Scenarios using it will report it missing.") if f.exists() else ""
    return f"""<div class="asio-card"><div class="row"><b class="grow">{B.esc(h.get('label', i))}</b><span class="asio-dim">{B.esc(i)}</span>{rm}</div><div class="asio-dim">{B.esc(h.get('src', ''))} {B.esc(h.get('date', ''))}</div>{summary_html(c, h)}
        <div class="asio-dim">Use it on the Build tab: a MoE engine of this model, expert heat "measured: {B.esc(h.get('label', i))}".</div></div>"""

# --- Bringing heat in ---

def matrix(rows: dict, layers: int = None, experts: int = None) -> list:
    """{layer: {expert: count}} to a full layers x experts list."""
    L = layers or 1 + max(rows, default=-1)
    E = experts or 1 + max((e for r in rows.values() for e in r), default=-1)
    return [[rows.get(l, {}).get(e, 0) for e in range(E)] for l in range(L)]

def parse(text: str) -> tuple:
    """(counts, tokens) from pasted text: a JSON matrix, Magpie's {"usage": {"heat": ...}} or {"heat": ...}, colibri usage lines "layer expert count" (layer -1 is the header: layers, experts), or Fleet trace lines (JSON per token)."""
    t = (text or "").strip()
    if not t: raise ValueError("nothing pasted")
    if t[0] == "[": return json.loads(t), None
    if t[0] == "{" and "\n" not in t.strip():
        d = json.loads(t)
        u = d.get("usage", d)
        return matrix({int(l): {int(e): c for e, c in r.items()} for l, r in u["heat"].items()}, u.get("layers"), u.get("experts")), None
    if t[0] == "{": return trace_counts(t.splitlines())
    rows, dims = collections.defaultdict(dict), {}
    for ln in t.splitlines():
        if not ln.strip(): continue
        l, e, c = (int(x) for x in ln.split()[:3])
        if l == -1: dims = {"layers": e, "experts": c}
        else: rows[l][e] = c
    return matrix(rows, dims.get("layers"), dims.get("experts")), None

def trace_counts(lines) -> tuple:
    """Counts from Fleet's trace format: {"tok", "route": [[experts] per layer]} or {"tok", "layer", "experts"} per line."""
    rows, toks = collections.defaultdict(collections.Counter), set()
    for ln in lines:
        if not ln.strip(): continue
        r = json.loads(ln)
        toks.add(r.get("tok"))
        for l, es in (enumerate(r["route"]) if "route" in r else [(int(r["layer"]), r["experts"])]):
            for e in es: rows[l][e] += 1
    return matrix({l: dict(c) for l, c in rows.items()}), len(toks)

async def from_magpie(conn_id: str, model: str) -> tuple:
    import httpx
    AIM = B.ENV["tools"]["ai_manager"]
    c = AIM.connections.load_conn_raw(conn_id)
    if not c: raise ValueError(f"no connection {conn_id}")
    async with httpx.AsyncClient(timeout=15.0) as cl: r = await cl.get(f"""{AIM.connections.base_url(c)}/experts/{model}""", headers=AIM.connections._auth_headers(c))
    if r.status_code != 200: raise ValueError(f"""the node answered {r.status_code}: {r.text[:200]}""")
    u = r.json().get("usage") or {}
    if not u.get("heat"): raise ValueError(f"""the node has no expert usage for {model} yet{': ' + u['note'] if u.get('note') else ''}""")
    return parse(json.dumps({"usage": u}))[0], None

def magpie_conns() -> list:
    AIM = (B.ENV.get("tools") or {}).get("ai_manager")
    return [(c["_id"], c.get("display_name", c["_id"])) for c in AIM.connections.list_conns(get_all=True) if c.get("connection_type") == "magpie"] if AIM else []

def form(ws) -> str:
    c, f = B.cat(), ws.form("heat")
    traces = sorted(p.stem for p in TRACES.glob("*.jsonl")) if TRACES.exists() else []
    opt = lambda opts, v: "".join(f"""<option value="{B.esc(k)}" {'selected' if str(k) == str(v) else ''}>{B.esc(l)}</option>""" for k, l in opts)
    src = f.get("source", "paste")
    return f"""<form class="asio-card" id="asio-heatform" onsubmit="return false" {B.act("asio_form", trigger="change", form="heat")}><b>Bring in expert heat</b>
        <div class="row asio-wrap">{F.field("model", f'<select class="asio-in asio-w14" name="model">{opt(moe_models(c), f.get("model"))}</select>')}{F.field("from", f'<select class="asio-in asio-w14" name="source">{opt([("paste", "pasted table or trace"), ("trace", "a Fleet routing trace"), ("magpie", "a Magpie node"), ("zipf", "a simulated skew")], src)}</select>')}
        {F.field("name", f'<input class="asio-in asio-w10" name="name" value="{B.esc(f.get("name", ""))}" placeholder="e.g. node1-week45">')}</div>
        <div class="row asio-wrap">{F.field("paste: JSON matrix, Magpie usage JSON, colibri usage lines or Fleet trace lines", f'<textarea class="asio-in asio-w30" name="text" rows="4">{B.esc(f.get("text", ""))}</textarea>')}
        {F.field("Fleet trace (" + B.esc(TRACES) + ")", f'<select class="asio-in asio-w14" name="trace">{opt([(t, t) for t in traces], f.get("trace")) or "<option>none found</option>"}</select>')}
        {F.field("Magpie connection, and the model folder on the node", f'<select class="asio-in asio-w14" name="conn">{opt(magpie_conns(), f.get("conn")) or "<option>no Magpie connections</option>"}</select><input class="asio-in asio-w14" name="folder" value="{B.esc(f.get("folder", ""))}" placeholder="model folder">')}
        {F.field("skew (Zipf exponent: 0 even, 1 strong)", f'<input class="asio-in asio-w4" name="skew" value="{B.esc(f.get("skew", "0.8"))}">')}</div>
        <div class="row">{B.btn("Preview", "asio_heat", op="preview", include="#asio-heatform")}<span class="asio-dim">a preview writes nothing; Save writes a new file</span></div></form><div id="asio-heat-out"></div>"""

async def h_heat(request, payload, imr):
    from asio_ui.build import say
    ws, op = B.ws(request), payload["op"]
    try:
        if op == "preview":
            ws.set_form("heat", {k: v for k, v in payload.items() if k not in ("type", "branch", "lvl", "op")})
            c, src, model = B.cat(), payload.get("source", "paste"), payload.get("model")
            if not model: raise ValueError("pick the model the heat is for")
            tokens = None
            if src == "paste": counts, tokens = parse(payload.get("text"))
            elif src == "trace": counts, tokens = trace_counts(open(TRACES / f"""{Path(payload.get("trace", "")).name}.jsonl""", encoding="utf-8"))
            elif src == "magpie": counts, tokens = await from_magpie(payload.get("conn", ""), (payload.get("folder") or "").strip() or model)
            else: counts = None
            m = c.get("models", model)
            entry = {"label": (payload.get("name") or "").strip() or f"{src} {time.strftime('%Y-%m-%d')}", "model": model, "src": {"paste": "pasted", "trace": f"""Fleet trace {payload.get('trace')}""", "magpie": f"""Magpie {payload.get('conn')} /experts/{payload.get('folder') or model}""", "zipf": "simulated"}[src], "date": time.strftime("%Y-%m-%d")}
            if counts is None: entry["zipf"] = F.parse("est", payload.get("skew")) or 0.8
            else:
                entry.update(counts=counts, tokens=tokens) if tokens else entry.update(counts=counts)
                if len(counts[0]) > m["experts_per_layer"]: raise ValueError(f"""the table has {len(counts[0])} experts per layer; {model} has {m['experts_per_layer']}""")
                counts = [r + [0] * (m["experts_per_layer"] - len(r)) for r in counts]   # experts a trace never saw chosen count zero
                entry["counts"] = counts
            pid = uuid.uuid4().hex[:10]
            PENDING[pid] = entry
            note = "" if counts is None or len(counts) == m.get("moe_layers", m["layers"]) else B.note(f"""The table has {len(counts)} layers and the model {m.get('moe_layers', m['layers'])}: layers are repeated or cut to fit.""")
            return imr.oob(f"""{note}{summary_html(c, entry, entry['label'])}{B.btn("Save as a heat table", "asio_heat", op="save", pid=pid)}""", "asio-heat-out")
        if op == "save":
            e = PENDING.pop(payload.get("pid", ""), None)
            if not e: raise ValueError("that preview is gone: preview again, then save")
            from asio_ui.workspace import safe
            base, n = safe(e["label"]).lower(), 2
            i = base
            while i in B.cat().data["heat"]: i, n = f"{base}-{n}", n + 1
            B.USER_CAT.mkdir(parents=True, exist_ok=True)
            (B.USER_CAT / f"heat_{i}.json").write_text(json.dumps({"heat": {i: e}}))
            ws.set_ui(catalog={**ws.ui().get("catalog", {}), "coll": "heat", "open": i})
        if op == "trash":
            import shutil
            shutil.move(str(B.USER_CAT / f"""heat_{payload['id']}.json"""), str(ws.root / "trash" / f"""heat_{payload['id']}.{time.strftime('%Y%m%d-%H%M%S')}.json"""))
            ws.set_ui(catalog={**ws.ui().get("catalog", {}), "open": None})
    except (ValueError, KeyError, OSError, json.JSONDecodeError) as e: return say(imr, f"{type(e).__name__}: {e}")
    from asio_ui import catalog_ui
    return imr.oob(catalog_ui.page(ws), "asio-catalog").oob("", "asio-msg")

INTENTS = {"asio_heat": h_heat}
