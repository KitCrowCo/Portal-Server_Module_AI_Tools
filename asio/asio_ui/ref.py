"""asio_ui/ref.py - the reference panel in AI Tools' right toolbar: the specs that matter, without leaving the page.
First the machines and models of the open scenario, then a search over the whole catalog, then the tables that are easy to forget (quant sizes, link types)."""
from asio_ui import base as B, fields as F
from asio_ui.build import node_line, node_spec

def model_line(c, m: dict) -> str:
    if m.get("kind") != "llm": return f"""{B.fmt(m.get('gb') or m.get('params_b', 0) * m.get('precision_bytes', 1.0))} GB, {B.esc(m.get('kind', ''))}"""
    kv = F.median(m.get("kv_kb_per_token"))
    if m.get("arch") == "moe": return f"""{m.get('params_b')}B, {m.get('active_b')}B active, {m.get('experts_per_layer')} experts x{m.get('active_experts')}, {B.fmt(F.median(m.get('resident_gb')))} GB held + {m.get('experts_gb', '?')} GB experts, KV {B.fmt(kv)} KB/token"""
    t = c.get("models", "quants")["table"]
    sizes = ", ".join(f"""{q} {B.fmt(m['params_b'] * t[q][0])}""" for q in ("Q4_K_M", "Q6_K", "Q8_0") if q in t)
    return f"""{m.get('params_b')}B, {sizes} GB, KV {B.fmt(kv)} KB/token, ctx {m.get('ctx_max', 0) // 1000}k{', MTP' if m.get('mtp') else ''}"""

def entry(c, coll: str, i: str) -> str:
    e = c.get(coll, i)
    line = node_line(e) if coll == "nodes" else model_line(c, e) if coll == "models" else ", ".join(f"""{F.label(k)[0]} {F.show(v, F.label(k)[1])}""" for k, v in list(F.flat(e).items())[:4] if k != "label")
    return f"""<div class="asio-ref-e" {B.act("asio_cat", op="open", coll=coll, id=i)} title="open in the Catalog tab"><b>{B.esc(e.get('label', i))}</b><div>{line}</div></div>"""

def hits(c, q: str) -> str:
    """Catalog entries whose id or name contains q."""
    if not q.strip(): return ""
    ql = q.lower()
    found = [(coll, i) for coll in ("nodes", "models", "backends", "links", "apis") for i in c.ids(coll) if ql in i.lower() or ql in str(c.get(coll, i).get("label", "")).lower()]
    return "".join(entry(c, coll, i) for coll, i in found[:20]) or '<div class="asio-dim">nothing matches</div>'

def panel(ws, q: str = "") -> str:
    c, name = B.cat(), ws.ui().get("scenario")
    sc = ws.get(name) if name and ws.exists(name) else {}
    here = "".join(f"""<div class="asio-ref-e"><b>{B.esc(n)}: {B.esc(node_spec(c, s).get('label', ''))}</b><div>{node_line(node_spec(c, s))}</div></div>""" for n, s in (sc.get("nodes") or {}).items())
    here += "".join(entry(c, "models", m) for m in dict.fromkeys(e["model"] for e in sc.get("engines") or []))
    t = c.get("models", "quants")["table"]
    quants = "".join(f"""<div class="asio-ref-row"><span>{q}</span><span>{b:g} B/param</span><span>quality {qq:.2f}</span></div>""" for q, (b, qq) in t.items())
    lks = "".join(f"""<div class="asio-ref-row"><span>{B.esc(c.get('links', i).get('label', i))}</span><span>{F.show(c.get('links', i).get('rate_gbps'), 'Gb/s')}</span></div>""" for i in c.ids("links"))
    return f"""<div class="ait-rp-hd">Asio reference</div><input class="asio-in" name="q" value="{B.esc(q)}" placeholder="search the catalog" {B.act("asio_ref", trigger="input changed delay:300ms")}><div id="asio-ref-hits">{hits(c, q)}</div>
        <details open><summary class="asio-ref-h">In {B.esc(name or 'no scenario')}</summary>{here or '<div class="asio-dim">nothing yet</div>'}</details>
        <details><summary class="asio-ref-h">Quant sizes</summary>{quants}</details><details><summary class="asio-ref-h">Links</summary>{lks}</details>"""

async def h_ref(request, payload, imr): return imr.oob(hits(B.cat(), payload.get("q", "")), "asio-ref-hits")

INTENTS = {"asio_ref": h_ref}
