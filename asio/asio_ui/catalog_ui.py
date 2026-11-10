"""asio_ui/catalog_ui.py - the Catalog tab: every machine, model, program, link, API and cost set as readable cards, side-by-side comparison, and your own entries and changes.
Changes never touch the built-in catalog: they go in data/ai_tools/asio/catalog/edits_<collection>.json, which overrides it field by field, and the file's previous content is appended to data/ai_tools/asio/catalog_history.jsonl before each write.
Measured values from Fleet (calibration) are previewed here and written as new measured_<name>.json files."""
import json, time, uuid
from asio_ui import base as B, fields as F
from asio_ui.build import node_line
from asio_ui.ref import model_line
from asio_ui import heat_ui, lookup

COLLS = [("nodes", "Machines"), ("models", "Models"), ("backends", "Programs"), ("links", "Links"), ("apis", "APIs"), ("economics", "Costs"), ("heat", "Expert heat"), ("measured", "Measured data"), ("lookup", "Look up")]
LOW = {"price_usd", "power_w.idle", "power_w.load", "latency_us", "in", "cached", "out", "os_reserve_gb", "step_ms", "token_overhead_ms", "electricity_usd_kwh", "internet_usd_month", "space_usd_month", "kv_kb_per_token", "state_mb", "resident_gb", "dense_gb_token", "expert_gb_token", "gflop_token", "rtf", "disk_gb"}
HISTORY = B.DATA_DIR / "catalog_history.jsonl"

def edits_file(coll: str): return B.USER_CAT / f"edits_{coll}.json"
def edits(coll: str) -> dict: return json.loads(edits_file(coll).read_text()).get(coll, {}) if edits_file(coll).exists() else {}

def write_edits(coll: str, items: dict, why: str):
    """Writes this collection's edits file, appending its previous content to the history first."""
    B.USER_CAT.mkdir(parents=True, exist_ok=True)
    f = edits_file(coll)
    with open(HISTORY, "a") as h: h.write(json.dumps({"t": time.time(), "file": f.name, "why": why, "before": json.loads(f.read_text()) if f.exists() else None}) + "\n")
    f.write_text(json.dumps({coll: items}, indent=1))

# --- Reading an entry ---

def line(c, coll: str, e: dict) -> str:
    if coll == "heat": return heat_ui.line(c, e)
    if coll == "nodes": return node_line(e)
    if coll == "models": return model_line(c, e)
    return ", ".join(f"""{F.label(k)[0]} {F.show(v, F.label(k)[1])}""" for k, v in F.flat(e).items() if k not in ("label", "src", "date") and not isinstance(v, (dict, list)))[:400]

def derived(c, coll: str, e: dict) -> dict:
    """Numbers worth seeing that the entry does not state: bandwidth from MT/s and bus width, usable memory, cost per GB/s; model sizes at each quant."""
    if coll == "nodes":
        bw = F.median(e.get("mem_bw_gbs")) or (e["mem_mts"] * e["bus_bits"] / 8000 if e.get("mem_mts") and e.get("bus_bits") else None)
        price = F.median(e.get("price_usd"))
        return {"memory bandwidth (GB/s, theory)": bw, "usable RAM (GB)": (e.get("ram_gb") or 0) - (F.median(e.get("os_reserve_gb")) or 2.5), "$ per GB/s": price / bw if bw and price else None, "W under load per GB/s": (F.median(e.get("power_w", {}).get("load")) or 0) / bw if bw else None}
    if coll == "models" and e.get("kind") == "llm" and e.get("arch") != "moe" and e.get("params_b"): return {f"weights at {q} (GB)": e["params_b"] * b for q, (b, _) in c.get("models", "quants")["table"].items()} | {"KV for 32k tokens (GB)": (F.median(e.get("kv_kb_per_token")) or 0) * 32768 * 1e3 * e.get("kv_quant_bytes", 1.0625) / 2 / 1e9}
    return {}

def src_of(v) -> str: return B.esc("; ".join(x for x in (v.get("src"), (v.get("measured") or {}).get("src"), v.get("date")) if x)) if F.is_spec(v) else ""

def card(c, coll: str, i: str) -> str:
    if coll == "heat": return heat_ui.card(c, i)
    e, mine = c.get(coll, i), i in edits(coll)
    rows = [[B.esc(F.label(k)[0]), F.show(v, F.label(k)[1]), f"""<span class="asio-dim">{src_of(v)}</span>"""] for k, v in F.flat(e).items() if k != "label"]
    rows += [[f"""<i>{B.esc(k)}</i>""", B.fmt(v), '<span class="asio-dim">worked out</span>'] for k, v in derived(c, coll, e).items() if v is not None]
    notes = "".join(f"""<div class="asio-dim">{B.esc(v)}</div>""" for k, v in c.data[coll].get(i, {}).items() if k.startswith("_"))
    base = c.data[coll][i].get("extends")
    acts = B.btn("Edit", "asio_cat", op="edit", coll=coll, id=i) + B.btn("Copy as new", "asio_cat", op="copy", coll=coll, id=i, title="your own entry that starts from this one") + (B.btn("Remove my changes", "asio_cat", cls="asio-btn asio-danger", op="revert", coll=coll, id=i, confirm=f"Remove your changes to {i}? It goes back to the built-in entry (or disappears, if you made it). The removed changes stay in the catalog history.") if mine else "")
    return f"""<div class="asio-card"><div class="row"><b class="grow">{B.esc(e.get('label', i))}</b><span class="asio-dim">{B.esc(i)}{f' (from {B.esc(base)})' if base else ''}{' - has your changes' if mine else ''}</span>{acts}</div>{notes}{B.tbl(["", "value", "source"], rows)}</div>"""

def compare(c, coll: str, ids: list) -> str:
    """Side by side: every field any of them has, the best of each row marked where better is clear. Heat tables are compared as curves."""
    if coll == "heat":
        from asio_core import heat as H
        curves = {c.get("heat", i).get("label", i): H.summary(heat_ui.q_of(c, c.get("heat", i)), 1)["curve"] for i in ids}
        return f"""<div class="row"><b class="grow">Comparing {len(ids)}</b>{B.btn("Clear", "asio_cat", op="compare_clear", coll=coll)}</div><div class="asio-fig">{heat_ui.curve_svg(curves, 560, 240)}</div><div class="asio-dim">The steeper a curve rises, the more a few experts take; the dashed diagonal is even heat.</div>"""
    es = [c.get(coll, i) for i in ids]
    flats = [{**F.flat(e), **{f"= {k}": v for k, v in derived(c, coll, e).items()}} for e in es]
    keys = list(dict.fromkeys(k for fl in flats for k in fl if k != "label"))
    rows = []
    for k in keys:
        nums = [F.median(fl.get(k)) if isinstance(F.median(fl.get(k)), (int, float)) and not isinstance(fl.get(k), bool) else None for fl in flats]
        spec = next((fl[k] for fl in flats if F.is_spec(fl.get(k))), None)
        low = (spec or {}).get("better") == "low" if spec else k.split(".")[-1] in LOW or k.startswith("= $") or k.startswith("= W")
        have = [x for x in nums if x is not None]
        best = (min(have) if low else max(have)) if len(have) > 1 and len(set(have)) > 1 else None
        lbl = k[2:] if k.startswith("= ") else F.label(k)[0]
        rows.append([f"""<i>{B.esc(lbl)}</i>""" if k.startswith("= ") else B.esc(lbl)] + [f"""<span class="{'asio-best' if best is not None and n == best else ''}">{B.fmt(fl.get(k)) if k.startswith('= ') else F.show(fl.get(k), F.label(k)[1])}</span>""" for fl, n in zip(flats, nums)])
    return f"""<div class="row"><b class="grow">Comparing {len(ids)}</b>{B.btn("Clear", "asio_cat", op="compare_clear", coll=coll)}</div>{B.tbl([""] + [B.esc(e.get('label', e['id'])) for e in es], rows, "asio-cmp")}<div class="asio-dim">Marked: the best of a row where lower or higher is clearly better.</div>"""

# --- Editing ---

def edit_form(c, coll: str, i: str) -> str:
    """Every field of the entry; a value with tiers shows its theory, estimate (low, most likely, high), measured values and source."""
    rows = []
    for k, v in F.flat(c.get(coll, i)).items():
        nm = B.esc(F.label(k)[0])
        if F.is_spec(v):
            m = v.get("measured") or {}
            rows.append([nm, f"""<input class="asio-in asio-w5" name="t:{B.esc(k)}" value="{B.esc(F.typed(v.get('theory')))}" placeholder="theory">""", f"""<input class="asio-in asio-w8" name="e:{B.esc(k)}" value="{B.esc(F.typed(v.get('est')))}" placeholder="low likely high">""", f"""<input class="asio-in asio-w8" name="m:{B.esc(k)}" value="{B.esc(', '.join(f'{x:g}' for x in m.get('values', [])))}" placeholder="measured values">""", f"""<input class="asio-in asio-w14" name="s:{B.esc(k)}" value="{B.esc(v.get('src', ''))}" placeholder="source">"""])
        elif isinstance(v, (dict, list)): rows.append([nm, f"""<input class="asio-in asio-w20" name="j:{B.esc(k)}" value="{B.esc(json.dumps(v))}">""", "", "", ""])
        else: rows.append([nm, f"""<input class="asio-in asio-w14" name="p:{B.esc(k)}" value="{B.esc(F.typed(v))}">""", "", "", ""])
    add = """<div class="row"><input class="asio-in asio-w14" name="newkey" placeholder="add a field: its path, e.g. power_w.idle"><input class="asio-in asio-w14" name="newval" placeholder="value (number, text or JSON)"></div>"""
    return f"""<form class="asio-card" id="asio-edit" onsubmit="return false"><div class="row"><b class="grow">Editing {B.esc(i)}</b>{B.btn("Save", "asio_cat", op="save", coll=coll, id=i, include="#asio-edit")}{B.btn("Cancel", "asio_cat", op="open", coll=coll, id=i)}</div>
        <div class="asio-dim">Only what you change is written, to edits_{coll}.json in your catalog folder. Measured values are a list of readings; an empty field keeps the catalog's.</div>{B.tbl(["field", "value or theory", "estimate", "measured", "source"], rows)}{add}</form>"""

def _num(s: str):
    try: return float(s)
    except ValueError: return s

def changes(c, coll: str, i: str, form: dict) -> dict:
    """The fields the form changed, as a nested patch; a value with tiers is written whole."""
    old, patch = F.flat(c.get(coll, i)), {}
    for k, v in old.items():
        if F.is_spec(v):
            new = {**v}
            for tag, f in (("t", "theory"), ("e", "est")):
                raw = form.get(f"{tag}:{k}", F.typed(v.get(f)))
                new[f] = F.parse("est", raw) if raw.strip() else None
            vals = [float(x) for x in form.get(f"m:{k}", "").replace(",", " ").split()]
            new["measured"] = {**(v.get("measured") or {}), "values": vals, "src": (v.get("measured") or {}).get("src", "edited by hand"), "date": (v.get("measured") or {}).get("date", time.strftime("%Y-%m-%d"))} if vals else None
            new["src"] = form.get(f"s:{k}", v.get("src", "")) or None
            new = {a: b for a, b in new.items() if b is not None}
            if new != v: F.set_path(patch, k, new)
        elif f"p:{k}" in form or f"j:{k}" in form:
            raw = form.get(f"p:{k}", form.get(f"j:{k}", ""))
            new = json.loads(raw) if f"j:{k}" in form else (raw if isinstance(v, str) else F.parse("bool", raw) if isinstance(v, bool) else _num(raw))
            if new != v and raw != F.typed(v): F.set_path(patch, k, new)
    if (form.get("newkey") or "").strip():
        raw = form.get("newval", "")
        F.set_path(patch, form["newkey"].strip(), json.loads(raw) if raw[:1] in "[{" else _num(raw))
    return patch

# --- Tab ---

def items(c, coll: str, q: str, picked: list) -> str:
    ql = q.lower()
    rows = []
    for i in c.ids(coll):
        e = c.get(coll, i)
        if ql and ql not in i.lower() and ql not in str(e.get("label", "")).lower(): continue
        tags = ("yours " if e.get("owned") else "") + ("to verify " if e.get("candidate") else "") + ("changed " if i in edits(coll) else "")
        rows.append(f"""<div class="asio-item"><input type="checkbox" name="v" value="{B.esc(i)}" {'checked' if i in picked else ''} title="compare"><div class="grow" {B.act("asio_cat", op="open", coll=coll, id=i)}><b>{B.esc(e.get('label', i))}</b> <span class="asio-tag">{tags}</span><div class="asio-line">{line(c, coll, e)}</div></div></div>""")
    return f"""<form class="asio-multi" onsubmit="return false" {B.act("asio_cat", trigger="change", op="compare", coll=coll)}>{''.join(rows) or f'<div class="asio-dim">{"nothing matches" if q else "nothing here yet"}</div>'}</form>"""

def detail(c, ws, coll: str) -> str:
    u = ws.ui().get("catalog", {})
    picked = [i for i in u.get("compare", []) if i in c.data.get(coll, {})]
    if len(picked) > 1: return compare(c, coll, picked)
    i = u.get("open")
    if i and i in c.data.get(coll, {}): return card(c, coll, i)
    return '<div class="asio-dim asio-pad">Pick an entry to read it, or tick two to four to compare them.</div>'

def page(ws) -> str:
    c, u = B.cat(), ws.ui().get("catalog", {})
    coll = u.get("coll", "nodes")
    chips = "".join(f"""<button type="button" class="chip {'asio-on' if k == coll else ''}" {B.act("asio_cat", op="coll", coll=k)}>{lbl}</button>""" for k, lbl in COLLS)
    if coll == "lookup": return f"""<div class="asio-pad"><div class="row">{chips}</div>{lookup.page(ws)}</div>"""
    if coll == "measured": return f"""<div class="asio-pad"><div class="row">{chips}</div>{measured_form(ws)}<div id="asio-meas-out"></div></div>"""
    return f"""<div class="asio-pad"><div class="row">{chips}<input class="asio-in asio-w14" name="q" value="{B.esc(u.get('q', ''))}" placeholder="filter" {B.act("asio_cat", trigger="input changed delay:300ms", op="filter", coll=coll)}><span class="asio-dim">{len(c.files)} catalog files; yours are in {B.esc(B.USER_CAT)}</span></div>
        {heat_ui.form(ws) if coll == "heat" else ""}<div class="asio-split"><div class="asio-list" id="asio-cat-list">{items(c, coll, u.get('q', ''), u.get('compare', []))}</div><div class="asio-detail" id="asio-cat-detail">{detail(c, ws, coll)}</div></div></div>"""

# --- Measured data (calibration) ---

FLEET_DB = B.DATA_DIR.parent / "fleet" / "fleet.db"
PROPOSALS: dict = {}
MAP_HINT = '{"nodes": {"<Fleet node id>": "getorli_h255"}, "models": {"qwen3.8-27b*": {"model": "qwen3.8-27b", "quant": "Q4_K_M", "backend": "llama_vulkan"}}}'

def measured_form(ws) -> str:
    f = ws.form("measured")
    ta = lambda n, rows, ph="": f"""<textarea class="asio-in" name="{n}" rows="{rows}" placeholder="{B.esc(ph)}">{B.esc(f.get(n, MAP_HINT if n == 'mapping' else ''))}</textarea>"""
    return f"""<form class="asio-card" id="asio-meas" onsubmit="return false" {B.act("asio_form", trigger="change", form="measured")}><div class="asio-dim">Fleet's request records ({B.esc(FLEET_DB)}{'' if FLEET_DB.exists() else ', not found'}) and anything pasted here are turned into measured values. A preview writes nothing.</div>
        {F.field("Fleet nodes and models to catalog ids (JSON)", ta("mapping", 3))}{F.field("Agent steps, one JSON object per line (optional)", ta("agent_steps", 3, '{"model": "...", "task": "t1", "valid": true, "after_bad": false, "progress": true}'))}
        {F.field("Readings, one JSON object per line (optional)", ta("readings", 3, '{"path": "nodes/getorli_h255/power_w/idle", "value": 10.2, "src": "wall meter"}'))}<div class="row">{F.field("days of Fleet records", f'<input class="asio-in asio-w4" name="days" value="{B.esc(f.get("days", "30"))}">')}{B.btn("Preview measured values", "asio_cat", op="measure", include="#asio-meas")}</div></form>"""

def _lines(text: str) -> list: return [json.loads(x) for x in (text or "").splitlines() if x.strip()]

def measure(payload: dict) -> str:
    from asio_core import calibrate
    ev = calibrate.fleet_events(str(FLEET_DB), time.time() - float(payload.get("days") or 30) * 86400) if FLEET_DB.exists() else []
    p = calibrate.propose(B.cat(), ev, json.loads(payload.get("mapping") or "{}"), _lines(payload.get("agent_steps")) + [e for e in ev if e.get("kind") == "agent.step"], _lines(payload.get("readings")))
    pid = uuid.uuid4().hex[:10]
    PROPOSALS[pid] = p
    rows = [[B.esc(k), B.fmt(v["now"]), B.fmt(v["measured_p50"]), v["n"], B.esc(v["src"])] for k, v in p["values"].items()]
    skipped = ", ".join(f"{k}: {v}" for k, v in p["skipped"].items() if v)
    write = f"""<form class="row" id="asio-mwrite" onsubmit="return false"><input class="asio-in asio-w14" name="mname" placeholder="file name (empty: date and time)">{B.btn("Write these as a new measured file", "asio_cat", op="measure_write", proposal=pid, include="#asio-mwrite")}</form>""" if rows else ""
    return f"""{B.note(f"{p['observations']} request observations; {p['clamped']} fell outside any plausible value and were left out.{' Skipped: ' + B.esc(skipped) if skipped else ''}")}{B.tbl(['catalog value', 'now (median)', 'measured (median)', 'n', 'from'], rows) if rows else '<div class="asio-dim">Nothing to change.</div>'}{write}"""

# --- Intents ---

async def h_cat(request, payload, imr):
    from asio_ui.build import say
    ws, op, coll = B.ws(request), payload["op"], payload.get("coll")
    u = ws.ui().get("catalog", {})
    try:
        if op == "coll": ws.set_ui(catalog={**u, "coll": coll, "open": None, "compare": []})
        elif op == "filter":
            ws.set_ui(catalog={**u, "q": payload.get("q", "")})
            return imr.oob(items(B.cat(), coll, payload.get("q", ""), u.get("compare", [])), "asio-cat-list")
        elif op == "open":
            ws.set_ui(catalog={**u, "coll": coll, "open": payload["id"], "compare": []})
            return await B.FOCUS(request, imr, "catalog")
        elif op == "compare":
            ws.set_ui(catalog={**u, "compare": F.parse("list", payload.get("v"))[:4]})
            return imr.oob(detail(B.cat(), ws, coll), "asio-cat-detail")
        elif op == "compare_clear": ws.set_ui(catalog={**u, "compare": []})
        elif op == "edit": return imr.oob(edit_form(B.cat(), coll, payload["id"]), "asio-cat-detail")
        elif op == "save":
            patch = changes(B.cat(), coll, payload["id"], payload)
            if not patch: return say(imr, "Nothing changed.", bad=False)
            from asio_core.catalog import merge
            ed = edits(coll)
            write_edits(coll, {**ed, payload["id"]: merge(ed.get(payload["id"], {}), patch)}, f"""edited {payload['id']}""")
        elif op == "copy":
            ed, i = edits(coll), payload["id"]
            new = next(f"{i}-mine{'' if k == 1 else k}" for k in range(1, 99) if f"{i}-mine{'' if k == 1 else k}" not in B.cat().data[coll])
            write_edits(coll, {**ed, new: {"extends": i, "label": f"""{B.cat().get(coll, i).get('label', i)} (mine)""", **({"owned": True, "abstract": False} if coll == "nodes" else {"abstract": False})}}, f"copied {i} as {new}")
            ws.set_ui(catalog={**u, "open": new, "compare": []})
            return imr.oob(page(ws), "asio-catalog").oob(edit_form(B.cat(), coll, new), "asio-cat-detail")
        elif op == "revert":
            ed = edits(coll)
            ed.pop(payload["id"], None)
            write_edits(coll, ed, f"""removed changes to {payload['id']}""")
        elif op == "measure": return imr.oob(measure(payload), "asio-meas-out")
        elif op == "measure_write":
            from asio_core import calibrate
            p = PROPOSALS.pop(payload.get("proposal", ""), None)
            if not p: return say(imr, "That preview is gone: preview again, then write.")
            name = "".join(ch for ch in (payload.get("mname") or "") if ch.isalnum() or ch in "_-") or None
            return imr.oob(B.note(f"""Written: {B.esc(calibrate.write(p, str(B.USER_CAT), name))}. Runs now read these as measured values; tier "estimates only" ignores them, and deleting the file undoes it."""), "asio-meas-out")
    except (ValueError, KeyError, json.JSONDecodeError) as e: return say(imr, f"{type(e).__name__}: {e}")
    return imr.oob(page(ws), "asio-catalog").oob("", "asio-msg")

INTENTS = {"asio_cat": h_cat}
