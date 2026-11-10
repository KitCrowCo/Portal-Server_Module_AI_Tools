"""asio_ui/build.py - the Build tab: a scenario as forms (machines, links, engines, routes, workloads, tweaks), saved to the workspace on every change.
A plain value change saves and refreshes only the fit check; a change that alters what other fields offer (a model, a node, a kind) refreshes its section too. The JSON stays available as the advanced view."""
import json, copy
from asio_ui import base as B, fields as F

KIND_FIELDS = {
    "stream": [("role", "route", "role"), ("prompt", "prompt tokens", "esti", 1000), ("out", "output tokens", "esti", 500)],
    "chat": [("role", "route", "role"), ("system", "context at start (tokens)", "esti", 0), ("turns", "turns", "esti", 5), ("prompt", "tokens per turn in", "esti", 200), ("out", "tokens per turn out", "esti", 300), ("think_s", "pause between turns (s)", "est", 20), ("session", "reuse KV between turns", "bool")],
    "agent": [("role", "route", "role"), ("steps_needed", "good steps a task needs", "esti", [5, 10, 20]), ("system", "system prompt (tokens)", "esti", 4000), ("step_out", "tokens written per step", "esti", [150, 400, 1200]), ("tool_out", "tool result (tokens)", "esti", [100, 600, 3000]), ("tool_s", "tool time (s)", "est", [0.5, 2, 15]),
              ("error_in", "error message (tokens)", "esti", 80), ("max_steps", "most steps", "int", 60), ("max_stuck", "bad steps in a row before giving up", "int", 4), ("kv", "keep KV between steps", "bool")],
    "routed": [("decide_role", "decision route", "role"), ("prompt", "request (tokens)", "esti"), ("p_hard", "share that is hard", "float"), ("fail_if_wrong", "a hard one sent easy fails", "float"),
               ("routes.easy.role", "easy route", "role"), ("routes.easy.out", "easy output (tokens)", "esti"), ("routes.hard.role", "hard route", "role"), ("routes.hard.out", "hard output (tokens)", "esti")],
    "pipeline": [("stages", "stages (JSON: id, after, do)", "json")],
    "image": [("role", "route", "role"), ("mp", "megapixels", "est"), ("steps", "steps", "esti")],
    "video": [("role", "route", "role"), ("frames", "frames", "esti"), ("steps", "steps", "esti")],
    "stt": [("role", "route", "role"), ("audio_s", "audio (s)", "est")], "tts": [("role", "route", "role"), ("chars", "characters", "esti")], "encode": [("role", "route", "role")],
    "train": [("role", "route", "role"), ("tokens", "training tokens", "esti"), ("factor", "cost against a forward pass", "float")]}
KIND_FIELDS["batch"] = KIND_FIELDS["stream"]
KIND_FIELDS["benchmark"] = KIND_FIELDS["stream"] + [("items", "the set: [[prompt, output], ...] (each job takes the next)", "json")]
KIND_FIELDS["mix"] = [("role", "route (for types that name none)", "role"), ("rate_per_h", "jobs per hour at full demand", "est", 10), ("duration_h", "for hours", "est", 24), ("start_hour", "starting at hour", "float", 0), ("profile", "demand by hour from midnight (24 numbers, 1 = full)", "nums", "flat: 1 every hour")]
COMMON = [("max_in_flight", "most at once (empty: no limit)", "int"), ("slo_s", "should end within (s)", "float")]
SUB_FIELDS = [("subagents.role", "subagent route", "role"), ("subagents.every", "every N good steps", "int"), ("subagents.n", "subagents at once", "int"), ("subagents.steps_needed", "their steps needed", "esti"), ("subagents.system", "their system prompt", "esti"), ("subagents.summary_out", "summary back (tokens)", "esti")]
KIND_DEFAULTS = {"stream": {"prompt": 1000, "out": 500}, "batch": {"prompt": 1000, "out": 500, "count": 50}, "benchmark": {"items": [[500, 200], [2000, 400], [300, 100]] * 4, "max_in_flight": 4},
                 "mix": {"rate_per_h": 20, "duration_h": 8, "types": [{"name": "chat", "kind": "stream", "share": 3, "prompt": [200, 600, 2000], "out": [100, 300, 800], "slo_s": 30}, {"name": "agent", "kind": "agent", "share": 1, "steps_needed": [4, 8, 16], "slo_s": 1800}]}, "chat": {"system": 2000, "turns": 4, "prompt": 200, "out": 300, "think_s": 20}, "agent": {"count": 4, "steps_needed": [4, 8, 16], "system": 5000, "step_out": [150, 350, 900], "tool_out": [100, 500, 2500], "tool_s": [0.5, 2, 15]},
                 "routed": {"count": 20, "decide_role": "decide", "prompt": [200, 600, 2000], "p_hard": 0.3, "fail_if_wrong": 0.6, "routes": {"easy": {"role": "easy", "out": [80, 200, 500]}, "hard": {"role": "hard", "out": [200, 500, 1200]}}},
                 "pipeline": {"stages": [{"id": "a", "do": {"llm": {"role": "main", "new": 500, "out": 200}}}, {"id": "b", "after": ["a"], "do": {"delay": 2}}]}, "image": {"count": 4, "mp": 1.0, "steps": 4}, "video": {"count": 1}, "stt": {"audio_s": [3, 6, 15]}, "tts": {"chars": [150, 500, 1500]}, "encode": {}, "train": {"tokens": 200000, "factor": 3.0}}
KIND_WORDS = {"stream": "single requests", "batch": "a batch of requests", "benchmark": "a benchmark set (fixed items)", "mix": "a demand mix (job types by share, daily profile)", "chat": "chat sessions", "agent": "agent tasks (tool loops)", "routed": "requests a decision model routes", "pipeline": "a pipeline of stages", "image": "images", "video": "video clips", "stt": "speech to text", "tts": "text to speech", "encode": "encoder calls", "train": "LoRA training"}
ARRIVALS = [("all", "all at the start"), ("every_s", "one every N seconds"), ("poisson_per_h", "random, N per hour"), ("at", "at these times (s)")]

# --- Fit check ---

def check(sc: dict, tier: str = "auto") -> str:
    """Builds the plan with every value at its median (no work run): memory per machine, and the first reason the plan cannot exist."""
    asio, _, _ = B.libs()
    from asio_core.sim import Sim
    V = asio.Valuer("median", tier, 0, sc.get("overrides", {}))
    try: sim = Sim(copy.deepcopy(sc), B.cat(), V, 0)
    except (MemoryError, ValueError, KeyError, TypeError) as e: return f"""<div class="asio-err"><b>This plan cannot run:</b> {B.esc(e)}</div>"""
    bars = "".join(f"""<div class="asio-mem" title="{B.esc('; '.join(f'{w} {g} GB' for w, g in n.uses))}"><span class="asio-mem-name">{B.esc(k)}</span><span class="asio-bar"><i style="width:{min(n.used_gb / n.usable_gb, 1) * 100 if n.usable_gb else 100:.0f}%"></i></span><span class="asio-mem-n">{n.used_gb:.1f} of {n.usable_gb:.1f} GB</span></div>""" for k, n in sim.pool.nodes.items())
    ws_ = [t for w in sc.get("workloads", []) for t in ([w] + [{"role": w.get("role"), **x} for x in w.get("types", [])] if w.get("kind") == "mix" else [w])]
    roles = {w.get(f) for w in ws_ for f in ("role", "decide_role") if w.get(f)} | {r.get("role") for w in ws_ for r in (w.get("routes") or {}).values()} - {None}
    missing = sorted(r for r in roles if r not in (sc.get("routes") or {}) and r not in sim.engines)
    warn = f"""<div class="asio-err">Workloads use routes no engine serves: {B.esc(', '.join(missing))}. Add them under Routes.</div>""" if missing else ""
    return f"""{warn}<div class="asio-fit"><span class="asio-ok">Fits</span>{bars}</div>"""

# --- Scenario bar ---

def bar(ws, name: str, sc: dict) -> str:
    names = [(n, n) for n in ws.names()]
    vs = ws.versions(name)
    return f"""<div class="asio-scbar row">{F.sel("", name, names, t="asio_sc", op="open", w="asio-w12")}
        {F.inp("label", sc.get("label", ""), "str", "what this scenario is", w="grow")}
        <span class="asio-dim" title="every change is saved; a version is a copy you can go back to">saved {B.ago(ws.changed(name))}, {len(vs)} versions</span>
        {B.btn("Save version", "asio_sc", op="checkpoint")}{B.btn("Versions", "asio_sc", op="versions")}{B.btn("Duplicate", "asio_sc", op="dup")}
        <input class="asio-in asio-w8" name="newname" placeholder="new name" id="asio-newname">{B.btn("Rename", "asio_sc", op="rename", include="#asio-newname")}
        {B.btn("Delete", "asio_sc", cls="asio-btn asio-danger", op="delete", confirm=f"Move scenario {name} and all its versions to the workspace trash?")}</div>
        <div class="asio-scbar row"><span class="asio-dim">New scenario from</span>{start_select()}{B.btn("Blank", "asio_sc", op="new_blank")}</div>"""

def start_select() -> str:
    c = B.cat()
    return F.sel("", "", [(i, f"""{i}: {c.get('scenarios', i).get('label', '')}""") for i in c.ids("scenarios")], t="asio_sc", op="new_from", blank="(a starting scenario)", w="asio-w20")

def versions_html(ws, name: str) -> str:
    rows = [[v["i"] + 1, B.ago(v["t"]), B.esc(v["why"]), B.btn("Restore", "asio_sc", op="restore", i=v["i"], confirm="Make this version the working copy? The current one is kept as a version first."), B.btn("Copy as new", "asio_sc", op="version_copy", i=v["i"])] for v in reversed(ws.versions(name))]
    return B.tbl(["#", "when", "why", "", ""], rows)

# --- Machines ---

def node_options(c, also: str = None) -> list:
    """Machines grouped yours first; also adds an abstract template a scenario already uses (a class of machine rather than one you have)."""
    own, other = [], []
    for i in c.ids("nodes") + ([also] if also and also not in c.ids("nodes") else []):
        e = c.get("nodes", i)
        (own if e.get("owned") else other).append((i, e.get("label", i)))
    return [("Yours", own), ("Others", other)]

def node_line(e: dict) -> str:
    """One machine in a line: RAM, bandwidth, compute, power, price."""
    bw = e.get("mem_bw_gbs") or (e["mem_mts"] * e["bus_bits"] / 8000 if e.get("mem_mts") and e.get("bus_bits") else None)
    gpu = f"""{e['gpu'].get('name', 'GPU')} {B.fmt(F.median(e['gpu'].get('tflops_fp16')))} TF""" if e.get("gpu") else "no GPU"
    ssd = sum(s.get("gb", 0) for s in e.get("ssd", []))
    return f"""{e.get('ram_gb', '?')} GB, {B.fmt(F.median(bw))} GB/s, {B.esc(gpu)}, CPU {B.fmt(F.median(e.get('cpu', {}).get('tflops')))} TF, SSD {ssd:g} GB, {B.fmt(F.median(e.get('power_w', {}).get('idle')))}-{B.fmt(F.median(e.get('power_w', {}).get('load')))} W, ${B.fmt(F.median(e.get('price_usd')))}"""

def node_spec(c, s) -> dict: return c.get("nodes", s) if isinstance(s, str) else {**c.get("nodes", s["template"]), **{k: v for k, v in s.items() if k != "template"}}

def nodes(sc: dict) -> str:
    c, rows = B.cat(), []
    for n, s in (sc.get("nodes") or {}).items():
        tid, e = s if isinstance(s, str) else s["template"], node_spec(c, s)
        ch = {} if isinstance(s, str) else s
        rows.append(f"""<div class="asio-card"><div class="row">{F.inp(n, n, "str", t="asio_node", op="rename", name=n, w="asio-w6", title="the name engines and links use")}{F.sel(n, tid, node_options(c, tid), t="asio_node", op="template", name=n, w="grow")}
            {F.field("RAM GB", F.inp(n, ch.get("ram_gb"), "float", e.get("ram_gb"), t="asio_node", op="change", name=n, field="ram_gb", w="asio-w4"), "change it to try a variant; empty uses the catalog's")}
            {F.field("MT/s", F.inp(n, ch.get("mem_mts"), "float", e.get("mem_mts"), t="asio_node", op="change", name=n, field="mem_mts", w="asio-w4"))}
            {B.btn("&#x2715;", "asio_node", cls="asio-btn asio-x", op="remove", name=n, title="remove this machine")}</div><div class="asio-line">{B.esc(ch["label"]) + ": " if ch.get("label") else ""}{node_line(e)}</div></div>""")
    add = f"""<div class="row asio-add">{F.sel("", "", node_options(c), t="asio_node", op="add", blank="+ add a machine", w="asio-w20")}</div>"""
    return "".join(rows) + add

# --- Links ---

def links(sc: dict) -> str:
    c, ns = B.cat(), list((sc.get("nodes") or {}))
    types = [(i, f"""{c.get('links', i).get('label', i)}""") for i in c.ids("links") if not i.startswith("magpie")]
    rows = []
    for i, l in enumerate(sc.get("links") or []):
        lt = c.get("links", l["type"])
        ends = F.multi(f"links.{i}.nodes", l.get("nodes"), [(n, n) for n in ns], t="asio_link", op="nodes", i=i) if "nodes" in l else f"""{F.sel(f"links.{i}.a", l.get("a"), [(n, n) for n in ns], w="asio-w6", re="links")} to {F.sel(f"links.{i}.b", l.get("b"), [(n, n) for n in ns], w="asio-w6", re="links")}"""
        rows.append(f"""<div class="asio-card row">{F.sel(f"links.{i}.type", l["type"], types, w="asio-w14", re="links")}{B.btn("shared" if "nodes" in l else "cable", "asio_link", op="mode", i=i, title="a cable joins two machines; a shared switch or hub joins any number")}{ends}
            <span class="asio-line">{F.show(lt.get('rate_gbps'), 'Gb/s')}, {F.show(lt.get('latency_us'), 'us per hop')}</span>{B.btn("&#x2715;", "asio_link", cls="asio-btn asio-x", op="remove", i=i)}</div>""")
    mag = F.sel("through_magpie", sc.get("through_magpie", ""), [(i, c.get("links", i).get("label", i)) for i in c.ids("links") if i.startswith("magpie")], blank="direct, not through Magpie", w="asio-w14", re="links")
    return "".join(rows) + f"""<div class="row asio-add">{F.sel("", "", types, t="asio_link", op="add", blank="+ add a link", w="asio-w14")}<span class="asio-dim">traffic between machines goes</span>{mag}</div>"""

# --- Engines ---

def model_options(c) -> list:
    groups = {}
    for i in c.ids("models"):
        m = c.get("models", i)
        groups.setdefault(m.get("kind", "other"), []).append((i, m.get("label", i)))
    return list(groups.items())

def engine_line(c, e: dict) -> str:
    """What the engine holds: weights at its quant, KV for its slots at full context."""
    m = c.get("models", e["model"])
    if m.get("kind") != "llm": return f"""{B.esc(m.get('label', ''))}: {B.fmt(m.get('gb') or m.get('params_b', 0) * m.get('precision_bytes', 1.0))} GB weights"""
    kv = F.median(m.get("kv_kb_per_token")) or 0
    ctx, keep = int(e.get("ctx", 8192)), max(int(e.get("slots", 1)), int(e.get("kv_keep", e.get("slots", 1))))
    kv_gb = keep * ctx * kv * 1e3 * m.get("kv_quant_bytes", 1.0625) / 2 / 1e9
    if m.get("arch") == "moe": return f"""{B.fmt(F.median(m.get('resident_gb')))} GB always held, {m.get('experts_gb', '?')} GB of experts cached as memory allows, KV {B.fmt(kv_gb)} GB ({keep} x {ctx:,} tokens)"""
    q = e.get("quant") or m.get("default_quant")
    w = m["params_b"] * c.get("models", "quants")["table"].get(q, [1, 1])[0]
    return f"""weights {B.fmt(w)} GB at {B.esc(q)}, KV {B.fmt(kv_gb)} GB ({keep} x {ctx:,} tokens), longest context {f"{m['ctx_max']:,}" if m.get('ctx_max') else 'not listed'}"""

def engines(sc: dict) -> str:
    c, ns, rows = B.cat(), list((sc.get("nodes") or {})), []
    quants = list(c.get("models", "quants")["table"])
    for i, e in enumerate(sc.get("engines") or []):
        m, p = c.get("models", e["model"]), f"engines.{i}"
        llm, moe = m.get("kind") == "llm", m.get("arch") == "moe"
        on = e.get("nodes") or ([e["node"]] if e.get("node") else [])
        place = F.field("placement", F.sel(f"{p}.placement", e.get("placement", ""), [("experts", "experts spread"), ("pipeline", "layers split")], blank="automatic", w="asio-w8")) if len(on) > 1 else ""
        llm_f = "".join([F.field("quant", F.sel(f"{p}.quant", e.get("quant", ""), [(q, q) for q in quants], blank=f"""default ({m.get('default_quant', '')})""", w="asio-w6", re="engines")) if not moe else "",
                         F.field("slots", F.inp(f"{p}.slots", e.get("slots", 1), "int", w="asio-w3", re="engines"), "streams served at once"), F.field("context", F.inp(f"{p}.ctx", e.get("ctx", 8192), "int", w="asio-w5", re="engines"), "tokens per slot"),
                         F.field("KV kept", F.inp(f"{p}.kv_keep", e.get("kv_keep"), "int", str(e.get("slots", 1)), w="asio-w3", re="engines"), "sessions whose KV stays for reuse"),
                         F.chk(f"{p}.mtp", e.get("mtp", True), "MTP") if m.get("mtp") else "", F.chk(f"{p}.draft", e.get("draft", True), "draft model") if m.get("draft") else "",
                         F.field("expert cache GB", F.inp(f"{p}.cache_gb", e.get("cache_gb"), "json", "all free memory", w="asio-w5", re="engines"), "a number for every machine, or {machine: GB}") if moe else "", heat_fields(c, i, e, m) if moe else ""]) if llm else F.field("slots", F.inp(f"{p}.slots", e.get("slots", 1), "int", w="asio-w3"))
        rows.append(f"""<div class="asio-card"><div class="row">{F.inp(p, e["id"], "str", t="asio_engine", op="rename", i=i, w="asio-w5", title="the name routes use")}{F.sel(f"{p}.model", e["model"], model_options(c), w="grow", re="engines,routes")}
            {F.field("program", F.sel(f"{p}.backend", e.get("backend", ""), [(b, c.get('backends', b).get('label', b)) for b in m.get("backends", [])], blank="first that fits", w="asio-w10"))}
            {F.field("on", F.multi(p, on, [(n, n) for n in ns], t="asio_engine", op="nodes", i=i))}{place}
            {F.chk(f"{p}.warm", e.get("warm", True), "loaded at start")}{B.btn("&#x2715;", "asio_engine", cls="asio-btn asio-x", op="remove", i=i)}</div>
            <div class="row">{llm_f}</div><div class="asio-line">{engine_line(c, e)}</div></div>""")
    return "".join(rows) + f"""<div class="row asio-add">{F.sel("", "", model_options(c), t="asio_engine", op="add", blank="+ add an engine (pick its model)", w="asio-w20")}</div>"""

HEAT_WORDS = [("", "unknown: the catalog's hit curve"), ("uniform", "every expert equally"), ("zipf", "skewed, typical range (uncertain)"), ("zipf_set", "skewed, exponent of my choice")]
ARR_WORDS = [("hottest", "known heat: hottest in RAM, load even"), ("learned", "unknown, caches learn (LRU)"), ("spread", "unknown, fixed random split")]

def heat_fields(c, i: int, e: dict, m: dict) -> str:
    """Which experts tokens choose (heat) and how experts are placed and cached (arrangement), for a MoE engine."""
    h = e.get("heat")
    mode = "" if not h else h if isinstance(h, str) and h in ("uniform",) else ("zipf" if h.get("zipf") is None else "zipf_set") if isinstance(h, dict) else f"id:{h}"
    entries = [(f"id:{x}", f"""measured: {c.get('heat', x).get('label', x)}""") for x in c.ids("heat") if c.get("heat", x).get("model") == m["id"]]
    skew = h.get("zipf") if isinstance(h, dict) and h.get("zipf") is not None else ""
    form = f"""<form class="asio-multi" onsubmit="return false" {B.act("asio_engine", trigger="change", op="heat", i=i)}><select class="asio-in asio-w14" name="mode">{''.join(f'<option value="{k}" {"selected" if k == mode else ""}>{B.esc(l)}</option>' for k, l in HEAT_WORDS + entries)}</select>{f'<input class="asio-in asio-w4" name="skew" value="{B.esc(F.typed(skew))}" placeholder="0.8" title="Zipf exponent: 0 is even, 1 strongly skewed; three numbers for a range">' if mode == "zipf_set" else ""}</form>"""
    arr = F.field("arrangement", F.sel(f"engines.{i}.arrangement", e.get("arrangement", "learned"), ARR_WORDS, w="asio-w14", re="engines")) if h else ""
    return F.field("expert heat", form, "which experts tokens choose; Catalog > Expert heat holds measured tables") + arr

# --- Routes ---

def routes(sc: dict) -> str:
    ids = [e["id"] for e in sc.get("engines") or []]
    rows = [f"""<div class="asio-card row"><span class="asio-w6 asio-role">{B.esc(r)}</span>{F.multi(f"routes.{r}", es, [(i, i) for i in ids], t="asio_route", op="set", role=r)}{B.btn("&#x2715;", "asio_route", cls="asio-btn asio-x", op="remove", role=r)}</div>""" for r, es in (sc.get("routes") or {}).items()]
    return "".join(rows) + f"""<div class="row asio-add"><input class="asio-in asio-w8" name="role" id="asio-newrole" placeholder="route name">{B.btn("+ add a route", "asio_route", op="add", include="#asio-newrole")}<span class="asio-dim">a route is a name workloads ask for; work goes to the least busy engine on it</span></div>"""

# --- Workloads ---

def arrival(i: int, a) -> str:
    mode = "all" if a in (None, "all") else next(iter(a))
    val = "" if mode == "all" else a[mode]
    return f"""<form class="row asio-multi" onsubmit="return false" {B.act("asio_work", trigger="change", op="arrival", i=i)}><select class="asio-in asio-w12" name="mode">{''.join(f'<option value="{k}" {"selected" if k == mode else ""}>{lbl}</option>' for k, lbl in ARRIVALS)}</select><input class="asio-in asio-w8" name="v" value="{B.esc(F.typed(val))}" placeholder="N"></form>"""

def wfields(p: str, w: dict, specs: list, roles: list) -> str:
    out = []
    for key, name, kind, *dflt in specs:
        v = F.get_path(w, key)
        ctl = F.sel(f"{p}.{key}", v, [(r, r) for r in roles], blank="(none)", w="asio-w8") if kind == "role" else F.chk(f"{p}.{key}", v if v is not None else True, "") if kind == "bool" else F.inp(f"{p}.{key}", v, kind, F.typed(dflt[0]) if dflt else "", w="asio-w16" if kind == "json" else "asio-w8")
        out.append(F.field(name, ctl, "one number, or low most-likely high" if kind in ("est", "esti") else ""))
    return "".join(out)

def mix_types(i: int, w: dict, roles: list) -> str:
    """The job types of a demand mix, each a small workload of its own with a share of the arrivals."""
    kinds = [(k, v) for k, v in KIND_WORDS.items() if k != "mix"]
    rows = "".join(f"""<div class="asio-card asio-type"><div class="row">{F.inp(f"workloads.{i}.types.{j}.name", t.get("name", ""), "str", w="asio-w8")}{F.field("share", F.inp(f"workloads.{i}.types.{j}.share", t.get("share", 1), "float", w="asio-w4"))}
        {F.sel(f"workloads.{i}.types.{j}.kind", t["kind"], kinds, t="asio_work", op="type_kind", i=i, j=j, w="asio-w14")}{B.btn("&#x2715;", "asio_work", cls="asio-btn asio-x", op="type_remove", i=i, j=j)}</div>
        <div class="row asio-wrap">{wfields(f"workloads.{i}.types.{j}", t, KIND_FIELDS.get(t["kind"], []) + COMMON[1:], roles)}</div></div>""" for j, t in enumerate(w.get("types", [])))
    share = sum(float(t.get("share", 1)) for t in w.get("types", [])) or 1
    split = ", ".join(f"""{B.esc(t.get('name', t['kind']))} {float(t.get('share', 1)) / share:.0%}""" for t in w.get("types", []))
    return f"""<div class="asio-dim">Job types ({split}):</div>{rows}<div class="row asio-add">{F.sel("", "", kinds, t="asio_work", op="type_add", i=i, blank="+ add a job type", w="asio-w14")}</div>"""

def work(sc: dict) -> str:
    roles, rows = sorted(set(sc.get("routes") or {}) | {e["id"] for e in sc.get("engines") or []}), []
    for i, w in enumerate(sc.get("workloads") or []):
        p, kind = f"workloads.{i}", w["kind"]
        sub = (F.chk(f"{p}.subagents", bool(w.get("subagents")), "subagents", t="asio_work", op="subagents", i=i) + (wfields(p, w, SUB_FIELDS, roles) if w.get("subagents") else "")) if kind == "agent" else ""
        when = "" if kind == "mix" else f"""{F.field("how many", F.inp(f"{p}.count", w.get("count"), "int", str(len(w["items"])) if w.get("items") else "1", w="asio-w4"))}{F.field("arriving", arrival(i, w.get("arrival")))}"""
        rows.append(f"""<div class="asio-card"><div class="row">{F.inp(f"{p}.name", w.get("name", ""), "str", w="asio-w8")}{F.sel(f"{p}.kind", kind, list(KIND_WORDS.items()), t="asio_work", op="kind", i=i, w="asio-w14")}
            {when}{F.field("start after (s)", F.inp(f"{p}.start_s", w.get("start_s"), "float", "0", w="asio-w4"))}
            {B.btn("Copy", "asio_work", op="dup", i=i)}{B.btn("&#x2715;", "asio_work", cls="asio-btn asio-x", op="remove", i=i)}</div>
            <div class="row asio-wrap">{wfields(p, w, KIND_FIELDS.get(kind, []) + ([] if kind == "mix" else COMMON), roles)}</div>{f'<div class="row asio-wrap">{sub}</div>' if sub else ''}{mix_types(i, w, roles) if kind == "mix" else ''}</div>""")
    return "".join(rows) + f"""<div class="row asio-add">{F.sel("", "", list(KIND_WORDS.items()), t="asio_work", op="add", blank="+ add work", w="asio-w14")}</div>"""

# --- Values and tweaks ---

def tweak_text(v):
    """A tweak as typed: a number, or the three numbers of an estimate."""
    return v["est"] if isinstance(v, dict) and set(v) == {"est"} else v

def values(sc: dict, tier: str = "auto") -> str:
    """Every value a median run of this scenario reads: the number used, its tiers and source, and a field to set it for this scenario only."""
    asio, _, _ = B.libs()
    from asio_core.sim import Sim
    from asio_core.workloads import arrivals
    V = asio.Valuer("median", tier, 0, sc.get("overrides", {}))
    try:
        sim = Sim(copy.deepcopy(sc), B.cat(), V, 0)
        sim.run(arrivals(sim, sc["workloads"], False), sc.get("horizon_s"))
    except (MemoryError, ValueError, KeyError, TypeError) as e:
        if not V.read: return B.err(f"The plan cannot run, so its values cannot be listed: {e}")
    ov = sc.get("overrides", {})
    rows = []
    for k in sorted(V.read):
        r = V.read[k]
        eff = r["override"] if r["override"] is not None else r["spec"]
        used = V.draws.get((eff.get("group") or k) if isinstance(eff, dict) else k, eff if isinstance(eff, (int, float)) else None)
        changed = k in ov
        rows.append([f"""<span title="{B.esc(k)}">{B.esc(F.value_label(k))}</span>""", f"""<b>{B.fmt(used)}</b>""" if changed else B.fmt(used), F.show(r["spec"]), F.inp("", tweak_text(ov.get(k)), "est", "catalog's", t="asio_tweak", key=k, w="asio-w8")])
    extra = [[f"""<span title="{B.esc(k)}">{B.esc(k)} (pattern)</span>""", "", "", F.inp("", tweak_text(v), "est", t="asio_tweak", key=k, w="asio-w8")] for k, v in ov.items() if k not in V.read]
    add = f"""<div class="row asio-add"><input class="asio-in asio-w14" name="key" id="asio-tweakkey" placeholder="key or pattern, e.g. node.*.throttle"><input class="asio-in asio-w8" name="v" id="asio-tweakv" placeholder="value">{B.btn("Set", "asio_tweak", include="#asio-tweakkey,#asio-tweakv", kind="est")}</div>"""
    return f"""<div class="asio-dim">{len(rows)} values, read at their median. A value set here applies to this scenario only; the catalog is unchanged. Hover a name for its key, a number for its source.</div>{B.tbl(["value", "used", "catalog", "this scenario"], rows + extra)}{add}"""

# --- Whole tab ---

SECTIONS = {"nodes": ("Machines", nodes), "links": ("Links", links), "engines": ("Engines", engines), "routes": ("Routes", routes), "work": ("Workloads", work)}

def section(k: str, sc: dict) -> str:
    try: return SECTIONS[k][1](sc)
    except (KeyError, ValueError, TypeError) as e: return B.err(f"{SECTIONS[k][0]} cannot be shown: {type(e).__name__}: {e}. The JSON view below can fix it.")

def page(ws, name: str) -> str:
    if not name: return f"""<div class="asio-pad"><div class="asio-note">No scenario yet. Start from one of the catalog's, or a blank one; it is saved in your workspace as you change it.</div><div class="row">{start_select()}{B.btn("Blank", "asio_sc", op="new_blank")}</div></div>"""
    sc = ws.get(name)
    secs = "".join(B.sec(t, f"""<div id="asio-b-{k}">{section(k, sc)}</div>""") for k, (t, _) in SECTIONS.items())
    vals = B.sec("Values and tweaks", f"""<div id="asio-b-values" {B.act("asio_values", trigger="toggle once from:closest details")}><span class="asio-dim">Opening lists every value this scenario reads.</span></div>""", open_=False)
    js = B.sec("JSON (advanced)", f"""<textarea class="asio-in asio-json" name="json" id="asio-jsonbox" spellcheck="false">{B.esc(json.dumps(sc, indent=1))}</textarea><div class="row">{B.btn("Apply JSON", "asio_sc", op="json", include="#asio-jsonbox")}<span class="asio-dim">replaces the working copy; the current one is kept as a version first</span></div>""", open_=False)
    return f"""<div class="asio-pad"><div id="asio-scbar">{bar(ws, name, sc)}</div><div id="asio-check">{check(sc)}</div><div class="row">{F.field("give up after (s of simulated time)", F.inp("horizon_s", sc.get("horizon_s"), "float", "until all work is done", w="asio-w8"))}</div>{secs}{vals}{js}</div>"""

# --- Intents ---

BLANK = {"label": "new scenario", "nodes": {}, "links": [], "engines": [], "routes": {}, "workloads": []}

def current(request):
    ws = B.ws(request)
    name = ws.ui().get("scenario")
    if not name or not ws.exists(name): raise ValueError("no scenario is open: pick or start one on the Build tab")
    return ws, name, ws.get(name)

def say(imr, msg: str, bad: bool = True):
    return imr.oob(f"""<div class="{'asio-err' if bad else 'asio-note'}">{B.esc(msg)}</div>""", "asio-msg")

def after(imr, ws, name: str, sc: dict, re: str = ""):
    """Saves the working copy, then refreshes the fit check, the sections named in re, and the reference panel when machines or engines changed."""
    from asio_ui import ref
    ws.put(name, sc)
    imr.oob(check(sc), "asio-check").oob("", "asio-msg")
    for k in filter(None, re.split(",")): imr.oob(section(k, sc), f"asio-b-{k}")
    if {"nodes", "engines"} & set(re.split(",")): imr.oob(ref.panel(ws), "asio-ref")
    return imr

def guarded(fn):
    """Intent wrapper: a value that cannot be read, or a scenario that is not open, is reported in the message line instead of failing silently."""
    async def h(request, payload, imr):
        try: return await fn(request, payload, imr)
        except (ValueError, KeyError, json.JSONDecodeError) as e: return say(imr, f"{type(e).__name__}: {e}")
    return h

@guarded
async def h_set(request, payload, imr):
    ws, name, sc = current(request)
    path, kind = payload["path"], payload.get("kind", "str")
    v = F.parse(kind, payload.get("v"))
    if path.endswith(".model"):
        e, m = F.get_path(sc, path.rsplit(".", 1)[0]), B.cat().get("models", v)
        if e.get("backend") not in m.get("backends", []): e.pop("backend", None)
        if m.get("kind") != "llm" or m.get("arch") == "moe": e.pop("quant", None)
    F.set_path(sc, path, v if v != "" else None)
    return after(imr, ws, name, sc, payload.get("re", ""))

@guarded
async def h_node(request, payload, imr):
    ws, name, sc = current(request)
    op, n, v, ns = payload["op"], payload.get("name"), payload.get("v"), sc.setdefault("nodes", {})
    if op == "add":
        if not v: return imr
        k = next(f"n{i}" for i in range(1, 999) if f"n{i}" not in ns)
        ns[k] = v
    elif op == "template": ns[n] = v if isinstance(ns[n], str) else {**ns[n], "template": v}
    elif op == "change":
        val = F.parse("float", v)
        s = {"template": ns[n]} if isinstance(ns[n], str) else ns[n]
        F.set_path(s, payload["field"], val)
        ns[n] = s["template"] if set(s) == {"template"} else s
    elif op == "rename":
        new = F.parse("str", v)
        if not new or new == n: return imr
        if new in ns: raise ValueError(f"there is already a machine called {new}")
        sc["nodes"] = {new if k == n else k: s for k, s in ns.items()}
        for e in sc.get("engines", []):
            if e.get("node") == n: e["node"] = new
            if e.get("nodes"): e["nodes"] = [new if x == n else x for x in e["nodes"]]
            if isinstance(e.get("cache_gb"), dict) and n in e["cache_gb"]: e["cache_gb"][new] = e["cache_gb"].pop(n)
        for l in sc.get("links", []):
            for f in ("a", "b"):
                if l.get(f) == n: l[f] = new
            if l.get("nodes"): l["nodes"] = [new if x == n else x for x in l["nodes"]]
        sc["overrides"] = {(f"node.{new}." + k[len(f"node.{n}."):] if k.startswith(f"node.{n}.") else k): x for k, x in sc.get("overrides", {}).items()}
    elif op == "remove":
        used = [e["id"] for e in sc.get("engines", []) if n == e.get("node") or n in (e.get("nodes") or [])]
        if used: raise ValueError(f"engines {used} run on {n}: move or remove them first")
        ns.pop(n)
        sc["links"] = [l for l in sc.get("links", []) if n not in (l.get("a"), l.get("b")) and (len([x for x in l.get("nodes", []) if x != n]) > 1 if "nodes" in l else True)]
        for l in sc["links"]:
            if "nodes" in l: l["nodes"] = [x for x in l["nodes"] if x != n]
    return after(imr, ws, name, sc, "nodes,links,engines")

@guarded
async def h_link(request, payload, imr):
    ws, name, sc = current(request)
    op, ls, ns = payload["op"], sc.setdefault("links", []), list(sc.get("nodes") or {})
    i = int(payload.get("i", -1))
    if op == "add":
        if not payload.get("v"): return imr
        if len(ns) < 2: raise ValueError("a link needs two machines: add machines first")
        ls.append({"type": payload["v"], "a": ns[0], "b": ns[1]})
    elif op == "mode": ls[i] = {"type": ls[i]["type"], "a": ls[i]["nodes"][0], "b": ls[i]["nodes"][-1]} if "nodes" in ls[i] else {"type": ls[i]["type"], "nodes": [ls[i]["a"], ls[i]["b"]]}
    elif op == "nodes": ls[i]["nodes"] = F.parse("list", payload.get("v"))
    elif op == "remove": ls.pop(i)
    return after(imr, ws, name, sc, "links")

def _eid(sc: dict, model: str) -> str:
    base = model.split("-")[0].replace(".", "")[:8] or "e"
    have = {e["id"] for e in sc.get("engines", [])}
    return base if base not in have else next(f"{base}{k}" for k in range(2, 999) if f"{base}{k}" not in have)

@guarded
async def h_engine(request, payload, imr):
    ws, name, sc = current(request)
    op, es = payload["op"], sc.setdefault("engines", [])
    i = int(payload.get("i", -1))
    if op == "add":
        if not payload.get("v"): return imr
        ns = list(sc.get("nodes") or {})
        if not ns: raise ValueError("an engine runs on a machine: add a machine first")
        es.append({"id": _eid(sc, payload["v"]), "model": payload["v"], "node": ns[0], "slots": 1, "ctx": 8192})
    elif op == "rename":
        new, old = F.parse("str", payload.get("v")), es[i]["id"]
        if not new or new == old: return imr
        if new in {e["id"] for e in es}: raise ValueError(f"there is already an engine called {new}")
        es[i]["id"] = new
        sc["routes"] = {r: [new if x == old else x for x in ids] for r, ids in (sc.get("routes") or {}).items()}
        sc["overrides"] = {(f"engine.{new}." + k[len(f"engine.{old}."):] if k.startswith(f"engine.{old}.") else k): x for k, x in sc.get("overrides", {}).items()}
    elif op == "nodes":
        on = F.parse("list", payload.get("v"))
        if not on: raise ValueError("an engine needs at least one machine")
        for k in ("node", "nodes"): es[i].pop(k, None)
        es[i].update({"node": on[0]} if len(on) == 1 else {"nodes": on})
    elif op == "heat":
        mode, e = payload.get("mode", ""), es[i]
        if not mode: e.pop("heat", None)
        elif mode == "zipf": e["heat"] = {"zipf": None}
        elif mode == "zipf_set": e["heat"] = {"zipf": F.parse("est", payload.get("skew")) or 0.8}
        else: e["heat"] = mode[3:] if mode.startswith("id:") else mode
        if e.get("heat"): e.setdefault("arrangement", "learned")
        else: e.pop("arrangement", None)
    elif op == "remove":
        old = es.pop(i)["id"]
        sc["routes"] = {r: [x for x in ids if x != old] for r, ids in (sc.get("routes") or {}).items()}
    return after(imr, ws, name, sc, "engines,routes,work")

@guarded
async def h_route(request, payload, imr):
    ws, name, sc = current(request)
    op, rs = payload["op"], sc.setdefault("routes", {})
    role = F.parse("str", payload.get("role"))
    if op == "add":
        if not role: raise ValueError("give the route a name")
        rs.setdefault(role, [])
    elif op == "set": rs[role] = F.parse("list", payload.get("v"))
    elif op == "remove": rs.pop(role, None)
    return after(imr, ws, name, sc, "routes,work")

@guarded
async def h_work(request, payload, imr):
    ws, name, sc = current(request)
    op, wl = payload["op"], sc.setdefault("workloads", [])
    i = int(payload.get("i", -1))
    if op == "add":
        if not payload.get("v"): return imr
        k = payload["v"]
        role = next(iter(sc.get("routes") or {}), None) or next((e["id"] for e in sc.get("engines", [])), "main")
        wl.append({"name": f"{k}{len(wl) + 1}", "kind": k, "count": 1, **({"role": role} if k not in ("routed", "pipeline") else {}), **copy.deepcopy(KIND_DEFAULTS.get(k, {}))})
    elif op == "kind":
        w, k = wl[i], payload["v"]
        keep = {f: w[f] for f in ("name", "count", "arrival", "start_s", "role") if f in w}
        wl[i] = {**keep, "kind": k, **copy.deepcopy(KIND_DEFAULTS.get(k, {}))}
    elif op == "arrival":
        mode, raw = payload.get("mode", "all"), payload.get("v")
        if mode == "all": wl[i].pop("arrival", None)
        elif mode == "at": wl[i]["arrival"] = {"at": [float(x) for x in str(raw or "0").replace(",", " ").split()]}
        else: wl[i]["arrival"] = {mode: F.parse("float", raw) or 1.0}
    elif op == "subagents":
        if F.parse("bool", payload.get("v")): wl[i]["subagents"] = {"every": 2, "n": 2, "role": wl[i].get("role", "main"), "steps_needed": [2, 4, 8], "system": 2000, "summary_out": 300}
        else: wl[i].pop("subagents", None)
    elif op == "type_add":
        if not payload.get("v"): return imr
        wl[i].setdefault("types", []).append({"name": f"""{payload["v"]}{len(wl[i].get("types", [])) + 1}""", "kind": payload["v"], "share": 1, **copy.deepcopy(KIND_DEFAULTS.get(payload["v"], {}))})
    elif op == "type_kind":
        t, j = wl[i]["types"], int(payload["j"])
        t[j] = {**{f: t[j][f] for f in ("name", "share", "role", "slo_s") if f in t[j]}, "kind": payload["v"], **copy.deepcopy(KIND_DEFAULTS.get(payload["v"], {}))}
    elif op == "type_remove": wl[i]["types"].pop(int(payload["j"]))
    elif op == "dup": wl.insert(i + 1, {**copy.deepcopy(wl[i]), "name": f"""{wl[i].get("name", "w")}-copy"""})
    elif op == "remove": wl.pop(i)
    return after(imr, ws, name, sc, "work")

@guarded
async def h_tweak(request, payload, imr):
    ws, name, sc = current(request)
    key = (payload.get("key") or "").strip()
    if not key: raise ValueError("give the key of the value to set")
    v = F.parse("est", payload.get("v"))
    ov = sc.setdefault("overrides", {})
    if v is None: ov.pop(key, None)
    else: ov[key] = {"est": v} if isinstance(v, list) else v
    after(imr, ws, name, sc)
    return imr.oob(values(sc), "asio-b-values")

@guarded
async def h_values(request, payload, imr):
    ws, name, sc = current(request)
    return imr.oob(values(sc), "asio-b-values")

@guarded
async def h_sc(request, payload, imr):
    """Scenario actions: open, new (from the catalog or blank), checkpoint, versions, restore, copy, rename, delete (to the trash), apply JSON."""
    ws, op = B.ws(request), payload["op"]
    name = ws.ui().get("scenario")
    if op == "open" and payload.get("v"): name = payload["v"]
    elif op == "new_from":
        if not payload.get("v"): return imr
        name = ws.create(payload["v"], {k: v for k, v in B.cat().get("scenarios", payload["v"]).items() if k != "id"}, f"""copied from the catalog's {payload["v"]}""")
    elif op == "new_blank": name = ws.create("new", copy.deepcopy(BLANK), "blank")
    elif op == "checkpoint": return say(imr, f"Saved as version {ws.checkpoint(name, 'saved by hand')}.", bad=False)
    elif op == "versions": return imr.oob(versions_html(ws, name), "asio-msg")
    elif op == "restore": ws.restore(name, int(payload["i"]))
    elif op == "version_copy": name = ws.create(f"{name}-v{int(payload['i']) + 1}", ws.version(name, int(payload["i"])), f"""copy of {name} version {int(payload['i']) + 1}""")
    elif op == "dup": name = ws.create(name, ws.get(name), f"copy of {name}")
    elif op == "rename":
        if not (payload.get("newname") or "").strip(): raise ValueError("type the new name first")
        name = ws.rename(name, payload["newname"])
    elif op == "delete":
        ws.trash(name)
        name = next(iter(ws.names()), None)
    elif op == "json":
        sc = json.loads(payload.get("json") or "{}")
        if not isinstance(sc, dict): raise ValueError("a scenario is a JSON object")
        ws.checkpoint(name, "before applying JSON")
        ws.put(name, sc)
    ws.set_ui(scenario=name)
    from asio_ui import ref
    return imr.oob(page(ws, name), "asio-build").oob(ref.panel(ws), "asio-ref").oob("", "asio-msg")

INTENTS = {"asio_set": h_set, "asio_node": h_node, "asio_link": h_link, "asio_engine": h_engine, "asio_route": h_route, "asio_work": h_work, "asio_tweak": h_tweak, "asio_values": h_values, "asio_sc": h_sc}
