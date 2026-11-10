"""Nodes - the Magpie nodes of this portal's resource pool, managed from AI Tools Settings: what each node is, which modules it runs or could install (and what installing takes), its services, file locations, settings and logs.
Every view and change here is a call to the node's own API with its Magpie connection's key (changes need an admin key). The node decides and keeps the record of what it is, so nothing about a node is stored in the portal; a node is added as a Magpie connection."""
import re, json, html, asyncio, httpx
from fastapi import APIRouter, Request
from fastapi.responses import HTMLResponse

TOOL_META = {"label": "Nodes", "group": "system", "icon": "&#x1F426;", "description": "Magpie nodes: modules, services, locations, settings and logs", "singleton": True, "settings_page": True}
router = APIRouter(redirect_slashes=False)
ENV, AIM, IM, _P = {}, None, None, "/module/ai_tools/nodes"
INSTALLS: dict = {}   # (connection id, module) -> {"lines": [...], "done": bool}: installs streaming in the background, shown while the page polls
_TASKS: set = set()
_STATE = {"active": "nd-ok", "installed": "nd-mid", "installable": "", "blocked": "nd-bad"}

def _esc(s) -> str: return html.escape("" if s is None else str(s), quote=True)
def _attr(name: str, v: str) -> str: return f""" {name}="{_esc(v)}" """.rstrip() if v else ""
def _act(t: str, confirm: str = None, include: str = None, trigger: str = None, **kw) -> str: return f"""hx-post="/im/in" hx-swap="none" hx-vals="{_esc(json.dumps({"type": t, "branch": IM.branch_id, "lvl": 2, **kw}))}"{_attr("hx-include", include)}{_attr("hx-trigger", trigger)}{_attr("hx-confirm", confirm)}"""
def _btn(label: str, t: str, cls: str = "ui-btn", **kw) -> str: return f"""<button type="button" class="{cls}" {_act(t, **kw)}>{label}</button>"""
def _dim(text: str) -> str: return f"""<div class="nd-dim">{text}</div>""" if text else ""
def _dur(s) -> str: return f"{int(s) // 86400}d {int(s) % 86400 // 3600}h" if s >= 86400 else f"{int(s) // 3600}h {int(s) % 3600 // 60}m"
def _nodes() -> list: return AIM.connections.list_conns(conn_type="magpie")
def _conn(cid: str = ""): return AIM.connections.get_conn(cid) if cid else next(iter(_nodes()), None)
def _value(raw: str):
    """A typed setting value: JSON when it parses (numbers, true, lists, objects), else the text itself (a path, a model name)."""
    try: return json.loads(raw)
    except ValueError: return raw

# --- Node API ---

async def _call(c: dict, method: str, path: str, body=None) -> tuple:
    """(HTTP status, answer) from the node; 0 and {"error"} when it cannot be reached - an unreachable node is something to show, not a fault here."""
    try:
        async with httpx.AsyncClient(timeout=60.0, trust_env=False) as cl: r = await cl.request(method, AIM.connections._base(c) + path, headers=AIM.connections._auth_headers(c), json=body)
    except httpx.ConnectError as e: return 0, {"error": f"{type(e).__name__}: {e}. {AIM.connections.unreachable_hint(c)}"}
    except httpx.HTTPError as e: return 0, {"error": f"{type(e).__name__}: {e} ({AIM.connections._base(c)})"}
    try: return r.status_code, r.json()
    except ValueError: return r.status_code, {"error": r.text[:300] or f"HTTP {r.status_code}"}

def _err(code: int, d: dict) -> str:
    hint = " - this needs the node's admin key in the connection" if code in (401, 403) else " - this node's Magpie is older than this page: update it" if code == 404 else ""
    return f"""<div class="nd-err">{_esc(d.get("error") or d.get("detail") or f"HTTP {code}")}{hint}</div>"""

async def _install(cid: str, name: str, body: dict):
    """Streams the node's install progress (NDJSON) into INSTALLS; the node keeps going if this portal restarts, only the view here is lost."""
    rec = INSTALLS[(cid, name)]
    c = _conn(cid)
    try:
        async with httpx.AsyncClient(timeout=httpx.Timeout(connect=10.0, read=None, write=30.0, pool=10.0), trust_env=False) as cl:
            async with cl.stream("POST", f"{AIM.connections._base(c)}/modules/{name}/install", headers=AIM.connections._auth_headers(c), json=body) as r:
                if r.status_code != 200: rec["lines"].append(f"HTTP {r.status_code}: {(await r.aread()).decode(errors='replace')[:300]}")
                async for line in r.aiter_lines():
                    if not line.strip(): continue
                    p = json.loads(line)
                    rec["lines"].append(str(p.get("status") or p.get("error") or p.get("message") or line))
    except httpx.HTTPError as e: rec["lines"].append(f"connection lost: {type(e).__name__}: {e} - the node may still be installing; refresh to see its state")
    rec["done"] = True

# --- Page parts ---

def _summary(c: dict, n: dict) -> str:
    h, cid = n["hardware"], c["_id"]
    gpus = ", ".join(g.get("vendor", "?") for g in h.get("gpus", [])) or "none"
    warn = " | <b class='nd-bad'>open: no key needed</b>" if n.get("auth_open") else ""
    hw = f"""{_esc(h.get("cpu"))} | {h.get("threads")} threads | {h.get("ram_gb")} GB | {_esc(h.get("arch"))} | GPU {_esc(gpus)} | up {_dur(n["uptime_s"])} | settings version {n["policy_version"]}{warn}"""
    can = f"""can do now: {_esc(", ".join(sorted(n["capabilities"])) or "nothing yet")}"""
    return f"""<div class="nd-sec"><div class="row"><b>{_esc(n["node"])}</b><span class="nd-dim mono">{_esc(AIM.connections._base(c))}</span><span class="grow"></span>{_btn("Refresh", "nodes_open", cid=cid)}{_btn("Restart node", "nodes_restart", cid=cid, confirm="Restart this node? Running work stops; it starts again with its current settings and modules.")}</div>{_dim(hw)}{_dim(can)}</div>"""

def _module_row(cid: str, m: dict) -> str:
    name, st = m["name"], m["state"]
    acts = {"active": _btn("Deactivate", "nodes_mod", cid=cid, name=name, verb="deactivate"), "installed": _btn("Activate", "nodes_mod", cid=cid, name=name, verb="activate") + _btn("Remove", "nodes_mod", cls="ui-btn nd-danger", cid=cid, name=name, verb="remove", confirm=f"Remove what {name} installed (its environment)? Its models and results stay."), "installable": _btn("Install...", "nodes_install_form", cid=cid, name=name)}.get(st, "")
    sets = _btn("Settings", "nodes_mod_settings", cid=cid, name=name) if m.get("settings") else ""
    info = " | ".join(x for x in (f"""driver {m["installed"]["driver"]}""" if m.get("installed") else "", f"""provides {", ".join(m["provides"])}""" if m.get("provides") else "") if x)
    blocks = "; ".join(f"""{d["driver"]}: {b["block"]}""" for d in (m.get("plan") or {}).get("drivers", []) for b in d.get("blocks", [])) if st == "blocked" else ""
    err = f"""<div class="nd-err">{_esc(m["error"])}</div>""" if m.get("error") else ""
    live = _install_status(cid, name) if (cid, name) in INSTALLS else ""
    return f"""<div class="list-row"><div class="row"><span class="{_STATE.get(st, "")}">&#x25CF;</span><b>{_esc(name)}</b><span class="chip">{st}</span><span class="nd-dim grow">{_esc(m.get("label", ""))}</span>{sets}{acts}</div>{_dim(_esc(info))}{_dim(_esc("blocked by " + blocks) if blocks else "")}{err}<div id="nd-m-{_esc(name)}">{live}</div></div>"""

def _modules(c: dict, n: dict) -> str:
    rows = "".join(_module_row(c["_id"], m) for m in sorted(n["modules"], key=lambda m: (m["state"] != "active", m["state"] != "installed", m["name"])))
    return f"""<div class="nd-sec"><b>Modules</b>{_dim("What this node runs or could run. Install shows its options and what it needs from the system before anything is done.")}{rows}</div>"""

def _option(v, selected: bool, label: str = None) -> str: return f"""<option value="{_esc(v)}" {"selected" if selected else ""}>{_esc(v if label is None else label)}</option>"""

def _opt_field(k: str, o: dict, must: bool) -> str:
    choices = "".join(_option(ch, str(ch) == str(o.get("default"))) for ch in o.get("choices") or [])
    field = f"""<select name="opt_{_esc(k)}" class="module-select">{choices}</select>""" if choices else f"""<input name="opt_{_esc(k)}" class="module-select grow mono" value="{_esc(o.get("default", ""))}" {'placeholder="must be decided"' if must else ""}>"""
    return f"""<div class="row"><b class="nd-key">{_esc(k)}</b>{field}<span class="nd-dim grow">{_esc(o.get("hint", ""))}</span></div>"""

def _install_form(cid: str, m: dict, driver: str = "") -> str:
    p, name = m["plan"], m["name"]
    d = next((x for x in p["drivers"] if x["driver"] == (driver or p["driver"])), None) or {"driver": "", "options": {}, "decide": [], "needs_system": None, "blocks": [], "ok": False}
    picks = "".join(f"""<option value="{_esc(x["driver"])}" {"selected" if x["driver"] == d["driver"] else ""}>{_esc(x["driver"])}{" - blocked" if x["blocks"] else "" if x["ok"] else " - needs system packages"}</option>""" for x in p["drivers"])
    ns = d.get("needs_system")
    sysnote = _dim(_esc(f"""Needs {", ".join(ns["missing"])}: {ns["manager"] or "no package manager"} would install {" ".join(ns["packages"]) or "nothing known"}. """ + ("The node owner allows system packages." if ns["allowed"] else "Not allowed on this node: install them there by hand, or set setup.allow_system in its node.json."))) if ns else ""
    blocks = _dim(_esc("Blocked here: " + "; ".join(b["block"] for b in d["blocks"]))) if d["blocks"] else ""
    return f"""<form class="nd-form" id="nd-if-{_esc(name)}" onsubmit="return false"><div class="row"><b class="nd-key">driver</b><select name="driver" class="module-select" {_act("nodes_install_form", cid=cid, name=name, include=f"#nd-if-{name}", trigger="change")}>{picks}</select><span class="nd-dim grow">{_esc((d.get("credits") or {}).get("project", ""))} {_esc((d.get("credits") or {}).get("license", ""))}</span></div>
        {"".join(_opt_field(k, o, k in d["decide"]) for k, o in d["options"].items())}{sysnote}{blocks}{_dim("Installing contacts what the driver's steps name (package indexes, project and model downloads) from the node itself.")}<div class="row">{_btn("Install", "nodes_install", cid=cid, name=name, include=f"#nd-if-{name}") if not d["blocks"] else ""}{_btn("Cancel", "nodes_open", cid=cid)}</div></form>"""

def _install_status(cid: str, name: str) -> str:
    rec = INSTALLS[(cid, name)]
    poll = "" if rec["done"] else f"""hx-get="{_P}/install_status?cid={_esc(cid)}&amp;name={_esc(name)}" hx-trigger="every 2s" hx-swap="outerHTML" """
    done = f"""<div class="row">{_dim("Finished.")}{_btn("Refresh", "nodes_open", cid=cid)}</div>""" if rec["done"] else _dim("Installing...")
    return f"""<div {poll}>{done}<div class="quote-block mono nd-log">{_esc(chr(10).join(rec["lines"][-15:]))}</div></div>"""

def _shown(v) -> str: return "" if v is None else v if isinstance(v, str) else json.dumps(v)

def _group(label: str, opts: str) -> str: return f"""<optgroup label="{_esc(label)}">{opts}</optgroup>""" if opts else ""

def _model_select(field: str, cap: str, cur: str, models: dict, machines: list, empty: str = "(not set)") -> str:
    """A model choice: what this node offers for cap (its own models and its sources'), then the models of the other machines this portal connects to. Those carry "@<connection>|<model>": choosing one makes that machine a source of this node first (_resolve_pick)."""
    names = models.get(cap, [])
    here = "".join(_option(n, n == cur) for n in names) + (_option(cur, True, f"{cur} (not listed by the node right now)") if cur and cur not in names else "")
    other = "".join(_group(f"""{x.get("display_name", x["_id"])} - becomes a source of this node""", "".join(_option(f"""@{x["_id"]}|{n}""", False, n) for n in ms)) for x, ms in machines)
    return f"""<select name="{field}" class="module-select grow mono">{_option("", not cur, empty)}{_group("This node offers", here)}{other}</select>"""

def _value_field(v: dict, models: dict, machines: list) -> str:
    """The input for one module setting: a model choice when the module marks it "pick": "model:<capability>", otherwise typed."""
    cur, pick = _shown(v.get("value")), v.get("pick") or ""
    return _model_select("value", pick[6:], cur, models, machines) if pick.startswith("model:") else f"""<input name="value" class="module-select grow mono" value="{_esc(cur)}">"""

async def _resolve_pick(c: dict, raw: str, username: str) -> tuple:
    """(value, note, error) for a chosen model: "@<connection>|<model>" becomes "<source>/<model>" after that machine is made a source of this node."""
    if not raw.startswith("@"): return raw, "", ""
    conn_id, model = raw[1:].split("|", 1)
    src, err = await _as_source(c, conn_id, username)
    return (f"{src}/{model}", f" Its machine is source {src} of this node.", "") if not err else ("", "", f"Could not add that machine as a source: {err}")

def _slug(s: str) -> str: return re.sub(r"[^\w.-]", "", s.lower().replace(" ", "-")) or "source"

async def _machines(cid: str, sources: dict) -> list:
    """[(connection, [models])] for this portal's Ollama and Magpie connections that are not this node and not already one of its sources (same address)."""
    have = {s.get("url", "").rstrip("/") for s in sources.values()}
    conns = [x for x in AIM.connections.list_conns(get_all=True) if x.get("connection_type") in ("ollama", "magpie") and x["_id"] != cid and AIM.connections._base(x).rstrip("/") not in have]
    return [(x, ms) for x, ms in zip(conns, await asyncio.gather(*(AIM.connections.list_models_async(x) for x in conns))) if ms]

async def _as_source(c: dict, conn_id: str, username: str) -> tuple:
    """(source name, "") for a portal connection, adding it to the node's remote.sources when no source has its address yet; ("", error) otherwise."""
    conn, cur = AIM.connections.get_conn(conn_id), await _sources(c)
    if not conn: return "", "that connection no longer exists in this portal"
    url = AIM.connections._base(conn).rstrip("/")
    if hit := next((n for n, s in cur.items() if s.get("url", "").rstrip("/") == url), None): return hit, ""
    base = _slug(conn.get("display_name", conn_id))
    name = next(n for n in (base, *(f"{base}-{i}" for i in range(2, 100))) if n not in cur)
    key = conn.get("values", {}).get("api_key")
    code, d = await _call(c, "PATCH", "/settings", {"path": "remote.sources", "value": {**cur, name: {"url": url, **({"key": key} if key else {})}}, "source": f"portal: {username}"})
    return (name, "") if code == 200 else ("", d.get("error") or d.get("detail") or f"HTTP {code}")

def _sources_editor(cid: str, value: dict) -> str:
    """remote.sources as a list: add one from this portal's own connections (Ollama or Magpie), remove one. The node offers each source's models as source/model."""
    rows = "".join(f"""<div class="list-row row"><b class="mono">{_esc(n)}</b><span class="mono nd-dim grow">{_esc(s.get("url"))}{" | key set" if s.get("key") else ""}</span>{_btn("Remove", "nodes_src_del", cls="ui-btn nd-danger", cid=cid, src=n, confirm=f"Stop offering the models of {n} on this node?")}</div>""" for n, s in (value or {}).items())
    conns = "".join(_option(x["_id"], False, f"""{x.get("display_name", x["_id"])} ({x.get("connection_type")}, {AIM.connections._base(x)})""") for x in AIM.connections.list_conns(get_all=True) if x.get("connection_type") in ("ollama", "magpie") and x["_id"] != cid)
    return f"""<div class="nd-form"><b>Model sources</b>{_dim("Other servers whose models this node offers as source/model, so its own programs (a knowledge server) can use a stronger machine. The calls go through this node.")}{rows or _dim("none yet")}
        <form class="row" id="nd-src-add" onsubmit="return false"><input name="src" class="module-select mono" placeholder="name, e.g. worknode2"><select name="conn" class="module-select grow">{conns}</select>{_btn("Add", "nodes_src_add", cid=cid, include="#nd-src-add")}</form>{_dim("The node uses the address this portal uses for that connection; if it reaches the server differently, set remote.sources under Settings. A Magpie source's key is stored on this node: use a key of that node with infer scope, not its admin key.")}</div>"""

# --- Knowledge groups (the knowledge module's "instances") ---

def _groups_editor(cid: str, kn: dict) -> str:
    if kn.get("error"): return f"""<div class="nd-form"><b>Knowledge groups</b>{_dim(_esc(kn["error"]))}</div>"""
    def row(g): return f"""<div class="list-row row"><b class="mono">{_esc(g["name"])}</b><span class="chip">{_esc(g["status"])}</span><span class="nd-dim grow">chat {_esc(_said(g["chat"]))} | embedding {_esc(_said(g["embed"]))} | {g["ram_gb"]} GB{" | holds data" if g["stored"] else ""}{" | " + _esc(g.get("notes", "")[:90]) if g.get("notes") else ""}</span>{_btn("Edit", "nodes_group_form", cid=cid, group=g["name"])}{_btn("Remove...", "nodes_group_form", cls="ui-btn nd-danger", cid=cid, group=g["name"], remove="1")}</div>"""
    rows = "".join(row(g) for g in kn.get("groups", []))
    return f"""<div class="nd-form"><div class="row"><b>Knowledge groups</b><span class="grow"></span>{_btn("New group", "nodes_group_form", cid=cid, group="")}</div>{_dim("Each group is its own knowledge server with its own documents. Each names the machine and model it uses for chat and for embedding. Kimi, Athena, pipelines and agents list each one as group (node).")}{rows or _dim("none yet")}<div id="nd-kg-form"></div></div>"""

def _said(c: dict) -> str: return f"""{c.get("model") or "not set"} on {c.get("source") or "this node"}""" if c.get("model") else "not set"

def _machine_select(field: str, cur: str, sources: list, others: list) -> str:
    """Where a model runs: this node, one of its model sources, or another machine this portal connects to ("@<connection>": made a source of this node when the group is saved)."""
    opts = _option("", not cur, "this node") + "".join(_option(n, n == cur, f"{n} (source of this node)") for n in sources) + "".join(_option(f"""@{x["_id"]}""", f"""@{x["_id"]}""" == cur, f"""{x.get("display_name", x["_id"])} (becomes a source)""") for x in others)
    return f"""<select name="{field}" class="module-select">{opts}</select>"""

async def _machine_models(c: dict, machine: str, cap: str) -> list:
    """Model names on one machine, as that machine names them: this node's own, a source's (offered here as <source>/<model>), or another portal connection's own list."""
    if machine.startswith("@"):
        conn = AIM.connections.get_conn(machine[1:])
        return sorted(await AIM.connections.list_models_async(conn)) if conn else []
    code, d = await _call(c, "GET", f"/models?capability={cap}")
    ms = d.get("models", []) if code == 200 else []
    return sorted(m["name"][len(machine) + 1:] for m in ms if m.get("source") == machine) if machine else sorted(m["name"] for m in ms if m.get("adapter") != "remote")

async def _group_form(c: dict, cid: str, g: dict, sources: list, others: list) -> str:
    """g: the group as the node lists it, or the form's own values while a machine is being changed. Picking a machine reloads the form with that machine's models."""
    name = g.get("name", "")
    rows = []
    for kind, label, note in (("chat", "chat", "reads documents into the graph and writes answers"), ("embed", "embedding", "fixed once documents are in")):
        cur = g.get(kind) or {}
        names, model = await _machine_models(c, cur.get("source", ""), kind), cur.get("model", "")
        opts = _option("", not model, "(choose a model)") + "".join(_option(n, n == model) for n in names) + (_option(model, True, f"{model} (not listed there now)") if model and model not in names else "")
        rows.append(f"""<div class="row"><b class="nd-key">{label}</b><span {_act("nodes_group_form", cid=cid, include="#nd-kg", trigger="change", reload="1")}>{_machine_select(f"{kind}_source", cur.get("source", ""), sources, others)}</span><select name="{kind}_model" class="module-select grow mono">{opts}</select><span class="nd-dim">{note}</span></div>""")
    warn = f"""<label class="row nd-dim"><input type="checkbox" name="embed_change_ok" value="1"> {_esc(name)} holds data embedded with {_esc(_said(g.get("embed") or {}))}: changing its embedding model means searches find nothing until the documents are added again. Tick to change it anyway.</label>""" if g.get("stored") else ""
    return f"""<form class="nd-form" id="nd-kg" onsubmit="return false"><b>{"Edit " + _esc(name) if g.get("existing") else "New knowledge group"}</b><input type="hidden" name="existing" value="{"1" if g.get("existing") else ""}"><input type="hidden" name="stored" value="{"1" if g.get("stored") else ""}">
        <div class="row"><b class="nd-key">name</b><input name="group" class="module-select mono" value="{_esc(name)}" {"readonly" if g.get("existing") else 'placeholder="letters, digits, - and _"'}></div>
        {"".join(rows)}
        <div class="row"><b class="nd-key">memory (GB)</b><input name="ram_gb" class="module-select mono" value="{_esc(g.get("ram_gb", 2))}"><span class="nd-dim">counted against the node while its server runs</span></div>
        <div class="row"><b class="nd-key">what it holds</b><input name="notes" class="module-select grow" value="{_esc(g.get("notes", ""))}" placeholder="e.g. work procedures (QWI) - agents read this to choose a group"></div>{warn}
        <div class="row">{_btn("Save", "nodes_group_save", cid=cid, include="#nd-kg")}{_btn("Cancel", "nodes_mod_settings", cid=cid, name="knowledge")}</div></form>"""

def _group_remove_form(cid: str, g: dict) -> str:
    gname = g["name"]
    return f"""<form class="nd-form" id="nd-kg" onsubmit="return false"><b>Remove {_esc(g["name"])}</b>{_dim("Its server stops and the group is no longer offered. Without the box below its stored documents stay on the node, and a new group of the same name finds them again.")}
        <label class="row nd-dim"><input type="checkbox" name="data" value="1"> also delete its stored documents, graph and embeddings ({_esc(g.get("data", ""))})</label>
        <div class="row">{_btn("Remove", "nodes_group_del", cls="ui-btn nd-danger", cid=cid, group=g["name"], include="#nd-kg", confirm=f"Remove the knowledge group {gname}?")}{_btn("Cancel", "nodes_mod_settings", cid=cid, name="knowledge")}</div></form>"""

def _mod_settings(cid: str, m: dict, models: dict, machines: list, kn: dict = None) -> str:
    name = m["name"]
    rows = "".join(_sources_editor(cid, v.get("value")) if v.get("pick") == "sources" else _groups_editor(cid, kn or {}) if v.get("pick") == "knowledge_groups" else f"""<form class="row" id="nd-ms-{_esc(name)}-{i}" onsubmit="return false"><b class="nd-key{"" if v.get("label") else " mono"}" title="{_esc(k)}">{_esc(v.get("label") or k)}</b>{_value_field(v, models, machines)}{_btn("Save", "nodes_set", cid=cid, path=f"{name}.{k}", include=f"#nd-ms-{name}-{i}")}<span class="nd-dim">{_esc(v.get("hint", ""))}{" (at its next start)" if v.get("when") == "runtime" else ""}</span></form>""" for i, (k, v) in enumerate(m["settings"].items()))
    return f"""<div class="nd-form">{rows}{_dim("Text is kept as text; numbers, true/false, lists and objects as JSON. Empty removes the value set here, so the module's default applies.")}<div class="row">{_btn("Close", "nodes_open", cid=cid)}</div></div>"""

def _services(c: dict, n: dict) -> str:
    cid = c["_id"]
    def info(s): return " | ".join(x for x in (s.get("owner") or "", f"""{s["rss_gb"]} GB""" if s.get("rss_gb") else "", f"""up {_dur(s["uptime_s"])}""" if s.get("uptime_s") else "", f"""{s["active"]} working""" if s.get("active") else "") if x)
    rows = "".join(f"""<div class="list-row row"><b class="mono">{_esc(s["name"])}</b><span class="chip">{_esc(s["status"])}</span><span class="nd-dim grow">{_esc(info(s))}</span>{"".join(_btn(v.title(), "nodes_svc", cid=cid, name=s["name"], verb=v) for v in ("start", "stop", "restart"))}{_btn("Log", "nodes_svc_log", cid=cid, name=s["name"])}</div>""" for s in n["services"])
    return f"""<div class="nd-sec"><b>Services</b>{_dim("Programs this node runs for its modules. Stop lets running work finish first. Most start by themselves when work needs them.")}{rows or _dim("none yet")}<div id="nd-svc-log"></div></div>"""

def _locations(c: dict, n: dict) -> str:
    rows = "".join(f"""<form class="list-row row" id="nd-loc-{_esc(k)}" onsubmit="return false"><b class="nd-key">{_esc(k)}</b><input name="path" class="module-select grow mono" value="{_esc(v["path"])}"><span class="chip">{_esc(v.get("kind", "long"))}</span><label class="nd-dim"><input type="checkbox" name="move" value="1"> move what is there</label>{_btn("Save", "nodes_loc", cid=c["_id"], name=k, include=f"#nd-loc-{k}")}</form>""" for k, v in n["locations"].items())
    return f"""<div class="nd-sec"><b>Locations</b>{_dim("Where this node keeps each kind of file. A new path is used from each program's next start; without move, files stay where they are (point it at a folder that already has them).")}{rows}</div>"""

async def _settings(c: dict) -> str:
    code, p = await _call(c, "GET", "/policy")
    if code != 200: return f"""<div class="nd-sec"><b>Settings</b>{_err(code, p)}</div>"""
    hist = "".join(f"<option>{v}</option>" for v in reversed(p["history"]))
    return f"""<div class="nd-sec"><b>Settings</b> <span class="nd-dim">version {p["version"]} - what portals have set on this node (its owner's node.json is shown under each module's defaults)</span><div class="quote-block mono">{_esc(json.dumps(p["rules"], indent=1))}</div>
        <form class="row" id="nd-set" onsubmit="return false"><input name="path" class="module-select mono" placeholder="dotted path, e.g. knowledge.instances"><input name="value" class="module-select grow mono" placeholder="value (JSON or text; empty removes)">{_btn("Set", "nodes_set", cid=c["_id"], include="#nd-set")}</form>
        <form class="row" id="nd-rb" onsubmit="return false"><span class="nd-dim">Earlier version</span><select name="version" class="module-select">{hist}</select>{_btn("Roll back to it", "nodes_rollback", cid=c["_id"], include="#nd-rb", confirm="Replace this node's current settings with that earlier version? The current one is kept as a version too.")}</form>{_dim("Never set from here: " + _esc(", ".join(p["never"]))) if p.get("never") else ""}</div>"""

async def _log(c: dict) -> str:
    code, d = await _call(c, "GET", "/node/log?n=200")
    body = _err(code, d) if code != 200 else f"""<div class="quote-block mono nd-log">{_esc(chr(10).join(d["tail"]) or "(empty)")}</div>{_dim(_esc(d["log"]))}"""
    return f"""<div class="nd-sec"><div class="row"><b>Node log</b><span class="grow"></span>{_btn("Refresh", "nodes_open", cid=c["_id"])}</div>{body}</div>"""

async def page(cid: str = "", msg: str = "") -> str:
    """The inside of #nd-page (refreshes swap only its content, so the page keeps one container)."""
    c = _conn(cid)
    pick = "".join(f"""<button type="button" class="chip{" nd-on" if c and x["_id"] == c["_id"] else ""}" {_act("nodes_open", cid=x["_id"])}>{_esc(x.get("display_name", x["_id"]))}</button>""" for x in _nodes())
    head = f"""<div class="row"><b>Nodes</b>{pick}</div>{_dim("A node joins as a Magpie connection (Settings - Connections) with its admin key.")}<div id="nd-msg">{msg}</div>"""
    if not c: return f"""{head}{_dim("No Magpie connections yet.")}"""
    code, n = await _call(c, "GET", "/node")
    body = _err(code, n) if code != 200 else f"""{_summary(c, n)}{_modules(c, n)}{_services(c, n)}{_locations(c, n)}{await _settings(c)}{await _log(c)}"""
    return f"""{head}{body}"""

# --- Intents ---

def _note(text: str, bad: bool = False) -> str: return f"""<div class="{"nd-err" if bad else "nd-note"}">{_esc(text)}</div>"""
async def _refresh(imr, cid: str, msg: str = ""): return imr.oob(await page(cid, msg), "nd-page")
def _admin(request) -> bool: return getattr(request.state.user, "role", "") == "admin"

async def _module(c: dict, name: str):
    code, m = await _call(c, "GET", f"/modules/{name}")
    return m if code == 200 else None

async def _h_open(request, payload, imr): return await _refresh(imr, payload.get("cid", ""))

async def _h_restart(request, payload, imr):
    if not _admin(request): return imr.oob(_note("Only a portal admin manages nodes.", True), "nd-msg")
    code, d = await _call(_conn(payload["cid"]), "POST", "/node/restart")
    return imr.oob(_note(f"""Restarting (kept running by {d["by"]}); refresh in a few seconds.""") if code == 200 else _err(code, d), "nd-msg")

async def _h_mod(request, payload, imr):
    if not _admin(request): return imr.oob(_note("Only a portal admin manages nodes.", True), "nd-msg")
    c, name, verb = _conn(payload["cid"]), payload["name"], payload["verb"]
    code, d = await (_call(c, "DELETE", f"/modules/{name}?confirm=1") if verb == "remove" else _call(c, "POST", f"/modules/{name}/{verb}"))
    return await _refresh(imr, payload["cid"], _note(f"{name}: {verb} done.") if code == 200 else _err(code, d))

async def _h_install_form(request, payload, imr):
    m = await _module(_conn(payload["cid"]), payload["name"])
    return imr.oob(_install_form(payload["cid"], m, payload.get("driver", "")) if m else _note("The node did not answer for this module.", True), f"""nd-m-{payload["name"]}""")

async def _h_install(request, payload, imr):
    if not _admin(request): return imr.oob(_note("Only a portal admin manages nodes.", True), "nd-msg")
    cid, name = payload["cid"], payload["name"]
    if (cid, name) in INSTALLS and not INSTALLS[(cid, name)]["done"]: return imr.oob(_install_status(cid, name), f"nd-m-{name}")
    INSTALLS[(cid, name)] = {"lines": [], "done": False}
    t = asyncio.create_task(_install(cid, name, {"driver": payload.get("driver", ""), "options": {k[4:]: v for k, v in payload.items() if k.startswith("opt_")}}))
    _TASKS.add(t)
    t.add_done_callback(_TASKS.discard)
    return imr.oob(_install_status(cid, name), f"nd-m-{name}")

async def _offered(c: dict, caps: set) -> dict:
    """{capability: [model names]} this node offers, for the settings that pick a model."""
    out = {}
    for cap in caps:
        code, d = await _call(c, "GET", f"/models?capability={cap}")
        out[cap] = sorted(m["name"] for m in d.get("models", [])) if code == 200 else []
    return out

async def _choices(c: dict, cid: str, caps: set) -> tuple: return (await _offered(c, caps), await _machines(cid, await _sources(c))) if caps else ({}, [])

async def _knowledge(c: dict) -> dict:
    code, d = await _call(c, "GET", "/knowledge")
    return d if code == 200 else {"error": "The knowledge module is not active on this node: install or activate it above." if code == 404 else d.get("error") or f"HTTP {code}"}

async def _settings_html(cid: str, name: str) -> str:
    c = _conn(cid)
    m = await _module(c, name)
    if not m: return _note("The node did not answer for this module.", True)
    picks = [str(v.get("pick", "")) for v in m["settings"].values()]
    models, machines = await _choices(c, cid, {p[6:] for p in picks if p.startswith("model:")})
    return _mod_settings(cid, m, models, machines, await _knowledge(c) if "knowledge_groups" in picks else None)

async def _h_mod_settings(request, payload, imr): return imr.oob(await _settings_html(payload["cid"], payload["name"]), f"""nd-m-{payload["name"]}""")

def _form_group(payload: dict) -> dict:
    """The group as the form holds it right now (while a machine is being changed)."""
    return {"name": payload.get("group", ""), "existing": payload.get("existing") == "1", "stored": payload.get("stored") == "1", "ram_gb": payload.get("ram_gb", 2), "notes": payload.get("notes", ""), **{k: {"source": payload.get(f"{k}_source", ""), "model": payload.get(f"{k}_model", "")} for k in ("chat", "embed")}}

async def _h_group_form(request, payload, imr):
    c, cid, name = _conn(payload["cid"]), payload["cid"], payload.get("group", "")
    kn = await _knowledge(c)
    if payload.get("reload") == "1": g = _form_group(payload)
    else: g = {**next((x for x in kn.get("groups", []) if x["name"] == name), {}), "existing": True} if name else {"chat": (kn.get("groups") or [{}])[-1].get("chat", {}), "embed": (kn.get("groups") or [{}])[-1].get("embed", {})}
    if payload.get("remove") == "1": return imr.oob(_group_remove_form(cid, g), "nd-kg-form")
    srcs = await _sources(c)
    others = [x for x in AIM.connections.list_conns(get_all=True) if x.get("connection_type") in ("ollama", "magpie") and x["_id"] != cid and AIM.connections._base(x).rstrip("/") not in {v.get("url", "").rstrip("/") for v in srcs.values()}]
    return imr.oob(await _group_form(c, cid, g, sorted(srcs), others), "nd-kg-form")

async def _h_group_save(request, payload, imr):
    if not _admin(request): return imr.oob(_note("Only a portal admin manages nodes.", True), "nd-msg")
    c, cid, name, user, g = _conn(payload["cid"]), payload["cid"], (payload.get("group") or "").strip(), request.state.user.username, _form_group(payload)
    added = []
    for k in ("chat", "embed"):
        if not g[k]["model"]: return imr.oob(_note(f"""Choose a {"chat" if k == "chat" else "embedding"} model.""", True), "nd-msg")
        if g[k]["source"].startswith("@"):   # another machine this portal knows: it becomes a source of this node first
            src, err = await _as_source(c, g[k]["source"][1:], user)
            if err: return imr.oob(_note(f"Could not add that machine as a source: {err}", True), "nd-msg")
            g[k]["source"] = src
            added.append(src)
    code, d = await _call(c, "PUT", f"/knowledge/{name}", {"chat": g["chat"], "embed": g["embed"], "ram_gb": payload.get("ram_gb") or 2, "notes": payload.get("notes", ""), "embed_change_ok": payload.get("embed_change_ok") == "1", "source": f"portal: {user}"})
    if code != 200: return imr.oob(_err(code, d), "nd-msg")
    AIM.connections.forget_groups(cid)
    extra = f""" Added as model sources of this node: {", ".join(sorted(set(added)))}.""" if added else ""
    return imr.oob(await _settings_html(cid, "knowledge"), "nd-m-knowledge").oob(_note(f"""Knowledge group {name} saved: chat {_said(d["chat"])}, embedding {_said(d["embed"])}.{extra} Kimi, Athena and agents list it as {name} (this node)."""), "nd-msg")

async def _h_group_del(request, payload, imr):
    if not _admin(request): return imr.oob(_note("Only a portal admin manages nodes.", True), "nd-msg")
    code, d = await _call(_conn(payload["cid"]), "DELETE", f"""/knowledge/{payload["group"]}?confirm=1{"&data=1" if payload.get("data") == "1" else ""}""")
    if code != 200: return imr.oob(_err(code, d), "nd-msg")
    AIM.connections.forget_groups(payload["cid"])
    return imr.oob(await _settings_html(payload["cid"], "knowledge"), "nd-m-knowledge").oob(_note(f"""Removed {d["removed"]}{" and its stored data" if d["data_deleted"] else "; its stored data stays on the node"}."""), "nd-msg")

async def _sources(c: dict) -> dict: return (((await _module(c, "remote")) or {}).get("settings", {}).get("sources") or {}).get("value") or {}

async def _set_sources(request, imr, cid: str, value: dict, msg: str):
    c = _conn(cid)
    code, d = await _call(c, "PATCH", "/settings", {"path": "remote.sources", "value": value, "source": f"portal: {request.state.user.username}"})
    if code != 200: return imr.oob(_err(code, d), "nd-msg")
    return imr.oob(_sources_editor(cid, value), "nd-m-remote").oob(_note(msg), "nd-msg")

async def _h_src_add(request, payload, imr):
    if not _admin(request): return imr.oob(_note("Only a portal admin manages nodes.", True), "nd-msg")
    name, conn = re.sub(r"[^\w.-]", "", payload.get("src") or ""), AIM.connections.get_conn(payload.get("conn") or "")
    if not name or not conn: return imr.oob(_note("Give the source a name (letters, digits, - . _) and pick a connection.", True), "nd-msg")
    cur = await _sources(_conn(payload["cid"]))
    if name in cur: return imr.oob(_note(f"There is already a source named {name}: remove it first, or pick another name.", True), "nd-msg")
    key = conn.get("values", {}).get("api_key")
    return await _set_sources(request, imr, payload["cid"], {**cur, name: {"url": AIM.connections._base(conn), **({"key": key} if key else {})}}, f"Added {name}: its models appear on this node as {name}/... within a minute.")

async def _h_src_del(request, payload, imr):
    if not _admin(request): return imr.oob(_note("Only a portal admin manages nodes.", True), "nd-msg")
    cur = await _sources(_conn(payload["cid"]))
    return await _set_sources(request, imr, payload["cid"], {n: v for n, v in cur.items() if n != payload["src"]}, f"""Removed {payload["src"]}.""")

async def _h_set(request, payload, imr):
    if not _admin(request): return imr.oob(_note("Only a portal admin manages nodes.", True), "nd-msg")
    if not (payload.get("path") or "").strip(): return imr.oob(_note("Give the setting's dotted path.", True), "nd-msg")
    c = _conn(payload["cid"])
    raw, note, err = await _resolve_pick(c, (payload.get("value") or "").strip(), request.state.user.username)
    if err: return imr.oob(_note(err, True), "nd-msg")
    code, d = await _call(c, "PATCH", "/settings", {"path": payload["path"].strip(), "value": _value(raw) if raw else None, "source": f"portal: {request.state.user.username}"})
    return await _refresh(imr, payload["cid"], _note(f"""{d["path"]} = {_shown(d.get("value"))} (version {d["version"]}).{note} Programs using it pick it up at their next start; module lists and knowledge instances at the node's restart.""") if code == 200 else _err(code, d))

async def _h_rollback(request, payload, imr):
    if not _admin(request): return imr.oob(_note("Only a portal admin manages nodes.", True), "nd-msg")
    code, d = await _call(_conn(payload["cid"]), "POST", "/policy/rollback", {"version": int(payload["version"])})
    return await _refresh(imr, payload["cid"], _note(f"""Rolled back to version {payload["version"]}.""") if code == 200 else _err(code, d))

async def _h_loc(request, payload, imr):
    if not _admin(request): return imr.oob(_note("Only a portal admin manages nodes.", True), "nd-msg")
    code, d = await _call(_conn(payload["cid"]), "PUT", f"""/locations/{payload["name"]}""", {"path": payload["path"].strip(), "move": payload.get("move") == "1"})
    return await _refresh(imr, payload["cid"], _note(f"""{payload["name"]} is now {payload["path"].strip()}.""") if code == 200 else _err(code, d))

async def _h_svc(request, payload, imr):
    if not _admin(request): return imr.oob(_note("Only a portal admin manages nodes.", True), "nd-msg")
    code, d = await _call(_conn(payload["cid"]), "POST", f"""/services/{payload["name"]}/{payload["verb"]}""")
    return await _refresh(imr, payload["cid"], _note(f"""{payload["name"]}: {d.get("status", payload["verb"])}""") if code == 200 else _err(code, d))

async def _h_svc_log(request, payload, imr):
    code, d = await _call(_conn(payload["cid"]), "GET", f"""/services/{payload["name"]}/log?n=150""")
    body = f"""<div class="row"><b class="mono">{_esc(payload["name"])}</b><span class="nd-dim grow mono">{_esc(d["log"])}</span>{_btn("Refresh", "nodes_svc_log", cid=payload["cid"], name=payload["name"])}</div><div class="quote-block mono nd-log">{_esc(chr(10).join(d["tail"]) or "(empty)")}</div>""" if code == 200 else _err(code, d)
    return imr.oob(body, "nd-svc-log")

# --- Routes and init ---

@router.get("")
@router.get("/")
async def root(request: Request):
    return ENV["templates"].TemplateResponse(name="base.html", request=request, context={"request": request, "user": request.state.user, "nesting_level": 2, "shell_id": IM.branch_id, "content": f"""<div class="scroll-pane nd" id="nd-page">{await page()}</div>""", "extra_css": CSS, "extra_script": ""})

@router.get("/install_status", response_class=HTMLResponse)
async def install_status(cid: str, name: str): return HTMLResponse(_install_status(cid, name) if (cid, name) in INSTALLS else "")

def init_tool(env, prefix):
    global ENV, AIM, IM, _P
    ENV, _P, AIM = env, prefix.rstrip("/"), env["tools"]["ai_manager"]
    IM = env["InterfaceManager"](nesting_level=2, db_path=f"""{env["data_dir"]}/nodes/im_registry.db""")
    IM.scripts.update({f"nodes_{k}": [fn] for k, fn in {"open": _h_open, "restart": _h_restart, "mod": _h_mod, "install_form": _h_install_form, "install": _h_install, "mod_settings": _h_mod_settings, "set": _h_set, "rollback": _h_rollback, "loc": _h_loc, "svc": _h_svc, "svc_log": _h_svc_log, "src_add": _h_src_add, "src_del": _h_src_del, "group_form": _h_group_form, "group_save": _h_group_save, "group_del": _h_group_del}.items()})
    print("[nodes] ready")

CSS = """
.nd{font-size:.8rem}.nd-sec{border:var(--border-thick) solid var(--border);border-radius:var(--radius);padding:.4rem .55rem;margin:.4rem 0;background:var(--surface)}
.nd-dim{font-size:.72rem;color:var(--text_muted)}.nd-key{min-width:7rem}.nd-on{border-color:var(--accent);background:var(--accent_dim)}
.nd-ok{color:#00ffa2}.nd-mid{color:#ffcc00}.nd-bad{color:#ff6b6b}.nd-danger{color:#ff6b6b}
.nd-form{border-left:3px solid var(--accent);padding:.3rem .5rem;margin:.3rem 0}.nd-log{max-height:22rem;font-size:.7rem}
.nd-note{border-left:3px solid var(--accent);padding:.3rem .5rem;margin:.3rem 0}.nd-err{border-left:3px solid #ff6b6b;padding:.3rem .5rem;margin:.3rem 0}
"""