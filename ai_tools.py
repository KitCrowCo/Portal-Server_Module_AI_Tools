"""
AI Tools Module
Multi-tool AI dashboard shell. Discovers sub-tools from subdirectories.
"""
import re, sys, json, uuid, importlib, importlib.util, pkg_resources, subprocess, httpx
from pathlib import Path
from datetime import datetime
from fastapi import APIRouter, Request, HTTPException
from fastapi.responses import HTMLResponse

MODULE_META = {"label": "AI Tools", "icon": "&#x25B3;", "description": "AI Multi-Tool Dashboard", "persistence": "user"}

router = APIRouter()
_P = "/module/ai_tools"
_MODULE_DIR = Path(__file__).parent
DATA_DIR = Path("./data/ai_tools")

ENV = {}
IM = None
TM = None
UI = None
AIM = None
Tools:dict = {}
_sub_css = ""

TOOL_GROUPS = {"model_tools": {"label": "Model Tools", "icon": "&#x25B3;"},
               "chat":        {"label": "Chat",        "icon": "&#x1F4AC;"},
               "knowledge":   {"label": "Knowledge",   "icon": "&#x1F4DA;"},
               "other":       {"label": "Other",       "icon": "&#x25A1;"},
               "system":      {"label": "System",      "icon": "&#x2699;"}}

# --- Access Policy ---

POLICY_FILE = DATA_DIR / "access_policy.json"

def _get_policy() -> dict: return json.loads(POLICY_FILE.read_text()) if POLICY_FILE.exists() else {"auto_redirect": None, "user_groups": {}, "role_access": {}}
def _save_policy(data: dict): POLICY_FILE.write_text(json.dumps(data, indent=2))

def _is_allowed(request: Request, tool_key: str) -> bool:
    # Admin always has access to everything
    if getattr(request.state.user, "role", "user") == "admin": return True
    policy = _get_policy()
    username = getattr(request.state.user, "username", "")
    role = getattr(request.state.user, "role", "user")
    if policy.get("auto_redirect"): return tool_key == policy["auto_redirect"]  # Check if this specific tool is the ONLY allowed auto-redirect for non-admins
    if tool_key in policy.get("user_groups", {}).get(username, []): return True  # Check Whitelist (Username or Role)
    if tool_key in policy.get("role_access", {}).get(role, []): return True
    return False #tool_key == "launcher" # Default to launcher access only if no specific lockdown

# --- Sub-module loader ---

def _valid(n: str) -> bool: return bool(n and not n.startswith((".", "_")) and re.match(r"^[a-zA-Z][a-zA-Z0-9_]*$", n))

def _check_deps(path: Path):
    req = path / "requirements.txt"
    if not req.exists(): return
    installed = {p.key for p in pkg_resources.working_set}
    missing = [d.strip() for d in req.read_text().splitlines() if d.strip() and not d.startswith("#") and d.split("==")[0].lower() not in installed]
    if missing:
        print(f"[ai_tools] installing deps: {missing}")
        subprocess.check_call([sys.executable, "-m", "pip", "install", *missing])

def _load_submodules():
    global _sub_css
    css = ""
    for item in sorted(_MODULE_DIR.iterdir()):
        if not item.is_dir() or not _valid(item.name): continue
        entry = item / f"{item.name}.py"
        if not entry.exists(): continue
        try:
            _check_deps(item)
            spec = importlib.util.spec_from_file_location(f"modules.ai_tools.{item.name}", entry)
            mod = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(mod)
            router.include_router(mod.router, prefix=f"/{item.name}", tags=[f"ai_tools:{item.name}"])
            if hasattr(mod, "init_tool"): mod.init_tool(ENV, f"{_P}/{item.name}")
            if hasattr(mod, "CSS"): css += mod.CSS
            Tools[item.name] = mod
            print(f"[ai_tools] loaded: {item.name}")
        except Exception as e:
            print(f"[ai_tools] error loading '{item.name}': {e}")
    _sub_css = css

# --- init_module ---

def init_module(env: dict):
    global ENV, IM, TM, UI, AIM
    ENV.update(env)
    UI = ENV["templates"].env.globals.get("UI")
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    IM = ENV["InterfaceManager"](nesting_level=1, db_path="ai_tools/im_registry.db")
    TM = ENV["tools"]["built_ins"].TabManager(namespace="ai_tools", tab_bar_id="ait-tab-bar", content_id="ait-workspace", render_content_fn=_tab_content, intent_prefix="ai_tools", IM=IM, scope="user", empty={"tabs": {"launcher": {"id": "launcher", "path": "launcher", "label": "Launcher", "icon": "", "order": 0}}, "active": "launcher"}, nesting_level=1)
    AIM = ENV["tools"]["ai_manager"]
    _load_submodules()
    if "refresh_system" in ENV: ENV["refresh_system"]() # This is to reupdate routes

    # Wrap tab intents to also update right panel via OOB ****************** This should be a more general function ****************
    def _wrap(base_fn):
        async def _h(req, pay, imr):
            imr = await base_fn(req, pay, imr)
            state = await TM._load(req)
            imr.oob(_right_panel(_tool_key(state.get("tabs", {}).get(state.get("active", ""), {}).get("path", "launcher"))), "ait-right-panel")
            return imr
        return _h

    for intent in ("open", "focus", "close"): IM.scripts[f"ai_tools_{intent}_tab"] = [_wrap(getattr(TM, f"_{intent}"))]
    print(f"[ai_tools] ready | tools: {list(Tools.keys())}")

# --- Helpers ---

def _url(*parts): return "/".join(p.strip("/") for p in [_P, *parts] if p)
def _tool_key(path: str) -> str: return path.replace(_P, "").strip("/").split("/")[0] or "launcher"

def _registry() -> dict:
    reg = {k: {"label": m.TOOL_META.get("label", k), "group": m.TOOL_META.get("group", "other"), "icon": m.TOOL_META.get("icon", "&#x25A1;"), "description": m.TOOL_META.get("description", ""), "singleton": m.TOOL_META.get("singleton", False)} for k, m in Tools.items()}
    reg["settings"] = {"label": "Settings", "group": "system", "icon": "&#x2699;", "description": "Connections and preferences", "singleton": True}
    return reg

async def _tab_content(request, state):
    active = state.get("active", "launcher")
    path = state.get("tabs", {}).get(active, {}).get("path", "launcher")
    clean = path.replace(_P, "").strip("/") or "launcher"
    return state, f'<div hx-get="{_P}/{clean}{"" if clean == "launcher" else "/"}" hx-trigger="load" hx-target="#ait-workspace" hx-swap="innerHTML" style="width:100%;height:100%;">Loading...</div>'

# --- Connection settings ---

def _tmpls() -> dict: return AIM.connections.list_templates()
def _save_conn(cid, data): AIM.connections.save_conn(cid, data)
def _load_conn(cid): return AIM.connections.load_conn_raw(cid)
def _del_conn(cid): AIM.connections.delete_conn(cid)

# --- HTML builders ---

def _right_panel(tool_key: str) -> str:
    if tool_key == "settings":
        return f"""
        <div class="ait-rp">
            <div class="ait-rp-hd">System Settings</div>
            <button class="ait-rp-btn" hx-get="{_P}/settings/connections" hx-target="#ait-workspace" hx-swap="innerHTML">&#x1F517; Connections</button>
            <button class="ait-rp-btn" hx-get="{_P}/settings/hw" hx-target="#ait-workspace" hx-swap="innerHTML">&#x1F4BB; Hardware Profiles</button>
            <button class="ait-rp-btn" hx-get="{_P}/settings/policy" hx-target="#ait-workspace" hx-swap="innerHTML">Access Policy</button>
        </div>"""
    # Check if it's a sub-module
    mod = Tools.get(tool_key)
    if mod and hasattr(mod, "right_panel"):
        try: return mod.right_panel()
        except Exception as e: return f'<div class="ait-rp" style="color:#ff5f5f;font-size:.75rem">Panel error: {e}</div>'
    return '<div class="ait-rp"><p style="color:var(--text_muted);font-size:.78rem;padding:.5rem">No options</p></div>'

def _conns_html() -> str:
    rows = "".join(f"""<div class="glass conn-card">
                            <div class="conn-hd">
                                <span>{_tmpls().get(c.get("connection_type", ""), {}).get("_meta", {}).get("icon", "&#x25A1;")}</span>
                                <span class="conn-name">{c.get("display_name", c["_id"])}</span>
                                <span class="conn-type">{c.get("connection_type","?")}</span>
                                <span id="conn-dot-{c['_id']}" class="conn-dot" hx-get="{_P}/conn_status/{c['_id']}" hx-trigger="load" hx-swap="outerHTML">&#x25CF;</span>
                                <div style="margin-left:auto;display:flex;gap:.3rem;">
                                    <button class="btn-icon" hx-get="{_P}/settings/edit_conn/{c['_id']}" hx-target="#conn-edit" hx-swap="innerHTML">&#x270E;</button>
                                    <button class="btn-icon" hx-post="{_P}/settings/test_conn/{c['_id']}" hx-target="#conn-edit" hx-swap="innerHTML">&#x25B6;</button>
                                    <button class="btn-icon" style="color:#ff5f5f" hx-delete="{_P}/settings/conn/{c['_id']}" hx-target="#ait-conns" hx-swap="outerHTML" hx-confirm="Delete?">&#x2715;</button>
                                </div>
                            </div>
                            <div style="font-size:.68rem;color:var(--text_muted);font-family:var(--font-mono);margin-top:.2rem">{c.get("values",{}).get("host","?")}{":" + str(c["values"]["port"]) if c.get("values",{}).get("port") else ""}</div>
                        </div>""" for c in AIM.connections.list_conns(get_all = True)) or '<div style="color:var(--text_muted);font-size:.8rem;padding:1rem 0">No connections. Add one below.</div>'
    type_opts = "".join(f'<option value="{k}">{v.get("_meta",{}).get("display_name",k)}</option>' for k, v in _tmpls().items())
    return f"""<div id="ait-conns" style="padding:1.5rem; height:100%; overflow:auto; box-sizing:border-box;">
                    <h2 style="margin:0 0 1rem;font-size:1rem">Connections</h2>
                    <div style="display:flex;flex-direction:column;gap:.4rem;margin-bottom:1.5rem">{rows}</div>
                    <details class="glass" style="padding:.8rem">
                        <summary style="cursor:pointer;font-size:.82rem;color:var(--text_muted)">+ Add Connection</summary>
                        <div style="display:flex; gap:.5rem; align-items:center; margin-top:.6rem">
                            <select id="nc-type" name="conn_type" class="module-select" style="flex:1">{type_opts}</select>
                            <button class="ui-btn" hx-get="{_P}/settings/new_conn_form" hx-include="#nc-type" hx-target="#nc-form" hx-swap="innerHTML">Configure</button>
                        </div>
                        <div id="nc-form"></div>
                    </details>
                    <div id="conn-edit" style="margin-top:1rem"></div>
                </div>"""

def _conn_form(ctype: str, existing: dict = None, cid: str = None) -> str:
    tmpl = _tmpls().get(ctype)
    if not tmpl: return f'<div style="color:#ff5f5f">Unknown type: {ctype}</div>'
    meta, schema = tmpl.get("_meta", {}), tmpl.get("connection", {})
    vals = (existing or {}).get("values", {})
    fields = ""
    for fn, fd in schema.items():
        ft, lbl, cur = fd.get("type", "string"), fd.get("label", fn), vals.get(fn, fd.get("default", ""))
        hs = f' <span style="opacity:.55;font-size:.67rem">- {fd["hint"]}</span>' if fd.get("hint") else ""
        if ft == "boolean":
            fields += f'<label style="flex-direction:row; align-items:center; gap:.4rem; font-size:.8rem"><input type="checkbox" name="field_{fn}" value="1" {"checked" if cur else ""}> {lbl}{hs}</label>'
        elif ft == "secret":
            fields += f'<label style="font-size:.75rem;color:var(--text_muted)">{lbl}{hs}<input type="password" name="field_{fn}" value="{cur}" class="module-select" style="font-family:var(--font-mono)"></label>'
        elif ft == "integer":
            fields += f'<label style="font-size:.75rem;color:var(--text_muted)">{lbl}{hs}<input type="number" name="field_{fn}" value="{"" if cur is None else cur}" step="1" class="module-select"></label>'
        else:
            fields += f'<label style="font-size:.75rem;color:var(--text_muted)">{lbl}{hs}<input type="text" name="field_{fn}" value="{cur}" class="module-select" {"required" if fd.get("required") else ""}></label>'
    action = f"{_P}/settings/save_conn/{cid}" if cid else f"{_P}/settings/create_conn"
    title  = f"Edit: {existing.get('display_name','')}" if existing else f"New {meta.get('display_name', ctype)}"
    return f"""<div class="glass" style="padding:1rem;margin-top:.5rem">
                    <div style="font-size:.85rem;font-weight:600;color:var(--accent);margin-bottom:.7rem">{title}</div>
                    <form hx-post="{action}" hx-target="#ait-conns" hx-swap="outerHTML" style="display:flex;flex-direction:column;gap:.45rem">
                        {"<input type=hidden name=cid value=" + cid + ">" if cid else ""}
                        <input type="hidden" name="connection_type" value="{ctype}">
                            <label style="font-size:.75rem;color:var(--text_muted)">Display Name
                                <input type="text" name="display_name" value='{existing.get("display_name","") if existing else ""}' class="module-select" required>
                            </label>
                            {fields}
                        <div style="display:flex;gap:.5rem;margin-top:.4rem">
                            <button type="submit" class="ui-btn">Save</button>
                            <button type="button" class="ui-btn" hx-get="{_P}/settings/connections" hx-target="#ait-workspace" hx-swap="innerHTML">Cancel</button>
                        </div>
                    </form>
                </div>"""

# --- CSS ---

CSS = """
.ait-nav{display:flex;flex-direction:column;height:100%;overflow-y:auto;}
.ait-grp{border-bottom:var(--border-thick) solid var(--border);}
.ait-grp summary{cursor:pointer;padding:.35rem .6rem;font-size:.7rem;text-transform:uppercase;letter-spacing:.06em;color:var(--text_muted);list-style:none;user-select:none;}
.ait-grp summary::-webkit-details-marker{display:none;}
.ait-grp-items{padding:.1rem 0;}
.ait-item{display:flex;align-items:center;gap:.45rem;padding:.32rem .85rem;cursor:pointer;font-size:.8rem;color:var(--text_muted);transition:background .12s,color .12s;}
.ait-item:hover{background:var(--accent_dim);color:var(--accent);}
.ait-rp{display:flex;flex-direction:column;padding:.5rem;gap:.2rem;height:100%;overflow-y:auto;}
.ait-rp-hd{font-size:.67rem;text-transform:uppercase;letter-spacing:.07em;color:var(--text_muted);padding:.4rem .2rem .25rem;}
.ait-rp-btn{width:100%;text-align:left;padding:.32rem .55rem;font-size:.78rem;cursor:pointer;border-radius:var(--radius);border:none;background:transparent;color:var(--text_muted);transition:all .12s;}
.ait-rp-btn:hover{background:var(--accent_dim);color:var(--accent);}
.ait-launch-grid{display:grid;grid-template-columns:repeat(auto-fill,minmax(19rem,1fr));gap:.65rem;}
.ait-lc{display:flex;align-items:center;gap:.65rem;padding:.7rem .9rem;cursor:pointer;border-radius:var(--radius);border:var(--border-thick) solid var(--border);background:var(--glass);transition:all .12s;}
.ait-lc:hover{border-color:var(--accent);background:var(--accent_dim);}
.ait-lc-icon{font-size:1.25rem;flex-shrink:0;}
.ait-lc-label{font-size:.84rem;font-weight:600;}
.ait-lc-desc{font-size:.68rem;color:var(--text_muted);margin-top:.1rem;}
#ait-workspace{width:100%;height:100%;overflow:auto;display:flex;flex-direction:column;}
.conn-card{padding:.6rem .8rem;margin-bottom:.35rem;}
.conn-hd{display:flex;align-items:center;gap:.45rem;font-size:.8rem;}
.conn-name{font-weight:600;}
.conn-type{font-size:.63rem;color:var(--text_muted);background:var(--surface);border:1px solid var(--border);padding:.08rem .35rem;border-radius:.3rem;font-family:var(--font-mono);}
.conn-dot{font-size:.55rem;cursor:pointer;}.conn-dot.ok{color:#00ffa2;}.conn-dot.warn{color:#ffcc00;}.conn-dot.err{color:#ff5f5f;}
"""

# --- Routes ---

@router.get("/", response_class=HTMLResponse)
async def index(request: Request):
    user = request.state.user
    policy = _get_policy()
    is_admin = getattr(user, "role", "") == "admin"
    state = await TM._load(request)
    # If not admin and a lockdown is active, ignore the requested path and force the redirect
    if not is_admin and policy.get("auto_redirect"):
        active_tool = policy["auto_redirect"]
        show_navigation = False
        state = {"tabs": {"_lock": {"id": "_lock", "path": active_tool, "label": active_tool, "order": 0}}, "active": "_lock"}
    else:
        active_tool = _tool_key(state.get("tabs", {}).get(state.get("active"), {}).get("path", "launcher"))
        show_navigation = True

    reg  = _registry()
    grps = ""
    for gk, gi in TOOL_GROUPS.items():
        items = [(tk, tv) for tk, tv in reg.items() if tv["group"] == gk and _is_allowed(request, tk)]
        if not items: continue
        rows = "".join(f"""<div class="ait-item" hx-post="/im/in" hx-vals='{json.dumps({"type": "ai_tools_open_tab", "path": "settings/connections" if tk == "settings" else tk, "label": tv["label"], "icon": tv["icon"], "id": f"ait-{tk}", "lvl": 1, "branch": "ai_tools"})}' hx-target="body" hx-swap="none" title="{tv["description"]}">{tv["icon"]} {tv["label"]}</div>""" for tk, tv in items)
        grps += f'<details class="ait-grp" open><summary>{gi["icon"]} {gi["label"]}</summary><div class="ait-grp-items">{rows}</div></details>'
    _left_bar = f'<nav class="ait-nav">{grps}</nav>'

    state, content = await _tab_content(request, state)
    ctx = {"request": request, "user": user, "nesting_level": 1, "extra_css": CSS + _sub_css, "shell_id": IM.branch_id, "content": f'<div id="ait-workspace">{content}</div>', "toolbars": {}}
    if show_navigation:
        tab_bar = await TM.tab_bar_fn(state, "ait-tab-bar", "ai_tools", 1)
        ctx["toolbars"] = {"top": UI.toolbar(side="top", content=tab_bar, size="2.5rem", id="ait-top",   nesting_level=1, start_open=True, resizable=False),
                           "left": UI.toolbar(side="left", content=_left_bar, size="12rem", id="ait-left", nesting_level=1, start_open=False, resizable=True),
                           "right": UI.toolbar(side="right", content=f'<div id="ait-right-panel">{_right_panel(active_tool)}</div>', size="12rem", id="ait-right", nesting_level=1, start_open=False, resizable=True)}
    return ENV["templates"].TemplateResponse(name = "base.html", request = request, context = ctx)

@router.get("/launcher", response_class=HTMLResponse)
async def launcher():
    reg   = _registry()
    cards = "".join(f"""<div class="ait-lc" hx-post="/im/in" hx-vals='{json.dumps({"type": "ai_tools_open_tab", "path": tk, "label": tv["label"], "icon": tv["icon"], "id": f"ait-{tk}", "lvl": 1, "branch": "ai_tools"})}' hx-target="body" hx-swap="none">
                            <span class="ait-lc-icon">{tv["icon"]}</span>
                            <div><div class="ait-lc-label">{tv["label"]}</div><div class="ait-lc-desc">{tv["description"]}</div></div>
                        </div>"""for tk, tv in reg.items() if tk != "settings")
    return HTMLResponse(f'<div style="padding:2rem 2.5rem;height:100%;overflow:auto;box-sizing:border-box;"><h1 style="opacity:.08;font-size:3rem;margin:0 0 1.5rem;">AI&#x25B3;TOOLS</h1><div class="ait-launch-grid">{cards}</div></div>')

@router.get("/right_panel/{tool_key}", response_class=HTMLResponse)
async def right_panel_route(tool_key: str): return HTMLResponse(_right_panel(tool_key))

@router.get("/conn_status/{cid}")
async def conn_status(cid: str):
    conn = _load_conn(cid)
    if not conn: return HTMLResponse(f'<span id="conn-dot-{cid}" class="conn-dot err">&#x25CF;</span>')
    tmpl = _tmpls().get(conn.get("connection_type", ""), {})
    ep = tmpl.get("endpoints", {}).get("health", {})
    url = f"{AIM.connections._base(conn)}{ep.get('path','/')}"
    cls, title = "err", "unreachable"
    try:
        async with httpx.AsyncClient(timeout=httpx.Timeout(connect=3.0, read=5.0, write=3.0, pool=3.0), follow_redirects=True) as c:
            r = await c.request(ep.get("method", "GET"), url)
            cls, title = ("ok", f"online - HTTP {r.status_code}") if r.status_code == 200 else ("warn", f"HTTP {r.status_code} @ {url}")
    except Exception as e: title = f"{str(e)[:60]} @ {url}"
    return HTMLResponse(f'<span id="conn-dot-{cid}" class="conn-dot {cls}" hx-get="{_P}/conn_status/{cid}" hx-trigger="every 30s" hx-swap="outerHTML" title="{title}">&#x25CF;</span>')

@router.get("/settings/connections/", response_class=HTMLResponse)
async def settings_conns(): return HTMLResponse(_conns_html())

@router.get("/settings/hw", response_class=HTMLResponse)
async def settings_hw():
    hw_mod = Tools.get("ai_calc")
    note = (f"""<p style="font-size:.8rem;color:var(--text_muted)">Hardware profiles are managed in <button class="ui-btn" style="display:inline;padding:.2rem .5rem" hx-post="/im/in" hx-vals='{json.dumps({"type":"ai_tools_open_tab","path":"ai_calc","label":"AI Calc","icon":"&#x25B3;","id":"ait-ai_calc","lvl":1,"branch":"ai_tools"})}' hx-target="body" hx-swap="none">AI Calc</button>.</p>""" if hw_mod else '<p style="font-size:.8rem;color:var(--text_muted)">AI Calc module not loaded.</p>')
    return HTMLResponse(f'<div style="padding:1.5rem"><h2 style="margin:0 0 1rem;font-size:1rem">Hardware Profiles</h2>{note}</div>')

@router.get("/settings/new_conn_form", response_class=HTMLResponse)
async def new_conn_form(request: Request, conn_type: str = "ollama"): return HTMLResponse(_conn_form(conn_type))

@router.post("/settings/create_conn", response_class=HTMLResponse)
async def create_conn(request: Request):
    form = await request.form()
    ctype = form.get("connection_type", "ollama")
    schema = _tmpls().get(ctype, {}).get("connection", {})
    vals = {}
    for fn, fd in schema.items():
        ft = fd.get("type", "string")
        if ft == "boolean": vals[fn] = bool(form.get(f"field_{fn}"))
        elif ft == "integer":
            raw_val = form.get(f"field_{fn}")
            if not raw_val:
                vals[fn] = None
            else:
                try: vals[fn] = int(raw_val)
                except: vals[fn] = fd.get("default", 0)
        else: vals[fn] = form.get(f"field_{fn}", fd.get("default", ""))
    cid = f"{ctype}_{uuid.uuid4().hex[:8]}"
    _save_conn(cid, {"connection_type": ctype, "display_name": form.get("display_name", "").strip() or ctype, "values": vals, "created": datetime.utcnow().isoformat()})
    return HTMLResponse(_conns_html())

@router.get("/settings/edit_conn/{cid}", response_class=HTMLResponse)
async def edit_conn(cid: str):
    c = _load_conn(cid)
    return HTMLResponse(_conn_form(c["connection_type"], c, cid) if c else '<div style="color:#ff5f5f">Not found.</div>')

@router.post("/settings/save_conn/{cid}", response_class=HTMLResponse)
async def save_conn(cid: str, request: Request):
    form = await request.form()
    c = _load_conn(cid)
    if not c: raise HTTPException(404)
    schema = _tmpls().get(c["connection_type"], {}).get("connection", {})
    c["display_name"] = form.get("display_name", c["display_name"])
    for fn, fd in schema.items():
        ft = fd.get("type", "string")
        if ft == "boolean": c["values"][fn] = bool(form.get(f"field_{fn}"))
        elif ft == "integer":
            raw_val = form.get(f"field_{fn}")
            if not raw_val:
                c["values"][fn] = None
            else:
                try: c["values"][fn] = int(raw_val)
                except: pass
        else: c["values"][fn] = form.get(f"field_{fn}", c["values"].get(fn, ""))
    c["modified"] = datetime.utcnow().isoformat()
    _save_conn(cid, c)
    return HTMLResponse(_conns_html())

@router.delete("/settings/conn/{cid}", response_class=HTMLResponse)
async def del_conn(cid: str):
    _del_conn(cid)
    return HTMLResponse(_conns_html())

@router.post("/settings/test_conn/{cid}", response_class=HTMLResponse)
async def test_conn(cid: str):
    c = _load_conn(cid)
    if not c: return HTMLResponse('<div style="color:#ff5f5f">Not found.</div>')
    tmpl = _tmpls().get(c.get("connection_type", ""), {})
    ep = tmpl.get("endpoints", {}).get("list_models", tmpl.get("endpoints", {}).get("health", {}))
    sc, msg, models_html = "#ff5f5f", "unreachable", ""
    try:
        async with httpx.AsyncClient(timeout=5.0, trust_env=False, follow_redirects=True) as cl:
            r = await cl.request(ep.get("method", "GET"), AIM.connections._base(c))
            print("TESTING2", r)
            if r.status_code == 200:
                sc, msg = "#00ffa2", f"Connected - HTTP {r.status_code}"
                try:
                    data   = r.json()
                    parser = tmpl.get("model_list_parser", {})
                    models = data.get(parser.get("root_key", "models"), []) if isinstance(data, dict) else []
                    if models:
                        nf   = parser.get("name_field", "name")
                        rows = "".join(f'<div style="font-size:.68rem;font-family:var(--font-mono);padding:.1rem 0;border-bottom:1px solid var(--border)">{m.get(nf,"?")}</div>' for m in models[:20])
                        models_html = f'<div style="margin-top:.5rem"><b style="font-size:.73rem">Models ({len(models)})</b>{rows}</div>'
                except: pass
            else:
                sc, msg = "#ffcc00", f"HTTP {r.status_code}"
    except httpx.ConnectTimeout:
        sc, msg = "#ff5f5f", "Connection timed out (Check host/port)"
    except Exception as e:
        sc, msg = "#ff5f5f", f"Error: {str(e)}"
    return HTMLResponse(f'<div class="glass" style="padding:.8rem;margin-top:.5rem"><div style="font-weight:600;font-size:.8rem;color:{sc}">{msg}</div><div style="font-size:.68rem;color:var(--text_muted);font-family:var(--font-mono);margin-top:.2rem">{AIM.connections._base(c)}</div>{models_html}</div>')

@router.get("/settings/policy", response_class=HTMLResponse)
async def settings_policy(request: Request):
    policy = _get_policy()
    return HTMLResponse(f"""
        <div style="padding:1.5rem">
            <h3>Module Lockdown</h3>
            <p style="font-size:.8rem;color:var(--text_muted)">Select a module to force-load for all non-admin users. This hides the sidebar and tab bar.</p>
            <form hx-post="{_P}/settings/save_policy" hx-target="#policy-msg">
                <select name="auto_redirect" class="module-select">
                    <option value="">None (Show Launcher)</option>
                    {"".join(f'<option value="{k}" {"selected" if policy.get("auto_redirect")==k else ""}>{k}</option>' for k in Tools.keys())}
                </select>
                <button type="submit" class="ui-btn" style="margin-top:1rem">Save Policy</button>
                <span id="policy-msg"></span>
            </form>
        </div>
    """)

@router.post("/settings/save_policy")
async def save_policy(request: Request):
    form = await request.form()
    policy = _get_policy()
    policy["auto_redirect"] = form.get("auto_redirect") or None
    _save_policy(policy)
    return HTMLResponse('<span style="color:#00ffa2; font-size:.8rem">Policy Updated!</span>')