# /modules/ai_tools/tessa/tessa.py
"""
Tessa - AI Document and Pipeline Workspace
Sub-module of ai_tools. Mounted at /module/ai_tools/tessa.
Data at data/ai_tools/tessa/. Shared knowledge at data/ai_tools/_knowledge/.
"""
import asyncio, json, uuid, pathlib, copy
from datetime import datetime
from pathlib import Path
import httpx
from fastapi import APIRouter, Request, Form, UploadFile, File
from fastapi.responses import HTMLResponse

TOOL_META = {"label": "Tessa", "icon": "&#x1F4C4;", "description": "AI document workspace and pipeline builder", "singleton": True} #"persistence": "user"}

router = APIRouter(redirect_slashes=False)

_P = "/module/ai_tools/tessa"
DATA_DIR = Path("./data/ai_tools/tessa")
PROJ_DIR = DATA_DIR / "projects"
COMMON_ROOT = Path("./data/_common")
KG_DIR = Path("./data/ai_tools/_knowledge")
COMMON_DIR = Path("./data/_common")
ENV = {}
UI = WS = IM = CM = BI = PE = AIM = _SETTINGS = None
_ACTIVE: set = set()
_STOP: dict = {}
_PIPE_TASKS: dict = {}
_STREAM_TASKS: dict = {}

# --- Helpers ---

def _u(*p): return "/" + "/".join(s.strip("/") for s in [_P.strip("/"), *p] if s)
def _esc(s): return str(s).replace("&","&amp;").replace("<","&lt;").replace(">","&gt;").replace('"',"&quot;")
def _tok(s): return max(1, len(str(s)) // 4)
def _dp(pid): return PROJ_DIR / f"{Path(pid).name}.json"
def _load(pid): p = _dp(pid); return json.loads(p.read_text()) if p.exists() else None

def _save(doc):
    doc["modified"] = datetime.utcnow().isoformat()
    _dp(doc["id"]).write_text(json.dumps(doc, indent=2))

STEP_PROMPTS_FILE = DATA_DIR / "step_prompts.json"

def _load_step_prompts(): return json.loads(STEP_PROMPTS_FILE.read_text()) if STEP_PROMPTS_FILE.exists() else []
def _save_step_prompts(p): STEP_PROMPTS_FILE.write_text(json.dumps(p, indent=2))

def _list_projects(username):
    if not PROJ_DIR.exists(): return []
    out = []
    for f in sorted(PROJ_DIR.glob("*.json"), key=lambda x: x.stat().st_mtime, reverse=True):
        try:
            d = json.loads(f.read_text())
            if d.get("username") == username: out.append(d)
        except: pass
    return out[:200]

def _new_project(user):
    cfg = _SETTINGS.get_group("defaults").load() if _SETTINGS else {}
    return {"id": f"proj_{uuid.uuid4().hex[:8]}", "title": "New Project",
            "username": user.username, "content": "",
            "conn_id": cfg.get("conn_id",""), "model": cfg.get("model",""),
            "model_ctx": int(cfg.get("model_ctx", 32768)),
            "system_prompt": cfg.get("system_prompt",""),
            "conversation": [], "selected_files": [], "pipelines": [],
            "context_summary": "",
            "settings": {"view":"edit","font":"mono","wrap":True},
            "created": datetime.utcnow().isoformat(), "modified": datetime.utcnow().isoformat()}

def _compress(doc):
    hist = [m for m in doc.get("conversation",[]) if not m.get("deleted")]
    if len(hist) <= 40: return doc
    old = hist[:len(hist)-20]; doc["conversation"] = hist[len(hist)-20:]
    lines = [f"{'User' if m['role']=='user' else 'AI'}: {m['content'][:100]}" for m in old]
    ex = doc.get("context_summary","")
    doc["context_summary"] = (ex + " | " if ex else "") + " | ".join(lines)
    return doc

async def _stream(conn, messages, model, num_ctx, think=False):
    """Wraps AIM.connections.stream_llm, preserving this module's (text, thinking, done, error) tuple shape for existing chat-rendering call sites."""
    try:
        async for text, thinking in AIM.connections.stream_llm(conn, messages, model, think=think, num_ctx=num_ctx, num_predict=4096):
            yield text, thinking, False, None
        yield "", "", True, None
    except asyncio.CancelledError: yield "", "", True, None
    except Exception as e: yield "", "", True, str(e)

def _build_messages(doc, user_msg, files_txt=""):
    num_ctx = doc.get("model_ctx", 32768); budget = int(num_ctx * 0.80); used = 0; msgs = []; sys_parts = []
    sys_p = doc.get("system_prompt","").strip()
    if sys_p: sys_parts.append(sys_p); used += _tok(sys_p)
    content = doc.get("content","").strip()
    if content:
        doc_budget = int(budget * 0.30); doc_tok = _tok(content)
        excerpt = content if doc_tok <= doc_budget else "...\n" + content[-(doc_budget*4):]
        sys_parts.append(f"CURRENT DOCUMENT:\n{excerpt}"); used += min(doc_tok, doc_budget)
    if files_txt and used + _tok(files_txt) < int(budget*0.60):
        sys_parts.append(f"[ATTACHED FILES]\n{files_txt}"); used += _tok(files_txt)
    summary = doc.get("context_summary","").strip()
    if summary and used + _tok(summary) < budget:
        sys_parts.append(f"[PRIOR CONTEXT]\n{summary}"); used += _tok(summary)
    if sys_parts: msgs.append({"role":"system","content":"\n\n---\n\n".join(sys_parts)})
    recent = []
    for m in reversed([x for x in doc.get("conversation",[]) if not x.get("deleted")]):
        t = _tok(m.get("content",""))
        if used + t + _tok(user_msg) + 300 > budget: break
        recent.insert(0, {"role":m["role"],"content":m["content"]}); used += t
    msgs.extend(recent); msgs.append({"role":"user","content":user_msg})
    return msgs

def _files_content(selected):
    parts = []
    for rel in selected:
        p = KG_DIR / rel
        if p.exists() and p.is_file():
            try: parts.append(f"--- {rel} ---\n{p.read_text(encoding='utf-8',errors='ignore')}")
            except: pass
    return "\n\n".join(parts)

def _chunk_text(text, max_tokens):
    text = str(text)
    try: max_tokens = int(max_tokens)
    except (TypeError, ValueError): max_tokens = 6000
    if max_tokens <= 0 or _tok(text) <= max_tokens: return [text]
    chars = max_tokens * 4
    chunks = []
    while len(text) > chars:
        split = text.rfind("\n", 0, chars)
        if split <= 0: split = chars
        chunks.append(text[:split])
        text = text[split:].lstrip("\n")
    if text: chunks.append(text)
    return chunks

# --- Init ---

def get_model_options(values=None):
    conn = AIM.connections.get_conn((values or {}).get("conn_id",""))
    return [(m, m) for m in AIM.connections.list_models_sync(conn)] if conn else []

def init_tool(env:dict, prefix:str):
    global ENV, UI, WS, IM, CM, BI, PE, _SETTINGS, AIM
    ENV = env
    UI = env["templates"].env.globals.get("UI")
    WS = env["ws"]
    for d in (PROJ_DIR, KG_DIR, DATA_DIR/"versions"): d.mkdir(parents=True, exist_ok=True)
    BI = env["tools"]["built_ins"]
    AIM = ENV["tools"]["ai_manager"]
    AIM.register_root("tessa", str(DATA_DIR))
    _SETTINGS = BI.SettingsPanel("Tessa", [BI.SettingsGroup("defaults", "Defaults", [BI.SettingField("title", "Title", "text", "Tessa"),
                                                                                     BI.SettingField("conn_id", "Default Connection", "select", options=[("","(none)")] + [(c["_id"], c.get("display_name",c["_id"])) for c in AIM.connections.list_conns()]),
                                                                                     BI.SettingField("model", "Default Model", "select", options=get_model_options),
                                                                                     BI.SettingField("model_ctx", "Context Tokens", "number", 32768),
                                                                                     BI.SettingField("system_prompt", "Default System Prompt", "textarea", "You are a helpful AI assistant."),
                                                                                     BI.SettingField("chunk_tokens", "Default Chunk Tokens", "number", 6000),
                                                                                     BI.SettingField("auto_snapshot", "Auto-snapshot on save", "checkbox", False)], json_path=str(DATA_DIR / "settings.json"))])
    IM = env["InterfaceManager"](nesting_level=2, db_path="tessa_im.db")
    CM = BI.ChatManager(namespace="tessa", base_url=_u(), view_style="bubble", stream_toggle=True, think_toggle=True, stop_enabled=True, pin_enabled=True, allow_edit=True, allow_delete=True, allow_copy=True, show_info=False, markdown_mode="standard", placeholder="Chat about this project\u2026 (Ctrl+Enter)", branch_id=IM.branch_id, nesting_level=2)

    class _TessaEditor(BI.PortalEditor):
        """PortalEditor already turns save/settings/preview/rename/search/info/clean/toggle_task into IM intents when constructed with IM
        Only override the two/three methods that need to persist into Tessa's own per-project JSON instead of a FileManager file."""
        async def _get_doc_from_state(self, request, payload):
            pid = payload.get("branch", "default"); doc = _load(pid)
            return {"id": pid, "title": (doc or {}).get("title","Untitled"), "content": payload.get("content", (doc or {}).get("content","")), "settings": (doc or {}).get("settings",{})}
        async def _im_save(self, request, payload, imr):
            pid = payload.get("branch", "default"); doc = _load(pid)
            if not doc or doc.get("username") != request.state.user.username: return imr.raw('<span style="color:#ff5f5f;font-size:.7rem">&#x26A0; denied</span>')
            doc["content"] = payload.get("content", ""); _save(doc)
            s = doc.get("settings", {})
            imr.oob(self.render_preview(doc["content"], zoom=s.get("zoom",1.0), task_interactive=s.get("interactive",False), doc_id=pid), f"editor-preview-{pid}")
            return imr.raw('<span style="color:var(--accent);font-size:.7rem">&#x2713;</span>')
        async def _im_settings(self, request, payload, imr):
            pid = payload.get("branch", "default"); doc = _load(pid)
            if not doc: return imr
            s = doc.setdefault("settings", {})
            for k in ("view","wrap","font","zoom","border","interactive"):
                if k in payload: s[k] = payload[k]
            _save(doc)
            return imr.raw(self.render_shell({"id": pid, "title": doc.get("title","Untitled"), "content": doc.get("content",""), "settings": s}, include_css=False))
        async def _im_rename(self, request, payload, imr):
            pid = payload.get("branch", ""); doc = _load(pid)
            if doc:
                doc["title"] = payload.get("value","").strip() or "Untitled"; _save(doc)
                imr.oob(_proj_list_html(request.state.user.username, pid), "tessa-proj-list")
            return imr.raw(f'<input id="doc-title-{pid}" type="text" value="{_esc(payload.get("value",""))}" name="value" class="doc-title-input">')
    PE = _TessaEditor(base_url=_u(), autosave_delay="2000ms", enable_graphviz=True, enable_ai=True, IM=IM, nesting_level=2, intent_prefix="tessa_doc")
    IM.scripts["submit"] = [_handle_submit]
    IM.scripts.update({"tessa_doc_apply_ai": [_h_doc_apply_ai], "tessa_doc_conn": [_h_doc_conn], "tessa_doc_model": [_h_doc_model], "tessa_doc_ctx": [_h_doc_ctx], "tessa_files_toggle": [_h_files_toggle]})
    print("[tessa] ready")

# --- Chat Stream ---

async def _handle_submit(request, payload, imr):
    pid = payload.get("cid","").strip(); content = payload.get("content","").strip()
    if not pid or not content: return imr
    imr.raw(CM.working_html(pid, _u("stop", pid)))
    imr.raw(f'<textarea id="cm-in-{pid}" name="content" class="cm-input" placeholder="Chat about this project\u2026 (Ctrl+Enter)" hx-swap-oob="outerHTML"></textarea>')
    _STREAM_TASKS[pid] = asyncio.create_task(_do_stream(request.state.user.username, payload, pid))
    await asyncio.sleep(0.05)
    return imr

async def _do_stream(username, payload, pid):
    content = payload.get("content","").strip(); think = payload.get("think") in ("1","true",True)
    async def _ws(html): await WS.send_personal_message(html, username)
    async def _err(msg): await _ws(f'<div id="cm-msgs-{pid}" hx-swap-oob="beforeend"><div style="color:#ff5f5f;font-size:.78rem;padding:.3rem .6rem">&#x26A0; {_esc(msg)}</div></div>{CM.working_hide_html(pid)}')
    full = ""; tb = ""
    try:
        doc = _load(pid)
        if not doc or doc.get("username") != username: await _err("Project not found."); return
        conn = AIM.connections.get_conn(doc.get("conn_id","")); model = doc.get("model","")
        if not conn: await _err("No connection configured. Set one in the top bar."); return
        if not model: await _err("No model selected. Choose one in the top bar."); return
        num_ctx = doc.get("model_ctx", 32768)
        if _tok(content) > int(num_ctx * 0.65): await _err(f"Input too long (~{_tok(content)}t, limit ~{int(num_ctx*0.65)}t for {num_ctx} context)"); return
        user_msg = {"id":uuid.uuid4().hex[:8],"role":"user","content":content,"user_name":username,"timestamp":datetime.utcnow().isoformat()}
        doc["conversation"].append(user_msg); _save(doc)
        await _ws(f'<div id="cm-msgs-{pid}" hx-swap-oob="beforeend">{CM.render_message(user_msg, is_me=True, can_delete=True, can_edit=True)}</div>')
        files_txt = _files_content(doc.get("selected_files",[])); _ACTIVE.add(pid)
        try:
            async for text, thinking, done, err in _stream(conn, _build_messages(doc, content, files_txt), model, num_ctx, think):
                if _STOP.pop(pid, False): break
                if err: await _err(err); return
                if text: full += text
                if thinking: tb = thinking
                think_html = f'<details class="cm-think" open><summary>\U0001f9e0 Thinking\u2026</summary><div class="cm-think-body">{_esc(tb[-2000:])}</div></details>' if tb.strip() else ""
                await _ws(f'<div id="cm-stream-{pid}" hx-swap-oob="innerHTML">{think_html}{"<div class=cm-stream-bubble>"+_esc(full)+"</div>" if full else ""}</div>')
                if done: break
        finally: _ACTIVE.discard(pid)
        if not full: await _ws(f'<div id="cm-stream-{pid}" hx-swap-oob="innerHTML"></div>{CM.working_hide_html(pid)}'); return
        doc = _load(pid)
        if doc:
            ai_msg = {"id":uuid.uuid4().hex[:8],"role":"assistant","content":full,"thinking":tb.strip(),"model":model,"timestamp":datetime.utcnow().isoformat()}
            doc["conversation"].append(ai_msg)
            if len(doc["conversation"]) == 2: doc["title"] = content[:50]
            _save(_compress(doc))
            await _ws(f"""<div id="cm-msgs-{pid}" hx-swap-oob="beforeend">{CM.render_message(ai_msg, is_me=False, can_delete=True, can_edit=False)}</div><div id="cm-stream-{pid}" hx-swap-oob="innerHTML"></div>{CM.working_hide_html(pid)}<div id="tessa-proj-list" hx-swap-oob="innerHTML">{_proj_list_html(username, pid)}</div>""")
    except Exception as e:
        print(f"[tessa] stream error {pid}: {e}")
        if full:
            try:
                doc2 = _load(pid)
                if doc2:
                    doc2["conversation"].append({"id":uuid.uuid4().hex[:8],"role":"assistant","content":full,"partial":True,"timestamp":datetime.utcnow().isoformat()})
                    _save(doc2)
            except Exception: pass
        await _err(f"Error: {e}")
    finally:
        _STREAM_TASKS.pop(pid, None)

# --- HTML Builders ---

def _proj_list_html(username, active_id=""):
    projects = _list_projects(username)
    if not projects: return '<div style="color:var(--text_muted);font-size:.8rem;padding:.5rem">No projects yet.</div>'
    out = ""
    for p in projects:
        pid = p["id"]; title = _esc((p.get("title","") or "Untitled")[:42]); date = (p.get("modified","") or "")[:10]
        act = "background:var(--glass);border-left:.15rem solid var(--accent);" if pid==active_id else ""
        out += (f"""<div id="tessa-pi-{pid}" style="padding:.3rem .3rem;cursor:pointer;border-bottom:var(--border-thick) solid var(--border);font-size:.8rem;{act}" hx-get="{_u("load",pid)}" hx-target="#tessa-center" hx-swap="innerHTML"><div style="display:flex;align-items:center;gap:.2rem"><span style="flex:1;font-weight:500;overflow:hidden;text-overflow:ellipsis;white-space:nowrap">{title}</span><button class="cm-qbtn" style="color:#ff5f5f" hx-delete="{_u("project",pid)}" hx-target="#tessa-proj-list" hx-swap="innerHTML" hx-confirm="Delete '{title}'?" onclick="event.stopPropagation()">&#x2715;</button></div><div style="font-size:.6rem;color:var(--text_muted)">{date}</div></div>""")
    return out

def _conn_bar_html(doc, conns, models):
    pid = doc["id"]; cid = doc.get("conn_id",""); mdl = doc.get("model",""); ctx = doc.get("model_ctx",32768)
    c_opts = AIM.connections.conn_opts_html(cid) or '<option value="">No connections</option>'
    m_opts = "".join(f'<option value="{m}" {"selected" if m==mdl else ""}>{m}</option>' for m in models) or AIM.connections.model_opts_html(cid, mdl)
    return f"""<div style="display:flex;align-items:center;gap:.4rem;height:100%;padding:0 .2rem;overflow:hidden;">
        <select class="module-select" style="font-size:.7rem;max-width:8rem;flex-shrink:0" name="value" hx-post="/im/in" hx-vals='{{"type":"tessa_doc_conn","branch":"{pid}","lvl":2}}' hx-trigger="change" hx-target="#tessa-model-wrap" hx-swap="innerHTML" hx-include="this">{c_opts}</select>
        <div id="tessa-model-wrap" style="flex-shrink:0"><select class="module-select" style="font-size:.7rem;max-width:11rem" name="value" hx-post="/im/in" hx-vals='{{"type":"tessa_doc_model","branch":"{pid}","lvl":2}}' hx-trigger="change" hx-include="this" hx-swap="none">{m_opts}</select></div>
        <label style="font-size:.6rem;color:var(--text_muted);white-space:nowrap;flex-shrink:0">ctx <input type="number" name="value" value="{ctx}" min="512" max="262144" class="module-select" style="width:5rem;font-size:.6rem;padding:.2rem .2rem" hx-post="/im/in" hx-vals='{{"type":"tessa_doc_ctx","branch":"{pid}","lvl":2}}' hx-trigger="change" hx-include="this" hx-swap="none"></label>
        <button class="btn-icon" style="font-size:.6rem;flex-shrink:0;margin-left:auto" hx-get="{_u("settings")}" hx-target="#tessa-center" hx-swap="innerHTML" title="Tessa Settings">&#x2699;</button>
    </div>"""

def _kg_html(doc):
    pid = doc["id"]; selected = set(doc.get("selected_files",[]))
    badge = f'<span style="background:var(--accent_dim);color:var(--accent);border-radius:.2rem;padding:.05rem .3rem;font-size:.6rem">{len(selected)}</span>' if selected else ""
    tree = UI.tree(items=KG_DIR, mode="file", selectable=True, selected=selected, post_url="/im/in", target="#tessa-kg-section", swap="outerHTML", extra_vals={"type": "tessa_files_toggle", "branch": pid, "lvl": 2}) if KG_DIR.exists() else '<div style="font-size:.7rem;color:var(--text_muted);padding:.2rem .4rem">No knowledge files yet.</div>'
    return f"""<div id="tessa-kg-section" style="border-top:var(--border-thick) solid var(--border)"><details><summary style="padding:.3rem .45rem;cursor:pointer;font-size:.7rem;text-transform:uppercase;letter-spacing:.05em;color:var(--text_muted);list-style:none;user-select:none;display:flex;align-items:center;gap:.3rem">&#x1F4DA; Knowledge {badge}</summary><div style="max-height:28vh;overflow-y:auto;padding:.25rem .4rem">{tree}</div><form hx-post="{_u("kg/upload",pid)}" hx-target="#tessa-kg-section" hx-swap="outerHTML" hx-encoding="multipart/form-data" style="padding:.2rem .4rem;border-top:var(--border-thick) solid var(--border)"><label class="btn-icon" style="cursor:pointer;font-size:.7rem;width:100%;justify-content:center" title="Upload knowledge files">&#x2B06; Upload files<input type="file" name="files" multiple style="display:none" onchange="this.closest('form').requestSubmit()"></label></form></details></div>"""

def _shadow_store_for(root: Path) -> "BI.ShadowStore": return BI.ShadowStore(BI.FileManager(root), root / "_shadow")

def _shadow_panel_html(pid: str) -> str:
    wiki_rows = BI.shadow_review_html(_shadow_store_for(COMMON_ROOT), _u("shadow/wiki/accept")+"/{path}", _u("shadow/wiki/reject")+"/{path}", _u("shadow/wiki/diff")+"/{path}", list_id="shadow-list-wiki")
    kg_rows = BI.shadow_review_html(_shadow_store_for(KG_DIR), _u("shadow/kg/accept")+"/{path}", _u("shadow/kg/reject")+"/{path}", _u("shadow/kg/diff")+"/{path}", list_id="shadow-list-kg")
    return f"""<div id="tessa-shadow-section" style="border-top:var(--border-thick) solid var(--border)">
                   <details>
                       <summary style="padding:.2rem .2rem;cursor:pointer;font-size:.7rem;text-transform:uppercase;letter-spacing:.05em;color:var(--text_muted);list-style:none;user-select:none">&#x1F441; Pending Reviews</summary>
                       <div style="max-height:26vh;overflow-y:auto;padding:.2rem .3rem">
                           <div style="font-size:.6rem;color:var(--text_muted);text-transform:uppercase;padding:.2rem 0">Wiki</div><div id="shadow-list-wiki">{wiki_rows}</div>
                           <div style="font-size:.6rem;color:var(--text_muted);text-transform:uppercase;padding:.4rem 0 .2rem">Knowledge</div><div id="shadow-list-kg">{kg_rows}</div>
                       </div>
                    </details>
                </div>"""

@router.post("/shadow/{scope}/accept/{rel_path:path}", response_class=HTMLResponse)
async def shadow_accept(scope: str, rel_path: str):
    shadow = _shadow_store_for(COMMON_ROOT if scope == "wiki" else KG_DIR)
    shadow.accept(rel_path)
    return HTMLResponse(BI.shadow_review_html(shadow, _u(f"shadow/{scope}/accept")+"/{path}", _u(f"shadow/{scope}/reject")+"/{path}", _u(f"shadow/{scope}/diff")+"/{path}", list_id=f"shadow-list-{scope}"))

@router.post("/shadow/{scope}/reject/{rel_path:path}", response_class=HTMLResponse)
async def shadow_reject(scope: str, rel_path: str):
    shadow = _shadow_store_for(COMMON_ROOT if scope == "wiki" else KG_DIR)
    shadow.reject(rel_path)
    return HTMLResponse(BI.shadow_review_html(shadow, _u(f"shadow/{scope}/accept")+"/{path}", _u(f"shadow/{scope}/reject")+"/{path}", _u(f"shadow/{scope}/diff")+"/{path}", list_id=f"shadow-list-{scope}"))

@router.get("/shadow/{scope}/diff/{rel_path:path}", response_class=HTMLResponse)
async def shadow_diff_route(scope: str, rel_path: str): return HTMLResponse(f'<pre style="white-space:pre-wrap;margin:0">{_esc(_shadow_store_for(COMMON_ROOT if scope == "wiki" else KG_DIR).diff(rel_path))}</pre>')

def _left_bottom_html(doc): return _kg_html(doc) + _pipelines_panel_html(doc["id"]) + _shadow_panel_html(doc["id"])

def _left_panel(username, doc):
    pid = doc["id"]
    return (f"""<div style="display:flex;flex-direction:column;height:100%;overflow:hidden">
                    <div style="flex-shrink:0;padding:.35rem .5rem;border-bottom:var(--border-thick) solid var(--border);display:flex;align-items:center;gap:.3rem">
                        <button class="btn-icon" style="font-size:1rem" hx-post="{_u("new")}" hx-target="#tessa-center" hx-swap="innerHTML" title="New project">+</button>
                        <span style="font-size:.7rem; text-transform:uppercase;letter-spacing:.05em;color:var(--text_muted);flex:1">Tessa</span>
                        <button class="btn-icon" style="font-size:.7rem" hx-get="{_u("settings")}" hx-target="#tessa-center" hx-swap="innerHTML" title="Settings">&#x2699;</button>
                    </div>
                    <div id="tessa-proj-list" style="flex:1;min-height:0;overflow-y:auto">{_proj_list_html(username, pid)}</div>
                    <div id="tessa-left-bottom" style="flex-shrink:0;overflow-y:auto;border-top:var(--border-thick) solid var(--border)">{_left_bottom_html(doc)}</div>
                </div>""")

# -- Pipeline Run --

def _project_pipelines(pid: str) -> list: return [p for p in AIM.engine.list_pipelines() if p.get("project_id") == pid]

def _new_pipeline_for_project(pid: str, name: str) -> dict:
    pl = AIM.engine.new_pipeline(owner="tessa", name=name)
    pl["project_id"], pl["tags"] = pid, ["tessa"]
    AIM.engine.save_pipeline(pl)
    return pl

def _recompute_next(flow: dict):
    """next is derived, never authored directly - every node's next = ids of nodes that list it in their own prev. Keeps the graph internally consistent regardless of edit order."""
    for n in flow["nodes"]: n["next"] = []
    by_id = {n["id"]: n for n in flow["nodes"]}
    for n in flow["nodes"]:
        for p in n.get("prev", []):
            if p in by_id: by_id[p]["next"].append(n["id"])

def _step_type_options(selected="") -> str:
    blank = '<option value="" selected disabled>-- select step type --</option>' if not selected else ""
    return blank + "".join(f'<option value="{t["type"]}" {"selected" if t["type"]==selected else ""}>{_esc(t.get("label",t["type"]))}</option>' for t in AIM.steps.list_step_types())


# def _step_config_form_fields(step_type: str, config: dict, pid: str) -> str: return BI.SettingsGroup(name="cfg", label="", fields=AIM.steps.get_step_type(step_type)["config_schema"], json_path="").render(config)
def _step_config_form_fields(step_type: str, config: dict, pid: str) -> str:
    schema = list(AIM.steps.get_step_type(step_type)["config_schema"])
    has_model = any(f.name == "model" for f in schema)
    if has_model:
        fields = []
        for f in schema:
            if f.name == "conn_id":
                f = copy.copy(f)
                f.hx_get, f.hx_target = _u("step_models", pid), "#cfg_model_wrap"
            fields.append(f)
        schema = fields
    return BI.SettingsGroup(name="cfg", label="", fields=schema, json_path="").render(config, name_prefix="cfg_")


def _parse_node_form(form, step_type: str = "") -> tuple:
    config = {}
    for field in AIM.steps.get_step_type(step_type)["config_schema"]:
        raw = form.get(f"cfg_{field.name}")
        if field.type == "number":
            try: config[field.name] = int(raw)
            except (ValueError, TypeError):
                try: config[field.name] = float(raw)
                except (ValueError, TypeError): config[field.name] = field.default or 0
        elif field.type == "checkbox": config[field.name] = raw is not None
        else: config[field.name] = raw or field.default or ""
    return config, form.getlist("prev"), form.get("join", "all")

def _node_multiselect(nodes: list, selected: list, exclude_id: str = "") -> str: return "".join(f"""<label style="display:flex;align-items:center;gap:.2rem; font-size:.7rem; padding:.1rem 0"><input type="checkbox" name="prev" value="{n["id"]}" {"checked" if n["id"] in selected else ""}> {_esc(n.get("name") or n["id"])} <span style="color:var(--text_muted)">({_esc(n.get("type",""))})</span></label>""" for n in nodes if n["id"] != exclude_id) or '<div style="font-size:.7rem; color:var(--text_muted)">No other nodes yet — this will be a start node.</div>'

def _node_form_html(pid: str, pl: dict, node: dict = None) -> str:
    nodes = pl.get("flow", {}).get("nodes", [])
    nid, is_new = (node or {}).get("id", ""), not (node or {}).get("id")
    ntype, config, prev = (node or {}).get("type", ""), (node or {}).get("config", {}), (node or {}).get("prev", [])
    cfg_fields = _step_config_form_fields(ntype, config, pid) if ntype else '<div style="color:var(--text_muted); font-size:.7rem">Pick a step type to configure it.</div>'
    action = _u("pipeline_node_save",pid,pl["id"],nid) if not is_new else _u("pipeline_node_add",pid,pl["id"])
    del_btn = f'<button type="button" class="btn-icon" style="color:#ff5f5f" hx-post="{_u("pipeline_node_delete",pid,pl["id"],nid)}" hx-target="#tessa-node-editor" hx-swap="innerHTML" hx-confirm="Remove node?">Remove</button>' if not is_new else ""
    join_sel = f"""<label style="font-size:.7rem;color:var(--text_muted)">Join (if multiple 'runs after')
                        <select name="join" class="module-select" style="font-size:.7rem">
                            <option value="all" {"selected" if (node or {}).get("join","all")=="all" else ""}>All must complete (default)</option>
                            <option value="any" {"selected" if (node or {}).get("join","all")=="any" else ""}>Any one is enough</option>
                       </select>
                   </label>"""
    return f"""<form hx-post="{action}" hx-target="#tessa-node-editor" hx-swap="innerHTML" style="display:flex; flex-direction:column; gap:.2rem; padding:.2rem">
                   <span style="font-weight:600; font-size:.8rem; color:var(--accent)">{"New Node" if is_new else "Edit Node"}</span>
                   <input type="text" name="name" value="{_esc((node or {}).get('name',''))}" placeholder="Node name" class="module-select" style="font-size:.8rem">
                   <label style="font-size:.7rem; color:var(--text_muted)">
                       Step Type
                       <select name="type" class="module-select" style="font-size:.7rem" hx-post="{_u("pipeline_node_type_change",pid,pl["id"])}" hx-trigger="change" hx-include="this" hx-target="#tessa-node-cfg" hx-swap="innerHTML">
                           {_step_type_options(ntype)}
                       </select>
                   </label>
                   <div id="tessa-node-cfg" style="display:flex;flex-direction:column; gap:.2rem">{cfg_fields}</div>
                   <label style="font-size:.7rem; color:var(--text_muted)">Runs after</label>
                   {join_sel}
                   {_node_multiselect(nodes, prev, nid)}
                   <div style="display:flex; gap:.1rem"><button type="submit" class="button" style="flex:1">{"Add Node" if is_new else "Save Node"}</button>{del_btn}</div>
               </form>"""

@router.post("/pipeline_node_type_change/{pid}/{pl_id}", response_class=HTMLResponse)
async def pipeline_node_type_change(pid: str, pl_id: str, request: Request):
    form = await request.form()
    return HTMLResponse(_step_config_form_fields(form.get("type",""), {}, pid))

@router.post("/step_models/{pid}", response_class=HTMLResponse)
async def step_models(pid: str, request: Request):
    form = await request.form(); conn = AIM.connections.get_conn(form.get("cfg_conn_id",""))
    models = AIM.connections.list_models_sync(conn) if conn else []
    opts = "".join(f'<option value="{m}">{m}</option>' for m in models) or '<option value="">No models</option>'
    return HTMLResponse(f'<label id="cfg_model_wrap" style="display:block;margin-bottom:1rem">Model<select name="cfg_model" class="module-select" style="width:100%">{opts}</select></label>')

@router.get("/pipeline_node_form/{pid}/{pl_id}", response_class=HTMLResponse)
async def pipeline_node_form_new(pid: str, pl_id: str):
    pl = AIM.engine.load_pipeline(pl_id)
    return HTMLResponse(_node_form_html(pid, pl) if pl else "")

@router.get("/pipeline_node_form/{pid}/{pl_id}/{nid}", response_class=HTMLResponse)
async def pipeline_node_form_edit(pid: str, pl_id: str, nid: str):
    pl = AIM.engine.load_pipeline(pl_id)
    if not pl: return HTMLResponse("")
    node = next((n for n in pl.get("flow",{}).get("nodes",[]) if n["id"]==nid), None)
    return HTMLResponse(_node_form_html(pid, pl, node))

@router.post("/pipeline_node_add/{pid}/{pl_id}", response_class=HTMLResponse)
async def pipeline_node_add(pid: str, pl_id: str, request: Request):
    form = await request.form(); pl = AIM.engine.load_pipeline(pl_id)
    if not pl: return HTMLResponse("")
    config, prev, join = _parse_node_form(form, form.get("type", ""))
    flow = pl.setdefault("flow", {"nodes": []})
    flow["nodes"].append({"id": f"n_{uuid.uuid4().hex[:8]}", "name": form.get("name","").strip(), "type": form.get("type",""), "config": config, "prev": prev, "join": join, "next": []})
    _recompute_next(flow); AIM.engine.save_pipeline(pl)
    return HTMLResponse(_pipeline_editor_html(pid, pl))

@router.post("/pipeline_node_save/{pid}/{pl_id}/{nid}", response_class=HTMLResponse)
async def pipeline_node_save(pid: str, pl_id: str, nid: str, request: Request):
    form = await request.form(); pl = AIM.engine.load_pipeline(pl_id)
    if not pl: return HTMLResponse("")
    flow = pl.setdefault("flow", {"nodes": []})
    node = next((n for n in flow["nodes"] if n["id"] == nid), None)
    if not node: return HTMLResponse("")
    config, prev, join = _parse_node_form(form, form.get("type", node["type"]))
    node["name"], node["type"], node["config"], node["prev"], node["join"] = form.get("name","").strip(), form.get("type", node["type"]), config, prev, join
    _recompute_next(flow); AIM.engine.save_pipeline(pl)
    return HTMLResponse(_pipeline_editor_html(pid, pl))

@router.post("/pipeline_node_delete/{pid}/{pl_id}/{nid}", response_class=HTMLResponse)
async def pipeline_node_delete(pid: str, pl_id: str, nid: str):
    pl = AIM.engine.load_pipeline(pl_id)
    if not pl: return HTMLResponse("")
    flow = pl.setdefault("flow", {"nodes": []})
    flow["nodes"] = [n for n in flow["nodes"] if n["id"] != nid]
    for n in flow["nodes"]: n["prev"] = [p for p in n.get("prev", []) if p != nid]
    _recompute_next(flow); AIM.engine.save_pipeline(pl)
    return HTMLResponse(_pipeline_editor_html(pid, pl))

def _pipeline_editor_html(pid: str, pl: dict) -> str:
    nodes = pl.get("flow", {}).get("nodes", [])
    rows = "".join(f"""<div style="display:flex; align-items:center; gap:.2rem; padding:.2rem .4rem; border-bottom:var(--border-thick) solid var(--border); font-size:.7rem">
                           <span style="flex:1; overflow:hidden;text-overflow:ellipsis;white-space:nowrap" title="after: {', '.join(n.get('prev',[])) or '(start)'}">
                               {_esc(n.get("name") or n["id"])}
                               <span style="color:var(--text_muted)">({_esc(n.get("type",""))})</span>
                           </span>
                           <button class="btn-icon" style="font-size:.6rem" hx-get="{_u("pipeline_node_form",pid,pl["id"],n["id"])}" hx-target="#tessa-node-editor" hx-swap="innerHTML">&#x270E;</button>
                       </div>""" for n in nodes) or '<div style="font-size:.7rem; color:var(--text_muted); padding:.2rem">No nodes yet.</div>'
    return f"""<div style="display:flex;flex-direction:column;height:100%;overflow:hidden">
                   <div style="flex-shrink:0;padding:.2rem .2rem;border-bottom:var(--border-thick) solid var(--border);display:flex;align-items:center;gap:.2rem">
                       <input type="text" value="{_esc(pl.get("name",""))}" class="module-select" style="flex:1; font-size:.8rem" hx-post="{_u("pipeline_rename",pid,pl["id"])}" hx-trigger="change" hx-swap="none" name="name">
                       <span style="font-size:.6rem;color:var(--text_muted)">tags: {", ".join(pl.get("tags",[]))}</span>
                   </div>
                   <div class="tessa-pl-split" style="display:flex;flex:1;min-height:0;overflow:hidden">
                       <div id="tessa-node-editor" style="flex:1;min-width:0;min-height:0;overflow-y:auto;border-right:var(--border-thick) solid var(--border)">
                           <div style="padding:2rem;color:var(--text_muted); font-size:.8rem; text-align:center">Select or add a node.</div>
                           <div class="tessa-pl-steps" style="flex:0 0 14rem; min-height:0;display:flex;flex-direction:column;overflow:hidden"></div>
                           <div style="flex:1;overflow-y:auto">{rows}</div>
                           <button class="btn-icon" style="font-size:.7rem;padding:.2rem" hx-get="{_u("pipeline_node_form",pid,pl["id"])}" hx-target="#tessa-node-editor" hx-swap="innerHTML">+ Add Node</button>
                       </div>
                   </div>
               </div>"""

@router.post("/pipeline_rename/{pid}/{pl_id}", response_class=HTMLResponse)
async def pipeline_rename(pid: str, pl_id: str, request: Request):
    form = await request.form(); pl = AIM.engine.load_pipeline(pl_id)
    if pl: pl["name"] = form.get("name","").strip() or pl["name"]; AIM.engine.save_pipeline(pl)
    return HTMLResponse("")

@router.get("/pipeline_editor/{pid}/{pl_id}", response_class=HTMLResponse)
async def pipeline_editor_route(pid: str, pl_id: str):
    pl = AIM.engine.load_pipeline(pl_id)
    if not pl: return HTMLResponse("")
    return HTMLResponse(f"""<style>
                            @media (max-width: 46rem) {{
                               .tessa-pl-modal {{ width:98vw !important; height:94vh !important; }}
                               .tessa-pl-split {{ flex-direction: column !important; }}
                               .tessa-pl-split #tessa-node-editor {{ flex: 1 1 60% !important; border-right:none !important; }}
                               .tessa-pl-split .tessa-pl-steps {{ flex: 0 0 auto !important; max-height: 8rem !important; border-top: var(--border-thick) solid var(--border) !important; }}
                               .tessa-pl-split select, .tessa-pl-split input {{ font-size: .8rem !important; }}
                            }}
                            </style>
                            <div class="glass tessa-pl-modal" style="overflow:hidden;display:flex;flex-direction:column;padding:0">
                                <div style="display:flex;align-items:center;padding:.2rem .2rem;border-bottom:var(--border-thick) solid var(--border);flex-shrink:0">
                                    <span style="font-weight:600;color:var(--accent);font-size:.8rem">Pipeline Editor</span>
                                    <button onclick="document.getElementById('tessa-modal').style.display='none';document.getElementById('tessa-modal').innerHTML=''" style="margin-left:auto;background:none;border:none;cursor:pointer;font-size:1rem;color:var(--text_muted)">&#x2715;</button>
                                </div>
                                <div style="flex:1;min-height:0;overflow:hidden">{_pipeline_editor_html(pid, pl)}</div>
                            </div>
                            <script>document.getElementById("tessa-modal").style.display="flex";</script>""")

def _pipelines_panel_html(pid: str) -> str:
    pls = _project_pipelines(pid)
    cards = "".join(_pipeline_card_html(pid, pl) for pl in pls) or '<div style="font-size:.7rem;color:var(--text_muted);padding:.2rem .2rem">No pipelines. Click + to create one.</div>'
    return f"""<div id="tessa-pipelines-section" style="border-top:var(--border-thick) solid var(--border)">
        <details open><summary style="paddtessa-pl-ne3rem;cursor:pointer;font-size:.7rem;text-transform:uppercase;letter-spacing:.05em;color:var(--text_muted);list-style:none;user-select:none;display:flex;align-items:center;gap:.2rem">&#x26A1; Pipelines
        <button class="btn-icon" style="margin-left:auto;font-size:.7rem" hx-get="{_u("pipeline_new_form",pid)}" hx-target="#tessa-pl-new" hx-swap="innerHTML" onclick="event.stopPropagation()">+</button></summary>
        <label class="btn-icon" style="cursor:pointer;font-size:.7rem" title="Import pipeline JSON">&#x2B06;<input type="file" accept=".json" style="display:none" onchange="tessaImportPipeline(this,'{pid}')"></label>
        <div id="tessa-pl-new"></div><div>{cards}</div></details></div>"""

@router.get("/pipeline_new_form/{pid}", response_class=HTMLResponse)
async def pipeline_new_form(pid: str): return HTMLResponse(f'<form hx-post="{_u("pipeline_create",pid)}" hx-target="#tessa-pipelines-section" hx-swap="outerHTML" style="display:flex;gap:.3rem;padding:.3rem"><input type="text" name="name" class="module-select" placeholder="Pipeline name" style="flex:1;font-size:.75rem" required autofocus><button type="submit" class="button" style="margin-top:0;font-size:.72rem">Create</button></form>')

@router.post("/pipeline_create/{pid}", response_class=HTMLResponse)
async def pipeline_create(pid: str, request: Request):
    form = await request.form(); _new_pipeline_for_project(pid, form.get("name","Pipeline").strip() or "Pipeline")
    return HTMLResponse(_pipelines_panel_html(pid))

@router.post("/pipeline_delete/{pid}/{pl_id}", response_class=HTMLResponse)
async def pipeline_delete(pid: str, pl_id: str):
    AIM.engine.delete_pipeline(pl_id)
    return HTMLResponse(_pipelines_panel_html(pid))

# --- Main View Builder ---

def _project_view(request, doc, models=None):
    username = request.state.user.username
    is_working = doc["id"] in _ACTIVE
    return (PE.render_shell(doc) + f"""<div id="tessa-conn-bar-content" hx-swap-oob="outerHTML">{_conn_bar_html(doc, AIM.connections.list_conns(), models or [])}</div><div id="tessa-chat-area" hx-swap-oob="outerHTML"><div id="tessa-chat-area" style="height:100%;overflow:hidden">{CM.shell(doc["id"], messages=doc.get("conversation",[]), viewer_name=username, is_working=is_working, stop_url=_u("stop",doc["id"]) if is_working else "")}</div></div><div id="tessa-proj-list" hx-swap-oob="innerHTML">{_proj_list_html(username, doc["id"])}</div><div id="tessa-left-bottom" hx-swap-oob="innerHTML">{_left_bottom_html(doc)}</div>""")

# --- Routes: Main ---

@router.get("")
@router.get("/")
async def root(request: Request):
    global CM, PE, BI
    user = request.state.user
    username = user.username
    did = await ENV["get_state"](request, scope="user", namespace="tessa", key="active_pid")
    doc = _load(did) if did else None
    if not doc or doc.get("username") != username:
        docs = _list_projects(username)
        doc = _load(docs[0]["id"]) if docs else None
    if not doc: doc = _new_project(user); _save(doc)
    await ENV["set_state"](request, doc["id"], scope="user", namespace="tessa", key="active_pid")
    conn = AIM.connections.get_conn(doc.get("conn_id",""))
    models = await AIM.connections.list_models_async(conn) if conn else []
    if not doc.get("model") and models: doc["model"] = models[0]; _save(doc)
    chat = f'<div id="tessa-chat-area" style="height:100%; overflow:hidden">{CM.shell(doc["id"], messages=doc.get("conversation",[]), viewer_name=username)}</div>'
    top = f'<div style="position:relative"><div id="tessa-conn-bar-content">{_conn_bar_html(doc, AIM.connections.list_conns(), models)}</div></div>'

    TESSA_SCRIPT = """['step_start','step_done','error'].forEach(k => document.addEventListener('pipeline:'+k, function(e){
                          var row = document.querySelector('tr[data-node="'+e.detail.node+'"]');
                          if(!row) return;
                          var status = row.querySelector('.tsn-status'), preview = row.querySelector('.tsn-preview');
                          if(status) status.textContent = k === 'step_start' ? 'running' : k;
                          if(preview){ if(e.detail.message) preview.textContent = e.detail.message; else if(e.detail.preview) preview.textContent = Object.values(e.detail.preview).join(' | '); }}));
                      function tessaExportPipeline(plId){
                          fetch('/tool/ai_manager/pipelines/'+plId+'/export').then(r=>r.json()).then(d=>{
                              var blob = new Blob([JSON.stringify(d, null, 2)], {type:'application/json'});
                              var a = document.createElement('a'); a.href = URL.createObjectURL(blob); a.download = (d.name||'pipeline')+'.json'; a.click();
                          });}
                      async function tessaImportPipeline(input, pid){
                          var file = input.files[0]; if(!file) return;
                          var text = await file.text();
                          var r = await fetch('/module/ai_tools/tessa/pipeline_import/'+pid, {method:'POST', headers:{'Content-Type':'application/json'}, body:text});
                          document.getElementById('tessa-pipelines-section').outerHTML = await r.text();
                      }
                    """

    return ENV["templates"].TemplateResponse(name="base.html", request=request, context={
        "request": request, "user": user, "nesting_level": 2, "shell_id": IM.branch_id, "code_mirror": True,
        "toolbars": {"top": UI.toolbar(side="top", content=top, size="3rem", overlay=False, start_open=True, locked=True, nesting_level=2),
                     "left":  UI.toolbar(side="left", content=_left_panel(username, doc), size="18rem", overlay=False, start_open=True, resizable=True, nesting_level=2),
                     "right": UI.toolbar(side="right", content=chat, size="22rem", overlay=False, start_open=True, resizable=True, nesting_level=2, id="tessa-right")},
        "content": f"""<div id="tessa-center">{PE.render_shell(doc)}</div><div id="tessa-modal" onclick="if(event.target===this){{this.style.display='none'; this.innerHTML=''}}"></div>""",
        "extra_css": CSS + CM.CSS + PE.CSS, "extra_script": BI.PORTAL_EDITOR_JS + CM.SCRIPT + TESSA_SCRIPT + BI.PROMPT_BLOCK_JS})

def _pipe_status_poll_js(pl_id: str) -> str:
    """Polls /tool/ai_manager/job_status for this pipeline's current job and updates the overall label, the recent-log strip, and every per-node table row - all from one fetch, independent of WS.
    Safe to call even if data-job is empty (it just returns immediately)."""
    base = AIM.job_status_url("")
    return f"""<script>
    (function poll(){{
        var el = document.getElementById('tessa-status-{pl_id}');
        if(!el || !el.dataset.job) return;
        fetch('{base}' + encodeURIComponent(el.dataset.job)).then(r=>r.json()).then(job=>{{
            var label = el.querySelector('.tessa-status-label');
            if(label) label.textContent = (job.status || 'unknown') + ' @ ' + new Date().toLocaleTimeString();
            var log = el.parentElement.querySelector('.tessa-status-log');
            if(log && job.log) log.innerHTML = job.log.slice(-8).map(l=>'<div>'+l+'</div>').join('');
            (job.flow && job.flow.nodes ? job.flow.nodes : []).forEach(function(n){{
                var row = document.querySelector('tr[data-node="'+n.id+'"]');
                if(!row) return;
                var s = row.querySelector('.tsn-status'), p = row.querySelector('.tsn-preview');
                if(s) s.textContent = (n.status || 'idle') + (n.ts ? ' (' + n.ts + ')' : '');
                if(p) p.textContent = n.message || (n.preview ? Object.values(n.preview).join(' | ') : '');
            }});
            if(job.status === 'running' || job.status === 'queued') setTimeout(poll, 1200);
        }}).catch(()=>setTimeout(poll, 2000));
    }})();
    </script>"""

def _status_block_html(pid: str, pl_id: str, job_id: str, err: str) -> str:
    """Renders the status row + log strip + poller for a JUST-STARTED job (from pipeline_run or pipeline_resume). This is the html returned directly by those two routes."""
    if err:
        return f"""<div id="tessa-status-{pl_id}" data-job="" style="display:flex;align-items:center;gap:.4rem;margin-top:.2rem">
                       <button class="cm-qbtn" hx-post="{_u("pipeline_run",pid,pl_id)}" hx-target="#tessa-status-{pl_id}" hx-swap="outerHTML" hx-include="#tessa-input-{pl_id}">&#x25B6; Run</button>
                       <span class="tessa-status-label" style="font-size:.6rem;color:#ff5f5f">error: {_esc(err)}</span>
                   </div>
                   <div class="tessa-status-log" style="font-size:.6rem;color:var(--text_muted);max-height:4rem;overflow-y:auto"></div>"""
    return f"""<div id="tessa-status-{pl_id}" data-job="{job_id}" style="display:flex;align-items:center;gap:.4rem;margin-top:.2rem">
                   <button class="cm-qbtn" style="color:#ff4444" hx-post="{_u("pipeline_stop",pid,pl_id,job_id)}" hx-target="#tessa-status-{pl_id}" hx-swap="outerHTML">&#x25FC; Stop</button>
                   <span class="tessa-status-label" style="font-size:.65rem;color:#00ffa2">running</span>
               </div>
               <div class="tessa-status-log" style="font-size:.62rem;color:var(--text_muted);max-height:4rem;overflow-y:auto"></div>""" + _pipe_status_poll_js(pl_id)

def _pipeline_card_html(pid: str, pl: dict) -> str:
    pl_id = pl["id"]
    last_job = AIM.engine.load_job(pl.get("last_job_id", "")) if pl.get("last_job_id") else None
    live = bool(last_job and last_job.get("status") in ("running", "queued"))
    interrupted = bool(last_job and last_job.get("status") == "interrupted")
    if live:
        status_html = f"""<div id="tessa-status-{pl_id}" data-job="{pl['last_job_id']}" style="display:flex;align-items:center;gap:.4rem;margin-top:.2rem">
                              <button class="cm-qbtn" style="color:#ff4444" hx-post="{_u("pipeline_stop",pid,pl_id,pl['last_job_id'])}" hx-target="#tessa-status-{pl_id}" hx-swap="outerHTML">&#x25FC; Stop</button>
                              <span class="tessa-status-label" style="font-size:.6rem;color:#00ffa2">{last_job['status']}</span>
                          </div>
                          <div class="tessa-status-log" style="font-size:.6rem;color:var(--text_muted);max-height:4rem;overflow-y:auto"></div>""" + _pipe_status_poll_js(pl_id)
    else:
        state_label = last_job["status"] if last_job else "idle"
        status_html = f"""<div id="tessa-status-{pl_id}" data-job="" style="display:flex;align-items:center;gap:.4rem;margin-top:.2rem">
                              <button class="cm-qbtn" hx-post="{_u("pipeline_run",pid,pl_id)}" hx-target="#tessa-status-{pl_id}" hx-swap="outerHTML" hx-include="#tessa-input-{pl_id}">&#x25B6; Run</button>
                              <span class="tessa-status-label" style="font-size:.6rem; color:var(--text_muted)">{state_label}</span>
                          </div>
                          <div class="tessa-status-log" style="font-size:.6rem; color:var(--text_muted);max-height: 4rem;overflow-y:auto"></div>"""
    resume_row = ""
    if interrupted: resume_row = f'<button class="cm-qbtn" style="width:100%;margin-top:.2rem" hx-post="{_u("pipeline_resume",pid,pl_id)}" hx-target="#tessa-status-{pl_id}" hx-swap="outerHTML">&#x21BB; Resume interrupted run</button>'
    nodes = (last_job["flow"]["nodes"] if last_job else pl.get("flow", {}).get("nodes", []))
    rows = "".join(f"""<tr data-node="{n["id"]}">
                           <td style="font-size:.6rem;padding:.1rem .3rem">{_esc(n.get("name") or n["id"])}</td>
                           <td style="font-size:.6rem;color:var(--text_muted);padding:.1rem .1rem">{_esc(n.get("type",""))}</td>
                           <td class="tsn-status" style="font-size:.6rem; color:var(--text_muted); padding:.1rem .3rem">{_esc(n.get("status","idle"))}{(' ('+n['ts']+')') if n.get('ts') else ''}</td>
                           <td class="tsn-preview" style="font-size:.6rem; color:var(--text_muted); padding:.1rem .3rem; max-width:14rem; overflow:hidden; text-overflow:ellipsis;white-space:nowrap">
                               {_esc(n.get('message') or ' | '.join((n.get('preview') or {}).values()))}
                           </td>
                       </tr>""" for n in nodes)
    return f"""<div class="glass" style="padding=.2rem .5rem;margin:.2rem .2rem; font-size:.7rem">
                   <div style="display:flex;align-items:center;gap:.3rem">
                       <span style="flex:1;overflow:hidden;text-overflow:ellipsis;white-space:nowrap;cursor:pointer" hx-get="{_u("pipeline_editor",pid,pl_id)}" hx-target="#tessa-modal" hx-swap="innerHTML">{_esc(pl.get("name",""))}</span>
                       <button class="btn-icon" style="color:#ff5f5f;font-size:.7rem" hx-post="{_u("pipeline_delete",pid,pl_id)}" hx-target="#tessa-pipelines-section" hx-swap="outerHTML" hx-confirm="Delete pipeline?">&#x2715;</button>
                       <div style="display:flex;gap:.3rem;margin-top:.2rem">
                           <button type="button" class="cm-qbtn" onclick="tessaExportPipeline('{pl_id}')">&#x2B07; Export</button>
                       </div>
                   </div>
                   <input type="text" id="tessa-input-{pl_id}" name="input_value" placeholder="Input for this run" class="module-select" style="width:100%;font-size:.7rem;margin:.2rem 0">
                   {status_html}
                   {resume_row}
                   <details style="margin-top:.2rem"><summary style="font-size:.6rem;color:var(--text_muted);cursor:pointer">Node status ({len(nodes)})</summary>
                   <table style="width:100%;border-collapse:collapse;margin-top:.2rem">{rows}</table></details>
               </div>"""

@router.post("/pipeline_run/{pid}/{pl_id}", response_class=HTMLResponse)
async def pipeline_run(pid: str, pl_id: str, request: Request):
    form = await request.form()
    job_id, err = AIM.engine.submit(request.state.user.username, kind="id", pipeline_id=pl_id, extra_config={"project_id": pid}, inputs={"input": form.get("input_value", "")})
    if not err:
        pl = AIM.engine.load_pipeline(pl_id)
        pl["last_job_id"] = job_id
        AIM.engine.save_pipeline(pl)
    return HTMLResponse(_status_block_html(pid, pl_id, job_id or "", err))

@router.post("/pipeline_resume/{pid}/{pl_id}", response_class=HTMLResponse)
async def pipeline_resume(pid: str, pl_id: str):
    pl = AIM.engine.load_pipeline(pl_id)
    if not pl or not pl.get("last_job_id"): return HTMLResponse(_status_block_html(pid, pl_id, "", "no previous job to resume"))
    job_id, err = AIM.engine.resume(pl["last_job_id"])
    return HTMLResponse(_status_block_html(pid, pl_id, job_id or "", err))

@router.post("/pipeline_stop/{pid}/{pl_id}/{job_id}", response_class=HTMLResponse)
async def pipeline_stop(pid: str, pl_id: str, job_id: str):
    AIM.engine.stop(job_id)
    return HTMLResponse(f"""<div id="tessa-status-{pl_id}" data-job="{job_id}" style="display:flex;align-items:center;gap:.4rem;margin-top:.2rem">
                                <span class="tessa-status-label" style="font-size:.65rem;color:#ffcc00">stopping&#x2026;</span>
                            </div>
                            <div class="tessa-status-log" style="font-size:.62rem;color:var(--text_muted);max-height:4rem;overflow-y:auto"></div>""" + _pipe_status_poll_js(pl_id))

@router.post("/new", response_class=HTMLResponse)
async def new_project(request: Request):
    doc = _new_project(request.state.user)
    _save(doc)
    await ENV["set_state"](request, doc["id"], scope="user", namespace="tessa", key="active_pid")
    models = []
    conn = AIM.connections.get_conn(doc.get("conn_id",""))
    if conn: models = await AIM.connections.list_models_async(conn)
    return HTMLResponse(_project_view(request, doc, models))

@router.get("/load/{pid}", response_class=HTMLResponse)
async def load_project(pid: str, request: Request):
    doc = _load(pid)
    if not doc or doc.get("username") != request.state.user.username: return HTMLResponse("Not found", status_code=404)
    await ENV["set_state"](request, pid, scope="user", namespace="tessa", key="active_pid")
    conn = AIM.connections.get_conn(doc.get("conn_id",""))
    models = await AIM.connections.list_models_async(conn) if conn else []
    return HTMLResponse(_project_view(request, doc, models))

@router.delete("/project/{pid}", response_class=HTMLResponse)
async def delete_project(pid: str, request: Request):
    user = request.state.user; doc = _load(pid)
    if doc and doc.get("username") == user.username: _dp(pid).unlink(missing_ok=True)
    active = await ENV["get_state"](request, scope="user", namespace="tessa", key="active_pid")
    if active == pid: await ENV["set_state"](request, "", scope="user", namespace="tessa", key="active_pid")
    return HTMLResponse(_proj_list_html(user.username, ""))

# --- Routes: Document ---

async def _h_doc_apply_ai(request, payload, imr):
    pid = payload.get("branch",""); doc = _load(pid)
    if not doc or doc.get("username") != request.state.user.username: return imr
    last_ai = next((m["content"] for m in reversed(doc.get("conversation",[])) if m.get("role")=="assistant" and not m.get("deleted")), None)
    if last_ai: doc["content"] = last_ai; _save(doc)
    return imr.raw(PE.render_shell(doc))

async def _h_doc_conn(request, payload, imr):
    pid = payload.get("branch",""); doc = _load(pid)
    if not doc: return imr
    doc["conn_id"] = payload.get("value",""); _save(doc)
    conn = AIM.connections.get_conn(doc["conn_id"]); models = await AIM.connections.list_models_async(conn) if conn else []
    cur = doc.get("model",""); opts = "".join(f'<option value="{m}" {"selected" if m==cur else ""}>{m}</option>' for m in models) or '<option value="">No models</option>'
    imr.oob(f"""<select class="module-select" style="font-size:.7rem; max-width:11rem" name="value" hx-post="/im/in" hx-vals='{{"type":"tessa_doc_model","branch":"{pid}","lvl":2}}' hx-trigger="change" hx-include="this" hx-swap="none">{opts}</select>""", "tessa-model-wrap")
    return imr

async def _h_doc_model(request, payload, imr):
    doc = _load(payload.get("branch",""))
    if doc: doc["model"] = payload.get("value",""); _save(doc)
    return imr

async def _h_doc_ctx(request, payload, imr):
    doc = _load(payload.get("branch",""))
    if doc: doc["model_ctx"] = max(512, int(payload.get("value",32768) or 32768)); _save(doc)
    return imr

async def _h_files_toggle(request, payload, imr):
    pid, path = payload.get("branch",""), payload.get("path","")
    is_dir = str(payload.get("is_dir","false")) == "true"
    doc = _load(pid)
    if not doc: return imr
    files = set(doc.get("selected_files",[]))
    if is_dir:
        full = KG_DIR/path
        children = {str(f.relative_to(KG_DIR)) for f in full.rglob("*") if f.is_file() and not f.name.startswith(".")} if full.is_dir() else set()
        files = files - children if children and children.issubset(files) else files | children
    else: files.discard(path) if path in files else files.add(path)
    doc["selected_files"] = list(files); _save(doc)
    imr.oob(_kg_html(doc), "tessa-kg-section", swap="outerHTML")
    return imr

@router.get("/doc/search_close/{pid}")
async def doc_search_close(pid: str): return HTMLResponse("")

@router.post("/stop/{pid}")
async def stop_stream(pid: str):
    _STOP[pid] = True
    task = _STREAM_TASKS.pop(pid, None)
    if task and not task.done(): task.cancel()
    return HTMLResponse("")

@router.post("/doc/toggle_task/{pid}")
async def doc_toggle_task(pid: str, request: Request):
    form = await request.form(); doc = _load(pid)
    if not doc or doc.get("username") != request.state.user.username: return HTMLResponse("")
    doc["content"] = PE._flip_task(doc.get("content",""), int(form.get("idx", -1)))
    _save(doc)
    return HTMLResponse(PE.render_preview(doc["content"], task_interactive=True, doc_id=pid))

# --- Routes: Messages ---

@router.post("/msg/delete")
async def msg_delete(request: Request):
    form = await request.form(); mid = form.get("id","")
    for doc in _list_projects(request.state.user.username):
        d = _load(doc["id"])
        if not d: continue
        for m in d.get("conversation",[]):
            if m.get("id") == mid:
                m["deleted"] = True
                _save(d)
                return HTMLResponse("")
    return HTMLResponse("")

@router.get("/msg/edit_form/{mid}")
async def msg_edit_form(mid: str, request: Request):
    for doc in _list_projects(request.state.user.username):
        d = _load(doc["id"])
        if not d: continue
        for m in d.get("conversation",[]):
            if m.get("id") != mid: continue
            return HTMLResponse(f"""<div class="cm-msg {"cm-me" if m.get("role")=="user" else "cm-other"}" id="cm-msg-{mid}" data-msg-id="{mid}">
                                        {CM._avatar_html(m.get("user_name","?"))}
                                        <div class="cm-bwrap" style="max-width:90%">
                                            <form hx-post="{_u("msg/edit_save",mid)}" hx-target="#cm-msg-{mid}" hx-swap="outerHTML" style="display:flex;flex-direction:column;gap:.3rem;width:100%">
                                                <textarea name="content" class="cm-input" style="min-height:4rem;overflow-y:auto">{_esc(m.get("content",""))}</textarea>
                                                <div style="display:flex;gap:.3rem">
                                                    <button type="submit" class="button" style="font-size:.7rem;margin-top:0">Save</button>
                                                    <button type="button" class="btn-icon" hx-get="{_u("msg/cancel_edit",mid)}" hx-target="#cm-msg-{mid}" hx-swap="outerHTML">Cancel</button>
                                                </div>
                                            </form>
                                        </div>
                                    </div>""")
    return HTMLResponse("")

@router.post("/msg/edit_save/{mid}")
async def msg_edit_save(mid: str, request: Request):
    form = await request.form(); user = request.state.user
    for doc in _list_projects(user.username):
        d = _load(doc["id"])
        if not d: continue
        for m in d.get("conversation",[]):
            if m.get("id") != mid: continue
            m["content"] = form.get("content","").strip(); m["edited"] = True; _save(d)
            is_me = m.get("role") == "user"
            return HTMLResponse(CM.render_message(m, is_me=is_me, can_delete=True, can_edit=is_me))
    return HTMLResponse("")

@router.get("/msg/cancel_edit/{mid}")
async def msg_cancel_edit(mid: str, request: Request):
    for doc in _list_projects(request.state.user.username):
        d = _load(doc["id"])
        if not d: continue
        for m in d.get("conversation",[]):
            if m.get("id") != mid: continue
            is_me = m.get("role") == "user"
            return HTMLResponse(CM.render_message(m, is_me=is_me, can_delete=True, can_edit=is_me))
    return HTMLResponse("")

@router.post("/msg/retry/{mid}")
async def msg_retry(mid: str, request: Request):
    for doc in _list_projects(request.state.user.username):
        d = _load(doc["id"])
        if not d: continue
        msgs = d.get("conversation",[]); idx = next((i for i,m in enumerate(msgs) if m.get("id")==mid), None)
        if idx is None: continue
        m = msgs[idx]; role_cls = "cm-me" if m.get("role")=="user" else "cm-other"
        return HTMLResponse(f"""<div class="cm-msg {role_cls}" id="cm-msg-{mid}" data-msg-id="{mid}">
                                    {CM._avatar_html(m.get("user_name","?"))}
                                    <div class="cm-bwrap" style="max-width:90%">
                                        <form hx-post="{_u("msg/retry_send",mid)}" hx-target="#cm-msg-{mid}" hx-swap="outerHTML" style="display:flex;flex-direction:column;gap:.3rem;width:100%">
                                            <textarea name="content" class="cm-input" style="min-height:4rem;overflow-y:auto">{_esc(m.get("content",""))}</textarea>
                                            <div style="display:flex;gap:.3rem">
                                                <button type="submit" class="button" style="font-size:.75rem;margin-top:0">&#x21BA; Retry</button>
                                                <button type="button" class="btn-icon" hx-get="{_u("msg/cancel_edit",mid)}" hx-target="#cm-msg-{mid}" hx-swap="outerHTML">Cancel</button>
                                              </div>
                                          </form>
                                      </div>
                                  </div>""")
    return HTMLResponse("")

@router.post("/msg/retry_send/{mid}")
async def msg_retry_send(mid: str, request: Request):
    form = await request.form()
    user = request.state.user
    new_content = form.get("content","").strip()
    for doc in _list_projects(user.username):
        d = _load(doc["id"])
        if not d: continue
        msgs = d.get("conversation",[]); idx = next((i for i,m in enumerate(msgs) if m.get("id")==mid), None)
        if idx is None: continue
        msgs[idx]["content"] = new_content; msgs[idx]["edited"] = True
        d["conversation"] = msgs[:idx+1]; _save(d); pid = d["id"]
        remaining = "".join(CM.render_message(m, is_me=(m.get("role")=="user"), can_delete=True, can_edit=(m.get("role")=="user")) for m in d["conversation"] if not m.get("deleted"))
        conn = AIM.connections.get_conn(d.get("conn_id","")); model = d.get("model","")
        if conn and model: asyncio.create_task(_run_chat_task(pid, user.username, conn, _build_messages(d, new_content), model, d.get("model_ctx", 32768)))
        return HTMLResponse(f'<div id="cm-msgs-{pid}" class="cm-msgs" data-pinned="true" hx-swap-oob="outerHTML">{remaining}</div>')
    return HTMLResponse("")

async def _run_chat_task(pid, username, conn, messages, model, num_ctx, think=False):
    _ACTIVE.add(pid)
    full = ""
    tb = ""
    try:
        async for text, thinking, done, err in _stream(conn, messages, model, num_ctx, think):
            if _STOP.pop(pid, False): break
            if err:
                await WS.send_personal_message(f'<div id="cm-msgs-{pid}" hx-swap-oob="beforeend">{CM.append_system_error_html(pid, err)}</div>{CM.working_hide_html(pid)}', username)
                return
            if text: full += text
            if thinking: tb = thinking
            await WS.send_personal_message(f'<div id="cm-stream-{pid}" hx-swap-oob="innerHTML">{"<div class=cm-stream-bubble>"+_esc(full)+"</div>" if full else ""}</div>', username)
            if done: break
    finally: _ACTIVE.discard(pid)
    if not full: await WS.send_personal_message(f'<div id="cm-stream-{pid}" hx-swap-oob="innerHTML"></div>{CM.working_hide_html(pid)}', username); return
    doc = _load(pid)
    if doc:
        ai_msg = {"id":uuid.uuid4().hex[:8],"role":"assistant","content":full,"model":model,"timestamp":datetime.utcnow().isoformat()}
        doc["conversation"].append(ai_msg); _save(_compress(doc))
        await WS.send_personal_message(f"""<div id="cm-msgs-{pid}" hx-swap-oob="beforeend">{CM.render_message(ai_msg, is_me=False, can_delete=True, can_edit=False)}</div><div id="cm-stream-{pid}" hx-swap-oob="innerHTML"></div>""" + CM.working_hide_html(pid), username)

# --- Routes: Knowledge Files ---

@router.post("/kg/upload/{pid}", response_class=HTMLResponse)
async def kg_upload(pid: str, request: Request, files: list[UploadFile] = File(...)):
    doc = _load(pid)
    if not doc or doc.get("username") != request.state.user.username: return HTMLResponse("Unauthorized", status_code=403)
    KG_DIR.mkdir(parents=True, exist_ok=True)
    for f in files:
        if not f.filename: continue
        dest = KG_DIR / pathlib.Path(f.filename).name
        dest.write_bytes(await f.read())
    return HTMLResponse(_kg_html(doc))

# --- Routes: Settings ---

@router.get("/settings", response_class=HTMLResponse)
async def settings_page(request: Request):
    group = _SETTINGS.get_group("defaults")
    values = group.load()
    body = f"""<form hx-post="{_u("settings/save")}" hx-target="#tessa-settings-status" style="display:flex;flex-direction:column;gap:.6rem">
                    <div id="tessa-settings-fields">{group.render(values)}</div>
                    <button type="submit" class="button" style="margin-top:.5rem">Save Settings</button>
                    <div id="tessa-settings-status" style="font-size:.75rem;min-height:1rem"></div>
                </form>"""
    return HTMLResponse(_SETTINGS.page_shell(body, close_url=_u("settings/close"), target_id="tessa-center", title="Tessa Settings"))

@router.get("/settings/close", response_class=HTMLResponse)
async def settings_close(request: Request):
    doc = _load(await ENV["get_state"](request, scope="user", namespace="tessa", key="active_pid"))
    return HTMLResponse(PE.render_shell(doc) if doc else "")

@router.post("/settings/save", response_class=HTMLResponse)
async def settings_save(request: Request):
    form = dict(await request.form())
    _SETTINGS.get_group("defaults").save(form)
    return HTMLResponse('<span style="color:#00ffa2">&#x2713; Saved</span>')

def _folder_picker(root_key: str, field_name: str, selected: str = "") -> str: return ENV["tools"]["built_ins"].FileManager(AIM.resolve_root(root_key)).folder_picker_html(selected).replace('name="parent"', f'name="{field_name}"')

def _destination_picker_html(field_prefix: str, config: dict, with_filename: bool) -> str:
    roots = AIM.list_roots()
    cur_root = config.get(f"{field_prefix}_root", "common")
    root_radios = "".join(f"""<label style="font-size:.7rem;margin-right:.6rem"><input type="radio" name="cfg_{field_prefix}_root" value="{k}" {"checked" if k==cur_root else ""} onchange="this.form.requestSubmit ? htmx.trigger(this,'change') : null" hx-get="{_u("destination_folder")}" hx-vals='{{"root":"{k}","field":"{field_prefix}"}}' hx-target="#{field_prefix}-folder-wrap" hx-swap="innerHTML" hx-include="this"> {k}</label>""" for k, _ in roots)
    folder_html = _folder_picker(cur_root, f"cfg_{field_prefix}_folder", config.get(f"{field_prefix}_folder", ""))
    filename_html = f'<input type="text" name="cfg_{field_prefix}_filename" value="{_esc(config.get(field_prefix+"_filename",""))}" placeholder="filename, e.g. {{input}}.md" class="module-select" style="font-size:.7rem;margin-top:.2rem">' if with_filename else ""
    return f"""<div style="font-size:.6rem;color:var(--text_muted)">Location<div style="margin:.2rem 0">{root_radios}</div><div id="{field_prefix}-folder-wrap" style="max-height:8rem;overflow-y:auto;border:var(--border-thick) solid var(--border);border-radius:var(--radius);padding:.2rem">{folder_html}</div>{filename_html}</div>"""

@router.get("/destination_folder", response_class=HTMLResponse)
async def destination_folder(root: str, field: str): return HTMLResponse(_folder_picker(root, f"cfg_{field}_folder"))

@router.post("/pipeline_import/{pid}", response_class=HTMLResponse)
async def pipeline_import(pid: str, request: Request):
    body = await request.json()
    body["id"] = f"pl_{uuid.uuid4().hex[:10]}"
    body["project_id"] = pid
    body.setdefault("tags", [])
    if "tessa" not in body["tags"]: body["tags"].append("tessa")
    AIM.engine.save_pipeline(body)
    return HTMLResponse(_pipelines_panel_html(pid))

def _pipelines_panel_html(pid: str) -> str:
    pls = _project_pipelines(pid)
    cards = "".join(_pipeline_card_html(pid, pl) for pl in pls) or '<div style="font-size:.7rem;color:var(--text_muted);padding:.2rem .2rem">No pipelines. Click + to create one.</div>'
    return f"""<div id="tessa-pipelines-section" style="border-top:var(--border-thick) solid var(--border)">
        <details open><summary style="padding:.3rem;cursor:pointer;font-size:.7rem;text-transform:uppercase;letter-spacing:.05em;color:var(--text_muted);list-style:none;user-select:none;display:flex;align-items:center;gap:.2rem">&#x26A1; Pipelines
        <button class="btn-icon" style="margin-left:auto;font-size:.7rem" hx-get="{_u("pipeline_new_form",pid)}" hx-target="#tessa-pl-new" hx-swap="innerHTML" onclick="event.stopPropagation()">+</button></summary>
        <label class="btn-icon" style="cursor:pointer;font-size:.7rem" title="Import pipeline JSON">&#x2B06;<input type="file" accept=".json" style="display:none" onchange="tessaImportPipeline(this,'{pid}')"></label>
        <div id="tessa-pl-new"></div><div>{cards}</div>{_orphaned_pipelines_html(pid)}</details></div>"""

# --- CSS ---

CSS = """
#tessa-proj-list .active-item{background:var(--glass);border-left:.1rem solid var(--accent);}
#tessa-modal{display:none;position:fixed;inset:0;z-index:2000;align-items:center;justify-content:center;background:rgba(0,0,0,0.65);}
.editor-shell{display:flex;flex-direction:column;height:100%;width:100%;overflow:hidden;}
#tessa-center{display:flex;flex-direction:column;height:100%;width:100%;overflow:hidden;}
#tessa-pipe-bar{height:100%;display:flex;align-items:stretch;}
.tessa-pl-modal{width:70rem;max-width:92vw;height:82vh;max-height:82vh;}
"""

async def _tessa_doc_read(config, ctx):
    doc = _load(config["project_id"])
    return {"text": (doc or {}).get("content", "")}
async def _tessa_doc_write(config, ctx):
    doc = _load(config["project_id"])
    if doc: doc["content"] = ctx.resolve(f"{{{config.get('from_node','')}.text}}"); _save(doc)
    return {"status": "ok"}

@router.post("/pipeline_claim/{pid}/{pl_id}", response_class=HTMLResponse)
async def pipeline_claim(pid: str, pl_id: str):
    """One-time fixup for pipelines saved before project_id was tracked - claims an orphaned pipeline for this project so it becomes visible in the sidebar."""
    pl = AIM.engine.load_pipeline(pl_id)
    if pl and not pl.get("project_id"): pl["project_id"] = pid; AIM.engine.save_pipeline(pl)
    return HTMLResponse(_pipelines_panel_html(pid))

def _orphaned_pipelines_html(pid: str) -> str:
    orphans = [p for p in AIM.engine.list_pipelines() if not p.get("project_id")]
    if not orphans: return ""
    rows = "".join(f'<div style="display:flex;align-items:center;gap:.3rem;font-size:.7rem;padding:.15rem .3rem"><span style="flex:1">{_esc(p.get("name",p["id"]))}</span><button class="cm-qbtn" hx-post="{_u("pipeline_claim",pid,p["id"])}" hx-target="#tessa-pipelines-section" hx-swap="outerHTML">Claim for this project</button></div>' for p in orphans)
    return f'<details style="border-top:var(--border-thick) solid var(--border)"><summary style="font-size:.65rem;color:var(--text_muted);padding:.2rem .4rem;cursor:pointer">Unclaimed pipelines ({len(orphans)})</summary>{rows}</details>'
