# /modules/ai_tools/tessa/tessa.py
"""
Tessa - AI Document and Pipeline Workspace
Sub-module of ai_tools. Mounted at /module/ai_tools/tessa.
Data at data/ai_tools/tessa/. Shared knowledge at data/ai_tools/_knowledge/.
"""
import asyncio, json, uuid, pathlib, copy, re
from datetime import datetime
from pathlib import Path
import httpx
from fastapi import APIRouter, Request, Form, UploadFile, File
from fastapi.responses import HTMLResponse

TOOL_META = {"label": "Tessa", "icon": "&#x1F4C4;", "description": "AI document workspace and pipeline builder", "singleton": True}

router = APIRouter(redirect_slashes=False)

_P = "/module/ai_tools/tessa"
DATA_DIR = Path("./data/ai_tools/tessa")
PROJ_DIR = DATA_DIR / "projects"
COMMON_ROOT = Path("./data/_common")
KG_DIR = Path("./data/ai_tools/_knowledge")
COMMON_DIR = Path("./data/_common")
ENV = {}
UI = WS = IM = CM = BI = PE = AIM = PB =_SETTINGS = _bottom_tm = None
_ACTIVE: set = set()
_STOP: dict = {}
_STREAM_TASKS: dict = {}

def _u(*p): return "/" + "/".join(s.strip("/") for s in [_P.strip("/"), *p] if s)
def _esc(s): return str(s).replace("&","&amp;").replace("<","&lt;").replace(">","&gt;").replace('"',"&quot;")
def _tok(s): return max(1, len(str(s)) // 4)
def _dp(pid): return PROJ_DIR / f"{Path(pid).name}.json"
def _load(pid): p = _dp(pid); return json.loads(p.read_text()) if p.exists() else None
def _save(doc): doc["modified"] = datetime.utcnow().isoformat(); _dp(doc["id"]).write_text(json.dumps(doc, indent=2))

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
    return {"id": f"stu_{uuid.uuid4().hex[:8]}", "username": user.username, "title": "New Project",
            "content": "", "conn_id": cfg.get("conn_id",""), "model": cfg.get("model",""),
            "model_ctx": int(cfg.get("model_ctx", 32768)), "system_prompt": cfg.get("system_prompt",""),
            "conversation": [], "selected_files": [], "context_summary": "",
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

def get_model_options(values=None):
    conn = AIM.connections.get_conn((values or {}).get("conn_id",""))
    return [(m, m) for m in AIM.connections.list_models_sync(conn)] if conn else []

def init_tool(env: dict, prefix: str):
    global ENV, UI, WS, IM, CM, BI, PE, PB, _SETTINGS, AIM
    ENV = env
    UI = env["templates"].env.globals.get("UI")
    WS = env["ws"]
    for d in (PROJ_DIR, KG_DIR, DATA_DIR/"versions"): d.mkdir(parents=True, exist_ok=True)
    BI = env["tools"]["built_ins"]
    AIM = ENV["tools"]["ai_manager"]
    AIM.register_root("tessa", str(DATA_DIR))
    _SETTINGS = BI.SettingsPanel("Tessa", [BI.SettingsGroup("defaults", "Defaults", [
        BI.SettingField("title", "Title", "text", "Tessa"),
        BI.SettingField("conn_id", "Default Connection", "select", options=[("","(none)")] + [(c["_id"], c.get("display_name",c["_id"])) for c in AIM.connections.list_conns()]),
        BI.SettingField("model", "Default Model", "select", options=get_model_options),
        BI.SettingField("model_ctx", "Context Tokens", "number", 32768),
        BI.SettingField("system_prompt", "Default System Prompt", "textarea", "You are a helpful AI assistant."),
        BI.SettingField("auto_snapshot", "Auto-snapshot on save", "checkbox", False)], json_path=str(DATA_DIR / "settings.json"))])
    IM = env["InterfaceManager"](nesting_level=2, db_path="tessa_im.db")
    CM = BI.ChatManager(namespace="tessa", base_url=_u(), view_style="bubble", stream_toggle=True, think_toggle=True, stop_enabled=True, pin_enabled=True, allow_edit=True, allow_delete=True, allow_copy=True, show_info=False, markdown_mode="standard", placeholder="Chat about this project\u2026 (Ctrl+Enter)", branch_id=IM.branch_id, nesting_level=2)

    class _PipelineEditor(BI.PortalEditor):
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
    PE = _PipelineEditor(base_url=_u(), autosave_delay="2000ms", enable_graphviz=True, enable_ai=True, IM=IM, nesting_level=2, intent_prefix="tessa_doc")
    PB = AIM.PipelineBuilderUI(IM, AIM, intent_prefix="tessa_pl", nesting_level=2, scope_key="project_id")
    IM.scripts["submit"] = [_handle_submit]
    IM.scripts.update({"tessa_doc_apply_ai": [_h_doc_apply_ai], "tessa_doc_conn": [_h_doc_conn], "tessa_doc_model": [_h_doc_model], "tessa_doc_ctx": [_h_doc_ctx], "tessa_files_toggle": [_h_files_toggle]})
    IM.scripts["tessa_shadow_action"] = [_h_shadow_action]
    print("[tessa] ready")

async def _handle_submit(request, payload, imr):
    pid = payload.get("cid","").strip(); content = payload.get("content","").strip()
    if not pid or not content: return imr
    imr.raw(CM.working_html(pid, _u("stop", pid)))
    imr.raw(f'<textarea id="cm-in-{pid}" name="content" class="cm-input" placeholder="Chat about this project\u2026 (Ctrl+Enter)" hx-swap-oob="outerHTML"></textarea>')
    _STREAM_TASKS[pid] = asyncio.create_task(_do_stream(request.state.user.username, payload, pid))
    await asyncio.sleep(0.05)
    return imr

async def _do_stream(username, payload, pid, skip_user_append=False):
    content = payload.get("content","").strip(); think = payload.get("think") in ("1","true",True)
    async def _ws(html): await WS.send_personal_message(html, username)
    async def _err(msg, retry_mid=None):
        retry_html = f' <button class="cm-qbtn" hx-post="{_u("msg/retry_send",retry_mid)}" hx-target="#cm-msgs-{pid}" hx-swap="outerHTML" hx-vals=\'{{"content":""}}\'>&#x21BA; Retry</button>' if retry_mid else ""
        await _ws(f'<div id="cm-msgs-{pid}" hx-swap-oob="beforeend"><div style="color:#ff5f5f;font-size:.8rem;padding:.3rem .6rem">&#x26A0; {_esc(msg)}{retry_html}</div></div>{CM.working_hide_html(pid)}')
    full = ""; tb = ""
    doc = _load(pid)
    if not doc or doc.get("username") != username: await _err("Project not found."); return
    user_msg = None
    if not skip_user_append:
        user_msg = {"id":uuid.uuid4().hex[:8],"role":"user","content":content,"user_name":username,"timestamp":datetime.utcnow().isoformat()}
        doc["conversation"].append(user_msg); _save(doc)
        await _ws(f'<div id="cm-msgs-{pid}" hx-swap-oob="beforeend">{CM.render_message(user_msg, is_me=True, can_delete=True, can_edit=True)}</div>')
    try:
        conn_id = doc.get("conn_id","")
        conn = AIM.connections.get_conn(conn_id)
        if not conn and conn_id:
            fallback = AIM.connections.get_conn("")
            if fallback:
                doc["conn_id"] = fallback["_id"]; _save(doc); conn = fallback
                await _ws(f'<div id="cm-msgs-{pid}" hx-swap-oob="beforeend"><div style="font-size:.6rem;color:#ffaa44;padding:.1rem .2rem">&#x26A0; Saved connection no longer exists - switched to {_esc(fallback.get("display_name",fallback["_id"]))}. Check the top bar.</div></div>')
        model = doc.get("model","")
        if not conn: await _err("No connection available. Add one in AI Tools > Settings.", retry_mid=user_msg["id"] if user_msg else None); return
        if not model: await _err("No model selected. Choose one in the top bar.", retry_mid=user_msg["id"] if user_msg else None); return
        num_ctx = doc.get("model_ctx", 32768)
        if _tok(content) > int(num_ctx * 0.65): await _err(f"Input too long (~{_tok(content)}t, limit ~{int(num_ctx*0.65)}t for {num_ctx} context). Edit the message above and retry.", retry_mid=user_msg["id"] if user_msg else None); return
        files_txt = _files_content(doc.get("selected_files",[])); _ACTIVE.add(pid)
        try:
            async for text, thinking, done, err in _stream(conn, _build_messages(doc, content, files_txt), model, num_ctx, think):
                if _STOP.pop(pid, False): break
                if err: await _err(err, retry_mid=user_msg["id"] if user_msg else None); return
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

def _shadow_rows_html():
    wiki_rows = BI.shadow_review_html(_shadow_store_for(COMMON_ROOT), "tessa_shadow_action", {"scope": "wiki"}, list_id="shadow-list-wiki")
    kg_rows = BI.shadow_review_html(_shadow_store_for(KG_DIR), "tessa_shadow_action", {"scope": "kg"}, list_id="shadow-list-kg")
    return f"""<div style="font-size:.6rem;color:var(--text_muted);text-transform:uppercase;padding:.2rem 0">Wiki</div><div id="shadow-list-wiki">{wiki_rows}</div>
               <div style="font-size:.6rem;color:var(--text_muted);text-transform:uppercase;padding:.4rem 0 .2rem">Knowledge</div><div id="shadow-list-kg">{kg_rows}</div>"""

async def _h_shadow_action(request, payload, imr):
    scope, action, path = payload.get("scope","wiki"), payload.get("action",""), payload.get("path","")
    shadow = _shadow_store_for(COMMON_ROOT if scope == "wiki" else KG_DIR)
    if action == "diff":
        imr.oob(f'<pre style="white-space:pre-wrap;margin:0">{_esc(shadow.diff(path))}</pre>', payload.get("diff_target",""))
        return imr
    if action == "accept": shadow.accept(path)
    elif action == "reject": shadow.reject(path)
    imr.oob(BI.shadow_review_html(shadow, "tessa_shadow_action", {"scope": scope}, list_id=f"shadow-list-{scope}"), f"shadow-list-{scope}", swap="innerHTML")
    return imr

def _left_bottom_html(doc): return _kg_html(doc) + PB.panel_html(doc["id"])

def _left_panel(username, doc):
    pid = doc["id"]
    return (f"""<div style="display:flex;flex-direction:column;height:100%;overflow:hidden">
                    <div style="flex-shrink:0;padding:.35rem .5rem;border-bottom:var(--border-thick) solid var(--border);display:flex;align-items:center;gap:.3rem">
                        <button class="btn-icon" style="font-size:1rem" hx-post="{_u("new")}" hx-target="#tessa-center" hx-swap="innerHTML" title="New project">+</button>
                        <span style="font-size:.7rem; text-transform:uppercase;letter-spacing:.05em;color:var(--text_muted);flex:1">Tessa</span>
                        <button class="btn-icon" style="font-size:.7rem" hx-get="{_u("settings")}" hx-target="#tessa-center" hx-swap="innerHTML" title="Settings">&#x2699;</button>
                    </div>
                    <div id="tessa-proj-list" style="flex:1;min-height:0;overflow-y:auto">{_proj_list_html(username, pid)}</div>
                    <div id="tessa-left-bottom" style="flex:0 1 auto;max-height:55vh;overflow-y:auto;border-top:var(--border-thick) solid var(--border)">{_left_bottom_html(doc)}</div>
                </div>""")

async def _project_view(request, doc, models=None):
    username = request.state.user.username
    is_working = doc["id"] in _ACTIVE
    return (PE.render_shell(doc) + f"""<div id="tessa-conn-bar-content" hx-swap-oob="outerHTML">{_conn_bar_html(doc, AIM.connections.list_conns(), models or [])}</div><div id="tessa-chat-area" hx-swap-oob="outerHTML"><div id="tessa-chat-area" style="height:100%;overflow:hidden">{CM.shell(doc["id"], messages=doc.get("conversation",[]), viewer_name=username, is_working=is_working, stop_url=_u("stop",doc["id"]) if is_working else "")}</div></div><div id="tessa-proj-list" hx-swap-oob="innerHTML">{_proj_list_html(username, doc["id"])}</div><div id="tessa-left-bottom" hx-swap-oob="innerHTML">{_left_bottom_html(doc)}</div>""")

class PipelineBuilderUI:
    """Generalized pipeline authoring surface over AIM.engine/AIM.steps. Fully intent-based - no bespoke
    GET routes, everything flows through /im/in and OOB updates to this instance's own generic ids.
    Any module embeds panel_html(scope_id) once instead of reimplementing node forms/editor/run controls.
    scope_key/scope_id decouple 'what owns this pipeline' from the engine - one module calls it a project,
    another a workspace; the builder only needs a stable string id and the key name it's stored under."""

    SCRIPT = """
    function plbExport(plId) {
        fetch('/tool/ai_manager/pipelines/' + plId + '/export').then(r => r.json()).then(d => {
            var blob = new Blob([JSON.stringify(d, null, 2)], {type: 'application/json'});
            var a = document.createElement('a'); a.href = URL.createObjectURL(blob); a.download = (d.name || 'pipeline') + '.json'; a.click();
        });
    }
    """

    def __init__(self, IM, AIM, intent_prefix="plb", nesting_level=2, scope_key="project_id"):
        self.IM, self.AIM, self.intent_prefix, self.nesting_level, self.scope_key = IM, AIM, intent_prefix, nesting_level, scope_key
        p = self.intent_prefix
        IM.scripts.update({f"{p}_new_form": [self._im_new_form], f"{p}_create": [self._im_create], f"{p}_delete": [self._im_delete],
                            f"{p}_editor_open": [self._im_editor_open], f"{p}_editor_close": [self._im_editor_close],
                            f"{p}_node_form": [self._im_node_form], f"{p}_node_type_change": [self._im_node_type_change],
                            f"{p}_node_add": [self._im_node_save], f"{p}_node_save": [self._im_node_save], f"{p}_node_delete": [self._im_node_delete],
                            f"{p}_rename": [self._im_rename], f"{p}_run": [self._im_run], f"{p}_stop": [self._im_stop], f"{p}_resume": [self._im_resume],
                            f"{p}_status": [self._im_status], f"{p}_step_models": [self._im_step_models], f"{p}_claim": [self._im_claim]})

    def _vals(self, action, **extra): return json.dumps({"type": f"{self.intent_prefix}_{action}", "branch": self.intent_prefix, "lvl": self.nesting_level, **extra})
    def _post(self, action, **extra): return f"""hx-post="/im/in" hx-target="body" hx-swap="none" hx-vals='{self._vals(action, **extra)}'"""
    def _pipelines(self, scope_id): return [p for p in self.AIM.engine.list_pipelines() if p.get(self.scope_key) == scope_id]

    def panel_html(self, scope_id: str) -> str:
        p = self.intent_prefix
        cards = "".join(self._card_html(scope_id, pl) for pl in self._pipelines(scope_id)) or '<div class="pl-empty">No pipelines. Click + to create one.</div>'
        return f"""<div id="pl-panel-{p}" class="pl-panel">
                       <div class="pl-panel-hd"><span class="pl-panel-title">Pipelines</span><button class="btn-icon" {self._post("new_form", scope=scope_id)}>+</button></div>
                       <div id="pl-new-{p}"></div>
                       <div id="pl-list-{p}" class="pl-list">{cards}</div>
                       <div id="pl-editor-modal-{p}"></div>
                   </div>"""

    def _card_html(self, scope_id, pl) -> str:
        pl_id = pl["id"]
        last_job = self.AIM.engine.load_job(pl.get("last_job_id","")) if pl.get("last_job_id") else None
        nodes = (last_job["flow"]["nodes"] if last_job else pl.get("flow",{}).get("nodes",[]))
        rows = "".join(self._node_status_row(n) for n in nodes)
        return f"""<div class="glass pl-card">
                       <div class="pl-card-hd">
                           <span class="pl-card-title" {self._post("editor_open", scope=scope_id, pl_id=pl_id)}>{UI.escape(pl.get("name",""))}</span>
                           <button type="button" class="cm-qbtn" onclick="plbExport('{pl_id}')">&#x2B07;</button>
                           <button class="btn-icon" style="color:#ff5f5f" {self._post("delete", scope=scope_id, pl_id=pl_id)} onclick="return confirm('Delete pipeline?')">&#x2715;</button>
                       </div>
                       <input type="text" id="pl-input-{pl_id}" name="value" placeholder="Input for this run" class="module-select">
                       <div id="pl-status-{pl_id}">{self._status_block(scope_id, pl_id, last_job)}</div>
                       <details class="pl-nodes"><summary>Node status ({len(nodes)})</summary><table class="pl-status-table">{rows}</table></details>
                   </div>"""

    def _status_block(self, scope_id, pl_id, job) -> str:
        live = bool(job and job.get("status") in ("running","queued"))
        if live:
            return f"""<div class="pl-status-row" hx-trigger="load delay:1.5s" {self._post("status", scope=scope_id, pl_id=pl_id)} hx-target="#pl-status-{pl_id}" hx-swap="innerHTML">
                           <button class="cm-qbtn" style="color:#ff4444" {self._post("stop", scope=scope_id, pl_id=pl_id, job_id=job["id"])}>&#x25FC; Stop</button>
                           <span class="pl-status-label" style="color:#00ffa2">{job["status"]}</span></div>"""
        label = job["status"] if job else "idle"
        resume = f"""<button class="cm-qbtn" {self._post("resume", scope=scope_id, pl_id=pl_id)}>&#x21BB; Resume</button>""" if job and job.get("status") == "interrupted" else ""
        return f"""<div class="pl-status-row">
                       <button class="cm-qbtn" {self._post("run", scope=scope_id, pl_id=pl_id)} hx-include="#pl-input-{pl_id}">&#x25B6; Run</button>
                       {resume}<span class="pl-status-label">{label}</span></div>"""

    @staticmethod
    def _node_slug(n) -> str:
        explicit = str(n.get("slug","")).strip().lower()
        if explicit: return re.sub(r'\W+', '_', explicit).strip('_') or n["id"]
        return re.sub(r'\W+', '_', (n.get("name") or "").strip().lower()).strip('_') or n["id"]

    def _node_status_row(self, n) -> str:
        slug, preview = self._node_slug(n), n.get("message") or " | ".join((n.get("preview") or {}).values())
        return f"""<tr><td class="qn">{UI.escape(n.get("name") or n["id"])}<br><code class="pl-slug">{slug}</code></td>
                       <td class="dim">{UI.escape(n.get("type",""))}</td><td class="dim">{UI.escape(n.get("status","idle"))}</td>
                       <td class="dim pl-preview">{UI.escape(preview)}</td></tr>"""

    def _step_type_options(self, selected=""):
        blank = '<option value="" selected disabled>-- select step type --</option>' if not selected else ""
        return blank + "".join(f'<option value="{t["type"]}" {"selected" if t["type"]==selected else ""}>{UI.escape(t.get("label",t["type"]))}</option>' for t in self.AIM.steps.list_step_types())

    def _step_config_form_fields(self, step_type, config):
        spec = self.AIM.steps.get_step_type(step_type)
        if not spec: return '<div class="dim">Pick a step type to configure it.</div>'
        schema = [copy.copy(f) if f.name == "conn_id" else f for f in spec["config_schema"]]
        for f in schema:
            if f.name == "conn_id": f.hx_intent, f.hx_target = f"{self.intent_prefix}_step_models", "#cfg_model_wrap"
        guide_html = f"""<details class="glass pl-guide"><summary>&#x2139; How this node works</summary><div>{UI.escape(spec.get("guide",""))}</div></details>""" if spec.get("guide") else ""
        return guide_html + BI.SettingsGroup(name="cfg", label="", fields=schema, json_path="").render(config, name_prefix="cfg_")

    def _node_multiselect(self, nodes, selected, exclude_id=""):
        rows = ""
        for n in nodes:
            if n["id"] == exclude_id: continue
            alias = self._node_slug(n)
            keys = (self.AIM.steps.get_step_type(n.get("type","")) or {}).get("output_keys", [])
            key_hint = " ".join(f'<code class="pl-keyhint">{{{alias}.{k}}}</code>' for k in keys)
            rows += f"""<label class="pl-check"><input type="checkbox" name="prev" value="{n["id"]}" {"checked" if n["id"] in selected else ""}> {UI.escape(n.get("name") or n["id"])} <span class="dim">({UI.escape(n.get("type",""))})</span>{key_hint}</label>"""
        return rows or '<div class="dim">No other nodes yet - this will be a start node.</div>'

    def _node_form_html(self, scope_id, pl, node=None) -> str:
        p, nodes = self.intent_prefix, pl.get("flow",{}).get("nodes",[])
        nid, is_new = (node or {}).get("id",""), not (node or {}).get("id")
        ntype, config, prev = (node or {}).get("type",""), (node or {}).get("config",{}), (node or {}).get("prev",[])
        del_btn = f"""<button type="button" class="btn-icon" style="color:#ff5f5f" {self._post("node_delete", scope=scope_id, pl_id=pl["id"], nid=nid)} onclick="return confirm('Remove node?')">Remove</button>""" if not is_new else ""
        slug_val = (node or {}).get("slug","") or self._node_slug(node or {})
        return f"""<form {self._post("node_save", scope=scope_id, pl_id=pl["id"], nid=nid) if not is_new else self._post("node_add", scope=scope_id, pl_id=pl["id"])} hx-include="this" class="pl-node-form">
                       <span class="pl-form-title">{"New Node" if is_new else "Edit Node"}</span>
                       <input type="text" name="name" value="{UI.escape((node or {}).get('name',''))}" placeholder="Node name" class="module-select">
                       <label class="dim">Reference name (used as <code>{{this.field}}</code>)<input type="text" name="slug" value="{UI.escape(slug_val)}" class="module-select"></label>
                       <label class="dim">Step Type<select name="type" class="module-select" {self._post("node_type_change", scope=scope_id, pl_id=pl["id"])} hx-trigger="change" hx-include="this" hx-target="#pl-node-cfg-{p}">{self._step_type_options(ntype)}</select></label>
                       <div id="pl-node-cfg-{p}">{self._step_config_form_fields(ntype, config) if ntype else '<div class="dim">Pick a step type to configure it.</div>'}</div>
                       <label class="dim">Runs after</label>
                       {self._node_multiselect(nodes, prev, nid)}
                       <div class="pl-form-actions"><button type="submit" class="button">{"Add Node" if is_new else "Save Node"}</button>{del_btn}</div>
                   </form>"""

    def _editor_html(self, scope_id, pl) -> str:
        p, nodes = self.intent_prefix, pl.get("flow",{}).get("nodes",[])
        rows = "".join(f"""<div class="pl-node-row" {self._post("node_form", scope=scope_id, pl_id=pl["id"], nid=n["id"])}>{UI.escape(n.get("name") or n["id"])} <span class="dim">({UI.escape(n.get("type",""))})</span></div>""" for n in nodes) or '<div class="dim">No nodes yet.</div>'
        return f"""<div class="pl-modal-backdrop" onclick="if(event.target===this) htmx.ajax('POST','/im/in',{{values:{self._vals("editor_close")},swap:'none'}})">
                       <div class="glass pl-modal">
                           <div class="pl-modal-hd">
                               <input type="text" value="{UI.escape(pl.get("name",""))}" class="module-select" {self._post("rename", scope=scope_id, pl_id=pl["id"])} hx-trigger="change" hx-include="this" name="name">
                               <button type="button" class="close-btn" {self._post("editor_close")}>&#x2715;</button>
                           </div>
                           <div class="pl-modal-body">
                               <div id="pl-node-editor-{p}" class="pl-node-editor">
                                   <div class="pl-placeholder">Select or add a node.</div>
                                   <div class="pl-node-list">{rows}</div>
                                   <button class="btn-icon" {self._post("node_form", scope=scope_id, pl_id=pl["id"])}>+ Add Node</button>
                               </div>
                           </div>
                       </div>
                   </div>"""

    @staticmethod
    def _recompute_next(flow):
        for n in flow["nodes"]: n["next"] = []
        by_id = {n["id"]: n for n in flow["nodes"]}
        for n in flow["nodes"]:
            for prv in n.get("prev", []):
                if prv in by_id: by_id[prv]["next"].append(n["id"])

    def _parse_node_form(self, form, step_type=""):
        config = {}
        for field in self.AIM.steps.get_step_type(step_type)["config_schema"]:
            raw = form.get(f"cfg_{field.name}")
            if field.type == "number":
                if raw is None or raw.strip() == "": config[field.name] = field.default
                else:
                    try: config[field.name] = int(raw)
                    except (ValueError, TypeError):
                        try: config[field.name] = float(raw)
                        except (ValueError, TypeError): config[field.name] = field.default
            elif field.type == "checkbox": config[field.name] = raw is not None
            else: config[field.name] = raw if raw is not None else field.default
        return config, form.getlist("prev"), form.get("slug","").strip()

    async def _im_new_form(self, request, payload, imr):
        return imr.oob(f"""<form {self._post("create", scope=payload.get("scope",""))} hx-include="this" class="pl-new-form"><input type="text" name="name" class="module-select" placeholder="Pipeline name" required autofocus><button type="submit" class="button">Create</button></form>""", f"pl-new-{self.intent_prefix}")

    async def _im_create(self, request, payload, imr):
        scope_id = payload.get("scope","")
        pl = self.AIM.engine.new_pipeline(owner=self.intent_prefix, name=payload.get("name","Pipeline").strip() or "Pipeline")
        pl[self.scope_key] = scope_id
        self.AIM.engine.save_pipeline(pl)
        imr.oob("", f"pl-new-{self.intent_prefix}")
        return imr.oob("".join(self._card_html(scope_id, p_) for p_ in self._pipelines(scope_id)) or '<div class="pl-empty">No pipelines.</div>', f"pl-list-{self.intent_prefix}")

    async def _im_delete(self, request, payload, imr):
        scope_id = payload.get("scope","")
        self.AIM.engine.delete_pipeline(payload.get("pl_id",""))
        return imr.oob("".join(self._card_html(scope_id, p_) for p_ in self._pipelines(scope_id)) or '<div class="pl-empty">No pipelines.</div>', f"pl-list-{self.intent_prefix}")

    async def _im_editor_open(self, request, payload, imr):
        pl = self.AIM.engine.load_pipeline(payload.get("pl_id",""))
        return imr.oob(self._editor_html(payload.get("scope",""), pl), f"pl-editor-modal-{self.intent_prefix}") if pl else imr

    async def _im_editor_close(self, request, payload, imr): return imr.oob("", f"pl-editor-modal-{self.intent_prefix}")

    async def _im_node_form(self, request, payload, imr):
        pl = self.AIM.engine.load_pipeline(payload.get("pl_id",""))
        if not pl: return imr
        node = next((n for n in pl.get("flow",{}).get("nodes",[]) if n["id"]==payload.get("nid")), None)
        return imr.oob(self._node_form_html(payload.get("scope",""), pl, node), f"pl-node-editor-{self.intent_prefix}")

    async def _im_node_type_change(self, request, payload, imr):
        return imr.oob(self._step_config_form_fields(payload.get("type",""), {}), f"pl-node-cfg-{self.intent_prefix}")

    async def _im_step_models(self, request, payload, imr):
        conn = self.AIM.connections.get_conn(payload.get("cfg_conn_id",""))
        models = self.AIM.connections.list_models_sync(conn) if conn else []
        opts = "".join(f'<option value="{m}">{m}</option>' for m in models) or '<option value="">No models</option>'
        return imr.oob(f'<label id="cfg_model_wrap" class="dim">Model<select name="cfg_model" class="module-select">{opts}</select></label>', "cfg_model_wrap")

    async def _im_node_save(self, request, payload, imr):
        pl = self.AIM.engine.load_pipeline(payload.get("pl_id",""))
        if not pl: return imr
        flow = pl.setdefault("flow", {"nodes": []})
        nid, ntype = payload.get("nid",""), payload.get("type","")
        config, prev, slug = self._parse_node_form(payload, ntype)
        node = next((n for n in flow["nodes"] if n["id"]==nid), None) if nid else None
        if node: node.update(slug=slug, name=payload.get("name","").strip(), type=ntype, config=config, prev=prev)
        else: flow["nodes"].append({"id": f"n_{uuid.uuid4().hex[:8]}", "slug": slug, "name": payload.get("name","").strip(), "type": ntype, "config": config, "prev": prev, "next": []})
        self._recompute_next(flow)
        self.AIM.engine.save_pipeline(pl)
        return imr.oob(self._editor_html(payload.get("scope",""), pl), f"pl-editor-modal-{self.intent_prefix}")

    async def _im_node_delete(self, request, payload, imr):
        pl = self.AIM.engine.load_pipeline(payload.get("pl_id",""))
        if not pl: return imr
        flow, nid = pl.setdefault("flow", {"nodes": []}), payload.get("nid","")
        flow["nodes"] = [n for n in flow["nodes"] if n["id"] != nid]
        for n in flow["nodes"]: n["prev"] = [pr for pr in n.get("prev",[]) if pr != nid]
        self._recompute_next(flow)
        self.AIM.engine.save_pipeline(pl)
        return imr.oob(self._editor_html(payload.get("scope",""), pl), f"pl-editor-modal-{self.intent_prefix}")

    async def _im_rename(self, request, payload, imr):
        pl = self.AIM.engine.load_pipeline(payload.get("pl_id",""))
        if pl: pl["name"] = payload.get("name","").strip() or pl["name"]; self.AIM.engine.save_pipeline(pl)
        return imr

    async def _im_run(self, request, payload, imr):
        scope_id, pl_id = payload.get("scope",""), payload.get("pl_id","")
        job_id, err = self.AIM.engine.submit(request.state.user.username, kind="id", pipeline_id=pl_id, extra_config={self.scope_key: scope_id}, inputs={"input": payload.get("value","")})
        if not err:
            pl = self.AIM.engine.load_pipeline(pl_id); pl["last_job_id"] = job_id; self.AIM.engine.save_pipeline(pl)
        return imr.oob(self._status_block(scope_id, pl_id, self.AIM.engine.load_job(job_id) if not err else None), f"pl-status-{pl_id}")

    async def _im_resume(self, request, payload, imr):
        pl = self.AIM.engine.load_pipeline(payload.get("pl_id",""))
        job_id, _ = self.AIM.engine.resume(pl.get("last_job_id","")) if pl and pl.get("last_job_id") else (None, "")
        return imr.oob(self._status_block(payload.get("scope",""), payload.get("pl_id",""), self.AIM.engine.load_job(job_id) if job_id else None), f"pl-status-{payload.get('pl_id','')}")

    async def _im_stop(self, request, payload, imr):
        self.AIM.engine.stop(payload.get("job_id",""))
        return imr.oob(self._status_block(payload.get("scope",""), payload.get("pl_id",""), self.AIM.engine.load_job(payload.get("job_id",""))), f"pl-status-{payload.get('pl_id','')}")

    async def _im_status(self, request, payload, imr):
        pl = self.AIM.engine.load_pipeline(payload.get("pl_id",""))
        job = self.AIM.engine.load_job(pl.get("last_job_id","")) if pl and pl.get("last_job_id") else None
        return imr.oob(self._status_block(payload.get("scope",""), payload.get("pl_id",""), job), f"pl-status-{payload.get('pl_id','')}")

    async def _im_claim(self, request, payload, imr):
        pl = self.AIM.engine.load_pipeline(payload.get("pl_id",""))
        if pl and not pl.get(self.scope_key): pl[self.scope_key] = payload.get("scope",""); self.AIM.engine.save_pipeline(pl)
        return imr


@router.get("")
@router.get("/")
async def root(request: Request):
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
    return ENV["templates"].TemplateResponse(name="base.html", request=request, context={
        "request": request, "user": user, "nesting_level": 2, "shell_id": IM.branch_id, "code_mirror": True,
        "toolbars": {"top": UI.toolbar(side="top", content=top, size="3rem", overlay=False, start_open=True, locked=True, nesting_level=2),
                     "left":  UI.toolbar(side="left", content=_left_panel(username, doc), size="18rem", overlay=False, start_open=True, resizable=True, nesting_level=2),
                     "right": UI.toolbar(side="right", content=chat, size="22rem", overlay=False, start_open=True, resizable=True, nesting_level=2, id="tessa-right")},
        "content": f"""<div id="tessa-center">{PE.render_shell(doc)}</div>""",
        "extra_css": CSS + CM.CSS + PE.CSS, "extra_script": BI.PORTAL_EDITOR_JS + CM.SCRIPT + AIM.PipelineBuilderUI.SCRIPT + BI.PROMPT_BLOCK_JS})

@router.post("/new", response_class=HTMLResponse)
async def new_project(request: Request):
    doc = _new_project(request.state.user)
    _save(doc)
    await ENV["set_state"](request, doc["id"], scope="user", namespace="tessa", key="active_pid")
    models = []
    conn = AIM.connections.get_conn(doc.get("conn_id",""))
    if conn: models = await AIM.connections.list_models_async(conn)
    return HTMLResponse(await _project_view(request, doc, models))

@router.get("/load/{pid}", response_class=HTMLResponse)
async def load_project(pid: str, request: Request):
    doc = _load(pid)
    if not doc or doc.get("username") != request.state.user.username: return HTMLResponse("Not found", status_code=404)
    await ENV["set_state"](request, pid, scope="user", namespace="tessa", key="active_pid")
    conn = AIM.connections.get_conn(doc.get("conn_id",""))
    models = await AIM.connections.list_models_async(conn) if conn else []
    return HTMLResponse(await _project_view(request, doc, models))

@router.delete("/project/{pid}", response_class=HTMLResponse)
async def delete_project(pid: str, request: Request):
    user = request.state.user; doc = _load(pid)
    if doc and doc.get("username") == user.username: _dp(pid).unlink(missing_ok=True)
    active = await ENV["get_state"](request, scope="user", namespace="tessa", key="active_pid")
    if active == pid: await ENV["set_state"](request, "", scope="user", namespace="tessa", key="active_pid")
    return HTMLResponse(_proj_list_html(user.username, ""))

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
        if not new_content: new_content = msgs[idx]["content"]
        else: msgs[idx]["content"] = new_content; msgs[idx]["edited"] = True
        d["conversation"] = msgs[:idx+1]; _save(d); pid = d["id"]
        remaining = "".join(CM.render_message(m, is_me=(m.get("role")=="user"), can_delete=True, can_edit=(m.get("role")=="user")) for m in d["conversation"] if not m.get("deleted"))
        asyncio.create_task(_do_stream(user.username, {"content": new_content}, pid, skip_user_append=True))
        return HTMLResponse(f'<div id="cm-msgs-{pid}" class="cm-msgs" data-pinned="true" hx-swap-oob="outerHTML">{remaining}</div>')
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

CSS = """
#tessa-proj-list .active-item{background:var(--glass);border-left:.1rem solid var(--accent);}
.editor-shell{display:flex;flex-direction:column;height:100%;width:100%;overflow:hidden;}
#tessa-center{display:flex;flex-direction:column;height:100%;width:100%;overflow:hidden;}
"""