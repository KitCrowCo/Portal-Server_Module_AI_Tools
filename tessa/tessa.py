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

def _compress(doc):
    hist = [m for m in doc.get("conversation",[]) if not m.get("deleted")]
    if len(hist) <= 40: return doc
    old = hist[:len(hist)-20]; doc["conversation"] = hist[len(hist)-20:]
    lines = [f"{'User' if m['role']=='user' else 'AI'}: {m['content'][:100]}" for m in old]
    ex = doc.get("context_summary","")
    doc["context_summary"] = (ex + " | " if ex else "") + " | ".join(lines)
    return doc

def _new_project(user):
    cfg = _SETTINGS.get_group("defaults").load() if _SETTINGS else {}
    return {"id": f"stu_{uuid.uuid4().hex[:8]}", "username": user.username, "title": "New Project", "content": "", "conn_id": cfg.get("conn_id",""), "model": cfg.get("model",""), "model_ctx": int(cfg.get("model_ctx", 32768)), "system_prompt": cfg.get("system_prompt",""), "temperature": float(cfg.get("temperature", 0.7)), "conversation": [], "selected_files": [], "context_summary": "", "settings": {"view":"edit","font":"mono","wrap":True}, "created": datetime.utcnow().isoformat(), "modified": datetime.utcnow().isoformat()}

async def _stream(conn, messages, model, num_ctx, think=False, temperature=0.7):
    try:
        async for text, thinking in AIM.connections.stream_llm(conn, messages, model, think=think, num_ctx=num_ctx, num_predict=4096, temperature=temperature):
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
    IM.scripts["tessa_git_action"] = [_h_git_action]
    IM.scripts.update({"tessa_bottom_shadow_wiki":[_h_bottom_shadow_wiki], "tessa_bottom_shadow_kg":[_h_bottom_shadow_kg], "tessa_bottom_git":[_h_bottom_git], "tessa_bottom_git_link":[_h_bottom_git_link], "tessa_bottom_git_create_ws":[_h_bottom_git_create_ws], "tessa_doc_temp": [_h_doc_temp], "tessa_stop": [_h_stop]})
    IM.scripts.update({"tessa_bottom_pipeline_form": [_h_bottom_pipeline_form], "tessa_bottom_pipeline_run": [_h_bottom_pipeline_run]})
    print("[tessa] ready")

async def _handle_submit(request, payload, imr):
    pid = payload.get("cid","").strip(); content = payload.get("content","").strip()
    if not pid or not content: return imr
    imr.raw(CM.working_html(pid, {"type":"tessa_stop","cid":pid,"lvl":2}))
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
            async for text, thinking, done, err in _stream(conn, _build_messages(doc, content, files_txt), model, num_ctx, think, doc.get("temperature", 0.7)):
                if _STOP.pop(pid, False): break
                if err: await _err(err, retry_mid=user_msg["id"] if user_msg else None); return
                if text: full += text
                if thinking: tb += thinking
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
    pid = doc["id"]; cid = doc.get("conn_id",""); mdl = doc.get("model",""); ctx = doc.get("model_ctx",32768); temp = doc.get("temperature", 0.7)
    c_opts = AIM.connections.conn_opts_html(cid) or '<option value="">No connections</option>'
    m_opts = "".join(f'<option value="{m}" {"selected" if m==mdl else ""}>{m}</option>' for m in models) or AIM.connections.model_opts_html(cid, mdl)
    return f"""<div style="display:flex;align-items:center;gap:.4rem;height:100%;padding:0 .2rem;overflow:hidden;">
        <select class="module-select" style="font-size:.7rem;max-width:8rem;flex-shrink:0" name="value" hx-post="/im/in" hx-vals='{{"type":"tessa_doc_conn","branch":"{pid}","lvl":2}}' hx-trigger="change" hx-target="#tessa-model-wrap" hx-swap="innerHTML" hx-include="this">{c_opts}</select>
        <div id="tessa-model-wrap" style="flex-shrink:0"><select class="module-select" style="font-size:.7rem;max-width:11rem" name="value" hx-post="/im/in" hx-vals='{{"type":"tessa_doc_model","branch":"{pid}","lvl":2}}' hx-trigger="change" hx-include="this" hx-swap="none">{m_opts}</select></div>
        <label style="font-size:.6rem;color:var(--text_muted);white-space:nowrap;flex-shrink:0">ctx <input type="number" name="value" value="{ctx}" min="512" max="262144" class="module-select" style="width:5rem;font-size:.6rem;padding:.2rem .2rem" hx-post="/im/in" hx-vals='{{"type":"tessa_doc_ctx","branch":"{pid}","lvl":2}}' hx-trigger="change" hx-include="this" hx-swap="none"></label>
        <label style="font-size:.6rem;color:var(--text_muted);white-space:nowrap;flex-shrink:0">temp <input type="number" name="value" value="{temp}" min="0" max="2" step="any" class="module-select" style="width:4rem;font-size:.6rem;padding:.2rem .2rem" hx-post="/im/in" hx-vals='{{"type":"tessa_doc_temp","branch":"{pid}","lvl":2}}' hx-trigger="change" hx-include="this" hx-swap="none"></label>
        <span id="tessa-prefix-warn-{pid}" style="font-size:.6rem;color:#ffaa44"></span>
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

def _left_bottom_html(doc): return _kg_html(doc) + PB.panel_html(doc["id"], include_modal_slot=False)

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
    return (PE.render_shell(doc) + f"""<div id="tessa-conn-bar-content" hx-swap-oob="outerHTML">{_conn_bar_html(doc, AIM.connections.list_conns(), models or [])}</div><div id="tessa-chat-area" hx-swap-oob="outerHTML"><div id="tessa-chat-area" style="height:100%;overflow:hidden">{CM.shell(doc["id"], messages=doc.get("conversation",[]), viewer_name=username, is_working=is_working, stop_intent={"type":"tessa_stop","cid":doc["id"],"lvl":2} if is_working else "")}</div></div><div id="tessa-proj-list" hx-swap-oob="innerHTML">{_proj_list_html(username, doc["id"])}</div><div id="tessa-left-bottom" hx-swap-oob="innerHTML">{_left_bottom_html(doc)}</div>""")

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
                     "right": UI.toolbar(side="right", content=chat, size="22rem", overlay=False, start_open=True, resizable=True, nesting_level=2, id="tessa-right"),
                     "bottom": UI.toolbar(side="bottom", content=_bottom_bar_html(doc), size="16rem", overlay=False, start_open=False, resizable=True, id="tessa-bottom", nesting_level=2)},
        "content": f"""<div id="tessa-center">{PE.render_shell(doc)}</div>{PB.modal_slot_html(doc["id"])}""",
        "extra_css": CSS + CM.CSS + PE.CSS, "extra_script": BI.PORTAL_EDITOR_JS + CM.SCRIPT + BI.PROMPT_BLOCK_JS})

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

def _prefix_warn_html(pid, conn, field, has_history):
    if not (conn and has_history and AIM.connections.is_prefix_breaking_change(conn, field)):
        return f'<span id="tessa-prefix-warn-{pid}" hx-swap-oob="outerHTML" style="font-size:.6rem;color:#ffaa44"></span>'
    return f'<span id="tessa-prefix-warn-{pid}" hx-swap-oob="outerHTML" style="font-size:.6rem;color:#ffaa44" title="This resets the connection\'s cached prompt prefix for this conversation - the next message reprocesses the full conversation instead of resuming.">&#x26A0; prefix reset on next message</span>'

async def _h_doc_conn(request, payload, imr):
    pid = payload.get("branch",""); doc = _load(pid)
    if not doc: return imr
    doc["conn_id"] = payload.get("value",""); _save(doc)
    conn = AIM.connections.get_conn(doc["conn_id"]); models = await AIM.connections.list_models_async(conn) if conn else []
    cur = doc.get("model",""); opts = "".join(f'<option value="{m}" {"selected" if m==cur else ""}>{m}</option>' for m in models) or '<option value="">No models</option>'
    imr.oob(f"""<select class="module-select" style="font-size:.7rem; max-width:11rem" name="value" hx-post="/im/in" hx-vals='{{"type":"tessa_doc_model","branch":"{pid}","lvl":2}}' hx-trigger="change" hx-include="this" hx-swap="none">{opts}</select>""", "tessa-model-wrap")
    imr.raw(_prefix_warn_html(pid, conn, "model", bool(doc.get("conversation"))))
    return imr

async def _h_doc_model(request, payload, imr):
    pid = payload.get("branch",""); doc = _load(pid)
    if doc: doc["model"] = payload.get("value",""); _save(doc)
    conn = AIM.connections.get_conn(doc.get("conn_id","")) if doc else None
    imr.raw(_prefix_warn_html(pid, conn, "model", bool(doc.get("conversation")) if doc else False))
    return imr

async def _h_doc_ctx(request, payload, imr):
    pid = payload.get("branch",""); doc = _load(pid)
    if doc: doc["model_ctx"] = max(512, int(payload.get("value",32768) or 32768)); _save(doc)
    conn = AIM.connections.get_conn(doc.get("conn_id","")) if doc else None
    imr.raw(_prefix_warn_html(pid, conn, "num_ctx", bool(doc.get("conversation")) if doc else False))
    return imr

async def _h_doc_temp(request, payload, imr):
    doc = _load(payload.get("branch",""))
    if doc: doc["temperature"] = max(0.0, min(float(payload.get("value", 0.7) or 0.7), 2.0)); _save(doc)
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

async def _h_stop(request, payload, imr):
    sid = payload.get("cid","")
    if sid: _STOP_FLAGS[sid] = True
    return imr

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

def _bottom_bar_html(doc):
    pid = doc["id"]
    vals = lambda action: json.dumps({"type": f"tessa_bottom_{action}", "lvl": 2, "pid": pid})
    return f"""<div style="display:flex;flex-direction:column;height:100%;overflow:hidden">
                   <div style="display:flex;gap:.3rem;padding:.3rem .5rem;border-bottom:var(--border-thick) solid var(--border);flex-shrink:0">
                       <button class="cm-qbtn" hx-post="/im/in" hx-target="body" hx-swap="none" hx-vals='{vals("shadow_wiki")}'>Shadow (Wiki)</button>
                       <button class="cm-qbtn" hx-post="/im/in" hx-target="body" hx-swap="none" hx-vals='{vals("shadow_kg")}'>Shadow (Knowledge)</button>
                       <button class="cm-qbtn" hx-post="/im/in" hx-target="body" hx-swap="none" hx-vals='{vals("git")}'>Git Diff</button>
                       <button class="cm-qbtn" hx-post="/im/in" hx-target="body" hx-swap="none" hx-vals='{vals("pipeline_form")}'>Run Pipeline on Project</button>
                       <button class="cm-qbtn" hx-post="/im/in" hx-target="body" hx-swap="none" hx-vals='{json.dumps({"type":"tessa_bottom_pipeline_run","lvl":2,"pid":pid})}'>Pipeline Builder</button>
                   </div>
                   <div id="tessa-bottom-content" style="flex:1;overflow-y:auto;padding:.4rem">{_shadow_rows_html()}</div>
               </div>"""

async def _h_bottom_shadow_wiki(request, payload, imr):
    return imr.oob(f'<div style="font-size:.6rem;color:var(--text_muted);text-transform:uppercase;padding:.2rem 0">Wiki</div><div id="shadow-list-wiki">{BI.shadow_review_html(_shadow_store_for(COMMON_ROOT), "tessa_shadow_action", {"scope":"wiki"}, list_id="shadow-list-wiki")}</div>', "tessa-bottom-content")

async def _h_bottom_shadow_kg(request, payload, imr):
    return imr.oob(f'<div style="font-size:.6rem;color:var(--text_muted);text-transform:uppercase;padding:.2rem 0">Knowledge</div><div id="shadow-list-kg">{BI.shadow_review_html(_shadow_store_for(KG_DIR), "tessa_shadow_action", {"scope":"kg"}, list_id="shadow-list-kg")}</div>', "tessa-bottom-content")

async def _h_bottom_git(request, payload, imr):
    pid = payload.get("pid","")
    doc = _load(pid)
    gm = ENV["tools"]["git_manager"]
    if not doc: return imr.oob("Project not found", "tessa-bottom-content")
    if not doc.get("git_project_id"):
        opts = "".join(f'<option value="{p["_id"]}">{p["label"]}</option>' for p in gm.list_projects())
        return imr.oob(f"""<div style="font-size:.8rem"><div style="margin-bottom:.4rem">Link this project to a Git Manager project to review AI-made code changes here.</div>
                                <form hx-post="/im/in" hx-target="body" hx-swap="none" style="display:flex;gap:.4rem">
                                    <input type="hidden" name="type" value="tessa_bottom_git_link"><input type="hidden" name="lvl" value="2"><input type="hidden" name="pid" value="{pid}">
                                    <select name="git_project_id" class="module-select">{opts or '<option value="">No git projects</option>'}</select>
                                    <button type="submit" class="button">Link</button>
                                </form></div>""", "tessa-bottom-content")
    if not doc.get("git_workspace_id"):
        return imr.oob(f"""<div style="font-size:.8rem"><div style="margin-bottom:.4rem">Linked. No AI workspace branch yet.</div>
                                <button class="button" hx-post="/im/in" hx-target="body" hx-swap="none" hx-vals='{json.dumps({"type":"tessa_bottom_git_create_ws","lvl":2,"pid":pid})}'>Create AI Workspace</button></div>""", "tessa-bottom-content")
    backend = gm.GitWorkspaceDiffBackend(doc["git_workspace_id"])
    return imr.oob(f'<div id="shadow-list-git">{BI.shadow_review_html(backend, "tessa_git_action", {"pid":pid}, list_id="shadow-list-git")}</div>', "tessa-bottom-content")

async def _h_bottom_git_link(request, payload, imr):
    doc = _load(payload.get("pid",""))
    if doc: doc["git_project_id"] = payload.get("git_project_id",""); _save(doc)
    return await _h_bottom_git(request, payload, imr)

async def _h_bottom_git_create_ws(request, payload, imr):
    doc = _load(payload.get("pid",""))
    if not doc or not doc.get("git_project_id"): return imr.oob("No git project linked", "tessa-bottom-content")
    gm = ENV["tools"]["git_manager"]
    ws, msg = gm.create_workspace(doc["git_project_id"], f"tessa-{doc['id']}")
    if ws: doc["git_workspace_id"] = ws["id"]; _save(doc)
    return await _h_bottom_git(request, payload, imr)

async def _h_git_action(request, payload, imr):
    pid, action, path = payload.get("pid",""), payload.get("action",""), payload.get("path","")
    doc = _load(pid)
    if not doc or not doc.get("git_workspace_id"): return imr
    backend = ENV["tools"]["git_manager"].GitWorkspaceDiffBackend(doc["git_workspace_id"])
    if action == "diff":
        imr.oob(f'<pre style="white-space:pre-wrap;margin:0;font-size:.7rem">{_esc(backend.diff(path))}</pre>', payload.get("diff_target",""))
        return imr
    if action == "accept": backend.accept(path)
    elif action == "reject": backend.reject(path)
    imr.oob(BI.shadow_review_html(backend, "tessa_git_action", {"pid":pid}, list_id="shadow-list-git"), "shadow-list-git", swap="innerHTML")
    return imr

async def _h_bottom_pipeline_form(request, payload, imr):
    pid = payload.get("pid","")
    doc = _load(pid)
    if not doc: return imr.oob("Project not found", "tessa-bottom-content")
    pl_opts = "".join(f'<option value="{p["id"]}">{_esc(p.get("name",p["id"]))}</option>' for p in AIM.engine.list_pipelines())
    return imr.oob(f"""<form hx-post="/im/in" hx-target="body" hx-swap="none" style="display:flex;gap:.4rem;align-items:flex-end;flex-wrap:wrap;font-size:.8rem">
                            <input type="hidden" name="type" value="tessa_bottom_pipeline_run"><input type="hidden" name="lvl" value="2"><input type="hidden" name="pid" value="{pid}">
                            <label style="flex:1;min-width:12rem;color:var(--text_muted)">Run pipeline against this project<select name="pipeline_id" class="module-select"><option value="">-- select --</option>{pl_opts}</select></label>
                            <label style="width:8rem;color:var(--text_muted)">Result Key<input type="text" name="result_key" value="text" class="module-select"></label>
                            <button type="submit" class="button">Run</button>
                        </form>
                        <div id="tessa-pipeline-out" style="margin-top:.5rem;font-size:.8rem;white-space:pre-wrap;font-family:var(--font-mono)"></div>""", "tessa-bottom-content")



async def _h_bottom_pipeline_run(request, payload, imr):
    pid = payload.get("pid","")
    doc = _load(pid)
    if not doc: return imr.oob("Project not found", "tessa-pipeline-out")
    plid = payload.get("pipeline_id","")
    if not plid: return imr.oob('<span style="color:#ff5f5f">Pick a pipeline first</span>', "tessa-pipeline-out")
    result_key = payload.get("result_key","text") or "text"
    convo_text = "\n\n".join(f"{'User' if m.get('role')=='user' else 'AI'}: {m.get('content','')}" for m in doc.get("conversation",[]) if not m.get("deleted"))
    job_id, err = AIM.engine.submit(request.state.user.username, kind="id", pipeline_id=plid, inputs={"document": doc.get("content",""), "conversation": convo_text, "input": doc.get("content","")})
    if err: return imr.oob(f'<span style="color:#ff5f5f">{_esc(err)}</span>', "tessa-pipeline-out")
    imr.oob('<span style="color:var(--text_muted)">Running\u2026</span>', "tessa-pipeline-out")
    asyncio.create_task(_watch_pipeline_run(request.state.user.username, job_id, result_key))
    return imr

async def _watch_pipeline_run(username, job_id, result_key):
    """Generic poll-to-completion for any pipeline run from Tessa - the pipeline itself decides what actually happens (handoff, summarize, translate, whatever), this just runs it and shows the declared result key."""
    job = None
    while True:
        await asyncio.sleep(1.0)
        job = AIM.engine.load_job(job_id)
        if not job or job["status"] in ("done","error","stopped","interrupted"): break
    if not job or job["status"] != "done":
        await WS.send_personal_message(f'<div id="tessa-pipeline-out" hx-swap-oob="innerHTML"><span style="color:#ff5f5f">Pipeline {job["status"] if job else "lost"}</span></div>', username)
        return
    full = str(job["data"].get(result_key,""))
    copy_btn = """<button type="button" class="ui-btn" style="margin-top:.4rem" onclick="cmCopyText(document.getElementById('tessa-pipeline-out').dataset.raw||'')">Copy Result</button>"""
    await WS.send_personal_message(f'<div id="tessa-pipeline-out" hx-swap-oob="innerHTML" data-raw="{_esc(full)}">{_esc(full) or "(no output under that result key)"}{copy_btn}</div>', username)

CSS = """
#tessa-proj-list .active-item{background:var(--glass);border-left:.1rem solid var(--accent);}
.editor-shell{display:flex;flex-direction:column;height:100%;width:100%;overflow:hidden;}
#tessa-center{display:flex;flex-direction:column;height:100%;width:100%;overflow:hidden;}
"""