# /modules/ai_tools/athena/athena.py
"""
Athena - Managed AI chat. Thin front end over ai_manager's connections/pipeline layer - not an AI implementation of its own.
Admin defines "capabilities" (labeled bundles of connection+model+system prompt+optional knowledge base+optional pipeline) as individually addable/editable blocks; users only ever pick a capability by its label, never a raw model name.
Capability resolution happens live at send-time from admin config, not snapshotted into the conversation, so an admin fixing a capability's settings takes effect on existing conversations immediately.
"""
import json, uuid, re, asyncio, base64
from pathlib import Path
from datetime import datetime
from fastapi import APIRouter, Request, HTTPException
from fastapi.responses import HTMLResponse, Response

TOOL_META = {"label": "Athena", "group": "chat", "icon": "&#x1F989;", "description": "AI chat", "singleton": True}

router = APIRouter(redirect_slashes=False)

ENV: dict = {}
_P = "/module/ai_tools/athena"
DATA_DIR = Path("./data/ai_tools/athena")

UI = None
WS = None
IM = None
CM = None
AIM = None
BI = None
cfg = {}

DEFAULT_CAP = {"id":"standard", "label":"Standard", "conn_id":"", "model":"", "system_prompt":"You are a helpful, professional assistant.", "think": False, "num_predict": 8192, "model_ctx": 16384, "knowledge_enabled":False, "knowledge_conn_id":"", "flow_pipeline_id":"", "flow_result_key":"text", "allowed_roles": [], "allowed_users": []}
_ACTIVE_TASKS:dict = {}  # sid -> asyncio.Task, guards against a second submit/retry/capability-switch racing an in-flight generation for the same conversation

def _u(*p): return "/".join(s.strip("/") for s in [_P,*p] if s)
def _iv(intent_type, **extra): return json.dumps({"type": intent_type, "lvl": 2, **extra})

def get_capability_id_options(values=None):
    caps = (values or {}).get("capabilities") or []
    return [(c.get("id",""), c.get("label",c.get("id",""))) for c in caps] or [("standard","Standard")]

def init_tool(env:dict, prefix:str):
    global ENV, _P, UI, WS, IM, CM, AIM, BI, cfg
    ENV = env
    _P = prefix.rstrip("/")
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    (DATA_DIR/"conversations").mkdir(exist_ok=True)
    UI=ENV["templates"].env.globals.get("UI")
    WS=ENV["ws"]
    BI = ENV["tools"]["built_ins"]
    IM=ENV["InterfaceManager"](nesting_level=2, db_path="ai_tools/athena/im_registry.db")
    AIM = ENV["tools"]["ai_manager"]
    cfg = BI.SettingsPanel("Athena Settings", [
            BI.SettingsGroup("general", "General", [
                BI.SettingField("title", "Title", "text", "Athena"),
                BI.SettingField("capabilities", "Capabilities (managed below - not edited here)", "json", default=[dict(DEFAULT_CAP)]),
                BI.SettingField("default_capability_id", "Default Capability for New Conversations", "select", options=get_capability_id_options),
                BI.SettingField("user_input_limit", "User Input Limit (tokens)", "number", 8000),
                BI.SettingField("temperature", "Temperature", "number", 0.3),
                BI.SettingField("num_predict", "Max Response Tokens", "number", 8192),
                BI.SettingField("allow_files", "Allow File Attachments", "checkbox", True),
                BI.SettingField("user_overrides", "Per-User Default Capability", "json", default={}, hint='{"username": "capability_id"} - overrides default_capability_id for specific people.')
            ], json_path="data/settings/athena.json")])
    CM=ENV["tools"]["built_ins"].ChatManager(namespace="athena", base_url=_u(), view_style="bubble", stream_toggle=True, think_toggle=False, stop_enabled=True, show_export=False, pin_enabled=True, allow_edit=True, allow_delete=True, allow_copy=True, show_info=True, markdown_mode="standard", branch_id=IM.branch_id, nesting_level=2, action_intent_prefix="athena")
    IM.scripts.update({"submit": [_handle_submit],
                       "athena_new": [_h_new],
                       "athena_load": [_h_load],
                       "athena_conv_delete": [_h_conv_delete],
                       "athena_conv_rename_form": [_h_conv_rename_form],
                       "athena_conv_rename": [_h_conv_rename],
                       "athena_folder_new_form": [_h_folder_new_form],
                       "athena_folder_cancel": [_h_folder_cancel],
                       "athena_folder_create": [_h_folder_create],
                       "athena_folder_delete": [_h_folder_delete],
                       "athena_folder_rename_form": [_h_folder_rename_form],
                       "athena_folder_rename": [_h_folder_rename],
                       "athena_folder_assign": [_h_folder_assign],
                       "athena_msg_delete": [_h_msg_delete],
                       "athena_msg_edit_form": [_h_msg_edit_form],
                       "athena_msg_edit_save": [_h_msg_edit_save],
                       "athena_msg_cancel_edit": [_h_msg_cancel_edit],
                       "athena_msg_retry": [_h_msg_retry],
                       "athena_msg_retry_send": [_h_msg_retry_send],
                       "athena_upload": [_h_upload],
                       "athena_delete_file": [_h_delete_file],
                       "athena_capability_change": [_h_capability_change],
                       "athena_conv_settings_open": [_h_conv_settings_open],
                       "athena_conv_settings_save": [_h_conv_settings_save],
                       "athena_import_form": [_h_import_form],
                       "athena_import": [_h_import],
                       "athena_admin_save": [_h_admin_save],
                       "athena_cap_add": [_h_cap_add],
                       "athena_cap_delete": [_h_cap_delete],
                       "athena_cap_save": [_h_cap_save],
                       "athena_cap_to_pipeline": [_h_cap_to_pipeline],
                       "athena_stop": [_h_stop],
                       "athena_cap_conn_change": [_h_cap_conn_change]})
    print(f"[athena] ready")

_STOP_FLAGS:dict = {}
_ACTIVE_STREAMS:set = set()
_STREAM_BUFFERS:dict = {}

def _cp(cid): return DATA_DIR/"conversations"/f"{Path(cid).name}.json"
def _esc(s): return str(s).replace("&","&amp;").replace("<","&lt;").replace(">","&gt;").replace('"',"&quot;")
def _tok(t): return max(1,len(str(t))//4)

def _load_conv(cid):
    p = _cp(cid)
    return json.loads(p.read_text()) if p.exists() else None

def _save_conv(c): c["modified"]=datetime.utcnow().isoformat(); _cp(c["id"]).write_text(json.dumps(c,indent=2))
def _list_convs(u): return [c for c in (json.loads(p.read_text()) for p in sorted((DATA_DIR/"conversations").glob("*.json"), key=lambda x:x.stat().st_mtime, reverse=True)) if c.get("username") == u]
def _del_conv(cid): p=_cp(cid); p.unlink() if p.exists() else None
def _org(u): p=DATA_DIR/f"org_{u}.json"; return json.loads(p.read_text()) if p.exists() else {"folders":{},"conv_folders":{}}
def _save_org(u,o): (DATA_DIR/f"org_{u}.json").write_text(json.dumps(o, indent=2))

def _think_effort(cap: dict) -> str:
    """Normalizes legacy boolean 'think' values from before effort-level support existed, alongside the current '', 'low', 'medium', 'high' string format."""
    v = cap.get("think", "")
    if v is True: return "medium"
    if v is False or v is None: return ""
    return v

def _default_capability_id(user):
    allowed = _visible_capabilities(user)
    override = cfg.get("user_overrides", {}).get(user.username)
    if override and any(c["id"] == override for c in allowed): return override
    default = cfg.get("default_capability_id")
    if default and any(c["id"] == default for c in allowed): return default
    return allowed[0]["id"] if allowed else DEFAULT_CAP["id"]

def _new_conv(user): return {"id": f"ath_{uuid.uuid4().hex[:8]}", "user_id": str(user.id), "username": user.username, "user_display": user.username, "title": "New Chat", "capability_id": _default_capability_id(user), "system_prompt_override": "", "model_ctx_override": None, "messages": [], "context_summary": "", "attached_files": [], "created": datetime.utcnow().isoformat(), "modified": datetime.utcnow().isoformat()}

def _resolve_capability(conv) -> dict:
    caps = cfg.get("capabilities", []) or [DEFAULT_CAP]
    return next((c for c in caps if c.get("id") == conv.get("capability_id")), None) or caps[0]

def _capability_allowed(cap, user) -> bool:
    """Empty allowed_roles AND empty allowed_users means everyone - restriction only applies once at least one is set. Admins always pass, matching the pattern used elsewhere on this platform (e.g. module access)."""
    if getattr(user, "role", "") == "admin": return True
    roles, users = cap.get("allowed_roles") or [], cap.get("allowed_users") or []
    if not roles and not users: return True
    return getattr(user, "role", "") in roles or getattr(user, "username", "") in users

def _visible_capabilities(user): return [c for c in (cfg.get("capabilities") or [DEFAULT_CAP]) if _capability_allowed(c, user)]

def _conv_ctx_info(conv):
    if not conv: return ""
    cap = _resolve_capability(conv)
    sys_p = conv.get("system_prompt_override") or cap.get("system_prompt","")
    sys_tok = _tok(sys_p)
    msgs = [m for m in conv.get("messages",[]) if not m.get("deleted")]
    msg_tok = sum(_tok(m.get("content","")) for m in msgs)
    total = sys_tok + msg_tok
    ctx = conv.get("model_ctx_override") or cap.get("model_ctx", 16384)
    pct = min(total/max(ctx,1)*100, 100)
    col = "#00ffa2" if pct<60 else "#ffcc00" if pct<80 else "#ff9944" if pct<95 else "#ff5f5f"
    return (f"""<div style="padding:.3rem .5rem;font-size:.62rem;color:var(--text_muted);border-top:var(--border-thick) solid var(--border);flex-shrink:0">
                    <div style="display:flex;justify-content:space-between"><span>sys {sys_tok:,}t + msgs {msg_tok:,}t = {total:,}t</span><span style="color:{col}">{pct:.0f}% of {ctx//1000}k ({_esc(cap.get('label',''))})</span></div>
                </div>""")

def _find_msg(username, mid):
    for c in _list_convs(username):
        conv=_load_conv(c.get("id",""))
        if not conv: continue
        for i,m in enumerate(conv.get("messages",[])):
            if m.get("id")==mid: return conv,i,m
    return None, None, None

def _attach_content(conv):
    text_parts=[]
    images=[]
    if not cfg.get("allow_files", True): return text_parts, images
    for f in conv.get("attached_files",[]):
        p=Path(f["path"])
        if not p.exists(): continue
        ext=f.get("ext","").lower()
        if ext in (".png",".jpg",".jpeg",".webp",".gif"): images.append(base64.b64encode(p.read_bytes()).decode())
        else:
            text = BI.extract_file_text(p)
            if text: text_parts.append(f"[File: {f['name']}]\n{text}")
    return text_parts,images

def _uploads_dir(cid): d=DATA_DIR/"uploads"/cid; d.mkdir(parents=True,exist_ok=True); return d

# --- Ollama streaming ---

async def _stream_ollama(conn, msgs, model, ctx, think="", images=None, temperature=0.7, num_predict=8192, top_k=40, top_p=0.8):
    if images and msgs: msgs[-1]["images"] = images
    async for text, thinking in AIM.connections.stream_llm(conn, msgs, model, think, temperature=temperature, num_ctx=ctx, num_predict=num_predict, top_k=top_k, top_p=top_p): yield text, thinking, False, None
    # async for text, thinking, done, err in _stream_ollama(conn, msgs, model, ctx, think, images, temperature, num_predict, top_k, top_p): yield text, thinking, False, None
    yield "", "", True, None

def _build_msgs(conv, user_msg, knowledge_context=""):
    cap = _resolve_capability(conv)
    ctx = conv.get("model_ctx_override") or cap.get("model_ctx", 16384)
    budget = int(ctx * 0.82)
    sys_p = conv.get("system_prompt_override") or cap.get("system_prompt", "")
    summary = conv.get("context_summary","").strip()
    out = []
    sys_parts = [sys_p] if sys_p else []
    if knowledge_context: sys_parts.append(knowledge_context)
    if summary: sys_parts.append(f"[Prior context]\n{summary}")
    if sys_parts: out.append({"role":"system","content":"\n\n---\n\n".join(sys_parts)})
    sys_tok = sum(_tok(m["content"]) for m in out)
    available = budget - sys_tok - _tok(user_msg) - 256
    if available < 100: raise ValueError(f"System prompt + knowledge context fills context window ({sys_tok}t sys, {_tok(user_msg)}t input, {budget}t budget)")
    history = [m for m in conv.get("messages",[]) if not m.get("deleted")]
    recent = []
    used = 0
    truncated = 0
    for m in reversed(history):
        t = _tok(m.get("content",""))
        if used + t > available: truncated += 1; continue
        recent.insert(0, {"role":m["role"], "content":m["content"]})
        used += t
    out.extend(recent)
    out.append({"role":"user", "content":user_msg})
    return out, truncated

# --- Submit + Stream ---

async def _handle_submit(request, payload:dict, imr):
    sid=payload.get("cid","").strip()
    content=payload.get("content","").strip()
    if not sid or not content: return imr
    if sid in _ACTIVE_TASKS and not _ACTIVE_TASKS[sid].done(): return imr.raw('<span style="color:#ffaa44;font-size:.7rem">&#x26A0; Still generating - wait for it to finish or click Stop first.</span>')
    user = request.state.user
    conv = _load_conv(sid)
    if not conv or conv.get("username") != user.username: return imr
    user_msg = {"id":uuid.uuid4().hex[:8],"role":"user","content":content,"user_name":conv.get("user_display",user.username),"timestamp":datetime.utcnow().isoformat()}
    conv["messages"].append(user_msg)
    _save_conv(conv)
    imr.oob(CM.render_message(user_msg, is_me=True, can_delete=True, can_edit=True), f"cm-msgs-{sid}", swap="beforeend")
    imr.oob(_left(user.username, sid), "ath-left", swap="innerHTML")
    imr.raw(CM.working_html(sid, {"type":"athena_stop","cid":sid,"lvl":2}))
    imr.raw(f"""<textarea id="cm-in-{sid}" name="content" class="cm-input" placeholder="Type a message\u2026 (Ctrl+Enter)" spellcheck="true" hx-swap-oob="outerHTML"></textarea>""")
    cap = _resolve_capability(conv)
    target = _do_stream_pipeline if cap.get("flow_pipeline_id") else _do_stream
    _ACTIVE_TASKS[sid] = asyncio.create_task(target(user.username, payload, sid, skip_user_append=True))
    return imr

async def _do_stream(username: str, payload: dict, sid: str, skip_user_append=False):
    content = payload.get("content","").strip()
    _STREAM_BUFFERS[sid] = {"full":"", "thinking":"", "done":False, "error":None, "username":username}
    async def _ws(html): await WS.send_personal_message(html, username); await asyncio.sleep(0.01)
    async def _err(msg):
        _STREAM_BUFFERS[sid].update({"error":msg,"done":True})
        await _ws(f'<div id="cm-msgs-{sid}" hx-swap-oob="beforeend"><div style="color:#ff5f5f; font-size:.8rem; padding:.2rem .2rem">&#x26A0; {_esc(msg)}</div></div>{CM.working_hide_html(sid)}')
    full = ""
    tb = ""
    try:
        conv = _load_conv(sid)
        if not conv: await _err("Conversation not found."); return
        cap = _resolve_capability(conv)
        conn = AIM.connections.get_conn(cap.get("conn_id",""))
        num_ctx = conv.get("model_ctx_override") or cap.get("model_ctx", 16384)
        if not conn: await _err(f"No connection configured for capability '{cap.get('label','')}'. Contact your admin."); return
        model = cap.get("model","")
        if not model: await _err(f"No model configured for capability '{cap.get('label','')}'. Contact your admin."); return
        max_input_tok = int(num_ctx * 0.65)
        if _tok(content) > max_input_tok: content = content[:max_input_tok * 4] + f"\n\n[Input was truncated: original length exceeded {max_input_tok} token limit for {num_ctx} context window]"
        knowledge_context = ""
        if cap.get("knowledge_enabled") and cap.get("knowledge_conn_id"):
            kg_conn = AIM.connections.get_conn(cap["knowledge_conn_id"], conn_type="lightrag")
            if not kg_conn:
                await _ws(f'<div id="cm-msgs-{sid}" hx-swap-oob="beforeend"><div style="font-size:.7rem;color:#ffaa44;padding:.2rem .4rem;border-left:var(--border-thick) solid #ffaa44">&#x26A0; Knowledge base enabled but connection is missing/deleted - answering without company knowledge.</div></div>')
            else:
                kg_result = await AIM.connections.lightrag_query_cached(kg_conn, content, "hybrid")
                if kg_result.get("error"):
                    await _ws(f'<div id="cm-msgs-{sid}" hx-swap-oob="beforeend"><div style="font-size:.7rem;color:#ff5f5f;padding:.2rem .4rem;border-left:var(--border-thick) solid #ff5f5f">&#x26A0; Knowledge lookup failed: {_esc(kg_result["error"][:200])} - answering without company knowledge.</div></div>')
                else:
                    kg_text = kg_result.get("response","").strip()
                    knowledge_context = f"[Company Knowledge]\n{kg_text}\n" if kg_text else ""
                    if not kg_text: await _ws(f'<div id="cm-msgs-{sid}" hx-swap-oob="beforeend"><div style="font-size:.65rem;color:var(--text_muted);padding:.1rem .4rem">(knowledge lookup returned no results for this question)</div></div>')
        try: built_msgs, truncated = _build_msgs(conv, content, knowledge_context)
        except ValueError as e: await _err(f"Context error: {e}"); return
        if truncated: await _ws(f'<div id="cm-msgs-{sid}" hx-swap-oob="beforeend"><div style="font-size:.7rem;color:#ffcc00;padding:.2rem .4rem;border-left:var(--border-thick) solid #ffcc00">&#x26A0; {truncated} older message{"s" if truncated>1 else ""} shifted out of context window.</div></div>')
        user_msg = None
        if not skip_user_append:
            user_msg = {"id":uuid.uuid4().hex[:8],"role":"user","content":content,"user_name":conv.get("user_display",username),"timestamp":datetime.utcnow().isoformat()}
            conv["messages"].append(user_msg)
            _save_conv(conv)
            await _ws(f'<div id="cm-msgs-{sid}" hx-swap-oob="beforeend">{CM.render_message(user_msg, is_me=True, can_delete=True, can_edit=True)}</div>')
        _,images = _attach_content(conv)
        _ACTIVE_STREAMS.add(sid)
        try:
            async for text, thinking, done, err in _stream_ollama(conn, built_msgs, model, num_ctx, cap.get("think", False), images=images or None, temperature=float(cfg.get("temperature", 0.7)), num_predict = int(cap.get("num_predict") or cfg.get("num_predict", 8192))):
                if _STOP_FLAGS.pop(sid, False): break
                if err: _STREAM_BUFFERS[sid]["error"] = err; await _err(f"Model error: {err}"); return
                if text: full += text
                if thinking: tb += thinking
                _STREAM_BUFFERS[sid].update({"full":full,"thinking":tb})
                think_html = (f'<details class="cm-think" open><summary>&#x1F9E0; Thinking ({len(tb)//4}t)\u2026</summary><div class="cm-think-body">{_esc(tb)}</div></details>') if tb.strip() else ""
                await _ws(f'<div id="cm-stream-{sid}" hx-swap-oob="innerHTML">{think_html}{"<div class=cm-stream-bubble>"+_esc(full)+"</div>" if full else ""}</div>')
                if done: break
        except Exception as stream_e:
            await _err(f"Connection error: {stream_e}"); return
        finally: _ACTIVE_STREAMS.discard(sid)
        _STREAM_BUFFERS[sid]["done"] = True
        if not full:
            await _ws(f'<div id="cm-stream-{sid}" hx-swap-oob="innerHTML"></div>{CM.working_hide_html(sid)}')
            if tb.strip():
                conv2 = _load_conv(sid)
                if conv2:
                    ai_msg = {"id":uuid.uuid4().hex[:8],"role":"assistant","content":"*(no visible reply - model used its whole token budget thinking)*","thinking":tb.strip(),"model":model,"timestamp":datetime.utcnow().isoformat(),"partial":True}
                    conv2["messages"].append(ai_msg)
                    _save_conv(conv2)
                    await _ws(f'<div id="cm-msgs-{sid}" hx-swap-oob="beforeend">{CM.render_message(ai_msg, is_me=False, can_delete=True, can_edit=True)}</div>')
            return
        conv2 = _load_conv(sid)
        if conv2:
            ai_msg = {"id":uuid.uuid4().hex[:8],"role":"assistant","content":full, "thinking":tb.strip() if tb.strip() else "","model":model, "timestamp":datetime.utcnow().isoformat(),"response_tokens":_tok(full)}
            conv2["messages"].append(ai_msg)
            if len(conv2["messages"]) == 2 and conv2.get("title","") in ("","New Chat"): conv2["title"] = conv2["messages"][0].get("content","")[:50]
            _save_conv(conv2)
            await _ws(f"""<div id="cm-msgs-{sid}" hx-swap-oob="beforeend">{CM.render_message(ai_msg, is_me=False, can_delete=True, can_edit=True)}</div>
                          <div id="cm-stream-{sid}" hx-swap-oob="innerHTML"></div>
                          {CM.working_hide_html(sid)}
                          <div id="ath-left" hx-swap-oob="innerHTML">{_left(username,sid)}</div>""")
    except Exception as e:
        print(f"[athena] stream error {sid}: {e}")
        if full:
            try:
                conv2 = _load_conv(sid)
                if conv2:
                    conv2["messages"].append({"id":uuid.uuid4().hex[:8],"role":"assistant","content":full,"thinking":tb.strip(),"model":"","partial":True,"timestamp":datetime.utcnow().isoformat(),"response_tokens":_tok(full)})
                    _save_conv(conv2)
            except Exception as save_err: print(f"[athena] partial save failed: {save_err}")
        await _err(f"Server error: {e}")
    finally:
        _STREAM_BUFFERS.pop(sid, None)

async def _do_stream_pipeline(username: str, payload: dict, sid: str, skip_user_append=False):
    """A capability with flow_pipeline_id set hands the whole turn to an ai_manager pipeline instead of a direct chat call.
    Live rendering rides the real pipeline:running / pipeline:stream / pipeline:error events interface_bridge.js already dispatches from engine.py's per-node pushes and steps.py's NodeContext.stream - job completion is still detected purely by polling below, never by a pushed event, since pipeline:done fires per-node rather than once for the whole job and would be misleading as a completion signal for multi-node pipelines."""
    content = payload.get("content","").strip()
    async def _ws(html): await WS.send_personal_message(html, username); await asyncio.sleep(0.01)
    async def _err(msg): await _ws(f'<div id="cm-msgs-{sid}" hx-swap-oob="beforeend"><div style="color:#ff5f5f;font-size:.8rem;padding:.2rem .2rem">&#x26A0; {_esc(msg)}</div></div>{CM.working_hide_html(sid)}')
    conv = _load_conv(sid)
    if not conv: await _err("Conversation not found."); return
    cap = _resolve_capability(conv)
    if not skip_user_append:
        user_msg = {"id":uuid.uuid4().hex[:8],"role":"user","content":content,"user_name":conv.get("user_display",username),"timestamp":datetime.utcnow().isoformat()}
        conv["messages"].append(user_msg); _save_conv(conv)
        await _ws(f'<div id="cm-msgs-{sid}" hx-swap-oob="beforeend">{CM.render_message(user_msg, is_me=True, can_delete=True, can_edit=True)}</div>')
    pid = cap.get("flow_pipeline_id","")
    job_id, err = AIM.engine.submit(username, kind="id", pipeline_id=pid, inputs={"input": content})
    if err: await _err(f"Pipeline error: {err}"); return
    await _ws(f"""<div id="cm-stream-{sid}" hx-swap-oob="innerHTML"><div data-pipeline-job="{job_id}" style="font-size:.75rem;color:var(--text_muted)">Starting pipeline\u2026</div></div>
                  <script>if(!window._athenaPipelineBound){{window._athenaPipelineBound=true;
                      document.addEventListener('pipeline:running',function(e){{var b=document.querySelector('[data-pipeline-job="'+e.detail.job_id+'"]');if(b)b.textContent='Running: '+(e.detail.name||e.detail.node||'');}});
                      document.addEventListener('pipeline:stream',function(e){{var b=document.querySelector('[data-pipeline-job="'+e.detail.job_id+'"]');if(b)b.textContent+=(e.detail.delta||'');}});
                      document.addEventListener('pipeline:error',function(e){{var b=document.querySelector('[data-pipeline-job="'+e.detail.job_id+'"]');if(b)b.textContent='Error: '+(e.detail.message||JSON.stringify(e.detail));}});
                  }}</script>""")
    job = None
    while True:
        await asyncio.sleep(1.0)
        job = AIM.engine.load_job(job_id)
        if not job or job["status"] in ("done","error","stopped","interrupted"): break
    if not job or job["status"] != "done":
        failed_node = next((n for n in (job["flow"]["nodes"] if job else []) if n.get("status") == "error"), None)
        detail = f" - node '{failed_node.get('name') or failed_node['id']}': {failed_node.get('message','')}" if failed_node else ""
        await _err(f"Pipeline {job['status'] if job else 'lost'}{detail}"); return
    result_key = cap.get("flow_result_key","text") or "text"
    full = str(job["data"].get(result_key,""))
    if not full: await _err(f"Pipeline finished but produced nothing under key '{result_key}' - check the pipeline's config with your admin."); return
    conv2 = _load_conv(sid)
    if not conv2: return
    ai_msg = {"id":uuid.uuid4().hex[:8],"role":"assistant","content":full,"model":f"pipeline:{pid}","timestamp":datetime.utcnow().isoformat()}
    conv2["messages"].append(ai_msg)
    if len(conv2["messages"]) == 2 and conv2.get("title","") in ("","New Chat"): conv2["title"] = content[:50]
    _save_conv(conv2)
    await _ws(f'<div id="cm-msgs-{sid}" hx-swap-oob="beforeend">{CM.render_message(ai_msg, is_me=False, can_delete=True, can_edit=False)}</div><div id="cm-stream-{sid}" hx-swap-oob="innerHTML"></div>{CM.working_hide_html(sid)}<div id="ath-left" hx-swap-oob="innerHTML">{_left(username,sid)}</div>')

# --- Left panel / conversation list ---

def _conv_item(c, active, org):
    cid = c.get("id","")
    title = _esc((c.get("title","") or "Untitled")[:42])
    ac = " active" if cid == active else ""
    short_title = (c.get("title","") or "Untitled")[:30]
    partial_badge = '<span style="font-size:.6rem;color:#ffaa44;margin-left:.2rem" title="Last response was partial">&#x25CC;</span>' if any(m.get("partial") for m in c.get("messages",[])) else ""
    folders = org.get("folders",{})
    folder_id = org.get("conv_folders",{}).get(cid,"") or ""
    folder_sel = ""
    if folders:
        opts = '<option value="">No folder</option>' + "".join(f'<option value="{fid}" {"selected" if fid==folder_id else ""}>{_esc(fd["name"])}</option>' for fid,fd in sorted(folders.items(),key=lambda x:x[1].get("order",0)))
        folder_sel = f"""<select class="side-folder-sel" name="value" hx-post="/im/in" hx-target="body" hx-swap="none" hx-vals='{_iv("athena_folder_assign", cid=cid)}' hx-include="this" onclick="event.stopPropagation()">{opts}</select>"""
    return f"""<div class="side-list-item{ac}" id="ath-ci-{cid}" hx-post="/im/in" hx-target="#ath-chat-area" hx-swap="innerHTML" hx-vals='{_iv("athena_load", cid=cid)}'>
                   <div style="display:flex;align-items:center;gap:.2rem; width:100%">
                       <span class="side-list-item-title" style="flex:1">{title}{partial_badge}</span>
                       <span class="side-list-actions" style="display:flex;gap:.1rem;flex-shrink:0">
                           <button class="cm-qbtn" hx-post="/im/in" hx-target="body" hx-swap="none" hx-vals='{_iv("athena_conv_rename_form", cid=cid)}' onclick="event.stopPropagation()">&#x270E;</button>
                           <button class="cm-qbtn" hx-post="/im/in" hx-target="body" hx-swap="none" hx-vals='{_iv("athena_conv_delete", cid=cid)}' hx-confirm="Delete '{short_title}'?" onclick="event.stopPropagation()">&#x2715;</button>
                       </span>
                   </div>
                   <details style="font-size:.6rem;color:var(--text_muted)" onclick="event.stopPropagation()">
                       <summary style="list-style:none;cursor:pointer;user-select:none">&#x25B8;</summary>
                       <div style="display:flex;justify-content:space-between;padding:.1rem 0">{folder_sel}</div>
                   </details>
               </div>"""

def _left(username, active=""):
    org = _org(username)
    convs = _list_convs(username)
    folders = org.get("folders", {})
    active_fid = org.get("conv_folders", {}).get(active)
    grouped = {fid: [] for fid in folders}
    ungrouped = []
    for c in convs:
        cid = c.get("id", "")
        fid = org.get("conv_folders", {}).get(cid)
        (grouped[fid] if fid and fid in grouped else ungrouped).append(c)
    folder_html = "".join(f"""<details class="side-folder" {"open" if fid==active_fid else ""}><summary class="side-folder-sum">&#x1F4C1; <span id="ath-fn-{fid}">{_esc(fd["name"])}</span><button class="cm-qbtn" hx-post="/im/in" hx-target="body" hx-swap="none" hx-vals='{_iv("athena_folder_rename_form", fid=fid)}' onclick="event.stopPropagation()">&#x270E;</button><button class="cm-qbtn" hx-post="/im/in" hx-target="body" hx-swap="none" hx-vals='{_iv("athena_folder_delete", fid=fid)}' hx-confirm="Delete folder?" style="margin-left:auto;color:#ff5f5f" onclick="event.stopPropagation()">&#x2715;</button></summary>{"".join(_conv_item(c,active,org) for c in grouped.get(fid,[]))}</details>""" for fid, fd in sorted(folders.items(), key=lambda x: x[1].get("order", 0)))
    ug_html = "".join(_conv_item(c, active, org) for c in ungrouped)
    ug_hdr = '<div class="side-ungrouped-hdr">Other</div>' if folder_html and ungrouped else ""
    active_conv = _load_conv(active) if active else None
    ctx_footer = _conv_ctx_info(active_conv)
    app_title = cfg.get("title", "Athena")
    return (f"""<div class="side-list-hdr">
                    <button class="btn-icon" hx-post="/im/in" hx-target="#ath-chat-area" hx-swap="innerHTML" hx-vals='{_iv("athena_new")}' title="New chat" style="font-size:1rem">+</button>
                    <span style="font-size:.7rem; text-transform:uppercase;letter-spacing:.05em;color:var(--text_muted);flex:1;padding:0 .3rem">{_esc(app_title)}</span>
                    <button class="btn-icon" hx-post="/im/in" hx-target="#ath-chat-area" hx-swap="innerHTML" hx-vals='{_iv("athena_import_form")}' style="font-size:.8rem" title="Import">&#x1F4E5;</button>
                    <button class="btn-icon" hx-post="/im/in" hx-target="body" hx-swap="none" hx-vals='{_iv("athena_folder_new_form")}' style="font-size:.75rem" title="New folder">&#x1F4C1;+</button>
                </div>
                <div id="ath-folder-new"></div>
                <div class="side-list">
                    {folder_html}
                    {ug_hdr}
                    {ug_html}
                </div>
                <button class="ait-rp-btn" hx-get="{_u("admin")}" hx-target="#ait-workspace" hx-swap="innerHTML">&#x2699; Admin Settings</button>{ctx_footer}""")

# --- Chat area ---

def _capability_bar_html(conv, user):
    caps = _visible_capabilities(user)
    cur = conv.get("capability_id","")
    opts = "".join(f'<option value="{_esc(c["id"])}" {"selected" if c["id"]==cur else ""}>{_esc(c.get("label",c["id"]))}</option>' for c in caps)
    return f"""<select class="module-select" style="font-size:.7rem;max-width:12rem;margin:0" name="value" hx-post="/im/in" hx-target="body" hx-swap="none"
                       hx-vals='{_iv("athena_capability_change", cid=conv["id"])}' hx-trigger="change" hx-include="this">{opts}</select>
               <span id="ath-capability-warn" style="font-size:.65rem;color:#ffaa44"></span>"""

def _file_chips_html(conv):
    sid = conv["id"]
    return "".join(f"""<span style="display:inline-flex;align-items:center;gap:.2rem;background:var(--accent_dim);border:var(--border-thick) solid var(--accent);color:var(--accent);padding:.1rem .1rem;border-radius:.3rem;font-size:.7rem; max-width:12rem">
                            <span style="overflow:hidden;text-overflow:ellipsis;white-space:nowrap" title="{_esc(f["name"])}">{_esc(f["name"][:20])}</span>
                            <button style="background:none;border:none;cursor:pointer;color:var(--accent);font-size:.8rem; padding:0; flex-shrink:0;line-height:1" hx-post="/im/in" hx-target="body" hx-swap="none" hx-vals='{_iv("athena_delete_file", cid=sid, fid=f["id"])}'>&#x2715;</button>
                        </span>""" for f in conv.get("attached_files",[]))

def _chat_html(conv, requests = None):
    sid=conv["id"]
    hdr=(f"""<span style="font-size:.8rem;font-weight:600;flex:1">{_esc(conv.get("title","Chat"))}</span>{_capability_bar_html(conv, requests.state.user)}""")
    extra_footer = ""
    if cfg.get("allow_files", True): extra_footer = (f"""<div style="display:flex;align-items:center; gap:.2rem; flex-wrap:wrap;padding-top:.1rem">
                                                             <label class="btn-icon" title="Attach file" style="cursor:pointer;font-size:.9rem;flex-shrink:0">&#x1F4CE;
                                                                 <input type="file" name="files" multiple style="display:none" accept="image/*,.csv,.txt,.md,.xlsx,.xls,.pdf" hx-post="/im/in" hx-target="body" hx-swap="none" hx-encoding="multipart/form-data" hx-trigger="change" hx-vals='{_iv("athena_upload", cid=sid)}'>
                                                             </label>
                                                             <div id="ath-files-{sid}" style="display:flex;gap:.2rem;flex-wrap:wrap;flex:1;min-width:0">{_file_chips_html(conv)}</div>
                                                             <a class="cm-qbtn" href="{_u("export",sid)}" download>&#x2B07; Export</a>
                                                         </div>""")
    buf=_STREAM_BUFFERS.get(sid)
    is_working=bool(buf and not buf.get("done"))
    shell=CM.shell(sid, messages=conv.get("messages",[]), viewer_name=conv.get("user_display",""), header_html=hdr, extra_footer=extra_footer, is_working=is_working, stop_intent={"type":"athena_stop","cid":sid,"lvl":2} if is_working else "", owns_conversation=True)
    resume=""
    if is_working:
        full=buf.get("full","")
        tb=buf.get("thinking","")
        think_html=(f"""<details class="cm-think" open><summary>&#x1F9E0; Thinking ({len(tb)//4}t)\u2026</summary><div class="cm-think-body">{_esc(tb)}</div></details>""") if tb.strip() else ""
        stream_content=think_html+(f'<div class="cm-stream-bubble">{_esc(full)}</div>' if full else "")
        resume=f'<script>document.getElementById("cm-stream-{sid}").innerHTML={json.dumps(stream_content)};</script>'
    scroll=f'<script>requestAnimationFrame(function(){{var m=document.getElementById("cm-msgs-{sid}");if(m)m.scrollTop=m.scrollHeight;}});</script>'
    return shell+resume+scroll

# --- Main route ---

@router.get("")
@router.get("/")
async def root(request:Request):
    user=request.state.user
    username=user.username
    cid=await ENV["get_state"](request,scope="user",namespace="athena",key="active_conv_id")
    conv=_load_conv(cid) if cid else None
    if not conv or conv.get("username")!=username:
        convs=_list_convs(username)
        conv=_load_conv(convs[0].get("id","")) if convs else None
    if not conv: conv=_new_conv(user); _save_conv(conv)
    await ENV["set_state"](request, conv["id"], scope="user", namespace="athena", key="active_conv_id")
    return ENV["templates"].TemplateResponse(name = "base.html", request = request, context = {"request": request,"user": user, "nesting_level": 2, "shell_id": IM.branch_id,
                                                                                               "toolbars": {"left": UI.toolbar(side="left", content=f'<div id="ath-left" style="display:flex;flex-direction:column;height:100%;overflow:hidden">{_left(username, conv["id"])}', size="16rem", overlay=False, start_open=False, id="ath-left-bar", nesting_level=2)},
                                                                                               "content":f'<div id="ath-chat-area" style="height:100%; overflow:hidden;">{_chat_html(conv,request)}</div>', "extra_css": CM.CSS, "extra_script": CM.SCRIPT})

# --- Intent handlers: conversation lifecycle ---

async def _h_new(request, payload, imr):
    user=request.state.user
    conv=_new_conv(user)
    _save_conv(conv)
    await ENV["set_state"](request,conv["id"],scope="user",namespace="athena",key="active_conv_id")
    return imr.raw(_chat_html(conv,request)+f'<div id="ath-left" hx-swap-oob="innerHTML">{_left(user.username,conv["id"])}</div>')

async def _h_load(request, payload, imr):
    user=request.state.user; cid=payload.get("cid","")
    conv=_load_conv(cid)
    if not conv or conv.get("username")!=user.username: return imr.raw("Not found")
    await ENV["set_state"](request,cid,scope="user",namespace="athena",key="active_conv_id")
    return imr.raw(_chat_html(conv,request)+f'<div id="ath-left" hx-swap-oob="innerHTML">{_left(user.username,cid)}</div>')

async def _h_conv_delete(request, payload, imr):
    user = request.state.user
    cid = payload.get("cid","")
    conv = _load_conv(cid)
    if conv and conv.get("username") == user.username: _del_conv(cid)
    org = _org(user.username)
    org.get("conv_folders",{}).pop(cid, None)
    _save_org(user.username, org)
    remaining = _list_convs(user.username)
    active = await ENV["get_state"](request, scope="user", namespace="athena", key="active_conv_id")
    if active != cid:
        imr.oob(_left(user.username, active), "ath-left", swap="innerHTML")
        return imr
    if remaining:
        next_cid = remaining[0].get("id","")
        next_conv = _load_conv(next_cid)
        if next_conv:
            await ENV["set_state"](request, next_cid, scope="user", namespace="athena", key="active_conv_id")
            imr.oob(_chat_html(next_conv, request), "ath-chat-area", swap="innerHTML")
            imr.oob(_left(user.username, next_cid), "ath-left", swap="innerHTML")
            return imr
    new_conv = _new_conv(user)
    _save_conv(new_conv)
    await ENV["set_state"](request, new_conv["id"], scope="user", namespace="athena", key="active_conv_id")
    imr.oob(_chat_html(new_conv, request), "ath-chat-area", swap="innerHTML")
    imr.oob(_left(user.username, new_conv["id"]), "ath-left", swap="innerHTML")
    return imr

async def _h_conv_rename_form(request, payload, imr):
    cid = payload.get("cid","")
    conv=_load_conv(cid)
    if not conv: return imr
    return imr.oob(f"""<div id="ath-ci-{cid}" style="padding:.25rem .4rem;display:flex;gap:.25rem"><form hx-post="/im/in" hx-target="body" hx-swap="none" style="display:flex;gap:.25rem;width:100%">
                            <input type="hidden" name="type" value="athena_conv_rename">
                            <input type="hidden" name="cid" value="{cid}">
                            <input type="hidden" name="lvl" value="2">
                            <input type="text" name="value" value="{_esc(conv.get("title",""))}" class="module-select" style="flex:1;font-size:.75rem" autofocus onclick="event.stopPropagation()">
                            <button type="submit" class="btn-icon" onclick="event.stopPropagation()">&#x2713;</button>
                            </form>
                        </div>""", f"ath-ci-{cid}", swap="outerHTML")

async def _h_conv_rename(request, payload, imr):
    user = request.state.user
    cid = payload.get("cid", "")
    conv = _load_conv(cid)
    if not conv or conv.get("username")!=user.username: return imr
    conv["title"]=payload.get("value","").strip() or conv.get("title","Chat")
    _save_conv(conv)
    return imr.oob(_left(user.username, cid), "ath-left", swap="innerHTML")

# --- Intent handlers: folders ---

async def _h_folder_new_form(request, payload, imr):
    return imr.oob(f"""<form hx-post="/im/in" hx-target="body" hx-swap="none" style="display:flex;gap:.3rem;padding:.3rem .5rem;border-bottom:var(--border-thick) solid var(--border)">
                            <input type="hidden" name="type" value="athena_folder_create"><input type="hidden" name="lvl" value="2">
                            <input type="text" name="name" class="module-select" placeholder="Folder name" style="flex:1;font-size:.75rem" required>
                            <button type="submit" class="btn-icon">&#x2713;</button>
                            <button type="button" class="btn-icon" hx-post="/im/in" hx-target="body" hx-swap="none" hx-vals='{_iv("athena_folder_cancel")}'>&#x2715;</button>
                        </form>""", "ath-folder-new", swap="innerHTML")

async def _h_folder_cancel(request, payload, imr): return imr.oob("", "ath-folder-new", swap="innerHTML")

async def _h_folder_create(request, payload, imr):
    user = request.state.user
    name=payload.get("name","").strip()
    if not name: return imr
    org=_org(user.username)
    org.setdefault("folders",{})[f"f_{uuid.uuid4().hex[:8]}"]={"name":name,"order":len(org.get("folders",{}))}
    _save_org(user.username,org)
    active=await ENV["get_state"](request, scope="user", namespace="athena", key="active_conv_id")
    imr.oob(_left(user.username, active or ""), "ath-left", swap="innerHTML")
    imr.oob("", "ath-folder-new", swap="innerHTML")
    return imr

async def _h_folder_delete(request, payload, imr):
    user = request.state.user
    fid=payload.get("fid","")
    org=_org(user.username)
    org.get("folders",{}).pop(fid,None)
    for cid,f in list(org.get("conv_folders",{}).items()):
        if f==fid: org["conv_folders"].pop(cid)
    _save_org(user.username,org)
    active=await ENV["get_state"](request,scope="user", namespace="athena", key="active_conv_id")
    return imr.oob(_left(user.username, active or ""), "ath-left", swap="innerHTML")

async def _h_folder_rename_form(request, payload, imr):
    fid = payload.get("fid","")
    return imr.oob(f"""<span id="ath-fn-{fid}" style="display:inline-flex;align-items:center;gap:.2rem">
                           <form hx-post="/im/in" hx-target="body" hx-swap="none" style="display:inline-flex;gap:.2rem" onclick="event.stopPropagation()">
                               <input type="hidden" name="type" value="athena_folder_rename">
                               <input type="hidden" name="fid" value="{fid}">
                               <input type="hidden" name="lvl" value="2">
                               <input type="text" name="name" class="module-select" style="font-size:.7rem;width:8rem" autofocus>
                               <button type="submit" class="btn-icon">&#x2713;</button>
                            </form>
                        </span>""", f"ath-fn-{fid}", swap="outerHTML")

async def _h_folder_rename(request, payload, imr):
    user = request.state.user
    name=payload.get("name","").strip()
    fid=payload.get("fid","")
    org=_org(user.username)
    if name and fid in org.get("folders",{}): org["folders"][fid]["name"]=name
    _save_org(user.username,org)
    active=await ENV["get_state"](request,scope="user",namespace="athena",key="active_conv_id")
    return imr.oob(_left(user.username,active or ""), "ath-left", swap="innerHTML")

async def _h_folder_assign(request, payload, imr):
    user = request.state.user
    cid=payload.get("cid","")
    fid=payload.get("value","")
    org=_org(user.username)
    org.setdefault("conv_folders",{})[cid]=fid if fid else None
    _save_org(user.username,org)
    active=await ENV["get_state"](request,scope="user",namespace="athena",key="active_conv_id")
    return imr.oob(_left(user.username,active or ""), "ath-left", swap="innerHTML")

# --- Intent handlers: messages ---

async def _h_msg_delete(request, payload, imr):
    mid = payload.get("id","")
    conv, idx, m = _find_msg(request.state.user.username, mid)
    if conv and m:
        conv["messages"][idx]["deleted"]=True
        _save_conv(conv)
    return imr.oob("", f"cm-msg-{mid}", swap="outerHTML")

async def _h_msg_edit_form(request, payload, imr):
    mid = payload.get("id","")
    conv, idx, m = _find_msg(request.state.user.username, mid)
    if not conv or not m: return imr
    role_cls="cm-me" if m.get("role")=="user" else "cm-other"
    avatar=CM._avatar_html(m.get("user_name","?"))
    return imr.oob(f"""<div class="cm-msg {role_cls}" id="cm-msg-{mid}" data-msg-id="{mid}">
                           {avatar}
                           <div class="cm-bwrap" style="max-width:90%">
                               <form hx-post="/im/in" hx-target="body" hx-swap="none" style="display:flex;flex-direction:column;gap:.3rem;width:100%">
                                   <input type="hidden" name="type" value="athena_msg_edit_save">
                                   <input type="hidden" name="id" value="{mid}"><input type="hidden" name="lvl" value="2">
                                   <textarea name="content" class="cm-input" style="min-height:4rem;overflow-y:auto">{_esc(m.get("content",""))}</textarea>
                                   <div style="display:flex;gap:.3rem">
                                       <button type="submit" class="button" style="font-size:.75rem;margin-top:0">Save</button>
                                       <button type="button" class="btn-icon" hx-post="/im/in" hx-target="body" hx-swap="none" hx-vals='{_iv("athena_msg_cancel_edit", id=mid)}'>Cancel</button>
                                   </div>
                               </form>
                            </div>
                        </div>""", f"cm-msg-{mid}", swap="outerHTML")

async def _h_msg_edit_save(request, payload, imr):
    mid = payload.get("id","")
    conv, idx, m = _find_msg(request.state.user.username, mid)
    if not conv or not m: return imr
    conv["messages"][idx]["content"]=payload.get("content","").strip()
    conv["messages"][idx]["edited"]=True
    _save_conv(conv)
    is_me=m.get("role")=="user"
    return imr.oob(CM.render_message(conv["messages"][idx],is_me=is_me,can_delete=True,can_edit=is_me), f"cm-msg-{mid}", swap="outerHTML")

async def _h_msg_cancel_edit(request, payload, imr):
    mid = payload.get("id","")
    conv, _, m = _find_msg(request.state.user.username, mid)
    if not conv or not m: return imr
    is_me=m.get("role")=="user"
    return imr.oob(CM.render_message(m,is_me=is_me,can_delete=True,can_edit=is_me), f"cm-msg-{mid}", swap="outerHTML")

async def _h_msg_retry(request, payload, imr):
    mid = payload.get("id","")
    conv, idx, m = _find_msg(request.state.user.username, mid)
    if not conv or not m: return imr
    role_cls="cm-me" if m.get("role")=="user" else "cm-other"
    avatar=CM._avatar_html(m.get("user_name","?"))
    return imr.oob(f"""<div class="cm-msg {role_cls}" id="cm-msg-{mid}" data-msg-id="{mid}">
                           {avatar}
                           <div class="cm-bwrap" style="max-width:90%">
                               <form hx-post="/im/in" hx-target="body" hx-swap="none" style="display:flex;flex-direction:column;gap:.3rem;width:100%">
                                   <input type="hidden" name="type" value="athena_msg_retry_send"><input type="hidden" name="id" value="{mid}">
                                   <input type="hidden" name="lvl" value="2">
                                   <textarea name="content" class="cm-input" style="min-height:4rem">{_esc(m.get("content",""))}</textarea>
                                   <div style="display:flex;gap:.3rem">
                                       <button type="submit" class="button" style="font-size:.75rem;margin-top:0">&#x21BA; Retry</button>
                                       <button type="button" class="btn-icon" hx-post="/im/in" hx-target="body" hx-swap="none" hx-vals='{_iv("athena_msg_cancel_edit", id=mid)}'>Cancel</button>
                                   </div>
                               </form>
                            </div>
                        </div>""", f"cm-msg-{mid}", swap="outerHTML")

async def _h_msg_retry_send(request, payload, imr):
    user = request.state.user
    new_content=payload.get("content","").strip()
    mid=payload.get("id","")
    conv, idx, m = _find_msg(user.username, mid)
    if not conv or not m: return imr
    conv["messages"][idx]["content"]=new_content
    conv["messages"][idx]["edited"]=True
    conv["messages"]=conv["messages"][:idx+1]
    _save_conv(conv)
    sid=conv["id"]
    remaining="".join(CM.render_message(msg,is_me=(msg.get("role")=="user"), can_delete=True, can_edit=(msg.get("role")=="user")) for msg in conv["messages"] if not msg.get("deleted"))
    cap = _resolve_capability(conv)
    target = _do_stream_pipeline if cap.get("flow_pipeline_id") else _do_stream
    _ACTIVE_TASKS[sid] = asyncio.create_task(target(user.username,{"content":new_content},sid,skip_user_append=True))
    return imr.oob(f'<div id="cm-msgs-{sid}" class="cm-msgs" data-pinned="true">{remaining}</div>', f"cm-msgs-{sid}", swap="outerHTML")

# --- Intent handlers: attachments ---

async def _h_upload(request, payload, imr):
    cid = payload.get("cid","")
    conv = _load_conv(cid)
    if not conv or conv.get("username") != request.state.user.username: return imr
    files_raw = payload.get("files")
    files = files_raw if isinstance(files_raw, list) else ([files_raw] if files_raw else [])
    up_dir = _uploads_dir(cid)
    for f in files:
        if not getattr(f, "filename", ""): continue
        fid = uuid.uuid4().hex[:8]
        ext = Path(f.filename).suffix
        save_path = up_dir / f"{fid}{ext}"
        save_path.write_bytes(await f.read())
        conv.setdefault("attached_files", []).append({"id": fid, "name": f.filename, "path": str(save_path), "ext": ext})
    _save_conv(conv)
    return imr.oob(_file_chips_html(conv), f"ath-files-{cid}", swap="innerHTML")

async def _h_delete_file(request, payload, imr):
    cid, fid = payload.get("cid",""), payload.get("fid","")
    conv = _load_conv(cid)
    if not conv or conv.get("username") != request.state.user.username: return imr
    files = conv.get("attached_files", [])
    for f in files:
        if f["id"] == fid:
            p = Path(f["path"])
            if p.exists(): p.unlink()
            break
    conv["attached_files"] = [f for f in files if f["id"] != fid]
    _save_conv(conv)
    return imr.oob(_file_chips_html(conv), f"ath-files-{cid}", swap="innerHTML")

async def _h_stop(request, payload, imr):
    sid = payload.get("cid","")
    if sid: _STOP_FLAGS[sid] = True
    return imr

# --- Intent handlers: capability + conversation settings ---

async def _h_capability_change(request, payload, imr):
    cid = payload.get("cid","")
    if cid in _ACTIVE_TASKS and not _ACTIVE_TASKS[cid].done(): return imr.oob('<span id="ath-capability-warn">&#x26A0; Wait for the current response to finish before switching capability.</span>', "ath-capability-warn", swap="outerHTML")
    conv = _load_conv(cid)
    if not conv or conv.get("username") != request.state.user.username: return imr
    target = payload.get("value","")
    if not any(c["id"] == target and _capability_allowed(c, request.state.user) for c in (cfg.get("capabilities") or [DEFAULT_CAP])): return imr
    old_cap = _resolve_capability(conv)
    conv["capability_id"] = target
    _save_conv(conv)
    new_cap = _resolve_capability(conv)
    conn = AIM.connections.get_conn(new_cap.get("conn_id",""))
    has_history = bool([m for m in conv.get("messages",[]) if not m.get("deleted")])
    changed = old_cap.get("conn_id") != new_cap.get("conn_id") or old_cap.get("model") != new_cap.get("model")
    if conn and has_history and changed and AIM.connections.is_prefix_breaking_change(conn, "model"):
        imr.oob('<span id="ath-capability-warn">&#x26A0; Switching capability resets this connection\'s cached prefix - the next reply reprocesses the full conversation.</span>', "ath-capability-warn", swap="outerHTML")
    else:
        imr.oob('<span id="ath-capability-warn"></span>', "ath-capability-warn", swap="outerHTML")
    imr.oob(_chat_html(conv, request), "ath-chat-area", swap="innerHTML")
    return imr

async def _h_conv_settings_open(request, payload, imr):
    cid = payload.get("cid","")
    conv = _load_conv(cid)
    if not conv or conv.get("username") != request.state.user.username: return imr
    cap = _resolve_capability(conv)
    return imr.raw(f"""<div style="padding:1rem;height:100%;overflow-y:auto;box-sizing:border-box;max-width:40rem;margin:0 auto">
                           <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:.8rem">
                               <h3 style="margin:0">Conversation Settings</h3>
                               <button class="close-btn" hx-post="/im/in" hx-target="#ath-chat-area" hx-swap="innerHTML" hx-vals='{_iv("athena_load", cid=cid)}'>&#x2715;</button>
                           </div>
                           <div style="font-size:.7rem;color:var(--text_muted);margin-bottom:.8rem">Currently: <b>{_esc(cap.get("label",""))}</b> - {_esc(cap.get("model","(no model configured)"))}. These fields override that capability's defaults for this conversation only.</div>
                           <form hx-post="/im/in" hx-target="#ath-chat-area" hx-swap="innerHTML" style="display:flex;flex-direction:column;gap:.7rem">
                               <input type="hidden" name="type" value="athena_conv_settings_save"><input type="hidden" name="cid" value="{cid}"><input type="hidden" name="lvl" value="2">
                               <label style="font-size:.8rem;color:var(--text_muted)">System Prompt Override (blank = use capability default)
                                   <textarea name="system_prompt_override" class="cm-input" rows="4">{_esc(conv.get("system_prompt_override",""))}</textarea>
                               </label>
                               <label style="font-size:.8rem;color:var(--text_muted)">Context Tokens Override (blank = use capability default: {cap.get("model_ctx",16384)})
                                   <input type="number" name="model_ctx_override" value="{conv.get("model_ctx_override") or ""}" class="module-select">
                               </label>
                               <button type="submit" class="button">Save</button>
                           </form>
                       </div>""")

async def _h_conv_settings_save(request, payload, imr):
    cid = payload.get("cid","")
    conv = _load_conv(cid)
    if not conv or conv.get("username") != request.state.user.username: return imr
    conv["system_prompt_override"] = payload.get("system_prompt_override","")
    ctx_raw = payload.get("model_ctx_override","")
    conv["model_ctx_override"] = int(ctx_raw) if str(ctx_raw).strip().isdigit() else None
    _save_conv(conv)
    return imr.raw(_chat_html(conv, request))

# --- Intent handlers: export / import ---

async def _h_import_form(request, payload, imr): return imr.raw(f"""<div style="padding:1rem;max-width:36rem;margin:0 auto">
                                                                        <h3>Import Conversation</h3>
                                                                        <p style="font-size:.75rem;color:var(--text_muted)">Paste text exported from this same feature (or hand-format as **You:**/**Assistant:** blocks).</p>
                                                                        <form hx-post="/im/in" hx-target="#ath-chat-area" hx-swap="innerHTML">
                                                                            <input type="hidden" name="type" value="athena_import"><input type="hidden" name="lvl" value="2">
                                                                            <textarea name="text" class="cm-input" rows="14" style="width:100%"></textarea>
                                                                            <button type="submit" class="button" style="margin-top:.5rem">Import</button>
                                                                        </form>
                                                                    </div>""")

async def _h_import(request, payload, imr):
    user = request.state.user
    raw = payload.get("text","").strip()
    if not raw: return imr
    conv = _new_conv(user)
    for block in re.split(r'\n(?=\*\*(?:You|Assistant):\*\*)', raw):
        m = re.match(r'\*\*(You|Assistant):\*\*\s*(.*)', block.strip(), re.S)
        if not m: continue
        conv["messages"].append({"id":uuid.uuid4().hex[:8],"role":"user" if m.group(1)=="You" else "assistant","content":m.group(2).strip(),"user_name":user.username,"timestamp":datetime.utcnow().isoformat()})
    conv["title"] = (conv["messages"][0]["content"][:50] if conv["messages"] else "Imported Chat")
    _save_conv(conv)
    await ENV["set_state"](request, conv["id"], scope="user", namespace="athena", key="active_conv_id")
    imr.raw(_chat_html(conv, request))
    imr.oob(_left(user.username,conv["id"]), "ath-left", swap="innerHTML")
    return imr

@router.get("/export/{cid}")
async def export_conv(cid: str, request: Request):
    """Real file download, browser-fetched - legitimate plain route, matches the pattern used for image/document downloads elsewhere in this codebase (not a mutation, not a fragment update)."""
    conv = _load_conv(cid)
    if not conv or conv.get("username") != request.state.user.username: raise HTTPException(404)
    lines = [f"# {conv.get('title','Chat')}", ""]
    for m in conv.get("messages",[]):
        if m.get("deleted"): continue
        lines.append(f"**{'You' if m.get('role')=='user' else 'Assistant'}:** {m.get('content','')}\n")
    md = "\n".join(lines)
    fname = re.sub(r'[^\w\-. ]', '_', conv.get("title","chat"))[:40] or "chat"
    return Response(md, media_type="text/markdown", headers={"Content-Disposition": f'attachment; filename="{fname}.md"'})

# --- Admin: general settings ---

async def _h_admin_save(request, payload, imr):
    """Loads current disk state and updates only the fields THIS form actually submits, skipping 'capabilities' entirely (managed by its own editor below).
    Deliberately does not call SettingsGroup.save() here - that method treats a call as authoritative for every registered field, which would silently reset capabilities to their default every time the general form is saved without them."""
    if getattr(request.state.user, "role", "") != "admin": return imr
    group = cfg.get_group("general")
    data = group.load()
    for field in group.fields:
        if field.name == "capabilities": continue
        raw = payload.get(field.name)
        if field.type == "checkbox": data[field.name] = raw is not None
        elif field.type == "json":
            try: data[field.name] = json.loads(raw) if raw and raw.strip() else field.default
            except Exception: data[field.name] = field.default
        elif field.type == "number":
            try: data[field.name] = float(raw) if field.step != 1 else int(raw)
            except Exception: data[field.name] = field.default
        else: data[field.name] = raw if raw is not None else field.default
    group.json_path.write_text(json.dumps(data, indent=2))
    return imr.raw('<span style="color:var(--accent)">&#x2713; Saved successfully.</span>')

# --- Admin: capability editor (add/remove/edit blocks) ---

def _save_capabilities(caps):
    """Writes just the 'capabilities' key, preserving every other saved field - same non-destructive load-merge-write pattern as _h_admin_save, for the same reason."""
    data = cfg.get_group("general").load()
    data["capabilities"] = caps
    cfg.get_group("general").json_path.write_text(json.dumps(data, indent=2))

def _cap_by_id(caps, cap_id): return next((c for c in caps if c.get("id")==cap_id), None)

def _capability_card_html(cap):
    cid_field = cap.get("id","")
    conn_opts = "".join(f'<option value="{c["_id"]}" {"selected" if c["_id"]==cap.get("conn_id") else ""}>{_esc(c.get("display_name",c["_id"]))}</option>' for c in AIM.connections.list_conns())
    models = AIM.connections.list_models_sync(AIM.connections.get_conn(cap.get("conn_id",""))) if cap.get("conn_id") else []
    model_opts = "".join(f'<option value="{_esc(m)}" {"selected" if m==cap.get("model") else ""}>{_esc(m)}{" (embedding - not for chat)" if AIM.steps.looks_like_embedding(m) else ""}</option>' for m in models)
    kg_opts = "".join(f'<option value="{c["_id"]}" {"selected" if c["_id"]==cap.get("knowledge_conn_id") else ""}>{_esc(c.get("display_name",c["_id"]))}</option>' for c in AIM.connections.list_conns(conn_type="lightrag"))
    pl_opts = "".join(f'<option value="{p["id"]}" {"selected" if p["id"]==cap.get("flow_pipeline_id") else ""}>{_esc(p.get("name",p["id"]))}</option>' for p in AIM.engine.list_pipelines())
    return f"""<div class="glass" style="padding:.7rem;margin-bottom:.6rem" id="cap-card-{cid_field}">
                   <form hx-post="/im/in" hx-target="body" hx-swap="none" style="display:flex;flex-direction:column;gap:.5rem">
                       <input type="hidden" name="type" value="athena_cap_save"><input type="hidden" name="lvl" value="2">
                       <input type="hidden" name="cap_id" value="{cid_field}">
                       <div style="display:flex;gap:.4rem;align-items:center">
                           <input type="text" name="label" value="{_esc(cap.get('label',''))}" placeholder="Label shown to users" class="module-select" style="flex:1;font-weight:600">
                           <span style="font-size:.65rem;color:var(--text_muted);font-family:var(--font-mono)">{cid_field}</span>
                           <button type="button" class="btn-icon" style="color:#ff5f5f" hx-post="/im/in" hx-target="body" hx-swap="none" hx-vals='{_iv("athena_cap_delete", cap_id=cid_field)}' hx-confirm="Delete this capability?">&#x2715;</button>
                       </div>
                       <div style="display:flex;gap:.4rem;flex-wrap:wrap">
                           <label style="flex:1;min-width:10rem;font-size:.7rem;color:var(--text_muted)">Connection
                               <select name="conn_id" class="module-select" hx-post="/im/in" hx-vals='{_iv("athena_cap_conn_change", cap_id=cid_field)}' hx-trigger="change" hx-include="closest form" hx-target="#cap-model-wrap-{cid_field}">
                                   <option value="">-- none --</option>{conn_opts}
                               </select>
                           </label>
                           <label id="cap-model-wrap-{cid_field}" style="flex:1;min-width:10rem;font-size:.7rem;color:var(--text_muted)">Model
                               <select name="model" class="module-select"><option value="">(select connection first)</option>{model_opts}</select>
                           </label>
                           <label style="flex:1;min-width:8rem;font-size:.7rem;color:var(--text_muted)">Context Tokens<input type="number" name="model_ctx" value="{cap.get('model_ctx',16384)}" class="module-select"></label>
                           <label style="flex:1;min-width:8rem;font-size:.7rem;color:var(--text_muted)">Max Response Tokens<input type="number" name="num_predict" value="{cap.get('num_predict', cfg.get('num_predict',8192))}" class="module-select" title="Raise this for thinking-heavy models - thinking tokens count against this budget too."></label>
                       </div>
                       <label style="font-size:.7rem;color:var(--text_muted)">
                           Thinking Effort
                           <select name="think" class="module-select">
                               <option value="" {"selected" if _think_effort(cap)=="" else ""}>Off</option>
                               <option value="low" {"selected" if _think_effort(cap)=="low" else ""}>Low</option>
                               <option value="medium" {"selected" if _think_effort(cap)=="medium" else ""}>Medium</option>
                               <option value="high" {"selected" if _think_effort(cap)=="high" else ""}>High</option>
                           </select>
                       </label>
                       <label style="font-size:.7rem;color:var(--text_muted)">System Prompt (not shown to users)<textarea name="system_prompt" class="cm-input" rows="3">{_esc(cap.get('system_prompt',''))}</textarea></label>
                       <label style="font-size:.7rem;color:var(--text_muted)">Allowed Roles (comma-sep, blank = everyone)<input type="text" name="allowed_roles" value="{','.join(cap.get('allowed_roles',[]))}" class="module-select"></label>
                       <label style="font-size:.7rem;color:var(--text_muted)">Allowed Usernames (comma-sep, blank = everyone)<input type="text" name="allowed_users" value="{','.join(cap.get('allowed_users',[]))}" class="module-select"></label>
                       <div style="border-top:var(--border-thick) solid var(--border);padding-top:.5rem;display:flex;gap:.4rem;flex-wrap:wrap;align-items:flex-end">
                           <label style="display:flex;align-items:center;gap:.3rem;font-size:.8rem"><input type="checkbox" name="knowledge_enabled" value="1" {"checked" if cap.get("knowledge_enabled") else ""}> Knowledge Base</label>
                           <label style="flex:1;min-width:10rem;font-size:.7rem;color:var(--text_muted)">Knowledge Connection<select name="knowledge_conn_id" class="module-select"><option value="">-- none --</option>{kg_opts}</select></label>
                       </div>
                       <div style="border-top:var(--border-thick) solid var(--border);padding-top:.5rem;display:flex;gap:.4rem;flex-wrap:wrap;align-items:flex-end">
                           <label style="flex:1;min-width:10rem;font-size:.7rem;color:var(--text_muted)">Run via Pipeline instead of plain chat<select name="flow_pipeline_id" class="module-select"><option value="">-- none --</option>{pl_opts}</select></label>
                           <button type="button" class="cm-qbtn" hx-post="/im/in" hx-target="body" hx-swap="none" hx-vals='{json.dumps({"type":"athena_cap_to_pipeline","cap_id":cid_field,"lvl":2})}' title="Generate a starter knowledge+generate pipeline from this capability's current fields, and point this capability at it">&#x2699; Build pipeline from this capability</button>
                           <label style="flex:1;min-width:8rem;font-size:.7rem;color:var(--text_muted)">Result Key<input type="text" name="flow_result_key" value="{_esc(cap.get('flow_result_key','text'))}" class="module-select" placeholder="text"></label>
                       </div>
                       <button type="submit" class="button" style="align-self:flex-start">Save Capability</button>
                   </form>
               </div>"""

def _capabilities_editor_html():
    caps = cfg.get("capabilities", []) or [DEFAULT_CAP]
    return f"""<div id="cap-editor">
        <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:.5rem">
            <h4 style="margin:0">Capabilities</h4>
            <button class="ui-btn" hx-post="/im/in" hx-target="body" hx-swap="none" hx-vals='{_iv("athena_cap_add")}'>+ Add Capability</button>
        </div>
        {"".join(_capability_card_html(c) for c in caps)}
    </div>"""

async def _h_cap_add(request, payload, imr):
    if getattr(request.state.user, "role", "") != "admin": return imr
    caps = cfg.get("capabilities", []) or []
    caps.insert(0, {**DEFAULT_CAP, "id": f"cap_{uuid.uuid4().hex[:6]}", "label": "New Capability"})
    _save_capabilities(caps)
    return imr.oob(_capabilities_editor_html(), "cap-editor", swap="outerHTML")

async def _h_cap_delete(request, payload, imr):
    if getattr(request.state.user, "role", "") != "admin": return imr
    caps = [c for c in cfg.get("capabilities", []) or [] if c.get("id") != payload.get("cap_id","")]
    if not caps: caps = [dict(DEFAULT_CAP)]
    _save_capabilities(caps)
    return imr.oob(_capabilities_editor_html(), "cap-editor", swap="outerHTML")

async def _h_cap_conn_change(request, payload, imr):
    cap_id = payload.get("cap_id","")
    conn = AIM.connections.get_conn(payload.get("conn_id",""))
    models = AIM.connections.list_models_sync(conn) if conn else []
    opts = "".join(f'<option value="{_esc(m)}">{_esc(m)}{" (embedding - not for chat)" if AIM.steps.looks_like_embedding(m) else ""}</option>' for m in models)
    return imr.oob(f'<label id="cap-model-wrap-{cap_id}" style="flex:1;min-width:10rem;font-size:.72rem;color:var(--text_muted)">Model<select name="model" class="module-select"><option value="">(auto)</option>{opts}</select></label>', f"cap-model-wrap-{cap_id}", swap="outerHTML")

async def _h_cap_save(request, payload, imr):
    if getattr(request.state.user, "role", "") != "admin": return imr
    cap_id = payload.get("cap_id","")
    caps = cfg.get("capabilities", []) or []
    existing = _cap_by_id(caps, cap_id)
    updated = {"id": cap_id, "label": payload.get("label","").strip() or cap_id,
               "conn_id": payload.get("conn_id",""),
               "model": payload.get("model",""),
               "system_prompt": payload.get("system_prompt",""),
                "think": payload.get("think","") or "",
               "model_ctx": int(payload.get("model_ctx", 16384) or 16384),
               "num_predict": int(payload.get("num_predict", 8192) or 8192),
               "knowledge_enabled": payload.get("knowledge_enabled")=="1",
               "knowledge_conn_id": payload.get("knowledge_conn_id",""),
               "allowed_roles": [r.strip() for r in payload.get("allowed_roles","").split(",") if r.strip()],
               "allowed_users": [u.strip() for u in payload.get("allowed_users","").split(",") if u.strip()],
               "flow_pipeline_id": payload.get("flow_pipeline_id",""),
               "flow_result_key": payload.get("flow_result_key","text") or "text"}
    if existing: caps[caps.index(existing)] = updated
    else: caps.append(updated)
    _save_capabilities(caps)
    return imr.oob(_capability_card_html(updated), f"cap-card-{cap_id}", swap="outerHTML")

async def _h_cap_to_pipeline(request, payload, imr):
    """Builds a starter knowledge-retrieval -> generate pipeline from this capability's current fields, pinned via conn_id so it runs with no CNode/resource-pool setup required. Points the capability's flow_pipeline_id at the result. Does not touch or remove the direct chat path - capabilities without a pipeline set still use it unchanged."""
    if getattr(request.state.user, "role", "") != "admin": return imr
    cap_id = payload.get("cap_id","")
    caps = cfg.get("capabilities", []) or []
    cap = _cap_by_id(caps, cap_id)
    if not cap: return imr
    pipeline = {"id": f"pl_{uuid.uuid4().hex[:10]}", "name": f"{cap.get('label',cap_id)} (from capability)", "owner": "athena", "tags": ["athena","knowledge"], "pool": AIM.engine.DEFAULT_POOL, "created": datetime.utcnow().isoformat(),
        "flow": {"nodes": [
            {"id":"n_retrieve","name":"Retrieve Context","type":"knowledge","config":{"mode":"query","query_template":"{input}","query_mode":"hybrid","conn_id":cap.get("knowledge_conn_id","")},"key_map":{"response":"kg_context"},"status":"idle"},
            {"id":"n_answer","name":"Answer","type":"generate","config":{"modality":"text","conn_id":cap.get("conn_id",""),"model":cap.get("model",""),"system_prompt":cap.get("system_prompt",""),"user_template":"Context:\n{kg_context}\n\nQuestion: {input}","think":_think_effort(cap)},"key_map":{"text":"answer"},"extra_in_keys":["kg_context"],"status":"idle"}
        ], "appearance": {}}}
    AIM.engine.save_pipeline(pipeline)
    cap["flow_pipeline_id"], cap["flow_result_key"] = pipeline["id"], "answer"
    _save_capabilities(caps)
    return imr.oob(_capability_card_html(cap), f"cap-card-{cap_id}", swap="outerHTML")

@router.get("/admin")
async def admin(request: Request):
    """Page-load navigation into the ai_tools shell's content area - not a mutation, follows the same GET convention used throughout ai_tools for switching between submodule panels."""
    if getattr(request.state.user, "role", "") != "admin": return HTMLResponse("Denied")
    group = cfg.get_group("general")
    values = cfg.get_all().get("general", {})
    general_fields_html = "".join(group._render_field(f, values) for f in group.fields if f.name != "capabilities")
    return HTMLResponse(f"""<div style="padding:0.5rem;position:relative;max-width:48rem;margin:0 auto">
                                <button type="button" class="close-btn" style="position:absolute;top:.2rem;right:.2rem" hx-get="{_u()}" hx-target="#ait-workspace" hx-swap="innerHTML">&#x2715;</button>
                                <h3 style="margin-top:0">Athena Admin Settings</h3>
                                <form hx-post="/im/in" hx-target="#status" hx-swap="innerHTML">
                                    <input type="hidden" name="type" value="athena_admin_save"><input type="hidden" name="lvl" value="2">
                                    <div id="athena-admin-fields">{general_fields_html}</div>
                                    <button type="submit" class="button" style="margin-top:1rem;">Save Settings</button>
                                    <div id="status" style="margin-top:0.5rem; font-size:0.7rem; color:#00ffa2;"></div>
                                </form>
                                <hr style="margin:1.2rem 0;border-color:var(--border)">
                                {_capabilities_editor_html()}
                            </div>""")

def right_panel() -> str: return f"""<div class="ait-rp"><div class="ait-rp-hd">Athena</div>
                                         <div style="font-size:.7rem; color:var(--text_muted); padding:.2rem .2rem .2rem">{_esc(cfg.get("title","Athena"))}</div>
                                         <button class="ait-rp-btn" hx-post="/im/in" hx-target="#ait-workspace" hx-swap="innerHTML" hx-vals='{_iv("athena_new")}'>+ New Conversation</button>
                                         <button class="ait-rp-btn" hx-get="{_u("admin")}" hx-target="#ait-workspace" hx-swap="innerHTML">&#x2699; Admin Settings</button>
                                     </div>"""