"""
Athena - Managed AI chat. One ChatManager per instance. Admin configures model/system prompt, users chat.
Sub-module of ai_tools. Mounted at /module/ai_tools/athena.
Conversation storage: JSON files per conversation.
User organization (folders): JSON metadata per user.
IM routes submit; HTMX routes conversation navigation and management.
"""
import json, uuid, re, asyncio, base64
from pathlib import Path
from datetime import datetime
import httpx
from fastapi import APIRouter, Request, UploadFile, File, Form
from fastapi.responses import HTMLResponse, JSONResponse
import openpyxl
import shutil

from modules.ai_tools.ai_utils import (get_conn, list_conns, list_models_sync, list_models_async, _base, tok_estimate, conn_opts_html, model_opts_html, KG_DIR)

# #*************************************************
# for f in Path("./data/ai_tools/athena/conversations").glob("*.json"):
#     d = json.loads(f.read_text())
#     changed = False
#     for m in d.get("messages", []):
#         if isinstance(m.get("content"), list):
#             m["content"] = "[corrupted - removed by repair script]"
#             changed = True
#     if changed: f.write_text(json.dumps(d, indent=2)); print("repaired", f.name)
# #*************************************************

TOOL_META = {"label": "Athena", "group": "chat", "icon": "&#x1F989;", "description": "AI chat", "singleton": True}

router = APIRouter(redirect_slashes=False)

ENV: dict = {}
_P = "/module/ai_tools/athena"
DATA_DIR = Path("./data/ai_tools/athena")
KG_DIR = Path("./data/ai_tools/_knowledge")
COMMON_DIR = Path("./data/_common")

UI = None
WS = None
IM = None
CM = None
AIM = None
cfg = {}

def _u(*p): return "/".join(s.strip("/") for s in [_P,*p] if s)

# --- Athena Initialization ---
def init_tool(env:dict, prefix:str):
    global ENV, _P, UI, WS, IM, CM, AIM, cfg
    ENV = env
    _P = prefix.rstrip("/")
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    (DATA_DIR/"conversations").mkdir(exist_ok=True)
    UI=ENV["templates"].env.globals.get("UI")
    WS=ENV["ws"]
    IM=ENV["InterfaceManager"](nesting_level=2, db_path="ai_tools/athena/im_registry.db")
    built_ins = ENV["tools"]["built_ins"]
    AIM = ENV["tools"]["ai_manager"]
    # Configure Athena using the agnostic framework
    cfg = built_ins.SettingsPanel("Athena Settings", [
            built_ins.SettingsGroup("general", "General", [
                built_ins.SettingField("title", "Title", "text", "Athena"),
                built_ins.SettingField("conn_id", "Connection", "select", options=get_connection_options, hx_get=_u("admin", "fields"), hx_target="#athena-admin-fields"),
                built_ins.SettingField("model", "Model", "select", options=get_model_options),
                built_ins.SettingField("think_model", "Think Model", "select", options=get_model_options),
                built_ins.SettingField("system_prompt", "System Prompt", "textarea", ""),
                built_ins.SettingField("model_ctx", "Context Tokens", "number", 8192),
                built_ins.SettingField("user_input_limit", "User Input Limit", "number", 2000),
                built_ins.SettingField("msg_buffer", "Message Buffer", "number", 500),
                built_ins.SettingField("allow_files", "Allow Files", "checkbox", True),
                built_ins.SettingField("user_overrides", "User Overrides", "json", default={}, hint='JSON dict mapping username to custom settings, e.g., {"user1": {"model": "llama3", "system_prompt": "..."}}') # JSON overrides field for the <10 users
            ], json_path="data/settings/athena.json")
        ])

    IM.scripts["submit"] = [_handle_submit]
    CM=ENV["tools"]["built_ins"].ChatManager(namespace="athena", base_url=_u(), view_style="bubble", stream_toggle=True, think_toggle=True, stop_enabled=True, show_export=True, pin_enabled=True, allow_edit=True, allow_delete=True, allow_copy=True, show_info=True, markdown_mode="standard", branch_id=IM.branch_id, nesting_level=2)
    print(f"[athena] ready at {_P}")

_STOP_FLAGS:dict = {}
_ACTIVE_STREAMS:set = set()
_STREAM_BUFFERS:dict = {}  # sid->{full,thinking,done,error,username} - server-side accumulator

def _cp(cid): return DATA_DIR/"conversations"/f"{Path(cid).name}.json"
def _esc(s): return str(s).replace("&","&amp;").replace("<","&lt;").replace(">","&gt;").replace('"',"&quot;")
def _tok(t): return max(1,len(str(t))//4)
def _load_conv(cid): return json.loads(_cp(cid).read_text())
def _save_conv(c): c["modified"]=datetime.utcnow().isoformat(); _cp(c["id"]).write_text(json.dumps(c,indent=2))
def _list_convs(u): return [c for c in (json.loads(p.read_text()) for p in sorted((DATA_DIR/"conversations").glob("*.json"), key=lambda x:x.stat().st_mtime, reverse=True)) if c.get("username") == u]
def _del_conv(cid): p=_cp(cid); p.unlink() if p.exists() else None
def _org(u): p=DATA_DIR/f"org_{u}.json"; return json.loads(p.read_text()) if p.exists() else {"folders":{},"conv_folders":{}}
def _save_org(u,o): (DATA_DIR/f"org_{u}.json").write_text(json.dumps(o, indent=2))

def _new_conv(user):
    # Extract admin-defined overrides for this specific user - Overrides take priority, fallback to global cfg
    overrides = cfg.get("user_overrides", {}).get(user.username, {})
    return {"id": f"ath_{uuid.uuid4().hex[:8]}", "user_id": str(user.id), "username": user.username, "user_display": user.username, "title": "New Chat", "model": overrides.get("model") or cfg.get("model", ""), "system_prompt": overrides.get("system_prompt") or cfg.get("system_prompt", ""), "messages": [], "context_summary": "", "attached_files": [], "created": datetime.utcnow().isoformat(), "modified": datetime.utcnow().isoformat()}

def _conv_ctx_info(conv):
    """Approximate token usage for sidebar display."""
    if not conv: return ""
    sys_p=conv.get("system_prompt","") or cfg.get("system_prompt","")
    total=_tok(sys_p)+sum(_tok(m.get("content","")) for m in conv.get("messages",[]) if not m.get("deleted"))
    ctx=conv.get("model_ctx",cfg.get("model_ctx",8192)); pct=min(total/max(ctx,1)*100,100)
    col="#00ffa2" if pct<60 else "#ffcc00" if pct<80 else "#ff9944" if pct<95 else "#ff5f5f"
    return (f"""<div style="padding:.3rem .5rem;font-size:.65rem;color:var(--text_muted);border-top:var(--border-thick) solid var(--border);flex-shrink:0;display:flex;justify-content:space-between"><span>~{total:,}t used</span><span style="color:{col}">{pct:.0f}% of {ctx//1000}k ctx</span></div>""")

def _find_msg(username, mid):
    for c in _list_convs(username):
        conv=_load_conv(c.get("id",""))
        if not conv: continue
        for i,m in enumerate(conv.get("messages",[])):
            if m.get("id")==mid: return conv,i,m
    return None, None, None

# --- System Prompts and uploads ---

def _prompts_dir(): d=DATA_DIR/"prompts"; d.mkdir(exist_ok=True); return d
def _uploads_dir(cid): d=DATA_DIR/"uploads"/cid; d.mkdir(parents=True,exist_ok=True); return d
def _list_prompts(): return [json.loads(f.read_text()) for f in sorted(_prompts_dir().glob("*.json"), key=lambda x:x.stat().st_mtime, reverse=True)]
def _load_prompt(pid): return json.loads((_prompts_dir()/f"{pid}.json").read_text())
def _save_prompt(p): (_prompts_dir()/f"{p['id']}.json").write_text(json.dumps(p,indent=2))
def _del_prompt(pid): p=_prompts_dir()/f"{pid}.json"; p.unlink() if p.exists() else None

def _attach_content(conv):
    """Returns (text_parts, image_b64_list) from conv attached_files."""
    text_parts=[]; images=[]
    for f in conv.get("attached_files",[]):
        p=Path(f["path"])
        if not p.exists(): continue
        ext=f.get("ext","").lower()
        if ext in (".png",".jpg",".jpeg",".webp",".gif"):
            images.append(base64.b64encode(p.read_bytes()).decode())
        elif ext in (".csv",".txt",".md"):
            text_parts.append(f"[File: {f['name']}]\n{p.read_text(errors='ignore')[:8000]}")
        elif ext in (".xlsx",".xls"):
            try:
                wb=openpyxl.load_workbook(p,read_only=True, data_only=True)
                rows=[]
                for ws in wb.worksheets:
                    rows.append(f"Sheet: {ws.title}")
                    for row in ws.iter_rows(values_only=True,max_row=500):
                        rows.append(",".join(str(c or "") for c in row))
                text_parts.append(f"[Excel: {f['name']}]\n"+"\n".join(rows))
            except Exception as e: text_parts.append(f"[Excel: {f['name']} - parse error: {e}]")
    return text_parts,images

# --- Ollama ---

def get_connection_options(values=None): return [(c["_id"], c.get("display_name", c["_id"])) for c in list_conns()]

def get_model_options(values=None):
    conn_id = (values or {}).get("conn_id") or cfg.get("conn_id","")
    conn = AIM.get_conn(conn_id) if conn_id else None
    return [(m, m) for m in AIM.list_models_sync(conn)] if conn else []

async def _stream_ollama(conn, msgs, model, ctx, think=False, images=None):
    if images and msgs: msgs[-1]["images"] = images
    pl = {"model":model,"messages":msgs,"stream":True,"options":{"num_ctx":ctx,"num_predict":4096,"temperature":0.3,"top_k":20,"top_p":0.5}}
    if think: pl["think"] = True
    tb = ""
    async with httpx.AsyncClient(timeout=httpx.Timeout(connect=30.0, read=1800.0, write=10.0, pool=30.0)) as c:
        async with c.stream("POST", f"{_base(conn)}/api/chat", json=pl) as resp:
            if resp.status_code == 503: yield "", "", True, "Ollama busy (503)"; return
            if resp.status_code != 200:
                body = await resp.aread()
                yield "", "", True, f"HTTP {resp.status_code}: {body.decode()[:300]}"; return
            async for line in resp.aiter_lines():
                if not line: continue
                try:
                    chunk = json.loads(line)
                    if chunk.get("error"): yield "", tb, True, chunk["error"]; return
                    msg = chunk.get("message",{})
                    tb += msg.get("thinking","")
                    text = msg.get("content","")
                    done = chunk.get("done",False)
                    if text or done or tb: yield text, tb, done, None
                    if done: return
                except: continue

# --- Context ---

def _build_msgs(conv, user_msg):
    ctx = conv.get("model_ctx", cfg.get("model_ctx", 8192))
    budget = int(ctx * 0.82)
    sys_p = conv.get("system_prompt","") or cfg.get("system_prompt","")
    summary = conv.get("context_summary","").strip()
    out = []
    sys_parts = [sys_p] if sys_p else []
    if summary: sys_parts.append(f"[Prior context]\n{summary}")
    if sys_parts: out.append({"role":"system","content":"\n\n---\n\n".join(sys_parts)})
    sys_tok = sum(_tok(m["content"]) for m in out)
    available = budget - sys_tok - _tok(user_msg) - 256
    if available < 100: raise ValueError(f"System prompt fills context window ({sys_tok}t sys, {_tok(user_msg)}t input, {budget}t budget)")
    history = [m for m in conv.get("messages",[]) if not m.get("deleted")]
    recent = []; used = 0; truncated = 0
    for m in reversed(history):
        t = _tok(m.get("content",""))
        if used + t > available: truncated += 1; continue
        recent.insert(0, {"role":m["role"],"content":m["content"]}); used += t
    out.extend(recent); out.append({"role":"user","content":user_msg})
    return out, truncated

# --- IM Submit + Stream ---

async def _handle_submit(request, payload:dict, imr):
    sid=payload.get("cid","").strip(); content=payload.get("content","").strip()
    if not sid or not content: return imr
    imr.raw(CM.working_html(sid, _u("stop",sid)))
    imr.raw(f"""<textarea id="cm-in-{sid}" name="content" class="cm-input" placeholder="Type a message\u2026 (Ctrl+Enter)" spellcheck="true" hx-swap-oob="outerHTML"></textarea>""")
    asyncio.create_task(_do_stream(request.state.user.username, payload, sid))
    await asyncio.sleep(0.1)
    return imr

async def _do_stream(username: str, payload: dict, sid: str):
    global _ACTIVE_STREAMS
    content = payload.get("content","").strip(); think = payload.get("think") in ("1","true",True)
    _STREAM_BUFFERS[sid] = {"full":"","thinking":"","done":False,"error":None,"username":username}
    async def _ws(html): await WS.send_personal_message(html, username); await asyncio.sleep(0.01)
    async def _err(msg):
        _STREAM_BUFFERS[sid].update({"error":msg,"done":True})
        await _ws(f'<div id="cm-msgs-{sid}" hx-swap-oob="beforeend"><div style="color:#ff5f5f;font-size:.78rem;padding:.3rem .6rem">&#x26A0; {_esc(msg)}</div></div>{CM.working_hide_html(sid)}')

    full = ""; tb = ""
    try:
        conv = _load_conv(sid)
        model = conv.get("model") or cfg.get("model","")
        if think: model = conv.get("think_model") or cfg.get("think_model","") or model
        conn = get_conn(conv.get("conn_id","") or cfg.get("conn_id",""))
        num_ctx = conv.get("model_ctx", cfg.get("model_ctx",8192))
        if not conn: await _err("No Ollama connection. Add one in AI Tools > Settings."); return
        if not model: await _err("No model configured in Athena Admin."); return
        max_input_tok = int(num_ctx * 0.65)
        if _tok(content) > max_input_tok: content = content[:max_input_tok * 4] + f"\n\n[Input was truncated: original length exceeded {max_input_tok} token limit for {num_ctx} context window]"
        # if _tok(content) > int(num_ctx*0.70): await _err(f"Input too long (~{_tok(content)}t, limit ~{int(num_ctx*0.70)}t)."); return

        try: built_msgs, truncated = _build_msgs(conv, content)
        except ValueError as e: await _err(f"Context error: {e}"); return
        if truncated: await _ws(f'<div id="cm-msgs-{sid}" hx-swap-oob="beforeend"><div style="font-size:.68rem;color:#ffcc00;padding:.2rem .6rem;border-left:2px solid #ffcc00">&#x26A0; {truncated} older message{"s" if truncated>1 else ""} shifted out of context window.</div></div>')

        user_msg = {"id":uuid.uuid4().hex[:8],"role":"user","content":content,"user_name":conv.get("user_display",username),"timestamp":datetime.utcnow().isoformat()}
        conv["messages"].append(user_msg)
        _save_conv(conv)
        await _ws(f'<div id="cm-msgs-{sid}" hx-swap-oob="beforeend">{CM.render_message(user_msg, is_me=True, can_delete=True, can_edit=True)}</div>')

        _,images = _attach_content(conv)
        _ACTIVE_STREAMS.add(sid)
        try:
            async for text, thinking, done, err in _stream_ollama(conn, built_msgs, model, num_ctx, think, images=images or None):
                if _STOP_FLAGS.pop(sid, False): break
                if err:
                    _STREAM_BUFFERS[sid]["error"] = err
                    await _err(f"Ollama: {err}"); return
                if text: full += text
                if thinking: tb = thinking
                _STREAM_BUFFERS[sid].update({"full":full,"thinking":tb})
                think_html = (f'<details class="cm-think" open><summary>&#x1F9E0; Thinking ({len(tb)//4}t)\u2026</summary><div class="cm-think-body">{_esc(tb)}</div></details>') if tb.strip() else ""
                await _ws(f'<div id="cm-stream-{sid}" hx-swap-oob="innerHTML">{think_html}{"<div class=cm-stream-bubble>"+_esc(full)+"</div>" if full else ""}</div>')
                if done: break
        finally:
            _ACTIVE_STREAMS.discard(sid)

        _STREAM_BUFFERS[sid]["done"] = True
        err_flag = _STREAM_BUFFERS[sid].get("error")

        if not full:
            await _ws(f'<div id="cm-stream-{sid}" hx-swap-oob="innerHTML"></div>{CM.working_hide_html(sid)}')
            if err_flag: await _err(f"Ollama error: {err_flag}")
            return

        conv = _load_conv(sid)
        if conv:
            ai_msg = {"id":uuid.uuid4().hex[:8],"role":"assistant","content":full,
                      "thinking":tb.strip() if tb.strip() else "","model":model,
                      "timestamp":datetime.utcnow().isoformat(),"response_tokens":_tok(full)}
            if err_flag: ai_msg["partial"] = True
            conv["messages"].append(ai_msg)
            if len(conv["messages"]) == 2 and conv.get("title","") in ("","New Chat"):
                conv["title"] = conv["messages"][0].get("content","")[:50]
            _save_conv(conv)
            partial_badge = '<span style="font-size:.65rem;color:#ffaa44;margin-left:.3rem">[partial]</span>' if err_flag else ""
            await _ws(f'<div id="cm-msgs-{sid}" hx-swap-oob="beforeend">{CM.render_message(ai_msg,is_me=False,can_delete=True,can_edit=False)}{partial_badge}</div>'
                      + f'<div id="cm-stream-{sid}" hx-swap-oob="innerHTML"></div>'
                      + CM.working_hide_html(sid)
                      + f'<div id="ath-left" hx-swap-oob="innerHTML">{_left(username,sid)}</div>')
            if err_flag: await _err(f"Ollama error (partial response saved): {err_flag}")
    except Exception as e:
        print(f"[athena] stream error {sid}: {e}")
        if full:
            try:
                conv = _load_conv(sid)
                if conv:
                    ai_msg = {"id":uuid.uuid4().hex[:8],"role":"assistant","content":full,"thinking":tb.strip(),"model":"","partial":True,"timestamp":datetime.utcnow().isoformat(),"response_tokens":_tok(full)}
                    conv["messages"].append(ai_msg); _save_conv(conv)
            except Exception as save_err: print(f"[athena] partial save failed: {save_err}")
        await _err(f"Server error: {e}")
    finally:
        _STREAM_BUFFERS.pop(sid, None)

def _conv_item(c, active, org):
    cid = c.get("id","")
    title = _esc((c.get("title","") or "Untitled")[:42])
    ac = " ath-conv-active" if cid == active else ""
    #date = (c.get("modified","") or "")[:10]
    short_title = (c.get("title","") or "Untitled")[:30]
    partial_badge = '<span style="font-size:.6rem;color:#ffaa44;margin-left:.2rem" title="Last response was partial">&#x25CC;</span>' if any(m.get("partial") for m in c.get("messages",[])) else ""
    folders = org.get("folders",{}); folder_id = org.get("conv_folders",{}).get(cid,"") or ""
    folder_sel = ""
    if folders:
        opts = '<option value="">No folder</option>' + "".join(f'<option value="{fid}" {"selected" if fid==folder_id else ""}>{_esc(fd["name"])}</option>' for fid,fd in sorted(folders.items(),key=lambda x:x[1].get("order",0)))
        folder_sel = f'<select class="ath-folder-sel" name="folder_id" hx-post="{_u("folder","assign",cid)}" hx-trigger="change" hx-target="#ath-left" hx-swap="innerHTML" onclick="event.stopPropagation()">{opts}</select>'
    return f"""<div class="ath-conv-item{ac}" id="ath-ci-{cid}" hx-get="{_u("load",cid)}" hx-target="#ath-chat-area" hx-swap="innerHTML">
        <div style="display:flex;align-items:center;gap:.2rem;width:100%">
            <span class="ath-conv-title" style="flex:1">{title}{partial_badge}</span>
            <span class="ath-conv-actions" style="display:flex;gap:.15rem;flex-shrink:0">
                <button class="cm-qbtn" hx-get="{_u("conv","rename_form",cid)}" hx-target="#ath-ci-{cid}" hx-swap="innerHTML" onclick="event.stopPropagation()">&#x270E;</button>
                <button class="cm-qbtn" hx-post="{_u("conv","delete",cid)}" hx-target="#ath-chat-area" hx-swap="innerHTML" hx-confirm="Delete '{short_title}'?" onclick="event.stopPropagation()">&#x2715;</button>
            </span>
        </div>
        <details style="font-size:.6rem;color:var(--text_muted)" onclick="event.stopPropagation()"><summary style="list-style:none;cursor:pointer;user-select:none">&#x25B8;</summary><div style="display:flex;justify-content:space-between;padding:.15rem 0">{folder_sel}</div></details>
    </div>"""

def _left(username, active=""):
    global cfg
    org = _org(username)
    convs = _list_convs(username)
    folders = org.get("folders", {})
    active_fid = org.get("conv_folders", {}).get(active)
    grouped = {fid: [] for fid in folders}; ungrouped = []
    for c in convs:
        cid = c.get("id", "") or c.get("_id", "")
        fid = org.get("conv_folders", {}).get(cid)
        (grouped[fid] if fid and fid in grouped else ungrouped).append(c)
    folder_html = "".join(f"""<details class="ath-folder" {"open" if fid==active_fid else ""}><summary class="ath-folder-sum">&#x1F4C1; <span id="ath-fn-{fid}">{_esc(fd["name"])}</span><button class="cm-qbtn" hx-get="{_u("folder","rename_form",fid)}" hx-target="#ath-fn-{fid}" hx-swap="outerHTML" onclick="event.stopPropagation()">&#x270E;</button><button class="cm-qbtn" hx-post="{_u("folder","delete",fid)}" hx-target="#ath-left" hx-swap="innerHTML" hx-confirm="Delete folder?" style="margin-left:auto;color:#ff5f5f" onclick="event.stopPropagation()">&#x2715;</button></summary>{"".join(_conv_item(c,active,org) for c in grouped.get(fid,[]))}</details>""" for fid, fd in sorted(folders.items(), key=lambda x: x[1].get("order", 0)))
    ug_html = "".join(_conv_item(c, active, org) for c in ungrouped)
    ug_hdr = '<div class="ath-ungrouped-hdr">Other</div>' if folder_html and ungrouped else ""
    active_conv = _load_conv(active) if active else None
    ctx_footer = _conv_ctx_info(active_conv)
    # Using the new cfg object to drive UI labels
    app_title = cfg.get("title", "Athena") 
    return (f"""<div class="ath-sb-hdr">
                    <button class="btn-icon" hx-post="{_u("new")}" hx-target="#ath-chat-area" hx-swap="innerHTML" title="New chat" style="font-size:1rem">+</button>
                    <span style="font-size:.68rem;text-transform:uppercase;letter-spacing:.05em;color:var(--text_muted);flex:1;padding:0 .3rem">{_esc(app_title)}</span>
                    <button class="btn-icon" hx-get="{_u("folder","new_form")}" hx-target="#ath-folder-new" hx-swap="innerHTML" style="font-size:.75rem">&#x1F4C1;+</button>
                </div>
                <div id="ath-folder-new"></div>
                <div class="ath-conv-list">
                    {folder_html}
                    {ug_hdr}
                    {ug_html}
                </div>
                <button class="ait-rp-btn" hx-get="{_u("admin")}" hx-target="#ait-workspace" hx-swap="innerHTML">&#x2699; Admin Settings</button>
    {ctx_footer}""")

def _chat_html(conv, requests = None):
    sid=conv["id"]; pipes=cfg.get("pipelines",[])
    pipe_sel=""
    if pipes:
        opts="".join(f'<option value="{p["name"]}" {"selected" if p["name"]==conv.get("pipeline","") else ""}>{_esc(p["label"])}</option>' for p in pipes)
        pipe_sel=(f"""<select class="module-select" style="font-size:.72rem;max-width:9rem" hx-post="{_u("pipeline",sid)}" hx-trigger="change" hx-target="#ath-pipe-{sid}" hx-include="this" name="pipeline"><option value="">General</option>{opts}</select>""")
    hdr=(f"""<span style="font-size:.88rem;font-weight:600;flex:1">{_esc(conv.get("title","Chat"))}</span>{pipe_sel}<div id="ath-pipe-{sid}" style="font-size:.68rem;color:var(--text_muted)"></div>""")
    attached = conv.get("attached_files",[])
    file_chips = "".join(
        f'<span style="display:inline-flex;align-items:center;gap:.2rem;background:var(--accent_dim);border:var(--border-thick) solid var(--accent);color:var(--accent);padding:.1rem .35rem;border-radius:.3rem;font-size:.7rem;max-width:12rem">'
        f'<span style="overflow:hidden;text-overflow:ellipsis;white-space:nowrap" title="{_esc(f["name"])}">{_esc(f["name"][:20])}</span>'
        f'<button onclick="athDelFile(\'{sid}\',\'{f["id"]}\')" style="background:none;border:none;cursor:pointer;color:var(--accent);font-size:.8rem;padding:0;flex-shrink:0;line-height:1">&#x2715;</button>'
        f'</span>'
        for f in attached)
    extra_footer = (f"""<div style="display:flex;align-items:center;gap:.3rem;flex-wrap:wrap;padding-top:.1rem">
                            <label class="btn-icon" title="Attach file" style="cursor:pointer;font-size:.9rem;flex-shrink:0">&#x1F4CE;<input type="file" style="display:none" accept="image/*,.csv,.txt,.md,.xlsx,.xls" onchange="athUpload(this,'{sid}')" multiple></label>
                            <div id="ath-files-{sid}" style="display:flex;gap:.2rem;flex-wrap:wrap;flex:1;min-width:0">{file_chips}</div>
                        </div>""")
    buf=_STREAM_BUFFERS.get(sid)
    is_working=bool(buf and not buf.get("done"))
    shell=CM.shell(sid, messages=conv.get("messages",[]), viewer_name=conv.get("user_display",""), header_html=hdr, extra_footer=extra_footer, is_working=is_working, stop_url=_u("stop",sid) if is_working else "")
    resume=""
    if is_working:
        full=buf.get("full",""); tb=buf.get("thinking","")
        think_html=(f"""<details class="cm-think" open><summary>&#x1F9E0; Thinking ({len(tb)//4}t)\u2026</summary><div class="cm-think-body">{_esc(tb)}</div></details>""") if tb.strip() else ""
        stream_content=think_html+(f'<div class="cm-stream-bubble">{_esc(full)}</div>' if full else "")
        resume=f'<script>document.getElementById("cm-stream-{sid}").innerHTML={json.dumps(stream_content)};</script>'
    scroll=f'<script>requestAnimationFrame(function(){{var m=document.getElementById("cm-msgs-{sid}");if(m)m.scrollTop=m.scrollHeight;}});</script>'
    return shell+resume+scroll

# --- Main Route ---

@router.get("")
@router.get("/")
async def root(request:Request):
    global _P, UI, WS, IM, CM, cfg
    user=request.state.user
    username=user.username
    cid=await ENV["get_state"](request,scope="user",namespace="athena",key="active_conv_id")
    conv=_load_conv(cid) if cid else None
    if not conv or conv.get("username")!=username:
        convs=_list_convs(username)
        conv=_load_conv(convs[0].get("id","") or convs[0].get("_id","")) if convs else None
    if not conv: conv=_new_conv(user); _save_conv(conv)
    await ENV["set_state"](request, conv["id"], scope="user", namespace="athena", key="active_conv_id")
    return ENV["templates"].TemplateResponse(name = "base.html", request = request, context = {"request": request,"user": user, "nesting_level": 2, "shell_id": IM.branch_id, "toolbars": {"left": UI.toolbar(side="left", content=f'<div id="ath-left" style="display:flex;flex-direction:column;height:100%;overflow:hidden">{_left(username, conv["id"])}', size="16rem", overlay=False, start_open=True, id="ath-left-bar", nesting_level=2)}, "content":f'<div id="ath-chat-area" style="height:100%;overflow:hidden;">{_chat_html(conv,request)}</div>' + '<script>' + EXTRA_JS + CM.SCRIPT + '</script>', "extra_css": CSS + CM.CSS})   #, "extra_script": EXTRA_JS + CM.SCRIPT})

@router.post("/new")
async def conv_new(request:Request):
    user=request.state.user; conv=_new_conv(user); _save_conv(conv)
    await ENV["set_state"](request,conv["id"],scope="user",namespace="athena",key="active_conv_id")
    return HTMLResponse(_chat_html(conv,request)+f'<div id="ath-left" hx-swap-oob="innerHTML">{_left(user.username,conv["id"])}</div>')

@router.get("/load/{cid}")
async def conv_load(cid:str, request:Request):
    user=request.state.user; conv=_load_conv(cid)
    if not conv or conv.get("username")!=user.username: return HTMLResponse("Not found",status_code=404)
    await ENV["set_state"](request,cid,scope="user",namespace="athena",key="active_conv_id")
    return HTMLResponse(_chat_html(conv,request)+f'<div id="ath-left" hx-swap-oob="innerHTML">{_left(user.username,cid)}</div>')

@router.post("/stop/{sid}")
async def conv_stop(sid:str): _STOP_FLAGS[sid]=True; return HTMLResponse("")

@router.post("/conv/delete/{cid}")
async def delete_conv(cid: str, request: Request):
#@router.post("/delete/{cid}")
#async def delete_conv(cid: str, request: Request):
    user = request.state.user
    conv = _load_conv(cid)
    if conv and conv.get("username") == user.username: _del_conv(cid)
    org = _org(user.username); org.get("conv_folders",{}).pop(cid, None); _save_org(user.username, org)
    remaining = _list_convs(user.username)
    if remaining:
        next_cid = remaining[0].get("id","") or remaining[0].get("_id","")
        next_conv = _load_conv(next_cid)
        if next_conv:
            await ENV["set_state"](request, next_cid, scope="user", namespace="athena", key="active_conv_id")
            return HTMLResponse(_chat_html(next_conv, request) + f'<div id="ath-left" hx-swap-oob="innerHTML">{_left(user.username, next_cid)}</div>')
    new_conv = _new_conv(user); _save_conv(new_conv)
    await ENV["set_state"](request, new_conv["id"], scope="user", namespace="athena", key="active_conv_id")
    return HTMLResponse(_chat_html(new_conv, request) + f'<div id="ath-left" hx-swap-oob="innerHTML">{_left(user.username, new_conv["id"])}</div>')

@router.post("/conv/rename/{cid}")
async def conv_rename(cid:str, request:Request):
    form=await request.form(); user=request.state.user; conv=_load_conv(cid)
    if not conv or conv.get("username")!=user.username: return HTMLResponse("")
    conv["title"]=form.get("value","").strip() or conv.get("title","Chat"); _save_conv(conv)
    return HTMLResponse(_left(user.username,cid))

@router.get("/conv/rename_form/{cid}")
async def conv_rename_form(cid:str):
    conv=_load_conv(cid)
    if not conv: return HTMLResponse("")
    return HTMLResponse(f"""<div id="ath-ci-{cid}" style="padding:.25rem .4rem;display:flex;gap:.25rem"><form hx-post="{_u("conv","rename",cid)}" hx-target="#ath-left" hx-swap="innerHTML" style="display:flex;gap:.25rem;width:100%"><input type="text" name="value" value="{_esc(conv.get("title",""))}" class="module-select" style="flex:1;font-size:.75rem" autofocus onclick="event.stopPropagation()"><button type="submit" class="btn-icon" onclick="event.stopPropagation()">&#x2713;</button></form></div>""")

@router.post("/msg/delete")
async def msg_delete(request:Request):
    form = await request.form()
    mid = form.get("id", "")
    user = request.state.user
    conv, idx, m=_find_msg(user.username,mid)
    if conv and m: conv["messages"][idx]["deleted"]=True; _save_conv(conv)
    return HTMLResponse("")

@router.get("/msg/edit_form/{mid}")
async def msg_edit_form(mid:str, request:Request):
    user=request.state.user; conv,idx,m=_find_msg(user.username,mid)
    if not conv or not m: return HTMLResponse("")
    role_cls="cm-me" if m.get("role")=="user" else "cm-other"; avatar=CM._avatar_html(m.get("user_name","?"))
    return HTMLResponse(f"""<div class="cm-msg {role_cls}" id="cm-msg-{mid}" data-msg-id="{mid}">{avatar}<div class="cm-bwrap" style="max-width:90%"><form hx-post="{_u("msg","edit_save",mid)}" hx-target="#cm-msg-{mid}" hx-swap="outerHTML" style="display:flex;flex-direction:column;gap:.3rem;width:100%"><textarea name="content" class="cm-input" style="min-height:4rem;overflow-y:auto">{_esc(m.get("content",""))}</textarea><div style="display:flex;gap:.3rem"><button type="submit" class="button" style="font-size:.75rem;margin-top:0">Save</button><button type="button" class="btn-icon" hx-get="{_u("msg","cancel_edit",mid)}" hx-target="#cm-msg-{mid}" hx-swap="outerHTML">Cancel</button></div></form></div></div>""")

@router.post("/msg/edit_save/{mid}")
async def msg_edit_save(mid:str, request:Request):
    form=await request.form(); user=request.state.user; conv,idx,m=_find_msg(user.username,mid)
    if not conv or not m: return HTMLResponse("")
    conv["messages"][idx]["content"]=form.get("content","").strip(); conv["messages"][idx]["edited"]=True; _save_conv(conv)
    is_me=m.get("role")=="user"
    return HTMLResponse(CM.render_message(conv["messages"][idx],is_me=is_me,can_delete=True,can_edit=is_me))

@router.get("/msg/cancel_edit/{mid}")
async def msg_cancel_edit(mid:str, request:Request):
    user=request.state.user; conv,_,m=_find_msg(user.username,mid)
    if not conv or not m: return HTMLResponse("")
    is_me=m.get("role")=="user"
    return HTMLResponse(CM.render_message(m,is_me=is_me,can_delete=True,can_edit=is_me))

@router.post("/msg/retry/{mid}")
async def msg_retry(mid:str, request:Request):
    user=request.state.user; conv,idx,m=_find_msg(user.username,mid)
    if not conv or not m: return HTMLResponse("")
    role_cls="cm-me" if m.get("role")=="user" else "cm-other"; avatar=CM._avatar_html(m.get("user_name","?"))
    return HTMLResponse(f"""<div class="cm-msg {role_cls}" id="cm-msg-{mid}" data-msg-id="{mid}">{avatar}<div class="cm-bwrap" style="max-width:90%"><form hx-post="{_u("msg","retry_send",mid)}" hx-target="#cm-msgs-{conv["id"]}" hx-swap="outerHTML" style="display:flex;flex-direction:column;gap:.3rem;width:100%"><textarea name="content" class="cm-input" style="min-height:4rem">{_esc(m.get("content",""))}</textarea><div style="display:flex;gap:.3rem"><button type="submit" class="button" style="font-size:.75rem;margin-top:0">&#x21BA; Retry</button><button type="button" class="btn-icon" hx-get="{_u("msg","cancel_edit",mid)}" hx-target="#cm-msg-{mid}" hx-swap="outerHTML">Cancel</button></div></form></div></div>""")

@router.post("/msg/retry_send/{mid}")
async def msg_retry_send(mid:str, request:Request):
    form=await request.form(); user=request.state.user; new_content=form.get("content","").strip()
    conv,idx,m=_find_msg(user.username,mid)
    if not conv or not m: return HTMLResponse("")
    conv["messages"][idx]["content"]=new_content; conv["messages"][idx]["edited"]=True
    conv["messages"]=conv["messages"][:idx+1]; _save_conv(conv); sid=conv["id"]
    remaining="".join(CM.render_message(msg,is_me=(msg.get("role")=="user"),can_delete=True,can_edit=(msg.get("role")=="user")) for msg in conv["messages"] if not msg.get("deleted"))
    asyncio.create_task(_do_stream(user.username,{"content":new_content,"cid":sid},sid))
    return HTMLResponse(f'<div id="cm-msgs-{sid}" class="cm-msgs" data-pinned="true" hx-swap-oob="outerHTML">{remaining}</div>')

@router.get("/folder/new_form")
async def folder_new_form(): return HTMLResponse(f"""<form hx-post="{_u("folder","create")}" hx-target="#ath-left" hx-swap="innerHTML" style="display:flex;gap:.3rem;padding:.3rem .5rem;border-bottom:var(--border-thick) solid var(--border)"><input type="text" name="name" class="module-select" placeholder="Folder name" style="flex:1;font-size:.75rem" required><button type="submit" class="btn-icon">&#x2713;</button><button type="button" class="btn-icon" hx-get="{_u("folder","cancel")}" hx-target="#ath-folder-new" hx-swap="innerHTML">&#x2715;</button></form>""")

@router.get("/folder/cancel")
async def folder_cancel(): return HTMLResponse("")

@router.post("/folder/create")
async def folder_create(request:Request):
    form=await request.form(); name=form.get("name","").strip(); user=request.state.user
    if not name: return HTMLResponse(_left(user.username,""))
    org=_org(user.username); fid=f"f_{uuid.uuid4().hex[:8]}"
    org.setdefault("folders",{})[fid]={"name":name,"order":len(org.get("folders",{}))}; _save_org(user.username,org)
    active=await ENV["get_state"](request,scope="user",namespace="athena",key="active_conv_id")
    return HTMLResponse(_left(user.username,active or ""))

@router.post("/folder/delete/{fid}")
async def folder_delete(fid:str, request:Request):
    user=request.state.user; org=_org(user.username)
    org.get("folders",{}).pop(fid,None)
    for cid,f in list(org.get("conv_folders",{}).items()):
        if f==fid: org["conv_folders"].pop(cid)
    _save_org(user.username,org)
    active=await ENV["get_state"](request,scope="user",namespace="athena",key="active_conv_id")
    return HTMLResponse(_left(user.username,active or ""))

@router.get("/folder/rename_form/{fid}")
async def folder_rename_form(fid:str): return HTMLResponse(f"""<span id="ath-fn-{fid}" style="display:inline-flex;align-items:center;gap:.2rem"><form hx-post="{_u("folder","rename",fid)}" hx-target="#ath-left" hx-swap="innerHTML" style="display:inline-flex;gap:.2rem" onclick="event.stopPropagation()"><input type="text" name="name" class="module-select" style="font-size:.7rem;width:8rem" autofocus><button type="submit" class="btn-icon">&#x2713;</button></form></span>""")

@router.post("/folder/rename/{fid}")
async def folder_rename(fid:str, request:Request):
    form=await request.form(); name=form.get("name","").strip(); user=request.state.user
    org=_org(user.username)
    if name and fid in org.get("folders",{}): org["folders"][fid]["name"]=name; _save_org(user.username,org)
    active=await ENV["get_state"](request,scope="user",namespace="athena",key="active_conv_id")
    return HTMLResponse(_left(user.username,active or ""))

@router.post("/folder/assign/{cid}")
async def folder_assign(cid:str, request:Request):
    form=await request.form(); fid=form.get("folder_id",""); user=request.state.user
    org=_org(user.username); org.setdefault("conv_folders",{})[cid]=fid if fid else None; _save_org(user.username,org)
    active=await ENV["get_state"](request,scope="user",namespace="athena",key="active_conv_id")
    return HTMLResponse(_left(user.username,active or ""))

@router.get("/prompts/dropdown")
async def get_prompt_dropdown():
    prompts = _list_prompts()
    options = "".join(f'<option value="{p["id"]}">{_esc(p["name"])}</option>' for p in prompts)
    return HTMLResponse(f"""<select name="system_prompt_id" class="ath-dropdown"><option value="">-- Select Saved Prompt --</option>{options}</select>""")

@router.post("/pipeline/{sid}")
async def pipeline(sid:str, request:Request):
    form=await request.form(); conv=_load_conv(sid)
    if not conv: return HTMLResponse("")
    pl=form.get("pipeline","")
    pipe=next((p for p in cfg.get("pipelines",[]) if p.get("name")==pl),None)
    conv["pipeline"]=pl; conv["system_prompt"]=pipe.get("system_prompt","") if pipe else cfg.get("system_prompt",""); _save_conv(conv)
    return HTMLResponse(f'<span style="color:#00ffa2;font-size:.68rem">Pipeline: {_esc(pipe["label"] if pipe else "General")}</span>')

@router.post("/admin/save")
async def save(request: Request):
    form = dict(await request.form())
    group = cfg.get_group("general")
    group.save(form)
    return HTMLResponse("""<div id="settings-modal-container" hx-swap-oob="true"></div><div id="status" hx-swap-oob="true">Saved successfully.</div>""")

@router.get("/admin")
async def admin(request: Request):
    if getattr(request.state.user, "role", "") != "admin": return HTMLResponse("Denied")
    all_data = cfg.get_all()
    group = cfg.get_group("general")
    return HTMLResponse(f"""<div style="padding:0.5rem;position:relative">
        <button type="button" class="close-btn" style="position:absolute;top:.2rem;right:.2rem" hx-get="{_u()}" hx-target="#ait-workspace" hx-swap="innerHTML">&#x2715;</button>
        <form hx-post="{_u("admin","save")}" hx-target="#status">
            <div id="athena-admin-fields">{group.render(all_data.get("general", {}))}</div>
            <button type="submit" class="button" style="margin-top:1rem;">Save Settings</button>
            <div id="status" style="margin-top:0.5rem; font-size:0.75rem; color:#00ffa2;"></div>
        </form>
    </div>""")

@router.get("/admin/fields")
async def admin_fields(request: Request):
    """Target endpoint fired by HTMX change events to dynamically calculate transient field lists."""
    if getattr(request.state.user, "role", "") != "admin": return HTMLResponse("Denied")
    all_data = cfg.get_all()
    values = all_data.get("general", {})
    # Intercept live UI changes from HTMX query params before submission saves them
    live_conn_id = request.query_params.get("conn_id")
    if live_conn_id is not None: values["conn_id"] = live_conn_id
    group = cfg.get_group("general")
    return HTMLResponse(group.render(values))

@router.post("/upload/{cid}")
async def upload_file(cid: str, request: Request):
    user = request.state.user
    conv = _load_conv(cid)
    if not conv or conv.get("username") != user.username: return HTMLResponse("Unauthorized", status_code=403)
    form = await request.form()
    files = form.getlist("files")
    up_dir = _uploads_dir(cid)
    for f in files:
        if not f.filename: continue
        fid = uuid.uuid4().hex[:8]
        ext = Path(f.filename).suffix
        save_path = up_dir / f"{fid}{ext}"
        with open(save_path, "wb") as out:
            shutil.copyfileobj(f.file, out)
        conv.setdefault("attached_files", []).append({ "id": fid, "name": f.filename, "path": str(save_path), "ext": ext})
    _save_conv(conv)
    return HTMLResponse("".join(f"""<span style="background:var(--glass);padding:.1rem .3rem;border-radius:.3rem; display:inline-flex;align-items:center;gap:.2rem">{_esc(f["name"])}<button onclick="athDelFile('{cid}','{f["id"]}')" style="background:none;border:none;cursor:pointer;color:#ff5f5f;font-size:.8rem;padding:0">&#x2715;</button></span>""" for f in conv.get("attached_files",[])))

    for f in files:
        if not f.filename: continue
        fid = uuid.uuid4().hex[:8]
        ext = Path(f.filename).suffix
        save_path = up_dir / f"{fid}{ext}"
        with open(save_path, "wb") as out:
            shutil.copyfileobj(f.file, out)
        conv.setdefault("attached_files", []).append({ "id": fid, "name": f.filename, "path": str(save_path), "ext": ext})
    _save_conv(conv)
    return HTMLResponse("".join(f"""<span style="background:var(--glass);padding:.1rem .3rem;border-radius:.3rem; display:inline-flex;align-items:center;gap:.2rem">{_esc(f["name"])}<button onclick="athDelFile('{cid}','{f["id"]}')" style="background:none;border:none;cursor:pointer;color:#ff5f5f;font-size:.8rem;padding:0">&#x2715;</button></span>""" for f in conv.get("attached_files",[])))

@router.post("/delete_file/{cid}/{fid}")
async def delete_file(cid: str, fid: str, request: Request):
    user = request.state.user
    conv = _load_conv(cid)
    if not conv or conv.get("username") != user.username: return HTMLResponse("")
    files = conv.get("attached_files", [])
    conv["attached_files"] = [f for f in files if f["id"] != fid]
    for f in files:
        if f["id"] == fid:
            p = Path(f["path"])
            if p.exists(): p.unlink()
            break
    _save_conv(conv)
    return HTMLResponse("".join(f"""<span style="background:var(--glass);padding:.1rem .3rem;border-radius:.3rem; display:inline-flex;align-items:center;gap:.2rem">{_esc(f["name"])}<button onclick="athDelFile('{cid}','{f["id"]}')" style="background:none;border:none;cursor:pointer;color:#ff5f5f;font-size:.8rem;padding:0">&#x2715;</button></span>""" for f in conv.get("attached_files",[])))


def right_panel() -> str: return (f"""<div class="ait-rp"><div class="ait-rp-hd">Athena</div><div style="font-size:.72rem;color:var(--text_muted);padding:.2rem .2rem .4rem">{_esc(cfg.get("title","Athena"))}</div><button class="ait-rp-btn" hx-post="{_u("new")}" hx-target="#ait-workspace" hx-swap="innerHTML">+ New Conversation</button><button class="ait-rp-btn" hx-get="{_u("admin")}" hx-target="#ait-workspace" hx-swap="innerHTML">&#x2699; Admin Settings</button><div class="ait-rp-hd" style="margin-top:.5rem">Model</div><div style="font-size:.7rem;color:var(--text_muted);font-family:var(--font-mono);padding:.1rem .2rem">{_esc(cfg.get("model","not configured"))}</div></div>""")

EXTRA_JS = """
function athUpload(input,sid){var fd=new FormData(); for(var i=0;i<input.files.length;i++) fd.append('files',input.files[i]); fetch('/module/ai_tools/athena/upload/'+sid,{method:'POST', body:fd}).then(r=>r.text()).then(html=>htmx.process(htmx.swap(document.getElementById('ath-files-'+sid), 'innerHTML', html))); input.value='';}
function athDelFile(sid,fid){ htmx.ajax('POST','/module/ai_tools/athena/delete_file/'+sid+'/'+fid,{target:'#ath-files-'+sid,swap:'innerHTML'}); }
"""

CSS = """
.ath-sb-hdr{padding:.35rem .5rem;border-bottom:var(--border-thick) solid var(--border);display:flex;align-items:center;gap:.3rem;flex-shrink:0;}
.ath-conv-list{flex:1;overflow-y:auto;}
.ath-conv-item{padding:.38rem .6rem;cursor:pointer;border-bottom:var(--border-thick) solid var(--border);font-size:.78rem;display:flex;flex-direction:column;gap:.06rem;transition:background .12s;}
.ath-conv-item:hover{background:var(--accent_dim);}
.ath-conv-active{background:var(--glass);border-left:.15rem solid var(--accent);}
.ath-conv-title{display:block;overflow:hidden;text-overflow:ellipsis;white-space:nowrap;font-weight:500;}
.ath-folder{border-bottom:var(--border-thick) solid var(--border);}
.ath-folder-sum{padding:.3rem .5rem;cursor:pointer;font-size:.72rem;color:var(--text_muted);list-style:none;display:flex;align-items:center;gap:.3rem;user-select:none;}
.ath-folder-sum::-webkit-details-marker{display:none;}
.ath-folder-sum:hover{color:var(--accent);}
.ath-folder-sel{background:var(--bg);color:var(--text);border:var(--border-thick) solid var(--border);border-radius:.2rem;font-size:.6rem;padding:.08rem .2rem;flex:1;min-width:0;}
.ath-ungrouped-hdr{padding:.3rem .5rem;font-size:.63rem;text-transform:uppercase;letter-spacing:.05em;color:var(--text_muted);border-top:var(--border-thick) solid var(--border);margin-top:.3rem;}
.ath-conv-actions{opacity:0;transition:opacity .15s;}
.ath-conv-item:hover .ath-conv-actions,.ath-conv-item.ath-conv-active .ath-conv-actions{opacity:1;}
.ath-conv-item{padding:.45rem .5rem;min-height:2.4rem;}
"""
