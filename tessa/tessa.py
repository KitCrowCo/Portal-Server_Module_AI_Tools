# /modules/ai_tools/tessa/tessa.py
"""
Tessa - AI Document and Pipeline Workspace
Sub-module of ai_tools. Mounted at /module/ai_tools/tessa.
Data at data/ai_tools/tessa/. Shared knowledge at data/ai_tools/_knowledge/.
"""
import asyncio, json, uuid, pathlib
from datetime import datetime
from pathlib import Path
import httpx
from fastapi import APIRouter, Request, Form, UploadFile, File
from fastapi.responses import HTMLResponse
from modules.ai_tools.ai_utils import (get_conn, list_conns, list_models_sync, list_models_async, _base, tok_estimate, conn_opts_html, model_opts_html, KG_DIR)

TOOL_META = {"label": "Tessa", "icon": "&#x1F4C4;", "description": "AI document workspace and pipeline builder", "singleton": True} #"persistence": "user"}

router = APIRouter(redirect_slashes=False)

_P = "/module/ai_tools/tessa"
DATA_DIR = Path("./data/ai_tools/tessa")
PROJ_DIR = DATA_DIR / "projects"

ENV = {}
UI = WS = IM = CM = _SETTINGS = None
_ACTIVE: set = set()
_STOP: dict = {}
_PIPE_TASKS: dict = {}
_STREAM_TASKS: dict = {}

S_COL = {"idle":"var(--text_muted)", "running":"#00ffa2", "paused":"#ffcc00", "done":"#3d9aff", "error":"#ff5f5f"}

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

# --- Connections (shared with ai_tools) ---

async def _stream(conn, messages, model, num_ctx, think=False):
    pl = {"model":model,"messages":messages,"stream":True,"options":{"num_ctx":num_ctx,"num_predict":4096}}
    if think: pl["think"] = True
    tb = ""
    try:
        async with httpx.AsyncClient(timeout=httpx.Timeout(connect=5.0, read=3000.0, write=5.0, pool=5.0)) as c:
            async with c.stream("POST", f"{_base(conn)}/api/chat", json=pl) as resp:
                if resp.status_code != 200: yield "", "", True, f"HTTP {resp.status_code}"; return
                async for line in resp.aiter_lines():
                    if not line: continue
                    try:
                        chunk = json.loads(line); msg = chunk.get("message",{})
                        tb += msg.get("thinking",""); text = msg.get("content",""); done = chunk.get("done",False)
                        if text or done or tb: yield text, tb, done, None
                        if done: return
                    except: continue
    except asyncio.CancelledError: yield "", tb, True, None
    except Exception as e: yield "", tb, True, str(e)

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
            try: parts.append(f"--- {rel} ---\n{p.read_text(encoding='utf-8',errors='ignore')[:8000]}")
            except: pass
    return "\n\n".join(parts)

def _chunk_text(text, max_tokens):
    if max_tokens <= 0 or _tok(text) <= max_tokens: return [text]
    chars = max_tokens * 4; chunks = []
    while len(text) > chars:
        split = text.rfind("\n", 0, chars)
        if split <= 0: split = chars
        chunks.append(text[:split])
        text = text[split:].lstrip("\n")
    if text: chunks.append(text)
    return chunks

# --- Init ---

def get_model_options(values=None):
    conn = get_conn((values or {}).get("conn_id",""))
    return [(m, m) for m in list_models_sync(conn)] if conn else []

def init_tool(env:dict, prefix:str):
    global ENV, UI, WS, IM, CM, _SETTINGS
    ENV = env
    UI = env["templates"].env.globals.get("UI")
    WS = env["ws"]
    for d in (PROJ_DIR, KG_DIR, DATA_DIR/"versions"): d.mkdir(parents=True, exist_ok=True)
    bi = env["tools"]["built_ins"]
    conn_opts = [("","(none)")] + [(c["_id"], c.get("display_name",c["_id"])) for c in list_conns()]
    _SETTINGS = bi.SettingsPanel("Tessa", [
        bi.SettingsGroup("defaults", "Defaults", [
            bi.SettingField("title", "Title", "text", "Tessa"),
            bi.SettingField("conn_id", "Default Connection", "select", options=conn_opts),
            bi.SettingField("model", "Default Model", "select", options=get_model_options),
            bi.SettingField("model_ctx", "Context Tokens", "number", 32768),
            bi.SettingField("system_prompt", "Default System Prompt", "textarea", "You are a helpful AI assistant."),
            bi.SettingField("chunk_tokens", "Default Chunk Tokens", "number", 6000),
            bi.SettingField("auto_snapshot", "Auto-snapshot on save", "checkbox", False),
        ], json_path=str(DATA_DIR / "settings.json")),
    ])
    IM = env["InterfaceManager"](nesting_level=2, db_path="tessa_im.db")
    CM = bi.ChatManager(namespace="tessa", base_url=_u(), view_style="bubble", stream_toggle=True, think_toggle=True, stop_enabled=True, pin_enabled=True, allow_edit=True, allow_delete=True, allow_copy=True, show_info=False, markdown_mode="standard", placeholder="Chat about this project\u2026 (Ctrl+Enter)", branch_id=IM.branch_id, nesting_level=2)
    IM.scripts["submit"] = [_handle_submit]
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
        conn = get_conn(doc.get("conn_id","")); model = doc.get("model","")
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

# --- Pipeline Execution ---

async def _run_pipeline(pid, pl_id, username):
    doc = _load(pid)
    if not doc: return
    pl = next((p for p in doc.get("pipelines",[]) if p["id"]==pl_id), None)
    if not pl: return
    pl["state"] = "running"; _save(doc)

    async def _push():
        d2 = _load(pid)
        if not d2: return
        p2 = next((p for p in d2.get("pipelines",[]) if p["id"]==pl_id), None)
        if not p2: return
        card = f'<div id="tessa-pipeline-{pl_id}" hx-swap-oob="outerHTML">{_pipeline_card(d2, p2)}</div>'
        bar  = f'<div id="tessa-pipe-bar" hx-swap-oob="innerHTML">{_pipe_progress_html(d2, pl_id)}</div>'
        await WS.send_personal_message(card + bar, username)

    for idx in range(pl.get("current_step",0), len(pl.get("steps",[]))):
        doc = _load(pid); pl = next((p for p in doc.get("pipelines",[]) if p["id"]==pl_id), None)
        if not pl or pl.get("state") != "running": break
        if _STOP.pop(f"pl_{pl_id}", False): pl["state"] = "paused"; _save(doc); await _push(); return
        step = pl["steps"][idx]; step["state"] = "running"; pl["current_step"] = idx
        _save(doc); await _push()
        ok = await _exec_step(pid, pl_id, step, username)
        doc = _load(pid); pl = next((p for p in doc.get("pipelines",[]) if p["id"]==pl_id), None)
        if not pl: break
        if not ok:
            if step.get("state") != "paused": step["state"] = "error"
            _save(doc); await _push(); break
        step["state"] = "done"; pl["current_step"] = idx+1; _save(doc); await _push()
        if step.get("pause_after") and idx+1 < len(pl.get("steps",[])):
            pl["state"] = "paused"; _save(doc); await _push(); return

    _STOP.pop(f"pl_{pl_id}", None)
    doc = _load(pid); pl = next((p for p in doc.get("pipelines",[]) if p["id"]==pl_id), None)
    if pl and pl.get("state") == "running":
        pl["state"] = "paused" if any(s.get("state")=="paused" for s in pl.get("steps",[])) else "done"
        _save(doc)
    await _push()
    _PIPE_TASKS.pop(pl_id, None)
    doc = _load(pid)
    if doc:
        ed = ENV["tools"]["built_ins"].PortalEditor(base_url=_u())
        await WS.send_personal_message(f'<div id="tessa-center" hx-swap-oob="innerHTML">{ed.render_shell(doc)}</div>', username)

async def _exec_step(pid, pl_id, step, username):
    doc = _load(pid)
    conn = get_conn(step.get("conn_id","") or doc.get("conn_id",""))
    model = step.get("model","") or doc.get("model","")
    if not conn or not model: step["state"] = "error"; return False

    num_ctx = step.get("model_ctx", doc.get("model_ctx", 32768))
    chunk_tokens = step.get("chunk_tokens", 6000)
    sys_p = step.get("system_prompt","").strip()
    tpl = step.get("user_template","").strip() or "{chunk_content}"
    sep = step.get("output_separator","\n\n---\n\n")
    prog = step.setdefault("progress", {"file_queue":[],"files_done":[],"current_file":"","log":[]})

    def _save_prog():
        d = _load(pid)
        pl = next((p for p in d.get("pipelines",[]) if p["id"]==pl_id), None)
        if pl:
            s = next((s for s in pl.get("steps",[]) if s.get("id")==step.get("id")), None)
            if s: s["progress"] = prog
            _save(d)

    if step.get("type") == "synthesis":
        doc = _load(pid); content = doc.get("content","").strip()
        if not content: step["state"] = "done"; return True
        chunks = _chunk_text(content, chunk_tokens)
        for ci, chunk in enumerate(chunks):
            if _STOP.get(f"pl_{pl_id}"): step["state"] = "paused"; return False
            user_msg = tpl.replace("{chunk_content}",chunk).replace("{chunk_number}",str(ci+1)).replace("{chunks_total}",str(len(chunks)))
            msgs = ([] if not sys_p else [{"role":"system","content":sys_p}]) + [{"role":"user","content":user_msg}]
            full = ""
            async for text, _tb, done, err in _stream(conn, msgs, model, num_ctx):
                if _STOP.get(f"pl_{pl_id}"): step["state"] = "paused"; return False
                if err: prog["log"].append(f"Err chunk {ci+1}: {err[:60]}"); break
                if text: full += text
                if full: await WS.send_personal_message(f"""<div id="tessa-pl-stream-{pl_id}" hx-swap-oob="innerHTML"><code style="font-size:.63rem">{_esc(full[-200:])}</code></div><div id="tessa-pl-stream-active" hx-swap-oob="innerHTML"><code style="font-size:.63rem">{_esc(full[-400:])}</code></div>""", username)
                if done: break
            if full:
                doc = _load(pid)
                doc["content"] += f"\n\n{full.strip()}{sep}"
                _save(doc)
        step["state"] = "done"
        return True

    # file_pass
    if not prog.get("file_queue") and not prog.get("files_done"):
        if step.get("use_selected"):
            doc_cur = _load(pid)
            prog["file_queue"] = sorted(f for f in (doc_cur.get("selected_files",[]) if doc_cur else []) if (KG_DIR/f).is_file())
        else:
            src = step.get("input_source","").strip("/")
            base = KG_DIR/src if src else KG_DIR
            prog["file_queue"] = sorted(str(f.relative_to(KG_DIR)) for f in base.rglob("*") if f.is_file() and not f.name.startswith(".")) if base.exists() else []
        if not prog["file_queue"]:
            step["state"] = "done"; _save_prog(); return True

    while prog.get("file_queue"):
        if _STOP.pop(f"pl_{pl_id}", False): step["state"] = "paused"; _save_prog(); return False
        cur = prog["file_queue"][0]
        prog["current_file"] = cur; prog["current_chunk"] = 0; prog["chunks_total"] = 0
        _save_prog()
        try: file_content = (KG_DIR/cur).read_text(encoding="utf-8", errors="ignore")
        except Exception as e: prog["log"].append(f"Read error {cur}: {e}"); prog["file_queue"].pop(0); continue

        chunks = _chunk_text(file_content, chunk_tokens)
        prog["chunks_total"] = len(chunks); _save_prog()

        for ci, chunk in enumerate(chunks):
            if _STOP.pop(f"pl_{pl_id}", False): step["state"] = "paused"; _save_prog(); return False
            prog["current_chunk"] = ci + 1; _save_prog()
            user_msg = (tpl.replace("{file_name}", pathlib.Path(cur).name).replace("{file_path}", cur)
                           .replace("{chunk_number}", str(ci+1)).replace("{chunks_total}", str(len(chunks)))
                           .replace("{chunk_content}", chunk))
            msgs = ([] if not sys_p else [{"role":"system","content":sys_p}]) + [{"role":"user","content":user_msg}]
            full = ""
            async for text, _tb, done, err in _stream(conn, msgs, model, num_ctx):
                if _STOP.get(f"pl_{pl_id}"): step["state"] = "paused"; _save_prog(); return False
                if err: prog["log"].append(f"Err {cur} c{ci+1}: {err[:60]}"); break
                if text: full += text
                if full: await WS.send_personal_message(
                    f"""<div id="tessa-pl-stream-{pl_id}" hx-swap-oob="innerHTML"><code style="font-size:.63rem">{_esc(full[-200:])}</code></div>""", username)
                if done: break
            if full:
                doc = _load(pid)
                doc["content"] += f"\n\n<!-- {_esc(step.get('name','step'))} | {_esc(cur)} chunk {ci+1}/{len(chunks)} -->\n{full.strip()}{sep}"
                _save(doc)
            prog["log"] = (prog["log"] + [f"{cur} c{ci+1}/{len(chunks)} +{_tok(full)}t"])[-30:]

        prog["files_done"].append(prog["file_queue"].pop(0))
        prog["current_file"] = ""; prog["current_chunk"] = 0; prog["chunks_total"] = 0
        _save_prog()
        d = _load(pid)
        if d: await WS.send_personal_message(f'<div id="tessa-pipe-bar" hx-swap-oob="innerHTML">{_pipe_progress_html(d, pl_id)}</div>', username)

    step["state"] = "done"
    return True
    
# --- HTML Builders ---

def _proj_list_html(username, active_id=""):
    projects = _list_projects(username)
    if not projects: return '<div style="color:var(--text_muted);font-size:.8rem;padding:.5rem">No projects yet.</div>'
    out = ""
    for p in projects:
        pid = p["id"]; title = _esc((p.get("title","") or "Untitled")[:42]); date = (p.get("modified","") or "")[:10]
        act = "background:var(--glass);border-left:.15rem solid var(--accent);" if pid==active_id else ""
        out += (f"""<div id="tessa-pi-{pid}" style="padding:.38rem .6rem;cursor:pointer;border-bottom:var(--border-thick) solid var(--border);font-size:.78rem;{act}" hx-get="{_u("load",pid)}" hx-target="#tessa-center" hx-swap="innerHTML"><div style="display:flex;align-items:center;gap:.2rem"><span style="flex:1;font-weight:500;overflow:hidden;text-overflow:ellipsis;white-space:nowrap">{title}</span><button class="cm-qbtn" style="color:#ff5f5f" hx-delete="{_u("project",pid)}" hx-target="#tessa-proj-list" hx-swap="innerHTML" hx-confirm="Delete '{title}'?" onclick="event.stopPropagation()">&#x2715;</button></div><div style="font-size:.6rem;color:var(--text_muted)">{date}</div></div>""")
    return out

def _conn_bar_html(doc, conns, models):
    pid = doc["id"]; cid = doc.get("conn_id",""); mdl = doc.get("model",""); ctx = doc.get("model_ctx",32768)
    c_opts = conn_opts_html(cid) or '<option value="">No connections</option>'
    m_opts = "".join(f'<option value="{m}" {"selected" if m==mdl else ""}>{m}</option>' for m in models) or model_opts_html(cid, mdl)
    return (f"""<div style="display:flex;align-items:center;gap:.4rem;height:100%;padding:0 .5rem;overflow:hidden;"><select class="module-select" style="font-size:.72rem;max-width:8rem;flex-shrink:0" name="conn_id" hx-post="{_u("doc/conn",pid)}" hx-trigger="change" hx-target="#tessa-model-wrap" hx-swap="innerHTML" hx-include="[name=conn_id]">{c_opts}</select><div id="tessa-model-wrap" style="flex-shrink:0"><select class="module-select" style="font-size:.72rem;max-width:11rem" name="model" hx-post="{_u("doc/model",pid)}" hx-trigger="change" hx-include="[name=model]" hx-swap="none">{m_opts}</select></div><label style="font-size:.63rem;color:var(--text_muted);white-space:nowrap;flex-shrink:0">ctx <input type="number" name="model_ctx" value="{ctx}" min="512" max="262144" class="module-select" style="width:4.5rem;font-size:.65rem;padding:.2rem .3rem" hx-post="{_u("doc/ctx",pid)}" hx-trigger="change" hx-include="[name=model_ctx]" hx-swap="none"></label><button class="btn-icon" style="font-size:.65rem;flex-shrink:0;margin-left:auto" hx-get="{_u("settings")}" hx-target="#tessa-center" hx-swap="innerHTML" title="Tessa Settings">&#x2699;</button></div>""")

def _kg_html(doc):
    pid = doc["id"]; selected = set(doc.get("selected_files",[]))
    badge = f'<span style="background:var(--accent_dim);color:var(--accent);border-radius:.2rem;padding:.05rem .3rem;font-size:.6rem">{len(selected)}</span>' if selected else ""
    tree = UI.tree(items=KG_DIR, mode="file", selectable=True, selected=selected, post_url=_u("files/toggle", pid), target="#tessa-kg-section", swap="outerHTML") if KG_DIR.exists() else '<div style="font-size:.7rem;color:var(--text_muted);padding:.2rem .4rem">No knowledge files yet.</div>'
    return (f"""<div id="tessa-kg-section" style="border-top:var(--border-thick) solid var(--border)"><details><summary style="padding:.3rem .45rem;cursor:pointer;font-size:.7rem;text-transform:uppercase;letter-spacing:.05em;color:var(--text_muted);list-style:none;user-select:none;display:flex;align-items:center;gap:.3rem">&#x1F4DA; Knowledge {badge}</summary><div style="max-height:28vh;overflow-y:auto;padding:.25rem .4rem">{tree}</div><form hx-post="{_u("kg/upload",pid)}" hx-target="#tessa-kg-section" hx-swap="outerHTML" hx-encoding="multipart/form-data" style="padding:.2rem .4rem;border-top:var(--border-thick) solid var(--border)"><label class="btn-icon" style="cursor:pointer;font-size:.7rem;width:100%;justify-content:center" title="Upload knowledge files">&#x2B06; Upload files<input type="file" name="files" multiple style="display:none" onchange="this.closest('form').requestSubmit()"></label></form></details></div>""")

def _kg_dir_selector_html(selected="", base=None, depth=0):
    base = base or KG_DIR
    if not base.exists(): return '<div style="color:var(--text_muted);font-size:.7rem;padding:.3rem">No knowledge files.</div>'
    pad = f"padding-left:{.4 + depth*.8:.1f}rem"
    out = ""
    if depth == 0:
        sel_style = "background:var(--accent_dim);color:var(--accent);" if selected == "" else ""
        out += f'<div class="kg-dir-opt" data-path="" style="{sel_style}padding:.22rem .4rem;cursor:pointer;font-size:.74rem;border-bottom:var(--border-thick) solid var(--border)" onclick="setStepSrc(this,\'\')">/ All files</div>'
    for d in sorted([x for x in base.iterdir() if x.is_dir() and not x.name.startswith(".")], key=lambda x: x.name):
        rel = str(d.relative_to(KG_DIR))
        sel_style = "background:var(--accent_dim);color:var(--accent);" if rel == selected else ""
        fc = sum(1 for _ in d.rglob("*") if _.is_file())
        out += f'<div class="kg-dir-opt" data-path="{rel}" style="{sel_style}{pad};padding:.22rem .4rem;cursor:pointer;font-size:.74rem;border-bottom:var(--border-thick) solid var(--border)" onclick="setStepSrc(this,\'{rel}\')">\u25B6 {d.name}/ <span style="opacity:.5;font-size:.63rem">({fc})</span></div>'
        out += _kg_dir_selector_html(selected, d, depth+1)
    return out

def _pipe_progress_html(doc=None, pl_id=None):
    """Bottom bar content: compact when idle, detailed when running."""
    if not doc: return '<div style="padding:.2rem .8rem;font-size:.72rem;color:var(--text_muted);display:flex;align-items:center;gap:.5rem">&#x26A1; No active pipeline <span style="opacity:.4;font-size:.65rem">- run one from the left panel</span></div>'
    pls = doc.get("pipelines", [])
    running = [p for p in pls if p.get("state") in ("running", "paused")]
    if not running:
        done = [p for p in pls if p.get("state")=="done"]
        err  = [p for p in pls if p.get("state")=="error"]
        parts = []
        if done: parts.append(f'<span style="color:#3d9aff">&#x2713; {len(done)} done</span>')
        if err:  parts.append(f'<span style="color:#ff5f5f">&#x26A0; {len(err)} error</span>')
        return f'<div style="padding:.2rem .8rem;font-size:.72rem;color:var(--text_muted);display:flex;align-items:center;gap:.5rem">&#x26A1; Pipelines &mdash; {" ".join(parts) if parts else "all idle"}</div>'

    pl = next((p for p in running if p["id"]==pl_id), running[0])
    pid = doc["id"]; state = pl.get("state","idle"); col = S_COL.get(state,"var(--text_muted)")
    cur_step = next((s for s in pl.get("steps",[]) if s.get("state")=="running"), None)
    prog = (cur_step or {}).get("progress",{})
    f_done = len(prog.get("files_done",[])); f_q = len(prog.get("file_queue",[])); f_total = f_done+f_q
    chunk_n = prog.get("current_chunk",0); chunk_t = prog.get("chunks_total",0)
    cur_file = _esc(prog.get("current_file",""))
    log = prog.get("log",[])[-3:]
    log_html = "".join(f'<div style="overflow:hidden;text-overflow:ellipsis;white-space:nowrap;color:var(--text_muted);font-size:.63rem">{_esc(l)}</div>' for l in log)
    pct = int(f_done/max(f_total,1)*100) if f_total else 0
    step_name = _esc((cur_step or {}).get("name",""))
    chunk_info = f" | chunk {chunk_n}/{chunk_t}" if chunk_t else ""
    stop_btn = f'<button class="cm-qbtn" style="color:#ff4444;flex-shrink:0;font-size:.75rem" hx-post="{_u("pipeline/stop",pid,pl["id"])}" hx-swap="none" title="Stop">&#x25FC; Stop</button>' if state=="running" else f'<span style="font-size:.65rem;color:#ffcc00">Paused</span>'
    return (f"""<div style="display:flex;align-items:flex-start;gap:.8rem;padding:.25rem .8rem;height:100%;box-sizing:border-box;overflow:hidden">
        <div style="flex:0 0 14rem;display:flex;flex-direction:column;gap:.1rem;overflow:hidden">
            <div style="display:flex;align-items:center;gap:.3rem"><span style="font-weight:600;font-size:.78rem;overflow:hidden;text-overflow:ellipsis;white-space:nowrap">{_esc(pl.get("name",""))}</span>{stop_btn}</div>
            <div style="font-size:.68rem;color:{col}">{state}{f" - {step_name}" if step_name else ""}</div>
            <div style="font-size:.63rem;color:var(--text_muted)">{f_done}/{f_total} files{chunk_info}</div>
            {"<div style=height:.3rem;background:var(--border);border-radius:.15rem;margin-top:.1rem><div style=height:100%;width:"+str(pct)+r"%;background:#00ffa2;border-radius:.15rem></div></div>" if f_total else ""}
        </div>
        <div style="flex:0 0 16rem;display:flex;flex-direction:column;gap:.06rem;overflow:hidden;border-left:var(--border-thick) solid var(--border);padding-left:.6rem">
            <div style="font-size:.63rem;color:var(--text_muted)">Current file:</div>
            <div style="font-size:.68rem;font-family:var(--font-mono);overflow:hidden;text-overflow:ellipsis;white-space:nowrap">{cur_file or "(none)"}</div>
            {f'<div style="font-size:.6rem;color:var(--accent)">chunk {chunk_n} of {chunk_t}</div>' if chunk_t else ""}
        </div>
        <div style="flex:1;min-width:0;border-left:var(--border-thick) solid var(--border);padding-left:.6rem;overflow:hidden">
            <div style="font-size:.63rem;color:var(--text_muted);margin-bottom:.1rem">Log:</div>
            {log_html}
        </div>
        <div id="tessa-pl-stream-active" style="flex:0 0 18rem;border-left:var(--border-thick) solid var(--border);padding-left:.6rem;font-size:.63rem;font-family:var(--font-mono);overflow:hidden;max-height:6rem;color:var(--text_muted)"></div>
    </div>""")
    
def _pipeline_card(doc, pl):
    pid = doc["id"]; pl_id = pl["id"]; state = pl.get("state","idle"); col = S_COL.get(state,"var(--text_muted)")
    steps = pl.get("steps",[])
    dots = "".join(f'<span style="width:.4rem;height:.4rem;border-radius:50%;background:{S_COL.get(s.get("state","idle"),"var(--text_muted)")};display:inline-block;margin:.05rem" title="{_esc(s.get("name",""))}"></span>' for s in steps)
    cur = next((s for s in steps if s.get("state")=="running"), None)
    prog = cur.get("progress",{}) if cur else {}
    prog_txt = _esc(prog.get("current_file",""))
    f_done = len(prog.get("files_done",[])); f_q = len(prog.get("file_queue",[]))
    prog_html = f'<div style="font-size:.63rem;color:var(--text_muted);font-family:var(--font-mono)">{prog_txt}{"  "+str(f_done)+"/"+str(f_done+f_q) if (f_done+f_q) else ""}</div>' if prog_txt else ""
    log = (cur.get("progress",{}).get("log",[]) if cur else [])[-2:]
    log_html = "".join(f'<div style="font-size:.6rem;color:var(--text_muted);overflow:hidden;text-overflow:ellipsis;white-space:nowrap">{_esc(l)}</div>' for l in log)
    run_btn  = f'<button class="cm-qbtn" hx-post="{_u("pipeline/run",pid,pl_id)}" hx-swap="none" title="Run">&#x25B6;</button>' if state in ("idle","paused","done","error") else ""
    stop_btn = f'<button class="cm-qbtn" style="color:#ff4444" hx-post="{_u("pipeline/stop",pid,pl_id)}" hx-swap="none" title="Stop">&#x25FC;</button>' if state == "running" else ""
    edit_btn = f'<button class="cm-qbtn" hx-get="{_u("pipeline/edit",pid,pl_id)}" hx-target="#tessa-modal" hx-swap="innerHTML" title="{"View (stop to edit)" if state=="running" else "Edit pipeline"}">{"&#x1F441;" if state=="running" else "&#x270E;"}</button>'
    del_btn  = f'<button class="cm-qbtn" style="color:#ff5f5f" hx-delete="{_u("pipeline",pid,pl_id)}" hx-target="#tessa-pipeline-{pl_id}" hx-swap="outerHTML" hx-confirm="Delete?" title="Delete">&#x2715;</button>' if state != "running" else ""
    return (f"""<div id="tessa-pipeline-{pl_id}" style="padding:.45rem .55rem;border-bottom:var(--border-thick) solid var(--border)"><div style="display:flex;align-items:center;gap:.3rem"><span style="flex:1;font-size:.78rem;font-weight:500;overflow:hidden;text-overflow:ellipsis;white-space:nowrap">{_esc(pl.get("name",""))}</span><span style="font-size:.65rem;color:{col}">{state}</span>{run_btn}{stop_btn}{edit_btn}{del_btn}</div><div style="display:flex;gap:.08rem;margin:.1rem 0">{dots}</div>{prog_html}{log_html}<div id="tessa-pl-stream-{pl_id}" style="font-size:.63rem;color:var(--text_muted);max-height:2rem;overflow:hidden"></div></div>""")

def _pipelines_html(doc):
    pid = doc["id"]; pls = doc.get("pipelines",[])
    cards = "".join(_pipeline_card(doc, pl) for pl in pls) or '<div style="font-size:.72rem;color:var(--text_muted);padding:.3rem .4rem">No pipelines. Click + to create one.</div>'
    return (f"""<div id="tessa-pipelines-section" style="border-top:var(--border-thick) solid var(--border)"><details open><summary style="padding:.3rem .45rem;cursor:pointer;font-size:.7rem;text-transform:uppercase;letter-spacing:.05em;color:var(--text_muted);list-style:none;user-select:none;display:flex;align-items:center;gap:.3rem">&#x26A1; Pipelines<button class="btn-icon" style="margin-left:auto;font-size:.7rem" hx-get="{_u("pipeline/new_form",pid)}" hx-target="#tessa-pl-editor" hx-swap="innerHTML" onclick="event.stopPropagation()">+</button></summary><div>{cards}</div><div id="tessa-pl-editor"></div></details></div>""")

def _left_bottom_html(doc): return _kg_html(doc) + _pipelines_html(doc)

def _left_panel(username, doc):
    pid = doc["id"]
    return (f"""<div style="display:flex;flex-direction:column;height:100%;overflow:hidden"><div style="flex-shrink:0;padding:.35rem .5rem;border-bottom:var(--border-thick) solid var(--border);display:flex;align-items:center;gap:.3rem"><button class="btn-icon" style="font-size:1.1rem" hx-post="{_u("new")}" hx-target="#tessa-center" hx-swap="innerHTML" title="New project">+</button><span style="font-size:.68rem;text-transform:uppercase;letter-spacing:.05em;color:var(--text_muted);flex:1">Tessa</span><button class="btn-icon" style="font-size:.75rem" hx-get="{_u("settings")}" hx-target="#tessa-center" hx-swap="innerHTML" title="Settings">&#x2699;</button></div><div id="tessa-proj-list" style="flex:1;min-height:0;overflow-y:auto">{_proj_list_html(username, pid)}</div><div id="tessa-left-bottom" style="flex-shrink:0;overflow-y:auto;border-top:var(--border-thick) solid var(--border)">{_left_bottom_html(doc)}</div></div>""")

# --- Pipeline Edit HTML ---

def _pl_edit_html(doc, pl):
    pid = doc["id"]; pl_id = pl["id"]
    rows = "".join(_step_row(doc, pl, s, i) for i, s in enumerate(pl.get("steps",[])))
    empty_state = '' if rows else '<div style="font-size:.75rem;color:var(--text_muted);padding:.5rem .3rem">No steps yet. Click + below.</div>'
    return (f"""<div style="display:flex;flex-direction:column;height:100%;overflow:hidden">
                    <div style="flex-shrink:0;padding:.4rem .5rem;border-bottom:var(--border-thick) solid var(--border);display:flex;align-items:center;gap:.3rem">
                        <input type="text" value="{_esc(pl.get("name",""))}" class="module-select" style="flex:1;font-size:.78rem" name="name" hx-post="{_u("pipeline/rename",pid,pl_id)}" hx-trigger="change" hx-include="this" hx-swap="none" placeholder="Pipeline name">
                        <button class="btn-icon" style="font-size:.7rem" title="Reset all step progress" hx-post="{_u("pipeline/reset",pid,pl_id)}" hx-target="#tessa-pipelines-section" hx-swap="outerHTML" hx-confirm="Reset all progress?" onclick="document.getElementById('tessa-modal').style.display='none';document.getElementById('tessa-modal').innerHTML=''">&#x21BA;</button>
                    </div>
                    <div class="tessa-pl-layout" style="display:flex;flex:1;min-height:0;overflow:hidden">
                        <div class="tessa-pl-steps" style="flex:0 0 min(14rem,38%);border-right:var(--border-thick) solid var(--border);display:flex;flex-direction:column;overflow:hidden">
                            <div style="flex:1;overflow-y:auto">{empty_state}{rows}</div>
                            <div style="border-top:var(--border-thick) solid var(--border);padding:.3rem .4rem;flex-shrink:0">
                                <button class="btn-icon" style="font-size:.75rem;width:100%;justify-content:center" hx-get="{_u("pipeline/step_form",pid,pl_id)}" hx-target="#tessa-modal-step-editor" hx-swap="innerHTML">+ Add Step</button>
                            </div>
                        </div>
                        <div id="tessa-modal-step-editor" style="flex:1;overflow-y:auto">
                            <div style="padding:2rem;color:var(--text_muted);font-size:.82rem;text-align:center">Select a step to edit, or add one.</div>
                        </div>
                    </div>
                </div>""")

def _step_row(doc, pl, step, idx):
    pid = doc["id"]; pl_id = pl["id"]; sid = step["id"]; col = S_COL.get(step.get("state","idle"),"var(--text_muted)")
    prog = step.get("progress",{})
    f_done = len(prog.get("files_done",[]))
    f_q = len(prog.get("file_queue",[]))
    f_total = f_done+f_q
    count = f' <span style="font-size:.6rem;color:var(--text_muted)">({f_done}/{f_total})</span>' if f_total else ""
    return (f"""<div style="display:flex;align-items:center;gap:.3rem;padding:.22rem .4rem;border-bottom:var(--border-thick) solid var(--border);font-size:.75rem"><span style="width:.45rem;height:.45rem;border-radius:50%;background:{col};flex-shrink:0"></span><span style="flex:1;overflow:hidden;text-overflow:ellipsis;white-space:nowrap">{_esc(step.get("name",""))}{count}</span><span style="font-size:.6rem;color:var(--text_muted);flex-shrink:0">{step.get("type","")}</span><button class="btn-icon" style="font-size:.65rem" hx-get="{_u("pipeline/step_form",pid,pl_id,sid)}" hx-target="#tessa-modal-step-editor" hx-swap="innerHTML" title="Edit step">&#x270E;</button><button class="btn-icon" style="font-size:.65rem;color:#ff5f5f" hx-delete="{_u("pipeline/step",pid,pl_id,sid)}" hx-target="#tessa-modal-step-editor" hx-swap="innerHTML" hx-confirm="Remove step?" title="Remove">&#x2715;</button></div>""")

def _step_form_html(doc, pl_id, step=None):
    pid = doc["id"]; s = step or {}; sid = s.get("id",""); is_new = not sid
    action = _u("pipeline/step/add",pid,pl_id) if is_new else _u("pipeline/step/save",pid,pl_id,sid)
    c_opts = conn_opts_html(s.get("conn_id","") or doc.get("conn_id",""))
    m_opts = model_opts_html(s.get("conn_id","") or doc.get("conn_id",""), s.get("model",""))
    t_opts = "".join(f'<option value="{v}" {"selected" if v==s.get("type","file_pass") else ""}>{l}</option>' for v,l in [("file_pass","File Pass - process knowledge files"),("synthesis","Synthesis - restructure document content")])
    sep = (s.get("output_separator","\n\n---\n\n") or "\n\n---\n\n").replace("\n","\\n")
    nsep = "\n"
    del_btn = f'<button type="button" class="btn-icon" style="color:#ff5f5f" hx-delete="{_u("pipeline/step",pid,pl_id,sid)}" hx-target="#tessa-modal-step-editor" hx-swap="innerHTML" hx-confirm="Remove step?">Remove Step</button>' if not is_new else ""
    back_btn = f'<button type="button" class="btn-icon" hx-get="{_u("pipeline/edit",pid,pl_id)}" hx-target="#tessa-modal-inner" hx-swap="innerHTML">&#x2190; Back</button>'
    kg_tree = _kg_dir_selector_html(s.get("input_source",""))
    prompt_save_url = _u("pipeline/step/prompt/save"); prompt_list_url = _u("pipeline/step/prompts")
    return f"""<form id="step-edit-form" hx-post="{action}" hx-target="#tessa-modal-step-editor" hx-swap="innerHTML" style="display:flex;flex-direction:column;gap:.32rem;padding:.75rem">
        <div style="display:flex;align-items:center;gap:.3rem;margin-bottom:.2rem">{back_btn}<span style="font-size:.82rem;font-weight:600;color:var(--accent)">{"New Step" if is_new else "Edit: "+_esc(s.get("name",""))}</span></div>
        <input type="text" name="name" value="{_esc(s.get("name","New Step"))}" class="module-select" style="font-size:.78rem" required placeholder="Step name">
        <select name="type" class="module-select" style="font-size:.73rem">{t_opts}</select>
        <div style="display:flex;gap:.3rem">
            <div style="flex:1"><label style="font-size:.65rem;color:var(--text_muted)">Connection</label><select name="conn_id" class="module-select" style="font-size:.73rem" hx-post="{_u("pipeline/step_models",pid,pl_id)}" hx-trigger="change" hx-include="[name=conn_id]" hx-target="#tessa-step-model-wrap" hx-swap="innerHTML">{c_opts}</select></div>
            <div id="tessa-step-model-wrap" style="flex:1"><label style="font-size:.65rem;color:var(--text_muted)">Model</label><select name="model" class="module-select" style="font-size:.73rem">{m_opts}</select></div>
        </div>
        <div style="display:flex;gap:.3rem">
            <div style="flex:1"><label style="font-size:.65rem;color:var(--text_muted)">Context tokens</label><input type="number" name="model_ctx" value="{s.get("model_ctx",32768)}" class="module-select" style="font-size:.73rem"></div>
            <div style="flex:1"><label style="font-size:.65rem;color:var(--text_muted)">Chunk tokens (0=no split)</label><input type="number" name="chunk_tokens" value="{s.get("chunk_tokens",6000)}" class="module-select" style="font-size:.73rem"></div>
        </div>
        <label style="font-size:.65rem;color:var(--text_muted)">Knowledge Source
            <div style="max-height:8rem;overflow-y:auto;border:var(--border-thick) solid var(--border);border-radius:var(--radius);margin-top:.2rem">{kg_tree}</div>
            <input type="hidden" name="input_source" id="step-input-src" value="{_esc(s.get("input_source",""))}">
        </label>
        <label style="display:flex;align-items:center;gap:.35rem;font-size:.76rem"><input type="checkbox" name="use_selected" value="1" {"checked" if s.get("use_selected") else ""}> Use document's selected files only (overrides above)</label>
        <details style="border:var(--border-thick) solid var(--border);border-radius:var(--radius);padding:.4rem"><summary style="cursor:pointer;font-size:.72rem;color:var(--text_muted);list-style:none;user-select:none">&#x1F4BE; Saved Prompts</summary>
            <div id="step-prompt-list" hx-get="{prompt_list_url}" hx-trigger="load" hx-swap="innerHTML" style="margin-top:.3rem"></div>
            <div style="display:flex;gap:.3rem;margin-top:.35rem">
                <input type="text" id="step-prompt-name" placeholder="Save current as..." class="module-select" style="flex:1;font-size:.72rem">
                <button type="button" class="btn-icon" style="font-size:.75rem" onclick="saveStepPrompt('{prompt_save_url}','{prompt_list_url}')">Save</button>
            </div>
        </details>
        <label style="font-size:.65rem;color:var(--text_muted)">System Prompt<textarea name="system_prompt" class="module-select" rows="4" style="font-family:var(--font-mono);font-size:.72rem;resize:vertical">{_esc(s.get("system_prompt",""))}</textarea></label>
        <label style="font-size:.65rem;color:var(--text_muted)">User Template <span style="font-size:.6rem;opacity:.7">{{file_name}} {{file_path}} {{chunk_number}} {{chunks_total}} {{chunk_content}}</span><textarea name="user_template" class="module-select" rows="3" style="font-family:var(--font-mono);font-size:.72rem;resize:vertical">{_esc(s.get("user_template", "File: {{file_name}} (chunk {{chunk_number}}/{{chunks_total}})"+nsep+nsep+"{{chunk_content}}"))}</textarea></label>
        <label style="font-size:.65rem;color:var(--text_muted)">Output separator ({nsep}=newline)<input type="text" name="output_separator" value="{_esc(sep)}" class="module-select" style="font-size:.72rem;font-family:var(--font-mono)"></label>
        <label style="display:flex;align-items:center;gap:.3rem;font-size:.76rem"><input type="checkbox" name="pause_after" value="1" {"checked" if s.get("pause_after") else ""}> Pause after this step completes</label>
        <div style="display:flex;gap:.3rem;margin-top:.2rem"><button type="submit" class="button" style="flex:1;font-size:.8rem;margin-top:0">{"Add Step" if is_new else "Save Step"}</button>{del_btn}</div>
    </form>
    <script>
    function setStepSrc(el, path) {{
        document.getElementById('step-input-src').value = path;
        document.querySelectorAll('.kg-dir-opt').forEach(function(d) {{ d.style.background=''; d.style.color=''; }});
        el.style.background='var(--accent_dim)'; el.style.color='var(--accent)';
    }}
    function loadStepPrompt(system, template) {{
        var f = document.getElementById('step-edit-form');
        var sp = f.querySelector('[name="system_prompt"]'); var ut = f.querySelector('[name="user_template"]');
        if(sp) sp.value=system; if(ut) ut.value=template;
    }}
    async function saveStepPrompt(saveUrl, listUrl) {{
        var name = document.getElementById('step-prompt-name').value.trim(); if(!name) return;
        var f = document.getElementById('step-edit-form');
        var fd = new FormData();
        fd.append('name', name);
        fd.append('system_prompt', f.querySelector('[name="system_prompt"]')?.value || '');
        fd.append('user_template', f.querySelector('[name="user_template"]')?.value || '');
        await fetch(saveUrl, {{method:'POST', body:fd}});
        htmx.ajax('GET', listUrl, {{target:'#step-prompt-list', swap:'innerHTML'}});
        document.getElementById('step-prompt-name').value='';
    }}
    </script>"""
                   
# --- Main View Builder ---

def _project_view(request, doc, models=None):
    username = request.state.user.username
    ed = ENV["tools"]["built_ins"].PortalEditor(base_url=_u())
    is_working = doc["id"] in _ACTIVE
    return (ed.render_shell(doc) + f"""<div id="tessa-conn-bar-content" hx-swap-oob="outerHTML">{_conn_bar_html(doc, list_conns(), models or [])}</div><div id="tessa-chat-area" hx-swap-oob="outerHTML"><div id="tessa-chat-area" style="height:100%;overflow:hidden">{CM.shell(doc["id"], messages=doc.get("conversation",[]), viewer_name=username, is_working=is_working, stop_url=_u("stop",doc["id"]) if is_working else "")}</div></div><div id="tessa-proj-list" hx-swap-oob="innerHTML">{_proj_list_html(username, doc["id"])}</div><div id="tessa-left-bottom" hx-swap-oob="innerHTML">{_left_bottom_html(doc)}</div>""")

# --- Routes: Main ---

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
    conn = get_conn(doc.get("conn_id",""))
    models = await list_models_async(conn) if conn else []
    if not doc.get("model") and models: doc["model"] = models[0]; _save(doc)
    ed = ENV["tools"]["built_ins"].PortalEditor(base_url=_u())
    chat = f'<div id="tessa-chat-area" style="height:100%; overflow:hidden">{CM.shell(doc["id"], messages=doc.get("conversation",[]), viewer_name=username)}</div>'
    top = f'<div style="position:relative"><div id="tessa-conn-bar-content">{_conn_bar_html(doc, list_conns(), models)}</div></div>'
    pipe_bar = f'<div id="tessa-pipe-bar" style="height:100%; overflow:hidden">{_pipe_progress_html(doc)}</div>'
    return ENV["templates"].TemplateResponse(name="base.html", request=request, context={
        "request": request, "user": user, "nesting_level": 2, "shell_id": IM.branch_id, "code_mirror": True,
        "toolbars": {"top": UI.toolbar(side="top", content=top, size="3rem", overlay=False, start_open=True, locked=True, nesting_level=2),
                     "left":  UI.toolbar(side="left", content=_left_panel(username, doc), size="18rem", overlay=False, start_open=True, resizable=True, nesting_level=2),
                     "right": UI.toolbar(side="right", content=chat, size="22rem", overlay=False, start_open=True, resizable=True, nesting_level=2, id="tessa-right"),
                     "bottom": UI.toolbar(side="bottom", content=pipe_bar, size="7rem", overlay=False, start_open=True, resizable=True, nesting_level=2, id="tessa-bottom")},
        "content": f"""<div id="tessa-center">{ed.render_shell(doc)}</div><div id="tessa-modal" onclick="if(event.target===this){{this.style.display='none'; this.innerHTML=''}}"></div>""",
        "extra_css": CSS + CM.CSS + ed.CSS, "extra_script": ENV["tools"]["built_ins"].PORTAL_EDITOR_JS + CM.SCRIPT})

@router.post("/new", response_class=HTMLResponse)
async def new_project(request: Request):
    user = request.state.user; doc = _new_project(user); _save(doc)
    await ENV["set_state"](request, doc["id"], scope="user", namespace="tessa", key="active_pid")
    models = []
    conn = get_conn(doc.get("conn_id",""))
    if conn: models = await list_models_async(conn)
    return HTMLResponse(_project_view(request, doc, models))

@router.get("/load/{pid}", response_class=HTMLResponse)
async def load_project(pid: str, request: Request):
    user = request.state.user; doc = _load(pid)
    if not doc or doc.get("username") != user.username: return HTMLResponse("Not found", status_code=404)
    await ENV["set_state"](request, pid, scope="user", namespace="tessa", key="active_pid")
    conn = get_conn(doc.get("conn_id",""))
    models = await list_models_async(conn) if conn else []
    return HTMLResponse(_project_view(request, doc, models))

@router.delete("/project/{pid}", response_class=HTMLResponse)
async def delete_project(pid: str, request: Request):
    user = request.state.user; doc = _load(pid)
    if doc and doc.get("username") == user.username: _dp(pid).unlink(missing_ok=True)
    active = await ENV["get_state"](request, scope="user", namespace="tessa", key="active_pid")
    if active == pid: await ENV["set_state"](request, "", scope="user", namespace="tessa", key="active_pid")
    return HTMLResponse(_proj_list_html(user.username, ""))

# --- Routes: Document ---

@router.post("/doc/save/{pid}")
async def doc_save(pid: str, request: Request):
    form = await request.form(); doc = _load(pid)
    if not doc or doc.get("username") != request.state.user.username: return HTMLResponse("")
    doc["content"] = form.get("content","")
    cfg = _SETTINGS.get_group("defaults").load() if _SETTINGS else {}
    if cfg.get("auto_snapshot_on_save",False) and doc["content"].strip():
        vdir = DATA_DIR/"versions"/pid; vdir.mkdir(parents=True, exist_ok=True)
        (vdir/f'{datetime.utcnow().strftime("%Y%m%dT%H%M%S")}.json').write_text(json.dumps({"content":doc["content"], "title":doc.get("title",""), "saved":doc.get("modified","")}, indent=2))
        for old in sorted(vdir.glob("*.json"))[:-30]: old.unlink()
    _save(doc); return HTMLResponse("")

@router.post("/doc/rename/{pid}")
async def doc_rename(pid: str, request: Request):
    form = await request.form(); doc = _load(pid)
    if not doc or doc.get("username") != request.state.user.username: return HTMLResponse("")
    doc["title"] = form.get("value","").strip() or "Untitled"; _save(doc)
    new_input = (f"""<input id="doc-title-{pid}" type="text" value="{_esc(doc["title"])}" name="value" hx-post="{_u("doc/rename",pid)}" hx-trigger="change" hx-target="#doc-title-{pid}" hx-swap="outerHTML" hx-include="this" style="background:transparent;border:none;border-bottom:var(--border-thick) solid var(--border);color:var(--text);font-size:.88rem;font-weight:600;padding:.2rem .3rem;outline:none;flex:1;min-width:5rem;">""")
    return HTMLResponse(new_input + f'<div id="tessa-proj-list" hx-swap-oob="innerHTML">{_proj_list_html(request.state.user.username, pid)}</div>')

@router.post("/doc/settings/{pid}")
async def doc_settings(pid: str, request: Request):
    form = dict(await request.form()); doc = _load(pid)
    if not doc or doc.get("username") != request.state.user.username: return HTMLResponse("")
    s = doc.setdefault("settings",{})
    for k in ("view","font"):
        if k in form: s[k] = form[k]
    for k in ("wrap","border","interactive"):
        if k in form: s[k] = form[k] == "true"
    if "zoom" in form:
        try: s["zoom"] = max(0.6, min(float(form["zoom"]), 2.0))
        except ValueError: pass
    _save(doc); return HTMLResponse(ENV["tools"]["built_ins"].PortalEditor(base_url=_u()).render_shell(doc))

@router.get("/doc/info/{pid}")
async def doc_info(pid: str): return HTMLResponse(ENV["tools"]["built_ins"].PortalEditor(base_url=_u()).info_html(_load(pid) or {}))

@router.get("/doc/search_form/{pid}")
async def doc_search_form(pid: str): return HTMLResponse(ENV["tools"]["built_ins"].PortalEditor(base_url=_u()).search_form_html(pid))

@router.get("/doc/search_close/{pid}")
async def doc_search_close(pid: str): return HTMLResponse("")

@router.post("/doc/search/{pid}")
async def doc_search(pid: str, request: Request):
    form = await request.form()
    doc = _load(pid)
    return HTMLResponse(ENV["tools"]["built_ins"].PortalEditor(base_url=_u()).search_results_html(doc or {}, form.get("query",""), bool(form.get("regex"))))

@router.post("/doc/apply_ai/{pid}")
async def doc_apply_ai(pid: str, request: Request):
    doc = _load(pid)
    if not doc or doc.get("username") != request.state.user.username: return HTMLResponse("")
    last_ai = next((m["content"] for m in reversed(doc.get("conversation",[])) if m.get("role")=="assistant" and not m.get("deleted")), None)
    if last_ai: doc["content"] = last_ai; _save(doc)
    return HTMLResponse(ENV["tools"]["built_ins"].PortalEditor(base_url=_u()).render_shell(doc))

@router.post("/doc/conn/{pid}")
async def doc_conn(pid: str, request: Request):
    form = await request.form(); doc = _load(pid)
    if not doc: return HTMLResponse(f'<select name="model" class="module-select" style="font-size:.72rem"><option>-</option></select>')
    doc["conn_id"] = form.get("conn_id",""); _save(doc)
    conn = get_conn(doc["conn_id"]); models = await list_models_async(conn) if conn else []
    cur = doc.get("model","")
    opts = "".join(f'<option value="{m}" {"selected" if m==cur else ""}>{m}</option>' for m in models) or '<option value="">No models</option>'
    return HTMLResponse(f'<select class="module-select" style="font-size:.72rem;max-width:11rem" name="model" hx-post="{_u("doc/model",pid)}" hx-trigger="change" hx-include="[name=model]" hx-swap="none">{opts}</select>')

@router.post("/doc/model/{pid}")
async def doc_model(pid: str, request: Request):
    form = await request.form(); doc = _load(pid)
    if not doc: return HTMLResponse("")
    doc["model"] = form.get("model",""); _save(doc); return HTMLResponse("")

@router.post("/doc/ctx/{pid}")
async def doc_ctx(pid: str, request: Request):
    form = await request.form(); doc = _load(pid)
    if not doc: return HTMLResponse("")
    doc["model_ctx"] = max(512, int(form.get("model_ctx",32768) or 32768)); _save(doc); return HTMLResponse("")

@router.post("/stop/{pid}")
async def stop_stream(pid: str):
    _STOP[pid] = True
    task = _STREAM_TASKS.pop(pid, None)
    if task and not task.done(): task.cancel()
    return HTMLResponse("")

# --- Routes: Messages ---

@router.post("/msg/delete")
async def msg_delete(request: Request):
    form = await request.form(); mid = form.get("id",""); user = request.state.user
    for doc in _list_projects(user.username):
        d = _load(doc["id"])
        if not d: continue
        for m in d.get("conversation",[]):
            if m.get("id") == mid: m["deleted"] = True; _save(d); return HTMLResponse("")
    return HTMLResponse("")

@router.get("/msg/edit_form/{mid}")
async def msg_edit_form(mid: str, request: Request):
    user = request.state.user
    for doc in _list_projects(user.username):
        d = _load(doc["id"])
        if not d: continue
        for m in d.get("conversation",[]):
            if m.get("id") != mid: continue
            role_cls = "cm-me" if m.get("role")=="user" else "cm-other"
            avatar = CM._avatar_html(m.get("user_name","?"))
            return HTMLResponse(f'<div class="cm-msg {role_cls}" id="cm-msg-{mid}" data-msg-id="{mid}">{avatar}<div class="cm-bwrap" style="max-width:90%"><form hx-post="{_u("msg/edit_save",mid)}" hx-target="#cm-msg-{mid}" hx-swap="outerHTML" style="display:flex;flex-direction:column;gap:.3rem;width:100%"><textarea name="content" class="cm-input" style="min-height:4rem;overflow-y:auto">{_esc(m.get("content",""))}</textarea><div style="display:flex;gap:.3rem"><button type="submit" class="button" style="font-size:.75rem;margin-top:0">Save</button><button type="button" class="btn-icon" hx-get="{_u("msg/cancel_edit",mid)}" hx-target="#cm-msg-{mid}" hx-swap="outerHTML">Cancel</button></div></form></div></div>')
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
    user = request.state.user
    for doc in _list_projects(user.username):
        d = _load(doc["id"])
        if not d: continue
        for m in d.get("conversation",[]):
            if m.get("id") != mid: continue
            is_me = m.get("role") == "user"
            return HTMLResponse(CM.render_message(m, is_me=is_me, can_delete=True, can_edit=is_me))
    return HTMLResponse("")

@router.post("/msg/retry/{mid}")
async def msg_retry(mid: str, request: Request):
    user = request.state.user
    for doc in _list_projects(user.username):
        d = _load(doc["id"])
        if not d: continue
        msgs = d.get("conversation",[]); idx = next((i for i,m in enumerate(msgs) if m.get("id")==mid), None)
        if idx is None: continue
        m = msgs[idx]; role_cls = "cm-me" if m.get("role")=="user" else "cm-other"
        return HTMLResponse(f'<div class="cm-msg {role_cls}" id="cm-msg-{mid}" data-msg-id="{mid}">{CM._avatar_html(m.get("user_name","?"))}<div class="cm-bwrap" style="max-width:90%"><form hx-post="{_u("msg/retry_send",mid)}" hx-target="#cm-msg-{mid}" hx-swap="outerHTML" style="display:flex;flex-direction:column;gap:.3rem;width:100%"><textarea name="content" class="cm-input" style="min-height:4rem;overflow-y:auto">{_esc(m.get("content",""))}</textarea><div style="display:flex;gap:.3rem"><button type="submit" class="button" style="font-size:.75rem;margin-top:0">&#x21BA; Retry</button><button type="button" class="btn-icon" hx-get="{_u("msg/cancel_edit",mid)}" hx-target="#cm-msg-{mid}" hx-swap="outerHTML">Cancel</button></div></form></div></div>')
    return HTMLResponse("")

@router.post("/msg/retry_send/{mid}")
async def msg_retry_send(mid: str, request: Request):
    form = await request.form(); user = request.state.user; new_content = form.get("content","").strip()
    for doc in _list_projects(user.username):
        d = _load(doc["id"])
        if not d: continue
        msgs = d.get("conversation",[]); idx = next((i for i,m in enumerate(msgs) if m.get("id")==mid), None)
        if idx is None: continue
        msgs[idx]["content"] = new_content; msgs[idx]["edited"] = True
        d["conversation"] = msgs[:idx+1]; _save(d); pid = d["id"]
        remaining = "".join(CM.render_message(m, is_me=(m.get("role")=="user"), can_delete=True, can_edit=(m.get("role")=="user")) for m in d["conversation"] if not m.get("deleted"))
        conn = get_conn(d.get("conn_id","")); model = d.get("model","")
        if conn and model: asyncio.create_task(_run_chat_task(pid, user.username, conn, _build_messages(d, new_content), model, d.get("model_ctx",32768)))
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

@router.post("/files/toggle/{pid}", response_class=HTMLResponse)
async def files_toggle(pid: str, request: Request):
    form = await request.form(); path = form.get("path",""); is_dir = form.get("is_dir","false")=="true"
    doc = _load(pid)
    if not doc: return HTMLResponse("")
    files = set(doc.get("selected_files",[]));
    if is_dir:
        full = KG_DIR/path
        children = {str(f.relative_to(KG_DIR)) for f in full.rglob("*") if f.is_file() and not f.name.startswith(".")} if full.is_dir() else set()
        if children and children.issubset(files): files -= children
        else: files |= children
    else:
        if path in files: files.discard(path)
        else: files.add(path)
    doc["selected_files"] = list(files); _save(doc)
    return HTMLResponse(_kg_html(doc))

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

# --- Routes: Pipelines ---

@router.get("/pipeline/new_form/{pid}", response_class=HTMLResponse)
async def pipeline_new_form(pid: str): return HTMLResponse(f'<form hx-post="{_u("pipeline/create",pid)}" hx-target="#tessa-pipelines-section" hx-swap="outerHTML" style="padding:.4rem;display:flex;gap:.3rem"><input type="text" name="name" class="module-select" placeholder="Pipeline name" style="flex:1;font-size:.78rem" required autofocus><button type="submit" class="button" style="margin-top:0;font-size:.75rem">Create</button></form>')

@router.post("/pipeline/create/{pid}", response_class=HTMLResponse)
async def pipeline_create(pid: str, request: Request):
    form = await request.form(); doc = _load(pid)
    if not doc or doc.get("username") != request.state.user.username: return HTMLResponse("")
    pl = {"id":f"pl_{uuid.uuid4().hex[:8]}","name":form.get("name","Pipeline").strip(),"state":"idle","current_step":0,"steps":[]}
    doc.setdefault("pipelines",[]).append(pl); _save(doc)
    return HTMLResponse(_pipelines_html(doc))

@router.delete("/pipeline/{pid}/{pl_id}", response_class=HTMLResponse)
async def pipeline_delete(pid: str, pl_id: str, request: Request):
    task = _PIPE_TASKS.pop(pl_id, None)
    if task: task.cancel()
    _STOP[f"pl_{pl_id}"] = True
    doc = _load(pid)
    if not doc: return HTMLResponse("")
    doc["pipelines"] = [p for p in doc.get("pipelines",[]) if p["id"] != pl_id]; _save(doc)
    return HTMLResponse("")  # hx-swap="outerHTML" targets the card itself

@router.get("/pipeline/edit/{pid}/{pl_id}", response_class=HTMLResponse)
async def pipeline_edit(pid: str, pl_id: str):
    doc = _load(pid)
    if not doc: return HTMLResponse("")
    pl = next((p for p in doc.get("pipelines",[]) if p["id"]==pl_id), None)
    if not pl: return HTMLResponse("")
    inner = _pl_edit_html(doc, pl)
    return HTMLResponse(f"""<div id="tessa-modal-inner" class="glass" style="width:min(92vw,64rem);height:min(85vh,45rem);overflow:hidden;display:flex;flex-direction:column;position:relative;padding:0"><div style="display:flex;align-items:center;padding:.5rem .75rem;border-bottom:var(--border-thick) solid var(--border);flex-shrink:0"><span style="font-weight:600;color:var(--accent);font-size:.9rem">&#x26A1; Pipeline Editor &mdash; {_esc(pl.get("name",""))}</span><button onclick="document.getElementById('tessa-modal').style.display='none';document.getElementById('tessa-modal').innerHTML=''" style="margin-left:auto;background:none;border:none;cursor:pointer;font-size:1.2rem;color:var(--text_muted);padding:.2rem .4rem">&#x2715;</button></div><div style="flex:1;min-height:0;overflow:hidden">{inner}</div></div><script>document.getElementById("tessa-modal").style.display="flex";</script>""")

@router.post("/pipeline/rename/{pid}/{pl_id}", response_class=HTMLResponse)
async def pipeline_rename(pid: str, pl_id: str, request: Request):
    form = await request.form(); doc = _load(pid)
    if not doc: return HTMLResponse("")
    pl = next((p for p in doc.get("pipelines",[]) if p["id"]==pl_id), None)
    if pl: pl["name"] = form.get("name","").strip() or pl["name"]; _save(doc)
    return HTMLResponse("")

@router.post("/pipeline/reset/{pid}/{pl_id}", response_class=HTMLResponse)
async def pipeline_reset(pid: str, pl_id: str):
    doc = _load(pid)
    if not doc: return HTMLResponse("")
    pl = next((p for p in doc.get("pipelines",[]) if p["id"]==pl_id), None)
    if pl:
        pl["state"] = "idle"; pl["current_step"] = 0
        for s in pl.get("steps",[]): s["state"] = "idle"; s.pop("progress", None)
        _save(doc)
    return HTMLResponse(_pipelines_html(doc))

@router.post("/pipeline/run/{pid}/{pl_id}", response_class=HTMLResponse)
async def pipeline_run(pid: str, pl_id: str, request: Request):
    user = request.state.user
    doc = _load(pid)
    if not doc: return HTMLResponse("")
    pl = next((p for p in doc.get("pipelines",[]) if p["id"]==pl_id), None)
    if not pl or pl.get("state") == "running": return HTMLResponse("")
    if pl.get("state") in ("done","error"): pl["current_step"] = 0
    for s in pl.get("steps",[]): s.get("state") != "done" and s.pop("progress", None)
    _STOP.pop(f"pl_{pl_id}", None)
    task = asyncio.create_task(_run_pipeline(pid, pl_id, user.username))
    _PIPE_TASKS[pl_id] = task
    return HTMLResponse("")  # pipeline card updates via WS OOB

@router.post("/pipeline/stop/{pid}/{pl_id}", response_class=HTMLResponse)
async def pipeline_stop(pid: str, pl_id: str, request: Request):
    _STOP[f"pl_{pl_id}"] = True

    user = request.state.user
    doc = _load(pid)
    if not doc: return HTMLResponse("")
    pl = next((p for p in doc.get("pipelines",[]) if p["id"]==pl_id), None)
    if not pl: return HTMLResponse("")
    pl["state"] = "done"
    return HTMLResponse("")  # pipeline card updates via WS OOB

@router.get("/pipeline/step_form/{pid}/{pl_id}", response_class=HTMLResponse)
async def step_form_new(pid: str, pl_id: str):
    doc = _load(pid)
    if not doc: return HTMLResponse("")
    return HTMLResponse(_step_form_html(doc, pl_id))

@router.get("/pipeline/step_form/{pid}/{pl_id}/{sid}", response_class=HTMLResponse)
async def step_form_edit(pid: str, pl_id: str, sid: str):
    doc = _load(pid)
    if not doc: return HTMLResponse("")
    pl = next((p for p in doc.get("pipelines",[]) if p["id"]==pl_id), None)
    if not pl: return HTMLResponse("")
    step = next((s for s in pl.get("steps",[]) if s.get("id")==sid), None)
    return HTMLResponse(_step_form_html(doc, pl_id, step))

@router.post("/pipeline/step/add/{pid}/{pl_id}", response_class=HTMLResponse)
async def step_add(pid: str, pl_id: str, request: Request):
    form = await request.form(); doc = _load(pid)
    if not doc: return HTMLResponse("")
    pl = next((p for p in doc.get("pipelines",[]) if p["id"]==pl_id), None)
    if not pl: return HTMLResponse("")
    step = _step_from_form(form)
    pl.setdefault("steps",[]).append(step); _save(doc)
    return HTMLResponse(_pl_edit_html(doc, pl))

@router.post("/pipeline/step/save/{pid}/{pl_id}/{sid}", response_class=HTMLResponse)
async def step_save(pid: str, pl_id: str, sid: str, request: Request):
    form = await request.form()
    doc = _load(pid)
    if not doc: return HTMLResponse("")
    pl = next((p for p in doc.get("pipelines",[]) if p["id"]==pl_id), None)
    if not pl: return HTMLResponse("")
    for s in pl.get("steps",[]):
        if s.get("id") == sid: s.update(_step_from_form(form, sid)); break
    _save(doc); return HTMLResponse(_pl_edit_html(doc, pl))

@router.delete("/pipeline/step/{pid}/{pl_id}/{sid}", response_class=HTMLResponse)
async def step_delete(pid: str, pl_id: str, sid: str, request: Request):
    doc = _load(pid)
    if not doc: return HTMLResponse("")
    pl = next((p for p in doc.get("pipelines",[]) if p["id"]==pl_id), None)
    if not pl: return HTMLResponse("")
    pl["steps"] = [s for s in pl.get("steps",[]) if s.get("id") != sid]; _save(doc)
    return HTMLResponse(_pl_edit_html(doc, pl))

def _step_from_form(form, existing_id=None):
    sep = form.get("output_separator","\n\n---\n\n").replace("\\n","\n")
    return {"id": existing_id or f"stp_{uuid.uuid4().hex[:8]}",
            "name": form.get("name","Step").strip(),
            "type": form.get("type","file_pass"),
            "conn_id": form.get("conn_id",""),
            "model": form.get("model","").strip(),
            "model_ctx": max(512, int(form.get("model_ctx",32768) or 32768)),
            "chunk_tokens": max(0, int(form.get("chunk_tokens",6000) or 6000)),
            "input_source": form.get("input_source","").strip("/"),
            "use_selected": bool(form.get("use_selected")),
            "system_prompt": form.get("system_prompt","").strip(),
            "user_template": form.get("user_template","").strip() or "{chunk_content}",
            "output_separator": sep,
            "pause_after": bool(form.get("pause_after")),
            "state": "idle"}

@router.post("/pipeline/step_models/{pid}/{pl_id}", response_class=HTMLResponse)
async def pipeline_step_models(pid: str, pl_id: str, request: Request):
    form = await request.form(); cid = form.get("conn_id","")
    conn = get_conn(cid); models = list_models_sync(conn) if conn else []
    opts = "".join(f'<option value="{m}">{m}</option>' for m in models) or '<option value="">No models</option>'
    return HTMLResponse(f'<label style="font-size:.65rem;color:var(--text_muted)">Model<select name="model" class="module-select" style="font-size:.73rem">{opts}</select></label>')

# --- Routes: Settings ---

@router.get("/settings", response_class=HTMLResponse)
async def settings_page(request: Request):
    group = _SETTINGS.get_group("defaults"); values = group.load()
    return HTMLResponse(f'<div style="padding:1.5rem"><h2 style="margin:0 0 1rem;font-size:1rem">Tessa Settings</h2><form hx-post="{_u("settings/save")}" hx-target="#tessa-settings-status" style="display:flex;flex-direction:column;gap:.6rem"><div id="tessa-settings-fields">{group.render(values)}</div><button type="submit" class="button" style="margin-top:.5rem">Save Settings</button><div id="tessa-settings-status" style="font-size:.75rem;min-height:1rem"></div></form></div>')

@router.post("/settings/save", response_class=HTMLResponse)
async def settings_save(request: Request):
    form = dict(await request.form()); _SETTINGS.get_group("defaults").save(form)
    return HTMLResponse('<span style="color:#00ffa2">&#x2713; Saved</span>')

@router.get("/pipeline/step/prompts", response_class=HTMLResponse)
async def step_prompts():
    prompts = _load_step_prompts()
    if not prompts: return HTMLResponse('<div style="color:var(--text_muted);font-size:.72rem;padding:.3rem">No saved prompts.</div>')
    rows = "".join(f"""<div style="display:flex;align-items:center;gap:.3rem;padding:.2rem 0;border-bottom:var(--border-thick) solid var(--border)"><span style="flex:1;font-size:.75rem;overflow:hidden;text-overflow:ellipsis;white-space:nowrap">{_esc(p["name"])}</span><button class="cm-qbtn" type="button" onclick="loadStepPrompt({json.dumps(p.get("system_prompt",""))},{json.dumps(p.get("user_template",""))})" title="Load into form">&#x21B3;</button><button class="cm-qbtn" type="button" style="color:#ff5f5f" hx-delete="{_u("pipeline/step/prompt",p["id"])}" hx-target="#step-prompt-list" hx-swap="innerHTML">&#x2715;</button></div>""" for p in prompts)
    return HTMLResponse(f'<div id="step-prompt-list">{rows}</div>')

@router.post("/pipeline/step/prompt/save", response_class=HTMLResponse)
async def step_prompt_save(request: Request):
    form = await request.form(); prompts = _load_step_prompts()
    prompts.append({"id": uuid.uuid4().hex[:8], "name": form.get("name","Untitled"), "system_prompt": form.get("system_prompt",""), "user_template": form.get("user_template","")})
    _save_step_prompts(prompts)
    return await step_prompts()

@router.delete("/pipeline/step/prompt/{pid}", response_class=HTMLResponse)
async def step_prompt_delete(pid: str):
    _save_step_prompts([p for p in _load_step_prompts() if p["id"] != pid])
    return await step_prompts()

@router.post("/doc/toggle_task/{pid}")
async def doc_toggle_task(pid: str, request: Request):
    form = await request.form(); doc = _load(pid)
    if not doc or doc.get("username") != request.state.user.username: return HTMLResponse("")
    ed = ENV["tools"]["built_ins"].PortalEditor(base_url=_u())
    doc["content"] = ed._flip_task(doc.get("content",""), int(form.get("idx", -1))); _save(doc)
    return HTMLResponse(ed.render_preview(doc["content"], task_interactive=True, doc_id=pid))

# --- CSS ---

CSS = """
#tessa-proj-list .active-item{background:var(--glass);border-left:.15rem solid var(--accent);}
#tessa-modal{display:none;position:fixed;inset:0;z-index:2000;align-items:center;justify-content:center;background:rgba(0,0,0,0.65);}
.editor-shell{display:flex;flex-direction:column;height:100%;width:100%;overflow:hidden;}
#tessa-center{display:flex;flex-direction:column;height:100%;width:100%;overflow:hidden;}
#tessa-pipe-bar{height:100%;display:flex;align-items:stretch;}
@media(max-width:68rem){.tessa-pl-layout{flex-direction:column!important;}.tessa-pl-steps{flex:0 0 auto!important;max-height:35vh!important;border-right:none!important;border-bottom:var(--border-thick) solid var(--border)!important;}}
"""