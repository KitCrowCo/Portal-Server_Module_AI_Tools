# /modules/ai_tools/code_agent/code_agent.py
"""
Code Agent - task queue and review surface for agent_loop runs. Sub-module of ai_tools, mounted at /module/ai_tools/code_agent.
A run is one pipeline job: the built-in single agent_loop node, or any saved pipeline that takes 'input' (the task) and produces the agent_loop keys (summary, agent_status, branch, workspace, transcript).
Runs on the same toolset share one engine lane, so they execute one at a time in submission order. The toolset node (code_workspace) owns the repo: each run works on its own ai/<id> branch, and merging stays with a person.
Data at data/ai_tools/code_agent/ (one small JSON record per run; the job record holds everything else).
"""
import json, uuid
from datetime import datetime
from pathlib import Path
from fastapi import APIRouter, Request

TOOL_META = {"label": "Code Agent", "icon": "&#x2387;", "group": "model_tools", "description": "Queue coding tasks for a local agent and review its branches", "singleton": True}

router = APIRouter(redirect_slashes=False)
DATA_DIR = Path("./data/ai_tools/code_agent")
_LIVE = ("queued", "running", "stopping")
_READ_ONLY_TOOLS = "list_files,read_file,grep,outline"
ENV = {}
UI = IM = BI = AIM = _FIELDS = None

# --- Run records ---

def _rp(rid): return DATA_DIR / f"{Path(rid).name}.json"
def _load(rid): return json.loads(_rp(rid).read_text()) if rid and _rp(rid).exists() else None
def _save(rec): _rp(rec["id"]).write_text(json.dumps(rec, indent=2))
def _runs(username): return sorted((r for r in (json.loads(f.read_text()) for f in DATA_DIR.glob("*.json")) if r.get("username") == username), key=lambda r: r["created"], reverse=True)
def _owned(request, rid): rec = _load(rid); return rec if rec and rec.get("username") == request.state.user.username else None
def _job(rec): return AIM.engine.load_job(rec["job_id"]) or {"status": "missing", "data": {}, "flow": {"nodes": []}, "log": []}

def _toolset(conn_id: str) -> dict:
    """Connection for review calls - the run's own toolset (recorded by agent_loop even when the pool chose it). Missing means it was deleted from Settings; that should surface, not be guessed around."""
    conn = AIM.connections.get_conn(conn_id)
    if not conn: raise RuntimeError(f"""toolset connection '{conn_id}' no longer exists - re-add it in AI Tools > Settings > Connections""")
    return conn

async def _call(conn_id: str, capability: str, payload: dict) -> dict: return await AIM.connections.call_capability(_toolset(conn_id), capability, payload)

# --- Task form ---

def _conn_opts(capability):
    def _opts(values=None): return [("", "(pool / first available)")] + [(c["_id"], c.get("display_name", c["_id"])) for c in AIM.connections.conns_matching(capability)]
    return _opts

def _model_opts(values=None):
    conn = AIM.connections.get_conn((values or {}).get("conn_id", "")) if (values or {}).get("conn_id") else None
    return [("", "(auto - first non-embedding model)")] + [(m, m) for m in (AIM.connections.list_models_sync(conn) if conn else [])]

def _build_fields():
    F = BI.SettingField
    return BI.SettingsGroup("code_agent", "", [
        F("tools_conn_id", "Workspace (toolset)", "select", "", options=_conn_opts("agent_tools_call")),
        F("conn_id", "Chat Connection", "select", "", options=_conn_opts("chat")),
        F("model", "Model", "select", "", options=_model_opts, hint="Lists the models of the connection saved last; queue once after changing the connection to refresh."),
        F("base", "Branch From (ref)", "text", "", hint="Blank = the node's BASE_REF (normally its current HEAD)."),
        F("max_steps", "Max Steps", "number", 60, step=1),
        F("read_only", "Read-only (review/explain, no edits)", "checkbox", False),
        F("pipeline_id", "Pipeline", "select", "", options=lambda values=None: [("", "(built-in: one agent_loop node)")] + [(p["id"], p.get("name", p["id"])) for p in AIM.engine.list_pipelines()], advanced=True, hint="A saved pipeline gets the task as 'input' and should produce the agent_loop keys."),
        F("num_ctx", "Context Window Tokens", "number", 32768, step=1, advanced=True),
        F("temperature", "Temperature", "number", 0.2, advanced=True),
        F("tool_protocol", "Tool Protocol", "select", "auto", options=[("auto", "Auto"), ("native", "Native tool calls"), ("text", "Text blocks (any model)")], advanced=True),
        F("think", "Thinking Effort", "select", "", options=[("", "Off"), ("low", "Low"), ("medium", "Medium"), ("high", "High")], advanced=True),
        F("system_prompt", "System Prompt Override", "textarea", "", advanced=True, hint="Blank = agent_loop's default. The repo's AGENT.md / AI_Context.md are added by the workspace either way.")], json_path="")

def _flow(s: dict) -> dict:
    cfg = {"tools_conn_id": s.get("tools_conn_id", ""), "conn_id": s.get("conn_id", ""), "model": s.get("model", ""), "max_steps": s.get("max_steps", 60), "num_ctx": s.get("num_ctx", 32768), "temperature": s.get("temperature", 0.2), "tool_protocol": s.get("tool_protocol", "auto"), "think": s.get("think", ""), "session_args_json": {"base": s["base"]} if s.get("base") else {}, "allow_tools": _READ_ONLY_TOOLS if s.get("read_only") else ""}
    if s.get("system_prompt"): cfg["system_prompt"] = s["system_prompt"]
    return {"nodes": [{"id": "agent", "name": "Agent", "type": "agent_loop", "config": cfg, "key_map": {}, "status": "idle"}]}

# --- HTML ---

def _post(action: str, **extra) -> str: return f"""hx-post="/im/in" hx-target="body" hx-swap="none" hx-vals='{json.dumps({"type": f"codeagent_{action}", "lvl": 2, **extra})}'"""

def _status_badge(status: str) -> str: return f"""<span class="status-badge" style="color:{ {"done": "#00ffa2", "error": "#ff5f5f", "running": "#ffcc00", "queued": "var(--text_muted)"}.get(status, "var(--text)") }">{UI.escape(status)}</span>"""

def _list_html(username: str, active: str = "") -> str:
    rows = "".join(f"""<div class="list-card glass" style="{"border-left:.15rem solid var(--accent)" if r["id"] == active else ""}"><div class="list-card-hd"><span class="list-card-title" {_post("open", rid=r["id"])}>{UI.escape(r["task"][:60])}</span>{_status_badge(_job(r)["status"])}</div><div class="status-label">{r["created"][:16].replace("T", " ")}</div></div>""" for r in _runs(username))
    return rows or '<div class="list-empty">No runs yet.</div>'

def _form_html(values: dict) -> str:
    return f"""<form {_post("queue")} hx-include="this" class="node-form glass" style="padding:.6rem">
                   <span class="form-title">New task</span>
                   <textarea name="task" class="cm-input" rows="5" placeholder="What should the agent do? Name files and functions when you know them." required></textarea>
                   {_FIELDS.render(values, name_prefix="ca_")}
                   <div class="form-actions"><button type="submit" class="ui-btn">Queue</button><button type="button" class="ui-btn" {_post("sessions")} hx-include="closest form">Workspace sessions</button></div>
               </form>"""

def _poller(rid: str) -> str: return f"""<div {_post("status", rid=rid)} hx-trigger="load delay:3s"></div>"""

def _transcript_html(messages: list) -> str:
    """One collapsible row per message after the system prompt; tool results and long replies stay folded so the run reads as a sequence of steps."""
    rows = []
    for m in messages[1:]:
        who = " ".join(p for p in (m["role"], m.get("tool_name", ""), "-> " + ", ".join(c["function"]["name"] for c in m["tool_calls"]) if m.get("tool_calls") else "") if p)
        body = str(m.get("content", "")) + ("\n" + json.dumps(m["tool_calls"], indent=1) if m.get("tool_calls") else "")
        rows.append(f"""<details class="log-line"><summary>{UI.escape(who)}: {UI.escape(str(m.get("content", ""))[:120])}</summary><pre class="ca-pre">{UI.escape(body)}</pre></details>""")
    return f"""<div class="log-box ca-log">{"".join(rows) or "(empty)"}</div>"""

def _result_html(rec: dict, job: dict) -> str:
    data, status = job.get("data", {}), job["status"]
    if status in _LIVE: return f"""<div class="info-bar">{_status_badge(status)} {"waiting for earlier runs on this workspace" if status == "queued" else "working"} <button class="ui-btn" {_post("stop", rid=rec["id"])}>&#x25FC; Stop</button></div>{_poller(rec["id"])}"""
    ws = data.get("workspace") or {}
    branch = UI.escape(ws.get("branch", ""))
    err = next((f"""{n.get("name") or n["id"]}: {n.get("message", "")}""" for n in job["flow"]["nodes"] if n.get("status") == "error"), "")
    files = "".join(f"""<tr><td>{UI.escape(f["status"])}</td><td class="ca-mono">{UI.escape(f["path"])}</td><td><button class="ui-btn" {_post("diff", rid=rec["id"], path=f["path"])}>Diff</button></td></tr>""" for f in ws.get("files", []))
    review = f"""<div class="info-bar"><span>branch <b class="ca-mono">{branch}</b></span><span>from <span class="ca-mono">{UI.escape(ws.get("base", "")[:10])}</span></span><span>{"commit " + UI.escape(ws["commit"][:10]) if ws.get("commit") else "no changes committed"}</span></div>
                 <table class="data-table">{files}</table>
                 <div class="form-actions"><button class="ui-btn" {_post("diff", rid=rec["id"])}>Full diff</button><button class="ui-btn" style="color:#ff5f5f" {_post("discard", rid=rec["id"])} hx-confirm="Delete branch {branch}? The agent's work on it is lost.">Discard branch</button></div>
                 <div class="status-label">Merging is yours: on the repo host, <span class="ca-mono">git diff HEAD...{branch}</span> then <span class="ca-mono">git merge {branch}</span>.</div>""" if branch and not rec.get("discarded") else ("""<div class="status-label">Branch discarded.</div>""" if branch else "")
    err_html = f"""<div class="err-box">{UI.escape(err)}</div>""" if err else ""
    summary = f"""<div class="glass ca-summary">{UI.escape(str(data["summary"]))}</div>""" if data.get("summary") else ""
    return f"""<div class="info-bar">{_status_badge(status)} <span>{UI.escape(str(data.get("agent_status", "")))}</span><span>{data.get("steps", "?")} step(s)</span></div>
               {err_html}{summary}
               {review}<div id="ca-diff"></div>
               <details class="status-details"><summary>Transcript</summary>{_transcript_html(data.get("transcript") or [])}</details>"""

def _run_html(rec: dict) -> str:
    job = _job(rec)
    live = f"""<pre class="log-box ca-pre ca-live" data-stream-job="{rec["job_id"]}" data-stream-node="agent"></pre>""" if job["status"] in _LIVE else ""
    return f"""<div class="ca-run"><div class="ca-task">{UI.escape(rec["task"])}</div>{live}<div id="ca-result">{_result_html(rec, job)}</div></div>"""

def _close_btn(conn_id: str, s: dict) -> str: return f"""<button class="ui-btn" {_post("session_close", conn_id=conn_id, session=s["id"])} hx-confirm="Commit what is in this worktree to its branch and remove the worktree? Only do this when no run is still using it - a running agent would fail on its next step.">Close &amp; commit</button>""" if s["status"] == "open" else ""

def _sessions_html(conn_id: str, sessions: list) -> str:
    rows = "".join(f"""<tr><td class="ca-mono">{UI.escape(s["branch"])}</td><td>{_status_badge(s["status"])}</td><td>{UI.escape(s.get("label", ""))}</td><td>{s["opened"][:16].replace("T", " ")}</td><td>{_close_btn(conn_id, s)}</td></tr>""" for s in sessions)
    return f"""<div class="status-label">Sessions on this workspace. An <b>open</b> session with no running job was cut off by a restart - close it to keep its work on its branch.</div><table class="data-table"><tr><th>Branch</th><th>Status</th><th>Label</th><th>Opened</th><th></th></tr>{rows}</table>"""

def _page_html(username: str, values: dict) -> str: return f"""<div class="ca-shell"><div class="list-panel ca-side"><div class="list-panel-hd"><span class="list-panel-title">Runs</span><button class="btn-icon" {_post("home")}>+</button></div><div id="ca-list" class="list-body">{_list_html(username)}</div></div><div id="ca-main" class="ca-main">{_form_html(values)}</div></div>"""

# --- Intent handlers ---

async def _defaults(request) -> dict: return await ENV["get_state"](request, scope="user", namespace="code_agent", key="defaults") or {f.name: f.default for f in _FIELDS.fields}

async def _h_home(request, payload, imr): return imr.oob(_form_html(await _defaults(request)), "ca-main")

async def _h_queue(request, payload, imr):
    task = str(payload.get("task", "")).strip()
    if not task: return imr
    s = _FIELDS.parse(payload, name_prefix="ca_")
    await ENV["set_state"](request, s, scope="user", namespace="code_agent", key="defaults")
    username = request.state.user.username
    kind, extra = ("id", {"pipeline_id": s["pipeline_id"]}) if s.get("pipeline_id") else ("inline", {"inline_flow": _flow(s)})
    job_id, err = AIM.engine.submit(username, kind=kind, inputs={"input": task}, lane=f"""code_workspace:{s.get("tools_conn_id") or "pool"}""", **extra)
    if err: return imr.oob(f"""<div class="err-box">{UI.escape(err)}</div>""", "ca-main")
    rec = {"id": f"car_{uuid.uuid4().hex[:8]}", "username": username, "task": task, "job_id": job_id, "settings": s, "created": datetime.utcnow().isoformat()}
    _save(rec)
    imr.oob(_run_html(rec), "ca-main")
    return imr.oob(_list_html(username, rec["id"]), "ca-list")

async def _h_open(request, payload, imr):
    rec = _owned(request, payload.get("rid", ""))
    if not rec: return imr
    imr.oob(_run_html(rec), "ca-main")
    return imr.oob(_list_html(request.state.user.username, rec["id"]), "ca-list")

async def _h_status(request, payload, imr):
    """Poll target while a run is live: re-renders the result block (which carries the next poll), and refreshes the list once the run settles."""
    rec = _owned(request, payload.get("rid", ""))
    if not rec: return imr
    job = _job(rec)
    imr.oob(_result_html(rec, job), "ca-result")
    if job["status"] not in _LIVE: imr.oob(_list_html(request.state.user.username, rec["id"]), "ca-list")
    return imr

async def _h_stop(request, payload, imr):
    rec = _owned(request, payload.get("rid", ""))
    if rec: AIM.engine.stop(rec["job_id"])
    return imr.oob(_result_html(rec, _job(rec)), "ca-result") if rec else imr

async def _h_diff(request, payload, imr):
    rec = _owned(request, payload.get("rid", ""))
    ws = (_job(rec)["data"].get("workspace") or {}) if rec else {}
    if not ws.get("session"): return imr
    r = await _call(ws["conn_id"], "agent_session_diff", {"session": ws["session"], **({"path": payload["path"]} if payload.get("path") else {})})
    return imr.oob(f"""<div class="err-box">{UI.escape(r["error"])}</div>""" if r.get("error") else f"""<pre class="log-box ca-pre ca-diff">{UI.escape(r["diff"]) or "(no differences)"}</pre>""", "ca-diff")

async def _h_discard(request, payload, imr):
    rec = _owned(request, payload.get("rid", ""))
    job = _job(rec) if rec else None
    ws = (job["data"].get("workspace") or {}) if job else {}
    if not ws.get("session"): return imr
    r = await _call(ws["conn_id"], "agent_session_discard", {"session": ws["session"], "confirm": True})
    if r.get("error"): return imr.oob(f"""<div class="err-box">{UI.escape(r["error"])}</div>""", "ca-diff")
    rec["discarded"] = datetime.utcnow().isoformat()
    _save(rec)
    return imr.oob(_result_html(rec, job), "ca-result")

async def _h_sessions(request, payload, imr):
    conn_id = payload.get("ca_tools_conn_id") or next((c["_id"] for c in AIM.connections.conns_matching("agent_tools_call")), "")
    if not conn_id: return imr.oob('<div class="err-box">No workspace connection configured - add a Code workspace connection in AI Tools > Settings.</div>', "ca-main")
    r = await _call(conn_id, "agent_session_list", {})
    return imr.oob(f"""<div class="err-box">{UI.escape(r["error"])}</div>""" if r.get("error") else _sessions_html(conn_id, r["sessions"]), "ca-main")

async def _h_session_close(request, payload, imr):
    r = await _call(payload.get("conn_id", ""), "agent_session_close", {"session": payload.get("session", ""), "message": "closed from Code Agent after an interrupted run"})
    if r.get("error"): return imr.oob(f"""<div class="err-box">{UI.escape(r["error"])}</div>""", "ca-main")
    return await _h_sessions(request, {"ca_tools_conn_id": payload.get("conn_id", "")}, imr)

# --- Init / route ---

def init_tool(env: dict, prefix: str):
    global ENV, UI, IM, BI, AIM, _FIELDS
    ENV = env
    UI = env["templates"].env.globals.get("UI")
    BI = env["tools"]["built_ins"]
    AIM = env["tools"]["ai_manager"]
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    _FIELDS = _build_fields()
    IM = env["InterfaceManager"](nesting_level=2, db_path="code_agent_im.db")
    IM.scripts.update({f"codeagent_{k}": [fn] for k, fn in {"home": _h_home, "queue": _h_queue, "open": _h_open, "status": _h_status, "stop": _h_stop, "diff": _h_diff, "discard": _h_discard, "sessions": _h_sessions, "session_close": _h_session_close}.items()})
    print("[code_agent] ready")

@router.get("")
@router.get("/")
async def root(request: Request):
    return ENV["templates"].TemplateResponse(name="base.html", request=request, context={"request": request, "user": request.state.user, "nesting_level": 2, "shell_id": IM.branch_id, "toolbars": {}, "content": _page_html(request.state.user.username, await _defaults(request)), "extra_css": CSS})

CSS = """
.ca-shell{display:flex;height:100%;width:100%;overflow:hidden;}
.ca-side{width:16rem;flex-shrink:0;border-right:var(--border-thick) solid var(--border);}
.ca-main{flex:1;overflow-y:auto;padding:.6rem;min-width:0;display:flex;flex-direction:column;gap:.5rem;}
.ca-run{display:flex;flex-direction:column;gap:.5rem;}
.ca-task{white-space:pre-wrap;font-weight:600;}
.ca-mono{font-family:var(--font-mono);}
.ca-pre{white-space:pre-wrap;margin:0;}
.ca-live,.ca-diff,.ca-log{max-height:50vh;}
.ca-summary{padding:.5rem .7rem;white-space:pre-wrap;}
@media (max-width:46rem){.ca-shell{flex-direction:column;}.ca-side{width:auto;max-height:30vh;border-right:none;}}
"""
