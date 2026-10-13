# /modules/ai_tools/agent/agent.py
"""
Agent - task queue and review surface for agent_loop runs. Sub-module of ai_tools, mounted at /module/ai_tools/agent.
A run is one pipeline job: the built-in single agent_loop node, or any saved pipeline that takes 'input' (the task) and produces the agent_loop keys (summary, agent_status, steps, transcript, workspace, branch, space).
What the agent can do is its toolsets: its own space, a toolset connection (e.g. a code workspace, where each run works on its own ai/<id> branch and merging stays with a person), knowledge search, and pipelines as tools.
The form is the agent_loop node's own settings, so a field added to the node appears here without a change in this file.
Runs that share a toolset (or, without one, a chat connection) share one engine lane, so they execute one at a time in submission order.
Data at data/ai_tools/agent/ (one small JSON record per run; the job record holds everything else).
"""
import re, json, uuid, copy, traceback
from datetime import datetime
from pathlib import Path
from fastapi import APIRouter, Request

TOOL_META = {"label": "Agent", "icon": "&#x2387;", "group": "model_tools", "description": "Queue tasks for a local tool-using agent and review what it did", "singleton": True}

router = APIRouter(redirect_slashes=False)
DATA_DIR = Path("./data/ai_tools/agent")
_OLD_DATA_DIR = Path("./data/ai_tools/code_agent")   # where run records lived while this was the Code Agent; moved once at startup
_LIVE = ("queued", "running", "stopping")
ENV = {}
UI = IM = BI = AIM = AL = _FIELDS = None

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

def _pipeline_opts(values=None): return [("", "(built-in: one agent_loop node)")] + [(p["id"], p.get("name", p["id"])) for p in AIM.engine.list_pipelines()]

def _node_fields() -> list: return [f for f in AIM.steps.get_node_type("agent_loop")["config_schema"] if f.name != "key_map"]

def _refreshing(f):
    """A copy of a field whose change re-renders the form, so options that follow it (a connection's models) update at once."""
    f = copy.copy(f); f.hx_intent, f.hx_target = "agent_form_refresh", "none"
    return f

def _build_fields():
    F = BI.SettingField
    return BI.SettingsGroup("agent", "", [*(_refreshing(f) if f.name == "conn_id" else f for f in _node_fields()), F("base", "Branch From (ref)", "text", "", advanced=True, hint="Toolset connections with sessions (git branches) only. Blank = the node's BASE_REF (normally its current HEAD)."),
        F("pipeline_id", "Pipeline", "select", "", options=_pipeline_opts, advanced=True, hint="Instead of the built-in node: a saved pipeline that gets the task as 'input' and produces the agent_loop keys.")], json_path="")

def _flow(s: dict) -> dict:
    cfg = {f.name: s.get(f.name, f.default) for f in _node_fields()}
    if s.get("base"): cfg["session_args_json"] = {**(cfg.get("session_args_json") or {}), "base": s["base"]}
    return {"nodes": [{"id": "agent", "name": "Agent", "type": "agent_loop", "config": cfg, "key_map": {}, "status": "idle"}]}

def _lane(s: dict) -> str: return f"""agent:{s.get("tools_conn_id") or s.get("conn_id") or "pool"}"""

# --- HTML ---

def _post(action: str, **extra) -> str: return f"""hx-post="/im/in" hx-target="body" hx-swap="none" hx-vals='{json.dumps({"type": f"agent_{action}", "lvl": 2, **extra})}'"""

def _status_badge(status: str) -> str: return f"""<span class="status-badge" style="color:{ {"done": "#00ffa2", "error": "#ff5f5f", "running": "#ffcc00", "queued": "var(--text_muted)"}.get(status, "var(--text)") }">{UI.escape(status)}</span>"""

def _run_row(r: dict, active: str) -> str:
    """One run in the list: opens on click; the delete button asks first. A run still working cannot be deleted (stop it first)."""
    status, title = _job(r)["status"], UI.escape(r["task"][:70])
    delete = "" if status in _LIVE else f"""<button class="cm-qbtn" {_post("delete", rid=r["id"])} hx-confirm="Delete this run and its job record? Changes it proposed stay in their folder's review list." onclick="event.stopPropagation()">&#x2715;</button>"""
    return f"""<div class="side-list-item{" active" if r["id"] == active else ""}" {_post("open", rid=r["id"])}>
                   <div style="display:flex;align-items:center;gap:.2rem;width:100%"><span class="side-list-item-title" style="flex:1">{title}</span><span class="side-list-actions">{delete}</span></div>
                   <div class="status-label">{_status_badge(status)} {r["created"][:16].replace("T", " ")}</div>
               </div>"""

def _list_html(username: str, active: str = "") -> str: return f"""<div class="list-panel-hd"><span class="list-panel-title">Runs</span><button class="btn-icon" {_post("home")} title="New task">+</button></div>{"".join(_run_row(r, active) for r in _runs(username)) or '<div class="list-empty">No runs yet.</div>'}"""

def _form_html(values: dict, task: str = "", again_of: str = "") -> str:
    """The task form; with again_of, prefilled from that run (Edit & run) and the new run records where it came from."""
    return f"""<form {_post("queue", **({"again_of": again_of} if again_of else {}))} hx-include="this" class="node-form glass" style="padding:.6rem">
                   <span class="form-title">{"Edit and run again" if again_of else "New task"}</span>
                   <textarea name="task" class="cm-input" rows="5" placeholder="What should the agent do? Name the files, functions or data it should work on when you know them." required>{UI.escape(task)}</textarea>
                   <div id="ca-fields">{_FIELDS.render(values, name_prefix="ca_")}</div><span id="ca-form-msg"></span>
                   <div class="form-actions"><button type="submit" class="ui-btn">Queue</button><button type="button" class="ui-btn" {_post("sessions")} hx-include="closest form">Workspace sessions</button></div>
               </form>"""

# --- Transcript (live while a run works, at a chosen level of detail) ---

_DETAIL = [("steps", "Steps only"), ("results", "Steps, results expandable"), ("full", "Everything (system prompt, thinking, all expanded)")]

def _poller(rid: str, seen: int) -> str: return f"""<div {_post("status", rid=rid, seen=seen)} hx-trigger="load delay:3s"></div>"""

def _messages(job: dict) -> list: return job["data"].get("transcript") or next((n["live"]["transcript"] for n in job["flow"]["nodes"] if n.get("live")), [])   # finished: the result; running: what the node has published so far

def _calls(m: dict) -> list: return [(c["function"]["name"], c["function"]["arguments"]) for c in m.get("tool_calls") or []] or ([(c["name"], c["arguments"]) for c in AL._text_calls(str(m.get("content", "")))] if m["role"] == "assistant" else [])

_RESP_TAG = re.compile(r"</?tool_response[^>]*>")

def _row_html(i: int, m: dict, detail: str) -> str:
    """One message: steps = a single line; results = a line that expands to the whole message; full = expanded, with the model's thinking."""
    content, calls = str(m.get("content", "")), _calls(m)
    who = {"system": "system prompt", "assistant": "model" + (" -> " + ", ".join(f"""{n}({json.dumps(a, ensure_ascii=False)[:160]})""" for n, a in calls) if calls else ""), "tool": f"""result: {m.get("tool_name", "")}""", "user": "task" if i == 1 else ("result: " + ", ".join(re.findall(r'<tool_response name="([^"]*)"', content)) if "<tool_response" in content else "to model")}[m["role"]]
    first = next((l.strip() for l in _RESP_TAG.sub("", AL._CALL_RE.sub("", content)).splitlines() if l.strip()), "")[:200]
    line = f"""<b class="ca-mono">{i}</b> {UI.escape(who)}{": " + UI.escape(first) if first else ""}"""
    if detail == "steps": return f"""<div class="ca-row">{line}</div>"""
    body = content + "".join(f"""\n-> {n} {json.dumps(a, indent=1, ensure_ascii=False)}""" for n, a in calls if m.get("tool_calls"))   # structured calls are not in the content; text-protocol calls already are
    thinking = f"""<div class="status-label">thinking</div><pre class="ca-pre ca-think">{UI.escape(m["_thinking"])}</pre>""" if detail == "full" and m.get("_thinking") else ""
    return f"""<details class="ca-row"{" open" if detail == "full" else ""}><summary>{line}</summary>{thinking}<pre class="ca-pre">{UI.escape(body)}</pre></details>"""

def _rows(messages: list, detail: str, start: int = 0) -> str: return "".join(_row_html(i, m, detail) for i, m in enumerate(messages) if i >= start and (i or detail == "full"))

def _option(v: str, label: str, selected: bool) -> str: return f"""<option value="{v}"{" selected" if selected else ""}>{label}</option>"""
def _view_controls(rid: str, v: dict) -> str:
    """Detail level and Follow (the transcript stays at its newest line while you are at the bottom; scrolling up holds your place). Both are per login and post together."""
    return f"""<span class="ca-view"><label class="ca-follow-tgl"><input type="checkbox" name="follow" value="1"{" checked" if v["follow"] else ""} {_post("view", rid=rid)} hx-trigger="change" hx-include="closest .ca-view"> Follow</label><select name="detail" class="module-select ca-detail" {_post("view", rid=rid)} hx-trigger="change" hx-include="closest .ca-view">{"".join(_option(k, l, k == v["detail"]) for k, l in _DETAIL)}</select></span>"""

def _live_bar(rec: dict, job: dict, v: dict, seen: int) -> str:
    live = next((n.get("live") or {} for n in job["flow"]["nodes"] if n.get("live")), {})
    doing = "waiting for earlier runs on this toolset" if job["status"] == "queued" else f"""working - step {live.get("steps", 0)}"""
    return f"""<div class="info-bar">{_status_badge(job["status"])} <span>{doing}</span>{_view_controls(rec["id"], v)}<button class="ui-btn" {_post("stop", rid=rec["id"])}>&#x25FC; Stop</button></div>{_poller(rec["id"], seen)}"""

def _transcript_box(messages: list, v: dict) -> str: return f"""<div class="status-label">Transcript ({len(messages)} messages)</div><div class="log-box ca-log{" ca-follow" if v["follow"] else ""}"><div id="ca-tx">{_rows(messages, v["detail"]) or "(nothing yet)"}</div></div>"""   # ca-follow: column-reverse scroll container - the browser keeps the view at the bottom as rows are appended, unless it has been scrolled up

def _result_html(rec: dict, job: dict, v: dict) -> str:
    data, status, msgs = job.get("data", {}), job["status"], _messages(job)
    if status in _LIVE: return f"""<div id="ca-live-bar">{_live_bar(rec, job, v, len(msgs))}</div>{_transcript_box(msgs, v)}"""
    ws = data.get("workspace") or {}
    branch = UI.escape(ws.get("branch", ""))
    err = next((f"""{n.get("name") or n["id"]}: {n.get("message", "")}""" for n in job["flow"]["nodes"] if n.get("status") == "error"), "")
    files = "".join(f"""<tr><td>{UI.escape(f["status"])}</td><td class="ca-mono">{UI.escape(f["path"])}</td><td><button class="ui-btn" {_post("diff", rid=rec["id"], path=f["path"])}>Diff</button></td></tr>""" for f in ws.get("files", []))
    review = f"""<div class="info-bar"><span>branch <b class="ca-mono">{branch}</b></span><span>from <span class="ca-mono">{UI.escape(ws.get("base", "")[:10])}</span></span><span>{"commit " + UI.escape(ws["commit"][:10]) if ws.get("commit") else "no changes committed"}</span></div>
                 <table class="data-table">{files}</table>
                 <div class="form-actions"><button class="ui-btn" {_post("diff", rid=rec["id"])}>Full diff</button><button class="ui-btn" style="color:#ff5f5f" {_post("discard", rid=rec["id"])} hx-confirm="Delete branch {branch}? The agent's work on it is lost.">Discard branch</button></div>
                 <div class="status-label">Merging is yours: on the repo host, <span class="ca-mono">git diff HEAD...{branch}</span> then <span class="ca-mono">git merge {branch}</span>.</div>""" if branch and not rec.get("discarded") else ("""<div class="status-label">Branch discarded.</div>""" if branch else "")
    err_html = f"""<div class="err-box">{UI.escape(err)}</div>""" if err else ""
    space = f"""<span>space <b class="ca-mono">{UI.escape(data["space"])}</b></span>""" if data.get("space") else ""
    summary = f"""<div class="glass ca-summary">{UI.escape(str(data["summary"]))}</div>""" if data.get("summary") else ""
    return f"""<div class="info-bar">{_status_badge(status)} <span>{UI.escape(str(data.get("agent_status", "")))}</span><span>{data.get("steps", "?")} step(s)</span>{space}<button class="ui-btn" {_post("again", rid=rec["id"])} title="Queue this task again with the same settings, as a new run">&#x21BB; Run again</button><button class="ui-btn" {_post("edit", rid=rec["id"])} title="Open this run's task and settings in the form, to change them before queueing a new run">&#x270E; Edit &amp; run</button>{_view_controls(rec["id"], v)}</div>
               {err_html}{summary}
               {review}<div id="ca-diff"></div>{_staged_html(rec, data)}
               {_transcript_box(msgs, v)}"""

def _staged_html(rec: dict, data: dict) -> str:
    if not data.get("files_root"): return ""
    return f"""<div class="status-label">Proposed changes in <b class="ca-mono">{UI.escape(data["files_root"])}</b> ({len(data.get("staged") or [])} from this run). The folder's review store is shared: other pending changes show here too, with their author.</div>
               <div id="ca-shadow">{BI.shadow_review_html(AL.shadow_for(data["files_root"]), "agent_shadow", {"rid": rec["id"]}, list_id="ca-shadow")}</div>"""

def _run_html(rec: dict, v: dict) -> str:
    job = _job(rec)
    live = f"""<div class="status-label">Model output as it is written</div><pre class="log-box ca-pre ca-live" data-stream-job="{rec["job_id"]}" data-stream-node="agent"></pre>""" if job["status"] in _LIVE else ""
    return f"""<div class="ca-run"><div class="ca-task">{UI.escape(rec["task"])}</div>{live}<div id="ca-result">{_result_html(rec, job, v)}</div></div>"""

def _close_btn(conn_id: str, s: dict) -> str: return f"""<button class="ui-btn" {_post("session_close", conn_id=conn_id, session=s["id"])} hx-confirm="Commit what is in this worktree to its branch and remove the worktree? Only do this when no run is still using it - a running agent would fail on its next step.">Close &amp; commit</button>""" if s["status"] == "open" else ""

def _sessions_html(conn_id: str, sessions: list) -> str:
    rows = "".join(f"""<tr><td class="ca-mono">{UI.escape(s["branch"])}</td><td>{_status_badge(s["status"])}</td><td>{UI.escape(s.get("label", ""))}</td><td>{s["opened"][:16].replace("T", " ")}</td><td>{_close_btn(conn_id, s)}</td></tr>""" for s in sessions)
    return f"""<div class="status-label">Sessions on this workspace. An <b>open</b> session with no running job was cut off by a restart - close it to keep its work on its branch.</div><table class="data-table"><tr><th>Branch</th><th>Status</th><th>Label</th><th>Opened</th><th></th></tr>{rows}</table>"""

def _runs_bar(username: str) -> str: return UI.toolbar(side="right", content=f"""<div id="ca-list" class="ca-list">{_list_html(username)}</div>""", size="16rem", overlay=False, start_open=True, id="ca-runs-bar", nesting_level=2)

def _page_html(values: dict) -> str: return f"""<div id="ca-main" class="ca-main">{_form_html(values)}</div>"""

# --- Intent handlers ---

async def _view(request) -> dict:
    """Transcript view settings per login: detail level and follow (on unless turned off)."""
    return {"detail": await ENV["get_state"](request, scope="user", namespace="agent", key="detail") or "results", "follow": await ENV["get_state"](request, scope="user", namespace="agent", key="follow") is not False}

async def _defaults(request) -> dict: return await ENV["get_state"](request, scope="user", namespace="agent", key="defaults") or {f.name: f.default for f in _FIELDS.fields}

async def _h_home(request, payload, imr): return imr.oob(_form_html(await _defaults(request)), "ca-main")

async def _h_queue(request, payload, imr):
    """Queues the form's task. A failure is shown under the form (and its traceback printed) - the dispatcher would otherwise only log it, leaving the Queue button looking dead."""
    task = str(payload.get("task", "")).strip()
    if not task: return imr.oob('<span class="err-box">Write a task first.</span>', "ca-form-msg")
    try:
        s = _FIELDS.parse(payload, name_prefix="ca_")
        await ENV["set_state"](request, s, scope="user", namespace="agent", key="defaults")
        return await _submit(request, imr, task, s, again_of=str(payload.get("again_of", "")))
    except Exception as e:
        traceback.print_exc()
        return imr.oob(f"""<span class="err-box">Not queued - {UI.escape(type(e).__name__)}: {UI.escape(str(e))}</span>""", "ca-form-msg")

async def _submit(request, imr, task: str, s: dict, again_of: str = ""):
    """Queues one run of task with settings s and opens it. again_of names the run this one repeats."""
    username = request.state.user.username
    if s.get("files_root") and request.state.user.role != "admin": return imr.oob('<div class="err-box">Only an admin can give the agent a Files folder.</div>', "ca-form-msg")
    kind, extra = ("id", {"pipeline_id": s["pipeline_id"]}) if s.get("pipeline_id") else ("inline", {"inline_flow": _flow(s)})
    job_id, err = AIM.engine.submit(username, kind=kind, inputs={"input": task}, lane=_lane(s), **extra)
    if err: return imr.oob(f"""<div class="err-box">{UI.escape(err)}</div>""", "ca-main")
    rec = {"id": f"car_{uuid.uuid4().hex[:8]}", "username": username, "task": task, "job_id": job_id, "settings": s, "created": datetime.utcnow().isoformat(), **({"again_of": again_of} if again_of else {})}
    _save(rec)
    await ENV["set_state"](request, rec["id"], scope="user", namespace="agent", key="open_rid")
    imr.oob(_run_html(rec, await _view(request)), "ca-main")
    return imr.oob(_list_html(username, rec["id"]), "ca-list")

async def _h_again(request, payload, imr):
    """Runs a finished run's task again with the same settings, as a new run - e.g. after the model server died mid-run. The earlier run and anything it proposed stay as they are."""
    rec = _owned(request, payload.get("rid", ""))
    return await _submit(request, imr, rec["task"], rec["settings"], again_of=rec["id"]) if rec and _job(rec)["status"] not in _LIVE else imr

async def _h_edit(request, payload, imr):
    """A run's task and settings loaded into the form, to change before queueing - the new run records which run it came from."""
    rec = _owned(request, payload.get("rid", ""))
    if not rec: return imr
    return imr.oob(_form_html({**await _defaults(request), **rec["settings"]}, rec["task"], rec["id"]), "ca-main")

async def _h_delete(request, payload, imr):
    """Deletes a run record and its job record (the owner's own data). Proposed changes stay in the folder's review store - they belong to that folder's review, not to the run."""
    rec, username = _owned(request, payload.get("rid", "")), request.state.user.username
    if not rec or _job(rec)["status"] in _LIVE: return imr
    AIM.engine.delete_job(rec["job_id"]); _rp(rec["id"]).unlink()
    if payload.get("rid") == await ENV["get_state"](request, scope="user", namespace="agent", key="open_rid"): imr.oob(_form_html(await _defaults(request)), "ca-main")
    return imr.oob(_list_html(username), "ca-list")

async def _h_open(request, payload, imr):
    rec = _owned(request, payload.get("rid", ""))
    if not rec: return imr
    await ENV["set_state"](request, rec["id"], scope="user", namespace="agent", key="open_rid")
    imr.oob(_run_html(rec, await _view(request)), "ca-main")
    return imr.oob(_list_html(request.state.user.username, rec["id"]), "ca-list")

async def _h_status(request, payload, imr):
    """Poll target while a run is live: appends the messages published since the last poll (rows already shown keep their scroll and open state) and renews the bar, which carries the next poll.
    Once the run settles, the whole result block and the run list are re-rendered."""
    rec = _owned(request, payload.get("rid", ""))
    if not rec: return imr
    job, v = _job(rec), await _view(request)
    if job["status"] not in _LIVE: imr.oob(_result_html(rec, job, v), "ca-result"); return imr.oob(_list_html(request.state.user.username, rec["id"]), "ca-list")
    msgs, seen = _messages(job), int(payload.get("seen") or 0)
    if len(msgs) > seen: imr.oob(_rows(msgs, v["detail"], seen) if seen else _rows(msgs, v["detail"]), "ca-tx", swap="beforeend" if seen else "innerHTML")
    return imr.oob(_live_bar(rec, job, v, len(msgs)), "ca-live-bar")

async def _h_view(request, payload, imr):
    """Detail level / Follow changed: stored per login, the open run re-rendered with them."""
    rec, v = _owned(request, payload.get("rid", "")), {"detail": payload.get("detail") if payload.get("detail") in dict(_DETAIL) else "results", "follow": bool(payload.get("follow"))}
    for k, val in v.items(): await ENV["set_state"](request, val, scope="user", namespace="agent", key=k)
    return imr.oob(_result_html(rec, _job(rec), v), "ca-result") if rec else imr

async def _h_stop(request, payload, imr):
    rec = _owned(request, payload.get("rid", ""))
    if rec: AIM.engine.stop(rec["job_id"])
    return imr.oob(_result_html(rec, _job(rec), await _view(request)), "ca-result") if rec else imr

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
    return imr.oob(_result_html(rec, job, await _view(request)), "ca-result")

async def _h_sessions(request, payload, imr):
    conn_id = payload["ca_tools_conn_id"] if payload.get("ca_tools_conn_id") not in (None, "", "pool") else next((c["_id"] for c in AIM.connections.conns_matching("agent_tools_call")), "")
    if not conn_id: return imr.oob('<div class="err-box">No toolset connection configured - add one (e.g. a Code workspace) in AI Tools > Settings.</div>', "ca-main")
    r = await _call(conn_id, "agent_session_list", {})
    return imr.oob(f"""<div class="err-box">{UI.escape(r["error"])}</div>""" if r.get("error") else _sessions_html(conn_id, r["sessions"]), "ca-main")

async def _h_form_refresh(request, payload, imr):
    """The chat connection changed: re-render the fields with the values typed so far, so the Model list is that connection's. The model is cleared - the old one belongs to the old connection."""
    try: s = _FIELDS.parse(payload, name_prefix="ca_")
    except BI.SettingsError as e: return imr.oob(f"""<span class="err-box">{UI.escape(str(e))}</span>""", "ca-form-msg")
    imr.oob("", "ca-form-msg")
    return imr.oob(_FIELDS.render({**s, "model": ""}, name_prefix="ca_"), "ca-fields")

async def _h_shadow(request, payload, imr):
    """Diff / accept / reject one proposed change. Accepting writes to the real folder (the previous version is kept in the store's history), so it is an admin's action."""
    rec = _owned(request, payload.get("rid", ""))
    root = (_job(rec)["data"].get("files_root") if rec else "") or ""
    if not root: return imr
    shadow, action, path = AL.shadow_for(root), payload.get("action", ""), payload.get("path", "")
    if action == "diff": return imr.oob(f"""<pre class="ca-pre">{UI.escape(shadow.diff(path)) or "(no differences)"}</pre>""", payload.get("diff_target", ""))
    if action == "accept" and request.state.user.role != "admin": return imr.oob('<div class="err-box">Only an admin can accept changes into the folder.</div>', "ca-diff")
    if action == "accept": shadow.accept(path)
    elif action == "reject": shadow.reject(path)
    return imr.oob(BI.shadow_review_html(shadow, "agent_shadow", {"rid": rec["id"]}, list_id="ca-shadow"), "ca-shadow")

async def _h_session_close(request, payload, imr):
    r = await _call(payload.get("conn_id", ""), "agent_session_close", {"session": payload.get("session", ""), "message": "closed from Agent after an interrupted run"})
    if r.get("error"): return imr.oob(f"""<div class="err-box">{UI.escape(r["error"])}</div>""", "ca-main")
    return await _h_sessions(request, {"ca_tools_conn_id": payload.get("conn_id", "")}, imr)

# --- Init / route ---

def init_tool(env: dict, prefix: str):
    global ENV, UI, IM, BI, AIM, AL, _FIELDS
    ENV = env
    UI = env["templates"].env.globals.get("UI")
    BI = env["tools"]["built_ins"]
    AIM = env["tools"]["ai_manager"]
    from tools.ai_manager import agent_loop as AL   # the node's own module: its review store for Files folders
    if _OLD_DATA_DIR.is_dir() and not DATA_DIR.exists(): _OLD_DATA_DIR.rename(DATA_DIR)
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    _FIELDS = _build_fields()
    IM = env["InterfaceManager"](nesting_level=2, db_path="agent_im.db")
    IM.scripts.update({f"agent_{k}": [fn] for k, fn in {"home": _h_home, "queue": _h_queue, "open": _h_open, "status": _h_status, "stop": _h_stop, "diff": _h_diff, "discard": _h_discard, "sessions": _h_sessions, "session_close": _h_session_close, "form_refresh": _h_form_refresh, "shadow": _h_shadow, "view": _h_view, "delete": _h_delete, "again": _h_again, "edit": _h_edit}.items()})
    print("[agent] ready")

@router.get("")
@router.get("/")
async def root(request: Request):
    return ENV["templates"].TemplateResponse(name="base.html", request=request, context={"request": request, "user": request.state.user, "nesting_level": 2, "shell_id": IM.branch_id, "toolbars": {"right": _runs_bar(request.state.user.username)}, "content": _page_html(await _defaults(request)), "extra_css": CSS})

CSS = """
.ca-main{height:100%;overflow-y:auto;padding:.6rem;box-sizing:border-box;display:flex;flex-direction:column;gap:.5rem;}
.ca-list{height:100%;overflow-y:auto;}
.ca-run{display:flex;flex-direction:column;gap:.5rem;}
.ca-task{white-space:pre-wrap;font-weight:600;}
.ca-mono{font-family:var(--font-mono);}
.ca-pre{white-space:pre-wrap;margin:0;overflow-wrap:anywhere;}
.ca-live{max-height:12rem;}
.ca-diff{max-height:50vh;}
.ca-log{max-height:60vh;resize:vertical;}
.ca-row{padding:.15rem 0;border-bottom:var(--border-thick) solid var(--border);overflow-wrap:anywhere;}
.ca-row summary{cursor:pointer;}
.ca-row .ca-pre{max-height:40vh;overflow-y:auto;padding:.2rem .4rem;}
.ca-think{opacity:.75;font-style:italic;}
.ca-view{display:flex;align-items:center;gap:.4rem;margin:0 0 0 auto;}
.ca-detail{width:auto;margin:0;}
.ca-follow-tgl{white-space:nowrap;font-size:.8rem;}
.ca-follow{display:flex;flex-direction:column-reverse;}
.ca-summary{padding:.5rem .7rem;white-space:pre-wrap;}
"""
