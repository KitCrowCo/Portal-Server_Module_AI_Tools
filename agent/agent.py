# /modules/ai_tools/agent/agent.py
"""
Agent - task queue and review surface for agent_loop runs. Sub-module of ai_tools, mounted at /module/ai_tools/agent.
A run is one pipeline job: the built-in single agent_loop node, or any saved pipeline that takes 'input' (the task) and produces the agent_loop keys (summary, agent_status, steps, transcript, workspace, branch, space).
What the agent can do is its toolsets: its own space, a toolset connection (e.g. a code workspace, where each run works on its own ai/<id> branch and merging stays with a person), knowledge search, and pipelines as tools.
The form is the agent_loop node's own settings, so a field added to the node appears here without a change in this file.
Runs that share a toolset (or, without one, a chat connection) share one engine lane, so they execute one at a time in submission order.
Data at data/ai_tools/agent/ (one small JSON record per run; the job record holds everything else).
"""
import re, json, uuid, copy, shutil, asyncio, traceback
from datetime import datetime
from pathlib import Path
from fastapi import APIRouter, Request

TOOL_META = {"label": "Agent", "icon": "&#x2387;", "group": "model_tools", "description": "Queue tasks for a local tool-using agent and review what it did", "singleton": True}

router = APIRouter(redirect_slashes=False)
DATA_DIR = Path("./data/ai_tools/agent")
_LIVE = ("queued", "running", "stopping")
_WATCHERS = set()   # background report tasks; held here because the event loop keeps only weak references to tasks
ENV = {}
UI = IM = BI = AIM = AL = MN = _FIELDS = None

# --- Run records ---

def _rp(rid): return DATA_DIR / f"{Path(rid).name}.json"
def _load(rid): return json.loads(_rp(rid).read_text()) if rid and _rp(rid).exists() else None
def _save(rec): _rp(rec["id"]).write_text(json.dumps(rec, indent=2))
def _runs(username): return sorted((r for r in (json.loads(f.read_text()) for f in DATA_DIR.glob("*.json")) if r.get("username") == username), key=lambda r: r["created"], reverse=True)
def _owned(request, rid): rec = _load(rid); return rec if rec and rec.get("username") == request.state.user.username else None
def _job(rec): return AIM.engine.load_job(rec["job_id"]) or {"status": "missing", "data": {}, "flow": {"nodes": []}, "log": []}
def _author(rec): return f"""agent:{rec["settings"].get("author") or rec["job_id"]}"""   # whose proposals in the Files review store are this run's (a project's runs share one)
def _root(rec, job): return job.get("data", {}).get("files_root") or rec["settings"].get("files_root") or ""   # the run's Files folder - from its settings while it works or when it ended without a result
def _accepted(root: str, paths: list):
    """Accepted changes are code changes like any other: a dev instance restarts for them (Git Manager's git_sync.code_saved decides by the sync config)."""
    saved = ENV["services"].get("git_sync", {}).get("code_saved")
    cwd = Path.cwd().resolve()
    for rel in paths:
        f = (Path(root) / rel).resolve()
        if saved and f.is_relative_to(cwd) and saved(str(f.relative_to(cwd))): break   # one restart covers them all

def _mine(rec, job) -> list: r = _root(rec, job); return [e for e in AL.shadow_for(r).pending() if e.get("author") == _author(rec)] if r else []

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
        F("pipeline_id", "Pipeline", "select", "", options=_pipeline_opts, advanced=True, hint="Instead of the built-in node: a saved pipeline that gets the task as 'input' and produces the agent_loop keys."),
        F("muninn_thread", "Muninn thread", "text", "", hint="When set, the run's outcome (status, task, summary) is added to this Muninn thread once the run finishes. Needs the Muninn tool.")], json_path="")

def _flow(s: dict) -> dict:
    cfg = {f.name: s.get(f.name, f.default) for f in _node_fields()}
    if s.get("base"): cfg["session_args_json"] = {**(cfg.get("session_args_json") or {}), "base": s["base"]}
    return {"nodes": [{"id": "agent", "name": "Agent", "type": "agent_loop", "config": cfg, "key_map": {}, "status": "idle"}]}

def _lane(s: dict) -> str: return f"""agent:{s.get("tools_conn_id") or s.get("conn_id") or "pool"}"""

# --- Muninn (a run's outcome added to a thread, once) ---

def _report(rid: str):
    """Adds a settled run's outcome to its Muninn thread. Idempotent: the record keeps the capture's id, and it is re-read here so two callers cannot both add it."""
    rec = _load(rid)
    if not rec or not rec["settings"].get("muninn_thread") or rec.get("muninn_entry"): return
    job = _job(rec)
    if job["status"] in _LIVE: return
    d = job.get("data", {})
    err = next((f"""{n.get("name") or n["id"]}: {n.get("message", "")}""" for n in job["flow"]["nodes"] if n.get("status") == "error"), "")
    text = f"""Agent run {job["status"]}{" - " + str(d["agent_status"]) if d.get("agent_status") else ""}, {d.get("steps", "?")} step(s): {rec["task"][:300]}\n\n{str(d.get("summary") or err or "(no summary)")[:4000]}"""
    rec["muninn_entry"] = MN.capture(rec["username"], text, source="agent", author="agent", refs=[f"""agent:{rec["id"]}"""], threads=[rec["settings"]["muninn_thread"]], plain=True)["id"]
    _save(rec)

def _spawn(coro):
    t = asyncio.create_task(coro)
    _WATCHERS.add(t); t.add_done_callback(_WATCHERS.discard)

async def _report_when_settled(rid: str):
    """Waits for the run in the background (it may sit queued behind other runs for hours), then reports it. Runs cut off by a restart are reported by the startup sweep in init_tool."""
    while (rec := _load(rid)) and _job(rec)["status"] in _LIVE: await asyncio.sleep(15)
    if rec: _report(rid)

# --- Runs for any caller (the form, and other tools through the "agent" service) ---

def submit_run(username: str, task: str, settings: dict, again_of: str = "", origin: str = "", label: str = "", resume: list = None) -> dict:
    """Queues one run and returns its record. settings are the form's values (missing ones take the form's defaults); origin names a tool that queued it (shown with the run); label is shown instead of the task's first words.
    resume: an earlier run's conversation (after its system prompt) - the new run continues it (Continue). Raises RuntimeError when it cannot be queued."""
    s = {**{f.name: f.default for f in _FIELDS.fields}, **settings}
    if s.get("muninn_thread") and not MN: raise RuntimeError("a Muninn thread is set but the Muninn tool is not installed - clear the field or install tools/muninn")
    kind, extra = ("id", {"pipeline_id": s["pipeline_id"]}) if s.get("pipeline_id") else ("inline", {"inline_flow": _flow(s)})
    job_id, err = AIM.engine.submit(username, kind=kind, inputs={"input": task, **({"resume_messages": resume} if resume else {})}, lane=_lane(s), **extra)
    if err: raise RuntimeError(err)
    rec = {"id": f"car_{uuid.uuid4().hex[:8]}", "username": username, "task": task, "job_id": job_id, "settings": s, "created": datetime.utcnow().isoformat(), **({"again_of": again_of} if again_of else {}), **({"origin": origin} if origin else {}), **({"label": label} if label else {})}
    _save(rec)
    if s.get("muninn_thread"): _spawn(_report_when_settled(rec["id"]))
    return rec

def run_status(rid: str) -> dict:
    """{"status", "summary", "task", "created"} of a run; None once it was deleted."""
    rec = _load(rid)
    if not rec: return None
    job = _job(rec)
    return {"status": job["status"], "summary": str(job.get("data", {}).get("summary", "")), "task": rec["task"], "created": rec["created"]}

def last_settings(username: str) -> dict:
    """The settings of the newest run username queued from the form themselves; None when there is none."""
    return next((r["settings"] for r in _runs(username) if not r.get("origin")), None)

def settings_summary(s: dict) -> str:
    """One line: model and what the agent can reach with these settings."""
    reach = [x for x in (s.get("space") and f"""space {s["space"]}""", s.get("files_root") and f"""files {s["files_root"]}""", s.get("knowledge_conn_id") and f"""knowledge {s["knowledge_conn_id"]}""", s.get("tools_conn_id") and f"""toolset {s["tools_conn_id"]}""", s.get("pipeline_tool_tag") and f"""pipelines tagged {s["pipeline_tool_tag"]}""") if x]
    return f"""{s.get("conn_id") or "first chat connection"} / {s.get("model") or "auto"} - {", ".join(reach) or "no tools beyond finish"}{" - read-only" if s.get("read_only") else ""}"""

# --- Projects: a goal worked through by a chain of runs - the first plans (plan.md, notes.md in the project's own space), each next run does one unchecked step and records it (log.md) ---
# Every run is a normal run in the list. The runs share one Files author, so each builds on the changes the earlier ones proposed; all of them wait in the folder's review store until a person accepts them.
# A project runs on the terms it was started with: it stops at its run budget, pauses after two runs in a row that changed nothing in plan.md, when a run was stopped or killed, and after a server restart waits for Resume.
# While it is not running: Reverse last step takes the newest run back (its proposed versions, and plan.md as it was before that run), plan.md can be edited, and the project can be deleted (what else goes is ticked by the person).

_PLAN_TASK = """Project goal:
{goal}

This is the project's first run: plan it and change nothing yet.
1. Find where the goal touches the code: search_code, find_symbol, references, file_map, read_symbol. Read only what you need.
2. Write notes.md in your space: the files, functions, intents and conventions that matter (with paths and line numbers), and anything risky.
3. Write plan.md in your space: a checklist, one step per line, each "- [ ] " followed by one change small enough to make and check in one run (one function, or one small new file), naming the file and the function. Order the steps so each builds on the ones before.
4. Call finish with a short summary of the plan."""

_STEP_TASK = """Project goal:
{goal}

You are continuing this project. Earlier runs left plan.md, notes.md and log.md in your space - read them first.
Do the first unchecked step of plan.md, and only that step:
1. Look at the code it touches (read_symbol, references, file_map). The code map shows the files as they are on disk; read_file and read_symbol show this project's pending changes, which earlier steps proposed and you build on.
2. Make the change with propose_edit (propose_file for a new file).
3. Check it: check_files, and references for anything you renamed or whose arguments changed.
4. Update your space: mark the step "- [x]" in plan.md, add one line to log.md (what changed, where, how you checked it), and add to notes.md what later steps need to know.
If the step is wrong, too large or already done, rewrite plan.md instead (split, correct or check off steps) and say why in log.md.
Then call finish."""

_PDIR = DATA_DIR / "projects"
_PROJECT_TASKS = {}   # pid -> its running loop task

def _pp(pid): return _PDIR / f"{Path(pid).name}.json"
def _pload(pid): return json.loads(_pp(pid).read_text()) if pid and _pp(pid).exists() else None
def _psave(p): _pp(p["id"]).write_text(json.dumps(p, indent=2))
def _projects(username): return sorted((p for p in (json.loads(f.read_text()) for f in _PDIR.glob("*.json")) if p.get("username") == username), key=lambda p: p["created"], reverse=True)
def _pset(pid, **kw): p = _pload(pid); p.update(kw); _psave(p); return p
def _space_dir(p: dict): return AL.SPACES_DIR / f"""project-{p["id"]}"""
def _space_file(p: dict, name: str) -> str: f = _space_dir(p) / name; return f.read_text(errors="replace") if f.is_file() else ""
def _project_pending(p: dict) -> list: return [e for e in AL.shadow_for(p["settings"]["files_root"]).pending() if e.get("author") == f"""agent:{p["id"]}"""] if p["settings"].get("files_root") else []
def _open_steps(plan: str) -> int: return len(re.findall(r"(?m)^\s*[-*]\s*\[ \]", plan))

def _project_run_settings(p: dict, planning: bool) -> dict:
    """A step's settings: the project's own, its space and Files author; the planning run may not propose changes, and step runs with a Files folder must pass check_files before finishing."""
    s = {**p["settings"], "space": f"""project-{p["id"]}""", "author": p["id"], "read_only": False}
    if planning: s["deny_tools"] = ",".join(x for x in [s.get("deny_tools", ""), "propose_edit,propose_file"] if x)
    elif s.get("files_root") and not s.get("check_tool"): s["check_tool"] = "check_files"
    return s

async def _project_loop(pid: str):
    while (p := _pload(pid)) and p["status"] == "running":
        if len(p["runs"]) >= int(p["max_runs"]): _pset(pid, status="paused", message=f"""Run budget reached ({p["max_runs"]} runs). Raise it and Resume to go on."""); break
        plan = _space_file(p, "plan.md")
        try: rec = submit_run(p["username"], (_STEP_TASK if plan.strip() else _PLAN_TASK).format(goal=p["goal"]), _project_run_settings(p, not plan.strip()), origin=f"""project: {p["title"]}""", label=f"""{p["title"]}: {f"step {plan.count('[x]') + 1}" if plan.strip() else "plan"}""")
        except RuntimeError as e: _pset(pid, status="paused", message=f"Could not queue the next run: {e}"); break
        p["plan_before"] = {**p.get("plan_before", {}), rec["id"]: plan}   # plan.md as this run found it, for Reverse last step
        p["runs"].append(rec["id"]); p["message"] = f"""Run {len(p["runs"])} {"(planning)" if not plan.strip() else ""} queued."""; _psave(p)
        while _job(rec)["status"] in _LIVE: await asyncio.sleep(10)
        job, p, after = _job(rec), _pload(pid), _space_file(_pload(pid), "plan.md")
        if job["status"] == "stopped" and p["status"] == "running": _pset(pid, status="paused", message="The last run was stopped or killed - look at it, then Resume (or Reverse last step)."); break
        if after.strip() and _open_steps(after) == 0: _pset(pid, status="done", message="Every step in plan.md is checked. Review the proposed changes below."); break
        stalls = 0 if after != plan and job["status"] == "done" else p.get("stalls", 0) + 1
        p = _pset(pid, stalls=stalls, message=f"""Last run: {job["status"]}{" - " + str(job.get("data", {}).get("agent_status", "")) if job.get("data") else ""}. {_open_steps(after)} step(s) open.""")
        if stalls >= 2: _pset(pid, status="paused", message="Two runs in a row changed nothing in plan.md (or failed). Look at the last runs, adjust the goal or plan.md (in the project's space), then Resume."); break
        if p["status"] == "running" and float(p.get("gap_min") or 0): await asyncio.sleep(float(p["gap_min"]) * 60)
    _PROJECT_TASKS.pop(pid, None)

def _start_loop(pid: str):
    if pid not in _PROJECT_TASKS: _PROJECT_TASKS[pid] = asyncio.create_task(_project_loop(pid))   # held here: the event loop keeps only weak references to tasks

# --- Command: "@agent <task>" in a Muninn capture queues a run with the settings of your last run; the outcome comes back to the capture's thread ---

_CMD_HELP = "@agent <task> - queues an Agent run with the settings of your last run in Agent; the outcome comes back to the capture's #thread (or #agent-runs)"

async def _cmd_parse(request, text: str, ctx: dict) -> dict:
    s, thread = last_settings(request.state.user.username), (ctx.get("threads") or ["agent-runs"])[0]
    if not text.strip(): return {"error": "write the task after @agent"}
    if not s: return {"error": "queue one run in Agent first - @agent uses the settings of your last run there"}
    return {"summary": f"Queue an agent run: {settings_summary(s)}; the outcome goes to #{thread}", "args": {"task": text.strip(), "thread": thread}}

async def _cmd_run(request, args: dict) -> str:
    user = request.state.user.username
    try: rec = submit_run(user, args["task"], {**last_settings(user), "muninn_thread": args["thread"]}, origin="muninn")
    except RuntimeError as e: raise ValueError(str(e))
    return f"""queued agent run {rec["id"]} (see Agent) - the outcome comes back to #{args["thread"]}"""

# --- HTML ---

def _post(action: str, **extra) -> str: return f"""hx-post="/im/in" hx-target="body" hx-swap="none" hx-vals='{json.dumps({"type": f"agent_{action}", "lvl": 2, **extra})}'"""

def _status_badge(status: str) -> str: return f"""<span class="status-badge" style="color:{ {"done": "#00ffa2", "error": "#ff5f5f", "running": "#ffcc00", "queued": "var(--text_muted)"}.get(status, "var(--text)") }">{UI.escape(status)}</span>"""

def _run_row(r: dict, active: str) -> str:
    """One run in the list: opens on click; the delete button asks first. A run still working is killed first."""
    status, title = _job(r)["status"], UI.escape((r.get("label") or r["task"])[:70])
    delete = f"""<button class="cm-qbtn" {_post("delete", rid=r["id"])} hx-confirm="{"This run is still working - kill it, then delete it and its job record?" if status in _LIVE else "Delete this run and its job record?"} Changes it proposed stay in their folder's review list." onclick="event.stopPropagation()">&#x2715;</button>"""
    return f"""<div class="side-list-item{" active" if r["id"] == active else ""}" {_post("open", rid=r["id"])}>
                   <div style="display:flex;align-items:center;gap:.2rem;width:100%"><span class="side-list-item-title" style="flex:1">{title}</span><span class="side-list-actions">{delete}</span></div>
                   <div class="status-label">{_status_badge(status)} {r["created"][:16].replace("T", " ")}{" &#xB7; by " + UI.escape(r["origin"]) if r.get("origin") else ""}</div>
               </div>"""

def _project_row(p: dict, active: str) -> str:
    """One project in the list: opens on click; the cross opens the project with its delete form."""
    delete = f"""<button class="cm-qbtn" {_post("project_delete", pid=p["id"])} onclick="event.stopPropagation()" title="Delete this project (choose what goes with it)">&#x2715;</button>"""
    return f"""<div class="side-list-item{" active" if p["id"] == active else ""}" {_post("project_open", pid=p["id"])}><div style="display:flex;align-items:center;gap:.2rem;width:100%"><span class="side-list-item-title" style="flex:1">&#x1F5C2; {UI.escape(p["title"])}</span><span class="side-list-actions">{delete}</span></div>
                   <div class="status-label">{_status_badge(p["status"])} {len(p["runs"])} / {p["max_runs"]} runs</div></div>"""

def _list_html(username: str, active: str = "") -> str: return f"""<div class="list-panel-hd"><span class="list-panel-title">Projects</span><button class="btn-icon" {_post("project_new")} title="New project: a goal worked through by a chain of runs">+</button></div>{"".join(_project_row(p, active) for p in _projects(username)) or '<div class="list-empty">No projects yet.</div>'}
               <div class="list-panel-hd"><span class="list-panel-title">Runs</span><button class="btn-icon" {_post("queue_view")} title="Queue: everything waiting or running, lanes, clearing finished runs">&#x2630;</button><button class="btn-icon" {_post("home")} title="New task">+</button></div>{"".join(_run_row(r, active) for r in _runs(username) if not r.get("origin", "").startswith("project:")) or '<div class="list-empty">No runs yet.</div>'}"""   # a project's runs are listed in the project

def _form_html(values: dict, task: str = "", again_of: str = "") -> str:
    """The task form; with again_of, prefilled from that run (Edit & run) and the new run records where it came from."""
    return f"""<form {_post("queue", **({"again_of": again_of} if again_of else {}))} hx-include="this" class="node-form glass" style="padding:.6rem">
                   <span class="form-title">{"Edit and run again" if again_of else "New task"}</span>
                   <textarea name="task" class="cm-input" rows="5" placeholder="What should the agent do? Name the files, functions or data it should work on when you know them." required>{UI.escape(task)}</textarea>
                   <div id="ca-fields">{_FIELDS.render(values, name_prefix="ca_")}</div><span id="ca-form-msg"></span>
                   <div class="form-actions"><button type="submit" class="ui-btn">Queue</button><button type="button" class="ui-btn" {_post("sessions")} hx-include="closest form">Workspace sessions</button></div>
               </form>"""

def _project_form_html(values: dict) -> str:
    return f"""<form {_post("project_start")} hx-include="this" class="node-form glass" style="padding:.6rem">
                   <span class="form-title">New project</span>
                   <div class="status-label">A goal too large for one run. The first run studies the code and writes a plan (plan.md, notes.md in the project's own space); each next run does one step, checks it and records it, building on the changes the earlier runs proposed. Nothing is applied until you accept it in the review list. Set Files to the folder the work is in.</div>
                   <input name="title" class="module-select" placeholder="Short title" required>
                   <textarea name="goal" class="cm-input" rows="6" placeholder="The goal: what should be different when it is done, where (files, modules) if you know, and what must not change." required></textarea>
                   <div class="row"><label class="status-label">Run budget <input type="number" name="max_runs" value="12" min="1" class="module-select" style="width:5rem"></label><label class="status-label">Minutes between runs <input type="number" name="gap_min" value="0" min="0" step="any" class="module-select" style="width:5rem" title="0 = one after another; more lets other work use the model in between"></label></div>
                   <div id="ca-fields">{_FIELDS.render(values, name_prefix="ca_")}</div><span id="ca-form-msg"></span>
                   <div class="form-actions"><button type="submit" class="ui-btn">Start project</button></div>
               </form>"""

def _space_box(p: dict, name: str, open_: bool = False) -> str:
    text = _space_file(p, name)
    return f"""<details class="sec"{" open" if open_ else ""}><summary>{name}{f" - {_open_steps(text)} open step(s)" if name == "plan.md" and text else ""}</summary><pre class="ca-pre log-box">{UI.escape(text) or "(not written yet)"}</pre></details>"""

def _project_live_html(p: dict) -> str:
    """Status, controls and the runs - redrawn every 10 seconds while the project runs."""
    ctl = {"running": f"""<button class="ui-btn" {_post("project_ctl", pid=p["id"], op="pause")} title="The current run finishes, then the project waits">&#x23F8; Pause</button>""",
           "paused": f"""<form class="row" {_post("project_ctl", pid=p["id"], op="resume")} hx-include="this"><label class="status-label">budget <input type="number" name="max_runs" value="{p["max_runs"]}" min="1" class="module-select" style="width:4.5rem"></label><button class="ui-btn">&#x25B6; Resume</button></form>"""}.get(p["status"], "")
    stop = f"""<button class="ui-btn" style="color:#ff5f5f" {_post("project_ctl", pid=p["id"], op="stop")} hx-confirm="Stop the project and its current run? Its proposed changes stay in the review list.">&#x25FC; Stop</button>""" if p["status"] in ("running", "paused") else ""
    runs = "".join(_run_row(r, "") for r in (_load(rid) for rid in reversed(p["runs"])) if r)
    poll = f"""<div {_post("project_live", pid=p["id"])} hx-trigger="load delay:10s"></div>""" if p["status"] == "running" else ""
    return f"""<div class="info-bar">{_status_badge(p["status"])}<span>{len(p["runs"])} / {p["max_runs"]} runs</span><span>{UI.escape(p.get("message", ""))}</span>{ctl}{stop}</div>{poll}
               <div class="status-label">Runs (newest first)</div>{runs or '<div class="list-empty">The first run is being queued.</div>'}"""

def _project_html(p: dict) -> str:
    last = next((r for r in (_load(rid) for rid in reversed(p["runs"])) if r), None)
    review = f"""<div class="status-label">Proposed changes in <b class="ca-mono">{UI.escape(p["settings"]["files_root"])}</b> - this project's runs share the author agent:{p["id"]}. Accept or reject them here or from any of its runs.</div><div class="row"><button class="ui-btn" {_post("project_accept", pid=p["id"])} hx-confirm="Write every pending change of this project into the folder? Files that changed since a change was proposed are left for you to look at.">Accept all of this project's changes</button></div><div id="ca-shadow">{BI.shadow_review_html(AL.shadow_for(p["settings"]["files_root"]), "agent_shadow", {"rid": last["id"]}, list_id="ca-shadow")}</div>""" if p["settings"].get("files_root") and last else ""
    return f"""<div class="ca-run"><div class="row"><span class="form-title grow">&#x1F5C2; {UI.escape(p["title"])}</span><span class="status-label">{settings_summary(p["settings"])}</span></div><div class="ca-task">{UI.escape(p["goal"])}</div>
               <div id="ca-proj-live">{_project_live_html(p)}</div>
               <div class="row"><span class="status-label grow">The project's space (project-{p["id"]}) - what the runs planned, did and learned</span><button class="ui-btn" {_post("project_open", pid=p["id"])}>&#x21BB; Reload</button>{_project_tools(p)}</div>
               {_plan_editor(p) if p["status"] != "running" else _space_box(p, "plan.md", True)}{_space_box(p, "log.md")}{_space_box(p, "notes.md")}<div id="ca-diff"></div>{review}</div>"""

def _project_tools(p: dict) -> str:
    """Reverse last step (only while the project is not running - Pause first) and Delete (always; a running project is stopped first)."""
    last = next((rid for rid in reversed(p["runs"]) if rid not in p.get("reversed", [])), None) if p["status"] != "running" else None
    rev = f"""<button class="ui-btn" {_post("project_reverse", pid=p["id"])} hx-confirm="Take the newest step back? Its proposed versions are removed (earlier steps' versions of those files come back) and plan.md returns to how that run found it. The run itself stays in the list.">&#x21B6; Reverse last step</button>""" if last else ""
    return f"""{rev}<button class="ui-btn" style="color:#ff5f5f" {_post("project_delete", pid=p["id"])}>&#x2715; Delete project</button>"""

def _plan_editor(p: dict) -> str:
    text = _space_file(p, "plan.md")
    return f"""<details class="sec" open><summary>plan.md{f" - {_open_steps(text)} open step(s)" if text else ""} - editable while not running</summary><div class="status-label">Uncheck a step to have it done again, split or rewrite steps, then Save and Resume.</div><form {_post("project_plan", pid=p["id"])} hx-include="this"><textarea name="plan" class="cm-input ca-mono" rows="12" style="width:100%;box-sizing:border-box">{UI.escape(text)}</textarea><div class="form-actions"><button class="ui-btn">Save plan.md</button></div></form></details>"""

# --- Queue: every job waiting or running (an admin sees all users'), lane capacities, clearing finished runs ---

def _job_label(job_id: str, recs: dict) -> str:
    r = recs.get(job_id)
    return UI.escape((r.get("label") or r["task"])[:60]) if r else f"""<span class="status-label">job {UI.escape(job_id)} (not an Agent run)</span>"""

def _queue_row(q: dict, recs: dict, username: str) -> str:
    rec = recs.get(q["id"]) if recs.get(q["id"], {}).get("username") == username else None
    place = "running" if q["place"] == 0 and q["status"] != "queued" else f"#{q['place']} waiting" if q["place"] else "-"
    gone = "" if q["live"] else '<span class="status-label" style="color:#ff9944" title="The record says queued / running but this server has no task for it - a restart or crash left it. Stop or Kill clears it.">left over</span>'
    open_btn = f"""<button class="cm-qbtn" {_post("open", rid=rec["id"])}>Open</button>""" if rec else ""
    return f"""<tr><td>{_status_badge(q["status"])}</td><td>{_job_label(q["id"], recs)}{"" if q["username"] == username else " &#xB7; " + UI.escape(q["username"])}</td><td class="ca-mono">{UI.escape(q["lane"] or "-")}</td><td>{place}</td><td>{gone}</td><td>{q["created"][5:16].replace("T", " ")}</td>
               <td class="row"><button class="cm-qbtn" {_post("queue_job", job=q["id"], op="stop")} title="Stops after the current model turn (clears a left-over record at once)">Stop</button><button class="cm-qbtn" style="color:#ff5f5f" {_post("queue_job", job=q["id"], op="kill")} hx-confirm="Kill this job now?">Kill</button>{open_btn}</td></tr>"""

async def _nodes_html(admin: bool) -> str:
    """Model servers with a control channel (capability node_status, e.g. Magpie): what each is doing right now, asked separately from the requests themselves. An admin can drop a node's requests or kill its model process."""
    conns = AIM.connections.conns_matching("node_status")
    if not conns: return ""
    sts = await asyncio.gather(*(AIM.connections.call_capability(c, "node_status", {}, timeout_s=4) for c in conns))
    def row(c, st):
        name = UI.escape(c.get("display_name", c["_id"]))
        if st.get("error"): return f"""<tr><td>{name}</td><td colspan="4" class="status-label" style="color:#ff5f5f">not answering: {UI.escape(str(st["error"])[:120])}</td></tr>"""
        ll, mem, reqs = st.get("llama", {}), st.get("memory", {}), st.get("requests", [])
        ctl = f"""<button class="cm-qbtn" {_post("node_ctl", conn=c["_id"], scope="requests")} hx-confirm="Drop every request this node is working on?">Drop requests</button><button class="cm-qbtn" style="color:#ff5f5f" {_post("node_ctl", conn=c["_id"], scope="server")} hx-confirm="Kill this node's model process now (no context save)? The next request starts it again.">Kill model process</button>""" if admin and AIM.connections.has_capability(c, "node_kill") else ""
        work = "; ".join(f"""{UI.escape(r["id"])} {UI.escape(r.get("phase", ""))} {r.get("age_s", 0):.0f}s""" for r in reqs) or "idle"
        return f"""<tr><td>{name}<div class="status-label">{UI.escape(str(st.get("node", "")))}</div></td><td>{UI.escape(str(ll.get("model") or "no model loaded"))}{"" if ll.get("alive") else " (not running)"}<div class="status-label">ctx {ll.get("ctx") or "-"} &#xB7; kv {UI.escape(str(ll.get("kv") or "-"))} &#xB7; {ll.get("rss_gb", 0)} GB</div></td>
                   <td class="status-label">{work}</td><td class="status-label">RAM {mem.get("ram_avail_gb", "?")} / {mem.get("ram_total_gb", "?")} GB free &#xB7; GPU {mem.get("gpu_used_gb", "?")} GB<br>{" &#xB7; ".join(f"{k} {v}" for k, v in st.get("counts", {}).items())}</td><td class="row">{ctl}</td></tr>"""
    return f"""<details class="sec" open><summary>Model servers ({len(conns)})</summary><table class="data-table">{"".join(row(c, st) for c, st in zip(conns, sts))}</table></details>"""

def _queue_html(username: str, admin: bool, note: str = "", nodes: str = "") -> str:
    recs = {r["job_id"]: r for r in (json.loads(f.read_text()) for f in DATA_DIR.glob("*.json"))}
    rows = "".join(_queue_row(q, recs, username) for q in AIM.engine.queue_snapshot(None if admin else username))
    lanes = "".join(f"""<tr><td class="ca-mono">{UI.escape(n)}</td><td>{l["running"]} running, {l["waiting"]} waiting</td><td><form class="row" {_post("lane_cap", lane=n)} hx-include="this"><input type="number" name="cap" value="{l["cap"]}" min="1" max="16" class="module-select" style="width:4rem"{"" if admin else " disabled"}>{'<button class="cm-qbtn">Set</button>' if admin else ""}</form></td></tr>""" for n, l in AIM.engine.lanes_snapshot().items())
    done = [r for r in _runs(username) if _job(r)["status"] not in _LIVE]
    clear = "".join(f"""<label class="list-row status-label"><input type="checkbox" name="rids" value="{r["id"]}" data-st="{UI.escape(_job(r)["status"])}"> {_status_badge(_job(r)["status"])} {UI.escape((r.get("label") or r["task"])[:70])} <span class="ca-mono">{r["created"][:16].replace("T", " ")}</span></label>""" for r in done)
    tick = lambda sts, label: f"""<button type="button" class="chip" onclick="this.closest('form').querySelectorAll('input[name=rids]').forEach(c=>{{if({repr(sts)}.includes(c.dataset.st))c.checked=true}})">{label}</button>"""   # repr: a list in single quotes, inside the double-quoted attribute
    return f"""<div class="ca-run"><div class="row"><span class="form-title grow">Queue</span><button class="ui-btn" {_post("queue_view")}>&#x21BB; Refresh</button></div>{f'<div class="status-label">{UI.escape(note)}</div>' if note else ""}
               <table class="data-table"><tr><th>Status</th><th>Run</th><th>Lane</th><th>Place</th><th></th><th>Queued</th><th></th></tr>{rows or '<tr><td colspan="7" class="list-empty">Nothing waiting or running.</td></tr>'}</table>{nodes}
               <details class="sec"><summary>Lanes - how many jobs of one lane run at once</summary><div class="status-label">A lane is one toolset or chat connection (agent:&lt;connection&gt;); different lanes always run side by side. More than 1 only helps when that server answers requests in parallel (a Magpie node runs one request at a time - extra requests wait inside it).</div><table class="data-table">{lanes or '<tr><td class="list-empty">No lanes used since the server started.</td></tr>'}</table></details>
               <details class="sec"><summary>Clear finished runs ({len(done)})</summary><form {_post("queue_clear")} hx-include="this" hx-confirm="Delete the ticked runs and their job records? Their proposed changes stay in the review lists." class="node-form"><div class="row">{tick(["error", "stopped", "interrupted", "missing"], "tick failed / stopped")}{tick(["done"], "tick done")}<button type="button" class="chip" onclick="this.closest('form').querySelectorAll('input[name=rids]').forEach(c=>c.checked=false)">none</button></div><div class="scroll-pane" style="max-height:40vh">{clear or '<div class="list-empty">No finished runs.</div>'}</div><div class="form-actions"><button class="ui-btn" style="color:#ff5f5f">Delete ticked</button></div></form></details></div>"""

# --- Transcript (live while a run works, at a chosen level of detail) ---

_DETAIL = [("steps", "Steps only"), ("results", "Steps, results expandable"), ("full", "Everything (system prompt, thinking, all expanded)")]

def _poller(rid: str, seen: int, staged: int = 0) -> str: return f"""<div {_post("status", rid=rid, seen=seen, staged=staged)} hx-trigger="load delay:3s"></div>"""

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
    """Status, Stop (after the current model turn) and Kill (at once). A record that says queued / running with no task in this server is left over from a restart or a crash: Stop clears it."""
    live = next((n.get("live") or {} for n in job["flow"]["nodes"] if n.get("live")), {})
    place = next((q["place"] for q in AIM.engine.queue_snapshot() if q["id"] == rec["job_id"]), None) if job["status"] == "queued" else None
    doing = ("not running in this server (left over from a restart or a crash) - Stop clears it" if not AIM.engine.is_live(rec["job_id"]) else f"""waiting in lane {UI.escape(job.get("lane", ""))}{f" - number {place} in line" if place else ""}""" if job["status"] == "queued" else "stopping after the current model turn - Kill ends it now" if job["status"] == "stopping" else f"""working - step {live.get("steps", 0)}""")
    kill = f"""<button class="ui-btn" style="color:#ff5f5f" {_post("kill", rid=rec["id"])} hx-confirm="Kill this run now? The model request is dropped and nothing waits for it to finish. Its transcript so far and its proposed changes are kept." title="Force stop: cancels at once, frees its place in the lane, tells the model server to drop the request">&#x2716; Kill</button>""" if AIM.engine.is_live(rec["job_id"]) else ""
    return f"""<div class="info-bar">{_status_badge(job["status"])} <span>{doing}</span>{_view_controls(rec["id"], v)}<button class="ui-btn" {_post("stop", rid=rec["id"])} title="Stops after the current model turn">&#x25FC; Stop</button>{kill}</div>{_poller(rec["id"], seen, len(_mine(rec, job)))}"""

def _transcript_box(messages: list, v: dict) -> str: return f"""<div class="status-label">Transcript ({len(messages)} messages)</div><div class="log-box ca-log{" ca-follow" if v["follow"] else ""}"><div id="ca-tx">{_rows(messages, v["detail"]) or "(nothing yet)"}</div></div>"""   # ca-follow: column-reverse scroll container - the browser keeps the view at the bottom as rows are appended, unless it has been scrolled up

def _result_html(rec: dict, job: dict, v: dict) -> str:
    data, status, msgs = job.get("data", {}), job["status"], _messages(job)
    if status in _LIVE: return f"""<div id="ca-live-bar">{_live_bar(rec, job, v, len(msgs))}</div><div id="ca-diff"></div>{_staged_html(rec, job)}{_transcript_box(msgs, v)}"""
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
    space += f"""<span class="status-label">added to Muninn #{UI.escape(rec["settings"]["muninn_thread"])}</span>""" if rec.get("muninn_entry") else ""
    summary = f"""<div class="glass ca-summary">{UI.escape(str(data["summary"]))}</div>""" if data.get("summary") else ""
    return f"""<div class="info-bar">{_status_badge(status)} <span>{UI.escape(str(data.get("agent_status", "")))}</span><span>{data.get("steps") or next((n["live"].get("steps") for n in job["flow"]["nodes"] if n.get("live")), "?")} step(s)</span>{space}<button class="ui-btn" {_post("again", rid=rec["id"])} title="Queue this task again with the same settings, as a new run">&#x21BB; Run again</button><button class="ui-btn" {_post("edit", rid=rec["id"])} title="Open this run's task and settings in the form, to change them before queueing a new run">&#x270E; Edit &amp; run</button><a class="ui-btn" href="/tool/ai_manager/job/{rec["job_id"]}" download="{rec["id"]}.json" title="The run's job record: settings, transcript and results, as JSON">&#x2B07; JSON</a>{_view_controls(rec["id"], v)}</div>
               {_continue_form(rec, job, msgs)}{err_html}{summary}
               {review}<div id="ca-diff"></div>{_staged_html(rec, job)}
               {_transcript_box(msgs, v)}"""

def _continue_form(rec: dict, job: dict, msgs: list) -> str:
    """Continue: a new run that picks up this one's conversation, proposals and space where it ended (stopped, failed, out of steps) - instead of starting over."""
    if len(msgs) < 3 or job["status"] in _LIVE: return ""
    return f"""<form class="row" {_post("continue", rid=rec["id"])} hx-include="this"><button class="ui-btn" title="Queue a new run that continues this conversation: same task, settings and proposals (it builds on this run's pending changes)">&#x23E9; Continue</button><label class="status-label">with up to <input type="number" name="steps" value="{rec["settings"].get("max_steps", 40)}" min="1" class="module-select" style="width:4.5rem"> more steps</label></form>"""

def _staged_html(rec: dict, job: dict) -> str:
    """The run's Files folder review list - while it works too: accept or reject a change as soon as it is proposed (the run then reads the folder as accepted; a rejected file goes back to the version on disk for it)."""
    root = _root(rec, job)
    if not root: return ""
    mine = len(_mine(rec, job))
    accept_all = f"""<button class="ui-btn" {_post("accept_run", rid=rec["id"])} hx-confirm="Write every pending change of this run into the folder? Files that changed since a change was proposed are left for you to look at.">Accept all {mine} of this run's changes</button>""" if mine else ""
    return f"""<div id="ca-staged"><div class="row"><span class="status-label grow">Proposed changes in <b class="ca-mono">{UI.escape(root)}</b> - {mine} pending from this run ({UI.escape(_author(rec))}). The folder's review store is shared: other pending changes show here too, with their author.</span>{accept_all}</div>
               <div id="ca-shadow">{BI.shadow_review_html(AL.shadow_for(root), "agent_shadow", {"rid": rec["id"]}, list_id="ca-shadow")}</div></div>"""

def _run_html(rec: dict, v: dict) -> str:
    job = _job(rec)
    live = f"""<div class="status-label">Model output as it is written</div><pre class="log-box ca-pre ca-live" data-stream-job="{rec["job_id"]}" data-stream-node="agent"></pre>""" if job["status"] in _LIVE else ""
    origin = f"""<div class="status-label">Queued by {UI.escape(rec["origin"])}</div>""" if rec.get("origin") else ""
    return f"""<div class="ca-run">{origin}<div class="ca-task">{UI.escape(rec["task"])}</div>{live}<div id="ca-result">{_result_html(rec, job, v)}</div></div>"""

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
    """Queues one run of task with settings s from this page and opens it. again_of names the run this one repeats. Errors raise (the caller shows them)."""
    username = request.state.user.username
    if s.get("files_root") and request.state.user.role != "admin": raise RuntimeError("only an admin can give the agent a Files folder")
    rec = submit_run(username, task, s, again_of=again_of)
    await ENV["set_state"](request, rec["id"], scope="user", namespace="agent", key="open_rid")
    imr.oob(_run_html(rec, await _view(request)), "ca-main")
    return imr.oob(_list_html(username, rec["id"]), "ca-list")

def _owned_project(request, pid): p = _pload(pid); return p if p and p["username"] == request.state.user.username else None

async def _show_project(request, imr, p: dict):
    await ENV["set_state"](request, f"""project:{p["id"]}""", scope="user", namespace="agent", key="open_rid")
    imr.oob(_project_html(p), "ca-main")
    return imr.oob(_list_html(request.state.user.username, p["id"]), "ca-list")

async def _h_project_new(request, payload, imr): return imr.oob(_project_form_html(await _defaults(request)), "ca-main")

async def _h_project_start(request, payload, imr):
    goal, title = str(payload.get("goal", "")).strip(), str(payload.get("title", "")).strip()
    if not goal or not title: return imr.oob('<span class="err-box">A title and a goal are needed.</span>', "ca-form-msg")
    try:
        s = _FIELDS.parse(payload, name_prefix="ca_")
        if s.get("files_root") and request.state.user.role != "admin": raise RuntimeError("only an admin can give the agent a Files folder")
    except Exception as e: return imr.oob(f"""<span class="err-box">Not started - {UI.escape(str(e))}</span>""", "ca-form-msg")
    await ENV["set_state"](request, s, scope="user", namespace="agent", key="defaults")
    p = {"id": f"prj_{uuid.uuid4().hex[:8]}", "username": request.state.user.username, "title": title, "goal": goal, "settings": s, "status": "running", "max_runs": max(1, int(payload.get("max_runs") or 12)), "gap_min": float(payload.get("gap_min") or 0), "runs": [], "stalls": 0, "message": "Starting.", "created": datetime.utcnow().isoformat()}
    _psave(p)
    _start_loop(p["id"])
    return await _show_project(request, imr, p)

async def _h_project_open(request, payload, imr):
    p = _owned_project(request, payload.get("pid", ""))
    return await _show_project(request, imr, p) if p else imr

async def _h_project_live(request, payload, imr):
    p = _owned_project(request, payload.get("pid", ""))
    if not p: return imr
    imr.oob(_list_html(request.state.user.username, p["id"]), "ca-list")
    return imr.oob(_project_live_html(p), "ca-proj-live")

async def _h_project_ctl(request, payload, imr):
    """pause: the current run finishes, then the project waits; resume (optionally with a new budget); stop: the current run is stopped too."""
    p, action = _owned_project(request, payload.get("pid", "")), payload.get("op", "")
    if not p: return imr
    if action == "pause" and p["status"] == "running": p = _pset(p["id"], status="paused", message="Paused - the current run finishes first.")
    elif action == "resume" and p["status"] == "paused":
        p = _pset(p["id"], status="running", stalls=0, max_runs=max(1, int(payload.get("max_runs") or p["max_runs"])), message="Resumed.")
        _start_loop(p["id"])
    elif action == "stop" and p["status"] in ("running", "paused"):
        p = _pset(p["id"], status="stopped", message="Stopped.")
        for r in (_load(rid) for rid in p["runs"]):
            if r and _job(r)["status"] in _LIVE: AIM.engine.stop(r["job_id"])
    return await _show_project(request, imr, p)

async def _h_project_accept(request, payload, imr):
    """Accepts every pending change this project's runs proposed (admin only). A change whose file was edited since it was proposed is left pending, for its own review."""
    p = _owned_project(request, payload.get("pid", ""))
    if not p or not p["settings"].get("files_root"): return imr
    if request.state.user.role != "admin": return imr.oob('<div class="err-box">Only an admin can accept changes into the folder.</div>', "ca-diff")
    sh = AL.shadow_for(p["settings"]["files_root"])
    mine = [e["path"] for e in sh.pending() if e.get("author") == f"""agent:{p["id"]}"""]
    done = [x for x in mine if sh.accept(x)]
    _accepted(p["settings"]["files_root"], done)
    left = [x for x in mine if x not in done]
    p = _pset(p["id"], message=f"""Accepted {len(done)} change(s){"; left for review (the file changed since): " + ", ".join(left) if left else ""}.""")
    return await _show_project(request, imr, p)

async def _h_project_reverse(request, payload, imr):
    """Takes the newest step (run) back: in the Files review store, the versions that run staged are removed - the earlier steps' versions of those files are the proposals again, or the change is rejected when that run made all of it; plan.md returns to how the run found it; log.md gets a line. Runs from before step reversal existed carry no version tags: their changes are left for the review list."""
    p = _owned_project(request, payload.get("pid", ""))
    if not p: return imr
    if p["status"] == "running": return imr.oob('<div class="err-box">Pause the project first.</div>', "ca-diff")
    rid = next((r for r in reversed(p["runs"]) if r not in p.get("reversed", [])), None)
    rec = _load(rid)
    if not rec: return imr.oob('<div class="err-box">No step left to reverse.</div>', "ca-diff")
    if _job(rec)["status"] in _LIVE: return imr.oob('<div class="err-box">That run is still working - Stop or Kill it first.</div>', "ca-diff")
    sh = AL.shadow_for(p["settings"]["files_root"]) if p["settings"].get("files_root") else None
    res = {e["path"]: sh.undo_tag(e["path"], rec["job_id"]) for e in _project_pending(p)} if sh else {}
    res = {k: v for k, v in res.items() if v}
    before = p.get("plan_before", {}).get(rid)
    if before is not None: _space_dir(p).mkdir(parents=True, exist_ok=True); (_space_dir(p) / "plan.md").write_text(before)
    later = [k for k, v in res.items() if v == "later"]
    line = f"""- {datetime.utcnow():%Y-%m-%d %H:%M} reversed by {request.state.user.username}: {rec.get("label") or rec["id"]} - {len(res) - len(later)} file(s) taken back{", plan.md restored" if before is not None else ""}\n"""
    with open(_space_dir(p) / "log.md", "a") as f: f.write(line)
    msg = f"""Reversed {rec.get("label") or rec["id"]}: {", ".join(f"{k} {v}" for k, v in res.items() if v != "later") or "no tagged versions (a run from before step reversal - its changes are in the review list)"}{"; plan.md restored" if before is not None else "; plan.md not recorded for that run - edit it below"}{"; kept (a later step built on it): " + ", ".join(later) if later else ""}."""
    p = _pset(p["id"], reversed=[*p.get("reversed", []), rid], stalls=0, status="paused" if p["status"] in ("done", "stopped") else p["status"], message=msg + " Resume to do it again.")   # paused: Resume picks the reversed step up
    return await _show_project(request, imr, p)

async def _h_project_plan(request, payload, imr):
    p = _owned_project(request, payload.get("pid", ""))
    if not p: return imr
    if p["status"] == "running": return imr.oob('<div class="err-box">Pause the project first - a run may be reading plan.md.</div>', "ca-diff")
    _space_dir(p).mkdir(parents=True, exist_ok=True)
    (_space_dir(p) / "plan.md").write_text(str(payload.get("plan", "")).replace("\r\n", "\n"))
    return await _show_project(request, imr, _pset(p["id"], status="paused" if p["status"] in ("done", "stopped") and _open_steps(_space_file(p, "plan.md")) else p["status"], message="plan.md saved."))

async def _h_project_delete(request, payload, imr):
    """First call: what deleting would remove, with nothing ticked. With go=1: the project record goes; its runs (and job records), its space and its pending proposals go only when ticked. A running project is stopped and its live runs killed first. Runs that stay are listed with the other runs."""
    p = _owned_project(request, payload.get("pid", ""))
    if not p: return imr.oob('<div class="err-box">That project is not yours or no longer exists - reload the list.</div>', "ca-main")
    try: return await _project_delete(request, payload, imr, p)
    except Exception as e:   # the intent dispatcher only logs a failure; show it where the person is looking
        traceback.print_exc()
        return imr.oob(f"""<div class="err-box">Delete failed - {UI.escape(type(e).__name__)}: {UI.escape(str(e))}</div>""", "ca-diff")

async def _project_delete(request, payload, imr, p: dict):
    pend = _project_pending(p)
    if payload.get("go") != "1":
        await _show_project(request, imr, p)   # from the list as well as from the project page: the form goes under the project
        return imr.oob(f"""<form class="node-form glass" style="padding:.6rem" {_post("project_delete", pid=p["id"], go="1")} hx-include="this" hx-confirm="Delete the project {UI.escape(p["title"])}?"><span class="form-title">Delete project {UI.escape(p["title"])}{" - it is running: it is stopped and its working run killed first" if p["status"] == "running" else ""}</span>
                               <label class="status-label"><input type="checkbox" name="runs" value="1"> also its {len(p["runs"])} run(s) and their job records (otherwise they move to the run list)</label>
                               <label class="status-label"><input type="checkbox" name="space" value="1"> also its space (plan.md, notes.md, log.md in project-{p["id"]})</label>
                               <label class="status-label"><input type="checkbox" name="proposals" value="1"> also reject its {len(pend)} pending proposed change(s)</label>
                               <div class="form-actions"><button class="ui-btn" style="color:#ff5f5f">Delete</button><button type="button" class="ui-btn" {_post("project_open", pid=p["id"])}>Cancel</button></div></form>""", "ca-diff")
    t = _PROJECT_TASKS.pop(p["id"], None)
    if t: t.cancel()
    recs = [r for r in (_load(rid) for rid in p["runs"]) if r]
    for r in recs:
        if _job(r)["status"] in _LIVE: await AIM.engine.kill(r["job_id"])
        if payload.get("runs") == "1": await AIM.engine.delete_job(r["job_id"]); _rp(r["id"]).unlink()
        else: r["origin"] = f"""deleted project {p["title"]}"""; _save(r)
    if payload.get("proposals") == "1" and pend:
        sh = AL.shadow_for(p["settings"]["files_root"])
        for e in pend: sh.reject(e["path"])
    if payload.get("space") == "1" and _space_dir(p).is_dir(): shutil.rmtree(_space_dir(p))
    _pp(p["id"]).unlink()
    imr.oob(_form_html(await _defaults(request)), "ca-main")
    return imr.oob(_list_html(request.state.user.username), "ca-list")

async def _h_again(request, payload, imr):
    """Runs a finished run's task again with the same settings, as a new run - e.g. after the model server died mid-run. The earlier run and anything it proposed stay as they are."""
    rec = _owned(request, payload.get("rid", ""))
    if not rec or _job(rec)["status"] in _LIVE: return imr
    try: return await _submit(request, imr, rec["task"], rec["settings"], again_of=rec["id"])
    except RuntimeError as e: return imr.oob(f"""<div class="err-box">Not queued - {UI.escape(str(e))}</div>""", "ca-diff")

async def _h_edit(request, payload, imr):
    """A run's task and settings loaded into the form, to change before queueing - the new run records which run it came from."""
    rec = _owned(request, payload.get("rid", ""))
    if not rec: return imr
    return imr.oob(_form_html({**await _defaults(request), **rec["settings"]}, rec["task"], rec["id"]), "ca-main")

async def _h_delete(request, payload, imr):
    """Deletes a run record and its job record (the owner's own data); a run still working is killed first. Proposed changes stay in the folder's review store - they belong to that folder's review, not to the run."""
    rec, username = _owned(request, payload.get("rid", "")), request.state.user.username
    if not rec: return imr
    await AIM.engine.delete_job(rec["job_id"]); _rp(rec["id"]).unlink()
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
    if job["status"] not in _LIVE:
        _report(rec["id"])
        imr.oob(_result_html(_load(rec["id"]), job, v), "ca-result"); return imr.oob(_list_html(request.state.user.username, rec["id"]), "ca-list")
    msgs, seen = _messages(job), int(payload.get("seen") or 0)
    if len(msgs) > seen: imr.oob(_rows(msgs, v["detail"], seen) if seen else _rows(msgs, v["detail"]), "ca-tx", swap="beforeend" if seen else "innerHTML")
    if _root(rec, job) and len(_mine(rec, job)) != int(payload.get("staged") or 0): imr.oob(_staged_html(rec, job), "ca-staged", swap="outerHTML")   # a new proposal (or one accepted elsewhere): the list is redrawn only then, so an open diff stays open
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

async def _h_kill(request, payload, imr):
    rec = _owned(request, payload.get("rid", ""))
    if not rec: return imr
    note = await AIM.engine.kill(rec["job_id"])
    imr.oob(_result_html(rec, _job(rec), await _view(request)), "ca-result")
    imr.oob(f"""<div class="status-label">{UI.escape(note)}</div>""", "ca-diff")
    return imr.oob(_list_html(request.state.user.username, rec["id"]), "ca-list")

async def _h_continue(request, payload, imr):
    """Queues a run that continues this one: its conversation (after the system prompt), its Files author (so it reads and builds on this run's pending changes), its space, and a new step budget."""
    rec = _owned(request, payload.get("rid", ""))
    if not rec: return imr
    job = _job(rec)
    if job["status"] in _LIVE: return imr
    s = {**rec["settings"], "author": rec["settings"].get("author") or rec["job_id"], "max_steps": max(1, int(payload.get("steps") or rec["settings"].get("max_steps", 40)))}
    try:
        if s.get("files_root") and request.state.user.role != "admin": raise RuntimeError("only an admin can give the agent a Files folder")
        new = submit_run(rec["username"], rec["task"], s, again_of=rec["id"], origin=rec.get("origin", ""), label=f"""continue: {(rec.get("label") or rec["task"])[:60]}""", resume=_messages(job)[1:])
    except RuntimeError as e: return imr.oob(f"""<div class="err-box">Not queued - {UI.escape(str(e))}</div>""", "ca-diff")
    await ENV["set_state"](request, new["id"], scope="user", namespace="agent", key="open_rid")
    imr.oob(_run_html(new, await _view(request)), "ca-main")
    return imr.oob(_list_html(request.state.user.username, new["id"]), "ca-list")

async def _h_queue_view(request, payload, imr, note: str = ""):
    admin = request.state.user.role == "admin"
    return imr.oob(_queue_html(request.state.user.username, admin, note, await _nodes_html(admin)), "ca-main")

async def _h_node_ctl(request, payload, imr):
    """Admin: drop a node's requests (scope requests) or kill its model process too (scope server), through the node's control channel."""
    conn = AIM.connections.get_conn(payload.get("conn", ""))
    if request.state.user.role != "admin" or not conn: return imr
    r = await AIM.connections.call_capability(conn, "node_kill", {"scope": "server" if payload.get("scope") == "server" else "requests"}, timeout_s=10)
    return await _h_queue_view(request, payload, imr, f"""{conn.get("display_name", conn["_id"])}: {r.get("error") or f"{r.get('cancelled', 0)} request(s) dropped" + (", model process killed" if r.get("server_killed") else "")}""")

async def _h_queue_job(request, payload, imr):
    """Stop / Kill one job from the queue - the owner's own, or any job for an admin."""
    job = AIM.engine.load_job(payload.get("job", ""))
    if not job or (job.get("username") != request.state.user.username and request.state.user.role != "admin"): return imr
    if payload.get("op") == "kill": note = await AIM.engine.kill(job["id"])
    else: AIM.engine.stop(job["id"]); note = "stop requested"
    imr.oob(_list_html(request.state.user.username), "ca-list")
    return await _h_queue_view(request, payload, imr, f"""{job["id"]}: {note}""")

async def _h_lane_cap(request, payload, imr):
    if request.state.user.role != "admin": return imr
    await AIM.engine.set_lane_cap(str(payload.get("lane", "")), max(1, min(16, int(payload.get("cap") or 1))))
    return await _h_queue_view(request, payload, imr, f"""lane {payload.get("lane", "")}: {payload.get("cap")} at once""")

async def _h_queue_clear(request, payload, imr):
    """Deletes the ticked finished runs (the owner's own) and their job records. Nothing is ticked until the person ticks it."""
    rids = payload.get("rids") or []
    recs = [r for r in (_owned(request, rid) for rid in ([rids] if isinstance(rids, str) else rids)) if r and _job(r)["status"] not in _LIVE]
    for r in recs: await AIM.engine.delete_job(r["job_id"]); _rp(r["id"]).unlink()
    imr.oob(_list_html(request.state.user.username), "ca-list")
    return await _h_queue_view(request, payload, imr, f"deleted {len(recs)} run(s)")

async def _h_accept_run(request, payload, imr):
    """Accepts every pending change of one run (admin only) - also while it works or after it failed. A change whose file was edited since it was proposed is left pending."""
    rec = _owned(request, payload.get("rid", ""))
    if not rec: return imr
    if request.state.user.role != "admin": return imr.oob('<div class="err-box">Only an admin can accept changes into the folder.</div>', "ca-diff")
    job = _job(rec)
    sh, mine = AL.shadow_for(_root(rec, job)), [e["path"] for e in _mine(rec, job)]
    done = [x for x in mine if sh.accept(x)]
    _accepted(_root(rec, job), done)
    imr.oob(f"""<div class="status-label">Accepted {len(done)} change(s){"; left for review (the file changed since): " + UI.escape(", ".join(x for x in mine if x not in done)) if len(done) < len(mine) else ""}.</div>""", "ca-diff")
    return imr.oob(_staged_html(rec, job), "ca-staged", swap="outerHTML")

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
    root = _root(rec, _job(rec)) if rec else ""
    if not root: return imr
    shadow, action, path = AL.shadow_for(root), payload.get("action", ""), payload.get("path", "")
    if action == "diff": return imr.oob(f"""<pre class="ca-pre">{UI.escape(shadow.diff(path)) or "(no differences)"}</pre>""", payload.get("diff_target", ""))
    if action == "accept" and request.state.user.role != "admin": return imr.oob('<div class="err-box">Only an admin can accept changes into the folder.</div>', "ca-diff")
    if action == "accept" and shadow.accept(path, force=payload.get("force") == "1"): _accepted(root, [path])
    elif action == "accept": imr.oob(f"""<div class="err-box">{UI.escape(path)} changed after the change was proposed - see its diff; Accept anyway replaces the newer file.</div>""", "ca-diff")
    elif action == "reject": shadow.reject(path)
    return imr.oob(BI.shadow_review_html(shadow, "agent_shadow", {"rid": rec["id"]}, list_id="ca-shadow"), "ca-shadow")

async def _h_session_close(request, payload, imr):
    r = await _call(payload.get("conn_id", ""), "agent_session_close", {"session": payload.get("session", ""), "message": "closed from Agent after an interrupted run"})
    if r.get("error"): return imr.oob(f"""<div class="err-box">{UI.escape(r["error"])}</div>""", "ca-main")
    return await _h_sessions(request, {"ca_tools_conn_id": payload.get("conn_id", "")}, imr)

# --- Init / route ---

def init_tool(env: dict, prefix: str):
    global ENV, UI, IM, BI, AIM, AL, MN, _FIELDS
    ENV = env
    UI = env["templates"].env.globals.get("UI")
    BI = env["tools"]["built_ins"]
    AIM = env["tools"]["ai_manager"]
    MN = env["tools"].get("muninn")   # optional: without it the Muninn thread field is refused at queue time
    # from tools.ai_manager import agent_loop as AL   # the node's own module: its review store for Files folders # do not import frm outside module folder
    AL = AIM.agent_loop
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    _FIELDS = _build_fields()
    _PDIR.mkdir(parents=True, exist_ok=True)
    for f in _PDIR.glob("*.json"):   # a restart is a point to re-confirm: running projects wait for Resume
        p = json.loads(f.read_text())
        if p["status"] == "running": _pset(p["id"], status="paused", message="The server restarted - press Resume to continue.")
    if MN:   # runs that finished (or were cut off) while the server was down: report them now
        for f in DATA_DIR.glob("*.json"): _report(f.stem)
    env["register_service"]("agent", {"submit_run": submit_run, "run_status": run_status, "last_settings": last_settings, "settings_summary": settings_summary, "command": {"help": _CMD_HELP, "parse": _cmd_parse, "run": _cmd_run}})
    IM = env["InterfaceManager"](nesting_level=2, db_path="agent_im.db")
    IM.scripts.update({f"agent_{k}": [fn] for k, fn in {"home": _h_home, "queue": _h_queue, "open": _h_open, "status": _h_status, "stop": _h_stop, "diff": _h_diff, "discard": _h_discard, "sessions": _h_sessions, "session_close": _h_session_close, "form_refresh": _h_form_refresh, "shadow": _h_shadow, "view": _h_view, "delete": _h_delete, "again": _h_again, "edit": _h_edit, "project_new": _h_project_new, "project_start": _h_project_start, "project_open": _h_project_open, "project_live": _h_project_live, "project_ctl": _h_project_ctl, "project_accept": _h_project_accept,
                      "kill": _h_kill, "continue": _h_continue, "accept_run": _h_accept_run, "queue_view": _h_queue_view, "queue_job": _h_queue_job, "lane_cap": _h_lane_cap, "queue_clear": _h_queue_clear, "node_ctl": _h_node_ctl,
                      "project_reverse": _h_project_reverse, "project_plan": _h_project_plan, "project_delete": _h_project_delete}.items()})
    print("[agent] ready")

@router.get("")
@router.get("/")
async def root(request: Request): return ENV["templates"].TemplateResponse(name="base.html", request=request, context={"request": request, "user": request.state.user, "nesting_level": 2, "shell_id": IM.branch_id, "toolbars": {"right": _runs_bar(request.state.user.username)}, "content": _page_html(await _defaults(request)), "extra_css": CSS})

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