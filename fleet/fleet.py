# /modules/ai_tools/fleet/fleet.py
"""
Fleet - the portal side of Fleet (an AI Tools tool, mounted at /module/ai_tools/fleet). Each Magpie node keeps a journal of what it does and holds (fleet.py on the node);
this tool reads every node's journal into its own database and shows it:
  Overview (each node now), Timeline (what each node held and did over time), Stats (speeds, waits, prefix reuse, loads, activity by hour), Jobs (where each job's requests went and how they went - the routing results),
  Caches (what each node holds, what its policies would change, the actions, and backtests of loading strategies), Tests (run tests on a node, compare results), Experts (expert placement for MoE models from routing traces),
  Events (the raw journal), Settings (policies per node, schedules, collector).
Policies only propose. A node in shadow mode (the default) shows proposals; a node in auto mode gets them sent; Apply sends one now. The node has the last word (its own fleet.json).
Helper files beside this one: store.py (database and collector), analysis.py (statistics, policies, backtests), viz.py (SVG charts), experts.py (expert placement).
Settings: data/ai_tools/fleet/fleet.json (admin, Settings tab).
"""
import json, time, uuid, asyncio, importlib.util
from datetime import datetime, timedelta
from pathlib import Path
from fastapi import APIRouter, Request

def _helper(name: str):
    """A helper file beside this one, loaded by path the way ai_tools loads this module."""
    spec = importlib.util.spec_from_file_location(f"modules.ai_tools.fleet_{name}", Path(__file__).parent / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod

store, analysis, viz, experts = _helper("store"), _helper("analysis"), _helper("viz"), _helper("experts")

TOOL_META = {"label": "Fleet", "group": "system", "icon": "&#x1F6F0;", "description": "Model nodes: what they hold and do, caches, tests, routing results", "singleton": True}
router = APIRouter(redirect_slashes=False)
ENV: dict = {}
_P = "/module/ai_tools/fleet"
UI = BI = IM = TM = AIM = None
CONFIG_PATH = store.DATA_DIR / "fleet.json"
DEFAULTS = {"collect_s": 10, "state_s": 60, "retention_days": 30, "plan_s": 300, "cooldown_s": 1800, "default": {"mode": "shadow", "policies": {"observe": {}}}, "nodes": {}, "schedule": [], "backtest": {"default_load_s": 30, "idle_s": 300}}
_TABS = {k: {"id": k, "order": i, "label": lbl, "icon": ic} for i, (k, lbl, ic) in enumerate((("overview", "Overview", "&#x1F6F0;"), ("timeline", "Timeline", "&#x1F4C8;"), ("stats", "Stats", "&#x1F4CA;"), ("jobs", "Jobs", "&#x1F9ED;"), ("caches", "Caches", "&#x1F5C4;"), ("tests", "Tests", "&#x1F9EA;"), ("experts", "Experts", "&#x1F9E9;"), ("events", "Events", "&#x1F4DC;"), ("settings", "Settings", "&#x2699;")))}
WINDOWS = ((1, "1 h"), (6, "6 h"), (24, "24 h"), (72, "3 d"), (168, "7 d"), (720, "30 d"), (0, "all"))   # 0: from the oldest event kept here
TEST_DEFAULTS = {"perf": {"lengths": "128,2048,8192", "repeat": 2, "max_tokens": 128}, "prefix": {"length": 4096, "max_tokens": 64}, "fidelity": {"max_tokens": 256}, "load": {"prefetch": False, "max_tokens": 16}}
_LOOP = {"task": None}
_LAST_PLAN: dict = {}       # node -> (time, proposal keys): shadow proposals are recorded when they change, auto ones sent at most once per cooldown
_PLAN_ERR: dict = {}        # node -> why its policies could not run
_EXPERT_RESULT: dict = {}   # user id -> the latest placement result shown (for saving its pin lists)

def _esc(s) -> str: return viz.esc(s)
def _vals(type_: str, **extra) -> str: return json.dumps({"type": type_, "branch": "fleet", "lvl": 2, **extra}).replace("'", "&#39;")
def _post(type_: str, **extra) -> str: return f"""hx-post="/im/in" hx-target="body" hx-swap="none" hx-vals='{_vals("fleet_" + type_, **extra)}'"""
def _admin(request) -> bool: return getattr(request.state.user, "role", "user") == "admin"
def _ago(t) -> str:
    if not t: return "never"
    d = time.time() - t
    return f"{d:.0f} s ago" if d < 120 else f"{d / 60:.0f} min ago" if d < 7200 else f"{d / 3600:.0f} h ago" if d < 172800 else f"{d / 86400:.0f} d ago"
def _ts(t, fmt: str = "%m-%d %H:%M:%S") -> str: return datetime.fromtimestamp(t).strftime(fmt) if t else "-"
def _n(v, fmt: str = ".1f", none: str = "-") -> str: return none if v is None else format(v, fmt)
def _pct(v) -> str: return "-" if v is None else f"{v:.0%}"
def _opt(value, label, selected: bool = False) -> str: return f"""<option value="{_esc(value)}"{" selected" if selected else ""}>{_esc(label)}</option>"""
def _num(name: str, label: str, value, width: str = "5rem") -> str: return f"""<label class="status-label">{_esc(label)} <input class="module-select" type="number" step="any" name="{name}" value="{value}" style="width:{width}"></label>"""
def _check(name: str, label: str, title: str = "", checked: bool = True) -> str: return f"""<label class="status-label" title="{_esc(title)}"><input type="checkbox" name="{name}" value="1"{" checked" if checked else ""}> {_esc(label)}</label>"""
def _confirm(text: str) -> str: return f""" hx-confirm="{_esc(text)}" """ if text else ""
def _red(on: bool) -> str: return ' style="color:#ff5f5f"' if on else ""   # a button that deletes
def _on(on: bool) -> str: return ' style="background:var(--accent);color:var(--bg)"' if on else ""   # a chip that is selected
def _err(text: str) -> str: return f"""<div class="err-box">{_esc(text)}</div>""" if text else ""
def _doc(name: str, doc: str, params: str = "") -> str: return f"""<div><b>{_esc(name)}</b> <span class="mono">{_esc(params)}</span><div class="status-label">{_esc(doc)}</div></div>"""
def _params_txt(p: dict) -> str: return f""", ctx {p.get("ctx")} kv {p.get("kv_cache_type")}""" if p else ""

# --- Settings ---

def cfg() -> dict: return {**DEFAULTS, **(json.loads(CONFIG_PATH.read_text()) if CONFIG_PATH.exists() else {})}
def node_cfg(node: str) -> dict: c = cfg(); return {**c["default"], **c["nodes"].get(node, {})}

def _check_cfg(c: dict) -> list:
    """Problems in a settings dict (nothing saved while there are any)."""
    out = [f"mode of {where} must be shadow or auto" for where, nc in [("default", c.get("default", {})), *c.get("nodes", {}).items()] if nc.get("mode", "shadow") not in ("shadow", "auto")]
    out += [f"unknown policy {p!r} in {where} - known: {sorted(analysis.POLICIES)}" for where, nc in [("default", c.get("default", {})), *c.get("nodes", {}).items()] for p in nc.get("policies", {}) if p not in analysis.POLICIES]
    out += [f"schedule {i}: at must be HH:MM" for i, e in enumerate(c.get("schedule", [])) if not str(e.get("at", "")).count(":") == 1]
    return out

# --- Init and background loop ---

def init_tool(env: dict, prefix: str):
    global UI, BI, IM, TM, AIM, _P
    ENV.update(env); _P = prefix.rstrip("/")
    UI, BI, AIM = env["templates"].env.globals.get("UI"), env["tools"]["built_ins"], env["tools"]["ai_manager"]
    IM = env["InterfaceManager"](nesting_level=2, db_path="ai_tools/fleet_im.db")
    store.init(AIM)
    TM = BI.TabManager(namespace="fleet", tab_bar_id="fleet-tab-bar", content_id="fleet-panel", render_content_fn=_render_panel, intent_prefix="fleet", IM=IM, scope="user", nesting_level=2, allow_new=False, closable=False, empty={"tabs": dict(_TABS), "active": "overview"})
    IM.scripts.update({f"fleet_{k}": [fn] for k, fn in {"view": _h_view, "refresh": _h_refresh, "collect": _h_collect, "apply": _h_apply, "apply_all": _h_apply_all, "backtest": _h_backtest, "test_run": _h_test_run, "test_cancel": _h_test_cancel,
                                                        "test_defaults": _h_test_defaults, "settings_save": _h_settings_save, "synth": _h_synth, "place": _h_place, "save_pins": _h_save_pins}.items()})
    env["register_service"]("fleet", {"advise": advise, "summary": node_summary})
    env["register_startup"](_start)
    print("[fleet] ready")

async def _start(): _ensure_loop()

def _ensure_loop():
    if _LOOP["task"] is None or _LOOP["task"].done(): _LOOP["task"] = asyncio.create_task(store.loop(cfg, _on_cycle))

async def _on_cycle():
    await _plan_cycle()
    await _schedule_cycle()

def _ctx(node: str, st: dict, now: float = 0) -> dict:
    """What a policy sees of one node: its state, its last 7 days of requests (tests left out), loads and action outcomes."""
    now = now or time.time()
    evs = store.events(node, now - 7 * 86400, 0, ("req.end", "engine.req", "llama.spawn", "llama.ready", "llama.crash", "engine.start", "engine.up", "engine.failed", "directive.done"))
    return {"node": node, "state": st, "reqs": analysis.requests(evs), "loads": analysis.loads(evs), "events": [e for e in evs if e["kind"] == "directive.done"], "now": now}

def _key(p: dict) -> str: return json.dumps([p["action"], p["args"]], sort_keys=True)

async def _plan_cycle():
    """Every plan_s per node: the node's policies propose; shadow nodes get new proposals recorded, auto nodes get them sent (one identical action per cooldown_s)."""
    c, now = cfg(), time.time()
    for conn in store.nodes():
        n = conn["_id"]
        if now - _LAST_PLAN.get(n, (0, set()))[0] < float(c["plan_s"]): continue
        nc, st = node_cfg(n), await asyncio.to_thread(store.state, n)
        try: props = analysis.plan(await asyncio.to_thread(_ctx, n, st), nc["policies"])
        except KeyError as e:   # a policy name the settings file has but this code does not (edited by hand): shown on the Overview until fixed; the other nodes go on
            _PLAN_ERR[n] = f"unknown policy {e} in data/ai_tools/fleet/fleet.json"
            continue
        _PLAN_ERR.pop(n, None)
        prev = _LAST_PLAN.get(n, (0, set()))[1]
        _LAST_PLAN[n] = (now, {_key(p) for p in props})
        for p in props:
            if nc["mode"] == "auto":
                sent = await asyncio.to_thread(store.q, "SELECT n FROM actions WHERE node = ? AND action = ? AND args = ? AND mode = 'sent' AND t > ?", (n, p["action"], json.dumps(p["args"], ensure_ascii=False), now - float(c["cooldown_s"])))
                if not sent: await _send(n, p["action"], p["args"], "plan", p["policy"], p["why"], p.get("value"))
            elif _key(p) not in prev: await asyncio.to_thread(store.record_action, n, "plan", p["policy"], p["action"], p["args"], p["why"], p.get("value"), "shadow", "proposed")

def _day_ok(days: str, wd: int) -> bool:
    """days: * | mon-fri | sat,sun | mon,wed-fri"""
    names = ["mon", "tue", "wed", "thu", "fri", "sat", "sun"]
    if str(days).strip() in ("*", ""): return True
    for part in str(days).lower().replace(" ", "").split(","):
        a, _, b = part.partition("-")
        if (names.index(a) <= wd <= names.index(b)) if b else names.index(a) == wd: return True
    return False

async def _schedule_cycle():
    """Scheduled actions (settings: schedule) run once on their day, within 30 minutes after their time - sent whatever the node's mode (a schedule is an explicit instruction; the node may still refuse).
    Whether one already ran today is read from the actions table, so a restart does not run it twice."""
    now = datetime.now()
    for i, e in enumerate(cfg()["schedule"]):
        if not e.get("enabled", True) or not _day_ok(e.get("days", "*"), now.weekday()): continue
        hh, mm = (int(x) for x in str(e["at"]).split(":"))
        due = now.replace(hour=hh, minute=mm, second=0, microsecond=0)
        if not timedelta(0) <= now - due < timedelta(minutes=30): continue
        for n in ([c["_id"] for c in store.nodes()] if e.get("node") in (None, "", "*") else [e["node"]]):
            if await asyncio.to_thread(store.q, "SELECT n FROM actions WHERE node = ? AND source = ? AND t >= ?", (n, f"schedule {i}", due.timestamp())): continue
            await _send(n, e["action"], e.get("args") or {}, f"schedule {i}", "", e.get("why") or f"""scheduled {e.get("days", "*")} {e["at"]}""")

async def _send(node: str, action: str, args: dict, source: str, policy: str = "", why: str = "", value=None, dry: bool = False) -> dict:
    """Sends one action to a node and records it; the node's outcome arrives later as its directive.done event (shown in the action history)."""
    conn = next((c for c in store.nodes() if c["_id"] == node), None)
    if conn is None: return {"error": f"no Fleet node {node!r}"}
    did = uuid.uuid4().hex[:10]
    n = await asyncio.to_thread(store.record_action, node, source, policy, action, args, why, value, "sent", "sending", {}, did)
    r = await AIM.connections.call_capability(conn, "fleet_apply", {"id": did, "action": action, "args": args, "dry_run": dry, "source": f"fleet {source}", "wait": False}, timeout_s=30)
    status = "error" if r.get("error") else "refused" if r.get("refused") else "dry run" if r.get("dry_run") else "started"
    await asyncio.to_thread(store.update_action, n, status, r, did)
    return {**r, "status": status}

# --- Services (for other tools: routing advice and summaries) ---

def node_summary(node: str, hours: float = 24) -> dict:
    """One node's last `hours`: request statistics, loads, what it holds now."""
    now = time.time()
    evs = store.events(node, now - hours * 3600, 0, ("req.end", "engine.req", "llama.spawn", "llama.ready", "llama.crash"))
    return {"stats": analysis.summary(analysis.requests(evs)), "loads": analysis.loads(evs), "state": store.state(node)}

def advise(model: str) -> list:
    """Nodes ranked for a request for `model`, best first: has it loaded and idle, loaded but busy, has its file (faster median decode first). [{"node", "score", "reason"}]. Advice only - nothing routes by it yet."""
    out = []
    for c in store.nodes():
        st = store.state(c["_id"])
        if not st: continue
        res, has = st.get("resident") or {}, any(m["name"] == model for m in st.get("models") or [])
        if not has: continue
        loaded = res.get("alive") and res.get("model") == model
        dec = analysis.summary([r for r in analysis.requests(store.events(c["_id"], time.time() - 7 * 86400, 0, ("req.end",))) if r.get("model") == model])["decode_p50"]
        score = (3 if loaded and not st.get("busy") else 2 if loaded else 1) * 1000 + (dec or 0)
        out.append({"node": c["_id"], "score": round(score, 1), "reason": f"""{"loaded" if loaded else "on disk"}{", busy" if st.get("busy") else ""}; decode {_n(dec)} tok/s (median, 7 d)"""})
    return sorted(out, key=lambda r: -r["score"])

# --- Per-person view state ---

async def _st(request, s: dict = None) -> dict:
    if s is not None:
        await ENV["set_state"](request, s, scope="user", namespace="fleet")
        return s
    s = await ENV["get_state"](request, scope="user", namespace="fleet") or {}
    for k, v in (("node", ""), ("window_h", 24), ("job", ""), ("ev_kind", ""), ("ev_q", ""), ("test", ""), ("trace", ""), ("tests_in_stats", False), ("msg", "")): s.setdefault(k, v)
    return s

def _node_of(s: dict, all_ok: bool = True) -> str:
    """The chosen node; "" = all nodes where that is allowed, else the first node Fleet has heard from."""
    ns = [c["_id"] for c in store.nodes()]
    return s["node"] if s["node"] in ns else ("" if all_ok else next((n for n in ns if store.state(n)), ns[0] if ns else ""))

def _since(s: dict) -> float:
    """Start of the chosen window; window 0 (all) starts at the oldest event kept here."""
    return time.time() - s["window_h"] * 3600 if s["window_h"] else (store.q("SELECT MIN(t) AS t FROM events")[0]["t"] or time.time() - 3600)

def _node_pick(s: dict, all_ok: bool = True) -> str:
    cur = _node_of(s, all_ok)
    opts = ([("", "all nodes")] if all_ok else []) + [(c["_id"], store.node_label(c)) for c in store.nodes()]
    return f"""<select class="module-select" name="node" {_post("view")} hx-trigger="change" hx-include="this">{"".join(_opt(v, l, v == cur) for v, l in opts)}</select>"""

def _window_pick(s: dict) -> str: return "".join(f"""<button class="chip"{_on(s["window_h"] == h)} {_post("view", window_h=h)}>{lbl}</button>""" for h, lbl in WINDOWS)

def _msg(s: dict) -> str: return f"""<div class="info-bar">{_esc(s["msg"])}</div>""" if s.get("msg") else ""

# --- Overview ---

def _node_card(c: dict, cur: dict, cs: float) -> str:
    n, now = c["_id"], time.time()
    st, evs = store.state(n), store.events(n, now - 86400, 0, ("req.end", "engine.req", "llama.spawn", "llama.ready", "llama.crash"))
    reqs, lds = analysis.requests(evs), analysis.loads(evs)
    sm, res, mem = analysis.summary(reqs), st.get("resident") or {}, st.get("memory") or {}
    ok = cur.get("ok_t") and (not cur.get("err_t") or cur["ok_t"] >= cur["err_t"]) and now - cur["ok_t"] < 3 * cs + 30
    dot = f"""<span style="color:{"hsl(140,45%,48%)" if ok else "hsl(0,65%,55%)"}">&#x25CF;</span>"""
    fl = st.get("fleet") or {}
    sets = fl.get("settings") or {}
    crashes = sum(1 for e in evs if e["kind"] == "llama.crash")
    loaded = f"""<b>{_esc(res.get("model"))}</b> ctx {res.get("ctx")} kv {_esc(res.get("kv"))}""" if res.get("alive") else "no model loaded"
    crash_txt = f""", <span style="color:hsl(0,65%,55%)">{crashes} crashes</span>""" if crashes else ""
    herr = f""", hook errors {fl["hook_errors"]["n"]}""" if (fl.get("hook_errors") or {}).get("n") else ""
    return f"""<div class="glass" style="padding:.6rem;display:flex;flex-direction:column;gap:.25rem;min-width:18rem;flex:1">
                   <div class="row"><b>{dot} {_esc(store.node_label(c))}</b><span class="status-label grow">{_esc(cur.get("node_name") or "")} - journal read {_ago(cur.get("ok_t"))}{" - " + _esc(cur.get("err") or "") if not ok and cur.get("err") else ""}{" - events were lost before Fleet read them (gap)" if cur.get("gap") else ""}</span></div>
                   <div>{loaded}{" - <span style='color:var(--accent)'>busy</span>" if st.get("busy") else " - idle" if st else ""} <span class="status-label">state {_ago(st.get("_collected"))}</span></div>
                   <div class="status-label">RAM available {_n(mem.get("ram_avail_gb"))} of {_n(mem.get("ram_total_gb"))} GB, GPU used {_n(mem.get("gpu_used_gb"))} GB, load {_n(mem.get("load1"), ".2f")}; engines up: {_esc(", ".join(mem.get("engines") or []) or "none")}</div>
                   <div>Last 24 h: <b>{sm["n"]}</b> requests ({sm["failed"]} failed, {sm["cancelled"]} cancelled), decode {_n(sm["decode_p50"])} tok/s, prefill {_n(sm["prefill_p50"], ".0f")} tok/s, prefix reused {_pct(sm["reuse"])}, waited {_n(sm["wait_sum"], ".0f")} s; {len(lds)} loads ({_n(sum(l["load_s"] for l in lds), ".0f")} s){crash_txt}</div>
                   <div class="row">{viz.spark(analysis.per_hour(reqs, now - 86400, now), title="requests per hour, last 24 h")}<span class="status-label">KV contexts {len(st.get("kv_ring") or [])}, pins {len(st.get("pins") or [])}; node accepts actions: {_esc((sets.get("apply") or {}).get("mode", "?"))}, tests: {_esc((sets.get("tests") or {}).get("allow", "?"))}{herr}; mode here: {_esc(node_cfg(n)["mode"])}</span></div>
               </div>"""

async def _p_overview(request, s) -> str:
    curs, c = await asyncio.to_thread(store.cursors), cfg()
    cards = "".join([await asyncio.to_thread(_node_card, n, curs.get(n["_id"], {}), float(c["collect_s"])) for n in store.nodes()]) or '<div class="status-label">No Fleet nodes. A Magpie connection (AI Tools > Settings > Connections, type Magpie) becomes one once its node runs Magpie with fleet.py.</div>'
    stt = store.STATUS
    return f"""<div class="scroll-pane" style="padding:.8rem">{_msg(s)}
                   <div class="row"><span class="status-label grow">Collector: {"running" if stt["running"] else "not started"}, last pass {_ago(stt["last_cycle"])} ({_n(stt["cycle_s"], ".2f")} s){" - " + _esc(stt["error"]) if stt["error"] else ""}</span>{"".join(_err(f"{n}: {e}") for n, e in _PLAN_ERR.items())}<button class="ui-btn" {_post("collect")}>&#x21BB; Collect now</button></div>
                   <div style="display:flex;flex-wrap:wrap;gap:.5rem">{cards}</div>
                   <div {_post("refresh", tab="overview")} hx-trigger="every 20s"></div>
               </div>"""

# --- Timeline ---

_MARKS = {"kv.save": ("&#x2193;", "var(--text_muted, #999)"), "kv.restore": ("&#x2191;", "var(--accent, #4aa3df)"), "llama.crash": ("&#x2715;", "hsl(0,65%,55%)"), "llama.crash_mid_stream": ("&#x2715;", "hsl(0,65%,55%)"), "directive": ("&#x25C6;", "hsl(280,50%,60%)"), "test.start": ("&#x25B2;", "hsl(40,80%,55%)"), "node.start": ("&#x27F3;", "var(--text, #ddd)")}

def _lanes(node: str, label: str, t0: float, now: float) -> list:
    res = analysis.residency(store.events(node, t0 - 7 * 86400, 0, ("llama.", "engine.start", "engine.up", "engine.stop", "engine.failed", "node.start")), now)
    evs = store.events(node, t0, 0, ("req.end", "engine.req", "mem", "kv.", "directive", "test.start", "llama.crash", "node.start"))
    bands = [(r["t0"], r["t1"], viz.hue(r["name"]), str(r["name"]), f"""{r["name"]} ({r["what"]}) {_ts(r["t0"])} - {_ts(r["t1"])}, {(r["t1"] - r["t0"]) / 60:.0f} min{_params_txt(r["params"])}""") for r in res if r["t1"] >= t0]
    reqs = analysis.requests(evs, tests=True)
    bars = [(r["start"], r["end"], min(1.0, (len(str((r.get("prompt_n") or 0) + (r.get("cache_n") or 0) + (r.get("gen_n") or 0))) - 1) / 5 + 0.2), viz.OUTCOME.get(r.get("outcome"), viz.MUTED),
             f"""{r.get("rid")} {r.get("model")} {r.get("outcome")}: {r.get("seconds")} s (waited {r.get("wait_s")} s), prompt {r.get("prompt_n")} + cached {r.get("cache_n")}, out {r.get("gen_n")}, decode {_n(r.get("decode_tps"))} tok/s""") for r in reqs]
    line = [(e["t"], (e.get("ram_avail_gb") or 0) / (e.get("ram_total_gb") or 1)) for e in evs if e["kind"] == "mem"]
    marks = [(e["t"], *_MARKS[e["kind"]], f"""{e["kind"]} {_ts(e["t"])}: {", ".join(f"{k}={v}" for k, v in e.items() if k not in ("id", "t", "kind", "node", "params", "est", "experiments", "smoke"))[:300]}""") for e in evs if e["kind"] in _MARKS]
    return [{"group": label, "label": "loaded", "bands": bands}, {"group": label, "label": "requests", "bars": bars, "line": line, "line_title": "RAM available (share of total)"}, {"group": label, "label": "events", "marks": marks}]

async def _p_timeline(request, s) -> str:
    now, t0 = time.time(), await asyncio.to_thread(_since, s)
    nodes = [c for c in store.nodes() if c["_id"] == _node_of(s) or not _node_of(s)]
    lanes = [ln for c in nodes for ln in await asyncio.to_thread(_lanes, c["_id"], store.node_label(c), t0, now)]
    legend = " ".join(f"""<span>{sym} {_esc(k)}</span>""" for k, (sym, col) in _MARKS.items() if not k.endswith("mid_stream"))
    return f"""<div class="scroll-pane" style="padding:.8rem">
                   <div class="row">{_node_pick(s)}{_window_pick(s)}</div>
                   {viz.timeline(t0, now, lanes) if lanes else '<div class="status-label">No nodes.</div>'}
                   <div class="status-label">Loaded: one band per model or engine while it was loaded. Requests: one bar each (height grows with its tokens; green served, red failed, amber cancelled), the line is RAM available. Events: {legend}. Hover for details.</div>
               </div>"""

# --- Stats ---

def _stats_html(s: dict) -> str:
    now, node, t0 = time.time(), _node_of(s), _since(s)
    evs = store.events(node, t0, 0, ("req.end", "engine.req", "llama.spawn", "llama.ready", "llama.crash", "engine.start", "engine.up", "engine.failed"))
    reqs, lds = analysis.requests(evs, tests=s["tests_in_stats"]), analysis.loads(evs)
    rows = "".join(f"""<tr><td>{_esc(n)}</td><td>{_esc(m)}</td><td>{sm["n"]}</td><td>{sm["failed"]}/{sm["cancelled"]}</td><td>{_n(sm["wait_p50"])} / {_n(sm["wait_p90"])}</td><td>{_n(sm["prefill_p50"], ".0f")}</td><td>{_n(sm["decode_p10"])} / {_n(sm["decode_p50"])} / {_n(sm["decode_p90"])}</td><td>{_pct(sm["reuse"])}</td><td>{sm["tokens_in"]} / {sm["tokens_cached"]} / {sm["tokens_out"]}</td><td>{_n(sm["busy_s"], ".0f")}</td></tr>"""
                   for (n, m), rs in analysis.by(reqs, "node", "model").items() for sm in [analysis.summary(rs)])
    lrows = "".join(f"""<tr><td>{_esc(n)}</td><td>{_esc(m)}</td><td>{len(ls)}</td><td>{sum(1 for l in ls if not l["ok"])}</td><td>{_n(analysis.pct([l["load_s"] for l in ls if l["ok"]], 50))}</td><td>{_n(max((l["load_s"] for l in ls), default=None))}</td></tr>""" for (n, m), ls in analysis.by(lds, "node", "model").items())
    dec = [(f"{n} / {m}", analysis.summary(rs)["decode_p50"], f"{n} {m}: median decode tok/s over {len(rs)} requests") for (n, m), rs in analysis.by([r for r in reqs if r.get("decode_tps")], "node", "model").items()]
    hm = analysis.heat(analysis.requests(store.events(node, now - 28 * 86400, 0, ("req.end", "engine.req"))))
    quiet = analysis.quiet_windows(hm)
    days = ("Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun")
    return f"""<div class="row">{_node_pick(s)}{_window_pick(s)}<button class="chip"{_on(s["tests_in_stats"])} {_post("view", tests_in_stats=not s["tests_in_stats"])}>with Fleet's tests</button></div>
               <h4>Requests by node and model</h4>
               <div style="overflow-x:auto"><table class="data-table" style="font-size:.78rem"><thead><tr><th>Node</th><th>Model</th><th>Requests</th><th>Failed / cancelled</th><th>Wait s (median / p90)</th><th>Prefill tok/s</th><th>Decode tok/s (p10 / median / p90)</th><th>Prefix reused</th><th>Tokens read / reused / written</th><th>Busy s</th></tr></thead><tbody>{rows or '<tr><td colspan="10" class="status-label">no requests in this window</td></tr>'}</tbody></table></div>
               <div class="status-label">Wait: from arrival until the node started on the request (loading a model, other requests). Prefill speed counts requests that read at least 32 tokens, decode those that wrote at least 8. Prefix reused: tokens taken from the cache over all prompt tokens.</div>
               <h4>Median decode speed</h4>{viz.bars(dec, unit=" tok/s") or '<div class="status-label">-</div>'}
               <h4>Loads</h4><table class="data-table" style="font-size:.78rem"><thead><tr><th>Node</th><th>Model / engine</th><th>Loads</th><th>Failed</th><th>Median s</th><th>Longest s</th></tr></thead><tbody>{lrows or '<tr><td colspan="6" class="status-label">none</td></tr>'}</tbody></table>
               <h4>When the {"node is" if node else "nodes are"} busy (last 4 weeks, busy seconds by weekday and hour)</h4>{viz.heat_week(hm)}
               <div class="status-label">Quietest 3-hour windows: {"; ".join(f"{days[d]} " + ", ".join(f"{a:02d}-{b:02d}h" for a, b, _ in w[:2]) for d, w in enumerate(quiet))} - where long jobs (an ingest, tests) disturb the fewest requests.</div>"""

async def _p_stats(request, s) -> str: return f"""<div class="scroll-pane" style="padding:.8rem">{await asyncio.to_thread(_stats_html, s)}</div>"""

# --- Jobs (routing results) ---

def _jobs_html(s: dict) -> str:
    reqs = analysis.requests(store.events(_node_of(s), _since(s), 0, ("req.end", "engine.req")), tests=True)
    if s["job"]:
        rs = sorted((r for r in reqs if r["job"] == s["job"]), key=lambda r: r["start"])
        rows = "".join(f"""<tr><td>{_ts(r["start"])}</td><td>{_esc(r["node"])}</td><td>{_esc(r.get("model"))}</td><td>{_esc(r.get("api"))}</td><td>{_esc(r.get("rid"))}</td><td>{_n(r.get("seconds"))}</td><td>{_n(r.get("wait_s"))}</td><td>{r.get("prompt_n", "-")} / {r.get("cache_n", "-")}</td><td>{r.get("gen_n", "-")}</td><td>{_n(r.get("decode_tps"))}</td><td style="color:{viz.OUTCOME.get(r.get("outcome"), "inherit")}">{_esc(r.get("outcome"))}</td></tr>""" for r in rs)
        lanes = [{"label": n, "bars": [(r["start"], r["end"], 0.9, viz.hue(r.get("model")), f"""{r.get("model")} {r.get("seconds")} s""") for r in rs if r["node"] == n]} for n in sorted({r["node"] for r in rs})]
        tl = viz.timeline(min(r["start"] for r in rs), max(r["end"] for r in rs) + 1, lanes) if rs else ""
        return f"""<div class="row"><button class="ui-btn" {_post("view", job="")}>&#x2039; All jobs</button><b>{_esc(s["job"])}</b><span class="status-label">{len(rs)} requests in this window</span></div>{tl}
                   <table class="data-table" style="font-size:.78rem"><thead><tr><th>Start</th><th>Node</th><th>Model</th><th>API</th><th>Request id</th><th>Seconds</th><th>Waited</th><th>Prompt read / reused</th><th>Written</th><th>Decode tok/s</th><th>Outcome</th></tr></thead><tbody>{rows}</tbody></table>"""
    js = analysis.jobs(reqs)[:300]
    rows = "".join(f"""<tr style="cursor:pointer" {_post("view", job=j["job"])}><td class="mono">{_esc(j["job"])}</td><td>{_esc(", ".join(j["nodes"]))}</td><td>{_esc(", ".join(j["models"]))}</td><td>{j["n"]}</td><td>{_ts(j["start"], "%m-%d %H:%M")} - {_ts(j["end"], "%H:%M")}</td><td>{_n(j["busy_s"], ".0f")}</td><td>{_n(j["wait_s"], ".0f")}</td><td>{j["tokens_in"]} / {j["tokens_cached"]}</td><td>{j["tokens_out"]}</td><td>{_n(j["decode_p50"])}</td><td>{_esc(", ".join(f"{k} {v}" for k, v in j["outcomes"].items()))}</td></tr>""" for j in js)
    return f"""<div class="row">{_node_pick(s)}{_window_pick(s)}<span class="status-label">A job is everything before the first dot of a request id: the portal sends &lt;job id&gt;.&lt;random&gt;, Fleet's tests fleettest_&lt;id&gt;.&lt;n&gt;; a caller that sends none gets one per request.</span></div>
               <table class="data-table" style="font-size:.78rem"><thead><tr><th>Job</th><th>Nodes</th><th>Models</th><th>Requests</th><th>When</th><th>Busy s</th><th>Waited s</th><th>Prompt read / reused</th><th>Written</th><th>Decode tok/s</th><th>Outcomes</th></tr></thead><tbody>{rows or '<tr><td colspan="11" class="status-label">no requests in this window</td></tr>'}</tbody></table>"""

async def _p_jobs(request, s) -> str: return f"""<div class="scroll-pane" style="padding:.8rem">{await asyncio.to_thread(_jobs_html, s)}</div>"""

# --- Caches: what a node holds, what its policies propose, actions ---

def _act_btn(request, node: str, action: str, args: dict, label: str, confirm: str = "", danger: bool = False, **rec) -> str:
    if not _admin(request): return ""
    return f"""<button class="cm-qbtn"{_red(danger)} {_post("apply", node=node, action=action, args=json.dumps(args), **rec)}{_confirm(confirm)}>{label}</button>"""

def _history_html(node: str) -> str:
    acts = store.actions(node, 100)
    done = {e.get("directive"): e for e in store.events(node, min((a["t"] for a in acts), default=time.time()) - 60, 0, ("directive.done",))}
    def outcome(a):
        d = done.get(a["directive"])
        if a["status"] != "started" or not d: return _esc(a["status"]) + (f""": {_esc((a["result"].get("refused") or a["result"].get("error") or "")[:120])}""" if a["result"].get("refused") or a["result"].get("error") else "")
        return f"""{"done" if d.get("ok") else "failed"} in {d.get("seconds")} s{": " + _esc(str(d.get("error"))[:160]) if d.get("error") else ""}"""
    rows = "".join(f"""<tr><td>{_ts(a["t"])}</td><td>{_esc(a["source"])}</td><td>{_esc(a["policy"] or "")}</td><td><b>{_esc(a["action"])}</b> <span class="mono">{_esc(json.dumps(a["args"], ensure_ascii=False)[:120])}</span></td><td>{_esc(a["mode"])}</td><td>{outcome(a)}</td><td class="status-label">{_esc(a["why"] or "")}</td></tr>""" for a in acts)
    return f"""<table class="data-table" style="font-size:.75rem"><thead><tr><th>When</th><th>From</th><th>Policy</th><th>Action</th><th>Mode</th><th>Outcome</th><th>Why</th></tr></thead><tbody>{rows or '<tr><td colspan="7" class="status-label">nothing yet</td></tr>'}</tbody></table>"""

def _caches_html(request, s: dict) -> str:
    node = _node_of(s, all_ok=False)
    if not node: return '<div class="status-label">No Fleet nodes.</div>'
    st, nc = store.state(node), node_cfg(node)
    try: props, perr = analysis.plan(_ctx(node, st), nc["policies"]), ""
    except KeyError as e: props, perr = [], f"unknown policy {e} in the settings"
    res = st.get("resident") or {}
    prow = "".join(f"""<tr><td>{_esc(p["policy"])}</td><td><b>{_esc(p["action"])}</b> <span class="mono">{_esc(json.dumps(p["args"], ensure_ascii=False))}</span></td><td>{_esc(p["why"])}</td><td>{_n(p.get("value"), ".0f")}</td><td>{_act_btn(request, node, p["action"], p["args"], "Apply", source="plan", policy=p["policy"], why=p["why"])}</td></tr>""" for p in props)
    ring = "".join(f"""<tr><td>{_esc(e["session"])}</td><td>{_esc(e["model"])}</td><td>{_esc(e["kv"])}</td><td>{e["size_mb"]} MB</td><td>{_ago(time.time() - e["age_s"])}</td><td>{"&#x1F4CC; pinned" if e["pinned"] else ""}</td><td>{_act_btn(request, node, "unpin" if e["pinned"] else "pin", {"session": e["session"]}, "Unpin" if e["pinned"] else "Pin")}{_act_btn(request, node, "kv_drop", {"file": e["file"]}, "Drop", "Delete this saved context? Its session re-reads its prompt next time.", True)}</td></tr>""" for e in st.get("kv_ring") or [])
    models = "".join(f"""<tr><td>{_esc(m["name"])}</td><td>{m["size_gb"]} GB</td><td>{_esc(m["source"])}</td><td>{"loaded" if res.get("alive") and res.get("model") == m["name"] else ""}</td><td>{_act_btn(request, node, "warm", {"model": m["name"]}, "Load", "Load this model now? A request for another model that arrives meanwhile waits.")}{_act_btn(request, node, "prefetch", {"model": m["name"]}, "Prefetch")}</td></tr>""" for m in st.get("models") or [])
    engines = "".join(f"""<tr><td>{_esc(e["name"])}</td><td>{_esc(e["status"])}</td><td>{"exclusive" if e["exclusive"] else ""}</td><td>{_act_btn(request, node, "engine_stop" if e["alive"] else "engine_start", {"name": e["name"]}, "Stop" if e["alive"] else "Start", "" if e["alive"] else "Start this engine? An exclusive engine stops the loaded model first.")}</td></tr>""" for e in st.get("engines") or [])
    pf = st.get("prefetch") or {}
    pol_docs = "".join(_doc(k, analysis.POLICIES[k]["doc"], json.dumps({**analysis.POLICIES[k]["params"], **(v or {})})) for k, v in nc["policies"].items() if k in analysis.POLICIES)
    manual = f"""<form class="row" {_post("apply", node=node, source="manual")} hx-include="this"><select class="module-select" name="action">{"".join(_opt(a, a) for a in (st.get("fleet") or {}).get("actions", []))}</select><input class="module-select grow" name="args" placeholder='args as JSON, e.g. {{"model": "name", "options": {{"num_ctx": 32768}}}}'><label class="status-label"><input type="checkbox" name="dry_run" value="1"> dry run</label><button class="ui-btn">Send</button></form>""" if _admin(request) else ""
    replay = f"""the last {s["window_h"]:g} h""" if s["window_h"] else "everything kept"
    apply_all = f"""<button class="ui-btn" {_post("apply_all", node=node)}{_confirm(f"Send all {len(props)} proposed actions to this node?")}>Apply all</button>""" if props and _admin(request) else ""
    loaded = f"""<b>{_esc(res.get("model"))}</b> ctx {res.get("ctx")} kv {_esc(res.get("kv"))} session {_esc(res.get("session"))}, estimate {_esc(json.dumps(res.get("est") or {}))}""" if res.get("alive") else "no model loaded"
    return f"""<div class="row">{_node_pick(s, all_ok=False)}<span class="status-label">state {_ago(st.get("_collected"))}{" - busy" if st.get("busy") else ""}</span></div>{_msg(s)}
               <h4>Proposed by this node's policies - mode <b>{_esc(nc["mode"])}</b> ({"proposals are only shown and recorded" if nc["mode"] == "shadow" else "proposals are sent every " + str(cfg()["plan_s"]) + " s"})</h4>{_err(perr)}
               <table class="data-table" style="font-size:.78rem"><thead><tr><th>Policy</th><th>Action</th><th>Why</th><th>Est. s saved</th><th></th></tr></thead><tbody>{prow or '<tr><td colspan="5" class="status-label">nothing to change</td></tr>'}</tbody></table>
               {apply_all}
               <details><summary class="status-label" style="cursor:pointer">Policies of this node (Settings tab to change)</summary>{pol_docs}</details>
               <h4>Loaded</h4><div>{loaded} {_act_btn(request, node, "unload", {}, "Unload", "Unload the model now? Its context is saved first.") if res.get("alive") else ""}</div>
               <h4>Saved KV contexts (the ring: newest kept, pinned ones too)</h4><table class="data-table" style="font-size:.78rem"><thead><tr><th>Session</th><th>Model</th><th>KV</th><th>Size</th><th>Saved</th><th></th><th></th></tr></thead><tbody>{ring or '<tr><td colspan="7" class="status-label">none</td></tr>'}</tbody></table>
               <h4>Models on this node</h4><table class="data-table" style="font-size:.78rem"><thead><tr><th>Model</th><th>Size</th><th>Source</th><th></th><th></th></tr></thead><tbody>{models}</tbody></table>
               <div class="status-label">Prefetch reads a model's file into the page cache, so its next load reads RAM; last: {_esc(json.dumps(pf.get("last")) if pf.get("last") else "none")}{" - running: " + _esc(pf["running"]) if pf.get("running") else ""}</div>
               <h4>Engines</h4><table class="data-table" style="font-size:.78rem"><tbody>{engines or '<tr><td class="status-label">none</td></tr>'}</tbody></table>
               {"<h4>Send an action</h4>" + manual if manual else ""}
               <h4>Which loading strategy would have served this node best?</h4>
               <div class="row"><button class="ui-btn" {_post("backtest", node=node)}>Replay {replay}</button><span class="status-label">Replays this node's recorded requests through its one model slot with each predictor, using its measured load times.</span></div><div id="fleet-backtest"></div>
               <h4>Actions</h4>{_history_html(node)}"""

async def _p_caches(request, s) -> str: return f"""<div class="scroll-pane" style="padding:.8rem">{await asyncio.to_thread(_caches_html, request, s)}</div>"""

def _backtest_html(node: str, since: float) -> str:
    c = cfg()["backtest"]
    evs = store.events(node, since, 0, ("req.end", "llama.spawn", "llama.ready", "llama.crash"))
    reqs, lds = analysis.requests(evs), analysis.loads(evs)
    load_s = {m: analysis.pct([l["load_s"] for l in ls if l["ok"]], 50) for (m,), ls in analysis.by(lds, "model").items()}
    load_s = {m: v for m, v in load_s.items() if v is not None}
    rows = analysis.backtest(reqs, load_s, float(c["default_load_s"]), float(c["idle_s"]))
    llama = [r for r in reqs if r.get("api") in analysis.LLAMA_APIS]
    actual = f"""Actual: {len(llama)} requests waited {_n(sum(r.get("wait_s") or 0 for r in llama), ".0f")} s in all, {sum(1 for l in lds if l["what"] == "model")} model loads."""
    body = "".join(f"""<tr><td><b>{_esc(r["predictor"])}</b><div class="status-label">{_esc(r["doc"])}</div></td><td>{r["wait_s"]}</td><td>{r["wait_per_request_s"]}</td><td>{r["loads"]}</td><td>{r["loads_ahead_hit"]}</td><td>{r["loads_ahead_wasted"]}</td></tr>""" for r in rows)
    return f"""<table class="data-table" style="font-size:.78rem"><thead><tr><th>Predictor</th><th>Waited s</th><th>Per request</th><th>Loads</th><th>Loaded ahead, used</th><th>Loaded ahead, wasted</th></tr></thead><tbody>{body}</tbody></table>
               <div class="status-label">{actual} Load times: {_esc(json.dumps({m: round(v) for m, v in load_s.items()}))} measured, {c["default_load_s"]} s assumed for others; a model is loaded ahead after {c["idle_s"]} s without requests (Settings: backtest). "stay" is what the node does without Fleet; compare it with Actual to see how close the replay is.</div>"""

# --- Tests ---

def _tests_html(request, s: dict) -> str:
    node = _node_of(s, all_ok=False)
    if not node: return '<div class="status-label">No Fleet nodes.</div>'
    st = store.state(node)
    kinds = (st.get("tests") or {}).get("kinds") or sorted(TEST_DEFAULTS)
    model = (st.get("resident") or {}).get("model") or next((m["name"] for m in st.get("models") or []), "")
    form = f"""<form {_post("test_run", node=node)} hx-include="this" style="display:flex;flex-direction:column;gap:.3rem;max-width:44rem">
                   <div class="row"><select class="module-select" name="kind" id="fleet-test-kind" {_post("test_defaults", node=node)} hx-trigger="change" hx-include="this">{"".join(_opt(k, k) for k in kinds)}</select><span class="status-label grow">perf: speed at several prompt lengths; prefix: what a cached prefix saves; fidelity: byte-identical answers; load: what a load costs</span></div>
                   <div id="fleet-test-params"><textarea class="module-select mono" name="params" rows="4" style="width:100%">{_esc(json.dumps({"model": model, **TEST_DEFAULTS[kinds[0]]}, indent=1))}</textarea></div>
                   <div class="row"><button class="ui-btn">&#x25B6; Run on this node</button><span class="status-label">Add "num_ctx" with the value this model's users send, or the model reloads with another context first. "engine": "name" tests an engine (OpenAI API) instead.</span></div></form>""" if _admin(request) else '<div class="status-label">Only an admin runs tests.</div>'
    evs = store.events(node, time.time() - 30 * 86400, 0, ("test.",))
    ends = {e["test"]: e for e in evs if e["kind"] == "test.end"}
    starts = {e["test"]: e for e in evs if e["kind"] == "test.start"}
    running = [t for t in starts if t not in ends]
    cancel = f"""<button class="cm-qbtn" {_post("test_cancel", node=node)}>Cancel</button>""" if _admin(request) else ""
    run_html = "".join(f"""<div class="info-bar">running: {_esc(t)} ({_esc(starts[t].get("test_kind"))}, since {_ts(starts[t]["t"])}) {cancel}</div>""" for t in running)
    rows = "".join(f"""<tr style="cursor:pointer" {_post("view", test=t)}><td>{_ts(e["t"])}</td><td class="mono">{_esc(t)}</td><td>{_esc(e.get("test_kind"))}</td><td>{_esc((starts.get(t, {}).get("params") or {}).get("model") or (starts.get(t, {}).get("params") or {}).get("engine") or "")}</td><td>{"ok" if e.get("ok") else _esc(e.get("error"))}</td><td>{e.get("seconds")}</td><td class="mono">{_esc(json.dumps(e.get("summary") or {})[:140])}</td></tr>""" for t, e in sorted(ends.items(), key=lambda kv: -kv[1]["t"]))
    detail = _test_detail(s["test"], evs, starts) if s["test"] else ""
    return f"""<div class="row">{_node_pick(s, all_ok=False)}</div>{_msg(s)}{form if _admin(request) else ""}{run_html}{detail}
               <h4>Tests on this node (30 days)</h4><table class="data-table" style="font-size:.78rem"><thead><tr><th>Ended</th><th>Test</th><th>Kind</th><th>Model</th><th>Result</th><th>Seconds</th><th>Summary</th></tr></thead><tbody>{rows or '<tr><td colspan="7" class="status-label">none yet</td></tr>'}</tbody></table>
               <div {_post("refresh", tab="tests")} hx-trigger="every 15s"></div>"""

def _test_detail(tid: str, evs: list, starts: dict) -> str:
    ms = [e for e in evs if e.get("test") == tid and e["kind"] in ("test.perf", "test.prefix", "test.fidelity", "test.load")]
    if not ms: return f"""<div class="info-bar">{_esc(tid)}: no measurements</div>"""
    kind, p = ms[0]["kind"].split(".")[1], (starts.get(tid, {}).get("params") or {})
    srv = lambda e, k: (e.get("server") or {}).get(k)
    head = f"""<h4>{_esc(tid)} - {kind} on {_esc(p.get("model") or p.get("engine"))} <button class="cm-qbtn" {_post("view", test="")}>&#x2715;</button></h4>"""
    rows = "".join(f"""<tr><td>{e.get("target_tokens", e.get("i", e.get("step", "")))}</td><td>{e.get("run", "")}</td><td>{_n(e.get("ttft_s"), ".2f")}</td><td>{_n(e.get("prompt_tokens"), ".0f")}</td><td>{_n(srv(e, "cache_n"), ".0f")}</td><td>{_n(srv(e, "prompt_tok_s"), ".0f")}</td><td>{_n(e.get("decode_tok_s"))}</td><td>{_n(e.get("total_s"), ".2f")}</td><td class="mono">{_esc((e.get("sha256") or "")[:12])}</td><td>{_esc(e.get("error") or "")}</td></tr>""" for e in ms)
    table = f"""<table class="data-table" style="font-size:.78rem"><thead><tr><th>Length / prompt / step</th><th>Run</th><th>TTFT s</th><th>Prompt tok</th><th>Reused</th><th>Prefill tok/s</th><th>Decode tok/s</th><th>Total s</th><th>Hash</th><th>Error</th></tr></thead><tbody>{rows}</tbody></table>"""
    chart = ""
    if kind == "perf":
        runs = sorted({e.get("run") for e in ms})
        ttft = viz.xy([{"name": f"TTFT run {r}", "points": [(e["target_tokens"], e.get("ttft_s"), f"""{e["target_tokens"]} tok: {e.get("ttft_s")} s""") for e in ms if e.get("run") == r]} for r in runs], "prompt tokens (target)", "time to first token, s", logx=True)
        dec = viz.xy([{"name": f"decode run {r}", "points": [(e["target_tokens"], e.get("decode_tok_s"), f"""{e.get("decode_tok_s")} tok/s""") for e in ms if e.get("run") == r]} for r in runs], "prompt tokens (target)", "decode tok/s", logx=True)
        chart = f"""<div class="row" style="align-items:flex-start;gap:1rem">{ttft}{dec}</div><div class="status-label">Run 1 is the first time this node saw the prompt; later runs show what its cache saves.</div>"""
    if kind == "fidelity":
        prev = sorted((t for t, st in starts.items() if t != tid and st.get("test_kind") == "fidelity" and (st.get("params") or {}).get("model") == p.get("model") and st["t"] < starts.get(tid, {"t": 0})["t"]), key=lambda t: starts[t]["t"])
        if prev:
            old = {e["i"]: e["sha256"] for e in evs if e.get("test") == prev[-1] and e["kind"] == "test.fidelity"}
            same = sum(1 for e in ms if old.get(e["i"]) == e["sha256"])
            chart = f"""<div class="info-bar">Against the previous fidelity test of this model ({_esc(prev[-1])}): {same} of {len(ms)} answers byte-identical{"" if same == len(ms) else " - prompts that differ: " + ", ".join(str(e["i"]) for e in ms if old.get(e["i"]) != e["sha256"])}</div>"""
    return head + chart + table

async def _p_tests(request, s) -> str: return f"""<div class="scroll-pane" style="padding:.8rem">{await asyncio.to_thread(_tests_html, request, s)}</div>"""

# --- Experts (MoE placement from routing traces) ---

def _experts_html(request, s: dict) -> str:
    ts = experts.traces()
    cur = s["trace"] if s["trace"] in ts else (ts[0] if ts else "")
    synth = f"""<form class="row" {_post("synth")} hx-include="this"><span class="status-label">Make a synthetic trace:</span>{"".join(_num(k, k, v) for k, v in (("layers", 43), ("experts", 256), ("k", 6), ("tokens", 4000), ("topics", 6), ("skew", 1.0)))}<button class="ui-btn">Make</button></form>""" if _admin(request) else ""
    picker = f"""<select class="module-select" name="trace" {_post("view")} hx-trigger="change" hx-include="this">{"".join(_opt(t, t, t == cur) for t in ts)}</select>""" if ts else '<span class="status-label">No traces yet.</span>'
    synth = synth if _admin(request) else ""
    if not cur: return f"""<div class="row">{picker}</div>{synth}<div class="status-label">Put routing traces in data/ai_tools/fleet/traces/ (format in experts.py) or make a synthetic one.</div>"""
    tr = experts.load(cur)
    form = f"""<form {_post("place", trace=cur)} hx-include="this" style="display:flex;flex-direction:column;gap:.3rem">
                   <div class="row">{"".join(_num(k, lbl, v, "5.5rem") for k, lbl, v in (("nodes", "nodes", 3), ("gb", "GB for experts per node", 22), ("holdout", "scored on last share", 0.3), ("replicate", "replicate_top: per layer", 8), ("balance", "coactivate: balance", 1.0)))}</div>
                   <div class="row">{"".join(_num(k, k, v) for k, v in experts.COST.items())}</div>
                   <div class="row">{"".join(_check(f"pol_{k}", k, v["doc"]) for k, v in experts.PLACEMENTS.items())}<button class="ui-btn">Plan placement</button></div></form>"""
    return f"""<div class="row">{picker}<span class="status-label">{tr["layers"]} layers x {tr["experts"]} experts, {len(tr["tokens"])} tokens{" - synthetic: shows how the planner behaves, not evidence about a model" if cur.startswith("synthetic") else ""}</span></div>{synth}
               <h4>Heat: how often each expert was chosen (each layer scaled to its own hottest expert)</h4>{viz.matrix(experts.heat(tr), cur)}
               <h4>Placement across nodes</h4>{form}<div id="fleet-place"></div>
               <div class="status-label">Each placement learns on the first part of the trace and is scored on the rest. Time per token is an estimate from the cost model above (the slowest node per layer, a round trip when another node is involved, a hop when the layer moves to another node) - replace the numbers with measured ones from the Tests tab and the node's disk and link.</div>"""

async def _p_experts(request, s) -> str: return f"""<div class="scroll-pane" style="padding:.8rem">{_msg(s)}{await asyncio.to_thread(_experts_html, request, s)}</div>"""

def _pins_btn(policy: str, admin: bool) -> str: return f"""<button class="cm-qbtn" {_post("save_pins", policy=policy)}>Save pin lists</button>""" if admin else ""

def _place_html(r: dict, admin: bool) -> str:
    rows = "".join(f"""<tr><td><b>{_esc(x["policy"])}</b><div class="status-label">{_esc(x["doc"])}</div></td><td>{_pct(x["hit_rate"])}</td><td>{_n(x["misses_per_token"])}</td><td>{_n(x["remote_per_layer"], ".2f")}</td><td>{_n(x["hops_per_token"], ".1f")}</td><td>{_n(x["ms_per_token"], ".0f")} ({_n(x["ms_dense"], ".0f")} / {_n(x["ms_experts"], ".0f")} / {_n(x["ms_network"], ".1f")})</td><td>{_n(x["tok_s"], ".2f")}</td><td>{" / ".join(f"{w:.0%}" for w in x["work_share"])}</td><td>{_pins_btn(x["policy"], admin)}</td></tr>""" for x in r["rows"])
    floor = r["all_from_disk"]
    return f"""<table class="data-table" style="font-size:.78rem"><thead><tr><th>Placement</th><th>In RAM</th><th>Disk reads / token</th><th>Other nodes / layer</th><th>Hops / token</th><th>ms / token (dense / experts / network)</th><th>tok/s</th><th>Expert work per node</th><th></th></tr></thead><tbody>{rows}
               <tr class="dim-row"><td>everything from disk</td><td>0%</td><td>{_n(floor["misses_per_token"])}</td><td>-</td><td>-</td><td>{_n(floor["ms_per_token"], ".0f")}</td><td>{_n(floor["tok_s"], ".2f")}</td><td></td><td></td></tr></tbody></table>
               <div class="status-label">{r["slots"]} expert slots per node, learned on {r["tokens_learn"]} tokens, scored on {r["tokens_test"]}. Estimates from the cost model, one stream, no batching.</div>"""

# --- Events ---

def _events_html(s: dict) -> str:
    node, kind, text = _node_of(s), s["ev_kind"], s["ev_q"]
    w, a = ["1 = 1"], []
    if node: w.append("node = ?"); a.append(node)
    if kind: w.append("kind LIKE ?"); a.append(kind + "%")
    if text: w.append("data LIKE ?"); a.append(f"%{text}%")
    rows = store.q(f"""SELECT node, data FROM events WHERE {" AND ".join(w)} ORDER BY t DESC LIMIT 300""", tuple(a))
    def line(e): return ", ".join(f"{k}={json.dumps(v, ensure_ascii=False) if isinstance(v, (dict, list)) else v}" for k, v in e.items() if k not in ("id", "t", "kind"))[:220]
    body = "".join(f"""<tr><td>{_ts(e["t"])}</td><td>{_esc(r["node"])}</td><td><b>{_esc(e["kind"])}</b></td><td><details><summary class="mono" style="cursor:pointer">{_esc(line(e))}</summary><pre class="mono" style="white-space:pre-wrap;font-size:.7rem">{_esc(json.dumps(e, indent=1, ensure_ascii=False))}</pre></details></td></tr>""" for r in rows for e in [json.loads(r["data"])])
    return f"""<form class="row" {_post("view")} hx-include="this" hx-trigger="change, submit">{_node_pick(s)}<input class="module-select" name="ev_kind" value="{_esc(kind)}" placeholder="kind starts with, e.g. req. llama. kv. directive test."><input class="module-select grow" name="ev_q" value="{_esc(text)}" placeholder="text anywhere in the event"></form>
               <table class="data-table" style="font-size:.75rem"><thead><tr><th>Time</th><th>Node</th><th>Kind</th><th>Fields</th></tr></thead><tbody>{body or '<tr><td colspan="4" class="status-label">no events</td></tr>'}</tbody></table>
               <div class="status-label">Newest 300 matching events. The node's own journal is the original: GET /fleet/events?after=0 on the node.</div>"""

async def _p_events(request, s) -> str: return f"""<div class="scroll-pane" style="padding:.8rem">{await asyncio.to_thread(_events_html, s)}</div>"""

# --- Settings ---

def _settings_html(request, s: dict, problems: list = None, text: str = "") -> str:
    c = cfg()
    pols = "".join(_doc(k, v["doc"], json.dumps(v["params"])) for k, v in analysis.POLICIES.items())
    nodes = "".join(f"""<tr><td>{_esc(store.node_label(n))}</td><td class="mono">{_esc(n["_id"])}</td><td>{_esc(node_cfg(n["_id"])["mode"])}</td><td class="mono">{_esc(json.dumps(node_cfg(n["_id"])["policies"]))}</td><td class="mono">{_esc(json.dumps({k: v for k, v in (((store.state(n["_id"]).get("fleet") or {}).get("settings") or {}).get("apply") or {}).items() if not k.startswith("_")}))}</td></tr>""" for n in store.nodes())
    edit = f"""<form {_post("settings_save")} hx-include="this"><textarea class="module-select mono" name="config" rows="18" style="width:100%">{_esc(text or json.dumps(c, indent=1))}</textarea>{"".join(_err(p) for p in problems or [])}<button class="ui-btn">Save settings</button></form>""" if _admin(request) else f"""<pre class="mono">{_esc(json.dumps(c, indent=1))}</pre>"""
    return f"""{_msg(s)}<h4>Nodes</h4><table class="data-table" style="font-size:.78rem"><thead><tr><th>Node</th><th>Connection id (key in "nodes")</th><th>Mode</th><th>Policies</th><th>What the node itself accepts (its fleet.json)</th></tr></thead><tbody>{nodes or '<tr><td colspan="5" class="status-label">none</td></tr>'}</tbody></table>
               <h4>Settings (data/ai_tools/fleet/fleet.json)</h4>
               <div class="status-label">collect_s: how often every node's journal is read; state_s: how often what it holds; retention_days: how long events are kept here; plan_s: how often policies run; cooldown_s: an auto node gets the same action at most once in this time.
               default / nodes.&lt;connection id&gt;: mode (shadow: proposals shown and recorded; auto: sent) and policies ({{name: parameters}}, parameters over the policy's defaults below).
               schedule: [{{"node": id or "*", "days": "mon-fri", "at": "07:45", "action": "warm", "args": {{"model": "...", "options": {{"num_ctx": 32768}}}}, "enabled": true}}] - sent at that time whatever the mode.
               backtest: default_load_s (for models never seen loading), idle_s (idle time before a predictor may load ahead).</div>{edit}
               <h4>Policies</h4>{pols}
               <h4>Predictors (backtest)</h4>{"".join(_doc(k, v["doc"]) for k, v in analysis.PREDICTORS.items())}"""

async def _p_settings(request, s) -> str: return f"""<div class="scroll-pane" style="padding:.8rem">{await asyncio.to_thread(_settings_html, request, s)}</div>"""

# --- Panel, tabs, main route ---

PANELS = {"overview": _p_overview, "timeline": _p_timeline, "stats": _p_stats, "jobs": _p_jobs, "caches": _p_caches, "tests": _p_tests, "experts": _p_experts, "events": _p_events, "settings": _p_settings}

async def _render_panel(request, state):
    s = await _st(request)
    html = await PANELS.get(state.get("active", "overview"), _p_overview)(request, s)
    if s["msg"]: s["msg"] = ""; await _st(request, s)   # a message shows once
    return state, html

async def _refresh(request, imr, msg: str = ""):
    if msg: s = await _st(request); s["msg"] = msg; await _st(request, s)
    state = await TM._load(request)
    _, html = await _render_panel(request, state)
    return imr.oob(html, "fleet-panel")

@router.get("")
@router.get("/")
async def root(request: Request):
    _ensure_loop()
    state = await TM._load(request)
    state["tabs"] = dict(_TABS)
    await TM._save(request, state)
    state, panel = await _render_panel(request, state)
    tab_bar = await TM.tab_bar_fn(state, "fleet-tab-bar", "fleet", 2, allow_new=False, closable=False)
    return ENV["templates"].TemplateResponse(name="base.html", request=request, context={"request": request, "user": request.state.user, "nesting_level": 2, "shell_id": IM.branch_id, "extra_css": "",
        "toolbars": {"top": UI.toolbar(side="top", content=tab_bar, size="2.5rem", id="fleet-top", nesting_level=2, start_open=True, locked=True)},
        "content": f"""<div id="fleet-panel" style="height:100%;overflow:hidden">{panel}</div>"""})

def right_panel() -> str: return """<div class="ait-rp"><div class="ait-rp-hd">Fleet</div><div style="font-size:.72rem;color:var(--text_muted);padding:.3rem">Model nodes: what each holds and does, its caches and the actions on them, tests, and where each job's requests went.</div></div>"""

# --- Intent handlers ---

async def _h_view(request, payload, imr):
    s = await _st(request)
    for k in ("node", "job", "ev_kind", "ev_q", "test", "trace"):
        if k in payload: s[k] = str(payload[k])
    if "window_h" in payload: s["window_h"] = float(payload["window_h"])
    if "tests_in_stats" in payload: s["tests_in_stats"] = str(payload["tests_in_stats"]).lower() == "true"
    await _st(request, s)
    return await _refresh(request, imr)

async def _h_refresh(request, payload, imr):
    """A panel's own timer: redraws it only while it is the open tab."""
    return await _refresh(request, imr) if (await TM._load(request)).get("active") == payload.get("tab") else imr

async def _h_collect(request, payload, imr):
    ns = store.nodes()
    got = await asyncio.gather(*(store.collect_events(n) for n in ns))
    await asyncio.gather(*(store.collect_state(n) for n in ns))
    return await _refresh(request, imr, f"Collected {sum(got)} new events from {len(ns)} node(s).")

def _args(payload: dict) -> dict:
    a = payload.get("args") or "{}"
    return a if isinstance(a, dict) else json.loads(a)

async def _h_apply(request, payload, imr):
    if not _admin(request): return imr
    try: args = _args(payload)
    except ValueError as e: return await _refresh(request, imr, f"Args are not valid JSON: {e}")
    r = await _send(payload["node"], payload["action"], args, payload.get("source") or "manual", payload.get("policy", ""), payload.get("why", ""), dry=str(payload.get("dry_run", "")) in ("1", "true", "on"))
    return await _refresh(request, imr, f"""{payload["action"]} on {payload["node"]}: {r["status"]}{" - " + str(r.get("refused") or r.get("error")) if r.get("refused") or r.get("error") else " (outcome under Actions once the node reports it)" if r["status"] == "started" else ""}""")

async def _h_apply_all(request, payload, imr):
    if not _admin(request): return imr
    node = payload["node"]
    props = analysis.plan(await asyncio.to_thread(_ctx, node, store.state(node)), node_cfg(node)["policies"])
    rs = [await _send(node, p["action"], p["args"], "plan", p["policy"], p["why"], p.get("value")) for p in props]
    return await _refresh(request, imr, f"""Sent {len(rs)} actions: {", ".join(r["status"] for r in rs)}""")

async def _h_backtest(request, payload, imr):
    s = await _st(request)
    return imr.oob(await asyncio.to_thread(_backtest_html, payload["node"], _since(s)), "fleet-backtest")

async def _h_test_defaults(request, payload, imr):
    st = store.state(payload["node"])
    model = (st.get("resident") or {}).get("model") or next((m["name"] for m in st.get("models") or []), "")
    return imr.oob(f"""<textarea class="module-select mono" name="params" rows="4" style="width:100%">{_esc(json.dumps({"model": model, **TEST_DEFAULTS.get(payload.get("kind"), {})}, indent=1))}</textarea>""", "fleet-test-params")

async def _h_test_run(request, payload, imr):
    if not _admin(request): return imr
    try: params = json.loads(payload.get("params") or "{}")
    except ValueError as e: return await _refresh(request, imr, f"Params are not valid JSON: {e}")
    conn = next((c for c in store.nodes() if c["_id"] == payload["node"]), None)
    r = await AIM.connections.call_capability(conn, "fleet_test", {"kind": payload["kind"], "params": params, "id": f"""{payload["kind"]}-{uuid.uuid4().hex[:6]}"""}, timeout_s=30)
    return await _refresh(request, imr, f"""Test {r.get("id")} queued on {payload["node"]} ({r.get("queued")} waiting{", running " + r["running"] if r.get("running") else ""})""" if not r.get("error") else f"""Not started: {r["error"]}""")

async def _h_test_cancel(request, payload, imr):
    if not _admin(request): return imr
    conn = next((c for c in store.nodes() if c["_id"] == payload["node"]), None)
    r = await AIM.connections.call_capability(conn, "fleet_test_cancel", {}, timeout_s=30)
    return await _refresh(request, imr, f"Cancel: {r}")

async def _h_settings_save(request, payload, imr):
    if not _admin(request): return imr
    text = payload.get("config", "")
    try: c = json.loads(text)
    except ValueError as e: return imr.oob(f"""<div class="scroll-pane" style="padding:.8rem">{_settings_html(request, await _st(request), [f"not valid JSON: {e}"], text)}</div>""", "fleet-panel")
    problems = _check_cfg({**DEFAULTS, **c})
    if problems: return imr.oob(f"""<div class="scroll-pane" style="padding:.8rem">{_settings_html(request, await _st(request), problems, text)}</div>""", "fleet-panel")
    CONFIG_PATH.write_text(json.dumps(c, indent=1))
    _LAST_PLAN.clear()
    return await _refresh(request, imr, "Settings saved.")

async def _h_synth(request, payload, imr):
    if not _admin(request): return imr
    name = await asyncio.to_thread(experts.synthetic, int(payload["layers"]), int(payload["experts"]), int(payload["k"]), int(payload["tokens"]), int(payload["topics"]), 200, float(payload["skew"]))
    s = await _st(request); s["trace"] = name; await _st(request, s)
    return await _refresh(request, imr, f"Made {name}.")

async def _h_place(request, payload, imr):
    f = lambda k: float(payload[k])
    slots = int(f("gb") * 1000 / f("expert_mb"))
    r = await asyncio.to_thread(experts.plan, payload["trace"], int(f("nodes")), slots, [k for k in experts.PLACEMENTS if payload.get(f"pol_{k}")], f("holdout"), {"replicate": f("replicate"), "balance": f("balance")}, {k: f(k) for k in experts.COST})
    _EXPERT_RESULT[str(getattr(request.state.user, "id", ""))] = r
    return imr.oob(_place_html(r, _admin(request)), "fleet-place")

async def _h_save_pins(request, payload, imr):
    if not _admin(request): return imr
    r = _EXPERT_RESULT.get(str(getattr(request.state.user, "id", "")))
    if not r: return await _refresh(request, imr, "Plan a placement first.")
    return await _refresh(request, imr, f"""Pin lists saved: {await asyncio.to_thread(experts.save_pins, r, payload["policy"])}""")
