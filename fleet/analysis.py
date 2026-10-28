"""analysis.py - what Fleet reads out of the events, as plain functions over event rows from store.py; nothing here talks to a node.
Requests and their statistics, model loads and residency, jobs (where each request went and how it went: the routing results), activity by weekday and hour,
cache plans (POLICIES: each proposes actions for one node from what it holds and what its users asked for), and backtests of model-residency predictors (PREDICTORS) on recorded traffic.
Speeds are computed only where the counts are large enough to mean something: prefill from 32 prompt tokens, decode from 8 generated tokens."""
import math, collections
from datetime import datetime

TEST_JOB = "fleettest"   # requests made by Fleet's own tests (job fleettest.<id>): left out of usage statistics unless asked for
LLAMA_APIS = ("chat", "generate")

def pct(xs, p: float):
    xs = sorted(x for x in xs if x is not None)
    if not xs: return None
    k = (len(xs) - 1) * p / 100
    f, c = math.floor(k), math.ceil(k)
    return xs[f] + (xs[c] - xs[f]) * (k - f)

# --- Requests ---

def requests(evs: list, tests: bool = False) -> list:
    """req.end events (llama.cpp requests) and engine.req events (requests to an engine such as colibri) as requests: start, end, speeds, prefix reuse."""
    out = []
    for e in evs:
        if e["kind"] not in ("req.end", "engine.req"): continue
        job = e.get("job") or (e.get("rid") or "").split(".", 1)[0]
        if job.startswith(TEST_JOB) and not tests: continue
        if e["kind"] == "engine.req":
            out.append({**e, "job": job, "api": "engine", "model": e.get("name"), "outcome": "served" if (e.get("status") or 500) < 400 else "failed", "end": e["t"], "start": e["t"] - (e.get("seconds") or 0), "prefill_tps": None, "decode_tps": None, "reuse": None})
            continue
        pn, pms, gn, gms, cn = e.get("prompt_n"), e.get("prompt_ms"), e.get("gen_n"), e.get("gen_ms"), e.get("cache_n")
        out.append({**e, "job": job, "end": e["t"], "start": e["t"] - (e.get("seconds") or 0), "prefill_tps": pn / pms * 1000 if pn and pms and pn >= 32 else None, "decode_tps": gn / gms * 1000 if gn and gms and gn >= 8 else None,
                    "reuse": cn / (cn + pn) if cn is not None and pn is not None and cn + pn else None})
    return out

def summary(reqs: list) -> dict:
    oc = collections.Counter(r.get("outcome") for r in reqs)
    tin, cached = sum(r.get("prompt_n") or 0 for r in reqs), sum(r.get("cache_n") or 0 for r in reqs)
    col = lambda k: [r.get(k) for r in reqs]
    return {"n": len(reqs), "served": oc.get("served", 0), "failed": oc.get("failed", 0), "cancelled": oc.get("cancelled", 0), "seconds_p50": pct(col("seconds"), 50), "seconds_p90": pct(col("seconds"), 90),
            "wait_p50": pct(col("wait_s"), 50), "wait_p90": pct(col("wait_s"), 90), "wait_sum": sum(r.get("wait_s") or 0 for r in reqs), "prefill_p50": pct(col("prefill_tps"), 50),
            "decode_p10": pct(col("decode_tps"), 10), "decode_p50": pct(col("decode_tps"), 50), "decode_p90": pct(col("decode_tps"), 90), "reuse": cached / (cached + tin) if cached + tin else None,
            "tokens_in": tin, "tokens_cached": cached, "tokens_out": sum(r.get("gen_n") or 0 for r in reqs), "busy_s": sum(r.get("seconds") or 0 for r in reqs)}

def by(rows: list, *keys) -> dict:
    g = collections.defaultdict(list)
    for r in rows: g[tuple(r.get(k) for k in keys)].append(r)
    return dict(sorted(g.items(), key=lambda kv: [str(x) for x in kv[0]]))

def per_hour(reqs: list, t0: float, t1: float) -> list:
    """Requests started in each hour from t0 to t1 (for sparklines)."""
    n = max(1, math.ceil((t1 - t0) / 3600))
    out = [0] * n
    for r in reqs:
        i = int((r["start"] - t0) // 3600)
        if 0 <= i < n: out[i] += 1
    return out

# --- Loads and residency ---

def loads(evs: list) -> list:
    """Model loads (llama.spawn to llama.ready, or to llama.crash: failed) and engine starts (engine.start to engine.up or engine.failed), each with its seconds."""
    pending, out = {}, []
    for e in evs:
        n, k = e["node"], e["kind"]
        if k == "llama.spawn": pending[(n, "llama")] = e
        elif k in ("llama.ready", "llama.crash") and (n, "llama") in pending:
            s = pending.pop((n, "llama"))
            out.append({"node": n, "what": "model", "model": s.get("model"), "t": s["t"], "load_s": e["t"] - s["t"], "ok": k == "llama.ready", "cause": e.get("kind_", ""), "params": s.get("params") or {}})
        elif k == "engine.start": pending[(n, "engine:" + str(e.get("name")))] = e
        elif k in ("engine.up", "engine.failed") and (n, "engine:" + str(e.get("name"))) in pending:
            s = pending.pop((n, "engine:" + str(e.get("name"))))
            out.append({"node": n, "what": "engine", "model": e.get("name"), "t": s["t"], "load_s": e["t"] - s["t"], "ok": k == "engine.up", "cause": e.get("status", ""), "params": {}})
    return out

def residency(evs: list, now: float) -> list:
    """What was loaded when, per node: a model from llama.ready until llama-server is respawned, stopped or crashes (or the node restarts); an engine from engine.up until engine.stop or engine.failed."""
    cur, out = {}, []
    def close(key, t):
        if key in cur: out.append({**cur.pop(key), "t1": t})
    for e in evs:
        n, k = e["node"], e["kind"]
        if k == "node.start":
            for key in [x for x in cur if x[0] == n]: close(key, e["t"])
        if k in ("llama.spawn", "llama.stop", "llama.crash", "llama.crash_mid_stream"): close((n, "llama"), e["t"])
        if k == "llama.ready": cur[(n, "llama")] = {"node": n, "what": "model", "name": e.get("model"), "t0": e["t"], "params": e.get("params") or {}}
        if k == "engine.up": cur[(n, "engine:" + str(e.get("name")))] = {"node": n, "what": "engine", "name": e.get("name"), "t0": e["t"], "params": {}}
        if k in ("engine.stop", "engine.failed"): close((n, "engine:" + str(e.get("name"))), e["t"])
    for key in list(cur): close(key, now)
    return out

# --- Jobs: where each job's requests went and how they went ---

def jobs(reqs: list) -> list:
    out = []
    for (job,), rs in by(reqs, "job").items():
        oc = collections.Counter(r.get("outcome") for r in rs)
        out.append({"job": job, "n": len(rs), "nodes": sorted({str(r["node"]) for r in rs}), "models": sorted({str(r.get("model")) for r in rs}), "start": min(r["start"] for r in rs), "end": max(r["end"] for r in rs),
                    "busy_s": sum(r.get("seconds") or 0 for r in rs), "wait_s": sum(r.get("wait_s") or 0 for r in rs), "tokens_in": sum(r.get("prompt_n") or 0 for r in rs), "tokens_cached": sum(r.get("cache_n") or 0 for r in rs),
                    "tokens_out": sum(r.get("gen_n") or 0 for r in rs), "decode_p50": pct([r.get("decode_tps") for r in rs], 50), "outcomes": dict(oc)})
    return sorted(out, key=lambda j: -j["end"])

# --- Activity by weekday and hour (portal's local time) ---

def heat(reqs: list, measure: str = "busy") -> list:
    """7 x 24: busy seconds (or request count) by weekday (Monday first) and hour of the start."""
    m = [[0.0] * 24 for _ in range(7)]
    for r in reqs:
        d = datetime.fromtimestamp(r["start"])
        m[d.weekday()][d.hour] += (r.get("seconds") or 0) if measure == "busy" else 1
    return m

def quiet_windows(m: list, hours: int = 3, top: int = 3) -> list:
    """The `top` runs of `hours` consecutive hours (wrapping over midnight) with the least activity, per weekday: where long jobs disturb the fewest requests."""
    out = []
    for d in range(7):
        runs = sorted(((sum(m[d][(h + i) % 24] for i in range(hours)), h) for h in range(24)))[:top]
        out.append([(h, (h + hours) % 24, v) for v, h in runs])
    return out

# --- Policies: actions proposed for one node ---
# A policy gets ctx: node (connection id), state (the node's latest /fleet/state), reqs (this node's requests in the last 7 days, tests left out), loads (this node's loads), events (this node's directive.done events),
# now, p (the policy's parameters: its defaults with the node's settings over them). It returns proposed actions: {"action", "args", "why", "value"} - value: estimated seconds saved, or None.

POLICIES: dict = {}

def policy(name: str, doc: str, **defaults):
    def reg(fn):
        POLICIES[name] = {"fn": fn, "doc": doc, "params": defaults}
        return fn
    return reg

def _demand(reqs: list, now: float, half_life_h: float) -> collections.Counter:
    """Requests per model, each weighted by 0.5 per half_life_h of age."""
    c = collections.Counter()
    for r in reqs: c[r.get("model")] += 0.5 ** ((now - r["end"]) / 3600 / max(half_life_h, 0.01))
    return c

def _warm_options(params: dict) -> dict: return {k: v for k, v in (("num_ctx", params.get("ctx")), ("kv_cache_type", params.get("kv_cache_type"))) if v is not None}   # the configuration the model last loaded with, so the next real request finds it as it expects

@policy("observe", "Proposes nothing: Fleet only records and shows this node.")
def _observe(ctx: dict) -> list: return []

@policy("keep_warm", "After idle_s without a request, load the model this node's users asked for most (weighted by recency, half-life half_life_h) when it is not loaded and has at least min_share of the demand - so the next request does not wait for the load. Never while the node is busy.", idle_s=600, half_life_h=12, min_share=0.5)
def _keep_warm(ctx: dict) -> list:
    st, p, now = ctx["state"], ctx["p"], ctx["now"]
    llama = [r for r in ctx["reqs"] if r.get("api") in LLAMA_APIS]
    if not st or st.get("busy") or not llama or now - max(r["end"] for r in llama) < p["idle_s"]: return []
    d = _demand(llama, now, p["half_life_h"])
    top, w = d.most_common(1)[0]
    res = st.get("resident") or {}
    if (res.get("alive") and res.get("model") == top) or w / sum(d.values()) < p["min_share"]: return []
    ok = [l for l in ctx["loads"] if l["model"] == top and l["ok"]]
    sessions = collections.Counter(r.get("session") for r in llama if r.get("model") == top)
    load = pct([l["load_s"] for l in ok], 50)
    return [{"action": "warm", "args": {"model": top, "options": _warm_options(ok[-1]["params"]) if ok else {}, "session": sessions.most_common(1)[0][0] if sessions else "default"},
             "why": f"""{top} has {w / sum(d.values()):.0%} of recent demand; loaded now: {res.get("model") if res.get("alive") else "nothing"}; its loads took {load:.0f} s (median)""" if load else f"""{top} has {w / sum(d.values()):.0%} of recent demand; loaded now: {res.get("model") if res.get("alive") else "nothing"}""",
             "value": round(w / sum(d.values()) * load, 1) if load else None}]

@policy("pin_sessions", "Pin the saved KV contexts of sessions that keep coming back (at least min_requests requests in window_h; skip: sessions never pinned), at most max_pins; unpin sessions with no request for unpin_after_h. A pinned context survives the ring's pruning, so its session resumes without re-reading its prompt after a model swap.", window_h=72, min_requests=5, max_pins=4, unpin_after_h=168, skip=["default"])
def _pin_sessions(ctx: dict) -> list:
    st, p, now = ctx["state"], ctx["p"], ctx["now"]
    if not st: return []
    pins = {x[8:] for x in st.get("pins") or [] if x.startswith("session:")}
    recent = [r for r in ctx["reqs"] if r.get("api") in LLAMA_APIS and r.get("session") and r["end"] >= now - p["window_h"] * 3600]
    counts = collections.Counter(r["session"] for r in recent)
    want = [s for s, n in counts.most_common() if n >= p["min_requests"] and s not in p["skip"]][:p["max_pins"]]
    last = {r["session"]: r["end"] for r in ctx["reqs"] if r.get("session")}
    ring = {e["session"] for e in st.get("kv_ring") or []}
    out = [{"action": "pin", "args": {"session": s}, "why": f"""{counts[s]} requests in {p["window_h"]} h{"" if s in ring else " (no saved context yet - pinned for when there is)"}""", "value": None} for s in want if s not in pins]
    out += [{"action": "unpin", "args": {"session": s}, "why": f"""no request for {(now - last[s]) / 3600:.0f} h""" if s in last else "no request in the recorded history", "value": None} for s in pins if s not in want and now - last.get(s, 0) > p["unpin_after_h"] * 3600]
    return out

@policy("prefetch_next", "Read the files of the models most likely to be asked for next (recency-weighted demand; not the loaded one) into the node's page cache, up to budget_gb in total, each at most once per refresh_h, after idle_s without requests - a later load then reads RAM instead of the disk. budget_gb 0 = off.", budget_gb=0, half_life_h=12, refresh_h=6, idle_s=300)
def _prefetch_next(ctx: dict) -> list:
    st, p, now = ctx["state"], ctx["p"], ctx["now"]
    llama = [r for r in ctx["reqs"] if r.get("api") in LLAMA_APIS]
    if not st or not p["budget_gb"] or st.get("busy") or not llama or now - max(r["end"] for r in llama) < p["idle_s"]: return []
    size = {m["name"]: m["size_gb"] for m in st.get("models") or []}
    done = {(e.get("result") or {}).get("model"): e["t"] for e in ctx["events"] if e["kind"] == "directive.done" and e.get("action") == "prefetch" and e.get("ok")}
    res, out, used = (st.get("resident") or {}).get("model"), [], 0.0
    for m, w in _demand(llama, now, p["half_life_h"]).most_common():
        if m == res or m not in size or used + size[m] > p["budget_gb"]: continue
        used += size[m]
        if now - done.get(m, 0) > p["refresh_h"] * 3600: out.append({"action": "prefetch", "args": {"model": m}, "why": f"""next most asked-for model ({size[m]:.1f} GB of the {p["budget_gb"]} GB budget)""", "value": None})
    return out

def plan(ctx: dict, policies: dict) -> list:
    """Every listed policy's proposals for one node; policies: {name: its parameters}. An unknown policy name is an error the caller shows."""
    out = []
    for name, params in policies.items():
        pol = POLICIES[name]
        out += [{**a, "policy": name} for a in pol["fn"]({**ctx, "p": {**pol["params"], **(params or {})}})]
    return out

# --- Backtest: model-residency predictors replayed on one node's recorded requests ---

PREDICTORS: dict = {}

def predictor(name: str, doc: str):
    def reg(fn):
        PREDICTORS[name] = {"fn": fn, "doc": doc}
        return fn
    return reg

@predictor("stay", "Never loads ahead: every change of model waits for its load (a node without Fleet).")
def _p_stay(hist: list, now: float): return None

@predictor("recent", "The model asked for most in the last 10 requests.")
def _p_recent(hist: list, now: float):
    c = collections.Counter(r["model"] for r in hist[-10:])
    return c.most_common(1)[0][0] if c else None

@predictor("hour", "The model asked for most at this hour of day on earlier days; recent where there is no such history.")
def _p_hour(hist: list, now: float):
    h = datetime.fromtimestamp(now).hour
    c = collections.Counter(r["model"] for r in hist if datetime.fromtimestamp(r["start"]).hour == h)
    return c.most_common(1)[0][0] if c else _p_recent(hist, now)

@predictor("markov", "The model that most often came right after the model asked for last.")
def _p_markov(hist: list, now: float):
    if len(hist) < 2: return None
    c = collections.Counter(b["model"] for a, b in zip(hist, hist[1:]) if a["model"] == hist[-1]["model"])
    return c.most_common(1)[0][0] if c else None

def backtest(reqs: list, load_s: dict, default_load_s: float, idle_s: float) -> list:
    """Replays one node's llama.cpp requests (by start time) through its one model slot, once per predictor. A request for a model that is not loaded waits for its load (load_s: measured median per model, default_load_s otherwise).
    After idle_s without a request the predictor may load a model ahead: a request for it then waits only for what is left of that load (a hit); a request for another model wasted it.
    Requests keep their recorded start times, so the replay shows waits and loads, not how much earlier work would have finished."""
    reqs = sorted((r for r in reqs if r.get("api") in LLAMA_APIS), key=lambda r: r["start"])
    out = []
    for name, P in PREDICTORS.items():
        resident, ready, ahead, wait, n_loads, wasted, hits, prev_end = None, 0.0, None, 0.0, 0, 0, 0, None
        for i, r in enumerate(reqs):
            if prev_end is not None and r["start"] - prev_end >= idle_s:
                g = P["fn"](reqs[:i], prev_end + idle_s)
                if g and g != resident:
                    if ahead: wasted += 1
                    resident, ready, ahead, n_loads = g, prev_end + idle_s + load_s.get(g, default_load_s), g, n_loads + 1
            if r["model"] == resident:
                wait += max(0.0, ready - r["start"])
                hits += 1 if ahead == r["model"] else 0
            else:
                wasted += 1 if ahead else 0
                resident, n_loads = r["model"], n_loads + 1
                ready = r["start"] + load_s.get(r["model"], default_load_s)
                wait += ready - r["start"]
            ahead, prev_end = None, max(prev_end or 0, r["end"])
        out.append({"predictor": name, "doc": P["doc"], "requests": len(reqs), "wait_s": round(wait, 1), "wait_per_request_s": round(wait / len(reqs), 2) if reqs else None, "loads": n_loads, "loads_ahead_hit": hits, "loads_ahead_wasted": wasted})
    return out
