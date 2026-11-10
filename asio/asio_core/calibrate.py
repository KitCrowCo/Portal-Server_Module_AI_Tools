"""asio/calibrate.py - turns what really ran into measured catalog values: Magpie's journal events (from Fleet's database or the node's own journal files), agent step records, and direct readings (a wall meter, ping, iperf3).
Request timings are inverted through the same physics the simulator uses: for each request, the one unknown value is solved so the model reproduces the time it took (decode: memory efficiency, or colibri's per-step overhead for MoE engines; prefill: compute efficiency). Every other value is held at its median, so the solved number absorbs their error too; the src line says so.
Nothing is written until asked: propose() gives the values and the file they would make, write() saves it as a new file in the user catalog folder, never over an existing one. Deleting that file undoes it.
mapping: {"nodes": {fleet node id or node name: catalog node id}, "models": {served model name or glob: {"model", "quant", "backend"}}}.
A catalog path is collection/id/field/...: slashes, because model ids have dots in them (models/qwen3.8-27b/agent/p_valid)."""
import json, time, glob, sqlite3, fnmatch, copy
from pathlib import Path
import numpy as np
from asio_core.values import Valuer
from asio_core.sim import Sim

MIN_GEN, MIN_PROMPT = 8, 32   # fewer tokens than this say more about fixed costs than about speed (Fleet uses the same cut-offs)

# --- Reading ---

def fleet_events(db: str, since: float = 0, kinds=("req.end", "engine.req", "agent.step")) -> list:
    """Events from Fleet's database, read-only; each gets its node from the row."""
    con = sqlite3.connect(f"file:{db}?mode=ro", uri=True)
    con.row_factory = sqlite3.Row
    rows = con.execute(f"""SELECT node, t, data FROM events WHERE t >= ? AND kind IN ({",".join("?" * len(kinds))}) ORDER BY t""", (since, *kinds)).fetchall()
    con.close()
    return [{**json.loads(r["data"]), "node": r["node"], "t": r["t"]} for r in rows]

def journal_events(pattern: str, node: str) -> list:
    """Events from a node's own journal files (JSON lines); node names the machine they came from."""
    out = []
    for f in sorted(glob.glob(pattern)):
        for line in open(f, encoding="utf-8", errors="replace"):
            try: out.append({**json.loads(line), "node": node})
            except json.JSONDecodeError: pass   # a line cut off by a crash
    return out

# --- Observations from requests ---

def _match(mapping: dict, model: str):
    for pat, m in mapping.get("models", {}).items():
        if model and (model == pat or fnmatch.fnmatchcase(model, pat)): return m
    return None

def observations(events: list, mapping: dict) -> tuple:
    """(observations, skipped counts). Concurrency is read from overlapping requests on the same node: a step served k streams, not one."""
    reqs, skipped = [], {"no node mapping": 0, "no model mapping": 0, "too short": 0, "failed": 0}
    for e in events:
        if e.get("kind") not in ("req.end", "engine.req") or (e.get("kind") == "engine.req" and not e.get("gen")): continue
        node, m = mapping.get("nodes", {}).get(e.get("node")) or mapping.get("nodes", {}).get(e.get("node_name")), _match(mapping, e.get("model"))
        if not node: skipped["no node mapping"] += 1
        elif not m: skipped["no model mapping"] += 1
        elif e.get("outcome", "served") != "served": skipped["failed"] += 1
        else: reqs.append({**e, "cat_node": node, **m, "start": e["t"] - (e.get("seconds") or 0)})
    obs = []
    for r in reqs:
        mid = (r["start"] + r["t"]) / 2
        k = sum(1 for o in reqs if o["node"] == r["node"] and o["model"] == r["model"] and o["start"] <= mid <= o["t"])
        pn, cn, gn = r.get("prompt_n") or 0, r.get("cache_n") or 0, r.get("gen_n") or 0
        if gn >= MIN_GEN and r.get("gen_ms"): obs.append({"what": "decode", "node": r["cat_node"], "model": r["model"], "quant": r.get("quant"), "backend": r.get("backend"), "k": max(k, 1), "depth": cn + pn + gn / 2, "s_per_tok": r["gen_ms"] / 1000 / gn, "src_node": r["node"]})
        if pn >= MIN_PROMPT and r.get("prompt_ms"): obs.append({"what": "prefill", "node": r["cat_node"], "model": r["model"], "quant": r.get("quant"), "backend": r.get("backend"), "k": 1, "depth": cn + pn / 2, "s_per_tok": r["prompt_ms"] / 1000 / pn, "src_node": r["node"]})
        if not (gn >= MIN_GEN or pn >= MIN_PROMPT): skipped["too short"] += 1
    return obs, skipped

# --- Solving one value per observation ---

def _target(cat, o: dict) -> tuple:
    """(value key in a run, catalog path, low, high, direction): which one number this observation decides."""
    model, back = cat.get("models", o["model"]), cat.get("backends", o["backend"] or cat.get("models", o["model"])["backends"][0])
    if o["what"] == "prefill": return "engine.e.compute_eff", f"""backends/{back["id"]}/compute_eff""", 0.005, 1.5, -1
    if model["arch"] == "moe" and back.get("token_overhead_ms"): return "engine.e.token_overhead_ms", f"""backends/{back["id"]}/token_overhead_ms""", 0.0, 60000.0, 1
    if back["device"] == "cpu" and cat.get("nodes", o["node"]).get("mem_eff_cpu"): return "node.n.mem_eff_cpu", f"""nodes/{o["node"]}/mem_eff_cpu""", 0.02, 1.5, -1
    return "engine.e.mem_eff.n", f"""backends/{back["id"]}/mem_eff""", 0.02, 1.5, -1

def _modeled(cat, o: dict, key: str, x: float) -> float:
    sc = {"nodes": {"n": o["node"]}, "engines": [{"id": "e", "model": o["model"], "quant": o.get("quant"), "backend": o.get("backend"), "node": "n", "slots": o["k"], "ctx": 1024}], "workloads": []}
    if not sc["engines"][0]["quant"]: sc["engines"][0].pop("quant")
    if not sc["engines"][0]["backend"]: sc["engines"][0].pop("backend")
    phys = Sim(sc, cat, Valuer("median", overrides={key: x, "node.n.os_reserve_gb": 0})).engines["e"].phys   # memory is not what is being measured: the reserve is lifted so any request size fits
    if o["what"] == "prefill": return phys.step([], 512, int(o["depth"]))[0] / 512
    t, _, mult = phys.step([int(o["depth"])] * o["k"])
    return t / mult

def solve(cat, o: dict, iters: int = 40) -> dict:
    """The value that makes the model take as long as the observation did; clamped at the search range, and flagged when it is."""
    key, path, lo, hi, sign = _target(cat, o)
    f = lambda x: _modeled(cat, o, key, x) - o["s_per_tok"]
    flo, fhi = f(lo), f(hi)
    if flo * fhi > 0: return {**o, "path": path, "value": lo if abs(flo) < abs(fhi) else hi, "clamped": True}
    for _ in range(iters):
        mid = (lo + hi) / 2
        if (f(mid) > 0) == (flo > 0): lo, flo = mid, f(mid)
        else: hi = mid
    return {**o, "path": path, "value": (lo + hi) / 2, "clamped": False}

# --- Agent steps and direct readings ---

def agent_values(steps: list, mapping: dict, boot: int = 200, seed: int = 0) -> dict:
    """steps: {"model", "task", "valid", "after_bad", "progress"} per agent step. Rates are bootstrapped over tasks, so their spread is how sure they are."""
    rng, out = np.random.default_rng(seed), {}
    by_model = {}
    for s in steps:
        m = _match(mapping, s.get("model"))
        if m: by_model.setdefault(m["model"], {}).setdefault(s.get("task"), []).append(s)
    for model, tasks in by_model.items():
        ids = list(tasks)
        rates = {"p_valid": lambda ss: [x["valid"] for x in ss if not x.get("after_bad")], "p_recover": lambda ss: [x["valid"] for x in ss if x.get("after_bad")], "p_progress": lambda ss: [x.get("progress", False) for x in ss if x["valid"]]}
        for name, pick in rates.items():
            vals = []
            for _ in range(boot):
                xs = [v for t in rng.choice(ids, len(ids)) for v in pick(tasks[t])]
                if xs: vals.append(round(float(np.mean(xs)), 4))
            n = sum(len(pick(ss)) for ss in tasks.values())
            if vals and n >= 5: out[f"models/{model}/agent/{name}"] = {"values": vals, "n": n, "tasks": len(ids)}
    return out

def direct(readings: list) -> dict:
    """{"path": catalog path (nodes/getorli_h255/power_w/idle), "value", "src", "date"} each, from a wall meter, ping, iperf3 or anything read by hand."""
    out = {}
    for r in readings: out.setdefault(r["path"], {"values": [], "src": r.get("src", "direct reading")})["values"].append(float(r["value"]))
    return out

# --- Proposing and writing ---

def _get(cat, path: str):
    coll, id_, *fields = path.split("/")
    v = cat.get(coll, id_)
    for f in fields: v = v.get(f) if isinstance(v, dict) else None
    return v

def propose(cat, events: list = (), mapping: dict = None, agent_steps: list = (), readings: list = (), max_obs: int = 400, date: str = None) -> dict:
    """What a calibration would change: {"values": {path: {now, measured p50, n, src}}, "file": the catalog file it would write, "skipped", "clamped"}."""
    mapping, date = mapping or {}, date or time.strftime("%Y-%m-%d")
    obs, skipped = observations(list(events), mapping)
    if len(obs) > max_obs: obs = [obs[i] for i in np.linspace(0, len(obs) - 1, max_obs).astype(int)]   # spread over the whole period, not its start
    solved = [solve(cat, o) for o in obs]
    found = {}
    for s in solved:
        if s["clamped"]: continue   # outside any plausible range: the request was waiting on something the model does not see
        d = found.setdefault(s["path"], {"values": [], "nodes": set(), "models": set()})
        d["values"].append(round(s["value"], 5))
        d["nodes"].add(s["src_node"])
        d["models"].add(s["model"])
    meas = {p: {"values": d["values"], "src": f"""solved from {len(d["values"])} requests ({", ".join(sorted(d["models"]))} on {", ".join(sorted(d["nodes"]))}) with every other value at its median"""} for p, d in found.items()}
    meas.update({p: {"values": d["values"], "src": f"""{d["n"]} agent steps over {d["tasks"]} tasks, bootstrapped"""} for p, d in agent_values(agent_steps, mapping).items()})
    meas.update(direct(readings))
    file, values = {}, {}
    for path, m in meas.items():
        now = _get(cat, path)
        spec = {**(copy.deepcopy(now) if isinstance(now, dict) else {"est": [now] * 3} if isinstance(now, (int, float)) else {}), "measured": {"values": m["values"], "src": m["src"], "date": date}}
        coll, id_, *fields = path.split("/")
        node = file.setdefault(coll, {}).setdefault(id_, {})
        for f in fields[:-1]: node = node.setdefault(f, {})
        node[fields[-1]] = spec
        values[path] = {"now": Valuer("median", tier="estimate")(now, path) if now is not None else None, "measured_p50": float(np.median(m["values"])), "n": len(m["values"]), "src": m["src"]}
    return {"values": values, "file": file, "skipped": skipped, "clamped": sum(1 for s in solved if s["clamped"]), "observations": len(obs)}

def write(proposal: dict, folder: str, name: str = None) -> str:
    """Saves the proposal's file as a new file in folder; refuses to replace one that exists."""
    Path(folder).mkdir(parents=True, exist_ok=True)
    p = Path(folder) / f"""measured_{name or time.strftime("%Y%m%d_%H%M%S")}.json"""
    if p.exists(): raise FileExistsError(f"{p} exists: pick another name")
    p.write_text(json.dumps({"_about": "Measured values written by asio.calibrate; delete this file to go back to the estimates.", **proposal["file"]}, indent=1))
    return str(p)

# --- Command line ---

def main():
    import argparse
    from asio_core.catalog import Catalog
    a = argparse.ArgumentParser(description="Measured catalog values from Fleet's database, journal files, agent steps and readings. Shows what it would write; --write saves it.")
    a.add_argument("--fleet-db"), a.add_argument("--journal", help="glob of a node's journal files"), a.add_argument("--journal-node", help="the Fleet node name those files belong to")
    a.add_argument("--map", required=True, help="mapping JSON file"), a.add_argument("--agent-steps", help="JSON lines of agent steps"), a.add_argument("--readings", help="JSON lines of direct readings")
    a.add_argument("--since-days", type=float, default=30), a.add_argument("--catalog", default="data/ai_tools/asio/catalog"), a.add_argument("--write", action="store_true"), a.add_argument("--name")
    o = a.parse_args()
    cat, mapping = Catalog(o.catalog), json.load(open(o.map))
    ev = (fleet_events(o.fleet_db, time.time() - o.since_days * 86400) if o.fleet_db else []) + (journal_events(o.journal, o.journal_node) if o.journal else [])
    lines = lambda f: [json.loads(x) for x in open(f) if x.strip()] if f else []
    p = propose(cat, ev, mapping, lines(o.agent_steps) + [e for e in ev if e.get("kind") == "agent.step"], lines(o.readings))
    print(json.dumps({k: p[k] for k in ("values", "skipped", "clamped", "observations")}, indent=1))
    if o.write: print("written:", write(p, o.catalog, o.name))

if __name__ == "__main__": main()
