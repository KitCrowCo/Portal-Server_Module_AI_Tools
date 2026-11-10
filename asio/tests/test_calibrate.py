"""python3 tests/test_calibrate.py - the importer recovers known values: events are made from the model itself at chosen "true" values, then solved back."""
import sys, json, sqlite3, pathlib, tempfile
ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT.parent / "Athene")]
import numpy as np
from asio_core import Catalog, once
from asio_core.calibrate import _modeled, propose, write, fleet_events, observations

FAILS = []
def check(name, ok, info=""):
    print(("ok   " if ok else "FAIL ") + name + ("" if ok else f"  {str(info)[:500]}"))
    if not ok: FAILS.append(name)

cat = Catalog()
mapping = {"nodes": {"conn-getorli": "getorli_h255", "conn-x1": "minisforum_x1pro"}, "models": {"qwen3.8-27b*": {"model": "qwen3.8-27b", "quant": "Q4_K_M", "backend": "llama_vulkan"}, "ds41*": {"model": "deepseek-v4.1-flash-1017", "backend": "colibri"}}}
rng, events, t = np.random.default_rng(0), [], 1000.0
def req(node, served, model, quant, backend, eff_key, eff, gen_n, prompt_n, depth0):
    global t
    o = {"node": {"conn-getorli": "getorli_h255", "conn-x1": "minisforum_x1pro"}[node], "model": model, "quant": quant, "backend": backend, "k": 1}
    dec = _modeled(cat, {**o, "what": "decode", "depth": depth0 + prompt_n + gen_n / 2}, eff_key, eff) * gen_n
    pre = _modeled(cat, {**o, "what": "prefill", "depth": depth0 + prompt_n / 2}, "engine.e.compute_eff", 0.22) * prompt_n
    t += dec + pre + 30   # one at a time: no overlap
    events.append({"kind": "req.end", "node": node, "t": t, "model": served, "outcome": "served", "seconds": dec + pre, "prompt_n": prompt_n, "prompt_ms": pre * 1000, "gen_n": gen_n, "gen_ms": dec * 1000, "cache_n": depth0})
for i in range(12): req("conn-getorli", "qwen3.8-27b-q4", "qwen3.8-27b", "Q4_K_M", "llama_vulkan", "engine.e.mem_eff.n", 0.66, int(rng.integers(50, 600)), int(rng.integers(100, 3000)), int(rng.integers(0, 20000)))
for i in range(6): req("conn-x1", "ds41-flash", "deepseek-v4.1-flash-1017", None, "colibri", "engine.e.token_overhead_ms", 900.0, int(rng.integers(20, 200)), int(rng.integers(100, 800)), 0)
events.append({"kind": "req.end", "node": "conn-unknown", "t": t, "model": "x", "gen_n": 50, "gen_ms": 1000})
events.append({"kind": "req.end", "node": "conn-getorli", "t": t, "model": "other-model", "gen_n": 50, "gen_ms": 1000})

steps = []
for task in range(40):
    bad = False
    for s in range(10):
        p = 0.4 if bad else 0.8
        v = bool(rng.random() < p)
        steps.append({"model": "qwen3.8-27b-q4", "task": task, "valid": v, "after_bad": bad, "progress": bool(v and rng.random() < 0.6)})
        bad = not v
readings = [{"path": "nodes/getorli_h255/power_w/idle", "value": v, "src": "wall meter"} for v in (9.8, 10.1, 10.4)]

p = propose(cat, events, mapping, steps, readings, date="2026-11-07")
v = p["values"]
check("unmapped nodes and models are counted, not guessed", p["skipped"]["no node mapping"] == 1 and p["skipped"]["no model mapping"] == 1, p["skipped"])
check("decode recovers memory efficiency", abs(v["backends/llama_vulkan/mem_eff"]["measured_p50"] - 0.66) < 0.01, v.get("backends/llama_vulkan/mem_eff"))
check("prefill recovers compute efficiency", abs(v["backends/llama_vulkan/compute_eff"]["measured_p50"] - 0.22) < 0.01, v.get("backends/llama_vulkan/compute_eff"))
check("colibri's per-step overhead is recovered", abs(v["backends/colibri/token_overhead_ms"]["measured_p50"] - 900) < 15, v.get("backends/colibri/token_overhead_ms"))
check("agent rates come out near the truth", abs(v["models/qwen3.8-27b/agent/p_valid"]["measured_p50"] - 0.8) < 0.08 and abs(v["models/qwen3.8-27b/agent/p_recover"]["measured_p50"] - 0.4) < 0.12 and abs(v["models/qwen3.8-27b/agent/p_progress"]["measured_p50"] - 0.6) < 0.08, {k: x["measured_p50"] for k, x in v.items() if "agent" in k})
check("direct readings pass through", v["nodes/getorli_h255/power_w/idle"]["measured_p50"] == 10.1)
spec = p["file"]["backends"]["llama_vulkan"]["mem_eff"]
check("a measured value keeps its estimate, group and improvement", spec.get("group") == "llama_vulkan_mem_eff" and spec.get("est") and spec.get("improve") and spec["measured"]["date"] == "2026-11-07", spec)

d = tempfile.mkdtemp()
f = write(p, d, "test")
try:
    write(p, d, "test"); check("never replaces an existing file", False)
except FileExistsError: check("never replaces an existing file", True)
cat2 = Catalog(d)
r1, r2 = once(cat, "check_27b_getorli"), once(cat2, "check_27b_getorli")
check("the catalog reads the measured file, and runs change with it", r2["draws"]["llama_vulkan_mem_eff"] == np.median(json.load(open(f))["backends"]["llama_vulkan"]["mem_eff"]["measured"]["values"]) and r1["engines"]["q27"]["decode_tps"] != r2["engines"]["q27"]["decode_tps"], (r1["draws"].get("llama_vulkan_mem_eff"), r2["draws"].get("llama_vulkan_mem_eff")))
check("an estimate-only run still ignores it", once(cat2, "check_27b_getorli", tier="estimate")["draws"]["llama_vulkan_mem_eff"] == r1["draws"]["llama_vulkan_mem_eff"])

db = pathlib.Path(d) / "fleet.db"
con = sqlite3.connect(db)
con.execute("CREATE TABLE events (node TEXT, epoch TEXT, id INTEGER, t REAL, kind TEXT, rid TEXT, job TEXT, model TEXT, data TEXT, PRIMARY KEY (node, epoch, id))")
for i, e in enumerate(events): con.execute("INSERT INTO events VALUES (?, 'e', ?, ?, ?, '', '', ?, ?)", (e["node"], i, e["t"], e["kind"], e["model"], json.dumps({k: x for k, x in e.items() if k != "node"})))
con.commit(); con.close()
fe = fleet_events(str(db))
check("Fleet's database reads back the same observations", len(observations(fe, mapping)[0]) == len(observations(events, mapping)[0]) > 0)
over = [dict(e) for e in events[:4]]
for e in over: e["t"] = events[0]["t"]; e["seconds"] = events[0]["seconds"]
check("overlapping requests count as streams served together", max(o["k"] for o in observations(over, mapping)[0]) == 4)
print("ALL PASSED" if not FAILS else f"FAILED: {FAILS}")
sys.exit(1 if FAILS else 0)
