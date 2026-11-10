"""python3 tests/test_heat.py - expert heat, arrangements, in-flight limits, benchmark items and demand mixes."""
import sys, copy, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2] / "Athene" / "tests"))
import numpy as np
from load import athene
from asio_core import Catalog, run, heat as H
run.use_athene(athene())
cat = Catalog()
FAILS = []
def check(name, ok, info=""):
    print(("ok   " if ok else "FAIL ") + name + ("" if ok else f"  {info}"))
    if not ok: FAILS.append(name)

q = np.full(1000, 0.01)
check("Che: an LRU cache under even demand hits in proportion to its size", abs(H._che(q, 250).mean() - 0.25) < 0.01, H._che(q, 250).mean())
p = 1 / np.arange(1, 385) ** 1.0
qz = np.tile(np.minimum(6 * p / p.sum(), 1), (40, 1))
hit = {a: H.Heat(qz, a, [None, None], [60, 60], [80, 80], [3, 3], 0.01875, 1).hit for a in H.ARRANGEMENTS}
check("with skewed heat and room for a third of the experts: known heat beats a learning cache, which beats a blind fixed split", hit["hottest"] > hit["learned"] > hit["spread"] and abs(hit["spread"] - 120 / 288) < 0.03, hit)
hu = {a: H.Heat(np.full((40, 384), 6 / 384), a, [None], [60], [80], [3], 0.01875, 1).hit for a in H.ARRANGEMENTS}
check("with even heat, no arrangement can do better than the cache's share", all(abs(v - 60 / 288) < 0.02 for v in hu.values()), hu)
h2 = H.Heat(qz, "hottest", [None, None], [60, 60], [100, 50], [3, 3], 0.01875, 1)
check("known heat loads machines in proportion to their memory bandwidth", h2.load[0] > h2.load[1] * 1.5, h2.load)
st1, st64 = sum(r + d for r, d in h2.step(1)), sum(r + d for r, d in h2.step(64))
check("a batch reads each chosen expert once: 64 tokens read far less than 64 times one token", 4.4 < st1 < 4.6 and st64 < 64 * st1 / 2, (st1, st64))

base = run.scenario(cat, "backlog_ds41_3nodes")
def with_heat(arr, skew=1.0):
    sc = copy.deepcopy(base)
    sc["engines"][0].update(heat={"zipf": skew}, arrangement=arr)
    return run.once(cat, sc, "median", 0)
r = {a: with_heat(a) for a in H.ARRANGEMENTS}
check("in a run, known heat finishes the backlog sooner than a blind split", r["hottest"]["metrics"]["span_s"] < r["spread"]["metrics"]["span_s"], {a: x["metrics"]["span_s"] for a, x in r.items()})
check("and the engine reports its expert hit rate", r["hottest"]["engines"]["ds"]["hit"] > r["spread"]["engines"]["ds"]["hit"])
sc = copy.deepcopy(base)
sc["engines"][0].update(heat={"zipf": None}, arrangement="learned")
mc = run.monte_carlo(cat, sc, 16)
check("unknown skew is an uncertain input Monte Carlo ranks", any(k == "model.deepseek-v4.1-flash-1017.heat_skew" for k, *_ in [tuple(x.values())[:1] for x in mc["sensitivity"]["span_s"]]) or "model.deepseek-v4.1-flash-1017.heat_skew" in str(mc["sensitivity"]), str(mc["sensitivity"])[:300])
cat.add("heat", "t1", {"model": "deepseek-v4.1-flash-1017", "counts": [[100] * 10 + [1] * 374] * 40, "src": "test"})
sc = copy.deepcopy(base)
sc["engines"][0].update(heat="t1", arrangement="hottest")
hc = run.once(cat, sc, "median", 0)["engines"]["ds"]["hit"]
check("heat from a catalog entry of counts: the hot tenth of choices sits in RAM", 0.73 < hc < 0.85, hc)

bench = {"nodes": {"n4": "getorli_h255"}, "engines": [{"id": "q9", "model": "qwen3.8-9b", "backend": "llama_vulkan", "node": "n4", "slots": 4, "ctx": 8192}], "routes": {"main": ["q9"]},
         "workloads": [{"name": "set", "kind": "benchmark", "role": "main", "items": [[500, 200], [2000, 400], [300, 100]] * 8}]}
def bench_with(limit):
    sc = copy.deepcopy(bench)
    sc["workloads"][0]["max_in_flight"] = limit
    return run.once(cat, sc, "median", 0)["metrics"]
b1, b4 = bench_with(1), bench_with(4)
check("a benchmark runs its whole set", b1["jobs"] == 24 and b4["ok"] == 24, b1)
check("letting four run at once fills the slots and finishes sooner", b4["span_s"] < b1["span_s"] * 0.6, (b1["span_s"], b4["span_s"]))

mixsc = {"nodes": {"n4": "getorli_h255"}, "engines": [{"id": "q9", "model": "qwen3.8-9b", "backend": "llama_vulkan", "node": "n4", "slots": 2, "ctx": 16384}], "routes": {"main": ["q9"]},
         "workloads": [{"name": "day", "kind": "mix", "role": "main", "rate_per_h": 30, "duration_h": 4, "profile": [0] * 2 + [1] * 22,
                        "types": [{"name": "chat", "kind": "stream", "share": 3, "prompt": 400, "out": 200, "slo_s": 60}, {"name": "agent", "kind": "agent", "share": 1, "count": 1, "steps_needed": 3, "slo_s": 600}]}]}
m = run.once(cat, mixsc, "median", 0)
wl = m["workloads"]
check("a mix reports each job type on its own", set(wl) == {"day.chat", "day.agent"} and wl["day.chat"]["jobs"] > wl["day.agent"]["jobs"], wl.keys())
check("no arrivals in hours the profile closes", all(j["t_arrive"] >= 7200 for j in m["jobs"]), min(j["t_arrive"] for j in m["jobs"]))
check("SLO shares are reported per type and overall", wl["day.chat"]["within_slo"] is not None and m["metrics"]["slo_share"] is not None, (wl["day.chat"]["within_slo"], m["metrics"]["slo_share"]))
# --- Rented machines and quotes ---
q = run.quote(cat, bench, ["q9"], "rent_rtx4090", 1, 1.0, 5.0, n=6)
o = q["options"]
check("a quote gives three options; rented machines finish the set sooner", o["overflow"]["span_p50"] < o["local"]["span_p50"] and o["planned"]["span_p50"] < o["local"]["span_p50"], {k: v["span_p50"] for k, v in o.items()})
check("local costs nothing rented and carries no fee; the others show the fee on its own line", o["local"]["rent_p50"] == 0 and o["local"]["fee"] == 0 and o["overflow"]["fee"] == 5.0 and o["overflow"]["total_p50"] > o["overflow"]["rent_p50"])
check("the ceiling sits above the 1-in-20 rented cost", o["overflow"]["ceiling"] >= o["overflow"]["rent_p95"] > 0)
quiet = copy.deepcopy(bench)
quiet["workloads"][0].update(items=[[300, 50]], max_in_flight=1)
oq = run.quote(cat, quiet, ["q9"], "rent_rtx4090", 1, 1.0, 0.0, n=4)["options"]
check("overflow that is never needed costs nothing; planned is billed for the whole run", oq["overflow"]["rent_p50"] == 0 and oq["planned"]["rent_p50"] > 0, (oq["overflow"]["rent_p50"], oq["planned"]["rent_p50"]))
r = run.once(cat, run.with_rented(cat, bench, ["q9"], "rent_rtx4090", 1, None), "median", 0)
check("a rented machine uses no local power and reports its rent", r["nodes"]["rent1"]["energy_wh"] == 0 and r["nodes"]["rent1"]["rent_usd"] > 0 and r["metrics"]["cost_rent_usd"] > 0, r["nodes"]["rent1"])
big = run.once(cat, run.with_rented(cat, base, ["ds"], "rent_rtx4090", 1, None), "median", 0)
check("a model that cannot fit a rented GPU is reported, not hidden", bool(big.get("error")) and "rent1" in big["error"], big.get("error"))
print("ALL PASSED" if not FAILS else f"FAILED: {FAILS}")
sys.exit(1 if FAILS else 0)
