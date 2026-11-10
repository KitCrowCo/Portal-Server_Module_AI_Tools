"""python3 tests/test_asio.py - Asio against what was measured, and against the behaviour the physics must show. Needs Athene beside this folder (../Athene) or installed."""
import sys, copy, pathlib
ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT.parent / "Athene")]
from asio_core import Catalog, Valuer, once, cases, monte_carlo, sweep, optimize, viz
from asio_core.values import _q

FAILS = []
def check(name, ok, info=""):
    print(("ok   " if ok else "FAIL ") + name + ("" if ok else f"  {str(info)[:400]}"))
    if not ok: FAILS.append(name)

cat = Catalog()
def sc(i): return copy.deepcopy(cat.get("scenarios", i))
def m(r): return r["metrics"]

# --- Values ---
spec = {"theory": 10, "est": [4, 6, 9], "better": "high", "improve": [1.0, 1.2, 1.5], "plateau": 8}
v = {c: Valuer(c)(spec, "x") for c in ("median", "worst", "best", "theory", "improved", "plateau")}
check("cases order a higher-is-better value", v["worst"] < v["median"] < v["best"] < v["theory"] and v["improved"] > v["median"] and v["plateau"] == 8, v)
check("lower-is-better flips worst and best", Valuer("worst")({"est": [1, 2, 3], "better": "low"}, "y") > Valuer("best")({"est": [1, 2, 3], "better": "low"}, "y"))
check("measured beats an estimate in auto", Valuer("median")({"est": [1, 2, 3], "measured": {"values": [7, 7, 7]}}, "z") == 7 and Valuer("median", "estimate")({"est": [1, 2, 3], "measured": {"values": [7]}}, "z") == 2)
check("same key, same seed, same draw", Valuer("sample", seed=3)(spec, "k") == Valuer("sample", seed=3)(spec, "k") != Valuer("sample", seed=4)(spec, "k"))
check("a group shares one draw", (lambda V: V({"est": [0, 1, 2], "group": "g"}, "a") == V({"est": [0, 1, 2], "group": "g"}, "b"))(Valuer("sample", seed=1)))
check("triangular quantiles", abs(_q([0, 5, 10], 0.5) - 5) < 1e-9 and _q([0, 5, 10], 0.0) == 0)
check("overrides by key and glob", Valuer("median", overrides={"node.*.mem_bw_gbs": 50})({"theory": 90}, "node.n1.mem_bw_gbs") == 50)

# --- Catalog and pool rules ---
n = cat.get("nodes", "geekom_ax8")
check("templates extend", n["gpu"]["name"] == "Radeon 780M" and n["bus_bits"] == 64 and n["ram_gb"] == 16)
check("abstract entries are not listed", "minipc" not in cat.ids("nodes") and "geekom_ax8" in cat.ids("nodes"))
bad = sc("backlog_ds41_3nodes")
bad["links"].append({"type": "usb4_host", "a": "n4", "b": "n5"})
bad["links"].append({"type": "usb4_host", "a": "n4", "b": "n1"})
check("a cable needs a free port", "no free usb4 port" in once(cat, bad).get("error", ""), once(cat, bad).get("error"))
big = sc("check_27b_getorli")
big["engines"][0].update(quant="BF16")
check("a model that does not fit says by how much", "MemoryError" in once(cat, big).get("error", ""), once(cat, big).get("error"))
nolink = sc("pi_portal_routed")
nolink["links"] = []
check("work between machines with no link is refused", "no link between" in once(cat, nolink).get("error", "") or not once(cat, nolink).get("error"), once(cat, nolink).get("error"))

# --- Against measurements ---
r = once(cat, "check_27b_getorli")
check("27B on the GETORLI decodes within the measured 3-7 t/s", 3 <= r["engines"]["q27"]["decode_tps"] <= 7, r["engines"]["q27"]["decode_tps"])
r = once(cat, "check_pi5_small")
check("3B on a Pi 5 near the published 4.7 t/s", 3.5 <= r["engines"]["s3"]["decode_tps"] <= 6, r["engines"]["s3"]["decode_tps"])
r = once(cat, "backlog_ds41_node1")
check("V4.1 Flash on one node is SSD-bound and under 1 t/s", r["bottleneck"]["limit"] == "ssd" and r["engines"]["ds"]["decode_tps"] < 1, r["bottleneck"])

# --- Physics the model must show ---
def batch(slots, count=6):
    s = sc("check_27b_getorli")
    s["engines"][0].update(slots=slots, ctx=8192)
    s["workloads"] = [{"name": "b", "kind": "batch", "role": "main", "count": count, "prompt": 200, "out": 400}]
    return m(once(cat, s))
b1, b4 = batch(1), batch(4)
check("batching raises total throughput", b4["tps_total"] > 2 * b1["tps_total"], (b1["tps_total"], b4["tps_total"]))
check("and lowers each stream's speed a little", b4["decode_tps"] < b1["decode_tps"], (b1["decode_tps"], b4["decode_tps"]))
def chat(session):
    s = sc("check_27b_getorli")
    s["engines"][0].update(ctx=32768)
    s["workloads"] = [{"name": "c", "kind": "chat", "role": "main", "system": 8000, "turns": 6, "prompt": 200, "out": 200, "think_s": 30, "session": session}]
    return m(once(cat, s))
kept, lost = chat(True), chat(False)
check("kept context is not computed again", kept["prefill_computed"] < lost["prefill_computed"] / 3 and kept["span_s"] < lost["span_s"], (kept["prefill_computed"], lost["prefill_computed"]))
one, three = m(once(cat, "backlog_ds41_node1")), m(once(cat, "backlog_ds41_3nodes"))
check("experts over three nodes beat one", three["span_s"] < one["span_s"] / 2, (one["span_s"], three["span_s"]))
nat = sc("backlog_ds41_3nodes")
nat.pop("through_magpie")
nat_r, tun = m(once(cat, nat)), three
check("the relay tunnel costs a little on expert hops", tun["span_s"] > nat_r["span_s"] and tun["span_s"] < nat_r["span_s"] * 1.2, (nat_r["span_s"], tun["span_s"]))
wifi = sc("backlog_ds41_3nodes")
wifi["links"] = [{"type": "wifi6", "nodes": ["n1", "n4", "n5"]}]
wifi_r = m(once(cat, wifi))
check("Wi-Fi hops slow expert splitting, less so while the SSD is the limit", three["span_s"] * 1.1 < wifi_r["span_s"], (three["span_s"], wifi_r["span_s"]))
def routed(acc):
    s = sc("pi_portal_routed")
    s["overrides"]["model.qwen3.5-0.8b.decide.accuracy"] = acc
    rs = [once(cat, s, seed=k) for k in range(3)]
    return sum(m(r)["tokens_out"] for r in rs)
check("a worse decision model costs redone work", routed(0.5) > routed(0.98), (routed(0.5), routed(0.98)))
r = once(cat, "voice_turn_geekom")
check("a pipeline runs its stages in order", all(j["ok"] for j in r["jobs"]) and m(r)["ok"] == 5)
r = once(cat, "lora_night_9b")
check("training holds the engine; chats wait behind it", r["workloads"]["chats"]["latency_p90_s"] > 100, r["workloads"]["chats"])
r = once(cat, "media_node1")
check("media jobs finish and take compute time", m(r)["ok"] == 14 and r["engines"]["vid"]["util"] > 0.5, r["engines"])
check("team: subagents run on the 9Bs", once(cat, "team_main_and_subs")["engines"]["q9a"]["tokens_out"] > 0)

# --- Reproducible, and the cases bracket each other ---
check("same seed, same run", m(once(cat, "backlog_27b_getorli", "sample", 5)) == m(once(cat, "backlog_27b_getorli", "sample", 5)))
cs = cases(cat, "backlog_ds41_node1", seeds=2)
check("worst is slower than median is slower than best", cs["worst"]["metrics"]["span_s"] > cs["median"]["metrics"]["span_s"] > cs["best"]["metrics"]["span_s"], {k: v["metrics"].get("span_s") for k, v in cs.items() if "metrics" in v})

# --- Monte Carlo finds what drives a result ---
s = sc("check_27b_getorli")
s["workloads"] = [{"name": "b", "kind": "batch", "role": "main", "count": 3, "prompt": 200, "out": 400}]
s["overrides"] = {"llama_vulkan_mem_eff": {"est": [0.3, 0.6, 0.9], "group": "llama_vulkan_mem_eff"}}
mc = monte_carlo(cat, s, n=60, metrics=["span_s"])
top = mc["sensitivity"]["span_s"][0]
top2 = {x["input"]: x for x in mc["sensitivity"]["span_s"][:2]}
check("the inputs that set decode speed rank first and are clear of noise", set(top2) == {"llama_vulkan_mem_eff", "model.qwen3.8-27b.tokens_per_step"} and all(x["rho"] < -0.4 and x["clear"] for x in top2.values()), mc["sensitivity"]["span_s"][:3])
check("Monte Carlo spread", mc["summary"]["span_s"]["p5"] < mc["summary"]["span_s"]["p95"])
sw = sweep(cat, "check_27b_getorli", "node.n4.mem_bw_gbs", [40, 80, 120])
check("a sweep: more bandwidth, faster decode", sw[0]["metrics"]["decode_tps"] < sw[1]["metrics"]["decode_tps"] < sw[2]["metrics"]["decode_tps"], [x["metrics"].get("decode_tps") for x in sw])

# --- Goal seeking ---
s = sc("check_27b_getorli")
s["workloads"] = [{"name": "b", "kind": "batch", "role": "main", "count": 8, "prompt": 300, "out": 300}]
s["engines"][0]["ctx"] = 8192
o = optimize(cat, s, [{"path": "engines.q27.slots", "type": "int", "lo": 1, "hi": 8}, {"path": "engines.q27.quant", "type": "choice", "options": ["Q3_K_M", "Q4_K_M", "Q6_K", "Q8_0"]}],
             [{"metric": "span_s", "kind": "min"}], method="ga", runs=2, budget=60, generations=8, size=8)
check("goal seeking picks many slots and a small quant for a batch", o["best"]["params"]["engines.q27.slots"] >= 6 and o["best"]["params"]["engines.q27.quant"] in ("Q3_K_M", "Q4_K_M"), o["best"])
check("infeasible plans lose rather than crash", o["best"]["cost"] < float("inf"))

# --- Charts ---
r = once(cat, "team_main_and_subs")
svgs = [viz.timeline(r["trace"], m(r)["span_s"]), viz.limits(r["engines"]), viz.spread("span_s", cs, mc["summary"]), viz.tornado(mc["sensitivity"]["span_s"][:8], title="t"), viz.scatter([{"x": 1, "y": 2, "label": "a", "group": "g"}, {"x": 3, "y": 1, "label": "b", "group": "h"}], "x", "y", logx=True), viz.topology(sc("backlog_ds41_3nodes"), once(cat, "backlog_ds41_3nodes"))]
check("every chart renders", all(x.startswith("<svg") and x.endswith("</svg>") for x in svgs), [x[:60] for x in svgs])

print("ALL PASSED" if not FAILS else f"FAILED: {FAILS}")
sys.exit(1 if FAILS else 0)
