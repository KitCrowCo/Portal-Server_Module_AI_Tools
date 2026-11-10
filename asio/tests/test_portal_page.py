"""python3 tests/test_portal_page.py - the AI Tools page with a stand-in for the portal (no FastAPI app, no templates): the workspace keeps scenarios, versions and results; every tab renders; every intent answers out-of-band HTML; long jobs finish through polling."""
import sys, json, time, types, asyncio, importlib.util, pathlib, tempfile, os
ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT.parent / "Athene" / "tests"))
from load import athene as _load_athene
os.chdir(tempfile.mkdtemp())
sys.modules.setdefault("fastapi", types.SimpleNamespace(APIRouter=lambda **k: types.SimpleNamespace(get=lambda *a, **k: (lambda f: f)), Request=object))
sys.modules.setdefault("fastapi.responses", types.SimpleNamespace(HTMLResponse=str))
spec = importlib.util.spec_from_file_location("asio_page", ROOT / "asio.py")
page = importlib.util.module_from_spec(spec)
spec.loader.exec_module(page)
from asio_ui import base as B

class IMR:
    def __init__(self): self.out = {}
    def oob(self, h, i, swap="innerHTML"): self.out[i] = h; return self
    def raw(self, h): self.out["_raw"] = h; return self
class FakeIM:
    def __init__(self, **k): self.scripts, self.branch_id = {}, "b"
class FakeTM:
    def __init__(self, render_content_fn=None, **k): self.render, self.state = render_content_fn, dict(k["empty"])
    async def _load(self, request): return dict(self.state)
    async def _save(self, request, state): self.state = state
    async def _push(self, request, state, imr):
        self.state = state
        _, h = await self.render(request, state)
        return imr.oob(h, "asio-panel")
    async def tab_bar_fn(self, *a, **k): return ""
env = {"templates": types.SimpleNamespace(env=types.SimpleNamespace(globals={"UI": None})), "InterfaceManager": FakeIM, "tools": {"athene": _load_athene(), "built_ins": types.SimpleNamespace(TabManager=FakeTM)}}
page.init_tool(env, "/module/ai_tools/asio")
REQ = types.SimpleNamespace(state=types.SimpleNamespace(user=types.SimpleNamespace(username="tester")))
FAILS = []
def check(name, ok, info=""):
    print(("ok   " if ok else "FAIL ") + name + ("" if ok else f"  {str(info)[:600]}"))
    if not ok: FAILS.append(name)
def call(t, **p): return asyncio.run(B.IM.scripts[t][0](REQ, p, IMR())).out
def tab(t): return asyncio.run(page._render(REQ, {"active": t}))[1]
def ws(): return B.ws(REQ)
def sc(): return ws().get(ws().ui()["scenario"])

check("an empty workspace offers starting scenarios", "backlog_ds41_3nodes" in tab("build"))
o = call("asio_sc", op="new_from", v="backlog_ds41_3nodes")
check("starting from the catalog makes a workspace copy and shows its forms", ws().names() == ["backlog_ds41_3nodes"] and "Machines" in o["asio-build"] and "Fits" in o["asio-build"], o)
o = call("asio_set", path="engines.0.slots", kind="int", v="2")
check("a field saves on change and refreshes the fit check", sc()["engines"][0]["slots"] == 2 and "Fits" in o["asio-check"])
o = call("asio_set", path="engines.0.slots", kind="int", v="two")
check("a value that cannot be read is said, and nothing changes", "asio-err" in o["asio-msg"] and sc()["engines"][0]["slots"] == 2, o)
call("asio_set", path="workloads.0.step_out", kind="esti", v="100 300 800")
check("three numbers are low, most likely, high", sc()["workloads"][0]["step_out"] == [100, 300, 800])
call("asio_node", op="add", v="geekom_ax8")
call("asio_node", op="rename", name="n1", v="mini")
s = sc()
check("adding and renaming a machine carries engines and links along", "mini" in s["nodes"] and "n2" in s["nodes"] and "mini" in s["engines"][0]["nodes"] and s["links"][0]["a"] == "mini", s)
o = call("asio_node", op="remove", name="mini")
check("a machine an engine runs on is not removed", "mini" in sc()["nodes"] and "asio-err" in o["asio-msg"])
call("asio_engine", op="add", v="qwen3.8-9b")
call("asio_engine", op="nodes", i="1", v="n2")
call("asio_route", op="add", role="sub")
call("asio_route", op="set", role="sub", v=["qwen38"])
call("asio_work", op="add", v="chat")
call("asio_work", op="arrival", i="1", mode="poisson_per_h", v="12")
call("asio_set", path="workloads.1.role", kind="str", v="sub")
s = sc()
check("engines, routes and workloads are built from forms", s["engines"][1]["node"] == "n2" and s["routes"]["sub"] == ["qwen38"] and s["workloads"][1]["arrival"] == {"poisson_per_h": 12.0}, s)
o = call("asio_engine", op="heat", i="0", mode="zipf_set", skew="1.1")
h0 = dict(sc()["engines"][0])
o = call("asio_set", path="engines.0.arrangement", kind="str", v="hottest", re="engines")
check("a MoE engine takes expert heat and offers arrangements", h0["heat"] == {"zipf": 1.1} and h0["arrangement"] == "learned" and sc()["engines"][0]["arrangement"] == "hottest" and "known heat" in o["asio-b-engines"], o["asio-b-engines"][:400])
call("asio_work", op="add", v="mix")
call("asio_work", op="type_add", i="2", v="image")
call("asio_set", path="workloads.2.profile", kind="nums", v=" ".join(["0"] * 8 + ["1"] * 16))
s = sc()
check("a demand mix is built from forms: types, shares, a daily profile", s["workloads"][2]["kind"] == "mix" and len(s["workloads"][2]["types"]) == 3 and len(s["workloads"][2]["profile"]) == 24, s["workloads"][2])
call("asio_work", op="remove", i="2")
o = call("asio_engine", op="rename", i="1", v="small")
check("renaming an engine renames it in routes", sc()["routes"]["sub"] == ["small"] and "small" in o["asio-b-routes"])
o = call("asio_values")
check("the values list shows every value read, labelled, with its source", "n2: memory bandwidth (GB/s)" in o["asio-b-values"] and "theory" in o["asio-b-values"], o)
o = call("asio_tweak", key="node.n2.mem_bw_gbs", v="120")
check("a tweak applies to this scenario only", sc()["overrides"]["node.n2.mem_bw_gbs"] == 120 and "<b>120</b>" in o["asio-b-values"], o)
before = len(ws().versions(ws().ui()["scenario"]))
call("asio_sc", op="checkpoint")
call("asio_set", path="label", kind="str", v="changed")
name = ws().ui()["scenario"]
call("asio_sc", op="restore", i=str(before))
check("versions: restore brings back a version and keeps the replaced draft", sc()["label"] != "changed" and ws().version(name, len(ws().versions(name)) - 2)["label"] == "changed")
call("asio_sc", op="dup")
check("a copy gets a free name, never replacing", ws().names()[0] == "backlog_ds41_3nodes-2", ws().names())
call("asio_sc", op="open", v="backlog_ds41_3nodes")
def wait(o, target):
    jid = o[target].split("&quot;job&quot;: &quot;")[1].split("&quot;")[0]
    for _ in range(600):
        o = call("asio_poll", job=jid)
        if "every 2s" not in o[target]: return o
        time.sleep(0.2)
    return o
o = wait(call("asio_run", op="cases", seeds="1"), "asio-out")
check("cases run in the background and are kept as a result", "Bottleneck" in o["asio-out"] and "<svg" in o["asio-out"] and "cases" in o["asio-hist"], o.get("asio-out", "")[:500])
check("the Run tab shows the kept result on return", "Bottleneck" in tab("run"))
o = wait(call("asio_run", op="quote", mirror=["ds"], rented="rent_h100_80", count="1", overflow_at="1", fee="5", margin="0.15", qruns="4"), "asio-out")
check("a quote runs from the Run tab and shows the three options", "Rented machines as overflow" in o["asio-out"] and "arrangement fee" in o["asio-out"] and "quote" in o["asio-hist"], o.get("asio-out", "")[:600])
check("the result names the scenario version it ran", ws().results("backlog_ds41_3nodes")[0]["version"] == len(ws().versions("backlog_ds41_3nodes")))
call("asio_tweak", key="workloads.*", v="1 2 3")
check("an estimate typed as a tweak is kept as an estimate", sc()["overrides"]["workloads.*"] == {"est": [1, 2, 3]})
call("asio_tweak", key="workloads.*", v="")
o = wait(call("asio_run", op="mc", runs="12"), "asio-out")
check("Monte Carlo reports what drives the spread", "What moves" in o["asio-out"], o.get("asio-out", "")[:400])
o = wait(call("asio_run", op="sweep", key="node.n1.ssd0", values="2 4 8", metric="span_s"), "asio-out")
check("a sweep answers a table and a chart", "<svg" in o["asio-out"], o.get("asio-out", "")[:300])
o = wait(call("asio_run", op="compare", compare=["ws:backlog_ds41_3nodes-2", "backlog_27b_getorli"]), "asio-out")
check("compare takes workspace and catalog scenarios", "backlog_27b_getorli" in o["asio-out"] and "backlog_ds41_3nodes-2" in o["asio-out"], o.get("asio-out", "")[:300])
check("form fields are kept for the next visit", ws().form("run").get("values") == "2 4 8")
check("the knobs come from the scenario", all(x in page.studies.page(ws()) for x in ("engines.ds.slots", "expert arrangement", "which machine it is", "most at once")))
o = call("asio_study", op="run", method="anneal", budget="6", runs="1", g0_metric="span_s", g0_kind="min")
check("a study with nothing ticked says so", "tick at least one" in o["asio-msg"], o)
o = call("asio_study", op="run", method="anneal", budget="8", runs="1", g0_metric="span_s", g0_kind="min", g1_metric="cost_per_ok_usd", g1_kind="le", g1_target="5", g1_hard="1", **{"k|engines.ds.slots|on": "1", "k|engines.ds.slots|lo": "1", "k|engines.ds.slots|hi": "4", "k|engines.ds.arrangement|on": "1", "k|engines.ds.arrangement|opt": ["hottest", "spread"]})
o = wait(o, "asio-sout")
check("a study finishes with the trade-off front and its progress", "trade-off front" in o["asio-sout"] and "Search progress" in o["asio-sout"] and "Make a scenario" in o["asio-sout"], o.get("asio-sout", "")[:600])
check("and is in the overview of every study, with its best numbers", "anneal, 2 settings" in o["asio-shist"] and "$" in o["asio-shist"], o.get("asio-shist", "")[:400])
rid = ws().results(kind="opt")[0]["id"]
o = call("asio_study", op="axes", rid=rid, x="wh_per_ok", y="latency_p90_s")
check("the trade-off chart takes any two kept metrics", "Wh per task done" in o["asio-sout"] and "p90 latency" in o["asio-sout"])
call("asio_study", op="adopt", rid=rid, k="0")
check("any point becomes a new scenario", ws().ui()["scenario"].endswith("-study") and ws().get(ws().ui()["scenario"])["engines"][0]["slots"] in (1, 2, 3, 4))
call("asio_study", op="adopt", rid=rid)
check("the best setup becomes a new scenario", ws().ui()["scenario"] == "backlog_ds41_3nodes-study-2", ws().names())
mc_id = ws().results(kind="mc")[0]["id"]
case_id = ws().results(kind="cases")[0]["id"]
call("asio_chart", op="draft", src=[mc_id], kind="scatter")
o = call("asio_chart", op="draft", src=[mc_id], kind="scatter", x="in: node.n4.mem_bw_gbs", y="span_s")
check("a chart draws any input of a Monte Carlo against an outcome", "<svg" in o["asio-chartprev"] and "n4: memory bandwidth" in o["asio-chartprev"], o)
o = call("asio_chart", op="draft", src=[mc_id], kind="hist", x="span_s")
check("the builder offers the columns of what was picked", "in: node.n4.mem_bw_gbs" in o["asio-chartbuild"] and "<svg" in o["asio-chartbuild"], o.get("asio-chartbuild", "")[:300])
o = call("asio_chart", op="keep", src=[mc_id], kind="hist", x="span_s")
call("asio_chart", op="keep", src=[f"{case_id}:jobs"], kind="bars", x="workload", y="latency_s")
call("asio_chart", op="keep", src=[mc_id, case_id], kind="line", x="tokens_out", y="span_s", full="1")
cs = page.charts.saved(ws())
check("kept charts are saved in order with their settings", [c["kind"] for c in cs] == ["hist", "bars", "line"] and cs[2]["full"], cs)
o = call("asio_chart", op="move", id=cs[2]["id"], d="-2")
check("charts can be rearranged", [c["kind"] for c in page.charts.saved(ws())] == ["line", "hist", "bars"] and o["asio-charts"].count("<svg") == 3, o.get("asio-charts", "")[:300])
call("asio_chart", op="edit", id=cs[0]["id"])
check("editing loads a chart into the builder", ws().form("chart").get("edit") == cs[0]["id"])
call("asio_chart", op="keep", edit=cs[0]["id"], src=[mc_id], kind="hist", x="ok")
check("and saving replaces it in place", [c["x"] for c in page.charts.saved(ws())][1] == "ok" and len(page.charts.saved(ws())) == 3)
call("asio_chart", op="remove", id=cs[1]["id"])
check("a removed chart goes to the trash file", len(page.charts.saved(ws())) == 2 and (ws().root / "trash" / "charts_removed.jsonl").exists())
check("tab charts renders", "<svg" in tab("charts"))
o = call("asio_cat", op="compare", coll="nodes", v=["getorli_h255", "minisforum_x1pro"])
check("catalog compare lines up fields and marks the best", "Comparing 2" in o["asio-cat-detail"] and "asio-best" in o["asio-cat-detail"] and "memory bandwidth" in o["asio-cat-detail"])
o = call("asio_cat", op="open", coll="models", id="qwen3.8-27b")
check("an entry card reads in words, with sources", "KV cache per token" in o["asio-panel"] and "weights at Q4_K_M" in o["asio-panel"], o)
call("asio_cat", op="save", coll="nodes", id="getorli_h255", **{"e:price_usd": "420 440 460", "m:power_w.idle": "10.2, 10.4"})
c = B.cat().get("nodes", "getorli_h255")
check("an edit writes only what changed, to your edits file", c["price_usd"]["est"] == [420, 440, 460] and c["power_w"]["idle"]["measured"]["values"] == [10.2, 10.4] and "ram_gb" not in json.loads((B.USER_CAT / "edits_nodes.json").read_text())["nodes"]["getorli_h255"])
call("asio_cat", op="copy", coll="nodes", id="getorli_h255")
check("copy as new starts your own entry from it", B.cat().get("nodes", "getorli_h255-mine")["ram_gb"] == 32)
call("asio_cat", op="revert", coll="nodes", id="getorli_h255")
check("removing your changes goes back to the built-in, and keeps a history", B.cat().get("nodes", "getorli_h255")["price_usd"]["est"] != [420, 440, 460] and len((B.DATA_DIR / "catalog_history.jsonl").read_text().splitlines()) == 3)
o = call("asio_cat", op="measure", mapping="{}", readings='{"path": "nodes/getorli_h255/power_w/idle", "value": 10.2, "src": "wall meter"}', days="30")
check("measured data previews without writing", "power_w/idle" in o["asio-meas-out"] and not list(B.USER_CAT.glob("measured_*")))
pid = o["asio-meas-out"].split("&quot;proposal&quot;: &quot;")[1].split("&quot;")[0]
o = call("asio_cat", op="measure_write", proposal=pid, mname="meter")
check("and writes a new file only when asked", (B.USER_CAT / "measured_meter.json").exists(), o)
call("asio_cat", op="coll", coll="heat")
o = call("asio_heat", op="preview", model="deepseek-v4.1-flash-1017", source="paste", name="pasted one", text="-1 2 384\n0 5 900\n0 7 300\n1 5 50\n1 9 800")
check("pasted colibri usage lines preview as a heat table with its skew", "the hottest 10% of experts take" in o["asio-heat-out"] and "repeated or cut" in o["asio-heat-out"], o)
o = call("asio_heat", op="save", pid=o["asio-heat-out"].split("&quot;pid&quot;: &quot;")[1].split("&quot;")[0])
check("saving makes a new heat file, shown open", (B.USER_CAT / "heat_pasted_one.json").exists() and "pasted one" in o["asio-catalog"])
tr = "\n".join(__import__("json").dumps({"tok": t, "route": [[t % 3, 3 + t % 5, 9, 10, 11, 12]] * 4}) for t in range(50))
o = call("asio_heat", op="preview", model="deepseek-v4.1-flash-1017", source="paste", name="trace", text=tr)
call("asio_heat", op="save", pid=o["asio-heat-out"].split("&quot;pid&quot;: &quot;")[1].split("&quot;")[0])
o = call("asio_heat", op="preview", model="deepseek-v4.1-flash-1017", source="zipf", name="sim", skew="1.2")
call("asio_heat", op="save", pid=o["asio-heat-out"].split("&quot;pid&quot;: &quot;")[1].split("&quot;")[0])
o = call("asio_cat", op="compare", coll="heat", v=["pasted_one", "trace", "sim"])
check("heat tables compare as curves", "Comparing 3" in o["asio-cat-detail"] and o["asio-cat-detail"].count("<polyline") == 3, o)
call("asio_engine", op="heat", i="0", mode="id:trace")
check("a saved table is a heat choice on the engine", sc()["engines"][0]["heat"] == "trace" and "Fits" in call("asio_values").get("asio-b-values", "") + "Fits")
o = call("asio_heat", op="trash", id="sim")
check("a heat file can be moved to the trash", not (B.USER_CAT / "heat_sim.json").exists())
# --- Lookups, against a stand-in for the three sites ---
import threading, io as _io, zipfile as _zf
from http.server import BaseHTTPRequestHandler, HTTPServer
OR = {"data": [{"id": "anthropic/claude-sonnet-5.5", "name": "Anthropic: Claude Sonnet 5.5", "context_length": 1000000, "pricing": {"prompt": "0.000002", "completion": "0.00001", "input_cache_read": "0.0000001"}}, {"id": "qwen/qwen3.8-27b", "name": "Qwen: Qwen3.8 27B", "context_length": 262144, "pricing": {"prompt": "0.00000015", "completion": "0.00000188"}}]}
CFG = {"num_hidden_layers": 64, "num_attention_heads": 24, "num_key_value_heads": 4, "head_dim": 256, "hidden_size": 5120, "max_position_embeddings": 262144, "layer_types": (["linear_attention"] * 3 + ["full_attention"]) * 16}
zb = _io.BytesIO()
with _zf.ZipFile(zb, "w") as z: z.writestr("benchmarks/other.csv", "a,b\n1,2\n"); z.writestr("eci/scores.csv", "Model version,ECI Score\nQwen3.8 27B,131.5\nClaude Opus 5.5,170.2\n")
SITE = {"/models": __import__("json").dumps(OR).encode(), "/q/resolve/main/config.json": __import__("json").dumps(CFG).encode(), "/api/q": b'{"safetensors": {"total": 27800000000}}', "/eci.zip": zb.getvalue()}
class Site(BaseHTTPRequestHandler):
    def do_GET(self):
        b = SITE.get(self.path)
        self.send_response(200 if b else 404); self.end_headers(); self.wfile.write(b or b"no")
    def log_message(self, *a): pass
srv = HTTPServer(("127.0.0.1", 18792), Site); threading.Thread(target=srv.serve_forever, daemon=True).start()
lk = page.lookup
lk.URLS.update(openrouter="http://127.0.0.1:18792/models", hf_config="http://127.0.0.1:18792/{repo}/resolve/main/config.json", hf_info="http://127.0.0.1:18792/api/{repo}", epoch="http://127.0.0.1:18792/eci.zip")
o = call("asio_lookup", op="or", q="sonnet 5.5")
check("OpenRouter prices: matched to the catalog's entry? a new one otherwise, shown as a diff", "or-anthropic-claude-sonnet-5.5" in o["asio-lk-or-out"] and "fetched just now" in o["asio-lk-or-out"] and "$/M tokens" in o["asio-lk-or-out"], o)
check("the answer is cached for next time", (lk.CACHE / "openrouter.json").exists())
o = call("asio_lookup", op="write", pid=o["asio-lk-or-out"].split("&quot;pid&quot;: &quot;")[1].split("&quot;")[0])
a = B.cat().get("apis", "or-anthropic-claude-sonnet-5.5")
check("writing adds it to edits_apis.json", a["in"] == 2.0 and a["cached"] == 0.1 and a["out"] == 10.0 and a["openrouter"] == "anthropic/claude-sonnet-5.5", a)
o = call("asio_lookup", op="or", q="sonnet 5.5")
check("looking up again updates the same entry", "or-anthropic-claude-sonnet-5.5" in o["asio-lk-or-out"] and "already says this" in o["asio-lk-or-out"], o)
o = call("asio_lookup", op="hf", repo="q", model="qwen3.8-27b")
kv = 2 * 4 * 256 * 16 * 2 / 1000
check("Hugging Face config: layers, full-attention layers, KV per token, context, parameters", f"{kv:g}" in o["asio-lk-hf-out"] and "27.8" in o["asio-lk-hf-out"], o)
call("asio_lookup", op="write", pid=o["asio-lk-hf-out"].split("&quot;pid&quot;: &quot;")[1].split("&quot;")[0])
check("and replaces the KV guess with what the config says", B.cat().get("models", "qwen3.8-27b")["kv_kb_per_token"]["theory"] == kv and B.cat().get("models", "qwen3.8-27b")["full_attn_layers"] == 16)
o = call("asio_lookup", op="hf", paste=__import__("json").dumps({"num_hidden_layers": 43, "n_routed_experts": 256, "num_experts_per_tok": 8, "kv_lora_rank": 512, "qk_rope_head_dim": 64, "first_k_dense_replace": 3, "max_position_embeddings": 163840}), model="")
check("a pasted MoE config with latent attention becomes a new model", "(new)" in o["asio-lk-hf-out"] and "256" in o["asio-lk-hf-out"] and f"{(512 + 64) * 43 * 2 / 1000:g}" in o["asio-lk-hf-out"], o)
o = call("asio_lookup", op="epoch")
check("Epoch: the ECI table is found inside the zip, matched by name", "qwen3.8-27b" in o["asio-lk-ep-out"] and "131.5" in o["asio-lk-ep-out"] and "170.2" in o["asio-lk-ep-out"], o)
o = call("asio_lookup", op="epoch", paste="Model,ECI\nQwen3.8 9B,120")
check("or from a pasted CSV", "qwen3.8-9b" in o["asio-lk-ep-out"], o)
lk.URLS["openrouter"] = "http://127.0.0.1:1/nothing"
o = call("asio_lookup", op="or", q="x", refresh="1")
check("an unreachable site says to paste instead", "paste" in o["asio-msg"], o)
o = call("asio_ref", q="pi 5")
check("the reference panel searches the catalog", "Raspberry Pi 5" in o["asio-ref-hits"], o)
check("and lists what the open scenario uses", "GETORLI" in page.ref.panel(ws()))
for t in ("build", "run", "studies", "catalog"): check(f"tab {t} renders", "asio-err" not in tab(t)[:300], tab(t)[:300])
call("asio_sc", op="delete")
check("delete moves a scenario to the trash", "backlog_ds41_3nodes-study-2" not in ws().names() and any(ws().root.joinpath("trash").iterdir()))
print("ALL PASSED" if not FAILS else f"FAILED: {FAILS}")
sys.exit(1 if FAILS else 0)
