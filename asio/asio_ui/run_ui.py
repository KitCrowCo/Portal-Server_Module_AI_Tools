"""asio_ui/run_ui.py - the Run tab, and the background jobs Studies uses too: runs of the open scenario, every result kept in the workspace and shown again on return.
Run: the named cases, Monte Carlo with what drives the spread, a sweep of one value, scenarios side by side.
A run checkpoints the scenario first, so each result names the version it ran."""
import json, threading, traceback, time, uuid, copy
from asio_ui import base as B, fields as F

METRICS = [("span_s", "time to finish", "t"), ("ok", "tasks done", "n"), ("success_rate", "success rate", "%"), ("decode_tps", "decode t/s per stream", "n"), ("tps_total", "total t/s", "n"), ("prefill_tps", "prefill t/s", "n"),
           ("latency_p90_s", "p90 latency", "t"), ("reuse_share", "context reused", "%"), ("energy_wh", "energy Wh", "n"), ("cost_usd", "cost $", "$"), ("cost_per_ok_usd", "$ per task done", "$"), ("wh_per_ok", "Wh per task done", "n")]
TIERS = [("auto", "measured, else estimate, else theory"), ("estimate", "estimates only"), ("theory", "theory only")]
JOBS: dict = {}
KIND_WORDS = {"cases": "cases", "mc": "Monte Carlo", "sweep": "sweep", "compare": "comparison", "opt": "study", "quote": "quote"}
QUOTE_WORDS = {"local": "Your machines only", "overflow": "Rented machines as overflow", "planned": "Rented machines from the start"}

# --- Result views ---

def bottleneck(r: dict) -> str:
    b = r.get("bottleneck")
    if not b: return ""
    e = r["engines"][b["engine"]]
    words = {"mem": "memory bandwidth (weights and context read each step)", "compute": "compute (prefill, batched or speculative steps)", "ssd": "the SSD (experts or weights streamed from disk)", "hop": "links between machines", "overhead": "fixed per-step costs"}
    return B.note(f"""<b>Bottleneck:</b> {b['engine']} ({e['model']} on {', '.join(e['nodes'])}) was busy {b['util']:.0%} of the run; work queued for it {B.fmt(b['queued_s'], 't')} in total; most of its time was limited by {words.get(b['limit'], b['limit'])}.""")

def run_detail(r: dict, sc: dict) -> str:
    _, viz, _ = B.libs()
    m = r["metrics"]
    eng = B.tbl(["engine", "model", "machines", "busy", "decode t/s", "prefill t/s", "tokens out", "queued", "held back by"], [[k, e["model"], ", ".join(e["nodes"]), f"""{e['util']:.0%}""", B.fmt(e.get("decode_tps")), B.fmt(e.get("prefill_tps")), B.fmt(e["tokens_out"]), B.fmt(e["wait_s"], "t"), ", ".join(f"{a} {b:.0%}" for a, b in sorted(e["limits"].items(), key=lambda t: -t[1]))] for k, e in r["engines"].items()])
    wl = B.tbl(["workload", "jobs", "done", "p50", "p90", "outcomes"], [[k, w["jobs"], w["ok"], B.fmt(w["latency_p50_s"], "t"), B.fmt(w["latency_p90_s"], "t"), ", ".join(f"{a} {b}" for a, b in w["outcomes"].items())] for k, w in r["workloads"].items()])
    nodes = B.tbl(["machine", "busy", "Wh", "RAM held", "holding"], [[k, f"""{n['util']:.0%}""", B.fmt(n["energy_wh"]), f"""{n['ram_used_gb']:.1f} of {n['ram_usable_gb']:.1f} GB""", "; ".join(f"{a} {b} GB" for a, b in n["holding"])] for k, n in r["nodes"].items()])
    apis = B.tbl(["the same tokens on", "cost"], [[k, f"${v:.4g}"] for k, v in sorted(r["api_cost"].items(), key=lambda t: t[1])] + [["<b>this setup</b>", f"""<b>${m['cost_usd']:.4g}</b>"""]])
    return f"""{bottleneck(r)}<div class="asio-fig">{viz.timeline(r['trace'], m['span_s'])}</div><div class="asio-fig">{viz.limits(r['engines'])}</div><div class="asio-fig">{viz.topology(sc, r)}</div>
        <div class="asio-h">Engines</div>{eng}<div class="asio-h">Workloads</div>{wl}<div class="asio-h">Machines</div>{nodes}<div class="asio-h">Against APIs</div><div class="asio-dim">API rows bill this run's tokens at each listed price, with every reused token at the cached rate (the API's best case). Dates and sources are in the catalog.</div>{apis}"""

def cases_html(cs: dict, sc: dict, mc: dict = None) -> str:
    _, viz, _ = B.libs()
    errs, ok = {c: v["error"] for c, v in cs.items() if v.get("error")}, {c: v for c, v in cs.items() if not v.get("error")}
    if not ok: return B.err("Every case failed: " + "; ".join(f"{c}: {e}" for c, e in errs.items()))
    head = ["metric"] + list(ok) + (["MC p5", "MC p50", "MC p95"] if mc else [])
    rows = [[lbl] + [B.fmt(v["metrics"].get(k), kind) for v in ok.values()] + ([B.fmt(mc[k][p], kind) for p in ("p5", "p50", "p95")] if mc and k in mc else ["-"] * 3 if mc else []) for k, lbl, kind in METRICS]
    figs = "".join(f"""<div class="asio-fig">{viz.spread(k, cs, mc, label=lbl)}</div>""" for k, lbl, _ in METRICS[:3] + METRICS[10:11])
    first = ok["median"]["first"] if "median" in ok else next(iter(ok.values()))["first"]
    return f"""{''.join(B.err(f'{c}: {e}') for c, e in errs.items())}{B.tbl(head, rows)}{figs}<div class="asio-h">Median case in detail</div>{run_detail(first, sc)}"""

def mc_html(res: dict, cs: dict = None) -> str:
    _, viz, _ = B.libs()
    s = res["summary"]
    rows = [[lbl, B.fmt(s[k]["p5"], kind), B.fmt(s[k]["p50"], kind), B.fmt(s[k]["p95"], kind), B.fmt(s[k]["mean"], kind)] for k, lbl, kind in METRICS if k in s]
    fails = B.note(f"""{s.get('_failed', 0)} of {res['n']} runs could not exist: {B.esc('; '.join(res['errors'][:3]))}""") if res["errors"] else ""
    tor = "".join(f"""<div class="asio-fig">{viz.tornado(v[:12], title=f'What moves {k}')}</div>""" for k, v in res["sensitivity"].items())
    figs = "".join(f"""<div class="asio-fig">{viz.spread(k, cs or {}, s, label=lbl)}</div>""" for k, lbl, _ in METRICS[:3] if k in s)
    return f"""{fails}{B.tbl(['metric', 'p5', 'p50', 'p95', 'mean'], rows)}{figs}<div class="asio-dim">Faded rows are within what chance gives with {res['n']} runs: run more to separate them.</div>{tor}"""

def sweep_html(d: dict) -> str:
    _, viz, _ = B.libs()
    key, metric, rows = d["key"], d["metric"], d["rows"]
    pts = [{"x": r["value"], "y": r["metrics"].get(metric), "label": str(r["value"]), "group": metric} for r in rows if isinstance(r["value"], (int, float)) and not r["error"]]
    t = B.tbl([B.esc(key)] + [l for _, l, _ in METRICS[:6]], [[B.esc(json.dumps(r["value"]))] + ([B.err(r["error"])] if r["error"] else [B.fmt(r["metrics"].get(k), kd) for k, _, kd in METRICS[:6]]) for r in rows])
    return f"""{t}<div class="asio-fig">{viz.scatter(pts, key, metric, title=f'{metric} as {key} changes')}</div>"""

def compare_html(rows: list) -> str:
    _, viz, _ = B.libs()
    c = B.cat()
    pts = [{"x": r["metrics"]["cost_per_ok_usd"], "y": r["metrics"]["span_s"] / max(r["metrics"]["ok"], 1), "label": r["id"], "group": "setups"} for r in rows if not r.get("error") and r["metrics"].get("ok")]
    eci = [{"x": a.get("out"), "y": a["eci"], "label": a["label"], "group": "APIs (output $/M)"} for a in c.data["apis"].values() if a.get("eci") and a.get("out")]
    for r in rows:
        if r.get("error"): continue
        e = c.get("models", next(iter(r["engines"].values()))["model"]).get("eci")
        if e and r["metrics"].get("cost_per_mtok_usd"): eci.append({"x": r["metrics"]["cost_per_mtok_usd"], "y": e, "label": r["id"], "group": "setups ($/M out)"})
    t = B.tbl(["scenario", "time", "done", "$ per task", "Wh per task", "decode t/s", "bottleneck"], [[B.esc(r["id"]), B.err(r["error"]) if r.get("error") else B.fmt(r["metrics"]["span_s"], "t"), "" if r.get("error") else f"""{r['metrics']['ok']}/{r['metrics']['jobs']}""", "" if r.get("error") else B.fmt(r["metrics"]["cost_per_ok_usd"], "$"), "" if r.get("error") else B.fmt(r["metrics"]["wh_per_ok"]), "" if r.get("error") else B.fmt(r["metrics"].get("decode_tps")), "" if r.get("error") or not r.get("bottleneck") else B.esc(f"""{r['bottleneck']['engine']}: {r['bottleneck']['limit']}""")] for r in rows])
    hint = "" if any(p["group"].startswith("setups") for p in eci) else '<div class="asio-dim">None of these setups\' models has an ECI in the catalog yet: add one to a model (Catalog, Edit) to place it on the capability chart.</div>'
    s2 = viz.scatter(eci, "$ per million output tokens", "Epoch Capabilities Index", logx=True, title="Capability against price") if eci else ""
    return f"""{t}<div class="asio-fig">{viz.scatter(pts, '$ per task done', 'time per task done (s)', logx=True, logy=True, title='Cost against time, per finished task')}</div>{hint}<div class="asio-fig">{s2}</div>"""

def quote_html(d: dict) -> str:
    """The three options side by side, as they would be shown to whoever the work is for: time, rented cost and its ceiling, the arrangement fee on its own line."""
    q, c = d["quote"], B.cat()
    o = q["options"]
    row = lambda lbl, f: [lbl] + [f(o[k]) for k in QUOTE_WORDS]
    rows = [row("time to finish, likely", lambda x: B.fmt(x["span_p50"], "t")), row("time to finish, 1 in 20 worst", lambda x: B.fmt(x["span_p95"], "t")), row("tasks done", lambda x: f"""{B.fmt(x['ok_p50'])} of {B.fmt(x['jobs'])}"""),
            row("rented machines, likely", lambda x: B.fmt(x["rent_p50"], "$")), row("rented machines, at most without asking", lambda x: B.fmt(x["ceiling"], "$") if x["ceiling"] else "-"), row("your machines (power, wear)", lambda x: B.fmt(x["own_p50"], "$")),
            row("arrangement fee", lambda x: B.fmt(x["fee"], "$") if x["fee"] else "-"), row("<b>total, likely</b>", lambda x: f"""<b>{B.fmt(x['total_p50'], '$')}</b>""")]
    errs = "".join(B.err(f"""{QUOTE_WORDS[k]}: {v.get('failed', '?')} of {v['n']} runs could not exist (left out of the numbers): {'; '.join(v['errors'][:2])}""") for k, v in o.items() if v["errors"])
    node = c.get("nodes", q["rented"]) if q["rented"] in c.data["nodes"] else {}
    note = f"""{q['count']} x {B.esc(node.get('label', q['rented']))} at {F.show((node.get('rented') or {}).get('usd_per_h'), '$/h')}, running copies of {B.esc(', '.join(q['mirror']))}. As overflow they start only when every local slot on their route is busy{'' if q['overflow_at'] == 1 else f" (load {q['overflow_at']:g})"} and are billed from then; planned, they are rented for the whole run. The ceiling is the 1-in-20 cost plus {q['margin']:.0%}: past it, the work should stop and ask. Rented prices and start times are planning ranges until measured; storage and data transfer are extra."""
    return f"""{errs}{B.tbl([""] + list(QUOTE_WORDS.values()), rows)}{B.note(note)}<div class="asio-dim">From {o['local']['n']} Monte Carlo runs per option. Privacy: work sent to rented machines leaves your machines; keep private domains local.</div>"""

def result_html(r: dict) -> str:
    d, k = r["data"], r["kind"]
    top = f"""<div class="row asio-rhead"><b class="grow">{B.esc(KIND_WORDS.get(k, k))} of {B.esc(r['scenario'])}{f" (version {r['version']})" if r.get('version') else ''}, {B.ago(r['t'])}</b>{B.btn("Move to trash", "asio_run", cls="asio-btn asio-danger", op="trash", rid=r['id'], confirm="Move this result to the workspace trash?")}</div>"""
    if d.get("error"): return top + B.err(d["error"])
    body = cases_html(d["cases"], d["sc"]) if k == "cases" else mc_html(d["mc"], d.get("cases")) if k == "mc" else sweep_html(d) if k == "sweep" else compare_html(d["rows"]) if k == "compare" else quote_html(d) if k == "quote" else B.err(f"no view for {k} here")
    return top + body

# --- Background jobs ---

def start(ws, scenario: str, kind: str, inputs: dict, fn) -> str:
    """Runs fn() in a thread; its dict (with "head", a one-line summary) is stored as a result. The job's state is set before the thread starts."""
    jid = uuid.uuid4().hex[:10]
    JOBS[jid] = {"kind": kind, "scenario": scenario, "status": "running", "t0": time.time(), "rid": None}
    def go():
        try: rid = ws.add_result(scenario, kind, inputs, fn())
        except Exception as e: rid = ws.add_result(scenario, kind, inputs, {"head": f"failed: {type(e).__name__}", "error": f"""{type(e).__name__}: {e}\n{traceback.format_exc()[-1500:]}"""})
        JOBS[jid].update(status="done", rid=rid)
    threading.Thread(target=go, daemon=True).start()
    return jid

def poll_html(jid: str) -> str:
    j = JOBS[jid]
    return f"""<div {B.act("asio_poll", trigger="every 2s", job=jid)}><div class="asio-note"><span class="spin-anim">[..]</span> {B.esc(KIND_WORDS.get(j['kind'], j['kind']))} of {B.esc(j['scenario'])} running for {time.time() - j['t0']:.0f}s. You can leave this tab; the result is kept.</div></div>"""

def history(ws, scenario: str = None, kind: str = None, open_id: str = None) -> str:
    rows = ws.results(scenario, kind)
    if not rows: return '<div class="asio-dim">No results yet.</div>'
    return "".join(f"""<div class="asio-item {'asio-on' if r['id'] == open_id else ''}" {B.act("asio_run", op="show", rid=r['id'])}><div class="grow"><b>{B.esc(KIND_WORDS.get(r['kind'], r['kind']))}</b> {B.esc(r.get('head', ''))}<div class="asio-dim">{B.esc(r['scenario'])}{f" v{r['version']}" if r.get('version') else ''}, {B.ago(r['t'])}</div></div></div>""" for r in rows)

# --- Run tab ---

def scenario_choices(ws) -> list: return [("Yours", [(f"ws:{n}", n) for n in ws.names()]), ("Catalog", [(i, i) for i in B.cat().ids("scenarios")])]

def run_page(ws) -> str:
    name, f = ws.ui().get("scenario"), ws.form("run")
    if not name or not ws.exists(name): return '<div class="asio-pad asio-dim">Open or start a scenario on the Build tab first.</div>'
    opt = lambda opts, v: "".join(f"""<option value="{B.esc(k)}" {'selected' if str(k) == str(v) else ''}>{B.esc(l)}</option>""" for k, l in opts)
    picked = f.get("compare") or []
    picked = [picked] if isinstance(picked, str) else picked
    cmp = "".join(f"""<optgroup label="{g}">{''.join(f'<option value="{B.esc(k)}" {"selected" if k in picked else ""}>{B.esc(l)}</option>' for k, l in items)}</optgroup>""" for g, items in scenario_choices(ws))
    form = f"""<form class="asio-card" id="asio-runform" onsubmit="return false" {B.act("asio_form", trigger="change", form="run")}><div class="row asio-wrap">
        {F.field("numbers from", f'<select class="asio-in asio-w14" name="tier">{opt(TIERS, f.get("tier", "auto"))}</select>')}{F.field("seeds per case", f'<input class="asio-in asio-w4" name="seeds" value="{B.esc(f.get("seeds", "4"))}">')}
        {B.btn("Run the cases", "asio_run", op="cases", include="#asio-runform")}{F.field("Monte Carlo runs", f'<input class="asio-in asio-w4" name="runs" value="{B.esc(f.get("runs", "64"))}">')}{B.btn("Monte Carlo and what drives it", "asio_run", op="mc", include="#asio-runform")}</div>
        <div class="row asio-wrap">{F.field("sweep a value (key or pattern)", f'<input class="asio-in asio-w14" name="key" value="{B.esc(f.get("key", ""))}" placeholder="node.n4.mem_bw_gbs" list="asio-keys">')}{F.field("over (numbers)", f'<input class="asio-in asio-w14" name="values" value="{B.esc(f.get("values", ""))}" placeholder="40 60 80 100">')}
        {F.field("plotting", f'<select class="asio-in asio-w10" name="metric">{opt([(k, l) for k, l, _ in METRICS], f.get("metric", "span_s"))}</select>')}{B.btn("Sweep", "asio_run", op="sweep", include="#asio-runform")}</div>
        <div class="row asio-wrap">{F.field("compare with (ctrl or shift to pick several)", f'<select class="asio-in asio-w20" name="compare" multiple size="4">{cmp}</select>')}{B.btn("Compare", "asio_run", op="compare", include="#asio-runform")}</div>
        <div class="row asio-wrap">{quote_fields(ws.get(name), f)}{B.btn("Quote with rented machines", "asio_run", op="quote", include="#asio-runform")}</div></form>"""
    keys = "".join(f'<option value="{B.esc(k)}">' for k in value_keys(ws.get(name)))
    open_id = ws.ui().get("run_open")
    shown = result_html(ws.result(open_id)) if open_id and (ws.rdir / f"{open_id}.json").exists() else '<div class="asio-dim">Run something; results stay here.</div>'
    return f"""<div class="asio-pad"><div class="asio-dim">Running <b>{B.esc(name)}</b>: {B.esc(ws.get(name).get('label', ''))}</div>{form}<datalist id="asio-keys">{keys}</datalist>
        <div class="asio-split"><div class="asio-list" id="asio-hist">{history(ws, name, "cases|mc|sweep|compare|quote", open_id)}</div><div class="asio-detail" id="asio-out">{shown}</div></div></div>"""

def quote_fields(sc: dict, f: dict) -> str:
    c = B.cat()
    opt = lambda opts, v: "".join(f"""<option value="{B.esc(k)}" {'selected' if str(k) in (v if isinstance(v, list) else [str(v)]) else ''}>{B.esc(l)}</option>""" for k, l in opts)
    rented = [(i, c.get("nodes", i)["label"]) for i in c.ids("nodes") if c.get("nodes", i).get("rented")]
    mirror = f.get("mirror") or [e["id"] for e in sc.get("engines", [])[:1]]
    return f"""{F.field("engines to copy onto rented machines", f'<select class="asio-in asio-w12" name="mirror" multiple size="3">{opt([(e["id"], e["id"]) for e in sc.get("engines", [])], mirror if isinstance(mirror, list) else [mirror])}</select>')}{F.field("rented machine", f'<select class="asio-in asio-w14" name="rented">{opt(rented, f.get("rented", "rent_rtx4090"))}</select>')}
        {F.field("how many", f'<input class="asio-in asio-w3" name="count" value="{B.esc(f.get("count", "1"))}">')}{F.field("overflow when local load reaches", f'<input class="asio-in asio-w4" name="overflow_at" value="{B.esc(f.get("overflow_at", "1"))}" title="1: every local slot busy; 2: a queue as long again">')}
        {F.field("arrangement fee $", f'<input class="asio-in asio-w4" name="fee" value="{B.esc(f.get("fee", "0"))}">')}{F.field("ceiling margin", f'<input class="asio-in asio-w4" name="margin" value="{B.esc(f.get("margin", "0.15"))}">')}{F.field("runs per option", f'<input class="asio-in asio-w4" name="qruns" value="{B.esc(f.get("qruns", "24"))}">')}"""

def value_keys(sc: dict) -> list:
    """Keys a sweep can change: what building this plan reads (machines, links, engines)."""
    asio, _, _ = B.libs()
    from asio_core.sim import Sim
    V = asio.Valuer("median", "auto", 0, sc.get("overrides", {}))
    try: Sim(copy.deepcopy(sc), B.cat(), V, 0)
    except (MemoryError, ValueError, KeyError, TypeError): pass
    return sorted(V.read)

def _nums(s: str) -> list: return [float(x) if "." in x or "e" in x else int(x) for x in (s or "").replace(",", " ").split()] if (s or "").strip()[:1] != "[" else json.loads(s)

async def h_run(request, payload, imr):
    from asio_ui.build import current, say
    ws, op = B.ws(request), payload["op"]
    if op == "show":
        ws.set_ui(run_open=payload["rid"])
        r = ws.result(payload["rid"])
        return imr.oob(result_html(r), "asio-out")
    if op == "trash":
        ws.trash_result(payload["rid"])
        return imr.oob(history(ws, ws.ui().get("scenario"), "cases|mc|sweep|compare|quote"), "asio-hist").oob(B.note("Moved to the trash."), "asio-out")
    try:
        ws, name, sc = current(request)
        _, _, run = B.libs()
        tier = payload.get("tier") or "auto"
        ins = {k: v for k, v in payload.items() if k not in ("type", "branch", "lvl", "op")}
        ws.set_form("run", ins)
        ws.checkpoint(name, f"before a run ({op})")
        if op == "cases":
            seeds = int(payload.get("seeds") or 4)
            def fn():
                cs = run.cases(B.cat(), sc, seeds=seeds, tier=tier)
                ok = cs.get("median", {}).get("metrics")
                return {"head": f"""median {B.fmt(ok['span_s'], 't')}, {ok['ok']:.0f} done, {B.fmt(ok['cost_per_ok_usd'], '$')} per task""" if ok else "median failed", "cases": cs, "sc": sc}
        elif op == "mc":
            n = max(4, min(int(payload.get("runs") or 64), 2000))
            def fn():
                res = run.monte_carlo(B.cat(), sc, n, tier, ["span_s", "cost_per_ok_usd", "ok"])
                s = res["summary"].get("span_s", {})
                return {"head": f"""{n} runs, time p5-p95 {B.fmt(s.get('p5'), 't')} to {B.fmt(s.get('p95'), 't')}""", "mc": res, "cases": run.cases(B.cat(), sc, 2, tier)}
        elif op == "sweep":
            key, vals, metric = (payload.get("key") or "").strip(), _nums(payload.get("values")), payload.get("metric") or "span_s"
            if not key or not vals: raise ValueError("give the value to sweep and the numbers to try")
            def fn(): return {"head": f"{key} over {len(vals)} values", "key": key, "metric": metric, "rows": run.sweep(B.cat(), sc, key, vals)}
        elif op == "compare":
            ids = F.parse("list", payload.get("compare"))
            if not ids: raise ValueError("pick the scenarios to compare")
            scs = {name: sc, **{i[3:] if i.startswith("ws:") else i: ws.get(i[3:]) if i.startswith("ws:") else i for i in ids}}
            def fn(): return {"head": f"{len(scs)} scenarios", "rows": [{"id": k, **run.once(B.cat(), v, "median", 0, tier)} for k, v in scs.items()]}
        elif op == "quote":
            mirror, rented = F.parse("list", payload.get("mirror")), payload.get("rented")
            if not mirror or not rented: raise ValueError("pick the engines to copy and the rented machine")
            count, at, fee, margin, n = int(payload.get("count") or 1), float(payload.get("overflow_at") or 1), float(payload.get("fee") or 0), float(payload.get("margin") or 0.15), max(4, min(int(payload.get("qruns") or 24), 500))
            def fn():
                q = run.quote(B.cat(), sc, mirror, rented, count, at, fee, margin, n, tier)
                o = q["options"]
                return {"head": f"""local {B.fmt(o['local']['span_p50'], 't')}; with {count} rented {B.fmt(o['overflow']['span_p50'], 't')} for about {B.fmt(o['overflow']['rent_p50'] + fee, '$')}""", "quote": q}
        else: raise ValueError(f"unknown run {op}")
        return imr.oob(poll_html(start(ws, name, op, ins, fn)), "asio-out")
    except (ValueError, KeyError, json.JSONDecodeError) as e: return say(imr, f"{type(e).__name__}: {e}")

async def h_poll(request, payload, imr):
    ws, j = B.ws(request), JOBS.get(payload.get("job", ""))
    if not j: return imr.oob(B.err("That job is gone (the portal restarted?). Finished results are in the list."), "asio-out")
    target = "asio-sout" if j["kind"] == "opt" else "asio-out"
    if j["status"] == "running": return imr.oob(poll_html(payload["job"]), target)
    if j["kind"] == "opt":
        from asio_ui import studies
        ws.set_ui(study_open=j["rid"])
        return imr.oob(studies.result_html(ws, ws.result(j["rid"])), target).oob(studies.overview(ws, j["rid"]), "asio-shist")
    ws.set_ui(run_open=j["rid"])
    return imr.oob(result_html(ws.result(j["rid"])), target).oob(history(ws, j["scenario"], "cases|mc|sweep|compare|quote", j["rid"]), "asio-hist")

async def h_form(request, payload, imr):
    """Keeps a form's fields in the workspace as they change, so they are there on return."""
    B.ws(request).set_form(payload["form"], {k: v for k, v in payload.items() if k not in ("type", "branch", "lvl", "form")})
    return imr

INTENTS = {"asio_run": h_run, "asio_poll": h_poll, "asio_form": h_form}
