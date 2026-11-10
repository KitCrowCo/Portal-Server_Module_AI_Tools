"""asio_ui/studies.py - the Studies tab: goal seeking over settings picked from the open scenario, an overview of every study, and the trade-offs each found.
Knobs come from the scenario itself (an engine's slots, quant, machine, expert arrangement and cache split; a machine's model; a workload's in-flight limit; a link's type), so nothing is typed as a path.
Goals: up to four, each a metric to lower or raise, or to keep under or over a target (hard: a setup that misses it never wins). Every setup's time, cost, energy and latency are kept, so the trade-off chart can show any two of them."""
import json, math
from asio_ui import base as B, fields as F
from asio_ui.run_ui import start, poll_html, METRICS

GOAL_METRICS = [(k, l) for k, l, _ in METRICS] + [("slo_share", "share within SLO"), ("latency_p50_s", "p50 latency"), ("ok_per_h", "tasks done per hour"), ("unfinished", "jobs left unfinished")]
WATCH_WORDS = {"span_s": ("time to finish", "t"), "cost_per_ok_usd": ("$ per task done", "$"), "wh_per_ok": ("Wh per task done", "n"), "ok": ("tasks done", "n"), "latency_p90_s": ("p90 latency", "t"), "slo_share": ("share within SLO", "%"), "tps_total": ("total t/s", "n"), "energy_wh": ("energy Wh", "n")}
CTX = [4096, 8192, 16384, 32768, 65536, 131072, 262144]

# --- Knobs ---

def node_ram(c, s) -> float: return (c.get("nodes", s) if isinstance(s, str) else {**c.get("nodes", s["template"]), **s}).get("ram_gb", 32)

def knobs(sc: dict) -> list:
    """Every setting of this scenario a study can change: {"path", "group", "label", "type", "lo", "hi", "options": [(value, label)], "pick": values on by default}."""
    c, out = B.cat(), []
    names = list(sc.get("nodes") or {})
    quants = list(c.get("models", "quants")["table"])
    for e in sc.get("engines") or []:
        m, g, p = c.get("models", e["model"]), f"""engine {e['id']} ({c.get('models', e['model']).get('label', e['model'])})""", f"""engines.{e['id']}"""
        llm, moe = m.get("kind") == "llm", m.get("arch") == "moe"
        out.append({"path": f"{p}.slots", "group": g, "label": "slots", "type": "int", "lo": 1, "hi": max(4, 2 * int(e.get("slots", 1)))})
        if llm:
            out.append({"path": f"{p}.ctx", "group": g, "label": "context per slot", "type": "choice", "options": [(x, f"{x // 1024}k") for x in CTX if x <= m.get("ctx_max", 262144)], "pick": [int(e.get("ctx", 8192))]})
            out.append({"path": f"{p}.kv_keep", "group": g, "label": "sessions whose KV is kept", "type": "int", "lo": int(e.get("slots", 1)), "hi": 4 * max(int(e.get("slots", 1)), 1)})
            if not moe: out.append({"path": f"{p}.quant", "group": g, "label": "quant", "type": "choice", "options": [(q, q) for q in quants], "pick": list(dict.fromkeys([e.get("quant") or m.get("default_quant"), "Q4_K_M", "Q6_K", "Q8_0"]))})
            if m.get("mtp"): out.append({"path": f"{p}.mtp", "group": g, "label": "MTP on or off", "type": "bool"})
        if e.get("node"): out.append({"path": f"{p}.node", "group": g, "label": "which machine", "type": "choice", "options": [(n, n) for n in names], "pick": [e["node"]]})
        if len(m.get("backends", [])) > 1: out.append({"path": f"{p}.backend", "group": g, "label": "program", "type": "choice", "options": [(b, b) for b in m["backends"]], "pick": [e.get("backend") or m["backends"][0]]})
        if moe and e.get("heat"): out.append({"path": f"{p}.arrangement", "group": g, "label": "expert arrangement", "type": "choice", "options": [("hottest", "known heat"), ("learned", "caches learn"), ("spread", "fixed random")], "pick": ["hottest", "learned", "spread"]})
        if moe and not isinstance(e.get("cache_gb"), (int, float)):
            for n in e.get("nodes") or [e.get("node")]: out.append({"path": f"{p}.cache_gb.{n}", "group": g, "label": f"expert cache on {n} (GB)", "type": "float", "lo": 0, "hi": float(node_ram(c, sc["nodes"][n]))})
    for n, s in (sc.get("nodes") or {}).items():
        cur = s if isinstance(s, str) else s["template"]
        out.append({"path": f"nodes.{n}", "group": f"machine {n}", "label": "which machine it is", "type": "choice", "options": [(i, c.get("nodes", i).get("label", i)) for i in c.ids("nodes")], "pick": [cur]})
    for i, w in enumerate(sc.get("workloads") or []):
        if w["kind"] != "mix": out.append({"path": f"workloads.{i}.max_in_flight", "group": f"""workload {w.get('name', i)}""", "label": "most at once", "type": "int", "lo": 1, "hi": max(8, 2 * int(w.get("max_in_flight") or 4))})
    for i, l in enumerate(sc.get("links") or []): out.append({"path": f"links.{i}.type", "group": f"""link {i + 1} ({l['type']})""", "label": "link type", "type": "choice", "options": [(t, c.get("links", t).get("label", t)) for t in c.ids("links") if not t.startswith("magpie")], "pick": [l["type"]]})
    return out

def space(sc: dict, f: dict) -> list:
    """The search space the form's ticked knobs make."""
    out = []
    for k in knobs(sc):
        key = f"""k|{k['path']}"""
        if not f.get(f"{key}|on"): continue
        if k["type"] in ("int", "float"):
            lo, hi = F.parse(k["type"], f.get(f"{key}|lo") or k["lo"]), F.parse(k["type"], f.get(f"{key}|hi") or k["hi"])
            if hi <= lo: raise ValueError(f"""{k['group']}, {k['label']}: the high end must be above the low end""")
            out.append({"path": k["path"], "type": k["type"], "lo": lo, "hi": hi})
        elif k["type"] == "bool": out.append({"path": k["path"], "type": "bool"})
        else:
            opts = F.parse("list", f.get(f"{key}|opt"))
            vals = [v for v, _ in k["options"] if str(v) in opts]
            if len(vals) < 2: raise ValueError(f"""{k['group']}, {k['label']}: tick at least two choices""")
            out.append({"path": k["path"], "type": "choice", "options": vals})
    if not out: raise ValueError("tick at least one setting to search over")
    return out

def goals(f: dict) -> list:
    out = []
    for g in range(4):
        m = f.get(f"g{g}_metric")
        if not m: continue
        kind, tgt = f.get(f"g{g}_kind", "min"), F.parse("float", f.get(f"g{g}_target"))
        if kind in ("le", "ge") and tgt is None: raise ValueError(f"goal {g + 1}: a limit needs its number")
        out.append({"metric": m, "kind": kind, "weight": F.parse("float", f.get(f"g{g}_weight")) or 1.0, "stat": f.get(f"g{g}_stat") or "mean", **({"target": tgt} if tgt is not None else {}), **({"hard": True} if f.get(f"g{g}_hard") and kind in ("le", "ge") else {})})
    if not out: raise ValueError("give at least one goal")
    return out

def knob_form(sc: dict, f: dict) -> str:
    rows, last = [], None
    for k in knobs(sc):
        key = f"""k|{k['path']}"""
        if k["group"] != last: rows.append(f"""<div class="asio-h">{B.esc(k['group'])}</div>""")
        last = k["group"]
        on = f"""<label class="asio-chk"><input type="checkbox" name="{B.esc(key)}|on" value="1" {'checked' if f.get(key + '|on') else ''}>{B.esc(k['label'])}</label>"""
        if k["type"] in ("int", "float"): ctl = f"""from <input class="asio-in asio-w4" name="{B.esc(key)}|lo" value="{B.esc(f.get(key + '|lo', k['lo']))}"> to <input class="asio-in asio-w4" name="{B.esc(key)}|hi" value="{B.esc(f.get(key + '|hi', k['hi']))}">"""
        elif k["type"] == "bool": ctl = '<span class="asio-dim">tries both</span>'
        else:
            picked = F.parse("list", f.get(f"{key}|opt")) if f.get(key + "|on") or f.get(f"{key}|opt") else [str(x) for x in k.get("pick", [])]
            ctl = "".join(f"""<label class="asio-chk"><input type="checkbox" name="{B.esc(key)}|opt" value="{B.esc(v)}" {'checked' if str(v) in picked else ''}>{B.esc(l)}</label>""" for v, l in k["options"])
            ctl = f"""<span class="asio-multi">{ctl}</span>"""
        rows.append(f"""<div class="row asio-knob">{on}{ctl}</div>""")
    return "".join(rows)

def goal_form(f: dict) -> str:
    opt = lambda opts, v: "".join(f"""<option value="{B.esc(k)}" {'selected' if str(k) == str(v) else ''}>{B.esc(l)}</option>""" for k, l in opts)
    dflt = {"g0_metric": "span_s"}
    rows = "".join(f"""<div class="row asio-wrap"><select class="asio-in asio-w12" name="g{g}_metric"><option value="">(no goal {g + 1})</option>{opt(GOAL_METRICS, f.get(f"g{g}_metric", dflt.get(f"g{g}_metric", "")))}</select>
        <select class="asio-in asio-w8" name="g{g}_kind">{opt([("min", "as low as possible"), ("max", "as high as possible"), ("le", "at most"), ("ge", "at least")], f.get(f"g{g}_kind", "min"))}</select><input class="asio-in asio-w5" name="g{g}_target" value="{B.esc(f.get(f"g{g}_target", ""))}" placeholder="limit">
        <select class="asio-in asio-w8" name="g{g}_stat">{opt([("mean", "on average"), ("p90", "in 9 of 10 runs"), ("worst", "even in the worst run")], f.get(f"g{g}_stat", "mean"))}</select>{F.field("weight", f'<input class="asio-in asio-w3" name="g{g}_weight" value="{B.esc(f.get(f"g{g}_weight", "1"))}">')}
        <label class="asio-chk" title="a limit that must hold: a setup that misses it never wins"><input type="checkbox" name="g{g}_hard" value="1" {'checked' if f.get(f'g{g}_hard') else ''}>must hold</label></div>""" for g in range(4))
    return rows

# --- Results ---

def setting_names(res: dict) -> dict:
    """Search paths in words, from the knobs of the scenario the study ran on."""
    try: return {k["path"]: f"""{k['group'].split(' (')[0]}: {k['label']}""" for k in knobs(res.get("base") or res["scenario"])}
    except (KeyError, ValueError, TypeError): return {}

def point_val(p: dict, m: str):
    return p["values"].get(f"{m}:mean")

def tradeoff(rid: str, res: dict, x: str, y: str) -> str:
    """Every setup scored, on two of the kept metrics; the trade-off front of those two marked, and listed with a button to make each a scenario."""
    _, viz, _ = B.libs()
    pts = [dict(p, k=i) for i, p in enumerate(res.get("points", []))]
    ok = [p for p in pts if isinstance(point_val(p, x), (int, float)) and isinstance(point_val(p, y), (int, float)) and math.isfinite(point_val(p, x)) and math.isfinite(point_val(p, y))]
    if not ok: return '<div class="asio-dim">No setup in this study could run, so there is nothing to compare.</div>'
    front = B.libs()[2].athene().pareto(ok, {f"{x}:mean": "min" if x not in ("ok", "slo_share", "tps_total") else "max", f"{y}:mean": "min" if y not in ("ok", "slo_share", "tps_total") else "max"})
    fk = {p["k"] for p in front}
    best = (res.get("best") or {}).get("params")
    data = [{"x": point_val(p, x), "y": point_val(p, y), "label": "", "group": "best by the goals" if p["params"] == best else "trade-off front" if p["k"] in fk else "other setups"} for p in ok]
    pos = lambda v: all(d[v] > 0 for d in data)
    unit = lambda m: WATCH_WORDS[m][0] + {"t": " (s)", "$": " ($)"}.get(WATCH_WORDS[m][1], "")
    names = setting_names(res)
    chart = viz.scatter(data, unit(x), unit(y), logx=pos("x") and max(d["x"] for d in data) > 20 * min(d["x"] for d in data), logy=pos("y") and max(d["y"] for d in data) > 20 * min(d["y"] for d in data), title=f"{len(ok)} setups: {WATCH_WORDS[y][0]} against {WATCH_WORDS[x][0]}")
    cols = [m for m in WATCH_WORDS if any(point_val(p, m) is not None for p in front)]
    rows = [[B.esc("; ".join(f"{names.get(k, k)} {json.dumps(v)}" for k, v in p["params"].items()))] + [B.fmt(point_val(p, m), WATCH_WORDS[m][1]) for m in cols] + [B.btn("Make a scenario", "asio_study", op="adopt", rid=rid, k=p["k"])] for p in sorted(front, key=lambda p: point_val(p, x))]
    axes = "".join(f"""<select class="asio-in asio-w12" name="{a}">{''.join(f'<option value="{k}" {"selected" if k == v else ""}>{l}</option>' for k, (l, _) in WATCH_WORDS.items())}</select>""" for a, v in (("x", x), ("y", y)))
    pick = f"""<form class="row" onsubmit="return false" {B.act("asio_study", trigger="change", op="axes", rid=rid)}><span class="asio-dim">chart</span>{axes}</form>"""
    return f"""{pick}<div class="asio-fig">{chart}</div><div class="asio-h">The trade-off front: no other setup beats these on both</div>{B.tbl(["settings"] + [WATCH_WORDS[m][0] for m in cols] + [""], rows)}"""

def result_html(ws, r: dict) -> str:
    _, viz, _ = B.libs()
    d = r["data"]
    top = f"""<div class="row asio-rhead"><b class="grow">study of {B.esc(r['scenario'])}, {B.ago(r['t'])}</b>{B.btn("Move to trash", "asio_study", cls="asio-btn asio-danger", op="trash", rid=r['id'], confirm="Move this study's result to the workspace trash?")}</div>"""
    if d.get("error"): return top + B.err(d["error"])
    res, f = d["res"], ws.form("pareto")
    best = res["best"] or {}
    goals_txt = "; ".join(f"""{g['metric']} {dict(min='as low as possible', max='as high as possible', le='at most', ge='at least', target='near').get(g['kind'])}{' ' + B.fmt(g['target']) if g.get('target') is not None else ''}{' (must hold)' if g.get('hard') else ''}""" for g in res.get("goals", []))
    names = setting_names(res)
    params = B.tbl(["setting", "best by the goals"], [[B.esc(names.get(k, k)), B.esc(json.dumps(v))] for k, v in (best.get("params") or {}).items()])
    vals = ", ".join(f"""{WATCH_WORDS.get(k.split(':')[0], (k,))[0]} {B.fmt(v, WATCH_WORDS.get(k.split(':')[0], ('', 'n'))[1])}""" for k, v in (best.get("values") or {}).items())
    hist = [{"x": h["evaluations"], "y": h["best"], "label": "", "group": "best so far"} for h in res["history"] if h["best"] is not None and math.isfinite(h["best"])]
    return f"""{top}{B.note(f"Goals: {B.esc(goals_txt)}. {res['evaluations']} setups tried, {res['ensemble']['runs']} runs ({res['ensemble']['hits']} reused). Best: {vals}.")}{params}{B.btn("Make the best a scenario", "asio_study", op="adopt", rid=r['id'])}
        {tradeoff(r['id'], res, f.get('x', 'cost_per_ok_usd'), f.get('y', 'span_s'))}<div class="asio-fig">{viz.scatter(hist, 'setups tried', 'best cost', title='Search progress')}</div>"""

def overview(ws, open_id: str = None) -> str:
    rows = ws.results(kind="opt", limit=60)
    if not rows: return '<div class="asio-dim">No studies yet.</div>'
    out = []
    for r in rows:
        best = (r.get("best") or {})
        out.append([f"""<a class="asio-link {'asio-on' if r['id'] == open_id else ''}" {B.act("asio_study", op="show", rid=r['id'])}>{B.esc(r['scenario'])}</a>""", B.ago(r["t"]), B.esc(r.get("head", ""))] + [B.fmt(best.get(f"{m}:mean"), WATCH_WORDS[m][1]) for m in ("span_s", "cost_per_ok_usd", "wh_per_ok", "latency_p90_s")])
    return B.tbl(["scenario", "when", "study", "time", "$ per task", "Wh per task", "p90 latency"], out)

# --- Tab ---

def page(ws) -> str:
    name = ws.ui().get("scenario")
    open_id = ws.ui().get("study_open")
    shown = result_html(ws, ws.result(open_id)) if open_id and (ws.rdir / f"{open_id}.json").exists() else '<div class="asio-dim">Run a study, or pick one above.</div>'
    if not name or not ws.exists(name): form = '<div class="asio-dim">Open or start a scenario on the Build tab to study it.</div>'
    else:
        sc, f = ws.get(name), ws.form(f"study.{name}")
        opt = lambda opts, v: "".join(f"""<option value="{k}" {'selected' if k == v else ''}>{l}</option>""" for k, l in opts)
        form = f"""<form class="asio-card" id="asio-studyform" onsubmit="return false" {B.act("asio_form", trigger="change", form=f"study.{name}")}><div class="asio-dim">Studying <b>{B.esc(name)}</b>: tick what may change and set its range; every setup is scored on the same seeded runs, so differences are the setup's, not luck's.</div>
            {B.sec("What may change", knob_form(sc, f))}{B.sec("Goals", goal_form(f))}
            <div class="row asio-wrap">{F.field("method", f'<select class="asio-in asio-w12" name="method">{opt([("ga", "genetic algorithm"), ("tabu", "tabu search"), ("anneal", "simulated annealing")], f.get("method", "ga"))}</select>')}{F.field("setups to try", f'<input class="asio-in asio-w4" name="budget" value="{B.esc(f.get("budget", "120"))}">')}
            {F.field("runs per setup", f'<input class="asio-in asio-w4" name="runs" value="{B.esc(f.get("runs", "6"))}">')}{B.btn("Seek the goal", "asio_study", op="run", include="#asio-studyform")}</div></form>"""
    return f"""<div class="asio-pad">{B.sec("Every study", f'<div id="asio-shist">{overview(ws, open_id)}</div>')}{form}<div id="asio-sout">{shown}</div></div>"""

# --- Intents ---

async def h_study(request, payload, imr):
    from asio_ui.build import current, say
    ws, op = B.ws(request), payload["op"]
    try:
        if op == "show":
            ws.set_ui(study_open=payload["rid"])
            return imr.oob(result_html(ws, ws.result(payload["rid"])), "asio-sout").oob(overview(ws, payload["rid"]), "asio-shist")
        if op == "axes":
            ws.set_form("pareto", {"x": payload.get("x"), "y": payload.get("y")})
            return imr.oob(result_html(ws, ws.result(payload["rid"])), "asio-sout")
        if op == "trash":
            ws.trash_result(payload["rid"])
            ws.set_ui(study_open=None)
            return imr.oob(overview(ws), "asio-shist").oob(B.note("Moved to the trash."), "asio-sout")
        if op == "adopt":
            r = ws.result(payload["rid"])
            res = r["data"]["res"]
            _, _, run = B.libs()
            sc = run.apply(res.get("base") or res["scenario"], res["points"][int(payload["k"])]["params"]) if payload.get("k") not in (None, "") else res["scenario"]
            new = ws.create(f"""{r['scenario']}-study""", sc, f"""a setup from study {r['id']}""")
            ws.set_ui(scenario=new)
            return say(imr, f"Made {new}; it is now the open scenario on the Build tab.", bad=False)
        ws, name, sc = current(request)
        _, _, run = B.libs()
        f = {k: v for k, v in payload.items() if k not in ("type", "branch", "lvl", "op")}
        ws.set_form(f"study.{name}", f)
        sp, gl = space(sc, f), goals(f)
        method, budget, runs = f.get("method") or "ga", int(f.get("budget") or 120), int(f.get("runs") or 6)
        kw = {"generations": max(budget // 12, 2), "size": 12} if method == "ga" else {"iterations": budget}
        ws.checkpoint(name, "before a study")
        def fn():
            res = run.optimize(B.cat(), sc, sp, gl, method, runs, budget, **kw)
            return {"head": f"""{method}, {len(sp)} settings, {res['evaluations']} setups""", "res": res, "best": (res["best"] or {}).get("values", {})}
        return imr.oob(poll_html(start(ws, name, "opt", f, fn)), "asio-sout")
    except (ValueError, KeyError, json.JSONDecodeError) as e: return say(imr, f"{type(e).__name__}: {e}")

INTENTS = {"asio_study": h_study}
