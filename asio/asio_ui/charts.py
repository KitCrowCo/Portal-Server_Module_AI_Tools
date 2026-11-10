"""asio_ui/charts.py - the Charts tab: any kept result as rows, drawn on the axes you choose, and saved charts arranged as you like.
Rows per source:
  Monte Carlo   one per run: its metrics and every value it drew ("in: <key>"), so an input can be set against an outcome
  sweep         one per value tried;  comparison  one per scenario;  cases  one per case
  cases, jobs   one per job of the median run: arrival, latency, workload, outcome
  study         one per setup scored: its settings ("set: <path>") and kept metrics
Kinds: scatter, line (each group's points joined in order of x), bars (one cluster per value of x), histogram (how often values of x came up).
Several sources on one chart are told apart by colour; a text column can group instead. Charts are kept in the workspace (charts.json), in the order and width you set; a removed chart is appended to trash/charts_removed.jsonl."""
import json, math, time, uuid
from asio_ui import base as B, fields as F
from asio_ui.run_ui import METRICS, KIND_WORDS

NAMES = {k: l + {"t": " (s)", "$": " ($)"}.get(u, "") for k, l, u in METRICS} | {"slo_share": "share within SLO", "latency_p50_s": "p50 latency", "jobs": "jobs", "unfinished": "unfinished", "tokens_out": "tokens out", "ok_per_h": "tasks done per hour", "jobs_per_h": "jobs per hour", "cost_per_mtok_usd": "$ per million tokens out",
         "latency_s": "latency (s)", "t_arrive": "arrived at (s)", "steps": "agent steps", "value": "value tried", "cost": "study cost"}
KINDS = [("scatter", "scatter"), ("line", "line"), ("bars", "bars"), ("hist", "histogram")]

# --- Rows ---

def sources(ws) -> list:
    """(source id, label) for every kept result; cases give a second source of their jobs."""
    out = []
    for r in ws.results(limit=80):
        lbl = f"""{KIND_WORDS.get(r['kind'], r['kind'])} of {r['scenario']}, {B.ago(r['t'])}: {r.get('head', '')}"""
        out.append((r["id"], lbl))
        if r["kind"] == "cases": out.append((f"""{r['id']}:jobs""", f"""jobs of the median run, {lbl}"""))
    return out

def rows(ws, src: str) -> list:
    rid, part = (src.split(":", 1) + [""])[:2]
    r = ws.result(rid)
    d, k = r["data"], r["kind"]
    if d.get("error"): return []
    if k == "mc": return [{**m, **{f"in: {a}": v for a, v in (ins or {}).items()}} for m, ins in zip(d["mc"]["runs"], d["mc"].get("inputs") or [{}] * len(d["mc"]["runs"]))]
    if k == "sweep": return [{"value": x["value"], **x["metrics"]} for x in d["rows"] if not x.get("error")]
    if k == "compare": return [{"scenario": x["id"], **x["metrics"]} for x in d["rows"] if not x.get("error")]
    if k == "cases" and part == "jobs":
        first = (d["cases"].get("median") or next((v for v in d["cases"].values() if not v.get("error")), {})).get("first") or {}
        return [{**{a: b for a, b in j.items() if a != "id"}, "latency_s": (j["t_end"] - j["t_arrive"]) if j.get("t_end") is not None else None, "ok": 1 if j.get("ok") else 0} for j in first.get("jobs", [])]
    if k == "cases": return [{"case": c, **v["metrics"]} for c, v in d["cases"].items() if not v.get("error")]
    if k == "quote": return [{"option": o, **v} for o, v in d["quote"]["options"].items() if not isinstance(v, str)]
    if k == "opt": return [{"cost": p["cost"], **{f"set: {a}": (json.dumps(b) if isinstance(b, (list, dict)) else b) for a, b in p["params"].items()}, **{a.split(":")[0]: b for a, b in p["values"].items()}} for p in d["res"].get("points", [])]
    return []

def num(v) -> bool: return isinstance(v, (int, float)) and not isinstance(v, bool) and math.isfinite(v)

def columns(rs: list) -> tuple:
    """(numeric columns, text columns), in first-seen order."""
    keys = list(dict.fromkeys(k for r in rs for k in r))
    return [k for k in keys if any(num(r.get(k)) for r in rs)], [k for k in keys if any(isinstance(r.get(k), str) for r in rs)]

def name(k: str) -> str:
    if not k: return ""
    if k.startswith("in: "): return f"in: {F.value_label(k[4:])}"
    return NAMES.get(k, k.replace("_", " "))

# --- Drawing ---

def draw(ws, ch: dict) -> str:
    _, viz, _ = B.libs()
    srcs = [s for s in ch.get("src", []) if (ws.rdir / f"""{s.split(':')[0]}.json""").exists()]
    if not srcs: return B.err("The results this chart read are gone (moved to the trash?): edit it to pick others.")
    labels = dict(sources(ws))
    data = [(labels.get(s, s)[:60], r) for s in srcs for r in rows(ws, s)]
    x, y, g, kind = ch.get("x"), ch.get("y"), ch.get("group") or "", ch.get("kind", "scatter")
    grp = lambda lab, r: (str(r.get(g)) if g and g != "source" else lab if len(srcs) > 1 or g == "source" else "")
    if not x or (kind != "hist" and not y): return B.note("Pick what goes on each axis.")
    title = ch.get("title") or (f"{name(y)} against {name(x)}" if kind in ("scatter", "line") else f"{name(y)} by {name(x)}" if kind == "bars" else f"how often: {name(x)}")
    if kind == "hist":
        groups = {}
        for lab, r in data:
            if num(r.get(x)): groups.setdefault(grp(lab, r) or name(x), []).append(r[x])
        return viz.hist(groups, name(x), title=title) or B.note("No numbers in that column.")
    if kind == "bars":
        cats = {}
        for lab, r in data:
            if num(r.get(y)): cats.setdefault(r.get(x) if not num(r.get(x)) else B.fmt(r[x]), {}).setdefault(grp(lab, r) or name(y), []).append(r[y])
        return viz.bars([(c, {a: sum(v) / len(v) for a, v in d.items()}) for c, d in cats.items()], name(y), logy=ch.get("logy", False), title=title) or B.note("No numbers in that column.")
    pts = [{"x": r[x], "y": r[y], "label": "", "group": grp(lab, r)} for lab, r in data if num(r.get(x)) and num(r.get(y))]
    return viz.scatter(pts, name(x), name(y), logx=ch.get("logx", False), logy=ch.get("logy", False), title=title, lines=kind == "line") or B.note("No rows have numbers in both columns (or, on a log axis, positive ones).")

# --- Saved charts ---

def saved(ws) -> list:
    f = ws.root / "charts.json"
    return json.loads(f.read_text()) if f.exists() else []

def save(ws, charts: list): (ws.root / "charts.json").write_text(json.dumps(charts, indent=1))

# --- Tab ---

def builder(ws) -> str:
    f = ws.form("chart")
    src = F.parse("list", f.get("src"))
    all_rows = [r for s in src if (ws.rdir / f"""{s.split(':')[0]}.json""").exists() for r in rows(ws, s)]
    nums, texts = columns(all_rows)
    opt = lambda opts, v: "".join(f"""<option value="{B.esc(k)}" {'selected' if str(k) == str(v) else ''}>{B.esc(l)}</option>""" for k, l in opts)
    src_opts = "".join(f"""<option value="{B.esc(k)}" {'selected' if k in src else ''}>{B.esc(l)}</option>""" for k, l in sources(ws))
    xs = [(k, name(k)) for k in (texts + nums if f.get("kind") == "bars" else nums)]
    chk = lambda n, lbl: f"""<label class="asio-chk"><input type="checkbox" name="{n}" value="1" {'checked' if f.get(n) else ''}>{lbl}</label>"""
    edit = f.get("edit") or ""
    return f"""<form class="asio-card" id="asio-chartform" onsubmit="return false" {B.act("asio_chart", trigger="change", op="draft")}><input type="hidden" name="edit" value="{B.esc(edit)}">
        <div class="row asio-wrap">{F.field("from these results (ctrl or shift for several)", f'<select class="asio-in asio-w30" name="src" multiple size="5">{src_opts}</select>')}
        <div>{F.field("kind", f'<select class="asio-in asio-w8" name="kind">{opt(KINDS, f.get("kind", "scatter"))}</select>')}{F.field("across (x)", f'<select class="asio-in asio-w14" name="x"><option value=""></option>{opt(xs, f.get("x"))}</select>')}
        {F.field("up (y)", f'<select class="asio-in asio-w14" name="y"><option value=""></option>{opt([(k, name(k)) for k in nums], f.get("y"))}</select>') if f.get("kind") != "hist" else ""}{F.field("colour by", f'<select class="asio-in asio-w14" name="group"><option value="">(each result)</option><option value="source" {"selected" if f.get("group") == "source" else ""}>result, always</option>{opt([(k, name(k)) for k in texts], f.get("group"))}</select>')}</div>
        <div>{F.field("title (optional)", f'<input class="asio-in asio-w14" name="title" value="{B.esc(f.get("title", ""))}">')}<div class="row">{chk("logx", "log x")}{chk("logy", "log y")}{chk("full", "full width")}</div>
        <div class="row">{B.btn("Save changes" if edit else "Keep this chart", "asio_chart", op="keep", include="#asio-chartform")}{B.btn("Clear", "asio_chart", op="clear")}</div></div></div></form>
        <div class="asio-card" id="asio-chartprev">{draw(ws, spec(f)) if src else '<div class="asio-dim">Pick one or more results, then what goes on each axis; the chart draws as you choose.</div>'}</div>"""

def spec(f: dict) -> dict: return {"src": F.parse("list", f.get("src")), "kind": f.get("kind", "scatter"), "x": f.get("x"), "y": f.get("y"), "group": f.get("group", ""), "title": f.get("title", ""), "logx": bool(f.get("logx")), "logy": bool(f.get("logy")), "full": bool(f.get("full"))}

def grid(ws) -> str:
    cs = saved(ws)
    if not cs: return '<div class="asio-dim">Kept charts show here, in the order and width you give them.</div>'
    cards = "".join(f"""<div class="asio-chart {'asio-full' if c.get('full') else ''}"><div class="row"><span class="grow"></span>{B.btn("&#x25B2;", "asio_chart", op="move", id=c['id'], d=-1, title="earlier")}{B.btn("&#x25BC;", "asio_chart", op="move", id=c['id'], d=1, title="later")}
        {B.btn("wide" if not c.get('full') else "half", "asio_chart", op="width", id=c['id'])}{B.btn("Edit", "asio_chart", op="edit", id=c['id'])}{B.btn("&#x2715;", "asio_chart", cls="asio-btn asio-x", op="remove", id=c['id'], confirm="Remove this chart? Its settings are kept in the workspace trash.")}</div>{draw(ws, c)}</div>""" for c in cs)
    return f"""<div class="asio-grid">{cards}</div>"""

def page(ws) -> str: return f"""<div class="asio-pad"><div id="asio-chartbuild">{builder(ws)}</div><div class="asio-h">Kept charts</div><div id="asio-chartgrid">{grid(ws)}</div></div>"""

# --- Intents ---

async def h_chart(request, payload, imr):
    ws, op = B.ws(request), payload["op"]
    f = {k: v for k, v in payload.items() if k not in ("type", "branch", "lvl", "op", "id", "d")}
    if op == "draft":
        old = ws.form("chart")
        ws.set_ui(forms={**ws.ui().get("forms", {}), "chart": f})
        if F.parse("list", old.get("src")) == F.parse("list", f.get("src")) and old.get("kind") == f.get("kind"): return imr.oob(draw(ws, spec(f)), "asio-chartprev")   # the axis lists stay as they are
        return imr.oob(builder(ws), "asio-chartbuild")
    cs = saved(ws)
    i = next((k for k, c in enumerate(cs) if c["id"] == payload.get("id")), None)
    if op == "keep":
        sp = spec(f)
        if not sp["src"] or not sp["x"]: return imr.oob(B.err("Pick results and an x axis first.") + builder(ws), "asio-chartbuild")
        j = next((k for k, c in enumerate(cs) if c["id"] == f.get("edit")), None)
        if j is None: cs.append({"id": uuid.uuid4().hex[:8], **sp})
        else: cs[j] = {"id": cs[j]["id"], **sp}
        save(ws, cs)
        ws.set_ui(forms={**ws.ui().get("forms", {}), "chart": {}})
    elif op == "clear": ws.set_ui(forms={**ws.ui().get("forms", {}), "chart": {}})
    elif op == "edit" and i is not None:
        c = cs[i]
        ws.set_ui(forms={**ws.ui().get("forms", {}), "chart": {**{k: v for k, v in c.items() if k not in ("id", "logx", "logy", "full")}, **{k: "1" for k in ("logx", "logy", "full") if c.get(k)}, "edit": c["id"]}})
    elif op == "move" and i is not None:
        j = min(max(i + int(payload.get("d", 0)), 0), len(cs) - 1)
        cs.insert(j, cs.pop(i))
        save(ws, cs)
    elif op == "width" and i is not None:
        cs[i]["full"] = not cs[i].get("full")
        save(ws, cs)
    elif op == "remove" and i is not None:
        with open(ws.root / "trash" / "charts_removed.jsonl", "a") as t: t.write(json.dumps({"t": time.time(), **cs.pop(i)}) + "\n")
        save(ws, cs)
    return imr.oob(page(ws), "asio-charts")

INTENTS = {"asio_chart": h_chart}
