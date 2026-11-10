"""asio/report.py - what one run says: a flat set of metrics (what goals and Monte Carlo summaries read), and detail for people (per engine, node, link, job; where the time went; what held it back; what it cost against APIs).
Bottlenecks: for each engine, the share of its busy time each limiter set (mem, compute, ssd, hop, overhead), how busy it was, and how long work queued for it. The engine with the most queueing and the highest busy share is where the next resource helps most."""
import collections, math
import numpy as np

H = 3600.0

def _pct(xs, p):
    xs = [x for x in xs if x is not None]
    return float(np.percentile(xs, p)) if xs else None

def economics(sim, span_s: float, econ: dict) -> dict:
    V = sim.V
    kwh_price, life_h = V(econ["electricity_usd_kwh"], "econ.electricity"), V(econ["life_months"], "econ.life_months") * 730
    cop = V(econ.get("cooling_cop", 0), "econ.cooling_cop")
    nodes, hw, wh, rent = {}, 0.0, 0.0, 0.0
    for n, nd in sim.pool.nodes.items():
        es = [e for e in sim.engines.values() if nd in e.nodes]
        busy = min(sum(e.stats["busy_s"] / len(e.nodes) for e in es), span_s)
        r = rent_usd(nd, es, span_s) if nd.rent else 0.0
        e_wh = 0.0 if nd.rent else (nd.idle_w * span_s + (nd.load_w - nd.idle_w) * busy) / H   # a rented machine's power is in its hourly price
        nodes[n] = {"busy_s": round(busy, 2), "util": round(busy / span_s, 4) if span_s else 0.0, "energy_wh": round(e_wh, 3), "ram_used_gb": round(nd.used_gb, 2), "ram_usable_gb": round(nd.usable_gb, 2), "holding": nd.uses, **({"rent_usd": round(r, 4), "rented_from": nd.rent["provider"]} if nd.rent else {})}
        hw += nd.price / life_h * span_s / H
        wh, rent = wh + e_wh, rent + r
    energy = wh / 1000 * kwh_price * (1 + (1 / cop if cop else 0))
    fixed = (V(econ.get("internet_usd_month", 0), "econ.internet") + V(econ.get("space_usd_month", 0), "econ.space")) / 730 * span_s / H
    return {"nodes": nodes, "energy_wh": wh, "cost_hw": hw, "cost_energy": energy, "cost_fixed": fixed, "cost_rent": rent, "cost_total": hw + energy + fixed + rent}

def rent_usd(nd, engines: list, span_s: float) -> float:
    """What a rented machine costs for this run: from the start (rented for the whole run), or from when it was first asked for until its last work ends (at least until it was ready), in whole billing steps. Never asked for: nothing."""
    used = [e for e in engines if e.used or e.overflow is None]
    if not used: return 0.0
    whole = nd.rent["from"] == "start" or any(e.overflow is None for e in used)
    first = 0.0 if whole else min(getattr(e, "t_req", span_s) for e in used)
    end = span_s if whole else max([e.stats.get("last_s", 0.0) for e in used] + [first + nd.rent["start_s"]])
    return math.ceil(max(end - first, 0.0) / nd.rent["bill_s"]) * nd.rent["bill_s"] / H * nd.rent["usd_h"]

def api_cost(cat, computed: int, reused: int, out: int, cached_share: float = None) -> dict:
    """What the same tokens would cost on each listed API. Input this node computed is billed as input; input it reused from held context is billed at the cached rate when cached_share is None (the API's best case), or at cached_share of it otherwise."""
    rows = {}
    for i, a in cat.data["apis"].items():
        if a.get("in") is None: continue
        reused_cached = reused if cached_share is None else reused * cached_share
        rows[i] = round((computed + reused - reused_cached) / 1e6 * a["in"] + reused_cached / 1e6 * a["cached"] + out / 1e6 * a["out"], 4)
    return rows

def build(sim, econ: dict) -> dict:
    jobs = [j for j in sim.done if not j.get("parent") and j["kind"] != "stage"]
    span = max([j["t_end"] for j in sim.done] + [0.0]) or 1e-9
    ok = [j for j in jobs if j.get("ok")]
    out_tok = sum(e.stats["tokens_out"] for e in sim.engines.values())
    computed, reused = sum(e.stats["prefill_computed"] for e in sim.engines.values()), sum(e.stats["prefill_reused"] for e in sim.engines.values())
    eco = economics(sim, span, econ)
    engines = {}
    for i, e in sim.engines.items():
        lim = {k: round(v / e.stats["busy_s"], 3) for k, v in e.stats["by_limit"].items()} if e.stats["busy_s"] else {}
        engines[i] = {"model": e.model["id"], "backend": e.backend["id"], "nodes": [n.name for n in e.nodes], "util": round(e.stats["busy_s"] / span, 4), **{k: (round(v, 3) if isinstance(v, float) else v) for k, v in e.stats.items() if k != "by_limit"}, "limits": lim, "quant": getattr(e.phys, "quant", None), "decode_tps": round(e.stats["tokens_out"] / e.stats["decode_stream_s"], 3) if e.stats["decode_stream_s"] else None, "prefill_tps": round(e.stats["prefill_computed"] / e.stats["prefill_s"], 2) if e.stats["prefill_s"] else None, "hit": round(getattr(e.phys, "hit", 0) or 0, 4) if e.phys.__dict__.get("hit") is not None else None}
    lat = [j["t_end"] - j["t_arrive"] for j in jobs]
    by_wl = collections.defaultdict(list)
    for j in jobs: by_wl[j["workload"]].append(j)
    wl = {w: {"jobs": len(js), "ok": sum(1 for j in js if j.get("ok")), "latency_p50_s": _pct([j["t_end"] - j["t_arrive"] for j in js], 50), "latency_p90_s": _pct([j["t_end"] - j["t_arrive"] for j in js], 90), "within_slo": (sum(1 for j in js if j["t_end"] - j["t_arrive"] <= j["slo"]) / len(js)) if js and js[0].get("slo") else None, "outcomes": dict(collections.Counter(j.get("outcome", "done" if j.get("ok") else "failed") for j in js))} for w, js in by_wl.items()}
    hot = max(engines.items(), key=lambda kv: (kv[1]["wait_s"], kv[1]["util"]), default=(None, {}))
    bottleneck = {"engine": hot[0], "limit": max(hot[1].get("limits") or {"-": 1}, key=(hot[1].get("limits") or {"-": 1}).get), "util": hot[1].get("util"), "queued_s": hot[1].get("wait_s")} if hot[0] else None
    apis = api_cost(sim.cat, computed, reused, out_tok)
    metrics = {"span_s": span, "jobs": len(jobs), "ok": len(ok), "success_rate": len(ok) / len(jobs) if jobs else 0.0, "unfinished": len(sim.unfinished), "tokens_out": out_tok, "prefill_computed": computed, "prefill_reused": reused,
               "reuse_share": reused / (computed + reused) if computed + reused else 0.0, "tps_total": out_tok / span, "jobs_per_h": len(jobs) / span * H, "ok_per_h": len(ok) / span * H,
               "decode_tps": out_tok / sum(e.stats["decode_stream_s"] for e in sim.engines.values()) if any(e.stats["decode_stream_s"] for e in sim.engines.values()) else None,
               "prefill_tps": computed / sum(e.stats["prefill_s"] for e in sim.engines.values()) if any(e.stats["prefill_s"] for e in sim.engines.values()) else None,
               "latency_p50_s": _pct(lat, 50), "latency_p90_s": _pct(lat, 90), "slo_share": (sum(1 for j in slo if j["t_end"] - j["t_arrive"] <= j["slo"]) / len(slo)) if (slo := [j for j in jobs if j.get("slo")]) else None, "wait_total_s": sum(e.stats["wait_s"] for e in sim.engines.values()),
               "energy_wh": eco["energy_wh"], "cost_usd": eco["cost_total"], "cost_rent_usd": eco["cost_rent"], "cost_per_ok_usd": eco["cost_total"] / len(ok) if ok else float("inf"), "cost_per_mtok_usd": eco["cost_total"] / out_tok * 1e6 if out_tok else None,
               "wh_per_ok": eco["energy_wh"] / len(ok) if ok else float("inf"), **{f"api_{k}_usd": v for k, v in apis.items()}}
    return {"metrics": metrics, "engines": engines, "nodes": eco["nodes"], "links": [{"type": l.spec["type"], "members": l.members, "busy_s": round(l.busy_s, 4), "gb": round(l.gb, 4), "messages": l.messages} for l in sim.pool.links],
            "workloads": wl, "bottleneck": bottleneck, "trace": {i: e.trace[-4000:] for i, e in sim.engines.items()}, "jobs": [{k: j.get(k) for k in ("id", "workload", "kind", "t_arrive", "t_end", "ok", "outcome", "steps")} for j in jobs][:2000], "economics": {k: v for k, v in eco.items() if k != "nodes"}, "api_cost": apis, "draws": dict(sim.V.draws)}
