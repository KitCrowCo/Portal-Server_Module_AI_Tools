"""asio/run.py - running scenarios: one run, the named cases side by side, a Monte Carlo ensemble with what moves the results, a sweep over one value, and goal seeking with Athene over a search space.
A scenario is a dict or a catalog id. "overrides" ({value key or glob: value}) change any number for a what-if; the keys a run read are in its result's "draws"."""
import copy, math, importlib

ATHENE = None   # the Athene library as its host provides it (the portal passes ENV["tools"]["athene"]); None imports a package named athene

def use_athene(mod):
    """Set the Athene library to use: the portal's tool module, or a package loaded any other way."""
    global ATHENE
    ATHENE = mod

def athene():
    if ATHENE is None: use_athene(importlib.import_module("athene"))
    return ATHENE

from asio_core.catalog import Catalog
from asio_core.values import Valuer, CASES
from asio_core.sim import Sim
from asio_core.workloads import arrivals
from asio_core import report

def scenario(cat: Catalog, sc) -> dict: return copy.deepcopy(cat.get("scenarios", sc) if isinstance(sc, str) else sc)

def once(cat: Catalog, sc, case: str = "median", seed: int = 0, tier: str = "auto", overrides: dict = None) -> dict:
    """One run. A plan that cannot exist (not enough memory, a missing link or port) comes back as {"error": why}, not an exception, so a search can score it."""
    try: sc = scenario(cat, sc)
    except KeyError as e: return {"error": f"KeyError: {e}", "metrics": {"error": str(e)}, "draws": {}}
    V = Valuer(case, tier, seed, {**sc.get("overrides", {}), **(overrides or {})})
    try:
        sim = Sim(sc, cat, V, seed)
        sim.run(arrivals(sim, sc["workloads"], case == "sample"), sc.get("horizon_s"))
    except (MemoryError, ValueError, KeyError) as e: return {"error": f"{type(e).__name__}: {e}", "metrics": {"error": str(e)}, "draws": dict(V.draws)}
    return {**report.build(sim, cat.get("economics", sc.get("economics", "default"))), "case": case, "seed": seed, "tier": tier}

def cases(cat: Catalog, sc, seeds: int = 4, tier: str = "auto", which=CASES[1:], overrides: dict = None) -> dict:
    """Each named case with its values fixed (median, worst, best, theory, improved, plateau); outcomes still vary by seed, so each case is the mean of a few seeds."""
    out = {}
    for c in which:
        runs = [once(cat, sc, c, s, tier, overrides) for s in range(seeds)]
        bad = next((r for r in runs if r.get("error")), None)
        out[c] = {"error": bad["error"]} if bad else {"metrics": _mean([r["metrics"] for r in runs]), "first": runs[0]}
    return out

def _mean(ms: list) -> dict:
    keys = {k for m in ms for k, v in m.items() if isinstance(v, (int, float)) and not isinstance(v, bool)}
    return {k: sum(m[k] for m in ms) / len(ms) for k in keys if all(isinstance(m.get(k), (int, float)) for m in ms)}

def monte_carlo(cat: Catalog, sc, n: int = 64, tier: str = "auto", metrics: list = None, overrides: dict = None) -> dict:
    """n runs, every value drawn. Returns the summary per metric, the ranked inputs that move each chosen metric, the runs' metrics, and the values each drew (inputs, in the same order)."""
    summarize, sensitivity = athene().summarize, athene().sensitivity
    runs = [once(cat, sc, "sample", s, tier, overrides) for s in range(n)]
    good = [r for r in runs if not r.get("error")]
    ms = [r["metrics"] for r in good]
    fin = lambda m: [r for r in good if isinstance(r["metrics"].get(m), (int, float)) and math.isfinite(r["metrics"][m])]
    sens = {m: sensitivity([r["draws"] for r in fin(m)], [r["metrics"][m] for r in fin(m)]) for m in (metrics or ["span_s", "cost_per_ok_usd", "tps_total"]) if len(fin(m)) >= 8}
    return {"summary": summarize(ms), "sensitivity": sens, "runs": ms, "inputs": [r["draws"] for r in good], "errors": sorted({r["error"] for r in runs if r.get("error")}), "n": n}

# --- Quotes with rented machines ---

def with_rented(cat: Catalog, sc, mirror: list, rented: str, count: int = 1, overflow_at: float = None, from_: str = "use") -> dict:
    """The scenario with count rented machines of catalog type rented, each running a copy of every engine in mirror, added to the same routes.
    overflow_at: the copies take work only once the local engines on their route have that load (1 = every slot busy), and are rented from first use; None: rented for the whole run and used like any engine."""
    out, node = scenario(cat, sc), cat.get("nodes", rented)
    backend = "llama_cuda" if (node.get("gpu") or {}).get("vendor") == "nvidia" else "llama_cpu"
    for k in range(count):
        name = f"rent{k + 1}"
        out.setdefault("nodes", {})[name] = {"template": rented, "rented": {**node["rented"], "from": "start" if overflow_at is None else from_}}
        for eid in mirror:
            e = next(x for x in out["engines"] if x["id"] == eid)
            m = cat.get("models", e["model"])
            copy_ = {**{a: b for a, b in e.items() if a not in ("node", "nodes", "placement", "cache_gb", "heat", "arrangement")}, "id": f"{eid}-{name}", "node": name, "backend": backend if m.get("kind") == "llm" else e.get("backend")}
            if overflow_at is not None: copy_["overflow"] = overflow_at
            out["engines"].append(copy_)
            routes = out.setdefault("routes", {})
            hit = [r for r, ids in routes.items() if eid in ids]
            for r in hit: routes[r] = routes[r] + [copy_["id"]]
            if eid not in routes and not hit: routes[eid] = [eid, copy_["id"]]
    return out

def quote(cat: Catalog, sc, mirror: list, rented: str, count: int = 1, overflow_at: float = 1.0, fee_usd: float = 0.0, margin: float = 0.15, n: int = 24, tier: str = "auto") -> dict:
    """Three ways to run the same work, each from n Monte Carlo runs: local only; local with rented machines allowed as overflow; rented machines planned from the start.
    For each: time to finish (median, 1 in 20 worst), rented cost (median, 1 in 20 worst), a ceiling the rented cost should not pass without asking (the 95th percentile plus margin), and the arrangement fee as its own line."""
    opts = {"local": scenario(cat, sc), "overflow": with_rented(cat, sc, mirror, rented, count, overflow_at), "planned": with_rented(cat, sc, mirror, rented, count, None)}
    out = {}
    for k, s in opts.items():
        mc = monte_carlo(cat, s, n, tier, ["span_s", "cost_rent_usd"])
        sm = mc["summary"]
        g = lambda m, p: (sm.get(m) or {}).get(p)
        rent95 = g("cost_rent_usd", "p95") or 0.0
        out[k] = {"span_p50": g("span_s", "p50"), "span_p95": g("span_s", "p95"), "rent_p50": g("cost_rent_usd", "p50") or 0.0, "rent_p95": rent95, "ceiling": math.ceil(rent95 * (1 + margin) * 2) / 2 if rent95 else 0.0,
                  "own_p50": (g("cost_usd", "p50") or 0.0) - (g("cost_rent_usd", "p50") or 0.0), "ok_p50": g("ok", "p50"), "jobs": g("jobs", "p50"), "fee": 0.0 if k == "local" else fee_usd, "errors": mc["errors"], "failed": n - len(mc["runs"]), "n": n, "sensitivity": mc["sensitivity"]}
        out[k]["total_p50"] = out[k]["own_p50"] + out[k]["rent_p50"] + out[k]["fee"]
    return {"options": out, "rented": rented, "count": count, "mirror": mirror, "overflow_at": overflow_at, "margin": margin}

def sweep(cat: Catalog, sc, key: str, values: list, case: str = "median", seeds: int = 2) -> list:
    """One value set to each of values in turn, everything else as the case says: [{"value", "metrics"}]."""
    out = []
    for v in values:
        rs = [once(cat, sc, case, s, overrides={key: v}) for s in range(seeds)]
        out.append({"value": v, "error": next((r["error"] for r in rs if r.get("error")), None), "metrics": _mean([r["metrics"] for r in rs if not r.get("error")]) if any(not r.get("error") for r in rs) else {}})
    return out

# --- Goal seeking ---

def _set(d: dict, path: str, v):
    """engines.ds.slots, engines.1.node, routes.main, workloads.0.count: dots walk dicts, list items by id or index."""
    parts = path.split(".")
    for p in parts[:-1]:
        d = next((x for x in d if isinstance(x, dict) and x.get("id") == p), None) or d[int(p)] if isinstance(d, list) else d.setdefault(p, {})
    if isinstance(d, list): d[int(parts[-1])] = v
    else: d[parts[-1]] = v

def _gene(g: dict):
    a = athene()
    t = g["type"]
    if t == "int": return a.IntGene(g["path"], g["lo"], g["hi"])
    if t == "float": return a.FloatGene(g["path"], g["lo"], g["hi"], g.get("step"), g.get("log", False))
    if t == "choice": return a.ChoiceGene(g["path"], g["options"])
    if t == "bool": return a.ChoiceGene(g["path"], [True, False])
    if t == "perm": return a.PermGene(g["path"], g["items"])
    if t == "subset": return a.SubsetGene(g["path"], g["items"], g.get("k_min", 0), g.get("k_max"))
    raise ValueError(f"unknown gene type {t}")

class Evaluator:
    """Picklable: a scenario plus a search point -> one sampled run's metrics, for athene.Ensemble."""
    def __init__(self, cat_dirs: list, sc: dict, tier: str = "auto"): self.dirs, self.sc, self.tier, self.cat = cat_dirs, sc, tier, None
    def __call__(self, params: dict, seed: int) -> dict:
        if self.cat is None: self.cat = Catalog(*self.dirs)
        sc = copy.deepcopy(self.sc)
        for path, v in params.items(): _set(sc, path, list(v) if isinstance(v, tuple) else v)
        r = once(self.cat, sc, "sample", seed, self.tier)
        return {"error": r["error"]} if r.get("error") else r["metrics"]

WATCH = ("span_s", "cost_per_ok_usd", "wh_per_ok", "ok", "latency_p90_s", "slo_share", "tps_total", "energy_wh")

class Watched:
    """A cost function that also keeps the mean of watched metrics for every point, for trade-off views; they add nothing to the cost."""
    def __init__(self, cost, watch): self.cost, self.watch = cost, watch
    def __call__(self, results: list) -> tuple:
        c, vals = self.cost(results)
        if results and not any(r.get("error") for r in results):
            for m in self.watch:
                xs = [r[m] for r in results if isinstance(r.get(m), (int, float)) and not isinstance(r.get(m), bool)]
                if xs and f"{m}:mean" not in vals: vals[f"{m}:mean"] = float(sum(xs) / len(xs))
        return c, vals
    def describe(self) -> list: return self.cost.describe()

def optimize(cat: Catalog, sc, space: list, goals: list, method: str = "ga", runs: int = 8, budget: int = 400, seed: int = 0, workers: int = 0, watch=WATCH, **kw) -> dict:
    """space: [{"path", "type": int | float | choice | bool | perm | subset, ...}]; goals: athene Goal specs over the run metrics. Every point is scored on the same runs (common random numbers).
    The result has every point scored ("points", each with the means of the watch metrics too) and "base", the scenario the paths apply to."""
    a = athene()
    sc = scenario(cat, sc)
    pool, cost = a.GenePool([_gene(g) for g in space]), Watched(a.CostFunction.from_spec(goals), watch)
    ens = a.Ensemble(Evaluator([str(d) for d in cat.dirs[1:]], sc), n=runs, seed=seed, workers=workers)
    cls = {"ga": a.GA, "tabu": a.Tabu, "anneal": a.Anneal}[method]
    s = cls(pool, cost, ens, seed=seed, budget=budget)
    res = s.run(**kw)
    ens.close()
    best = copy.deepcopy(sc)
    for path, v in (res["best"] or {}).get("params", {}).items(): _set(best, path, list(v) if isinstance(v, tuple) else v)
    return {**res, "scenario": best, "base": sc, "space": pool.describe(), "goals": cost.describe()}

def apply(sc: dict, params: dict) -> dict:
    """A copy of the scenario with a search point's settings in place."""
    out = copy.deepcopy(sc)
    for path, v in params.items(): _set(out, path, list(v) if isinstance(v, tuple) else v)
    return out
