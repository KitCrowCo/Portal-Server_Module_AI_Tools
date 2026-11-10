"""asio/workloads.py - kinds of work, as small programs the simulator runs (see asio/sim.py for the commands they yield). A scenario lists workloads; each has a kind, a role (which engines serve it), a count and an arrival pattern.
Quantities can be numbers or est forms ([lo, mode, hi], {"normal": ...}): drawn per job in Monte Carlo runs, their median otherwise. Outcomes (a tool call that fails, a wrong route) are always drawn, from each job's own seeded stream.
Kinds: stream (one request; batch and benchmark are the same with a fixed set in mind), mix (demand: job types by share at a rate with a daily profile), chat (turns in one session), agent (a tool loop, with subagents), routed (a decision model picks the engine), pipeline (stages with dependencies), image, video, stt, tts, encode, train."""
import zlib, collections
import numpy as np
from asio_core.values import _q

def draw(spec, rng, sample: bool) -> float:
    if spec is None or isinstance(spec, (int, float)): return spec
    return float(_q(spec, rng.random() if sample else 0.5))

class Ctx:
    def __init__(self, sim, w: dict, i: int, sample: bool):
        self.sim, self.w, self.i, self.sample = sim, w, i, sample
        self.rng = np.random.default_rng([sim.seed, zlib.crc32(f"""{w["name"]}.{i}""".encode())])
    def q(self, key: str, default=None): return draw(self.w.get(key, default), self.rng, self.sample)
    def n(self, key: str, default=None) -> int: return max(int(round(self.q(key, default) or 0)), 0)
    def p(self, x: float) -> bool: return self.rng.random() < x
    def agent(self, role: str) -> dict:
        """The agentic outcome rates of the model behind a role."""
        e = self.sim.engines[self.sim.routes.get(role, [role])[0]]
        a = e.model.get("agent") or {}
        return {k: self.sim.V(a[k], f"""model.{e.model["id"]}.agent.{k}""") for k in ("p_valid", "p_progress", "p_recover")}

def _llm(c: Ctx, role: str, new: int, out: int, total: int = None, session: str = None):
    r = yield ("llm", role, {"new": new, "total": total or new, "out": out, "session": session})
    return r

# --- Kinds ---

def stream(c: Ctx):
    """One request. "items" ([[prompt, out], ...] or [{"prompt", "out"}, ...]) gives each job its own sizes in turn, as a benchmark's set does."""
    it = c.w["items"][c.i % len(c.w["items"])] if c.w.get("items") else None
    if it is not None: p, o = (it["prompt"], it["out"]) if isinstance(it, dict) else it
    r = yield from _llm(c, c.w["role"], int(p) if it is not None else c.n("prompt", 1000), int(o) if it is not None else c.n("out", 500))
    return {"ok": True, "tokens_out": r["out"], "computed": r["computed"], "reused": r["reused"], "wait_s": r["wait_s"]}

def chat(c: Ctx):
    sid, ctx, out_n, comp, reuse, wait = f"""s.{c.w["name"]}.{c.i}""", c.n("system", 0), 0, 0, 0, 0.0
    for t in range(c.n("turns", 5)):
        new = c.n("prompt", 200) + (ctx if t == 0 else 0)
        r = yield from _llm(c, c.w["role"], new if t == 0 else c.n("prompt", 200), c.n("out", 300), ctx + c.n("prompt", 200) if t else new, sid if c.w.get("session", True) else None)
        ctx, out_n, comp, reuse, wait = r["reused"] + r["computed"] + r["out"], out_n + r["out"], comp + r["computed"], reuse + r["reused"], wait + r["wait_s"]
        yield ("delay", c.q("think_s", 20))
    return {"ok": True, "tokens_out": out_n, "computed": comp, "reused": reuse, "wait_s": wait}

def agent(c: Ctx, w: dict = None, depth: int = 0):
    """A tool loop. Each step: the model reads what is new and writes a step; a valid tool call runs the tool, and a valid step moves the task on with p_progress.
    A bad step costs a step; after one, the next is good with p_recover, else the stuck count grows. The task ends done, stuck (max_stuck bad in a row) or out of steps.
    subagents {"every": steps, "n", "role", plus any agent fields}: every that many good steps, n subtasks run in parallel and their summaries join the context."""
    w = w or c.w
    role, a = w["role"], c.agent(w["role"])
    sid = f"""a.{w["name"]}.{c.i}.{depth}.{c.rng.integers(1 << 30)}""" if w.get("kv", True) else None
    need, ctx, steps, good, stuck, bad_last = max(int(round(draw(w.get("steps_needed", [5, 10, 20]), c.rng, c.sample))), 1), int(draw(w.get("system", 4000), c.rng, c.sample)), 0, 0, 0, False
    tot = {"tokens_out": 0, "computed": 0, "reused": 0, "wait_s": 0.0, "llm_calls": 0, "bad_steps": 0, "subtasks": 0, "tool_s": 0.0}
    pending = ctx
    while good < need and steps < w.get("max_steps", 60) and stuck < w.get("max_stuck", 4):
        out = int(draw(w.get("step_out", [150, 400, 1200]), c.rng, c.sample))
        r = yield ("llm", role, {"new": pending, "total": ctx, "out": out, "session": sid})
        steps, tot["llm_calls"], ctx = steps + 1, tot["llm_calls"] + 1, ctx + out
        for k in ("computed", "reused"): tot[k] += r[k]
        tot["tokens_out"] += r["out"]
        tot["wait_s"] += r["wait_s"]
        valid = c.p(a["p_recover"] if bad_last else a["p_valid"])
        if not valid:
            tot["bad_steps"] += 1
            stuck, bad_last, pending = stuck + 1, True, int(draw(w.get("error_in", 80), c.rng, c.sample))
            ctx += pending
            continue
        stuck, bad_last = 0, False
        ts = draw(w.get("tool_s", [0.5, 2, 15]), c.rng, c.sample)
        tot["tool_s"] += ts
        yield ("delay", ts)
        pending = int(draw(w.get("tool_out", [100, 600, 3000]), c.rng, c.sample))
        if c.p(a["p_progress"]):
            good += 1
            sub = w.get("subagents")
            if sub and depth < 2 and good % sub.get("every", 5) == 0:
                kids = [({"kind": "agent", "role": sub["role"]}, agent(c, {**w, "max_steps": 30, "kv": True, **sub, "subagents": sub.get("subagents"), "name": f"""{w["name"]}.sub"""}, depth + 1)) for _ in range(sub.get("n", 2))]
                res = yield ("spawn", kids)
                tot["subtasks"] += len(res)
                pending += sum(int(draw(sub.get("summary_out", 300), c.rng, c.sample)) for _ in res)
                for k in ("tokens_out", "computed", "reused"): tot[k] += sum(x.get(k, 0) for x in res)
        ctx += pending
    return {"ok": good >= need, "outcome": "done" if good >= need else "stuck" if stuck >= w.get("max_stuck", 4) else "out of steps", "steps": steps, **tot}

def routed(c: Ctx):
    """A decision model reads the request and picks a route. With its accuracy it picks the right one; a hard task sent to the easy route fails with fail_if_wrong and is redone on the hard one."""
    dr = c.w["decide_role"]
    e = c.sim.engines[c.sim.routes[dr][0]]
    acc = c.sim.V(e.model["decide"]["accuracy"], f"""model.{e.model["id"]}.decide.accuracy""") if e.model.get("decide") else 0.85
    r0 = yield from _llm(c, dr, c.n("prompt", 800), 2)
    hard = c.p(c.w.get("p_hard", 0.3))
    right = c.p(acc)
    pick = ("hard" if hard else "easy") if right else ("easy" if hard else "hard")
    rt = c.w["routes"]
    r = yield from _llm(c, rt[pick]["role"], c.n("prompt", 800), int(draw(rt[pick].get("out", 400), c.rng, c.sample)))
    redo = hard and pick == "easy" and c.p(c.w.get("fail_if_wrong", 0.6))
    if redo: r2 = yield from _llm(c, rt["hard"]["role"], c.n("prompt", 800), int(draw(rt["hard"].get("out", 400), c.rng, c.sample)))
    outs = r["out"] + r0["out"] + (r2["out"] if redo else 0)
    return {"ok": True, "tokens_out": outs, "route": pick, "misrouted": not right, "redone": redo, "computed": r0["computed"] + r["computed"], "reused": 0, "wait_s": r0["wait_s"] + r["wait_s"]}

def _do(c: Ctx, d: dict):
    (k, v), = d.items()
    if k == "llm": return (yield ("llm", v["role"], {"new": int(draw(v.get("new", 1000), c.rng, c.sample)), "total": int(draw(v.get("total", v.get("new", 1000)), c.rng, c.sample)), "out": int(draw(v.get("out", 300), c.rng, c.sample)), "session": v.get("session")}))
    if k == "op": return (yield ("op", v["role"], {kk: draw(vv, c.rng, c.sample) if kk != "role" else vv for kk, vv in v.items()}))
    if k == "transfer": return (yield ("transfer", {"a": v["a"], "b": v["b"], "gb": draw(v["gb"], c.rng, c.sample)}))
    if k == "delay": return (yield ("delay", draw(v, c.rng, c.sample)))
    if k == "train": return (yield ("train", v["role"], {"tokens": int(draw(v["tokens"], c.rng, c.sample)), "factor": v.get("factor", 3.0)}))
    raise ValueError(f"unknown stage action {k}")

def pipeline(c: Ctx):
    """stages: [{"id", "after": [ids], "do": {"llm" | "op" | "transfer" | "delay" | "train": {...}}}]: every stage whose dependencies are done starts at once."""
    stages, done, order = {s["id"]: s for s in c.w["stages"]}, {}, []
    while len(done) < len(stages):
        ready = [s for i, s in stages.items() if i not in done and all(a in done for a in s.get("after", []))]
        if not ready: raise ValueError(f"""pipeline {c.w["name"]}: stages wait on each other or on unknown ids: {sorted(set(stages) - set(done))}""")
        def one(s):
            r = yield from _do(c, s["do"])
            return {"stage": s["id"], **(r or {})}
        res = yield ("spawn", [({"kind": "stage", "stage": s["id"]}, one(s)) for s in ready])
        for s, r in zip(ready, res): done[s["id"]] = r
        order.append([s["id"] for s in ready])
    return {"ok": True, "waves": order, "tokens_out": sum(r.get("out", 0) for r in done.values())}

def _op(kind):
    def prog(c: Ctx):
        spec = {k: draw(v, c.rng, c.sample) for k, v in c.w.items() if k in ("mp", "steps", "frames", "audio_s", "chars")}
        r = yield ("op", c.w["role"], spec)
        return {"ok": True, "op": kind, "wait_s": r["wait_s"], "seconds": r["t_end"] - r["t_start"]}
    return prog

def train(c: Ctx):
    r = yield ("train", c.w["role"], {"tokens": c.n("tokens", 200000), "factor": c.w.get("factor", 3.0)})
    return {"ok": True, "wait_s": r["wait_s"], "seconds": r["t_end"] - r["t_start"]}

KINDS = {"stream": stream, "batch": stream, "benchmark": stream, "chat": chat, "agent": agent, "routed": routed, "pipeline": pipeline, "image": _op("image"), "video": _op("video"), "stt": _op("stt"), "tts": _op("tts"), "encode": _op("encode"), "train": train}

def _spec(w: dict, i: int) -> dict: return {"id": f"""{w["name"]}.{i}""", "kind": w["kind"], "workload": w["name"], "role": w.get("role"), "limit": w.get("max_in_flight"), "slo": w.get("slo_s")}

def mix(sim, w: dict, sample: bool) -> list:
    """Demand as a mix: jobs arrive at random at rate_per_h (times the hour's profile value, 24 numbers from midnight, starting at start_hour) for duration_h,
    and each is one of "types" by share. A type is a workload of its own (kind, role and that kind's fields) and is reported as <mix name>.<type name>."""
    rng = np.random.default_rng([sim.seed, zlib.crc32(w["name"].encode())])
    prof = w.get("profile") or [1.0] * 24
    rate, hours, h0 = draw(w.get("rate_per_h", 10), rng, sample), draw(w.get("duration_h", 24), rng, sample), float(w.get("start_hour", 0))
    types = [{**t, "name": f"""{w["name"]}.{t.get("name") or t["kind"]}""", "role": t.get("role", w.get("role"))} for t in w["types"]]
    share = np.asarray([float(t.get("share", 1)) for t in types])
    peak, t, out, n = rate * max(prof), float(w.get("start_s", 0)), [], collections.Counter()
    while peak > 0:
        t += rng.exponential(3600 / peak)
        if t > float(w.get("start_s", 0)) + hours * 3600: break
        if rng.random() > rate * prof[int(h0 + t / 3600) % 24] / peak: continue
        tw = types[rng.choice(len(types), p=share / share.sum())]
        if tw["kind"] not in KINDS or tw["kind"] == "mix": raise ValueError(f"""mix {w["name"]}: unknown kind {tw["kind"]} for a type""")
        out.append((t, _spec(tw, n[tw["name"]]), KINDS[tw["kind"]](Ctx(sim, tw, n[tw["name"]], sample))))
        n[tw["name"]] += 1
    return out

def arrivals(sim, workloads: list, sample: bool) -> list:
    """[(t, job spec, program)] for every job of every workload. arrival: "all" (t=0), {"every_s"}, {"poisson_per_h"}, {"at": [t, ...]}; "start_s" shifts a workload.
    max_in_flight holds a workload's jobs until fewer than that many of them are running (a benchmark client, a team of agents); slo_s marks jobs that should end within that many seconds of arriving."""
    out = []
    for wi, w in enumerate(workloads):
        w = {"name": w.get("name") or f"""w{wi}.{w["kind"]}""", **w}
        if w["kind"] == "mix":
            out += mix(sim, w, sample)
            continue
        if w["kind"] not in KINDS: raise ValueError(f"""unknown workload kind {w["kind"]} (known: {sorted(KINDS) + ["mix"]})""")
        rng, arr, t = np.random.default_rng([sim.seed, zlib.crc32(w["name"].encode())]), w.get("arrival", "all"), float(w.get("start_s", 0))
        for i in range(int(w.get("count", len(w["items"]) if w.get("items") else 1))):
            if isinstance(arr, dict) and "at" in arr: t = float(arr["at"][i % len(arr["at"])])
            elif isinstance(arr, dict) and "every_s" in arr: t = float(w.get("start_s", 0)) + i * arr["every_s"]
            elif isinstance(arr, dict) and "poisson_per_h" in arr: t += rng.exponential(3600 / arr["poisson_per_h"]) if i or arr.get("first_wait") else 0.0
            c = Ctx(sim, w, i, sample)
            out.append((t, _spec(w, i), KINDS[w["kind"]](c)))
    return out
