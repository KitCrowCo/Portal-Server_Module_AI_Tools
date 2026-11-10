"""asio/sim.py - one run of a scenario: discrete events over the resource pool.
Engines hold their slots and queues; jobs are generator programs (asio/workloads.py) that ask for work and wait for it:
  ("llm", role, {"new", "total", "out", "session"})   prefill and decode on an engine of that role; returns timings and tokens
  ("op", role, {...})                                 one image, video, speech or encoder operation; returns its timings
  ("train", role, {"tokens"})                         a LoRA pass on an LLM engine, holding all its slots
  ("delay", seconds)                                  a tool runs, a person reads
  ("transfer", {"a", "b", "gb"})                      bytes between machines over the best link
  ("spawn", [programs])                               subagents or parallel stages; returns their results when all are done
Time advances in blocks: while an engine's mix of streams does not change, every step costs the same, so many steps are one event (the smallest block that changes anything). A new stream joining a running engine splits the block it lands in."""
import heapq, itertools, math, collections
from asio_core.pool import Pool
from asio_core import perf

class Req:
    __slots__ = ("job", "kind", "spec", "prefill", "decode", "depth", "t_queue", "t_start", "computed", "reused", "out", "eng")
    def __init__(self, job, kind: str, spec: dict, t: float): self.job, self.kind, self.spec, self.t_queue, self.t_start, self.computed, self.reused, self.out, self.eng = job, kind, spec, t, None, 0, 0, 0, None

class Engine:
    def __init__(self, sim, spec: dict):
        cat, V = sim.cat, sim.V
        self.sim, self.spec, self.id = sim, spec, spec["id"]
        self.model = cat.get("models", spec["model"])
        self.backend = cat.get("backends", spec.get("backend") or self.model["backends"][0])
        nodes = [sim.pool.nodes[n] for n in (spec.get("nodes") or [spec["node"]])]
        quants = cat.get("models", "quants")["table"]
        self.phys = perf.LLM(spec, self.model, self.backend, nodes, sim.pool, V, quants) if self.model["kind"] == "llm" else perf.Op(spec, self.model, self.backend, nodes, sim.pool, V, cat)
        self.nodes, self.slots, self.llm = nodes, self.phys.slots, self.model["kind"] == "llm"
        self.queue, self.active = collections.deque(), []
        self.kv = collections.OrderedDict()   # session -> depth held, most recent last
        self.keep = int(spec.get("kv_keep", self.slots)) if self.llm else 0
        self.ready_at = 0.0 if spec.get("warm", True) else self.phys.load_s()
        self.spin_s = max([n.rent["start_s"] for n in nodes if n.rent] or [0.0])
        self.overflow, self.used = (float(spec["overflow"]) if spec.get("overflow") is not None else None), False   # overflow: takes work only once the other engines on its route have this load (1 = every slot busy)
        if self.overflow is None: self.ready_at += self.spin_s   # rented for the whole run: ready once started
        self.block, self.gen, self.trace = None, 0, []   # trace: [t0, t1, limiter, streams], neighbours with the same limiter and streams merged
        self.stats = {"busy_s": 0.0, "by_limit": collections.Counter(), "tokens_out": 0, "prefill_computed": 0, "prefill_reused": 0, "ops": 0, "wait_s": 0.0, "max_queue": 0, "load_s": self.ready_at, "steps": 0, "decode_stream_s": 0.0, "prefill_s": 0.0}

    def load(self) -> float: return (len(self.queue) + len(self.active)) / self.slots

    def submit(self, r: Req):
        r.eng = self
        self.queue.append(r)
        self.stats["max_queue"] = max(self.stats["max_queue"], len(self.queue))
        if self.block and len(self.active) < self.slots: self._interrupt()
        elif not self.block: self.sim.at(max(self.sim.t, self.ready_at), self._schedule)

    # --- Blocks ---

    def _admit(self):
        cap = self.slots if self.llm else 1   # single operations run one after another; parallel media work is more engines
        while self.queue and len(self.active) < cap and (self.queue[0].kind != "train" or not self.active):
            r = self.queue.popleft()
            r.t_start = self.sim.t
            self.stats["wait_s"] += r.t_start - r.t_queue
            if r.kind == "llm":
                s, total, new = r.spec.get("session"), r.spec.get("total", r.spec["new"]), r.spec["new"]
                held = self.kv.get(s, 0) if s else 0
                use = held if held and held >= total - new else 0   # a session's earlier context is reused only if this engine still holds all of it
                r.prefill, r.decode, r.depth, r.reused = total - use, r.spec["out"], use, use
                if s: self.kv.pop(s, None)
            self.active.append(r)
            if r.kind == "train": break

    def _steal(self):
        """A ready engine with free slots takes waiting work from engines it shares a route with (oldest first), as a shared queue would: a rented machine still starting gives its queue back to local ones that free up, and takes theirs once it is ready. Work tied to a session's held context stays put."""
        free = self.slots - len(self.queue) - len(self.active)
        for sib in sorted(self.sim.siblings(self), key=lambda e: -len(e.queue)):
            while free > 0 and sib.queue and not sib.queue[0].spec.get("session"):
                r = sib.queue.popleft()
                r.eng = self
                self.queue.append(r)
                free -= 1

    def _schedule(self):
        if self.block or self.sim.t < self.ready_at: return
        self._steal()
        self._admit()
        if not self.active: return
        self.gen += 1
        g = self.gen
        if not self.llm or self.active[0].kind == "train":
            r = self.active[0] if self.active[0].kind == "train" else min(self.active, key=lambda r: r.t_start)
            dur, lim = self.phys.train_s(r.spec["tokens"], r.spec.get("factor", 3.0)) if r.kind == "train" else self.phys.duration(r.spec)
            self.block = {"t0": self.sim.t, "n": 1, "step": dur, "lim": lim, "reqs": [r], "op": True}
            self.sim.at(self.sim.t + dur, lambda: self._end(g))
            return
        dec = [r for r in self.active if r.prefill <= 0]
        pre = next((r for r in self.active if r.prefill > 0), None)
        chunk = min(perf.UBATCH, pre.prefill) if pre else 0
        step, lim, mult = self.phys.step([r.depth for r in dec], chunk, pre.depth if pre else 0)
        n = min([math.ceil(r.decode / mult) for r in dec] + ([math.ceil(pre.prefill / perf.UBATCH)] if pre else []) + [256])
        self.block = {"t0": self.sim.t, "n": max(n, 1), "step": step, "lim": lim, "dec": dec, "pre": pre, "mult": mult, "op": False}
        self.sim.at(self.sim.t + max(n, 1) * step, lambda: self._end(g))

    def _apply(self, steps: int):
        b = self.block
        dt = steps * b["step"]
        self.stats["busy_s"] += dt
        if dt > 0: self.stats["first_s"], self.stats["last_s"] = min(self.stats.get("first_s", b["t0"]), b["t0"]), max(self.stats.get("last_s", 0.0), b["t0"] + dt)
        self.stats["by_limit"][b["lim"]] += dt
        self.stats["steps"] += steps
        if dt > 0:
            nst = 1 if b["op"] else len(b["dec"]) + (1 if b["pre"] else 0)
            tr = self.trace
            if tr and tr[-1][2] == b["lim"] and tr[-1][3] == nst and abs(tr[-1][1] - b["t0"]) < 1e-6: tr[-1][1] = b["t0"] + dt
            else: tr.append([b["t0"], b["t0"] + dt, b["lim"], nst])
        if b["op"]: return
        self.stats["decode_stream_s"] += dt * len(b["dec"])
        if b["pre"]: self.stats["prefill_s"] += dt
        for r in b["dec"]:
            got = min(r.decode, steps * b["mult"])
            r.decode -= got
            r.depth += got
            r.out += got
            self.stats["tokens_out"] += got
        if b["pre"]:
            got = min(b["pre"].prefill, steps * perf.UBATCH)
            b["pre"].prefill -= got
            b["pre"].depth += got
            b["pre"].computed += got
            self.stats["prefill_computed"] += got

    def _interrupt(self):
        b = self.block
        done = int((self.sim.t - b["t0"]) / b["step"]) if b["step"] > 0 else 0
        if b["op"] or done >= b["n"]: return   # an operation runs to its end; a block about to end ends on its own
        self._apply(done)
        self.block, self.gen = None, self.gen + 1
        self._schedule()

    def _end(self, g: int):
        if g != self.gen or not self.block: return
        b = self.block
        self._apply(b["n"])
        self.block = None
        finished = b["reqs"] if b["op"] else [r for r in self.active if r.prefill <= 0 and r.decode <= 0]
        for r in finished:
            self.active.remove(r)
            if r.kind == "llm":
                self.stats["prefill_reused"] += r.reused
                if r.spec.get("session") and self.keep:
                    self.kv[r.spec["session"]] = r.depth
                    while len(self.kv) > self.keep: self.kv.popitem(last=False)
            else: self.stats["ops"] += 1
            self.sim.resume(r.job, {"engine": self.id, "t_queue": r.t_queue, "t_start": r.t_start, "t_end": self.sim.t, "wait_s": r.t_start - r.t_queue, "out": r.out, "computed": r.computed, "reused": r.reused})
        self._schedule()

class Sim:
    def __init__(self, scenario: dict, catalog, V, seed: int = 0):
        self.sc, self.cat, self.V, self.seed = scenario, catalog, V, seed
        self.t, self.events, self.seq = 0.0, [], itertools.count()
        self.pool = Pool(scenario, catalog, V)
        self.engines = {e["id"]: Engine(self, e) for e in scenario["engines"]}
        self.routes = scenario.get("routes") or {}
        for role, ids in self.routes.items():
            missing = [i for i in ids if i not in self.engines]
            if missing: raise KeyError(f"route {role} names engines that do not exist: {missing}")
        self.jobs, self.done, self.sticky = {}, [], {}
        self.inflight, self.held = collections.Counter(), collections.defaultdict(collections.deque)   # per workload with max_in_flight: jobs running, jobs waiting their turn
        self.transfers = collections.Counter()

    def at(self, t: float, fn): heapq.heappush(self.events, (t, next(self.seq), fn))

    def siblings(self, e: Engine) -> list:
        """Engines that share a route with e."""
        if not hasattr(self, "_sib"):
            self._sib = {i: set() for i in self.engines}
            for ids in self.routes.values():
                for i in ids:
                    if i in self._sib: self._sib[i] |= {self.engines[x] for x in ids if x != i and x in self.engines}
        return list(self._sib[e.id])

    def engine_for(self, role: str, spec: dict) -> Engine:
        ids = self.routes.get(role) or ([role] if role in self.engines else None)
        if not ids: raise KeyError(f"no engine serves role {role}: add it to routes")
        s = spec.get("session")
        if s and self.sticky.get(s) in ids: return self.engines[self.sticky[s]]
        es = [self.engines[i] for i in ids]
        base = [e for e in es if e.overflow is None] or es
        e = min(base, key=lambda e: (e.load(), e.ready_at))
        spill = [x for x in es if x.overflow is not None and e.load() >= x.overflow]
        if spill: e = min([e] + spill, key=lambda e: (e.load(), e.ready_at))
        if e.overflow is not None and not e.used: e.ready_at, e.used, e.t_req = max(e.ready_at, self.t + e.spin_s), True, self.t   # a rented machine is asked for when first needed, and is ready spin_s later
        e.used = True
        if s: self.sticky[s] = e.id
        return e

    # --- Jobs ---

    def admit(self, job: dict, prog):
        """A top-level job arrives; a workload with max_in_flight ("limit") holds it until one of its running jobs ends. Its latency counts from arrival."""
        job.setdefault("t_arrive", self.t)
        lim = job.get("limit")
        if lim and self.inflight[job["workload"]] >= lim: return self.held[job["workload"]].append((job, prog))
        if lim: self.inflight[job["workload"]] += 1
        self.start(job, prog)

    def start(self, job: dict, prog):
        job.update(t_arrive=job.get("t_arrive", self.t), t_start=self.t, prog=prog, waiting=0, results=None)
        self.jobs[job["id"]] = job
        self.resume(job, None)

    def resume(self, job: dict, value):
        try: cmd = job["prog"].send(value)
        except StopIteration as e:
            job.update({k: v for k, v in (e.value or {}).items() if k not in ("id", "t_arrive", "t_end", "parent", "prog")}, t_end=self.t)
            self.done.append(job)
            if job.get("parent"): self._child_done(job)
            elif job.get("limit"):
                self.inflight[job["workload"]] -= 1
                if self.held[job["workload"]]: self.admit(*self.held[job["workload"]].popleft())
            return
        kind = cmd[0]
        if kind in ("llm", "op", "train"):
            eng = self.engine_for(cmd[1], cmd[2])
            eng.submit(Req(job, kind, cmd[2], self.t))
        elif kind == "delay": self.at(self.t + max(cmd[1], 0.0), lambda: self.resume(job, None))
        elif kind == "transfer":
            link, secs = self.pool.path(cmd[1]["a"], cmd[1]["b"], cmd[1]["gb"])
            start = max(self.t, link.free_at) if link else self.t
            if link:
                link.free_at, link.busy_s, link.gb, link.messages = start + secs, link.busy_s + secs, link.gb + cmd[1]["gb"], link.messages + 1
            self.at(start + secs, lambda: self.resume(job, {"seconds": start + secs - self.t}))
        elif kind == "spawn":
            kids = cmd[1]
            job["waiting"], job["results"] = len(kids), []
            if not kids: return self.at(self.t, lambda: self.resume(job, []))
            for i, (spec, prog) in enumerate(kids): self.start({**spec, "id": f"""{job["id"]}.{i}""", "parent": job["id"]}, prog)
        else: raise ValueError(f"unknown command {kind}")

    def _child_done(self, kid: dict):
        p = self.jobs[kid["parent"]]
        p["results"].append(kid)
        p["waiting"] -= 1
        if p["waiting"] == 0: self.resume(p, p["results"])

    def run(self, arrivals: list, horizon_s: float = None) -> "Sim":
        """arrivals: [(t, job spec, program)]."""
        for t, spec, prog in arrivals: self.at(t, lambda s=spec, p=prog: self.admit(s, p))
        while self.events:
            t, _, fn = heapq.heappop(self.events)
            if horizon_s is not None and t > horizon_s: break
            self.t = t
            fn()
        self.unfinished = [j for j in self.jobs.values() if "t_end" not in j] + [j for q in self.held.values() for j, _ in q]
        return self
