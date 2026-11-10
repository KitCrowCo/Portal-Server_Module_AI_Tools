"""asio/pool.py - the resource pool of one scenario: machines with their numbers read for this run, and the paths between them.
scenario "nodes": {name: catalog id, or {"template": id, ...fields to change}}; "links": [{"type", "a", "b"}] for a direct cable, or {"type", "nodes": [...]} for a switch or hub every member joins.
Rules: a direct link uses a port of its kind on both ends (usb4 ports are counted); a switch needs an eth port, wifi a radio. path(a, b) is the fastest way between two machines for a given message size, through Magpie's own overhead when the scenario says traffic goes through it."""
from asio_core.catalog import merge

class Node:
    def __init__(self, name: str, spec: dict, V):
        self.name, self.spec, self.V = name, spec, V
        k = lambda f: f"node.{name}.{f}"
        self.ram_gb = float(spec["ram_gb"])
        self.reserve_gb = V(spec.get("os_reserve_gb", 2.5), k("os_reserve_gb"))
        self.bw_gbs = V(spec.get("mem_bw_gbs") or {"theory": spec["mem_mts"] * spec["bus_bits"] / 8000}, k("mem_bw_gbs"))
        self.cpu_tflops = V(spec["cpu"]["tflops"], k("cpu_tflops"))
        self.gpu_tflops = V(spec["gpu"]["tflops_fp16"], k("gpu_tflops")) if spec.get("gpu") else 0.0
        self.mem_eff_cpu = V(spec["mem_eff_cpu"], k("mem_eff_cpu")) if spec.get("mem_eff_cpu") else None
        self.throttle = V(spec.get("throttle", 1.0), k("throttle"))
        self.ssd = [V(s["read_gbs"], f"""{k('ssd')}{i}""") for i, s in enumerate(spec.get("ssd", []))]
        self.ssd_gb = sum(s.get("gb", 0) for s in spec.get("ssd", []))
        self.idle_w, self.load_w = V(spec["power_w"]["idle"], k("idle_w")), V(spec["power_w"]["load"], k("load_w"))
        self.price = V(spec.get("price_usd", 0), k("price_usd"))
        r = spec.get("rented")   # a machine paid by the hour: {"usd_per_h", "start_s" (to be ready), "bill_s" (billing step), "from": "start" (rented for the whole run) or "use" (from first use)}
        self.rent = {"usd_h": V(r["usd_per_h"], k("rent_usd_h")), "start_s": V(r.get("start_s", 120), k("rent_start_s")), "bill_s": float(r.get("bill_s", 1)), "from": r.get("from", "use"), "provider": r.get("provider", "")} if r else None
        self.usable_gb = self.ram_gb - self.reserve_gb
        self.used_gb, self.uses = 0.0, []

    def tflops(self, device: str) -> float:
        t = self.gpu_tflops if device == "gpu" and self.gpu_tflops else self.cpu_tflops
        return t * self.throttle

    def ssd_gbs(self, n: int = None) -> float:
        """Reads spread over the first n drives (all by default): colibri splits experts across independent drives."""
        return sum(self.ssd[:n] if n else self.ssd) or 0.0

    def reserve(self, gb: float, what: str):
        self.used_gb += gb
        self.uses.append((what, round(gb, 2)))
        if self.used_gb > self.usable_gb + 1e-9: raise MemoryError(f"""{self.name}: {what} needs {gb:.1f} GB; {self.used_gb:.1f} of {self.usable_gb:.1f} GB usable ({self.ram_gb:g} GB less {self.reserve_gb:.1f} kept for the system). Holding: {self.uses}""")

class Link:
    def __init__(self, i: int, spec: dict, members: list, ltype: dict, V):
        key = f"""link.{spec.get("id") or i}"""
        self.spec, self.members, self.type = spec, members, ltype
        self.rate_gbs = V(ltype["rate_gbps"], f"{key}.rate") / 8
        self.latency_s = V(ltype["latency_us"], f"{key}.latency") / 1e6
        self.busy_s, self.gb, self.messages = 0.0, 0.0, 0
        self.free_at = 0.0

    def cost(self, gb: float) -> float: return self.latency_s + gb / self.rate_gbs

class Pool:
    def __init__(self, scenario: dict, catalog, V):
        self.V, self.cat = V, catalog
        self.nodes = {n: Node(n, self._spec(s), V) for n, s in scenario["nodes"].items()}
        self.links, ports = [], {n: {"usb4": int(nd.spec.get("ports", {}).get("usb4", 0)), "eth": len(nd.spec.get("ports", {}).get("eth_gbps", [])), "wifi": 1 if nd.spec.get("ports", {}).get("wifi") else 0} for n, nd in self.nodes.items()}
        for i, l in enumerate(scenario.get("links", [])):
            lt = catalog.get("links", l["type"])
            members = l.get("nodes") or [l["a"], l["b"]]
            for m in members:
                if m not in self.nodes: raise KeyError(f"link {l} names {m}, which is not a node here")
                need = lt.get("needs_port")
                if need:
                    if ports[m][need] <= 0: raise ValueError(f"""{m} has no free {need} port for link {l["type"]} ({self.nodes[m].spec.get("label", m)})""")
                    if "nodes" not in l: ports[m][need] -= 1   # a cable takes a port; joining a switch or radio network does not use one up
                if lt.get("min_port_gbps") and max(self.nodes[m].spec.get("ports", {}).get("eth_gbps", [0]) or [0]) < lt["min_port_gbps"]: raise ValueError(f"""{m} has no {lt["min_port_gbps"]} Gb/s port for {l["type"]}""")
            self.links.append(Link(i, l, members, lt, V))
        ov = scenario.get("through_magpie")
        self.overhead = Link(-1, {"id": "magpie"}, [], catalog.get("links", ov if isinstance(ov, str) else "magpie_tunnel"), V) if ov else None

    def _spec(self, s) -> dict:
        if isinstance(s, str): return self.cat.get("nodes", s)
        return merge(self.cat.get("nodes", s["template"]), {k: v for k, v in s.items() if k != "template"})

    def path(self, a: str, b: str, gb: float = 1e-5):
        """(link, seconds) for the fastest single hop between a and b for this size, or (None, 0) for the same machine."""
        if a == b: return None, 0.0
        best = min(((l, l.cost(gb)) for l in self.links if a in l.members and b in l.members), key=lambda t: t[1], default=(None, None))
        if best[0] is None: raise ValueError(f"""no link between {a} and {b}: add one to the scenario's links""")
        return best[0], best[1] + (self.overhead.cost(gb) if self.overhead else 0.0)
