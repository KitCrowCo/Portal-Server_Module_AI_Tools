"""asio/perf.py - how long one step of work takes on a placement of machines, from bytes moved and arithmetic done (a roofline: whichever runs out first sets the time), plus fixed costs.
LLM engines are stepped: every step serves all decoding streams at once (one read of the weights for all of them: why batching is nearly free until compute binds) and up to one prefill chunk. Media and speech are single operations with a duration.
Every function returns (seconds, limiter) so a run can say what held each second back: mem, compute, ssd, hop, overhead."""
import math

UBATCH = 512

def _hidden(params_b: float, layers: int) -> float: return math.sqrt(params_b * 1e9 / (12 * max(layers, 1)))

def _limit(parts: dict) -> str: return max(parts, key=parts.get)

class LLM:
    """placement: single (one node), experts (MoE experts spread, the first node holds the dense part), pipeline (layers split by free memory)."""
    def __init__(self, eng: dict, model: dict, backend: dict, nodes: list, pool, V, quants: dict):
        k = lambda f: f"""engine.{eng["id"]}.{f}"""
        self.id, self.model, self.nodes, self.pool, self.V = eng["id"], model, nodes, pool, V
        self.placement = eng.get("placement", "single" if len(nodes) == 1 else "experts" if model["arch"] == "moe" else "pipeline")
        self.moe = model["arch"] == "moe"
        self.device = backend["device"]
        self.mem_eff = [n.mem_eff_cpu if self.device == "cpu" and n.mem_eff_cpu else V(backend["mem_eff"], f"""{k('mem_eff')}.{n.name}""") for n in nodes]
        self.comp_gflops = [n.tflops(self.device) * V(backend["compute_eff"], k("compute_eff")) * 1e3 for n in nodes]
        self.step_s = V(backend["step_ms"], k("step_ms")) / 1e3
        self.tok_over_s = V(backend["token_overhead_ms"], k("token_overhead_ms")) / 1e3 if backend.get("token_overhead_ms") else 0.0
        self.overlap = V(backend["ssd_overlap"], k("ssd_overlap")) if backend.get("ssd_overlap") else 0.0
        self.load_eff = V(backend.get("load_eff", 0.7), k("load_eff"))
        self.slots, self.ctx = int(eng.get("slots", 1)), int(eng.get("ctx", 8192))
        self.kv_gb_tok = V(model["kv_kb_per_token"], f"""model.{model["id"]}.kv_kb""") * 1e3 * model.get("kv_quant_bytes", 1.0625) / 2 / 1e9
        self.state_gb = V(model["state_mb"], f"""model.{model["id"]}.state_mb""") / 1e3 if model.get("state_mb") else 0.0
        mult = model.get("mtp") if eng.get("mtp", True) and model.get("mtp") else model.get("draft") if eng.get("draft", True) else None
        self.mult = V(mult["tokens_per_step"], f"""model.{model["id"]}.tokens_per_step""") if mult else 1.0
        self.verify = 4 if model.get("mtp") and mult is model.get("mtp") else 1   # MTP checks its 3 draft tokens plus one in the same step
        if self.moe: self._moe(eng, k)
        else:
            q = eng.get("quant") or model["default_quant"]
            if q not in quants: raise KeyError(f"""{self.id}: quant {q} is not in the quant table ({sorted(quants)})""")
            self.quant, self.quality = q, quants[q][1]
            self.weights_gb, self.gflop_tok = model["params_b"] * quants[q][0], 2 * model["params_b"]
            self.hidden = _hidden(model["params_b"], model.get("layers", 32))
            self._dense(eng)

    # --- Memory ---

    def kv_gb(self) -> float: return max(self.slots, int(self.eng_keep)) * self.ctx * self.kv_gb_tok + self.slots * self.state_gb

    def _dense(self, eng: dict):
        self.eng_keep = eng.get("kv_keep", self.slots)
        if self.placement == "single": self.share = [1.0]
        else:
            free = [max(n.usable_gb - n.used_gb, 0.01) for n in self.nodes]
            self.share = [f / sum(free) for f in free]
        for n, s in zip(self.nodes, self.share): n.reserve(self.weights_gb * s + self.kv_gb() * s + 0.5, f"{self.id} weights and KV")
        self.resident_gb = self.weights_gb

    def _moe(self, eng: dict, k):
        m, V, mid = self.model, self.V, self.model["id"]
        self.quant, self.quality, self.eng_keep = "native", 1.0, eng.get("kv_keep", self.slots)
        self.dense_gb = V(m["dense_gb_token"], f"model.{mid}.dense_gb_token")
        self.expert_gb = V(m["expert_gb_token"], f"model.{mid}.expert_gb_token")
        self.gflop_tok = V(m["gflop_token"], f"model.{mid}.gflop_token")
        self.alpha = V(m["hit_alpha"], f"model.{mid}.hit_alpha")
        self.E, self.a = m["experts_per_layer"], m["active_experts"]
        self.hidden = 4096
        resident = V(m["resident_gb"], f"model.{mid}.resident_gb")
        self.nodes[0].reserve(resident + self.kv_gb() + 0.5, f"{self.id} dense part and KV")
        want = eng.get("cache_gb") or {}
        self.cache = []
        for n in self.nodes if self.placement == "experts" else self.nodes[:1]:
            c = float(want.get(n.name, max(n.usable_gb - n.used_gb, 0)) if isinstance(want, dict) else want)
            n.reserve(c, f"{self.id} expert cache")
            self.cache.append(c)
        tot = sum(self.cache)
        self.share = [c / tot if tot else 1 / len(self.cache) for c in self.cache]
        self.hit = min(1.0, (tot / m["experts_gb"])) ** self.alpha if tot else 0.0
        self.heat = None
        if eng.get("heat"):   # which experts are chosen and where they live (asio_core/heat.py); without it, the hit rate follows hit_alpha
            from asio_core import heat as H
            cn = self.nodes[:len(self.cache)]
            self.heat = H.Heat(H.table(eng["heat"], m, self.pool.cat, V), eng.get("arrangement", "learned"), cn, self.cache, [n.bw_gbs * eff * n.throttle for n, eff in zip(cn, self.mem_eff)], [n.ssd_gbs() or 0.01 for n in cn], self.expert_gb / (self.a * m.get("moe_layers", m["layers"])), V.seed)
            self.hit, self.share = self.heat.hit, self.heat.load
        self.resident_gb = resident + tot
        missing = [n.name for n in self.nodes[:len(self.cache)] if not n.ssd]
        if missing: raise ValueError(f"{self.id}: {missing} have no SSD to stream experts from")

    def load_s(self) -> float: return 3.0 + self.resident_gb / max(min(n.ssd_gbs() or 0.5 for n in self.nodes) * self.load_eff, 0.05)

    # --- One step ---

    def _hops(self) -> float:
        if len(self.nodes) == 1: return 0.0
        per = max(self.pool.path(self.nodes[0].name, n.name)[1] for n in self.nodes[1:])
        return 2 * self.model.get("moe_layers", 1) * per if self.moe and self.placement == "experts" else (len(self.nodes) - 1) * per

    def step(self, depths: list, chunk: int = 0, chunk_depth: int = 0) -> tuple:
        """depths: context depth of each decoding stream; chunk: prefill tokens in this step (one stream at chunk_depth). (seconds, limiter, tokens per decoding stream)."""
        k = len(depths)
        toks = k * self.verify + chunk
        attn = (chunk * (chunk_depth + chunk / 2) + k * self.verify * (sum(depths) / k if k else 0)) * self.model.get("full_attn_layers", self.model.get("layers", 32)) * 4 * self.hidden / 1e9
        kv = sum(depths) * self.kv_gb_tok + 2 * k * self.state_gb
        if self.moe: return self._moe_step(k, toks, kv, attn)
        parts = {"mem": 0.0, "compute": 0.0}
        for n, s, eff, comp in zip(self.nodes, self.share, self.mem_eff, self.comp_gflops):
            mem, cp = (self.weights_gb * s + kv * s) / (n.bw_gbs * eff * n.throttle), (toks * self.gflop_tok + attn) * s / comp
            parts["mem"] += mem
            parts["compute"] += cp
            parts.setdefault("_stage", 0.0)
            parts["_stage"] += max(mem, cp)
        t = parts.pop("_stage") + self._hops() + self.step_s
        return t, _limit({**parts, "hop": self._hops(), "overhead": self.step_s}), self.mult

    def _moe_step(self, k: int, toks: int, kv: float, attn: float) -> tuple:
        n0 = self.nodes[0]
        union = (self.E / self.a) * (1 - (1 - self.a / self.E) ** max(toks, 1))
        egb = self.expert_gb * union
        dense = (self.dense_gb + kv) / (n0.bw_gbs * self.mem_eff[0] * n0.throttle)
        comp = (toks * self.gflop_tok + attn) / self.comp_gflops[0]
        if self.heat: ex = [(r / (n.bw_gbs * eff * n.throttle), d / max(n.ssd_gbs(), 0.01) * (1 - self.overlap)) for n, eff, (r, d) in zip(self.nodes, self.mem_eff, self.heat.step(toks))]
        else: ex = [(egb * s * self.hit / (n.bw_gbs * eff * n.throttle), egb * s * (1 - self.hit) / max(n.ssd_gbs(), 0.01) * (1 - self.overlap)) for n, s, eff in zip(self.nodes, self.share, self.mem_eff)]
        slow = max(ex, key=sum)
        hops, over = self._hops(), self.tok_over_s + self.step_s
        t = max(dense, comp) + sum(slow) + hops + over
        return t, _limit({"mem": dense + slow[0], "compute": comp, "ssd": slow[1], "hop": hops, "overhead": over}), self.mult

    def train_s(self, tokens: int, factor: float) -> tuple:
        """A LoRA pass over tokens: forward and backward through the frozen weights (factor x the forward arithmetic), in ubatch steps."""
        steps = math.ceil(tokens / UBATCH)
        per, lim, _ = self.step([], UBATCH, 0)
        return steps * per * factor, lim

# --- Single operations ---

REF_GFLOPS = 16.6 * 0.3 * 1e3   # a 780M at 30% of peak: what the media reference times below scale from

class Op:
    """Image, video, encoder, speech: each call is one block of work with a duration, run one at a time per slot."""
    def __init__(self, eng: dict, model: dict, backend: dict, nodes: list, pool, V, cat):
        k = lambda f: f"""engine.{eng["id"]}.{f}"""
        self.id, self.model, self.node, self.V, self.kind = eng["id"], model, nodes[0], V, model["kind"]
        self.device = backend["device"]
        self.mem_eff = V(backend["mem_eff"], k("mem_eff"))
        self.comp = self.node.tflops(self.device) * V(backend["compute_eff"], k("compute_eff")) * 1e3
        self.step_s = V(backend["step_ms"], k("step_ms")) / 1e3
        self.load_eff = V(backend.get("load_eff", 0.7), k("load_eff"))
        self.slots, self.mid = int(eng.get("slots", 1)), model["id"]
        self.weights_gb = model.get("gb") or model["params_b"] * model.get("precision_bytes", 1.0)
        self.enc = cat.get("models", model["encoder"]) if model.get("encoder") else None
        self.enc_gb = self.enc["params_b"] * self.enc.get("precision_bytes", 1.0) if self.enc else 0.0
        self.resident_gb = self.weights_gb + self.enc_gb
        act = {"image": 3.0, "video": 6.0}.get(self.kind, 0.3)
        self.node.reserve(self.resident_gb + act, f"{self.id} {self.kind} weights")
        self.bw = self.node.bw_gbs * self.mem_eff * self.node.throttle

    def load_s(self) -> float: return 2.0 + self.resident_gb / max((self.node.ssd_gbs() or 0.5) * self.load_eff, 0.05)

    def _encode(self) -> float: return (self.enc["tokens"] * 2 * self.enc["params_b"] / self.comp + self.enc_gb / self.bw) if self.enc else 0.0

    def duration(self, job: dict) -> tuple:
        V, m = self.V, self.model
        if self.kind == "image":
            mp, steps = job.get("mp", 1.0), job.get("steps") or m["steps"]
            step = max(V(m["tflop_step_mp"], f"model.{self.mid}.tflop_step_mp") * mp * 1e3 / self.comp, self.weights_gb / self.bw) + self.step_s
            vae = V(m["vae_s_mp_ref"], f"model.{self.mid}.vae_s") * mp * REF_GFLOPS / self.comp
            return self._encode() + steps * step + vae, "compute"
        if self.kind == "video":
            steps = job.get("steps") or m["steps"]
            scale = job.get("frames", m["frames"]) / m["frames"]
            step = max(V(m["tflop_step"], f"model.{self.mid}.tflop_step") * scale * 1e3 / self.comp, self.weights_gb / self.bw) + self.step_s
            return self._encode() + steps * step + V(m["vae_s_ref"], f"model.{self.mid}.vae_s") * scale * REF_GFLOPS / self.comp, "compute"
        if self.kind in ("stt", "tts"):
            audio = job.get("audio_s") or job.get("chars", 150) / 15
            return audio * V(m["rtf"], f"model.{self.mid}.rtf") * m.get("ref_cpu_tflops", 0.7) * 1e3 / self.comp, "compute"
        if self.kind == "encoder": return self._encode(), "compute"
        raise ValueError(f"{self.id}: no duration model for {self.kind}")
