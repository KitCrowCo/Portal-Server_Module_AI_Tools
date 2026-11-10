"""asio_core/heat.py - which experts a MoE model's tokens choose, and what that means for where the experts live.
Heat is per layer: q[l, e], the chance one token picks expert e in layer l (a layer's q sums to the experts each token picks). Sources (an engine's "heat"):
  "uniform"              every expert equally likely
  {"zipf": s}            popularity falls as 1 / rank^s; s may be an uncertain value (null reads the model's heat_skew), drawn per run as model.<id>.heat_skew
  "<id>"                 a catalog heat entry: counts per layer (from Fleet's traces, Magpie's expert usage, or a pasted table)
Arrangements (an engine's "arrangement") decide which machine owns each expert (reads it from its SSD when not in RAM) and what each keeps in RAM:
  hottest   heat is known: experts go out hottest first, each to the machine with the least load for its memory bandwidth that still has room, so RAM holds the hottest and load stays even
  spread    heat is not known and placement is fixed: experts dealt out at random in proportion to each machine's cache, and a random part of each machine's own kept in RAM
  learned   heat is not known and each cache adapts (least recently used): a machine's RAM holds its own experts in proportion to how often they are chosen (Che's approximation)
A step that batches t tokens reads, in every layer, each expert at least one of them chose: 1 - (1 - q)^t of the time."""
import zlib
import numpy as np

ARRANGEMENTS = ("hottest", "spread", "learned")

def table(source, model: dict, cat, V) -> np.ndarray:
    """q as a (layers, experts) array."""
    E, L, a = model["experts_per_layer"], model.get("moe_layers", model.get("layers", 1)), model["active_experts"]
    if source == "uniform": return np.full((L, E), a / E)
    if isinstance(source, dict) and "zipf" in source:
        s = V(source["zipf"] if source["zipf"] is not None else model.get("heat_skew", {"est": [0.4, 0.8, 1.2]}), f"""model.{model["id"]}.heat_skew""")
        p = 1.0 / np.arange(1, E + 1) ** s
        return np.tile(np.minimum(a * p / p.sum(), 1.0), (L, 1))
    h = cat.get("heat", source)
    if h.get("zipf") is not None: return table({"zipf": h["zipf"]}, model, cat, V)
    c = np.asarray(h["counts"], dtype=float) + 1e-9
    if c.ndim == 1: c = c[None, :]
    q = np.minimum(a * c / c.sum(axis=1, keepdims=True), 1.0)
    return np.resize(q, (L, q.shape[1])) if q.shape[0] != L else q

class Heat:
    """One engine's experts over its machines. nodes: the perf Node objects holding caches; cache_gb: RAM each gives to experts; bw: memory GB/s each reaches."""
    def __init__(self, q: np.ndarray, arrangement: str, nodes: list, cache_gb: list, bw: list, ssd: list, expert_gb: float, seed: int):
        if arrangement not in ARRANGEMENTS: raise ValueError(f"arrangement must be one of {ARRANGEMENTS}, not {arrangement!r}")
        self.q, self.b, self.k = q.ravel(), expert_gb, len(nodes)
        n = len(self.q)
        cap = [int(c / expert_gb) for c in cache_gb]
        rng = np.random.default_rng([seed, zlib.crc32(arrangement.encode())])
        self.owner, self.h = np.zeros(n, dtype=int), np.zeros(n)
        if arrangement == "hottest":
            load, sload, held = np.zeros(self.k), np.zeros(self.k), np.zeros(self.k, dtype=int)
            for i in np.argsort(-self.q):
                room = held < cap
                if room.any():
                    j = int(np.argmin(np.where(room, (load + self.q[i]) / np.asarray(bw), np.inf)))
                    load[j], held[j], self.h[i] = load[j] + self.q[i], held[j] + 1, 1.0
                else:
                    j = int(np.argmin((sload + self.q[i]) / np.maximum(ssd, 1e-3)))
                    sload[j] += self.q[i]
                self.owner[i] = j
        else:
            w = np.asarray(cache_gb, dtype=float) if sum(cache_gb) else np.asarray(ssd, dtype=float) + 1e-9
            self.owner = rng.choice(self.k, size=n, p=w / w.sum())
            for j in range(self.k):
                mine = np.flatnonzero(self.owner == j)
                if arrangement == "spread": self.h[rng.permutation(mine)[:cap[j]]] = 1.0
                else: self.h[mine] = _che(self.q[mine], cap[j])
        self.hit = float((self.q * self.h).sum() / self.q.sum())
        self.load = [float(self.q[self.owner == j].sum() / self.q.sum()) for j in range(self.k)]
        self._memo = {}

    def step(self, tokens: int) -> list:
        """[(GB read from RAM, GB read from SSD)] per machine for one step that batches this many tokens."""
        t = max(int(tokens), 1)
        if t not in self._memo:
            u = 1.0 - (1.0 - self.q) ** t
            self._memo[t] = [(self.b * float((u * self.h)[self.owner == j].sum()), self.b * float((u * (1 - self.h))[self.owner == j].sum())) for j in range(self.k)]
        return self._memo[t]

def _che(q: np.ndarray, slots: int) -> np.ndarray:
    """Hit chance of each item in an LRU cache of this many slots under independent requests at rates q (Che's approximation): h = 1 - exp(-q T), T where the hits fill the slots."""
    if slots <= 0 or len(q) == 0: return np.zeros(len(q))
    if slots >= len(q): return np.ones(len(q))
    lo, hi = 0.0, 1.0
    while (1 - np.exp(-q * hi)).sum() < slots: hi *= 2
    for _ in range(60):
        mid = (lo + hi) / 2
        lo, hi = (mid, hi) if (1 - np.exp(-q * mid)).sum() < slots else (lo, mid)
    return 1 - np.exp(-q * hi)

def summary(q: np.ndarray, a: int) -> dict:
    """How skewed a heat table is: share of choices the hottest 10% and 25% of experts take (per layer, averaged), and the experts per layer."""
    s = -np.sort(-q, axis=1)
    E = q.shape[1]
    return {"layers": q.shape[0], "experts": E, "top10": float((s[:, :max(E // 10, 1)].sum(axis=1) / s.sum(axis=1)).mean()), "top25": float((s[:, :max(E // 4, 1)].sum(axis=1) / s.sum(axis=1)).mean()), "curve": (s.sum(axis=0) / s.sum()).cumsum()[::max(E // 64, 1)].round(4).tolist()}
