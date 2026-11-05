"""experts.py - where a mixture-of-experts model's experts should live across nodes, worked out from routing traces (which experts each token used in each layer).
Traces: data/ai_tools/fleet/traces/<name>.jsonl, one line per token, either {"tok": n, "route": [[experts of layer 0], [experts of layer 1], ...]} or one line per token and layer {"tok": n, "layer": l, "experts": [...]}.
  colibri can write every routing decision to a file (ROUTE_TRACE=<path> in its environment). Its format is not documented: a converter into this format is written once a real sample exists.
  synthetic() makes a trace with skewed, topic-dependent popularity for trying the planner before real traces exist. It shows how the planner behaves, it is not evidence about any model.
Placements (PLACEMENTS) decide which experts each node keeps in RAM, given how many expert slots each node has. They learn from the first part of a trace and are scored on the rest,
so a placement that memorises one conversation does not look better than it is.
Score, per token and layer: hits (the chosen expert is in some node's RAM), misses (read from disk), remote nodes contacted, hops between nodes, and an estimated time per token from a stated cost model
(memory and disk bandwidth, link round trip and speed, dense bytes per token). The cost model is an estimate, to be replaced by measurements."""
import json, math, random, itertools, collections
from pathlib import Path

TRACE_DIR = Path("./data/ai_tools/fleet/traces")
PLACEMENT_DIR = Path("./data/ai_tools/fleet/placements")
COST = {"expert_mb": 13.4, "dense_gb": 5.6, "act_kb": 8, "ram_gbs": 80, "disk_gbs": 5, "link_gbs": 1.25, "rtt_us": 100}   # DeepSeek V4 Flash per expert and per token (derived, see the memory map); APU node RAM, Gen4 NVMe, 10 GbE: estimates

# --- Traces ---

def traces() -> list: return sorted(p.stem for p in TRACE_DIR.glob("*.jsonl")) if TRACE_DIR.exists() else []

def load(name: str, limit: int = 20000) -> dict:
    """{"name", "layers", "experts", "tokens": [[experts per layer] per token]} - at most limit tokens."""
    per_tok = collections.defaultdict(dict)
    with open(TRACE_DIR / f"{Path(name).name}.jsonl", encoding="utf-8") as f:
        for line in f:
            if not line.strip(): continue
            r = json.loads(line)
            if "route" in r:
                per_tok[r["tok"]] = dict(enumerate(r["route"]))
            else: per_tok[r["tok"]][int(r["layer"])] = r["experts"]
            if len(per_tok) > limit: break
    keys = sorted(per_tok)[:limit]
    layers = 1 + max((l for k in keys for l in per_tok[k]), default=-1)
    tokens = [[list(per_tok[k].get(l, [])) for l in range(layers)] for k in keys]
    experts = 1 + max((e for t in tokens for es in t for e in es), default=-1)
    return {"name": name, "layers": layers, "experts": experts, "tokens": tokens}

def heat(tr: dict, tokens: list = None) -> list:
    m = [[0] * tr["experts"] for _ in range(tr["layers"])]
    for t in tokens if tokens is not None else tr["tokens"]:
        for l, es in enumerate(t):
            for e in es: m[l][e] += 1
    return m

def synthetic(layers: int = 43, experts: int = 256, k: int = 6, tokens: int = 4000, topics: int = 6, turn: int = 200, skew: float = 1.0, seed: int = 7) -> str:
    """Writes traces/synthetic-<layers>x<experts>-k<k>-<tokens>.jsonl: tokens come in turns of `turn` tokens, each turn on one of `topics` topics; every topic and layer prefers its own experts (Zipf weights, exponent skew),
    which gives the trace heat, co-activation and drift between turns, as real routing has. Returns the trace name."""
    rnd = random.Random(seed)
    cum = list(itertools.accumulate(1 / (r + 1) ** skew for r in range(experts)))
    prefs = [[rnd.sample(range(experts), experts) for _ in range(layers)] for _ in range(topics)]
    name = f"synthetic-{layers}x{experts}-k{k}-{tokens}"
    TRACE_DIR.mkdir(parents=True, exist_ok=True)
    with open(TRACE_DIR / f"{name}.jsonl", "w", encoding="utf-8") as f:
        for n in range(tokens):
            topic = random.Random(n // turn * 7919 + seed).randrange(topics)   # one topic per turn, the same on every run
            route = []
            for l in range(layers):
                chosen = set()
                while len(chosen) < k: chosen.add(prefs[topic][l][rnd.choices(range(experts), cum_weights=cum)[0]])
                route.append(sorted(chosen))
            f.write(json.dumps({"tok": n, "route": route}) + "\n")
    return name

# --- Placements: which experts each node holds; who owns (and reads from disk) each expert; which node runs each layer's attention ---
# A placement gets the heat of the learning part (layers x experts), co-activation counts (only coactivate uses them), nodes and slots per node,
# and returns {"holds": [set of (layer, expert)] per node, "owner": {(layer, expert): node}, "home": [node per layer]}.

PLACEMENTS: dict = {}

def placement(name: str, doc: str):
    def reg(fn):
        PLACEMENTS[name] = {"fn": fn, "doc": doc}
        return fn
    return reg

def _ranked(h: list) -> list: return [sorted(range(len(r)), key=lambda e: -r[e]) for r in h]   # per layer: experts hottest first

def _fill(h: list, owner: dict, nodes: int, slots: int, layers_of: list) -> list:
    """Each node keeps its own experts, hottest first, slots spread evenly over the layers it serves."""
    holds = [set() for _ in range(nodes)]
    for n in range(nodes):
        ls = layers_of[n]
        if not ls: continue
        q = slots // len(ls)
        for l in ls: holds[n].update((l, e) for e in [e for e in sorted(range(len(h[l])), key=lambda e: -h[l][e]) if owner[(l, e)] == n][:q])
    return holds

@placement("layer_split", "Each node serves a block of consecutive layers (attention and experts): one stream passes node to node (a pipeline); each node keeps the hottest experts of its own layers.")
def _layer_split(h, co, nodes, slots, p):
    L, E = len(h), len(h[0])
    home = [min(nodes - 1, l * nodes // L) for l in range(L)]
    owner = {(l, e): home[l] for l in range(L) for e in range(E)}
    return {"holds": _fill(h, owner, nodes, slots, [[l for l in range(L) if home[l] == n] for n in range(nodes)]), "owner": owner, "home": home}

@placement("shard_hot", "One node runs attention for every layer; each layer's experts are dealt out to the nodes in order of heat (so heat is spread evenly), and each node keeps its hottest ones in its slots. Every layer sends the activation to the nodes holding the chosen experts.")
def _shard_hot(h, co, nodes, slots, p):
    owner = {(l, e): i % nodes for l, r in enumerate(_ranked(h)) for i, e in enumerate(r)}
    return {"holds": _fill(h, owner, nodes, slots, [list(range(len(h)))] * nodes), "owner": owner, "home": [0] * len(h)}

@placement("replicate_top", "Like shard_hot, but each layer's `replicate` hottest experts are kept on every node, so most layers need no other node at all; fewer distinct experts fit in the cluster's RAM as a whole.")
def _replicate_top(h, co, nodes, slots, p):
    R, L = int(p.get("replicate", 8)), len(h)
    ranked = _ranked(h)
    owner = {(l, e): (0 if i < R else (i - R) % nodes) for l, r in enumerate(ranked) for i, e in enumerate(r)}
    rest = _fill(h, {k: (v if k[1] not in ranked[k[0]][:R] else -1) for k, v in owner.items()}, nodes, max(0, slots - R * L), [list(range(L))] * nodes)
    return {"holds": [rest[n] | {(l, e) for l in range(L) for e in ranked[l][:R]} for n in range(nodes)], "owner": owner, "home": [0] * L}

@placement("coactivate", "Like shard_hot, but experts often chosen for the same token go to the same node (greedy, by co-activation counts of the learning part), so a token's experts sit on fewer nodes and each layer contacts fewer of them. balance (default 1) weighs keeping the nodes' heat even against keeping co-chosen experts together.")
def _coactivate(h, co, nodes, slots, p):
    L, E, bal = len(h), len(h[0]), float(p.get("balance", 1.0))
    cap, owner = math.ceil(E / nodes), {}
    for l, r in enumerate(_ranked(h)):
        members, load, total = [[] for _ in range(nodes)], [0] * nodes, sum(h[l]) or 1
        for e in r:
            aff = [sum(co[l].get((min(e, o), max(e, o)), 0) for o in members[n]) for n in range(nodes)]
            norm = sum(aff) or 1
            best = max((n for n in range(nodes) if len(members[n]) < cap), key=lambda n: aff[n] / norm - bal * load[n] / total)
            owner[(l, e)], load[best] = best, load[best] + h[l][e]
            members[best].append(e)
    return {"holds": _fill(h, owner, nodes, slots, [list(range(L))] * nodes), "owner": owner, "home": [0] * L}

def _coactivation(tokens: list, layers: int) -> list:
    co = [collections.Counter() for _ in range(layers)]
    for t in tokens:
        for l, es in enumerate(t):
            for i, a in enumerate(es):
                for b in es[i + 1:]: co[l][(min(a, b), max(a, b))] += 1
    return co

# --- Scoring ---

def score(pl: dict, tokens: list, nodes: int, c: dict) -> dict:
    """Replays tokens over a placement. Per layer: each chosen expert is computed on a node holding it (the layer's home node first, then a node this layer already contacts), else read from disk by its owner (a miss).
    The layer costs its slowest node's expert work plus one round trip when a node other than the layer's home is involved; a change of home node between layers is one hop (pipeline). Dense work: dense_gb per token on the home nodes."""
    em, ram, disk = c["expert_mb"] / 1000, c["ram_gbs"], c["disk_gbs"]
    rtt, act = c["rtt_us"] / 1e6, c["act_kb"] / 1e6 / c["link_gbs"]
    hits = misses = remote = hops = 0
    t_exp = t_net = 0.0
    work = [0.0] * nodes
    holders = collections.defaultdict(list)
    for n, hs in enumerate(pl["holds"]):
        for k in hs: holders[k].append(n)
    home = pl["home"]
    for t in tokens:
        prev = home[0]
        for l, es in enumerate(t):
            hn, w, used = home[l], [0.0] * nodes, set()
            if hn != prev: hops += 1; t_net += rtt / 2 + act
            prev = hn
            for e in es:
                hl = holders.get((l, e), [])
                n = hn if hn in hl else next((x for x in hl if x in used), hl[0] if hl else None)
                if n is None: n = pl["owner"][(l, e)]; misses += 1; w[n] += em / disk
                else: hits += 1; w[n] += em / ram
                used.add(n)
            far = used - {hn}
            remote += len(far)
            for n in range(nodes): work[n] += w[n]
            t_exp += max(w)
            if far: t_net += rtt + 2 * act
        if prev != home[0]: hops += 1; t_net += rtt / 2 + act   # back to the first node for the next token
    T, lt = max(len(tokens), 1), max(sum(len(t) for t in tokens), 1)
    dense = c["dense_gb"] / ram
    ms = (dense + (t_exp + t_net) / T) * 1000
    return {"hit_rate": hits / max(hits + misses, 1), "misses_per_token": misses / T, "remote_per_layer": remote / lt, "hops_per_token": hops / T, "ms_per_token": ms, "tok_s": 1000 / ms if ms else None,
            "ms_dense": dense * 1000, "ms_experts": t_exp / T * 1000, "ms_network": t_net / T * 1000, "held": [len(hs) for hs in pl["holds"]], "work_share": [w / (sum(work) or 1) for w in work]}

def plan(name: str, nodes: int, slots: int, policies: list, holdout: float = 0.3, params: dict = None, cost: dict = None) -> dict:
    """Learns each placement on the first (1 - holdout) of the trace and scores it on the rest; also scores everything-from-disk as the floor."""
    tr, c, p = load(name), {**COST, **(cost or {})}, params or {}
    cut = max(1, int(len(tr["tokens"]) * (1 - holdout)))
    learn, test = tr["tokens"][:cut], tr["tokens"][cut:] or tr["tokens"][-1:]
    h = heat(tr, learn)
    co = _coactivation(learn, tr["layers"]) if "coactivate" in policies else None
    rows = []
    for pol in policies:
        pl = PLACEMENTS[pol]["fn"](h, co, nodes, slots, p)
        rows.append({"policy": pol, "doc": PLACEMENTS[pol]["doc"], **score(pl, test, nodes, c), "pins": {n: sorted([l, e] for l, e in hs) for n, hs in enumerate(pl["holds"])}})
    floor = score({"holds": [set()] * nodes, "owner": {(l, e): 0 for l in range(tr["layers"]) for e in range(tr["experts"])}, "home": [0] * tr["layers"]}, test, nodes, c)
    return {"trace": name, "layers": tr["layers"], "experts": tr["experts"], "tokens_learn": len(learn), "tokens_test": len(test), "nodes": nodes, "slots": slots, "cost": c, "rows": rows, "all_from_disk": floor}

def save_pins(result: dict, policy: str) -> str:
    """The chosen placement's experts per node ({node: {layer: [experts]}}) as data/ai_tools/fleet/placements/<trace>__<policy>.json - what a node would pin once its engine takes a pin list."""
    row = next(r for r in result["rows"] if r["policy"] == policy)
    out = {str(n): {str(l): [e for ll, e in pins if ll == l] for l in sorted({ll for ll, _ in pins})} for n, pins in row["pins"].items()}
    PLACEMENT_DIR.mkdir(parents=True, exist_ok=True)
    path = PLACEMENT_DIR / f"""{result["trace"]}__{policy}.json"""
    path.write_text(json.dumps({"trace": result["trace"], "policy": policy, "nodes": result["nodes"], "slots": result["slots"], "pins": out}, indent=1))
    return str(path)
