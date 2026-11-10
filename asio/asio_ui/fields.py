"""asio_ui/fields.py - turning catalog and scenario data into words people read, and form fields that save as they change.
LABELS names every catalog field (a dotted path inside an entry) with its unit; VALUE_LABELS names the last part of a simulator value key (node.n4.mem_bw_gbs).
Numbers that carry uncertainty are typed as one number, or three: low, most likely, high (\"150 400 1200\"); a JSON spec ({\"theory\": 16.6, \"est\": [...]}) also works."""
import json, re
from asio_ui import base as B

LABELS = {
    "label": ("name", ""), "kind": ("kind", ""), "owned": ("yours", ""), "candidate": ("to verify before use", ""), "license": ("license", ""),
    "ram_gb": ("RAM", "GB"), "mem_mts": ("memory speed", "MT/s"), "bus_bits": ("memory bus", "bits"), "mem_bw_gbs": ("memory bandwidth", "GB/s"), "os_reserve_gb": ("kept for the system", "GB"), "mem_eff_cpu": ("CPU share of bandwidth reached", ""),
    "cpu.name": ("CPU", ""), "cpu.cores": ("CPU cores", ""), "cpu.ghz": ("CPU clock", "GHz"), "cpu.tflops": ("CPU compute", "TFLOPS fp32"),
    "gpu.name": ("GPU", ""), "gpu.arch": ("GPU architecture", ""), "gpu.cu": ("GPU compute units", ""), "gpu.tflops_fp16": ("GPU compute", "TFLOPS fp16"), "npu.name": ("NPU", ""), "npu.tops_int8": ("NPU compute", "TOPS int8"),
    "power_w.idle": ("power, idle", "W"), "power_w.load": ("power, under load", "W"), "price_usd": ("price", "$"), "throttle": ("sustained share of peak", ""),
    "ssd": ("SSDs", ""), "ssd_slots": ("SSD slots", ""), "ports.usb4": ("USB4 ports", ""), "ports.eth_gbps": ("Ethernet", "Gb/s"), "ports.wifi": ("Wi-Fi", ""), "ports.oculink": ("OCuLink ports", ""),
    "arch": ("architecture", ""), "params_b": ("parameters", "B"), "active_b": ("active parameters", "B"), "layers": ("layers", ""), "full_attn_layers": ("full-attention layers", ""), "moe_layers": ("MoE layers", ""),
    "experts_per_layer": ("experts per layer", ""), "active_experts": ("experts per token", ""), "ctx_max": ("longest context", "tokens"), "kv_kb_per_token": ("KV cache per token", "KB"), "kv_quant_bytes": ("KV bytes per value", ""),
    "state_mb": ("fixed state per stream", "MB"), "mtp.tokens_per_step": ("tokens per step with MTP", ""), "draft.tokens_per_step": ("tokens per step with a draft model", ""), "default_quant": ("default quant", ""),
    "disk_gb": ("on disk", "GB"), "experts_gb": ("all experts", "GB"), "resident_gb": ("always in memory", "GB"), "dense_gb_token": ("dense bytes read per token", "GB"), "expert_gb_token": ("expert bytes read per token", "GB"),
    "gflop_token": ("compute per token", "GFLOP"), "hit_alpha": ("expert cache hit curve exponent", ""), "agent.p_valid": ("agent: a step is a usable tool call", ""), "agent.p_progress": ("agent: a good step moves the task on", ""),
    "agent.p_recover": ("agent: the step after a bad one is good", ""), "decide.accuracy": ("picks the right route", ""), "eci": ("Epoch Capabilities Index", ""), "backends": ("runs on", ""), "roles": ("good for", ""),
    "steps": ("steps", ""), "res": ("resolution", ""), "frames": ("frames", ""), "tflop_step_mp": ("compute per step per megapixel", "TFLOP"), "tflop_step": ("compute per step", "TFLOP"), "vae_s_mp_ref": ("decode per megapixel (780M)", "s"),
    "vae_s_ref": ("decode (780M)", "s"), "encoder": ("text encoder", ""), "tokens": ("tokens encoded", ""), "gb": ("weights", "GB"), "precision_bytes": ("bytes per parameter", ""), "rtf": ("real-time factor", ""), "ref_cpu_tflops": ("reference CPU", "TFLOPS"), "clone": ("clones voices", ""),
    "device": ("runs on", ""), "mem_eff": ("share of memory bandwidth reached", ""), "compute_eff": ("share of peak compute reached", ""), "step_ms": ("fixed cost per step", "ms"), "token_overhead_ms": ("fixed cost per token", "ms"), "ssd_overlap": ("SSD reads hidden behind compute", ""), "load_eff": ("share of SSD speed when loading", ""),
    "rate_gbps": ("rate", "Gb/s"), "latency_us": ("latency per hop", "us"), "needs_port": ("needs a port", ""), "min_port_gbps": ("needs a port of at least", "Gb/s"), "hub": ("shared hub", ""), "reduce": ("reduces in the network", ""), "overhead": ("adds to every message", ""),
    "in": ("input", "$/M tokens"), "cached": ("cached input", "$/M tokens"), "out": ("output", "$/M tokens"), "tps": ("speed", "tokens/s"), "model": ("model", ""), "date": ("checked", ""), "src": ("source", ""),
    "electricity_usd_kwh": ("electricity", "$/kWh"), "life_months": ("hardware life", "months"), "internet_usd_month": ("internet", "$/month"), "space_usd_month": ("space", "$/month"), "labor_usd_hour": ("labor", "$/hour"), "cooling_cop": ("cooling efficiency (COP)", ""), "payment_fee": ("payment fee", "share")}

VALUE_LABELS = {"mem_bw_gbs": ("memory bandwidth", "GB/s"), "cpu_tflops": ("CPU compute", "TFLOPS"), "gpu_tflops": ("GPU compute", "TFLOPS"), "mem_eff_cpu": ("CPU share of bandwidth", ""), "throttle": ("sustained share of peak", ""), "os_reserve_gb": ("kept for the system", "GB"),
    "idle_w": ("power idle", "W"), "load_w": ("power under load", "W"), "price_usd": ("price", "$"), "rate": ("link rate", "Gb/s"), "latency": ("link latency", "us"), "mem_eff": ("share of memory bandwidth", ""), "compute_eff": ("share of peak compute", ""),
    "step_ms": ("fixed cost per step", "ms"), "token_overhead_ms": ("fixed cost per token", "ms"), "ssd_overlap": ("SSD reads hidden", ""), "load_eff": ("loading speed share", ""), "kv_kb": ("KV per token", "KB"), "state_mb": ("state per stream", "MB"),
    "tokens_per_step": ("tokens per step", ""), "dense_gb_token": ("dense GB per token", "GB"), "expert_gb_token": ("expert GB per token", "GB"), "gflop_token": ("GFLOP per token", ""), "hit_alpha": ("expert hit curve exponent", ""), "resident_gb": ("always in memory", "GB"),
    "p_valid": ("agent: usable tool call", ""), "p_progress": ("agent: step moves task on", ""), "p_recover": ("agent: recovers after a bad step", ""), "accuracy": ("decision accuracy", ""), "tflop_step_mp": ("TFLOP per step per MP", ""), "tflop_step": ("TFLOP per step", ""),
    "vae_s": ("decode time", "s"), "rtf": ("real-time factor", ""), "electricity": ("electricity", "$/kWh"), "life_months": ("hardware life", "months"), "internet": ("internet", "$/month"), "space": ("space", "$/month"), "cooling_cop": ("cooling COP", "")}

def label(path: str) -> tuple: return LABELS.get(path) or LABELS.get(path.split(".")[-1]) or (path.replace("_", " "), "")

def value_label(key: str) -> str:
    """node.n4.mem_bw_gbs -> 'n4: memory bandwidth (GB/s)'; ssd0 and mem_eff.<node> keep their qualifier."""
    parts = key.split(".")
    if parts[0] == "node" and parts[-1].startswith("ssd"): return f"""{parts[1]}: SSD {int(parts[-1][3:] or 0) + 1} read (GB/s)"""
    if parts[0] == "engine" and len(parts) > 3 and parts[2] == "mem_eff": return f"""{parts[1]} on {parts[3]}: share of memory bandwidth"""
    name, unit = VALUE_LABELS.get(parts[-1], (parts[-1].replace("_", " "), ""))
    who = "economics" if parts[0] == "econ" else ".".join(parts[1:-1]) if len(parts) > 2 else parts[0]
    return f"""{who}: {name}{f' ({unit})' if unit else ''}"""

# --- Values with tiers ---

def is_spec(v) -> bool: return isinstance(v, dict) and any(k in v for k in ("theory", "est", "measured"))

def median(v):
    """The number a run uses by default (measured, else estimate, else theory)."""
    if not is_spec(v): return v
    asio, _, _ = B.libs()
    return asio.Valuer("median")(v, "_")

def show(v, unit: str = "") -> str:
    """A catalog value for reading: the number a run uses, then the tiers behind it, the source on hover."""
    u = f""" <span class="asio-unit">{B.esc(unit)}</span>""" if unit else ""
    if isinstance(v, bool): return "yes" if v else "no"
    if isinstance(v, list) and v and isinstance(v[0], dict): return "; ".join(show_ssd(s) for s in v)
    if isinstance(v, list): return B.esc(", ".join(map(str, v))) + u
    if v is None: return "-"
    if not is_spec(v): return B.esc(B.fmt(v) if isinstance(v, float) else v) + u
    asio, _, _ = B.libs()
    d, tiers = asio.describe(v), []
    if "measured" in d: tiers.append(f"""measured n={d['measured']['n']}""")
    if "est" in d: tiers.append(f"""est {B.fmt(d['est']['p5'])}-{B.fmt(d['est']['p95'])}""")
    if "theory" in d: tiers.append(f"""theory {B.fmt(d['theory'])}""")
    src = "; ".join(x for x in (v.get("src"), (v.get("measured") or {}).get("src"), v.get("date")) if x)
    return f"""<span class="asio-v" title="{B.esc(src)}">{B.fmt(median(v))}{u} <span class="asio-tier">{', '.join(tiers)}</span></span>"""

def show_ssd(s: dict) -> str: return f"""{s.get('gb', '?')} GB, {show(s.get('read_gbs'), 'GB/s read')}"""

ORDER = {k: i for i, k in enumerate(LABELS)}

def flat(e: dict, p: str = "") -> dict:
    """An entry as {dotted path: value} in the order LABELS lists fields; value specs and lists kept whole; notes (keys starting with _) and empty values left out."""
    out = {}
    for k, v in e.items():
        if k.startswith("_") or k in ("id", "abstract", "extends") or v is None: continue
        if isinstance(v, dict) and not is_spec(v) and k not in ("table", "tps"): out.update(flat(v, f"{p}{k}."))
        else: out[p + k] = v
    return out if p else dict(sorted(out.items(), key=lambda kv: ORDER.get(kv[0], ORDER.get(kv[0].split(".")[-1], len(ORDER)))))

# --- Typed input ---

def parse(kind: str, raw):
    """Text from a form field to a value; empty means unset (the default applies). Raises ValueError with what to type instead."""
    if kind == "bool": return raw not in (None, "", "0", "false", False)
    if kind == "list": return [x for x in ([raw] if isinstance(raw, str) else raw or []) if x]
    s = (raw or "").strip() if isinstance(raw, str) else raw
    if s in ("", None): return None
    if kind == "str": return s
    if kind == "json" or (kind in ("est", "esti") and s[:1] in "{["): return json.loads(s)
    nums = [float(x) for x in re.split(r"[\s,/]+", s) if x]
    if kind == "nums": return nums
    as_n = (lambda x: int(round(x))) if kind in ("int", "esti") else float
    if kind in ("int", "float"):
        if len(nums) != 1: raise ValueError(f"one number, not {s!r}")
        return as_n(nums[0])
    if len(nums) == 1: return as_n(nums[0])
    if len(nums) == 3 and nums[0] <= nums[1] <= nums[2]: return [as_n(x) for x in nums]
    raise ValueError(f"one number, or three from low to high (low, most likely, high), not {s!r}")

def typed(v) -> str:
    """A value back into the text a person would type."""
    if v is None: return ""
    if isinstance(v, list) and v and all(isinstance(x, (int, float)) and not isinstance(x, bool) for x in v) and (len(v) == 3 or len(v) > 4): return " ".join(f"{x:g}" for x in v)
    if isinstance(v, (dict, list)): return json.dumps(v)
    return f"{v:g}" if isinstance(v, float) else str(v)

def get_path(d, path: str, default=None):
    for p in path.split("."):
        if isinstance(d, list):
            if not p.isdigit() or int(p) >= len(d): return default
            d = d[int(p)]
        elif isinstance(d, dict) and p in d: d = d[p]
        else: return default
    return d

def set_path(d, path: str, v):
    """Sets (or with v None removes) a dotted path; list items by index, missing dicts made on the way."""
    parts = path.split(".")
    for p in parts[:-1]: d = d[int(p)] if isinstance(d, list) else d.setdefault(p, {})
    last = parts[-1]
    if isinstance(d, list): d[int(last)] = v
    elif v is None: d.pop(last, None)
    else: d[last] = v

# --- Fields that save on change ---

def inp(path: str, v, kind: str = "str", ph: str = "", w: str = "", t: str = "asio_set", title: str = "", **kw) -> str:
    return f"""<input class="asio-in {w}" name="v" value="{B.esc(typed(v))}" placeholder="{B.esc(ph)}" title="{B.esc(title)}" {B.act(t, trigger="change", path=path, kind=kind, **kw)}>"""

def sel(path: str, v, options: list, t: str = "asio_set", blank: str = None, w: str = "", **kw) -> str:
    """options: [(value, label)] or [(group, [(value, label)])]."""
    one = lambda o: f"""<option value="{B.esc(o[0])}" {'selected' if str(o[0]) == str(v) else ''}>{B.esc(o[1])}</option>"""
    opts = "".join(f"""<optgroup label="{B.esc(o[0])}">{''.join(one(x) for x in o[1])}</optgroup>""" if isinstance(o[1], list) else one(o) for o in options)
    return f"""<select class="asio-in {w}" name="v" {B.act(t, trigger="change", path=path, kind="str", **kw)}>{f'<option value="">{B.esc(blank)}</option>' if blank is not None else ''}{opts}</select>"""

def chk(path: str, v, label: str = "", t: str = "asio_set", **kw) -> str: return f"""<label class="asio-chk"><input type="checkbox" name="v" value="1" {'checked' if v else ''} {B.act(t, trigger="change", path=path, kind="bool", **kw)}>{B.esc(label)}</label>"""

def multi(path: str, values: list, options: list, t: str = "asio_set", **kw) -> str:
    """A group of checkboxes saved together as a list."""
    boxes = "".join(f"""<label class="asio-chk"><input type="checkbox" name="v" value="{B.esc(o)}" {'checked' if o in (values or []) else ''}>{B.esc(lbl)}</label>""" for o, lbl in options)
    return f"""<form class="asio-multi" onsubmit="return false" {B.act(t, trigger="change", path=path, kind="list", **kw)}>{boxes}</form>"""

def field(name: str, control: str, hint: str = "") -> str: return f"""<label class="asio-f" title="{B.esc(hint)}"><span>{B.esc(name)}</span>{control}</label>"""
