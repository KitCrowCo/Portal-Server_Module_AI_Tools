"""
AI Calc - Speed/memory calculator, model finder, session monitor, hardware profiles, compare/sweep tools.
Sub-module of ai_tools. Mounted at /module/ai_tools/ai_calc.

Hardware profiles ARE CNodes (tools/ai_manager/resources.py) - no separate store. Connections (Ollama, etc.) come from tools/ai_manager/connections.py.
Nearly everything here is an IM.scripts intent, not a route - the one GET route is the page load, per the platform's dispatch architecture.
"""
import json, math, re, csv, threading, traceback, uuid, asyncio
from pathlib import Path
from datetime import datetime
import httpx
from fastapi import APIRouter, Request
from fastapi.responses import HTMLResponse

TOOL_META = {"label": "AI Calc", "group": "model_tools", "icon": "&#x25B3;", "description": "Speed, context, benchmarks, model finder, compare", "singleton": True}
router = APIRouter(redirect_slashes=False)
ENV = {}
UI = BI = AIM = IM = TM = None
_P = "/module/ai_tools/ai_calc"
DATA_DIR = Path("./data/ai_tools/ai_calc")

QUANTS, IMAGE_MODELS, TASK_PRESETS, KNOWN_BENCHMARKS, DEFAULT_WEIGHTS, SCORE_DIM_LABELS, SCORE_DIM_COLORS, _KNOWN_SHORT = {}, {}, {}, {}, {}, {}, {}, {}
ARCH_TABLE, REFERENCE_LINES, PROPRIETARY_REFS, SCORE_DIMS = [], [], [], []
STANDARD_SIZES = [1, 3, 7, 8, 9, 11, 12, 13, 14, 20, 27, 30, 32, 34, 70, 72]

def load_config():
    global ARCH_TABLE, PROPRIETARY_REFS, REFERENCE_LINES, SCORE_DIMS
    cfg = json.loads((Path(__file__).parent / "ai_calc_config.json").read_text())
    QUANTS.update(cfg.get("QUANTS", {})); IMAGE_MODELS.update(cfg.get("IMAGE_MODELS", {})); TASK_PRESETS.update(cfg.get("TASK_PRESETS", {}))
    KNOWN_BENCHMARKS.update(cfg.get("KNOWN_BENCHMARKS", {})); DEFAULT_WEIGHTS.update(cfg.get("DEFAULT_WEIGHTS", {}))
    SCORE_DIM_LABELS.update(cfg.get("SCORE_DIM_LABELS", {})); SCORE_DIM_COLORS.update(cfg.get("SCORE_DIM_COLORS", {}))
    ARCH_TABLE = cfg.get("ARCH_TABLE", []); REFERENCE_LINES = cfg.get("REFERENCE_LINES", []); PROPRIETARY_REFS = cfg.get("PROPRIETARY_REFS", []); SCORE_DIMS = cfg.get("SCORE_DIMS", [])
    _KNOWN_SHORT.update({k.split("/")[-1].lower(): k for k in KNOWN_BENCHMARKS})

# --- Hardware (CNode-backed) ---

_HW_FIELDS = ("vram_gb", "shared_gb", "sys_ram_gb", "os_overhead_gb", "mem_bw_gbps", "sys_ram_bw_gbps", "gpu_tflops_fp16")
_HW_DEFAULTS = {"vram_gb": 8.0, "shared_gb": 0.0, "sys_ram_gb": 16.0, "os_overhead_gb": 2.5, "mem_bw_gbps": 50.0, "sys_ram_bw_gbps": 50.0, "gpu_tflops_fp16": 4.0}
_HW_LABELS = {"vram_gb": "Dedicated VRAM (GB)", "shared_gb": "Shared/iGPU VRAM (GB)", "sys_ram_gb": "System RAM (GB)", "os_overhead_gb": "OS Overhead (GB)", "mem_bw_gbps": "Dedicated/iGPU Memory Bandwidth (GB/s)", "sys_ram_bw_gbps": "System RAM Bandwidth (GB/s)", "gpu_tflops_fp16": "GPU TFLOPS FP16 (matrix/tensor path, not vector)"}
KNOWN_HW_REFS = [("Radeon 760M (iGPU, RDNA3 ~8CU)", 5.323, None), ("Radeon 780M (iGPU, RDNA3 12CU)", 8.91, None), ("Radeon 890M (iGPU, RDNA3.5 16CU, matrix/tensor path)", 5.94, None), ("Raspberry Pi 5 (CPU only, no usable GPU compute)", 0.15, 17.1)]

GPU_EFF = 0.72  # fixed derate applied to raw bandwidth/flops - not yet per-node tunable, candidate for calibration once real logged data exists

def hw_cnodes() -> list: return [c for c in AIM.resources.list_cnodes() if any(k in c for k in _HW_FIELDS)]

def cnode_hw(cnode: dict = None) -> dict:
    hw = dict(_HW_DEFAULTS)
    if cnode:
        for k in _HW_FIELDS:
            if k in cnode: hw[k] = float(cnode[k])
    hw["gpu_eff"] = GPU_EFF
    return hw

def get_hw(cid: str = "") -> dict:
    if cid:
        c = AIM.resources.get_cnode(cid)
        if c: return cnode_hw(c)
    nodes = hw_cnodes()
    return cnode_hw(nodes[0]) if nodes else cnode_hw(None)

def _hw_select_html(selected: str = "") -> str:
    opts = "".join(f'<option value="{c["id"]}" {"selected" if c["id"]==selected else ""}>{UI.escape(c.get("label",c["id"]))} ({c.get("vram_gb",0)+c.get("shared_gb",0):.0f}+{c.get("sys_ram_gb",0):.0f}GB)</option>' for c in hw_cnodes())
    return f"""<select name="cnode_id" class="module-select" style="width:auto">{opts or "<option value=''>(no hardware profiles - see Hardware tab)</option>"}</select>"""

def _hw_ref_html() -> str:
    rows = "".join(f"""<div style="display:flex;justify-content:space-between;align-items:center;padding:.2rem 0;font-size:.72rem;border-bottom:1px solid var(--border)">
                            <span>{label}</span>
                            <button type="button" class="cm-qbtn" onclick="document.querySelector('.hwin[name=gpu_tflops_fp16]').value={tflops};{f"document.querySelector('.hwin[name=sys_ram_bw_gbps]').value={bw};" if bw else ""}syncHW()">Apply {tflops} TFLOPS{f' / {bw} GB/s' if bw else ''}</button>
                        </div>""" for label, tflops, bw in KNOWN_HW_REFS)
    return f"""<details style="margin-bottom:.5rem;font-size:.75rem"><summary style="cursor:pointer;color:var(--text_muted)">Known hardware FP16 TFLOPS reference (click to apply)</summary>
                   <div style="padding:.3rem 0">{rows}
                       <div style="font-size:.65rem;color:var(--text_muted);padding-top:.3rem">NPUs aren't modeled - Ollama/llama.cpp inference doesn't currently route through the NPU on any known consumer setup, only iGPU/CPU/dGPU paths. Worth adding if that changes for your backend, not before.</div>
                   </div>
               </details>"""

# --- Core Physics ---

def arch_for(params_b):
    for lo, hi, layers, hidden, attn, kv in ARCH_TABLE:
        if lo <= params_b < hi: return layers, hidden, attn, kv
    return 32, 4096, 32, 8

def total_vram(hw): return hw["vram_gb"] + hw["shared_gb"]
def usable_ram(hw): return max(hw["sys_ram_gb"] - hw["shared_gb"] - hw.get("os_overhead_gb", 0), 0.0)
def total_mem(hw): return total_vram(hw) + usable_ram(hw)
def model_gb(params_b, quant): return params_b * 1e9 * QUANTS.get(quant, QUANTS["Q4_K_M"])["bpp"] / 1e9

def kv_bytes_per_token(params_b, kv_bits=8):
    layers, hidden, attn, kv_heads = arch_for(params_b)
    return 2 * layers * kv_heads * (hidden // attn) * (kv_bits // 8)

def kv_cache_gb_for_ctx(params_b, ctx, kv_bits=8): return kv_bytes_per_token(params_b, kv_bits) * ctx / (1024**3)
def max_ctx_tokens(params_b, avail_gb, kv_bits=8): return int(avail_gb * 1024**3 / max(kv_bytes_per_token(params_b, kv_bits), 1))

def estimate_tps(params_b: float, quant: str, hw: dict) -> dict:
    """Weights each half of the model separately by whichever bus it actually sits on - dedicated/iGPU bandwidth for the VRAM-resident portion, system-RAM bandwidth for anything spilled - rather than one blended figure."""
    mgb  = model_gb(params_b, quant)
    vt, tm, eff = total_vram(hw), total_mem(hw), hw["gpu_eff"]
    vram_bw, ram_bw = hw["mem_bw_gbps"], hw.get("sys_ram_bw_gbps", hw["mem_bw_gbps"])
    if mgb > tm: return {"tps":0.0,"prefill_tps":0.0,"mode":"OOM","model_gb":round(mgb,2), "fits_vram":False,"fits_total":False,"eff_bw":0.0,"vram_used":0.0,"ram_used":round(mgb,2)}
    vram_used = min(mgb, vt); ram_used = max(mgb - vt, 0.0)
    eff_bw = eff * (vram_used*vram_bw + ram_used*ram_bw) / mgb if mgb > 0 else 0.0
    return {"tps":round((eff_bw*1e9)/(mgb*1e9*1.10),2), "prefill_tps":round(hw.get("gpu_tflops_fp16",8.9)*1e12*eff/(2*params_b*1e9),1), "mode":f"VRAM+RAM ({ram_used:.1f}GB spill)" if ram_used > 0 else "VRAM", "model_gb":round(mgb,2), "fits_vram":mgb<=vt, "fits_total":True, "eff_bw":round(eff_bw,1), "vram_used":round(vram_used,2), "ram_used":round(ram_used,2)}

def full_perf(params_b, quant, ctx, hw, kv_bits=8):
    p = estimate_tps(params_b, quant, hw)
    kv, vt, tm = kv_cache_gb_for_ctx(params_b, ctx, kv_bits), total_vram(hw), total_mem(hw)
    kv_in_vram = min(kv, max(vt - p["model_gb"], 0.0)); kv_in_ram = max(kv - kv_in_vram, 0.0)
    total_used = p["model_gb"] + kv
    fits = p["fits_total"] and total_used <= tm
    tps, mode = p["tps"], p["mode"]
    if fits and kv_in_ram > 0 and tps > 0: tps, mode = round(tps * (1.0 - (kv_in_ram / total_used) * 0.20), 2), mode + f" +KV({kv_in_ram:.1f}->RAM)"
    qi = QUANTS.get(quant, QUANTS["Q4_K_M"])
    return {**p, "tps": tps, "mode": mode, "kv_gb": round(kv, 3), "kv_in_vram": round(kv_in_vram, 3), "kv_in_ram": round(kv_in_ram, 3),
            "total_mem_gb": round(total_used, 2), "mem_ok": fits, "time_10k_s": round(10000 / tps, 0) if tps > 0 and fits else None,
            "quality_idx": qi["quality"], "creative_ok": qi["creative_ok"], "quant": quant, "params_b": params_b}

def image_estimate(model_key, hw, steps=20):
    m = IMAGE_MODELS.get(model_key, next(iter(IMAGE_MODELS.values()), {}))
    if not m or m.get("vram_min_gb", 0) > total_mem(hw): return {"feasible": False, "note": f"Needs {m.get('vram_min_gb',0):.0f} GB, have {total_mem(hw):.0f} GB"}
    its = m["its_per_90gbps"] * (hw["mem_bw_gbps"] / 90.0) * hw.get("gpu_eff", 0.72)
    return {"feasible": True, "its": round(its, 2), "total_s": round(steps / max(its, 0.01), 1), "vram_min": m["vram_min_gb"], "note": m["note"]}

# --- Scoring ---
# All dimensions in SCORE_DIMS must be 0-1 normalized before weighting - _total_score does a single weighted average across them with no per-dim rescaling.

_INTEL_WEIGHTS = {"hle": 4.0, "gpqa": 3.0, "mmlu_pro": 2.5, "ifeval": 2.5, "bbh": 2.0, "gsm8k": 1.5, "mmlu": 1.0, "arc": 1.0, "hellaswag": 0.5}

def _compute_intelligence(scores):
    """Weighted average of whatever fine-grained benchmarks are present, on the raw 0-100 scale those benchmarks report in. Caller normalizes to 0-1 - this function does not."""
    total = weight = 0.0
    for bench, w in _INTEL_WEIGHTS.items():
        if bench in scores: total += float(scores[bench]) * w; weight += w
    return round(total / max(weight, 0.001), 2)

def _sub_scores(perf, model, hw, target_tps):
    raw_avg = (model.get("leaderboard_avg") or 0.0) / 100.0
    lb_avg = min(raw_avg * model.get("bench_confidence", 0.0), 1.0) if model.get("bench_confidence", 0) > 0 else 0.0
    tps, tm, used, vt = perf.get("tps", 0.0), total_mem(hw), perf.get("total_mem_gb", 0), total_vram(hw)
    needed = perf.get("model_gb", 0.0) + perf.get("kv_gb", 0.0)
    pop_raw = math.log1p(model.get("likes", 0) or 0) * 0.4 + math.log1p(model.get("downloads", 0) or 0) * 0.6
    intel_raw = _compute_intelligence(model.get("lb_detail", {}))  # 0-100 scale, or 0 if no fine-grained benchmarks matched
    intel = intel_raw / 100.0 if intel_raw > 0 else raw_avg  # FIX: was stored un-normalized (0-100) against every other 0-1 dim, and fell to 0 with no fallback when lb_detail was empty even though leaderboard_avg had already been resolved
    return {"lb_avg": round(lb_avg, 4), "quant_qual": round(perf.get("quality_idx", 0.0), 4),
            "speed": round(min(tps / max(target_tps * 2.0, 1.0), 1.0) if tps > 0 else 0.0, 4),
            "mem_head": round(max(0.0, (tm - used) / max(tm, 1.0)), 4),
            "ctx_fit": round(min(needed, vt) / max(needed, 0.001) if needed > 0 else 1.0, 4),
            "popularity": round(min(pop_raw / 16.0, 1.0), 4),
            "intel": round(intel, 4)}

def _total_score(sub, weights):
    tw = sum(weights.get(d, DEFAULT_WEIGHTS.get(d, 1)) for d in SCORE_DIMS)
    return round(sum(sub.get(d, 0) * weights.get(d, DEFAULT_WEIGHTS.get(d, 1)) for d in SCORE_DIMS) / max(tw, 0.001), 4)

def _rank_filtered(filtered: list, hw: dict, ctx: int, target_tps: float, weights: dict, top_n: int = 40, kv_bits: int = 8) -> list:
    rows = []
    for m in filtered:
        for q in m.get("avail_quants",[]):
            if q not in QUANTS: continue
            perf = full_perf(m["params_b"], q, ctx, hw, kv_bits)
            if not perf["mem_ok"]: continue
            sub = _sub_scores(perf, m, hw, target_tps)
            rows.append({**{k: m[k] for k in ("id","params_b","likes","downloads","tags","avail_quants","leaderboard_avg","lb_detail","bench_confidence","bench_source","bench_inferred")},
                         "author": m["id"].split("/")[-1] if "/" in m["id"] else "", "name": m["id"].split("/")[-1], "quant": q, "perf": perf, "sub_scores": sub,
                         "total_score": _total_score(sub, weights), "below_floor": perf["tps"] < target_tps})
    rows.sort(key=lambda r: r["total_score"], reverse=True)
    return rows[:top_n]

def _parse_weights(f): return {d: max(0.0, min(float(f.get(f"w_{d}", DEFAULT_WEIGHTS.get(d, 1))), 10.0)) for d in SCORE_DIMS}

# --- Benchmark Resolution ---
# Single cascade: curated static table (config-editable) -> HF card metadata -> base-model inheritance -> README table scrape.

_BENCH_ALIASES = {"mmlu":"mmlu","massive multitask": "mmlu","mmlu_pro":"mmlu_pro","arc":"arc","arc_challenge":"arc","ai2_arc":"arc","hellaswag":"hellaswag","truthfulqa":"truthfulqa","truthful_qa":"truthfulqa","winogrande":"winogrande","gsm8k":"gsm8k","ifeval":"ifeval","humaneval":"humaneval","pass@1":"humaneval","bbh":"bbh","big bench hard":"bbh","gpqa":"gpqa","gpqa_diamond":"gpqa","hle":"hle"}

def _norm_bench(raw):
    r = raw.lower().strip().replace("-", "_").replace(" ", "_")
    if r in _BENCH_ALIASES: return _BENCH_ALIASES[r]
    return next((v for k, v in _BENCH_ALIASES.items() if k.replace(" ", "_") in r), None)

def _bench_to_avg(scores):
    w = {"mmlu": 3, "mmlu_pro": 3.5, "arc": 2, "hellaswag": 1.5, "truthfulqa": 1.5, "winogrande": 1, "gsm8k": 1.5, "ifeval": 2, "bbh": 2, "gpqa": 2.5, "hle": 4}
    tv = tw = 0.0
    for b, wt in w.items():
        v = scores.get(b)
        if v and v > 0: tv += float(v) * wt; tw += wt
    return round(tv / tw, 2) if tw > 0 else 0.0

def _scores_from_static(model_id):
    if model_id in KNOWN_BENCHMARKS: return dict(KNOWN_BENCHMARKS[model_id])
    short = model_id.split("/")[-1].lower()
    if short in _KNOWN_SHORT: return dict(KNOWN_BENCHMARKS[_KNOWN_SHORT[short]])
    return next((dict(KNOWN_BENCHMARKS[kfull]) for ks, kfull in _KNOWN_SHORT.items() if ks in short or short in ks), {})

def _scores_from_card(model):
    card = model.get("cardData") or {}
    if isinstance(card, str):
        try: card = json.loads(card)
        except Exception: return {}
    scores = {}
    for entry in (card.get("model-index") or []):
        for result in (entry.get("results") or []):
            for metric in (result.get("metrics") or []):
                mval = metric.get("value")
                if mval is None: continue
                try: mval = float(str(mval).replace("%", "").strip())
                except Exception: continue
                if 0 < mval <= 1.0: mval *= 100.0
                canonical = _norm_bench(str(metric.get("name", "") or metric.get("type", "")))
                if canonical and mval > 0 and scores.get(canonical, 0) < mval: scores[canonical] = mval
    return scores

_README_RE = re.compile(r'(?:^|\|)\s*(?P<bench>mmlu|arc[^|]*|hellaswag|truthfulqa|winogrande|gsm8k|ifeval|bbh|gpqa|humaneval|pass@1)[^\|]*\|[^\|]*?\|\s*(?P<val>\d{1,3}(?:\.\d{1,4})?)\s*(?:%|\|)', re.I | re.M)

def _scores_from_readme(text):
    scores = {}
    for m in _README_RE.finditer(text):
        bench = _norm_bench(m.group("bench"))
        if not bench: continue
        try: val = float(m.group("val"))
        except Exception: continue
        if 0 < val <= 1.0: val *= 100.0
        if 1.0 < val <= 100.0 and scores.get(bench, 0) < val: scores[bench] = val
    return scores

async def _fetch_readme(client, model_id):
    try:
        r = await client.get(f"https://huggingface.co/{model_id}/raw/main/README.md", timeout=httpx.Timeout(connect=4.0, read=8.0, write=4.0, pool=4.0))
        return r.text[:80000] if r.status_code == 200 else ""
    except Exception: return ""

def _base_chain(model):
    card = model.get("cardData") or {}
    if isinstance(card, str):
        try: card = json.loads(card)
        except Exception: card = {}
    bm = card.get("base_model") or model.get("base_model") or []
    if isinstance(bm, str): bm = [bm]
    bases = [b for b in bm if b]
    name_l = model.get("id", "").split("/")[-1].lower()
    for ks, kfull in _KNOWN_SHORT.items():
        if ks in name_l and kfull not in bases: bases.append(kfull)
    return bases[:4]

async def _resolve_bench(client, model, deep=False):
    mid = model.get("id", "")
    s = _scores_from_static(mid)
    if s and s.get("average", 0) > 0: return s
    cs = _scores_from_card(model)
    if cs:
        avg = _bench_to_avg(cs)
        if avg > 0: return {**cs, "average": avg, "confidence": 0.9, "source": "card-metadata", "inferred": False}
    for base_id in _base_chain(model):
        bs = _scores_from_static(base_id)
        if bs and bs.get("average", 0) > 0:
            return {**{k: v for k, v in bs.items() if k not in ("confidence","source","inferred")}, "confidence": bs.get("confidence",1.0)*0.8, "source": f"base-inherit:{base_id.split('/')[-1]}", "inferred": True, "average": bs.get("average",0)}
    if not deep: return {}
    readme = await _fetch_readme(client, mid)
    if readme:
        rs = _scores_from_readme(readme)
        avg = _bench_to_avg(rs)
        if avg > 0: return {**rs, "average": avg, "confidence": 0.85, "source": "readme-table", "inferred": False}
    return {}

async def _resolve_all_bench(models, deep=False):
    results = {}
    async with httpx.AsyncClient(timeout=httpx.Timeout(connect=5.0, read=8.0, write=4.0, pool=10.0), headers={"User-Agent": "Mozilla/5.0"}) as client:
        for i in range(0, len(models), 20):
            batch = models[i:i+20]
            try: batch_r = await asyncio.wait_for(asyncio.gather(*[_resolve_bench(client, m, deep) for m in batch], return_exceptions=True), timeout=60.0)
            except asyncio.TimeoutError: batch_r = [{}] * len(batch)
            for m, r in zip(batch, batch_r): results[m.get("id", "")] = r if isinstance(r, dict) else {}
    return results

# --- HuggingFace Model Fetch ---

def parse_params_b(name):
    nl = name.lower().replace("_", "-")
    for pat in (r'(\d+\.?\d*)-?b(?:illion)?(?:\b|[_\-])', r'(?:^|[_\-])(\d+\.?\d*)b(?:[_\-]|$)', r'\b(\d+\.?\d*)b\b'):
        m = re.search(pat, nl)
        if m:
            v = float(m.group(1))
            if 0.1 <= v <= 500: return v
    return None

def extract_params(m):
    st = (m.get("safetensors") or {}).get("total")
    if st and st > 1e6: return round(st / 1e9, 2)
    return next((parse_params_b(str(m.get(f,""))) for f in ("id","modelId") if parse_params_b(str(m.get(f,"")))), None)

def gguf_quants_from_siblings(siblings):
    found = set()
    for s in (siblings or []):
        fn = s.get("rfilename", "").upper()
        if not fn.endswith(".GGUF"): continue
        for q in QUANTS:
            if q.upper() in fn.replace("-", "_"): found.add(q); break
    return list(found)

async def _fetch_gguf_models(preset, extra_query):
    seen, all_m = set(), []
    queries = ([extra_query.strip()] if extra_query.strip() else []) + list(preset.get("queries", []))
    reqs = []
    for q in queries[:3]:
        for sort in ("likes", "downloads"): reqs.append({"search": q, "filter": ["text-generation", "gguf"], "library": "gguf", "sort": sort, "direction": "-1", "limit": 100, "full": "true"})
    for sort in ("likes", "downloads"): reqs.append({"filter": ["text-generation", "gguf"], "library": "gguf", "sort": sort, "direction": "-1", "limit": 200, "full": "true"})
    for tag in preset.get("any_tags", [])[:3]: reqs.append({"filter": ["text-generation", "gguf", tag], "library": "gguf", "sort": "likes", "direction": "-1", "limit": 60, "full": "true"})
    async def _one(c, p):
        try: r = await c.get("https://huggingface.co/api/models", params=p); return r.json() if r.status_code == 200 else []
        except Exception: return []
    async with httpx.AsyncClient(timeout=httpx.Timeout(connect=5.0, read=12.0, write=3.0, pool=10.0), headers={"User-Agent": "Mozilla/5.0"}) as c:
        try: batches = await asyncio.wait_for(asyncio.gather(*[_one(c, p) for p in reqs]), timeout=30.0)
        except asyncio.TimeoutError: batches = []
    for batch in (batches or []):
        if isinstance(batch, list):
            for m in batch:
                mid = m.get("id", "")
                if mid and mid not in seen: seen.add(mid); all_m.append(m)
    return all_m

# --- Search Job (background thread + poll) ---

_job = {"running": False, "status": "idle", "step": "", "progress": [], "filtered": None, "result": None, "error": None, "params": None}
_job_lock = threading.Lock()

def _job_up(**kw):
    with _job_lock:
        _job.update(kw)
        if "step" in kw: _job["progress"].append(f"[{datetime.now().strftime('%H:%M:%S')}] {kw['step']}")

def _job_reset():
    with _job_lock: _job.update(running=False, status="idle", step="", progress=[], filtered=None, result=None, error=None, params=None)

def _run_thread(params):
    loop = asyncio.new_event_loop(); asyncio.set_event_loop(loop)
    try: _job_up(running=False, status="Done.", result=loop.run_until_complete(_pipeline(params)))
    except Exception: _job_up(running=False, status="Failed.", error=traceback.format_exc())
    finally: loop.close()

def _write_csv(rows, key):
    if not rows: return
    fields = ["rank","id","quant","total_score","lb_avg","quant_qual","speed","mem_head","ctx_fit","intel","popularity","leaderboard_avg","params_b","tps","model_gb","kv_gb","total_mem_gb","time_10k_s","mode","likes","downloads"]
    try:
        with open(DATA_DIR / f"analysis_{key}.csv", "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore"); w.writeheader()
            for i, r in enumerate(rows):
                p, ss = r["perf"], r["sub_scores"]
                w.writerow({"rank": i+1, "id": r["id"], "quant": r["quant"], "total_score": r["total_score"], **{d: ss[d] for d in SCORE_DIMS}, "leaderboard_avg": r["leaderboard_avg"], "params_b": r["params_b"],
                            "tps": p["tps"], "model_gb": p["model_gb"], "kv_gb": p["kv_gb"], "total_mem_gb": p["total_mem_gb"], "time_10k_s": p["time_10k_s"], "mode": p["mode"], "likes": r["likes"], "downloads": r["downloads"]})
    except Exception as e: print(f"[ai_calc] CSV write error: {e}")

async def _pipeline(params):
    hw, deep = params["hw"], params.get("deep_scan", False)
    must_w = [w.strip().lower() for w in params["must_contain"].split(",") if w.strip()]
    any_w = [w.strip().lower() for w in params["any_contain"].split(",") if w.strip()]
    excl_w = [w.strip().lower() for w in params["exclude"].split(",") if w.strip()]
    preset = TASK_PRESETS.get(params["task_preset"], next(iter(TASK_PRESETS.values()), {}))
    key = re.sub(r'[^\w]', '_', f"{params['task_preset']}_{params['min_params']}_{params['max_params']}_{params['extra_query'][:20]}")[:60]
    cache = DATA_DIR / f"hf_{key}.json"
    raw = None
    if not params.get("force_refresh") and cache.exists():
        age = (datetime.now() - datetime.fromtimestamp(cache.stat().st_mtime)).total_seconds()
        if age < 21600:
            try: raw = json.loads(cache.read_text()); _job_up(step=f"Loaded {len(raw)} from cache ({int(age/60)}m old)")
            except Exception: raw = None
    if raw is None:
        _job_up(status="Fetching from HuggingFace...", step="Starting HF queries")
        raw = await _fetch_gguf_models(preset, params["extra_query"])
        _job_up(step=f"Fetched {len(raw)} raw models - caching")
        cache.write_text(json.dumps(raw))
    params["fetched"] = len(raw)
    min_quality = QUANTS.get(params.get("min_quant","Q2_K"), QUANTS["Q2_K"])["quality"]
    _job_up(status=f"Filtering {len(raw)} models...", step=f"Params {params['min_params']}-{params['max_params']} | min_tps={params['min_tps']} | min_quant={params.get('min_quant','Q2_K')}")
    filtered, counts = [], {"oom": 0, "speed": 0, "params": 0, "quant": 0, "tags": 0, "exclude": 0}
    for idx, m in enumerate(raw):
        if idx % 50 == 0 and idx > 0: _job_up(step=f"  {idx}/{len(raw)} checked, {len(filtered)} passing")
        blob = (m.get("id", "") + " " + " ".join(t.lower() for t in (m.get("tags") or []))).lower()
        if excl_w and any(w in blob for w in excl_w): counts["exclude"] += 1; continue
        if must_w and not all(w in blob for w in must_w): counts["tags"] += 1; continue
        if any_w and not any(w in blob for w in any_w): counts["tags"] += 1; continue
        pb = extract_params(m)
        if pb is None or not (params["min_params"] <= pb <= params["max_params"]): counts["params"] += 1; continue
        avail = {q for q in gguf_quants_from_siblings(m.get("siblings") or []) if q in QUANTS and QUANTS[q]["quality"] >= min_quality}  # quality floor applied before a model is even kept, so a high-scoring Q2 can't win a search where it's below the acceptable floor (e.g. creative writing)
        if not avail: counts["quant"] += 1; continue
        any_fits = any_fast = False
        for q in sorted(avail, key=lambda x: QUANTS[x]["bpp"]):
            p = full_perf(pb, q, params["ctx_tokens"], hw, params.get("kv_bits", 8))
            if p["mem_ok"]:
                any_fits = True
                if p["tps"] >= params["min_tps"]: any_fast = True; break
        if not any_fits: counts["oom"] += 1; continue
        if not any_fast: counts["speed"] += 1; continue
        filtered.append({**m, "params_b": pb, "avail_quants": sorted(avail, key=lambda q: QUANTS[q]["bpp"]), "leaderboard_avg": 0.0, "lb_detail": {}, "bench_confidence": 0.0, "bench_source": "none", "bench_inferred": False})
    _job_up(step=f"Filter done: {len(filtered)} pass all hard gates")
    _job_up(status="Resolving benchmarks...", step=f"Checking {len(filtered)} models")
    bench_res = await _resolve_all_bench(filtered, deep)
    lb_hits = 0
    for m in filtered:
        bs = bench_res.get(m.get("id", ""), {}); avg = bs.get("average", 0.0) or 0.0
        m.update(leaderboard_avg=round(float(avg), 2) if avg else 0.0, lb_detail={k: v for k, v in bs.items() if k not in ("confidence","source","inferred","average")},
                  bench_confidence=float(bs.get("confidence", 0.0)), bench_source=str(bs.get("source", "none")), bench_inferred=bool(bs.get("inferred", False)))
        if avg > 0: lb_hits += 1
    _job_up(step=f"Bench resolved: {lb_hits}/{len(filtered)} have scores")
    with _job_lock: _job["filtered"] = filtered
    _job_up(status="Scoring...", step=f"target_tps={params['target_tps']}")
    rows = _rank_filtered(filtered, hw, params["ctx_tokens"], params["target_tps"], params["weights"], params["top_n"], params.get("kv_bits", 8))
    _write_csv(rows, key)
    top = rows[0] if rows else None
    _job_up(step=f"Done. Top: {top['id']} [{top['quant']}] score={top['total_score']}" if top else "Done - no results")
    return {"rows": rows, "stats": {"fetched": params["fetched"], "passed": len(filtered), "ranked": len(rows), **counts, "lb_hits": lb_hits, "csv": f"data/ai_tools/ai_calc/analysis_{key}.csv", "deep": deep}, "weights": params["weights"]}

# --- Session Monitor ---

def _tcolor(tps): return "#00ffa2" if tps >= 8 else "#ffcc00" if tps >= 3 else "#ff8c42" if tps >= 0.5 else "#ff4444"

def _parse_loaded(name):
    pb = parse_params_b(name) or parse_params_b(name.split(":")[-1])
    nu = name.upper().replace("-", "_")
    return pb, next((q for q in sorted(QUANTS, key=len, reverse=True) if q.upper() in nu), "Q4_K_M")

async def _fetch_running(conn):
    try:
        async with httpx.AsyncClient(timeout=httpx.Timeout(connect=3.0, read=5.0, write=3.0, pool=3.0)) as c:
            r = await c.get(f"{AIM.connections._base(conn)}/api/ps")
            return r.json().get("models", []) if r.status_code == 200 else []
    except Exception: return []

def _session_card_html(m, hw):
    name = m.get("name", "")
    vram_gb = m.get("size_vram", m.get("size", 0)) / 1e9
    pb, q = _parse_loaded(name)
    tps = full_perf(pb, q, 32768, hw)["tps"] if pb else 0
    pct = min(vram_gb / max(total_vram(hw), 0.001) * 100, 100)
    qlabel = f'{q} ({QUANTS.get(q, QUANTS["Q4_K_M"]).get("quality",0)*100:.0f}% quality)' if pb else ""
    return f"""<div class="glass" style="padding:.6rem .8rem;margin-bottom:.4rem">
                   <div style="font-family:var(--font-mono);font-weight:700;font-size:.8rem;margin-bottom:.25rem">{UI.escape(name)}</div>
                   <div style="display:flex;gap:.7rem;font-size:.74rem;flex-wrap:wrap"><span style="color:{_tcolor(tps)}">~{tps:.1f} t/s</span><span>{qlabel}</span><span style="color:var(--text_muted)">{pb or '?'}B params</span></div>
                   <div style="display:flex;height:8px;border-radius:4px;overflow:hidden;background:var(--bg);border:1px solid var(--border);margin-top:.3rem"><div style="width:{pct:.1f}%;background:#3d9aff"></div></div>
                   <div style="font-size:.65rem;color:var(--text_muted);margin-top:.15rem">{vram_gb:.1f} / {total_vram(hw):.0f} GB VRAM</div>
               </div>"""

# --- SVG chart helpers (server-rendered, no client charting lib - matches this platform's no-framework-JS default) ---

def _svg_bar_chart(items: list, value_key: str, label_key: str, width=680, height=300, title="", color_key=None, color_true="#3d9aff", color_false="#ffaa44") -> str:
    if not items: return '<div class="dim tiny" style="padding:1rem">No data to chart.</div>'
    pad_l, pad_b, pad_t = 40, 60, 24
    plot_w, plot_h = width - pad_l - 20, height - pad_b - pad_t
    maxv = max(i[value_key] for i in items) or 1
    bw = plot_w / len(items)
    bars = labels = ""
    for idx, it in enumerate(items):
        v = it[value_key]; bh = (v / maxv) * plot_h
        x, y = pad_l + idx * bw + bw * 0.12, pad_t + plot_h - (v / maxv) * plot_h
        color = color_true if (color_key is None or it.get(color_key)) else color_false
        bars += f'<rect x="{x:.1f}" y="{y:.1f}" width="{bw*0.76:.1f}" height="{bh:.1f}" fill="{color}" rx="3"/><text x="{x+bw*0.38:.1f}" y="{y-4:.1f}" font-size="10" fill="var(--text)" text-anchor="middle">{v:.1f}</text>'
        labels += f'<text x="{x+bw*0.38:.1f}" y="{pad_t+plot_h+16:.1f}" font-size="9" fill="var(--text_muted)" text-anchor="middle" transform="rotate(28 {x+bw*0.38:.1f} {pad_t+plot_h+16:.1f})">{UI.escape(str(it[label_key])[:16])}</text>'
    axis = f'<line x1="{pad_l}" y1="{pad_t+plot_h}" x2="{pad_l+plot_w}" y2="{pad_t+plot_h}" stroke="var(--border)"/><line x1="{pad_l}" y1="{pad_t}" x2="{pad_l}" y2="{pad_t+plot_h}" stroke="var(--border)"/>'
    title_html = f'<text x="{width/2}" y="14" font-size="11" fill="var(--text_muted)" text-anchor="middle">{UI.escape(title)}</text>' if title else ""
    legend = f'<div style="margin-top:.3rem;font-size:.7rem;color:var(--text_muted)"><span style="color:{color_true}">&#9632;</span> your search results &nbsp; <span style="color:{color_false}">&#9632;</span> frontier reference</div>' if color_key else ""
    return f'<div><svg viewBox="0 0 {width} {height}" style="width:100%;max-width:{width}px;height:auto;font-family:var(--font-mono)">{title_html}{axis}{bars}{labels}</svg>{legend}</div>'

def _svg_path_d(pts: list) -> str: return " ".join(f"{'M' if j==0 else 'L'}{x:.1f},{y:.1f}" for j, (x, y) in enumerate(pts))

def _svg_line_chart(series: list, width=680, height=320, title="") -> str:
    """series: [{"label":str, "points":[(x,y),...]}, ...]. Auto-scales both axes across all series combined."""
    all_pts = [p for s in series for p in s["points"]]
    if not all_pts: return '<div class="dim" style="padding:1rem">No data to chart.</div>'
    pad_l, pad_b, pad_t, pad_r = 54, 36, 24, 20
    plot_w, plot_h = width - pad_l - pad_r, height - pad_b - pad_t
    xs, ys = [p[0] for p in all_pts], [p[1] for p in all_pts]
    xmin, xmax, ymin, ymax = min(xs), max(xs) or 1, 0, max(ys) or 1
    def _sx(x): return pad_l + ((x - xmin) / max(xmax - xmin, 1e-9)) * plot_w
    def _sy(y): return pad_t + plot_h - ((y - ymin) / max(ymax - ymin, 1e-9)) * plot_h
    paths = dots = legend = ""
    palette = ["#3d9aff", "#ff9a3c", "#00ffa2", "#b06aff"]
    for i, s in enumerate(series):
        pts = sorted(s["points"]); col = palette[i % len(palette)]
        sx_pts = [(_sx(x), _sy(y)) for x, y in pts]
        paths += f'<path d="{_svg_path_d(sx_pts)}" fill="none" stroke="{col}" stroke-width="2"/>'
        dots += "".join(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="2.5" fill="{col}"/>' for x, y in sx_pts)
        legend += f'<span style="display:inline-flex;align-items:center;gap:.25rem;margin-right:.8rem;font-size:.7rem;color:var(--text_muted)"><span style="width:.6rem;height:.6rem;border-radius:50%;background:{col};display:inline-block"></span>{UI.escape(s["label"])}</span>'
    axis = f'<line x1="{pad_l}" y1="{pad_t+plot_h}" x2="{pad_l+plot_w}" y2="{pad_t+plot_h}" stroke="var(--border)"/><line x1="{pad_l}" y1="{pad_t}" x2="{pad_l}" y2="{pad_t+plot_h}" stroke="var(--border)"/>'
    yticks = "".join(f'<text x="{pad_l-6}" y="{_sy(ymin+frac*(ymax-ymin))+3:.1f}" font-size="9" fill="var(--text_muted)" text-anchor="end">{ymin+frac*(ymax-ymin):.1f}</text>' for frac in (0,.25,.5,.75,1))
    xticks = "".join(f'<text x="{_sx(xmin+frac*(xmax-xmin)):.1f}" y="{pad_t+plot_h+14}" font-size="9" fill="var(--text_muted)" text-anchor="middle">{xmin+frac*(xmax-xmin):.0f}</text>' for frac in (0,.25,.5,.75,1))
    title_html = f'<text x="{width/2}" y="14" font-size="11" fill="var(--text_muted)" text-anchor="middle">{UI.escape(title)}</text>' if title else ""
    return f'<div><svg viewBox="0 0 {width} {height}" style="width:100%;max-width:{width}px;height:auto;font-family:var(--font-mono)">{title_html}{axis}{yticks}{xticks}{paths}{dots}</svg><div style="margin-top:.3rem">{legend}</div></div>'

# --- UI: Calculator ---

def _quant_opts(selected="Q4_K_M"): return "".join(f'<option value="{q}" {"selected" if q==selected else ""}>{q} ({round(QUANTS[q]["quality"]*100)}% quality, {QUANTS[q]["bpp"]:.3f} bpp)</option>' for q in QUANTS)

def _panel_calc():
    model_opts = "".join(f'<option value="{k}">{k} - {v["note"]}</option>' for k, v in IMAGE_MODELS.items())
    return f"""<div style="padding:.9rem;height:100%;overflow-y:auto;box-sizing:border-box">
        <div class="fsect-hd">Model Speed / Memory / Context Calculator</div>
        <form class="pform" hx-post="/im/in" hx-target="#calc-out" hx-swap="innerHTML">
            <input type="hidden" name="type" value="ai_calc_run"><input type="hidden" name="branch" value="ai_calc"><input type="hidden" name="lvl" value="2">
            <div class="frow wrap">
                <label>Hardware{_hw_select_html()}</label>
                <label>Params (B)<input class="fin" name="params_b" type="number" step="any" value="7"></label>
                <label>Quant<select class="fin" name="quant">{_quant_opts()}</select></label>
                <label>Context (tokens)<input class="fin" name="ctx_tokens" type="number" value="20992"></label>
                <label>KV Quant<select class="fin" name="kv_bits"><option value="16">FP16</option><option value="8" selected>INT8</option><option value="4">INT4</option></select></label>
                <button class="rbtn" type="submit">Calculate</button>
            </div>
        </form>
        <div id="calc-out" class="rzone"><div class="placeholder">Enter parameters above - shows speed, memory layout, context limits, and every quantization side by side.</div></div>
        <details style="margin-top:.9rem">
            <summary style="cursor:pointer;font-size:.75rem;color:var(--text_muted)">Image/Video Model Estimate</summary>
            <form class="pform" hx-post="/im/in" hx-target="#img-calc-out" hx-swap="innerHTML" style="margin-top:.4rem">
                <input type="hidden" name="type" value="ai_calc_image"><input type="hidden" name="branch" value="ai_calc"><input type="hidden" name="lvl" value="2">
                <div class="frow wrap">
                    <label>Hardware{_hw_select_html()}</label>
                    <label>Model<select class="fin" name="image_model">{model_opts}</select></label>
                    <label>Steps<input class="fin" name="steps" type="number" value="20"></label>
                    <button class="rbtn" type="submit">Estimate</button>
                </div>
            </form>
            <div id="img-calc-out" class="rzone"></div>
        </details>
    </div>"""

def _calc_result_html(pb, quant, ctx, hw, kv_bits, cnode_id=""):
    p = full_perf(pb, quant, ctx, hw, kv_bits)
    tm = total_mem(hw)
    layers, hidden, attn, kv_heads = arch_for(pb)
    avail = max(tm - p["model_gb"], 0)
    max_ctx = max_ctx_tokens(pb, avail, kv_bits)
    t10 = f"{p['time_10k_s']:.0f}s ({p['time_10k_s']/60:.1f}m)" if p.get("time_10k_s") else "N/A"
    okc = "#00ffa2" if p["mem_ok"] else "#ff5f5f"
    stats = f"""<div style="display:grid;grid-template-columns:repeat(auto-fit,minmax(130px,1fr));gap:.5rem;margin-bottom:.6rem">
        <div class="stat" style="--sc:{_tcolor(p['tps'])}"><div class="sl">Generation</div><div class="sv">{p['tps']}<span class="su"> t/s</span></div><div class="ss">{p['mode']}</div></div>
        <div class="stat" style="--sc:#3d9aff"><div class="sl">Prefill</div><div class="sv">{p['prefill_tps']:.0f}<span class="su"> t/s</span></div></div>
        <div class="stat" style="--sc:#b06aff"><div class="sl">Model Size</div><div class="sv">{p['model_gb']}<span class="su"> GB</span></div></div>
        <div class="stat" style="--sc:#ff9a3c"><div class="sl">KV Cache</div><div class="sv">{p['kv_gb']:.3f}<span class="su"> GB</span></div></div>
        <div class="stat" style="--sc:{okc}"><div class="sl">Memory</div><div class="sv" style="font-size:1rem;padding-top:.2rem">{"Fits" if p["mem_ok"] else "OOM"}</div><div class="ss">{p['total_mem_gb']:.1f} / {tm:.1f} GB</div></div>
        <div class="stat" style="--sc:#00ffa2"><div class="sl">{ctx//1000}K Token Run</div><div class="sv" style="font-size:1rem;padding-top:.2rem">{t10}</div></div>
    </div>
    <div style="display:grid;grid-template-columns:1fr 1fr;gap:.5rem;font-size:.78rem;margin-bottom:.7rem">
        <div class="glass" style="padding:.5rem .7rem"><div style="color:var(--text_muted);font-size:.63rem;text-transform:uppercase;margin-bottom:.2rem">Architecture (estimated)</div>
            <div style="font-family:var(--font-mono);line-height:1.7;font-size:.7rem">Layers {layers} | Hidden {hidden} | Attn {attn} | KV-heads {kv_heads}<br>{kv_bytes_per_token(pb,kv_bits)//1024:.1f} KB/token</div></div>
        <div class="glass" style="padding:.5rem .7rem"><div style="color:var(--text_muted);font-size:.63rem;text-transform:uppercase;margin-bottom:.2rem">Context Limits</div>
            <div style="font-family:var(--font-mono);line-height:1.7;font-size:.7rem">Free after model: {avail:.1f} GB<br>Max context: <span style="color:var(--accent)">{max_ctx//1000:.0f}K tokens</span></div></div>
    </div>
    <div style="display:flex;gap:.5rem;align-items:center;margin-bottom:.5rem">
        <input type="text" id="calc-actual-tps" placeholder="Actual observed t/s (optional)" class="fin" style="max-width:14rem">
        <button class="rbtn" type="button" onclick="calcLogActual('{cnode_id}',{pb},'{quant}',{ctx})">Log for calibration</button>
        <span id="calc-log-msg" style="font-size:.7rem;color:var(--text_muted)"></span>
    </div>"""
    rows = "".join(f"""<tr class="{'oom' if not pq['fits_total'] else ''}"><td class="qn">{q}</td><td>{pq["model_gb"]} GB</td><td style="color:{'#555' if not pq['fits_total'] else _tcolor(pq['tps'])};font-weight:700">{pq["tps"]}</td><td>{pq.get("kv_gb",0):.3f}</td><td>{pq.get("total_mem_gb",0):.1f}</td><td style="color:#3d9aff">{round(QUANTS[q]["quality"]*100)}%</td></tr>"""
                     for q, pq in ((q, full_perf(pb, q, ctx, hw)) for q in QUANTS))
    return stats + f"""<div style="font-size:.68rem;color:var(--text_muted);margin:.3rem 0">All quantizations at this size/context:</div>
    <div class="tbl-scroll"><table class="cmp-table"><thead><tr><th>Quant</th><th>Size</th><th>T/s</th><th>KV GB</th><th>Total GB</th><th>Quality</th></tr></thead><tbody>{rows}</tbody></table></div>"""

# --- UI: Model Finder ---

def _weight_slider(dim, weights):
    val = weights.get(dim, DEFAULT_WEIGHTS.get(dim, 1))
    return (f'<label class="wsl" style="--wc:{SCORE_DIM_COLORS.get(dim,"#888")}"><span class="wsln">{SCORE_DIM_LABELS.get(dim,dim)}</span>'
            f'<input class="wslider" type="range" name="w_{dim}" min="0" max="10" step="1" value="{val}" oninput="this.parentElement.querySelector(\'.wsv\').textContent=this.value"><span class="wsv">{val}</span></label>')

def _score_bar(dim, raw, w, total_w):
    color, contrib = SCORE_DIM_COLORS.get(dim, "#888"), raw * w / max(total_w, 0.001)
    return (f'<div class="sbrow"><span class="sbl" style="color:{color}">{SCORE_DIM_LABELS.get(dim,dim)}</span><div class="sbbar"><div class="sbfill" style="width:{raw*100:.1f}%;background:{color}"></div></div>'
            f'<span class="sbv">{raw:.3f}</span><span class="sbw">w={w:.0f}</span><span class="sbc" style="color:{color}">+{contrib:.4f}</span></div>')

def _lb_pills(lb, source="", confidence=0, inferred=False):
    if not lb: return '<span class="dim tiny">no benchmark data</span>'
    parts = []
    for key, short in (("mmlu","MMLU"),("arc","ARC"),("hellaswag","HS"),("gsm8k","GSM8K"),("ifeval","IFEval"),("bbh","BBH"),("gpqa","GPQA"),("hle","HLE"),("average","Avg")):
        v = lb.get(key)
        if v and float(v) > 0:
            v = float(v); col = "#00ffa2" if v>=75 else "#ffcc00" if v>=55 else "#ff9a3c" if v>=40 else "#888"
            parts.append(f'<span class="lbp" style="color:{col}">{short} {v:.1f}</span>')
    src_labels = {"official": ("anchor", "#00ffa2"), "reported": ("~anchor", "#88c"), "card-metadata": ("card", "#3d9aff"), "readme-table": ("readme", "#ffcc00")}
    if source and source != "none":
        slabel, scol = next((v for k, v in src_labels.items() if source.startswith(k)), (source, "#888"))
        if source.startswith("base-inherit"): slabel, scol = "base&#x2191;", "#ff9a3c"
        parts.append(f'<span class="lbp" style="color:{scol}">{slabel}{" ~inferred" if inferred else ""}{f" {confidence*100:.0f}%" if confidence<0.95 else ""}</span>')
    return "".join(parts)

def _panel_search():
    task_opts = "".join(f'<option value="{k}">{v["label"]}</option>' for k, v in TASK_PRESETS.items())
    wsliders = "".join(_weight_slider(d, DEFAULT_WEIGHTS) for d in SCORE_DIMS)
    return f"""<div style="padding:.9rem;height:100%;overflow-y:auto;box-sizing:border-box">
        <div class="fsect-hd">Model Finder</div>
        <form class="pform" hx-post="/im/in" hx-target="#srch-out" hx-swap="innerHTML">
            <input type="hidden" name="type" value="ai_calc_search"><input type="hidden" name="branch" value="ai_calc"><input type="hidden" name="lvl" value="2">
            <div class="frow wrap">
                <label>Hardware{_hw_select_html()}</label>
                <label>Task<select class="fin" name="task_preset" onchange="applyPreset(this.value)">{task_opts}</select></label>
                <label>Min params (B)<input class="fin" name="min_params" type="number" step="any" value="1"></label>
                <label>Max params (B)<input class="fin" name="max_params" type="number" step="any" value="35"></label>
                <label>Context tokens<input class="fin" name="ctx_tokens" type="number" value="20992"></label>
            </div>
            <div class="frow wrap">
                <label>Min T/s (hard floor)<input class="fin" name="min_tps" type="number" step="any" value="1.0"></label>
                <label>Target T/s (score ref)<input class="fin" name="target_tps" type="number" step="any" value="5.0"></label>
                <label>KV Cache Quant<select class="fin" name="kv_bits"><option value="16">FP16</option><option value="8" selected>INT8</option><option value="4">INT4</option></select></label>
                <label>Minimum Model Quant<select class="fin" name="min_quant">{"".join(f'<option value="{q}" {"selected" if q=="Q3_K_M" else ""}>{q}</option>' for q in QUANTS)}</select><span class="dim tiny">excludes anything below this quality floor regardless of score</span></label>
            </div>
            <div class="frow wrap">
                <label style="flex:2">Must contain ALL<input class="fin" name="must_contain" type="text" placeholder="instruct, chat"></label>
                <label style="flex:2">Must contain ANY<input class="fin" name="any_contain" type="text" placeholder="creative, roleplay"></label>
                <label style="flex:2">Exclude terms<input class="fin" name="exclude" type="text" placeholder="vision, embed, base"></label>
            </div>
            <div class="frow wrap">
                <label style="flex:2">Extra HF query<input class="fin" name="extra_query" type="text"></label>
                <label>Results cap<input class="fin" name="top_n" type="number" value="40"></label>
            </div>
            <div class="fsect-hd" style="margin-top:.7rem">Scoring Weights</div>
            <div class="wsblock">{wsliders}</div>
            <div class="frow" style="align-items:center;gap:1rem;margin-top:.5rem;flex-wrap:wrap">
                <label style="flex-direction:row;align-items:center;gap:.4rem;flex:0"><input type="checkbox" name="force_refresh" value="1"> Force HF refresh</label>
                <label style="flex-direction:row;align-items:center;gap:.4rem;flex:0" title="Also scrapes each candidate's README.md - slower, higher hit rate"><input type="checkbox" name="deep_scan" value="1"> Deep bench scan</label>
                <button class="rbtn" type="submit">Search &amp; Rank</button>
                <button type="button" class="rbtn" style="background:var(--bg_panel)" hx-post="/im/in" hx-target="#srch-out" hx-swap="innerHTML" hx-include="closest form" hx-vals='{{"type":"ai_calc_rerank","branch":"ai_calc","lvl":2}}' title="Re-scores the last fetched result set with the current weights above, without re-querying HuggingFace">&#x21BB; Re-rank cached results</button>
            </div>
        </form>
        <div id="srch-out" class="rzone"><div class="placeholder">Configure and run search above</div></div>
    </div>"""

def _search_html(res):
    rows, s, weights = res["rows"], res["stats"], res.get("weights", DEFAULT_WEIGHTS)
    total_w = sum(weights.get(d, 0) for d in SCORE_DIMS)
    sbar = (f'<div class="info-bar">Fetched <b>{s.get("fetched","?")}</b> | Filtered <b>{s.get("passed","?")}</b> | Ranked <b>{s.get("ranked",len(rows))}</b> | '
            f'OOM <b>{s.get("oom","?")}</b> | Slow <b>{s.get("speed","?")}</b> | Bench <b>{s.get("lb_hits","?")}</b> hits{" [deep]" if s.get("deep") else ""} <span class="dim">&#8594; {Path(s.get("csv","")).name}</span></div>')
    if not rows: return sbar + '<div class="placeholder">No results. Try a wider param range, lower min T/s, a lower minimum quant floor, or fewer filters.</div>'
    frontier_html = _frontier_overlay_chart(rows)
    wleg = '<div class="wleg">' + "".join(f'<span class="wli"><span class="wld" style="background:{SCORE_DIM_COLORS.get(d,"#888")}"></span>{SCORE_DIM_LABELS.get(d,d)} <b>w={weights.get(d,0):.0f}</b></span>' for d in SCORE_DIMS) + '</div>'
    RC = ["#ffd700", "#c0c0c0", "#cd7f32"]
    cards = ""
    for i, r in enumerate(rows):
        p, ss = r["perf"], r["sub_scores"]
        hfu = f"https://huggingface.co/{r['id']}"
        lb_v = r.get("leaderboard_avg", 0.0)
        lb_bdg = f'<span class="bdg bl">LB {"~" if r.get("bench_inferred") else ""}{lb_v:.1f}</span>' if lb_v > 0 else '<span class="bdg" style="opacity:.35">No bench</span>'
        score_bars = "".join(_score_bar(d, ss[d], weights.get(d, 0), total_w) for d in SCORE_DIMS)
        avail_qt = "".join(f'<span class="qt" style="border-color:{"var(--accent)" if q==r["quant"] else "var(--border)"};color:{"var(--accent)" if q==r["quant"] else "var(--text_muted)"}">{q}</span>' for q in r.get("avail_quants", []))
        cards += f"""<div class="mcard" style="--rc:{RC[i] if i<3 else 'var(--border)'}"><div class="mrank">#{i+1}</div><div class="mbody">
            <div class="mhead"><a class="mname" href="{hfu}" target="_blank" rel="noopener">{r['name']}</a><span class="mauthor">{r['author']}</span>{lb_bdg}
            <span class="bdg" style="background:var(--accent_dim);color:var(--accent)">{r['quant']}</span><span class="tscore">{r['total_score']:.4f}</span></div>
            <div class="mstats"><span class="ms" style="color:{_tcolor(p['tps'])}">~{p['tps']} t/s</span><span class="ms">{r['params_b']}B</span><span class="ms">{p['model_gb']} GB</span></div>
            <div class="sb-block">{score_bars}</div>
            <details class="mdet"><summary class="mdet-sum">&#9656; Benchmarks + quants</summary><div class="mdet-body">
                <div class="lb-row">{_lb_pills(r.get("lb_detail",{}), r.get("bench_source","none"), r.get("bench_confidence",0), r.get("bench_inferred",False))}</div>
                <div class="mqts" style="margin-top:.3rem">{avail_qt}</div>
                <div class="mpop" style="margin-top:.3rem"><span>&#9829; {r.get("likes",0):,}</span><span>&#8659; {r.get("downloads",0):,}</span><a href="{hfu}" target="_blank" class="hfl">HuggingFace &#8594;</a></div>
            </div></details></div></div>"""
    return f'{sbar}{frontier_html}{wleg}<div class="clist">{cards}</div>'

def _search_status_html():
    with _job_lock: running, status, progress, result, error = _job["running"], _job["status"], list(_job["progress"]), _job["result"], _job["error"]
    if error: return f'<div class="err-box">Pipeline failed:<br><pre style="font-size:.68rem;white-space:pre-wrap;margin-top:.4rem">{error[:1500]}</pre></div>'
    if result is not None and not running: return _search_html(result)
    log = "".join(f'<div class="log-line">{l}</div>' for l in progress[-25:])
    return f"""<div hx-post="/im/in" hx-vals='{{"type":"ai_calc_search_status","branch":"ai_calc","lvl":2}}' hx-trigger="every 2s" hx-target="#srch-out" hx-swap="innerHTML">
                   <div class="status-bar"><span class="spin-anim">[..]</span> {status}</div><div class="log-wrap">{log}</div></div>"""

# --- UI: Hardware ---

def _hw_form_html(cnode):
    fields = "".join(f'<label style="font-size:.73rem;color:var(--text_muted)">{_HW_LABELS[k]}<input class="hwin" name="{k}" type="number" step="any" value="{cnode.get(k, _HW_DEFAULTS[k])}"></label>' for k in _HW_FIELDS)
    return f"""<form class="glass" style="padding:.7rem;display:flex;flex-direction:column;gap:.4rem" hx-post="/im/in" hx-target="body" hx-swap="none">
        <input type="hidden" name="type" value="ai_calc_hw_save"><input type="hidden" name="branch" value="ai_calc"><input type="hidden" name="lvl" value="2"><input type="hidden" name="cid" value="{cnode['id']}">
        <div style="font-weight:600;font-size:.8rem">{UI.escape(cnode.get("label",cnode["id"]))}</div>
        {_hw_ref_html()}
        <div class="hwg">{fields}</div>
        <button type="submit" class="rbtn" style="align-self:flex-start">Save Hardware Specs</button>
    </form>"""

def _panel_hardware():
    rows = ""
    for c in AIM.resources.list_cnodes():
        has_hw = any(k in c for k in _HW_FIELDS)
        spec = f'{c.get("vram_gb",0)+c.get("shared_gb",0):.0f}+{c.get("sys_ram_gb",0):.0f}GB | {c.get("mem_bw_gbps",0):.0f}/{c.get("sys_ram_bw_gbps",0):.0f}GB/s' if has_hw else "no hardware specs yet"
        rows += f"""<div class="glass" style="padding:.5rem .7rem;margin-bottom:.3rem"><div style="display:flex;align-items:center;gap:.5rem">
            <span style="flex:1;font-weight:600;font-size:.82rem">{UI.escape(c.get("label",c["id"]))}</span><span class="dim tiny">{UI.escape(", ".join(c.get("tags",[])))}</span><span class="dim tiny" style="font-family:var(--font-mono)">{spec}</span>
            <button class="cm-qbtn" hx-post="/im/in" hx-target="body" hx-swap="none" hx-vals='{{"type":"ai_calc_hw_form","branch":"ai_calc","lvl":2,"cid":"{c["id"]}"}}'>Edit specs</button>
        </div></div>"""
    return f"""<div style="padding:.9rem;height:100%;overflow-y:auto;box-sizing:border-box;max-width:44rem">
        <div class="fsect-hd">Hardware Profiles (CNodes)</div>
        <p style="font-size:.75rem;color:var(--text_muted)">Hardware specs live directly on your CNodes - the same machines used for pipeline connection routing. Add specs to a CNode here to make it usable as a calculator/search hardware profile; a CNode with no specs still works fine for connection routing.</p>
        {rows or '<div class="dim tiny" style="padding:.5rem 0">No CNodes yet - add one via AI Manager Resource Pool button, then set its hardware specs here.</div>'}
        <div id="hw-form-slot" style="margin-top:.6rem"></div>
    </div>"""

# --- UI: Compare (frontier reference + sweep) ---

def _frontier_chart_html():
    items = sorted(({"name": r[0], "provider": r[1], "quality": r[2]} for r in PROPRIETARY_REFS), key=lambda i: i["quality"], reverse=True)
    chart = _svg_bar_chart(items, "quality", "name", title="Leaderboard-style composite quality score")
    rows = "".join(f'<tr><td>{UI.escape(r[0])}</td><td class="dim">{UI.escape(r[1])}</td><td>{r[2]}</td><td>{r[3]}</td><td>${r[4]}/M tok</td><td class="dim tiny">{UI.escape(r[5])}</td></tr>' for r in sorted(PROPRIETARY_REFS, key=lambda x: x[2], reverse=True))
    return f'{chart}<div class="tbl-scroll" style="margin-top:.5rem"><table class="cmp-table"><thead><tr><th>Model</th><th>Provider</th><th>Quality</th><th>Coding</th><th>Cost</th><th>Note</th></tr></thead><tbody>{rows}</tbody></table></div>'

def _frontier_overlay_chart(rows: list, top_n: int = 8) -> str:
    """Puts your top local search results on the SAME 0-100 quality scale as the proprietary reference table, so a score is legible against something externally verifiable rather than only against other unknown open-weight models."""
    local = [{"name": f"{r['name']} [{r['quant']}]", "quality": r.get("leaderboard_avg") or (r["sub_scores"]["intel"] * 100), "is_local": True} for r in rows[:top_n]]
    local = [i for i in local if i["quality"] > 0]
    frontier = [{"name": p[0], "quality": p[2], "is_local": False} for p in PROPRIETARY_REFS]
    items = sorted(local + frontier, key=lambda i: i["quality"], reverse=True)
    if not local: return '<div class="dim tiny" style="padding:.4rem 0">None of your top results have benchmark data yet - frontier reference alone shown below in the Compare tab. Try Deep bench scan to fill this in.</div>'
    return _svg_bar_chart(items, "quality", "name", title="Your top results vs frontier models (same quality scale)", color_key="is_local")

def _panel_compare():
    return f"""<div style="padding:.9rem;height:100%;overflow-y:auto;box-sizing:border-box">
        <div class="fsect-hd">Frontier Model Reference</div>
        <p class="dim tiny">What your local setup's speed/quality trade-off is actually competing against. Edit PROPRIETARY_REFS in ai_calc_config.json to update figures - not fetched live.</p>
        <div id="cmp-frontier">{_frontier_chart_html()}</div>
        <div class="fsect-hd" style="margin-top:1.2rem">Sweep: Context Window vs Speed</div>
        <p class="dim tiny">Compares up to 3 model/quant combinations across a context-token range on the selected hardware - e.g. is a KV-cache size increase worth the speed you give up.</p>
        <form class="pform" hx-post="/im/in" hx-target="#cmp-sweep-out" hx-swap="innerHTML">
            <input type="hidden" name="type" value="ai_calc_sweep"><input type="hidden" name="branch" value="ai_calc"><input type="hidden" name="lvl" value="2">
            <div class="frow wrap">
                <label>Hardware{_hw_select_html()}</label>
                <label>Ctx range min-max (tokens)<input class="fin" name="ctx_range" value="4096-65536"></label>
                <label>Steps<input class="fin" name="ctx_steps" type="number" value="8"></label>
                <label>KV Quant<select class="fin" name="kv_bits"><option value="16">FP16</option><option value="8" selected>INT8</option><option value="4">INT4</option></select></label>
            </div>
            {"".join(f'''<div class="frow wrap"><label>Series {i+1} Params (B){(" - required" if i==0 else " - optional")}<input class="fin" name="s{i}_params" type="number" step="any" value="{"7" if i==0 else ""}"></label><label>Quant<select class="fin" name="s{i}_quant">{_quant_opts()}</select></label></div>''' for i in range(3))}
            <button class="rbtn" type="submit">Run Sweep</button>
        </form>
        <div id="cmp-sweep-out" class="rzone"></div>
    </div>"""

# --- Intent Handlers ---

async def _render_panel(request, state):
    active = state.get("active", "calc")
    if active == "search": return state, _panel_search()
    if active == "session": return state, '<div style="padding:.9rem;height:100%;overflow-y:auto;box-sizing:border-box" hx-post="/im/in" hx-vals=\'{"type":"ai_calc_session_poll","branch":"ai_calc","lvl":2}\' hx-trigger="load, every 5s" hx-target="this" hx-swap="innerHTML">Loading...</div>'
    if active == "hardware": return state, _panel_hardware()
    if active == "compare": return state, _panel_compare()
    return state, _panel_calc()

async def _h_calc_run(request, payload, imr):
    hw = get_hw(payload.get("cnode_id", ""))
    pb = max(0.1, min(float(payload.get("params_b", 7) or 7), 500))
    quant = payload.get("quant", "Q4_K_M") if payload.get("quant") in QUANTS else "Q4_K_M"
    ctx = max(512, int(payload.get("ctx_tokens", 20992) or 20992))
    kv_bits = int(payload.get("kv_bits", 8) or 8)
    return imr.raw(_calc_result_html(pb, quant, ctx, hw, kv_bits, payload.get("cnode_id","")))

async def _h_image_calc(request, payload, imr):
    hw = get_hw(payload.get("cnode_id",""))
    r = image_estimate(payload.get("image_model", next(iter(IMAGE_MODELS), "")), hw, max(1, min(int(payload.get("steps",20) or 20), 500)))
    if not r["feasible"]: return imr.raw(f'<div class="err-box">Cannot run: {r["note"]}</div>')
    return imr.raw(f"""<div style="display:grid;grid-template-columns:repeat(auto-fit,minmax(130px,1fr));gap:.5rem;margin-top:.4rem">
        <div class="stat" style="--sc:{'#00ffa2' if r['its']>=1.5 else '#ffcc00' if r['its']>=0.5 else '#ff8c42'}"><div class="sl">Speed</div><div class="sv">{r['its']}<span class="su"> it/s</span></div></div>
        <div class="stat" style="--sc:#3d9aff"><div class="sl">Gen Time</div><div class="sv">{r['total_s']}<span class="su"> s</span></div></div>
        <div class="stat" style="--sc:#b06aff"><div class="sl">VRAM Min</div><div class="sv">{r['vram_min']}<span class="su"> GB</span></div></div>
    </div><div style="font-size:.72rem;color:var(--text_muted);margin-top:.3rem">{r['note']}</div>""")

async def _h_log_actual(request, payload, imr):
    AIM.resources.log_usage(payload.get("cnode_id","") or "unspecified", "calc_estimate_check", 0,
                             extra={"params_b": float(payload.get("params_b",0) or 0), "quant": payload.get("quant",""), "ctx_tokens": int(payload.get("ctx_tokens",0) or 0), "actual_tps": float(payload.get("actual_tps",0) or 0)})
    return imr

async def _h_search(request, payload, imr):
    with _job_lock:
        if _job["running"]: return imr.raw('<div class="sbar">Job already running.</div>')
    hw = get_hw(payload.get("cnode_id", ""))
    params = {"task_preset": payload.get("task_preset", next(iter(TASK_PRESETS), "")), "min_params": max(0.1, float(payload.get("min_params",1) or 1)),
              "max_params": min(float(payload.get("max_params",35) or 35), total_mem(hw) / QUANTS["Q2_K"]["bpp"]), "ctx_tokens": max(512, int(payload.get("ctx_tokens",20992) or 20992)),
              "min_tps": max(0.0, float(payload.get("min_tps",1.0) or 0)), "target_tps": max(0.1, float(payload.get("target_tps",5.0) or 0.1)),
              "extra_query": payload.get("extra_query",""), "must_contain": payload.get("must_contain",""), "any_contain": payload.get("any_contain",""), "exclude": payload.get("exclude",""),
              "hw": hw, "weights": _parse_weights(payload), "force_refresh": payload.get("force_refresh") == "1", "deep_scan": payload.get("deep_scan") == "1",
              "kv_bits": int(payload.get("kv_bits", 8) or 8), "min_quant": payload.get("min_quant","Q2_K") if payload.get("min_quant","Q2_K") in QUANTS else "Q2_K",
              "top_n": max(5, min(int(payload.get("top_n",40) or 40), 200)), "fetched": 0}
    _job_reset(); _job_up(running=True, status="Starting...", params=params)
    threading.Thread(target=_run_thread, args=(params,), daemon=True).start()
    return imr.raw(_search_status_html())

async def _h_search_status(request, payload, imr): return imr.raw(_search_status_html())

async def _h_rerank(request, payload, imr):
    """Re-scores the already-fetched candidate set against new weight sliders without re-querying HuggingFace - lets you compare weighting strategies live off one fetch, which is the same underlying need as the sweep tool: exploring a configuration space rather than accepting one fixed point."""
    with _job_lock: filtered, params = _job.get("filtered"), _job.get("params")
    if not filtered: return imr.raw('<div class="err-box">No cached search results to re-rank - run a search first.</div>')
    weights = _parse_weights(payload)
    rows = _rank_filtered(filtered, params["hw"], params["ctx_tokens"], params["target_tps"], weights, params["top_n"], params.get("kv_bits", 8))
    _write_csv(rows, "rerank")
    with _job_lock:
        prior_stats = (_job.get("result") or {}).get("stats", {})
        result = {"rows": rows, "stats": {**prior_stats, "csv": "data/ai_tools/ai_calc/analysis_rerank.csv"}, "weights": weights}
        _job["result"] = result
    return imr.raw(_search_html(result))

async def _h_hw_form(request, payload, imr):
    c = AIM.resources.get_cnode(payload.get("cid",""))
    return imr.oob(_hw_form_html(c), "hw-form-slot") if c else imr

async def _h_hw_save(request, payload, imr):
    cid = payload.get("cid",""); c = AIM.resources.get_cnode(cid) or {}
    for k in _HW_FIELDS:
        if k in payload:
            try: c[k] = float(payload[k])
            except Exception: pass
    AIM.resources.save_cnode(cid, c)
    return imr.oob(_panel_hardware(), TM.content_id)

async def _h_session_poll(request, payload, imr):
    conns = AIM.connections.list_conns(conn_type="ollama")
    if not conns: return imr.raw('<div class="placeholder">No Ollama connection configured - add one in AI Manager.</div>')
    hw = get_hw("")
    cards = ""
    for conn in conns:
        loaded = await _fetch_running(conn)
        cards += f'<div class="dim tiny" style="margin:.4rem 0 .2rem">{UI.escape(conn.get("display_name", conn["_id"]))}</div>'
        cards += "".join(_session_card_html(m, hw) for m in loaded) or '<div class="dim tiny" style="padding:.2rem 0">No models loaded.</div>'
    return imr.raw(cards)

async def _h_sweep(request, payload, imr):
    hw = get_hw(payload.get("cnode_id",""))
    kv_bits = int(payload.get("kv_bits", 8) or 8)
    try: lo, hi = (float(x) for x in payload.get("ctx_range","4096-65536").split("-"))
    except Exception: lo, hi = 4096.0, 65536.0
    steps = max(2, min(int(payload.get("ctx_steps",8) or 8), 30))
    ctx_vals = [lo + (hi-lo)*i/(steps-1) for i in range(steps)]
    series = []
    for i in range(3):
        raw = (payload.get(f"s{i}_params") or "").strip()
        if not raw: continue
        try: pb = float(raw)
        except Exception: continue
        quant = payload.get(f"s{i}_quant","Q4_K_M")
        if quant not in QUANTS: continue
        pts = [(c, full_perf(pb, quant, int(c), hw, kv_bits)["tps"]) for c in ctx_vals]
        series.append({"label": f"{pb}B {quant}", "points": pts})
    if not series: return imr.raw('<div class="err-box">Enter at least one series (Params + Quant) to sweep.</div>')
    return imr.raw(_svg_line_chart(series, title="Tokens/sec vs context window"))

def _hw_from_form(f) -> dict:
    hw = dict(_HW_DEFAULTS)
    for k, lo, hi in [("vram_gb", 0.5, 512), ("shared_gb", 0, 512), ("sys_ram_gb", 0, 1024), ("os_overhead_gb", 0, 64), ("mem_bw_gbps", 1, 10000), ("sys_ram_bw_gbps", 1, 10000), ("gpu_tflops_fp16", 0.1, 1000)]:
        hw[k] = max(lo, min(float(f.get(k, hw[k])), hi))
    hw["gpu_eff"] = GPU_EFF
    return hw

def estimate_for_pipeline(cnode_id: str, params_b: float, quant: str, ctx: int, kv_bits: int = 8) -> dict: return full_perf(params_b, quant, ctx, get_hw(cnode_id), kv_bits) #Stable entry point for other modules (engine.py's future sDAG optimizer, PipelineBuilderUI's preflight) to get a speed/memory estimate without importing ai_calc's UI internals.
    

# --- Routes / Init ---

def _script():
    preset_weights = json.dumps({k: {d: v.get("weights", {}).get(d, DEFAULT_WEIGHTS.get(d, 1)) for d in SCORE_DIMS} for k, v in TASK_PRESETS.items()})
    return f"""(function(){{
        var PRESET_WEIGHTS = {preset_weights};
        window.applyPreset = function(preset){{
            var ws = PRESET_WEIGHTS[preset]; if(!ws) return;
            Object.keys(ws).forEach(function(d){{
                var sl = document.querySelector('.wsblock input[name="w_'+d+'"]');
                if(sl){{ sl.value = ws[d]; var lbl = sl.parentElement.querySelector('.wsv'); if(lbl) lbl.textContent = ws[d]; }}
            }});
        }};
        window.syncHW = function(){{}};  // hooked so _hw_ref_html's inline onclick has a stable target even before any hw form field listens; individual .hwin values are read generically on form submit, no per-field JS needed
    }})();
    function calcLogActual(cnodeId, paramsB, quant, ctx) {{
        var v = document.getElementById('calc-actual-tps').value; if (!v) return;
        htmx.ajax('POST', '/im/in', {{values:{{type:'ai_calc_log_actual', branch:'ai_calc', lvl:2, cnode_id:cnodeId, params_b:paramsB, quant:quant, ctx_tokens:ctx, actual_tps:v}}, swap:'none'}})
            .then(function(){{ var m=document.getElementById('calc-log-msg'); if(m) m.textContent = 'Logged.'; }});
    }}"""

def init_tool(env, prefix):
    global ENV, UI, BI, AIM, IM, TM, _P
    ENV, _P = env, prefix.rstrip("/")
    UI = env["templates"].env.globals.get("UI")
    BI = env["tools"]["built_ins"]
    AIM = env["tools"]["ai_manager"]
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    load_config()
    IM = env["InterfaceManager"](nesting_level=2, db_path="ai_tools/ai_calc_im.db")
    TM = BI.TabManager(namespace="ai_calc", tab_bar_id="calc-tab-bar", content_id="calc-panel", render_content_fn=_render_panel, intent_prefix="ai_calc", IM=IM, scope="user", nesting_level=2, allow_new=False, closable=False,
                        empty={"tabs": {"calc":{"id":"calc","order":0,"label":"Calculator","icon":"&#x223C;"}, "search":{"id":"search","order":1,"label":"Model Finder","icon":"&#x1F50D;"},
                                        "compare":{"id":"compare","order":2,"label":"Compare","icon":"&#x1F4CA;"}, "session":{"id":"session","order":3,"label":"Session","icon":"&#x25CF;"}, "hardware":{"id":"hardware","order":4,"label":"Hardware","icon":"&#x1F5A5;"}}, "active":"calc"})
    IM.scripts.update({"ai_calc_run": [_h_calc_run], "ai_calc_image": [_h_image_calc], "ai_calc_log_actual": [_h_log_actual], "ai_calc_search": [_h_search],
                        "ai_calc_search_status": [_h_search_status], "ai_calc_rerank": [_h_rerank], "ai_calc_sweep": [_h_sweep],
                        "ai_calc_hw_form": [_h_hw_form], "ai_calc_hw_save": [_h_hw_save], "ai_calc_session_poll": [_h_session_poll]})
    print("[ai_calc] ready")

@router.get("")
@router.get("/")
async def root(request: Request):
    state = await TM._load(request)
    state, panel_html = await _render_panel(request, state)
    tab_bar = await TM.tab_bar_fn(state, "calc-tab-bar", "ai_calc", 2, allow_new=False, closable=False)
    return ENV["templates"].TemplateResponse(name="base.html", request=request, context={
        "request": request, "user": request.state.user, "nesting_level": 2, "shell_id": IM.branch_id,
        "toolbars": {"top": UI.toolbar(side="top", content=tab_bar, size="2.5rem", id="calc-top", nesting_level=2, start_open=True, locked=True)},
        "content": f'<div id="calc-panel" style="height:100%;overflow:hidden">{panel_html}</div>', "extra_css": CSS, "extra_script": _script()})

def right_panel(): return '<div class="ait-rp"><div class="ait-rp-hd">AI Calc</div><div style="font-size:.72rem;color:var(--text_muted);padding:.3rem">Speed/memory calculator, model finder, hardware profiles, frontier compare, sweep.</div></div>'

# --- CSS ---

CSS = """
.pform{background:var(--surface);border:1px solid var(--border);border-radius:6px;padding:.8rem;margin-bottom:.5rem}
.frow{display:flex;gap:.5rem;align-items:flex-start;margin-bottom:.55rem}.frow.wrap{flex-wrap:wrap}
.frow label{display:flex;flex-direction:column;gap:.15rem;font-size:.73rem;color:var(--text_muted);flex:1;min-width:100px}
.frow label:has(input[type=checkbox]){flex-direction:row;flex:0;align-items:center}
.fin{background:var(--bg);border:1px solid var(--border);color:var(--text);border-radius:5px;padding:.32rem .48rem;font-family:var(--font-mono);font-size:.8rem;width:100%;box-sizing:border-box}
.rbtn{background:var(--accent_dim);color:var(--accent);border:1px solid var(--accent);border-radius:6px;padding:.4rem .85rem;font-size:.82rem;font-weight:700;cursor:pointer;white-space:nowrap}
.rbtn:hover{background:var(--accent);color:#000}
.rzone{margin-top:.7rem}.placeholder{color:var(--text_muted);font-size:.84rem;padding:1.2rem 0;text-align:center}
.fsect-hd{font-size:.73rem;font-weight:700;color:var(--text_muted);text-transform:uppercase;letter-spacing:.05em;margin-bottom:.5rem}
.dim{color:var(--text_muted)}.tiny{font-size:.7rem}
.qn{font-family:var(--font-mono);font-weight:700;color:var(--accent)}
.tbl-scroll{overflow-x:auto}
.wsblock{display:flex;flex-direction:column;gap:.28rem;padding:.3rem 0}
.wsl{display:flex;align-items:center;gap:.55rem;font-size:.75rem;color:var(--text_muted)}
.wsln{min-width:130px;color:var(--wc);font-weight:600;flex-shrink:0}.wslider{flex:1;accent-color:var(--wc)}.wsv{min-width:1.5rem;text-align:right;font-family:var(--font-mono);color:var(--wc);font-weight:700}
.wleg{display:flex;flex-wrap:wrap;gap:.3rem .65rem;margin:.45rem 0 .5rem;font-size:.7rem;color:var(--text_muted)}.wli{display:flex;align-items:center;gap:.22rem}.wld{width:7px;height:7px;border-radius:50%}
.sb-block{display:flex;flex-direction:column;gap:.15rem;margin:.28rem 0}.sbrow{display:flex;align-items:center;gap:.4rem;font-size:.7rem}
.sbl{min-width:110px;color:var(--text_muted);font-size:.67rem;flex-shrink:0}.sbbar{flex:1;height:6px;background:var(--bg);border-radius:3px;overflow:hidden;border:1px solid var(--border)}
.sbfill{height:100%;border-radius:3px}.sbv{min-width:3.2rem;text-align:right;font-family:var(--font-mono)}.sbw{min-width:2.2rem;color:var(--text_muted);font-size:.65rem}.sbc{min-width:3.8rem;text-align:right;font-family:var(--font-mono);font-size:.67rem}
.clist{display:flex;flex-direction:column;gap:.5rem}
.mcard{background:var(--bg);border:1px solid var(--border);border-left:3px solid var(--rc);border-radius:6px;padding:.6rem .75rem;display:flex;gap:.5rem}
.mrank{font-size:.7rem;font-weight:800;color:var(--rc);min-width:1.8rem;text-align:right;flex-shrink:0}.mbody{flex:1;min-width:0}
.mhead{display:flex;align-items:baseline;flex-wrap:wrap;gap:.28rem;margin-bottom:.28rem}
.mname{font-weight:700;font-size:.88rem;color:var(--accent);text-decoration:none}.mauthor{font-size:.68rem;color:var(--text_muted)}
.tscore{font-size:.8rem;font-weight:800;font-family:var(--font-mono);margin-left:auto}
.mstats{display:flex;flex-wrap:wrap;gap:.25rem;font-size:.73rem;margin-bottom:.22rem}.ms{background:var(--surface);border:1px solid var(--border);padding:.1rem .35rem;border-radius:4px}
.mqts{display:flex;flex-wrap:wrap;gap:.2rem}.qt{font-size:.65rem;padding:.08rem .32rem;border-radius:3px;border:1px solid;font-family:var(--font-mono)}
.mpop{display:flex;align-items:center;gap:.6rem;font-size:.7rem;color:var(--text_muted);flex-wrap:wrap}.hfl{color:var(--accent);text-decoration:none;margin-left:auto}
.bdg{font-size:.63rem;padding:.08rem .32rem;border-radius:3px;font-weight:700}.bl{color:#00ffa2}
.mdet summary{cursor:pointer;font-size:.7rem;color:var(--text_muted);list-style:none}.mdet-body{padding:.45rem 0 .15rem;border-top:1px solid var(--border);margin-top:.28rem}
.lb-row{display:flex;flex-wrap:wrap;gap:.28rem}.lbp{font-size:.68rem;padding:.08rem .38rem;border-radius:3px;background:var(--surface);border:1px solid var(--border);font-family:var(--font-mono)}
.sbar{display:flex;flex-wrap:wrap;gap:.3rem .7rem;font-size:.72rem;color:var(--text_muted);margin-bottom:.5rem;padding:.4rem .6rem;background:var(--bg);border:1px solid var(--border);border-radius:5px}
.status-bar{display:flex;align-items:center;gap:.6rem;padding:.45rem .65rem;background:var(--surface);border:1px solid var(--accent);border-radius:5px;font-size:.8rem;margin-bottom:.4rem}
.spin-anim{color:var(--accent);font-family:var(--font-mono)}
.log-wrap{font-family:var(--font-mono);font-size:.7rem;color:var(--text_muted);max-height:170px;overflow-y:auto;background:var(--bg);border:1px solid var(--border);border-radius:4px;padding:.35rem .55rem}
.log-line{padding:.06rem 0;border-bottom:1px solid var(--border)}
.hwg{display:flex;flex-wrap:wrap;gap:.5rem}.hwg label{display:flex;flex-direction:column;gap:.15rem;font-size:.73rem;color:var(--text_muted);min-width:120px;flex:1}
.hwin{background:var(--bg);border:1px solid var(--border);color:var(--text);border-radius:4px;padding:.27rem .4rem;font-family:var(--font-mono);font-size:.8rem;width:100%;box-sizing:border-box}
"""