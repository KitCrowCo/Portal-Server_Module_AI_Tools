"""
AI Calc - Speed estimator, context planner, model finder, session monitor.
Sub-module of ai_tools. Mounted at /module/ai_tools/ai_calc.
All results are detailed by design. Speed is not a priority.
"""
import json, math, re, csv, threading, traceback, uuid
import psutil
import httpx, asyncio
import pandas as pd
import io
from pathlib import Path
from datetime import datetime
from fastapi import APIRouter, Request
from fastapi.responses import HTMLResponse, JSONResponse
from huggingface_hub import model_info

TOOL_META = {"label": "AI Calc", "group": "model_tools", "icon": "&#x25B3;", "description": "Speed, context, benchmarks, model finder", "singleton": True}

router = APIRouter()
ENV = {}
_P = "/module/ai_tools/ai_calc"
DATA_DIR = Path("./data/ai_tools/ai_calc")

QUANTS = {}
ARCH_TABLE = {}
IMAGE_MODELS = {}
REFERENCE_LINES = {}
TASK_PRESETS = {}
KNOWN_BENCHMARKS = {}
PROPRIETARY_REFS = {}
DEFAULT_WEIGHTS = {}
SCORE_DIMS = {} # {"params", "speed", "mem", "ctx", "intel"}
SCORE_DIM_LABELS = {} #{"params": "Parameters", "speed": "Speed", "mem": "Memory", "ctx": "Context", "intel": "Intelligence"}
SCORE_DIM_COLORS = {} #{"params": "#ff9a3c", "speed": "#00ffa2", "mem": "#3d9aff", "ctx": "#ffcc00", "intel": "#b06aff"}
_KNOWN_SHORT = {}

def init_tool(env: dict, prefix: str):
    global ENV, _P
    ENV = env
    _P = prefix
    load_config()

def load_config():
    """Load all configuration data from a single JSON file."""
    global QUANTS, ARCH_TABLE, IMAGE_MODELS, REFERENCE_LINES, TASK_PRESETS, KNOWN_BENCHMARKS, PROPRIETARY_REFS, DEFAULT_WEIGHTS, SCORE_DIMS, SCORE_DIM_LABELS, SCORE_DIM_COLORS, _KNOWN_SHORT
    config_path = Path(__file__).parent / "ai_calc_config.json"
    with open(config_path, 'r') as f:
        config = json.load(f)
    QUANTS.update(config.get("QUANTS", {}))
    ARCH_TABLE = config.get("ARCH_TABLE", ARCH_TABLE) # (params_b_lo, params_b_hi, layers, hidden, attn_heads, kv_heads) kv_heads drives GQA-accurate KV cache math. Most modern models use 8 KV heads.
    IMAGE_MODELS.update(config.get("IMAGE_MODELS", {}))
    REFERENCE_LINES = config.get("REFERENCE_LINES", REFERENCE_LINES)
    TASK_PRESETS.update(config.get("TASK_PRESETS", {}))
    KNOWN_BENCHMARKS.update(config.get("KNOWN_BENCHMARKS", {}))
    PROPRIETARY_REFS = config.get("PROPRIETARY_REFS", PROPRIETARY_REFS)
    DEFAULT_WEIGHTS.update(config.get("DEFAULT_WEIGHTS", {}))
    SCORE_DIMS = set(config.get("SCORE_DIMS", list(SCORE_DIMS)))
    SCORE_DIM_LABELS.update(config.get("SCORE_DIM_LABELS", SCORE_DIM_LABELS))
    SCORE_DIM_COLORS.update(config.get("SCORE_DIM_COLORS", SCORE_DIM_COLORS))
    _KNOWN_SHORT = {k.split("/")[-1].lower(): k for k in KNOWN_BENCHMARKS}

def create_config(path: Path):
    """Create a default configuration file with all the current data."""
    config = {"TOOL_META": TOOL_META,
              "QUANTS": QUANTS,
              "ARCH_TABLE": ARCH_TABLE,
              "IMAGE_MODELS": IMAGE_MODELS,
              "REFERENCE_LINES": REFERENCE_LINES,
              "TASK_PRESETS": TASK_PRESETS,
              "KNOWN_BENCHMARKS": KNOWN_BENCHMARKS,
              "PROPRIETARY_REFS": PROPRIETARY_REFS, # Proprietary model comparison (intelligence index ~ MMLU+reasoning composite, 95=frontier)
              "DEFAULT_WEIGHTS": DEFAULT_WEIGHTS,
              "SCORE_DIMS": list(SCORE_DIMS),
              "SCORE_DIM_LABELS": SCORE_DIM_LABELS,
              "SCORE_DIM_COLORS": SCORE_DIM_COLORS}
    with open(path, 'w') as f:
        json.dump(config, f, indent=2)

DEFAULT_HW = {"vram_gb": 16.0, "shared_gb": 8.0, "sys_ram_gb": 16.0, "os_overhead_gb": 3.5, "mem_bw_gbps": 89.0, "gpu_tflops_fp16": 8.9, "gpu_eff": 0.72}
STANDARD_SIZES = [1, 3, 7, 8, 9, 11, 12, 13, 14, 20, 27, 30, 32, 34, 70, 72]

# Benchmark name normalizer
_BENCH_ALIASES = {"mmlu":"mmlu","massive multitask":"mmlu","mmlu_pro":"mmlu", "arc":"arc","arc_challenge":"arc","ai2_arc":"arc","arc challenge":"arc","arc-challenge":"arc",
                  "hellaswag":"hellaswag","hella swag":"hellaswag", "truthfulqa":"truthfulqa","truthful_qa":"truthfulqa","truthfulqa mc":"truthfulqa","truthfulqa_mc2":"truthfulqa",
                  "winogrande":"winogrande","gsm8k":"gsm8k","gsm 8k":"gsm8k", "ifeval":"ifeval","humaneval":"humaneval","human_eval":"humaneval","pass@1":"humaneval",
                  "bbh":"bbh","big bench hard":"bbh","gpqa":"gpqa","gpqa_diamond":"gpqa", "hle": "intel", "mmlu_pro": "intel"}

def _norm_bench(raw: str) -> str | None:
    r = raw.lower().strip().replace("-","_").replace(" ","_")
    if r in _BENCH_ALIASES: return _BENCH_ALIASES[r]
    for k, v in _BENCH_ALIASES.items():
        if k.replace(" ","_") in r: return v
    return None

def _bench_to_avg(scores: dict) -> float:
    W = {"mmlu":3, "arc":2, "hellaswag":1.5, "truthfulqa":1.5, "winogrande":1, "gsm8k":1.5, "ifeval":2, "bbh":2, "gpqa":2.5}
    tw = tv = 0.0
    for b, w in W.items():
        v = scores.get(b)
        if v and v > 0: tv += float(v) * w; tw += w
    return round(tv / tw, 2) if tw > 0 else 0.0

_INTEL_WEIGHTS = {"hle": 4.0,           # Humanity's Last Exam (Top tier reasoning)
                  "gpqa": 3.0,          # PhD level science
                  "mmlu_pro": 2.5,      # Harder version of MMLU
                  "ifeval": 2.5,        # Instruction following
                  "bbh": 2.0,           # Big Bench Hard
                  "gsm8k": 1.5,         # Math
                  "mmlu": 1.0,          # General knowledge (de-prioritized due to saturation)
                  "arc": 1.0,           # Science
                  "hellaswag": 0.5}      # Commonsense (mostly solved in 2026)

def _compute_intelligence(scores: dict) -> float:
    total = weight = 0.0
    for bench, w in _INTEL_WEIGHTS.items():
        if bench in scores:
            # Most benchmarks are 0-100; ensure they are floats
            total += float(scores[bench]) * w
            weight += w
    return round(total / max(weight, 0.001), 2)

# def _compute_intelligence(scores: dict) -> float:
#     relevant = {"mmlu":3.0, "arc":2.0, "hellaswag":1.5, "truthfulqa":1.5, "gsm8k":1.5, "ifeval":2.0, "bbh":2.0, "gpqa":2.5}
#     total = weight = 0.0
#     for bench, w in relevant.items():
#         if bench in scores:
#             total += scores[bench] * w
#             weight += w
#     return round(total / max(weight, 0.001), 2)

def _url(*parts): return "/".join(p.strip("/") for p in [_P, *parts] if p)

# --- Core physics ---

def arch_for(params_b: float):
    for lo, hi, layers, hidden, attn, kv in ARCH_TABLE:
        if lo <= params_b < hi: return layers, hidden, attn, kv
    return 32, 4096, 32, 8

def total_vram(hw): return hw["vram_gb"] + hw["shared_gb"]
def usable_ram(hw): return max(hw["sys_ram_gb"] - hw["shared_gb"] - hw.get("os_overhead_gb",0), 0.0)
def total_mem(hw): return total_vram(hw) + usable_ram(hw)
def model_gb(params_b, quant): return params_b * 1e9 * QUANTS.get(quant, QUANTS["Q4_K_M"])["bpp"] / 1e9

def kv_bytes_per_token(params_b: float, kv_bits: int = 8) -> int:
    """KV cache bytes per token with GQA-accurate kv_heads."""
    layers, hidden, attn, kv_heads = arch_for(params_b)
    head_dim = hidden // attn
    return 2 * layers * kv_heads * head_dim * (kv_bits // 8)

def kv_cache_gb_for_ctx(params_b: float, ctx: int, kv_bits: int = 8) -> float: return kv_bytes_per_token(params_b, kv_bits) * ctx / (1024**3)
def max_ctx_tokens(params_b: float, avail_gb: float, kv_bits: int = 8) -> int: return int(avail_gb * 1024**3 / max(kv_bytes_per_token(params_b, kv_bits), 1))

def estimate_tps(params_b: float, quant: str, hw: dict) -> dict:
    mgb  = model_gb(params_b, quant)
    vt, tm, bw, eff = total_vram(hw), total_mem(hw), hw["mem_bw_gbps"], hw["gpu_eff"]
    if mgb > tm: return {"tps":0.0,"prefill_tps":0.0,"mode":"OOM","model_gb":round(mgb,2), "fits_vram":False,"fits_total":False,"eff_bw":0.0,"vram_used":0.0,"ram_used":round(mgb,2)}
    vram_used = min(mgb, vt); ram_used = max(mgb - vt, 0.0)
    eff_bw    = bw * eff * (1.0 - (ram_used / mgb) * 0.15) if ram_used > 0 else bw * eff
    return {"tps":round((eff_bw*1e9)/(mgb*1e9*1.10),2), "prefill_tps":round(hw.get("gpu_tflops_fp16",8.9)*1e12*eff/(2*params_b*1e9),1), "mode":f"VRAM+RAM ({ram_used:.1f}GB spill)" if ram_used > 0 else "VRAM", "model_gb":round(mgb,2), "fits_vram":mgb<=vt, "fits_total":True, "eff_bw":round(eff_bw,1), "vram_used":round(vram_used,2), "ram_used":round(ram_used,2)}

def full_perf(params_b: float, quant: str, ctx: int, hw: dict, kv_bits: int = 8) -> dict:
    p    = estimate_tps(params_b, quant, hw)
    kv   = kv_cache_gb_for_ctx(params_b, ctx, kv_bits)
    vt, tm = total_vram(hw), total_mem(hw)
    kv_in_vram = min(kv, max(vt - p["model_gb"], 0.0))
    kv_in_ram  = max(kv - kv_in_vram, 0.0)
    total_used = p["model_gb"] + kv
    fits = p["fits_total"] and total_used <= tm
    tps, mode = p["tps"], p["mode"]
    if fits and kv_in_ram > 0 and tps > 0:
        tps  = round(tps * (1.0 - (kv_in_ram / total_used) * 0.20), 2)
        mode = mode + f" +KV({kv_in_ram:.1f}->RAM)"
    t10k = (10000 / tps) if tps > 0 and fits else None
    qi   = QUANTS.get(quant, QUANTS["Q4_K_M"])
    return {**p, "tps":tps, "prefill_tps":p["prefill_tps"], "mode":mode, "kv_gb":round(kv,3),
            "kv_in_vram":round(kv_in_vram,3), "kv_in_ram":round(kv_in_ram,3), "total_mem_gb":round(total_used,2), "mem_ok":fits,
            "time_10k_s":round(t10k,0) if t10k else None,
            "quality_idx":qi["quality"], "creative_ok":qi["creative_ok"],
            "quant":quant, "params_b":params_b}

# def sweep_data(quants: list, hw: dict) -> dict: return {"series":{q:[{"params_b":s,**{k:estimate_tps(s,q,hw)[k] for k in ("tps","model_gb","mode")}]} for s in STANDARD_SIZES for q in quants}, "refs":REFERENCE_LINES, "sizes":STANDARD_SIZES}
def sweep_data(quants: list, hw: dict) -> dict: return {"series":{q:[{"params_b":s,**{k:estimate_tps(s,q,hw)[k] for k in ("tps","model_gb","mode")}} for s in STANDARD_SIZES] for q in quants}, "refs":REFERENCE_LINES, "sizes":STANDARD_SIZES}

def image_estimate(model_key: str, hw: dict, steps: int = 20) -> dict:
    m   = IMAGE_MODELS.get(model_key, IMAGE_MODELS["SDXL"])
    vt  = total_vram(hw)
    if m["vram_min_gb"] > total_mem(hw): return {"feasible":False,"note":f"Needs {m['vram_min_gb']:.0f} GB, have {total_mem(hw):.0f} GB"}
    its = m["its_per_90gbps"] * (hw["mem_bw_gbps"] / 90.0) * hw.get("gpu_eff",0.72)
    return {"feasible":True,"its":round(its,2),"total_s":round(steps/max(its,0.01),1), "vram_min":m["vram_min_gb"],"fits_vram":m["vram_min_gb"]<=vt, "mode":"VRAM" if m["vram_min_gb"]<=vt else "RAM offload (slower)","note":m["note"],"steps":steps}

# --- Scoring ---

def _sub_scores(perf: dict, model: dict, hw: dict, target_tps: float) -> dict:
    raw_avg   = (model.get("leaderboard_avg") or 0.0) / 100.0
    lb_avg    = min(raw_avg * model.get("bench_confidence",0.0), 1.0) if model.get("bench_confidence",0) > 0 else 0.0
    tps, tm   = perf.get("tps",0.0), total_mem(hw)
    used, vt  = perf.get("total_mem_gb",tm), total_vram(hw)
    mgb, kv   = perf.get("model_gb",0.0), perf.get("kv_gb",0.0)
    needed    = mgb + kv
    pop_raw   = math.log1p(model.get("likes",0) or 0) * 0.4 + math.log1p(model.get("downloads",0) or 0) * 0.6
    intel     = _compute_intelligence(model.get("bench_detail", {}))
    return {"lb_avg":     round(lb_avg, 4),
            "quant_qual": round(perf.get("quality_idx",0.0), 4),
            "speed":      round(min(tps / max(target_tps * 2.0, 1.0), 1.0) if tps > 0 else 0.0, 4),
            "mem_head":   round(max(0.0, (tm - used) / max(tm, 1.0)), 4),
            "ctx_fit":    round(min(needed, vt) / max(needed, 0.001) if needed > 0 else 1.0, 4),
            "popularity": round(min(pop_raw / 16.0, 1.0), 4),
            "intel":      round(intel, 4)}

def _total_score(sub: dict) -> float:
    tw = sum(sub.get(d,0) for d in SCORE_DIMS)
    return round(sum(sub.get(d, 0) * (sub.get(f"{d}_weight", DEFAULT_WEIGHTS[d]) / 10.0) for d in SCORE_DIMS) / max(tw, 1), 4)

def _parse_weights(f) -> dict: return {d: max(0.0, min(float(f.get(f"w_{d}", DEFAULT_WEIGHTS[d])), 10.0)) for d in SCORE_DIMS}

def _rank_filtered(filtered: list, hw: dict, ctx: int, target_tps: float, weights: dict, top_n: int = 40) -> list:
    rows = []
    for m in filtered:
        for q in m.get("avail_quants",[]):
            if q not in QUANTS: continue
            perf = full_perf(m["params_b"], q, ctx, hw)
            if not perf["mem_ok"]: continue
            sub = _sub_scores(perf, m, hw, target_tps)
            total = _total_score(sub)
            rows.append({**{k:m[k] for k in ("id", "params_b", "likes", "downloads", "tags", "avail_quants", "leaderboard_avg","lb_detail","bench_confidence", "bench_source","bench_inferred")}, 
                         "author":m["id"].split("/")[-1] if "/" in m["id"] else "", "name":m["id"].split("/")[-1], "quant":q,"perf":perf,"sub_scores":sub,"total_score":total, "below_floor":perf["tps"] < target_tps})
    rows.sort(key=lambda r: r["total_score"], reverse=True)
    return rows[:top_n]

def _write_csv(rows: list, key: str):
    if not rows: return
    fields = ["rank","id","quant","total_score","lb_avg","quant_qual","speed","mem_head","ctx_fit","intel","popularity","leaderboard_avg","params_b","tps","prefill_tps","model_gb","kv_gb","total_mem_gb","time_10k_s","mode","likes","downloads","creative_ok"]
    try:
        with open(DATA_DIR / f"analysis_{key}.csv", "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
            w.writeheader()
            for i, r in enumerate(rows):
                p = r["perf"]; ss = r["sub_scores"]
                w.writerow({"rank":i+1,"id":r["id"],"quant":r["quant"],"total_score":r["total_score"], **{d:ss[d] for d in SCORE_DIMS}, "leaderboard_avg":r["leaderboard_avg"],"params_b":r["params_b"],
                            "tps":p["tps"],"prefill_tps":p["prefill_tps"],"model_gb":p["model_gb"], "kv_gb":p["kv_gb"],"total_mem_gb":p["total_mem_gb"],
                            "time_10k_s":p["time_10k_s"],"mode":p["mode"], "likes":r["likes"],"downloads":r["downloads"],"creative_ok":p["creative_ok"]})
    except Exception as e: print(f"[ai_calc] CSV write error: {e}")

# Benchmark intelligence (multi-source)

def _scores_from_static(model_id: str) -> dict:
    if model_id in KNOWN_BENCHMARKS: return dict(KNOWN_BENCHMARKS[model_id])
    short = model_id.split("/")[-1].lower()
    if short in _KNOWN_SHORT: return dict(KNOWN_BENCHMARKS[_KNOWN_SHORT[short]])
    for ks, kfull in _KNOWN_SHORT.items():
        if ks in short or short in ks: return dict(KNOWN_BENCHMARKS[kfull])
    return {}

def _scores_from_card(model: dict) -> dict:
    card = model.get("cardData") or {}
    if isinstance(card, str):
        try: card = json.loads(card)
        except: return {}
    scores = {}
    for entry in (card.get("model-index") or []):
        for result in (entry.get("results") or []):
            for metric in (result.get("metrics") or []):
                mname = str(metric.get("name","") or metric.get("type","")).lower()
                mval  = metric.get("value")
                if mval is None: continue
                try: mval = float(str(mval).replace("%","").strip())
                except: continue
                if 0 < mval <= 1.0: mval *= 100.0
                canonical = _norm_bench(mname)
                if canonical and mval > 0 and scores.get(canonical,0) < mval: scores[canonical] = mval
    return scores

_README_RE = re.compile(r'(?:^|\|)\s*(?P<bench>mmlu|arc[^|]*|hellaswag|truthfulqa|winogrande|gsm8k|ifeval|bbh|gpqa|humaneval|pass@1)[^\|]*\|[^\|]*?\|\s*(?P<val>\d{1,3}(?:\.\d{1,4})?)\s*(?:%|\|)', re.I | re.M)
_README_RE2 = re.compile(r'(?P<bench>mmlu|arc.challenge|hellaswag|truthfulqa|winogrande|gsm8k|ifeval|bbh|gpqa|humaneval)\s*[:\|]\s*(?P<val>\d{1,3}(?:\.\d{1,4})?)(?:\s*%)?', re.I)

def _scores_from_readme(text: str) -> dict:
    scores = {}
    for pat in (_README_RE, _README_RE2):
        for m in pat.finditer(text):
            bench = _norm_bench(m.group("bench"))
            if not bench: continue
            try: val = float(m.group("val"))
            except: continue
            if 0 < val <= 1.0: val *= 100.0
            if 1.0 < val <= 100.0 and scores.get(bench,0) < val: scores[bench] = val
    return scores

async def _fetch_readme(client: httpx.AsyncClient, model_id: str) -> str:
    try:
        r = await client.get(f"https://huggingface.co/{model_id}/raw/main/README.md",
                             timeout=httpx.Timeout(connect=4.0, read=8.0))
        return r.text[:80000] if r.status_code == 200 else ""
    except: return ""

def _base_chain(model: dict) -> list:
    card = model.get("cardData") or {}
    if isinstance(card, str):
        try: card = json.loads(card)
        except: card = {}
    bm = card.get("base_model") or model.get("base_model") or []
    if isinstance(bm, str): bm = [bm]
    bases = [b for b in bm if b]
    name_l = model.get("id","").split("/")[-1].lower()
    for ks, kfull in _KNOWN_SHORT.items():
        if ks in name_l and kfull not in bases: bases.append(kfull)
    return bases[:4]

async def fetch_leaderboard_scores() -> dict:
    """
    Fetches benchmark scores anonymously from the Open LLM Leaderboard dataset.
    Returns a dict mapping: { "base_model_id": { "mmlu": 75.2, "hle": 24.1, ... } }
    """
    scores_map = {}
    # Dataset Server API for the Open LLM Leaderboard (v2)
    url = "https://datasets-server.huggingface.co/rows?dataset=open-llm-leaderboard%2Fresults&config=default&split=train&limit=150"
    
    try:
        async with httpx.AsyncClient(timeout=20.0) as client:
            resp = await client.get(url)
            if resp.status_code == 200:
                data = resp.json()
                for row in data.get("rows", []):
                    content = row.get("row", {})
                    # The dataset structure uses 'model_name' or 'config' to identify the LLM
                    m_id = content.get("model_name") or content.get("config")
                    if m_id:
                        # Extract the actual scores (keys vary slightly by leaderboard version)
                        scores_map[m_id] = {
                            "mmlu": content.get("mmlu", 0),
                            "mmlu_pro": content.get("mmlu_pro", 0),
                            "arc": content.get("arc", 0),
                            "hle": content.get("hle", 0),
                            "gpqa": content.get("gpqa", 0),
                            "ifeval": content.get("ifeval", 0),
                            "bbh": content.get("bbh", 0),
                            "gsm8k": content.get("gsm8k", 0),
                            "hellaswag": content.get("hellaswag", 0)
                        }
    except Exception as e:
        print(f"Error fetching benchmark data: {e}")
    
    return scores_map


async def _fetch_leaderboard_snapshot() -> dict:
    """Fetches public leaderboard data without requiring tokens."""
    url = "https://datasets-server.huggingface.co/rows?dataset=open-llm-leaderboard%2Fresults&config=default&split=train&limit=150"
    scores = {}
    # try:
    async with httpx.AsyncClient(timeout=15.0) as client:
        resp = await client.get(url)
        if resp.status_code == 200:
            rows = resp.json().get("rows", [])
            for r in rows:
                row_data = r.get("row", {})
                # Clean the ID to handle inconsistent naming in the dataset
                m_id = (row_data.get("model_name") or row_data.get("config") or "").lower()
                if m_id: scores[m_id] = row_data
    # except Exception:
    #     pass # Pipeline will continue with 0.0 scores if fetch fails
    return scores

async def _resolve_all_bench(filtered_models: list, quant_map: dict, deep: bool) -> dict:
    """
    Downloads leaderboard data via Parquet (bypassing broken JSON API),
    saves raw JSON to DATA_DIR, and matches models.
    """
    # 1. Setup paths and download raw data
    # The Datasets Server is broken, so we fetch the Parquet export instead.
    # Note: Using the primary result file for V2
    lb_cache_path = DATA_DIR / "leaderboard_raw_snapshot.json"
    
    # Direct link to the converted parquet file on HF
    parquet_url = "https://huggingface.co/datasets/open-llm-leaderboard/results/resolve/main/default/train/0000.parquet"
    
    lb_data = {}
    
    try:
        async with httpx.AsyncClient(timeout=30.0) as client:
            resp = await client.get(parquet_url)
            if resp.status_code == 200:
                # Load Parquet from memory
                parquet_content = io.BytesIO(resp.content)
                df = pd.read_parquet(parquet_content)
                
                # Convert DataFrame to dictionary format for your matching logic
                # We use model_name or config as the primary key
                raw_rows = df.to_dict(orient="records")
                
                # SAVE RAW DATA TO DISK (matching your previous workflow)
                lb_cache_path.write_text(json.dumps(raw_rows, indent=2, default=str))
                
                for content in raw_rows:
                    # Schema V2 uses 'config' or 'model_name'
                    m_id = (content.get("model_name") or content.get("config") or "").lower()
                    if m_id:
                        lb_data[m_id] = content
                
                print(f"DEBUG: Leaderboard fetch success! Loaded {len(lb_data)} models.")
            else:
                print(f"DEBUG: Leaderboard fetch failed. Status: {resp.status_code}")
    except Exception as e:
        print(f"DEBUG: Leaderboard fetch failed with error: {e}")

    # 2. Matching Logic (remains mostly the same, but more robust)
    results = {}
    for m in filtered_models:
        mid = m.get("id", "")
        mid_clean = mid.lower()
        
        core_name = mid_clean.split('/')[-1].replace("-gguf", "").replace(".gguf", "").replace("-quantized", "")
        
        matched_row = None
        for lb_id, row in lb_data.items():
            lb_id_clean = lb_id.split('/')[-1].lower()
            # Check for exact match or substring containment
            if lb_id in mid_clean or core_name in lb_id_clean or lb_id_clean in core_name:
                matched_row = row
                break
        
        if matched_row:
            # 3. Apply your _INTEL_WEIGHTS scoring
            score_sum = 0.0
            weight_sum = 0.0
            details = {}

            # IMPORTANT: V2 Leaderboard nested values often live inside the 'results' or 'metrics' keys
            # or are flattened. This logic assumes flattened keys.
            for alias, internal_name in _BENCH_ALIASES.items():
                val = matched_row.get(internal_name)
                
                # Handle cases where the value might be a dict (common in V2)
                if isinstance(val, dict):
                    val = val.get("score") # Adjust based on specific benchmark structure
                
                if val is not None and isinstance(val, (int, float)):
                    w = _INTEL_WEIGHTS.get(internal_name, 1.0)
                    score_sum += (val * w)
                    weight_sum += w
                    details[internal_name] = val

            final_avg = score_sum / weight_sum if weight_sum > 0 else 0.0
            
            results[mid] = {
                "average": round(final_avg, 2),
                "confidence": 1.0 if core_name in str(matched_row.get("model_name", "")).lower() else 0.6,
                "source": "HF_Leaderboard_V2_Parquet",
                "inferred": True,
                **details
            }
        else:
            results[mid] = {"average": 0.0, "confidence": 0.0, "source": "none"}
    return results

async def _resolve_bench(client, model: dict, quant: str, deep: bool) -> dict:
    mid = model.get("id", "")
    try:
        # Fetch structured evaluation results directly from HF Metadata expand=["evalResults"] pulls the verified scores from the leaderboard
        info = model_info(mid, expand=["evalResults"])
        scores = {}
        if hasattr(info, 'eval_results') and info.eval_results:
            for res in info.eval_results:
                # Normalize the name (e.g., 'cais/hle' -> 'hle')
                metric = res.metric_type.lower()
                # Check against our aliases
                canonical = _BENCH_ALIASES.get(metric) or metric
                if canonical in _INTEL_WEIGHTS:
                    scores[canonical] = res.metric_value
        # If no direct scores, check the "Base Model" (crucial for GGUFs)
        if not scores:
            base_model = getattr(info, 'base_model', None)
            if base_model:
                # Recursively fetch for the base model
                return await _resolve_bench(client, {"id": base_model}, quant, deep)
        if scores:
            avg = sum(scores.values()) / len(scores)
            return {**scores, "average": avg, "confidence": 1.0, "source": "hf-metadata"}
    except Exception as e:
        print(f"Metadata fetch failed for {mid}: {e}")

    s = _scores_from_static(mid)
    if s and s.get("average",0) > 0:
        s.setdefault("average", _bench_to_avg(s))
        return s
    cs = _scores_from_card(model)
    if cs:
        avg = _bench_to_avg(cs)
        if avg > 0: return {**cs,"average":avg,"confidence":0.9,"source":"card-metadata","inferred":False}
    for hop, base_id in enumerate(_base_chain(model)):
        bs = _scores_from_static(base_id)
        if bs and bs.get("average",0) > 0:
            conf = bs.get("confidence",1.0) * (0.80 ** (hop+1))
            return {**{k:v for k,v in bs.items() if k not in ("confidence","source","inferred")}, 
                    "confidence":conf,"source":f"base-inherit:{base_id.split('/')[-1]}","inferred":True, 
                    "average":bs.get("average",0)}
    readme = await _fetch_readme(client, mid)
    if readme:
        rs = _scores_from_readme(readme)
        avg = _bench_to_avg(rs)
        if avg > 0: return {**rs,"average":avg,"confidence":0.85,"source":"readme-table","inferred":False}
        return rs
    return {}

# async def _resolve_all_bench(models: list, quant_map: dict, deep: bool) -> dict:
#     results = {}
#     timeout = httpx.Timeout(5.0, connect=4.0, read=8.0)
#     async with httpx.AsyncClient(timeout=timeout, headers={"User-Agent":"Mozilla/5.0"}) as client:
#         for i in range(0, len(models), 20):
#             batch = models[i:i+20]
#             tasks = [_resolve_bench(client, m, quant_map.get(m.get("id",""),"Q4_K_M"), deep) for m in batch]
#             try:
#                 batch_r = await asyncio.wait_for(asyncio.gather(*tasks, return_exceptions=True), timeout=60.0)
#             except asyncio.TimeoutError:
#                 batch_r = [{}] * len(batch)
#             for m, r in zip(batch, batch_r):
#                 results[m.get("id","")] = r if isinstance(r, dict) else {}
#     return results

# -- HuggingFace fetch --

# _HF_HEADERS = {"User-Agent": "Mozilla/5.0 (compatible; AICalc/1.0)"}

def parse_params_b(name: str) -> float | None:
    nl = name.lower().replace("_","-")
    for pat in [r'(\d+\.?\d*)-?b(?:illion)?(?:\b|[_\-])', r'(?:^|[_\-])(\d+\.?\d*)b(?:[_\-]|$)', r'\b(\d+\.?\d*)b\b']:
        m = re.search(pat, nl)
        if m:
            v = float(m.group(1))
            if 0.1 <= v <= 500: return v
    return None

def extract_params(m: dict) -> float | None:
    st = (m.get("safetensors") or {}).get("total")
    if st and st > 1e6: return round(st / 1e9, 2)
    for field in ("id","modelId"):
        v = parse_params_b(str(m.get(field,"")))
        if v: return v
    return None

def gguf_quants_from_siblings(siblings: list) -> list:
    found = set()
    for s in (siblings or []):
        fn = s.get("rfilename","").upper()
        if not fn.endswith(".GGUF"): continue
        for q in QUANTS:
            if q.upper() in fn.replace("-","_"): found.add(q); break
    return list(found)

async def _fetch_gguf_models(preset: dict, extra_query: str) -> list:
    seen, all_m = set(), []
    queries = ([extra_query.strip()] if extra_query.strip() else []) + list(preset.get("queries",[]))
    timeout = httpx.Timeout(connect=5.0, read=12.0, write=3.0, pool=2.0)
    reqs = []
    for q in queries[:3]:
        for sort in ("likes","downloads"):
            reqs.append({"search":q,"filter":["text-generation","gguf"],"library":"gguf","sort":sort,"direction":"-1","limit":100,"full":"true"})
    for sort in ("likes","downloads"):
        reqs.append({"filter":["text-generation","gguf"],"library":"gguf","sort":sort,"direction":"-1","limit":200,"full":"true"})
    for tag in preset.get("any_tags",[])[:3]:
        reqs.append({"filter":["text-generation","gguf",tag],"library":"gguf","sort":"likes","direction":"-1","limit":60,"full":"true"})

    async def _one(c, p):
        try: r = await c.get("https://huggingface.co/api/models", params=p); return r.json() if r.status_code==200 else []
        except: return []

    async with httpx.AsyncClient(timeout=timeout, headers={"User-Agent":"Mozilla/5.0"}) as c:
        try:   batches = await asyncio.wait_for(asyncio.gather(*[_one(c,p) for p in reqs]), timeout=30.0)
        except asyncio.TimeoutError: batches = []
    for batch in (batches or []):
        if isinstance(batch, list):
            for m in batch:
                mid = m.get("id","")
                if mid and mid not in seen: seen.add(mid); all_m.append(m)
    return all_m

# --- Search pipeline (background thread) ---

_job = {"running":False,"status":"idle","step":"","progress":[],"filtered":None,"result":None,"error":None,"params":None}
_job_lock = threading.Lock()

def _job_up(**kw):
    with _job_lock:
        _job.update(kw)
        if "step" in kw: _job["progress"].append(f"[{datetime.now().strftime('%H:%M:%S')}] {kw['step']}")

def _job_reset():
    with _job_lock:
        _job.update(running=False,status="idle",step="",progress=[],filtered=None,result=None,error=None,params=None)

def _run_thread(params: dict):
    loop = asyncio.new_event_loop(); asyncio.set_event_loop(loop)
    try:
        result = loop.run_until_complete(_pipeline(params))
        _job_up(running=False, status="Done.", result=result)
    except Exception:
        _job_up(running=False, status="Failed.", error=traceback.format_exc())
    finally:
        loop.close()

async def _pipeline(params: dict) -> dict:
    hw      = params["hw"]
    deep    = params.get("deep_scan", False)
    must_w  = [w.strip().lower() for w in params["must_contain"].split(",") if w.strip()]
    any_w   = [w.strip().lower() for w in params["any_contain"].split(",")  if w.strip()]
    excl_w  = [w.strip().lower() for w in params["exclude"].split(",")      if w.strip()]
    preset  = TASK_PRESETS.get(params["task_preset"], TASK_PRESETS["instruct"])
    key     = re.sub(r'[^\w]','_',f"{params['task_preset']}_{params['min_params']}_{params['max_params']}_{params['extra_query'][:20]}")[:60]
    cache   = DATA_DIR / f"hf_{key}.json"

    raw = None
    if not params.get("force_refresh") and cache.exists():
        age = (datetime.now() - datetime.fromtimestamp(cache.stat().st_mtime)).total_seconds()
        if age < 21600:
            try: raw = json.loads(cache.read_text()); _job_up(step=f"Loaded {len(raw)} from cache ({int(age/60)}m old)")
            except: raw = None
    if raw is None:
        _job_up(status="Fetching from HuggingFace...", step="Starting HF queries")
        raw = await _fetch_gguf_models(preset, params["extra_query"])
        _job_up(step=f"Fetched {len(raw)} raw models - caching")
        cache.write_text(json.dumps(raw))
    params["fetched"] = len(raw)
    _job_up(status=f"Filtering {len(raw)} models...", step=f"Params {params['min_params']}-{params['max_params']} | min_tps={params['min_tps']}")
    filtered, counts = [], {"oom":0,"speed":0,"params":0,"quant":0,"tags":0,"exclude":0}
    for idx, m in enumerate(raw):
        if idx % 50 == 0 and idx > 0: _job_up(step=f"  {idx}/{len(raw)} checked, {len(filtered)} passing")
        blob = (m.get("id","") + " " + " ".join(t.lower() for t in (m.get("tags") or []))).lower()
        if excl_w and any(w in blob for w in excl_w): counts["exclude"] += 1; continue
        if must_w and not all(w in blob for w in must_w): counts["tags"] += 1; continue
        if any_w  and not any(w in blob for w in any_w):  counts["tags"] += 1; continue
        pb = extract_params(m)
        if pb is None or not (params["min_params"] <= pb <= params["max_params"]): counts["params"] += 1; continue
        avail = set(gguf_quants_from_siblings(m.get("siblings") or []))
        if not avail: counts["quant"] += 1; continue
        any_fits = any_fast = False
        for q in sorted(avail & set(QUANTS), key=lambda x: QUANTS[x]["bpp"]):
            p = full_perf(pb, q, params["ctx_tokens"], hw)
            if p["mem_ok"]:
                any_fits = True
                if p["tps"] >= params["min_tps"]: any_fast = True; break
        if not any_fits:  counts["oom"]   += 1; continue
        if not any_fast:  counts["speed"] += 1; continue
        filtered.append({**m,"params_b":pb,"avail_quants":sorted(avail&set(QUANTS),key=lambda q:QUANTS[q]["bpp"]),
                         "leaderboard_avg":0.0,"lb_detail":{},"bench_confidence":0.0,"bench_source":"none","bench_inferred":False})
    _job_up(step=f"Filter done: {len(filtered)} pass all hard gates")
    _job_up(status="Resolving benchmarks (multi-source)...", step=f"Checking {len(filtered)} models")
    quant_map = {m.get("id",""): max((set(m.get("avail_quants",[]))&set(QUANTS)) or {"Q4_K_M"}, key=lambda x:QUANTS[x]["quality"]) for m in filtered}
    bench_res = await _resolve_all_bench(filtered, quant_map, deep)
    lb_hits   = 0
    for m in filtered:
        mid = m.get("id",""); bs = bench_res.get(mid,{}); avg = bs.get("average",0.0) or 0.0
        m.update(leaderboard_avg=round(float(avg),2) if avg else 0.0,
                 lb_detail={k:v for k,v in bs.items() if k not in ("confidence","source","inferred","average")},
                 bench_confidence=float(bs.get("confidence",0.0)),
                 bench_source=str(bs.get("source","none")),
                 bench_inferred=bool(bs.get("inferred",False)))
        if avg > 0: lb_hits += 1
    _job_up(step=f"Bench resolved: {lb_hits}/{len(filtered)} have scores")
    with _job_lock: _job["filtered"] = filtered

    _job_up(status="Scoring...", step=f"target_tps={params['target_tps']}")
    rows = _rank_filtered(filtered, hw, params["ctx_tokens"], params["target_tps"], params["weights"], params["top_n"])
    _job_up(step=f"Ranked {len(rows)} quant-variants")
    _write_csv(rows, key)
    top = rows[0] if rows else None
    _job_up(step=f"Done. Top: {top['id']} [{top['quant']}] score={top['total_score']}" if top else "Done - no results")
    return {"rows":rows,"stats":{"fetched":params["fetched"],"passed":len(filtered),"ranked":len(rows),
            "oom":counts["oom"],"speed":counts["speed"],"params":counts["params"],"quant":counts["quant"],
            "tags":counts["tags"],"exclude":counts["exclude"],"lb_hits":lb_hits,
            "csv":f"data/ai_tools/analysis_{key}.csv","deep":deep},"weights":params["weights"]}

# HW Profiles

def _load_hw_profiles() -> list: return json.loads((DATA_DIR / "hw_profiles.json").read_text()) if (DATA_DIR / "hw_profiles.json").exists() else []
def _save_hw_profiles(p: list): (DATA_DIR / "hw_profiles.json").write_text(json.dumps(p, indent=2))

def _hw_from_form(f) -> dict:
    hw = DEFAULT_HW.copy()
    for k, lo, hi in [("vram_gb", 0.5, 512), ("shared_gb", 0, 512), ("sys_ram_gb", 0, 1024), ("os_overhead_gb", 0, 64),("mem_bw_gbps", 1, 10000), ("gpu_tflops_fp16", 0.1, 1000)]:
        hw[k] = max(lo, min(float(f.get(k, hw[k])), hi))
    hw["gpu_eff"] = DEFAULT_HW["gpu_eff"]
    return hw

# --- Session monitor ---

def _get_ollama_conn() -> dict | None:
    for f in sorted((DATA_DIR / "_connections").glob("*.json")):
        try:
            c = json.loads(f.read_text())
            if c.get("connection_type") == "ollama": c["_id"] = f.stem; return c
        except: pass
    return None

def _ollama_base(conn: dict) -> str:
    v = conn.get("values",{})
    return f"{'https' if v.get('tls') else 'http'}://{v.get('host','127.0.0.1')}:{v.get('port',11434)}{v.get('base_path','')}" 

def _fetch_running(conn: dict) -> list:
    try:
        async def _async():
            async with httpx.AsyncClient(timeout=httpx.Timeout(5.0, connect=3.0,read=5.0)) as c:
                r = await c.get(f"{_ollama_base(conn)}/api/ps")
                return r.json().get("models",[]) if r.status_code == 200 else []
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        return loop.run_until_complete(_async())
    except: return []

async def _fetch_running(conn: dict) -> list:
    try:  
        async with httpx.AsyncClient(timeout=httpx.Timeout(5.0, connect=3.0,read=5.0)) as c:  
            r = await c.get(f"{_ollama_base(conn)}/api/ps")  
            return r.json().get("models",[]) if r.status_code == 200 else []  
    except: return []  


def _parse_loaded(name: str):
    pb = parse_params_b(name) or parse_params_b(name.split(":")[-1])
    nu = name.upper().replace("-","_")
    q = next((q for q in sorted(QUANTS, key=lambda x: len(x), reverse=True) if q.upper() in nu), "Q4_K_M")
    return pb, q

def _session_card_html(m: dict, hw: dict, alt_profiles: list) -> str:
    name = m.get("name","")
    size_gb = m.get("size",0) / 1e9
    vram_gb = m.get("size_vram", m.get("size",0)) / 1e9
    pb, q = _parse_loaded(name)
    perf = full_perf(pb, q, 32768, hw) if pb else {}
    tps = perf.get("tps",0); tc = "#00ffa2" if tps > 0 else "#ff9a3c"
    vt = total_vram(hw)
    pct_m = min(vram_gb / max(vt,0.001) * 100, 100)
    kv_16k = kv_cache_gb_for_ctx(pb, 16384) if pb else 0
    kv_pct = min(kv_16k / max(vt,0.001) * 100, 100)
    free_pct = max(100 - pct_m - kv_pct, 0)
    expires = m.get("expires_at",""); exp_str = f' | expires {expires[:16].replace("T"," ")}' if expires else ""
    qlabel = f'{q} ({QUANTS.get(q, QUANTS["Q4_K_M"]).get("quality",0) * 100:.0f}% quality)' if pb else ""
    ctx_16k_ok = (perf.get("model_gb",99) + kv_16k) <= total_mem(hw) if pb else False

    quant_rows = ""
    if pb:
        alternatives = [(aq, QUANTS[aq], full_perf(pb, aq, 32768, hw)) for aq in QUANTS if aq != q and full_perf(pb, aq, 32768, hw)["fits_total"]]
        up   = sorted([(aq,qi,p) for aq,qi,p in alternatives if qi["quality"] > QUANTS.get(q,q).get("quality",0)], key=lambda x: x[1]["quality"] - QUANTS[q]["quality"])[:3]
        down = sorted([(aq,qi,p) for aq,qi,p in alternatives if qi["quality"] < QUANTS.get(q,q).get("quality",0)], key=lambda x: QUANTS[q]["quality"] - x[1]["quality"])[:2]
        for aq, qi, ap in up + down:
            delta_q  = (qi["quality"] - QUANTS.get(q,q).get("quality",0)) * 100
            delta_gb = round(model_gb(pb,aq) - model_gb(pb,q), 2)
            col = "#00ffa2" if delta_q > 0 else "#ff9a3c"
            quant_rows += f'<tr><td class="qn">{aq}</td><td style="color:{col}">{delta_q:+.0f}%</td><td>{delta_gb:+.1f} GB</td><td style="color:{_tcolor(ap["tps"])}">{ap["tps"]:.1f} t/s</td></tr>'

    quant_table = f'<details class="s-opts"><summary class="s-opts-sum">Quantization alternatives</summary><div class="tbl-scroll"><table class="cmp-table"><thead><tr><th>Quant</th><th>Quality</th><th>Size</th><th>T/s</th></tr></thead><tbody>{quant_rows}</tbody></table></div></details>' if quant_rows else ""

    alt_rows = ""
    if pb and alt_profiles:
        for profile in alt_profiles:
            ahw = _hw_from_form(profile)
            ap = estimate_tps(pb, q, ahw)
            if not ap["fits_total"]: ac, note = "#ff4444", "OOM"
            else:
                ratio = ap["tps"] / max(tps,0.001)
                note = f'+{(ratio-1)*100:.0f}%' if ratio > 1.05 else f'{(ratio-1)*100:.0f}%'
                ac = "#00ffa2" if ratio > 1.1 else "#ffcc00" if ratio > 0.9 else "#ff8c42"
            alt_rows += f'<tr><td>{profile["name"]}</td><td style="color:{ac}">{ap["tps"]:.1f} t/s <span style="font-size:.65rem;color:var(--text_muted)">{note}</span></td><td style="font-size:.68rem;color:var(--text_muted)">{ahw["vram_gb"]:.0f}+{ahw["shared_gb"]:.0f} GB</td></tr>'

    alt_table = f'<details class="s-opts"><summary class="s-opts-sum">Hardware comparison</summary><div class="tbl-scroll"><table class="cmp-table"><thead><tr><th>Profile</th><th>T/s</th><th>VRAM</th></tr></thead><tbody>{alt_rows}</tbody></table></div></details>' if alt_rows else ""

    layers, hidden, attn, kv_heads = arch_for(pb) if pb else (0,0,0,0)
    bpt = kv_bytes_per_token(pb, kv_bits) if pb else 0
    arch_info = f"<div style='font-size:.66rem;color:var(--text_muted);font-family:var(--font-mono);margin-top:.2rem'>Layers:{layers} Hidden:{hidden} KV-heads:{kv_heads} | {bpt//1024:.1f} KB/token | {int(1024**3/max(bpt,1))//1000:.0f}K tokens/GB</div>" if pb else ""

    return f"""<div class="glass s-card">
    <div class="s-name">{name}</div>
    <div class="s-stats">
        <span style="color:{tc}">~{tps:.1f} t/s</span>
        <span>{qlabel}</span>
        <span style='color:var(--text_muted)'>{pb}B params</span>
        <span style='color:var(--text_muted)'>prefill {perf.get('prefill_tps',0):.1f} t/s</span>
        <span style='color:{("#00ffa2" if ctx_16k_ok else "#ffcc00") if pb else "var(--text_muted)"}'>{("16K ctx OK" if ctx_16k_ok else "16K ctx: RAM spill") if pb else ""}</span>
    </div>
    <div class="s-mem-bar">
        <div class="s-seg s-seg-m" style="width:{pct_m:.1f}%"></div>
        <div class="s-seg s-seg-k" style="width:{kv_pct:.1f}%"></div>
        <div class="s-seg s-seg-f" style="width:{free_pct:.1f}%"></div>
    </div>
    <div class="s-mem-labels">
        <span style="color:#3d9aff">Model VRAM {vram_gb:.1f} GB</span>
        <span style="color:#b06aff">Model RAM {perf.get('ram_used',0):.1f} GB</span>
        <span style="color:var(--text_muted)">Free {(vt - perf.get('model_gb',0) - perf.get('kv_in_vram',0)):.2f} GB</span>
    </div>
    {quant_table}
    {alt_table}
</div>"""

# --- HTML helpers ---

def _tcolor(tps: float) -> str:  return "#00ffa2" if tps >= 8 else "#ffcc00" if tps >= 3 else "#ff8c42" if tps >= 0.5 else "#ff4444"
def _bar(used, total, color) -> str: return f'<div class="bar-bg"><div class="bar-fill" style="width:{min(used / max(total,0.001) * 100, 100):.1f}%;background:{color}"></div></div>'
def _quant_opts(selected="Q4_K_M") -> str: return "".join(f'<option value="{q}" {"selected" if q==selected else ""}>{q} ({round(QUANTS[q]["quality"]*100):.0f}% quality, {QUANTS[q]["bpp"]:.3f} bpp)</option>' for q in QUANTS)

OH_OPTS = [("Linux bare (1.5 GB)",1.5), ("Linux+Docker (2.5 GB)",2.5), ("Windows bare (3.5 GB)",3.5), ("Windows+Docker (5.0 GB)",5.0)]

def _hw_bar(hw: dict) -> str:
    vt, ur = total_vram(hw), usable_ram(hw)  
    oh_sel = "".join(f'<option value="{v}" {"selected" if abs(v - hw.get("os_overhead_gb",3.5)) < 0.05 else ""}>{lbl}</option>' for lbl,v in OH_OPTS)
    return f"""<details class="hwp">  
        <summary>[HW] {vt:.0f} GB VRAM + {ur:.1f} GB usable RAM = <b>{vt+ur:.1f} GB pool</b> | {hw["mem_bw_gbps"]:.0f} GB/s <span style="opacity:.6;font-size:.7rem">(click to edit)</span></summary>  
        <div class="hwg">  
            <label>Dedicated VRAM (GB)<input class="hwin" name="vram_gb" type="number" step="any" value="{hw['vram_gb']}"></label>  
            <label>Shared/iGPU VRAM (GB)<input class="hwin" name="shared_gb" type="number" step="any" value="{hw['shared_gb']}"><span class="dim tiny">iGPU reserved from RAM</span></label>  
            <label>System RAM (GB)<input class="hwin" name="sys_ram_gb" type="number" step="any" value="{hw['sys_ram_gb']}"></label>  
            <label>OS Overhead<div style="display:flex;gap:.3rem"><select class="hwin" style="flex:0;min-width:8rem" onchange="document.querySelector('.hwin[name=os_overhead_gb]').value=this.value;syncHW()">{oh_sel}</select><input class="hwin" name="os_overhead_gb" type="number" step="0.5" value="{hw.get('os_overhead_gb',3.5)}" style="flex:1"></div></label>  
            <label>Memory BW (GB/s)<input class="hwin" name="mem_bw_gbps" type="number" step="any" value="{hw['mem_bw_gbps']}"><span class="dim tiny">GPU-CPU unified for iGPU</span></label>  
            <label>GPU TFLOPS FP16<input class="hwin" name="gpu_tflops_fp16" type="number" step="any" value="{hw['gpu_tflops_fp16']}"><span class="dim tiny">for prefill speed estimate</span></label>  
        </div>  
    </details>  
    <div id="hw-fields" style="display:none"></div>"""

def _weight_slider(dim: str, weights: dict) -> str:  
    val = weights.get(dim, DEFAULT_WEIGHTS[dim])  
    return (f'<label class="wsl" style="--wc:{SCORE_DIM_COLORS[dim]}">'  
            f'<span class="wsln">{SCORE_DIM_LABELS[dim]}</span>'  
            f'<input class="wslider" type="range" name="w_{dim}" min="0" max="10" step="1" value="{val}" oninput="this.parentElement.querySelector(\'.wsv\').textContent=this.value">'  
            f'<span class="wsv">{val}</span></label>')

def _score_bar(dim: str, raw: float, w: float, total_w: float) -> str:  
    color = SCORE_DIM_COLORS[dim]; contrib = raw * w / max(total_w,0.001)  
    return (f'<div class="sbrow"><span class="sbl" style="color:{color}">{SCORE_DIM_LABELS[dim]}</span>'  
            f'<div class="sbbar"><div class="sbfill" style="width:{raw*100:.1f}%;background:{color}"></div></div>'  
            f'<span class="sbv">{raw:.3f}</span><span class="sbw">w={w:.0f}</span>'  
            f'<span class="sbc" style="color:{color}">+{contrib:.4f}</span></div>')

def _lb_pills(lb: dict, source: str = "", confidence: float = 0, inferred: bool = False) -> str:  
    if not lb: return '<span class="dim tiny">no benchmark data</span>'  
    bench_show = [("mmlu","MMLU"),("arc","ARC"),("hellaswag","HS"),("truthfulqa","TQA"),  
                  ("winogrande","Wino"),("gsm8k","GSM8K"),("ifeval","IFEval"),("bbh","BBH"),  
                  ("gpqa","GPQA"),("humaneval","HEval"),("average","Avg")]  
    parts = []  
    for key, short in bench_show:  
        v = lb.get(key)  
        if v and float(v) > 0:  
            v = float(v); col = "#00ffa2" if v>=75 else "#ffcc00" if v>=55 else "#ff9a3c" if v>=40 else "#888"  
            parts.append(f'<span class="lbp" style="color:{col}">{short} {v:.1f}</span>')  
    if source and source != "none":  
        src_map = {"static-anchor":("anchor","#00ffa2"),"static-fuzzy":("~anchor","#88c"),  
                   "card-metadata":("card","#3d9aff"),"readme-table":("readme","#ffcc00")}  
        slabel, scol = source, "#888"  
        for k,(l,c) in src_map.items():  
            if source.startswith(k): slabel,scol = l,c; break  
        if source.startswith("base-inherit"): slabel,scol = "base&#x2191;","#ff9a3c"  
        inf_tag = " ~inferred" if inferred else ""  
        conf_pct = f" {confidence*100:.0f}%" if confidence<0.95 else ""  
        parts.append(f'<span class="lbp" style="color:{scol}">{slabel}{inf_tag}{conf_pct}</span>')  
    return "".join(parts)

def _fmt(num: int) -> str: return "{:,}".format(num)

# -- Panel HTML --

def _panel_session() -> str:
    return f"""<div class="ait-rp">
        <div class="ait-rp-hd">AI Calc</div>
        <button class="ait-rp-btn" hx-get="{_P}/view/session"    hx-target="#ait-workspace" hx-swap="innerHTML">&#x25CF; Session Monitor</button>
        <button class="ait-rp-btn" hx-get="{_P}/view/calc"       hx-target="#ait-workspace" hx-swap="innerHTML">&#x223C; Speed Calc</button>
        <button class="ait-rp-btn" hx-get="{_P}/view/ctx"        hx-target="#ait-workspace" hx-swap="innerHTML">&#x25A4; Context Planner</button>
        <button class="ait-rp-btn" hx-get="{_P}/view/compare"    hx-target="#ait-workspace" hx-swap="innerHTML">&#x2261; Quant Compare</button>
        <button class="ait-rp-btn" hx-get="{_P}/view/sweep"      hx-target="#ait-workspace" hx-swap="innerHTML">&#x2227; Sweep Chart</button>
        <button class="ait-rp-btn" hx-get="{_P}/view/search"     hx-target="#ait-workspace" hx-swap="innerHTML">&#x1F50D; Model Finder</button>
        <button class="ait-rp-btn" hx-get="{_P}/view/image"      hx-target="#ait-workspace" hx-swap="innerHTML">&#x1F5BC; Image Models</button>
        <button class="ait-rp-btn" hx-get="{_P}/view/hwprofiles" hx-target="#ait-workspace" hx-swap="innerHTML">&#x1F9E0; HW Profiles</button>
        <button class="ait-rp-btn" hx-get="{_P}/view/files"      hx-target="#ait-workspace" hx-swap="innerHTML">&#x1F4C2; Cache Files</button>
    </div>"""

def _panel_calc(hw: dict) -> str:  
    kv_opts = '<option value="16">FP16 (full precision)</option><option value="8" selected>INT8 (recommended)</option><option value="4">INT4 (saves RAM, slight quality loss)</option>'  
    return f"""<div style="padding:.75rem .6rem">  
        <div class="fsect-hd">Speed + Memory Calculator</div>  
        <p style="font-size:.72rem;color:var(--text_muted);margin:.2rem 0 .7rem">Estimates token generation speed, prefill speed, KV cache cost, and memory layout for a given model configuration.</p>  
        {_hw_bar(hw)}  
        <form class="pform" hx-post="{_P}/calc" hx-target="#calc-out" hx-include="#hw-fields">  
            <div class="frow">  
                <label>Params (B)<input class="fin" name="params_b" type="number" step="any" value="7"></label>  
                <label>Quantization<select class="fin" name="quant">{_quant_opts()}</select></label>  
                <label>Context (tokens)<input class="fin" name="ctx_tokens" type="number" value="20992"></label>  
                <label>KV Quant<select class="fin" name="kv_bits">{kv_opts}</select></label>  
                <button class="rbtn" type="submit">Calculate</button>  
            </div>  
        </form>  
        <div id="calc-out" class="rzone"><div class="placeholder">Enter parameters above</div></div>  
    </div>"""

def _panel_ctx(hw: dict) -> str:  
    kv_opts = '<option value="16">FP16</option><option value="8" selected>INT8</option><option value="4">INT4</option>'  
    return f"""<div style="padding:.75rem .6rem">  
        <div class="fsect-hd">Context Window Planner</div>  
        <p style="font-size:.72rem;color:var(--text_muted);margin:.2rem 0 .7rem">  
            Given a model and hardware, see how many tokens fit and design your context layout.  
            Essential for planning pinned blocks vs sliding history vs knowledge context.  
        </p>  
        {_hw_bar(hw)}  
        <form class="pform" hx-post="{_P}/ctx_breakdown" hx-target="#ctx-out" hx-include="#hw-fields">  
            <div class="frow">  
                <label>Params (B)<input class="fin" name="params_b" type="number" step="any" value="7"></label>  
                <label>Quantization<select class="fin" name="quant">{_quant_opts()}</select></label>  
                <label>KV Quant<select class="fin" name="kv_bits">{kv_opts}</select></label>  
                <button class="rbtn" type="submit">Analyze</button>  
            </div>  
        </form>  
        <div id="ctx-out" class="rzone"><div class="placeholder">Analyze a model to see context breakdown</div></div>  
        <div id="ctx-block-planner" style="margin-top:1.2rem">  
            <div class="fsect-hd">Context Block Budget</div>  
            <p style="font-size:.72rem;color:var(--text_muted);margin:.2rem 0 .5rem">  
                Add named blocks to see if they fit. Analyze above first to get your token budget.  
                Save as a chat profile to reuse in the Chat module.  
            </p>  
            <div id="ctx-block-list" style="display:flex;flex-direction:column;gap:.25rem;margin-bottom:.5rem"></div>  
            <div style="display:flex;gap:.4rem;align-items:center;flex-wrap:wrap">  
                <input type="text" id="blk-name" placeholder="Block name" class="fin" style="flex:2;min-width:8rem">  
                <input type="number" id="blk-tokens" placeholder="Tokens" class="fin" style="flex:1;min-width:5rem" value="1000">  
                <select id="blk-type" class="fin" style="flex:1;min-width:6rem">  
                    <option value="pinned">Pinned (always in ctx)</option>  
                    <option value="knowledge">Knowledge (large ref)</option>  
                    <option value="goal">Goal/outline</option>  
                    <option value="history">History (sliding)</option>  
                    <option value="workspace">Workspace/scratch</option>  
                </select>  
                <button class="rbtn" type="button" onclick="addCtxBlock()">Add</button>  
            </div>  
            <div id="ctx-block-budget" style="margin-top:.5rem;font-size:.75rem;color:var(--text_muted)">Run analysis above to see token budget</div>  
            <div style="margin-top:.5rem;display:flex;gap:.4rem">  
                <button class="rbtn" type="button" onclick="saveCtxProfile()" style="font-size:.75rem">Save as Chat Profile</button>  
                <button class="rbtn" type="button" onclick="clearCtxBlocks()" style="background:none;font-size:.75rem">Clear</button>  
            </div>  
        </div>  
    </div>"""

def _panel_compare(hw: dict) -> str:  
    return f"""<div style="padding:.75rem .6rem">  
        <div class="fsect-hd">Quantization Comparison</div>  
        <p style="font-size:.72rem;color:var(--text_muted);margin:.2rem 0 .7rem">Compare all 15 quantizations for a model size. Shows exact tradeoffs between speed, size, memory, and quality.</p>  
        {_hw_bar(hw)}  
        <form class="pform" hx-post="{_P}/compare" hx-target="#cmp-out" hx-include="#hw-fields">  
            <div class="frow">  
                <label>Params (B)<input class="fin" name="params_b" type="number" step="any" value="14"></label>  
                <label>Context (tokens)<input class="fin" name="ctx_tokens" type="number" value="20992"></label>  
                <button class="rbtn" type="submit">Compare All Quants</button>  
            </div>  
        </form>  
        <div id="cmp-out" class="rzone"><div class="placeholder">Compare every quantization for a model size</div></div>  
    </div>"""

def _panel_sweep(hw: dict) -> str:  
    mqopts = "".join(f'<option value="{q}" {"selected" if q in ("Q4_K_M","Q5_K_M","Q8_0") else ""}>{q}</option>' for q in QUANTS)  
    return f"""<div style="padding:.75rem .6rem">  
        <div class="fsect-hd">T/s Sweep Chart</div>  
        <p style="font-size:.72rem;color:var(--text_muted);margin:.2rem 0 .7rem">T/s vs model size for selected quantizations. Reference lines show minimum readable speeds for different use cases.</p>  
        {_hw_bar(hw)}  
        <form class="pform" hx-post="{_P}/sweep" hx-target="#sw-out" hx-include="#hw-fields">  
            <div class="frow" style="align-items:flex-start">  
                <label>Select Quants<select class="fin" name="quants" multiple size="6">{mqopts}</select><span class="dim tiny">Ctrl/Cmd for multiple</span></label>  
                <button class="rbtn" type="submit" style="align-self:flex-end">Generate Chart</button>  
            </div>  
        </form>  
        <div id="sw-out" class="rzone"><div class="placeholder">Select quantizations and generate chart</div></div>  
    </div>"""

def _panel_image(hw: dict) -> str:  
    model_opts = "".join(f'<option value="{k}">{k} - {v["note"]}</option>' for k, v in IMAGE_MODELS.items())  
    return f"""<div style="padding:.75rem .6rem">  
        <div class="fsect-hd">Image / Video Model Estimator</div>  
        <p style="font-size:.72rem;color:var(--text_muted);margin:.2rem 0 .7rem">Estimates image/video generation speed and memory requirements based on your hardware.</p>  
        {_hw_bar(hw)}  
        <form class="pform" hx-post="{_P}/image_calc" hx-target="#img-out" hx-include="#hw-fields">  
            <div class="frow wrap">  
                <label>Model<select class="fin" name="image_model">{model_opts}</select></label>  
                <label>Steps<input class="fin" name="steps" type="number" value="20"></label>  
                <button class="rbtn" type="submit">Estimate</button>  
            </div>  
        </form>  
        <div id="img-out" class="rzone"><div class="placeholder">Select model to estimate</div></div>  
    </div>"""

def _panel_search() -> str:  
    task_opts  = "".join(f'<option value="{k}" {"selected" if k=="instruct" else ""}>{v["label"]}</option>' for k,v in TASK_PRESETS.items())  
    wsliders   = "".join(_weight_slider(d, DEFAULT_WEIGHTS) for d in SCORE_DIMS)  
    return f"""<div style="padding:.75rem .6rem">  
        <div class="fsect-hd">[1] Hard Filters <span class="dim tiny">- models failing any gate are excluded regardless of score</span></div>  
        <form class="pform" hx-post="{_P}/search" hx-target="#srch-out" hx-swap="innerHTML">  
            <div class="frow wrap">  
                <label>Task Preset<select class="fin" name="task_preset" id="task-preset-sel" onchange="applyPreset(this.value)">{task_opts}</select><span class="dim tiny">loads suggested weights below</span></label>  
                <label>Min params (B)<input class="fin" name="min_params" type="number" step="any" value="1"></label>  
                <label>Max params (B)<input class="fin" name="max_params" type="number" step="any" value="35"></label>  
                <label>Context tokens<input class="fin" name="ctx_tokens" type="number" value="20992"></label>  
            </div>  
            <div class="frow wrap">  
                <label>Min T/s (hard floor)<input class="fin" name="min_tps" type="number" step="any" value="1.0"><span class="dim tiny">below this = excluded entirely</span></label>  
                <label>Target T/s (score ref)<input class="fin" id="target-tps-in" name="target_tps" type="number" step="any" value="5.0"><span class="dim tiny">2x target = perfect speed score</span></label>  
                <label style="flex:2">Must contain ALL (comma)<input class="fin" name="must_contain" type="text" placeholder="e.g. instruct, chat"></label>  
                <label style="flex:2">Must contain ANY (comma)<input class="fin" name="any_contain" type="text" placeholder="e.g. creative, roleplay"></label>  
            </div>  
            <div class="frow wrap">  
                <label style="flex:2">Extra HF query<input class="fin" name="extra_query" type="text" placeholder="e.g. mistral uncensored"></label>  
                <label style="flex:2">Exclude terms (comma)<input class="fin" name="exclude" type="text" placeholder="e.g. vision, embed, base"></label>  
                <label>Results cap<input class="fin" name="top_n" type="number" value="40"></label>  
            </div>  
            <div class="fsect-hd" style="margin-top:.8rem">[2] Scoring Weights <span class="dim tiny">- weighted sum / total weight = total score</span></div>  
            <div class="wsblock">{wsliders}</div>  
            <div class="frow" style="align-items:center;gap:1rem;flex-wrap:wrap;margin-top:.7rem">  
                <label style="flex-direction:row;align-items:center;gap:.4rem;flex:0;white-space:nowrap"><input type="checkbox" name="force_refresh" value="1"> Force HF refresh</label>  
                <label style="flex-direction:row;align-items:center;gap:.4rem;flex:0;white-space:nowrap" title="Fetches README.md per model for higher benchmark hit rate. Significantly slower."><input type="checkbox" name="deep_scan" value="1"> Deep bench scan (README)</label>  
                <button class="rbtn" type="submit">Search &amp; Rank <span class="htmx-indicator spin">[..]</span></button>  
            </div>  
            <div class="dim tiny" style="margin-top:.4rem">Stage 1: HF fetch + hard filter (memory, speed, params, text). Stage 2: score each (model x quant) on 6 weighted dimensions.</div>  
        </form>  
        <div id="srch-rerank" style="display:none;margin-top:.7rem">  
            <form class="pform" hx-post="{_P}/rerank" hx-target="#srch-out" hx-swap="innerHTML">  
                <div class="fsect-hd">Re-rank Cached Results <span class="dim tiny">- adjust weights without re-fetching</span></div>  
                <div class="wsblock" id="rr-wsblock">{"".join(_weight_slider(d, DEFAULT_WEIGHTS) for d in SCORE_DIMS)}</div>  
                <div class="frow" style="margin-top:.6rem;gap:1rem;align-items:center;flex-wrap:wrap">  
                    <label style="flex:0;min-width:110px">Target T/s<input class="fin" id="rr-tps" name="target_tps" type="number" step="any" value="5.0"></label>  
                    <label style="flex:0;min-width:90px">Results cap<input class="fin" name="top_n" type="number" value="40"></label>  
                    <button class="rbtn" type="submit">Re-rank <span class="htmx-indicator spin">[..]</span></button>  
                </div>  
            </form>  
        </div>  
        <div id="srch-out" class="rzone"><div class="placeholder">Configure and run search above</div></div>  
    </div>"""

def _panel_hw_profiles() -> str:  
    profiles = _load_hw_profiles()  
    cards    = "".join(  
        f'<div class="glass" style="padding:.6rem .8rem;margin-bottom:.3rem">'  
        f'<div style="display:flex;align-items:center;gap:.5rem">'  
        f'<span style="flex:1;font-weight:600;font-size:.82rem">{p["name"]}</span>'  
        f'<span style="font-size:.68rem;color:var(--text_muted);font-family:var(--font-mono)">'  
        f'{p["hw"].get("vram_gb",0):.0f}+{p["hw"].get("shared_gb",0):.0f} GB VRAM | {p["hw"].get("mem_bw_gbps",0):.0f} GB/s | {p["hw"].get("gpu_tflops_fp16",0):.1f} TFLOPS</span>'  
        f'<button class="btn-icon" style="color:#ff5f5f" hx-delete="{_P}/hw_profiles/{p["id"]}" hx-target="#hw-profiles-panel" hx-swap="outerHTML" hx-confirm="Delete?">&#x2715;</button></div></div>'  
        for p in profiles) or '<div style="color:var(--text_muted);font-size:.8rem;padding:.5rem 0">No profiles yet.</div>'  
    hw_fields = [("vram_gb","Dedicated VRAM (GB)",16),("shared_gb","Shared VRAM (GB)",8),("sys_ram_gb","System RAM (GB)",16),  
                 ("os_overhead_gb","OS Overhead (GB)",3.5),("mem_bw_gbps","Mem BW (GB/s)",89),("gpu_tflops_fp16","GPU TFLOPS FP16",8.9)]  
    return f"""<div id="hw-profiles-panel" style="padding:1rem;max-width:600px">  
        <div class="fsect-hd">Hardware Profiles</div>  
        <p style="font-size:.75rem;color:var(--text_muted);margin:0 0 .8rem">Named hardware configs for side-by-side comparison in Session Monitor and model search.</p>  
        {cards}  
        <details class="glass" style="padding:.8rem;margin-top:.5rem">  
            <summary style="cursor:pointer;font-size:.82rem;color:var(--text_muted)">+ Add Profile</summary>  
            <form hx-post="{_P}/hw_profiles/save" hx-target="#hw-profiles-panel" hx-swap="outerHTML" style="display:flex;flex-direction:column;gap:.4rem;margin-top:.5rem">  
                <label style="font-size:.73rem;color:var(--text_muted)">Profile Name<input type="text" name="name" class="module-select" placeholder="e.g. RTX 3090 Desktop" required></label>  
                <div class="hwg">{"".join(f'<label style="font-size:.73rem;color:var(--text_muted)">{lbl}<input class="hwin" name="{fn}" type="number" step="any" value="{dflt}"></label>' for fn,lbl,dflt in hw_fields)}</div>  
                <button class="rbtn" type="submit" style="margin-top:.3rem">Save</button>  
            </form>  
        </details>  
    </div>"""

def _panel_files() -> str:  
    files = sorted((set(DATA_DIR.glob("*.json")) | set(DATA_DIR.glob("*.csv"))), key=lambda f: f.stat().st_mtime, reverse=True)  
    rows  = "".join(f'<tr><td class="qn" style="font-size:.72rem">{f.name}</td><td style="font-size:.72rem">{f.stat().st_size//1024} KB</td><td style="font-size:.68rem;color:var(--text_muted)">{datetime.fromtimestamp(f.stat().st_mtime).strftime("%m-%d %H:%M")}</td></tr>' for f in list(files)[:30])  
    return f'<div style="padding:.75rem .6rem"><div class="fsect-hd">Cached Files</div><div class="tbl-scroll"><table class="cmp-table"><thead><tr><th>File</th><th>Size</th><th>Modified</th></tr></thead><tbody>{rows}</tbody></table></div></div>' if rows else '<div style="padding:.75rem .6rem"><div class="placeholder">No cached files</div></div>'

# -- Result HTML --

def _calc_html(p: dict, hw: dict, kv_gb: float, kv_bits: int, pb: float, ctx: int) -> str:  
    vt, ur, tm = total_vram(hw), usable_ram(hw), total_mem(hw)  
    tc   = _tcolor(p["tps"])  
    tot  = p["model_gb"] + kv_gb  
    ok   = p["fits_total"] and tot <= tm  
    okc  = "#00ffa2" if ok else "#ff5f5f"  
    t10  = f"{p['time_10k_s']:.0f}s ({p['time_10k_s']/60:.1f}m)" if p.get("time_10k_s") else "N/A"  
    cre  = "Good for creative" if p.get("creative_ok") else "Quality marginal for creative"  
    layers, hidden, attn_heads, kv_heads = arch_for(pb)  
    bpt    = kv_bytes_per_token(pb, kv_bits)  
    avail  = max(tm - p["model_gb"], 0)  
    max_ctx = max_ctx_tokens(pb, avail, kv_bits)  
    kv_in_vram = min(kv_gb, max(vt - p["model_gb"], 0)); kv_in_ram = max(kv_gb - kv_in_vram, 0)  
    return f"""<div style="display:grid;grid-template-columns:repeat(auto-fit,minmax(130px,1fr));gap:.5rem;margin-bottom:.6rem">  
        <div class="stat" style="--sc:{tc}"><div class="sl">Generation</div><div class="sv">{p['tps']}<span class="su"> t/s</span></div><div class="ss">{p['mode']}</div></div>  
        <div class="stat" style="color:#3d9aff"><div class="sl">Prefill</div><div class="sv">{p['prefill_tps']:.0f}<span class="su"> t/s</span></div><div class="ss">prompt processing speed</div></div>  
        <div class="stat" style="color:#b06aff"><div class="sl">Model Size</div><div class="sv">{p['model_gb']}<span class="su"> GB</span></div><div class="ss">eff BW {p['eff_bw']} GB/s</div></div>  
        <div class="stat" style="color:#ff9a3c"><div class="sl">KV Cache ({ctx//1000}K ctx, INT{kv_bits})</div><div class="sv">{kv_gb:.3f}<span class="su"> GB</span></div><div class="ss">{bpt//1024:.1f} KB/token | {kv_heads} KV-heads</div></div>  
        <div class="stat" style="color:{okc}"><div class="sl">Memory</div><div class="sv" style="font-size:1rem;padding-top:.2rem">{"Fits" if ok else "OOM"}</div><div class="ss">total needed {tot:.1f} / {tm:.1f} GB</div></div>  
        <div class="stat" style="color:#00ffa2"><div class="sl">{ctx//1000}K Token Run</div><div class="sv" style="font-size:1rem;padding-top:.2rem">{t10}</div><div class="ss">{cre}</div></div>  
    </div>  
    <div style="display:flex;flex-direction:column;gap:.28rem;margin-bottom:.8rem">  
        <div style="display:flex;align-items:center;gap:.5rem;font-size:.73rem"><span style="min-width:80px;color:var(--text_muted);text-align:right">VRAM</span>{_bar(p['vram_used'],vt,'#3d9aff')}<span style="min-width:90px;color:var(--text_muted)">{p['vram_used']:.1f}/{vt:.0f} GB</span></div>  
        <div style="display:flex;align-items:center;gap:.5rem;font-size:.73rem"><span style="min-width:80px;color:var(--text_muted);text-align:right">RAM spill</span>{_bar(p['ram_used'],ur,'#b06aff')}<span style="min-width:90px;color:var(--text_muted)">{p['ram_used']:.1f}/{ur:.1f} GB usable</span></div>  
        <div style="display:flex;align-items:center;gap:.5rem;font-size:.73rem"><span style="min-width:80px;color:var(--text_muted);text-align:right">KV (VRAM)</span>{_bar(kv_in_vram,vt,'#ff9a3c')}<span style="min-width:90px;color:var(--text_muted)">{kv_in_vram:.3f} GB</span></div>  
        {"<div style='display:flex;align-items:center;gap:.5rem;font-size:.73rem'><span style='min-width:80px;color:var(--text_muted);text-align:right'>KV (RAM)</span>" + _bar(kv_in_ram,ur,'#ff6a00') + f"<span style='min-width:90px;color:#ff9a3c'>{kv_in_ram:.3f} GB spill</span></div>" if kv_in_ram > 0.001 else ""}  
    </div>  
    <div style="display:grid;grid-template-columns:1fr 1fr;gap:.5rem;font-size:.78rem;margin-bottom:.5rem">  
        <div class="glass" style="padding:.55rem .75rem">  
            <div style="color:var(--text_muted);font-size:.65rem;text-transform:uppercase;margin-bottom:.25rem">Architecture (estimated)</div>  
            <div style="font-family:var(--font-mono);line-height:1.8;font-size:.72rem">  
                Layers: {layers} | Hidden: {hidden}<br>  
                Attn-heads: {attn_heads} | KV-heads: {kv_heads} (GQA)<br>  
                Head-dim: {hidden//attn_heads} | KV-bits: INT{kv_bits}<br>  
                {bpt//1024:.1f} KB/token | {int(1024**3/max(bpt,1))//1000:.0f}K tokens/GB  
            </div>  
        </div>  
        <div class="glass" style="padding:.55rem .75rem">  
            <div style="color:var(--text_muted);font-size:.65rem;text-transform:uppercase;margin-bottom:.25rem">Context Limits</div>  
            <div style="font-family:var(--font-mono);line-height:1.8;font-size:.72rem">  
                Free after model: {avail:.1f} GB<br>  
                Max context: <span style="color:var(--accent)">{max_ctx//1000:.0f}K tokens</span><br>  
                At {ctx//1000}K: {kv_gb:.3f} GB KV {"(VRAM)" if kv_in_ram < 0.001 else f"({kv_in_ram:.2f} GB->RAM)"}<br>  
                Quality: {round(p.get("quality_idx",0)*100):.0f}% | {p.get("quant","")}  
            </div>  
        </div>  
    </div>"""

def _compare_html(pb: float, quants_perfs: list) -> str:  
    tbody = ""  
    for q, p in quants_perfs:  
        kv_gb = p.get("kv_gb",0); tot = p["model_gb"] + kv_gb  
        if not p["fits_total"]: tc = mc = cls = "#555"; cls = "oom"  
        else: tc = _tcolor(p["tps"]); mc = "#00ffa2" if p["mem_ok"] else "#ffcc00"; cls = ""  
        t10  = f"{p['time_10k_s']:.0f}s" if p.get("time_10k_s") else "inf"  
        cre  = "&#x2713;" if p.get("creative_ok") else "--"  
        qi   = QUANTS[q]  
        tbody += (f"""<tr class="{cls}"><td class="qn">{q}</td><td style="font-family:var(--font-mono)">{p["model_gb"]} GB</td><td style="color:{tc};font-weight:700">{p["tps"]}</td><td style="font-family:var(--font-mono)">{kv_gb:.3f}</td><td style="color:{mc};font-family:var(--font-mono)">{tot:.1f}</td><td style="color:{"#3d9aff"}">{round(qi["quality"]*100):.0f}%</td><td style="font-size:.7rem">{qi["bpp"]:.3f}</td><td style="color:{"#00ffa2" if cre=="&#x2713;" else "#777"}">{cre}</td></tr>""")
    return (f"""<h3 style="color:var(--accent);margin:0 0 .5rem;font-size:.9rem">{pb}B - All Quantizations</h3><div class="tbl-scroll"><table class="cmp-table"><thead><tr><th>Quant</th><th>Size</th><th>T/s</th><th>KV GB</th><th>Total GB</th><th>Quality</th><th>BPP</th><th>Creative</th></tr></thead><tbody>{tbody}</tbody></table></div><div class="legend"><span style="color:#00ffa2">&#x25A0;</span> >=8 t/s &nbsp;<span style="color:#ffcc00">&#x25A0;</span> 3-8 &nbsp;<span style="color:#ff8c42">&#x25A0;</span> 0.5-3 &nbsp;<span style="color:#ff4444">&#x25A0;</span> &lt;0.5 / OOM</div>""")












def _sweep_html(data: dict, quants: list) -> str:
    series, sizes, refs = data["series"], data["sizes"], data["refs"]
    all_tps = [pt["tps"] for s in series.values() for pt in s if pt["tps"] > 0]
    y_max   = max(all_tps) * 1.2 if all_tps else 20.0
    W, H    = 700, 300; pl, pr, pt_, pb = 52, 90, 15, 48; cw, ch = W-pl-pr, H-pt_-pb
    QCOLS   = {"Q2_K":"#ff4444","Q3_K_S":"#ff6b35","Q3_K_M":"#ff8c42","Q3_K_L":"#ffa05a",
                "Q4_0":"#ffd166","Q4_K_S":"#c8e08a","Q4_K_M":"#06d6a0","Q4_K_L":"#00c4a0",
                "Q5_0":"#00b8cc","Q5_K_S":"#00b4d8","Q5_K_M":"#0096c7","Q6_K":"#3d9aff",
                "Q8_0":"#b06aff","FP16":"#ff79c6","BF16":"#ff79c6"}
    sx = lambda i: pl + (i/(len(sizes)-1)) * cw
    sy = lambda v: pt_ + ch - (min(v,y_max)/y_max) * ch
    parts = [f'<svg viewBox="0 0 {W} {H}" xmlns="http://www.w3.org/2000/svg" style="width:100%;max-height:340px;overflow:visible;display:block">',
             f'<rect width="{W}" height="{H}" fill="var(--bg,#0a0d14)" rx="6"/>']
    for i in range(6):
        v = y_max * i / 5; y = sy(v)
        parts += [f'<line x1="{pl}" y1="{y:.1f}" x2="{pl+cw}" y2="{y:.1f}" stroke="var(--border,#222)" stroke-width="0.5"/>',
                  f'<text x="{pl-5}" y="{y+4:.1f}" text-anchor="end" font-size="9" fill="#777">{v:.1f}</text>']
    for ref in refs:
        if ref["tps"] <= y_max:
            y = sy(ref["tps"])
            lbl = ref["label"].split("~")[1].split(" t/s")[0].strip() if "~" in ref["label"] else ref["label"][:15]
            parts += [f'<line x1="{pl}" y1="{y:.1f}" x2="{pl+cw}" y2="{y:.1f}" stroke="{ref["color"]}" stroke-width="1.2" stroke-dasharray="5,4" opacity="0.75"/>',
                      f'<text x="{pl+cw+4}" y="{y+4:.1f}" font-size="8" fill="{ref["color"]}">{lbl}</text>']
    for i, s in enumerate(sizes):
        parts.append(f'<text x="{sx(i):.1f}" y="{pt_+ch+16}" text-anchor="middle" font-size="9" fill="#777">{s}B</text>')
    parts += [f'<text x="{pl+cw//2}" y="{H-2}" text-anchor="middle" font-size="10" fill="#777">Parameters (B)</text>',
              f'<text x="10" y="{pt_+ch//2}" text-anchor="middle" font-size="10" fill="#777" transform="rotate(-90 10 {pt_+ch//2})">T/s</text>']
    for q in quants:
        pts = series.get(q,[]); color = QCOLS.get(q,"#aaa")
        valid = [(i,p) for i,p in enumerate(pts) if p["tps"] > 0]
        if len(valid) >= 2:
            ld = f"M{sx(valid[0][0]):.1f},{sy(valid[0][1]['tps']):.1f}" + "".join(f" L{sx(i):.1f},{sy(p['tps']):.1f}" for i,p in valid[1:])
            d  = ld + f" L{sx(valid[-1][0]):.1f},{pt_+ch} L{sx(valid[0][0]):.1f},{pt_+ch} Z"
            parts += [f'<path d="{d}" fill="{color}" opacity="0.09"/>',
                      f'<path d="{ld}" fill="none" stroke="{color}" stroke-width="2" stroke-linejoin="round"/>']
        for i, p in enumerate(pts):
            x = sx(i)
            if p["tps"] > 0:
                y = sy(p["tps"])
                parts += [f'<circle cx="{x:.1f}" cy="{y:.1f}" r="4" fill="{color}"/>',
                          f'<text x="{x:.1f}" y="{y-7:.1f}" text-anchor="middle" font-size="8" fill="{color}">{p["tps"]:.1f}</text>']
            else:
                parts.append(f'<text x="{x:.1f}" y="{pt_+ch-3:.1f}" text-anchor="middle" font-size="9" fill="#ff4444" opacity="0.5">X</text>')
    parts.append('</svg>')
    leg_q = "".join(f'<span class="li"><span class="ld" style="background:{QCOLS.get(q,"#aaa")}"></span>{q}</span>' for q in quants)
    leg_r = "".join(f'<span class="li"><span class="lr" style="border-color:{r["color"]}"></span>{r["label"].split("(")[0].strip()}</span>' for r in refs)
    return f'<div class="sweep-wrap">{"".join(parts)}<div class="leg-bar">{leg_q}</div><div class="leg-bar" style="margin-top:.2rem">{leg_r}</div></div>'

def _proprietary_html(user_avgs: list) -> str:
    top = sorted([s for s in user_avgs[:5] if s > 0])
    if not top: return ""
    user_rep = top[len(top)//2]
    rows     = "".join(
        f'<tr><td class="pname">{name}</td><td style="font-size:.7rem;color:var(--text_muted)">{prov}</td>'
        f'<td><div class="pbar-bg"><div class="pbar-fill" style="width:{min(intel/95*100,100):.1f}%"></div></div><span class="pval">{intel}</span></td>'
        f'<td style="color:{"#888" if abs(intel-user_rep)<3 else "#00ffa2" if intel>user_rep else "#ff9a3c"};font-family:var(--font-mono);font-size:.72rem">{("+" if intel>user_rep else "")}{intel-user_rep:.0f}</td>'
        f'<td style="color:{"#00ffa2" if cost<=1 else "#ffcc00" if cost<=10 else "#ff8c42"};font-family:var(--font-mono);font-size:.72rem">${cost:.2f}/1M</td>'
        f'<td class="dim tiny">{notes}</td></tr>'
        for name,prov,_,intel,cost,notes in sorted(PROPRIETARY_REFS, key=lambda r: r[3], reverse=True))
    user_bar = min(user_rep/95*100,100)
    user_row = (f'<tr style="border-top:2px solid var(--accent)"><td class="pname" style="color:var(--accent)">&#9651; Your local models</td>'
                f'<td class="dim" style="font-size:.7rem">local/free</td>'
                f'<td><div class="pbar-bg"><div class="pbar-fill" style="width:{user_bar:.1f}%;background:var(--accent)"></div></div><span class="pval" style="color:var(--accent)">{user_rep:.0f}</span></td>'
                f'<td style="color:var(--accent);font-family:var(--font-mono)">--</td>'
                f'<td style="color:#00ffa2;font-family:var(--font-mono)">$0/1M</td><td class="dim tiny">est from benchmarks</td></tr>')
    return (f'<details class="prop-panel"><summary class="prop-sum">&#9651; Proprietary Model Comparison <span class="dim tiny">(click to expand)</span></summary>'
            f'<div class="prop-body"><div class="prop-note dim tiny">Intelligence index ~= MMLU+reasoning composite (95=frontier). Delta vs your top local result. Cost = USD/1M blended tokens. Scores approximate.</div>'
            f'<div class="tbl-scroll"><table class="prop-table"><thead><tr><th>Model</th><th>Provider</th><th>Intelligence (0-95)</th><th>vs Local</th><th>Cost</th><th>Notes</th></tr></thead>'
            f'<tbody>{user_row}{rows}</tbody></table></div></details>')

def _search_html(res: dict) -> str:
    rows    = res["rows"]; s = res["stats"]; weights = res.get("weights", DEFAULT_WEIGHTS)
    total_w = sum(weights.get(d,0) for d in SCORE_DIMS)
    rerank  = ' <span class="dim tiny">(re-ranked)</span>' if s.get("reranked") else ""
    deep    = ' <span class="dim tiny">[deep scan]</span>' if s.get("deep") else ""
    sbar    = (f'<div class="sbar">Fetched <b>{s.get("fetched","?")}</b> | Filtered <b>{s.get("passed","?")}</b> | '
               f'Ranked <b>{s.get("ranked",len(rows))}</b> | OOM <b>{s.get("oom","?")}</b> | '
               f'Slow <b>{s.get("speed","?")}</b> | Bench <b>{s.get("lb_hits","?")}</b> hits{rerank}{deep}'
               f' <span class="dim">&#8594; {Path(s["csv"]).name}</span></div>')
    if not rows: return sbar + '<div class="placeholder">No results. Try wider param range, lower min T/s, or fewer filters.</div>'
    wleg   = '<div class="wleg">' + "".join(f'<span class="wli"><span class="wld" style="background:{SCORE_DIM_COLORS[d]}"></span>{SCORE_DIM_LABELS[d]} <b>w={weights.get(d,0):.0f}</b></span>' for d in SCORE_DIMS) + '</div>'
    prop   = _proprietary_html([r.get("leaderboard_avg",0) for r in rows[:5]])
    RC     = ["#ffd700","#c0c0c0","#cd7f32"]
    cards  = ""
    for i, r in enumerate(rows):
        p = r["perf"]; ss = r["sub_scores"]; tc = _tcolor(p["tps"])
        rc = RC[i] if i < 3 else "var(--border)"
        t10 = f"{p['time_10k_s']:.0f}s ({p['time_10k_s']/60:.1f}m)" if p.get("time_10k_s") else "N/A"
        hfu = f"https://huggingface.co/{r['id']}"
        mbdg = ('<span class="bdg" style="background:#ff44441a;color:#ff4444;border:1px solid #ff444433">OOM</span>' if "OOM" in p["mode"] else
                '<span class="bdg bh">Hybrid</span>' if "RAM" in p["mode"] else '<span class="bdg bv">VRAM</span>')
        lb_v = r.get("leaderboard_avg",0.0); bconf = r.get("bench_confidence",0.0)
        bsrc = r.get("bench_source","none"); binf = r.get("bench_inferred",False)
        if lb_v > 0:
            src_cols = {"static-anchor":"#00ffa2","card-metadata":"#3d9aff","readme-table":"#ffcc00"}
            bc = next((c for k,c in src_cols.items() if bsrc.startswith(k)), "#ff9a3c")
            lb_bdg = f'<span class="bdg bl" style="border-color:{bc}33;background:{bc}1a;color:{bc}">LB {"~" if binf else ""}{lb_v:.1f}{f" {bconf*100:.0f}%" if bconf<0.95 else ""}</span>'
        else:
            lb_bdg = '<span class="bdg" style="opacity:.3;border:1px solid var(--border)">No bench</span>'
        score_bars = "".join(_score_bar(d, ss[d], weights.get(d,0), total_w) for d in SCORE_DIMS)
        avail_qt   = "".join(f'<span class="qt" style="border-color:{"var(--accent)" if q==r["quant"] else "var(--border)"};color:{"var(--accent)" if q==r["quant"] else "var(--text_muted)"};{"font-weight:800;" if q==r["quant"] else ""}">{q}</span>' for q in r.get("avail_quants",[]))
        lb_detail  = _lb_pills(r.get("lb_detail",{}), bsrc, bconf, binf)
        tags_html  = " ".join(f'<span class="tag">{t}</span>' for t in (r.get("tags") or [])[:8] if not any(t.startswith(p2) for p2 in ("arxiv","doi","region","license","language")) and len(t) < 35)
        below      = ' <span class="dim tiny">[below T/s target]</span>' if r.get("below_floor") else ""
        cards += f"""<div class="mcard" style="--rc:{rc}">
            <div class="mrank">#{i+1}</div>
            <div class="mbody">
                <div class="mhead"><a class="mname" href="{hfu}" target="_blank" rel="noopener">{r['name']}</a>
                <span class="mauthor">{r['author']}</span>{lb_bdg}{mbdg}
                <span class="bdg" style="background:var(--accent_dim);color:var(--accent);border:1px solid var(--accent)33">{r['quant']}</span>
                <span class="tscore">{r['total_score']:.4f}</span>{below}</div>
                <div class="mstats">
                    <span class="ms" style="color:{tc}">~{p['tps']} t/s</span>
                    <span class="ms">^{p['prefill_tps']:.0f} t/s pf</span>
                    <span class="ms">{r['params_b']}B</span>
                    <span class="ms">{p['model_gb']} GB</span>
                    <span class="ms">KV {p['kv_gb']} GB</span>
                    <span class="ms">{p['total_mem_gb']} GB total</span>
                    <span class="ms">{t10}</span>
                    <span class="ms" style="color:{"#00ffa2" if p.get("creative_ok") else "#777"}">{"creative OK" if p.get("creative_ok") else "q-marginal"}</span>
                </div>
                <div class="sb-block">{score_bars}</div>
                <details class="mdet"><summary class="mdet-sum">&#9656; Benchmark scores + all quants</summary>
                <div class="mdet-body">
                    <div class="lb-row">{lb_detail}</div>
                    <div class="mqts" style="margin-top:.35rem">{avail_qt}</div>
                    {f'<div class="mtags" style="margin-top:.3rem">{tags_html}</div>' if tags_html else ""}
                    <div class="mpop" style="margin-top:.3rem"><span>&#9829; {r.get("likes",0):,}</span><span>&#8659; {_fmt(r.get("downloads",0) or 0)}</span><a href="{hfu}" target="_blank" class="hfl">HuggingFace &#8594;</a></div>
                </div></details>
            </div>
        </div>"""
    return f'{sbar}{prop}{wleg}<div class="clist">{cards}</div>'

def _ctx_breakdown_html(pb: float, q: str, hw: dict, kv_bits: int) -> str:
    mgb  = model_gb(pb, q); vt, ur, tm = total_vram(hw), usable_ram(hw), total_mem(hw)
    if mgb > tm: return f'<div class="err-box">Model ({mgb:.1f} GB) exceeds total memory ({tm:.1f} GB). OOM.</div>'
    vram_used = min(mgb, vt); ram_used = max(mgb-vt,0); avail = max(tm-mgb,0)
    avail_vram = max(vt-vram_used,0)
    bpt        = kv_bytes_per_token(pb, kv_bits)
    layers, hidden, attn, kv_heads = arch_for(pb); head_dim = hidden//attn
    m_pct  = min(vram_used/max(vt,0.001)*100,100); rm_pct = min(ram_used/max(vt,0.001)*100,100)
    ctx_rows = ""
    for ctx in [4096,8192,16384,32768,65536,100000,131072]:
        kv_gb = kv_bytes_per_token(pb,kv_bits)*ctx/(1024**3); total_req = mgb+kv_gb
        kv_in_vram = min(kv_gb,avail_vram); kv_in_ram = max(kv_gb-kv_in_vram,0)
        if total_req > tm: ctx_rows += f'<tr><td style="font-family:var(--font-mono)">{ctx//1000}K</td><td style="color:#ff5f5f">OOM ({total_req:.1f}/{tm:.1f} GB)</td><td>-</td><td>-</td></tr>'
        elif kv_in_ram > 0: ctx_rows += f'<tr><td style="font-family:var(--font-mono)">{ctx//1000}K</td><td style="color:#ffcc00">KV: {kv_gb:.3f} GB</td><td style="color:#ff9a3c">{kv_in_ram:.3f} GB->RAM</td><td style="color:#ffcc00">hybrid</td></tr>'
        else: ctx_rows += f'<tr><td style="font-family:var(--font-mono)">{ctx//1000}K</td><td style="color:#00ffa2">KV: {kv_gb:.3f} GB</td><td style="color:var(--text_muted)">-</td><td style="color:#00ffa2">VRAM</td></tr>'
    return f"""<div style="display:flex;flex-direction:column;gap:.8rem;padding:.3rem 0">
        <div>
            <div style="font-size:.68rem;color:var(--text_muted);margin-bottom:.25rem">Memory layout ({q}, INT{kv_bits} KV)</div>
            <div style="display:flex;height:16px;border-radius:4px;overflow:hidden;border:1px solid var(--border)">
                <div style="width:{m_pct:.1f}%;background:#3d9aff" title="Model in VRAM {vram_used:.1f} GB"></div>
                <div style="width:{rm_pct:.1f}%;background:#b06aff" title="Model in RAM {ram_used:.1f} GB"></div>
                <div style="flex:1;background:var(--bg)"></div>
            </div>
            <div style="display:flex;gap:.8rem;margin-top:.2rem;font-size:.65rem;flex-wrap:wrap">
                <span style="color:#3d9aff">Model VRAM {vram_used:.1f} GB</span>
                {"<span style='color:#b06aff'>Model RAM " + f"{ram_used:.1f} GB</span>" if ram_used > 0 else ""}
                <span style="color:var(--text_muted)">Free {avail:.2f} GB ({avail/max(tm,0.001)*100:.0f}%)</span>
            </div>
        </div>
        <div style="display:grid;grid-template-columns:1fr 1fr;gap:.4rem;font-size:.78rem">
            <div class="glass" style="padding:.5rem .7rem"><div style="color:var(--text_muted);font-size:.63rem;text-transform:uppercase;margin-bottom:.2rem">Architecture</div>
                <div style="font-family:var(--font-mono);line-height:1.75;font-size:.7rem">Layers: {layers} | Hidden: {hidden}<br>Attn: {attn} heads | KV: {kv_heads} heads (GQA)<br>Head-dim: {head_dim}</div></div>
            <div class="glass" style="padding:.5rem .7rem"><div style="color:var(--text_muted);font-size:.63rem;text-transform:uppercase;margin-bottom:.2rem">KV Cache Rate</div>
                <div style="font-family:var(--font-mono);line-height:1.75;font-size:.7rem">{bpt//1024:.1f} KB/token (INT{kv_bits})<br>{int(1024**3/max(bpt,1))//1000:.0f}K tokens/GB<br>Max: <span style="color:var(--accent)">{max_ctx_tokens(pb,avail,kv_bits)//1000:.0f}K</span> in {avail:.1f} GB</div></div>
        </div>
        <div>
            <div style="font-size:.68rem;color:var(--text_muted);margin-bottom:.2rem">Context vs memory</div>
            <div class="tbl-scroll"><table class="cmp-table" style="font-size:.75rem">
                <thead><tr><th>Context</th><th>KV Total</th><th>KV->RAM</th><th>Location</th></tr></thead>
                <tbody>{ctx_rows}</tbody>
            </table></div>
        </div>
    </div>"""

# right_panel() sub-module contract

def right_panel() -> str:
    hw = DEFAULT_HW; vt = total_vram(hw); ur = usable_ram(hw)
    return f"""<div class="ait-rp">
        <div class="ait-rp-hd">AI Calc</div>
        <button class="ait-rp-btn" hx-get="{_P}/view/session"    hx-target="#ait-workspace" hx-swap="innerHTML">&#x25CF; Session Monitor</button>
        <button class="ait-rp-btn" hx-get="{_P}/view/calc"       hx-target="#ait-workspace" hx-swap="innerHTML">&#x223C; Speed Calc</button>
        <button class="ait-rp-btn" hx-get="{_P}/view/ctx"        hx-target="#ait-workspace" hx-swap="innerHTML">&#x25A4; Context Planner</button>
        <button class="ait-rp-btn" hx-get="{_P}/view/compare"    hx-target="#ait-workspace" hx-swap="innerHTML">&#x2261; Quant Compare</button>
        <button class="ait-rp-btn" hx-get="{_P}/view/sweep"      hx-target="#ait-workspace" hx-swap="innerHTML">&#x2227; Sweep Chart</button>
        <button class="ait-rp-btn" hx-get="{_P}/view/search"     hx-target="#ait-workspace" hx-swap="innerHTML">&#x1F50D; Model Finder</button>
        <button class="ait-rp-btn" hx-get="{_P}/view/image"      hx-target="#ait-workspace" hx-swap="innerHTML">&#x1F5BC; Image Models</button>
        <button class="ait-rp-btn" hx-get="{_P}/view/hwprofiles" hx-target="#ait-workspace" hx-swap="innerHTML">&#x1F9E0; HW Profiles</button>
        <button class="ait-rp-btn" hx-get="{_P}/view/files"      hx-target="#ait-workspace" hx-swap="innerHTML">&#x1F4C2; Cache Files</button>
        <div class="ait-rp-hd" style="margin-top:.6rem">Default HW</div>
        <div style="font-size:.68rem;color:var(--text_muted);font-family:var(--font-mono);line-height:1.8;padding:0 .2rem">
            VRAM {vt:.0f} GB ({hw['vram_gb']:.0f}+{hw['shared_gb']:.0f}sh)<br>
            RAM  {ur:.1f} GB usable<br>
            Pool <span style="color:var(--accent)">{vt+ur:.1f} GB</span><br>
            BW   {hw['mem_bw_gbps']:.0f} GB/s | {hw['gpu_tflops_fp16']:.1f} TFLOPS
        </div>
    </div>"""

# Routes

@router.get("/", response_class=HTMLResponse)
async def root(): return HTMLResponse(_panel_session())

@router.get("/view/session", response_class=HTMLResponse)
async def view_session(): return HTMLResponse(_panel_session())

@router.get("/view/calc", response_class=HTMLResponse)
async def view_calc(): return HTMLResponse(_panel_calc(DEFAULT_HW))
    
@router.get("/view/ctx", response_class=HTMLResponse)
async def view_ctx(): return HTMLResponse(_panel_ctx(DEFAULT_HW))
    
@router.get("/view/compare", response_class=HTMLResponse)
async def view_compare(): return HTMLResponse(_panel_compare(DEFAULT_HW))
    
@router.get("/view/sweep", response_class=HTMLResponse)
async def view_sweep(): return HTMLResponse(_panel_sweep(DEFAULT_HW))
    
@router.get("/view/search", response_class=HTMLResponse)
async def view_search(): return HTMLResponse(_panel_search())
    
@router.get("/view/image", response_class=HTMLResponse)
async def view_image(): return HTMLResponse(_panel_image(DEFAULT_HW))
    
@router.get("/view/hwprofiles", response_class=HTMLResponse)
async def view_hwprofiles(): return HTMLResponse(_panel_hw_profiles())
    
@router.get("/view/files", response_class=HTMLResponse)
async def view_files(): return HTMLResponse(_panel_files())

@router.post("/calc", response_class=HTMLResponse)
async def calc_route(request: Request):
    f  = await request.form()
    hw = _hw_from_form(f)
    pb = max(0.1, min(float(f.get("params_b",7)), 500))
    q  = f.get("quant","Q4_K_M") if f.get("quant") in QUANTS else "Q4_K_M"
    ctx    = max(512, int(f.get("ctx_tokens",20992)))
    kv_bits = int(f.get("kv_bits",8))
    p    = full_perf(pb, q, ctx, hw, kv_bits)
    kv_gb = kv_cache_gb_for_ctx(pb, ctx, kv_bits)
    return HTMLResponse(_calc_html(p, hw, kv_gb, kv_bits, pb, ctx))

@router.post("/ctx_breakdown", response_class=HTMLResponse)
async def ctx_breakdown_route(request: Request):
    f  = await request.form()
    hw = _hw_from_form(f)
    pb = max(0.1, min(float(f.get("params_b",7)), 500))
    q  = f.get("quant","Q4_K_M") if f.get("quant") in QUANTS else "Q4_K_M"
    kv_bits = int(f.get("kv_bits",8))
    mgb   = model_gb(pb, q); avail = max(total_mem(hw) - mgb, 0)
    max_ctx = max_ctx_tokens(pb, avail, kv_bits)
    result = _ctx_breakdown_html(pb, q, hw, kv_bits)
    script = f"<script>window._ctxBudget={{maxCtx:{max_ctx},modelGb:{mgb:.2f},availGb:{avail:.2f}}};updateCtxBudget();</script>"
    return HTMLResponse(result + script)

@router.post("/compare", response_class=HTMLResponse)
async def compare_route(request: Request):
    f  = await request.form()
    hw = _hw_from_form(f)
    pb = max(0.1, min(float(f.get("params_b",14)), 500))
    ctx = max(512, int(f.get("ctx_tokens",20992)))
    qp  = [(q, full_perf(pb, q, ctx, hw)) for q in QUANTS]
    return HTMLResponse(_compare_html(pb, qp))

@router.post("/sweep", response_class=HTMLResponse)
async def sweep_route(request: Request):
    f      = await request.form()
    hw     = _hw_from_form(f)
    quants = [q for q in f.getlist("quants") if q in QUANTS] or ["Q4_K_M"]
    return HTMLResponse(_sweep_html(sweep_data(quants, hw), quants))

@router.post("/image_calc", response_class=HTMLResponse)
async def image_calc_route(request: Request):
    f     = await request.form()
    hw    = _hw_from_form(f)
    model = f.get("image_model","SDXL")
    steps = max(1, min(int(f.get("steps",20)), 500))
    r     = image_estimate(model, hw, steps)
    if not r["feasible"]: return HTMLResponse(f'<div class="err-box">Cannot run {model}: {r["note"]}</div>')
    tc  = "#00ffa2" if r["its"] >= 1.5 else "#ffcc00" if r["its"] >= 0.5 else "#ff8c42"
    vok = "VRAM only" if r["fits_vram"] else "RAM offload (may be significantly slower)"
    return HTMLResponse(f"""<div style="display:grid;grid-template-columns:repeat(auto-fit,minmax(130px,1fr));gap:.5rem">
        <div class="stat" style="--sc:{tc}"><div class="sl">Speed</div><div class="sv">{r['its']}<span class="su"> it/s</span></div><div class="ss">{steps} steps</div></div>
        <div class="stat" style="--sc:#3d9aff"><div class="sl">Gen Time</div><div class="sv">{r['total_s']}<span class="su"> s</span></div><div class="ss">{r['total_s']/60:.1f} min</div></div>
        <div class="stat" style="--sc:#b06aff"><div class="sl">VRAM Min</div><div class="sv">{r['vram_min']}<span class="su"> GB</span></div><div class="ss">{vok}</div></div>
    </div><div style="font-size:.73rem;color:var(--text_muted);margin-top:.4rem">{model}: {r['note']}</div>""")

@router.post("/search", response_class=HTMLResponse)
async def search_route(request: Request):
    f = await request.form()
    with _job_lock:
        if _job["running"]: return HTMLResponse('<div class="sbar"><span>Job already running. Wait or reload.</span></div>')
    hw = _hw_from_form(f)
    params = {
        "task_preset":   f.get("task_preset","instruct"),
        "min_params":    max(0.1, float(f.get("min_params",1))),
        "max_params":    min(float(f.get("max_params",35)), total_mem(hw) / QUANTS["Q2_K"]["bpp"]),
        "ctx_tokens":    max(512, int(f.get("ctx_tokens",20992))),
        "min_tps":       max(0.0, float(f.get("min_tps",1.0))),
        "target_tps":    max(0.1, float(f.get("target_tps",5.0))),
        "extra_query":   f.get("extra_query",""),
        "must_contain":  f.get("must_contain",""),
        "any_contain":   f.get("any_contain",""),
        "exclude":       f.get("exclude",""),
        "hw":            hw, "weights": _parse_weights(f),
        "force_refresh": f.get("force_refresh") == "1",
        "deep_scan":     f.get("deep_scan") == "1",
        "top_n":         max(5, min(int(f.get("top_n",40)), 200)),
        "fetched":       0,
    }
    _job_reset()
    _job_up(running=True, status="Starting...", params=params)
    threading.Thread(target=_run_thread, args=(params,), daemon=True).start()
    return HTMLResponse(f'<div id="srch-status" hx-get="{_P}/search-status" hx-trigger="every 2s" hx-target="#srch-out" hx-swap="innerHTML"><div class="status-bar"><span class="spin-anim">[  ]</span> Starting pipeline...</div></div>')

@router.get("/search-status", response_class=HTMLResponse)
async def search_status():
    with _job_lock:
        running = _job["running"]; status = _job["status"]
        progress = list(_job["progress"]); result = _job["result"]; error = _job["error"]
    if error:
        return HTMLResponse(f'<div class="err-box">Pipeline failed:<br><pre style="font-size:.68rem;white-space:pre-wrap;margin-top:.4rem">{error[:1500]}</pre></div>')
    if result is not None and not running:
        return HTMLResponse(_search_html(result))
    log = "".join(f'<div class="log-line">{l}</div>' for l in progress[-25:])
    return HTMLResponse(f'<div id="srch-status" hx-get="{_P}/search-status" hx-trigger="every 2s" hx-target="#srch-out" hx-swap="innerHTML"><div class="status-bar"><span class="spin-anim">[  ]</span> {status}</div><div class="log-wrap">{log}</div></div>')

@router.post("/rerank", response_class=HTMLResponse)
async def rerank_route(request: Request):
    f = await request.form()
    with _job_lock:
        filtered = _job.get("filtered"); params = _job.get("params"); running = _job.get("running")
    if running:   return HTMLResponse('<div class="err-box">Search still running.</div>')
    if not filtered or not params: return HTMLResponse('<div class="err-box">No cached results. Run a search first.</div>')
    target_tps = max(0.1, float(f.get("target_tps", params.get("target_tps",5.0))))
    weights    = _parse_weights(f)
    top_n      = max(5, min(int(f.get("top_n",40)), 200))
    rows       = _rank_filtered(filtered, params["hw"], params["ctx_tokens"], target_tps, weights, top_n)
    key        = re.sub(r'[^\w]','_',f"{params['task_preset']}_rerank")[:50]
    _write_csv(rows, key)
    stats      = {**{k:params.get("fetched",0) if k=="fetched" else _job.get("stats",{}).get(k,"?") for k in ("fetched","passed","oom","speed","params","quant","tags","exclude","lb_hits")},
                  "passed":len(filtered),"ranked":len(rows),"csv":f"data/ai_tools/analysis_{key}.csv","reranked":True}
    return HTMLResponse(_search_html({"rows":rows,"stats":stats,"weights":weights}))

@router.get("/session_poll", response_class=HTMLResponse)
async def session_poll():
    conn = await _get_ollama_conn()
    if not conn: return HTMLResponse('<div class="placeholder">No Ollama connection configured. Add one in Settings.</div>')
    loaded = await _fetch_running(conn)
    if not loaded: return HTMLResponse('<div style="color:var(--text_muted);font-size:.82rem;padding:.5rem">No models loaded in Ollama.</div>')
    alt_profiles = _load_hw_profiles()
    cards = "".join(_session_card_html(m, DEFAULT_HW, alt_profiles) for m in loaded)
    return HTMLResponse(f'<div>{cards}<div style="font-size:.65rem;color:var(--text_muted);margin-top:.3rem">Updates every 5s | Alt hardware requires saved HW Profiles</div></div>')

@router.post("/hw_profiles/save", response_class=HTMLResponse)
async def hw_profiles_save(request: Request):
    f = await request.form(); profiles = _load_hw_profiles(); name = f.get("name","").strip()
    if not name: return HTMLResponse(_panel_hw_profiles())
    hw_keys = ["vram_gb","shared_gb","sys_ram_gb","os_overhead_gb","mem_bw_gbps","gpu_tflops_fp16"]
    hw      = {k: float(f.get(k, DEFAULT_HW.get(k,0))) for k in hw_keys}
    profiles.append({"id":f"hwp_{uuid.uuid4().hex[:6]}","name":name,"hw":hw})
    _save_hw_profiles(profiles)
    return HTMLResponse(_panel_hw_profiles())

@router.delete("/hw_profiles/{pid}", response_class=HTMLResponse)
async def hw_profiles_delete(pid: str):
    _save_hw_profiles([p for p in _load_hw_profiles() if p["id"] != pid])
    return HTMLResponse(_panel_hw_profiles())

@router.post("/ctx_profile_save", response_class=HTMLResponse)
async def ctx_profile_save(request: Request):
    body = await request.json()
    pid  = f"cp_{uuid.uuid4().hex[:8]}"; name = body.get("name","Untitled")
    prof_dir = DATA_DIR / "ctx_profiles"; prof_dir.mkdir(exist_ok=True)
    (prof_dir / f"{pid}.json").write_text(json.dumps({"id":pid,"name":name,"blocks":body.get("blocks",[]),"created":datetime.utcnow().isoformat()},indent=2))
    return JSONResponse({"status":"ok","id":pid,"name":name})

# --- Quick estimate API (for other modules to call) ---

@router.get("/api/model_estimate", response_class=JSONResponse)
async def api_model_estimate(model_name: str = "", conn_id: str = ""):
    """
    Returns model size estimate, available memory, and recommended num_ctx.
    Called by chat module to set smart context defaults.
    Priority: Ollama model info -> name parsing -> defaults.
    """
    hw       = DEFAULT_HW.copy()
    # Load saved HW profile if exists (first one found)
    profiles = _load_hw_profiles()
    if profiles: hw.update(profiles[0].get("hw", {}))

    # Get RAM available from OS
    sys_ram = psutil.virtual_memory().available / (1024**3)
    hw["sys_ram_gb"] = max(sys_ram, hw["sys_ram_gb"])

    vt, ur = total_vram(hw), usable_ram(hw)
    pool   = total_mem(hw)

    model_gb_est = None; params_b_est = None; quant_est = "Q4_K_M"
    arch_info    = {}; ollama_ctx = None; model_size_bytes = None

    # Try Ollama /api/show for real model info
    if model_name and conn_id:
        conn = None
        for f in sorted((DATA_DIR / "_connections").glob("*.json")):
            try:
                c = json.loads(f.read_text())
                if f.stem == conn_id or c.get("_id") == conn_id:
                    c["_id"] = f.stem; conn = c; break
            except Exception: pass
        if conn:
            try:
                async with httpx.AsyncClient(timeout=5.0) as c:
                    r = await c.post(f"{_ollama_base(conn)}/api/show", json={"name": model_name})
                    if r.status_code == 200:
                        data = r.json()
                        # Extract num_ctx from parameters section
                        params_text = data.get("parameters","")
                        ctx_match = re.search(r'num_ctx\s+(\d+)', str(params_text))
                        if ctx_match: ollama_ctx = int(ctx_match.group(1))
                        # Size from details
                        details = data.get("details", {})
                        model_size_bytes = data.get("size", 0)
                        if model_size_bytes: model_gb_est = round(model_size_bytes / 1e9, 2)
                        # Parse quant from quantization_level
                        ql = details.get("quantization_level","")
                        for q in QUANTS:
                            if q.lower() in ql.lower().replace("-","_"): quant_est = q; break
            except Exception: pass

    # Fallback: parse from name
    if params_b_est is None and model_name:
        params_b_est = parse_params_b(model_name)
    if model_gb_est is None and params_b_est:
        model_gb_est = round(model_gb(params_b_est, quant_est), 2)

    # Architecture info for context math
    if params_b_est:
        layers, hidden, attn, kv_heads = arch_for(params_b_est)
        bpt     = kv_bytes_per_token(params_b_est)
        avail   = max(pool - model_gb_est, 0) if model_gb_est else pool * 0.5
        max_ctx = max_ctx_tokens(params_b_est, avail) if params_b_est else 4096
        arch_info = {"layers":layers,"hidden":hidden,"kv_heads":kv_heads, "bpt_bytes":bpt,"bpt_kb":round(bpt/1024,1), "tokens_per_gb":int(1024**3/max(bpt,1))//1000}
    else:
        avail   = pool * 0.4
        max_ctx = 8192

    # Recommended ctx: max_ctx * 0.75 as comfortable target, capped to ollama reported
    recommended = int(min(max_ctx * 0.75, ollama_ctx or max_ctx))
    recommended = max(2048, min(recommended, 131072))

    warning = None
    if model_gb_est and model_gb_est > pool:  warning = f"Model ({model_gb_est:.1f}GB) exceeds total memory ({pool:.1f}GB)"
    elif model_gb_est and model_gb_est > vt:  warning = f"Model ({model_gb_est:.1f}GB) will spill to RAM (VRAM={vt:.0f}GB)"

    return JSONResponse({
        "model_name":     model_name,
        "model_gb":       model_gb_est,
        "params_b":       params_b_est,
        "quant":          quant_est,
        "pool_gb":        round(pool, 2),
        "vram_gb":        round(vt, 2),
        "ram_gb":         round(ur, 2),
        "avail_gb":       round(avail, 2),
        "max_ctx":        max_ctx,
        "recommended_ctx":recommended,
        "ollama_ctx":     ollama_ctx,
        "arch":           arch_info,
        "warning":        warning,
        "hw_source":      "hw_profile" if profiles else "default",
    })

@router.get("/api/memory_status", response_class=JSONResponse)
async def api_memory_status():
    """Current memory pool status - useful for live display."""
    hw = DEFAULT_HW.copy()
    profiles = _load_hw_profiles()
    if profiles: hw.update(profiles[0].get("hw", {}))
    vm = psutil.virtual_memory()
    return JSONResponse({"vram_gb": round(total_vram(hw), 2), "ram_available_gb": round(vm.available/(1024**3), 2), "ram_total_gb": round(vm.total/(1024**3), 2), "pool_gb": round(total_mem(hw), 2)})

# CSS

CSS = """:root{--c-fast:#00ffa2;--c-ok:#ffcc00;--c-slow:#ff8c42;--c-bad:#ff4444;}
.hwp{background:var(--surface);border:1px solid var(--border);border-radius:7px;padding:.5rem .75rem;margin-bottom:.7rem;font-size:.8rem;}
.hwp summary{cursor:pointer;color:var(--text_muted);list-style:none;user-select:none;}
.hwp summary::-webkit-details-marker{display:none;}
.hwg{display:flex;flex-wrap:wrap;gap:.5rem;margin-top:.6rem;}
.hwg label{display:flex;flex-direction:column;gap:.15rem;font-size:.73rem;color:var(--text_muted);min-width:120px;flex:1;}
.hwin{background:var(--bg);border:1px solid var(--border);color:var(--text);border-radius:4px;padding:.27rem .4rem;font-family:var(--font-mono);font-size:.8rem;width:100%;box-sizing:border-box;}
.pform{background:var(--surface);border:1px solid var(--border);border-radius:6px;padding:.8rem;margin-bottom:.5rem;}
.frow{display:flex;gap:.5rem;align-items:flex-start;flex-wrap:nowrap;margin-bottom:.55rem;}
.frow.wrap{flex-wrap:wrap;}
.frow label{display:flex;flex-direction:column;gap:.15rem;font-size:.73rem;color:var(--text_muted);flex:1;min-width:100px;}
.frow label:has(input[type=checkbox]){flex-direction:row;flex:0;align-items:center;}
.fin{background:var(--bg);border:1px solid var(--border);color:var(--text);border-radius:5px;padding:.32rem .48rem;font-family:var(--font-mono);font-size:.8rem;width:100%;box-sizing:border-box;}
.fin:focus{outline:none;border-color:var(--accent);}
.fin option{background:var(--bg);color:var(--text);}
.rbtn{background:var(--accent_dim);color:var(--accent);border:1px solid var(--accent);border-radius:6px;padding:.4rem .85rem;font-size:.82rem;font-weight:700;cursor:pointer;white-space:nowrap;flex-shrink:0;transition:all .12s;}
.rbtn:hover{background:var(--accent);color:#000;}
.rzone{margin-top:.7rem;}
.placeholder{color:var(--text_muted);font-size:.84rem;padding:1.5rem 0;text-align:center;}
.fsect-hd{font-size:.73rem;font-weight:700;color:var(--text_muted);text-transform:uppercase;letter-spacing:.05em;margin-bottom:.5rem;}
.dim{color:var(--text_muted);}.tiny{font-size:.7rem;}
.spin{display:none;}.htmx-request .spin{display:inline;}
.stat{background:var(--bg);border:1px solid var(--border);border-top:2px solid var(--sc);border-radius:6px;padding:.55rem .7rem;}
.sl{font-size:.65rem;color:var(--text_muted);text-transform:uppercase;letter-spacing:.04em;}
.sv{font-size:1.35rem;font-weight:800;color:var(--sc);line-height:1.1;}.su{font-size:.7rem;font-weight:400;}
.ss{font-size:.66rem;color:var(--text_muted);margin-top:.15rem;}
.bar-bg{flex:1;height:8px;background:var(--bg);border:1px solid var(--border);border-radius:4px;overflow:hidden;}
.bar-fill{height:100%;border-radius:4px;min-width:2px;}
.cmp-table{width:100%;border-collapse:collapse;font-size:.8rem;white-space:nowrap;}
.cmp-table th{padding:.3rem .6rem;text-align:left;border-bottom:1px solid var(--border);color:var(--text_muted);font-weight:600;}
.cmp-table td{padding:.28rem .6rem;border-bottom:1px solid var(--border);}
.cmp-table tr.oom td{opacity:.35;}
.cmp-table tr:not(.oom):hover td{background:var(--accent_dim);}
.qn{font-family:var(--font-mono);font-weight:700;color:var(--accent);}
.legend{font-size:.72rem;margin-top:.5rem;color:var(--text_muted);}
.sweep-wrap{width:100%;}
.leg-bar{display:flex;flex-wrap:wrap;gap:.5rem;font-size:.72rem;margin-top:.4rem;}
.li{display:flex;align-items:center;gap:.25rem;color:var(--text_muted);}
.ld{width:10px;height:10px;border-radius:50%;flex-shrink:0;}
.lr{width:16px;height:0;border-top:2px dashed;flex-shrink:0;}
.sbar{display:flex;flex-wrap:wrap;gap:.3rem .7rem;font-size:.72rem;color:var(--text_muted);margin-bottom:.5rem;padding:.4rem .6rem;background:var(--bg);border:1px solid var(--border);border-radius:5px;}
.sbar b{color:var(--text);}
.wsblock{display:flex;flex-direction:column;gap:.28rem;padding:.3rem 0;}
.wsl{display:flex;align-items:center;gap:.55rem;font-size:.75rem;color:var(--text_muted);}
.wsln{min-width:130px;color:var(--wc);font-weight:600;flex-shrink:0;}
.wslider{flex:1;accent-color:var(--wc);cursor:pointer;}
.wsv{min-width:1.5rem;text-align:right;font-family:var(--font-mono);font-size:.8rem;color:var(--wc);font-weight:700;}
.wleg{display:flex;flex-wrap:wrap;gap:.3rem .65rem;margin:.45rem 0 .5rem;font-size:.7rem;color:var(--text_muted);}
.wli{display:flex;align-items:center;gap:.22rem;}.wld{width:7px;height:7px;border-radius:50%;flex-shrink:0;}
.sb-block{display:flex;flex-direction:column;gap:.15rem;margin:.28rem 0;}
.sbrow{display:flex;align-items:center;gap:.4rem;font-size:.7rem;}
.sbl{min-width:110px;color:var(--text_muted);font-size:.67rem;flex-shrink:0;}
.sbbar{flex:1;height:6px;background:var(--bg);border-radius:3px;overflow:hidden;border:1px solid var(--border);}
.sbfill{height:100%;border-radius:3px;min-width:1px;}
.sbv{min-width:3.2rem;text-align:right;font-family:var(--font-mono);color:var(--text);flex-shrink:0;}
.sbw{min-width:2.2rem;color:var(--text_muted);font-size:.65rem;flex-shrink:0;}
.sbc{min-width:3.8rem;text-align:right;font-family:var(--font-mono);font-size:.67rem;flex-shrink:0;}
.clist{display:flex;flex-direction:column;gap:.5rem;}
.mcard{background:var(--bg);border:1px solid var(--border);border-left:3px solid var(--rc);border-radius:6px;padding:.6rem .75rem;display:flex;gap:.5rem;}
.mrank{font-size:.7rem;font-weight:800;color:var(--rc);min-width:1.8rem;text-align:right;padding-top:.1rem;flex-shrink:0;}
.mbody{flex:1;min-width:0;}
.mhead{display:flex;align-items:baseline;flex-wrap:wrap;gap:.28rem;margin-bottom:.28rem;}
.mname{font-weight:700;font-size:.88rem;color:var(--accent);text-decoration:none;word-break:break-all;}
.mname:hover{text-decoration:underline;}
.mauthor{font-size:.68rem;color:var(--text_muted);}
.tscore{font-size:.8rem;font-weight:800;color:var(--text);font-family:var(--font-mono);margin-left:auto;}
.mstats{display:flex;flex-wrap:wrap;gap:.25rem;font-size:.73rem;margin-bottom:.22rem;}
.ms{background:var(--surface);border:1px solid var(--border);padding:.1rem .35rem;border-radius:4px;}
.mqts{display:flex;flex-wrap:wrap;gap:.2rem;}
.qt{font-size:.65rem;padding:.08rem .32rem;border-radius:3px;border:1px solid;font-family:var(--font-mono);}
.mtags{display:flex;flex-wrap:wrap;gap:.15rem;}
.tag{font-size:.63rem;padding:.08rem .3rem;border-radius:3px;background:var(--surface);border:1px solid var(--border);color:var(--text_muted);}
.mpop{display:flex;align-items:center;gap:.6rem;font-size:.7rem;color:var(--text_muted);flex-wrap:wrap;}
.hfl{color:var(--accent);text-decoration:none;margin-left:auto;font-size:.73rem;}
.hfl:hover{text-decoration:underline;}
.bdg{font-size:.63rem;padding:.08rem .32rem;border-radius:3px;font-weight:700;flex-shrink:0;}
.bv{background:#3d9aff1a;color:#3d9aff;border:1px solid #3d9aff33;}
.bh{background:#b06aff1a;color:#b06aff;border:1px solid #b06aff33;}
.bl{color:#00ffa2;}
.mdet summary{cursor:pointer;font-size:.7rem;color:var(--text_muted);list-style:none;user-select:none;}
.mdet summary::-webkit-details-marker{display:none;}
.mdet summary:hover{color:var(--accent);}
.mdet-body{padding:.45rem 0 .15rem;border-top:1px solid var(--border);margin-top:.28rem;}
.lb-row{display:flex;flex-wrap:wrap;gap:.28rem;}
.lbp{font-size:.68rem;padding:.08rem .38rem;border-radius:3px;background:var(--surface);border:1px solid var(--border);font-family:var(--font-mono);}
.err-box{color:#ff4444;background:#ff44441a;border:1px solid #ff444433;border-radius:6px;padding:.6rem .8rem;font-size:.82rem;}
.prop-panel{background:var(--surface);border:1px solid var(--border);border-radius:6px;margin:.55rem 0;}
.prop-sum{cursor:pointer;font-size:.8rem;color:var(--text_muted);padding:.5rem .8rem;list-style:none;user-select:none;}
.prop-sum::-webkit-details-marker{display:none;}
.prop-sum:hover{color:var(--accent);}
.prop-body{padding:.5rem .8rem .8rem;}
.prop-note{margin-bottom:.5rem;max-width:640px;}
.prop-table{width:100%;border-collapse:collapse;font-size:.76rem;white-space:nowrap;}
.prop-table th{padding:.27rem .5rem;text-align:left;border-bottom:1px solid var(--border);color:var(--text_muted);}
.prop-table td{padding:.24rem .5rem;border-bottom:1px solid var(--border);}
.pname{font-weight:600;color:var(--text);}
.pprov{font-size:.68rem;}
.pbar-bg{display:inline-block;width:90px;height:6px;background:var(--bg);border-radius:3px;overflow:hidden;border:1px solid var(--border);vertical-align:middle;margin-right:.35rem;}
.pbar-fill{height:100%;background:var(--accent);border-radius:3px;}
.pval{font-size:.72rem;font-family:var(--font-mono);color:var(--text_muted);vertical-align:middle;}
.status-bar{display:flex;align-items:center;gap:.6rem;padding:.45rem .65rem;background:var(--surface);border:1px solid var(--accent);border-radius:5px;font-size:.8rem;margin-bottom:.4rem;}
.spin-anim{color:var(--accent);font-family:var(--font-mono);}
.log-wrap{font-family:var(--font-mono);font-size:.7rem;color:var(--text_muted);max-height:170px;overflow-y:auto;background:var(--bg);border:1px solid var(--border);border-radius:4px;padding:.35rem .55rem;}
.log-line{padding:.06rem 0;border-bottom:1px solid var(--border);}
.log-line:last-child{color:var(--text);}
.tbl-scroll{overflow-x:auto;-webkit-overflow-scrolling:touch;}
.s-card{padding:.75rem .85rem;margin-bottom:.4rem;}
.s-name{font-family:var(--font-mono);font-size:.8rem;font-weight:700;margin-bottom:.28rem;overflow:hidden;text-overflow:ellipsis;white-space:nowrap;}
.s-stats{display:flex;gap:.7rem;font-size:.74rem;margin-bottom:.35rem;flex-wrap:wrap;}
.s-mem-bar{display:flex;height:9px;border-radius:4px;overflow:hidden;background:var(--bg);border:1px solid var(--border);margin-bottom:.22rem;}
.s-seg{height:100%;transition:width .4s;}.s-seg-m{background:#3d9aff;}.s-seg-k{background:#ff9a3c;}.s-seg-f{background:var(--border);opacity:.3;flex:1;}
.s-mem-labels{font-size:.66rem;display:flex;gap:.55rem;flex-wrap:wrap;margin-bottom:.3rem;}
.s-opts summary{cursor:pointer;font-size:.7rem;color:var(--text_muted);list-style:none;user-select:none;margin-top:.25rem;}
.s-opts summary::-webkit-details-marker{display:none;}
.s-opts-sum:hover{color:var(--accent);}
"""

# JavaScript

SCRIPT = """
(function(){
    // HW field sync - copies .hwin values to hidden #hw-fields for hx-include
    function syncHW(){
        var hw = document.getElementById('hw-fields');
        if(!hw) return;
        hw.innerHTML = '';
        document.querySelectorAll('.hwp .hwin[name]').forEach(function(inp){
            if(inp.tagName === 'SELECT') return;
            var h = document.createElement('input');
            h.type='hidden'; h.name=inp.name; h.value=inp.value;
            hw.appendChild(h);
        });
    }
    window.syncHW = syncHW;
    document.querySelectorAll('.hwp .hwin').forEach(function(inp){ inp.addEventListener('input', syncHW); });
    syncHW();

    // Task preset weights
    var PRESET_WEIGHTS = """ + json.dumps({k:{d:v["weights"].get(d,DEFAULT_WEIGHTS[d]) for d in SCORE_DIMS} for k,v in TASK_PRESETS.items()}) + """;
    window.applyPreset = function(preset){
        var ws = PRESET_WEIGHTS[preset]; if(!ws) return;
        ['lb_avg','quant_qual','speed','mem_head','ctx_fit','popularity', 'intel'].forEach(function(d){
            var sl = document.querySelector('.wsblock input[name="w_'+d+'"]');
            if(sl){ sl.value = ws[d]; var lbl = sl.parentElement.querySelector('.wsv'); if(lbl) lbl.textContent = ws[d]; }
        });
    };

    // Context block planner
    var _blocks = [];
    var BCOLORS = {pinned:'#3d9aff',knowledge:'#00ffa2',goal:'#ffcc00',history:'#ff9a3c',workspace:'#b06aff'};
    window.addCtxBlock = function(){
        var name = document.getElementById('blk-name').value.trim();
        var tok  = parseInt(document.getElementById('blk-tokens').value) || 0;
        var type = document.getElementById('blk-type').value;
        if(!name || !tok) return;
        _blocks.push({name, tok, type, id: Date.now()});
        renderCtxBlocks();
        document.getElementById('blk-name').value = '';
    };
    window.clearCtxBlocks = function(){ _blocks = []; renderCtxBlocks(); };
    window.removeCtxBlock = function(id){ _blocks = _blocks.filter(function(b){ return b.id !== id; }); renderCtxBlocks(); };
    function renderCtxBlocks(){
        var list = document.getElementById('ctx-block-list'); if(!list) return;
        list.innerHTML = _blocks.map(function(b){
            return '<div style="display:flex;align-items:center;gap:.45rem;padding:.28rem .45rem;background:var(--surface);border:1px solid var(--border);border-radius:4px;font-size:.76rem">'
                +'<span style="width:7px;height:7px;border-radius:50%;background:'+(BCOLORS[b.type]||'#888')+';flex-shrink:0"></span>'
                +'<span style="flex:1">'+b.name+'</span>'
                +'<span style="font-family:var(--font-mono);color:var(--text_muted)">'+b.tok.toLocaleString()+' t</span>'
                +'<span style="font-size:.63rem;color:var(--text_muted)">'+b.type+'</span>'
                +'<button onclick="removeCtxBlock('+b.id+')" style="background:none;border:none;color:#ff5f5f;cursor:pointer;padding:0 .2rem">&#x2715;</button>'
                +'</div>';
        }).join('');
        updateCtxBudget();
    }
    window.updateCtxBudget = function(){
        var used = _blocks.reduce(function(s,b){ return s+b.tok; }, 0);
        var max  = window._ctxBudget ? window._ctxBudget.maxCtx : 0;
        var el   = document.getElementById('ctx-block-budget'); if(!el) return;
        if(!max){ el.textContent = 'Run analysis above to see token budget'; return; }
        var pct = Math.min(used/max*100, 100);
        var col = pct < 70 ? '#00ffa2' : pct < 88 ? '#ffcc00' : '#ff5f5f';
        var rem = max - used;
        el.innerHTML = '<div style="display:flex;align-items:center;gap:.5rem;margin-bottom:.2rem">'
            +'<div style="flex:1;height:6px;background:var(--bg);border:1px solid var(--border);border-radius:3px;overflow:hidden">'
            +'<div style="height:100%;width:'+pct.toFixed(1)+'%;background:'+col+';border-radius:3px"></div></div>'
            +'<span style="font-size:.72rem;color:'+col+'">'+used.toLocaleString()+' / '+max.toLocaleString()+' tokens ('+pct.toFixed(0)+'%)</span></div>'
            +'<div style="font-size:.68rem;color:var(--text_muted)">Remaining: '+(rem>0 ? rem.toLocaleString()+' tokens for history+response' : '<span style=\"color:#ff5f5f\">OVER BUDGET</span>')+'</div>';
    };
    window.saveCtxProfile = async function(){
        var name = prompt('Profile name?'); if(!name) return;
        var r = await fetch('/module/ai_tools/ai_calc/ctx_profile_save', {method:'POST', headers:{'Content-Type':'application/json'}, body:JSON.stringify({name, blocks:_blocks})});
        if(r.ok) alert('Saved: '+name);
    };

    // Show re-rank form after search completes
    document.addEventListener('htmx:afterSwap', function(e){
        if(e.detail && e.detail.target && e.detail.target.id === 'srch-out'){
            var rr = document.getElementById('srch-rerank');
            if(rr){ rr.style.display = 'block'; }
            var tgt = document.getElementById('target-tps-in');
            var rrt = document.getElementById('rr-tps');
            if(tgt && rrt) rrt.value = tgt.value;
        }
        document.querySelectorAll('.hwp .hwin').forEach(function(inp){
            inp.removeEventListener('input', syncHW);
            inp.addEventListener('input', syncHW);
        });
        syncHW();
    });
})();
"""