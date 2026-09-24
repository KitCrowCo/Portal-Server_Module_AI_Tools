"""
main.py — Flux2 Klein image generation server (FastAPI).

Changes vs previous version:
  - _progress dict + /system/progress endpoint (issue 8)
  - offload_mode field: "none" | "vae_cpu" (VAE offloaded to CPU during denoising, back for decode, saves ~1 GB VRAM for larger images) (issue 1)
  - mask_image field now accepts a file-relative path as well as data URIs, so the frontend can save masks to the shared volume and reference them by path (issue 2)
  - Aggressive cache clearing between steps when offload_mode is active
"""

import os, sys, torch, gc, logging, threading, json, re, io, base64, time
import asyncio
from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, ConfigDict
from typing import Optional, List
from PIL.PngImagePlugin import PngInfo
from datetime import datetime
from types import ModuleType
import traceback
import inspect
import numpy as np
from PIL import Image

# =========================================================================
# PATCH 1: Metaclass-Driven Metamorphic Mock for 'flex_attention'
if not hasattr(torch.nn, "attention") or not hasattr(torch.nn.attention, "flex_attention"):
    if not hasattr(torch.nn, "attention"):
        attn_module = ModuleType("torch.nn.attention")
        sys.modules["torch.nn.attention"] = attn_module
        torch.nn.attention = attn_module

    class MetaMock(type):
        def __call__(cls, *args, **kwargs): return cls
        def __getattr__(cls, name): return type(name, (UniversalMock,), {})
        def __getitem__(cls, item): return cls
        def endswith(cls, *args, **kwargs): return False

    class UniversalMock(metaclass=MetaMock):
        def __call__(self, *args, **kwargs): return self
        def __getattr__(self, name): return type(name, (UniversalMock,), {})()
        def __getitem__(self, item): return self
        def endswith(self, *args, **kwargs): return False

    flex_mock = type("flex_attention", (UniversalMock,), {})
    sys.modules["torch.nn.attention.flex_attention"] = flex_mock
    torch.nn.attention.flex_attention = flex_mock

# PATCH 2: Temporary, Isolated Type Registry Hook
original_infer_schema = None
try:
    import torch._library.infer_schema
    original_infer_schema = torch._library.infer_schema.infer_schema

    def temporary_infer_schema(prototype_function, *args, **kwargs):
        try:
            if hasattr(prototype_function, "__annotations__"):
                eval_env = {
                    'torch': torch, 'Optional': Optional, 'tuple': tuple,
                    'list': list, 'Union': __import__('typing').Union,
                    'NoneType': type(None), 'str': str, 'int': int,
                    'float': float, 'bool': bool,
                    'Sequence': __import__('typing').Sequence,
                    'List': __import__('typing').List,
                }
                for param_name, current_val in list(prototype_function.__annotations__.items()):
                    if isinstance(current_val, str):
                        try:
                            prototype_function.__annotations__[param_name] = eval(current_val, eval_env)
                        except Exception:
                            pass
        except Exception:
            pass
        return original_infer_schema(prototype_function, *args, **kwargs)

    torch._library.infer_schema.infer_schema = temporary_infer_schema
except (ImportError, AttributeError):
    pass
# =========================================================================

import diffusers
import accelerate
from diffusers.quantizers.gguf import utils as gguf_utils

import diffusers.utils.import_utils
from safetensors.torch import load_file
from peft.utils import set_peft_model_state_dict

diffusers.utils.import_utils._accelerate_available = True
diffusers.utils.import_utils._accelerate_version = "1.1.0"

ACCELERATE_IMPORT_ERROR = """
{0} requires the accelerate library but it was not found. pip install accelerate
"""
diffusers.utils.import_utils.BACKENDS_MAPPING["accelerate"] = (diffusers.utils.import_utils.is_accelerate_available, ACCELERATE_IMPORT_ERROR)

def patched_dtype(self): return torch.bfloat16
gguf_utils.GGUFParameter.dtype = property(patched_dtype)

import torch.nn.functional as F
original_sdpa = F.scaled_dot_product_attention

def patched_sdpa(*args, **kwargs):
    for arg in ["enable_gqa"]:
        kwargs.pop(arg, None)
    return original_sdpa(*args, **kwargs)

F.scaled_dot_product_attention = patched_sdpa
print("DEBUG: Global F.scaled_dot_product_attention patched.", flush=True)

sys.path.insert(0, os.path.abspath("local_overrides"))
from diffusers.quantizers.gguf import utils as utils  # noqa: F811
print(f"DEBUG: Using utils from {utils.__file__}", flush=True)

from accelerate.hooks import remove_hook_from_submodules
from pipeline_flux2_klein_inpaint import Flux2KleinInpaintPipeline
from diffusers import Flux2Transformer2DModel, AutoencoderKLFlux2, FlowMatchEulerDiscreteScheduler, GGUFQuantizationConfig
import peft  # noqa: F401

# ---------------------------------------------------------------------------
app = FastAPI()
OUTPUT_DIR = "/app/outputs"
LORA_DIR = os.getenv("LORA_DIR", "/app/models/lora/flux2")
LOG_FILE = os.path.join(OUTPUT_DIR, "logs.txt")
os.makedirs(OUTPUT_DIR, exist_ok=True)
app.mount("/outputs", StaticFiles(directory=OUTPUT_DIR), name="outputs")

logging.basicConfig(filename=LOG_FILE, level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger("ai-image")
logger.addHandler(logging.StreamHandler())

EMBED_VAULT_DIR = os.getenv("EMBED_VAULT_DIR", "/app/shared_vault/embeds")
JOBS_DIR = os.path.join(EMBED_VAULT_DIR, "jobs")
EXPECTED_TEXT_DIM = int(os.getenv("EXPECTED_TEXT_DIM", str(4096 * 3)))
TRANSFORMER_DIR = "/app/models/transformers"
VAE_PATH = "/app/models/VAE"

# ---------------------------------------------------------------------------
# Progress tracking — written each denoising step, polled by the frontend
_progress: dict = {"step": 0, "total": 0, "job_id": "", "ts": 0.0, "status": "idle", "pct": 0}

def _make_step_callback(job_id: str, total: int):
    def cb(pipe, step: int, timestep, kwargs: dict):
        done = step + 1
        _progress.update({"step": done, "total": total, "job_id": job_id, "ts": time.time(), "status": "running", "pct": int(done * 100 / max(total, 1))})
        logger.info(f"[progress] job={job_id} step={done}/{total} ({_progress['pct']}%)")
        return kwargs
    return cb

# -----------------------------------------------------------
class GenerationRequest(BaseModel):
    model_config = ConfigDict(extra="allow")
    prompt: Optional[str] = ""
    embed_job_id: Optional[str] = None
    image_path: Optional[str] = None
    image: Optional[str] = None
    mask_image: Optional[str] = None   # data URI OR path relative to OUTPUT_DIR
    reference_image: Optional[str] = None
    strength: Optional[float] = 0.75
    width: int = 1600
    height: int = 900
    steps: int = 4
    guidance_scale: float = 1.0
    shift: Optional[float] = None
    seed: int = -1
    model_path: str = "flux-2-klein-9b-Q6_K.gguf"
    vae_path: str = "flux2"
    loras: list = []
    vae_tiling: bool = True
    offload_mode: str = "none"   # "none" | "vae_cpu"
    meta: list = ["prompt", "width", "height", "guidance_scale", "steps", "seed", "loras"]
    sequence_dir: Optional[str] = None

class SilentlyEmptyDict(dict):
    def __len__(self): return 0
    def __bool__(self): return False
# ------------------------------------------------------------

class APUEngine:
    def __init__(self):
        self.pipe = None
        self.is_loaded = False
        self.interrupt_flag = threading.Event()
        self.active_loras: list = []
        self.last_error: Optional[str] = None
        self._base_scheduler_config = None
        self._lora_hooks: list = []
        self._last_lora_config: list = []

    def execution_device(self) -> torch.device:
        if self.pipe is None: return torch.device("cpu")
        dev = self.pipe._execution_device
        if torch.cuda.is_available() and torch.device(dev).type == "cpu": logger.warning(f"cuda available but pipe._execution_device={dev}. Unload and reload to fix.")
        return torch.device(dev)

    def resolve_embed_path(self, request: "GenerationRequest"):
        job_dir = os.path.join(JOBS_DIR, request.embed_job_id)
        embeds_path = os.path.join(job_dir, "embeds.npy")
        meta_path = os.path.join(job_dir, "meta.json")
        if not os.path.exists(embeds_path): raise FileNotFoundError(f"No embeds.npy for embed_job_id={request.embed_job_id} at {embeds_path}")
        meta = None
        if os.path.exists(meta_path):
            with open(meta_path) as f:
                meta = json.load(f)
        return embeds_path, meta

    def load_local_embeddings(self, request: "GenerationRequest") -> torch.Tensor:
        embeds_path, meta = self.resolve_embed_path(request)
        logger.info(f"Loading embeddings from: {embeds_path}")
        np_arr = np.load(embeds_path)
        logger.info(f"Loaded array shape={np_arr.shape} dtype={np_arr.dtype}")
        if np_arr.ndim != 3 or np_arr.shape[0] != 1: raise ValueError(f"Expected (1, seq_len, dim), got {np_arr.shape}")
        if np_arr.shape[-1] != EXPECTED_TEXT_DIM: raise ValueError(f"Embeds last dim {np_arr.shape[-1]} != expected {EXPECTED_TEXT_DIM}.")
        if not np.isfinite(np_arr).all(): raise ValueError("Embeds contain NaN/Inf — re-encode this prompt.")
        if meta and meta.get("stats", {}).get("std", 1.0) < 1e-6: logger.warning(f"Near-zero std in embeddings: {meta.get('stats')}")
        return torch.from_numpy(np_arr).to(device=self.execution_device(), dtype=torch.bfloat16)

    # -- LoRA helpers (unchanged from original) --
    def clear_loras(self):
        removed = 0
        for handle in self._lora_hooks:
            try:
                handle.remove()
                removed += 1
            except Exception:
                pass
        self._lora_hooks.clear()
        if removed: logger.info(f"Removed {removed} LoRA forward hooks.")
        self.active_loras = []
        gc.collect()
        if torch.cuda.is_available(): torch.cuda.empty_cache()

    def apply_lora_weights(self, loras_list: list):
        normalized = [{"path": l.get("path", ""), "scale": round(float(l.get("scale", 1.0)), 4)} for l in loras_list if l.get("path")]
        if normalized == self._last_lora_config:
            logger.info(f"[LoRA] config unchanged, skipping re-apply")
            return
        self.clear_loras()
        if not normalized or self.pipe is None:
            self._last_lora_config = []
            return
        model = self.pipe.transformer
        dev = next(model.parameters()).device
        dtype = next(model.parameters()).dtype
        module_index = {name: mod for name, mod in model.named_modules() if hasattr(mod, "weight") and mod.weight is not None}
        for i, lora_cfg in enumerate(loras_list[:3]):
            raw_path = lora_cfg.get("path", "")
            if not raw_path: continue
            path = os.path.join(LORA_DIR, raw_path)
            scale = float(lora_cfg.get("scale", 1.0))
            name = lora_cfg.get("name") or f"lora_{i}_{os.path.basename(raw_path)}"
            try:
                self._apply_single_lora(path, name, scale, module_index, dev, dtype)
            except Exception as e:
                logger.error(f"[LoRA] Failed to apply {name}: {e}\n{traceback.format_exc()}")
        self._last_lora_config = normalized

    def _register_lora_hook(self, module, lora_down, lora_up, eff_scale, tag, resolved_name):
        import torch.nn.functional as F_local
        in_features = getattr(module, "in_features", None)
        out_features = getattr(module, "out_features", None)
        if in_features is None or out_features is None: out_features, in_features = module.weight.shape[0], module.weight.shape[1]

        if lora_down.shape[1] == in_features and lora_up.shape[0] == out_features:
            _ld, _lu, _s = lora_down, lora_up, eff_scale

            def _hook_lora(mod, inp, out, ld=_ld, lu=_lu, s=_s):
                x = inp[0]
                if ld.device != x.device or ld.dtype != x.dtype:
                    ld = ld.to(x.device, x.dtype)
                    lu = lu.to(x.device, x.dtype)
                orig_shape = out.shape
                x2 = x.reshape(-1, x.shape[-1])
                delta = F_local.linear(F_local.linear(x2, ld), lu)
                return out + s * delta.view(orig_shape)

            self._lora_hooks.append(module.register_forward_hook(_hook_lora))
            return True

        A, B = lora_down, lora_up
        m, n = A.shape
        p, q = B.shape
        if m * p == out_features and n * q == in_features:
            _A, _B, _s, _m, _n, _p, _q = A, B, eff_scale, m, n, p, q

            def _hook_lokr(mod, inp, out, A=_A, B=_B, s=_s, m=_m, n=_n, p=_p, q=_q):
                x = inp[0]
                if A.device != x.device or A.dtype != x.dtype:
                    A = A.to(x.device, x.dtype)
                    B = B.to(x.device, x.dtype)
                orig_shape = out.shape
                x_r = x.reshape(*x.shape[:-1], n, q)
                temp = torch.einsum('ij,...jl->...il', A, x_r)
                y = torch.einsum('kl,...il->...ik', B, temp)
                return out + s * y.reshape(*orig_shape[:-1], m * p)

            self._lora_hooks.append(module.register_forward_hook(_hook_lokr))
            logger.info(f"[LoRA] {tag}: '{resolved_name}' LoKR A={tuple(A.shape)} B={tuple(B.shape)}")
            return True

        logger.warning(f"[LoRA] {tag}: shape mismatch '{resolved_name}' in={in_features} out={out_features} down={tuple(lora_down.shape)} up={tuple(lora_up.shape)}")
        return False

    def _resolve_fused_target(self, module_index, block_prefix, lora_up, lora_down):
        total_out = lora_up.shape[0]
        in_dim = lora_down.shape[1]
        candidates = [(n, m) for n, m in module_index.items() if n.startswith(block_prefix + ".") and hasattr(m, "out_features") and hasattr(m, "in_features")]
        if not candidates: return None
        for n, m in candidates:
            if m.out_features == total_out and m.in_features == in_dim: return [(m, n, slice(0, total_out))]
        same_in = [(n, m) for n, m in candidates if m.in_features == in_dim]
        running, chosen = 0, []
        for n, m in same_in:
            if running >= total_out: break
            chosen.append((n, m, running))
            running += m.out_features
        if running == total_out and chosen: return [(m, n, slice(off, off + m.out_features)) for n, m, off in chosen]
        return None

    def _apply_single_lora(self, path, name, scale, module_index, dev, dtype):
        state_dict = load_file(path, device="cpu")
        all_keys = sorted(state_dict.keys())
        logger.info(f"[LoRA] {name} — {len(all_keys)} tensors. First 5: {all_keys[:5]}")

        down_map, up_map, alpha_map = {}, {}, {}
        for k, v in state_dict.items():
            if k.endswith(".alpha"):
                alpha_map[k[:-len(".alpha")]] = float(v.item())
                continue
            is_weight = k.endswith(".weight")
            stem = k[:-7] if is_weight else k
            if stem.endswith(".lora_down") or stem.endswith(".lora_A") or stem.endswith(".down"):
                down_map[stem.rsplit(".", 1)[0]] = v
            elif stem.endswith(".lora_up") or stem.endswith(".lora_B") or stem.endswith(".up"):
                up_map[stem.rsplit(".", 1)[0]] = v

        matched, skipped_keys = 0, []
        for base in down_map:
            if base not in up_map: continue
            lora_down = down_map[base].to(dev, dtype)
            lora_up = up_map[base].to(dev, dtype)
            rank = lora_down.shape[0]
            alpha = alpha_map.get(base, float(rank))
            eff_scale = scale * alpha / rank

            if base.endswith("img_attn.qkv") or base.endswith("txt_attn.qkv"):
                is_txt = base.endswith("txt_attn.qkv")
                block_prefix = base.rsplit(".", 2)[0]
                for pfx in ("diffusion_model.", "transformer.", "base_model.model."):
                    if block_prefix.startswith(pfx): block_prefix = block_prefix[len(pfx):]
                block_prefix = (block_prefix.replace("double_blocks", "transformer_blocks").replace("single_blocks", "single_transformer_blocks"))
                targets = ["add_q_proj", "add_k_proj", "add_v_proj"] if is_txt else ["to_q", "to_k", "to_v"]
                out_dim = lora_up.shape[0]
                if out_dim % 3 != 0:
                    skipped_keys.append(base)
                    continue
                chunk = out_dim // 3
                sub_applied = 0
                for idx, tgt in enumerate(targets):
                    resolved_key = f"{block_prefix}.attn.{tgt}"
                    module = module_index.get(resolved_key)
                    if module is None: continue
                    up_slice = lora_up[idx * chunk:(idx + 1) * chunk, :]
                    if self._register_lora_hook(module, lora_down, up_slice, eff_scale, name, resolved_key): sub_applied += 1
                if sub_applied == 0:
                    skipped_keys.append(base)
                else:
                    matched += 1
                continue

            module, resolved = self._find_module(module_index, base)
            if module is not None:
                if self._register_lora_hook(module, lora_down, lora_up, eff_scale, name, resolved):
                    matched += 1
                else:
                    skipped_keys.append(base)
                continue

            block_m = re.search(r'^(.*?\.\d+)\.', base)
            if block_m:
                fused = self._resolve_fused_target(module_index, block_m.group(1), lora_up, lora_down)
                if fused:
                    sub_applied = 0
                    for sub_mod, sub_name, sl in fused:
                        up_slice = lora_up[sl, :]
                        if self._register_lora_hook(sub_mod, lora_down, up_slice, eff_scale, name, sub_name): sub_applied += 1
                    if sub_applied:
                        matched += 1
                        logger.info(f"[LoRA] {name}: '{base}' fused split -> {[s for _,s,_ in fused]}")
                        continue
            skipped_keys.append(base)

        if skipped_keys: logger.warning(f"[LoRA] {name}: {len(skipped_keys)} unmatched. First: {skipped_keys[:5]}")
        if matched:
            self.active_loras.append(name)
            logger.info(f"[LoRA] {name}: {matched}/{len(down_map)} adapters applied")
        else:
            logger.error(f"[LoRA] {name}: ZERO adapters matched — check key format.")

    def _find_module(self, module_index, key):
        for prefix in ["diffusion_model.", "transformer.", "base_model.model."]:
            if key.startswith(prefix): key = key[len(prefix):]
#        if key in _DIRECT_MODULE_MAP and _DIRECT_MODULE_MAP[key] in module_index:
#            return module_index[_DIRECT_MODULE_MAP[key]], _DIRECT_MODULE_MAP[key]
        name_swaps = {"double_blocks": "transformer_blocks",
                      "single_blocks": "single_transformer_blocks",
                      "img_attn.proj": "attn.to_out.0",
                      "txt_attn.proj": "attn.to_add_out",
                      "img_mlp.0": "ff.net.0.proj",
                      "img_mlp.2": "ff.net.2",
                      "txt_mlp.0": "ff_context.net.0.proj",
                      "txt_mlp.2": "ff_context.net.2",
                      "linear1": "proj_mlp",
                      "linear2": "proj_out",
                      "img_in": "x_embedder",
                      "final_layer.linear": "proj_out",
                      "final_layer.adaLN_modulation.1": "norm_out.linear",
                      "double_stream_modulation_img.lin": "time_text_embed.timestep_embedder.linear_2",
                      "double_stream_modulation_txt.lin": "time_text_embed.text_embedder.linear_2"}

        mapped_key = key
        for old, new in name_swaps.items():
            mapped_key = mapped_key.replace(old, new)
        if mapped_key in module_index: return module_index[mapped_key], mapped_key
        alt_key = mapped_key.replace("net.0.proj", "linear_in").replace("net.2", "linear_out")
        if alt_key in module_index: return module_index[alt_key], alt_key
        block_match = re.search(r'\.(\d+)\.', mapped_key)
        block_id = block_match.group(0) if block_match else None
        key_norm = mapped_key.replace("_", ".").lower()
        candidates = []
        for m_name, m_obj in module_index.items():
            if block_id and block_id not in m_name: continue
            m_norm = m_name.replace("_", ".").lower()
            if m_norm.endswith(key_norm) or key_norm.endswith(m_norm): candidates.append((m_name, m_obj))
        if len(candidates) == 1: return candidates[0][1], candidates[0][0]
        return None, key

    def set_vae_tiling(self, enabled: bool):
        try:
            vae = getattr(self.pipe, "vae", None) if self.pipe else None
            if vae is None: return
            if enabled and hasattr(vae, "enable_tiling"):
                vae.enable_tiling()
            elif not enabled and hasattr(vae, "disable_tiling"):
                vae.disable_tiling()
        except Exception as e:
            logger.warning(f"VAE tiling toggle failed (non-fatal): {e}")

    def maybe_override_scheduler_shift(self, shift: Optional[float]):
        if shift is None or self.pipe is None or self._base_scheduler_config is None: return
        try:
            cfg = dict(self._base_scheduler_config)
            cfg["shift"] = shift
            self.pipe.scheduler = FlowMatchEulerDiscreteScheduler.from_config(cfg)
            logger.info(f"Scheduler shift overridden to {shift}")
        except Exception as e:
            logger.warning(f"Failed to override scheduler shift={shift}: {e}")

    def load(self, model_path: str, vae_path: str = "flux2"):
        if self.is_loaded: return
        cuda_dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model_path = os.path.join(TRANSFORMER_DIR, model_path)
        vae_path_dir = os.path.join(VAE_PATH, vae_path)
        logger.info(f"Loading VAE from {vae_path_dir}, transformer from {model_path}")

        with open(os.path.join(vae_path_dir, "config.json")) as f:
            vae_config = json.load(f)
        vae = AutoencoderKLFlux2.from_config(vae_config)
        vae_state_dict = load_file(os.path.join(vae_path_dir, "flux2-vae.safetensors"))
        vae.load_state_dict(vae_state_dict)
        vae.to(torch.bfloat16).to(cuda_dev)
        logger.info(f"VAE on {cuda_dev} dtype={next(vae.parameters()).dtype}")

        logger.info(f"Loading GGUF transformer from {model_path}")
        transformer = Flux2Transformer2DModel.from_single_file(model_path, config=os.path.join(os.path.dirname(model_path), "config.json"), quantization_config=GGUFQuantizationConfig(compute_dtype=torch.bfloat16), torch_dtype=torch.bfloat16, low_cpu_mem_usage=True, local_files_only=True)

        # PATCH 3: GGUF dtype inference bypass
        class GGUFPatchedTransformer(transformer.__class__):
            @property
            def dtype(self): return torch.bfloat16
        transformer.__class__ = GGUFPatchedTransformer

        gc.collect()
        torch.cuda.empty_cache()

        scheduler_config = {"num_train_timesteps": 1000, "shift": 1.0}
        self._base_scheduler_config = scheduler_config
        self.pipe = Flux2KleinInpaintPipeline(transformer=transformer, vae=vae, text_encoder=None, tokenizer=None, scheduler=FlowMatchEulerDiscreteScheduler.from_config(scheduler_config), is_distilled=True)

        logger.info("Stripping accelerate hooks from transformer...")
        remove_hook_from_submodules(self.pipe.transformer)

        logger.info(f"Pinning transformer and VAE to {cuda_dev}...")
        self.pipe.transformer.to(cuda_dev)
        self.pipe.vae.to(cuda_dev)
        logger.info(f"Pipeline execution device: {self.pipe._execution_device}")

        self.pipe._original_state_dict = SilentlyEmptyDict()
        self.pipe.transformer._original_state_dict = SilentlyEmptyDict()
        self.pipe.vae._original_state_dict = SilentlyEmptyDict()

        self.set_vae_tiling(True)
        self.is_loaded = True
        gc.collect()
        torch.cuda.empty_cache()
        logger.info("Engine load complete.")

    def interrupt(self):
        logger.info("Interrupt signal sent...")
        self.interrupt_flag.set()

    def unload(self):
        self.interrupt()
        logger.info("Unloading models and clearing VRAM...")
        self.clear_loras()
        self.pipe = None
        self.is_loaded = False
        gc.collect()
        torch.cuda.empty_cache()

engine = APUEngine()

def _mem_info_str() -> str:
    """
    Note: the reported "total capacity" for an APU/iGPU is a driver-level GTT/VRAM split, not something torch or this process can change.
    The two real levers are (a) the host kernel param `amdgpu.gttsize=<MB>` (needs a reboot) or (b) the BIOS UMA framebuffer size — not fixable from Python.
    PYTORCH_ALLOC_CONF=expandable_segments:True (already set) only reduces fragmentation within whatever ceiling the driver reports.
    """
    if not torch.cuda.is_available(): return "cuda not available"
    try:
        free, total = torch.cuda.mem_get_info()
        return f"{free/1e9:.2f} GB free / {total/1e9:.2f} GB total"
    except Exception as e:
        return f"mem_get_info failed: {e}"

class _VaeDeviceGuard:
    """
    Wraps vae.encode AND vae.decode so the VAE only lives on GPU for the duration of each call.
    Fixes the dtype/device mismatch that occurred when only decode() was special-cased — encode() runs earlier, during prepare_latents/prepare_image_latents, while image tensors are already on CUDA.
    """
    def __init__(self, vae, device):
        self.vae = vae
        self.device = device
        self._orig_encode = vae.encode
        self._orig_decode = vae.decode

    def __enter__(self):
        def _wrap(fn):
            def wrapped(sample, *args, **kwargs):
                self.vae.to(self.device)
                try:
                    return fn(sample, *args, **kwargs)
                finally:
                    self.vae.to("cpu")
                    torch.cuda.empty_cache()
            return wrapped
        self.vae.encode = _wrap(self._orig_encode)
        self.vae.decode = _wrap(self._orig_decode)
        self.vae.to("cpu")
        logger.info(f"[vae-cpu-offload] enabled — {_mem_info_str()}")
        return self

    def __exit__(self, exc_type, exc, tb):
        self.vae.encode = self._orig_encode
        self.vae.decode = self._orig_decode
        self.vae.to(self.device)
        gc.collect()
        torch.cuda.empty_cache()
        logger.info(f"[vae-cpu-offload] disabled — {_mem_info_str()}")
        return False

class _SequentialBlockOffload:
    """
    Experimental "wait-overnight" mode.
    Moves each transformer block to GPU immediately before its forward pass and back to CPU immediately after, using plain PyTorch hooks — deliberately NOT accelerate's cpu_offload/AlignDevicesHook, which conflicts with this GGUF-quantized transformer's own .to() calls during forward (see the comment in APUEngine.load()).
    Much slower, frees nearly all transformer VRAM between blocks.
    """
    def __init__(self, transformer, device):
        self.transformer = transformer
        self.device = device
        self.handles = []
        self.block_lists = [getattr(transformer, attr) for attr in ("transformer_blocks", "single_transformer_blocks") if getattr(transformer, attr, None) is not None]

    def __enter__(self):
        def _pre(mod, inp):
            mod.to(self.device)
            return inp

        def _post(mod, inp, out):
            mod.to("cpu")
            return out

        for blocks in self.block_lists:
            for block in blocks:
                self.handles.append(block.register_forward_pre_hook(_pre))
                self.handles.append(block.register_forward_hook(_post))
                block.to("cpu")
        gc.collect()
        torch.cuda.empty_cache()
        logger.info(f"[sequential-offload] enabled — {_mem_info_str()}")
        return self

    def __exit__(self, exc_type, exc, tb):
        for h in self.handles:
            h.remove()
        for blocks in self.block_lists:
            for block in blocks:
                block.to(self.device)
        gc.collect()
        torch.cuda.empty_cache()
        logger.info(f"[sequential-offload] disabled — {_mem_info_str()}")
        return False

# --- Helpers ---

def _load_mask_image(mask_field: str, width: int, height: int) -> Image.Image:
    """
    Load mask image from either a data URI or a path relative to OUTPUT_DIR.
    Returns a grayscale PIL image resized to (width, height).
    White = inpaint, Black = keep.
    """

    if not mask_field: return Image.new("L", (width, height), 255)  # all white = full inpaint
    if mask_field.startswith("data:"): return decode_b64_image(mask_field).resize((width, height)).convert("L")

    # File path relative to OUTPUT_DIR
    mask_path = os.path.join(OUTPUT_DIR, mask_field.lstrip("/"))
    if not os.path.exists(mask_path): raise FileNotFoundError(f"Mask file not found at {mask_path}")
    return Image.open(mask_path).resize((width, height)).convert("L")

def decode_b64_image(b64_str: str) -> Image.Image:
    if "base64," in b64_str: b64_str = b64_str.split("base64,")[1]
    return Image.open(io.BytesIO(base64.b64decode(b64_str))).convert("RGB")

# --- System endpoints ---

@app.post("/system/load")
def load_system(model_path: str = "flux-2-klein-9b-Q6_K.gguf", vae_path: str = "flux2"):
    try:
        engine.load(model_path, vae_path)
        return {"status": "Loaded into memory"}
    except Exception as e:
        tb = traceback.format_exc()
        logging.error(f"Load failed:\n{tb}")
        engine.last_error = str(e)
        return {"status": "Error", "detail": str(e), "traceback": tb}

@app.post("/system/stop")
def stop_generation():
    engine.interrupt()
    _progress.update({"status": "stopped"})
    return {"status": "Interrupting..."}

@app.post("/system/clear_error")
def clear_error():
    engine.last_error = None
    engine.interrupt_flag.clear()
    _progress.update({"status": "idle", "step": 0, "total": 0, "pct": 0})
    return {"status": "ok"}

@app.post("/system/unload")
def unload_system():
    engine.unload()
    return {"status": "Memory Cleared"}

@app.get("/system/status")
def get_status():
    log_tail = ""
    try:
        with open(LOG_FILE, "r") as f:
            log_tail = "".join(f.readlines()[-40:])
    except Exception:
        pass
    return {"loaded": engine.is_loaded, "active_loras": engine.active_loras,
            "last_error": engine.last_error, "execution_device": str(engine.execution_device()) if engine.is_loaded else None,
            "torch_cuda_is_available": torch.cuda.is_available(), "vram": _mem_info_str(),
            "log_tail": log_tail, "progress": dict(_progress)}

@app.get("/system/progress")
def get_progress(): return dict(_progress) # Lightweight endpoint polled every 2s by the frontend during generation.

@app.delete("/outputs/{filename}")
def delete_output(filename: str):
    if not re.match(r'^[\w\-. /]+$', filename): return {"error": "invalid filename"}
    p = os.path.join(OUTPUT_DIR, filename)
    if os.path.exists(p):
        os.remove(p)
        return {"status": "deleted"}
    return {"error": "not found"}

# --- Generation ---

@app.post("/generate")
async def process_unified_generation(request: GenerationRequest):
    if not engine.is_loaded: load_system(request.model_path, request.vae_path)

    t0 = time.time()
    logger.info(f"[generate] starting. size={request.width}x{request.height} steps={request.steps} offload={request.offload_mode}")
    if not request.embed_job_id: raise HTTPException(status_code=400, detail="Provide `embed_job_id`")

    try:
        prompt_embeds = engine.load_local_embeddings(request)
    except Exception as e:
        logging.error(f"Embedding preflight failed:\n{traceback.format_exc()}")
        raise HTTPException(status_code=400, detail=f"Embedding validation failed: {e}")

    # Load base image if given by path
    if request.image is None and request.image_path:
        safe = os.path.basename(request.image_path)
        p = os.path.join(OUTPUT_DIR, safe)
        if not os.path.exists(p): raise HTTPException(400, f"image_path not found: {safe}")
        with open(p, "rb") as f:
            request.image = "data:image/png;base64," + base64.b64encode(f.read()).decode()
        logger.info(f"Loaded base image: {safe}")

    do_cfg = request.guidance_scale > 1.0
    negative_prompt_embeds = None
    if do_cfg:
        logger.warning("CFG active on distilled model — using zeroed negative embeds.")
        negative_prompt_embeds = torch.zeros_like(prompt_embeds)

    if request.seed is None or request.seed < 0:
        execution_seed = torch.randint(0, 2**32 - 1, (1,)).item()
    else:
        execution_seed = request.seed

    generator = torch.Generator(device=engine.execution_device()).manual_seed(execution_seed)
    engine.set_vae_tiling(request.vae_tiling)
    engine.maybe_override_scheduler_shift(request.shift)
    engine.apply_lora_weights(request.loras)

    pipe_kwargs = {
        "prompt_embeds": prompt_embeds,
        "negative_prompt_embeds": negative_prompt_embeds,
        "num_inference_steps": request.steps,
        "guidance_scale": request.guidance_scale,
        "height": request.height,
        "width": request.width,
        "output_type": "pil",
        "generator": generator,
        "callback_on_step_end": _make_step_callback(request.embed_job_id or "gen", request.steps),
        "callback_on_step_end_tensor_inputs": ["latents"],
    }

    if request.reference_image: pipe_kwargs["image_reference"] = decode_b64_image(request.reference_image).resize((request.width, request.height))

    # Mode routing
    if request.image and request.mask_image:
        pipe_kwargs["image"] = decode_b64_image(request.image).resize((request.width, request.height))
        pipe_kwargs["mask_image"] = _load_mask_image(request.mask_image, request.width, request.height)
        mask_arr = np.array(pipe_kwargs["mask_image"])
        logger.info(f"[mask-debug] shape={mask_arr.shape} unique_vals={np.unique(mask_arr)[:10]} min={mask_arr.min()} max={mask_arr.max()}")
        pipe_kwargs["strength"] = request.strength
    elif request.image:
        pipe_kwargs["image"] = decode_b64_image(request.image).resize((request.width, request.height))
        pipe_kwargs["mask_image"] = Image.new("L", (request.width, request.height), 255)
        pipe_kwargs["strength"] = request.strength
    else:
        pipe_kwargs["image"] = Image.new("RGB", (request.width, request.height), (0, 0, 0))
        pipe_kwargs["mask_image"] = Image.new("L", (request.width, request.height), 255)
        pipe_kwargs["strength"] = 1.0

    if os.getenv("DEBUG_INPAINT", "1") == "1" and request.mask_image:
        _mask_arr = np.array(pipe_kwargs["mask_image"])
        logger.info(f"[inpaint-debug] mask_mean={_mask_arr.mean():.2f} white_pct={(_mask_arr > 127).mean() * 100:.1f}% strength={pipe_kwargs.get('strength')}")

    offload_mode = request.offload_mode
    cuda_dev = engine.execution_device()
    vae_guard = None
    block_offload = None

    if offload_mode in ("vae_cpu", "sequential") and engine.pipe is not None:
        vae_guard = _VaeDeviceGuard(engine.pipe.vae, cuda_dev)
        vae_guard.__enter__()

    if offload_mode == "sequential" and engine.pipe is not None:
        block_offload = _SequentialBlockOffload(engine.pipe.transformer, cuda_dev)
        block_offload.__enter__()

    logger.info(f"[generate] offload_mode={offload_mode} — {_mem_info_str()}")

    _progress.update({"status": "running", "step": 0, "total": request.steps, "pct": 0, "job_id": request.embed_job_id or "gen"})

    try:
        print("--- TRACE: PIPELINE INFERENCE START ---", flush=True)
        with torch.no_grad():
            output = engine.pipe(**pipe_kwargs)
        generated_image = output.images[0]
        save_dir = OUTPUT_DIR
        rel_prefix = ""
        if request.sequence_dir:
            safe_dir = re.sub(r'[^\w\-]', '_', request.sequence_dir)[:64]
            save_dir = os.path.join(OUTPUT_DIR, "_sequences", safe_dir)
            os.makedirs(save_dir, exist_ok=True)
            rel_prefix = f"_sequences/{safe_dir}/"

        filename = f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_{execution_seed}.png"
        filepath = os.path.join(save_dir, filename)
        meta_values = {"prompt": request.prompt, "width": request.width, "height": request.height, "guidance_scale": request.guidance_scale, "steps": request.steps, "seed": execution_seed, "loras": json.dumps(request.loras), "embed_job_id": request.embed_job_id}
        png_info = PngInfo()
        for key in request.meta:
            if key in meta_values: png_info.add_text(key, str(meta_values[key]))
        for k in ("embed_job_id",):
            if k not in request.meta and meta_values.get(k): png_info.add_text(k, str(meta_values[k]))

        generated_image.save(filepath, format="PNG", pnginfo=png_info)
        elapsed = time.time() - t0
        logger.info(f"[generate] done in {elapsed:.1f}s → {filename}")
        _progress.update({"status": "idle", "pct": 100})
        return {"status": "success", "message": "Pipeline complete.", "seed": execution_seed, "elapsed_seconds": round(elapsed, 2), "file_name": filename, "absolute_path": filepath, "uri_endpoint": f"/outputs/{filename}", "active_loras": engine.active_loras}

    except torch.cuda.OutOfMemoryError:
        tb = traceback.format_exc()
        logging.error(f"OOM during generation:\n{tb}")
        engine.last_error = (f"HIP/CUDA out of memory. {_mem_info_str()}. Try a smaller size, a heavier offload_mode, or fewer/no LoRAs.")
        _progress.update({"status": "error"})
        raise HTTPException(status_code=500, detail=engine.last_error)

    except Exception as e:
        tb = traceback.format_exc()
        logging.error(f"Generation crash:\n{tb}")
        engine.last_error = str(e)
        _progress.update({"status": "error"})
        raise HTTPException(status_code=500, detail=f"Pipeline failed: {str(e)}")

    finally:
        if block_offload is not None: block_offload.__exit__(None, None, None)
        if vae_guard is not None: vae_guard.__exit__(None, None, None)
        gc.collect()
        torch.cuda.empty_cache()

    # Reset progress
    _progress.update({"status": "running", "step": 0, "total": request.steps, "pct": 0, "job_id": request.embed_job_id or "gen"})

@app.get("/lora/debug")
def lora_debug(block: str = ""):
    if not engine.is_loaded: return {"error": "Engine not loaded"}
    model = engine.pipe.transformer
    module_paths = [name for name, mod in model.named_modules() if hasattr(mod, "weight") and mod.weight is not None]
    if block: return {"block_filter": block, "matched": [n for n in module_paths if block in n]}
    lora_files = {}
    try:
        for f in os.listdir(LORA_DIR):
            if f.endswith(".safetensors"):
                sd = load_file(os.path.join(LORA_DIR, f), device="cpu")
                lora_files[f] = sorted(sd.keys())[:10]
    except Exception as e:
        lora_files["error"] = str(e)
    return {"model_module_paths_sample": module_paths[:30], "total_weighted_modules": len(module_paths), "active_lora_hooks": len(engine._lora_hooks), "lora_files_sample_keys": lora_files}

@app.post("/generate/sequence")
async def generate_sequence_endpoint(request: GenerationRequest):
    if not engine.is_loaded: raise HTTPException(400, "Engine not loaded")
    num_frames = max(2, min(int(request.model_extra.get("num_frames") or 8), 120))
    loop_gif = bool(request.model_extra.get("loop", True))
    prefix = str(request.model_extra.get("output_prefix") or "seq").strip() or "seq"
    logger.info(f"[sequence] {num_frames} frames seed={request.seed}")
    try:
        frames = []
        for i in range(num_frames):
            if engine.interrupt_flag.is_set():
                logger.info(f"[sequence] interrupted at frame {i}/{num_frames}")
                break
            seed = (request.seed + i) if (request.seed is not None and request.seed >= 0) else -1
            frame_req = request.model_copy(update={"seed": seed})
            result = await process_unified_generation(frame_req)
            if result.get("status") != "success": raise RuntimeError(f"Frame {i} failed: {result}")
            frames.append(Image.open(os.path.join(OUTPUT_DIR, result["file_name"])).convert("RGBA"))
            logger.info(f"[sequence] frame {i+1}/{num_frames}: {result['file_name']}")
            gc.collect()
            torch.cuda.empty_cache()
            await asyncio.sleep(0.05)

        if not frames: raise RuntimeError("No frames completed")
        out_name = f"{prefix}_{int(time.time())}.gif"
        out_path = os.path.join(OUTPUT_DIR, out_name)
        frames[0].save(out_path, format="GIF", save_all=True, append_images=frames[1:], loop=0 if loop_gif else 1, duration=120, optimize=False)
        logger.info(f"[sequence] saved {out_name}")
        return {"status": "success", "file_name": out_name, "frames": len(frames), "uri_endpoint": f"/outputs/{out_name}"}
    except Exception as e:
        tb = traceback.format_exc()
        logger.error(f"Sequence crash:\n{tb}")
        engine.last_error = str(e)
        raise HTTPException(500, str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
