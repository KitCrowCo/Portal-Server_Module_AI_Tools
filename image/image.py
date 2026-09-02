"""
image.py — Flux2 Klein multi-stage image generation workspace.
Sub-module of ai_tools. Mounted at /module/ai_tools/image.
"""
import json, uuid, asyncio, base64, os, re, time, mimetypes, traceback
import httpx
from pathlib import Path
from typing import Optional
from datetime import datetime
from fastapi import APIRouter, Request, Form, HTTPException
from fastapi.responses import HTMLResponse, FileResponse
from PIL import Image as PILImage

TOOL_META = {"label": "Image", "group": "model_tools", "icon": "&#x1F5BC;", "description": "Flux2 Klein multi-stage image pipeline", "singleton": True}

router = APIRouter(redirect_slashes=False)
_P = "/module/ai_tools/image"
DATA_DIR = Path("/app/data/ai_tools/image")
PROMPTS_DIR = DATA_DIR / "prompts"
JOB_RECORDS_DIR = DATA_DIR / "job_records"
INPAINT_DIR = DATA_DIR / "inpaint_inputs"
IMG_EXTS = {".png", ".jpg", ".jpeg", ".webp", ".gif"}
KG_DIR = Path("./data/ai_tools/_knowledge")
COMMON_DIR = Path("./data/_common")

_gallery_tool = None
ENV = {}
BI = UI = WS = IM = TM = AIM = _SETTINGS = _base_picker = _mask_picker = _ref_picker = None
_WORKER_TASK = None
thumb_data_uri = None

def _u(*p): return "/" + "/".join(s.strip("/") for s in [_P.strip("/"), *p] if s)
def _esc(s): return str(s).replace("&","&amp;").replace("<","&lt;").replace(">","&gt;").replace('"',"&quot;")

# --- Init ---

def _conn_options(values=None): return [("", "(none)")] + [(c["_id"], f'{c.get("display_name", c["_id"])} [{c.get("connection_type","?")}]') for c in AIM.connections.list_conns(get_all=True)]

def init_tool(env: dict, prefix: str):
    global ENV, BI, UI, WS, IM, TM, AIM, _SETTINGS, _P, _gallery_tool, thumb_data_uri, _base_picker, _mask_picker, _ref_picker
    ENV = env
    _P = prefix.rstrip("/")
    UI = env["templates"].env.globals.get("UI")
    WS = env["ws"]
    for d in (PROMPTS_DIR, JOB_RECORDS_DIR, INPAINT_DIR): d.mkdir(parents=True, exist_ok=True)
    BI = env["tools"]["built_ins"]
    AIM = ENV["tools"]["ai_manager"]
    _SETTINGS = BI.SettingsPanel("Image", [BI.SettingsGroup("defaults", "Defaults", [
        BI.SettingField("text_encoder_conn_id", "Text Encoder Connection", "select", "", options=_conn_options),
        BI.SettingField("image_gen_conn_id", "Image Generation Connection", "select", "", options=_conn_options),
        BI.SettingField("model_name", "Model Filename", "text", "flux-2-klein-9b-Q6_K.gguf", hint="Filename — relative to node's transformer dir"),
        BI.SettingField("vae_name", "VAE Directory Name", "text", "flux2", hint="Directory name — relative to node's VAE dir"),
        BI.SettingField("output_dir", "Output Directory (local path)", "text", str(DATA_DIR / "outputs"), hint="Full path on THIS container — for gallery scanning"),
        BI.SettingField("lora_dir", "LoRA Directory (local path)", "text", str(DATA_DIR / "models/loras"), hint="Full path on THIS container — for dropdown scanning"),
        BI.SettingField("max_sequence_length", "Max Sequence Length", "number", 1024, hint="Tokens; 512–2048. Longer needs more VRAM on encoder node."),
        BI.SettingField("vae_tiling", "VAE Tiling", "checkbox", True),
        BI.SettingField("offload_mode", "Memory Offload Mode", "select", "none", options=[("none", "None (fastest, most VRAM)"), ("vae_cpu", "VAE CPU Offload (~1 GB freed, slower)")]),
    ], json_path=str(DATA_DIR / "settings.json"))])
    IM = env["InterfaceManager"](nesting_level=2, db_path="ai_tools/image_im.db")
    TM = BI.TabManager(namespace="image", tab_bar_id="img-tab-bar", content_id="img-panel", render_content_fn=_render_panel, intent_prefix="image", IM=IM, scope="user", nesting_level=2, allow_new=False, closable=False,
        empty={"tabs": {"generate": {"id":"generate","order":0,"label":"Generate","icon":"&#x25B6;"},
                        "inpaint":  {"id":"inpaint","order":1,"label":"Inpaint","icon":"&#x1F58C;"},
                        "gallery":  {"id":"gallery","order":2,"label":"Gallery","icon":"&#x1F5BC;"},
                        "settings": {"id":"settings","order":3,"label":"Settings","icon":"&#x2699;"}}, "active":"generate"})
    thumb_data_uri = BI.thumb_data_uri

    # Tab-switch wrapper: update bottom toolbar on tab focus
    def _wrap_tab_action(base_fn):
        async def _h(req, pay, imr):
            imr = await base_fn(req, pay, imr)
            state = await TM._load(req)
            s = await _ui_state(req)
            bottom_inner = await _bottom_toolbar_html(req, state.get("active", "generate"), _list("prompt"), s["selected_prompt_id"], s["selected_mask"])
            imr.oob(bottom_inner or "", "img-bottom-content")
            return imr
        return _h

    async def _im_select_mask(request, payload, imr):
        rel = payload.get("path", "")
        await _ui_state(request, {"selected_mask": rel})
        imr.oob(f'<input id="img-mask-path" type="hidden" name="mask_path" form="img-inpaint-form" value="{_esc(rel)}">', "img-mask-path", swap="outerHTML")
        imr.oob(f'<span id="img-mask-status" style="font-size:.7rem;color:var(--text_muted);flex:1">&#x2713; Using saved mask: {_esc(rel)}</span>', "img-mask-status")
        return imr

    async def _im_select_reference(request, payload, imr):
        rel = payload.get("path", "")
        imr.oob(f'<input id="img-ref-path" type="hidden" name="reference_path" form="img-inpaint-form" value="{_esc(rel)}">', "img-ref-path", swap="outerHTML")
        imr.oob(f'<span id="img-ref-status" style="font-size:.6rem;color:var(--text_muted)">Reference: {_esc(rel)}</span>', "img-ref-status", swap="outerHTML")
        return imr

    async def _im_save_mask(request, payload, imr):
        print(f"[image_save_mask] called. mask_data length={len(payload.get('mask_data',''))}")
        mask_data = payload.get("mask_data", "")
        base_name = payload.get("base_name", "mask")
        if not mask_data:
            print("[image_save_mask] ABORT: mask_data empty on server side")
            imr.oob('<span style="color:#ff5f5f">No mask data reached the server</span>', "img-mask-status")
            return imr
        raw = mask_data.split("base64,")[-1] if "base64," in mask_data else mask_data
        ts = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
        fname = f"{ts}_{Path(base_name).stem[:30]}_mask.png"
        rel_path = f"_masks/{fname}"
        try:
            fm = _fm()
            dest = fm.resolve(rel_path)
            print(f"[image_save_mask] fm.root={fm.root}  dest={dest}  dest.parent.exists()={dest.parent.exists()}")
            dest.parent.mkdir(parents=True, exist_ok=True)   # <-- mkdir the SAME path we're about to write, not a separate constant
            decoded = base64.b64decode(raw)
            dest.write_bytes(decoded)
            print(f"[image_save_mask] wrote {len(decoded)} bytes -> {dest} (exists={dest.exists()})")
            fm._trigger_change(rel_path, "created")
        except Exception as e:
            traceback.print_exc()
            imr.oob(f'<span style="color:#ff5f5f">Save failed: {_esc(str(e)[:80])}</span>', "img-mask-status")
            return imr
        imr.oob(f'<span style="color:#00ffa2">&#x2713; Saved {_esc(fname)}</span>', "img-mask-status")
        imr.oob(f'<input id="img-mask-path" type="hidden" value="{_esc(rel_path)}">', "img-mask-path", swap="outerHTML")
        return imr

    for _intent in ("open", "focus", "close"): IM.scripts[f"image_{_intent}_tab"] = [_wrap_tab_action(getattr(TM, f"_{_intent}"))]
    IM.scripts["image_select_reference"] = [_im_select_reference]
    IM.scripts["image_select_mask"] = [_im_select_mask]
    IM.scripts["image_save_mask"] = [_im_save_mask]
    IM.scripts["image_assemble_gif"] = [_im_assemble_gif]
    _gallery_tool = BI.ImageGallery(root_dir=_output_dir(), IM=IM, intent_prefix="image_gallery", nesting_level=2, file_manager=_fm(), full_url_fn=lambda rel: _u("full", rel), transfer_targets={"common": "./data/_common", "knowledge": "./data/ai_tools/_knowledge"})
    _base_picker = BI.ImageGallery(root_dir=_output_dir(), IM=IM, intent_prefix="image_pick_base", nesting_level=2, file_manager=_fm(), select_mode=True, on_select=_pick_base_image)
    _mask_picker = BI.ImageGallery(root_dir=_output_dir(), IM=IM, intent_prefix="image_pick_mask", nesting_level=2, file_manager=_fm(), select_mode=True, on_select=_pick_mask_image)
    _ref_picker  = BI.ImageGallery(root_dir=_output_dir(), IM=IM, intent_prefix="image_pick_ref",  nesting_level=2, file_manager=_fm(), select_mode=True, on_select=_pick_reference_image)
    print("[image] ready")

async def _pick_base_image(request, payload, imr):
    rel = payload.get("path", "")
    fm = _fm()
    imr.oob(f'<input type="hidden" id="img-base-path" value="{_esc(rel)}"><input type="hidden" name="image_path" id="img-base-path-mirror" form="img-inpaint-form" value="{_esc(rel)}">', "img-base-path-mirror", swap="outerHTML")
    imr.oob(f'<div style="font-size:.7rem;color:var(--accent)">{_esc(rel)}</div>', "img-inpaint-base-hint", swap="outerHTML")
    imr.raw(f'<script>imgLoadBaseCanvas({json.dumps(_data_uri(rel, fm))})</script>')
    return imr

async def _pick_mask_image(request, payload, imr):
    rel = payload.get("path", "")
    fm = _fm()
    await _ui_state(request, {"selected_mask": rel})
    imr.oob(f'<input type="hidden" id="img-mask-path" name="mask_path" form="img-inpaint-form" value="{_esc(rel)}">', "img-mask-path", swap="outerHTML")
    imr.oob(f'<span style="font-size:.7rem;color:var(--accent)">&#x2713; Using saved mask: {_esc(rel)}</span>', "img-mask-status", swap="outerHTML")
    imr.raw(f'<script>imgLoadMaskCanvas({json.dumps(_data_uri(rel, fm))})</script>')
    return imr

async def _pick_reference_image(request, payload, imr):
    rel = payload.get("path", "")
    fm = _fm()
    imr.oob(f'<input type="hidden" id="img-ref-path" name="reference_path" form="img-inpaint-form" value="{_esc(rel)}">', "img-ref-path", swap="outerHTML")
    imr.oob(f'<div style="display:flex;align-items:center;gap:.3rem;font-size:.6rem;color:var(--text_muted)"><img src="{_u("thumb", rel)}" style="width:2rem;height:2rem;object-fit:cover;border-radius:.2rem"> {_esc(rel)}</div>', "img-ref-status", swap="outerHTML")
    return imr

def _ensure_worker():
    global _WORKER_TASK
    if _WORKER_TASK is None or _WORKER_TASK.done(): _WORKER_TASK = asyncio.create_task(_queue_worker())

# Config / paths

def _cfg(): return _SETTINGS.get_group("defaults").load()
def _conn(key): return AIM.connections.get_conn(_cfg().get(key, ""))
def _output_dir():
    p = Path(_cfg().get("output_dir") or DATA_DIR / "outputs")
    p.mkdir(parents=True, exist_ok=True)
    return p
def _lora_dir(): return Path(_cfg().get("lora_dir") or DATA_DIR / "models/loras")
def _fm(): return ENV["tools"]["built_ins"].FileManager(_output_dir())

# Shared helpers

def _parse_targets(raw) -> list:
    """Normalize gallery delete targets: hx-vals js:{array} arrives as JSON string."""
    if isinstance(raw, list): return raw
    if isinstance(raw, str):
        raw = raw.strip()
        if raw.startswith("["):
            try:
                return json.loads(raw)
            except Exception:
                pass
        return [raw] if raw else []
    return []

# --- Storage: prompts + job records ---

def _dp(kind, id_): return (PROMPTS_DIR if kind == "prompt" else JOB_RECORDS_DIR) / f"{Path(id_).name}.json"

def _load(kind, id_):
    p = _dp(kind, id_)
    return json.loads(p.read_text()) if p.exists() else None

def _save(kind, doc):
    doc["modified"] = datetime.utcnow().isoformat()
    _dp(kind, doc["id"]).write_text(json.dumps(doc, indent=2))

def _list(kind):
    d = PROMPTS_DIR if kind == "prompt" else JOB_RECORDS_DIR
    out = []
    for f in sorted(d.glob("*.json"), key=lambda x: x.stat().st_mtime, reverse=True):
        try:
            out.append(json.loads(f.read_text()))
        except Exception:
            pass
    return out

# --- Per-user WIP state ---

def _default_form(): return {"width": 512, "height": 1024, "steps": 4, "cfg": 1.0, "shift": 1.0, "seed": -1, "batch": 1, "output_prefix": "img", "loras": []}

async def _ui_state(request, patch=None):
    if patch is not None:
        s = await ENV["get_state"](request, scope="user", namespace="image") or {}
        s.update(patch)
        await ENV["set_state"](request, s, scope="user", namespace="image")
        return s
    s = await ENV["get_state"](request, scope="user", namespace="image") or {}
    s.setdefault("selected_prompt_id", "")
    s.setdefault("selected_mask", "")
    s.setdefault("form", _default_form())
    return s

# --- Prompt UI --- 

def _new_prompt(username): return {"id": f"pr_{uuid.uuid4().hex[:10]}", "username": username, "text": "", "title": "New Prompt", "status": "draft", "error": "", "meta": {}, "created": datetime.utcnow().isoformat()}
async def _push_prompts(username, selected=""): await WS.send_personal_message(f'<div id="img-prompts" hx-swap-oob="innerHTML">{_prompts_html(username, selected)}</div>', username)

def _prompts_html(username, selected=""):
    rows = ""
    for p in _list("prompt"):
        badge = {"ready": "#00ffa2", "encoding": "#ffcc00", "error": "#ff5f5f"}.get(p["status"], "var(--text_muted)")
        est = f"~{len(p.get('text',''))//4}t"
        rows += (f"""<div class="img-prompt-item{" active" if p["id"]==selected else ""}" hx-post="{_u("prompts/select")}" hx-vals='{{"id":"{p["id"]}"}}' hx-target="#img-prompt-panel" hx-swap="innerHTML">
                         <span style="width:.5rem;height:.5rem;border-radius:50%;background:{badge};flex-shrink:0"></span>
                         <span style="flex:1;overflow:hidden;text-overflow:ellipsis;white-space:nowrap">{_esc(p["title"])}</span>
                         <span style="font-size:.6rem;color:var(--text_muted);flex-shrink:0">{est}</span>
                         <button class="cm-qbtn" style="color:#ff5f5f" hx-delete="{_u("prompts",p["id"])}" hx-target="#img-prompts" hx-swap="innerHTML" hx-confirm="Delete prompt?" onclick="event.stopPropagation()">&#x2715;</button>
                     </div>""")
    return rows or '<div style="color:var(--text_muted);font-size:.75rem;padding:.5rem">No prompts yet.</div>'

def _prompt_editor_html(prompt):
    if not prompt: return '<div style="color:var(--text_muted);font-size:.78rem;padding:.5rem">Select or create a prompt.</div>'
    pid = prompt["id"]
    cfg = _cfg()
    max_len = int(cfg.get("max_sequence_length", 1024))
    tok_est = len(prompt.get("text", "")) // 4
    warn = ""
    if tok_est > max_len: warn = (f"""<div style="color:#ffaa44;font-size:.68rem;padding:.15rem 0">&#x26A0; ~{tok_est} tokens exceeds max ({max_len}). Prompt will be truncated at encode time.</div>""")
    status_line = {"ready": f'<span style="color:#00ffa2">&#x2713; ready — {prompt.get("meta",{}).get("tokenized_len","?")} tokens encoded</span>',
                   "encoding": '<span style="color:#ffcc00">&#x25CC; encoding&hellip;</span>',
                   "error": f'<span style="color:#ff5f5f">&#x26A0; {_esc(prompt.get("error",""))}</span>'}.get(prompt["status"], '<span style="color:var(--text_muted)">not encoded</span>')
    return (f"""<form hx-post="{_u("prompts",pid,"save")}" hx-target="#img-prompt-panel" hx-swap="innerHTML" style="display:flex;flex-direction:column;gap:.4rem;padding:.5rem">
                    <input type="text" name="title" value="{_esc(prompt["title"])}" class="module-select" style="font-size:.8rem;font-weight:600" placeholder="Title">
                    <textarea name="text" rows="18" class="module-select" style="font-size:.8rem;resize:vertical;min-height:16rem" oninput="var t=document.getElementById('pr-tok-{pid}');if(t)t.textContent='~'+(this.value.length>>2)+'t'">{_esc(prompt["text"])}</textarea>
                    <div style="display:flex;justify-content:space-between;align-items:center">
                        <span style="font-size:.68rem;color:var(--text_muted)">{status_line}</span>
                        <span id="pr-tok-{pid}" style="font-size:.66rem;color:var(--text_muted)">~{tok_est}t / {max_len}</span>
                    </div>
                    {warn}
                    <div style="display:flex;gap:.3rem">
                        <button type="submit" class="ui-btn" style="flex:1">Save</button>
                        <button type="button" class="ui-btn" hx-post="{_u("prompts",pid,"encode")}" hx-target="#img-prompt-panel" hx-swap="innerHTML">
                        {"Re-encode" if prompt["status"]!="draft" else "Encode"}</button>
                    </div>
                </form>""")

@router.post("/prompts/create")
async def prompts_create(request: Request):
    username = request.state.user.username
    p = _new_prompt(username)
    _save("prompt", p)
    await _ui_state(request, {"selected_prompt_id": p["id"]})
    return HTMLResponse(f'<div id="img-prompts">{_prompts_html(username, p["id"])}</div><div id="img-prompt-panel" hx-swap-oob="innerHTML">{_prompt_editor_html(p)}</div>')

@router.post("/prompts/select")
async def prompts_select(request: Request, id: str = Form(...)):
    await _ui_state(request, {"selected_prompt_id": id})
    username = request.state.user.username
    return HTMLResponse(_prompt_editor_html(_load("prompt", id)) + f'<div id="img-prompts" hx-swap-oob="innerHTML">{_prompts_html(username, id)}</div>')

@router.post("/prompts/{pid}/save")
async def prompts_save(pid: str, request: Request, title: str = Form(""), text: str = Form("")):
    p = _load("prompt", pid)
    username = request.state.user.username
    if not p or p["username"] != username: return HTMLResponse("Not found", status_code=404)
    if p["text"] != text: p["status"] = "draft"
    p["title"] = title.strip() or text[:40] or "Untitled"
    p["text"] = text
    _save("prompt", p)
    return HTMLResponse(_prompt_editor_html(p) + f'<div id="img-prompts" hx-swap-oob="innerHTML">{_prompts_html(username, pid)}</div>')

@router.post("/prompts/{prompt_id}/encode")
async def _encode_prompt(request: Request, prompt_id: str):
    p = _load("prompt", prompt_id)
    if not p: return HTMLResponse("Not found", status_code=404)
    conn = _conn("text_encoder_conn_id")
    if not conn:
        p["status"], p["error"] = "error", "No text encoder connection — set one in Settings"
        _save("prompt", p)
        await _push_prompts(request.state.user.username, prompt_id)
        return HTMLResponse(_prompt_editor_html(p))
    p["status"] = "encoding"
    _save("prompt", p)
    asyncio.create_task(_do_encode(request.state.user.username, prompt_id))
    return HTMLResponse(_prompt_editor_html(p))

async def _do_encode(username: str, pid: str):
    p = _load("prompt", pid)
    if not p: return
    conn = _conn("text_encoder_conn_id")
    cfg = _cfg()
    max_len = int(cfg.get("max_sequence_length", 1024))
    r = await AIM.connections.flux2_encode(conn, p["text"], job_id=pid, max_sequence_length=max_len, hard_truncate=True, force_recompute=True)
    p = _load("prompt", pid)
    if r.get("error"):
        p["status"], p["error"] = "error", r["error"]
    else:
        p["status"], p["error"], p["meta"] = "ready", "", r.get("meta", {})
    _save("prompt", p)
    await _push_prompts(username, pid)
    await WS.send_personal_message(f'<div id="img-prompt-panel" hx-swap-oob="innerHTML">{_prompt_editor_html(p)}</div>', username)

@router.delete("/prompts/{pid}")
async def prompts_delete(pid: str, request: Request):
    p = _load("prompt", pid)
    username = request.state.user.username
    if p and p["username"] == username: _dp("prompt", pid).unlink(missing_ok=True)
    s = await _ui_state(request)
    if s["selected_prompt_id"] == pid: await _ui_state(request, {"selected_prompt_id": ""})
    return HTMLResponse(_prompts_html(username) + f'<div id="img-prompt-panel" hx-swap-oob="innerHTML">{_prompt_editor_html(None)}</div>')

# Generate tab

def _lora_row_html(idx, selected_path="", selected_scale=1.0):
    opts = "".join(f'<option value="{_esc(n)}" {"selected" if n==selected_path else ""}>{_esc(n)}</option>' for n in AIM.connections.list_loras(_lora_dir()))
    return (f"""<div style="display:flex;gap:.3rem;align-items:center">
                    <select name="lora_{idx}" class="module-select" style="flex:2;font-size:.72rem">
                    <option value="">(none)</option>{opts}</select>
                    <input type="number" name="lora_{idx}_scale" value="{selected_scale}" step="0.05" min="0" max="2" class="module-select" style="width:4rem;font-size:.72rem">
                </div>""")

def _generate_form_html(form, prompts, selected_prompt_id, system_status):
    p_opts = (('<option value="" disabled selected>— create a prompt first —</option>' if not prompts else "") + "".join(f"""<option value="{p["id"]}" {"selected" if p["id"]==selected_prompt_id else ""}>{_esc(p["title"])} {"&#x2713;" if p["status"]=="ready" else "&#x25CC;" if p["status"]=="encoding" else ""}</option>""" for p in prompts))
    loras = form.get("loras", [])
    lora_rows = "".join(_lora_row_html(i, (loras[i] or {}).get("path", "") if i < len(loras) else "", (loras[i] or {}).get("scale", 1.0) if i < len(loras) else 1.0) for i in range(3))
    cfg = _cfg()
    offload_now = cfg.get("offload_mode", "none")
    offload_opts = "".join(f'<option value="{v}" {"selected" if v==offload_now else ""}>{l}</option>' for v, l in [("none", "None (fastest, most VRAM)"), ("vae_cpu", "VAE CPU offload (~1 GB freed)"), ("sequential", "Sequential block offload (slowest, max VRAM — overnight renders)")])
    no_conn_html = ('<div style="font-size:.7rem;color:#ffaa44">&#x26A0; No image-gen connection — set one in Settings.</div>' if system_status.get("_no_conn") else "")
    active_loras = system_status.get("active_loras", [])
    lora_active = (f'<div style="font-size:.7rem;color:#ffcc00">Active LoRAs: {", ".join(active_loras)}</div>' if active_loras else "")
    loaded = system_status.get("loaded", False)
    status_dot = f'<span style="color:{"#00ffa2" if loaded else "var(--text_muted)"}">&#x25CF; {"loaded" if loaded else "not loaded"}</span>'
    return (f"""<form id="img-gen-form" hx-post="{_u("generate/submit")}" hx-target="#img-gen-status" style="padding:.75rem;display:flex;flex-direction:column;gap:.5rem;overflow-y:auto;height:100%;box-sizing:border-box">
                    {no_conn_html}{lora_active}
                    <label style="font-size:.72rem;color:var(--text_muted)">
                        Prompt {status_dot}
                        <select name="prompt_id" class="module-select" style="font-size:.8rem" required>{p_opts}</select>
                    </label>
                    <div style="display:flex;gap:.4rem;flex-wrap:wrap">
                        <label style="font-size:.72rem;color:var(--text_muted);flex:1;min-width:6rem">
                            Width
                            <input type="number" name="width" value="{form.get("width", 512)}" step="1" class="module-select" style="font-size:.78rem">
                        </label>
                        <label style="font-size:.7rem;color:var(--text_muted);flex:1;min-width:6rem">
                            Height
                            <input type="number" name="height" value="{form.get("height", 512)}" step="1" class="module-select" style="font-size:.8rem">
                        </label>
                    </div>
                    <div style="display:flex;gap:.4rem;flex-wrap:wrap">
                        <label style="font-size:.72rem;color:var(--text_muted);flex:1">
                            Steps
                            <input type="number" name="steps" value="{form.get("steps", 4)}" class="module-select" style="font-size:.8rem">
                        </label>
                        <label style="font-size:.72rem;color:var(--text_muted);flex:1">
                            CFG
                            <input type="number" name="cfg" value="{form.get("cfg", 1.0)}" step="0.1" class="module-select" style="font-size:.8rem">
                        </label>
                        <label style="font-size:.72rem;color:var(--text_muted);flex:1">
                            Shift
                            <input type="number" name="shift" value="{form.get("shift", 1.0)}" step="0.1" class="module-select" style="font-size:.8rem">
                        </label>
                    </div>
                    <div style="display:flex;gap:.4rem;flex-wrap:wrap">
                        <label style="font-size:.72rem;color:var(--text_muted);flex:1">
                            Seed (-1=random)
                            <input type="number" name="seed" value="{form.get("seed", -1)}" class="module-select" style="font-size:.78rem">
                        </label>
                        <label style="font-size:.72rem;color:var(--text_muted);flex:1">
                            Batch
                            <input type="number" name="batch" value="{form.get("batch", 1)}" min="1" class="module-select" style="font-size:.8rem">
                        </label>
                    </div>
                    <label style="font-size:.7rem;color:var(--text_muted)">
                        Output prefix
                        <input type="text" name="output_prefix" value="{_esc(form.get("output_prefix", "img"))}" class="module-select" style="font-size:.8rem">
                    </label>
                    <label style="font-size:.7rem;color:var(--text_muted)">
                        Memory offload
                        <select name="offload_mode" class="module-select" style="font-size:.8rem">{offload_opts}</select>
                    </label>
                    <details class="glass" style="padding:.5rem">
                        <summary style="cursor:pointer;font-size:.7rem;color:var(--text_muted);list-style:none">LoRAs (up to 3)</summary>
                        <div style="display:flex;flex-direction:column;gap:.3rem;margin-top:.4rem">{lora_rows}</div>
                    </details>
                    <details class="glass" style="padding:.5rem">
                        <summary style="cursor:pointer;font-size:.7rem;color:var(--text_muted);list-style:none">Sequence / GIF</summary>
                        <div style="display:flex;gap:.4rem;margin-top:.3rem;align-items:center;flex-wrap:wrap">
                        <label style="display:flex;align-items:center;gap:.3rem;font-size:.8rem;flex-shrink:0">
                            <input type="checkbox" name="is_sequence" value="1">
                            Enable GIF
                        </label>
                        <label style="font-size:.7rem;color:var(--text_muted);flex:1">
                            Frames
                            <input type="number" name="num_frames" value="8" min="2" max="120" class="module-select" style="font-size:.8rem">
                        </label>
                        <label style="font-size:.7rem;color:var(--text_muted);flex:1">
                            Frame Strength
                            <input type="number" name="frame_strength" value="0.35" step="0.05" min="0" max="1" class="module-select" style="font-size:.8rem">
                        </label>
                    </div>
                </details>
                <button type="submit" class="button" style="margin-top:.3rem">&#x25B6; Queue</button>
                <div id="img-gen-status" style="font-size:.73rem;min-height:1rem;color:var(--text_muted)"></div>
            </form>""")

@router.post("/generate/submit")
async def generate_submit(request: Request):
    f = await request.form()
    prompt = _load("prompt", f.get("prompt_id", ""))
    if not prompt: return HTMLResponse('<span style="color:#ff5f5f">No prompt selected</span>')
    if prompt["status"] != "ready": return HTMLResponse('<span style="color:#ff5f5f">Prompt not encoded yet</span>')
    loras = [{"path": f.get(f"lora_{i}", ""), "scale": float(f.get(f"lora_{i}_scale", 1.0) or 1.0)} for i in range(3) if f.get(f"lora_{i}", "")]
    is_sequence = f.get("is_sequence") == "1"
    base_form = {"width": int(f.get("width", 512)), "height": int(f.get("height", 512)), "steps": int(f.get("steps", 4)), "cfg": float(f.get("cfg", 1.0)), "shift": float(f.get("shift", 1.0)), "seed": int(f.get("seed", -1)), "batch": (max(1, int(f.get("batch", 1))) if not is_sequence else 1), "output_prefix": f.get("output_prefix", "img").strip() or "img", "loras": loras, "offload_mode": f.get("offload_mode", "none")}
    if is_sequence:
        base_form["num_frames"] = max(2, min(int(f.get("num_frames", 8) or 8), 120))
        base_form["frame_strength"] = float(f.get("frame_strength", 0.35) or 0.35)
    await _ui_state(request, {"form": base_form})
    for i in range(base_form["batch"]):
        job_seed = base_form["seed"] if base_form["seed"] < 0 else base_form["seed"] + i
        job = {"id": f"job_{uuid.uuid4().hex[:10]}", "username": request.state.user.username, "kind": "sequence" if is_sequence else "txt2img", "status": "queued", "prompt_id": prompt["id"], "params": {**base_form, "seed": job_seed}, "batch_index": i, "created": datetime.utcnow().isoformat()}
        _save("job", job)
    _ensure_worker()
    label = "GIF sequence" if is_sequence else f"{base_form['batch']} job(s)"
    return HTMLResponse(f'<span style="color:var(--accent)">&#x2713; Queued {label}</span>')

# --- Inpainting ---

def _inpaint_panel_html(prompts, selected_prompt_id):
    """Canvas panel only — controls live in the bottom toolbar."""
    return (f"""<div style="display:flex;flex-direction:column;height:100%;overflow:hidden">
                    <details style="border-bottom:var(--border-thick) solid var(--border); flex-shrink:0">
                        <summary style="cursor:pointer; font-size:.7rem; color:var(--text_muted); list-style:none">Select base image</summary>
                        <div style="max-height:20rem;overflow-y:auto">{_base_picker.render_shell(include_css=False)}</div>
                    </details>
                    <div style="flex:1;overflow:auto;padding:.75rem;position:relative;text-align:center">
                        <div id="img-inpaint-canvas-wrap" style="position:relative;display:inline-block;max-width:100%">
                            <canvas id="img-base-canvas" style="display:block;max-width:100%;background:var(--surface);position:relative;z-index:1"></canvas>
                            <canvas id="img-mask-canvas" style="position:absolute;top:0;left:0;max-width:100%;opacity:.6;cursor:crosshair;z-index:2"></canvas>
                        </div>
                        <div id="img-inpaint-base-hint" style="margin-top:.4rem; font-size:.7rem;color:var(--text_muted)">Select an image above</div>
                    </div>
                </div>""")

def _bottom_toolbar_generate_html(sequence_state: dict):
    return f"""<div id="img-seq-fields" style="display:flex;gap:.5rem;flex-wrap:wrap;align-items:center;padding:.4rem .6rem">
                   <span style="font-size:.7rem;color:var(--text_muted);text-transform:uppercase">Assemble GIF</span>
                   <input type="text" name="sequence_dir" value="{_esc(sequence_state.get("sequence_dir",""))}" placeholder="sequence name" class="module-select" style="flex:1;min-width:8rem;font-size:.7rem">
                   <input type="number" name="frame_ms" value="{sequence_state.get("frame_ms",120)}" min="10" max="5000" class="module-select" style="width:5rem;font-size:.7rem" title="ms/frame"><label style="font-size:.7rem;color:var(--text_muted);display:flex;align-items:center;gap:.2rem">
                   <input type="checkbox" name="loop" value="1" {"checked" if sequence_state.get("loop",True) else ""}> loop</label>
                   <button class="button" hx-post="/im/in" hx-target="body" hx-swap="none" hx-include="#img-seq-fields" hx-vals='{json.dumps({"type": "image_assemble_gif", "branch": IM.branch_id, "lvl": 2})}'>&#x25B6; Assemble</button>
                   <span id="img-assemble-status" style="font-size:.7rem;color:var(--text_muted);width:100%"></span>
               </div>"""

def _bottom_toolbar_inpaint_html(prompts, selected_prompt_id, selected_mask=""):
    p_opts = "".join(f'<option value="{p["id"]}" {"selected" if p["id"]==selected_prompt_id else ""}>{_esc(p["title"])} {"&#x2713;" if p["status"]=="ready" else ""}</option>' for p in prompts)
    mask_status = f"&#x2713; Using saved mask: {_esc(selected_mask)}" if selected_mask else "No mask saved yet — draw then Save Mask, or pick a saved one above."
    return f"""<div style="display:flex;flex-direction:column;gap:.05rem;padding:.05rem .2rem">
                   <input type="hidden" id="img-base-path" value="">
                   <div style="display:flex;gap:.05rem;flex-wrap:wrap;align-items:center">
                       <select id="img-inpaint-prompt" name="prompt_id" form="img-inpaint-form" class="module-select" style="font-size:.7rem;flex:1;min-width:8rem">{p_opts}</select>
                       <label style="font-size:.7rem;color:var(--text_muted);display:flex;align-items:center;gap:.05rem">Brush<input type="range" min="5" max="200" value="40" oninput="ImgMask.setBrush(this.value)" style="width:5rem"></label>
                       <input type="number" id="img-inpaint-strength" name="strength" form="img-inpaint-form" value="1" step="0.05" min="0.05" max="1" class="module-select" style="width:3.5rem" title="Strength">
                       <input type="number" id="img-inpaint-steps" name="steps" form="img-inpaint-form" value="20" min="4" max="50" class="module-select" style="width:3.5rem" title="Steps (inpaint needs more than txt2img)">
                       <button type="button" class="cm-qbtn" onclick="ImgMask.clear()">Clear</button>
                       <button type="button" class="ui-btn" style="flex-shrink:0" onclick="imgSaveMask('{IM.branch_id}')">&#x1F4BE; Save Mask</button>
                       <form id="img-inpaint-form" hx-post="{_u("inpaint/submit")}" hx-target="#img-inpaint-status" style="display:flex;gap:.05rem;align-items:center">
                           <input type="hidden" name="image_path" id="img-base-path-mirror" form="img-inpaint-form" value="">
                           <input type="hidden" name="mask_path" id="img-mask-path" form="img-inpaint-form" value="">
                           <button type="button" class="button" onclick="imgQueueInpaint('{IM.branch_id}')">&#x25B6; Queue Inpaint</button>
                       </form>
                   </div>
                   <div id="img-mask-debug" style="font-size:.65rem;color:var(--text_muted);font-family:var(--font-mono)">save-mask: idle (never clicked)</div>
                   <details style="border-top:var(--border-thick) solid var(--border);padding-top:.3rem">
                       <summary style="cursor:pointer;font-size:.7rem;color:var(--text_muted);list-style:none">Or use a previously saved mask</summary>
                       <div style="max-height:20rem;overflow-y:auto">{_mask_picker.render_shell(include_css=False)}</div>
                   </details>
                   <details style="border-top:var(--border-thick) solid var(--border);padding-top:.3rem">
                       <summary style="cursor:pointer;font-size:.7rem;color:var(--text_muted);list-style:none">Reference image (optional)</summary>
                       {mask_status}
                       <div style="max-height:20rem;overflow-y:auto">{_ref_picker.render_shell(include_css=False)}</div>
                   </details>
                   <div id="img-inpaint-status" style="font-size:.7rem;color:var(--text_muted);width:100%"></div>
                   <span id="img-mask-status" style="font-size:.7rem;color:var(--text_muted);flex:1">No mask saved yet — draw then Save Mask, or pick a saved one above.</span>
               </div>"""

def _data_uri(rel: str, fm) -> str:
    try:
        p = fm.resolve(rel)
        mime = mimetypes.guess_type(p.name)[0] or "application/octet-stream"
        return f"data:{mime};base64,{base64.b64encode(p.read_bytes()).decode()}"
    except Exception:
        return ""

@router.post("/inpaint/submit")
async def inpaint_submit(request: Request):
    """Queue an inpaint job using a saved mask file path."""
    f = await request.form()
    username = request.state.user.username
    image_path = f.get("image_path", "").strip()
    mask_path = f.get("mask_path", "").strip()
    reference_path = f.get("reference_path", "").strip()
    prompt_id = f.get("prompt_id", "")
    strength = float(f.get("strength", 0.75) or 0.75)
    if not image_path: return HTMLResponse('<span style="color:#ff5f5f">Select a base image first</span>')
    if not mask_path: return HTMLResponse('<span style="color:#ff5f5f">Save a mask first (draw then click &#x1F4BE; Save Mask)</span>')
    prompt = _load("prompt", prompt_id)
    if not prompt or prompt["status"] != "ready": return HTMLResponse('<span style="color:#ff5f5f">Select a ready (encoded) prompt</span>')
    form = dict((await _ui_state(request))["form"])
    form["steps"] = int(f.get("steps", 20) or 20)
    # Match geometry to base image
    try:
        with PILImage.open(_output_dir() / image_path) as im:
            iw, ih = im.size
        form["width"] = max(32, (iw // 32) * 32)
        form["height"] = max(32, (ih // 32) * 32)
    except Exception as e:
        pass  # use form defaults
    job = {"id": f"job_{uuid.uuid4().hex[:10]}", "username": username, "kind": "inpaint", "status": "queued", "prompt_id": prompt_id, "image_path": image_path, "mask_path": mask_path, "reference_path": reference_path, "params": {**form, "strength": strength}, "created": datetime.utcnow().isoformat()}
    _save("job", job)
    _ensure_worker()
    return HTMLResponse('<span style="color:var(--accent)">&#x2713; Inpaint queued</span>')

# Queue worker + progress push

async def _queue_worker():
    while True:
        job = _next_queued_job()
        if not job:
            await asyncio.sleep(2)
            continue
        await _run_job(job)

def _next_queued_job():
    jobs = [j for j in _list("job") if j["status"] == "queued"]
    return sorted(jobs, key=lambda j: j["created"])[0] if jobs else None

async def _poll_progress(username: str, stop_evt: asyncio.Event):
    conn = _conn("image_gen_conn_id")
    if not conn: return
    try:
        while not stop_evt.is_set():
            await asyncio.sleep(1.5)
            if stop_evt.is_set(): break
            try:
                async with httpx.AsyncClient(timeout=httpx.Timeout(3.0)) as c:
                    r = await c.get(f"{AIM.connections._base(conn)}/system/progress")
                    if r.status_code != 200: continue
                    prog = r.json()
            except Exception:
                continue
            step = prog.get("step", 0)
            total = max(prog.get("total", 1), 1)
            pct = prog.get("pct", int(step * 100 / total))
            if prog.get("status") != "running": continue
            filled = int(pct / 10)
            bar = "&#x2588;" * filled + "&#x2591;" * (10 - filled)
            html = (f"""<div id="img-progress-bar" hx-swap-oob="innerHTML"><span style="color:#ffcc00">{bar}</span> step {step}/{total} ({pct}%)</div>""")
            await WS.send_personal_message(html, username)
    finally:
        await WS.send_personal_message('<div id="img-progress-bar" hx-swap-oob="innerHTML"></div>', username)

async def _run_job(job):
    job["status"] = "running"
    job["started"] = datetime.utcnow().isoformat()
    _save("job", job)
    await _push_job(job)
    conn = _conn("image_gen_conn_id")
    if not conn:
        job["status"] = "error"
        job["error"] = "No image-gen connection"
        job["finished"] = datetime.utcnow().isoformat()
        _save("job", job)
        await _push_job(job)
        return
    cfg = _cfg()
    p = job["params"]
    prompt_doc = _load("prompt", job.get("prompt_id", ""))
    payload = {"prompt": (prompt_doc or {}).get("text", ""),
               "embed_job_id": job.get("prompt_id", ""),"width": p["width"], "height": p["height"],
               "steps": p["steps"], "guidance_scale": p["cfg"],
               "shift": p["shift"], "seed": p["seed"],
               "model_path": cfg.get("model_name", "flux-2-klein-9b-Q6_K.gguf"),
               "vae_path": cfg.get("vae_name", "flux2"),
               "vae_tiling": cfg.get("vae_tiling", True),
               "offload_mode": p.get("offload_mode", cfg.get("offload_mode", "none")),
               "loras": [{"path": l["path"], "scale": l["scale"], "name": Path(l["path"]).stem} for l in p.get("loras", []) if l.get("path")]}
    if job["kind"] == "inpaint":
        payload["image_path"] = job.get("image_path", "")
        payload["mask_image"] = job.get("mask_path", "")  # file path, not inline data
        payload["strength"] = p.get("strength", 0.75)
        if job.get("reference_path"):
            with open(_output_dir() / job["reference_path"], "rb") as rf:
                payload["reference_image"] = "data:image/png;base64," + base64.b64encode(rf.read()).decode()
    # Start progress polling
    stop_poll = asyncio.Event()
    poll_task = asyncio.create_task(_poll_progress(job["username"], stop_poll))
    try:
        if job["kind"] == "sequence":
            payload["num_frames"] = p.get("num_frames", 8)
            payload["frame_strength"] = p.get("frame_strength", 0.35)
            payload["loop"] = True
            r = await AIM.connections.flux2_generate_sequence(conn, payload)
        else:
            r = await AIM.connections.flux2_generate(conn, payload)
        if r.get("error"): raise RuntimeError(r["error"])
        result_file = r["file_name"]
        if not (_output_dir() / result_file).exists(): raise RuntimeError(f"Node reported '{result_file}' but file is not in outputs")
        job["status"] = "done"
        job["result_file"] = result_file
    except Exception as e:
        job["status"] = "error"
        job["error"] = str(e)
    finally:
        stop_poll.set()
        poll_task.cancel()
    job["finished"] = datetime.utcnow().isoformat()
    _save("job", job)
    await _push_job(job)

async def _push_job(job): await WS.send_personal_message(f'<div id="img-queue-status" hx-swap-oob="innerHTML">{_queue_status_html()}</div>', job["username"])

# Right panel + system actions

def _queue_status_html():
    running = [j for j in _list("job") if j["status"] == "running"]
    queued = [j for j in _list("job") if j["status"] == "queued"]
    if running: return (f"""<div style="font-size:.73rem"><span style="color:#ffcc00">&#x25CC; running</span>{running[0]["kind"]} — {running[0]["id"]}</div><div style="font-size:.68rem;color:var(--text_muted)">{len(queued)} queued</div>""")
    return f'<div style="font-size:.73rem;color:var(--text_muted)">Idle — {len(queued)} queued</div>'

def _recent_jobs_html():
    jobs = sorted(_list("job"), key=lambda j: j.get("created",""), reverse=True)[:15]
    rows = "".join(f"""<div style="font-size:.66rem;padding:.2rem 0;border-bottom:var(--border-thick) solid var(--border)"><span style="color:{"#00ffa2" if j["status"]=="done" else "#ff5f5f" if j["status"]=="error" else "#ffcc00" if j["status"]=="running" else "var(--text_muted)"}">{j["status"]}</span> {j["kind"]} {_esc(j.get("error","")[:50])}</div>""" for j in jobs)
    return rows or '<div style="color:var(--text_muted);font-size:.7rem">No jobs yet.</div>'

async def _right_panel_html():
    conn = _conn("image_gen_conn_id")
    status = await AIM.connections.flux2_system_status(conn) if conn else {}
    loaded = status.get("loaded", False)
    active_loras = status.get("active_loras", [])
    last_error = status.get("last_error", "")
    log_tail = status.get("log_tail", "")
    prog = status.get("progress", {})
    no_conn = not bool(conn)
    lora_html = (f'<div style="font-size:.7rem; color:#ffcc00; margin-top:.2rem">LoRAs: {", ".join(active_loras)}</div>' if active_loras else "")
    err_html = (f'<div style="font-size:.7rem; color:#ff5f5f; margin-top:.2rem">&#x26A0; {_esc(last_error[:120])}</div>' if last_error else "")
    no_conn_html = ('<div style="font-size:.7rem; color:#ffaa44; margin-top:.2rem">&#x26A0; No connection — set one in Settings.</div>' if no_conn else "")
    # Progress bar from last status call
    prog_step = prog.get("step", 0)
    prog_total = max(prog.get("total", 1), 1)
    prog_pct = prog.get("pct", int(prog_step * 100 / prog_total))
    prog_status = prog.get("status", "idle")
    prog_html = '<div id="img-progress-bar" style="font-size:.68rem;margin-top:.3rem;font-family:var(--font-mono)">'
    if prog_status == "running" and prog_total > 1:
        filled = int(prog_pct / 10)
        bar = "&#x2588;" * filled + "&#x2591;" * (10 - filled)
        prog_html += f'<span style="color:#ffcc00">{bar}</span> {prog_step}/{prog_total} ({prog_pct}%)'
    prog_html += '</div>'
    return (f"""<div style="display:flex;flex-direction:column;height:100%;overflow:hidden">
                    <div style="padding:.5rem;border-bottom:var(--border-thick) solid var(--border);flex-shrink:0">
                        <div style="font-size:.7rem; text-transform:uppercase; color:var(--text_muted); margin-bottom:.3rem">Image Node</div>
                        <div style="font-size:.7rem">{"<span style=color:#00ffa2>&#x25CF; loaded</span>" if loaded else "<span style=color:var(--text_muted)>&#x25CF; not loaded</span>"}</div>
                        {lora_html}
                        {err_html}
                        {no_conn_html}
                        {prog_html}
                        <div style="display:flex;gap:.25rem;margin-top:.35rem;flex-wrap:wrap">
                            <button class="cm-qbtn" hx-post="{_u("system/load")}" hx-target="#img-right-panel" hx-swap="innerHTML">Load</button>
                            <button class="cm-qbtn" hx-post="{_u("system/unload")}" hx-target="#img-right-panel" hx-swap="innerHTML">Unload</button>
                            <button class="cm-qbtn" style="color:#ff5f5f" hx-post="{_u("system/stop")}" hx-target="#img-right-panel" hx-swap="innerHTML">Stop</button>
                            <button class="cm-qbtn" hx-get="{_u("system/status_check")}" hx-target="#img-right-panel" hx-swap="innerHTML" title="Refresh">&#x21BA;</button>
                            <button class="cm-qbtn" hx-post="{_u("system/clear_error")}" hx-target="#img-right-panel" hx-swap="innerHTML" title="Clear error">Clr</button>
                        </div>
                    </div>
                    <div id="img-queue-status" style="padding:.5rem;border-bottom:var(--border-thick) solid var(--border);flex-shrink:0">{_queue_status_html()}</div>
                    <div style="flex:1;overflow-y:auto;padding:.5rem;display:flex;flex-direction:column;gap:.4rem">
                        <div style="font-size:.7rem;text-transform:uppercase;color:var(--text_muted)">Recent Jobs</div>
                        {_recent_jobs_html()}
                        {"<details><summary style=cursor:pointer;font-size:.68rem;color:var(--text_muted);list-style:none>Node Logs</summary><pre style=font-size:.58rem;white-space:pre-wrap;max-height:10rem;overflow-y:auto>" + _esc(log_tail[-2000:]) + "</pre></details>" if log_tail else ""}
                    </div>
                </div>""")

@router.post("/system/{action}")
async def system_action(action: str):
    conn = _conn("image_gen_conn_id")
    if conn:
        cfg = _cfg()
        if action == "load": await AIM.connections.flux2_system_load(conn, cfg.get("model_name","flux-2-klein-9b-Q6_K.gguf"), cfg.get("vae_name","flux2"))
        elif action == "unload": await AIM.connections.flux2_system_unload(conn)
        elif action == "stop": await AIM.connections.flux2_system_stop(conn)
        elif action == "clear_error":
            async with httpx.AsyncClient(timeout=httpx.Timeout(connect=3.0, read=5.0, write=3.0, pool=3.0)) as c:
                try: await c.post(f"{AIM.connections._base(conn)}/system/clear_error")
                except Exception: pass
    return HTMLResponse(await _right_panel_html())

# --- Settings ---

def _settings_html():
    g = _SETTINGS.get_group("defaults")
    return f"""<div style="padding:1.5rem; max-width:36rem; height:100%; overflow-y:auto; box-sizing:border-box;">
        <h2 style="margin:0 0 1rem;font-size:1rem">Image Settings</h2>
        <form hx-post="{_u("settings/save")}" hx-target="#img-settings-status" style="display:flex;flex-direction:column;gap:.5rem">
            {g.render(g.load())}
            <button type="submit" class="button">Save</button>
            <div id="img-settings-status" style="font-size:.75rem;min-height:1rem"></div>
        </form>
        <hr style="margin:1.5rem 0;border-color:var(--border)">
        <div style="display:flex;align-items:center;gap:.6rem">
            <button class="ui-btn" hx-post="{_u("jobs/clear")}" hx-target="#img-jobs-clear-status" hx-confirm="Delete all job history records? Outputs on disk are not affected.">Clear job records</button>
            <span id="img-jobs-clear-status" style="font-size:.75rem;color:var(--text_muted)"></span>
        </div>
    </div>"""

@router.post("/jobs/clear")
async def jobs_clear(request: Request):
    n = 0
    for f in JOB_RECORDS_DIR.glob("*.json"):
        f.unlink(missing_ok=True)
        n += 1
    return HTMLResponse(f'<span style="color:var(--accent)">&#x2713; Cleared {n} job record(s)</span>')

@router.post("/settings/save")
async def settings_save(request: Request):
    _SETTINGS.get_group("defaults").save(dict(await request.form()))
    return HTMLResponse('<span style="color:var(--accent)">&#x2713; Saved</span>')

# --- Bottom toolbar ---

async def _bottom_toolbar_html(request, active: str, prompts, selected_prompt_id, selected_mask):
    if active == "inpaint": return _bottom_toolbar_inpaint_html(prompts, selected_prompt_id, selected_mask)
    if active == "generate":
        seq = await ENV["get_state"](request, scope="user", namespace="image_sequence") or {}
        return _bottom_toolbar_generate_html(seq)
    return ""

def _bottom_toolbar_generate_html(sequence_state: dict):
    return (f"""<div style="display:flex;gap:.5rem;flex-wrap:wrap;align-items:center;padding:.4rem .6rem">
                    <span style="font-size:.7rem;color:var(--text_muted);text-transform:uppercase">Assemble GIF</span>
                    <input type="text" name="sequence_dir" value="{_esc(sequence_state.get("sequence_dir",""))}" placeholder="sequence name" class="module-select" style="flex:1;min-width:8rem;font-size:.7rem" hx-post="/im/in" hx-target="body" hx-swap="none" hx-trigger="change" hx-vals='{{"type":"image_assemble_gif_field","branch":"{IM.branch_id}","lvl":2,"field":"sequence_dir"}}' hx-include="this">
                    <input type="number" id="img-frame-ms" value="{sequence_state.get("frame_ms",120)}" min="10" max="5000" class="module-select" style="width:5rem;font-size:.7rem" title="ms/frame">
                    <label style="font-size:.68rem;color:var(--text_muted);display:flex;align-items:center;gap:.2rem">
                    <input type="checkbox" id="img-loop" {"checked" if sequence_state.get("loop",True) else ""}> loop</label>
                    <button class="button" hx-post="/im/in" hx-target="body" hx-swap="none" hx-vals='js:{{"type":"image_assemble_gif","branch":"{IM.branch_id}", "lvl":2, "sequence_dir":document.querySelector("[name=sequence_dir]").value, "frame_ms":document.getElementById("img-frame-ms").value, "loop":document.getElementById("img-loop").checked?"1":""}}'>&#x25B6; Assemble</button>
                    <span id="img-assemble-status" style="font-size:.7rem;color:var(--text_muted);width:100%"></span>
                </div>""")

async def _im_assemble_gif_field(request, payload, imr):
    field = payload.get("field")
    if field:
        seq = await ENV["get_state"](request, scope="user", namespace="image_sequence") or {}
        seq[field] = payload.get(field, "")
        await ENV["set_state"](request, seq, scope="user", namespace="image_sequence")
    return imr

async def _im_assemble_gif(request, payload, imr):
    seq_dir = (payload.get("sequence_dir") or "").strip()
    frame_ms = int(payload.get("frame_ms") or 120)
    loop = payload.get("loop") == "1"
    await ENV["set_state"](request, {"sequence_dir": seq_dir, "frame_ms": frame_ms, "loop": loop}, scope="user", namespace="image_sequence")
    if not seq_dir:
        imr.oob('<span style="color:#ff5f5f">No sequence name</span>', "img-assemble-status")
        return imr
    out_name, n = assemble_gif(seq_dir, frame_ms=frame_ms, loop=loop)
    if not out_name: imr.oob(f'<span style="color:#ff5f5f">Need ≥2 frames, found {n}</span>', "img-assemble-status")
    else: imr.oob(f'<span style="color:var(--accent)">&#x2713; {n} frames → {out_name}</span>', "img-assemble-status")
    return imr

def assemble_gif(sequence_dir: str, frame_ms: int = 120, fps=None, loop: bool = True, output_name=None):
    safe_dir = re.sub(r'[^\w\-]', '_', sequence_dir)[:64]
    src_dir = os.path.join(_output_dir(), "_sequences", safe_dir)
    if not os.path.isdir(src_dir): return None, 0
    files = sorted(f for f in os.listdir(src_dir) if f.lower().endswith(".png"))
    if len(files) < 2: return None, len(files)
    frames = [PILImage.open(os.path.join(src_dir, f)).convert("RGBA") for f in files]
    duration = int(1000 / fps) if fps else frame_ms
    out_name = output_name or f"{safe_dir}_{int(time.time())}.gif"
    if not out_name.lower().endswith(".gif"): out_name += ".gif"
    out_path = os.path.join(_output_dir(), out_name)
    frames[0].save(out_path, format="GIF", save_all=True, append_images=frames[1:], loop=0 if loop else 1, duration=duration, optimize=False)
    return out_name, len(files)

# --- Tab Rendering---

async def _render_panel(request, state):
    active = state.get("active", "generate")
    s = await _ui_state(request)
    if active == "inpaint": return state, _inpaint_panel_html(_list("prompt"), s["selected_prompt_id"])
    if active == "gallery": return state, f'<div style="height:100%">{_gallery_tool.render_shell()}</div>'
    if active == "settings": return state, _settings_html()
    conn = _conn("image_gen_conn_id")
    status = (await AIM.connections.flux2_system_status(conn)) if conn else {"_no_conn": True}
    return state, f'<div id="img-gen-panel" style="height:100%">{_generate_form_html(s["form"], _list("prompt"), s["selected_prompt_id"], status)}</div>'

@router.get("")
@router.get("/")
async def root(request: Request):
    _ensure_worker()
    username = request.state.user.username
    state = await TM._load(request)
    state, panel_html = await _render_panel(request, state)
    tab_bar = await TM.tab_bar_fn(state, "img-tab-bar", "image", 2, allow_new=False, closable=False)
    s = await _ui_state(request)
    selected = _load("prompt", s["selected_prompt_id"]) if s["selected_prompt_id"] else None
    bottom_inner = await _bottom_toolbar_html(request, state.get("active", "generate"), _list("prompt"), s["selected_prompt_id"], s["selected_mask"])
    left = (f"""<div style="display:flex;flex-direction:column;height:100%;overflow:hidden">
                    <div style="padding:.4rem .5rem;border-bottom:var(--border-thick) solid var(--border); display:flex;align-items:center;gap:.3rem">
                        <span style="font-size:.68rem;text-transform:uppercase;color:var(--text_muted);flex:1">Prompts</span>
                        <button class="btn-icon" hx-post="{_u("prompts/create")}" hx-target="#img-prompts" hx-swap="innerHTML" title="New prompt">+</button>
                     </div>
                     <div id="img-prompts" style="flex:1;overflow-y:auto">{_prompts_html(username, s["selected_prompt_id"])}</div>
                     <div id="img-prompt-panel" style="border-top:var(--border-thick) solid var(--border); max-height:55%;overflow-y:auto">{_prompt_editor_html(selected)}</div>
                 </div>""")
    right = f'<div id="img-right-panel" style="height:100%">{await _right_panel_html()}</div>'
    return ENV["templates"].TemplateResponse(name="base.html", request=request, context={"request": request, "user": request.state.user, "nesting_level": 2, "shell_id": IM.branch_id,
            "toolbars": {"top": UI.toolbar(side="top", content=tab_bar, size="2.5rem", id="img-top", nesting_level=2, start_open=True, locked=True),
                         "bottom": UI.toolbar(side="bottom", content=f'<div id="img-bottom-content">{bottom_inner}</div>', size="14rem", id="img-bottom", nesting_level=2, start_open=False, resizable=True),
                         "left": UI.toolbar(side="left", content=left, size="18rem", id="img-left", nesting_level=2, start_open=False, resizable=True),
                         "right": UI.toolbar(side="right", content=right, size="18rem", id="img-right", nesting_level=2, start_open=False, resizable=True)},
            "content": (f'<div id="img-panel" style="height:100%;overflow:hidden">{panel_html}</div>'), "extra_css": CSS, "extra_script": SCRIPT + BI.IMAGE_GALLERY_JS})

@router.get("/full/{path:path}")
async def serve_full(path: str):
    p = _fm().resolve(path)
    if not p.exists(): raise HTTPException(404)
    return FileResponse(p, headers={"Cache-Control": "public, max-age=3600"})

CSS = """
.img-prompt-item{display:flex;align-items:center;gap:.2rem;padding:.3rem .3rem;cursor:pointer;font-size:.8rem; border-bottom:var(--border-thick) solid var(--border)}
.img-prompt-item:hover{background:var(--accent_dim)}
.img-prompt-item.active{background:var(--glass);border-left:.1rem solid var(--accent)}
.igal-meta-card{padding:.8rem;max-width:22rem; max-height:80vh; overflow-y:auto}
#igal-action-msg{font-size:.7rem;padding:.2rem .4rem}
"""

SCRIPT = """
const ImgMask = (() => {
    let canvas, ctx, drawing = false, brush = 40;
    function init(c) {
        canvas = c; ctx = c.getContext('2d');
        ctx.strokeStyle = 'white'; ctx.lineWidth = brush;
        ctx.lineCap = 'round'; ctx.lineJoin = 'round';
        canvas.onmousedown = e => { drawing = true; ctx.beginPath(); move(e); };
        canvas.onmousemove = e => { if (!drawing) return; const p = pos(e); ctx.lineTo(p.x,p.y); ctx.stroke(); ctx.beginPath(); ctx.moveTo(p.x,p.y); };
        canvas.onmouseup = canvas.onmouseleave = () => drawing = false;
        canvas.ontouchstart = e => { e.preventDefault(); drawing = true; ctx.beginPath(); move(e.touches[0]); };
        canvas.ontouchmove = e => { e.preventDefault(); if (!drawing) return; const p = tpos(e); ctx.lineTo(p.x,p.y); ctx.stroke(); ctx.beginPath(); ctx.moveTo(p.x,p.y); };
        canvas.ontouchend = () => drawing = false;
    }
    function move(e) { const p = pos(e); ctx.moveTo(p.x,p.y); }
    function pos(e)  { const r = canvas.getBoundingClientRect(); return {x:(e.clientX-r.left)*canvas.width/r.width,y:(e.clientY-r.top)*canvas.height/r.height}; }
    function tpos(e) { return pos(e.touches[0]); }
    function setBrush(sz) { brush = parseInt(sz); if(ctx) ctx.lineWidth = brush; }
    function clear() { if(ctx) ctx.clearRect(0,0,canvas.width,canvas.height); }
    return {init, setBrush, clear};
})();

function imgLoadBaseCanvas(dataUri) {
    var baseC = document.getElementById('img-base-canvas'), maskC = document.getElementById('img-mask-canvas');
    if (!baseC || !dataUri) return;
    var img = new Image();
    img.onload = function() {
        baseC.width = maskC.width = img.naturalWidth;
        baseC.height = maskC.height = img.naturalHeight;
        var dispW = Math.min(img.naturalWidth, 680) + 'px';
        baseC.style.width = maskC.style.width = dispW;
        baseC.style.height = maskC.style.height = 'auto';
        baseC.getContext('2d').drawImage(img, 0, 0);
        maskC.getContext('2d').clearRect(0, 0, maskC.width, maskC.height);
        ImgMask.init(maskC);
    };
    img.src = dataUri;
}

function imgLoadMaskCanvas(dataUri) {
    var mc = document.getElementById('img-mask-canvas');
    if (!mc || !mc.width || !dataUri) return;
    var img = new Image();
    img.onload = function() {
        var mctx = mc.getContext('2d');
        mctx.clearRect(0, 0, mc.width, mc.height);
        mctx.drawImage(img, 0, 0, mc.width, mc.height);
    };
    img.src = dataUri;
}

function imgCanvasIsBlank(canvas) {
    var ctx = canvas.getContext('2d');
    var data = ctx.getImageData(0, 0, canvas.width, canvas.height).data;
    for (var i = 3; i < data.length; i += 4) { if (data[i] !== 0) return false; }
    return true;
}

function imgSaveMask(branchId) {
    const dbg = document.getElementById('img-mask-debug');
    function setDbg(msg, color) { if (dbg) { dbg.textContent = 'save-mask: ' + msg; dbg.style.color = color || 'var(--text_muted)'; } }
    return new Promise(function(resolve, reject) {
        const mc = document.getElementById('img-mask-canvas');
        if (!mc || !mc.width) { setDbg('FAILED - no canvas or canvas has zero width (select a base image first)', '#ff5f5f'); reject('no canvas'); return; }
        let dataUrl;
        try {
            const tc = document.createElement('canvas');
            tc.width = mc.width; tc.height = mc.height;
            const cx = tc.getContext('2d');
            cx.fillStyle = 'black'; cx.fillRect(0, 0, tc.width, tc.height);
            cx.drawImage(mc, 0, 0);
            dataUrl = tc.toDataURL('image/png');
        } catch (e) { setDbg('FAILED - canvas extraction threw: ' + e.message, '#ff5f5f'); reject(e); return; }
        const basePath = document.getElementById('img-base-path').value || 'mask';
        htmx.ajax('POST', '/im/in', { values: { type: 'image_save_mask', branch: branchId, lvl: 2, mask_data: dataUrl, base_name: basePath }, swap: 'none' })
            .then(function(){ setDbg('saved', '#00ffa2'); resolve(); })
            .catch(function(e){ setDbg('FAILED - request rejected: ' + e, '#ff5f5f'); reject(e); });
    });
}

function imgQueueInpaint(branchId) {
    const maskPathInput = document.getElementById('img-mask-path');
    const mc = document.getElementById('img-mask-canvas');
    const form = document.getElementById('img-inpaint-form');
    if (!maskPathInput.value && mc && mc.width && !imgCanvasIsBlank(mc)) {
        imgSaveMask(branchId).then(function(){ htmx.trigger(form, 'submit'); });
    } else {
        htmx.trigger(form, 'submit');
    }
}
""" + f'const _IMG_BASE = "{_u()}";'
