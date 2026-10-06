"""motion.py - the Image module's Video and Animatic tabs: clips from text, a start frame, or start + end frames, and storyboards of keyframes animated gap by gap.
Loaded by image.py by file path (the way ai_tools loads image.py itself); init(host) hands it a live view of image.py's globals - settings, job queue, galleries, helpers.
Every node call goes through connection capabilities (model_list, text_to_video, image_to_video, start_end_to_video, video_join), so another video backend is a profile, not code here.
All image paths are relative to the Image output folder, which is also the image node's outputs folder."""
import json, uuid, time, hashlib, asyncio, shutil, math
from pathlib import Path
from datetime import datetime
from PIL import Image as PILImage

H = None                     # image.py's globals, read live
BOARDS_DIR: Path = None
PIPES_FILE = Path(__file__).parent / "animatic_pipelines.json"
FORMATS = ("gif", "webp", "mp4")
SAMPLERS = [("", "(model default)"), ("ddim", "DDIM"), ("cog_ddim", "CogVideoX DDIM"), ("cog_dpm", "CogVideoX DPM")]
OFFLOADS = [("none", "None"), ("sequential", "Sequential blocks (slowest, least memory)")]
MODE_LABEL = {"text_to_video": "text only", "image_to_video": "start frame", "start_end_to_video": "start + end frame"}
PIXEL_KEYS = ("model", "negative_prompt", "num_frames", "steps", "cfg", "scheduler", "seed")   # settings that change a gap's pixels (fps, formats and offload do not)
_MODELS = {"t": 0.0, "conn": None, "data": {}}
_PICKERS: dict = {}
_MOCK_MODELS = {"image": [{"name": "mock-flux.gguf", "modes": ["text_to_image", "image_to_image", "inpaint"]}],
                "video": [{"name": "Mock-Fun-InP", "kind": "inp", "modes": ["text_to_video", "image_to_video", "start_end_to_video"], "defaults": {"width": 256, "height": 160, "num_frames": 9, "fps": 8, "steps": 4, "guidance_scale": 6.0, "scheduler": "ddim"}},
                          {"name": "Mock-I2V", "kind": "i2v", "modes": ["image_to_video"], "defaults": {"width": 256, "height": 160, "num_frames": 9, "fps": 8, "steps": 4, "guidance_scale": 6.0, "scheduler": "cog_ddim"}}]}

def init(host):
    global H, BOARDS_DIR
    H = host
    BOARDS_DIR = H.DATA_DIR / "storyboards"
    (BOARDS_DIR / "_deleted").mkdir(parents=True, exist_ok=True)

def register(IM):
    """The two tabs' intents and their four gallery pickers."""
    for name, fn in (("video_field", _im_video_field), ("video_submit", _im_video_submit), ("video_clear", _im_video_clear), ("models_refresh", _im_models_refresh),
                     ("anim_select", _im_anim_select), ("anim_new", _im_anim_new), ("anim_rename", _im_anim_rename), ("anim_delete", _im_anim_delete), ("anim_field", _im_anim_field), ("anim_settings", _im_anim_settings),
                     ("anim_gap", _im_anim_gap), ("anim_move", _im_anim_move), ("anim_remove", _im_anim_remove), ("anim_from_prompt", _im_anim_from_prompt), ("anim_inpaint", _im_anim_inpaint), ("anim_render", _im_anim_render),
                     ("anim_mask_clear", _im_anim_mask_clear), ("pipelines", _im_pipelines)):
        IM.scripts[f"image_{name}"] = [fn]
    for prefix, fn in (("image_pick_vstart", _pick_vstart), ("image_pick_vend", _pick_vend), ("image_pick_kf", _pick_kf), ("image_pick_kfmask", _pick_kfmask)):
        _PICKERS[prefix] = H.BI.ImageGallery(root_dir=H._output_dir(), IM=IM, intent_prefix=prefix, nesting_level=2, file_manager=H._fm(), select_mode=True, on_select=fn)

# --- Shared helpers ---

def _vals(**kw) -> str: return json.dumps({"branch": H.IM.branch_id, "lvl": 2, **kw}).replace("'", "&#39;")   # hx-vals JSON, safe inside a single-quoted attribute
def _num(v, cast=int): return None if v in (None, "") else cast(float(v))   # blank form field = unset (the model's default)
def _mode(start, end) -> str: return "start_end_to_video" if start and end else "image_to_video" if start else "text_to_video"
def options_html(pairs, selected) -> str: return "".join(f"""<option value="{H._esc(v)}" {"selected" if str(v) == str(selected) else ""}>{H._esc(l)}</option>""" for v, l in pairs)
def _formats(v) -> list: return [f for f in ([v] if isinstance(v, str) else v or []) if f in FORMATS]
def _now() -> str: return datetime.utcnow().isoformat()
def _thumb(rel: str) -> str: return H.thumb_data_uri(H._fm(), rel) if rel else ""
def _note(text: str, color: str = "var(--text_muted)") -> str: return f"""<div style="font-size:.66rem;color:{color}">{text}</div>""" if text else ""
def _field(label: str, inner: str, grow: int = 1) -> str: return f"""<label style="font-size:.7rem;color:var(--text_muted);flex:{grow};min-width:5rem">{label}{inner}</label>"""
def _input(name: str, value, placeholder="", kind="number", step="1") -> str: return f"""<input type="{kind}" name="{name}" value="{H._esc("" if value is None else value)}" placeholder="{H._esc(placeholder)}" step="{step}" class="module-select" style="font-size:.78rem;width:100%">"""
def _select(name: str, pairs, selected) -> str: return f"""<select name="{name}" class="module-select" style="font-size:.78rem;width:100%">{options_html(pairs, selected)}</select>"""
def _textarea(name: str, value: str, rows: int = 3, attrs: str = "", placeholder: str = "") -> str: return f"""<textarea name="{name}" rows="{rows}" placeholder="{H._esc(placeholder)}" class="module-select" style="font-size:.78rem;width:100%;resize:vertical" {attrs}>{H._esc(value)}</textarea>"""
def _btn(label: str, title: str, cls: str = "cm-qbtn", extra: str = "", **vals) -> str: return f"""<button type="button" class="{cls}" title="{H._esc(title)}" hx-post="/im/in" hx-target="body" hx-swap="none" hx-vals='{_vals(**vals)}' {extra}>{label}</button>"""
def _status(imr, html: str, target: str, ok: bool = False): return imr.oob(f"""<span style="color:{"var(--accent)" if ok else "#ff5f5f"}">{html}</span>""", target)
def _details(summary: str, body: str, open_: bool = False) -> str: return f"""<details {"open" if open_ else ""} style="border-top:var(--border-thick) solid var(--border);padding-top:.3rem"><summary style="cursor:pointer;font-size:.72rem;color:var(--text_muted);list-style:none">&#x25B8; {summary}</summary><div style="display:flex;flex-direction:column;gap:.35rem;margin-top:.35rem">{body}</div></details>"""
def _link(rel: str, text: str, download: bool = False) -> str: return f"""<a href="{H._u("full", rel)}" {"download" if download else 'target="_blank"'} style="color:var(--accent);font-size:.68rem">{text}</a>"""
def _picker_html(prefix: str) -> str: return f"""<div style="max-height:18rem;overflow-y:auto">{_PICKERS[prefix].render_shell(include_css=False)}</div>"""

def _vconn():
    """The connection clips go to: the Video Connection setting, else the Image Generation Connection (one node serves both)."""
    cid = H._cfg().get("video_conn_id") or H._cfg().get("image_gen_conn_id")
    return H.AIM.connections.get_conn(cid) if cid else None

def _queue(username: str, kind: str, params: dict, **extra) -> dict:
    job = {"id": f"job_{uuid.uuid4().hex[:10]}", "username": username, "kind": kind, "status": "queued", "params": params, "created": _now(), **extra}
    H._save("job", job)
    H._ensure_worker()
    return job

# --- Node models ---

async def models(force: bool = False) -> dict:
    """The node's /models answer ({image: [...], video: [...]}), cached 30 s per connection; {"error": ...} when there is no connection or the node does not answer."""
    if H._cfg().get("mock_mode"): return _MOCK_MODELS
    conn = _vconn()
    if not conn: return {"error": "no image / video connection - set one in Settings"}
    if force or _MODELS["conn"] != conn["_id"] or time.time() - _MODELS["t"] > 30: _MODELS.update(t=time.time(), conn=conn["_id"], data=await H.AIM.connections.call_capability(conn, "model_list", {}, timeout_s=8.0))
    return _MODELS["data"]

def _model(mdl: dict, name: str) -> dict: return next((m for m in mdl.get("video", []) if m["name"] == name), None) or {} if name else {}
def _model_label(m: dict) -> str: return f"""{m["name"]} - {", ".join(MODE_LABEL.get(x, x) for x in m.get("modes", [])) or "listed only"}"""

def _model_select(name: str, mdl: dict, selected: str, need: str = "") -> str:
    """Video model select; need limits it to models with that mode. Blank = the node picks (the loaded model when it can, else the first that can)."""
    pairs = [("", "(node picks)")] + [(m["name"], _model_label(m)) for m in mdl.get("video", []) if m.get("modes") and (not need or need in m["modes"])]
    if selected and selected not in [v for v, _ in pairs]: pairs.append((selected, f"{selected} (saved, not listed by the node)"))
    err = _note(f"""&#x26A0; node: {H._esc(str(mdl["error"])[:200])}""", "#ffaa44") if mdl.get("error") else ""
    return f"""<div style="display:flex;gap:.3rem;align-items:center">{_select(name, pairs, selected)}{_btn("&#x21BA;", "Ask the node again", type="image_models_refresh")}</div>{err}"""

async def _im_models_refresh(request, payload, imr):
    await models(force=True)
    return await H.TM._push(request, None, imr)

# --- Clips (the node, or placeholders in mock mode) ---

async def _clip(params: dict) -> dict:
    """One clip; the capability follows the frames given. Returns the node's answer (file_name, files, frames_dir, frame_count, seed, width, height, ...)."""
    if H._cfg().get("mock_mode"): return _mock_clip(params)
    conn = _vconn()
    if not conn: raise RuntimeError("no video connection - set the Image Generation (or Video) Connection in Settings")
    r = await H.AIM.connections.call_capability(conn, _mode(params.get("start_image"), params.get("end_image")), {"prompt": "", **{k: v for k, v in params.items() if k == "prompt" or v not in (None, "", [])}}, timeout_s=float(conn.get("values", {}).get("timeout_s") or 14400))   # clips take minutes to hours; a connection saved before its profile had timeout_s gets 4 h
    if r.get("error"): raise RuntimeError(r["error"])
    return r

async def _join(clips: list, fps, formats: list, name: str) -> dict:
    if H._cfg().get("mock_mode"): return _mock_join(clips, fps, formats, name)
    r = await H.AIM.connections.call_capability(_vconn(), "video_join", {"clips": clips, "fps": fps, "formats": formats, "drop_shared": True, "name": name}, timeout_s=900.0)
    if r.get("error"): raise RuntimeError(r["error"])
    return r

def _mock_stem(name: str, seed: int) -> str:
    base = f"""{"".join(c if c.isalnum() or c in "_-" else "_" for c in (name or "vid"))[:40]}_{datetime.now():%Y%m%d_%H%M%S}_{seed}"""
    return next(s for s in (base if i == 1 else f"{base}_{i}" for i in range(1, 100000)) if not (H._output_dir() / "_video" / s).exists())

def _mock_write(frames: list, stem: str, fps) -> dict:
    """Writes frames the way the node does: _video/<stem>/frame_NNNN.png plus <stem>.gif (mock mode writes gif whatever formats were asked)."""
    d = H._output_dir() / "_video" / stem
    d.mkdir(parents=True)
    for i, f in enumerate(frames): f.save(d / f"frame_{i:04d}.png")
    frames[0].save(H._output_dir() / f"{stem}.gif", save_all=True, append_images=frames[1:], duration=round(1000 / float(fps or 8)), loop=0)
    return {"status": "success", "file_name": f"{stem}.gif", "files": {"frames_dir": f"_video/{stem}", "gif": f"{stem}.gif"}, "frames_dir": f"_video/{stem}", "frame_count": len(frames), "fps": fps or 8}

def _mock_clip(p: dict) -> dict:
    m = _model(_MOCK_MODELS, p.get("model") or "Mock-Fun-InP") or _MOCK_MODELS["video"][0]
    d, seed = m["defaults"], 1234 if p.get("seed") in (None, "", -1) else int(p["seed"])
    w, h, n = int(p.get("width") or d["width"]) // 16 * 16, int(p.get("height") or d["height"]) // 16 * 16, int(p.get("num_frames") or d["num_frames"])
    load = lambda rel: PILImage.open(H._output_dir() / rel).convert("RGB").resize((w, h)) if rel else PILImage.new("RGB", (w, h), (seed % 200, 90, 160))
    a, b = load(p.get("start_image")), load(p.get("end_image") or p.get("start_image"))
    return {**_mock_write([PILImage.blend(a, b, i / max(1, n - 1)) for i in range(n)], _mock_stem(p.get("name"), seed), p.get("fps")), "seed": seed, "width": w, "height": h, "model": m["name"], "mode": _mode(p.get("start_image"), p.get("end_image"))}

def _mock_join(clips: list, fps, formats: list, name: str) -> dict:
    frames = [PILImage.open(f).convert("RGB") for k, c in enumerate(clips) for f in sorted((H._output_dir() / c).glob("frame_*.png"))[(1 if k else 0):]]
    if len({f.size for f in frames}) > 1: raise RuntimeError("clips differ in size - render every gap at the same width and height")
    return _mock_write(frames, _mock_stem(name or "joined", int(time.time() * 1000) % 1000), fps)

def _clip_links(files: dict) -> str: return " ".join(_link(files[f], f"&#x2B07; {f}", download=True) for f in FORMATS if files.get(f))

def _preview_html(file_name: str) -> str:
    src = H._u("full", file_name)
    return f"""<video src="{src}" controls loop autoplay muted style="width:100%;max-height:55vh;object-fit:contain"></video>""" if file_name.endswith(".mp4") else f"""<img src="{src}" style="width:100%;max-height:55vh;object-fit:contain">"""   # full width, capped height, aspect kept: inside a flex column an image is stretched, so contain is what keeps it undistorted

def _clip_params(s: dict, prompt: str) -> dict:
    """Saved form values -> the capability payload; blank values are left out so the node applies the model's defaults."""
    seed = _num(s.get("seed"))
    return {"model": s.get("model") or None, "prompt": prompt, "negative_prompt": s.get("negative_prompt") or None, "width": _num(s.get("width")), "height": _num(s.get("height")), "num_frames": _num(s.get("num_frames")), "fps": _num(s.get("fps"), float), "steps": _num(s.get("steps")),
            "guidance_scale": _num(s.get("cfg"), float), "scheduler": s.get("scheduler") or None, "seed": -1 if seed is None else seed, "formats": _formats(s.get("formats")) or ["gif"], "offload_mode": s.get("offload") or None}

def _check(mdl: dict, model: str, mode: str) -> str:
    """Why a model / frames combination cannot run ("" when it can), checked before anything queues."""
    m = _model(mdl, model)
    if m: return "" if mode in m.get("modes", []) else f"""{model} cannot do {MODE_LABEL[mode]}"""
    return f"""no model on the node can do {MODE_LABEL[mode]}""" if not model and mdl.get("video") and not any(mode in x.get("modes", []) for x in mdl["video"]) else ""

# --- Video tab ---

_VIDEO_DEFAULT = {"prompt": "", "negative_prompt": "", "model": "", "start": "", "end": "", "width": "", "height": "", "num_frames": "", "fps": "", "steps": "", "cfg": "", "seed": -1, "scheduler": "", "formats": ["gif"], "offload": "none", "name": "vid"}
_VIDEO_FIELDS = ("prompt", "negative_prompt", "model", "width", "height", "num_frames", "fps", "steps", "cfg", "seed", "scheduler", "offload", "name")

async def _video_state(request, patch: dict = None) -> dict:
    v = {**_VIDEO_DEFAULT, **(await H._ui_state(request)).get("video", {}), **(patch or {})}
    if patch is not None: await H._ui_state(request, {"video": v})
    return v

def _frame_slot_html(which: str, rel: str) -> str:
    """One of the Video tab's frame slots: thumbnail, path, clear button, and the hidden field the form submits."""
    body = f"""<img src="{_thumb(rel)}" style="width:4.5rem;height:4.5rem;object-fit:contain;background:var(--bg);border-radius:.2rem"><span style="flex:1;font-size:.65rem;word-break:break-all">{H._esc(rel)}</span>{_btn("&#x2715;", "Clear", type="image_video_clear", which=which)}""" if rel else """<span style="font-size:.68rem;color:var(--text_muted)">none - pick one from the gallery below</span>"""
    return f"""<input type="hidden" name="{which}" value="{H._esc(rel)}"><div style="display:flex;gap:.4rem;align-items:center;min-height:2rem">{body}</div>"""

def _hints_html(v: dict, mdl: dict) -> str:
    """The mode the frames give, a warning when the chosen model cannot do it, and the model's defaults."""
    mode, m = _mode(v["start"], v["end"]), _model(mdl, v["model"])
    d = m.get("defaults", {})
    warn = "&#x26A0; an end frame needs a start frame" if v["end"] and not v["start"] else f"""&#x26A0; {H._esc(_check(mdl, v["model"], mode))} - pick a model that can, or change the frames""" if _check(mdl, v["model"], mode) else ""
    dflt = f"""Model defaults: {d.get("width")}x{d.get("height")}, {d.get("num_frames")} frames, {d.get("fps")} fps, {d.get("steps")} steps, guidance {d.get("guidance_scale")}, {d.get("scheduler")}. Blank fields use them.""" if d else "Blank fields use the model's defaults."
    return f"""<div style="font-size:.72rem">Mode: <b>{MODE_LABEL[mode]}</b> <span style="color:var(--text_muted)">(follows the frames: none = text, start = start frame, both = start + end)</span></div>{_note(warn, "#ffaa44")}{_note(dflt)}"""

def _video_form_html(v: dict, mdl: dict) -> str:
    d = _model(mdl, v["model"]).get("defaults", {})
    fmt = "".join(f"""<label style="font-size:.72rem;display:flex;align-items:center;gap:.2rem"><input type="checkbox" name="formats" value="{f}" {"checked" if f in v["formats"] else ""}> {f}</label>""" for f in FORMATS)
    return f"""<form id="img-vid-form" hx-post="/im/in" hx-trigger="change" hx-swap="none" hx-vals='{_vals(type="image_video_field")}' onsubmit="return false" style="display:flex;flex-direction:column;gap:.45rem">
                   {_field("Prompt", _textarea("prompt", v["prompt"], 4, placeholder="what happens in the clip"))}
                   {_details("Negative prompt (blank = the node's default)", _textarea("negative_prompt", v["negative_prompt"], 2))}
                   {_field("Model", _model_select("model", mdl, v["model"]))}
                   <span style="font-size:.7rem;color:var(--text_muted)">Start frame</span><div id="img-vid-start">{_frame_slot_html("start", v["start"])}</div>
                   <span style="font-size:.7rem;color:var(--text_muted)">End frame</span><div id="img-vid-end">{_frame_slot_html("end", v["end"])}</div>
                   <div id="img-vid-hints">{_hints_html(v, mdl)}</div>
                   <div style="display:flex;gap:.4rem;flex-wrap:wrap">{_field("Width", _input("width", v["width"], d.get("width", ""), step="16"))}{_field("Height", _input("height", v["height"], d.get("height", ""), step="16"))}{_field("Frames", _input("num_frames", v["num_frames"], d.get("num_frames", "")))}{_field("FPS", _input("fps", v["fps"], d.get("fps", ""), step="any"))}</div>
                   <div style="display:flex;gap:.4rem;flex-wrap:wrap">{_field("Steps", _input("steps", v["steps"], d.get("steps", "")))}{_field("Guidance", _input("cfg", v["cfg"], d.get("guidance_scale", ""), step="any"))}{_field("Seed (-1 = random)", _input("seed", v["seed"]))}</div>
                   <div style="display:flex;gap:.4rem;flex-wrap:wrap">{_field("Sampler", _select("scheduler", SAMPLERS, v["scheduler"]))}{_field("Memory offload", _select("offload", OFFLOADS, v["offload"]))}</div>
                   <div style="display:flex;gap:.6rem;align-items:center;flex-wrap:wrap"><span style="font-size:.7rem;color:var(--text_muted)">Formats</span>{fmt}{_field("Name (file prefix)", _input("name", v["name"], "vid", kind="text"), 2)}</div>
                   <button type="button" class="button" hx-post="/im/in" hx-target="body" hx-swap="none" hx-include="#img-vid-form" hx-vals='{_vals(type="image_video_submit")}'>&#x25B6; Queue clip</button>
                   <div id="img-vid-status" style="font-size:.72rem;min-height:1rem;color:var(--text_muted)"></div>
               </form>"""

def _video_result_html(username: str) -> str:
    """The newest clip job (preview, downloads, what made it) and links to the five before it."""
    jobs = [j for j in H._list("job") if j.get("username") == username and j.get("kind") == "video"][:6]
    if not jobs: return """<div style="color:var(--text_muted);font-size:.75rem;padding:1rem">Finished clips show here.</div>"""
    j, r = jobs[0], jobs[0].get("result", {})
    head = {"queued": "&#x25CC; queued", "running": "&#x25CC; rendering - progress is in the right panel; Stop is there too", "error": f"""<span style="color:#ff5f5f">&#x26A0; {H._esc(str(j.get("error", ""))[:400])}</span>"""}.get(j["status"], "")
    made = f"""{H._esc(r.get("model", ""))} &middot; {MODE_LABEL.get(r.get("mode"), "")} &middot; {r.get("width")}x{r.get("height")} &middot; {r.get("frame_count")} frames @ {r.get("fps")} fps &middot; seed {r.get("seed")} &middot; {r.get("elapsed_seconds", "?")} s"""
    body = f"""{_preview_html(j["result_file"])}{_note(made)}<div>{_clip_links(r.get("files", {}))}</div>""" if j["status"] == "done" and j.get("result_file") else ""
    rest = "".join(f"""<div style="font-size:.66rem">{_link(x["result_file"], H._esc(x["result_file"]))} <span style="color:var(--text_muted)">{H._esc(x.get("params", {}).get("prompt", "")[:60])}</span></div>""" for x in jobs[1:] if x.get("result_file"))
    return f"""<div style="display:flex;flex-direction:column;gap:.4rem;padding:.6rem"><div style="font-size:.72rem">{head}</div>{_note(H._esc(j.get("params", {}).get("prompt", "")[:200]))}{body}{_note("Earlier clips" if rest else "")}{rest}</div>"""

async def video_panel_html(request) -> str:
    v, mdl = await _video_state(request), await models()
    return f"""<div style="display:flex;height:100%;overflow:hidden;gap:.5rem;flex-wrap:wrap">
                   <div style="flex:1 1 22rem;overflow-y:auto;padding:.75rem;box-sizing:border-box;max-height:100%"><div id="img-vid-form-wrap">{_video_form_html(v, mdl)}</div>{_details("Pick the start frame", _picker_html("image_pick_vstart"))}{_details("Pick the end frame", _picker_html("image_pick_vend"))}</div>
                   <div id="img-vid-result" style="flex:1 1 22rem;overflow-y:auto;max-height:100%">{_video_result_html(request.state.user.username)}</div>
               </div>"""

def _form_video(p: dict, v: dict) -> dict: return {**v, **{k: p.get(k, v[k]) for k in _VIDEO_FIELDS}, "formats": _formats(p.get("formats"))}   # frame paths stay as they were: the pickers own them

async def _im_video_field(request, payload, imr):
    """Any change saves the form; a model change re-renders it (the placeholders follow the model's defaults)."""
    old = await _video_state(request)
    v = await _video_state(request, _form_video(payload, old))
    mdl = await models()
    return imr.oob(_video_form_html(v, mdl), "img-vid-form-wrap") if v["model"] != old["model"] else imr.oob(_hints_html(v, mdl), "img-vid-hints")

async def _set_frame(request, imr, which: str, rel: str):
    v = await _video_state(request, {which: rel})
    imr.oob(_frame_slot_html(which, rel), f"img-vid-{which}")
    return imr.oob(_hints_html(v, await models()), "img-vid-hints")

async def _pick_vstart(request, payload, imr): return await _set_frame(request, imr, "start", payload.get("path", ""))
async def _pick_vend(request, payload, imr): return await _set_frame(request, imr, "end", payload.get("path", ""))
async def _im_video_clear(request, payload, imr): return await _set_frame(request, imr, "end" if payload.get("which") == "end" else "start", "")

async def _im_video_submit(request, payload, imr):
    v = await _video_state(request, _form_video(payload, await _video_state(request)))
    mode, username = _mode(v["start"], v["end"]), request.state.user.username
    if not v["prompt"].strip() and mode == "text_to_video": return _status(imr, "Write a prompt first - a text-only clip has nothing else to go on", "img-vid-status")
    if v["end"] and not v["start"]: return _status(imr, "An end frame needs a start frame", "img-vid-status")
    if not (_vconn() or H._cfg().get("mock_mode")): return _status(imr, "No image / video connection - set one in Settings", "img-vid-status")
    if not v["formats"]: return _status(imr, "Pick at least one format", "img-vid-status")
    why = _check(await models(), v["model"], mode)
    if why: return _status(imr, f"&#x26A0; {H._esc(why)} - nothing queued", "img-vid-status")
    _queue(username, "video", {**_clip_params(v, v["prompt"].strip()), "start_image": v["start"] or None, "end_image": v["end"] or None, "name": v["name"] or "vid"})
    imr.oob(_video_result_html(username), "img-vid-result")
    return _status(imr, f"&#x2713; Queued: {MODE_LABEL[mode]}", "img-vid-status", ok=True)

# --- Storyboards ---
# A storyboard is an ordered keyframe list; each adjacent pair is a gap, rendered as one start + end clip. Gap prompts are keyed by the pair ("a|b"), so a prompt follows its two frames when keyframes move.
# A rendered gap is cached under a hash of its two frames (path + file time), its prompt and the pixel-affecting settings: re-animating renders only what changed, and finished gaps survive a later failure or a Stop.
# Element ids carry the board id, so a push for one board never lands in a view of another.

_BOARD_SETTINGS = {"model": "", "width": "", "height": "", "num_frames": "", "fps": "", "steps": "", "cfg": "", "seed": -1, "scheduler": "", "formats": ["gif"], "offload": "none", "negative_prompt": ""}
_BOARD_FIELDS = tuple(k for k in _BOARD_SETTINGS if k != "formats")

def _bp(bid: str) -> Path: return BOARDS_DIR / f"{Path(bid).name}.json"
def _board(bid: str) -> dict: return json.loads(_bp(bid).read_text()) if bid and _bp(bid).exists() else None
def _boards(username: str) -> list: return sorted((b for b in (json.loads(f.read_text()) for f in BOARDS_DIR.glob("*.json")) if b.get("username") == username), key=lambda b: b.get("created", ""))

def _board_save(b: dict):
    b["modified"] = _now()
    _bp(b["id"]).write_text(json.dumps(b, indent=2))

def _new_board(username: str, name: str) -> dict:
    b = {"id": f"sb_{uuid.uuid4().hex[:10]}", "username": username, "name": (name or "").strip() or "storyboard", "motion_prompt": "", "keyframes": [], "gap_prompts": {}, "settings": dict(_BOARD_SETTINGS), "gaps": {}, "renders": [], "created": _now()}
    _board_save(b)
    return b

def _pair(a: str, b: str) -> str: return f"{a}|{b}"
def _gap_prompt(b: dict, i: int) -> str: return (b["gap_prompts"].get(_pair(b["keyframes"][i], b["keyframes"][i + 1])) or "").strip() or b["motion_prompt"].strip()

def _mtime(rel: str) -> float:
    p = H._output_dir() / rel
    return p.stat().st_mtime if p.exists() else 0.0

def _gap_key(a: str, b: str, prompt: str, s: dict, size) -> str: return hashlib.sha256(json.dumps([a, _mtime(a), b, _mtime(b), prompt, [str(s.get(k, "")) for k in PIXEL_KEYS], list(size)]).encode()).hexdigest()[:24]

def _cached(b: dict, key: str) -> dict:
    hit = b["gaps"].get(key)
    return hit if hit and (H._output_dir() / hit["frames_dir"]).is_dir() else None   # a deleted clip folder renders again

def _size(b: dict, mdl: dict) -> tuple:
    """One size for every gap (joined clips must match): the settings when both are set, else the first keyframe's shape at the model's default pixel count, in multiples of 16."""
    s, w, h = b["settings"], _num(b["settings"]["width"]), _num(b["settings"]["height"])
    if w and h: return w // 16 * 16, h // 16 * 16
    d = _model(mdl, s["model"]).get("defaults") or next((m["defaults"] for m in mdl.get("video", []) if "start_end_to_video" in m.get("modes", [])), {"width": 672, "height": 384})
    first = H._output_dir() / b["keyframes"][0] if b["keyframes"] else None
    if not (first and first.exists()): return (w or d["width"]) // 16 * 16, (h or d["height"]) // 16 * 16
    with PILImage.open(first) as im: a = im.width / im.height
    if w: return w // 16 * 16, max(16, round(w / a / 16) * 16)
    if h: return max(16, round(h * a / 16) * 16), h // 16 * 16
    return max(16, round(math.sqrt(d["width"] * d["height"] * a) / 16) * 16), max(16, round(math.sqrt(d["width"] * d["height"] / a) / 16) * 16)

def _gap_state(b: dict, i: int, size) -> str:
    hit = _cached(b, _gap_key(b["keyframes"][i], b["keyframes"][i + 1], _gap_prompt(b, i), b["settings"], size))
    return _link(hit["file"], "&#x2713; rendered") if hit else """<span style="font-size:.6rem;color:var(--text_muted)">not rendered</span>"""

def _kf_card(b: dict, i: int) -> str:
    rel, n = b["keyframes"][i], len(b["keyframes"])
    moves = (_btn("&#x2190;", "Move left", type="image_anim_move", board=b["id"], index=i, dir=-1) if i else "") + (_btn("&#x2192;", "Move right", type="image_anim_move", board=b["id"], index=i, dir=1) if i < n - 1 else "")
    return f"""<div style="display:flex;flex-direction:column;align-items:center;gap:.15rem;width:7.5rem;flex-shrink:0;border:var(--border-thick) solid var(--border);border-radius:var(--radius);padding:.25rem">
                   <img src="{_thumb(rel)}" title="{H._esc(rel)}" style="width:7rem;height:5.5rem;object-fit:contain;background:var(--bg)">
                   <span style="font-size:.6rem;max-width:7rem;overflow:hidden;text-overflow:ellipsis;white-space:nowrap" title="{H._esc(rel)}">{i + 1}. {H._esc(Path(rel).name)}</span>
                   <div style="display:flex;gap:.1rem">{moves}{_btn("&#x2715;", "Remove from the storyboard (the image stays in the gallery)", type="image_anim_remove", board=b["id"], index=i)}</div>
               </div>"""

def _gap_card(b: dict, i: int, size) -> str:
    gap = f"""hx-post="/im/in" hx-trigger="change" hx-swap="none" hx-vals='{_vals(type="image_anim_gap", board=b["id"], index=i)}'"""
    return f"""<div style="display:flex;flex-direction:column;justify-content:center;gap:.2rem;width:9rem;flex-shrink:0">
                   <span style="font-size:.6rem;color:var(--text_muted)">gap {i + 1} &#x2192;</span>
                   {_textarea("prompt", b["gap_prompts"].get(_pair(b["keyframes"][i], b["keyframes"][i + 1]), ""), 3, gap, b["motion_prompt"] or "motion for this gap")}
                   <span id="img-gap-{b["id"]}-{i}">{_gap_state(b, i, size)}</span>
               </div>"""

def _board_job_html(b: dict) -> str:
    """The newest Animate job of this board: its progress while queued or running, the reason when it failed (finished gaps stay cached); nothing once done - the render is listed."""
    j = next((x for x in H._list("job") if x.get("kind") == "animatic" and x.get("params", {}).get("board") == b["id"]), None)
    if not j or j["status"] == "done": return ""
    if j["status"] == "error": return _note(f"""&#x26A0; the last Animate stopped: {H._esc(str(j.get("error", ""))[:300])} - finished gaps are kept, so Animate again renders only the rest""", "#ff5f5f")
    return _note(f"""&#x25CC; Animate {j["status"]}{" - " + H._esc(j["progress"]) if j.get("progress") else ""} (progress and Stop are in the right panel)""", "#ffcc00")

def _board_html(b: dict, mdl: dict) -> str:
    """The part of the Animatic tab that follows the board: default motion prompt, keyframe strip with gap prompts, renders."""
    size, n = _size(b, mdl), len(b["keyframes"])
    strip = "".join(_kf_card(b, i) + (_gap_card(b, i, size) if i < n - 1 else "") for i in range(n)) or """<span style="font-size:.72rem;color:var(--text_muted)">No keyframes yet - add some below.</span>"""
    renders = "".join(f"""<div style="font-size:.68rem;display:flex;gap:.4rem;align-items:center;flex-wrap:wrap">{_link(r["file"], H._esc(r["file"]))} {_clip_links(r.get("files", {}))} <span style="color:var(--text_muted)">{r.get("gaps")} gaps ({r.get("rendered")} rendered, {r.get("reused")} reused) &middot; {r.get("width")}x{r.get("height")} &middot; {H._esc(r.get("created", "")[:16])}</span></div>""" for r in b["renders"]) or _note("None yet.")
    motion = f"""hx-post="/im/in" hx-trigger="change" hx-swap="none" hx-vals='{_vals(type="image_anim_field", board=b["id"])}'"""
    return f"""<div style="display:flex;flex-direction:column;gap:.5rem">
                   {_field("Default motion prompt (every gap left blank uses it)", _textarea("motion_prompt", b["motion_prompt"], 2, motion))}
                   <div style="display:flex;gap:.3rem;overflow-x:auto;padding-bottom:.3rem;align-items:stretch">{strip}</div>
                   {_note(f"Every gap renders at {size[0]}x{size[1]}." if n > 1 else "")}{_board_job_html(b)}
                   <div style="font-size:.7rem;text-transform:uppercase;color:var(--text_muted)">Renders</div>{_preview_html(b["renders"][0]["file"]) if b["renders"] else ""}{renders}
               </div>"""

def _src_wrap(b: dict) -> str: return f"""<div id="img-anim-src-{b["id"]}">{_src_select(b)}</div>"""
def _src_select(b: dict) -> str: return _select("source", [(i, f"{i + 1}. {Path(k).name}") for i, k in enumerate(b["keyframes"])], len(b["keyframes"]) - 1) if b and b["keyframes"] else _note("Add a keyframe first.")

def _mask_slot(rel: str) -> str: return f"""<div style="display:flex;gap:.4rem;align-items:center"><img src="{_thumb(rel)}" style="width:3rem;height:3rem;object-fit:contain;background:var(--bg)"><span style="font-size:.65rem;flex:1;word-break:break-all">{H._esc(rel)}</span>{_btn("&#x2715;", "No mask (repaint the whole frame)", type="image_anim_mask_clear")}</div>""" if rel else _note("No mask: the whole frame is repainted at the strength below. Pick a saved mask (Inpaint tab - Save Mask) to keep part of the frame.")

def _settings_form(b: dict, mdl: dict) -> str:
    s = b["settings"]
    d = _model(mdl, s["model"]).get("defaults", {})
    fmt = "".join(f"""<label style="font-size:.72rem;display:flex;align-items:center;gap:.2rem"><input type="checkbox" name="formats" value="{f}" {"checked" if f in s["formats"] else ""}> {f}</label>""" for f in FORMATS)
    return f"""<form id="img-anim-settings" hx-post="/im/in" hx-trigger="change" hx-swap="none" hx-vals='{_vals(type="image_anim_settings", board=b["id"])}' onsubmit="return false" style="display:flex;flex-direction:column;gap:.4rem">
                   {_field("Model (start + end capable)", _model_select("model", mdl, s["model"], "start_end_to_video"))}
                   <div style="display:flex;gap:.4rem;flex-wrap:wrap">{_field("Width", _input("width", s["width"], "first keyframe's shape", step="16"))}{_field("Height", _input("height", s["height"], "first keyframe's shape", step="16"))}{_field("Frames per gap", _input("num_frames", s["num_frames"], d.get("num_frames", "")))}{_field("FPS", _input("fps", s["fps"], d.get("fps", 8), step="any"))}</div>
                   <div style="display:flex;gap:.4rem;flex-wrap:wrap">{_field("Steps", _input("steps", s["steps"], d.get("steps", "")))}{_field("Guidance", _input("cfg", s["cfg"], d.get("guidance_scale", ""), step="any"))}{_field("Seed (-1 = random)", _input("seed", s["seed"]))}{_field("Sampler", _select("scheduler", SAMPLERS, s["scheduler"]))}{_field("Memory offload", _select("offload", OFFLOADS, s["offload"]))}</div>
                   {_details("Negative prompt (blank = the node's default)", _textarea("negative_prompt", s["negative_prompt"], 2))}
                   <div style="display:flex;gap:.6rem;align-items:center;flex-wrap:wrap"><span style="font-size:.7rem;color:var(--text_muted)">Formats</span>{fmt}<label style="font-size:.72rem;display:flex;align-items:center;gap:.2rem" title="Ignore the cache once and render every gap again"><input type="checkbox" name="force" value="1"> Re-render gaps</label></div>
                   <button type="button" class="button" hx-post="/im/in" hx-target="body" hx-swap="none" hx-include="#img-anim-settings" hx-vals='{_vals(type="image_anim_render", board=b["id"])}'>&#x25B6; Animate</button>
                   <div id="img-anim-status" style="font-size:.72rem;min-height:1rem;color:var(--text_muted)"></div>
               </form>"""

def _add_section(b: dict, mask: str) -> str:
    """Three ways to add a keyframe: from the gallery, from a prompt (Flux), or by inpainting a keyframe. Keyframe jobs run in the normal queue and add themselves when done."""
    ready = [(p["id"], p["title"]) for p in H._list("prompt") if p["status"] == "ready"]
    pick = _select("prompt_id", ready, "") if ready else _note("No encoded prompt yet - write one in the Prompts panel (left) and press Encode.")
    from_prompt = f"""<form id="img-anim-fp" onsubmit="return false" style="display:flex;gap:.3rem;align-items:center;flex-wrap:wrap">{_field("Prompt", pick, 3)}<button type="button" class="ui-btn" hx-post="/im/in" hx-target="body" hx-swap="none" hx-include="#img-anim-fp" hx-vals='{_vals(type="image_anim_from_prompt", board=b["id"])}'>Queue keyframe</button></form>{_note("Uses the Generate tab's size, steps, model and LoRAs; the image is appended when it finishes.")}"""
    inpaint = f"""<form id="img-anim-ip" onsubmit="return false" style="display:flex;flex-direction:column;gap:.3rem">
                      <div style="display:flex;gap:.4rem;flex-wrap:wrap">{_field("Keyframe to inpaint", _src_wrap(b), 2)}{_field("Prompt", pick, 3)}</div>
                      <span style="font-size:.7rem;color:var(--text_muted)">Mask</span><div id="img-anim-mask">{_mask_slot(mask)}</div>
                      <div style="display:flex;gap:.4rem;flex-wrap:wrap;align-items:flex-end">{_field("Strength", _input("strength", 0.85, step="0.05"))}{_field("Steps", _input("steps", 20))}<button type="button" class="ui-btn" hx-post="/im/in" hx-target="body" hx-swap="none" hx-include="#img-anim-ip" hx-vals='{_vals(type="image_anim_inpaint", board=b["id"])}'>Queue inpaint</button></div>
                      {_details("Pick a mask", _picker_html("image_pick_kfmask"))}
                  </form>{_note("The result goes right after the keyframe it came from, so the gap between them animates the change.")}"""
    return _details("Add keyframes", f"""{_details("From the gallery", _picker_html("image_pick_kf"))}{_details("From a prompt (Flux)", from_prompt)}{_details("Inpaint a keyframe (Flux)", inpaint)}<div id="img-anim-add-status" style="font-size:.72rem;min-height:1rem"></div>""", open_=True)

async def animatic_panel_html(request) -> str:
    username, s = request.state.user.username, await H._ui_state(request)
    boards = _boards(username)
    b = next((x for x in boards if x["id"] == s.get("storyboard")), boards[-1] if boards else None)
    mdl = await models()
    bar = f"""<div style="display:flex;gap:.3rem;align-items:center;flex-wrap:wrap">
                  <span style="font-size:.7rem;text-transform:uppercase;color:var(--text_muted)">Storyboard</span>
                  <select name="board" class="module-select" style="font-size:.78rem;min-width:10rem" hx-post="/im/in" hx-trigger="change" hx-swap="none" hx-vals='{_vals(type="image_anim_select")}'>{options_html([(x["id"], x["name"]) for x in boards], b["id"] if b else "")}</select>
                  {_btn("+ New", "New storyboard", "ui-btn", 'hx-prompt="Name for the new storyboard (also the output file prefix):"', type="image_anim_new")}
                  {_btn("Rename", "Rename this storyboard", "cm-qbtn", 'hx-prompt="New name:"', type="image_anim_rename", board=b["id"]) if b else ""}
                  {_btn("Delete", "Delete this storyboard (images and clips stay)", "cm-qbtn", 'hx-confirm="Delete this storyboard? It moves to storyboards/_deleted (move it back to restore); its images and clips stay in the gallery."', type="image_anim_delete", board=b["id"]) if b else ""}
              </div>"""
    if not b: return f"""<div style="padding:.75rem">{bar}{_note("A storyboard is an ordered list of keyframes; Animate renders each gap between two keyframes as a start + end clip and joins them.")}</div>"""
    return f"""<div style="display:flex;flex-direction:column;gap:.6rem;height:100%;overflow-y:auto;padding:.75rem;box-sizing:border-box">
                   {bar}
                   <div id="img-anim-board-{b["id"]}">{_board_html(b, mdl)}</div>
                   {_add_section(b, s.get("anim_mask", ""))}
                   {_details("Animate", _settings_form(b, mdl), open_=True)}
               </div>"""

async def _push_board(username: str, b: dict):
    """Live update of one board's views (keyframe strip, renders, inpaint source list) for its owner."""
    if b: await H.WS.send_personal_message(f"""<div id="img-anim-board-{b["id"]}" hx-swap-oob="innerHTML">{_board_html(b, await models())}</div><div id="img-anim-src-{b["id"]}" hx-swap-oob="innerHTML">{_src_select(b)}</div>""", username)

def _owned(request, payload) -> dict:
    b = _board(payload.get("board", ""))
    return b if b and b["username"] == request.state.user.username else None

async def _board_changed(request, imr, b: dict):
    _board_save(b)
    imr.oob(_board_html(b, await models()), f"""img-anim-board-{b["id"]}""")
    return imr.oob(_src_select(b), f"""img-anim-src-{b["id"]}""")

async def _im_anim_select(request, payload, imr):
    await H._ui_state(request, {"storyboard": payload.get("board", "")})
    return await H.TM._push(request, None, imr)

async def _im_anim_new(request, payload, imr):
    if payload.get("name") is None: return imr   # prompt dismissed
    await H._ui_state(request, {"storyboard": _new_board(request.state.user.username, payload.get("name", ""))["id"]})
    return await H.TM._push(request, None, imr)

async def _im_anim_rename(request, payload, imr):
    b = _owned(request, payload)
    if b and (payload.get("name") or "").strip():
        b["name"] = payload["name"].strip()
        _board_save(b)
    return await H.TM._push(request, None, imr)

async def _im_anim_delete(request, payload, imr):
    b = _owned(request, payload)
    if b: _bp(b["id"]).rename(BOARDS_DIR / "_deleted" / f"""{b["id"]}_{datetime.now():%Y%m%d_%H%M%S}.json""")   # recoverable: moving the file back restores the board; its keyframes, clips and renders never leave the gallery
    return await H.TM._push(request, None, imr)

async def _im_anim_field(request, payload, imr):
    """Default motion prompt: saved; gap states refresh (a blank gap's prompt just changed) without re-rendering the text being edited."""
    b = _owned(request, payload)
    if not b: return imr
    b["motion_prompt"] = payload.get("motion_prompt", b["motion_prompt"])
    _board_save(b)
    size = _size(b, await models())
    for i in range(len(b["keyframes"]) - 1): imr.oob(_gap_state(b, i, size), f"""img-gap-{b["id"]}-{i}""")
    return imr

async def _im_anim_gap(request, payload, imr):
    b, i = _owned(request, payload), int(payload.get("index", -1))
    if not b or not 0 <= i < len(b["keyframes"]) - 1: return imr
    b["gap_prompts"][_pair(b["keyframes"][i], b["keyframes"][i + 1])] = payload.get("prompt", "").strip()
    _board_save(b)
    return imr.oob(_gap_state(b, i, _size(b, await models())), f"""img-gap-{b["id"]}-{i}""")

async def _im_anim_settings(request, payload, imr):
    b = _owned(request, payload)
    if not b: return imr
    b["settings"] = {**b["settings"], **{k: payload.get(k, b["settings"][k]) for k in _BOARD_FIELDS}, "formats": _formats(payload.get("formats"))}
    return await _board_changed(request, imr, b)

async def _im_anim_move(request, payload, imr):
    b, i, d = _owned(request, payload), int(payload.get("index", -1)), int(payload.get("dir", 0))
    if not b or not (0 <= i < len(b["keyframes"]) and 0 <= i + d < len(b["keyframes"])): return imr
    k = b["keyframes"]
    k[i], k[i + d] = k[i + d], k[i]
    return await _board_changed(request, imr, b)

async def _im_anim_remove(request, payload, imr):
    b, i = _owned(request, payload), int(payload.get("index", -1))
    if not b or not 0 <= i < len(b["keyframes"]): return imr
    b["keyframes"].pop(i)
    return await _board_changed(request, imr, b)

async def _pick_kf(request, payload, imr):
    """Gallery pick: appended to the open storyboard (a new one is made when there is none)."""
    username, b = request.state.user.username, _board((await H._ui_state(request)).get("storyboard", ""))
    if not b or b["username"] != username:
        b = (_boards(username) or [None])[-1] or _new_board(username, "storyboard")
        await H._ui_state(request, {"storyboard": b["id"]})
    b["keyframes"].append(payload.get("path", ""))
    return await _board_changed(request, imr, b)

async def _pick_kfmask(request, payload, imr):
    await H._ui_state(request, {"anim_mask": payload.get("path", "")})
    return imr.oob(_mask_slot(payload.get("path", "")), "img-anim-mask")

async def _im_anim_mask_clear(request, payload, imr):
    await H._ui_state(request, {"anim_mask": ""})
    return imr.oob(_mask_slot(""), "img-anim-mask")

def _ready_prompt(pid: str) -> dict:
    p = H._load("prompt", pid) if pid else None
    return p if p and p["status"] == "ready" else None

async def _im_anim_from_prompt(request, payload, imr):
    b, p = _owned(request, payload), _ready_prompt(payload.get("prompt_id", ""))
    if not b or not p: return _status(imr, "Pick an encoded prompt", "img-anim-add-status")
    form = (await H._ui_state(request))["form"]
    _queue(request.state.user.username, "txt2img", {**form, "batch": 1}, prompt_id=p["id"], storyboard=b["id"], insert_at=len(b["keyframes"]))
    return _status(imr, f"""&#x2713; Keyframe queued from "{H._esc(p["title"])}" ({form.get("width")}x{form.get("height")})""", "img-anim-add-status", ok=True)

async def _im_anim_inpaint(request, payload, imr):
    b, p, s = _owned(request, payload), _ready_prompt(payload.get("prompt_id", "")), await H._ui_state(request)
    i = _num(payload.get("source"))
    if not b or i is None or not 0 <= i < len(b["keyframes"]): return _status(imr, "Pick the keyframe to inpaint", "img-anim-add-status")
    if not p: return _status(imr, "Pick an encoded prompt", "img-anim-add-status")
    with PILImage.open(H._output_dir() / b["keyframes"][i]) as im: w, h = max(32, im.width // 32 * 32), max(32, im.height // 32 * 32)   # the Inpaint tab's rounding
    params = {**s["form"], "batch": 1, "width": w, "height": h, "steps": _num(payload.get("steps")) or 20, "strength": float(payload.get("strength") or 0.85)}
    _queue(request.state.user.username, "inpaint", params, prompt_id=p["id"], image_path=b["keyframes"][i], mask_path=s.get("anim_mask", ""), reference_path="", storyboard=b["id"], insert_at=i + 1)
    return _status(imr, f"""&#x2713; Inpaint of keyframe {i + 1} queued ({"mask " + H._esc(s.get("anim_mask")) if s.get("anim_mask") else "whole frame"}, strength {params["strength"]})""", "img-anim-add-status", ok=True)

async def _im_anim_render(request, payload, imr):
    b = _owned(request, payload)
    if not b: return imr
    b["settings"] = {**b["settings"], **{k: payload.get(k, b["settings"][k]) for k in _BOARD_FIELDS}, "formats": _formats(payload.get("formats"))}
    _board_save(b)
    mdl, n = await models(), len(b["keyframes"])
    if n < 2: return _status(imr, "An animatic needs at least two keyframes", "img-anim-status")
    if not (_vconn() or H._cfg().get("mock_mode")): return _status(imr, "No image / video connection - set one in Settings", "img-anim-status")
    if not b["settings"]["formats"]: return _status(imr, "Pick at least one format", "img-anim-status")
    why = _check(mdl, b["settings"]["model"], "start_end_to_video")
    if why: return _status(imr, f"&#x26A0; {H._esc(why)} - nothing queued", "img-anim-status")
    missing = [k for k in b["keyframes"] if not (H._output_dir() / k).exists()]
    if missing: return _status(imr, f"&#x26A0; keyframe file missing: {H._esc(missing[0])}", "img-anim-status")
    d = _model(mdl, b["settings"]["model"]).get("defaults") or next((m["defaults"] for m in mdl.get("video", []) if "start_end_to_video" in m.get("modes", [])), {})
    _queue(request.state.user.username, "animatic", {"board": b["id"], "name": b["name"], "keyframes": list(b["keyframes"]), "prompts": [_gap_prompt(b, i) for i in range(n - 1)], "settings": dict(b["settings"]), "size": list(_size(b, mdl)), "fps": _num(b["settings"]["fps"], float) or d.get("fps") or 8, "force": payload.get("force") == "1"})
    return _status(imr, f"""&#x2713; Queued: {n - 1} gaps{" (all re-rendered)" if payload.get("force") == "1" else ", unchanged ones reused"}""", "img-anim-status", ok=True)

# --- Jobs (called by image.py's queue worker) ---

async def run_job(job: dict):
    """Video and animatic jobs: one at a time in the module's queue, like every node job. Each finished animatic gap is saved as it completes."""
    if job["kind"] == "video": await H.WS.send_personal_message(f"""<div id="img-vid-result" hx-swap-oob="innerHTML">{_video_result_html(job["username"])}</div>""", job["username"])
    else: await _push_board(job["username"], _board(job["params"]["board"]))
    stop, conn = asyncio.Event(), _vconn()
    poll = asyncio.create_task(H._poll_progress(job["username"], stop, conn, lambda: job.get("progress", ""))) if conn and not H._cfg().get("mock_mode") else None
    try:
        if job["kind"] == "video":
            r = await _clip(job["params"])
            job.update(status="done", result_file=r["file_name"], result=r)
        else: job.update(status="done", **(await _render_animatic(job)))
    except Exception as e: job.update(status="error", error=str(e))
    finally:
        stop.set()
        if poll: poll.cancel()
    job.update(finished=_now(), progress="")
    H._save("job", job)
    await H._push_job(job)
    if job["kind"] == "video": await H.WS.send_personal_message(f"""<div id="img-vid-result" hx-swap-oob="innerHTML">{_video_result_html(job["username"])}</div>""", job["username"])
    else: await _push_board(job["username"], _board(job["params"]["board"]))

async def _render_animatic(job: dict) -> dict:
    """Every gap as a start + end clip (cached gaps reused unless forced), then one joined clip. A failure or Stop keeps the gaps already finished."""
    p = job["params"]
    kfs, s, size, n = p["keyframes"], p["settings"], tuple(p["size"]), len(p["keyframes"]) - 1
    clips, rendered = [], 0
    for i in range(n):
        key = _gap_key(kfs[i], kfs[i + 1], p["prompts"][i], s, size)
        hit = None if p.get("force") else _cached(_board(p["board"]) or {"gaps": {}}, key)
        if not hit:
            job["progress"] = f"gap {i + 1}/{n}"
            H._save("job", job)
            await _push_board(job["username"], _board(p["board"]))
            r = await _clip({**_clip_params(s, p["prompts"][i]), "start_image": kfs[i], "end_image": kfs[i + 1], "width": size[0], "height": size[1], "formats": ["gif"], "name": f"""{p["name"]}_gap{i + 1}"""})
            hit, rendered = {"frames_dir": r["frames_dir"], "file": r["file_name"], "seed": r.get("seed"), "created": _now()}, rendered + 1
            b = _board(p["board"])
            if b:
                b["gaps"][key] = hit
                _board_save(b)
                await _push_board(job["username"], b)
        clips.append(hit["frames_dir"])
    job["progress"] = "joining"
    j = await _join(clips, p["fps"], _formats(s.get("formats")) or ["gif"], p["name"])
    entry = {"file": j["file_name"], "files": j.get("files", {}), "frame_count": j.get("frame_count"), "gaps": n, "rendered": rendered, "reused": n - rendered, "width": size[0], "height": size[1], "job": job["id"], "created": _now()}
    b = _board(p["board"])
    if b:
        b["renders"].insert(0, entry)
        _board_save(b)
    return {"result_file": j["file_name"], "result": {**j, **entry}}

async def keyframe_done(job: dict):
    """A storyboard keyframe job finished: its image joins the board where it was asked for (the end for prompts, after its source for inpaints)."""
    b = _board(job.get("storyboard", ""))
    if not b or job.get("status") != "done" or not job.get("result_file"): return
    b["keyframes"].insert(min(int(job.get("insert_at", len(b["keyframes"]))), len(b["keyframes"])), job["result_file"])
    _board_save(b)
    await _push_board(job["username"], b)

# --- Animatic pipelines (installed on request, never automatically) ---

def _shipped() -> list: return json.loads(PIPES_FILE.read_text())["pipelines"]
def _flow_sig(flow: dict) -> str: return json.dumps([{k: n.get(k) for k in ("id", "type", "config", "key_map", "extra_in_keys", "extra_out_keys")} for n in flow.get("nodes", [])], sort_keys=True)

def pipelines_html() -> str:
    eng, rows = H.AIM.engine, ""
    for doc in _shipped():
        cur = eng.load_pipeline(doc["id"])
        state = """<span style="color:var(--text_muted)">not installed</span>""" if not cur else """<span style="color:#00ffa2">installed</span>""" if _flow_sig(cur["flow"]) == _flow_sig(doc["flow"]) else """<span style="color:#ffcc00">installed, edited since</span>"""
        rows += f"""<div style="font-size:.72rem;padding:.2rem 0;border-bottom:var(--border-thick) solid var(--border)"><b>{H._esc(doc["name"])}</b> <code style="font-size:.62rem">{doc["id"]}</code> - {state}<div style="font-size:.64rem;color:var(--text_muted)">{H._esc(doc.get("description", ""))}</div></div>"""
    return f"""<h3 style="margin:0 0 .4rem;font-size:.85rem">Animatic pipelines</h3>
               {_note("Pipelines for the AI Manager that do what the Animatic tab does, for automation (e.g. an LLM writing the keyframe prompts). Nothing is installed until you click; Reset first copies each installed pipeline to <id>.json.bak-<time>.")}
               {rows}
               <div style="display:flex;gap:.4rem;margin-top:.5rem">{_btn("Install missing", "Install the ones not installed yet", "ui-btn", type="image_pipelines", action="install")}{_btn("Reset to shipped", "Back up, then replace every one with the shipped version", "ui-btn", 'hx-confirm="Replace all four with the shipped versions? Each current one is backed up first."', type="image_pipelines", action="reset")}</div>
               <div id="img-pipes-status" style="font-size:.72rem;min-height:1rem"></div>"""

async def _im_pipelines(request, payload, imr):
    eng, done = H.AIM.engine, []
    for doc in _shipped():
        cur = eng.load_pipeline(doc["id"])
        if cur and payload.get("action") != "reset": continue
        if cur: shutil.copy2(eng._pdp(doc["id"]), eng.PIPE_DIR / f"""{doc["id"]}.json.bak-{datetime.now():%Y%m%d_%H%M%S}""")
        eng.save_pipeline({**doc, "owner": request.state.user.username, "created": _now()})
        done.append(doc["id"])
    imr.oob(pipelines_html(), "img-pipes")
    return _status(imr, f"""&#x2713; {"Reset" if payload.get("action") == "reset" else "Installed"}: {", ".join(done)}""" if done else "Nothing to install - all are there", "img-pipes-status", ok=True)