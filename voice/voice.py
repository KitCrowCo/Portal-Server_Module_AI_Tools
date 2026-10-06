# /modules/ai_tools/voice/voice.py
"""
Voice - the speech lab. Sub-module of ai_tools, mounted at /module/ai_tools/voice.
Conversations run through the shared ChatManager (voice=True: hold-to-talk, hands-free call mode with barge-in, audio parts with replay); the tuning
controls (prompts, model, voice, language, call detection, pipeline) sit around them, per conversation.
Every turn is one pipeline run: the built-in flow (transcribe -> prompt -> generate -> speak) or any saved pipeline that follows the same key convention:
  in:  audio_b64 (recorded audio, data URL or bare base64) or transcript (typed text), plus history (prior turns as text, "" on the first turn)
  out: transcript, reply, reply_audio (base64 wav) - any of them may be missing; the view shows whatever the run produced.
Speech itself runs on whichever connections declare speech_to_text / text_to_speech (the speech_direct node today). This module records, shows, plays and tunes.
Data at data/ai_tools/voice/: one JSON file per conversation, audio at audio/<username>/<turn id>.wav (replies always, the user's recordings when keep_recordings is on).
"""
import json, uuid, base64, asyncio
from datetime import datetime
from pathlib import Path
from fastapi import APIRouter, Request, HTTPException
from fastapi.responses import FileResponse

TOOL_META = {"label": "Voice", "icon": "&#x1F3A4;", "group": "chat", "description": "Speech lab: talk with a pipeline, tune voices, prompts and call detection", "singleton": True}

router = APIRouter(redirect_slashes=False)
DATA_DIR = Path("./data/ai_tools/voice")
AUDIO_DIR = DATA_DIR / "audio"
_DONE = ("done", "error", "stopped", "interrupted")
ENV = {}
PREFIX = ""
UI = WS = IM = BI = AIM = CM = _FIELDS = None

# --- Conversation storage ---

def _dp(sid): return DATA_DIR / f"{Path(sid).name}.json"
def _load(sid): return json.loads(_dp(sid).read_text()) if sid and _dp(sid).exists() else None
def _save(doc): doc["modified"] = datetime.utcnow().isoformat(); _dp(doc["id"]).write_text(json.dumps(doc, indent=2))
def _list(username): return sorted((d for d in (json.loads(f.read_text()) for f in DATA_DIR.glob("*.json")) if d.get("username") == username), key=lambda d: d.get("modified", ""), reverse=True)
def _owned(request, sid): doc = _load(sid); return doc if doc and doc.get("username") == request.state.user.username else None

def _new(username, settings=None) -> dict:
    doc = {"id": f"vc_{uuid.uuid4().hex[:8]}", "username": username, "title": "New conversation", "settings": settings or {f.name: f.default for f in _FIELDS.fields}, "turns": [], "created": datetime.utcnow().isoformat()}
    _save(doc)
    return doc

async def _active(request) -> dict:
    """The user's active conversation, creating the first one on demand. A new conversation inherits the previous one's settings."""
    username = request.state.user.username
    doc = _owned(request, await ENV["get_state"](request, scope="user", namespace="voice", key="active_sid"))
    if not doc:
        recent = _list(username)
        doc = recent[0] if recent else _new(username)
        await ENV["set_state"](request, doc["id"], scope="user", namespace="voice", key="active_sid")
    return doc

def _add_turn(sid: str, role: str, text: str, job_id: str, audio_b64: str = "") -> dict:
    """Appends a turn; audio (data URL or bare base64 wav) is written next to the user's other audio and the turn keeps only the file name."""
    doc, tid = _load(sid), uuid.uuid4().hex[:8]
    turn = {"id": tid, "role": role, "text": text, "job_id": job_id, "ts": datetime.utcnow().isoformat(), "audio": _write_audio(doc["username"], tid, audio_b64) if audio_b64 else ""}
    doc["turns"].append(turn)
    if role == "user" and doc["title"] == "New conversation": doc["title"] = text[:48] or doc["title"]
    _save(doc)
    return turn

# --- Audio files ---

def _audio_path(username: str, name: str) -> Path: return AUDIO_DIR / Path(username).name / Path(name).name

def _write_audio(username: str, stem: str, b64: str) -> str:
    p = _audio_path(username, f"{stem}.wav")
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_bytes(base64.b64decode(b64.split(",", 1)[-1]))   # recordings arrive as data URLs, node replies as bare base64
    return p.name

def _backfill(doc: dict) -> dict:
    """Conversations from before audio files existed: copy each reply's audio out of its job record, once. Turns with nothing to copy get audio "" so they are not looked up again."""
    todo = [t for t in doc["turns"] if "audio" not in t]
    for t in todo: t["audio"] = _write_audio(doc["username"], t["id"], b64) if t["role"] == "assistant" and t.get("job_id") and (b64 := ((AIM.engine.load_job(t["job_id"]) or {}).get("data") or {}).get("reply_audio")) else ""
    if todo: _save(doc)
    return doc

# --- Settings (per conversation) ---

def _conn_opts(capability):
    def _opts(values=None): return [("", "(pool / first available)")] + [(c["_id"], c.get("display_name", c["_id"])) for c in AIM.connections.conns_matching(capability)]
    return _opts

def _conn_for(values, key: str, capability: str):
    """The connection a setting points at, or - when it is blank - the one a turn would actually use (the first declaring the capability), so dependent lists show real options either way."""
    cid = (values or {}).get(key, "")
    return AIM.connections.get_conn(cid) if cid else next(iter(AIM.connections.conns_matching(capability)), None)

def _listed(values, key: str, items: list, blank_label: str, note: str = "") -> list:
    """Select options for a node-provided list. A saved value the node no longer offers stays visible and selected, marked, instead of silently becoming a different choice."""
    cur = (values or {}).get(key, "")
    return [("", blank_label)] + [(i, i) for i in items] + ([(cur, f"""{cur} ({note or "saved - not offered by this connection"})""")] if cur and cur not in items else [])

def _model_opts(values=None):
    conn = _conn_for(values, "chat_conn", "chat")
    return _listed(values, "model", AIM.connections.list_models_sync(conn) if conn else [], "(auto - first non-embedding model)")

def _node_list(values, conn_key: str, capability: str, list_capability: str) -> dict:
    conn = _conn_for(values, conn_key, capability)
    if not conn: return {"error": f"no connection declares {capability}"}
    return AIM.connections.call_capability_sync(conn, list_capability) if AIM.connections.has_capability(conn, list_capability) else {"error": f"""connection {conn["_id"]} does not list {list_capability}"""}

def _voice_opts(values=None):
    r = _node_list(values, "tts_conn", "text_to_speech", "voice_list")
    return _listed(values, "voice", r.get("installed", []), f"""(node default{": " + r["default"] if r.get("default") else ""})""", note=f"""speech node unreachable: {r["error"][:60]}""" if r.get("error") else "")

def _lang_opts(values=None):
    r = _node_list(values, "stt_conn", "speech_to_text", "stt_languages")
    return _listed(values, "language", r.get("languages", []), "(auto-detect)", note=f"""speech node unreachable: {r["error"][:60]}""" if r.get("error") else "")

_CALL_KEYS = ("threshold_db", "bargein_db", "start_ms", "end_ms", "preroll_ms", "bargein_ms", "min_voiced_ms", "voicing")
_NEW_DEFAULTS = {"max_reply_tokens": 300, "trim_unfinished": True, "call_bargein_ms": 400, "call_min_voiced_ms": 250, "call_voicing": 0.45, "ignore_phrases": "", "max_no_speech_prob": None}

def _opt(s: dict, k: str):
    """A conversation setting, or the shipped default when the conversation was saved before the setting existed (or left it blank)."""
    return _NEW_DEFAULTS[k] if k in _NEW_DEFAULTS and s.get(k) in (None, "") else s.get(k)

_TRIM_UNFINISHED = """input[:max(input.rfind(c) for c in ".!?\u2026") + 1] if len(input) >= {cut} and max(input.rfind(c) for c in ".!?\u2026") > 0 else input"""   # transform expr: a reply near the cap ends at its last . ! ? or ellipsis; shorter replies pass unchanged

def _tune_flow(flow: dict, s: dict) -> dict:
    """Applies one conversation's reply cap, unfinished-sentence trim and nothing-said filters to the flow _flow() built. Nodes are found by type, so ids and key names do not matter."""
    nodes = flow["nodes"]
    gen, tr, sp = (next((n for n in nodes if n["type"] == t), None) for t in ("generate", "transcribe", "speak"))
    cap = int(_opt(s, "max_reply_tokens") or 0)
    if gen and cap > 0: gen["config"]["num_predict"] = cap
    if tr: tr["config"].update({"skip_empty": True, "ignore_phrases": _opt(s, "ignore_phrases") or "", "max_no_speech_prob": _opt(s, "max_no_speech_prob")})
    if gen and sp and cap > 0 and _opt(s, "trim_unfinished"):
        said = sp.setdefault("key_map", {}).get("text", "text")   # the key the speak node reads (the reply)
        nodes.append({"id": "trim_unfinished", "type": "transform", "name": "end speech at the last full sentence", "config": {"mode": "expr", "expr": _TRIM_UNFINISHED.format(cut=int(cap * 3.6))}, "key_map": {"input": said, "value": f"{said}_spoken"}, "extra_in_keys": [], "extra_out_keys": [], "appearance": {}})
        sp["key_map"]["text"] = f"{said}_spoken"   # the chat still shows the reply as written; only the voice stops at the sentence
    return flow

async def _heard(username: str, sid: str, job: dict) -> bool:
    """For a finished turn: True when words were heard (or the turn was typed). Tells the browser either way with a `cm-voice` trigger {sid, heard};
    heard=false resumes a reply that the utterance paused. Nothing heard = the transcribe node ended with 'dropped_text' and no 'text'."""
    tr = next((n for n in job.get("flow", {}).get("nodes", []) if n.get("type") == "transcribe"), None)
    if not tr: return True
    heard = tr.get("key_map", {}).get("text", "text") in job.get("data", {})
    await ENV["push_to_client"](username, {"t": "trigger", "event": "cm-voice", "detail": {"sid": sid, "heard": heard}})
    return heard

def _call_cfg(s: dict) -> dict: return {k: s.get(f"call_{k}") for k in _CALL_KEYS}   # blank values are dropped by ChatManager, so the browser script's defaults apply

def _build_fields():
    F = BI.SettingField
    refresh = {"hx_intent": "voice_settings_refresh", "hx_target": "#voice-settings-fields"}   # changing a connection re-renders the dependent lists (models, voices, languages) from the live form values
    return BI.SettingsGroup("voice", "", [
        F("consent_prefix", "System Prompt Prefix (consent / standing terms)", "textarea", "", hint="Sent first, before the system prompt, on every turn. It stays in the model's prompt cache, so after the first turn it costs almost nothing - see Prepare."),
        F("system_prompt", "System Prompt", "textarea", "You are a friendly voice assistant. Answer in short, natural spoken sentences - no lists, tables or code unless asked."),
        F("chat_conn", "Chat Connection", "select", "", options=_conn_opts("chat"), **refresh),
        F("model", "Model", "select", "", options=_model_opts),
        F("tts_conn", "Text-to-Speech Connection", "select", "", options=_conn_opts("text_to_speech"), **refresh),
        F("voice", "Voice", "select", "", options=_voice_opts),
        F("speak_replies", "Speak replies", "checkbox", True),
        F("language", "Spoken Language", "select", "", options=_lang_opts),
        F("keep_recordings", "Keep my recordings (replayable in the conversation)", "checkbox", True, hint="Off: your audio is transcribed and discarded; only the text is kept."),
        F("pipeline_id", "Pipeline", "select", "", options=lambda values=None: [("", "(built-in: transcribe -> generate -> speak)")] + [(p["id"], p.get("name", p["id"])) for p in AIM.engine.list_pipelines()], advanced=True, hint="A saved pipeline must follow the key convention: in audio_b64 or transcript + history, out transcript / reply / reply_audio."),
        F("stt_conn", "Speech-to-Text Connection", "select", "", options=_conn_opts("speech_to_text"), advanced=True, **refresh),
        F("length_scale", "Speaking Pace (length scale, >1 slower)", "number", None, advanced=True),
        F("num_ctx", "Context Window Tokens", "number", 16384, step=1, advanced=True),
        F("history_chars", "History Carried Into Each Turn (chars)", "number", 8000, step=1, advanced=True),
        F("stream_node", "Live-Text Node Id", "text", "llm", advanced=True, hint="Node whose text streams into the chat while it is generated. The built-in flow's generate node is 'llm'."),
        F("call_threshold_db", "Call: Speech Threshold (dB over room noise, default 10)", "number", None, advanced=True, hint="Lower picks up quieter speech but also more noise."),
        F("call_bargein_db", "Call: Extra dB to Interrupt a Reply (default 6)", "number", None, advanced=True, hint="Added to the threshold while a reply is playing, so speaker leakage does not interrupt it."),
        F("call_start_ms", "Call: Speech Needed to Start (ms, default 180)", "number", None, step=1, advanced=True),
        F("call_end_ms", "Call: Silence That Ends a Turn (ms, default 800)", "number", None, step=1, advanced=True, hint="Raise it if you get cut off while pausing to think."),
        BI.SettingField("max_reply_tokens", "Longest Reply (tokens)", "number", 300, step=1, hint="Hard cap sent to the model; 300 is about a minute of speech. 0 = no cap."),
        BI.SettingField("trim_unfinished", "Drop an Unfinished Last Sentence", "checkbox", True, hint="When a reply reaches the cap, speech ends at its last complete sentence instead of a fragment."),
        BI.SettingField("call_bargein_ms", "Call: Talk-Over Time Before a Reply Pauses (ms)", "number", 400, step=1, advanced=True, hint="Speaking over a reply this long pauses it; the reply resumes if nothing was said."),
        BI.SettingField("call_min_voiced_ms", "Call: Voiced Speech Needed to Send (ms)", "number", 250, step=1, advanced=True, hint="An utterance with less voiced sound is dropped in the browser (coughs, knocks). Raise to 400 if noise still sends."),
        BI.SettingField("call_voicing", "Call: Voice Detection Strictness (0-1)", "number", 0.45, advanced=True, hint="How periodic (voice-like) the speech band must be. Raise to 0.55 if noise still counts as speech."),
        BI.SettingField("ignore_phrases", "Ignore These Transcripts (one per line)", "textarea", "", advanced=True, hint="Whole transcripts treated as nothing said - the recognizer's noise inventions, e.g. Thank you. / Thanks for watching!"),
        BI.SettingField("max_no_speech_prob", "Drop Recognized Segments Rated Not-Speech Above (0-1)", "number", None, advanced=True, hint="Uses the recognizer's own per-segment score (speech_direct reports it). Blank = keep all; 0.6 is a reasonable start."),
        F("call_preroll_ms", "Call: Audio Kept From Before Speech Starts (ms, default 400)", "number", None, step=1, advanced=True)], json_path="")

# --- Turn execution ---

def _system(s: dict) -> str: return "\n\n".join(p for p in (str(s.get("consent_prefix") or "").strip(), str(s.get("system_prompt") or "").strip()) if p)   # prefix first and unchanged turn to turn, so it is the cached part of every prompt

def _flow(s: dict) -> dict:
    """The built-in turn as an ordinary inline flow - 'Save as pipeline' writes exactly this, so it can be edited in the pipeline builder and selected above."""
    nodes = [{"id": "stt", "name": "Transcribe", "type": "transcribe", "config": {"conn_id": s.get("stt_conn", ""), "language": s.get("language", ""), "vad_filter": True}, "key_map": {"text": "transcript"}},
             {"id": "prompt", "name": "Prompt", "type": "transform", "config": {"mode": "template", "template": "{history}{transcript}"}, "key_map": {"input": "transcript", "value": "prompt"}, "extra_in_keys": ["history"]},
             {"id": "llm", "name": "Reply", "type": "generate", "config": {"conn_id": s.get("chat_conn", ""), "model": s.get("model", ""), "system_prompt": _system(s), "user_template": "{prompt}", "num_ctx": s.get("num_ctx", 16384), "think": ""}, "key_map": {"input": "prompt", "text": "reply"}}]
    if s.get("speak_replies"): nodes.append({"id": "tts", "name": "Speak", "type": "speak", "config": {"conn_id": s.get("tts_conn", ""), "voice": s.get("voice", ""), "length_scale": s.get("length_scale")}, "key_map": {"text": "reply", "audio_b64": "reply_audio"}})
    #return _tune_flow(<flow dict>, <this conversation's settings dict>)
    return {"nodes": [{**n, "status": "idle"} for n in nodes]}

def _history(doc: dict) -> str:
    """Prior turns as plain text, newest kept whole, oldest dropped first to fit history_chars. Appending a turn only extends the text, so the model's cached prefix stays valid."""
    lines, size, budget = [], 0, int(doc["settings"].get("history_chars") or 0)
    for t in reversed([t for t in doc["turns"] if t.get("text")]):
        line = f"""{"User" if t["role"] == "user" else "Assistant"}: {t["text"]}"""
        if size + len(line) > budget: break
        lines.insert(0, line); size += len(line) + 1
    return f"""Conversation so far:\n{chr(10).join(lines)}\n\nThe user now says: """ if lines else ""

async def _start_turn(request, doc: dict, inputs: dict, imr):
    s, sid, username = doc["settings"], doc["id"], request.state.user.username
    kind, extra = ("id", {"pipeline_id": s["pipeline_id"]}) if s.get("pipeline_id") else ("inline", {"inline_flow": _flow(s)})
    job_id, err = AIM.engine.submit(username, kind=kind, inputs={**inputs, "history": _history(doc)}, lane=f"voice:{username}", **extra)
    if err: return imr.raw(CM.append_system_error_html(sid, err))
    imr.raw(CM.working_html(sid, {"type": "voice_stop", "job_id": job_id, "branch": IM.branch_id, "lvl": 2}))   # the call-mode barge-in presses this Stop button
    imr.raw(f"""<div id="cm-stream-{sid}" hx-swap-oob="innerHTML"><div class="cm-content voice-live" data-stream-job="{job_id}" data-stream-node="{UI.escape(s.get("stream_node") or "llm")}"></div></div>""")
    asyncio.create_task(_watch(username, sid, job_id, typed="transcript" in inputs, recording=inputs.get("audio_b64", "") if s.get("keep_recordings") else ""))
    return imr

def _failed(job: dict) -> str: return next((f"""{n.get("name") or n["id"]}: {n.get("message", "")}""" for n in job["flow"]["nodes"] if n.get("status") == "error"), job["log"][-1] if job.get("log") else job["status"])

async def _push(username, html: str): await WS.send_personal_message(html, username)

async def _watch(username: str, sid: str, job_id: str, typed: bool, recording: str = ""):
    """Follows one turn's job: the user's words appear once transcribed (with their recording, when kept); the reply appears with its audio once spoken - its text streams live meanwhile - or as text alone if the run ends without audio."""
    shown, job = set(), None
    while True:
        await asyncio.sleep(0.4)
        try: job = AIM.engine.load_job(job_id)
        except json.JSONDecodeError: continue   # the engine rewrites the job file between waves; a read can land mid-write
        data, done = job.get("data", {}), job["status"] in _DONE
        if "transcript" in data and not typed and "user" not in shown:
            shown.add("user")
            await _push(username, CM.append_message_html(sid, _msg(_add_turn(sid, "user", data["transcript"], job_id, recording), username), viewer_name=username))
        if "reply" in data and "reply" not in shown and ("reply_audio" in data or done):
            shown.add("reply")
            await _push(username, CM.append_message_html(sid, _msg(_add_turn(sid, "assistant", data["reply"], job_id, data.get("reply_audio", "")), username, autoplay=True), viewer_name=username) + CM.stream_clear_html(sid))
        if done: break
    tail = CM.working_hide_html(sid) + CM.stream_clear_html(sid)
    if job["status"] == "stopped": tail += f"""<div id="voice-status" hx-swap-oob="innerHTML"><span class="status-label">Turn stopped.</span></div>"""
    elif job["status"] != "done": tail += CM.append_system_error_html(sid, f"""Turn {job["status"]} - {_failed(job)}""")
    await _push(username, tail)

# --- HTML ---

def _post(action: str, **extra) -> str: return f"""hx-post="/im/in" hx-target="body" hx-swap="none" hx-vals='{json.dumps({"type": f"voice_{action}", "lvl": 2, **extra})}'"""

def _msg(t: dict, username: str, autoplay: bool = False) -> dict:
    """A stored turn as a ChatManager message. Autoplay is only for the freshly pushed reply, never for re-renders."""
    return {"id": t["id"], "role": t["role"], "content": t["text"], "user_name": username if t["role"] == "user" else "", "persona_name": "" if t["role"] == "user" else "Assistant", "timestamp": t.get("ts", ""), "audio": {"url": f"""{PREFIX}/audio/{t["audio"]}""", "autoplay": autoplay} if t.get("audio") else None}

def _list_html(username: str, active: str) -> str:
    rows = "".join(f"""<div class="list-card glass" style="{"border-left:.15rem solid var(--accent)" if d["id"] == active else ""}"><div class="list-card-hd"><span class="list-card-title" {_post("open", sid=d["id"])}>{UI.escape(d.get("title", ""))}</span><button class="btn-icon" style="color:#ff5f5f" {_post("delete", sid=d["id"])} hx-confirm="Delete this conversation and its audio?">&#x2715;</button></div><div class="status-label">{d.get("modified", "")[:16].replace("T", " ")}</div></div>""" for d in _list(username))
    return rows or '<div class="list-empty">No conversations.</div>'

def _main_html(doc: dict, username: str, status: str = "") -> str:
    sid = doc["id"]
    return f"""<div class="voice-head"><b>{UI.escape(doc["title"])}</b>
                   <details class="status-details" {"open" if status else ""}><summary>Settings</summary><form {_post("settings_save", sid=sid)} hx-include="this" class="node-form"><div id="voice-settings-fields">{_FIELDS.render(doc["settings"], name_prefix="vs_")}</div><div class="form-actions"><button type="submit" class="ui-btn">Save</button><button type="button" class="ui-btn" {_post("save_flow", sid=sid)}>Save built-in flow as pipeline</button><span id="voice-saved"></span></div></form></details>
                   <div class="form-actions"><button class="ui-btn" {_post("prepare", sid=sid)} title="Loads the model and caches the system prompt so the first real turn starts fast">Prepare</button><button class="ui-btn" {_post("new")} title="Starts a fresh conversation with these settings; this one stays in the list">Restart conversation</button><span id="voice-status">{status}</span></div></div>
               <div class="voice-chat">{CM.shell(sid, messages=[_msg(t, username) for t in doc["turns"]], viewer_name=username, voice_opts=_call_cfg(doc["settings"]))}</div>"""

def _page_html(username: str, doc: dict) -> str: return f"""<div class="voice-shell"><div class="list-panel voice-side"><div class="list-panel-hd"><span class="list-panel-title">Conversations</span><button class="btn-icon" {_post("new")}>+</button></div><div id="voice-list" class="list-body">{_list_html(username, doc["id"])}</div></div><div id="voice-main" class="voice-main">{_main_html(_backfill(doc), username)}</div></div>"""

def _refresh(imr, username: str, doc: dict, status: str = ""):
    imr.oob(_main_html(_backfill(doc), username, status), "voice-main")
    return imr.oob(_list_html(username, doc["id"]), "voice-list")

# --- Intent handlers ---

async def _h_new(request, payload, imr):
    prev = await _active(request)
    doc = _new(request.state.user.username, dict(prev["settings"]))
    await ENV["set_state"](request, doc["id"], scope="user", namespace="voice", key="active_sid")
    return _refresh(imr, request.state.user.username, doc)

async def _h_open(request, payload, imr):
    doc = _owned(request, payload.get("sid", ""))
    if not doc: return imr
    await ENV["set_state"](request, doc["id"], scope="user", namespace="voice", key="active_sid")
    return _refresh(imr, request.state.user.username, doc)

async def _h_delete(request, payload, imr):
    doc = _owned(request, payload.get("sid", ""))
    if doc:
        for t in doc["turns"]:
            if t.get("audio"): _audio_path(doc["username"], t["audio"]).unlink(missing_ok=True)
        _dp(doc["id"]).unlink()
    return _refresh(imr, request.state.user.username, await _active(request))

async def _h_settings_save(request, payload, imr):
    """Saves, then re-renders the conversation so footer-level options (call detection) take effect. An open call is ended by the re-render - start it again to use the new values."""
    doc = _owned(request, payload.get("sid", ""))
    if not doc: return imr
    doc["settings"] = _FIELDS.parse(payload, name_prefix="vs_")
    _save(doc)
    return _refresh(imr, request.state.user.username, doc, '<span class="status-label">&#x2713; Saved</span>')

async def _h_save_flow(request, payload, imr):
    doc = _owned(request, payload.get("sid", ""))
    if not doc: return imr
    pl = {**AIM.engine.new_pipeline(owner="voice", name=f"""Voice turn ({datetime.utcnow().strftime("%Y-%m-%d %H:%M")})"""), "flow": _flow({**doc["settings"], "speak_replies": True}), "tags": ["voice"]}
    AIM.engine.save_pipeline(pl)
    return imr.oob(f"""<span class="status-label">&#x2713; Saved as pipeline {pl["id"]} - edit it in a pipeline builder, then pick it as this conversation's Pipeline.</span>""", "voice-saved")

async def _h_settings_refresh(request, payload, imr): return imr.raw(_FIELDS.render(_FIELDS.parse(payload, name_prefix="vs_"), name_prefix="vs_"))   # main swap into #voice-settings-fields (the select carries hx-target); unsaved edits ride along in the payload

async def _h_prepare(request, payload, imr):
    """One short generation with the real system prompt, model and context size: loads the model and fills its prompt cache, so the first spoken turn only processes the new words. The reply is not added to the conversation."""
    doc = _owned(request, payload.get("sid", ""))
    if not doc: return imr
    llm = next(n for n in _flow(doc["settings"])["nodes"] if n["id"] == "llm")
    flow = {"nodes": [{**llm, "config": {**llm["config"], "user_template": "Reply with the single word: ready", "num_predict": 16}, "key_map": {"input": "go", "text": "ready"}}]}
    username = request.state.user.username
    job_id, err = AIM.engine.submit(username, kind="inline", inline_flow=flow, inputs={"go": "1"}, lane=f"voice:{username}")
    if err: return imr.oob(f"""<div class="err-box">{UI.escape(err)}</div>""", "voice-status")
    asyncio.create_task(_watch_prepare(username, job_id))
    return imr.oob('<span class="status-label">Preparing - loading the model and caching the system prompt&#x2026;</span>', "voice-status")

async def _watch_prepare(username: str, job_id: str):
    t0, job = asyncio.get_event_loop().time(), None
    while True:
        await asyncio.sleep(0.5)
        try: job = AIM.engine.load_job(job_id)
        except json.JSONDecodeError: continue
        if job["status"] in _DONE: break
    msg = f"""<span class="status-label">&#x2713; Ready - model loaded and system prompt cached ({asyncio.get_event_loop().time() - t0:.0f} s)</span>""" if job["status"] == "done" else f"""<div class="err-box">Prepare {job["status"]} - {UI.escape(next((n.get("message", "") for n in job["flow"]["nodes"] if n.get("status") == "error"), ""))}</div>"""
    await _push(username, f"""<div id="voice-status" hx-swap-oob="innerHTML">{msg}</div>""")

async def _h_text(request, payload, imr):
    """ChatManager form submit: {cid, content}."""
    doc = _owned(request, payload.get("cid", ""))
    text = str(payload.get("content", "")).strip()
    if not doc or not text: return imr
    username = request.state.user.username
    imr.oob("", "voice-status").raw(CM.append_message_html(doc["id"], _msg(_add_turn(doc["id"], "user", text, ""), username), viewer_name=username))
    return await _start_turn(request, doc, {"transcript": text}, imr)   # doc as loaded before this turn was added, so history does not repeat it

async def _h_audio(request, payload, imr):
    """ChatManager mic (hold-to-talk or call mode): {cid, audio_b64 = 16 kHz mono WAV data URL}."""
    doc = _owned(request, payload.get("cid", ""))
    if not doc or not payload.get("audio_b64"): return imr
    return await _start_turn(request, doc, {"audio_b64": payload["audio_b64"]}, imr.oob("", "voice-status"))

async def _h_stop(request, payload, imr):
    AIM.engine.stop(payload.get("job_id", ""))
    return imr.oob('<span class="status-label">Stopping after the current step&#x2026;</span>', "voice-status")

# --- Init / routes ---

def init_tool(env: dict, prefix: str):
    global ENV, PREFIX, UI, WS, IM, BI, AIM, CM, _FIELDS
    ENV, PREFIX = env, prefix
    UI = env["templates"].env.globals.get("UI")
    WS = env["ws"]
    BI = env["tools"]["built_ins"]
    AIM = env["tools"]["ai_manager"]
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    AUDIO_DIR.mkdir(parents=True, exist_ok=True)
    _FIELDS = _build_fields()
    IM = env["InterfaceManager"](nesting_level=2, db_path="voice_im.db")
    CM = BI.ChatManager(namespace="voice", base_url=prefix, view_style="bubble", stream_toggle=False, think_toggle=False, stop_enabled=True, pin_enabled=True, allow_edit=False, allow_delete=False, allow_copy=True, show_info=False, markdown_mode="standard", placeholder="Type, hold \U0001F3A4 to talk, or \u260E for a hands-free call (Ctrl+Enter sends)", branch_id=IM.branch_id, nesting_level=2, intent_name="voice_text", voice=True, audio_intent="voice_audio")
    IM.scripts.update({f"voice_{k}": [fn] for k, fn in {"new": _h_new, "open": _h_open, "delete": _h_delete, "settings_save": _h_settings_save, "save_flow": _h_save_flow, "text": _h_text, "audio": _h_audio, "stop": _h_stop, "settings_refresh": _h_settings_refresh, "prepare": _h_prepare}.items()})
    print("[voice] ready")

@router.get("")
@router.get("/")
async def root(request: Request):
    doc = await _active(request)
    return ENV["templates"].TemplateResponse(name="base.html", request=request, context={"request": request, "user": request.state.user, "nesting_level": 2, "shell_id": IM.branch_id, "toolbars": {}, "content": _page_html(request.state.user.username, doc), "extra_css": CSS + CM.CSS, "extra_script": CM.SCRIPT})

@router.get("/audio/{name}")
async def audio(request: Request, name: str):
    """A turn's audio - the requesting user's own folder only."""
    p = _audio_path(request.state.user.username, name)
    if not p.is_file(): raise HTTPException(404)
    return FileResponse(p, media_type="audio/wav")

CSS = """
.voice-shell{display:flex;height:100%;width:100%;overflow:hidden;}
.voice-side{width:14rem;flex-shrink:0;border-right:var(--border-thick) solid var(--border);}
.voice-main{flex:1;display:flex;flex-direction:column;gap:.4rem;padding:.6rem;overflow:hidden;min-width:0;}
.voice-head{flex-shrink:0;max-height:45vh;overflow-y:auto;}
.voice-chat{flex:1;min-height:0;display:flex;}
.voice-live{white-space:pre-wrap;opacity:.8;}
@media (max-width:46rem){.voice-side{display:none;}}
"""
