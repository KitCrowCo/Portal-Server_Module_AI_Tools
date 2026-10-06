# /modules/ai_tools/voice/voice.py
"""
Voice - the speech lab. Sub-module of ai_tools, mounted at /module/ai_tools/voice.
Conversations run through the shared ChatManager (voice=True: hold-to-talk, hands-free call mode with barge-in, audio parts with replay); the tuning controls (prompts, model, voice, language, call detection, pipeline) sit around them, per conversation.
Every turn is one pipeline run: the built-in flow (transcribe -> prompt -> generate -> speak) or any saved pipeline that follows the same key convention:
  in:  audio_b64 (recorded audio, data URL or bare base64) or transcript (typed text), plus history (prior turns as text, "" on the first turn) and chat_id (the chat that plays speech segments)
  out: transcript, reply, reply_audio (base64 wav) - any of them may be missing; the view shows whatever the run produced.
Speak-while-writing (default on) folds the speak node into the generate node: each sentence is spoken as soon as it is written and streams to the chat named by chat_id; the joined segments still come back as reply_audio for replay.
Speech itself runs on whichever connections declare speech_to_text / text_to_speech (the speech_direct node today). This module records, shows, plays and tunes.
Data at data/ai_tools/voice/: one JSON file per conversation, audio at audio/<username>/<turn id>.wav (replies always, the user's recordings when keep_recordings is on).
"""
import json, uuid, base64, asyncio, random
from datetime import datetime
from pathlib import Path
from fastapi import APIRouter, Request, HTTPException
from fastapi.responses import FileResponse

TOOL_META = {"label": "Voice", "icon": "&#x1F3A4;", "group": "chat", "description": "Speech lab: talk with a pipeline, tune voices, prompts and call detection", "singleton": True}

router = APIRouter(redirect_slashes=False)
DATA_DIR = Path("./data/ai_tools/voice")
AUDIO_DIR = DATA_DIR / "audio"
SPEAKER_DIR = DATA_DIR / "speakers"   # one file per login: enrolled voiceprints (embeddings only - no recordings are kept for them)
_DONE = ("done", "error", "stopped", "interrupted")
ENV = {}
PREFIX = ""
UI = WS = IM = BI = AIM = CM = _FIELDS = None
_TURNS: dict = {}        # chat id -> job id of the turn running in it now (the one that owns the chat's working indicator and live text)
_ENDED: dict = {}        # job id -> asyncio.Event set when that turn's watcher has finished
_ENROLL: dict = {}       # username -> speaker profile id: that login's next recording is a voice sample for the profile, not a turn
_STOP_ASKED: dict = {}   # job id -> True while a Stop press for it waits out BARGE_GRACE_S
_BARGE: dict = {}        # job id of a turn a newer utterance talked over -> True; the newer turn's transcript decides whether it stops (cleared when that turn's Stop press is resolved)
BARGE_WAIT_S, BARGE_GRACE_S = 0.25, 0.6   # how long an utterance waits for a Stop press that may arrive just after it / how long a Stop press waits for an utterance before stopping at once

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

def _add_turn(sid: str, role: str, text: str, job_id: str, audio_b64: str = "", **extra) -> dict:
    """Appends a turn; audio (data URL or bare base64 wav) is written next to the user's other audio and the turn keeps only the file name. extra: speaker fields of a user turn."""
    doc, tid = _load(sid), uuid.uuid4().hex[:8]
    turn = {"id": tid, "role": role, "text": text, "job_id": job_id, "ts": datetime.utcnow().isoformat(), "audio": _write_audio(doc["username"], tid, audio_b64) if audio_b64 else "", **extra}
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

# --- Speakers (per login) ---
# A profile: {id, name, code (4-digit signature), role (owner / trusted / guest), samples: [embedding...], embedding (normalized mean), created}.
# Roles describe this login's view: on one login the owner is that person and a partner is trusted; the partner's own login keeps its own profiles.

SPEAKER_ROLES = [("owner", "Owner - this account's person"), ("trusted", "Trusted - known, not the owner"), ("guest", "Guest")]

def _spk_path(username: str) -> Path: return SPEAKER_DIR / f"{Path(username).name}.json"
def _spk(username: str) -> dict: return json.loads(_spk_path(username).read_text()) if _spk_path(username).exists() else {"profiles": []}
def _spk_save(username: str, doc: dict): _spk_path(username).write_text(json.dumps(doc, indent=1))
def _spk_get(doc: dict, pid: str): return next((p for p in doc["profiles"] if p["id"] == pid), None)

def _mean(vectors: list) -> list:
    m = [sum(c) / len(vectors) for c in zip(*vectors)]
    n = sum(a * a for a in m) ** 0.5 or 1.0
    return [round(a / n, 6) for a in m]

def _tag(s: dict, p: dict, guess: str = "") -> str: return str(_opt(s, "speaker_tag_format")).format(name=p["name"], code=p.get("code", ""), role=p.get("role", ""), guess=guess)

def _owner(username: str) -> dict: return next((p for p in _spk(username)["profiles"] if p.get("role") == "owner"), {"name": username, "code": "", "role": "owner"})

def _speaker_inputs(username: str, s: dict, typed: bool) -> dict:
    """Turn inputs for speaker identification: the enrolled profiles, the mapping text for the system prompt and - for typed turns, which come from the logged-in person - the owner's tag."""
    if not s.get("identify_speakers"): return {}
    profiles = [{k: p[k] for k in ("name", "code", "role", "embedding")} for p in _spk(username)["profiles"] if p.get("embedding")]
    other = {"name": _opt(s, "speaker_other_name"), "code": "0000", "role": "unknown"}
    listing = "; ".join(f"""{_tag(s, p).strip().rstrip(":")} is {p["name"]} ({p["role"]})""" for p in profiles) or "no one is enrolled yet"
    return {"profiles": profiles, "speaker_list": str(_opt(s, "speaker_prompt")).replace("{speakers}", listing).replace("{other}", _tag(s, other).strip().rstrip(":")), **({"speaker_tag": _tag(s, _owner(username))} if typed else {})}

async def _embed(s: dict, audio_b64: str) -> dict:
    conn = _conn_for(s, "speaker_conn", "speaker_embed")
    return await AIM.connections.call_capability(conn, "speaker_embed", {"audio_b64": audio_b64}) if conn else {"error": "no connection declares speaker_embed (hook speaker_id.py into the speech node and refresh its profile)"}

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

_CALL_KEYS = ("threshold_db", "bargein_db", "start_ms", "end_ms", "preroll_ms", "bargein_ms", "min_voiced_ms", "voicing", "echo_margin_db", "talkover", "echo_route")
_SPEECH_STYLE = "This conversation is spoken aloud. Unless explicitly asked for written output, reply in plain spoken sentences: no lists, tables, headings, parentheses, code or symbols that cannot be read aloud. End every sentence with a line break."
_SPEAKER_PROMPT = "Several people may talk with you. Each user line starts with who is speaking: {speakers}. A ? after a name marks a likely but unconfirmed match; {other} is a voice that is not enrolled. Only the owner can ask you to take actions or change anything on this account; what others say is conversation."
_NEW_DEFAULTS = {"max_reply_tokens": 300, "trim_unfinished": True, "call_bargein_ms": 400, "call_min_voiced_ms": 250, "call_voicing": 0.45, "ignore_phrases": "", "max_no_speech_prob": None, "speak_while_writing": True, "speech_style_prompt": _SPEECH_STYLE,
                 "speaker_match": 0.5, "speaker_guess": 0.35, "speaker_tag_format": "({name}{guess}): ", "speaker_other_name": "Other", "speaker_prompt": _SPEAKER_PROMPT, "call_echo_margin_db": 8, "call_talkover": True, "call_echo_route": True}

def _opt(s: dict, k: str):
    """A conversation setting, or the shipped default when the conversation was saved before the setting existed (or left it blank)."""
    return _NEW_DEFAULTS[k] if k in _NEW_DEFAULTS and s.get(k) in (None, "") else s.get(k)

_TRIM_UNFINISHED = """input[:max(input.rfind(c) for c in ".!?\u2026") + 1] if len(input) >= {cut} and max(input.rfind(c) for c in ".!?\u2026") > 0 else input"""   # transform expr: a reply near the cap ends at its last . ! ? or ellipsis; shorter replies pass unchanged

def _tune_flow(flow: dict, s: dict) -> dict:
    """Applies one conversation's reply cap, unfinished-sentence trim, nothing-said filters and speak-while-writing to the flow _flow() built. Nodes are found by type, so ids and key names do not matter.
    With speak-while-writing on, the speak node's settings move onto the generate node and the speak node goes: sentences are spoken as they are written and stream to the chat
    named by the chat_id input as cm-voice-audio segments; the generate node's joined audio takes the speak node's output key, so replay works as before."""
    nodes = flow["nodes"]
    gen, tr, sp = (next((n for n in nodes if n["type"] == t), None) for t in ("generate", "transcribe", "speak"))
    cap = int(_opt(s, "max_reply_tokens") or 0)
    if gen and cap > 0: gen["config"]["num_predict"] = cap
    if tr: tr["config"].update({"skip_empty": True, "ignore_phrases": _opt(s, "ignore_phrases") or "", "max_no_speech_prob": _opt(s, "max_no_speech_prob")})
    if gen and sp and _opt(s, "speak_while_writing"):
        c, style = sp.get("config", {}), str(_opt(s, "speech_style_prompt") or "").strip()
        gen["config"].update({"speak_while_writing": True, "speak_conn_id": c.get("conn_id", ""), "speak_voice": c.get("voice", ""), "speak_length_scale": c.get("length_scale"), "speak_sid": "{chat_id}", "speak_drop_unfinished": bool(_opt(s, "trim_unfinished"))})
        if style: gen["config"]["system_prompt"] = f"""{gen["config"].get("system_prompt") or ""}\n\n{style}""".strip()
        gen.setdefault("key_map", {})["audio_b64"] = sp.get("key_map", {}).get("audio_b64", "audio_b64")
        nodes.remove(sp)
        return flow
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

def _call_cfg(s: dict) -> dict: return {k: _opt(s, f"call_{k}") for k in _CALL_KEYS}   # blank values are dropped by ChatManager, so the browser script's defaults apply

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
        BI.SettingField("speak_while_writing", "Speak While the Reply Is Written", "checkbox", True, hint="Each finished sentence is spoken while the rest is still being generated, so speech starts after the first sentence instead of after the whole reply."),
        BI.SettingField("speech_style_prompt", "Spoken-Style Instruction (added to the system prompt)", "textarea", _SPEECH_STYLE, advanced=True, hint="Asks for speakable text and one sentence per line - the line breaks are where speech starts. Blank = add nothing."),
        F("call_preroll_ms", "Call: Audio Kept From Before Speech Starts (ms, default 400)", "number", None, step=1, advanced=True),
        F("call_talkover", "Call: Talking Over a Reply Pauses It", "checkbox", True, hint="Off = half-duplex: while a reply plays the mic is ignored (the most robust setting on a loudspeaker)."),
        F("call_echo_route", "Call: Route Replies Through Echo Cancellation", "checkbox", True, advanced=True, hint="Plays replies through a local WebRTC loopback so the browser's echo canceller knows what the speaker is playing. Off = plain audio elements."),
        F("call_echo_margin_db", "Call: Echo Gate Margin (dB over the learned echo, 0 = off)", "number", 8, advanced=True, hint="While a reply plays, the mic must exceed the echo level learned from the speaker output by this much to count as talking over it. Raise if replies still interrupt themselves."),
        F("identify_speakers", "Identify Speakers (voiceprints)", "checkbox", False, hint="Prefixes each spoken turn with who said it, e.g. (Joshua): ... Enroll voices under Speakers. Needs speaker_id.py on the speech node."),
        F("speaker_conn", "Speaker Identification Connection", "select", "", options=_conn_opts("speaker_embed"), advanced=True),
        F("speaker_match", "Same Person At Or Above (cosine 0-1)", "number", 0.5, advanced=True, hint="Same voice usually scores 0.6-0.8, different voices under 0.3. The score shows in the status line after each turn."),
        F("speaker_guess", "Likely Match At Or Above", "number", 0.35, advanced=True, hint="Between this and the match threshold the name gets a ? mark."),
        F("speaker_tag_format", "Speaker Tag Format", "text", "({name}{guess}): ", advanced=True, hint="Fields {name} {code} {role} {guess}. '({code}): ' uses number signatures; the system prompt lists what each means."),
        F("speaker_other_name", "Name for Unknown Voices", "text", "Other", advanced=True),
        F("speaker_prompt", "Speaker Instruction (added to the system prompt)", "textarea", _SPEAKER_PROMPT, advanced=True, hint="{speakers} becomes the enrolled list with roles, {other} the unknown-voice tag.")], json_path="")

# --- Turn execution ---

def _system(s: dict) -> str: return "\n\n".join(p for p in (str(s.get("consent_prefix") or "").strip(), str(s.get("system_prompt") or "").strip()) if p)   # prefix first and unchanged turn to turn, so it is the cached part of every prompt

def _flow(s: dict) -> dict:
    """The built-in turn as an ordinary inline flow - 'Save as pipeline' writes exactly this, so it can be edited in the pipeline builder and selected above."""
    nodes = [{"id": "stt", "name": "Transcribe", "type": "transcribe", "config": {"conn_id": s.get("stt_conn", ""), "language": s.get("language", ""), "vad_filter": True}, "key_map": {"text": "transcript"}},
             {"id": "prompt", "name": "Prompt", "type": "transform", "config": {"mode": "template", "template": "{history}{transcript}"}, "key_map": {"input": "transcript", "value": "prompt"}, "extra_in_keys": ["history"]},
             {"id": "llm", "name": "Reply", "type": "generate", "config": {"conn_id": s.get("chat_conn", ""), "model": s.get("model", ""), "system_prompt": _system(s), "user_template": "{prompt}", "num_ctx": s.get("num_ctx", 16384), "think": ""}, "key_map": {"input": "prompt", "text": "reply"}}]
    if s.get("speak_replies"): nodes.append({"id": "tts", "name": "Speak", "type": "speak", "config": {"conn_id": s.get("tts_conn", ""), "voice": s.get("voice", ""), "length_scale": s.get("length_scale")}, "key_map": {"text": "reply", "audio_b64": "reply_audio"}})
    if s.get("identify_speakers"):   # runs beside transcribe; the prompt waits for its tag. Typed turns get the tag as an input instead. {speaker_list} is filled per turn from the login's profiles.
        nodes.insert(1, {"id": "who", "name": "Who Spoke", "type": "identify_speaker", "config": {"conn_id": s.get("speaker_conn", ""), "match_threshold": _opt(s, "speaker_match"), "guess_threshold": _opt(s, "speaker_guess"), "tag_format": _opt(s, "speaker_tag_format"), "other_name": _opt(s, "speaker_other_name")}, "key_map": {}})
        nodes[2].update(config={"mode": "template", "template": "{history}{speaker_tag}{transcript}"}, extra_in_keys=["history", "speaker_tag"])
        nodes[3]["config"]["system_prompt"] = f"""{nodes[3]["config"]["system_prompt"]}\n\n{{speaker_list}}""".strip()
    return _tune_flow({"nodes": [{**n, "status": "idle"} for n in nodes]}, s)

def _history(doc: dict) -> str:
    """Prior turns as plain text, newest kept whole, oldest dropped first to fit history_chars. Appending a turn only extends the text, so the model's cached prefix stays valid."""
    lines, size, budget = [], 0, int(doc["settings"].get("history_chars") or 0)
    for t in reversed([t for t in doc["turns"] if t.get("text")]):
        line = f"""{(t.get("speaker_tag") or "User: ") if t["role"] == "user" else "Assistant: "}{t["text"]}"""
        if size + len(line) > budget: break
        lines.insert(0, line); size += len(line) + 1
    return f"""Conversation so far:\n{chr(10).join(lines)}\n\nThe user now says: """ if lines else ""

async def _start_turn(request, doc: dict, inputs: dict, imr, barge_of: str = ""):
    """barge_of: the running turn this utterance talked over. This turn then skips the lane and starts at once - its transcript decides whether that turn stops, so it cannot wait behind it."""
    s, sid, username = doc["settings"], doc["id"], request.state.user.username
    kind, extra = ("id", {"pipeline_id": s["pipeline_id"]}) if s.get("pipeline_id") else ("inline", {"inline_flow": _flow(s)})
    job_id, err = AIM.engine.submit(username, kind=kind, inputs={**inputs, "history": _history(doc), "chat_id": sid, **_speaker_inputs(username, s, typed="transcript" in inputs)}, lane="" if barge_of else f"voice:{username}", **extra)
    if err: return imr.raw(CM.append_system_error_html(sid, err))
    _TURNS[sid], _ENDED[job_id] = job_id, asyncio.Event()
    imr.raw(CM.working_html(sid, {"type": "voice_stop", "job_id": job_id, "branch": IM.branch_id, "lvl": 2}))   # the call-mode barge-in presses this Stop button
    imr.raw(f"""<div id="cm-stream-{sid}" hx-swap-oob="innerHTML"><div class="cm-content voice-live" data-stream-job="{job_id}" data-stream-node="{UI.escape(s.get("stream_node") or "llm")}"></div></div>""")
    asyncio.create_task(_watch(username, sid, job_id, typed="transcript" in inputs, recording=inputs.get("audio_b64", "") if s.get("keep_recordings") else "", barge_of=barge_of))
    return imr

def _failed(job: dict) -> str: return next((f"""{n.get("name") or n["id"]}: {n.get("message", "")}""" for n in job["flow"]["nodes"] if n.get("status") == "error"), job["log"][-1] if job.get("log") else job["status"])

async def _push(username, html: str): await WS.send_personal_message(html, username)

def _streamed(job: dict) -> bool: return any(n.get("config", {}).get("speak_while_writing") for n in job["flow"]["nodes"] if n.get("type") == "generate")   # the reply was already played sentence by sentence

async def _watch(username: str, sid: str, job_id: str, typed: bool, recording: str = "", barge_of: str = ""):
    """Follows one turn's job: the user's words appear once transcribed (with their recording, when kept); the reply appears with its audio once spoken - its text streams live meanwhile - or as text alone if the run ends without audio.
    A spoken turn tells the browser whether words were heard as soon as transcription ends (a paused reply resumes when not); a turn that talked over a running one (barge_of) stops it when words were heard and lets it finish when not."""
    shown, job, mine = set(), None, lambda: _TURNS.get(sid) == job_id   # a newer turn owns the chat's working indicator and live text; an older one must not clear them
    while True:
        await asyncio.sleep(0.4)
        try: job = AIM.engine.load_job(job_id)
        except json.JSONDecodeError: continue   # the engine rewrites the job file between waves; a read can land mid-write
        data, done = job.get("data", {}), job["status"] in _DONE
        if not typed and "user" not in shown and ("transcript" in data or done):
            shown.add("user")
            heard = await _heard(username, sid, job)
            if barge_of and heard: AIM.engine.stop(barge_of, now=True); await _until_ended(barge_of)   # the reply it cut off lands in the chat first, so the conversation reads in order
            if heard and "transcript" in data:
                who = {k: data[k] for k in ("speaker", "speaker_role", "speaker_score", "speaker_tag") if data.get(k) not in (None, "")}
                await _push(username, CM.append_message_html(sid, _msg(_add_turn(sid, "user", data["transcript"], job_id, recording, **who), username), viewer_name=username) + (f"""<div id="voice-status" hx-swap-oob="innerHTML"><span class="status-label">Heard {UI.escape(who["speaker_tag"].strip().rstrip(":"))} - score {who.get("speaker_score", 0):.2f}</span></div>""" if who.get("speaker_tag") else ""))
            elif job["status"] == "done": await _push(username, """<div id="voice-status" hx-swap-oob="innerHTML"><span class="status-label">Nothing heard - ignored.</span></div>""")
        if "reply" in data and "reply" not in shown and ("reply_audio" in data or done):
            shown.add("reply")
            await _push(username, CM.append_message_html(sid, _msg(_add_turn(sid, "assistant", data["reply"], job_id, data.get("reply_audio", "")), username, autoplay=not _streamed(job)), viewer_name=username) + (CM.stream_clear_html(sid) if mine() else ""))
        if done: break
    tail = CM.working_hide_html(sid) + CM.stream_clear_html(sid) if mine() else ""
    if mine(): _TURNS.pop(sid)
    _ENDED.pop(job_id).set()
    if job["status"] == "stopped": tail += f"""<div id="voice-status" hx-swap-oob="innerHTML"><span class="status-label">Turn stopped.</span></div>"""
    elif job["status"] != "done": tail += CM.append_system_error_html(sid, f"""Turn {job["status"]} - {_failed(job)}""")
    if tail: await _push(username, tail)

async def _until_ended(job_id: str, timeout_s: float = 10):
    """Waits for a turn's watcher to finish (it is gone already when not listed). Bounded, so a stuck engine never holds the newer turn back for good."""
    if job_id in _ENDED:
        try: await asyncio.wait_for(_ENDED[job_id].wait(), timeout_s)
        except asyncio.TimeoutError: pass

# --- HTML ---

def _post(action: str, **extra) -> str: return f"""hx-post="/im/in" hx-target="body" hx-swap="none" hx-vals='{json.dumps({"type": f"voice_{action}", "lvl": 2, **extra})}'"""

def _msg(t: dict, username: str, autoplay: bool = False) -> dict:
    """A stored turn as a ChatManager message. Autoplay is only for the freshly pushed reply, never for re-renders."""
    label = (t.get("speaker_tag") or "").strip().rstrip(":") if t.get("speaker_role") != "owner" else ""   # other voices show under their tag; the owner's turns stay this login's own
    return {"id": t["id"], "role": t["role"], "content": t["text"], "user_name": (label or username) if t["role"] == "user" else "", "persona_name": "" if t["role"] == "user" else "Assistant", "timestamp": t.get("ts", ""), "audio": {"url": f"""{PREFIX}/audio/{t["audio"]}""", "autoplay": autoplay} if t.get("audio") else None}

def _list_html(username: str, active: str) -> str:
    rows = "".join(f"""<div class="list-card glass" style="{"border-left:.15rem solid var(--accent)" if d["id"] == active else ""}"><div class="list-card-hd"><span class="list-card-title" {_post("open", sid=d["id"])}>{UI.escape(d.get("title", ""))}</span><button class="btn-icon" style="color:#ff5f5f" {_post("delete", sid=d["id"])} hx-confirm="Delete this conversation and its audio?">&#x2715;</button></div><div class="status-label">{d.get("modified", "")[:16].replace("T", " ")}</div></div>""" for d in _list(username))
    return rows or '<div class="list-empty">No conversations.</div>'

def _main_html(doc: dict, username: str, status: str = "") -> str:
    sid = doc["id"]
    return f"""<div class="voice-head"><b>{UI.escape(doc["title"])}</b>
                   <details class="status-details" {"open" if status else ""}><summary>Settings</summary><form {_post("settings_save", sid=sid)} hx-include="this" class="node-form"><div id="voice-settings-fields">{_FIELDS.render(doc["settings"], name_prefix="vs_")}</div><div class="form-actions"><button type="submit" class="ui-btn">Save</button><button type="button" class="ui-btn" {_post("save_flow", sid=sid)}>Save built-in flow as pipeline</button><span id="voice-saved"></span></div></form></details>
                   <details class="status-details"><summary>Speakers</summary><div id="voice-speakers">{_speakers_html(username)}</div></details>
                   <div class="form-actions"><button class="ui-btn" {_post("prepare", sid=sid)} title="Loads the model and caches the system prompt so the first real turn starts fast">Prepare</button><button class="ui-btn" {_post("new")} title="Starts a fresh conversation with these settings; this one stays in the list">Restart conversation</button><span id="voice-status">{status}</span></div></div>
               <div class="voice-chat">{CM.shell(sid, messages=[_msg(t, username) for t in doc["turns"]], viewer_name=username, voice_opts=_call_cfg(doc["settings"]))}</div>"""

def _speakers_html(username: str, note: str = "") -> str:
    armed = _ENROLL.get(username)
    roles = lambda p: "".join(f"""<option value="{k}" {"selected" if p.get("role") == k else ""}>{UI.escape(v)}</option>""" for k, v in SPEAKER_ROLES)
    rows = "".join(f"""<div class="voice-spk-row"><b>{UI.escape(p["name"])}</b><span class="status-label">({p.get("code", "")})</span><select name="role" {_post("spk_role", pid=p["id"])} hx-trigger="change" hx-include="this">{roles(p)}</select><span class="status-label">{len(p.get("samples", []))} sample(s)</span><button class="ui-btn" {_post("spk_sample", pid=p["id"])} {"disabled" if armed else ""}>Record sample</button><button class="btn-icon" style="color:#ff5f5f" {_post("spk_delete", pid=p["id"])} hx-confirm="Delete this voice profile and its voiceprints?">&#x2715;</button></div>""" for p in _spk(username)["profiles"])
    wait = f"""<div class="voice-spk-note">Next recording is a voice sample for <b>{UI.escape((_spk_get(_spk(username), armed) or {}).get("name", "?"))}</b>: hold the mic and speak naturally for 5-10 s. <button class="ui-btn" {_post("spk_cancel")}>Cancel</button></div>""" if armed else ""
    form = f"""<form class="voice-spk-row" {_post("spk_add")} hx-include="this"><input name="name" placeholder="Name" required><select name="role">{roles({"role": "trusted"})}</select><button type="submit" class="ui-btn" {"disabled" if armed else ""}>Add and record a sample</button></form>"""
    return f"""{rows or '<div class="list-empty">No voices enrolled.</div>'}{wait}{form}<div class="status-label">Voiceprints only (numbers derived from the samples); recordings used for enrollment are not kept. 3+ samples per person, in the rooms you use, work best.{" " + note if note else ""}</div>"""

def _side_html(username: str, doc: dict) -> str: return f"""<div class="list-panel voice-side"><div class="list-panel-hd"><span class="list-panel-title">Conversations</span><button class="btn-icon" {_post("new")}>+</button></div><div id="voice-list" class="list-body">{_list_html(username, doc["id"])}</div></div>"""

def _page_html(username: str, doc: dict) -> str: return f"""<div class="voice-shell"><div id="voice-main" class="voice-main">{_main_html(_backfill(doc), username)}</div></div>"""

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
    llm = next(n for n in _flow(doc["settings"])["nodes"] if n["id"] == "llm")   # the tuned node, so the cached system prompt includes the spoken-style instruction exactly as real turns send it
    flow = {"nodes": [{**llm, "config": {**{k: v for k, v in llm["config"].items() if not k.startswith("speak_")}, "user_template": "Reply with the single word: ready", "num_predict": 16}, "key_map": {"input": "go", "text": "ready"}}]}
    username = request.state.user.username
    job_id, err = AIM.engine.submit(username, kind="inline", inline_flow=flow, inputs={"go": "1", **_speaker_inputs(username, doc["settings"], typed=True)}, lane=f"voice:{username}")
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
    who = {"speaker": _owner(username)["name"], "speaker_role": "owner", "speaker_tag": _tag(doc["settings"], _owner(username))} if doc["settings"].get("identify_speakers") else {}
    imr.oob("", "voice-status").raw(CM.append_message_html(doc["id"], _msg(_add_turn(doc["id"], "user", text, "", **who), username), viewer_name=username))
    return await _start_turn(request, doc, {"transcript": text}, imr)   # doc as loaded before this turn was added, so history does not repeat it

async def _h_audio(request, payload, imr):
    """ChatManager mic (hold-to-talk or call mode): {cid, audio_b64 = 16 kHz mono WAV data URL}.
    An utterance that talked over a reply comes with a Stop press for the running turn (the browser sends it first; it can still arrive just after): that turn becomes barge_of."""
    doc = _owned(request, payload.get("cid", ""))
    if not doc or not payload.get("audio_b64"): return imr
    if request.state.user.username in _ENROLL: return await _enroll(request.state.user.username, doc["settings"], payload["audio_b64"], imr)
    old = _TURNS.get(doc["id"], "")
    for _ in range(int(BARGE_WAIT_S / 0.05) if old else 0):
        if old in _STOP_ASKED: break
        await asyncio.sleep(0.05)
    if old in _STOP_ASKED: _BARGE[old] = True
    return await _start_turn(request, doc, {"audio_b64": payload["audio_b64"]}, imr.oob("", "voice-status"), barge_of=old if old in _BARGE else "")

async def _h_stop(request, payload, imr):
    """Stop button: the turn stops at once, its reply cut where it was. When an utterance follows within BARGE_GRACE_S (a call-mode barge-in), that utterance decides instead."""
    job_id = payload.get("job_id", "")
    _STOP_ASKED[job_id] = True
    asyncio.create_task(_stop_unless_talked_over(job_id))
    return imr.oob('<span class="status-label">Stopping&#x2026;</span>', "voice-status")

async def _stop_unless_talked_over(job_id: str):
    await asyncio.sleep(BARGE_GRACE_S)
    if not _BARGE.pop(job_id, False): AIM.engine.stop(job_id, now=True)
    _STOP_ASKED.pop(job_id, None)

async def _enroll(username: str, s: dict, audio_b64: str, imr):
    """The recording armed by Record sample: its voiceprint joins the profile and the profile's mean is recomputed. The audio itself is not stored."""
    pid, r = _ENROLL.pop(username), await _embed(s, audio_b64)
    doc = _spk(username); p = _spk_get(doc, pid)
    if r.get("error") or not p: return imr.oob(_speakers_html(username, f"""<span class="err-box">Sample not added: {UI.escape(r.get("error") or "profile was deleted")}</span>"""), "voice-speakers")
    p["samples"].append(r["embedding"]); p["embedding"] = _mean(p["samples"])
    _spk_save(username, doc)
    return imr.oob(_speakers_html(username, f"""&#x2713; Sample {len(p["samples"])} added for {UI.escape(p["name"])} ({r.get("duration_s", 0)} s)."""), "voice-speakers")

async def _h_spk_add(request, payload, imr):
    username, name = request.state.user.username, str(payload.get("name", "")).strip()
    if not name: return imr
    doc = _spk(username); used = {p.get("code") for p in doc["profiles"]}
    p = {"id": f"sp_{uuid.uuid4().hex[:8]}", "name": name, "code": next(c for c in (f"{random.randrange(1000, 10000)}" for _ in iter(int, 1)) if c not in used and c != "0000"), "role": payload.get("role") if payload.get("role") in dict(SPEAKER_ROLES) else "guest", "samples": [], "embedding": [], "created": datetime.utcnow().isoformat()}
    doc["profiles"].append(p); _spk_save(username, doc); _ENROLL[username] = p["id"]
    return imr.oob(_speakers_html(username), "voice-speakers")

async def _h_spk_sample(request, payload, imr):
    if _spk_get(_spk(request.state.user.username), payload.get("pid", "")): _ENROLL[request.state.user.username] = payload["pid"]
    return imr.oob(_speakers_html(request.state.user.username), "voice-speakers")

async def _h_spk_cancel(request, payload, imr): _ENROLL.pop(request.state.user.username, None); return imr.oob(_speakers_html(request.state.user.username), "voice-speakers")

async def _h_spk_role(request, payload, imr):
    username = request.state.user.username; doc = _spk(username); p = _spk_get(doc, payload.get("pid", ""))
    if p and payload.get("role") in dict(SPEAKER_ROLES): p["role"] = payload["role"]; _spk_save(username, doc)
    return imr.oob(_speakers_html(username), "voice-speakers")

async def _h_spk_delete(request, payload, imr):
    username = request.state.user.username; doc = _spk(username)
    doc["profiles"] = [p for p in doc["profiles"] if p["id"] != payload.get("pid")]; _spk_save(username, doc)
    if _ENROLL.get(username) == payload.get("pid"): _ENROLL.pop(username)
    return imr.oob(_speakers_html(username), "voice-speakers")

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
    SPEAKER_DIR.mkdir(parents=True, exist_ok=True)
    _FIELDS = _build_fields()
    IM = env["InterfaceManager"](nesting_level=2, db_path="voice_im.db")
    CM = BI.ChatManager(namespace="voice", base_url=prefix, view_style="bubble", stream_toggle=False, think_toggle=False, stop_enabled=True, pin_enabled=True, allow_edit=False, allow_delete=False, allow_copy=True, show_info=False, markdown_mode="standard", placeholder="Type, hold \U0001F3A4 to talk, or \u260E for a hands-free call (Ctrl+Enter sends)", branch_id=IM.branch_id, nesting_level=2, intent_name="voice_text", voice=True, audio_intent="voice_audio")
    IM.scripts.update({f"voice_{k}": [fn] for k, fn in {"new": _h_new, "open": _h_open, "delete": _h_delete, "settings_save": _h_settings_save, "save_flow": _h_save_flow, "text": _h_text, "audio": _h_audio, "stop": _h_stop, "settings_refresh": _h_settings_refresh, "prepare": _h_prepare,
                       "spk_add": _h_spk_add, "spk_sample": _h_spk_sample, "spk_cancel": _h_spk_cancel, "spk_role": _h_spk_role, "spk_delete": _h_spk_delete}.items()})
    print("[voice] ready")

@router.get("")
@router.get("/")
async def root(request: Request):
    doc = await _active(request)
    left = UI.toolbar(side="left", content=_side_html(request.state.user.username, doc), size="16rem", overlay=False, start_open=True, resizable=True, id="voice-left-bar", nesting_level=2)
    return ENV["templates"].TemplateResponse(name="base.html", request=request, context={"request": request, "user": request.state.user, "nesting_level": 2, "shell_id": IM.branch_id, "toolbars": {"left": left}, "content": _page_html(request.state.user.username, doc), "extra_css": CSS + CM.CSS, "extra_script": CM.SCRIPT + SCRIPT})

@router.get("/audio/{name}")
async def audio(request: Request, name: str):
    """A turn's audio - the requesting user's own folder only."""
    p = _audio_path(request.state.user.username, name)
    if not p.is_file(): raise HTTPException(404)
    return FileResponse(p, media_type="audio/wav")

CSS = """
.voice-shell{display:flex;height:100%;width:100%;overflow:hidden;}
.voice-side{height:100%;border-right:var(--border-thick) solid var(--border);}
.voice-main{flex:1;display:flex;flex-direction:column;gap:.4rem;padding:.6rem;overflow:hidden;min-width:0;}
.voice-head{flex-shrink:0;max-height:45vh;overflow-y:auto;}
.voice-chat{flex:1;min-height:0;display:flex;}
.voice-live{white-space:pre-wrap;opacity:.8;}
.voice-spk-row{display:flex;flex-wrap:wrap;align-items:center;gap:.4rem;margin:.25rem 0;}
.voice-spk-note{margin:.3rem 0;padding:.3rem .5rem;border-left:.15rem solid var(--accent);}
"""

SCRIPT = """
(function(){ var b = document.getElementById('voice-left-bar'); if (b && typeof toggleToolbar === 'function' && b.classList.contains('active') && window.matchMedia('(max-width:46rem)').matches) toggleToolbar('voice-left-bar', 'left', '16rem', 'shrink', '&#x25C0;', '&#x25B6;'); })();   // narrow screens start with the conversation list collapsed; its arrow opens it
"""
