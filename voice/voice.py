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
SPEAKER_DIR = DATA_DIR / "speakers"   # one file per login: enrolled voice profiles (voiceprints + the names of their sample recordings, kept under audio/<login>/)
DEFAULTS_DIR = DATA_DIR / "defaults"  # one file per login: the settings a conversation starts from when there is none to copy (written by every settings save)
MEMORY_DIR = DATA_DIR / "memory"      # one file per login: things to remember, sent with every turn's system prompt when Use Memory is on
BACKUP_DIR = DATA_DIR / "backups"     # one folder per login: a full bundle written before every import
_DONE = ("done", "error", "stopped", "interrupted")
ENV = {}
PREFIX = ""
UI = WS = IM = BI = AIM = CM = _FIELDS = BUNDLE = None
_TURNS: dict = {}        # chat id -> job id of the turn running in it now (the one that owns the chat's working indicator and live text)
_ENDED: dict = {}        # job id -> asyncio.Event set when that turn's watcher has finished
_ENROLL: dict = {}       # username -> speaker profile id: that login's next recording is a voice sample for the profile, not a turn
_STOP_ASKED: dict = {}   # job id -> True while a Stop press for it waits out BARGE_GRACE_S
_BARGE: dict = {}        # job id of a turn a newer utterance talked over -> True; the newer turn's transcript decides whether it stops (cleared when that turn's Stop press is resolved)
_AWAKE: dict = {}        # chat id -> loop time until which a line needs no wake word (a wake word, an addressed line or a finished reply extends it; the sleep command ends it)
BARGE_WAIT_S, BARGE_GRACE_S = 0.25, 0.6   # how long an utterance waits for a Stop press that may arrive just after it / how long a Stop press waits for an utterance before stopping at once

# --- Conversation storage ---

def _dp(sid): return DATA_DIR / f"{Path(sid).name}.json"
def _load(sid): return json.loads(_dp(sid).read_text()) if sid and _dp(sid).exists() else None
def _save(doc): doc["modified"] = datetime.utcnow().isoformat(); BI.write_json_atomic(_dp(doc["id"]), doc)
def _list(username): return sorted((d for d in (json.loads(f.read_text()) for f in DATA_DIR.glob("*.json")) if d.get("username") == username), key=lambda d: d.get("modified", ""), reverse=True)
def _owned(request, sid): doc = _load(sid); return doc if doc and doc.get("username") == request.state.user.username else None

def _def_path(username: str) -> Path: return DEFAULTS_DIR / f"{Path(username).name}.json"
def _defaults(username: str) -> dict: return {**{f.name: f.default for f in _FIELDS.fields}, **(json.loads(_def_path(username).read_text()) if _def_path(username).exists() else {})}   # this login's last saved settings over the field defaults, so fields added later keep theirs
def _spk_save(username: str, doc: dict): BI.write_json_atomic(_spk_path(username), doc, indent=1)

def _new(username, settings=None) -> dict:
    doc = {"id": f"vc_{uuid.uuid4().hex[:8]}", "username": username, "title": "New conversation", "settings": settings or _defaults(username), "turns": [], "created": datetime.utcnow().isoformat()}
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
# A profile: {id, name, code (4-digit signature), role (owner / trusted / guest), samples, embedding (normalized mean of the samples' voiceprints), created}.
# A sample is {audio (file under audio/<login>/), embedding (or None when measuring failed)}; profiles enrolled before recordings were kept hold bare voiceprints (lists) - both forms are read.

SPEAKER_ROLES = [("owner", "Owner - this account's person"), ("trusted", "Trusted - known, not the owner"), ("guest", "Guest")]

def _spk_path(username: str) -> Path: return SPEAKER_DIR / f"{Path(username).name}.json"
def _spk(username: str) -> dict: return json.loads(_spk_path(username).read_text()) if _spk_path(username).exists() else {"profiles": []}
def _spk_save(username: str, doc: dict): BI.write_json_atomic(_spk_path(username), doc, indent=1)
def _spk_get(doc: dict, pid: str): return next((p for p in doc["profiles"] if p["id"] == pid), None)

def _mean(vectors: list) -> list:
    m = [sum(c) / len(vectors) for c in zip(*vectors)]
    n = sum(a * a for a in m) ** 0.5 or 1.0
    return [round(a / n, 6) for a in m]

def _sample_vec(x): return x if isinstance(x, list) else x.get("embedding")
def _voiceprint(p: dict) -> list: return _mean([v for v in map(_sample_vec, p["samples"]) if v])
def _sample_note(p: dict) -> str:
    n, missing = len(p["samples"]), sum(1 for x in p["samples"] if not _sample_vec(x))
    return f"{n} sample(s), {missing} without voiceprint" if missing else f"{n} sample(s)"

def _free_code(doc: dict) -> str:
    used = {p.get("code") for p in doc["profiles"]}
    return next(c for c in (f"{random.randrange(1000, 10000)}" for _ in iter(int, 1)) if c not in used and c != "0000")

def _new_profile(doc: dict, name: str, role: str) -> dict:
    p = {"id": f"sp_{uuid.uuid4().hex[:8]}", "name": name, "code": _free_code(doc), "role": role if role in dict(SPEAKER_ROLES) else "guest", "samples": [], "embedding": [], "created": datetime.utcnow().isoformat()}
    doc["profiles"].append(p)
    return p

def _profile_named(username: str, name: str) -> dict:
    """This login's profile with that name (any case), created as a guest when there is none."""
    doc = _spk(username)
    p = next((p for p in doc["profiles"] if p["name"].lower() == name.strip().lower()), None)
    if not p: p = _new_profile(doc, name.strip().title(), "guest"); _spk_save(username, doc)
    return p

async def _add_sample(username: str, s: dict, pid: str, audio_b64: str) -> str:
    """Keeps the recording (audio/<login>/spk_<profile>_<id>.wav), measures it and adds its voiceprint to the profile's mean. A failed measurement still keeps the recording for Rebuild. Returns the note (HTML)."""
    r = await _embed(s, audio_b64)
    doc = _spk(username); p = _spk_get(doc, pid)   # read after the measurement, so a change made meanwhile is not overwritten
    if not p: return '<span class="err-box">Sample not added: the profile was deleted</span>'
    p["samples"].append({"audio": _write_audio(username, f"spk_{pid}_{uuid.uuid4().hex[:6]}", audio_b64), "embedding": None if r.get("error") else r["embedding"]})
    p["embedding"] = _voiceprint(p)
    _spk_save(username, doc)
    return f"""<span class="err-box">Recording kept, but no voiceprint was made: {UI.escape(r["error"])} - press Rebuild once the speaker connection works.</span>""" if r.get("error") else f"""&#x2713; Sample {len(p["samples"])} added for {UI.escape(p["name"])} ({r.get("duration_s", 0)} s)."""

async def _enroll(username: str, s: dict, audio_b64: str, imr): return imr.oob(_speakers_html(username, await _add_sample(username, s, _ENROLL.pop(username), audio_b64)), "voice-speakers")

async def _sample_from_turn(username: str, s: dict, name: str, audio: str) -> str:
    """The model's sample action: this turn's own recording becomes a voice sample for the named person."""
    if not audio: return "Not done: a voice sample needs a spoken turn."
    if not name.strip(): return "Not done: the sample tag named nobody."
    note = await _add_sample(username, s, _profile_named(username, name)["id"], audio)
    await _push(username, f"""<div id="voice-speakers" hx-swap-oob="innerHTML">{_speakers_html(username)}</div>""")
    return note

def _tag(s: dict, p: dict, guess: str = "") -> str: return str(_opt(s, "speaker_tag_format")).format(name=p["name"], code=p.get("code", ""), role=p.get("role", ""), guess=guess)

def _owner(username: str) -> dict: return next((p for p in _spk(username)["profiles"] if p.get("role") == "owner"), {"name": username, "code": "", "role": "owner"})

def _speaker_inputs(username: str, s: dict, typed: bool) -> dict:
    """Turn inputs for speaker identification: the enrolled profiles, the mapping text for the system prompt and - for typed turns, which come from the logged-in person - the owner's tag and role.
    With identification off nobody is identified, so every turn counts as the owner's (the route node and model actions read speaker_role)."""
    if not s.get("identify_speakers"): return {"speaker_role": "owner"}
    profiles = [{k: p[k] for k in ("name", "code", "role", "embedding")} for p in _spk(username)["profiles"] if p.get("embedding")]
    other = {"name": _opt(s, "speaker_other_name"), "code": "0000", "role": "unknown"}
    listing = "; ".join(f"""{_tag(s, p).strip().rstrip(":")} is {p["name"]} ({p["role"]})""" for p in profiles) or "no one is enrolled yet"
    return {"profiles": profiles, "speaker_list": str(_opt(s, "speaker_prompt")).replace("{speakers}", listing).replace("{other}", _tag(s, other).strip().rstrip(":")), **({"speaker_tag": _tag(s, _owner(username)), "speaker_role": "owner"} if typed else {})}

async def _embed(s: dict, audio_b64: str) -> dict:
    conn = _conn_for(s, "speaker_conn", "speaker_embed")
    return await AIM.connections.call_capability(conn, "speaker_embed", {"audio_b64": audio_b64}) if conn else {"error": "no connection declares speaker_embed (hook speaker_id.py into the speech node and refresh its profile)"}

# --- Memory (per login) ---
# One file per login: {"items": [...], "requests": [...]}. An item is {id, kind, about, text, by, ts}:
#   fact  - about the people here; style - how to understand and talk with them. Added by the login, spoken commands, or the model (role-gated); the login edits them freely.
#   note  - the assistant's own notes. It adds, edits and removes them; the login can read them and asks before changing one (a request the assistant answers),
#           or removes one anyway - then the assistant is told that a note was removed, not what it said.
# Styles and notes are always sent with each turn's system prompt (Use Memory on); facts are sent newest first within memory_chars, and the panel marks those left out.

_MEM_KINDS = {"style": "How to understand and talk with the people here", "fact": "About the people here", "note": "Your own notes (kept for you between conversations; the owner can read them and asks before changing one)"}

def _mem_path(username: str) -> Path: return MEMORY_DIR / f"{Path(username).name}.json"
def _mem_save(username: str, d: dict): BI.write_json_atomic(_mem_path(username), d)
def _mem_budget(username: str) -> int: return int(_defaults(username).get("memory_chars") or 0) or 10**9   # memory is per login, so its budget comes from the login's last saved settings; 0 = no limit
def _mem_line(i: dict) -> str: return f"""- [{i["id"]}] {"(" + i["about"] + ") " if i.get("about") else ""}{i["text"]}"""
def _now_iso() -> str: return datetime.utcnow().isoformat()

def _mem(username: str) -> dict:
    d = json.loads(_mem_path(username).read_text()) if _mem_path(username).exists() else {"items": [], "requests": []}
    return {"items": [{"kind": "fact", **i} for i in d], "requests": []} if isinstance(d, list) else d   # files from before kinds existed held a plain list of facts

def _mem_put(username: str, kind: str, text: str, by: str, about: str = "", mid: str = "") -> dict:
    """Adds an item, or replaces the text of item mid when it exists with the same kind."""
    d, text = _mem(username), " ".join(str(text).split())
    item = next((i for i in d["items"] if i["id"] == mid and i["kind"] == kind), None)
    if item: item.update(text=text, by=by, ts=_now_iso(), **({"about": about} if about else {}))
    else: item = {"id": f"m_{uuid.uuid4().hex[:8]}", "kind": kind, "about": about, "text": text, "by": by, "ts": _now_iso()}; d["items"].append(item)
    _mem_save(username, d)
    return item

def _facts_sent(d: dict, budget: int) -> list:
    """Facts newest first while they fit the budget (chars), returned oldest first."""
    kept, used = [], 0
    for i in sorted((i for i in d["items"] if i["kind"] == "fact"), key=lambda i: i["ts"], reverse=True):
        if used + len(_mem_line(i)) > budget: break
        kept.append(i); used += len(_mem_line(i)) + 1
    return kept[::-1]

def _mem_text(username: str) -> str:
    """The memory block for the system prompt: styles, facts that fit the budget, the assistant's notes, then the owner's open requests about those notes."""
    d = _mem(username)
    groups = {"style": [i for i in d["items"] if i["kind"] == "style"], "fact": _facts_sent(d, _mem_budget(username)), "note": [i for i in d["items"] if i["kind"] == "note"]}
    parts = [f"""{_MEM_KINDS[k]}:\n{chr(10).join(map(_mem_line, v))}""" for k, v in groups.items() if v]
    asks = [r for r in d["requests"] if r["status"] == "open"]
    if asks: parts.append("Requests about your notes (answer each with a note tag):\n" + "\n".join(f"""- [{r["id"]}] the owner asks you to change or remove [{r["note_id"]}]: {r.get("reason", "")}""" for r in asks))
    return "\n\n".join(parts)

def _note_act(username: str, attrs: dict, body: str) -> str:
    """The assistant's notes: add (no id), or edit / remove / keep one of its notes by id (keep = answer an owner's request with its reason in the body). An answer closes the open requests on that note."""
    d, nid = _mem(username), attrs.get("id", "")
    act = str(attrs.get("action") or ("edit" if nid else "add")).lower()
    if act == "add":
        if not body.strip(): return "Not done: an empty note."
        return f"""Assistant note added: {UI.escape(_mem_put(username, "note", body, "assistant")["text"])}"""
    note = next((i for i in d["items"] if i["id"] == nid and i["kind"] == "note"), None)
    if not note: return f"""Not done: the assistant referred to a note {UI.escape(nid)} that does not exist."""
    if act == "edit": note.update(text=" ".join(body.split()), ts=_now_iso())
    elif act == "remove": d["items"].remove(note)
    elif act != "keep": return f"""Not done: unknown note action {UI.escape(act)}."""
    for r in d["requests"]:
        if r["note_id"] == nid and r["status"] == "open": r.update(status={"remove": "granted", "keep": "declined"}.get(act, "edited"), answer=" ".join(body.split()), answered=_now_iso())
    _mem_save(username, d)
    return f"""Assistant {"removed" if act == "remove" else "kept" if act == "keep" else "edited"} its note {UI.escape(nid)}{": " + UI.escape(body) if body.strip() and act == "keep" else ""}."""

def _mem_note(item: dict) -> str: return f"""Remembered: {UI.escape(item["text"])} <button class="ui-btn" {_post("mem_delete", mid=item["id"])}>Undo</button>"""

def _memory_html(username: str) -> str:
    d = _mem(username); sent = {i["id"] for i in _facts_sent(d, _mem_budget(username))}
    meta = lambda i: f"""{UI.escape(i.get("about", ""))} {UI.escape(i.get("by", ""))} {i.get("ts", "")[:10]}{" - over budget, not sent" if i["kind"] == "fact" and i["id"] not in sent else ""}"""
    edit = lambda i: f"""<div class="voice-spk-row"><input name="text" value="{UI.escape(i["text"])}" style="flex:1" {_post("mem_edit", mid=i["id"])} hx-trigger="change" hx-include="this"><span class="status-label">{meta(i)}</span><button class="btn-icon" style="color:#ff5f5f" {_post("mem_delete", mid=i["id"])}>&#x2715;</button></div>"""
    note = lambda i: f"""<div class="voice-spk-row"><span style="flex:1">{UI.escape(i["text"])}</span><span class="status-label">{i["id"]} {i.get("ts", "")[:10]}</span><form class="voice-spk-row" {_post("note_ask", nid=i["id"])} hx-include="this"><input name="reason" placeholder="Ask to change or remove - why?" required><button type="submit" class="ui-btn">Ask</button></form><button class="ui-btn" {_post("note_force", nid=i["id"])} hx-confirm="Remove this note without the assistant's answer? It will be told a note was removed, not what it said.">Remove anyway</button></div>"""
    sec = lambda kind, render: f"""<div class="voice-mem-sec"><b>{_MEM_KINDS[kind].split(" (")[0]}</b>{"".join(render(i) for i in d["items"] if i["kind"] == kind) or '<div class="list-empty">None yet.</div>'}</div>"""
    reqs = "".join(f"""<div class="status-label">[{r["id"]}] {r["note_id"]}: {UI.escape(r.get("reason", ""))} - {r["status"]}{": " + UI.escape(r["answer"]) if r.get("answer") else ""}</div>""" for r in d["requests"][-10:])
    form = f"""<form class="voice-spk-row" {_post("mem_add")} hx-include="this"><select name="kind"><option value="fact">Fact</option><option value="style">Style</option></select><input name="about" placeholder="About (optional)" style="width:8rem"><input name="text" placeholder="Something to remember" required style="flex:1"><button type="submit" class="ui-btn">Add</button></form>"""
    return f"""{sec("style", edit)}{sec("fact", edit)}{sec("note", note)}{f'<div class="voice-mem-sec"><b>Requests</b>{reqs}</div>' if reqs else ""}{form}<div class="status-label">Sent with every turn when Use Memory is on. Styles and the assistant's notes always; facts newest first within the memory budget.</div>"""

def _memory_oob(username: str) -> str: return f"""<div id="voice-memory" hx-swap-oob="innerHTML">{_memory_html(username)}</div>"""

# --- Bundle parts (built_ins.DataBundle does the transfer; these say what each part holds) ---

def _owned_doc(username: str, ctx: dict) -> dict:
    doc = _load(ctx["sid"])
    if doc["username"] != username: raise PermissionError("voice bundle: that conversation belongs to another login")
    return doc

def _settings_out(username: str, ctx: dict) -> dict: return {"settings": _owned_doc(username, ctx)["settings"], "defaults": json.loads(_def_path(username).read_text()) if _def_path(username).exists() else {}}

def _settings_in(username: str, ctx: dict, data: dict, replace: bool) -> str:
    doc = _owned_doc(username, ctx)
    doc["settings"] = dict(data["settings"]) if replace else {**doc["settings"], **data["settings"]}; _save(doc)
    cur = {} if replace or not _def_path(username).exists() else json.loads(_def_path(username).read_text())
    _save_defaults(username, {**cur, **(data.get("defaults") or {})})
    return "settings"

def _speakers_out(username: str, ctx: dict) -> dict:
    spk = _spk(username); names = [x["audio"] for p in spk["profiles"] for x in p["samples"] if isinstance(x, dict)]
    return {"profiles": spk["profiles"], "recordings": {n: base64.b64encode(_audio_path(username, n).read_bytes()).decode() for n in names if _audio_path(username, n).exists()}, "missing_recordings": [n for n in names if not _audio_path(username, n).exists()]}

def _speakers_in(username: str, ctx: dict, data: dict, replace: bool) -> str:
    """Voices merge by name: new samples are added and existing recordings kept; replace starts from no profiles (old recording files stay on disk)."""
    for name, b64 in (data.get("recordings") or {}).items():
        f = _audio_path(username, name)
        if not f.exists(): f.parent.mkdir(parents=True, exist_ok=True); f.write_bytes(base64.b64decode(b64))
    cur = {"profiles": []} if replace else _spk(username)
    for q in data["profiles"]:
        p = next((p for p in cur["profiles"] if p["name"].lower() == q["name"].lower()), None)
        if not p:
            if q.get("code") in {x.get("code") for x in cur["profiles"]}: q["code"] = _free_code(cur)
            cur["profiles"].append(q); continue
        have = {x["audio"] for x in p["samples"] if isinstance(x, dict)}
        p["samples"] += [x for x in q["samples"] if not ((isinstance(x, dict) and x["audio"] in have) or (isinstance(x, list) and x in p["samples"]))]
        p["embedding"] = _voiceprint(p)
    _spk_save(username, cur)
    return f"""{len(data["profiles"])} voice profile(s)"""

def _memory_in(username: str, ctx: dict, data, replace: bool) -> str:
    m = data if isinstance(data, dict) else {"items": [{"kind": "fact", **i} for i in data], "requests": []}
    cur = {"items": [], "requests": []} if replace else _mem(username)
    for k in ("items", "requests"): ids = {i["id"] for i in cur[k]}; cur[k] += [i for i in m.get(k, []) if i["id"] not in ids]
    _mem_save(username, cur)
    return "memory"

async def _after_import(request, ctx: dict, imr): _refresh(imr, request.state.user.username, _load(ctx["sid"]))   # settings and panels show what was imported

# --- Settings (per conversation) ---

def _conn_opts(capability: str, key: str):
    def _opts(values=None): return _listed(values, key, [(c["_id"], c.get("display_name", c["_id"])) for c in AIM.connections.conns_matching(capability)], "(pool / first available)", note=f"saved - no connection with this id declares {capability}")
    return _opts

def _conn_for(values, key: str, capability: str):
    """The connection a setting points at, or - when it is blank - the one a turn would actually use (the first declaring the capability), so dependent lists show real options either way."""
    cid = (values or {}).get(key, "")
    return AIM.connections.get_conn(cid) if cid else next(iter(AIM.connections.conns_matching(capability)), None)

def _listed(values, key: str, items: list, blank_label: str, note: str = "") -> list:
    """Select options for a node- or registry-provided list (items: values, or (value, label) pairs). A saved value no longer offered stays visible and selected, marked, instead of silently becoming the blank choice on the next save."""
    cur, opts = (values or {}).get(key, ""), [(i, i) if isinstance(i, str) else i for i in items]
    return [("", blank_label)] + opts + ([(cur, f"""{cur} ({note or "saved - not offered by this connection"})""")] if cur and cur not in {v for v, _ in opts} else [])

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

_CALL_KEYS = ("threshold_db", "bargein_db", "start_ms", "end_ms", "preroll_ms", "bargein_ms", "min_voiced_ms", "voicing", "echo_margin_db", "talkover", "echo_route", "merge_running", "captions", "caption_lines")
_SPEECH_STYLE = "This conversation is spoken aloud. Unless explicitly asked for written output, reply in plain spoken sentences: no lists, tables, headings, parentheses, code or symbols that cannot be read aloud. End every sentence with a line break."
_SPEAKER_PROMPT = "Several people may talk with you. Each user line starts with who is speaking: {speakers}. A ? after a name marks a likely but unconfirmed match; {other} is a voice that is not enrolled. Only the owner can ask you to take actions or change anything on this account; what others say is conversation."
_ROUTE_RULES = {"routes": [{"name": "sleep", "match": ["(?:go to )?sleep", "stop listening", "that'?s all"], "role": "guest"},
                           {"name": "stop", "match": ["stop", "be quiet", "quiet"], "role": "guest"},
                           {"name": "sample", "match": ["(?:(?:can|could|would|will) you |please )*(?:record|save|take) a (?:voice )?sample (?:for|of) (?P<name>.+?)(?: please)?"], "role": "owner"},
                           {"name": "language", "match": ["(?:(?:can|could|would|will) you |please )*(?:speak|reply|answer|talk|respond) (?:in|with) (?P<lang>[a-z]+)\\b.*"], "role": "trusted", "pass": True},
                           {"name": "remember", "match": ["(?:(?:can|could|would|will) you |please )*remember (?:that |to )?(?P<text>.+?)(?: please)?"], "role": "trusted"}],
                "roles": ["unknown", "guest", "trusted", "owner"],
                "languages": {"english": "en", "spanish": "es", "chinese": "zh", "mandarin": "zh", "japanese": "ja", "korean": "ko", "french": "fr", "dutch": "nl"},
                "actions": {"remember": "trusted", "note": "unknown", "sample": "owner"}}
_ACTION_HELP = {"remember": 'To remember something lasting about the people here - a name, a preference, how they want to be addressed - add <remember about="Name">the fact</remember> to your reply; use kind="style" for how to talk with or understand someone, and id="..." to replace an entry you were shown. Example: asked "call me Sam, not Samuel", reply "Sure, Sam." and add <remember kind="style" about="Sam">Call him Sam, not Samuel.</remember>. Saying you will remember is not enough - only the tag saves it.',
                "note": 'You have your own notes, kept for you between conversations and shown below; the owner can read them and asks before changing one. Write <note>text</note> to add one, <note id="..." action="edit">new text</note> or <note id="..." action="remove"/> to change your own, and answer a request with <note id="..." action="keep">your reason</note>, action="edit" or action="remove".',
                "sample": 'To save the voice of the person speaking now as a sample, add <sample name="Their name"/>.'}
_TOOLS_PROMPT = 'Tags are read by the system and never spoken. Text in another language goes inside <lang code="xx">...</lang>, using one of these codes: {languages}; text outside tags is spoken in the default voice. {actions}'
_NEW_DEFAULTS = {"max_reply_tokens": 300, "trim_unfinished": True, "call_bargein_ms": 400, "call_min_voiced_ms": 250, "call_voicing": 0.45, "ignore_phrases": "", "max_no_speech_prob": None, "speak_while_writing": True, "speech_style_prompt": _SPEECH_STYLE,
                 "speaker_match": 0.5, "speaker_guess": 0.35, "speaker_tag_format": "({name}{guess}): ", "speaker_other_name": "Other", "speaker_prompt": _SPEAKER_PROMPT, "call_echo_margin_db": 8, "call_talkover": True, "call_echo_route": True, "wake_fuzz": 0.8,
                 "call_merge_running": True, "call_captions": True, "call_caption_lines": 4, "route_turns": True, "wake_words": "", "wake_window_s": 20, "route_rules": _ROUTE_RULES, "ai_actions": True, "lang_voices": {}, "use_memory": True, "tools_prompt": _TOOLS_PROMPT, "memory_chars": 16000}

def _opt(s: dict, k: str): return _NEW_DEFAULTS[k] if k in _NEW_DEFAULTS and s.get(k) is None else s.get(k) # A conversation setting, or the shipped default when the conversation was saved before the setting existed. A blank value is kept: blank text means nothing (blank numbers already parse to their field default).
  
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
        gen["config"].update({"speak_while_writing": True, "speak_conn_id": c.get("conn_id", ""), "speak_voice": c.get("voice", ""), "speak_length_scale": c.get("length_scale"), "speak_sid": "{chat_id}", "speak_drop_unfinished": bool(_opt(s, "trim_unfinished")), "speak_lang_voices": c.get("lang_voices") or {}, "speak_action_tags": c.get("action_tags", "")})
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
    """For a spoken turn: True when words were heard and addressed - the routed transcript or a command came out (a run with no transcribe node counts as heard).
    Tells the browser either way with a `cm-voice` trigger {sid, heard}: heard=false resumes a reply the utterance paused, so talking to someone else in the room never cuts the assistant off."""
    if not any(n.get("type") == "transcribe" for n in job.get("flow", {}).get("nodes", [])): return True
    heard = "transcript" in job.get("data", {}) or "command" in job.get("data", {})
    await ENV["push_to_client"](username, {"t": "trigger", "event": "cm-voice", "detail": {"sid": sid, "heard": heard}})
    return heard

def _call_cfg(s: dict) -> dict: return {k: _opt(s, f"call_{k}") for k in _CALL_KEYS}   # blank values are dropped by ChatManager, so the browser script's defaults apply

def _build_fields():
    F = BI.SettingField
    refresh = {"hx_intent": "voice_settings_refresh", "hx_target": "#voice-settings-fields"}   # changing a connection re-renders the dependent lists (models, voices, languages) from the live form values
    return BI.SettingsGroup("voice", "", [
        F("consent_prefix", "System Prompt Prefix (consent / standing terms)", "textarea", "", hint="Sent first, before the system prompt, on every turn. It stays in the model's prompt cache, so after the first turn it costs almost nothing - see Prepare."),
        F("system_prompt", "System Prompt", "textarea", "You are a friendly voice assistant. Answer in short, natural spoken sentences - no lists, tables or code unless asked."),
        F("chat_conn", "Chat Connection", "select", "", options=_conn_opts("chat", "chat_conn"), **refresh),
        F("tts_conn", "Text-to-Speech Connection", "select", "", options=_conn_opts("text_to_speech", "tts_conn"), **refresh),
        F("stt_conn", "Speech-to-Text Connection", "select", "", options=_conn_opts("speech_to_text", "stt_conn"), advanced=True, **refresh),
        F("speaker_conn", "Speaker Identification Connection", "select", "", options=_conn_opts("speaker_embed", "speaker_conn"), advanced=True),
        F("model", "Model", "select", "", options=_model_opts),
        F("voice", "Voice", "select", "", options=_voice_opts),
        F("speak_replies", "Speak replies", "checkbox", True),
        F("language", "Spoken Language", "select", "", options=_lang_opts),
        F("keep_recordings", "Keep my recordings (replayable in the conversation)", "checkbox", True, hint="Off: your audio is transcribed and discarded; only the text is kept."),
        F("pipeline_id", "Pipeline", "select", "", options=lambda values=None: _listed(values, "pipeline_id", [(p["id"], p.get("name", p["id"])) for p in AIM.engine.list_pipelines()], "(built-in: transcribe -> generate -> speak)", note="saved - pipeline no longer exists"), advanced=True, hint="A saved pipeline must follow the key convention: in audio_b64 or transcript + history, out transcript / reply / reply_audio."),
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
        F("call_merge_running", "Call: Talking Again Before the Reply Is Spoken Continues Your Turn", "checkbox", True, advanced=True, hint="A pause mid-thought sends what you said so far; if you go on talking while that reply is still being written, it is cut and both parts are answered together. Off = every utterance gets its own reply."),
        F("call_captions", "Captions for Spoken Replies", "checkbox", True, hint="Shows each sentence above the input as it is spoken - including sentences the speech node could not speak."),
        F("call_caption_lines", "Caption Lines Kept", "number", 4, step=1, advanced=True),
        F("route_turns", "Route Turns (wake words and commands)", "checkbox", True, hint="Each turn passes the rules below before the model: a wake word can be required, and command phrases run here instead of going to the model. Built-in flow only."),
        F("wake_words", "Wake Words (comma-sep; blank = always listening)", "text", "", hint="e.g. Tessa, Tesa - add the ways the recognizer mishears the name."),
        F("wake_fuzz", "Wake Word Tolerance (0-1, 1 = exact only)", "number", 0.8, advanced=True, hint="How far the recognizer's spelling may be from a wake word. 0.8 accepts Tesa and Tess a for Tessa; lower accepts more, including false wakes."),
        F("wake_window_s", "Stay Addressed For (s after a wake word or a reply)", "number", 20, step=1),
        F("route_rules", "Route Rules (JSON)", "json", _ROUTE_RULES, advanced=True, hint="routes: {name, match: [regex over the whole line after the wake word; named groups become args], role}; roles: lowest to highest; languages: spoken name -> code; actions: model tag -> role needed."),
        F("ai_actions", "Let the Model Act Through Tags", "checkbox", True, hint="The model may write remember and sample tags; each runs only when the turn's speaker holds the role the rules give it."),
        F("lang_voices", "Voice per Language (JSON: code -> voice)", "json", {}, hint='e.g. {"es": "es_MX-claude-high", "zh": "zh_CN-huayan-medium"}. The model marks other-language text with <lang code="xx">; that text uses this voice. Leave Spoken Language blank so recognition detects each utterance.'),
        F("reply_language", "Reply Language (blank = the model's choice)", "text", "", advanced=True, hint="Also set by the 'speak in ...' command."),
        F("use_memory", "Use Memory", "checkbox", True),
        F("memory_chars", "Memory Budget for Facts (chars, 0 = no limit)", "number", 16000, step=1, advanced=True, hint="Style lines and the assistant's own notes are always sent; facts past this, oldest first, are left out and marked in the Memory panel. Memory is per login, so this comes from your last saved settings."),
        F("tools_prompt", "Tag Instruction (added to the system prompt)", "textarea", _TOOLS_PROMPT, advanced=True, hint="{languages} becomes the codes that have a voice; {actions} the allowed action tags. Blank = add nothing."),
        F("identify_speakers", "Identify Speakers (voiceprints)", "checkbox", False, hint="Prefixes each spoken turn with who said it, e.g. (Joshua): ... Enroll voices under Speakers. Needs speaker_id.py on the speech node."),
        F("speaker_match", "Same Person At Or Above (cosine 0-1)", "number", 0.5, advanced=True, hint="Same voice usually scores 0.6-0.8, different voices under 0.3. The score shows in the status line after each turn."),
        F("speaker_guess", "Likely Match At Or Above", "number", 0.35, advanced=True, hint="Between this and the match threshold the name gets a ? mark."),
        F("speaker_tag_format", "Speaker Tag Format", "text", "({name}{guess}): ", advanced=True, hint="Fields {name} {code} {role} {guess}. '({code}): ' uses number signatures; the system prompt lists what each means."),
        F("speaker_other_name", "Name for Unknown Voices", "text", "Other", advanced=True),
        F("speaker_prompt", "Speaker Instruction (added to the system prompt)", "textarea", _SPEAKER_PROMPT, advanced=True, hint="{speakers} becomes the enrolled list with roles, {other} the unknown-voice tag.")], json_path="")

# --- Routing (wake words, commands, model actions) ---    

def _rank(s: dict, role: str) -> int: roles = _rules(s).get("roles") or _ROUTE_RULES["roles"]; return roles.index(role) if role in roles else 0
def _action_tags(s: dict) -> list: return list(_rules(s).get("actions") or {})
def _lang_code(s: dict, word: str) -> str: return (_rules(s).get("languages") or {}).get(str(word).lower(), str(word).lower())
def _turn_role(s: dict, data: dict, typed: bool) -> str: return "owner" if typed or not s.get("identify_speakers") else (data.get("speaker_role") or "unknown")   # with identification off every voice counts as this login's owner
def _wake_words(s: dict) -> list: return [w.strip() for w in [*(_rules(s).get("wake") or []), *str(_opt(s, "wake_words") or "").split(",")] if w.strip()]
def _wake(sid: str, s: dict): _AWAKE[sid] = asyncio.get_event_loop().time() + float(_opt(s, "wake_window_s") or 0)
def _awake(sid: str) -> bool: return _AWAKE.get(sid, 0) > asyncio.get_event_loop().time()
def _routed(s: dict) -> bool: return bool(_opt(s, "route_turns")) and not s.get("pipeline_id")   # only the built-in flow carries the route node
def _status_html(text: str) -> str: return f"""<div id="voice-status" hx-swap-oob="innerHTML"><span class="status-label">{text}</span></div>"""

def _voice_for(s: dict) -> str:
    """The default voice: the reply language's voice when one is set and mapped, else the Voice setting."""
    return ((_opt(s, "lang_voices") or {}).get(_lang_code(s, s["reply_language"])) if s.get("reply_language") else "") or s.get("voice", "")

def _rules(s: dict) -> dict:
    """The shipped route rules with the conversation's saved ones over them: routes merge by name (a saved route with "off": true removes a shipped one), languages and actions merge by key, roles and wake words are replaced when saved."""
    saved = _opt(s, "route_rules") or {}
    routes = {r["name"]: r for r in _ROUTE_RULES["routes"]}
    routes.update({r["name"]: r for r in saved.get("routes") or []})
    return {**_ROUTE_RULES, **saved, "routes": [r for r in routes.values() if not r.get("off")], "languages": {**_ROUTE_RULES["languages"], **(saved.get("languages") or {})}, "actions": {**_ROUTE_RULES["actions"], **(saved.get("actions") or {})}}

def _tools_text(s: dict) -> str:
    """The tag instruction, placed last in the system prompt so no earlier style rule (no symbols in spoken replies) overrides it. Only languages that have a voice and actions that are on; empty when neither applies."""
    langs = ", ".join(sorted(_opt(s, "lang_voices") or {}))
    acts = " ".join(_ACTION_HELP[a] for a in _action_tags(s) if a in _ACTION_HELP) if _opt(s, "ai_actions") else ""
    body = str(_opt(s, "tools_prompt") or "").replace("{languages}", langs or "none (speak only the default language)").replace("{actions}", acts).strip()
    return f"""Tags are the one exception to the spoken-style rules above: write them exactly as shown. {body}""" if body and (langs or acts) else ""

async def _run_command(username: str, sid: str, s: dict, name: str, args: dict) -> str:
    """A routed command, run by this module; returns the status text (HTML). A rule naming a command this module has no handler for says so."""
    if name == "wake": _wake(sid, s); return "Listening."
    if name == "sleep": _AWAKE.pop(sid, None); return f"""Sleeping - start with {UI.escape((_wake_words(s) or ["the wake word"])[0])} to talk to me again."""
    if name == "stop": return "Stopped."
    if name == "remember": return _mem_note(_mem_put(username, "fact", args.get("text", ""), "command"))
    if name == "language":
        doc = _load(sid); doc["settings"]["reply_language"] = str(args.get("lang", "")).lower(); _save(doc)
        await _push(username, f"""<div id="voice-settings-fields" hx-swap-oob="innerHTML">{_FIELDS.render(doc["settings"], name_prefix="vs_")}</div>""")   # the open settings form autosaves, so it must show the new value or it would write the old one back
        mapped = (_opt(s, "lang_voices") or {}).get(_lang_code(s, doc["settings"]["reply_language"]))
        return f"""Replies in {UI.escape(doc["settings"]["reply_language"])} from the next turn{"" if mapped else " (no voice is mapped for it - set Voice per Language)"}."""
    if name == "sample":
        p = _profile_named(username, args.get("name", ""))
        _ENROLL[username] = p["id"]
        await _push(username, f"""<div id="voice-speakers" hx-swap-oob="innerHTML">{_speakers_html(username)}</div>""")
        return f"""The next recording is a voice sample for {UI.escape(p["name"])} - let them talk for 5-10 s."""
    return f"""No handler here for the command {UI.escape(name)} - check the route rules."""

async def _run_actions(username: str, s: dict, actions: list, role: str, audio: str) -> list:
    """Actions the model wrote as tags. Each runs only when Let the Model Act is on and the turn's speaker holds the role the rules give it; refused or unknown ones are reported, never dropped silently."""
    notes, need = [], _rules(s).get("actions") or {}
    for a in actions:
        if not _opt(s, "ai_actions"): notes.append(f"""Not done: {UI.escape(a["name"])} (model actions are off)."""); continue
        if _rank(s, role) < _rank(s, need.get(a["name"], "owner")): notes.append(f"""Not done: {UI.escape(a["name"])} needs the {UI.escape(need.get(a["name"], "owner"))} role (speaker: {UI.escape(role)})."""); continue
        if a["name"] == "remember": notes.append(_mem_note(_mem_put(username, "style" if a["attrs"].get("kind") == "style" else "fact", a["body"], "assistant", about=a["attrs"].get("about", ""), mid=a["attrs"].get("id", ""))) if a["body"].strip() else "Not done: an empty remember tag.")
        elif a["name"] == "note": notes.append(_note_act(username, a["attrs"], a["body"]))
        elif a["name"] == "sample": notes.append(await _sample_from_turn(username, s, a["attrs"].get("name") or a["body"], audio))
        else: notes.append(f"""No handler here for the action {UI.escape(a["name"])}.""")
    return notes

# --- Turn execution ---

def _system(s: dict) -> str: return "\n\n".join(p for p in (str(s.get("consent_prefix") or "").strip(), str(s.get("system_prompt") or "").strip(), f"""Reply in {s["reply_language"]} unless asked otherwise.""" if s.get("reply_language") else "") if p)   # prefix first and unchanged turn to turn, so it is the cached part of every prompt

def _flow(s: dict) -> dict:
    """The built-in turn as an ordinary inline flow - 'Save as pipeline' writes exactly this, so it can be edited in the pipeline builder and selected above.
    With routing on, transcribe writes 'heard' and the route node turns addressed lines into 'transcript'; commands and unaddressed lines never reach the prompt."""
    nodes = [{"id": "stt", "name": "Transcribe", "type": "transcribe", "config": {"conn_id": s.get("stt_conn", ""), "language": s.get("language", ""), "vad_filter": True}, "key_map": {"text": "transcript"}},
             {"id": "prompt", "name": "Prompt", "type": "transform", "config": {"mode": "template", "template": "{history}{transcript}"}, "key_map": {"input": "transcript", "value": "prompt"}, "extra_in_keys": ["history"]},
             {"id": "llm", "name": "Reply", "type": "generate", "config": {"conn_id": s.get("chat_conn", ""), "model": s.get("model", ""), "system_prompt": _system(s), "user_template": "{prompt}", "num_ctx": s.get("num_ctx", 16384), "think": ""}, "key_map": {"input": "prompt", "text": "reply"}}]
    if s.get("speak_replies"): nodes.append({"id": "tts", "name": "Speak", "type": "speak", "config": {"conn_id": s.get("tts_conn", ""), "voice": _voice_for(s), "length_scale": s.get("length_scale"), "lang_voices": _opt(s, "lang_voices") or {}, "action_tags": ",".join(_action_tags(s))}, "key_map": {"text": "reply", "audio_b64": "reply_audio"}})
    by = {n["id"]: n for n in nodes}
    if s.get("identify_speakers"):   # runs beside transcribe; the prompt waits for its tag. Typed turns get the tag as an input instead. {speaker_list} is filled per turn from the login's profiles.
        nodes.insert(1, {"id": "who", "name": "Who Spoke", "type": "identify_speaker", "config": {"conn_id": s.get("speaker_conn", ""), "match_threshold": _opt(s, "speaker_match"), "guess_threshold": _opt(s, "speaker_guess"), "tag_format": _opt(s, "speaker_tag_format"), "other_name": _opt(s, "speaker_other_name")}, "key_map": {}})
        by["prompt"].update(config={"mode": "template", "template": "{history}{speaker_tag}{transcript}"}, extra_in_keys=["history", "speaker_tag"])
        by["llm"]["config"]["system_prompt"] = f"""{by["llm"]["config"]["system_prompt"]}\n\n{{speaker_list}}""".strip()
    if _opt(s, "use_memory"): by["llm"]["config"]["system_prompt"] = f"""{by["llm"]["config"]["system_prompt"]}\n\n{{memory}}""".strip()   # after the stable parts, so a memory change only re-processes the tail of the system prompt
    if _opt(s, "route_turns"):
        by["stt"]["key_map"] = {"text": "heard"}
        nodes.insert(1, {"id": "route", "name": "Route", "type": "route_text", "config": {"rules_json": _rules(s), "wake_words": _opt(s, "wake_words") or "", "wake_fuzz": _opt(s, "wake_fuzz")}, "key_map": {"input": "heard", "role": "speaker_role", "text": "transcript"}, "extra_in_keys": ["awake", "speaker_role"]})
    flow = _tune_flow({"nodes": [{**n, "status": "idle"} for n in nodes]}, s)
    tools = _tools_text(s)
    if tools: llm = next(n for n in flow["nodes"] if n["id"] == "llm"); llm["config"]["system_prompt"] = f"""{llm["config"]["system_prompt"]}\n\n{tools}"""   # last, after the spoken-style instruction, so that rule cannot cancel the tags
    return flow

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
    s, sid, username, typed = doc["settings"], doc["id"], request.state.user.username, "audio_b64" not in inputs
    kind, extra = ("id", {"pipeline_id": s["pipeline_id"]}) if s.get("pipeline_id") else ("inline", {"inline_flow": _flow(s)})
    job_id, err = AIM.engine.submit(username, kind=kind, inputs={**inputs, "history": _history(doc), "chat_id": sid, "awake": "1" if typed or _awake(sid) else "", "memory": _mem_text(username) if _opt(s, "use_memory") else "", **_speaker_inputs(username, s, typed=typed)}, lane="" if barge_of else f"voice:{username}", **extra)
    if err: return imr.raw(CM.append_system_error_html(sid, err))
    _TURNS[sid], _ENDED[job_id] = job_id, asyncio.Event()
    imr.raw(CM.working_html(sid, {"type": "voice_stop", "job_id": job_id, "branch": IM.branch_id, "lvl": 2}))   # the call-mode barge-in presses this Stop button
    imr.raw(f"""<div id="cm-stream-{sid}" hx-swap-oob="innerHTML"><div class="cm-content voice-live" data-stream-job="{job_id}" data-stream-node="{UI.escape(s.get("stream_node") or "llm")}"></div></div>""")
    asyncio.create_task(_watch(username, sid, job_id, typed=typed, recording=inputs.get("audio_b64", "") if s.get("keep_recordings") else "", barge_of=barge_of, audio=inputs.get("audio_b64", "")))
    return imr

def _failed(job: dict) -> str: return next((f"""{n.get("name") or n["id"]}: {n.get("message", "")}""" for n in job["flow"]["nodes"] if n.get("status") == "error"), job["log"][-1] if job.get("log") else job["status"])

async def _push(username, html: str): await WS.send_personal_message(html, username)

def _streamed(job: dict) -> bool: return any(n.get("config", {}).get("speak_while_writing") for n in job["flow"]["nodes"] if n.get("type") == "generate")   # the reply was already played sentence by sentence

async def _watch(username: str, sid: str, job_id: str, typed: bool, recording: str = "", barge_of: str = "", audio: str = ""):
    """Follows one turn's job: the user's words appear once transcribed (with their recording, when kept); the reply appears with its audio once spoken - its text streams live meanwhile - or as text alone if the run ends without audio.
    A spoken turn tells the browser whether words were heard and addressed as soon as that is known (a paused reply resumes when not); a turn that talked over a running one (barge_of) stops it only when it was heard.
    A routed turn may end as a command (run after the user's words are shown) or unaddressed (nothing is stored or sent to the model). Known tags in the reply are kept out of the shown text; allowed actions run once the reply lands."""
    shown, job, mine, s = set(), None, lambda: _TURNS.get(sid) == job_id, _load(sid)["settings"]   # a newer turn owns the chat's working indicator and live text; an older one must not clear them
    while True:
        await asyncio.sleep(0.4)
        try: job = AIM.engine.load_job(job_id)
        except json.JSONDecodeError: continue   # the engine rewrites the job file between waves; a read can land mid-write
        data, done = job.get("data", {}), job["status"] in _DONE
        if not typed and "user" not in shown and ("transcript" in data or "command" in data or done):
            shown.add("user")
            heard = await _heard(username, sid, job)
            if barge_of and heard: AIM.engine.stop(barge_of, now=True); await _until_ended(barge_of)   # the reply it cut off lands in the chat first, so the conversation reads in order
            said = data.get("transcript") or (data.get("heard", "") if "command" in data else "")
            if heard and said:
                who = {k: data[k] for k in ("speaker", "speaker_role", "speaker_score", "speaker_tag") if data.get(k) not in (None, "")}
                await _push(username, CM.append_message_html(sid, _msg(_add_turn(sid, "user", said, job_id, recording, **who), username), viewer_name=username) + (_status_html(f"""Heard {UI.escape(who["speaker_tag"].strip().rstrip(":"))} - score {who.get("speaker_score", 0):.2f}""") if who.get("speaker_tag") else ""))
            elif job["status"] == "done": await _push(username, _status_html("Not addressed - start with a wake word." if data.get("route") == "ignored" else "Nothing heard - ignored."))
        if ("transcript" in data or "command" in data) and "awake" not in shown: shown.add("awake"); _wake(sid, s)   # an addressed line keeps the conversation open; a sleep command below closes it again
        if "command" in data and "command" not in shown: shown.add("command"); await _push(username, _status_html(await _run_command(username, sid, s, data["command"], data.get("args") or {})) + _memory_oob(username))
        if "note" in data and "note" not in shown: shown.add("note"); await _push(username, _status_html(UI.escape(data["note"])))
        if "reply" in data and "reply" not in shown and ("reply_audio" in data or done):
            shown.add("reply")
            parsed = BI.parse_tagged(data["reply"], _action_tags(s))
            await _push(username, CM.append_message_html(sid, _msg(_add_turn(sid, "assistant", parsed["plain"], job_id, data.get("reply_audio", "")), username, autoplay=not _streamed(job)), viewer_name=username) + (CM.stream_clear_html(sid) if mine() else ""))
            notes = await _run_actions(username, s, parsed["actions"], _turn_role(s, data, typed), audio)
            if notes: await _push(username, _status_html(" - ".join(notes)) + _memory_oob(username))
        if done: break
    if job["status"] == "done" and "reply" in job.get("data", {}): _wake(sid, s)   # the follow-up window counts from the end of the reply
    tail = CM.working_hide_html(sid) + CM.stream_clear_html(sid) if mine() else ""
    if mine(): _TURNS.pop(sid)
    _ENDED.pop(job_id).set()
    if job["status"] == "stopped": tail += _status_html("Turn stopped.")
    elif job["status"] == "done": await _push(username, _status_html(_unheard_note(s, data)))
    if tail: await _push(username, tail)

async def _until_ended(job_id: str, timeout_s: float = 10):
    """Waits for a turn's watcher to finish (it is gone already when not listed). Bounded, so a stuck engine never holds the newer turn back for good."""
    if job_id in _ENDED:
        try: await asyncio.wait_for(_ENDED[job_id].wait(), timeout_s)
        except asyncio.TimeoutError: pass

def _unheard_note(s: dict, data: dict) -> str:
    """Why a spoken turn did not reach the model, with what was heard - so a missed wake word or a filtered transcript can be seen and tuned."""
    said = UI.escape(str(data.get("dropped_text") or "").strip()[:120])
    if data.get("route") == "ignored": return f"""Not addressed - heard "{said}"; start with {UI.escape(" / ".join(_wake_words(s)) or "a wake word")} (or lower Wake Word Tolerance)."""
    return f"""Nothing heard - the recognizer's text "{said}" was filtered (empty, an ignored phrase, or rated not-speech).""" if said else "Nothing heard - the recognizer returned no words."    
    
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
                   <details class="status-details" {"open" if status else ""}><summary>Settings</summary><form {_post("settings_save", sid=sid)} hx-include="this" hx-trigger="change delay:300ms, submit" class="node-form"><div id="voice-settings-fields">{_FIELDS.render(doc["settings"], name_prefix="vs_")}</div><div class="form-actions"><button type="submit" class="ui-btn">Save</button><button type="button" class="ui-btn" {_post("save_flow", sid=sid)}>Save built-in flow as pipeline</button><span id="voice-saved"></span></div></form></details>
                   <details class="status-details"><summary>Speakers</summary><div id="voice-speakers">{_speakers_html(username)}</div></details>
                   <details class="status-details"><summary>Memory</summary><div id="voice-memory">{_memory_html(username)}</div></details>
                   <details class="status-details"><summary>Backup / Transfer</summary>{BUNDLE.panel_html({"sid": sid})}</details>
                   <div class="form-actions"><button class="ui-btn" {_post("prepare", sid=sid)} title="Loads the model and caches the system prompt so the first real turn starts fast">Prepare</button><button class="ui-btn" {_post("new")} title="Starts a fresh conversation with these settings; this one stays in the list">Restart conversation</button><span id="voice-status">{status}</span></div></div>
               <div class="voice-chat">{CM.shell(sid, messages=[_msg(t, username) for t in doc["turns"]], viewer_name=username, voice_opts=_call_cfg(doc["settings"]))}</div>"""

def _rebuild_btn(p: dict, armed) -> str: return f"""<button class="ui-btn" {_post("spk_rebuild", pid=p["id"])} {"disabled" if armed else ""} title="Measure this person's kept recordings again - after a speaker-model change, or for samples without a voiceprint">Rebuild</button>""" if any(isinstance(x, dict) for x in p["samples"]) else ""

def _speakers_html(username: str, note: str = "") -> str:
    armed = _ENROLL.get(username)
    roles = lambda p: "".join(f"""<option value="{k}" {"selected" if p.get("role") == k else ""}>{UI.escape(v)}</option>""" for k, v in SPEAKER_ROLES)
    rows = "".join(f"""<div class="voice-spk-row"><b>{UI.escape(p["name"])}</b><span class="status-label">({p.get("code", "")})</span><select name="role" {_post("spk_role", pid=p["id"])} hx-trigger="change" hx-include="this">{roles(p)}</select><span class="status-label">{_sample_note(p)}</span><button class="ui-btn" {_post("spk_sample", pid=p["id"])} {"disabled" if armed else ""}>Record sample</button>{_rebuild_btn(p, armed)}<button class="btn-icon" style="color:#ff5f5f" {_post("spk_delete", pid=p["id"])} hx-confirm="Delete this voice profile, its voiceprints and its recordings?">&#x2715;</button></div>""" for p in _spk(username)["profiles"])
    wait = f"""<div class="voice-spk-note">Next recording is a voice sample for <b>{UI.escape((_spk_get(_spk(username), armed) or {}).get("name", "?"))}</b>: hold the mic and speak naturally for 5-10 s. <button class="ui-btn" {_post("spk_cancel")}>Cancel</button></div>""" if armed else ""
    form = f"""<form class="voice-spk-row" {_post("spk_add")} hx-include="this"><input name="name" placeholder="Name" required><select name="role">{roles({"role": "trusted"})}</select><button type="submit" class="ui-btn" {"disabled" if armed else ""}>Add and record a sample</button></form>"""
    return f"""{rows or '<div class="list-empty">No voices enrolled.</div>'}{wait}{form}<div class="status-label">Voiceprints and the recordings they come from are kept on this portal, under this login only; deleting a profile deletes both. 3+ samples per person, in the rooms you use, work best.{" " + note if note else ""}</div>"""

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
    """Saves on every form change and on Save, and makes these settings this login's defaults for new conversations. Values equal to the shipped defaults are not stored, so improved defaults reach existing conversations.
    Re-renders the conversation only when call-detection values changed - they live in the footer, and an open call ends with that re-render (start it again to use them)."""
    doc = _owned(request, payload.get("sid", ""))
    if not doc: return imr
    try: new = _FIELDS.parse(payload, name_prefix="vs_")
    except BI.SettingsError as e: return imr.oob(f"""<span class="err-box">Not saved - {UI.escape(str(e))}</span>""", "voice-saved")
    old, doc["settings"] = _call_cfg(doc["settings"]), {k: v for k, v in new.items() if not (k in _NEW_DEFAULTS and v == _NEW_DEFAULTS[k])}
    _save(doc); _save_defaults(doc["username"], doc["settings"])
    if _call_cfg(doc["settings"]) != old: return _refresh(imr, request.state.user.username, doc, '<span class="status-label">&#x2713; Saved</span>')
    return imr.oob('<span class="status-label">&#x2713; Saved</span>', "voice-saved")

async def _h_save_flow(request, payload, imr):
    doc = _owned(request, payload.get("sid", ""))
    if not doc: return imr
    pl = {**AIM.engine.new_pipeline(owner="voice", name=f"""Voice turn ({datetime.utcnow().strftime("%Y-%m-%d %H:%M")})"""), "flow": _flow({**doc["settings"], "speak_replies": True}), "tags": ["voice"]}
    AIM.engine.save_pipeline(pl)
    return imr.oob(f"""<span class="status-label">&#x2713; Saved as pipeline {pl["id"]} - edit it in a pipeline builder, then pick it as this conversation's Pipeline.</span>""", "voice-saved")

async def _h_settings_refresh(request, payload, imr):
    try: return imr.oob(_FIELDS.render(_FIELDS.parse(payload, name_prefix="vs_"), name_prefix="vs_"), "voice-settings-fields")   # OOB: the select inherits the settings form's hx-swap="none"; unsaved edits ride along in the payload
    except BI.SettingsError: return imr   # the autosave of the same change reports the bad field

async def _h_prepare(request, payload, imr):
    """One short generation with the real system prompt, model and context size: loads the model and fills its prompt cache, so the first spoken turn only processes the new words. The reply is not added to the conversation."""
    doc = _owned(request, payload.get("sid", ""))
    if not doc: return imr
    llm = next(n for n in _flow(doc["settings"])["nodes"] if n["id"] == "llm")   # the tuned node, so the cached system prompt includes the spoken-style instruction exactly as real turns send it
    flow = {"nodes": [{**llm, "config": {**{k: v for k, v in llm["config"].items() if not k.startswith("speak_")}, "user_template": "Reply with the single word: ready", "num_predict": 16}, "key_map": {"input": "go", "text": "ready"}}]}
    username = request.state.user.username
    job_id, err = AIM.engine.submit(username, kind="inline", inline_flow=flow, inputs={"go": "1", "memory": _mem_text(username) if _opt(doc["settings"], "use_memory") else "", **_speaker_inputs(username, doc["settings"], typed=True)}, lane=f"voice:{username}")
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
    return await _start_turn(request, doc, {"heard" if _routed(doc["settings"]) else "transcript": text}, imr)   # doc as loaded before this turn was added, so history does not repeat it; routed turns enter through the route node

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

async def _h_spk_add(request, payload, imr):
    username, name = request.state.user.username, str(payload.get("name", "")).strip()
    if not name: return imr
    doc = _spk(username); p = _new_profile(doc, name, payload.get("role", "guest")); _spk_save(username, doc); _ENROLL[username] = p["id"]
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
    username = request.state.user.username; doc = _spk(username); p = _spk_get(doc, payload.get("pid", ""))
    if p:
        for x in p["samples"]:
            if isinstance(x, dict): _audio_path(username, x["audio"]).unlink(missing_ok=True)
        doc["profiles"].remove(p); _spk_save(username, doc)
    if _ENROLL.get(username) == payload.get("pid"): _ENROLL.pop(username)
    return imr.oob(_speakers_html(username), "voice-speakers")

async def _h_spk_rebuild(request, payload, imr):
    """Measures every kept recording of a profile again with the active conversation's speaker connection and replaces those voiceprints - after a speaker-model change, or for samples whose measurement failed.
    A recording that fails keeps its previous voiceprint; bare voiceprints (no recording) are left as they are."""
    username, pid = request.state.user.username, payload.get("pid", "")
    s, p = (await _active(request))["settings"], _spk_get(_spk(username), pid)
    if not p: return imr.oob(_speakers_html(username), "voice-speakers")
    new, failed = {}, []
    for x in (x for x in p["samples"] if isinstance(x, dict)):
        r = await _embed(s, base64.b64encode(_audio_path(username, x["audio"]).read_bytes()).decode())
        if r.get("error"): failed.append(r["error"])
        else: new[x["audio"]] = r["embedding"]
    doc = _spk(username); p = _spk_get(doc, pid)   # re-read: samples recorded during the measurements stay
    if not p: return imr.oob(_speakers_html(username), "voice-speakers")
    for x in p["samples"]:
        if isinstance(x, dict) and x["audio"] in new: x["embedding"] = new[x["audio"]]
    p["embedding"] = _voiceprint(p)
    _spk_save(username, doc)
    note = f"""&#x2713; {len(new)} voiceprint(s) rebuilt for {UI.escape(p["name"])}.""" + (f""" <span class="err-box">{len(failed)} failed: {UI.escape(failed[0])}</span>""" if failed else "")
    return imr.oob(_speakers_html(username, note), "voice-speakers")

async def _h_mem_add(request, payload, imr):
    username = request.state.user.username
    if str(payload.get("text", "")).strip(): _mem_put(username, payload.get("kind") if payload.get("kind") in ("fact", "style") else "fact", payload["text"], "you", about=str(payload.get("about", "")).strip())
    return imr.oob(_memory_html(username), "voice-memory")

async def _h_mem_edit(request, payload, imr):
    """Edits a fact or style line in place. The assistant's notes change only through a request or Remove anyway."""
    username = request.state.user.username
    item = next((i for i in _mem(username)["items"] if i["id"] == payload.get("mid") and i["kind"] != "note"), None)
    if item and str(payload.get("text", "")).strip(): _mem_put(username, item["kind"], payload["text"], "you", mid=item["id"])
    return imr.oob(_memory_html(username), "voice-memory")

async def _h_mem_delete(request, payload, imr):
    username = request.state.user.username; d = _mem(username)
    d["items"] = [i for i in d["items"] if not (i["id"] == payload.get("mid") and i["kind"] != "note")]
    _mem_save(username, d)
    return imr.oob(_memory_html(username), "voice-memory").oob('<span class="status-label">Removed from memory.</span>', "voice-status")

async def _h_note_ask(request, payload, imr):
    """Asks the assistant to change or remove one of its notes; the request goes into its next turn and it answers with a note tag."""
    username = request.state.user.username; d = _mem(username)
    if any(i["id"] == payload.get("nid") and i["kind"] == "note" for i in d["items"]):
        d["requests"].append({"id": f"r_{uuid.uuid4().hex[:6]}", "note_id": payload["nid"], "reason": " ".join(str(payload.get("reason", "")).split()), "status": "open", "ts": _now_iso()})
        _mem_save(username, d)
    return imr.oob(_memory_html(username), "voice-memory")

async def _h_note_force(request, payload, imr):
    """Removes one of the assistant's notes without its answer - the login has the final say over what is stored. The assistant gets a note of its own saying a note was removed (not its content), which it may keep or remove."""
    username = request.state.user.username; d = _mem(username)
    note = next((i for i in d["items"] if i["id"] == payload.get("nid") and i["kind"] == "note"), None)
    if note:
        d["items"].remove(note)
        for r in d["requests"]:
            if r["note_id"] == note["id"] and r["status"] == "open": r.update(status="overridden", answered=_now_iso())
        _mem_save(username, d)
        _mem_put(username, "note", f"""The owner removed your note [{note["id"]}] on {_now_iso()[:10]} without waiting for your answer.""", "owner")
    return imr.oob(_memory_html(username), "voice-memory")

# --- Init / routes ---

def init_tool(env: dict, prefix: str):
    global ENV, PREFIX, UI, WS, IM, BI, AIM, CM, _FIELDS, BUNDLE
    ENV, PREFIX = env, prefix
    UI = env["templates"].env.globals.get("UI")
    WS = env["ws"]
    BI = env["tools"]["built_ins"]
    AIM = env["tools"]["ai_manager"]
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    AUDIO_DIR.mkdir(parents=True, exist_ok=True)
    SPEAKER_DIR.mkdir(parents=True, exist_ok=True)
    DEFAULTS_DIR.mkdir(parents=True, exist_ok=True)
    MEMORY_DIR.mkdir(parents=True, exist_ok=True)
    _FIELDS = _build_fields()
    IM = env["InterfaceManager"](nesting_level=2, db_path="voice_im.db")
    BUNDLE = BI.DataBundle("voice", IM, {"settings": ("Settings (this conversation and your defaults)", _settings_out, _settings_in), "speakers": ("Voices (profiles, voiceprints, recordings)", _speakers_out, _speakers_in), "memory": ("Memory", lambda u, ctx: _mem(u), _memory_in)}, BACKUP_DIR, nesting_level=2, on_import=_after_import)
    CM = BI.ChatManager(namespace="voice", base_url=prefix, view_style="bubble", stream_toggle=False, think_toggle=False, stop_enabled=True, pin_enabled=True, allow_edit=False, allow_delete=False, allow_copy=True, show_info=False, markdown_mode="standard", placeholder="Type, hold \U0001F3A4 to talk, or \u260E for a hands-free call (Ctrl+Enter sends)", branch_id=IM.branch_id, nesting_level=2, intent_name="voice_text", voice=True, audio_intent="voice_audio")
    IM.scripts.update({f"voice_{k}": [fn] for k, fn in {"new": _h_new, "open": _h_open, "delete": _h_delete, "settings_save": _h_settings_save, "save_flow": _h_save_flow, "text": _h_text, "audio": _h_audio, "stop": _h_stop, "settings_refresh": _h_settings_refresh, "prepare": _h_prepare, "spk_add": _h_spk_add, "spk_sample": _h_spk_sample, "spk_cancel": _h_spk_cancel, "spk_role": _h_spk_role, "spk_delete": _h_spk_delete, "spk_rebuild": _h_spk_rebuild, "mem_add": _h_mem_add, "mem_delete": _h_mem_delete, "mem_edit": _h_mem_edit, "note_ask": _h_note_ask, "note_force": _h_note_force}.items()})
    print("[voice] ready")

@router.get("")
@router.get("/")
async def root(request: Request):
    doc = await _active(request)
    left = UI.toolbar(side="left", content=_side_html(request.state.user.username, doc), size="16rem", overlay=False, start_open=True, resizable=True, id="voice-left-bar", nesting_level=2)
    return ENV["templates"].TemplateResponse(name="base.html", request=request, context={"request": request, "user": request.state.user, "nesting_level": 2, "shell_id": IM.branch_id, "toolbars": {"left": left}, "content": _page_html(request.state.user.username, doc), "extra_css": CSS + CM.CSS + BI.DATA_BUNDLE_CSS, "extra_script": CM.SCRIPT + BI.DATA_BUNDLE_JS + SCRIPT})

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
.voice-mem-sec{margin:.5rem 0;}
"""

SCRIPT = """
(function(){ var b = document.getElementById('voice-left-bar'); if (b && typeof toggleToolbar === 'function' && b.classList.contains('active') && window.matchMedia('(max-width:46rem)').matches) toggleToolbar('voice-left-bar', 'left', '16rem', 'shrink', '&#x25C0;', '&#x25B6;'); })();   // narrow screens start with the conversation list collapsed; its arrow opens it
"""
