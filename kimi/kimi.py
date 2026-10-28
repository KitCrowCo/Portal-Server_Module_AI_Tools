# /modules/ai_tools/kimi/kimi.py
"""
Kimi - Knowledge Integration Manager Interface: the UI over knowledge groups (LightRAG servers). Sub-module of ai_tools, mounted at /module/ai_tools/kimi.
Sources: the shared _knowledge dir (also used by Tessa/Athena pipelines) and the server-wide _common dir.
Tabs: Query (modes and groups side by side, per-group default options, or retrieval only with any chat model writing the answer), Paste Text, Documents (paged list with status counts, the server's indexing pipeline, delete, reprocess failed), Graph (interactive, around a label), Server (the routes the server offers vs the ones its profile uses).
Ingest: a file goes in under its folder path (notes/a.md -> notes__a.md) and replaces what the group holds for an earlier version of it; progress is saved per file, so a pass cut off by a restart continues.
All protocol logic lives in AIM.connections.lightrag_* and the connection profile (tools/ai_manager/_connections/lightrag.json) - this module is the UI.
"""

import re, json, time, asyncio, hashlib, fnmatch, docx
from datetime import datetime
from pathlib import Path
from fastapi import APIRouter, Request, HTTPException
from fastapi.responses import HTMLResponse, FileResponse

TOOL_META = {"label": "Kimi", "group": "knowledge", "icon": "&#x1F4DA;", "description": "Knowledge Integration Manager Interface", "singleton": True}

router = APIRouter(redirect_slashes=False)

ENV: dict = {}
_P = "/module/ai_tools/kimi"
SYNC_STATE_FILE = Path("./data/ai_tools/kimi_sync_state.json")
KG_DIR = Path("./data/ai_tools/_knowledge")
COMMON_DIR = Path("./data/_common")
SOURCESETS_PATH = Path("./data/ai_tools/kimi_sourcesets.json")
SKIP_DEFAULT = "static/fonts/, static/js/codemirror/, static/js/katex/, static/information/, *.min.js, *.min.css, *.map, LICENSE, LICENSE.*, LICENCE, LICENCE.*, COPYING, COPYING.*, AUTHORS, AUTHORS.*, NOTICE, NOTICE.*, OFL.txt, *.ttf, *.otf, *.woff, *.woff2, *.eot, *.ico, *.png, *.jpg, *.jpeg, *.gif, *.webp, *.svg, *.zip, *.gz, *.tar, *.db, *.sqlite"   # Never send, until changed in Settings: vendored fonts and editor/math libraries, the portal's help pages, minified bundles, license texts, files the knowledge server cannot read
_MODES = ("hybrid", "local", "global", "naive", "mix")
_TABS = {"query": {"id": "query", "order": 0, "label": "Query", "icon": "&#x1F50D;"}, "paste": {"id": "paste", "order": 1, "label": "Paste Text", "icon": "&#x1F4DD;"}, "docs": {"id": "docs", "order": 2, "label": "Documents", "icon": "&#x1F4C4;"}, "graph": {"id": "graph", "order": 3, "label": "Graph", "icon": "&#x1F578;"}, "server": {"id": "server", "order": 4, "label": "Server", "icon": "&#x2699;"}, "settings": {"id": "settings", "order": 5, "label": "Settings", "icon": "&#x1F527;"}}
_ANSWER_SYS = "Answer the question from the knowledge context below. When the context does not hold the answer, say so instead of guessing. Name the source files in [brackets] when the context gives them."

UI = FM_KG = FM_COMMON = BI = IM = TM = AIM = cfg = ANSWER = None
_SYNC_TASK = None
_INGEST_LOCK = asyncio.Lock()   # one ingest pass at a time - a knowledge server indexes one batch at a time anyway; later passes wait their turn
_TASKS = set()                  # background passes, held because the event loop keeps only weak references to tasks
_DELETING = {}                  # knowledge group id -> {document id: time the server accepted its deletion, 0 while waiting for a busy server}

def _u(*p): return "/" + "/".join(s.strip("/") for s in [_P.strip("/"), *p] if s)
def _esc(s): return str(s).replace("&","&amp;").replace("<","&lt;").replace(">","&gt;").replace('"',"&quot;")
def _vals(type_: str, **extra) -> str: return json.dumps({"type": type_, "branch": "kimi", "lvl": 2, **extra})
def _post(type_: str, **extra) -> str: return f"""hx-post="/im/in" hx-target="body" hx-swap="none" hx-vals='{_vals(type_, **extra)}'"""
def _now() -> str: return datetime.now().isoformat(timespec="seconds")
def _div(text) -> str: return f"""<div>{_esc(text)}</div>"""

# --- Init ---

def init_tool(env: dict, prefix: str):
    global ENV, UI, FM_KG, FM_COMMON, BI, cfg, IM, TM, AIM, _P, ANSWER
    ENV = env; _P = prefix.rstrip("/")
    UI = env["templates"].env.globals.get("UI")
    BI = env["tools"]["built_ins"]
    IM = env["InterfaceManager"](nesting_level=2, db_path="ai_tools/kimi_im.db")
    AIM = ENV["tools"]["ai_manager"]
    cfg = BI.SettingsPanel("Kimi", [BI.SettingsGroup("general", "General", [
        BI.SettingField("title", "Title", "text", "Kimi"),
        BI.SettingField("preserve_structure", "Keep the folder path in document names", "checkbox", True, hint="notes/a.md goes in as notes__a.md - same-named files in different folders stay apart. Off = the file name alone."),
        BI.SettingField("name_strip", "Leave out of document names (folders, comma-sep)", "text", "", hint="e.g. code - a file at code/core_files/main.py is named core_files__main.py, as it was before the sources moved under code/; its sync entry moves with it, so unchanged files are not sent again."),
        BI.SettingField("code_chunk_ext", "Code-aware chunks for (file endings, comma-sep)", "text", ".py,.js,.ts,.css,.html,.sh,.json,.yml,.yaml,.toml", hint="Files with these endings go in as text cut at whole definitions (functions, classes, section headers; Python by its syntax tree), each chunk headed with its file, section, the names it defines and its lines - the pipelines' Chunk code split. Other files go to the server as files and are cut by the server. Blank = every file as a file."),
        BI.SettingField("code_chunk_chars", "Code chunk size (characters)", "number", 4000, hint="Whole definitions are packed up to this size; one definition over twice this is split inside itself. Keep it at the size the group's other code was chunked with."),
        BI.SettingField("skip_patterns", "Never send (patterns, comma-sep)", "text", SKIP_DEFAULT, hint="Files matching any of these never go into a group, even inside a ticked folder, and ticking a folder leaves them unticked. folder/ = that folder anywhere in the path (static/fonts/); a pattern without / = file names (*.min.js, LICENSE.*); a/b* = a path from any folder down. Not case-sensitive. Documents a group already holds from such files stay until you remove them (Documents tab)."),
        BI.SettingField("sync_enabled", "Enable scheduled sync", "checkbox", False),
        BI.SettingField("sync_window", "Sync window (24h, HH:MM-HH:MM)", "text", "02:00-08:00", hint="Watched folders are re-ingested for changed files only while the current time falls in this window."),
        BI.SettingField("sync_conn_id", "Sync Knowledge Group", "select", "", options=lambda values=None: [("", "(none - scheduled sync does nothing)")] + [(c["_id"], c.get("display_name", c["_id"])) for c in AIM.connections.list_conns(conn_type="lightrag")]),
        BI.SettingField("sync_sourceset", "Sync Source Set (name)", "text", "", hint="A source set saved in Kimi's left panel: the files and folders the scheduled sync watches. Blank = scheduled sync does nothing."),
        BI.SettingField("sync_git_pull", "Pull git checkouts before syncing", "checkbox", False, hint="Sync, Continue and the scheduled sync first run git pull --ff-only in every git checkout at or inside the checked folders, then send only files whose content changed. A checkout with local changes or a diverged branch is left as it is and listed as failed."),
    ], json_path="data/ai_tools/kimi_settings.json")])
    ANSWER = BI.SettingsGroup("kimi_answer", "Answer with", AIM.steps.chat_fields("kimi_answer_refresh"), json_path="data/ai_tools/kimi_answer.json")   # rendered per person from state; json_path unused
    FM_KG = BI.FileManager(KG_DIR)
    FM_COMMON = BI.FileManager(COMMON_DIR)
    TM = BI.TabManager(namespace="kimi", tab_bar_id="kimi-tab-bar", content_id="kimi-panel", render_content_fn=_render_panel, intent_prefix="kimi", IM=IM, scope="user", nesting_level=2, allow_new=False, closable=False, empty={"tabs": dict(_TABS), "active": "query"})
    IM.scripts.update({f"kimi_{k}": [fn] for k, fn in {"move_modal": _h_move_modal, "move": _h_move, "conn_select": _h_conn_select, "health": _h_health, "select_file": _h_select_file, "ingest_selected": _h_ingest_selected, "sync_now": _h_sync_now, "ingest_status": _h_ingest_status, "ingest_continue": _h_ingest_continue, "insert_text": _h_insert_text, "query": _h_query, "save_defaults": _h_save_defaults, "answer_refresh": _h_answer_refresh, "clear_all": _h_clear_all, "doc_delete": _h_doc_delete, "docs_view": _h_docs_view, "pipeline": _h_pipeline, "reprocess": _h_reprocess, "cancel_pipeline": _h_cancel_pipeline, "graph_show": _h_graph_show, "upload_modal": _h_upload_modal, "upload_kg": lambda r, p, i: _h_upload(r, p, i, "kg"), "upload_common": lambda r, p, i: _h_upload(r, p, i, "common"), "export_docx": _h_export_docx, "chunk_selected": _h_chunk_selected, "combine_selected": _h_combine_selected, "sourceset_save": _h_sourceset_save, "sourceset_load": _h_sourceset_load, "sourceset_delete": _h_sourceset_delete, "settings_save": _h_settings_save, "uncheck": _h_uncheck, "remove_unsent": _h_remove_unsent}.items()})
    env["register_service"]("kimi", {"sync": sync_service})
    print("[kimi] ready")

def _ensure_sync_task():
    global _SYNC_TASK
    if _SYNC_TASK is None or _SYNC_TASK.done(): _SYNC_TASK = asyncio.create_task(_scheduled_sync_loop())

def _spawn(coro):
    t = asyncio.create_task(coro)
    _TASKS.add(t)
    t.add_done_callback(_TASKS.discard)

# --- State ---

async def _kg_state(request, state=None):
    if state is not None:
        await ENV["set_state"](request, state, scope="user", namespace="knowledge")
        return state
    s = await ENV["get_state"](request, scope="user", namespace="knowledge") or {}
    if not s.get("conn_id"):
        conns = AIM.connections.list_conns(conn_type="lightrag")
        s["conn_id"] = conns[0]["_id"] if conns else ""
    s.setdefault("selected_kg", []); s.setdefault("selected_common", [])
    s.setdefault("last_query", ""); s.setdefault("last_modes", [s.get("last_mode") or "hybrid"]); s.setdefault("last_result", "")
    return s

def _conn_of(s: dict): return AIM.connections.get_conn(s["conn_id"], conn_type="lightrag") if s.get("conn_id") else None
async def _active_conn(request): return _conn_of(await _kg_state(request))

def _chunk_text(text: str, target_chars: int = 4000) -> list:
    """Splits on paragraph boundaries, packing consecutive paragraphs up to target_chars per chunk - never splits mid-paragraph."""
    chunks, cur = [], ""
    for p in text.split("\n\n"):
        if cur and len(cur) + len(p) + 2 > target_chars: chunks.append(cur); cur = p
        else: cur = f"{cur}\n\n{p}" if cur else p
    if cur: chunks.append(cur)
    return chunks

def _load_sourcesets() -> dict: return json.loads(SOURCESETS_PATH.read_text()) if SOURCESETS_PATH.exists() else {}
def _save_sourcesets(d: dict): SOURCESETS_PATH.parent.mkdir(parents=True, exist_ok=True); SOURCESETS_PATH.write_text(json.dumps(d, indent=2))

def _sourceset_select_html() -> str:
    sets = _load_sourcesets()
    opts = "".join(f'<option value="{_esc(n)}">{_esc(n)} ({len(v.get("kg",[]))+len(v.get("common",[]))} files)</option>' for n,v in sets.items())
    return f"""<div id="kg-sourceset-select" style="display:flex;gap:.2rem;align-items:center">
                   <select id="kg-sourceset-picker" class="module-select" style="flex:1;font-size:.7rem;margin:0"><option value="">-- source sets --</option>{opts}</select>
                   <button type="button" class="cm-qbtn" onclick="htmx.ajax('POST','/im/in',{{values:{{type:'kimi_sourceset_load',branch:'kimi',lvl:2,name:document.getElementById('kg-sourceset-picker').value}},swap:'none'}})">Load</button>
                   <button type="button" class="cm-qbtn" style="color:#ff5f5f" onclick="if(confirm('Delete this source set?'))htmx.ajax('POST','/im/in',{{values:{{type:'kimi_sourceset_delete',branch:'kimi',lvl:2,name:document.getElementById('kg-sourceset-picker').value}},swap:'none'}})">&#x2715;</button>
               </div>"""

async def _h_sourceset_save(request, payload, imr):
    name = (payload.get("name") or "").strip()
    if not name: return imr
    s = await _kg_state(request)
    sets = _load_sourcesets()
    sets[name] = {"kg": s.get("selected_kg", []), "common": s.get("selected_common", [])}
    _save_sourcesets(sets)
    return imr.oob(_sourceset_select_html(), "kg-sourceset-select", swap="outerHTML")

async def _h_sourceset_load(request, payload, imr):
    entry = _load_sourcesets().get(payload.get("name",""))
    if not entry: return imr
    s = await _kg_state(request)
    s["selected_kg"], s["selected_common"] = entry.get("kg",[]), entry.get("common",[])
    await _kg_state(request, s)
    return _trees_oob(imr, s)

async def _h_sourceset_delete(request, payload, imr):
    sets = _load_sourcesets()
    sets.pop(payload.get("name",""), None)
    _save_sourcesets(sets)
    return imr.oob(_sourceset_select_html(), "kg-sourceset-select", swap="outerHTML")

async def _h_chunk_selected(request, payload, imr):
    s = await _kg_state(request)
    target = int(payload.get("chunk_size", 4000) or 4000)
    log = []
    for src, fm in (("kg", FM_KG), ("common", FM_COMMON)):
        for rel in list(s.get(f"selected_{src}", [])):
            try:
                p = fm.resolve(rel)
                if not p.is_file(): continue
                chunks = _chunk_text(p.read_text(encoding="utf-8", errors="ignore"), target)
                if len(chunks) < 2: log.append(f"{rel}: already fits in one chunk, skipped"); continue
                stem, parent = p.stem, str(Path(rel).parent).lstrip("./")
                for i, chunk in enumerate(chunks, 1):
                    fm.write(f"{parent}/{stem}_{i:03d}.md" if parent else f"{stem}_{i:03d}.md", chunk)
                log.append(f"{rel}: split into {len(chunks)} chunks")
            except Exception as e: log.append(f"{rel}: error {e}")
    imr.oob("".join(f'<div>{_esc(l)}</div>' for l in log) or '<div style="color:var(--text_muted)">Nothing selected.</div>', "kg-ingest-log")
    return _trees_oob(imr, s)

async def _h_combine_selected(request, payload, imr):
    s = await _kg_state(request)
    out_name = (payload.get("combine_name") or "combined").strip()
    parts = []
    for src, fm in (("kg", FM_KG), ("common", FM_COMMON)):
        for rel in sorted(s.get(f"selected_{src}", [])):
            try: parts.append(f"# {rel}\n\n{fm.resolve(rel).read_text(encoding='utf-8', errors='ignore')}")
            except Exception as e: parts.append(f"# {rel}\n\n[read error: {e}]")
    if not parts: return imr.oob('<div style="color:var(--text_muted)">Select files from Knowledge or Common to combine.</div>', "kg-ingest-log")
    dest = out_name if out_name.endswith(".md") else f"{out_name}.md"
    FM_KG.write(dest, "\n\n---\n\n".join(parts))
    imr.oob(f'<div style="color:var(--accent)">Combined {len(parts)} file(s) into {_esc(dest)}</div>', "kg-ingest-log")
    imr.oob(_source_tree_html(FM_KG, s.get("selected_kg", []), "kg"), "kg-tree-kg", swap="outerHTML")
    return imr

async def _h_export_docx(request, payload, imr):
    s = await _kg_state(request)
    out_name = (payload.get("docx_name") or "export").strip()
    parts = []
    for src, fm in (("kg", FM_KG), ("common", FM_COMMON)):
        for rel in s.get(f"selected_{src}", []):
            try: parts.append(f"# {rel}\n\n{fm.resolve(rel).read_text(encoding='utf-8', errors='ignore')}")
            except Exception as e: parts.append(f"# {rel}\n\n[read error: {e}]")
    if not parts: return imr.oob('<div style="color:var(--text_muted)">Select files to export.</div>', "kg-ingest-log")
    dest_name = out_name if out_name.endswith(".docx") else f"{out_name}.docx"
    _md_to_docx("\n\n".join(parts), COMMON_DIR / dest_name)
    imr.oob(f'<div style="color:var(--accent)">Exported to Common/{_esc(dest_name)} - open from the Wiki file browser to download.</div>', "kg-ingest-log")
    imr.oob(_source_tree_html(FM_COMMON, s.get("selected_common", []), "common"), "kg-tree-common", swap="outerHTML")
    return imr

# --- Ingest (Ingest Selected, Sync Now, uploads and the scheduled sync all run the same pass) ---

def _load_sync_state() -> dict: return json.loads(SYNC_STATE_FILE.read_text()) if SYNC_STATE_FILE.exists() else {}
def _save_sync_state(s: dict): SYNC_STATE_FILE.parent.mkdir(parents=True, exist_ok=True); SYNC_STATE_FILE.write_text(json.dumps(s, indent=2))
def _hash(b: bytes) -> str: return hashlib.sha256(b).hexdigest()[:20]
def _strips() -> list: return [x.strip().strip("/") + "/" for x in str(cfg.get_group("general").load().get("name_strip", "")).split(",") if x.strip().strip("/")]
def _named(rel: str) -> str: return next((rel[len(x):] for x in _strips() if rel.startswith(x)), rel)   # the path a document is named after: a folder that only groups sources (code/) is left out, so moving sources under it keeps their document names
def _upload_name(rel: str) -> str: return AIM.connections.lightrag_upload_name(_named(rel)) if cfg.get_group("general").load().get("preserve_structure", True) else Path(rel).name
def _unchanged(prev, f: Path, data: bytes) -> bool: return prev == _hash(data) or (isinstance(prev, (int, float)) and f.stat().st_mtime <= prev) # Same content as when the file last went into this group. Entries saved before content hashes were kept hold the file's modification time instead.

def _code_chunked(rel: str) -> bool: return Path(rel).suffix.lower() in {x.strip().lower() for x in str(cfg.get_group("general").load().get("code_chunk_ext") or "").split(",") if x.strip()}
def _rel(fm, f: Path) -> str: return str(f.relative_to(fm.root)).replace("\\", "/")
def _skip_rules() -> list: return [x.strip().lower().lstrip("/") for x in str(cfg.get_group("general").load().get("skip_patterns") or "").split(",") if x.strip().lstrip("/")]

def _sendable(rel: str, rules: list) -> bool:
    """A file a pass may send: nothing hidden (.git, .env), nothing in __pycache__, nothing matching a Never send rule (_skip_rules): folder/ anywhere in the path, a name pattern on the file name, a/b* on the path from any folder down."""
    parts = rel.lower().split("/")
    folders = "/" + "/".join(parts[:-1]) + "/"
    return not any(x.startswith(".") or x == "__pycache__" for x in parts) and not any(("/" + r in folders) if r.endswith("/") else fnmatch.fnmatchcase(parts[-1], r) if "/" not in r else any(fnmatch.fnmatchcase("/".join(parts[i:]), r) for i in range(len(parts))) for r in rules)

async def _send(conn: dict, rel: str, data: bytes, index: dict) -> dict:
    """One source file into the group, replacing what the group holds under its name. Code files (code_chunk_ext) go in as text cut at whole definitions (AIM.steps.chunk_code); other files, and code that is not UTF-8, go as files the server cuts itself. An empty code file only removes what was stored."""
    name = _upload_name(rel)
    try: text = data.decode("utf-8") if _code_chunked(rel) else None
    except UnicodeDecodeError: text = None
    if text is None: return await AIM.connections.lightrag_replace_file(conn, name, data, index=index)
    pieces = AIM.steps.chunk_code(text, _named(rel), int(cfg.get_group("general").load().get("code_chunk_chars") or 4000))
    if not pieces: return await AIM.connections.lightrag_delete_file(conn, name, index=index)
    r = await AIM.connections.lightrag_insert_named(conn, name, pieces=pieces, on_exists="replace", index=index)
    return r if r.get("error") else {**(r.get("response") or {}), "replaced": r["deleted"]}

def _files(sources: dict) -> list:
    """(prefix, file manager, file) for every checked file and every file inside a checked folder that a pass may send (_sendable). Runs in a worker thread - a large folder must not stall the server."""
    out, rules = [], _skip_rules()
    for prefix, fm in (("kg", FM_KG), ("common", FM_COMMON)):
        for rel in sources.get(prefix, []):
            p = fm.resolve(rel)
            out += [(prefix, fm, f) for f in ([p] if p.is_file() else sorted(p.rglob("*")) if p.is_dir() else []) if f.is_file() and _sendable(_rel(fm, f), rules)]
    return out

def _git_repos(sources: dict) -> list:
    """Every git checkout at or inside the checked folders."""
    roots = [fm.resolve(rel) for prefix, fm in (("kg", FM_KG), ("common", FM_COMMON)) for rel in sources.get(prefix, [])]
    return sorted({g.parent for p in roots if p.is_dir() for g in p.rglob(".git")})

async def _git_pull(sources: dict) -> list:
    """git pull --ff-only in each checkout; one line per checkout that failed. Fast-forward only: a checkout with local changes or a diverged branch is not touched."""
    errs = []
    for repo in await asyncio.to_thread(_git_repos, sources):
        proc = await asyncio.create_subprocess_exec("git", "-c", "safe.directory=*", "-C", str(repo), "pull", "--ff-only", "-q", stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.STDOUT)   # safe.directory: checkouts made on the host belong to another user than the container's
        try: out = (await asyncio.wait_for(proc.communicate(), 300))[0].decode(errors="replace")
        except asyncio.TimeoutError:
            proc.kill()
            out = (await proc.communicate())[0].decode(errors="replace") + " (timed out after 300 s)"
        print(f"""[kimi] git pull {repo}: {"ok" if proc.returncode == 0 else out.strip()[:300]}""", flush=True)
        if proc.returncode: errs.append(f"""git pull {repo.name}: {out.strip()[:200]}""")
    return errs

def _new_run(state: dict, conn: dict, label: str, sources: dict) -> dict:
    """A fresh progress record for one pass, kept in the sync state as _run (shown under Sync in the left panel)."""
    state["_run"] = {"conn": conn["_id"], "group": conn.get("display_name", conn["_id"]), "label": label, "sources": sources, "started": _now(), "total": 0, "done": 0, "skipped": 0, "replaced": 0, "removed": 0, "errors": [], "current": "", "finished": ""}
    return state["_run"]

async def _ingest(conn: dict, sources: dict, only_changed: bool, label: str):
    """Puts the sources' files into one knowledge group. Each replaces the documents the group holds for an earlier version of it; only_changed skips files whose content is the same as when they last went into this group. Progress and the outcome are kept in the sync state ("_run") after every file, so a pass cut off by a restart can be continued."""
    async with _INGEST_LOCK:
        state = _load_sync_state()
        run = _new_run(state, conn, label, sources)
        pull = only_changed and cfg.get_group("general").load().get("sync_git_pull")
        if pull: run["current"] = "git pull"; _save_sync_state(state)
        if pull: run["errors"] += await _git_pull(sources)
        items = await asyncio.to_thread(_files, sources)
        run["total"] = len(items)
        _save_sync_state(state)
        try:
            index = await AIM.connections.lightrag_doc_index(conn)
            for prefix, fm, f in items:
                rel = _rel(fm, f)
                key, data = f"""{conn["_id"]}:{prefix}:{rel}""", f.read_bytes()
                run["current"] = rel
                old = f"""{conn["_id"]}:{prefix}:{_named(rel)}"""   # the entry from before the sources moved under a stripped folder
                if key not in state and old != key and old in state: state[key] = state.pop(old)
                if only_changed and _unchanged(state.get(key), f, data): state[key] = _hash(data); run["skipped"] += 1; continue
                r = await _send(conn, rel, data, index)
                if r.get("error") or r.get("status") in ("failure", "duplicated"): run["errors"].append(f"""{rel}: {str(r.get("error") or r.get("message") or r.get("status"))[:200]}""")
                else: state[key] = _hash(data); run["done"] += 1; run["replaced"] += bool(r.get("replaced"))
                _save_sync_state(state)
            if only_changed: await _remove_keys(conn, _gone_keys(conn["_id"], sources, items, state), "gone from the source", state, run, index)
        except Exception as e: run["errors"].append(f"{type(e).__name__}: {e}"[:300]); print(f"[kimi] ingest pass stopped: {e}", flush=True)
        run["current"], run["finished"] = "", _now()
        _save_sync_state(state)

def _gone_keys(conn_id: str, sources: dict, items: list, state: dict) -> list:
    """Sync-state keys of this group for files under a checked folder that are no longer on disk. A checked folder with no file on disk right now (an unmounted volume, a clone being swapped) is left out, so a missing folder never empties the group.
    A file still on disk that a pass does not send (Never send) is not gone: its documents go only when a person removes them (_remove_unsent)."""
    live, fms = {f"""{conn_id}:{p}:{_rel(fm, f)}""" for p, fm, f in items}, {"kg": FM_KG, "common": FM_COMMON}
    roots = [r for r in (f"""{conn_id}:{p}:{rel.strip("/")}""" for p, rels in sources.items() for rel in rels if rel.strip("/")) if any(k.startswith(r + "/") or k == r for k in live)]
    return [k for k in state if k not in live and any(k.startswith(r + "/") for r in roots) and not fms[k.split(":", 2)[1]].resolve(k.split(":", 2)[2]).exists()]

def _unsent_keys(conn_id: str, state: dict, rules: list) -> list:
    """Sync-state keys of this group for files a pass no longer sends (Never send rules): the group still holds documents Kimi put in for them."""
    return [k for k in state if k.startswith(conn_id + ":") and not _sendable(k.split(":", 2)[2], rules)]

async def _remove_keys(conn: dict, keys: list, why: str, state: dict, run: dict, index: dict):
    """Deletes from the group, in one request, the documents of these sync-state keys (files Kimi itself put in), then drops the keys; why names the reason in the progress line and in a failure."""
    if not keys: return
    run["removed_why"], run["current"] = why, f"removing {len(keys)} file(s) ({why})"
    _save_sync_state(state)
    r = await AIM.connections.lightrag_delete_files(conn, [_upload_name(k.split(":", 2)[2]) for k in keys], index=index)
    if r.get("error"): run["errors"].append(f"""{len(keys)} file(s) ({why}) not removed: {str(r["error"])[:200]}"""); return
    for k in keys: state.pop(k)
    run["removed"] += len(keys)
    _save_sync_state(state)

async def _remove_unsent(conn: dict):
    """Deletes from the group the documents of files under Never send, started by a person from the Documents tab. Sync passes never do this by themselves, so a pattern typed by mistake removes nothing until someone confirms it."""
    async with _INGEST_LOCK:
        state = _load_sync_state()
        keys = _unsent_keys(conn["_id"], state, _skip_rules())
        run = _new_run(state, conn, "Remove Never send documents", {})
        run["total"] = len(keys)
        _save_sync_state(state)
        try: await _remove_keys(conn, keys, "Never send", state, run, await AIM.connections.lightrag_doc_index(conn))
        except Exception as e: run["errors"].append(f"{type(e).__name__}: {e}"[:300]); print(f"[kimi] removal pass stopped: {e}", flush=True)
        run["current"], run["finished"] = "", _now()
        _save_sync_state(state)

def _run_html(queued: bool = False) -> str:
    """The latest ingest pass: counts, what it is working on, failures; polls itself while a pass runs (or was just queued); a pass that never finished (server restarted) can be continued."""
    run, live = _load_sync_state().get("_run"), _INGEST_LOCK.locked() or queued
    if not run: return f"""<div {_post("kimi_ingest_status")} hx-trigger="load delay:2s"></div>""" if live else ""
    state = f"""working on {_esc(run["current"])}""" if live and run["current"] else "finished " + run["finished"][5:16].replace("T", " ") if run["finished"] else "waiting" if live else "cut off before it finished (server restarted?)"
    errs = f"""<details><summary style="color:#ff5f5f;cursor:pointer">{len(run["errors"])} failed</summary>{"".join(_div(e) for e in run["errors"][-30:])}</details>""" if run["errors"] else ""
    cont = f"""<button class="ui-btn" {_post("kimi_ingest_continue")} title="Runs the same pass again; files already done are skipped">Continue</button>""" if not live and not run["finished"] and run.get("sources") else ""
    poll = f"""<div {_post("kimi_ingest_status")} hx-trigger="load delay:4s"></div>""" if live else ""
    return f"""<div class="status-label">{_esc(run["label"])} into {_esc(run["group"])}: {run["done"]} in ({run["replaced"]} replaced an earlier version), {run["skipped"]} unchanged{", " + str(run["removed"]) + " removed (" + run.get("removed_why", "gone from the source") + ")" if run.get("removed") else ""}, of {run["total"]} - {state}</div>{errs}{cont}{poll}"""

def _start_ingest(conn: dict, sources: dict, only_changed: bool, label: str) -> str:
    _spawn(_ingest(conn, sources, only_changed, label))
    return f"""<span class="status-label" style="color:var(--accent)">&#x2713; {_esc(label)} queued: {sum(len(v) for v in sources.values())} checked item(s) into {_esc(conn.get("display_name", conn["_id"]))}{" - waits for the pass already running" if _INGEST_LOCK.locked() else ""}</span>"""

def sync_service(group: str, sourceset: str) -> str:
    """For other tools (the "kimi" service, e.g. Git Manager's sync after a push): starts a Sync pass - changed files only, documents of deleted files removed - of a saved source set into a knowledge group (its connection id or display name)."""
    conn = next((c for c in AIM.connections.list_conns(conn_type="lightrag") if group in (c["_id"], c.get("display_name"))), None)
    if not conn: raise ValueError(f"no knowledge group {group}")
    sources = _load_sourcesets().get(sourceset)
    if not sources: raise ValueError(f"no source set {sourceset} - save one in Kimi's left panel")
    busy = _INGEST_LOCK.locked()
    _spawn(_ingest(conn, sources, True, f"Sync ({sourceset})"))
    return f"""Sync of source set {sourceset} into {conn.get("display_name", conn["_id"])} started{" - it waits for the pass already running" if busy else ""} (progress in Kimi)"""

async def _sources_or_error(request, imr):
    s = await _kg_state(request)
    conn, sources = _conn_of(s), {"kg": s.get("selected_kg", []), "common": s.get("selected_common", [])}
    if not conn or not any(sources.values()): imr.oob('<span style="color:#ff5f5f;font-size:.7rem">Pick a knowledge group and check the files or folders first.</span>', "kg-run-msg"); return None, None
    return conn, sources

async def _h_ingest_selected(request, payload, imr):
    """The checked files (a checked folder = every file in it) go in now, each replacing an earlier version of itself in the group."""
    conn, sources = await _sources_or_error(request, imr)
    if conn: imr.oob(_start_ingest(conn, sources, False, "Ingest"), "kg-run-msg")
    return imr.oob(_run_html(bool(conn)), "kg-run")

async def _h_sync_now(request, payload, imr):
    """Like Ingest Selected, but only files whose content changed since they last went into this group."""
    conn, sources = await _sources_or_error(request, imr)
    if conn: imr.oob(_start_ingest(conn, sources, True, "Sync"), "kg-run-msg")
    return imr.oob(_run_html(bool(conn)), "kg-run")

async def _h_ingest_continue(request, payload, imr):
    run = _load_sync_state().get("_run") or {}
    conn = AIM.connections.get_conn(run.get("conn", ""), conn_type="lightrag")
    if conn: imr.oob(_start_ingest(conn, run["sources"], True, "Continue"), "kg-run-msg")
    return imr.oob(_run_html(bool(conn)), "kg-run")

async def _h_ingest_status(request, payload, imr): return imr.oob(_run_html(), "kg-run")

# --- Left panel (persistent, not part of tab switching) ---

def _conn_select_html(active_id):
    conns = AIM.connections.list_conns(conn_type="lightrag")
    if not conns: return '<div style="font-size:.7rem;color:var(--text_muted)">No LightRAG connections - add one in AI Tools &rarr; Settings &rarr; Connections.</div>'
    active = AIM.connections.get_conn(active_id, conn_type="lightrag") or conns[0]
    notes = active.get("values", {}).get("domain_notes", "")
    sel = UI.select("conn_id", [(c["_id"], c.get("display_name", c["_id"])) for c in conns], selected=active["_id"], htmx={"post":"/im/in","trigger":"change","target":"#kg-health", "vals": _vals("kimi_conn_select"), "include":"this"})
    notes_html = f'<div style="font-size:.68rem;color:var(--text_muted);margin-top:.2rem">{_esc(notes)}</div>' if notes else ""
    return sel + notes_html

def _checked_html(fm, selected, prefix) -> str:
    """What is ticked in one tree, by folder, each folder with a cross that unticks its files - ticked files inside closed folders show here, and ticked files no longer on disk are counted."""
    if not selected: return ""
    groups = {}
    for rel in sorted(selected): groups.setdefault(rel.rpartition("/")[0], []).append(rel)
    gone, rules = {rel for rel in selected if not fm.resolve(rel).is_file()}, _skip_rules()
    unsent = sum(1 for rel in selected if not _sendable(rel, rules))
    rows = "".join(f"""<div style="display:flex;align-items:center;gap:.3rem"><span class="mono" style="flex:1;min-width:0;overflow:hidden;text-overflow:ellipsis;white-space:nowrap" title="{_esc(", ".join(x.rpartition("/")[2] for x in rels[:60]))}">{_esc(rels[0] if len(rels) == 1 else (folder or "(top level)") + "/")}</span><span class="status-label">{len(rels) if len(rels) > 1 else ""}{f" ({len(gone & set(rels))} not on disk)" if gone & set(rels) else ""}</span><button type="button" class="cm-qbtn" style="color:#ff5f5f" {_post("kimi_uncheck", src=prefix, folder=folder)} title="Untick these">&#x2715;</button></div>""" for folder, rels in groups.items())
    return f"""<details style="font-size:.68rem;margin-bottom:.2rem"><summary class="status-label" style="cursor:pointer;color:var(--accent)">{len(selected)} ticked in {len(groups)} folder(s){f", {len(gone)} no longer on disk" if gone else ""}{f", {unsent} never sent (Never send setting)" if unsent else ""}</summary><div style="max-height:12rem;overflow-y:auto">{rows}</div></details>"""

def _trees_oob(imr, s: dict):
    """Both source trees, with their ticked lists, redrawn for this person."""
    for src, fm in (("kg", FM_KG), ("common", FM_COMMON)): imr.oob(_source_tree_html(fm, s.get(f"selected_{src}", []), src), f"kg-tree-{src}", swap="outerHTML")
    return imr

def _source_tree_html(fm, selected, prefix):
    if not fm.root.exists(): return '<div style="color:var(--text_muted);font-size:.75rem;padding:.3rem">No files yet.</div>'
    return _checked_html(fm, selected, prefix) + UI.tree(items=fm.root, mode="file", selectable=True, selected=set(selected), post_url="/im/in", target=f"#kg-tree-{prefix}", swap="outerHTML", extra_vals={"type":"kimi_select_file","branch":"kimi","lvl":2,"src":prefix}, context_menu_url=f"{_u('ctx_menu')}?src={prefix}")

async def _left_panel(request):
    s = await _kg_state(request)
    return f"""<div style="display:flex;flex-direction:column;height:100%;overflow:hidden">
                    <div style="padding:.5rem;border-bottom:var(--border-thick) solid var(--border)">
                        {UI.field("Knowledge Group", _conn_select_html(s["conn_id"]))}
                        <div class="row" style="font-size:.7rem;color:var(--text_muted)"><div id="kg-health" class="grow" hx-post="/im/in" hx-vals='{_vals("kimi_health")}' hx-trigger="load" hx-swap="innerHTML">checking...</div><button type="button" class="btn-icon" style="font-size:.7rem" hx-post="/im/in" hx-vals='{_vals("kimi_health")}' hx-target="#kg-health" hx-swap="innerHTML" title="Check the knowledge server now">&#x21BB; Health</button></div>
                    </div>
                    <div style="flex:1;min-height:8rem;overflow-y:auto">
                        <details open style="border-bottom:var(--border-thick) solid var(--border)">
                            <summary style="padding:.3rem .5rem;cursor:pointer;font-size:.7rem;color:var(--text_muted);text-transform:uppercase;list-style:none">&#x1F4DA; Shared Knowledge<button class="btn-icon" style="float:right;font-size:.7rem" {_post("kimi_upload_modal", src="kg")} onclick="event.stopPropagation()">&#x2795;</button></summary>
                            <div id="kg-tree-kg" style="padding:.2rem .4rem">{_source_tree_html(FM_KG, s["selected_kg"], "kg")}</div>
                        </details>
                        <details open style="border-bottom:var(--border-thick) solid var(--border)">
                            <summary style="padding:.3rem .5rem;cursor:pointer;font-size:.7rem;color:var(--text_muted);text-transform:uppercase;list-style:none">&#x1F310; Common (server-wide)<button class="btn-icon" style="float:right;font-size:.7rem" {_post("kimi_upload_modal", src="common")} onclick="event.stopPropagation()">&#x2795;</button></summary>
                            <div id="kg-tree-common" style="padding:.2rem .4rem">{_source_tree_html(FM_COMMON, s["selected_common"], "common")}</div>
                        </details>
                    </div>
                    <div style="padding:.3rem .5rem;border-top:var(--border-thick) solid var(--border);display:flex;flex-direction:column;gap:.2rem">
                        <div style="display:flex;align-items:center;gap:.2rem"><span class="status-label" style="text-transform:uppercase;font-size:.62rem">Source sets</span><input type="text" id="kg-sourceset-name" placeholder="save checked as..." class="module-select" style="flex:1;min-width:0;font-size:.7rem;margin:0"><button type="button" class="cm-qbtn" onclick="htmx.ajax('POST','/im/in',{{values:{{type:'kimi_sourceset_save',branch:'kimi',lvl:2,name:document.getElementById('kg-sourceset-name').value}},swap:'none'}})">Save</button></div>
                        {_sourceset_select_html()}
                    </div>
                    <details class="sec" style="padding:.2rem .5rem;border-top:var(--border-thick) solid var(--border)"><summary class="status-label" style="text-transform:uppercase;font-size:.62rem;cursor:pointer">Batch tools (act on checked files)</summary>
                        <div style="display:flex;flex-direction:column;gap:.2rem;padding-top:.2rem">
                            <div class="row" style="align-items:center;gap:.2rem"><input type="number" id="kg-chunk-size" value="4000" step="100" class="module-select" style="flex:1;font-size:.7rem;margin:0" title="Approximate characters per chunk"><button class="cm-qbtn" hx-post="/im/in" hx-target="body" hx-swap="none" hx-vals='js:{{"type":"kimi_chunk_selected","branch":"kimi","lvl":2,"chunk_size":document.getElementById("kg-chunk-size").value}}'>&#x2702; Chunk</button></div>
                            <div class="row" style="align-items:center;gap:.2rem"><input type="text" id="kg-combine-name" placeholder="combined file name" class="module-select" style="flex:1;font-size:.7rem;margin:0"><button class="cm-qbtn" hx-post="/im/in" hx-target="body" hx-swap="none" hx-vals='js:{{"type":"kimi_combine_selected","branch":"kimi","lvl":2,"combine_name":document.getElementById("kg-combine-name").value,"combine_src":"kg"}}'>&#x1F517; Combine</button></div>
                            <div class="row" style="align-items:center;gap:.2rem"><input type="text" id="kg-docx-name" placeholder="export file name" class="module-select" style="flex:1;font-size:.7rem;margin:0"><button class="cm-qbtn" hx-post="/im/in" hx-target="body" hx-swap="none" hx-vals='js:{{"type":"kimi_export_docx","branch":"kimi","lvl":2,"docx_name":document.getElementById("kg-docx-name").value}}'>&#x1F4C4; Export .docx</button></div>
                            <div id="kg-ingest-log" style="font-size:.7rem;max-height:8rem;overflow-y:auto;font-family:var(--font-mono)"></div>
                        </div>
                    </details>
                    <div style="padding:.5rem;border-top:var(--border-thick) solid var(--border)">
                        <div class="row" style="align-items:center"><button class="ui-btn grow" {_post("kimi_ingest_selected")} title="The checked files go in now, each replacing an earlier version of itself">&#x2191; Ingest Selected</button><button class="ui-btn grow" {_post("kimi_sync_now")} title="Only checked files whose content changed since they last went into this group">&#x21BB; Sync Changed</button></div>
                        <div id="kg-run-msg" style="font-size:.7rem"></div>
                        <div id="kg-run" style="font-size:.7rem;margin-top:.2rem">{_run_html()}</div>
                    </div>
                    <div id="kg-modal"></div>
                </div>"""

# --- Query tab ---

def _opt_field(k: str, spec: dict, val) -> str:
    """One query option, pre-set to the group's default. Booleans have three states, so a group default can be turned off for one query and "server default" sends nothing."""
    label, hint = _esc(spec.get("label", k)), _esc(spec.get("hint", ""))
    if spec.get("type") == "boolean":
        cur = "" if val is None else "1" if val else "0"
        opts = "".join(f"""<option value="{v}"{" selected" if cur == v else ""}>{l}</option>""" for v, l in (("", "server default"), ("1", "on"), ("0", "off")))
        return f"""<label class="row status-label" title="{hint}">{label}<select name="opt_{k}" class="module-select">{opts}</select></label>"""
    return f"""<label class="row status-label" title="{hint}">{label}<input type="number" name="opt_{k}" class="module-select" value="{"" if val is None else val}"></label>"""

def _opts(payload: dict, schema: dict) -> dict:
    """opt_* form values -> typed query options; blank = not given (the group's default applies)."""
    out = {}
    for k, spec in schema.items():
        v = payload.get(f"opt_{k}")
        v = v[-1] if isinstance(v, list) else v
        if v in ("", None): continue
        out[k] = v == "1" if spec.get("type") == "boolean" else int(v) if re.fullmatch(r"-?\d+", str(v)) else float(v)
    return out

def _as_list(v) -> list: return v if isinstance(v, list) else [v] if v else []

async def _panel_query(request, msg: str = ""):
    s = await _kg_state(request)
    conn = _conn_of(s)
    schema, defaults = (AIM.connections.lightrag_query_options_schema(conn), AIM.connections.lightrag_query_defaults(conn)) if conn else ({}, {})
    ans = {"temperature": 0.2, **(s.get("answer") or {})}
    opt_fields = "".join(_opt_field(k, spec, defaults.get(k)) for k, spec in schema.items())
    modes = "".join(f"""<label class="row status-label"><input type="checkbox" name="modes" value="{m}"{" checked" if m in s["last_modes"] else ""}>{m}</label>""" for m in _MODES)
    groups = "".join(f"""<label class="row status-label"><input type="checkbox" name="conn_ids" value="{c["_id"]}"{" checked" if c["_id"] == s["conn_id"] else ""}>{_esc(c.get("display_name", c["_id"]))}</label>""" for c in AIM.connections.list_conns(conn_type="lightrag"))
    answer = f"""<label class="row status-label"><input type="checkbox" name="use_model" value="1"{" checked" if ans.get("use_model") else ""}> Retrieve only from the knowledge group; this model writes the answer</label><div id="kg-answer-fields" class="settings-compact">{ANSWER.render(ans, name_prefix="ka_")}</div>"""
    return f"""<div class="scroll-pane" style="padding:.8rem">
                    <form {_post("kimi_query")} hx-indicator="#kg-query-spin" style="display:flex;flex-direction:column;gap:.4rem;margin-bottom:.8rem">
                        <div class="row"><input type="text" name="q" value="{_esc(s.get('last_query',''))}" placeholder="Ask the knowledge base..." class="module-select grow" style="min-width:14rem"><button class="ui-btn">Ask</button><span id="kg-query-spin" class="htmx-indicator spin" title="Working...">&#x25CC;</span></div>
                        <div class="row"><span class="status-label">Modes (several = side by side):</span>{modes}</div>
                        <div class="row">{opt_fields}<button type="button" class="ui-btn" {_post("kimi_save_defaults")} hx-include="closest form" title="These option values become this group's defaults - for Kimi, knowledge nodes and context blocks alike">Save as group defaults</button><span class="status-label">{msg}</span></div>
                        <details><summary class="status-label" style="cursor:pointer">Compare across groups</summary><div class="row">{groups}</div></details>
                        <details{" open" if ans.get("use_model") else ""}><summary class="status-label" style="cursor:pointer">Answer with a chat model{" (on)" if ans.get("use_model") else ""}</summary>{answer}</details>
                    </form>
                    <div id="kg-query-result" style="font-size:.85rem;display:flex;flex-direction:column;gap:.6rem">{s.get("last_result","")}</div>
                </div>"""

async def _cell(cid: str, q: str, mode: str, extra: dict, ans: dict) -> tuple:
    """(header, answer, retrieved context) for one group and mode."""
    conn = AIM.connections.get_conn(cid, conn_type="lightrag")
    if not conn: return cid, "connection not found", ""
    head = f"""{conn.get("display_name", cid)} - {mode}"""
    if not ans.get("use_model"):
        r = await AIM.connections.lightrag_query_cached(conn, q, mode, **extra)
        return head, r.get("response") or r.get("error") or json.dumps(r), ""
    d = await AIM.connections.lightrag_query_data(conn, q, mode, **extra)
    if d.get("error"): return head, d["error"], ""
    if not d["context"].strip(): return head, "(nothing retrieved for this question)", ""
    try:
        chat, model = AIM.steps.resolve_chat(ans.get("conn_id", ""), ans.get("model", ""), "Kimi answer")
        msgs = [{"role": "system", "content": _ANSWER_SYS}, {"role": "user", "content": f"""Knowledge context:\n{d["context"]}\n\nQuestion: {q}"""}]
        text = "".join([t async for t, _ in AIM.connections.stream_llm(chat, msgs, model, think=False, num_ctx=int(ans.get("num_ctx") or 16384), temperature=float(ans.get("temperature") or 0.2))])
        return f"{head} - answered by {model}", text, d["context"]
    except RuntimeError as e: return head, str(e), d["context"]

def _cell_html(head: str, text: str, context: str) -> str: return f"""<div class="glass" style="padding:.6rem"><div style="font-weight:600; font-size:var(--font_size);margin-bottom:.2rem">{_esc(head)}</div><div class="pre-wrap">{_esc(text)}</div>{f'<details><summary class="status-label" style="cursor:pointer">Retrieved context ({len(context)} chars)</summary><div class="pre-wrap" style="font-size:var(--font_size)">{_esc(context)}</div></details>' if context else ''}</div>"""

async def _h_query(request, payload, imr):
    q, s = payload.get("q", "").strip(), await _kg_state(request)
    if not q: return imr.oob('<div style="color:#ff5f5f">Enter a question.</div>', "kg-query-result")
    targets, modes = [c for c in (_as_list(payload.get("conn_ids")) or [s["conn_id"]]) if c], [m for m in _as_list(payload.get("modes")) if m in _MODES] or ["hybrid"]
    if not targets: return imr.oob('<div style="color:#ff5f5f">No knowledge group selected.</div>', "kg-query-result")
    ans = {**ANSWER.parse(payload, name_prefix="ka_"), "use_model": bool(payload.get("use_model"))}
    async def _group(cid):
        conn = AIM.connections.get_conn(cid, conn_type="lightrag")
        extra = _opts(payload, AIM.connections.lightrag_query_options_schema(conn)) if conn else {}
        return [await _cell(cid, q, m, extra, ans) for m in modes]   # one group's modes in turn (one server); groups in parallel
    cells = [c for g in await asyncio.gather(*[_group(c) for c in targets]) for c in g]
    html = "".join(_cell_html(*c) for c in cells)
    s.update(last_query=q, last_modes=modes, last_result=html, answer=ans)
    await _kg_state(request, s)
    return imr.oob(html, "kg-query-result")

async def _h_save_defaults(request, payload, imr):
    """The option values in the form become the selected group's defaults (stored on the connection record, which the connection editor keeps when it saves)."""
    s = await _kg_state(request)
    conn = _conn_of(s)
    if not conn: return imr
    raw = AIM.connections.load_conn_raw(conn["_id"])
    raw["query_defaults"] = _opts(payload, AIM.connections.lightrag_query_options_schema(conn))
    AIM.connections.save_conn(conn["_id"], raw)
    return imr.oob(await _panel_query(request, f"""saved for {_esc(conn.get("display_name", conn["_id"]))}"""), "kimi-panel")

async def _h_answer_refresh(request, payload, imr): return imr.oob(ANSWER.render({**ANSWER.parse(payload, name_prefix="ka_"), "model": ""}, name_prefix="ka_"), "kg-answer-fields") # The answer model's connection changed: the fields again, Model listing that connection's models (cleared - the old one belongs to the old connection).

# --- Paste tab ---

async def _panel_paste(request):
    return f"""<div style="padding:1rem;height:100%;overflow-y:auto;box-sizing:border-box">
                    <form {_post("kimi_insert_text")} style="display:flex;flex-direction:column;gap:.5rem">
                        {UI.field("Source label (optional)", UI.input("source"))}
                        {UI.field("Text", UI.textarea("text", rows=14))}
                        <button class="ui-btn" hx-indicator="#kg-insert-spin">Insert <span id="kg-insert-spin" class="htmx-indicator spin">&#x25CC;</span></button>
                    </form>
                    <div id="kg-ingest-log2" style="margin-top:.6rem;font-size:.8rem"></div>
                </div>"""

async def _h_insert_text(request, payload, imr):
    conn = await _active_conn(request)
    if not conn: return imr.oob('<div style="color:#ff5f5f">No knowledge group selected.</div>', "kg-ingest-log2")
    r = await AIM.connections.lightrag_insert_text(conn, payload.get("text",""), payload.get("source",""))
    return imr.oob(f'<div style="color:{"#ff5f5f" if "error" in r else "var(--accent)"}">{_esc(str(r.get("error") or "Inserted"))}</div>', "kg-ingest-log2")
    
# --- Documents tab ---

async def _pipeline_html(conn, p: dict = None) -> str:
    """The knowledge server's indexing pipeline: busy or idle, which batch, its latest messages; refreshes itself while busy."""
    p = p if p is not None else await AIM.connections.lightrag_op(conn, "pipeline_status")
    if p.get("error"): return f"""<div id="kg-pipeline" class="status-label">Pipeline status not offered by this server: {_esc(str(p["error"])[:200])}</div>"""
    busy = bool(p.get("busy"))
    hist = "".join(_div(m) for m in (p.get("history_messages") or [])[-12:])
    cancel = f"""<button class="ui-btn" {_post("kimi_cancel_pipeline")} hx-confirm="Ask the server to stop indexing after the current step? Documents not finished stay pending or failed and can be reprocessed later.">Cancel</button>""" if busy else ""
    poll = f"""<div {_post("kimi_pipeline")} hx-trigger="load delay:5s"></div>""" if busy else ""
    started = f""" - started {_esc(str(p.get("job_start", ""))[:16].replace("T", " "))}""" if busy and p.get("job_start") else ""
    return f"""<div id="kg-pipeline" class="info-bar" style="display:block">
                   <div class="row">
                       <b>{"&#x25CF; indexing" if busy else "idle"}</b>
                       <span class="status-label grow">{_esc(p.get("job_name", ""))}{started} - batch {p.get("cur_batch", 0)} of {p.get("batchs", 0)}, {p.get("docs", 0)} document(s)</span>
                       <button class="ui-btn" {_post("kimi_reprocess")} title="Documents that failed or were left pending go through indexing again">Reprocess failed</button>
                       {cancel}
                   </div>
                   <div class="status-label">{_esc(p.get("latest_message", ""))}</div>
                   <details><summary class="status-label" style="cursor:pointer">Recent messages</summary><div class="mono" style="font-size:.7rem">{hist}</div></details>
                   {poll}
               </div>"""

def _doc_row(d: dict, pend: dict) -> str:
    err = d.get("error_msg") or d.get("error")
    t = pend.get(str(d.get("id", "")))
    status = d.get("status", "") if t is None else "waiting to delete (server busy)" if not t else f"""deleting since {datetime.fromtimestamp(t).strftime("%H:%M")}"""
    when = str(d.get("updated_at") or d.get("created_at") or "")[:16].replace("T", " ")
    delete = f"""<button class="cm-qbtn" style="color:#ff5f5f" hx-post="/im/in" hx-target="body" hx-swap="none" hx-vals='{_vals("kimi_doc_delete", doc_id=str(d.get("id", "")))}' hx-confirm="Delete this document and what was extracted only from it? Entities other documents share are rebuilt; the server's uploaded copy goes too.">&#x2715;</button>"""
    err_html = f"""<div style="color:#ff5f5f">{_esc(str(err)[:300])}</div>""" if err else ""
    return f"""<tr title="{_esc(str(d.get("content_summary", ""))[:400])}"><td class="mono">{_esc(d.get("file_path", ""))}{err_html}</td><td>{_esc(status)}</td><td>{d.get("chunks_count") or ""}</td><td>{d.get("content_length") or ""}</td><td>{when}</td><td class="status-label" title="{_esc(d.get("track_id") or "")}">{_esc(str(d.get("track_id") or "")[-14:])}</td><td>{delete}</td></tr>"""

DOC_SORTS = {"file_path": ("Name", "asc"), "updated_at": ("Updated", "desc"), "created_at": ("Added", "desc")}   # the knowledge server's sort fields: names A-Z, dates newest first

async def _panel_docs(request, msg: str = ""):
    s = await _kg_state(request)
    conn = _conn_of(s)
    if not conn: return '<div style="padding:1rem;color:var(--text_muted)">No connection selected.</div>'
    v = {"page": 1, "status": "", "sort": "file_path", **(s.get("docs_view") or {})}
    sort = v["sort"] if v["sort"] in DOC_SORTS else "file_path"
    r, p = await AIM.connections.lightrag_documents(conn, page=int(v["page"]), page_size=50, status=v["status"], sort_field=sort, sort_direction=DOC_SORTS[sort][1]), await AIM.connections.lightrag_op(conn, "pipeline_status")
    if r.get("error"): return f"""<div class="scroll-pane" style="padding:.8rem">{await _pipeline_html(conn, p)}<div style="color:#ff5f5f">{_esc(r["error"])} </div></div>"""
    note, recheck = _deletions_note(conn, p, {str(d.get("id", "")) for d in r.get("documents", [])})
    recheck += _unsent_bar(len(_unsent_keys(conn["_id"], _load_sync_state(), _skip_rules())))
    msg += note
    counts, pg = r.get("status_counts") or {}, r.get("pagination") or {}
    chips = "".join(f"""<button class="chip"{' style="background:var(--accent);color:var(--bg)"' if v["status"] == k else ""} {_post("kimi_docs_view", status=k, page=1)}>{_esc(k or "all")}{n}</button>""" for k, n in [("", sum(c for k2, c in counts.items() if k2.lower() != "all")) ] + sorted(counts.items()) if k.lower() != "all" or not k)
    sorts = "".join(f"""<button class="chip"{' style="background:var(--accent);color:var(--bg)"' if k == sort else ""} {_post("kimi_docs_view", sort=k, page=1)}>{lbl}</button>""" for k, (lbl, _) in DOC_SORTS.items())
    nav = f"""<div class="row"><button class="ui-btn" {_post("kimi_docs_view", status=v["status"], page=max(1, int(v["page"]) - 1))}{"" if pg.get("has_prev") else " disabled"}>&#x2039; Prev</button><span class="status-label">page {pg.get("page", 1)} of {pg.get("total_pages", 1)} - {pg.get("total_count", 0)} document(s)</span><button class="ui-btn" {_post("kimi_docs_view", status=v["status"], page=int(v["page"]) + 1)}{"" if pg.get("has_next") else " disabled"}>Next &#x203A;</button></div>"""
    rows = "".join(_doc_row(d, _DELETING.get(conn["_id"], {})) for d in r.get("documents", []))
    clear = f"""<button class="ui-btn" style="margin-top:1rem;color:#ff5f5f" {_post("kimi_clear_all")}>Clear Entire Knowledge Group <span id="kg-clear-spin" class="htmx-indicator spin">&#x25CC;</span></button>"""

    return f"""<div class="scroll-pane" style="padding:.8rem">{await _pipeline_html(conn, p)}{recheck}<div class="row">{chips}<span class="status-label">{msg}</span></div>
                <div class="row"><span class="status-label">Sort</span>{sorts}</div>
                <div style="overflow-x:auto"><table class="data-table" style="width:100%;font-size:.8rem"><thead><tr><th>File</th><th>Status</th><th>Chunks</th><th>Length</th><th>Updated</th><th>Track</th><th></th></tr></thead><tbody>{rows}</tbody></table></div>{nav}{clear}</div>"""

async def _docs_refresh(request, imr, msg: str = ""): return imr.oob(await _panel_docs(request, msg), "kimi-panel")

async def _h_docs_view(request, payload, imr):
    s = await _kg_state(request)
    s["docs_view"] = {**(s.get("docs_view") or {}), **{k: payload[k] for k in ("status", "sort") if k in payload}, "page": max(1, int(payload.get("page") or 1))}   # a status chip keeps the sort, a sort chip keeps the status
    await _kg_state(request, s)
    return await _docs_refresh(request, imr)

def _deletions_note(conn: dict, p: dict, listed: set) -> tuple:
    """(a note, a re-check trigger) for this group's deletions in progress. Once the server is idle, a deletion accepted over 20 s ago is finished: it leaves the in-progress list, and one whose document is still listed is reported as not completed."""
    pend, idle = _DELETING.get(conn["_id"], {}), not p.get("busy") and not p.get("error")
    over = [i for i, t in pend.items() if idle and t and time.time() - t > 20]
    stuck = [i for i in over if i in listed]
    for i in over: pend.pop(i)
    note = f""" {len(stuck)} deletion(s) did not complete - the server is idle and the document is still here; its Recent messages above (or its log) say why.""" if stuck else ""
    return note, f"""<div {_post("kimi_pipeline")} hx-trigger="load delay:5s"></div>""" if idle and pend else ""   # just accepted, the server not busy yet: look again shortly

def _unsent_bar(n: int) -> str:
    """The Documents tab's offer to remove what the group still holds from files under Never send (nothing is removed without this button and its confirm)."""
    return f"""<div class="info-bar">{n} document(s) here came from files Kimi no longer sends (Never send, Settings tab). <button class="ui-btn" {_post("kimi_remove_unsent")} hx-confirm="Delete the {n} document(s) Kimi put in for files under Never send? What only they gave the group goes with them; sending such a file again later puts it back.">Remove them</button></div>""" if n else ""

async def _h_remove_unsent(request, payload, imr):
    """Starts the removal pass for the selected group (Remove them, after its confirm); it waits for a pass already running."""
    conn = await _active_conn(request)
    if not conn: return await _docs_refresh(request, imr, "no knowledge group selected")
    _spawn(_remove_unsent(conn))
    imr.oob(_run_html(True), "kg-run")
    return await _docs_refresh(request, imr, "Removing them - progress under the Sync buttons in the left panel; each row goes once the server has finished with it.")

async def _delete_when_free(conn: dict, doc_id: str):
    """A deletion refused because the server was indexing: asked again every 15 s, for up to 6 hours."""
    for _ in range(1440):
        await asyncio.sleep(15)
        r = await AIM.connections.lightrag_delete_document(conn, doc_id)
        if r.get("status") != "busy": break
    pend = _DELETING.setdefault(conn["_id"], {})
    if r.get("error") or r.get("status") == "busy": pend.pop(doc_id, None); print(f"""[kimi] deleting {doc_id} failed: {r.get("error") or "the server stayed busy for 6 h"}""", flush=True)
    else: pend[doc_id] = time.time()

async def _h_doc_delete(request, payload, imr):
    """The server deletes in the background (it rebuilds what the document shared with others) and refuses while it is indexing; both are shown on the row until the document is gone."""
    conn, doc_id = await _active_conn(request), payload.get("doc_id", "")
    if not conn: return await _docs_refresh(request, imr, "no knowledge group selected")
    r, pend = await AIM.connections.lightrag_delete_document(conn, doc_id), _DELETING.setdefault(conn["_id"], {})
    if r.get("error"): return await _docs_refresh(request, imr, _esc(f"""Not deleted: {r["error"]}"""))
    if r.get("status") == "busy": pend[doc_id] = 0; _spawn(_delete_when_free(conn, doc_id)); return await _docs_refresh(request, imr, "The server is indexing and refuses deletions until it is done - this document is deleted as soon as it is free (asked again every 15 s).")
    pend[doc_id] = time.time()
    return await _docs_refresh(request, imr, _esc(f"""{r.get("message") or r.get("status") or "Deletion started"} - the row stays, marked deleting, until the server has finished."""))

async def _h_clear_all(request, payload, imr):
    conn = await _active_conn(request)
    if conn: await AIM.connections.lightrag_clear_all(conn)
    return await _docs_refresh(request, imr)

async def _h_pipeline(request, payload, imr):
    conn = await _active_conn(request)
    if conn and _DELETING.get(conn["_id"]): return await _docs_refresh(request, imr)   # deletions in progress: the whole list, so finished documents leave it
    return imr.oob(await _pipeline_html(conn), "kg-pipeline", swap="outerHTML") if conn else imr

async def _h_reprocess(request, payload, imr):
    conn = await _active_conn(request)
    r = await AIM.connections.lightrag_op(conn, "reprocess_failed") if conn else {}
    return await _docs_refresh(request, imr, _esc(r.get("error") or r.get("message") or ""))

async def _h_cancel_pipeline(request, payload, imr):
    conn = await _active_conn(request)
    r = await AIM.connections.lightrag_op(conn, "cancel_pipeline") if conn else {}
    return await _docs_refresh(request, imr, _esc(r.get("error") or r.get("message") or ""))

# --- Graph tab (interactive, /static/js/graph_view.js) ---

GRAPH_LAYOUTS = (("force", "Force"), ("ring", "Ring (by type)"), ("rings", "Rings (hubs inside)"))   # names graph_view.js knows

def _graph_data(g: dict) -> str:
    """The server's nodes and edges in graph_view's shape, as JSON safe to embed in a script tag. LightRAG joins merged descriptions with <SEP>."""
    def desc(p, n): return str((p or {}).get("description", "")).replace("<SEP>", "\n")[:n]
    nodes = [{"id": str(n.get("id")), "label": str((n.get("labels") or [n.get("id")])[0]), "type": (n.get("properties") or {}).get("entity_type", ""), "info": desc(n.get("properties"), 800)} for n in g["nodes"]]
    edges = [{"s": str(e.get("source")), "t": str(e.get("target")), "info": desc(e.get("properties"), 300)} for e in g["edges"]]
    return json.dumps({"nodes": nodes, "edges": edges}, ensure_ascii=False).replace("</", "<\\/")

async def _panel_graph(request):
    s = await _kg_state(request)
    conn = _conn_of(s)
    if not conn: return '<div style="padding:1rem;color:var(--text_muted)">No connection selected.</div>'
    g = {"label": "*", "depth": 2, "max_nodes": int(s.get("graph_limit") or 300), "layout": "force", **(s.get("graph") or {})}
    labels = await AIM.connections.lightrag_op(conn, "graph_labels")
    data = await AIM.connections.lightrag_graph(conn, g["label"], int(g["depth"]), int(g["max_nodes"]))
    opts = "".join(f"""<option value="{_esc(l)}">""" for l in (labels if isinstance(labels, list) else [])[:3000])
    layouts = "".join(f"""<option value="{v}"{" selected" if v == g["layout"] else ""}>{l}</option>""" for v, l in GRAPH_LAYOUTS)
    form = f"""<form class="row" {_post("kimi_graph_show")}><input name="label" list="kg-labels" value="{_esc(g["label"])}" class="module-select" style="width:14rem" placeholder="entity, or * for everything"><datalist id="kg-labels">{opts}</datalist>
               <label class="row status-label">Depth<input type="number" name="depth" value="{g["depth"]}" min="1" max="6" class="module-select"></label><label class="row status-label">Max nodes<input type="number" name="max_nodes" value="{g["max_nodes"]}" min="10" step="10" class="module-select"></label><label class="row status-label">Layout<select name="layout" class="module-select" onchange="this.form.requestSubmit()">{layouts}</select></label><button class="ui-btn">Show</button>
               <span class="status-label">{"" if data.get("error") else str(len(data["nodes"])) + " nodes, " + str(len(data["edges"])) + " edges" + (" - cut at max nodes" if data.get("is_truncated") else "")}</span><span class="grow"></span><a class="ui-btn" href="{_esc(AIM.connections.base_url(conn))}/webui/" target="_blank" rel="noopener" title="The knowledge server's own WebUI, in a new browser tab">Server WebUI</a></form>"""
    if data.get("error"): return f"""<div style="padding:.5rem 1rem">{form}<div style="color:#ff5f5f">Graph fetch failed: {_esc(data["error"])}</div></div>"""
    init = """(function go() { var b = document.getElementById('kg-graph'); if (!b || b.dataset.ready) return; if (!window.graphView) return setTimeout(go, 50); b.dataset.ready = 1; graphView(b, {layout: b.dataset.layout, onOpen: function (n) { htmx.ajax('POST', '/im/in', {values: {type: 'kimi_graph_show', branch: 'kimi', lvl: 2, label: n.id}, swap: 'none'}); }}); })();"""
    return f"""<div style="padding:.3rem 1rem;border-bottom:var(--border-thick) solid var(--border)">{form}</div><div id="kg-graph" data-layout="{_esc(g["layout"])}" style="height:calc(100% - 3rem)"><script type="application/json">{_graph_data(data)}</script></div><script>{init}</script>"""

async def _h_graph_show(request, payload, imr):
    """Label, depth and size of the graph shown; double-tapping a node centres the graph on it."""
    s = await _kg_state(request)
    prev = {"depth": 2, "max_nodes": 300, **(s.get("graph") or {})}
    s["graph"] = {"label": payload.get("label") or "*", "depth": max(1, min(6, int(payload.get("depth") or prev["depth"]))), "max_nodes": max(10, int(payload.get("max_nodes") or prev["max_nodes"])), "layout": payload.get("layout") if payload.get("layout") in dict(GRAPH_LAYOUTS) else prev.get("layout", "force")}
    await _kg_state(request, s)
    return imr.oob(await _panel_graph(request), "kimi-panel")

# --- Server tab: what this knowledge server offers vs what its profile uses ---

async def _panel_settings(request, note: str = "") -> str:
    """Kimi's own settings (data/ai_tools/kimi_settings.json): document naming, the scheduled sync. They apply to every user of this portal, so an admin saves them."""
    g = cfg.get_group("general")
    save = '<div class="form-actions"><button class="ui-btn">Save settings</button></div>' if request.state.user.role == "admin" else '<div class="status-label">Only an admin changes these.</div>'
    return f"""<form {_post("kimi_settings_save")} hx-include="this" class="settings-compact" style="padding:.8rem;max-width:42rem">{f'<div class="status-label" style="color:var(--accent)">{_esc(note)}</div>' if note else ""}{g.render(g.load())}{save}</form>"""

async def _h_settings_save(request, payload, imr):
    if request.state.user.role != "admin": return imr
    cfg.get_group("general").save(payload)
    s = await _kg_state(request)
    conn = _conn_of(s)
    n = len(_unsent_keys(conn["_id"], _load_sync_state(), _skip_rules())) if conn else 0
    _trees_oob(imr, s)   # the ticked lists count files the new patterns leave out
    return imr.oob(await _panel_settings(request, f"""Saved.{f" {n} document(s) in the selected group came from files now under Never send - the Documents tab removes them." if n else ""}"""), "kimi-panel")

async def _panel_server(request):
    conn = await _active_conn(request)
    if not conn: return '<div style="padding:1rem;color:var(--text_muted)">No connection selected.</div>'
    api = await AIM.connections.lightrag_op(conn, "openapi")
    if api.get("error"): return f"""<div style="padding:1rem;color:#ff5f5f">This server does not publish its API description: {_esc(api["error"])}</div>"""
    used = {(ep.get("method", "GET").upper(), ep.get("path", "")): name for name, ep in AIM.connections.list_templates()["lightrag"].get("endpoints", {}).items()}
    offered = {(m.upper(), p): spec for p, ms in api.get("paths", {}).items() for m, spec in ms.items() if isinstance(spec, dict)}
    rows = "".join(f"""<tr><td>{m}</td><td class="mono">{_esc(p)}</td><td>{_esc(spec.get("summary", ""))}</td><td class="status-label">{_esc(used.get((m, p), ""))}</td></tr>""" for (m, p), spec in sorted(offered.items(), key=lambda x: x[0][1]))
    missing = "".join(f"""<div class="mono">{m} {_esc(p)} <span class="status-label">({_esc(n)})</span></div>""" for (m, p), n in used.items() if (m, p) not in offered and n != "openapi")   # the description itself is never listed in its own paths
    info = api.get("info") or {}
    missing_html = f"""<div class="info-bar" style="display:block"><b>In the profile but not offered here</b> (those features answer with an error on this server):{missing}</div>""" if missing else ""
    return f"""<div class="scroll-pane" style="padding:.8rem">
                   <div class="row">
                       <b>{_esc(info.get("title", "Server"))}</b>
                       <span class="status-label">version {_esc(info.get("version", "?"))} - {len(offered)} route(s); the last column names the profile operation using it</span>
                   </div>
                   {missing_html}
                   <table class="data-table" style="width:100%;font-size:.78rem"><thead><tr><th>Method</th><th>Path</th><th>Summary</th><th>Used as</th></tr></thead><tbody>{rows}</tbody></table>
               </div>"""

# --- Tabs / main route ---

async def _render_panel(request, state):
    active = state.get("active", "query")
    panel = {"paste": _panel_paste, "docs": _panel_docs, "graph": _panel_graph, "server": _panel_server, "settings": _panel_settings}.get(active, _panel_query)
    return state, await panel(request)

@router.get("")
@router.get("/")
async def root(request: Request):
    _ensure_sync_task()
    state = await TM._load(request)
    state["tabs"] = dict(_TABS)
    await TM._save(request, state)   # saved, so focusing a tab added since the person's state was written (Server) works
    state, panel_html = await _render_panel(request, state)
    tab_bar = await TM.tab_bar_fn(state, "kimi-tab-bar", "kimi", 2, allow_new=False, closable=False)
    left = await _left_panel(request)
    return ENV["templates"].TemplateResponse(name="base.html", request=request, context={
        "request": request, "user": request.state.user, "nesting_level": 2, "shell_id": IM.branch_id,
        "extra_css": BI.MD_BLOCK_CSS,
        "toolbars": {"top": UI.toolbar(side="top", content=tab_bar, size="2.5rem", id="kimi-top", nesting_level=2, start_open=True, locked=True), "left": UI.toolbar(side="left", content=left, size="18rem", overlay=False, start_open=True, resizable=True, nesting_level=2)},
        "content": f'<script src="/static/js/graph_view.js"></script><div id="kimi-panel" style="height:100%;overflow:hidden">{panel_html}</div>'})

async def _h_conn_select(request, payload, imr):
    s = await _kg_state(request); s["conn_id"] = payload.get("conn_id",""); await _kg_state(request, s)
    return imr.raw(await _health_html(request))

async def _health_html(request):
    conn = await _active_conn(request)
    if not conn: return '<span style="color:var(--text_muted)">No connection selected.</span>'
    h = await AIM.connections.lightrag_health(conn)
    if not h.get("ok"): return f"""<span style="color:#ff5f5f">&#x25CF; {_esc(str(h.get("detail","unreachable")))}</span>"""
    d = h["detail"] if isinstance(h.get("detail"), dict) else {}
    cfg = d.get("configuration") or {}
    facts = [f"v{d['core_version']}" if d.get("core_version") else "", f"LLM {cfg['llm_model']}" if cfg.get("llm_model") else "", f"embed {cfg['embedding_model']}" if cfg.get("embedding_model") else "", "busy" if d.get("pipeline_busy") else ""]   # LightRAG /health fields; other servers show only online
    return f"""<span style="color:#00ffa2">&#x25CF; online</span> {_esc(" - ".join(x for x in facts if x))}"""

async def _h_health(request, payload, imr): return imr.raw(await _health_html(request))

async def _h_select_file(request, payload, imr):
    path, is_dir, src = payload.get("path",""), str(payload.get("is_dir","false"))=="true", payload.get("src","kg")
    fm = FM_KG if src == "kg" else FM_COMMON
    s = await _kg_state(request); key = f"selected_{src}"; sel = set(s.get(key, []))
    if is_dir:
        full, rules = fm.resolve(path), _skip_rules()
        children = {r for r in (_rel(fm, f) for f in full.rglob("*") if f.is_file()) if _sendable(r, rules)} if full.is_dir() else set()   # the files a pass would send: hidden ones, caches and Never send matches are never ticked
        sel = {x for x in sel if not x.startswith(path.rstrip("/") + "/")} if children and children.issubset(sel) else sel | children   # unticking a folder unticks everything under it, files a pass would not send included
    else: sel.discard(path) if path in sel else sel.add(path)
    s[key] = list(sel); await _kg_state(request, s)
    return imr.oob(_source_tree_html(fm, sel, src), f"kg-tree-{src}", swap="outerHTML")

async def _h_uncheck(request, payload, imr):
    """Unticks the ticked files directly in one folder (the ticked list under a tree)."""
    src, folder = payload.get("src", "kg"), payload.get("folder", "")
    s = await _kg_state(request); key = f"selected_{src}"
    s[key] = [x for x in s.get(key, []) if x.rpartition("/")[0] != folder]
    await _kg_state(request, s)
    return imr.oob(_source_tree_html(FM_KG if src == "kg" else FM_COMMON, s[key], src), f"kg-tree-{src}", swap="outerHTML")

async def _h_upload_modal(request, payload, imr):
    src = payload.get("src","kg")
    fm = FM_KG if src == "kg" else FM_COMMON
    return imr.oob(fm.new_item_modal_html(f"kg-upload-{src}", target_id=f"kg-tree-{src}", swap="outerHTML", intent_type=f"kimi_upload_{src}", branch="kimi", lvl=2), "kg-modal")

async def _h_upload(request, payload, imr, src):
    """New files, folders or uploads. Uploaded files go into the selected knowledge group straight away (an ingest pass, replacing earlier versions of the same files)."""
    fm = FM_KG if src == "kg" else FM_COMMON
    s = await _kg_state(request)
    conn = _conn_of(s)
    parent, kind, name = payload.get("parent",""), payload.get("kind","file"), payload.get("name","")
    upload_raw = payload.get("upload")
    files = upload_raw if isinstance(upload_raw, list) else ([upload_raw] if upload_raw else [])
    try: rel_paths = json.loads(payload.get("rel_paths","[]") or "[]")
    except Exception: rel_paths = []
    async def _auto_ingest(saved_rels):
        if conn and saved_rels: _start_ingest(conn, {src: list(saved_rels)}, False, "Upload")
    if kind in ("upload", "upload_folder") and files:
        await fm.save_uploads(parent, files, rel_paths, on_complete=_auto_ingest)
    elif kind == "folder" and name.strip():
        fm.safe_join(parent, name.strip()).mkdir(parents=True, exist_ok=True)
    elif kind == "file" and name.strip():
        p = fm.safe_join(parent, name.strip())
        p.parent.mkdir(parents=True, exist_ok=True)
        if not p.exists(): p.write_text("", encoding="utf-8")
    imr.oob(_run_html(bool(conn) and kind in ("upload", "upload_folder") and bool(files)), "kg-run")
    return imr.oob(_source_tree_html(fm, s.get(f"selected_{src}", []), src), f"kg-tree-{src}", swap="outerHTML")

def _md_to_docx(markdown_text: str, out_path: Path):
    """First-pass markdown->docx converter covering the common cases (headers, paragraphs, bold/italic, lists). Not a full transpile of every markdown extension this codebase supports - tables/images/code blocks are left for a later pass."""
    d = docx.Document()
    for line in markdown_text.split("\n"):
        stripped = line.strip()
        if not stripped: d.add_paragraph(); continue
        h = re.match(r'^(#{1,6})\s+(.*)$', stripped)
        if h: d.add_heading(h.group(2), level=len(h.group(1))); continue
        bullet = re.match(r'^[-*]\s+(.*)$', stripped)
        if bullet: d.add_paragraph(bullet.group(1), style="List Bullet"); continue
        numbered = re.match(r'^\d+\.\s+(.*)$', stripped)
        if numbered: d.add_paragraph(numbered.group(1), style="List Number"); continue
        p = d.add_paragraph()
        pos = 0
        for m in re.finditer(r'\*\*(.+?)\*\*|\*(.+?)\*', stripped):
            if m.start() > pos: p.add_run(stripped[pos:m.start()])
            run = p.add_run(m.group(1) or m.group(2))
            if m.group(1): run.bold = True
            else: run.italic = True
            pos = m.end()
        if pos < len(stripped): p.add_run(stripped[pos:])
    d.save(str(out_path))

# --- Scheduled sync ---

def _in_window(window: str) -> bool:
    try:
        start_s, end_s = window.split("-")
        now = datetime.now().time()
        start, end = datetime.strptime(start_s.strip(), "%H:%M").time(), datetime.strptime(end_s.strip(), "%H:%M").time()
        return (start <= now <= end) if start <= end else (now >= start or now <= end)
    except Exception: return False

async def _scheduled_sync_loop():
    """Default-off (sync_enabled checkbox) - checks every 15min; does real work only inside the configured window."""
    while True:
        try:
            settings = cfg.get_group("general").load()
            if settings.get("sync_enabled") and _in_window(settings.get("sync_window", "")):
                conn, sources = AIM.connections.get_conn(settings.get("sync_conn_id", ""), conn_type="lightrag") if settings.get("sync_conn_id") else None, _load_sourcesets().get(settings.get("sync_sourceset", ""))
                if conn and sources: await _ingest(conn, sources, True, "Scheduled sync")
                else: print("[kimi] scheduled sync skipped - set Sync Knowledge Group and Sync Source Set in Kimi settings")
        except Exception as e: print(f"[kimi] sync loop error: {e}")
        await asyncio.sleep(900)

def right_panel() -> str: return """<div class="ait-rp"><div class="ait-rp-hd">Kimi</div><div style="font-size:.72rem;color:var(--text_muted);padding:.3rem">Knowledge Integration Manager - pick a knowledge group and ingest sources from the left panel.</div></div>"""

async def _h_move_modal(request, payload, imr):
    src = payload.get("src","kg")
    fm = FM_KG if src == "kg" else FM_COMMON
    path = payload.get("path","")
    return imr.oob(BI.move_modal_html(f"kimi-move-{src}", "/im/in", fm.folder_picker_html(), path, target_id=f"kg-tree-{src}", swap="outerHTML"), "kg-modal")

async def _h_move(request, payload, imr):
    src = payload.get("src","kg")
    fm = FM_KG if src == "kg" else FM_COMMON
    fm.move(payload.get("path",""), payload.get("parent",""))
    s = await _kg_state(request)
    return imr.oob(_source_tree_html(fm, s.get(f"selected_{src}", []), src), f"kg-tree-{src}", swap="outerHTML")

@router.get("/raw/{src}/{path:path}")
async def serve_raw(src: str, path: str):
    fm = FM_KG if src == "kg" else FM_COMMON
    p = fm.resolve(path)
    if not p.exists(): raise HTTPException(404)
    return FileResponse(p)

@router.get("/ctx_menu")
async def ctx_menu(path: str, src: str):
    return HTMLResponse(f"""<div style="padding:.2rem .4rem;display:flex;flex-direction:column;gap:.2rem;font-size:.7rem">
        <a href="{_u('raw', src, path)}" target="_blank" style="color:var(--accent)">&#x1F4C4; View</a>
        <button class="btn-icon" hx-post="/im/in" hx-target="body" hx-swap="none" hx-vals='{json.dumps({"type":"kimi_move_modal","branch":"kimi","lvl":2,"src":src,"path":path})}'>&#x21C4; Move</button>
    </div>""")
