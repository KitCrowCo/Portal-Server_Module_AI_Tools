# /modules/ai_tools/kimi/kimi.py
"""
Kimi - Knowledge Integration Manager Interface: the UI over knowledge groups (LightRAG servers). Sub-module of ai_tools, mounted at /module/ai_tools/kimi.
Sources: the shared _knowledge dir (also used by Tessa/Athena pipelines) and the server-wide _common dir.
Tabs: Query (modes and groups side by side, per-group default options, or retrieval only with any chat model writing the answer), Paste Text,
Documents (paged list with status counts, the server's indexing pipeline, delete, reprocess failed), Graph (interactive, around a label), Server (the routes the server offers vs the ones its profile uses).
Ingest: a file goes in under its folder path (notes/a.md -> notes__a.md) and replaces what the group holds for an earlier version of it; progress is saved per file, so a pass cut off by a restart continues.
All protocol logic lives in AIM.connections.lightrag_* and the connection profile (tools/ai_manager/_connections/lightrag.json) - this module is the UI.
"""
import re, json, asyncio, hashlib, docx
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
_MODES = ("hybrid", "local", "global", "naive", "mix")
_TABS = {"query": {"id": "query", "order": 0, "label": "Query", "icon": "&#x1F50D;"}, "paste": {"id": "paste", "order": 1, "label": "Paste Text", "icon": "&#x1F4DD;"}, "docs": {"id": "docs", "order": 2, "label": "Documents", "icon": "&#x1F4C4;"},
         "graph": {"id": "graph", "order": 3, "label": "Graph", "icon": "&#x1F578;"}, "server": {"id": "server", "order": 4, "label": "Server", "icon": "&#x2699;"}}
_ANSWER_SYS = "Answer the question from the knowledge context below. When the context does not hold the answer, say so instead of guessing. Name the source files in [brackets] when the context gives them."

UI = FM_KG = FM_COMMON = BI = IM = TM = AIM = cfg = ANSWER = None
_SYNC_TASK = None
_INGEST_LOCK = asyncio.Lock()   # one ingest pass at a time - a knowledge server indexes one batch at a time anyway; later passes wait their turn
_TASKS = set()                  # background passes, held because the event loop keeps only weak references to tasks

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
        BI.SettingField("sync_enabled", "Enable scheduled sync", "checkbox", False),
        BI.SettingField("sync_window", "Sync window (24h, HH:MM-HH:MM)", "text", "02:00-08:00", hint="Watched folders are re-ingested for changed files only while the current time falls in this window."),
        BI.SettingField("sync_conn_id", "Sync Knowledge Group", "select", "", options=lambda values=None: [("", "(none - scheduled sync does nothing)")] + [(c["_id"], c.get("display_name", c["_id"])) for c in AIM.connections.list_conns(conn_type="lightrag")]),
        BI.SettingField("sync_sourceset", "Sync Source Set (name)", "text", "", hint="A source set saved in Kimi's left panel: the files and folders the scheduled sync watches. Blank = scheduled sync does nothing."),
    ], json_path="data/ai_tools/kimi_settings.json")])
    ANSWER = BI.SettingsGroup("kimi_answer", "Answer with", AIM.steps.chat_fields("kimi_answer_refresh"), json_path="data/ai_tools/kimi_answer.json")   # rendered per person from state; json_path unused
    FM_KG = BI.FileManager(KG_DIR)
    FM_COMMON = BI.FileManager(COMMON_DIR)
    TM = BI.TabManager(namespace="kimi", tab_bar_id="kimi-tab-bar", content_id="kimi-panel", render_content_fn=_render_panel, intent_prefix="kimi", IM=IM, scope="user", nesting_level=2, allow_new=False, closable=False, empty={"tabs": dict(_TABS), "active": "query"})
    IM.scripts.update({f"kimi_{k}": [fn] for k, fn in {"move_modal": _h_move_modal, "move": _h_move, "conn_select": _h_conn_select, "health": _h_health, "select_file": _h_select_file, "ingest_selected": _h_ingest_selected, "sync_now": _h_sync_now,
                       "ingest_status": _h_ingest_status, "ingest_continue": _h_ingest_continue, "insert_text": _h_insert_text, "query": _h_query, "save_defaults": _h_save_defaults, "answer_refresh": _h_answer_refresh,
                       "clear_all": _h_clear_all, "doc_delete": _h_doc_delete, "docs_view": _h_docs_view, "pipeline": _h_pipeline, "reprocess": _h_reprocess, "cancel_pipeline": _h_cancel_pipeline, "graph_show": _h_graph_show,
                       "upload_modal": _h_upload_modal, "upload_kg": lambda r, p, i: _h_upload(r, p, i, "kg"), "upload_common": lambda r, p, i: _h_upload(r, p, i, "common"), "export_docx": _h_export_docx, "chunk_selected": _h_chunk_selected,
                       "combine_selected": _h_combine_selected, "sourceset_save": _h_sourceset_save, "sourceset_load": _h_sourceset_load, "sourceset_delete": _h_sourceset_delete}.items()})
    print("[kimi] ready")

def _ensure_sync_task():
    global _SYNC_TASK
    if _SYNC_TASK is None or _SYNC_TASK.done(): _SYNC_TASK = asyncio.create_task(_scheduled_sync_loop())

def _spawn(coro):
    t = asyncio.create_task(coro)
    _TASKS.add(t); t.add_done_callback(_TASKS.discard)

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
    paras = text.split("\n\n")
    chunks, cur = [], ""
    for p in paras:
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
                   <select id="kg-sourceset-picker" class="module-select" style="flex:1;font-size:.7rem"><option value="">-- source sets --</option>{opts}</select>
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
    imr.oob(_source_tree_html(FM_KG, s["selected_kg"], "kg"), "kg-tree-kg", swap="outerHTML")
    imr.oob(_source_tree_html(FM_COMMON, s["selected_common"], "common"), "kg-tree-common", swap="outerHTML")
    return imr

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
    imr.oob(_source_tree_html(FM_KG, s.get("selected_kg", []), "kg"), "kg-tree-kg", swap="outerHTML")
    imr.oob(_source_tree_html(FM_COMMON, s.get("selected_common", []), "common"), "kg-tree-common", swap="outerHTML")
    return imr

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
def _upload_name(rel: str) -> str: return AIM.connections.lightrag_upload_name(rel) if cfg.get_group("general").load().get("preserve_structure", True) else Path(rel).name

def _unchanged(prev, f: Path, data: bytes) -> bool:
    """Same content as when the file last went into this group. Entries saved before content hashes were kept hold the file's modification time instead."""
    return prev == _hash(data) or (isinstance(prev, (int, float)) and f.stat().st_mtime <= prev)

def _files(sources: dict) -> list:
    """(prefix, file manager, file) for every checked file and every file inside a checked folder. Runs in a worker thread - a large folder must not stall the server."""
    out = []
    for prefix, fm in (("kg", FM_KG), ("common", FM_COMMON)):
        for rel in sources.get(prefix, []):
            p = fm.resolve(rel)
            out += [(prefix, fm, f) for f in ([p] if p.is_file() else sorted(p.rglob("*")) if p.is_dir() else []) if f.is_file()]
    return out

async def _ingest(conn: dict, sources: dict, only_changed: bool, label: str):
    """Puts the sources' files into one knowledge group. Each replaces the documents the group holds for an earlier version of it; only_changed skips files whose content is the same as when
    they last went into this group. Progress and the outcome are kept in the sync state ("_run") after every file, so a pass cut off by a restart can be continued."""
    async with _INGEST_LOCK:
        state = _load_sync_state()
        items = await asyncio.to_thread(_files, sources)
        run = state["_run"] = {"conn": conn["_id"], "group": conn.get("display_name", conn["_id"]), "label": label, "sources": sources, "started": _now(), "total": len(items), "done": 0, "skipped": 0, "replaced": 0, "errors": [], "current": "", "finished": ""}
        _save_sync_state(state)
        try:
            index = await AIM.connections.lightrag_doc_index(conn)
            for prefix, fm, f in items:
                rel = str(f.relative_to(fm.root)).replace("\\", "/")
                key, data = f"""{conn["_id"]}:{prefix}:{rel}""", f.read_bytes()
                run["current"] = rel
                if only_changed and _unchanged(state.get(key), f, data): state[key] = _hash(data); run["skipped"] += 1; continue
                r = await AIM.connections.lightrag_replace_file(conn, _upload_name(rel), data, index=index)
                if r.get("error") or r.get("status") in ("failure", "duplicated"): run["errors"].append(f"""{rel}: {str(r.get("error") or r.get("message") or r.get("status"))[:200]}""")
                else: state[key] = _hash(data); run["done"] += 1; run["replaced"] += bool(r.get("replaced"))
                _save_sync_state(state)
        except Exception as e: run["errors"].append(f"{type(e).__name__}: {e}"[:300]); print(f"[kimi] ingest pass stopped: {e}", flush=True)
        run["current"], run["finished"] = "", _now()
        _save_sync_state(state)

def _run_html(queued: bool = False) -> str:
    """The latest ingest pass: counts, what it is working on, failures; polls itself while a pass runs (or was just queued); a pass that never finished (server restarted) can be continued."""
    run, live = _load_sync_state().get("_run"), _INGEST_LOCK.locked() or queued
    if not run: return f"""<div {_post("kimi_ingest_status")} hx-trigger="load delay:2s"></div>""" if live else ""
    state = f"""working on {_esc(run["current"])}""" if live and run["current"] else "finished " + run["finished"][5:16].replace("T", " ") if run["finished"] else "waiting" if live else "cut off before it finished (server restarted?)"
    errs = f"""<details><summary style="color:#ff5f5f;cursor:pointer">{len(run["errors"])} failed</summary>{"".join(_div(e) for e in run["errors"][-30:])}</details>""" if run["errors"] else ""
    cont = f"""<button class="ui-btn" {_post("kimi_ingest_continue")} title="Runs the same pass again; files already done are skipped">Continue</button>""" if not live and not run["finished"] else ""
    poll = f"""<div {_post("kimi_ingest_status")} hx-trigger="load delay:4s"></div>""" if live else ""
    return f"""<div class="status-label">{_esc(run["label"])} into {_esc(run["group"])}: {run["done"]} in ({run["replaced"]} replaced an earlier version), {run["skipped"]} unchanged, of {run["total"]} - {state}</div>{errs}{cont}{poll}"""

def _start_ingest(conn: dict, sources: dict, only_changed: bool, label: str) -> str:
    _spawn(_ingest(conn, sources, only_changed, label))
    return f"""<span class="status-label" style="color:var(--accent)">&#x2713; {_esc(label)} queued: {sum(len(v) for v in sources.values())} checked item(s) into {_esc(conn.get("display_name", conn["_id"]))}{" - waits for the pass already running" if _INGEST_LOCK.locked() else ""}</span>"""

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

def _source_tree_html(fm, selected, prefix):
    if not fm.root.exists(): return '<div style="color:var(--text_muted);font-size:.75rem;padding:.3rem">No files yet.</div>'
    return UI.tree(items=fm.root, mode="file", selectable=True, selected=set(selected), post_url="/im/in", target=f"#kg-tree-{prefix}", swap="outerHTML", extra_vals={"type":"kimi_select_file","branch":"kimi","lvl":2,"src":prefix}, context_menu_url=f"{_u('ctx_menu')}?src={prefix}")

async def _left_panel(request):
    s = await _kg_state(request)
    return f"""<div style="display:flex;flex-direction:column;height:100%;overflow:hidden">
                    <div style="padding:.5rem;border-bottom:var(--border-thick) solid var(--border)">
                        {UI.field("Knowledge Group", _conn_select_html(s["conn_id"]))}
                        <div id="kg-health" style="font-size:.7rem;color:var(--text_muted)" hx-post="/im/in" hx-vals='{_vals("kimi_health")}' hx-trigger="load" hx-swap="innerHTML">checking...</div>
                    </div>
                    <div style="flex:1;overflow-y:auto">
                        <details open style="border-bottom:var(--border-thick) solid var(--border)">
                            <summary style="padding:.3rem .5rem;cursor:pointer;font-size:.7rem;color:var(--text_muted);text-transform:uppercase;list-style:none">&#x1F4DA; Shared Knowledge<button class="btn-icon" style="float:right;font-size:.7rem" {_post("kimi_upload_modal", src="kg")} onclick="event.stopPropagation()">&#x2795;</button></summary>
                            <div id="kg-tree-kg" style="padding:.2rem .4rem">{_source_tree_html(FM_KG, s["selected_kg"], "kg")}</div>
                        </details>
                        <details open style="border-bottom:var(--border-thick) solid var(--border)">
                            <summary style="padding:.3rem .5rem;cursor:pointer;font-size:.7rem;color:var(--text_muted);text-transform:uppercase;list-style:none">&#x1F310; Common (server-wide)<button class="btn-icon" style="float:right;font-size:.7rem" {_post("kimi_upload_modal", src="common")} onclick="event.stopPropagation()">&#x2795;</button></summary>
                            <div id="kg-tree-common" style="padding:.2rem .4rem">{_source_tree_html(FM_COMMON, s["selected_common"], "common")}</div>
                        </details>
                    </div>
                    <div style="padding:.5rem;border-top:var(--border-thick) solid var(--border)">
                        <div style="font-size:.65rem;color:var(--text_muted);text-transform:uppercase;margin-bottom:.2rem">Source Sets</div>
                        <div style="display:flex;gap:.2rem">
                            <input type="text" id="kg-sourceset-name" placeholder="save selection as..." class="module-select" style="flex:1;font-size:.7rem">
                            <button class="cm-qbtn" onclick="htmx.ajax('POST','/im/in',{{values:{{type:'kimi_sourceset_save',branch:'kimi',lvl:2,name:document.getElementById('kg-sourceset-name').value}},swap:'none'}})">Save</button>
                        </div>
                        <div id="kg-sourceset-select" style="margin-top:.2rem">{_sourceset_select_html()}</div>
                    </div>
                    <div style="padding:.5rem;border-top:var(--border-thick) solid var(--border)">
                        <div style="font-size:.65rem;color:var(--text_muted);text-transform:uppercase;margin-bottom:.2rem">Batch Tools (act on files checked above)</div>
                        <div style="display:flex;gap:.2rem">
                            <input type="number" id="kg-chunk-size" value="4000" step="100" class="module-select" style="flex:1;font-size:.7rem" title="Approximate characters per chunk">
                            <button class="ui-btn" hx-post="/im/in" hx-target="body" hx-swap="none" hx-vals='js:{{"type":"kimi_chunk_selected","branch":"kimi","lvl":2,"chunk_size":document.getElementById("kg-chunk-size").value}}'>&#x2702; Chunk</button>
                        </div>
                        <div style="display:flex;gap:.2rem;margin-top:.3rem">
                            <input type="text" id="kg-combine-name" placeholder="combined file name" class="module-select" style="flex:1;font-size:.7rem">
                            <button class="ui-btn" hx-post="/im/in" hx-target="body" hx-swap="none" hx-vals='js:{{"type":"kimi_combine_selected","branch":"kimi","lvl":2,"combine_name":document.getElementById("kg-combine-name").value,"combine_src":"kg"}}'>&#x1F517; Combine</button>
                        </div>
                        <div style="display:flex;gap:.2rem;margin-top:.3rem">
                            <input type="text" id="kg-docx-name" placeholder="export file name" class="module-select" style="flex:1;font-size:.7rem">
                            <button class="ui-btn" hx-post="/im/in" hx-target="body" hx-swap="none" hx-vals='js:{{"type":"kimi_export_docx","branch":"kimi","lvl":2,"docx_name":document.getElementById("kg-docx-name").value}}'>&#x1F4C4; Export .docx</button>
                        </div>
                        <div id="kg-ingest-log" style="font-size:.7rem;margin-top:.4rem;max-height:8rem;overflow-y:auto;font-family:var(--font-mono)"></div>
                    </div>
                    <div style="padding:.5rem;border-top:var(--border-thick) solid var(--border)">
                        <div class="row"><button class="ui-btn grow" {_post("kimi_ingest_selected")} title="The checked files go in now, each replacing an earlier version of itself">&#x2191; Ingest Selected</button><button class="ui-btn grow" {_post("kimi_sync_now")} title="Only checked files whose content changed since they last went into this group">&#x21BB; Sync Changed</button></div>
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

def _cell_html(head: str, text: str, context: str) -> str:
    ctx = f"""<details><summary class="status-label" style="cursor:pointer">Retrieved context ({len(context)} chars)</summary><div class="pre-wrap" style="font-size:.75rem">{_esc(context)}</div></details>""" if context else ""
    return f"""<div class="glass" style="padding:.6rem"><div style="font-weight:600;font-size:.8rem;margin-bottom:.3rem">{_esc(head)}</div><div class="pre-wrap">{_esc(text)}</div>{ctx}</div>"""

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

async def _h_answer_refresh(request, payload, imr):
    """The answer model's connection changed: the fields again, Model listing that connection's models (cleared - the old one belongs to the old connection)."""
    return imr.oob(ANSWER.render({**ANSWER.parse(payload, name_prefix="ka_"), "model": ""}, name_prefix="ka_"), "kg-answer-fields")

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

async def _pipeline_html(conn) -> str:
    """The knowledge server's indexing pipeline: busy or idle, which batch, its latest messages; refreshes itself while busy."""
    p = await AIM.connections.lightrag_op(conn, "pipeline_status")
    if p.get("error"): return f"""<div id="kg-pipeline" class="status-label">Pipeline status not offered by this server: {_esc(str(p["error"])[:200])}</div>"""
    busy = bool(p.get("busy"))
    hist = "".join(_div(m) for m in (p.get("history_messages") or [])[-12:])
    cancel = f"""<button class="ui-btn" {_post("kimi_cancel_pipeline")} hx-confirm="Ask the server to stop indexing after the current step? Documents not finished stay pending or failed and can be reprocessed later.">Cancel</button>""" if busy else ""
    poll = f"""<div {_post("kimi_pipeline")} hx-trigger="load delay:5s"></div>""" if busy else ""
    started = f""" - started {_esc(str(p.get("job_start", ""))[:16].replace("T", " "))}""" if busy and p.get("job_start") else ""
    return f"""<div id="kg-pipeline" class="info-bar" style="display:block"><div class="row"><b>{"&#x25CF; indexing" if busy else "idle"}</b><span class="status-label grow">{_esc(p.get("job_name", ""))}{started} - batch {p.get("cur_batch", 0)} of {p.get("batchs", 0)}, {p.get("docs", 0)} document(s)</span>
               <button class="ui-btn" {_post("kimi_reprocess")} title="Documents that failed or were left pending go through indexing again">Reprocess failed</button>{cancel}</div>
               <div class="status-label">{_esc(p.get("latest_message", ""))}</div><details><summary class="status-label" style="cursor:pointer">Recent messages</summary><div class="mono" style="font-size:.7rem">{hist}</div></details>{poll}</div>"""

def _doc_row(d: dict) -> str:
    err = d.get("error_msg") or d.get("error")
    when = str(d.get("updated_at") or d.get("created_at") or "")[:16].replace("T", " ")
    delete = f"""<button class="cm-qbtn" style="color:#ff5f5f" hx-post="/im/in" hx-target="body" hx-swap="none" hx-vals='{_vals("kimi_doc_delete", doc_id=str(d.get("id", "")))}' hx-confirm="Delete this document and what was extracted only from it? Entities other documents share are rebuilt; the server's uploaded copy goes too.">&#x2715;</button>"""
    err_html = f"""<div style="color:#ff5f5f">{_esc(str(err)[:300])}</div>""" if err else ""
    return f"""<tr title="{_esc(str(d.get("content_summary", ""))[:400])}"><td class="mono">{_esc(d.get("file_path", ""))}{err_html}</td><td>{_esc(d.get("status", ""))}</td><td>{d.get("chunks_count") or ""}</td><td>{d.get("content_length") or ""}</td><td>{when}</td><td class="status-label" title="{_esc(d.get("track_id") or "")}">{_esc(str(d.get("track_id") or "")[-14:])}</td><td>{delete}</td></tr>"""

async def _panel_docs(request, msg: str = ""):
    s = await _kg_state(request)
    conn = _conn_of(s)
    if not conn: return '<div style="padding:1rem;color:var(--text_muted)">No connection selected.</div>'
    v = s.get("docs_view") or {"page": 1, "status": ""}
    r = await AIM.connections.lightrag_documents(conn, page=int(v["page"]), page_size=50, status=v["status"])
    if r.get("error"): return f"""<div class="scroll-pane" style="padding:.8rem">{await _pipeline_html(conn)}<div style="color:#ff5f5f">{_esc(r["error"])}</div></div>"""
    counts, pg = r.get("status_counts") or {}, r.get("pagination") or {}
    chips = "".join(f"""<button class="chip"{' style="background:var(--accent);color:var(--bg)"' if v["status"] == k else ""} {_post("kimi_docs_view", status=k, page=1)}>{_esc(k or "all")} {n}</button>""" for k, n in [("", sum(c for k2, c in counts.items() if k2.lower() != "all"))] + sorted(counts.items()) if k.lower() != "all" or not k)
    nav = f"""<div class="row"><button class="ui-btn" {_post("kimi_docs_view", status=v["status"], page=max(1, int(v["page"]) - 1))}{"" if pg.get("has_prev") else " disabled"}>&#x2039; Prev</button><span class="status-label">page {pg.get("page", 1)} of {pg.get("total_pages", 1)} - {pg.get("total_count", 0)} document(s)</span><button class="ui-btn" {_post("kimi_docs_view", status=v["status"], page=int(v["page"]) + 1)}{"" if pg.get("has_next") else " disabled"}>Next &#x203A;</button></div>"""
    rows = "".join(_doc_row(d) for d in r.get("documents", []))
    clear = f"""<button class="ui-btn" style="margin-top:1rem;color:#ff5f5f" {_post("kimi_clear_all")} hx-indicator="#kg-clear-spin" hx-confirm="Delete the ENTIRE knowledge graph for this group? This cannot be undone.">Clear Entire Knowledge Group <span id="kg-clear-spin" class="htmx-indicator spin">&#x25CC;</span></button>"""
    return f"""<div class="scroll-pane" style="padding:.8rem">{await _pipeline_html(conn)}<div class="row">{chips}<span class="status-label">{msg}</span></div>
               <div style="overflow-x:auto"><table class="data-table" style="width:100%;font-size:.8rem"><thead><tr><th>File</th><th>Status</th><th>Chunks</th><th>Length</th><th>Updated</th><th>Track</th><th></th></tr></thead><tbody>{rows}</tbody></table></div>{nav}{clear}</div>"""

async def _docs_refresh(request, imr, msg: str = ""): return imr.oob(await _panel_docs(request, msg), "kimi-panel")

async def _h_docs_view(request, payload, imr):
    s = await _kg_state(request)
    s["docs_view"] = {"page": max(1, int(payload.get("page") or 1)), "status": payload.get("status", "")}
    await _kg_state(request, s)
    return await _docs_refresh(request, imr)

async def _h_doc_delete(request, payload, imr):
    conn = await _active_conn(request)
    r = await AIM.connections.lightrag_delete_document(conn, payload.get("doc_id", "")) if conn else {"error": "no knowledge group selected"}
    return await _docs_refresh(request, imr, _esc(r.get("error") or r.get("message") or r.get("status") or "deleted"))

async def _h_clear_all(request, payload, imr):
    conn = await _active_conn(request)
    if conn: await AIM.connections.lightrag_clear_all(conn)
    return await _docs_refresh(request, imr)

async def _h_pipeline(request, payload, imr):
    conn = await _active_conn(request)
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
    g = {"label": "*", "depth": 2, "max_nodes": int(s.get("graph_limit") or 300), **(s.get("graph") or {})}
    labels = await AIM.connections.lightrag_op(conn, "graph_labels")
    data = await AIM.connections.lightrag_graph(conn, g["label"], int(g["depth"]), int(g["max_nodes"]))
    opts = "".join(f"""<option value="{_esc(l)}">""" for l in (labels if isinstance(labels, list) else [])[:3000])
    form = f"""<form class="row" {_post("kimi_graph_show")}><input name="label" list="kg-labels" value="{_esc(g["label"])}" class="module-select" style="width:14rem" placeholder="entity, or * for everything"><datalist id="kg-labels">{opts}</datalist>
               <label class="row status-label">Depth<input type="number" name="depth" value="{g["depth"]}" min="1" max="6" class="module-select"></label><label class="row status-label">Max nodes<input type="number" name="max_nodes" value="{g["max_nodes"]}" min="10" step="10" class="module-select"></label><button class="ui-btn">Show</button>
               <span class="status-label">{"" if data.get("error") else str(len(data["nodes"])) + " nodes, " + str(len(data["edges"])) + " edges" + (" - cut at max nodes" if data.get("is_truncated") else "")}</span><span class="grow"></span><a class="ui-btn" href="{_esc(AIM.connections.base_url(conn))}/webui/" target="_blank" rel="noopener" title="The knowledge server's own WebUI, in a new browser tab">Server WebUI</a></form>"""
    if data.get("error"): return f"""<div style="padding:.5rem 1rem">{form}<div style="color:#ff5f5f">Graph fetch failed: {_esc(data["error"])}</div></div>"""
    init = """(function go() { var b = document.getElementById('kg-graph'); if (!b || b.dataset.ready) return; if (!window.graphView) return setTimeout(go, 50); b.dataset.ready = 1;
                  graphView(b, {onOpen: function (n) { htmx.ajax('POST', '/im/in', {values: {type: 'kimi_graph_show', branch: 'kimi', lvl: 2, label: n.id}, swap: 'none'}); }}); })();"""
    return f"""<div style="padding:.3rem 1rem;border-bottom:var(--border-thick) solid var(--border)">{form}</div><div id="kg-graph" style="height:calc(100% - 3rem)"><script type="application/json">{_graph_data(data)}</script></div><script>{init}</script>"""

async def _h_graph_show(request, payload, imr):
    """Label, depth and size of the graph shown; double-tapping a node centres the graph on it."""
    s = await _kg_state(request)
    prev = {"depth": 2, "max_nodes": 300, **(s.get("graph") or {})}
    s["graph"] = {"label": payload.get("label") or "*", "depth": max(1, min(6, int(payload.get("depth") or prev["depth"]))), "max_nodes": max(10, int(payload.get("max_nodes") or prev["max_nodes"]))}
    await _kg_state(request, s)
    return imr.oob(await _panel_graph(request), "kimi-panel")

# --- Server tab: what this knowledge server offers vs what its profile uses ---

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
    return f"""<div class="scroll-pane" style="padding:.8rem"><div class="row"><b>{_esc(info.get("title", "Server"))}</b><span class="status-label">version {_esc(info.get("version", "?"))} - {len(offered)} route(s); the last column names the profile operation using it</span></div>
               {missing_html}
               <table class="data-table" style="width:100%;font-size:.78rem"><thead><tr><th>Method</th><th>Path</th><th>Summary</th><th>Used as</th></tr></thead><tbody>{rows}</tbody></table></div>"""

# --- Tabs / main route ---

async def _render_panel(request, state):
    active = state.get("active", "query")
    panel = {"paste": _panel_paste, "docs": _panel_docs, "graph": _panel_graph, "server": _panel_server}.get(active, _panel_query)
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
        "toolbars": {"top": UI.toolbar(side="top", content=tab_bar, size="2.5rem", id="kimi-top", nesting_level=2, start_open=True, locked=True),
                     "left": UI.toolbar(side="left", content=left, size="18rem", overlay=False, start_open=True, resizable=True, nesting_level=2)},
        "content": f'<script src="/static/js/graph_view.js"></script><div id="kimi-panel" style="height:100%;overflow:hidden">{panel_html}</div>'})

async def _h_conn_select(request, payload, imr):
    s = await _kg_state(request); s["conn_id"] = payload.get("conn_id",""); await _kg_state(request, s)
    return imr.raw(await _health_html(request))

async def _health_html(request):
    conn = await _active_conn(request)
    if not conn: return '<span style="color:var(--text_muted)">No connection selected.</span>'
    h = await AIM.connections.lightrag_health(conn)
    return f'<span style="color:{"#00ffa2" if h.get("ok") else "#ff5f5f"}">{"&#x25CF; online" if h.get("ok") else "&#x25CF; " + _esc(str(h.get("detail","unreachable")))}</span>'

async def _h_health(request, payload, imr): return imr.raw(await _health_html(request))

async def _h_select_file(request, payload, imr):
    path, is_dir, src = payload.get("path",""), str(payload.get("is_dir","false"))=="true", payload.get("src","kg")
    fm = FM_KG if src == "kg" else FM_COMMON
    s = await _kg_state(request); key = f"selected_{src}"; sel = set(s.get(key, []))
    if is_dir:
        full = fm.resolve(path)
        children = {str(f.relative_to(fm.root)).replace("\\","/") for f in full.rglob("*") if f.is_file()} if full.is_dir() else set()
        sel = sel - children if children and children.issubset(sel) else sel | children
    else: sel.discard(path) if path in sel else sel.add(path)
    s[key] = list(sel); await _kg_state(request, s)
    return imr.oob(_source_tree_html(fm, sel, src), f"kg-tree-{src}", swap="outerHTML")

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
    return HTMLResponse(f"""<div style="padding:.3rem .5rem;display:flex;flex-direction:column;gap:.25rem;font-size:.72rem">
        <a href="{_u('raw', src, path)}" target="_blank" style="color:var(--accent)">&#x1F4C4; View</a>
        <button class="btn-icon" hx-post="/im/in" hx-target="body" hx-swap="none" hx-vals='{json.dumps({"type":"kimi_move_modal","branch":"kimi","lvl":2,"src":src,"path":path})}'>&#x21C4; Move</button>
    </div>""")
