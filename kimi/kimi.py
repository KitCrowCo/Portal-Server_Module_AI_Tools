# /modules/ai_tools/kimi/kimi.py
"""
Kimi - Knowledge Integration Manager Interface. LightRAG ingestion dashboard.
Sub-module of ai_tools. Mounted at /module/ai_tools/kimi.
Sources: the shared _knowledge dir (also used by Tessa/Athena pipelines) and the server-wide _common dir.
All LightRAG protocol logic lives in ai_utils.lightrag_* - this module is the UI, nothing duplicated here.
"""
import json, asyncio
from datetime import datetime
from pathlib import Path
from typing import List
from fastapi import APIRouter, Request, Form, UploadFile, File
from fastapi.responses import HTMLResponse
from modules.ai_tools.ai_utils import *

TOOL_META = {"label": "Kimi", "group": "knowledge", "icon": "&#x1F4DA;", "description": "Knowledge Integration Manager Interface", "singleton": True}

router = APIRouter(redirect_slashes=False)

ENV: dict = {}
_P = "/module/ai_tools/kimi"
SYNC_STATE_FILE = Path("./data/ai_tools/kimi_sync_state.json")

UI = FM_KG = FM_COMMON = BI = IM = TM = cfg = None
_SYNC_TASK = None

def _u(*p): return "/" + "/".join(s.strip("/") for s in [_P.strip("/"), *p] if s)
def _esc(s): return str(s).replace("&","&amp;").replace("<","&lt;").replace(">","&gt;").replace('"',"&quot;")

# --- Init ---

def init_tool(env: dict, prefix: str):
    global ENV, UI, FM_KG, FM_COMMON, BI, cfg, IM, TM, _P, _SYNC_TASK
    ENV = env; _P = prefix.rstrip("/")
    UI = env["templates"].env.globals.get("UI")
    BI = env["tools"]["built_ins"]
    IM = env["InterfaceManager"](nesting_level=2, db_path="ai_tools/kimi_im.db")
    cfg = BI.SettingsPanel("Kimi", [BI.SettingsGroup("general", "General", [
        BI.SettingField("title", "Title", "text", "Kimi"),
        BI.SettingField("preserve_structure", "Preserve folder structure on ingest", "checkbox", True, hint="Uses the file's relative path as its source label instead of just the filename - prevents same-named files in different folders from colliding."),
        BI.SettingField("sync_enabled", "Enable scheduled sync", "checkbox", False),
        BI.SettingField("sync_window", "Sync window (24h, HH:MM-HH:MM)", "text", "02:00-08:00", hint="Watched folders are re-ingested for changed files only while the current time falls in this window."),
    ], json_path="data/ai_tools/kimi_settings.json")])
    FM_KG = BI.FileManager(KG_DIR)
    FM_COMMON = BI.FileManager(COMMON_DIR)
    TM = BI.TabManager(namespace="kimi", tab_bar_id="kimi-tab-bar", content_id="kimi-panel", render_content_fn=_render_panel, intent_prefix="kimi", IM=IM, scope="user", nesting_level=2, allow_new=False, closable=False,
                        empty={"tabs": {"query":{"id":"query","order":0,"label":"Query","icon":"&#x1F50D;"},
                                        "paste":{"id":"paste","order":1,"label":"Paste Text","icon":"&#x1F4DD;"},
                                        "docs":{"id":"docs","order":2,"label":"Documents","icon":"&#x1F4C4;"},
                                        "graph":{"id":"graph","order":3,"label":"Graph","icon":"&#x1F578;"}}, "active":"query"})
    print("[kimi] ready")

def _ensure_sync_task():
    global _SYNC_TASK
    if _SYNC_TASK is None or _SYNC_TASK.done(): _SYNC_TASK = asyncio.create_task(_scheduled_sync_loop())

# --- State ---

async def _kg_state(request, state=None):
    if state is not None:
        await ENV["set_state"](request, state, scope="user", namespace="knowledge")
        return state
    s = await ENV["get_state"](request, scope="user", namespace="knowledge") or {}
    if not s.get("conn_id"):
        conns = list_conns(conn_type="lightrag")
        s["conn_id"] = conns[0]["_id"] if conns else ""  # auto-default to the only/first connection - no blank placeholder requiring a manual re-select
    s.setdefault("selected_kg", []); s.setdefault("selected_common", [])
    return s

# --- Left panel (persistent, not part of tab switching) ---

def _conn_select_html(active_id):
    conns = list_conns(conn_type="lightrag")
    if not conns: return '<div style="font-size:.75rem;color:var(--text_muted)">No LightRAG connections - add one in AI Tools &rarr; Settings &rarr; Connections.</div>'
    active = get_conn(active_id, conn_type="lightrag") or conns[0]
    notes = active.get("values", {}).get("domain_notes", "")
    sel = UI.select("conn_id", [(c["_id"], c.get("display_name", c["_id"])) for c in conns], selected=active["_id"], htmx={"post": _u("conn/select"), "trigger": "change", "target": "#kg-health", "include": "this"})
    notes_html = f'<div style="font-size:.68rem;color:var(--text_muted);margin-top:.2rem">{_esc(notes)}</div>' if notes else ""
    return sel + notes_html

def _source_tree_html(fm, selected, prefix):
    if not fm.root.exists(): return '<div style="color:var(--text_muted);font-size:.75rem;padding:.3rem">No files yet.</div>'
    return UI.tree(items=fm.root, mode="file", selectable=True, selected=set(selected), post_url=_u(f"select/{prefix}"), target=f"#kg-tree-{prefix}", swap="outerHTML")

async def _left_panel(request):
    s = await _kg_state(request)
    return f"""<div style="display:flex;flex-direction:column;height:100%;overflow:hidden">
                    <div style="padding:.5rem;border-bottom:var(--border-thick) solid var(--border)">
                        {UI.field("Knowledge Group", _conn_select_html(s["conn_id"]))}
                        <div id="kg-health" style="font-size:.7rem;color:var(--text_muted)" hx-get="{_u('health')}" hx-trigger="load" hx-swap="innerHTML">checking...</div>
                    </div>
                    <div style="flex:1;overflow-y:auto">
                        <details open style="border-bottom:var(--border-thick) solid var(--border)">
                            <summary style="padding:.3rem .5rem;cursor:pointer;font-size:.72rem;color:var(--text_muted);text-transform:uppercase;list-style:none">&#x1F4DA; Shared Knowledge<button class="btn-icon" style="float:right;font-size:.7rem" hx-get="{_u('upload_modal/kg')}" hx-target="#kg-modal" hx-swap="innerHTML" onclick="event.stopPropagation()">&#x2795;</button></summary>
                            <div id="kg-tree-kg" style="padding:.2rem .4rem">{_source_tree_html(FM_KG, s["selected_kg"], "kg")}</div>
                        </details>
                        <details open style="border-bottom:var(--border-thick) solid var(--border)">
                            <summary style="padding:.3rem .5rem;cursor:pointer;font-size:.72rem;color:var(--text_muted);text-transform:uppercase;list-style:none">&#x1F310; Common (server-wide)<button class="btn-icon" style="float:right;font-size:.7rem" hx-get="{_u('upload_modal/common')}" hx-target="#kg-modal" hx-swap="innerHTML" onclick="event.stopPropagation()">&#x2795;</button></summary>
                            <div id="kg-tree-common" style="padding:.2rem .4rem">{_source_tree_html(FM_COMMON, s["selected_common"], "common")}</div>
                        </details>
                    </div>
                    <div style="padding:.5rem;border-top:var(--border-thick) solid var(--border)">
                        <button class="ui-btn" style="width:100%;justify-content:center" hx-post="{_u('ingest_selected')}" hx-target="#kg-ingest-log" hx-swap="innerHTML">&#x2191; Ingest Selected</button>
                        <div id="kg-ingest-log" style="font-size:.72rem;margin-top:.4rem;max-height:8rem;overflow-y:auto;font-family:var(--font-mono)"></div>
                    </div>
                    <div id="kg-modal"></div>
                </div>"""

# --- Tab panels ---

async def _panel_query(request):
    s = await _kg_state(request)
    multi_opts = "".join(f'<label style="display:flex;align-items:center;gap:.3rem;font-size:.76rem"><input type="checkbox" name="conn_ids" value="{c["_id"]}" {"checked" if c["_id"]==s["conn_id"] else ""}> {_esc(c.get("display_name",c["_id"]))}</label>' for c in list_conns(conn_type="lightrag"))
    return f"""<div style="padding:1rem;height:100%;overflow-y:auto;box-sizing:border-box">
                    <form hx-post="{_u('query')}" hx-target="#kg-query-result" style="display:flex;flex-direction:column;gap:.5rem;margin-bottom:.8rem">
                        <div style="display:flex;gap:.4rem">
                            <input type="text" name="q" placeholder="Ask the knowledge base..." class="module-select" style="flex:1;margin:0">
                            {UI.select("mode", [(m,m) for m in ("hybrid","local","global","naive","mix")], selected="hybrid", style="width:8rem;margin:0")}
                            <input type="number" name="top_k" placeholder="top_k" class="module-select" style="width:6rem;margin:0" title="Optional - only some LightRAG versions support this">
                            <button class="ui-btn">Ask</button>
                        </div>
                        <details><summary style="cursor:pointer;font-size:.74rem;color:var(--text_muted);list-style:none">Compare across groups</summary>
                            <div style="display:flex;flex-direction:column;gap:.2rem;margin-top:.4rem">{multi_opts}</div>
                        </details>
                    </form>
                    <div id="kg-query-result" style="font-size:.85rem;white-space:pre-wrap;display:flex;flex-direction:column;gap:.6rem"></div>
                </div>"""

async def _panel_paste(request):
    return f"""<div style="padding:1rem;height:100%;overflow-y:auto;box-sizing:border-box">
                    <form hx-post="{_u('insert_text')}" hx-target="#kg-ingest-log2" hx-swap="innerHTML" style="display:flex;flex-direction:column;gap:.5rem">
                        {UI.field("Source label (optional)", UI.input("source"))}
                        {UI.field("Text", UI.textarea("text", rows=14))}
                        <button class="ui-btn">Insert</button>
                    </form>
                    <div id="kg-ingest-log2" style="margin-top:.6rem;font-size:.8rem"></div>
                </div>"""

async def _panel_docs(request):
    s = await _kg_state(request)
    conn = get_conn(s["conn_id"], conn_type="lightrag") if s["conn_id"] else None
    body = '<div style="color:var(--text_muted)">No connection selected.</div>'
    if conn:
        r = await lightrag_list_documents(conn)
        if "error" in r: body = f'<div style="color:#ff5f5f">{_esc(r["error"])}</div>'
        else:
            rows = r if isinstance(r, list) else (r.get("documents") or r.get("statuses") or [])
            if isinstance(rows, dict): rows = [v for vs in rows.values() for v in (vs if isinstance(vs, list) else [vs])]
            if rows and isinstance(rows[0], dict):
                headers = list(rows[0].keys())
                body = UI.table(headers, [[str(row.get(h,""))[:80] for h in headers] for row in rows])
            else:
                body = f'<pre style="font-size:.72rem;white-space:pre-wrap">{_esc(json.dumps(r, indent=2))}</pre>'
    return f"""<div style="padding:1rem;height:100%;overflow-y:auto;box-sizing:border-box">
                   {body}
                   <button class="ui-btn" style="margin-top:1rem;color:#ff5f5f" hx-post="{_u('clear_all')}" hx-target="#kimi-panel" hx-confirm="Delete the ENTIRE knowledge graph for this group? This cannot be undone.">Clear Entire Knowledge Group</button>
               </div>"""

async def _panel_graph(request):
    s = await _kg_state(request)
    conn = get_conn(s["conn_id"], conn_type="lightrag") if s["conn_id"] else None
    if not conn: return '<div style="padding:1rem;color:var(--text_muted)">No connection selected.</div>'
    dot = await lightrag_graph_dot(conn)
    if not dot: return '<div style="padding:1rem;color:var(--text_muted)">No graph data available (or this LightRAG version does not expose a listing endpoint at the paths this dashboard tries - check /docs on your instance).</div>'
    return f'<div style="padding:1rem;height:100%;overflow:auto;box-sizing:border-box">{BI.render_graphviz_block(dot, {})}</div>'

async def _render_panel(request, state):
    active = state.get("active", "query")
    if active == "paste": return state, await _panel_paste(request)
    if active == "docs": return state, await _panel_docs(request)
    if active == "graph": return state, await _panel_graph(request)
    return state, await _panel_query(request)

# --- Main route ---

@router.get("")
@router.get("/")
async def root(request: Request):
    _ensure_sync_task()
    state = await TM._load(request)
    state.update({"tabs": {"query":{"id":"query","order":0,"label":"Query","icon":"&#x1F50D;"}, "paste":{"id":"paste","order":1,"label":"Paste Text","icon":"&#x1F4DD;"}, "docs":{"id":"docs","order":2,"label":"Documents","icon":"&#x1F4C4;"}, "graph":{"id":"graph","order":3,"label":"Graph","icon":"&#x1F578;"}}})#, "active":"query"})
    state, panel_html = await _render_panel(request, state)
    tab_bar = await TM.tab_bar_fn(state, "kimi-tab-bar", "kimi", 2, allow_new=False, closable=False)
    left = await _left_panel(request)
    return ENV["templates"].TemplateResponse(name="base.html", request=request, context={
        "request": request, "user": request.state.user, "nesting_level": 2, "shell_id": IM.branch_id,
        "extra_css": BI.MD_BLOCK_CSS,
        "toolbars": {"top": UI.toolbar(side="top", content=tab_bar, size="2.5rem", id="kimi-top", nesting_level=2, start_open=True, locked=True),
                     "left": UI.toolbar(side="left", content=left, size="18rem", overlay=False, start_open=True, resizable=True, nesting_level=2)},
        "content": f'<div id="kimi-panel" style="height:100%;overflow:hidden">{panel_html}</div>'})

# --- Connection / health ---

@router.post("/conn/select", response_class=HTMLResponse)
async def conn_select(request: Request, conn_id: str = Form("")):
    s = await _kg_state(request); s["conn_id"] = conn_id; await _kg_state(request, s)
    return await health(request)

@router.get("/health", response_class=HTMLResponse)
async def health(request: Request):
    s = await _kg_state(request)
    conn = get_conn(s["conn_id"], conn_type="lightrag") if s["conn_id"] else None
    if not conn: return HTMLResponse('<span style="color:var(--text_muted)">No connection selected.</span>')
    h = await lightrag_health(conn)
    return HTMLResponse(f'<span style="color:{"#00ffa2" if h.get("ok") else "#ff5f5f"}">{"&#x25CF; online" if h.get("ok") else "&#x25CF; " + _esc(str(h.get("detail","unreachable")))}</span>')

# --- Source selection / ingestion ---

@router.post("/select/{src}", response_class=HTMLResponse)
async def select_file(src: str, request: Request):
    form = await request.form(); path, is_dir = form.get("path",""), form.get("is_dir","false")=="true"
    fm = FM_KG if src == "kg" else FM_COMMON
    s = await _kg_state(request); key = f"selected_{src}"; sel = set(s.get(key, []))
    if is_dir:
        full = fm.resolve(path)
        children = {str(f.relative_to(fm.root)).replace("\\","/") for f in full.rglob("*") if f.is_file()} if full.is_dir() else set()
        sel = sel - children if children and children.issubset(sel) else sel | children
    else:
        sel.discard(path) if path in sel else sel.add(path)
    s[key] = list(sel); await _kg_state(request, s)
    return HTMLResponse(_source_tree_html(fm, sel, src))

@router.post("/ingest_selected", response_class=HTMLResponse)
async def ingest_selected(request: Request):
    s = await _kg_state(request)
    conn = get_conn(s["conn_id"], conn_type="lightrag") if s["conn_id"] else None
    if not conn: return HTMLResponse('<div style="color:#ff5f5f">No knowledge group selected.</div>')
    preserve = cfg.get_group("general").load().get("preserve_structure", True)
    log = []
    for src, fm in (("kg", FM_KG), ("common", FM_COMMON)):
        for rel in s.get(f"selected_{src}", []):
            try:
                r = await lightrag_insert_file(conn, rel if preserve else Path(rel).name, fm.resolve(rel).read_bytes())
                log.append(f"{rel}: {'ok' if 'error' not in r else r['error'][:80]}")
            except Exception as e: log.append(f"{rel}: error {e}")
    return HTMLResponse("".join(f'<div>{_esc(l)}</div>' for l in log) or '<div style="color:var(--text_muted)">Nothing selected.</div>')

@router.post("/insert_text", response_class=HTMLResponse)
async def insert_text(request: Request, text: str = Form(...), source: str = Form("")):
    s = await _kg_state(request)
    conn = get_conn(s["conn_id"], conn_type="lightrag") if s["conn_id"] else None
    if not conn: return HTMLResponse('<div style="color:#ff5f5f">No knowledge group selected.</div>')
    r = await lightrag_insert_text(conn, text, source)
    return HTMLResponse(f'<div style="color:{"#ff5f5f" if "error" in r else "var(--accent)"}">{_esc(str(r.get("error") or "Inserted"))}</div>')

# --- Query ---

async def _query_one(conn_id, q, mode, top_k):
    conn = get_conn(conn_id, conn_type="lightrag")
    name = conn.get("display_name", conn_id) if conn else conn_id
    if not conn: return name, "connection not found"
    r = await lightrag_query_cached(conn, q, mode, top_k=top_k)
    return name, r.get("response") or r.get("error") or json.dumps(r)

@router.post("/query", response_class=HTMLResponse)
async def query(request: Request, q: str = Form(...), mode: str = Form("hybrid"), top_k: int = Form(None), conn_ids: List[str] = Form(default=[])):
    targets = [c for c in (conn_ids or [(await _kg_state(request))["conn_id"]]) if c]
    if not targets: return HTMLResponse('<div style="color:#ff5f5f">No knowledge group selected.</div>')
    results = await asyncio.gather(*[_query_one(c, q, mode, top_k) for c in targets])
    if len(results) == 1: return HTMLResponse(_esc(results[0][1]))
    return HTMLResponse("".join(f'<div class="glass" style="padding:.6rem"><div style="font-weight:600;font-size:.8rem;margin-bottom:.3rem">{_esc(name)}</div>{_esc(text)}</div>' for name, text in results))

@router.post("/clear_all", response_class=HTMLResponse)
async def clear_all(request: Request):
    s = await _kg_state(request)
    conn = get_conn(s["conn_id"], conn_type="lightrag") if s["conn_id"] else None
    if conn: await lightrag_clear_all(conn)
    return await _panel_docs(request)

# --- Upload ---

@router.get("/upload_modal/{src}", response_class=HTMLResponse)
async def upload_modal(src: str, request: Request):
    fm = FM_KG if src == "kg" else FM_COMMON
    return HTMLResponse(fm.new_item_modal_html(f"kg-upload-{src}", _u(f"upload/{src}"), target_id=f"kg-tree-{src}", swap="outerHTML"))

@router.post("/upload/{src}", response_class=HTMLResponse)
async def upload(src: str, request: Request, parent: str = Form(""), kind: str = Form("file"), name: str = Form(""), upload: List[UploadFile] = File(default=[]), rel_paths: str = Form("[]")):
    fm = FM_KG if src == "kg" else FM_COMMON
    s = await _kg_state(request)
    conn = get_conn(s["conn_id"], conn_type="lightrag") if s["conn_id"] else None
    preserve = cfg.get_group("general").load().get("preserve_structure", True)

    async def _auto_ingest(saved_rels):
        if not conn: return
        for rel in saved_rels:
            try: await lightrag_insert_file(conn, rel if preserve else Path(rel).name, fm.resolve(rel).read_bytes())
            except Exception: pass

    if kind in ("upload", "upload_folder"):
        await fm.save_uploads(parent, upload, json.loads(rel_paths or "[]"), on_complete=_auto_ingest)
    elif kind == "folder":
        if name.strip(): fm.safe_join(parent, name.strip()).mkdir(parents=True, exist_ok=True)
    else:
        if name.strip():
            p = fm.safe_join(parent, name.strip())
            p.parent.mkdir(parents=True, exist_ok=True)
            if not p.exists(): p.write_text("", encoding="utf-8")
    return HTMLResponse(_source_tree_html(fm, s.get(f"selected_{src}", []), src))

# --- Scheduled sync ---

def _load_sync_state() -> dict: return json.loads(SYNC_STATE_FILE.read_text()) if SYNC_STATE_FILE.exists() else {}
def _save_sync_state(s: dict): SYNC_STATE_FILE.parent.mkdir(parents=True, exist_ok=True); SYNC_STATE_FILE.write_text(json.dumps(s, indent=2))

def _in_window(window: str) -> bool:
    try:
        start_s, end_s = window.split("-")
        now = datetime.now().time()
        start, end = datetime.strptime(start_s.strip(), "%H:%M").time(), datetime.strptime(end_s.strip(), "%H:%M").time()
        return (start <= now <= end) if start <= end else (now >= start or now <= end)
    except Exception: return False

async def _run_sync_pass():
    state = _load_sync_state()
    conns = list_conns(conn_type="lightrag")
    if not conns: return
    conn = conns[0]  # scheduled sync targets the default connection - per-connection schedules are future work
    preserve = cfg.get_group("general").load().get("preserve_structure", True)
    for fm, prefix in ((FM_KG, "kg"), (FM_COMMON, "common")):
        for f in fm.root.rglob("*"):
            if not f.is_file(): continue
            rel = str(f.relative_to(fm.root)).replace("\\","/")
            key = f"{prefix}:{rel}"
            mtime = f.stat().st_mtime
            if state.get(key, 0) >= mtime: continue
            try:
                await lightrag_insert_file(conn, rel if preserve else f.name, f.read_bytes())
                state[key] = mtime
            except Exception as e: print(f"[kimi] sync failed for {rel}: {e}")
    _save_sync_state(state)

async def _scheduled_sync_loop():
    """Default-off (sync_enabled checkbox) - checks every 15min; does real work only inside the configured window."""
    while True:
        try:
            settings = cfg.get_group("general").load()
            if settings.get("sync_enabled") and _in_window(settings.get("sync_window", "")): await _run_sync_pass()
        except Exception as e: print(f"[kimi] sync loop error: {e}")
        await asyncio.sleep(900)

def right_panel() -> str: return """<div class="ait-rp"><div class="ait-rp-hd">Kimi</div><div style="font-size:.72rem;color:var(--text_muted);padding:.3rem">Knowledge Integration Manager - pick a knowledge group and ingest sources from the left panel.</div></div>"""