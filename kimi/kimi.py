# /modules/ai_tools/kimi/kimi.py
"""
Kimi - Knowledge Integration Manager Interface. LightRAG ingestion dashboard.
Sub-module of ai_tools. Mounted at /module/ai_tools/kimi.
Sources: the shared _knowledge dir (also used by Tessa/Athena pipelines) and the server-wide _common dir.
All LightRAG protocol logic lives in AIM.connections.lightrag_* - this module is the UI, nothing duplicated here.
"""
import json, asyncio, docx
from datetime import datetime
from pathlib import Path
from typing import List
from fastapi import APIRouter, Request, Form, UploadFile, File
from fastapi.responses import HTMLResponse

TOOL_META = {"label": "Kimi", "group": "knowledge", "icon": "&#x1F4DA;", "description": "Knowledge Integration Manager Interface", "singleton": True}

router = APIRouter(redirect_slashes=False)

ENV: dict = {}
_P = "/module/ai_tools/kimi"
SYNC_STATE_FILE = Path("./data/ai_tools/kimi_sync_state.json")
KG_DIR = Path("./data/ai_tools/_knowledge")
COMMON_DIR = Path("./data/_common")
SOURCESETS_PATH = Path("./data/ai_tools/kimi_sourcesets.json")

UI = FM_KG = FM_COMMON = BI = IM = TM = AIM = cfg = None
_SYNC_TASK = None

def _u(*p): return "/" + "/".join(s.strip("/") for s in [_P.strip("/"), *p] if s)
def _esc(s): return str(s).replace("&","&amp;").replace("<","&lt;").replace(">","&gt;").replace('"',"&quot;")

# --- Init ---

def init_tool(env: dict, prefix: str):
    global ENV, UI, FM_KG, FM_COMMON, BI, cfg, IM, TM, AIM, _P, _SYNC_TASK
    ENV = env; _P = prefix.rstrip("/")
    UI = env["templates"].env.globals.get("UI")
    BI = env["tools"]["built_ins"]
    IM = env["InterfaceManager"](nesting_level=2, db_path="ai_tools/kimi_im.db")
    AIM = ENV["tools"]["ai_manager"]
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
    IM.scripts["kimi_graph_limit"] = [_h_graph_limit]
    IM.scripts["kimi_graph_limit"] = [_h_graph_limit]
    IM.scripts.update({"kimi_move_modal": [_h_move_modal], "kimi_move": [_h_move]})
    IM.scripts.update({"kimi_conn_select": [_h_conn_select], "kimi_health": [_h_health], "kimi_select_file": [_h_select_file], "kimi_ingest_selected": [_h_ingest_selected], "kimi_insert_text": [_h_insert_text], "kimi_query": [_h_query], "kimi_clear_all": [_h_clear_all], "kimi_doc_delete": [_h_doc_delete], "kimi_upload_modal": [_h_upload_modal], "kimi_sync_now": [_h_sync_now], "kimi_upload_kg": [lambda r,p,i: _h_upload(r,p,i,"kg")], "kimi_upload_common": [lambda r,p,i: _h_upload(r,p,i,"common")], "kimi_export_docx": [_h_export_docx], "kimi_chunk_selected": [_h_chunk_selected], "kimi_combine_selected": [_h_combine_selected]})
    IM.scripts.update({"kimi_sourceset_save":[_h_sourceset_save],"kimi_sourceset_load":[_h_sourceset_load],"kimi_sourceset_delete":[_h_sourceset_delete]})
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
        conns = AIM.connections.list_conns(conn_type="lightrag")
        s["conn_id"] = conns[0]["_id"] if conns else ""
    s.setdefault("selected_kg", []); s.setdefault("selected_common", [])
    s.setdefault("last_query", ""); s.setdefault("last_mode", "hybrid"); s.setdefault("last_result", "")
    return s

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
    src = payload.get("combine_src", "kg")
    fm = FM_KG if src == "kg" else FM_COMMON
    rels = sorted(s.get(f"selected_{src}", []))
    if not rels: return imr.oob('<div style="color:var(--text_muted)">Select files from Knowledge or Common to combine.</div>', "kg-ingest-log")
    parts = []
    for rel in rels:
        try: parts.append(f"# {rel}\n\n{fm.resolve(rel).read_text(encoding='utf-8', errors='ignore')}")
        except Exception as e: parts.append(f"# {rel}\n\n[read error: {e}]")
    dest = out_name if out_name.endswith(".md") else f"{out_name}.md"
    fm.write(dest, "\n\n---\n\n".join(parts))
    imr.oob(f'<div style="color:var(--accent)">Combined {len(rels)} file(s) into {_esc(dest)}</div>', "kg-ingest-log")
    imr.oob(_source_tree_html(fm, s.get(f"selected_{src}", []), src), f"kg-tree-{src}", swap="outerHTML")
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

# --- Left panel (persistent, not part of tab switching) ---

def _conn_select_html(active_id):
    conns = AIM.connections.list_conns(conn_type="lightrag")
    if not conns: return '<div style="font-size:.7rem;color:var(--text_muted)">No LightRAG connections - add one in AI Tools &rarr; Settings &rarr; Connections.</div>'
    active = AIM.connections.get_conn(active_id, conn_type="lightrag") or conns[0]
    notes = active.get("values", {}).get("domain_notes", "")
    sel = UI.select("conn_id", [(c["_id"], c.get("display_name", c["_id"])) for c in conns], selected=active["_id"], htmx={"post":"/im/in","trigger":"change","target":"#kg-health", "vals": json.dumps({"type": "kimi_conn_select", "branch": "kimi", "lvl": 2}), "include":"this"})
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
                        <div id="kg-health" style="font-size:.7rem;color:var(--text_muted)" hx-post="/im/in" hx-vals='{{"type":"kimi_health","branch":"kimi","lvl":2}}' hx-trigger="load" hx-swap="innerHTML">checking...</div>
                    </div>
                    <div style="flex:1;overflow-y:auto">
                        <details open style="border-bottom:var(--border-thick) solid var(--border)">
                            <summary style="padding:.3rem .5rem;cursor:pointer;font-size:.7rem;color:var(--text_muted);text-transform:uppercase;list-style:none">&#x1F4DA; Shared Knowledge<button class="btn-icon" style="float:right;font-size:.7rem" hx-post="/im/in" hx-target="body" hx-swap="none" hx-vals='{{"type":"kimi_upload_modal","branch":"kimi","lvl":2,"src":"kg"}}' onclick="event.stopPropagation()">&#x2795;</button></summary>
                            <div id="kg-tree-kg" style="padding:.2rem .4rem">{_source_tree_html(FM_KG, s["selected_kg"], "kg")}</div>
                        </details>
                        <details open style="border-bottom:var(--border-thick) solid var(--border)">
                            <summary style="padding:.3rem .5rem;cursor:pointer;font-size:.7rem;color:var(--text_muted);text-transform:uppercase;list-style:none">&#x1F310; Common (server-wide)<button class="btn-icon" style="float:right;font-size:.7rem" hx-post="/im/in" hx-target="body" hx-swap="none" hx-vals='{{"type":"kimi_upload_modal","branch":"kimi","lvl":2,"src":"common"}}' onclick="event.stopPropagation()">&#x2795;</button></summary>
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
                    </div>
                    <div style="padding:.5rem;border-top:var(--border-thick) solid var(--border)">
                        <button class="ui-btn" style="width:100%;justify-content:center" hx-post="/im/in" hx-target="body" hx-swap="none" hx-indicator="#kg-ingest-spin" hx-vals='{{"type":"kimi_ingest_selected","branch":"kimi","lvl":2}}'>&#x2191; Ingest Selected <span id="kg-ingest-spin" class="htmx-indicator spin" title="Working...">&#x25CC;</span></button>
                        <div id="kg-ingest-log" style="font-size:.7rem;margin-top:.4rem;max-height:8rem;overflow-y:auto;font-family:var(--font-mono)"></div>
                        <button class="ui-btn" style="width:100%;margin-top:.3rem" hx-post="/im/in" hx-target="body" hx-swap="none" hx-indicator="#kg-sync-spin" hx-vals='{{"type":"kimi_sync_now","branch":"kimi","lvl":2}}'>&#x21BB; Sync Now <span id="kg-sync-spin" class="htmx-indicator spin" title="Working...">&#x25CC;</span></button>
                        <div id="kg-sync-status" style="font-size:.7rem;margin-top:.2rem"></div>
                    </div>
                    <div id="kg-modal"></div>
                </div>"""
# --- Tab panels ---

def build_field(k, spec):
    label = _esc(spec.get("label", k))
    if spec.get("type") == "boolean":
        hint = _esc(spec.get("hint", ""))
        checked = " checked" if spec.get("default") else ""
        return f'<label style="display:flex;align-items:center;gap:.3rem;font-size:.7rem" title="{hint}"><input type="checkbox" name="opt_{k}" value="1"{checked}> {label}</label>'
    else:
        val = f' value="{spec["default"]}"' if "default" in spec else ""
        return f'<label style="font-size:.7rem;color:var(--text_muted)">{label}<input type="number" name="opt_{k}" class="module-select" style="width:5rem"{val}></label>'

async def _panel_query(request):
    s = await _kg_state(request)
    conn = AIM.connections.get_conn(s["conn_id"], conn_type="lightrag") if s["conn_id"] else None
    schema = AIM.connections.lightrag_query_options_schema(conn) if conn else {}
    def build_field(k, spec):
        label = _esc(spec.get("label", k))
        if spec.get("type") == "boolean":
            hint = _esc(spec.get("hint", ""))
            checked = " checked" if spec.get("default") else ""
            return f'<label style="display:flex;align-items:center;gap:.3rem;font-size:.72rem" title="{hint}"><input type="checkbox" name="opt_{k}" value="1"{checked}> {label}</label>'
        val = f' value="{spec["default"]}"' if "default" in spec else ""
        return f'<label style="font-size:.72rem;color:var(--text_muted)">{label}<input type="number" name="opt_{k}" class="module-select" style="width:5rem"{val}></label>'
    opt_fields = "".join(build_field(k, spec) for k, spec in schema.items())
    multi_opts = "".join(f'<label style="display:flex;align-items:center;gap:.3rem;font-size:.76rem"><input type="checkbox" name="conn_ids" value="{c["_id"]}" {"checked" if c["_id"]==s["conn_id"] else ""}> {_esc(c.get("display_name",c["_id"]))}</label>' for c in AIM.connections.list_conns(conn_type="lightrag"))
    return f"""<div style="padding:1rem;height:100%;overflow-y:auto;box-sizing:border-box">
                    <form hx-post="/im/in" hx-target="body" hx-swap="none" hx-indicator="#kg-query-spin" style="display:flex;flex-direction:column;gap:.5rem;margin-bottom:.8rem">
                        <input type="hidden" name="type" value="kimi_query"><input type="hidden" name="branch" value="kimi"><input type="hidden" name="lvl" value="2">
                        <div style="display:flex;gap:.4rem;flex-wrap:wrap;align-items:center">
                            <input type="text" name="q" value="{_esc(s.get('last_query',''))}" placeholder="Ask the knowledge base..." class="module-select" style="flex:1;margin:0;min-width:14rem">
                            {UI.select("mode", [(m,m) for m in ("hybrid","local","global","naive","mix")], selected=s.get("last_mode","hybrid"), style="width:8rem;margin:0")}
                            <button class="ui-btn">Ask</button>
                            <span id="kg-query-spin" class="htmx-indicator spin" style="font-size:.9rem" title="Working...">&#x25CC;</span>
                        </div>
                        <div style="display:flex;gap:.6rem;flex-wrap:wrap;align-items:center">{opt_fields}</div>
                        <details><summary style="cursor:pointer;font-size:.74rem;color:var(--text_muted);list-style:none">Compare across groups</summary>
                            <div style="display:flex;flex-direction:column;gap:.2rem;margin-top:.4rem">{multi_opts}</div>
                        </details>
                    </form>
                    <div id="kg-query-result" style="font-size:.85rem;white-space:pre-wrap;display:flex;flex-direction:column;gap:.6rem">{s.get("last_result","")}</div>
                </div>"""

async def _panel_paste(request):
    return f"""<div style="padding:1rem;height:100%;overflow-y:auto;box-sizing:border-box">
                    <form hx-post="/im/in" hx-target="body" hx-swap="none" style="display:flex;flex-direction:column;gap:.5rem">
                        <input type="hidden" name="type" value="kimi_insert_text"><input type="hidden" name="branch" value="kimi"><input type="hidden" name="lvl" value="2">
                        {UI.field("Source label (optional)", UI.input("source"))}
                        {UI.field("Text", UI.textarea("text", rows=14))}
                        <button class="ui-btn" hx-indicator="#kg-insert-spin">Insert <span id="kg-insert-spin" class="htmx-indicator spin">&#x25CC;</span></button>
                    </form>
                    <div id="kg-ingest-log2" style="margin-top:.6rem;font-size:.8rem"></div>
                </div>"""

async def _panel_docs(request):
    s = await _kg_state(request)
    conn = AIM.connections.get_conn(s["conn_id"], conn_type="lightrag") if s["conn_id"] else None
    body = '<div style="color:var(--text_muted)">No connection selected.</div>'
    if conn:
        r = await AIM.connections.lightrag_list_documents(conn)
        if "error" in r: body = f'<div style="color:#ff5f5f">{_esc(r["error"])}</div>'
        else:
            rows = r if isinstance(r, list) else (r.get("documents") or r.get("statuses") or [])
            if isinstance(rows, dict): rows = [v for vs in rows.values() for v in (vs if isinstance(vs, list) else [vs])]
            if rows and isinstance(rows[0], dict):
                doc_id_field = next((f for f in ("id","doc_id","document_id") if f in rows[0]), None)
                headers = list(rows[0].keys())
                priority = [h for h in headers if any(k in h.lower() for k in ("file","path","name","source"))]
                headers = priority + [h for h in headers if h not in priority]
                th = "".join(f'<th style="padding:.32rem .6rem;border-bottom:var(--border-thick) solid var(--border);text-align:left;">{_esc(h)}</th>' for h in headers) + ("<th></th>" if doc_id_field else "")
                table_rows = "".join("<tr>" + "".join(f"""<td style="padding:.2rem .4rem;border-bottom:var(--border-thick) solid var(--border);">{_esc(str(row.get(h,""))[:80])}</td>""" for h in headers) + (f"""<td style="padding:.2rem .4rem;border-bottom:var(--border-thick) solid var(--border)"><button class="cm-qbtn" style="color:#ff5f5f" hx-post="/im/in" hx-target="body" hx-swap="none" hx-indicator="closest button" hx-vals='{{"type":"kimi_doc_delete","branch":"kimi","lvl":2,"doc_id":"{_esc(str(row.get(doc_id_field,"")))}"}}' hx-confirm="Delete this document from the knowledge base?">&#x2715;</button></td>""" if doc_id_field else "") + "</tr>" for row in rows)
                body = f'<div style="overflow-x:auto"><table style="width:100%;border-collapse:collapse;font-size:.8rem"><thead><tr>{th}</tr></thead><tbody>{table_rows}</tbody></table></div>'
            else:
                body = f'<pre style="font-size:.72rem;white-space:pre-wrap">{_esc(json.dumps(r, indent=2))}</pre>'
    return f"""<div style="padding:1rem;height:100%;overflow-y:auto;box-sizing:border-box">
                   {body}
                   <button class="ui-btn" style="margin-top:1rem;color:#ff5f5f" hx-post="/im/in" hx-target="body" hx-swap="none" hx-indicator="#kg-clear-spin" hx-vals='{{"type":"kimi_clear_all","branch":"kimi","lvl":2}}' hx-confirm="Delete the ENTIRE knowledge graph for this group? This cannot be undone.">Clear Entire Knowledge Group <span id="kg-clear-spin" class="htmx-indicator spin">&#x25CC;</span></button>
               </div>"""

async def _panel_graph(request):
    s = await _kg_state(request)
    conn = AIM.connections.get_conn(s["conn_id"], conn_type="lightrag") if s["conn_id"] else None
    if not conn: return '<div style="padding:1rem;color:var(--text_muted)">No connection selected.</div>'
    limit = int((await ENV["get_state"](request, scope="user", namespace="knowledge", key="graph_limit")) or 1000)
    try: dot = await AIM.connections.lightrag_graph_dot(conn, limit=limit)
    except Exception as e: return f'<div style="padding:1rem;color:#ff5f5f">Graph fetch failed: {_esc(str(e))}</div>'
    body = dot and BI.render_graphviz_block(dot, {}) or '<div style="padding:1rem;color:var(--text_muted)">No graph data available.</div>'
    return f"""<div style="padding:.5rem 1rem;display:flex;align-items:center;gap:.4rem;border-bottom:var(--border-thick) solid var(--border)">
                   <label style="font-size:.75rem;color:var(--text_muted)">Max nodes<input type="number" name="value" value="{limit}" min="1" class="module-select" style="width:6rem" hx-post="/im/in" hx-vals='{{"type":"kimi_graph_limit"}}' hx-trigger="change" hx-include="this" hx-target="#kimi-panel" hx-swap="innerHTML"></label>
               </div>
               <div style="padding:1rem;height:calc(100% - 3rem);overflow:auto;box-sizing:border-box">{body}</div>"""

async def _h_graph_limit(request, payload, imr):
    await ENV["set_state"](request, int(payload.get("value", 5000) or 5000), scope="user", namespace="knowledge", key="graph_limit")
    return imr.raw(f'<div id="kimi-panel" style="height:100%;overflow:hidden">{await _panel_graph(request)}</div>')

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

async def _h_conn_select(request, payload, imr):
    s = await _kg_state(request); s["conn_id"] = payload.get("conn_id",""); await _kg_state(request, s)
    return imr.raw(await _health_html(request))

async def _health_html(request):
    s = await _kg_state(request)
    conn = AIM.connections.get_conn(s["conn_id"], conn_type="lightrag") if s["conn_id"] else None
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

async def _h_ingest_selected(request, payload, imr):
    s = await _kg_state(request)
    conn = AIM.connections.get_conn(s["conn_id"], conn_type="lightrag") if s["conn_id"] else None
    if not conn: return imr.oob('<div style="color:#ff5f5f">No knowledge group selected.</div>', "kg-ingest-log")
    preserve = cfg.get_group("general").load().get("preserve_structure", True)
    log = []
    for src, fm in (("kg", FM_KG), ("common", FM_COMMON)):
        for rel in s.get(f"selected_{src}", []):
            try:
                r = await AIM.connections.lightrag_insert_file(conn, rel if preserve else Path(rel).name, fm.resolve(rel).read_bytes())
                log.append(f"{rel}: {'ok' if 'error' not in r else r['error'][:80]}")
            except Exception as e: log.append(f"{rel}: error {e}")
    return imr.oob("".join(f'<div>{_esc(l)}</div>' for l in log) or '<div style="color:var(--text_muted)">Nothing selected.</div>', "kg-ingest-log")

async def _h_insert_text(request, payload, imr):
    s = await _kg_state(request)
    conn = AIM.connections.get_conn(s["conn_id"], conn_type="lightrag") if s["conn_id"] else None
    if not conn: return imr.oob('<div style="color:#ff5f5f">No knowledge group selected.</div>', "kg-ingest-log2")
    r = await AIM.connections.lightrag_insert_text(conn, payload.get("text",""), payload.get("source",""))
    return imr.oob(f'<div style="color:{"#ff5f5f" if "error" in r else "var(--accent)"}">{_esc(str(r.get("error") or "Inserted"))}</div>', "kg-ingest-log2")

async def _query_one(conn_id, q, mode, extra):
    conn = AIM.connections.get_conn(conn_id, conn_type="lightrag")
    name = conn.get("display_name", conn_id) if conn else conn_id
    if not conn: return name, "connection not found"
    r = await AIM.connections.lightrag_query_cached(conn, q, mode, **(extra or {}))
    return name, r.get("response") or r.get("error") or json.dumps(r)

async def _h_query(request, payload, imr):
    q, mode = payload.get("q","").strip(), payload.get("mode","hybrid")
    if not q: return imr.oob('<div style="color:#ff5f5f">Enter a question.</div>', "kg-query-result")
    extra = {}
    for k, v in payload.items():
        if not k.startswith("opt_") or v in ("", None): continue
        extra[k[4:]] = True if v in ("1","true","on") else (int(v) if str(v).strip().lstrip("-").isdigit() else v)
    conn_ids = payload.get("conn_ids", [])
    if isinstance(conn_ids, str): conn_ids = [conn_ids] if conn_ids else []
    s = await _kg_state(request)
    targets = [c for c in (conn_ids or [s["conn_id"]]) if c]
    if not targets: return imr.oob('<div style="color:#ff5f5f">No knowledge group selected.</div>', "kg-query-result")
    results = await asyncio.gather(*[_query_one(c, q, mode, extra) for c in targets])
    html = _esc(results[0][1]) if len(results)==1 else "".join(f'<div class="glass" style="padding:.6rem"><div style="font-weight:600;font-size:.8rem;margin-bottom:.3rem">{_esc(name)}</div>{_esc(text)}</div>' for name, text in results)
    s["last_query"], s["last_mode"], s["last_result"] = q, mode, html
    await _kg_state(request, s)
    return imr.oob(html, "kg-query-result")

async def _h_clear_all(request, payload, imr):
    s = await _kg_state(request)
    conn = AIM.connections.get_conn(s["conn_id"], conn_type="lightrag") if s["conn_id"] else None
    if conn: await AIM.connections.lightrag_clear_all(conn)
    return imr.raw(f'<div id="kimi-panel" style="height:100%;overflow:hidden">{await _panel_docs(request)}</div>')

async def _h_upload_modal(request, payload, imr):
    src = payload.get("src","kg")
    fm = FM_KG if src == "kg" else FM_COMMON
    return imr.oob(fm.new_item_modal_html(f"kg-upload-{src}", target_id=f"kg-tree-{src}", swap="outerHTML", intent_type=f"kimi_upload_{src}", branch="kimi", lvl=2), "kg-modal")

async def _h_upload(request, payload, imr, src):
    fm = FM_KG if src == "kg" else FM_COMMON
    s = await _kg_state(request)
    conn = AIM.connections.get_conn(s["conn_id"], conn_type="lightrag") if s["conn_id"] else None
    preserve = cfg.get_group("general").load().get("preserve_structure", True)
    parent, kind, name = payload.get("parent",""), payload.get("kind","file"), payload.get("name","")
    upload_raw = payload.get("upload")
    files = upload_raw if isinstance(upload_raw, list) else ([upload_raw] if upload_raw else [])
    try: rel_paths = json.loads(payload.get("rel_paths","[]") or "[]")
    except Exception: rel_paths = []

    async def _auto_ingest(saved_rels):
        if not conn: return
        for rel in saved_rels:
            try: await AIM.connections.lightrag_insert_file(conn, rel if preserve else Path(rel).name, fm.resolve(rel).read_bytes())
            except Exception: pass

    if kind in ("upload", "upload_folder") and files:
        await fm.save_uploads(parent, files, rel_paths, on_complete=_auto_ingest)
    elif kind == "folder" and name.strip():
        fm.safe_join(parent, name.strip()).mkdir(parents=True, exist_ok=True)
    elif kind == "file" and name.strip():
        p = fm.safe_join(parent, name.strip())
        p.parent.mkdir(parents=True, exist_ok=True)
        if not p.exists(): p.write_text("", encoding="utf-8")
    return imr.oob(_source_tree_html(fm, s.get(f"selected_{src}", []), src), f"kg-tree-{src}", swap="outerHTML")

async def _h_doc_delete(request, payload, imr):
    s = await _kg_state(request)
    conn = AIM.connections.get_conn(s["conn_id"], conn_type="lightrag") if s["conn_id"] else None
    if conn: await AIM.connections.lightrag_delete_document(conn, payload.get("doc_id",""))
    return imr.raw(f'<div id="kimi-panel" style="height:100%;overflow:hidden">{await _panel_docs(request)}</div>')

async def _h_sync_now(request, payload, imr):
    asyncio.create_task(_run_sync_pass())
    return imr.oob('<span style="color:var(--accent);font-size:.7rem">&#x2713; Sync started (running in background)</span>', "kg-sync-status")

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
    conns = AIM.connections.list_conns(conn_type="lightrag")
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
                await AIM.connections.lightrag_insert_file(conn, rel if preserve else f.name, f.read_bytes())
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

async def _h_move_modal(request, payload, imr):
    src = payload.get("src","kg")
    fm = FM_KG if src == "kg" else FM_COMMON
    path = payload.get("path","")
    return imr.oob(BI.move_modal_html(f"kimi-move-{src}", f"/im/in", fm.folder_picker_html(), path, target_id=f"kg-tree-{src}", swap="outerHTML"), "kg-modal")

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