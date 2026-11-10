"""Asio - simulates AI work on a pool of machines: what fits where, how long it takes, what holds it back, what it costs, how sure that is, and which setup is best for a goal. An ai_tools tool: this repository is the tool folder, placed at modules/ai_tools/asio/ and mounted at /module/ai_tools/asio.
This file is the entry the AI Tools module loads: page, tabs and intent registry. The simulator is asio_core (beside this file), the page parts are asio_ui (beside it too), the optimizer is Athene (the portal tool at tools/athene/, reached through ENV["tools"]["athene"]).
Data: data/ai_tools/asio/catalog holds your catalog files (each overrides or adds to the built-in catalog); data/ai_tools/asio/workspace/<user>/ holds each user's scenarios, versions and results."""
import sys
from pathlib import Path
from fastapi import APIRouter, Request
from fastapi.responses import HTMLResponse

HERE = str(Path(__file__).resolve().parent)
if HERE not in sys.path: sys.path.insert(0, HERE)
from asio_ui import base as B, build, run_ui, studies, charts, catalog_ui, ref, heat_ui, lookup

TOOL_META = {"label": "Asio", "group": "model_tools", "icon": "&#x1F989;", "description": "Simulate AI work on your machines: what fits, bottlenecks, spread, cost, goal seeking", "singleton": True}
router = APIRouter(redirect_slashes=False)
UI, TM = None, None
TABS = {"build": {"id": "build", "order": 0, "label": "Build", "icon": "&#x1F527;"}, "run": {"id": "run", "order": 1, "label": "Run", "icon": "&#x25B6;"}, "studies": {"id": "studies", "order": 2, "label": "Studies", "icon": "&#x1F3AF;"}, "charts": {"id": "charts", "order": 3, "label": "Charts", "icon": "&#x1F4C8;"}, "catalog": {"id": "catalog", "order": 4, "label": "Catalog", "icon": "&#x1F4D6;"}}

# --- Tabs ---

async def _render(request, state):
    """The active tab's content; each tab reads the user's workspace, so it shows what was there when it was left."""
    ws, active = B.ws(request), state.get("active", "build")
    try: B.libs()
    except Exception as e: return state, B.err(f"Asio cannot load its libraries: {type(e).__name__}: {e}")
    if active == "run": return state, f"""<div id="asio-runtab">{run_ui.run_page(ws)}</div>"""
    if active == "studies": return state, f"""<div id="asio-studies">{studies.page(ws)}</div>"""
    if active == "charts": return state, f"""<div id="asio-charts">{charts.page(ws)}</div>"""
    if active == "catalog": return state, f"""<div id="asio-catalog">{catalog_ui.page(ws)}</div>"""
    name = ws.ui().get("scenario")
    return state, f"""<div id="asio-build">{build.page(ws, name if name and ws.exists(name) else (ws.names() or [None])[0])}</div>"""

async def _focus(request, imr, tab: str):
    state = await TM._load(request)
    state["active"] = tab
    return await TM._push(request, state, imr)

async def _state(request) -> dict:
    """The saved tab state, with tabs added since it was saved."""
    state = await TM._load(request)
    if any(k not in state["tabs"] or state["tabs"][k].get("order") != v["order"] for k, v in TABS.items()):
        state["tabs"] = {k: {**v, **state["tabs"].get(k, {}), "order": v["order"]} for k, v in TABS.items()}
        await TM._save(request, state)
    return state

# --- Routes ---

@router.get("")
@router.get("/")
async def root(request: Request):
    state = await _state(request)
    state, panel = await _render(request, state)
    bar = await TM.tab_bar_fn(state, "asio-tab-bar", "asio", 2, allow_new=False, closable=False)
    return B.ENV["templates"].TemplateResponse(name="base.html", request=request, context={"request": request, "user": request.state.user, "nesting_level": 2, "shell_id": B.IM.branch_id,
        "toolbars": {"top": UI.toolbar(side="top", content=bar, size="2.5rem", id="asio-top", nesting_level=2, start_open=True, locked=True)},
        "content": f"""<div class="asio-root"><div id="asio-msg"></div><div id="asio-panel" class="asio-panel">{panel}</div></div>""", "extra_css": CSS, "extra_script": ""})

@router.get("/ref", response_class=HTMLResponse)
async def ref_route(request: Request): return HTMLResponse(ref.panel(B.ws(request)))

def right_panel() -> str: return f"""<div class="ait-rp asio-ref" id="asio-ref" hx-get="{B.P}/ref" hx-trigger="load" hx-swap="innerHTML"><div class="ait-rp-hd">Asio reference</div></div>"""

# --- Init ---

def init_tool(env, prefix):
    global UI, TM
    B.ENV, B.P = env, prefix.rstrip("/")
    UI = env["templates"].env.globals.get("UI")
    B.DATA_DIR.mkdir(parents=True, exist_ok=True)
    B.IM = env["InterfaceManager"](nesting_level=2, db_path=f"""{env["data_dir"]}/asio/im_registry.db""")
    TM = env["tools"]["built_ins"].TabManager(namespace="asio", tab_bar_id="asio-tab-bar", content_id="asio-panel", render_content_fn=_render, intent_prefix="asio", IM=B.IM, scope="user", nesting_level=2, allow_new=False, closable=False, empty={"tabs": dict(TABS), "active": "build"})
    B.FOCUS = _focus
    for part in (build, run_ui, studies, charts, catalog_ui, ref, heat_ui, lookup): B.IM.scripts.update({k: [v] for k, v in part.INTENTS.items()})
    print("[asio] ready")

CSS = """
.asio-root{height:100%;display:flex;flex-direction:column}.asio-panel{flex:1;min-height:0;overflow-y:auto}.asio-pad{padding:.6rem .8rem}
.asio-sec{border:var(--border-thick) solid var(--border);border-radius:var(--radius);margin:.45rem 0;background:var(--surface)}.asio-sec>summary{cursor:pointer;padding:.35rem .6rem;font-size:.78rem;font-weight:700;color:var(--text_muted);text-transform:uppercase;letter-spacing:.04em}.asio-sec-body{padding:.2rem .6rem .6rem}
.asio-card{border-bottom:var(--border-thick) solid var(--border);padding:.35rem 0}.asio-card:last-child{border-bottom:none}.asio-line,.asio-dim{font-size:.72rem;color:var(--text_muted)}.asio-line{margin-top:.15rem}
.asio-in{background:var(--bg);color:var(--text);border:var(--border-thick) solid var(--border);border-radius:calc(var(--radius) / 2);padding:.15rem .3rem;font-size:.78rem;font-family:inherit;box-sizing:border-box;max-width:100%}
.asio-w3{width:3rem}.asio-w4{width:4rem}.asio-w5{width:5rem}.asio-w6{width:6rem}.asio-w8{width:8rem}.asio-w10{width:10rem}.asio-w12{width:12rem}.asio-w14{width:14rem}.asio-w16{width:16rem}.asio-w20{width:20rem}.asio-w30{width:30rem}
.asio-json{width:100%;min-height:14rem;font-family:var(--font-mono);font-size:.72rem;white-space:pre}
.asio-f{display:inline-flex;flex-direction:column;font-size:.66rem;color:var(--text_muted);gap:.05rem}.asio-chk{display:inline-flex;align-items:center;gap:.2rem;font-size:.74rem}.asio-multi{display:inline-flex;flex-wrap:wrap;gap:.1rem .5rem;margin:0}.asio-wrap{flex-wrap:wrap;gap:.3rem .6rem}
.asio-btn{background:var(--accent_dim);color:var(--accent);border:var(--border-thick) solid var(--accent);border-radius:calc(var(--radius) / 2);padding:.15rem .5rem;font-size:.74rem;font-weight:600;cursor:pointer;white-space:nowrap}.asio-btn:hover{background:var(--accent);color:#000}
.asio-danger{color:#ff6b6b;border-color:#ff6b6b;background:none}.asio-x{padding:.1rem .35rem}.asio-add{margin-top:.35rem}.asio-scbar{margin-bottom:.3rem}.asio-on{border-color:var(--accent)!important;background:var(--accent_dim)!important}
.asio-tscroll{overflow-x:auto}.asio-tbl{border-collapse:collapse;font-size:.74rem;margin:.3rem 0 .7rem;width:100%}.asio-tbl th,.asio-tbl td{border-bottom:1px solid var(--border);padding:.18rem .4rem;text-align:left;vertical-align:top}.asio-tbl th{color:var(--text_muted);font-weight:600}
.asio-cmp td:first-child{color:var(--text_muted);white-space:nowrap}.asio-best{color:var(--accent);font-weight:700}
.asio-v .asio-tier,.asio-unit{font-size:.66rem;color:var(--text_muted)}.asio-tag{font-size:.64rem;color:var(--accent)}
.asio-fig{margin:.5rem 0;overflow-x:auto}.asio-h{font-size:.72rem;font-weight:700;color:var(--text_muted);text-transform:uppercase;margin:.6rem 0 .2rem}
.asio-note{border-left:3px solid var(--accent);padding:.35rem .55rem;margin:.35rem 0;font-size:.78rem;background:var(--bg)}.asio-err{border-left:3px solid #ff6b6b;padding:.35rem .55rem;margin:.35rem 0;font-size:.78rem;background:var(--bg)}
.asio-fit{display:flex;flex-wrap:wrap;gap:.3rem .9rem;align-items:center;font-size:.72rem;margin:.3rem 0}.asio-ok{color:var(--accent);font-weight:700}.asio-mem{display:inline-flex;gap:.3rem;align-items:center}.asio-bar{display:inline-block;width:5rem;height:.45rem;border:1px solid var(--border);border-radius:.2rem;overflow:hidden}.asio-bar i{display:block;height:100%;background:var(--accent)}
.asio-split{display:flex;gap:.6rem;align-items:flex-start;margin-top:.4rem}.asio-list{flex:0 0 18rem;max-height:75vh;overflow-y:auto}.asio-detail{flex:1;min-width:0}
.asio-item{display:flex;gap:.35rem;align-items:flex-start;padding:.25rem .3rem;border:var(--border-thick) solid transparent;border-bottom-color:var(--border);cursor:pointer;font-size:.76rem}.asio-item:hover{background:var(--glass)}
.asio-rhead{margin-bottom:.3rem}.asio-grid{display:flex;flex-wrap:wrap;gap:.6rem}.asio-chart{flex:1 1 42%;min-width:18rem;border:var(--border-thick) solid var(--border);border-radius:var(--radius);padding:.3rem}.asio-full{flex-basis:100%}.asio-knob{gap:.2rem .6rem;padding:.1rem 0;font-size:.74rem}.asio-link{cursor:pointer;color:var(--accent)}.asio-role{font-weight:700;font-size:.8rem}
.asio-ref{font-size:.72rem}.asio-ref .asio-in{width:100%;margin:.2rem 0}.asio-ref-h{cursor:pointer;font-weight:700;color:var(--text_muted);margin-top:.4rem}.asio-ref-e{padding:.2rem 0;border-bottom:1px solid var(--border);cursor:pointer}.asio-ref-e div{color:var(--text_muted);font-size:.66rem}.asio-ref-row{display:flex;justify-content:space-between;gap:.3rem;font-size:.66rem;border-bottom:1px solid var(--border)}
@media (max-width:46rem){.asio-split{flex-direction:column}.asio-list{flex:none;width:100%;max-height:40vh}.asio-w20,.asio-w30{width:100%}}
"""
