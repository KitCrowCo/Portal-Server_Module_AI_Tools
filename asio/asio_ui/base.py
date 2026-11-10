"""asio_ui/base.py - what every part of the Asio page shares: the portal's handles, paths, the simulator and catalog, the user's workspace, and small HTML helpers.
The page modules (build, run_ui, catalog_ui, ref) import this as B and read B.ENV, B.IM and B.P once init_tool has set them."""
import json, html, time
from pathlib import Path

ENV, IM, FOCUS = {}, None, None   # FOCUS(request, imr, tab_id): shows one of the page's tabs; set by asio.py
P = "/module/ai_tools/asio"
DATA_DIR = Path("./data/ai_tools/asio")
USER_CAT = DATA_DIR / "catalog"
WS_ROOT = DATA_DIR / "workspace"
_CAT = {"key": None, "cat": None}

def libs():
    """The simulator package and Athene. Athene is the portal tool at tools/athene/, handed over in ENV["tools"]."""
    import asio_core as asio
    from asio_core import viz, run
    if run.ATHENE is None:
        t = (ENV.get("tools") or {}).get("athene")
        if t is None: raise ModuleNotFoundError("the Athene tool is not loaded: put the Athene repository at tools/athene/ (so tools/athene/athene.py exists) and restart the portal")
        run.use_athene(t)
    return asio, viz, run

def cat():
    """The merged catalog (built-in, then the user catalog folder), rebuilt only when a file in the user folder changes."""
    asio, _, _ = libs()
    key = tuple((str(f), f.stat().st_mtime) for f in sorted(USER_CAT.rglob("*.json"))) if USER_CAT.exists() else ()
    if _CAT["key"] != key: _CAT.update(key=key, cat=asio.Catalog(USER_CAT))
    return _CAT["cat"]

def ws(request):
    from asio_ui.workspace import Workspace
    return Workspace(WS_ROOT, request.state.user.username)

# --- HTML helpers ---

esc = lambda s: html.escape("" if s is None else str(s), quote=True)

def act(t: str, trigger: str = None, include: str = None, confirm: str = None, **kw) -> str:
    """Attributes that send intent t with kw through /im/in; every answer is out-of-band swaps, so the element itself is never replaced."""
    extra = (f' hx-trigger="{trigger}"' if trigger else "") + (f' hx-include="{include}"' if include else "") + (f' hx-confirm="{esc(confirm)}"' if confirm else "")
    return f"""hx-post="/im/in" hx-swap="none" hx-vals="{esc(json.dumps({"type": t, "branch": IM.branch_id, "lvl": 2, **kw}))}"{extra}"""

def btn(label: str, t: str, cls: str = "asio-btn", title: str = "", **kw) -> str: return f"""<button type="button" class="{cls}" title="{esc(title)}" {act(t, **kw)}>{label}</button>"""

def tbl(head: list, rows: list, cls: str = "") -> str:
    th = "".join(f"<th>{h}</th>" for h in head)
    tr = "".join("<tr>" + "".join(f"<td>{c}</td>" for c in r) + "</tr>" for r in rows)
    return f"""<div class="asio-tscroll"><table class="asio-tbl {cls}"><thead><tr>{th}</tr></thead><tbody>{tr}</tbody></table></div>"""

def err(msg: str) -> str: return f"""<div class="asio-err">{esc(msg)}</div>"""
def note(msg: str) -> str: return f"""<div class="asio-note">{msg}</div>"""
def sec(title: str, body: str, extra: str = "", open_: bool = True, sid: str = "") -> str: return f"""<details class="asio-sec" {'open' if open_ else ''} {f'id="{sid}"' if sid else ''}><summary><span>{title}</span>{extra}</summary><div class="asio-sec-body">{body}</div></details>"""
def ago(t: float) -> str:
    d = time.time() - t
    return "just now" if d < 60 else f"{d / 60:.0f} min ago" if d < 3600 else f"{d / 3600:.0f} h ago" if d < 86400 else time.strftime("%Y-%m-%d %H:%M", time.localtime(t))

def fmt(v, kind: str = "n") -> str:
    """t: a duration, %: a share, $: money, n: a number with sensible digits."""
    if v is None or (isinstance(v, float) and v != v): return "-"
    if isinstance(v, float) and v == float("inf"): return "none done"
    _, viz, _ = libs()
    return viz.fmt_t(v) if kind == "t" else f"{v:.0%}" if kind == "%" else f"${v:.4g}" if kind == "$" else viz.fmt(v)
