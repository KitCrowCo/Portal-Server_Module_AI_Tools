"""Context memory - the page for AI Manager's context instances (tools/ai_manager/context.py): which folders each domain holds, what its material's status is, which nodes embed and rerank it, and a search to try it.
An instance is what an agent picks as a knowledge group "<name> (context memory)". Nothing here changes the material itself: status rules, settings, claims, decisions and the index live under data/_tools/ai_manager/context/."""
import json, html, asyncio
from fastapi import APIRouter, Request

TOOL_META = {"label": "Context memory", "group": "model_tools", "icon": "&#x1F9E0;", "description": "Domains of notes, code and papers as cited, status-aware context for agents", "singleton": True}
router = APIRouter(redirect_slashes=False)
ENV, UI, AIM, IM, _P = {}, None, None, None, "/module/ai_tools/memory"
JOBS: dict = {}
_TASKS: set = set()

def _esc(s) -> str: return html.escape("" if s is None else str(s), quote=True)
def _act(t: str, trigger: str = None, include: str = None, confirm: str = None, **kw) -> str: return f"""hx-post="/im/in" hx-swap="none" hx-vals="{_esc(json.dumps({"type": t, "branch": IM.branch_id, "lvl": 2, **kw}))}"{f' hx-trigger="{trigger}"' if trigger else ""}{f' hx-include="{include}"' if include else ""}{f' hx-confirm="{_esc(confirm)}"' if confirm else ""}"""
def _btn(label: str, t: str, cls: str = "mem-btn", **kw) -> str: return f"""<button type="button" class="{cls}" {_act(t, **kw)}>{label}</button>"""
def _C(): return AIM.context

# --- Page parts ---

def _card(name: str, i: dict, open_: str) -> str:
    ctx = _C().Context(name)
    ctx.update()   # cheap when nothing changed: only sizes and times are compared
    st = ctx.stats()
    emb = f""", {st['embedded']} of {st['pieces']} embedded""" if st["embedded"] is not None else ", words only"
    emb += f""", {st['claimed']} of {st['pieces']} read for claims""" if st.get("claimed") is not None else ""
    by = ", ".join(f"{k} {v}" for k, v in sorted(st["by_status"].items()))
    job = JOBS.get(name, "")
    return f"""<div class="mem-card {'mem-on' if name == open_ else ''}"><div class="row"><b class="grow" {_act("mem_open", name=name)} style="cursor:pointer">{_esc(name)}</b>{_btn("Update now", "mem_update", name=name)}{_btn("Embed all", "mem_embed", name=name) if i.get("embed") else ""}{_btn("Read claims", "mem_claims", name=name) if i.get("claims") else ""}{_btn("Remove", "mem_remove", cls="mem-btn mem-danger", name=name, confirm=f"Remove {name}? Its settings, index and status rules go to the context trash; the material is untouched.")}</div>
        <div class="mem-dim">{_esc(i.get("notes", ""))}</div><div class="mem-dim">{_esc(", ".join(i["folders"]))}: {st['files']} files, {st['pieces']} pieces, about {st['tokens']:,} tokens{emb}; {_esc(by)}{" - " + _esc(job) if job else ""}</div></div>"""

def _conn_select(field: str, cap: str, cur: dict) -> str:
    opts = "".join(f"""<option value="{_esc(c["_id"])}" {'selected' if c["_id"] == (cur or {}).get("conn") else ''}>{_esc(c.get("display_name", c["_id"]))}</option>""" for c in AIM.connections.conns_matching(cap))
    return f"""<select class="mem-in" name="{field}_conn"><option value="">(none)</option>{opts}</select><input class="mem-in" name="{field}_model" value="{_esc((cur or {}).get("model", ""))}" placeholder="model">"""

def _form(name: str) -> str:
    i = _C().instances().get(name, {"folders": [], "notes": "", "budget": 6000, "exclude": []})
    return f"""<form class="mem-card" id="mem-form" onsubmit="return false"><b>{'Edit ' + _esc(name) if name else 'New instance'}</b>
        <div class="mem-grid"><label>name<input class="mem-in" name="name" value="{_esc(name)}" {'readonly' if name else ''}></label><label>budget (tokens per search)<input class="mem-in" name="budget" value="{_esc(i.get('budget', 6000))}"></label>
        <label>folders, one per line<textarea class="mem-in" name="folders" rows="3">{_esc(chr(10).join(i['folders']))}</textarea></label><label>leave out (globs, one per line)<textarea class="mem-in" name="exclude" rows="3">{_esc(chr(10).join(i.get('exclude', [])))}</textarea></label>
        <label>what it holds (agents read this)<textarea class="mem-in" name="notes" rows="2">{_esc(i.get('notes', ''))}</textarea></label><div><label>meaning search: embedding node and model{_conn_select("embed", "embed", i.get("embed"))}</label><label>reranking node and model{_conn_select("rerank", "rerank", i.get("rerank"))}</label><label>claims and conflicts: chat node and model{_conn_select("claims", "chat", i.get("claims"))}</label></div></div>
        <div class="row">{_btn("Save", "mem_save", include="#mem-form")}<span class="mem-dim">Changing the embedding model re-embeds every piece; changing the claims model reads every piece again.</span></div></form>"""

def _rules(name: str) -> str:
    if not name: return ""
    C = _C()
    rows = "".join(f"""<tr><td class="mono">{_esc(g)}</td><td>{_esc(s)}</td><td>{_btn("&#x2715;", "mem_rule_del", name=name, i=k, confirm=f"Remove the rule {g} -> {s}?")}</td></tr>""" for k, (g, s) in enumerate(C.Context(name).own_rules()))
    defaults = ", ".join(f"{g} -> {s}" for g, s in C.DEFAULT_RULES[::2])
    opts = "".join(f"<option>{s}</option>" for s in C.STATUS)
    return f"""<div class="mem-card"><b>Status of the material in {_esc(name)}</b><div class="mem-dim">Later rules win. Paths start with the folder's own name. Defaults: {_esc(defaults)}; everything else is current. Superseded and abandoned material is only returned when asked for; a proposal is checked against canon and current only.</div>
        <table class="mem-tbl">{rows or '<tr><td class="mem-dim">no rules of your own yet</td></tr>'}</table><form class="row" id="mem-rule" onsubmit="return false"><input class="mem-in" name="glob" placeholder="e.g. world/era2/* or world/notes/old_magic.md"><select class="mem-in" name="status">{opts}</select>{_btn("Add rule", "mem_rule_add", name=name, include="#mem-rule")}</form></div>"""

def _try(name: str) -> str:
    if not name: return ""
    return f"""<form class="mem-card" id="mem-try" onsubmit="return false"><b>Try it</b><div class="row"><input class="mem-in grow" name="q" placeholder="a question, or a proposal to check"><select class="mem-in" name="include"><option value="current">current material</option><option value="binding">settled only (as a proposal check)</option><option value="all">everything</option></select>{_btn("Search", "mem_try", name=name, include="#mem-try")}</div><pre class="mem-out" id="mem-out"></pre></form>"""

def _claim_row(name: str, cid: str, x: dict) -> str:
    return f"""<tr><td><b>{_esc(x["value"])}</b><div class="mem-dim">{_esc(x["quote"])}</div></td><td class="mono mem-dim">{_esc(x["path"])}:{x["a"]}{"-" + str(x["b"]) if x["b"] != x["a"] else ""} | {_esc(x["status"])}</td><td>{_btn("This holds", "mem_decide", name=name, cid=cid, value=x["value"], include=f"#mem-k-{cid}")}</td></tr>"""

def _conflicts(name: str) -> str:
    ctx = _C().Context(name)
    if not ctx.cfg.get("claims"): return ""
    ks, done = ctx.conflicts(), ctx.decisions()[-5:][::-1]
    body = "".join(f"""<div class="mem-k"><b>{_esc(k["subject"])} - {_esc(k["attribute"])}</b><table class="mem-tbl">{"".join(_claim_row(name, k["id"], x) for x in k["claims"])}</table>
        <form class="row" id="mem-k-{k["id"]}" onsubmit="return false"><input class="mem-in" name="own" placeholder="or what holds, in your words"><input class="mem-in grow" name="note" placeholder="note (why; agents see it)">{_btn("Use mine", "mem_decide", name=name, cid=k["id"], include=f"#mem-k-{k['id']}")}{_btn("Different contexts", "mem_decide", name=name, cid=k["id"], value="contexts", include=f"#mem-k-{k['id']}")}</form></div>""" for k in ks)
    past = "".join(f"""<div class="mem-dim">{_esc(d["subject"])} - {_esc(d["attribute"])}: {_esc("both stand, in different contexts" if d["value"] == "contexts" else d["value"])}{" (" + _esc(d["note"]) + ")" if d.get("note") else ""}</div>""" for d in done)
    return f"""<div class="mem-card"><b>Conflicts in {_esc(name)}</b> <span class="mem-dim">({len(ks)} open)</span><div class="mem-dim">Current and canon sources that say different things about the same subject. Settling one changes nothing in the material: the answer is kept in the instance and comes first in every search about it. A later source saying something new opens it again.</div>
        {body or '<div class="mem-dim">none open - read claims after the material changes to look again</div>'}{'<div class="mem-dim" style="margin-top:.3rem"><b>Settled lately</b></div>' + past if past else ""}</div>"""

def page(open_: str = "") -> str:
    inst = _C().instances()
    open_ = open_ if open_ in inst else ""
    cards = "".join(_card(n, i, open_) for n, i in inst.items()) or '<div class="mem-dim">No instances yet: make one for each domain (a codebase, a body of notes, household papers...).</div>'
    return f"""<div class="mem-pad" id="mem-page"><div id="mem-msg"></div>{cards}<div class="row">{_btn("New instance", "mem_open", name="")}</div>{_form(open_)}{_rules(open_)}{_conflicts(open_) if open_ else ""}{_try(open_)}</div>"""

# --- Intents ---

def _say(imr, msg: str, bad: bool = True): return imr.oob(f"""<div class="{'mem-err' if bad else 'mem-note'}">{_esc(msg)}</div>""", "mem-msg")
def _lines(s: str) -> list: return [x.strip() for x in (s or "").splitlines() if x.strip()]

async def _h_open(request, payload, imr): return imr.oob(page(payload.get("name", "")), "mem-page", swap="outerHTML")

async def _h_save(request, payload, imr):
    name = (payload.get("name") or "").strip()
    if not name or not _lines(payload.get("folders")): return _say(imr, "give the instance a name and at least one folder")
    pick = lambda f: {"conn": payload[f"{f}_conn"], "model": (payload.get(f"{f}_model") or "").strip()} if payload.get(f"{f}_conn") else None
    _C().set_instance(name, _lines(payload.get("folders")), payload.get("notes", ""), int(payload.get("budget") or 6000), _lines(payload.get("exclude")), pick("embed"), pick("rerank"), pick("claims"))
    return imr.oob(page(name), "mem-page", swap="outerHTML")

def _spawn(name: str, label: str, coro):
    JOBS[name] = f"{label}..."
    async def run():
        try: JOBS[name] = f"{label}: {await coro}"
        except Exception as e: JOBS[name] = f"{label} failed: {type(e).__name__}: {e}"
    t = asyncio.create_task(run())
    _TASKS.add(t)
    t.add_done_callback(_TASKS.discard)

async def _h_update(request, payload, imr):
    _spawn(payload["name"], "update", asyncio.to_thread(_C().Context(payload["name"]).update))
    return _say(imr, f"""Updating {payload['name']}; open it again to see the counts.""", bad=False)

async def _h_embed(request, payload, imr):
    _spawn(payload["name"], "embedding", _C().Context(payload["name"]).embed_all())
    return _say(imr, f"""Embedding {payload['name']} in the background; searches work by words meanwhile.""", bad=False)

async def _h_claims(request, payload, imr):
    _spawn(payload["name"], "reading claims", _C().Context(payload["name"]).extract_all())
    return _say(imr, f"""Reading claims in {payload['name']} in the background (one model call per new or changed piece); open it again to see conflicts.""", bad=False)

async def _h_decide(request, payload, imr):
    value = (payload.get("value") or payload.get("own") or "").strip()
    if not value: return _say(imr, "pick one of the claims, or write what holds")
    try: _C().Context(payload["name"]).decide(payload["cid"], value, (payload.get("note") or "").strip())
    except ValueError as e: return _say(imr, str(e))
    return imr.oob(page(payload["name"]), "mem-page", swap="outerHTML")

async def _h_remove(request, payload, imr):
    to = _C().remove_instance(payload["name"])
    return imr.oob(page(), "mem-page", swap="outerHTML").oob(f"""<div class="mem-note">Moved to {_esc(to)}.</div>""", "mem-msg")

async def _h_rule_add(request, payload, imr):
    ctx, g = _C().Context(payload["name"]), (payload.get("glob") or "").strip()
    if not g: return _say(imr, "give the path or glob the rule is for")
    ctx.set_rules(ctx.own_rules() + [[g, payload.get("status", "current")]])
    return imr.oob(page(payload["name"]), "mem-page", swap="outerHTML")

async def _h_rule_del(request, payload, imr):
    ctx = _C().Context(payload["name"])
    rules = ctx.own_rules()
    rules.pop(int(payload["i"]))
    ctx.set_rules(rules)
    return imr.oob(page(payload["name"]), "mem-page", swap="outerHTML")

async def _h_try(request, payload, imr):
    if not (payload.get("q") or "").strip(): return _say(imr, "type something to search for")
    try: text = await _C().Context(payload["name"]).aquery(payload["q"], include=payload.get("include") or "current")
    except RuntimeError as e: return _say(imr, str(e))
    return imr.oob(_esc(text), "mem-out")

# --- Init and route ---

@router.get("")
@router.get("/")
async def root(request: Request):
    return ENV["templates"].TemplateResponse(name="base.html", request=request, context={"request": request, "user": request.state.user, "nesting_level": 2, "shell_id": IM.branch_id, "content": page(), "extra_css": CSS, "extra_script": ""})

def init_tool(env, prefix):
    global ENV, UI, AIM, IM, _P
    ENV, _P = env, prefix.rstrip("/")
    UI, AIM = env["templates"].env.globals.get("UI"), env["tools"]["ai_manager"]
    IM = env["InterfaceManager"](nesting_level=2, db_path=f"""{env["data_dir"]}/memory/im_registry.db""")
    IM.scripts.update({"mem_open": [_h_open], "mem_save": [_h_save], "mem_update": [_h_update], "mem_embed": [_h_embed], "mem_remove": [_h_remove], "mem_rule_add": [_h_rule_add], "mem_rule_del": [_h_rule_del], "mem_try": [_h_try], "mem_claims": [_h_claims], "mem_decide": [_h_decide]})
    print("[memory] ready")

CSS = """
.mem-pad{padding:.6rem .8rem;overflow-y:auto;height:100%;box-sizing:border-box}.mem-card{border:var(--border-thick) solid var(--border);border-radius:var(--radius);padding:.45rem .6rem;margin:.4rem 0;background:var(--surface)}.mem-on{border-color:var(--accent)}
.mem-dim{font-size:.72rem;color:var(--text_muted)}.mem-in{background:var(--bg);color:var(--text);border:var(--border-thick) solid var(--border);border-radius:calc(var(--radius) / 2);padding:.15rem .3rem;font-size:.78rem;font-family:inherit;box-sizing:border-box;max-width:100%}
.mem-grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(16rem,1fr));gap:.3rem .8rem;margin:.3rem 0}.mem-grid label{display:flex;flex-direction:column;font-size:.68rem;color:var(--text_muted);gap:.1rem}
.mem-btn{background:var(--accent_dim);color:var(--accent);border:var(--border-thick) solid var(--accent);border-radius:calc(var(--radius) / 2);padding:.12rem .45rem;font-size:.74rem;font-weight:600;cursor:pointer;white-space:nowrap}.mem-danger{color:#ff6b6b;border-color:#ff6b6b;background:none}
.mem-tbl{font-size:.74rem;border-collapse:collapse;margin:.3rem 0}.mem-tbl td{padding:.1rem .5rem;border-bottom:1px solid var(--border)}.mem-out{white-space:pre-wrap;font-size:.72rem;max-height:50vh;overflow-y:auto;margin:.3rem 0 0}
.mem-k{border-top:1px solid var(--border);padding:.3rem 0;margin-top:.3rem}.mem-k .mem-tbl{width:100%}
.mem-note{border-left:3px solid var(--accent);padding:.3rem .5rem;margin:.3rem 0;font-size:.78rem}.mem-err{border-left:3px solid #ff6b6b;padding:.3rem .5rem;margin:.3rem 0;font-size:.78rem}
"""
