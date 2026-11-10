"""asio_ui/lookup.py - filling the catalog from outside sources, only when asked: OpenRouter's model list (API prices and context), a Hugging Face model's config.json (layers, experts, KV size, context), Epoch's benchmark data (Capabilities Index).
Every lookup is a click: the portal server fetches the page (no keys, no client library, no other contact), keeps the answer in data/ai_tools/asio/lookup_cache/ (shown with its age; Refresh fetches again),
and shows what would change against the catalog. Nothing is written until Write; writing goes to edits_<collection>.json like any catalog edit (the file's previous content is kept in catalog_history.jsonl).
When the network blocks a source, paste what it would have answered (the JSON, config.json or CSV) instead: the same parsing and the same diff follow.
URLS can be pointed elsewhere (a mirror, a saved copy served locally)."""
import json, re, io, csv, time, zipfile, uuid
from asio_ui import base as B, fields as F

URLS = {"openrouter": "https://openrouter.ai/api/v1/models", "hf_config": "https://huggingface.co/{repo}/resolve/main/config.json", "hf_info": "https://huggingface.co/api/models/{repo}", "epoch": "https://epoch.ai/data/benchmark_data.zip"}
CACHE = B.DATA_DIR / "lookup_cache"
PENDING: dict = {}

async def fetch(key: str, url: str, refresh: bool = False) -> tuple:
    """(bytes, fetched at) from the cache, or from the URL when asked to refresh or never fetched."""
    import httpx
    CACHE.mkdir(parents=True, exist_ok=True)
    f, meta = CACHE / re.sub(r"[^A-Za-z0-9_.-]", "_", key), CACHE / (re.sub(r"[^A-Za-z0-9_.-]", "_", key) + ".t")
    if f.exists() and not refresh: return f.read_bytes(), float(meta.read_text())
    async with httpx.AsyncClient(timeout=httpx.Timeout(connect=10.0, read=60.0, write=10.0, pool=10.0), follow_redirects=True) as cl: r = await cl.get(url)
    if r.status_code != 200: raise ValueError(f"""{url} answered {r.status_code}{' (gated or private: paste config.json instead)' if r.status_code in (401, 403) else ''}""")
    f.write_bytes(r.content)
    meta.write_text(str(time.time()))
    return r.content, time.time()

def slug(s: str) -> str: return re.sub(r"[^a-z0-9.]+", "-", s.lower()).strip("-")
def norm(s: str) -> str: return re.sub(r"[^a-z0-9]", "", (s or "").lower())

def diff(c, coll: str, i: str, patch: dict) -> str:
    """Field by field: what the catalog says now, what the lookup says."""
    now = F.flat(c.get(coll, i)) if i in c.data[coll] else {}
    exact = lambda v, u: f"""{v:g} <span class="asio-unit">{B.esc(u)}</span>""" if isinstance(v, float) else F.show(v, u)   # looked-up numbers in full, so a small change shows
    rows = [[B.esc(F.label(k)[0]), exact(now.get(k), F.label(k)[1]), exact(v.get("theory"), F.label(k)[1]) + " theory" if F.is_spec(v) and set(v) <= {"theory", "better", "src"} else exact(v, F.label(k)[1])] for k, v in F.flat(patch).items() if now.get(k) != v]
    return B.tbl(["field", "catalog now", "looked up"], rows) if rows else '<div class="asio-dim">The catalog already says this.</div>'

def offer(c, coll: str, i: str, patch: dict, why: str) -> str:
    pid = uuid.uuid4().hex[:10]
    PENDING[pid] = (coll, i, patch, why)
    return f"""<div class="asio-card"><div class="row"><b class="grow">{B.esc(coll)}: {B.esc(i)}{'' if i in c.data[coll] else ' (new)'}</b>{B.btn("Write", "asio_lookup", op="write", pid=pid)}</div>{diff(c, coll, i, patch)}</div>"""

# --- OpenRouter: API prices ---

def openrouter_rows(raw: bytes) -> list:
    """[{"id", "name", "in", "cached", "out", "ctx"}] from OpenRouter's /models answer; prices in $ per million tokens."""
    per_m = lambda v: round(float(v) * 1e6, 6) if v not in (None, "", "-1") else None
    return [{"id": m["id"], "name": m.get("name", m["id"]), "in": per_m((m.get("pricing") or {}).get("prompt")), "cached": per_m((m.get("pricing") or {}).get("input_cache_read")), "out": per_m((m.get("pricing") or {}).get("completion")), "ctx": m.get("context_length")} for m in json.loads(raw)["data"]]

def api_patch(r: dict, when: str) -> dict:
    return {"label": f"""{r['name']} (OpenRouter)""", "openrouter": r["id"], "in": r["in"], "cached": r["cached"] if r["cached"] is not None else round((r["in"] or 0) * 0.1, 6), "out": r["out"], "date": when, "src": f"""OpenRouter /api/v1/models, {r['id']}{'' if r['cached'] is not None else '; cached rate not listed, 10% of input assumed'}"""}

def api_id(c, r: dict) -> str:
    """The catalog API entry this OpenRouter model updates: one that names it, else a new id."""
    return next((i for i, a in c.data["apis"].items() if a.get("openrouter") == r["id"]), f"""or-{slug(r['id'])}""")

# --- Hugging Face: model shape ---

def hf_patch(cfg: dict, info: dict, repo: str, when: str) -> dict:
    """Catalog model fields from config.json (and the model page's parameter count): layers, experts, attention layers, KV per token, longest context. KV per token is f16 bytes, as the catalog counts it."""
    t = cfg.get("text_config") or cfg
    L = t.get("num_hidden_layers") or t.get("n_layer") or t.get("num_layers")
    E, a = t.get("n_routed_experts") or t.get("num_experts") or t.get("num_local_experts"), t.get("num_experts_per_tok") or t.get("num_experts_per_token") or t.get("moe_topk")
    types = t.get("layer_types") or []
    full = sum(1 for x in types if "full" in str(x) or str(x) == "attention") if types else (L // t["full_attention_interval"] if t.get("full_attention_interval") else L)
    heads = t.get("num_attention_heads")
    kvh, hd = t.get("num_key_value_heads") or heads, t.get("head_dim") or ((t.get("hidden_size") // heads) if heads and t.get("hidden_size") else None)
    src = {"src": f"""config.json of {repo}, read {when}"""}
    kv = (t["kv_lora_rank"] + t.get("qk_rope_head_dim", 0)) * L * 2 / 1000 if t.get("kv_lora_rank") else (2 * kvh * hd * full * 2 / 1000 if kvh and hd and full else None)
    total = ((info.get("safetensors") or {}).get("total") or 0) / 1e9
    out = {"layers": L, "full_attn_layers": full if full and full != L else None, "ctx_max": t.get("max_position_embeddings"), "kv_kb_per_token": {"theory": round(kv, 3), "better": "low", **src} if kv else None, "params_b": round(total, 2) if total else None, "hf": repo if "/" in repo else None}
    if E: out.update(arch="moe", experts_per_layer=E, active_experts=a, moe_layers=L - (t.get("first_k_dense_replace") or 0))
    elif types and full != L: out["arch"] = "hybrid"
    return {k: v for k, v in out.items() if v is not None}

# --- Epoch: Capabilities Index ---

def epoch_scores(raw: bytes) -> dict:
    """{model name: ECI} from Epoch's benchmark zip, or a pasted CSV: the first table with a model column and a column named ECI (any case, "ECI score" too). The zip's layout is not documented, so every CSV in it is tried."""
    tables = []
    if raw[:2] == b"PK":
        z = zipfile.ZipFile(io.BytesIO(raw))
        tables = [z.read(n).decode("utf-8", "replace") for n in z.namelist() if n.lower().endswith(".csv")]
    else: tables = [raw.decode("utf-8", "replace")]
    for t in tables:
        rd = list(csv.DictReader(io.StringIO(t)))
        if not rd: continue
        cols = {c.lower().strip(): c for c in rd[0]}
        ec = next((cols[k] for k in ("eci", "eci score", "eci_score", "epoch capabilities index") if k in cols), None)
        mc = next((cols[k] for k in ("model", "model version", "model_version", "name", "display name") if k in cols), None)
        if ec and mc: return {r[mc]: float(r[ec]) for r in rd if r.get(ec) not in (None, "") and re.match(r"^-?[0-9.]+$", r[ec].strip())}
    raise ValueError("no table with a model column and an ECI column was found")

def close(a: str, b: str) -> bool:
    """One normalised name inside the other, not followed by a digit (so "claudeopus5" does not match "claudeopus55"), at least 6 characters."""
    s, l = sorted((a, b), key=len)
    i = l.find(s)
    return len(s) >= 6 and i >= 0 and not l[i + len(s):i + len(s) + 1].isdigit() and not l[i - 1:i].isdigit()

def epoch_matches(c, scores: dict) -> list:
    """(collection, id, label, ECI now, ECI found, Epoch's name) for catalog models and APIs whose name matches an Epoch model's."""
    keyed = {norm(k): (k, v) for k, v in scores.items()}
    out = []
    for coll in ("models", "apis"):
        for i in c.ids(coll):
            e = c.get(coll, i)
            n = norm(re.sub(r"\(.*?\)", "", e.get("label", i)))
            hit = keyed.get(n) or next((kv for k, kv in keyed.items() if close(n, k)), None)
            if hit: out.append((coll, i, e.get("label", i), e.get("eci"), hit[1], hit[0]))
    return out

# --- Tab section ---

def age(t: float) -> str: return f"""fetched {B.ago(t)}"""

def page(ws) -> str:
    f = ws.form("lookup")
    c = B.cat()
    models = "".join(f"""<option value="{B.esc(i)}">{B.esc(c.get('models', i).get('label', i))}</option>""" for i in c.ids("models"))
    ta = lambda n, ph: f"""<textarea class="asio-in asio-w30" name="{n}" rows="2" placeholder="{B.esc(ph)}"></textarea>"""
    return f"""<div class="asio-dim">Each lookup contacts only the site named, from the portal server, when you click; answers are cached in {B.esc(CACHE)}. If the network blocks a site, paste what it would answer.</div>
        <form class="asio-card" id="asio-lk-or" onsubmit="return false"><b>API prices: OpenRouter</b><div class="row asio-wrap">{F.field("search", f'<input class="asio-in asio-w14" name="q" value="{B.esc(f.get("or_q", ""))}" placeholder="e.g. claude, qwen3.8">')}
            {B.btn("Look up", "asio_lookup", op="or", include="#asio-lk-or")}{B.btn("Refresh", "asio_lookup", op="or", refresh="1", include="#asio-lk-or")}{F.field("or paste the /api/v1/models JSON", ta("paste", '{"data": [...]}'))}</div><div id="asio-lk-or-out"></div></form>
        <form class="asio-card" id="asio-lk-hf" onsubmit="return false"><b>Model shape: Hugging Face config.json</b><div class="row asio-wrap">{F.field("repository", f'<input class="asio-in asio-w14" name="repo" value="{B.esc(f.get("hf_repo", ""))}" placeholder="org/model">')}
            {F.field("catalog model it describes", f'<select class="asio-in asio-w14" name="model"><option value="">(a new entry)</option>{models}</select>')}
            {B.btn("Look up", "asio_lookup", op="hf", include="#asio-lk-hf")}{B.btn("Refresh", "asio_lookup", op="hf", refresh="1", include="#asio-lk-hf")}{F.field("or paste config.json", ta("paste", '{"num_hidden_layers": ...}'))}</div><div id="asio-lk-hf-out"></div></form>
        <form class="asio-card" id="asio-lk-ep" onsubmit="return false"><b>Capabilities: Epoch ECI</b><div class="row asio-wrap">{B.btn("Look up", "asio_lookup", op="epoch", include="#asio-lk-ep")}{B.btn("Refresh", "asio_lookup", op="epoch", refresh="1", include="#asio-lk-ep")}{F.field("or paste a CSV with model and ECI columns", ta("paste", "Model,ECI"))}</div><div id="asio-lk-ep-out"></div></form>"""

async def h_lookup(request, payload, imr):
    from asio_ui.build import say
    from asio_ui.catalog_ui import edits, write_edits
    ws, op, c = B.ws(request), payload["op"], B.cat()
    paste, refresh, today = (payload.get("paste") or "").strip(), bool(payload.get("refresh")), time.strftime("%Y-%m-%d")
    try:
        if op == "or":
            ws.set_form("lookup", {"or_q": payload.get("q", "")})
            raw, t = (paste.encode(), time.time()) if paste else await fetch("openrouter.json", URLS["openrouter"], refresh)
            q = norm(payload.get("q"))
            rs = [r for r in openrouter_rows(raw) if not q or q in norm(r["id"]) or q in norm(r["name"])][:12]
            body = "".join(offer(c, "apis", api_id(c, r), api_patch(r, today), f"OpenRouter {r['id']}") for r in rs) or '<div class="asio-dim">nothing matches</div>'
            return imr.oob(f"""<div class="asio-dim">{'pasted' if paste else age(t)}; {len(rs)} shown</div>{body}""", "asio-lk-or-out")
        if op == "hf":
            repo = (payload.get("repo") or "").strip().strip("/")
            ws.set_form("lookup", {"hf_repo": repo})
            if not repo and not paste: raise ValueError("give the repository (org/model)")
            if paste: cfg, info, t = json.loads(paste), {}, time.time()
            else:
                cb, t = await fetch(f"hf_{repo}_config.json", URLS["hf_config"].format(repo=repo), refresh)
                try: info = json.loads((await fetch(f"hf_{repo}_info.json", URLS["hf_info"].format(repo=repo), refresh))[0])
                except ValueError: info = {}
                cfg = json.loads(cb)
            patch = hf_patch(cfg, info, repo or "a pasted config", today)
            i = payload.get("model") or slug(repo.split("/")[-1] if repo else "pasted-model")
            if i not in c.data["models"]: patch = {"extends": "llm", "label": repo.split("/")[-1] if repo else i, "kind": "llm", "default_quant": "Q4_K_M", **patch}
            return imr.oob(f"""<div class="asio-dim">{'pasted' if paste else age(t)}</div>{offer(c, "models", i, patch, f"Hugging Face {repo}")}""", "asio-lk-hf-out")
        if op == "epoch":
            raw, t = (paste.encode(), time.time()) if paste else await fetch("epoch_benchmark_data.zip", URLS["epoch"], refresh)
            m = epoch_matches(c, epoch_scores(raw))
            rows = "".join(offer(c, coll, i, {"eci": v, "eci_src": f"Epoch ({name}), {today}"}, "Epoch ECI") for coll, i, lbl, now, v, name in m if now != v)
            return imr.oob(f"""<div class="asio-dim">{'pasted' if paste else age(t)}; {len(m)} catalog entries matched an Epoch model by name; check each match before writing</div>{rows or '<div class="asio-dim">Nothing to change.</div>'}""", "asio-lk-ep-out")
        if op == "write":
            item = PENDING.pop(payload.get("pid", ""), None)
            if not item: raise ValueError("that lookup is gone: look it up again, then write")
            coll, i, patch, why = item
            from asio_core.catalog import merge
            ed = edits(coll)
            write_edits(coll, {**ed, i: merge(ed.get(i, {}), patch)}, f"lookup: {why}")
            return say(imr, f"Written to edits_{coll}.json: {i}.", bad=False)
    except (ValueError, KeyError, json.JSONDecodeError, zipfile.BadZipFile, OSError) as e: return say(imr, f"{type(e).__name__}: {e}")
    except Exception as e:
        if type(e).__module__.startswith("httpx") or type(e).__module__.startswith("httpcore"): return say(imr, f"""could not reach the site ({type(e).__name__}: {e}); paste its answer instead""")
        raise
    return imr

INTENTS = {"asio_lookup": h_lookup}
