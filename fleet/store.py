"""store.py - Fleet's database (data/ai_tools/fleet/fleet.db, SQLite) and the collector that fills it from every node's journal. Loaded by fleet.py, which passes AIM in init().
Tables:
  events   every journal event of every node, unique by (node, epoch, id): t (node clock), kind, rid, job, model, data (the whole event as JSON)
  cursors  per node: the node's own name, the epoch and last id read, when it last answered, its last error, its clock offset (node time minus portal time, seconds)
  states   per node: the latest /fleet/state answer
  actions  every action Fleet proposed or sent: where it came from (plan, schedule, manual), the policy, mode (shadow = shown only, sent = sent to the node), the node's answer
Nodes are connections whose profile declares the capability fleet_events (Magpie); the node key is the connection id. Two connections to the same server are collected once.
The collector only reads nodes. A node's journal is the source of truth: this file can be deleted, and fills again from what the nodes still keep."""
import json, time, sqlite3, asyncio
from pathlib import Path

DATA_DIR = Path("./data/ai_tools/fleet")
DB_PATH = DATA_DIR / "fleet.db"
SCHEMA = """CREATE TABLE IF NOT EXISTS events (node TEXT, epoch TEXT, id INTEGER, t REAL, kind TEXT, rid TEXT, job TEXT, model TEXT, data TEXT, PRIMARY KEY (node, epoch, id));
CREATE INDEX IF NOT EXISTS ev_node_t ON events (node, t);
CREATE INDEX IF NOT EXISTS ev_kind_t ON events (kind, t);
CREATE INDEX IF NOT EXISTS ev_job ON events (job);
CREATE TABLE IF NOT EXISTS cursors (node TEXT PRIMARY KEY, node_name TEXT, epoch TEXT, last_id INTEGER, ok_t REAL, err TEXT, err_t REAL, offset REAL, gap INTEGER DEFAULT 0);
CREATE TABLE IF NOT EXISTS states (node TEXT PRIMARY KEY, t REAL, data TEXT);
CREATE TABLE IF NOT EXISTS actions (n INTEGER PRIMARY KEY AUTOINCREMENT, t REAL, node TEXT, source TEXT, policy TEXT, action TEXT, args TEXT, why TEXT, value REAL, mode TEXT, status TEXT, result TEXT, directive TEXT);
CREATE INDEX IF NOT EXISTS act_node_t ON actions (node, t);"""
AIM = None
STATUS = {"running": False, "last_cycle": None, "cycle_s": None, "error": ""}

def init(aim):
    global AIM
    AIM = aim
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    with _con() as c: c.execute("PRAGMA journal_mode=WAL"); c.executescript(SCHEMA)

def _con() -> sqlite3.Connection:
    c = sqlite3.connect(DB_PATH, timeout=30)
    c.row_factory = sqlite3.Row
    return c

def q(sql: str, args: tuple = ()) -> list:
    with _con() as c: return [dict(r) for r in c.execute(sql, args)]

def x(sql: str, args: tuple = ()) -> int:
    with _con() as c: return c.execute(sql, args).lastrowid

def events(node: str = "", since: float = 0, until: float = 0, kinds: tuple = (), job: str = "", limit: int = 200000) -> list:
    """Events as dicts (the event's own fields plus node), oldest first. kinds: exact kinds, or prefixes ending in '.' (e.g. 'llama.')."""
    w, a = ["t >= ?"], [since]
    if until: w.append("t <= ?"); a.append(until)
    if node: w.append("node = ?"); a.append(node)
    if job: w.append("job = ?"); a.append(job)
    if kinds:
        w.append("(" + " OR ".join("kind LIKE ?" if k.endswith(".") else "kind = ?" for k in kinds) + ")")
        a += [k + "%" if k.endswith(".") else k for k in kinds]
    return [{**json.loads(r["data"]), "node": r["node"]} for r in q(f"""SELECT node, data FROM events WHERE {" AND ".join(w)} ORDER BY t, id LIMIT ?""", (*a, limit))]

def state(node: str) -> dict:
    r = q("SELECT t, data FROM states WHERE node = ?", (node,))
    return {**json.loads(r[0]["data"]), "_collected": r[0]["t"]} if r else {}

def cursors() -> dict: return {r["node"]: r for r in q("SELECT * FROM cursors")}

# --- Nodes ---

def nodes() -> list:
    """Connections with a Fleet journal (capability fleet_events), one per server address."""
    seen, out = set(), []
    for c in AIM.connections.conns_matching("fleet_events"):
        base = AIM.connections.base_url(c)
        if base not in seen: seen.add(base); out.append(c)
    return out

def node_label(conn: dict) -> str: return conn.get("display_name") or conn["_id"]

# --- Collector ---

def _insert(node: str, epoch: str, evs: list):
    rows = [(node, epoch, e["id"], e["t"], e["kind"], e.get("rid", ""), e.get("job", "") or (e.get("rid", "").split(".", 1)[0] if e.get("rid") else ""), e.get("model") or e.get("name") or "", json.dumps(e, ensure_ascii=False)) for e in evs]
    with _con() as c: c.executemany("INSERT OR IGNORE INTO events (node, epoch, id, t, kind, rid, job, model, data) VALUES (?,?,?,?,?,?,?,?,?)", rows)

def _cursor_set(node: str, **f):
    with _con() as c:
        c.execute("INSERT OR IGNORE INTO cursors (node, last_id) VALUES (?, 0)", (node,))
        c.execute(f"""UPDATE cursors SET {", ".join(k + " = ?" for k in f)} WHERE node = ?""", (*f.values(), node))

async def collect_events(conn: dict, batch: int = 2000, rounds: int = 20) -> int:
    """Pulls new events from one node until it has no more (at most rounds x batch per call). A changed epoch (the node's journal was reset) starts that node over from id 0; events already stored keep their old epoch."""
    node, got = conn["_id"], 0
    cur = cursors().get(node) or {"epoch": "", "last_id": 0}
    for _ in range(rounds):
        r = await AIM.connections.call_capability(conn, "fleet_events", {"after": cur["last_id"] or 0, "limit": batch}, timeout_s=30)
        if r.get("error"): await asyncio.to_thread(_cursor_set, node, err=str(r["error"])[:300], err_t=time.time()); return got
        if r["epoch"] != cur["epoch"]:   # a new node, or its journal was reset
            await asyncio.to_thread(_cursor_set, node, epoch=r["epoch"], last_id=0)
            restart, cur = cur["last_id"] != 0, {"epoch": r["epoch"], "last_id": 0}
            if restart: continue   # the answer was for an id of the old epoch: ask again from the start
        evs = r.get("events") or []
        if evs: await asyncio.to_thread(_insert, node, r["epoch"], evs)
        cur["last_id"] = evs[-1]["id"] if evs else cur["last_id"]
        await asyncio.to_thread(_cursor_set, node, node_name=r["node"], last_id=cur["last_id"], ok_t=time.time(), err="", offset=round(r["now"] - time.time(), 2), gap=1 if r.get("gap") else 0)
        got += len(evs)
        if not r.get("more"): return got
    return got

async def collect_state(conn: dict):
    r = await AIM.connections.call_capability(conn, "fleet_state", {}, timeout_s=30)
    if r.get("error"): return await asyncio.to_thread(_cursor_set, conn["_id"], err=str(r["error"])[:300], err_t=time.time())
    await asyncio.to_thread(x, "INSERT OR REPLACE INTO states (node, t, data) VALUES (?, ?, ?)", (conn["_id"], time.time(), json.dumps(r, ensure_ascii=False)))

def prune(days: float) -> int:
    with _con() as c: return c.execute("DELETE FROM events WHERE t < ?", (time.time() - days * 86400,)).rowcount

async def loop(cfg, on_cycle=None):
    """The collector: every collect_s pulls every node's new events, every state_s their state, hourly deletes events older than retention_days. cfg() returns the current settings; on_cycle (async) runs after each cycle (plans, schedules)."""
    last_state, last_prune = 0.0, 0.0
    STATUS["running"] = True
    while True:
        c, t0 = cfg(), time.time()
        try:
            ns = nodes()
            await asyncio.gather(*(collect_events(n) for n in ns))
            if time.time() - last_state >= float(c["state_s"]): await asyncio.gather(*(collect_state(n) for n in ns)); last_state = time.time()
            if time.time() - last_prune >= 3600: await asyncio.to_thread(prune, float(c["retention_days"])); last_prune = time.time()
            if on_cycle: await on_cycle()
            STATUS.update(last_cycle=time.time(), cycle_s=round(time.time() - t0, 2), error="")
        except Exception as e:
            STATUS["error"] = f"{type(e).__name__}: {e}"[:300]
            print(f"""[fleet] collector cycle failed: {STATUS["error"]}""", flush=True)
        await asyncio.sleep(max(1.0, float(c["collect_s"]) - (time.time() - t0)))

# --- Actions record ---

def record_action(node: str, source: str, policy: str, action: str, args: dict, why: str, value: float, mode: str, status: str, result: dict = None, directive: str = "") -> int:
    return x("INSERT INTO actions (t, node, source, policy, action, args, why, value, mode, status, result, directive) VALUES (?,?,?,?,?,?,?,?,?,?,?,?)", (time.time(), node, source, policy, action, json.dumps(args, ensure_ascii=False), why, value, mode, status, json.dumps(result or {}, ensure_ascii=False), directive))

def update_action(n: int, status: str, result: dict = None, directive: str = ""): x("UPDATE actions SET status = ?, result = ?, directive = COALESCE(NULLIF(?, ''), directive) WHERE n = ?", (status, json.dumps(result or {}, ensure_ascii=False), directive, n))

def actions(node: str = "", limit: int = 200) -> list:
    rows = q("SELECT * FROM actions WHERE (? = '' OR node = ?) ORDER BY n DESC LIMIT ?", (node, node, limit))
    return [{**r, "args": json.loads(r["args"] or "{}"), "result": json.loads(r["result"] or "{}")} for r in rows]
