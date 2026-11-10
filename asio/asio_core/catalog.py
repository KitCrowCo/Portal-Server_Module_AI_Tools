"""asio_core/catalog.py - the swappable data: every JSON file in the catalog folders, merged by collection and id, later folders over earlier ones.
A file is {"<collection>": {"<id>": {...}}}, any collections in any files, so new hardware, models or workloads are a file dropped in, and a measurement is a file that overrides one field.
An entry with "extends": "<id>" (or a list) starts from those entries of the same collection. Keys starting with "_" are notes, kept but never read as values."""
import json, copy
from pathlib import Path

BUILTIN = Path(__file__).with_name("catalog")
COLLECTIONS = ("nodes", "links", "backends", "models", "apis", "economics", "workloads", "scenarios", "heat")

def merge(a: dict, b: dict) -> dict:
    out = copy.deepcopy(a)
    for k, v in b.items(): out[k] = merge(out[k], v) if isinstance(v, dict) and isinstance(out.get(k), dict) and not _leaf(v) else copy.deepcopy(v)
    return out

def _leaf(v: dict) -> bool: return any(k in v for k in ("theory", "est", "measured"))   # a value spec replaces, it does not merge

class Catalog:
    def __init__(self, *dirs):
        self.dirs, self.data, self.files = [BUILTIN, *map(Path, dirs)], {c: {} for c in COLLECTIONS}, []
        for d in self.dirs:
            for f in sorted(d.rglob("*.json")) if d.exists() else []:
                for coll, items in json.loads(f.read_text()).items():
                    if coll.startswith("_"): continue
                    if coll not in self.data: raise ValueError(f"{f}: unknown collection {coll} (known: {COLLECTIONS})")
                    for i, e in items.items(): self.data[coll][i] = merge(self.data[coll].get(i, {}), e)
                self.files.append(str(f))

    def get(self, coll: str, id_: str, _seen=()) -> dict:
        try: e = self.data[coll][id_]
        except KeyError: raise KeyError(f"{coll} has no {id_!r} (has: {sorted(self.data[coll])})")
        if id_ in _seen: raise ValueError(f"{coll}.{id_} extends itself")
        bases = e.get("extends") or []
        out = {}
        for b in [bases] if isinstance(bases, str) else bases: out = merge(out, self.get(coll, b, (*_seen, id_)))
        return {**merge(out, {k: v for k, v in e.items() if k != "extends"}), "id": id_}

    def ids(self, coll: str, concrete: bool = True) -> list: return sorted(i for i, e in self.data[coll].items() if not (concrete and e.get("abstract")))
    def add(self, coll: str, id_: str, entry: dict): self.data[coll][id_] = merge(self.data[coll].get(id_, {}), entry)
