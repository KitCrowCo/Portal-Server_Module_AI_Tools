"""asio_ui/workspace.py - each user's Asio workspace on the server, so nothing is lost moving between tabs, reloading or restarting the portal.
Layout under data/ai_tools/asio/workspace/<user>/:
  ui.json                     page state: the open scenario, the active part of each tab, the last inputs of every form
  scenarios/<name>/draft.json the working copy, saved on every change
  scenarios/<name>/versions.jsonl  full copies, appended at each checkpoint (a run, Save version, before a restore); never rewritten
  results/<id>.json           every run's result, with the scenario version it ran; results/index.jsonl lists them (appended only)
  trash/                      deleted scenarios and results, moved here rather than removed
Names are kept free rather than replaced: a new scenario named like an existing one becomes name-2."""
import json, re, time, uuid, shutil
from pathlib import Path

def safe(s: str) -> str: return re.sub(r"[^A-Za-z0-9_.-]", "_", str(s or "")).strip("._")[:64] or "unnamed"

def _plain(o):
    """numpy numbers and booleans in results, as plain JSON values."""
    if hasattr(o, "item"): return o.item()
    raise TypeError(f"{type(o).__name__} is not JSON serializable")

class Workspace:
    def __init__(self, root: Path, user: str):
        self.root = Path(root) / safe(user)
        self.sdir, self.rdir = self.root / "scenarios", self.root / "results"
        for d in (self.sdir, self.rdir, self.root / "trash"): d.mkdir(parents=True, exist_ok=True)

    # --- Page state ---

    def ui(self) -> dict:
        f = self.root / "ui.json"
        return json.loads(f.read_text()) if f.exists() else {}

    def set_ui(self, **kw) -> dict:
        u = {**self.ui(), **kw}
        (self.root / "ui.json").write_text(json.dumps(u, indent=1))
        return u

    def form(self, name: str) -> dict: return self.ui().get("forms", {}).get(name, {})
    def set_form(self, name: str, values: dict): self.set_ui(forms={**self.ui().get("forms", {}), name: {**self.form(name), **values}})

    # --- Scenarios ---

    def names(self) -> list:
        """Scenario names, most recently changed first."""
        return [d.name for d in sorted((d for d in self.sdir.iterdir() if (d / "draft.json").exists()), key=lambda d: -(d / "draft.json").stat().st_mtime)]

    def exists(self, name: str) -> bool: return (self.sdir / safe(name) / "draft.json").exists()
    def get(self, name: str) -> dict: return json.loads((self.sdir / safe(name) / "draft.json").read_text())
    def changed(self, name: str) -> float: return (self.sdir / safe(name) / "draft.json").stat().st_mtime
    def put(self, name: str, sc: dict): (self.sdir / safe(name) / "draft.json").write_text(json.dumps(sc, indent=1))

    def free(self, name: str) -> str:
        base, n = safe(name), 2
        out = base
        while (self.sdir / out).exists(): out, n = f"{base}-{n}", n + 1
        return out

    def create(self, name: str, sc: dict, why: str) -> str:
        """A new scenario under a free name, with its first version; returns the name used."""
        name = self.free(name)
        (self.sdir / name).mkdir(parents=True)
        self.put(name, sc)
        self.checkpoint(name, why)
        return name

    def checkpoint(self, name: str, why: str) -> int:
        """Appends the draft as a version unless it equals the last one; returns the version count."""
        vs = self.versions(name)
        sc = self.get(name)
        if vs and self.version(name, len(vs) - 1) == sc: return len(vs)
        with open(self.sdir / safe(name) / "versions.jsonl", "a") as f: f.write(json.dumps({"t": time.time(), "why": why, "sc": sc}) + "\n")
        return len(vs) + 1

    def _vlines(self, name: str) -> list:
        f = self.sdir / safe(name) / "versions.jsonl"
        return [json.loads(x) for x in f.read_text().splitlines() if x.strip()] if f.exists() else []

    def versions(self, name: str) -> list: return [{"i": i, "t": v["t"], "why": v["why"]} for i, v in enumerate(self._vlines(name))]
    def version(self, name: str, i: int) -> dict: return self._vlines(name)[i]["sc"]

    def restore(self, name: str, i: int):
        """The draft becomes version i; the draft it replaces is kept as a version first."""
        self.checkpoint(name, "before restoring an earlier version")
        self.put(name, self.version(name, i))
        self.checkpoint(name, f"restored version {i + 1}")

    def rename(self, name: str, new: str) -> str:
        new = self.free(new)
        (self.sdir / safe(name)).rename(self.sdir / new)
        return new

    def trash(self, name: str):
        """Moves a scenario (draft and every version) to trash/ under a time-stamped name."""
        shutil.move(str(self.sdir / safe(name)), str(self.root / "trash" / f"""{safe(name)}.{time.strftime("%Y%m%d-%H%M%S")}"""))

    # --- Results ---

    def add_result(self, scenario: str, kind: str, inputs: dict, data: dict) -> str:
        rid = f"""{time.strftime("%Y%m%d-%H%M%S")}-{uuid.uuid4().hex[:6]}"""
        head = {"id": rid, "t": time.time(), "scenario": scenario, "version": len(self.versions(scenario)) if self.exists(scenario) else None, "kind": kind, "head": data.get("head", ""), **({"best": data["best"]} if "best" in data else {})}
        (self.rdir / f"{rid}.json").write_text(json.dumps({**head, "inputs": inputs, "data": data}, default=_plain))
        with open(self.rdir / "index.jsonl", "a") as f: f.write(json.dumps(head) + "\n")
        return rid

    def result(self, rid: str) -> dict: return json.loads((self.rdir / f"{safe(rid)}.json").read_text())

    def results(self, scenario: str = None, kind: str = None, limit: int = 40) -> list:
        """Newest first, from the index; a result moved to trash is left out."""
        f = self.rdir / "index.jsonl"
        rows = [json.loads(x) for x in f.read_text().splitlines() if x.strip()] if f.exists() else []
        return [r for r in reversed(rows) if (scenario is None or r["scenario"] == scenario) and (kind is None or r["kind"] in kind.split("|")) and (self.rdir / f"""{r["id"]}.json""").exists()][:limit]

    def trash_result(self, rid: str): shutil.move(str(self.rdir / f"{safe(rid)}.json"), str(self.root / "trash" / f"result-{safe(rid)}.json"))
