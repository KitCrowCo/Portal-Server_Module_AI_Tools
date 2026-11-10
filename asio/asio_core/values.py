"""asio/values.py - every number in the catalog says how sure it is. A value is a plain number, or a spec with tiers:
  {"theory": ceiling from physics or a datasheet, "est": a distribution, "measured": {"values": [...], "src", "date"}, "better": "high" | "low", "improve": factor distribution, "plateau": value, "group": shared draw, "unit", "src", "date"}
est forms: [lo, mode, hi] (triangular); {"normal": [mean, sd]}; {"lognormal": [median, geometric sd]}; {"uniform": [lo, hi]}.
A Valuer reads values for one run: which tier ("auto" = measured, else est, else theory; or one named) and which case:
  sample - a random draw (Monte Carlo); median; worst / best - the 5th or 95th percentile on the unfavorable or favorable side; theory - the ceiling;
  improved - median times the improvement we are 90% sure of; plateau - where it is expected to stop improving.
Each key draws once per run from its own seeded stream, so the same key gets the same luck in every plan evaluated with that seed (common random numbers). "group" makes several keys share one draw."""
import zlib, fnmatch
from statistics import NormalDist
import numpy as np

CASES = ("sample", "median", "worst", "best", "theory", "improved", "plateau")
TIERS = ("auto", "measured", "estimate", "theory")

def _q(est, q: float) -> float:
    """Quantile of an est form."""
    if isinstance(est, (list, tuple)):
        lo, m, hi = map(float, est)
        if hi <= lo: return lo
        c = (m - lo) / (hi - lo)
        return lo + np.sqrt(q * (hi - lo) * (m - lo)) if q < c else hi - np.sqrt((1 - q) * (hi - lo) * (hi - m))
    (k, a), = est.items()
    if k == "normal": return NormalDist(a[0], a[1]).inv_cdf(min(max(q, 1e-9), 1 - 1e-9))
    if k == "lognormal": return float(a[0] * np.exp(np.log(a[1]) * NormalDist().inv_cdf(min(max(q, 1e-9), 1 - 1e-9))))
    if k == "uniform": return a[0] + q * (a[1] - a[0])
    raise ValueError(f"unknown distribution {k}")

class Valuer:
    def __init__(self, case: str = "median", tier: str = "auto", seed: int = 0, overrides: dict = None):
        if case not in CASES: raise ValueError(f"case must be one of {CASES}")
        if tier not in TIERS: raise ValueError(f"tier must be one of {TIERS}")
        self.case, self.tier, self.seed, self.overrides, self.draws, self.read = case, tier, seed, overrides or {}, {}, {}

    def _override(self, key: str):
        for pat, v in self.overrides.items():
            if key == pat or fnmatch.fnmatchcase(key, pat): return v
        return None

    def __call__(self, spec, key: str) -> float:
        """The value of key for this run. Every key read is kept in self.read with its catalog spec and any override, so a page can show what a run used and where each number came from."""
        o = self._override(key)
        self.read.setdefault(key, {"spec": spec, "override": o})
        if o is not None: spec = o
        if spec is None: raise KeyError(f"no value for {key}")
        if isinstance(spec, (int, float)) and not isinstance(spec, bool): return float(spec)
        if not isinstance(spec, dict): raise TypeError(f"{key}: a value is a number or a spec, not {spec!r}")
        dk = spec.get("group") or key
        if dk not in self.draws: self.draws[dk] = self._value(spec, dk)
        return self.draws[dk]

    def _tier(self, spec: dict):
        order = {"auto": ("measured", "est", "theory"), "measured": ("measured", "est", "theory"), "estimate": ("est", "theory", "measured"), "theory": ("theory", "est", "measured")}[self.tier]
        for k in order:
            if spec.get(k) is not None: return k, spec[k]
        raise ValueError(f"empty value spec {spec}")

    def _quant(self, kind, v, q: float) -> float:
        if kind == "theory": return float(v)
        if kind == "measured": return float(np.percentile(v["values"], q * 100))
        return float(_q(v, q))

    def _value(self, spec: dict, key: str) -> float:
        high = spec.get("better", "high") == "high"
        if self.case == "theory" and spec.get("theory") is not None: return float(spec["theory"])
        if self.case == "plateau" and spec.get("plateau") is not None: return float(spec["plateau"])
        kind, v = self._tier(spec)
        if self.case == "sample":
            if kind == "theory": return float(v)
            rng = np.random.default_rng([self.seed, zlib.crc32(key.encode())])
            return float(rng.choice(v["values"])) if kind == "measured" else float(_q(v, rng.random()))
        if self.case in ("worst", "best", "theory", "plateau"): return self._quant(kind, v, 0.95 if (self.case == "worst") != high else 0.05)
        base = self._quant(kind, v, 0.5)
        if self.case == "improved" and spec.get("improve") is not None: return base * float(_q(spec["improve"], 0.1 if high else 0.9))
        return base

def describe(spec) -> dict:
    """A value's quantiles and tiers, for showing where a number came from."""
    if not isinstance(spec, dict): return {"value": spec}
    out = {k: spec[k] for k in ("theory", "plateau", "better", "unit", "src", "date", "group") if k in spec}
    if spec.get("est") is not None: out["est"] = {f"p{int(q * 100)}": round(float(_q(spec["est"], q)), 6) for q in (0.05, 0.5, 0.95)}
    if spec.get("measured"): out["measured"] = {"n": len(spec["measured"]["values"]), "p50": float(np.median(spec["measured"]["values"])), **{k: spec["measured"][k] for k in ("src", "date") if k in spec["measured"]}}
    return out
