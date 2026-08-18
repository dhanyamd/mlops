"""Append-only registry of every strategy variant ever evaluated.

WHY THIS EXISTS
---------------
The Deflated Sharpe Ratio (Bailey & Lopez de Prado 2014) deflates an observed
Sharpe by the expected maximum Sharpe across N zero-skill trials. Both inputs --
N and the spread of Sharpes across those trials -- describe the SEARCH, not the
strategy. They cannot be read off the winning backtest.

Which means they are normally supplied by the researcher from memory. That is
precisely the failure mode DSR was invented to detect: Lopez de Prado's central
claim is that researchers systematically UNDERCOUNT their own trials, without
any intent to deceive. A hand-typed N in a paper whose defence is "we corrected
for multiple testing" is circular -- it assumes the number the correction exists
to establish.

So every configuration that gets evaluated is logged HERE, by the code that
evaluates it, and DSR reads N and sd from this file. The researcher never types
either number.

DEDUPLICATION
-------------
A trial is a distinct CONFIGURATION, not a distinct execution. Re-running the
same config after a bug fix is one trial, not two; counting it twice would
inflate N and make the DSR look artificially strong (a larger N raises the
hurdle, so over-counting is conservative for DSR but wrong for the sd estimate,
and it corrupts any other use of the registry). Records are therefore keyed by a
hash of the canonicalised config, and the LATEST record for a key wins.

The file is JSONL so it appends atomically, survives interruption, diffs
readably in git, and can be inspected by a referee without running anything.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path

DEFAULT_PATH = Path(__file__).resolve().parent.parent / "research" / "trials" / "srp.jsonl"


def _canonical(config: dict) -> str:
    """Stable text form of a config, so equal configs hash equal.

    Sorted keys, and floats normalised via repr of the float itself -- 0.5 and
    0.50 must not produce different hashes, while 0.5 and 0.6 must.
    """
    def norm(v):
        if isinstance(v, float):
            # collapse -0.0 -> 0.0 and any float that is exactly integral
            if v == int(v):
                return float(int(v))
            return round(v, 12)
        if isinstance(v, (list, tuple)):
            return [norm(x) for x in v]
        if isinstance(v, dict):
            return {k: norm(v[k]) for k in sorted(v)}
        return v

    return json.dumps(norm(config), sort_keys=True, separators=(",", ":"))


def config_hash(config: dict) -> str:
    return hashlib.sha256(_canonical(config).encode()).hexdigest()[:16]


def _git_sha() -> str | None:
    """Code version the trial ran under, so a referee can tie result to source."""
    try:
        out = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True, text=True, timeout=5,
            cwd=Path(__file__).resolve().parent.parent,
        )
        return out.stdout.strip() or None
    except Exception:
        return None


@dataclass(frozen=True)
class Trial:
    family: str
    config_hash: str
    config: dict
    sharpe_weekly: float
    sharpe_ann: float
    n_obs: int
    ts: float
    git_sha: str | None = None
    note: str | None = None


def log_trial(
    family: str,
    config: dict,
    *,
    sharpe_weekly: float,
    n_obs: int,
    periods_per_year: int = 52,
    note: str | None = None,
    path: Path | str = DEFAULT_PATH,
) -> str:
    """Record one evaluated configuration. Returns its config hash.

    Called by the evaluator itself, never by hand. A trial that produced an
    unusable result (too few observations, degenerate book) is still logged --
    a failed search branch is still a search branch, and omitting it is exactly
    the undercount DSR guards against. Such rows carry a non-finite Sharpe and
    are counted in N but excluded from the sd estimate.
    """
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    h = config_hash(config)
    sw = float(sharpe_weekly)
    rec = {
        "family": family,
        "config_hash": h,
        "config": config,
        "sharpe_weekly": sw if math.isfinite(sw) else None,
        "sharpe_ann": sw * math.sqrt(periods_per_year) if math.isfinite(sw) else None,
        "n_obs": int(n_obs),
        "ts": time.time(),
        "git_sha": _git_sha(),
        "note": note,
    }
    with p.open("a") as f:
        f.write(json.dumps(rec) + "\n")
        f.flush()
        os.fsync(f.fileno())
    return h


def load_trials(family: str | None = None, path: Path | str = DEFAULT_PATH) -> list[Trial]:
    """All DISTINCT configurations logged, latest record per config wins."""
    p = Path(path)
    if not p.exists():
        return []
    latest: dict[tuple[str, str], dict] = {}
    for line in p.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            r = json.loads(line)
        except json.JSONDecodeError:
            continue
        if family is not None and r.get("family") != family:
            continue
        key = (r.get("family", ""), r.get("config_hash", ""))
        prev = latest.get(key)
        if prev is None or r.get("ts", 0) >= prev.get("ts", 0):
            latest[key] = r
    out = []
    for r in latest.values():
        sw = r.get("sharpe_weekly")
        sa = r.get("sharpe_ann")
        out.append(
            Trial(
                family=r.get("family", ""),
                config_hash=r.get("config_hash", ""),
                config=r.get("config", {}),
                sharpe_weekly=float("nan") if sw is None else float(sw),
                sharpe_ann=float("nan") if sa is None else float(sa),
                n_obs=int(r.get("n_obs", 0)),
                ts=float(r.get("ts", 0.0)),
                git_sha=r.get("git_sha"),
                note=r.get("note"),
            )
        )
    return sorted(out, key=lambda t: t.ts)


def trial_stats(family: str | None = None, path: Path | str = DEFAULT_PATH) -> dict:
    """N and sd(Sharpe) for DSR -- both MEASURED from the registry.

    ``n_trials`` counts every distinct configuration evaluated, including ones
    that failed to produce a usable series. ``sd_weekly`` is estimated only over
    the finite ones, since a non-finite Sharpe carries no information about
    spread.
    """
    ts = load_trials(family, path)
    finite = [t for t in ts if math.isfinite(t.sharpe_weekly)]
    n = len(ts)
    if not finite:
        return {"n_trials": n, "n_finite": 0, "sd_weekly": float("nan"),
                "best_weekly": float("nan"), "best_config": None, "trials": ts}
    vals = [t.sharpe_weekly for t in finite]
    mean = sum(vals) / len(vals)
    var = sum((v - mean) ** 2 for v in vals) / (len(vals) - 1) if len(vals) > 1 else 0.0
    best = max(finite, key=lambda t: t.sharpe_weekly)
    return {
        "n_trials": n,
        "n_finite": len(finite),
        "sd_weekly": math.sqrt(var),
        "mean_weekly": mean,
        "best_weekly": best.sharpe_weekly,
        "best_config": best.config,
        "best_hash": best.config_hash,
        "trials": ts,
    }


if __name__ == "__main__":  # inspection only
    import argparse

    ap = argparse.ArgumentParser(description="inspect the trial registry")
    ap.add_argument("--family", default=None)
    ap.add_argument("--path", default=str(DEFAULT_PATH))
    a = ap.parse_args()
    st = trial_stats(a.family, a.path)
    print(f"registry: {a.path}")
    print(f"  distinct configurations : {st['n_trials']}")
    print(f"  with a finite Sharpe    : {st['n_finite']}")
    if st["n_finite"]:
        print(f"  sd of trial Sharpes (wk): {st['sd_weekly']:.4f}")
        print(f"  best weekly Sharpe      : {st['best_weekly']:.4f}"
              f"  (ann {st['best_weekly'] * math.sqrt(52):.3f}, {st['best_hash']})")
