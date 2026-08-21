"""Local, serverless verification of the FAS+SMB strategy against the Iceberg lake.

Why DuckDB: it is an embedded OLAP engine that reads Apache Iceberg tables directly
from MinIO (no server, no Snowflake round-trip). That lets us independently
recompute the ``asym`` (FAS_avg + SMB) cross-sectional Sharpe from the GOLD
features mart on-laptop — the user's "is the Sharpe actually real?" question — using
the exact method as ``scripts/research_broad.py::metrics``.

The lake tier is optional (``LAKE_ENABLED``). This script probes MinIO, loads the
GOLD features Iceberg table, prints its schema, and (if a signal column and a
forward-return column are present) recomputes the strategy Sharpe. Column detection
is heuristic so nothing is hardcoded; if the columns aren't found it prints the
schema and stops.

Usage:
    uv sync --extra lake
    uv run --extra lake python scripts/lake_sharpe_verify.py [--table gold.features] [--ppy 52]
"""

from __future__ import annotations

import argparse

import numpy as np
import pandas as pd

from config.logging import configure_logging, get_logger
from config.settings import get_settings
from flows.lake_export import catalog_properties

logger = get_logger(__name__)

# 10 bps round-trip is the cost assumption used across the backtests.
COST_BPS = 10.0


def _host(endpoint: str) -> str:
    return endpoint.replace("https://", "").replace("http://", "")


def _load_iceberg_df(table: str, settings) -> pd.DataFrame:
    """Return the Iceberg table as a DataFrame via DuckDB + the PyIceberg catalog.

    The catalog (SqlCatalog over sqlite) gives us the current metadata file
    location; DuckDB's iceberg extension reads the Parquet data straight from
    MinIO. No hardcoded bucket/table paths — everything comes from Settings.
    """
    try:
        from pyiceberg.catalog.sql import SqlCatalog
    except ImportError:  # pragma: no cover - guarded by the lake extra
        raise SystemExit("pyiceberg not installed; run `uv sync --extra lake`")

    namespace, _, table_name = table.partition(".")
    catalog = SqlCatalog("quant_lake", **catalog_properties(settings))
    ident = (
        (namespace, table_name)
        if table_name
        else (settings.lake_namespace, settings.lake_table_features)
    )
    try:
        tbl = catalog.load_table(ident)
    except Exception as exc:  # noqa: BLE001 - surface a clear "table missing" message
        available = [f"{n}.{t}" for (n, t) in catalog.list_tables()]
        raise SystemExit(
            f"could not load Iceberg table {ident} ({exc}). Available tables: {available or 'none'}"
        )
    metadata_location = tbl.metadata_location
    logger.info("iceberg_metadata", table=ident, metadata_location=metadata_location)

    import duckdb

    con = duckdb.connect()
    con.execute("INSTALL iceberg; LOAD iceberg;")
    con.execute("INSTALL httpfs; LOAD httpfs;")
    con.execute("SET s3_endpoint = ?", [_host(settings.lake_endpoint)])
    con.execute("SET s3_access_key_id = ?", [settings.lake_access_key or "minioadmin"])
    con.execute("SET s3_secret_access_key = ?", [settings.lake_secret_key or "minioadmin"])
    con.execute("SET s3_region = ?", [settings.lake_region])
    con.execute("SET s3_url_style = 'path';")
    return con.execute(
        f"SELECT * FROM iceberg_scan('{metadata_location}', allow_moved_paths=true)"
    ).df()


def _detect_columns(columns: list[str]) -> tuple[str | None, str | None, str | None]:
    """Heuristically pick the signal, forward-return and regime columns.

    Nothing is hardcoded: we match by name fragments so the script adapts to
    whatever the GOLD mart actually contains.
    """
    cols = {c.lower(): c for c in columns}
    sig = next(
        (
            c
            for k, c in cols.items()
            if any(f in k for f in ("asym", "fas", "smb", "signal", "score", "zscore"))
        ),
        None,
    )
    fwd = next(
        (
            c
            for k, c in cols.items()
            if any(f in k for f in ("fwd", "forward", "ret_1", "future_ret", "realized", "target"))
        ),
        None,
    )
    regime = next((c for k, c in cols.items() if "regime" in k), None)
    return sig, fwd, regime


def _cross_sectional_returns(
    df: pd.DataFrame, sig_col: str, fwd_col: str, regime_col: str | None, periods_per_year: int
) -> pd.Series:
    """Replicate scripts/research_broad.py::backtest (quintile L/S, 10 bps)."""
    need = ["ts", "symbol", sig_col, fwd_col]
    if regime_col:
        need.append(regime_col)
    missing = [c for c in need if c not in df.columns]
    if missing:
        raise SystemExit(f"lake missing required columns: {missing}")

    df = df.copy()
    df["ts"] = pd.to_datetime(
        df["ts"], unit="ms" if str(df["ts"].dtype).startswith("int") else None
    )
    wide_sig = df.pivot(index="ts", columns="symbol", values=sig_col)
    wide_ret = df.pivot(index="ts", columns="symbol", values=fwd_col)
    regime = df.pivot(index="ts", columns="symbol", values=regime_col) if regime_col else None
    dates = wide_sig.index
    ret: list[float] = []
    prev = None
    for date in dates:
        s = wide_sig.loc[date].dropna()
        if regime is not None and (
            date not in regime.index or float(regime.loc[date].dropna().mean()) <= 0
        ):
            ret.append(0.0)
            prev = None
            continue
        if len(s) < 10:
            ret.append(0.0)
            prev = None
            continue
        q = s.rank(pct=True)
        longs = q[q > 0.8].index
        shorts = q[q < 0.2].index
        w = pd.Series(0.0, index=s.index)
        w[longs] = 1.0 / max(1, len(longs))
        w[shorts] = -1.0 / max(1, len(shorts))
        fr = wide_ret.loc[date].reindex(w.index).fillna(0.0)
        r = float((w * fr).sum())
        if prev is not None:
            turn = float((w.reindex(prev.index).fillna(0) - prev).abs().sum())
            r -= COST_BPS / 1e4 * turn
        ret.append(r if np.isfinite(r) else 0.0)
        prev = w
    return pd.Series(ret, index=dates)


def _metrics(ret: pd.Series, periods_per_year: int) -> dict:
    """Exact port of research_broad.py::metrics (bootstrap CI, annualized)."""
    ret = ret.dropna()
    n = len(ret)
    if n < 20:
        return {"n": n, "sharpe": float("nan")}
    ann = ret.mean() * periods_per_year
    vol = ret.std() * np.sqrt(periods_per_year)
    sharpe = ann / vol if vol > 0 else 0.0
    wealth = (1 + ret).cumprod()
    dd = float((wealth / wealth.cummax() - 1).min())
    rng = np.random.default_rng(0)
    vals = ret.values
    boot = []
    for _ in range(1000):
        sm = rng.choice(vals, n).sum() / n * periods_per_year
        sd = rng.choice(vals, n).std() * np.sqrt(periods_per_year)
        boot.append(sm / sd if sd > 0 else 0.0)
    ci = (float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5)))
    return {
        "n": n,
        "ann": float(ann),
        "vol": float(vol),
        "sharpe": float(sharpe),
        "ci": ci,
        "maxdd": dd,
        "pct_flat": float((ret == 0).mean()),
    }


def main() -> None:
    configure_logging()
    ap = argparse.ArgumentParser()
    ap.add_argument("--table", default=f"{get_settings().lake_namespace}.features")
    ap.add_argument(
        "--ppy", type=int, default=52, help="periods per year for annualization (52=weekly)"
    )
    args = ap.parse_args()

    settings = get_settings()
    logger.info("lake_verify_start", table=args.table, endpoint=settings.lake_endpoint)

    df = _load_iceberg_df(args.table, settings)
    print(f"\n== {args.table} ==  rows={len(df)}  cols={len(df.columns)}")
    if "ts" in df.columns:
        ts = pd.to_datetime(df["ts"], unit="ms" if str(df["ts"].dtype).startswith("int") else None)
        print(
            f"   span: {ts.min()} -> {ts.max()}  symbols={df['symbol'].nunique() if 'symbol' in df.columns else '?'}"
        )

    sig, fwd, regime = _detect_columns(list(df.columns))
    print(f"   detected signal='{sig}' forward_return='{fwd}' regime='{regime}'")
    if not (sig and fwd):
        print("\nSchema introspection only (no signal/forward-return column detected):")
        for c in df.columns:
            print(f"   - {c} ({df[c].dtype})")
        return

    rets = _cross_sectional_returns(df, sig, fwd, regime, args.ppy)
    m = _metrics(rets, args.ppy)
    if "sharpe" not in m or np.isnan(m["sharpe"]):
        print(f"insufficient data after pivoting (n={m.get('n')})")
        return
    print(
        f"\n   FAS+SMB Sharpe={m['sharpe']:.2f}  CI[{m['ci'][0]:.2f},{m['ci'][1]:.2f}]  "
        f"ann={m['ann'] * 100:5.1f}%  vol={m['vol'] * 100:4.1f}%  "
        f"maxDD={m['maxdd'] * 100:6.1f}%  %flat={m['pct_flat'] * 100:.0f}%  (n={m['n']})"
    )


if __name__ == "__main__":
    main()
