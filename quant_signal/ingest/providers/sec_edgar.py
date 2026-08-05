"""SEC EDGAR fundamentals provider — real external source, no API key.

Pulls XBRL company facts (us-gaap) and extracts the FULL history of annual
filings — every 10-K/20-F with its SEC ``filed`` date — so features can be
built point-in-time (as-of ``filed_at``) with NO lookahead bias. This is the
correctness gap the platform flags in ``docs/architecture.md``.

Two things are NOT hardcoded:

- The ticker→CIK mapping is loaded from SEC's official keyless
  ``company_tickers.json`` registry (~10k companies).
- The XBRL concept mapping (which us-gaap concepts a canonical metric like
  "Revenues" is tagged under) is REFERENCE DATA in ``us_gaap_concepts.json``.
  Different filers tag the same economic metric under different concepts, and
  even a single filer switches concepts over time (e.g. AAPL: SalesRevenueNet
  pre-2018, Revenues in FY2018, RevenueFromContractWithCustomerExcludingAssessedTax
  post-2019). We UNION across every candidate concept for a metric and keep
  every annual filing as its own row keyed by
  (ticker, metric, fiscal_year, filed_at) so restatements (10-K/A) survive.

Two SEC data quirks, verified live, are handled by the extraction:

- A single 10-K/20-F contains comparative prior-period rows (1-2 years back)
  alongside the fiscal-year figure. The ANNUAL value is the row whose ``end``
  equals the fiscal-year end — the max ``end`` within the filing (e.g. AAPL's
  original FY2009 10-K reported $36.5B revenue; the 10-K/A restated it to
  $42.9B — both rows are kept, each with its own ``filed`` date).
- SEC's ``fy`` label can disagree with the period's ``end`` date (NVDA's old
  filings are off-by-one), so ``fiscal_year`` is derived from the ``end`` date.

Output columns: ``ticker, cik, metric, fiscal_year, value, unit, filed_at,
loaded_at``.
"""

from __future__ import annotations

import datetime as dt
import json
import time
from pathlib import Path

import pandas as pd
import requests

from config.settings import csv_list, get_settings

_SEC_BASE = "https://data.sec.gov/api/xbrl/companyfacts"
# SEC's official ticker→CIK registry, updated daily by the agency.
_SEC_TICKER_MAP_URL = "https://www.sec.gov/files/company_tickers.json"
# SEC fair-access policy: max 10 requests/second per IP. 0.15s = ~6.7 req/s,
# well under the limit and polite.
_REQUEST_INTERVAL_S = 0.15
# Canonical metric -> candidate us-gaap concepts (reference data, not logic).
_CONCEPT_MAP_PATH = Path(__file__).parent / "us_gaap_concepts.json"


def _load_concept_map() -> dict[str, list[str]]:
    return json.loads(_CONCEPT_MAP_PATH.read_text())


def _is_annual_filing(entry: dict) -> bool:
    """True for a fiscal-year filing (10-K/20-F), never a quarterly one."""
    if entry.get("fp") != "FY" or entry.get("val") is None:
        return False
    form = str(entry.get("form") or "")
    return form.startswith("10-K") or form.startswith("20-F")


def _extract_annual_filings(facts: dict, concepts: list[str]) -> list[tuple[int, float, dt.date]]:
    """Union of annual-filing values across candidate concepts.

    Returns ``(fiscal_year, value, filed_date)`` per annual filing, one row per
    SEC filing. Two structural facts about EDGAR's data drive the rules:

    - A 10-K/20-F contains comparative prior-period rows (1-2 years back) in
      addition to the fiscal-year figure. The ANNUAL value is the row whose
      ``end`` equals the fiscal-year end, i.e. the max ``end`` within the
      filing (``fp='FY'`` keeps only fiscal-year rows; ``10-Q`` is excluded).
      Each filing contributes EXACTLY ONE row — its fiscal-year-end value —
      and never one per comparative period.
    - Filers switch XBRL concepts over time (e.g. AAPL revenue: SalesRevenueNet
      -> Revenues -> RevenueFromContractWithCustomerExcludingAssessedTax), so
      we union across every candidate concept. When a filing appears under more
      than one candidate, the earliest candidate in ``concepts`` wins — that
      precedence is reference data (e.g. SalesRevenueNet total revenue before
      SalesRevenueGoodsNet product-only revenue).

    ``fiscal_year`` is derived from the fiscal-year-end date (``year(end)``),
    not SEC's ``fy`` label, because ``fy`` can disagree with ``end`` for some
    filings (NVDA's older 10-Ks are off-by-one). Restatements filed on a later
    date are kept as their own rows (point-in-time correct); the same filing
    surfacing the same value multiple times is deduplicated on
    (filed_date, accession).
    """
    gaap = facts.get("facts", {}).get("us-gaap", {})
    merged: dict[tuple[str, str], dict] = {}
    for concept in concepts:
        key = next((k for k in gaap if k.rsplit(":", 1)[-1] == concept), None)
        if key is None:
            continue
        units = gaap[key].get("units", {})
        # Prefer the primary currency unit when present (raw USD, not per-share).
        unit_keys = ["USD"] if "USD" in units else list(units.keys())
        best: dict[tuple[str, str], dict] = {}
        for unit in unit_keys:
            for entry in units.get(unit, []):
                if not _is_annual_filing(entry):
                    continue
                filed = str(entry.get("filed") or "")
                end = str(entry.get("end") or "")
                if not filed or not end:
                    continue
                filing = (filed[:10], str(entry.get("accn") or ""))
                if filing not in best or end > best[filing].get("end", ""):
                    best[filing] = entry
        for filing, entry in best.items():
            # First candidate to claim a filing owns it (precedence order).
            merged.setdefault(filing, entry)
    rows = [
        (int(entry["end"][:4]), float(entry["val"]), dt.date.fromisoformat(filed))
        for (filed, _accn), entry in merged.items()
    ]
    return sorted(rows, key=lambda r: (r[2], r[0]))


class EdgarFundamentalsProvider:
    name = "sec_edgar"

    def __init__(self, user_agent: str | None = None, timeout: int = 30) -> None:
        # SEC will throttle/block requests without a real UA. Prefer env; the
        # fallback is clearly marked as non-production.
        self._user_agent = (
            user_agent
            or get_settings().edgar_user_agent
            or ("quant-signal-learning learn@example.com")
        )
        self._timeout = timeout
        self._ticker_to_cik: dict[str, str] | None = None
        self._concept_map = _load_concept_map()

    def _headers(self) -> dict[str, str]:
        # requests sets the Host header from the URL automatically. SEC vhosts
        # www.sec.gov and data.sec.gov, so a manually pinned Host header caused
        # 404s on the wrong vhost. Only the descriptive UA is required.
        return {
            "User-Agent": self._user_agent,
            "Accept-Encoding": "gzip, deflate",
        }

    def _fetch_ticker_map(self) -> dict[str, str]:
        """Load the official SEC ticker→CIK registry (keyless, ~10k companies)."""
        resp = requests.get(_SEC_TICKER_MAP_URL, headers=self._headers(), timeout=self._timeout)
        resp.raise_for_status()
        return {
            row["ticker"].upper(): str(row["cik_str"]).zfill(10) for row in resp.json().values()
        }

    def _ticker_map(self) -> dict[str, str]:
        if self._ticker_to_cik is None:
            self._ticker_to_cik = self._fetch_ticker_map()
        return self._ticker_to_cik

    def _fetch_facts(self, cik: str) -> dict:
        resp = requests.get(
            f"{_SEC_BASE}/CIK{cik}.json", headers=self._headers(), timeout=self._timeout
        )
        resp.raise_for_status()
        return resp.json()

    def fetch_facts(
        self,
        tickers: list[str],
        metrics: list[str] | None = None,
    ) -> pd.DataFrame:
        """Full annual-filing timeline per (ticker, metric); empty if none found."""
        metric_list = metrics or csv_list(get_settings().ingest_default_metrics)
        ticker_map = self._ticker_map()
        now = dt.datetime.now(dt.timezone.utc).replace(tzinfo=None)
        rows: list[tuple] = []
        for ticker in tickers:
            cik = ticker_map.get(ticker.upper())
            if not cik:
                continue
            facts = self._fetch_facts(cik)
            time.sleep(_REQUEST_INTERVAL_S)  # stay well under SEC's rate limit
            for metric in metric_list:
                for fy, value, filed_at in _extract_annual_filings(
                    facts, self._concept_map.get(metric, [])
                ):
                    rows.append((ticker.upper(), cik, metric, fy, value, "USD", filed_at, now))
        return pd.DataFrame(
            rows,
            columns=[
                "ticker",
                "cik",
                "metric",
                "fiscal_year",
                "value",
                "unit",
                "filed_at",
                "loaded_at",
            ],
        )
