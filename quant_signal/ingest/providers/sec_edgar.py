"""SEC EDGAR fundamentals provider — real external source, no API key.

Pulls XBRL company facts (us-gaap) and extracts the latest annual value for a
small set of metrics. SEC requires a descriptive User-Agent ("Sample Company
AdminContact@example.com"); set EDGAR_USER_AGENT in .env.

The ticker→CIK mapping is NOT hardcoded: it is loaded from SEC's official
keyless ``company_tickers.json`` registry (~10k companies, authoritative).

Output columns: ``ticker, cik, metric, fiscal_year, value, unit, loaded_at``.
"""

from __future__ import annotations

import datetime as dt
import time

import pandas as pd
import requests

from config.settings import csv_list, get_settings

_SEC_BASE = "https://data.sec.gov/api/xbrl/companyfacts"
# SEC's official ticker→CIK registry, updated daily by the agency.
_SEC_TICKER_MAP_URL = "https://www.sec.gov/files/company_tickers.json"
# SEC fair-access policy: max 10 requests/second per IP. 0.15s = ~6.7 req/s,
# well under the limit and polite.
_REQUEST_INTERVAL_S = 0.15


def _extract_latest_annual(facts: dict, metric: str) -> tuple[int, float] | None:
    """Return (fiscal_year, value) of the most recent annual filing for ``metric``.

    The JSON nests taxonomy -> concept -> units -> entries; concept names may
    carry a ``us-gaap:`` prefix, so match by suffix. Only USD-denominated
    annual filings count: ``fp == "FY"`` AND a 10-K/20-F form type, so a rare
    FY-tagged quarterly filing can never double-count into the series.
    """
    gaap = facts.get("facts", {}).get("us-gaap", {})
    candidates: list[tuple[int, float]] = []
    for concept, payload in gaap.items():
        if concept.rsplit(":", 1)[-1] != metric:
            continue
        units = payload.get("units", {})
        for entries in units.get("USD", []):
            if entries.get("fp") != "FY" or entries.get("val") is None:
                continue
            form = str(entries.get("form") or "")
            if not (form.startswith("10-K") or form.startswith("20-F")):
                continue
            end = entries.get("end")
            if not end:
                continue
            fy = entries.get("fy") or int(str(end)[:4])
            candidates.append((fy, float(entries["val"])))
    if not candidates:
        return None
    latest_fy = max(fy for fy, _ in candidates)
    best = max(v for fy, v in candidates if fy == latest_fy)
    return latest_fy, best


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
        metrics: tuple[str, ...] | None = None,
    ) -> pd.DataFrame:
        """Latest annual fundamental per (ticker, metric); empty if none found."""
        metric_set = metrics or tuple(csv_list(get_settings().ingest_default_metrics))
        ticker_map = self._ticker_map()
        now = dt.datetime.now(dt.timezone.utc).replace(tzinfo=None)
        rows: list[tuple] = []
        for ticker in tickers:
            cik = ticker_map.get(ticker.upper())
            if not cik:
                continue
            facts = self._fetch_facts(cik)
            time.sleep(_REQUEST_INTERVAL_S)  # stay well under SEC's rate limit
            for metric in metric_set:
                extracted = _extract_latest_annual(facts, metric)
                if extracted is None:
                    continue
                fy, value = extracted
                rows.append((ticker.upper(), cik, metric, fy, value, "USD", now))
        return pd.DataFrame(
            rows,
            columns=["ticker", "cik", "metric", "fiscal_year", "value", "unit", "loaded_at"],
        )
