"""Provider interface — one contract, many sources.

A provider fetches raw market data and returns a pandas DataFrame whose
columns match the contract in ``ingest.schemas``. The orchestration layer
never talks to a vendor directly: swapping a source = adding a provider.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

import pandas as pd


class BarProvider(ABC):
    """Produces equity minute bars with columns:
    ``symbol, ts, open, high, low, close, volume, loaded_at``."""

    name: str = "base"

    @abstractmethod
    def fetch_bars(self, symbols: list[str], days: int) -> pd.DataFrame:
        """Return minute bars for ``symbols`` over the last ``days`` calendar days."""
        raise NotImplementedError
