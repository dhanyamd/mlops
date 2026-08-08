"""Provider registry — the single place that maps a source name to a provider.

Both the batch flows and the streaming producer build providers *by name* from
here, so no caller ever hardcodes a venue. Construction differs per provider
(Alpaca needs API keys, Yahoo a cache dir), which is why the factory lives here
instead of in each caller.
"""

from __future__ import annotations

from config.settings import Settings
from ingest.providers.alpaca import AlpacaBarProvider
from ingest.providers.base import BarProvider
from ingest.providers.binance import BinanceBarProvider
from ingest.providers.synthetic import SyntheticBarProvider
from ingest.providers.yahoo import YahooBarProvider

PROVIDERS: dict[str, type[BarProvider]] = {
    "yahoo": YahooBarProvider,
    "binance": BinanceBarProvider,
    "alpaca": AlpacaBarProvider,
    "synthetic": SyntheticBarProvider,
}


def build_bar_provider(provider_name: str, settings: Settings) -> BarProvider:
    """Build a provider by name, validating keys/config from Settings."""
    provider_cls = PROVIDERS.get(provider_name)
    if provider_cls is None:
        raise ValueError(
            f"unknown bar provider: {provider_name!r} (choose from {sorted(PROVIDERS)})"
        )
    if provider_name == "yahoo":
        return YahooBarProvider(cache_dir=settings.yahoo_cache_dir)
    if provider_name == "alpaca":
        if (
            not settings.ingest_provider_alpaca_api_key
            or not settings.ingest_provider_alpaca_secret_key
        ):
            raise ValueError(
                "provider 'alpaca' needs INGEST_PROVIDER_ALPACA_API_KEY and "
                "INGEST_PROVIDER_ALPACA_SECRET_KEY set (free alpaca.markets "
                "account); the keyless default (yahoo) is used otherwise"
            )
        return AlpacaBarProvider(
            api_key=settings.ingest_provider_alpaca_api_key,
            api_secret=settings.ingest_provider_alpaca_secret_key,
        )
    return provider_cls()
