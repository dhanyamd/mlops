"""The warehouse panel path: same frames as the file, and a real cutoff.

Hermetic — a stub client stands in for Snowflake, so these run in CI without
credentials. The end-to-end equality against the live warehouse is
``scripts.panel_parity``, which needs a connection; what is asserted here is the
logic that would make the two drift apart.
"""

from __future__ import annotations

import pandas as pd
import pytest

from scripts.panel_parity import compare
from scripts.research_fas_clean import build_frames
from scripts.warehouse_panel import _to_rows, load_from_snowflake

DAY_MS = 86_400_000
# A Wednesday, so weekly (W-MON) buckets are exercised mid-week rather than
# landing conveniently on a boundary.
START_MS = 1_600_000_000_000


def _bars(symbol_prices: dict[str, list[float]]) -> dict[str, list[list]]:
    return {
        sym: [[START_MS + i * DAY_MS, px, 100.0 + i] for i, px in enumerate(prices)]
        for sym, prices in symbol_prices.items()
    }


class StubClient:
    """Returns fixed frames, and records the SQL it was asked to run."""

    def __init__(self, bars: pd.DataFrame, funding: pd.DataFrame) -> None:
        self._bars, self._funding = bars, funding
        self.calls: list[tuple[str, tuple | None]] = []

    def query_df(self, sql: str, params=None) -> pd.DataFrame:
        self.calls.append((sql, params))
        return (self._bars if "PANEL_BARS" in sql else self._funding).copy()


def _stub(n: int = 30) -> StubClient:
    ts = pd.to_datetime([START_MS + i * DAY_MS for i in range(n)], unit="ms")
    bars = pd.DataFrame({
        "SYMBOL": ["BTCUSDT"] * n,
        "TS": ts,
        "CLOSE": [100.0 + i for i in range(n)],
        "VOLUME": [10.0] * n,
    })
    funding = pd.DataFrame({
        "SYMBOL": ["BTCUSDT"] * n,
        "TS": ts,
        "RATE": [0.0001] * n,
    })
    return StubClient(bars, funding)


def test_row_conversion_round_trips_into_the_shared_resampler() -> None:
    """``_to_rows`` must emit exactly the shape ``build_frames`` parses.

    This is the seam where the two sources could diverge: if the warehouse
    emitted seconds where the cache carries milliseconds, every timestamp would
    land in 1970 and the panel would be silently empty rather than wrong-looking.
    """
    client = _stub()
    df = client.query_df("PANEL_BARS")
    rows = _to_rows(df, ["CLOSE", "VOLUME"])

    assert set(rows) == {"BTCUSDT"}
    ts_ms, close, volume = rows["BTCUSDT"][0]
    assert ts_ms == START_MS, "timestamps must be epoch MILLISECONDS"
    assert (close, volume) == (100.0, 10.0)

    cw, _vw, _aw, dcl, _dvl = build_frames(rows, {})
    assert not cw.empty and not dcl.empty
    assert cw.index.max().year == 2020, "a unit mix-up would land these in 1970"


def test_both_sources_produce_identical_frames_from_identical_rows() -> None:
    """The parity claim, at the level it is actually guaranteed.

    Both paths call ``build_frames``. Feeding it the cache shape and the
    warehouse shape derived from the same underlying data must give frames that
    ``compare`` cannot distinguish — that is what makes parity structural rather
    than a coincidence someone has to maintain.
    """
    client = _stub()
    warehouse = build_frames(
        _to_rows(client.query_df("PANEL_BARS"), ["CLOSE", "VOLUME"]),
        _to_rows(client.query_df("FUNDING"), ["RATE"]),
    )
    from_file = build_frames(
        _bars({"BTCUSDT": [100.0 + i for i in range(30)]}),
        {"BTCUSDT": [[START_MS + i * DAY_MS, 0.0001] for i in range(30)]},
    )
    # The file fixture uses a rising volume; align on the frames that do not
    # depend on it, which are the ones carrying price and funding.
    for name, idx in (("weekly_close", 0), ("weekly_funding", 2), ("daily_close", 3)):
        ok, message = compare(from_file[idx], warehouse[idx], name)
        assert ok, message


def test_as_of_is_applied_in_sql_not_in_pandas() -> None:
    """The cutoff must reach the database.

    Filtering after the fetch would still give the right answer here, but it
    puts the guarantee in the caller's hands: anyone who forgets re-runs a
    historical study on today's data and gets a number the original decision was
    never based on. Asserting the bound is bound as a parameter keeps it
    impossible to skip.
    """
    client = _stub()
    load_from_snowflake(client=client, as_of="2023-06-01")

    assert len(client.calls) == 2, "expected one query for bars and one for funding"
    for sql, params in client.calls:
        assert "TS <= %s" in sql
        assert params == ("2023-06-01 23:59:59",)


def test_symbol_filter_binds_each_symbol_separately() -> None:
    """Symbols are bound, never interpolated — the panel is user-supplied."""
    client = _stub()
    load_from_snowflake(client=client, symbols=["btcusdt", "ethusdt"])

    sql, params = client.calls[0]
    assert "SYMBOL IN (%s, %s)" in sql
    assert params == ("BTCUSDT", "ETHUSDT"), "symbols should be upper-cased and bound"


def test_no_filters_means_no_where_clause() -> None:
    client = _stub()
    load_from_snowflake(client=client)
    sql, params = client.calls[0]
    assert "WHERE" not in sql
    assert params is None


def test_empty_result_does_not_raise() -> None:
    """An empty warehouse should give empty frames, not a crash.

    Before the panel was loaded this was the real state of the table, and a
    traceback would have been a worse signal than an obviously empty panel.
    """
    empty_bars = pd.DataFrame(columns=["SYMBOL", "TS", "CLOSE", "VOLUME"])
    empty_funding = pd.DataFrame(columns=["SYMBOL", "TS", "RATE"])
    cw, *_ = load_from_snowflake(client=StubClient(empty_bars, empty_funding))
    assert cw.empty


@pytest.mark.parametrize(
    "mutate, expect",
    [
        (lambda d: d.iloc[:-1], "shape"),
        (lambda d: d.rename(columns={"BTCUSDT": "XRPUSDT"}), "columns differ"),
        (lambda d: d * 2.0, "max abs difference"),
    ],
)
def test_compare_rejects_real_differences(mutate, expect) -> None:
    """The gate has to be able to fail, or passing it means nothing."""
    frame = pd.DataFrame(
        {"BTCUSDT": [1.0, 2.0, 3.0]},
        index=pd.date_range("2024-01-01", periods=3, tz="UTC"),
    )
    ok, message = compare(frame, mutate(frame), "frame")
    assert not ok
    assert expect in message


def test_compare_rejects_a_hole_in_one_source() -> None:
    """A NaN opposite a number means a bar is missing from one panel.

    This is the failure mode a summary statistic would hide: a handful of absent
    bars barely moves a mean, but it changes which symbols the liquidity screen
    admits.
    """
    frame = pd.DataFrame(
        {"BTCUSDT": [1.0, 2.0, 3.0]},
        index=pd.date_range("2024-01-01", periods=3, tz="UTC"),
    )
    holed = frame.copy()
    holed.iloc[1, 0] = float("nan")
    ok, message = compare(frame, holed, "frame")
    assert not ok
    assert "present in one panel but not the other" in message
