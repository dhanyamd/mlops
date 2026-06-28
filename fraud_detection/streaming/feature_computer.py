"""Real-time feature computation for streaming transactions (Spark Streaming stand-in)."""

from __future__ import annotations

from collections import defaultdict, deque
from dataclasses import dataclass, field
from datetime import datetime, timedelta


@dataclass
class StreamingFeatureComputer:
    """
    In-memory velocity feature store for real-time path.
    Production: Spark Streaming / Flink writes to Feast Online Store.
    """

    window_1h: timedelta = timedelta(hours=1)
    window_24h: timedelta = timedelta(hours=24)
    _history: dict[str, deque] = field(default_factory=lambda: defaultdict(deque))

    def compute(self, user_id: str, amount: float, timestamp: datetime | None = None) -> dict:
        ts = timestamp or datetime.now()
        history = self._history[user_id]
        history.append((ts, amount))

        # Prune old entries
        cutoff = ts - self.window_24h
        while history and history[0][0] < cutoff:
            history.popleft()

        amounts = [a for _, a in history]
        timestamps = [t for t, _ in history]
        window_1h = [a for t, a in zip(timestamps, amounts) if t >= ts - self.window_1h]
        window_24h = amounts

        txn_1h = len(window_1h)
        txn_24h = len(window_24h)
        amt_sum = sum(window_24h)
        amt_mean = amt_sum / txn_24h if txn_24h else 0
        amt_std = (
            (sum((a - amt_mean) ** 2 for a in window_24h) / txn_24h) ** 0.5 if txn_24h > 1 else 0
        )

        return {
            "amount": amount,
            "txn_count_1h": txn_1h,
            "txn_count_24h": txn_24h,
            "amount_sum_24h": amt_sum,
            "amount_mean_24h": amt_mean,
            "amount_std_24h": amt_std,
            "velocity_ratio": txn_1h / max(txn_24h, 1),
            "amount_deviation": amount / max(amt_mean, 1),
            "hour_of_day": ts.hour,
            "is_night": int(ts.hour < 6 or ts.hour > 22),
        }
