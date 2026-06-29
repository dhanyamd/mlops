"""Replay real fraud dataset transactions into Kafka for streaming pipeline testing."""

from __future__ import annotations

import argparse
import json
import time
from datetime import datetime

from confluent_kafka import Producer

from shared.clients import ClickHouseClient
from shared.config import KAFKA


def produce_from_clickhouse(n_messages: int = 100, delay_sec: float = 0.1) -> None:
    ch = ClickHouseClient()
    df = ch.query_df(
        f"""
        SELECT transaction_id, time_seconds, amount,
               v1,v2,v3,v4,v5,v6,v7,v8,v9,v10,
               v11,v12,v13,v14,v15,v16,v17,v18,v19,v20,
               v21,v22,v23,v24,v25,v26,v27,v28
        FROM fraud.transactions
        ORDER BY time_seconds
        LIMIT {n_messages}
        """
    )

    producer = Producer({"bootstrap.servers": KAFKA.bootstrap_servers})

    for _, row in df.iterrows():
        card_id = f"CARD_{int(row['time_seconds']) // 3600}"
        txn = {
            "transaction_id": row["transaction_id"],
            "card_id": card_id,
            "amount": float(row["amount"]),
            "time_seconds": float(row["time_seconds"]),
            "timestamp": datetime.utcnow().isoformat(),
            **{f"v{i}": float(row[f"v{i}"]) for i in range(1, 29)},
        }
        producer.produce(
            KAFKA.transactions_topic,
            key=card_id.encode(),
            value=json.dumps(txn).encode(),
        )
        producer.poll(0)
        time.sleep(delay_sec)

    producer.flush()
    from shared.observability.logging import get_logger
    log = get_logger(__name__)
    log.info("produced_transactions", count=len(df), topic=KAFKA.transactions_topic)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--count", type=int, default=100)
    parser.add_argument("--delay", type=float, default=0.05)
    args = parser.parse_args()
    produce_from_clickhouse(args.count, args.delay)
