"use client";

import { useEffect, useRef, useState } from "react";

export type LiveBar = {
  symbol: string;
  ts: string;
  open: number;
  high: number;
  low: number;
  close: number;
  volume: number;
};

type StreamMessage =
  | { type: "snapshot"; symbol: string; bars: LiveBar[] }
  | { type: "bar"; symbol: string; bar: LiveBar }
  | { type: "error"; message: string };

const MAX_BARS = 120;
const MAX_RETRY_MS = 15000;

function wsBase() {
  const base = process.env.NEXT_PUBLIC_API_BASE_URL ?? "http://127.0.0.1:8000";
  return base.replace(/^http/, "ws");
}

/**
 * Subscribe to the API's live minute-bar stream (`/ws/market`) with automatic
 * exponential-backoff reconnection. Raw bars are produced every ~20s, so the
 * chart this feeds visibly moves between the 5-minute feature windows that
 * drive everything else on the Signal page.
 */
export function useMarketStream(symbol: string) {
  const [bars, setBars] = useState<LiveBar[]>([]);
  const [connected, setConnected] = useState(false);
  const [retries, setRetries] = useState(0);
  const wsRef = useRef<WebSocket | null>(null);
  const barsRef = useRef<LiveBar[]>([]);
  const retriesRef = useRef(0);

  useEffect(() => {
    let disposed = false;
    let ws: WebSocket | null = null;

    const connect = () => {
      if (disposed) return;
      const url = `${wsBase()}/ws/market?symbol=${encodeURIComponent(symbol)}`;
      ws = new WebSocket(url);
      wsRef.current = ws;

      ws.onopen = () => {
        retriesRef.current = 0;
        setRetries(0);
        setConnected(true);
      };

      ws.onmessage = (event) => {
        let msg: StreamMessage;
        try {
          msg = JSON.parse(event.data as string) as StreamMessage;
        } catch {
          return;
        }
        if (msg.type === "snapshot") {
          barsRef.current = (msg.bars ?? []).slice(-MAX_BARS);
        } else if (msg.type === "bar" && msg.bar) {
          const next = [...barsRef.current, msg.bar];
          barsRef.current = next.slice(-MAX_BARS);
        } else {
          return;
        }
        if (!disposed) setBars(barsRef.current);
      };

      ws.onclose = () => {
        setConnected(false);
        if (disposed) return;
        const delay = Math.min(1000 * 2 ** retriesRef.current, MAX_RETRY_MS);
        retriesRef.current += 1;
        setRetries(retriesRef.current);
        window.setTimeout(connect, delay);
      };

      ws.onerror = () => {
        ws?.close();
      };
    };

    connect();
    return () => {
      disposed = true;
      ws?.close();
      wsRef.current = null;
      barsRef.current = [];
      setBars([]);
    };
  }, [symbol]);

  return { bars, connected, retries };
}
