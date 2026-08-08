"use client";

import { useEffect, useState } from "react";

const WINDOW_MS = 5 * 60 * 1000;

function fmt(ms: number) {
  const s = Math.max(0, Math.floor(ms / 1000));
  const m = Math.floor(s / 60);
  return `${String(m).padStart(2, "0")}:${String(s % 60).padStart(2, "0")}`;
}

/**
 * Counts down to the next 5-minute feature window boundary so the periodic
 * nature of the prediction/simulation refresh is visible instead of reading as
 * a frozen page.
 */
export function NextWindowCountdown() {
  const [left, setLeft] = useState(() => WINDOW_MS - (Date.now() % WINDOW_MS));

  useEffect(() => {
    const id = setInterval(
      () => setLeft(WINDOW_MS - (Date.now() % WINDOW_MS)),
      500
    );
    return () => clearInterval(id);
  }, []);

  return (
    <span className="font-mono text-xs tabular-nums text-zinc-400 dark:text-zinc-600">
      next window in {fmt(left)}
    </span>
  );
}
