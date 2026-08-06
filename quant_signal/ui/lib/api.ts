"use client";

export type Bar = {
  TRADE_DATE: string;
  DAY_OPEN: number | null;
  DAY_HIGH: number | null;
  DAY_LOW: number | null;
  DAY_CLOSE: number | null;
  VOLUME: number | null;
};

export type Fact = {
  TICKER: string;
  METRIC: string;
  FISCAL_YEAR: number;
  VALUE: number | null;
  UNIT: string | null;
  FILED_AT: string;
};

export type QuintileRow = {
  quintile: string;
  n: number;
  mean_sue: number | null;
} & Record<string, number | null>;

export type PeadResult = {
  metric: string;
  windows: number[];
  n_events: number;
  n_tickers: number;
  unlabeled: number;
  quintiles: QuintileRow[];
  spread: Record<string, number | null>;
};

export type MetricRow = {
  RUN_ID: string;
  FLOW: string;
  STAGE: string;
  STARTED_AT: string;
  ELAPSED_MS: number;
  N_ROWS: number | null;
  LOADED_AT: string;
};

export type MacroPoint = {
  SERIES_ID: string;
  DATE: string;
  VALUE: number | null;
};

async function get<T>(path: string): Promise<T> {
  const res = await fetch(path, { cache: "no-store" });
  if (!res.ok) {
    throw new Error(`${res.status} ${res.statusText} — ${path}`);
  }
  return (await res.json()) as T;
}

export const api = {
  tickers: () => get<{ tickers: string[] }>("/api/tickers"),
  metrics: () => get<{ metrics: string[] }>("/api/metrics"),
  macroSeries: () => get<{ series: string[] }>("/api/macro/series"),
  market: (symbol: string, days: number) =>
    get<{ symbol: string; count: number; bars: Bar[] }>(
      `/api/market/${encodeURIComponent(symbol)}?days=${days}`
    ),
  fundamentals: (ticker: string, metric?: string) =>
    get<{ ticker: string; metric: string | null; count: number; facts: Fact[] }>(
      `/api/fundamentals/${encodeURIComponent(ticker)}${
        metric ? `?metric=${encodeURIComponent(metric)}` : ""
      }`
    ),
  pead: (metric: string, windows: string, minPrior = 5, quintiles = 5) =>
    get<PeadResult>(
      `/api/pead?metric=${encodeURIComponent(metric)}&windows=${encodeURIComponent(
        windows
      )}&min_prior=${minPrior}&quintiles=${quintiles}`
    ),
  pipeline: (limit = 100, flow?: string) =>
    get<{ runs: MetricRow[] }>(
      `/api/metrics/pipeline?limit=${limit}${flow ? `&flow=${encodeURIComponent(flow)}` : ""}`
    ),
  macro: (series?: string, limit = 1000) =>
    get<{ count: number; points: MacroPoint[] }>(
      `/api/macro?limit=${limit}${series ? `&series=${encodeURIComponent(series)}` : ""}`
    ),
};
