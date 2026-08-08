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

export type Prediction = {
  symbol: string;
  window_end_ms?: number | null;
  predicted_return: number;
  interval_low: number;
  interval_high: number;
  direction: "LONG" | "SHORT" | "FLAT";
  alpha?: number | null;
  coverage?: number | null;
  updated_at?: string | null;
};

export type Simulation = {
  symbol: string;
  base_price: number;
  horizon_steps: number;
  n_paths: number;
  sigma_annualized: number;
  percentiles: Record<string, number[]>;
  median_path: number[];
  sample_paths?: number[][];
  var95: number;
  es95: number;
  prob_up: number;
  returns_histogram: { counts: number[]; edges: number[] };
  surface_grid?: {
    steps: number;
    edges: number[];
    counts: number[][];
  };
  confidence_interval: { p10: number; p90: number };
};

export type Strategy = {
  symbol: string;
  n_windows: number;
  n_trades: number;
  n_wins: number;
  win_rate: number | null;
  strategy_equity: number[];
  buyhold_equity: number[];
  total_return_strategy: number;
  total_return_buyhold: number;
  updated_at?: string | null;
};

export type SamplePath = {
  equity: number[];
  outcome: "passed" | "busted" | "neutral";
  terminal_return: number;
  max_drawdown: number;
};

export type Validation = {
  n_periods: number;
  n_sims: number;
  max_drawdown_rule: number;
  target: number | null;
  pass_probability: number;
  bust_rate: number;
  neutral_rate: number;
  expected_terminal: number;
  expected_return: number;
  median_terminal: number;
  best10_terminal: number;
  worst10_terminal: number;
  avg_max_drawdown: number;
  median_max_drawdown: number;
  p95_max_drawdown: number;
  equity_fan: Record<string, number[]>;
  median_path: number[];
  sample_paths: SamplePath[];
  terminal_histogram: {
    counts: number[];
    passed: number[];
    busted: number[];
    neutral: number[];
    edges: number[];
  };
  drawdown_histogram: { counts: number[]; edges: number[] };
};

export type HealthStatus = "healthy" | "stale" | "warming";

export type HealthStage = {
  name: "produce" | "features" | "predict" | "simulate" | "strategy";
  symbol: string;
  status: HealthStatus;
  age_seconds: number | null;
  venue?: string | null;
  detail: string;
};

export type HealthSummary = {
  enabled: boolean;
  healthy: boolean | null;
  threshold_seconds: number;
  stages: HealthStage[];
};

export type EdgeSweep = {
  edges_bps: number[];
  pass: number[];
  current_edge_bps: number;
};

export type GeometryGrid = {
  grid: number[][];
  wr_axis: number[];
  rr_axis: number[];
  ev: number;
  sigma?: number;
  edge_sweep?: EdgeSweep;
};

export type FeatureWindow = {
  symbol: string;
  window_start_ms: number;
  window_end_ms: number;
  open: number;
  high: number;
  low: number;
  close: number;
  volume: number;
  vwap?: number | null;
  bar_count?: number | null;
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
  marketSymbols: () => get<{ symbols: string[] }>("/api/market/symbols"),
  prediction: (symbol: string) =>
    get<{ symbol: string; enabled: boolean; prediction: Prediction | null }>(
      `/api/market/predict/${encodeURIComponent(symbol)}`
    ),
  simulation: (symbol: string) =>
    get<{ symbol: string; enabled: boolean; simulation: Simulation | null }>(
      `/api/market/simulation/${encodeURIComponent(symbol)}`
    ),
  strategy: (symbol: string) =>
    get<{ symbol: string; enabled: boolean; strategy: Strategy | null }>(
      `/api/market/strategy/${encodeURIComponent(symbol)}`
    ),
  validation: (symbol: string) =>
    get<{ symbol: string; enabled: boolean; validation: Validation | null }>(
      `/api/market/validation/${encodeURIComponent(symbol)}`
    ),
  features: (symbol: string, limit = 12) =>
    get<{ symbol: string; enabled: boolean; count: number; features: FeatureWindow[] }>(
      `/api/market/features/${encodeURIComponent(symbol)}?limit=${limit}`
    ),
  healthSummary: () => get<HealthSummary>("/api/market/health/summary"),
  geometry: (symbol: string) =>
    get<{ symbol: string; enabled: boolean; grid: GeometryGrid | null }>(
      `/api/market/validation/${encodeURIComponent(symbol)}/geometry`
    ),
};
