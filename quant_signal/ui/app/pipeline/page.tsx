"use client";

import { useMemo, useState } from "react";
import { Card } from "@/components/Card";
import { Bars } from "@/components/Charts";
import { DataTable } from "@/components/DataTable";
import { Select } from "@/components/Select";
import { api, type MetricRow } from "@/lib/api";
import { useQuery } from "@/lib/useQuery";

export default function PipelinePage() {
  const [flow, setFlow] = useState<string>("");

  const { data: rows, error } = useQuery([flow], async () =>
    (await api.pipeline(200, flow || undefined)).runs
  );

  const flows = useMemo(() => {
    if (!rows) return [];
    return [...new Set(rows.map((r) => r.FLOW))];
  }, [rows]);

  const latestRuns = useMemo(() => {
    if (!rows) return [];
    const byRun = new Map<string, MetricRow[]>();
    for (const r of rows) {
      const list = byRun.get(r.RUN_ID) ?? [];
      list.push(r);
      byRun.set(r.RUN_ID, list);
    }
    // RUN_IDs are already chronological (newest last); show the last 3 runs.
    return [...byRun.values()].slice(-3).reverse();
  }, [rows]);

  const latencyByStage = useMemo(() => {
    if (latestRuns.length === 0) return [];
    const stages = latestRuns[0];
    return stages.map((s) => ({
      name: s.STAGE,
      seconds: +(s.ELAPSED_MS / 1000).toFixed(2),
    }));
  }, [latestRuns]);

  return (
    <div className="space-y-6">
      <div className="flex flex-wrap items-end justify-between gap-4">
        <div>
          <h1 className="text-2xl font-semibold tracking-tight">Pipeline latency</h1>
          <p className="mt-1 text-sm text-zinc-500 dark:text-zinc-400">
            Per-stage wall-clock from PIPELINE_METRICS telemetry
          </p>
        </div>
        <Select
          label="Flow"
          value={flow}
          onChange={setFlow}
          options={[
            { value: "", label: "All flows" },
            ...flows.map((f) => ({ value: f, label: f })),
          ]}
        />
      </div>

      {error ? (
        <div className="rounded-lg border border-red-300 bg-red-50 px-4 py-3 text-sm text-red-700 dark:border-red-800 dark:bg-red-950/40 dark:text-red-300">
          {error}
        </div>
      ) : null}

      {rows === null ? (
        <p className="py-16 text-center text-sm text-zinc-500">Loading…</p>
      ) : (
        <div className="grid grid-cols-1 gap-6 lg:grid-cols-2">
          <Card title="Latest run — stage latency (s)" subtitle={latestRuns[0] ? latestRuns[0][0]?.FLOW : "…"}>
            <Bars data={latencyByStage} dataKey="seconds" xKey="name" color="#f43f5e" height={280} />
          </Card>

          <Card title="Recent runs">
            <DataTable
              columns={[
                { key: "FLOW", label: "Flow" },
                { key: "STAGE", label: "Stage" },
                { key: "ELAPSED_MS", label: "ms" },
                { key: "N_ROWS", label: "Rows" },
                { key: "STARTED_AT", label: "Started" },
              ]}
              rows={latestRuns.flat().map((r) => ({
                ...r,
                STARTED_AT: r.STARTED_AT.slice(0, 19).replace("T", " "),
              }))}
            />
          </Card>
        </div>
      )}
    </div>
  );
}
