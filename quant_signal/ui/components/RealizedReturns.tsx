"use client";

import { useMemo } from "react";
import {
  Bar,
  BarChart,
  CartesianGrid,
  Cell,
  ReferenceLine,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";
import type { FeatureWindow } from "@/lib/api";

/**
 * Realized per-window 5m returns derived from the actual feature windows the
 * predictor consumes. Unlike the near-zero online predictions, these are the
 * genuine market moves the pipeline is reacting to — so they visibly change
 * every 5-minute window.
 */
export function RealizedReturns({
  features,
  height = 220,
}: {
  features: FeatureWindow[];
  height?: number;
}) {
  const data = useMemo(() => {
    const rows: { label: string; ret: number }[] = [];
    for (let i = 0; i < features.length - 1; i++) {
      const newer = features[i].close;
      const older = features[i + 1].close;
      if (newer && older) {
        rows.push({
          label: new Date(features[i].window_end_ms).toLocaleTimeString([], {
            hour: "2-digit",
            minute: "2-digit",
          }),
          ret: (newer - older) / older,
        });
      }
    }
    return rows.reverse();
  }, [features]);

  return (
    <div style={{ height }} className="w-full">
      <ResponsiveContainer width="100%" height="100%">
        <BarChart data={data} margin={{ top: 8, right: 8, bottom: 0, left: 0 }}>
          <CartesianGrid strokeDasharray="3 3" stroke="currentColor" opacity={0.1} />
          <XAxis
            dataKey="label"
            tick={{ fontSize: 10 }}
            tickLine={false}
            axisLine={false}
            minTickGap={28}
          />
          <YAxis
            tick={{ fontSize: 10 }}
            tickLine={false}
            axisLine={false}
            width={52}
            tickFormatter={(v: number) => `${(v * 100).toFixed(2)}%`}
          />
          <Tooltip
            formatter={(v) => [`${(Number(v) * 100).toFixed(3)}%`, "realized 5m return"]}
            labelFormatter={(label) => String(label)}
            contentStyle={{
              background: "rgba(15,15,15,0.9)",
              border: "none",
              borderRadius: 8,
              color: "#eee",
              fontSize: 12,
            }}
          />
          <ReferenceLine y={0} stroke="currentColor" strokeOpacity={0.3} />
          <Bar dataKey="ret" radius={[2, 2, 0, 0]} isAnimationActive={false}>
            {data.map((d, i) => (
              <Cell
                key={i}
                fill={d.ret >= 0 ? "#10b981" : "#ef4444"}
                fillOpacity={0.85}
              />
            ))}
          </Bar>
        </BarChart>
      </ResponsiveContainer>
    </div>
  );
}
