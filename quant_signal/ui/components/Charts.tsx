"use client";

import {
  Area,
  AreaChart,
  Bar,
  BarChart,
  CartesianGrid,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";

export function LineChart({
  data,
  dataKey,
  xKey = "TRADE_DATE",
  color = "#3b82f6",
  height = 320,
}: {
  data: Record<string, unknown>[];
  dataKey: string;
  xKey?: string;
  color?: string;
  height?: number;
}) {
  return (
    <div style={{ height }} className="w-full">
      <ResponsiveContainer width="100%" height="100%">
        <AreaChart data={data} margin={{ top: 8, right: 8, bottom: 0, left: 0 }}>
          <defs>
            <linearGradient id={`fill-${dataKey}`} x1="0" y1="0" x2="0" y2="1">
              <stop offset="0%" stopColor={color} stopOpacity={0.25} />
              <stop offset="100%" stopColor={color} stopOpacity={0} />
            </linearGradient>
          </defs>
          <CartesianGrid strokeDasharray="3 3" stroke="currentColor" opacity={0.1} />
          <XAxis dataKey={xKey} tick={{ fontSize: 11 }} tickLine={false} axisLine={false} minTickGap={40} />
          <YAxis tick={{ fontSize: 11 }} tickLine={false} axisLine={false} width={64} domain={["auto", "auto"]} />
          <Tooltip
            contentStyle={{
              background: "rgba(15,15,15,0.9)",
              border: "none",
              borderRadius: 8,
              color: "#eee",
              fontSize: 12,
            }}
          />
          <Area
            type="monotone"
            dataKey={dataKey}
            stroke={color}
            strokeWidth={1.5}
            fill={`url(#fill-${dataKey})`}
            dot={false}
          />
        </AreaChart>
      </ResponsiveContainer>
    </div>
  );
}

export function Bars({
  data,
  dataKey,
  xKey = "name",
  height = 320,
  color = "#8b5cf6",
}: {
  data: Record<string, unknown>[];
  dataKey: string;
  xKey?: string;
  height?: number;
  color?: string;
}) {
  return (
    <div style={{ height }} className="w-full">
      <ResponsiveContainer width="100%" height="100%">
        <BarChart data={data} margin={{ top: 8, right: 8, bottom: 0, left: 0 }}>
          <CartesianGrid strokeDasharray="3 3" stroke="currentColor" opacity={0.1} />
          <XAxis dataKey={xKey} tick={{ fontSize: 11 }} tickLine={false} axisLine={false} />
          <YAxis tick={{ fontSize: 11 }} tickLine={false} axisLine={false} width={64} />
          <Tooltip
            contentStyle={{
              background: "rgba(15,15,15,0.9)",
              border: "none",
              borderRadius: 8,
              color: "#eee",
              fontSize: 12,
            }}
          />
          <Bar dataKey={dataKey} fill={color} radius={[4, 4, 0, 0]} />
        </BarChart>
      </ResponsiveContainer>
    </div>
  );
}
