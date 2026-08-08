"use client";

import { useEffect, useRef } from "react";
import * as echarts from "echarts/core";
import { Bar3DChart } from "echarts-gl/charts";
import { Grid3DComponent } from "echarts-gl/components";
import { TooltipComponent } from "echarts/components";
import { VisualMapComponent } from "echarts/components";
import { CanvasRenderer } from "echarts/renderers";

echarts.use([Bar3DChart, Grid3DComponent, TooltipComponent, VisualMapComponent, CanvasRenderer]);

type SurfaceGrid = {
  steps: number;
  edges: number[];
  counts: number[][];
};

function formatPct(v: number) {
  return `${(v * 100).toFixed(1)}%`;
}

/**
 * 3D Monte Carlo probability surface: for every forward step (x), the density
 * of simulated path returns (y) is drawn as a WebGL bar3D mountain (z). The
 * surface is rebuilt from the full 10k-path ensemble on every window, so the
 * mountain visibly re-forms as new simulations arrive — the "probability
 * landscape" of the forward fan chart.
 */
export function ProbSurface3D({
  surface,
  basePrice,
  height = 320,
}: {
  surface: SurfaceGrid;
  basePrice: number;
  height?: number;
}) {
  const ref = useRef<HTMLDivElement>(null);
  const chartRef = useRef<echarts.ECharts | null>(null);

  useEffect(() => {
    if (!ref.current) return;
    const chart = echarts.init(ref.current);
    chartRef.current = chart;
    const onResize = () => chart.resize();
    window.addEventListener("resize", onResize);
    return () => {
      window.removeEventListener("resize", onResize);
      chart.dispose();
      chartRef.current = null;
    };
  }, []);

  useEffect(() => {
    const chart = chartRef.current;
    if (!chart) return;

    const { edges, counts } = surface;
    const binMids = edges.slice(0, -1).map((e, i) => (e + edges[i + 1]) / 2);
    const data: [number, number, number][] = [];
    counts.forEach((row, step) => {
      row.forEach((count, bin) => {
        if (count > 0) data.push([step, binMids[bin], count]);
      });
    });

    chart.setOption(
      {
        animationDuration: 500,
        animationEasing: "cubicOut",
        tooltip: {
          formatter: (params: { value: [number, number, number] }) => {
            const [step, ret, count] = params.value;
            return `step ${step} · return ${formatPct(ret)}<br/>${count} paths (of ${counts[0].reduce((a, b) => a + b, 0)})`;
          },
        },
        visualMap: {
          max: Math.max(1, ...data.map((d) => d[2])),
          dimension: 2,
          inRange: {
            color: ["#1e3a8a", "#2563eb", "#38bdf8", "#22d3ee", "#a5f3fc"],
          },
          show: false,
        },
        xAxis3D: {
          type: "value",
          name: "step",
          nameTextStyle: { color: "#71717a" },
          axisLabel: { color: "#a1a1aa" },
        },
        yAxis3D: {
          type: "value",
          name: "return",
          nameTextStyle: { color: "#71717a" },
          axisLabel: {
            color: "#a1a1aa",
            formatter: (v: number) => formatPct(v),
          },
        },
        zAxis3D: {
          type: "value",
          name: "paths",
          nameTextStyle: { color: "#71717a" },
          axisLabel: { color: "#a1a1aa" },
        },
        grid3D: {
          boxWidth: 110,
          boxDepth: 90,
          boxHeight: 60,
          viewControl: {
            autoRotate: true,
            autoRotateSpeed: 6,
            distance: 210,
            alpha: 28,
            beta: 30,
          },
          light: {
            main: { intensity: 1.2, shadow: true, alpha: 30, beta: 45 },
            ambient: { intensity: 0.35 },
          },
        },
        series: [
          {
            type: "bar3D",
            data,
            barSize: 0.55,
            shading: "lambert",
            emphasis: { itemStyle: { color: "#fde68a" } },
            itemStyle: { opacity: 0.88 },
          },
        ],
      },
      { replaceMerge: ["series", "visualMap"] }
    );
  }, [surface]);

  return (
    <div className="w-full">
      <div ref={ref} style={{ height }} className="w-full" />
      <div className="mt-1 flex items-center justify-between text-[11px] text-zinc-500 dark:text-zinc-400">
        <span>
          return density per forward step · base {basePrice.toLocaleString(undefined, { maximumFractionDigits: 0 })}
        </span>
        <span className="font-mono">{surface.steps}×5m horizon</span>
      </div>
    </div>
  );
}
