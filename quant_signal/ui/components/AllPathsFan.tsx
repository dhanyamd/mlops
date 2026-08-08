"use client";

import { useEffect, useRef } from "react";

type AllPathsFanProps = {
  paths: number[][];
  percentiles: Record<string, number[]>;
  basePrice: number;
  height?: number;
};

const BAND_ORDER: [string, string, string][] = [
  ["10", "90", "rgba(59,130,246,0.10)"],
  ["25", "75", "rgba(59,130,246,0.14)"],
];

/**
 * All-paths fan: every simulated path drawn as a thin canvas stroke — the full
 * Monte Carlo distribution on one page (the "thousand futures" spaghetti fan).
 * Canvas, not SVG, so thousands of lines render in a single frame. Percentile
 * bands and the median sit under/over the raw strokes. Re-renders whenever a
 * new 5m window ships a fresh ensemble.
 */
export function AllPathsFan({
  paths,
  percentiles,
  basePrice,
  height = 280,
}: AllPathsFanProps) {
  const ref = useRef<HTMLCanvasElement>(null);

  useEffect(() => {
    const canvas = ref.current;
    if (!canvas) return;
    const ctx = canvas.getContext("2d");
    if (!ctx) return;

    const dpr = Math.min(window.devicePixelRatio || 1, 2);
    const steps = paths.length ? Math.max(0, paths[0].length - 1) : 0;
    if (steps === 0) return;

    let minPrice = basePrice;
    let maxPrice = basePrice;
    for (const p of paths) {
      for (const v of p) {
        if (v < minPrice) minPrice = v;
        if (v > maxPrice) maxPrice = v;
      }
    }
    const pad = (maxPrice - minPrice) * 0.08 || 1;
    minPrice -= pad;
    maxPrice += pad;

    const layout = () => {
      const rect = canvas.getBoundingClientRect();
      canvas.width = Math.round(rect.width * dpr);
      canvas.height = Math.round(rect.height * dpr);
      ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    };
    layout();

    const w = () => canvas.getBoundingClientRect().width;
    const h = () => canvas.getBoundingClientRect().height;
    const x = (i: number) => 8 + (i / steps) * (w() - 16);
    const y = (v: number) => 10 + (1 - (v - minPrice) / (maxPrice - minPrice)) * (h() - 20);

    ctx.clearRect(0, 0, w(), h());

    // base price reference
    ctx.strokeStyle = "rgba(128,128,128,0.15)";
    ctx.lineWidth = 1;
    ctx.setLineDash([3, 4]);
    ctx.beginPath();
    ctx.moveTo(0, y(basePrice));
    ctx.lineTo(w(), y(basePrice));
    ctx.stroke();
    ctx.setLineDash([]);

    // percentile bands
    for (const [loKey, hiKey, fill] of BAND_ORDER) {
      const lo = percentiles[loKey];
      const hi = percentiles[hiKey];
      if (!lo || !hi) continue;
      ctx.fillStyle = fill;
      ctx.beginPath();
      ctx.moveTo(x(0), y(lo[0]));
      for (let i = 1; i <= steps; i++) ctx.lineTo(x(i), y(lo[i]));
      for (let i = steps; i >= 0; i--) ctx.lineTo(x(i), y(hi[i]));
      ctx.closePath();
      ctx.fill();
    }

    // all raw paths, faint
    ctx.lineWidth = 1;
    for (const p of paths) {
      ctx.strokeStyle = "rgba(96,165,250,0.16)";
      ctx.beginPath();
      for (let i = 0; i <= steps; i++) {
        const px = x(i);
        const py = y(p[i]);
        if (i === 0) ctx.moveTo(px, py);
        else ctx.lineTo(px, py);
      }
      ctx.stroke();
    }

    // median
    const median = percentiles["50"];
    if (median) {
      ctx.strokeStyle = "rgba(251,113,133,0.95)";
      ctx.lineWidth = 1.8;
      ctx.beginPath();
      for (let i = 0; i <= steps; i++) {
        const px = x(i);
        const py = y(median[i]);
        if (i === 0) ctx.moveTo(px, py);
        else ctx.lineTo(px, py);
      }
      ctx.stroke();
    }

    ctx.fillStyle = "rgba(161,161,170,0.9)";
    ctx.font = "11px ui-monospace, SFMono-Regular, Menlo, monospace";
    ctx.fillText(
      `${paths.length} paths · bands 10–90 / 25–75 · median`,
      8,
      h() - 8
    );

    const onResize = () => layout();
    window.addEventListener("resize", onResize);
    return () => window.removeEventListener("resize", onResize);
  }, [paths, percentiles, basePrice]);

  return (
    <div style={{ height }} className="w-full">
      <canvas ref={ref} className="h-full w-full" />
    </div>
  );
}
