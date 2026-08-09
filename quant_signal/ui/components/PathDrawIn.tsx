"use client";

import { useEffect, useRef } from "react";

type PathFanProps = {
  paths: number[][];
  basePrice: number;
  height?: number;
  /** Fixed [min, max] price window (e.g. base ± 4σ·√steps). When omitted the
   * canvas auto-normalizes to each ensemble's min/max, which makes every
   * window look identical no matter how volatility actually moved. */
  domain?: [number, number];
};

/**
 * Animated Monte Carlo path draw-in: the sampled GBM paths draw themselves
 * forward on a canvas, ticking step-by-step in a continuous loop — a live,
 * breathing simulation visual instead of a static fan chart. The reveal
 * replay starts fresh every `epoch`, so the eye always has motion even when
 * the 5m window hasn't changed yet.
 */
export function PathDrawIn({ paths, basePrice, height = 280, domain }: PathFanProps) {
  const canvasRef = useRef<HTMLCanvasElement>(null);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext("2d");
    if (!ctx) return;

    const dpr = Math.min(window.devicePixelRatio || 1, 2);
    let raf = 0;

    const steps = paths.length ? Math.max(0, paths[0].length - 1) : 0;
    if (steps === 0) return;

    let minPrice = basePrice;
    let maxPrice = basePrice;
    if (domain) {
      [minPrice, maxPrice] = domain;
    } else {
      for (const p of paths) {
        for (const v of p) {
          if (v < minPrice) minPrice = v;
          if (v > maxPrice) maxPrice = v;
        }
      }
      const pad = (maxPrice - minPrice) * 0.08 || 1;
      minPrice -= pad;
      maxPrice += pad;
    }

    const layout = () => {
      const rect = canvas.getBoundingClientRect();
      canvas.width = Math.round(rect.width * dpr);
      canvas.height = Math.round(rect.height * dpr);
      ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    };
    layout();

    const CYCLE_MS = 2400;
    const HOLD_MS = 1600;
    const EPOCH_MS = CYCLE_MS + HOLD_MS;

    const x = (i: number) => {
      const w = canvas.getBoundingClientRect().width;
      return 8 + (i / steps) * (w - 16);
    };
    const y = (v: number) => {
      const h = canvas.getBoundingClientRect().height;
      return 10 + (1 - (v - minPrice) / (maxPrice - minPrice)) * (h - 20);
    };

    const drawFrame = (t: number) => {
      const rect = canvas.getBoundingClientRect();
      const w = rect.width;
      const h = rect.height;
      ctx.clearRect(0, 0, w, h);

      // base price reference line + grid
      ctx.strokeStyle = "rgba(128,128,128,0.12)";
      ctx.lineWidth = 1;
      ctx.setLineDash([3, 4]);
      ctx.beginPath();
      ctx.moveTo(0, y(basePrice));
      ctx.lineTo(w, y(basePrice));
      ctx.stroke();
      ctx.setLineDash([]);

      const phase = t % EPOCH_MS;
      const progress = Math.min(1, phase / CYCLE_MS);
      const reveal = Math.pow(progress, 0.85); // ease-in: fast start, slow finish
      const maxStep = reveal * steps;

      // draw percentile band by plotting all paths up to maxStep
      ctx.lineWidth = 1;
      for (let pi = 0; pi < paths.length; pi++) {
        const p = paths[pi];
        ctx.strokeStyle = "rgba(96,165,250,0.16)";
        ctx.beginPath();
        const end = Math.floor(maxStep);
        for (let i = 0; i <= end; i++) {
          const px = x(i);
          const py = y(p[i]);
          if (i === 0) ctx.moveTo(px, py);
          else ctx.lineTo(px, py);
        }
        // partial segment for the next step
        const frac = maxStep - end;
        if (frac > 0 && end + 1 <= steps) {
          const p0 = p[end];
          const p1 = p[end + 1];
          const px = x(end + frac);
          const py = y(p0 + (p1 - p0) * frac);
          ctx.lineTo(px, py);
        }
        ctx.stroke();
      }

      // trailing glow on the leading edge
      const leading = [];
      for (const p of paths) {
        const frac = maxStep - Math.floor(maxStep);
        const i = Math.floor(maxStep);
        const v = p[i] + (p[Math.min(i + 1, steps)] - p[i]) * frac;
        leading.push(v);
      }
      const leadMin = Math.min(...leading);
      const leadMax = Math.max(...leading);
      ctx.fillStyle = "rgba(59,130,246,0.9)";
      for (const p of paths) {
        const frac = maxStep - Math.floor(maxStep);
        const i = Math.floor(maxStep);
        const v = p[i] + (p[Math.min(i + 1, steps)] - p[i]) * frac;
        ctx.beginPath();
        ctx.arc(x(maxStep), y(v), 1.4, 0, Math.PI * 2);
        ctx.fill();
      }
      ctx.fillStyle = "rgba(248,113,113,0.85)";
      ctx.beginPath();
      ctx.arc(x(maxStep), y(leadMax), 2, 0, Math.PI * 2);
      ctx.arc(x(maxStep), y(leadMin), 2, 0, Math.PI * 2);
      ctx.fill();

      // step counter
      ctx.fillStyle = "rgba(161,161,170,0.9)";
      ctx.font = "11px ui-monospace, SFMono-Regular, Menlo, monospace";
      ctx.fillText(
        `step ${Math.floor(maxStep).toString().padStart(2, "0")}/${steps} · ${paths.length} paths`,
        8,
        h - 8
      );
    };

    const loop = (t: number) => {
      drawFrame(t);
      raf = requestAnimationFrame(loop);
    };
    raf = requestAnimationFrame(loop);
    const onResize = () => {
      layout();
      drawFrame(performance.now());
    };
    window.addEventListener("resize", onResize);

    return () => {
      cancelAnimationFrame(raf);
      window.removeEventListener("resize", onResize);
    };
  }, [paths, basePrice, domain]);

  return (
    <div style={{ height }} className="w-full">
      <canvas ref={canvasRef} className="h-full w-full" />
    </div>
  );
}
