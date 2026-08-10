import type { NextConfig } from "next";

const nextConfig: NextConfig = {
  // Lean Docker runtime: emits a self-contained .next/standalone tree so the
  // image only carries the server, not node_modules (next dev ignores this).
  output: "standalone",
  allowedDevOrigins: ["127.0.0.1", "localhost"],
  async rewrites() {
    // Proxy all /api/* calls to the FastAPI backend during development. Keeps
    // the browser same-origin (no CORS at all); the API also allows :3000 for
    // direct calls. In production this destination becomes the deployed API.
    const apiBase = process.env.API_BASE_URL ?? "http://127.0.0.1:8000";
    return [{ source: "/api/:path*", destination: `${apiBase}/api/:path*` }];
  },
};

export default nextConfig;
