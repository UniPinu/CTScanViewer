import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

/**
 * The dev server no longer spawns Python itself.
 *
 * It used to run `fetch_patient.py` and stream its stdout, which worked for one
 * script but does not generalise to the cohort, inference and job endpoints the
 * pipeline now exposes (CONTEXT.md §7). Those live in the FastAPI service, and
 * this proxies to them so the browser still sees a single origin and there is
 * no CORS to configure in development.
 *
 * Start both:
 *     .venv/Scripts/uvicorn server.app:app --reload --port 8000
 *     npm run dev
 */
const API_TARGET = process.env.CT_API ?? "http://127.0.0.1:8000";

export default defineConfig({
  plugins: [react()],
  server: {
    port: 5173,
    proxy: {
      "/api": {
        target: API_TARGET,
        changeOrigin: true,
        rewrite: (path) => path.replace(/^\/api/, ""),
        // Job progress is server-sent events; buffering would hold every
        // update until the job finished, which defeats the point.
        configure: (proxy) => {
          proxy.on("proxyRes", (proxyRes) => {
            if (proxyRes.headers["content-type"]?.includes("text/event-stream")) {
              proxyRes.headers["cache-control"] = "no-cache";
            }
          });
        },
      },
    },
  },
});
