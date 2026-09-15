import { spawn, type ChildProcess } from "node:child_process";
import { fileURLToPath } from "node:url";
import { defineConfig, type Plugin } from "vite";
import react from "@vitejs/plugin-react";

const projectRoot = fileURLToPath(new URL("..", import.meta.url));

/**
 * Dev-server API that runs ../fetch_patient.py and streams its JSON progress
 * lines back to the browser as NDJSON. One fetch at a time.
 *
 *   POST /api/fetch-patient[?patient=ID]   -> stream of {"stage": ...} lines
 *   GET  /api/fetch-status                 -> {"running": bool}
 */
function fetchPatientApi(): Plugin {
  let running: ChildProcess | null = null;

  return {
    name: "fetch-patient-api",
    configureServer(server) {
      server.middlewares.use("/api/fetch-status", (_req, res) => {
        res.setHeader("Content-Type", "application/json");
        res.end(JSON.stringify({ running: running !== null }));
      });

      server.middlewares.use("/api/fetch-patient", (req, res) => {
        if (req.method !== "POST") {
          res.statusCode = 405;
          res.end("POST only");
          return;
        }
        if (running) {
          res.statusCode = 409;
          res.end("A fetch is already running");
          return;
        }

        const patient = new URL(req.url ?? "/", "http://localhost").searchParams.get("patient");
        const args = ["-u", "fetch_patient.py", "--json"];
        if (patient) args.push("--patient", patient);

        res.writeHead(200, {
          "Content-Type": "application/x-ndjson",
          "Cache-Control": "no-cache",
          "X-Accel-Buffering": "no",
        });
        const send = (obj: unknown) => res.write(JSON.stringify(obj) + "\n");

        const proc = spawn(process.env.PYTHON ?? "python", args, { cwd: projectRoot });
        running = proc;

        // Forward complete stdout lines only, so a chunk boundary can't split a JSON object.
        let buf = "";
        proc.stdout.on("data", (chunk: Buffer) => {
          buf += chunk.toString();
          const lines = buf.split(/\r?\n/);
          buf = lines.pop() ?? "";
          for (const line of lines) if (line.trim()) res.write(line + "\n");
        });
        proc.stderr.on("data", (chunk: Buffer) => {
          for (const line of chunk.toString().split(/\r?\n/)) if (line.trim()) send({ stage: "log", msg: line });
        });
        proc.on("error", (e) => {
          running = null;
          send({ stage: "error", msg: `Could not start python: ${e.message}` });
          res.end();
        });
        proc.on("close", (code) => {
          running = null;
          if (buf.trim()) res.write(buf + "\n");
          send({ stage: "exit", code });
          res.end();
        });
      });
    },
  };
}

export default defineConfig({
  plugins: [react(), fetchPatientApi()],
  server: { port: 5173 },
});
