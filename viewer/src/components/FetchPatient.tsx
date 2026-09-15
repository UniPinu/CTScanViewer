import { useEffect, useRef, useState } from "react";

/** One progress line from fetch_patient.py --json (see that file for the stages). */
interface Event {
  stage: string;
  msg?: string;
  patient?: string;
  series?: number;
  visits?: number;
  mb?: number;
  done?: number;
  total?: number;
  i?: number;
  n?: number;
  count?: number;
  code?: number | null;
}

type Status = "idle" | "running" | "done" | "error";

interface Props {
  /** Called once the new patient's data is on disk and exported. */
  onComplete: () => void;
  /** Render as a large call-to-action instead of a top-bar button. */
  prominent?: boolean;
}

const fmtMB = (bytes: number) => `${Math.round(bytes / 1e6)} MB`;

/**
 * "Fetch a random patient" button. Deletes the current data, downloads every
 * useful CT series for one randomly chosen patient, and exports it for the
 * viewer — all via the dev-server endpoint in vite.config.ts.
 */
export function FetchPatient({ onComplete, prominent }: Props) {
  const [status, setStatus] = useState<Status>("idle");
  const [headline, setHeadline] = useState("");
  const [detail, setDetail] = useState("");
  const [fraction, setFraction] = useState<number | null>(null);
  const [log, setLog] = useState<string[]>([]);
  const [open, setOpen] = useState(false);
  const abort = useRef<AbortController | null>(null);

  // If a fetch was started elsewhere (another tab, the CLI), don't allow a second one.
  useEffect(() => {
    fetch("/api/fetch-status")
      .then((r) => r.json())
      .then((s: { running: boolean }) => {
        if (s.running) {
          setStatus("running");
          setHeadline("A fetch is already running in another tab or terminal…");
        }
      })
      .catch(() => undefined);
  }, []);

  const handle = (ev: Event) => {
    switch (ev.stage) {
      case "index":
        setHeadline("Loading the IDC index…");
        setDetail("Matching the manifest against the catalogue of scans.");
        break;
      case "selected":
        setHeadline(`Patient ${ev.patient}`);
        setDetail(`${ev.series} scans over ${ev.visits} visit(s), about ${ev.mb} MB.`);
        break;
      case "clearing":
        setHeadline((h) => h || "Clearing old data…");
        setDetail("Removing the previous patient's files.");
        break;
      case "download":
        if (ev.total) {
          setFraction((ev.done ?? 0) / ev.total);
          setDetail(`Downloading ${fmtMB(ev.done ?? 0)} of ${fmtMB(ev.total)}`);
        }
        break;
      case "export":
        setFraction(null);
        setDetail(`Preparing scan ${ev.i} of ${ev.n} for the viewer…`);
        break;
      case "done":
        setStatus("done");
        setFraction(1);
        setDetail(`Ready: ${ev.count} scans.`);
        onComplete();
        break;
      case "error":
        setStatus("error");
        setDetail(ev.msg ?? "Unknown error");
        break;
      case "log":
        if (ev.msg) setLog((l) => [...l.slice(-30), ev.msg!]);
        break;
      case "exit":
        setStatus((s) => (s === "done" ? s : ev.code === 0 ? "done" : "error"));
        if (ev.code !== 0) setDetail((d) => d || `Script exited with code ${ev.code}`);
        break;
    }
  };

  const start = async () => {
    const ok = window.confirm(
      "This deletes the current scans (data/patient and viewer/public/data) and downloads one random patient — " +
        "every CT scan they have, typically 100–600 MB. Continue?",
    );
    if (!ok) return;

    setStatus("running");
    setOpen(true);
    setHeadline("Starting…");
    setDetail("");
    setFraction(null);
    setLog([]);
    abort.current = new AbortController();

    try {
      const res = await fetch("/api/fetch-patient", { method: "POST", signal: abort.current.signal });
      if (!res.ok || !res.body) {
        setStatus("error");
        setDetail(await res.text());
        return;
      }
      const reader = res.body.getReader();
      const decoder = new TextDecoder();
      let buf = "";
      for (;;) {
        const { done, value } = await reader.read();
        if (done) break;
        buf += decoder.decode(value, { stream: true });
        const lines = buf.split("\n");
        buf = lines.pop() ?? "";
        for (const line of lines) {
          if (!line.trim()) continue;
          try {
            handle(JSON.parse(line) as Event);
          } catch {
            setLog((l) => [...l.slice(-30), line]);
          }
        }
      }
    } catch (e) {
      setStatus("error");
      setDetail(e instanceof Error ? e.message : String(e));
    }
  };

  const busy = status === "running";
  const button = (
    <button className={prominent ? "cta" : "chip"} onClick={start} disabled={busy} title="Replace the current data with one random patient">
      {busy ? "Fetching…" : "Fetch a random patient"}
    </button>
  );

  return (
    <>
      {button}
      {!prominent && (status !== "idle" || open) && (
        <button className="chip" onClick={() => setOpen((o) => !o)}>{open ? "Hide" : "Progress"}</button>
      )}
      {open && (
        <div className="fetch-panel" role="status">
          <div className="fetch-head">
            <b>{headline || (status === "idle" ? "Ready" : "")}</b>
            <span className={`badge ${status}`}>{status}</span>
          </div>
          <div className="fetch-detail">{detail}</div>
          {busy && (
            <div className={`progress ${fraction === null ? "indeterminate" : ""}`}>
              <div style={{ width: fraction === null ? undefined : `${Math.round(fraction * 100)}%` }} />
            </div>
          )}
          {log.length > 0 && (
            <details>
              <summary>Log</summary>
              <pre>{log.join("\n")}</pre>
            </details>
          )}
          {(status === "done" || status === "error") && (
            <button className="chip" onClick={() => setOpen(false)}>Close</button>
          )}
        </div>
      )}
    </>
  );
}
