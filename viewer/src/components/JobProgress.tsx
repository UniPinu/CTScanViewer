import type { Job } from "../api";

/**
 * Staged, honest progress (CONTEXT.md §8.4).
 *
 * Loading is frequent here — the data is streamed from a public bucket on
 * demand — so the brief asks for it to be a first-class experience showing the
 * *stage*, not a spinner. Each stage is named as the server reports it, with
 * earlier stages marked complete, so the wait reads as progress through a known
 * sequence rather than an indefinite hang.
 */

/** The order stages actually occur in, so the list can show what is behind and ahead. */
const FETCH_STAGES = [
  { id: "clearing", label: "Clearing previous patient" },
  { id: "download", label: "Downloading from Imaging Data Commons" },
  { id: "preprocess", label: "Preparing volumes" },
];

const INFER_STAGES = [
  { id: "risk", label: "Estimating risk" },
  { id: "saliency", label: "Rendering attention" },
  { id: "change", label: "Registering rounds and detecting change" },
  { id: "done", label: "Assembling results" },
];

export function JobProgress({ job, onDismiss }: { job: Job; onDismiss?: () => void }) {
  const stages = job.kind === "fetch" ? FETCH_STAGES : INFER_STAGES;
  const currentIndex = stages.findIndex((s) => s.id === job.stage);
  const running = job.status === "running" || job.status === "queued";

  return (
    <div className="panel" role="status" aria-live="polite">
      <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center" }}>
        <h3 style={{ margin: 0 }}>{job.kind === "fetch" ? "Fetching patient" : "Running inference"}</h3>
        <span className={`badge ${job.status}`}>{job.status}</span>
      </div>

      <p className="hint" style={{ marginTop: 8 }}>
        {job.message || (running ? "Starting…" : "")}
        {job.elapsed > 0 && <span className="faint"> · {job.elapsed.toFixed(0)}s</span>}
      </p>

      {running && (
        <div className={`progress ${job.progress === null ? "indeterminate" : ""}`} style={{ marginTop: 8 }}>
          <div style={{ width: job.progress === null ? undefined : `${Math.round(job.progress * 100)}%` }} />
        </div>
      )}

      <div className="stages">
        {stages.map((stage, i) => {
          const state =
            currentIndex < 0 ? "" : i < currentIndex ? "complete" : i === currentIndex ? "active" : "";
          return (
            <div key={stage.id} className={`stage-step ${job.status === "done" ? "complete" : state}`}>
              <span className="stage-marker" />
              {stage.label}
            </div>
          );
        })}
      </div>

      {job.error && (
        <div className="notice" style={{ marginTop: 10 }}>
          <strong>That did not work</strong>
          {job.error}
          {job.log.length > 0 && (
            <details style={{ marginTop: 8 }}>
              <summary>Log</summary>
              <pre>{job.log.join("\n")}</pre>
            </details>
          )}
        </div>
      )}

      {!running && onDismiss && (
        <button className="chip" style={{ marginTop: 10 }} onClick={onDismiss}>
          Dismiss
        </button>
      )}
    </div>
  );
}
