/**
 * Typed client for the FastAPI service (CONTEXT.md §7).
 *
 * Everything goes through `/api`, which Vite proxies to the Python service in
 * development (see vite.config.ts) so the browser only ever sees one origin.
 */

const API = "/api";

export type Split = "train" | "val" | "test";
export type JobStatus = "queued" | "running" | "done" | "error" | "cancelled";
export type Task = "risk" | "change" | "saliency";

export interface CohortPatient {
  patient_id: string;
  split: Split;
  n_rounds: number;
  /** 1 cancer, 0 no cancer, -1 unknown. Unknown is the norm until CDAS lands. */
  label: number;
  label_source: string;
}

export interface RoundInfo {
  round: string;
  study_date: string;
  series_uid: string;
  kernel: string | null;
  kernel_style: string | null;
  slice_mm: number | null;
  instances: number;
  size_mb: number;
  cached: boolean;
  manufacturer: string;
  model: string;
  alternate_reconstructions: number;
  /** The official NLST read for this round — known without downloading it. */
  screen_result: string | null;
  screen_days: number | null;
}

/** A decoded field from the NLST participant table. */
export interface ClinicalField {
  column: string;
  label: string;
  value: string | null;
}

/** The NLST participant record: the trial's own outcome for this person. */
export interface ClinicalRecord {
  patient_id: string;
  found: boolean;
  label?: number;
  label_source?: string;
  error?: string;
  outcome?: {
    diagnosed: boolean;
    days_to_diagnosis: number | null;
    years_to_diagnosis: number | null;
    diagnosis_study_year: string | null;
    cancer_free_days: number | null;
    screen_result_at_diagnosis: string | null;
  };
  demographics?: ClinicalField[];
  screening?: { round: string; result: string | null; days_from_randomisation: number | null }[];
  tumour?: ClinicalField[];
  locations?: string[];
}

export interface PatientOverview {
  patient_id: string;
  split: Split;
  n_rounds: number;
  n_series: number;
  total_mb: number;
  primary_mb: number;
  cached_mb: number;
  rounds: RoundInfo[];
  clinical: ClinicalRecord;
  /** 1 cancer, 0 no diagnosis on record, -1 unknown. */
  label: number;
  label_source: string;
  has_results: boolean;
  exported: boolean;
}

export interface Job {
  job_id: string;
  kind: string;
  status: JobStatus;
  stage: string;
  message: string;
  progress: number | null;
  patient_id: string | null;
  elapsed: number;
  error: string | null;
  result: unknown;
  log: string[];
}

/** Risk at years 1–6, or an explicit statement that no model ran (§11). */
export interface RiskBlock {
  unavailable?: boolean;
  reason?: string;
  model?: string;
  round?: string;
  calibrated?: boolean;
  curve?: number[];
  cohort_percentile?: number | null;
  ci_1?: [number, number];
  year_1?: number;
  year_2?: number;
  year_3?: number;
  year_4?: number;
  year_5?: number;
  year_6?: number;
}

export interface ChangeRegion {
  slice: number;
  bbox: number[];
  centroid_mm: number[];
  volume_mm3: number;
  delta_mm: number;
  mean_delta_hu: number;
  peak_delta_hu: number;
  baseline_hu: number;
  followup_hu: number;
  fill_fraction: number;
  elongation: number;
  kind: "new" | "growth";
  side: "left" | "right";
}

export interface ChangePair {
  from: string;
  to: string;
  regions: ChangeRegion[];
  n_regions: number;
  registration: {
    alignment_before_hu?: number;
    alignment_after_hu?: number;
    alignment_improved?: boolean;
    registration_failed?: boolean;
    selected_stage?: string;
    alignment_by_stage_hu?: Record<string, number>;
    inflation_shift_hu?: number;
    deformable?: boolean;
    stages?: { stage: string; metric?: number; seconds?: number }[];
  };
  spacing_mm: number;
  shape: number[];
  lung_volume_ml: { from: number; to: number };
  overlay_dir: string | null;
  note?: string;
  seconds: number;
}

export interface OverlayMeta {
  name: string;
  file: string;
  shape: [number, number, number];
  value_range: [number, number];
  units: string;
  top_slices: number[];
  per_slice: number[];
  overlay_dir?: string;
  /** The screening round this overlay's geometry belongs to. */
  round?: string | null;
  /** The patient it describes, so it cannot be drawn over someone else's scan. */
  patient_id?: string | null;
}

export interface Results {
  patient_id: string;
  split: Split;
  rounds: string[];
  tasks: Task[];
  risk?: RiskBlock;
  attention?: OverlayMeta;
  change?: { pairs: ChangePair[]; note?: string };
  summary: string;
  provenance: {
    model: string;
    code_rev: string;
    generated_at: string;
    research_only: boolean;
    disclaimer: string;
  };
}

export interface SystemInfo {
  risk: { available: boolean; name: string; ok?: boolean; reason?: string; cuda?: boolean };
  cohort: { built: boolean; splits: boolean };
  cache: { series: number; bytes: number; cap_bytes: number; used_fraction: number };
}

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  const res = await fetch(`${API}${path}`, {
    headers: init?.body ? { "Content-Type": "application/json" } : undefined,
    ...init,
  });
  if (!res.ok) {
    // FastAPI puts the useful message in `detail`; fall back to the status.
    const detail = await res
      .json()
      .then((b: { detail?: string }) => b.detail)
      .catch(() => null);
    throw new Error(detail ?? `${path}: HTTP ${res.status}`);
  }
  return res.json() as Promise<T>;
}

export const api = {
  system: () => request<SystemInfo>("/system"),

  cohort: (split: Split | "all", limit = 60, minRounds = 1, q = "", label?: number) => {
    const params = new URLSearchParams({ limit: String(limit), min_rounds: String(minRounds) });
    if (split !== "all") params.set("split", split);
    if (q) params.set("q", q);
    if (label !== undefined) params.set("label", String(label));
    return request<{ patients: CohortPatient[]; count: number }>(`/cohort?${params}`);
  },

  patient: (id: string) => request<PatientOverview>(`/patient/${id}`),

  randomPatient: (split: Split | "all" = "test", minRounds = 2) => {
    const params = new URLSearchParams({ min_rounds: String(minRounds) });
    if (split !== "all") params.set("split", split);
    return request<PatientOverview>(`/random-patient?${params}`);
  },

  fetchPatient: (id: string) => request<Job>(`/patient/${id}/fetch`, { method: "POST" }),

  infer: (patient_id: string, tasks: Task[]) =>
    request<Job>("/infer", { method: "POST", body: JSON.stringify({ patient_id, tasks }) }),

  job: (jobId: string) => request<Job>(`/jobs/${jobId}`),

  results: (id: string) => request<Results>(`/results/${id}`),
};

/**
 * Follow a job to completion over server-sent events.
 *
 * Progress matters here: a fetch is tens of seconds and §8.4 asks for the
 * *stage* to be visible rather than a spinner. Returns a cancel function.
 */
export function watchJob(
  jobId: string,
  onUpdate: (job: Job) => void,
  onSettled: (job: Job) => void,
): () => void {
  const source = new EventSource(`${API}/jobs/${jobId}/stream`);
  let settled = false;

  source.onmessage = (event) => {
    const job = JSON.parse(event.data) as Job;
    onUpdate(job);
    if (job.status === "done" || job.status === "error" || job.status === "cancelled") {
      settled = true;
      source.close();
      onSettled(job);
    }
  };

  // If the stream drops before the job finishes, fall back to one poll so the
  // UI resolves instead of sitting on a stale "running".
  source.onerror = () => {
    source.close();
    if (settled) return;
    api
      .job(jobId)
      .then((job) => {
        onUpdate(job);
        if (job.status !== "running" && job.status !== "queued") onSettled(job);
      })
      .catch(() => undefined);
  };

  return () => source.close();
}
