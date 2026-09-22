import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import {
  api, watchJob,
  type ChangePair, type ChangeRegion, type Job, type PatientOverview,
  type Results, type SystemInfo, type Task,
} from "./api";
import { BrowseView } from "./components/BrowseView";
import { CohortRail } from "./components/CohortRail";
import { CompareView } from "./components/CompareView";
import { JobProgress } from "./components/JobProgress";
import { OverlayControls } from "./components/OverlayControls";
import { ResultsPanel } from "./components/ResultsPanel";
import { VolumeView } from "./components/VolumeView";
import { clearVolumeCache, loadIndex, planeDepth } from "./data";
import { useVolume } from "./hooks";
import { clearOverlayCache, loadOverlay, overlayBelongsTo, type OverlayVolume } from "./overlay";
import type { HUWindow, Plane, SeriesMeta } from "./types";
import { DEFAULT_WINDOW } from "./windowing";

type Tab = "browse" | "compare" | "volume";
const TABS: { id: Tab; name: string }[] = [
  { id: "browse", name: "Slices" },
  { id: "compare", name: "Before / after" },
  { id: "volume", name: "3-D" },
];

const ALL_TASKS: Task[] = ["risk", "change", "saliency"];

const middleSlice = (m: SeriesMeta): Record<Plane, number> => ({
  axial: Math.floor(planeDepth(m, "axial") / 2),
  coronal: Math.floor(planeDepth(m, "coronal") / 2),
  sagittal: Math.floor(planeDepth(m, "sagittal") / 2),
});

export default function App() {
  const [system, setSystem] = useState<SystemInfo | null>(null);
  const [patient, setPatient] = useState<PatientOverview | null>(null);
  const [results, setResults] = useState<Results | null>(null);
  const [series, setSeries] = useState<SeriesMeta[]>([]);
  const [job, setJob] = useState<Job | null>(null);
  const [error, setError] = useState<string | null>(null);

  const [tab, setTab] = useState<Tab>("browse");
  const [selected, setSelected] = useState<SeriesMeta | null>(null);
  const [plane, setPlane] = useState<Plane>("axial");
  const [indices, setIndices] = useState<Record<Plane, number>>({ axial: 0, coronal: 0, sagittal: 0 });
  const [win, setWin] = useState<HUWindow>(DEFAULT_WINDOW);
  const [left, setLeft] = useState<SeriesMeta | null>(null);
  const [right, setRight] = useState<SeriesMeta | null>(null);

  const [activePair, setActivePair] = useState<string | null>(null);
  const [activeRegion, setActiveRegion] = useState<ChangeRegion | null>(null);
  const [overlayOpacity, setOverlayOpacity] = useState(0.6);
  const [overlayOn, setOverlayOn] = useState({ attention: false, change: true });
  const [overlays, setOverlays] = useState<OverlayVolume[]>([]);

  const cancelWatch = useRef<(() => void) | null>(null);
  const volumeState = useVolume(selected);

  useEffect(() => {
    api.system().then(setSystem).catch((e: Error) => setError(e.message));
    return () => cancelWatch.current?.();
  }, []);

  /** Re-read what is on disk for the viewer, and reset the selection. */
  const reloadSeries = useCallback(async () => {
    clearVolumeCache();
    clearOverlayCache();
    const list = await loadIndex();
    setSeries(list);
    setSelected(list[0] ?? null);
    if (list[0]) setIndices(middleSlice(list[0]));
    setLeft(list[0] ?? null);
    setRight(list[1] ?? list[0] ?? null);
    return list;
  }, []);

  /** Select a patient: load their overview and any results already on disk. */
  const selectPatient = useCallback(async (patientId: string) => {
    setError(null);
    setActiveRegion(null);
    try {
      const overview = await api.patient(patientId);
      setPatient(overview);
      if (overview.has_results) {
        const r = await api.results(patientId).catch(() => null);
        setResults(r);
        setActivePair(r?.change?.pairs?.[0] ? `${r.change.pairs[0].from}_${r.change.pairs[0].to}` : null);
      } else {
        setResults(null);
        setActivePair(null);
      }
      if (overview.exported) await reloadSeries();
      else setSeries([]);
    } catch (e) {
      setError((e as Error).message);
    }
  }, [reloadSeries]);

  /** Run a job to completion, streaming its progress into the UI. */
  const runJob = useCallback((start: () => Promise<Job>, onDone: (job: Job) => void) => {
    setError(null);
    start()
      .then((started) => {
        setJob(started);
        cancelWatch.current?.();
        cancelWatch.current = watchJob(started.job_id, setJob, (finished) => {
          setJob(finished);
          if (finished.status === "done") onDone(finished);
        });
      })
      .catch((e: Error) => setError(e.message));
  }, []);

  const fetchPatient = useCallback((patientId: string) => {
    runJob(() => api.fetchPatient(patientId), async () => {
      await reloadSeries();
      await selectPatient(patientId);
    });
  }, [runJob, reloadSeries, selectPatient]);

  const runInference = useCallback((patientId: string) => {
    runJob(() => api.infer(patientId, ALL_TASKS), async () => {
      const r = await api.results(patientId).catch(() => null);
      setResults(r);
      const first = r?.change?.pairs?.[0];
      setActivePair(first ? `${first.from}_${first.to}` : null);
      await reloadSeries();
    });
  }, [runJob, reloadSeries]);

  const pickRandom = useCallback(() => {
    api.randomPatient("test", 2)
      .then((p) => { setPatient(p); fetchPatient(p.patient_id); })
      .catch((e: Error) => setError(e.message));
  }, [fetchPatient]);

  // Load the overlay volumes the current results point at.
  useEffect(() => {
    if (!results) { setOverlays([]); return; }
    const wanted: Promise<OverlayVolume>[] = [];
    const pair = results.change?.pairs.find((p) => `${p.from}_${p.to}` === activePair)
      ?? results.change?.pairs[0];
    if (pair?.overlay_dir) wanted.push(loadOverlay(pair.overlay_dir, "change"));
    if (results.attention?.overlay_dir) wanted.push(loadOverlay(results.attention.overlay_dir, "attention"));

    let cancelled = false;
    Promise.allSettled(wanted).then((settled) => {
      if (cancelled) return;
      setOverlays(settled.flatMap((s) => (s.status === "fulfilled" ? [s.value] : [])));
    });
    return () => { cancelled = true; };
  }, [results, activePair]);

  // Deep link: ?patient=100002 reopens a case, and selecting one updates the URL
  // so a specific patient can be shared or returned to.
  useEffect(() => {
    const id = new URLSearchParams(window.location.search).get("patient");
    if (id) selectPatient(id);
  }, [selectPatient]);

  useEffect(() => {
    if (!patient) return;
    const url = new URL(window.location.href);
    if (url.searchParams.get("patient") === patient.patient_id) return;
    url.searchParams.set("patient", patient.patient_id);
    window.history.replaceState(null, "", url);
  }, [patient]);

  const select = (m: SeriesMeta) => { setSelected(m); setIndices(middleSlice(m)); };

  /** Clicking a round in the timeline loads that round into the viewer. */
  const selectRound = (round: string) => {
    const match = series.find((s) => s.round === round);
    if (match) select(match);
  };

  /** Clicking a change region jumps to its slice and frames it. */
  const selectRegion = (region: ChangeRegion, pair: ChangePair) => {
    setActiveRegion(region);
    setActivePair(`${pair.from}_${pair.to}`);
    setOverlayOn((o) => ({ ...o, change: true }));
    setPlane("axial");
    const baseline = series.find((s) => s.round === pair.from);
    if (baseline) {
      setSelected(baseline);
      // Region coordinates are on the 1.5 mm registration grid; rescale to the
      // series the viewer is actually showing.
      const scale = baseline.shape[0] / (pair.shape[0] || baseline.shape[0]);
      setIndices((prev) => ({ ...prev, axial: Math.round(region.slice * scale) }));
    }
    setTab("browse");
  };

  // Keyboard: arrows step slices, [ and ] step rounds (§8.2).
  useEffect(() => {
    if (tab !== "browse" || !selected) return;
    const depth = planeDepth(selected, plane);
    const onKey = (e: KeyboardEvent) => {
      if (e.target instanceof HTMLInputElement || e.target instanceof HTMLSelectElement) return;
      const step = e.shiftKey ? 5 : 1;
      if (e.key === "ArrowRight" || e.key === "ArrowUp") {
        setIndices((p) => ({ ...p, [plane]: Math.min(depth - 1, p[plane] + step) }));
      } else if (e.key === "ArrowLeft" || e.key === "ArrowDown") {
        setIndices((p) => ({ ...p, [plane]: Math.max(0, p[plane] - step) }));
      } else if (e.key === "[" || e.key === "]") {
        const ordered = series.filter((s) => s.round).sort((a, b) => (a.round ?? "").localeCompare(b.round ?? ""));
        const at = ordered.findIndex((s) => s.id === selected.id);
        const next = ordered[at + (e.key === "]" ? 1 : -1)];
        if (next) select(next);
      } else return;
      e.preventDefault();
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [tab, selected, plane, series]);

  const busy = job?.status === "running" || job?.status === "queued";
  // Only overlays belonging to the round on screen. Attention is computed on the
  // latest round and change maps on a pair's earlier round, so showing either
  // over a different scan would colour the wrong anatomy.
  const forThisRound = useMemo(
    () => overlays.filter((o) => overlayBelongsTo(o, selected?.round ?? null, selected?.patient ?? null)),
    [overlays, selected],
  );
  const activeOverlays = useMemo(
    () => forThisRound.filter((o) => overlayOn[o.kind]),
    [forThisRound, overlayOn],
  );
  const available = useMemo(() => ({
    change: forThisRound.some((o) => o.kind === "change"),
    attention: forThisRound.some((o) => o.kind === "attention"),
  }), [forThisRound]);

  const highlight = useMemo(() => {
    if (!activeRegion || !selected) return null;
    const pair = results?.change?.pairs.find((p) => `${p.from}_${p.to}` === activePair);
    const scaleY = selected.shape[1] / (pair?.shape[1] || selected.shape[1]);
    const scaleX = selected.shape[2] / (pair?.shape[2] || selected.shape[2]);
    const [, y0, x0, , y1, x1] = activeRegion.bbox;
    return {
      slice: indices.axial,
      box: [y0 * scaleY, x0 * scaleX, y1 * scaleY, x1 * scaleX] as [number, number, number, number],
    };
  }, [activeRegion, selected, results, activePair, indices.axial]);

  return (
    <div className="app">
      <header className="topbar">
        <div className="brand">
          Lung screening assistant
          <small>NLST · tissue change &amp; cancer risk</small>
        </div>
        <div className="topbar-spacer" />

        {patient && (
          <>
            <button className="cta secondary" onClick={() => fetchPatient(patient.patient_id)} disabled={busy}>
              {patient.exported ? "Re-fetch" : "Fetch scans"}
            </button>
            <button
              className="cta"
              onClick={() => runInference(patient.patient_id)}
              disabled={busy || !patient.exported}
              title={patient.exported ? "Run risk, change detection and saliency" : "Fetch the scans first"}
            >
              Analyse
            </button>
          </>
        )}
        <button className="cta secondary" onClick={pickRandom} disabled={busy}>
          Random held-out patient
        </button>

        <nav className="tabs">
          {TABS.map((t) => (
            <button key={t.id} className={`tab ${tab === t.id ? "active" : ""}`} onClick={() => setTab(t.id)}>
              {t.name}
            </button>
          ))}
        </nav>
      </header>

      <div className="workspace">
        <CohortRail selected={patient?.patient_id ?? null} onSelect={selectPatient} busy={!!busy} />

        <main className="center">
          {error && (
            <div className="notice">
              <strong>Something went wrong</strong>
              {error}
            </div>
          )}

          {job && (job.status !== "done" || busy) && (
            <JobProgress job={job} onDismiss={() => setJob(null)} />
          )}

          {series.length > 0 && (
            <OverlayControls
              available={available}
              active={overlayOn}
              onToggle={(kind) =>
                // One heavy overlay at a time by default (§8.3).
                setOverlayOn((o) => ({
                  attention: kind === "attention" ? !o.attention : false,
                  change: kind === "change" ? !o.change : false,
                }))
              }
              opacity={overlayOpacity}
              onOpacityChange={setOverlayOpacity}
              loaded={forThisRound}
            />
          )}

          {series.length === 0 ? (
            <EmptyStage patient={patient} busy={!!busy} onFetch={fetchPatient} onRandom={pickRandom} />
          ) : tab === "browse" ? (
            <BrowseView
              meta={selected} state={volumeState} plane={plane} onPlaneChange={setPlane}
              index={indices[plane]} onIndexChange={(i) => setIndices((p) => ({ ...p, [plane]: i }))}
              window={win} onWindowChange={setWin}
              overlays={activeOverlays} overlayOpacity={overlayOpacity} highlight={highlight}
              series={series} onSelect={select}
            />
          ) : tab === "compare" && left && right ? (
            <CompareView
              series={series} left={left} right={right} onLeftChange={setLeft} onRightChange={setRight}
              plane={plane} onPlaneChange={setPlane} window={win} onWindowChange={setWin}
              overlays={activeOverlays} overlayOpacity={overlayOpacity}
            />
          ) : tab === "volume" ? (
            <VolumeView
              meta={selected} state={volumeState} window={win}
              sliceIndex={indices.axial}
              onSliceIndexChange={(i) => setIndices((p) => ({ ...p, axial: i }))}
            />
          ) : (
            <div className="empty">Need at least two scans to compare.</div>
          )}
        </main>

        <ResultsPanel
          patient={patient}
          results={results}
          activeRound={selected?.round ?? null}
          onRoundSelect={selectRound}
          activePair={activePair}
          onPairSelect={setActivePair}
          activeRegion={activeRegion}
          onRegionSelect={selectRegion}
        />
      </div>

      <footer className="footer">
        <span>Research prototype on NLST data. Not a medical device; not for clinical use.</span>
        <span className="sep">|</span>
        <span>
          Risk model:{" "}
          {system?.risk.available ? system.risk.name : "not installed — change detection only"}
        </span>
        {system && (
          <>
            <span className="sep">|</span>
            <span>Cache {(system.cache.bytes / 1e9).toFixed(1)} GB of {(system.cache.cap_bytes / 1e9).toFixed(0)} GB</span>
          </>
        )}
      </footer>
    </div>
  );
}

/** The empty state (§8.4) — tells the reader what to do, not just that nothing is here. */
function EmptyStage({
  patient, busy, onFetch, onRandom,
}: {
  patient: PatientOverview | null;
  busy: boolean;
  onFetch: (id: string) => void;
  onRandom: () => void;
}) {
  if (!patient) {
    return (
      <div className="empty">
        <h2>Pick a patient</h2>
        <p>
          Choose someone from the cohort on the left, or jump straight to a random held-out patient
          the model has never been trained or tuned on.
        </p>
        <button className="cta" onClick={onRandom} disabled={busy}>Random held-out patient</button>
      </div>
    );
  }
  return (
    <div className="empty">
      <h2>Patient {patient.patient_id}</h2>
      <p>
        {patient.n_rounds} screening round{patient.n_rounds === 1 ? "" : "s"} ·{" "}
        {patient.total_mb.toFixed(0)} MB. Nothing is downloaded until you ask — fetching streams
        the scans from the Imaging Data Commons and replaces whatever is currently loaded.
      </p>
      <button className="cta" onClick={() => onFetch(patient.patient_id)} disabled={busy}>
        Fetch scans
      </button>
    </div>
  );
}
