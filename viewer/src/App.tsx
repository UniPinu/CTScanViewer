import { useCallback, useEffect, useMemo, useState } from "react";
import { BrowseView } from "./components/BrowseView";
import { CompareView } from "./components/CompareView";
import { FetchPatient } from "./components/FetchPatient";
import { SeriesList } from "./components/SeriesList";
import { VolumeView } from "./components/VolumeView";
import { clearVolumeCache, loadIndex, planeDepth } from "./data";
import { useVolume } from "./hooks";
import type { HUWindow, Plane, SeriesMeta } from "./types";
import { DEFAULT_WINDOW } from "./windowing";

type Tab = "browse" | "compare" | "volume";
const TABS: { id: Tab; name: string }[] = [
  { id: "browse", name: "Browse" },
  { id: "compare", name: "Before / after" },
  { id: "volume", name: "3-D" },
];

/** Pick a sensible default pair: the same person's earliest and next visit, same kernel style if possible. */
function defaultPair(series: SeriesMeta[]): [SeriesMeta, SeriesMeta] | null {
  if (series.length < 2) return null;
  const byPatient = new Map<string, SeriesMeta[]>();
  for (const m of series) byPatient.set(m.patient, [...(byPatient.get(m.patient) ?? []), m]);
  for (const items of byPatient.values()) {
    const dates = [...new Set(items.map((m) => m.studyDate))].sort();
    if (dates.length < 2) continue;
    const first = items.filter((m) => m.studyDate === dates[0]);
    const second = items.filter((m) => m.studyDate === dates[1]);
    const left = first[0];
    const right = second.find((m) => m.kernelStyle === left.kernelStyle) ?? second[0];
    return [left, right];
  }
  return [series[0], series[1]];
}

const middle = (m: SeriesMeta): Record<Plane, number> => ({
  axial: Math.floor(planeDepth(m, "axial") / 2),
  coronal: Math.floor(planeDepth(m, "coronal") / 2),
  sagittal: Math.floor(planeDepth(m, "sagittal") / 2),
});

export default function App() {
  const [series, setSeries] = useState<SeriesMeta[] | null>(null);
  const [loadError, setLoadError] = useState<string | null>(null);
  const [tab, setTab] = useState<Tab>("browse");

  const [selected, setSelected] = useState<SeriesMeta | null>(null);
  const [plane, setPlane] = useState<Plane>("axial");
  const [indices, setIndices] = useState<Record<Plane, number>>({ axial: 0, coronal: 0, sagittal: 0 });
  const [win, setWin] = useState<HUWindow>(DEFAULT_WINDOW);

  const [left, setLeft] = useState<SeriesMeta | null>(null);
  const [right, setRight] = useState<SeriesMeta | null>(null);

  const volumeState = useVolume(selected);

  /** (Re)read index.json and reset the selection to the first scan. */
  const reload = useCallback(() => {
    clearVolumeCache();
    loadIndex()
      .then((list) => {
        setSeries(list);
        setLoadError(null);
        setSelected(list[0] ?? null);
        if (list.length) setIndices(middle(list[0]));
        const pair = defaultPair(list);
        setLeft(pair ? pair[0] : null);
        setRight(pair ? pair[1] : null);
      })
      .catch((e: Error) => setLoadError(e.message));
  }, []);

  useEffect(reload, [reload]);

  const select = (m: SeriesMeta) => {
    setSelected(m);
    setIndices(middle(m));
  };

  const setIndex = (i: number) => setIndices((prev) => ({ ...prev, [plane]: i }));
  const setAxial = (i: number) => setIndices((prev) => ({ ...prev, axial: i }));

  // Arrow keys step through slices in the browse view.
  useEffect(() => {
    if (tab !== "browse" || !selected) return;
    const depth = planeDepth(selected, plane);
    const onKey = (e: KeyboardEvent) => {
      if (e.target instanceof HTMLInputElement || e.target instanceof HTMLSelectElement) return;
      const step = e.shiftKey ? 5 : 1;
      if (e.key === "ArrowUp" || e.key === "ArrowRight") setIndices((p) => ({ ...p, [plane]: Math.min(depth - 1, p[plane] + step) }));
      else if (e.key === "ArrowDown" || e.key === "ArrowLeft") setIndices((p) => ({ ...p, [plane]: Math.max(0, p[plane] - step) }));
      else return;
      e.preventDefault();
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [tab, selected, plane]);

  const totalMB = useMemo(
    () => (series ?? []).reduce((s, m) => s + (m.shape[0] * m.shape[1] * m.shape[2] * 2) / 1e6, 0),
    [series],
  );
  const patientLabel = useMemo(() => {
    const ids = [...new Set((series ?? []).map((m) => m.patient))];
    return ids.length === 1 ? `patient ${ids[0]}` : `${ids.length} patients`;
  }, [series]);

  if (loadError || (series && series.length === 0)) {
    return (
      <div className="empty">
        <h2>No scans on disk yet</h2>
        <p className="muted">
          Fetch one random patient from the Imaging Data Commons — every CT scan they have, usually 100–600 MB.
          Or, from the project folder, run <code>python fetch_patient.py</code>.
        </p>
        <FetchPatient onComplete={reload} prominent />
      </div>
    );
  }
  if (!series) return <div className="empty">Loading index…</div>;

  return (
    <div className="app">
      <header className="topbar">
        <div className="brand">
          CT Viewer <span className="muted">· {patientLabel} · {series.length} scans · {totalMB.toFixed(0)} MB</span>
        </div>
        <div className="topbar-actions">
          <FetchPatient onComplete={reload} />
        </div>
        <nav className="tabs">
          {TABS.map((t) => (
            <button key={t.id} className={`tab ${tab === t.id ? "active" : ""}`} onClick={() => setTab(t.id)}>{t.name}</button>
          ))}
        </nav>
      </header>

      <div className="body">
        {tab !== "compare" && <SeriesList series={series} selectedId={selected?.id ?? null} onSelect={select} />}

        {tab === "browse" && (
          <BrowseView meta={selected} state={volumeState} plane={plane} onPlaneChange={setPlane}
            index={indices[plane]} onIndexChange={setIndex} window={win} onWindowChange={setWin} />
        )}
        {tab === "compare" && left && right && (
          <CompareView series={series} left={left} right={right} onLeftChange={setLeft} onRightChange={setRight}
            plane={plane} onPlaneChange={setPlane} window={win} onWindowChange={setWin} />
        )}
        {tab === "compare" && !(left && right) && <div className="empty">Need at least two scans to compare.</div>}
        {tab === "volume" && (
          <VolumeView meta={selected} state={volumeState} window={win}
            sliceIndex={indices.axial} onSliceIndexChange={setAxial} />
        )}
      </div>
    </div>
  );
}
