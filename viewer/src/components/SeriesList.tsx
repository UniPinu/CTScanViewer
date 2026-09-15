import { kernelLabel, visitLabel } from "../data";
import type { SeriesMeta } from "../types";

interface Props {
  series: SeriesMeta[];
  selectedId: string | null;
  onSelect: (m: SeriesMeta) => void;
}

/** Sidebar listing every exported series, grouped by patient. */
export function SeriesList({ series, selectedId, onSelect }: Props) {
  const byPatient = new Map<string, SeriesMeta[]>();
  for (const m of series) {
    const list = byPatient.get(m.patient) ?? [];
    list.push(m);
    byPatient.set(m.patient, list);
  }
  return (
    <nav className="series-list">
      {[...byPatient.entries()].map(([patient, items]) => (
        <section key={patient}>
          <h4>Patient {patient}</h4>
          {items.map((m) => (
            <button key={m.id} className={`series-item ${m.id === selectedId ? "active" : ""}`} onClick={() => onSelect(m)}>
              <span className="series-visit">{visitLabel(m)}</span>
              <span className="series-detail">
                {kernelLabel(m)} · {m.shape[0]} slices · {+m.spacing[0].toFixed(2)} mm
              </span>
            </button>
          ))}
        </section>
      ))}
    </nav>
  );
}

/** Compact dropdown alternative, used by the compare view. */
export function SeriesSelect({ series, value, onChange }: { series: SeriesMeta[]; value: string; onChange: (m: SeriesMeta) => void }) {
  return (
    <select className="series-select" value={value} onChange={(e) => {
      const m = series.find((s) => s.id === e.target.value);
      if (m) onChange(m);
    }}>
      {series.map((m) => (
        <option key={m.id} value={m.id}>
          {m.patient} · {visitLabel(m)} · {kernelLabel(m)}
        </option>
      ))}
    </select>
  );
}
