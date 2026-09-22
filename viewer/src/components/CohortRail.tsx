import { useEffect, useState } from "react";
import { api, type CohortPatient, type Split } from "../api";

/**
 * The left rail (CONTEXT.md §8.2): a quiet, scannable cohort list.
 *
 * Filtering by split is the point of it. A demo that can say "these are
 * held-out test patients only" is making an honest claim about what the model
 * has seen, and the filter is what backs that claim up.
 */

interface Props {
  selected: string | null;
  onSelect: (patientId: string) => void;
  busy: boolean;
}

const SPLITS: (Split | "all")[] = ["test", "val", "train", "all"];

export function CohortRail({ selected, onSelect, busy }: Props) {
  const [split, setSplit] = useState<Split | "all">("test");
  const [minRounds, setMinRounds] = useState(2);
  const [label, setLabel] = useState<number | undefined>(undefined);
  const [query, setQuery] = useState("");
  const [patients, setPatients] = useState<CohortPatient[] | null>(null);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    let cancelled = false;
    setPatients(null);
    setError(null);
    // Debounce the search so typing an id does not fire a request per keystroke.
    const timer = setTimeout(() => {
      api
        .cohort(split, 60, minRounds, query, label)
        .then((r) => !cancelled && setPatients(r.patients))
        .catch((e: Error) => !cancelled && setError(e.message));
    }, query ? 250 : 0);
    return () => {
      cancelled = true;
      clearTimeout(timer);
    };
  }, [split, minRounds, query, label]);

  return (
    <nav className="rail rail-left">
      <div className="rail-head">
        <h3>Cohort</h3>
        <div className="chips">
          {SPLITS.map((s) => (
            <button key={s} className={`chip ${split === s ? "active" : ""}`} onClick={() => setSplit(s)}>
              {s}
            </button>
          ))}
        </div>
        <div className="chips" style={{ marginTop: 6 }}>
          <button
            className={`chip ${minRounds === 2 ? "active" : ""}`}
            onClick={() => setMinRounds(minRounds === 2 ? 1 : 2)}
            title="Change detection needs at least two rounds"
          >
            2+ rounds
          </button>
          {/* Outcome is trial ground truth from the NLST participant table, so
              cases with a known diagnosis can be found without downloading. */}
          <button
            className={`chip ${label === 1 ? "active" : ""}`}
            onClick={() => setLabel(label === 1 ? undefined : 1)}
            title="Patients diagnosed with lung cancer during the trial"
          >
            cancer
          </button>
          <button
            className={`chip ${label === 0 ? "active" : ""}`}
            onClick={() => setLabel(label === 0 ? undefined : 0)}
            title="Patients with no diagnosis on record"
          >
            no cancer
          </button>
        </div>
        <input
          className="rail-search"
          placeholder="Find a patient id…"
          value={query}
          onChange={(e) => setQuery(e.target.value)}
        />
      </div>

      <div className="cohort-list">
        {error && <p className="hint">{error}</p>}
        {!patients && !error && <p className="hint">Loading cohort…</p>}
        {patients?.length === 0 && <p className="hint">No patient matches these filters.</p>}
        {patients?.map((p) => (
          <button
            key={p.patient_id}
            className={`cohort-row ${selected === p.patient_id ? "active" : ""}`}
            onClick={() => onSelect(p.patient_id)}
            disabled={busy}
          >
            <span>
              <span className="cohort-id">{p.patient_id}</span>
              <span className="cohort-meta"> · {p.n_rounds} rounds</span>
              {p.label === 1 && <span className="region-kind"> · cancer</span>}
            </span>
            {/* Redundant when the list is already filtered to one split (§8.2: no noise). */}
            {split === "all" && <span className={`split-chip ${p.split}`}>{p.split}</span>}
          </button>
        ))}
      </div>
    </nav>
  );
}
