import type { ChangePair, ChangeRegion, PatientOverview, Results } from "../api";
import { PatientCard } from "./PatientCard";
import { RiskPanel } from "./RiskPanel";

/**
 * The right rail (CONTEXT.md §8.2, §8.3): risk, screening timeline, change
 * regions, the serif summary, and a persistent provenance line.
 *
 * The summary card restates the numbers from the results object verbatim — it
 * is rendered server-side from a template (`ml/summary.py`) precisely so that
 * the prose and the figures beside it cannot drift apart.
 */

interface Props {
  patient: PatientOverview | null;
  results: Results | null;
  activeRound: string | null;
  onRoundSelect: (round: string) => void;
  activePair: string | null;
  onPairSelect: (pair: string) => void;
  activeRegion: ChangeRegion | null;
  onRegionSelect: (region: ChangeRegion, pair: ChangePair) => void;
}

export function ResultsPanel({
  patient, results, activeRound, onRoundSelect, activePair, onPairSelect,
  activeRegion, onRegionSelect,
}: Props) {
  if (!patient) {
    return (
      <div className="rail rail-right">
        <div className="panel">
          <h3>Results</h3>
          <p className="hint">Choose a patient to see their screening rounds and results.</p>
        </div>
      </div>
    );
  }

  const pairs = results?.change?.pairs ?? [];

  return (
    <div className="rail rail-right">
      <ScreeningTimeline
        patient={patient}
        activeRound={activeRound}
        onRoundSelect={onRoundSelect}
        activePair={activePair}
        pairs={pairs}
      />

      <RiskPanel risk={results?.risk} />

      <PatientCard patient={patient} />

      {results?.change && (
        <ChangePanel
          change={results.change}
          activePair={activePair}
          onPairSelect={onPairSelect}
          activeRegion={activeRegion}
          onRegionSelect={onRegionSelect}
        />
      )}

      {results?.summary && <SummaryCard summary={results.summary} results={results} />}
    </div>
  );
}

/** T0 · T1 · T2 as dots on a line; clicking one loads that round (§8.3). */
function ScreeningTimeline({
  patient, activeRound, onRoundSelect, activePair, pairs,
}: {
  patient: PatientOverview;
  activeRound: string | null;
  onRoundSelect: (round: string) => void;
  activePair: string | null;
  pairs: ChangePair[];
}) {
  const pair = pairs.find((p) => `${p.from}_${p.to}` === activePair);
  const inPair = new Set(pair ? [pair.from, pair.to] : []);

  return (
    <div className="panel">
      <h3>Screening timeline</h3>
      <div className="timeline">
        {patient.rounds.map((round) => (
          <button
            key={round.round}
            className={`timeline-round ${activeRound === round.round ? "active" : ""} ${
              inPair.has(round.round) ? "in-pair" : ""
            }`}
            onClick={() => onRoundSelect(round.round)}
            title={`${round.instances} slices · ${round.size_mb} MB · ${round.kernel ?? "?"}`}
          >
            <span className="timeline-dot" />
            <span className="timeline-label">{round.round}</span>
            <span className="timeline-date">{round.study_date}</span>
          </button>
        ))}
      </div>
      <p className="hint">
        Patient {patient.patient_id} · <span className={`split-chip ${patient.split}`}>{patient.split}</span>
        {patient.split === "test" && " — never used for training or tuning."}
      </p>
    </div>
  );
}

function ChangePanel({
  change, activePair, onPairSelect, activeRegion, onRegionSelect,
}: {
  change: { pairs: ChangePair[]; note?: string };
  activePair: string | null;
  onPairSelect: (pair: string) => void;
  activeRegion: ChangeRegion | null;
  onRegionSelect: (region: ChangeRegion, pair: ChangePair) => void;
}) {
  if (change.note && change.pairs.length === 0) {
    return (
      <div className="panel">
        <h3>Tissue change</h3>
        <p className="hint">{change.note}</p>
      </div>
    );
  }

  const selected = change.pairs.find((p) => `${p.from}_${p.to}` === activePair) ?? change.pairs[0];
  const registration = selected?.registration ?? {};
  const failed = registration.registration_failed === true;

  return (
    <div className="panel">
      <h3>Tissue change</h3>
      <div className="chips">
        {change.pairs.map((p) => {
          const key = `${p.from}_${p.to}`;
          return (
            <button
              key={key}
              className={`chip ${key === (activePair ?? `${change.pairs[0].from}_${change.pairs[0].to}`) ? "active" : ""}`}
              onClick={() => onPairSelect(key)}
            >
              {p.from} → {p.to}
            </button>
          );
        })}
      </div>

      {failed && (
        <div className="notice" style={{ marginTop: 10 }}>
          <strong>Could not align these rounds</strong>
          {selected.note ??
            "No registration stage aligned these rounds better than leaving them unregistered, " +
              "so no regions are reported — anything found would be breathing, not growth."}
        </div>
      )}

      {selected && (
        <>
          <dl className="facts" style={{ marginTop: 12 }}>
            <dt>Regions</dt>
            <dd>{selected.n_regions}</dd>
            <dt>Alignment</dt>
            <dd>
              {registration.alignment_before_hu} → {registration.alignment_after_hu} HU
              {registration.selected_stage && registration.selected_stage !== "none" && (
                <span className="faint"> ({registration.selected_stage})</span>
              )}
            </dd>
            <dt>Lung volume</dt>
            <dd>
              {selected.lung_volume_ml.from.toFixed(0)} → {selected.lung_volume_ml.to.toFixed(0)} mL
            </dd>
          </dl>

          <div className="regions">
            {selected.regions.slice(0, 8).map((region, i) => (
              <button
                key={i}
                className={`region-row ${activeRegion === region ? "active" : ""}`}
                onClick={() => onRegionSelect(region, selected)}
              >
                <span className="region-size tabular">{region.delta_mm.toFixed(1)} mm</span>
                <span className="region-where">
                  {region.side} lung · slice {region.slice} · {region.baseline_hu.toFixed(0)} →{" "}
                  {region.followup_hu.toFixed(0)} HU
                </span>
                <span className="region-kind">{region.kind}</span>
              </button>
            ))}
          </div>
          {selected.regions.length === 0 && !failed && (
            <p className="hint">No region met the change thresholds for this pair.</p>
          )}
          <p className="hint">
            Candidate regions for a human to review, not findings. Thresholds are physically
            motivated but not yet validated against annotated positives.
          </p>
        </>
      )}
    </div>
  );
}

/** Serif prose, generously set — where the tool "speaks" (§8.3). */
function SummaryCard({ summary, results }: { summary: string; results: Results }) {
  // The disclaimer is the last sentence of the composed summary; pull it out so
  // it can be set quietly rather than as part of the prose.
  const marker = "Research prototype";
  const index = summary.indexOf(marker);
  const body = index > 0 ? summary.slice(0, index).trim() : summary;
  const disclaimer = index > 0 ? summary.slice(index).trim() : null;

  return (
    <div className="panel">
      <h3>Summary</h3>
      <div className="summary">
        <p>{body}</p>
        {disclaimer && <p className="disclaimer">{disclaimer}</p>}
      </div>
      <details style={{ marginTop: 10 }}>
        <summary>Provenance</summary>
        <dl className="facts" style={{ marginTop: 8 }}>
          <dt>Model</dt>
          <dd>{results.provenance.model}</dd>
          <dt>Code</dt>
          <dd>{results.provenance.code_rev}</dd>
          <dt>Generated</dt>
          <dd>{new Date(results.provenance.generated_at).toLocaleString()}</dd>
          <dt>Split</dt>
          <dd>{results.split}</dd>
        </dl>
      </details>
    </div>
  );
}
