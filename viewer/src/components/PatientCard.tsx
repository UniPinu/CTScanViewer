import type { PatientOverview } from "../api";

/**
 * Everything known about a patient before any imaging is downloaded.
 *
 * This is the answer to "is this case worth 200 MB and four minutes?". Both
 * halves come from metadata alone: what is *stored* (rounds, reconstructions,
 * slice thickness, sizes, what is already cached) from IDC's imaging index, and
 * what the *outcome* was — diagnosed or not, when, stage, histology, and the
 * official read of each screening round — from IDC's NLST participant table.
 *
 * The outcome is shown factually and without alarm styling. It is trial ground
 * truth, not a prediction, and §11's tone rules apply to it just as much: the
 * accent is reserved for the one line that says a diagnosis is on record.
 */

export function PatientCard({ patient }: { patient: PatientOverview }) {
  const clinical = patient.clinical;
  const outcome = clinical?.outcome;
  const diagnosed = outcome?.diagnosed === true;

  return (
    <div className="panel">
      <h3>Before you download</h3>

      <dl className="facts">
        <dt>Patient</dt>
        <dd>
          {patient.patient_id} <span className={`split-chip ${patient.split}`}>{patient.split}</span>
        </dd>
        <dt>Rounds</dt>
        <dd>{patient.n_rounds} screening round{patient.n_rounds === 1 ? "" : "s"}</dd>
        <dt>Download</dt>
        <dd>
          {patient.primary_mb.toFixed(0)} MB
          {patient.cached_mb > 0 && (
            <span className="faint"> · {patient.cached_mb.toFixed(0)} MB already cached</span>
          )}
        </dd>
        <dt>All series</dt>
        <dd className="faint">
          {patient.n_series} including alternates · {patient.total_mb.toFixed(0)} MB
        </dd>
      </dl>

      {/* Trial outcome — the ground truth, not a model output. */}
      {clinical?.found ? (
        <div className={`outcome ${diagnosed ? "positive" : ""}`}>
          <div className="outcome-headline">
            {diagnosed ? "Lung cancer diagnosed during the trial" : "No lung-cancer diagnosis on record"}
          </div>
          <div className="outcome-detail">
            {diagnosed ? (
              <>
                {outcome?.diagnosis_study_year && <>Study year {outcome.diagnosis_study_year} · </>}
                {outcome?.years_to_diagnosis != null && (
                  <>{outcome.years_to_diagnosis.toFixed(1)} years from randomisation</>
                )}
                {outcome?.screen_result_at_diagnosis && <> · {outcome.screen_result_at_diagnosis}</>}
              </>
            ) : (
              outcome?.cancer_free_days != null && (
                <>Followed {(outcome.cancer_free_days / 365.25).toFixed(1)} years without a diagnosis</>
              )
            )}
          </div>
        </div>
      ) : (
        <p className="hint">
          No NLST participant record for this patient
          {clinical?.error ? ` (${clinical.error})` : ""}.
        </p>
      )}

      {clinical?.tumour && clinical.tumour.length > 0 && (
        <>
          <h3 style={{ marginTop: 16 }}>Tumour</h3>
          <dl className="facts">
            {clinical.tumour.map((f) => (
              <Fragment key={f.column} label={f.label} value={f.value} />
            ))}
            {clinical.locations && clinical.locations.length > 0 && (
              <>
                <dt>Location</dt>
                <dd>{clinical.locations.join(", ")}</dd>
              </>
            )}
          </dl>
        </>
      )}

      {clinical?.demographics && clinical.demographics.length > 0 && (
        <>
          <h3 style={{ marginTop: 16 }}>Participant</h3>
          <dl className="facts">
            {clinical.demographics.map((f) => (
              <Fragment key={f.column} label={f.label} value={f.value} />
            ))}
          </dl>
        </>
      )}

      <h3 style={{ marginTop: 16 }}>Rounds on file</h3>
      <div className="rounds">
        {patient.rounds.map((r) => (
          <div key={r.series_uid} className="round-row">
            <span className="round-name">{r.round}</span>
            <span className="round-detail">
              {r.study_date} · {r.instances} slices · {r.slice_mm ?? "?"} mm · {r.kernel ?? "?"}
              {r.alternate_reconstructions > 0 && (
                <span className="faint"> · +{r.alternate_reconstructions} alt</span>
              )}
              {r.screen_result && <div className="round-screen">{r.screen_result}</div>}
            </span>
            <span className="round-size tabular">
              {r.size_mb.toFixed(0)} MB
              {r.cached && <span className="cached-dot" title="already cached" />}
            </span>
          </div>
        ))}
      </div>

      <p className="hint">
        All of the above comes from IDC's metadata and NLST participant tables — no imaging has
        been downloaded to show it.
      </p>
    </div>
  );
}

/** A <dt>/<dd> pair that renders nothing when the value is absent. */
function Fragment({ label, value }: { label: string; value: string | null }) {
  if (!value) return null;
  return (
    <>
      <dt>{label}</dt>
      <dd>{value}</dd>
    </>
  );
}
