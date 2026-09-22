import type { RiskBlock } from "../api";

/**
 * The risk gauge and 1–6-year curve (CONTEXT.md §8.3).
 *
 * The brief is emphatic about tone: a calm horizontal bar, not a red dial; the
 * number large but neutral; the accent colour only once the value is clearly
 * above the cohort median. This is decision support, and making a screening
 * number feel alarming would be a design failure, not a safety feature.
 *
 * When no risk model is installed the panel says exactly that. It does not
 * render a zero, a dash that reads as "low", or a placeholder curve — §11 rules
 * out anything that implies a prediction the system did not make.
 */

interface Props {
  risk: RiskBlock | undefined;
  /** Ratio of the gauge width used for the full scale. 1-year risk is small. */
  scaleMax?: number;
}

const PERCENT = (v: number) => `${(v * 100).toFixed(1)}%`;

export function RiskPanel({ risk, scaleMax = 0.1 }: Props) {
  if (!risk) {
    return (
      <div className="panel">
        <h3>Risk</h3>
        <p className="hint">Run inference to estimate risk for this patient.</p>
      </div>
    );
  }

  if (risk.unavailable) {
    return (
      <div className="panel">
        <h3>Risk</h3>
        <div className="notice">
          <strong>No risk model installed</strong>
          Sybil needs its own Python 3.10 environment, so this deployment reports tissue change
          only. No estimate is shown rather than a placeholder.
          {risk.reason && (
            <details style={{ marginTop: 8 }}>
              <summary>How to enable it</summary>
              <pre>{risk.reason}</pre>
            </details>
          )}
        </div>
      </div>
    );
  }

  const year1 = risk.year_1 ?? 0;
  const percentile = risk.cohort_percentile;
  const aboveMedian = percentile !== null && percentile !== undefined && percentile >= 50;
  const pct = (v: number) => `${Math.min(100, (v / scaleMax) * 100)}%`;

  return (
    <div className="panel">
      <h3>1-year risk</h3>
      <div className="risk-value tabular">{PERCENT(year1)}</div>
      <div className="risk-caption">
        {percentile === null || percentile === undefined
          ? "No cohort reference computed yet, so no percentile is shown."
          : `${aboveMedian ? "Above" : "Below"} the screening cohort median · ${percentile}th percentile`}
      </div>

      <div className="gauge" role="img" aria-label={`1-year risk ${PERCENT(year1)}`}>
        {risk.ci_1 && (
          <div
            className="gauge-band"
            style={{ left: pct(risk.ci_1[0]), width: pct(risk.ci_1[1] - risk.ci_1[0]) }}
          />
        )}
        <div className={`gauge-fill ${aboveMedian ? "above" : ""}`} style={{ width: pct(year1) }} />
        {/* The cohort median, marked as a tick rather than a second bar. */}
        <div className="gauge-tick" style={{ left: pct(scaleMax / 2) }} title="cohort median" />
      </div>
      <div className="gauge-scale">
        <span>0%</span>
        <span>{(scaleMax * 50).toFixed(0)}%</span>
        <span>{(scaleMax * 100).toFixed(0)}%</span>
      </div>

      {risk.curve && risk.curve.length > 1 && <RiskCurve curve={risk.curve} />}

      <p className="hint">
        {risk.calibrated
          ? "Recalibrated on this cohort's training split."
          : "Raw model output — not yet recalibrated on this cohort, so treat the absolute value as indicative."}
        {risk.model ? ` Model: ${risk.model}.` : ""}
      </p>
    </div>
  );
}

/** Understated sparkline of cumulative risk by horizon, with hairline gridlines. */
function RiskCurve({ curve }: { curve: number[] }) {
  const width = 300;
  const height = 74;
  const pad = { top: 8, right: 8, bottom: 16, left: 30 };
  const max = Math.max(...curve, 0.01) * 1.15;

  const x = (i: number) => pad.left + (i / (curve.length - 1)) * (width - pad.left - pad.right);
  const y = (v: number) => pad.top + (1 - v / max) * (height - pad.top - pad.bottom);
  const path = curve.map((v, i) => `${i === 0 ? "M" : "L"}${x(i).toFixed(1)},${y(v).toFixed(1)}`).join(" ");

  return (
    <>
      <h3 style={{ marginTop: 18 }}>Risk by horizon</h3>
      <svg className="curve" viewBox={`0 0 ${width} ${height}`} role="img" aria-label="cumulative risk by year">
        {[0, max / 2, max].map((v) => (
          <g key={v}>
            <line
              x1={pad.left} x2={width - pad.right} y1={y(v)} y2={y(v)}
              stroke="var(--line)" strokeWidth="1"
            />
            <text x={pad.left - 5} y={y(v) + 3} textAnchor="end" fontSize="9" fill="var(--ink-faint)">
              {(v * 100).toFixed(0)}%
            </text>
          </g>
        ))}
        <path d={path} fill="none" stroke="var(--accent)" strokeWidth="1.5" strokeLinejoin="round" />
        {curve.map((v, i) => (
          <circle key={i} cx={x(i)} cy={y(v)} r="2" fill="var(--accent)" />
        ))}
        {curve.map((_, i) => (
          <text key={i} x={x(i)} y={height - 4} textAnchor="middle" fontSize="9" fill="var(--ink-faint)">
            {i + 1}
          </text>
        ))}
      </svg>
    </>
  );
}
