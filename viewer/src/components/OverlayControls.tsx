import type { OverlayVolume } from "../overlay";

/**
 * Overlay toggles, opacity and legend (CONTEXT.md §8.3).
 *
 * "Only one heavy overlay on at a time by default" — turning one on turns the
 * other off unless the reader deliberately holds both, which keeps the image
 * legible and keeps the palette down to one accent at a time.
 */

interface Props {
  available: { attention: boolean; change: boolean };
  active: { attention: boolean; change: boolean };
  onToggle: (kind: "attention" | "change") => void;
  opacity: number;
  onOpacityChange: (v: number) => void;
  loaded: OverlayVolume[];
}

export function OverlayControls({ available, active, onToggle, opacity, onOpacityChange, loaded }: Props) {
  const anyAvailable = available.attention || available.change;
  if (!anyAvailable) return null;

  const showing = loaded.filter((o) => active[o.kind]);

  return (
    <div className="center-controls">
      <span className="muted" style={{ fontSize: 12 }}>Overlay</span>
      <button
        className={`chip ${active.change ? "active" : ""}`}
        onClick={() => onToggle("change")}
        disabled={!available.change}
        title={available.change ? "Regions that densified between rounds" : "Run change detection first"}
      >
        Change
      </button>
      <button
        className={`chip ${active.attention ? "active" : ""}`}
        onClick={() => onToggle("attention")}
        disabled={!available.attention}
        title={available.attention ? "Where the risk model attended" : "Needs a risk model"}
      >
        Attention
      </button>

      {showing.length > 0 && (
        <>
          <label style={{ display: "flex", alignItems: "center", gap: 8, minWidth: 190 }}>
            <span className="muted" style={{ fontSize: 12 }}>Opacity</span>
            <input
              type="range" min={0} max={100} value={Math.round(opacity * 100)}
              onChange={(e) => onOpacityChange(+e.target.value / 100)}
            />
            <span className="slider-value" style={{ minWidth: 34 }}>{Math.round(opacity * 100)}%</span>
          </label>
          {showing.map((o) => (
            <span key={o.kind} className="legend" style={{ minWidth: 150 }}>
              <span className={`legend-ramp ${o.kind}`} />
              <span>{o.meta.units}</span>
            </span>
          ))}
        </>
      )}
    </div>
  );
}
