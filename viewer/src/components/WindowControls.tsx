import { PRESETS, presetName } from "../windowing";
import type { HUWindow } from "../types";

interface Props {
  value: HUWindow;
  onChange: (w: HUWindow) => void;
}

/** Preset buttons plus centre/width sliders for the display window. */
export function WindowControls({ value, onChange }: Props) {
  const active = presetName(value);
  const lo = value.center - value.width / 2;
  const hi = value.center + value.width / 2;
  return (
    <div className="panel">
      <h3>Window</h3>
      <div className="chips">
        {Object.entries(PRESETS).map(([name, w]) => (
          <button key={name} className={`chip ${active === name ? "active" : ""}`} onClick={() => onChange(w)}>
            {name}
          </button>
        ))}
      </div>
      <label className="field">
        <span>Centre <b>{value.center} HU</b></span>
        <input type="range" min={-1000} max={1500} step={5} value={value.center}
          onChange={(e) => onChange({ ...value, center: +e.target.value })} />
      </label>
      <label className="field">
        <span>Width <b>{value.width} HU</b></span>
        <input type="range" min={10} max={3000} step={10} value={value.width}
          onChange={(e) => onChange({ ...value, width: +e.target.value })} />
      </label>
      <p className="hint">
        Showing {Math.round(lo)} … {Math.round(hi)} HU as black … white.
        Drag on the image to adjust: left/right = width, up/down = centre.
      </p>
    </div>
  );
}
