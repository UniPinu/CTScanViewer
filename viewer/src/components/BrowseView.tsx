import { planeDepth, planePositionMm, visitLabel } from "../data";
import type { VolumeState } from "../hooks";
import type { HUWindow, Plane, SeriesMeta } from "../types";
import { SliceCanvas } from "./SliceCanvas";
import { WindowControls } from "./WindowControls";

const PLANES: { id: Plane; name: string; blurb: string }[] = [
  { id: "axial", name: "Axial", blurb: "horizontal cuts, viewed from the feet" },
  { id: "coronal", name: "Coronal", blurb: "front-to-back sheets, viewed from the front" },
  { id: "sagittal", name: "Sagittal", blurb: "side-to-side sheets, viewed from the left" },
];

interface Props {
  meta: SeriesMeta | null;
  state: VolumeState;
  plane: Plane;
  onPlaneChange: (p: Plane) => void;
  index: number;
  onIndexChange: (i: number) => void;
  window: HUWindow;
  onWindowChange: (w: HUWindow) => void;
}

/** Single-series viewer: pick a plane, scroll through slices, adjust the window. */
export function BrowseView({ meta, state, plane, onPlaneChange, index, onIndexChange, window: win, onWindowChange }: Props) {
  if (!meta) return <div className="empty">Pick a scan on the left.</div>;
  const depth = planeDepth(meta, plane);
  const clamped = Math.min(index, depth - 1);
  const [d, h, w] = meta.shape;
  const [sz, sy, sx] = meta.spacing;

  return (
    <div className="view">
      <div className="stage">
        <SliceCanvas
          volume={state.volume}
          plane={plane}
          index={clamped}
          window={win}
          progress={state.progress}
          error={state.error}
          onIndexChange={onIndexChange}
          onWindowChange={onWindowChange}
          label={
            <>
              <b>Patient {meta.patient}</b> · {visitLabel(meta)}
              <br />
              {PLANES.find((p) => p.id === plane)!.name} {clamped + 1} / {depth} · {planePositionMm(meta, plane, clamped).toFixed(0)} mm
            </>
          }
        />
        <div className="slider-row">
          <button className="chip" onClick={() => onIndexChange(Math.max(0, clamped - 1))}>−</button>
          <input type="range" min={0} max={depth - 1} value={clamped} onChange={(e) => onIndexChange(+e.target.value)} />
          <button className="chip" onClick={() => onIndexChange(Math.min(depth - 1, clamped + 1))}>+</button>
          <span className="slider-value">{clamped + 1} / {depth}</span>
        </div>
      </div>

      <aside className="side">
        <div className="panel">
          <h3>Plane</h3>
          <div className="chips">
            {PLANES.map((p) => (
              <button key={p.id} className={`chip ${plane === p.id ? "active" : ""}`} onClick={() => onPlaneChange(p.id)} title={p.blurb}>
                {p.name}
              </button>
            ))}
          </div>
          <p className="hint">{PLANES.find((p) => p.id === plane)!.blurb}. Scroll the mouse wheel over the image (shift = ×5) or use ↑ ↓.</p>
        </div>

        <WindowControls value={win} onChange={onWindowChange} />

        <div className="panel">
          <h3>About this scan</h3>
          <dl className="facts">
            <dt>Patient</dt><dd>{meta.patient}</dd>
            <dt>Visit</dt><dd>{visitLabel(meta)}</dd>
            <dt>Scanner</dt><dd>{meta.manufacturer} {meta.model}</dd>
            <dt>Kernel</dt><dd>{meta.kernel} ({meta.kernelStyle})</dd>
            <dt>Volume</dt><dd>{d} slices × {h} × {w} px</dd>
            <dt>Voxel size</dt><dd>{sx.toFixed(2)} × {sy.toFixed(2)} × {sz.toFixed(1)} mm</dd>
            <dt>Extent</dt><dd>{(w * sx / 10).toFixed(0)} × {(h * sy / 10).toFixed(0)} × {(d * sz / 10).toFixed(0)} cm</dd>
            <dt>HU range</dt><dd>{meta.huRange[0]} … {meta.huRange[1]}</dd>
          </dl>
        </div>
      </aside>
    </div>
  );
}
