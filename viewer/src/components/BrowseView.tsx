import { planeDepth, planePositionMm, visitLabel } from "../data";
import type { VolumeState } from "../hooks";
import type { OverlayVolume } from "../overlay";
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
  overlays: OverlayVolume[];
  overlayOpacity: number;
  highlight: { slice: number; box: [number, number, number, number] } | null;
  series: SeriesMeta[];
  onSelect: (m: SeriesMeta) => void;
}

/** Single-series viewer: pick a round and plane, scroll slices, adjust the window. */
export function BrowseView({
  meta, state, plane, onPlaneChange, index, onIndexChange, window: win, onWindowChange,
  overlays, overlayOpacity, highlight, series, onSelect,
}: Props) {
  if (!meta) return <div className="empty">Pick a scan to view.</div>;

  const depth = planeDepth(meta, plane);
  const clamped = Math.min(index, depth - 1);
  const [d, h, w] = meta.shape;
  const [sz, sy, sx] = meta.spacing;

  return (
    <>
      <div className="center-controls">
        {series.map((s) => (
          <button
            key={s.id}
            className={`chip ${s.id === meta.id ? "active" : ""}`}
            onClick={() => onSelect(s)}
            title={`${s.kernel} · ${s.shape[0]} slices · ${s.studyDate}`}
          >
            {s.round ?? visitLabel(s)}
          </button>
        ))}
        <span style={{ flex: 1 }} />
        {PLANES.map((p) => (
          <button
            key={p.id}
            className={`chip ${plane === p.id ? "active" : ""}`}
            onClick={() => onPlaneChange(p.id)}
            title={p.blurb}
          >
            {p.name}
          </button>
        ))}
      </div>

      <SliceCanvas
        volume={state.volume}
        plane={plane}
        index={clamped}
        window={win}
        progress={state.progress}
        error={state.error}
        overlays={overlays}
        overlayOpacity={overlayOpacity}
        highlight={highlight}
        onIndexChange={onIndexChange}
        onWindowChange={onWindowChange}
        label={
          <>
            <b>Patient {meta.patient}</b> · {meta.round ?? visitLabel(meta)} · {meta.studyDate}
            <br />
            {PLANES.find((p) => p.id === plane)!.name} {clamped + 1} / {depth} ·{" "}
            {planePositionMm(meta, plane, clamped).toFixed(0)} mm
          </>
        }
      />

      <div className="slider-row">
        <button className="chip" onClick={() => onIndexChange(Math.max(0, clamped - 1))}>−</button>
        <input
          type="range" min={0} max={depth - 1} value={clamped}
          onChange={(e) => onIndexChange(+e.target.value)}
        />
        <button className="chip" onClick={() => onIndexChange(Math.min(depth - 1, clamped + 1))}>+</button>
        <span className="slider-value">{clamped + 1} / {depth}</span>
      </div>

      <details>
        <summary>Scan details and display window</summary>
        <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 12, marginTop: 10 }}>
          <div className="panel">
            <h3>About this scan</h3>
            <dl className="facts">
              <dt>Round</dt><dd>{meta.round ?? "—"} · {meta.studyDate}</dd>
              <dt>Scanner</dt><dd>{meta.manufacturer} {meta.model}</dd>
              <dt>Kernel</dt><dd>{meta.kernel} ({meta.kernelStyle})</dd>
              <dt>Volume</dt><dd>{d} × {h} × {w} px</dd>
              <dt>Voxel</dt><dd>{sx.toFixed(2)} × {sy.toFixed(2)} × {sz.toFixed(1)} mm</dd>
              <dt>Extent</dt>
              <dd>{(w * sx / 10).toFixed(0)} × {(h * sy / 10).toFixed(0)} × {(d * sz / 10).toFixed(0)} cm</dd>
              {meta.lungVolumeMl != null && <><dt>Lung volume</dt><dd>{meta.lungVolumeMl.toFixed(0)} mL</dd></>}
              <dt>Slice order</dt>
              <dd>{meta.flippedForSybil ? "reversed to feet→head" : "as stored"}</dd>
            </dl>
          </div>
          <WindowControls value={win} onChange={onWindowChange} />
        </div>
      </details>
    </>
  );
}
