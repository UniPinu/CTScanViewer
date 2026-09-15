import { useRef, useState } from "react";
import { kernelLabel, planeDepth, visitLabel } from "../data";
import { useVolume } from "../hooks";
import type { HUWindow, Plane, SeriesMeta } from "../types";
import { SeriesSelect } from "./SeriesList";
import { SliceCanvas } from "./SliceCanvas";
import { WindowControls } from "./WindowControls";

interface Props {
  series: SeriesMeta[];
  left: SeriesMeta;
  right: SeriesMeta;
  onLeftChange: (m: SeriesMeta) => void;
  onRightChange: (m: SeriesMeta) => void;
  plane: Plane;
  onPlaneChange: (p: Plane) => void;
  window: HUWindow;
  onWindowChange: (w: HUWindow) => void;
}

/**
 * Before / after comparison of two series. The slider is a fraction of the
 * way through each volume, so scans with different slice counts stay roughly
 * aligned. Side-by-side or swipe (one image over the other with a divider).
 */
export function CompareView({ series, left, right, onLeftChange, onRightChange, plane, onPlaneChange, window: win, onWindowChange }: Props) {
  const leftState = useVolume(left);
  const rightState = useVolume(right);
  const [fraction, setFraction] = useState(0.5);
  const [mode, setMode] = useState<"side" | "swipe">("side");
  const [split, setSplit] = useState(0.5);
  const swipeBox = useRef<HTMLDivElement>(null);

  const leftDepth = planeDepth(left, plane);
  const rightDepth = planeDepth(right, plane);
  const leftIndex = Math.round(fraction * (leftDepth - 1));
  const rightIndex = Math.round(fraction * (rightDepth - 1));
  const maxDepth = Math.max(leftDepth, rightDepth);

  const setFromIndex = (i: number, depth: number) => setFraction(depth > 1 ? i / (depth - 1) : 0);

  const onSplitDrag = (e: React.PointerEvent<HTMLDivElement>) => {
    if (e.buttons !== 1 || !swipeBox.current) return;
    const r = swipeBox.current.getBoundingClientRect();
    setSplit(Math.min(1, Math.max(0, (e.clientX - r.left) / r.width)));
  };

  const label = (m: SeriesMeta, tag: string, idx: number, depth: number) => (
    <>
      <b>{tag}</b> · Patient {m.patient} · {visitLabel(m)}
      <br />
      {kernelLabel(m)} · slice {idx + 1} / {depth}
    </>
  );

  const sameStudy = left.patient === right.patient && left.studyDate === right.studyDate;
  const samePatient = left.patient === right.patient;

  return (
    <div className="view">
      <div className="stage">
        <div className="compare-header">
          <SeriesSelect series={series} value={left.id} onChange={onLeftChange} />
          <span className="arrow">→</span>
          <SeriesSelect series={series} value={right.id} onChange={onRightChange} />
        </div>

        {mode === "side" ? (
          <div className="compare-grid">
            <SliceCanvas volume={leftState.volume} plane={plane} index={leftIndex} window={win}
              progress={leftState.progress} error={leftState.error}
              onIndexChange={(i) => setFromIndex(i, leftDepth)} onWindowChange={onWindowChange}
              label={label(left, "Before", leftIndex, leftDepth)} />
            <SliceCanvas volume={rightState.volume} plane={plane} index={rightIndex} window={win}
              progress={rightState.progress} error={rightState.error}
              onIndexChange={(i) => setFromIndex(i, rightDepth)} onWindowChange={onWindowChange}
              label={label(right, "After", rightIndex, rightDepth)} />
          </div>
        ) : (
          <div className="swipe-box" ref={swipeBox} onPointerMove={onSplitDrag} onPointerDown={onSplitDrag}>
            <SliceCanvas volume={leftState.volume} plane={plane} index={leftIndex} window={win}
              progress={leftState.progress} error={leftState.error} className="swipe-layer"
              label={label(left, "Before", leftIndex, leftDepth)} />
            <div className="swipe-layer" style={{ clipPath: `inset(0 0 0 ${split * 100}%)` }}>
              <SliceCanvas volume={rightState.volume} plane={plane} index={rightIndex} window={win}
                progress={rightState.progress} error={rightState.error} className="swipe-layer" />
            </div>
            <div className="swipe-divider" style={{ left: `${split * 100}%` }} />
            <div className="slice-label" style={{ left: `calc(${split * 100}% + 12px)` }}>
              {label(right, "After", rightIndex, rightDepth)}
            </div>
          </div>
        )}

        <div className="slider-row">
          <input type="range" min={0} max={maxDepth - 1} value={Math.round(fraction * (maxDepth - 1))}
            onChange={(e) => setFromIndex(+e.target.value, maxDepth)} />
          <span className="slider-value">{Math.round(fraction * 100)}% through the chest</span>
        </div>
      </div>

      <aside className="side">
        <div className="panel">
          <h3>Layout</h3>
          <div className="chips">
            <button className={`chip ${mode === "side" ? "active" : ""}`} onClick={() => setMode("side")}>Side by side</button>
            <button className={`chip ${mode === "swipe" ? "active" : ""}`} onClick={() => setMode("swipe")}>Swipe</button>
          </div>
          <div className="chips" style={{ marginTop: 8 }}>
            {(["axial", "coronal", "sagittal"] as Plane[]).map((p) => (
              <button key={p} className={`chip ${plane === p ? "active" : ""}`} onClick={() => onPlaneChange(p)}>
                {p[0].toUpperCase() + p.slice(1)}
              </button>
            ))}
          </div>
          <p className="hint">
            {sameStudy
              ? "Same visit, two renderings: identical anatomy, different sharpening filter."
              : samePatient
                ? "Same person, different visits. Slices are matched by position through the scan, not registered, so expect small shifts."
                : "Two different people."}
          </p>
        </div>
        <WindowControls value={win} onChange={onWindowChange} />
      </aside>
    </div>
  );
}
