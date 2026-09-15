import { useEffect, useMemo, useRef, type ReactNode } from "react";
import { extractSlice, planeDepth } from "../data";
import { windowToImageData } from "../windowing";
import type { HUWindow, Plane, Volume } from "../types";

interface Props {
  volume: Volume | null;
  plane: Plane;
  index: number;
  window: HUWindow;
  /** Download progress 0..1 while `volume` is null. */
  progress?: number;
  error?: string | null;
  /** Mouse wheel scrolls through slices. */
  onIndexChange?: (index: number) => void;
  /** Click-drag adjusts the window: horizontal = width, vertical = centre. */
  onWindowChange?: (w: HUWindow) => void;
  /** Overlay drawn in the top-left corner. */
  label?: ReactNode;
  /** Extra class for the outer box. */
  className?: string;
}

/**
 * Draws one windowed slice of a volume onto a canvas, letterboxed inside its
 * container with the correct physical aspect ratio.
 */
export function SliceCanvas({
  volume, plane, index, window: win, progress = 0, error, onIndexChange, onWindowChange, label, className,
}: Props) {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const offscreen = useRef<HTMLCanvasElement | null>(null);
  const imageData = useRef<ImageData | null>(null);
  const drag = useRef<{ x: number; y: number; win: HUWindow } | null>(null);

  const slice = useMemo(() => (volume ? extractSlice(volume, plane, index) : null), [volume, plane, index]);

  // Paint: HU -> grey via window -> offscreen (native pixels) -> display canvas (aspect-corrected).
  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas || !slice) return;
    const { width, height, pixelMm } = slice;
    if (!offscreen.current) offscreen.current = document.createElement("canvas");
    const off = offscreen.current;
    if (off.width !== width || off.height !== height) {
      off.width = width;
      off.height = height;
      imageData.current = null;
    }
    if (!imageData.current) imageData.current = new ImageData(width, height);
    windowToImageData(slice, win, imageData.current);
    off.getContext("2d")!.putImageData(imageData.current, 0, 0);

    const dispH = Math.round((height * pixelMm[1]) / pixelMm[0]);
    if (canvas.width !== width || canvas.height !== dispH) {
      canvas.width = width;
      canvas.height = dispH;
    }
    const ctx = canvas.getContext("2d")!;
    ctx.imageSmoothingEnabled = true;
    ctx.imageSmoothingQuality = "high";
    ctx.drawImage(off, 0, 0, width, dispH);
  }, [slice, win]);

  // Wheel must be a non-passive listener so we can stop the page scrolling.
  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas || !volume || !onIndexChange) return;
    const depth = planeDepth(volume.meta, plane);
    const onWheel = (e: WheelEvent) => {
      e.preventDefault();
      const step = (e.deltaY > 0 ? 1 : -1) * (e.shiftKey ? 5 : 1);
      onIndexChange(Math.min(depth - 1, Math.max(0, index + step)));
    };
    canvas.addEventListener("wheel", onWheel, { passive: false });
    return () => canvas.removeEventListener("wheel", onWheel);
  }, [volume, plane, index, onIndexChange]);

  const onPointerDown = (e: React.PointerEvent<HTMLCanvasElement>) => {
    if (!onWindowChange || e.button !== 0) return;
    drag.current = { x: e.clientX, y: e.clientY, win };
    e.currentTarget.setPointerCapture(e.pointerId);
  };
  const onPointerMove = (e: React.PointerEvent<HTMLCanvasElement>) => {
    if (!drag.current || !onWindowChange) return;
    const dx = e.clientX - drag.current.x;
    const dy = e.clientY - drag.current.y;
    onWindowChange({
      width: Math.max(1, Math.round(drag.current.win.width + dx * 4)),
      center: Math.round(drag.current.win.center - dy * 2),
    });
  };
  const onPointerUp = (e: React.PointerEvent<HTMLCanvasElement>) => {
    drag.current = null;
    e.currentTarget.releasePointerCapture(e.pointerId);
  };

  return (
    <div className={`slice-box ${className ?? ""}`}>
      <canvas
        ref={canvasRef}
        className="slice-canvas"
        onPointerDown={onPointerDown}
        onPointerMove={onPointerMove}
        onPointerUp={onPointerUp}
        onPointerCancel={onPointerUp}
        style={{ cursor: onWindowChange ? "crosshair" : "default" }}
      />
      {label && <div className="slice-label">{label}</div>}
      {!volume && !error && (
        <div className="slice-overlay">
          <div className="progress"><div style={{ width: `${Math.round(progress * 100)}%` }} /></div>
          <span>Loading… {Math.round(progress * 100)}%</span>
        </div>
      )}
      {error && <div className="slice-overlay error">{error}</div>}
    </div>
  );
}
