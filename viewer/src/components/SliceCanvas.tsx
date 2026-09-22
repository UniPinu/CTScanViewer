import { useEffect, useMemo, useRef, type ReactNode } from "react";
import { extractSlice, planeDepth } from "../data";
import { extractOverlaySlice, OVERLAY_COLOURS, overlayMatches, type OverlayVolume } from "../overlay";
import { blendOverlay, windowToImageData } from "../windowing";
import type { HUWindow, Plane, Volume } from "../types";

interface Props {
  volume: Volume | null;
  plane: Plane;
  index: number;
  window: HUWindow;
  /** Download progress 0..1 while `volume` is null. */
  progress?: number;
  error?: string | null;
  /** Heatmaps drawn over the slice, in order. */
  overlays?: OverlayVolume[];
  overlayOpacity?: number;
  /** Box drawn on the current slice, in [y0, x0, y1, x1] voxels of this volume. */
  highlight?: { slice: number; box: [number, number, number, number] } | null;
  /** Mouse wheel scrolls through slices. */
  onIndexChange?: (index: number) => void;
  /** Click-drag adjusts the window: horizontal = width, vertical = centre. */
  onWindowChange?: (w: HUWindow) => void;
  /** Overlay drawn in the top-left corner. */
  label?: ReactNode;
  className?: string;
}

/**
 * Draws one windowed slice, letterboxed inside its container at the correct
 * physical aspect ratio, with any overlays composited on top.
 */
export function SliceCanvas({
  volume, plane, index, window: win, progress = 0, error, overlays = [], overlayOpacity = 0.6,
  highlight, onIndexChange, onWindowChange, label, className,
}: Props) {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const offscreen = useRef<HTMLCanvasElement | null>(null);
  const imageData = useRef<ImageData | null>(null);
  const drag = useRef<{ x: number; y: number; win: HUWindow } | null>(null);

  const slice = useMemo(() => (volume ? extractSlice(volume, plane, index) : null), [volume, plane, index]);

  // HU -> grey -> overlay blend -> offscreen (native px) -> display (aspect-corrected).
  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas || !slice || !volume) return;
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

    for (const overlay of overlays) {
      // A mismatched overlay would paint colour onto the wrong anatomy.
      if (!overlayMatches(overlay, volume.meta.shape)) continue;
      blendOverlay(
        imageData.current,
        extractOverlaySlice(overlay, plane, index),
        OVERLAY_COLOURS[overlay.kind],
        overlayOpacity,
      );
    }
    off.getContext("2d")!.putImageData(imageData.current, 0, 0);

    const displayHeight = Math.round((height * pixelMm[1]) / pixelMm[0]);
    if (canvas.width !== width || canvas.height !== displayHeight) {
      canvas.width = width;
      canvas.height = displayHeight;
    }
    const ctx = canvas.getContext("2d")!;
    ctx.imageSmoothingEnabled = true;
    ctx.imageSmoothingQuality = "high";
    ctx.drawImage(off, 0, 0, width, displayHeight);

    // The selected change region, drawn as a hairline box on its own slice only.
    if (highlight && plane === "axial" && highlight.slice === index) {
      const [y0, x0, y1, x1] = highlight.box;
      const scaleY = displayHeight / height;
      ctx.strokeStyle = "#e9c7bc";
      ctx.lineWidth = Math.max(1, width / 400);
      ctx.strokeRect(x0, y0 * scaleY, x1 - x0, (y1 - y0) * scaleY);
    }
  }, [slice, win, overlays, overlayOpacity, highlight, plane, index, volume]);

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
    onWindowChange({
      width: Math.max(1, Math.round(drag.current.win.width + (e.clientX - drag.current.x) * 4)),
      center: Math.round(drag.current.win.center - (e.clientY - drag.current.y) * 2),
    });
  };
  const onPointerUp = (e: React.PointerEvent<HTMLCanvasElement>) => {
    drag.current = null;
    e.currentTarget.releasePointerCapture(e.pointerId);
  };

  return (
    <div className={`stage ${className ?? ""}`}>
      <canvas
        ref={canvasRef}
        onPointerDown={onPointerDown}
        onPointerMove={onPointerMove}
        onPointerUp={onPointerUp}
        onPointerCancel={onPointerUp}
        style={{ cursor: onWindowChange ? "crosshair" : "default" }}
      />
      {label && <div className="stage-label">{label}</div>}
      {!volume && !error && (
        <div className="stage-overlay">
          <div className="progress" style={{ width: 200 }}>
            <div style={{ width: `${Math.round(progress * 100)}%` }} />
          </div>
          <span>Loading volume… {Math.round(progress * 100)}%</span>
        </div>
      )}
      {error && <div className="stage-overlay error">{error}</div>}
    </div>
  );
}
