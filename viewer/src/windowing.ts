import type { HUWindow, Slice } from "./types";

/** Standard CT display windows (centre, width in HU) — same presets as dicom_to_png.py. */
export const PRESETS: Record<string, HUWindow> = {
  Lung: { center: -600, width: 1500 },
  "Soft tissue": { center: 40, width: 400 },
  Bone: { center: 400, width: 1800 },
  Full: { center: 250, width: 2500 },
};

export const DEFAULT_WINDOW = PRESETS.Lung;

export function presetName(w: HUWindow): string | null {
  for (const [name, p] of Object.entries(PRESETS)) {
    if (p.center === w.center && p.width === w.width) return name;
  }
  return null;
}

/**
 * Map HU values to 8-bit grey and write them into an ImageData. Everything
 * below the window is black, everything above is white — exactly what
 * `to_uint8` does in dicom_to_png.py.
 */
export function windowToImageData(slice: Slice, win: HUWindow, out: ImageData): void {
  const lo = win.center - win.width / 2;
  const scale = 255 / Math.max(win.width, 1);
  const px = new Uint32Array(out.data.buffer);
  const src = slice.data;
  for (let i = 0; i < src.length; i++) {
    let v = (src[i] - lo) * scale;
    v = v < 0 ? 0 : v > 255 ? 255 : v;
    const g = v | 0;
    px[i] = 0xff000000 | (g << 16) | (g << 8) | g;
  }
}

/**
 * Composite a coloured overlay onto an already-windowed greyscale ImageData.
 *
 * Done in one pass over the pixels rather than with a second canvas and a CSS
 * blend mode: the overlay has to line up exactly with the grey underneath at
 * every zoom level, and a screen blend would wash out the lung parenchyma the
 * reader is trying to judge. Alpha is scaled by the overlay's own intensity, so
 * weak signal stays faint instead of painting a flat wash of colour.
 */
export function blendOverlay(
  out: ImageData,
  overlay: { data: Uint8Array; width: number; height: number },
  colour: [number, number, number],
  opacity: number,
): void {
  if (opacity <= 0) return;
  if (overlay.width * overlay.height !== out.width * out.height) return;

  const px = out.data;
  const [r, g, b] = colour;
  const src = overlay.data;
  for (let i = 0; i < src.length; i++) {
    const v = src[i];
    if (v === 0) continue;
    const a = (v / 255) * opacity;
    const o = i * 4;
    px[o] = px[o] + (r - px[o]) * a;
    px[o + 1] = px[o + 1] + (g - px[o + 1]) * a;
    px[o + 2] = px[o + 2] + (b - px[o + 2]) * a;
  }
}
