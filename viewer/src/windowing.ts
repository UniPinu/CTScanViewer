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
