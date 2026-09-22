import type { OverlayMeta } from "./api";
import type { Plane } from "./types";
import { clamp } from "./data";

/**
 * Overlay volumes: attention and change, loaded and sampled like the CT itself.
 *
 * `ml/saliency.py` writes these as `uint8` volumes rather than pre-rendered
 * images, which is what lets the opacity slider and the coronal/sagittal planes
 * work without asking the server to re-render anything.
 */

export interface OverlayVolume {
  meta: OverlayMeta;
  /** uint8, [slice][row][col], same axis order as the CT volume. */
  data: Uint8Array;
  kind: "attention" | "change";
}

const DATA_URL = "data/";

const cache = new Map<string, Promise<OverlayVolume>>();

export function clearOverlayCache(): void {
  cache.clear();
}

/** Fetch an overlay volume described by a results-object overlay block. */
export function loadOverlay(dir: string, kind: "attention" | "change"): Promise<OverlayVolume> {
  const key = `${dir}:${kind}`;
  let pending = cache.get(key);
  if (!pending) {
    pending = fetchOverlay(dir, kind);
    cache.set(key, pending);
    pending.catch(() => cache.delete(key));
  }
  return pending;
}

async function fetchOverlay(dir: string, kind: "attention" | "change"): Promise<OverlayVolume> {
  const metaRes = await fetch(`${DATA_URL}${dir}/meta.json`);
  if (!metaRes.ok) throw new Error(`${dir}/meta.json: HTTP ${metaRes.status}`);
  const meta = (await metaRes.json()) as OverlayMeta;

  const res = await fetch(`${DATA_URL}${dir}/${meta.file}`);
  if (!res.ok) throw new Error(`${dir}/${meta.file}: HTTP ${res.status}`);
  const buffer = await res.arrayBuffer();

  const [d, h, w] = meta.shape;
  if (buffer.byteLength !== d * h * w) {
    throw new Error(`${meta.file}: expected ${d * h * w} bytes, got ${buffer.byteLength}`);
  }
  return { meta, data: new Uint8Array(buffer), kind };
}

/**
 * Pull the 2-D plane of an overlay matching a CT slice.
 *
 * Mirrors `extractSlice` in data.ts, including the head-up reorientation for
 * the non-axial planes, so the overlay lands on the anatomy it describes.
 */
export function extractOverlaySlice(
  overlay: OverlayVolume,
  plane: Plane,
  index: number,
): { data: Uint8Array; width: number; height: number } {
  const [d, h, w] = overlay.meta.shape;
  const src = overlay.data;
  const stride = h * w;

  if (plane === "axial") {
    const z = clamp(index, 0, d - 1);
    return { data: src.subarray(z * stride, (z + 1) * stride), width: w, height: h };
  }

  if (plane === "coronal") {
    const y = clamp(index, 0, h - 1);
    const out = new Uint8Array(w * d);
    for (let z = 0; z < d; z++) {
      const row = (d - 1 - z) * w;
      const base = z * stride + y * w;
      for (let x = 0; x < w; x++) out[row + x] = src[base + x];
    }
    return { data: out, width: w, height: d };
  }

  const x = clamp(index, 0, w - 1);
  const out = new Uint8Array(h * d);
  for (let z = 0; z < d; z++) {
    const row = (d - 1 - z) * h;
    const base = z * stride + x;
    for (let y = 0; y < h; y++) out[row + y] = src[base + y * w];
  }
  return { data: out, width: h, height: d };
}

/**
 * Overlay colours.
 *
 * §8.1 asks for a single accent, and change — the overlay that carries the
 * project's novel signal — gets it. Attention uses the calm green token rather
 * than a second accent, so the two are distinguishable on the rare occasion
 * both are enabled without introducing a new colour to the palette.
 */
export const OVERLAY_COLOURS: Record<"attention" | "change", [number, number, number]> = {
  change: [193, 95, 60], // --accent
  attention: [79, 122, 91], // --ok
};

/**
 * Whether an overlay's shape matches the volume it is being drawn over.
 *
 * Change detection runs on a resampled grid, so a mismatch is possible if the
 * results and the exported volume fall out of step. Drawing a mismatched
 * overlay would put colour on the wrong anatomy, which is worse than drawing
 * none — so the caller checks this and shows a warning instead.
 */
export function overlayMatches(overlay: OverlayVolume, shape: [number, number, number]): boolean {
  return overlay.meta.shape.every((n, i) => n === shape[i]);
}

/**
 * Whether this overlay belongs to the round currently on screen.
 *
 * Attention is computed on the latest round and a change map lives in the
 * geometry of its pair's earlier round, so neither is valid over an arbitrary
 * scan. Shapes alone are not a safe proxy — two rounds of the same patient can
 * have identical dimensions and completely different anatomy on a given slice.
 */
export function overlayBelongsTo(
  overlay: OverlayVolume,
  round: string | null,
  patientId?: string | null,
): boolean {
  if (patientId && overlay.meta.patient_id && overlay.meta.patient_id !== patientId) return false;
  if (!overlay.meta.round) return true; // unlabelled: fall back to the shape check
  return overlay.meta.round === round;
}
