import type { Plane, SeriesMeta, Slice, Volume } from "./types";

const DATA_URL = "data/";

export async function loadIndex(): Promise<SeriesMeta[]> {
  const res = await fetch(`${DATA_URL}index.json`);
  if (!res.ok) throw new Error(`index.json: HTTP ${res.status} — run export_volumes.py first`);
  return res.json();
}

const cache = new Map<string, Promise<Volume>>();

/** Drop cached volumes, e.g. after the data on disk has been replaced. */
export function clearVolumeCache(): void {
  cache.clear();
}

/** Fetch a series' raw HU volume (cached), reporting download progress 0..1. */
export function loadVolume(meta: SeriesMeta, onProgress?: (p: number) => void): Promise<Volume> {
  let p = cache.get(meta.id);
  if (!p) {
    p = fetchVolume(meta, onProgress);
    cache.set(meta.id, p);
    p.catch(() => cache.delete(meta.id));
  } else {
    onProgress?.(1);
  }
  return p;
}

async function fetchVolume(meta: SeriesMeta, onProgress?: (p: number) => void): Promise<Volume> {
  const res = await fetch(`${DATA_URL}${meta.file}`);
  if (!res.ok || !res.body) throw new Error(`${meta.file}: HTTP ${res.status}`);

  const [d, h, w] = meta.shape;
  const total = d * h * w * 2;
  const bytes = new Uint8Array(total);
  const reader = res.body.getReader();
  let offset = 0;
  for (;;) {
    const { done, value } = await reader.read();
    if (done) break;
    bytes.set(value, offset);
    offset += value.length;
    onProgress?.(offset / total);
  }
  if (offset !== total) throw new Error(`${meta.file}: expected ${total} bytes, got ${offset}`);
  // File is little-endian int16; every platform we target is little-endian too.
  return { meta, data: new Int16Array(bytes.buffer) };
}

/** Number of slices available along a plane. */
export function planeDepth(meta: SeriesMeta, plane: Plane): number {
  const [d, h, w] = meta.shape;
  return plane === "axial" ? d : plane === "coronal" ? h : w;
}

/** Physical position (mm) of slice `index` along a plane, measured from the first slice. */
export function planePositionMm(meta: SeriesMeta, plane: Plane, index: number): number {
  const [sz, sy, sx] = meta.spacing;
  return index * (plane === "axial" ? sz : plane === "coronal" ? sy : sx);
}

/**
 * Pull a 2-D slice out of the volume. Axial is a straight copy; coronal and
 * sagittal are re-slices with the head at the top of the image, following the
 * usual radiology conventions (coronal: patient's right on image left;
 * sagittal: anterior on image left).
 */
export function extractSlice(vol: Volume, plane: Plane, index: number): Slice {
  const [d, h, w] = vol.meta.shape;
  const [sz, sy, sx] = vol.meta.spacing;
  const src = vol.data;
  const stride = h * w;

  if (plane === "axial") {
    const z = clamp(index, 0, d - 1);
    return { data: src.subarray(z * stride, (z + 1) * stride), width: w, height: h, pixelMm: [sx, sy] };
  }

  if (plane === "coronal") {
    const y = clamp(index, 0, h - 1);
    const out = new Int16Array(w * d);
    for (let z = 0; z < d; z++) {
      const row = (d - 1 - z) * w; // head up
      const base = z * stride + y * w;
      for (let x = 0; x < w; x++) out[row + x] = src[base + x];
    }
    return { data: out, width: w, height: d, pixelMm: [sx, sz] };
  }

  const x = clamp(index, 0, w - 1);
  const out = new Int16Array(h * d);
  for (let z = 0; z < d; z++) {
    const row = (d - 1 - z) * h;
    const base = z * stride + x;
    for (let y = 0; y < h; y++) out[row + y] = src[base + y * w];
  }
  return { data: out, width: h, height: d, pixelMm: [sy, sz] };
}

export function clamp(v: number, lo: number, hi: number): number {
  return v < lo ? lo : v > hi ? hi : v;
}

/** Human-readable one-liners for the UI. */
export function visitLabel(m: SeriesMeta): string {
  return m.visit === null ? m.studyDate : `Year ${m.visit} · ${m.studyDate}`;
}

export function kernelLabel(m: SeriesMeta): string {
  const style = m.kernelStyle === "other" ? "" : ` (${m.kernelStyle})`;
  return `${m.kernel || "?"}${style}`;
}

export function shortLabel(m: SeriesMeta): string {
  return `${m.patient} · ${visitLabel(m)} · ${kernelLabel(m)}`;
}
