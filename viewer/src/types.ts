/** One exported series, as written by export_volumes.py into data/index.json. */
export interface SeriesMeta {
  id: string;
  patient: string;
  studyDate: string;
  /** NLST screening year: 0, 1 or 2. */
  visit: number | null;
  /** "T0" | "T1" | "T2" — the round label used throughout the pipeline. */
  round: string | null;
  kernel: string;
  kernelStyle: "sharp" | "smooth" | "other";
  manufacturer: string;
  model: string;
  description: string;
  /** [slices, rows, cols] */
  shape: [number, number, number];
  /** mm per voxel, same order as shape: [z, y, x] */
  spacing: [number, number, number];
  huRange: [number, number];
  file: string;
  seriesUid: string;
  /** Which split this patient's scans belong to, so the UI can say so. */
  split: "train" | "val" | "test" | null;
  /** Whether the stored slice order had to be reversed (see preprocess/volume.py). */
  flippedForSybil: boolean;
  lungVolumeMl?: number;
  maskFile?: string;
}

/** A loaded series: metadata plus the raw Hounsfield-unit voxels, [slice][row][col]. */
export interface Volume {
  meta: SeriesMeta;
  data: Int16Array;
}

export type Plane = "axial" | "coronal" | "sagittal";

/** A display window in Hounsfield units. */
export interface HUWindow {
  center: number;
  width: number;
}

/** A 2-D slice pulled out of a volume, ready to be windowed and drawn. */
export interface Slice {
  data: Int16Array;
  width: number;
  height: number;
  /** Physical size of one pixel, in mm: [x, y]. Used to un-squash non-axial planes. */
  pixelMm: [number, number];
}
