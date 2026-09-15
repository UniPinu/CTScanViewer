import { useEffect, useState } from "react";
import { loadVolume } from "./data";
import type { SeriesMeta, Volume } from "./types";

export interface VolumeState {
  volume: Volume | null;
  progress: number;
  error: string | null;
}

/** Load (and cache) the volume for a series, tracking progress for the UI. */
export function useVolume(meta: SeriesMeta | null): VolumeState {
  const [state, setState] = useState<VolumeState>({ volume: null, progress: 0, error: null });

  useEffect(() => {
    if (!meta) {
      setState({ volume: null, progress: 0, error: null });
      return;
    }
    let cancelled = false;
    setState({ volume: null, progress: 0, error: null });
    loadVolume(meta, (p) => {
      if (!cancelled) setState((s) => ({ ...s, progress: p }));
    })
      .then((volume) => {
        if (!cancelled) setState({ volume, progress: 1, error: null });
      })
      .catch((e: Error) => {
        if (!cancelled) setState({ volume: null, progress: 0, error: e.message });
      });
    return () => {
      cancelled = true;
    };
  }, [meta]);

  return state;
}
