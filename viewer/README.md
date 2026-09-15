# CT Viewer

A browser app for looking at the downloaded NLST chest CT scans. Three views:

- **Browse** — scroll through one scan slice by slice in any of the three
  planes (axial / coronal / sagittal), with the usual radiology display
  windows. Drag on the image to fine-tune the window.
- **Before / after** — two scans side by side or with a swipe divider, kept
  at the same relative position. Defaults to the same person's year-0 and
  year-1 visits.
- **3-D** — GPU ray-marched rendering of the whole volume: translucent
  layers (lungs / soft tissue / bone), a solid surface at a chosen density
  (skin or skeleton), or a see-through X-ray projection. "Cut at slice"
  removes everything above an axial slice and draws that slice in place, so
  you can see where the 2-D picture sits inside the body.

See `../context.md` for what the pictures show and what the terms mean.

## Run it

```bash
pip install -r ../requirements.txt   # once, from the project root
npm install
npm run dev                          # http://localhost:5173
```

Then press **Fetch a random patient** in the app. It wipes `../data/patient`
and `public/data`, downloads every usable CT series for one random patient
from the Imaging Data Commons, and exports it for the viewer — so nothing
under `public/data` ever needs to be committed. Press it again for another
patient.

The button talks to a tiny API that the Vite dev server provides
(`vite.config.ts` → `fetchPatientApi`), which simply runs
`../fetch_patient.py --json` and streams its progress lines back to the
browser. It therefore only exists under `npm run dev`; a static `dist/`
build has no button backend. Set `PYTHON=/path/to/python` if `python` isn't
the interpreter with `idc-index` installed.

If you'd rather manage the data yourself, `python export_volumes.py --in <dicom dir>`
exports any folder of downloaded series into `public/data`.

## How the data flows

```
manifest.s5cmd  --fetch_patient.py-->  ../data/patient/**/*.dcm
                                       public/data/<id>/hu.i16 + meta.json
                                       public/data/index.json
```

`export_volumes.py` reuses the slice sorting and Hounsfield conversion from
`dicom_to_png.py`, but writes raw 16-bit HU instead of windowed 8-bit PNGs.
The app applies the window itself (`src/windowing.ts` mirrors `to_uint8` in
the Python), which is what makes the presets and drag-to-window interactive.
For the 3-D view the volume is downsampled to ≤256 px in-plane, packed to
8 bits (10 HU per level) and uploaded as a WebGL2 3-D texture; the shader in
`src/volume/shaders.ts` marches rays through it.

## Layout

```
src/
  App.tsx                 tabs, selected scan, shared window/slice state
  data.ts                 index/volume loading, coronal & sagittal re-slicing
  windowing.ts            HU -> grey presets and conversion
  hooks.ts                useVolume (cached loading with progress)
  components/
    FetchPatient.tsx      the "Fetch a random patient" button + progress panel
    SliceCanvas.tsx       one windowed slice on a canvas; wheel + drag interaction
    BrowseView.tsx        single-scan viewer
    CompareView.tsx       before/after (side-by-side and swipe)
    VolumeView.tsx        three.js scene, 3-D texture, cut plane
    WindowControls.tsx    presets + centre/width sliders
    SeriesList.tsx        sidebar and dropdown
  volume/shaders.ts       ray-marching vertex/fragment shaders
```
