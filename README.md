# CT

Tools for pulling chest CT scans from the [Imaging Data Commons](https://portal.imaging.datacommons.cancer.gov/)
(NLST collection), converting them for machine learning, and looking at them
in a browser. No scan data is stored in this repo — it is fetched on demand,
one patient at a time.

New to CT / DICOM? Read [context.md](context.md) first: it explains what the
pictures show and what every abbreviation means.

## Quick start

```bash
# Python side (downloading + conversion)
pip install -r requirements.txt

# Viewer
cd viewer
npm install
npm run dev            # http://localhost:5173
```

Open the app and press **Fetch a random patient**. That deletes whatever
scans were on disk, downloads every usable CT series for one randomly chosen
patient (all their visits and reconstructions, typically 100–600 MB), and
prepares them for the viewer. Press it again for a different patient.

The same thing from the command line:

```bash
python fetch_patient.py                  # random patient
python fetch_patient.py --patient 100012 # a specific one
python fetch_patient.py --dry-run        # see what would be fetched
```

## What's here

| File | Purpose |
|---|---|
| `manifest_*.s5cmd` | The IDC manifest: a list of ~14 000 series (one line each) to choose from. Small enough to commit. |
| `fetch_patient.py` | Select one patient → clear old data → download → export for the viewer. Backs the button in the app. |
| `download_sample.py` | Download the first / a random *N* series from the manifest (the original small-sample script). |
| `dicom_to_png.py` | Convert downloaded DICOM series into per-slice PNGs (windowed 8-bit or lossless 16-bit). |
| `export_volumes.py` | Convert DICOM series into raw 16-bit HU volumes + JSON metadata for the viewer. |
| `viewer/` | React + TypeScript app: slice browser, before/after comparison, 3-D rendering. See [viewer/README.md](viewer/README.md). |
| `context.md` | Plain-language explanation of the data. |

## Where data lives (all git-ignored)

```
data/patient/          DICOM files for the current patient   (fetch_patient.py)
data/sample/           DICOM files from download_sample.py
data/sample_png/       PNGs from dicom_to_png.py
viewer/public/data/    volumes the viewer reads               (export_volumes.py / fetch_patient.py)
```

`fetch_patient.py` clears `data/patient/` and `viewer/public/data/` before
each download; it leaves `data/sample*` alone.

## Making PNGs for training

```bash
python dicom_to_png.py --in data/patient            # lung window, 8-bit
python dicom_to_png.py --in data/patient --bits 16  # full HU range
```
