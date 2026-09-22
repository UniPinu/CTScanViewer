# CONTEXT.md — Lung Tissue Change & Cancer-Risk Pipeline (MVP)

> This document is the engineering brief for turning the current CT-Scanner Viewer
> (see [README.md](README.md), [context.md](context.md)) into a full machine-learning
> pipeline that predicts lung-cancer risk, detects longitudinal tissue change, and
> presents it in a clean, professional viewer.
>
> Audience: an experienced dev/ML team. It states _what_ to build, _why_ each choice
> was made, and _how_ the pieces connect. It is deliberately opinionated so that the
> team can start without re-litigating the architecture.

---

## 0. TL;DR

We are building a screening-assistant pipeline on top of the **National Lung Screening
Trial (NLST)** data. It does three things for a chosen patient:

1. **Risk** — a calibrated probability that the patient develops lung cancer within 1–6
   years, from a single low-dose CT (LDCT).
2. **Change** — a map of how lung tissue changed between a patient's screening rounds
   (NLST scanned each participant at baseline + up to two annual follow-ups), highlighting
   growing or new regions.
3. **Explanation** — attention/saliency overlays showing _where_ the model looked, plus a
   short plain-language summary.

Two design decisions make this tractable:

- **We do not download the dataset and we do not train from scratch.** The full NLST
  imaging is on the order of ~10+ TB. We treat the dataset as a _queryable index_ and
  stream only the series we need, on demand, caching a small compact form. And we stand on
  **Sybil**, the open-source model already trained on NLST, rather than reinventing a
  weaker one.
- **The train/validation/test split is deterministic and patient-level**, derived by
  hashing the patient ID. This is the whole answer to "randomly hold out some data to test
  on": it needs no full download, it is reproducible, and it guarantees no patient's scans
  leak across splits.

The MVP is a FastAPI service exposing risk/change/explanation endpoints, consumed by the
existing React viewer, restyled into an Anthropic/Claude-like clean, minimal UI.

---

## 1. The data you're working with

New team members should read [context.md](context.md) for the imaging basics (what HU is,
windowing, DICOM). This section is about the _dataset_, not the pixels.

### 1.1 NLST in one paragraph

NLST was a randomized trial (enrolled ~53,000 high-risk current/former heavy smokers,
aged 55–74, 2002–2004) that showed a ~20% relative reduction in lung-cancer mortality from
screening with low-dose CT versus chest X-ray. The public imaging release contains LDCT
scans for **~26,254 participants**, with a **baseline scan (T0) plus up to two annual
follow-ups (T1, T2)** — roughly **200,000 CT series** in total. About **~2,050
participants were diagnosed with lung cancer** during the trial. That longitudinal,
mostly-negative structure drives the whole design.

### 1.2 Where the data actually lives (and how to get it)

| Source                               | What it has                                                                                                                                                       | Access                                                                     |
| ------------------------------------ | ----------------------------------------------------------------------------------------------------------------------------------------------------------------- | -------------------------------------------------------------------------- |
| **IDC (Imaging Data Commons)**       | Harmonized DICOM for all NLST CT; organ/tumor segmentations; research annotations (Sybil, NLSTseg, TotalSegmentator). Public cloud buckets (AWS S3 + GCS mirror). | **Free, no auth.** Best programmatic path.                                 |
| **TCIA**                             | Same imaging + a _limited_ subset of clinical variables; pathology slides (SVS).                                                                                  | Free.                                                                      |
| **CDAS (Cancer Data Access System)** | The **full clinical/outcome tables**: cancer confirmation status, days-from-randomization-to-diagnosis, study-year of diagnosis, stage, histology, mortality.     | **Free but requires a formal application** (turnaround measured in weeks). |

**Implication for labels (read this carefully):** the public IDC/TCIA release ships only a
_limited_ set of clinical variables. The authoritative per-patient cancer label lives in
the **CDAS Participant dataset**. So the project has two tracks that run in parallel:

- **Prototype track (start day 1, no waiting):** use publicly available annotated subsets
  to build and test the entire pipeline end-to-end:
     - **Sybil annotations** — bounding boxes of suspicious lesions on the LDCTs of
       participants who developed cancer within a year (published as JSON; also mirrored in
       IDC as `NLSTSybil`, ~601 patients).
     - **NLSTseg** — pixel-level tumor masks (~581 patients).
     - **TotalSegmentator-CT-Segmentations** — organ segmentations + radiomics for ~26,000
       NLST patients (useful for lung masking / QC).
- **Full-label track (apply immediately, in parallel):** submit a CDAS application for the
  Participant + Lung Cancer tables. When it lands, join it to the imaging on `PatientID`
  and you have the real training labels for the risk head.

### 1.3 The one number that dictates ML choices: class imbalance

At the patient level, positives are ~2,050 / ~26,254 ≈ **8%**; at the scan/slice level the
positive fraction is far lower. This must be designed for from the start:

- Split **stratified by label** so each of train/val/test has a representative positive
  rate.
- **Oversample positives** (or weight the loss) during training.
- **Never report accuracy.** Report **ROC-AUC**, **AUPRC** (precision-recall area, the
  honest metric under imbalance), calibration (reliability curve + Brier), and a per-year
  **concordance index** for the time-to-event framing.

### 1.4 Reference numbers to sanity-check against (Sybil, published)

Sybil (the model we build on) reports ROC-AUC ≈ **0.92 at 1 year** on a held-out NLST test
set of 6,282 LDCTs, and a 6-year concordance index ≈ **0.75** on NLST. Treat these as the
"good" bar. If our risk head lands wildly above these on our own held-out split, suspect
leakage.

---

## 2. Core design decisions (the "why")

These are the four decisions that everything else follows from. If you disagree with one,
raise it before building — but they are chosen deliberately.

### 2.1 Stand on Sybil; don't train a worse model from scratch

**Sybil** is an open-source (MIT/Harvard) deep-learning model trained on NLST that predicts
1–6-year lung-cancer risk from a _single_ LDCT, with no clinical inputs or manual
annotation required at inference. It is pip-installable, ships an ensemble of pretrained
weights, returns per-year risk scores, and — critically for us — exposes **attention maps**
and a helper that overlays them on the slices and can export a GIF.

```python
from sybil import Serie, Sybil, visualize_attentions

model  = Sybil("sybil_ensemble")                      # pretrained ensemble
serie  = Serie(list_of_dicom_paths)                   # one CT series
result = model.predict([serie], return_attentions=True)

risk       = result.scores[0]          # 6 numbers: risk at years 1..6
attentions = result.attentions[0]      # per-slice (volume) + per-pixel (image) attention

overlays = visualize_attentions([serie], attentions=result.attentions,
                                save_directory="out/", gain=3)   # heatmap PNGs / GIF
```

This single dependency delivers **the risk factor** and **the "highlight places of
importance"** requirements out of the box. Sybil's attention is _supervised_ by the lesion
bounding boxes during training, so its heatmaps localize suspicious tissue rather than being
generic saliency.

> **Note on Sybil input ordering.** Sybil expects an axial LDCT where the **first frame is
> the abdomen and the last frame is the clavicles** (i.e. inferior → superior). Our
> ingestion step must detect and, if needed, flip slice order using DICOM
> `ImagePositionPatient` before handing a series to Sybil. This is the #1 silent-failure
> gotcha.

**What "the ML that trains" then means for us.** We are not throwing away the training
requirement — we keep a genuine, held-out-evaluated training loop, but we spend the compute
where it pays off:

- a small **trainable head** on top of Sybil (recalibration + optional fine-tune of the
  last block on our split), so the system provably "trains and is tested on held-out data";
- a **longitudinal change model** (Section 6.2), which is the genuinely novel, differentiated
  piece and the direct answer to "detect changes in lung tissue."

A full from-scratch 3D CNN is documented as a _later_ option (Section 6.4), not the MVP.

### 2.2 Treat the dataset as an index, not a download

The dataset is too big to hold, so we never hold it. Use **`idc-index`** (the official IDC
Python package): it queries ~100 TB of DICOM metadata with SQL (DuckDB, local, no
download) and does high-speed parallel pulls from the public buckets via the bundled
`s5cmd`. Files are addressed as `<crdc_series_uuid>/<crdc_instance_uuid>.dcm`.

```python
from idc_index import IDCClient
client = IDCClient.client()
cohort = client.sql_query("""
  SELECT PatientID, StudyInstanceUID, SeriesInstanceUID, series_size_MB
  FROM index
  WHERE collection_id = 'nlst' AND Modality = 'CT'
""")                                   # metadata only — nothing downloaded yet
```

The existing `manifest_*.s5cmd` file is a frozen snapshot of exactly this query; keep it as
the offline fallback, but prefer `idc-index` so cohorts are rebuildable and versioned. The
existing `fetch_patient.py` (clear → download one patient → export) becomes one worker
inside the streaming layer rather than the whole story.

### 2.3 The split is deterministic, patient-level, and hash-derived

This is the whole "randomly mark some for out-of-training" mechanism, and it is boring on
purpose:

```python
import hashlib
def assign_split(patient_id: str, seed: str = "nlst-v1") -> str:
    h = int(hashlib.sha256(f"{seed}:{patient_id}".encode()).hexdigest(), 16)
    r = (h % 10_000) / 10_000            # stable pseudo-random in [0,1)
    return "train" if r < 0.70 else "val" if r < 0.85 else "test"   # 70/15/15
```

Why this instead of `random.shuffle`:

- **No leakage.** A patient's T0/T1/T2 scans are highly correlated; hashing the _patient_
  keeps all of them in one split. A naive per-scan shuffle would put a patient's baseline in
  train and their follow-up in test and inflate every metric.
- **Reproducible & download-free.** You can compute any patient's split from its ID alone,
  before fetching a single byte. Changing `seed` gives a fresh split; keeping it guarantees
  the test set is stable forever.
- **Streamable.** A training run for the "train" split simply skips any patient whose hash
  says val/test — no central shuffle file needed.

For imbalance, apply the hash _within label strata_ (compute the split separately for the
positive and negative pools) so the 8% positive rate is preserved in each split. Persist the
result to `splits.parquet` for auditability, but the hash remains the source of truth.

### 2.4 Longitudinal change is the spine, not an afterthought

NLST is longitudinal by design and the viewer already has a **before/after comparison**.
So the "detect changes in lung tissue" requirement maps onto: register a patient's
consecutive rounds (T0→T1→T2), compute a change map, and surface growing/new regions. This
is the piece that makes the tool feel novel rather than "we ran a public model," and it can
be demoed _without_ CDAS labels (registration + differencing is self-supervised; labels only
validate it).

---

## 3. Architecture overview

```
                          ┌─────────────────────────────────────────────┐
                          │  React viewer (existing, restyled)           │
                          │  slice browser · before/after · 3D · results │
                          └───────────────▲─────────────────────────────┘
                                          │  REST / SSE (job status)
                          ┌───────────────┴─────────────────────────────┐
                          │  FastAPI service (new: server/)              │
                          │   /cohort  /patient  /infer  /jobs  /results │
                          └───┬───────────┬───────────┬──────────────────┘
             ┌────────────────┘           │           └─────────────────┐
             ▼                            ▼                             ▼
   ┌───────────────────┐      ┌────────────────────────┐     ┌────────────────────┐
   │ Cohort + Split    │      │ Ingestion / Streaming  │     │ Inference workers  │
   │ (idc-index, hash) │      │ (idc download, s5cmd,  │     │  · Risk  (Sybil)   │
   │ → splits.parquet  │      │  bounded LRU cache)    │     │  · Change (reg.)   │
   └───────────────────┘      └───────────┬────────────┘     │  · Saliency        │
                                          ▼                  └─────────┬──────────┘
                              ┌────────────────────────┐               │
                              │ Preprocess             │               ▼
                              │ (export_volumes.py++)  │     ┌────────────────────┐
                              │ HU volume + meta.json  │────►│ Evaluation + report│
                              └────────────────────────┘     │ (held-out only)    │
                                                             └────────────────────┘
```

Everything below the FastAPI line is a set of Python modules that can run in-process for the
MVP and be split into a job queue later. The viewer talks only to the FastAPI service.

---

## 4. The pipeline, stage by stage

Each stage lists its **job**, **in → out**, **key libraries**, and **how it relates to the
current repo**.

### Stage 0 — Cohort & split service

- **Job:** produce the master list of usable series and each patient's split assignment.
- **In:** `collection_id = 'nlst'` (+ optional filters: only reconstructions matching
  Sybil's slice-thickness expectations, exclude scouts/localizers).
- **Out:** `data/cohort/series.parquet` (one row per series: `PatientID`,
  `StudyInstanceUID`, `SeriesInstanceUID`, study year T0/T1/T2, size, s5cmd URL) and
  `data/cohort/splits.parquet`.
- **Libs:** `idc-index`, `duckdb`, `pandas`.
- **Repo relation:** supersedes the static `manifest_*.s5cmd` (keep it as fallback).

**Series-selection rule of thumb:** NLST has multiple reconstructions per study (different
kernels/thicknesses). For the risk head, pick _one_ series per study deterministically
(prefer thin-slice soft-kernel reconstructions; break ties by `SeriesInstanceUID` hash) so
results are reproducible. Record which was chosen in metadata.

### Stage 1 — Ingestion / streaming

- **Job:** given a set of series UIDs, fetch their DICOMs on demand and never let the disk
  footprint exceed a budget.
- **In:** list of `SeriesInstanceUID`. **Out:** local DICOM dirs (transient).
- **Behaviour:** a **bounded LRU cache** keyed by series UID with a hard size cap (e.g.
  50 GB). On cache miss → `idc download <uid>` (or `s5cmd cp`) → on cap exceeded → evict
  least-recently-used. Training iterates the split by streaming shards; the working set is
  a rolling window, never the whole 11 TB.
- **Libs:** `idc-index` / `s5cmd`, a small LRU manager.
- **Repo relation:** generalizes `fetch_patient.py` and `download_sample.py`. Keep
  `fetch_patient.py`'s "clear patient dir + export for viewer" behaviour as the _interactive_
  single-patient path used by the app button.

### Stage 2 — Preprocessing

- **Job:** turn a raw DICOM series (100–600 MB) into a compact, model-ready volume (a few
  MB) plus metadata — computed **once per series** and cached.
- **In:** DICOM dir. **Out:** `volume.npy`/`.zarr` (int16 HU) + `meta.json`.
- **Steps:**
     1. Stack slices by `ImagePositionPatient`; rescale to HU (`RescaleSlope`/`Intercept`).
     2. Resample to isotropic spacing (e.g. 1×1×1 mm or Sybil's expected geometry).
     3. Clip HU to a sane range (e.g. `[-1024, 400]`); keep a raw copy for the model,
        compute a lung-window view (level ≈ −600, width ≈ 1500) for display.
     4. Optional **lung mask** (e.g. `lungmask` / TotalSegmentator output) to focus change
        detection on parenchyma and drop bed/air.
     5. Emit `meta.json`: spacing, orientation, slice order (and whether it was flipped for
        Sybil), patient/study/series IDs, study year, split.
- **Libs:** `pydicom`, `SimpleITK`, `numpy`, optionally `lungmask`.
- **Repo relation:** this is `export_volumes.py` extended (it already produces "raw 16-bit
  HU volumes + JSON metadata for the viewer"). `dicom_to_png.py` stays as the PNG-for-quick-
  training/QC path.

### Stage 3 — Inference

Three sub-models, one results object (see Section 5.3 for the schema).

- **3a Risk (Sybil):** run the ensemble on the selected series, `return_attentions=True`.
  Output: 6 calibrated risk numbers (years 1–6) + attention tensors.
- **3b Change (longitudinal):** for a patient with ≥2 rounds, register the later scan to the
  earlier (see Section 6.2), compute a change/growth map, threshold into regions of interest.
- **3c Saliency:** convert Sybil's image + volume attention into per-slice heat overlays and
  a "top-K most attended slices" list; overlay change regions similarly.
- **Libs:** `sybil`, `SimpleITK`/`antspyx` (registration), `torch`.

### Stage 4 — Evaluation (held-out only)

- **Job:** measure the system honestly on the **test** split, which no training touched.
- **Metrics:** ROC-AUC and **AUPRC** per horizon (1/2/.../6 yr), calibration (reliability
  curve + Brier), 6-yr concordance index; for change detection, agreement of detected growth
  regions against NLSTseg/Sybil annotations where available (Dice / detection recall).
- **Out:** `reports/eval_<run>.json` + plots. Emit a leakage guard that asserts no test
  `PatientID` appears in the train manifest.
- **Libs:** `scikit-learn`, `lifelines` (C-index), `matplotlib`.

### Stage 5 — Explanation & summary

- **Job:** assemble the human-facing result.
- **Contents:** the risk gauge + 1–6-yr curve; the top attended slices with overlays; the
  change map; and a **plain-language summary** ("Estimated 1-year risk 4.1% [CI …], below /
  above the screening cohort median. Attention concentrated in the right upper lobe;
  compared with the 2003 baseline, a ~6 mm region there enlarged.").
- **Summary generation:** MVP uses a deterministic **template** filled from the results
  object (fully offline, reproducible, no hallucination risk). A later option is to pass the
  structured results to an LLM for nicer prose — but the numbers must always come from the
  template, never be invented by the model.

### Stage 6 — Serving

- **Job:** expose the above to the viewer; run inference asynchronously.
- **MVP:** FastAPI, in-process background tasks, results written to `results/<jobid>.json`
  and volumes/overlays under `viewer/public/data/`. **Later:** a real queue (Redis/RQ or
  Celery) + a worker pool + GPU scheduling.

---

## 5. On-disk contracts (so the team can build stages in parallel)

Fixed schemas let front-end, ingestion, and ML proceed independently.

### 5.1 `splits.parquet`

`PatientID · split(train|val|test) · label(0|1|unknown) · label_source(cdas|sybil|nlstseg|none) · n_rounds`

### 5.2 `meta.json` (per series)

```json
{
	"patient_id": "100012",
	"study_year": "T0",
	"series_uid": "1.2.840...",
	"spacing_mm": [1, 1, 1],
	"orientation": "IS",
	"flipped_for_sybil": true,
	"hu_clip": [-1024, 400],
	"lung_mask": "mask.npy",
	"split": "test"
}
```

### 5.3 `results/<jobid>.json` (the object the viewer renders)

```json
{
  "patient_id": "100012",
  "risk": { "year_1": 0.041, "year_2": 0.06, "year_6": 0.14,
            "ci_1": [0.03, 0.055], "cohort_percentile": 62 },
  "attention": { "top_slices": [88, 91, 84],
                 "overlay_dir": "public/data/100012/attn/" },
  "change": { "pairs": [{"from":"T0","to":"T1",
                         "regions":[{"slice":90,"bbox":[...],"delta_mm":6.2,"kind":"growth"}],
                         "overlay_dir":"public/data/100012/change_T0_T1/"}] },
  "summary": "Estimated 1-year risk 4.1% …",
  "provenance": { "model":"sybil_ensemble", "code_rev":"<git sha>", "research_only": true }
}
```

---

## 6. ML specifics

### 6.1 Risk head

- **MVP:** Sybil ensemble inference + a **calibration layer** (isotonic/Platt) fit on the
  _train_ split so probabilities are honest on our cohort. This alone is a legitimate
  "trained + held-out-tested" artifact.
- **Stretch:** unfreeze Sybil's final block and fine-tune on the train split with a
  class-weighted survival loss; validate on val; report on test. Expect only marginal gains
  over the pretrained ensemble — Sybil was already trained on this distribution.

### 6.2 Change head (the novel piece)

- **Registration:** rigid → affine → deformable (B-spline / SyN) via `SimpleITK` or
  `antspyx`, aligning T(n+1) to T(n) inside the lung mask.
- **Change signal (MVP):** post-registration HU difference, denoised, masked to lung, then
  connected-component regions with size/HU-delta filters → candidate growth/new-lesion
  regions. Cheap, explainable, needs no labels.
- **Change signal (stretch):** a **Siamese/change-detection CNN** trained on registered
  pairs, supervised where NLSTseg masks or Sybil boxes exist, to distinguish true nodule
  growth from noise/atelectasis/registration error.
- **Output:** per-pair region list + overlay volumes that drop straight into the existing
  before/after view.

### 6.3 Evaluation protocol (non-negotiable)

Train touches **only** `train`; hyperparameters chosen on `val`; every headline number
comes from `test`, computed once. A CI check fails the build if any test `PatientID` is
found in the training manifest. Report AUPRC alongside AUC because of the 8% base rate.

### 6.4 From-scratch option (explicitly _not_ MVP)

A 3D CNN (e.g. a 3D ResNet / nnU-Net-style encoder) trained end-to-end on streamed shards.
Document it, keep the streaming/split infra compatible with it, but do not block the MVP on
it — it needs far more compute and will, at best, match Sybil.

---

## 7. Backend API surface (MVP)

| Method / path                                              | Does                                                     |
| ---------------------------------------------------------- | -------------------------------------------------------- | ------- | ---- | --------------------------------- |
| `GET /cohort?split=test&limit=50`                          | list patients (id, n_rounds, label if known, split)      |
| `GET /patient/{id}`                                        | metadata + available rounds + whether volumes are cached |
| `POST /patient/{id}/fetch`                                 | trigger stream+preprocess (async) → returns `job_id`     |
| `POST /infer` `{patient_id, tasks:[risk,change,saliency]}` | run inference → `job_id`                                 |
| `GET /jobs/{job_id}`                                       | status (`queued                                          | running | done | error`) + progress (SSE friendly) |
| `GET /results/{patient_id}`                                | the `results/<jobid>.json` object                        |
| `GET /random-patient`                                      | pick a random cohort patient (backs the existing button) |

Keep the existing "Fetch a random patient" button; it now calls `/random-patient` →
`/patient/{id}/fetch` → `/infer` and streams progress via `/jobs`.

---

## 8. UI / UX specification — Anthropic/Claude style

The aesthetic goal: **calm, editorial, and obviously intentional.** It should read like a
well-made reading tool, not a flashy dashboard and not a clinical alarm system. Restraint is
the brief. (Colours below are a _starting palette in that spirit_ — tune to taste; they are
not claimed to be exact brand values.)

### 8.1 Design tokens

```
--bg            #F5F2EC   /* warm off-white / ivory "paper", not pure white     */
--surface       #FBFAF7   /* cards sit a shade lighter than the page            */
--ink           #1A1A18   /* near-black text, warm not blue-black               */
--ink-muted     #6B6862   /* secondary text, captions                          */
--line          #E5E0D8   /* hairline borders — 1px, used instead of shadows    */
--accent        #C15F3C   /* one muted clay/terracotta accent, used sparingly   */
--accent-soft   #E9C7BC   /* accent tint for fills/hover                        */
--ok            #4F7A5B   /* calm green for "below cohort median"               */
--warn          #B98A2E   /* amber, reserved; never a screaming red             */

font-sans: "Inter", system-ui, sans-serif        /* UI, numbers               */
font-serif: "Tiempos"/"Georgia" fallback          /* summary prose, headings   */

radius: 10px   spacing base: 8px grid   shadow: avoid; prefer 1px --line borders
```

Principles: **generous whitespace**, a strict 8px spacing grid, hairline borders instead of
drop shadows, **one** accent colour, a serif for the human-readable summary to make it feel
like considered writing, and **no gratuitous motion** (transitions ≤150ms, ease-out, used
only for state changes).

### 8.2 Layout — three calm columns

```
┌───────────┬─────────────────────────────────────┬──────────────────────┐
│  LEFT      │            CENTER (viewer)           │   RIGHT (results)     │
│  cohort    │  ┌───────────────────────────────┐  │  Risk gauge           │
│  · patients│  │  slice browser / before-after │  │  1–6 yr risk curve    │
│  · split   │  │  / 3D  (existing components)   │  │  Screening timeline   │
│  · search  │  │                                │  │   T0 · T1 · T2        │
│            │  │  overlay toggles:              │  │  Overlay controls     │
│            │  │   [ Attention ] [ Change ]     │  │  Summary card (serif) │
│            │  └───────────────────────────────┘  │  Provenance / disclaimer│
└───────────┴─────────────────────────────────────┴──────────────────────┘
```

- **Left rail** — the cohort list. Each row: patient id, small split chip (`test`), round
  count. Filter by split so a demo can show "held-out test patients only." Quiet, scannable,
  no avatars or noise.
- **Center** — the existing viewer (slice browser, before/after, 3D) is the hero. Add two
  overlay toggles and an **opacity slider** for heatmaps. Keyboard: `←/→` slices, `[`/`]`
  rounds, `space` play through slices.
- **Right rail** — the results panel (below).

### 8.3 Key components

- **Risk gauge.** A single calm horizontal bar, not a red dial. Show the 1-year risk as a
  filled segment with the cohort median marked as a tick, the number large in `--ink`, and a
  confidence interval as a lighter band. Colour stays neutral; only cross the accent when
  clearly above median. **Never make risk feel alarmist** — this is decision support, not a
  diagnosis.
- **1–6-year risk curve.** A small line/sparkline of risk by horizon. Understated axis,
  hairline gridlines.
- **Screening timeline.** T0 · T1 · T2 as three dots on a line with dates; clicking one
  loads that round into the viewer; the active pair for change detection is highlighted with
  `--accent-soft`.
- **Overlay controls.** Two toggles (Attention, Change) + opacity slider + a legend. Only
  one heavy overlay on at a time by default.
- **Summary card.** Serif prose, ~2–4 sentences, generously set. This is where the tool
  "speaks." It restates the numbers from the results object verbatim (no new numbers).
- **Provenance / disclaimer footer.** Persistent, quiet line: _"Research prototype on NLST
  data. Not a medical device; not for clinical use."_ Non-negotiable given the domain.

### 8.4 States (design them, don't bolt them on)

Empty (no patient), loading (streaming/preprocess/inference — show the _stage_, e.g.
"Downloading 214 MB · Preprocessing · Running model"), error (with a retry), and
result-ready. Loading is frequent here (data is streamed), so make it a first-class, honest,
staged progress experience rather than a spinner.

### 8.5 What to avoid

Neon/red risk theatrics, dense multi-KPI dashboards, heavy drop shadows, more than one
accent colour, dark-mode-only, and animation for its own sake. When in doubt, remove
something.

---

## 9. Repo layout after the upgrade

```
CTScanViewer/
├─ CONTEXT.md                 ← this file
├─ context.md                 ← imaging primer (existing)
├─ requirements.txt
├─ manifest_*.s5cmd           ← frozen fallback cohort (existing)
├─ cohort/
│   ├─ build_cohort.py        ← idc-index query → series.parquet
│   └─ make_splits.py         ← deterministic hash split → splits.parquet
├─ ingest/
│   ├─ stream.py              ← LRU cache + idc download / s5cmd
│   └─ fetch_patient.py       ← interactive single-patient path (existing, moved)
├─ preprocess/
│   ├─ export_volumes.py      ← HU volume + meta.json (existing, extended)
│   └─ dicom_to_png.py        ← PNGs for QC / quick training (existing)
├─ ml/
│   ├─ risk.py                ← Sybil inference + calibration head
│   ├─ change.py              ← registration + change map (+ optional Siamese)
│   ├─ saliency.py            ← attention → overlays
│   ├─ train.py               ← trains calibration/fine-tune + change head (train only)
│   └─ evaluate.py            ← held-out metrics + leakage guard
├─ server/
│   └─ app.py                 ← FastAPI: cohort/patient/infer/jobs/results
├─ viewer/                    ← React app (existing, restyled per §8)
└─ data/, results/, reports/  ← all git-ignored
```

---

## 10. Roadmap (suggested milestones)

1. **M0 — Cohort & split (no ML).** `idc-index` cohort + deterministic splits; CI leakage
   guard. _Exit:_ `splits.parquet` exists; test-set patient list is stable across runs.
2. **M1 — Stream → preprocess → view one held-out patient.** Wire streaming + extended
   `export_volumes.py` into the viewer's existing button. _Exit:_ pick a random _test_
   patient, see its volume, footprint stays under budget.
3. **M2 — Risk + saliency.** Sybil inference + attention overlays + risk gauge/curve in the
   UI. _Exit:_ a test patient shows a calibrated risk and correct heat overlays.
4. **M3 — Change detection.** Registration + change map in the before/after view. _Exit:_ a
   multi-round test patient shows a plausible growth region.
5. **M4 — Evaluation + summary + UI polish.** Held-out AUPRC/AUC/calibration report;
   template summary; full Anthropic-style restyle and loading/error states.
6. **M5 (parallel, gated on CDAS) — full labels.** Join CDAS Participant table; refit
   calibration/fine-tune; publish honest test-set metrics.

Milestones M0–M4 need **only public data** and can ship before the CDAS application returns.

---

## 11. Risks, limitations, and ethics

- **Not a medical device.** Everything is research-only; the disclaimer is persistent in the
  UI and in every results object (`research_only: true`). No language anywhere should imply
  diagnosis.
- **Attention ≠ explanation.** Heatmaps show where the model attended, not proven causation.
  Present them as "regions the model weighted," not "the cancer."
- **Distribution shift.** Sybil was trained on a ~60% male, heavy-smoker US cohort. Do not
  imply validity outside that population.
- **Label latency.** The best labels require a CDAS application; the prototype track exists
  precisely so the team is never blocked waiting on it.
- **Leakage is the silent killer.** Patient-level splits + the CI guard are mandatory, not
  optional; correlated T0/T1/T2 scans make per-scan splitting quietly catastrophic.
- **Registration artifacts** can masquerade as tissue change; the change head must filter
  by size/HU-delta and, ideally, be validated against NLSTseg masks.

---

## 12. References

- NLST primary result — _Reduced Lung-Cancer Mortality with Low-Dose Computed Tomographic
  Screening_, NEJM 2011. https://doi.org/10.1056/NEJMoa1102873
- NLST collection (TCIA) — https://www.cancerimagingarchive.net/collection/nlst/
- NLST via CDAS (full clinical/outcome data + access process) — https://cdas.cancer.gov/nlst/
  and the cancer-diagnosis variable reference https://cdas.cancer.gov/learn/nlst/cancer-dx/
- NCI Imaging Data Commons — https://portal.imaging.datacommons.cancer.gov/ ;
  `idc-index` package — https://github.com/ImagingDataCommons/idc-index
- **Sybil** — _A Validated Deep Learning Model to Predict Future Lung Cancer Risk From a
  Single Low-Dose Chest CT_, J Clin Oncol 2023. Code:
  https://github.com/reginabarzilaygroup/Sybil
- NLSTseg (pixel-level tumor masks) — _Scientific Data_ 2025.
  https://www.nature.com/articles/s41597-025-05742-x
- IDC-hosted NLST annotations (Sybil, NLSTSeg, TotalSegmentator) —
  https://arxiv.org/abs/2609.10858

---

_Document status: MVP design brief. Treat Sections 2 (design decisions) and 8 (UI) as the
load-bearing parts; the rest is how to realize them._
