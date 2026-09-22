# Lung tissue change & cancer-risk pipeline

A screening-assistant prototype built on the [National Lung Screening
Trial](https://cdas.cancer.gov/nlst/) (NLST) imaging in the [Imaging Data
Commons](https://portal.imaging.datacommons.cancer.gov/). For a chosen patient it

1. **detects tissue change** between their screening rounds — registers T0 → T1 → T2
   and surfaces regions that densified, which is what a new or growing nodule looks like;
2. **estimates cancer risk** at 1–6 years from a single low-dose CT, using
   [Sybil](https://github.com/reginabarzilaygroup/Sybil);
3. **explains both** with heat overlays and a plain-language summary.

Nothing is downloaded until you ask for it. The full NLST imaging is ~11 TB; this
treats it as a queryable index and streams only the series it needs, into a
size-capped cache.

> **Research prototype. Not a medical device; not for clinical use.**

The engineering brief this implements is [context.md](context.md). Section
references below (§2.3, §6.3, …) point into it.

---

## Quick start

```bash
python -m venv .venv
.venv/Scripts/pip install -r requirements.txt     # bin/pip on macOS/Linux

# 1. Build the cohort and the split. Metadata only — no imaging is fetched.
.venv/Scripts/python -m cohort.build_cohort       # ~1 min → data/cohort/series.parquet
.venv/Scripts/python -m cohort.make_splits        #        → data/cohort/splits.parquet

# 2. Start the service and the viewer, in two terminals.
.venv/Scripts/uvicorn server.app:app --reload --port 8000
cd viewer && npm install && npm run dev           # http://localhost:5173
```

Open the viewer, press **Random held-out patient**, then **Analyse**.

A full run for one patient takes roughly 30 seconds of download, a few seconds of
preprocessing, and about two minutes per round-pair for registration.

### Enabling the risk model

Change detection works out of the box. Risk needs a second environment, and until
it exists the UI says so rather than showing a placeholder.

Sybil declares `python_requires = ">=3.8,<3.11"` and pins `torch==1.13.1`,
`numpy==1.24.1`, `pydicom==2.3.0`. The pipeline runs on Python 3.13 with numpy 2.x,
so the two cannot share an interpreter. Give Sybil its own:

```bash
py -3.10 -m venv .venv-sybil
.venv-sybil/Scripts/pip install -r requirements-sybil.txt
.venv/Scripts/python -m ml.risk --check           # should report: available
```

> **Do not `pip install sybil`.** That name on PyPI belongs to an unrelated
> documentation-testing package. It installs without error, imports without
> error, and contains no `Serie`, no `Sybil.predict` and no
> `visualize_attentions` — so the failure only shows up later, as an
> `AttributeError` deep in a run. The lung model has never been published to PyPI
> under that name; `requirements-sybil.txt` installs it from the repository,
> which is the only correct source. To confirm what you actually got:
> `pip show sybil` should say **Author: Peter G. Mikhael** (first author of the
> JCO paper), and `.../sybil-*.dist-info/direct_url.json` should name
> `github.com/reginabarzilaygroup/Sybil.git`. The PyPI package is Chris Withers'
> `simplistix/sybil` doctest runner, currently at v10.x.

On **Windows this installs CPU-only torch**, and silently: Sybil's marker for the
CUDA build is `platform_machine == "x86_64"`, and Windows reports `AMD64`, so it
never matches. For GPU inference, follow up with

```bash
.venv-sybil/Scripts/pip install torch==1.13.1+cu117 torchvision==0.14.1+cu117 --index-url https://download.pytorch.org/whl/cu117
```

`python -m ml.risk --check` reports which device you ended up with.

`.venv-sybil` is found automatically; set `$SYBIL_PYTHON` to put it elsewhere.
`ml/risk.py` then invokes `ml/sybil_runner.py` in that interpreter and reads JSON
back, so nothing above that seam has to care.

**The attention collation happens on Sybil's side of that seam**, deliberately.
Its raw tensors are internal to the architecture — `image_attention_1` is
`(5, 1, 25, 256)`, meaning 5 ensemble members over 25 downsampled slices and a
_flattened_ 16×16 grid, in logits. Turning that into something drawable means
exponentiating, averaging the ensemble, multiplying by the per-slice weight,
reshaping to 16×16 and upsampling, so the runner calls Sybil's own
`collate_attentions` and hands back an `(n_slices, 512, 512)` volume aligned to
the scan. Anything downstream just quantises it.

---

## How it fits together

```
 viewer/  React — cohort rail · slice browser · before/after · 3-D · results panel
    │  REST + server-sent events, proxied through Vite at /api
 server/  FastAPI — /cohort /patient /infer /jobs /results /random-patient
    │     jobs.py runs one heavy job at a time; pipeline.py sequences the stages
    ├── cohort/      idc-index SQL → series.parquet; hash split → splits.parquet
    ├── ingest/      LRU-capped DICOM cache streaming from the public buckets
    ├── preprocess/  DICOM → Hounsfield volume + lung mask + meta.json
    └── ml/          risk (Sybil) · change (registration) · saliency · train · evaluate
```

### The four decisions that shape everything

**The dataset is an index, not a download** (§2.2). `idc-index` ships the IDC
catalogue's _metadata_ as a local DuckDB file, so the whole 26,235-patient cohort is
built with a SQL query and no network transfer. NLST encodes the screening year,
kernel and slice thickness into `SeriesDescription`, so even series _selection_
stays metadata-only. Imaging is fetched per series into a cache with a hard size
cap (`CT_CACHE_GB`, default 50), least-recently-used evicted.

**The split is a hash of the patient ID** (§2.3). `assign_split("100002")` is
`train` on any machine, in any order, before a byte is fetched. A patient's T0/T1/T2
scans are near-duplicates, so hashing the _patient_ is what stops a baseline landing
in train and its own follow-up in test. `cohort/split.py` also offers an
exact-proportion stratified mode for preserving the ~8% positive rate.

**We stand on Sybil rather than training a weaker model** (§2.1). What _is_ trained
here is the calibration layer on top of it — fit on train, selected on val, reported
on test (`ml/train.py`).

**Longitudinal change is the spine** (§2.4). It is the novel piece, and it needs no
labels, so it ships while the CDAS application is pending.

---

## Commands

| Command                                               | Does                                                                                            |
| ----------------------------------------------------- | ----------------------------------------------------------------------------------------------- |
| `python -m cohort.build_cohort`                       | Query IDC → `series.parquet`. `--summary` describes an existing one.                            |
| `python -m cohort.make_splits --stratify`             | Assign splits with real outcome labels → `splits.parquet`. `--check` re-runs the leakage guard. |
| `python -m cohort.clinical --patient 100619`          | Everything known about a patient before downloading: outcome, stage, histology, rounds.         |
| `python -m ingest.stream --patient 100012`            | Fetch a patient's series into the cache. `--status`, `--evict-all`.                             |
| `python -m ingest.fetch_patient --split test`         | Fetch a random held-out patient and export for the viewer.                                      |
| `python -m preprocess.export_volumes --in data/cache` | DICOM → viewer volumes. `--masks` adds lung masks.                                              |
| `python -m ml.change --earlier DIR --later DIR`       | Register two rounds and list change regions.                                                    |
| `python -m ml.risk --check`                           | Report whether a risk model is installed.                                                       |
| `python -m ml.train --scores FILE`                    | Fit calibration on the train split.                                                             |
| `python -m ml.evaluate --predictions FILE`            | Held-out metrics, leakage guard first.                                                          |
| `python -m pytest`                                    | The test suite.                                                                                 |

---

## Where things are written

Everything below is git-ignored and reproducible; deleting any of it costs time, not
information.

```
data/cohort/      series.parquet, splits.parquet
data/cache/       DICOM, LRU-capped — the only large thing on disk
data/change/      delta maps from CLI runs of ml/change.py
viewer/public/data/   volumes + overlays the browser reads
results/          results/<patient>.json — the object the viewer renders (§5.3)
reports/          held-out evaluation reports
models/           the fitted calibrator
```

---

## Two corrections to the brief

**§1.2 is out of date: the outcome labels are not behind a CDAS application.**
IDC publishes the NLST clinical tables next to the imaging index, and
`idc-index` fetches them in about two seconds — including `nlst_prsn`, the
participant table §1.2 says requires a formal application with a turnaround
"measured in weeks". All 26,235 imaging patients join to it, so labels are
complete: **1,059 positives, 25,176 negatives, zero unknowns**, with diagnosis
timing, stage, histology, lesion size and location, and the official read of
every screening round. `cohort/clinical.py` reads it; `cohort/make_splits.py`
uses it by default.

**§1.3's headline number is wrong by a factor of two.** The brief says positives
are "~2,050 / ~26,254 ≈ 8%" and calls it "the one number that dictates ML
choices". The 2,050 is right, but it counts the _whole trial_ — 2,058 patients
across both the CT and chest-X-ray arms, 53,452 people. Our cohort is the CT arm
alone, so the real rate is **1,059 / 26,235 = 4.04%**. The imbalance is twice as
severe as planned for, which makes AUPRC-over-accuracy and positive
oversampling more important, not less.

---

## Evaluation, and what is honest to claim today

The protocol in §6.3 is enforced, not just documented: `ml/evaluate.py` runs the
leakage guard _before_ computing anything and raises rather than warns. Accuracy is
never reported — at an 8% positive rate, predicting "no cancer" for everyone scores
92% — so the metrics are ROC-AUC, **AUPRC against its base-rate floor**, Brier,
a reliability curve, and a concordance index.

**Labels are real and complete**, so supervised evaluation is unblocked. Splits
are stratified on the true outcome, holding 4.04% positives in each:

```
train   18,364 patients   741 positive   4.04%
val      3,935 patients   159 positive   4.04%
test     3,936 patients   159 positive   4.04%
```

`label` remains three-valued — `1` / `0` / `-1` — because the annotation sets
(Sybil boxes, NLSTseg masks) list positives only and say nothing about anyone
they omit. `ml/evaluate.py` still drops `-1` rather than assuming a negative.
With IDC's participant table in play, `-1` is now the empty set.

The remaining work is running Sybil across the cohort to produce the score table
that calibration and evaluation consume:

```bash
python -m cohort.make_splits --stratify                  # real labels, already the default
python -m ml.train --scores sybil_train.parquet          # fit calibration
python -m ml.evaluate --predictions sybil_test.parquet --split test
```

Reference points to sanity-check against (§1.4): Sybil reports ROC-AUC ≈ 0.92 at
1 year and a 6-year C-index ≈ 0.75 on NLST. Landing far _above_ those on our own
split is evidence of leakage, not of success.

---

## Known limitations

- **Change-detection thresholds are motivated, not validated.** They are physical
  (a region must have been air, must be tissue now, must be compact) and they take
  645 raw components down to ~10 on a real pair. But §6.2's validation against
  NLSTseg masks and Sybil boxes has not been done, so the regions are candidates
  for review, not findings. This is the most valuable next piece of work.
- **The effective sensitivity floor is ~5 mm, not the nominal 4 mm**, and reported
  diameters run 1–2 mm high. Both come from the Gaussian smoothing before
  thresholding: it erodes blobs a few voxels across and spreads the edges of
  larger ones. Measured on synthetic spheres and pinned in `tests/test_change.py`
  so it cannot drift silently.
- **Registration is verified, not trusted — because it sometimes fails.** ITK
  returns wherever the optimiser stopped, which can be worse than where it
  started. Every stage is resampled and scored, and the best is kept; on one test
  patient the scores were `{bspline: 86.8, affine: 203.2, rigid: 347.7}` HU against
  188.5 unregistered, i.e. the rigid stage alone was worse than doing nothing. If
  no stage beats the unregistered baseline, the summary says so and marks the
  regions unreliable rather than presenting them.
- **Registration is not bit-reproducible, and on some pairs it is genuinely
  unstable.** ITK accumulates the Mattes MI metric across threads, so summation
  order varies between runs even with the sampling seed fixed. Usually this shifts
  region counts by one or two. On hard pairs it is much larger: patient 110647's
  T0→T1 scored `{bspline: 86.8, affine: 203.2, rigid: 347.7}` on one run and
  `{none: 188.5, rigid: 280.6, affine: 406.2, bspline: 406.4}` on the next, from
  identical inputs. **The stage verification makes this safe rather than fixing
  it** — the worst case is a pair reported as unalignable. Diagnosing why the
  rigid stage diverges from an already-good starting position (these two volumes
  are 188 HU apart before any registration) is the open question. A line-search
  optimiser was tried for the affine stage and made it worse.
- **Overlays are large.** A change overlay is one byte per voxel at the viewer's
  resolution — ~33 MB alongside a ~66 MB volume. Fine locally, heavy over a network.
- **One job at a time.** Deliberate: concurrent fetches would evict each other from
  the shared cache. §4 marks a real queue as the later step.
- **Risk is verified against Sybil's own published reference scores.** Their
  `tests/regression_test.py` pins six expected values for a demo series; this
  pipeline reproduces all six **exactly** (absolute difference 0.0, not merely
  within the `rel_tol=1e-6` they assert), which shows the out-of-process seam —
  subprocess, JSON round-trip, score ordering — does not perturb the model.
  `tests/test_sybil_integration.py` keeps that as a standing check; it skips
  when no Sybil environment is present.
- **Attention is verified too.** On patient 100002, **71.6% of total attention
  falls inside the lung mask, which is only 17.2% of the volume** — a 4.2×
  enrichment. If that ratio ever drops toward 1.0, the collation or the slice
  alignment has broken.
- **Risk scores are uncalibrated**, and the UI says so on every reading. `0.1%`
  at one year for patient 100002 is the model's raw output; no cohort reference
  distribution has been computed yet, so no percentile is shown either.

## Ethics

Everything is research-only and every results object carries
`"research_only": true`. Attention maps show where a model attended, not where
disease is, and the UI says so. Sybil was trained on a ~60% male, heavy-smoker US
cohort aged 55–74; nothing here implies validity outside that population. The
disclaimer is persistent in the footer and non-negotiable.

## References

- NLST primary result — NEJM 2011, https://doi.org/10.1056/NEJMoa1102873
- NLST via CDAS (full clinical data + access process) — https://cdas.cancer.gov/nlst/
- NCI Imaging Data Commons — https://portal.imaging.datacommons.cancer.gov/
- `idc-index` — https://github.com/ImagingDataCommons/idc-index
- Sybil — _J Clin Oncol_ 2023; code at https://github.com/reginabarzilaygroup/Sybil
- NLSTseg (pixel-level tumour masks) — _Scientific Data_ 2025,
  https://www.nature.com/articles/s41597-025-05742-x
