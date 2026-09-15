# What these pictures are — a plain-language guide

This file explains, with no assumed medical or radiology background, what the
images in `data/sample` and `data/sample_png` actually show, how the folders
are organised, and what every abbreviation means.

---

## 1. Where the data comes from

- **The pictures are chest CT scans** (explained in §2) of adults who took
  part in a large American medical study called the **National Lung Screening
  Trial (NLST)**. Roughly 53,000 current or former heavy smokers aged 55–74
  were scanned once a year for three years (2002–2007) to see whether
  scanning catches lung cancer early. The scans are anonymised: names, exact
  dates and other identifying details have been removed or shifted.
- The scans are hosted for free by the **Imaging Data Commons (IDC)**, a US
  National Cancer Institute service that keeps public cancer imaging in the
  cloud. The `manifest_…aws.s5cmd` file in this folder is simply a shopping
  list of which scans to fetch from IDC's storage. `download_sample.py`
  trims that list to a handful and downloads them.

---

## 2. What one picture shows

Open any PNG in `data/sample_png`, for example
`…/CT_1.2.840…810792/0080.png`. You are looking at a **horizontal slice
through a person's chest**, as if the body were a loaf of bread cut across
and you were looking at one cut face.

- **Orientation.** The image is viewed *from the feet looking up towards the
  head* (the standard convention in radiology). So the **patient's right side
  is on the left of the picture**, and vice-versa. The **spine is at the
  bottom** of the picture (the person lay on their back), the **breastbone is
  at the top**.
- **What the greys mean.** A CT records how much each tiny bit of tissue
  blocks X-rays. Dense things block more and appear brighter:
  - **Black** – air. The two big dark regions are the **lungs** (mostly air).
    The black around the outside of the body is the air in the room.
  - **Dark grey speckles and lines inside the lungs** – blood vessels and
    airways branching through the lung.
  - **Mid grey** – soft tissue: muscle, fat, the heart (the rounded grey
    shape between the lungs at the front).
  - **White** – bone: the ribs (the arc of white blobs around the outside),
    the spine (bottom centre), the breastbone (top centre).
  - **The faint grey ring around the whole body** – the skin and the padded
    scanner table underneath.
- **One PNG is one slice.** A full scan is a *stack* of 70–260 such slices,
  a few millimetres apart, running from the base of the lungs up to the
  shoulders. The PNG filenames `0000.png, 0001.png, …` are that stack in
  order from **bottom of the chest (0000) to top**. Flick through them and
  you will see the lungs appear, swell, and shrink away.

---

## 3. How the files are organised

Every scan is stored the way hospitals store them, in a format called
**DICOM** (§5). The download tool lays them out in nested folders. Here is
the tree you currently have, with plain-language labels:

```
data/sample/
└─ nlst/                          the study the scans come from (§1)
   ├─ 100012/                     ONE PERSON (anonymous patient number)
   │  ├─ 1.2.840…956831/          ONE VISIT to the scanner: "year 0", dated 1999-01-02*
   │  │  ├─ CT_…810792/  162 files   one full chest scan, "sharp" rendering (§6)
   │  │  ├─ CT_…500079/  162 files   THE SAME scan, "smooth" rendering (§6)
   │  │  └─ CT_…147023/    1 file    a "scout" picture (§5) – not a real scan
   │  └─ 1.2.840…760484/          A SECOND VISIT one year later: "year 1", 2000-01-02*
   │     ├─ CT_…430925/  157 files   full scan, smooth rendering
   │     └─ CT_…122196/  157 files   the same scan, sharp rendering
   ├─ 102136/ └─ (visit, year 0) └─ CT_…/   71 files   full scan
   ├─ 107610/ └─ (visit, year 0) └─ CT_…/    1 file    scout only – nothing useful
   ├─ 120011/ └─ (visit, year 0) └─ CT_…/  114 files   full scan
   ├─ 122025/ └─ (visit, year 0) └─ CT_…/  160 files   full scan
   └─ 217501/ └─ (visit, year 2) └─ CT_…/  263 files   full scan
```
\* dates are deliberately fake, but the *gap* between them is real.

The four levels are always the same:

| Level | Folder name looks like | Plain meaning |
|---|---|---|
| 1 | `nlst` | Which research collection |
| 2 | `100012` | Which **person** |
| 3 | `1.2.840.113654.2.55.238…` | Which **visit** (one trip to the scanner). NLST people have up to 3 |
| 4 | `CT_1.2.840.113654.2.55.240…` | One **version of the pictures** from that visit (see §6 for why there can be several) |
| files | `xxxx.dcm` | One **slice** each |

**Why does person 100012 have more folders than the others?**
Only because of how the sample was picked. The first download took the first
five entries in the list, and the list is grouped person-by-person, so all
five belonged to 100012 — covering two visits and both renderings of each
visit. The second (random) download landed on five *different* people, one
folder each. Those people have other visits and renderings in the full
dataset; they simply weren't downloaded.

`data/sample_png/` has exactly the same folder tree, but with the `.dcm`
files replaced by `.png` files, and the 1-file scout folders left out.

---

## 4. What is in your sample right now

| Person | Visit | Files | Machine | Rendering | Slice spacing | Usable? |
|---|---|---|---|---|---|---|
| 100012 | year 0 | 162 | Siemens Volume Zoom | B50f (sharp) | 2 mm | yes |
| 100012 | year 0 | 162 | Siemens Volume Zoom | B30f (smooth) | 2 mm | yes, but duplicate of the row above |
| 100012 | year 0 | 1 | Siemens Volume Zoom | scout | – | no |
| 100012 | year 1 | 157 | Siemens Volume Zoom | B30f (smooth) | 2 mm | yes |
| 100012 | year 1 | 157 | Siemens Volume Zoom | B50f (sharp) | 2 mm | yes, but duplicate of the row above |
| 102136 | year 0 | 71 | Siemens Volume Zoom | B50f (sharp) | 5 mm | yes (coarser) |
| 107610 | year 0 | 1 | Siemens Sensation 16 | scout | – | no |
| 120011 | year 0 | 114 | GE LightSpeed QX/i | BONE (sharp) | 2.5 mm | yes |
| 122025 | year 0 | 160 | Siemens Sensation 16 | B50f (sharp) | 2 mm | yes |
| 217501 | year 2 | 263 | GE LightSpeed QX/i | STANDARD (smooth) | 2.5 mm | yes |

So: **10 folders, 6 people, 8 real scans, of which 2 are near-duplicates, and
2 scouts** — effectively 6 distinct chest volumes.

---

## 5. Glossary

Every abbreviation or term you will meet in the folder names, the scripts, or
my earlier messages.

### The dataset and the download

| Term | Meaning |
|---|---|
| **NLST** | National Lung Screening Trial — the medical study the scans come from (§1). |
| **IDC** | Imaging Data Commons — the free cloud library the scans are downloaded from. |
| **idc-index** | The Python package that knows what is in IDC and does the downloading. |
| **manifest** | The `.s5cmd` text file: a list of everything you asked IDC for, one line per scan folder. |
| **s5cmd** | A command-line tool for copying files out of cloud storage very fast. The manifest is written in its command language (`cp s3://… .` means "copy this folder here"). |
| **s3 / AWS** | Amazon's cloud storage, where IDC keeps the files. |
| **collection** | IDC's word for one dataset. Here there is only one: `nlst`. |
| **series** | IDC's (and DICOM's) word for one folder of slices — one "version of the pictures". Each line in the manifest is one series. |
| **CT / SEG / SR** | The three kinds of thing in the manifest. **CT** = the actual scan pictures. **SEG** = "segmentation": a computer- or human-drawn outline of some organ or nodule, stored as a mask, not a photo. **SR** = "structured report": a text/table of measurements, no picture at all. Only CT is useful as training images, so the download script keeps only CT. |

### The file format

| Term | Meaning |
|---|---|
| **DICOM** | *Digital Imaging and Communications in Medicine.* The universal file format for medical scans. A DICOM file holds one picture **plus** a long header of labelled facts about it (who, when, which machine, how it was set up). |
| **.dcm** | File extension for one DICOM file. In a CT, one `.dcm` = one slice. |
| **pydicom** | The Python library used to read `.dcm` files. |
| **UID** | *Unique Identifier* — the very long dotted numbers like `1.2.840.113654.2.55.238…`. Just a globally unique name, like a serial number. Not meaningful to read. |
| **PatientID** | The anonymous person number (`100012`). |
| **StudyInstanceUID** | The UID of one *visit* to the scanner. ("Study" in medical jargon means one imaging appointment, not a research study.) |
| **SeriesInstanceUID** | The UID of one folder of slices. |
| **instance / InstanceNumber** | One slice, and its position number in the stack. |
| **ImagePositionPatient** | Header field giving the physical x, y, z position (in mm) of a slice inside the scanner. The scripts sort slices by z so they come out in the right order. |
| **Modality** | The type of machine that made the picture. `CT` here. (Others you may meet elsewhere: MR = MRI, CR/DX = ordinary X-ray, PT = PET.) |
| **SeriesDescription** | A free-text label the scanner attached to the series. In NLST it is a coded string like `0,OPA,SE,VZOOM,B50f,300,2,120,75,40,na` — decoded in §6. |
| **Transfer syntax / "Explicit VR Little Endian"** | How the pixels are packed inside the file. This one means *uncompressed*, which is why no extra decoding libraries were needed. |

### The scanner

| Term | Meaning |
|---|---|
| **CT** | *Computed Tomography.* An X-ray tube spins around the patient while they slide through a ring; a computer turns the hundreds of X-ray views into a stack of cross-section pictures. |
| **low-dose CT** | A CT run with less radiation than usual, because NLST scanned healthy people every year. The pictures are grainier than a hospital diagnostic CT. |
| **axial** | The slice direction: horizontal cuts across the body. (The other two directions are *coronal* = front-to-back sheets, and *sagittal* = left-to-right sheets. All your pictures are axial.) |
| **slice thickness** | How thick each cut is, in mm. 2 mm means the slices are 2 mm apart; 5 mm scans have fewer, coarser slices. |
| **localizer / scout / topogram** | A quick, low-quality flat picture (like an ordinary chest X-ray) taken *before* the real scan so the operator can choose where to scan. It lives in its own 1-file folder, is labelled modality CT, but is not a slice of anything. The scripts skip it. |
| **kernel / convolution kernel / reconstruction kernel** | The sharpening filter the scanner applies when turning raw X-ray data into pictures. Each manufacturer has its own code names (see §6). Think of it as "render mode". |
| **reconstruction** | Producing pictures from the raw scanner data. A scanner can reconstruct the *same* raw data several times with different kernels, giving several near-identical folders. |
| **kVp, mA, mAs, pitch** | X-ray tube settings (voltage, current, exposure, how fast the table moves). They affect image noise and radiation dose. You can ignore them. |
| **FOV / ReconstructionDiameter** | *Field of view*: how wide (in mm) the circle in the picture is. 300 mm means the 512-pixel-wide picture spans 30 cm, so each pixel is ~0.59 mm. |
| **PixelSpacing** | The real-world size of one pixel, in mm. Varies from scan to scan (0.55–0.78 mm here). |
| **Siemens Volume Zoom / Sensation 16, GE LightSpeed QX/i** | Model names of the CT machines used. |

### Pixel values and the PNG conversion

| Term | Meaning |
|---|---|
| **HU / Hounsfield unit** | The standard scale for CT pixel values. Air = −1000, water = 0, fat ≈ −100, muscle/organs ≈ +40, bone ≈ +400 to +1500 or more. Lung tissue (mostly air) is around −800. |
| **RescaleSlope / RescaleIntercept** | Two header numbers that convert the raw stored integers into HU: `HU = raw × slope + intercept`. Here slope = 1, intercept = −1024. |
| **12-bit / 16-bit / 8-bit** | How many grey levels a pixel can hold. The scanner stores 12 bits (4096 levels). A normal PNG is 8 bits (256 levels), so you must throw away detail to make one — hence *windowing*. `--bits 16` keeps everything. |
| **window / windowing** | Choosing which slice of the HU range to spread across black-to-white. Everything below the window is black, everything above is white. Specified as a **centre** (level) and a **width**. |
| **lung window** | Centre −600, width 1500 → shows −1350…+150 HU. Good for seeing detail *inside* the lungs; bone and organs all saturate to white. This is the default the PNGs were made with. |
| **soft-tissue window** | Centre 40, width 400 → shows −160…+240 HU. Organs, muscle and fat become distinguishable; lungs go solid black. |
| **bone window** | Centre 400, width 1800 → shows −500…+1300 HU. Detail within bone. |
| **PNG** | An ordinary lossless picture file. One per slice. |

---

## 6. Decoding the SeriesDescription code

NLST labelled each series with a comma-separated code. Example:

```
0,OPA,SE,VZOOM,B50f,300,2,120,75,40,na
```

| Position | Example | Meaning |
|---|---|---|
| 1 | `0` | Screening year: `0` = first visit, `1` = one year later, `2` = two years later |
| 2 | `OPA` | Image kind: `OPA` = Original Primary **Axial** (a real slice stack). `OPL` / `DSL` = a **localizer** (scout picture) |
| 3 | `SE` / `GE` | Manufacturer: **Siemens** / **GE** |
| 4 | `VZOOM`, `SEN16`, `LSQX` | Scanner model: Siemens **Volume Zoom**, Siemens **Sensation 16**, GE **LightSpeed QX/i** |
| 5 | `B50f` | Reconstruction kernel — see below |
| 6 | `300` | Field of view, mm |
| 7 | `2` | Slice thickness, mm (`na` for scouts) |
| 8 | `120` | Tube voltage, kVp |
| 9–11 | `75,40,na` | Radiation-dose settings (tube current / exposure / pitch). Not needed for anything here. |

**Kernel names, translated:**

| Code | Manufacturer | What it does |
|---|---|---|
| `B30f` | Siemens | *Smooth.* Less noise, softer edges. Standard for looking at organs. |
| `B50f` | Siemens | *Sharp.* More noise, crisper edges. Standard for looking at lung detail. |
| `STANDARD` | GE | Smooth (≈ B30f). |
| `BONE` | GE | Sharp (≈ B50f). Despite the name, it's the usual lung-detail setting. |
| `T20s` | Siemens | The kernel used for scout pictures. Ignore. |

When a visit folder has both a `B30f` and a `B50f` series with the same
slice count, they are **the same scan rendered twice** — same person, same
moment, same slices, just filtered differently.

---

## 7. What this means for training a model

1. **A "sample" is a slice or a volume, not a `.dcm` folder.** For 2-D
   models each PNG is one input; for 3-D models each series folder is one
   input.
2. **Split by person, not by file.** The two renderings of one scan, and the
   year-0/year-1 scans of one person, are highly correlated. If some end up
   in training and others in validation, your validation score will be
   inflated.
3. **Skip 1-file folders** (scouts). Both scripts already do.
4. **Pixel sizes and slice thicknesses differ between scans** (0.55–0.78 mm
   per pixel; 2–5 mm per slice). Many pipelines resample everything to a
   common spacing first.
5. **The 8-bit PNGs have been windowed for lungs**; anything denser than
   about +150 HU is clipped to pure white. If you want the model to see
   bone/organ detail too, regenerate with `python dicom_to_png.py --bits 16`
   and normalise in your own data loader.
