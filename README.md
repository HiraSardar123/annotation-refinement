# SAM2 Annotation Refinement + Review Tool

Tools for welding high-speed-camera annotations (`molten_consumable` and
`droplet`):

- **Part A -- SAM2 refinement.** Takes human annotations (a COCO JSON with
  polygons + the grayscale frames), converts them to RGB masks, and lets
  **SAM2.1 Hiera-Large** refine the boundary of every molten and droplet
  object. Result: one refined RGB mask per frame, plus human | SAM comparison
  images and a per-image evaluation CSV.
- **Part B -- Review tool.** A local browser app that shows the human
  annotation and the SAM result side by side, object by object, and lets you
  correct the SAM result and save the reviewed masks.

Every command below is for **Windows PowerShell**, run **from the repository
root**.

---

## Contents

1. [Repository layout](#1-repository-layout)
2. [Setup (once)](#2-setup-once)
3. [Input data format](#3-input-data-format)
4. **[Part A -- SAM2 refinement, step by step](#part-a----sam2-refinement-step-by-step)**
5. **[Part B -- Review tool, step by step](#part-b----review-tool-step-by-step)**
6. [Where everything goes (inputs and outputs)](#6-where-everything-goes)
7. [All commands together](#7-all-commands-together)
8. [How the refinement works](#8-how-the-refinement-works)
9. [All pipeline options](#9-all-pipeline-options)
10. [Running one step on its own](#10-running-one-step-on-its-own)
11. [Review tool reference](#11-review-tool-reference)
12. [Troubleshooting](#12-troubleshooting)
13. [What is not in the repository](#13-what-is-not-in-the-repository)

---

## 1. Repository layout

```text
annotations/
|-- coco_to_rgb.py          Part A step 1: COCO JSON + grayscale frames -> RGB frames + RGB masks
|-- run_dataset.py          Part A: run the SAM2 refinement on a whole dataset
|-- run_folder.py           Part A: run it on one experiment folder
|-- check_env.py            checks that SAM2 loads and predicts
|-- pipeline/               the refinement, one file per step (section 8)
|   |-- main.py             runs every step; called by run_dataset.py / run_folder.py
|   |-- config.py           every option and its default
|   |-- common.py           palette, reading/writing, finding masks and frames
|   |-- segregate.py  boxes.py  prompts.py  sam_predict.py  relabel.py
|   |-- instances.py  molten_top.py  gaps.py  spatter.py
|   |-- visualize.py        human | SAM comparison images
|   |-- evaluate.py         evaluation.csv
|   `-- step_cli.py         run one step on one mask
|-- annotate.py             Part B: starts the review tool
|-- annotator/              Part B: the review tool
|   |-- server.py           local web server (127.0.0.1 only)
|   |-- data.py             loads human / SAM masks, saves edits
|   `-- static/             the browser app (index.html, app.js, style.css, selftest.js)
|-- requirements-sam2.txt   Python packages
|
|-- data/                   your data -- NOT in git (section 3)
|-- third_party/sam2/       SAM2 source code -- NOT in git, clone it (section 2)
|-- sam2.1_hiera_large.pt   SAM2 weights -- NOT in git, download it (section 2)
`-- .venv/                  Python environment -- NOT in git, create it (section 2)
```

---

## 2. Setup (once)

Needs **Python 3.10 or newer** (tested with 3.14, torch 2.14) and **git**.
A CUDA GPU is optional: on CPU SAM takes about 5 s per image.

**2.1 Get the repository**

```powershell
git clone <this repository's URL> annotations
cd annotations
```

**2.2 Get the SAM2 source code** into `third_party\sam2` (the version this
was tested with):

```powershell
git clone https://github.com/facebookresearch/sam2.git third_party\sam2
git -C third_party\sam2 checkout 2b90b9f
```

**2.3 Create the Python environment and install the packages**

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install -r .\requirements-sam2.txt
```

This installs torch, torchvision, opencv-python, numpy, pillow and SAM2
itself (from `third_party\sam2`). With an NVIDIA GPU, install the CUDA build of
torch first, using the command from <https://pytorch.org/get-started/locally/>,
then run the line above.

**2.4 Download the SAM2.1 Hiera-Large weights** (about 860 MB) and save them
as `sam2.1_hiera_large.pt` in the repository root:

```text
https://dl.fbaipublicfiles.com/segment_anything_2/072824/sam2.1_hiera_large.pt
```

```powershell
Invoke-WebRequest https://dl.fbaipublicfiles.com/segment_anything_2/072824/sam2.1_hiera_large.pt -OutFile .\sam2.1_hiera_large.pt
```

**2.5 Check that SAM2 loads and predicts:**

```powershell
.\.venv\Scripts\python.exe .\check_env.py --root . --checkpoint .\sam2.1_hiera_large.pt
```

> Always run from the repository root. The first `import torch` can take up
> to a minute on Windows -- it is loading, not frozen.

---

## 3. Input data format

Put the data in `data\` (it is git-ignored):

```text
data/
|-- annotations/
|   `-- <annotations>.json                  COCO JSON, polygons
`-- original_data_grayscale/droplet/unlabeled/
    |-- train/
    |   |-- <experiment 1>/
    |   |   |-- <experiment 1>_f000058.png  grayscale frames
    |   |   `-- ...
    |   `-- <experiment 2>/...
    `-- test/
        `-- <experiment>/...
```

**Frames:** `<frames folder>\<split>\<experiment>\<frame>.png`, where
`<split>` is `train` or `test`. Grayscale (or RGB) PNG, any size (here
960x960). Frame file names must be unique across all folders.

**JSON:** standard COCO, one entry in `images` per annotated frame, polygons
in `segmentation`:

```json
{
  "images":      [{"id": 1, "file_name": "<frame>.png", "width": 960, "height": 960}],
  "categories":  [{"id": 1, "name": "molten_consumable"}, {"id": 2, "name": "droplet"}],
  "annotations": [{"id": 1, "image_id": 1, "category_id": 1,
                   "segmentation": [[x1, y1, x2, y2, ...]], "bbox": [...], "area": ...}]
}
```

- `file_name` is matched to the frame **by file name only** (folders in it
  are ignored).
- Category names are matched loosely ("Molten Consumable" =
  "molten_consumable"). Other categories are skipped with a warning.
- Frames that are not in the JSON are skipped (nothing to refine).
- RLE segmentations are not supported, only polygons.

**Classes and colours** (`class_rgb_mapping.txt`, written by
`coco_to_rgb.py`):

```text
Background: 0, 0, 0
molten_consumable: 255, 128, 0
droplet: 255, 0, 0
```

**Overlap priority** (where polygons overlap):
`molten_consumable` > `droplet` > `background`.

---

## Part A -- SAM2 refinement, step by step

### A1. Convert the JSON to RGB

The refinement does not read JSON. It reads RGB mask images (one colour per
class) next to a `class_rgb_mapping.txt`, plus the frames.
`coco_to_rgb.py` does this:

1. takes each grayscale frame and saves it as a 3-channel RGB image;
2. draws its JSON polygons into an RGB mask of the same size -- background
   black, then droplet red, then molten orange on top (molten wins overlaps);
3. writes `class_rgb_mapping.txt` next to the masks, and each experiment's
   part of the JSON (used by the review tool to tell objects apart).

```powershell
.\.venv\Scripts\python.exe .\coco_to_rgb.py `
    --coco   .\data\annotations\<annotations>.json `
    --frames .\data\original_data_grayscale\droplet\unlabeled `
    --out    .\data\prepared
```

Takes about a minute. Example output:

```text
INFO: JSON: 1152 images, 2898 annotations; frames found: 1407
INFO: written: droplet=1743, masks_train=1152, molten_consumable=1155, overlap_px=35
INFO: skipped 255 test frames that the JSON does not annotate
INFO: 35 px where molten and droplet polygons overlap (molten kept)
INFO: done -> ...\data\prepared  (18 experiment folders)
```

Result:

```text
data/prepared/<split>/<experiment>/
|-- rgb/images/<frame>.png                      frame, RGB
|-- rgb/rgb_masks/<frame>.png_rgb_mask.png      human annotation, RGB mask
|-- rgb/rgb_masks/class_rgb_mapping.txt
`-- <experiment>_coco.json
```

| Option | Meaning |
|---|---|
| `--coco` | the COCO JSON (required) |
| `--frames` | folder with `<split>\<experiment>\<frame>.png` (required) |
| `--out` | where the converted dataset goes (required) |
| `--splits` | split folders to look in (default `train test`) |

Nothing is written into `--frames`. Running it again overwrites `--out`.

### A2. Dry run -- check that every mask finds its frame

Loads nothing, writes nothing:

```powershell
.\.venv\Scripts\python.exe .\run_dataset.py --root .\data\prepared --out-root .\data\refined --dry-run
```

The last line must give the number of masks from A1, e.g.
`INFO: dry run: 1152 masks would be processed`.

### A3. Small trial

3 images from each of the first 2 folders, into a separate folder:

```powershell
.\.venv\Scripts\python.exe .\run_dataset.py --root .\data\prepared --out-root .\data\refined_trial --pilot 3 --limit-dirs 2
```

Open `data\refined_trial\spatter_refined_comp\train\<experiment>\`: each image
shows the human outline (left) and the SAM-refined outline (right) on the
photo -- molten green, droplet red. If they look right, delete
`data\refined_trial` and continue. To try other settings, add them and
`--overwrite`, e.g. `--pad-molten 2` (the molten may grow less; see
[section 9](#9-all-pipeline-options)).

### A4. Full run

```powershell
.\.venv\Scripts\python.exe .\run_dataset.py --root .\data\prepared --out-root .\data\refined --final-only
```

- About 5 s per image on CPU (1,152 images: about 1.5 hours), much faster on
  a GPU. Each image prints a line like
  `[17/1152] <mask>  comp=2 sam=2 blank=310px gaps=0px ... 4.8s`.
- **Interrupted?** Run the same command again -- finished images are
  skipped. `--overwrite` redoes them.
- `--final-only` keeps only the final masks and their comparisons. Without
  it, the intermediate stages are saved too (`refined\`, `gaps_refined\`, see
  [section 8](#8-how-the-refinement-works)).
- One experiment folder only:

  ```powershell
  .\.venv\Scripts\python.exe .\run_folder.py .\data\prepared\train\<experiment> --out-root .\data\refined
  ```

### A5. Results

```text
data/refined/
|-- spatter_refined/<split>/<experiment>/<frame>.png_rgb_mask.png   THE REFINED MASKS
|-- spatter_refined_comp/<split>/<experiment>/<frame>.png.png       human | refined images
|-- evaluation.csv                                                  per-image numbers
|-- run_report.json                                                 settings, per-image details
`-- manifest.json                                                   output -> input mask
```

- **Refined masks**: RGB PNG, same size and file name as the input masks in
  `data\prepared`, colours as in section 3.
- **`evaluation.csv`**, one row per image: `changed_px` (pixels whose class
  changed) and per class `<class>_human_px`, `<class>_refined_px`,
  `<class>_diff_px`, `<class>_boundary_dist_px` (mean distance in px between
  the human and the refined outline). Recompute it with
  `.\.venv\Scripts\python.exe -m pipeline.evaluate --out-root .\data\refined --simple`.
- The console ends with per-class totals:

  ```text
  class                   human            step10
  molten_consumable       35695      35076 (-619)
  droplet                  1752       1708 (-44)
  ```

---

## Part B -- Review tool, step by step

The tool compares the human annotation with the SAM result and lets you
correct the SAM result. It runs in the browser, but everything stays on your
PC (the server listens on 127.0.0.1 only). It needs only the environment from
section 2 -- no SAM, no GPU, no internet.

### B1. What it needs

| Input | Option | In this workflow |
|---|---|---|
| **Human side** (left, read-only): a dataset with `<split>\<experiment>\` folders containing RGB masks + `class_rgb_mapping.txt` + frames (+ optionally a COCO JSON to tell objects apart) | `--dataset` | `.\data\prepared` (from A1) |
| **SAM side** (right, editable): RGB masks with the same file names | `--result` | `.\data\refined\spatter_refined` (from A4) |
| **Where reviewed masks go** | `--out` | `.\data\edited` |

### B2. Start it

```powershell
.\.venv\Scripts\python.exe .\annotate.py --dataset .\data\prepared --result .\data\refined\spatter_refined --out .\data\edited
```

It prints how many images it found and how many have a SAM result, then opens
`http://127.0.0.1:8000/` in the browser (a short "How to use" appears the
first time; `?` brings it back). Stop it with **Ctrl+C** in the PowerShell
window. If the port is taken, add `--port 8001`.

| Option | Default | Meaning |
|---|---|---|
| `--dataset` | repository root | human side |
| `--result` | `final\spatter_refined` | SAM side |
| `--out` | `edited` | reviewed folders; work in progress is kept in `<out>_in_progress` |
| `--splits` | `train test` | splits to show |
| `--port` | 8000 | local port |
| `--no-browser` | -- | do not open the browser automatically |

### B3. Review an image

1. Pick an experiment folder and an image in the list.
2. **Left** = human annotation, **right** = "My edit" (starts as the SAM
   result). Switch the view with keys 1-4 (Photo, Outline, Overlay, Mask);
   **O** draws the human outline on My edit, so differences stand out.
3. Click an object (**V**) to see its card and its match on the other side.
4. Fix what is wrong, e.g. **Copy from human** (**H**) or *Replace with the
   human version* where SAM was worse, **Brush** (**B**) / **Eraser** (**E**)
   for small fixes. Ctrl+Z undoes.
5. **Save & next ▶** (Ctrl+Enter) -- also when nothing needed changing. An
   image counts as reviewed only when it is saved.

All tools and keys: [section 11](#11-review-tool-reference).

### B4. Outputs

```text
data/edited_in_progress/<split>/<experiment>/   folders being reviewed
data/edited/<split>/<experiment>/               fully reviewed folders
    <frame>.png_rgb_mask.png      the reviewed RGB mask (same name and size as the input)
    class_rgb_mapping.txt         its colours
    instances_coco.json           one annotation per object (uncompressed RLE),
                                  with the human / SAM object it came from
```

When the last image of a folder is saved, the folder is moved from
`edited_in_progress\` to `edited\`. Nothing is ever written into the
`--dataset` or `--result` folders.

---

## 6. Where everything goes

| Step | Reads | Writes |
|---|---|---|
| A1 `coco_to_rgb.py` | `data\annotations\*.json`, `data\original_data_grayscale\...` | `data\prepared\` |
| A2-A4 `run_dataset.py` | `data\prepared\`, `sam2.1_hiera_large.pt` | `data\refined\` (or `--out-root`) |
| B `annotate.py` | `data\prepared\`, `data\refined\spatter_refined\` | `data\edited_in_progress\`, `data\edited\` |

Without `--out-root` the pipeline writes to `outputs\`; without options the
review tool uses the repository root, `final\spatter_refined` and `edited\`.
All of these are git-ignored.

---

## 7. All commands together

```powershell
# setup (once)
git clone https://github.com/facebookresearch/sam2.git third_party\sam2
git -C third_party\sam2 checkout 2b90b9f
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r .\requirements-sam2.txt
Invoke-WebRequest https://dl.fbaipublicfiles.com/segment_anything_2/072824/sam2.1_hiera_large.pt -OutFile .\sam2.1_hiera_large.pt
.\.venv\Scripts\python.exe .\check_env.py --root . --checkpoint .\sam2.1_hiera_large.pt

# A1  JSON -> RGB
.\.venv\Scripts\python.exe .\coco_to_rgb.py --coco .\data\annotations\<annotations>.json --frames .\data\original_data_grayscale\droplet\unlabeled --out .\data\prepared
# A2  check
.\.venv\Scripts\python.exe .\run_dataset.py --root .\data\prepared --out-root .\data\refined --dry-run
# A3  trial
.\.venv\Scripts\python.exe .\run_dataset.py --root .\data\prepared --out-root .\data\refined_trial --pilot 3 --limit-dirs 2
# A4  full run
.\.venv\Scripts\python.exe .\run_dataset.py --root .\data\prepared --out-root .\data\refined --final-only

# B   review
.\.venv\Scripts\python.exe .\annotate.py --dataset .\data\prepared --result .\data\refined\spatter_refined --out .\data\edited
```

---

## 8. How the refinement works

Per image, in order. Each step is its own file in `pipeline/`. The pipeline
knows seven classes (background, molten, droplet, external arc, internal
arc, spatter, weld pool) by their RGB value; with the 2-class data of
section 3, the steps for the other classes simply find nothing to do.

| Class | RGB |
|---|---|
| background | `(0, 0, 0)` |
| molten_consumable | `(255, 128, 0)` |
| droplet | `(255, 0, 0)` |
| external_arc | `(0, 0, 255)` |
| internal_arc | `(255, 255, 0)` |
| spatter | `(0, 255, 255)` |
| weld_pool | `(255, 0, 255)` |

Classes are identified by RGB value, never by label spelling. Where a
folder's `class_rgb_mapping.txt` gives a class a slightly different colour,
that colour is translated to the standard one when the mask is read.

**Step 1 -- Segregation (`segregate.py`).** The human mask is split into the
pixels of `molten_consumable` and of `droplet`, and each class into its
8-connected components. A molten and a droplet component are never merged,
even where they touch.

**Step 2 -- Boxes (`boxes.py`).** Every component gets its own box: the tight
bounding box plus **4 px for molten** (`--pad-molten`) and **1 px for
droplet** (`--pad-droplet`). SAM's prediction is clipped to this box, so the
padding is the most a boundary can move outward. Molten and droplet stay
separate: refining one may never take pixels the human labelled as the
other.

**Step 3 -- Prompts (`prompts.py`).** 2 positive points inside the component
(the centroid, or the deepest pixel of the cross-section through it if the
centroid falls outside the shape, and a second one further along the long
axis) and 3 negative points (2 inside the box but outside the human mask, one
on each side of the long axis at the growth limit, and 1 within
`--neg-outside-px` (3) px outside the box).

**Step 4 -- SAM predictions (`sam_predict.py`).** One SAM2 call per component
with its box and points. The highest-scoring of SAM's three masks is used,
clipped to the box, kept off the other target class's human pixels, and
reduced to its largest piece. If nothing is left, the human component is
kept.

**Step 5 -- Relabel blank pixels (`relabel.py`).** A pixel the human labelled
molten or droplet that SAM left out is blank. It gets SAM's label only when
SAM is at least `--sam-conf` (80%) sure **and** that class actually surrounds
the pixel (within `--surround-tol-px` of its nearest labelled class);
otherwise it keeps its human label. Molten can never become droplet and vice
versa.

**Instances (`instances.py`)**, after steps 5, 7, 9: each molten / droplet
instance becomes one solid region -- fragments cut off from it take their
neighbouring class, and small holes and cuts (up to about
2 x `--instance-close-px`) are filled.

**Top of the molten (`molten_top.py`)**, after steps 5, 7, 9: the wire above
the molten is dark, so SAM tends to grow into it. The top
`--molten-top-keep-px` (10) rows of each molten keep the human outline, and
the strip directly above the tip may only be molten or background.

**Step 6 -- Save** to `refined/` and `refined_comp/`.

**Step 7 -- Gaps (`gaps.py`).** Thin background slivers (up to about 4 px) and
small enclosed holes (up to `--gap-max-hole-px`, 500 px) between labelled
classes are filled: with the surrounding class if one class surrounds them,
otherwise per pixel by SAM (>= 80% confident), the nearest neighbour, or the
human label.

**Step 8 -- Save** to `gaps_refined/` and `gaps_refined_comp/`.

**Step 9 -- Static spatter (`spatter.py`).** "Spatter" on the same pixels in
(almost) every mask of a folder is sensor noise; it is removed and the folder
is reported as corrupted. (No effect on data without spatter.)

**Step 10 -- Save** to `spatter_refined/` and `spatter_refined_comp/` -- the
final result.

**Step 11 -- Evaluation (`evaluate.py`)** writes `evaluation.csv`.

**Comparison images**: the human class boundaries on the left, the stage's on
the right, 1 mask pixel wide, on the photo dimmed to `--overlay-dim` (0.6)
and upscaled `--overlay-scale` (2x). Colours: molten green, droplet red,
external arc light blue, internal arc yellow, spatter cyan, weld pool
magenta.

**Full evaluation** (without `--final-only`): per image and class the pixel
count in the human mask and after steps 6, 8 and 10 (`<class>_step6`, ...),
the differences (`<class>_d6`, ...) and `changed_step6/8/10`.

**`run_report.json`**, per image: components, SAM predictions accepted,
blank pixels after step 4, gap pixels, removed spatter, and how many pixels
each step gave each class and why (`<class>:sam|agree|human|nn|surround`).

---

## 9. All pipeline options

`run_dataset.py`, `run_folder.py FOLDER`, `python -m pipeline.main` and the
single-step commands all accept these (`--help` lists them with defaults).

| Option | Default | Meaning |
|---|---|---|
| `--root` | repository root | dataset with `train\` / `test\` folders |
| `--folder PATH` | -- | one folder only (what `run_folder.py` sets) |
| `--out-root` | `outputs` | where all outputs go |
| `--splits` | train test | splits to process |
| `--pilot N` | all | first N masks per folder |
| `--limit-dirs N` | all | first N folders |
| `--overwrite` | off | redo images that already have outputs |
| `--dry-run` | off | list the work, load nothing |
| `--checkpoint` | `sam2.1_hiera_large.pt` | SAM2 weights |
| `--model-config` | `configs/sam2.1/sam2.1_hiera_l.yaml` | SAM2 config |
| `--device` | auto | cuda / cpu / auto |
| `--pad-molten` | 4 | molten box padding, px |
| `--pad-droplet` | 1 | droplet box padding, px |
| `--pad-other` | 3 | box padding for other classes' prompts (steps 5/7/9) |
| `--neg-outside-px` | 3 | how far beyond the box the outside negative may be |
| `--sam-conf` | 0.8 | SAM's label counts only at or above this confidence |
| `--relabel-ring-px` | 9 | classes within this distance are candidates |
| `--surround-tol-px` | 1.5 | a class surrounds a pixel if within this of its closest labelled class |
| `--instance-close-px` | 2 | holes / cuts inside an instance up to ~2x this wide are filled |
| `--molten-top-keep-px` | 10 | top of the molten keeps its human outline down to this many px (0 = off) |
| `--gap-close-px` | 2 | closing radius for thin gaps |
| `--gap-max-hole-px` | 500 | largest enclosed hole treated as a gap |
| `--spatter-max-missing` | 4 | static if spatter in all but at most this many masks |
| `--spatter-min-masks` | 8 | smaller folders are not judged |
| `--spatter-overlap` | 0.5 | share of a blob on the static map to remove it |
| `--visualize` / `--no-visualize` | folder: on, dataset: off | box and point debug images in `debug\` |
| `--no-comp` | -- | skip comparison images |
| `--overlay-scale` / `-dim` / `-thickness` | 2 / 0.6 / 1 | look of the images |
| `--final-only` | off | save only `spatter_refined\` + `spatter_refined_comp\`; simple evaluation |
| `--skip-eval` | off | skip the evaluation at the end |
| `--log-level` | INFO | DEBUG shows failed SAM calls |

---

## 10. Running one step on its own

Each step file runs the pipeline up to that step on one mask and saves the
result to `outputs\single\<frame>_<step>.png`. Steps 1-3 need no SAM.

```powershell
$M = ".\data\prepared\train\<experiment>\rgb\rgb_masks\<frame>.png_rgb_mask.png"

.\.venv\Scripts\python.exe -m pipeline.segregate   $M   # step 1
.\.venv\Scripts\python.exe -m pipeline.boxes       $M   # step 2: boxes image
.\.venv\Scripts\python.exe -m pipeline.prompts     $M   # step 3: points image
.\.venv\Scripts\python.exe -m pipeline.sam_predict $M   # step 4: raw SAM result
.\.venv\Scripts\python.exe -m pipeline.relabel     $M   # steps 4-5 (= step 6 result)
.\.venv\Scripts\python.exe -m pipeline.gaps        $M   # steps 4-7 (= step 8 result)
.\.venv\Scripts\python.exe -m pipeline.spatter     $M   # steps 4-9 (= step 10 result)
```

Any option from section 9 works too, e.g. `--pad-molten 5`.

---

## 11. Review tool reference

### The screen

- **Left -- Human annotation (reference, read-only)**: the `--dataset` RGB
  masks, pixel for pixel. The COCO JSON in the experiment folder is used only
  to tell separate objects of one class apart (droplet 1, droplet 2, ...).
- **Right -- My edit**: what gets saved. Starts from the `--result` mask, or
  from your last save of that image; the header says which.
- **Colours** are each folder's own `class_rgb_mapping.txt` colours, on
  screen and in the saved mask. White marks the selected object.
- **Left column**: tools, their options, a box explaining the active tool,
  and the view settings. **Right column**: the selected object's card and the
  object lists (*My edit*, *Human*, *Overlaps*).

### Tools (they always act on My edit)

| Tool | Key | What it does |
|---|---|---|
| Inspect | V | click an object in either image; its card opens and its match on the other side is highlighted. **Shift+click** to select several |
| Pan | Space | drag to move (right-drag also pans; the wheel zooms) |
| Brush | B | paint the chosen class; extends the selected object, or makes a new one. Touching other objects of the same class merges them |
| Eraser | E | make pixels background |
| Fill gap in box | G | drag a box; only unlabelled pixels inside get the chosen class |
| Change class in box | R | drag a box; pixels of the "from" class inside get the chosen class |
| Copy from human | H | paint: copy the human labels under the brush |
| Copy from SAM | S | paint: copy the original SAM labels back under the brush |

**Object card**: change class, *Merge with…*, *Replace with the human
version*, *Replace with the SAM version*, *Zoom to it*, *Delete* / *Restore*.
For a human object: *Copy it into My edit*. With nothing selected: *Reset My
edit* to the SAM or the human mask. **Merging**: Shift+click two or more
objects, then *Merge these N into one object*.

**Views**: *Photo*, *Outline*, *Overlay* (with *Mask opacity*), *Mask* --
keys 1 / 2 / 3 / 4. **Image adjustments** (display only): *Brightness*,
*Contrast*, *Local contrast*; *Reset image to default*.

**Overlaps**: where two human objects of the same class overlap in the JSON,
the Overlaps tab lets you pick which owns the shared pixels. **Objects only in
the JSON** are listed under *Human → Only in the JSON*.

Other keys: F fit, O human outline on My edit, [ ] brush size, Ctrl+Z /
Ctrl+Y undo / redo, Ctrl+S save, ← → previous / next image, Esc clear
selection. To open one image directly add
`?key=<split>/<experiment>/<mask file name>` to the address.

### Saving and progress

- **Save & next ▶** (Ctrl+Enter): save as reviewed, open the next image.
- **Save** (Ctrl+S): save without moving on.
- **⏭ Next unreviewed**: jump to the next image not reviewed yet.
- **Save all in folder**: mark every remaining image of the folder reviewed,
  as it currently is.
- **Only what's left**: show only started folders and their unreviewed
  images.
- A folder you have started must be finished before you leave it; a dialog
  lists the images still marked ○.
- Moving on with ▶ or the arrow keys does **not** save. Leaving an image with
  unsaved changes asks first.

---

## 12. Troubleshooting

| Problem | Fix |
|---|---|
| `no frame for JSON image ...` (A1) | The JSON names a frame that is not under `--frames\<split>\<experiment>\`. Check the path and the file name. |
| Dry run finds 0 masks / `nothing to do` | `--root` must be the `--out` folder of A1 (`.\data\prepared`), which contains `train\` / `test\`. |
| Looks frozen after `discovered N mask dirs` | `import torch` and loading the checkpoint are slow on Windows; wait for `SAM2 device : ...`. Ctrl+C there prints a torch traceback -- not a bug. |
| `checkpoint not found` | Download it (2.4) or pass `--checkpoint`. `--dry-run` works without it. |
| `No module named sam2` / SAM2 from the wrong place | Do 2.2 and 2.3, and run from the repository root. The log line `SAM2 package:` must end in `third_party\sam2\sam2\build_sam.py`. |
| `.ps1 cannot be loaded` when activating the venv | Not needed: call `.\.venv\Scripts\python.exe` directly as shown. |
| Comparison lines are hard to see | They are 1 mask pixel wide on purpose; zoom in or raise `--overlay-scale`. |
| `Port 8000 is already in use` (B2) | The tool is already open; use that window, stop it with Ctrl+C, or add `--port 8001`. |
| Review tool says `0 with a result mask` | `--result` must point at `spatter_refined\` of the A4 output, and the file names must match the masks in `--dataset`. |

---

## 13. What is not in the repository

`.gitignore` keeps out everything that is confidential, large, or generated:

| Not in git | Why | How to get it |
|---|---|---|
| `data/`, `train/`, `test/`, `final/`, `edited*/`, `outputs/`, `*.json` / `*.iap` / images outside the code | confidential annotation data and results | from the project's data storage |
| `sam2.1_hiera_large.pt`, `*.pt` | 860 MB weights | download (2.4) |
| `third_party/sam2/` | upstream SAM2 code | clone (2.2) |
| `.venv/` | Python environment | create (2.3) |
| `tried_code/`, `tests/`, `refine_boundaries.py`, `evaluate_refinement.py`, `sam2/` | old experiments, not used by the pipeline or the tool | -- |
