r"""Convert grayscale frames + a COCO JSON (polygons) into RGB frames and RGB masks.

The pipeline never reads JSON: it needs one RGB mask per frame, in a folder
with a class_rgb_mapping.txt, and the frame next to it. This script builds
that, one folder per experiment:

    <out>/<split>/<experiment>/
        rgb/images/<frame>.png                      the grayscale frame as 3-channel RGB
        rgb/rgb_masks/<frame>.png_rgb_mask.png      the JSON drawn as an RGB mask
        rgb/rgb_masks/class_rgb_mapping.txt         Background / molten_consumable / droplet
        <experiment>_coco.json                      this experiment's part of the JSON
                                                    (lets annotate.py tell objects apart)

Frames are found under --frames as <split>/<experiment>/<frame>.png and
matched to the JSON by file name. Frames the JSON does not mention are
skipped (they are unlabelled). Classes are matched by name ("Molten
Consumable", "molten_consumable", ...).

Colours and overlap priority:

    Background:        0,   0, 0
    molten_consumable: 255, 128, 0     priority 1 (wins every overlap)
    droplet:           255,   0, 0     priority 2
    background                         priority 3 (only where nothing is drawn)

    .\.venv\Scripts\python.exe .\coco_to_rgb.py --coco .\data\annotations\<annotations>.json --frames .\data\original_data_grayscale\droplet\unlabeled --out .\data\prepared

Writes nothing into --frames. Existing outputs are overwritten.
"""

from __future__ import annotations

import argparse
import json
import logging
from collections import Counter, defaultdict
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

from pipeline.common import IMAGE_SUFFIXES, RGB_MAPPING_NAME, normalise_class, write_rgb

log = logging.getLogger("coco_to_rgb")

# class_rgb_mapping.txt, written as is into every mask folder
CLASSES = {
    "Background": (0, 0, 0),
    "molten_consumable": (255, 128, 0),
    "droplet": (255, 0, 0),
}
# overlap priority, lowest first: each class is painted over the ones before it
PAINT_ORDER = ("droplet", "molten_consumable")


def polygon_mask(seg, h: int, w: int) -> np.ndarray:
    """Same rasterisation as the annotation tool (annotator/data.py)."""
    if isinstance(seg, dict):
        raise ValueError("RLE segmentations are not supported, only polygons")
    m = np.zeros((h, w), np.uint8)
    for poly in seg:
        pts = np.asarray(poly, np.float64).reshape(-1, 2).round().astype(np.int32)
        if len(pts) >= 3:
            cv2.fillPoly(m, [pts], 1)
    return m.astype(bool)


def mapping_text() -> str:
    return "".join(f"{name}: {r}, {g}, {b}\n" for name, (r, g, b) in CLASSES.items())


def find_frames(frames_root: Path, splits: list[str]) -> dict[str, tuple[str, str, Path]]:
    """frame file name -> (split, experiment, path)."""
    found: dict[str, tuple[str, str, Path]] = {}
    for split in splits:
        split_dir = frames_root / split
        if not split_dir.is_dir():
            log.warning("no %s folder in %s", split, frames_root)
            continue
        for exp_dir in sorted(d for d in split_dir.iterdir() if d.is_dir()):
            for p in sorted(exp_dir.iterdir()):
                if p.suffix.lower() in IMAGE_SUFFIXES:
                    if p.name in found:
                        log.warning("frame name used twice, keeping the first: %s", p)
                        continue
                    found[p.name] = (split, exp_dir.name, p)
    return found


def main() -> int:
    p = argparse.ArgumentParser(description="Grayscale frames + COCO JSON -> RGB frames and RGB masks.")
    p.add_argument("--coco", type=Path, required=True, help="COCO JSON with polygon annotations")
    p.add_argument("--frames", type=Path, required=True,
                   help="folder with <split>/<experiment>/<frame>.png")
    p.add_argument("--out", type=Path, required=True, help="where the prepared dataset goes")
    p.add_argument("--splits", nargs="+", default=["train", "test"])
    args = p.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    data = json.loads(args.coco.read_text(encoding="utf-8"))
    cats: dict[int, str] = {}
    for c in data.get("categories", []):
        cls = normalise_class(c["name"])
        if cls in PAINT_ORDER:
            cats[c["id"]] = cls
        else:
            log.warning("category %r is not molten_consumable/droplet; its annotations are skipped",
                        c["name"])
    images = {im["id"]: im for im in data.get("images", [])}
    anns_of: dict[int, list] = defaultdict(list)
    for a in data.get("annotations", []):
        anns_of[a["image_id"]].append(a)

    frames = find_frames(args.frames, args.splits)
    log.info("JSON: %d images, %d annotations; frames found: %d",
             len(images), sum(map(len, anns_of.values())), len(frames))

    stats: Counter = Counter()
    per_exp: dict[tuple[str, str], dict] = {}
    for image_id, im in sorted(images.items()):
        name = Path(im["file_name"]).name
        if name not in frames:
            stats["json_image_without_frame"] += 1
            log.warning("no frame for JSON image %s", name)
            continue
        split, exp, frame_path = frames[name]
        with Image.open(frame_path) as f:
            frame = np.asarray(f.convert("RGB"))
        h, w = frame.shape[:2]
        if (im.get("width"), im.get("height")) not in {(w, h), (None, None)}:
            log.warning("%s: JSON says %sx%s, frame is %dx%d -- using the frame size",
                        name, im.get("width"), im.get("height"), w, h)
            stats["size_mismatch"] += 1

        mask = np.zeros((h, w, 3), np.uint8)  # Background
        drawn = np.zeros((h, w), bool)
        anns = [a for a in anns_of.get(image_id, []) if a["category_id"] in cats]
        anns.sort(key=lambda a: PAINT_ORDER.index(cats[a["category_id"]]))
        for a in anns:
            cls = cats[a["category_id"]]
            region = polygon_mask(a["segmentation"], h, w)
            stats["overlap_px"] += int((region & drawn & ~np.all(mask == CLASSES[cls], axis=2)).sum())
            mask[region] = CLASSES[cls]
            drawn |= region
            stats[cls] += 1

        rgb_dir = args.out / split / exp / "rgb"
        write_rgb(rgb_dir / "images" / name, frame)
        write_rgb(rgb_dir / "rgb_masks" / f"{name}_rgb_mask.png", mask)
        (rgb_dir / "rgb_masks" / RGB_MAPPING_NAME).write_text(mapping_text(), encoding="utf-8")

        sub = per_exp.setdefault((split, exp), {"images": [], "annotations": [],
                                                "categories": data.get("categories", [])})
        sub["images"].append(im)
        sub["annotations"].extend(anns_of.get(image_id, []))
        stats[f"masks_{split}"] += 1

    for (split, exp), sub in per_exp.items():
        (args.out / split / exp / f"{exp}_coco.json").write_text(json.dumps(sub), encoding="utf-8")

    annotated = {Path(im["file_name"]).name for im in images.values()}
    unlabelled = Counter(s for n, (s, _, _) in frames.items() if n not in annotated)
    log.info("written: %s", ", ".join(f"{k}={v}" for k, v in sorted(stats.items())))
    for split, n in sorted(unlabelled.items()):
        log.info("skipped %d %s frames that the JSON does not annotate", n, split)
    if stats["overlap_px"]:
        log.info("%d px where molten and droplet polygons overlap (molten kept)", stats["overlap_px"])
    log.info("done -> %s  (%d experiment folders)", args.out.resolve(), len(per_exp))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
