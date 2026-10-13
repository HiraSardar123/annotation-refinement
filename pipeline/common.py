"""Shared constants and helpers: palette, image IO, morphology, discovery."""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

REPO_ROOT = Path(__file__).resolve().parent.parent
log = logging.getLogger("pipeline")

# Canonical palette, from class_rgb_mapping.txt. Identity is the RGB triple,
# never the label spelling (folders disagree on "spatter"/"Spatter").
CLASS_COLORS: dict[str, tuple[int, int, int]] = {
    "background": (0, 0, 0),
    "molten_consumable": (255, 128, 0),
    "droplet": (255, 0, 0),
    "external_arc": (0, 0, 255),
    "internal_arc": (255, 255, 0),
    "spatter": (0, 255, 255),
    "weld_pool": (255, 0, 255),
}
COLOR_TO_CLASS = {v: k for k, v in CLASS_COLORS.items()}
BACKGROUND = CLASS_COLORS["background"]
MOLTEN = CLASS_COLORS["molten_consumable"]
DROPLET = CLASS_COLORS["droplet"]
SPATTER = CLASS_COLORS["spatter"]
TARGET_CLASSES = ("molten_consumable", "droplet")

# A pixel the human gave one of these classes may never end up as the other.
FORBIDDEN = {MOLTEN: DROPLET, DROPLET: MOLTEN}

RGB_MAPPING_NAME = "class_rgb_mapping.txt"
IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}
MASK_SUFFIXES = ("_rgb_mask.png", "_mask.png", "_segmentation.png")
OUTPUT_DIR_NAMES = {"refined", "overlays", ".project_backups", "outputs"}

Color = tuple[int, int, int]


# --------------------------------------------------------------------------
# IO and pixel helpers
# --------------------------------------------------------------------------
def read_rgb(path: Path) -> np.ndarray:
    return np.asarray(Image.open(path).convert("RGB"))


def write_rgb(path: Path, array: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(np.ascontiguousarray(array, dtype=np.uint8), mode="RGB").save(path)


def normalise_class(name: str) -> str | None:
    """Map the many label spellings ("Molten Consumable", "weilding pool",
    " External arc", "splatter", ...) to the canonical class names."""
    key = re.sub(r"[^a-z]", "", name.lower())
    return {"background": "background", "moltenconsumable": "molten_consumable",
            "droplet": "droplet", "externalarc": "external_arc", "internalarc": "internal_arc",
            "weldpool": "weld_pool", "weildingpool": "weld_pool", "weldingpool": "weld_pool",
            "spatter": "spatter", "splatter": "spatter"}.get(key)


@lru_cache(maxsize=128)
def folder_palette(mask_dir: Path) -> tuple[tuple[Color, Color], ...]:
    """(folder colour, canonical colour) pairs where a folder's
    class_rgb_mapping.txt uses a non-standard colour for a class -- e.g.
    molten (255, 125, 0) instead of (255, 128, 0)."""
    f = mask_dir / RGB_MAPPING_NAME
    if not f.is_file():
        return ()
    pairs = []
    for line in f.read_text(encoding="utf-8", errors="replace").splitlines():
        if ":" not in line:
            continue
        name, rgb = line.split(":", 1)
        nums = [int(x) for x in re.findall(r"\d+", rgb)]
        cls = normalise_class(name)
        if cls in CLASS_COLORS and len(nums) == 3 and tuple(nums) != CLASS_COLORS[cls]:
            pairs.append((tuple(nums), CLASS_COLORS[cls]))
    return tuple(pairs)


def folder_colors(mask_dir: Path) -> dict[str, Color]:
    """class -> colour as this folder's class_rgb_mapping.txt defines it
    (standard colours for classes the file does not list)."""
    colors = dict(CLASS_COLORS)
    for src, dst in folder_palette(mask_dir.resolve()):
        cls = next(c for c, v in CLASS_COLORS.items() if v == dst)
        colors[cls] = src
    return colors


def read_mask(path: Path, mapping_dir: Path | None = None) -> np.ndarray:
    """A human RGB mask with its folder's colours translated to the canonical
    palette (using class_rgb_mapping.txt in `mapping_dir`, default the mask's
    own folder). Always use this -- never read_rgb -- for human masks."""
    rgb = read_rgb(path)
    pairs = folder_palette((mapping_dir or path.parent).resolve())
    if not pairs:
        return rgb
    out = rgb.copy()
    for src, dst in pairs:
        out[class_binary(rgb, src)] = dst
    return out


def class_binary(mask: np.ndarray, color: Color) -> np.ndarray:
    return np.all(mask == np.array(color, dtype=mask.dtype), axis=2)


def as_color(value) -> Color:
    return tuple(int(v) for v in value)


def colors_in(region: np.ndarray, mask: np.ndarray) -> list[Color]:
    """Known palette colours present in `mask` under `region`."""
    if not region.any():
        return []
    found = np.unique(mask[region].reshape(-1, 3), axis=0)
    return [as_color(c) for c in found if as_color(c) in COLOR_TO_CLASS]


def disc(radius: int) -> np.ndarray:
    k = max(1, 2 * radius + 1)
    return cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))


def dilate(binary: np.ndarray, radius: int) -> np.ndarray:
    if radius <= 0:
        return binary.copy()
    return cv2.dilate(binary.astype(np.uint8), disc(radius)).astype(bool)


def components(binary: np.ndarray) -> list[np.ndarray]:
    """8-connected components, every size."""
    n, labels = cv2.connectedComponents(binary.astype(np.uint8), connectivity=8)
    return [labels == i for i in range(1, n)]


def distance_to(binary: np.ndarray) -> np.ndarray:
    """Per-pixel distance to the nearest True pixel of `binary`."""
    return cv2.distanceTransform((~binary).astype(np.uint8), cv2.DIST_L2, 5)


def largest_piece(binary: np.ndarray) -> np.ndarray:
    n, labels, stats, _ = cv2.connectedComponentsWithStats(binary.astype(np.uint8), connectivity=8)
    if n <= 2:
        return binary
    return labels == 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))


def box_mask(box, shape: tuple[int, int]) -> np.ndarray:
    x0, y0, x1, y1 = (int(v) for v in box)
    out = np.zeros(shape, dtype=bool)
    out[y0:y1 + 1, x0:x1 + 1] = True
    return out


def padded_box(region: np.ndarray, pad: int) -> np.ndarray:
    """Tight bbox of `region` plus `pad` px on every side, clipped to the image."""
    ys, xs = np.nonzero(region)
    h, w = region.shape
    return np.array([max(0, xs.min() - pad), max(0, ys.min() - pad),
                     min(w - 1, xs.max() + pad), min(h - 1, ys.max() + pad)], dtype=np.int32)


# --------------------------------------------------------------------------
# discovery
# --------------------------------------------------------------------------
@dataclass
class Job:
    mask_path: Path
    image_path: Path
    split: str
    experiment: str
    mask_dir: Path

    @property
    def stem(self) -> str:
        return image_stem_for_mask(self.mask_path)


def is_mask_file(p: Path) -> bool:
    return p.name.endswith(MASK_SUFFIXES) or p.parent.name.lower() in {"rgb_masks", "rgb-masks", "masks"}


def image_stem_for_mask(mask_path: Path) -> str:
    for suffix in MASK_SUFFIXES:
        if mask_path.name.endswith(suffix):
            return mask_path.name[: -len(suffix)]
    return mask_path.stem


def split_of(path: Path, splits: list[str]) -> str:
    for part in path.parts:
        if part in splits:
            return part
    return "folder"


def experiment_root(mask_dir: Path, splits: list[str], fallback: Path) -> Path:
    """The per-experiment folder: a direct child of train/train or test/test."""
    for parent in mask_dir.parents:
        if parent.parent.name in splits:
            return parent
    return fallback


def index_images(root: Path) -> dict[str, list[Path]]:
    index: dict[str, list[Path]] = {}
    for p in root.rglob("*"):
        if not p.is_file() or p.suffix.lower() not in IMAGE_SUFFIXES or is_mask_file(p):
            continue
        if {s.lower() for s in p.parts} & OUTPUT_DIR_NAMES:
            continue
        index.setdefault(p.stem, []).append(p)
        index.setdefault(p.name, []).append(p)
    return index


def _is_output(p: Path) -> bool:
    return bool({s.lower() for s in p.parts} & OUTPUT_DIR_NAMES)


def jobs_for_mask_dirs(mask_dirs: list[Path], splits: list[str], pilot: int | None,
                       fallback_root) -> list[Job]:
    jobs, unpaired = [], 0
    for mask_dir in mask_dirs:
        exp_root = experiment_root(mask_dir, splits, fallback_root(mask_dir))
        images = index_images(exp_root)
        masks = sorted(mask_dir.glob("*.png"))
        if pilot:
            masks = masks[:pilot]
        for mask_path in masks:
            stem = image_stem_for_mask(mask_path)
            candidates = [c for c in (images.get(stem) or images.get(f"{stem}.png") or [])
                          if c != mask_path]
            if not candidates:
                unpaired += 1
                log.debug("no source image for %s", mask_path.name)
                continue
            candidates.sort(key=lambda c: (c.parent.name.lower() != "images", len(c.parts)))
            jobs.append(Job(mask_path, candidates[0], split_of(mask_path, splits),
                            exp_root.name, mask_dir))
    log.info("discovered %d mask dirs, %d paired masks, %d unpaired",
             len(mask_dirs), len(jobs), unpaired)
    return jobs


def discover_dataset(root: Path, splits: list[str], pilot: int | None,
                     limit_dirs: int | None) -> list[Job]:
    mask_dirs = sorted(p.parent for p in root.rglob(RGB_MAPPING_NAME) if not _is_output(p))
    mask_dirs = [d for d in mask_dirs if split_of(d, splits) in splits]
    if limit_dirs:
        mask_dirs = mask_dirs[:limit_dirs]
    return jobs_for_mask_dirs(mask_dirs, splits, pilot, lambda d: d.parent)


def discover_folder(folder: Path, splits: list[str], pilot: int | None) -> list[Job]:
    """Every mask directory under `folder` (or `folder` itself if it holds masks)."""
    folder = folder.resolve()
    mask_dirs = sorted(p.parent for p in folder.rglob(RGB_MAPPING_NAME) if not _is_output(p))
    if not mask_dirs and any(folder.glob("*_rgb_mask.png")):
        mask_dirs = [folder]
    return jobs_for_mask_dirs(mask_dirs, splits, pilot,
                              lambda d: folder if d != folder else folder.parent)
