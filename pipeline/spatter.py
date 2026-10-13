"""Step 9 -- remove static spatter (noise).

Real spatter flies. When "spatter" sits on the same pixels in all, or all but
a few (--spatter-max-missing, default 4), of the masks in a folder, it is a
lens speck / sensor noise that got annotated, and the folder is reported as
corrupted.

1. For each mask folder the number of masks in which each pixel is spatter
   is counted over ALL masks of the folder (independent of --pilot). Pixels
   with count >= N - --spatter-max-missing form the static map. Folders with
   fewer than --spatter-min-masks masks are not judged.
2. In each frame, a spatter blob is removed when at least --spatter-overlap
   (50%) of it lies on the static map (dilated by 1 px for jitter). Moving
   spatter is kept.
3. The removed pixels are relabelled with step 5's rule; spatter itself is
   excluded as a candidate and the human's "spatter" label is ignored.
"""

from __future__ import annotations

import logging
from collections import Counter
from functools import lru_cache
from pathlib import Path

import cv2
import numpy as np

from .common import SPATTER, class_binary, dilate, read_mask
from .relabel import relabel

log = logging.getLogger("pipeline")


@lru_cache(maxsize=8)
def static_map(mask_dir: Path, max_missing: int, min_masks: int) -> np.ndarray | None:
    masks = sorted(mask_dir.glob("*.png"))
    if len(masks) < min_masks:
        return None
    count = None
    for path in masks:
        spatter = class_binary(read_mask(path), SPATTER)
        if count is None:
            count = np.zeros(spatter.shape, np.int32)
        if spatter.shape == count.shape:
            count += spatter
    static = count >= len(masks) - max_missing
    if static.any():
        log.warning("folder %s looks corrupted: %d px of spatter present in at least %d of %d masks",
                    mask_dir, int(static.sum()), len(masks) - max_missing, len(masks))
    return static


def remove_static_spatter(mask: np.ndarray, human: np.ndarray, static: np.ndarray | None,
                          predictor, cfg) -> tuple[np.ndarray, np.ndarray, Counter]:
    """(cleaned mask, removed pixels, per class:reason pixel counts)."""
    removed = np.zeros(mask.shape[:2], bool)
    spatter = class_binary(mask, SPATTER)
    if static is None or static.shape != removed.shape or not static.any() or not spatter.any():
        return mask, removed, Counter()
    near_static = dilate(static, 1)
    n, cc, stats, _ = cv2.connectedComponentsWithStats(spatter.astype(np.uint8), connectivity=8)
    for i in range(1, n):
        blob = cc == i
        if (blob & near_static).sum() >= cfg.spatter_overlap * stats[i, cv2.CC_STAT_AREA]:
            removed |= blob
    if not removed.any():
        return mask, removed, Counter()
    cleaned, decided = relabel(mask, removed, human, predictor, cfg, exclude=(SPATTER,))
    return cleaned, removed, decided


if __name__ == "__main__":
    from .step_cli import run_single
    raise SystemExit(run_single(9))
