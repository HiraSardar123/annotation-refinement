"""Step 7 -- fill gaps between annotated classes.

A gap is background (0,0,0) sitting between labelled classes (molten,
droplet, arcs, weld pool, spatter):
  * thin gaps: background that a morphological closing of all labelled
    pixels (radius --gap-close-px) fills in, i.e. slivers up to about
    2 x radius wide between labelled regions;
  * holes: background areas completely enclosed by labelled pixels, up to
    --gap-max-hole-px.
Only gaps that touch at least two different classes, or are enclosed by a
single class, are kept -- a notch in the outline of one class that opens
onto the real background is not a gap.

Thin unlabelled slivers next to molten / droplet: background less than about
2 x --gap-close-px wide (what a morphological opening of the background
removes) that touches a molten or droplet instance. These are annotation
gaps, not real background, so they are given a real class: the same rule as
below, with background excluded ("sliver/..." in the report). Wide background
next to the molten -- e.g. around the wire -- is not a sliver and stays.

Each gap region is then filled:
  * surrounded all the way by one class  -> that class ("surround")
  * otherwise, per pixel, with step 5's rule (nearest neighbour + human
    annotation + SAM label + SAM confidence) with protect_human=True: a
    pixel the human labelled keeps its human class unless SAM is at least
    --sam-conf sure of another; molten and droplet never swap.
"""

from __future__ import annotations

from collections import Counter

import cv2
import numpy as np

from .common import (BACKGROUND, COLOR_TO_CLASS, DROPLET, MOLTEN, as_color, class_binary,
                     colors_in, dilate, disc)
from .relabel import relabel


def find_gaps(mask: np.ndarray, cfg) -> list[np.ndarray]:
    labelled = np.any(mask != 0, axis=2)
    background = ~labelled
    closed = cv2.morphologyEx(labelled.astype(np.uint8), cv2.MORPH_CLOSE,
                              disc(cfg.gap_close_px)).astype(bool)
    candidate = closed & background

    n, cc, stats, _ = cv2.connectedComponentsWithStats(background.astype(np.uint8), connectivity=8)
    h, w = background.shape
    for i in range(1, n):
        x, y, bw, bh, area = stats[i]
        touches_border = x == 0 or y == 0 or x + bw == w or y + bh == h
        if not touches_border and area <= cfg.gap_max_hole_px:
            candidate |= cc == i

    gaps = []
    n, cc = cv2.connectedComponents(candidate.astype(np.uint8), connectivity=8)
    for i in range(1, n):
        region = cc == i
        ring = dilate(region, 1) & ~region
        around = colors_in(ring, mask)
        classes = [c for c in around if c != BACKGROUND]
        enclosed = len(around) == 1 and classes
        if enclosed or len(classes) >= 2:
            gaps.append(region)
    return gaps


def find_target_slivers(mask: np.ndarray, cfg) -> np.ndarray:
    """Thin background touching molten / droplet (see module docstring)."""
    background = ~np.any(mask != 0, axis=2)
    wide = cv2.morphologyEx(background.astype(np.uint8), cv2.MORPH_OPEN,
                            disc(cfg.gap_close_px)).astype(bool)
    thin = background & ~wide
    target = class_binary(mask, MOLTEN) | class_binary(mask, DROPLET)
    n, cc = cv2.connectedComponents(thin.astype(np.uint8), connectivity=8)
    touching = np.unique(cc[thin & dilate(target, 1)])
    return np.isin(cc, touching[touching > 0])


def fill_gaps(mask: np.ndarray, human: np.ndarray, predictor, cfg
              ) -> tuple[np.ndarray, Counter, int]:
    """(filled mask, per class:reason pixel counts, gap pixels found)."""
    gaps = find_gaps(mask, cfg)
    out = mask.copy()
    stats: Counter = Counter()
    unknown = np.zeros(mask.shape[:2], bool)
    for region in gaps:
        ring = dilate(region, 1) & ~region
        around = colors_in(ring, mask)
        if len(around) == 1 and around[0] != BACKGROUND:
            out[region] = around[0]
            stats[f"{COLOR_TO_CLASS[as_color(around[0])]}:surround"] += int(region.sum())
        else:
            unknown |= region
    if unknown.any():
        out, decided = relabel(out, unknown, human, predictor, cfg, protect_human=True)
        stats.update(decided)
    slivers = find_target_slivers(out, cfg)
    if slivers.any():
        out, decided = relabel(out, slivers, human, predictor, cfg, protect_human=True,
                               exclude=(BACKGROUND,))
        stats.update({f"sliver/{k}": v for k, v in decided.items()})
    return out, stats, int(sum(r.sum() for r in gaps)) + int(slivers.sum())


if __name__ == "__main__":
    from .step_cli import run_single
    raise SystemExit(run_single(7))
