"""Step 3 -- SAM prompts: 2 positive and 3 negative points per region.

Positive (inside the region):
  1. the centroid; if the centroid falls outside the shape (curved molten
     stream), the deepest region pixel on the cross-section through it;
  2. the deepest pixel of a cross-section further along the region's long
     axis, so elongated regions are covered end to end.

Negative:
  * 2 inside the box but outside the human region, one on each side of the
    region's long axis, level with the centroid. They are taken from the
    outer band of the box (at least pad-1 px from the region), i.e. at the
    growth limit, so they do not stop SAM from using the padding. If one
    side has no room (region at the image edge), both come from the other
    side, as far apart as possible -- there are always two.
  * 1 outside the box, within --neg-outside-px beyond its edge. A pixel the
    human gave another class is preferred over background, and among those
    the one closest to the region's centroid.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .common import box_mask, distance_to


@dataclass
class Prompt:
    points: np.ndarray   # (N, 2) x, y
    labels: np.ndarray   # (N,) 1 positive, 0 negative
    box: np.ndarray      # x0, y0, x1, y1
    outside: np.ndarray  # (N,) True for the outside-box negative (visualisation only)


def principal_axis(region: np.ndarray):
    ys, xs = np.nonzero(region)
    cy, cx = ys.mean(), xs.mean()
    offsets = np.stack([xs - cx, ys - cy], axis=1)
    if len(xs) > 1:
        _, vectors = np.linalg.eigh(offsets.T @ offsets)
        axis = vectors[:, -1]
    else:
        axis = np.array([1.0, 0.0])
    return cy, cx, axis


def positive_points(region: np.ndarray, count: int = 2) -> np.ndarray:
    ys, xs = np.nonzero(region)
    if len(xs) == 0:
        return np.empty((0, 2), np.float32)
    cy, cx, axis = principal_axis(region)
    along = np.stack([xs - cx, ys - cy], axis=1) @ axis
    depth = distance_to(~region)

    def peak(position: float) -> tuple[float, float]:
        gap = np.abs(along - position)
        idx = np.flatnonzero(gap <= max(1.0, float(gap.min())))
        i = idx[int(np.argmax(depth[ys[idx], xs[idx]]))]
        return float(xs[i]), float(ys[i])

    iy, ix = int(round(cy)), int(round(cx))
    picked = [(float(ix), float(iy)) if region[iy, ix] else peak(0.0)]
    for q in ([0.8] if count == 2 else np.linspace(0.15, 0.85, max(0, count - 1))):
        if len(picked) >= count:
            break
        point = peak(float(np.quantile(along, q)))
        if point not in picked:
            picked.append(point)
    return np.array(picked, np.float32)


def negative_points(region: np.ndarray, box: np.ndarray, pad: int, mask: np.ndarray,
                    outside_px: int, avoid: np.ndarray | None = None
                    ) -> tuple[np.ndarray, np.ndarray]:
    """(points, is_outside) -- 2 inside-box negatives + 1 outside-box negative."""
    shape = region.shape
    inside = box_mask(box, shape)
    blocked = region | (avoid if avoid is not None else np.zeros(shape, bool))
    dist = distance_to(region)
    cy, cx, axis = principal_axis(region)
    normal = np.array([-axis[1], axis[0]])
    picked, outside_flags = [], []

    # two inside the box, one per side of the long axis, at the growth limit
    pool = inside & ~blocked & (dist >= max(1, pad - 1))
    if not pool.any():
        pool = inside & ~blocked
    ys, xs = np.nonzero(pool)
    if len(xs):
        rel = np.stack([xs - cx, ys - cy], axis=1)
        along, side = rel @ axis, rel @ normal
        for sign in (1, -1):
            idx = np.flatnonzero(side * sign > 0)
            if not len(idx):
                continue
            i = idx[np.lexsort((dist[ys[idx], xs[idx]], np.abs(along[idx])))[0]]
            picked.append((float(xs[i]), float(ys[i])))
            outside_flags.append(False)
        # always two: if one side had no room (region at the image edge),
        # take the pool pixel farthest from the first negative
        if len(picked) == 1 and len(xs) > 1:
            far = (xs - picked[0][0]) ** 2 + (ys - picked[0][1]) ** 2
            i = int(np.argmax(far))
            picked.append((float(xs[i]), float(ys[i])))
            outside_flags.append(False)

    # one just outside the box, preferring another labelled class
    ring = box_mask(_grow(box, outside_px, shape), shape) & ~inside & ~blocked
    ys, xs = np.nonzero(ring)
    if len(xs):
        labelled = np.any(mask[ys, xs] != 0, axis=1)
        closeness = (ys - cy) ** 2 + (xs - cx) ** 2
        i = int(np.lexsort((closeness, ~labelled))[0])
        picked.append((float(xs[i]), float(ys[i])))
        outside_flags.append(True)
    return np.array(picked, np.float32).reshape(-1, 2), np.array(outside_flags, bool)


def _grow(box: np.ndarray, px: int, shape: tuple[int, int]) -> np.ndarray:
    h, w = shape
    x0, y0, x1, y1 = (int(v) for v in box)
    return np.array([max(0, x0 - px), max(0, y0 - px), min(w - 1, x1 + px), min(h - 1, y1 + px)])


def make_prompt(region: np.ndarray, box: np.ndarray, pad: int, mask: np.ndarray, cfg,
                avoid: np.ndarray | None = None) -> Prompt:
    pos = positive_points(region, 2)
    neg, neg_outside = negative_points(region, box, pad, mask, cfg.neg_outside_px, avoid)
    points = np.concatenate([pos, neg]).astype(np.float32)
    labels = np.concatenate([np.ones(len(pos)), np.zeros(len(neg))]).astype(np.int32)
    outside = np.concatenate([np.zeros(len(pos), bool), neg_outside])
    return Prompt(points, labels, np.asarray(box, np.float32), outside)


def component_prompt(comp, human: np.ndarray, cfg) -> Prompt:
    """Prompt for one step-2 component. A negative may land on the other
    target class: a droplet pixel is a useful "not molten" example."""
    return make_prompt(comp.mask, comp.box, comp.pad, human, cfg)


if __name__ == "__main__":
    from .step_cli import run_single
    raise SystemExit(run_single(3))
