"""Step 5 -- label the pixels the refinement left blank.

`assemble` paints the step-4 predictions over the human mask (molten first,
droplet second). Pixels the human labelled molten/droplet that no prediction
covers are left blank ("unknown").

Step 5 (`relabel(..., mode="target")`): every blank pixel was molten or
droplet in the human mask and SAM now says it is not. It is labelled by SAM,
but only when both hold:
  * SAM is confident: the most likely class has probability >= --sam-conf
    (0.8). SAM2 is prompted once per candidate class around the blank region
    (classes within --relabel-ring-px); background's score is
    1 - the best class score ("SAM gives it to nothing");
  * the class surrounds the pixel: it is one of the pixel's nearest labelled
    classes -- its distance from the pixel is within --surround-tol-px
    (1.5 px) of the distance to the closest labelled pixel of any class.
    Only classes whose boundary the pixel actually sits against count.
Otherwise the pixel falls back to its human label ("human"). Example: blank
pixels between molten and internal arc, SAM says external_arc -- external arc
is further away than molten and internal arc, so the pixels stay molten.

Steps 7 and 9 (`relabel(..., mode="general")`) decide gap / spatter pixels,
first match wins:
  1. SAM confidence >= --sam-conf              -> SAM label           ("sam")
  2. nearest neighbour == human annotation     -> that class          ("agree")
  3. protect_human (step 7) and the human
     labelled the pixel (not background)       -> human annotation    ("human")
  4. otherwise                                 -> nearest neighbour   ("nn")
and the same surround check applies to every pixel: if the chosen class is
not one of the pixel's nearest classes, a pixel the human labelled molten or
droplet keeps its human label, any other pixel takes its nearest-neighbour
class ("surround").

Always: a pixel the human labelled molten can never become droplet and vice
versa. Classes in `exclude` (spatter, in step 9) are never assigned and the
human annotation is ignored for pixels that carry one.
"""

from __future__ import annotations

from collections import Counter

import cv2
import numpy as np

from .common import (BACKGROUND, CLASS_COLORS, COLOR_TO_CLASS, DROPLET, FORBIDDEN, MOLTEN,
                     class_binary, colors_in, dilate, distance_to)
from .sam_predict import Prediction, class_probabilities
from .segregate import Segregation

SURROUND_DEFAULT_TOL = 1.5


def assemble(seg: Segregation, predictions: list[Prediction]) -> tuple[np.ndarray, np.ndarray]:
    """(mask with predictions painted, unknown pixels)."""
    target = seg.target
    mask = seg.human.copy()
    mask[target] = BACKGROUND
    painted = np.zeros(target.shape, bool)
    for cls in ("molten_consumable", "droplet"):
        for pred in predictions:
            if pred.cls == cls:
                mask[pred.mask] = CLASS_COLORS[cls]
                painted |= pred.mask
    return mask, target & ~painted


# --------------------------------------------------------------------------
# evidence
# --------------------------------------------------------------------------
def class_distances(ys: np.ndarray, xs: np.ndarray, classes: list, current: np.ndarray,
                    unknown: np.ndarray) -> np.ndarray:
    """(len(classes), N) distance from each pixel to the nearest labelled pixel
    of each class (unknown pixels do not count)."""
    return np.stack([distance_to(class_binary(current, c) & ~unknown)[ys, xs] for c in classes])


def surrounds(labels: np.ndarray, classes: list, dist: np.ndarray, cfg) -> np.ndarray:
    """True where a pixel's label is one of its nearest classes: within
    --surround-tol-px of the closest labelled pixel of any class."""
    tol = getattr(cfg, "surround_tol_px", SURROUND_DEFAULT_TOL)
    near = dist <= dist.min(axis=0, keepdims=True) + tol
    ok = np.zeros(labels.shape[0], bool)
    for k, color in enumerate(classes):
        ok |= near[k] & np.all(labels == np.array(color), axis=1)
    return ok


def _candidates(region, current, unknown, cfg, exclude):
    h, w = region.shape
    radius = max(1, cfg.relabel_ring_px)
    while True:
        near = dilate(region, radius) & ~unknown
        candidates = [c for c in colors_in(near, current) if c not in exclude]
        if candidates or radius >= max(h, w):
            return candidates, near
        radius *= 2


def _forbidden(hum: np.ndarray, color) -> np.ndarray:
    bad = np.zeros(len(hum), bool)
    for human_color, never in FORBIDDEN.items():
        if color == never:
            bad |= np.all(hum == np.array(human_color), axis=1)
    return bad


def _sam(region, candidates, near, current, unknown, hum, predictor, cfg, exclude=()):
    """(SAM label, SAM confidence) per region pixel. Background is a possible
    label unless it is in `exclude`."""
    classes = [c for c in candidates if c != BACKGROUND]
    probs = class_probabilities(predictor, region, classes, current, unknown, near, cfg)
    for k, color in enumerate(classes):
        probs[k, _forbidden(hum, color)] = 0.0
    n = int(region.sum())
    if BACKGROUND in exclude:
        if not classes:
            return np.tile(np.array(BACKGROUND, np.uint8), (n, 1)), np.zeros(n, np.float32)
        labels, scores = np.array(classes, np.uint8), probs
    else:
        background = 1.0 - probs.max(axis=0) if len(classes) else np.ones(n, np.float32)
        scores = np.vstack([probs, background[None]])
        labels = np.array(classes + [BACKGROUND], np.uint8)
    return labels[np.argmax(scores, axis=0)], scores.max(axis=0)


# --------------------------------------------------------------------------
# decisions
# --------------------------------------------------------------------------
def decide_target(region, current, human, unknown, predictor, cfg):
    """Step 5: SAM label if confident and surrounding the pixel, else human."""
    ys, xs = np.nonzero(region)
    hum = human[ys, xs]
    out = hum.copy()
    reason = np.full(len(ys), "human", dtype=object)
    candidates, near = _candidates(region, current, unknown, cfg, ())
    if not candidates:
        return out, reason
    sam_label, sam_conf = _sam(region, candidates, near, current, unknown, hum, predictor, cfg)
    dist = class_distances(ys, xs, candidates, current, unknown)
    ok = (sam_conf >= cfg.sam_conf) & surrounds(sam_label, candidates, dist, cfg)
    out[ok], reason[ok] = sam_label[ok], "sam"
    return out, reason


def decide(region, current, human, unknown, predictor, cfg, protect_human=False, exclude=()):
    """Steps 7 / 9: SAM + nearest neighbour + human rule, then the surround
    check (a pixel's own human label is always allowed)."""
    ys, xs = np.nonzero(region)
    hum = human[ys, xs]
    candidates, near = _candidates(region, current, unknown, cfg, exclude)
    if not candidates:
        kept = hum.copy()
        kept[[tuple(c) in exclude for c in hum.tolist()]] = BACKGROUND
        return kept, np.full(len(ys), "kept", dtype=object)

    raw = class_distances(ys, xs, candidates, current, unknown)
    dist = raw.copy()
    for k, color in enumerate(candidates):
        dist[k, _forbidden(hum, color)] = np.inf
    nn = np.array(candidates, np.uint8)[np.argmin(dist, axis=0)]
    stranded = ~np.isfinite(dist.min(axis=0))
    nn[stranded] = hum[stranded]
    sam_label, sam_conf = _sam(region, candidates, near, current, unknown, hum, predictor, cfg,
                               exclude)

    out = nn.copy()
    reason = np.full(len(ys), "nn", dtype=object)
    human_ok = np.array([tuple(c) not in exclude for c in hum.tolist()], bool)
    agree = human_ok & np.all(nn == hum, axis=1)
    out[agree], reason[agree] = hum[agree], "agree"
    if protect_human:
        keep = human_ok & ~agree & np.any(hum != 0, axis=1)
        out[keep], reason[keep] = hum[keep], "human"
    confident = sam_conf >= cfg.sam_conf
    out[confident], reason[confident] = sam_label[confident], "sam"

    target = np.all(hum == np.array(MOLTEN), axis=1) | np.all(hum == np.array(DROPLET), axis=1)
    bad = ~surrounds(out, candidates, raw, cfg) & ~np.all(out == hum, axis=1)
    fallback = np.where(target[:, None], hum, nn)
    out[bad], reason[bad] = fallback[bad], "surround"
    return out, reason


def relabel(current: np.ndarray, unknown: np.ndarray, human: np.ndarray, predictor, cfg,
            mode: str = "general", protect_human: bool = False, exclude: tuple = ()
            ) -> tuple[np.ndarray, Counter]:
    """Decide every connected region of `unknown` against the same snapshot.
    mode="target" is step 5; mode="general" is steps 7 and 9."""
    stats: Counter = Counter()
    if not unknown.any():
        return current, stats
    snapshot = current.copy()
    out = current.copy()
    n, labels = cv2.connectedComponents(unknown.astype(np.uint8), connectivity=8)
    for i in range(1, n):
        region = labels == i
        if mode == "target":
            colors, reasons = decide_target(region, snapshot, human, unknown, predictor, cfg)
        else:
            colors, reasons = decide(region, snapshot, human, unknown, predictor, cfg,
                                     protect_human, exclude)
        out[region] = colors
        for color, why in zip(map(tuple, colors.tolist()), reasons):
            stats[f"{COLOR_TO_CLASS.get(color, str(color))}:{why}"] += 1
    return out, stats


if __name__ == "__main__":
    from .step_cli import run_single
    raise SystemExit(run_single(5))
