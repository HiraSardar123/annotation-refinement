"""One solid mask per molten / droplet instance.

Molten appears once per frame, droplets can be several, but every instance
is one region with one boundary and nothing of another class inside it --
never droplet / other class / droplet.

For each target class (molten, then droplet):
  1. main pieces: for every instance of the human mask, the connected piece
     of that class in the current mask that overlaps it most;
  2. fragments: any other piece of that class (cut off from its instance,
     e.g. a few pixels left behind when SAM trimmed the boundary). Each
     fragment pixel takes its nearest neighbouring class, never the other
     target class;
  3. each main piece is closed (radius --instance-close-px) and its holes
     filled: pixels of other classes enclosed in it, or cutting into it less
     than about 2 x radius wide, take the instance's class. Pixels that are,
     or were labelled by the human as, the other target class are never
     taken (molten never swallows a droplet and vice versa).

Applied after steps 5, 7 and 9.
"""

from __future__ import annotations

from collections import Counter

import cv2
import numpy as np

from .common import (BACKGROUND, COLOR_TO_CLASS, DROPLET, FORBIDDEN, MOLTEN, class_binary,
                     colors_in, components, dilate, disc, distance_to, padded_box)
from .segregate import Segregation


def fill_holes(binary: np.ndarray) -> np.ndarray:
    """`binary` plus every region it fully encloses."""
    padded = np.pad(binary.astype(np.uint8), 1)
    flood = padded.copy()
    cv2.floodFill(flood, None, (0, 0), 2)
    return binary | (flood[1:-1, 1:-1] == 0)


def _solid(piece: np.ndarray, radius: int) -> np.ndarray:
    x0, y0, x1, y1 = (int(v) for v in padded_box(piece, radius + 2))
    crop = piece[y0:y1 + 1, x0:x1 + 1]
    closed = cv2.morphologyEx(crop.astype(np.uint8), cv2.MORPH_CLOSE, disc(radius)).astype(bool)
    out = np.zeros_like(piece)
    out[y0:y1 + 1, x0:x1 + 1] = fill_holes(closed | crop)
    return out


def _relabel_fragment(frag: np.ndarray, mask: np.ndarray, color, exclude: set) -> np.ndarray:
    """Nearest neighbouring class for each fragment pixel."""
    ys, xs = np.nonzero(frag)
    radius = 3
    while True:
        ring = dilate(frag, radius) & ~frag
        cands = [c for c in colors_in(ring, mask) if c != color and c not in exclude]
        if cands or radius > max(mask.shape):
            break
        radius *= 2
    if not cands:
        return np.tile(np.array(BACKGROUND, np.uint8), (len(ys), 1))
    dist = np.stack([distance_to(class_binary(mask, c) & ~frag)[ys, xs] for c in cands])
    return np.array(cands, np.uint8)[np.argmin(dist, axis=0)]


def consolidate(mask: np.ndarray, seg: Segregation, cfg) -> tuple[np.ndarray, Counter]:
    out = mask.copy()
    stats: Counter = Counter()
    for cls, color in (("molten_consumable", MOLTEN), ("droplet", DROPLET)):
        other = FORBIDDEN[color]
        pieces = components(class_binary(out, color))
        if not pieces:
            continue
        mains = set()
        for inst in seg.components[cls]:
            overlap = [int((p & inst).sum()) for p in pieces]
            if max(overlap) > 0:
                mains.add(int(np.argmax(overlap)))
        for i, piece in enumerate(pieces):
            if i in mains:
                continue
            out[piece] = _relabel_fragment(piece, out, color, {other})
            stats[f"{cls}:fragment_removed"] += int(piece.sum())
        protected = class_binary(out, other) | class_binary(seg.human, other)
        for i in mains:
            add = _solid(pieces[i], cfg.instance_close_px) & ~pieces[i] & ~protected
            if add.any():
                for c in colors_in(add, out):
                    stats[f"{cls}:filled_{COLOR_TO_CLASS[c]}"] += int((add & class_binary(out, c)).sum())
                out[add] = color
    return out, stats


if __name__ == "__main__":
    import sys
    from pathlib import Path

    from . import config
    from .common import read_mask, write_rgb
    from .segregate import segregate

    if len(sys.argv) < 2:
        sys.exit("usage: python -m pipeline.instances MASK_PNG [options]   (on the human mask)")
    cfg = config.parse(sys.argv[2:])
    human = read_mask(Path(sys.argv[1]))
    fixed, stats = consolidate(human, segregate(human), cfg)
    out = cfg.out_root.resolve() / "single" / f"{Path(sys.argv[1]).stem}_instances.png"
    write_rgb(out, fixed)
    print(dict(stats), "->", out)
