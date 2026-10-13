"""Step 2 -- one box per component.

Every molten and droplet component from step 1 is its own item with its own
box: the tight bounding box plus --pad-molten (4 px) or --pad-droplet (1 px)
on every side. SAM's prediction is later clipped to this box, so the padding
is the most a boundary can move outward.

Molten and droplet stay separate: each component remembers its class, and
`forbidden` gives the pixels of the *other* class that SAM may not take
(a human-molten pixel never becomes droplet, and vice versa).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .common import CLASS_COLORS, TARGET_CLASSES, padded_box
from .config import class_pad
from .segregate import Segregation


@dataclass
class Component:
    cls: str
    mask: np.ndarray        # the human component
    box: np.ndarray         # x0, y0, x1, y1 (inclusive)
    pad: int
    forbidden: np.ndarray   # human pixels of the other target class

    @property
    def color(self):
        return CLASS_COLORS[self.cls]


def build_boxes(seg: Segregation, cfg) -> list[Component]:
    out = []
    for cls in TARGET_CLASSES:
        other = [c for c in TARGET_CLASSES if c != cls][0]
        pad = class_pad(CLASS_COLORS[cls], cfg)
        for comp in seg.components[cls]:
            out.append(Component(cls, comp, padded_box(comp, pad), pad, seg.binary[other]))
    return out


if __name__ == "__main__":
    from .step_cli import run_single
    raise SystemExit(run_single(2))
