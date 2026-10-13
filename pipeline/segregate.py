"""Step 1 -- split the human mask into molten_consumable and droplet regions.

Each class is taken on its own colour, so a molten component and a droplet
component can never be merged, even where they touch. The result is passed
to step 2.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .common import CLASS_COLORS, TARGET_CLASSES, class_binary, components


@dataclass
class Segregation:
    human: np.ndarray                                   # the human RGB mask
    binary: dict[str, np.ndarray] = field(default_factory=dict)       # class -> all its pixels
    components: dict[str, list[np.ndarray]] = field(default_factory=dict)  # class -> pieces

    @property
    def target(self) -> np.ndarray:
        """Every pixel the human labelled molten or droplet."""
        out = np.zeros(self.human.shape[:2], dtype=bool)
        for cls in TARGET_CLASSES:
            out |= self.binary[cls]
        return out


def segregate(human: np.ndarray) -> Segregation:
    seg = Segregation(human=human)
    for cls in TARGET_CLASSES:
        binary = class_binary(human, CLASS_COLORS[cls])
        seg.binary[cls] = binary
        seg.components[cls] = components(binary)
    return seg


if __name__ == "__main__":
    from .step_cli import run_single
    raise SystemExit(run_single(1))
