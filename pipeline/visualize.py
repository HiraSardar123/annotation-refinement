"""Debug images (boxes, points) and human | SAM comparison images.

Boundaries are 1 mask pixel wide, drawn on the photo dimmed to
--overlay-dim, then upscaled --overlay-scale times (nearest neighbour) so
they are visible without being thickened. Each class has one boundary
colour, the same on both halves of a comparison.
"""

from __future__ import annotations

import cv2
import numpy as np

from .common import CLASS_COLORS, class_binary

BOUNDARY_COLORS = {
    "molten_consumable": (0, 255, 0),
    "droplet": (255, 0, 0),
    "external_arc": (0, 140, 255),
    "internal_arc": (255, 255, 0),
    "spatter": (0, 255, 255),
    "weld_pool": (255, 0, 255),
}
SHORT = {"molten_consumable": "molten", "droplet": "droplet", "external_arc": "ext arc",
         "internal_arc": "int arc", "spatter": "spatter", "weld_pool": "weld pool"}
POSITIVE, NEGATIVE, NEGATIVE_OUT = (0, 255, 0), (255, 0, 0), (255, 160, 0)


def _base(image: np.ndarray, cfg) -> np.ndarray:
    return np.clip(image.astype(np.float32) * cfg.overlay_dim, 0, 255).astype(np.uint8)


def draw_boundaries(canvas: np.ndarray, mask: np.ndarray, cfg, classes=None) -> np.ndarray:
    canvas = canvas.copy()
    for cls, color in BOUNDARY_COLORS.items():
        if classes is not None and cls not in classes:
            continue
        binary = class_binary(mask, CLASS_COLORS[cls]).astype(np.uint8)
        if binary.any():
            contours, _ = cv2.findContours(binary, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
            cv2.drawContours(canvas, contours, -1, color, cfg.overlay_thickness, lineType=cv2.LINE_8)
    return canvas


def _finish(canvas: np.ndarray, title: str, legend: list[tuple[str, tuple]], cfg) -> np.ndarray:
    scale = max(1, cfg.overlay_scale)
    if scale > 1:
        canvas = cv2.resize(canvas, None, fx=scale, fy=scale, interpolation=cv2.INTER_NEAREST)
    font, size = cv2.FONT_HERSHEY_SIMPLEX, 0.55
    strip = np.zeros((28, canvas.shape[1], 3), np.uint8)
    x = 8
    for text, color in [(title, (255, 255, 255))] + legend:
        cv2.putText(strip, text, (x, 19), font, size, color, 1, cv2.LINE_AA)
        x += cv2.getTextSize(text, font, size, 1)[0][0] + 18
    return np.vstack([strip, canvas])


def _class_legend(*masks) -> list[tuple[str, tuple]]:
    present = [cls for cls in BOUNDARY_COLORS
               if any(class_binary(m, CLASS_COLORS[cls]).any() for m in masks)]
    return [(SHORT[c], BOUNDARY_COLORS[c]) for c in present]


def comparison(image: np.ndarray, human: np.ndarray, result: np.ndarray, right_title: str,
               cfg) -> np.ndarray:
    """Human boundaries on the left, the stage's boundaries on the right."""
    base = _base(image, cfg)
    legend = _class_legend(human, result)
    left = _finish(draw_boundaries(base, human, cfg), "HUMAN", legend, cfg)
    right = _finish(draw_boundaries(base, result, cfg), right_title, legend, cfg)
    gap = np.full((left.shape[0], 6, 3), 255, np.uint8)
    return np.hstack([left, gap, right])


def boxes_image(image: np.ndarray, human: np.ndarray, components, cfg) -> np.ndarray:
    canvas = draw_boundaries(_base(image, cfg), human, cfg, classes=("molten_consumable", "droplet"))
    for comp in components:
        x0, y0, x1, y1 = (int(v) for v in comp.box)
        cv2.rectangle(canvas, (x0, y0), (x1, y1), (255, 255, 255), 1)
    legend = [("molten", BOUNDARY_COLORS["molten_consumable"]),
              ("droplet", BOUNDARY_COLORS["droplet"]), ("box", (255, 255, 255))]
    return _finish(canvas, f"BOXES (molten +{cfg.pad_molten}px, droplet +{cfg.pad_droplet}px)",
                   legend, cfg)


def points_image(image: np.ndarray, human: np.ndarray, components, prompts, cfg) -> np.ndarray:
    canvas = draw_boundaries(_base(image, cfg), human, cfg, classes=("molten_consumable", "droplet"))
    for comp, prompt in zip(components, prompts):
        x0, y0, x1, y1 = (int(v) for v in comp.box)
        cv2.rectangle(canvas, (x0, y0), (x1, y1), (255, 255, 255), 1)
        for (x, y), label, outside in zip(prompt.points.astype(int), prompt.labels, prompt.outside):
            color = POSITIVE if label else (NEGATIVE_OUT if outside else NEGATIVE)
            cv2.circle(canvas, (int(x), int(y)), 2, color, -1)
    legend = [("positive", POSITIVE), ("negative in box", NEGATIVE),
              ("negative outside box", NEGATIVE_OUT), ("box", (255, 255, 255))]
    return _finish(canvas, "POINTS", legend, cfg)
