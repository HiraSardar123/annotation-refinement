"""The top of the molten consumable.

Two rules, applied after steps 5, 7 and 9 (after the instance clean-up), so
no later step can undo them:

1. Keep the human top outline. The wire above the molten is dark and the top
   of the molten is dim, so SAM tends to grow the molten up into the wire;
   the human outline is the better one there. For every molten instance of
   the human mask, in the band from the top of its box down to
   --molten-top-keep-px rows below its highest pixel (full box width),
   molten is exactly the human molten: molten grown outside the human
   outline goes back to the human label, human molten that was trimmed
   comes back. Below the band SAM's refinement stands.

2. Nothing but molten or background on top of the molten. The molten hangs
   from the wire, so the strip directly above its tip -- the --pad-molten
   rows above the molten's highest pixel, over the columns the tip occupies
   (columns of molten in its top --pad-molten rows, widened by 1 px) -- may
   only be molten or background; any other class there becomes background.
   The rest of the box, e.g. external arc beside a wide molten, is untouched.
"""

from __future__ import annotations

import numpy as np

from .common import BACKGROUND, MOLTEN, class_binary, components, padded_box


def keep_human_top(mask: np.ndarray, human: np.ndarray, cfg) -> tuple[np.ndarray, int]:
    """Rule 1. (mask, pixels changed)."""
    keep = getattr(cfg, "molten_top_keep_px", 10)
    if keep <= 0:
        return mask, 0
    out = mask.copy()
    changed = 0
    for inst in components(class_binary(human, MOLTEN)):
        x0, y0, x1, y1 = (int(v) for v in padded_box(inst, cfg.pad_molten))
        top = int(np.nonzero(inst)[0].min())
        yb = min(y1, top + keep)
        view = out[y0:yb + 1, x0:x1 + 1]
        hum = human[y0:yb + 1, x0:x1 + 1]
        inst_band = inst[y0:yb + 1, x0:x1 + 1]
        now = class_binary(view, MOLTEN)
        grown = now & ~inst_band & ~class_binary(hum, MOLTEN)
        lost = inst_band & ~now
        view[grown] = hum[grown]
        view[lost] = MOLTEN
        changed += int(grown.sum() + lost.sum())
    return out, changed


def clear_above_tip(mask: np.ndarray, cfg) -> tuple[np.ndarray, int]:
    """Rule 2. (mask, pixels changed to background)."""
    molten = class_binary(mask, MOLTEN)
    if not molten.any():
        return mask, 0
    out = mask.copy()
    changed = 0
    for comp in components(molten):
        x0, y0, x1, y1 = (int(v) for v in padded_box(comp, cfg.pad_molten))
        ys, xs = np.nonzero(comp)
        top = int(ys.min())
        if top <= y0:
            continue
        tip = xs[ys < top + max(1, cfg.pad_molten)]
        cx0, cx1 = max(x0, int(tip.min()) - 1), min(x1, int(tip.max()) + 1)
        strip = out[y0:top, cx0:cx1 + 1]
        bad = np.any(strip != 0, axis=2) & ~class_binary(strip, MOLTEN)
        if bad.any():
            strip[bad] = BACKGROUND
            changed += int(bad.sum())
    return out, changed


def enforce_molten_top(mask: np.ndarray, cfg, human: np.ndarray | None = None
                       ) -> tuple[np.ndarray, int]:
    """Both rules (rule 1 only when the human mask is given). (mask, pixels changed)."""
    kept = 0
    if human is not None:
        mask, kept = keep_human_top(mask, human, cfg)
    mask, cleared = clear_above_tip(mask, cfg)
    return mask, kept + cleared


if __name__ == "__main__":
    import sys
    from pathlib import Path

    from . import config
    from .common import read_mask, write_rgb

    if len(sys.argv) < 2:
        sys.exit("usage: python -m pipeline.molten_top MASK_PNG [options]")
    cfg = config.parse(sys.argv[2:])
    mask = read_mask(Path(sys.argv[1]))
    fixed, n = enforce_molten_top(mask, cfg)
    out = cfg.out_root.resolve() / "single" / f"{Path(sys.argv[1]).stem}_molten_top.png"
    write_rgb(out, fixed)
    print(f"{n} px on top of the molten changed to background -> {out}")
