"""Run the pipeline up to one step on a single mask and save that step's output.

Every step module calls this from its `__main__`, e.g.

    .\\.venv\\Scripts\\python.exe -m pipeline.boxes <mask.png> [options]

Outputs go to <out-root>/single/<mask stem>_<step>.png (default
<repo>/outputs/single). Steps 1-3 do not need SAM; 4 onwards load it.
Every main.py option is accepted (e.g. --pad-molten 5, --sam-conf 0.9).
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

import cv2
import numpy as np

from . import config
from .common import CLASS_COLORS, TARGET_CLASSES, discover_folder, read_mask, read_rgb, write_rgb

STEP_NAMES = {1: "segregate", 2: "boxes", 3: "prompts", 4: "sam_predict", 5: "relabel",
              7: "gaps", 9: "spatter"}


def run_single(step: int) -> int:
    argv = sys.argv[1:]
    if not argv or argv[0].startswith("-"):
        print(f"usage: python -m pipeline.{STEP_NAMES[step]} MASK_PNG [options]")
        return 2
    mask_path = Path(argv[0]).resolve()
    cfg = config.parse(argv[1:])
    logging.basicConfig(level=getattr(logging, cfg.log_level), format="%(levelname)s: %(message)s")
    job = next((j for j in discover_folder(mask_path.parent, cfg.splits, None)
                if j.mask_path.resolve() == mask_path), None)
    if job is None:
        print(f"could not pair {mask_path} with a source image")
        return 1
    out = cfg.out_root.resolve() / "single"
    name = f"{job.stem}_{step}_{STEP_NAMES[step]}.png"

    from .boxes import build_boxes
    from .prompts import component_prompt
    from .segregate import segregate
    from .visualize import boxes_image, comparison, points_image

    human = read_mask(job.mask_path)
    image = read_rgb(job.image_path)
    if image.shape[:2] != human.shape[:2]:
        image = cv2.resize(image, (human.shape[1], human.shape[0]), interpolation=cv2.INTER_AREA)

    seg = segregate(human)
    if step == 1:
        view = np.zeros_like(human)
        for cls in TARGET_CLASSES:
            view[seg.binary[cls]] = CLASS_COLORS[cls]
            print(f"{cls}: {len(seg.components[cls])} components, "
                  f"sizes {[int(c.sum()) for c in seg.components[cls]]}")
        write_rgb(out / name, view)
        print(f"-> {out / name}")
        return 0

    comps = build_boxes(seg, cfg)
    if step == 2:
        for c in comps:
            print(f"{c.cls}: box {c.box.tolist()} (pad {c.pad}px)")
        write_rgb(out / name, boxes_image(image, human, comps, cfg))
        print(f"-> {out / name}")
        return 0

    prompts = [component_prompt(c, human, cfg) for c in comps]
    if step == 3:
        for c, p in zip(comps, prompts):
            print(f"{c.cls}: positive {p.points[p.labels == 1].astype(int).tolist()}  "
                  f"negative {p.points[p.labels == 0].astype(int).tolist()}")
        write_rgb(out / name, points_image(image, human, comps, prompts, cfg))
        print(f"-> {out / name}")
        return 0

    from .gaps import fill_gaps
    from .relabel import assemble, relabel
    from .sam_predict import load_predictor, predict_component
    from .spatter import remove_static_spatter, static_map

    predictor = load_predictor(cfg)
    predictor.set_image(image)
    predictions = [predict_component(predictor, c, p) for c, p in zip(comps, prompts)]
    mask, unknown = assemble(seg, predictions)
    if step == 4:
        for p in predictions:
            print(f"{p.cls}: score {p.score:.3f} accepted {p.accepted} px {int(p.mask.sum())}")
        print(f"pixels left blank for step 5: {int(unknown.sum())}")
        write_rgb(out / name, comparison(image, human, mask, "SAM (step 4, blanks unlabelled)", cfg))
        print(f"-> {out / name}")
        return 0

    from .instances import consolidate
    from .molten_top import enforce_molten_top
    mask, stats = relabel(mask, unknown, human, predictor, cfg, mode="target")
    mask, _ = consolidate(mask, seg, cfg)
    mask, _ = enforce_molten_top(mask, cfg, human)
    if step == 5:
        print("blank pixels relabelled as:", dict(stats))
        write_rgb(out / name, comparison(image, human, mask, "SAM REFINED (step 6)", cfg))
        print(f"-> {out / name}")
        return 0

    mask, stats, gap_px = fill_gaps(mask, human, predictor, cfg)
    mask, _ = consolidate(mask, seg, cfg)
    mask, _ = enforce_molten_top(mask, cfg, human)
    if step == 7:
        print(f"gap pixels: {gap_px}; filled as:", dict(stats))
        write_rgb(out / name, comparison(image, human, mask, "GAPS REFINED (step 8)", cfg))
        print(f"-> {out / name}")
        return 0

    static = static_map(job.mask_dir, cfg.spatter_max_missing, cfg.spatter_min_masks)
    mask, removed, stats = remove_static_spatter(mask, human, static, predictor, cfg)
    mask, _ = consolidate(mask, seg, cfg)
    mask, _ = enforce_molten_top(mask, cfg, human)
    print(f"static spatter removed: {int(removed.sum())} px; relabelled as:", dict(stats))
    write_rgb(out / name, comparison(image, human, mask, "SPATTER REFINED (step 10)", cfg))
    print(f"-> {out / name}")
    return 0
