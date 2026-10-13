"""Step 12 -- run the whole pipeline.

Per image:
  1  segregate       human mask -> molten / droplet components
  2  boxes           one box per component (molten +4 px, droplet +1 px)
  3  prompts         2 positive + 3 negative points per component
  4  sam_predict     SAM2 prediction per component
  5  relabel         label the pixels the refinement left blank
     instances       one solid mask per molten / droplet instance (after 5, 7, 9)
     molten_top      nothing but molten/background above the molten (after 5, 7, 9)
  6  save            <out-root>/refined/            + refined_comp/
  7  gaps            fill gaps between classes
  8  save            <out-root>/gaps_refined/       + gaps_refined_comp/
  9  spatter         remove static (noise) spatter
  10 save            <out-root>/spatter_refined/    + spatter_refined_comp/
then
  11 evaluate        <out-root>/evaluation.csv

Box / point images (--visualize) go to <out-root>/debug/{boxes,points}/.
Run with run_dataset.py (whole dataset) or run_folder.py (one folder), or
directly:  python -m pipeline.main [--folder PATH] [options]
"""

from __future__ import annotations

import json
import logging
import time
from collections import Counter
from pathlib import Path

import cv2
import numpy as np

from . import config
from .boxes import build_boxes
from .common import Job, discover_dataset, discover_folder, read_mask, read_rgb, write_rgb
from .evaluate import evaluate, stage_path
from .gaps import fill_gaps
from .instances import consolidate
from .molten_top import enforce_molten_top
from .prompts import component_prompt
from .relabel import assemble, relabel
from .sam_predict import load_predictor, predict_component
from .segregate import segregate
from .spatter import remove_static_spatter, static_map
from .visualize import boxes_image, comparison, points_image

log = logging.getLogger("pipeline")


def save_stage(out_root: Path, stage: str, title: str, job: Job, image: np.ndarray,
               human: np.ndarray, mask: np.ndarray, cfg) -> None:
    if cfg.final_only and stage != "spatter_refined":
        return
    write_rgb(stage_path(out_root, stage, job.split, job.experiment, job.mask_path.name), mask)
    if cfg.comp:
        write_rgb(out_root / f"{stage}_comp" / job.split / job.experiment / f"{job.stem}.png",
                  comparison(image, human, mask, title, cfg))


def process(job: Job, predictor, cfg, out_root: Path) -> dict:
    started = time.perf_counter()
    human = read_mask(job.mask_path)
    image = read_rgb(job.image_path)
    if image.shape[:2] != human.shape[:2]:
        image = cv2.resize(image, (human.shape[1], human.shape[0]), interpolation=cv2.INTER_AREA)
    predictor.set_image(image)

    # 1-3: components, boxes, prompts
    seg = segregate(human)
    comps = build_boxes(seg, cfg)
    prompts = [component_prompt(c, human, cfg) for c in comps]
    if cfg.visualize:
        debug = out_root / "debug"
        write_rgb(debug / "boxes" / job.split / job.experiment / f"{job.stem}.png",
                  boxes_image(image, human, comps, cfg))
        write_rgb(debug / "points" / job.split / job.experiment / f"{job.stem}.png",
                  points_image(image, human, comps, prompts, cfg))

    # 4: SAM
    predictions = [predict_component(predictor, c, p) for c, p in zip(comps, prompts)]

    # 5-6: relabel what the refinement left blank, save
    refined, unknown = assemble(seg, predictions)
    refined, step5 = relabel(refined, unknown, human, predictor, cfg, mode="target")
    refined, inst5 = consolidate(refined, seg, cfg)
    refined, top5 = enforce_molten_top(refined, cfg, human)
    save_stage(out_root, "refined", "SAM REFINED (step 6)", job, image, human, refined, cfg)

    # 7-8: gaps between classes, save
    gapped, step7, gap_px = fill_gaps(refined, human, predictor, cfg)
    gapped, inst7 = consolidate(gapped, seg, cfg)
    gapped, top7 = enforce_molten_top(gapped, cfg, human)
    save_stage(out_root, "gaps_refined", "GAPS REFINED (step 8)", job, image, human, gapped, cfg)

    # 9-10: static spatter, save
    static = static_map(job.mask_dir, cfg.spatter_max_missing, cfg.spatter_min_masks)
    cleaned, removed, step9 = remove_static_spatter(gapped, human, static, predictor, cfg)
    cleaned, inst9 = consolidate(cleaned, seg, cfg)
    cleaned, top9 = enforce_molten_top(cleaned, cfg, human)
    save_stage(out_root, "spatter_refined", "SPATTER REFINED (step 10)", job, image, human,
               cleaned, cfg)

    return {
        "split": job.split, "experiment": job.experiment, "mask": job.mask_path.name,
        "components": len(comps), "sam_accepted": sum(p.accepted for p in predictions),
        "step5_unknown_px": int(unknown.sum()), "step5": dict(step5),
        "step7_gap_px": gap_px, "step7": dict(step7),
        "step9_spatter_removed_px": int(removed.sum()), "step9": dict(step9),
        "molten_top_cleared_px": top5 + top7 + top9,
        "instances": dict(inst5 + inst7 + inst9),
        "folder_corrupted": bool(static is not None and static.any()),
        "seconds": round(time.perf_counter() - started, 2),
    }


def main(argv: list[str] | None = None) -> int:
    cfg = config.parse(argv)
    logging.basicConfig(level=getattr(logging, cfg.log_level), format="%(levelname)s: %(message)s")
    root = cfg.root.resolve()
    out_root = cfg.out_root.resolve()

    if cfg.folder is not None:
        jobs = discover_folder(cfg.folder, cfg.splits, cfg.pilot)
    else:
        jobs = discover_dataset(root, cfg.splits, cfg.pilot, cfg.limit_dirs)
    log.info("visualisation %s, comparisons %s, output -> %s",
             "on" if cfg.visualize else "off", "on" if cfg.comp else "off", out_root)
    if not jobs:
        log.error("nothing to do")
        return 1
    if cfg.dry_run:
        for job in jobs[:20]:
            log.info("  %s/%s  %s  <-  %s", job.split, job.experiment, job.mask_path.name,
                     job.image_path.name)
        log.info("dry run: %d masks would be processed", len(jobs))
        return 0
    if not cfg.checkpoint.exists():
        log.error("checkpoint not found: %s", cfg.checkpoint)
        return 2

    predictor = load_predictor(cfg)
    manifest_path = out_root / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8")) if manifest_path.exists() else {}
    report_path = out_root / "run_report.json"
    records, corrupted = [], set()
    started = time.perf_counter()
    for index, job in enumerate(jobs, 1):
        final = stage_path(out_root, "spatter_refined", job.split, job.experiment, job.mask_path.name)
        key = f"{job.split}/{job.experiment}/{job.mask_path.name}"
        if final.exists() and not cfg.overwrite:
            manifest.setdefault(key, {"split": job.split, "experiment": job.experiment,
                                      "mask": job.mask_path.name, "human": str(job.mask_path)})
            continue
        try:
            record = process(job, predictor, cfg, out_root)
        except Exception:
            log.exception("failed on %s", job.mask_path)
            continue
        records.append(record)
        manifest[key] = {"split": job.split, "experiment": job.experiment,
                         "mask": job.mask_path.name, "human": str(job.mask_path)}
        if record["folder_corrupted"]:
            corrupted.add(str(job.mask_dir))
        log.info("[%d/%d] %s  comp=%d sam=%d blank=%dpx gaps=%dpx spatter-=%dpx "
                 "above-molten=%dpx  %.1fs",
                 index, len(jobs), job.mask_path.name, record["components"],
                 record["sam_accepted"], record["step5_unknown_px"], record["step7_gap_px"],
                 record["step9_spatter_removed_px"], record["molten_top_cleared_px"],
                 record["seconds"])
        if index % 25 == 0:  # keep progress if the run is interrupted
            out_root.mkdir(parents=True, exist_ok=True)
            manifest_path.write_text(json.dumps(manifest, indent=1), encoding="utf-8")

    out_root.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest, indent=1), encoding="utf-8")
    totals: dict[str, Counter] = {k: Counter() for k in ("step5", "step7", "step9")}
    for r in records:
        for k in totals:
            totals[k].update(r[k])
    report_path.write_text(json.dumps({
        "settings": {k: str(v) for k, v in vars(cfg).items()},
        "images": len(records), "seconds": round(time.perf_counter() - started, 1),
        "corrupted_folders": sorted(corrupted),
        "pixels_by_class_and_reason": {k: dict(v) for k, v in totals.items()},
        "files": records}, indent=2), encoding="utf-8")
    log.info("done: %d images in %.0fs -> %s", len(records), time.perf_counter() - started, out_root)
    for folder in sorted(corrupted):
        log.info("corrupted (static spatter): %s", folder)

    if not cfg.skip_eval:
        evaluate(out_root, simple=cfg.final_only)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
