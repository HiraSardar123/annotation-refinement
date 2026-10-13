"""Step 11 -- per-image, per-class pixel counts for every stage, in one file.

For every image processed by main.py (listed in <out-root>/manifest.json):

  <class>_human, <class>_step6, <class>_step8, <class>_step10
      pixel count of the class in the human mask, refined/ (step 6),
      gaps_refined/ (step 8) and spatter_refined/ (step 10)
  <class>_d6, <class>_d8, <class>_d10
      that stage minus human (positive = the class grew)
  changed_step6, changed_step8, changed_step10
      pixels whose class differs from the human mask at that stage

Written to <out-root>/evaluation.csv, one row per image; a per-class total is
printed at the end.

Simple mode (--simple, and what --final-only runs): human vs the final result
only -- per image and class: human_px, refined_px, diff_px and
boundary_dist_px (average distance in px between the human and the refined
boundary), plus changed_px per image.

Run on its own:
    .\\.venv\\Scripts\\python.exe -m pipeline.evaluate --out-root .\\outputs
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
from pathlib import Path

import cv2
import numpy as np

from .common import CLASS_COLORS, REPO_ROOT, class_binary, read_mask

log = logging.getLogger("pipeline")
STAGES = (("refined", "step6"), ("gaps_refined", "step8"), ("spatter_refined", "step10"))


def stage_path(out_root: Path, stage: str, split: str, experiment: str, mask_name: str) -> Path:
    return out_root / stage / split / experiment / mask_name


def boundary_distance(a: np.ndarray, b: np.ndarray) -> float | str:
    """Average symmetric boundary distance in px: the mean distance from each
    boundary pixel of one mask to the other mask's boundary. "" if either is empty."""
    if not a.any() or not b.any():
        return ""
    def edge(m):
        return m & ~cv2.erode(m.astype(np.uint8), np.ones((3, 3), np.uint8)).astype(bool)
    ea, eb = edge(a), edge(b)
    to_b = cv2.distanceTransform((~eb).astype(np.uint8), cv2.DIST_L2, 5)[ea]
    to_a = cv2.distanceTransform((~ea).astype(np.uint8), cv2.DIST_L2, 5)[eb]
    return round(float(np.concatenate([to_b, to_a]).mean()), 2)


def evaluate_simple(out_root: Path, entries: dict) -> Path | None:
    """Human vs the final result (spatter_refined/) only."""
    rows = []
    for key in sorted(entries):
        e = entries[key]
        p = stage_path(out_root, "spatter_refined", e["split"], e["experiment"], e["mask"])
        if not p.exists():
            continue
        human, final = read_mask(Path(e["human"])), read_mask(p, Path(e["human"]).parent)
        row = {"split": e["split"], "experiment": e["experiment"], "mask": e["mask"],
               "changed_px": int(np.any(final != human, axis=2).sum())}
        for cls, color in CLASS_COLORS.items():
            h, f = class_binary(human, color), class_binary(final, color)
            row[f"{cls}_human_px"] = int(h.sum())
            row[f"{cls}_refined_px"] = int(f.sum())
            row[f"{cls}_diff_px"] = int(f.sum()) - int(h.sum())
            row[f"{cls}_boundary_dist_px"] = boundary_distance(h, f)
        rows.append(row)
    if not rows:
        log.error("no final masks to evaluate")
        return None
    out = out_root / "evaluation.csv"
    with open(out, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    print(f"\n{len(rows)} images -- human vs refined (spatter_refined)")
    print(f"{'class':<18}{'human px':>12}{'refined px':>12}{'diff':>10}{'boundary dist':>16}")
    for cls in CLASS_COLORS:
        h = sum(r[f"{cls}_human_px"] for r in rows)
        f_ = sum(r[f"{cls}_refined_px"] for r in rows)
        d = [r[f"{cls}_boundary_dist_px"] for r in rows if r[f"{cls}_boundary_dist_px"] != ""]
        dist = f"{np.mean(d):.2f} px" if d and cls != "background" else "-"
        print(f"{cls:<18}{h:>12}{f_:>12}{f_ - h:>+10}{dist:>16}")
    changed = [r["changed_px"] for r in rows]
    print(f"pixels changed per image: median {int(np.median(changed))}, max {max(changed)}")
    print(f"-> {out}")
    return out


def evaluate(out_root: Path, simple: bool = False) -> Path | None:
    manifest_path = out_root / "manifest.json"
    if not manifest_path.exists():
        log.error("no %s -- run the pipeline first", manifest_path)
        return None
    entries = json.loads(manifest_path.read_text(encoding="utf-8"))
    if simple:
        return evaluate_simple(out_root, entries)
    rows = []
    for key in sorted(entries):
        e = entries[key]
        human = read_mask(Path(e["human"]))
        row = {"split": e["split"], "experiment": e["experiment"], "mask": e["mask"]}
        stages = {}
        for folder, name in STAGES:
            p = stage_path(out_root, folder, e["split"], e["experiment"], e["mask"])
            stages[name] = read_mask(p, Path(e["human"]).parent) if p.exists() else None
        for cls, color in CLASS_COLORS.items():
            h = int(class_binary(human, color).sum())
            row[f"{cls}_human"] = h
            for _, name in STAGES:
                row[f"{cls}_{name}"] = "" if stages[name] is None else int(class_binary(stages[name], color).sum())
            for _, name in STAGES:
                v = row[f"{cls}_{name}"]
                row[f"{cls}_d{name[4:]}"] = "" if v == "" else v - h
        for _, name in STAGES:
            m = stages[name]
            row[f"changed_{name}"] = "" if m is None else int(np.any(m != human, axis=2).sum())
        rows.append(row)

    out = out_root / "evaluation.csv"
    if not rows:
        log.error("manifest is empty")
        return None
    with open(out, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    print(f"\n{len(rows)} images -- total pixels per class (difference vs human)")
    print(f"{'class':<18}{'human':>11}{'step6':>18}{'step8':>18}{'step10':>18}")
    for cls in CLASS_COLORS:
        h = sum(r[f"{cls}_human"] for r in rows)
        cells = []
        for _, name in STAGES:
            vals = [r[f"{cls}_{name}"] for r in rows if r[f"{cls}_{name}"] != ""]
            total = sum(vals)
            cells.append(f"{total} ({total - h:+d})" if vals else "-")
        print(f"{cls:<18}{h:>11}{cells[0]:>18}{cells[1]:>18}{cells[2]:>18}")
    for _, name in STAGES:
        vals = [r[f"changed_{name}"] for r in rows if r[f"changed_{name}"] != ""]
        if vals:
            print(f"pixels changed vs human at {name}: {sum(vals)} "
                  f"(median per image {int(np.median(vals))})")
    print(f"-> {out}")
    return out


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Step 11: per-image class pixel counts per stage.")
    p.add_argument("--out-root", type=Path, default=REPO_ROOT / "outputs")
    p.add_argument("--simple", action="store_true",
                   help="human vs final result only: pixel counts, difference, boundary distance")
    args = p.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    return 0 if evaluate(args.out_root.resolve(), simple=args.simple) else 1


if __name__ == "__main__":
    raise SystemExit(main())
