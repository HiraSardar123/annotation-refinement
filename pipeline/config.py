"""All command-line settings, shared by main.py, run_dataset.py and run_folder.py."""

from __future__ import annotations

import argparse
from pathlib import Path

from .common import REPO_ROOT


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="SAM2 refinement pipeline (steps 1-11). See README.md.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    g = p.add_argument_group("input / output")
    g.add_argument("--root", type=Path, default=REPO_ROOT, help="dataset root with train/ and test/")
    g.add_argument("--folder", type=Path, default=None,
                   help="process only this folder (an experiment folder or a mask folder)")
    g.add_argument("--out-root", type=Path, default=REPO_ROOT / "outputs",
                   help="refined/, gaps_refined/, spatter_refined/ and the *_comp/ folders go here")
    g.add_argument("--splits", nargs="+", default=["train", "test"])
    g.add_argument("--pilot", type=int, default=None, help="only the first N masks per mask folder")
    g.add_argument("--limit-dirs", type=int, default=None, help="only the first N mask folders")
    g.add_argument("--overwrite", action="store_true", help="redo images that already have outputs")
    g.add_argument("--dry-run", action="store_true", help="list what would run, load nothing")

    g = p.add_argument_group("SAM2")
    g.add_argument("--checkpoint", type=Path, default=REPO_ROOT / "sam2.1_hiera_large.pt")
    g.add_argument("--model-config", default="configs/sam2.1/sam2.1_hiera_l.yaml")
    g.add_argument("--device", choices=("auto", "cuda", "cpu"), default="auto")

    g = p.add_argument_group("step 2: boxes")
    g.add_argument("--pad-molten", type=int, default=4, help="box padding for molten_consumable, px")
    g.add_argument("--pad-droplet", type=int, default=1, help="box padding for droplet, px")
    g.add_argument("--pad-other", type=int, default=3,
                   help="box padding for the other classes (relabel / gap / spatter prompts), px")

    g = p.add_argument_group("step 3: prompts")
    g.add_argument("--neg-outside-px", type=int, default=3,
                   help="the outside-box negative is taken within this many px beyond the box")

    g = p.add_argument_group("steps 5 / 7 / 9: relabelling")
    g.add_argument("--sam-conf", type=float, default=0.8,
                   help="SAM's label is used only when its confidence is at least this")
    g.add_argument("--relabel-ring-px", type=int, default=9,
                   help="classes within this many px of a region are its candidates")
    g.add_argument("--surround-tol-px", type=float, default=1.5,
                   help="a class 'surrounds' a pixel if it is within this many px of the "
                        "pixel's closest labelled class")

    g = p.add_argument_group("instances (after steps 5, 7, 9)")
    g.add_argument("--instance-close-px", type=int, default=2,
                   help="holes and cuts of other classes inside a molten/droplet instance "
                        "up to about 2x this wide are filled with the instance's class")

    g.add_argument("--molten-top-keep-px", type=int, default=10,
                   help="the top of the molten keeps its human outline: from the box top down "
                        "to this many px below the molten's highest pixel (0 = off)")

    g = p.add_argument_group("step 7: gaps")
    g.add_argument("--gap-close-px", type=int, default=2,
                   help="background gaps up to about 2x this wide between classes are found")
    g.add_argument("--gap-max-hole-px", type=int, default=500,
                   help="fully enclosed background holes up to this area are gaps")

    g = p.add_argument_group("step 9: static spatter")
    g.add_argument("--spatter-max-missing", type=int, default=4,
                   help="a pixel is static noise if it is spatter in all but at most this many masks")
    g.add_argument("--spatter-min-masks", type=int, default=8,
                   help="folders with fewer masks are not judged")
    g.add_argument("--spatter-overlap", type=float, default=0.5,
                   help="a spatter blob is removed if at least this share of it is static")

    g = p.add_argument_group("visualisation")
    g.add_argument("--visualize", dest="visualize", action="store_true", default=None,
                   help="write box and point images (default: on for --folder, off for the dataset)")
    g.add_argument("--no-visualize", dest="visualize", action="store_false")
    g.add_argument("--no-comp", dest="comp", action="store_false",
                   help="skip the human|SAM comparison images")
    g.add_argument("--overlay-scale", type=int, default=2)
    g.add_argument("--overlay-dim", type=float, default=0.6)
    g.add_argument("--overlay-thickness", type=int, default=1)

    g = p.add_argument_group("other")
    g.add_argument("--final-only", action="store_true",
                   help="save only the final result: spatter_refined/ and spatter_refined_comp/ "
                        "(no refined/, gaps_refined/ or their comparisons), and write the simple "
                        "evaluation")
    g.add_argument("--skip-eval", action="store_true", help="do not run step 11 at the end")
    g.add_argument("--log-level", default="INFO", choices=("DEBUG", "INFO", "WARNING", "ERROR"))
    return p


def parse(argv: list[str] | None = None) -> argparse.Namespace:
    cfg = build_parser().parse_args(argv)
    if cfg.visualize is None:
        cfg.visualize = cfg.folder is not None
    return cfg


def class_pad(color, cfg) -> int:
    from .common import DROPLET, MOLTEN
    if color == MOLTEN:
        return cfg.pad_molten
    if color == DROPLET:
        return cfg.pad_droplet
    return cfg.pad_other
