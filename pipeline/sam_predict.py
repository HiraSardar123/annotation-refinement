"""Step 4 -- SAM2 predictions.

`predict_component`: one SAM2 call per step-2 component with its step-3
prompt. Of SAM's three candidate masks the highest-scoring one is used, and
it is clipped to the component's box and kept off the other target class's
human pixels (molten never takes human-droplet pixels, droplet never takes
human-molten pixels). Only the largest connected piece is kept (drops
speckle). If nothing is left, the human component is kept unchanged.

`class_probabilities`: used by steps 5 / 7 / 9. For a region of unlabelled
pixels and each candidate class, SAM2 is prompted for that class (positives
on the class's pixels next to the region, negatives chosen by step 3's rule,
never on the class itself or on the region) and returns, per pixel, the
probability that the pixel belongs to the class.
"""

from __future__ import annotations

import logging
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .common import REPO_ROOT, box_mask, class_binary, largest_piece, padded_box
from .config import class_pad
from .prompts import Prompt, make_prompt

log = logging.getLogger("pipeline")


def load_predictor(cfg):
    """Import SAM2 with the repo root off sys.path: the `sam2/` checkout next
    to the scripts would otherwise shadow the package in third_party/sam2."""
    import torch

    sys.path = [e for e in sys.path if Path(e or ".").resolve() != REPO_ROOT]
    from sam2.build_sam import build_sam2
    from sam2.sam2_image_predictor import SAM2ImagePredictor
    import sam2.build_sam as build_module

    device = cfg.device
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    log.info("SAM2 package: %s", build_module.__file__)
    log.info("SAM2 device : %s", device)
    return SAM2ImagePredictor(build_sam2(cfg.model_config, str(cfg.checkpoint), device=device))


def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(x, -30.0, 30.0)))


def _best_logits(predictor, prompt: Prompt) -> tuple[np.ndarray, float] | None:
    try:
        masks, scores, _ = predictor.predict(
            point_coords=prompt.points, point_labels=prompt.labels, box=prompt.box,
            multimask_output=True, return_logits=True)
    except Exception as exc:  # a bad prompt must not kill the run
        log.debug("SAM2 predict failed: %s", exc)
        return None
    best = int(np.argmax(scores))
    return np.asarray(masks[best], dtype=np.float32), float(scores[best])


@dataclass
class Prediction:
    cls: str
    mask: np.ndarray     # refined component (or the human one if SAM gave nothing)
    score: float
    accepted: bool


def predict_component(predictor, comp, prompt: Prompt) -> Prediction:
    out = _best_logits(predictor, prompt)
    if out is None:
        return Prediction(comp.cls, comp.mask.copy(), 0.0, False)
    logits, score = out
    mask = (logits > predictor.mask_threshold) & box_mask(comp.box, comp.mask.shape) & ~comp.forbidden
    if not mask.any():
        return Prediction(comp.cls, comp.mask.copy(), score, False)
    return Prediction(comp.cls, largest_piece(mask), score, True)


def class_probabilities(predictor, region: np.ndarray, candidates: list, current: np.ndarray,
                        unknown: np.ndarray, near: np.ndarray, cfg) -> np.ndarray:
    """(len(candidates), N) probability per region pixel (np.nonzero order)."""
    ys, xs = np.nonzero(region)
    probs = np.zeros((len(candidates), len(ys)), np.float32)
    for k, color in enumerate(candidates):
        own = class_binary(current, color) & ~unknown
        support = near & own
        if not support.any():
            continue
        pad = class_pad(color, cfg)
        box = padded_box(region | support, pad)
        prompt = make_prompt(support, box, pad, current, cfg, avoid=unknown | own)
        out = _best_logits(predictor, prompt)
        if out is not None:
            probs[k] = _sigmoid(out[0][ys, xs])
    return probs


if __name__ == "__main__":
    from .step_cli import run_single
    raise SystemExit(run_single(4))
