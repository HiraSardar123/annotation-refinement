r"""Step 0: verify the environment and that SAM2 actually loads + predicts.

Run:
    .\.venv\Scripts\python.exe .\check_env.py --root . --checkpoint .\sam2.1_hiera_large.pt
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent


def load_sam2(checkpoint: Path, model_config: str, device: str | None = None):
    """Import SAM2 without the local ``sam2/`` checkout shadowing the package.

    A directory named ``sam2`` sits next to this script; leaving the repo root
    on sys.path makes ``import sam2`` resolve to that empty-ish checkout instead
    of the installed editable package in third_party/sam2.
    """
    import torch

    sys.path = [e for e in sys.path if Path(e or ".").resolve() != REPO_ROOT]
    from sam2.build_sam import build_sam2
    from sam2.sam2_image_predictor import SAM2ImagePredictor
    import sam2.build_sam as build_module

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    model = build_sam2(model_config, str(checkpoint), device=device)
    return SAM2ImagePredictor(model), device, build_module.__file__


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=REPO_ROOT)
    parser.add_argument("--checkpoint", type=Path, default=REPO_ROOT / "sam2.1_hiera_large.pt")
    parser.add_argument("--model-config", default="configs/sam2.1/sam2.1_hiera_l.yaml")
    args = parser.parse_args()

    import numpy as np
    import torch
    import cv2
    from PIL import Image

    print("=" * 60)
    print("ENVIRONMENT")
    print("=" * 60)
    print(f"python      : {sys.executable}")
    in_venv = "\\.venv\\" in sys.executable.lower() or "/.venv/" in sys.executable.lower()
    print(f"in .venv    : {in_venv}")
    if not in_venv:
        print("  !! Not running the project .venv. Use:")
        print("     .\\.venv\\Scripts\\python.exe check_env.py")
    print(f"torch       : {torch.__version__}  cuda={torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"gpu         : {torch.cuda.get_device_name(0)}")
    else:
        print("  note: CPU-only torch. SAM2 Hiera-Large on CPU is ~5-15 s/prompt.")
    print(f"opencv      : {cv2.__version__}")
    print(f"numpy       : {np.__version__}")
    print(f"pillow      : {Image.__version__}")

    print()
    print("=" * 60)
    print("DATA")
    print("=" * 60)
    root = args.root.resolve()
    mask_dirs = sorted(p.parent for p in root.rglob("class_rgb_mapping.txt")
                       if "refined" not in p.parts and "overlays" not in p.parts)
    print(f"mask directories found: {len(mask_dirs)}")
    total = 0
    for d in mask_dirs:
        n = len([p for p in d.glob("*.png")])
        total += n
    print(f"mask png files        : {total}")
    for d in mask_dirs[:3]:
        print(f"  e.g. {d.relative_to(root)}")

    print()
    print("=" * 60)
    print("SAM2")
    print("=" * 60)
    if not args.checkpoint.exists():
        print(f"  !! checkpoint missing: {args.checkpoint}")
        return 1
    size_gb = args.checkpoint.stat().st_size / 1e9
    print(f"checkpoint  : {args.checkpoint.name} ({size_gb:.2f} GB)")

    predictor, device, module_path = load_sam2(args.checkpoint, args.model_config)
    print(f"sam2 module : {module_path}")
    ok = "third_party" in module_path
    print(f"correct pkg : {ok}")
    print(f"device      : {device}")

    # Smoke test: a synthetic bright blob on dark background, box + point prompt.
    img = np.zeros((256, 256, 3), dtype=np.uint8)
    cv2.circle(img, (128, 128), 40, (255, 180, 60), -1)
    predictor.set_image(img)
    masks, scores, _ = predictor.predict(
        point_coords=np.array([[128.0, 128.0]], dtype=np.float32),
        point_labels=np.array([1], dtype=np.int32),
        box=np.array([80.0, 80.0, 176.0, 176.0], dtype=np.float32),
        multimask_output=True,
    )
    best = int(np.argmax(scores))
    area = int(np.asarray(masks[best], dtype=bool).sum())
    print(f"smoke test  : {len(masks)} masks, best score {scores[best]:.3f}, area {area} px "
          f"(expected ~{int(np.pi * 40 ** 2)})")
    print()
    print("ALL GOOD" if ok and area > 3000 else "CHECK THE WARNINGS ABOVE")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
