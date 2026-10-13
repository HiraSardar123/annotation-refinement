r"""Start the annotation tool and open it in the browser.

    .\.venv\Scripts\python.exe .\annotate.py
    .\.venv\Scripts\python.exe .\annotate.py --result .\final\spatter_refined --out .\edited

Left side  : the human annotations (dataset root with train/ and test/, and
             each experiment's COCO JSON for the instances).
Right side : the folder being annotated, matched by mask file name
             (default: final\spatter_refined). Edits are saved under --out.
Stop with Ctrl+C.
"""

import argparse
import logging
import webbrowser
from pathlib import Path

from annotator.data import Catalog
from annotator.server import serve

ROOT = Path(__file__).resolve().parent


def main() -> int:
    p = argparse.ArgumentParser(description="Annotation review / editing tool (runs in your browser).")
    p.add_argument("--dataset", type=Path, default=ROOT, help="root with train/ and test/ (human side)")
    p.add_argument("--result", type=Path, default=ROOT / "final" / "spatter_refined",
                   help="folder being annotated (RGB masks, matched by file name)")
    p.add_argument("--out", type=Path, default=ROOT / "edited",
                   help="fully reviewed experiment folders end up here; work in progress is "
                        "kept in <out>_in_progress")
    p.add_argument("--splits", nargs="+", default=["train", "test"])
    p.add_argument("--port", type=int, default=8000)
    p.add_argument("--no-browser", action="store_true", help="do not open the browser")
    args = p.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    cat = Catalog(args.dataset.resolve(), args.result.resolve(), args.out.resolve(), args.splits)
    with_result = sum(e.result_path is not None for e in cat.entries.values())
    print(f"{len(cat.entries)} images, {with_result} with a result mask in {args.result}")
    print(f"edits are saved to {cat.work_root}")
    print(f"an experiment folder is copied to {args.out.resolve()} once all its images are reviewed")
    try:
        server = serve(cat, args.port)
    except OSError:
        print()
        print(f"Port {args.port} is already in use -- the tool is probably already running.")
        print(f"Open http://127.0.0.1:{args.port}/ , close the other copy (Ctrl+C in its window),")
        print("or start this one on another port, e.g.  --port 8001")
        return 1
    url = f"http://127.0.0.1:{args.port}/"
    print(f"open {url}   (Ctrl+C to stop)")
    if not args.no_browser:
        webbrowser.open(url)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("stopped")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
