r"""Run the pipeline on the complete dataset (train + test).

Box / point visualisations are OFF by default (--visualize turns them on);
human|SAM comparison images are written (--no-comp skips them).

    .\.venv\Scripts\python.exe .\run_dataset.py
    .\.venv\Scripts\python.exe .\run_dataset.py --splits test --pilot 5 --limit-dirs 3

All options: .\.venv\Scripts\python.exe .\run_dataset.py --help
"""

import sys

from pipeline.main import main

if __name__ == "__main__":
    if any(a == "--folder" or a.startswith("--folder=") for a in sys.argv[1:]):
        sys.exit("run_dataset.py processes the whole dataset; use run_folder.py for one folder")
    raise SystemExit(main(sys.argv[1:]))
