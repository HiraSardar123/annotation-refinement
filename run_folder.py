r"""Run the pipeline on one folder: an experiment folder or a mask folder.

Box / point visualisations are ON by default (--no-visualize turns them off).

    .\.venv\Scripts\python.exe .\run_folder.py .\data\prepared\train\<experiment>
    .\.venv\Scripts\python.exe .\run_folder.py <folder> --pilot 5 --out-root .\outputs_one

All options: .\.venv\Scripts\python.exe .\run_folder.py --help
"""

import sys

from pipeline.main import main

if __name__ == "__main__":
    args = sys.argv[1:]
    if not args or args[0].startswith("-"):
        if "-h" in args or "--help" in args:
            raise SystemExit(main(["--help"]))
        sys.exit("usage: run_folder.py FOLDER [options]   (see --help)")
    raise SystemExit(main(["--folder", args[0], *args[1:]]))
