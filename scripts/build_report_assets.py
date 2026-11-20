"""Copy the generated tables and figures the report uses next to it (Phase 11).

Run from the repo root inside the conda env `script-gap`:

    python -m scripts.build_report_assets --config configs/report.yaml          # copy
    python -m scripts.build_report_assets --config configs/report.yaml --check  # only compare

Overleaf compiles one uploaded folder, so the report cannot reach into
``results/``. The files listed under ``assets`` are copied into ``assets_dir``
(``docs/report/generated/``) unchanged; ``--check`` exits with an error if any
copy differs from its source, so the report never shows stale numbers.
"""

import argparse
import filecmp
import shutil
import sys

from src.utils.config import load_config, resolve_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--config", required=True)
    parser.add_argument("--check", action="store_true", help="compare only, copy nothing")
    args = parser.parse_args()
    cfg = load_config(args.config)

    out_dir = resolve_path(cfg["assets_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)
    stale = []
    for source in cfg["assets"]:
        src = resolve_path(source)
        dst = out_dir / src.name
        same = dst.exists() and filecmp.cmp(src, dst, shallow=False)
        if args.check:
            if not same:
                stale.append(source)
            continue
        if not same:
            shutil.copyfile(src, dst)
        print(f"{'unchanged' if same else 'copied   '} {source} -> {cfg['assets_dir']}/{src.name}")
    if stale:
        sys.exit("out of date (run without --check): " + ", ".join(stale))
    if args.check:
        print(f"all {len(cfg['assets'])} assets match their sources")


if __name__ == "__main__":
    main()
