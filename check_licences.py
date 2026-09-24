#!/usr/bin/env python3
"""Print the Hugging Face licence tag of every model this pipeline uses.

Writes reports/licences.md and reports/licences.json so the documentation
can be filled without hand-editing licence strings.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from src.config import load_config
from src.licences import run_licence_check


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="configs/default.yaml")
    args = p.parse_args()
    cfg = load_config(args.config)
    rows = run_licence_check(cfg, Path(cfg.paths.reports_dir))
    print()
    print(f"wrote {cfg.paths.reports_dir}/licences.md  ({len(rows)} model(s))")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
