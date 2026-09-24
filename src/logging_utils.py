"""Stdout logging for the pipeline, plus a short experiments.md append."""

from __future__ import annotations

import json
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

LOG = logging.getLogger("er")


def setup_logging(verbose: bool = True) -> logging.Logger:
    """Idempotent. Verbose = DEBUG (row counts, cache hits); always INFO+ on stdout."""
    if LOG.handlers:
        LOG.setLevel(logging.DEBUG if verbose else logging.INFO)
        return LOG
    LOG.setLevel(logging.DEBUG if verbose else logging.INFO)
    handler = logging.StreamHandler(sys.stdout)
    handler.setLevel(logging.DEBUG)
    handler.setFormatter(
        logging.Formatter("%(asctime)s | %(levelname)-5s | %(message)s", datefmt="%H:%M:%S")
    )
    LOG.addHandler(handler)
    LOG.propagate = False
    return LOG


def banner(stage: str, index: int | None = None, total: int | None = None) -> None:
    label = f"stage {index}/{total}  {stage}" if index and total else stage
    bar = "=" * 72
    LOG.info(bar)
    LOG.info("%s", label)
    LOG.info(bar)


def log_experiment(path: str | Path, stage: str, message: str, metrics: dict | None = None) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not path.exists():
        path.write_text("# Experiments\n\n", encoding="utf-8")
    ts = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
    lines = [f"## {ts} — `{stage}`", "", message.strip(), ""]
    if metrics:
        lines.append("```json")
        lines.append(json.dumps(metrics, indent=2, default=str))
        lines.append("```")
        lines.append("")
    with path.open("a", encoding="utf-8") as f:
        f.write("\n".join(lines))
    extra = ""
    if metrics:
        keys = [k for k in ("macro_f05", "n_pairs", "n_s1", "avg_candidates", "n_pred_links") if k in metrics]
        if "holdout" in metrics and isinstance(metrics["holdout"], dict):
            extra = f"  holdout_F0.5={metrics['holdout'].get('macro_f05')}"
        elif keys:
            extra = "  " + " ".join(f"{k}={metrics[k]}" for k in keys)
    LOG.info("done %s%s", stage, extra)


def write_json(path: str | Path, obj: Any) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, default=str) + "\n", encoding="utf-8")
    LOG.debug("wrote %s", path)


def read_json(path: str | Path) -> Any:
    return json.loads(Path(path).read_text(encoding="utf-8"))
