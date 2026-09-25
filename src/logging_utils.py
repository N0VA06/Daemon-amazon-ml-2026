"""Stdout + file logging for the pipeline, plus a short experiments.md append."""

from __future__ import annotations

import json
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

LOG = logging.getLogger("er")

_FILE_HANDLER_KEY = "er_file"


def _fmt_elapsed(seconds: float) -> str:
    seconds = max(0.0, float(seconds))
    if seconds < 60:
        return f"{seconds:.1f}s"
    mins, sec = divmod(int(seconds), 60)
    if mins < 60:
        return f"{mins}m {sec}s"
    hours, mins = divmod(mins, 60)
    return f"{hours}h {mins}m {sec}s"


def fmt_elapsed(seconds: float) -> str:
    return _fmt_elapsed(seconds)


def setup_logging(verbose: bool = True, log_file: str | Path | None = None) -> logging.Logger:
    """Idempotent. Always INFO+ on stdout; DEBUG when verbose. File gets DEBUG."""
    level = logging.DEBUG if verbose else logging.INFO
    LOG.setLevel(logging.DEBUG)
    LOG.propagate = False

    if not any(isinstance(h, logging.StreamHandler) and not isinstance(h, logging.FileHandler) for h in LOG.handlers):
        stream = logging.StreamHandler(sys.stdout)
        stream.setLevel(logging.DEBUG if verbose else logging.INFO)
        stream.setFormatter(
            logging.Formatter("%(asctime)s | %(levelname)-5s | %(message)s", datefmt="%H:%M:%S")
        )
        LOG.addHandler(stream)
    else:
        for h in LOG.handlers:
            if isinstance(h, logging.StreamHandler) and not isinstance(h, logging.FileHandler):
                h.setLevel(logging.DEBUG if verbose else logging.INFO)

    if log_file is not None:
        log_file = Path(log_file)
        log_file.parent.mkdir(parents=True, exist_ok=True)
        already = any(getattr(h, "baseFilename", None) == str(log_file.resolve()) for h in LOG.handlers)
        if not already:
            fh = logging.FileHandler(log_file, encoding="utf-8")
            fh.setLevel(logging.DEBUG)
            fh.setFormatter(
                logging.Formatter(
                    "%(asctime)s | %(levelname)-5s | %(message)s",
                    datefmt="%Y-%m-%d %H:%M:%S",
                )
            )
            fh._er_key = _FILE_HANDLER_KEY  # type: ignore[attr-defined]
            LOG.addHandler(fh)
            LOG.info("log file  %s", log_file)
            latest = log_file.parent / "latest.log"
            try:
                if latest.exists() or latest.is_symlink():
                    latest.unlink()
                latest.symlink_to(log_file.resolve())
            except OSError:
                latest.write_text(f"{log_file.resolve()}\n", encoding="utf-8")
    return LOG


def default_log_path(reports_dir: str | Path) -> Path:
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    return Path(reports_dir) / "logs" / f"run_{ts}.log"


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
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, indent=2, default=str) + "\n", encoding="utf-8")
    tmp.replace(path)
    LOG.debug("wrote %s", path)


def read_json(path: str | Path) -> Any:
    return json.loads(Path(path).read_text(encoding="utf-8"))
