"""Append-only experiment log. Every stage writes one block to experiments.md."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def log_experiment(path: str | Path, stage: str, message: str, metrics: dict | None = None) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not path.exists():
        path.write_text(
            "# Experiments log\n\n"
            "Append-only. Every pipeline stage records what changed and any metrics.\n\n",
            encoding="utf-8",
        )
    ts = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
    lines = [f"## {ts} — `{stage}`", "", message.strip(), ""]
    if metrics:
        lines.append("```json")
        lines.append(json.dumps(metrics, indent=2, default=str))
        lines.append("```")
        lines.append("")
    with path.open("a", encoding="utf-8") as f:
        f.write("\n".join(lines))


def write_json(path: str | Path, obj: Any) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, default=str) + "\n", encoding="utf-8")


def read_json(path: str | Path) -> Any:
    return json.loads(Path(path).read_text(encoding="utf-8"))
