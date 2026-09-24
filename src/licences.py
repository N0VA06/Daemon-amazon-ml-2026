"""Print Hugging Face licence tags for every model the pipeline can load."""

from __future__ import annotations

from pathlib import Path

from src.config import BACKBONE_PRESETS
from src.logging_utils import write_json


def hf_licence(repo: str) -> dict:
    info = {"repo": repo, "licence": "unknown", "card": {}}
    try:
        from huggingface_hub import model_info

        mi = model_info(repo)
        card = mi.cardData or {}
        lic = card.get("license") or card.get("licence") or getattr(mi, "license", None)
        info["licence"] = str(lic) if lic else "unknown"
        info["card"] = {k: card.get(k) for k in ("license", "license_name", "language") if k in card}
    except Exception as exc:
        info["error"] = str(exc)
    return info


def collect_licences(cfg=None) -> list[dict]:
    repos = []
    if cfg is not None:
        repos.append(cfg.backbone.repo)
        other = (
            BACKBONE_PRESETS["snowflake"]["repo"]
            if cfg.backbone.preset == "jina"
            else BACKBONE_PRESETS["jina"]["repo"]
        )
        repos.append(other)
    else:
        repos = [p["repo"] for p in BACKBONE_PRESETS.values()]
    # unique, preserve order
    seen = set()
    ordered = []
    for r in repos:
        if r not in seen:
            seen.add(r)
            ordered.append(r)
    return [hf_licence(r) for r in ordered]


def render_licence_md(rows: list[dict]) -> str:
    lines = [
        "# Model licences",
        "",
        "Printed by `python check_licences.py`. The PDF asks for MIT / Apache-2.0;",
        "jina-embeddings-v5 is CC-BY-NC-4.0. The backbone is a single config key",
        "(`backbone.preset` / `backbone.repo`) so Snowflake/arctic-embed-l-v2.0",
        "(Apache-2.0, prefix `query: `) is a drop-in replacement.",
        "",
        "| repo | HF licence tag | notes |",
        "| --- | --- | --- |",
    ]
    notes = {
        "jinaai/jina-embeddings-v5-text-small-text-matching":
            "Primary backbone (user choice). Text-matching LoRA merged. Swap via config.",
        "Snowflake/snowflake-arctic-embed-l-v2.0":
            "Tested Apache-2.0 drop-in. Same code path, prefix `query: `.",
    }
    for r in rows:
        lines.append(
            f"| `{r['repo']}` | `{r['licence']}` | {notes.get(r['repo'], '')} |"
        )
    lines.append("")
    return "\n".join(lines)


def run_licence_check(cfg, reports_dir: Path) -> list[dict]:
    rows = collect_licences(cfg)
    reports_dir.mkdir(parents=True, exist_ok=True)
    write_json(reports_dir / "licences.json", rows)
    (reports_dir / "licences.md").write_text(render_licence_md(rows), encoding="utf-8")
    for r in rows:
        print(f"{r['repo']}\t{r['licence']}")
    return rows
