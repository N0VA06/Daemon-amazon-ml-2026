"""Stage checkpoints so `--stage all` can continue after a kill.

A stage is "done" when its required artifacts exist on disk. Progress is
also written to `cache/progress.json` (start / done / fail + elapsed).
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from src.logging_utils import LOG, write_json


# Ordered CLI stages (not including "all").
PIPELINE_STAGES = (
    "eda",
    "split",
    "normalize",
    "block",
    "train_biencoder",
    "features",
    "train_matcher",
    "train_crossencoder",
    "stack",
    "decide",
    "evaluate",
    "predict",
)

# Relative to the student_resource/ root (cwd when you run python -m src.run).
STAGE_ARTIFACTS: dict[str, tuple[str, ...]] = {
    "eda": ("reports/eda.json",),
    "split": (
        "cache/splits.json",
        "cache/val_gallery_s2.parquet",
        "cache/val_gallery_s3.parquet",
    ),
    "normalize": (
        "cache/normalized/train_s1.parquet",
        "cache/normalized/train_s2.parquet",
        "cache/normalized/train_s3.parquet",
        "cache/normalized/test_s1.parquet",
        "cache/normalized/test_s2.parquet",
        "cache/normalized/test_s3.parquet",
        "cache/idf.json",
    ),
    "block": ("cache/val_pairs_zs.parquet",),
    "train_biencoder": ("models/biencoder/merged",),
    "features": ("cache/val_features.parquet",),
    "train_matcher": ("models/lgbm.joblib",),
    "train_crossencoder": ("models/crossencoder/crossencoder.pt",),
    "stack": ("models/stacker.joblib",),
    "decide": ("reports/decision.json",),
    "evaluate": ("reports/evaluate.json",),
    "predict": (
        "output/matching_results.tsv",
        "output/candidate_pairs.tsv",
    ),
}


def _now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")


def progress_path(cfg) -> Path:
    return Path(cfg.paths.cache_dir) / "progress.json"


def load_progress(cfg) -> dict:
    p = progress_path(cfg)
    if not p.exists():
        return {"stages": {}, "updated_at": None}
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        return {"stages": {}, "updated_at": None}


def save_progress(cfg, payload: dict) -> None:
    payload["updated_at"] = _now()
    write_json(progress_path(cfg), payload)


def _root_paths(cfg) -> dict[str, str]:
    return {
        "cache": str(Path(cfg.paths.cache_dir)),
        "reports": str(Path(cfg.paths.reports_dir)),
        "models": str(Path(cfg.paths.models_dir)),
        "output": str(Path(cfg.paths.output_dir)),
    }


def artifact_paths(cfg, stage: str) -> list[Path]:
    """Resolve STAGE_ARTIFACTS entries against the configured dirs."""
    roots = _root_paths(cfg)
    out = []
    for rel in STAGE_ARTIFACTS.get(stage, ()):
        parts = Path(rel).parts
        if parts and parts[0] in roots:
            out.append(Path(roots[parts[0]], *parts[1:]))
        else:
            out.append(Path(rel))
    return out


def artifacts_exist(cfg, stage: str) -> bool:
    paths = artifact_paths(cfg, stage)
    return bool(paths) and all(p.exists() for p in paths)


def missing_artifacts(cfg, stage: str) -> list[str]:
    return [str(p) for p in artifact_paths(cfg, stage) if not p.exists()]


def stage_enabled(cfg, stage: str) -> bool:
    if stage == "train_biencoder":
        return bool(getattr(cfg.biencoder, "enabled", True)) and not bool(
            getattr(cfg.debug, "fast", False)
        )
    if stage == "train_crossencoder":
        return bool(getattr(cfg.cross_encoder, "enabled", True)) and not bool(
            getattr(cfg.debug, "fast", False)
        )
    if stage == "stack":
        return bool(getattr(cfg.stacker, "enabled", True))
    return True


def features_need_refresh(cfg) -> bool:
    """True if val_features exists but is missing finetuned cosines after LoRA."""
    feats = Path(cfg.paths.cache_dir) / "val_features.parquet"
    merged = Path(cfg.paths.models_dir) / "biencoder" / "merged"
    if not feats.exists() or not merged.exists():
        return False
    if not bool(getattr(cfg.features, "compute_finetuned_cosine", True)):
        return False
    try:
        import pyarrow.parquet as pq

        cols = set(pq.read_schema(feats).names)
    except Exception:
        return False
    return "cosine_ft_combined" not in cols


def is_complete(cfg, stage: str) -> bool:
    if not stage_enabled(cfg, stage):
        return True
    if not artifacts_exist(cfg, stage):
        return False
    if stage == "features" and features_need_refresh(cfg):
        LOG.info(
            "features artifact is stale (LoRA merged exists, no cosine_ft_combined) — will re-run"
        )
        return False
    if stage == "train_matcher":
        feats = Path(cfg.paths.cache_dir) / "val_features.parquet"
        if feats.exists():
            try:
                import pyarrow.parquet as pq

                cols = pq.read_schema(feats).names
                if "p_gbm" not in cols:
                    return False
            except Exception:
                pass
    return True


def mark_stage(cfg, stage: str, status: str, **extra: Any) -> None:
    payload = load_progress(cfg)
    stages = payload.setdefault("stages", {})
    rec = stages.get(stage, {})
    rec.update(extra)
    rec["status"] = status
    rec["updated_at"] = _now()
    stages[stage] = rec
    payload["stages"] = stages
    save_progress(cfg, payload)


def log_artifact_status(cfg, stage: str) -> None:
    for p in artifact_paths(cfg, stage):
        if p.exists():
            if p.is_file():
                mb = p.stat().st_size / (1024 * 1024)
                LOG.info("  artifact  %s  (%.1f MB)", p, mb)
            else:
                LOG.info("  artifact  %s  (dir)", p)
        else:
            LOG.info("  missing   %s", p)


def print_status(cfg) -> None:
    prog = load_progress(cfg)
    LOG.info("=" * 72)
    LOG.info("checkpoint status  progress=%s", progress_path(cfg))
    LOG.info("=" * 72)
    recorded = prog.get("stages", {})
    for stage in PIPELINE_STAGES:
        enabled = stage_enabled(cfg, stage)
        done = is_complete(cfg, stage)
        rec = recorded.get(stage, {})
        rec_status = rec.get("status", "—")
        elapsed = rec.get("elapsed_s")
        extra = f"  elapsed={elapsed:.0f}s" if isinstance(elapsed, (int, float)) else ""
        flag = "DONE" if done else ("SKIP" if not enabled else "TODO")
        LOG.info(
            "  %-20s  %s  recorded=%-8s%s",
            stage,
            flag,
            rec_status,
            extra,
        )
        if not done and enabled:
            miss = missing_artifacts(cfg, stage)
            if miss:
                LOG.info("    waiting on: %s", ", ".join(miss))
    LOG.info("re-run:  python -m src.run --stage all")
    LOG.info("force:   python -m src.run --stage all --force")
    LOG.info("from:    python -m src.run --stage all --force-from features")
