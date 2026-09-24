"""YAML config loader, seed pinning, and backbone preset resolution."""

from __future__ import annotations

import os
import random
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import yaml

BACKBONE_PRESETS = {
    "jina": {
        "repo": "jinaai/jina-embeddings-v5-text-small-text-matching",
        "prefix": "Document: ",
        "dim": 1024,
        "trust_remote_code": True,
        "licence_expected": "cc-by-nc-4.0",
    },
    "snowflake": {
        "repo": "Snowflake/snowflake-arctic-embed-l-v2.0",
        "prefix": "query: ",
        "dim": 1024,
        "trust_remote_code": False,
        "licence_expected": "apache-2.0",
    },
}


def _to_ns(obj: Any) -> Any:
    if isinstance(obj, dict):
        return SimpleNamespace(**{k: _to_ns(v) for k, v in obj.items()})
    if isinstance(obj, list):
        return [_to_ns(v) for v in obj]
    return obj


def _to_dict(obj: Any) -> Any:
    if isinstance(obj, SimpleNamespace):
        return {k: _to_dict(v) for k, v in vars(obj).items()}
    if isinstance(obj, list):
        return [_to_dict(v) for v in obj]
    return obj


def load_config(path: str | Path) -> SimpleNamespace:
    path = Path(path)
    with path.open("r", encoding="utf-8") as f:
        raw = yaml.safe_load(f)
    cfg = _to_ns(raw)
    cfg._raw = raw
    cfg._path = str(path)
    apply_backbone_preset(cfg)
    return cfg


def apply_backbone_preset(cfg: SimpleNamespace) -> None:
    """Fill missing backbone fields from the named preset. Explicit YAML wins."""
    bb = cfg.backbone
    preset = str(getattr(bb, "preset", "jina")).lower()
    defaults = BACKBONE_PRESETS.get(preset, BACKBONE_PRESETS["jina"])
    for key, value in defaults.items():
        if not hasattr(bb, key) or getattr(bb, key) in (None, ""):
            setattr(bb, key, value)
    bb.preset = preset


def as_dict(cfg: SimpleNamespace) -> dict:
    return _to_dict(cfg)


def set_seeds(seed: int) -> None:
    seed = int(seed)
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
    except ImportError:
        pass


def resolve_path(cfg: SimpleNamespace, *parts: str) -> Path:
    return Path(*parts)
