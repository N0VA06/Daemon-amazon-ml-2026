"""Swappable backbone loader (jina / snowflake) and embedding cache.

Loading contract (both presets):
    SentenceTransformer(repo, model_kwargs={"dtype": torch.bfloat16})
    model.max_seq_length = 128
    embeddings L2-normalised.
Prefix is applied in normalize.attach_record_texts, never inside the encoder,
so training and inference stay identical.
"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import numpy as np

from src.logging_utils import LOG

_MODEL_CACHE: dict[str, object] = {}


def _torch_dtype(name: str):
    import torch

    return {
        "bfloat16": torch.bfloat16,
        "bf16": torch.bfloat16,
        "float16": torch.float16,
        "fp16": torch.float16,
        "float32": torch.float32,
    }.get(str(name).lower(), torch.bfloat16)


def load_sentence_transformer(
    cfg: SimpleNamespace,
    adapter_path: str | Path | None = None,
    use_cache: bool = True,
):
    """Load the configured backbone. adapter_path is an optional LoRA / merged dir.

    Training code that mutates the module (LoRA) must pass use_cache=False so a
    later zero-shot encode does not receive the adapted weights.
    """
    from sentence_transformers import SentenceTransformer

    repo = adapter_path or cfg.backbone.repo
    key = str(repo)
    if use_cache and key in _MODEL_CACHE:
        LOG.info("model cache hit  %s", key)
        return _MODEL_CACHE[key]
    LOG.info("loading SentenceTransformer  %s  dtype=%s  max_len=%s  cache=%s",
             key, getattr(cfg.backbone, "dtype", "bfloat16"), cfg.backbone.max_seq_length, use_cache)

    import torch

    kwargs = {"trust_remote_code": bool(getattr(cfg.backbone, "trust_remote_code", True))}
    model_kwargs = {"dtype": _torch_dtype(getattr(cfg.backbone, "dtype", "bfloat16"))}
    try:
        model = SentenceTransformer(
            key,
            model_kwargs=model_kwargs,
            trust_remote_code=kwargs["trust_remote_code"],
        )
    except TypeError:
        # older ST: no model_kwargs
        model = SentenceTransformer(key, trust_remote_code=kwargs["trust_remote_code"])
        try:
            model = model.to(model_kwargs["dtype"])
        except Exception:
            pass

    model.max_seq_length = int(cfg.backbone.max_seq_length)
    if use_cache:
        _MODEL_CACHE[key] = model
    return model


def get_auto_model(st_model):
    """Best-effort handle for the underlying HF decoder (LoRA / CE head)."""
    try:
        return st_model[0].auto_model
    except Exception:
        pass
    try:
        return st_model._first_module().auto_model
    except Exception:
        pass
    raise RuntimeError("Could not locate AutoModel inside SentenceTransformer")


def encode_texts(
    model,
    texts: list[str],
    batch_size: int,
    normalize: bool = True,
    show_progress: bool = True,
) -> np.ndarray:
    embs = model.encode(
        list(texts),
        batch_size=int(batch_size),
        convert_to_numpy=True,
        normalize_embeddings=bool(normalize),
        show_progress_bar=show_progress,
    )
    return np.asarray(embs, dtype=np.float32)


def embedding_path(cache_dir: Path, split: str, source: str, view: str, tag: str) -> Path:
    return cache_dir / "embeddings" / f"{split}_{source}_{view}_{tag}.npy"


def ids_path(cache_dir: Path, split: str, source: str) -> Path:
    return cache_dir / "embeddings" / f"{split}_{source}_ids.npy"


def save_embeddings(path: Path, matrix: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.save(path, np.asarray(matrix, dtype=np.float32))


def load_embeddings(path: Path) -> np.ndarray:
    return np.load(path)


def encode_frame_views(
    cfg: SimpleNamespace,
    df,
    split: str,
    source: str,
    tag: str,
    model=None,
    overwrite: bool = False,
) -> dict[str, Path]:
    """Encode combined / name / address views. Cache .npy + id alignment."""
    cache = Path(cfg.paths.cache_dir)
    views = list(cfg.backbone.encode_views)
    paths = {}
    id_p = ids_path(cache, split, source)
    if overwrite or not id_p.exists():
        id_p.parent.mkdir(parents=True, exist_ok=True)
        np.save(id_p, df["entity_id"].astype(str).to_numpy())
    need = [v for v in views if overwrite or not embedding_path(cache, split, source, v, tag).exists()]
    if not need:
        for v in views:
            p = embedding_path(cache, split, source, v, tag)
            mb = p.stat().st_size / (1024 * 1024) if p.exists() else 0
            LOG.info("embedding cache hit  %s/%s/%s tag=%s  %.1f MB", split, source, v, tag, mb)
        return {v: embedding_path(cache, split, source, v, tag) for v in views}
    if model is None:
        model = load_sentence_transformer(cfg)
    batch = int(cfg.hardware.encode_batch_size)
    col = {"combined": "text_combined", "name": "text_name", "address": "text_address"}
    for view in need:
        LOG.info("encode  %s/%s/%s/%s  n=%s  batch=%s", split, source, view, tag, f"{len(df):,}", batch)
        texts = df[col[view]].astype(str).tolist()
        embs = encode_texts(model, texts, batch_size=batch, normalize=cfg.backbone.normalize)
        p = embedding_path(cache, split, source, view, tag)
        save_embeddings(p, embs)
        mb = p.stat().st_size / (1024 * 1024)
        LOG.info("wrote embeddings  %s  shape=%s  %.1f MB  (resume-safe)", p, embs.shape, mb)
        paths[view] = p
    for view in views:
        paths.setdefault(view, embedding_path(cache, split, source, view, tag))
    return paths


def load_id_index(cache_dir: Path, split: str, source: str) -> dict[str, int]:
    ids = np.load(ids_path(cache_dir, split, source), allow_pickle=True)
    return {str(i): int(k) for k, i in enumerate(ids)}
