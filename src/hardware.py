"""GPU detection and automatic batch-size selection."""

from __future__ import annotations

from types import SimpleNamespace


def detect_device() -> dict:
    info = {
        "device": "cpu",
        "n_gpu": 0,
        "name": "cpu",
        "vram_gb": 0.0,
        "bf16": False,
    }
    try:
        import torch

        if torch.cuda.is_available():
            props = torch.cuda.get_device_properties(0)
            vram_gb = props.total_memory / (1024 ** 3)
            info.update(
                {
                    "device": "cuda",
                    "n_gpu": torch.cuda.device_count(),
                    "name": props.name,
                    "vram_gb": round(vram_gb, 2),
                    "bf16": bool(
                        getattr(torch.cuda, "is_bf16_supported", lambda: False)()
                    ),
                }
            )
    except ImportError:
        pass
    return info


def apply_auto_batch(cfg: SimpleNamespace) -> dict:
    """Overwrite encode / train batch sizes from VRAM. Returns the hardware info."""
    info = detect_device()
    if info["device"] == "cuda":
        try:
            import torch

            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
        except Exception:
            pass
    if not getattr(cfg.hardware, "auto_batch", True):
        return info
    vram = info["vram_gb"]
    if info["device"] == "cpu":
        cfg.hardware.encode_batch_size = 8
        cfg.biencoder.mini_batch_size = 4
        cfg.biencoder.effective_batch_size = 64
        cfg.cross_encoder.batch_size = 2
    elif vram >= 40:
        cfg.hardware.encode_batch_size = 256
        cfg.biencoder.mini_batch_size = 64
        cfg.biencoder.effective_batch_size = 1024
        cfg.cross_encoder.batch_size = 16
    elif vram >= 20:
        # L4 / 3090 / A10 class (24 GB). Qwen3-0.6B @ 128 tokens fits the
        # top of the spec range: mini 64, InfoNCE 1024, encode 256.
        cfg.hardware.encode_batch_size = 256
        cfg.biencoder.mini_batch_size = 64
        cfg.biencoder.effective_batch_size = 1024
        cfg.cross_encoder.batch_size = 16
    elif vram >= 14:
        cfg.hardware.encode_batch_size = 64
        cfg.biencoder.mini_batch_size = 16
        cfg.biencoder.effective_batch_size = 256
        cfg.cross_encoder.batch_size = 8
    else:
        cfg.hardware.encode_batch_size = 16
        cfg.biencoder.mini_batch_size = 8
        cfg.biencoder.effective_batch_size = 128
        cfg.cross_encoder.batch_size = 4
    info["encode_batch_size"] = cfg.hardware.encode_batch_size
    info["mini_batch_size"] = cfg.biencoder.mini_batch_size
    info["effective_batch_size"] = cfg.biencoder.effective_batch_size
    info["ce_batch_size"] = cfg.cross_encoder.batch_size
    return info
