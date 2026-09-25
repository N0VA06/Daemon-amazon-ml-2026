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
    """Detect GPU and enable TF32. Batch sizes always come from the YAML."""
    info = detect_device()
    if info["device"] == "cuda":
        try:
            import torch

            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
        except Exception:
            pass
    info["auto_batch"] = bool(getattr(cfg.hardware, "auto_batch", False))
    info["encode_batch_size"] = int(cfg.hardware.encode_batch_size)
    info["mini_batch_size"] = int(cfg.biencoder.mini_batch_size)
    info["effective_batch_size"] = int(cfg.biencoder.effective_batch_size)
    info["ce_batch_size"] = int(cfg.cross_encoder.batch_size)
    return info
