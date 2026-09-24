"""jina/Qwen3 cross-encoder: last-token hidden → linear → sigmoid, BCE, LoRA r=16.

Input (same prefix on the whole sequence):
    Document: Record A: {name_a} | {addr_a} | {country_a}
    Record B: {name_b} | {addr_b} | {country_b}

Only the top-N candidates per S1 (by p_gbm) are scored at inference.
"""

from __future__ import annotations

import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from tqdm import tqdm

from src.embeddings import load_sentence_transformer, get_auto_model
from src.logging_utils import write_json


def pair_text(prefix: str, a: pd.Series, b: pd.Series) -> str:
    ra = f"Record A: {a['name_exp']} | {a['address_exp']} | {a['country_raw']}"
    rb = f"Record B: {b['name_exp']} | {b['address_exp']} | {b['country_raw']}"
    return f"{prefix}{ra}\n{rb}"


class CrossEncoderHead(nn.Module):
    def __init__(self, hidden: int):
        super().__init__()
        self.proj = nn.Linear(hidden, 1)

    def forward(self, last_hidden: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        idx = attention_mask.sum(dim=1) - 1
        pooled = last_hidden[torch.arange(last_hidden.size(0), device=last_hidden.device), idx]
        return self.proj(pooled).squeeze(-1)


def _attach_lora_ce(auto, cfg):
    from peft import LoraConfig, get_peft_model, TaskType

    lcfg = LoraConfig(
        r=int(cfg.cross_encoder.lora_r),
        lora_alpha=int(cfg.cross_encoder.lora_alpha),
        lora_dropout=float(cfg.cross_encoder.lora_dropout),
        target_modules=list(cfg.biencoder.target_modules),
        bias="none",
        task_type=TaskType.FEATURE_EXTRACTION,
    )
    return get_peft_model(auto, lcfg)


class JinaCrossEncoder(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        st = load_sentence_transformer(cfg)
        auto = get_auto_model(st)
        self.tokenizer = st.tokenizer
        self.backbone = _attach_lora_ce(auto, cfg)
        hidden = getattr(auto.config, "hidden_size", None) or getattr(auto.config, "d_model", 1024)
        self.head = CrossEncoderHead(int(hidden))
        self.max_length = int(cfg.cross_encoder.max_length)

    def forward(self, texts: list[str], device: torch.device) -> torch.Tensor:
        enc = self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        )
        enc = {k: v.to(device) for k, v in enc.items()}
        out = self.backbone(**enc, return_dict=True)
        hidden = out.last_hidden_state
        return self.head(hidden, enc["attention_mask"])


def _bce(logits: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    return F.binary_cross_entropy_with_logits(logits, y)


def _focal(logits: torch.Tensor, y: torch.Tensor, gamma: float) -> torch.Tensor:
    p = torch.sigmoid(logits)
    pt = p * y + (1 - p) * (1 - y)
    w = (1 - pt) ** gamma
    return (w * F.binary_cross_entropy_with_logits(logits, y, reduction="none")).mean()


def build_ce_texts(
    pairs: pd.DataFrame,
    s1: pd.DataFrame,
    gallery: pd.DataFrame,
    prefix: str,
    swap: bool,
    seed: int,
) -> tuple[list[str], np.ndarray]:
    rng = random.Random(seed)
    s1_map = s1.set_index("entity_id", drop=False)
    g_map = gallery.set_index("entity_id", drop=False)
    texts, labels = [], []
    for row in pairs.itertuples(index=False):
        a = s1_map.loc[row.s1_id]
        b = g_map.loc[row.cand_id]
        if swap and rng.random() < 0.5:
            a, b = b, a
        texts.append(pair_text(prefix, a, b))
        labels.append(int(getattr(row, "label", 0)))
    return texts, np.asarray(labels, dtype=np.float32)


def train_crossencoder(
    cfg,
    pairs: pd.DataFrame,
    s1: pd.DataFrame,
    gallery: pd.DataFrame,
    out_dir: Path,
) -> dict:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = JinaCrossEncoder(cfg).to(device)
    # keep natural ratio or subsample to 1:8, keep hard negs
    labelled = pairs.dropna(subset=["label"]).copy()
    pos = labelled[labelled["label"] == 1]
    neg = labelled[labelled["label"] == 0]
    ratio = float(cfg.cross_encoder.negative_sample_ratio)
    n_neg = min(len(neg), int(len(pos) * ratio))
    if "house_num_mismatch" in neg.columns:
        hard = neg[neg["house_num_mismatch"] == 0]
        easy = neg[neg["house_num_mismatch"] == 1]
        n_easy = max(n_neg - len(hard), 0)
        easy = easy.sample(n=min(n_easy, len(easy)), random_state=int(cfg.seed)) if n_easy else easy.iloc[0:0]
        train_df = pd.concat([pos, hard, easy], ignore_index=True)
    else:
        neg = neg.sample(n=n_neg, random_state=int(cfg.seed)) if n_neg < len(neg) else neg
        train_df = pd.concat([pos, neg], ignore_index=True)
    train_df = train_df.sample(frac=1.0, random_state=int(cfg.seed)).reset_index(drop=True)

    texts, y = build_ce_texts(
        train_df, s1, gallery, cfg.backbone.prefix,
        bool(cfg.cross_encoder.swap_order), int(cfg.seed),
    )
    opt = AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=float(cfg.cross_encoder.lr),
    )
    bs = int(cfg.cross_encoder.batch_size)
    use_focal = str(cfg.cross_encoder.loss) == "focal"
    history = []
    model.train()
    for epoch in range(int(cfg.cross_encoder.epochs)):
        running = 0.0
        n = 0
        for start in tqdm(range(0, len(texts), bs), desc=f"ce epoch {epoch+1}"):
            batch_t = texts[start : start + bs]
            batch_y = torch.tensor(y[start : start + bs], device=device)
            ctx = torch.autocast(device_type="cuda", dtype=torch.bfloat16) if device.type == "cuda" else torch.nullcontext()
            with ctx:
                logits = model(batch_t, device)
                loss = _focal(logits, batch_y, float(cfg.cross_encoder.focal_gamma)) if use_focal else _bce(logits, batch_y)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            running += float(loss.detach())
            n += 1
        history.append({"epoch": epoch + 1, "loss": running / max(n, 1)})

    out_dir.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "head": model.head.state_dict(),
            "backbone": model.backbone.state_dict(),
        },
        out_dir / "crossencoder.pt",
    )
    model.backbone.save_pretrained(out_dir / "adapter")
    metrics = {"history": history, "n_train": int(len(train_df)), "pos": int(len(pos))}
    write_json(Path(cfg.paths.reports_dir) / "crossencoder.json", metrics)
    return {"model": model, "metrics": metrics}


@torch.no_grad()
def score_pairs_ce(
    model: JinaCrossEncoder,
    pairs: pd.DataFrame,
    s1: pd.DataFrame,
    gallery: pd.DataFrame,
    prefix: str,
    batch_size: int,
) -> np.ndarray:
    device = next(model.parameters()).device
    model.eval()
    texts, _ = build_ce_texts(pairs, s1, gallery, prefix, swap=False, seed=0)
    probs = []
    for start in range(0, len(texts), batch_size):
        logits = model(texts[start : start + batch_size], device)
        probs.append(torch.sigmoid(logits).float().cpu().numpy())
    return np.concatenate(probs) if probs else np.zeros(0, dtype=np.float32)


def apply_ce_topn(
    model: JinaCrossEncoder,
    pairs: pd.DataFrame,
    s1: pd.DataFrame,
    gallery: pd.DataFrame,
    cfg,
    score_col: str = "p_gbm",
) -> pd.DataFrame:
    df = pairs.copy()
    df["p_ce"] = np.nan
    topn = int(cfg.cross_encoder.top_n)
    take_idx = []
    for s1_id, g in df.groupby("s1_id"):
        take_idx.extend(g.sort_values(score_col, ascending=False).head(topn).index.tolist())
    sub = df.loc[take_idx]
    if sub.empty:
        return df
    scores = score_pairs_ce(
        model, sub, s1, gallery, cfg.backbone.prefix, int(cfg.cross_encoder.batch_size)
    )
    df.loc[sub.index, "p_ce"] = scores
    return df


def load_crossencoder(cfg, ckpt_dir: Path) -> JinaCrossEncoder:
    model = JinaCrossEncoder(cfg)
    blob = torch.load(ckpt_dir / "crossencoder.pt", map_location="cpu")
    model.head.load_state_dict(blob["head"])
    try:
        model.backbone.load_state_dict(blob["backbone"])
    except Exception:
        pass
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return model.to(device)
