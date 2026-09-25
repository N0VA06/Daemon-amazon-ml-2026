"""LoRA bi-encoder fine-tune: Phase A InfoNCE (GradCache) + Phase B CoSENT.

Backbone is whatever `cfg.backbone` points at (jina or snowflake). Prefixes are
already on the record texts. LoRA keeps multilingual knowledge (France unseen).
"""

from __future__ import annotations

import math
import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.optim import AdamW
from tqdm import tqdm

from src.augment import augment_record, make_hard_negative
from src.embeddings import encode_texts, get_auto_model, load_sentence_transformer
from src.evaluate import recall_at_k_curve
from src.logging_utils import LOG, write_json
from src.normalize import build_record_text
from src.sampler import EntityAwareBatchSampler


def _unwrap_model(st_model):
    return get_auto_model(st_model)


def attach_lora(st_model, cfg) -> object:
    if cfg.biencoder.full_finetune:
        return st_model
    from peft import LoraConfig, get_peft_model, TaskType

    auto = _unwrap_model(st_model)
    lcfg = LoraConfig(
        r=int(cfg.biencoder.lora_r),
        lora_alpha=int(cfg.biencoder.lora_alpha),
        lora_dropout=float(cfg.biencoder.lora_dropout),
        target_modules=list(cfg.biencoder.target_modules),
        bias="none",
        task_type=TaskType.FEATURE_EXTRACTION,
    )
    try:
        peft_model = get_peft_model(auto, lcfg)
    except ValueError:
        # some decoders register targets under different names
        lcfg.target_modules = list(cfg.biencoder.target_modules)
        peft_model = get_peft_model(auto, lcfg)
    # splice back
    try:
        st_model[0].auto_model = peft_model
    except Exception:
        st_model._first_module().auto_model = peft_model
    if cfg.biencoder.gradient_checkpointing:
        try:
            peft_model.gradient_checkpointing_enable()
        except Exception:
            pass
    return st_model


def save_lora(st_model, path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)
    auto = _unwrap_model(st_model)
    if hasattr(auto, "save_pretrained"):
        auto.save_pretrained(path)
    else:
        st_model.save(str(path))


def merge_lora(st_model, out_dir: Path) -> Path:
    auto = _unwrap_model(st_model)
    if hasattr(auto, "merge_and_unload"):
        merged = auto.merge_and_unload()
        try:
            st_model[0].auto_model = merged
        except Exception:
            st_model._first_module().auto_model = merged
    out_dir.mkdir(parents=True, exist_ok=True)
    st_model.save(str(out_dir))
    return out_dir


def build_positive_pairs(
    s1: pd.DataFrame,
    gallery: pd.DataFrame,
    gt: pd.DataFrame,
    prefix: str,
    template: str,
    max_pairs: int,
    seed: int,
) -> pd.DataFrame:
    """Symmetric S1↔S2/S3 plus S2↔S3 that share an S1."""
    rec = pd.concat(
        [
            s1[["entity_id", "name_exp", "address_exp", "country_raw", "text_combined"]],
            gallery[["entity_id", "name_exp", "address_exp", "country_raw", "text_combined"]],
        ],
        ignore_index=True,
    ).drop_duplicates("entity_id").set_index("entity_id")

    rows = []
    for row in gt.itertuples(index=False):
        if not row.matched_list:
            continue
        if row.source1_entity_id not in rec.index:
            continue
        matches = [m for m in row.matched_list if m in rec.index]
        for m in matches:
            rows.append((row.source1_entity_id, m, row.source1_entity_id))
            rows.append((m, row.source1_entity_id, row.source1_entity_id))
        for i in range(len(matches)):
            for j in range(i + 1, len(matches)):
                rows.append((matches[i], matches[j], row.source1_entity_id))
                rows.append((matches[j], matches[i], row.source1_entity_id))
    df = pd.DataFrame(rows, columns=["anchor_id", "positive_id", "entity_id"])
    if len(df) > max_pairs:
        df = df.sample(n=max_pairs, random_state=seed)
    return df.reset_index(drop=True)


def select_mine_gallery(
    gallery: pd.DataFrame,
    pairs: pd.DataFrame,
    max_n: int,
    seed: int,
) -> pd.DataFrame:
    """Positives that appear in train pairs, plus a random slice. Never the full S2/S3."""
    needed = set(pairs["positive_id"].astype(str)) | set(pairs["anchor_id"].astype(str))
    gal_ids = gallery["entity_id"].astype(str)
    must = gallery.loc[gal_ids.isin(needed)]
    rest = gallery.loc[~gal_ids.isin(needed)]
    extra_n = max(0, int(max_n) - len(must))
    if extra_n and len(rest) > extra_n:
        rest = rest.sample(n=extra_n, random_state=seed)
    elif extra_n <= 0:
        rest = rest.iloc[0:0]
    out = pd.concat([must, rest], ignore_index=True).drop_duplicates("entity_id")
    LOG.info(
        "hard-neg mine gallery %s  (must=%s extra=%s; full S2/S3 was %s — not encoded)",
        f"{len(out):,}", f"{len(must):,}", f"{len(out) - len(must):,}", f"{len(gallery):,}",
    )
    return out.reset_index(drop=True)


def select_mine_s1(s1: pd.DataFrame, pairs: pd.DataFrame) -> pd.DataFrame:
    ids = set(pairs["anchor_id"].astype(str)) | set(pairs["entity_id"].astype(str))
    out = s1[s1["entity_id"].astype(str).isin(ids)].reset_index(drop=True)
    LOG.info("hard-neg mine S1 %s / %s (pair anchors only)", f"{len(out):,}", f"{len(s1):,}")
    return out


def mine_hard_negatives(
    pairs: pd.DataFrame,
    s1: pd.DataFrame,
    gallery: pd.DataFrame,
    emb_s1: np.ndarray | None,
    emb_gal: np.ndarray | None,
    truth_map: dict[str, set[str]],
    k: int,
    per_anchor: int,
    seed: int,
) -> dict[str, list[str]]:
    """Top-k same-country non-matches from a zero-shot / Phase-A index."""
    rng = random.Random(seed)
    gal_ids = gallery["entity_id"].astype(str).to_numpy()
    gal_country = gallery["country"].astype(str).to_numpy()
    s1_ids = s1["entity_id"].astype(str).to_numpy()
    s1_country = s1["country"].astype(str).to_numpy()
    s1_pos = {eid: i for i, eid in enumerate(s1_ids)}
    hard: dict[str, list[str]] = {}
    if emb_s1 is None or emb_gal is None:
        return hard
    # sample anchors to keep mining tractable
    anchors = list(dict.fromkeys(pairs["anchor_id"].tolist()))
    rng.shuffle(anchors)
    anchors = anchors[: min(len(anchors), 50000)]
    country_index: dict[str, np.ndarray] = {}
    for c in np.unique(gal_country):
        country_index[str(c)] = np.flatnonzero(gal_country == c)

    for aid in anchors:
        if aid not in s1_pos:
            continue
        i = s1_pos[aid]
        c = s1_country[i]
        pool = country_index.get(c)
        if pool is None or len(pool) == 0:
            continue
        # random subset of the country gallery
        if len(pool) > 4000:
            pool = pool[rng.sample(range(len(pool)), 4000)]
        sims = emb_gal[pool] @ emb_s1[i]
        truth = truth_map.get(aid, set())
        order = np.argsort(-sims)
        picked = []
        for j in order:
            cid = gal_ids[pool[j]]
            if cid in truth or cid == aid:
                continue
            picked.append(str(cid))
            if len(picked) >= k:
                break
        hard[aid] = picked[:per_anchor] if picked else []
    return hard


def _tokenize(st_model, texts: list[str], device: torch.device, max_len: int):
    tok = st_model.tokenizer
    enc = tok(
        texts,
        padding=True,
        truncation=True,
        max_length=max_len,
        return_tensors="pt",
    )
    return {k: v.to(device) for k, v in enc.items()}


def _forward_norm(st_model, features: dict) -> torch.Tensor:
    """Last-token / ST pooling then L2-normalise."""
    out = st_model(features)
    if "sentence_embedding" in out:
        emb = out["sentence_embedding"]
    else:
        hidden = out["token_embeddings"]
        mask = features["attention_mask"]
        idx = mask.sum(dim=1) - 1
        emb = hidden[torch.arange(hidden.size(0), device=hidden.device), idx]
    return F.normalize(emb, p=2, dim=1)


def infonce_loss(anchor: torch.Tensor, positive: torch.Tensor, scale: float) -> torch.Tensor:
    """L = −log( exp(cos(a,p)/τ) / Σ_c exp(cos(a,c)/τ) ) with in-batch candidates."""
    logits = (anchor @ positive.T) * scale
    labels = torch.arange(anchor.size(0), device=anchor.device)
    return F.cross_entropy(logits, labels)


def cosent_loss(cosines: torch.Tensor, labels: torch.Tensor, lam: float) -> torch.Tensor:
    """L = log(1 + Σ_{y_i>y_j} exp(λ · (cos_j − cos_i)))."""
    pos = cosines[labels > 0.5]
    neg = cosines[labels < 0.5]
    if pos.numel() == 0 or neg.numel() == 0:
        return cosines.new_zeros(())
    # pair every pos with every neg
    diff = neg.unsqueeze(0) - pos.unsqueeze(1)  # (P, N)
    return torch.log1p(torch.exp(lam * diff).sum())


def online_contrastive(cosines: torch.Tensor, labels: torch.Tensor, margin: float) -> torch.Tensor:
    pos = cosines[labels > 0.5]
    neg = cosines[labels < 0.5]
    loss = cosines.new_zeros(())
    if pos.numel():
        loss = loss + ((1.0 - pos) ** 2).mean()
    if neg.numel():
        loss = loss + (F.relu(neg - (1.0 - margin)) ** 2).mean()
    return loss


class PairDataset:
    def __init__(
        self,
        pairs: pd.DataFrame,
        rec: pd.DataFrame,
        hard: dict[str, list[str]],
        prefix: str,
        template: str,
        augment: bool,
        seed: int,
    ):
        self.pairs = pairs.reset_index(drop=True)
        self.rec = rec
        self.hard = hard
        self.prefix = prefix
        self.template = template
        self.augment = augment
        self.rng = random.Random(seed)

    def __len__(self) -> int:
        return len(self.pairs)

    def text_of(self, eid: str, do_aug: bool) -> str:
        r = self.rec.loc[eid]
        name, addr = str(r.name_exp), str(r.address_exp)
        if do_aug:
            name, addr = augment_record(name, addr, self.rng, p=0.5)
        return build_record_text(name, addr, str(r.country_raw), self.prefix, self.template)

    def item(self, i: int) -> dict:
        row = self.pairs.iloc[i]
        aug = self.augment
        negs = self.hard.get(row.anchor_id, [])
        return {
            "anchor": self.text_of(row.anchor_id, aug),
            "positive": self.text_of(row.positive_id, aug),
            "negatives": [self.text_of(n, False) for n in negs if n in self.rec.index],
            "entity_id": row.entity_id,
            "label": 1,
        }


def _device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _cap_eval_gallery(s1: pd.DataFrame, gallery: pd.DataFrame, truth_map, max_n: int, seed: int) -> pd.DataFrame:
    if max_n <= 0 or len(gallery) <= max_n:
        return gallery
    must: set[str] = set()
    for sid in s1["entity_id"].astype(str):
        must |= set(truth_map.get(sid, set()))
    gal_ids = gallery["entity_id"].astype(str)
    keep = gallery.loc[gal_ids.isin(must)]
    rest = gallery.loc[~gal_ids.isin(must)]
    extra = max(0, max_n - len(keep))
    if extra and len(rest) > extra:
        rest = rest.sample(n=extra, random_state=seed)
    out = pd.concat([keep, rest], ignore_index=True).drop_duplicates("entity_id")
    LOG.info("eval gallery capped %s → %s (truth matches kept)", f"{len(gallery):,}", f"{len(out):,}")
    return out


def _eval_retrieval(st_model, s1: pd.DataFrame, gallery: pd.DataFrame, truth_map, cfg, ks) -> dict:
    batch = int(cfg.hardware.encode_batch_size)
    if len(s1) > 4000:
        idx = np.linspace(0, len(s1) - 1, 4000).astype(int)
        s1 = s1.iloc[idx]
    cap = int(getattr(cfg.biencoder, "eval_gallery_size", 50000) or 50000)
    gallery = _cap_eval_gallery(s1, gallery, truth_map, cap, int(cfg.seed))
    q = encode_texts(st_model, s1["text_combined"].tolist(), batch, True, False)
    g = encode_texts(st_model, gallery["text_combined"].tolist(), batch, True, False)
    scores = q @ g.T
    gal_ids = gallery["entity_id"].astype(str).to_numpy()
    ranked = {}
    kmax = max(ks)
    for i, s1_id in enumerate(s1["entity_id"].astype(str)):
        top = np.argpartition(-scores[i], kth=min(kmax, scores.shape[1] - 1))[:kmax]
        order = top[np.argsort(-scores[i, top])]
        ranked[s1_id] = gal_ids[order].tolist()
    sub_truth = {s: truth_map.get(s, set()) for s in ranked}
    return {"recall_at_k": recall_at_k_curve(ranked, sub_truth, ks)}


def train_biencoder(
    cfg,
    s1: pd.DataFrame,
    gallery: pd.DataFrame,
    gt: pd.DataFrame,
    truth_map: dict[str, set[str]],
    zs_s1: np.ndarray | None,
    zs_gal: np.ndarray | None,
    out_dir: Path,
    val_s1: pd.DataFrame | None = None,
    val_gallery: pd.DataFrame | None = None,
) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)
    device = _device()
    pairs = build_positive_pairs(
        s1, gallery, gt, cfg.backbone.prefix, cfg.backbone.record_template,
        int(cfg.biencoder.max_train_pairs), int(cfg.seed),
    )
    mine_n = int(getattr(cfg.biencoder, "mine_gallery_size", 400000) or 400000)
    mine_gal = select_mine_gallery(gallery, pairs, mine_n, int(cfg.seed))
    mine_s1 = select_mine_s1(s1, pairs)
    rec = pd.concat(
        [
            s1[["entity_id", "name_exp", "address_exp", "country_raw", "text_combined"]],
            mine_gal[["entity_id", "name_exp", "address_exp", "country_raw", "text_combined"]],
        ],
        ignore_index=True,
    ).drop_duplicates("entity_id").set_index("entity_id")

    st_model = load_sentence_transformer(cfg, use_cache=False)
    LOG.info("biencoder  device=%s  positive pairs=%s  mine_gallery=%s  (full gallery texts=%s, not encoded)",
             device, f"{len(pairs):,}", f"{len(mine_gal):,}", f"{len(gallery):,}")
    if zs_s1 is None or zs_gal is None:
        LOG.info("encoding hard-neg mine subset only — not full S2/S3")
        zs_s1 = encode_texts(
            st_model, mine_s1["text_combined"].tolist(),
            int(cfg.hardware.encode_batch_size), True, True,
        )
        zs_gal = encode_texts(
            st_model, mine_gal["text_combined"].tolist(),
            int(cfg.hardware.encode_batch_size), True, True,
        )
    st_model = attach_lora(st_model, cfg)
    st_model.to(device)
    st_model.train()
    hard = mine_hard_negatives(
        pairs, mine_s1, mine_gal, zs_s1, zs_gal, truth_map,
        k=int(cfg.biencoder.max_hard_neg_pool),
        per_anchor=int(cfg.biencoder.hard_negatives_per_anchor),
        seed=int(cfg.seed),
    )

    # synthetic hard negatives (house-digit / W↔E / token swap) attached as extra texts
    rng = random.Random(int(cfg.seed))
    extra_neg_text: dict[int, str] = {}
    for i, row in pairs.iterrows():
        if rng.random() < 0.25 and row.anchor_id in rec.index:
            r = rec.loc[row.anchor_id]
            n, a = make_hard_negative(str(r.name_exp), str(r.address_exp), rng)
            extra_neg_text[i] = build_record_text(
                n, a, str(r.country_raw), cfg.backbone.prefix, cfg.backbone.record_template
            )

    ds = PairDataset(
        pairs, rec, hard, cfg.backbone.prefix, cfg.backbone.record_template, True, int(cfg.seed)
    )
    scale = float(cfg.biencoder.scale)
    mini = int(cfg.biencoder.mini_batch_size)
    eff = int(cfg.biencoder.effective_batch_size)
    max_len = int(cfg.backbone.max_seq_length)
    lr = float(cfg.biencoder.full_finetune_lr if cfg.biencoder.full_finetune else cfg.biencoder.lr)
    params = [p for p in st_model.parameters() if p.requires_grad]
    opt = AdamW(params, lr=lr)
    entity_ids = pairs["entity_id"].astype(str).tolist()

    def run_phase(epochs: int, remine: bool, aux: str | None, tag: str) -> list[dict]:
        nonlocal hard, ds
        if remine:
            LOG.info("phase-B remine: encode mine subset only (S1=%s gallery=%s), not full S2/S3",
                     f"{len(mine_s1):,}", f"{len(mine_gal):,}")
            st_model.eval()
            with torch.no_grad():
                s1_e = encode_texts(
                    st_model, mine_s1["text_combined"].tolist(),
                    int(cfg.hardware.encode_batch_size), True, True,
                )
                g_e = encode_texts(
                    st_model, mine_gal["text_combined"].tolist(),
                    int(cfg.hardware.encode_batch_size), True, True,
                )
            hard = mine_hard_negatives(
                pairs, mine_s1, mine_gal, s1_e, g_e, truth_map,
                k=int(cfg.biencoder.max_hard_neg_pool),
                per_anchor=int(cfg.biencoder.hard_negatives_per_anchor),
                seed=int(cfg.seed) + 7,
            )
            ds = PairDataset(
                pairs, rec, hard, cfg.backbone.prefix, cfg.backbone.record_template, True, int(cfg.seed)
            )
            st_model.train()

        history = []
        n_steps = max(len(ds) // eff, 1) * epochs
        warmup = max(int(n_steps * float(cfg.biencoder.warmup_ratio)), 1)
        step = 0
        best_r10 = -1.0
        for epoch in range(epochs):
            sampler = EntityAwareBatchSampler(entity_ids, eff, seed=int(cfg.seed) + epoch)
            if not cfg.biencoder.use_entity_aware_sampler:
                order = list(range(len(ds)))
                rng.shuffle(order)
                sampler = [order[i : i + eff] for i in range(0, len(order), eff)]
            running = 0.0
            nseen = 0
            for batch_idx in tqdm(list(sampler), desc=f"{tag} epoch {epoch+1}"):
                items = [ds.item(i) for i in batch_idx if i < len(ds)]
                if len(items) < 4:
                    continue
                # GradCache-style: encode mini-batches, InfoNCE on the full batch
                def encode_list(texts: list[str]) -> torch.Tensor:
                    chunks = []
                    for s in range(0, len(texts), mini):
                        feats = _tokenize(st_model, texts[s : s + mini], device, max_len)
                        use_amp = device.type == "cuda"
                        ctx = torch.autocast(device_type="cuda", dtype=torch.bfloat16) if use_amp else torch.nullcontext()
                        with ctx:
                            chunks.append(_forward_norm(st_model, feats))
                    return torch.cat(chunks, dim=0)

                anchors = encode_list([it["anchor"] for it in items])
                positives = encode_list([it["positive"] for it in items])
                loss = infonce_loss(anchors, positives, scale)
                # extra hard negatives as additional columns (optional)
                extra = []
                for it in items:
                    extra.extend(it["negatives"][:1])
                if extra:
                    neg_emb = encode_list(extra)
                    # pull each anchor away from its first hard neg if aligned
                    n_use = min(len(items), neg_emb.size(0))
                    hard_cos = (anchors[:n_use] * neg_emb[:n_use]).sum(-1)
                    loss = loss + F.relu(hard_cos - 0.3).mean()

                if aux == "cosent":
                    pos_cos = (anchors * positives).sum(-1)
                    # in-batch negatives: off-diagonal
                    all_cos = (anchors @ positives.T).reshape(-1)
                    lab = torch.eye(anchors.size(0), device=device).reshape(-1)
                    loss = loss + 0.5 * cosent_loss(all_cos, lab, float(cfg.biencoder.cosent_lambda))
                elif aux == "online_contrastive":
                    all_cos = (anchors @ positives.T).reshape(-1)
                    lab = torch.eye(anchors.size(0), device=device).reshape(-1)
                    loss = loss + 0.5 * online_contrastive(
                        all_cos, lab, float(cfg.biencoder.online_contrastive_margin)
                    )

                opt.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(params, 1.0)
                # warmup
                step += 1
                warm = min(1.0, step / warmup)
                for pg in opt.param_groups:
                    pg["lr"] = lr * warm
                opt.step()
                running += float(loss.detach())
                nseen += 1
            rec_m = {"epoch": epoch + 1, "loss": running / max(nseen, 1), "phase": tag}
            if val_s1 is not None and val_gallery is not None:
                st_model.eval()
                with torch.no_grad():
                    ev = _eval_retrieval(
                        st_model, val_s1, val_gallery, truth_map, cfg, list(cfg.biencoder.eval_ks)
                    )
                rec_m.update(ev)
                r10 = ev["recall_at_k"].get(10, 0.0)
                if r10 > best_r10:
                    best_r10 = r10
                    save_lora(st_model, out_dir / "best_adapter")
                st_model.train()
            history.append(rec_m)
        return history

    hist_a = []
    if int(cfg.biencoder.phase_a_epochs) > 0:
        hist_a = run_phase(int(cfg.biencoder.phase_a_epochs), remine=False, aux=None, tag="phase_a")
    hist_b = []
    if int(cfg.biencoder.phase_b_epochs) > 0:
        hist_b = run_phase(
            int(cfg.biencoder.phase_b_epochs),
            remine=True,
            aux=str(cfg.biencoder.phase_b_aux_loss),
            tag="phase_b",
        )
    save_lora(st_model, out_dir / "last_adapter")
    merged = merge_lora(st_model, out_dir / "merged")
    metrics = {"phase_a": hist_a, "phase_b": hist_b, "merged": str(merged), "n_pairs": int(len(pairs))}
    write_json(Path(cfg.paths.reports_dir) / "biencoder.json", metrics)
    return metrics
