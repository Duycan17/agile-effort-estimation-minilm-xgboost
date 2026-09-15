"""Frozen transformer embeddings with content-addressed caching."""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np


@dataclass(frozen=True)
class EmbeddingConfig:
    model_name: str = "sentence-transformers/all-MiniLM-L6-v2"
    revision: str = "main"
    max_length: int = 256
    batch_size: int = 16
    device: str = "auto"
    normalize: bool = True
    seed: int = 42


def _cache_key(row_ids: np.ndarray, texts: list[str], config: EmbeddingConfig) -> str:
    digest = hashlib.sha256()
    digest.update(json.dumps(asdict(config), sort_keys=True).encode())
    for row_id, value in zip(row_ids, texts, strict=True):
        digest.update(str(row_id).encode())
        digest.update(b"\0")
        digest.update(value.encode("utf-8", errors="replace"))
        digest.update(b"\0")
    return digest.hexdigest()


def _resolve_device(requested: str, torch: Any) -> str:
    if requested != "auto":
        return requested
    if torch.cuda.is_available():
        return "cuda"
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def load_or_create_embeddings(
    row_ids: np.ndarray,
    texts: list[str],
    config: EmbeddingConfig,
    cache_dir: Path,
) -> tuple[np.ndarray, dict[str, Any]]:
    import torch
    from transformers import AutoModel, AutoTokenizer

    cache_dir.mkdir(parents=True, exist_ok=True)
    key = _cache_key(row_ids, texts, config)
    cache_path = cache_dir / f"{key}.npz"
    metadata_path = cache_dir / f"{key}.json"
    if cache_path.exists() and metadata_path.exists():
        with np.load(cache_path) as archive:
            embeddings = archive["embeddings"]
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        metadata["cache_hit"] = True
        return embeddings, metadata

    torch.manual_seed(config.seed)
    device = _resolve_device(config.device, torch)
    tokenizer = AutoTokenizer.from_pretrained(
        config.model_name, revision=config.revision
    )
    model = AutoModel.from_pretrained(config.model_name, revision=config.revision)
    model.to(device)
    model.eval()

    batches: list[np.ndarray] = []
    with torch.inference_mode():
        for start in range(0, len(texts), config.batch_size):
            batch = texts[start : start + config.batch_size]
            tokens = tokenizer(
                batch,
                padding=True,
                truncation=True,
                max_length=config.max_length,
                return_tensors="pt",
            )
            tokens = {name: value.to(device) for name, value in tokens.items()}
            hidden = model(**tokens).last_hidden_state
            mask = tokens["attention_mask"].unsqueeze(-1).to(hidden.dtype)
            pooled = (hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1)
            if config.normalize:
                pooled = torch.nn.functional.normalize(pooled, p=2, dim=1)
            batches.append(pooled.cpu().numpy().astype(np.float32))

    embeddings = np.concatenate(batches, axis=0)
    resolved_commit = getattr(model.config, "_commit_hash", None)
    metadata = {
        **asdict(config),
        "resolved_device": device,
        "resolved_model_commit": resolved_commit,
        "rows": len(texts),
        "embedding_dimensions": int(embeddings.shape[1]),
        "pooling": "attention_mask_mean",
        "target_used": False,
        "cross_row_statistics_fitted": False,
        "cache_key": key,
        "cache_hit": False,
    }
    np.savez_compressed(cache_path, embeddings=embeddings)
    metadata_path.write_text(
        json.dumps(metadata, indent=2, sort_keys=True), encoding="utf-8"
    )
    return embeddings, metadata
