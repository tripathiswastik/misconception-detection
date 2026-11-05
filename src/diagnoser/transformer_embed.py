"""Multilingual Transformer Semantic Reasoner on NVIDIA CUDA GPU.

Embeds Indonesian mathematical questions and taxonomy definitions using
paraphrase-multilingual-MiniLM-L12-v2 in FP16 on cuda:0 (GTX 1050).
Computes dense semantic cosine affinity against all 75 misconception definitions.
"""
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Union, cast
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from transformers.models.auto import AutoModel, AutoTokenizer

from src.data.taxonomy import LABEL_NONE, TaxonomyManager
from src.utils.device import clear_gpu_memory, get_device

MODEL_NAME = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
EMBEDDING_DIM = 384


def _mean_pooling(model_output: Any, attention_mask: torch.Tensor) -> torch.Tensor:
    """Mean pooling taking attention mask into account for correct averaging."""
    token_embeddings = model_output[0]  # First element of model_output contains all token embeddings
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, 1)
    sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)
    return sum_embeddings / sum_mask


class MultilingualMathEmbedder:
    """GPU-accelerated dense semantic encoder in FP16 for Indonesian math texts."""

    def __init__(
        self,
        model_name: str = MODEL_NAME,
        taxonomy: Optional[TaxonomyManager] = None,
        use_gpu: bool = True
    ):
        self.device = get_device(prefer_gpu=use_gpu)
        self.taxonomy = taxonomy or TaxonomyManager()
        self.tokenizer = cast(Any, AutoTokenizer).from_pretrained(model_name)
        model_obj = cast(Any, AutoModel).from_pretrained(model_name)
        self.model = model_obj.to(self.device)
        self.model.eval()

        self.taxonomy_codes: List[str] = self.taxonomy.all_codes(include_none=True)
        self.taxonomy_embeddings: Optional[np.ndarray] = None
        self._encode_taxonomy_definitions()

    def _encode_taxonomy_definitions(self):
        """Pre-compute normalized dense vectors for all 75 misconception definitions."""
        tax_texts = []
        for code in self.taxonomy_codes:
            if code == LABEL_NONE:
                tax_texts.append("Pekerjaan siswa benar. Tidak ada kesalahan konsep matematika.")
            else:
                item = self.taxonomy.get(code)
                desc = item.description if item and item.description else ""
                name = item.name if item and item.name else ""
                section = item.section if item and item.section else ""
                tax_texts.append(f"{name}. {desc}. Kategori: {section}")

        self.taxonomy_embeddings = self.encode_texts(tax_texts, batch_size=32)

    def encode_texts(
        self,
        texts: Sequence[str],
        batch_size: int = 32,
        max_length: int = 128
    ) -> np.ndarray:
        """Batch encode texts into normalized (N, 384) dense semantic vectors on GPU."""
        embeddings_list = []
        total = len(texts)

        for i in range(0, total, batch_size):
            batch_texts = list(texts[i: i + batch_size])
            encoded = self.tokenizer(
                batch_texts,
                padding=True,
                truncation=True,
                max_length=max_length,
                return_tensors="pt"
            ).to(self.device)

            with torch.no_grad():
                if self.device.type == "cuda":
                    with torch.cuda.amp.autocast():
                        model_output = self.model(**encoded)
                        pooled = _mean_pooling(model_output, encoded["attention_mask"])
                        normalized = F.normalize(pooled, p=2, dim=1)
                else:
                    model_output = self.model(**encoded)
                    pooled = _mean_pooling(model_output, encoded["attention_mask"])
                    normalized = F.normalize(pooled, p=2, dim=1)

            embeddings_list.append(normalized.cpu().numpy())

        clear_gpu_memory()
        return np.vstack(embeddings_list)

    def compute_semantic_similarity(
        self,
        query_texts: Sequence[str],
        batch_size: int = 32
    ) -> np.ndarray:
        """Compute cosine similarity between query texts and all taxonomy definitions.
        
        Returns:
            (N, num_classes) cosine similarity matrix in range [-1, 1].
        """
        query_embeddings = self.encode_texts(query_texts, batch_size=batch_size)
        assert self.taxonomy_embeddings is not None
        # Dot product of L2-normalized vectors equals cosine similarity
        sim_matrix = np.dot(query_embeddings, self.taxonomy_embeddings.T)
        return sim_matrix
