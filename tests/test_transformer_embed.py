"""Unit tests for MultilingualMathEmbedder on GPU."""
import numpy as np
from src.data.taxonomy import TaxonomyManager
from src.diagnoser.transformer_embed import MultilingualMathEmbedder


def test_multilingual_embedder_gpu():
    embedder = MultilingualMathEmbedder(use_gpu=True)
    assert embedder.device.type in ("cuda", "cpu")
    assert embedder.taxonomy_embeddings is not None
    assert embedder.taxonomy_embeddings.shape[1] == 384
    assert len(embedder.taxonomy_codes) >= 76

    # Test single query encoding
    queries = [
        "Sebuah lingkaran memiliki diameter 14 cm. Berapakah luasnya?",
        "Hitunglah hasil dari 1/4 + 2/4"
    ]
    embeddings = embedder.encode_texts(queries, batch_size=2)
    assert embeddings.shape == (2, 384)
    # L2 normalized
    assert np.allclose(np.linalg.norm(embeddings, axis=1), 1.0, atol=1e-3)

    # Test semantic similarity computation
    sim_matrix = embedder.compute_semantic_similarity(queries, batch_size=2)
    assert sim_matrix.shape == (2, len(embedder.taxonomy_codes))
    assert not np.isnan(sim_matrix).any()
