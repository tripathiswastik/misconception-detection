"""Unit tests for HandwritingReaderEngine and Vision OCR."""
import tempfile
from pathlib import Path
import numpy as np
from PIL import Image
import torch

from src.reader.ocr_engine import (
    HandwritingReaderEngine,
    HandwritingVisionEncoder,
    preprocess_handwriting_image
)
from src.utils.device import get_device


def test_preprocess_handwriting_image():
    # Create synthetic test image
    synthetic = Image.fromarray((np.random.rand(200, 200, 3) * 255).astype(np.uint8))
    preprocessed = preprocess_handwriting_image(synthetic, target_size=(256, 256))
    assert preprocessed.size == (256, 256)
    assert preprocessed.mode == "RGB"


def test_vision_encoder_forward():
    device = get_device()
    encoder = HandwritingVisionEncoder(embedding_dim=64).to(device)
    encoder.eval()

    sample_tensor = torch.randn(1, 3, 256, 256).to(device)
    with torch.no_grad():
        out = encoder(sample_tensor)

    assert out.shape == (1, 64)
    if device.type == "cuda":
        assert out.is_cuda


def test_reader_transcribe_and_cache():
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_path = Path(tmpdir)
        cache_file = tmp_path / "cache.json"

        # Create dummy image
        img_file = tmp_path / "sample.jpeg"
        dummy_img = Image.fromarray(np.ones((300, 300, 3), dtype=np.uint8) * 255)
        dummy_img.save(img_file)

        reader = HandwritingReaderEngine(cache_path=cache_file)
        text = reader.transcribe_image(img_file, trace_id="trace_test_01")
        assert "L1 |" in text

        # Test cache hit
        assert "trace_test_01" in reader.cache
        cached_text = reader.transcribe_image(img_file, trace_id="trace_test_01")
        assert cached_text == text
