"""Unit tests for GPU device management."""
import torch
from src.utils.device import get_device, get_gpu_info, clear_gpu_memory


def test_device_selection():
    device = get_device()
    assert isinstance(device, torch.device)
    if torch.cuda.is_available():
        assert device.type == "cuda"
    else:
        assert device.type == "cpu"


def test_gpu_info_retrieval():
    info = get_gpu_info()
    assert isinstance(info, dict)
    assert "cuda_available" in info
    if info["cuda_available"]:
        assert "GeForce" in info["device_name"] or "NVIDIA" in info["device_name"]
        assert info["total_memory_gb"] > 0


def test_clear_gpu_memory():
    # Smoke test that clear_gpu_memory executes cleanly
    clear_gpu_memory()
