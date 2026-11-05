"""Device utilities for NVIDIA GPU acceleration and VRAM monitoring."""
import gc
from typing import Dict, Any
import torch
import torch.version


def get_device(prefer_gpu: bool = True) -> torch.device:
    """Return the NVIDIA CUDA device if available, otherwise CPU."""
    if prefer_gpu and torch.cuda.is_available():
        return torch.device("cuda:0")
    return torch.device("cpu")


def get_gpu_info() -> Dict[str, Any]:
    """Retrieve detailed hardware information for the active GPU."""
    if not torch.cuda.is_available():
        return {"cuda_available": False, "device": "CPU"}

    dev = 0
    props = torch.cuda.get_device_properties(dev)
    total_mem = props.total_memory / (1024 ** 3)
    allocated_mem = torch.cuda.memory_allocated(dev) / (1024 ** 3)
    reserved_mem = torch.cuda.memory_reserved(dev) / (1024 ** 3)

    return {
        "cuda_available": True,
        "device_name": props.name,
        "cuda_version": torch.version.cuda,
        "total_memory_gb": round(total_mem, 2),
        "allocated_memory_gb": round(allocated_mem, 3),
        "reserved_memory_gb": round(reserved_mem, 3),
        "free_memory_gb": round(total_mem - reserved_mem, 3)
    }


def clear_gpu_memory():
    """Clear cached memory from GPU to prevent Out-Of-Memory errors on 4GB VRAM."""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def print_gpu_status(prefix: str = ""):
    """Print current GPU allocation status."""
    info = get_gpu_info()
    if info.get("cuda_available"):
        print(
            f"[GPU {prefix}] {info['device_name']} | "
            f"VRAM Allocated: {info['allocated_memory_gb']:.2f}GB / "
            f"Reserved: {info['reserved_mem_gb'] if 'reserved_mem_gb' in info else info['reserved_memory_gb']:.2f}GB / "
            f"Total: {info['total_memory_gb']:.2f}GB"
        )
    else:
        print("[GPU] CUDA not available, using CPU.")


if __name__ == "__main__":
    import json
    print(json.dumps(get_gpu_info(), indent=2))
