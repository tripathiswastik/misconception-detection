import sys
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", line_buffering=True)

print("Starting check...", flush=True)
import torch
print("PyTorch loaded...", flush=True)
import transformers
print("Transformers loaded...", flush=True)
import PIL
print("PIL loaded...", flush=True)

print("Python Version:", sys.version, flush=True)
print("PyTorch Version:", torch.__version__, flush=True)
print("CUDA Available:", torch.cuda.is_available(), flush=True)
if torch.cuda.is_available():
    print("Device Name:", torch.cuda.get_device_name(0), flush=True)
    print("Total VRAM (GB):", round(torch.cuda.get_device_properties(0).total_memory / (1024**3), 2), flush=True)
print("Transformers Version:", transformers.__version__, flush=True)
print("PIL Version:", PIL.__version__, flush=True)
print("Done!", flush=True)
