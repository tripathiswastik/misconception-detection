import sys
from pathlib import Path
from typing import Any, cast
import torch

print("Testing Florence-2-base load...", flush=True)
try:
    from transformers import AutoProcessor, AutoModelForCausalLM
    model_id = "microsoft/Florence-2-base"
    print("Loading processor...", flush=True)
    processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)
    print("Processor loaded successfully!", flush=True)

    print("Loading model...", flush=True)
    model = cast(Any, AutoModelForCausalLM).from_pretrained(model_id, trust_remote_code=True)
    print("Model loaded successfully!", flush=True)
    print("Parameter count:", sum(p.numel() for p in model.parameters()) / 1e6, "M")
except Exception as e:
    import traceback
    print("Error loading Florence-2-base:", e, flush=True)
    traceback.print_exc()

