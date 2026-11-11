import sys
from typing import Any, cast
import torch

print("Testing Salesforce/blip-vqa-base load...", flush=True)
try:
    from transformers import BlipProcessor, BlipForQuestionAnswering
    model_id = "Salesforce/blip-vqa-base"
    print("Loading processor...", flush=True)
    processor = BlipProcessor.from_pretrained(model_id)
    print("Processor loaded successfully!", flush=True)

    print("Loading model...", flush=True)
    model = cast(Any, BlipForQuestionAnswering).from_pretrained(model_id)
    print("Model loaded successfully!", flush=True)
    params = sum(p.numel() for p in model.parameters()) / 1e6
    print(f"Parameter count: {params:.1f} M")
except Exception as e:
    import traceback
    print("Error loading BLIP:", e, flush=True)
    traceback.print_exc()

