"""Test PyTorch forward and backward pass with Layer 11 trainable on CUDA:0."""
import os
import sys
import time
import torch
import torch.nn as nn
from transformers import AutoModel, AutoTokenizer

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")

model_name = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
tok = AutoTokenizer.from_pretrained(model_name, local_files_only=True)
encoder = AutoModel.from_pretrained(model_name, local_files_only=True).to(device)

# Freeze layers 0-10, unfreeze layer 11
for param in encoder.parameters():
    param.requires_grad = False

# Layer 11 parameters
for name, param in encoder.encoder.layer[-1].named_parameters():
    param.requires_grad = True

trainable_params = sum(p.numel() for p in encoder.parameters() if p.requires_grad)
total_params = sum(p.numel() for p in encoder.parameters())
print(f"Trainable params in Layer 11: {trainable_params:,} / {total_params:,} ({trainable_params/total_params:.1%})")

# Test forward + backward pass
batch_a = ["Soal: Berapakah 2 + 2?", "Soal: Hitung nilai x dari 3x = 9"]
batch_b = ["Miskonsepsi: SAL-01: Salah hitung penjumlahan", "Miskonsepsi: ALG-02: Salah konsep variabel"]

enc = tok(batch_a, batch_b, padding=True, truncation=True, max_length=128, return_tensors="pt").to(device)
out = encoder(**enc)
cls_rep = out.last_hidden_state[:, 0, :]

head = nn.Linear(384, 1).to(device)
logits = head(cls_rep).squeeze(-1)
targets = torch.tensor([1.0, 0.0], device=device)

loss_fn = nn.BCEWithLogitsLoss()
loss = loss_fn(logits, targets)
loss.backward()

print(f"Forward/backward pass successful! Loss: {loss.item():.4f}")
print(f"GPU memory allocated: {torch.cuda.memory_allocated() / (1024**2):.1f} MB")

