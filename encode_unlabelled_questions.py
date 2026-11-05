"""Encode 284 unlabelled questions using paraphrase-multilingual-MiniLM-L12-v2."""
from typing import Any, List, cast
import numpy as np
import pandas as pd
from pathlib import Path
import torch
from transformers import AutoModel, AutoTokenizer

data = Path("d:/ag-ai/misconception-detection/data")
art = Path("d:/ag-ai/misconception-detection/artifacts")
unl = cast(pd.DataFrame, pd.read_csv(data / "unlabelled.csv"))
dup_q = cast(pd.DataFrame, unl.drop_duplicates(subset=["question_id"]))
unique_q = dup_q.reset_index(drop=True)

model_name = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
print(f"Loading {model_name} for {len(unique_q)} questions...")

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
tokenizer = cast(Any, AutoTokenizer).from_pretrained(model_name)
model = cast(Any, AutoModel).from_pretrained(model_name).to(device).eval()

texts: List[str] = []
for _, r in unique_q.iterrows():
    t = f"Topik: {r['topic_name']}. Kelas: {r['grade']}. Pertanyaan: {r['question_text']}. Jawaban Benar: {r['correct_answer']}"
    texts.append(t)

batch_size = 32
all_embeds: List[np.ndarray] = []

with torch.no_grad():
    for i in range(0, len(texts), batch_size):
        batch_texts = texts[i : i + batch_size]
        encoded = tokenizer(batch_texts, padding=True, truncation=True, max_length=256, return_tensors="pt").to(device)
        out = model(**encoded)
        # Mean pooling
        mask = encoded["attention_mask"].unsqueeze(-1).expand(out[0].size()).float()
        sum_emb = torch.sum(out[0] * mask, dim=1)
        sum_mask = torch.clamp(mask.sum(dim=1), min=1e-9)
        mean_emb = sum_emb / sum_mask
        mean_emb = torch.nn.functional.normalize(mean_emb, p=2, dim=1)
        all_embeds.append(mean_emb.cpu().numpy())

q_embeds = np.vstack(all_embeds)
print("Done! Shape:", q_embeds.shape)

np.save(art / "unlabelled_284_q_embeddings_384.npy", q_embeds)
print(f"Saved: {art / 'unlabelled_284_q_embeddings_384.npy'}")
