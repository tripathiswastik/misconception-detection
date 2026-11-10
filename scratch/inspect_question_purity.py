import pandas as pd
import numpy as np

df = pd.read_csv('d:/ag-ai/misconception-detection/data/train.csv')
stats = []
for q, g in df.groupby('question_id'):
    nq = len(g)
    labels = g['root_code'].fillna('NONE').values
    u_labels, counts = np.unique(labels, return_counts=True)
    probs = counts / nq
    entropy = -np.sum(probs * np.log(probs + 1e-12))
    top_cnt = np.max(counts)
    margin = (counts[np.argsort(-counts)][0] - counts[np.argsort(-counts)][1]) / nq if len(counts) > 1 else 1.0
    stats.append({
        'qid': q,
        'nq': nq,
        'n_unique': len(u_labels),
        'top_ratio': top_cnt / nq,
        'margin': margin,
        'entropy': entropy
    })

s_df = pd.DataFrame(stats)
print("Question Support Count (nq) summary:")
print(s_df['nq'].describe())

print("\nPurity Statistics:")
print(f"Total Questions: {len(s_df)}")
print(f"100% Pure Questions (all traces identical label): {(s_df['n_unique'] == 1).sum()} ({s_df['n_unique'].eq(1).mean():.1%})")

p3 = s_df[s_df['nq'] >= 3]
print(f"\nFor questions with >= 3 traces ({len(p3)} questions):")
print(f"  100% Pure: {p3['n_unique'].eq(1).sum()} ({p3['n_unique'].eq(1).mean():.1%})")
print(f"  Dominant label >= 75%: {(p3['top_ratio'] >= 0.75).sum()} ({(p3['top_ratio'] >= 0.75).mean():.1%})")
print(f"  High Entropy (>= 0.8): {(p3['entropy'] >= 0.8).sum()} ({(p3['entropy'] >= 0.8).mean():.1%})")

