import pandas as pd
import numpy as np

train = pd.read_csv('data/train.csv')
test = pd.read_csv('data/test.csv')

train_qids = set(train.question_id.unique())
test_qids = set(test.question_id.unique())
overlap = train_qids & test_qids
unseen = test_qids - train_qids

print('=== TEST ===')
print(f'Rows: {len(test)}')
print(f'Unique questions: {test.question_id.nunique()}')
print(f'Seen questions: {len(overlap)}')
print(f'Unseen questions: {len(unseen)}')
print(f'Test rows from seen: {test[test.question_id.isin(overlap)].shape[0]}')
print(f'Test rows from unseen: {test[test.question_id.isin(unseen)].shape[0]}')

# Traces per question
test_qc = test.groupby('question_id').size()
print(f'\nTest traces/question: min={test_qc.min()}, max={test_qc.max()}, median={test_qc.median():.0f}, mean={test_qc.mean():.1f}')

train_qc = train.groupby('question_id').size()
print(f'Train traces/question: min={train_qc.min()}, max={train_qc.max()}, median={train_qc.median():.0f}, mean={train_qc.mean():.1f}')

# Root code diversity per question
q_div = train.groupby('question_id')['root_code'].nunique()
print(f'\nRoot codes per question: min={q_div.min()}, max={q_div.max()}, median={q_div.median():.0f}, mean={q_div.mean():.2f}')
print(f'Questions with 1 code: {(q_div == 1).sum()} / {len(q_div)} ({(q_div == 1).mean()*100:.1f}%)')
print(f'Questions with 2 codes: {(q_div == 2).sum()}')
print(f'Questions with 3+ codes: {(q_div >= 3).sum()}')

# NONE distribution
q_none_count = train.groupby('question_id').apply(lambda g: (g.root_code == 'NONE').sum())
q_total = train.groupby('question_id').size()
q_none_frac = q_none_count / q_total
print(f'\nQuestions with >0 NONE: {(q_none_count > 0).sum()} / {len(q_none_count)}')
all_none = (q_none_count == q_total)
print(f'Questions with ALL NONE: {all_none.sum()}')

# Clean column analysis
print(f'\nClean: {train["clean"].value_counts().to_dict()}')
print(f'Final answer correct: {train["final_answer_correct"].value_counts().to_dict()}')

# Current OOF analysis
oof = np.load('artifacts/oof_adaptive_fusion.npy')
print(f'\nOOF matrix shape: {oof.shape}')

# Per-question OOF analysis
import sys
sys.path.insert(0, '.')
from src.data.taxonomy import TaxonomyManager
tax = TaxonomyManager('data/taxonomy_v1.csv')
classes = tax.all_codes(include_none=True)

y_true = train['root_code'].values
top3 = np.argsort(-oof, axis=1)[:, :3]

# Per-sample MAP@3 scores
scores = np.zeros(len(y_true))
for i, target in enumerate(y_true):
    for rank in range(3):
        if classes[top3[i, rank]] == target:
            scores[i] = 1.0 / (rank + 1)
            break

print(f'\nOverall MAP@3: {scores.mean():.5f}')

# Breakdown by whether it's NONE or not
none_mask = np.array([y == 'NONE' for y in y_true])
print(f'NONE samples MAP@3: {scores[none_mask].mean():.5f} (n={none_mask.sum()})')
non_none_count = (~none_mask).sum()
print(f'Non-NONE samples MAP@3: {scores[~none_mask].mean():.5f} (n={non_none_count})')

# Per-question analysis
for qid in train.question_id.unique()[:5]:
    mask = train.question_id == qid
    q_scores = scores[mask.values]
    q_true = y_true[mask.values]
    print(f'\nQ {qid}: n={mask.sum()}, MAP@3={q_scores.mean():.4f}, codes={pd.Series(q_true).value_counts().to_dict()}')

# How many samples have rank-1 correct?
rank1_correct = sum(1 for i in range(len(y_true)) if classes[top3[i, 0]] == y_true[i])
rank2_correct = sum(1 for i in range(len(y_true)) if classes[top3[i, 1]] == y_true[i])
rank3_correct = sum(1 for i in range(len(y_true)) if classes[top3[i, 2]] == y_true[i])
print(f'\nRank-1 correct: {rank1_correct} ({rank1_correct/len(y_true)*100:.1f}%)')
print(f'Rank-2 correct: {rank2_correct} ({rank2_correct/len(y_true)*100:.1f}%)')
print(f'Rank-3 correct: {rank3_correct} ({rank3_correct/len(y_true)*100:.1f}%)')
print(f'Not in top3: {len(y_true) - rank1_correct - rank2_correct - rank3_correct} ({(len(y_true) - rank1_correct - rank2_correct - rank3_correct)/len(y_true)*100:.1f}%)')

# Examine confused predictions on non-NONE
print('\n=== CONFUSION ANALYSIS (Non-NONE samples) ===')
confused = []
for i in range(len(y_true)):
    if y_true[i] != 'NONE' and scores[i] == 0:
        confused.append({
            'idx': i,
            'true': y_true[i],
            'pred1': classes[top3[i, 0]],
            'pred2': classes[top3[i, 1]],
            'pred3': classes[top3[i, 2]],
            'qid': train.iloc[i]['question_id']
        })
print(f'Non-NONE samples NOT in top3: {len(confused)}')
# Most common true codes that get missed
from collections import Counter
missed_codes = Counter(c['true'] for c in confused)
print(f'Most missed codes: {missed_codes.most_common(10)}')

# Most common pred1 when non-NONE is missed
pred1_when_missed = Counter(c['pred1'] for c in confused)
print(f'Most common pred1 when non-NONE missed: {pred1_when_missed.most_common(5)}')
