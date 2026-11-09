import pandas as pd
from pathlib import Path
import sys

PROJECT_ROOT = Path("d:/ag-ai/misconception-detection")
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.reader.diagnostic_verifier import enhanced_safe_verifier

train_df = pd.read_csv(PROJECT_ROOT / "data/train.csv")
print(f"Total train rows: {len(train_df)}", flush=True)

steps_list = train_df["transcription_steps"].fillna("").astype(str).tolist()
correct_list = train_df["correct_answer"].fillna("").astype(str).tolist()
root_list = train_df["root_code"].fillna("NONE").astype(str).tolist()

results = []
for steps, correct_ans, root_code in zip(steps_list, correct_list, root_list):
    is_correct, ext_ans, err_cat = enhanced_safe_verifier(steps, correct_ans)
    true_is_none = (root_code == "NONE")
    results.append({
        "is_correct": is_correct,
        "ext_ans": ext_ans,
        "err_cat": err_cat,
        "true_is_none": true_is_none
    })

res_df = pd.DataFrame(results)
print("\nVerifier Output Counts on Train (1500 rows):", flush=True)
print(res_df["is_correct"].value_counts(dropna=False), flush=True)

# When verifier says False (student got wrong answer)
wrong_ans = res_df[res_df["is_correct"] == False]
true_error_rate = (~wrong_ans["true_is_none"]).mean()
n_wrong = len(wrong_ans)
n_true_err = sum(~wrong_ans["true_is_none"])
n_clean_fa = sum(wrong_ans["true_is_none"])

print(f"\nWhen Verifier flags WRONG answer (Student error detected):", flush=True)
print(f"  Total flagged rows: {n_wrong} / 1500 ({n_wrong/1500:.1%})", flush=True)
print(f"  True Misconceptions: {n_true_err} / {n_wrong} -> Precision = {true_error_rate:.2%}", flush=True)
print(f"  Clean Students (False Alarms): {n_clean_fa} / {n_wrong} -> False Alarm Rate = {n_clean_fa/n_wrong:.2%}", flush=True)

# When verifier says True (student got right answer)
right_ans = res_df[res_df["is_correct"] == True]
clean_rate = right_ans["true_is_none"].mean()
n_right = len(right_ans)
n_clean = sum(right_ans["true_is_none"])
n_err_fa = sum(~right_ans["true_is_none"])

print(f"\nWhen Verifier flags CORRECT answer (Student solved correctly):", flush=True)
print(f"  Total flagged rows: {n_right} / 1500 ({n_right/1500:.1%})", flush=True)
print(f"  True Clean (NONE): {n_clean} / {n_right} -> Precision = {clean_rate:.2%}", flush=True)
print(f"  Misconception missed: {n_err_fa} / {n_right}", flush=True)
