import pandas as pd

df = pd.read_csv("d:/ag-ai/misconception-detection/data/train.csv")
print("Non-null transcription_steps count:", df["transcription_steps"].notna().sum())
for i in range(5):
    print(f"\n--- Trace {df.iloc[i]['trace_id']} (root_code={df.iloc[i]['root_code']}) ---")
    print(f"Image File: {df.iloc[i]['image_file']}")
    print(f"Transcription Steps: {df.iloc[i]['transcription_steps']}")
