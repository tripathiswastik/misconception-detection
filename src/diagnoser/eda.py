from typing import cast
import pandas as pd

df = cast(pd.DataFrame, pd.read_csv("data/train.csv"))
print("Total train traces:", len(df))
print("Unique questions in train:", df["question_id"].nunique())
print("\nTop 10 root_code counts:")
print(df["root_code"].value_counts().head(10))

none_count = (df["root_code"] == "NONE").sum()
print("\nNONE count:", none_count, f"({none_count / len(df):.1%})")

print("\nWhen final_answer_correct is True (count = {}):".format((df["final_answer_correct"] == True).sum()))
print(df[df["final_answer_correct"] == True]["root_code"].value_counts().head(5))

print("\nWhen clean is True (count = {}):".format((df["clean"] == True).sum()))
print(df[df["clean"] == True]["root_code"].value_counts())
