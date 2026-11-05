import pandas as pd

sub_dir = "d:/ag-ai/misconception-detection/submissions"
quad = pd.read_csv(f"{sub_dir}/blend_rank1_locked_quad_champion_submission.csv")
cons = pd.read_csv(f"{sub_dir}/blend_rank1_locked_conservative_champion_submission.csv")
champ = pd.read_csv(f"{sub_dir}/blend_m15_recalibrated_composite_triblend_448_submission.csv")

diffs = [i for i in range(len(quad)) if quad.iloc[i]["root_code"] != cons.iloc[i]["root_code"]]
r2_diffs = [i for i in range(len(quad)) if quad.iloc[i]["root_code"].split()[1] != cons.iloc[i]["root_code"].split()[1]]
r3_diffs = [i for i in range(len(quad)) if quad.iloc[i]["root_code"].split()[2] != cons.iloc[i]["root_code"].split()[2]]

print(f"Total differences between the two 0.54716 champions: {len(diffs)}")
print(f"Rank-2 differences: {len(r2_diffs)}")
print(f"Rank-3 differences: {len(r3_diffs)}")

print("\nDetailed breakdown of differing rows:")
for i in diffs:
    print(f"Row {i:4d}: Orig=\"{champ.iloc[i]['root_code']}\"")
    print(f"          Quad=\"{quad.iloc[i]['root_code']}\"")
    print(f"          Cons=\"{cons.iloc[i]['root_code']}\"")

