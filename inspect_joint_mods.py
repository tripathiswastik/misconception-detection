import pandas as pd

sub_dir = "d:/ag-ai/misconception-detection/submissions"
quad = pd.read_csv(f"{sub_dir}/blend_rank1_locked_quad_champion_submission.csv")
cons = pd.read_csv(f"{sub_dir}/blend_rank1_locked_conservative_champion_submission.csv")
champ = pd.read_csv(f"{sub_dir}/blend_m15_recalibrated_composite_triblend_448_submission.csv")

# Find rows where BOTH quad and cons agree, but differ from champ
joint_mods = []
for i in range(len(quad)):
    if quad.iloc[i]["root_code"] == cons.iloc[i]["root_code"] and quad.iloc[i]["root_code"] != champ.iloc[i]["root_code"]:
        joint_mods.append((i, champ.iloc[i]["root_code"], quad.iloc[i]["root_code"]))

print(f"Total joint modifications that powered the +0.00100 surge to 0.54716: {len(joint_mods)}")
for idx, old_c, new_c in joint_mods:
    print(f"Row {idx:4d}: Old=\"{old_c}\" -> New=\"{new_c}\"")

