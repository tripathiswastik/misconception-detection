"""Submit Candidate S560 to Kaggle Competition and Monitor Score.

Candidate:
- File: submissions/blend_sota560_convnext448_protected_fusion_submission.csv
- Specs: ConvNeXt-448 FiLM Fusion (w=0.04) + Strict Fold-Safe S550 NONE-Anchor
- CV Composite MAP@3: 0.54286 (Delta vs S550: +0.00072)
- Held-Out Gate: Rare R@10 +2 (220/298), False NONE Demotions -3 (101/506)
- Test Trace Behavior: 40 false demotions blocked, 401 NONE rank-2/3 refinements
"""

import json
import os
from pathlib import Path
import sys
import time

PROJECT_ROOT = Path("d:/ag-ai/misconception-detection")
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.fusion.validator import validate_submission_file

SUB_DIR = PROJECT_ROOT / "submissions"
ART_DIR = PROJECT_ROOT / "artifacts"
COMPETITION = "misconception-detection-in-k-12-student-math-workings"
SUB_NAME = "blend_sota560_convnext448_protected_fusion_submission.csv"
SUB_FILE = SUB_DIR / SUB_NAME
MESSAGE = "blend_sota560_convnext448_protected_fusion (ConvNeXt-448 FiLM w=0.04 + fold-safe NONE anchor, CV 0.54286, delta +0.00072)"


def main():
    print("=" * 85)
    print("SUBMITTING CANDIDATE S560 TO KAGGLE COMPETITION")
    print("=" * 85)

    if not SUB_FILE.exists():
        raise FileNotFoundError(f"Missing submission file: {SUB_FILE}")

    print(f"Validating {SUB_FILE.name}...")
    val_res = validate_submission_file(SUB_FILE)
    if not val_res["valid"]:
        raise ValueError(f"Submission validation failed: {val_res['errors']}")
    print("Validation passed: 3,498 rows, taxonomy compliant, non-empty rankings.")

    # Authenticate Kaggle API
    print("\nAuthenticating Kaggle API...")
    kaggle_dir = Path.home() / ".kaggle"
    access_token_file = kaggle_dir / "access_token"
    kaggle_json_file = kaggle_dir / "kaggle.json"

    if access_token_file.exists():
        token = access_token_file.read_text(encoding="utf-8").strip()
        os.environ["KAGGLE_KEY"] = token
        os.environ["KAGGLE_API_TOKEN"] = token
        os.environ["KAGGLE_USERNAME"] = "Swastik.T"
        print("Loaded token from ~/.kaggle/access_token")
    elif kaggle_json_file.exists():
        with open(kaggle_json_file) as f:
            cred = json.load(f)
            os.environ["KAGGLE_USERNAME"] = cred["username"]
            os.environ["KAGGLE_KEY"] = cred["key"]
        print("Loaded credentials from ~/.kaggle/kaggle.json")
    else:
        raise FileNotFoundError("Could not find Kaggle credentials in ~/.kaggle/")

    from kaggle.api.kaggle_api_extended import KaggleApi
    api = KaggleApi()
    api.authenticate()
    print("Kaggle API authenticated successfully.")

    print(f"\nSubmitting {SUB_NAME} to {COMPETITION}...")
    submit_result = api.competition_submit(
        file_name=str(SUB_FILE),
        message=MESSAGE,
        competition=COMPETITION
    )
    print(f"Submission response: {submit_result}")

    # Monitor submission status
    print("\nMonitoring Kaggle submission status...")
    score_recorded = None
    status_recorded = None

    for attempt in range(1, 35):
        time.sleep(10)
        submissions = api.competition_submissions(COMPETITION)
        if submissions:
            latest = submissions[0]
            fname = getattr(latest, "fileName", "")
            status = getattr(latest, "status", "")
            score = getattr(latest, "publicScore", None)
            status_str = str(status).upper()

            print(f"[{attempt:02d} | {time.strftime('%H:%M:%S')}] File: {fname} | Status: {status} | Score: {score}")

            if "COMPLETE" in status_str:
                score_recorded = score
                status_recorded = status_str
                print("\n" + "=" * 85)
                print("SUBMISSION COMPLETE!")
                print(f"Candidate: {fname}")
                print(f"Public Score: {score}")
                print(f"Active Champion S550 Reference: 0.55049")
                print(f"Unprotected S559 Reference:     0.54616")

                if score is not None:
                    try:
                        sc_f = float(score)
                        delta_champ = sc_f - 0.55049
                        delta_s559 = sc_f - 0.54616
                        print(f"Delta vs Active Champion S550: {delta_champ:+.5f}")
                        print(f"Delta vs Unprotected S559:     {delta_s559:+.5f}")
                        if delta_champ > 0:
                            print(f"Outcome: NEW ALL-TIME BEST LEADERBOARD PEAK! (+{delta_champ:.5f})")
                        elif delta_champ == 0:
                            print(f"Outcome: TIED LEADERBOARD PEAK (0.55049)")
                        else:
                            print(f"Outcome: Below S550 Champion ({delta_champ:+.5f})")
                    except Exception as e:
                        print(f"Score parse notice: {e}")
                print("=" * 85)
                break
            elif "ERROR" in status_str or "FAIL" in status_str:
                status_recorded = status_str
                print(f"\n[Error] Submission failed: {latest}")
                break

    # Save submission result artifact
    result_data = {
        "candidate": SUB_NAME,
        "message": MESSAGE,
        "status": status_recorded,
        "public_score": score_recorded,
        "champion_reference": 0.55049,
        "cv_composite": 0.54286,
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S")
    }
    with open(ART_DIR / "s560_submission_result.json", "w") as f:
        json.dump(result_data, f, indent=2)
    print(f"\nSaved result to {ART_DIR / 's560_submission_result.json'}")


if __name__ == "__main__":
    main()
