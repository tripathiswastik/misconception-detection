"""Submit Candidate S558 to Kaggle Competition and monitor scoring."""
from pathlib import Path
import sys
import time
import pandas as pd
from kaggle.api.kaggle_api_extended import KaggleApi

PROJECT_ROOT = Path("d:/ag-ai/misconception-detection")
SUB_FILE = PROJECT_ROOT / "submissions" / "blend_sota558_domain_gated_arbitration_submission.csv"
COMPETITION = "misconception-detection-in-k-12-student-math-workings"
MESSAGE = "blend_sota558_domain_gated_arbitration_submission (nested CV validated Candidate B, 5 surgical switches)"

def main():
    print("=" * 80)
    print("SUBMITTING S558 TO KAGGLE COMPETITION")
    print("=" * 80)

    # 1. Verification
    assert SUB_FILE.exists(), f"Submission file {SUB_FILE} does not exist!"
    df = pd.read_csv(SUB_FILE)
    print(f"File verified: {SUB_FILE.name}")
    print(f"Row count: {len(df)}")
    print(f"Columns: {list(df.columns)}")
    assert len(df) == 3498, f"Expected 3498 rows, got {len(df)}"
    assert list(df.columns) == ["trace_id", "root_code"], f"Columns mismatch"
    assert df["root_code"].isna().sum() == 0, "Null values detected"

    # 2. Authentication with verified token
    token = (Path.home() / ".kaggle" / "access_token").read_text().strip()
    import os
    os.environ["KAGGLE_KEY"] = token
    os.environ["KAGGLE_API_TOKEN"] = token
    os.environ["KAGGLE_USERNAME"] = "Swastik.T"

    api = KaggleApi()
    api.authenticate()
    print("Kaggle API authenticated successfully.")

    # 3. Submission
    print(f"\nSubmitting {SUB_FILE.name} to {COMPETITION}...")
    submit_result = api.competition_submit(
        file_name=str(SUB_FILE),
        message=MESSAGE,
        competition=COMPETITION
    )
    print(f"Submit call response: {submit_result}")

    # 4. Polling for Score
    print("\nWaiting for scoring to complete...")
    for attempt in range(1, 30):
        time.sleep(10)
        submissions = api.competition_submissions(COMPETITION)
        if submissions:
            latest = submissions[0]
            desc = getattr(latest, "description", "")
            fname = getattr(latest, "fileName", "")
            status = getattr(latest, "status", "")
            score = getattr(latest, "publicScore", None)
            date = str(getattr(latest, "date", ""))[:19]

            print(f"[{attempt:2d} | {time.strftime('%H:%M:%S')}] File: {fname} | Status: {status} | Score: {score}")

            if status.lower() == "complete":
                print("\n" + "=" * 80)
                print("SUBMISSION COMPLETE!")
                print(f"Candidate: {fname}")
                print(f"Public Score: {score}")
                print(f"Active Champion Reference: 0.55049")
                if score is not None:
                    try:
                        s_float = float(score)
                        delta = s_float - 0.55049
                        print(f"Delta vs Champion (0.55049): {delta:+.5f}")
                    except ValueError:
                        pass
                print("=" * 80)
                break
            elif status.lower() in ["error", "failed"]:
                print(f"\n[ERROR] Submission failed with status: {status}")
                error_desc = getattr(latest, "errorDescription", "Unknown error")
                print(f"Error details: {error_desc}")
                break

if __name__ == "__main__":
    main()
