import json
from pathlib import Path
import urllib.request

token_path = Path.home() / ".kaggle" / "access_token"
if not token_path.exists():
    print("Token file missing")
    exit(1)

token = token_path.read_text().strip()
url = "https://www.kaggle.com/api/v1/competitions/submissions/list/misconception-detection-in-k-12-student-math-workings"
req = urllib.request.Request(url, headers={"Authorization": f"Bearer {token}"})

try:
    with urllib.request.urlopen(req) as resp:
        data = json.loads(resp.read().decode("utf-8"))
        print(f"{'Date':24s} | {'File Name':50s} | {'Public Score':12s} | {'Status':10s}")
        print("-" * 105)
        for sub in data[:20]:
            d = sub.get("date", "")[:19]
            fn = sub.get("fileName", "")
            ps = sub.get("publicScore", "None")
            st = sub.get("status", "")
            print(f"{d:24s} | {fn:50s} | {str(ps):12s} | {st:10s}")
except Exception as e:
    print(f"Error fetching submissions: {e}")

