"""Kaggle REST API Client with Bearer Token Authentication.

Provides programmatic interaction with Kaggle:
- List competition submissions and live leaderboard scores.
- Direct programmatic submission of prediction CSVs.
- Kernel / Notebook synchronization.
"""
import os
from pathlib import Path
from typing import Any, Dict, List, Optional
import requests  # type: ignore[import-untyped]

DEFAULT_COMPETITION = "misconception-detection-in-k-12-student-math-workings"
TOKEN_PATH = Path.home() / ".kaggle" / "bearer_token"


class KaggleClient:
    """Authenticated Kaggle REST API Client."""

    def __init__(self, token: Optional[str] = None):
        self.token = token or os.environ.get("KAGGLE_API_TOKEN")
        if not self.token and TOKEN_PATH.exists():
            self.token = TOKEN_PATH.read_text(encoding="utf-8").strip()

        if not self.token:
            raise ValueError(
                "Missing Kaggle Bearer Token. Provide token, set KAGGLE_API_TOKEN, "
                "or write to ~/.kaggle/bearer_token."
            )

        self.base_url = "https://www.kaggle.com/api/v1"
        self.session = requests.Session()
        self.session.headers.update({
            "Authorization": f"Bearer {self.token}",
            "User-Agent": "Misconception-Kaggle-Client/1.0"
        })

    def list_submissions(self, competition: str = DEFAULT_COMPETITION) -> List[Dict[str, Any]]:
        """List past submissions and their public/private scores."""
        url = f"{self.base_url}/competitions/submissions/list/{competition}"
        resp = self.session.get(url, timeout=30)
        resp.raise_for_status()
        return resp.json()

    def get_latest_submission(self, competition: str = DEFAULT_COMPETITION) -> Optional[Dict[str, Any]]:
        """Fetch the most recent submission and its status."""
        subs = self.list_submissions(competition)
        return subs[0] if subs else None

    def get_kernel_status(self, user_name: str = "tripathiswastik", kernel_slug: str = "misconception-vlm-verification") -> Dict[str, Any]:
        """Fetch kernel execution status from Kaggle."""
        url = f"{self.base_url}/kernels/status?userName={user_name}&kernelSlug={kernel_slug}"
        resp = self.session.get(url, timeout=30)
        resp.raise_for_status()
        return resp.json()

    def get_kernel_output(self, user_name: str = "tripathiswastik", kernel_slug: str = "misconception-vlm-verification") -> Dict[str, Any]:
        """Fetch kernel execution log and output files."""
        url = f"{self.base_url}/kernels/output?userName={user_name}&kernelSlug={kernel_slug}"
        resp = self.session.get(url, timeout=45)
        resp.raise_for_status()
        return resp.json()

    def list_entered_competitions(self) -> List[Dict[str, Any]]:
        """List active competitions the user is enrolled in."""
        url = f"{self.base_url}/competitions/list?group=entered"
        resp = self.session.get(url, timeout=30)
        resp.raise_for_status()
        return resp.json()


if __name__ == "__main__":
    client = KaggleClient()
    latest = client.get_latest_submission()
    print("Latest Submission:", latest)
    status = client.get_kernel_status()
    print("Kernel Status:", status)

    print("Latest Submission:", latest)
