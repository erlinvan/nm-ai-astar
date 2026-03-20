import time
import requests
from typing import Optional
from config import API_BASE_URL, API_TOKEN


class AstarIslandClient:
    def __init__(self, token: Optional[str] = None, base_url: str = API_BASE_URL):
        self.base_url = base_url.rstrip("/")
        self.session = requests.Session()
        token = token or API_TOKEN
        if token:
            self.session.headers["Authorization"] = f"Bearer {token}"
        self._last_request_time = 0.0
        self._min_interval = 0.22  # ~5 req/s rate limit

    def _rate_limit(self):
        elapsed = time.time() - self._last_request_time
        if elapsed < self._min_interval:
            time.sleep(self._min_interval - elapsed)
        self._last_request_time = time.time()

    def _get(self, path: str):
        resp = self.session.get(f"{self.base_url}{path}")
        resp.raise_for_status()
        return resp.json()

    def _post(self, path: str, data: dict):
        self._rate_limit()
        resp = self.session.post(f"{self.base_url}{path}", json=data)
        resp.raise_for_status()
        return resp.json()

    def get_rounds(self) -> list:
        return self._get("/astar-island/rounds")

    def get_active_round(self) -> Optional[dict]:
        rounds = self.get_rounds()
        return next((r for r in rounds if r["status"] == "active"), None)

    def get_round_detail(self, round_id: str) -> dict:
        return self._get(f"/astar-island/rounds/{round_id}")

    def get_budget(self) -> dict:
        return self._get("/astar-island/budget")

    def simulate(
        self,
        round_id: str,
        seed_index: int,
        viewport_x: int = 0,
        viewport_y: int = 0,
        viewport_w: int = 15,
        viewport_h: int = 15,
    ) -> dict:
        return self._post(
            "/astar-island/simulate",
            {
                "round_id": round_id,
                "seed_index": seed_index,
                "viewport_x": viewport_x,
                "viewport_y": viewport_y,
                "viewport_w": viewport_w,
                "viewport_h": viewport_h,
            },
        )

    def submit(self, round_id: str, seed_index: int, prediction: list) -> dict:
        return self._post(
            "/astar-island/submit",
            {
                "round_id": round_id,
                "seed_index": seed_index,
                "prediction": prediction,
            },
        )

    def get_my_rounds(self) -> list:
        return self._get("/astar-island/my-rounds")

    def get_my_predictions(self, round_id: str) -> list:
        return self._get(f"/astar-island/my-predictions/{round_id}")

    def get_analysis(self, round_id: str, seed_index: int) -> dict:
        return self._get(f"/astar-island/analysis/{round_id}/{seed_index}")

    def get_leaderboard(self) -> list:
        return self._get("/astar-island/leaderboard")
