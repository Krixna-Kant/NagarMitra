import unittest
from unittest.mock import AsyncMock, patch

from fastapi.testclient import TestClient

from app.main import app


class RoutesApiTests(unittest.TestCase):
    def setUp(self):
        self.client = TestClient(app)
        self.payload = {
            "origin": {"latitude": 28.6139, "longitude": 77.2090},
            "destination": {"latitude": 28.5355, "longitude": 77.3910},
            "profile": "driving",
            "corridor_radius_km": 2.5,
            "max_routes": 2,
            "preference_alpha": 0.4,
        }

    def test_compare_routes_with_mocked_services(self):
        fake_routes = [
            {
                "route_id": "route_1",
                "distance_m": 12000.0,
                "duration_sec": 1700.0,
                "geometry": [[77.209, 28.6139], [77.391, 28.5355]],
                "sample_points": [
                    {"index": 0, "latitude": 28.6139, "longitude": 77.209, "distance_from_start_km": 0.0},
                    {"index": 1, "latitude": 28.56, "longitude": 77.30, "distance_from_start_km": 7.0},
                ],
            }
        ]
        fake_ranked = [
            {
                "route_id": "route_1",
                "distance_m": 12000.0,
                "duration_sec": 1700.0,
                "geometry": [[77.209, 28.6139], [77.391, 28.5355]],
                "sample_points": [],
                "pollution": {
                    "score": 41.2,
                    "avg_aqi": 132.0,
                    "avg_wind_speed_ms": 2.8,
                    "avg_construction_density_per_sq_km": 0.4,
                    "confidence": "Medium",
                    "confidence_value": 0.64,
                    "risk_segments": [],
                },
                "analytics": {"overall_score": 0.32},
                "recommendation_reason": "Best balance of lower pollution exposure and travel time.",
            }
        ]

        with patch(
            "app.routes.routes.get_route_alternatives",
            new=AsyncMock(return_value=fake_routes),
        ), patch(
            "app.routes.routes.score_and_rank_routes",
            new=AsyncMock(return_value=fake_ranked),
        ):
            response = self.client.post("/api/v1/routes/compare", json=self.payload)

        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertEqual(data["recommendation"]["preferred_route_id"], "route_1")
        self.assertEqual(len(data["routes"]), 1)
        self.assertIn("geometry_latlon", data["routes"][0])

    def test_compare_routes_falls_back_when_env_provider_fails(self):
        fake_routes = [
            {
                "route_id": "route_1",
                "distance_m": 10000.0,
                "duration_sec": 1600.0,
                "geometry": [[77.209, 28.6139], [77.28, 28.59], [77.391, 28.5355]],
                "sample_points": [
                    {"index": 0, "latitude": 28.6139, "longitude": 77.209, "distance_from_start_km": 0.0},
                    {"index": 1, "latitude": 28.59, "longitude": 77.28, "distance_from_start_km": 5.2},
                    {"index": 2, "latitude": 28.5355, "longitude": 77.391, "distance_from_start_km": 10.0},
                ],
            }
        ]

        with patch(
            "app.routes.routes.get_route_alternatives",
            new=AsyncMock(return_value=fake_routes),
        ), patch(
            "app.services.pollution_score_service.get_environmental_snapshot",
            new=AsyncMock(side_effect=RuntimeError("provider down")),
        ):
            response = self.client.post("/api/v1/routes/compare", json=self.payload)

        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertEqual(len(data["routes"]), 1)
        pollution = data["routes"][0]["pollution"]
        self.assertEqual(pollution["confidence"], "Low")
        self.assertGreater(pollution["score"], 0)


if __name__ == "__main__":
    unittest.main()
