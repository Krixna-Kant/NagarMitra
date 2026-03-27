import unittest

from app.services.pollution_score_service import _point_pollution_score, rank_routes


class PollutionScoreServiceTests(unittest.TestCase):
    def test_point_score_is_deterministic(self):
        score_a = _point_pollution_score(
            aqi_value=180.0,
            dust_index=0.52,
            wind_speed_ms=2.0,
            humidity_pct=44.0,
            traffic_congestion_index=0.45,
            traffic_density=9.4,
            construction_density=0.8,
        )
        score_b = _point_pollution_score(
            aqi_value=180.0,
            dust_index=0.52,
            wind_speed_ms=2.0,
            humidity_pct=44.0,
            traffic_congestion_index=0.45,
            traffic_density=9.4,
            construction_density=0.8,
        )
        self.assertEqual(score_a, score_b)

    def test_rank_routes_prefers_cleaner_when_alpha_low(self):
        routes = [
            {
                "route_id": "route_fast",
                "duration_sec": 900,
                "distance_m": 10000,
                "pollution": {"score": 70},
            },
            {
                "route_id": "route_clean",
                "duration_sec": 1150,
                "distance_m": 10800,
                "pollution": {"score": 25},
            },
        ]
        ranked = rank_routes(routes, alpha=0.2)
        self.assertEqual(ranked[0]["route_id"], "route_clean")

    def test_rank_routes_prefers_faster_when_alpha_high(self):
        routes = [
            {
                "route_id": "route_fast",
                "duration_sec": 900,
                "distance_m": 10000,
                "pollution": {"score": 70},
            },
            {
                "route_id": "route_clean",
                "duration_sec": 1150,
                "distance_m": 10800,
                "pollution": {"score": 25},
            },
        ]
        ranked = rank_routes(routes, alpha=0.95)
        self.assertEqual(ranked[0]["route_id"], "route_fast")


if __name__ == "__main__":
    unittest.main()
