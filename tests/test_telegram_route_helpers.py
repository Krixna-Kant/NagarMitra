import unittest
from urllib.parse import parse_qs, urlparse

from app.bot import telegram_webhook


class TelegramRouteHelperTests(unittest.TestCase):
    def test_extract_destination_query_from_route_phrase(self):
        query = telegram_webhook._extract_destination_query("I want to go to DLF CyberHub")
        self.assertEqual(query, "DLF CyberHub")

    def test_extract_destination_query_returns_none_for_non_route_text(self):
        query = telegram_webhook._extract_destination_query("What is AQI in Rohini?")
        self.assertIsNone(query)

    def test_pick_closest_candidate(self):
        candidates = [
            {"latitude": 28.495, "longitude": 77.09, "display_name": "Far"},
            {"latitude": 28.503, "longitude": 77.101, "display_name": "Near"},
        ]
        best = telegram_webhook._pick_closest_candidate(28.502, 77.102, candidates)
        self.assertEqual(best["display_name"], "Near")
        self.assertIn("distance_from_origin_km", best)
        self.assertGreaterEqual(best["distance_from_origin_km"], 0)

    def test_build_route_ui_link_contains_expected_params(self):
        link = telegram_webhook._build_route_ui_link(
            origin_lat=28.6139,
            origin_lon=77.2090,
            destination_lat=28.4953,
            destination_lon=77.0899,
            profile="driving",
        )
        parsed = urlparse(link)
        params = parse_qs(parsed.query)

        self.assertEqual(parsed.scheme, "https")
        self.assertEqual(params["from"][0], "28.613900,77.209000")
        self.assertEqual(params["to"][0], "28.495300,77.089900")
        self.assertEqual(params["profiling"][0], "driving")
        self.assertEqual(params["profile"][0], "driving")


if __name__ == "__main__":
    unittest.main()
