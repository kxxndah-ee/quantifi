import unittest
from unittest.mock import patch, MagicMock
from src.collectors.investor_data import fetch_investor_trend, _parse_int, _parse_float


class TestInvestorData(unittest.TestCase):

    def test_parse_helpers(self):
        self.assertEqual(_parse_int("+1,234,567"), 1234567)
        self.assertEqual(_parse_int("-500,000"), -500000)
        self.assertEqual(_parse_int("0"), 0)
        self.assertEqual(_parse_int(None), 0)

        self.assertEqual(_parse_float("45.96%"), 45.96)
        self.assertEqual(_parse_float("+12.5%"), 12.5)
        self.assertEqual(_parse_float("-3.2%"), -3.2)
        self.assertEqual(_parse_float(None), 0.0)

    @patch("src.collectors.investor_data.requests.get")
    def test_fetch_investor_trend_mock(self, mock_get):
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = [
            {
                "bizdate": "20261117",
                "foreignerPureBuyQuant": "+1,500,000",
                "organPureBuyQuant": "+800,000",
                "individualPureBuyQuant": "-2,300,000",
                "foreignerHoldRatio": "52.30%",
                "closePrice": "75,000"
            },
            {
                "bizdate": "20261116",
                "foreignerPureBuyQuant": "+1,100,000",
                "organPureBuyQuant": "+400,000",
                "individualPureBuyQuant": "-1,500,000",
                "foreignerHoldRatio": "52.20%",
                "closePrice": "74,200"
            },
            {
                "bizdate": "20261115",
                "foreignerPureBuyQuant": "-200,000",
                "organPureBuyQuant": "+100,000",
                "individualPureBuyQuant": "+100,000",
                "foreignerHoldRatio": "52.10%",
                "closePrice": "73,800"
            }
        ]
        mock_get.return_value = mock_resp

        res = fetch_investor_trend("005930")
        self.assertTrue(res["has_data"])
        self.assertEqual(res["foreigner_latest_buy"], 1500000)
        self.assertEqual(res["organ_latest_buy"], 800000)
        self.assertEqual(res["foreigner_hold_ratio"], 52.3)
        self.assertTrue(res["is_dual_buying"])
        self.assertEqual(res["consecutive_foreigner_days"], 2)
        self.assertEqual(res["consecutive_organ_days"], 3)
        self.assertEqual(res["foreigner_3d_sum"], 2400000)
        self.assertIn("쌍끌이 순매수", res["summary_text"])

    def test_fetch_investor_trend_fallback_on_network_error(self):
        # Even if connection fails or invalid code, it must not throw and return safe dictionary
        res = fetch_investor_trend("999999")
        self.assertIsInstance(res, dict)
        self.assertIn("has_data", res)
        self.assertIn("summary_text", res)


if __name__ == "__main__":
    unittest.main()
