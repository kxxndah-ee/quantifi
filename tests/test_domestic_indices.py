import unittest
from unittest.mock import patch, MagicMock
from src.collectors.domestic_indices import (
    fetch_domestic_indices_summary,
    fetch_btc_price,
    _safe_float,
    DOMESTIC_INDEX_KEYS
)


class TestDomesticIndicesCollector(unittest.TestCase):
    def test_safe_float(self):
        self.assertEqual(_safe_float("1,234.56"), 1234.56)
        self.assertEqual(_safe_float("-81.12"), -81.12)
        self.assertEqual(_safe_float(None, 10.0), 10.0)
        self.assertEqual(_safe_float("invalid", 5.0), 5.0)

    @patch("requests.Session.get")
    def test_fetch_btc_price_upbit_mock(self, mock_get):
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = [{
            "market": "KRW-BTC",
            "trade_price": 105000000.0,
            "signed_change_rate": 0.025,
            "signed_change_price": 2500000.0,
            "prev_closing_price": 102500000.0
        }]
        mock_get.return_value = mock_resp

        import requests
        session = requests.Session()
        res = fetch_btc_price(session)
        self.assertEqual(res["code"], "BTC")
        self.assertEqual(res["name"], "비트코인")
        self.assertEqual(res["close"], 105000000.0)
        self.assertEqual(res["change_pct"], 2.5)
        self.assertIn("105,000,000", res["display_price"])

    @patch("requests.Session.get")
    def test_fetch_domestic_indices_summary_mock(self, mock_get):
        def mock_get_side_effect(url, **kwargs):
            m = MagicMock()
            m.status_code = 200
            if "polling.finance.naver.com" in url:
                m.json.return_value = {
                    "datas": [
                        {"itemCode": "KOSPI", "stockName": "코스피", "closePrice": "2,600.50", "fluctuationsRatio": "1.25", "compareToPreviousClosePrice": "32.10"},
                        {"itemCode": "KOSDAQ", "stockName": "코스닥", "closePrice": "750.20", "fluctuationsRatio": "-0.50", "compareToPreviousClosePrice": "-3.80"},
                        {"itemCode": "KPI200", "stockName": "코스피 200", "closePrice": "350.10", "fluctuationsRatio": "0.80", "compareToPreviousClosePrice": "2.80"},
                        {"itemCode": "FUT", "stockName": "코스피 200 선물", "closePrice": "349.50", "fluctuationsRatio": "0.75", "compareToPreviousClosePrice": "2.60"},
                        {"itemCode": "KQI150", "stockName": "코스닥 150", "closePrice": "1,205.30", "fluctuationsRatio": "-1.10", "compareToPreviousClosePrice": "-13.40"},
                        {"itemCode": "KVALUE", "stockName": "코리아 밸류업", "closePrice": "1,010.50", "fluctuationsRatio": "1.05", "compareToPreviousClosePrice": "10.50"},
                    ]
                }
            elif "upbit.com" in url:
                m.json.return_value = [{
                    "trade_price": 105000000.0,
                    "signed_change_rate": 0.015,
                    "signed_change_price": 1500000.0,
                    "prev_closing_price": 103500000.0
                }]
            return m

        mock_get.side_effect = mock_get_side_effect
        summary = fetch_domestic_indices_summary(force_refresh=True)

        self.assertIn("items", summary)
        self.assertEqual(len(summary["items"]), 7)
        codes = [it["code"] for it in summary["items"]]
        self.assertEqual(codes, ["KOSPI", "KOSDAQ", "KPI200", "FUT", "KQI150", "KVALUE", "BTC"])

        names = [it["name"] for it in summary["items"]]
        self.assertEqual(names, ["코스피", "코스닥", "코스피200", "코스피200선물", "코스닥 150", "코리아밸류업", "비트코인"])

        self.assertIn("코스피 +1.25%", summary["summary_text"])
        self.assertIn("비트코인 +1.50%", summary["summary_text"])

    def test_live_fetch_or_fallback(self):
        summary = fetch_domestic_indices_summary()
        self.assertIn("items", summary)
        self.assertEqual(len(summary["items"]), 7)
        self.assertTrue(all(it["close"] > 0 for it in summary["items"]))


if __name__ == "__main__":
    unittest.main()
