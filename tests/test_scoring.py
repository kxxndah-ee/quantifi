import unittest
import pandas as pd
import numpy as np
from src.core.scoring import (
    calculate_rsi,
    calculate_bollinger_bands,
    compute_point_in_time_indicators,
    calculate_score_for_row
)


class TestScoring(unittest.TestCase):
    def test_rsi_bounds(self):
        prices = pd.Series([100 + i + (i % 3) * 2 for i in range(50)])
        rsi = calculate_rsi(prices, 14)
        self.assertTrue((rsi >= 0).all() and (rsi <= 100).all())

    def test_bollinger_bands(self):
        prices = pd.Series([1000 + np.sin(i / 5.0) * 50 for i in range(60)])
        mid, upper, lower, pct_b = calculate_bollinger_bands(prices, 20, 2.0)
        self.assertEqual(len(upper), len(prices))
        self.assertTrue((upper.dropna() >= lower.dropna()).all())

    def test_score_calculation(self):
        row = pd.Series({
            "RSI14": 55.0,        # Momentum zone -> 100%
            "MA5": 105.0,
            "MA20": 100.0,
            "MA60": 95.0,         # Full alignment -> 100%
            "VolRatio": 2.2,      # 200%+ -> 100%
            "BB_PctB": 0.5,       # Mid band -> 100%
            "Close": 105.0
        })
        res = calculate_score_for_row(row)
        self.assertEqual(res["score"], 100.0)
        self.assertEqual(res["label"], "강세")
        self.assertEqual(res["tp_pct"], 3.0)

    def test_bearish_score(self):
        row = pd.Series({
            "RSI14": 82.0,        # Overbought
            "MA5": 90.0,
            "MA20": 95.0,
            "MA60": 100.0,        # Inverted MA
            "VolRatio": 0.5,      # Low volume
            "BB_PctB": 0.95,      # Band top
            "Close": 90.0
        })
        res = calculate_score_for_row(row)
        self.assertTrue(res["score"] < 50.0)
        self.assertEqual(res["label"], "약세")

    def test_falling_stock_drawdown_and_ma_breakdown_penalty(self):
        # A declining stock where lagged MAs are 5 > 20 > 60, but Close < MA20 and Drawdown is -22%
        row = pd.Series({
            "RSI14": 52.2,
            "MA5": 21220.0,
            "MA20": 20919.5,
            "MA60": 17176.7,
            "VolRatio": 0.59,
            "BB_PctB": 0.47,
            "Close": 20650.0,
            "High20": 26500.0,
            "Drawdown20": -22.08,
            "MA5_Diff": -550.0
        })
        res = calculate_score_for_row(row)
        self.assertLess(res["score"], 50.0)
        self.assertEqual(res["label"], "약세")
        self.assertIn("이탈", res["breakdown"]["ma_status"])
        self.assertGreater(res["breakdown"]["penalty"], 20.0)


if __name__ == "__main__":
    unittest.main()
