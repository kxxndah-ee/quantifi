"""
Unit tests for Upper-Limit (상한가) and Super-Surge Market Leader Evaluation
Verifies that market leaders like Hyundai AutoEver (307950) are properly screened with top conviction scores
across all screening modes (Daytrade, Closing Bet, 5% Surge Target, Premarket).
"""

import unittest
from datetime import datetime
import pandas as pd
import numpy as np

from src.core.high_winrate_strategies import (
    evaluate_intraday_daytrade_candidate,
    evaluate_closing_bet_candidate,
    evaluate_5pct_surge_candidate,
    evaluate_market_regime
)
from src.collectors.market_data import get_last_completed_trading_day


class TestLimitUpLeader(unittest.TestCase):
    def setUp(self):
        # Create a 60-day price history with 280,000 baseline
        dates = pd.date_range(end="2026-11-23", periods=60, freq="B")
        prices = [280000.0 + (i % 5) * 1000.0 for i in range(59)]
        volumes = [50000 + (i % 3) * 10000 for i in range(59)]

        # Day 59 (T-1): Close 282,000
        prices.append(282000.0)
        volumes.append(60000)

        self.df_base = pd.DataFrame({
            "Open": prices,
            "High": [p * 1.01 for p in prices],
            "Low": [p * 0.99 for p in prices],
            "Close": prices,
            "Volume": volumes,
            "Change": [0.0] * 60
        }, index=dates)

        # Create bearish benchmark where Close < 20MA
        bm_dates = pd.date_range(end="2026-11-24", periods=60, freq="B")
        bm_closes = [85000.0 - i * 100.0 for i in range(60)] # declining
        self.df_benchmark_bear = pd.DataFrame({
            "Open": bm_closes,
            "High": bm_closes,
            "Low": bm_closes,
            "Close": bm_closes,
            "Volume": [1000000] * 60,
            "Change": [-0.01] * 60
        }, index=bm_dates)

    def test_limit_up_candidate_in_daytrade(self):
        """Verifies that an upper limit (+29.8%) stock with massive volume is recognized as 초강세(상한가대장)."""
        limit_row = pd.DataFrame([{
            "Open": 295000.0,
            "High": 366000.0,
            "Low": 295000.0,
            "Close": 366000.0, # +29.79% limit-up
            "Volume": 320000,   # 1,171억 KRW
            "Change": 0.2979
        }], index=[pd.to_datetime("2026-11-24")])
        df_limit = pd.concat([self.df_base, limit_row])

        res = evaluate_intraday_daytrade_candidate(
            df_limit,
            df_benchmark=self.df_benchmark_bear,
            min_today_val_krw=20_000_000_000,
            min_intraday_gain=3.0,
            max_intraday_gain=8.5
        )
        self.assertIsNotNone(res)
        self.assertEqual(res["score"], 98.0)
        self.assertIn("상한가", res["label"])
        self.assertIn("상한가", res["name_tag"])

    def test_super_surge_candidate_in_daytrade(self):
        """Verifies that a stock gaining +17.6% from open (Hyundai AutoEver pattern) is recognized as 초강세(급등주도)."""
        surge_row = pd.DataFrame([{
            "Open": 295000.0,
            "High": 362500.0,
            "Low": 295000.0,
            "Close": 347000.0, # +23.05%
            "Volume": 295016,   # 1,023억 KRW
            "Change": 0.2305
        }], index=[pd.to_datetime("2026-11-24")])
        df_surge = pd.concat([self.df_base, surge_row])

        res = evaluate_intraday_daytrade_candidate(
            df_surge,
            df_benchmark=self.df_benchmark_bear,
            min_today_val_krw=20_000_000_000,
            min_intraday_gain=3.0,
            max_intraday_gain=8.5
        )
        self.assertIsNotNone(res)
        self.assertEqual(res["score"], 97.0)
        self.assertIn("급등주도", res["label"])

    def test_limit_up_in_closing_bet(self):
        """Verifies that upper limit and super surge stocks pass closing bet even with disparity > 115%."""
        surge_row = pd.DataFrame([{
            "Open": 295000.0,
            "High": 362500.0,
            "Low": 295000.0,
            "Close": 347000.0, # disparity ~122%
            "Volume": 295016,
            "Change": 0.2305
        }], index=[pd.to_datetime("2026-11-24")])
        df_surge = pd.concat([self.df_base, surge_row])

        res = evaluate_closing_bet_candidate(
            df_surge,
            df_benchmark=self.df_benchmark_bear,
            min_today_val_krw=30_000_000_000
        )
        self.assertIsNotNone(res)
        self.assertGreaterEqual(res["score"], 97.0)

    def test_limit_up_in_5pct_surge(self):
        """Verifies that 5% surge mode identifies limit-up breakout stocks with high conviction."""
        limit_row = pd.DataFrame([{
            "Open": 295000.0,
            "High": 366000.0,
            "Low": 295000.0,
            "Close": 366000.0,
            "Volume": 320000,
            "Change": 0.2979
        }], index=[pd.to_datetime("2026-11-24")])
        df_limit = pd.concat([self.df_base, limit_row])

        res = evaluate_5pct_surge_candidate(
            df_limit,
            df_benchmark=self.df_benchmark_bear,
            min_today_val_krw=30_000_000_000,
            min_day_return=10.0
        )
        self.assertIsNotNone(res)
        self.assertEqual(res["score"], 98.0)
        self.assertIn("상한가", res["label"])

    def test_calendar_last_completed_trading_day(self):
        """Verifies that get_last_completed_trading_day accounts for weekday calendar and does not get stuck in the past."""
        last_day = get_last_completed_trading_day()
        self.assertTrue(isinstance(last_day, str))
        self.assertEqual(len(last_day), 10)
        # Should be a recent date, not older than 7 calendar days
        parsed = datetime.strptime(last_day, "%Y-%m-%d")
        diff_days = (datetime.now() - parsed).days
        self.assertLessEqual(diff_days, 7)


if __name__ == "__main__":
    unittest.main()
