import unittest
from unittest.mock import patch, MagicMock
import numpy as np
import pandas as pd
import streamlit as st

from src.backtest.btc_regime import (
    DataPipeline,
    MarketRegimeModel,
    PerformanceEvaluator,
    Visualizer,
    run_or_load_btc_regime_analysis,
)


class TestBitcoinRegimeEngine(unittest.TestCase):
    def setUp(self):
        # Create synthetic BTC price series for unit testing
        dates = pd.date_range("2020-01-01", periods=300, freq="D")
        np.random.seed(42)
        returns = np.random.normal(0.001, 0.03, size=len(dates))
        prices = 10000.0 * np.exp(np.cumsum(returns))
        volumes = np.random.uniform(1e8, 1e9, size=len(dates))

        self.mock_df = pd.DataFrame(
            {"price": prices, "volume": volumes},
            index=dates
        )

    def test_feature_engineering(self):
        feat_df = DataPipeline.build_features(self.mock_df)
        self.assertIn("log_ret", feat_df.columns)
        self.assertIn("realized_vol", feat_df.columns)
        self.assertIn("trend_ratio", feat_df.columns)
        self.assertFalse(feat_df.isnull().any().any())
        self.assertGreater(len(feat_df), 50)

    def test_hmm_market_regime_model(self):
        feat_df = DataPipeline.build_features(self.mock_df)
        feature_cols = ["log_ret", "realized_vol", "trend_ratio"]
        X = feat_df[feature_cols].values
        returns = feat_df["log_ret"].values

        model = MarketRegimeModel(n_states=2, random_state=42)
        model.fit(X, returns)

        self.assertIn(model.bull_state_idx, [0, 1])
        pred = model.predict_state(X[-5:])
        self.assertIn(pred, [0, 1])

    def test_performance_evaluator(self):
        df = self.mock_df.copy()
        df["log_ret"] = np.log(df["price"] / df["price"].shift(1))
        df["regime"] = np.random.choice([0, 1], size=len(df))
        df.dropna(inplace=True)

        metrics, eval_df = PerformanceEvaluator.evaluate(df)
        self.assertIn("Regime Switching Strategy (OOS)", metrics)
        self.assertIn("Buy and Hold (OOS)", metrics)
        self.assertIn("CAGR", metrics["Regime Switching Strategy (OOS)"])
        self.assertIn("MDD", metrics["Regime Switching Strategy (OOS)"])
        self.assertIn("Sharpe", metrics["Regime Switching Strategy (OOS)"])
        self.assertIn("cum_strat", eval_df.columns)
        self.assertIn("cum_bnh", eval_df.columns)

    def test_visualizer_build_figure(self):
        df = self.mock_df.copy()
        df["log_ret"] = np.log(df["price"] / df["price"].shift(1))
        df["regime"] = 1
        df["cum_strat"] = 1.2
        df["cum_bnh"] = 1.1
        df.dropna(inplace=True)

        fig = Visualizer.build_figure(df)
        self.assertIsNotNone(fig)
        self.assertGreaterEqual(len(fig.data), 4)

    def test_run_or_load_btc_regime_analysis_cached(self):
        metrics, eval_df = run_or_load_btc_regime_analysis(force_refresh=False)
        self.assertIsInstance(metrics, dict)
        self.assertIsInstance(eval_df, pd.DataFrame)
        self.assertIn("regime", eval_df.columns)
        self.assertFalse(eval_df.empty)

    def test_tab5_in_main_tab_names(self):
        import app
        self.assertEqual(len(app.main_tab_names), 5)
        self.assertEqual(app.main_tab_names[4], "5. 비트코인")

    def test_render_btc_regime_tab_runs(self):
        import app
        self.assertTrue(hasattr(app, "render_btc_regime_tab"))

        def mock_columns(spec, *args, **kwargs):
            n = len(spec) if isinstance(spec, (list, tuple)) else int(spec)
            return [MagicMock() for _ in range(n)]

        with patch.object(st, "columns", side_effect=mock_columns):
            with patch.object(st, "markdown"):
                with patch.object(st, "button", return_value=False):
                    with patch.object(st, "metric"):
                        with patch.object(st, "plotly_chart"):
                            with patch.object(st, "dataframe"):
                                with patch.object(st, "expander", return_value=MagicMock()):
                                    app.render_btc_regime_tab()


if __name__ == "__main__":
    unittest.main()
