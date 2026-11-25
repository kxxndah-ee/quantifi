import unittest
from unittest.mock import patch
from src.collectors.us_market import (
    fetch_us_market_summary,
    determine_global_market_regime,
    get_sector_macro_note,
    get_sector_macro_boost,
    detect_premarket_gap_anomaly
)
from src.core.quant_analyst import generate_quant_expert_answer


class TestUSMarketQuant(unittest.TestCase):
    def setUp(self):
        self.mock_metrics_bull = {
            "NASDAQ": {"close": 18500.0, "change_pct": 1.2},
            "SP500": {"close": 5850.0, "change_pct": 0.8},
            "SOXX": {"close": 540.0, "change_pct": 2.5},
            "DOW": {"close": 43000.0, "change_pct": 0.4},
            "USDKRW": {"close": 1335.0, "change_pct": -0.6},
            "VIX": {"close": 14.5, "change_pct": -5.2},
        }

        self.mock_metrics_bear = {
            "NASDAQ": {"close": 17800.0, "change_pct": -2.1},
            "SP500": {"close": 5650.0, "change_pct": -1.8},
            "SOXX": {"close": 490.0, "change_pct": -4.2},
            "DOW": {"close": 41500.0, "change_pct": -1.1},
            "USDKRW": {"close": 1395.0, "change_pct": 1.1},
            "VIX": {"close": 24.5, "change_pct": 22.0},
        }

        self.mock_metrics_caution = {
            "NASDAQ": {"close": 18100.0, "change_pct": -0.9},
            "SP500": {"close": 5750.0, "change_pct": -0.7},
            "SOXX": {"close": 510.0, "change_pct": -1.8},
            "DOW": {"close": 42000.0, "change_pct": -0.3},
            "USDKRW": {"close": 1360.0, "change_pct": 0.3},
            "VIX": {"close": 17.5, "change_pct": 4.0},
        }

        self.mock_metrics_neutral = {
            "NASDAQ": {"close": 18200.0, "change_pct": 0.1},
            "SP500": {"close": 5790.0, "change_pct": 0.0},
            "SOXX": {"close": 522.0, "change_pct": 0.2},
            "DOW": {"close": 42200.0, "change_pct": 0.1},
            "USDKRW": {"close": 1342.0, "change_pct": 0.0},
            "VIX": {"close": 15.8, "change_pct": -1.0},
        }

    def test_determine_regime_bull(self):
        regime = determine_global_market_regime(self.mock_metrics_bull)
        self.assertEqual(regime["regime"], "우호")
        self.assertIn("반도체", regime["sector_notes"])
        self.assertIn("갭상승", regime["sector_notes"]["반도체"])

    def test_determine_regime_bear(self):
        regime = determine_global_market_regime(self.mock_metrics_bear)
        self.assertEqual(regime["regime"], "위험")
        self.assertIn("반도체", regime["sector_notes"])
        self.assertIn("급락", regime["sector_notes"]["반도체"])
        self.assertIn("50%", regime["action_guide"])

    def test_determine_regime_caution(self):
        regime = determine_global_market_regime(self.mock_metrics_caution)
        self.assertEqual(regime["regime"], "경계")
        self.assertIn("약세", regime["regime_desc"])

    def test_determine_regime_neutral(self):
        regime = determine_global_market_regime(self.mock_metrics_neutral)
        self.assertEqual(regime["regime"], "중립")
        self.assertIn("혼조세", regime["regime_desc"])

    def test_sector_macro_note_mapping(self):
        us_summary = {
            "sector_notes": {
                "반도체": "SOXX 급등",
                "기술주": "나스닥 호조",
                "외인수급": "환율 하락 안정"
            }
        }
        note_semi = get_sector_macro_note("반도체 소부장", us_summary)
        self.assertEqual(note_semi, "SOXX 급등")

        note_bio = get_sector_macro_note("바이오헬스", us_summary)
        self.assertEqual(note_bio, "나스닥 호조")

        note_finance = get_sector_macro_note("금융지주", us_summary)
        self.assertEqual(note_finance, "환율 하락 안정")

    def test_fetch_us_market_summary_has_required_keys(self):
        summary = fetch_us_market_summary()
        self.assertIn("metrics", summary)
        self.assertIn("regime", summary)
        self.assertIn("action_guide", summary)
        self.assertIn("sector_notes", summary)
        self.assertIn("NASDAQ", summary["metrics"])
        self.assertIn("SOXX", summary["metrics"])
        self.assertIn("SP500", summary["metrics"])
        self.assertIn("USDKRW", summary["metrics"])

    def test_quant_analyst_incorporates_macro(self):
        sample_stock = {
            "name": "삼성전자",
            "code": "005930",
            "market": "KOSPI",
            "sector": "반도체/전기전자",
            "close": 75000,
            "change_pct": 1.5,
            "score": 85.0,
            "label": "고확신",
            "strategy_tag": "장전 시초가",
            "tp_pct": 3.0,
            "sl_pct": 2.0,
            "vol_ratio": 2.5,
            "rsi": 58.0,
            "ma_status": "5일>20일 정배열"
        }
        ans = generate_quant_expert_answer(sample_stock, "지금 사도 되나요?")
        self.assertIn("글로벌 미장 연동 점검", ans)
        self.assertIn("나스닥", ans)

    def test_multi_stock_comparison_incorporates_macro(self):
        stock_a = {
            "name": "삼성전자",
            "code": "005930",
            "market": "KOSPI",
            "sector": "반도체",
            "close": 75000,
            "change_pct": 1.0,
            "score": 80.0,
            "label": "고확신",
            "tp_pct": 3.0,
            "sl_pct": 2.0,
            "vol_ratio": 2.0,
            "rsi": 55.0,
        }
        stock_b = {
            "name": "HLB",
            "code": "028300",
            "market": "KOSDAQ",
            "sector": "바이오",
            "close": 90000,
            "change_pct": 4.5,
            "score": 88.0,
            "label": "고확신",
            "tp_pct": 5.0,
            "sl_pct": 2.5,
            "vol_ratio": 3.5,
            "rsi": 65.0,
        }
        ans = generate_quant_expert_answer(stock_a, "HLB와 삼성전자 중 상승 가능성이 더 높은 종목은?", compare_stocks=[stock_a, stock_b])
        self.assertIn("글로벌 미장 연동 점검", ans)
        self.assertIn("나스닥", ans)



    def test_get_sector_macro_boost_semi_bull(self):
        ctx = {"soxx_pct": 2.5, "nasdaq_pct": 1.0, "usdkrw_rate": 1340.0, "vix_val": 15.0}
        res = get_sector_macro_boost("반도체/전자부품", ctx)
        self.assertEqual(res["boost"], 4.0)
        self.assertEqual(res["status"], "STRONG_BOOST")
        self.assertIn("SOXX", res["tag"])

    def test_get_sector_macro_boost_semi_bear(self):
        ctx = {"soxx_pct": -3.0, "nasdaq_pct": -1.5, "usdkrw_rate": 1390.0, "vix_val": 22.0}
        res = get_sector_macro_boost("반도체 소부장", ctx)
        self.assertEqual(res["boost"], -5.0)
        self.assertEqual(res["status"], "PENALTY")

    def test_get_sector_macro_boost_tech(self):
        ctx_bull = {"soxx_pct": 0.5, "nasdaq_pct": 2.0, "usdkrw_rate": 1340.0, "vix_val": 15.0}
        res = get_sector_macro_boost("소프트웨어/AI", ctx_bull)
        self.assertEqual(res["boost"], 3.0)
        self.assertEqual(res["status"], "BOOST")

    def test_detect_premarket_gap_anomaly_trap(self):
        ctx_bear = {"regime": "위험", "soxx_pct": -3.0, "nasdaq_pct": -2.0}
        res = detect_premarket_gap_anomaly("반도체", 4.0, ctx_bear)
        self.assertTrue(res["is_gap_trap"])
        self.assertEqual(res["score_adjustment"], -8.0)
        self.assertIn("갭트랩", res["anomaly_tag"])

    def test_detect_premarket_gap_anomaly_aligned(self):
        ctx_bull = {"regime": "우호", "soxx_pct": 2.0, "nasdaq_pct": 1.5}
        res = detect_premarket_gap_anomaly("반도체", 2.5, ctx_bull)
        self.assertFalse(res["is_gap_trap"])
        self.assertTrue(res["is_momentum_aligned"])
        self.assertEqual(res["score_adjustment"], 3.0)
        self.assertIn("모멘텀", res["anomaly_tag"])


if __name__ == "__main__":
    unittest.main()
