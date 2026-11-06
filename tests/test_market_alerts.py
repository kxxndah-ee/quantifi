"""
Unit tests for Market Warning Alerts & Sidecar Detection.
Strictly NO EMOJIS in code, logs, comments, or output.
"""

import unittest
from unittest.mock import patch, MagicMock

from src.collectors.market_alerts import (
    evaluate_market_warning_alerts,
    get_realtime_market_alerts,
    clear_alert_state
)
from datetime import datetime, timedelta


class TestMarketAlerts(unittest.TestCase):
    def test_normal_market_regime(self):
        domestic_summary = {
            "items": [
                {"code": "KOSPI", "name": "코스피", "close": 2600.0, "change_pct": 0.50},
                {"code": "KOSDAQ", "name": "코스닥", "close": 750.0, "change_pct": -0.80},
                {"code": "KPI200", "name": "코스피200", "close": 350.0, "change_pct": 0.40},
                {"code": "FUT", "name": "코스피200선물", "close": 350.0, "change_pct": 0.35},
                {"code": "KQI150", "name": "코스닥 150", "close": 1200.0, "change_pct": -0.90},
                {"code": "KVALUE", "name": "코리아밸류업", "close": 1000.0, "change_pct": 0.20},
                {"code": "BTC", "name": "비트코인", "close": 105000000.0, "change_pct": 1.10},
            ]
        }
        res = evaluate_market_warning_alerts(domestic_summary=domestic_summary)
        self.assertEqual(res["level"], "NORMAL")
        self.assertIn("정상", res["badge_text"])
        self.assertEqual(len(res["alerts"]), 0)

    def test_sidecar_watch_fut(self):
        # KOSPI 200 futures at -3.80% (watch threshold: +/- 3.5%)
        domestic_summary = {
            "items": [
                {"code": "KOSPI", "name": "코스피", "close": 2500.0, "change_pct": -2.20},
                {"code": "KOSDAQ", "name": "코스닥", "close": 740.0, "change_pct": -1.50},
                {"code": "FUT", "name": "코스피200선물", "close": 335.0, "change_pct": -3.80},
                {"code": "KQI150", "name": "코스닥 150", "close": 1180.0, "change_pct": -2.00},
            ]
        }
        res = evaluate_market_warning_alerts(domestic_summary=domestic_summary)
        self.assertEqual(res["level"], "WARNING")
        self.assertIn("사이드카 주의보", res["badge_text"])
        fut_alerts = [a for a in res["alerts"] if a["target_code"] == "FUT"]
        self.assertEqual(len(fut_alerts), 1)
        self.assertEqual(fut_alerts[0]["type"], "SIDECAR_WATCH")
        self.assertIn("-3.80%", fut_alerts[0]["title"])

    def test_sidecar_trigger_fut_sell(self):
        # KOSPI 200 futures at -5.20% (trigger threshold: -5.0%)
        domestic_summary = {
            "items": [
                {"code": "KOSPI", "name": "코스피", "close": 2400.0, "change_pct": -4.20},
                {"code": "FUT", "name": "코스피200선물", "close": 320.0, "change_pct": -5.20},
            ]
        }
        res = evaluate_market_warning_alerts(domestic_summary=domestic_summary)
        self.assertEqual(res["level"], "CRITICAL")
        self.assertIn("사이드카", res["badge_text"])
        self.assertIn("매도 사이드카", res["alerts"][0]["title"])

    def test_sidecar_trigger_kosdaq_buy(self):
        # KOSDAQ 150 at +6.20% (trigger threshold: +6.0%)
        domestic_summary = {
            "items": [
                {"code": "KQI150", "name": "코스닥 150", "close": 1350.0, "change_pct": 6.20},
            ]
        }
        res = evaluate_market_warning_alerts(domestic_summary=domestic_summary)
        self.assertEqual(res["level"], "CRITICAL")
        self.assertIn("매수 사이드카", res["alerts"][0]["title"])

    def test_circuit_breaker_stage1(self):
        # KOSPI at -8.50% (trigger threshold: -8.0%)
        domestic_summary = {
            "items": [
                {"code": "KOSPI", "name": "코스피", "close": 2350.0, "change_pct": -8.50},
                {"code": "FUT", "name": "코스피200선물", "close": 310.0, "change_pct": -4.00},
            ]
        }
        res = evaluate_market_warning_alerts(domestic_summary=domestic_summary)
        self.assertEqual(res["level"], "CRITICAL")
        cb_alerts = [a for a in res["alerts"] if a["type"] == "CIRCUIT_BREAKER"]
        self.assertEqual(len(cb_alerts), 1)
        self.assertIn("서킷브레이커 1단계", cb_alerts[0]["title"])

    def test_circuit_breaker_watch(self):
        # KOSDAQ at -5.50% (watch threshold: -5.0%)
        domestic_summary = {
            "items": [
                {"code": "KOSDAQ", "name": "코스닥", "close": 700.0, "change_pct": -5.50},
            ]
        }
        res = evaluate_market_warning_alerts(domestic_summary=domestic_summary)
        self.assertEqual(res["level"], "WARNING")
        self.assertIn("서킷브레이커 주의보", res["badge_text"])

    def test_global_macro_vix_shock(self):
        domestic_summary = {"items": []}
        us_summary = {
            "vix_val": 27.5,
            "vix_pct": 22.0,
            "usdkrw_rate": 1380.0
        }
        res = evaluate_market_warning_alerts(domestic_summary=domestic_summary, us_summary=us_summary)
        self.assertEqual(res["level"], "WARNING")
        vix_alerts = [a for a in res["alerts"] if a["target_code"] == "VIX"]
        self.assertEqual(len(vix_alerts), 1)
        self.assertIn("VIX 공포지수", vix_alerts[0]["title"])

    def test_sidecar_trigger_and_expiration_timing(self):
        clear_alert_state()
        t0 = datetime(2026, 11, 6, 10, 15, 0)
        domestic_summary = {
            "items": [
                {"code": "KOSPI", "name": "코스피", "close": 2400.0, "change_pct": -4.20},
                {"code": "FUT", "name": "코스피200선물", "close": 320.0, "change_pct": -5.20},
            ]
        }
        res1 = evaluate_market_warning_alerts(domestic_summary=domestic_summary, now_kst=t0)
        self.assertEqual(res1["level"], "CRITICAL")
        self.assertIn("사이드카", res1["badge_text"])
        self.assertEqual(res1["triggered_time"], "10:15:00")
        self.assertEqual(res1["end_time"], "10:20:00")
        self.assertIn("5분간 효력 정지", res1["duration_str"])
        self.assertIn("발동 10:15:00", res1["time_info"])
        self.assertIn("종료 10:20:00", res1["time_info"])

        # Check alert item
        alt = res1["alerts"][0]
        self.assertEqual(alt["triggered_time"], "10:15:00")
        self.assertEqual(alt["end_time"], "10:20:00")
        self.assertEqual(alt["duration_minutes"], 5)
        self.assertIn("발동 10:15:00", alt["title"])
        self.assertIn("종료 10:20:00", alt["title"])

        # 10 seconds polling later - trigger time must be frozen, countdown updated
        t1 = t0 + timedelta(seconds=10)
        res2 = evaluate_market_warning_alerts(domestic_summary=domestic_summary, now_kst=t1)
        self.assertEqual(res2["triggered_time"], "10:15:00")
        self.assertEqual(res2["end_time"], "10:20:00")
        self.assertEqual(res2["remaining_str"], "잔여 4분 50초")
        self.assertEqual(res2["alert_status"], "발동 중")

        # 5 minutes and 10 seconds later - status should be expired
        t2 = t0 + timedelta(seconds=310)
        res3 = evaluate_market_warning_alerts(domestic_summary=domestic_summary, now_kst=t2)
        self.assertEqual(res3["triggered_time"], "10:15:00")
        self.assertEqual(res3["end_time"], "10:20:00")
        self.assertEqual(res3["remaining_str"], "해제 완료")
        self.assertEqual(res3["alert_status"], "발동 종료 (해제)")
        clear_alert_state()

    def test_circuit_breaker_timing(self):
        clear_alert_state()
        t0 = datetime(2026, 11, 6, 9, 30, 0)
        domestic_summary = {
            "items": [
                {"code": "KOSPI", "name": "코스피", "close": 2350.0, "change_pct": -8.50},
            ]
        }
        res = evaluate_market_warning_alerts(domestic_summary=domestic_summary, now_kst=t0)
        self.assertEqual(res["level"], "CRITICAL")
        self.assertEqual(res["triggered_time"], "09:30:00")
        self.assertEqual(res["end_time"], "09:50:00")
        self.assertIn("20분간 매매 중단", res["duration_str"])
        self.assertIn("종료 09:50:00", res["time_info"])

        # CB Stage 3 (Market closed)
        clear_alert_state()
        domestic_summary_stage3 = {
            "items": [
                {"code": "KOSPI", "name": "코스피", "close": 2000.0, "change_pct": -21.00},
            ]
        }
        res_cb3 = evaluate_market_warning_alerts(domestic_summary=domestic_summary_stage3, now_kst=t0)
        self.assertEqual(res_cb3["triggered_time"], "09:30:00")
        self.assertEqual(res_cb3["end_time"], "당일 장 마감")
        self.assertIn("당일 거래 전면 마감", res_cb3["duration_str"])
        clear_alert_state()


if __name__ == "__main__":
    unittest.main()
