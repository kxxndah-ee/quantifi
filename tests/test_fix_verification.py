import unittest
from unittest.mock import patch, MagicMock
import pandas as pd
import streamlit as st


class TestTab4ActivationAndNavigation(unittest.TestCase):
    def test_main_tab_names_defined(self):
        import app
        self.assertTrue(hasattr(app, "main_tab_names"))
        self.assertEqual(len(app.main_tab_names), 4)
        self.assertEqual(app.main_tab_names[0], "1. 스크리너")
        self.assertEqual(app.main_tab_names[1], "2. Q&A")
        self.assertEqual(app.main_tab_names[2], "3. 관심종목")
        self.assertEqual(app.main_tab_names[3], "4. 검증 및 최적화")

    def test_tab_pending_navigation_logic(self):
        main_tab_names = [
            "1. 스크리너",
            "2. Q&A",
            "3. 관심종목",
            "4. 검증 및 최적화",
        ]
        session = {}
        # Simulate button click setting pending_active_tab
        session["pending_active_tab"] = main_tab_names[3]

        # Simulate tab navigation lifecycle
        if "pending_active_tab" in session:
            session["main_active_tab_nav"] = session.pop("pending_active_tab")

        if "main_active_tab_nav" in session:
            _cur_nav = str(session["main_active_tab_nav"])
            if _cur_nav not in main_tab_names:
                if any(w in _cur_nav for w in ["4", "5", "검증", "최적화", "다이어리", "워크포워드"]):
                    session["main_active_tab_nav"] = main_tab_names[3]
                elif any(w in _cur_nav for w in ["3", "관심", "스코어보드"]):
                    session["main_active_tab_nav"] = main_tab_names[2]
                elif any(w in _cur_nav for w in ["2", "애널리스트", "팩트시트", "Q&A"]):
                    session["main_active_tab_nav"] = main_tab_names[1]
                else:
                    session["main_active_tab_nav"] = main_tab_names[0]
        else:
            session["main_active_tab_nav"] = main_tab_names[0]

        self.assertEqual(session["main_active_tab_nav"], "4. 검증 및 최적화")
        self.assertNotIn("pending_active_tab", session)

    def test_tab_fallback_heuristics(self):
        main_tab_names = [
            "1. 스크리너",
            "2. Q&A",
            "3. 관심종목",
            "4. 검증 및 최적화",
        ]
        test_cases = [
            ("검증", "4. 검증 및 최적화"),
            ("최적화", "4. 검증 및 최적화"),
            ("워크포워드", "4. 검증 및 최적화"),
            ("3. 관심", "3. 관심종목"),
            ("애널리스트", "2. Q&A"),
            ("알수없는탭", "1. 스크리너"),
        ]
        for input_tab, expected_tab in test_cases:
            session = {"main_active_tab_nav": input_tab}
            _cur_nav = str(session["main_active_tab_nav"])
            if _cur_nav not in main_tab_names:
                if any(w in _cur_nav for w in ["4", "5", "검증", "최적화", "다이어리", "워크포워드"]):
                    session["main_active_tab_nav"] = main_tab_names[3]
                elif any(w in _cur_nav for w in ["3", "관심", "스코어보드"]):
                    session["main_active_tab_nav"] = main_tab_names[2]
                elif any(w in _cur_nav for w in ["2", "애널리스트", "팩트시트", "Q&A"]):
                    session["main_active_tab_nav"] = main_tab_names[1]
                else:
                    session["main_active_tab_nav"] = main_tab_names[0]
            self.assertEqual(session["main_active_tab_nav"], expected_tab)

    def test_pit_verification_cached_function(self):
        import app
        self.assertTrue(hasattr(app, "_get_quick_pit_verification_res"))
        res = app._get_quick_pit_verification_res("2026-09-22", "실시간 당일 단타 (5% 익절)")
        self.assertIsInstance(res, dict)
        self.assertIn("df_results", res)

    def test_render_unified_verification_and_tuning_section_runs(self):
        import app

        def mock_columns(spec, *args, **kwargs):
            n = len(spec) if isinstance(spec, (list, tuple)) else int(spec)
            return [MagicMock() for _ in range(n)]

        # Mock streamlit widgets to test execution of render_unified_verification_and_tuning_section
        with patch.object(st, "columns", side_effect=mock_columns):
            with patch.object(st, "markdown"):
                with patch.object(st, "caption"):
                    with patch.object(st, "button", return_value=False):
                        with patch.object(st, "metric"):
                            with patch.object(st, "table"):
                                with patch.object(st, "dataframe"):
                                    with patch.object(st, "expander", return_value=MagicMock()):
                                        # Should run without error
                                        app.render_unified_verification_and_tuning_section()


if __name__ == "__main__":
    unittest.main()
