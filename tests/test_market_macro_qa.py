import unittest
from src.core.quant_analyst import (
    detect_market_macro_intent,
    extract_stock_and_question_from_query,
    generate_market_macro_answer,
    generate_quant_expert_answer,
    _built_in_market_macro_reasoning
)


class TestMarketMacroQA(unittest.TestCase):
    def test_detect_domestic_market_intent(self):
        queries = [
            "오늘 국내장 어때?",
            "코스피 전망 알려줘",
            "코스닥 왜 빠져?",
            "국내 증시 흐름 분석해줘",
            "선물 지수 흐름 어때?",
            "외인 수급이랑 코스피 방향성"
        ]
        for q in queries:
            res = detect_market_macro_intent(q)
            self.assertTrue(res["is_market_macro"], f"Failed for {q}")
            self.assertIn(res["category"], ["domestic_market", "market_general"], f"Wrong category for {q}: {res['category']}")

    def test_detect_us_market_intent(self):
        queries = [
            "미국증시 분석해줘",
            "오늘 미장 어때?",
            "뉴욕증시 마감 브리핑",
            "나스닥 흐름 어때?",
            "S&P500 전망 알려줘",
            "필라델피아 반도체 지수 어때?",
            "다우존스 흐름 분석"
        ]
        for q in queries:
            res = detect_market_macro_intent(q)
            self.assertTrue(res["is_market_macro"], f"Failed for {q}")
            self.assertIn(res["category"], ["us_market", "market_general"], f"Wrong category for {q}: {res['category']}")

    def test_detect_macro_economy_intent(self):
        queries = [
            "거시경제 분석해줘",
            "매크로 환경 진단해줘",
            "환율이랑 금리 영향 분석해줘",
            "원달러 환율 1400원 영향은?",
            "기준금리 인하하면 증시에 어떤 영향이 있어?",
            "인플레이션이랑 CPI 발표 영향",
            "유가 상승 영향 알려줘",
            "VIX 공포지수 어때?"
        ]
        for q in queries:
            res = detect_market_macro_intent(q)
            self.assertTrue(res["is_market_macro"], f"Failed for {q}")
            self.assertIn(res["category"], ["macro", "market_general"], f"Wrong category for {q}: {res['category']}")

    def test_detect_general_market_intent(self):
        queries = [
            "오늘 시장 전체적인 분위기 어때?",
            "글로벌 증시 흐름 알려줘",
            "지금 시장 리스크는 뭐야?"
        ]
        for q in queries:
            res = detect_market_macro_intent(q)
            self.assertTrue(res["is_market_macro"], f"Failed for {q}")

    def test_stock_queries_not_misclassified_as_macro(self):
        queries = [
            "지금 사도 되나요?",
            "목표가 & 손절가 얼마야?",
            "왜 급등하나요?",
            "차트 지표 진단해줘",
            "눌림목 매수 타점 알려줘"
        ]
        for q in queries:
            res = detect_market_macro_intent(q)
            self.assertFalse(res["is_market_macro"], f"Should not be macro: {q}")

    def test_extract_stock_and_question_routing(self):
        # 1. Macro query -> routed to MARKET
        parsed_macro = extract_stock_and_question_from_query("오늘 국내장 어때?")
        self.assertTrue(parsed_macro.get("is_market_macro"))
        self.assertEqual(parsed_macro.get("code"), "MARKET")
        self.assertEqual(parsed_macro.get("matched_by"), "market_macro")

        parsed_us = extract_stock_and_question_from_query("미국증시 분석해줘")
        self.assertTrue(parsed_us.get("is_market_macro"))
        self.assertEqual(parsed_us.get("code"), "MARKET")

        parsed_fx = extract_stock_and_question_from_query("환율이랑 금리 영향 분석해줘")
        self.assertTrue(parsed_fx.get("is_market_macro"))
        self.assertEqual(parsed_fx.get("code"), "MARKET")

        # 2. Explicit single stock query -> routed to that stock
        parsed_stock = extract_stock_and_question_from_query("삼성전자 어때?")
        self.assertFalse(parsed_stock.get("is_market_macro"))
        self.assertEqual(parsed_stock.get("code"), "005930")

        # 3. Stock query with macro context -> still targets the stock
        parsed_mixed = extract_stock_and_question_from_query("미국 증시 빠졌는데 하이닉스 어때?")
        self.assertFalse(parsed_mixed.get("is_market_macro"))
        self.assertEqual(parsed_mixed.get("code"), "000660")

        # 4. Stock follow-up query without name -> preserves active stock context
        parsed_follow = extract_stock_and_question_from_query("목표가 얼마야?", current_code="326030")
        self.assertFalse(parsed_follow.get("is_market_macro"))
        self.assertEqual(parsed_follow.get("code"), "326030")

    def test_generate_market_macro_answer_content(self):
        ans = generate_market_macro_answer("오늘 코스피와 미국 증시 흐름 분석해줘", category="market_general")
        self.assertIn("> **[수석 스트래티지스트 핵심 직답]**", ans)
        self.assertIn("실시간 국내 및 글로벌 핵심 마켓 지표 진단", ans)
        self.assertIn("시장 수급 및 매크로 리스크 팩트체크", ans)
        self.assertIn("섹터별 영향도 및 퀀트 포트폴리오 대응 전략", ans)
        self.assertIn("코스피", ans)
        self.assertIn("나스닥", ans)
        self.assertIn("원/달러 환율", ans)

    def test_generate_quant_expert_answer_delegation(self):
        # When passed a dummy stock with is_market_macro: True, delegates cleanly
        ans = generate_quant_expert_answer({"is_market_macro": True, "code": "MARKET"}, "국내장 어때?")
        self.assertIn("> **[수석 스트래티지스트 핵심 직답]**", ans)

        # When question itself is purely macro without explicit stock
        ans2 = generate_quant_expert_answer({"code": "005930", "name": "삼성전자"}, "거시경제 및 환율 분석해줘")
        self.assertIn("> **[수석 스트래티지스트 핵심 직답]**", ans2)


if __name__ == "__main__":
    unittest.main()
