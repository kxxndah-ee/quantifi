import unittest
from src.core.quant_analyst import generate_quant_expert_answer

class TestQuantAnalyst(unittest.TestCase):
    def setUp(self):
        self.sample_stock = {
            "name": "삼성전자",
            "code": "005930",
            "market": "KOSPI",
            "sector": "전기전자",
            "close": 75000,
            "change_pct": 5.4,
            "score": 88.5,
            "label": "고확신",
            "strategy_tag": "당일단타 5% 돌파",
            "tp_pct": 5.0,
            "sl_pct": 2.0,
            "vol_ratio": 3.8,
            "rsi": 62.5,
            "ma_status": "5일>20일>60일 정배열",
            "reason_core": "거래대금 전일비 3.8배 폭발 및 20일선 상향 돌파",
            "rule_note": "장중 78,750원 도달 시 즉시 익절, 73,500원 이탈 시 손절",
            "breakdown": {
                "rsi_val": 62.5,
                "vol_ratio": 3.8,
                "ma_status": "5일>20일>60일 정배열"
            }
        }

    def test_default_briefing(self):
        answer = generate_quant_expert_answer(self.sample_stock, "종목 종합 브리핑")
        self.assertIn("삼성전자", answer)
        self.assertIn("005930", answer)
        self.assertIn("전기전자", answer)
        self.assertIn("목표가", answer)

    def test_surge_reason_query(self):
        answer = generate_quant_expert_answer(self.sample_stock, "왜 오늘 이렇게 급등하나요?")
        self.assertIn("상승 원인", answer)
        self.assertIn("3.8배", answer)
        self.assertIn("정배열", answer)

    def test_entry_timing_query_normal(self):
        answer = generate_quant_expert_answer(self.sample_stock, "지금 신규 매수 들어가도 되나요?")
        self.assertIn("진입 적합도", answer)
        self.assertIn("손익비", answer)

    def test_entry_timing_query_overheated(self):
        hot_stock = dict(self.sample_stock)
        hot_stock["change_pct"] = 22.0
        hot_stock["rsi"] = 82.0
        hot_stock["breakdown"]["rsi_val"] = 82.0
        answer = generate_quant_expert_answer(hot_stock, "지금 사도 되나요?")
        self.assertIn("추격매수 절대 금지", answer)

    def test_target_price_query(self):
        answer = generate_quant_expert_answer(self.sample_stock, "목표가와 손절가는 얼마로 잡아야 하나요?")
        self.assertIn("목표 익절가", answer)
        self.assertIn("손절 기준가", answer)
        self.assertIn("78,750원", answer)
        self.assertIn("73,500원", answer)

    def test_risk_factcheck_query(self):
        answer = generate_quant_expert_answer(self.sample_stock, "어떤 리스크를 조심해야 하나요?")
        self.assertIn("리스크 팩트체크", answer)
        self.assertIn("윗꼬리", answer)

    def test_technical_indicators_query(self):
        answer = generate_quant_expert_answer(self.sample_stock, "차트와 RSI 등 지표 분석해줘")
        self.assertIn("기술적 지표", answer)
        self.assertIn("62.5", answer)

    def test_api_fallback_resilience(self):
        # Invalid API key should cleanly fallback without crashing
        answer = generate_quant_expert_answer(self.sample_stock, "상승 원인이 뭐야?", api_key="INVALID_TEST_KEY_123456789")
        self.assertIn("상승 원인", answer)
        self.assertIn("내장 퀀트", answer)

    def test_null_and_zero_division_safety(self):
        # Stock info with None values, missing breakdown, and zero sl_pct
        weird_stock = {
            "name": "테스트주",
            "code": "999999",
            "close": None,
            "change_pct": None,
            "score": None,
            "tp_pct": None,
            "sl_pct": 0.0,
            "breakdown": None
        }
        ans_entry = generate_quant_expert_answer(weird_stock, "지금 사도 되나요?")
        self.assertIn("테스트주", ans_entry)
        self.assertIn("손익비", ans_entry)

        ans_tp = generate_quant_expert_answer(weird_stock, "목표가 얼마야?")
        self.assertIn("목표 익절가", ans_tp)

        ans_why = generate_quant_expert_answer(weird_stock, "왜 급등해?")
        self.assertIn("상승 원인", ans_why)

    def test_all_five_quick_questions_unique(self):
        prompts = [
            ("왜 급등하나요?", "왜 급등하나요? 거래량과 수급 상승 원인을 정밀 분석해줘.", "상승 원인"),
            ("지금 사도 되나요?", "지금 사도 되나요? 신규 매수 진입 적합도와 손익비를 진단해줘.", "신규 진입"),
            ("목표가 & 손절가", "목표가 & 손절가 알고리즘 권장 익절가와 스탑 가격을 알려줘.", "목표가 & 손절가"),
            ("리스크 팩트체크", "리스크 팩트체크 주의해야 할 핵심 투자 위험 요소를 알려줘.", "투자 리스크"),
            ("차트 & 지표 진단", "차트 & 지표 진단 이평선과 RSI 기술적 지표를 종합 분석해줘.", "기술적 지표")
        ]
        answers = []
        for label, prompt, expected_keyword in prompts:
            ans = generate_quant_expert_answer(self.sample_stock, prompt)
            self.assertIn(expected_keyword, ans, f"Prompt '{label}' should contain '{expected_keyword}'")
            answers.append(ans)
        self.assertEqual(len(set(answers)), 5, "All 5 questions must produce distinct answers")

    def test_pre_market_opening_strategy_qa(self):
        pre_stock = dict(self.sample_stock)
        pre_stock["strategy_tag"] = "장전시초가 (주도 갭상승)"
        pre_stock["timing_label"] = "장전 (주도 갭상승)"
        ans = generate_quant_expert_answer(pre_stock, "지금 사도 되나요?")
        self.assertIn("장전 시초가", ans)
        self.assertIn("08:00~08:50", ans)
        self.assertIn("동시호가", ans)

    def test_pullback_dip_entry_query(self):
        # 1. Exact query: "로보티즈 진입 눌림목 알려줘"
        robotis = {
            "name": "로보티즈",
            "code": "108490",
            "market": "KOSDAQ",
            "sector": "로봇",
            "close": 251250,
            "change_pct": 11.67,
            "open": 235000,
            "high": 258000,
            "low": 232000,
            "prev_close": 225000,
            "volume": 1500000,
            "score": 57.5,
            "rsi": 46.4,
            "tp_pct": 5.0,
            "sl_pct": 2.5
        }
        ans = generate_quant_expert_answer(robotis, "로보티즈 진입 눌림목 알려줘")
        self.assertIn("> **[수석 애널리스트 핵심 직답]**:", ans)
        self.assertIn("눌림목 구간", ans)
        self.assertIn("1차 공격적 눌림선", ans)
        self.assertIn("2차 정석 황금 지지선", ans)
        self.assertIn("3차 최종 마지노선", ans)
        self.assertIn("246,000원 ~ 248,000원", ans)

        # 2. Query variations
        ans_calc = generate_quant_expert_answer(robotis, "진입을 위해서 눌림목 구간 계산해줘")
        self.assertIn("눌림목 구간", ans_calc)
        self.assertIn("246,000원 ~ 248,000원", ans_calc)


if __name__ == "__main__":
    unittest.main()

