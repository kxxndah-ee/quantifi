import unittest
import os
import sys

# Ensure project root is in path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.database.models import get_db_connection
from src.collectors.market_data import get_latest_market_info, fetch_ohlcv
from src.core.scoring import evaluate_stock_latest
from src.core.quant_analyst import generate_quant_expert_answer, extract_stock_and_question_from_query


class TestAnalystTab(unittest.TestCase):
    def test_factsheet_payload_logic(self):
        code = "005930"
        info = get_latest_market_info(code)
        self.assertIsNotNone(info)
        self.assertIn("close", info)
        self.assertIn("change_pct", info)

        df = fetch_ohlcv(code)
        self.assertFalse(df.empty)

        score_info = evaluate_stock_latest(df)
        self.assertIsNotNone(score_info)
        self.assertIn("score", score_info)
        self.assertIn("label", score_info)
        self.assertIn("breakdown", score_info)

        close = float(info.get("close", 0.0))
        tp_pct = float(score_info.get("tp_pct", 5.0))
        sl_pct = float(score_info.get("sl_pct", 2.0))
        tp_price = close * (1.0 + tp_pct / 100.0) if close > 0 else 0.0
        sl_price = close * (1.0 - sl_pct / 100.0) if close > 0 else 0.0
        rr_ratio = tp_pct / max(0.1, sl_pct)

        self.assertGreater(tp_price, close)
        self.assertLess(sl_price, close)
        self.assertGreater(rr_ratio, 0.0)

    def test_analyst_qa_answers_for_stock(self):
        sample = {
            "name": "삼성전자",
            "code": "005930",
            "market": "KOSPI",
            "sector": "전기/전자",
            "close": 75000,
            "change_pct": 3.5,
            "score": 80.0,
            "label": "우수",
            "strategy_tag": "고확신 퀀트 모멘텀",
            "tp_pct": 5.0,
            "sl_pct": 2.5,
            "vol_ratio": 2.5,
            "rsi": 62.0,
            "ma_status": "이평 정배열",
            "reason_core": "거래량 2.5배 폭발 및 정배열 모멘텀",
            "rule_note": "목표가 도달 시 50% 분할 익절",
            "breakdown": {
                "rsi_val": 62.0,
                "vol_ratio": 2.5,
                "ma_status": "이평 정배열"
            }
        }
        ans = generate_quant_expert_answer(sample, "왜 급등하나요?")
        self.assertIn("삼성전자", ans)
        self.assertIn("상승 원인", ans)

        ans_buy = generate_quant_expert_answer(sample, "지금 사도 되나요?")
        self.assertIn("진입 적합도", ans_buy)

    def test_app_py_tab_configuration(self):
        # Inspect app.py text to verify main_tab_names definition
        app_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "app.py")
        with open(app_path, "r", encoding="utf-8") as f:
            content = f.read()

        self.assertIn('"1. 스크리너"', content)
        self.assertIn('"2. Q&A"', content)
        self.assertIn('"3. 관심종목"', content)
        self.assertIn('"4. 검증 및 최적화"', content)
        self.assertIn("tab1, tab2, tab3, tab4 = st.tabs(main_tab_names", content)
        self.assertIn("render_analyst_factsheet_and_qa_tab()", content)
        self.assertIn("with tab1:", content)
        self.assertIn("with tab2:", content)
        self.assertIn("with tab3:", content)
        self.assertIn("with tab4:", content)
        self.assertIn('st.session_state["sc_scope_sel"] = "전 상장사 전수조사 (약 2,700개)"', content)
        self.assertIn('[스크리너 필터 설정] 전 상장사 전수조사', content)

        # Tab 2 structure assertions:
        # 1. Direct search only (no quick chips, no dropdown)
        self.assertIn("qa_direct_search_kw", content)
        self.assertNotIn("빠른 종목 선택 칩", content)
        self.assertNotIn("qa_stock_search_selectbox", content)

        # 2. Compact metrics present
        self.assertIn("종합 퀀트 스코어", content)
        self.assertIn("가격 전략 & 손익비", content)
        self.assertIn("수급 및 거래량 폭발력", content)
        self.assertIn("기술적 지표 & 이평선", content)

        # 3. Console 1: 목표가&손절가, 리스크 팩트체크, 차트&지표진단 (3열 요약 카드 + 접이식 전문 보기, 선택 버튼 삭제)
        self.assertNotIn("btn_diag_tp_", content)
        self.assertNotIn("btn_diag_risk_", content)
        self.assertNotIn("btn_diag_chart_", content)
        self.assertIn("목표가 & 손절가", content)
        self.assertIn("리스크 팩트체크", content)
        self.assertIn("차트 & 지표 진단", content)
        self.assertIn("상세 진단 리포트 전문 보기", content)

        # 4. Tab 1 label & Tab 2 layout assertions:
        self.assertNotIn("스크리닝 전략 모드 선택 (아래 6개 중 1개 선택)", content)
        self.assertIn('"스크리닝 전략 모드 선택"', content)

        tab2_fn_text = content[content.index("def render_analyst_factsheet_and_qa_tab():"):content.index("main_tab_names =")]
        self.assertNotIn('<div class="office-subheading">Q&A</div>', tab2_fn_text)
        self.assertNotIn("수석 애널리스트 실시간 Q&A 콘솔", tab2_fn_text)
        self.assertIn('placeholder="질문을 입력하세요"', tab2_fn_text)
        self.assertNotIn("비교 종목 팩트시트 전환:", tab2_fn_text)
        self.assertIn("qa_custom_form_", tab2_fn_text)
        self.assertIn("질문: <span", tab2_fn_text)
        self.assertNotIn("선택 질문:", tab2_fn_text)
        self.assertIn("btn_qa_buy_", tab2_fn_text)
        self.assertIn("btn_qa_why_", tab2_fn_text)
        # Question Input -> Answer Output -> Stock Header Box (with Quick Buttons inside) -> Core Cards
        self.assertLess(tab2_fn_text.index("qa_custom_form_"), tab2_fn_text.index("질문: <span"))
        self.assertLess(tab2_fn_text.index("질문: <span"), tab2_fn_text.index("btn_qa_buy_"))
        self.assertLess(tab2_fn_text.index("btn_qa_buy_"), tab2_fn_text.index("btn_qa_why_"))
        self.assertLess(tab2_fn_text.index("btn_qa_why_"), tab2_fn_text.index("종합 퀀트 스코어"))

    def test_question_first_stock_and_question_extraction(self):
        # 1. Stock name + question
        res1 = extract_stock_and_question_from_query("삼성전자 지금 사도 되나요?", current_code="000660")
        self.assertEqual(res1["code"], "005930")
        self.assertEqual(res1["matched_by"], "name")
        self.assertFalse(res1["is_stock_only"])
        self.assertIn("지금 사도 되나요", res1["question"])

        # 2. 6-digit stock code + question
        res2 = extract_stock_and_question_from_query("000660 내일 시초가에 어떻게 대응해야 하나요?", current_code="005930")
        self.assertEqual(res2["code"], "000660")
        self.assertEqual(res2["matched_by"], "code")
        self.assertFalse(res2["is_stock_only"])

        # 3. General market recommendation query
        res3 = extract_stock_and_question_from_query("지금 매수할 만한 종목은?", current_code="005930", fallback_top_code="035420")
        self.assertEqual(res3["code"], "035420")
        self.assertEqual(res3["matched_by"], "recommend")
        self.assertFalse(res3["is_stock_only"])

        # 4. Follow-up question without stock name keeps current_code
        res4 = extract_stock_and_question_from_query("목표가와 손절가는 얼마인가요?", current_code="005380")
        self.assertEqual(res4["code"], "005380")
        self.assertEqual(res4["matched_by"], "context")
        self.assertFalse(res4["is_stock_only"])

        # 5. Stock name only triggers default entry suitability question
        res5 = extract_stock_and_question_from_query("삼성전자", current_code="000660")
        self.assertEqual(res5["code"], "005930")
        self.assertTrue(res5["is_stock_only"])

        # 6. Multi-stock comparative question ('HLB와 삼성전자 중 오후에 상승 가능성이 더 높은 종목은?')
        res6 = extract_stock_and_question_from_query("HLB와 삼성전자 중 오후에 상승 가능성이 더 높은 종목은?", current_code="000660")
        self.assertEqual(res6["matched_by"], "multi")
        self.assertTrue(res6["is_comparative"])
        self.assertEqual(res6["codes"], ["028300", "005930"])

    def test_multi_stock_comparative_quant_answer(self):
        hlb_sample = {
            "name": "HLB",
            "code": "028300",
            "market": "KOSDAQ",
            "sector": "제약/바이오",
            "close": 48900,
            "open": 49500,
            "high": 50200,
            "low": 48500,
            "prev_close": 50350,
            "volume": 1200000,
            "change_pct": -2.88,
            "score": 47.0,
            "label": "중립",
            "strategy_tag": "일반 퀀트 관찰",
            "tp_pct": 5.0,
            "sl_pct": 2.5,
            "vol_ratio": 0.6,
            "rsi": 39.2,
            "ma_status": "이평 역배열",
            "bb_pct": 0.42,
            "breakdown": {"rsi_val": 39.2, "vol_ratio": 0.6, "ma_status": "이평 역배열", "bb_pct": 0.42}
        }
        sec_sample = {
            "name": "삼성전자",
            "code": "005930",
            "market": "KOSPI",
            "sector": "전기/전자",
            "close": 75000,
            "open": 73500,
            "high": 75300,
            "low": 73100,
            "prev_close": 72500,
            "volume": 15000000,
            "change_pct": 3.45,
            "score": 80.0,
            "label": "우수",
            "strategy_tag": "고확신 퀀트 모멘텀",
            "tp_pct": 5.0,
            "sl_pct": 2.5,
            "vol_ratio": 2.5,
            "rsi": 62.0,
            "ma_status": "이평 정배열",
            "bb_pct": 0.72,
            "breakdown": {"rsi_val": 62.0, "vol_ratio": 2.5, "ma_status": "이평 정배열", "bb_pct": 0.72}
        }
        q = "HLB와 삼성전자 중 오후에 상승 가능성이 더 높은 종목은?"
        comp_ans = generate_quant_expert_answer(sec_sample, q, compare_stocks=[hlb_sample, sec_sample])
        self.assertIn("HLB", comp_ans)
        self.assertIn("삼성전자", comp_ans)
        self.assertIn("오후장 상승 가능성 점수", comp_ans)
        self.assertIn("최우선 공략 1위", comp_ans)

    def test_custom_questions_produce_distinct_quant_results(self):
        sample = {
            "name": "삼성전자",
            "code": "005930",
            "market": "KOSPI",
            "sector": "전기/전자",
            "close": 75000,
            "open": 73500,
            "high": 76200,
            "low": 73100,
            "prev_close": 72500,
            "volume": 15000000,
            "change_pct": 3.45,
            "score": 80.0,
            "label": "우수",
            "strategy_tag": "고확신 퀀트 모멘텀",
            "tp_pct": 5.0,
            "sl_pct": 2.5,
            "vol_ratio": 2.5,
            "rsi": 62.0,
            "ma_status": "이평 정배열",
            "bb_pct": 0.72,
            "reason_core": "거래량 2.5배 폭발 및 정배열 모멘텀",
            "rule_note": "목표가 도달 시 50% 분할 익절",
            "breakdown": {
                "rsi_val": 62.0,
                "vol_ratio": 2.5,
                "ma_status": "이평 정배열",
                "bb_pct": 0.72
            }
        }
        custom_questions = [
            "오후에 상승 가능성이 얼마나 되나요?",
            "내일 시초가에 어떻게 대응해야 하나요?",
            "보유 중인데 언제 매도할까요?",
            "지지선과 저항선이 어디인가요?",
            "수급이랑 거래량 상태는 어때?",
            "비중은 얼마나 들어가면 될까요?",
            "단타가 좋나요 스윙이 좋나요?",
            "앞으로 전반적인 주가 전망이 어때요?"
        ]
        answers = []
        for q in custom_questions:
            ans = generate_quant_expert_answer(sample, q)
            self.assertIn(q, ans)
            self.assertIn("퀀트 산출 결과값", ans)
            answers.append(ans)
        self.assertEqual(len(set(answers)), len(custom_questions))

    def test_calculate_analyzed_rise_rate(self):
        from src.core.scoring import calculate_analyzed_rise_rate

        # Strong bullish stock
        rate_bull = calculate_analyzed_rise_rate(
            total_score=85.0,
            vol_ratio=3.5,
            rsi_val=60.0,
            ma_status="정배열",
            change_pct=4.2,
            strategy_tag="당일단타 수급돌파",
            bb_pct=0.75
        )
        self.assertGreaterEqual(rate_bull, 5.0)

        # Neutral stock
        rate_neutral = calculate_analyzed_rise_rate(
            total_score=55.0,
            vol_ratio=1.1,
            rsi_val=50.0,
            ma_status="수렴",
            change_pct=0.5,
            strategy_tag="-",
            bb_pct=0.5
        )
        self.assertGreater(rate_neutral, 1.0)
        self.assertLess(rate_neutral, rate_bull)

    def test_watchlist_tab_uses_analyzed_rise_rate(self):
        app_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "app.py")
        with open(app_path, "r", encoding="utf-8") as f:
            content = f.read()

        # Locate Tab 3 block
        tab3_start = content.find("with tab3:")
        tab4_start = content.find("with tab4:")
        self.assertGreater(tab3_start, 0)
        self.assertGreater(tab4_start, tab3_start)

        tab3_content = content[tab3_start:tab4_start]

        # Verify Tab 3 has "분석 상승률" in table and sorting
        self.assertIn('"분석 상승률 높은 순"', tab3_content)
        self.assertIn('"analyzed_gain_pct"', tab3_content)
        self.assertIn('"분석 상승률"', tab3_content)
        self.assertIn("분석 상승률 / 손절", tab3_content)
        self.assertIn("분석 상승률 & 손절선", tab3_content)
        self.assertIn("분석 상승률: +", tab3_content)

        # Verify Tab 3 does NOT have "목표익절" or "목표 익절가 높은 순"
        self.assertNotIn('"목표 익절가 높은 순"', tab3_content)
        self.assertNotIn('"목표익절"', tab3_content)

    def test_screener_qa_hub_cleanup_and_no_scrollbar(self):
        app_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "app.py")
        with open(app_path, "r", encoding="utf-8") as f:
            content = f.read()

        # 1. Verify text and selectbox from media_1791877801284.png are removed
        self.assertNotIn("AI 퀀트 수석 애널리스트 실시간 질의응답 (Q&A 허브)", content)
        self.assertNotIn("스크리닝된 상승 유력 종목 중 궁금한 종목을 선택하여", content)
        self.assertNotIn("분석 및 질의응답할 스크리닝 종목 선택", content)
        self.assertNotIn("qa_hub_stock_selectbox", content)
        self.assertNotIn("[안내] 상단 테이블 클릭 또는 카드 버튼 클릭 시 자동 연동", content)

        # 2. Verify scrollbar prevention CSS on question buttons
        self.assertIn("overflow-x: hidden !important;", content)
        self.assertIn("scrollbar-width: none !important;", content)
        self.assertIn("display: none !important;", content)

    def test_sync_and_reset_buttons_uniform_sizing(self):
        app_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "app.py")
        with open(app_path, "r", encoding="utf-8") as f:
            content = f.read()

        # Both buttons should be in identical column width ratio [0.9, 0.9, 4.2]
        self.assertIn("st.columns([0.9, 0.9, 4.2], gap=\"small\")", content)

        # sync_btn should NOT have help= parameter which caused unequal size/DOM wrapper
        self.assertIn('sync_btn = st.button("시세 동기화", key="btn_qa_sync_market", use_container_width=True)', content)
        self.assertIn('reset_btn = st.button("질의 초기화", key=f"reset_chat_{curr_target_code}_tab2", use_container_width=True)', content)

        # CSS rule should uniformly enforce height and font across all buttons
        self.assertIn('height: 2rem !important;', content)
        self.assertIn('min-height: 2rem !important;', content)

    def test_screener_mode_default_setting(self):
        app_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "app.py")
        with open(app_path, "r", encoding="utf-8") as f:
            content = f.read()

        # Verify that "실시간 당일 단타 (5% 익절)" is set as the top default for sc_mode
        self.assertIn('"실시간 당일 단타 (5% 익절)"', content)
        self.assertIn('st.session_state["sc_mode_radio_sel"] = "실시간 당일 단타 (5% 익절)"', content)

    def test_screener_qa_box_title_and_order(self):
        app_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "app.py")
        with open(app_path, "r", encoding="utf-8") as f:
            content = f.read()

        qa_box_text = content[content.index("def render_quant_analyst_qa_box(stock_row, key_suffix=\"\"):"):content.index("def render_analyst_factsheet_and_qa_tab():")]

        # 1. Verify "수석 애널리스트 실시간 Q&A 콘솔" is deleted from Tab 1 screener box
        self.assertNotIn("수석 애널리스트 실시간 Q&A 콘솔", qa_box_text)

        # 2. Verify "지금 사도 되나요?" is first and "왜 급등하나요?" is second in quick_queries and buttons
        self.assertLess(qa_box_text.index('"btn_quick_buy_'), qa_box_text.index('"btn_quick_why_'))

    def test_top_sync_clears_cache_and_qa_market_data(self):
        app_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "app.py")
        with open(app_path, "r", encoding="utf-8") as f:
            content = f.read()

        # Verify top sync button clears cache, Q&A caches, and active answers
        top_sync_section = content[content.index('key="btn_top_time_sync"'):content.index('def _safe_float_val')]
        self.assertIn("st.cache_data.clear()", top_sync_section)
        self.assertIn('qa_ans_cache_', top_sync_section)
        self.assertIn('quant_cache_', top_sync_section)
        self.assertIn('qa_active_ans_', top_sync_section)
        self.assertIn('quant_active_ans_', top_sync_section)
        self.assertIn('st.session_state["sc_trigger_time_sync"] = True', top_sync_section)

        # Verify Tab 2 sync button pops active answer and clears compared caches
        tab2_sync_section = content[content.index('if sync_btn:'):content.index('if reset_btn:')]
        self.assertIn("st.session_state.pop(qa_active_ans_key, None)", tab2_sync_section)
        self.assertIn("qa_compared_codes", tab2_sync_section)

    def test_expander_no_scrollbar_css(self):
        app_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "app.py")
        with open(app_path, "r", encoding="utf-8") as f:
            content = f.read()

        # Verify expander overflow visible and scrollbar suppression
        self.assertIn('div[data-testid="stExpanderDetails"]', content)
        self.assertIn('scrollbar-width: none !important;', content)
        self.assertIn('div[data-testid="stExpanderDetails"]::-webkit-scrollbar', content)

    def test_qa_header_container_no_scrollbar(self):
        app_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "app.py")
        with open(app_path, "r", encoding="utf-8") as f:
            content = f.read()

        # 1. Verify stVerticalBlockBorderWrapper has overflow visible and scrollbar none
        self.assertIn('div[data-testid="stVerticalBlockBorderWrapper"]', content)
        self.assertIn('div[data-testid="stVerticalBlockBorderWrapper"]::-webkit-scrollbar', content)

    def test_quant_analyst_multi_turn_chat_history(self):
        sample = {
            "name": "삼성전자",
            "code": "005930",
            "market": "KOSPI",
            "sector": "전기/전자",
            "close": 75000,
            "change_pct": 3.5,
            "score": 80.0,
            "label": "우수",
            "strategy_tag": "고확신 퀀트 모멘텀",
            "tp_pct": 5.0,
            "sl_pct": 2.5,
            "vol_ratio": 2.5,
            "rsi": 62.0,
            "ma_status": "이평 정배열",
            "reason_core": "거래량 2.5배 폭발 및 정배열 모멘텀",
            "rule_note": "목표가 도달 시 50% 분할 익절",
            "breakdown": {
                "rsi_val": 62.0,
                "vol_ratio": 2.5,
                "ma_status": "이평 정배열"
            }
        }
        # 1. First turn: initial question without history
        ans_1 = generate_quant_expert_answer(sample, "지금 사도 되나요?", chat_history=[])
        self.assertIn("진입 적합도", ans_1)
        self.assertNotIn("[직전 질문 연계 심화(디벨롭) 분석]", ans_1)

        # 2. Second turn: follow-up question with history
        history = [{"q": "지금 사도 되나요?", "ans": ans_1}]
        ans_2 = generate_quant_expert_answer(sample, "그럼 손절가는 얼마로 잡을까?", chat_history=history)
        self.assertIn("[직전 질문 연계 심화(디벨롭) 분석]", ans_2)
        self.assertIn("지금 사도 되나요?", ans_2)

        # 3. Verify app.py wire-up for chat_history and reset
        app_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "app.py")
        with open(app_path, "r", encoding="utf-8") as f:
            content = f.read()

        # Tab 1 wires chat_history
        self.assertIn("chat_history=st.session_state.get(hist_key, [])", content)
        # Tab 2 wires chat_history
        self.assertIn("chat_history=st.session_state[target_hist_key]", content)
        self.assertIn("chat_history=st.session_state.get(qa_hist_key, [])", content)
        # Reset button clears history
        self.assertIn("st.session_state[hist_key] = []", content)
        self.assertIn("st.session_state[qa_hist_key] = []", content)

    def test_compact_font_sizes_and_overflow_styling(self):
        app_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "app.py")
        with open(app_path, "r", encoding="utf-8") as f:
            content = f.read()

        # 1. Verify CSS styles for compact Q&A fonts (0.80rem)
        self.assertIn(".quant-qa-body-wrap", content)
        self.assertIn('div[data-testid="stChatMessage"] p', content)
        self.assertIn('font-size: 0.80rem !important;', content)

        # 2. Verify Tab 1 and Tab 2 wrap answers in quant-qa-body-wrap
        self.assertIn("<div class='quant-qa-body-wrap'>", content)

        # 3. Verify compact alert styling and theme clustering warning styling
        self.assertIn('div[data-testid="stAlert"]', content)
        self.assertIn('<strong>[테마 몰림 경고]</strong>', content)

        # 4. Verify overflow: visible and scrollbar suppression for element-container and stVerticalBlock
        self.assertIn('div[data-testid="element-container"]', content)
        self.assertIn('div[data-testid="element-container"]::-webkit-scrollbar', content)

    def test_direct_answer_block_and_afternoon_upside_routing(self):
        sample = {
            "name": "HT로보틱스",
            "code": "396300",
            "market": "KOSDAQ",
            "sector": "로봇/자동화",
            "close": 40000,
            "open": 37500,
            "high": 41200,
            "low": 37000,
            "prev_close": 36500,
            "volume": 3500000,
            "change_pct": 9.59,
            "score": 82.5,
            "label": "고확신",
            "strategy_tag": "당일단타 수급돌파",
            "tp_pct": 5.0,
            "sl_pct": 2.0,
            "vol_ratio": 2.8,
            "rsi": 66.5,
            "ma_status": "이평 정배열",
            "bb_pct": 0.78,
            "reason_core": "거래량 2.8배 폭발 및 로봇 섹터 수급 집중",
            "rule_note": "목표가 42,000원 분할 익절 및 39,200원 원칙 손절",
            "breakdown": {
                "rsi_val": 66.5,
                "vol_ratio": 2.8,
                "ma_status": "이평 정배열",
                "bb_pct": 0.78
            }
        }

        # 1. Exact user question: "HT로보틱스 1시 이후에 상승 여력은?"
        user_q = "HT로보틱스 1시 이후에 상승 여력은?"
        ans_upside = generate_quant_expert_answer(sample, user_q)

        # Must have clear direct answer block at the top
        self.assertIn("> **[수석 애널리스트 핵심 직답]**:", ans_upside)
        self.assertIn("1시 이후", ans_upside)
        self.assertIn("추가 상승 여력", ans_upside)
        # Must NOT be diverted into generic "상승 원인"
        self.assertNotIn("상승 원인 퀀트 정밀 진단", ans_upside)
        self.assertIn("1시 이후", ans_upside)
        self.assertIn("상승 여력 퀀트 정밀 진단", ans_upside)

        # 1-1. Real-time dynamic question: "HT로보틱스 현시각 이후에 추가 상승 여력은?"
        user_q_now = "HT로보틱스 현시각 이후에 추가 상승 여력은?"
        ans_now = generate_quant_expert_answer(sample, user_q_now)
        self.assertIn("> **[수석 애널리스트 핵심 직답]**:", ans_now)
        self.assertIn("현시각(", ans_now)
        self.assertNotIn("1시 이후", ans_now)
        self.assertIn("추가 상승 여력", ans_now)

        # 2. Preset and custom questions must ALL contain the direct answer block
        test_questions = [
            ("왜 급등하나요?", "급등 원인"),
            ("지금 사도 되나요?", "지금 매수"),
            ("목표가와 손절가는?", "목표 익절가"),
            ("리스크는 무엇인가요?", "핵심 리스크"),
            ("지지선과 저항선은 어디인가요?", "핵심 1차 지지선"),
            ("단타가 좋나요 스윙이 좋나요?", "매매에 가장 최적화"),
            ("비중은 얼마나 들어갈까요?", "권장 진입 비중"),
            ("내일 시초가 대응은?", "시초가 대응 판정"),
            ("수급 상태는 어때?", "수급 상태는"),
        ]

        for q, expected_phrase in test_questions:
            res = generate_quant_expert_answer(sample, q)
            self.assertIn("> **[수석 애널리스트 핵심 직답]**:", res, f"Missing direct answer block for query: {q}")
            self.assertIn(expected_phrase, res, f"Missing expected phrase '{expected_phrase}' for query: {q}")

    def test_lg_cns_query_maps_to_064400(self):
        # 1. 엘지씨앤에스 variations must resolve to 064400 (LG씨엔에스)
        for q in [
            "엘지씨앤에스 오늘 전망해줘",
            "lg씨앤에스 오늘 전망해줘",
            "LG씨앤에스 오늘 전망해줘",
            "LG CNS 오늘 전망해줘",
            "씨앤에스 주가 어때?",
            "엘지cns 전망 분석해줘",
        ]:
            res = extract_stock_and_question_from_query(q, current_code="005930")
            self.assertEqual(res["code"], "064400", f"Failed code for query: {q}")
            self.assertEqual(res["matched_by"], "name", f"Failed matched_by for query: {q}")
            self.assertFalse(res["is_stock_only"], f"Should not be stock only for query: {q}")

        # 2. Stock-only input for LG CNS triggers default entry question for 064400
        res_stock_only = extract_stock_and_question_from_query("LG CNS", current_code="005930")
        self.assertEqual(res_stock_only["code"], "064400")
        self.assertTrue(res_stock_only["is_stock_only"])

        # 3. 6-digit stock code query for 064400 directly resolves
        res_code = extract_stock_and_question_from_query("064400 오늘 전망해줘", current_code="005930")
        self.assertEqual(res_code["code"], "064400")
        self.assertEqual(res_code["matched_by"], "code")

    def test_lg_cns_quant_answer_content(self):
        lg_cns_payload = {
            "name": "LG씨엔에스",
            "code": "064400",
            "market": "KOSPI",
            "sector": "소프트웨어/IT",
            "close": 72700.0,
            "change_pct": 0.55,
            "score": 71.2,
            "label": "우수",
            "strategy_tag": "고확신 퀀트 모멘텀",
            "tp_pct": 2.0,
            "sl_pct": 2.0,
            "vol_ratio": 1.85,
            "rsi": 55.7,
            "ma_status": "단기 정배열",
            "reason_core": "IT 서비스 및 클라우드 DX 모멘텀",
            "rule_note": "목표가 분할 익절 및 손절가 원칙 대응",
            "breakdown": {"rsi_val": 55.7, "vol_ratio": 1.85, "ma_status": "단기 정배열"},
            "indicators": {"ma5": 69500, "ma20": 68910, "ma60": 70600},
        }

        ans = generate_quant_expert_answer(lg_cns_payload, "lg씨앤에스 오늘 전망")
        self.assertIn("LG씨엔에스", ans)
        self.assertIn("064400", ans)
        self.assertNotIn("삼성전자", ans)
        self.assertNotIn("비상장", ans)

    def test_unrecognized_stock_guardrail(self):
        # 1. Unrecognized companies must be classified as 'unrecognized' without defaulting to Samsung Electronics
        res_coupang = extract_stock_and_question_from_query("쿠팡 오늘 전망해줘", current_code="005930")
        self.assertEqual(res_coupang["matched_by"], "unrecognized")
        self.assertEqual(res_coupang["unrecognized_name"], "쿠팡")
        self.assertEqual(res_coupang["codes"], [])

        res_apple = extract_stock_and_question_from_query("애플 주가 어때?", current_code="005930")
        self.assertEqual(res_apple["matched_by"], "unrecognized")
        self.assertEqual(res_apple["unrecognized_name"], "애플")

        # 2. General follow-up questions must NOT be misclassified as unrecognized
        res_buy = extract_stock_and_question_from_query("지금 사도 되나요?", current_code="005930")
        self.assertEqual(res_buy["matched_by"], "context")
        self.assertEqual(res_buy["code"], "005930")

        res_tp = extract_stock_and_question_from_query("목표가와 손절가는 얼마인가요?", current_code="005930")
        self.assertEqual(res_tp["matched_by"], "context")
        self.assertEqual(res_tp["code"], "005930")


if __name__ == "__main__":
    unittest.main()




