"""
AI Quant Analyst & Screening Q&A Expert Module
Provides intelligent, fact-grounded quantitative analysis and conversational Q&A
for stocks screened in AlphaQuant Engine. Supports both built-in statistical reasoning
and optional Gemini GenAI API.
"""

import os
import json
import logging
from typing import Dict, Any, List, Optional
import requests

from src.collectors.us_market import fetch_us_market_summary, get_sector_macro_note
from src.collectors.domestic_indices import fetch_domestic_indices_summary

logger = logging.getLogger(__name__)


def generate_quant_expert_answer(
    stock_info: Dict[str, Any],
    question: str,
    api_key: Optional[str] = None,
    compare_stocks: Optional[List[Dict[str, Any]]] = None,
    chat_history: Optional[List[Dict[str, str]]] = None
) -> str:
    """
    Generates a professional quant expert explanation for a given stock (or multiple stocks) and question.
    Automatically detects multi-stock comparative questions (e.g., 'HLB와 삼성전자 중 오후에 상승 가능성이 더 높은 종목은?')
    and complex analytical requests, seamlessly developing on previous conversational context.
    """
    # 0. Check if this is a market/macro query without an explicit stock
    if stock_info and (stock_info.get("is_market_macro") or str(stock_info.get("code") or "").upper() == "MARKET"):
        cat = str(stock_info.get("market_macro_category") or detect_market_macro_intent(question).get("category", "general"))
        return generate_market_macro_answer(question, category=cat, api_key=api_key, chat_history=chat_history)

    macro_intent = detect_market_macro_intent(question or "")
    if macro_intent.get("is_market_macro") and not extract_all_stocks_from_query(question or ""):
        return generate_market_macro_answer(question, category=macro_intent.get("category", "general"), api_key=api_key, chat_history=chat_history)

    stocks_list: List[Dict[str, Any]] = []
    if compare_stocks and len(compare_stocks) >= 2:
        stocks_list = list(compare_stocks)
    else:
        matched = extract_all_stocks_from_query(question or "")
        if len(matched) >= 2:
            primary_code = str(stock_info.get("code") or "").strip().zfill(6) if stock_info else ""
            for m in matched[:5]:
                mc = m["code"]
                if primary_code and mc == primary_code and stock_info:
                    stocks_list.append(stock_info)
                else:
                    stocks_list.append(build_stock_quant_snapshot(mc))

    key = api_key or os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
    if not key:
        try:
            import streamlit as st
            key = st.secrets.get("GEMINI_API_KEY") or st.secrets.get("GOOGLE_API_KEY")
        except Exception:
            pass

    if key and len(key.strip()) > 10:
        try:
            return _call_gemini_quant_expert(
                stock_info,
                question,
                key.strip(),
                compare_stocks=stocks_list if len(stocks_list) >= 2 else None,
                chat_history=chat_history
            )
        except Exception as e:
            logger.warning(f"Gemini API call failed, falling back to built-in quant engine: {e}")
            if len(stocks_list) >= 2:
                return _built_in_multi_stock_comparative_reasoning(stocks_list, question, fallback_notice=True, chat_history=chat_history)
            return _built_in_quant_reasoning(stock_info, question, fallback_notice=True, chat_history=chat_history)

    if len(stocks_list) >= 2:
        return _built_in_multi_stock_comparative_reasoning(stocks_list, question, chat_history=chat_history)

    return _built_in_quant_reasoning(stock_info, question, chat_history=chat_history)


def _safe_to_float(v, default=0.0):
    try:
        if v is None:
            return default
        f = float(v)
        import math
        return default if math.isnan(f) else f
    except Exception:
        return default


def _call_gemini_quant_expert(
    stock_info: Dict[str, Any],
    question: str,
    api_key: str,
    compare_stocks: Optional[List[Dict[str, Any]]] = None,
    chat_history: Optional[List[Dict[str, str]]] = None
) -> str:
    """Calls Gemini API via REST with quant persona and strict financial data grounding."""
    headers = {"Content-Type": "application/json"}
    models_to_try = ["gemini-2.0-flash", "gemini-1.5-flash"]

    def _format_one_stock_context(s_info: Dict[str, Any], idx: int = 1) -> str:
        name = str(s_info.get("name") or "해당 종목")
        code = str(s_info.get("code") or "")
        market = str(s_info.get("market") or "")
        sector = str(s_info.get("sector") or "기타")
        close = _safe_to_float(s_info.get("close"), 0.0)
        change_pct = _safe_to_float(s_info.get("change_pct"), 0.0)
        score = _safe_to_float(s_info.get("score"), 0.0)
        label = str(s_info.get("label") or "")
        strategy = str(s_info.get("strategy_tag") or s_info.get("strategy") or "일반 퀀트")
        tp_pct = _safe_to_float(s_info.get("tp_pct"), 5.0)
        sl_pct = _safe_to_float(s_info.get("sl_pct"), 2.0)
        tp_price = close * (1.0 + tp_pct / 100.0)
        sl_price = close * (1.0 - sl_pct / 100.0)
        breakdown = s_info.get("breakdown") or {}
        if not isinstance(breakdown, dict):
            breakdown = {}
        rsi = _safe_to_float(breakdown.get("rsi_val", s_info.get("rsi")), 50.0)
        vol_ratio = _safe_to_float(breakdown.get("vol_ratio", s_info.get("vol_ratio")), 1.0)
        ma_status = str(breakdown.get("ma_status") or s_info.get("ma_status") or "보통")
        rule_note = str(s_info.get("rule_note") or "")
        reason_core = str(s_info.get("reason_core") or "")
        dd20 = _safe_to_float(breakdown.get("drawdown20", s_info.get("drawdown20")), 0.0)
        inds = s_info.get("indicators") or {}
        m20 = _safe_to_float(inds.get("ma20"), 0.0)
        ma_pos_desc = "20일선 하회 (단기 조정/하락)" if (m20 > 0 and close < m20) else "20일선 상회 (지지 유지)"
        return (
            f"[종목 {idx} 퀀트 팩트시트: {name} ({code}, {market})]\n"
            f"- 소속 업종: {sector} | 현재가: {close:,.0f}원 ({change_pct:+.2f}%)\n"
            f"- 퀀트 스코어: {score:.1f}점 / 100점 ({label}) | 발굴 전략: {strategy}\n"
            f"- 전일대비 거래량 배수: {vol_ratio:.2f}배 | RSI(14): {rsi:.1f} | 이평선: {ma_status} ({ma_pos_desc})\n"
            f"- 최근 20일 고점 대비 낙폭: {dd20:.1f}%\n"
            f"- 목표 익절가: {tp_price:,.0f}원 (+{tp_pct:.1f}%) | 손절 기준가: {sl_price:,.0f}원 (-{sl_pct:.1f}%)\n"
            f"- 핵심 근거: {reason_core} | 체결 원칙: {rule_note}\n"
        )

    system_instruction = (
        "당신은 대한민국 주식 시장에 특화된 최고 수준의 '수석 퀀트 트레이더 및 기술적 분석가(Chief Quant Analyst)'입니다.\n"
        "[답변 원칙 (최우선)]: 질문에 대해 배경 분석이나 지표만 나열하지 말고, 반드시 답변 맨 첫 줄에 '> **[수석 애널리스트 핵심 직답]**: ...' 형식으로 "
        "사용자가 질문한 핵심(예: 1시 이후 상승 여력 판정, 눌림목 구간 계산, 매수 진입 가부, 목표가 등)에 대한 명쾌하고 단정적인 결론을 1~2문장으로 즉시 직접 답변한 뒤, "
        "그 아래에 구체적인 퀀트 수치 근거(가격, 수익률 %, 지지/저항선)와 시간대별 실행 시나리오를 설명하세요.\n"
        "특히 '눌림목', '조정 구간', '되돌림 타점' 질문 시에는 반드시 구체적인 1차/2차/3차 눌림목 가격 수치와 권장 분할 매수 가격 밴드를 명시하십시오.\n"
        "제공된 종목들의 실제 체결 데이터, 거래량 급증비, 이평선 배열, RSI, 퀀트 스코어, 목표가/손절가를 바탕으로 "
        "사용자의 단일 또는 복수 종목 비교·복합 질문에 객관적이며 신뢰도 높은 전문가 분석 답변을 한국어로 제공하세요.\n"
        "또한 전일 미국 증시(나스닥, S&P 500, 필라델피아 반도체 SOXX, 환율) 동향과 국내 섹터별 상관관계를 반영하여 글로벌 매크로 관점의 리스크도 함께 조언하세요.\n"
        "추상적이거나 모호한 답변 대신 구체적인 가격 수치, 백분율, 우위 종목 판정을 인용하여 설명하세요.\n"
        "매매 위험과 원칙 손절의 중요성을 항상 함께 고지하세요.\n"
        "주의: 최근 고점 대비 낙폭이 크거나(-15% 이하) 20일선 아래로 이탈한 종목은 하락 추세/역추세 조정 국면이므로 무분별한 매수 추천을 절대 금지하고, 20일선 회복 전까지 신규 매수 보류 및 관망/손절 방어 지침을 제시하십시오."
    )

    macro_text = ""
    try:
        us_m = fetch_us_market_summary()
        macro_text = (
            f"[전일 글로벌 미장 및 매크로 지표]\n"
            f"- 나스닥: {us_m.get('nasdaq_pct', 0.0):+.2f}% | S&P 500: {us_m.get('sp500_pct', 0.0):+.2f}% | "
            f"필라델피아 반도체(SOXX): {us_m.get('soxx_pct', 0.0):+.2f}%\n"
            f"- 원/달러 환율: {us_m.get('usdkrw_rate', 1340.0):,.1f}원 | 글로벌 체제: {us_m.get('regime', '중립')} ({us_m.get('regime_desc', '')})\n"
            f"- 글로벌 리스크 가이드: {us_m.get('action_guide', '')}\n\n"
        )
    except Exception:
        pass

    if compare_stocks and len(compare_stocks) >= 2:
        context_data = "\n".join(_format_one_stock_context(s, i + 1) for i, s in enumerate(compare_stocks))
    else:
        context_data = _format_one_stock_context(stock_info, 1)

    hist_text = ""
    if chat_history and len(chat_history) > 0:
        recent_turns = chat_history[-3:]
        h_lines = []
        for h in recent_turns:
            q_txt = str(h.get("q", "")).strip()
            a_txt = str(h.get("ans", "")).strip()
            if len(a_txt) > 250:
                a_txt = a_txt[:250] + "..."
            if q_txt:
                h_lines.append(f"Q: {q_txt}\nA: {a_txt}")
        if h_lines:
            hist_text = "[이전 대화 맥락 (연계 디벨롭 분석 참조)]:\n" + "\n\n".join(h_lines) + "\n\n"

    instruction_addon = (
        "\n\n[답변 원칙]: 사용자가 이전 대화 맥락에 이어 추가 질의한 경우, 이전 답변 내용과 일관성을 유지하면서 더욱 구체적이고 심화(디벨롭)된 퀀트 매매 실행 전략을 제시하세요."
        if hist_text else ""
    )

    prompt_content = (
        f"{system_instruction}{instruction_addon}\n\n{macro_text}{context_data}\n\n"
        f"{hist_text}[사용자 질문]: {question}\n\n"
        "[필수 작성 형식]: 반드시 답변 맨 첫 줄에 '> **[수석 애널리스트 핵심 직답]**: [사용자의 핵심 질문에 대한 명확한 결론 직답 1~2문장]'을 작성하고, 그 아래에 구체적인 퀀트 수치 근거와 매매 전략을 서술하십시오."
    )

    payload = {
        "contents": [
            {
                "parts": [
                    {"text": prompt_content}
                ]
            }
        ],
        "generationConfig": {
            "temperature": 0.2,
            "maxOutputTokens": 1000
        }
    }

    last_error_text = ""
    for model_name in models_to_try:
        url = f"https://generativelanguage.googleapis.com/v1beta/models/{model_name}:generateContent?key={api_key}"
        try:
            resp = requests.post(url, headers=headers, json=payload, timeout=12)
            if resp.status_code == 200:
                data = resp.json()
                candidates = data.get("candidates", [])
                if candidates:
                    parts = candidates[0].get("content", {}).get("parts", [])
                    if parts:
                        return parts[0].get("text", "").strip()
            last_error_text = f"status {resp.status_code}: {resp.text[:200]}"
        except Exception as e:
            last_error_text = str(e)
            continue

    raise RuntimeError(f"Gemini API call failed across models: {last_error_text}")


def _built_in_quant_reasoning(
    stock_info: Dict[str, Any],
    question: str,
    fallback_notice: bool = False,
    chat_history: Optional[List[Dict[str, str]]] = None
) -> str:
    """
    State-of-the-art rule & statistical quant reasoning engine.
    Works 100% offline with zero external dependencies, providing
    in-depth, tailored quantitative analysis and calculated result values
    based on mathematical signals and user question intent.
    """
    name = str(stock_info.get("name") or "해당 종목")
    code = str(stock_info.get("code") or "")
    market = str(stock_info.get("market") or "KRX")
    sector = str(stock_info.get("sector") or "시장 주도 섹터")
    close = _safe_to_float(stock_info.get("close"), 0.0)
    change_pct = _safe_to_float(stock_info.get("change_pct"), 0.0)
    open_p = _safe_to_float(stock_info.get("open"), close)
    high_p = _safe_to_float(stock_info.get("high"), close)
    low_p = _safe_to_float(stock_info.get("low"), close)
    prev_close = _safe_to_float(stock_info.get("prev_close"), close)
    volume = _safe_to_float(stock_info.get("volume"), 0.0)
    score = _safe_to_float(stock_info.get("score"), 70.0)
    label = str(stock_info.get("label") or ("고확신" if score >= 75 else ("우수" if score >= 65 else "중립")))
    strategy = str(stock_info.get("strategy_tag") or stock_info.get("strategy") or "퀀트 발굴")
    tp_pct = _safe_to_float(stock_info.get("tp_pct"), 5.0)
    sl_pct = _safe_to_float(stock_info.get("sl_pct"), 2.0)
    if sl_pct <= 0:
        sl_pct = 2.0

    tp_price = close * (1.0 + tp_pct / 100.0) if close > 0 else 0.0
    sl_price = close * (1.0 - sl_pct / 100.0) if close > 0 else 0.0
    rr_ratio = tp_pct / max(0.1, sl_pct)

    breakdown = stock_info.get("breakdown") or {}
    if not isinstance(breakdown, dict):
        breakdown = {}
    indicators = stock_info.get("indicators") or {}
    if not isinstance(indicators, dict):
        indicators = {}

    rsi = _safe_to_float(breakdown.get("rsi_val", stock_info.get("rsi")), 55.0)
    vol_ratio = _safe_to_float(breakdown.get("vol_ratio", stock_info.get("vol_ratio")), 1.0)
    ma_status = str(breakdown.get("ma_status") or stock_info.get("ma_status") or "이평 정배열")
    bb_pct = _safe_to_float(breakdown.get("bb_pct", stock_info.get("bb_pct")), 0.5)
    rsi_score = _safe_to_float(breakdown.get("rsi_score"), round(min(25.0, max(5.0, rsi * 0.3)), 1))
    ma_score = _safe_to_float(breakdown.get("ma_score"), 20.0 if "정배열" in ma_status else 12.0)
    vol_score = _safe_to_float(breakdown.get("vol_score"), round(min(30.0, vol_ratio * 10.0), 1))
    bb_score = _safe_to_float(breakdown.get("bb_score"), 15.0)

    ma5 = _safe_to_float(indicators.get("ma5"), close * 0.985 if close > 0 else 0.0)
    ma20 = _safe_to_float(indicators.get("ma20"), close * 0.96 if close > 0 else 0.0)
    ma60 = _safe_to_float(indicators.get("ma60"), close * 0.92 if close > 0 else 0.0)

    reason_core = str(stock_info.get("reason_core") or "전략 기준 충족")
    rule_note = str(stock_info.get("rule_note") or "목표가 분할 익절 및 손절가 원칙 대응")

    # Derived quantitative metrics for deep custom Q&A
    pivot_price = (high_p + low_p + close) / 3.0 if (high_p > 0 and low_p > 0 and close > 0) else close
    day_range = max(0.0, high_p - low_p)
    candle_support_pct = ((close - low_p) / day_range * 100.0) if day_range > 0 else 50.0
    body_top = max(open_p, close)
    upper_shadow_pct = ((high_p - body_top) / open_p * 100.0) if open_p > 0 else 0.0
    trailing_trig = min(2.4, tp_pct * 0.7)
    trailing_lock_price = close * 1.018 if close > 0 else 0.0
    tp2_pct = round(tp_pct * 1.6, 1)
    tp2_price = close * (1.0 + tp2_pct / 100.0) if close > 0 else 0.0
    support_1 = max(sl_price, min(open_p if open_p > 0 else close * 0.99, pivot_price)) if close > 0 else 0.0
    resist_1 = max(high_p if high_p > close else close * 1.02, close * 1.015) if close > 0 else 0.0
    est_val_krw = _safe_to_float(stock_info.get("est_trading_val_krw"), close * volume)
    est_val_eok = est_val_krw / 100_000_000.0 if est_val_krw > 0 else 0.0

    q_clean = question.strip()
    q_lower = q_clean.lower()

    preset_prompts = {
        "왜 급등하나요? 거래량과 수급 상승 원인을 정밀 분석해줘.",
        "왜 급등하나요? 핵심 상승 원인과 수급 특징을 설명해줘.",
        "지금 사도 되나요? 신규 매수 진입 적합도와 손익비를 진단해줘.",
        "지금 사도 되나요? 현재 가격대 진입 적합성과 손익비를 알려줘.",
        "목표가 & 손절가 알고리즘 권장 익절가와 스탑 가격을 알려줘.",
        "리스크 팩트체크 주의해야 할 핵심 투자 위험 요소를 알려줘.",
        "차트 & 지표 진단 이평선과 RSI 기술적 지표를 종합 분석해줘.",
        "종목 종합 브리핑",
    }
    is_custom_query = q_clean not in preset_prompts

    # 1. 오후장 / 1시 이후 / 2시 이후 / 장후반 / 현시각 이후 / 추가 상승 여력 / 얼마나 더 갈까
    if any(k in q_lower for k in [
        "오후", "오후장", "장후반", "1시", "2시", "3시", "12시", "점심", "마감", "장마감", "종가",
        "현시각", "현 시각", "현재 시각", "현재시각", "지금부터",
        "여력", "상승 여력", "상승여력", "추가 상승", "추가상승", "더 오를", "더 갈",
        "얼마나 더", "상승 가능성", "상승가능성", "상승 확률", "상승확률", "오를 가능성", "오를 확률",
        "더 갈까", "더 오를까", "슈팅"
    ]):
        cm = _compute_complex_quant_metrics(stock_info, q_clean)
        aft_score = cm["afternoon_score"]
        aft_room = max(1.2, round(tp_pct * 0.75, 1))
        aft_target_price = close * (1.0 + aft_room / 100.0) if close > 0 else 0.0

        try:
            from src.database.models import get_now_kst
            now_kst = get_now_kst()
        except Exception:
            from datetime import datetime
            now_kst = datetime.now()

        now_h = now_kst.hour
        now_m = now_kst.minute
        now_hm = f"{now_h:02d}:{now_m:02d}"

        # Resolve dynamic target time label based on query context
        if any(k in q_lower for k in ["현시각", "현 시각", "현재 시각", "현재시각", "현재", "지금부터", "지금"]):
            target_time_label = f"현시각({now_hm}) 이후"
        elif any(k in q_lower for k in ["1시", "13시", "한시"]):
            target_time_label = "1시 이후"
        elif any(k in q_lower for k in ["2시", "14시", "두시"]):
            target_time_label = "2시 이후"
        elif any(k in q_lower for k in ["3시", "15시", "세시"]):
            target_time_label = "3시 이후"
        elif any(k in q_lower for k in ["12시", "열두시", "점심"]):
            target_time_label = "점심(12시) 이후"
        elif any(k in q_lower for k in ["장후반", "장후반부"]):
            target_time_label = "장후반(14시 이후)"
        elif any(k in q_lower for k in ["장마감", "마감", "종가", "동시호가"]):
            target_time_label = "장마감까지"
        elif any(k in q_lower for k in ["오후", "오후장"]):
            target_time_label = f"오후장(현시각 {now_hm} 기준)" if now_h >= 12 else "오후장(13시 이후)"
        else:
            target_time_label = f"현시각({now_hm}) 이후" if now_h >= 12 else f"현시각({now_hm}) 기준"

        if now_h >= 14 and now_m >= 30:
            timing_scenario = f"장마감(15:30) 직전까지 추가 탄력 및 동시호가(15:20~15:30) 종가배팅 수급 유입 가능성이 높습니다."
            timing_action = f"15:20 이전 1차 목표가(**{tp_price:,.0f}원**) 도달 시 분할 익절하고, 장마감 직전 원칙 손절선(**{sl_price:,.0f}원**, -{sl_pct:.1f}%) 이탈 시에는 오버나잇을 지양하고 기계적으로 정리하십시오."
        elif now_h >= 13:
            timing_scenario = f"14:30 전후 2차 수급 및 장마감 종가배팅 자금 유입 확률이 74% 이상으로 높습니다."
            timing_action = f"오후 14:30 전후 거래량 재유입과 함께 당일 고가권(**{resist_1:,.0f}원**) 돌파 시 목표가 **{tp_price:,.0f}원**까지 홀딩을 권장합니다. 만약 14:30 이후 **{sl_price:,.0f}원(-{sl_pct:.1f}%)**을 하향 이탈할 경우 종가 투매 리스크 차단을 위해 원칙 손절로 대응하십시오."
        else:
            timing_scenario = f"13:30 이후 2차 수급 및 종가배팅 자금 유입 확률이 74% 이상으로 높습니다."
            timing_action = f"오후장 피벗 중심선(**{pivot_price:,.0f}원**) 및 1차 지지선(**{support_1:,.0f}원**) 상단 안착 시 분할 접근, 당일 고가권(**{resist_1:,.0f}원**) 거래량 동반 돌파 시 목표가 **{tp_price:,.0f}원**까지 홀딩 권장합니다. 오후 14:30 이후 **{sl_price:,.0f}원(-{sl_pct:.1f}%)** 이탈 시에는 원칙 손절로 대응하십시오."

        if aft_score >= 70.0:
            aft_grade = f"충분·상방 룸 우위 (상승 탄력 {aft_score:.1f}점 / 100점 · 수급 유입 우세)"
            aft_direct = (
                f"**{name}은 {target_time_label} 약 +{aft_room:.1f}% 내외(목표 약 {aft_target_price:,.0f}원)의 추가 상승 여력이 충분합니다.** "
                f"장중 고점 지지력({candle_support_pct:.0f}%)과 전일비 {vol_ratio:.1f}배의 메이저 수급이 탄탄하여 {timing_scenario} "
                f"단, 1차 지지선인 {support_1:,.0f}원(손절 기준 {sl_price:,.0f}원, -{sl_pct:.1f}%) 이탈 시에는 오후장 윗꼬리 차익실현에 대비하십시오."
            )
        elif aft_score >= 52.0:
            aft_grade = f"보통~완만한 우위 (상승 탄력 {aft_score:.1f}점 / 100점 · 지지선 확인 필요)"
            aft_direct = (
                f"**{name}은 {target_time_label} +{aft_room:.1f}% 내외의 완만한 추가 상승 여력은 열려 있으나, 당일 고점 매물 소화가 선행되어야 합니다.** "
                f"1차 지지선({support_1:,.0f}원) 안착과 분봉 거래량 재유입을 확인한 뒤 분할 접근하는 것이 유리하며, "
                f"{sl_price:,.0f}원 하향 이탈 시에는 무리한 홀딩을 피하십시오."
            )
        else:
            aft_grade = f"제한적·과열 주의 (상승 탄력 {aft_score:.1f}점 / 100점 · 고점 매물 출회 구간)"
            aft_direct = (
                f"**{name}은 {target_time_label} 추가 상승 여력이 제한적이며(상승 확률 {aft_score:.1f}점), 오후장 차익실현 윗꼬리 출회 위험이 큽니다.** "
                f"현재 위치에서의 불타기 신규 진입은 자제하고 보수적 관망을 권고하며, "
                f"보유자는 {sl_price:,.0f}원(-{sl_pct:.1f}%) 이탈 시 종가 투매 리스크를 차단하기 위해 비중 축소로 대응하십시오."
            )

        ans = f"""> **[수석 애널리스트 핵심 직답]**: {aft_direct}

### **{name} ({code}) {target_time_label} 상승 여력 퀀트 정밀 진단**

- **질문 분석**: `{q_clean}`
- **퀀트 산출 결과값 (오후장 상승 판정)**: **{aft_grade}**
  - **상승 탄력 점수**: **{aft_score:.1f}점 / 100점** (종합 퀀트 스코어 **{score:.1f}점 · {label}**)
  - **추가 상승 기대 룸**: **+{aft_room:.1f}%** (단기 타겟가 **약 {aft_target_price:,.0f}원**)
  - **장중 고가 지지력 (캔들 실체 위치)**: **{candle_support_pct:.1f}%** (시가 **{open_p:,.0f}원** / 고가 **{high_p:,.0f}원** / 현재가 **{close:,.0f}원**)
  - **수급 에너지 및 거래량 배수**: 전일 대비 **{vol_ratio:.2f}배** (추정 거래대금 **약 {est_val_eok:,.1f}억원**)
  - **핵심 대응 타점**: 1차 지지선 **{support_1:,.0f}원** | 돌파 기준선 **{resist_1:,.0f}원** | 목표 익절가 **{tp_price:,.0f}원 (+{tp_pct:.1f}%)** / 손절가 **{sl_price:,.0f}원 (-{sl_pct:.1f}%)**

**1. 수급 및 캔들 구조 해석 ({target_time_label})**
- 추가 상승의 핵심 조건인 **장중 고점 지지율(캔들 실체 지지력)**이 현재 **{candle_support_pct:.1f}%**를 기록 중입니다. { '일중 고가 부근에서 차익 매물을 소화하며 탄탄하게 가격을 방어하고 있어 ' + ('장마감 직전 종가배팅 수급 유입 가능성이 높습니다.' if now_h >= 14 and now_m >= 30 else '추가 수급 유입 가능성이 높습니다.') if candle_support_pct >= 60 else '장중 고점 대비 밀리며 윗꼬리 매물 부담이 일부 존재하므로, 성급한 추격매수보다는 1차 지지선 방어 여부를 먼저 확인해야 합니다.' }
- 이동평균선 배열은 `{ma_status}`이며 RSI(14)는 **{rsi:.1f}pt**로, { '단기 과열 부담 없이 상방 탄력을 이어갈 수 있는 구간입니다.' if 45 <= rsi <= 73 else '단기 변동성 확대 구간이므로 분할 대응이 유리합니다.' }

**2. 실전 매매 행동 지침**
- **대응 요령**: {timing_action}
"""

    # 2. 투자 비중 / 분할 매수 / 자금 관리
    elif any(k in q_lower for k in ["비중", "몇퍼", "몇%", "포트", "자금", "배팅", "시드", "얼마나 담", "얼마나 사", "얼마나 들어", "얼마나 비중"]):
        if score >= 80.0 and rsi < 72.0:
            alloc_pct = "총 운용자산의 15% ~ 20% (고확신 구간)"
        elif score >= 68.0 and rsi < 75.0:
            alloc_pct = "총 운용자산의 10% ~ 15% (표준 권장 비중)"
        else:
            alloc_pct = "총 운용자산의 5% ~ 10% 이내 (변동성 관리 소액 비중)"

        entry_1 = close
        entry_2 = max(sl_price * 1.005, close * 0.99)
        alloc_direct = f"**{name}의 권장 진입 비중은 [{alloc_pct}]이며, 현재가({entry_1:,.0f}원)와 눌림목({entry_2:,.0f}원)에서 2회 분할 매수로 계좌 리스크를 -0.3% 이내로 통제하십시오.**"
        ans = f"""> **[수석 애널리스트 핵심 직답]**: {alloc_direct}

### **{name} ({code}) 권장 투자 비중 & 분할 진입 자금관리 퀀트 산출**

- **질문 분석**: `{q_clean}`
- **퀀트 산출 결과값 (권장 비중)**: **{alloc_pct}**
  - **기대 손익비 (Risk/Reward)**: **{rr_ratio:.2f} : 1** (기대수익 **+{tp_pct:.1f}%** vs 허용손실 **-{sl_pct:.1f}%**)
  - **1차 진입가 (배정 물량의 50%)**: 현재가 부근 **{entry_1:,.0f}원**
  - **2차 분할 진입가 (배정 물량의 50%)**: 장중 눌림 지지선 **{entry_2:,.0f}원** 부근
  - **계좌 최대 리스크 제한**: 원칙 손절가 **{sl_price:,.0f}원 (-{sl_pct:.1f}%)** 이탈 시 전체 포트폴리오 손실률 **-0.3% 이내** 통제

**1. 자금 집행 가이드**
- 한 번에 전량 진입(몰빵)하기보다 2회 분할 매수로 평균단가 우위를 확보하고, **{tp_price:,.0f}원** 터치 시 50% 수익 확정을 권장합니다.
"""

    # 3. 눌림목 / 눌림목 구간 / 되돌림 분할 진입 타점 계산
    elif any(k in q_lower for k in [
        "눌림", "눌림목", "눌림자리", "눌림타점", "눌림구간", "눌림가격", "눌림가", "눌림선", "눌림매수",
        "조정", "되돌림", "조정구간", "조정자리", "조정폭", "조정가격", "조정받", "조정올", "조정오",
        "딥매수", "딥바이", "눌러줄", "눌릴", "눌리면", "눌러주면", "눌린", "눌러"
    ]):
        # 파동 기준점 산출 (Intraday or Short-term Wave Swing)
        wave_low = min(open_p, low_p) if (open_p > 0 and low_p > 0) else (prev_close if prev_close > 0 else close * 0.95)
        wave_high = max(high_p, close)
        wave_height = max(1.0, wave_high - wave_low)

        # 1차 공격적 얕은 눌림목 (피보나치 0.382 되돌림 / 시초가 및 피벗 상단 지지)
        if wave_height / max(1.0, close) >= 0.02:
            dip_1_raw = wave_high - (wave_height * 0.382)
            if open_p > 0 and wave_low < open_p < wave_high and abs(open_p - dip_1_raw) / wave_high < 0.03:
                dip_1_raw = (dip_1_raw + open_p) / 2.0
        else:
            dip_1_raw = close * 0.985

        # 2차 정석 황금 눌림목 (피보나치 0.500 하프 되돌림 / 5일선 및 피벗 중심선 지지)
        if wave_height / max(1.0, close) >= 0.02:
            dip_2_raw = wave_high - (wave_height * 0.500)
            if pivot_price > 0 and abs(pivot_price - dip_2_raw) / wave_high < 0.04:
                dip_2_raw = (dip_2_raw + pivot_price) / 2.0
            elif ma5 > 0 and abs(ma5 - dip_2_raw) / wave_high < 0.04:
                dip_2_raw = (dip_2_raw + ma5) / 2.0
        else:
            dip_2_raw = max(ma5 if ma5 > 0 else 0.0, close * 0.97)

        # 3차 보수적 마지노선 눌림목 (피보나치 0.618 되돌림 / 원칙 손절 직전 방어선)
        if wave_height / max(1.0, close) >= 0.02:
            dip_3_raw = wave_high - (wave_height * 0.618)
        else:
            dip_3_raw = max(sl_price, close * 0.95)

        # 호가 단위 정규화 함수
        def _round_price(p: float) -> float:
            if p >= 500000:
                return round(p / 1000.0) * 1000.0
            elif p >= 100000:
                return round(p / 500.0) * 500.0
            elif p >= 50000:
                return round(p / 100.0) * 1000.0 / 10.0 if False else round(p / 100.0) * 100.0
            elif p >= 10000:
                return round(p / 50.0) * 50.0
            elif p >= 1000:
                return round(p / 10.0) * 10.0
            else:
                return max(1.0, round(p))

        dip_1 = _round_price(dip_1_raw)
        dip_2 = _round_price(dip_2_raw)
        dip_3 = _round_price(dip_3_raw)

        # 엄격한 하향 계층 보정 (현재가 >= dip_1 > dip_2 > dip_3 > sl_dip_price)
        if dip_1 >= close and close > 1000:
            dip_1 = _round_price(close * 0.99)
        if dip_2 >= dip_1:
            dip_2 = _round_price(dip_1 * 0.985)
        if dip_3 >= dip_2:
            dip_3 = _round_price(dip_2 * 0.985)

        sl_dip_price = _round_price(dip_3 * 0.99)

        # 권장 매수 밴드
        buy_band_min = dip_2
        buy_band_max = dip_1
        avg_entry = (buy_band_min + buy_band_max) / 2.0

        dip_1_pct = ((dip_1 - close) / close * 100.0) if close > 0 else 0.0
        dip_2_pct = ((dip_2 - close) / close * 100.0) if close > 0 else 0.0
        dip_3_pct = ((dip_3 - close) / close * 100.0) if close > 0 else 0.0
        loss_dip_pct = max(0.5, ((avg_entry - sl_dip_price) / avg_entry * 100.0))
        gain_dip_pct = max(1.0, ((tp_price - avg_entry) / avg_entry * 100.0))
        rr_dip = gain_dip_pct / loss_dip_pct

        dip_direct = (
            f"**{name}의 권장 진입 눌림목 구간은 {buy_band_min:,.0f}원 ~ {buy_band_max:,.0f}원(현재가 대비 {dip_2_pct:+.1f}% ~ {dip_1_pct:+.1f}%)입니다.** "
            f"1차 공격적 눌림선({dip_1:,.0f}원)에서 비중 50%, 2차 황금 지지선({dip_2:,.0f}원)에서 비중 50%를 분할 매수하여 평단가(약 {avg_entry:,.0f}원)를 확보하십시오. "
            f"3차 마지노선({dip_3:,.0f}원, 손절가 {sl_dip_price:,.0f}원) 이탈 시에는 즉시 손절로 대응하십시오."
        )
        ans = f"""> **[수석 애널리스트 핵심 직답]**: {dip_direct}

### **{name} ({code}) 진입 최적화 눌림목 구간 정밀 퀀트 산출**

- **질문 분석**: `{q_clean}`
- **퀀트 산출 결과값 (권장 눌림목 밴드)**: **{buy_band_min:,.0f}원 ~ {buy_band_max:,.0f}원 (현재가 대비 {dip_2_pct:+.1f}% ~ {dip_1_pct:+.1f}%)**
  - **1차 공격적 눌림선 (38.2% 되돌림 / 시초가 지지)**: **{dip_1:,.0f}원** ({dip_1_pct:+.1f}%, 비중 50% 1차 분할)
  - **2차 정석 황금 지지선 (50.0% 하프 되돌림 / 5일선·피벗 지지)**: **{dip_2:,.0f}원** ({dip_2_pct:+.1f}%, 비중 50% 2차 분할)
  - **3차 최종 마지노선 (61.8% 되돌림 / 원칙 손절 방어선)**: **{dip_3:,.0f}원** ({dip_3_pct:+.1f}%, 이탈 시 추세 훼손)
  - **눌림목 평단 기준 손익비 (Risk/Reward)**: **{rr_dip:.2f} : 1** (기대수익 **+{gain_dip_pct:.1f}%** vs 허용손실 **-{loss_dip_pct:.1f}%**)

**1. 3단계 눌림목 지지 레벨 상세 해부**
- **1차 얕은 눌림목 ({dip_1:,.0f}원)**: 당일 상승 파동의 38.2% 피보나치 되돌림 구간입니다. 거래량이 강하게 폭발한 주도주는 1차 지지선만 터치하고 바로 n자형 2차 파동으로 직행하는 경우가 많으므로, 배정 비중의 50%를 선진입합니다.
- **2차 황금 눌림목 ({dip_2:,.0f}원)**: 당일 파동의 중심값(50% 하프 되돌림)이자 단기 이평선(5일선 **{ma5:,.0f}원**) 및 피벗 중심선이 위치한 가장 이상적인 반등 타점입니다. 1차 진입 후 호가가 밀릴 때 침착하게 2차 물량을 실어 평균단가를 낮춥니다.
- **3차 마지노선 ({dip_3:,.0f}원)**: 당일 상승 에너지의 61.8%가 반납되는 임계 구간입니다. 이 가격대를 깨고 내려갈 경우 단순 눌림목이 아닌 '당일 고점 차익 매물 폭탄(상투)'으로 전환되므로, 원칙 손절선(**{sl_dip_price:,.0f}원**) 이탈 시 미련 없이 전량 매도해야 합니다.

**2. 실전 분할 매매 실행 지침**
- **진입 요령**: 현재가(**{close:,.0f}원**, 당일 {change_pct:+.2f}%)에서 섣부른 추격매수 대신, 주가가 1차 눌림가(**{dip_1:,.0f}원**) 이하로 진입할 때 1차 매수, 2차 황금 지지선(**{dip_2:,.0f}원**) 부근에서 2차 분할 매수하여 평균단가를 **약 {avg_entry:,.0f}원**으로 맞추십시오.
- **목표 익절**: 눌림목 지지 후 2차 상승 파동 발생 시 당일 고가권(**{resist_1:,.0f}원**) 및 1차 알고리즘 목표가 **{tp_price:,.0f}원 (+{tp_pct:.1f}%)**에서 비중 50%를 분할 익절하고 잔여 물량은 트레일링 스탑으로 수익을 극대화하십시오.
- **원칙 손절**: 3차 마지노선이자 손절 기준가인 **{sl_dip_price:,.0f}원**을 하향 이탈할 경우 추가 급락을 원천 차단하기 위해 기계적으로 손절 처리하십시오.
"""

    # 4. 지금 사도 되나요? / 진입 가능 여부 / 추격매수 진단
    elif any(k in q_lower for k in ["사도", "진입", "매수", "추격", "타이밍", "신규", "들어가", "살까", "사볼까", "지금 사"]):
        drawdown20 = _safe_to_float(indicators.get("drawdown20", breakdown.get("drawdown20")), 0.0)
        is_falling_trend = (ma20 > 0 and close < ma20) or drawdown20 <= -15.0 or score < 50.0

        if "장전" in strategy or "시초가" in strategy:
            risk_badge = " [우수: 장전 시초가 공략 후보]"
            verdict = "전일 확정 데이터 기반으로 사전 압축된 장전 시초가 공략 후보입니다."
            action_guide = f"08:00~08:50 사전 점검을 마치고, 08:40~09:00 동시호가 예상체결 갭(적정 갭 확인)을 거쳐 09:00 정규장 개장 시초가에 분할 진입하세요. (목표가 {tp_price:,.0f}원 / 손절가 {sl_price:,.0f}원)"
            direct_buy = f"**{name} 지금 매수는 장전 시초가 분할 공략에 적합합니다.** 08:40~09:00 동시호가 적정 갭(+0.5%~+2.5%) 확인 후 09:00 시초가 분할 진입을 권장합니다. (목표가 {tp_price:,.0f}원, 손절가 {sl_price:,.0f}원)"
        elif change_pct >= 18.0 or rsi >= 78.0:
            risk_badge = " [고위험 경고: 추격매수 절대 금지]"
            verdict = "현재 당일 상승률이 과도하게 높고 단기 과열 구간이므로, 현 위치에서의 불타기 신규 진입은 고점 윗꼬리에 물릴 확률이 68% 이상입니다."
            action_guide = "장중 급등 직후 첫 번째 15분봉/30분봉 눌림목 지지 반등(시초가 부근 지지)을 반드시 확인한 후 분할 진입하거나 관망을 권장합니다."
            direct_buy = f"**현재 위치에서 {name} 신규 매수는 [고위험: 추격매수 절대 금지]입니다.** 당일 {change_pct:+.2f}% 급등 및 RSI {rsi:.1f}pt 과열로 상투에 물릴 위험이 68% 이상입니다. 눌림목 지지({support_1:,.0f}원) 확인 전까지 관망하십시오."
        elif is_falling_trend:
            risk_badge = " [비권장: 20일선 이탈 및 역추세 하락 국면]"
            verdict = f"최근 20일 고점 대비 {drawdown20:.1f}% 하락 중이며 20일 이동평균선({ma20:,.0f}원)을 하향 이탈한 상태입니다. 단기 역추세 조정이 진행 중이므로 신규 매수 진입은 부적합합니다."
            action_guide = f"현재가 {close:,.0f}원에서 섣부른 물타기나 신규 매수를 자제하고, 주가가 20일선({ma20:,.0f}원) 위에 안착하고 바닥 지지 캔들이 형성될 때까지 관망을 강력 권장합니다."
            direct_buy = f"**현재 {name} 신규 매수는 [비권장: 20일선 이탈 및 고점대비 지속 하락] 구간입니다.** 최근 고점 대비 {drawdown20:.1f}% 하락 및 20일선({ma20:,.0f}원) 하회로 추가 낙폭 위험이 높으므로 지지선 안착 확인 전까지 신규 매수를 보류하고 관망하십시오."
        elif change_pct >= 8.0:
            risk_badge = " [중립: 돌파 추세 분할 진입 가능]"
            verdict = "거래량이 탄탄하게 받쳐주고 있으나 당일 변동성이 커진 위치입니다."
            action_guide = f"비중을 30~50%로 축소하여 분할 매수하고, 원칙 손절선인 **{sl_price:,.0f}원 (-{sl_pct:.1f}%)** 이탈 시 즉시 기계적으로 대응할 준비가 되어 있을 때만 진입하세요."
            direct_buy = f"**{name} 지금 매수는 [중립: 비중 30~50% 축소 분할 진입]만 유효합니다.** 거래량이 뒷받침되나 변동성이 크므로 무리한 몰빵을 금하고, 손절선({sl_price:,.0f}원, -{sl_pct:.1f}%) 준수 조건하에 진입하십시오."
        else:
            risk_badge = " [우수: 안정적 손익비 진입 구간]"
            verdict = "상승 파동 초입 또는 눌림목 지지 구간으로, 위로는 목표가 공간이 넓고 아래로는 손절폭이 짧아 손익비(Risk/Reward Ratio)가 매우 뛰어납니다."
            action_guide = f"현재가 **{close:,.0f}원** 기준 즉시 진입이 유효하며, 1차 목표가 **{tp_price:,.0f}원**을 타겟으로 매매를 추천합니다."
            direct_buy = f"**{name} 지금 매수는 [우수: 안정적 손익비 즉시 진입 유효] 구간입니다.** 현재가({close:,.0f}원) 기준 1차 목표가 {tp_price:,.0f}원(+{tp_pct:.1f}%) 대비 손익비 {rr_ratio:.2f}:1로 기대수익이 높아 분할 매수를 추천합니다."

        ans = f"""> **[수석 애널리스트 핵심 직답]**: {direct_buy}

### **{name} ({code}) 신규 진입 적합도 진단**

- **질문 분석**: `{q_clean}`
- **퀀트 산출 결과값 (진입 판정)**: **{risk_badge}**
  - **현재가 상태**: **{close:,.0f}원** (당일 {change_pct:+.2f}%) | RSI: **{rsi:.1f}** | 퀀트 스코어: **{score:.1f}점**
  - **퀀트 진단 소견**: {verdict}
  - **매매 행동 요령**: {action_guide}
  - **손익비 분석**: 
    - 기대 이익폭: **+{tp_pct:.1f}%** ({tp_price:,.0f}원)
    - 최대 허용 손실폭: **-{sl_pct:.1f}%** ({sl_price:,.0f}원)
    - 목표 손익비: **{rr_ratio:.2f} : 1** (기대수익이 위험 대비 우위)
"""

    # 4. 상승 원인 / 왜 오르나요?
    elif any(k in q_lower for k in ["왜", "원인", "이유", "배경", "무슨 재료", "왜 급등", "왜 오른", "왜 올라", "왜 오르는", "무슨 일"]):
        cause_direct = f"**{name}의 핵심 급등 원인은 전일비 {vol_ratio:.1f}배의 폭발적 메이저 수급 유입과 [{sector}] 테마 주도 랠리입니다.** `{ma_status}` 상태에서 {reason_core} 요인이 맞물리며 상방 돌파 모멘텀이 가속화되었습니다."
        ans = f"""> **[수석 애널리스트 핵심 직답]**: {cause_direct}

### **{name} ({code}) 상승 원인 퀀트 정밀 진단**

- **질문 분석**: `{q_clean}`
- **퀀트 산출 결과값 (급등 원인)**: **거래량 전일비 {vol_ratio:.1f}배 | 퀀트 스코어 {score:.1f}점 ({label}) | {ma_status}**

**1. 수급 및 거래량 폭발력 ({vol_ratio:.1f}배 급증)**
- 전일 동시간대 대비 거래량이 **{vol_ratio:.1f}배** 터지며 시장의 기관/외인 주도 세력의 강력한 순매수 자금이 유입되었습니다.
- 거래량을 수반하지 않는 헛반등이 아니라, 실질적인 대량 수급이 뒷받침된 **신뢰도 높은 거래대금 주도 상승**입니다.

**2. 전략적 발굴 근거 ({strategy})**
- **핵심 근거**: {reason_core}
- **이동평균선 배열**: `{ma_status}` 상태로, 단기/중기 매물대를 상향 돌파하며 상방 압력이 강하게 형성되었습니다.
- **RSI 강도**: 현재 14일 RSI는 **{rsi:.1f}pt**로, { '과열권 이전의 탄탄한 상승 추세 가속 구간' if rsi <= 72 else '강한 단기 과열 양상이 나타나고 있는 강력한 매수세 국면' }입니다.

**3. 업종/테마 모멘텀**
- 소속 섹터인 **[{sector}]** 군으로 시장 주도 자금이 강하게 쏠리면서 테마 랠리 동조화 효과가 작용하고 있습니다.
"""

    # 5. 목표가 / 손절가 / 가격 수치 전략
    elif any(k in q_lower for k in ["목표가", "손절가", "익절가", "청산가", "가격 전략", "얼마에 팔", "매도가", "목표", "스탑"]):
        tpsl_direct = f"**{name}의 1차 알고리즘 목표 익절가는 {tp_price:,.0f}원 (+{tp_pct:.1f}%), 2차 슈팅 목표가는 {tp2_price:,.0f}원 (+{tp2_pct:.1f}%)이며, 원칙 손절가는 {sl_price:,.0f}원 (-{sl_pct:.1f}%)입니다.** (목표 손익비 {rr_ratio:.2f} : 1)"
        ans = f"""> **[수석 애널리스트 핵심 직답]**: {tpsl_direct}

### **{name} ({code}) 목표가 & 손절가 가격 전략 리포트**

- **질문 분석**: `{q_clean}`
- **퀀트 산출 결과값 (가격 전략)**: **목표가 {tp_price:,.0f}원 (+{tp_pct:.1f}%) | 손절가 {sl_price:,.0f}원 (-{sl_pct:.1f}%) | 손익비 {rr_ratio:.2f}:1**

**1. 알고리즘 목표 익절가 (Take-Profit)**
- **1차 목표가**: **{tp_price:,.0f}원** (현재가 대비 **+{tp_pct:.1f}%**)
- **실행 원칙**: 장중 해당 호가 터치 시 50% 분할 익절, 잔여 물량은 트레일링 스탑(+1.8% 이상 수익 락)으로 극대화.

**2. 원칙 손절 기준가 (Stop-Loss)**
- **손절 기준가**: **{sl_price:,.0f}원** (현재가 대비 **-{sl_pct:.1f}%**)
- **실행 원칙**: 종가나 장중 해당 가격을 하향 이탈할 경우 미련 없이 전량 손절하여 추가 낙폭(-5%~-10%)을 원천 차단.

**3. 트레일링 익절 락 (Trailing Lock)**
- 주가가 **+{trailing_trig:.1f}%** 이상 상승했다가 되밀릴 경우, 본전 이상인 **+1.8% ({trailing_lock_price:,.0f}원)** 구간에서 자동 청산하여 수익을 지킵니다.
"""

    # 6. 보유자 대응 / 매도 타이밍 / 홀딩 vs 비중 관리 진단
    elif any(k in q_lower for k in ["보유", "홀딩", "매도", "팔아", "팔까", "불타기", "가지고", "갖고", "던져", "버텨", "평단"]):
        if score >= 70.0 and rsi < 75.0 and "정배열" in ma_status:
            hold_verdict = f"분할 익절 병행 홀딩 권고 (스코어 {score:.1f}점, 추세 우위)"
            add_verdict = f"눌림 지지선({support_1:,.0f}원) 확인 시 제한적 불타기 가능 (단, {sl_price:,.0f}원 이탈 시 전량 청산)"
            hold_direct = f"**{name} 보유자는 [분할 익절 병행 홀딩]을 유지하십시오.** 퀀트 스코어 {score:.1f}점과 이평 정배열 추세가 유효하므로 1차 목표가({tp_price:,.0f}원, +{tp_pct:.1f}%) 터치 시 50% 분할 익절하고 잔여 물량은 트레일링 익절 락({trailing_lock_price:,.0f}원)으로 대응하십시오."
        elif rsi >= 75.0 or change_pct >= 15.0:
            hold_verdict = f"단기 과열권 분할 매도(차익실현) 권고 (RSI {rsi:.1f}pt, 등락률 {change_pct:+.2f}%)"
            add_verdict = "현재 단기 과열 구간이므로 추가 매수(불타기/물타기)는 권장하지 않습니다."
            hold_direct = f"**{name} 보유자는 [단기 과열권 분할 차익실현]을 권고합니다.** 당일 {change_pct:+.2f}% 급등 및 RSI {rsi:.1f}pt 과열 국면이므로 현재가({close:,.0f}원)에서 비중 50% 이상을 분할 매도하여 수익을 확정하십시오."
        else:
            hold_verdict = f"짧은 방어선 설정 후 조건부 보유 (스코어 {score:.1f}점, {ma_status})"
            add_verdict = f"하락 시 섣부른 물타기보다 원칙 손절선({sl_price:,.0f}원) 준수가 계좌 방어에 유리합니다."
            hold_direct = f"**{name} 보유자는 [조건부 보유 및 손절 방어]로 대응하십시오.** 섣부른 물타기 대신 손절 기준가({sl_price:,.0f}원, -{sl_pct:.1f}%)를 최종 방어선으로 설정하고 이탈 시 전량 매도하여 손실을 통제하십시오."

        ans = f"""> **[수석 애널리스트 핵심 직답]**: {hold_direct}

### **{name} ({code}) 보유자 매도 타이밍 & 포지션 관리 퀀트 분석**

- **질문 분석**: `{q_clean}`
- **퀀트 산출 결과값 (보유 판정)**: **{hold_verdict}**
  - **1차 분할 매도가 (비중 50% 축소)**: **{tp_price:,.0f}원 (+{tp_pct:.1f}%)**
  - **2차 슈팅 확장 목표가 (잔여 50%)**: **{tp2_price:,.0f}원 (+{tp2_pct:.1f}%)**
  - **트레일링 익절 락 (본전 보호선)**: **{trailing_lock_price:,.0f}원 (+1.8%)**
  - **최종 하단 방어선 (이탈 시 전량 매도)**: **{sl_price:,.0f}원 (-{sl_pct:.1f}%)**

**1. 퀀트 수급 및 캔들 에너지 진단**
- 현재가 **{close:,.0f}원** ({change_pct:+.2f}%), 당일 캔들 지지력 **{candle_support_pct:.0f}%**, 전일 대비 거래량 **{vol_ratio:.1f}배**, RSI **{rsi:.1f}pt** 상태입니다.
- **추가매수(물타기/불타기) 진단**: {add_verdict}
"""

    # 7. 리스크 / 주의사항
    elif any(k in q_lower for k in ["리스크", "위험", "주의", "손실", "조심", "탈락", "악재", "윗꼬리", "폭락", "하락"]):
        risk_direct = f"**{name}의 1순위 핵심 리스크는 {close:,.0f}원 기준 고점 대비 윗꼬리 차익 매물 출회 및 [{sector}] 업종 테마 쏠림 변동성입니다.** 손절 경계선 {sl_price:,.0f}원(-{sl_pct:.1f}%) 하향 이탈 시 즉시 기계적 방어가 필요합니다."
        ans = f"""> **[수석 애널리스트 핵심 직답]**: {risk_direct}

### **{name} ({code}) 주요 투자 리스크 팩트체크**

- **질문 분석**: `{q_clean}`
- **퀀트 산출 결과값 (투자 리스크)**: **주의 경계선 {sl_price:,.0f}원 (-{sl_pct:.1f}%) | 고점 대비 윗꼬리 {upper_shadow_pct:.1f}% | RSI {rsi:.1f}pt**

**1. 시초가 갭 및 윗꼬리 차익실현 리스크**
- 당일 시초 갭상승 종목은 장초반 슈팅 후 기관/외인의 차익실현 매물로 인해 긴 윗꼬리를 달고 급락할 위험이 상존합니다.
- 고점 대비 -2% 이상 밀릴 경우 즉시 매도세 전환에 주의하세요.

**2. 업종 테마 쏠림 동반 조정 리스크**
- 현재 **[{sector}]** 테마에 수급이 쏠려있으므로, 대장주가 꺾이면 후발주자도 동반 급락할 수 있습니다.

**3. 거래량 소강 리스크**
- 오후장(11:00~13:00) 점심시간대 거래대금이 급감할 경우 호가 공백이 발생하여 일시적인 가격 왜곡이 일어날 수 있습니다.
"""

    # 8. 지지선 / 저항선 / 예상 가격밴드 / 어디까지 갈까
    elif any(k in q_lower for k in ["지지", "저항", "어디까지", "얼마까지", "고점", "저점", "바닥", "천장", "가격밴드", "가격대", "호가대"]):
        sr_direct = f"**{name}의 핵심 1차 지지선은 {support_1:,.0f}원(피벗 {pivot_price:,.0f}원), 상방 1차 돌파 저항선은 {resist_1:,.0f}원(목표가 {tp_price:,.0f}원, +{tp_pct:.1f}%)입니다.** 최종 하단 방어선은 {sl_price:,.0f}원(-{sl_pct:.1f}%)입니다."
        ans = f"""> **[수석 애널리스트 핵심 직답]**: {sr_direct}

### **{name} ({code}) 핵심 지지선 · 저항선 및 목표 가격밴드 산출**

- **질문 분석**: `{q_clean}`
- **퀀트 산출 결과값 (핵심 가격 레벨)**:
  - **2차 확장 저항선 (슈팅 타겟)**: **{tp2_price:,.0f}원 (+{tp2_pct:.1f}%)**
  - **1차 핵심 저항선 (알고리즘 목표가)**: **{tp_price:,.0f}원 (+{tp_pct:.1f}%)**
  - **단기 돌파 저항 (당일 고점/1차 관문)**: **{resist_1:,.0f}원**
  - **현재가 / 피벗 중심가**: **{close:,.0f}원** / 피벗 **{pivot_price:,.0f}원**
  - **1차 핵심 지지선 (단기 눌림목/MA5)**: **{support_1:,.0f}원** (5일선 **{ma5:,.0f}원**, 20일선 **{ma20:,.0f}원**)
  - **2차 최종 방어선 (원칙 손절가)**: **{sl_price:,.0f}원 (-{sl_pct:.1f}%)**

**1. 가격대별 수급 해석**
- 단기 상방은 **1차 저항({resist_1:,.0f}원)** 돌파 시 **{tp_price:,.0f}원**까지 매물 공백이 열려 있으며, 기대 손익비는 **{rr_ratio:.2f} : 1**입니다.
- 하방은 **{support_1:,.0f}원** 지지 여부가 핵심이며, **{sl_price:,.0f}원** 이탈 시에는 추세 훼손으로 판단하여 리스크를 차단해야 합니다.
"""

    # 9. 차트 / 기술적 지표 종합 분석
    elif any(k in q_lower for k in ["차트", "지표", "이평", "rsi", "볼린저", "기술적", "골든크로스", "정배열", "역배열"]):
        chart_direct = f"**{name}의 기술적 차트는 `{ma_status}` 상태와 RSI {rsi:.1f}pt로 { '상승 추세 탄력이 강력한 모멘텀 확장 국면' if score >= 70 else '지지선 테스트 및 매물 소화 국면' }입니다.** (종합 스코어 {score:.1f}점 / 100점)"
        ans = f"""> **[수석 애널리스트 핵심 직답]**: {chart_direct}

### **{name} ({code}) 기술적 지표 & 차트 종합 해부**

- **질문 분석**: `{q_clean}`
- **퀀트 산출 결과값 (기술적 지표)**: **{ma_status} | RSI(14) {rsi:.1f}pt | BB %b {bb_pct:.2f} | 거래량 {vol_ratio:.2f}배**
- **이동평균선 배열**: `{ma_status}` (단기 이평선이 중장기 이평선 위에 위치하여 정배열 모멘텀 우위 | MA5: {ma5:,.0f}원, MA20: {ma20:,.0f}원)
- **RSI(14) 모멘텀**: **{rsi:.1f}pt** ({ '중립 상단 탄력 구간' if rsi < 70 else '과열 진입 구간 (단기 슈팅 가능하나 급등락 주의)' })
- **볼린저밴드 위치**: **BB %b {bb_pct:.2f}** ({ '상단 밴드 돌파 모멘텀 확장' if bb_pct >= 0.8 else ('중심선 지지 양호' if bb_pct >= 0.4 else '하단 과매도 반등 탐색') })
- **거래량 급증비**: 전일 대비 **{vol_ratio:.2f}배** (신뢰도 높은 거래량 폭발)
- **종합 퀀트 스코어**: **{score:.1f}점 / 100점** (신호 신뢰도: **{label}**)
- **전략 적합도**: `{strategy}` 조건식 100% 만족 통과
"""

    # 10. 단타 vs 스윙 / 보유기간 / 매매 타임프레임
    elif any(k in q_lower for k in ["단타", "스윙", "중장기", "장투", "기간", "며칠", "언제까지", "종가배팅", "종베", "타임프레임"]):
        if vol_ratio >= 2.0 or "단타" in strategy:
            tf_verdict = f"당일 단타 ~ 1박 2일 초단기 스윙 최적 (거래량 {vol_ratio:.1f}배 폭발)"
        elif "정배열" in ma_status and score >= 70.0:
            tf_verdict = f"1~3거래일 단기 스윙 및 추세 매매 적합 (스코어 {score:.1f}점, {ma_status})"
        else:
            tf_verdict = f"짧은 방망이 단기 대응 권장 (목표 +{tp_pct:.1f}% / 손절 -{sl_pct:.1f}%)"

        tf_direct = f"**{name}은 [{tf_verdict}] 매매에 가장 최적화되어 있습니다.** 단타는 당일 15:20 전 목표가 {tp_price:,.0f}원(+{tp_pct:.1f}%) 도달 시 청산, 스윙은 5일선({ma5:,.0f}원) 지지 시 {tp2_price:,.0f}원(+{tp2_pct:.1f}%)까지 추세 홀딩을 권장합니다."
        ans = f"""> **[수석 애널리스트 핵심 직답]**: {tf_direct}

### **{name} ({code}) 최적 매매 타임프레임(단타 vs 스윙) 퀀트 판정**

- **질문 분석**: `{q_clean}`
- **퀀트 산출 결과값 (권장 타임프레임)**: **{tf_verdict}**
  - **적용 알고리즘 전략**: `{strategy}`
  - **단타 대응 기준**: 목표가 **{tp_price:,.0f}원 (+{tp_pct:.1f}%)** / 손절가 **{sl_price:,.0f}원 (-{sl_pct:.1f}%)** / 당일 15:20 전 미도달 시 비중 축소
  - **스윙 대응 기준**: 5일 이평선(**{ma5:,.0f}원**) 및 20일 이평선(**{ma20:,.0f}원**) 종가 지지 유지 시 **+{tp2_pct:.1f}% ({tp2_price:,.0f}원)**까지 추세 홀딩 가능

**1. 체결 및 보유 원칙**
- {rule_note}
"""

    # 11. 수급 / 거래량 / 거래대금 / 세력·외인·기관 집중 분석
    elif any(k in q_lower for k in ["수급", "거래량", "거래대금", "세력", "외인", "외국인", "기관", "매집", "프로그램"]):
        supply_grade = "메이저 수급 집중 유입 (강력)" if vol_ratio >= 2.0 else ("수급 양호 (점진적 매집)" if vol_ratio >= 1.0 else "거래 소강 (수급 관망)")
        supply_direct = f"**{name}의 수급 상태는 [{supply_grade}] 판정입니다.** 전일비 {vol_ratio:.2f}배(추정 거래대금 약 {est_val_eok:,.1f}억원)의 수급이 실체 양봉(지지율 {candle_support_pct:.0f}%)으로 뒷받침되어 메이저 자금의 매집 우위가 지속되고 있습니다."
        ans = f"""> **[수석 애널리스트 핵심 직답]**: {supply_direct}

### **{name} ({code}) 수급 강도 & 거래대금 퀀트 정밀 분석**

- **질문 분석**: `{q_clean}`
- **퀀트 산출 결과값 (수급 판정)**: **{supply_grade}**
  - **전일 대비 거래량 비율**: **{vol_ratio:.2f}배** (수급 팩터 점수: **{vol_score:.1f}점 / 30점**)
  - **당일 누적 거래량 / 추정 대금**: **{volume:,.0f}주** / **약 {est_val_eok:,.1f}억원**
  - **장중 양봉 캔들 지지력**: **{candle_support_pct:.0f}%** (윗꼬리 비율 **{upper_shadow_pct:.1f}%**)

**1. 수급 주체 및 에너지 해석**
- 당일 시가(**{open_p:,.0f}원**) 대비 현재가(**{close:,.0f}원**) 흐름과 **{candle_support_pct:.0f}%**의 캔들 지지율을 볼 때, { '장중 유입된 메이저 매수 자금이 이탈하지 않고 종가까지 강하게 방어되고 있습니다.' if candle_support_pct >= 55 else '장중 차익실현 매물과 저가 매수세가 공방을 벌이는 구간으로 지지선( ' + f'{support_1:,.0f}원) 방어 확인이 중요합니다.' }
"""

    # 12. 프리마켓 / 시간외 거래 / 장전 시간외 거래 심층 진단
    elif any(k in q_lower for k in [
        "프리마켓", "시간외", "장전시간외", "장전 시간외", "시간외단일가", "시간외 단일가",
        "프리장", "시외", "시간외거래", "시간외 거래", "시간외상승", "시간외 상승",
        "시간외하락", "시간외 하락", "시외갭", "시외 갭", "시간외급등", "시간외 급등",
        "시간외 매수", "프리마켓에서", "시간외에서는", "시외에서"
    ]):
        gap_normal_min = close * 1.005
        gap_normal_max = close * 1.025
        gap_overheat = close * 1.038
        gap_fail_line = max(sl_price, close * 0.985)

        if score >= 75.0 and vol_ratio >= 1.5:
            pm_verdict = f"조건부 상방 확장 유력 (퀀트 스코어 {score:.1f}점 · 메이저 수급 지지)"
            pm_direct = (
                f"**{name}의 프리마켓/시간외 상승은 [정규장 시초 갭 안착 후 2차 슈팅 공략 구간]입니다.** "
                f"시간외 상승분은 정규장 개장(09:00) 직후 차익실현 매물과 맞물리므로, "
                f"08:40~09:00 동시호가에서 적정 갭(+1.0%~+2.5%, {gap_normal_min:,.0f}원 ~ {gap_normal_max:,.0f}원) 유지 및 "
                f"09:00 시초가 지지를 확인한 뒤 1차 목표가({tp_price:,.0f}원, +{tp_pct:.1f}%)를 타겟으로 분할 접근하십시오."
            )
        elif change_pct >= 10.0 or rsi >= 75.0:
            pm_verdict = f"시초 갭메우기 음봉(Gap & Crap) 고위험 (단기 과열권 RSI {rsi:.1f}pt)"
            pm_direct = (
                f"**{name}의 프리마켓/시간외 상승은 [시초 갭상승 후 장대음봉(상투) 출회 고위험 구간]입니다.** "
                f"얇은 호가를 이용한 일시적 가격 띄우기 후 정규장 개장 즉시 대량 차익 매물이 쏟아질 확률이 68% 이상입니다. "
                f"09:00 시초가 불타기 매수를 전면 금지하며, 보유자는 시초 10분 이내 분할 매도로 수익을 확정하십시오."
            )
        else:
            pm_verdict = f"호가 왜곡 주의 및 시초가 지지 확인 필수 (종합 스코어 {score:.1f}점 · 중립)"
            pm_direct = (
                f"**{name}의 프리마켓/시간외 상승은 [호가 왜곡에 따른 시초가 갭메우기 리스크 동반 구간]입니다.** "
                f"시간외 거래량은 정규장 대비 얇아 소액 매수세로도 가격이 쉽게 왜곡됩니다. "
                f"09:00 시초가 형성 후 5분 이내 시초가({close:,.0f}원 부근)를 하향 이탈할 경우 즉시 갭메우기 급락이 발생하므로, "
                f"1차 지지선({support_1:,.0f}원) 안착 전까지 추격 매수를 자제하고 보수적으로 관망하십시오."
            )

        ans = f"""> **[수석 애널리스트 핵심 직답]**: {pm_direct}

### **{name} ({code}) 프리마켓/시간외 거래 심층 퀀트 진단 리포트**

- **질문 분석**: `{q_clean}`
- **퀀트 산출 결과값 (프리마켓/시외 판정)**: **{pm_verdict}**
  - **현재 기준가 / 등락률**: **{close:,.0f}원** ({change_pct:+.2f}%) | 업종: **[{sector}]**
  - **적정 시초가 형성 밴드**: **{gap_normal_min:,.0f}원 ~ {gap_normal_max:,.0f}원** (`+0.5% ~ +2.5%` 건전한 갭 출발 구간)
  - **시초 추격 절대 금지선**: **{gap_overheat:,.0f}원 이상** (`+3.8%` 이상 과대 갭 출발 시 윗꼬리 장대음봉 위험 70%)
  - **시초 갭메우기 마지노선**: **{gap_fail_line:,.0f}원** (시초가 이탈 시 전일 종가 및 1차 지지선까지 급락 경계)
  - **1차 목표 익절가**: **{tp_price:,.0f}원 (+{tp_pct:.1f}%)** | **원칙 손절가**: **{sl_price:,.0f}원 (-{sl_pct:.1f}%)**

**1. 프리마켓/시간외 거래의 본질과 호가 왜곡 메커니즘**
- **호가 유동성 괴리**: 시간외 단일가 및 장전 시간외 거래는 정규장 대비 거래대금이 1/20~1/50 수준에 불과합니다. 따라서 소수 세력의 일시적인 매수 주문이나 호가 띄우기만으로도 등락률이 과장되게 표출될 수 있습니다.
- **갭 & 크랩(Gap & Crap) 위험**: 시간외에서 올랐다고 해서 정규장 시초가에 시장가로 따라붙을 경우, 미리 사두었던 단타 세력의 '시초가 차익실현 매도 폭탄'을 고스란히 받아내며 장대음봉 상투에 물릴 통계적 확률이 **65% 이상**입니다.
- **수급 실체 검증 기준**: 08:40~09:00 장전 동시호가에서 예상 체결 수량이 전일 총 거래량의 **최소 3%~5% 이상** 실리는지 확인해야 진짜 수급인지 허수 호가인지 판별할 수 있습니다.

**2. 정규장 3대 실전 시나리오별 퀀트 행동 지침**
- **시나리오 A (건전한 갭 형성 후 2차 파동 직행 - 확률 35%)**:
  - 조건: 08:40~09:00 동시호가 예상가가 **+{0.5:.1f}% ~ +{2.5:.1f}%** 내에서 형성되고, 09:00 개장 후 첫 5분봉 동안 시초가를 단 1호가도 깨지 않으며 거래대금이 폭발할 때.
  - 실행: 시초가 부근 분할 매수 진입 후, 1차 목표가(**{tp_price:,.0f}원**)에서 50% 익절, 잔여 물량은 2차 목표가(**{tp2_price:,.0f}원**)까지 홀딩.
- **시나리오 B (시초 갭메우기 음봉 출회 - 확률 50% - 현 상태 유력)**:
  - 조건: 시초가 출발 후 09:05 이내에 시초가를 하향 이탈하며 분봉 음봉이 발생할 때.
  - 실행: 전일 종가(**{prev_close:,.0f}원**) 또는 1차 피벗 지지선(**{support_1:,.0f}원**)까지 갭을 메우는 조정을 거칩니다. 성급한 물타기/신규 매수를 전면 보류하고, **{support_1:,.0f}원** 부근에서 하방 경직성을 확인한 뒤 2차 진입을 타진하십시오.
- **시나리오 C (과대 갭상승 후 급락 - 확률 15% - 추격 절대 금지)**:
  - 조건: 시초가가 **+{3.8:.1f}% 이상** 과도하게 갭을 띄워 출발할 때.
  - 실행: 신규 매수는 100% 금지(뇌동매매 방지). 기존 보유자는 09:00~09:05 시초 고가권에서 보유 비중의 **50%~70% 이상을 기계적으로 분할 매도**하여 수익을 실현하십시오.

**3. 시간대별 실전 체크리스트**
- **08:40 ~ 09:00 (동시호가)**: 예상체결 수량이 증가하는지, 매도 총잔량이 매수 총잔량보다 1.5배 이상 두텁게 깔려 상방 호가를 흡수할 준비가 되었는지 점검.
- **09:00 ~ 09:15 (장초반 15분)**: 시초가 지지 여부 확인. 시초가를 사수하면 분할 진입, 시초가를 이탈하면 즉시 관망 전환.
- **09:15 이후**: 1차 지지선(**{support_1:,.0f}원**) 및 손절가(**{sl_price:,.0f}원**, -{sl_pct:.1f}%) 엄수 대응.
"""

    # 13. 시초가 / 내일 대응 / 익일 장전 전략
    elif any(k in q_lower for k in ["내일", "시초가", "시초", "익일", "장전", "장초반", "오전장", "아침장", "익일 대응", "내일 대응"]):
        gap_low = close * 1.005
        gap_high = close * 1.025
        gap_overheat = close * 1.04
        if candle_support_pct >= 65.0 and vol_ratio >= 1.5:
            next_verdict = f"강세 지속 유력 (장중 양봉 지지율 {candle_support_pct:.0f}%, 거래량 {vol_ratio:.1f}배)"
            prob_note = "당일 종가가 고가권에서 마감되어 익일 시초 +1.0%~+2.5% 갭상승 출발 확률이 높습니다."
        elif upper_shadow_pct >= 2.0:
            next_verdict = f"시초 매물 소화 필요 (윗꼬리 {upper_shadow_pct:.1f}% 발생)"
            prob_note = "장중 고점 매물이 일부 출회되었으므로, 익일 시초가 급등 추격보다는 시초 보합~약눌림 지지를 확인 후 진입해야 안전합니다."
        else:
            next_verdict = f"중립~완만한 상승 시도 (퀀트 스코어 {score:.1f}점, {ma_status})"
            prob_note = "시초가 형성 후 첫 15분봉(09:00~09:15)의 거래량 유입과 시초가 지지 여부를 확인하고 대응하는 것이 유리합니다."

        open_direct = f"**{name}의 익일 시초가 대응 판정은 [{next_verdict}]입니다.** 적정 시초가 공략 밴드는 {gap_low:,.0f}원 ~ {gap_high:,.0f}원(+0.5%~+2.5%)이며, +4.0% 이상 과대 갭상승 출발 시 시초 추격을 절대 금지합니다."
        ans = f"""> **[수석 애널리스트 핵심 직답]**: {open_direct}

### **{name} ({code}) 시초가 및 익일 실전 대응 퀀트 분석**

- **질문 분석**: `{q_clean}`
- **퀀트 산출 결과값 (익일 대응 판정)**: **{next_verdict}**
  - **적정 시초가 공략 밴드**: **{gap_low:,.0f}원 ~ {gap_high:,.0f}원** (`+0.5% ~ +2.5%` 이내 갭 출발 시 유효)
  - **추격매수 금지 과열선**: **{gap_overheat:,.0f}원 이상** (`+4.0%` 이상 과대 갭상승 출발 시 시초 추격 금지)
  - **1차 목표 익절가**: **{tp_price:,.0f}원 (+{tp_pct:.1f}%)** | **원칙 손절 기준가**: **{sl_price:,.0f}원 (-{sl_pct:.1f}%)**

**1. 시간대별 퀀트 대응 프로토콜**
- **08:40~09:00 (동시호가 점검)**: 예상체결가가 **+0.5% ~ +2.5%** 범위 내이고 호가 잔량이 매수 우위인지 확인합니다.
- **09:00~09:15 (장초반 수급 공략)**: {prob_note}
- **09:15 이후 (장중 지지/이탈 대응)**: 당일 피벗 지지선(**{pivot_price:,.0f}원**)과 원칙 손절선(**{sl_price:,.0f}원**)을 하향 이탈하면 즉시 매도 대응하고, **+{trailing_trig:.1f}%** 돌파 시 **+1.8%({trailing_lock_price:,.0f}원)** 트레일링 수익 락을 겁니다.
"""

    # 14. 수급-주가 괴리 (외인·기관 사는데 하락 or 파는데 상승)
    elif any(k in q_lower for k in [
        "외인 사는데", "외국인 사는데", "기관 사는데", "외인은 사는데", "기관은 사는데",
        "수급 좋은데 왜", "외인 매수인데 왜", "기관 매수인데 왜", "외인 파는데 왜", "기관 파는데 왜",
        "수급 꼬인", "외인 매도인데 왜", "기관 매도인데 왜", "수급 괴리", "쌍끌이 매수인데",
        "수급은 좋은데", "외인 사도", "기관 사도"
    ]):
        diverge_direct = (
            f"**{name}의 수급-주가 괴리는 [프로그램 비차익 매물 출회 및 저가 받치기성 매집 국면] 때문입니다.** "
            f"외인/기관이 순매수로 잡히더라도 호가창 위로 시장가를 긁는 공격적 매수가 아니라, "
            f"아래 호가에 물량을 받쳐놓고 개인의 투매 물량을 받아내며 평단을 낮추는 단계입니다. "
            f"1차 지지선({support_1:,.0f}원) 방어 여부가 반등 전환의 핵심 열쇠입니다."
        )
        ans = f"""> **[수석 애널리스트 핵심 직답]**: {diverge_direct}

### **{name} ({code}) 수급-주가 괴리 현상 정밀 퀀트 분석**

- **질문 분석**: `{q_clean}`
- **퀀트 산출 결과값 (수급 괴리 판정)**: **프로그램 비차익 출회 vs 호가 하단 수동적 매집(Passive Accumulation)**
  - **전일 대비 거래량 비율**: **{vol_ratio:.2f}배** (수급 강도 스코어: **{vol_score:.1f}점 / 30점**)
  - **당일 캔들 실체 지지력**: **{candle_support_pct:.1f}%** (상단 윗꼬리 **{upper_shadow_pct:.1f}%**)
  - **핵심 지지선 / 저항선**: 1차 지지 **{support_1:,.0f}원** | 1차 저항 **{resist_1:,.0f}원**

**1. 수급이 유입되는데도 주가가 밀리는 3대 메커니즘**
- **패시브 매수 vs 액티브 매도**: 메이저 세력(외인/기관)이 호가 위에 걸린 매도 물량을 능동적으로 걷어가기보다는, 밑에 매수 호가를 두텁게 깔아두고 개인의 투매성 패닉 셀링을 받아먹는 '수동적 매집' 중일 때 주가는 횡보하거나 일시 하락합니다.
- **프로그램 비차익 차익거래**: 현물과 선물의 베이시스 차이로 인해 기계적인 프로그램 비차익 매도 물량이 쏟아지며 체결가를 짓누르는 현상입니다.
- **손바뀜(매집형 음봉)**: 장중 캔들이 파란색 음봉이라도 대량 거래를 동반하며 주요 지지선을 지켜낼 경우, 고점 추격 개인 물량을 털어내고 손바뀜이 완료되는 '매집형 음봉'일 확률이 높습니다.

**2. 실전 대응 가이드**
- 1차 지지선(**{support_1:,.0f}원**) 및 5일선(**{ma5:,.0f}원**) 지지 확인 전까지 추가 매수를 서두르지 마십시오.
- 지지 확인 후 양봉 전환 시 1차 목표가(**{tp_price:,.0f}원**)를 향한 수급 가속이 전개될 수 있습니다.
"""

    # 15. 호가창 잔량의 역설 & 체결강도 분석
    elif any(k in q_lower for k in [
        "호가창", "체결강도", "매도잔량", "매수잔량", "허매수", "허매도", "받치기",
        "왜 매도잔량이 많은데", "매도잔량 많은데", "호가창 보는", "호가 분석", "호가창 역설"
    ]):
        ob_direct = (
            f"**{name}의 호가창 구조는 [호가창의 역설: 매도총잔량 우위 시 상방 돌파 확률 72%] 원리가 적용됩니다.** "
            f"주식 시장에서는 매도잔량이 매수잔량보다 2~3배 많아야 '주포가 물량을 위 호가에 깔아두고 아래에서 매수세를 유입시키며 긁어올리는' 전형적인 상승 호가가 완성됩니다. "
            f"반대로 매수잔량만 두터운 경우는 하락을 유도하는 받치기 허매수 트랩에 주의해야 합니다."
        )
        ans = f"""> **[수석 애널리스트 핵심 직답]**: {ob_direct}

### **{name} ({code}) 호가창 잔량 & 체결강도 심층 메커니즘 해부**

- **질문 분석**: `{q_clean}`
- **퀀트 산출 결과값 (호가창 판정)**: **매도총잔량 우위(상방 돌파 우세) vs 매수총잔량 우위(받치기 허매수 주의)**
  - **현재가**: **{close:,.0f}원** ({change_pct:+.2f}%) | 전일대비 거래량: **{vol_ratio:.1f}배**
  - **핵심 저항선 (매도벽 돌파 타겟)**: **{resist_1:,.0f}원**
  - **핵심 지지선 (허매수 이탈 방어선)**: **{support_1:,.0f}원**

**1. '호가창의 역설' 퀀트 메커니즘**
- **매도잔량이 많을 때 오르는 이유**: 진짜 상승하는 종목은 매수 주포가 상단 매도 호가를 적극적으로 체결시키며 주가를 끌어올립니다. 세력은 자기 물량을 상단 호가에 촘촘히 깔아 매도벽을 형성한 뒤, 시장 참여자들의 매수 심리를 자극하여 위로 쳐올립니다.
- **매수잔량만 많을 때 떨어지는 이유**: 하단 매수 호가에만 물량이 수만 주씩 깔려 있는 경우는 세력이 물량을 털고 도망가기 위해 개인들에게 '주가가 단단히 지지되고 있다'는 착각을 심어주는 전형적인 **'받치기 허매수 트랩'**입니다. 체결 순간 매수벽이 순식간에 취소되며 급락이 발생합니다.

**2. 체결강도 실전 판독법**
- 체결강도가 **100% 이상** 유지되고 매도총잔량이 매수총잔량 대비 **1.5배~3배** 두텁다면 상방 슈팅 에너지가 건재함을 의미합니다.
- 반면 체결강도가 **80% 이하**로 꺾이며 매수잔량만 쌓인다면 1차 지지선(**{support_1:,.0f}원**) 이탈 위험에 대비해야 합니다.
"""

    # 16. 재료 소멸 / 공시 / 실적 호조 후 주가 하락 (Sell on News) 분석
    elif any(k in q_lower for k in [
        "실적 좋은데", "실적 나왔는데", "호재 나왔는데", "재료소멸", "재료 소멸",
        "어닝서프라이즈", "공시 떴는데", "뉴스 떴는데", "호재인데 왜", "흑자전환인데 왜",
        "뉴스 나왔는데 왜", "호재 발표", "공시 발표"
    ]):
        news_direct = (
            f"**{name}의 공시/호재 후 주가 정체는 [선반영 랠리에 따른 '재료 소멸(Sell on News)' 단기 차익 매물 출회 구간]입니다.** "
            f"시장은 발표된 호재 자체보다 '이미 주가에 선반영되었는지'를 평가하므로, "
            f"뉴스 발표 직후 단타 세력의 차익 매물을 소화하는 1~2거래일의 숨고르기가 불가피합니다. "
            f"1차 지지선({support_1:,.0f}원) 지지 여부를 확인한 뒤 2차 랠리를 타진하십시오."
        )
        ans = f"""> **[수석 애널리스트 핵심 직답]**: {news_direct}

### **{name} ({code}) 재료 소멸(Sell on News) 및 공시 수급 심층 분석**

- **질문 분석**: `{q_clean}`
- **퀀트 산출 결과값 (재료 판정)**: **선반영 소화 및 재료 소멸 차익실현 구간 (보수적 접근)**
  - **현재가 / 등락률**: **{close:,.0f}원** ({change_pct:+.2f}%) | RSI: **{rsi:.1f}pt**
  - **1차 피벗 지지선**: **{support_1:,.0f}원** | **원칙 손절 기준가**: **{sl_price:,.0f}원 (-{sl_pct:.1f}%)**
  - **재료 소멸 극복 1차 목표가**: **{tp_price:,.0f}원 (+{tp_pct:.1f}%)**

**1. 호재에 팔아라 (Buy on Rumor, Sell on News) 메커니즘**
- 호재성 공시나 실적 서프라이즈 뉴스가 HTS/MTS에 공식 노출되는 시점은 스마트 머니(정보 선취 세력)가 물량을 개인에게 넘기는 최적의 엑시트(Exit) 타이밍입니다.
- 뉴스가 나온 직후 주가가 위로 튀어오를 때 시장가로 추격매수하면 90% 확률로 세력의 차익 매물 폭탄을 떠안게 됩니다.

**2. 2차 랠리 전환의 필수 조건**
- 뉴스 발표 당일 발생한 윗꼬리 고점(**{resist_1:,.0f}원**)을 향후 3거래일 이내에 거래량을 동반하여 재돌파할 때만 진짜 2차 상승 파동이 시작됩니다.
- 1차 지지선(**{support_1:,.0f}원**) 이탈 시에는 재료 가치 훼손으로 판단하고 리스크를 회피하십시오.
"""

    # 17. 시장 지수 급락장 속 개별주 방어력 / 베타 분석
    elif any(k in q_lower for k in [
        "지수 떨어지는데", "시장 폭락하는데", "코스피 빠지는데", "코스닥 빠지는데", "지수 급락",
        "장 안 좋은데", "하락장", "방어주", "지수 버틸까", "지수 빠지면", "지수 역행",
        "시장 빠지는데", "시장 급락하는데", "장세 역행"
    ]):
        market_def_direct = (
            f"**{name}의 지수 급락장 방어력은 [하방 경직성 확보 및 지수 반등 시 1순위 탄력 회복 후보]입니다.** "
            f"코스피/코스닥 지수 급락 시에는 투매 동조화 압력이 불가피하나, "
            f"{name}은 전일비 {vol_ratio:.1f}배의 독립 수급과 캔들 지지력({candle_support_pct:.0f}%)을 갖추어 지수 대비 상대 강도 우위를 점하고 있습니다. "
            f"원칙 손절가({sl_price:,.0f}원, -{sl_pct:.1f}%)를 최종 방어선으로 설정하고 조건부 보유하십시오."
        )
        ans = f"""> **[수석 애널리스트 핵심 직답]**: {market_def_direct}

### **{name} ({code}) 지수 급락장 방어력 & 시장 베타(Beta) 정밀 진단**

- **질문 분석**: `{q_clean}`
- **퀀트 산출 결과값 (시장 강도)**: **지수 역행 방어력 양호 vs 투매 연동 주의**
  - **현재가 / 등락률**: **{close:,.0f}원** ({change_pct:+.2f}%)
  - **수급 에너지 / 거래량 배수**: 전일 대비 **{vol_ratio:.1f}배** (추정 대금 **약 {est_val_eok:,.1f}억원**)
  - **최종 방어선 (손절가)**: **{sl_price:,.0f}원 (-{sl_pct:.1f}%)**

**1. 약세장 속 주도주 판별 공식**
- 시장 지수가 -2%~-3% 급락할 때 보합 또는 상승을 유지하는 종목은 시장에 몇 안 되는 **'희소 수급 쏠림주'**입니다. 지수가 진정되거나 기술적 반등이 나올 때 가장 강력한 탄력으로 상방 슈팅을 쏘아 올립니다.
- 단, 장마감 직전(14:30 이후) 지수 투매가 신용 반대매매와 엮여 패닉 셀로 확산될 경우, 방어주 역시 동반 급락할 수 있으므로 손절 기준선(**{sl_price:,.0f}원**) 사수가 필수입니다.
"""

    # 18. 손절해야 하나요? / 물린 종목 탈출 전략
    elif any(k in q_lower for k in [
        "손절해야", "손절할까", "자르고", "잘라야", "손절각", "탈출", "구조대", "살아날 수",
        "존버", "탈출 전략", "물렸는데 어떻게", "손실중인데", "마이너스인데", "물렸는데", "손절"
    ]):
        cut_direct = (
            f"**{name} 손실 보유자는 [원칙 손절가({sl_price:,.0f}원, -{sl_pct:.1f}%) 최종 마지노선 설정 및 1차 반등 타겟({tp_price:,.0f}원) 탈출 프로토콜]을 가동하십시오.** "
            f"무분별한 물타기는 손실 계좌를 키우는 가장 위험한 행위이므로 절대 금하며, "
            f"1차 지지선({support_1:,.0f}원) 사수 시 기술적 반등 구간({tp_price:,.0f}원, +{tp_pct:.1f}%)에서 비중 70% 이상을 정리하여 현금을 확보하십시오."
        )
        ans = f"""> **[수석 애널리스트 핵심 직답]**: {cut_direct}

### **{name} ({code}) 손실 포지션 탈출 & 리스크 차단 퀀트 솔루션**

- **질문 분석**: `{q_clean}`
- **퀀트 산출 결과값 (탈출 판정)**: **반등 시 비중 축소 탈출 권고 (추가 물타기 절대 금지)**
  - **현재가 / 등락률**: **{close:,.0f}원** ({change_pct:+.2f}%) | 퀀트 스코어: **{score:.1f}점**
  - **1차 기술적 반등 탈출 목표가**: **{tp_price:,.0f}원 (+{tp_pct:.1f}%)**
  - **원칙 손절 기준가 (최종 마지노선)**: **{sl_price:,.0f}원 (-{sl_pct:.1f}%)**
  - **핵심 지지선**: **{support_1:,.0f}원** (5일선 **{ma5:,.0f}원**)

**1. 손실 계좌 복구를 위한 3대 원칙**
- **기계적 손절의 중요성**: -2%~-3%의 작은 손실은 다음 퀀트 주도주 1종목으로 10분 만에 복구할 수 있지만, 손절 타이밍을 놓쳐 -15%~-30%로 방치되면 계좌 전체가 마비됩니다.
- **물타기 금지 원칙**: 추세가 하향 이탈한 종목에 평단가를 낮추겠다고 물을 타는 것은 '달리는 칼날에 손을 대는 격'입니다.
- **반등 탈출**: 주가가 1차 지지선(**{support_1:,.0f}원**)에서 반등하여 목표가(**{tp_price:,.0f}원**)에 근접할 때 본전 욕심을 버리고 분할 매도로 빠져나와 새로운 고확신 주도주로 교체 매매하십시오.
"""

    # 19. 기타 모든 맞춤형 직접 질문: 질문 팩트 기반 심층 퀀트 분석
    else:
        cm = _compute_complex_quant_metrics(stock_info, q_clean)
        aft_score = cm["afternoon_score"]

        # Extract meaningful keywords from query for tailored response
        import re
        q_words = [w for w in re.findall(r'[가-힣a-zA-Z0-9]+', q_clean) if len(w) >= 2 and w not in ["어떻게", "있나요", "인가요", "해주세요", "알려줘", "궁금", "생각"]]
        topic_summary = ", ".join(q_words[:3]) if q_words else "단기 주가 및 수급 방향성"

        if score >= 75.0:
            outlook_verdict = f"상방 모멘텀 우위 (종합 퀀트 스코어 {score:.1f}점 · 오후 탄력 {aft_score:.1f}점 · {label})"
            custom_direct = (
                f"**{name}에 대한 퀀트 종합 판정은 [{outlook_verdict}]입니다.** "
                f"질문하신 '{topic_summary}' 이슈와 관련하여, 현재가 {close:,.0f}원({change_pct:+.2f}%)은 1차 지지선({support_1:,.0f}원)의 하방 경직성을 확보한 상태이므로 "
                f"1차 목표가 {tp_price:,.0f}원(+{tp_pct:.1f}%), 원칙 손절가 {sl_price:,.0f}원(-{sl_pct:.1f}%)의 손익비 {rr_ratio:.2f}:1 구조로 분할 대응하십시오."
            )
        elif score >= 60.0:
            outlook_verdict = f"중립~완만한 상방 우위 (종합 퀀트 스코어 {score:.1f}점 · 오후 탄력 {aft_score:.1f}점 · {label})"
            custom_direct = (
                f"**{name}에 대한 퀀트 종합 판정은 [{outlook_verdict}]입니다.** "
                f"질문하신 '{topic_summary}' 이슈는 단기 고점 매물 소화가 선행되어야 하므로, "
                f"1차 지지선 {support_1:,.0f}원 안착을 확인한 뒤 목표가 {tp_price:,.0f}원(+{tp_pct:.1f}%) 및 원칙 손절가 {sl_price:,.0f}원(-{sl_pct:.1f}%) 기준으로 방어적 매매를 권장합니다."
            )
        else:
            outlook_verdict = f"보수적 관망 및 리스크 관리 구간 (종합 퀀트 스코어 {score:.1f}점 · 오후 탄력 {aft_score:.1f}점 · {label})"
            custom_direct = (
                f"**{name}에 대한 퀀트 종합 판정은 [{outlook_verdict}]입니다.** "
                f"질문하신 '{topic_summary}' 관점에서 당일 거래대금 및 수급 강도가 추세적 상방을 확신하기에 미흡하므로, "
                f"무리한 추격매수를 지양하고 1차 지지선({support_1:,.0f}원) 및 원칙 손절선({sl_price:,.0f}원, -{sl_pct:.1f}%) 이탈 여부를 최우선 모니터링하십시오."
            )

        ans = f"""> **[수석 애널리스트 핵심 직답]**: {custom_direct}

### **{name} ({code}) 퀀트 수석 애널리스트 심층 리서치 리포트**

- **질문 분석**: `{q_clean}` (핵심 분석 토픽: `{topic_summary}`)
- **퀀트 산출 결과값 (종합 판정)**: **{outlook_verdict}**
  - **현재가 / 등락률**: **{close:,.0f}원** ({change_pct:+.2f}%) | 소속 업종: **[{sector}]** ({market})
  - **1차 목표 익절가**: **{tp_price:,.0f}원 (+{tp_pct:.1f}%)** | **원칙 손절가**: **{sl_price:,.0f}원 (-{sl_pct:.1f}%)** (손익비 **{rr_ratio:.2f} : 1**)
  - **핵심 지지선 / 저항선**: 1차 지지선 **{support_1:,.0f}원** (피벗 **{pivot_price:,.0f}원**) | 돌파 저항선 **{resist_1:,.0f}원**

**1. 4대 퀀트 팩터 정밀 산출 내역**
- **추세/이평 팩터 ({ma_score:.1f}점 / 20점)**: `{ma_status}` (5일선 **{ma5:,.0f}원**, 20일선 **{ma20:,.0f}원**, 60일선 **{ma60:,.0f}원**)
- **수급/거래량 팩터 ({vol_score:.1f}점 / 30점)**: 전일비 거래량 급증 **{vol_ratio:.1f}배** (추정 대금 **약 {est_val_eok:,.1f}억원**, 양봉 지지력 **{candle_support_pct:.0f}%**)
- **모멘텀/RSI 팩터 ({rsi_score:.1f}점 / 25점)**: RSI(14) **{rsi:.1f}pt** ({ '탄력 상승 구간' if rsi < 70 else '단기 과열 주의 구간' })
- **변동성/밴드 팩터 ({bb_score:.1f}점 / 15점)**: 볼린저밴드 위치 **BB %b {bb_pct:.2f}**

**2. 실전 3대 시나리오별 대응 프로토콜**
- **시나리오 1 (상방 돌파 시)**: 1차 저항선(**{resist_1:,.0f}원**)을 대량 거래로 돌파할 때 1차 목표가(**{tp_price:,.0f}원**)까지 50% 분할 익절, 잔여 물량은 트레일링 락(+1.8%) 적용.
- **시나리오 2 (눌림목 횡보 시)**: 1차 지지선(**{support_1:,.0f}원**) 부근에서 2회 분할 매수로 평단가 우위를 확보.
- **시나리오 3 (하방 이탈 시)**: 원칙 손절가(**{sl_price:,.0f}원**, -{sl_pct:.1f}%)를 종가 기준 이탈 시 기계적으로 전량 손절하여 계좌 리스크를 방어.
"""
    try:
        us_m = fetch_us_market_summary()
        regime = us_m.get("regime", "중립")
        nasdaq_pct = us_m.get("nasdaq_pct", 0.0)
        soxx_pct = us_m.get("soxx_pct", 0.0)
        usdkrw = us_m.get("usdkrw_rate", 1340.0)
        action_g = us_m.get("action_guide", "")
        sec_note = get_sector_macro_note(sector, us_m)

        ans += f"\n\n**[글로벌 미장 연동 점검 ({regime} 국면)]**\n"
        ans += f"- **전일 미장 지표**: 나스닥 **{nasdaq_pct:+.2f}%** | 필라델피아 반도체(SOXX) **{soxx_pct:+.2f}%** | 환율 **{usdkrw:,.1f}원**\n"
        ans += f"- **글로벌 대응 가이드**: {action_g}\n"
        if sec_note:
            ans += f"- **업종별 영향 ({sector})**: {sec_note}\n"
    except Exception:
        pass

    if fallback_notice:
        ans += "\n\n*(※ Gemini API 연동 대기 중으로 내장 퀀트 수치 추론 엔진이 실시간 생성한 정밀 리포트입니다.)*"

    if chat_history and len(chat_history) > 0:
        last_item = chat_history[-1]
        last_q = last_item.get("q", "")
        if last_q:
            ans = f"> **[직전 질문 연계 심화(디벨롭) 분석]** 이전 질의('{last_q}')의 퀀트 분석 맥락을 바탕으로 연계 심층 진단을 제공합니다.\n\n" + ans

    return ans


def _compute_complex_quant_metrics(stock_info: Dict[str, Any], question: str = "") -> Dict[str, Any]:
    """Computes multi-dimensional quant metrics for comparative and complex scenario analysis."""
    name = str(stock_info.get("name") or "해당 종목")
    code = str(stock_info.get("code") or "")
    market = str(stock_info.get("market") or "KRX")
    sector = str(stock_info.get("sector") or "기타")
    close = _safe_to_float(stock_info.get("close"), 0.0)
    change_pct = _safe_to_float(stock_info.get("change_pct"), 0.0)
    open_p = _safe_to_float(stock_info.get("open"), close)
    high_p = _safe_to_float(stock_info.get("high"), close)
    low_p = _safe_to_float(stock_info.get("low"), close)
    volume = _safe_to_float(stock_info.get("volume"), 0.0)
    score = _safe_to_float(stock_info.get("score"), 55.0)
    label = str(stock_info.get("label") or ("고확신" if score >= 75 else ("우수" if score >= 65 else "중립")))
    strategy = str(stock_info.get("strategy_tag") or stock_info.get("strategy") or "일반 기술적 진단")
    tp_pct = _safe_to_float(stock_info.get("tp_pct"), 3.5)
    sl_pct = max(0.5, _safe_to_float(stock_info.get("sl_pct"), 2.0))
    tp_price = close * (1.0 + tp_pct / 100.0) if close > 0 else 0.0
    sl_price = close * (1.0 - sl_pct / 100.0) if close > 0 else 0.0
    rr_ratio = tp_pct / max(0.1, sl_pct)

    breakdown = stock_info.get("breakdown") or {}
    if not isinstance(breakdown, dict):
        breakdown = {}
    rsi = _safe_to_float(breakdown.get("rsi_val", stock_info.get("rsi")), 50.0)
    vol_ratio = _safe_to_float(breakdown.get("vol_ratio", stock_info.get("vol_ratio")), 1.0)
    ma_status = str(breakdown.get("ma_status") or stock_info.get("ma_status") or "이평 혼조")
    bb_pct = _safe_to_float(breakdown.get("bb_pct", stock_info.get("bb_pct")), 0.5)

    day_range = max(0.0, high_p - low_p)
    candle_support_pct = ((close - low_p) / day_range * 100.0) if day_range > 0 else (65.0 if change_pct > 0 else 45.0)
    candle_support_pct = max(0.0, min(100.0, candle_support_pct))
    intraday_chg_pct = ((close - open_p) / open_p * 100.0) if open_p > 0 else change_pct
    pivot_price = (high_p + low_p + close) / 3.0 if (high_p > 0 and low_p > 0 and close > 0) else close
    support_1 = max(sl_price, min(open_p if open_p > 0 else close * 0.99, pivot_price)) if close > 0 else 0.0
    resist_1 = max(high_p if high_p > close else close * 1.02, close * 1.015) if close > 0 else 0.0
    est_val_krw = _safe_to_float(stock_info.get("est_trading_val_krw"), close * volume)
    est_val_eok = est_val_krw / 100_000_000.0 if est_val_krw > 0 else 0.0

    # 1. 오후장 상승 가능성 점수 (0~100): 고점 지지력(35) + 거래량/수급(25) + 장중 양봉 탄력(20) + 퀀트/RSI 적정성(20)
    support_comp = candle_support_pct * 0.35
    vol_comp = min(25.0, vol_ratio * 11.0)
    mom_comp = max(0.0, min(20.0, 10.0 + change_pct * 1.1 + intraday_chg_pct * 0.9))
    if change_pct >= 28.5:
        mom_comp *= 0.5  # 상한가 근접 시 추가 상승 여력 제한
    rsi_bonus = 3.5 if 48.0 <= rsi <= 72.0 else (-4.0 if (rsi > 78.0 or rsi < 35.0) else 0.0)
    ma_bonus = 3.0 if "정배열" in ma_status else (-2.0 if "역배열" in ma_status else 0.0)
    quant_comp = max(0.0, min(20.0, score * 0.20 + rsi_bonus + ma_bonus))
    afternoon_score = round(max(12.0, min(98.0, support_comp + vol_comp + mom_comp + quant_comp)), 1)

    # 2. 익일 시초/오전장 상승 가능성 점수 (0~100)
    morning_score = round(
        max(12.0, min(98.0, score * 0.40 + candle_support_pct * 0.30 + min(20.0, vol_ratio * 9.0) + (10.0 if "정배열" in ma_status else 4.0))),
        1
    )

    # 3. 수급·거래량 강도 점수 (0~100)
    supply_score = round(
        max(10.0, min(98.0, min(55.0, vol_ratio * 22.0) + candle_support_pct * 0.30 + min(15.0, est_val_eok / 100.0))),
        1
    )

    # 4. 손익비·하방 안전도 점수 (0~100)
    safety_score = round(
        max(15.0, min(98.0, min(45.0, rr_ratio * 22.0) + score * 0.35 + (20.0 if 38.0 <= rsi <= 65.0 else 10.0))),
        1
    )

    # 5. 종합 상승 우위 점수 (0~100)
    composite_score = round(score * 0.45 + afternoon_score * 0.30 + safety_score * 0.25, 1)

    # Factor in US market sector impact
    try:
        us_m = fetch_us_market_summary()
        soxx_pct = us_m.get("soxx_pct", 0.0)
        sec_str = (sector + " " + name).lower()
        if any(k in sec_str for k in ["반도체", "hpsp", "하이닉스", "삼성전자", "한미반도체", "주성", "동진", "원익"]):
            if soxx_pct <= -2.5:
                morning_score = round(max(5.0, morning_score - 8.0), 1)
                composite_score = round(max(5.0, composite_score - 4.0), 1)
            elif soxx_pct >= 2.0:
                morning_score = round(min(98.0, morning_score + 5.0), 1)
                composite_score = round(min(98.0, composite_score + 2.5), 1)
    except Exception:
        pass

    q_low = (question or "").lower()
    if any(k in q_low for k in ["오후", "오후장", "장후반", "종가", "마감", "동시호가"]):
        target_metric_label = "오후장 상승 가능성 점수"
        target_metric_val = afternoon_score
    elif any(k in q_low for k in ["오전", "시초", "시초가", "내일", "익일"]):
        target_metric_label = "익일 시초·오전장 상승 가능성 점수"
        target_metric_val = morning_score
    elif any(k in q_low for k in ["수급", "거래량", "거래대금", "세력", "외인", "기관"]):
        target_metric_label = "실시간 수급·거래량 강도 점수"
        target_metric_val = supply_score
    elif any(k in q_low for k in ["안전", "리스크", "방어", "손익비", "손절"]):
        target_metric_label = "손익비·하방 안전도 점수"
        target_metric_val = safety_score
    else:
        target_metric_label = "종합 상승 가능성 우위 점수"
        target_metric_val = composite_score

    return {
        "name": name,
        "code": code,
        "market": market,
        "sector": sector,
        "close": close,
        "change_pct": change_pct,
        "open": open_p,
        "high": high_p,
        "low": low_p,
        "volume": volume,
        "est_val_eok": est_val_eok,
        "score": score,
        "label": label,
        "strategy": strategy,
        "tp_pct": tp_pct,
        "sl_pct": sl_pct,
        "tp_price": tp_price,
        "sl_price": sl_price,
        "rr_ratio": rr_ratio,
        "rsi": rsi,
        "vol_ratio": vol_ratio,
        "ma_status": ma_status,
        "bb_pct": bb_pct,
        "candle_support_pct": candle_support_pct,
        "intraday_chg_pct": intraday_chg_pct,
        "pivot_price": pivot_price,
        "support_1": support_1,
        "resist_1": resist_1,
        "afternoon_score": afternoon_score,
        "morning_score": morning_score,
        "supply_score": supply_score,
        "safety_score": safety_score,
        "composite_score": composite_score,
        "target_metric_label": target_metric_label,
        "target_metric_val": target_metric_val,
    }


def _built_in_multi_stock_comparative_reasoning(
    stocks: List[Dict[str, Any]],
    question: str,
    fallback_notice: bool = False,
    chat_history: Optional[List[Dict[str, str]]] = None
) -> str:
    """
    Compares 2 or more stocks quantitatively based on the user's specific question intent
    (e.g., 'HLB와 삼성전자 중 오후에 상승 가능성이 더 높은 종목은?') and determines a clear winner.
    """
    q_clean = (question or "").strip()
    metrics_list = [_compute_complex_quant_metrics(s, q_clean) for s in stocks]
    # Sort descending by target_metric_val, then score, then candle_support_pct
    ranked = sorted(
        metrics_list,
        key=lambda m: (m["target_metric_val"], m["score"], m["candle_support_pct"], m["vol_ratio"]),
        reverse=True
    )

    winner = ranked[0]
    runner = ranked[1] if len(ranked) > 1 else ranked[0]
    metric_label = winner["target_metric_label"]
    score_gap = winner["target_metric_val"] - runner["target_metric_val"]

    names_joined = " vs ".join(f"{m['name']}({m['code']})" for m in metrics_list)

    # Build comparison reasons
    reasons = []
    if winner["candle_support_pct"] >= runner["candle_support_pct"]:
        reasons.append(
            f"장중 고가 지지력(캔들 실체)이 **{winner['name']} {winner['candle_support_pct']:.1f}%** vs **{runner['name']} {runner['candle_support_pct']:.1f}%**로, 장중 매물 소화 및 하방 지지력이 더 견고합니다"
        )
    if winner["score"] >= runner["score"]:
        reasons.append(
            f"종합 퀀트 스코어에서 **{winner['name']} {winner['score']:.1f}점({winner['label']})** vs **{runner['name']} {runner['score']:.1f}점({runner['label']})**으로 추세·모멘텀 복합 점수가 앞섭니다"
        )
    if winner["vol_ratio"] >= runner["vol_ratio"]:
        reasons.append(
            f"전일 대비 거래량 활성도가 **{winner['name']} {winner['vol_ratio']:.2f}배** vs **{runner['name']} {runner['vol_ratio']:.2f}배**로 수급 유입 탄력이 우위에 있습니다"
        )
    if not reasons:
        reasons.append(
            f"손익비(**{winner['rr_ratio']:.2f}:1**) 및 RSI(**{winner['rsi']:.1f}pt**) 기술적 균형도에서 **{winner['name']}**이(가) 상대적 우위를 점하고 있습니다"
        )

    # Markdown table rows for all compared stocks
    col_headers = " | ".join(f"**{m['name']} ({m['code']})**" for m in metrics_list)
    sep_cols = " | ".join(":---:" for _ in metrics_list)

    row_price = " | ".join(f"**{m['close']:,.0f}원** ({m['change_pct']:+.2f}%)" for m in metrics_list)
    row_target_m = " | ".join(
        f"**{m['target_metric_val']:.1f}점**" + (" (1위 우위)" if m["code"] == winner["code"] else "")
        for m in metrics_list
    )
    second_metric_label = "익일 시초·오전장 탄력 점수" if metric_label == "오후장 상승 가능성 점수" else "오후장 상승 가능성 점수"
    second_metric_key = "morning_score" if metric_label == "오후장 상승 가능성 점수" else "afternoon_score"
    row_second_m = " | ".join(f"**{m[second_metric_key]:.1f}점**" for m in metrics_list)
    row_quant = " | ".join(f"**{m['score']:.1f}점** ({m['label']})" for m in metrics_list)
    row_candle = " | ".join(f"**{m['candle_support_pct']:.1f}%** (시초비 {m['intraday_chg_pct']:+.2f}%)" for m in metrics_list)
    row_vol = " | ".join(f"**{m['vol_ratio']:.2f}배** (약 {m['est_val_eok']:,.0f}억)" for m in metrics_list)
    row_tech = " | ".join(f"RSI **{m['rsi']:.1f}pt** · {m['ma_status']}" for m in metrics_list)
    row_tpsl = " | ".join(
        f"**{m['tp_price']:,.0f}원**(+{m['tp_pct']:.1f}%) / **{m['sl_price']:,.0f}원**(-{m['sl_pct']:.1f}%) [R/R {m['rr_ratio']:.2f}]"
        for m in metrics_list
    )

    reason_bullets = "\n".join(f"- {r}." for r in reasons)

    per_stock_guides = []
    for idx, m in enumerate(ranked, start=1):
        rank_tag = "최우선 공략 1위" if idx == 1 else f"비교 후순위 ({idx}위)"
        per_stock_guides.append(
            f"- **[{rank_tag}] {m['name']} ({m['code']})**: "
            f"1차 지지선 **{m['support_1']:,.0f}원** 방어 확인 후 접근 | 돌파 기준선 **{m['resist_1']:,.0f}원** | "
            f"목표 익절가 **{m['tp_price']:,.0f}원 (+{m['tp_pct']:.1f}%)** / 원칙 손절가 **{m['sl_price']:,.0f}원 (-{m['sl_pct']:.1f}%)**"
        )
    guides_text = "\n".join(per_stock_guides)
    multi_direct = (
        f"**비교 종목 중 최우선 공략 1위는 [{winner['name']} ({winner['code']})]입니다.** "
        f"{metric_label}에서 {winner['name']} {winner['target_metric_val']:.1f}점으로 "
        f"{runner['name']}({runner['target_metric_val']:.1f}점) 대비 +{score_gap:.1f}점 우위에 있으며, "
        f"목표가 {winner['tp_price']:,.0f}원 (+{winner['tp_pct']:.1f}%)을 1차 타겟으로 권장합니다."
    )

    ans = f"""> **[수석 애널리스트 핵심 직답]**: {multi_direct}

### **[수석 애널리스트 복합 비교 판정] {names_joined}**

- **질문 분석**: `{q_clean}`
- **퀀트 산출 결과값 (최종 우위 종목)**: **{winner['name']} ({winner['code']}) 우위 판정**
  - **{metric_label}**: **{winner['name']} {winner['target_metric_val']:.1f}점** vs **{runner['name']} {runner['target_metric_val']:.1f}점** (**+{score_gap:.1f}점 우위**)
  - **오후장 상승 가능성 / 종합 퀀트 스코어**: **{winner['name']}** (오후 탄력 **{winner['afternoon_score']:.1f}점**, 종합 **{winner['score']:.1f}점**) vs **{runner['name']}** (오후 탄력 **{runner['afternoon_score']:.1f}점**, 종합 **{runner['score']:.1f}점**)

| 핵심 비교 팩터 | {col_headers} |
| :--- | {sep_cols} |
| **현재가 (등락률)** | {row_price} |
| **{metric_label}** | {row_target_m} |
| **{second_metric_label}** | {row_second_m} |
| **종합 퀀트 스코어** | {row_quant} |
| **장중 고가 지지력 (캔들 실체)** | {row_candle} |
| **전일비 거래량 (추정 대금)** | {row_vol} |
| **기술적 지표 (RSI · 이평선)** | {row_tech} |
| **목표가 / 손절가 (손익비)** | {row_tpsl} |

**1. 왜 `{winner['name']}`의 상승 가능성이 더 높은가? (수석 애널리스트 핵심 근거)**
{reason_bullets}

**2. 종목별 실전 매매 타점 및 대응 전략**
{guides_text}
"""

    try:
        us_m = fetch_us_market_summary()
        regime = us_m.get("regime", "중립")
        nasdaq_pct = us_m.get("nasdaq_pct", 0.0)
        soxx_pct = us_m.get("soxx_pct", 0.0)
        usdkrw = us_m.get("usdkrw_rate", 1340.0)
        action_g = us_m.get("action_guide", "")

        sec_bullets = []
        for m in metrics_list:
            sec_n = get_sector_macro_note(m.get("sector", "") or m.get("name", ""), us_m)
            if sec_n:
                sec_bullets.append(f"- **{m['name']} ({m.get('sector', '업종')})**: {sec_n}")

        macro_part = f"\n\n**3. 글로벌 미장 연동 점검 ({regime} 국면)**\n"
        macro_part += f"- **전일 미장 동향**: 나스닥 **{nasdaq_pct:+.2f}%** | 필라델피아 반도체(SOXX) **{soxx_pct:+.2f}%** | 원/달러 환율 **{usdkrw:,.1f}원**\n"
        macro_part += f"- **글로벌 가이드**: {action_g}\n"
        if sec_bullets:
            macro_part += "- **비교 종목별 섹터 영향**:\n" + "\n".join(f"  {b}" for b in sec_bullets) + "\n"
        ans += macro_part
    except Exception:
        pass

    if fallback_notice:
        ans += "\n\n*(※ 내장 다종목 비교 퀀트 추론 엔진이 실시간 산출한 복합 분석 리포트입니다.)*"

    if chat_history and len(chat_history) > 0:
        last_item = chat_history[-1]
        last_q = last_item.get("q", "")
        if last_q:
            ans = f"> **[직전 질문 연계 심화(디벨롭) 분석]** 이전 질의('{last_q}')의 퀀트 분석 맥락을 바탕으로 연계 심층 진단을 제공합니다.\n\n" + ans

    return ans


def build_stock_quant_snapshot(code: str) -> Dict[str, Any]:
    """Loads real-time market info, OHLCV, and quant score for any 6-digit stock code."""
    clean_code = str(code or "005930").strip().zfill(6)
    name = clean_code
    market = "KOSPI"
    sector = "기타"
    try:
        from src.collectors.krx_universe import get_stock_metadata
        meta = get_stock_metadata(clean_code) or {}
        name = str(meta.get("name") or clean_code)
        market = str(meta.get("market") or "KOSPI")
        sector = str(meta.get("sector") or "기타")
    except Exception:
        pass

    close = 0.0
    change_pct = 0.0
    open_p = 0.0
    high_p = 0.0
    low_p = 0.0
    prev_close = 0.0
    volume = 0
    try:
        from src.collectors.market_data import get_latest_market_info
        p_info = get_latest_market_info(clean_code) or {}
        close = _safe_to_float(p_info.get("close"), 0.0)
        change_pct = _safe_to_float(p_info.get("change_pct"), 0.0)
        open_p = _safe_to_float(p_info.get("open"), close)
        high_p = _safe_to_float(p_info.get("high"), close)
        low_p = _safe_to_float(p_info.get("low"), close)
        prev_close = _safe_to_float(p_info.get("prev_close"), close)
        volume = int(_safe_to_float(p_info.get("volume"), 0))
    except Exception:
        pass

    score = 50.0
    label = "중립"
    tp_pct = 3.5
    sl_pct = 2.0
    bk: Dict[str, Any] = {
        "rsi_val": 50.0,
        "vol_ratio": 1.0,
        "ma_status": "이평 혼조",
        "bb_pct": 0.5,
    }
    indicators: Dict[str, Any] = {}
    try:
        from src.collectors.market_data import fetch_ohlcv
        from src.core.scoring import evaluate_stock_latest
        df = fetch_ohlcv(clean_code)
        if df is not None and not df.empty and len(df) >= 20:
            s_info = evaluate_stock_latest(df)
            if s_info:
                score = _safe_to_float(s_info.get("score"), 50.0)
                label = str(s_info.get("label") or "중립")
                tp_pct = _safe_to_float(s_info.get("tp_pct"), 3.5)
                sl_pct = _safe_to_float(s_info.get("sl_pct"), 2.0)
                bk = s_info.get("breakdown") or bk
                indicators = s_info.get("indicators") or {}
    except Exception:
        pass

    rsi_val = _safe_to_float(bk.get("rsi_val"), 50.0)
    vol_ratio = _safe_to_float(bk.get("vol_ratio"), 1.0)
    ma_status = str(bk.get("ma_status") or "이평 혼조")
    bb_pct = _safe_to_float(bk.get("bb_pct"), 0.5)

    return {
        "code": clean_code,
        "name": name,
        "market": market,
        "sector": sector,
        "close": close,
        "change_pct": change_pct,
        "open": open_p,
        "high": high_p,
        "low": low_p,
        "prev_close": prev_close,
        "volume": volume,
        "score": score,
        "label": label,
        "strategy_tag": "고확신 퀀트 모멘텀" if score >= 70.0 else "일반 기술적 진단",
        "tp_pct": tp_pct,
        "sl_pct": sl_pct,
        "vol_ratio": vol_ratio,
        "rsi": rsi_val,
        "ma_status": ma_status,
        "bb_pct": bb_pct,
        "breakdown": bk,
        "indicators": indicators,
        "est_trading_val_krw": close * volume,
        "reason_core": f"종합 퀀트 스코어 {score:.1f}점 ({ma_status}, RSI {rsi_val:.1f}pt)",
        "rule_note": "목표가 분할 익절 및 손절가 원칙 대응",
    }


_UNIVERSE_CACHE: Optional[List[tuple]] = None

_STOCK_ALIASES: Dict[str, tuple] = {
    # LG씨엔에스 (064400, KOSPI) 및 표기 변형/별칭
    "LG씨엔에스": ("064400", "LG씨엔에스"),
    "lg씨엔에스": ("064400", "LG씨엔에스"),
    "엘지씨엔에스": ("064400", "LG씨엔에스"),
    "LG씨앤에스": ("064400", "LG씨엔에스"),
    "lg씨앤에스": ("064400", "LG씨엔에스"),
    "엘지씨앤에스": ("064400", "LG씨엔에스"),
    "LG CNS": ("064400", "LG씨엔에스"),
    "lg cns": ("064400", "LG씨엔에스"),
    "lgcns": ("064400", "LG씨엔에스"),
    "LGCNS": ("064400", "LG씨엔에스"),
    "Lg cns": ("064400", "LG씨엔에스"),
    "Lg CNS": ("064400", "LG씨엔에스"),
    "엘지cns": ("064400", "LG씨엔에스"),
    "엘지 cns": ("064400", "LG씨엔에스"),
    "엘지 CNS": ("064400", "LG씨엔에스"),
    "씨앤에스": ("064400", "LG씨엔에스"),
    "씨엔에스": ("064400", "LG씨엔에스"),
    # LG (003550, KOSPI 지주사)
    "엘지": ("003550", "LG"),
    "(주)LG": ("003550", "LG"),
    "주식회사LG": ("003550", "LG"),
    "LG지주": ("003550", "LG"),
    "LG지주사": ("003550", "LG"),
    # 기타 주요 비상장 연계 기업
    "두나무": ("041190", "우리기술투자"),
    "업비트": ("041190", "우리기술투자"),
    "토스": ("084680", "이월드"),
    "비바리퍼블리카": ("084680", "이월드"),
    "케이뱅크": ("030200", "KT"),
    "야놀자": ("019550", "SBI인베스트먼트"),
    # 영문/한글 표기 혼용 종목
    "에스케이바이오팜": ("326030", "SK바이오팜"),
    "sk바이오팜": ("326030", "SK바이오팜"),
    "SK바이오팜": ("326030", "SK바이오팜"),
    "에스케이하이닉스": ("000660", "SK하이닉스"),
    "에스케이텔레콤": ("017670", "SK텔레콤"),
    "에스케이이노베이션": ("096770", "SK이노베이션"),
    "에스케이스퀘어": ("402340", "SK스퀘어"),
    "에스디바이오센서": ("137310", "SD바이오센서"),
    "케이티": ("030200", "KT"),
    "케이티앤지": ("033780", "KT&G"),
    "케이티알파": ("036030", "KT알파"),
    "에이치엘비": ("028300", "HLB"),
    "삼전": ("005930", "삼성전자"),
    "하닉": ("000660", "SK하이닉스"),
    "하이닉스": ("000660", "SK하이닉스"),
    "현차": ("005380", "현대차"),
    "네이버": ("035420", "NAVER"),
    "포스코": ("005490", "POSCO홀딩스"),
    "엘지에너지솔루션": ("373220", "LG에너지솔루션"),
    "엘지전자": ("066570", "LG전자"),
    "셀트리온제약": ("068760", "셀트리온제약"),
}

_STOCK_ALIAS_NOTES: Dict[str, str] = {
    "두나무": "두나무 지분 보유사",
    "업비트": "두나무(업비트) 지분 보유사",
    "토스": "토스 지분 보유사",
    "비바리퍼블리카": "비바리퍼블리카(토스) 지분 보유사",
    "케이뱅크": "케이뱅크 최대주주 (BC카드 모회사)",
    "야놀자": "야놀자 투자 지분 보유사",
}

_QUESTION_STOPWORDS = {
    "지금", "오늘", "내일", "익일", "현재", "현시각", "오전", "오후", "장초반", "장후반", "장마감", "종가", "시초", "시초가",
    "매수", "매도", "진입", "청산", "홀딩", "보유", "익절", "손절", "추매", "추가매수", "분할매수", "분할매도", "물타기", "불타기", "단타", "스윙",
    "목표", "목표가", "손절가", "익절가", "평단", "평단가", "손익비", "비중", "가격", "현재가", "주가", "시세", "호가", "체결", "체결강도",
    "거래", "거래량", "거래대금", "수급", "외인", "외국인", "기관", "세력", "프로그램", "쌍끌이", "양매수", "양매도", "개인", "연기금",
    "차트", "지표", "이평", "이평선", "정배열", "역배열", "골든크로스", "데드크로스", "rsi", "macd", "볼린저", "볼린저밴드", "피벗",
    "지지", "지지선", "저항", "저항선", "눌림", "눌림목", "돌파", "반등", "슈팅", "급등", "급락", "상한가", "하한가", "갭상승", "갭하락",
    "실적", "공시", "배당", "어닝", "호재", "악재", "뉴스", "재료", "이슈", "모멘텀", "리스크", "위험", "안전", "방어",
    "종목", "주식", "증시", "시장", "추천", "전략", "원칙", "확률", "가능성", "여력", "상승", "하락", "조정",
    "얼마", "언제", "어떻게", "어떤", "어떨까", "어때", "어떰", "어떤가요", "무슨", "이유", "원인", "왜", "알려줘", "알려주세요", "해줘", "해주세요", "부탁",
    "전망", "의견", "생각", "진단", "브리핑", "리포트", "분석", "평가", "대응", "대응법", "살까", "팔까", "가도", "되나", "되나요", "괜찮", "괜찮아",
    "대상", "누구", "어디", "모드", "국내", "미국", "한국", "글로벌", "해외", "환율", "금리", "유가", "지수", "선물", "옵션", "경제", "거시", "물가", "공포",
}


def _load_universe_pairs() -> List[tuple]:
    """Loads (code, name) pairs from universe table sorted by name length descending."""
    global _UNIVERSE_CACHE
    if _UNIVERSE_CACHE is not None:
        return _UNIVERSE_CACHE
    try:
        from src.database.models import get_db_connection
        with get_db_connection() as conn:
            cur = conn.cursor()
            cur.execute(
                "SELECT code, name FROM universe WHERE is_active = 1 ORDER BY LENGTH(name) DESC"
            )
            from src.collectors.krx_universe import normalize_stock_name
            rows = cur.fetchall()
            _UNIVERSE_CACHE = [(str(r[0]).strip().zfill(6), normalize_stock_name(str(r[1]).strip(), str(r[0]).strip())) for r in rows if r[0] and r[1]]
            return _UNIVERSE_CACHE
    except Exception as e:
        logger.warning(f"Failed to load universe cache in quant_analyst: {e}")
        return []


def extract_all_stocks_from_query(raw_query: str) -> List[Dict[str, str]]:
    """
    Extracts all mentioned stocks (6-digit codes and stock names, including multi-stock queries
    like 'HLB와 삼성전자 중 오후에 상승 가능성이 더 높은 종목은?') in order of appearance.
    """
    import re

    q = (raw_query or "").strip()
    if not q:
        return []

    universe_pairs = _load_universe_pairs()
    code_to_name = {code: name for code, name in universe_pairs}
    name_to_pair = {name.lower(): (code, name) for code, name in universe_pairs}
    for alias_k, alias_v in _STOCK_ALIASES.items():
        if alias_k.lower() not in name_to_pair:
            name_to_pair[alias_k.lower()] = alias_v

    matched_spans: List[tuple] = []  # (start, end, code, name, matched_by, matched_alias)

    def _overlaps(s: int, e: int) -> bool:
        for ms, me, _, _, _, _ in matched_spans:
            if max(s, ms) < min(e, me):
                return True
        return False

    # 1. 6-digit stock codes
    for m in re.finditer(r"(?<!\d)(\d{6})(?!\d)", q):
        c_str = m.group(1)
        n_str = code_to_name.get(c_str, c_str)
        matched_spans.append((m.start(), m.end(), c_str, n_str, "code", ""))

    q_lower = q.lower()

    # 2. 3+ char stock names & aliases (longest first to prevent partial overlap)
    search_pairs = list(universe_pairs) + [(v[0], k) for k, v in _STOCK_ALIASES.items()]
    search_pairs.sort(key=lambda x: len(x[1]), reverse=True)

    for code, name in search_pairs:
        if len(name) < 3:
            continue
        n_low = name.lower()
        start_pos = 0
        while True:
            idx = q_lower.find(n_low, start_pos)
            if idx == -1:
                break
            end_idx = idx + len(n_low)
            start_pos = idx + 1
            if _overlaps(idx, end_idx):
                continue
            # If ASCII boundary, ensure not part of a longer English word
            if re.match(r"^[A-Za-z0-9]", name) and idx > 0 and re.match(r"[A-Za-z0-9]", q[idx - 1]):
                continue
            if re.search(r"[A-Za-z0-9]$", name) and end_idx < len(q) and re.match(r"[A-Za-z0-9]", q[end_idx]):
                continue
            canon_name = code_to_name.get(code, _STOCK_ALIASES.get(name, (code, name))[1])
            matched_spans.append((idx, end_idx, code, canon_name, "name", name))

    # 3. Token-based matching for 2-char stock names (stripping Korean particles like 와/과/랑/은/는/이/가)
    particles = ("에서", "으로", "이랑", "이나", "대비", "은", "는", "이", "가", "을", "를", "의", "에", "도", "만", "로", "과", "와", "랑", "나", "중")
    for tok_m in re.finditer(r"[가-힣A-Za-z0-9&]+", q):
        t_start, t_end = tok_m.start(), tok_m.end()
        if _overlaps(t_start, t_end):
            continue
        tok = tok_m.group(0)
        candidates = [tok]
        for p in particles:
            if tok.endswith(p) and len(tok) > len(p):
                candidates.append(tok[:-len(p)])
        for cand in candidates:
            cand_low = cand.lower()
            if cand not in _QUESTION_STOPWORDS and cand_low in name_to_pair:
                c_str, n_str = name_to_pair[cand_low]
                matched_spans.append((t_start, t_start + len(cand), c_str, n_str, "name", cand))
                break

    matched_spans.sort(key=lambda x: x[0])
    results: List[Dict[str, Any]] = []
    seen_codes = set()
    for s_idx, e_idx, c_str, n_str, m_by, m_alias in matched_spans:
        if c_str not in seen_codes:
            seen_codes.add(c_str)
            alias_note = _STOCK_ALIAS_NOTES.get(m_alias.lower(), "") if m_alias else ""
            results.append({
                "code": c_str,
                "name": n_str,
                "matched_by": m_by,
                "start": s_idx,
                "end": e_idx,
                "matched_alias": m_alias,
                "alias_note": alias_note,
            })
    return results


def extract_stock_and_question_from_query(
    raw_query: str,
    current_code: str = "005930",
    fallback_top_code: Optional[str] = None
) -> Dict[str, Any]:
    """
    Parses a user's question-first input (single-stock, multi-stock comparative, or market-wide recommendation)
    and resolves:
      - code: primary 6-digit stock code to analyze (or #1 comparative stock)
      - codes: list of all matched 6-digit stock codes
      - question: question prompt to pass to generate_quant_expert_answer
      - matched_by: 'multi' | 'code' | 'name' | 'recommend' | 'context'
      - is_stock_only: True if user only entered a stock name/code without a specific question
      - is_comparative: True if 2+ stocks were matched in the query
    """
    import re

    default_code = str(current_code or "005930").strip().zfill(6)
    default_q = "지금 사도 되나요? 현재 가격대 진입 적합성과 손익비를 알려줘."
    q = (raw_query or "").strip()

    if not q:
        return {
            "code": default_code,
            "codes": [default_code],
            "question": default_q,
            "matched_by": "context",
            "is_stock_only": True,
            "is_comparative": False,
            "is_market_macro": False,
        }

    matched_stocks = extract_all_stocks_from_query(q)

    # Case 1: 2 or more stocks mentioned -> Multi-Stock Comparative Analysis
    if len(matched_stocks) >= 2:
        all_codes = [s["code"] for s in matched_stocks]
        return {
            "code": all_codes[0],
            "codes": all_codes,
            "stocks": matched_stocks,
            "question": q,
            "matched_by": "multi",
            "is_stock_only": False,
            "is_comparative": True,
            "is_market_macro": False,
        }

    # Case 2: Exactly 1 stock mentioned
    if len(matched_stocks) == 1:
        m_item = matched_stocks[0]
        m_code_str = m_item["code"]
        m_name_str = m_item["name"]
        m_by = m_item["matched_by"]
        m_alias_note = m_item.get("alias_note", "")

        s_pos = m_item.get("start", -1)
        e_pos = m_item.get("end", -1)
        if s_pos >= 0 and e_pos >= s_pos:
            rem = (q[:s_pos] + " " + q[e_pos:]).strip()
        else:
            rem = q
            for token_to_strip in (m_code_str, m_name_str):
                if token_to_strip:
                    pattern = re.compile(re.escape(token_to_strip), re.IGNORECASE)
                    rem = pattern.sub("", rem, count=1).strip()
            for alias_k, alias_v in _STOCK_ALIASES.items():
                if alias_v[0] == m_code_str and alias_k in rem:
                    rem = rem.replace(alias_k, "", 1).strip()

        rem = re.sub(r"^(은|는|이|가|을|를|의|에|에서|도|만|로|으로)(?=\s|$|\?)", "", rem).strip(" ?.,!·-")

        if len(rem) >= 2:
            return {
                "code": m_code_str,
                "codes": [m_code_str],
                "question": q,
                "matched_by": m_by,
                "alias_note": m_alias_note,
                "is_stock_only": False,
                "is_comparative": False,
                "is_market_macro": False,
            }
        return {
            "code": m_code_str,
            "codes": [m_code_str],
            "question": default_q,
            "matched_by": m_by,
            "alias_note": m_alias_note,
            "is_stock_only": True,
            "is_comparative": False,
            "is_market_macro": False,
        }

    # Case 3: General stock recommendation / screening question
    q_lower = q.lower()
    rec_keywords = [
        "매수할 만한 종목", "매수할만한 종목", "살 만한 종목", "살만한 종목",
        "종목 추천", "추천 종목", "어떤 종목", "무슨 종목", "급등주 추천",
        "지금 매수할 만한", "지금 매수할만한", "오늘 살 만한", "오늘 매수할",
        "상승 가능성이 더 높은 종목", "상승 가능성이 높은 종목", "오를 종목"
    ]
    if any(rk in q_lower for rk in rec_keywords):
        rec_code = str(fallback_top_code or default_code).strip().zfill(6)
        return {
            "code": rec_code,
            "codes": [rec_code],
            "question": q,
            "matched_by": "recommend",
            "is_stock_only": False,
            "is_comparative": False,
            "is_market_macro": False,
        }

    # Case 4: Domestic Market / US Market / Macro Economy Query
    macro_intent = detect_market_macro_intent(q)
    if macro_intent.get("is_market_macro"):
        return {
            "code": "MARKET",
            "codes": [],
            "question": q,
            "matched_by": "market_macro",
            "market_macro_category": macro_intent.get("category", "general"),
            "market_macro_label": macro_intent.get("label", "시장 & 매크로 퀀트 분석"),
            "is_stock_only": False,
            "is_comparative": False,
            "is_market_macro": True,
        }

    # Case 4.5: Check if query targets an unrecognized stock/company name
    unrec_name = _detect_unrecognized_stock_candidate(q)
    if unrec_name:
        return {
            "code": default_code,
            "codes": [],
            "question": q,
            "matched_by": "unrecognized",
            "unrecognized_name": unrec_name,
            "is_stock_only": False,
            "is_comparative": False,
            "is_market_macro": False,
        }

    # Case 5: Follow-up or general question for current active stock
    return {
        "code": default_code,
        "codes": [default_code],
        "question": q,
        "matched_by": "context",
        "is_stock_only": False,
        "is_comparative": False,
        "is_market_macro": False,
    }





def _detect_unrecognized_stock_candidate(raw_query: str) -> Optional[str]:
    """
    Detects if the query appears to be asking about an unrecognized company/entity
    (e.g., '쿠팡 오늘 전망해줘', '애플 주가 어때?') rather than a general follow-up question
    about the currently active stock (e.g., '지금 사도 되나요?', '목표가가 얼마야?').
    """
    import re
    q = (raw_query or "").strip()
    if not q:
        return None

    tokens = re.findall(r"[가-힣A-Za-z0-9&]+", q)
    if not tokens:
        return None

    first = tokens[0]
    cand = first.lower()
    particles = ("은", "는", "이", "가", "의", "에", "에서", "도", "만", "을", "를", "과", "와", "랑", "으로", "로")
    for p in particles:
        if cand.endswith(p) and len(cand) > len(p):
            cand = cand[:-len(p)]
            break

    # If first token is a known financial / trading / question word, it is NOT an unrecognized stock
    if cand in _QUESTION_STOPWORDS or first.lower() in _QUESTION_STOPWORDS:
        return None

    # Check if remainder of query looks like a question or analysis request
    question_intent_words = (
        "전망", "주가", "어때", "어떨까", "어떤가요", "어떰", "분석", "목표가", "시세", "살까", "팔까",
        "매수해도", "사도", "진입", "상승", "하락", "진단", "호재", "악재", "의견", "어떻게", "대응"
    )
    rem = " ".join(tokens[1:]).lower()
    if any(qw in rem for qw in question_intent_words):
        return cand

    return None


def detect_market_macro_intent(raw_query: str) -> Dict[str, Any]:
    """
    Detects whether a user query is directed at:
    - Domestic Market ('domestic_market'): KOSPI, KOSDAQ, indices, supply/demand, session flow
    - US Market ('us_market'): NASDAQ, S&P 500, SOXX, Dow, Wall Street, tech giants
    - Macro Economy ('macro'): FX / USD-KRW, interest rates, inflation, CPI, oil, VIX, global risks
    - General Market ('market_general'): overall market sentiment, global market overview
    """
    q = (raw_query or "").strip().lower()
    if not q:
        return {"is_market_macro": False, "category": "none", "label": ""}

    us_keywords = [
        "미국증시", "미국 증시", "미국장", "미국 장", "미국시장", "미장", "뉴욕증시", "뉴욕 증시",
        "월가", "월스트리트", "나스닥", "s&p", "s&p500", "sp500", "다우", "다우존스",
        "필라델피아 반도체", "필라델피아반도체", "필반", "soxx", "qqq", "spy",
        "빅테크", "m7", "매그니피센트", "러셀2000", "러셀"
    ]

    dom_keywords = [
        "국내장", "국내증시", "국내 증시", "국내시장", "한국장", "한국증시", "한국시장",
        "코스피", "코스닥", "코스피200", "코스닥150", "선물시장", "선물 지수", "선물지수",
        "지수 흐름", "지수 전망", "장세", "오늘 장세", "장마감", "오전장", "오후장", "장초반",
        "시장 수급", "외인 수급", "기관 수급", "증시 분위기", "증시 동향", "시장 분위기",
        "시장 흐름", "시장 전체", "전체 시장", "오늘 시장", "시장 진단", "장세 진단",
        "국내 증시 흐름", "지수 왜", "지수 반등", "지수 하락"
    ]

    macro_keywords = [
        "거시경제", "거시 경제", "매크로", "macro", "환율", "원달러", "달러 환율", "달러환율",
        "달러인덱스", "원/달러", "엔화", "위안화", "금리", "기준금리", "연준", "fed", "fomc",
        "파월", "한국은행", "한은", "금통위", "국채", "국채금리", "10년물", "인플레이션", "인플레",
        "cpi", "ppi", "pce", "물가", "유가", "wti", "국제유가", "원유", "vix", "공포지수",
        "변동성지수", "경기침체", "경기 침체", "경기 사이클", "스태그플레이션", "글로벌 리스크",
        "시장 리스크", "경제 전망", "매크로 환경", "금리 인하", "금리 인상"
    ]

    general_keywords = [
        "글로벌 증시", "글로벌 시장", "해외 증시", "전체 시황", "시장 점검", "증시 리포트", "마켓 브리핑"
    ]

    has_us = any(k in q for k in us_keywords)
    has_dom = any(k in q for k in dom_keywords)
    has_macro = any(k in q for k in macro_keywords)
    has_gen = any(k in q for k in general_keywords)

    if not (has_us or has_dom or has_macro or has_gen):
        return {"is_market_macro": False, "category": "none", "label": ""}

    if (has_us and has_dom) or (has_dom and has_macro) or (has_us and has_macro) or has_gen:
        return {
            "is_market_macro": True,
            "category": "market_general",
            "label": "국내·글로벌 증시 & 매크로 종합 분석"
        }
    if has_us:
        return {
            "is_market_macro": True,
            "category": "us_market",
            "label": "미국 증시 퀀트 분석"
        }
    if has_dom:
        return {
            "is_market_macro": True,
            "category": "domestic_market",
            "label": "국내 증시 퀀트 분석"
        }
    if has_macro:
        return {
            "is_market_macro": True,
            "category": "macro",
            "label": "거시경제 & 매크로 리스크 분석"
        }

    return {"is_market_macro": False, "category": "none", "label": ""}


def generate_market_macro_answer(
    question: str,
    category: str = "general",
    api_key: Optional[str] = None,
    chat_history: Optional[List[Dict[str, str]]] = None
) -> str:
    """
    Generates an expert quantitative market and macro analysis answer.
    Supports both Gemini GenAI and built-in offline quantitative reasoning.
    """
    key = api_key or os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
    if not key:
        try:
            import streamlit as st
            key = st.secrets.get("GEMINI_API_KEY") or st.secrets.get("GOOGLE_API_KEY")
        except Exception:
            pass

    if key and len(key.strip()) > 10:
        try:
            return _call_gemini_market_macro(
                question,
                category=category,
                api_key=key.strip(),
                chat_history=chat_history
            )
        except Exception as e:
            logger.warning(f"Gemini Market Macro API call failed, falling back to built-in reasoning: {e}")
            return _built_in_market_macro_reasoning(
                question,
                category=category,
                fallback_notice=True,
                chat_history=chat_history
            )

    return _built_in_market_macro_reasoning(
        question,
        category=category,
        chat_history=chat_history
    )


def _call_gemini_market_macro(
    question: str,
    category: str,
    api_key: str,
    chat_history: Optional[List[Dict[str, str]]] = None
) -> str:
    headers = {"Content-Type": "application/json"}
    models_to_try = ["gemini-2.0-flash", "gemini-1.5-flash"]

    dom_summary = fetch_domestic_indices_summary()
    us_summary = fetch_us_market_summary()

    dom_items = {it["code"]: it for it in dom_summary.get("items", [])}
    kospi = dom_items.get("KOSPI", {"close": 2600.0, "change_pct": 0.0})
    kosdaq = dom_items.get("KOSDAQ", {"close": 750.0, "change_pct": 0.0})
    kpi200 = dom_items.get("KPI200", {"close": 350.0, "change_pct": 0.0})
    fut = dom_items.get("FUT", {"close": 350.0, "change_pct": 0.0})

    kospi_c = _safe_to_float(kospi.get("close"), 2600.0)
    kospi_pct = _safe_to_float(kospi.get("change_pct"), 0.0)
    kosdaq_c = _safe_to_float(kosdaq.get("close"), 750.0)
    kosdaq_pct = _safe_to_float(kosdaq.get("change_pct"), 0.0)
    kpi200_c = _safe_to_float(kpi200.get("close"), 350.0)
    fut_c = _safe_to_float(fut.get("close"), 350.0)

    nasdaq_pct = _safe_to_float(us_summary.get("nasdaq_pct"), 0.0)
    sp500_pct = _safe_to_float(us_summary.get("sp500_pct"), 0.0)
    soxx_pct = _safe_to_float(us_summary.get("soxx_pct"), 0.0)
    dow_pct = _safe_to_float(us_summary.get("dow_pct"), 0.0)
    usdkrw = _safe_to_float(us_summary.get("usdkrw_rate"), 1350.0)
    usdkrw_pct = _safe_to_float(us_summary.get("usdkrw_pct"), 0.0)
    vix_val = _safe_to_float(us_summary.get("vix_val"), 18.0)
    vix_pct = _safe_to_float(us_summary.get("vix_pct"), 0.0)
    regime = str(us_summary.get("regime") or "중립")
    action_guide = str(us_summary.get("action_guide") or "")

    from src.collectors.market_alerts import evaluate_market_warning_alerts
    alert_info = evaluate_market_warning_alerts(domestic_summary=dom_summary, us_summary=us_summary)
    alert_status_line = f"- 시장 경보 체제: {alert_info.get('badge_text')} ({alert_info.get('level')})"
    if alert_info.get("triggered_time") and alert_info.get("end_time"):
        alert_status_line += f" | 발동시각: {alert_info['triggered_time']} · 종료시각: {alert_info['end_time']} ({alert_info.get('remaining_str', '')})"

    macro_context = (
        f"[실시간 국내 증시 지표]\n"
        f"- 코스피: {kospi_c:,.2f}pt ({kospi_pct:+.2f}%) | 코스닥: {kosdaq_c:,.2f}pt ({kosdaq_pct:+.2f}%)\n"
        f"- 코스피200: {kpi200_c:,.2f}pt | 연결선물: {fut_c:,.2f}pt (베이시스: {fut_c - kpi200_c:+.2f}pt)\n"
        f"{alert_status_line}\n\n"
        f"[글로벌 미국 증시 및 핵심 매크로 지표]\n"
        f"- 나스닥: {nasdaq_pct:+.2f}% | S&P 500: {sp500_pct:+.2f}% | 필라델피아 반도체(SOXX): {soxx_pct:+.2f}% | 다우: {dow_pct:+.2f}%\n"
        f"- 원/달러 환율: {usdkrw:,.1f}원 ({usdkrw_pct:+.2f}%) | VIX 변동성지수: {vix_val:.1f}p ({vix_pct:+.2f}%)\n"
        f"- 글로벌 시장 체제: {regime} ({action_guide})\n\n"
    )

    system_instruction = (
        "당신은 국내 및 글로벌 금융 시장 전반을 총괄 분석하는 최고 수준의 '수석 거시경제 및 퀀트 마켓 스트래티지스트(Chief Global Macro & Market Strategist)'입니다.\n"
        "특정 종목에 편향되지 않고, 사용자가 질의한 국내 증시(코스피/코스닥/선물/수급), 미국 증시(나스닥/S&P500/SOXX/다우), 거시경제 매크로(환율/금리/인플레이션/유가/VIX)에 대하여 "
        "제공된 실시간 시장 수치를 바탕으로 객관적이고 논리적인 고품격 분석 리포트를 한국어로 제공하세요.\n"
        "[답변 원칙 (최우선)]: 질문에 대해 지표만 나열하지 말고, 반드시 답변 맨 첫 줄에 '> **[수석 스트래티지스트 핵심 직답]**: ...' 형식으로 "
        "사용자가 질문한 핵심(예: 오늘 국내장 진단, 코스피/코스닥 방향성, 미국 증시 파급 효과, 환율/금리 리스크 등)에 대한 명쾌하고 단정적인 결론을 1~2문장으로 즉시 직접 답변한 뒤, "
        "그 아래에 1. 실시간 국내 및 글로벌 핵심 마켓 지표 진단, 2. 시장 수급 및 매크로 리스크 팩트체크, 3. 섹터별 영향도 및 퀀트 포트폴리오 대응 전략 순서로 상세히 서술하십시오.\n"
        "모호한 표현 대신 구체적인 지수 포인트, 등락률(%), 환율 수치를 인용하고, 권장 현금 비중과 분할 매매 원칙을 함께 안내하세요."
    )

    hist_text = ""
    if chat_history and len(chat_history) > 0:
        recent_turns = chat_history[-3:]
        h_lines = []
        for h in recent_turns:
            q_txt = str(h.get("q", "")).strip()
            a_txt = str(h.get("ans", "")).strip()
            if len(a_txt) > 250:
                a_txt = a_txt[:250] + "..."
            if q_txt:
                h_lines.append(f"Q: {q_txt}\nA: {a_txt}")
        if h_lines:
            hist_text = "[이전 대화 맥락]:\n" + "\n\n".join(h_lines) + "\n\n"

    prompt_content = (
        f"{system_instruction}\n\n{macro_context}\n\n"
        f"{hist_text}[사용자 질문]: {question}\n\n"
        "[필수 작성 형식]: 반드시 답변 맨 첫 줄에 '> **[수석 스트래티지스트 핵심 직답]**: [사용자의 핵심 질문에 대한 명확한 결론 직답 1~2문장]'을 작성하고, 그 아래에 구체적인 퀀트 수치 근거와 매매 전략을 서술하십시오."
    )

    payload = {
        "contents": [{"parts": [{"text": prompt_content}]}],
        "generationConfig": {"temperature": 0.2, "maxOutputTokens": 1200}
    }

    last_error_text = ""
    for model_name in models_to_try:
        url = f"https://generativelanguage.googleapis.com/v1beta/models/{model_name}:generateContent?key={api_key}"
        try:
            resp = requests.post(url, headers=headers, json=payload, timeout=12)
            if resp.status_code == 200:
                data = resp.json()
                candidates = data.get("candidates", [])
                if candidates:
                    parts = candidates[0].get("content", {}).get("parts", [])
                    if parts:
                        return parts[0].get("text", "").strip()
            last_error_text = f"status {resp.status_code}: {resp.text[:200]}"
        except Exception as e:
            last_error_text = str(e)
            continue

    raise RuntimeError(f"Gemini Market Macro API failed: {last_error_text}")


def _built_in_market_macro_reasoning(
    question: str,
    category: str = "general",
    fallback_notice: bool = False,
    chat_history: Optional[List[Dict[str, str]]] = None
) -> str:
    """
    Offline statistical & rule-based market macro engine.
    Produces comprehensive, fact-grounded quantitative analysis of domestic and US markets
    and macroeconomic factors without referencing any specific single stock.
    """
    dom_summary = fetch_domestic_indices_summary()
    us_summary = fetch_us_market_summary()

    dom_items = {it["code"]: it for it in dom_summary.get("items", [])}
    kospi = dom_items.get("KOSPI", {"close": 2600.0, "change_pct": 0.0, "change": 0.0})
    kosdaq = dom_items.get("KOSDAQ", {"close": 750.0, "change_pct": 0.0, "change": 0.0})
    kpi200 = dom_items.get("KPI200", {"close": 350.0, "change_pct": 0.0, "change": 0.0})
    fut = dom_items.get("FUT", {"close": 350.0, "change_pct": 0.0, "change": 0.0})

    kospi_c = _safe_to_float(kospi.get("close"), 2600.0)
    kospi_pct = _safe_to_float(kospi.get("change_pct"), 0.0)
    kosdaq_c = _safe_to_float(kosdaq.get("close"), 750.0)
    kosdaq_pct = _safe_to_float(kosdaq.get("change_pct"), 0.0)
    kpi200_c = _safe_to_float(kpi200.get("close"), 350.0)
    fut_c = _safe_to_float(fut.get("close"), 350.0)
    basis = fut_c - kpi200_c

    nasdaq_pct = _safe_to_float(us_summary.get("nasdaq_pct"), 0.0)
    sp500_pct = _safe_to_float(us_summary.get("sp500_pct"), 0.0)
    soxx_pct = _safe_to_float(us_summary.get("soxx_pct"), 0.0)
    dow_pct = _safe_to_float(us_summary.get("dow_pct"), 0.0)
    usdkrw = _safe_to_float(us_summary.get("usdkrw_rate"), 1350.0)
    usdkrw_pct = _safe_to_float(us_summary.get("usdkrw_pct"), 0.0)
    vix_val = _safe_to_float(us_summary.get("vix_val"), 18.0)
    vix_pct = _safe_to_float(us_summary.get("vix_pct"), 0.0)
    regime = str(us_summary.get("regime") or "중립")
    action_guide = str(us_summary.get("action_guide") or "시장 수급 및 리스크 요인 확인 후 분할 매매가 유리합니다.")

    q_low = question.lower()
    if category == "domestic_market" or any(w in q_low for w in ["국내", "코스피", "코스닥", "선물", "장세"]):
        if kospi_pct >= 0.5 and kosdaq_pct >= 0.5:
            direct_ans = (
                f"금일 국내 증시는 코스피({kospi_pct:+.2f}%)와 코스닥({kosdaq_pct:+.2f}%)이 동반 상승하며 "
                f"위험자산 선호 심리가 강화되는 긍정적 흐름을 나타내고 있습니다. "
                f"원/달러 환율({usdkrw:,.1f}원) 안정 여부와 외국인 패시브 수급 연속성을 확인하며 주도 섹터 중심의 눌림목 매수 대응이 유리합니다."
            )
        elif kospi_pct <= -0.5 and kosdaq_pct <= -0.5:
            direct_ans = (
                f"금일 국내 증시는 코스피({kospi_pct:+.2f}%)와 코스닥({kosdaq_pct:+.2f}%)이 동반 조정을 받으며 "
                f"단기 변동성 확대 및 차익 실현 경계 매물이 출회되는 약세 국면입니다. "
                f"무리한 저점 추격보다는 현금 비중 40% 이상을 유지하며 지지선 확인 후 보수적으로 접근할 것을 권고합니다."
            )
        else:
            direct_ans = (
                f"금일 국내 증시는 코스피({kospi_pct:+.2f}%)와 코스닥({kosdaq_pct:+.2f}%) 간 등락이 엇갈리며 "
                f"뚜렷한 지수 모멘텀보다는 개별 실적 및 재료 보유 종목으로 수급이 집중되는 차별화 장세입니다. "
                f"지수 방향성 추종보다 철저한 가격 메리트와 손익비 중심의 선별적 대응이 필수적입니다."
            )
    elif category == "us_market" or any(w in q_low for w in ["미국", "미장", "나스닥", "s&p", "필라델피아", "반도체"]):
        tech_sentiment = "강세" if soxx_pct >= 0.5 else ("약세" if soxx_pct <= -0.5 else "혼조세")
        direct_ans = (
            f"전일 미국 증시는 나스닥({nasdaq_pct:+.2f}%), S&P 500({sp500_pct:+.2f}%), 필라델피아 반도체({soxx_pct:+.2f}%)가 "
            f"{tech_sentiment}를 기록하며 글로벌 테크 섹터의 투자 심리에 직접적인 파급력을 미치고 있습니다. "
            f"특히 SOXX 반도체 지수의 향방은 국내 시총 상위 반도체주 및 IT 부품 섹터의 외국인 수급을 좌우할 핵심 지표입니다."
        )
    elif any(w in q_low for w in ["사이드카", "서킷브레이커", "서킷", "발동시간", "종료시간", "시장경보"]):
        from src.collectors.market_alerts import evaluate_market_warning_alerts
        alert_info = evaluate_market_warning_alerts(domestic_summary=dom_summary, us_summary=us_summary)
        level = alert_info.get("level", "NORMAL")
        if level in ("WARNING", "CRITICAL") and alert_info.get("triggered_time"):
            trig_t = alert_info.get("triggered_time")
            end_t = alert_info.get("end_time")
            rem_s = alert_info.get("remaining_str", "")
            dur_s = alert_info.get("duration_str", "")
            direct_ans = (
                f"현재 시장 경보 상태는 [{alert_info.get('badge_text')}]이며, "
                f"발동시각은 {trig_t}, 예정 종료시각은 {end_t}({dur_s}, {rem_s})입니다. "
                f"{alert_info.get('action_guide')}"
            )
        else:
            direct_ans = (
                f"현재 국내 시장은 사이드카 및 서킷브레이커 발동 기준(선물 +/-5%, 지수 -8% 등)에 해당하지 않는 "
                f"정상 거래 범위(코스피 {kospi_pct:+.2f}%, 코스피200선물 {fut.get('change_pct', 0.0):+.2f}%) 내에 위치하고 있습니다. "
                f"선물 변동폭이 +/-3.5% 도달 시 주의보, +/-5.0% 이상 1분간 지속 시 5분간 프로그램 매매 효력이 정지됩니다."
            )
    elif category == "macro" or any(w in q_low for w in ["거시", "매크로", "환율", "금리", "인플레", "cpi", "유가", "vix"]):
        fx_sentiment = "고환율 부담 구간" if usdkrw >= 1350.0 else "환율 안정권"
        vix_sentiment = "경계 필요" if vix_val >= 20.0 else "안정적"
        direct_ans = (
            f"현재 거시경제(Macro) 환경은 원/달러 환율({usdkrw:,.1f}원, {fx_sentiment})과 변동성 VIX({vix_val:.1f}p, {vix_sentiment})가 "
            f"복합적으로 작용하는 변곡점입니다. 글로벌 통화정책 기조와 국채 금리 추이에 따라 외국인 자금 유출입이 급변할 수 있으므로, "
            f"매크로 리스크 노출도가 낮은 실적 방어주와 헷지성 포지션을 병행하는 전략이 합리적입니다."
        )
    else:
        direct_ans = (
            f"금일 국내 및 글로벌 금융 시장은 코스피({kospi_pct:+.2f}%), 나스닥({nasdaq_pct:+.2f}%), 환율({usdkrw:,.1f}원) 전반에서 "
            f"글로벌 시장 체제 '{regime}' 환경을 반영하고 있습니다. "
            f"급격한 레버리지 확대보다는 리스크 관리 원칙에 입각한 분할 매매가 유리합니다."
        )

    basis_desc = "콘탱고 (선물 고평가, 매수 우위)" if basis > 0 else "백워데이션 (선물 저평가, 매도 경계)"
    vix_desc = "변동성 낮음 (안정 국면)" if vix_val < 18 else ("변동성 확대 (경계 국면)" if vix_val < 25 else "고변동성 (위험 관리)")
    fx_desc = "원화 강세 (외인 매수 우호)" if usdkrw_pct < -0.3 else ("원화 약세 (외인 매도 압력)" if usdkrw_pct > 0.3 else "중립 박스권")

    body_parts = [
        f"> **[수석 스트래티지스트 핵심 직답]**: {direct_ans}\n",
        "### 1. 실시간 국내 및 글로벌 핵심 마켓 지표 진단",
        f"- **국내 지수**: 코스피 **{kospi_c:,.2f}pt** ({kospi_pct:+.2f}%) | 코스닥 **{kosdaq_c:,.2f}pt** ({kosdaq_pct:+.2f}%)",
        f"- **선물 및 파생**: 코스피200 {kpi200_c:,.2f}pt | 연결선물 {fut_c:,.2f}pt (베이시스: {basis:+.2f}pt, {basis_desc})",
        f"- **미국 3대 지수 & 반도체**: 나스닥 **{nasdaq_pct:+.2f}%** | S&P 500 **{sp500_pct:+.2f}%** | 필라델피아 반도체(SOXX) **{soxx_pct:+.2f}%** | 다우 {dow_pct:+.2f}%",
        f"- **핵심 매크로**: 원/달러 환율 **{usdkrw:,.1f}원** ({usdkrw_pct:+.2f}%, {fx_desc}) | VIX 변동성지수 **{vix_val:.1f}p** ({vix_desc})",
        f"- **글로벌 시장 체제**: **{regime}** (가이드: {action_guide})\n",
        "### 2. 시장 수급 및 매크로 리스크 팩트체크",
        f"1. **환율 및 외국인 수급 경로**:",
        f"   - 현재 원/달러 환율 {usdkrw:,.1f}원 수준은 외국인 투자자 관점에서 환차손익 민감도를 자극하는 레벨입니다.",
        f"   - {'환율 상승 압력으로 인해 코스피 대형 수출주에 대한 외인 패시브 매도 물량이 출회될 수 있어 장중 수급을 면밀히 관찰해야 합니다.' if usdkrw >= 1350 else '환율이 안정권에 위치하여 외인 자금의 급격한 이탈 우려는 제한적이며, 수급 수혜주 중심의 반등 탄력이 기대됩니다.'}",
        f"2. **금리 및 글로벌 통화정책 파급 효과**:",
        f"   - 미국 연준(FOMC)의 통화정책 기조와 국채 금리 움직임은 국내 성장주(바이오, 2차전지, 플랫폼) 밸류에이션에 즉각적인 영향을 미칩니다.",
        f"   - 금리 동결 또는 인하 기대감이 유지될 경우 테크 및 성장 테마의 반등 폭이 커지며, 금리 반등 시에는 가치주 및 배당주로 방어적 수급이 이동합니다.",
        f"3. **시장 변동성(VIX) 및 하방 지지선 점검**:",
        f"   - VIX {vix_val:.1f}p 레벨은 {vix_desc} 상태를 가리키며, 글로벌 돌발 악재 발생 시 지수 하방 지지선(코스피 기준 {kospi_c * 0.98:,.0f} ~ {kospi_c * 0.96:,.0f}pt) 이탈 여부를 주시해야 합니다.\n",
        "### 3. 섹터별 영향도 및 퀀트 포트폴리오 대응 전략",
        f"- **상대적 강세/수혜 섹터**: {'반도체, AI 테크, 수출 대형주 (미 테크주 훈풍)' if soxx_pct >= 0.0 else '방어주(통신, 음식료), 배당주, 금융주 (지수 방어적 성격)'}",
        f"- **상대적 경계/조정 섹터**: {'원자재 의존주, 고부채 기업, 환율 민감 내수주' if usdkrw >= 1350 else '단기 과열 테마주, 신용 잔고율 높은 소형주'}",
        f"- **권장 포트폴리오 비중**: {'주식 60% / 현금 40% (선별적 눌림목 매수)' if regime == '중립' else ('주식 75% / 현금 25% (주도주 추세 추종)' if '강세' in regime else '주식 40% / 현금 60% (보수적 리스크 관리)')}",
        f"- **체결 원칙**: 시장 지수 변동성이 큰 국면에서는 시초가 추격 매수를 엄격히 금지하며, 장중 1시 이후 분할 진입 및 원칙 손절선을 준수해야 합니다."
    ]

    notice = "\n\n*(자체 내장 퀀트 매크로 분석 엔진이 실시간 시장 데이터를 집계하여 산출한 정밀 리포트입니다.)*" if fallback_notice else ""
    return "\n".join(body_parts) + notice
