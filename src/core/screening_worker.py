"""
Background Screening Worker & Task Manager
Runs heavy screening tasks asynchronously in background threads so users
can navigate between tabs and menus without cancelling or interrupting the process.
"""

import time
import threading
from datetime import datetime, timedelta
from typing import Dict, Any, Optional, List
from concurrent.futures import ThreadPoolExecutor, as_completed
import pandas as pd

from src.database.models import get_db_connection, get_now_kst
from src.collectors.market_data import (
    fetch_ohlcv,
    get_benchmark_ohlcv,
    fetch_realtime_quotes,
    get_last_completed_trading_day
)
from src.collectors.krx_universe import get_universe
from src.core.scoring import evaluate_stock_latest
from src.core.high_winrate_strategies import (
    evaluate_sniper_candidate,
    evaluate_closing_bet_candidate,
    evaluate_5pct_surge_candidate,
    evaluate_intraday_daytrade_candidate
)


def _fetch_live_market_leader_codes(target_market: str = "ALL") -> List[str]:
    """
    Fetches real-time market cap, volume, and surging leader codes from Naver mobile API in parallel.
    Returns prioritized list of 6-digit stock codes (surging stocks first, then trading value leaders).
    """
    import requests
    from concurrent.futures import ThreadPoolExecutor

    headers = {"User-Agent": "Mozilla/5.0"}
    session = requests.Session()
    markets = ["KOSPI", "KOSDAQ"] if target_market == "ALL" else [target_market.upper()]

    def _fetch_one_market(m: str) -> List[Dict[str, Any]]:
        url = f"https://m.stock.naver.com/api/stocks/marketValue/{m}?page=1&pageSize=100"
        try:
            r = session.get(url, headers=headers, verify=False, timeout=4)
            if r.status_code == 200:
                return r.json().get("stocks", [])
        except Exception:
            pass
        return []

    with ThreadPoolExecutor(max_workers=min(2, len(markets) or 1)) as executor:
        market_stocks = list(executor.map(_fetch_one_market, markets))

    surging_codes: List[str] = []
    high_val_codes: List[str] = []

    for stocks in market_stocks:
        if not stocks:
            continue
        # 1. Stocks surging today (fluctuationsRatio >= 2.5%) sorted by trading value
        surging = [
            s for s in stocks
            if float(str(s.get("fluctuationsRatio", 0)).replace("%", "").replace(",", "") or 0) >= 2.5
        ]
        surging.sort(key=lambda s: float(s.get("accumulatedTradingValueRaw", 0) or 0), reverse=True)
        surging_codes.extend([s["itemCode"].zfill(6) for s in surging if s.get("itemCode")])

        # 2. Top trading value leaders
        by_val = sorted(stocks, key=lambda s: float(s.get("accumulatedTradingValueRaw", 0) or 0), reverse=True)
        high_val_codes.extend([s["itemCode"].zfill(6) for s in by_val if s.get("itemCode")])

    return list(dict.fromkeys(surging_codes + high_val_codes))


def _evaluate_single_stock(
    row: pd.Series,
    realtime_quotes: Dict[str, Dict[str, Any]],
    now_dt: datetime,
    today_str: str,
    last_trading_day_str: str,
    sc_mode: str,
    bm_ohlcv: Optional[pd.DataFrame],
    params: Dict[str, Any]
) -> Optional[Dict[str, Any]]:
    code = row["code"]
    name = row["name"]
    sector = row.get("sector", "기타")
    market = row["market"]

    min_daytrade_val_b = params.get("min_daytrade_val_b", 200)
    intraday_gain_range = params.get("intraday_gain_range", (3.0, 8.5))
    min_daytrade_vol_ratio = params.get("min_daytrade_vol_ratio", 0.6)

    min_val_krw_b = params.get("min_val_krw_b", 100)
    max_disparity_val = params.get("max_disparity_val", 103.5)
    ignore_market_filter = params.get("ignore_market_filter", False)

    min_today_val_b = params.get("min_today_val_b", 200)
    min_day_ret_val = params.get("min_day_ret_val", 3.0)

    min_surge_val_b = params.get("min_surge_val_b", 300)
    min_day_surge_pct = params.get("min_day_surge_pct", 10.0)

    min_screener_score = params.get("min_screener_score", 70.0)
    min_vol_surge = params.get("min_vol_surge", 1.5)
    require_ma_align = params.get("require_ma_align", True)

    try:
        # High-Speed Pre-filter using real-time quotes: bypass disk OHLCV loads for non-candidates
        rt_info = realtime_quotes.get(code)
        if rt_info:
            rt_val_raw = rt_info.get("trading_val_raw", 0.0)
            rt_chg_pct = rt_info.get("change_pct", 0.0)

            if "당일 단타" in sc_mode:
                min_val_threshold = float(min_daytrade_val_b) * 100_000_000.0 * 0.90
                if rt_val_raw > 0 and rt_val_raw < min_val_threshold:
                    return None
                if rt_chg_pct < (intraday_gain_range[0] - 1.5):
                    return None
            elif "5% 급등" in sc_mode or "5% 돌파" in sc_mode:
                min_val_threshold = float(min_surge_val_b) * 100_000_000.0 * 0.90
                if rt_val_raw > 0 and rt_val_raw < min_val_threshold:
                    return None
                if rt_chg_pct < (float(min_day_surge_pct) - 2.0):
                    return None
            elif "종가배팅" in sc_mode:
                min_val_threshold = float(min_today_val_b) * 100_000_000.0 * 0.90
                if rt_val_raw > 0 and rt_val_raw < min_val_threshold:
                    return None
                if rt_chg_pct < (float(min_day_ret_val) - 1.5):
                    return None
            elif "스나이퍼" in sc_mode or "장전 시초가" in sc_mode or "08:00~09:00" in sc_mode:
                if rt_val_raw > 0 and rt_val_raw < 1_000_000_000.0 and rt_chg_pct < -5.0:
                    return None

        # Synchronize real-time live quote if available
        if rt_info and rt_info.get("close", 0) > 0 and (now_dt.weekday() < 5 and now_dt.hour >= 9):
            df_stock = fetch_ohlcv(code, end=last_trading_day_str)
            close_p = float(rt_info["close"])
            open_p = float(rt_info["open"])
            high_p = float(rt_info["high"])
            low_p = float(rt_info["low"])
            prev_p = float(rt_info["prev_close"])
            chg_pct = float(rt_info["change_pct"])
            open_pct = float(rt_info["open_pct"])
            high_pct = float(rt_info["high_pct"])
            low_pct = float(rt_info["low_pct"])
            vol_p = int(rt_info["volume"])

            # Inject live quote into latest bar of df_stock for technical indicators
            today_dt = pd.to_datetime(rt_info.get("date") or today_str)
            df_live_bar = pd.DataFrame([{
                "Open": open_p, "High": high_p, "Low": low_p, "Close": close_p, "Volume": vol_p, "Change": chg_pct / 100.0
            }], index=[today_dt])
            df_stock = pd.concat([df_stock, df_live_bar])
        else:
            df_stock = fetch_ohlcv(code, end=last_trading_day_str)
            min_bars = 1 if ("당일 단타" in sc_mode or "5% 급등" in sc_mode or "종가배팅" in sc_mode) else 20
            if df_stock.empty or len(df_stock) < min_bars:
                return None

            close_p = float(df_stock["Close"].iloc[-1])
            open_p = float(df_stock["Open"].iloc[-1]) if "Open" in df_stock.columns else close_p
            high_p = float(df_stock["High"].iloc[-1]) if "High" in df_stock.columns else close_p
            low_p = float(df_stock["Low"].iloc[-1]) if "Low" in df_stock.columns else close_p
            prev_p = float(df_stock["Close"].iloc[-2]) if len(df_stock) >= 2 else close_p

            chg_pct = ((close_p - prev_p) / prev_p) * 100.0 if prev_p > 0 else 0.0
            open_pct = ((open_p - prev_p) / prev_p) * 100.0 if prev_p > 0 else 0.0
            high_pct = ((high_p - prev_p) / prev_p) * 100.0 if prev_p > 0 else 0.0
            low_pct = ((low_p - prev_p) / prev_p) * 100.0 if prev_p > 0 else 0.0
            vol_p = int(df_stock["Volume"].iloc[-1])

        min_bars = 1 if ("당일 단타" in sc_mode or "5% 급등" in sc_mode or "종가배팅" in sc_mode) else 20
        if df_stock.empty or len(df_stock) < min_bars:
            return None

        try:
            last_bar_dt = df_stock.index[-1]
            last_bar_str = last_bar_dt.strftime("%Y-%m-%d") if hasattr(last_bar_dt, "strftime") else str(last_bar_dt)[:10]
            is_today_intraday = (last_bar_str == now_dt.strftime("%Y-%m-%d") and now_dt.hour < 15)
        except Exception:
            is_today_intraday = False
        df_snp_eval = df_stock.iloc[:-1] if is_today_intraday and len(df_stock) >= 31 else df_stock

        if "장전 시초가" in sc_mode or "08:00~09:00" in sc_mode:
            bm_for_eval = None if ignore_market_filter else bm_ohlcv

            # 트랙 1: 20일선 눌림목 첫 반등 (Pullback Rebound)
            snp = evaluate_sniper_candidate(df_snp_eval, df_benchmark=bm_for_eval, min_daily_val_krw=min_val_krw_b * 100_000_000)
            is_pullback_cand = False
            if snp and snp["metrics"]["disparity"] <= max_disparity_val:
                is_pullback_cand = True

            # 트랙 2: 전일 대규모 거래대금 폭발 주도 대장주 (Top Trading Value Breakout)
            df_eval_lead = df_snp_eval
            is_lead_gap_cand = False
            lead_metrics = {}
            if len(df_eval_lead) >= 20:
                c_last = float(df_eval_lead["Close"].iloc[-1])
                c_prev = float(df_eval_lead["Close"].iloc[-2]) if len(df_eval_lead) >= 2 else c_last
                h_last = float(df_eval_lead["High"].iloc[-1])
                l_last = float(df_eval_lead["Low"].iloc[-1])
                v_last = float(df_eval_lead["Volume"].iloc[-1])

                yesterday_val_krw = c_last * v_last
                yesterday_val_b = yesterday_val_krw / 100_000_000.0

                ma5 = float(df_eval_lead["Close"].tail(5).mean())
                ma20 = float(df_eval_lead["Close"].tail(20).mean())
                vol20_avg = float(df_eval_lead["Volume"].tail(20).mean())
                vol_ratio = (v_last / vol20_avg) if vol20_avg > 0 else 1.0

                day_ret_pct = ((c_last - c_prev) / c_prev) * 100.0 if c_prev > 0 else 0.0
                high_close_ratio = ((c_last - l_last) / (h_last - l_last) * 100.0) if (h_last - l_last) > 0 else 100.0

                lead_val_threshold = max(300.0, float(min_val_krw_b) * 3.0)
                if yesterday_val_b >= lead_val_threshold and day_ret_pct >= 2.5 and high_close_ratio >= 65.0 and c_last >= ma5 and ma5 >= ma20:
                    is_lead_gap_cand = True
                    lead_metrics = {
                        "val_b": yesterday_val_b,
                        "ret_pct": day_ret_pct,
                        "hc_ratio": high_close_ratio,
                        "vol_ratio": vol_ratio,
                        "ma5": ma5,
                        "ma20": ma20
                    }

            if not is_pullback_cand and not is_lead_gap_cand:
                return None

            # Dual-track priority
            if is_lead_gap_cand and (not is_pullback_cand or lead_metrics.get("val_b", 0) >= 1200.0):
                val_b = lead_metrics["val_b"]
                ret_p = lead_metrics["ret_pct"]
                hc_r = lead_metrics["hc_ratio"]
                vr = lead_metrics["vol_ratio"]
                pm_ctx = params.get("premarket_ctx") or {}
                pm_regime = pm_ctx.get("regime", "중립")
                pm_gap_guide = pm_ctx.get("gap_guide", "적정 갭 +1.5%~+4.5% 확인 후 진입")
                pm_risk = pm_ctx.get("gap_risk", "보통 (중립)")
                pm_mod = pm_ctx.get("score_mod", 0.0)
                lead_score = min(99.0, max(60.0, min(98.0, 78.0 + min(12.0, val_b / 250.0) + min(8.0, ret_p / 2.0)) + pm_mod))

                reason_summary = f"[주도 갭상승] 전일대금 {val_b:,.0f}억 폭발 | 종가 +{ret_p:.1f}% 고가마감 | 거래량 {vr:.1f}배"
                reason_core = "전일 대규모 주도 거래대금을 분출하며 일봉 최고가권으로 마감한 시장 1등 대장주입니다. 장전 08:00~08:50 동시호가 예상체결가 점검 후 09:00 시초가 적정 갭상승(+1.5%~+4.5%) 출발 시 장초반 강력한 2차 슈팅 파동을 타겟합니다."
                reason_criteria = [
                    f"전일 거래대금 {val_b:,.0f}억 원 폭발 (기준 {lead_val_threshold:,.0f}억 원 이상 시장 주도 수급)",
                    f"전일 종가상승률 +{ret_p:.1f}% 장대양봉 마감 (강력한 단기 상승 탄력)",
                    f"고점 지지율 {hc_r:.1f}% (윗꼬리 적은 마루보즈형 종가 관리 매수세 확인)",
                    "이평선 정배열 (종가 > 5일선 > 20일선 단기 정배열 유지)",
                    "장전 시초가 체결 원칙: 예상 갭 +1.5% ~ +4.5% 적정 갭 확인 후 진입 (+5% 초과 과열 시 진입 금지)",
                    f"글로벌 프리마켓 분석: 미증시/매크로 [{pm_regime}] (나스닥 {pm_ctx.get('nasdaq_pct', 0.0):+.2f}%, SOXX {pm_ctx.get('soxx_pct', 0.0):+.2f}%) - {pm_gap_guide}",
                    "목표 익절 +4.0% / 원칙 손절 -2.5% 자동 스탑로스"
                ]

                return {
                    "code": code,
                    "name": name,
                    "market": market,
                    "sector": sector,
                    "close": close_p,
                    "change_pct": chg_pct,
                    "open_pct": open_pct,
                    "high_pct": high_pct,
                    "low_pct": low_pct,
                    "open": open_p,
                    "high": high_p,
                    "low": low_p,
                    "volume": vol_p,
                    "timing_label": "장전 (주도 갭상승)",
                    "strategy_mode": "PRE_MARKET_OPEN",
                    "strategy_tag": "장전시초가 (주도 갭상승)",
                    "premarket_regime": pm_regime,
                    "premarket_gap_guide": pm_gap_guide,
                    "premarket_risk": pm_risk,
                    "score": lead_score,
                    "label": "강력 주도주" if lead_score >= 88 else "우량 주도주",
                    "tp_pct": 4.0,
                    "sl_pct": 2.5,
                    "vol_ratio": vr,
                    "rsi": 68.0,
                    "ma_status": "5·20일선 정배열 대장주",
                    "rule_note": "08:00~08:50 분석 완료 -> 09:00 적정 갭상승(+1.5%~+4.5%) 시초가 진입 -> 슈팅 시 +4.0% 분할 익절",
                    "reason_summary": reason_summary,
                    "reason_core": reason_core,
                    "reason_criteria": reason_criteria,
                    "breakdown": {
                        "rsi_val": 68.0,
                        "ma_status": "주도 대장주 장대양봉",
                        "vol_ratio": vr,
                        "bb_pct": hc_r / 100.0
                    }
                }
            else:
                m_snp = snp["metrics"]
                disp_val = m_snp.get("disparity", 100.0)
                avg_val_b = m_snp.get("avg_trading_val_억", 0)
                rsi_val = m_snp.get("rsi", 50.0)
                vol_r_val = m_snp.get("vol_ratio", 1.0)

                reason_summary = f"[눌림 반등] 20일선 이격도 {disp_val:.1f}% 지지 | 20일평균 대금 {avg_val_b:,.0f}억 | RSI {rsi_val:.1f}"
                reason_core = "전일 확정 데이터 기준 20일 이동평균선 눌림목 첫 지지 반등이 검증된 풍부한 유동성의 우량주입니다. 08:00~08:50 후보 압축 후 09:00 시초가 보합권(-1.5%~+1.5%) 진입 시 안정적인 승률과 손익비를 목표로 합니다."
                pm_ctx = params.get("premarket_ctx") or {}
                pm_regime = pm_ctx.get("regime", "중립")
                pm_gap_guide = pm_ctx.get("gap_guide", "보합권 -1.5%~+1.5% 진입")
                pm_risk = pm_ctx.get("gap_risk", "보통 (중립)")

                reason_criteria = [
                    "KODEX 200 지수 추세 필터 통과 (시스템적 하락장 차단)" if not ignore_market_filter else "개별 종목 기술적 지표 단독 검증",
                    f"20일 일평균 거래대금 {avg_val_b:,.0f}억 원 (기준 {min_val_krw_b}억 원 이상 충족)",
                    f"20일선 이격도 {disp_val:.1f}% (설정 기준 98.0% ~ {max_disparity_val:.1f}% 내 20일선 지지 반등)",
                    f"RSI(14) {rsi_val:.1f} (에너지 응축 건전한 반등 구간)",
                    "장전 시초가 체결 원칙: 시초가 갭 -1.5% ~ +1.5% 적정 보합권 진입 (+1.5% 초과 갭상승 시 진입 금지)",
                    f"글로벌 프리마켓 점검: 미증시/매크로 [{pm_regime}] 상태 연동 - 20일선 눌림목 첫 반등 지지력 확인",
                    "목표 익절 +2.5% / 원칙 손절 -2.0% 자동 스탑로스"
                ]

                return {
                    "code": code,
                    "name": name,
                    "market": market,
                    "sector": sector,
                    "close": close_p,
                    "change_pct": chg_pct,
                    "open_pct": open_pct,
                    "high_pct": high_pct,
                    "low_pct": low_pct,
                    "open": open_p,
                    "high": high_p,
                    "low": low_p,
                    "volume": vol_p,
                    "timing_label": "장전 (눌림 반등)",
                    "strategy_mode": "PRE_MARKET_OPEN",
                    "strategy_tag": "장전시초가 (눌림 반등)",
                    "premarket_regime": pm_regime,
                    "premarket_gap_guide": pm_gap_guide,
                    "premarket_risk": pm_risk,
                    "score": snp["score"],
                    "label": snp["label"],
                    "tp_pct": 2.5,
                    "sl_pct": 2.0,
                    "vol_ratio": vol_r_val,
                    "rsi": rsi_val,
                    "ma_status": m_snp.get("ma_status", "20일선 지지"),
                    "rule_note": "08:00~08:50 후보 압축 -> 09:00 시초가 갭(-1.5%~+1.5%) 진입 -> 익절 +2.5% / 손절 -2.0%",
                    "reason_summary": reason_summary,
                    "reason_core": reason_core,
                    "reason_criteria": reason_criteria,
                    "breakdown": {
                        "rsi_val": rsi_val,
                        "ma_status": "20일선 지지 첫반등",
                        "vol_ratio": vol_r_val,
                        "bb_pct": 0.5
                    }
                }
        elif "당일 단타" in sc_mode:
            dtrade = evaluate_intraday_daytrade_candidate(
                df_stock,
                df_benchmark=bm_ohlcv,
                min_today_val_krw=min_daytrade_val_b * 100_000_000,
                min_intraday_gain=intraday_gain_range[0],
                max_intraday_gain=intraday_gain_range[1]
            )
            if not dtrade:
                return None

            m_dt = dtrade["metrics"]
            if m_dt.get("vol_ratio_vs_prev", 0) < min_daytrade_vol_ratio:
                return None

            today_val_b_val = m_dt.get('today_trading_val_억', 0)
            gain_from_open_val = m_dt.get('gain_from_open', 0)
            supp_ratio_val = m_dt.get('support_ratio', 0)
            vol_ratio_prev_val = m_dt.get('vol_ratio_vs_prev', 1.0)
            room_limit_val = m_dt.get('room_to_limit_pct', 0)

            reason_summary = f"대금 {today_val_b_val:,.0f}억 유입 | 시초대비 {gain_from_open_val:+.1f}% 돌파 | 양봉지지 {supp_ratio_val:.0f}%"
            reason_core = "장중 대규모 수급(거래대금)이 유입되며 시초가를 강력하게 상향 돌파하였고, 윗꼬리가 짧은 견고한 양봉 지지력을 유지하여 당일 장중 +5% 추가 슈팅 확률이 매우 높은 당일단타 주도주입니다."
            pm_ctx = params.get("premarket_ctx") or {}
            pm_regime = pm_ctx.get("regime", "중립")
            reason_criteria = [
                f"당일 거래대금 {today_val_b_val:,.0f}억 원 유입 (최소 기준 {min_daytrade_val_b}억 원 충족)",
                f"시초가 대비 {gain_from_open_val:+.1f}% 돌파 (유효 돌파 구간 +{intraday_gain_range[0]}% ~ +{intraday_gain_range[1]}% 충족)",
                f"장중 전일 거래량 대비 {vol_ratio_prev_val*100:.0f}% 돌파 (수급 집중 기준 {min_daytrade_vol_ratio*100:.0f}% 충족)",
                f"당일 진폭 중 고점 지지율 {supp_ratio_val:.1f}% (윗꼬리 짧은 탄탄한 양봉 지지, 65% 이상 충족)",
                f"프리마켓 매크로 점검: 글로벌 체제 [{pm_regime}] 상태 연동 (목표 익절 +5.0% / 손절 -2.5% 당일 청산 준수)",
                f"상한가(+30%)까지 잔여 상승폭 +{room_limit_val:.1f}% 확보 (목표 +5% 익절 공간 여유)"
            ]

            return {
                "code": code,
                "name": name,
                "market": market,
                "sector": sector,
                "close": close_p,
                "change_pct": chg_pct,
                "open_pct": open_pct,
                "high_pct": high_pct,
                "low_pct": low_pct,
                "open": open_p,
                "high": high_p,
                "low": low_p,
                "volume": vol_p,
                "timing_label": "장중 실시간 (09:10~14:30)",
                "strategy_mode": "DAY_TRADE_5PCT",
                "strategy_tag": "당일단타 (당일 +5%)",
                "premarket_regime": pm_regime,
                "premarket_gap_guide": "장중 09:10~14:30 실시간 수급 폭발 및 시초가 돌파 확인 후 진입",
                "premarket_risk": pm_ctx.get("gap_risk", "보통 (중립)"),
                "score": dtrade["score"],
                "label": dtrade["label"],
                "tp_pct": dtrade["tp_pct"],
                "sl_pct": dtrade["sl_pct"],
                "vol_ratio": vol_ratio_prev_val,
                "rsi": 65.0,
                "ma_status": "장중 수급폭발 돌파",
                "rule_note": dtrade["rule_note"],
                "reason_summary": reason_summary,
                "reason_core": reason_core,
                "reason_criteria": reason_criteria,
                "breakdown": {
                    "rsi_val": 65.0,
                    "ma_status": f"시초대비 {gain_from_open_val:+.1f}% 돌파봉",
                    "vol_ratio": vol_ratio_prev_val,
                    "bb_pct": supp_ratio_val / 100.0
                }
            }
        elif "스나이퍼" in sc_mode:
            bm_for_eval = None if ignore_market_filter else bm_ohlcv
            snp = evaluate_sniper_candidate(df_snp_eval, df_benchmark=bm_for_eval, min_daily_val_krw=min_val_krw_b * 100_000_000)
            if not snp or snp["metrics"]["disparity"] > max_disparity_val:
                return None

            m_snp = snp["metrics"]
            disp_val = m_snp.get("disparity", 100.0)
            avg_val_b = m_snp.get("avg_trading_val_억", 0)
            rsi_val = m_snp.get("rsi", 50.0)
            vol_r_val = m_snp.get("vol_ratio", 1.0)

            reason_summary = f"20일선 이격도 {disp_val:.1f}% 눌림목 | 20일평균 대금 {avg_val_b:,.0f}억 | RSI {rsi_val:.1f}"
            reason_core = "KODEX 200 지수 상승장 확인 후, 일평균 거래대금이 풍부한 우량주가 20일 이동평균선 눌림목에서 첫 반등을 시작하여 시초가 갭 기준 부합 시 승률 80%를 타겟하는 고확신 스나이퍼 종목입니다."
            reason_criteria = [
                "KODEX 200 지수 20일 이동평균선 상회 (지수 하락장 시스템 위험 사전 차단 필터 통과)" if not ignore_market_filter else "시장 필터 미적용 (개별 종목 기술적 지표 단독 검증)",
                f"20일 일평균 거래대금 {avg_val_b:,.0f}억 원 (기준 {min_val_krw_b}억 원 이상 풍부한 유동성)",
                f"20일선 이격도 {disp_val:.1f}% (설정 기준 98.0% ~ {max_disparity_val:.1f}% 내 20일선 첫 지지 반등)",
                "5일선 > 20일선 상단 위치 (단기 상승 모멘텀 유지)",
                f"RSI(14) {rsi_val:.1f} (과열 없는 건전한 에너지 응축 구간, 45~68 충족)",
                f"20일 평균 대비 거래량 {vol_r_val:.2f}배 반등 유입 (1.2배 이상 기준 충족)"
            ]

            return {
                "code": code,
                "name": name,
                "market": market,
                "sector": sector,
                "close": close_p,
                "change_pct": chg_pct,
                "open_pct": open_pct,
                "high_pct": high_pct,
                "low_pct": low_pct,
                "open": open_p,
                "high": high_p,
                "low": low_p,
                "volume": vol_p,
                "timing_label": "장초 (09:00 시초가)",
                "strategy_mode": "SNIPER",
                "strategy_tag": "스나이퍼 (오전 9시)",
                "score": snp["score"],
                "label": snp["label"],
                "tp_pct": snp["tp_pct"],
                "sl_pct": snp["sl_pct"],
                "vol_ratio": vol_r_val,
                "rsi": rsi_val,
                "ma_status": m_snp.get("ma_status", "20일선 지지"),
                "rule_note": "익일 시초가 갭 -1.5%~+1.5% 이내 시에만 체결 (+1.5% 초과 갭상승 시 진입금지)",
                "reason_summary": reason_summary,
                "reason_core": reason_core,
                "reason_criteria": reason_criteria,
                "breakdown": {
                    "rsi_val": rsi_val,
                    "ma_status": "20일선 지지 첫반등",
                    "vol_ratio": vol_r_val,
                    "bb_pct": 0.5
                }
            }
        elif "종가배팅" in sc_mode:
            cbet = evaluate_closing_bet_candidate(df_stock, min_today_val_krw=min_today_val_b * 100_000_000)
            if not cbet or cbet["metrics"]["day_return"] < min_day_ret_val:
                return None

            m_cb = cbet["metrics"]
            cb_val_b = m_cb.get("today_trading_val_억", 0)
            cb_ret = m_cb.get("day_return", 0)
            cb_hc = m_cb.get("high_close_ratio", 0)

            reason_summary = f"당일대금 {cb_val_b:,.0f}억 폭발 | 당일 +{cb_ret:.1f}% | 고가마감 {cb_hc:.1f}%"
            reason_core = "장마감 시점(15:20) 대규모 주도 거래대금과 함께 종가를 최고가 부근으로 마감하여, 익일 09:00 시초가 갭상승 및 장초반 슈팅(+1.5% 이상) 청산 확률이 극대화된 종가배팅 후보입니다."
            reason_criteria = [
                f"당일 거래대금 {cb_val_b:,.0f}억 원 폭발 (기준 {min_today_val_b}억 원 이상 시장 주도 수급)",
                f"당일 주가 +{cb_ret:.1f}% 장대양봉 마감 (기준 +{min_day_ret_val:.1f}% 이상 충족)",
                f"당일 고점 대비 종가 지지율 {cb_hc:.1f}% (기준 82% 이상, 장마감까지 차익실현 없는 탄탄한 종가)",
                "5일선 및 20일선 상단 돌파 (단기 이평선 정배열 확인)"
            ]

            return {
                "code": code,
                "name": name,
                "market": market,
                "sector": sector,
                "close": close_p,
                "change_pct": chg_pct,
                "open_pct": open_pct,
                "high_pct": high_pct,
                "low_pct": low_pct,
                "open": open_p,
                "high": high_p,
                "low": low_p,
                "volume": vol_p,
                "timing_label": "마감 직전 (15:20 종가)",
                "strategy_mode": "CLOSING_BET",
                "strategy_tag": "종가배팅 (마감 직전)",
                "score": cbet["score"],
                "label": cbet["label"],
                "tp_pct": cbet["tp_pct"],
                "sl_pct": cbet["sl_pct"],
                "vol_ratio": 2.0,
                "rsi": 65.0,
                "ma_status": "5·20일선 상단 고가마감",
                "rule_note": "15:20 종가 진입 -> 익일 09:00 시초가 갭익절(+1.5%↑) 또는 장초반 +1.5% 슈팅 청산",
                "reason_summary": reason_summary,
                "reason_core": reason_core,
                "reason_criteria": reason_criteria,
                "breakdown": {
                    "rsi_val": 65.0,
                    "ma_status": "주도주 고가마감",
                    "vol_ratio": 2.5,
                    "bb_pct": 0.8
                }
            }
        elif "5% 급등" in sc_mode or "5% 돌파" in sc_mode:
            s5 = evaluate_5pct_surge_candidate(
                df_stock,
                min_today_val_krw=min_surge_val_b * 100_000_000,
                min_day_return_pct=min_day_surge_pct
            )
            if not s5 and is_today_intraday and len(df_stock) >= 31:
                s5 = evaluate_5pct_surge_candidate(
                    df_stock.iloc[:-1],
                    min_today_val_krw=min_surge_val_b * 100_000_000,
                    min_day_return_pct=min_day_surge_pct
                )
            if not s5:
                return None

            m_s5 = s5["metrics"]
            s5_val_b = m_s5.get("today_trading_val_억", 0)
            s5_ret = m_s5.get("day_return", 0)
            s5_hc = m_s5.get("high_close_ratio", 0)

            reason_summary = f"초대형 대금 {s5_val_b:,.0f}억 | 당일 +{s5_ret:.1f}% 급등 | 종가마감 {s5_hc:.1f}%"
            reason_core = "시장의 주도 자금이 집중된 초대형 거래대금과 함께 장대양봉(상한가/준상한가 마루보즈)으로 최고가권 마감하여, 1~2일 내 장중 +5.0% 연속 돌파 슈팅 가능성이 검증된 초강세 대장주입니다."
            reason_criteria = [
                f"당일 거래대금 {s5_val_b:,.0f}억 원 집중 (기준 {min_surge_val_b}억 원 이상 최상위 주도주 수급)",
                f"당일 주가 +{s5_ret:.1f}% 급등 마감 (기준 +{min_day_surge_pct:.1f}% 이상 강력한 모멘텀)",
                f"당일 고점 대비 종가 유지율 {s5_hc:.1f}% (기준 88% 이상, 윗꼬리 극소화 마루보즈봉)",
                "매매 원칙: 15:20 종가 매수 -> 1~2일 내 장중 +5.0% 터치 시 즉시 전량 자동 익절"
            ]

            return {
                "code": code,
                "name": name,
                "market": market,
                "sector": sector,
                "close": close_p,
                "change_pct": chg_pct,
                "open_pct": open_pct,
                "high_pct": high_pct,
                "low_pct": low_pct,
                "open": open_p,
                "high": high_p,
                "low": low_p,
                "volume": vol_p,
                "timing_label": "마감 직전 (15:20 종가)",
                "strategy_mode": "SURGE_5PCT",
                "strategy_tag": "5%타겟 (마감 직전)",
                "score": s5["score"],
                "label": s5["label"],
                "tp_pct": s5["tp_pct"],
                "sl_pct": s5["sl_pct"],
                "vol_ratio": m_s5.get("vol_ratio", 3.0),
                "rsi": 70.0,
                "ma_status": "신고가/상한가 돌파",
                "rule_note": "전일 15:20 종가 매수 -> 1~2일 내 장중 +5.0% 돌파 시 즉시 전량 익절 (손절 -4.0%)",
                "reason_summary": reason_summary,
                "reason_core": reason_core,
                "reason_criteria": reason_criteria,
                "breakdown": {
                    "rsi_val": 70.0,
                    "ma_status": "주도주 상한가/급등돌파",
                    "vol_ratio": m_s5.get("vol_ratio", 3.0),
                    "bb_pct": 0.95
                }
            }
        else:
            score_res = evaluate_stock_latest(df_stock)
            if not score_res:
                return None
            score = score_res["score"]
            vol_ratio = score_res["breakdown"]["vol_ratio"]
            ma_status = score_res["breakdown"]["ma_status"]

            if score < min_screener_score or vol_ratio < min_vol_surge:
                return None
            if require_ma_align and ("정배열" not in ma_status and "골든" not in ma_status):
                return None

            reason_summary = f"종합 스코어 {score:.1f}점 | 거래량 {vol_ratio:.1f}배 급증 | {ma_status}"
            reason_core = "이동평균선 추세, 거래량 급증, RSI 모멘텀, 볼린저 밴드 지표의 다중 팩터 가중치 종합 퀀트 분석 결과 상위권에 랭크된 기술적 우량 후보입니다."
            reason_criteria = [
                f"종합 퀀트 스코어 {score:.1f}점 달성 (기준 {min_screener_score}점 이상 {score_res['label']} 등급)",
                f"20일 평균 대비 거래량 {vol_ratio:.1f}배 급증 (기준 {min_vol_surge}배 이상 수급 유입)",
                f"이평선 배열: {ma_status} (추세 상승 국면 확인)",
                f"RSI(14) {score_res['breakdown']['rsi_val']:.1f} (과열 없는 안정적 상승 탄력)"
            ]

            return {
                "code": code,
                "name": name,
                "market": market,
                "sector": sector,
                "close": close_p,
                "change_pct": chg_pct,
                "open_pct": open_pct,
                "high_pct": high_pct,
                "low_pct": low_pct,
                "open": open_p,
                "high": high_p,
                "low": low_p,
                "volume": vol_p,
                "timing_label": "장초 (09:00 시초가)",
                "strategy_mode": "NORMAL",
                "strategy_tag": "일반퀀트",
                "score": score,
                "label": score_res["label"],
                "tp_pct": score_res["tp_pct"],
                "sl_pct": score_res["sl_pct"],
                "vol_ratio": vol_ratio,
                "rsi": score_res["breakdown"]["rsi_val"],
                "ma_status": ma_status,
                "rule_note": "익일 시초가 진입 -> 점수 연동 동적 익절선(+1.5%~+3.0%)",
                "reason_summary": reason_summary,
                "reason_core": reason_core,
                "reason_criteria": reason_criteria,
                "breakdown": score_res["breakdown"]
            }
    except Exception:
        return None


class ScreeningWorker:
    """Manages an asynchronous stock screening task for a user session."""

    def __init__(self, session_id: str = "default"):
        self.session_id = session_id
        self._lock = threading.Lock()
        self._thread: Optional[threading.Thread] = None
        self._task_id = 0

        self._state: Dict[str, Any] = {
            "task_id": 0,
            "status": "IDLE",  # IDLE, RUNNING, COMPLETED, FAILED, CANCELLED
            "progress": 0.0,
            "current_index": 0,
            "total_count": 0,
            "current_name": "",
            "current_code": "",
            "mode_label": "",
            "results": [],
            "results_df": None,
            "error_message": None,
            "cancel_requested": False,
            "start_time": None,
            "end_time": None,
            "params": {},
        }

    def is_running(self) -> bool:
        with self._lock:
            return self._state["status"] == "RUNNING" and (self._thread is not None and self._thread.is_alive())

    def get_status(self) -> Dict[str, Any]:
        with self._lock:
            # Defensive check: if status says RUNNING but thread died unexpectedly
            if self._state["status"] == "RUNNING" and (self._thread is None or not self._thread.is_alive()):
                if self._state["results_df"] is not None:
                    self._state["status"] = "COMPLETED"
                elif self._state["cancel_requested"]:
                    self._state["status"] = "CANCELLED"
                else:
                    self._state["status"] = "FAILED"
                    if not self._state["error_message"]:
                        self._state["error_message"] = "백그라운드 스레드가 비정상 종료되었습니다."
            return dict(self._state)

    def cancel(self):
        with self._lock:
            if self._state["status"] == "RUNNING":
                self._state["cancel_requested"] = True

    def reset_status(self):
        with self._lock:
            self._state["status"] = "IDLE"
            self._state["cancel_requested"] = False
            self._state["error_message"] = None

    def start_screening(self, params: Dict[str, Any]) -> bool:
        """Starts screening in a daemon thread. Returns True if started, False if already running."""
        with self._lock:
            if self._state["status"] == "RUNNING" and self._thread is not None and self._thread.is_alive():
                return False

            self._task_id += 1
            current_task_id = self._task_id
            self._state = {
                "task_id": current_task_id,
                "status": "RUNNING",
                "progress": 0.0,
                "current_index": 0,
                "total_count": 0,
                "current_name": "유니버스 준비 중...",
                "current_code": "",
                "mode_label": params.get("sc_mode", "퀀트 스크리닝"),
                "results": [],
                "results_df": None,
                "error_message": None,
                "cancel_requested": False,
                "start_time": time.time(),
                "end_time": None,
                "params": params,
            }

            self._thread = threading.Thread(
                target=self._run_screening_thread,
                args=(params, current_task_id),
                daemon=True,
                name=f"ScreeningWorker-{self.session_id}-{current_task_id}"
            )
            self._thread.start()
            return True

    def _run_screening_thread(self, params: Dict[str, Any], task_id: int):
        try:
            sc_mode = params.get("sc_mode", "")
            target_market = params.get("target_market", "전체")
            sc_scope = params.get("sc_scope", "")

            # 1. Fetch universe
            mkt_param = None if target_market == "전체" else target_market
            df_univ = get_universe(market=mkt_param, active_only=True)

            if df_univ.empty:
                with self._lock:
                    if self._task_id == task_id:
                        self._state["status"] = "COMPLETED"
                        self._state["results_df"] = pd.DataFrame()
                        self._state["end_time"] = time.time()
                return

            # 2. Filter sample pool by scope
            if "관심종목" in sc_scope:
                with get_db_connection() as conn:
                    cursor = conn.cursor()
                    cursor.execute(
                        "SELECT w.code, w.name, w.market, COALESCE(u.sector, '기타') as sector "
                        "FROM watchlist w LEFT JOIN universe u ON w.code = u.code"
                    )
                    wl_recs = cursor.fetchall()
                sample_pool = pd.DataFrame([dict(r) for r in wl_recs]) if wl_recs else df_univ.head(100)
            elif "300선" in sc_scope or "급등" in sc_scope or "주도주" in sc_scope:
                # 1. Fetch real-time market cap, volume, and surging leader codes
                leader_codes = _fetch_live_market_leader_codes(target_market)

                # 2. Add watchlist codes
                with get_db_connection() as conn:
                    cursor = conn.cursor()
                    cursor.execute("SELECT code FROM watchlist")
                    wl_codes = [r["code"] for r in cursor.fetchall()]

                # 3. Add cached liquid universe stocks
                import glob, os
                cache_files = glob.glob(os.path.join("data", "cache", "*.parquet"))
                cached_codes = [
                    os.path.splitext(os.path.basename(f))[0]
                    for f in cache_files
                    if os.path.splitext(os.path.basename(f))[0].isdigit() and len(os.path.splitext(os.path.basename(f))[0]) == 6
                ]

                # Combined unique codes: real-time surging/leaders first, then watchlist, then liquid cache!
                combined_codes = list(dict.fromkeys(leader_codes + wl_codes + cached_codes))
                df_top = df_univ[df_univ["code"].isin(combined_codes)].copy()
                code_rank = {c: i for i, c in enumerate(combined_codes)}
                df_top["_order"] = df_top["code"].map(code_rank).fillna(9999)
                df_top = df_top.sort_values("_order").drop(columns=["_order"])

                df_rest = df_univ[~df_univ["code"].isin(combined_codes)]
                sample_pool = pd.concat([df_top, df_rest]).head(300).reset_index(drop=True)
            else:
                sample_pool = df_univ

            total_pool_cnt = len(sample_pool)
            with self._lock:
                if self._task_id != task_id:
                    return
                self._state["total_count"] = total_pool_cnt

            # 3. Strategy setup & benchmark
            bm_ohlcv = get_benchmark_ohlcv() if ("스나이퍼" in sc_mode or "당일 단타" in sc_mode or "장전 시초가" in sc_mode or "08:00~09:00" in sc_mode) else None
            try:
                from src.collectors.us_market import get_premarket_analysis_context
                params["premarket_ctx"] = get_premarket_analysis_context()
            except Exception:
                params["premarket_ctx"] = {}
            screened_results = []

            # Batch fetch real-time live quotes for sample pool to match HTS 100%
            pool_codes = sample_pool["code"].astype(str).str.zfill(6).tolist()
            realtime_quotes = fetch_realtime_quotes(pool_codes)

            # 4. Multi-threaded Evaluation Loop
            now_dt = get_now_kst()
            today_str = now_dt.strftime("%Y-%m-%d")
            last_trading_day_str = get_last_completed_trading_day()

            eval_workers = min(16, max(1, total_pool_cnt)) if total_pool_cnt > 0 else 1
            with ThreadPoolExecutor(max_workers=eval_workers) as executor:
                def _task_eval(row_item):
                    try:
                        res = _evaluate_single_stock(
                            row=row_item,
                            realtime_quotes=realtime_quotes,
                            now_dt=now_dt,
                            today_str=today_str,
                            last_trading_day_str=last_trading_day_str,
                            sc_mode=sc_mode,
                            bm_ohlcv=bm_ohlcv,
                            params=params
                        )
                        return row_item["name"], row_item["code"], res
                    except Exception:
                        return row_item["name"], row_item["code"], None

                futures = [executor.submit(_task_eval, row) for _, row in sample_pool.iterrows()]

                completed_cnt = 0
                for fut in as_completed(futures):
                    # Check cancellation
                    with self._lock:
                        if self._task_id != task_id:
                            for f in futures:
                                f.cancel()
                            return
                        if self._state["cancel_requested"]:
                            self._state["status"] = "CANCELLED"
                            self._state["end_time"] = time.time()
                            for f in futures:
                                f.cancel()
                            return

                    try:
                        stock_name, stock_code, cand_res = fut.result()
                    except Exception:
                        stock_name, stock_code, cand_res = "", "", None

                    with self._lock:
                        completed_cnt += 1
                        self._state["current_index"] = completed_cnt
                        self._state["progress"] = completed_cnt / total_pool_cnt if total_pool_cnt > 0 else 1.0
                        if stock_name:
                            self._state["current_name"] = stock_name
                            self._state["current_code"] = stock_code

                    if cand_res:
                        screened_results.append(cand_res)

            # 5. Finalize results
            df_screened = pd.DataFrame(screened_results)
            if not df_screened.empty:
                df_screened = df_screened.sort_values("score", ascending=False).reset_index(drop=True)

            with self._lock:
                if self._task_id == task_id:
                    self._state["status"] = "COMPLETED"
                    self._state["results"] = screened_results
                    self._state["results_df"] = df_screened
                    self._state["progress"] = 1.0
                    self._state["end_time"] = time.time()

        except Exception as e:
            with self._lock:
                if self._task_id == task_id:
                    self._state["status"] = "FAILED"
                    self._state["error_message"] = str(e)
                    self._state["end_time"] = time.time()


# Global Registry mapping session_id -> ScreeningWorker
_WORKER_REGISTRY: Dict[str, ScreeningWorker] = {}
_REGISTRY_LOCK = threading.Lock()


def get_screening_worker(session_id: str = "default") -> ScreeningWorker:
    """Returns or creates the ScreeningWorker for the given session_id."""
    with _REGISTRY_LOCK:
        if session_id not in _WORKER_REGISTRY:
            _WORKER_REGISTRY[session_id] = ScreeningWorker(session_id=session_id)
        return _WORKER_REGISTRY[session_id]
