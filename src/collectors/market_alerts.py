"""
Real-time Market Alert & Warning System.
Evaluates KRX Sidecar, Circuit Breakers, Market Shocks, and Global Volatility.
Provides quantitative condition checks, timing calculation, and real-time alert payload for UI.
Strictly NO EMOJIS in code, logs, comments, or output.
"""

import os
import json
import logging
from typing import Dict, Any, List, Optional
from datetime import datetime, timedelta

from src.database.models import get_now_kst
from src.collectors.domestic_indices import fetch_domestic_indices_summary
from src.collectors.us_market import fetch_us_market_summary

logger = logging.getLogger(__name__)

CACHE_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "data", "cache")
ALERT_STATE_FILE = os.path.join(CACHE_DIR, "market_alerts_state.json")

_ALERT_STATE_CACHE: Dict[str, Any] = {}


def load_alert_state() -> Dict[str, Any]:
    """Loads alert timing state from cache file or in-memory cache."""
    global _ALERT_STATE_CACHE
    if _ALERT_STATE_CACHE:
        return _ALERT_STATE_CACHE
    try:
        if os.path.exists(ALERT_STATE_FILE):
            with open(ALERT_STATE_FILE, "r", encoding="utf-8") as f:
                _ALERT_STATE_CACHE = json.load(f)
                return _ALERT_STATE_CACHE
    except Exception as e:
        logger.warning(f"Failed to load alert state file: {e}")
    _ALERT_STATE_CACHE = {}
    return _ALERT_STATE_CACHE


def save_alert_state(state: Dict[str, Any]) -> None:
    """Saves alert timing state to cache file and in-memory cache."""
    global _ALERT_STATE_CACHE
    _ALERT_STATE_CACHE = state
    try:
        os.makedirs(os.path.dirname(ALERT_STATE_FILE), exist_ok=True)
        with open(ALERT_STATE_FILE, "w", encoding="utf-8") as f:
            json.dump(state, f, ensure_ascii=False, indent=2)
    except Exception as e:
        logger.warning(f"Failed to save alert state file: {e}")


def clear_alert_state() -> None:
    """Clears alert timing state cache (used for daily resets and testing)."""
    global _ALERT_STATE_CACHE
    _ALERT_STATE_CACHE = {}
    if os.path.exists(ALERT_STATE_FILE):
        try:
            os.remove(ALERT_STATE_FILE)
        except Exception:
            pass


def resolve_alert_timing(
    alert_id: str,
    alert_type: str,
    duration_minutes: int,
    now_kst: Optional[datetime] = None
) -> Dict[str, Any]:
    """
    Resolves stable trigger and end times for Sidecar and Circuit Breakers.
    Uses daily cached state so that 10-second polling does not slide the trigger time forward.
    """
    if now_kst is None:
        now_kst = get_now_kst()

    date_str = now_kst.strftime("%Y-%m-%d")
    state = load_alert_state()

    # Reset state if date changed
    if state.get("date") != date_str:
        state = {"date": date_str, "alerts": {}}

    alerts_dict = state.setdefault("alerts", {})
    alert_record = alerts_dict.get(alert_id)

    if alert_record:
        try:
            triggered_at = datetime.fromisoformat(alert_record["triggered_iso"])
        except Exception:
            triggered_at = now_kst
            alert_record = None

    if not alert_record:
        triggered_at = now_kst
        if duration_minutes > 0:
            end_at = triggered_at + timedelta(minutes=duration_minutes)
            end_iso = end_at.isoformat()
            end_time_str = end_at.strftime("%H:%M:%S")
        else:
            end_at = None
            end_iso = None
            end_time_str = "당일 장 마감"

        alert_record = {
            "alert_id": alert_id,
            "alert_type": alert_type,
            "duration_minutes": duration_minutes,
            "triggered_iso": triggered_at.isoformat(),
            "triggered_time": triggered_at.strftime("%H:%M:%S"),
            "end_iso": end_iso,
            "end_time": end_time_str,
        }
        alerts_dict[alert_id] = alert_record
        save_alert_state(state)

    triggered_time_str = alert_record.get("triggered_time", triggered_at.strftime("%H:%M:%S"))
    end_time_str = alert_record.get("end_time", "")

    if duration_minutes > 0:
        try:
            end_at = datetime.fromisoformat(alert_record["end_iso"])
            diff_sec = int((end_at - now_kst).total_seconds())
        except Exception:
            diff_sec = 0

        duration_str = f"{duration_minutes}분간 효력 정지" if alert_type == "SIDECAR" else f"{duration_minutes}분간 매매 중단"
        if diff_sec > 0:
            rem_m = diff_sec // 60
            rem_s = diff_sec % 60
            rem_str = f"잔여 {rem_m}분 {rem_s:02d}초"
            status = "발동 중"
        else:
            rem_str = "해제 완료"
            status = "발동 종료 (해제)"
        time_info = f"발동 {triggered_time_str} · 종료 {end_time_str} ({duration_str}, {rem_str})"
    else:
        diff_sec = 0
        duration_str = "당일 거래 전면 마감"
        rem_str = "당일 장 마감"
        status = "장 마감 발동"
        time_info = f"발동 {triggered_time_str} · 종료 {end_time_str} (당일 거래 전면 마감)"

    return {
        "triggered_time": triggered_time_str,
        "end_time": end_time_str,
        "duration_minutes": duration_minutes,
        "duration_str": duration_str,
        "remaining_seconds": max(0, diff_sec),
        "remaining_str": rem_str,
        "status": status,
        "time_info": time_info,
    }


def evaluate_market_warning_alerts(
    domestic_summary: Optional[Dict[str, Any]] = None,
    us_summary: Optional[Dict[str, Any]] = None,
    now_kst: Optional[datetime] = None
) -> Dict[str, Any]:
    """
    Evaluates real-time market warnings including:
    1. KRX Sidecar:
       - KOSPI 200 Futures (FUT): +/- 5.0% (Trigger, 5 min halt), +/- 3.5% (Watch)
       - KOSDAQ 150 (KQI150): +/- 6.0% (Trigger, 5 min halt), +/- 4.5% (Watch)
    2. KRX Circuit Breakers:
       - KOSPI / KOSDAQ: -8.0% (Stage 1, 20 min halt), -15.0% (Stage 2, 20 min halt),
         -20.0% (Stage 3, market closed), -5.0% (Watch)
    3. Market Shocks:
       - KOSPI / KOSDAQ: <= -3.0% (Drop alert), >= +3.0% (Surge alert)
    4. Global Macro:
       - VIX >= 25.0 or VIX surge > +18.0%
       - USD/KRW >= 1430.0 or FX surge > +1.5%

    Returns structured alert payload:
        - level: 'CRITICAL' | 'WARNING' | 'NORMAL'
        - badge_text: str
        - badge_color: str (hex)
        - bg_color: str (hex)
        - border_color: str (hex)
        - text_color: str (hex)
        - title: str
        - summary_text: str
        - summary_key: str (unique signature to track state transitions)
        - alerts: List[Dict[str, Any]]
        - action_guide: str
        - updated_at_str: str
        - triggered_time: Optional[str]
        - end_time: Optional[str]
        - duration_str: str
        - remaining_str: str
        - alert_status: str
        - time_info: str
    """
    if domestic_summary is None:
        try:
            domestic_summary = fetch_domestic_indices_summary()
        except Exception as e:
            logger.warning(f"Failed to fetch domestic summary for market alerts: {e}")
            domestic_summary = {"items": []}

    items = domestic_summary.get("items", [])
    item_map = {it.get("code"): it for it in items if "code" in it}

    if now_kst is None:
        now_kst = get_now_kst()
    updated_at_str = now_kst.strftime("%H:%M:%S KST")

    critical_alerts: List[Dict[str, Any]] = []
    warning_alerts: List[Dict[str, Any]] = []

    # ---------------------------------------------------------
    # 1. KRX Sidecar Evaluation
    # ---------------------------------------------------------
    # KOSPI 200 Futures (FUT)
    fut_item = item_map.get("FUT")
    if fut_item:
        chg_pct = fut_item.get("change_pct", 0.0)
        curr_val = fut_item.get("close", 0.0)
        if chg_pct >= 5.0:
            timing = resolve_alert_timing("SIDECAR_BUY_KOSPI", "SIDECAR", duration_minutes=5, now_kst=now_kst)
            critical_alerts.append({
                "id": "SIDECAR_BUY_KOSPI",
                "type": "SIDECAR",
                "target": "코스피 200 선물",
                "target_code": "FUT",
                "level": "CRITICAL",
                "change_pct": chg_pct,
                "current_val": curr_val,
                "threshold": "+5.0%",
                "triggered_time": timing["triggered_time"],
                "end_time": timing["end_time"],
                "duration_minutes": timing["duration_minutes"],
                "duration_str": timing["duration_str"],
                "remaining_seconds": timing["remaining_seconds"],
                "remaining_str": timing["remaining_str"],
                "status": timing["status"],
                "time_info": timing["time_info"],
                "title": f"[매수 사이드카 발동] 코스피 200 선물 급등 (+{chg_pct:.2f}%) (발동 {timing['triggered_time']} · 종료 {timing['end_time']})",
                "message": f"코스피 200 선물이 +5.0% 이상 급등하여 매수 사이드카 발동 기준에 도달했습니다. 프로그램 매수호가 효력이 {timing['duration_minutes']}분간 정지됩니다. (발동시각: {timing['triggered_time']} / 예정 종료시각: {timing['end_time']})",
                "action_guide": f"프로그램 매수호가 효력 정지({timing['time_info']}) 대비, 고점 추격 매수 자제 및 호가 스프레드 확대 주의"
            })
        elif chg_pct <= -5.0:
            timing = resolve_alert_timing("SIDECAR_SELL_KOSPI", "SIDECAR", duration_minutes=5, now_kst=now_kst)
            critical_alerts.append({
                "id": "SIDECAR_SELL_KOSPI",
                "type": "SIDECAR",
                "target": "코스피 200 선물",
                "target_code": "FUT",
                "level": "CRITICAL",
                "change_pct": chg_pct,
                "current_val": curr_val,
                "threshold": "-5.0%",
                "triggered_time": timing["triggered_time"],
                "end_time": timing["end_time"],
                "duration_minutes": timing["duration_minutes"],
                "duration_str": timing["duration_str"],
                "remaining_seconds": timing["remaining_seconds"],
                "remaining_str": timing["remaining_str"],
                "status": timing["status"],
                "time_info": timing["time_info"],
                "title": f"[매도 사이드카 발동] 코스피 200 선물 급락 ({chg_pct:.2f}%) (발동 {timing['triggered_time']} · 종료 {timing['end_time']})",
                "message": f"코스피 200 선물이 -5.0% 이하로 급락하여 매도 사이드카 발동 기준에 도달했습니다. 프로그램 매도호가 효력이 {timing['duration_minutes']}분간 정지됩니다. (발동시각: {timing['triggered_time']} / 예정 종료시각: {timing['end_time']})",
                "action_guide": f"프로그램 매도호가 효력 정지({timing['time_info']}) 대비, 투매 동참 자제 및 분할 방어 대응"
            })
        elif chg_pct <= -3.5 or chg_pct >= 3.5:
            direction = "급등" if chg_pct > 0 else "급락"
            warning_alerts.append({
                "id": "SIDECAR_WATCH_KOSPI",
                "type": "SIDECAR_WATCH",
                "target": "코스피 200 선물",
                "target_code": "FUT",
                "level": "WARNING",
                "change_pct": chg_pct,
                "current_val": curr_val,
                "threshold": "+/-3.5%",
                "triggered_time": now_kst.strftime("%H:%M:%S"),
                "end_time": None,
                "duration_minutes": 0,
                "duration_str": "주의보 감시",
                "remaining_seconds": 0,
                "remaining_str": "실시간 감시",
                "status": "주의보",
                "time_info": f"주의보 감지 {now_kst.strftime('%H:%M:%S')}",
                "title": f"[사이드카 주의보] 코스피 200 선물 {direction} ({chg_pct:+.2f}%)",
                "message": f"코스피 200 선물이 {chg_pct:+.2f}% 변동 중입니다. 변동폭 확대 시 사이드카(5.0%, 5분간 효력정지) 발동 가능성이 있습니다.",
                "action_guide": "선물 변동성 확대 구간. 신규 포지션 진입 시 슬리피지 주의 및 손절 기준 준수"
            })

    # KOSDAQ 150 (KQI150)
    kq_item = item_map.get("KQI150")
    if kq_item:
        chg_pct = kq_item.get("change_pct", 0.0)
        curr_val = kq_item.get("close", 0.0)
        if chg_pct >= 6.0:
            timing = resolve_alert_timing("SIDECAR_BUY_KOSDAQ", "SIDECAR", duration_minutes=5, now_kst=now_kst)
            critical_alerts.append({
                "id": "SIDECAR_BUY_KOSDAQ",
                "type": "SIDECAR",
                "target": "코스닥 150",
                "target_code": "KQI150",
                "level": "CRITICAL",
                "change_pct": chg_pct,
                "current_val": curr_val,
                "threshold": "+6.0%",
                "triggered_time": timing["triggered_time"],
                "end_time": timing["end_time"],
                "duration_minutes": timing["duration_minutes"],
                "duration_str": timing["duration_str"],
                "remaining_seconds": timing["remaining_seconds"],
                "remaining_str": timing["remaining_str"],
                "status": timing["status"],
                "time_info": timing["time_info"],
                "title": f"[매수 사이드카 발동] 코스닥 150 급등 (+{chg_pct:.2f}%) (발동 {timing['triggered_time']} · 종료 {timing['end_time']})",
                "message": f"코스닥 150 지수가 +6.0% 이상 급등하여 코스닥 매수 사이드카 발동 기준에 도달했습니다. 프로그램 매수호가 효력이 {timing['duration_minutes']}분간 정지됩니다. (발동시각: {timing['triggered_time']} / 예정 종료시각: {timing['end_time']})",
                "action_guide": f"코스닥 프로그램 매수 정지({timing['time_info']}) 대비, 추격 매수 자제"
            })
        elif chg_pct <= -6.0:
            timing = resolve_alert_timing("SIDECAR_SELL_KOSDAQ", "SIDECAR", duration_minutes=5, now_kst=now_kst)
            critical_alerts.append({
                "id": "SIDECAR_SELL_KOSDAQ",
                "type": "SIDECAR",
                "target": "코스닥 150",
                "target_code": "KQI150",
                "level": "CRITICAL",
                "change_pct": chg_pct,
                "current_val": curr_val,
                "threshold": "-6.0%",
                "triggered_time": timing["triggered_time"],
                "end_time": timing["end_time"],
                "duration_minutes": timing["duration_minutes"],
                "duration_str": timing["duration_str"],
                "remaining_seconds": timing["remaining_seconds"],
                "remaining_str": timing["remaining_str"],
                "status": timing["status"],
                "time_info": timing["time_info"],
                "title": f"[매도 사이드카 발동] 코스닥 150 급락 ({chg_pct:.2f}%) (발동 {timing['triggered_time']} · 종료 {timing['end_time']})",
                "message": f"코스닥 150 지수가 -6.0% 이하로 급락하여 코스닥 매도 사이드카 발동 기준에 도달했습니다. 프로그램 매도호가 효력이 {timing['duration_minutes']}분간 정지됩니다. (발동시각: {timing['triggered_time']} / 예정 종료시각: {timing['end_time']})",
                "action_guide": f"코스닥 프로그램 매도 정지({timing['time_info']}) 대비, 투매 자제 및 리스크 관리"
            })
        elif chg_pct <= -4.5 or chg_pct >= 4.5:
            direction = "급등" if chg_pct > 0 else "급락"
            warning_alerts.append({
                "id": "SIDECAR_WATCH_KOSDAQ",
                "type": "SIDECAR_WATCH",
                "target": "코스닥 150",
                "target_code": "KQI150",
                "level": "WARNING",
                "change_pct": chg_pct,
                "current_val": curr_val,
                "threshold": "+/-4.5%",
                "triggered_time": now_kst.strftime("%H:%M:%S"),
                "end_time": None,
                "duration_minutes": 0,
                "duration_str": "주의보 감시",
                "remaining_seconds": 0,
                "remaining_str": "실시간 감시",
                "status": "주의보",
                "time_info": f"주의보 감지 {now_kst.strftime('%H:%M:%S')}",
                "title": f"[사이드카 주의보] 코스닥 150 {direction} ({chg_pct:+.2f}%)",
                "message": f"코스닥 150 지수가 {chg_pct:+.2f}% 급변동 중입니다. 사이드카(6.0%, 5분간 효력정지) 발동 기준에 근접했습니다.",
                "action_guide": "코스닥 고변동성 구간. 단기 단타 진입 시 철저한 비중 축소"
            })

    # ---------------------------------------------------------
    # 2. KRX Circuit Breakers Evaluation (KOSPI & KOSDAQ)
    # ---------------------------------------------------------
    for code, name in [("KOSPI", "코스피"), ("KOSDAQ", "코스닥")]:
        idx_item = item_map.get(code)
        if not idx_item:
            continue
        chg_pct = idx_item.get("change_pct", 0.0)
        curr_val = idx_item.get("close", 0.0)

        if chg_pct <= -20.0:
            timing = resolve_alert_timing(f"CB_LV3_{code}", "CIRCUIT_BREAKER", duration_minutes=0, now_kst=now_kst)
            critical_alerts.append({
                "id": f"CB_LV3_{code}",
                "type": "CIRCUIT_BREAKER",
                "target": name,
                "target_code": code,
                "level": "CRITICAL",
                "change_pct": chg_pct,
                "current_val": curr_val,
                "threshold": "-20.0%",
                "triggered_time": timing["triggered_time"],
                "end_time": timing["end_time"],
                "duration_minutes": timing["duration_minutes"],
                "duration_str": timing["duration_str"],
                "remaining_seconds": timing["remaining_seconds"],
                "remaining_str": timing["remaining_str"],
                "status": timing["status"],
                "time_info": timing["time_info"],
                "title": f"[서킷브레이커 3단계] {name} -20% 이상 폭락 ({chg_pct:.2f}%) (발동 {timing['triggered_time']} · 당일 장 마감)",
                "message": f"{name} 지수가 -20% 이상 폭락하여 서킷브레이커 3단계 발동 요건에 도달했습니다. 당일 전 종목 매매가 전면 종료됩니다. (발동시각: {timing['triggered_time']} / 당일 장 마감)",
                "action_guide": "시장 조기 마감 대비. 모든 미체결 주문 상태 점검"
            })
        elif chg_pct <= -15.0:
            timing = resolve_alert_timing(f"CB_LV2_{code}", "CIRCUIT_BREAKER", duration_minutes=20, now_kst=now_kst)
            critical_alerts.append({
                "id": f"CB_LV2_{code}",
                "type": "CIRCUIT_BREAKER",
                "target": name,
                "target_code": code,
                "level": "CRITICAL",
                "change_pct": chg_pct,
                "current_val": curr_val,
                "threshold": "-15.0%",
                "triggered_time": timing["triggered_time"],
                "end_time": timing["end_time"],
                "duration_minutes": timing["duration_minutes"],
                "duration_str": timing["duration_str"],
                "remaining_seconds": timing["remaining_seconds"],
                "remaining_str": timing["remaining_str"],
                "status": timing["status"],
                "time_info": timing["time_info"],
                "title": f"[서킷브레이커 2단계] {name} -15% 이상 폭락 ({chg_pct:.2f}%) (발동 {timing['triggered_time']} · 종료 {timing['end_time']})",
                "message": f"{name} 지수가 -15% 이상 폭락하여 서킷브레이커 2단계 발동 요건에 도달했습니다. 모든 주식 거래가 20분간 전면 중단됩니다. (발동시각: {timing['triggered_time']} / 중단종료: {timing['end_time']}, 이후 10분간 단일가매매)",
                "action_guide": f"20분간 매매 중단({timing['time_info']}) 및 10분간 단일가매매 진행. 호가 안정화 대기"
            })
        elif chg_pct <= -8.0:
            timing = resolve_alert_timing(f"CB_LV1_{code}", "CIRCUIT_BREAKER", duration_minutes=20, now_kst=now_kst)
            critical_alerts.append({
                "id": f"CB_LV1_{code}",
                "type": "CIRCUIT_BREAKER",
                "target": name,
                "target_code": code,
                "level": "CRITICAL",
                "change_pct": chg_pct,
                "current_val": curr_val,
                "threshold": "-8.0%",
                "triggered_time": timing["triggered_time"],
                "end_time": timing["end_time"],
                "duration_minutes": timing["duration_minutes"],
                "duration_str": timing["duration_str"],
                "remaining_seconds": timing["remaining_seconds"],
                "remaining_str": timing["remaining_str"],
                "status": timing["status"],
                "time_info": timing["time_info"],
                "title": f"[서킷브레이커 1단계] {name} -8% 이상 급락 ({chg_pct:.2f}%) (발동 {timing['triggered_time']} · 종료 {timing['end_time']})",
                "message": f"{name} 지수가 -8% 이상 급락하여 서킷브레이커 1단계 발동 요건에 도달했습니다. 모든 주식 거래가 20분간 전면 중단됩니다. (발동시각: {timing['triggered_time']} / 중단종료: {timing['end_time']}, 이후 10분간 단일가매매)",
                "action_guide": f"전 종목 거래 일시 정지({timing['time_info']}) 대비. 재개 후 극심한 호가 변동성 유의"
            })
        elif chg_pct <= -5.0:
            warning_alerts.append({
                "id": f"CB_WATCH_{code}",
                "type": "CIRCUIT_BREAKER_WATCH",
                "target": name,
                "target_code": code,
                "level": "WARNING",
                "change_pct": chg_pct,
                "current_val": curr_val,
                "threshold": "-5.0%",
                "triggered_time": now_kst.strftime("%H:%M:%S"),
                "end_time": None,
                "duration_minutes": 0,
                "duration_str": "주의보 감시",
                "remaining_seconds": 0,
                "remaining_str": "실시간 감시",
                "status": "주의보",
                "time_info": f"주의보 감지 {now_kst.strftime('%H:%M:%S')}",
                "title": f"[서킷브레이커 주의보] {name} -5% 이상 급락 ({chg_pct:.2f}%)",
                "message": f"{name} 지수가 -5% 이상 급락 중입니다. -8% 도달 시 서킷브레이커(20분 매매중단)가 발동되므로 시장 경계가 필요합니다.",
                "action_guide": "서킷브레이커 기준(-8%) 근접. 신규 매수 전면 자제 및 현금 비중 확대"
            })
        elif chg_pct <= -3.0:
            warning_alerts.append({
                "id": f"SHOCK_DROP_{code}",
                "type": "MARKET_SHOCK",
                "target": name,
                "target_code": code,
                "level": "WARNING",
                "change_pct": chg_pct,
                "current_val": curr_val,
                "threshold": "-3.0%",
                "triggered_time": now_kst.strftime("%H:%M:%S"),
                "end_time": None,
                "duration_minutes": 0,
                "duration_str": "단기 급락 경보",
                "remaining_seconds": 0,
                "remaining_str": "실시간 감시",
                "status": "경보",
                "time_info": f"경보 감지 {now_kst.strftime('%H:%M:%S')}",
                "title": f"[시장 급락 경보] {name} {chg_pct:.2f}% 급락세",
                "message": f"{name} 지수가 -3.0%를 초과하여 급락 중입니다.",
                "action_guide": "지수 급락 구간. 보수적 진입 및 손절선 방어 필수"
            })
        elif chg_pct >= 3.0:
            warning_alerts.append({
                "id": f"SHOCK_SURGE_{code}",
                "type": "MARKET_SURGE",
                "target": name,
                "target_code": code,
                "level": "WARNING",
                "change_pct": chg_pct,
                "current_val": curr_val,
                "threshold": "+3.0%",
                "triggered_time": now_kst.strftime("%H:%M:%S"),
                "end_time": None,
                "duration_minutes": 0,
                "duration_str": "단기 급등 경보",
                "remaining_seconds": 0,
                "remaining_str": "실시간 감시",
                "status": "경보",
                "time_info": f"경보 감지 {now_kst.strftime('%H:%M:%S')}",
                "title": f"[시장 급등 경보] {name} +{chg_pct:.2f}% 급등세",
                "message": f"{name} 지수가 +3.0%를 초과하여 급등 중입니다.",
                "action_guide": "단기 과열 구간. 추격 매수 자제 및 분할 익절 고려"
            })

    # ---------------------------------------------------------
    # 3. Global Macro Volatility Evaluation
    # ---------------------------------------------------------
    if us_summary:
        vix_val = us_summary.get("vix_val", 16.0)
        vix_pct = us_summary.get("vix_pct", 0.0)
        usdkrw = us_summary.get("usdkrw_rate", 1340.0)
        if vix_val >= 25.0 or vix_pct > 18.0:
            warning_alerts.append({
                "id": "GLOBAL_VIX_SURGE",
                "type": "GLOBAL_RISK",
                "target": "VIX 공포지수",
                "target_code": "VIX",
                "level": "WARNING",
                "change_pct": vix_pct,
                "current_val": vix_val,
                "threshold": "25.0pt / +18%",
                "triggered_time": now_kst.strftime("%H:%M:%S"),
                "end_time": None,
                "duration_minutes": 0,
                "duration_str": "글로벌 위험",
                "remaining_seconds": 0,
                "remaining_str": "실시간 감시",
                "status": "경보",
                "time_info": f"경보 감지 {now_kst.strftime('%H:%M:%S')}",
                "title": f"[글로벌 변동성 경보] VIX 공포지수 {vix_val:.1f}pt 급등",
                "message": f"미국 VIX 지수가 {vix_val:.1f}pt (+{vix_pct:.1f}%)로 급등하여 글로벌 위험 회피 심리가 고조되었습니다.",
                "action_guide": "해외발 변동성 충격 전이 대비. 방어적 관망 권장"
            })
        if usdkrw >= 1430.0:
            warning_alerts.append({
                "id": "FX_SHOCK",
                "type": "GLOBAL_RISK",
                "target": "원/달러 환율",
                "target_code": "USD/KRW",
                "level": "WARNING",
                "change_pct": 0.0,
                "current_val": usdkrw,
                "threshold": "1,430원",
                "triggered_time": now_kst.strftime("%H:%M:%S"),
                "end_time": None,
                "duration_minutes": 0,
                "duration_str": "환율 급등",
                "remaining_seconds": 0,
                "remaining_str": "실시간 감시",
                "status": "경보",
                "time_info": f"경보 감지 {now_kst.strftime('%H:%M:%S')}",
                "title": f"[환율 급등 경보] 원/달러 {usdkrw:,.1f}원 기록",
                "message": "원/달러 환율이 1,430원선에 도달하여 외국인 자금 이탈 압력이 높아졌습니다.",
                "action_guide": "외국인 수급 점검 및 대형주 수급 모니터링"
            })

    # ---------------------------------------------------------
    # 4. Synthesize Overall Market Warning Status
    # ---------------------------------------------------------
    all_alerts = critical_alerts + warning_alerts

    primary_timing_alert = next((a for a in critical_alerts if a.get("triggered_time")), None)
    if not primary_timing_alert:
        primary_timing_alert = next((a for a in warning_alerts if a.get("triggered_time")), None)

    triggered_time = primary_timing_alert.get("triggered_time") if primary_timing_alert else None
    end_time = primary_timing_alert.get("end_time") if primary_timing_alert else None
    time_info = primary_timing_alert.get("time_info", "") if primary_timing_alert else ""
    duration_str = primary_timing_alert.get("duration_str", "") if primary_timing_alert else ""
    remaining_str = primary_timing_alert.get("remaining_str", "") if primary_timing_alert else ""
    alert_status = primary_timing_alert.get("status", "") if primary_timing_alert else ""

    if critical_alerts:
        level = "CRITICAL"
        has_sidecar_crit = any(a["type"] == "SIDECAR" for a in critical_alerts)
        has_cb_crit = any(a["type"] == "CIRCUIT_BREAKER" for a in critical_alerts)
        if has_sidecar_crit and has_cb_crit:
            badge_text = "[서킷브레이커 & 사이드카 발동]"
        elif has_cb_crit:
            cb_types = [a["id"] for a in critical_alerts if a["type"] == "CIRCUIT_BREAKER"]
            if any("LV3" in cid for cid in cb_types):
                badge_text = "[서킷브레이커 3단계 발동]"
            elif any("LV2" in cid for cid in cb_types):
                badge_text = "[서킷브레이커 2단계 발동]"
            else:
                badge_text = "[서킷브레이커 1단계 발동]"
        elif has_sidecar_crit:
            if any("BUY" in a["id"] for a in critical_alerts):
                badge_text = "[매수 사이드카 발동]"
            else:
                badge_text = "[매도 사이드카 발동]"
        else:
            badge_text = "[시장 비상 발동 경보]"

        badge_color = "#DC2626"
        bg_color = "#FEF2F2"
        border_color = "#F87171"
        text_color = "#991B1B"
        title = critical_alerts[0]["title"]
        summary_text = " · ".join([a["title"].split("] ")[-1] for a in critical_alerts[:2]])
        action_guide = critical_alerts[0]["action_guide"]
    elif warning_alerts:
        level = "WARNING"
        has_sidecar_watch = any(a["type"] == "SIDECAR_WATCH" for a in warning_alerts)
        has_cb_watch = any(a["type"] == "CIRCUIT_BREAKER_WATCH" for a in warning_alerts)
        if has_sidecar_watch and has_cb_watch:
            badge_text = "[사이드카 & 서킷브레이커 주의보]"
        elif has_sidecar_watch:
            badge_text = "[사이드카 주의보]"
        elif has_cb_watch:
            badge_text = "[서킷브레이커 주의보]"
        else:
            badge_text = "[시장 급변동 경보]"

        badge_color = "#D97706"
        bg_color = "#FFFBEB"
        border_color = "#FBBF24"
        text_color = "#92400E"
        title = warning_alerts[0]["title"]
        summary_text = " · ".join([a["title"].split("] ")[-1] for a in warning_alerts[:2]])
        action_guide = warning_alerts[0]["action_guide"]
    else:
        level = "NORMAL"
        badge_text = "[시장 안정 / 정상 거래]"
        badge_color = "#059669"
        bg_color = "#F0FDF4"
        border_color = "#86EFAC"
        text_color = "#166534"
        title = "[시장 정상] 사이드카·서킷브레이커 정상 범위 유지 중"
        summary_text = "코스피 200 선물 및 주요 지수 변동성 안정 (사이드카/서킷 발동 조건 없음)"
        action_guide = "시장 전체 지수가 정상 변동 범위 내에서 거래되고 있습니다. 주도주 중심의 정상 매매 가능."

    summary_key = f"{level}_{len(critical_alerts)}_{len(warning_alerts)}_" + "_".join([a["id"] for a in all_alerts[:3]])

    return {
        "level": level,
        "badge_text": badge_text,
        "badge_color": badge_color,
        "bg_color": bg_color,
        "border_color": border_color,
        "text_color": text_color,
        "title": title,
        "summary_text": summary_text,
        "summary_key": summary_key,
        "critical_count": len(critical_alerts),
        "warning_count": len(warning_alerts),
        "alerts": all_alerts,
        "action_guide": action_guide,
        "updated_at_str": updated_at_str,
        "triggered_time": triggered_time,
        "end_time": end_time,
        "duration_str": duration_str,
        "remaining_str": remaining_str,
        "alert_status": alert_status,
        "time_info": time_info,
    }


def get_realtime_market_alerts(force_refresh: bool = False) -> Dict[str, Any]:
    """
    Fetches latest domestic & US summary and evaluates real-time market warnings.
    """
    domestic_summary = fetch_domestic_indices_summary(force_refresh=force_refresh)
    us_summary = None
    try:
        us_summary = fetch_us_market_summary()
    except Exception as e:
        logger.warning(f"Could not load US summary for alerts: {e}")

    return evaluate_market_warning_alerts(domestic_summary=domestic_summary, us_summary=us_summary)
