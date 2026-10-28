"""
US Market and Global Macro Data Collector & Quantitative Regime Analyzer.
Fetches US major indices (NASDAQ, S&P 500, SOXX, DOW), USD/KRW exchange rate, and VIX.
Provides caching and deterministic quant market regime classification.
"""

import os
import json
import logging
from datetime import datetime, timedelta
from typing import Dict, Any, Optional

try:
    import truststore
    truststore.inject_into_ssl()
except Exception:
    pass

import pandas as pd
import FinanceDataReader as fdr

from src.database.models import CACHE_DIR, get_now_kst

logger = logging.getLogger(__name__)

US_CACHE_FILE = os.path.join(CACHE_DIR, "us_market_summary.json")

# Target symbols in FinanceDataReader
US_SYMBOLS = {
    "NASDAQ": {"sym": "IXIC", "name": "나스닥 종합", "category": "tech"},
    "SP500": {"sym": "US500", "name": "S&P 500", "category": "broad"},
    "SOXX": {"sym": "SOXX", "name": "필라델피아 반도체", "category": "semiconductor"},
    "DOW": {"sym": "DJI", "name": "다우 존스", "category": "industrial"},
    "USDKRW": {"sym": "USD/KRW", "name": "원/달러 환율", "category": "fx"},
    "VIX": {"sym": "VIX", "name": "변동성지수(VIX)", "category": "volatility"},
}


def _safe_float(val: Any, default: float = 0.0) -> float:
    try:
        if val is None or val == "":
            return default
        return float(val)
    except (ValueError, TypeError):
        return default


def determine_global_market_regime(metrics: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
    """
    Evaluates quantitative global market regime based on US indices, SOXX, FX, and VIX.
    Returns:
        - regime: "우호", "중립", "경계", "위험"
        - regime_desc: brief summary description
        - color: badge color hex code
        - action_guide: practical trading guideline for Korean market open
        - sector_notes: sector-specific lead-lag impact notes
    """
    nasdaq_pct = metrics.get("NASDAQ", {}).get("change_pct", 0.0)
    sp500_pct = metrics.get("SP500", {}).get("change_pct", 0.0)
    soxx_pct = metrics.get("SOXX", {}).get("change_pct", 0.0)
    dow_pct = metrics.get("DOW", {}).get("change_pct", 0.0)
    usdkrw_rate = metrics.get("USDKRW", {}).get("close", 1340.0)
    usdkrw_pct = metrics.get("USDKRW", {}).get("change_pct", 0.0)
    vix_val = metrics.get("VIX", {}).get("close", 16.0)
    vix_pct = metrics.get("VIX", {}).get("change_pct", 0.0)

    # 1. Extreme Risk: Plunge or high volatility panic
    if nasdaq_pct < -1.7 or sp500_pct < -1.5 or soxx_pct < -3.0 or vix_val >= 23.0 or vix_pct > 18.0:
        regime = "위험"
        regime_desc = "글로벌 급락 및 위험회피 심화"
        color = "#DC2626"  # red
        action_guide = "지수 동반 갭하락 및 외인 투매 경계. 신규 진입을 전면 보류하거나 현금 비중 50% 이상 확보 권장"
    # 2. Caution: Weakness or pullback
    elif nasdaq_pct < -0.7 or sp500_pct < -0.6 or soxx_pct < -1.5 or usdkrw_rate >= 1390.0:
        regime = "경계"
        regime_desc = "미장 약세 조정 및 관망 국면"
        color = "#D97706"  # orange
        action_guide = "시초가 갭하락 주의. 장 초반 무리한 추격매수 금지, 09:30 이후 하방 지지 확인 후 보수적 분할 매수"
    # 3. Favorable: Strong bull momentum
    elif (nasdaq_pct >= 0.5 and sp500_pct >= 0.3) or soxx_pct >= 1.5:
        regime = "우호"
        regime_desc = "글로벌 위험선호 및 상승 탄력 우수"
        color = "#059669"  # green
        action_guide = "글로벌 훈풍 지속. 단, +4% 초과 과도한 갭상승 종목은 윗꼬리 차익실현 경계, 주도 섹터 눌림목 공략"
    # 4. Neutral: Mixed
    else:
        regime = "중립"
        regime_desc = "글로벌 혼조세 및 지수 영향 제한적"
        color = "#475569"  # slate
        action_guide = "지수 영향 제한적. 개별 종목 고유 모멘텀 및 당일 거래대금 상위 주도주 위주 선별 공략"

    # Sector specific impact assessment
    sector_notes = {}

    # Semiconductor note
    if soxx_pct <= -2.5:
        sector_notes["반도체"] = f"필라델피아 반도체({soxx_pct:+.2f}%) 급락으로 국내 반도체/소부장 시초가 갭하락 및 외인 매도 압력 주의"
    elif soxx_pct <= -1.0:
        sector_notes["반도체"] = f"필라델피아 반도체({soxx_pct:+.2f}%) 약세로 시초가 보수적 대응 및 장중 지지선 확인 필요"
    elif soxx_pct >= 2.0:
        sector_notes["반도체"] = f"필라델피아 반도체({soxx_pct:+.2f}%) 강세로 국내 반도체/소부장 갭상승 출발 우호"
    else:
        sector_notes["반도체"] = f"필라델피아 반도체({soxx_pct:+.2f}%) 보합권으로 개별 실적 및 수급 위주 등락 전망"

    # FX note
    if usdkrw_pct >= 0.8 or usdkrw_rate >= 1380.0:
        sector_notes["외인수급"] = f"원/달러 환율({usdkrw_rate:,.1f}원, {usdkrw_pct:+.2f}%) 급등세로 외국인 수급 이탈 경계"
    elif usdkrw_pct <= -0.5:
        sector_notes["외인수급"] = f"원/달러 환율({usdkrw_rate:,.1f}원, {usdkrw_pct:+.2f}%) 하락 안정세로 외국인 순매수 유입 우호"
    else:
        sector_notes["외인수급"] = f"원/달러 환율({usdkrw_rate:,.1f}원, {usdkrw_pct:+.2f}%) 안정권 유지 중"

    # Tech note
    if nasdaq_pct <= -1.5:
        sector_notes["기술주"] = f"나스닥({nasdaq_pct:+.2f}%) 급락으로 국내 성장주/IT/소프트웨어 단기 차익 매물 출회 가능성"
    elif nasdaq_pct >= 1.0:
        sector_notes["기술주"] = f"나스닥({nasdaq_pct:+.2f}%) 강세로 AI/소프트웨어/성장주 투자심리 개선"

    return {
        "regime": regime,
        "regime_desc": regime_desc,
        "color": color,
        "action_guide": action_guide,
        "sector_notes": sector_notes,
        "nasdaq_pct": nasdaq_pct,
        "sp500_pct": sp500_pct,
        "soxx_pct": soxx_pct,
        "dow_pct": dow_pct,
        "usdkrw_rate": usdkrw_rate,
        "usdkrw_pct": usdkrw_pct,
        "vix_val": vix_val,
        "vix_pct": vix_pct,
    }


def fetch_us_market_summary(force_refresh: bool = False) -> Dict[str, Any]:
    """
    Fetches latest US market metrics from FinanceDataReader or local cache.
    Includes fallback data if offline or error occurs to guarantee zero crashes.
    """
    now_kst = get_now_kst()
    today_str = now_kst.strftime("%Y-%m-%d")

    # 1. Check local cache
    if not force_refresh and os.path.exists(US_CACHE_FILE):
        try:
            with open(US_CACHE_FILE, "r", encoding="utf-8") as f:
                cached = json.load(f)
            cached_time = cached.get("updated_at", "")
            # Cache valid for 30 minutes, or if created today outside US market trading hours
            if cached_time:
                c_dt = datetime.fromisoformat(cached_time)
                age_minutes = (now_kst - c_dt).total_seconds() / 60.0
                if age_minutes < 30.0:
                    return cached
                # If US market is closed (KST 06:00 to 22:30) and cache is from today, reuse it
                if 6 <= now_kst.hour < 22 and c_dt.strftime("%Y-%m-%d") == today_str:
                    return cached
        except Exception as e:
            logger.warning(f"Failed to read US market cache: {e}")

    # 2. Fetch live data for each symbol
    metrics: Dict[str, Dict[str, Any]] = {}
    fetch_success = False

    for key, info in US_SYMBOLS.items():
        sym = info["sym"]
        name = info["name"]
        cat = info["category"]
        try:
            df = fdr.DataReader(sym)
            if df is not None and not df.empty and len(df) >= 2:
                last_row = df.iloc[-1]
                prev_row = df.iloc[-2]
                close = _safe_float(last_row.get("Close"))
                prev_close = _safe_float(prev_row.get("Close"))
                change = close - prev_close
                change_pct = (change / prev_close * 100.0) if prev_close > 0 else 0.0
                trade_date = df.index[-1].strftime("%Y-%m-%d")

                metrics[key] = {
                    "symbol": sym,
                    "name": name,
                    "category": cat,
                    "close": round(close, 2),
                    "prev_close": round(prev_close, 2),
                    "change": round(change, 2),
                    "change_pct": round(change_pct, 2),
                    "trade_date": trade_date
                }
                fetch_success = True
        except Exception as e:
            logger.warning(f"Error fetching US market symbol {sym}: {e}")

    # 3. Handle fallback if fetch failed or incomplete
    if not fetch_success or "NASDAQ" not in metrics:
        # Check if old cache exists as fallback
        if os.path.exists(US_CACHE_FILE):
            try:
                with open(US_CACHE_FILE, "r", encoding="utf-8") as f:
                    return json.load(f)
            except Exception:
                pass

        # Safe default fallback
        metrics = {
            "NASDAQ": {"symbol": "IXIC", "name": "나스닥 종합", "category": "tech", "close": 18000.0, "prev_close": 18000.0, "change": 0.0, "change_pct": 0.0, "trade_date": today_str},
            "SP500": {"symbol": "US500", "name": "S&P 500", "category": "broad", "close": 5800.0, "prev_close": 5800.0, "change": 0.0, "change_pct": 0.0, "trade_date": today_str},
            "SOXX": {"symbol": "SOXX", "name": "필라델피아 반도체", "category": "semiconductor", "close": 520.0, "prev_close": 520.0, "change": 0.0, "change_pct": 0.0, "trade_date": today_str},
            "DOW": {"symbol": "DJI", "name": "다우 존스", "category": "industrial", "close": 42000.0, "prev_close": 42000.0, "change": 0.0, "change_pct": 0.0, "trade_date": today_str},
            "USDKRW": {"symbol": "USD/KRW", "name": "원/달러 환율", "category": "fx", "close": 1340.0, "prev_close": 1340.0, "change": 0.0, "change_pct": 0.0, "trade_date": today_str},
            "VIX": {"symbol": "VIX", "name": "변동성지수(VIX)", "category": "volatility", "close": 16.0, "prev_close": 16.0, "change": 0.0, "change_pct": 0.0, "trade_date": today_str},
        }

    # 4. Determine quantitative regime
    regime_info = determine_global_market_regime(metrics)

    summary_result = {
        "metrics": metrics,
        "regime": regime_info["regime"],
        "regime_desc": regime_info["regime_desc"],
        "color": regime_info["color"],
        "action_guide": regime_info["action_guide"],
        "sector_notes": regime_info["sector_notes"],
        "nasdaq_pct": regime_info["nasdaq_pct"],
        "sp500_pct": regime_info["sp500_pct"],
        "soxx_pct": regime_info["soxx_pct"],
        "dow_pct": regime_info["dow_pct"],
        "usdkrw_rate": regime_info["usdkrw_rate"],
        "usdkrw_pct": regime_info["usdkrw_pct"],
        "vix_val": regime_info["vix_val"],
        "vix_pct": regime_info["vix_pct"],
        "updated_at": now_kst.isoformat(),
        "date_str": now_kst.strftime("%Y-%m-%d %H:%M KST")
    }

    # 5. Save to cache
    try:
        os.makedirs(os.path.dirname(US_CACHE_FILE), exist_ok=True)
        with open(US_CACHE_FILE, "w", encoding="utf-8") as f:
            json.dump(summary_result, f, ensure_ascii=False, indent=2)
    except Exception as e:
        logger.warning(f"Failed to write US market cache: {e}")

    return summary_result


def get_sector_macro_note(sector_name: str, us_summary: Optional[Dict[str, Any]] = None) -> str:
    """
    Returns relevant macro correlation note for given domestic sector name.
    """
    if us_summary is None:
        us_summary = fetch_us_market_summary()

    notes = us_summary.get("sector_notes", {})
    sec = str(sector_name or "").lower()

    if any(k in sec for k in ["반도체", "it", "전자", "전기", "하드웨어"]):
        return notes.get("반도체", "")
    elif any(k in sec for k in ["제약", "바이오", "생명", "헬스"]):
        return notes.get("기술주", "")
    elif any(k in sec for k in ["2차전지", "배터리", "에너지", "화학"]):
        return notes.get("기술주", "")
    elif any(k in sec for k in ["금융", "지주", "은행", "증권"]):
        return notes.get("외인수급", "")

    return ""

def get_premarket_analysis_context(us_summary: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """
    Returns quantitative pre-market and global macro context for KOSPI/KOSDAQ market open:
    - regime: '우호', '중립', '경계', '위험'
    - score_mod: Score modifier between -5.0 and +3.0
    - gap_risk: '낮음 (우호)', '보통 (중립)', '주의 (경계)', '높음 (위험)'
    - gap_guide: practical open gap guideline (+1.5%~+4.5%)
    - nasdaq_pct, sp500_pct, soxx_pct, vix_val, usdkrw_rate, action_guide
    """
    if us_summary is None:
        us_summary = fetch_us_market_summary()

    regime = us_summary.get("regime", "중립")
    nasdaq = us_summary.get("nasdaq_pct", 0.0)
    sp500 = us_summary.get("sp500_pct", 0.0)
    soxx = us_summary.get("soxx_pct", 0.0)
    vix = us_summary.get("vix_val", 16.0)
    usdkrw = us_summary.get("usdkrw_rate", 1340.0)

    if regime == "우호":
        score_mod = 3.0
        gap_risk = "낮음 (우호)"
        gap_guide = "글로벌 훈풍으로 시초 갭상승 종목의 연속 슈팅 확률 우수 (적정 갭 +1.5%~+4.5% 공략, +5% 초과 과열은 윗꼬리 차익실현 경계)"
    elif regime == "경계":
        score_mod = -2.0
        gap_risk = "주의 (경계)"
        gap_guide = "미장 약세 조정 영향으로 시초 갭상승 후 음봉 전환(트랩) 주의 (무리한 시초가 추격 금지, 09:20 이후 저점 지지 확인 필수)"
    elif regime == "위험":
        score_mod = -5.0
        gap_risk = "높음 (위험)"
        gap_guide = "글로벌 급락 및 위험회피 고조 (시초 갭상승 추격매수 절대 금지, 20일선 첫 지지 반등 및 보수적 관망 권장)"
    else:
        score_mod = 0.0
        gap_risk = "보통 (중립)"
        gap_guide = "지수 영향 중립 (전일 800억~1,000억+ 폭발 주도 대장주 중심 적정 갭 +1.5%~+4.0% 선별 공략)"

    return {
        "regime": regime,
        "score_mod": score_mod,
        "gap_risk": gap_risk,
        "gap_guide": gap_guide,
        "nasdaq_pct": nasdaq,
        "sp500_pct": sp500,
        "soxx_pct": soxx,
        "vix_val": vix,
        "usdkrw_rate": usdkrw,
        "action_guide": us_summary.get("action_guide", "")
    }

