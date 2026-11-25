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


def get_sector_macro_boost(sector_name: str, premarket_ctx: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """
    Computes pinpoint score boost/penalty and directional macro guideline for a domestic sector
    based on US leading indicators (SOXX, NASDAQ, USD/KRW, VIX).
    Returns:
        - boost: float
        - tag: str
        - status: "STRONG_BOOST", "BOOST", "NEUTRAL", "CAUTION", "PENALTY"
        - guide: str
    """
    if premarket_ctx is None:
        premarket_ctx = get_premarket_analysis_context()

    soxx_pct = float(premarket_ctx.get("soxx_pct", 0.0))
    nasdaq_pct = float(premarket_ctx.get("nasdaq_pct", 0.0))
    usdkrw_rate = float(premarket_ctx.get("usdkrw_rate", 1340.0))
    vix_val = float(premarket_ctx.get("vix_val", 16.0))

    sec = str(sector_name or "").lower()

    # 1. Semiconductor & Hardware: linked to SOXX
    if any(k in sec for k in ["반도체", "소부장", "it부품", "전자부품", "하드웨어"]):
        if soxx_pct >= 2.0:
            return {
                "boost": 4.0,
                "tag": f"SOXX 랠리 훈풍 (+4.0점)",
                "status": "STRONG_BOOST",
                "guide": f"필라델피아 반도체 지수({soxx_pct:+.2f}%) 폭등으로 국내 반도체 대장주 시초가 적정 갭상승 및 장초반 2차 슈팅 탄력 최우수"
            }
        elif soxx_pct >= 0.7:
            return {
                "boost": 2.0,
                "tag": f"SOXX 반도체 양호 (+2.0점)",
                "status": "BOOST",
                "guide": f"필라델피아 반도체 지수({soxx_pct:+.2f}%) 상승세로 반도체 소부장 시초 갭상승 우호"
            }
        elif soxx_pct <= -2.5:
            return {
                "boost": -5.0,
                "tag": f"SOXX 반도체 급락 경계 (-5.0점)",
                "status": "PENALTY",
                "guide": f"필라델피아 반도체 지수({soxx_pct:+.2f}%) 급락으로 국내 반도체 섹터 동반 갭하락 및 외인 매도세 주의"
            }
        elif soxx_pct <= -1.0:
            return {
                "boost": -2.5,
                "tag": f"SOXX 반도체 조정 (-2.5점)",
                "status": "CAUTION",
                "guide": f"필라델피아 반도체 지수({soxx_pct:+.2f}%) 약세로 시초가 보수적 분할 대응 필요"
            }
        else:
            return {
                "boost": 0.0,
                "tag": "SOXX 반도체 중립",
                "status": "NEUTRAL",
                "guide": "필라델피아 반도체 보합권 유지로 개별 수급 위주 등락"
            }

    # 2. Software, Platform, AI, Tech, Bio: linked to NASDAQ
    elif any(k in sec for k in ["소프트웨어", "it서비스", "인터넷", "게임", "미디어", "인공지능", "ai", "바이오", "제약", "생명공학", "헬스케어"]):
        if nasdaq_pct >= 1.5:
            return {
                "boost": 3.0,
                "tag": f"나스닥 기술주 랠리 (+3.0점)",
                "status": "BOOST",
                "guide": f"미 나스닥({nasdaq_pct:+.2f}%) 강세로 국내 성장주/IT/바이오 투자심리 개선 및 시초 갭 안착 우호"
            }
        elif nasdaq_pct <= -1.7:
            return {
                "boost": -4.0,
                "tag": f"나스닥 기술주 급락 경계 (-4.0점)",
                "status": "PENALTY",
                "guide": f"미 나스닥({nasdaq_pct:+.2f}%) 급락으로 국내 성장주 밸류에이션 부담 및 차익 매물 주의"
            }
        elif nasdaq_pct <= -0.8:
            return {
                "boost": -1.5,
                "tag": f"나스닥 기술주 약세 (-1.5점)",
                "status": "CAUTION",
                "guide": f"미 나스닥({nasdaq_pct:+.2f}%) 약세로 기술주 시초가 추격 자제"
            }
        else:
            return {
                "boost": 0.0,
                "tag": "나스닥 기술주 중립",
                "status": "NEUTRAL",
                "guide": "나스닥 보합권으로 개별 재료 및 당일 수급 중심 대응"
            }

    # 3. Finance, Large Cap, Holdings: linked to FX (USD/KRW)
    elif any(k in sec for k in ["금융", "지주", "은행", "증권", "보험"]):
        if usdkrw_rate >= 1420.0 or vix_val >= 22.0:
            return {
                "boost": -3.0,
                "tag": f"환율/VIX 외인 이탈 주의 (-3.0점)",
                "status": "CAUTION",
                "guide": f"환율({usdkrw_rate:,.1f}원) 상승 및 VIX({vix_val:.1f}) 급등으로 대형 금융주 외인 패시브 매도 압력 주의"
            }
        elif usdkrw_rate <= 1350.0 and vix_val <= 16.0:
            return {
                "boost": 2.0,
                "tag": f"환율 안정 외인 우호 (+2.0점)",
                "status": "BOOST",
                "guide": f"환율({usdkrw_rate:,.1f}원) 및 VIX 안정세로 대형 금융주 외인 순매수 유입 우호"
            }
        else:
            return {
                "boost": 0.0,
                "tag": "금융 매크로 중립",
                "status": "NEUTRAL",
                "guide": "환율 및 거시경제 지표 보합권"
            }

    # 4. Default / General Sectors
    return {
        "boost": 0.0,
        "tag": "섹터 매크로 중립",
        "status": "NEUTRAL",
        "guide": "글로벌 매크로 지수 영향 제한적, 개별 종목 수급 및 차트 집중"
    }


def detect_premarket_gap_anomaly(
    stock_sector: str,
    open_pct: float,
    premarket_ctx: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """
    Evaluates whether the domestic stock's open gap (+% from previous close) aligns with or contradicts
    overnight US leading market conditions, catching Gap Traps (fake spikes) and Golden Alignments.
    Returns:
        - is_gap_trap: bool
        - is_momentum_aligned: bool
        - anomaly_tag: str
        - anomaly_guide: str
        - score_adjustment: float
    """
    if premarket_ctx is None:
        premarket_ctx = get_premarket_analysis_context()

    regime = premarket_ctx.get("regime", "중립")
    soxx_pct = float(premarket_ctx.get("soxx_pct", 0.0))
    nasdaq_pct = float(premarket_ctx.get("nasdaq_pct", 0.0))

    sec = str(stock_sector or "").lower()
    is_semi = any(k in sec for k in ["반도체", "소부장", "it부품", "전자부품"])
    is_tech = any(k in sec for k in ["소프트웨어", "it서비스", "인터넷", "게임", "바이오"])

    # 1. Bear Gap Trap: US plunged, but domestic stock gaps up abnormally high -> High risk of opening dump!
    if (regime in ("위험", "경계")) or (is_semi and soxx_pct <= -2.0) or (is_tech and nasdaq_pct <= -1.8):
        if open_pct >= 3.5:
            return {
                "is_gap_trap": True,
                "is_momentum_aligned": False,
                "anomaly_tag": "[미장 역행 갭트랩 경고]",
                "anomaly_guide": f"간밤 미장/SOXX 급락({soxx_pct:+.1f}%)에도 불구하고 시초가가 +{open_pct:.1f}% 이상 무리하게 높게 형성되었습니다. 개장 직후 외인/기관의 시초가 차익실현 및 윗꼬리 장대음봉 전환 위험이 극히 높으므로 시초가 진입을 전면 배제합니다.",
                "score_adjustment": -8.0
            }

    # 2. Golden Momentum Alignment: US surged and domestic stock opens with solid appropriate gap
    if (regime == "우호") or (is_semi and soxx_pct >= 1.5) or (is_tech and nasdaq_pct >= 1.2):
        if 1.5 <= open_pct <= 4.5:
            return {
                "is_gap_trap": False,
                "is_momentum_aligned": True,
                "anomaly_tag": "[글로벌 모멘텀 일치]",
                "anomaly_guide": f"미장 주도 섹터 호조와 시초가 적정 갭상승(+{open_pct:.1f}%)이 완벽히 일치합니다. 09:00 시초가 체결 후 장초반 강력한 2차 슈팅 및 익절 확률이 극대화된 주도주 구간입니다.",
                "score_adjustment": 3.0
            }
        elif open_pct > 5.0:
            return {
                "is_gap_trap": False,
                "is_momentum_aligned": False,
                "anomaly_tag": "[시초가 과열 주의]",
                "anomaly_guide": f"글로벌 훈풍은 우수하나 시초가 갭(+{open_pct:.1f}%)이 +5%를 초과하여 장초반 단기 과열권입니다. 시초가 추격매수 대신 눌림목 재확인이 필수적입니다.",
                "score_adjustment": -2.0
            }

    # 3. Healthy Pullback Rebound Open
    if -1.5 <= open_pct <= 1.0:
        return {
            "is_gap_trap": False,
            "is_momentum_aligned": False,
            "anomaly_tag": "[보합권 건전 시초가]",
            "anomaly_guide": f"시초가({open_pct:+.1f}%)가 보합권에서 차분하게 출발하여 20일선 눌림목 첫 지지 반등을 노리기에 안정적인 손익비를 제공합니다.",
            "score_adjustment": 1.0
        }

    return {
        "is_gap_trap": False,
        "is_momentum_aligned": False,
        "anomaly_tag": "시초가 중립 체결",
        "anomaly_guide": f"시초가 갭 {open_pct:+.1f}%로 일반적인 수급 범위 내 출발입니다.",
        "score_adjustment": 0.0
    }

