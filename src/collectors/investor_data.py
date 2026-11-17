"""
Investor Data Collector: Fetches foreign and institutional net purchases (외국인·기관 순매매 동향).
Retrieves real-time/historical investor trends from Naver Finance mobile API:
- Foreigner net buying quantity (외국인 순매수량)
- Institutional net buying quantity (기관 순매수량)
- Individual net buying quantity (개인 순매수량)
- Foreign ownership ratio (외국인 지분율)
- Consecutive buying/selling days (연속 순매수/순매도 일수)
"""

import time
from typing import Dict, Any, List, Optional
import requests
import urllib3

try:
    import truststore
    truststore.inject_into_ssl()
except Exception:
    pass

urllib3.disable_warnings()

_INVESTOR_CACHE: Dict[str, Dict[str, Any]] = {}
_INVESTOR_CACHE_TS: Dict[str, float] = {}
CACHE_TTL_SECONDS = 600.0  # 10 minutes cache


def _parse_int(val: Any) -> int:
    if val is None or val == "":
        return 0
    clean = str(val).replace(",", "").replace("+", "").strip()
    try:
        return int(float(clean))
    except (ValueError, TypeError):
        return 0


def _parse_float(val: Any) -> float:
    if val is None or val == "":
        return 0.0
    clean = str(val).replace(",", "").replace("%", "").replace("+", "").strip()
    try:
        return float(clean)
    except (ValueError, TypeError):
        return 0.0


def fetch_investor_trend(code: str, limit: int = 10) -> Dict[str, Any]:
    """
    Fetches the latest investor buying/selling trend for a given stock code.
    Returns parsed metrics including foreign/institutional net buying,
    foreign ownership ratio, 3-day and 5-day cumulative net buying, and consecutive streak.
    Gracefully falls back to safe empty metrics if network is unavailable.
    """
    clean_code = str(code).strip().zfill(6)
    now = time.time()

    # In-memory cache check
    if clean_code in _INVESTOR_CACHE:
        cached_ts = _INVESTOR_CACHE_TS.get(clean_code, 0.0)
        if (now - cached_ts) < CACHE_TTL_SECONDS:
            return _INVESTOR_CACHE[clean_code]

    empty_fallback = {
        "code": clean_code,
        "has_data": False,
        "latest_bizdate": "",
        "foreigner_latest_buy": 0,
        "organ_latest_buy": 0,
        "individual_latest_buy": 0,
        "foreigner_hold_ratio": 0.0,
        "foreigner_3d_sum": 0,
        "organ_3d_sum": 0,
        "foreigner_5d_sum": 0,
        "organ_5d_sum": 0,
        "consecutive_foreigner_days": 0,
        "consecutive_organ_days": 0,
        "is_dual_buying": False,
        "is_dual_selling": False,
        "trend_list": [],
        "summary_text": "외국인·기관 순매매 데이터를 조회하는 중입니다."
    }

    url = f"https://m.stock.naver.com/api/stock/{clean_code}/trend"
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
        "Referer": f"https://m.stock.naver.com/domestic/stock/{clean_code}/trend"
    }

    try:
        resp = requests.get(url, headers=headers, verify=False, timeout=3.5)
        if resp.status_code != 200:
            return empty_fallback

        raw_data = resp.json()
        if not isinstance(raw_data, list) or len(raw_data) == 0:
            return empty_fallback

        parsed_list = []
        for row in raw_data[:limit]:
            bdate_raw = str(row.get("bizdate", ""))
            bdate_fmt = f"{bdate_raw[:4]}-{bdate_raw[4:6]}-{bdate_raw[6:8]}" if len(bdate_raw) == 8 else bdate_raw
            f_buy = _parse_int(row.get("foreignerPureBuyQuant"))
            o_buy = _parse_int(row.get("organPureBuyQuant"))
            i_buy = _parse_int(row.get("individualPureBuyQuant"))
            h_ratio = _parse_float(row.get("foreignerHoldRatio"))
            c_price = _parse_int(row.get("closePrice"))

            parsed_list.append({
                "date": bdate_fmt,
                "bizdate_raw": bdate_raw,
                "foreigner_buy": f_buy,
                "organ_buy": o_buy,
                "individual_buy": i_buy,
                "hold_ratio": h_ratio,
                "close_price": c_price
            })

        latest = parsed_list[0]
        f_latest = latest["foreigner_buy"]
        o_latest = latest["organ_buy"]
        i_latest = latest["individual_buy"]
        h_ratio_latest = latest["hold_ratio"]
        latest_date = latest["date"]

        # 3-day and 5-day sums
        f_3d = sum(p["foreigner_buy"] for p in parsed_list[:3])
        o_3d = sum(p["organ_buy"] for p in parsed_list[:3])
        f_5d = sum(p["foreigner_buy"] for p in parsed_list[:5])
        o_5d = sum(p["organ_buy"] for p in parsed_list[:5])

        # Consecutive streak calculation
        consec_f = 0
        if f_latest > 0:
            for p in parsed_list:
                if p["foreigner_buy"] > 0:
                    consec_f += 1
                else:
                    break
        elif f_latest < 0:
            for p in parsed_list:
                if p["foreigner_buy"] < 0:
                    consec_f -= 1
                else:
                    break

        consec_o = 0
        if o_latest > 0:
            for p in parsed_list:
                if p["organ_buy"] > 0:
                    consec_o += 1
                else:
                    break
        elif o_latest < 0:
            for p in parsed_list:
                if p["organ_buy"] < 0:
                    consec_o -= 1
                else:
                    break

        is_dual_buy = (f_latest > 0 and o_latest > 0)
        is_dual_sell = (f_latest < 0 and o_latest < 0)

        # Build natural Korean summary
        f_desc = f"외국인 {f_latest:+,.0f}주" + (f" ({abs(consec_f)}일 연속 순매수)" if consec_f >= 2 else (f" ({abs(consec_f)}일 연속 순매도)" if consec_f <= -2 else ""))
        o_desc = f"기관 {o_latest:+,.0f}주" + (f" ({abs(consec_o)}일 연속 순매수)" if consec_o >= 2 else (f" ({abs(consec_o)}일 연속 순매도)" if consec_o <= -2 else ""))

        if is_dual_buy:
            bias_phrase = "외인·기관 쌍끌이 순매수 유입 (강력한 수급 호재)"
        elif is_dual_sell:
            bias_phrase = "외인·기관 동반 순매도 출회 (수급 차익실현 주의)"
        elif f_latest > 0 and o_latest <= 0:
            bias_phrase = "외국인 단독 순매수 (기관 매물 소화 중)"
        elif f_latest <= 0 and o_latest > 0:
            bias_phrase = "기관 단독 순매수 (외인 매물 소화 중)"
        else:
            bias_phrase = "외인·기관 수급 관망세"

        summary_text = (
            f"최근 거래일 기준 {f_desc}, {o_desc}를 기록하며 [{bias_phrase}] 상태입니다. "
            f"(외국인 지분율: {h_ratio_latest:.2f}%, 최근 5일 누적: 외인 {f_5d:+,.0f}주 / 기관 {o_5d:+,.0f}주)"
        )

        result = {
            "code": clean_code,
            "has_data": True,
            "latest_bizdate": latest_date,
            "foreigner_latest_buy": f_latest,
            "organ_latest_buy": o_latest,
            "individual_latest_buy": i_latest,
            "foreigner_hold_ratio": h_ratio_latest,
            "foreigner_3d_sum": f_3d,
            "organ_3d_sum": o_3d,
            "foreigner_5d_sum": f_5d,
            "organ_5d_sum": o_5d,
            "consecutive_foreigner_days": consec_f,
            "consecutive_organ_days": consec_o,
            "is_dual_buying": is_dual_buy,
            "is_dual_selling": is_dual_sell,
            "trend_list": parsed_list,
            "summary_text": summary_text
        }

        _INVESTOR_CACHE[clean_code] = result
        _INVESTOR_CACHE_TS[clean_code] = now
        return result

    except Exception:
        return empty_fallback
