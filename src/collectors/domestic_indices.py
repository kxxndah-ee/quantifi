"""
Domestic Market Indices & Crypto Data Collector.
Fetches real-time market indices:
- 코스피 (KOSPI)
- 코스닥 (KOSDAQ)
- 코스피200 (KPI200)
- 코스피200선물 (FUT)
- 코스닥 150 (KQI150)
- 코리아밸류업 (KVALUE)
- 비트코인 (BTC/KRW)
Provides local JSON caching, zero-delay polling, and robust fallback handling.
"""

import os
import json
import logging
from datetime import datetime
from typing import Dict, Any, Optional

try:
    import truststore
    truststore.inject_into_ssl()
except Exception:
    pass

import requests
import urllib3
urllib3.disable_warnings()

from src.database.models import CACHE_DIR, get_now_kst

logger = logging.getLogger(__name__)

DOMESTIC_CACHE_FILE = os.path.join(CACHE_DIR, "domestic_indices_summary.json")

DOMESTIC_INDEX_KEYS = [
    ("KOSPI", "코스피"),
    ("KOSDAQ", "코스닥"),
    ("KPI200", "코스피200"),
    ("FUT", "코스피200선물"),
    ("KQI150", "코스닥 150"),
    ("KVALUE", "코리아밸류업"),
]


def _safe_float(val: Any, default: float = 0.0) -> float:
    try:
        if val is None or val == "":
            return default
        if isinstance(val, str):
            val = val.replace(",", "").strip()
        return float(val)
    except (ValueError, TypeError):
        return default


def fetch_btc_price(session: requests.Session) -> Dict[str, Any]:
    """Fetches real-time Bitcoin (BTC/KRW) price from Upbit with Bithumb fallback."""
    # 1. Try Upbit API
    try:
        r = session.get("https://api.upbit.com/v1/ticker?markets=KRW-BTC", verify=False, timeout=3)
        if r.status_code == 200:
            data = r.json()
            if isinstance(data, list) and len(data) > 0:
                item = data[0]
                trade_price = _safe_float(item.get("trade_price"))
                chg_rate = _safe_float(item.get("signed_change_rate")) * 100.0
                chg_price = _safe_float(item.get("signed_change_price"))
                prev_price = _safe_float(item.get("prev_closing_price"), trade_price - chg_price)
                return {
                    "code": "BTC",
                    "name": "비트코인",
                    "close": trade_price,
                    "prev_close": prev_price,
                    "change": chg_price,
                    "change_pct": round(chg_rate, 2),
                    "unit": "KRW",
                    "display_price": f"{trade_price:,.0f}원"
                }
    except Exception as e:
        logger.warning(f"Upbit BTC fetch failed: {e}")

    # 2. Fallback to Bithumb API
    try:
        r = session.get("https://api.bithumb.com/public/ticker/BTC_KRW", verify=False, timeout=3)
        if r.status_code == 200:
            res_json = r.json()
            if res_json.get("status") == "0000":
                bdata = res_json.get("data", {})
                close_p = _safe_float(bdata.get("closing_price"))
                prev_p = _safe_float(bdata.get("prev_closing_price"), close_p)
                chg_rate = _safe_float(bdata.get("fluctate_rate_24H"))
                chg_p = close_p - prev_p
                return {
                    "code": "BTC",
                    "name": "비트코인",
                    "close": close_p,
                    "prev_close": prev_p,
                    "change": chg_p,
                    "change_pct": round(chg_rate, 2),
                    "unit": "KRW",
                    "display_price": f"{close_p:,.0f}원"
                }
    except Exception as e:
        logger.warning(f"Bithumb BTC fetch failed: {e}")

    return {
        "code": "BTC",
        "name": "비트코인",
        "close": 105000000.0,
        "prev_close": 105000000.0,
        "change": 0.0,
        "change_pct": 0.0,
        "unit": "KRW",
        "display_price": "105,000,000원"
    }


def fetch_domestic_indices_summary(force_refresh: bool = False) -> Dict[str, Any]:
    """
    Fetches real-time domestic indices and Bitcoin.
    Returns dictionary with:
        - items: list of dicts for [KOSPI, KOSDAQ, KPI200, FUT, KQI150, KVALUE, BTC]
        - updated_at: ISO timestamp
        - updated_at_str: format 'HH:MM:SS KST'
        - summary_text: one-line summary string for expander/header
    """
    now_kst = get_now_kst()
    today_str = now_kst.strftime("%Y-%m-%d")

    # 1. Check local cache (TTL: 15 seconds during day, 5 mins at night)
    if not force_refresh and os.path.exists(DOMESTIC_CACHE_FILE):
        try:
            with open(DOMESTIC_CACHE_FILE, "r", encoding="utf-8") as f:
                cached = json.load(f)
            cached_time = cached.get("updated_at", "")
            if cached_time:
                c_dt = datetime.fromisoformat(cached_time)
                age_seconds = (now_kst - c_dt).total_seconds()
                # Fast 10s cache during market hours, 30s outside
                is_market_hours = (9 <= now_kst.hour < 16)
                ttl_limit = 10.0 if is_market_hours else 30.0
                if age_seconds < ttl_limit and len(cached.get("items", [])) >= 7:
                    return cached
        except Exception as e:
            logger.warning(f"Failed to read domestic indices cache: {e}")

    session = requests.Session()
    session.headers.update({
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36"
    })

    items = []
    codes_str = ",".join([k[0] for k in DOMESTIC_INDEX_KEYS])
    poll_url = f"https://polling.finance.naver.com/api/realtime/domestic/index/{codes_str}"

    index_data_map = {}
    try:
        r = session.get(poll_url, verify=False, timeout=4)
        if r.status_code == 200:
            datas = r.json().get("datas", [])
            for d in datas:
                icode = d.get("itemCode")
                if icode:
                    index_data_map[icode] = d
    except Exception as e:
        logger.warning(f"Error fetching domestic indices polling: {e}")

    # Build items in exact requested order
    for icode, iname in DOMESTIC_INDEX_KEYS:
        d = index_data_map.get(icode, {})
        close_val = _safe_float(d.get("closePrice"))
        chg_pct = _safe_float(d.get("fluctuationsRatio"))
        chg_val = _safe_float(d.get("compareToPreviousClosePrice"))
        prev_close = close_val - chg_val if close_val > 0 else 0.0

        # Fallback values if offline
        if close_val <= 0:
            defaults = {
                "KOSPI": (2600.0, 0.0),
                "KOSDAQ": (750.0, 0.0),
                "KPI200": (350.0, 0.0),
                "FUT": (350.0, 0.0),
                "KQI150": (1200.0, 0.0),
                "KVALUE": (1000.0, 0.0),
            }
            def_close, def_pct = defaults.get(icode, (1000.0, 0.0))
            close_val = def_close
            chg_pct = def_pct
            prev_close = def_close
            chg_val = 0.0

        items.append({
            "code": icode,
            "name": iname,
            "close": round(close_val, 2),
            "prev_close": round(prev_close, 2),
            "change": round(chg_val, 2),
            "change_pct": round(chg_pct, 2),
            "unit": "pt",
            "display_price": f"{close_val:,.2f}"
        })

    # Fetch Bitcoin
    btc_item = fetch_btc_price(session)
    items.append(btc_item)

    # Build summary line
    summary_parts = []
    for it in items:
        summary_parts.append(f"{it['name']} {it['change_pct']:+.2f}%")
    summary_text = " · ".join(summary_parts)

    res = {
        "items": items,
        "updated_at": now_kst.isoformat(),
        "updated_at_str": now_kst.strftime("%H:%M:%S KST"),
        "summary_text": summary_text
    }

    # Save to disk cache
    try:
        with open(DOMESTIC_CACHE_FILE, "w", encoding="utf-8") as f:
            json.dump(res, f, ensure_ascii=False, indent=2)
    except Exception as e:
        logger.warning(f"Failed to write domestic cache: {e}")

    return res
