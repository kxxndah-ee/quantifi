"""
Market Data Collector with local parquet caching and incremental update support.
Uses FinanceDataReader with truststore SSL injection for robust Windows execution.
"""

import os
import sys
from datetime import datetime, timedelta
from typing import Optional, Dict, Any

try:
    import truststore
    truststore.inject_into_ssl()
except Exception:
    pass

import pandas as pd
import numpy as np
import FinanceDataReader as fdr

from src.database.models import CACHE_DIR, get_now_kst

BENCHMARK_CODE = "069500"  # KODEX 200


def get_cache_path(code: str) -> str:
    """Returns absolute path to cached parquet file for given stock code."""
    return os.path.join(CACHE_DIR, f"{code}.parquet")


import requests
import urllib3
urllib3.disable_warnings()


def fetch_realtime_quotes(codes: list) -> Dict[str, Dict[str, Any]]:
    """
    Fetches zero-delay real-time domestic stock quotes from Naver Finance realtime polling API.
    Supports batch querying in chunks of 50 codes for maximum performance (~0.1s per 50 stocks).
    Returns dict: code -> quote_dict containing:
        - code: str
        - name: str
        - close: float
        - open: float
        - high: float
        - low: float
        - prev_close: float
        - change_pct: float
        - open_pct: float
        - high_pct: float
        - low_pct: float
        - volume: int
        - market_status: str
        - date: str (YYYY-MM-DD)
    """
    if not codes:
        return {}

    clean_codes = [c.strip().zfill(6) for c in codes if c and c.strip()]
    unique_codes = list(dict.fromkeys(clean_codes))
    result: Dict[str, Dict[str, Any]] = {}

    chunk_size = 100
    chunks = [unique_codes[i:i + chunk_size] for i in range(0, len(unique_codes), chunk_size)]

    session = requests.Session()
    adapter = requests.adapters.HTTPAdapter(pool_connections=15, pool_maxsize=15, max_retries=2)
    session.mount("https://", adapter)

    def _fetch_chunk(chunk: list) -> list:
        url = "https://polling.finance.naver.com/api/realtime/domestic/stock/" + ",".join(chunk)
        for _ in range(2):
            try:
                r = session.get(url, verify=False, timeout=4)
                if r.status_code == 200:
                    return r.json().get("datas", [])
            except Exception:
                pass
        return []

    from concurrent.futures import ThreadPoolExecutor
    with ThreadPoolExecutor(max_workers=min(14, len(chunks) or 1)) as executor:
        chunk_results = list(executor.map(_fetch_chunk, chunks))

    def _safe_float(val, default=0.0) -> float:
        try:
            if val is None or val == "":
                return default
            return float(val)
        except (ValueError, TypeError):
            return default

    def _safe_int(val, default=0) -> int:
        try:
            if val is None or val == "":
                return default
            return int(float(val))
        except (ValueError, TypeError):
            return default

    now_kst_str = get_now_kst().strftime("%Y-%m-%d")
    for datas in chunk_results:
        for item in datas:
            try:
                code = item.get("itemCode")
                if not code:
                    continue
                close_p = _safe_float(item.get("closePriceRaw"), 0.0)
                diff_p = _safe_float(item.get("compareToPreviousClosePriceRaw"), 0.0)
                prev_p = close_p - diff_p if diff_p != 0 else _safe_float(item.get("closePriceRaw"), 0.0)
                chg_pct = _safe_float(item.get("fluctuationsRatioRaw"), 0.0)
                if prev_p <= 0 and close_p > 0 and chg_pct != 0:
                    prev_p = close_p / (1.0 + chg_pct / 100.0)
                elif prev_p <= 0:
                    prev_p = close_p

                open_p = _safe_float(item.get("openPriceRaw"), close_p)
                high_p = _safe_float(item.get("highPriceRaw"), close_p)
                low_p = _safe_float(item.get("lowPriceRaw"), close_p)
                vol_p = _safe_int(item.get("accumulatedTradingVolumeRaw"), 0)
                val_raw = _safe_float(item.get("accumulatedTradingValueRaw"), 0.0)

                open_pct = ((open_p - prev_p) / prev_p * 100.0) if prev_p > 0 else 0.0
                high_pct = ((high_p - prev_p) / prev_p * 100.0) if prev_p > 0 else 0.0
                low_pct = ((low_p - prev_p) / prev_p * 100.0) if prev_p > 0 else 0.0

                local_time = item.get("localTradedAt", "")
                date_str = local_time[:10] if local_time else now_kst_str

                raw_sname = item.get("stockName", "")
                try:
                    from src.collectors.krx_universe import normalize_stock_name
                    clean_sname = normalize_stock_name(raw_sname, code)
                except Exception:
                    clean_sname = raw_sname

                result[code] = {
                    "code": code,
                    "name": clean_sname,
                    "close": close_p,
                    "open": open_p,
                    "high": high_p,
                    "low": low_p,
                    "prev_close": prev_p,
                    "change_pct": chg_pct,
                    "open_pct": open_pct,
                    "high_pct": high_pct,
                    "low_pct": low_pct,
                    "volume": vol_p,
                    "trading_val_raw": val_raw,
                    "market_status": item.get("marketStatus", ""),
                    "date": date_str
                }
            except Exception:
                continue

    return result


_LAST_COMPLETED_TRADING_DAY: Optional[str] = None
_LAST_COMPLETED_CHECK_TIME: float = 0.0


def get_last_completed_trading_day() -> str:
    """
    Returns the most recent completed market trading date string (YYYY-MM-DD) before today.
    Uses cached KODEX 200 benchmark dates to accurately account for market holidays and weekends.
    """
    global _LAST_COMPLETED_TRADING_DAY, _LAST_COMPLETED_CHECK_TIME
    import time
    now = time.time()
    now_kst = get_now_kst()
    today_str = now_kst.strftime("%Y-%m-%d")

    if _LAST_COMPLETED_TRADING_DAY and (now - _LAST_COMPLETED_CHECK_TIME < 300):
        return _LAST_COMPLETED_TRADING_DAY

    try:
        cache_path = get_cache_path(BENCHMARK_CODE)
        if os.path.exists(cache_path) and os.path.getsize(cache_path) > 0:
            bm = pd.read_parquet(cache_path)
            if not bm.empty:
                bm.index = pd.to_datetime(bm.index)
                bm_past = bm[bm.index.strftime("%Y-%m-%d") < today_str]
                if not bm_past.empty:
                    _LAST_COMPLETED_TRADING_DAY = bm_past.index.max().strftime("%Y-%m-%d")
                    _LAST_COMPLETED_CHECK_TIME = now
                    return _LAST_COMPLETED_TRADING_DAY
    except Exception:
        pass

    # Fallback to weekday subtraction
    dt = now_kst - timedelta(days=1)
    while dt.weekday() >= 5:
        dt -= timedelta(days=1)
    _LAST_COMPLETED_TRADING_DAY = dt.strftime("%Y-%m-%d")
    _LAST_COMPLETED_CHECK_TIME = now
    return _LAST_COMPLETED_TRADING_DAY


def fetch_ohlcv(
    code: str,
    start: str = "2017-01-01",
    end: Optional[str] = None,
    force_refresh: bool = False
) -> pd.DataFrame:
    """
    Fetches adjusted OHLCV for given stock code.
    Loads from parquet cache if present, fetching only missing dates incrementally.
    During live market hours, today's bar is refreshed in real time to match HTS.
    """
    cache_path = get_cache_path(code)
    now_kst = get_now_kst()
    today_str = now_kst.strftime("%Y-%m-%d")
    target_end = end or today_str
    last_trading_day = get_last_completed_trading_day()

    is_market_session = (now_kst.weekday() < 5 and now_kst.hour < 16)
    is_target_today = (target_end >= today_str)

    df_cached: Optional[pd.DataFrame] = None

    if not force_refresh and os.path.exists(cache_path) and os.path.getsize(cache_path) > 0:
        try:
            df_cached = pd.read_parquet(cache_path)
            if not df_cached.empty:
                df_cached.index = pd.to_datetime(df_cached.index)

                # During market session, strip today's stale intraday bar from cache if present
                if is_market_session and is_target_today:
                    df_cached = df_cached[df_cached.index.strftime("%Y-%m-%d") < today_str]

                last_cached_date = df_cached.index.max().strftime("%Y-%m-%d") if not df_cached.empty else "1900-01-01"

                # If target is historical (prior to today) and cache covers it or last completed trading day
                if not is_target_today:
                    if last_cached_date >= target_end or last_cached_date >= last_trading_day:
                        return df_cached.loc[start:target_end]
                else:
                    # Target is today: if outside market session and cache covers target_end
                    if not is_market_session and last_cached_date >= target_end:
                        return df_cached.loc[start:target_end]

                # Incremental fetch
                next_day = (df_cached.index.max() + timedelta(days=1)).strftime("%Y-%m-%d")
                # If target is historical and next_day is after last completed trading day, no new trading bars exist
                if not is_target_today and next_day > last_trading_day:
                    return df_cached.loc[start:target_end]

                if next_day <= target_end:
                    try:
                        df_new = fdr.DataReader(code, next_day, target_end)
                    except Exception:
                        df_new = None

                    if df_new is not None and not df_new.empty:
                        df_new.index = pd.to_datetime(df_new.index)
                        df_combined = pd.concat([df_cached, df_new])
                        df_combined = df_combined[~df_combined.index.duplicated(keep="last")].sort_index()

                        # Only persist today's bar to disk after market closes (>= 16:00 KST) or on weekend
                        if not is_market_session:
                            df_combined.to_parquet(cache_path)
                        else:
                            # Save historical portion only
                            df_hist = df_combined[df_combined.index.strftime("%Y-%m-%d") < today_str]
                            if not df_hist.empty:
                                df_hist.to_parquet(cache_path)

                        return df_combined.loc[start:target_end]

                return df_cached.loc[start:target_end]
        except Exception:
            pass

    # Full fetch
    try:
        df = fdr.DataReader(code, start, target_end)
    except Exception as e:
        # If network error and cache exists, fallback to cache
        if df_cached is not None and not df_cached.empty:
            return df_cached.loc[start:target_end]
        raise RuntimeError(f"Failed to fetch data for {code}: {e}")

    if df is None or df.empty:
        return pd.DataFrame(columns=["Open", "High", "Low", "Close", "Volume", "Change"])

    df.index = pd.to_datetime(df.index)
    # Ensure column casing
    rename_dict = {}
    for col in df.columns:
        c_lower = col.lower()
        if c_lower == "open":
            rename_dict[col] = "Open"
        elif c_lower == "high":
            rename_dict[col] = "High"
        elif c_lower == "low":
            rename_dict[col] = "Low"
        elif c_lower == "close":
            rename_dict[col] = "Close"
        elif c_lower == "volume":
            rename_dict[col] = "Volume"
        elif c_lower in ("change", "chg"):
            rename_dict[col] = "Change"
    df = df.rename(columns=rename_dict)

    # Save to parquet cache (strip today during live market session)
    try:
        os.makedirs(CACHE_DIR, exist_ok=True)
        if not is_market_session:
            df.to_parquet(cache_path)
        else:
            df_hist = df[df.index.strftime("%Y-%m-%d") < today_str]
            if not df_hist.empty:
                df_hist.to_parquet(cache_path)
    except Exception:
        pass

    return df.loc[start:target_end]


def get_benchmark_ohlcv(start: str = "2017-01-01", end: Optional[str] = None) -> pd.DataFrame:
    """Fetches and caches KODEX 200 (069500) ETF daily OHLCV."""
    return fetch_ohlcv(BENCHMARK_CODE, start=start, end=end)


def get_latest_market_info(code: str) -> Dict[str, Any]:
    """
    Returns latest price, daily change %, volume, and date for given ticker.
    Uses real-time quote API first for 100% live HTS accuracy, falling back to OHLCV.
    """
    try:
        rt_dict = fetch_realtime_quotes([code])
        if code in rt_dict:
            q = rt_dict[code]
            return {
                "close": q["close"],
                "change_pct": q["change_pct"],
                "volume": q["volume"],
                "date": q["date"],
                "high": q["high"],
                "low": q["low"],
                "open": q["open"],
                "open_pct": q["open_pct"],
                "high_pct": q["high_pct"],
                "low_pct": q["low_pct"],
            }

        df = fetch_ohlcv(code)
        if df.empty:
            return {"close": 0.0, "change_pct": 0.0, "volume": 0, "date": "N/A"}
        latest = df.iloc[-1]
        prev_close = df.iloc[-2]["Close"] if len(df) > 1 else latest["Close"]
        change_pct = ((latest["Close"] - prev_close) / prev_close) * 100.0 if prev_close > 0 else 0.0

        return {
            "close": float(latest["Close"]),
            "change_pct": float(change_pct),
            "volume": int(latest["Volume"]),
            "date": df.index[-1].strftime("%Y-%m-%d"),
            "high": float(latest["High"]),
            "low": float(latest["Low"]),
            "open": float(latest["Open"]),
            "open_pct": ((float(latest["Open"]) - prev_close) / prev_close * 100.0) if prev_close > 0 else 0.0,
            "high_pct": ((float(latest["High"]) - prev_close) / prev_close * 100.0) if prev_close > 0 else 0.0,
            "low_pct": ((float(latest["Low"]) - prev_close) / prev_close * 100.0) if prev_close > 0 else 0.0,
        }
    except Exception as e:
        return {"close": 0.0, "change_pct": 0.0, "volume": 0, "date": "Error", "error": str(e)}


if __name__ == "__main__":
    print("Testing market_data collector...")
    df_samsung = fetch_ohlcv("005930", "2024-01-01", "2024-01-10")
    print("Samsung Electronics (005930):")
    print(df_samsung.head(2))
    info = get_latest_market_info("005930")
    print("Latest info:", info)
