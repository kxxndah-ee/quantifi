"""
Point-in-Time Technical Scoring Engine.
Computes technical indicators (RSI-14, MA alignment, Volume surge ratio, Bollinger %b)
using strictly data available up to day T close (no look-ahead bias).
Generates 0~100 score, Bullish/Neutral/Bearish label, and indicator breakdown.
"""

from typing import Dict, Any, Optional
import pandas as pd
import numpy as np


# Default Indicator Weights (Total = 100)
DEFAULT_WEIGHTS = {
    "rsi": 30.0,
    "ma": 25.0,
    "volume": 25.0,
    "bb": 20.0,
}


def calculate_rsi(series: pd.Series, period: int = 14) -> pd.Series:
    """Computes Wilder's Relative Strength Index (RSI)."""
    delta = series.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)

    avg_gain = gain.ewm(alpha=1.0 / period, min_periods=period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1.0 / period, min_periods=period, adjust=False).mean()

    rs = avg_gain / avg_loss.replace(0, np.nan)
    rsi = 100.0 - (100.0 / (1.0 + rs))
    return rsi.fillna(50.0)


def calculate_bollinger_bands(series: pd.Series, period: int = 20, num_std: float = 2.0):
    """Computes Bollinger Bands middle, upper, lower, and %b."""
    mid = series.rolling(window=period).mean()
    std = series.rolling(window=period).std()
    upper = mid + (num_std * std)
    lower = mid - (num_std * std)
    bandwidth = upper - lower
    pct_b = (series - lower) / bandwidth.replace(0, np.nan)
    return mid, upper, lower, pct_b.fillna(0.5)


def compute_point_in_time_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """
    Computes rolling indicators across historical OHLCV.
    Guarantees that at row i, only rows <= i are utilized.
    """
    if len(df) < 60:
        return pd.DataFrame()

    df = df.copy()
    close = df["Close"]
    volume = df["Volume"]

    # Moving Averages
    df["MA5"] = close.rolling(window=5).mean()
    df["MA20"] = close.rolling(window=20).mean()
    df["MA60"] = close.rolling(window=60).mean()

    # RSI 14
    df["RSI14"] = calculate_rsi(close, 14)

    # Volume Surge Ratio (today / 20-day average volume)
    df["VolMA20"] = volume.rolling(window=20).mean()
    df["VolRatio"] = volume / df["VolMA20"].replace(0, np.nan)
    df["VolRatio"] = df["VolRatio"].fillna(1.0)

    # Bollinger Bands
    mid, upper, lower, pct_b = calculate_bollinger_bands(close, 20, 2.0)
    df["BB_Mid"] = mid
    df["BB_Upper"] = upper
    df["BB_Lower"] = lower
    df["BB_PctB"] = pct_b

    # 20-day High and Drawdown
    high_series = df["High"] if "High" in df.columns else close
    df["High20"] = high_series.rolling(window=20, min_periods=1).max()
    df["Drawdown20"] = ((close - df["High20"]) / df["High20"].replace(0, np.nan)) * 100.0
    df["Drawdown20"] = df["Drawdown20"].fillna(0.0)

    # MA5 Slope
    df["MA5_Diff"] = df["MA5"].diff().fillna(0.0)

    return df


def calculate_score_for_row(row: pd.Series, weights: Optional[Dict[str, float]] = None) -> Dict[str, Any]:
    """
    Calculates 0~100 technical score and breakdown for a single point-in-time record.
    """
    w = weights or DEFAULT_WEIGHTS
    w_rsi = w.get("rsi", 30.0)
    w_ma = w.get("ma", 25.0)
    w_vol = w.get("volume", 25.0)
    w_bb = w.get("bb", 20.0)
    total_w = w_rsi + w_ma + w_vol + w_bb

    rsi_val = float(row.get("RSI14", 50.0))
    ma5 = float(row.get("MA5", 0.0))
    ma20 = float(row.get("MA20", 0.0))
    ma60 = float(row.get("MA60", 0.0))
    vol_ratio = float(row.get("VolRatio", 1.0))
    bb_pct = float(row.get("BB_PctB", 0.5))
    close = float(row.get("Close", 0.0))
    change_pct = float(row.get("change_pct", row.get("Change", 0.0)))
    drawdown20 = float(row.get("Drawdown20", 0.0))
    high20 = float(row.get("High20", 0.0))
    ma5_diff = float(row.get("MA5_Diff", 0.0))

    # 1. RSI Score (Base 0~1 multiplier * w_rsi)
    # 45~65: momentum/pullback rebound zone -> 100%
    # <=30: oversold rebound zone -> 85%
    # 30~45: recovery zone -> 60%
    # 65~75: extended momentum -> 60%
    # >75: overbought risk -> 20%
    if 45.0 <= rsi_val <= 65.0:
        # If in severe drawdown or broken below 20MA & 5MA, RSI is downward drift, not momentum
        if drawdown20 <= -15.0 or (ma20 > 0 and close < ma20 and close < ma5):
            rsi_factor = 0.40
        else:
            rsi_factor = 1.0
    elif rsi_val <= 30.0:
        rsi_factor = 0.85
    elif 30.0 < rsi_val < 45.0:
        rsi_factor = 0.60
    elif 65.0 < rsi_val <= 75.0:
        rsi_factor = 0.60
    else:
        rsi_factor = 0.20
    rsi_score = rsi_factor * w_rsi

    # 2. Moving Average Alignment & Price Position Score
    # Moving averages lag actual price. Verify actual price position relative to MA20 and MA5.
    if ma5 > ma20 > ma60 > 0:
        if close >= ma20 and close >= ma5:
            ma_factor = 1.0
            ma_status = "정배열 (5>20>60)"
        elif close >= ma20:
            ma_factor = 0.80
            ma_status = "20일선 지지 (5일선 하회 눌림)"
        else:
            # Breakdown below 20MA despite lagging alignment
            ma_factor = 0.20
            ma_status = "20일선 이탈 (하향조정)"
    elif ma5 > ma20 > 0:
        if close >= ma20:
            ma_factor = 0.65
            ma_status = "단기골든 (5>20)"
        else:
            ma_factor = 0.15
            ma_status = "단기선 이탈 (20일선 하회)"
    else:
        ma_factor = 0.10
        ma_status = "역배열/혼조"
    ma_score = ma_factor * w_ma

    # 3. Volume Surge Ratio Score
    # >= 2.0 (200%+): 100%
    # 1.5 ~ 2.0: 80%
    # 1.2 ~ 1.5: 60%
    # 1.0 ~ 1.2: 40%
    # < 1.0: 15%
    if vol_ratio >= 2.0:
        vol_factor = 1.0
    elif vol_ratio >= 1.5:
        vol_factor = 0.80
    elif vol_ratio >= 1.2:
        vol_factor = 0.60
    elif vol_ratio >= 1.0:
        vol_factor = 0.40
    else:
        vol_factor = 0.15
    vol_score = vol_factor * w_vol

    # 4. Bollinger Band (%b) Score
    # 0.2 ~ 0.8 (rebounding within bands): 100%
    # < 0.2 (oversold band touch): 60%
    # > 0.8 (band top resistance): 25%
    if 0.2 <= bb_pct <= 0.8:
        if ma20 > 0 and close < ma20 and bb_pct < 0.50:
            bb_factor = 0.50
        else:
            bb_factor = 1.0
    elif bb_pct < 0.2:
        bb_factor = 0.60
    else:
        bb_factor = 0.25
    bb_score = bb_factor * w_bb

    # 5. Peak Drawdown & Trend Breakdown Penalty
    penalty = 0.0
    if drawdown20 <= -20.0:
        penalty += 25.0
    elif drawdown20 <= -15.0:
        penalty += 18.0
    elif drawdown20 <= -10.0 and ma20 > 0 and close < ma20:
        penalty += 10.0

    if ma20 > 0 and close < ma20:
        penalty += 10.0

    raw_total = rsi_score + ma_score + vol_score + bb_score
    raw_score = (raw_total / total_w) * 100.0
    final_score = max(5.0, min(100.0, raw_score - penalty))
    total_score = round(final_score, 1)

    # Label assignment: Disallow "강세" for stocks under 20MA or suffering steep drawdown
    if (ma20 > 0 and close < ma20) or drawdown20 <= -15.0:
        if total_score >= 50.0:
            label = "중립"
        else:
            label = "약세"
    else:
        if total_score >= 70.0:
            label = "강세"
        elif total_score >= 50.0:
            label = "중립"
        else:
            label = "약세"

    # Dynamic TP / SL targets based on Score
    # Section 2-4:
    # Score >= 80 -> +3.0%
    # Score 70~79 -> +2.0%
    # Score 60~69 -> +1.5%
    # SL -> default -2.0%
    if total_score >= 80.0:
        tp_pct = 3.0
    elif total_score >= 70.0:
        tp_pct = 2.0
    elif total_score >= 60.0:
        tp_pct = 1.5
    else:
        tp_pct = 1.0
    sl_pct = 2.0

    analyzed_gain_pct = calculate_analyzed_rise_rate(
        total_score=total_score,
        vol_ratio=vol_ratio,
        rsi_val=rsi_val,
        ma_status=ma_status,
        change_pct=change_pct,
        bb_pct=bb_pct
    )

    return {
        "score": total_score,
        "label": label,
        "tp_pct": tp_pct,
        "sl_pct": sl_pct,
        "analyzed_gain_pct": analyzed_gain_pct,
        "breakdown": {
            "rsi_score": round(rsi_score, 1),
            "rsi_val": round(rsi_val, 1),
            "ma_score": round(ma_score, 1),
            "ma_status": ma_status,
            "vol_score": round(vol_score, 1),
            "vol_ratio": round(vol_ratio, 2),
            "bb_score": round(bb_score, 1),
            "bb_pct": round(bb_pct, 2),
            "drawdown20": round(drawdown20, 1),
            "penalty": round(penalty, 1),
        },
        "indicators": {
            "close": close,
            "ma5": round(ma5, 1),
            "ma20": round(ma20, 1),
            "ma60": round(ma60, 1),
            "high20": round(high20, 1),
            "drawdown20": round(drawdown20, 1),
            "rsi14": round(rsi_val, 1),
            "vol_ratio": round(vol_ratio, 2),
            "bb_pct": round(bb_pct, 2),
        }
    }


def calculate_analyzed_rise_rate(
    total_score: float,
    vol_ratio: float = 1.0,
    rsi_val: float = 50.0,
    ma_status: str = "보통",
    change_pct: float = 0.0,
    strategy_tag: str = "",
    bb_pct: float = 0.5
) -> float:
    """
    Computes the quant analyst's modeled expected upside potential / surge rate (분석 상승률, %).
    Grounded in multi-factor momentum and statistical expectations:
    - Base expected surge derived from composite quant score: 1.0% + (total_score * 0.05)
      (e.g., 80점 -> 1.0 + 4.0 = 5.0%, 60점 -> 1.0 + 3.0 = 4.0%, 40점 -> 1.0 + 2.0 = 3.0%)
    - Volume surge velocity: additional upside for high volume explosion:
      min(2.5, max(0.0, (vol_ratio - 1.0) * 0.8))
    - Moving average alignment bonus:
      +0.8% for 정배열, +0.3% for 수렴/밀집, -0.6% for 역배열
    - RSI sweet spot bonus (48~68pt: +0.6%), overbought penalty (>75pt: -0.7%)
    - High-conviction strategy floor:
      if '당일단타' or '5%타겟' in strategy_tag: floor of 5.0%
      elif '스나이퍼' in strategy_tag: target +1.5% ~ +4.5%
      elif '종가배팅' in strategy_tag: target +1.8% ~ +4.8%
    Result is rounded to 1 decimal place, constrained between +1.0% and +12.0%.
    """
    base_gain = 1.0 + (float(total_score) * 0.05)
    vol_boost = min(2.5, max(0.0, (float(vol_ratio) - 1.0) * 0.8))
    ma_boost = 0.8 if "정배열" in ma_status else (0.3 if any(w in ma_status for w in ["수렴", "밀집"]) else (-0.6 if "역배열" in ma_status else 0.0))
    rsi_boost = 0.6 if 48.0 <= float(rsi_val) <= 68.0 else (-0.7 if float(rsi_val) > 75.0 else 0.0)

    est_gain = base_gain + vol_boost + ma_boost + rsi_boost

    if any(k in strategy_tag for k in ["당일단타", "5%타겟", "5% 급등"]):
        est_gain = max(5.0, est_gain)
    elif "스나이퍼" in strategy_tag:
        est_gain = max(1.5, min(4.5, est_gain * 0.7))
    elif "종가배팅" in strategy_tag:
        est_gain = max(1.8, min(4.8, est_gain * 0.8))

    return round(max(1.0, min(12.0, est_gain)), 1)


def evaluate_intraday_outlook(row_today: pd.Series, prev_close: float = 0.0) -> Dict[str, Any]:
    """
    Evaluates real-time intraday trading momentum and structure for day T.
    Categorizes into:
    - [수급돌파] (Strong intraday breakout, candidate for day trading)
    - [눌림지지] (Consolidating near Open with support, holding floor)
    - [윗꼬리경계] (Pullback from day high, upper shadow profit-taking)
    - [시초하회] (Below Open, intraday selling pressure)
    - [거래소강] (Low volume sideways)
    """
    open_p = float(row_today.get("Open", 0.0))
    high_p = float(row_today.get("High", 0.0))
    low_p = float(row_today.get("Low", 0.0))
    close_p = float(row_today.get("Close", 0.0))
    vol_ratio = float(row_today.get("VolRatio", 1.0))

    if open_p <= 0:
        return {
            "tag": "[거래소강]",
            "desc": "장중 시세 데이터 대기 중",
            "open_gain_pct": 0.0,
            "candle_support_ratio": 0.5,
            "upper_shadow_pct": 0.0
        }

    open_gain_pct = ((close_p - open_p) / open_p) * 100.0
    day_change_pct = ((close_p - prev_close) / prev_close * 100.0) if prev_close > 0 else open_gain_pct
    day_range = high_p - low_p
    candle_support_ratio = (close_p - low_p) / day_range if day_range > 0 else 0.5
    body_top = max(open_p, close_p)
    upper_shadow_pct = ((high_p - body_top) / open_p * 100.0) if open_p > 0 else 0.0

    # 1. Bearish drop below open (when below open significantly)
    if open_gain_pct <= -1.2 or (open_gain_pct < 0 and candle_support_ratio < 0.35):
        tag = "[시초하회]"
        desc = f"시초가 대비 {open_gain_pct:.1f}% 하회 음봉 진행 (당일 매수 자제)"
    # 2. Upper shadow warning (profit taking / dump above body)
    elif upper_shadow_pct >= 2.5 and candle_support_ratio < 0.50:
        tag = "[윗꼬리경계]"
        desc = f"고점 대비 +{upper_shadow_pct:.1f}% 윗꼬리 매물 출회 (차익실현 경계)"
    # 3. Bullish breakout (high candle support, above open by >= 2.0%, decent volume)
    elif open_gain_pct >= 2.0 and candle_support_ratio >= 0.60:
        tag = "[수급돌파]"
        desc = f"시초가 대비 +{open_gain_pct:.1f}% 돌파 & 양봉 지지율 {candle_support_ratio*100:.0f}% (단타 강세)"
    # 4. Pullback support (near open -1.0% ~ +2.0%, solid floor)
    elif -1.0 <= open_gain_pct <= 2.0 and candle_support_ratio >= 0.35 and day_change_pct >= -2.0:
        tag = "[눌림지지]"
        desc = f"시초가 부근 지지력 유지 ({open_gain_pct:+.1f}%) & 반등 수급 대기"
    # 5. Low volume / sideways
    else:
        tag = "[거래소강]"
        desc = f"장중 거래량 둔화 및 횡보 흐름 (관망 권장)"

    return {
        "tag": tag,
        "desc": desc,
        "open_gain_pct": round(open_gain_pct, 2),
        "candle_support_ratio": round(candle_support_ratio, 2),
        "upper_shadow_pct": round(upper_shadow_pct, 2),
    }


def evaluate_stock_latest(df: pd.DataFrame, weights: Optional[Dict[str, float]] = None) -> Optional[Dict[str, Any]]:
    """
    Evaluates latest point-in-time technical score and indicators for given stock OHLCV.
    Returns both intraday real-time outlook and next-day swing outlook.
    """
    if df is None or len(df) < 60:
        return None

    df_ind = compute_point_in_time_indicators(df)
    if df_ind.empty:
        return None

    latest_row = df_ind.iloc[-1]
    res = calculate_score_for_row(latest_row, weights=weights)
    res["date"] = df_ind.index[-1].strftime("%Y-%m-%d")

    # Intraday Outlook
    prev_close = float(df_ind["Close"].iloc[-2]) if len(df_ind) >= 2 else float(latest_row["Close"])
    res["intraday_outlook"] = evaluate_intraday_outlook(latest_row, prev_close=prev_close)

    # Next-Day Swing Outlook
    total_score = res["score"]
    drawdown20 = float(latest_row.get("Drawdown20", 0.0))
    close_val = float(latest_row.get("Close", 0.0))
    ma20_val = float(latest_row.get("MA20", 0.0))

    if (ma20_val > 0 and close_val < ma20_val) or drawdown20 <= -15.0:
        next_tag = f"약세/조정 ({total_score:.1f}점)"
        if drawdown20 <= -15.0:
            next_desc = f"고점대비 {drawdown20:.1f}% 하락 및 20일선 하회 (추세 이탈/관망 권장)"
        else:
            next_desc = "20일 이동평균선 하향 이탈 (단기 조정/신규 매수 자제)"
    elif total_score >= 70.0:
        next_tag = f"강세 ({total_score:.1f}점)"
        next_desc = "이평 정배열 & 모멘텀 우수 (스윙 긍정적)"
    elif total_score >= 50.0:
        next_tag = f"중립 ({total_score:.1f}점)"
        next_desc = "추세 횡보/방향성 탐색 구간 (분할 접근)"
    else:
        next_tag = f"약세 ({total_score:.1f}점)"
        next_desc = "이평 역배열/조정 국면 (보수적 대응 권장)"

    res["nextday_outlook"] = {
        "tag": next_tag,
        "desc": next_desc,
        "score": total_score,
        "label": res["label"],
        "tp_pct": res["tp_pct"],
        "sl_pct": res["sl_pct"]
    }
    return res


if __name__ == "__main__":
    from src.collectors.market_data import fetch_ohlcv
    df = fetch_ohlcv("005930", "2023-01-01", "2024-01-10")
    score_res = evaluate_stock_latest(df)
    print("Samsung score result:")
    print(score_res)
