"""
Execution feasibility and realistic transaction cost model.
Includes:
- Upper limit (+30%) and gap-up (+15%) order cancellation filters
- Realistic broker commission (0.015% each side)
- Korean securities transaction tax (0.18% on sell)
- Differential slippage model (0.10% normal liquidity / 0.50% low liquidity)
"""

from typing import Tuple, Dict, Any
import numpy as np


# Cost parameters
COMMISSION_BUY = 0.00015   # 0.015%
COMMISSION_SELL = 0.00015  # 0.015%
TRANSACTION_TAX = 0.0018   # 0.18% (매도 증권거래세)
SLIPPAGE_NORMAL = 0.0010   # 0.10% (정상/대형 유동성)
SLIPPAGE_ILLIQUID = 0.0050 # 0.50% (유동성 부족)
LIQUIDITY_THRESHOLD_KRW = 5_000_000_000  # 50억 원 (20일 평균 일거래대금 기준)


def is_trade_feasible(
    prev_close: float,
    next_open: float,
    prev_prev_close: float = 0.0,
    allow_gap_down_filter: bool = True
) -> Tuple[bool, str]:
    """
    Checks whether a buy order can realistically execute on next day open.
    Rejects:
    1. If previous day reached upper limit (+30% price ceiling): cannot buy on open.
    2. If next day open gap > +15% above previous close: gap surge pursuit prohibited.
    3. If next day open gap < -1.5% below previous close: institutional dumping / breakdown risk.
    """
    if prev_close <= 0 or next_open <= 0:
        return False, "INVALID_PRICE"

    # 1. Previous day upper limit check (+29.5% or higher)
    if prev_prev_close > 0:
        prev_day_gain = (prev_close - prev_prev_close) / prev_prev_close
        if prev_day_gain >= 0.295:
            return False, "UPPER_LIMIT_PREV_DAY"

    # 2. Open gap > +15% check
    open_gap = (next_open - prev_close) / prev_close
    if open_gap >= 0.15:
        return False, "GAP_UP_EXCEEDS_15PCT"

    # 3. Open gap down < -1.5% check
    if allow_gap_down_filter and open_gap <= -0.015:
        return False, "GAP_DOWN_EXCEEDS_1_5PCT"

    return True, "FEASIBLE"


def get_slippage_rate(avg_daily_volume_val: float) -> float:
    """
    Returns slippage rate based on liquidity.
    Normal/High liquidity: 0.10%
    Low liquidity: 0.50%
    """
    if avg_daily_volume_val >= LIQUIDITY_THRESHOLD_KRW:
        return SLIPPAGE_NORMAL
    return SLIPPAGE_ILLIQUID


def calculate_net_trade_return(
    raw_entry_price: float,
    raw_exit_price: float,
    avg_daily_val: float = LIQUIDITY_THRESHOLD_KRW
) -> Dict[str, Any]:
    """
    Calculates precise net return after entry/exit slippage, commissions, and transaction tax.
    """
    slippage = get_slippage_rate(avg_daily_val)

    # Effective prices with slippage
    entry_price = raw_entry_price * (1.0 + slippage)
    exit_price = raw_exit_price * (1.0 - slippage)

    # Cost breakdown
    total_entry_cost = entry_price * (1.0 + COMMISSION_BUY)
    net_exit_revenue = exit_price * (1.0 - COMMISSION_SELL - TRANSACTION_TAX)

    net_return = (net_exit_revenue - total_entry_cost) / total_entry_cost
    net_return_pct = net_return * 100.0

    gross_return = (raw_exit_price - raw_entry_price) / raw_entry_price
    gross_return_pct = gross_return * 100.0

    return {
        "raw_entry": raw_entry_price,
        "raw_exit": raw_exit_price,
        "effective_entry": entry_price,
        "effective_exit": exit_price,
        "slippage_rate": slippage,
        "gross_return_pct": round(gross_return_pct, 4),
        "net_return_pct": round(net_return_pct, 4),
        "net_return": net_return,
    }


if __name__ == "__main__":
    feasible, reason = is_trade_feasible(10000, 11600)
    print("Gap 16% Feasible?", feasible, reason)
    ret = calculate_net_trade_return(10000, 10300)
    print("Return on 10,000 -> 10,300 (+3% raw):", ret)
