import unittest
from src.core.execution import (
    is_trade_feasible,
    calculate_net_trade_return,
    get_slippage_rate,
    COMMISSION_BUY,
    COMMISSION_SELL,
    TRANSACTION_TAX
)


class TestExecution(unittest.TestCase):
    def test_gap_up_rejection(self):
        # +16% gap open -> should be rejected
        feasible, reason = is_trade_feasible(prev_close=10000, next_open=11600)
        self.assertFalse(feasible)
        self.assertEqual(reason, "GAP_UP_EXCEEDS_15PCT")

    def test_upper_limit_rejection(self):
        # Prev day gained 30% -> cannot buy at open
        feasible, reason = is_trade_feasible(prev_close=13000, next_open=13500, prev_prev_close=10000)
        self.assertFalse(feasible)
        self.assertEqual(reason, "UPPER_LIMIT_PREV_DAY")

    def test_normal_trade_feasible(self):
        feasible, reason = is_trade_feasible(prev_close=10000, next_open=10200, prev_prev_close=9900)
        self.assertTrue(feasible)
        self.assertEqual(reason, "FEASIBLE")

    def test_gap_down_rejection(self):
        # -2.0% gap open -> rejected
        feasible, reason = is_trade_feasible(prev_close=10000, next_open=9800)
        self.assertFalse(feasible)
        self.assertEqual(reason, "GAP_DOWN_EXCEEDS_1_5PCT")

    def test_cost_calculation(self):
        # 10,000 -> 10,300 (+3.0% gross)
        res = calculate_net_trade_return(10000, 10300, avg_daily_val=10_000_000_000)
        self.assertEqual(res["slippage_rate"], 0.001)  # 0.1% for normal liquidity
        self.assertGreater(res["gross_return_pct"], res["net_return_pct"])
        # Net return should account for 0.1% buy slippage, 0.1% sell slippage, 0.03% commissions, 0.18% tax
        self.assertTrue(2.4 < res["net_return_pct"] < 2.7)


if __name__ == "__main__":
    unittest.main()
