import unittest
import sqlite3
from src.collectors.krx_universe import (
    normalize_stock_name,
    search_ticker,
    CANONICAL_NAME_OVERRIDES,
)
from src.core.quant_analyst import extract_all_stocks_from_query
from src.database.models import migrate_stock_names


class TestStockNameNormalization(unittest.TestCase):
    def test_canonical_name_overrides(self):
        self.assertEqual(normalize_stock_name("에스케이바이오팜", "326030"), "SK바이오팜")
        self.assertEqual(normalize_stock_name("케이티", "030200"), "KT")
        self.assertEqual(normalize_stock_name("케이티앤지", "033780"), "KT&G")
        self.assertEqual(normalize_stock_name("케이티알파", "036030"), "KT알파")
        self.assertEqual(normalize_stock_name("에스디바이오센서", "137310"), "SD바이오센서")
        self.assertEqual(normalize_stock_name("케이비아이동국실업", "001620"), "KBI동국실업")
        self.assertEqual(normalize_stock_name("지에스이", "053050"), "GSE")
        self.assertEqual(normalize_stock_name("에스엠벡셀", "010580"), "SM벡셀")

    def test_pattern_rules(self):
        self.assertEqual(normalize_stock_name("에스케이스퀘어"), "SK스퀘어")
        self.assertEqual(normalize_stock_name("에스케이텔레콤"), "SK텔레콤")
        self.assertEqual(normalize_stock_name("엘지에너지솔루션"), "LG에너지솔루션")
        self.assertEqual(normalize_stock_name("엘지화학"), "LG화학")
        self.assertEqual(normalize_stock_name("씨제이제일제당"), "CJ제일제당")
        self.assertEqual(normalize_stock_name("케이비금융"), "KB금융")
        self.assertEqual(normalize_stock_name("엔에이치투자증권"), "NH투자증권")

    def test_legitimate_names_preserved(self):
        self.assertEqual(normalize_stock_name("피에스케이"), "피에스케이")
        self.assertEqual(normalize_stock_name("디에스케이"), "디에스케이")
        self.assertEqual(normalize_stock_name("삼성전자"), "삼성전자")
        self.assertEqual(normalize_stock_name("현대차"), "현대차")
        self.assertEqual(normalize_stock_name("포스코DX"), "포스코DX")

    def test_strip_company_affixes(self):
        self.assertEqual(normalize_stock_name("(주)에스케이바이오팜", "326030"), "SK바이오팜")
        self.assertEqual(normalize_stock_name("주식회사 에스케이바이오팜", "326030"), "SK바이오팜")

    def test_search_ticker_bidirectional(self):
        res_sk = search_ticker("SK바이오팜")
        self.assertTrue(any(r["code"] == "326030" and r["name"] == "SK바이오팜" for r in res_sk))

        res_hangul = search_ticker("에스케이바이오팜")
        self.assertTrue(any(r["code"] == "326030" and r["name"] == "SK바이오팜" for r in res_hangul))

        res_code = search_ticker("326030")
        self.assertTrue(any(r["code"] == "326030" and r["name"] == "SK바이오팜" for r in res_code))

    def test_quant_analyst_stock_extraction(self):
        stocks_hangul = extract_all_stocks_from_query("에스케이바이오팜 앞으로 전망 분석해줘")
        self.assertTrue(any(s["code"] == "326030" and s["name"] == "SK바이오팜" for s in stocks_hangul))

        stocks_canonical = extract_all_stocks_from_query("SK바이오팜 매수 전략 알려줘")
        self.assertTrue(any(s["code"] == "326030" and s["name"] == "SK바이오팜" for s in stocks_canonical))

    def test_migrate_stock_names_database(self):
        conn = sqlite3.connect(":memory:")
        conn.row_factory = sqlite3.Row
        cur = conn.cursor()
        cur.execute("CREATE TABLE universe (code TEXT PRIMARY KEY, name TEXT)")
        cur.execute("CREATE TABLE watchlist (code TEXT PRIMARY KEY, name TEXT)")
        cur.execute("CREATE TABLE predictions (id INTEGER PRIMARY KEY, code TEXT, name TEXT)")

        cur.execute("INSERT INTO universe VALUES ('326030', '에스케이바이오팜')")
        cur.execute("INSERT INTO watchlist VALUES ('326030', '에스케이바이오팜')")
        cur.execute("INSERT INTO predictions VALUES (1, '326030', '에스케이바이오팜')")
        conn.commit()

        updated = migrate_stock_names(conn)
        self.assertEqual(updated, 3)

        cur.execute("SELECT name FROM universe WHERE code = '326030'")
        self.assertEqual(cur.fetchone()["name"], "SK바이오팜")

        cur.execute("SELECT name FROM watchlist WHERE code = '326030'")
        self.assertEqual(cur.fetchone()["name"], "SK바이오팜")

        cur.execute("SELECT name FROM predictions WHERE id = 1")
        self.assertEqual(cur.fetchone()["name"], "SK바이오팜")

        conn.close()


if __name__ == "__main__":
    unittest.main()
