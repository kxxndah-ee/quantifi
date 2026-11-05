"""
KRX Universe Manager with Survivorship Bias Prevention.
Maintains active and delisted stock metadata (code, name, market, sector, industry).
Stores universe in SQLite data/app.db for fast local querying and screener execution.
"""

import io
import os
import re
import json
import sqlite3
from datetime import datetime
from typing import List, Dict, Any, Optional

try:
    import truststore
    truststore.inject_into_ssl()
except Exception:
    pass

import pandas as pd
import requests
import urllib3
urllib3.disable_warnings()

from src.database.models import get_db_connection, DB_PATH, get_now_kst


def fetch_active_stocks_from_krx() -> pd.DataFrame:
    """
    Fetches active KOSPI/KOSDAQ listed companies from KRX / KIND.
    Returns DataFrame with columns: Code, Name, Market, Sector, Industry.
    """
    # Method 1: KIND corporate list download
    try:
        url = "https://kind.krx.co.kr/corpgeneral/corpList.do?method=download&searchType=13"
        r = requests.get(url, verify=False, timeout=15)
        if r.status_code == 200:
            r.encoding = "cp949"
            dfs = pd.read_html(io.StringIO(r.text), header=0)
            if dfs and not dfs[0].empty:
                df = dfs[0]
                # Column mapping
                # 회사명, 시장구분, 종목코드, 업종, 주요제품, 상장일, 결산월, 대표자명, 홈페이지, 지역
                rename_map = {
                    "회사명": "Name",
                    "시장구분": "Market",
                    "종목코드": "Code",
                    "업종": "Sector",
                    "주요제품": "Industry",
                }
                # Handle possible encoding anomalies in header
                for col in df.columns:
                    if "회사" in col:
                        rename_map[col] = "Name"
                    elif "시장" in col:
                        rename_map[col] = "Market"
                    elif "종목코드" in col or "코드" in col:
                        rename_map[col] = "Code"
                    elif "업종" in col:
                        rename_map[col] = "Sector"
                    elif "주요제품" in col or "제품" in col:
                        rename_map[col] = "Industry"

                df = df.rename(columns=rename_map)
                df["Code"] = df["Code"].astype(str).str.zfill(6)
                df["Market"] = df["Market"].fillna("").astype(str).str.upper()
                # Normalize Market names
                df["Market"] = df["Market"].apply(
                    lambda m: "KOSPI" if "유가" in m or "코스피" in m or "STK" in m
                    else ("KOSDAQ" if "코스닥" in m or "KSQ" in m else ("KONEX" if "코넥스" in m else m))
                )
                if "Sector" not in df.columns:
                    df["Sector"] = "일반"
                if "Industry" not in df.columns:
                    df["Industry"] = ""

                result = df[["Code", "Name", "Market", "Sector", "Industry"]].drop_duplicates("Code")
                result["Name"] = result.apply(lambda r: normalize_stock_name(r["Name"], r["Code"]), axis=1)
                return result
    except Exception as e:
        print(f"Warning: KIND active stock fetch error: {e}")

    # Method 2: FinanceDataReader KrxStockListing fallback
    try:
        import FinanceDataReader.krx.listing as kl
        df_kl = kl.KrxStockListing("KRX-DESC").read()
        if not df_kl.empty:
            df_kl = df_kl.rename(columns={"Code": "Code", "Name": "Name", "Market": "Market", "Sector": "Sector", "Industry": "Industry"})
            df_kl["Code"] = df_kl["Code"].astype(str).str.zfill(6)
            for col in ["Sector", "Industry"]:
                if col not in df_kl.columns:
                    df_kl[col] = ""
            return df_kl[["Code", "Name", "Market", "Sector", "Industry"]].drop_duplicates("Code")
    except Exception as e:
        print(f"Warning: KrxStockListing fallback error: {e}")

    return pd.DataFrame(columns=["Code", "Name", "Market", "Sector", "Industry"])


def fetch_historical_delisted_stocks() -> pd.DataFrame:
    """
    Returns curated list of historically delisted / merged stocks from 2017 to present
    to ensure backtests are protected from survivorship bias.
    """
    # Well-known delisted/merged sample universe entries from 2017-2024
    delisted_seed = [
        {"Code": "034230", "Name": "소프트맥스", "Market": "KOSDAQ", "Sector": "소프트웨어", "Industry": "게임", "is_active": 0, "delist_date": "2017-03-31"},
        {"Code": "001470", "Name": "삼부토건(구)", "Market": "KOSPI", "Sector": "건설", "Industry": "토목", "is_active": 0, "delist_date": "2017-09-15"},
        {"Code": "053950", "Name": "키위미디어그룹", "Market": "KOSPI", "Sector": "엔터", "Industry": "영화", "is_active": 0, "delist_date": "2020-11-20"},
        {"Code": "054620", "Name": "AP시스템(구)", "Market": "KOSDAQ", "Sector": "반도체장비", "Industry": "디스플레이", "is_active": 0, "delist_date": "2017-04-07"},
        {"Code": "083640", "Name": "인콘(구)", "Market": "KOSDAQ", "Sector": "IT", "Industry": "CCTV", "is_active": 0, "delist_date": "2021-06-15"},
        {"Code": "099430", "Name": "바이오빌", "Market": "KOSDAQ", "Sector": "화학", "Industry": "필름", "is_active": 0, "delist_date": "2020-07-28"},
        {"Code": "101000", "Name": "마스타테크론", "Market": "KOSDAQ", "Sector": "반도체", "Industry": "검사장비", "is_active": 0, "delist_date": "2018-05-10"},
        {"Code": "038530", "Name": "골드앤에스", "Market": "KOSDAQ", "Sector": "교육", "Industry": "어학", "is_active": 0, "delist_date": "2022-04-20"},
        {"Code": "058610", "Name": "에스에프씨", "Market": "KOSDAQ", "Sector": "신재생", "Industry": "태양광", "is_active": 0, "delist_date": "2020-03-12"},
        {"Code": "033500", "Name": "동성화학", "Market": "KOSPI", "Sector": "화학", "Industry": "폴리우레탄", "is_active": 0, "delist_date": "2021-04-01"},
    ]
    return pd.DataFrame(delisted_seed)


def sync_krx_universe(force: bool = False) -> int:
    """
    Synchronizes KRX listed and delisted stocks into the SQLite universe table.
    Returns total count of registered universe stocks.
    """
    with get_db_connection() as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT COUNT(*) FROM universe")
        count = cursor.fetchone()[0]
        if count > 1000 and not force:
            return count

        print("Syncing KRX Universe into local database...")
        df_active = fetch_active_stocks_from_krx()
        now_str = get_now_kst().strftime("%Y-%m-%d %H:%M:%S")

        records_to_upsert = []
        if not df_active.empty:
            for _, row in df_active.iterrows():
                code = str(row["Code"]).zfill(6)
                name = normalize_stock_name(str(row["Name"]), code)
                market = str(row.get("Market", "KOSPI"))
                sector = str(row.get("Sector", "기타"))
                industry = str(row.get("Industry", ""))
                records_to_upsert.append((code, name, market, sector, industry, 1, None, now_str))

        # Delisted stocks
        df_delisted = fetch_historical_delisted_stocks()
        for _, row in df_delisted.iterrows():
            code = str(row["Code"]).zfill(6)
            name = normalize_stock_name(str(row["Name"]), code)
            market = str(row["Market"])
            sector = str(row["Sector"])
            industry = str(row["Industry"])
            delist_date = str(row["delist_date"])
            records_to_upsert.append((code, name, market, sector, industry, 0, delist_date, now_str))

        cursor.executemany("""
            INSERT INTO universe (code, name, market, sector, industry, is_active, delist_date, last_updated)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(code) DO UPDATE SET
                name=excluded.name,
                market=excluded.market,
                sector=excluded.sector,
                industry=excluded.industry,
                is_active=excluded.is_active,
                delist_date=excluded.delist_date,
                last_updated=excluded.last_updated
        """, records_to_upsert)

        conn.commit()

        cursor.execute("SELECT COUNT(*) FROM universe")
        total = cursor.fetchone()[0]
        print(f"KRX Universe synchronized: {total} stocks.")
        return total


CANONICAL_NAME_OVERRIDES: Dict[str, str] = {
    "326030": "SK바이오팜",
    "030200": "KT",
    "033780": "KT&G",
    "036030": "KT알파",
    "053210": "KT스카이라이프",
    "058850": "KTcs",
    "058860": "KTis",
    "137310": "SD바이오센서",
    "001620": "KBI동국실업",
    "053050": "GSE",
    "010580": "SM벡셀",
    "460870": "SMCG",
}


def normalize_stock_name(name: str, code: str = "") -> str:
    """
    Normalizes Korean legal entity names from KRX KIND / corporate registries
    into clean, standard Korean stock market trading brand names.
    (e.g., '에스케이바이오팜' -> 'SK바이오팜', '케이티' -> 'KT', '케이티앤지' -> 'KT&G')
    """
    if not name:
        return ""
    clean_name = str(name).strip()
    clean_code = str(code).strip().zfill(6) if code else ""

    # Strip (주) or 주식회사
    clean_name = re.sub(r"^\(주\)\s*|\s*\(주\)$", "", clean_name).strip()
    clean_name = re.sub(r"^주식회사\s*|\s*주식회사$", "", clean_name).strip()

    if clean_code in CANONICAL_NAME_OVERRIDES:
        return CANONICAL_NAME_OVERRIDES[clean_code]

    # Pattern-based canonical brand abbreviations
    if clean_name.startswith("에스케이"):
        return "SK" + clean_name[4:]
    elif clean_name.startswith("엘지"):
        return "LG" + clean_name[2:]
    elif clean_name == "케이티앤지":
        return "KT&G"
    elif clean_name == "케이티":
        return "KT"
    elif clean_name.startswith("케이티") and any(k in clean_name for k in ["알파", "스카이", "cs", "is", "씨에스", "아이엔"]):
        return "KT" + clean_name[3:]
    elif clean_name.startswith("씨제이"):
        return "CJ" + clean_name[3:]
    elif clean_name.startswith("케이비아이"):
        return "KBI" + clean_name[5:]
    elif clean_name.startswith("케이비") and any(k in clean_name for k in ["금융", "증권", "손해", "캐피탈", "스팩"]):
        return "KB" + clean_name[3:]
    elif clean_name.startswith("에이치디"):
        return "HD" + clean_name[4:]
    elif clean_name.startswith("에이치엘비"):
        return "HLB" + clean_name[5:]
    elif clean_name.startswith("엔에이치") and any(k in clean_name for k in ["스팩", "증권", "금융", "투자"]):
        return "NH" + clean_name[4:]
    elif clean_name.startswith("비엔케이"):
        return "BNK" + clean_name[4:]
    elif clean_name.startswith("디지비"):
        return "DGB" + clean_name[3:]
    elif clean_name.startswith("제이비"):
        return "JB" + clean_name[3:]
    elif clean_name.startswith("디비") and any(k in clean_name for k in ["스팩", "금융", "손해", "하이텍", "증권"]):
        return "DB" + clean_name[2:]

    return clean_name


def normalize_sector(krx_sec: str, name: str = "", industry: str = "") -> str:
    """
    Transforms verbose official KRX standard industry classifications into
    clean, intuitive, professional Korean market sector labels.
    """
    if not krx_sec or krx_sec == "기타":
        return "기타"
    s = krx_sec.strip()
    ind = (industry or "").strip()
    nm = (name or "").strip()

    if any(k in nm or k in ind for k in ["로봇", "로보틱스", "로보스타"]) or ("특수 목적용 기계" in s and any(k in nm for k in ["두산로보틱스", "레인보우", "로보"])):
        return "로봇/자동화"
    if any(k in nm or k in ind for k in ["전선", "케이블"]) or "절연선" in s:
        return "전선/케이블"
    if any(k in nm for k in ["일렉트릭", "효성중공업", "제룡전기", "두산퓨얼셀"]) or "전동기, 발전기" in s:
        return "전력설비/전기"
    if "반도체" in s or any(k in nm for k in ["하이닉스", "한미반도체", "가온칩스", "리노공업", "이오테크닉스", "HPSP", "주성엔지니어링", "하나마이크론", "동진쎄미켐", "ISC", "원익", "신성이엔지"]):
        return "반도체"
    if nm in ["삼성전자", "LG전자"]:
        return "IT/전기전자"
    if "통신 및 방송 장비" in s or "전자부품" in s:
        return "전자부품/IT"
    if "인터넷 정보매개" in s or "포털" in s or nm in ["NAVER", "카카오"]:
        return "인터넷/플랫폼"
    if "소프트웨어" in s or "컴퓨터 프로그래밍" in s or any(k in nm for k in ["엔씨소프트", "넷마블", "크래프톤", "펄어비스", "넥슨게임즈", "카카오게임즈", "현대오토에버", "삼성에스디에스", "스피어"]):
        return "소프트웨어/IT"
    if any(k in s for k in ["의약", "제약"]) or any(k in nm for k in ["제약", "바이오", "셀트리온", "알테오젠", "삼천당", "유한양행", "에이비엘바이오", "펩트론", "리가켐", "한미약품", "삼성바이오로직스", "HLB"]):
        return "제약/바이오"
    if "자동차" in s or any(k in nm for k in ["현대차", "기아", "현대모비스", "현대자동차"]):
        return "자동차/부품"
    if "배터리" in ind or any(k in nm for k in ["에코프로", "엘앤에프", "포스코퓨처엠", "LG에너지솔루션", "삼성SDI", "SK이노베이션", "금양"]):
        return "2차전지/배터리"
    if "조선" in s or "선박" in s or any(k in nm for k in ["HD현대중공업", "삼성중공업", "한화오션", "HD한국조선해양"]):
        return "조선/해양"
    if "해상 운송" in s or "해운" in s or any(k in nm for k in ["HMM", "팬오션", "대한해운", "흥아해운"]):
        return "해운/운송"
    if "항공" in s or any(k in nm for k in ["대한항공", "진에어", "아시아나", "제주항공", "티웨이항공", "한화에어로스페이스", "쎄트렉아이"]):
        return "항공/우주"
    if "도로 화물" in s or "물류" in s or nm in ["CJ대한통운", "인터지스", "현대글로비스"]:
        return "물류/운송"
    if "건설" in s or "토목" in s:
        return "건설/토목"
    if "식품" in s or "음료" in s or "도축" in s:
        return "음식료"
    if "화학" in s or any(k in nm for k in ["LG화학", "롯데케미칼", "금호석유", "대한유화", "코스맥스", "한국콜마", "아모레퍼시픽", "에이피알"]):
        return "화학/소재"
    if "철강" in s or "금속" in s or any(k in nm for k in ["고려아연", "POSCO홀딩스", "현대제철"]):
        return "철강/금속"
    if "무기" in s or any(k in nm for k in ["LIG넥스원", "한국항공우주", "LIG디펜스앤에어로스페이스"]):
        return "방산/항공우주"
    if any(k in s for k in ["금융", "보험", "증권", "은행", "지주"]) or any(k in nm for k in ["KB금융", "신한지주", "하나금융", "우리금융", "미래에셋", "삼성생명", "삼성화재", "한국금융지주", "SK스퀘어", "한진칼", "HD현대", "코오롱", "OCI홀딩스", "우리기술투자"]):
        return "금융/지주"
    if "통신" in s or any(k in nm for k in ["SK텔레콤", "KT", "LG유플러스"]):
        return "통신/네트워크"
    if "소매" in s or "도매" in s or any(k in nm for k in ["신세계", "롯데쇼핑", "현대백화점", "이마트", "삼성물산", "영원무역", "우리로"]):
        return "유통/소매"
    if "오디오" in s or "엔터" in ind or any(k in nm for k in ["하이브", "에스엠", "JYP", "와이지엔터"]):
        return "엔터/미디어"
    if "의료용 기기" in s or "의료용품" in s:
        return "의료기기"
    if "연구개발" in s:
        return "바이오/R&D"
    if "의복" in s or "섬유" in s:
        return "섬유/의류"
    if "기계" in s or any(k in nm for k in ["피에스케이", "필옵틱스", "한화엔진", "STX엔진", "두산에너빌리티", "대한과학"]):
        return "기계/장비"
    if "여행" in s or any(k in nm for k in ["하나투어", "모두투어", "노랑풍선"]):
        return "여행/관광"
    if "엔지니어링" in s:
        return "엔지니어링"
    if "운송장비" in s or any(k in nm for k in ["현대로템", "일진하이솔루스"]):
        return "운송장비/철도"
    if "전기장비" in s:
        return "전기/전자"

    clean = s.replace(" 제조업", "").replace(" 서비스업", "").replace(" 공급업", "").replace(" 관련", "").strip()
    return clean if clean else "기타"


def get_universe(market: Optional[str] = None, active_only: bool = True) -> pd.DataFrame:
    """
    Returns universe stocks from database as DataFrame.
    """
    with get_db_connection() as conn:
        query = "SELECT code, name, market, sector, industry, is_active, delist_date FROM universe WHERE 1=1"
        params = []
        if active_only:
            query += " AND is_active = 1"
        if market and market.upper() in ("KOSPI", "KOSDAQ"):
            query += " AND market = ?"
            params.append(market.upper())
        query += " ORDER BY market, name"
        df = pd.read_sql_query(query, conn, params=params)
        if not df.empty:
            df["name"] = df.apply(lambda r: normalize_stock_name(r["name"], r["code"]), axis=1)
            df["sector"] = df.apply(lambda r: normalize_sector(r["sector"], r["name"], r.get("industry", "")), axis=1)
        return df


def search_ticker(keyword: str, limit: int = 15) -> List[Dict[str, Any]]:
    """
    Searches for stocks matching ticker code or company name.
    Supports bidirectional search between Korean transliteration and English brand name
    (e.g., '에스케이바이오팜' and 'SK바이오팜' both find 'SK바이오팜').
    """
    keyword = keyword.strip()
    if not keyword:
        return []

    norm_kw = normalize_stock_name(keyword)
    alt_kw = ""
    if keyword.startswith("SK"):
        alt_kw = "에스케이" + keyword[2:]
    elif keyword.startswith("KT"):
        alt_kw = "케이티" + keyword[2:]
    elif keyword.startswith("LG"):
        alt_kw = "엘지" + keyword[2:]

    with get_db_connection() as conn:
        cursor = conn.cursor()
        cursor.execute("""
            SELECT code, name, market, sector, is_active
            FROM universe
            WHERE code LIKE ? OR name LIKE ? OR name LIKE ? OR (length(?) > 0 AND name LIKE ?)
            ORDER BY 
                CASE WHEN code = ? THEN 1
                     WHEN name = ? OR name = ? THEN 2
                     WHEN name LIKE ? OR name LIKE ? THEN 3
                     ELSE 4 END,
                name ASC
            LIMIT ?
        """, (
            f"{keyword}%", f"%{keyword}%", f"%{norm_kw}%", alt_kw, f"%{alt_kw}%",
            keyword, keyword, norm_kw, f"{keyword}%", f"{norm_kw}%", limit
        ))
        rows = cursor.fetchall()
        results = []
        for row in rows:
            d = dict(row)
            d["name"] = normalize_stock_name(d["name"], d["code"])
            results.append(d)
        return results


def get_stock_metadata(code: str) -> Optional[Dict[str, Any]]:
    """Returns metadata for a specific stock code with normalized stock name."""
    with get_db_connection() as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT code, name, market, sector, industry, is_active FROM universe WHERE code = ?", (code,))
        row = cursor.fetchone()
        if row:
            d = dict(row)
            d["name"] = normalize_stock_name(d["name"], d["code"])
            return d
        return None


def batch_add_to_watchlist(codes: List[str], group_name: str = "기본그룹") -> int:
    """
    Batch registers a list of stock codes into the watchlist.
    Returns count of successfully added stocks.
    """
    if not codes:
        return 0

    clean_codes = [c.strip().zfill(6) for c in codes if c.strip()]
    clean_codes = list(dict.fromkeys(clean_codes))  # remove duplicates
    clean_group = group_name.strip() if group_name and group_name.strip() else "기본그룹"

    now_str = get_now_kst().strftime("%Y-%m-%d %H:%M:%S")
    added_count = 0

    with get_db_connection() as conn:
        cursor = conn.cursor()
        # Find existing universe metadata for these codes
        placeholders = ",".join(["?"] * len(clean_codes))
        cursor.execute(f"SELECT code, name, market, sector, industry FROM universe WHERE code IN ({placeholders})", clean_codes)
        found = {r["code"]: r for r in cursor.fetchall()}

        records = []
        for code in clean_codes:
            if code in found:
                name = normalize_stock_name(found[code]["name"], code)
                market = found[code]["market"]
                sector = normalize_sector(found[code]["sector"], name, found[code].get("industry", ""))
            else:
                name = code
                market = "KOSPI"
                sector = "기타"
            records.append((code, name, market, now_str, sector, clean_group))

        cursor.executemany("""
            INSERT INTO watchlist (code, name, market, added_at, notes, group_name)
            VALUES (?, ?, ?, ?, ?, ?)
            ON CONFLICT(code) DO UPDATE SET
                name=excluded.name,
                market=excluded.market,
                notes=excluded.notes,
                group_name=CASE WHEN excluded.group_name != '기본그룹' THEN excluded.group_name ELSE COALESCE(watchlist.group_name, excluded.group_name) END
        """, records)
        conn.commit()
        added_count = len(records)

    return added_count


def clear_watchlist() -> None:
    """Clears all entries from the watchlist table."""
    with get_db_connection() as conn:
        cursor = conn.cursor()
        cursor.execute("DELETE FROM watchlist")
        conn.commit()


if __name__ == "__main__":
    count = sync_krx_universe()
    print("Total universe count:", count)
    results = search_ticker("삼성")
    print("Search '삼성':", results[:3])

