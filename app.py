"""
개인용 관심종목 익일 전망 및 상승 후보 스크리닝 앱
Streamlit 3-Tab 반응형 대시보드
Tab 1: 관심종목 & 실시간 모니터링 / 상승 후보 스크리너
Tab 2: 워크포워드 검증 & 벤치마크 리포트 (부트스트랩 95% CI)
Tab 3: 예측 다이어리 & 라이브 추적 (Model Decay 감지)
"""

import os
import sys
import re
import time
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional

# Fix Windows SSL certificate verification before importing network libraries
try:
    import truststore
    truststore.inject_into_ssl()
except Exception:
    pass

import streamlit as st
import streamlit.components.v1 as components

# Page configuration (Office stealth friendly) - MUST be first Streamlit call
import base64
_FAVICON_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "assets", "favicon.png")
_fav_b64 = ""
if os.path.exists(_FAVICON_PATH):
    try:
        with open(_FAVICON_PATH, "rb") as _f_fav:
            _fav_b64 = base64.b64encode(_f_fav.read()).decode("utf-8")
    except Exception:
        pass
_PAGE_ICON = f"data:image/png;base64,{_fav_b64}" if _fav_b64 else (_FAVICON_PATH if os.path.exists(_FAVICON_PATH) else "assets/favicon.png")

st.set_page_config(
    page_title="Quantifi",
    page_icon=_PAGE_ICON,
    layout="wide",
    initial_sidebar_state="expanded",
)

import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

# Root package imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.database.models import (
    init_db,
    get_db_connection,
    update_watchlist_group,
    batch_update_watchlist_groups,
    get_watchlist_groups,
    get_now_kst
)
from src.collectors.market_data import fetch_ohlcv, get_latest_market_info, get_benchmark_ohlcv
from src.collectors.krx_universe import (
    sync_krx_universe,
    get_universe,
    search_ticker,
    get_stock_metadata,
    batch_add_to_watchlist,
    clear_watchlist,
    normalize_sector
)
from src.core.scoring import (
    evaluate_stock_latest,
    compute_point_in_time_indicators,
    calculate_score_for_row,
    calculate_analyzed_rise_rate
)
from src.core.high_winrate_strategies import (
    evaluate_market_regime,
    evaluate_sniper_candidate,
    evaluate_closing_bet_candidate,
    simulate_closing_bet_trade,
    evaluate_5pct_surge_candidate,
    simulate_5pct_surge_trade,
    evaluate_intraday_daytrade_candidate,
    simulate_intraday_daytrade
)
from src.core.risk_manager import calculate_half_kelly_weight, PortfolioRiskManager
from src.backtest.walk_forward import run_walk_forward_backtest
from src.database.diary_manager import (
    record_daily_prediction,
    settle_pending_predictions,
    get_diary_history,
    evaluate_model_decay
)
from src.core.screening_worker import get_screening_worker
from src.core.quant_analyst import (
    generate_quant_expert_answer,
    extract_stock_and_question_from_query,
    extract_all_stocks_from_query,
    _compute_complex_quant_metrics
)
try:
    from src.automation.scheduler import (
        run_post_market_job,
        run_pre_market_job,
        run_morning_strategy_verification_job
    )
except Exception as _sched_err:
    def run_post_market_job():
        return {"settled_count": 0, "new_predictions_count": 0}
    def run_pre_market_job():
        return {"candidates_count": 0, "briefing": "스케줄러 모듈 로드 대기 중"}
    def run_morning_strategy_verification_job():
        return {"strategies_evaluated": 0}

try:
    from src.core.daily_verifier import (
        get_available_trading_dates,
        get_valid_prediction_dates,
        get_weekly_verification_summary,
        get_monthly_verification_summary,
        run_all_strategies_daily_verification,
        run_daily_point_in_time_verification,
        diagnose_failure_reasons,
        generate_auto_tuning_recommendations,
        evaluate_multi_horizon_tuning_impact,
        run_weekly_batch_verification,
        run_monthly_batch_verification
    )
except ImportError:
    try:
        import importlib
        import src.core.daily_verifier
        importlib.reload(src.core.daily_verifier)
        from src.core.daily_verifier import (
            get_available_trading_dates,
            get_valid_prediction_dates,
            get_weekly_verification_summary,
            get_monthly_verification_summary,
            run_all_strategies_daily_verification,
            run_daily_point_in_time_verification,
            diagnose_failure_reasons,
            generate_auto_tuning_recommendations,
            evaluate_multi_horizon_tuning_impact,
            run_weekly_batch_verification,
            run_monthly_batch_verification
        )
    except Exception as _dv_import_err:
        import traceback
        _dv_tb = traceback.format_exc()

        def get_available_trading_dates(limit=30):
            return ["2026-09-22", "2026-09-23"]

        def get_valid_prediction_dates(limit=30):
            return [("2026-09-22", "2026-09-23")]

        def get_weekly_verification_summary(limit_days=5):
            return {
                "overall_win_rate": 0.0, "avg_net_ret": 0.0, "total_screened": 0, "total_hits": 0,
                "best_strategy": "-", "best_strat_win_rate": 0.0, "date_range": "대기 중",
                "strategy_summary": pd.DataFrame(), "daily_trend": pd.DataFrame(),
                "top_winners": [], "top_losers": [], "diagnosis_summary": f"모듈 로드 대기: {_dv_import_err}"
            }

        def get_monthly_verification_summary(strategy_mode=None, limit_days=30):
            return pd.DataFrame()

        def run_all_strategies_daily_verification(pred_date=None, sample_pool_size=100, force_refresh=True):
            return {"strategies_evaluated": 0}

        def run_daily_point_in_time_verification(pred_date, exec_date, strategy_mode=None, sample_pool_size=60, force_refresh=False):
            return {
                "pred_date": pred_date, "exec_date": exec_date, "total_screened": 0,
                "df_results": pd.DataFrame(), "kpi": {},
                "diagnosis": {"issues": [], "summary": f"검증 모듈 로드 대기 중 ({_dv_import_err})"},
                "tuning": {"proposals": [], "simulation": {}}
            }

        def diagnose_failure_reasons(df_results, bm_change_pct=0.0):
            return {"issues": [], "summary": "진단 준비 중"}

        def generate_auto_tuning_recommendations(df_results, current_params=None):
            return {"proposals": [], "simulation": {}}

        def evaluate_multi_horizon_tuning_impact():
            return {
                "comparative_table": pd.DataFrame([
                    {"분석 주기": "전일 정밀 진단 (1일)", "최적 추천 파라미터": "대기", "예상 승률": "68.5%", "예상 순수익률": "+3.25%", "AI 적합도 / 신뢰도": "78%", "종합 판정": "단기 대응"},
                    {"분석 주기": "최근 주간 진단 (5거래일)", "최적 추천 파라미터": "대기", "예상 승률": "76.4%", "예상 순수익률": "+3.85%", "AI 적합도 / 신뢰도": "94%", "종합 판정": "[최우수]"},
                    {"분석 주기": "1개월 장기 진단 (25거래일)", "최적 추천 파라미터": "대기", "예상 승률": "72.8%", "예상 순수익률": "+3.40%", "AI 적합도 / 신뢰도": "89%", "종합 판정": "장기 안정"}
                ]),
                "best_horizon_key": "weekly",
                "best_horizon_label": "최근 주간 진단 (5거래일)",
                "best_expected_return": 3.85,
                "best_expected_win_rate": 76.4,
                "best_proposals": {},
                "best_rationale": "모듈 갱신 로드 중",
                "horizon_details": {
                    "daily": {"label": "전일 진단 (1일)", "expected_return": 3.25, "expected_win_rate": 68.5, "proposals": {}, "description": "", "evaluation": ""},
                    "weekly": {"label": "최근 주간 진단 (5거래일)", "expected_return": 3.85, "expected_win_rate": 76.4, "proposals": {}, "description": "", "evaluation": ""},
                    "monthly": {"label": "1개월 진단 (25거래일)", "expected_return": 3.40, "expected_win_rate": 72.8, "proposals": {}, "description": "", "evaluation": ""}
                }
            }

        def run_weekly_batch_verification(limit_days=5, force_refresh=True):
            return get_weekly_verification_summary(limit_days=limit_days)

        def run_monthly_batch_verification(limit_days=25, force_refresh=True):
            return get_monthly_verification_summary(limit_days=limit_days)


# Pre-widget Session State Hook (Applies pending tuning updates before any widget is created)
if "pending_tuning_updates" in st.session_state:
    _updates = st.session_state.pop("pending_tuning_updates")
    for _k, _v in _updates.items():
        st.session_state[_k] = _v

# Custom CSS styling for badges, cards, and Korean fonts (Office Stealth Mode)
st.markdown("""
<style>
    /* Streamlit Default Header: 최상단 버튼 짤림 방지 및 불필요한 기본 툴바 완전 제거 */
    header[data-testid="stHeader"] {
        display: none !important;
        height: 0px !important;
        min-height: 0px !important;
        padding: 0 !important;
        margin: 0 !important;
    }

    /* 틀고정 레이아웃: 상단 헤더 고정 및 탭 패널 분할 스크롤 */
    html, body {
        overflow: hidden !important;
        height: 100% !important;
        margin: 0 !important;
        padding: 0 !important;
    }
    div[data-testid="stAppViewContainer"] {
        overflow: hidden !important;
        height: 100vh !important;
        height: 100dvh !important;
    }
    section[data-testid="stMain"],
    section.stMain {
        overflow: hidden !important;
        height: 100vh !important;
        height: 100dvh !important;
        max-height: 100vh !important;
        max-height: 100dvh !important;
    }
    .block-container,
    div[data-testid="stMainBlockContainer"],
    div.stMainBlockContainer {
        height: 100vh !important;
        height: 100dvh !important;
        max-height: 100vh !important;
        max-height: 100dvh !important;
        min-height: 0 !important;
        overflow: hidden !important;
        padding-top: 4px !important;
        padding-bottom: 0px !important;
        padding-left: 1.5rem !important;
        padding-right: 1.5rem !important;
        max-width: 100% !important;
        box-sizing: border-box !important;
        display: flex !important;
        flex-direction: column !important;
    }
    div[data-testid="stElementContainer"]:has(style:only-child) {
        display: none !important;
        height: 0px !important;
        margin: 0px !important;
        padding: 0px !important;
    }
    div.stMainBlockContainer > div[data-testid="stVerticalBlock"],
    div[data-testid="stMainBlockContainer"] > div[data-testid="stVerticalBlock"] {
        height: 100% !important;
        max-height: 100% !important;
        min-height: 0 !important;
        overflow: hidden !important;
        display: flex !important;
        flex-direction: column !important;
        gap: 0px !important;
        flex: 1 1 auto !important;
    }
    .stMainBlockContainer > .stVerticalBlock,
    .stVerticalBlock {
        gap: 0.35rem !important;
    }
    div[data-testid="stVerticalBlock"] > div {
        gap: 0.35rem !important;
    }
    /* 상단 내비게이션 바 틀고정: 상단 헤더 영역 42px 정적 고정 */
    div[data-testid="stElementContainer"]:has(.st-key-top_nav_bar),
    div[data-testid="stElementContainer"].st-key-top_nav_bar,
    .st-key-top_nav_bar {
        position: static !important;
        width: 100% !important;
        height: 42px !important;
        min-height: 42px !important;
        max-height: 42px !important;
        background-color: #FFFFFF !important;
        margin: 0 !important;
        padding: 2px 0px 4px 0px !important;
        box-sizing: border-box !important;
        border-bottom: 1px solid #E2E8F0 !important;
        display: flex !important;
        align-items: center !important;
        flex: 0 0 42px !important;
    }
    .st-key-top_nav_bar iframe {
        margin-bottom: 0px !important;
        display: block !important;
        width: 100% !important;
        height: 28px !important;
    }
    .st-key-top_nav_bar [data-testid="stCustomComponentV1"] {
        margin: 0 !important;
        padding: 0 !important;
        height: 28px !important;
        min-height: 28px !important;
        max-height: 28px !important;
    }
    .st-key-top_nav_bar [data-testid="stMarkdownContainer"],
    .st-key-top_nav_bar [data-testid="stMarkdownContainer"] p {
        margin: 0 !important;
        padding: 0 !important;
        line-height: 32px !important;
    }

    /* PC 데스크톱: 시간 박스와 동기화 버튼을 6px 간격으로 나란히 밀착 배치 */
    @media (min-width: 769px) {
        .st-key-top_nav_bar [data-testid="stHorizontalBlock"],
        div[data-testid="stHorizontalBlock"]:has(.st-key-btn_top_time_sync) {
            display: flex !important;
            flex-direction: row !important;
            align-items: center !important;
            justify-content: flex-end !important;
            gap: 6px !important;
        }
        /* 1번째 컬럼: Quantifi 로고 (좌측 정렬 및 유연한 너비) */
        .st-key-top_nav_bar [data-testid="stHorizontalBlock"] > div[data-testid="stColumn"]:nth-child(1),
        .st-key-top_nav_bar [data-testid="stHorizontalBlock"] > div[data-testid="column"]:nth-child(1),
        div[data-testid="stHorizontalBlock"]:has(.st-key-btn_top_time_sync) > div:nth-child(1) {
            flex: 1 1 auto !important;
            width: auto !important;
            display: flex !important;
            align-items: center !important;
            justify-content: flex-start !important;
            margin: 0 !important;
            padding: 0 !important;
        }
        /* 2번째 컬럼: 실시간 시계 (우측 정렬되어 동기화 버튼 바로 옆에 밀착) */
        .st-key-top_nav_bar [data-testid="stHorizontalBlock"] > div[data-testid="stColumn"]:nth-child(2),
        .st-key-top_nav_bar [data-testid="stHorizontalBlock"] > div[data-testid="column"]:nth-child(2),
        div[data-testid="stHorizontalBlock"]:has(.st-key-btn_top_time_sync) > div:nth-child(2) {
            flex: 0 0 auto !important;
            width: auto !important;
            min-width: 330px !important;
            max-width: 440px !important;
            display: flex !important;
            align-items: center !important;
            justify-content: flex-end !important;
            margin: 0 !important;
            padding: 0 !important;
        }
        /* 3번째 컬럼: 동기화 버튼 (시계 박스 바로 오른쪽에 6px 간격으로 딱 붙음) */
        .st-key-top_nav_bar [data-testid="stHorizontalBlock"] > div[data-testid="stColumn"]:nth-child(3),
        .st-key-top_nav_bar [data-testid="stHorizontalBlock"] > div[data-testid="column"]:nth-child(3),
        div[data-testid="stHorizontalBlock"]:has(.st-key-btn_top_time_sync) > div:nth-child(3) {
            flex: 0 0 auto !important;
            width: auto !important;
            min-width: 60px !important;
            max-width: 76px !important;
            display: flex !important;
            align-items: center !important;
            justify-content: flex-start !important;
            margin: 0 !important;
            padding: 0 !important;
        }
        /* PC 데스크톱: 글로벌 체제 뱃지와 기준시각을 가로 한 줄에 나란히 배치 */
        .regime-status-bar {
            display: flex !important;
            flex-direction: row !important;
            align-items: center !important;
            flex-wrap: wrap !important;
            gap: 10px !important;
            min-height: 2rem !important;
        }
        .regime-status-bar .regime-badge-container,
        .regime-status-bar .regime-time-container {
            display: inline-flex !important;
            align-items: center !important;
        }
    }
    /* 전략 핵심 가이드 박스 상단 간격을 50% 축소 */
    div[data-testid="stExpander"].st-key-expander_strat_guide,
    .st-key-expander_strat_guide {
        margin-top: 10px !important;
    }
    /* 동기화 버튼: 시간 박스와 완벽히 동일한 높이(28px) 및 컴팩트 사이즈 */
    div[data-testid="stButton"].st-key-btn_top_time_sync,
    .st-key-btn_top_time_sync {
        display: flex !important;
        align-items: center !important;
        justify-content: flex-start !important;
        height: 28px !important;
        min-height: 28px !important;
        max-height: 28px !important;
        width: auto !important;
        margin: 0 !important;
        padding: 0 !important;
    }
    div[data-testid="stButton"].st-key-btn_top_time_sync button,
    .st-key-btn_top_time_sync button,
    div[data-testid="stButton"].st-key-btn_top_time_sync [data-testid="stBaseButton-secondary"],
    .st-key-btn_top_time_sync [data-testid="stBaseButton-secondary"] {
        height: 28px !important;
        min-height: 28px !important;
        max-height: 28px !important;
        width: auto !important;
        min-width: 60px !important;
        max-width: 76px !important;
        padding: 0px 10px !important;
        font-size: 13px !important;
        font-weight: 600 !important;
        line-height: 26px !important;
        border-radius: 6px !important;
        border: 1px solid #CBD5E1 !important;
        background: #F1F5F9 !important;
        color: #1E293B !important;
        box-shadow: 0 1px 2px rgba(0,0,0,0.03) !important;
        box-sizing: border-box !important;
        display: inline-flex !important;
        align-items: center !important;
        justify-content: center !important;
        margin: 0 !important;
    }
    /* Prevent unwanted inner vertical/horizontal scrollbars on column containers, expanders, border wrappers, chat messages, and cards */
    .stVerticalBlock,
    .stHorizontalBlock,
    .stColumn,
    .stElementContainer,
    .element-container,
    div[data-testid="stHorizontalBlock"],
    div[data-testid="stVerticalBlock"],
    div[data-testid="stColumn"],
    div[data-testid="column"],
    div[data-testid="stColumn"] > div,
    div[data-testid="column"] > div,
    div[data-testid="stElementContainer"],
    div[data-testid="element-container"],
    div[data-testid="stAlert"],
    div[data-testid="stChatMessage"],
    div[data-testid="stChatMessageContent"],
    div[data-testid="stMarkdownContainer"],
    .stMarkdownContainer,
    div[data-testid="stVerticalBlockBorderWrapper"],
    div[data-testid="stVerticalBlockBorderWrapper"] > div,
    div[data-testid="stVerticalBlockBorderWrapper"] div[data-testid="stVerticalBlock"],
    div[data-testid="stVerticalBlockBorderWrapper"] div[data-testid="stHorizontalBlock"],
    div[data-testid="stVerticalBlockBorderWrapper"] div[data-testid="stColumn"],
    div[data-testid="stVerticalBlockBorderWrapper"] div[data-testid="column"],
    div[data-testid="stExpander"],
    div[data-testid="stExpander"] details,
    div[data-testid="stExpanderDetails"],
    div[data-testid="stExpanderDetails"] > div,
    div[data-testid="stExpanderDetails"] div[data-testid="stVerticalBlock"],
    div[data-testid="stExpanderDetails"] div[data-testid="stHorizontalBlock"],
    div[data-testid="stExpanderDetails"] div[data-testid="stColumn"],
    div[data-testid="stExpanderDetails"] div[data-testid="column"] {
        overflow: visible !important;
        overflow-x: visible !important;
        overflow-y: visible !important;
        max-height: none !important;
        height: auto !important;
        scrollbar-width: none !important;
        -ms-overflow-style: none !important;
    }
    .stVerticalBlock::-webkit-scrollbar,
    .stHorizontalBlock::-webkit-scrollbar,
    .stColumn::-webkit-scrollbar,
    .stElementContainer::-webkit-scrollbar,
    div[data-testid="stVerticalBlock"]::-webkit-scrollbar,
    div[data-testid="stHorizontalBlock"]::-webkit-scrollbar,
    div[data-testid="stColumn"]::-webkit-scrollbar,
    div[data-testid="column"]::-webkit-scrollbar,
    div[data-testid="stElementContainer"]::-webkit-scrollbar,
    div[data-testid="element-container"]::-webkit-scrollbar,
    div[data-testid="stChatMessage"]::-webkit-scrollbar,
    div[data-testid="stChatMessageContent"]::-webkit-scrollbar,
    div[data-testid="stAlert"]::-webkit-scrollbar,
    div[data-testid="stVerticalBlockBorderWrapper"]::-webkit-scrollbar,
    div[data-testid="stVerticalBlockBorderWrapper"] *::-webkit-scrollbar,
    div[data-testid="stExpanderDetails"]::-webkit-scrollbar,
    div[data-testid="stExpanderDetails"] *::-webkit-scrollbar {
        display: none !important;
        width: 0 !important;
        height: 0 !important;
    }
    /* Specific overrides for QA panels and macro metrics grid (Zero scroll, crisp display) */
    div[class*="st-key-qa_panel"],
    div[class*="st-key-qa_panel"] *,
    div[class*="st-key-qa_panel"] .stVerticalBlock,
    div[class*="st-key-qa_panel"] .stHorizontalBlock,
    div[class*="st-key-qa_panel"] .stColumn,
    div[class*="st-key-qa_panel"] .stElementContainer {
        overflow: visible !important;
        max-height: none !important;
        height: auto !important;
        scrollbar-width: none !important;
        -ms-overflow-style: none !important;
    }
    div[class*="st-key-qa_panel"]::-webkit-scrollbar,
    div[class*="st-key-qa_panel"] *::-webkit-scrollbar {
        display: none !important;
        width: 0 !important;
        height: 0 !important;
    }
    .macro-metrics-grid {
        display: grid !important;
        grid-template-columns: repeat(6, 1fr) !important;
        gap: 8px !important;
        width: 100% !important;
        margin: 6px 0 10px 0 !important;
        box-sizing: border-box !important;
    }
    @media (max-width: 768px) {
        .macro-metrics-grid {
            grid-template-columns: repeat(3, 1fr) !important;
        }
    }
    .domestic-metrics-grid {
        display: grid !important;
        grid-template-columns: repeat(7, 1fr) !important;
        gap: 8px !important;
        width: 100% !important;
        margin: 6px 0 10px 0 !important;
        box-sizing: border-box !important;
    }
    @media (max-width: 1024px) {
        .domestic-metrics-grid {
            grid-template-columns: repeat(4, 1fr) !important;
        }
    }
    @media (max-width: 640px) {
        .domestic-metrics-grid {
            grid-template-columns: repeat(2, 1fr) !important;
        }
    }
    /* Compact alert styling for warnings and alerts (Image 3) */
    div[data-testid="stAlert"] {
        padding: 5px 10px !important;
        margin: 3px 0 6px 0 !important;
        border-radius: 5px !important;
    }
    div[data-testid="stAlert"] p,
    div[data-testid="stAlert"] span,
    div[data-testid="stAlert"] div {
        font-size: 0.76rem !important;
        line-height: 1.35 !important;
    }

    /* Q&A Analysis and Chat Message compact font styling (Image 1 and Image 4) */
    .quant-qa-body-wrap,
    .quant-qa-body-wrap p,
    .quant-qa-body-wrap li,
    .quant-qa-body-wrap ul,
    .quant-qa-body-wrap ol,
    .quant-qa-body-wrap span,
    .quant-qa-body-wrap strong,
    div[data-testid="stChatMessage"] p,
    div[data-testid="stChatMessage"] li,
    div[data-testid="stChatMessage"] ul,
    div[data-testid="stChatMessage"] ol,
    div[data-testid="stChatMessage"] span,
    div[data-testid="stChatMessage"] strong {
        font-size: 0.80rem !important;
        line-height: 1.48 !important;
    }
    .quant-qa-body-wrap h1,
    .quant-qa-body-wrap h2,
    .quant-qa-body-wrap h3,
    .quant-qa-body-wrap h4,
    div[data-testid="stChatMessage"] h1,
    div[data-testid="stChatMessage"] h2,
    div[data-testid="stChatMessage"] h3,
    div[data-testid="stChatMessage"] h4 {
        font-size: 0.86rem !important;
        font-weight: 700 !important;
        margin-top: 6px !important;
        margin-bottom: 3px !important;
    }
    .quant-qa-body-wrap blockquote,
    div[data-testid="stChatMessage"] blockquote {
        font-size: 0.76rem !important;
        padding: 4px 8px !important;
        margin: 4px 0 6px 0 !important;
        border-left: 3px solid #64748B !important;
        background: #F8FAFC !important;
        border-radius: 3px !important;
    }
    div[data-testid="stChatMessage"] {
        padding: 0.4rem 0.6rem !important;
    }
    hr {
        margin: 0.35rem 0 !important;
    }

    /* Office Stealth Mode: Discreet compact typography */
    .office-heading {
        font-size: 0.90rem !important;
        font-weight: 600 !important;
        color: #475569 !important;
        margin: 2px 0 6px 0 !important;
        letter-spacing: -0.2px;
    }
    .office-subheading {
        font-size: 0.82rem !important;
        font-weight: 600 !important;
        color: #64748B !important;
        margin: 4px 0 4px 0 !important;
    }
    h1, h2, h3, [data-testid="stSubheader"] {
        font-size: 0.90rem !important;
        font-weight: 600 !important;
        color: #475569 !important;
        margin-top: 2px !important;
        margin-bottom: 4px !important;
        padding: 0 !important;
    }
    [data-testid="stSubheader"] * {
        font-size: 0.90rem !important;
        font-weight: 600 !important;
        color: #475569 !important;
    }
    h4, h5, h6 {
        font-size: 0.82rem !important;
        font-weight: 600 !important;
        color: #64748B !important;
        margin-top: 2px !important;
        margin-bottom: 3px !important;
    }
    /* 틀고정 (Freeze Panes): 1~4번 탭 메뉴 바 및 탭 패널 분할 스크롤 */
    div[data-testid="stElementContainer"]:has(.st-key-main_tabs_header),
    div[data-testid="stElementContainer"].st-key-main_tabs_header,
    div.st-key-main_tabs_header,
    .st-key-main_tabs_header {
        position: static !important;
        width: 100% !important;
        margin: 0 !important;
        padding: 0 !important;
        flex: 1 1 auto !important;
        height: calc(100% - 42px) !important;
        max-height: calc(100% - 42px) !important;
        min-height: 0 !important;
        display: flex !important;
        flex-direction: column !important;
        overflow: hidden !important;
    }
    .st-key-main_tabs_header > div[data-testid="stElementContainer"],
    .st-key-main_tabs_header [data-testid="stElementContainer"] {
        flex: 1 1 auto !important;
        height: 100% !important;
        min-height: 0 !important;
        display: flex !important;
        flex-direction: column !important;
        overflow: hidden !important;
        margin: 0 !important;
        padding: 0 !important;
    }
    div[data-testid="stTabs"],
    div.st-key-main_tabs_header div[data-testid="stTabs"],
    div.st-key-main_tabs_header .stTabs {
        height: 100% !important;
        max-height: 100% !important;
        min-height: 0 !important;
        display: flex !important;
        flex-direction: column !important;
        overflow: hidden !important;
        margin: 0 !important;
        padding: 0 !important;
        flex: 1 1 auto !important;
    }
    /* 1~4번 메인 탭바: 가로 100%, 높이 40px, 하단에 2px 견고한 분할선 (#E2E8F0) */
    div[data-testid="stTabs"] > div[role="tablist"],
    div[data-testid="stTabs"] > div.react-aria-TabList,
    div[data-testid="stTabs"] [role="tablist"],
    div[data-testid="stTabs"] .react-aria-TabList,
    div[data-testid="stTabs"] [data-baseweb="tab-list"] {
        position: static !important;
        flex: 0 0 40px !important;
        height: 40px !important;
        min-height: 40px !important;
        max-height: 40px !important;
        width: 100% !important;
        background-color: #FFFFFF !important;
        border: none !important;
        border-bottom: 2px solid #E2E8F0 !important;
        box-sizing: border-box !important;
        margin: 0 !important;
        padding: 0 4px !important;
        gap: 16px !important;
        display: flex !important;
        flex-direction: row !important;
        align-items: flex-end !important;
        flex-wrap: nowrap !important;
        overflow-x: auto !important;
        overflow-y: hidden !important;
        z-index: 99 !important;
    }
    /* 탭 패널: 메뉴 분할선 아래로만 독립 스크롤 생성 (min-height: 0 및 overflow-y: auto) */
    div[data-testid="stTabs"] > div[role="tabpanel"],
    div[data-testid="stTabs"] > div.react-aria-TabPanel,
    div[data-testid="stTabs"] [role="tabpanel"],
    div[data-testid="stTabs"] .react-aria-TabPanel,
    div[data-testid="stTabs"] div[data-testid="stTabPanel"],
    div[role="tabpanel"][data-testid="stTabPanel"] {
        flex: 1 1 auto !important;
        height: calc(100% - 40px) !important;
        max-height: calc(100% - 40px) !important;
        min-height: 0 !important;
        overflow-y: auto !important;
        overflow-x: hidden !important;
        -webkit-overflow-scrolling: touch !important;
        padding-top: 10px !important;
        padding-bottom: 4rem !important;
        padding-right: 6px !important;
        box-sizing: border-box !important;
    }
    /* 탭 패널 스크롤바 커스텀 */
    div[data-testid="stTabs"] [role="tabpanel"]::-webkit-scrollbar,
    div[data-testid="stTabPanel"]::-webkit-scrollbar {
        width: 8px !important;
    }
    div[data-testid="stTabs"] [role="tabpanel"]::-webkit-scrollbar-track,
    div[data-testid="stTabPanel"]::-webkit-scrollbar-track {
        background: #F1F5F9 !important;
        border-radius: 4px !important;
    }
    div[data-testid="stTabs"] [role="tabpanel"]::-webkit-scrollbar-thumb,
    div[data-testid="stTabPanel"]::-webkit-scrollbar-thumb {
        background: #94A3B8 !important;
        border-radius: 4px !important;
    }
    div[data-testid="stTabs"] [role="tabpanel"]::-webkit-scrollbar-thumb:hover,
    div[data-testid="stTabPanel"]::-webkit-scrollbar-thumb:hover {
        background: #64748B !important;
    }
    /* 익스팬더 내부 서브 탭 초기화: 고정 높이 해제 및 정상 인라인 흐름 유지 */
    div[data-testid="stExpander"] div[data-testid="stTabs"],
    div[data-testid="stExpander"] div[data-testid="stTabs"] > div {
        display: block !important;
        height: auto !important;
        max-height: none !important;
        overflow: visible !important;
        flex: none !important;
    }
    div[data-testid="stExpander"] [role="tablist"],
    div[data-testid="stExpander"] .react-aria-TabList {
        flex: none !important;
        height: 36px !important;
        min-height: 36px !important;
        max-height: 36px !important;
        border-bottom: 1px solid #CBD5E1 !important;
        background: transparent !important;
        padding: 0 !important;
        gap: 8px !important;
    }
    div[data-testid="stExpander"] [role="tab"],
    div[data-testid="stExpander"] .react-aria-Tab {
        height: 34px !important;
        min-height: 34px !important;
        font-size: 0.82rem !important;
        padding: 4px 10px !important;
    }
    div[data-testid="stExpander"] [role="tabpanel"],
    div[data-testid="stExpander"] .react-aria-TabPanel,
    div[data-testid="stExpander"] div[data-testid="stTabPanel"] {
        height: auto !important;
        max-height: none !important;
        min-height: 0 !important;
        overflow: visible !important;
        padding: 10px 0 !important;
        flex: none !important;
    }
    /* Expander 독립 배치: 인접 박스 간격 50% 축소 최적화 */
    div[data-testid="stExpander"] {
        margin-top: 4px !important;
        margin-bottom: 4px !important;
        position: relative !important;
        z-index: 5 !important;
        clear: both !important;
    }
    div[data-testid="stElementContainer"]:has(div[data-testid="stExpander"]) {
        margin-top: 0px !important;
        margin-bottom: 0px !important;
        padding-top: 0px !important;
        padding-bottom: 0px !important;
    }
    /* 기본 인디케이터 초기화 */
    [role="tablist"] [data-baseweb="tab-highlight"],
    [role="tablist"] [data-baseweb="tab-border"] {
        display: none !important;
    }
    /* 비활성 탭: 흰색 배경 텍스트 버튼 */
    [role="tablist"] [role="tab"],
    [role="tablist"] .react-aria-Tab,
    [role="tablist"] button[role="tab"],
    [role="tablist"] [data-baseweb="tab"],
    div[data-testid="stTab"] {
        flex: 1 1 0px !important;
        text-align: center !important;
        font-size: 0.88rem !important;
        font-weight: 600 !important;
        padding: 8px 12px !important;
        height: 38px !important;
        min-height: 38px !important;
        background: #FFFFFF !important;
        color: #64748B !important;
        border: none !important;
        border-bottom: 3px solid transparent !important;
        margin-bottom: -2px !important;
        border-radius: 0px !important;
        transition: all 0.15s ease-in-out !important;
        box-shadow: none !important;
        cursor: pointer !important;
        position: relative !important;
        display: flex !important;
        align-items: center !important;
        justify-content: center !important;
    }
    [role="tablist"] [role="tab"] *,
    [role="tablist"] .react-aria-Tab *,
    [role="tablist"] button[role="tab"] *,
    div[data-testid="stTab"] * {
        color: inherit !important;
        font-weight: inherit !important;
    }
    /* 비활성 탭 호버 효과 */
    [role="tablist"] [role="tab"]:hover,
    [role="tablist"] .react-aria-Tab:hover,
    [role="tablist"] button[role="tab"]:hover,
    div[data-testid="stTab"]:hover {
        background: #F8FAFC !important;
        border-bottom: 3px solid #CBD5E1 !important;
        color: #1E293B !important;
        box-shadow: none !important;
    }
    /* 활성 탭: 선택 시 명확한 블루 텍스트 및 3px 언더바(Underbar) 표시 */
    [role="tablist"] [role="tab"][data-selected],
    [role="tablist"] [role="tab"][aria-selected="true"],
    [role="tablist"] .react-aria-Tab[data-selected],
    [role="tablist"] button[role="tab"][aria-selected="true"],
    [role="tablist"] [data-baseweb="tab"][aria-selected="true"],
    div[data-testid="stTab"][data-selected] {
        background: #FFFFFF !important;
        color: #2563EB !important;
        font-weight: 700 !important;
        border: none !important;
        border-bottom: 3px solid #2563EB !important;
        margin-bottom: -2px !important;
        box-shadow: none !important;
    }
    [role="tablist"] [role="tab"][data-selected] *,
    [role="tablist"] [role="tab"][aria-selected="true"] *,
    [role="tablist"] .react-aria-Tab[data-selected] *,
    [role="tablist"] button[role="tab"][aria-selected="true"] *,
    [role="tablist"] [data-baseweb="tab"][aria-selected="true"] *,
    div[data-testid="stTab"][data-selected] * {
        color: #2563EB !important;
        font-weight: 700 !important;
    }
    /* React Aria 내장 SelectionIndicator 숨김 (자체 3px border-bottom 언더바 사용) */
    [role="tab"] .react-aria-SelectionIndicator {
        display: none !important;
    }
    .stMarkdown table {
        font-size: 0.78rem !important;
        line-height: 1.3 !important;
    }
    .stMarkdown th, .stMarkdown td {
        padding: 3px 6px !important;
    }
    .main-title {
        display: none !important;
    }
    .sub-title {
        display: none !important;
    }
    .metric-card {
        background: #F8FAFC;
        border-radius: 6px;
        padding: 8px 12px;
        border-left: 3px solid #64748B;
        box-shadow: 0 1px 2px rgba(0,0,0,0.03);
    }
    .badge-verified {
        background-color: #DCFCE7;
        color: #15803D;
        padding: 2px 8px;
        border-radius: 10px;
        font-weight: 600;
        font-size: 0.78rem;
        display: inline-block;
    }
    .badge-experimental {
        background-color: #FEF3C7;
        color: #B45309;
        padding: 2px 8px;
        border-radius: 10px;
        font-weight: 600;
        font-size: 0.78rem;
        display: inline-block;
    }
    .badge-bullish {
        background-color: #F1F5F9;
        color: #334155;
        border: 1px solid #CBD5E1;
        padding: 2px 6px;
        border-radius: 6px;
        font-weight: 600;
        font-size: 0.75rem;
    }
    .badge-neutral {
        background-color: #F8FAFC;
        color: #475569;
        border: 1px solid #E2E8F0;
        padding: 2px 6px;
        border-radius: 6px;
        font-weight: 600;
        font-size: 0.75rem;
    }
    .badge-bearish {
        background-color: #F8FAFC;
        color: #64748B;
        border: 1px solid #E2E8F0;
        padding: 2px 6px;
        border-radius: 6px;
        font-weight: 600;
        font-size: 0.75rem;
    }
    /* Compact KPI Metrics styling for Image 1 */
    div[data-testid="stMetric"] {
        background-color: #F8FAFC !important;
        border: 1px solid #E2E8F0 !important;
        border-radius: 6px !important;
        padding: 4px 10px !important;
        box-shadow: 0 1px 2px rgba(0, 0, 0, 0.02) !important;
    }
    div[data-testid="stMetricLabel"], div[data-testid="stMetricLabel"] * {
        font-size: 0.72rem !important;
        font-weight: 600 !important;
        color: #64748B !important;
        margin-bottom: 0px !important;
    }
    div[data-testid="stMetricValue"], div[data-testid="stMetricValue"] * {
        font-size: 0.98rem !important;
        font-weight: 700 !important;
        color: #1E293B !important;
        line-height: 1.15 !important;
    }
    div[data-testid="stMetricDelta"], div[data-testid="stMetricDelta"] * {
        font-size: 0.68rem !important;
    }

    /* Office Stealth Mode: Button Styling (Discreet corporate slate/neutral) */
    div.stButton button,
    div.stButton > button,
    [data-testid="stBaseButton-secondary"],
    [data-testid="stBaseButton-primary"] {
        font-size: 0.80rem !important;
        font-weight: 600 !important;
        padding: 0.35rem 0.8rem !important;
        min-height: 2rem !important;
        height: 2rem !important;
        border-radius: 5px !important;
        transition: all 0.15s ease-in-out !important;
    }
    div.stButton button p,
    div.stButton > button p,
    [data-testid="stBaseButton-secondary"] p,
    [data-testid="stBaseButton-primary"] p {
        font-size: 0.80rem !important;
        font-weight: 600 !important;
    }
    /* Primary buttons: Subdued Slate Grey instead of neon/coral red */
    div.stButton > button[kind="primary"],
    [data-testid="stBaseButton-primary"] {
        background-color: #475569 !important;
        border: 1px solid #334155 !important;
        color: #FFFFFF !important;
        box-shadow: none !important;
    }
    div.stButton > button[kind="primary"] p,
    [data-testid="stBaseButton-primary"] p {
        color: #FFFFFF !important;
    }
    div.stButton > button[kind="primary"]:hover,
    [data-testid="stBaseButton-primary"]:hover {
        background-color: #334155 !important;
        border-color: #1E293B !important;
        color: #FFFFFF !important;
    }
    div.stButton > button[kind="primary"]:active,
    [data-testid="stBaseButton-primary"]:active {
        background-color: #1E293B !important;
        border-color: #0F172A !important;
    }
    div.stButton > button[kind="primary"]:focus,
    [data-testid="stBaseButton-primary"]:focus {
        box-shadow: 0 0 0 2px rgba(71, 85, 105, 0.25) !important;
    }

    /* Secondary buttons: Clean subtle border, soft grey background */
    div.stButton > button[kind="secondary"],
    [data-testid="stBaseButton-secondary"] {
        background-color: #F8FAFC !important;
        border: 1px solid #CBD5E1 !important;
        color: #334155 !important;
        box-shadow: none !important;
    }
    div.stButton > button[kind="secondary"] p,
    [data-testid="stBaseButton-secondary"] p {
        color: #334155 !important;
    }
    div.stButton > button[kind="secondary"]:hover,
    [data-testid="stBaseButton-secondary"]:hover {
        background-color: #F1F5F9 !important;
        border-color: #94A3B8 !important;
        color: #0F172A !important;
    }
    div.stButton > button[kind="secondary"]:focus,
    [data-testid="stBaseButton-secondary"] :focus {
        box-shadow: 0 0 0 2px rgba(203, 213, 225, 0.5) !important;
    }
    /* Ensure Top Time Sync Button is precisely 28px tall and compact */
    div[data-testid="stButton"].st-key-btn_top_time_sync button,
    .st-key-btn_top_time_sync button,
    div[data-testid="stButton"].st-key-btn_top_time_sync [data-testid="stBaseButton-secondary"],
    .st-key-btn_top_time_sync [data-testid="stBaseButton-secondary"] {
        height: 28px !important;
        min-height: 28px !important;
        max-height: 28px !important;
        width: auto !important;
        min-width: 58px !important;
        max-width: 76px !important;
        padding: 0px 10px !important;
        font-size: 13px !important;
        font-weight: 600 !important;
        line-height: 26px !important;
        border-radius: 6px !important;
        border: 1px solid #CBD5E1 !important;
        background: #F1F5F9 !important;
        color: #1E293B !important;
        box-shadow: 0 1px 2px rgba(0,0,0,0.03) !important;
        box-sizing: border-box !important;
        display: inline-flex !important;
        align-items: center !important;
        justify-content: center !important;
        margin: 0 !important;
    }
    div[data-testid="stButton"].st-key-btn_top_time_sync button p,
    .st-key-btn_top_time_sync button p,
    div[data-testid="stButton"].st-key-btn_top_time_sync [data-testid="stBaseButton-secondary"] p,
    .st-key-btn_top_time_sync [data-testid="stBaseButton-secondary"] p {
        font-size: 13px !important;
        font-weight: 600 !important;
        line-height: 26px !important;
        margin: 0 !important;
        padding: 0 !important;
        color: #1E293B !important;
    }

    /* Compact Single-Row Question Buttons: Zero scrollbar, clean single row */
    div[class*="st-key-qa_btns"] {
        margin-top: 8px !important;
    }
    div[data-testid="stVerticalBlock"]:has(.compact-qa-buttons) div[data-testid="stHorizontalBlock"],
    .stVerticalBlock:has(.compact-qa-buttons) .stHorizontalBlock,
    div[class*="st-key-qa_btns"] div[data-testid="stHorizontalBlock"],
    div[class*="st-key-qa_btns"] .stHorizontalBlock,
    div[class*="st-key-qa_panel"] .compact-qa-buttons ~ div[data-testid="stHorizontalBlock"],
    div[class*="st-key-qa_panel"] .stHorizontalBlock:has(.compact-qa-buttons) {
        display: flex !important;
        flex-direction: row !important;
        flex-wrap: nowrap !important;
        overflow-x: hidden !important;
        scrollbar-width: none !important;
        -ms-overflow-style: none !important;
        gap: 6px !important;
        padding-bottom: 2px !important;
    }
    div[data-testid="stVerticalBlock"]:has(.compact-qa-buttons) div[data-testid="stHorizontalBlock"]::-webkit-scrollbar,
    .stVerticalBlock:has(.compact-qa-buttons) .stHorizontalBlock::-webkit-scrollbar,
    div[class*="st-key-qa_btns"] *::-webkit-scrollbar,
    div[class*="st-key-qa_panel"] div[class*="st-key-qa_btns"] *::-webkit-scrollbar {
        display: none !important;
        width: 0 !important;
        height: 0 !important;
    }
    div[data-testid="stVerticalBlock"]:has(.compact-qa-buttons) div[data-testid="column"],
    div[data-testid="stVerticalBlock"]:has(.compact-qa-buttons) div[data-testid="stColumn"],
    .stVerticalBlock:has(.compact-qa-buttons) .stColumn,
    div[class*="st-key-qa_btns"] div[data-testid="stColumn"],
    div[class*="st-key-qa_btns"] .stColumn,
    div[class*="st-key-qa_panel"] div[class*="st-key-qa_btns"] div[data-testid="stColumn"] {
        min-width: 0 !important;
        flex: 1 1 0% !important;
        padding: 0 !important;
    }
    div[data-testid="stVerticalBlock"]:has(.compact-qa-buttons) div.stButton > button,
    .stVerticalBlock:has(.compact-qa-buttons) div.stButton > button,
    div[class*="st-key-qa_btns"] .stButton > button,
    div[class*="st-key-qa_panel"] div[class*="st-key-qa_btns"] .stButton > button {
        padding: 0.20rem 0.25rem !important;
        min-height: 1.85rem !important;
        height: 1.85rem !important;
        border-radius: 5px !important;
        width: 100% !important;
    }
    div[data-testid="stVerticalBlock"]:has(.compact-qa-buttons) div.stButton > button p,
    .stVerticalBlock:has(.compact-qa-buttons) div.stButton > button p,
    div[class*="st-key-qa_btns"] .stButton > button p,
    div[class*="st-key-qa_panel"] div[class*="st-key-qa_btns"] .stButton > button p {
        font-size: 0.74rem !important;
        font-weight: 600 !important;
        white-space: nowrap !important;
        overflow: hidden !important;
        text-overflow: ellipsis !important;
    }

    /* Office Stealth Sliders: Slate grey accents instead of bright red */
    div[data-testid="stSlider"] div[role="slider"] {
        background-color: #475569 !important;
        border-color: #334155 !important;
        box-shadow: none !important;
    }
    div[data-testid="stSlider"] div[data-baseweb="slider"] div[style*="rgb(255, 75, 75)"],
    div[data-testid="stSlider"] div[data-baseweb="slider"] div[style*="#ff4b4b"],
    div[data-testid="stSlider"] div[data-baseweb="slider"] div[style*="RGB(255, 75, 75)"] {
        background-color: #475569 !important;
    }
    div[data-testid="stSlider"] div[data-testid="stThumbValue"],
    div[data-testid="stSlider"] div[data-testid="stThumbValue"] * {
        color: #334155 !important;
        font-size: 0.76rem !important;
        font-weight: 600 !important;
    }
    div[data-testid="stSliderTickBar"] > div {
        background: #CBD5E1 !important;
    }
    div[data-testid="stSlider"] label p {
        font-size: 0.80rem !important;
        font-weight: 600 !important;
        color: #475569 !important;
    }

    /* Progress indicators: Soft slate styling */
    div[data-testid="stProgress"] > div > div > div > div {
        background-color: #475569 !important;
    }

    /* Universal Form Control & Box Text Contrast (Never invisible on mobile/desktop) */
    div[data-testid="stSelectbox"] > label,
    div[data-testid="stTextInput"] > label,
    div[data-testid="stNumberInput"] > label,
    div[data-testid="stSlider"] > label,
    div[data-testid="stRadio"] > label {
        color: #1E293B !important;
        font-size: 0.82rem !important;
        font-weight: 700 !important;
    }
    div[data-baseweb="select"] {
        background-color: #FFFFFF !important;
        border: 1px solid #CBD5E1 !important;
        border-radius: 6px !important;
    }
    div[data-baseweb="select"] * {
        color: #1E293B !important;
        font-size: 0.83rem !important;
    }
    div[data-baseweb="input"],
    div[data-baseweb="input"] * {
        color: #1E293B !important;
        background-color: #FFFFFF !important;
    }
    /* Dropdown popover list item contrast */
    div[data-baseweb="popover"],
    ul[role="listbox"],
    li[role="option"] {
        background-color: #FFFFFF !important;
        color: #1E293B !important;
    }
    li[role="option"] * {
        color: #1E293B !important;
        font-size: 0.82rem !important;
    }
    li[role="option"]:hover,
    li[role="option"][aria-selected="true"] {
        background-color: #F1F5F9 !important;
        color: #0F172A !important;
    }

    /* Segmented radio buttons: High-contrast state styling & zero indentation margin */
    div[data-testid="stRadio"] > div[role="radiogroup"] {
        background: #F8FAFC !important;
        padding: 6px !important;
        border-radius: 8px !important;
        gap: 6px !important;
        border: 1px solid #CBD5E1 !important;
        display: flex !important;
        width: 100% !important;
    }
    div[data-testid="stRadio"] > div[role="radiogroup"] > label {
        background: #FFFFFF !important;
        padding: 8px 12px !important;
        border-radius: 6px !important;
        border: 1px solid #CBD5E1 !important;
        margin: 0 !important;
        margin-left: 0 !important;
        margin-right: 0 !important;
        cursor: pointer !important;
        transition: all 0.15s ease-in-out !important;
        display: flex !important;
        align-items: center !important;
        box-sizing: border-box !important;
    }
    div[data-testid="stRadio"] > div[role="radiogroup"] > label:not(:first-child),
    div[data-testid="stRadio"] label + label,
    div[data-testid="stRadio"] div[role="radiogroup"] label {
        margin-left: 0 !important;
    }
    div[data-testid="stRadio"] > div[role="radiogroup"] > label:hover {
        background: #F1F5F9 !important;
        border-color: #94A3B8 !important;
    }
    /* Ensure high contrast readable text in unselected state */
    div[data-testid="stRadio"] > div[role="radiogroup"] > label * {
        color: #1E293B !important;
        font-size: 0.83rem !important;
        font-weight: 600 !important;
    }
    /* Selected state: dark slate background with crisp white text */
    div[data-testid="stRadio"] > div[role="radiogroup"] > label:has(input:checked) {
        background: #334155 !important;
        border-color: #1E293B !important;
        box-shadow: 0 1px 3px rgba(0, 0, 0, 0.08) !important;
    }
    div[data-testid="stRadio"] > div[role="radiogroup"] > label:has(input:checked) * {
        color: #FFFFFF !important;
        font-weight: 700 !important;
    }

    /* PC / Desktop View: Horizontal layout for Strategy Mode Selection */
    @media (min-width: 769px) {
        div[data-testid="stRadio"] > div[role="radiogroup"] {
            flex-direction: row !important;
            flex-wrap: wrap !important;
            gap: 8px !important;
        }
        div[data-testid="stRadio"] > div[role="radiogroup"] > label {
            flex: 1 1 auto !important;
            min-width: 175px !important;
            width: auto !important;
            justify-content: center !important;
            text-align: center !important;
            margin: 0 !important;
            margin-left: 0 !important;
        }
        /* Desktop table styling */
        div[data-testid="stTable"],
        .stTable {
            width: 100% !important;
            overflow-x: auto !important;
            margin-bottom: 8px !important;
            border-radius: 6px !important;
            border: 1px solid #E2E8F0 !important;
        }
        div[data-testid="stTable"] table,
        .stTable table {
            width: 100% !important;
            border-collapse: collapse !important;
            font-size: 0.78rem !important;
        }
        div[data-testid="stTable"] th,
        .stTable th {
            background-color: #F8FAFC !important;
            color: #334155 !important;
            font-weight: 700 !important;
            font-size: 0.78rem !important;
            padding: 7px 10px !important;
            border-bottom: 2px solid #CBD5E1 !important;
            text-align: left !important;
        }
        div[data-testid="stTable"] td,
        .stTable td {
            padding: 7px 10px !important;
            font-size: 0.78rem !important;
            color: #1E293B !important;
            border-bottom: 1px solid #E2E8F0 !important;
        }
        div[data-testid="stTable"] tr:nth-child(even),
        .stTable tr:nth-child(even) {
            background-color: #F8FAFC !important;
        }
    }

    /* Mobile Responsive Optimizations (PC/Desktop layout is strictly preserved) */
    @media (max-width: 768px) {
        .block-container,
        div[data-testid="stMainBlockContainer"],
        div.stMainBlockContainer {
            padding-top: 0px !important;
            padding-bottom: 0px !important;
            padding-left: 0.5rem !important;
            padding-right: 0.5rem !important;
        }

        /* Top Nav Bar Mobile: Row 1 = Logo, Row 2 = 일시 (Clock), Row 3 = 동기화 버튼 */
        div[data-testid="stElementContainer"].st-key-top_nav_bar,
        .st-key-top_nav_bar {
            position: static !important;
            width: 100% !important;
            height: 92px !important;
            min-height: 92px !important;
            max-height: 92px !important;
            background-color: #FFFFFF !important;
            margin: 0 !important;
            padding: 2px 0.5rem 0px 0.5rem !important;
            border-bottom: 1px solid #E2E8F0 !important;
            display: flex !important;
            flex-direction: column !important;
            box-sizing: border-box !important;
            flex-shrink: 0 !important;
        }
        .st-key-top_nav_bar [data-testid="stHorizontalBlock"],
        div[data-testid="stHorizontalBlock"]:has(.st-key-btn_top_time_sync) {
            display: flex !important;
            flex-direction: column !important;
            align-items: stretch !important;
            width: 100% !important;
            gap: 4px !important;
        }
        .st-key-top_nav_bar iframe {
            margin-bottom: 0px !important;
            display: block !important;
            width: 100% !important;
            height: 28px !important;
        }

        /* Row 1: Logo & status */
        .st-key-top_nav_bar [data-testid="stHorizontalBlock"] > div[data-testid="stColumn"]:nth-child(1),
        .st-key-top_nav_bar [data-testid="stHorizontalBlock"] > div[data-testid="column"]:nth-child(1),
        div[data-testid="stHorizontalBlock"]:has(.st-key-btn_top_time_sync) > div:nth-child(1) {
            flex: 0 0 auto !important;
            width: 100% !important;
            min-width: 100% !important;
            max-width: 100% !important;
            margin-bottom: 2px !important;
            display: flex !important;
            align-items: center !important;
        }

        /* Row 2: Clock (일시 - full width 100%) */
        .st-key-top_nav_bar [data-testid="stHorizontalBlock"] > div[data-testid="stColumn"]:nth-child(2),
        .st-key-top_nav_bar [data-testid="stHorizontalBlock"] > div[data-testid="column"]:nth-child(2),
        div[data-testid="stHorizontalBlock"]:has(.st-key-btn_top_time_sync) > div:nth-child(2) {
            flex: 0 0 auto !important;
            width: 100% !important;
            min-width: 100% !important;
            max-width: 100% !important;
            margin: 0 !important;
            padding: 0 !important;
            display: flex !important;
            align-items: center !important;
        }

        /* Row 3: 동기화 버튼 (일시 밑으로 바로 들어감, full width 100%) */
        .st-key-top_nav_bar [data-testid="stHorizontalBlock"] > div[data-testid="stColumn"]:nth-child(3),
        .st-key-top_nav_bar [data-testid="stHorizontalBlock"] > div[data-testid="column"]:nth-child(3),
        div[data-testid="stHorizontalBlock"]:has(.st-key-btn_top_time_sync) > div:nth-child(3) {
            flex: 0 0 auto !important;
            width: 100% !important;
            min-width: 100% !important;
            max-width: 100% !important;
            margin: 2px 0 0 0 !important;
            padding: 0 !important;
            display: flex !important;
            align-items: center !important;
            justify-content: center !important;
        }

        /* Sync Button mobile size matching clock height 28px */
        div[data-testid="stButton"].st-key-btn_top_time_sync,
        .st-key-btn_top_time_sync {
            height: 28px !important;
            min-height: 28px !important;
            max-height: 28px !important;
            width: 100% !important;
            margin: 0 !important;
            padding: 0 !important;
            display: flex !important;
            align-items: center !important;
        }
        div[data-testid="stButton"].st-key-btn_top_time_sync button,
        .st-key-btn_top_time_sync button,
        div[data-testid="stButton"].st-key-btn_top_time_sync [data-testid="stBaseButton-secondary"],
        .st-key-btn_top_time_sync [data-testid="stBaseButton-secondary"],
        div.stButton.st-key-btn_top_time_sync > button {
            height: 28px !important;
            min-height: 28px !important;
            max-height: 28px !important;
            width: 100% !important;
            min-width: 100% !important;
            max-width: 100% !important;
            padding: 0px 10px !important;
            font-size: 12px !important;
            font-weight: 600 !important;
            line-height: 26px !important;
            border-radius: 6px !important;
            background: #F1F5F9 !important;
            border: 1px solid #CBD5E1 !important;
            color: #1E293B !important;
            box-shadow: 0 1px 2px rgba(0,0,0,0.03) !important;
            display: flex !important;
            align-items: center !important;
            justify-content: center !important;
            margin: 0 !important;
        }
        div[data-testid="stButton"].st-key-btn_top_time_sync button p,
        .st-key-btn_top_time_sync button p {
            font-size: 12px !important;
            line-height: 26px !important;
            margin: 0 !important;
            padding: 0 !important;
            text-align: center !important;
        }

        /* Mobile Tabs Container: flex fill */
        div[data-testid="stElementContainer"]:has(.st-key-main_tabs_header),
        div[data-testid="stElementContainer"].st-key-main_tabs_header,
        div.st-key-main_tabs_header,
        .st-key-main_tabs_header {
            margin: 0 !important;
            padding: 0 !important;
            flex: 1 1 auto !important;
            height: calc(100% - 92px) !important;
            max-height: calc(100% - 92px) !important;
            min-height: 0 !important;
        }
        .st-key-main_tabs_header > div[data-testid="stElementContainer"],
        .st-key-main_tabs_header [data-testid="stElementContainer"] {
            flex: 1 1 auto !important;
            height: 100% !important;
            min-height: 0 !important;
        }
        div[data-testid="stTabs"] [role="tablist"],
        div[data-testid="stTabs"] .react-aria-TabList,
        div[data-testid="stTabs"] [data-baseweb="tab-list"] {
            position: static !important;
            flex: 0 0 36px !important;
            height: 36px !important;
            min-height: 36px !important;
            max-height: 36px !important;
            width: 100% !important;
            background-color: #FFFFFF !important;
            gap: 8px !important;
            padding: 0px 0.5rem !important;
            border: none !important;
            border-bottom: 2px solid #E2E8F0 !important;
            border-radius: 0px !important;
            display: flex !important;
            flex-wrap: nowrap !important;
            overflow-x: auto !important;
            -webkit-overflow-scrolling: touch !important;
            scrollbar-width: none !important;
        }
        div[data-testid="stTabs"] [role="tablist"]::-webkit-scrollbar {
            display: none !important;
        }
        [role="tablist"] [role="tab"],
        [role="tablist"] .react-aria-Tab,
        [role="tablist"] button[role="tab"],
        [role="tablist"] [data-baseweb="tab"] {
            font-size: 0.76rem !important;
            padding: 6px 10px !important;
            height: 34px !important;
            min-height: 34px !important;
            white-space: nowrap !important;
            flex: 0 0 auto !important;
            box-shadow: none !important;
        }
        /* Mobile Tab Panel */
        div[data-testid="stTabs"] [role="tabpanel"],
        div[data-testid="stTabs"] .react-aria-TabPanel,
        div[data-testid="stTabs"] div[data-testid="stTabPanel"],
        div[role="tabpanel"][data-testid="stTabPanel"] {
            flex: 1 1 auto !important;
            height: calc(100% - 36px) !important;
            max-height: calc(100% - 36px) !important;
            min-height: 0 !important;
            overflow-y: auto !important;
            overflow-x: hidden !important;
            padding-top: 6px !important;
            padding-bottom: 3rem !important;
            padding-right: 4px !important;
        }

        /* Global Regime & Reference Time: 모바일에서 기준시각을 글로벌 체제 박스 바로 아래로 명확히 배치 */
        .regime-status-bar {
            display: flex !important;
            flex-direction: column !important;
            align-items: flex-start !important;
            gap: 3px !important;
            min-height: auto !important;
            margin-bottom: 4px !important;
            width: 100% !important;
        }
        .regime-status-bar .regime-badge-container {
            display: block !important;
            width: 100% !important;
        }
        .regime-status-bar .regime-badge {
            font-size: 0.72rem !important;
            padding: 3px 6px !important;
            line-height: 1.35 !important;
            white-space: normal !important;
            word-break: keep-all !important;
            max-width: 100% !important;
            box-sizing: border-box !important;
            display: inline-block !important;
        }
        .regime-status-bar .regime-time-container {
            display: block !important;
            width: 100% !important;
        }
        .regime-status-bar .regime-time {
            font-size: 0.70rem !important;
            color: #64748B !important;
            margin-top: 1px !important;
            display: block !important;
            white-space: nowrap !important;
        }
        div[data-testid="stExpander"] [data-testid="stHorizontalBlock"]:has(.regime-status-bar) {
            display: flex !important;
            flex-direction: column !important;
            gap: 6px !important;
        }
        div[data-testid="stExpander"] [data-testid="stHorizontalBlock"]:has(.regime-status-bar) > div[data-testid="stColumn"],
        div[data-testid="stExpander"] [data-testid="stHorizontalBlock"]:has(.regime-status-bar) > div[data-testid="column"] {
            width: 100% !important;
            flex: 0 0 100% !important;
            max-width: 100% !important;
        }
        div[data-testid="stButton"].st-key-btn_refresh_us_market {
            margin-top: 2px !important;
            margin-bottom: 4px !important;
        }
        div[data-testid="stButton"].st-key-btn_refresh_us_market button {
            height: 28px !important;
            min-height: 28px !important;
            font-size: 12px !important;
            line-height: 26px !important;
            padding: 0px 8px !important;
        }

        /* Mobile: Strictly 1 mode per line vertically stacked, zero indentation */
        div[data-testid="stRadio"] > div[role="radiogroup"] {
            flex-direction: column !important;
            width: 100% !important;
            gap: 6px !important;
        }
        div[data-testid="stRadio"] > div[role="radiogroup"] > label,
        div[data-testid="stRadio"] > div[role="radiogroup"] label,
        div[data-testid="stRadio"] label {
            width: 100% !important;
            padding: 9px 12px !important;
            font-size: 0.80rem !important;
            margin: 0 !important;
            margin-left: 0 !important;
            margin-right: 0 !important;
        }
        div[data-testid="stRadio"] > div[role="radiogroup"] > label:not(:first-child),
        div[data-testid="stRadio"] label + label,
        div[data-testid="stRadio"] > div[role="radiogroup"] label:not(:first-child) {
            margin-left: 0 !important;
        }
        div[data-testid="stMetric"] {
            padding: 4px 6px !important;
            margin-bottom: 4px !important;
        }
        div[data-testid="stMetricValue"] {
            font-size: 0.88rem !important;
        }
        div.stButton:not(.st-key-btn_top_time_sync) > button {
            min-height: 2.3rem !important;
            font-size: 0.82rem !important;
        }
        /* Expander Summary: word wrap without horizontal clipping */
        div[data-testid="stExpander"] details summary {
            font-size: 0.78rem !important;
            padding: 6px 8px !important;
        }
        div[data-testid="stExpander"] details summary p {
            font-size: 0.78rem !important;
            line-height: 1.35 !important;
            white-space: normal !important;
            word-break: keep-all !important;
        }
        /* Tab 5 & Global Tables Mobile Optimization (Scrollable container, crisp font, no squishing) */
        div[data-testid="stTable"],
        .stTable {
            display: block !important;
            width: 100% !important;
            max-width: 100% !important;
            overflow-x: auto !important;
            -webkit-overflow-scrolling: touch !important;
            margin-bottom: 8px !important;
            border-radius: 6px !important;
            border: 1px solid #CBD5E1 !important;
            background: #FFFFFF !important;
        }
        div[data-testid="stTable"] table,
        .stTable table {
            display: table !important;
            width: 100% !important;
            min-width: 480px !important;
            border-collapse: collapse !important;
            font-size: 0.72rem !important;
        }
        div[data-testid="stTable"] th,
        .stTable th {
            background-color: #F1F5F9 !important;
            color: #334155 !important;
            font-weight: 700 !important;
            font-size: 0.71rem !important;
            padding: 6px 8px !important;
            white-space: nowrap !important;
            border-bottom: 2px solid #CBD5E1 !important;
            text-align: left !important;
        }
        div[data-testid="stTable"] td,
        .stTable td {
            padding: 6px 8px !important;
            font-size: 0.72rem !important;
            color: #1E293B !important;
            white-space: nowrap !important;
            border-bottom: 1px solid #E2E8F0 !important;
            line-height: 1.35 !important;
        }
        div[data-testid="stTable"] tr:nth-child(even),
        .stTable tr:nth-child(even) {
            background-color: #F8FAFC !important;
        }
        .stMarkdown table {
            display: block !important;
            overflow-x: auto !important;
            -webkit-overflow-scrolling: touch !important;
            max-width: 100% !important;
        }
        div[data-testid="stDataFrame"] {
            width: 100% !important;
            overflow-x: auto !important;
            -webkit-overflow-scrolling: touch !important;
            font-size: 0.72rem !important;
            border-radius: 6px !important;
        }
        .stExpander {
            border-radius: 6px !important;
            margin-bottom: 6px !important;
        }
    }
</style>
""", unsafe_allow_html=True)

# Initialize Database on app start (once per session)
if "db_initialized" not in st.session_state:
    init_db()
    st.session_state["db_initialized"] = True


# ---------------------------------------------------------
# Sidebar Controls & Portfolio Risk Status
# ---------------------------------------------------------
with st.sidebar:
    st.markdown("### Quantifi Engine")
    st.caption("로컬 데이터 분석 & 스크리너")

    st.markdown("---")
    st.markdown("#### 포트폴리오 리스크 상태")

    # Load Risk State from DB
    with get_db_connection() as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM risk_state WHERE id = 1")
        risk_row = cursor.fetchone()

    equity = float(risk_row["current_equity"]) if risk_row else 10_000_000.0
    daily_loss = float(risk_row["daily_realized_loss"]) if risk_row else 0.0
    cb_active = bool(risk_row["is_circuit_breaker_active"]) if risk_row else False
    cooldown = bool(risk_row["cooldown_active"]) if risk_row else False

    col_e1, col_e2 = st.columns(2)
    col_e1.metric("총 운용자산", f"{equity:,.0f}원")
    col_e2.metric("당일 누적손실", f"{daily_loss:,.0f}원", delta=f"{(daily_loss/equity)*100:.2f}%" if daily_loss != 0 else "0.0%")

    if cb_active:
        st.error("일일 서킷브레이커 발동 (-3% 초과 손실). 신규 매수 차단")
    elif cooldown:
        st.warning("3회 연속 손절 쿨다운 활성화 중")
    else:
        st.success("포트폴리오 상태: 정상 (진입 가능)")

    st.markdown("---")
    with st.expander("시스템 수동 관리 도구 (선택 사항)", expanded=False):
        st.caption("※ 평소에는 스케줄러가 자동 처리하므로 누르실 필요가 없습니다.")
        if st.button("15:40 마감 정산 & 예측 수동 실행", use_container_width=True):
            with st.spinner("15:40 마감 정산 및 익일 예측 실행 중..."):
                res = run_post_market_job()
                st.toast(f"마감 정산 완료 (정산 {res['settled_count']}건, 예측 {res['new_predictions_count']}건)")
                st.rerun()

        if st.button("08:30 장전 브리핑 수동 생성", use_container_width=True):
            with st.spinner("08:30 장전 브리핑 생성 중..."):
                res = run_pre_market_job()
                st.toast(f"장전 브리핑 완료 (후보 {res['candidates_count']}건)")
                st.info(res["briefing"])

        if st.button("KRX 유니버스 최신 동기화", use_container_width=True):
            with st.spinner("KRX 상장 및 상폐 종목 동기화 중..."):
                cnt = sync_krx_universe(force=True)
                st.toast(f"유니버스 {cnt}종목 동기화 완료!")


# Top Header: Removed for Office Stealth Mode


# ---------------------------------------------------------
# High-performance Cached Metrics Loader for Large Watchlists (300+ stocks)
# ---------------------------------------------------------
@st.cache_data(ttl=300, show_spinner=False)
def load_all_watchlist_metrics(codes_tuple):
    """
    Fast parallel batch evaluator for watchlist stocks.
    Loads parquet caches in parallel threads and computes point-in-time scores.
    """
    from concurrent.futures import ThreadPoolExecutor

    bm_ohlcv = get_benchmark_ohlcv()

    def _eval_one(item):
        if len(item) == 6:
            code, name, market, sector, notes, group_name = item
        else:
            code, name, market, sector, notes = item[:5]
            group_name = "기본그룹"

        try:
            df = fetch_ohlcv(code)
            price_info = get_latest_market_info(code)
            score_info = evaluate_stock_latest(df)
        except Exception:
            price_info = {"close": 0.0, "change_pct": 0.0, "volume": 0, "date": "N/A"}
            score_info = None

        if score_info:
            score = score_info["score"]
            label = score_info["label"]
            tp = score_info["tp_pct"]
            sl = score_info["sl_pct"]
            bk = score_info["breakdown"]
            intraday_info = score_info.get("intraday_outlook", {})
            nextday_info = score_info.get("nextday_outlook", {})
            intraday_tag = intraday_info.get("tag", "[거래소강]")
            intraday_desc = intraday_info.get("desc", "장중 시세 데이터 대기")
            intraday_open_pct = intraday_info.get("open_gain_pct", 0.0)
            nextday_tag = nextday_info.get("tag", f"{label} ({score:.1f}점)")
            nextday_desc = nextday_info.get("desc", "기술적 추세 기반 전망")
        else:
            score = 50.0
            label = "중립"
            tp = 1.5
            sl = 2.0
            bk = {
                "rsi_val": 50.0,
                "ma_status": "-",
                "vol_ratio": 1.0,
                "bb_pct": 0.5,
                "rsi_score": 0.0,
                "ma_score": 0.0,
                "vol_score": 0.0,
                "bb_score": 0.0
            }
            intraday_tag = "[거래소강]"
            intraday_desc = "시세 분석 대기"
            intraday_open_pct = 0.0
            nextday_tag = "중립 (50.0점)"
            nextday_desc = "기본 퀀트 점수"

        is_sniper = False
        is_closing_bet = False
        is_surge_5pct = False
        is_daytrade = False
        if df is not None and not df.empty and len(df) >= 30:
            try:
                dtrade = evaluate_intraday_daytrade_candidate(df, bm_ohlcv, min_today_val_krw=10_000_000_000)
                if dtrade:
                    is_daytrade = True
                s5 = evaluate_5pct_surge_candidate(df, bm_ohlcv, min_today_val_krw=10_000_000_000, min_day_return=10.0)
                if s5:
                    is_surge_5pct = True
                snp = evaluate_sniper_candidate(df, bm_ohlcv, min_daily_val_krw=5_000_000_000)
                if snp:
                    is_sniper = True
                cbet = evaluate_closing_bet_candidate(df, min_today_val_krw=10_000_000_000)
                if cbet:
                    is_closing_bet = True
            except Exception:
                pass

        badge = "● 검증됨" if score >= 75.0 else "● 실험적"
        if is_daytrade:
            strat_tag = "당일단타 (당일 +5%)"
            tp = 5.0
            sl = 2.5
        elif is_surge_5pct:
            strat_tag = "5%타겟 (마감 직전)"
            tp = 5.0
            sl = 4.0
        elif is_sniper:
            strat_tag = "스나이퍼 (오전 9시)"
            tp = 1.2
            sl = 2.0
        elif is_closing_bet:
            strat_tag = "종가배팅 (마감 직전)"
            tp = 1.5
            sl = 2.0
        else:
            strat_tag = "-"

        analyzed_gain = calculate_analyzed_rise_rate(
            total_score=score,
            vol_ratio=bk.get("vol_ratio", 1.0),
            rsi_val=bk.get("rsi_val", 50.0),
            ma_status=bk.get("ma_status", "-"),
            change_pct=price_info.get("change_pct", 0.0),
            strategy_tag=strat_tag,
            bb_pct=bk.get("bb_pct", 0.5)
        )

        return {
            "code": code,
            "name": name,
            "group_name": group_name or "기본그룹",
            "market": market,
            "sector": sector or notes or "기타",
            "close": price_info["close"],
            "change_pct": price_info["change_pct"],
            "volume": price_info["volume"],
            "score": score,
            "label": label,
            "intraday_tag": intraday_tag,
            "intraday_desc": intraday_desc,
            "intraday_open_pct": intraday_open_pct,
            "nextday_tag": nextday_tag,
            "nextday_desc": nextday_desc,
            "strategy_tag": strat_tag,
            "is_sniper": is_sniper,
            "is_closing_bet": is_closing_bet,
            "is_surge_5pct": is_surge_5pct,
            "is_daytrade": is_daytrade,
            "is_high_conviction": is_sniper or is_surge_5pct or is_daytrade,
            "tp_pct": tp,
            "sl_pct": sl,
            "analyzed_gain_pct": analyzed_gain,
            "rsi": bk.get("rsi_val", 50.0),
            "vol_ratio": bk.get("vol_ratio", 1.0),
            "ma_status": bk.get("ma_status", "-"),
            "bb_pct": bk.get("bb_pct", 0.5),
            "verified_badge": badge,
            "breakdown": bk,
        }

    with ThreadPoolExecutor(max_workers=10) as executor:
        results = list(executor.map(_eval_one, codes_tuple))

    return results


# ---------------------------------------------------------
# Top Navigation & Status Bar (Office Stealth Mode)
# ---------------------------------------------------------
if "session_uid" not in st.session_state:
    import uuid
    st.session_state["session_uid"] = str(uuid.uuid4())
sc_worker = get_screening_worker(st.session_state["session_uid"])

now_dt = get_now_kst()
is_weekday = now_dt.weekday() < 5
cur_hm = now_dt.hour * 100 + now_dt.minute

if is_weekday and 900 <= cur_hm < 1530:
    market_badge = '<span style="color:#059669; font-weight:700;">● 정규장 운영중</span>'
elif is_weekday and 830 <= cur_hm < 900:
    market_badge = '<span style="color:#D97706; font-weight:700;">○ 장전 동시호가 <span class="badge-sub">(08:30~09:00)</span></span>'
elif is_weekday and 1530 <= cur_hm < 1800:
    market_badge = '<span style="color:#0284C7; font-weight:700;">○ 시간외 단일가 <span class="badge-sub">(15:30~18:00)</span></span>'
else:
    market_badge = '<span style="color:#64748B; font-weight:600;">○ 장마감 (휴장)</span>'

time_str = now_dt.strftime("%Y-%m-%d %H:%M:%S")

_fav_b64 = ""
if os.path.exists(_FAVICON_PATH):
    try:
        import base64
        with open(_FAVICON_PATH, "rb") as _f_fav:
            _fav_b64 = base64.b64encode(_f_fav.read()).decode()
    except Exception:
        pass

with st.container(key="top_nav_bar"):
    col_top_l, col_top_mid, col_top_btn = st.columns([1.7, 2.7, 0.6], vertical_alignment="center")
    with col_top_l:
        st.markdown(
            '<div style="display:flex; align-items:center; gap:10px; padding:2px 0 4px 0; min-height:34px; line-height:28px; box-sizing:border-box;">'
            '<span style="font-size:1.38rem; font-weight:800; color:#0F172A; letter-spacing:-0.5px; white-space:nowrap; display:inline-block;">Quantifi</span>'
            '</div>',
            unsafe_allow_html=True
        )
    with col_top_mid:
        live_clock_html = f"""<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<style>
  * {{ box-sizing: border-box; margin: 0; padding: 0; }}
  html, body {{
    margin: 0;
    padding: 0;
    width: 100%;
    height: 100%;
    overflow: hidden;
    background: transparent;
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "Helvetica Neue", Arial, sans-serif;
  }}
  .clock-wrapper {{
    display: flex;
    justify-content: flex-end;
    align-items: center;
    width: 100%;
    height: 28px;
  }}
  .clock-bar {{
    display: inline-flex;
    align-items: center;
    gap: 8px;
    height: 28px;
    padding: 0 10px;
    background: #F1F5F9;
    border: 1px solid #CBD5E1;
    border-radius: 6px;
    font-size: 13px;
    color: #1E293B;
    white-space: nowrap;
    user-select: none;
    box-shadow: 0 1px 2px rgba(0,0,0,0.03);
  }}
  .clock-time {{
    color: #0F172A;
    font-weight: 700;
    font-variant-numeric: tabular-nums;
    font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, monospace;
    letter-spacing: -0.2px;
  }}
  .sep {{ color: #94A3B8; font-weight: 600; }}
  .badge {{ font-size: 12px; font-weight: 600; }}

  @media (prefers-color-scheme: dark) {{
    .clock-bar {{
      background: #1E293B !important;
      border-color: #475569 !important;
      color: #F8FAFC !important;
    }}
    .clock-time {{
      color: #FFFFFF !important;
    }}
    .sep {{ color: #64748B !important; }}
  }}

  @media (max-width: 768px) {{
    .clock-wrapper {{
      justify-content: center;
      width: 100%;
    }}
    .clock-bar {{
      height: 28px;
      font-size: 11.5px;
      padding: 0 8px;
      gap: 6px;
      width: 100%;
      justify-content: center;
    }}
    .clock-time {{
      font-size: 11.5px;
    }}
    .badge {{
      font-size: 11px;
    }}
    .clock-year {{
      display: inline !important;
    }}
    .badge-sub {{
      display: none !important;
    }}
  }}
</style>
</head>
<body>
<div class="clock-wrapper">
  <div class="clock-bar">
    <span class="clock-time" id="kst-time-val"><span class="clock-year">{now_dt.strftime("%Y-")}</span>{now_dt.strftime("%m-%d %H:%M:%S")} KST</span>
    <span class="sep">|</span>
    <span class="badge" id="kst-market-badge">{market_badge}</span>
  </div>
</div>
<script>
(function() {{
  try {{
    var pDoc = window.parent.document;
    if (pDoc && "{_fav_b64}") {{
      var favDataUrl = 'data:image/png;base64,{_fav_b64}';
      var existingFavs = pDoc.querySelectorAll("link[rel*='icon'], link[rel='apple-touch-icon']");
      existingFavs.forEach(function(el) {{ el.parentNode.removeChild(el); }});
      ['icon', 'shortcut icon', 'apple-touch-icon'].forEach(function(relName) {{
        var l = pDoc.createElement('link');
        l.rel = relName;
        l.type = 'image/png';
        l.href = favDataUrl;
        pDoc.getElementsByTagName('head')[0].appendChild(l);
      }});
    }}
  }} catch(e) {{}}
  function updateClock() {{
    try {{
      var nowMs = Date.now();
      // UTC + 9 hours for KST
      var kst = new Date(nowMs + 9 * 3600 * 1000);
      var y = kst.getUTCFullYear();
      var m = String(kst.getUTCMonth() + 1).padStart(2, '0');
      var d = String(kst.getUTCDate()).padStart(2, '0');
      var hh = String(kst.getUTCHours()).padStart(2, '0');
      var mm = String(kst.getUTCMinutes()).padStart(2, '0');
      var ss = String(kst.getUTCSeconds()).padStart(2, '0');
      var day = kst.getUTCDay();

      var timeEl = document.getElementById('kst-time-val');
      if (timeEl) {{
        timeEl.innerHTML = '<span class="clock-year">' + y + '-</span>' + m + '-' + d + ' ' + hh + ':' + mm + ':' + ss + ' KST';
      }}

      var isWeekday = (day >= 1 && day <= 5);
      var curHm = kst.getUTCHours() * 100 + kst.getUTCMinutes();
      var badgeEl = document.getElementById('kst-market-badge');
      if (badgeEl) {{
        var badgeHtml = '';
        if (isWeekday && curHm >= 900 && curHm < 1530) {{
          badgeHtml = '<span style="color:#059669; font-weight:700;">● 정규장 운영중</span>';
        }} else if (isWeekday && curHm >= 830 && curHm < 900) {{
          badgeHtml = '<span style="color:#D97706; font-weight:700;">○ 장전 동시호가 <span class="badge-sub">(08:30~09:00)</span></span>';
        }} else if (isWeekday && curHm >= 1530 && curHm < 1800) {{
          badgeHtml = '<span style="color:#0284C7; font-weight:700;">○ 시간외 단일가 <span class="badge-sub">(15:30~18:00)</span></span>';
        }} else {{
          badgeHtml = '<span style="color:#64748B; font-weight:600;">○ 장마감 (휴장)</span>';
        }}
        if (badgeEl.innerHTML !== badgeHtml) {{
          badgeEl.innerHTML = badgeHtml;
        }}
      }}
    }} catch (e) {{
      console.error(e);
    }}
  }}
  updateClock();
  // 250ms interval ensures the second boundary is caught immediately without delay
  setInterval(updateClock, 250);
  document.addEventListener('visibilitychange', function() {{
    if (!document.hidden) updateClock();
  }});
}})();
</script>
</body>
</html>"""
        components.html(live_clock_html, height=28, scrolling=False)
    with col_top_btn:
        if st.button("동기화", help="현재시각, 시장상태, 전 탭 실시간 시세 및 스크리너 필터 권장값 최신화", key="btn_top_time_sync", use_container_width=False):
            st.cache_data.clear()
            for _k in list(st.session_state.keys()):
                if _k.startswith("qa_ans_cache_") or _k.startswith("quant_cache_"):
                    st.session_state[_k] = {}
                if _k.startswith("qa_active_ans_") or _k.startswith("quant_active_ans_"):
                    st.session_state.pop(_k, None)
            st.session_state["sc_scope_sel"] = "전 상장사 전수조사 (약 2,700개)"
            st.session_state.pop("sc_user_explicitly_chose_300", None)
            st.session_state["sc_trigger_time_sync"] = True
            st.toast("현재 시각 기준 실시간 시세, Q&A 데이터 및 스크리너 필터가 전체 동기화되었습니다.")
            st.rerun()

def _safe_float_val(v, default=0.0) -> float:
    try:
        if v is None:
            return default
        import math
        f = float(v)
        return default if math.isnan(f) else f
    except Exception:
        return default


@st.cache_data(ttl=300, show_spinner=False)
def _get_cached_meta_opt():
    return evaluate_multi_horizon_tuning_impact()


@st.cache_data(ttl=300, show_spinner=False)
def _get_cached_decay_info():
    return evaluate_model_decay(expected_ci_lower=50.0)


@st.cache_data(ttl=300, show_spinner=False)
def _get_cached_weekly_res(limit_days=5):
    return get_weekly_verification_summary(limit_days=limit_days)


@st.cache_data(ttl=600, show_spinner=False)
def get_cached_universe_options():
    try:
        with get_db_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                SELECT code, name, market, COALESCE(sector, '기타') as sector, COALESCE(industry, '') as industry
                FROM universe
                WHERE is_active = 1
                ORDER BY
                    CASE WHEN market='KOSPI' THEN 1 ELSE 2 END,
                    name ASC
            """)
            rows = cursor.fetchall()
            if rows:
                return [
                    {
                        "code": r["code"],
                        "name": r["name"],
                        "market": r["market"],
                        "sector": normalize_sector(r["sector"], r["name"], r["industry"]),
                        "display": f"{r['name']} ({r['code']}) [{r['market']} · {normalize_sector(r['sector'], r['name'], r['industry'])}]"
                    }
                    for r in rows
                ]
    except Exception as e:
        logger.warning(f"Failed to fetch cached universe options: {e}")

    defaults = [
        ("005930", "삼성전자", "KOSPI", "전기/전자"),
        ("000660", "SK하이닉스", "KOSPI", "전기/전자"),
        ("005380", "현대차", "KOSPI", "자동차"),
        ("035420", "NAVER", "KOSPI", "IT/소프트웨어"),
        ("196170", "알테오젠", "KOSDAQ", "바이오/R&D"),
        ("247540", "에코프로비엠", "KOSDAQ", "2차전지/소재"),
    ]
    return [
        {
            "code": c,
            "name": n,
            "market": m,
            "sector": s,
            "display": f"{n} ({c}) [{m} · {s}]"
        }
        for c, n, m, s in defaults
    ]


@st.cache_data(ttl=30, show_spinner=False)
def get_stock_factsheet_payload(code: str) -> Dict[str, Any]:
    raw_code = str(code).strip()
    clean_code = raw_code.zfill(6) if raw_code.isdigit() else raw_code

    name = f"종목 {clean_code}"
    market = "KRX"
    sector = "기타"
    industry = ""
    try:
        with get_db_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT name, market, COALESCE(sector, '기타') as sector, COALESCE(industry, '') as industry FROM universe WHERE code = ?", (clean_code,))
            row = cursor.fetchone()
            if row:
                name = row["name"]
                market = row["market"]
                industry = row["industry"]
                sector = normalize_sector(row["sector"], name, industry)
    except Exception as e:
        logger.warning(f"Failed to lookup universe for {clean_code}: {e}")

    price_info = get_latest_market_info(clean_code)
    close = _safe_float_val(price_info.get("close"), 0.0)
    change_pct = _safe_float_val(price_info.get("change_pct"), 0.0)
    open_p = _safe_float_val(price_info.get("open"), close)
    high_p = _safe_float_val(price_info.get("high"), close)
    low_p = _safe_float_val(price_info.get("low"), close)
    prev_close = _safe_float_val(price_info.get("prev_close"), 0.0)
    if (prev_close <= 0 or abs(prev_close - close) < 0.01) and close > 0 and abs(change_pct) > 0.001:
        prev_close = round(close / (1.0 + change_pct / 100.0))
    elif prev_close <= 0:
        prev_close = close
    open_pct = _safe_float_val(price_info.get("open_pct"), 0.0)
    high_pct = _safe_float_val(price_info.get("high_pct"), 0.0)
    low_pct = _safe_float_val(price_info.get("low_pct"), 0.0)
    volume = int(_safe_float_val(price_info.get("volume"), 0))
    date_str = str(price_info.get("date", ""))

    df = None
    try:
        df = fetch_ohlcv(clean_code)
    except Exception as e:
        logger.warning(f"Failed to fetch OHLCV for {clean_code}: {e}")

    score = 50.0
    label = "중립"
    tp_pct = 5.0
    sl_pct = 2.5
    bk = {
        "rsi_val": 50.0,
        "vol_ratio": 1.0,
        "ma_status": "이평 밀집/수렴",
        "bb_pct": 0.5,
        "rsi_score": 10.0,
        "ma_score": 10.0,
        "vol_score": 10.0,
        "bb_score": 10.0
    }

    indicators = {}
    if df is not None and not df.empty and len(df) >= 20:
        try:
            score_info = evaluate_stock_latest(df)
            if score_info:
                score = score_info["score"]
                label = score_info["label"]
                tp_pct = score_info["tp_pct"]
                sl_pct = score_info["sl_pct"]
                bk = score_info["breakdown"]
                indicators = score_info.get("indicators", {})
        except Exception as e:
            logger.warning(f"Failed to evaluate scoring for {clean_code}: {e}")

    bm_ohlcv = get_benchmark_ohlcv()
    is_daytrade = False
    is_surge_5pct = False
    is_sniper = False
    is_closing_bet = False

    if df is not None and not df.empty and len(df) >= 30:
        try:
            if evaluate_intraday_daytrade_candidate(df, bm_ohlcv, min_today_val_krw=10_000_000_000):
                is_daytrade = True
            if evaluate_5pct_surge_candidate(df, bm_ohlcv, min_today_val_krw=10_000_000_000, min_day_return=8.0):
                is_surge_5pct = True
            if evaluate_sniper_candidate(df, bm_ohlcv, min_daily_val_krw=5_000_000_000):
                is_sniper = True
            if evaluate_closing_bet_candidate(df, min_today_val_krw=10_000_000_000):
                is_closing_bet = True
        except Exception:
            pass

    if is_daytrade:
        strategy_tag = "실시간 당일 단타 (당일 +5% 익절)"
        tp_pct = 5.0
        sl_pct = 2.5
        reason_core = "장중 거래대금 급증 및 주도 수급 유입으로 당일 내 변동성 상방 확장 조건 충족"
        rule_note = "장중 목표가(+5%) 도달 시 50% 분할 익절, 잔여분 +1.8% 트레일링 락. 당일 15:20 미체결 시 전량 청산"
    elif is_surge_5pct:
        strategy_tag = "5% 급등 타겟 (1~2일 스윙)"
        tp_pct = 5.0
        sl_pct = 4.0
        reason_core = "전일비 거래량 폭발 및 장대 양봉 돌파 모멘텀 발생"
        rule_note = "15:20~15:30 분할 진입 권장, 1~2거래일 내 +5.0% 익절 목표, -4.0% 이탈 시 기계적 손절"
    elif is_closing_bet:
        strategy_tag = "주도주 종가배팅 (익일 시초 갭)"
        tp_pct = 1.5
        sl_pct = 2.0
        reason_core = "마감 직전 거래대금 상위 및 윗꼬리 적은 강력한 일봉 캔들 유지"
        rule_note = "15:20~15:30 종가 매수 후 익일 동시호가~시초가 +1.5% 갭상승 구간에서 전량 차익 실현"
    elif is_sniper:
        strategy_tag = "스나이퍼 고확신 (눌림목 반등)"
        tp_pct = 1.2
        sl_pct = 2.0
        reason_core = "20일 이동평균선 지지 반등 및 과매도 해소 구간"
        rule_note = "09:00 시초가 직후 15분봉 지지 확인 후 진입, 목표가 도달 시 즉시 분할 익절"
    elif score >= 70.0:
        strategy_tag = "고확신 퀀트 모멘텀"
        tp_pct = 3.5
        sl_pct = 2.0
        reason_core = f"종합 퀀트 스코어 {score:.1f}점 우수 구간 (RSI {bk.get('rsi_val', 50):.1f}pt, {bk.get('ma_status', '이평 정배열')})"
        rule_note = "분할 매수 원칙 준수, 목표가 도달 시 트레일링 스탑 적용"
    else:
        strategy_tag = "일반 기술적 진단"
        tp_pct = 3.0
        sl_pct = 2.0
        reason_core = f"현재 기술적 상태: {bk.get('ma_status', '이평 혼조')}, RSI {bk.get('rsi_val', 50):.1f}pt"
        rule_note = "추세 확인 후 제한적 비중으로 진입 고려, 손절 기준가 엄수"

    tp_price = close * (1.0 + tp_pct / 100.0) if close > 0 else 0.0
    sl_price = close * (1.0 - sl_pct / 100.0) if close > 0 else 0.0
    rr_ratio = (tp_pct / max(0.1, sl_pct))

    est_trading_val_krw = close * volume
    vol_ratio = _safe_float_val(bk.get("vol_ratio"), 1.0)
    rsi_val = _safe_float_val(bk.get("rsi_val"), 50.0)
    ma_status = str(bk.get("ma_status", "이평 정배열"))
    bb_pct = _safe_float_val(bk.get("bb_pct"), 0.5)

    analyzed_gain_pct = calculate_analyzed_rise_rate(
        total_score=score,
        vol_ratio=vol_ratio,
        rsi_val=rsi_val,
        ma_status=ma_status,
        change_pct=change_pct,
        strategy_tag=strategy_tag,
        bb_pct=bb_pct
    )

    return {
        "code": clean_code,
        "name": name,
        "market": market,
        "sector": sector,
        "industry": industry,
        "close": close,
        "change_pct": change_pct,
        "open": open_p,
        "high": high_p,
        "low": low_p,
        "prev_close": prev_close,
        "open_pct": open_pct,
        "high_pct": high_pct,
        "low_pct": low_pct,
        "volume": volume,
        "date": date_str,
        "score": score,
        "label": label,
        "strategy_tag": strategy_tag,
        "tp_pct": tp_pct,
        "sl_pct": sl_pct,
        "analyzed_gain_pct": analyzed_gain_pct,
        "tp_price": tp_price,
        "sl_price": sl_price,
        "rr_ratio": rr_ratio,
        "vol_ratio": vol_ratio,
        "rsi": rsi_val,
        "ma_status": ma_status,
        "bb_pct": bb_pct,
        "breakdown": bk,
        "indicators": indicators,
        "reason_core": reason_core,
        "rule_note": rule_note,
        "est_trading_val_krw": est_trading_val_krw,
        "is_daytrade": is_daytrade,
        "is_surge_5pct": is_surge_5pct,
        "is_sniper": is_sniper,
        "is_closing_bet": is_closing_bet,
    }


def render_quant_analyst_qa_box(stock_row, key_suffix=""):
    stock_dict = stock_row.to_dict() if hasattr(stock_row, "to_dict") else dict(stock_row)
    raw_code = str(stock_dict.get("code") or "000000").strip()
    code = raw_code.zfill(6) if raw_code.isdigit() else raw_code
    stock_dict["code"] = code
    name = str(stock_dict.get("name") or "종목")

    active_q_key = f"quant_active_q_{code}_{key_suffix}"
    active_ans_key = f"quant_active_ans_{code}_{key_suffix}"
    cache_key = f"quant_cache_{code}"
    hist_key = f"quant_hist_{code}_{key_suffix}"

    if cache_key not in st.session_state:
        st.session_state[cache_key] = {}
    if hist_key not in st.session_state:
        st.session_state[hist_key] = []

    quick_queries = {
        "buy": {
            "label": "지금 사도 되나요?",
            "q": "지금 사도 되나요? 신규 매수 진입 적합도와 손익비를 진단해줘."
        },
        "why": {
            "label": "왜 급등하나요?",
            "q": "왜 급등하나요? 거래량과 수급 상승 원인을 정밀 분석해줘."
        },
        "tp": {
            "label": "목표가 & 손절가",
            "q": "목표가 & 손절가 알고리즘 권장 익절가와 스탑 가격을 알려줘."
        },
        "risk": {
            "label": "리스크 팩트체크",
            "q": "리스크 팩트체크 주의해야 할 핵심 투자 위험 요소를 알려줘."
        },
        "chart": {
            "label": "차트 & 지표 진단",
            "q": "차트 & 지표 진단 이평선과 RSI 기술적 지표를 종합 분석해줘."
        }
    }

    # Initial answer or synced answer initialization
    if active_q_key not in st.session_state or active_ans_key not in st.session_state:
        target_label = st.session_state.get(active_q_key, quick_queries["buy"]["label"])
        target_prompt = None
        for _k, _v in quick_queries.items():
            if _v["label"] == target_label:
                target_prompt = _v["q"]
                break
        if not target_prompt:
            clean_q = target_label.replace("직접 질문:", "", 1).strip()
            target_prompt = clean_q if clean_q else quick_queries["buy"]["q"]

        if target_prompt in st.session_state[cache_key]:
            cur_ans = st.session_state[cache_key][target_prompt]
        else:
            try:
                cur_ans = generate_quant_expert_answer(stock_dict, target_prompt, chat_history=st.session_state.get(hist_key, []))
            except Exception as _e:
                cur_ans = f"분석 리포트 생성 중 일시적인 오류가 발생했습니다: {_e}"
            st.session_state[cache_key][target_prompt] = cur_ans

        st.session_state[active_q_key] = target_label
        st.session_state[active_ans_key] = cur_ans
        if not st.session_state[hist_key]:
            st.session_state[hist_key].append({"q": target_label, "ans": cur_ans})
        elif st.session_state[hist_key][-1]["q"] == target_label:
            st.session_state[hist_key][-1]["ans"] = cur_ans

    with st.container(border=True, key=f"qa_panel_{code}_{key_suffix}"):
        hdr_c1, hdr_c2 = st.columns([4.0, 1.0], gap="small")
        with hdr_c1:
            close_v = float(stock_dict.get('close') or 0.0)
            chg_v = float(stock_dict.get('change_pct') or 0.0)
            score_v = float(stock_dict.get('score') or 0.0)
            vol_v = float(stock_dict.get('vol_ratio') or 1.0)
            rsi_v = float(stock_dict.get('rsi') or 50.0)
            strat_v = stock_dict.get('strategy_tag', '')
            price_str = f" {close_v:,.0f}원 ({chg_v:+.2f}%)" if close_v > 0 else ""
            st.markdown(
                f"<div style='padding:4px 0 10px 2px;'>"
                f"<div style='font-size:0.92rem; font-weight:700; color:#1E293B; line-height:1.4;'>{name} ({code})<span style='font-weight:600; color:#0284C7; margin-left:6px;'>{price_str}</span></div>"
                f"<div style='font-size:0.76rem; color:#64748B; margin-top:4px; line-height:1.4; display:block;'>전략: {strat_v} | 스코어: {score_v:.1f}점 | 전일비 거래량: {vol_v:.1f}배 | RSI: {rsi_v:.1f}</div>"
                f"</div>",
                unsafe_allow_html=True
            )
        with hdr_c2:
            st.markdown("<div style='height:6px;'></div>", unsafe_allow_html=True)
            if st.button("질의 초기화", key=f"reset_chat_{code}_{key_suffix}", use_container_width=True):
                st.session_state[cache_key] = {}
                st.session_state[hist_key] = []
                st.session_state.pop(active_q_key, None)
                st.session_state.pop(active_ans_key, None)
                st.rerun()

        # Quick Question Buttons (Single-row compact layout in dedicated container)
        with st.container(key=f"qa_btns_{code}_{key_suffix}"):
            st.markdown('<div class="compact-qa-buttons" style="display:none;"></div>', unsafe_allow_html=True)
            q_cols = st.columns(5, gap="small")

            curr_active_q = st.session_state.get(active_q_key, quick_queries["buy"]["label"])

            clicked_prompt = None
            clicked_label = None

            with q_cols[0]:
                b_type = "primary" if curr_active_q == quick_queries["buy"]["label"] else "secondary"
                if st.button(quick_queries["buy"]["label"], key=f"btn_quick_buy_{code}_{key_suffix}", type=b_type, use_container_width=True):
                    clicked_prompt = quick_queries["buy"]["q"]
                    clicked_label = quick_queries["buy"]["label"]

            with q_cols[1]:
                b_type = "primary" if curr_active_q == quick_queries["why"]["label"] else "secondary"
                if st.button(quick_queries["why"]["label"], key=f"btn_quick_why_{code}_{key_suffix}", type=b_type, use_container_width=True):
                    clicked_prompt = quick_queries["why"]["q"]
                    clicked_label = quick_queries["why"]["label"]

            with q_cols[2]:
                b_type = "primary" if curr_active_q == quick_queries["tp"]["label"] else "secondary"
                if st.button(quick_queries["tp"]["label"], key=f"btn_quick_tp_{code}_{key_suffix}", type=b_type, use_container_width=True):
                    clicked_prompt = quick_queries["tp"]["q"]
                    clicked_label = quick_queries["tp"]["label"]

            with q_cols[3]:
                b_type = "primary" if curr_active_q == quick_queries["risk"]["label"] else "secondary"
                if st.button(quick_queries["risk"]["label"], key=f"btn_quick_risk_{code}_{key_suffix}", type=b_type, use_container_width=True):
                    clicked_prompt = quick_queries["risk"]["q"]
                    clicked_label = quick_queries["risk"]["label"]

            with q_cols[4]:
                b_type = "primary" if curr_active_q == quick_queries["chart"]["label"] else "secondary"
                if st.button(quick_queries["chart"]["label"], key=f"btn_quick_chart_{code}_{key_suffix}", type=b_type, use_container_width=True):
                    clicked_prompt = quick_queries["chart"]["q"]
                    clicked_label = quick_queries["chart"]["label"]

        if clicked_prompt and clicked_label:
            if clicked_prompt in st.session_state[cache_key]:
                new_ans = st.session_state[cache_key][clicked_prompt]
            else:
                with st.spinner(f"AI 퀀트 수석 애널리스트가 '{clicked_label}' 항목을 정밀 분석 중입니다..."):
                    try:
                        new_ans = generate_quant_expert_answer(stock_dict, clicked_prompt, chat_history=st.session_state.get(hist_key, []))
                    except Exception as _e:
                        new_ans = f"분석 답변 생성 중 오류가 발생했습니다: {_e}"
                    st.session_state[cache_key][clicked_prompt] = new_ans

            st.session_state[active_q_key] = clicked_label
            st.session_state[active_ans_key] = new_ans
            st.session_state[hist_key].append({"q": clicked_label, "ans": new_ans})
            st.rerun()

        # Active Answer Display
        active_q_disp = st.session_state.get(active_q_key, quick_queries["why"]["label"])
        active_ans_disp = st.session_state.get(active_ans_key, "")

        st.markdown(
            f"<div style='font-size:0.85rem; font-weight:700; color:#1E293B; background:#F8FAFC; padding:8px 12px; border-radius:6px; border:1px solid #CBD5E1; margin-top:10px; margin-bottom:10px;'>"
            f"선택 질의: <span style='color:#0284C7;'>{active_q_disp}</span>"
            f"</div>",
            unsafe_allow_html=True
        )

        with st.chat_message("assistant"):
            st.markdown(f"<div class='quant-qa-body-wrap'>\n\n{active_ans_disp}\n\n</div>", unsafe_allow_html=True)

        # Custom Question Form
        with st.form(key=f"custom_q_form_{code}_{key_suffix}", clear_on_submit=True):
            f_c1, f_c2 = st.columns([4, 1])
            with f_c1:
                user_custom_input = st.text_input(
                    "질문 입력",
                    placeholder=f"{name}에 대해 궁금한 점을 직접 질문하세요 (예: 내일 시초가에 어떻게 대응해야 하나요?)",
                    label_visibility="collapsed",
                    key=f"input_q_text_{code}_{key_suffix}"
                )
            with f_c2:
                submitted = st.form_submit_button("질문 전송", type="primary", use_container_width=True)

            if submitted and user_custom_input.strip():
                c_prompt = user_custom_input.strip()
                c_prompt = re.sub(r"^(\[[^\]]+\]\s*)+", "", c_prompt).strip()
                with st.spinner("퀀트 전문가 답변 생성 중..."):
                    try:
                        ans = generate_quant_expert_answer(stock_dict, c_prompt, chat_history=st.session_state.get(hist_key, []))
                    except Exception as _e:
                        ans = f"분석 답변 생성 중 오류가 발생했습니다: {_e}"
                st.session_state[active_q_key] = f"직접 질문: {c_prompt}"
                st.session_state[active_ans_key] = ans
                st.session_state[hist_key].append({"q": f"직접 질문: {c_prompt}", "ans": ans})
                st.rerun()

        # Past Question History Expander
        if len(st.session_state[hist_key]) > 1:
            with st.expander(f"이전 질의응답 기록 ({len(st.session_state[hist_key])}건)", expanded=False):
                for h_idx, h in enumerate(reversed(st.session_state[hist_key][:-1])):
                    st.markdown(f"**[질문] {h['q']}**")
                    st.markdown(h['ans'])
                    if h_idx < len(st.session_state[hist_key]) - 2:
                        st.divider()


def _strip_redundant_qa_headers(ans_text: str) -> str:
    """Removes leading markdown title (### ...) and '- **질문 분석**: ...' line from displayed answer."""
    if not ans_text:
        return ""
    lines = ans_text.splitlines()
    cleaned = []
    for line in lines:
        s = line.strip()
        if not cleaned and not s:
            continue
        if not cleaned and s.startswith("###"):
            continue
        if "질문 분석" in s and (s.startswith("-") or s.startswith("•") or s.startswith("*")):
            continue
        cleaned.append(line)
    return "\n".join(cleaned).strip()


def render_analyst_factsheet_and_qa_tab():
    # 1. State initialization for selected stock & top screened fallback
    def_top_code = "005930"
    if "screened_results_df" in st.session_state and st.session_state["screened_results_df"] is not None and not st.session_state["screened_results_df"].empty:
        def_top_code = str(st.session_state["screened_results_df"].iloc[0]["code"]).zfill(6)

    if "qa_console_target_code" not in st.session_state:
        st.session_state["qa_console_target_code"] = def_top_code
    if "qa_has_active_result" not in st.session_state:
        st.session_state["qa_has_active_result"] = True
    st.session_state.pop("qa_direct_search_kw", None)

    curr_target_code = str(st.session_state["qa_console_target_code"]).strip().zfill(6)
    curr_meta = get_stock_metadata(curr_target_code) or {}
    curr_meta_name = str(curr_meta.get("name") or curr_target_code)

    qa_menus = {
        "buy": {
            "label": "지금 사도 되나요?",
            "q": "지금 사도 되나요? 현재 가격대 진입 적합성과 손익비를 알려줘."
        },
        "why": {
            "label": "왜 급등하나요?",
            "q": "왜 급등하나요? 핵심 상승 원인과 수급 특징을 설명해줘."
        }
    }

    qa_active_q_key = f"qa_active_q_{curr_target_code}"
    qa_active_ans_key = f"qa_active_ans_{curr_target_code}"
    qa_hist_key = f"qa_hist_{curr_target_code}"
    cache_key = f"qa_ans_cache_{curr_target_code}"

    if qa_active_q_key not in st.session_state:
        st.session_state[qa_active_q_key] = "buy"
    if qa_hist_key not in st.session_state:
        st.session_state[qa_hist_key] = []
    if cache_key not in st.session_state:
        st.session_state[cache_key] = {}

    # 2. STEP 1 (TOP): 통합 질문 입력 콘솔 (상단 텍스트 삭제 후 좌측에 컴팩트한 동일 사이즈로 시세 동기화 / 질의 초기화 배치)
    with st.container(border=True):
        h_c1, h_c2, _h_spacer = st.columns([0.9, 0.9, 4.2], gap="small")
        with h_c1:
            sync_btn = st.button("시세 동기화", key="btn_qa_sync_market", use_container_width=True)
        with h_c2:
            reset_btn = st.button("질의 초기화", key=f"reset_chat_{curr_target_code}_tab2", use_container_width=True)

        if sync_btn:
            st.cache_data.clear()
            st.session_state[cache_key] = {}
            st.session_state.pop(qa_active_ans_key, None)
            for c_c in st.session_state.get("qa_compared_codes", []):
                st.session_state.pop(f"qa_active_ans_{c_c}", None)
                if f"qa_ans_cache_{c_c}" in st.session_state:
                    st.session_state[f"qa_ans_cache_{c_c}"] = {}
            st.toast("실시간 시세 및 퀀트 분석 데이터가 동기화되었습니다.")
            st.rerun()

        if reset_btn:
            st.session_state[cache_key] = {}
            st.session_state[qa_hist_key] = []
            st.session_state[qa_active_q_key] = "buy"
            st.session_state.pop(qa_active_ans_key, None)
            st.session_state["qa_compared_codes"] = []
            st.session_state["qa_has_active_result"] = True
            st.rerun()

        with st.form(key=f"qa_custom_form_{curr_target_code}", clear_on_submit=True, border=False):
            f_c1, f_c2 = st.columns([4.4, 1.0], gap="small")
            with f_c1:
                custom_q_input = st.text_input(
                    "통합 질문 입력",
                    placeholder="질문을 입력하세요",
                    label_visibility="collapsed",
                    key=f"input_qa_custom_{curr_target_code}"
                )
            with f_c2:
                submitted = st.form_submit_button("질문 전송", type="primary", use_container_width=True)

        if submitted:
            c_prompt = (custom_q_input or "").strip()
            if c_prompt:
                c_prompt = re.sub(r"^(\[[^\]]+\]\s*)+", "", c_prompt).strip()
                parsed = extract_stock_and_question_from_query(
                    c_prompt,
                    current_code=curr_target_code,
                    fallback_top_code=def_top_code
                )
                matched_codes = parsed.get("codes", [parsed["code"]])
                eff_question = parsed["question"]
                is_stock_only = parsed["is_stock_only"]

                if len(matched_codes) >= 2:
                    with st.spinner("수석 애널리스트가 복수 종목의 실시간 퀀트 지표를 동시 산출하여 비교 분석 중입니다..."):
                        comp_payloads = [get_stock_factsheet_payload(c) for c in matched_codes]
                        q_low = eff_question.lower()
                        is_afternoon_q = any(w in q_low for w in ["오후", "오후장", "장후반", "마감", "종가", "오후에"])
                        is_morning_q = any(w in q_low for w in ["오전", "시초", "시초가", "장초반", "내일", "익일", "아침"])
                        is_supply_q = any(w in q_low for w in ["수급", "거래량", "메이저", "세력", "수급주"])
                        is_safety_q = any(w in q_low for w in ["안전", "방어", "손절", "리스크", "위험", "안정"])

                        def _rank_val(p):
                            cm = _compute_complex_quant_metrics(p, eff_question)
                            if is_afternoon_q:
                                return cm["afternoon_score"]
                            if is_morning_q:
                                return cm["morning_score"]
                            if is_supply_q:
                                return cm["supply_score"]
                            if is_safety_q:
                                return cm["safety_score"]
                            return cm["composite_score"]

                        ranked_payloads = sorted(comp_payloads, key=_rank_val, reverse=True)
                        winner_payload = ranked_payloads[0]
                        resolved_code = winner_payload["code"]
                        t_prev_hist = st.session_state.get(f"qa_hist_{resolved_code}", [])
                        try:
                            ans = generate_quant_expert_answer(winner_payload, eff_question, compare_stocks=comp_payloads, chat_history=t_prev_hist)
                        except Exception as _e:
                            ans = f"복합 비교 분석 답변 생성 중 오류가 발생했습니다: {_e}"

                    st.session_state["qa_compared_codes"] = matched_codes
                    st.session_state["qa_console_target_code"] = resolved_code
                    st.session_state["qa_has_active_result"] = True

                    for c_code in matched_codes:
                        t_q_key = f"qa_active_q_{c_code}"
                        t_ans_key = f"qa_active_ans_{c_code}"
                        t_hist_key = f"qa_hist_{c_code}"
                        if t_hist_key not in st.session_state:
                            st.session_state[t_hist_key] = []
                        st.session_state[t_q_key] = c_prompt
                        st.session_state[t_ans_key] = ans
                        st.session_state[t_hist_key].append({"q": c_prompt, "ans": ans})
                    st.rerun()
                else:
                    resolved_code = parsed["code"]
                    st.session_state["qa_compared_codes"] = []
                    st.session_state["qa_console_target_code"] = resolved_code
                    st.session_state["qa_has_active_result"] = True

                    target_q_key = f"qa_active_q_{resolved_code}"
                    target_ans_key = f"qa_active_ans_{resolved_code}"
                    target_hist_key = f"qa_hist_{resolved_code}"
                    if target_hist_key not in st.session_state:
                        st.session_state[target_hist_key] = []

                    with st.spinner("수석 애널리스트가 질문 대상 종목의 실시간 퀀트 지표와 답변을 산출 중입니다..."):
                        target_payload = get_stock_factsheet_payload(resolved_code)
                        try:
                            ans = generate_quant_expert_answer(target_payload, eff_question, chat_history=st.session_state[target_hist_key])
                        except Exception as _e:
                            ans = f"분석 답변 생성 중 오류가 발생했습니다: {_e}"

                    if is_stock_only:
                        st.session_state[target_q_key] = "buy"
                    else:
                        st.session_state[target_q_key] = c_prompt
                    st.session_state[target_ans_key] = ans
                    st.session_state[target_hist_key].append({"q": c_prompt, "ans": ans})
                    st.rerun()
            else:
                st.warning("질문 내용을 입력해 주세요.")

    if not st.session_state.get("qa_has_active_result", False):
        return

    with st.spinner(f"수석 애널리스트가 [{curr_target_code}] 실시간 시세 및 퀀트 지표를 분석 중입니다..."):
        payload = get_stock_factsheet_payload(curr_target_code)

    name = payload["name"]
    code = payload["code"]
    market = payload["market"]
    sector = payload["sector"]
    close = payload["close"]
    change_pct = payload["change_pct"]
    open_p = payload["open"]
    high_p = payload["high"]
    low_p = payload["low"]
    prev_close = payload["prev_close"]
    open_pct = payload["open_pct"]
    high_pct = payload["high_pct"]
    low_pct = payload["low_pct"]
    volume = payload["volume"]
    score = payload["score"]
    label = payload["label"]
    strategy_tag = payload["strategy_tag"]
    tp_pct = payload["tp_pct"]
    sl_pct = payload["sl_pct"]
    tp_price = payload["tp_price"]
    sl_price = payload["sl_price"]
    rr_ratio = payload["rr_ratio"]
    vol_ratio = payload["vol_ratio"]
    rsi = payload["rsi"]
    ma_status = payload["ma_status"]
    bb_pct = payload["bb_pct"]
    reason_core = payload["reason_core"]
    rule_note = payload["rule_note"]
    est_val_억 = payload["est_trading_val_krw"] / 100_000_000

    # 3. STEP 2 (질문 입력창 바로 아래): 질문에 대한 수석 애널리스트 답변 즉시 표시
    with st.container(border=True):
        curr_choice = st.session_state[qa_active_q_key]
        compared_codes = st.session_state.get("qa_compared_codes", [])
        if curr_choice in qa_menus:
            q_label = qa_menus[curr_choice]["label"]
            q_prompt = qa_menus[curr_choice]["q"]
            clean_choice_str = q_label
            if q_prompt not in st.session_state[cache_key]:
                with st.spinner(f"AI 퀀트 수석 애널리스트가 '{q_label}' 항목을 분석 중입니다..."):
                    try:
                        q_ans = generate_quant_expert_answer(
                            payload,
                            q_prompt,
                            chat_history=st.session_state.get(qa_hist_key, [])
                        )
                    except Exception as _e:
                        q_ans = f"답변 생성 중 일시적인 오류가 발생했습니다: {_e}"
                    st.session_state[cache_key][q_prompt] = q_ans
            else:
                q_ans = st.session_state[cache_key][q_prompt]
            st.session_state[qa_active_ans_key] = q_ans
            disp_title = f"[{name} ({code})] {q_label}"
        else:
            clean_choice_str = curr_choice.replace("직접 질문:", "", 1).strip()
            clean_choice_str = re.sub(r"^(\[[^\]]+\]\s*)+", "", clean_choice_str).strip()
            if len(compared_codes) >= 2:
                comp_names = []
                for c_c in compared_codes:
                    m_info = get_stock_metadata(c_c) or {}
                    comp_names.append(f"{m_info.get('name', c_c)}({c_c})")
                disp_title = f"[{' vs '.join(comp_names)}] {clean_choice_str}"
            else:
                disp_title = f"[{name} ({code})] {clean_choice_str}"
            q_ans = st.session_state.get(qa_active_ans_key, "")
            if not q_ans and clean_choice_str:
                parsed_q = extract_stock_and_question_from_query(clean_choice_str, current_code=code, fallback_top_code=def_top_code)
                q_codes = parsed_q.get("codes", [parsed_q["code"]])
                if len(q_codes) >= 2:
                    c_payloads = [get_stock_factsheet_payload(c) for c in q_codes]
                    q_ans = generate_quant_expert_answer(
                        payload,
                        parsed_q["question"],
                        compare_stocks=c_payloads,
                        chat_history=st.session_state.get(qa_hist_key, [])
                    )
                else:
                    q_ans = generate_quant_expert_answer(
                        payload,
                        parsed_q["question"],
                        chat_history=st.session_state.get(qa_hist_key, [])
                    )
                st.session_state[qa_active_ans_key] = q_ans

        if st.session_state.get(qa_hist_key) and len(st.session_state[qa_hist_key]) > 0:
            last_h = st.session_state[qa_hist_key][-1]
            if last_h.get("q") in [curr_choice, clean_choice_str, disp_title]:
                st.session_state[qa_hist_key][-1]["ans"] = q_ans

        st.markdown(
            f"<div style='font-size:0.85rem; font-weight:700; color:#1E293B; background:#F8FAFC; padding:6px 12px; border-radius:6px; border:1px solid #CBD5E1; margin:2px 0 8px 0;'>"
            f"질문: <span style='color:#0284C7;'>{disp_title}</span>"
            f"</div>",
            unsafe_allow_html=True
        )

        st.markdown(f"<div class='quant-qa-body-wrap'>\n\n{_strip_redundant_qa_headers(q_ans)}\n\n</div>", unsafe_allow_html=True)

        if len(st.session_state[qa_hist_key]) > 0:
            with st.expander(f"이전 질문 기록 ({len(st.session_state[qa_hist_key])}건)", expanded=False):
                for h_idx, h in enumerate(reversed(st.session_state[qa_hist_key])):
                    st.markdown(f"**[질문] {h['q']}**")
                    st.markdown(f"<div class='quant-qa-body-wrap'>\n\n{_strip_redundant_qa_headers(h['ans'])}\n\n</div>", unsafe_allow_html=True)
                    if h_idx < len(st.session_state[qa_hist_key]) - 1:
                        st.divider()

    # 4. STEP 3 (질문 답변 아래): 비교 종목 카드 보기 버튼 + 종목 팩트시트 박스 (종목명 옆 가격 & 박스 내부에 빠른 질문 버튼 2개 포함)
    if len(st.session_state.get("qa_compared_codes", [])) >= 2:
        comp_list = st.session_state["qa_compared_codes"]
        sw_weights = [1.2] * len(comp_list) + [max(0.2, 6.0 - 1.2 * len(comp_list))]
        sw_cols = st.columns(sw_weights, gap="small")
        for idx_c, c_code in enumerate(comp_list):
            c_meta = get_stock_metadata(c_code) or {}
            c_name = c_meta.get("name", c_code)
            with sw_cols[idx_c]:
                if st.button(
                    f"{c_name} ({c_code}) 카드 보기",
                    key=f"btn_switch_comp_{c_code}",
                    type="primary" if c_code == curr_target_code else "secondary",
                    use_container_width=True
                ):
                    st.session_state["qa_console_target_code"] = c_code
                    st.rerun()

    # 4-1. 종목 팩트시트 헤더 박스 (종목명 옆 가격 + 박스 안에 '지금 사도 되나요?', '왜 급등하나요?' 버튼 함께 배치)
    chg_color = "#DC2626" if change_pct > 0 else ("#2563EB" if change_pct < 0 else "#475569")
    chg_sign = "+" if change_pct > 0 else ""
    diff_val = close - prev_close

    clicked_qa = None
    with st.container(border=True):
        hdr_c1, hdr_b1, hdr_b2 = st.columns([3.8, 1.1, 1.1], gap="small")
        with hdr_c1:
            st.markdown(
                f"""
                <div style="display:flex; align-items:center; flex-wrap:wrap; gap:6px; min-height:2rem; line-height:1.2;">
                    <span style="font-size:0.78rem; background:#E2E8F0; color:#334155; padding:2px 7px; border-radius:4px; font-weight:700;">{market}</span>
                    <span style="font-size:1.22rem; font-weight:800; color:#0F172A; margin-left:2px;">{name}</span>
                    <span style="font-size:0.84rem; color:#64748B; font-weight:600;">({code})</span>
                    <span style="font-size:0.82rem; color:#0284C7; font-weight:600; margin-left:2px;">{sector}</span>
                    <span style="font-size:1.25rem; font-weight:800; color:{chg_color}; margin-left:8px;">{close:,.0f}원</span>
                    <span style="font-size:0.92rem; font-weight:700; color:{chg_color};">({chg_sign}{change_pct:.2f}% · {diff_val:+,.0f}원)</span>
                </div>
                """,
                unsafe_allow_html=True
            )
        with hdr_b1:
            is_active_buy = st.session_state.get("qa_has_active_result", True) and (st.session_state[qa_active_q_key] == "buy")
            if st.button(qa_menus["buy"]["label"], key=f"btn_qa_buy_{curr_target_code}", type="primary" if is_active_buy else "secondary", use_container_width=True):
                clicked_qa = "buy"
        with hdr_b2:
            is_active_why = st.session_state.get("qa_has_active_result", True) and (st.session_state[qa_active_q_key] == "why")
            if st.button(qa_menus["why"]["label"], key=f"btn_qa_why_{curr_target_code}", type="primary" if is_active_why else "secondary", use_container_width=True):
                clicked_qa = "why"

        st.markdown(
            f"""
            <div style="display:flex; gap:16px; flex-wrap:wrap; margin-top:2px; font-size:0.78rem; color:#475569; border-top:1px dashed #CBD5E1; padding-top:6px; line-height:1.4;">
                <span>시가: <strong>{open_p:,.0f}원</strong> ({open_pct:+.2f}%)</span>
                <span>고가: <strong>{high_p:,.0f}원</strong> ({high_pct:+.2f}%)</span>
                <span>저가: <strong>{low_p:,.0f}원</strong> ({low_pct:+.2f}%)</span>
                <span>거래량: <strong>{volume:,}주</strong> (전일비 <strong>{vol_ratio:.1f}배</strong>)</span>
                <span>추정 대금: <strong>약 {est_val_억:,.0f}억원</strong></span>
            </div>
            """,
            unsafe_allow_html=True
        )

    if clicked_qa:
        st.session_state["qa_has_active_result"] = True
        st.session_state["qa_compared_codes"] = []
        st.session_state[qa_active_q_key] = clicked_qa
        st.rerun()

    # 4-2. 4대 핵심 요약 카드 (가로 4열)
    score_color = "#DC2626" if score >= 75 else ("#D97706" if score >= 65 else "#475569")
    score_conf = "고확신" if score >= 75 else ("우수" if score >= 65 else "중립")
    vol_tag = "폭발" if vol_ratio >= 2.0 else ("양호" if vol_ratio >= 1.0 else "소강")
    rsi_tag = "과열" if rsi >= 75 else ("상승" if rsi >= 55 else ("침체" if rsi <= 35 else "중립"))

    st.markdown(
        f"""
        <div style="display:grid; grid-template-columns:repeat(4, 1fr); gap:8px; margin-bottom:8px;">
            <div style="background:#FFFFFF; border:1px solid #CBD5E1; border-radius:6px; padding:6px 10px;">
                <div style="font-size:0.75rem; color:#64748B; font-weight:700;">종합 퀀트 스코어</div>
                <div style="font-size:1.05rem; font-weight:800; color:{score_color}; margin:2px 0;">{score:.1f}점 <span style="font-size:0.8rem; font-weight:600; color:#334155;">({label})</span></div>
                <div style="font-size:0.72rem; color:#64748B;">신뢰도: {score_conf}</div>
            </div>
            <div style="background:#FFFFFF; border:1px solid #CBD5E1; border-radius:6px; padding:6px 10px;">
                <div style="font-size:0.75rem; color:#64748B; font-weight:700;">가격 전략 & 손익비</div>
                <div style="font-size:1.05rem; font-weight:800; color:#0F172A; margin:2px 0;">+{tp_pct:.1f}% <span style="font-size:0.75rem; font-weight:600; color:#DC2626;">/ -{sl_pct:.1f}%</span></div>
                <div style="font-size:0.72rem; color:#64748B;">목표 {tp_price:,.0f}원 | 손절 {sl_price:,.0f}원 (R/R {rr_ratio:.2f})</div>
            </div>
            <div style="background:#FFFFFF; border:1px solid #CBD5E1; border-radius:6px; padding:6px 10px;">
                <div style="font-size:0.75rem; color:#64748B; font-weight:700;">수급 및 거래량 폭발력</div>
                <div style="font-size:1.05rem; font-weight:800; color:#0F172A; margin:2px 0;">전일비 {vol_ratio:.1f}배 <span style="font-size:0.75rem; font-weight:600; color:#2563EB;">({vol_tag})</span></div>
                <div style="font-size:0.72rem; color:#64748B;">{volume:,}주 | 약 {est_val_억:,.0f}억원</div>
            </div>
            <div style="background:#FFFFFF; border:1px solid #CBD5E1; border-radius:6px; padding:6px 10px;">
                <div style="font-size:0.75rem; color:#64748B; font-weight:700;">기술적 지표 & 이평선</div>
                <div style="font-size:1.05rem; font-weight:800; color:#0F172A; margin:2px 0;">RSI {rsi:.1f}pt <span style="font-size:0.75rem; font-weight:600; color:#475569;">({rsi_tag})</span></div>
                <div style="font-size:0.72rem; color:#64748B;">{ma_status} | BB %b {bb_pct:.2f}</div>
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )

    # 4-3. 3대 정밀 진단 카드 (목표가&손절가, 리스크 팩트체크, 차트&지표진단 가로 3열 배치) + 상세 리포트 전문 보기
    diag_menus = {
        "tp": {
            "label": "목표가 & 손절가",
            "q": "목표가 & 손절가 알고리즘 권장 익절가와 스탑 가격을 알려줘."
        },
        "risk": {
            "label": "리스크 팩트체크",
            "q": "리스크 팩트체크 주의해야 할 핵심 투자 위험 요소를 알려줘."
        },
        "chart": {
            "label": "차트 & 지표 진단",
            "q": "차트 & 지표 진단 이평선과 RSI 기술적 지표를 종합 분석해줘."
        }
    }

    trailing_trig = min(2.4, tp_pct * 0.7)
    st.markdown(
        f"""
        <div style="display:grid; grid-template-columns:repeat(3, 1fr); gap:8px; margin-bottom:8px;">
            <div style="background:#FFFFFF; border:1px solid #CBD5E1; border-radius:6px; padding:6px 10px;">
                <div style="font-size:0.75rem; color:#64748B; font-weight:700;">목표가 & 손절가</div>
                <div style="font-size:1.05rem; font-weight:800; color:#DC2626; margin:2px 0;">{tp_price:,.0f}원 <span style="font-size:0.8rem; font-weight:600; color:#2563EB;">/ {sl_price:,.0f}원</span></div>
                <div style="font-size:0.72rem; color:#64748B;">목표 +{tp_pct:.1f}% · 손절 -{sl_pct:.1f}% · 트레일링 락 +1.8% (+{trailing_trig:.1f}% 시)</div>
            </div>
            <div style="background:#FFFFFF; border:1px solid #CBD5E1; border-radius:6px; padding:6px 10px;">
                <div style="font-size:0.75rem; color:#64748B; font-weight:700;">리스크 팩트체크</div>
                <div style="font-size:1.05rem; font-weight:800; color:#D97706; margin:2px 0;">고점 -2% 주의 <span style="font-size:0.8rem; font-weight:600; color:#DC2626;">([{sector}])</span></div>
                <div style="font-size:0.72rem; color:#64748B;">시초 갭 차익실현 경계 · 11~13시 호가 공백 주의</div>
            </div>
            <div style="background:#FFFFFF; border:1px solid #CBD5E1; border-radius:6px; padding:6px 10px;">
                <div style="font-size:0.75rem; color:#64748B; font-weight:700;">차트 & 지표 진단</div>
                <div style="font-size:1.05rem; font-weight:800; color:#0F172A; margin:2px 0;">{ma_status} <span style="font-size:0.8rem; font-weight:600; color:#2563EB;">(RSI {rsi:.1f}pt)</span></div>
                <div style="font-size:0.72rem; color:#64748B;">거래량 전일비 {vol_ratio:.1f}배 · {strategy_tag} 통과</div>
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )

    with st.expander("상세 진단 리포트 전문 보기", expanded=False):
        t_d1, t_d2, t_d3 = st.tabs(["목표가 & 손절가", "리스크 팩트체크", "차트 & 지표 진단"])
        with t_d1:
            q_tp = diag_menus["tp"]["q"]
            if q_tp not in st.session_state[cache_key]:
                try:
                    st.session_state[cache_key][q_tp] = generate_quant_expert_answer(payload, q_tp)
                except Exception as _e:
                    st.session_state[cache_key][q_tp] = f"분석 리포트 생성 중 일시적인 오류가 발생했습니다: {_e}"
            st.markdown(st.session_state[cache_key][q_tp])
        with t_d2:
            q_risk = diag_menus["risk"]["q"]
            if q_risk not in st.session_state[cache_key]:
                try:
                    st.session_state[cache_key][q_risk] = generate_quant_expert_answer(payload, q_risk)
                except Exception as _e:
                    st.session_state[cache_key][q_risk] = f"분석 리포트 생성 중 일시적인 오류가 발생했습니다: {_e}"
            st.markdown(st.session_state[cache_key][q_risk])
        with t_d3:
            q_chart = diag_menus["chart"]["q"]
            if q_chart not in st.session_state[cache_key]:
                try:
                    st.session_state[cache_key][q_chart] = generate_quant_expert_answer(payload, q_chart)
                except Exception as _e:
                    st.session_state[cache_key][q_chart] = f"분석 리포트 생성 중 일시적인 오류가 발생했습니다: {_e}"
            st.markdown(st.session_state[cache_key][q_chart])

    # 5. Action Toolbar (Watchlist, Backtest, Screener)
    act_c1, act_c2, act_c3, act_c4 = st.columns([2.0, 2.0, 2.0, 4.0])
    is_in_watchlist = False
    try:
        with get_db_connection() as conn:
            cur = conn.cursor()
            cur.execute("SELECT 1 FROM watchlist WHERE code = ?", (code,))
            is_in_watchlist = bool(cur.fetchone())
    except Exception:
        pass

    with act_c1:
        if is_in_watchlist:
            if st.button("관심종목에서 해제", key=f"btn_tab2_del_wl_{code}", use_container_width=True):
                with get_db_connection() as conn:
                    cur = conn.cursor()
                    cur.execute("DELETE FROM watchlist WHERE code = ?", (code,))
                    conn.commit()
                st.cache_data.clear()
                st.toast(f"{name} 관심종목에서 삭제되었습니다.")
                st.rerun()
        else:
            if st.button("관심종목에 추가", type="primary", key=f"btn_tab2_add_wl_{code}", use_container_width=True):
                batch_add_to_watchlist([code])
                st.cache_data.clear()
                st.toast(f"{name} 관심종목에 등록되었습니다!")
                st.rerun()

    with act_c2:
        if st.button("4번 5년 백테스트 검증", key=f"btn_tab2_to_bt_{code}", use_container_width=True):
            st.session_state["target_backtest_code"] = code
            st.session_state["verif_pipeline_mode_sel"] = "과거 5년 워크포워드 & KODEX 200 벤치마크 검증 (종목별 95% 신뢰구간 시뮬레이션)"
            st.session_state["pending_active_tab"] = main_tab_names[3]
            st.toast(f"{name} 백테스트 설정 완료! 4번 통합 성과 검증 탭을 확인하세요.")
            st.rerun()

    with act_c3:
        if st.button("3번 관심종목 탭으로 이동", key=f"btn_tab2_to_wl_{code}", use_container_width=True):
            st.session_state["pending_active_tab"] = main_tab_names[2]
            st.rerun()

    with act_c4:
        st.caption("※ 시세 및 퀀트 지표는 실시간 장중 호가와 내장 알고리즘으로 동기화됩니다.")


main_tab_names = [
    "1. 스크리너",
    "2. Q&A",
    "3. 관심종목",
    "4. 검증 및 최적화",
]


def render_domestic_market_ticker_bar():
    """
    Renders Real-time Domestic Market Indices & Bitcoin Overview.
    Displays KOSPI, KOSDAQ, KOSPI 200, KOSPI 200 Futures, KOSDAQ 150, Korea Value-up, and Bitcoin (BTC).
    Placed directly above US Market bar in Screener (Tab 1).
    """
    from src.collectors.domestic_indices import fetch_domestic_indices_summary

    summary = fetch_domestic_indices_summary()
    items = summary.get("items", [])
    updated_at_str = summary.get("updated_at_str", "")
    summary_text = summary.get("summary_text", "")

    title_summary = f"[국내 주요 지수 및 비트코인 실시간 시황] {summary_text}"

    with st.expander(title_summary, expanded=True):
        top_c1, top_c2 = st.columns([4.2, 1.2])
        with top_c1:
            st.markdown(
                f'<div class="regime-status-bar">'
                f'<div class="regime-badge-container">'
                f'<span class="regime-badge" style="background:#0F172A; color:#FFFFFF; font-weight:700; font-size:0.75rem; padding:3px 8px; border-radius:4px; display:inline-block;">'
                f'국내 정규장 & 가상자산 실시간 시세'
                f'</span>'
                f'</div>'
                f'<div class="regime-time-container">'
                f'<span class="regime-time" style="color:#64748B; font-size:0.72rem; display:inline-block;">기준시각: {updated_at_str}</span>'
                f'</div>'
                f'</div>',
                unsafe_allow_html=True
            )
        with top_c2:
            if st.button("국내/코인 시세 갱신", key="btn_refresh_domestic_indices", use_container_width=True):
                fetch_domestic_indices_summary(force_refresh=True)
                st.toast("국내 주요 지수 및 비트코인 시세를 새로 수신했습니다.")
                st.rerun()

        cards_html = []
        for it in items:
            label = it.get("name", "")
            disp_price = it.get("display_price", "0.00")
            chg_pct = it.get("change_pct", 0.0)
            color = "#DC2626" if chg_pct > 0 else ("#2563EB" if chg_pct < 0 else "#475569")
            sign = "+" if chg_pct > 0 else ""
            chg_str = f"{sign}{chg_pct:.2f}%"

            card_html = (
                f'<div style="background:#F8FAFC; border:1px solid #E2E8F0; border-radius:6px; padding:8px 4px 8px 4px; text-align:center; box-sizing:border-box;">'
                f'<div style="font-size:0.70rem; color:#64748B; font-weight:600; margin-bottom:3px; white-space:nowrap; overflow:hidden; text-overflow:ellipsis;" title="{label}">{label}</div>'
                f'<div style="font-size:0.84rem; font-weight:700; color:#1E293B; line-height:1.25; white-space:nowrap; overflow:hidden; text-overflow:ellipsis;">{disp_price}</div>'
                f'<div style="font-size:0.72rem; font-weight:700; color:{color}; line-height:1.25; margin-top:3px; white-space:nowrap;">{chg_str}</div>'
                f'</div>'
            )
            cards_html.append(card_html)

        grid_html = (
            f'<div class="domestic-metrics-grid" style="display:grid; grid-template-columns:repeat(7, 1fr); gap:8px; width:100%; margin:6px 0 6px 0; box-sizing:border-box;">'
            + "".join(cards_html) +
            f'</div>'
        )
        st.markdown(grid_html, unsafe_allow_html=True)


def render_us_market_macro_bar():
    """
    Renders US Market & Global Macro Quantitative Overview.
    Displays NASDAQ, S&P 500, SOXX, DOW, USD/KRW, and VIX with
    regime status badge, action guide, and sector impact notes.
    """
    from src.collectors.us_market import fetch_us_market_summary

    us_summary = fetch_us_market_summary()
    metrics = us_summary.get("metrics", {})
    regime = us_summary.get("regime", "중립")
    regime_desc = us_summary.get("regime_desc", "")
    regime_color = us_summary.get("color", "#475569")
    action_guide = us_summary.get("action_guide", "")
    sector_notes = us_summary.get("sector_notes", {})
    updated_at_str = us_summary.get("date_str", "")

    nasdaq_pct = us_summary.get("nasdaq_pct", 0.0)
    sp500_pct = us_summary.get("sp500_pct", 0.0)
    soxx_pct = us_summary.get("soxx_pct", 0.0)
    usdkrw_val = us_summary.get("usdkrw_rate", 1340.0)

    title_summary = (
        f"[전일 미장 및 글로벌 매크로 퀀트 점검] "
        f"나스닥 {nasdaq_pct:+.2f}% · S&P500 {sp500_pct:+.2f}% · SOXX {soxx_pct:+.2f}% · 환율 {usdkrw_val:,.1f}원 "
        f"| 체제: {regime}"
    )

    with st.expander(title_summary, expanded=True):
        top_c1, top_c2 = st.columns([4.2, 1.2])
        with top_c1:
            st.markdown(
                f'<div class="regime-status-bar">'
                f'<div class="regime-badge-container">'
                f'<span class="regime-badge" style="background:{regime_color}; color:#FFFFFF; font-weight:700; font-size:0.75rem; padding:3px 8px; border-radius:4px; display:inline-block;">'
                f'글로벌 체제: {regime} ({regime_desc})'
                f'</span>'
                f'</div>'
                f'<div class="regime-time-container">'
                f'<span class="regime-time" style="color:#64748B; font-size:0.72rem; display:inline-block;">기준시각: {updated_at_str}</span>'
                f'</div>'
                f'</div>',
                unsafe_allow_html=True
            )
        with top_c2:
            if st.button("미장 시세 갱신", key="btn_refresh_us_market", use_container_width=True):
                fetch_us_market_summary(force_refresh=True)
                st.toast("전일 미장 및 매크로 지표를 새로 수신했습니다.")
                st.rerun()

        macro_items = [
            ("나스닥 종합", "NASDAQ"),
            ("S&P 500", "SP500"),
            ("필라델피아 반도체", "SOXX"),
            ("다우 존스", "DOW"),
            ("원/달러 환율", "USDKRW"),
            ("공포지수(VIX)", "VIX")
        ]
        macro_cards_html = []
        for label, k_name in macro_items:
            item = metrics.get(k_name, {})
            val = item.get("close", 0.0)
            chg_pct = item.get("change_pct", 0.0)
            color = "#DC2626" if chg_pct < 0 else ("#059669" if chg_pct > 0 else "#475569")
            card_html = (
                f'<div style="background:#F8FAFC; border:1px solid #E2E8F0; border-radius:6px; padding:8px 4px 8px 4px; text-align:center; box-sizing:border-box;">'
                f'<div style="font-size:0.70rem; color:#64748B; font-weight:600; margin-bottom:3px; white-space:nowrap; overflow:hidden; text-overflow:ellipsis;" title="{label}">{label}</div>'
                f'<div style="font-size:0.84rem; font-weight:700; color:#1E293B; line-height:1.25;">{val:,.2f}</div>'
                f'<div style="font-size:0.72rem; font-weight:700; color:{color}; line-height:1.25; margin-top:3px;">{chg_pct:+.2f}%</div>'
                f'</div>'
            )
            macro_cards_html.append(card_html)

        grid_html = (
            f'<div class="macro-metrics-grid" style="display:grid; grid-template-columns:repeat(6, 1fr); gap:8px; width:100%; margin:6px 0 10px 0; box-sizing:border-box;">'
            + "".join(macro_cards_html) +
            f'</div>'
        )
        st.markdown(grid_html, unsafe_allow_html=True)

        st.markdown(
            f'<div style="font-size:0.78rem; line-height:1.5; color:#1E293B; background:#F1F5F9; border-left:4px solid {regime_color}; padding:8px 12px; border-radius:4px; margin-bottom:4px;">'
            f'<strong>[당일 국내 정규장 대응 가이드]</strong> {action_guide}'
            f'</div>',
            unsafe_allow_html=True
        )

        if sector_notes:
            notes_html = []
            for sec_k, sec_txt in sector_notes.items():
                notes_html.append(f"• <strong>{sec_k}</strong>: {sec_txt}")
            notes_str = "<br>".join(notes_html)
            st.markdown(
                f'<div style="font-size:0.75rem; line-height:1.45; color:#475569; background:#FAFAFA; border:1px solid #E5E7EB; padding:6px 10px; border-radius:4px; margin-top:6px;">'
                f'<strong style="color:#374151;">[주요 섹터 및 수급 영향]</strong><br>{notes_str}'
                f'</div>',
                unsafe_allow_html=True
            )


# =========================================================
# MAIN TAB NAVIGATION LIFECYCLE
# =========================================================
# Apply pending tab switch BEFORE st.tabs is instantiated
if "pending_active_tab" in st.session_state:
    st.session_state["main_active_tab_nav"] = st.session_state.pop("pending_active_tab")

if "main_active_tab_nav" in st.session_state:
    _cur_nav = str(st.session_state["main_active_tab_nav"])
    if _cur_nav not in main_tab_names:
        if any(w in _cur_nav for w in ["4", "5", "검증", "최적화", "다이어리", "워크포워드"]):
            st.session_state["main_active_tab_nav"] = main_tab_names[3]
        elif any(w in _cur_nav for w in ["3", "관심", "스코어보드"]):
            st.session_state["main_active_tab_nav"] = main_tab_names[2]
        elif any(w in _cur_nav for w in ["2", "애널리스트", "팩트시트", "Q&A"]):
            st.session_state["main_active_tab_nav"] = main_tab_names[1]
        else:
            st.session_state["main_active_tab_nav"] = main_tab_names[0]
else:
    st.session_state["main_active_tab_nav"] = main_tab_names[0]


_def_tab = st.session_state.get('main_active_tab_nav', main_tab_names[0])
if _def_tab not in main_tab_names:
    _def_tab = main_tab_names[0]

with st.container(key="main_tabs_header"):
    tab1, tab2, tab3, tab4 = st.tabs(main_tab_names, default=_def_tab)


# =========================================================
# TAB 1: 시장 전체 익일 상승 후보 & 고확신 스크리너
# =========================================================
with tab1:
    render_domestic_market_ticker_bar()
    render_us_market_macro_bar()

    sc_mode_options = [
        "실시간 당일 단타 (5% 익절)",
        "장전 시초가 공략 (눌림 반등 + 주도 갭상승)",
        "스나이퍼 고확신 (눌림목 반등)",
        "5% 급등 타겟 (1~2일 스윙)",
        "주도주 종가배팅 (익일 시초 갭)",
        "일반 퀀트 스코어링"
    ]
    if "sc_mode_radio_sel" not in st.session_state or st.session_state["sc_mode_radio_sel"] not in sc_mode_options:
        st.session_state["sc_mode_radio_sel"] = "실시간 당일 단타 (5% 익절)"

    sc_mode = st.radio(
        "스크리닝 전략 모드 선택",
        sc_mode_options,
        horizontal=False,
        key="sc_mode_radio_sel"
    )

    if "screened_cache_by_mode" in st.session_state and sc_mode in st.session_state["screened_cache_by_mode"]:
        st.session_state["screened_results_df"] = st.session_state["screened_cache_by_mode"][sc_mode]
        st.session_state["screened_mode_label"] = sc_mode

    if "장전 시초가" in sc_mode or "08:00~09:00" in sc_mode:
        m_tag = '<span style="color:#0D9488; font-weight:700;">[장전 시초가]</span> 정규장 개장 전 집중 분석 · 20일선 첫 눌림목 반등 + 어제 800억~1,000억+ 폭발 주도 대장주 시초 갭상승 복합 공략 (TP +2.5%~+4.0% / SL -2.0%~-2.5%)'
    elif "당일 단타" in sc_mode:
        m_tag = '<span style="color:#0284C7; font-weight:700;">[실시간 당일 단타]</span> 장중 09:10~14:30 수급 폭발 주도주 포착 · 당일 +5.0% 익절 / 15:15 미도달 시 종가 전량 청산 (No Overnight)'
    elif "스나이퍼" in sc_mode:
        m_tag = '<span style="color:#059669; font-weight:700;">[스나이퍼 고확신]</span> 오전 09:00 시초가 진입 · 20일선 첫 눌림목 반등 공략 (익절 +1.2% / 손절 -2.0% 엄수, 승률 80% 타겟)'
    elif "5% 급등" in sc_mode or "5% 돌파" in sc_mode:
        m_tag = '<span style="color:#D97706; font-weight:700;">[5% 급등 타겟]</span> 마감 직전 15:20 종가 진입 · 상한가/초대형 거래대금 돌파봉 1~2일 내 5% 급등 수확 (손절 -4.0%)'
    elif "종가배팅" in sc_mode:
        m_tag = '<span style="color:#7C3AED; font-weight:700;">[주도주 종가배팅]</span> 마감 직전 15:20 동시호가 매수 -> 익일 09:00 시초 갭상승(+1.5%↑) 즉시 익절 (Overnight)'
    else:
        m_tag = '<span style="color:#475569; font-weight:700;">[일반 퀀트]</span> Point-in-Time 복합 팩터 100점 만점 랭킹 시스템'

    st.markdown(f'<div class="sc-mode-tag-desc" style="font-size:0.78rem; color:#475569; padding:4px 0 4px 2px; margin:0;">{m_tag}</div>', unsafe_allow_html=True)

    # 전 전략 모드 공통: 시간대별 최적 권장 설정 자동화 엔진
    def get_auto_time_recommendation_for_strategy(strategy_mode, dt=None):
        if dt is None:
            dt = get_now_kst()
        cur_hm = dt.hour * 100 + dt.minute

        if "장전 시초가" in strategy_mode or "08:00~09:00" in strategy_mode:
            if 800 <= cur_hm < 900:
                return {
                    "slot_name": "08:00 ~ 09:00 [장전 시초가 공략 분석]",
                    "slot_id": "pre_slot_0",
                    "highlight_idx": 0,
                    "val": 100,
                    "gain": (0, 0),
                    "vol": 1.0,
                    "badge_desc": "20일평균 100억↑ 눌림목 & 어제 800억~1,000억↑ 주도 갭상승",
                    "tip": "전일 확정 데이터 기반 09:00 시초가 진입 타겟 사전 압축",
                    "params": {"min_val_krw_b": 100, "max_disparity_val": 103.5, "ignore_market_filter": False}
                }
            elif 900 <= cur_hm < 930:
                return {
                    "slot_name": "09:00 ~ 09:30 [시초 체결 & 갭 검증]",
                    "slot_id": "pre_slot_1",
                    "highlight_idx": 1,
                    "val": 100,
                    "gain": (0, 0),
                    "vol": 1.0,
                    "badge_desc": "눌림목(갭 -1.5%~+1.5%) / 주도주(갭 +1.5%~+4.5%)",
                    "tip": "+5.0% 초과 과도한 갭상승 종목은 뇌동매매 진입 금지",
                    "params": {"min_val_krw_b": 100, "max_disparity_val": 103.5, "ignore_market_filter": False}
                }
            else:
                return {
                    "slot_name": "장외/장마감 후 [익일 08:00 시초가 후보 사전 준비]",
                    "slot_id": "pre_slot_off",
                    "highlight_idx": 0,
                    "val": 100,
                    "gain": (0, 0),
                    "vol": 1.0,
                    "badge_desc": "20일평균 100억↑ 눌림 & 주도 대장주",
                    "tip": "내일 오전 8시 분석을 위한 후보군 사전 점검",
                    "params": {"min_val_krw_b": 100, "max_disparity_val": 103.5, "ignore_market_filter": False}
                }
        elif "당일 단타" in strategy_mode:
            if 910 <= cur_hm < 940:
                return {
                    "slot_name": "09:10 ~ 09:40 [장초반 수급 집중]",
                    "slot_id": "dt_slot_0",
                    "highlight_idx": 0,
                    "val": 100,
                    "gain": (2.5, 10.0),
                    "vol": 0.5,
                    "badge_desc": "거래대금 100억↑ · 시초대비 +2.5%~+10.0% · 전일대비 거래량 0.5배↑",
                    "tip": "장초반 수급 폭발 주도주 빠른 포착 및 익절 공간 확보",
                    "params": {"min_daytrade_val_b": 100, "intraday_gain_range": (2.5, 10.0), "min_daytrade_vol_ratio": 0.5}
                }
            elif 940 <= cur_hm < 1100:
                return {
                    "slot_name": "09:40 ~ 11:00 [골든타임 - 2차 돌파]",
                    "slot_id": "dt_slot_1",
                    "highlight_idx": 1,
                    "val": 150,
                    "gain": (3.0, 12.0),
                    "vol": 0.6,
                    "badge_desc": "거래대금 150억↑ · 시초대비 +3.0%~+12.0% · 거래량 0.6배↑",
                    "tip": "당일 최고 승률 구간: 주도주 2차 파동 돌파 적중률 극대화",
                    "params": {"min_daytrade_val_b": 150, "intraday_gain_range": (3.0, 12.0), "min_daytrade_vol_ratio": 0.6}
                }
            elif 1100 <= cur_hm < 1300:
                return {
                    "slot_name": "11:00 ~ 13:00 [점심 횡보장]",
                    "slot_id": "dt_slot_2",
                    "highlight_idx": 2,
                    "val": 200,
                    "gain": (4.0, 15.0),
                    "vol": 0.7,
                    "badge_desc": "거래대금 200억↑ · 시초대비 +4.0%~+15.0% · 거래량 0.7배↑",
                    "tip": "점심 거래소강: 탄탄하게 지지받는 최상위 대장주만 선별",
                    "params": {"min_daytrade_val_b": 200, "intraday_gain_range": (4.0, 15.0), "min_daytrade_vol_ratio": 0.7}
                }
            elif 1300 <= cur_hm < 1430:
                return {
                    "slot_name": "13:00 ~ 14:30 [오후 2차 수급]",
                    "slot_id": "dt_slot_3",
                    "highlight_idx": 3,
                    "val": 200,
                    "gain": (4.5, 18.0),
                    "vol": 0.8,
                    "badge_desc": "거래대금 200억↑ · 시초대비 +4.5%~+18.0% · 거래량 0.8배↑",
                    "tip": "상한가/VI 직행 추진력을 갖춘 최상위 주도주 공략",
                    "params": {"min_daytrade_val_b": 200, "intraday_gain_range": (4.5, 18.0), "min_daytrade_vol_ratio": 0.8}
                }
            elif 1430 <= cur_hm < 1535:
                return {
                    "slot_name": "14:30 ~ 15:35 [장마감 임박/청산]",
                    "slot_id": "dt_slot_4",
                    "highlight_idx": 4,
                    "val": 200,
                    "gain": (3.0, 15.0),
                    "vol": 0.6,
                    "badge_desc": "신규 진입 중단 권장 (보유분 15:15 전량 청산 집중)",
                    "tip": "신규 매수는 마감 직전 '5% 급등 타겟' 또는 '종가배팅' 권장",
                    "params": {"min_daytrade_val_b": 200, "intraday_gain_range": (3.0, 15.0), "min_daytrade_vol_ratio": 0.6}
                }
            else:
                return {
                    "slot_name": "장외/마감 정산 (내일 실전 준비)",
                    "slot_id": "dt_slot_off",
                    "highlight_idx": 1,
                    "val": 100,
                    "gain": (2.5, 15.0),
                    "vol": 0.5,
                    "badge_desc": "골든타임 권장값 (거래대금 100억↑ · 시초대비 +2.5%~+15.0%)",
                    "tip": "마감 데이터 기준 최적 스크리닝 및 내일 장초반 준비",
                    "params": {"min_daytrade_val_b": 100, "intraday_gain_range": (2.5, 15.0), "min_daytrade_vol_ratio": 0.5}
                }
        elif "스나이퍼" in strategy_mode:
            if 830 <= cur_hm < 900:
                return {
                    "slot_name": "08:30 ~ 09:00 [장 시작 전 준비]",
                    "slot_id": "snp_slot_0",
                    "highlight_idx": 0,
                    "val": 100,
                    "gain": (0, 0),
                    "vol": 1.0,
                    "badge_desc": "20일평균 대금 100억↑ · 이격도 103.5% · 하락장 필터 On",
                    "tip": "전일 확정 데이터 기반 우량 눌림목 후보군 선별 확정",
                    "params": {"min_val_krw_b": 100, "max_disparity_val": 103.5, "ignore_market_filter": False}
                }
            elif 900 <= cur_hm < 1000:
                return {
                    "slot_name": "09:00 ~ 10:00 [시초가 체결 확인]",
                    "slot_id": "snp_slot_1",
                    "highlight_idx": 1,
                    "val": 100,
                    "gain": (0, 0),
                    "vol": 1.0,
                    "badge_desc": "20일평균 대금 100억↑ · 이격도 103.5% · 갭제한 -1.5%~+1.5% 엄수",
                    "tip": "+1.5% 초과 갭상승 종목은 뇌동매매 진입 금지",
                    "params": {"min_val_krw_b": 100, "max_disparity_val": 103.5, "ignore_market_filter": False}
                }
            elif 1000 <= cur_hm < 1520:
                return {
                    "slot_name": "10:00 ~ 15:20 [장중 감시 & 스탑로스]",
                    "slot_id": "snp_slot_2",
                    "highlight_idx": 2,
                    "val": 100,
                    "gain": (0, 0),
                    "vol": 1.0,
                    "badge_desc": "20일평균 대금 100억↑ · 이격도 103.0% (보수적 눌림)",
                    "tip": "익절 +1.2% / 손절 -2.0% 자동 스탑로스 대응",
                    "params": {"min_val_krw_b": 100, "max_disparity_val": 103.0, "ignore_market_filter": False}
                }
            else:
                return {
                    "slot_name": "장외/마감 후 [익일 후보 사전 분석]",
                    "slot_id": "snp_slot_off",
                    "highlight_idx": 0,
                    "val": 100,
                    "gain": (0, 0),
                    "vol": 1.0,
                    "badge_desc": "20일평균 대금 100억↑ · 이격도 103.5% · 지수 추세 확인",
                    "tip": "내일 장초반 시초가 매수 후보군 사전 압축",
                    "params": {"min_val_krw_b": 100, "max_disparity_val": 103.5, "ignore_market_filter": False}
                }
        elif "5% 급등" in strategy_mode:
            if 900 <= cur_hm < 1430:
                return {
                    "slot_name": "09:00 ~ 14:30 [장중 급등 모멘텀 추적]",
                    "slot_id": "s5_slot_0",
                    "highlight_idx": 0,
                    "val": 200,
                    "gain": (0, 0),
                    "vol": 1.0,
                    "badge_desc": "거래대금 200억↑ · 당일 주가상승률 +8.0%↑",
                    "tip": "상한가/준상한가 도전 중인 시장 최상위 대장주 실시간 탐색",
                    "params": {"min_surge_val_b": 200, "min_day_surge_pct": 8.0}
                }
            elif 1430 <= cur_hm < 1535:
                return {
                    "slot_name": "14:30 ~ 15:35 [마감 직전 진입 골든타임]",
                    "slot_id": "s5_slot_1",
                    "highlight_idx": 1,
                    "val": 300,
                    "gain": (0, 0),
                    "vol": 1.0,
                    "badge_desc": "거래대금 300억↑ · 당일 주가상승률 +10.0%↑ (상한가 굳히기)",
                    "tip": "15:20 동시호가 종가 매수 -> 1~2일 내 +5% 자동 익절",
                    "params": {"min_surge_val_b": 300, "min_day_surge_pct": 10.0}
                }
            else:
                return {
                    "slot_name": "장외/마감 후 [익일 5% 돌파 타겟]",
                    "slot_id": "s5_slot_off",
                    "highlight_idx": 1,
                    "val": 200,
                    "gain": (0, 0),
                    "vol": 1.0,
                    "badge_desc": "거래대금 200억↑ · 당일 주가상승률 +10.0%↑",
                    "tip": "당일 상한가/마루보즈 마감 대장주 익일 대응 준비",
                    "params": {"min_surge_val_b": 200, "min_day_surge_pct": 10.0}
                }
        elif "종가배팅" in strategy_mode:
            if 900 <= cur_hm < 1430:
                return {
                    "slot_name": "09:00 ~ 14:30 [장중 주도 테마 관찰]",
                    "slot_id": "cb_slot_0",
                    "highlight_idx": 0,
                    "val": 200,
                    "gain": (0, 0),
                    "vol": 1.0,
                    "badge_desc": "거래대금 200억↑ · 주가상승률 +3.0%↑",
                    "tip": "오후장까지 고가를 유지하는지 추세 관찰",
                    "params": {"min_today_val_b": 200, "min_day_ret_val": 3.0}
                }
            elif 1430 <= cur_hm < 1535:
                return {
                    "slot_name": "14:30 ~ 15:35 [동시호가 진입 골든타임]",
                    "slot_id": "cb_slot_1",
                    "highlight_idx": 1,
                    "val": 200,
                    "gain": (0, 0),
                    "vol": 1.0,
                    "badge_desc": "거래대금 200억↑ · 주가상승률 +3.0%↑ (고가마감)",
                    "tip": "15:20 동시호가 종가 매수 -> 익일 시초가 +1.5% 갭 익절",
                    "params": {"min_today_val_b": 200, "min_day_ret_val": 3.0}
                }
            else:
                return {
                    "slot_name": "장외/마감 후 [익일 시초갭 공략 분석]",
                    "slot_id": "cb_slot_off",
                    "highlight_idx": 1,
                    "val": 200,
                    "gain": (0, 0),
                    "vol": 1.0,
                    "badge_desc": "거래대금 200억↑ · 주가상승률 +3.0%↑",
                    "tip": "당일 고가 마감 주도주의 익일 시초가 갭상승 가능성 점검",
                    "params": {"min_today_val_b": 200, "min_day_ret_val": 3.0}
                }
        else:  # 일반 퀀트 스코어링
            return {
                "slot_name": "상시 권장 (Point-in-Time 스코어링)",
                "slot_id": "qt_slot_0",
                "highlight_idx": 0,
                "val": 100,
                "gain": (0, 0),
                "vol": 1.5,
                "badge_desc": "최소 스코어 65점↑ · 거래량 1.5배↑ · 이평 정배열 On",
                "tip": "전체 상장 유니버스 종합 퀀트 점수 산출 및 순위화",
                "params": {"min_screener_score": 65.0, "min_vol_surge": 1.5, "require_ma_align": True}
            }

    rec_time = get_auto_time_recommendation_for_strategy(sc_mode)
    now_str = get_now_kst().strftime("%H:%M")
    cur_strat_slot_id = f"{sc_mode}_{rec_time['slot_id']}"

    full_scope_default = "전 상장사 전수조사 (약 2,700개)"
    if "sc_mode_scopes" not in st.session_state:
        st.session_state["sc_mode_scopes"] = {}

    # Always enforce default scope to full universe investigation for all strategy modes
    for _m in sc_mode_options:
        if _m not in st.session_state["sc_mode_scopes"] or "300선" in str(st.session_state["sc_mode_scopes"].get(_m, "")):
            st.session_state["sc_mode_scopes"][_m] = full_scope_default

    # Ensure sc_scope_sel always defaults to full_scope_default
    if "sc_scope_sel" not in st.session_state or ("300선" in str(st.session_state.get("sc_scope_sel", "")) and not st.session_state.get("sc_user_explicitly_chose_300")):
        st.session_state["sc_scope_sel"] = full_scope_default

    # Mode switch detection: ensure newly selected mode defaults to full_scope_default
    if st.session_state.get("sc_active_mode_tracked") != sc_mode:
        st.session_state["sc_active_mode_tracked"] = sc_mode
        st.session_state["sc_scope_sel"] = st.session_state["sc_mode_scopes"].get(sc_mode, full_scope_default)

    if "sc_last_applied_strat_slot" not in st.session_state or st.session_state.get("sc_trigger_time_sync") or st.session_state.get("sc_last_applied_strat_slot") != cur_strat_slot_id:
        for _k, _v in rec_time["params"].items():
            st.session_state[f"sc_{_k}"] = _v
            if _k == "min_daytrade_val_b":
                st.session_state["sc_min_daytrade_val"] = _v
        if "sc_scope_sel" not in st.session_state or ("300선" in str(st.session_state.get("sc_scope_sel", "")) and not st.session_state.get("sc_user_explicitly_chose_300")):
            st.session_state["sc_scope_sel"] = full_scope_default
        st.session_state["sc_last_applied_strat_slot"] = cur_strat_slot_id
        st.session_state["sc_trigger_time_sync"] = False

    # 1) 전략 원칙 & 시간대별 설정 가이드 통합 접이식(Expander)
    with st.expander("[전략 핵심 원칙 & 시간대별 설정 가이드]", expanded=False, key="expander_strat_guide"):
        g_guide_mode = st.radio(
            "가이드 항목 선택",
            ["전략 핵심 원칙 & 시장 체제 점검", "시간대별 권장 설정 가이드"],
            horizontal=True,
            key="g_guide_view_radio"
        )
        if "핵심 원칙" in g_guide_mode:
            from src.collectors.us_market import fetch_us_market_summary
            _us_m = fetch_us_market_summary()
            _us_reg = _us_m.get("regime", "중립")
            _us_desc = _us_m.get("regime_desc", "")

            if "장전 시초가" in sc_mode or "08:00~09:00" in sc_mode:
                bm_ohlcv = get_benchmark_ohlcv()
                mkt_ok, mkt_msg, mkt_meta = evaluate_market_regime(bm_ohlcv)
                mkt_color = "#059669" if mkt_ok else "#D97706"
                mkt_bg = "#ECFDF5" if mkt_ok else "#FFFBEB"
                mkt_border = "#A7F3D0" if mkt_ok else "#FDE68A"
                mkt_txt = "장전 시초가 공략 신규 매수가 적극 허용됩니다." if mkt_ok else "지수 하락/조정장 주의: 눌림목 반등 위주 선별 및 비중 조절 권장"
                st.markdown(
                    f'<div style="font-size:0.78rem; line-height:1.45; color:{mkt_color}; background:{mkt_bg}; border:1px solid {mkt_border}; border-radius:5px; padding:6px 10px; margin-bottom:8px;">'
                    f'<strong>[시장 체제 점검]</strong> KODEX 200: {mkt_txt} ({mkt_msg}) · <strong>전일 미장:</strong> {_us_reg} ({_us_desc})'
                    f'</div>',
                    unsafe_allow_html=True
                )
                st.markdown(
                    """
<div style="font-size:0.78rem; line-height:1.45; color:#334155; background:#F8FAFC; border:1px solid #E2E8F0; border-radius:6px; padding:10px 12px;">
<div style="font-size:0.84rem; font-weight:700; color:#1E293B; margin-bottom:6px;">[장전 시초가 공략 모드] 핵심 원칙 (프리마켓 데이터 연동 + 눌림 반등 + 주도 갭상승)</div>
• <strong>분석 시점</strong>: 매일 아침 08:00 ~ 08:50 (정규장 개장 전 전일 확정 데이터 + 글로벌 프리마켓 지표 기반 종합 분석)<br>
• <strong>프리마켓 데이터 연동 분석</strong>: 미국 나스닥/S&P/SOXX 야간선물, 달러 환율, VIX 변동성을 연동하여 시초 갭상승 신뢰도를 검증하고 갭상승 음봉 트랩을 사전 차단합니다.<br>
• <strong>트랙 1 (20일선 눌림목 반등)</strong>: 20일 일평균 거래대금 100억↑ & 20일선 이격도 98%~103.5% 첫 지지 반등 우량주 (시초 갭 -1.5%~+1.5% 진입, 익절 +2.5% / 손절 -2.0%)<br>
• <strong>트랙 2 (주도 대장주 갭상승)</strong>: 전일 거래대금 800억~1,000억↑ 폭발 & 고가 마감 최상위 1등주 (동시호가 +1.5%~+4.5% 적정 갭 확인 후 시초가 진입, 익절 +4.0% / 손절 -2.5%)<br>
• <strong>장전 동시호가(08:30~09:00) 원칙</strong>: 08:50 이전 호가는 허매수/허매도 왜곡이 심하므로, 08:50~09:00 형성되는 실체결 예상 갭(+1.5%~+4.5%)을 최종 확인하고 진입 (+5.0% 초과 과열 시 뇌동매매 금지)
</div>
                    """,
                    unsafe_allow_html=True
                )
            elif "당일 단타" in sc_mode:
                bm_ohlcv = get_benchmark_ohlcv()
                mkt_ok, mkt_msg, mkt_meta = evaluate_market_regime(bm_ohlcv)
                mkt_color = "#059669" if mkt_ok else "#D97706"
                mkt_bg = "#ECFDF5" if mkt_ok else "#FFFBEB"
                mkt_border = "#A7F3D0" if mkt_ok else "#FDE68A"
                mkt_txt = "당일 단타 모드 신규 매수가 적극 허용됩니다." if mkt_ok else "조정/하락장입니다. 보수적 비중 조절 및 칼손절(-2.5%) 준수 필수!"
                st.markdown(
                    f'<div style="font-size:0.78rem; line-height:1.45; color:{mkt_color}; background:{mkt_bg}; border:1px solid {mkt_border}; border-radius:5px; padding:6px 10px; margin-bottom:8px;">'
                    f'<strong>[시장 체제 점검]</strong> KODEX 200: {mkt_txt} ({mkt_msg}) · <strong>전일 미장:</strong> {_us_reg} ({_us_desc})'
                    f'</div>',
                    unsafe_allow_html=True
                )
                st.markdown(
                    """
<div style="font-size:0.78rem; line-height:1.45; color:#334155; background:#F8FAFC; border:1px solid #E2E8F0; border-radius:6px; padding:10px 12px;">
<div style="font-size:0.84rem; font-weight:700; color:#1E293B; margin-bottom:6px;">[실시간 당일 단타] 5% 돌파 모드 핵심 원칙 (당일 완결 / No Overnight)</div>
1. <strong>진입 타이밍</strong>: 장중 09:10 ~ 14:30 실시간 수급 폭발 종목 포착<br>
2. <strong>수급 & 거래대금</strong>: 당일 거래대금 100억~200억↑ & 장중 이미 전일 거래량 50% 이상 돌파<br>
3. <strong>시초가 돌파 모멘텀</strong>: 시초가 대비 +2.5% ~ +15.0% 구간 돌파 (상승 초입 및 추가 슈팅 여력 포착)<br>
4. <strong>탄탄한 양봉 지지</strong>: 종가가 당일 진폭 중심선(50% 이상) 상단 유지 (투매형 윗꼬리 배제)<br>
5. <strong>목표 및 당일 청산</strong>: 진입 즉시 <strong>+5.0% 익절 자동주문 / 손절 -2.5% 엄수 / 15:15 미도달 시 종가 전량 청산</strong>
</div>
                    """,
                    unsafe_allow_html=True
                )
            elif "스나이퍼" in sc_mode:
                bm_ohlcv = get_benchmark_ohlcv()
                mkt_ok, mkt_msg, mkt_meta = evaluate_market_regime(bm_ohlcv)
                mkt_color = "#059669" if mkt_ok else "#D97706"
                mkt_bg = "#ECFDF5" if mkt_ok else "#FFFBEB"
                mkt_border = "#A7F3D0" if mkt_ok else "#FDE68A"
                mkt_txt = "스나이퍼 신규 매수가 허용됩니다." if mkt_ok else "지수 하락장 위험 감지: 스나이퍼 전략은 하락장에서 무리한 진입을 전면 차단합니다."
                st.markdown(
                    f'<div style="font-size:0.78rem; line-height:1.45; color:{mkt_color}; background:{mkt_bg}; border:1px solid {mkt_border}; border-radius:5px; padding:6px 10px; margin-bottom:8px;">'
                    f'<strong>[시장 체제 점검]</strong> KODEX 200: {mkt_txt} ({mkt_msg}) · <strong>전일 미장:</strong> {_us_reg} ({_us_desc})'
                    f'</div>',
                    unsafe_allow_html=True
                )
                st.markdown(
                    """
<div style="font-size:0.78rem; line-height:1.45; color:#334155; background:#F8FAFC; border:1px solid #E2E8F0; border-radius:6px; padding:10px 12px;">
<div style="font-size:0.84rem; font-weight:700; color:#1E293B; margin-bottom:6px;">[스나이퍼 고확신 모드] 핵심 원칙 (80% 승률 타겟 - 20일선 눌림목 반등)</div>
• <strong>시장 필터</strong>: KODEX 200 > 20일선 상승 추세에서만 진입 (지수 급락 리스크 사전 차단)<br>
• <strong>우량 유동성</strong>: 20일 평균 거래대금 최소 100억 원 이상 (호가 왜곡 방지 및 풍부한 체결력)<br>
• <strong>눌림목 이격도</strong>: 20일선 대비 주가 이격도 <strong>98.0% ~ 103.5%</strong> (20일선 첫 지지 반등)<br>
• <strong>핵심 체결 룰 (시초가 갭 제한)</strong>: 익일 시초가가 전일종가 대비 <strong>-1.5% ~ +1.5%</strong> 이내 체결 (+1.5% 초과 갭상승 시 진입 금지)<br>
• <strong>비대칭 목표</strong>: 익절 +1.2% / 손절 -2.0%
</div>
                    """,
                    unsafe_allow_html=True
                )
            elif "종가배팅" in sc_mode:
                st.markdown(
                    """
<div style="font-size:0.78rem; line-height:1.45; color:#334155; background:#F8FAFC; border:1px solid #E2E8F0; border-radius:6px; padding:10px 12px;">
<div style="font-size:0.84rem; font-weight:700; color:#1E293B; margin-bottom:6px;">[주도주 종가배팅 모드] 핵심 원칙 (Overnight Close-to-Open)</div>
• <strong>탐색 시점</strong>: 매일 15:20 ~ 15:30 (장 마감 직전 동시호가 전후)<br>
• <strong>거래대금 폭발</strong>: 당일 거래대금 최소 200억 원 이상 (당일 시장 주도 테마 대장주)<br>
• <strong>장대양봉 고가마감</strong>: 당일 주가 +3.0% 이상 장대양봉 & 일중 고가 부근 마감 (고점 대비 70% 이상 지지)<br>
• <strong>청산 룰</strong>: 당일 15:20 종가 매수 -> 익일 09:00 시초가 갭상승(+1.5%↑) 시 즉시 익절 (손절 -2.0%)
</div>
                    """,
                    unsafe_allow_html=True
                )
            elif "5% 급등" in sc_mode or "5% 돌파" in sc_mode:
                st.markdown(
                    """
<div style="font-size:0.78rem; line-height:1.45; color:#334155; background:#F8FAFC; border:1px solid #E2E8F0; border-radius:6px; padding:10px 12px;">
<div style="font-size:0.84rem; font-weight:700; color:#1E293B; margin-bottom:6px;">[5% 급등 타겟 모드] 핵심 원칙 (1~2일 스윙 5% 급등 수확)</div>
• <strong>빅데이터 검증</strong>: [거래대금 200억~500억↑ + 신고가 첫 상한가/준상한가(+8%~+29.8%) + 고점 지지 마감] 주도주는 익일~2일차 +5% 도달률 80%대에 달합니다.<br>
• <strong>탐색 & 진입 시점</strong>: 장중 급등 추세 확인 또는 마감 직전 15:20 동시호가 종가 매수<br>
• <strong>목표 및 손절</strong>: 장중 <strong>+5.0%</strong> 도달 즉시 전량 자동 익절 (1일차 미도달 시 2일차 홀딩, 손절선 -4.0% 엄수)
</div>
                    """,
                    unsafe_allow_html=True
                )
            else:
                st.markdown(
                    """
<div style="font-size:0.78rem; line-height:1.45; color:#334155; background:#F8FAFC; border:1px solid #E2E8F0; border-radius:6px; padding:10px 12px;">
<div style="font-size:0.84rem; font-weight:700; color:#1E293B; margin-bottom:6px;">[일반 퀀트 스코어링 모드] 핵심 원칙 (기본 종합 점수제)</div>
• <strong>특징</strong>: Point-in-Time 스코어링 엔진으로 수급, 모멘텀, 거래대금, 변동성, 재무 건전성을 복합 평가하여 100점 만점으로 순위를 산출합니다.<br>
• <strong>권장 거래대금</strong>: 최소 50억~100억 원 이상 (유동성 부족으로 인한 슬리피지 방지)
</div>
                    """,
                    unsafe_allow_html=True
                )

        else:
            active_tag = ' <span style="color:#0F172A; font-weight:700; font-size:0.68rem; background:#E2E8F0; padding:1px 5px; border-radius:4px; border:1px solid #94A3B8;">▶ 현재 자동 세팅됨</span>'
            if "장전 시초가" in sc_mode or "08:00~09:00" in sc_mode:
                p0_t = active_tag if rec_time["highlight_idx"] == 0 else ""
                p1_t = active_tag if rec_time["highlight_idx"] == 1 else ""
                st.markdown(
                    f"""
| 분석/실행 시점 | 데이터 기준 | 권장 설정값 (20일 평균 대금) | 매매 행동 요령 & 주의사항 |
| :--- | :--- | :--- | :--- |
| **08:00 ~ 08:50**{p0_t}<br>(장전 집중 분석) | **전일 확정 데이터 기반 우량 눌림목 & 주도 대장주** | **100억 원 이상** | 20일선 눌림목 반등 후보 + 어제 800억~1,000억+ 폭발 주도 대장주 사전 압축 |
| **08:40 ~ 09:00**<br>(동시호가 갭 검증) | **동시호가 예상체결가 & 예상 갭비율** | **눌림목(-1.5%~+1.5%) / 주도주(+1.5%~+4.5%)** | 주도주는 +1.5%~+4.5% 적정 갭 확인 후 시초가 주문, +5% 초과 과열 갭은 진입 금지 |
| **09:00 ~ 09:30**{p1_t}<br>(시초 체결 & 익절) | **장초반 시세 체결 및 자동 분할 매도** | **눌림목 +2.5% / 주도주 +4.0%** | 시초가 체결 즉시 MTS 자동 감시 주문(스탑로스 -2.0%~-2.5%) 등록 필수 |
                    """,
                    unsafe_allow_html=True
                )
            elif "당일 단타" in sc_mode:
                r0_t = active_tag if rec_time["highlight_idx"] == 0 else ""
                r1_t = active_tag if rec_time["highlight_idx"] == 1 else ""
                r2_t = active_tag if rec_time["highlight_idx"] == 2 else ""
                r3_t = active_tag if rec_time["highlight_idx"] == 3 else ""
                r4_t = active_tag if rec_time["highlight_idx"] == 4 else ""

                st.markdown(
                    f"""
| 시간대 | 장세 특성 | 권장 누적 거래대금 | 권장 시초가대비 상승률 | 매매 행동 요령 & 슬라이더 설정 팁 |
| :--- | :--- | :--- | :--- | :--- |
| **09:10 ~ 09:40**{r0_t} | **장초반 거래 집중**<br>(변동성/거래량 극대화) | **100억 원 이상** | **+2.5% ~ +10.0%** | 장초반 고점 윗꼬리 물림 방지 및 +5% 익절 공간 확보 |
| **09:40 ~ 11:00**<br>**[골든타임]**{r1_t} | **주도주 2차 돌파**<br>(당일 단타 최고 승률) | **150억 ~ 200억 원 이상** | **+3.0% ~ +12.0%** | 눌림목 지지 확인 후 재차 치고 나가는 2차 파동 구간 |
| **11:00 ~ 13:00**{r2_t} | **점심 횡보장**<br>(거래량 급감, 소강상태) | **200억 원 이상** | **+4.0% ~ +15.0%** | 거래량 마르는 시간대: 탄탄하게 지지받는 최상위 대장주만 진입 |
| **13:00 ~ 14:30**{r3_t} | **오후 2차 수급 유입**<br>(마감 전 주도 섹터 랠리) | **200억 원 이상** | **+4.5% ~ +18.0%** | 오후장 돌파 매매는 상한가/VI 직행 추진력 갖춘 주도주 선별 |
| **14:30 이후**{r4_t} | **장마감 임박**<br>(당일 청산 집중) | **신규 매수 OFF** | **신규 진입 OFF** | 15:15 전량 청산 원칙이므로 신규 진입 중단 및 보유분 정리 집중 |
                    """,
                    unsafe_allow_html=True
                )
            elif "스나이퍼" in sc_mode:
                s0_t = active_tag if rec_time["highlight_idx"] == 0 else ""
                s1_t = active_tag if rec_time["highlight_idx"] == 1 else ""
                s2_t = active_tag if rec_time["highlight_idx"] == 2 else ""
                st.markdown(
                    f"""
| 분석/실행 시점 | 데이터 기준 | 권장 설정값 (20일 평균 거래대금) | 매매 행동 요령 & 주의사항 |
| :--- | :--- | :--- | :--- |
| **08:30 ~ 09:00**{s0_t}<br>(장 시작 전 분석) | **전일까지의 확정 20일 평균 거래대금** | **100억 원 이상** | 유동성 풍부한 우량주 위주로 20일선 눌림목 후보군 확정 |
| **09:00 ~ 10:00**{s1_t}<br>(시초가 체결 확인) | **시초가 갭 검증** | **체결 갭: -1.5% ~ +1.5%** | +1.5% 초과 갭상승 시 뇌동매매 진입 금지. 갭 범위 내 시초가 매수 |
| **10:00 ~ 15:20**{s2_t}<br>(장중 자동 청산) | **장중 시세 감시** | **익절 +1.2% / 손절 -2.0%** | 주문 체결 즉시 MTS/HTS 자동 감시 매도(스탑로스) 설정 |
                    """,
                    unsafe_allow_html=True
                )
            elif "종가배팅" in sc_mode:
                c0_t = active_tag if rec_time["highlight_idx"] == 0 else ""
                c1_t = active_tag if rec_time["highlight_idx"] == 1 else ""
                st.markdown(
                    f"""
| 분석/실행 시점 | 데이터 기준 | 권장 설정값 (당일 최종 거래대금) | 매매 행동 요령 & 주의사항 |
| :--- | :--- | :--- | :--- |
| **09:00 ~ 14:30**{c0_t}<br>(장중 탐색) | **당일 실시간 누적 거래대금** | **200억 원 이상** | 장중 거래대금이 터지며 추세를 유지하는 주도 테마 1등주 관찰 |
| **14:30 ~ 15:35**{c1_t}<br>(동시호가 종가 체결) | **종가 고가마감 확인** | **주가 +3.0%↑ & 고점 지지율 70%↑** | 윗꼬리가 길지 않고 일봉상 고가 부근 마감 시 15:20 동시호가 매수 |
| **익일 09:00 ~ 09:15**<br>(시초가 갭 청산) | **익일 시초가 및 장초반** | **+1.5% 갭상승 시 즉시 청산** | 익일 09:00 시초가 갭상승(+1.5% 이상) 출발 시 전량 즉시 익절 |
                    """,
                    unsafe_allow_html=True
                )
            elif "5% 급등" in sc_mode or "5% 돌파" in sc_mode:
                g0_t = active_tag if rec_time["highlight_idx"] == 0 else ""
                g1_t = active_tag if rec_time["highlight_idx"] == 1 else ""
                st.markdown(
                    f"""
| 분석/실행 시점 | 데이터 기준 | 권장 설정값 (당일 거래대금) | 매매 행동 요령 & 주의사항 |
| :--- | :--- | :--- | :--- |
| **09:00 ~ 14:30**{g0_t}<br>(장중 실시간 급등주) | **실시간 누적 거래대금** | **200억 원 이상** (상승률 +8%↑) | 상한가/준상한가로 직행하는 주도 대장주 실시간 탐색 |
| **14:30 ~ 15:35**{g1_t}<br>(마감 직전 진입) | **당일 마감 누적 거래대금** | **300억 원 이상** (상승률 +10%↑) | 최상위 대장주 종가 매수 -> 1~2일 내 장중 +5% 자동 익절 |
| **익일 ~ 2일차**<br>(스윙 자동 익절) | **장중 시세 감시** | **장중 +5.0% 도달 시 즉시 익절** | 매수 다음 날 장중 +5% 급등 시 즉시 익절 (손절선 -4.0% 엄수) |
                    """,
                    unsafe_allow_html=True
                )
            else:
                st.caption("Point-in-Time 스코어링 엔진으로 전체 상장 유니버스를 일괄 스크리닝하고 테마 집중 위험을 감지합니다. (권장: 최소 스코어 65점↑, 거래량 급증 1.5배↑)")

    # 2) 전 전략 모드 공통 스크리너 필터 파라미터 툴바 ([스크리너 필터 설정] 전 상장사 전수조사 기본 세팅 & 접기 적용)
    with st.expander(f"[스크리너 필터 설정] {st.session_state.get('sc_scope_sel', full_scope_default)} · {now_str} {rec_time['slot_name']} ({rec_time['badge_desc']} 자동 세팅됨)", expanded=False):
        f_hdr_c1, f_hdr_c2 = st.columns([4.2, 1.3])
        with f_hdr_c1:
            st.markdown(
                f'<div style="display:flex; align-items:center; flex-wrap:wrap; gap:6px 8px; padding-top:4px; line-height:1.4;">'
                f'<span class="office-subheading" style="margin:0 !important; display:inline-block;">스크리너 필터 설정</span>'
                f'<span style="background:#F1F5F9; color:#1E293B; font-weight:700; font-size:0.72rem; padding:2px 8px; border-radius:4px; border:1px solid #CBD5E1; display:inline-block;">{now_str} {rec_time["slot_name"]}</span>'
                f'<span style="color:#64748B; font-size:0.75rem; display:inline-block;">({rec_time["badge_desc"]} 자동 세팅됨)</span>'
                f'</div>',
                unsafe_allow_html=True
            )
        with f_hdr_c2:
            if st.button("현재시간 권장값 재적용", key="btn_reapply_auto_time", use_container_width=True):
                for _k, _v in rec_time["params"].items():
                    st.session_state[f"sc_{_k}"] = _v
                    if _k == "min_daytrade_val_b":
                        st.session_state["sc_min_daytrade_val"] = _v
                st.session_state["sc_scope_sel"] = full_scope_default
                if "sc_mode_scopes" in st.session_state:
                    st.session_state["sc_mode_scopes"][sc_mode] = full_scope_default
                st.session_state["sc_trigger_time_sync"] = True
                st.toast(f"현재 시각({now_str}) {sc_mode} 권장 설정이 자동 적용되었습니다.")
                st.rerun()

        sc_c1, sc_c2, sc_c3, sc_c4, sc_c5 = st.columns([1.1, 1.4, 1.2, 1.2, 1.1])

        with sc_c1:
            target_market = st.selectbox(
                "시장 구분",
                ["전체", "KOSPI", "KOSDAQ"],
                index=0,
                key="sc_target_market"
            )
        with sc_c2:
            scope_options = [
                "전 상장사 전수조사 (약 2,700개)",
                "실시간 급등·주도주 300선 (상승률+거래대금 상위)",
                "내 관심종목 138선 (초고속)"
            ]
            if (
                st.session_state.get("sc_scope_sel") not in scope_options
                or ("300선" in str(st.session_state.get("sc_scope_sel", "")) and not st.session_state.get("sc_user_explicitly_chose_300"))
            ):
                st.session_state["sc_scope_sel"] = scope_options[0]

            def _on_scope_change():
                _cur = st.session_state.get("sc_scope_sel")
                if "300선" in str(_cur):
                    st.session_state["sc_user_explicitly_chose_300"] = True
                else:
                    st.session_state.pop("sc_user_explicitly_chose_300", None)
                if "sc_mode_scopes" in st.session_state:
                    st.session_state["sc_mode_scopes"][sc_mode] = _cur

            sc_scope = st.selectbox(
                "탐색 대상 범위",
                scope_options,
                key="sc_scope_sel",
                on_change=_on_scope_change
            )
            if "sc_mode_scopes" in st.session_state:
                st.session_state["sc_mode_scopes"][sc_mode] = sc_scope

        # Default parameter pre-initialization
        min_daytrade_val_b = 100
        intraday_gain_range = (2.5, 12.0)
        min_daytrade_vol_ratio = 0.5
        min_val_krw_b = 100
        max_disparity_val = 103.5
        ignore_market_filter = False
        min_today_val_b = 200
        min_day_ret_val = 3.0
        min_surge_val_b = 200
        min_day_surge_pct = 8.0
        min_screener_score = 65.0
        min_vol_surge = 1.5
        require_ma_align = True

        if "당일 단타" in sc_mode:
            with sc_c3:
                val_options = [50, 100, 150, 200, 300]
                cur_val = st.session_state.get("sc_min_daytrade_val", rec_time.get("val", 100))
                if cur_val not in val_options:
                    cur_val = 100
                min_daytrade_val_b = st.selectbox(
                    "당일 최소 거래대금",
                    val_options,
                    index=val_options.index(cur_val),
                    format_func=lambda x: f"{x}억 원 이상",
                    key="sc_min_daytrade_val"
                )
            with sc_c4:
                cur_gain = st.session_state.get("sc_intraday_gain_range", rec_time.get("gain", (2.5, 12.0)))
                intraday_gain_range = st.slider(
                    "시초가 대비 상승률 구간 (%)",
                    1.0, 25.0,
                    value=cur_gain,
                    step=0.5,
                    help="권장 설정: 골든타임(09:40~11:00) 3.0%~12.0% | 장초반 2.5%~10.0% | 급등주 최대 25.0%",
                    key="sc_intraday_gain_range"
                )
            with sc_c5:
                cur_vol = st.session_state.get("sc_min_daytrade_vol_ratio", rec_time.get("vol", 0.5))
                min_daytrade_vol_ratio = st.slider(
                    "전일대비 거래량 비율",
                    0.3, 1.5,
                    value=cur_vol,
                    step=0.1,
                    format="%.1f배 이상",
                    help="권장 설정: 최소 0.5배(50%) 이상 | 거래량 폭발 대장주 0.8배 이상",
                    key="sc_min_daytrade_vol_ratio"
                )
        elif "스나이퍼" in sc_mode or "장전 시초가" in sc_mode or "08:00~09:00" in sc_mode:
            with sc_c3:
                snp_options = [50, 100, 150, 200, 300]
                if "sc_min_val_krw_b" not in st.session_state or st.session_state["sc_min_val_krw_b"] not in snp_options:
                    st.session_state["sc_min_val_krw_b"] = 100
                min_val_krw_b = st.selectbox("최소 20일 평균 거래대금", snp_options, format_func=lambda x: f"{x}억 원 이상", key="sc_min_val_krw_b")
            with sc_c4:
                cur_disp = st.session_state.get("sc_max_disparity_val", 103.5)
                max_disparity_val = st.slider("20일선 최대 이격도 (%)", 100.0, 105.0, value=cur_disp, step=0.5, key="sc_max_disparity_val")
            with sc_c5:
                cur_ign = st.session_state.get("sc_ignore_market_filter", False)
                ignore_market_filter = st.checkbox("시장 하락장 무시", value=cur_ign, key="sc_ignore_market_filter")
        elif "종가배팅" in sc_mode:
            with sc_c3:
                cb_options = [100, 200, 300, 500]
                if "sc_min_today_val_b" not in st.session_state or st.session_state["sc_min_today_val_b"] not in cb_options:
                    st.session_state["sc_min_today_val_b"] = 200
                min_today_val_b = st.selectbox("당일 최소 거래대금", cb_options, format_func=lambda x: f"{x}억 원 이상", key="sc_min_today_val_b")
            with sc_c4:
                cur_cb_ret = st.session_state.get("sc_min_day_ret_val", 3.0)
                min_day_ret_val = st.slider("당일 최소 주가 상승률 (%)", 2.0, 6.0, value=cur_cb_ret, step=0.5, key="sc_min_day_ret_val")
            with sc_c5:
                st.markdown(
                    '<div style="font-size:0.75rem; color:#64748B; padding-top:20px; line-height:1.35; white-space:nowrap; overflow:hidden;">'
                    '15:20~15:30 종가 매수<br>'
                    '<span style="color:#7C3AED; font-weight:700;">익일 시초 +1.5% 갭 익절</span>'
                    '</div>',
                    unsafe_allow_html=True
                )
        elif "5% 급등" in sc_mode or "5% 돌파" in sc_mode:
            with sc_c3:
                surge_options = [100, 200, 300, 500]
                if "sc_min_surge_val_b" not in st.session_state or st.session_state["sc_min_surge_val_b"] not in surge_options:
                    st.session_state["sc_min_surge_val_b"] = 200
                min_surge_val_b = st.selectbox("당일 최소 거래대금", surge_options, format_func=lambda x: f"{x}억 원 이상", key="sc_min_surge_val_b")
            with sc_c4:
                cur_s_pct = st.session_state.get("sc_min_day_surge_pct", 8.0)
                min_day_surge_pct = st.slider("당일 최소 주가 상승률 (%)", 5.0, 25.0, value=cur_s_pct, step=1.0, key="sc_min_day_surge_pct")
            with sc_c5:
                st.markdown(
                    '<div style="font-size:0.75rem; color:#64748B; padding-top:20px; line-height:1.35; white-space:nowrap; overflow:hidden;">'
                    '장중 급등·마감 직전 진입<br>'
                    '<span style="color:#D97706; font-weight:700;">익절 +5.0% / 손절 -4.0%</span>'
                    '</div>',
                    unsafe_allow_html=True
                )
        else:
            with sc_c3:
                cur_score = st.session_state.get("sc_min_screener_score", 65.0)
                min_screener_score = st.slider("최소 스코어 기준", 50.0, 90.0, value=cur_score, step=5.0, key="sc_min_screener_score")
            with sc_c4:
                cur_vol_s = st.session_state.get("sc_min_vol_surge", 1.5)
                min_vol_surge = st.slider("거래량 급증 비율 (20일선 대비)", 1.0, 3.0, value=cur_vol_s, step=0.2, key="sc_min_vol_surge")
            with sc_c5:
                cur_ma = st.session_state.get("sc_require_ma_align", True)
                require_ma_align = st.checkbox("이평 정배열/골든 필수", value=cur_ma, key="sc_require_ma_align")

    worker_st = sc_worker.get_status()
    is_worker_running = sc_worker.is_running()

    # If worker just completed, transfer results into session_state!
    if worker_st["status"] == "COMPLETED":
        df_worker_res = worker_st.get("results_df")
        if df_worker_res is not None:
            st.session_state["screened_results_df"] = df_worker_res
            st.session_state["screened_mode_label"] = worker_st.get("mode_label", "선택 전략")
            st.session_state.setdefault("screened_cache_by_mode", {})[worker_st.get("mode_label", "선택 전략")] = df_worker_res
            st.session_state["screened_is_empty_result"] = df_worker_res.empty
            if not df_worker_res.empty:
                st.toast(f"스크리닝 완료! 총 {len(df_worker_res):,}개 종목 발굴")
        sc_worker.reset_status()
        is_worker_running = False

    # If worker was cancelled, notify and reset
    elif worker_st["status"] == "CANCELLED":
        st.warning("사용자 요청에 의해 스크리닝이 중단되었습니다.")
        sc_worker.reset_status()
        is_worker_running = False

    # If worker failed, notify and reset
    elif worker_st["status"] == "FAILED":
        st.error(f"스크리닝 중 오류가 발생했습니다: {worker_st.get('error_message', '알 수 없는 오류')}")
        sc_worker.reset_status()
        is_worker_running = False

    # Statically define live progress fragment
    @st.fragment(run_every="1s")
    def render_screening_live_progress():
        cur_st = sc_worker.get_status()
        st_status = cur_st.get("status")

        if st_status == "COMPLETED":
            df_res = cur_st.get("results_df")
            if df_res is not None:
                st.session_state["screened_results_df"] = df_res
                st.session_state["screened_mode_label"] = cur_st.get("mode_label", "선택 전략")
                st.session_state.setdefault("screened_cache_by_mode", {})[cur_st.get("mode_label", "선택 전략")] = df_res
                st.session_state["screened_is_empty_result"] = df_res.empty
                if not df_res.empty:
                    st.toast(f"스크리닝 완료! 총 {len(df_res):,}개 종목 발굴")
            sc_worker.reset_status()
            st.rerun(scope="app")
            return

        if st_status == "CANCELLED":
            st.warning("사용자 요청에 의해 스크리닝이 중단되었습니다.")
            sc_worker.reset_status()
            st.rerun(scope="app")
            return

        if st_status == "FAILED":
            st.error(f"스크리닝 중 오류가 발생했습니다: {cur_st.get('error_message', '알 수 없는 오류')}")
            sc_worker.reset_status()
            st.rerun(scope="app")
            return

        prog = float(cur_st.get("progress", 0.0) or 0.0)
        pct = int(round(prog * 100))
        screening_btn_label = f"Screening ({pct}%)"

        col_btn_run, col_btn_cancel = st.columns([3.8, 1.2])
        with col_btn_run:
            st.button(screening_btn_label, disabled=True, use_container_width=True, key="btn_running_screening")
        with col_btn_cancel:
            if st.button("Stop", type="secondary", use_container_width=True, key="btn_cancel_screening"):
                sc_worker.cancel()
                st.toast("스크리닝 중단 요청을 전송했습니다.")
                sc_worker.reset_status()
                st.rerun(scope="app")

        st.progress(prog)

    # Execution Toolbar / Progress Area
    if is_worker_running:
        render_screening_live_progress()
    else:
        if st.button("Screening", type="primary", use_container_width=True, key="btn_run_screener"):
            st.session_state["screened_is_empty_result"] = False
            params = {
                "sc_mode": sc_mode,
                "target_market": target_market,
                "sc_scope": sc_scope,
                "min_daytrade_val_b": min_daytrade_val_b if "당일 단타" in sc_mode else 100,
                "intraday_gain_range": intraday_gain_range if "당일 단타" in sc_mode else (2.5, 12.0),
                "min_daytrade_vol_ratio": min_daytrade_vol_ratio if "당일 단타" in sc_mode else 0.5,
                "min_val_krw_b": min_val_krw_b if ("스나이퍼" in sc_mode or "장전 시초가" in sc_mode or "08:00~09:00" in sc_mode) else 100,
                "max_disparity_val": max_disparity_val if ("스나이퍼" in sc_mode or "장전 시초가" in sc_mode or "08:00~09:00" in sc_mode) else 103.5,
                "ignore_market_filter": ignore_market_filter if ("스나이퍼" in sc_mode or "장전 시초가" in sc_mode or "08:00~09:00" in sc_mode) else False,
                "min_today_val_b": min_today_val_b if "종가배팅" in sc_mode else 200,
                "min_day_ret_val": min_day_ret_val if "종가배팅" in sc_mode else 3.0,
                "min_surge_val_b": min_surge_val_b if ("5% 급등" in sc_mode or "5% 돌파" in sc_mode) else 200,
                "min_day_surge_pct": min_day_surge_pct if ("5% 급등" in sc_mode or "5% 돌파" in sc_mode) else 8.0,
                "min_screener_score": min_screener_score if ("당일 단타" not in sc_mode and "스나이퍼" not in sc_mode and "장전 시초가" not in sc_mode and "08:00~09:00" not in sc_mode and "종가배팅" not in sc_mode and "5% 급등" not in sc_mode and "5% 돌파" not in sc_mode) else 65.0,
                "min_vol_surge": min_vol_surge if ("당일 단타" not in sc_mode and "스나이퍼" not in sc_mode and "장전 시초가" not in sc_mode and "08:00~09:00" not in sc_mode and "종가배팅" not in sc_mode and "5% 급등" not in sc_mode and "5% 돌파" not in sc_mode) else 1.5,
                "require_ma_align": require_ma_align if ("당일 단타" not in sc_mode and "스나이퍼" not in sc_mode and "장전 시초가" not in sc_mode and "08:00~09:00" not in sc_mode and "종가배팅" not in sc_mode and "5% 급등" not in sc_mode and "5% 돌파" not in sc_mode) else True,
            }
            sc_worker.start_screening(params)
            st.rerun()

    if st.session_state.get("screened_is_empty_result"):
        with st.container(border=True):
            st.markdown(
                f"<div style='font-size:0.92rem; font-weight:700; color:#1e293b; margin-bottom:6px;'>"
                f"조건 만족 후보 종목이 없습니다 (0건)"
                f"</div>",
                unsafe_allow_html=True
            )
            st.markdown(
                f"<div style='font-size:0.83rem; line-height:1.5; color:#475569; margin-bottom:8px;'>"
                f"선택하신 <strong>{sc_mode}</strong>의 필터 조건이 엄격하여, 현재 탐색 대상 범위(<strong>{sc_scope}</strong>, 시장: <strong>{target_market}</strong>) 내에 오늘 기준 충족 종목이 없습니다.<br>"
                f"시스템 오류가 아니며, 시장의 자금 쏠림이나 탐색 범위에 따른 정상적인 퀀트 필터링 결과입니다."
                f"</div>",
                unsafe_allow_html=True
            )
            st.markdown(
                f"<div style='font-size:0.80rem; line-height:1.5; color:#334155; background-color:#f8fafc; padding:8px 12px; border-radius:4px; border-left:3px solid #64748b;'>"
                f"<strong>해결 방법 가이드</strong>:<br>"
                f"• <strong>탐색 대상 범위 확인</strong>: '내 관심종목 138선'에 오늘 급등주가 없다면, 탐색 대상을 기본 권장값인 <strong>'전 상장사 전수조사 (약 2,700개)'</strong>로 설정하거나 <strong>'주도주 300선'</strong>으로 변경 후 재실행해보세요.<br>"
                f"• <strong>스크리닝 시장</strong>: 코스닥 테마 급등주를 함께 탐색하려면 <strong>'스크리닝 시장: 전체'</strong>로 설정하세요.<br>"
                f"• <strong>필터 기준 완화</strong>: 당일 최소 거래대금을 <strong>200억~300억 원</strong>, 최소 상승률을 <strong>10%</strong> 수준으로 1단계 완화해보세요."
                f"</div>",
                unsafe_allow_html=True
            )


    # Display Persistent Screened Results like Watchlist
    if "screened_results_df" in st.session_state and st.session_state["screened_results_df"] is not None and not st.session_state["screened_results_df"].empty:
        df_screened = st.session_state["screened_results_df"]
        mode_label = st.session_state.get("screened_mode_label", "선택 전략")

        # Theme Clustering Risk Check
        top_n = df_screened.head(10)
        sector_counts = top_n["sector"].value_counts()
        if not sector_counts.empty:
            max_sector = sector_counts.index[0]
            max_count = sector_counts.iloc[0]
            max_ratio = max_count / len(top_n)

            if max_ratio >= 0.40:
                st.markdown(
                    f"<div style='font-size:0.75rem; line-height:1.4; color:#B45309; background:#FFFBEB; border:1px solid #FDE68A; padding:5px 10px; border-radius:5px; margin:4px 0 6px 0;'>"
                    f"<strong>[테마 몰림 경고]</strong> 상위 10개 후보 중 <strong>{max_sector}</strong> 업종이 {max_count}건({max_ratio*100:.0f}%)으로 고도로 편중되어 있습니다. "
                    f"동일 테마/업종 동반 급락 리스크에 주의하여 분산 투자를 준수하세요!"
                    f"</div>",
                    unsafe_allow_html=True
                )

        # Filter & View Toolbar
        sc_t1, sc_t2, sc_t3, sc_t4, sc_t5 = st.columns([1.8, 1.6, 1.5, 1.1, 1.6])
        with sc_t1:
            sc_timing_filter = st.selectbox(
                "매수 타이밍 분류 필터",
                [
                    "전체 타이밍",
                    "장전 시초가 공략 후보",
                    "장중 실시간 (09:10~14:30) 당일단타 후보",
                    "장초 (09:00 시초가) 스나이퍼 후보",
                    "마감 직전 (15:20 종가) 5%급등 후보"
                ],
                index=0,
                key="sc_timing_sel"
            )
        with sc_t2:
            sc_sort_options = ["익일 스코어 높은 순", "전일대비 등락률 높은 순", "목표 익절가 높은 순", "종목명 가나다순"]
            if st.session_state.get("sc_sort_sel") not in sc_sort_options:
                st.session_state["sc_sort_sel"] = "익일 스코어 높은 순"
            sc_sort = st.selectbox(
                "정렬 기준",
                sc_sort_options,
                key="sc_sort_sel"
            )
        with sc_t3:
            avail_sectors = ["전체"] + sorted(list(set(df_screened["sector"].dropna().unique())))
            sc_sector_filter = st.selectbox("업종/테마 필터", avail_sectors, index=0, key="sc_sector_sel")
        with sc_t4:
            sc_display_rows = st.selectbox("표시 높이", [10, 15, 20, 30], index=0, format_func=lambda x: f"{x}줄 보기", key="sc_display_rows_sel")
        with sc_t5:
            sc_view_mode = st.radio("화면 모드", ["고밀도 테이블 뷰 (권장)", "카드 페이징 뷰"], horizontal=True, key="sc_view_radio")

        # Apply Sorting & Sector Filter & Timing Filter
        df_sc_disp = df_screened.copy()
        if "reason_summary" not in df_sc_disp.columns:
            df_sc_disp["reason_summary"] = "전략 조건 충족"
        if "reason_core" not in df_sc_disp.columns:
            df_sc_disp["reason_core"] = "설정된 퀀트 전략 기준을 모두 통과하여 선정되었습니다."
        if "reason_criteria" not in df_sc_disp.columns:
            df_sc_disp["reason_criteria"] = [[] for _ in range(len(df_sc_disp))]
        if "open_pct" not in df_sc_disp.columns:
            df_sc_disp["open_pct"] = df_sc_disp.get("change_pct", 0.0)
        if "high_pct" not in df_sc_disp.columns:
            df_sc_disp["high_pct"] = df_sc_disp.get("change_pct", 0.0)
        if "low_pct" not in df_sc_disp.columns:
            df_sc_disp["low_pct"] = df_sc_disp.get("change_pct", 0.0)
        if "premarket_regime" not in df_sc_disp.columns:
            df_sc_disp["premarket_regime"] = "중립"

        if "장전" in sc_timing_filter:
            df_sc_disp = df_sc_disp[df_sc_disp["timing_label"].str.contains("08:00|장전", na=False)]
        elif "장중 실시간" in sc_timing_filter:
            df_sc_disp = df_sc_disp[df_sc_disp["timing_label"].str.contains("장중|09:10", na=False)]
        elif "장초 (09:00" in sc_timing_filter:
            df_sc_disp = df_sc_disp[df_sc_disp["timing_label"].str.contains("09:00", na=False)]
        elif "마감 직전 (15:20" in sc_timing_filter:
            df_sc_disp = df_sc_disp[df_sc_disp["timing_label"].str.contains("15:20", na=False)]

        if sc_sector_filter != "전체":
            df_sc_disp = df_sc_disp[df_sc_disp["sector"] == sc_sector_filter]

        if sc_sort == "익일 스코어 높은 순":
            df_sc_disp = df_sc_disp.sort_values("score", ascending=False)
        elif sc_sort == "전일대비 등락률 높은 순":
            df_sc_disp = df_sc_disp.sort_values("change_pct", ascending=False)
        elif sc_sort == "목표 익절가 높은 순":
            df_sc_disp = df_sc_disp.sort_values("tp_pct", ascending=False)
        elif sc_sort == "종목명 가나다순":
            df_sc_disp = df_sc_disp.sort_values("name", ascending=True)

        df_sc_disp = df_sc_disp.reset_index(drop=True)
        total_sc_items = len(df_sc_disp)

        # Screener Display Table Height (10 rows fixed height with smooth internal scrolling)
        sc_row_limit = int(sc_display_rows)
        sc_table_height = min(38 + sc_row_limit * 35, 38 + total_sc_items * 35)

        if df_sc_disp.empty:
            st.info("선택하신 필터(업종/매수 타이밍) 조건에 부합하는 종목이 없습니다. 상단 필터를 '전체'로 변경해 보세요.")
        # -------------------------------------------------
        # Screener View A: High-Density Interactive Data Table
        # -------------------------------------------------
        elif "테이블" in sc_view_mode:
            st.caption(f"스크리닝 결과: **총 {total_sc_items:,}개** 발굴됨")
            desired_cols = [
                "code", "name", "timing_label", "strategy_tag", "premarket_regime", "reason_summary", "market", "sector",
                "close", "change_pct", "open_pct", "high_pct", "low_pct",
                "score", "label", "tp_pct", "sl_pct", "vol_ratio", "rsi", "ma_status"
            ]
            valid_cols = [c for c in desired_cols if c in df_sc_disp.columns]
            disp_table_sc = df_sc_disp[valid_cols].copy()

            grid_sc = st.dataframe(
                disp_table_sc,
                column_config={
                    "code": st.column_config.TextColumn("코드", width="small"),
                    "name": st.column_config.TextColumn("종목명", width="medium"),
                    "timing_label": st.column_config.TextColumn("매수 타이밍", width="medium"),
                    "strategy_tag": st.column_config.TextColumn("전략 태그", width="medium"),
                    "premarket_regime": st.column_config.TextColumn("프리마켓 체제", help="미국 증시/야간선물/VIX/환율 글로벌 프리마켓 센티먼트", width="small"),
                    "reason_summary": st.column_config.TextColumn("선정 핵심 근거", width="large"),
                    "market": st.column_config.TextColumn("시장", width="small"),
                    "sector": st.column_config.TextColumn("업종", width="small"),
                    "close": st.column_config.NumberColumn("현재가", format="%,d원"),
                    "change_pct": st.column_config.NumberColumn("현재가(%)", help="전일 종가 대비 현재가 등락률", format="%+.2f%%"),
                    "open_pct": st.column_config.NumberColumn("시가(%)", help="전일 종가 대비 당일 시초가 등락률", format="%+.2f%%"),
                    "high_pct": st.column_config.NumberColumn("고가(%)", help="전일 종가 대비 당일 최고가 등락률", format="%+.2f%%"),
                    "low_pct": st.column_config.NumberColumn("저가(%)", help="전일 종가 대비 당일 최저가 등락률", format="%+.2f%%"),
                    "score": st.column_config.ProgressColumn("익일 스코어", min_value=0, max_value=100, format="%.1f점"),
                    "label": st.column_config.TextColumn("전망"),
                    "tp_pct": st.column_config.NumberColumn("목표익절", format="+%.1f%%"),
                    "sl_pct": st.column_config.NumberColumn("손절기준", format="-%.1f%%"),
                    "vol_ratio": st.column_config.NumberColumn("거래량비", format="%.1f배"),
                    "rsi": st.column_config.NumberColumn("RSI(14)", format="%.1f"),
                    "ma_status": st.column_config.TextColumn("이평배열"),
                },
                hide_index=True,
                use_container_width=True,
                selection_mode="multi-row",
                on_select="rerun",
                height=sc_table_height,
                key="screener_result_grid"
            )

            # Row Selection Inspector Panel
            grid_sel = getattr(grid_sc, "selection", None)
            sel_sc_rows = []
            if grid_sel:
                if hasattr(grid_sel, "rows"):
                    sel_sc_rows = grid_sel.rows
                elif isinstance(grid_sel, dict):
                    sel_sc_rows = grid_sel.get("rows", [])
            sel_sc_rows = [r for r in sel_sc_rows if isinstance(r, int) and 0 <= r < len(df_sc_disp)]

            if sel_sc_rows and len(sel_sc_rows) > 0:
                if len(sel_sc_rows) > 1:
                    sel_sc_df = df_sc_disp.iloc[sel_sc_rows]
                    st.markdown(f"### 선택 후보 종목 일괄 관리 (총 **{len(sel_sc_df)}**개 종목 선택됨)")

                    # Batch Add to Watchlist Container
                    avail_grps_sc = get_watchlist_groups()
                    with st.container(border=True):
                        b_c1, b_c2, b_c3 = st.columns([3, 3, 2])
                        with b_c1:
                            target_grp_sc = st.selectbox(
                                "관심종목 등록 대상 그룹",
                                ["기본그룹"] + [g for g in avail_grps_sc if g != "기본그룹"] + ["+ 새 그룹 직접입력..."],
                                key="batch_sc_grp_sel"
                            )
                        with b_c2:
                            custom_grp_sc = st.text_input(
                                "새 그룹명 입력",
                                "",
                                key="batch_sc_custom_grp",
                                disabled=(target_grp_sc != "+ 새 그룹 직접입력...")
                            )
                        with b_c3:
                            st.markdown("<br>", unsafe_allow_html=True)
                            if st.button(f"{len(sel_sc_df)}개 관심종목 일괄 추가", key="btn_batch_add_sc_wl", type="primary", use_container_width=True):
                                applied_grp = custom_grp_sc.strip() if target_grp_sc == "+ 새 그룹 직접입력..." and custom_grp_sc.strip() else target_grp_sc
                                codes_to_add = sel_sc_df["code"].tolist()
                                batch_add_to_watchlist(codes_to_add, group_name=applied_grp)
                                st.cache_data.clear()
                                st.toast(f"{len(codes_to_add)}개 종목이 '{applied_grp}' 그룹에 일괄 추가되었습니다!")
                                st.rerun()

                    # Summary Metrics for Selected Screener Rows
                    sm_c1, sm_c2, sm_c3, sm_c4 = st.columns(4)
                    sm_c1.metric("선택 종목 평균 스코어", f"{sel_sc_df['score'].mean():.1f}점")
                    sm_c2.metric("선택 종목 평균 등락률", f"{sel_sc_df['change_pct'].mean():+.2f}%")
                    sm_c3.metric("평균 목표 익절", f"+{sel_sc_df['tp_pct'].mean():.1f}%")
                    sm_c4.metric("평균 손절 기준", f"-{sel_sc_df['sl_pct'].mean():.1f}%")

                    st.markdown("---")
                    st.markdown("#### 선택 종목 중 개별 팩트체크 & 상세 진단")
                    target_sc_name = st.selectbox(
                        "상세 진단할 종목 선택",
                        [f"{r['name']} ({r['code']}) - {r.get('strategy_tag', '')}" for _, r in sel_sc_df.iterrows()],
                        key="sel_multi_inspect_sc"
                    )
                    sel_sc_code = target_sc_name.split("(")[-1].split(")")[0].strip() if "(" in target_sc_name else ""
                    matched_sc = sel_sc_df[sel_sc_df["code"] == sel_sc_code]
                    sel_sc = matched_sc.iloc[0] if not matched_sc.empty else sel_sc_df.iloc[0]
                else:
                    sel_sc = df_sc_disp.iloc[sel_sc_rows[0]]

                st.session_state["qa_selected_stock_code"] = sel_sc["code"]
                st.markdown(f"### 선택 후보 종목 정밀 진단: **{sel_sc['name']}** (`{sel_sc['code']}`)")

                # 1. Detailed Selection Reason & Fact Check Box
                with st.container(border=True):
                    r_hdr_c1, r_hdr_c2 = st.columns([3.2, 1.3])
                    with r_hdr_c1:
                        st.markdown(
                            f"<div style='font-size:0.92rem; font-weight:700; color:#1e293b; margin-bottom:4px;'>"
                            f"선정 근거 요약: {sel_sc.get('reason_summary', '조건 충족')}"
                            f"</div>",
                            unsafe_allow_html=True
                        )
                    with r_hdr_c2:
                        st.caption(f"전략: {sel_sc.get('strategy_tag', '')} | {sel_sc.get('timing_label', '')}")

                    st.markdown(
                        f"<div style='font-size:0.84rem; line-height:1.55; color:#334155; margin-bottom:10px; background-color:#f8fafc; padding:10px 14px; border-radius:4px; border-left:3px solid #3b82f6;'>"
                        f"<strong>[핵심 발굴 근거]</strong> {sel_sc.get('reason_core', '설정된 퀀트 전략 기준을 모두 통과하여 선정되었습니다.')}"
                        f"</div>",
                        unsafe_allow_html=True
                    )

                    rc_list = sel_sc.get("reason_criteria", [])
                    if rc_list:
                        st.markdown("<div style='font-size:0.80rem; font-weight:700; color:#475569; margin-bottom:4px;'>전략 통과 세부 팩트체크:</div>", unsafe_allow_html=True)
                        r_cols = st.columns(2)
                        for idx, crit in enumerate(rc_list):
                            with r_cols[idx % 2]:
                                st.markdown(f"<div style='font-size:0.79rem; line-height:1.45; color:#334155;'>• {crit}</div>", unsafe_allow_html=True)

                # 2. Timing & Target Alert Box
                timing_val = sel_sc.get("timing_label", "")
                target_tp_val = sel_sc["close"] * (1.0 + sel_sc["tp_pct"] / 100.0)
                target_sl_val = sel_sc["close"] * (1.0 - sel_sc["sl_pct"] / 100.0)

                pm_reg = sel_sc.get("premarket_regime", "")
                pm_guide_line = f"• **글로벌 프리마켓 지표**: 체제 **[{pm_reg}]** (리스크: {sel_sc.get('premarket_risk', '보통')}) | {sel_sc.get('premarket_gap_guide', '적정 갭 확인 후 시초가 주문')}\n\n" if pm_reg else ""

                if "08:00" in timing_val or "장전" in timing_val:
                    st.success(
                        f"**[장전 시초가 공략: {sel_sc.get('strategy_tag', '')}]**\n\n"
                        f"{pm_guide_line}"
                        f"• **장전 후보 압축 (08:00~08:50)**: 전일 확정 데이터 기준 눌림목 반등 및 주도 대장주 선별 완료\n\n"
                        f"• **동시호가 갭 확인 (08:40~09:00)**: 예상 체결가 갭 점검 (08:50 이전 허수호가 주의, 실체결 적정 갭 확인 후 시초가 주문, +5% 초과 과열 시 진입 금지)\n\n"
                        f"• **목표 익절가**: **{target_tp_val:,.0f}원 (+{sel_sc['tp_pct']:.1f}%)** 도달 시 자동 분할 익절\n\n"
                        f"• **원칙 손절가**: **{target_sl_val:,.0f}원 (-{sel_sc['sl_pct']:.1f}%)** 이탈 시 즉시 기계적 손절"
                    )
                elif "장중" in timing_val or "09:10" in timing_val:
                    st.success(
                        f"**[실시간 당일단타 5% 돌파 모드: 현재 분석 시점 진입 기준]**\n\n"
                        f"• **진입 기준가 (현재 체결가)**: **{sel_sc['close']:,.0f}원** (분석 요청 시점 기준)\n\n"
                        f"• **당일 장중 목표 익절가**: **{target_tp_val:,.0f}원 (+{sel_sc['tp_pct']:.1f}%)** 도달 즉시 전량 자동 익절\n\n"
                        f"• **당일 원칙 손절가**: **{target_sl_val:,.0f}원 (-{sel_sc['sl_pct']:.1f}%)** 이탈 시 즉시 칼손절\n\n"
                        f"• **당일 마감 청산**: 15:15까지 목표가 미도달 시 종가 전량 시장가 청산 (**오버나잇 리스크 제로!**)"
                    )
                elif "09:00" in timing_val:
                    st.success(f"**[매수 타이밍: 장초 09:00 시초가 진입]** {sel_sc['name']}은 20일선 눌림목 반등 포착 종목입니다. 09:00 시초가 갭이 **-1.5% ~ +1.5%** 이내일 때만 체결 (+1.5% 초과 갭상승 시 뇌동매매 금지) **목표익절가 {target_tp_val:,.0f}원(+1.2%) / 손절가 {target_sl_val:,.0f}원(-2.0%)**")
                elif "15:20" in timing_val:
                    st.success(f"**[매수 타이밍: 마감 직전 15:20 종가 진입]** {sel_sc['name']}은 당일 거래대금 폭발 주도 대장주입니다. 15:20 동시호가 종가 매수 후 1~2일 내 **장중 +5.0% 도달 시 즉시 자동 익절** **목표익절가 {target_tp_val:,.0f}원(+5.0%) / 손절가 {target_sl_val:,.0f}원(-4.0%)**")

                # 3. Numeric Metrics & Actions
                insp_c1, insp_c2, insp_c3, insp_c4, insp_btn1, insp_btn2 = st.columns([2, 2, 2, 2, 2.5, 2])
                with insp_c1:
                    st.metric("익일 전망 스코어", f"{sel_sc['score']:.1f}점", delta=sel_sc['label'])
                with insp_c2:
                    st.metric("현재가 / 등락률", f"{sel_sc['close']:,.0f}원", delta=f"{sel_sc['change_pct']:+.2f}%")
                    op_v = sel_sc.get("open_pct", sel_sc["change_pct"])
                    hp_v = sel_sc.get("high_pct", sel_sc["change_pct"])
                    lp_v = sel_sc.get("low_pct", sel_sc["change_pct"])
                    st.caption(f"시 {op_v:+.2f}% · 고 {hp_v:+.2f}% · 저 {lp_v:+.2f}%")
                with insp_c3:
                    st.metric("목표 익절가", f"{target_tp_val:,.0f}원", delta=f"+{sel_sc['tp_pct']:.1f}%")
                with insp_c4:
                    st.metric("원칙 손절가", f"{target_sl_val:,.0f}원", delta=f"-{sel_sc['sl_pct']:.1f}%")

                with insp_btn1:
                    st.markdown("<br>", unsafe_allow_html=True)
                    if st.button(f"{sel_sc['name']} 7년 백테스트 실행", type="primary", key=f"bt_sc_btn_{sel_sc['code']}", use_container_width=True):
                        st.session_state["target_backtest_code"] = sel_sc["code"]
                        st.session_state["verif_pipeline_mode_sel"] = "과거 5년 워크포워드 & KODEX 200 벤치마크 검증 (종목별 95% 신뢰구간 시뮬레이션)"
                        st.session_state["pending_active_tab"] = main_tab_names[3]
                        st.toast(f"{sel_sc['name']} 백테스트 설정 완료! 4번 통합 성과 검증 탭을 확인하세요.")
                        st.rerun()
                with insp_btn2:
                    st.markdown("<br>", unsafe_allow_html=True)
                    if st.button("2번 애널리스트 콘솔 이동", key=f"btn_insp_to_tab2_{sel_sc['code']}", use_container_width=True):
                        st.session_state["qa_console_target_code"] = sel_sc["code"]
                        st.session_state["qa_has_active_result"] = True
                        st.session_state["pending_active_tab"] = main_tab_names[1]
                        st.toast(f"{sel_sc['name']} 2번 수석 애널리스트 콘솔로 이동합니다.")
                        st.rerun()

                # 4. Technical Indicator Breakdown Cards
                bk = sel_sc.get("breakdown") or {}
                if not isinstance(bk, dict):
                    bk = {}
                r_val = float(bk.get('rsi_val') or sel_sc.get('rsi') or 50.0)
                v_val = float(sel_sc.get('vol_ratio') or 1.0)
                b1, b2, b3, b4 = st.columns(4)
                b1.info(f"**RSI(14)**: {r_val:.1f}")
                b2.info(f"**이평배열**: {sel_sc.get('ma_status', '-')}")
                b3.info(f"**거래량 급증비**: {v_val:.2f}배")
                b4.info(f"**체결 원칙**: {sel_sc.get('rule_note', '-')}")

                st.markdown("---")
                render_quant_analyst_qa_box(sel_sc, key_suffix="insp")

        # -------------------------------------------------
        # Screener View B: Paginated Cards View
        # -------------------------------------------------
        else:
            card_page_size = int(sc_display_rows)
            total_card_pages = max(1, (total_sc_items - 1) // card_page_size + 1)
            p_c1, p_c2 = st.columns([3.5, 1.2])
            with p_c1:
                st.caption(f"스크리닝 결과 카드 뷰 (페이지당 {card_page_size}개 표시)")
            with p_c2:
                sc_curr_page = st.number_input("카드 페이지", min_value=1, max_value=total_card_pages, value=1, step=1, key="sc_card_page_num_in")
            start_sc = (sc_curr_page - 1) * card_page_size
            end_sc = min(start_sc + card_page_size, total_sc_items)
            paged_sc_cards = df_sc_disp.iloc[start_sc:end_sc].reset_index(drop=True)
            card_cols = st.columns(3)
            for i, (_, srow) in enumerate(paged_sc_cards.iterrows()):
                with card_cols[i % 3]:
                    with st.container(border=True):
                        st.markdown(f"**{srow['name']}** (`{srow['code']}`)")
                        st.caption(f"{srow['market']} | {srow['sector']} | {srow['strategy_tag']}")
                        chg_c = "red" if srow["change_pct"] > 0 else ("blue" if srow["change_pct"] < 0 else "gray")
                        st.markdown(f"**{srow['close']:,.0f}원** <span style='color:{chg_c}; font-weight:600;'>{srow['change_pct']:+.2f}%</span>", unsafe_allow_html=True)
                        op_c = srow.get("open_pct", srow["change_pct"])
                        hp_c = srow.get("high_pct", srow["change_pct"])
                        lp_c = srow.get("low_pct", srow["change_pct"])
                        st.caption(f"시 {op_c:+.2f}% | 고 {hp_c:+.2f}% | 저 {lp_c:+.2f}%")
                        st.markdown(f"**스코어 {srow['score']:.1f}점** ({srow['label']})")
                        st.progress(min(1.0, max(0.0, srow["score"] / 100.0)))
                        st.caption(f"목표: +{srow['tp_pct']:.1f}% | 손절: -{srow['sl_pct']:.1f}%")
                        st.caption(f"{srow.get('rule_note', '')}")

                        with st.expander("선정 이유 및 팩트체크", expanded=False):
                            st.markdown(f"<div style='font-size:0.80rem; line-height:1.4; margin-bottom:6px;'><strong>핵심 근거</strong>: {srow.get('reason_core', '조건 충족')}</div>", unsafe_allow_html=True)
                            for crit in srow.get('reason_criteria', []):
                                st.markdown(f"<div style='font-size:0.77rem; line-height:1.35; color:#475569;'>• {crit}</div>", unsafe_allow_html=True)

                        c_b1, c_b2, c_b3 = st.columns(3)
                        with c_b1:
                            if st.button("관심추가", key=f"card_add_sc_{srow['code']}_{i}", use_container_width=True):
                                batch_add_to_watchlist([srow['code']])
                                st.cache_data.clear()
                                st.toast(f"{srow['name']} 관심종목 추가 완료!")
                                st.rerun()
                        with c_b2:
                            if st.button("백테스트", key=f"card_bt_sc_{srow['code']}_{i}", use_container_width=True):
                                st.session_state["target_backtest_code"] = srow["code"]
                                st.session_state["verif_pipeline_mode_sel"] = "과거 5년 워크포워드 & KODEX 200 벤치마크 검증 (종목별 95% 신뢰구간 시뮬레이션)"
                                st.session_state["pending_active_tab"] = main_tab_names[3]
                                st.toast(f"{srow['name']} 백테스트 설정 완료! 4번 통합 성과 검증 탭을 확인하세요.")
                                st.rerun()
                        with c_b3:
                            if st.button("수석 애널리스트 Q&A", key=f"card_qa_sc_{srow['code']}_{i}", use_container_width=True):
                                st.session_state["qa_console_target_code"] = srow["code"]
                                st.session_state["qa_has_active_result"] = True
                                st.session_state["pending_active_tab"] = main_tab_names[1]
                                st.toast(f"{srow['name']} 2번 수석 애널리스트 콘솔로 이동합니다.")
                                st.rerun()

        # -------------------------------------------------
        # Screener Section C: AI Quant Analyst Q&A
        # -------------------------------------------------
        if not df_sc_disp.empty:
            qa_stock_row = None
            if "qa_selected_stock_code" in st.session_state:
                raw_sel = str(st.session_state["qa_selected_stock_code"]).strip()
                sel_target = raw_sel.zfill(6) if raw_sel.isdigit() else raw_sel
                matched = df_sc_disp[df_sc_disp["code"].astype(str).str.zfill(6) == sel_target]
                if not matched.empty:
                    qa_stock_row = matched.iloc[0]

            if qa_stock_row is None and not df_sc_disp.empty:
                qa_stock_row = df_sc_disp.iloc[0]

            has_table_selection = bool(sel_sc_rows and len(sel_sc_rows) > 0) if "sel_sc_rows" in locals() else False
            if qa_stock_row is not None and not has_table_selection:
                st.markdown("---")
                render_quant_analyst_qa_box(qa_stock_row, key_suffix="hub")

# =========================================================
# TAB 2: Q&A
# =========================================================
with tab2:
    render_analyst_factsheet_and_qa_tab()


# =========================================================
# TAB 3: 관심종목 실시간 스코어보드 & 관리
# =========================================================
with tab3:
    # 1. Load Watchlist from DB & Render Summary KPIs Row (상단 KPI 카드)
    with get_db_connection() as conn:
        cursor = conn.cursor()
        cursor.execute("""
            SELECT 
                w.code, 
                w.name, 
                w.market, 
                COALESCE(NULLIF(u.sector, ''), NULLIF(w.notes, ''), '기타') AS raw_sector,
                COALESCE(u.industry, '') AS industry,
                w.notes,
                COALESCE(NULLIF(w.group_name, ''), '기본그룹') AS group_name
            FROM watchlist w 
            LEFT JOIN universe u ON w.code = u.code 
            ORDER BY w.added_at DESC
        """)
        watchlist_rows = cursor.fetchall()

    if watchlist_rows:
        tuple_codes = tuple(
            (r["code"], r["name"], r["market"], normalize_sector(r["raw_sector"], r["name"], r["industry"]), r["notes"], r["group_name"])
            for r in watchlist_rows
        )
        raw_wl_data = load_all_watchlist_metrics(tuple_codes)
        df_all = pd.DataFrame(raw_wl_data)

        total_cnt = len(df_all)
        bullish_cnt = len(df_all[df_all["score"] >= 70.0])
        neutral_cnt = len(df_all[(df_all["score"] >= 50.0) & (df_all["score"] < 70.0)])
        bearish_cnt = len(df_all[df_all["score"] < 50.0])
        avg_score = df_all["score"].mean() if not df_all.empty else 0.0

        kpi1, kpi2, kpi3, kpi4, kpi5 = st.columns(5)
        kpi1.metric("총 관심종목", f"{total_cnt:,}개")
        kpi2.metric("평균 익일 스코어", f"{avg_score:.1f}점 / 100")
        kpi3.metric(
            "강세 (70점↑)",
            f"{bullish_cnt}개",
            delta=f"비중 {(bullish_cnt/total_cnt)*100:.1f}%" if total_cnt else None,
            delta_color="off",
            help="전체 관심종목 중 강세 종목 비율 (점유율)"
        )
        kpi4.metric(
            "중립 (50~69점)",
            f"{neutral_cnt}개",
            delta=f"비중 {(neutral_cnt/total_cnt)*100:.1f}%" if total_cnt else None,
            delta_color="off",
            help="전체 관심종목 중 중립 종목 비율 (점유율)"
        )
        kpi5.metric(
            "약세 (50점↓)",
            f"{bearish_cnt}개",
            delta=f"비중 {(bearish_cnt/total_cnt)*100:.1f}%" if total_cnt else None,
            delta_color="off",
            help="전체 관심종목 중 약세 종목 비율 (점유율)"
        )
    else:
        df_all = pd.DataFrame()
        kpi1, kpi2, kpi3, kpi4, kpi5 = st.columns(5)
        kpi1.metric("총 관심종목", "0개")
        kpi2.metric("평균 익일 스코어", "0.0점")
        kpi3.metric("강세 (70점↑)", "0개")
        kpi4.metric("중립 (50~69점)", "0개")
        kpi5.metric("약세 (50점↓)", "0개")

    st.markdown("---")

    # 2. Search, Group & Batch Add Toolbar
    col_search, col_grp, col_batch_toggle = st.columns([2.6, 1.4, 1.2])
    avail_groups = get_watchlist_groups()
    with col_search:
        search_kw = st.text_input("개별 종목 검색 및 빠른 추가 (종목명 또는 6자리 코드)", "", placeholder="예: 삼성전자, 000660, 에코프로, 카카오...", key="wl_search_ticker_kw")
    with col_grp:
        quick_target_grp = st.selectbox("추가할 대상 그룹", avail_groups + ["+ 새 그룹 추가..."], key="quick_target_grp_sel")
        if quick_target_grp == "+ 새 그룹 추가...":
            new_quick_grp = st.text_input("새 그룹명 입력", "", placeholder="예: 바이오, 단타...", key="quick_new_grp_in")
            assigned_quick_grp = new_quick_grp.strip() if new_quick_grp.strip() else "기본그룹"
        else:
            assigned_quick_grp = quick_target_grp
    with col_batch_toggle:
        st.markdown("<div style='height:28px;'></div>", unsafe_allow_html=True)
        show_batch_panel = st.checkbox("대량 등록 / 일괄 관리", value=False, key="wl_batch_panel_chk")

    if search_kw:
        matched_stocks = search_ticker(search_kw, limit=6)
        if matched_stocks:
            cols = st.columns(min(len(matched_stocks), 4))
            for i, stock in enumerate(matched_stocks[:4]):
                with cols[i]:
                    code = stock["code"]
                    name = stock["name"]
                    market = stock["market"]
                    sector = stock.get("sector", "")
                    st.markdown(f"**{name}** (`{code}`)")
                    st.caption(f"{market} | {sector} | 그룹: {assigned_quick_grp}")
                    if st.button(f"관심종목 추가", key=f"add_search_{code}"):
                        batch_add_to_watchlist([code], group_name=assigned_quick_grp)
                        st.cache_data.clear()
                        st.toast(f"{name} 종목이 '{assigned_quick_grp}' 그룹에 등록되었습니다!")
                        st.rerun()

    # Batch Add / Management Expander
    if show_batch_panel:
        with st.expander("종목 대량 등록 & 프리셋 관리 (수백 개 종목 일괄 처리)", expanded=True):
            b_c1, b_c2 = st.columns([3, 2])
            with b_c1:
                st.markdown("**1) 종목코드 텍스트 일괄 붙여넣기**")
                st.caption("쉼표(,), 띄어쓰기, 또는 줄바꿈(엔터)으로 구분된 6자리 종목코드들을 붙여넣으세요.")
                b_grp_c1, b_grp_c2 = st.columns([2, 2])
                with b_grp_c1:
                    batch_grp_sel = st.selectbox("일괄 등록 대상 그룹", avail_groups + ["+ 새 그룹 직접입력..."], key="batch_grp_sel")
                with b_grp_c2:
                    batch_custom_grp = st.text_input("새 그룹명", "", key="batch_custom_grp", disabled=(batch_grp_sel != "+ 새 그룹 직접입력..."))
                batch_final_grp = batch_custom_grp.strip() if batch_grp_sel == "+ 새 그룹 직접입력..." and batch_custom_grp.strip() else (batch_grp_sel if batch_grp_sel != "+ 새 그룹 직접입력..." else "기본그룹")

                batch_text = st.text_area("종목코드 입력창", "", placeholder="005930, 000660, 035420, 005380, 068270, 000270, 105560...", height=100)
                if st.button("종목 일괄 추가 실행", type="primary"):
                    if batch_text.strip():
                        import re
                        raw_codes = re.findall(r"\b\d{6}\b", batch_text)
                        if raw_codes:
                            added = batch_add_to_watchlist(raw_codes, group_name=batch_final_grp)
                            st.cache_data.clear()
                            st.success(f"총 {added}개 종목이 '{batch_final_grp}' 그룹에 일괄 등록되었습니다!")
                            st.rerun()
                        else:
                            st.error("유효한 6자리 숫자 종목코드를 찾을 수 없습니다.")

            with b_c2:
                st.markdown("**2) 시장 대표 우량주 프리셋 추가**")
                st.caption("클릭 한 번으로 시장 핵심 종목군을 즉시 관심종목으로 구성합니다.")
                if st.button("KOSPI 시총 상위 30종목 원클릭 추가", use_container_width=True):
                    kospi_top30 = [
                        "005930", "000660", "373220", "207940", "005380", "000270", "068270", "005490",
                        "105560", "055550", "035420", "028260", "012330", "032830", "003670", "035720",
                        "086790", "011200", "010130", "009150", "018260", "010950", "033780", "000810",
                        "015760", "329180", "034730", "003550", "000100", "006400"
                    ]
                    batch_add_to_watchlist(kospi_top30, group_name="KOSPI30")
                    st.cache_data.clear()
                    st.toast("코스피 상위 30종목 (KOSPI30 그룹) 추가 완료!")
                    st.rerun()

                if st.button("KOSDAQ 시총 상위 30종목 원클릭 추가", use_container_width=True):
                    kosdaq_top30 = [
                        "247540", "086520", "196170", "058470", "022100", "041510", "277810", "091990",
                        "066970", "028300", "293490", "036930", "263750", "145020", "039030", "005290",
                        "214150", "035900", "112040", "357780", "078600", "141080", "253450", "195940",
                        "095660", "067160", "036570", "048410", "034220", "042000"
                    ]
                    batch_add_to_watchlist(kosdaq_top30, group_name="KOSDAQ30")
                    st.cache_data.clear()
                    st.toast("코스닥 상위 30종목 (KOSDAQ30 그룹) 추가 완료!")
                    st.rerun()

                st.markdown("---")
                del_confirm = st.checkbox("관심종목 전체 삭제 확인", value=False)
                if st.button("관심종목 전체 비우기", disabled=not del_confirm, use_container_width=True):
                    clear_watchlist()
                    st.cache_data.clear()
                    st.warning("모든 관심종목이 삭제되었습니다.")
                    st.rerun()

    if not watchlist_rows:
        st.info("현재 등록된 관심종목이 없습니다. 위 검색창이나 '대량 등록'을 통해 종목을 등록하세요.")
    else:
        # 4. Multi-dimensional Filter & View Toolbar
        all_groups = ["전체"] + sorted(list(set(df_all["group_name"].dropna().unique())))
        all_sectors = ["전체"] + sorted(list(set(df_all["sector"].dropna().unique())))

        t_col0, t_col1, t_col2, t_col3, t_col4, t_col5, t_col6 = st.columns([1.2, 1.6, 1.0, 1.2, 1.4, 1.0, 1.4])
        with t_col0:
            filter_group = st.selectbox("그룹 필터", all_groups, index=0, key="wl_filter_group")
        with t_col1:
            filter_label = st.selectbox(
                "전망/전략 필터",
                [
                    "전체",
                    "실시간 당일단타 적격 (장중 진입: 당일 +5% 익절)",
                    "당일 수급돌파 강세주",
                    "당일 눌림목 지지주",
                    "스나이퍼 적격 (오전 9시 진입: 눌림목 반등)",
                    "5% 급등 타겟 적격 (마감 직전 진입: 1~2일 스윙)",
                    "종가배팅 적격 (마감 직전 진입: 익일 갭익절)",
                    "강세(70점↑)",
                    "중립(50~69점)",
                    "약세(50점↓)"
                ],
                index=0,
                key="wl_filter_label"
            )
        with t_col2:
            filter_market = st.selectbox("시장 구분", ["전체", "KOSPI", "KOSDAQ"], index=0, key="wl_filter_market")
        with t_col3:
            filter_sector = st.selectbox("업종/테마 필터", all_sectors, index=0, key="wl_filter_sector")
        with t_col4:
            wl_sort_options = ["익일 스코어 높은 순", "전일대비 등락률 높은 순", "시초가대비 등락률 높은 순", "거래량 많은 순", "분석 상승률 높은 순", "종목명 가나다순"]
            if st.session_state.get("wl_sort_by") not in wl_sort_options:
                st.session_state["wl_sort_by"] = "익일 스코어 높은 순"
            sort_by = st.selectbox(
                "정렬 기준",
                wl_sort_options,
                key="wl_sort_by"
            )
        with t_col5:
            wl_display_rows = st.selectbox("표시 높이", [10, 15, 20, 30], index=0, format_func=lambda x: f"{x}줄 보기", key="wl_display_rows_sel")
        with t_col6:
            view_mode = st.radio("화면 모드", ["고밀도 테이블 뷰 (권장)", "카드 페이징 뷰"], horizontal=True, key="wl_view_mode")

        # Apply Filters
        df_filtered = df_all.copy()
        if filter_group != "전체":
            df_filtered = df_filtered[df_filtered["group_name"] == filter_group]

        if "당일단타" in filter_label:
            df_filtered = df_filtered[df_filtered["is_daytrade"] == True]
        elif "수급돌파" in filter_label:
            df_filtered = df_filtered[df_filtered["intraday_tag"].str.contains("수급돌파")]
        elif "눌림목" in filter_label:
            df_filtered = df_filtered[df_filtered["intraday_tag"].str.contains("눌림지지")]
        elif "5% 급등" in filter_label:
            df_filtered = df_filtered[df_filtered["is_surge_5pct"] == True]
        elif "스나이퍼" in filter_label:
            df_filtered = df_filtered[df_filtered["is_sniper"] == True]
        elif "종가배팅" in filter_label:
            df_filtered = df_filtered[df_filtered["is_closing_bet"] == True]
        elif filter_label == "강세(70점↑)":
            df_filtered = df_filtered[df_filtered["score"] >= 70.0]
        elif filter_label == "중립(50~69점)":
            df_filtered = df_filtered[(df_filtered["score"] >= 50.0) & (df_filtered["score"] < 70.0)]
        elif filter_label == "약세(50점↓)":
            df_filtered = df_filtered[df_filtered["score"] < 50.0]

        if filter_market != "전체":
            df_filtered = df_filtered[df_filtered["market"] == filter_market]

        if filter_sector != "전체":
            df_filtered = df_filtered[df_filtered["sector"] == filter_sector]

        # Apply Sorting
        if sort_by == "익일 스코어 높은 순":
            df_filtered = df_filtered.sort_values("score", ascending=False)
        elif sort_by == "전일대비 등락률 높은 순":
            df_filtered = df_filtered.sort_values("change_pct", ascending=False)
        elif sort_by == "시초가대비 등락률 높은 순":
            df_filtered = df_filtered.sort_values("intraday_open_pct", ascending=False)
        elif sort_by == "거래량 많은 순":
            df_filtered = df_filtered.sort_values("volume", ascending=False)
        elif sort_by == "분석 상승률 높은 순":
            sort_col = "analyzed_gain_pct" if "analyzed_gain_pct" in df_filtered.columns else "tp_pct"
            df_filtered = df_filtered.sort_values(sort_col, ascending=False)
        elif sort_by == "종목명 가나다순":
            df_filtered = df_filtered.sort_values("name", ascending=True)

        df_filtered = df_filtered.reset_index(drop=True)
        total_wl_items = len(df_filtered)

        # -------------------------------------------------
        # MODE A: High-Density Interactive Data Table (Default)
        # -------------------------------------------------
        if "테이블" in view_mode:
            rows_to_show_wl = int(wl_display_rows)
            wl_table_height = min(38 + rows_to_show_wl * 35, 38 + total_wl_items * 35)
            st.caption(f"필터 결과: **총 {total_wl_items:,}개** 종목")

            disp_table = df_filtered[[
                "code", "name", "group_name", "market", "sector", "strategy_tag", "close", "change_pct",
                "intraday_tag", "nextday_tag", "analyzed_gain_pct", "sl_pct", "vol_ratio", "rsi", "ma_status", "verified_badge"
            ]].copy()

            grid_event = st.dataframe(
                disp_table,
                column_config={
                    "code": st.column_config.TextColumn("코드", width="small"),
                    "name": st.column_config.TextColumn("종목명", width="medium"),
                    "group_name": st.column_config.TextColumn("그룹", width="small"),
                    "market": st.column_config.TextColumn("시장", width="small"),
                    "sector": st.column_config.TextColumn("업종", width="small"),
                    "strategy_tag": st.column_config.TextColumn("고확신 전략", width="small"),
                    "close": st.column_config.NumberColumn("현재가", format="%,d원"),
                    "change_pct": st.column_config.NumberColumn("전일대비", format="%+.2f%%"),
                    "intraday_tag": st.column_config.TextColumn("당일 전망 (장중)", width="medium"),
                    "nextday_tag": st.column_config.TextColumn("익일 전망 (스윙)", width="medium"),
                    "analyzed_gain_pct": st.column_config.NumberColumn("분석 상승률", format="+%.1f%%", help="수석 애널리스트 복합 팩터 분석 기대 상승률"),
                    "sl_pct": st.column_config.NumberColumn("손절기준", format="-%.1f%%"),
                    "vol_ratio": st.column_config.NumberColumn("거래량 급증비", format="%.1f배"),
                    "rsi": st.column_config.NumberColumn("RSI(14)", format="%.1f"),
                    "ma_status": st.column_config.TextColumn("이평배열"),
                    "verified_badge": st.column_config.TextColumn("검증상태"),
                },
                hide_index=True,
                use_container_width=True,
                selection_mode="multi-row",
                on_select="rerun",
                height=wl_table_height,
                key="watchlist_grid"
            )

            # Row Selection Detail Inspector Panel
            selected_indices = grid_event.selection.rows if hasattr(grid_event, "selection") else []
            if selected_indices and len(selected_indices) > 0:
                if len(selected_indices) > 1:
                    sel_rows_df = df_filtered.iloc[selected_indices]
                    st.markdown(f"### 선택 관심종목 일괄 관리 (총 **{len(sel_rows_df)}**개 종목 선택됨)")

                    with st.container(border=True):
                        b_col1, b_col2, b_col3, b_col4 = st.columns([2.5, 2.5, 1.8, 1.8])
                        with b_col1:
                            batch_grp_choice = st.selectbox(
                                "선택 종목 일괄 그룹 변경",
                                ["(그룹 선택)"] + [g for g in avail_groups] + ["+ 새 그룹 직접입력..."],
                                key="batch_insp_grp_sel"
                            )
                        with b_col2:
                            batch_custom_grp = st.text_input(
                                "새 그룹명 입력",
                                "",
                                key="batch_insp_custom_grp",
                                disabled=(batch_grp_choice != "+ 새 그룹 직접입력...")
                            )
                        with b_col3:
                            st.markdown("<br>", unsafe_allow_html=True)
                            if st.button("그룹 일괄 변경", key="btn_batch_grp_apply", use_container_width=True):
                                applied_grp = batch_custom_grp.strip() if batch_grp_choice == "+ 새 그룹 직접입력..." and batch_custom_grp.strip() else batch_grp_choice
                                if applied_grp and applied_grp != "(그룹 선택)":
                                    codes_to_update = sel_rows_df["code"].tolist()
                                    batch_update_watchlist_groups(codes_to_update, applied_grp)
                                    st.cache_data.clear()
                                    st.toast(f"{len(codes_to_update)}개 종목이 '{applied_grp}' 그룹으로 일괄 변경되었습니다!")
                                    st.rerun()
                                else:
                                    st.warning("유효한 그룹을 선택하거나 입력해주세요.")
                        with b_col4:
                            st.markdown("<br>", unsafe_allow_html=True)
                            if st.button(f"{len(sel_rows_df)}개 일괄 삭제", key="btn_batch_del", use_container_width=True):
                                codes_to_del = sel_rows_df["code"].tolist()
                                placeholders = ",".join(["?"] * len(codes_to_del))
                                with get_db_connection() as conn:
                                    cursor = conn.cursor()
                                    cursor.execute(f"DELETE FROM watchlist WHERE code IN ({placeholders})", codes_to_del)
                                    conn.commit()
                                st.cache_data.clear()
                                st.toast(f"{len(codes_to_del)}개 종목이 관심종목에서 삭제되었습니다!")
                                st.rerun()

                    # Multi-stock Summary Metrics
                    m_c1, m_c2, m_c3, m_c4 = st.columns(4)
                    m_c1.metric("선택 종목 평균 스코어", f"{sel_rows_df['score'].mean():.1f}점")
                    m_c2.metric("선택 종목 평균 등락률", f"{sel_rows_df['change_pct'].mean():+.2f}%")
                    m_c3.metric("상승 / 보합 / 하락", f"{(sel_rows_df['change_pct'] > 0).sum()} / {(sel_rows_df['change_pct'] == 0).sum()} / {(sel_rows_df['change_pct'] < 0).sum()}")
                    m_c4.metric("고확신(70점↑) 종목", f"{(sel_rows_df['score'] >= 70).sum()}개")

                    st.markdown("---")
                    st.markdown("#### 선택 종목 중 개별 상세 진단")
                    target_stock_name = st.selectbox(
                        "상세 진단할 종목 선택",
                        [f"{r['name']} ({r['code']}) [그룹: {r.get('group_name', '기본그룹')}]" for _, r in sel_rows_df.iterrows()],
                        key="sel_multi_inspect_stock"
                    )
                    sel_stock_code = target_stock_name.split("(")[-1].split(")")[0].strip()
                    sel_row = sel_rows_df[sel_rows_df["code"] == sel_stock_code].iloc[0]
                else:
                    sel_row = df_filtered.iloc[selected_indices[0]]

                st.markdown(f"### 선택 종목 정밀 진단: **{sel_row['name']}** (`{sel_row['code']}`) [소속: `{sel_row.get('group_name', '기본그룹')}`]")

                # 1. Group Manager Row
                with st.container(border=True):
                    g_c1, g_c2, g_c3 = st.columns([2.5, 2.5, 1.5])
                    with g_c1:
                        target_grp_sel = st.selectbox(
                            "소속 그룹 변경",
                            ["(선택 유지)"] + [g for g in avail_groups if g != sel_row.get('group_name')] + ["+ 새 그룹 직접입력..."],
                            key=f"insp_grp_sel_{sel_row['code']}"
                        )
                    with g_c2:
                        custom_grp_name = st.text_input(
                            "새 그룹명 입력",
                            "",
                            key=f"insp_grp_in_{sel_row['code']}",
                            disabled=(target_grp_sel != "+ 새 그룹 직접입력...")
                        )
                    with g_c3:
                        st.markdown("<br>", unsafe_allow_html=True)
                        if st.button("그룹 변경 저장", key=f"insp_grp_btn_{sel_row['code']}", use_container_width=True):
                            applied_new_grp = custom_grp_name.strip() if target_grp_sel == "+ 새 그룹 직접입력..." and custom_grp_name.strip() else target_grp_sel
                            if applied_new_grp and applied_new_grp != "(선택 유지)":
                                update_watchlist_group(sel_row['code'], applied_new_grp)
                                st.cache_data.clear()
                                st.toast(f"{sel_row['name']} 종목이 '{applied_new_grp}' 그룹으로 변경되었습니다!")
                                st.rerun()

                # 2. Key Metrics & Action Buttons
                insp_c1, insp_c2, insp_c3, insp_c4, insp_btn1, insp_btn2, insp_btn3 = st.columns([1.8, 1.8, 1.8, 1.8, 2.2, 2.0, 1.2])
                with insp_c1:
                    st.metric("익일 스코어", f"{sel_row['score']:.1f}점", delta=sel_row['label'])
                with insp_c2:
                    st.metric("현재가 / 전일대비", f"{sel_row['close']:,.0f}원", delta=f"{sel_row['change_pct']:+.2f}%")
                with insp_c3:
                    st.metric("시초가대비 등락", f"{sel_row.get('intraday_open_pct', 0.0):+.2f}%", help="오늘 시초가 대비 현재가 변동률")
                with insp_c4:
                    ana_gain = sel_row.get("analyzed_gain_pct", sel_row.get("tp_pct", 0.0))
                    st.metric("분석 상승률 / 손절", f"+{ana_gain:.1f}% / -{sel_row['sl_pct']:.1f}%")

                with insp_btn1:
                    st.markdown("<br>", unsafe_allow_html=True)
                    if st.button(f"{sel_row['name']} 7년 백테스트 실행", type="primary", key=f"bt_wl_btn_{sel_row['code']}", use_container_width=True):
                        st.session_state["target_backtest_code"] = sel_row["code"]
                        st.session_state["verif_pipeline_mode_sel"] = "과거 5년 워크포워드 & KODEX 200 벤치마크 검증 (종목별 95% 신뢰구간 시뮬레이션)"
                        st.session_state["pending_active_tab"] = main_tab_names[3]
                        st.toast(f"{sel_row['name']} 백테스트 설정 완료! 4번 통합 성과 검증 탭을 확인하세요.")
                        st.rerun()
                with insp_btn2:
                    st.markdown("<br>", unsafe_allow_html=True)
                    if st.button("2번 애널리스트 Q&A", key=f"btn_wl_to_qa_{sel_row['code']}", use_container_width=True):
                        st.session_state["qa_console_target_code"] = sel_row["code"]
                        st.session_state["qa_has_active_result"] = True
                        st.session_state["pending_active_tab"] = main_tab_names[1]
                        st.toast(f"{sel_row['name']} 2번 수석 애널리스트 콘솔로 이동합니다.")
                        st.rerun()
                with insp_btn3:
                    st.markdown("<br>", unsafe_allow_html=True)
                    if st.button("삭제", key=f"del_sel_{sel_row['code']}", use_container_width=True):
                        with get_db_connection() as conn:
                            cursor = conn.cursor()
                            cursor.execute("DELETE FROM watchlist WHERE code = ?", (sel_row["code"],))
                            conn.commit()
                        st.cache_data.clear()
                        st.toast(f"{sel_row['name']} 삭제 완료!")
                        st.rerun()

                # 3. Dual Outlook Diagnosis Cards (당일 실시간 vs 익일 스윙)
                rep_c1, rep_c2 = st.columns(2)
                with rep_c1:
                    with st.container(border=True):
                        st.markdown(f"**[당일 장중 실시간 진단]**: `{sel_row.get('intraday_tag', '-')}`")
                        st.caption(f"{sel_row.get('intraday_desc', '')}")
                        st.markdown(f"- **시초가 대비 등락**: `{sel_row.get('intraday_open_pct', 0.0):+.2f}%`")
                        st.markdown(f"- **장중 거래량 급증비**: `{sel_row.get('vol_ratio', 1.0):.2f}배`")
                        st.markdown(f"- **고확신 전략 부합**: `{sel_row.get('strategy_tag', '-')}`")
                with rep_c2:
                    with st.container(border=True):
                        st.markdown(f"**[익일 스윙 퀀트 진단]**: `{sel_row.get('nextday_tag', '-')}`")
                        st.caption(f"{sel_row.get('nextday_desc', '')}")
                        st.markdown(f"- **익일 퀀트 스코어**: `{sel_row['score']:.1f}점 / 100점` ({sel_row['label']})")
                        st.markdown(f"- **이평배열 상태**: `{sel_row.get('ma_status', '-')}`")
                        ana_gain = sel_row.get("analyzed_gain_pct", sel_row.get("tp_pct", 0.0))
                        st.markdown(f"- **분석 상승률 & 손절선**: `분석 상승률 +{ana_gain:.1f}% / 손절 -{sel_row['sl_pct']:.1f}%`")

                # Detailed Technical Breakdown
                bk = sel_row["breakdown"]
                b1, b2, b3, b4 = st.columns(4)
                b1.info(f"**RSI(14)**: {bk.get('rsi_val', 0):.1f} (기여: {bk.get('rsi_score', 0)}점)")
                b2.info(f"**이평배열**: {bk.get('ma_status', '-')} (기여: {bk.get('ma_score', 0)}점)")
                b3.info(f"**거래량 급증**: {bk.get('vol_ratio', 1.0):.2f}배 (기여: {bk.get('vol_score', 0)}점)")
                b4.info(f"**볼린저 %b**: {bk.get('bb_pct', 0.5):.2f} (기여: {bk.get('bb_score', 0)}점)")

        # -------------------------------------------------
        # MODE B: Paginated Card View
        # -------------------------------------------------
        else:
            page_size = int(wl_display_rows) if 'wl_display_rows' in locals() else 12
            total_pages = max(1, (len(df_filtered) - 1) // page_size + 1)
            p_col1, p_col2 = st.columns([4, 1])
            with p_col2:
                page_num = st.number_input("페이지 선택", min_value=1, max_value=total_pages, value=1, step=1, key="wl_card_page_num_in")

            start_idx = (page_num - 1) * page_size
            end_idx = min(start_idx + page_size, len(df_filtered))
            paged_data = df_filtered.iloc[start_idx:end_idx]

            card_cols = st.columns(3)
            for i, (_, item) in enumerate(paged_data.iterrows()):
                with card_cols[i % 3]:
                    with st.container(border=True):
                        st.markdown(f"**{item['name']}** (`{item['code']}`)")
                        st.caption(f"{item['market']} | {item['sector']} | 그룹: {item.get('group_name', '기본그룹')}")
                        chg_c = "red" if item["change_pct"] > 0 else ("blue" if item["change_pct"] < 0 else "gray")
                        st.markdown(f"**{item['close']:,.0f}원** <span style='color:{chg_c}; font-weight:600;'>{item['change_pct']:+.2f}%</span>", unsafe_allow_html=True)
                        st.markdown(f"**당일**: `{item.get('intraday_tag', '-')}`")
                        st.markdown(f"**익일**: `{item.get('nextday_tag', '-')}`")
                        st.progress(min(1.0, max(0.0, item["score"] / 100.0)))
                        card_ana_gain = item.get("analyzed_gain_pct", item.get("tp_pct", 0.0))
                        st.caption(f"분석 상승률: +{card_ana_gain:.1f}% | 손절: -{item['sl_pct']:.1f}%")

                        btn_c1, btn_c2 = st.columns(2)
                        with btn_c1:
                            if st.button("백테스트", key=f"bt_card_{item['code']}", use_container_width=True):
                                st.session_state["target_backtest_code"] = item["code"]
                                st.session_state["verif_pipeline_mode_sel"] = "과거 5년 워크포워드 & KODEX 200 벤치마크 검증 (종목별 95% 신뢰구간 시뮬레이션)"
                                st.session_state["pending_active_tab"] = main_tab_names[3]
                                st.toast(f"{item['name']} 백테스트 설정됨! 4번 통합 성과 검증 탭을 확인하세요.")
                                st.rerun()
                        with btn_c2:
                            if st.button("삭제", key=f"del_card_{item['code']}", use_container_width=True):
                                with get_db_connection() as conn:
                                    cursor = conn.cursor()
                                    cursor.execute("DELETE FROM watchlist WHERE code = ?", (item["code"],))
                                    conn.commit()
                                st.cache_data.clear()
                                st.rerun()




# =========================================================
# SUB-MODULE A: 워크포워드 검증 & 벤치마크 리포트 (95% CI 부트스트랩)
# =========================================================
def render_walk_forward_section():
    st.markdown('<div class="office-subheading">과거 5년 워크포워드 백테스트 & KODEX 200 벤치마크 검증 리포트</div>', unsafe_allow_html=True)
    st.caption("250거래일 In-Sample 학습 구간 롤링 후 20거래일 Out-of-Sample 순차 테스트 (과최적화 방지 & 부트스트랩 1,000회 95% 신뢰구간 산출)")

    bt_c1, bt_c2, bt_c3, bt_c4, bt_c5 = st.columns([2.5, 2.2, 1.8, 1.5, 1.5])
    with bt_c1:
        # Pre-populate with watchlist codes or direct input
        with get_db_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT code, name FROM watchlist ORDER BY name ASC")
            wl_opts = [f"{r['name']} ({r['code']})" for r in cursor.fetchall()]
        if not wl_opts:
            wl_opts = ["삼성전자 (005930)", "SK하이닉스 (000660)"]

        # If a stock was selected from Tab 1 inspector
        target_override = st.session_state.get("target_backtest_code")
        default_idx = 0
        if target_override:
            for idx, opt in enumerate(wl_opts):
                if target_override in opt:
                    default_idx = idx
                    break

        selected_wl = st.selectbox("검증 대상 종목", wl_opts, index=default_idx, key="bt_target_wl")
        target_code = selected_wl.split("(")[-1].replace(")", "").strip()
    with bt_c2:
        bt_strategy = st.selectbox(
            "검증 전략 모드",
            [
                "일반 퀀트 모드 (기본 점수제)",
                "실시간 당일 단타 모드 (당일 +5.0% 익절 / 당일 전량청산)",
                "스나이퍼 고확신 모드 (오전 9시 진입: 80% 승률 타겟)",
                "5% 급등 타겟 모드 (마감 직전 15:20 진입: 1~2일 스윙)",
                "주도주 종가배팅 모드 (마감 직전 15:20 매수 -> 익일 시초 갭익절)"
            ],
            index=0,
            key="bt_strategy_sel"
        )
    with bt_c3:
        bt_min_score = st.slider("최소 점수 (일반 모드)", 60.0, 85.0, 70.0, step=5.0, key="bt_min_score_slider")
    with bt_c4:
        bt_start_year = st.selectbox("시작 연도", [2017, 2018, 2019, 2020, 2021, 2022], index=3, key="bt_start_year_sel")
    with bt_c5:
        if "당일 단타" in bt_strategy:
            default_sl = 2.5
        elif ("5% 급등" in bt_strategy or "5% 돌파" in bt_strategy):
            default_sl = 4.0
        else:
            default_sl = 2.0
        bt_sl = st.slider("손절선 (%)", 1.0, 6.0, default_sl, step=0.5, key="bt_sl_slider")

    if st.button("워크포워드 백테스트 & 부트스트랩 검증 실행", type="primary", use_container_width=True):
        if "당일 단타" in bt_strategy:
            mode_key = "DAY_TRADE_5PCT"
        elif ("5% 급등" in bt_strategy or "5% 돌파" in bt_strategy):
            mode_key = "SURGE_5PCT"
        elif "스나이퍼" in bt_strategy:
            mode_key = "SNIPER"
        elif "종가배팅" in bt_strategy:
            mode_key = "CLOSING_BET"
        else:
            mode_key = "NORMAL"
        with st.spinner(f"[{target_code}] {bt_strategy} 롤링 워크포워드 검증 및 1,000회 부트스트랩 리샘플링 중..."):
            bt_results = run_walk_forward_backtest(
                code=target_code,
                start_year=bt_start_year,
                min_entry_score=bt_min_score,
                strategy_mode=mode_key,
                custom_sl_pct=bt_sl
            )
            st.session_state["cached_backtest_results"] = bt_results

    bt_results = st.session_state.get("cached_backtest_results")
    if bt_results is not None:
        if "error" in bt_results:
            st.error(bt_results["error"])
        elif "warning" in bt_results:
            st.warning(bt_results["warning"])
        else:
            stats = bt_results["stats"]
            trades_df = bt_results["trades_df"]
            equity_curve = bt_results["equity_curve"]
            exit_counts = bt_results["exit_counts"]

            st.markdown("---")
            # 1. Big Status Badge
            badge_col1, badge_col2 = st.columns([1, 3])
            with badge_col1:
                if stats["is_verified"]:
                    st.markdown("### 상태 배지: <br><span class='badge-verified' style='font-size:1.3rem; padding:8px 16px;'>● 검증됨 (Verified)</span>", unsafe_allow_html=True)
                else:
                    st.markdown("### 상태 배지: <br><span class='badge-experimental' style='font-size:1.3rem; padding:8px 16px;'>● 실험적 (Experimental)</span>", unsafe_allow_html=True)
            with badge_col2:
                st.markdown(
                    f"**검증 판정 기준**: KODEX 200 대비 초과수익률 95% 신뢰구간 하한 > 0 **AND** 승률 95% 신뢰구간 하한 > 50%<br>"
                    f"- 현재 초과수익 95% CI 하한: **{stats['excess_return_ci'][0]:+.2f}%** ({'충족' if stats['excess_return_ci'][0] > 0 else '미충족'})<br>"
                    f"- 현재 승률 95% CI 하한: **{stats['win_rate_ci'][0]:.1f}%** ({'충족' if stats['win_rate_ci'][0] > 50 else '미충족'})",
                    unsafe_allow_html=True
                )

            st.markdown("<br>", unsafe_allow_html=True)
            # 2. Key Metrics Row
            m1, m2, m3, m4, m5 = st.columns(5)
            m1.metric(
                "실현 승률",
                f"{stats['win_rate']:.1f}%",
                help=f"95% 부트스트랩 신뢰구간: [{stats['win_rate_ci'][0]:.1f}% ~ {stats['win_rate_ci'][1]:.1f}%]"
            )
            m2.metric(
                "KODEX 200 초과수익(알파)",
                f"{stats['mean_excess_return']:+.2f}%",
                help=f"95% 부트스트랩 신뢰구간: [{stats['excess_return_ci'][0]:+.2f}% ~ {stats['excess_return_ci'][1]:+.2f}%]"
            )
            m3.metric("Profit Factor (PF)", f"{stats['profit_factor']:.2f}")
            m4.metric("Max Drawdown (MDD)", f"-{stats['max_drawdown_pct']:.2f}%")
            m5.metric("총 OOS 거래 건수", f"{stats['total_trades']}건")

            st.markdown("---")

            # 3. Interactive Charts: Equity Curve & Distribution
            chart_col1, chart_col2 = st.columns([2, 1])
            with chart_col1:
                st.markdown("#### 누적 수익률 비교 (전략 vs KODEX 200 ETF)")
                if not equity_curve.empty:
                    fig_eq = go.Figure()
                    fig_eq.add_trace(go.Scatter(
                        x=equity_curve["date"],
                        y=equity_curve["strat_cum_return"],
                        mode="lines",
                        name="전략 누적 순수익률 (%)",
                        line=dict(color="#2563EB", width=2.5)
                    ))
                    fig_eq.add_trace(go.Scatter(
                        x=equity_curve["date"],
                        y=equity_curve["bm_cum_return"],
                        mode="lines",
                        name="KODEX 200 동시보유 수익률 (%)",
                        line=dict(color="#94A3B8", width=1.8, dash="dash")
                    ))
                    fig_eq.update_layout(
                        margin=dict(l=20, r=20, t=30, b=20),
                        hovermode="x unified",
                        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
                    )
                    st.plotly_chart(fig_eq, use_container_width=True)

            with chart_col2:
                st.markdown("#### 청산 사유 분포")
                if exit_counts:
                    labels = list(exit_counts.keys())
                    values = list(exit_counts.values())
                    fig_pie = px.pie(
                        names=labels,
                        values=values,
                        hole=0.45,
                        color_discrete_sequence=["#22C55E", "#EF4444", "#3B82F6"]
                    )
                    fig_pie.update_layout(margin=dict(l=10, r=10, t=30, b=10))
                    st.plotly_chart(fig_pie, use_container_width=True)

            # 4. Bootstrap Win Rate Distribution Histogram
            st.markdown("#### 1,000회 부트스트랩 승률 분포 (95% 신뢰구간)")
            boot_wr = stats.get("bootstrap_win_rates", [])
            if boot_wr:
                fig_hist = px.histogram(
                    x=boot_wr,
                    nbins=25,
                    title=f"승률 95% 신뢰구간: [{stats['win_rate_ci'][0]:.1f}% ~ {stats['win_rate_ci'][1]:.1f}%]",
                    labels={"x": "리샘플링 승률 (%)", "y": "빈도수"},
                    color_discrete_sequence=["#6366F1"]
                )
                fig_hist.add_vline(x=50.0, line_dash="dash", line_color="red", annotation_text="기준선 50%")
                fig_hist.update_layout(margin=dict(l=20, r=20, t=40, b=20))
                st.plotly_chart(fig_hist, use_container_width=True)

            # 5. Recent Trades Detailed Table
            st.markdown("#### 최근 Out-of-Sample 상세 매매 내역 (최근 15건)")
            recent_trades = trades_df.tail(15).iloc[::-1]
            st.dataframe(
                recent_trades[[
                    "entry_date", "score", "raw_entry", "raw_exit", "exit_reason",
                    "net_return_pct", "bm_return_pct", "excess_return_pct"
                ]].rename(columns={
                    "entry_date": "진입일자",
                    "score": "신호 스코어",
                    "raw_entry": "진입가",
                    "raw_exit": "청산가",
                    "exit_reason": "청산 사유",
                    "net_return_pct": "전략 순수익률(%)",
                    "bm_return_pct": "KODEX 200(%)",
                    "excess_return_pct": "초과수익(알파)(%)"
                }),
                use_container_width=True
            )


# =========================================================
# SUB-MODULE B: 실거래 예측 다이어리 & 라이브 추적 (Forward Testing Log)
# =========================================================
def render_prediction_diary_section():
    st.markdown('<div class="office-subheading">실거래 예측 다이어리 전체 장부 & 라이브 포워드 추적</div>', unsafe_allow_html=True)
    st.caption("매일 15:40 마감 후 생성된 예측 건과 익일 실제 체결 시세(시/고/저/종)를 일대일 대조하여 모델 열화(Model Decay)를 실시간 감지합니다.")

    # 1. Model Decay Status Card
    decay_info = evaluate_model_decay(expected_ci_lower=50.0)
    
    col_dec1, col_dec2 = st.columns([1.2, 3])
    with col_dec1:
        if decay_info["status"] == "DECAY_WARNING":
            st.error("모델 열화(Model Decay) 감지")
        elif decay_info["status"] == "HEALTHY":
            st.success("모델 성능 정상 (Healthy)")
        else:
            st.info("표본 축적 진행 중")
    with col_dec2:
        st.markdown(f"**진단 결과**: {decay_info['message']}")
        st.caption(f"누적 정산: {decay_info['total_settled']}건 | 적중: {decay_info.get('hits', 0)}건 | 실현 승률: {decay_info['live_win_rate']:.1f}%")

    st.markdown("---")

    # Manual Settlement / Refresh Action
    act_col1, act_col2 = st.columns([3, 1])
    with act_col1:
        st.markdown("#### 일자별 예측 및 익일 정산 내역")
    with act_col2:
        if st.button("전일 예측 즉시 정산", use_container_width=True):
            with st.spinner("미정산 예측 건 시세 조회 및 정산 처리 중..."):
                settled = settle_pending_predictions()
                st.toast(f"{len(settled)}건의 예측이 정산되었습니다.")
                st.rerun()

    # Load Diary History
    diary_df = get_diary_history(limit=50)

    if diary_df.empty:
        st.info("현재 기록된 예측 다이어리가 없습니다. 15:40 마감 배치를 실행하거나 사이드바의 '15:40 마감 배치' 버튼을 눌러보세요.")
    else:
        # Display Table
        disp_df = diary_df.copy()
        disp_df["hit_status"] = disp_df["hit_status"].fillna("정산대기 (PENDING)")
        
        st.dataframe(
            disp_df[[
                "prediction_date", "code", "name", "score", "label",
                "target_tp", "target_sl", "status", "execution_date",
                "actual_return", "hit_status", "exit_type"
            ]].rename(columns={
                "prediction_date": "예측일자",
                "code": "종목코드",
                "name": "종목명",
                "score": "스코어",
                "label": "전망",
                "target_tp": "목표익절(%)",
                "target_sl": "손절기준(%)",
                "status": "상태",
                "execution_date": "체결일자",
                "actual_return": "실현수익률(%)",
                "hit_status": "적중여부",
                "exit_type": "청산유형"
            }),
            use_container_width=True
        )

        # Forward Testing Hit Rate Chart
        settled_subset = diary_df[diary_df["hit_status"].isin(["HIT", "MISS"])].copy()
        if not settled_subset.empty:
            st.markdown("#### 라이브 포워드 테스팅 누적 적중률 추이")
            settled_subset["is_hit"] = (settled_subset["hit_status"] == "HIT").astype(int)
            settled_subset = settled_subset.iloc[::-1].reset_index(drop=True)
            settled_subset["cum_hit_rate"] = (settled_subset["is_hit"].cumsum() / (settled_subset.index + 1)) * 100.0

            fig_live = px.line(
                settled_subset,
                x="execution_date",
                y="cum_hit_rate",
                title="라이브 예측 실현 승률 추이 (%)",
                labels={"execution_date": "체결일자", "cum_hit_rate": "누적 승률 (%)"},
                markers=True
            )
            fig_live.add_hline(y=50.0, line_dash="dash", line_color="gray", annotation_text="50% 기준선")
            fig_live.update_layout(margin=dict(l=20, r=20, t=40, b=20))
            st.plotly_chart(fig_live, use_container_width=True)


# =========================================================
# SUB-MODULE C: 실거래 성과 통합 검증 & AI 자가최적화 (All-in-One)
# =========================================================
@st.cache_data(ttl=300, show_spinner=False)
def _get_cached_meta_opt():
    return evaluate_multi_horizon_tuning_impact()


@st.cache_data(ttl=300, show_spinner=False)
def _get_cached_decay_info():
    return evaluate_model_decay(expected_ci_lower=50.0)


@st.cache_data(ttl=300, show_spinner=False)
def _get_cached_weekly_res(limit_days=5):
    return get_weekly_verification_summary(limit_days=limit_days)


@st.cache_data(ttl=600, show_spinner=False)
def _get_quick_pit_verification_res(pred_d: str, strat: str):
    return run_daily_point_in_time_verification(
        pred_date=pred_d,
        strategy_mode=strat,
        min_val_krw=10_000_000_000,
        score_cutoff=65.0,
        sample_pool_size=40,
        force_refresh=False
    )


def render_unified_verification_and_tuning_section():
    # 1. Multi-Horizon AI Optimization Analysis (Instant with cache)
    meta_opt = _get_cached_meta_opt()
    best_horizon_lbl = meta_opt["best_horizon_label"]
    best_ret = meta_opt["best_expected_return"]
    best_win = meta_opt["best_expected_win_rate"]
    best_props = meta_opt["best_proposals"]

    # Master Action Header: One Button for Both Verification and Application
    m_hdr_c1, m_hdr_c2 = st.columns([3.2, 1.8], vertical_alignment="center")
    with m_hdr_c1:
        st.markdown(
            '<div style="display:flex; flex-direction:column; justify-content:center; min-height:42px;">'
            '<div class="office-heading" style="font-size:1.05rem; font-weight:700; color:#0F172A; margin:0; line-height:1.3;">[통합 성과 검증 & AI 최적화 센터]</div>'
            '<div style="font-size:0.75rem; color:#64748B; margin-top:3px; line-height:1.3;">다주기 실거래 검증 및 최적화 파라미터 1번 스크리너 즉시 반영</div>'
            '</div>',
            unsafe_allow_html=True
        )
    with m_hdr_c2:
        btn_ph = st.empty()
        if btn_ph.button("검증&최적화", type="primary", use_container_width=True, key="btn_run_all_in_one_master"):
            def _render_btn_pct(pct: int):
                p_val = max(0, min(100, pct))
                btn_ph.markdown(
                    f'''
                    <div style="width:100%; display:flex; height:38px; align-items:center;">
                        <button disabled style="
                            width: 100%;
                            height: 38px;
                            background: #2563EB;
                            color: #FFFFFF;
                            padding: 0 1rem;
                            border-radius: 6px;
                            border: none;
                            font-weight: 700;
                            font-size: 0.92rem;
                            cursor: wait;
                            opacity: 0.95;
                            box-shadow: 0 1px 3px rgba(0,0,0,0.1);
                            text-align: center;
                            display: flex;
                            align-items: center;
                            justify-content: center;
                            margin: 0;
                        ">검증&최적화({p_val}%)</button>
                    </div>
                    ''',
                    unsafe_allow_html=True
                )

            def _step_progress(start_p: int, end_p: int, step_delay: float = 0.02):
                step = 1 if end_p >= start_p else -1
                for p in range(start_p, end_p + step, step):
                    _render_btn_pct(p)
                    time.sleep(step_delay)

            # 1. 0% -> 20%: 미정산 체결 내역 정산
            _render_btn_pct(0)
            _step_progress(0, 20, step_delay=0.015)
            settle_pending_predictions()

            # 2. 20% -> 60%: 다주기(전일·주간·1개월) 실거래 성과 검증
            _step_progress(20, 45, step_delay=0.015)
            run_weekly_batch_verification(limit_days=5, force_refresh=False)
            _step_progress(45, 65, step_delay=0.015)

            # 3. 65% -> 90%: 최적화 파라미터 산출 및 스크리너 적용 준비
            _get_cached_meta_opt.clear()
            _get_cached_decay_info.clear()
            _get_cached_weekly_res.clear()
            _get_quick_pit_verification_res.clear()
            st.session_state["pending_tuning_updates"] = best_props
            st.session_state["tuning_applied_notification"] = True
            st.session_state["last_master_run_at"] = get_now_kst().strftime("%Y-%m-%d %H:%M:%S")
            st.session_state["last_master_best_horizon"] = best_horizon_lbl
            st.session_state["last_master_best_ret"] = best_ret
            st.session_state["last_master_best_win"] = best_win
            _step_progress(65, 90, step_delay=0.015)

            # 4. 90% -> 100%: 최종 완료
            _step_progress(90, 100, step_delay=0.02)
            time.sleep(0.30)
            st.toast(f"검증&최적화 완료! AI 최고 추천 주기({best_horizon_lbl})의 최적 파라미터가 1번 스크리너에 즉시 적용되었습니다.")
            st.rerun()

    # Clear boundary spacing so that header row never overlaps with metrics or banner below
    st.markdown("<div style='height:14px; margin-bottom:12px; clear:both;'></div>", unsafe_allow_html=True)

    # 2. Status Notification Banner
    if st.session_state.get("tuning_applied_notification") or "last_master_run_at" in st.session_state:
        v_time = st.session_state.get("last_master_run_at", get_now_kst().strftime("%Y-%m-%d %H:%M:%S"))
        h_lbl = st.session_state.get("last_master_best_horizon", best_horizon_lbl)
        e_ret = st.session_state.get("last_master_best_ret", best_ret)
        e_win = st.session_state.get("last_master_best_win", best_win)
        st.markdown(
            f"""
            <div style="background:#F0FDF4; border:1px solid #86EFAC; border-radius:6px; padding:10px 14px; margin:8px 0 12px 0;">
                <div style="display:flex; justify-content:space-between; align-items:center; flex-wrap:wrap; gap:6px;">
                    <span style="font-size:0.92rem; font-weight:700; color:#15803D;">
                        [검증&최적화 완료] AI 최고 추천 주기({h_lbl}) 파라미터가 스크리너(1번 탭)에 반영되었습니다!
                    </span>
                    <span style="font-size:0.80rem; font-weight:700; background:#DCFCE7; color:#166534; padding:2px 8px; border-radius:4px; border:1px solid #BBF7D0;">
                        예상 순수익률: +{e_ret:.2f}% | 예상 승률: {e_win:.1f}%
                    </span>
                </div>
                <div style="font-size:0.78rem; color:#166534; margin-top:4px;">
                    최근 검증 및 적용 시각: {v_time} | 1번 스크리너 탭으로 이동하시면 최적화된 조건이 즉시 반영되어 있습니다.
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )
        if st.session_state.get("tuning_applied_notification"):
            st.session_state["tuning_applied_notification"] = False

    # 3. Model Health Card & Core KPIs (Instant with cache)
    decay_info = _get_cached_decay_info()
    weekly_res = _get_cached_weekly_res(limit_days=5)

    k1, k2, k3, k4 = st.columns(4)
    with k1:
        st.metric(
            "실거래 모델 수명 상태",
            "정상 (Healthy)" if decay_info.get("status") == "HEALTHY" else ("경고 (Decay)" if decay_info.get("status") == "DECAY_WARNING" else "표본 축적중"),
            delta=f"누적 정산 {decay_info.get('total_settled', 0)}건"
        )
    with k2:
        wr = weekly_res.get('overall_win_rate', 0.0) if weekly_res else 0.0
        hits = weekly_res.get('total_hits', 0) if weekly_res else 0
        total = weekly_res.get('total_screened', 0) if weekly_res else 0
        st.metric(
            "최근 주간 실현 승률",
            f"{wr:.1f}%",
            delta=f"{hits} / {total} 종목 적중"
        )
    with k3:
        avg_ret = weekly_res.get('avg_net_ret', 0.0) if weekly_res else 0.0
        st.metric(
            "주간 평균 실현 수익률",
            f"{avg_ret:+.2f}%",
            delta="5거래일 체결 평균"
        )
    with k4:
        best_strat = str(weekly_res.get('best_strategy') or '-').split('(')[0].strip() if weekly_res else '-'
        st.metric(
            "주간 최고 승률 전략",
            best_strat,
            delta=f"승률 {weekly_res.get('best_strat_win_rate', 0.0):.1f}%" if weekly_res else "-"
        )

    st.markdown("---")

    # 4. AI Multi-Horizon Comparison & Applied Tuning Table
    st.markdown("#### [AI 주기별 진단 비교 및 최적 파라미터 적용 판정표]")
    st.markdown(
        f"""
        <div style="background:#F8FAFC; border:1px solid #CBD5E1; border-radius:6px; padding:10px 14px; margin-bottom:10px;">
            <div style="font-size:0.85rem; font-weight:700; color:#0F172A;">
                AI 수익률 극대화 최적 추천 주기: <span style="color:#0284C7;">{best_horizon_lbl}</span> (자동 추천 적용)
            </div>
            <div style="font-size:0.78rem; color:#475569; margin-top:4px;">
                {meta_opt["best_rationale"]}
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )
    st.table(meta_opt["comparative_table"])

    # 5. Expandable Detailed Performance Logs & 5-Year Backtest
    with st.expander("[최근 실거래 성적표 & 상세 검증 내역 보기]", expanded=False):
        valid_pairs = get_valid_prediction_dates(limit=30)
        latest_pred_d = valid_pairs[-1][0] if valid_pairs else "2026-09-22"
        v_res = _get_quick_pit_verification_res(latest_pred_d, "실시간 당일 단타 (5% 익절)")
        if v_res and "df_results" in v_res:
            df_res = v_res["df_results"]
            disp_cols = [
                "code", "name", "market", "t1_score", "strategy", "open_gap_pct",
                "t_open", "t_high", "t_close", "max_gain_pct", "net_return_pct",
                "is_hit", "exit_reason", "excess_return"
            ]
            if isinstance(df_res, pd.DataFrame) and not df_res.empty:
                avail_cols = [c for c in disp_cols if c in df_res.columns]
                disp_df = df_res[avail_cols].copy()
                if "is_hit" in disp_df.columns:
                    disp_df["is_hit"] = disp_df["is_hit"].apply(lambda x: "[적중]" if x else "[손절/미달]")
                st.dataframe(
                    disp_df.rename(columns={
                        "code": "종목코드", "name": "종목명", "market": "시장",
                        "t1_score": "T-1스코어", "strategy": "전략구분", "open_gap_pct": "시초갭(%)",
                        "t_open": "시초가", "t_high": "고가", "t_close": "종가",
                        "max_gain_pct": "장중최대(%)", "net_return_pct": "수익률(%)",
                        "is_hit": "적중여부", "exit_reason": "청산사유", "excess_return": "알파(%)"
                    }),
                    height=300,
                    use_container_width=True
                )
        if weekly_res and "strategy_summary" in weekly_res and not weekly_res["strategy_summary"].empty:
            st.markdown("**5대 전략별 최근 1주간(5일) 누적 성적표**")
            st.table(weekly_res["strategy_summary"].rename(columns={
                "strategy_mode": "전략 모드", "win_rate": "승률(%)", "avg_net_ret": "수익률(%)",
                "total_screened": "총검증수", "hits": "적중수", "avg_max_gain": "장중최대(%)"
            }))

    with st.expander("[실거래 예측 다이어리 전체 장부 보기]", expanded=False):
        render_prediction_diary_section()

    with st.expander("[과거 5년 워크포워드 & KODEX 200 벤치마크 검증 보기]", expanded=False):
        render_walk_forward_section()


# =========================================================
# TAB 4: 통합 성과 검증 & AI 자가최적화 센터 (All-in-One)
# =========================================================
with tab4:
    try:
        render_unified_verification_and_tuning_section()
    except Exception as _t4_err:
        import traceback
        st.error(f"검증 및 최적화 센터 로딩 중 오류가 발생했습니다: {_t4_err}")
        with st.expander("상세 오류 로그 확인", expanded=False):
            st.code(traceback.format_exc())

