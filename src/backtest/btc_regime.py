"""
Bitcoin Market Regime Classifier & Walk-Forward Validation Engine
----------------------------------------------------------------
- Model: Gaussian Hidden Markov Model (2-State: Bull vs Bear)
- Validation: Expanding Window Walk-Forward (Out-of-Sample)
- Features: Log Returns, Realized Volatility, Trend Deviation (Mayer Multiple proxy)
"""

import os
import warnings
from typing import Tuple, Optional, Callable, Dict, Any
import numpy as np
import pandas as pd
import yfinance as yf
from hmmlearn.hmm import GaussianHMM
from sklearn.preprocessing import RobustScaler
import plotly.graph_objects as go
from plotly.subplots import make_subplots

warnings.filterwarnings("ignore")

try:
    import truststore
    truststore.inject_into_ssl()
except Exception:
    pass

CACHE_FILE_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
    "data",
    "btc_regime_cache.parquet"
)
RAW_CACHE_FILE_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
    "data",
    "btc_raw_cache.parquet"
)


# =====================================================================
# 1. DATA INGESTION & FEATURE ENGINEERING
# =====================================================================
class DataPipeline:
    @staticmethod
    def fetch_btc_data(ticker: str = "BTC-USD", start_date: str = "2014-01-01", force_online: bool = False) -> pd.DataFrame:
        """야후 파이낸스 및 FDR에서 비트코인 일봉 데이터를 다운로드하고 정제합니다."""
        df = None

        # 1. Check local raw cache first if not forcing online refresh
        if not force_online and os.path.exists(RAW_CACHE_FILE_PATH):
            try:
                cached_raw = pd.read_parquet(RAW_CACHE_FILE_PATH)
                if not cached_raw.empty and len(cached_raw) > 500:
                    return cached_raw
            except Exception:
                pass

        # 2. Try yfinance first
        try:
            df = yf.download(ticker, start=start_date, progress=False)
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.get_level_values(0)
            if not df.empty and "Close" in df.columns and "Volume" in df.columns:
                df = df[["Close", "Volume"]].dropna()
                df.rename(columns={"Close": "price", "Volume": "volume"}, inplace=True)
            else:
                df = None
        except Exception:
            df = None

        # 3. Fallback to FinanceDataReader
        if df is None or df.empty:
            try:
                import FinanceDataReader as fdr
                fdr_df = fdr.DataReader("BTC/USD", start=start_date)
                if not fdr_df.empty:
                    df = fdr_df[["Close", "Volume"]].dropna()
                    df.rename(columns={"Close": "price", "Volume": "volume"}, inplace=True)
            except Exception:
                pass

        # 4. Fallback to offline cached file if fetch failed
        if (df is None or df.empty) and os.path.exists(RAW_CACHE_FILE_PATH):
            try:
                df = pd.read_parquet(RAW_CACHE_FILE_PATH)
            except Exception:
                pass

        if df is None or df.empty:
            raise RuntimeError("비트코인 데이터를 불러올 수 없습니다. 인터넷 연결 및 소스를 확인해주세요.")

        # Ensure index is datetime
        df.index = pd.to_datetime(df.index)
        df.sort_index(inplace=True)

        # Save to raw cache
        try:
            os.makedirs(os.path.dirname(RAW_CACHE_FILE_PATH), exist_ok=True)
            df.to_parquet(RAW_CACHE_FILE_PATH)
        except Exception:
            pass

        return df

    @staticmethod
    def build_features(df: pd.DataFrame) -> pd.DataFrame:
        """
        비트코인 거시 레짐 분류를 위한 3대 핵심 퀀트 피처를 생성합니다.
        (전부 점진적 과거 지표만 사용하여 Look-ahead Bias 방지)
        """
        data = df.copy()

        # 1. 일일 로그 수익률 (Daily Log Return)
        data["log_ret"] = np.log(data["price"] / data["price"].shift(1))

        # 2. 30일 실현 변동성 (Realized Volatility)
        data["realized_vol"] = data["log_ret"].rolling(window=30).std() * np.sqrt(365)

        # 3. 200일 장기 추세 괴리율 (Mayer Multiple Proxy)
        sma_200 = data["price"].rolling(window=200).mean()
        data["trend_ratio"] = (data["price"] - sma_200) / sma_200

        # 초기 윈도우 결측치 제거
        data.dropna(inplace=True)
        return data


# =====================================================================
# 2. REGIME CLASSIFICATION ENGINE (HMM)
# =====================================================================
class MarketRegimeModel:
    def __init__(self, n_states: int = 2, random_state: int = 42):
        self.n_states = n_states
        self.model = GaussianHMM(
            n_components=n_states,
            covariance_type="full",
            n_iter=50,
            tol=1e-3,
            random_state=random_state
        )
        self.scaler = RobustScaler()
        self.bull_state_idx = None

    def fit(self, X: np.ndarray, returns: np.ndarray) -> None:
        """
        과거 데이터(In-Sample)로만 모델을 피팅합니다.
        평균 수익률이 더 높은 상태를 자동으로 'Bull(1)'로 태깅합니다.
        """
        X_scaled = self.scaler.fit_transform(X)
        self.model.fit(X_scaled)

        # 각 은닉 상태별 평균 일일 수익률 계산하여 Bull/Bear 결정
        hidden_states = self.model.predict(X_scaled)
        state_means = [returns[hidden_states == i].mean() for i in range(self.n_states)]
        self.bull_state_idx = int(np.argmax(state_means))

    def predict_state(self, X_t: np.ndarray) -> int:
        """
        신규 시점(Out-of-Sample) 데이터 1스텝의 레짐을 판별합니다 (0: Bear, 1: Bull).
        """
        X_t_scaled = self.scaler.transform(X_t)
        raw_state = self.model.predict(X_t_scaled)[-1]
        return 1 if raw_state == self.bull_state_idx else 0


# =====================================================================
# 3. WALK-FORWARD VALIDATION ENGINE
# =====================================================================
class WalkForwardValidator:
    def __init__(self, initial_train_years: int = 3, rebalance_freq_days: int = 7):
        """
        Parameters:
        - initial_train_years: 초기 학습 기간 (사이클 학습을 위해 최소 3년 권장)
        - rebalance_freq_days: 모델 재학습 및 포지션 조정 주기 (1: 일간, 7: 주간)
        """
        self.init_train_len = initial_train_years * 365
        self.freq = rebalance_freq_days

    def run(
        self,
        df: pd.DataFrame,
        existing_df: Optional[pd.DataFrame] = None,
        progress_callback: Optional[Callable[[float, str], None]] = None
    ) -> pd.DataFrame:
        """
        Expanding Window 방식으로 시점을 과거로 되돌려 
        '오직 과거 데이터만으로 익일~익주를 예측'하는 실전 검증을 진행합니다.
        기존 사전 계산된 데이터(existing_df)가 존재할 경우 증분(Incremental) 구간만 효율적으로 계산합니다.
        """
        feature_cols = ["log_ret", "realized_vol", "trend_ratio"]
        total_len = len(df)

        if total_len <= self.init_train_len:
            raise ValueError("데이터 길이가 초기 학습 윈도우보다 짧습니다.")

        regime_preds = np.zeros(total_len)
        regime_preds[:] = np.nan

        start_idx = self.init_train_len
        # Reuse existing regimes if available to eliminate redundant CPU usage
        if existing_df is not None and not existing_df.empty and "regime" in existing_df.columns:
            common_idx = df.index.intersection(existing_df.index)
            if len(common_idx) > self.init_train_len:
                for dt in common_idx:
                    loc = df.index.get_loc(dt)
                    regime_preds[loc] = existing_df.loc[dt, "regime"]
                valid_locs = np.where(~np.isnan(regime_preds))[0]
                if len(valid_locs) > 0:
                    start_idx = max(self.init_train_len, valid_locs[-1] + 1)

        i = start_idx
        current_model = None

        while i < total_len:
            # 1. In-Sample 학습 (현재 시점 i 이전 데이터만 사용)
            train_sub = df.iloc[:i]
            X_train = train_sub[feature_cols].values
            ret_train = train_sub["log_ret"].values

            current_model = MarketRegimeModel(n_states=2)
            current_model.fit(X_train, ret_train)

            # 2. Out-of-Sample 예측 (다음 freq 구간 예측)
            eval_end = min(i + self.freq, total_len)
            for step in range(i, eval_end):
                # t 시점 예측을 위해 t 시점 직전 피처 벡터 전달
                X_step = df.iloc[max(0, step - 60):step][feature_cols].values
                pred = current_model.predict_state(X_step)
                regime_preds[step] = pred

            if progress_callback:
                pct = min(1.0, (i - self.init_train_len) / (total_len - self.init_train_len))
                date_str = df.index[min(i, total_len - 1)].strftime("%Y-%m-%d")
                progress_callback(pct, f"OOS 검증 진행 중 ({date_str})")

            i = eval_end

        if progress_callback:
            progress_callback(1.0, "OOS 전진 검증 완료")

        result_df = df.copy()
        result_df["regime"] = regime_preds
        return result_df.dropna(subset=["regime"])


# =====================================================================
# 4. PERFORMANCE EVALUATION & BACKTEST
# =====================================================================
class PerformanceEvaluator:
    @staticmethod
    def evaluate(df: pd.DataFrame) -> Tuple[Dict[str, Dict[str, str]], pd.DataFrame]:
        """
        Walk-Forward OOS 구간에서 Regime 전략과 단순 보유(Buy & Hold) 성과를 비교 분석합니다.
        Strategy Rule: Bull(1) 구간에서 Long 100%, Bear(0) 구간에서 Cash(0%)
        """
        eval_df = df.copy()

        # 전략 일별 수익률 (시그널은 당일 종가에 확정되므로 익일 수익률에 적용: shift(1))
        eval_df["signal"] = eval_df["regime"].shift(1).fillna(0)
        eval_df["strat_ret"] = eval_df["signal"] * eval_df["log_ret"]

        # 누적 수익률 계산
        eval_df["cum_bnh"] = np.exp(eval_df["log_ret"].cumsum())
        eval_df["cum_strat"] = np.exp(eval_df["strat_ret"].cumsum())

        # 메트릭 계산 함수
        def calc_metrics(cum_series: pd.Series, daily_ret: pd.Series):
            days = (cum_series.index[-1] - cum_series.index[0]).days
            cagr = (cum_series.iloc[-1] ** (365 / days)) - 1 if days > 0 else 0
            roll_max = cum_series.cummax()
            drawdown = (cum_series - roll_max) / roll_max
            mdd = drawdown.min()
            sharpe = (daily_ret.mean() / daily_ret.std()) * np.sqrt(365) if daily_ret.std() != 0 else 0
            total_ret = cum_series.iloc[-1] - 1.0
            return {
                "CAGR": f"{cagr * 100:.2f}%",
                "MDD": f"{mdd * 100:.2f}%",
                "Sharpe": f"{sharpe:.2f}",
                "TotalReturn": f"{total_ret * 100:.2f}%"
            }

        metrics = {
            "Regime Switching Strategy (OOS)": calc_metrics(eval_df["cum_strat"], eval_df["strat_ret"]),
            "Buy and Hold (OOS)": calc_metrics(eval_df["cum_bnh"], eval_df["log_ret"]),
        }
        return metrics, eval_df


# =====================================================================
# 5. CYCLE PROJECTION & DYNAMIC SCENARIO ENGINE
# =====================================================================
class CycleProjectionEngine:
    """
    비트코인 4년 반감기 주기론, Mayer Multiple 괴리율, 200주 이동평균선(200W SMA),
    가우시안 HMM 레짐 상태를 결합하여 현재 시황 데이터를 기반으로
    1) 가장 낮을 시점 (바닥 구간 및 지지선)
    2) 상승기 진입 시점 (1단계 축적기 & 2단계 폭발적 대세 상승기)
    3) 상승기 지속 기간 및 최종 사이클 정점 도달 시점
    4) 목표 가격 밴드
    를 실시간으로 동적 산출(Dynamic Computation)하는 분석 엔진입니다.
    """

    @staticmethod
    def calculate(df: pd.DataFrame) -> Dict[str, Any]:
        if df is None or df.empty or "price" not in df.columns:
            raise ValueError("유효한 비트코인 시세 데이터가 필요합니다.")

        prices = df["price"].dropna()
        total_records = len(prices)
        if total_records < 200:
            raise ValueError("사이클 분석을 위해 최소 200거래일 이상의 데이터가 필요합니다.")

        current_date = prices.index[-1]
        current_price = float(prices.iloc[-1])

        # 1. Macro High & Drawdown
        ath_price = float(prices.max())
        ath_date = prices.idxmax()
        days_since_ath = (current_date - ath_date).days
        current_drawdown_pct = ((current_price - ath_price) / ath_price) * 100.0

        post_ath_series = prices.loc[ath_date:]
        post_ath_min_price = float(post_ath_series.min())
        post_ath_min_date = post_ath_series.idxmin()
        post_ath_max_dd_pct = ((post_ath_min_price - ath_price) / ath_price) * 100.0
        days_ath_to_min = (post_ath_min_date - ath_date).days

        # 2. Moving Averages & Mayer Multiple
        sma_200 = float(prices.rolling(200).mean().iloc[-1])
        sma_200w = float(prices.rolling(min(total_records, 1400)).mean().iloc[-1]) if total_records >= 400 else sma_200 * 0.85
        mayer_multiple = current_price / sma_200 if sma_200 > 0 else 1.0
        realized_vol = float(df["realized_vol"].iloc[-1]) if "realized_vol" in df.columns else 0.35
        current_regime = int(df["regime"].iloc[-1]) if "regime" in df.columns else 0

        # 3. Halving Schedule
        # Historical: 2012-11-28, 2016-07-09, 2020-05-11, 2024-04-20
        # Next Halving is approx 1,460 days (4 years) from 2024-04-20 -> 2028-04-19
        last_halving = pd.Timestamp("2024-04-20")
        next_halving = last_halving + pd.Timedelta(days=1460)

        # 4. Dynamic Bottom Analysis (가장 낮을 시점)
        # Historical peak-to-bottom: 364 days to 411 days (mean 384 days)
        bottom_window_start = ath_date + pd.Timedelta(days=364)
        bottom_window_avg = ath_date + pd.Timedelta(days=384)
        bottom_window_end = ath_date + pd.Timedelta(days=411)

        # Support bands
        mayer_band_low = sma_200 * 0.75   # Deep value floor (0.75x)
        mayer_band_high = sma_200 * 0.85  # Accumulation floor (0.85x)

        # Dynamic status judgment based on days elapsed and price vs post_ath_min
        if days_since_ath < 180:
            bottom_status = "고점 형성 직후 초기 하락장 (바닥 미도달, 추가 조정 주의)"
            bottom_badge = "하락 국면"
            bottom_badge_color = "#DC2626"
            bottom_desc = f"고점 대비 {days_since_ath}일 경과. 역대 통계상 사이클 바닥은 고점 후 평균 384일 전후에 형성되므로 섣부른 매수보다 리스크 관리가 우선됩니다."
        elif days_since_ath < 350:
            bottom_status = f"하락장 심화 및 저점 탐색 구간 (예상 바닥 창구: {bottom_window_start.strftime('%Y-%m')} ~ {bottom_window_end.strftime('%Y-%m')})"
            bottom_badge = "저점 탐색"
            bottom_badge_color = "#EA580C"
            bottom_desc = f"고점 대비 {days_since_ath}일 경과. 역대 평균 바닥 일수(384일) 도달을 앞두고 지지선 탐색 및 분할 매집 준비가 유효한 구간입니다."
        elif days_since_ath <= 430:
            if current_price > post_ath_min_price * 1.15:
                bottom_status = f"1차 최저점(${post_ath_min_price:,.0f}) 통과 후 2차 지지(Double Bottom/Higher Low) 다지기 국면"
                bottom_badge = "2차 바닥 다지기"
                bottom_badge_color = "#2563EB"
                bottom_desc = f"고점 후 {days_since_ath}일 경과로 역대 통계적 바닥 창구(364~411일)를 통과 중입니다. 지난 {post_ath_min_date.strftime('%Y-%m-%d')}에 기록한 ${post_ath_min_price:,.0f} 대비 {((current_price / post_ath_min_price) - 1.0) * 100:+.1f}% 반등하여 200일선(${sma_200:,.0f}) 및 200주선(${sma_200w:,.0f}) 지지력을 시험하고 있습니다."
            else:
                bottom_status = "역대 평균 바닥 윈도우 진입 (사이클 최저점 및 극단적 저평가 테스트 진행 중)"
                bottom_badge = "바닥 테스트"
                bottom_badge_color = "#D97706"
                bottom_desc = f"고점 대비 {days_since_ath}일 경과로 역대 평균 바닥 소요 기간(384일) 부근입니다. 이전 최저점(${post_ath_min_price:,.0f}) 및 200주선(${sma_200w:,.0f}) 부근에서 최저 가격 방어가 이루어지는 핵심 변곡점입니다."
        else:
            bottom_status = f"사이클 바닥 윈도우 통과 완료 (최저점 ${post_ath_min_price:,.0f} 유효 지지 확인)"
            bottom_badge = "바닥 통과 확인"
            bottom_badge_color = "#16A34A"
            bottom_desc = f"고점 대비 {days_since_ath}일이 경과하여 하락장 시간 프레임을 벗어났으며, 차기 반감기를 향한 구조적 축적 및 저점 상승(Higher Low) 추세로 전환되었습니다."

        # 5. Dynamic Bull Run Start (상승기 진입 시점)
        # Stage 1: Accumulation / Recovery
        if current_regime == 1:
            stage1_status = "상승 국면 진입 완료 (HMM Bull 레짐 100% 롱 활성화)"
            stage1_badge = "상승 레짐 가동"
            stage1_badge_color = "#16A34A"
            stage1_desc = f"HMM 모델이 상승 국면으로 판정 중이며, 200일선(${sma_200:,.0f}) 상회 기반의 회복 추세가 가동되었습니다."
            stage1_date_str = "현재 진행 중"
        else:
            stage1_est_dt = max(current_date + pd.Timedelta(days=60), bottom_window_end + pd.Timedelta(days=60))
            stage1_status = f"{stage1_est_dt.strftime('%Y년 %m월')} ~ {(stage1_est_dt + pd.Timedelta(days=60)).strftime('%m월')} 전후 예상"
            stage1_badge = "축적 회복 대기"
            stage1_badge_color = "#64748B"
            stage1_desc = "바닥 다지기 완료 후 200일선 안착 및 HMM 모델의 Bull 레짐 전환 시그널이 발생할 때 1단계 점진적 상승기가 개막됩니다."
            stage1_date_str = stage1_status

        # Stage 2: Parabolic Bull Run (본격 대세 폭발 상승기)
        # Historically: Halving + 150~180 days (mean 165 days)
        parabolic_start_min = next_halving + pd.Timedelta(days=150)
        parabolic_start_avg = next_halving + pd.Timedelta(days=165)
        parabolic_start_max = next_halving + pd.Timedelta(days=180)
        parabolic_desc = f"역대 모든 사이클에서 전고점을 돌파하는 메가 랠리는 반감기 당일이 아니라 채굴 보상 반감 충격이 누적되는 반감기 후 150~180일(약 5~6개월) 시점에 점화되었습니다. 제5차 반감기({next_halving.strftime('%Y-%m-%d')} 기준) 대입 시 {parabolic_start_avg.strftime('%Y년 %m월')}(2028년 가을)부터 수직 상승장이 시작될 것으로 계산됩니다."

        # 6. Dynamic Peak & Bull Duration (언제까지 오를 것인가)
        # Historically (Cycles 2-4): 525, 548, 534 days (mean 536 days, range 520-550 days)
        peak_window_min = next_halving + pd.Timedelta(days=520)
        peak_window_avg = next_halving + pd.Timedelta(days=536)
        peak_window_max = next_halving + pd.Timedelta(days=550)

        # Duration calculations
        parabolic_duration_days = (peak_window_avg - parabolic_start_avg).days
        parabolic_duration_months = round(parabolic_duration_days / 30.4)
        total_cycle_duration_months = round((peak_window_avg - bottom_window_avg).days / 30.4)

        # Target Price Calculation (Diminishing Multiplier Model from ATH)
        target_conservative = ath_price * 1.45
        target_median = ath_price * 1.70
        target_optimistic = ath_price * 1.95

        # 7. Comparison Table DataFrame
        comparison_records = [
            {
                "사이클": "1차 사이클 (2012)",
                "반감기 일자": "2012-11-28",
                "사이클 고점": "$1,163 (2013-11-29)",
                "반감기->고점": "366일",
                "사이클 바닥": "$152 (2015-01-14)",
                "고점->바닥": "411일",
                "최대낙폭(MDD)": "-86.9%",
                "바닥 Mayer": "0.45",
            },
            {
                "사이클": "2차 사이클 (2016)",
                "반감기 일자": "2016-07-09",
                "사이클 고점": "$19,666 (2017-12-16)",
                "반감기->고점": "525일",
                "사이클 바닥": "$3,122 (2018-12-15)",
                "고점->바닥": "364일",
                "최대낙폭(MDD)": "-84.1%",
                "바닥 Mayer": "0.51",
            },
            {
                "사이클": "3차 사이클 (2020)",
                "반감기 일자": "2020-05-11",
                "사이클 고점": "$68,789 (2021-11-10)",
                "반감기->고점": "548일",
                "사이클 바닥": "$15,599 (2022-11-21)",
                "고점->바닥": "376일",
                "최대낙폭(MDD)": "-77.3%",
                "바닥 Mayer": "0.71",
            },
            {
                "사이클": "4차 사이클 (2024)",
                "반감기 일자": "2024-04-20",
                "사이클 고점": f"${ath_price:,.0f} ({ath_date.strftime('%Y-%m-%d')})",
                "반감기->고점": f"{(ath_date - last_halving).days}일",
                "사이클 바닥": f"${post_ath_min_price:,.0f} ({post_ath_min_date.strftime('%Y-%m-%d')})",
                "고점->바닥": f"{days_ath_to_min}일 (현재 {days_since_ath}일 경과)",
                "최대낙폭(MDD)": f"{post_ath_max_dd_pct:.1f}% (현재 {current_drawdown_pct:+.1f}%)",
                "바닥 Mayer": f"{(post_ath_min_price / sma_200):.2f}",
            },
            {
                "사이클": "5차 사이클 (차기 예상)",
                "반감기 일자": f"{next_halving.strftime('%Y-%m-%d')} (추정)",
                "사이클 고점": f"${target_median:,.0f} ({peak_window_avg.strftime('%Y-%m-%d')} 추정)",
                "반감기->고점": "536일 예상 (520~550일)",
                "사이클 바닥": f"${mayer_band_low:,.0f} ~ ${sma_200w:,.0f} (지지 밴드)",
                "고점->바닥": "384일 기준 모델링",
                "최대낙폭(MDD)": "수익률 감쇄 적용",
                "바닥 Mayer": "0.75 ~ 0.85 밴드",
            },
        ]
        comparison_df = pd.DataFrame(comparison_records)

        return {
            "current_date": current_date.strftime("%Y-%m-%d"),
            "current_price": current_price,
            "current_regime": current_regime,
            "ath_price": ath_price,
            "ath_date": ath_date.strftime("%Y-%m-%d"),
            "days_since_ath": days_since_ath,
            "current_drawdown_pct": current_drawdown_pct,
            "post_ath_min_price": post_ath_min_price,
            "post_ath_min_date": post_ath_min_date.strftime("%Y-%m-%d"),
            "post_ath_max_dd_pct": post_ath_max_dd_pct,
            "days_ath_to_min": days_ath_to_min,
            "sma_200": sma_200,
            "sma_200w": sma_200w,
            "mayer_multiple": mayer_multiple,
            "realized_vol_pct": realized_vol * 100.0,
            # Bottom window
            "bottom_window_start": bottom_window_start.strftime("%Y-%m-%d"),
            "bottom_window_avg": bottom_window_avg.strftime("%Y-%m-%d"),
            "bottom_window_end": bottom_window_end.strftime("%Y-%m-%d"),
            "bottom_status": bottom_status,
            "bottom_badge": bottom_badge,
            "bottom_badge_color": bottom_badge_color,
            "bottom_desc": bottom_desc,
            "mayer_band_low": mayer_band_low,
            "mayer_band_high": mayer_band_high,
            # Bull start
            "stage1_date_str": stage1_date_str,
            "stage1_status": stage1_status,
            "stage1_badge": stage1_badge,
            "stage1_badge_color": stage1_badge_color,
            "stage1_desc": stage1_desc,
            "next_halving": next_halving.strftime("%Y-%m-%d"),
            "parabolic_start_min": parabolic_start_min.strftime("%Y-%m-%d"),
            "parabolic_start_avg": parabolic_start_avg.strftime("%Y-%m-%d"),
            "parabolic_start_max": parabolic_start_max.strftime("%Y-%m-%d"),
            "parabolic_desc": parabolic_desc,
            # Peak & duration
            "peak_window_min": peak_window_min.strftime("%Y-%m-%d"),
            "peak_window_avg": peak_window_avg.strftime("%Y-%m-%d"),
            "peak_window_max": peak_window_max.strftime("%Y-%m-%d"),
            "parabolic_duration_months": parabolic_duration_months,
            "total_cycle_duration_months": total_cycle_duration_months,
            "target_conservative": target_conservative,
            "target_median": target_median,
            "target_optimistic": target_optimistic,
            "comparison_df": comparison_df,
        }


# =====================================================================
# 6. VISUALIZATION ENGINE
# =====================================================================
class Visualizer:
    @staticmethod
    def build_figure(eval_df: pd.DataFrame) -> go.Figure:
        """인터랙티브 Plotly 차트로 레짐 분류와 누적 성과 곡선을 생성합니다."""
        fig = make_subplots(
            rows=2, cols=1, shared_xaxes=True,
            vertical_spacing=0.08,
            subplot_titles=(
                "BTC Price & Walk-Forward Regime Classification (Green=Bull, Red=Bear)",
                "Cumulative Equity: Regime Strategy vs Buy & Hold (Log Scale)"
            )
        )

        # 1. Base price curve
        fig.add_trace(
            go.Scatter(
                x=eval_df.index,
                y=eval_df["price"],
                name="BTC Price",
                line=dict(color="#94A3B8", width=1.2)
            ),
            row=1, col=1
        )

        # 200-Day SMA
        sma_200 = eval_df["price"].rolling(200).mean()
        fig.add_trace(
            go.Scatter(
                x=eval_df.index,
                y=sma_200,
                name="200D SMA (추세선)",
                line=dict(color="#F59E0B", width=1.3, dash="dash")
            ),
            row=1, col=1
        )

        # 200-Week SMA (approx 1400 days)
        if len(eval_df) >= 300:
            sma_200w = eval_df["price"].rolling(1400).mean()
            fig.add_trace(
                go.Scatter(
                    x=eval_df.index,
                    y=sma_200w,
                    name="200W SMA (바닥 지지선)",
                    line=dict(color="#8B5CF6", width=1.3, dash="dot")
                ),
                row=1, col=1
            )

        # Bull and Bear highlight markers
        bull_df = eval_df[eval_df["regime"] == 1]
        bear_df = eval_df[eval_df["regime"] == 0]

        fig.add_trace(
            go.Scatter(
                x=bull_df.index,
                y=bull_df["price"],
                name="Bull Regime (Long 100%)",
                mode="markers",
                marker=dict(color="#16A34A", size=3, opacity=0.85)
            ),
            row=1, col=1
        )
        fig.add_trace(
            go.Scatter(
                x=bear_df.index,
                y=bear_df["price"],
                name="Bear Regime (Cash 0%)",
                mode="markers",
                marker=dict(color="#DC2626", size=3, opacity=0.85)
            ),
            row=1, col=1
        )

        # 2. Strategy vs BnH
        fig.add_trace(
            go.Scatter(
                x=eval_df.index,
                y=eval_df["cum_strat"],
                name="Regime Strategy",
                line=dict(color="#2563EB", width=2)
            ),
            row=2, col=1
        )
        fig.add_trace(
            go.Scatter(
                x=eval_df.index,
                y=eval_df["cum_bnh"],
                name="Buy & Hold",
                line=dict(color="#64748B", width=1.5, dash="dot")
            ),
            row=2, col=1
        )

        fig.update_yaxes(type="log", title_text="Price (USD)", row=1, col=1)
        fig.update_yaxes(type="log", title_text="Equity Multiple", row=2, col=1)
        fig.update_layout(
            height=750,
            template="plotly_white",
            hovermode="x unified",
            margin=dict(l=40, r=20, t=50, b=40),
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
        )
        return fig

    @staticmethod
    def plot_results(eval_df: pd.DataFrame):
        fig = Visualizer.build_figure(eval_df)
        fig.show()


# =====================================================================
# 7. CACHING & CONTROLLER FOR STREAMLIT INTEGRATION
# =====================================================================
def run_or_load_btc_regime_analysis(
    force_refresh: bool = False,
    progress_callback: Optional[Callable[[float, str], None]] = None
) -> Tuple[Dict[str, Dict[str, str]], pd.DataFrame]:
    """
    비트코인 레짐 분석을 로컬 캐시에서 즉시 불러오거나, 필요 시 신규 분석을 수행하여 캐싱합니다.
    """
    if not force_refresh and os.path.exists(CACHE_FILE_PATH):
        try:
            cached_eval_df = pd.read_parquet(CACHE_FILE_PATH)
            if not cached_eval_df.empty and "regime" in cached_eval_df.columns:
                metrics, evaluated_df = PerformanceEvaluator.evaluate(cached_eval_df)
                return metrics, evaluated_df
        except Exception:
            pass

    # Check if we have existing cached results for incremental validation
    existing_cached_df = None
    if not force_refresh and os.path.exists(CACHE_FILE_PATH):
        try:
            prev_df = pd.read_parquet(CACHE_FILE_PATH)
            if not prev_df.empty and "regime" in prev_df.columns:
                existing_cached_df = prev_df
        except Exception:
            pass

    # Execute workflow (incremental if previous cache exists)
    if progress_callback:
        progress_callback(0.05, "비트코인 과거 시세 다운로드 중...")

    raw_data = DataPipeline.fetch_btc_data("BTC-USD", start_date="2014-01-01", force_online=force_refresh)
    feature_data = DataPipeline.build_features(raw_data)

    if progress_callback:
        progress_callback(0.15, "HMM 2-State 워크포워드 증분 검증 시작 (Expanding Window)...")

    validator = WalkForwardValidator(initial_train_years=3, rebalance_freq_days=7)
    validated_df = validator.run(feature_data, existing_df=existing_cached_df, progress_callback=progress_callback)

    metrics, evaluated_df = PerformanceEvaluator.evaluate(validated_df)

    # Save to local cache
    try:
        os.makedirs(os.path.dirname(CACHE_FILE_PATH), exist_ok=True)
        evaluated_df.to_parquet(CACHE_FILE_PATH)
    except Exception:
        pass

    return metrics, evaluated_df


if __name__ == "__main__":
    print("[1/4] Fetching historical BTC data...")
    raw = DataPipeline.fetch_btc_data("BTC-USD", start_date="2014-01-01")
    features = DataPipeline.build_features(raw)

    print("[2/4] Running Walk-Forward validation with HMM...")
    val = WalkForwardValidator(initial_train_years=3, rebalance_freq_days=7)
    val_df = val.run(features)

    print("[3/4] Evaluating Out-of-Sample Performance...")
    res_metrics, ev_df = PerformanceEvaluator.evaluate(val_df)

    print("\n" + "=" * 55)
    print("           WALK-FORWARD BACKTEST METRICS (OOS)")
    print("=" * 55)
    for model_name, stat in res_metrics.items():
        print(f"[{model_name}]")
        for k, v in stat.items():
            print(f"  - {k:<12}: {v}")
    print("=" * 55 + "\n")

    print("[4/4] Generating Interactive Plotly Chart...")
    Visualizer.plot_results(ev_df)
