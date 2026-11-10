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
            n_iter=150,
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

    def run(self, df: pd.DataFrame, progress_callback: Optional[Callable[[float, str], None]] = None) -> pd.DataFrame:
        """
        Expanding Window 방식으로 시점을 과거로 되돌려 
        '오직 과거 데이터만으로 익일~익주를 예측'하는 실전 검증을 진행합니다.
        """
        feature_cols = ["log_ret", "realized_vol", "trend_ratio"]
        total_len = len(df)

        if total_len <= self.init_train_len:
            raise ValueError("데이터 길이가 초기 학습 윈도우보다 짧습니다.")

        regime_preds = np.zeros(total_len)
        regime_preds[:] = np.nan

        i = self.init_train_len
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
                X_step = df.iloc[step - 1:step][feature_cols].values
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
# 5. VISUALIZATION ENGINE
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
# 6. CACHING & CONTROLLER FOR STREAMLIT INTEGRATION
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

    # Execute full workflow
    if progress_callback:
        progress_callback(0.05, "비트코인 과거 시세 다운로드 중...")

    raw_data = DataPipeline.fetch_btc_data("BTC-USD", start_date="2014-01-01", force_online=force_refresh)
    feature_data = DataPipeline.build_features(raw_data)

    if progress_callback:
        progress_callback(0.15, "HMM 2-State 워크포워드 검증 시작 (Expanding Window)...")

    validator = WalkForwardValidator(initial_train_years=3, rebalance_freq_days=7)
    validated_df = validator.run(feature_data, progress_callback=progress_callback)

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
