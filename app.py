"""
日経平均 翌日予測アプリ v5
─────────────────────────────────────
予測対象: 前日終値からの騰落率(%)
モデル  : LightGBM / SVR / ランダムフォレスト / アンサンブル / LSTM
タブ    : ダッシュボード / 過去10日(方向性) / 過去10日(高値/安値) / 週着地 /
          各モデル(5) + ボリンジャーバンド
v5 新機能:
  1. 予測レンジ Hit/Miss フラグ + 精度トラッキング
  2. バイアス補正アンサンブル予測
  3. 週着地タグ (新タブ)
  4. 主要イベントカレンダー (FOMC/BOJ/SQ)
  5. 大ブレ警戒シグナル
  6. 直近100日Maxからのボリバンタグ (dist_from_max100, bb_max_tag)
"""

# ─── TensorFlow ログ抑制（インポート前に設定）───
import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

import warnings
warnings.filterwarnings("ignore")

import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, timedelta
import lightgbm as lgb
from sklearn.svm import SVR, SVC
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_absolute_error, accuracy_score

# ─── TensorFlow / Keras（任意）───
try:
    import tensorflow as tf
    tf.get_logger().setLevel("ERROR")
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import LSTM, Dense, Dropout
    from tensorflow.keras.callbacks import EarlyStopping
    from tensorflow.keras.optimizers import Adam
    LSTM_AVAILABLE = True
except Exception:
    LSTM_AVAILABLE = False

# ─────────────────────────────────────────
# ページ設定
# ─────────────────────────────────────────
st.set_page_config(page_title="日経平均 翌日予測", page_icon="📈", layout="centered")
st.markdown("""
<style>
.stMetric { text-align: center; }
@media (max-width: 600px) { .block-container { padding: 1rem 0.5rem; } }
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────
# 定数
# ─────────────────────────────────────────
TICKERS = {
    "nikkei":     "^N225",
    "nasdaq":     "^IXIC",
    "usdjpy":     "USDJPY=X",
    "nikkei_fut": "NKD=F",
}
END_DATE   = datetime.now().strftime("%Y-%m-%d")
START_DATE = (datetime.now() - timedelta(days=365 * 10 + 30)).strftime("%Y-%m-%d")
SEQ_LEN    = 20   # LSTM の参照日数

FEATURE_COLS = [
    "ret_lag1",       "ret_lag2",       "ret_lag3",
    "ret_lag5",       "ret_lag10",
    "high_ret_lag1",  "high_ret_lag2",
    "low_ret_lag1",   "low_ret_lag2",
    "vol_ret_lag1",
    "bb_pct",         "bb_width",
    "rsi14",          "atr_pct",
    "ma5_dev",        "ma10_dev",       "ma20_dev",       "ma60_dev",
    "ma5_slope_pct",  "ma20_slope_pct",
    "usdjpy_ret_lag1","nasdaq_ret_lag1",
    "dow",            "month",
    "dist_from_max100", "bb_max_tag",
]
TARGETS = ["target_high_pct", "target_low_pct", "target_dir"]

# ─── イベントカレンダー定数 ───
FOMC_DATES = [
    # 2024
    "2024-01-31","2024-03-20","2024-05-01","2024-06-12",
    "2024-07-31","2024-09-18","2024-11-07","2024-12-18",
    # 2025
    "2025-01-29","2025-03-19","2025-05-07","2025-06-18",
    "2025-07-30","2025-09-17","2025-10-29","2025-12-10",
    # 2026
    "2026-01-28","2026-03-18","2026-05-06","2026-06-17",
    "2026-07-29","2026-09-16","2026-10-28","2026-12-16",
]
BOJ_DATES = [
    # 2024
    "2024-01-23","2024-03-19","2024-04-26","2024-06-14",
    "2024-07-31","2024-09-20","2024-10-31","2024-12-19",
    # 2025
    "2025-01-24","2025-03-19","2025-04-30","2025-06-17",
    "2025-07-31","2025-09-22","2025-10-29","2025-12-19",
    # 2026
    "2026-01-23","2026-03-19","2026-04-28","2026-06-16",
    "2026-07-30","2026-09-25","2026-10-29","2026-12-18",
]

# bb_max_tag の説明辞書
BB_MAX_TAG_INFO = {
    5: ("🔥 最高値圏BB突破",  "100日高値付近でBBを上抜け"),
    4: ("📈 高値圏BB上限",    "100日高値-3%以内、BB上限付近"),
    3: ("📊 高値圏中央以上",  "100日高値から-3%以上下落、BB中央以上"),
    2: ("⚖️ 中立圏",          "標準的な位置"),
    1: ("🧲 下落圏BB下限",    "100日高値から-10%以上、BB下限付近"),
    0: ("⚠️ 下落圏BB割れ",    "100日高値から-15%以上、BBを下抜け"),
}

# ─────────────────────────────────────────
# SQ 日（各月第2金曜日）計算
# ─────────────────────────────────────────
def get_sq_dates(year_from: int, year_to: int) -> list:
    """各月の第2金曜日（日本SQ日）を計算して返す"""
    sq_list = []
    for year in range(year_from, year_to + 1):
        for month in range(1, 13):
            # その月の1日
            first_day = datetime(year, month, 1)
            # 1日の曜日 (0=月, 4=金)
            dow = first_day.weekday()
            # 最初の金曜日まで何日か
            days_to_first_fri = (4 - dow) % 7
            first_fri = first_day + timedelta(days=days_to_first_fri)
            second_fri = first_fri + timedelta(days=7)
            sq_list.append(second_fri.strftime("%Y-%m-%d"))
    return sq_list

# ─────────────────────────────────────────
# データ取得
# ─────────────────────────────────────────
@st.cache_data(ttl=3600, show_spinner=False)
def fetch_data() -> dict:
    result = {}
    for name, ticker in TICKERS.items():
        try:
            df = yf.download(ticker, start=START_DATE, end=END_DATE,
                             auto_adjust=True, progress=False)
            if df.empty:
                continue
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = [c[0].lower() for c in df.columns]
            else:
                df.columns = [c.lower() for c in df.columns]
            df.index = pd.to_datetime(df.index).tz_localize(None)
            result[name] = df
        except Exception:
            pass
    return result

# ─────────────────────────────────────────
# 特徴量エンジニアリング（全て相対値）
# ─────────────────────────────────────────
def build_features(dfs: dict) -> pd.DataFrame:
    nk = dfs["nikkei"][["open", "high", "low", "close", "volume"]].copy()
    c  = nk["close"]

    nk["ret"] = c.pct_change() * 100
    for lag in [1, 2, 3, 5, 10]:
        nk[f"ret_lag{lag}"] = nk["ret"].shift(lag)

    nk["high_ret"] = (nk["high"] / c - 1) * 100
    nk["low_ret"]  = (nk["low"]  / c - 1) * 100
    for lag in [1, 2]:
        nk[f"high_ret_lag{lag}"] = nk["high_ret"].shift(lag)
        nk[f"low_ret_lag{lag}"]  = nk["low_ret"].shift(lag)

    nk["vol_ret_lag1"] = nk["volume"].pct_change().shift(1) * 100

    nk["bb_mid"]   = c.rolling(20).mean()
    nk["bb_std"]   = c.rolling(20).std()
    nk["bb_upper"] = nk["bb_mid"] + 2 * nk["bb_std"]
    nk["bb_lower"] = nk["bb_mid"] - 2 * nk["bb_std"]
    bb_rng         = (nk["bb_upper"] - nk["bb_lower"]).replace(0, np.nan)
    nk["bb_pct"]   = (c - nk["bb_lower"]) / bb_rng
    nk["bb_width"] = bb_rng / nk["bb_mid"].replace(0, np.nan) * 100

    delta = c.diff()
    gain  = delta.clip(lower=0).rolling(14).mean()
    loss  = (-delta.clip(upper=0)).rolling(14).mean()
    nk["rsi14"] = 100 - (100 / (1 + gain / loss.replace(0, np.nan)))

    tr = pd.concat([
        nk["high"] - nk["low"],
        (nk["high"] - c.shift(1)).abs(),
        (nk["low"]  - c.shift(1)).abs(),
    ], axis=1).max(axis=1)
    nk["atr_pct"] = tr.rolling(14).mean() / c * 100

    for w in [5, 10, 20, 60]:
        ma = c.rolling(w).mean()
        nk[f"ma{w}"]     = ma
        nk[f"ma{w}_dev"] = (c / ma - 1) * 100
    nk["ma5_slope_pct"]  = nk["ma5"].diff(3)  / c * 100
    nk["ma20_slope_pct"] = nk["ma20"].diff(3) / c * 100

    if "usdjpy" in dfs:
        usdjpy = dfs["usdjpy"]["close"].reindex(nk.index).ffill()
        nk["usdjpy_ret_lag1"] = usdjpy.pct_change().shift(1) * 100
    else:
        nk["usdjpy_ret_lag1"] = np.nan

    if "nasdaq" in dfs:
        nasdaq = dfs["nasdaq"]["close"].reindex(nk.index).ffill()
        nk["nasdaq_ret_lag1"] = nasdaq.pct_change().shift(1) * 100
    else:
        nk["nasdaq_ret_lag1"] = np.nan

    nk["dow"]   = nk.index.dayofweek
    nk["month"] = nk.index.month

    # ── Feature 6: 直近100日Max からのボリバンタグ ──
    nk["max100"] = nk["high"].rolling(100).max()
    nk["dist_from_max100"] = (c / nk["max100"] - 1) * 100  # always <= 0

    def _bb_max_tag(bb_pct, dist):
        if pd.isna(bb_pct) or pd.isna(dist):
            return np.nan
        if bb_pct > 1.0 and dist >= -3:
            return 5  # 最高値圏BB突破
        if bb_pct > 0.8 and dist >= -3:
            return 4  # 高値圏BB上限
        if bb_pct > 0.8:
            return 3  # 高値圏中央以上
        if bb_pct < 0.0 and dist < -15:
            return 0  # 下落圏BB割れ
        if bb_pct < 0.2:
            return 1  # 下落圏BB下限
        return 2       # 中立圏

    nk["bb_max_tag"] = nk.apply(
        lambda row: _bb_max_tag(row["bb_pct"], row["dist_from_max100"]), axis=1
    )

    next_high  = nk["high"].shift(-1)
    next_low   = nk["low"].shift(-1)
    next_close = c.shift(-1)
    nk["target_high_pct"] = (next_high  / c - 1) * 100
    nk["target_low_pct"]  = (next_low   / c - 1) * 100
    nk["target_dir"]      = (next_close > c).astype(int)

    return nk

# ─────────────────────────────────────────
# sklearn 3モデル学習
# ─────────────────────────────────────────
@st.cache_resource(show_spinner=False)
def train_sklearn_models(cache_key: str, _df: pd.DataFrame, years: int):
    cutoff = _df.index[-1] - pd.DateOffset(years=years)
    df = _df[_df.index >= cutoff].dropna(subset=FEATURE_COLS + TARGETS)
    X  = df[FEATURE_COLS].values
    n  = len(df)
    sp = int(n * 0.8)
    X_tr, X_te = X[:sp], X[sp:]
    yh = df["target_high_pct"].values
    yl = df["target_low_pct"].values
    yd = df["target_dir"].values

    cb  = [lgb.early_stopping(50, verbose=False), lgb.log_evaluation(-1)]
    lgp = dict(n_estimators=500, learning_rate=0.03, num_leaves=31,
               feature_fraction=0.8, bagging_fraction=0.8, bagging_freq=5,
               random_state=42, n_jobs=-1, verbose=-1)

    mh_lgb = lgb.LGBMRegressor(**lgp)
    mh_lgb.fit(X_tr, yh[:sp], eval_set=[(X_te, yh[sp:])], callbacks=cb)
    ml_lgb = lgb.LGBMRegressor(**lgp)
    ml_lgb.fit(X_tr, yl[:sp], eval_set=[(X_te, yl[sp:])], callbacks=cb)
    md_lgb = lgb.LGBMClassifier(**lgp)
    md_lgb.fit(X_tr, yd[:sp], eval_set=[(X_te, yd[sp:])], callbacks=cb)

    eps = float(np.std(yh[:sp]) * 0.1)
    svr = lambda: Pipeline([("sc", StandardScaler()),
                             ("m",  SVR(C=5, epsilon=eps, kernel="rbf"))])
    svc = lambda: Pipeline([("sc", StandardScaler()),
                             ("m",  SVC(C=5, kernel="rbf", probability=True,
                                        random_state=42))])
    mh_svr = svr(); mh_svr.fit(X_tr, yh[:sp])
    ml_svr = svr(); ml_svr.fit(X_tr, yl[:sp])
    md_svr = svc(); md_svr.fit(X_tr, yd[:sp])

    rfr = lambda: RandomForestRegressor(n_estimators=300, max_depth=10,
                                        random_state=42, n_jobs=-1)
    rfc = lambda: RandomForestClassifier(n_estimators=300, max_depth=10,
                                         random_state=42, n_jobs=-1)
    mh_rf = rfr(); mh_rf.fit(X_tr, yh[:sp])
    ml_rf = rfr(); ml_rf.fit(X_tr, yl[:sp])
    md_rf = rfc(); md_rf.fit(X_tr, yd[:sp])

    def pack(mh, ml, md):
        ph  = mh.predict(X_te)
        pl  = ml.predict(X_te)
        pd_ = md.predict(X_te)
        prb = md.predict_proba(X_te)
        return dict(
            mh=mh, ml=ml, md=md,
            ph=ph, pl=pl, pd=pd_, prb=prb,
            mae_h=mean_absolute_error(yh[sp:], ph),
            mae_l=mean_absolute_error(yl[sp:], pl),
            acc  =accuracy_score(yd[sp:], pd_),
        )

    models = {
        "LightGBM":       pack(mh_lgb, ml_lgb, md_lgb),
        "SVR":            pack(mh_svr, ml_svr, md_svr),
        "ランダムフォレスト": pack(mh_rf,  ml_rf,  md_rf),
    }
    test_info = dict(
        dates      = df.index[sp:],
        actual_h   = yh[sp:],
        actual_l   = yl[sp:],
        actual_dir = yd[sp:],
        test_start = df.index[sp].strftime("%Y/%m/%d"),
        n_test     = n - sp,
    )
    return models, test_info

# ─────────────────────────────────────────
# LSTM 学習
# ─────────────────────────────────────────
@st.cache_resource(show_spinner=False)
def train_lstm(cache_key: str, _df: pd.DataFrame, years: int):
    if not LSTM_AVAILABLE:
        return None, None

    cutoff = _df.index[-1] - pd.DateOffset(years=years)
    df = _df[_df.index >= cutoff].dropna(subset=FEATURE_COLS + TARGETS)
    X_raw = df[FEATURE_COLS].values.astype("float32")
    yh = df["target_high_pct"].values.astype("float32")
    yl = df["target_low_pct"].values.astype("float32")
    yd = df["target_dir"].values.astype("float32")
    n  = len(df)

    scaler  = StandardScaler()
    X_sc    = scaler.fit_transform(X_raw).astype("float32")
    n_feat  = len(FEATURE_COLS)

    # シーケンス生成
    X_seq = np.array([X_sc[i - SEQ_LEN:i] for i in range(SEQ_LEN, n)],
                     dtype="float32")
    yh_s  = yh[SEQ_LEN:]
    yl_s  = yl[SEQ_LEN:]
    yd_s  = yd[SEQ_LEN:]
    n_seq = len(X_seq)
    sp    = int(n_seq * 0.8)
    X_tr, X_te = X_seq[:sp], X_seq[sp:]

    es = EarlyStopping(patience=8, restore_best_weights=True, verbose=0)

    def reg_model():
        m = Sequential([
            LSTM(64, input_shape=(SEQ_LEN, n_feat)),
            Dropout(0.2),
            Dense(32, activation="relu"),
            Dense(1),
        ])
        m.compile(optimizer=Adam(0.001), loss="mse")
        return m

    def cls_model():
        m = Sequential([
            LSTM(64, input_shape=(SEQ_LEN, n_feat)),
            Dropout(0.2),
            Dense(32, activation="relu"),
            Dense(1, activation="sigmoid"),
        ])
        m.compile(optimizer=Adam(0.001), loss="binary_crossentropy")
        return m

    mh = reg_model()
    mh.fit(X_tr, yh_s[:sp], validation_data=(X_te, yh_s[sp:]),
           epochs=50, batch_size=32, callbacks=[es], verbose=0)

    ml = reg_model()
    ml.fit(X_tr, yl_s[:sp], validation_data=(X_te, yl_s[sp:]),
           epochs=50, batch_size=32, callbacks=[es], verbose=0)

    md = cls_model()
    md.fit(X_tr, yd_s[:sp], validation_data=(X_te, yd_s[sp:]),
           epochs=50, batch_size=32, callbacks=[es], verbose=0)

    ph      = mh.predict(X_te, verbose=0).flatten()
    pl      = ml.predict(X_te, verbose=0).flatten()
    prob_up = md.predict(X_te, verbose=0).flatten()
    pd_     = (prob_up >= 0.5).astype(int)
    prb     = np.column_stack([1 - prob_up, prob_up])

    lstm_result = dict(
        mh=mh, ml=ml, md=md, scaler=scaler,
        ph=ph, pl=pl, pd=pd_, prb=prb,
        mae_h=mean_absolute_error(yh_s[sp:], ph),
        mae_l=mean_absolute_error(yl_s[sp:], pl),
        acc  =accuracy_score(yd_s[sp:], pd_),
    )
    lstm_test_info = dict(
        dates     = df.index[SEQ_LEN:][sp:],
        actual_h  = yh_s[sp:],
        actual_l  = yl_s[sp:],
        actual_dir= yd_s[sp:].astype(int),
        test_start= df.index[SEQ_LEN + sp].strftime("%Y/%m/%d"),
        n_test    = len(X_te),
    )
    return lstm_result, lstm_test_info

# ─────────────────────────────────────────
# 翌日予測
# ─────────────────────────────────────────
def predict_tomorrow_sklearn(models: dict, df: pd.DataFrame, last_close: float) -> dict:
    last = df[FEATURE_COLS].dropna().iloc[[-1]].values
    out  = {}
    for name, m in models.items():
        ph  = float(m["mh"].predict(last)[0])
        pl  = float(m["ml"].predict(last)[0])
        prb = m["md"].predict_proba(last)[0]
        pd_ = int(m["md"].predict(last)[0])
        out[name] = dict(
            high_pct=ph, low_pct=pl,
            high_yen=last_close * (1 + ph / 100),
            low_yen =last_close * (1 + pl / 100),
            dir=pd_, prob_up=float(prb[1]), prob_down=float(prb[0]),
        )
    return out

def predict_tomorrow_lstm(lstm_result, df: pd.DataFrame, last_close: float):
    if lstm_result is None:
        return None
    feat = df[FEATURE_COLS].dropna()
    if len(feat) < SEQ_LEN:
        return None
    X_sc  = lstm_result["scaler"].transform(
                feat.iloc[-SEQ_LEN:].values.astype("float32"))
    X_seq = X_sc.reshape(1, SEQ_LEN, len(FEATURE_COLS)).astype("float32")
    ph     = float(lstm_result["mh"].predict(X_seq, verbose=0)[0][0])
    pl     = float(lstm_result["ml"].predict(X_seq, verbose=0)[0][0])
    prob_up= float(lstm_result["md"].predict(X_seq, verbose=0)[0][0])
    pd_    = 1 if prob_up >= 0.5 else 0
    return dict(
        high_pct=ph, low_pct=pl,
        high_yen=last_close * (1 + ph / 100),
        low_yen =last_close * (1 + pl / 100),
        dir=pd_, prob_up=prob_up, prob_down=1 - prob_up,
    )

def build_ensemble_pred(all_preds: list, last_close: float) -> dict:
    """全モデルの予測値を平均してアンサンブル予測を生成"""
    ph  = float(np.mean([p["high_pct"] for p in all_preds]))
    pl  = float(np.mean([p["low_pct"]  for p in all_preds]))
    pu  = float(np.mean([p["prob_up"]  for p in all_preds]))
    pd_ = 1 if pu >= 0.5 else 0
    return dict(
        high_pct=ph, low_pct=pl,
        high_yen=last_close * (1 + ph / 100),
        low_yen =last_close * (1 + pl / 100),
        dir=pd_, prob_up=pu, prob_down=1 - pu,
    )

# ─────────────────────────────────────────
# アンサンブル テスト評価（sklearn 3モデルで算出）
# ─────────────────────────────────────────
def ensemble_test_metrics(models: dict, test_info: dict) -> dict:
    ph_ens  = np.mean([m["ph"] for m in models.values()], axis=0)
    pl_ens  = np.mean([m["pl"] for m in models.values()], axis=0)
    pu_ens  = np.mean([m["prb"][:, 1] for m in models.values()], axis=0)
    pd_ens  = (pu_ens >= 0.5).astype(int)
    prb_ens = np.column_stack([1 - pu_ens, pu_ens])
    return dict(
        ph=ph_ens, pl=pl_ens, pd=pd_ens, prb=prb_ens,
        mae_h=mean_absolute_error(test_info["actual_h"], ph_ens),
        mae_l=mean_absolute_error(test_info["actual_l"], pl_ens),
        acc  =accuracy_score(test_info["actual_dir"], pd_ens),
    )

# ─────────────────────────────────────────
# Feature 2: バイアス補正アンサンブル予測
# ─────────────────────────────────────────
def apply_bias_correction(pred: dict, test_info: dict, models_eval: dict,
                           last_close: float, n_days: int = 10) -> dict:
    """
    直近 n_days 間のアンサンブル予測誤差（バイアス）を計算し、
    翌日予測値を補正して返す。
    """
    n = test_info["n_test"]
    start = max(0, n - n_days)

    # アンサンブル(sklearn 3モデル)のテスト予測
    ph_ens = np.mean([m["ph"] for m in models_eval.values()
                      if "ph" in m and not callable(m["ph"])], axis=0)
    pl_ens = np.mean([m["pl"] for m in models_eval.values()
                      if "pl" in m and not callable(m["pl"])], axis=0)

    actual_h = test_info["actual_h"][start:]
    actual_l = test_info["actual_l"][start:]
    pred_h   = ph_ens[start:]
    pred_l   = pl_ens[start:]

    bias_h = float(np.mean(actual_h - pred_h))
    bias_l = float(np.mean(actual_l - pred_l))

    corr_high_pct = pred["high_pct"] + bias_h
    corr_low_pct  = pred["low_pct"]  + bias_l

    return dict(
        high_pct  = corr_high_pct,
        low_pct   = corr_low_pct,
        high_yen  = last_close * (1 + corr_high_pct / 100),
        low_yen   = last_close * (1 + corr_low_pct  / 100),
        dir       = pred["dir"],
        prob_up   = pred["prob_up"],
        prob_down = pred["prob_down"],
        bias_h    = bias_h,
        bias_l    = bias_l,
        n_days    = n_days,
    )

# ─────────────────────────────────────────
# Feature 5: 大ブレ警戒シグナル
# ─────────────────────────────────────────
def calc_swing_warning(ensemble_pred: dict, test_info: dict, models_eval: dict) -> dict:
    """
    翌日予測レンジが過去テストセットの分布の何パーセンタイルにあるかを計算する。
    80パーセンタイル以上なら大ブレ警戒。
    """
    # アンサンブルのテスト内予測レンジ
    ph_ens = np.mean([m["ph"] for m in models_eval.values()
                      if "ph" in m and not callable(m["ph"])], axis=0)
    pl_ens = np.mean([m["pl"] for m in models_eval.values()
                      if "pl" in m and not callable(m["pl"])], axis=0)
    hist_ranges = ph_ens - pl_ens  # 各テスト日のレンジ

    pred_range = ensemble_pred["high_pct"] - ensemble_pred["low_pct"]

    if len(hist_ranges) == 0:
        return dict(is_big_swing=False, percentile=50.0, pred_range=pred_range)

    percentile = float(np.mean(hist_ranges <= pred_range) * 100)
    is_big_swing = percentile >= 80

    return dict(
        is_big_swing = is_big_swing,
        percentile   = percentile,
        pred_range   = pred_range,
        hist_ranges  = hist_ranges,
    )

# ─────────────────────────────────────────
# Feature 3: 週着地テーブル
# ─────────────────────────────────────────
def build_weekly_landing(df: pd.DataFrame, n_weeks: int = 12) -> pd.DataFrame:
    """
    週次リサンプルして週着地タグ付きテーブルを生成する。
    """
    ohlc = df[["open", "high", "low", "close"]].dropna()

    weekly = ohlc.resample("W-FRI").agg({
        "open":  "first",
        "high":  "max",
        "low":   "min",
        "close": "last",
    }).dropna()

    if len(weekly) == 0:
        return pd.DataFrame()

    weekly["騰落率%"] = (weekly["close"] / weekly["open"] - 1) * 100

    def _tag(r):
        if r > 1.5:
            return "🟢 強い陽線"
        elif r > 0.3:
            return "🟡 小陽線"
        elif r > -0.3:
            return "⚪ 横ばい"
        elif r > -1.5:
            return "🟠 小陰線"
        else:
            return "🔴 強い陰線"

    weekly["タグ"] = weekly["騰落率%"].apply(_tag)

    # 当週（未完了）の判定
    last_data_date = ohlc.index[-1]
    today = pd.Timestamp.now().normalize()
    # 直近の週末（金曜日）
    days_to_fri = (4 - today.weekday()) % 7
    this_week_fri = today + pd.Timedelta(days=days_to_fri)

    rows = []
    # 最新 n_weeks 週を取得（降順）
    subset = weekly.tail(n_weeks)
    for week_end, row in subset.iloc[::-1].iterrows():
        is_current = (week_end >= last_data_date) and (week_end >= today)
        rows.append({
            "週末(金)":    week_end.strftime("%Y/%m/%d"),
            "始値":        f"¥{row['open']:,.0f}",
            "終値":        f"¥{row['close']:,.0f}",
            "高値":        f"¥{row['high']:,.0f}",
            "安値":        f"¥{row['low']:,.0f}",
            "騰落率%":     f"{row['騰落率%']:+.2f}%",
            "タグ":        row["タグ"] + (" ⚠️未確定" if is_current else ""),
        })

    return pd.DataFrame(rows)

# ─────────────────────────────────────────
# Feature 4: イベントカレンダー (upcoming)
# ─────────────────────────────────────────
def get_upcoming_events(days_ahead: int = 30) -> pd.DataFrame:
    """今後 days_ahead 日以内のFOMC/BOJ/SQイベントを返す"""
    today = datetime.now().date()
    end   = today + timedelta(days=days_ahead)

    sq_dates = get_sq_dates(today.year - 1, today.year + 2)

    events = []
    for d in FOMC_DATES:
        dt = datetime.strptime(d, "%Y-%m-%d").date()
        if today <= dt <= end:
            events.append({"日付": dt.strftime("%Y/%m/%d"), "イベント": "🇺🇸 FOMC"})
    for d in BOJ_DATES:
        dt = datetime.strptime(d, "%Y-%m-%d").date()
        if today <= dt <= end:
            events.append({"日付": dt.strftime("%Y/%m/%d"), "イベント": "🏦 日銀政策決定会合"})
    for d in sq_dates:
        dt = datetime.strptime(d, "%Y-%m-%d").date()
        if today <= dt <= end:
            events.append({"日付": dt.strftime("%Y/%m/%d"), "イベント": "📋 SQ（特別清算指数）"})

    if not events:
        return pd.DataFrame(columns=["日付", "イベント"])

    ev_df = pd.DataFrame(events).sort_values("日付").reset_index(drop=True)
    return ev_df

# ─────────────────────────────────────────
# チャート
# ─────────────────────────────────────────
def plot_bollinger(df: pd.DataFrame) -> go.Figure:
    last30 = df.tail(30)
    idx    = last30.index
    fig    = go.Figure()
    fig.add_trace(go.Scatter(
        x=list(idx) + list(idx[::-1]),
        y=list(last30["bb_upper"]) + list(last30["bb_lower"][::-1]),
        fill="toself", fillcolor="rgba(99,160,255,0.12)",
        line=dict(color="rgba(0,0,0,0)"), name="BB帯域", hoverinfo="skip",
    ))
    for col, label, dash in [
        ("bb_upper", "BB Upper +2σ", "dot"),
        ("bb_lower", "BB Lower −2σ", "dot"),
        ("bb_mid",   "BB Mid (SMA20)", "solid"),
    ]:
        fig.add_trace(go.Scatter(
            x=idx, y=last30[col],
            line=dict(color="rgba(99,160,255,0.75)", dash=dash, width=1.5),
            name=label,
        ))
    fig.add_trace(go.Candlestick(
        x=idx, open=last30["open"], high=last30["high"],
        low=last30["low"], close=last30["close"],
        name="日経225",
        increasing_line_color="#EF5350",
        decreasing_line_color="#26A69A",
    ))

    # ── Feature 4: イベントライン追加 ──
    chart_start = idx[0]
    chart_end   = idx[-1]

    sq_dates_all = get_sq_dates(chart_start.year - 1, chart_end.year + 1)

    event_lines = []
    for d in FOMC_DATES:
        dt = pd.Timestamp(d)
        if chart_start <= dt <= chart_end:
            event_lines.append((dt, "FOMC", "#5599FF", "dash"))
    for d in BOJ_DATES:
        dt = pd.Timestamp(d)
        if chart_start <= dt <= chart_end:
            event_lines.append((dt, "BOJ", "#FF9933", "dash"))
    for d in sq_dates_all:
        dt = pd.Timestamp(d)
        if chart_start <= dt <= chart_end:
            event_lines.append((dt, "SQ", "#44BB44", "dot"))

    for dt, label, color, dash_style in event_lines:
        fig.add_vline(
            x=dt.timestamp() * 1000,
            line_color=color, line_dash=dash_style, line_width=1.2,
            annotation_text=label,
            annotation_font_color=color,
            annotation_font_size=9,
            annotation_position="top",
        )

    fig.update_layout(
        title="日経225  直近30営業日 ＋ ボリンジャーバンド (20日, 2σ)　｜ 縦線: FOMC(青) BOJ(橙) SQ(緑)",
        yaxis_title="価格 (円)", height=450,
        margin=dict(l=8, r=8, t=50, b=8),
        legend=dict(orientation="h", y=-0.28, font_size=11),
        xaxis_rangeslider_visible=False,
        plot_bgcolor="#0e1117", paper_bgcolor="#0e1117", font_color="#fafafa",
    )
    return fig

def plot_scatter_pct(test_info: dict, all_eval: dict, target: str) -> go.Figure:
    actual = test_info["actual_h"] if target == "high" else test_info["actual_l"]
    label  = "高値乖離率" if target == "high" else "安値乖離率"
    colors = {"LightGBM":"#63A0FF","SVR":"#FF6B6B",
              "ランダムフォレスト":"#63FF8F","アンサンブル":"#FFD700"}
    fig = go.Figure()
    for name, ev in all_eval.items():
        pred = ev["ph"] if target == "high" else ev["pl"]
        fig.add_trace(go.Scatter(
            x=actual, y=pred, mode="markers",
            marker=dict(size=4, color=colors.get(name, "#aaa"), opacity=0.6),
            name=name,
        ))
    lim = max(float(np.abs(actual).max()), 3.0)
    fig.add_shape(type="line", x0=-lim, y0=-lim, x1=lim, y1=lim,
                  line=dict(color="white", dash="dot", width=1))
    fig.update_layout(
        title=f"{label}：予測 vs 実績 (%)",
        xaxis_title="実績 (%)", yaxis_title="予測 (%)",
        height=380, margin=dict(l=8, r=8, t=44, b=8),
        plot_bgcolor="#0e1117", paper_bgcolor="#0e1117", font_color="#fafafa",
        legend=dict(orientation="h", y=-0.25),
    )
    return fig

def plot_accuracy_bar(acc_data: dict) -> go.Figure:
    df_acc = pd.DataFrame([{"モデル": k, "正解率": v} for k, v in acc_data.items()])
    fig = px.bar(df_acc, x="モデル", y="正解率", range_y=[0, 1],
                 text=df_acc["正解率"].map("{:.1%}".format),
                 color="正解率", color_continuous_scale="Blues")
    fig.update_traces(textposition="outside")
    fig.update_layout(
        height=320, margin=dict(l=8, r=8, t=20, b=8),
        plot_bgcolor="#0e1117", paper_bgcolor="#0e1117", font_color="#fafafa",
        showlegend=False, coloraxis_showscale=False,
    )
    return fig

# ─────────────────────────────────────────
# タブ内の予測パネル（共通）
# ─────────────────────────────────────────
def render_pred_tab(title: str, pred: dict, last_close: float,
                    mae_h: float, mae_l: float, acc: float,
                    test_start: str, n_test: int) -> None:
    st.markdown(f"### {title}　翌営業日予測")
    st.caption(f"基準終値 ¥{last_close:,.0f} からの騰落率(%)と参考価格")

    c1, c2, c3 = st.columns(3)
    sh = "+" if pred["high_pct"] >= 0 else ""
    sl = "+" if pred["low_pct"]  >= 0 else ""
    with c1:
        st.metric("🔺 高値乖離率", f"{sh}{pred['high_pct']:.2f}%",
                  delta=f"参考 ¥{pred['high_yen']:,.0f}")
    with c2:
        st.metric("🔻 安値乖離率", f"{sl}{pred['low_pct']:.2f}%",
                  delta=f"参考 ¥{pred['low_yen']:,.0f}")
    with c3:
        dl = "上昇 ↑" if pred["dir"] == 1 else "下落 ↓"
        dp = pred["prob_up"] if pred["dir"] == 1 else pred["prob_down"]
        st.metric("📊 終値の方向", dl, delta=f"確率 {dp:.1%}")

    st.markdown("#### 上昇 / 下落 確率")
    p1, p2 = st.columns(2)
    with p1:
        st.markdown(f"**上昇 ↑**　`{pred['prob_up']:.1%}`")
        st.progress(pred["prob_up"])
    with p2:
        st.markdown(f"**下落 ↓**　`{pred['prob_down']:.1%}`")
        st.progress(pred["prob_down"])

    st.divider()
    st.markdown(f"#### テストデータ精度　（{test_start} ～ 直近 / {n_test} 営業日）")
    m1, m2, m3 = st.columns(3)
    with m1: st.metric("高値乖離率 MAE", f"{mae_h:.3f}%")
    with m2: st.metric("安値乖離率 MAE", f"{mae_l:.3f}%")
    with m3: st.metric("方向性 正解率",  f"{acc:.1%}")
    st.caption(
        "※ 時系列分割（前80%学習 / 後20%テスト）シャッフルなし　"
        "｜ **MAEは小さいほど誤差が少ない**　"
        "｜ **方向性正解率は50%がランダム基準、55〜60%以上で実用的**"
    )

# ─────────────────────────────────────────
# Feature 1: 過去10日テーブル作成（Hit/Missフラグ付き）
# ─────────────────────────────────────────
def build_backtest_rows(models_eval: dict, test_info: dict,
                        lstm_result, lstm_test_info) -> tuple:
    """方向性テーブルと高値/安値テーブル（Hit/Missフラグ付き）を生成"""
    n     = test_info["n_test"]
    start = max(0, n - 10)
    dates = test_info["dates"][start:]
    ah    = test_info["actual_h"][start:]
    al    = test_info["actual_l"][start:]
    ad    = test_info["actual_dir"][start:]

    # LSTM テスト結果を日付引きで取得できるように辞書化
    lstm_date_ph, lstm_date_pl, lstm_date_pd = {}, {}, {}
    if lstm_result is not None and lstm_test_info is not None:
        for i, dt in enumerate(lstm_test_info["dates"]):
            lstm_date_ph[dt] = lstm_result["ph"][i]
            lstm_date_pl[dt] = lstm_result["pl"][i]
            lstm_date_pd[dt] = int(lstm_result["pd"][i])

    dir_rows = []
    hl_rows  = []

    for i, (dt, rh, rl, rd) in enumerate(zip(dates, ah, al, ad)):
        si = start + i   # sklearn テスト内のインデックス

        # ── 方向性テーブル ──
        drow = {
            "日付":     dt.strftime("%m/%d(%a)"),
            "実際": "上昇↑" if rd == 1 else "下落↓",
        }
        ens_preds = []
        for mname, ev in models_eval.items():
            dp  = int(ev["pd"][si])
            hit = "✓" if dp == rd else "✗"
            ens_preds.append(dp)
            short = {"LightGBM":"LGB","SVR":"SVR","ランダムフォレスト":"RF","アンサンブル":"ENS"}[mname]
            drow[f"{short}({hit})"] = "上昇↑" if dp == 1 else "下落↓"
        if dt in lstm_date_pd:
            dp  = lstm_date_pd[dt]
            hit = "✓" if dp == rd else "✗"
            drow[f"LSTM({hit})"] = "上昇↑" if dp == 1 else "下落↓"
        dir_rows.append(drow)

        # ── 高値/安値テーブル（Hit/Missフラグ付き）──
        hrow = {
            "日付":       dt.strftime("%m/%d(%a)"),
            "実際 高値%":  f"{rh:+.2f}%",
            "実際 安値%":  f"{rl:+.2f}%",
        }
        for mname, ev in models_eval.items():
            short = {"LightGBM":"LGB","SVR":"SVR","ランダムフォレスト":"RF","アンサンブル":"ENS"}[mname]
            ph_val = ev["ph"][si]
            pl_val = ev["pl"][si]
            # Hit判定: 実際の高値が予測高値以上なら到達✓
            high_hit = "✓" if rh >= ph_val else "✗"
            # Hit判定: 実際の安値が予測安値以下なら到達✓
            low_hit  = "✓" if rl <= pl_val else "✗"
            hrow[f"{short} 高値%({high_hit})"] = f"{ph_val:+.2f}%"
            hrow[f"{short} 安値%({low_hit})"]  = f"{pl_val:+.2f}%"
        if dt in lstm_date_ph:
            ph_val = lstm_date_ph[dt]
            pl_val = lstm_date_pl[dt]
            high_hit = "✓" if rh >= ph_val else "✗"
            low_hit  = "✓" if rl <= pl_val else "✗"
            hrow[f"LSTM 高値%({high_hit})"] = f"{ph_val:+.2f}%"
            hrow[f"LSTM 安値%({low_hit})"]  = f"{pl_val:+.2f}%"
        hl_rows.append(hrow)

    # 最新日付を先頭に（降順）
    return (pd.DataFrame(dir_rows[::-1]),
            pd.DataFrame(hl_rows[::-1]))


def calc_hit_rate_summary(models_eval: dict, test_info: dict,
                          lstm_result, lstm_test_info) -> pd.DataFrame:
    """
    Feature 1: 各モデルの高値/安値Hit率と平均乖離（バイアス）を計算
    """
    n     = test_info["n_test"]
    start = max(0, n - 10)
    ah    = test_info["actual_h"][start:]
    al    = test_info["actual_l"][start:]
    dates = test_info["dates"][start:]

    lstm_date_ph, lstm_date_pl = {}, {}
    if lstm_result is not None and lstm_test_info is not None:
        for i, dt in enumerate(lstm_test_info["dates"]):
            lstm_date_ph[dt] = lstm_result["ph"][i]
            lstm_date_pl[dt] = lstm_result["pl"][i]

    rows = []
    for mname, ev in models_eval.items():
        ph_arr = ev["ph"][start:]
        pl_arr = ev["pl"][start:]
        n10    = len(ah)
        high_hits = int(np.sum(ah >= ph_arr))
        low_hits  = int(np.sum(al <= pl_arr))
        bias_h    = float(np.mean(ah - ph_arr))
        bias_l    = float(np.mean(al - pl_arr))
        rows.append({
            "モデル":         mname,
            "高値到達率":     f"{high_hits}/{n10} ({high_hits/n10:.0%})",
            "安値到達率":     f"{low_hits}/{n10} ({low_hits/n10:.0%})",
            "高値平均乖離":   f"{bias_h:+.3f}%",
            "安値平均乖離":   f"{bias_l:+.3f}%",
        })

    # LSTM
    if lstm_result is not None and lstm_test_info is not None:
        ph_list, pl_list, ah_list, al_list = [], [], [], []
        for dt, rh, rl in zip(dates, ah, al):
            if dt in lstm_date_ph:
                ph_list.append(lstm_date_ph[dt])
                pl_list.append(lstm_date_pl[dt])
                ah_list.append(rh)
                al_list.append(rl)
        if ph_list:
            n_l = len(ph_list)
            ph_arr_l = np.array(ph_list)
            pl_arr_l = np.array(pl_list)
            ah_arr_l = np.array(ah_list)
            al_arr_l = np.array(al_list)
            high_hits_l = int(np.sum(ah_arr_l >= ph_arr_l))
            low_hits_l  = int(np.sum(al_arr_l <= pl_arr_l))
            bias_h_l    = float(np.mean(ah_arr_l - ph_arr_l))
            bias_l_l    = float(np.mean(al_arr_l - pl_arr_l))
            rows.append({
                "モデル":         "LSTM",
                "高値到達率":     f"{high_hits_l}/{n_l} ({high_hits_l/n_l:.0%})",
                "安値到達率":     f"{low_hits_l}/{n_l} ({low_hits_l/n_l:.0%})",
                "高値平均乖離":   f"{bias_h_l:+.3f}%",
                "安値平均乖離":   f"{bias_l_l:+.3f}%",
            })

    return pd.DataFrame(rows)


# ─────────────────────────────────────────
# メインUI
# ─────────────────────────────────────────
st.title("📈 日経平均 翌日予測")
st.caption(
    f"データ: Yahoo Finance　|　モデル: LightGBM / SVR / RF / アンサンブル / LSTM"
    f"　|　更新: {datetime.now().strftime('%Y/%m/%d %H:%M')}"
)

# ── サイドバー ──
with st.sidebar:
    st.header("⚙️ 設定")
    train_years = st.slider("学習期間（年）", 1, 10, 3, 1,
                             help="直近N年分のデータで学習。短いほど直近トレンド重視")
    st.divider()
    if st.button("🔄 最新データに更新", use_container_width=True,
                 help="Yahoo Finance から最新データを再取得します（キャッシュクリア）"):
        st.cache_data.clear()
        st.rerun()
    st.caption(f"最終取得: {datetime.now().strftime('%Y/%m/%d %H:%M')}　※1時間ごと自動更新")
    st.divider()
    st.markdown("""
**指標の見方**
- **高値/安値乖離率**: 前日終値を0%とした騰落率
- **参考価格**: 乖離率から逆算した円換算（目安）
- **方向性正解率**: 50%=ランダム、55〜60%以上で実用的
- **MAE**: 小さいほど誤差が少ない

⚠️ 投資判断には使用しないでください
""")

# ── データ取得 ──
with st.spinner("データを取得しています..."):
    dfs = fetch_data()

if "nikkei" not in dfs or dfs["nikkei"].empty:
    st.error("日経平均データを取得できませんでした。")
    st.stop()

with st.expander("取得データ概要", expanded=False):
    rows = []
    for label, key in [("日経225","nikkei"),("日経先物","nikkei_fut"),
                        ("NASDAQ","nasdaq"),("USD/JPY","usdjpy")]:
        d = dfs.get(key)
        if d is not None and not d.empty:
            rows.append({"銘柄":label, "開始":d.index[0].strftime("%Y/%m/%d"),
                          "終了":d.index[-1].strftime("%Y/%m/%d"), "日数":len(d)})
        else:
            rows.append({"銘柄":label, "開始":"N/A", "終了":"N/A", "日数":0})
    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

# ── 特徴量構築 ──
df = build_features(dfs)
last_close = float(df["close"].dropna().iloc[-1])
last_date  = df["close"].dropna().index[-1].strftime("%Y/%m/%d")
st.info(f"直近終値: **¥{last_close:,.0f}**　（{last_date}）← 乖離率の基準値")

# ── モデル学習 ──
ck = df["close"].dropna().index[-1].strftime("%Y%m%d") + f"_y{train_years}"
with st.spinner(f"直近{train_years}年データで LightGBM / SVR / ランダムフォレスト を学習中..."):
    sklearn_models, test_info = train_sklearn_models(ck, df, train_years)

with st.spinner("LSTM を学習中...（初回のみ時間がかかります）"):
    lstm_result, lstm_test_info = train_lstm(ck, df, train_years)

# ── アンサンブル評価（sklearn 3モデル平均）──
ens_eval = ensemble_test_metrics(sklearn_models, test_info)

# ── 全評価モデル（散布図・テーブル用）──
all_eval = {**sklearn_models, "アンサンブル": ens_eval}

# ── 翌日予測 ──
sklearn_preds  = predict_tomorrow_sklearn(sklearn_models, df, last_close)
lstm_pred      = predict_tomorrow_lstm(lstm_result, df, last_close)
all_pred_list  = list(sklearn_preds.values()) + ([lstm_pred] if lstm_pred else [])
ensemble_pred  = build_ensemble_pred(all_pred_list, last_close)

# ── Feature 2: バイアス補正予測 ──
bias_corrected = apply_bias_correction(ensemble_pred, test_info, sklearn_models, last_close)

# ── Feature 5: 大ブレ警戒シグナル ──
swing_warn = calc_swing_warning(ensemble_pred, test_info, sklearn_models)

# ── 現在のbb_max_tag ──
bb_max_tag_last = df["bb_max_tag"].dropna().iloc[-1] if "bb_max_tag" in df.columns and df["bb_max_tag"].dropna().shape[0] > 0 else np.nan
dist_from_max100_last = df["dist_from_max100"].dropna().iloc[-1] if "dist_from_max100" in df.columns and df["dist_from_max100"].dropna().shape[0] > 0 else np.nan

# ─────────────────────────────────────────
# タブ (10つ)
# ─────────────────────────────────────────
(tab_dash, tab_dir, tab_hl, tab_week, tab_lgb, tab_svr, tab_rf,
 tab_ens, tab_lstm, tab_bb) = st.tabs([
    "🏠 ダッシュボード",
    "🔍 過去10日（方向性）",
    "📈 過去10日（高値/安値）",
    "📅 週着地",
    "🤖 LightGBM",
    "📐 SVR",
    "🌲 ランダムフォレスト",
    "🔗 アンサンブル",
    "🧠 LSTM",
    "📊 ボリンジャーバンド",
])

# ════════════════════════════════════════
# ── ダッシュボードタブ ──
# ════════════════════════════════════════
with tab_dash:
    st.markdown(f"### 翌営業日の予測まとめ　（基準終値 ¥{last_close:,.0f} / {last_date}）")

    # ── 1. 大ブレ警戒バナー + ボリバンタグ ──
    col_warn, col_bbtag = st.columns([2, 1])
    with col_warn:
        if swing_warn["is_big_swing"]:
            st.markdown(
                f"<div style='background:#5c1a1a;border:1px solid #EF5350;"
                f"border-radius:6px;padding:10px 14px;margin-bottom:6px;'>"
                f"⚠️ <b>大ブレ警戒</b>: 予測レンジが過去テスト比 上位"
                f"<b>{swing_warn['percentile']:.0f}%ile</b>"
                f" （予測レンジ: {swing_warn['pred_range']:.2f}%）"
                f"</div>",
                unsafe_allow_html=True,
            )
            st.progress(min(swing_warn["percentile"] / 100, 1.0),
                        text=f"過去テスト内パーセンタイル: {swing_warn['percentile']:.0f}%")
        else:
            st.success(
                f"✅ レンジ警戒なし　予測レンジ: {swing_warn['pred_range']:.2f}%"
                f"（過去比 {swing_warn['percentile']:.0f}%ile）"
            )
    with col_bbtag:
        if not pd.isna(bb_max_tag_last):
            tag_key = int(bb_max_tag_last)
            tag_name, tag_desc = BB_MAX_TAG_INFO.get(tag_key, ("不明", ""))
            st.markdown(f"**現在のボリバンタグ**")
            st.markdown(f"### {tag_name}")
            st.caption(tag_desc)
        else:
            st.markdown("**ボリバンタグ**: データ不足")

    st.divider()

    # ── 2. 直近・今後のイベント ──
    st.markdown("#### 📅 直近・今後のイベント（30日以内）")
    upcoming_df = get_upcoming_events(30)
    if upcoming_df.empty:
        st.info("今後30日以内に主要イベントはありません。")
    else:
        st.dataframe(upcoming_df, use_container_width=True, hide_index=True)

    st.divider()

    # ── 3. 高値・安値 棒グラフ ──
    all_tomorrow = {
        "LightGBM":       sklearn_preds["LightGBM"],
        "SVR":            sklearn_preds["SVR"],
        "ランダムフォレスト": sklearn_preds["ランダムフォレスト"],
        "アンサンブル":    ensemble_pred,
    }
    if lstm_pred:
        all_tomorrow["LSTM"] = lstm_pred

    model_names = list(all_tomorrow.keys())

    st.markdown(f"#### 予測 高値 / 安値　（前日終値 ¥{last_close:,.0f} 基準）")
    fig_dash = go.Figure()
    high_deltas = [all_tomorrow[m]["high_yen"] - last_close for m in model_names]
    low_deltas  = [all_tomorrow[m]["low_yen"]  - last_close for m in model_names]
    high_yens   = [all_tomorrow[m]["high_yen"] for m in model_names]
    low_yens    = [all_tomorrow[m]["low_yen"]  for m in model_names]
    fig_dash.add_trace(go.Bar(
        name="予測 高値（前日比）", x=model_names, y=high_deltas,
        marker_color="#EF5350",
        text=[f"+¥{v:,.0f}<br>(¥{y:,.0f})" for v, y in zip(high_deltas, high_yens)],
        textposition="outside",
    ))
    fig_dash.add_trace(go.Bar(
        name="予測 安値（前日比）", x=model_names, y=low_deltas,
        marker_color="#26A69A",
        text=[f"¥{v:,.0f}<br>(¥{y:,.0f})" for v, y in zip(low_deltas, low_yens)],
        textposition="outside",
    ))
    # ボリンジャーバンド水平線
    bb_last = df[["bb_upper", "bb_mid", "bb_lower"]].dropna().iloc[-1]
    bb_upper_d = bb_last["bb_upper"] - last_close
    bb_mid_d   = bb_last["bb_mid"]   - last_close
    bb_lower_d = bb_last["bb_lower"] - last_close
    for val, label, color in [
        (bb_upper_d, f"BB Upper ¥{bb_last['bb_upper']:,.0f}", "rgba(99,160,255,0.9)"),
        (bb_mid_d,   f"BB Mid   ¥{bb_last['bb_mid']:,.0f}",  "rgba(99,160,255,0.6)"),
        (bb_lower_d, f"BB Lower ¥{bb_last['bb_lower']:,.0f}","rgba(99,160,255,0.9)"),
    ]:
        fig_dash.add_hline(
            y=val, line_color=color, line_dash="dash", line_width=1.2,
            annotation_text=label, annotation_position="top left",
            annotation_font_color=color,
        )
    fig_dash.add_hline(y=0, line_color="white", line_dash="dot", line_width=1.5,
                       annotation_text=f"前日終値 ¥{last_close:,.0f}",
                       annotation_position="top right")
    fig_dash.update_layout(
        barmode="group",
        yaxis_title="前日終値比 (円)",
        yaxis=dict(tickformat="+,.0f", zeroline=True, zerolinecolor="white",
                   zerolinewidth=1),
        height=440, margin=dict(l=8, r=8, t=20, b=8),
        legend=dict(orientation="h", y=-0.2),
        plot_bgcolor="#0e1117", paper_bgcolor="#0e1117", font_color="#fafafa",
    )
    st.plotly_chart(fig_dash, use_container_width=True)

    st.divider()

    # ── 4. バイアス補正アンサンブル予測 ──
    st.markdown("#### 🎯 バイアス補正アンサンブル予測")
    st.caption(
        f"直近{bias_corrected['n_days']}日の予測誤差（バイアス）を補正した予測値"
    )
    bc1, bc2, bc3, bc4 = st.columns(4)
    sh_bc = "+" if bias_corrected["high_pct"] >= 0 else ""
    sl_bc = "+" if bias_corrected["low_pct"]  >= 0 else ""
    with bc1:
        st.metric("🔺 補正後 高値乖離率",
                  f"{sh_bc}{bias_corrected['high_pct']:.2f}%",
                  delta=f"参考 ¥{bias_corrected['high_yen']:,.0f}")
    with bc2:
        st.metric("🔻 補正後 安値乖離率",
                  f"{sl_bc}{bias_corrected['low_pct']:.2f}%",
                  delta=f"参考 ¥{bias_corrected['low_yen']:,.0f}")
    sh_bias = "+" if bias_corrected["bias_h"] >= 0 else ""
    sl_bias = "+" if bias_corrected["bias_l"] >= 0 else ""
    with bc3:
        st.metric("高値バイアス補正値",
                  f"{sh_bias}{bias_corrected['bias_h']:.3f}%",
                  help="正=実際が予測より高かった（過去平均）")
    with bc4:
        st.metric("安値バイアス補正値",
                  f"{sl_bias}{bias_corrected['bias_l']:.3f}%",
                  help="負=実際が予測より低かった（過去平均）")

    st.divider()

    # ── 5. 上昇確率 横並び ──
    st.markdown("#### 上昇確率　各モデル比較")
    cols = st.columns(len(all_tomorrow))
    for col, (mname, p) in zip(cols, all_tomorrow.items()):
        with col:
            color = "🟢" if p["prob_up"] >= 0.5 else "🔴"
            st.markdown(f"**{mname}**")
            st.markdown(f"{color} 上昇 `{p['prob_up']:.1%}`")
            st.progress(p["prob_up"])

    st.divider()

    # ── 6. 予測まとめテーブル（バイアス補正行追加）──
    st.markdown("#### 予測一覧テーブル")
    summary_rows = []
    for mname, p in all_tomorrow.items():
        dir_label = "上昇 ↑" if p["dir"] == 1 else "下落 ↓"
        dprob = p["prob_up"] if p["dir"] == 1 else p["prob_down"]
        summary_rows.append({
            "モデル":   mname,
            "予測 高値": f"¥{p['high_yen']:,.0f}",
            "予測 安値": f"¥{p['low_yen']:,.0f}",
            "終値方向":  dir_label,
            "確率":     f"{dprob:.1%}",
        })
    # バイアス補正アンサンブル行
    bc_dir_label = "上昇 ↑" if bias_corrected["dir"] == 1 else "下落 ↓"
    bc_dprob = bias_corrected["prob_up"] if bias_corrected["dir"] == 1 else bias_corrected["prob_down"]
    summary_rows.append({
        "モデル":   "バイアス補正アンサンブル",
        "予測 高値": f"¥{bias_corrected['high_yen']:,.0f}",
        "予測 安値": f"¥{bias_corrected['low_yen']:,.0f}",
        "終値方向":  bc_dir_label,
        "確率":     f"{bc_dprob:.1%}",
    })
    st.dataframe(
        pd.DataFrame(summary_rows).set_index("モデル"),
        use_container_width=True,
    )

# ════════════════════════════════════════
# ── 各モデルタブ ──
# ════════════════════════════════════════
for tab, name, pred in [
    (tab_lgb, "LightGBM",       sklearn_preds["LightGBM"]),
    (tab_svr, "SVR",             sklearn_preds["SVR"]),
    (tab_rf,  "ランダムフォレスト", sklearn_preds["ランダムフォレスト"]),
]:
    with tab:
        m = sklearn_models[name]
        render_pred_tab(name, pred, last_close,
                        m["mae_h"], m["mae_l"], m["acc"],
                        test_info["test_start"], test_info["n_test"])

# ── アンサンブルタブ ──
with tab_ens:
    n_models = len(all_pred_list)
    render_pred_tab(
        f"アンサンブル（{n_models}モデル平均）",
        ensemble_pred, last_close,
        ens_eval["mae_h"], ens_eval["mae_l"], ens_eval["acc"],
        test_info["test_start"], test_info["n_test"],
    )
    st.caption(f"テスト評価: LightGBM / SVR / ランダムフォレスト の3モデル平均　"
               f"| 翌日予測: {n_models}モデル平均（LSTM{'含む' if lstm_pred else '除く'}）")

# ── LSTM タブ ──
with tab_lstm:
    if not LSTM_AVAILABLE:
        st.warning("TensorFlow が見つかりません。`pip install tensorflow` を実行してください。")
    elif lstm_result is None or lstm_pred is None:
        st.warning("LSTM の学習に失敗しました。データ量を確認してください。")
    else:
        render_pred_tab(
            f"LSTM（参照日数 {SEQ_LEN}日）",
            lstm_pred, last_close,
            lstm_result["mae_h"], lstm_result["mae_l"], lstm_result["acc"],
            lstm_test_info["test_start"], lstm_test_info["n_test"],
        )

# ════════════════════════════════════════
# ── ボリンジャーバンドタブ ──
# ════════════════════════════════════════
with tab_bb:
    st.subheader("ボリンジャーバンド（直近30営業日）")

    # Feature 6: bb_max_tag 表示
    if not pd.isna(bb_max_tag_last):
        tag_key  = int(bb_max_tag_last)
        tag_name, tag_desc = BB_MAX_TAG_INFO.get(tag_key, ("不明", ""))
        col_t1, col_t2, col_t3 = st.columns(3)
        with col_t1:
            st.markdown("**現在のボリバンタグ (bb_max_tag)**")
            st.markdown(f"## {tag_name}")
            st.caption(tag_desc)
        with col_t2:
            st.metric("bb_max_tag 番号", f"{tag_key}")
        with col_t3:
            if not pd.isna(dist_from_max100_last):
                st.metric(
                    "100日高値からの乖離 (dist_from_max100)",
                    f"{dist_from_max100_last:.2f}%",
                    help="0%=直近100日高値と同水準、負=それ以下"
                )
        st.divider()

    st.plotly_chart(plot_bollinger(df), use_container_width=True)

    with st.expander("直近10日 数値データ"):
        dcols = ["open","high","low","close","bb_upper","bb_mid","bb_lower",
                 "rsi14","atr_pct","dist_from_max100","bb_max_tag"]
        available_cols = [c for c in dcols if c in df.columns]
        st.dataframe(df[available_cols].tail(10).round(2), use_container_width=True)

    # BB_MAX_TAG_INFO 凡例
    with st.expander("ボリバンタグ (bb_max_tag) 凡例"):
        tag_rows = [{"タグ番号": k, "名称": v[0], "説明": v[1]}
                    for k, v in sorted(BB_MAX_TAG_INFO.items(), reverse=True)]
        st.dataframe(pd.DataFrame(tag_rows), use_container_width=True, hide_index=True)

# ════════════════════════════════════════
# ── 過去10日（方向性）タブ ──
# ════════════════════════════════════════
with tab_dir:
    st.subheader("過去10営業日の予測 vs 実績（方向性）")
    st.caption("✓=正解 / ✗=不正解　｜　テストセット末尾10件（学習データ非含有）")

    dir_df, hl_df = build_backtest_rows(all_eval, test_info, lstm_result, lstm_test_info)
    st.dataframe(dir_df.set_index("日付"), use_container_width=True)

    st.markdown("---")
    st.markdown("#### 方向性正解率（過去10営業日）")
    n   = test_info["n_test"]
    acc_data = {}
    for mname, ev in all_eval.items():
        p10 = ev["pd"][max(0, n - 10):]
        a10 = test_info["actual_dir"][max(0, n - 10):]
        acc_data[mname] = accuracy_score(a10, p10)
    if lstm_result is not None and lstm_test_info is not None:
        lstm_map = {dt: i for i, dt in enumerate(lstm_test_info["dates"])}
        last10_dates = test_info["dates"][max(0, n - 10):]
        lp, la = [], []
        for dt, act in zip(last10_dates, test_info["actual_dir"][max(0, n - 10):]):
            if dt in lstm_map:
                lp.append(lstm_result["pd"][lstm_map[dt]])
                la.append(act)
        if lp:
            acc_data["LSTM"] = accuracy_score(la, lp)

    st.plotly_chart(plot_accuracy_bar(acc_data), use_container_width=True)
    st.info(
        "**方向性正解率の見方**\n\n"
        "- **50%** = ランダムと同等（コイントスと変わらない）\n"
        "- **55%以上** = 実用的な予測精度の目安\n"
        "- **60%以上** = かなり高い予測精度とされる\n\n"
        "過去10日のサンプルは少ないため、全期間の正解率も合わせて参照してください。"
    )

# ════════════════════════════════════════
# ── 過去10日（高値/安値）タブ ──
# ════════════════════════════════════════
with tab_hl:
    st.subheader("過去10営業日の予測 vs 実績（高値/安値乖離率）")
    st.caption(
        "前日終値からの乖離率(%)で比較　｜　テストセット末尾10件（学習データ非含有）\n"
        "✓=到達（高値: 実際≥予測 / 安値: 実際≤予測）　✗=未到達"
    )
    st.dataframe(hl_df.set_index("日付"), use_container_width=True)

    # Feature 1: Hit率サマリー
    st.markdown("---")
    st.markdown("#### 予測レンジ Hit率 & バイアスサマリー（過去10日）")
    hit_df = calc_hit_rate_summary(all_eval, test_info, lstm_result, lstm_test_info)
    st.dataframe(hit_df.set_index("モデル"), use_container_width=True)
    st.caption(
        "**高値到達率**: 実際の高値が予測高値以上だった日の割合\n"
        "**安値到達率**: 実際の安値が予測安値以下だった日の割合\n"
        "**平均乖離（バイアス）**: 正＝実際が予測より高い（過小予測）、負＝実際が予測より低い（過大予測）"
    )

    st.markdown("---")
    st.markdown("#### 予測 vs 実績 散布図（テスト全期間）")
    s1, s2 = st.columns(2)
    with s1:
        st.plotly_chart(plot_scatter_pct(test_info, all_eval, "high"),
                        use_container_width=True)
    with s2:
        st.plotly_chart(plot_scatter_pct(test_info, all_eval, "low"),
                        use_container_width=True)

    st.markdown("---")
    st.markdown("#### モデル精度比較（テスト全期間）")
    st.info(
        "**指標の見方**\n\n"
        "- **MAE（平均絶対誤差）は小さいほど良い** — 予測が平均何%ずれているかを示します。\n"
        "  例）日経平均 ¥38,000 のとき: MAE 0.5% → 平均 ±190円のずれ、MAE 0.2% → 平均 ±76円のずれ\n\n"
        "- **方向性正解率は大きいほど良い** — 50%はランダムと同等。55〜60%以上で実用的とされます。\n"
        "  MAEが小さくても方向性が外れることはあるため、両方を合わせて評価してください。"
    )
    mae_rows = []
    for mname, ev in all_eval.items():
        p10h = ev["ph"][max(0, n - 10):]
        p10l = ev["pl"][max(0, n - 10):]
        a10h = test_info["actual_h"][max(0, n - 10):]
        a10l = test_info["actual_l"][max(0, n - 10):]
        mae_rows.append({
            "モデル":            mname,
            "高値MAE 全期間":    f"{ev['mae_h']:.3f}%",
            "高値MAE 過去10日":  f"{mean_absolute_error(a10h, p10h):.3f}%",
            "安値MAE 全期間":    f"{ev['mae_l']:.3f}%",
            "安値MAE 過去10日":  f"{mean_absolute_error(a10l, p10l):.3f}%",
            "方向正解率 全期間": f"{ev['acc']:.1%}",
        })
    st.dataframe(pd.DataFrame(mae_rows).set_index("モデル"),
                 use_container_width=True)

# ════════════════════════════════════════
# ── 週着地タブ（Feature 3）──
# ════════════════════════════════════════
with tab_week:
    st.subheader("週着地タグ（直近12週）")
    st.caption(
        "週次（月〜金）のOHLCを集計し、週の騰落パターンをタグ付け\n"
        "🟢 強い陽線(>+1.5%) | 🟡 小陽線(>+0.3%) | ⚪ 横ばい | 🟠 小陰線(>-1.5%) | 🔴 強い陰線"
    )

    weekly_df = build_weekly_landing(df, n_weeks=12)
    if weekly_df.empty:
        st.warning("週次データを生成できませんでした。データ量を確認してください。")
    else:
        st.dataframe(weekly_df, use_container_width=True, hide_index=True)

    st.markdown("---")
    st.markdown("#### 週次騰落率ヒストグラム（直近24週）")
    ohlc_wk = df[["open", "high", "low", "close"]].dropna()
    weekly_all = ohlc_wk.resample("W-FRI").agg({
        "open":  "first",
        "high":  "max",
        "low":   "min",
        "close": "last",
    }).dropna().tail(24)
    if not weekly_all.empty:
        weekly_all["騰落率%"] = (weekly_all["close"] / weekly_all["open"] - 1) * 100
        fig_wk = go.Figure()
        colors_wk = ["#EF5350" if r >= 0 else "#26A69A" for r in weekly_all["騰落率%"]]
        fig_wk.add_trace(go.Bar(
            x=[d.strftime("%m/%d") for d in weekly_all.index],
            y=weekly_all["騰落率%"].values,
            marker_color=colors_wk,
            name="週次騰落率%",
        ))
        fig_wk.add_hline(y=0, line_color="white", line_dash="dot", line_width=1)
        fig_wk.add_hline(y=1.5,  line_color="#EF5350", line_dash="dash", line_width=0.8,
                         annotation_text="+1.5%", annotation_position="top right")
        fig_wk.add_hline(y=-1.5, line_color="#26A69A", line_dash="dash", line_width=0.8,
                         annotation_text="-1.5%", annotation_position="bottom right")
        fig_wk.update_layout(
            yaxis_title="週次騰落率 (%)",
            height=350, margin=dict(l=8, r=8, t=20, b=8),
            plot_bgcolor="#0e1117", paper_bgcolor="#0e1117", font_color="#fafafa",
        )
        st.plotly_chart(fig_wk, use_container_width=True)

st.markdown("---")
st.caption("免責事項: このアプリはデモ・教育目的のみです。投資判断には使用しないでください。")
