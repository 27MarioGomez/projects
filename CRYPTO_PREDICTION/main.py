import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, GRU, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.regularizers import l2

# Activate mixed precision if GPU is available (ignored on CPU)
if tf.config.list_physical_devices('GPU'):
    from tensorflow.keras.mixed_precision import set_global_policy
    set_global_policy('mixed_float16')

import yfinance as yf
import requests
import certifi
import os
import shutil
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import ElasticNetCV
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
import torch
from joblib import Parallel, delayed
from textblob import TextBlob
from dateutil.parser import parse as date_parse
from urllib3.util.retry import Retry
from requests.adapters import HTTPAdapter
from newsapi import NewsApiClient
from prophet import Prophet

# Essential technical indicators
from ta.momentum import RSIIndicator
from ta.trend import MACD, SMAIndicator, EMAIndicator
from ta.volatility import BollingerBands, AverageTrueRange
from ta.volume import OnBalanceVolumeIndicator
from transformers.pipelines import pipeline
from xgboost import XGBRegressor
from numba import njit

# Try to import Keras Tuner (installed as keras-tuner)
try:
    import keras_tuner as kt
except ModuleNotFoundError:
    import kerastuner as kt

# Set up the page configuration for the dashboard
st.set_page_config(page_title="Crypto Price Predictions 🔮", layout="wide")

# ---------------------------------------------------------------------
# Global Definitions
# ---------------------------------------------------------------------
coincap_ids = {
    "Bitcoin (BTC)": "bitcoin",
    "Ethereum (ETH)": "ethereum",
    "Ripple (XRP)": "xrp",
    "Binance Coin (BNB)": "binance-coin",
    "Cardano (ADA)": "cardano",
    "Solana (SOL)": "solana",
    "Dogecoin (DOGE)": "dogecoin",
    "Polkadot (DOT)": "polkadot",
    "Polygon (MATIC)": "polygon",
    "Litecoin (LTC)": "litecoin",
    "TRON (TRX)": "tron",
    "Stellar (XLM)": "stellar"
}
coinid_to_symbol = {v: k.split(" (")[1][:-1] for k, v in coincap_ids.items()}

# ---------------------------------------------------------------------
# Transformer to preserve DataFrame structure during imputation
# ---------------------------------------------------------------------
class DataFrameTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, transformer):
        self.transformer = transformer
    def fit(self, X, y=None):
        self.transformer.fit(X, y)
        return self
    def transform(self, X):
        X_trans = self.transformer.transform(X)
        if isinstance(X, pd.DataFrame):
            return pd.DataFrame(X_trans, columns=X.columns, index=X.index)
        return X_trans

# ---------------------------------------------------------------------
# Feature Selection Transformer
# ---------------------------------------------------------------------
class FeatureSelector(BaseEstimator, TransformerMixin):
    """
    Performs automatic feature selection using ElasticNetCV,
    refining the result with feature importance computed by XGBoost.
    """
    def __init__(self, feature_cols, target_col, enet_threshold=0.01, importance_threshold=0.01):
        self.feature_cols = feature_cols
        self.target_col = target_col
        self.enet_threshold = enet_threshold
        self.importance_threshold = importance_threshold
        self.selected_features_ = None

    def fit(self, X, y=None):
        df = X.copy()
        if y is not None:
            df[self.target_col] = y
        y_arr = df[self.target_col].values
        X_arr = df[self.feature_cols].values

        # ElasticNetCV
        enet = ElasticNetCV(cv=5, random_state=42).fit(X_arr, y_arr)
        coefs = enet.coef_
        initial_selected = [
            self.feature_cols[i]
            for i in range(len(self.feature_cols))
            if abs(coefs[i]) > self.enet_threshold
        ]
        if not initial_selected:
            initial_selected = self.feature_cols

        # XGBoost
        xgb = XGBRegressor(n_estimators=50, max_depth=3, learning_rate=0.1, random_state=42)
        xgb.fit(df[initial_selected].values, y_arr)
        importances = xgb.feature_importances_
        refined = [
            initial_selected[i]
            for i in range(len(initial_selected))
            if importances[i] > self.importance_threshold
        ]
        self.selected_features_ = refined if refined else initial_selected
        return self

    def transform(self, X):
        return X[self.selected_features_]

# ---------------------------------------------------------------------
# Sentiment Analysis Functions
# ---------------------------------------------------------------------
@st.cache_resource(show_spinner=False)
def load_sentiment_pipeline():
    hf_token = st.secrets.get("hf_token")
    if hf_token:
        os.environ["HUGGINGFACE_HUB_TOKEN"] = hf_token
    return pipeline(
        "sentiment-analysis",
        model="distilbert-base-uncased-finetuned-sst-2-english",
        revision="714eb0f"
    )

def get_advanced_sentiment(text):
    pipe = load_sentiment_pipeline()
    result = pipe(text)[0]
    return 50 + (result["score"] * 50) if result["label"].upper() == "POSITIVE" else 50 - (result["score"] * 50)

@st.cache_data(ttl=43200)
def get_newsapi_articles(coin_id, show_warning=True):
    newsapi_key = st.secrets.get("newsapi_key", "")
    if not newsapi_key:
        st.error("News API key not found in secrets.")
        return []
    try:
        query = f"{coin_id} crypto"
        newsapi = NewsApiClient(api_key=newsapi_key)
        data = newsapi.get_everything(q=query, language="en", sort_by="relevancy", page_size=5)
        articles = []
        if data.get("articles"):
            for art in data["articles"]:
                image_url = art.get("urlToImage", "")
                title = art.get("title") or "No title"
                description = art.get("description") or "No description"
                pub_date = art.get("publishedAt") or "Date not available"
                link = art.get("url") or "#"
                try:
                    parsed_date = date_parse(pub_date)
                except:
                    parsed_date = datetime(1970, 1, 1)
                pub_date_str = parsed_date.strftime("%Y-%m-%d %H:%M:%S")
                articles.append({
                    "title": title,
                    "description": description,
                    "pubDate": pub_date_str,
                    "link": link,
                    "image": image_url,
                    "parsed_date": parsed_date
                })
            articles = sorted(articles, key=lambda x: x["parsed_date"], reverse=True)
        return articles
    except Exception as e:
        if ("rateLimited" in str(e) or "429" in str(e)) and show_warning:
            st.warning("Request rate limit exceeded. Please try again in 12 hours.")
        elif show_warning:
            st.error(f"Error fetching news: {e}")
        return []

def get_news_sentiment(coin_id):
    articles = get_newsapi_articles(coin_id, show_warning=False)
    if not articles:
        return 50.0
    sentiments_tb = []
    sentiments_trans = []
    for article in articles:
        text = (article["title"] or "") + " " + (article["description"] or "")
        blob = TextBlob(text)
        polarity_tb = blob.sentiment.polarity
        sentiments_tb.append(50 + (polarity_tb * 50))
        sentiments_trans.append(get_advanced_sentiment(text))
    return (np.mean(sentiments_tb) + np.mean(sentiments_trans)) / 2.0

def get_fear_greed_index():
    try:
        data = requests.get("https://api.alternative.me/fng/?format=json", timeout=10).json()
        return float(data["data"][0]["value"])
    except Exception:
        st.warning("Fear & Greed Index not available. Using default value 50.0.")
        return 50.0

def get_crypto_sentiment_combined(coin_id):
    news_sent = get_news_sentiment(coin_id)
    market_sent = get_fear_greed_index()
    gauge_val = 50 + (news_sent - market_sent)
    return news_sent, market_sent, max(0, min(100, gauge_val))

def adjust_predictions_for_sentiment(preds_array, gauge_val, current_price):
    offset = (gauge_val - 50) / 50.0
    factor = offset * 0.05
    preds_adj = preds_array * (1 + factor)
    if gauge_val > 60 and preds_adj[0] < current_price:
        extra_factor = min((current_price / preds_adj[0] - 1), 0.10)
        preds_adj = preds_adj * (1 + extra_factor)
    return preds_adj

# ---------------------------------------------------------------------
# Technical Indicators Calculation (Parallel and Vectorized)
# ---------------------------------------------------------------------
def compute_rsi(df):
    return RSIIndicator(close=df["close_price"], window=14).rsi()

def compute_macd(df):
    return MACD(close=df["close_price"], window_fast=12, window_slow=26, window_sign=9).macd()

def compute_bollinger_upper(df):
    return BollingerBands(close=df["close_price"], window=20, window_dev=2).bollinger_hband()

def compute_bollinger_lower(df):
    return BollingerBands(close=df["close_price"], window=20, window_dev=2).bollinger_lband()

def compute_sma50(df):
    return SMAIndicator(close=df["close_price"], window=50).sma_indicator()

def compute_atr(df):
    return AverageTrueRange(high=df["high"], low=df["low"], close=df["close_price"], window=14).average_true_range()

def compute_base_indicators(df):
    results = Parallel(n_jobs=-1)(
        delayed(func)(df) for func in [
            compute_rsi, compute_macd, compute_bollinger_upper,
            compute_bollinger_lower, compute_sma50, compute_atr
        ]
    )
    df["RSI"] = results[0]
    df["rsi_norm"] = df["RSI"] / 100.0
    df["macd"] = results[1]
    df["bollinger_upper"] = results[2]
    df["bollinger_lower"] = results[3]
    df["sma50"] = results[4]
    df["atr"] = results[5]
    return df

def compute_additional_features(df):
    df["log_return"] = np.log(df["close_price"] / df["close_price"].shift(1)).fillna(0.0)
    df["vol_30d"] = df["log_return"].rolling(window=30).std().fillna(0.0)
    df["obv"] = OnBalanceVolumeIndicator(close=df["close_price"], volume=df["volume"]).on_balance_volume()
    df["ema200"] = EMAIndicator(close=df["close_price"], window=200).ema_indicator()
    return df

# ---------------------------------------------------------------------
# Sequence Creation (Using Numba)
# ---------------------------------------------------------------------
@njit
def create_sequences_numba(data, window_size):
    n = data.shape[0]
    num_features = data.shape[1]
    m = n - window_size
    X = np.empty((m, window_size, num_features), dtype=data.dtype)
    y = np.empty(m, dtype=data.dtype)
    for i in range(m):
        X[i] = data[i:i+window_size]
        y[i] = data[i+window_size, 0]
    return X, y

def create_sequences(data, window_size):
    if data.shape[0] <= window_size:
        return None, None
    return create_sequences_numba(data, window_size)

def flatten_sequences(X_seq):
    return X_seq.reshape((X_seq.shape[0], X_seq.shape[1] * X_seq.shape[2]))

# ---------------------------------------------------------------------
# Models with Keras Tuner (LSTM and GRU)
# ---------------------------------------------------------------------
def build_lstm_model_tuner(input_shape):
    """Search space for LSTM."""
    def model_builder(hp):
        lstm_units1 = hp.Int('lstm_units1', min_value=64, max_value=128, step=32)
        lstm_units2 = hp.Int('lstm_units2', min_value=32, max_value=64, step=16)
        dropout_rate = hp.Float('dropout_rate', 0.1, 0.4, step=0.05)
        dense_units = hp.Int('dense_units', min_value=50, max_value=100, step=25)
        learning_rate = hp.Float('learning_rate', 1e-4, 1e-3, sampling='log')
        l2_lambda = hp.Float('l2_lambda', 1e-4, 1e-2, sampling='log')
        model = Sequential([
            LSTM(lstm_units1, return_sequences=True, input_shape=input_shape, kernel_regularizer=l2(l2_lambda)),
            Dropout(dropout_rate),
            LSTM(lstm_units2, kernel_regularizer=l2(l2_lambda)),
            Dropout(dropout_rate),
            Dense(dense_units, activation="relu", kernel_regularizer=l2(l2_lambda)),
            Dense(1)
        ])
        model.compile(optimizer=Adam(learning_rate), loss="mse")
        return model
    return model_builder

def build_gru_model_tuner(input_shape):
    """Search space for GRU."""
    def model_builder(hp):
        gru_units1 = hp.Int('gru_units1', min_value=64, max_value=128, step=32)
        gru_units2 = hp.Int('gru_units2', min_value=32, max_value=64, step=16)
        dropout_rate = hp.Float('dropout_rate', 0.1, 0.4, step=0.05)
        dense_units = hp.Int('dense_units', min_value=50, max_value=100, step=25)
        learning_rate = hp.Float('learning_rate', 1e-4, 1e-3, sampling='log')
        l2_lambda = hp.Float('l2_lambda', 1e-4, 1e-2, sampling='log')
        model = Sequential([
            GRU(gru_units1, return_sequences=True, input_shape=input_shape, kernel_regularizer=l2(l2_lambda)),
            Dropout(dropout_rate),
            GRU(gru_units2, kernel_regularizer=l2(l2_lambda)),
            Dropout(dropout_rate),
            Dense(dense_units, activation="relu", kernel_regularizer=l2(l2_lambda)),
            Dense(1)
        ])
        model.compile(optimizer=Adam(learning_rate), loss="mse")
        return model
    return model_builder

# ---------------------------------------------------------------------
# Iterative Forecast Functions
# ---------------------------------------------------------------------
def iterative_rnn_forecast(model, current_input, scaler, feature_cols, horizon_days):
    """Generic forecast for LSTM/GRU; input shape is assumed to be consistent."""
    preds = []
    for _ in range(horizon_days):
        p_scaled = model.predict(current_input, verbose=0)[0][0]
        reconst = np.concatenate([[p_scaled], np.zeros((len(feature_cols)-1,))]).reshape(1, -1)
        inv = scaler.inverse_transform(reconst)
        preds.append(inv[0, 0])
        new_feature = np.copy(current_input[:, -1:, :])
        new_feature[0, 0, 0] = p_scaled
        current_input = np.append(current_input[:, 1:, :], new_feature, axis=1)
    return np.expm1(np.array(preds))

def iterative_xgb_forecast(model, current_input, scaler, feature_cols, horizon_days):
    preds = []
    for _ in range(horizon_days):
        X_flat = current_input.reshape(1, -1)
        xgb_p_scaled = model.predict(X_flat)[0]
        reconst = np.concatenate([[xgb_p_scaled], np.zeros((len(feature_cols)-1,))]).reshape(1, -1)
        inv = scaler.inverse_transform(reconst)
        preds.append(inv[0, 0])
        new_feature = np.copy(current_input[:, -1:, :])
        new_feature[0, 0, 0] = xgb_p_scaled
        current_input = np.append(current_input[:, 1:, :], new_feature, axis=1)
    return np.expm1(np.array(preds))

# ---------------------------------------------------------------------
# Ensemble Function for 4 Models: LSTM, GRU, XGB, and Prophet
# ---------------------------------------------------------------------
def ensemble_prediction_4(lstm_pred, gru_pred, xgb_pred, prophet_pred, w_lstm=0.3, w_gru=0.3, w_xgb=0.2, w_prophet=0.2):
    """Combine predictions from 4 models using configurable weights."""
    return (
        w_lstm * lstm_pred +
        w_gru * gru_pred +
        w_xgb * xgb_pred +
        w_prophet * prophet_pred
    )

def medium_long_term_prediction(df, days=180, current_price=None):
    df_prophet = df[["ds", "close_price"]].copy()
    df_prophet.rename(columns={"close_price": "y"}, inplace=True)
    df_prophet["y"] = np.log1p(df_prophet["y"])
    model = Prophet()
    model.fit(df_prophet)
    future_dates = model.make_future_dataframe(periods=days)
    forecast = model.predict(future_dates)
    forecast["exp_yhat"] = np.expm1(forecast["yhat"])
    if current_price is not None and days > 0:
        forecast.loc[forecast.index[-days], "exp_yhat"] = current_price
    return model, forecast

# ---------------------------------------------------------------------
# Function to Load Historical Data
# ---------------------------------------------------------------------
def load_crypto_data(coin_id, start_date=None, end_date=None):
    ticker_ids = {
        "bitcoin": "BTC-USD",
        "ethereum": "ETH-USD",
        "xrp": "XRP-USD",
        "binance-coin": "BNB-USD",
        "cardano": "ADA-USD",
        "solana": "SOL-USD",
        "dogecoin": "DOGE-USD",
        "polkadot": "DOT-USD",
        "polygon": "MATIC-USD",
        "litecoin": "LTC-USD",
        "tron": "TRX-USD",
        "stellar": "XLM-USD"
    }
    ticker = ticker_ids.get(coin_id)
    if not ticker:
        st.error("Ticker not found.")
        return None

    if start_date is None or end_date is None:
        df = yf.download(ticker, period="max", progress=False)
    else:
        df = yf.download(ticker, start=start_date, end=end_date, progress=False)

    if df.empty:
        st.warning("Data not available from yfinance.")
        return None

    df = df.reset_index()
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    df.rename(
        columns={"Date": "ds", "Close": "close_price", "Volume": "volume",
                 "High": "high", "Low": "low"},
        inplace=True
    )

    df = compute_base_indicators(df)
    df = compute_additional_features(df)
    df.dropna(inplace=True)

    current_sent = get_news_sentiment(coin_id)
    df["sentiment"] = current_sent

    return df

# ---------------------------------------------------------------------
# Main Function for Training and Prediction
# ---------------------------------------------------------------------
def train_and_predict_with_sentiment(coin_id, horizon_days, start_date=None, end_date=None, training_period_years=1):
    st.info("Training and prediction process may take a while. Please wait...")
    progress_text = st.empty()
    progress_bar = st.progress(0)

    # ----------------------
    # Load and Prepare Data
    # ----------------------
    full_df = load_crypto_data(coin_id, start_date, end_date)
    if full_df is None or full_df.empty:
        st.error("Could not load historical data.")
        return None

    last_date = full_df["ds"].max()
    period_start = last_date - pd.DateOffset(years=training_period_years)
    df_pred = full_df[full_df["ds"] >= period_start].copy()
    if df_pred.empty:
        st.error("Not enough recent data for training.")
        return None

    progress_text.text("Preparing dataset and scaling data...")
    progress_bar.progress(25)

    df_pred["log_price"] = np.log1p(df_pred["close_price"])

    feature_cols = [
        "log_price", "volume", "high", "low", "rsi_norm", "macd",
        "bollinger_upper", "bollinger_lower", "atr", "obv", "ema200",
        "log_return", "vol_30d", "sentiment"
    ]
    pipe = Pipeline([
        ('imputer', DataFrameTransformer(SimpleImputer(strategy="median"))),
        ('selector', FeatureSelector(feature_cols, target_col='log_price', enet_threshold=0.01, importance_threshold=0.01)),
        ('scaler', MinMaxScaler())
    ])
    scaled_data = pipe.fit_transform(df_pred[feature_cols])
    selected_features = pipe.named_steps['selector'].selected_features_
    if not selected_features:
        selected_features = feature_cols

    window_size = 60 if horizon_days <= 30 else 90
    X, y = create_sequences(scaled_data, window_size)
    if X is None:
        st.error("Not enough data to create sequences.")
        return None

    split = int(len(X) * 0.8)
    X_train, X_test = X[:split], X[split:]
    y_train, y_test = y[:split], y[split:]
    val_split = int(len(X_train) * 0.9)
    X_val, y_val = X_train[val_split:], y_train[val_split:]
    X_train, y_train = X_train[:val_split], y_train[:val_split]

    epochs = 8
    batch_size = 32
    input_shape = (window_size, len(selected_features))

    # ----------------------
    # 1) Train LSTM Model
    # ----------------------
    progress_text.text("Searching hyperparameters for LSTM...")
    progress_bar.progress(30)

    if os.path.exists('kt_dir_lstm'):
        shutil.rmtree('kt_dir_lstm')

    callbacks = [
        tf.keras.callbacks.EarlyStopping(patience=3, restore_best_weights=True),
        tf.keras.callbacks.ReduceLROnPlateau(patience=2, factor=0.5, min_lr=1e-6)
    ]
    
    tuner_lstm = kt.Hyperband(
        build_lstm_model_tuner(input_shape),
        objective='val_loss',
        max_epochs=epochs,
        factor=3,
        directory='kt_dir_lstm',
        project_name=f'{coin_id}_crypto_lstm'
    )
    tuner_lstm.search(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=epochs,
        batch_size=batch_size,
        callbacks=callbacks,
        verbose=0
    )
    lstm_model = tuner_lstm.get_best_models(num_models=1)[0]

    # ----------------------
    # 2) Train GRU Model
    # ----------------------
    progress_text.text("Searching hyperparameters for GRU...")
    progress_bar.progress(35)

    if os.path.exists('kt_dir_gru'):
        shutil.rmtree('kt_dir_gru')

    tuner_gru = kt.Hyperband(
        build_gru_model_tuner(input_shape),
        objective='val_loss',
        max_epochs=epochs,
        factor=3,
        directory='kt_dir_gru',
        project_name=f'{coin_id}_crypto_gru'
    )
    tuner_gru.search(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=epochs,
        batch_size=batch_size,
        callbacks=callbacks,
        verbose=0
    )
    gru_model = tuner_gru.get_best_models(num_models=1)[0]

    # ----------------------
    # Test Predictions for LSTM/GRU
    # ----------------------
    progress_text.text("Predicting on test data (LSTM/GRU)...")
    progress_bar.progress(45)

    # LSTM
    preds_test_scaled_lstm = lstm_model.predict(X_test, verbose=0)
    reconst_test_lstm = np.concatenate([preds_test_scaled_lstm, np.zeros((len(preds_test_scaled_lstm), len(selected_features)-1))], axis=1)
    reconst_test_inv_lstm = pipe.named_steps['scaler'].inverse_transform(reconst_test_lstm)
    preds_test_log_lstm = reconst_test_inv_lstm[:, 0]
    lstm_test_preds = np.expm1(preds_test_log_lstm)

    # GRU
    preds_test_scaled_gru = gru_model.predict(X_test, verbose=0)
    reconst_test_gru = np.concatenate([preds_test_scaled_gru, np.zeros((len(preds_test_scaled_gru), len(selected_features)-1))], axis=1)
    reconst_test_inv_gru = pipe.named_steps['scaler'].inverse_transform(reconst_test_gru)
    preds_test_log_gru = reconst_test_inv_gru[:, 0]
    gru_test_preds = np.expm1(preds_test_log_gru)

    # Reconstruct y_test from scaled data
    reconst_y = np.concatenate([y_test.reshape(-1, 1), np.zeros((len(y_test), len(selected_features)-1))], axis=1)
    reconst_y_inv = pipe.named_steps['scaler'].inverse_transform(reconst_y)
    y_test_log = reconst_y_inv[:, 0]
    y_test_real = np.expm1(y_test_log)

    # Calculate MAPE for LSTM
    lstm_mape = np.mean(np.abs((y_test_real - lstm_test_preds) / np.maximum(np.abs(y_test_real), 1e-9))) * 100
    # Calculate MAPE for GRU
    gru_mape = np.mean(np.abs((y_test_real - gru_test_preds) / np.maximum(np.abs(y_test_real), 1e-9))) * 100

    # ----------------------
    # 3) Train XGBoost and Prophet (Ensemble)
    # ----------------------
    progress_text.text("Training XGBoost and Prophet (ensemble)...")
    progress_bar.progress(55)

    X_train_val = np.concatenate([X_train, X_val], axis=0)
    y_train_val = np.concatenate([y_train, y_val], axis=0)
    X_train_val_flat = flatten_sequences(X_train_val)

    xgb_model = XGBRegressor(
        n_estimators=150, max_depth=6, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.8
    )
    xgb_model.fit(X_train_val_flat, y_train_val)

    X_test_flat = flatten_sequences(X_test)
    xgb_test_scaled = xgb_model.predict(X_test_flat)
    xgb_reconst = np.concatenate([xgb_test_scaled.reshape(-1, 1), np.zeros((len(xgb_test_scaled), len(selected_features)-1))], axis=1)
    xgb_test_inv = pipe.named_steps['scaler'].inverse_transform(xgb_reconst)
    xgb_test_log = xgb_test_inv[:, 0]
    xgb_test_preds = np.expm1(xgb_test_log)

    df_prophet = full_df[["ds", "close_price"]].copy()
    df_prophet.rename(columns={"close_price": "y"}, inplace=True)
    df_prophet["y"] = np.log1p(df_prophet["y"])
    prophet_model = Prophet()
    prophet_model.fit(df_prophet)

    future_prophet = prophet_model.make_future_dataframe(periods=len(X_test))
    forecast = prophet_model.predict(future_prophet)
    prophet_preds_log = forecast["yhat"].tail(len(X_test)).values
    prophet_test_preds = np.expm1(prophet_preds_log)

    # ----------------------
    # 4) Final Ensemble (Test)
    # ----------------------
    test_ens_preds = ensemble_prediction_4(
        lstm_test_preds, gru_test_preds, xgb_test_preds, prophet_test_preds,
        w_lstm=0.3, w_gru=0.3, w_xgb=0.2, w_prophet=0.2
    )
    ens_mape = np.mean(np.abs((y_test_real - test_ens_preds) / np.maximum(np.abs(y_test_real), 1e-9))) * 100
    ens_accuracy = max(0, 100 - ens_mape)

    # ----------------------
    # 5) Future Prediction (LSTM, GRU, XGBoost, Prophet)
    # ----------------------
    progress_text.text("Generating future predictions (LSTM, GRU, XGBoost, Prophet)...")
    progress_bar.progress(70)

    last_window = pipe.named_steps['scaler'].transform(df_pred[selected_features].values[-window_size:])
    current_input_lstm = last_window.reshape(1, window_size, len(selected_features))
    current_input_gru = np.copy(current_input_lstm)
    current_input_xgb = np.copy(current_input_lstm)

    # Future prediction using LSTM
    lstm_future_preds = iterative_rnn_forecast(lstm_model, current_input_lstm, pipe.named_steps['scaler'], selected_features, horizon_days)
    # Future prediction using GRU
    gru_future_preds = iterative_rnn_forecast(gru_model, current_input_gru, pipe.named_steps['scaler'], selected_features, horizon_days)
    # Future prediction using XGBoost
    xgb_future_preds = iterative_xgb_forecast(xgb_model, current_input_xgb, pipe.named_steps['scaler'], selected_features, horizon_days)

    current_price = full_df["close_price"].iloc[-1]
    future_prophet2 = prophet_model.make_future_dataframe(periods=horizon_days)
    forecast2 = prophet_model.predict(future_prophet2)
    prophet_preds_log2 = forecast2["yhat"].tail(horizon_days).values
    prophet_future_preds = np.expm1(prophet_preds_log2)
    prophet_future_preds[0] = current_price

    final_future_preds = ensemble_prediction_4(
        lstm_future_preds, gru_future_preds, xgb_future_preds, prophet_future_preds,
        w_lstm=0.3, w_gru=0.3, w_xgb=0.2, w_prophet=0.2
    )
    # Set the first value to the current price
    final_future_preds[0] = current_price

    # Adjust predictions based on sentiment analysis
    _, _, gauge_val = get_crypto_sentiment_combined(coin_id)
    final_future_preds = adjust_predictions_for_sentiment(final_future_preds, gauge_val, current_price)
    future_dates = pd.date_range(start=full_df["ds"].iloc[-1] + timedelta(days=1), periods=horizon_days).tolist()

    progress_text.text("Prediction completed successfully!")
    progress_bar.progress(100)

    # ----------------------
    # Final Output
    # ----------------------
    return {
        "df": full_df,
        "df_train": df_pred,
        "symbol": coinid_to_symbol[coin_id],
        "test_preds": test_ens_preds,
        "accuracy": ens_accuracy,
        "crypto_sent": get_news_sentiment(coin_id),
        "market_sent": get_fear_greed_index(),
        "gauge_val": gauge_val,
        "future_preds": final_future_preds,
        "future_dates": future_dates,
        "test_dates": full_df["ds"].iloc[-len(test_ens_preds):].values,
        "real_prices": y_test_real,
        "lstm_mape": lstm_mape,
        "gru_mape": gru_mape
    }

def main_app():
    st.title("Crypto Price Predictions 🔮")
    st.markdown("""
    More information [here](https://github.com/27MarioGomez/projects/blob/18467c6d0ff4ec1ce63999ded0e12dd2f0be7469/CRYPTO_PREDICTION/README.md).
    
    **NFA:** Not Financial Advice.
    """)
    st.sidebar.title("Prediction Settings")
    crypto_name = st.sidebar.selectbox("Select a cryptocurrency:", list(coincap_ids.keys()))
    coin_id = coincap_ids[crypto_name]

    training_period = st.sidebar.select_slider(
        "Training period (years):",
        options=[1, 2, 3],
        value=1,
        help="A longer period requires more training time."
    )
    horizon = st.sidebar.slider("Days to predict:", 1, 60, 5, help="A longer prediction horizon requires more processing time.")

    use_custom_range = st.sidebar.checkbox("Enable date range", value=False)
    default_end = datetime.utcnow()
    default_start = default_end - timedelta(days=7)
    if use_custom_range:
        start_date = st.sidebar.date_input("Start date", default_start.date())
        end_date = st.sidebar.date_input("End date", default_end.date())
        if start_date > end_date:
            st.sidebar.error("Start date cannot be later than end date.")
            return
        if (end_date - start_date).days > 7:
            st.sidebar.warning("The range exceeds 7 days. It will be adjusted to 7 days.")
            end_date = start_date + timedelta(days=7)
        if start_date > datetime.utcnow().date():
            start_date = datetime.utcnow().date() - timedelta(days=7)
            st.sidebar.warning("Start date cannot be in the future. Adjusted to 7 days ago.")
        if end_date > datetime.utcnow().date():
            end_date = datetime.utcnow().date()
            st.sidebar.warning("End date cannot be in the future. Adjusted to today.")
        end_date_with_offset = end_date + timedelta(days=1)
    else:
        start_date = None
        end_date_with_offset = None

    df_prices = load_crypto_data(coin_id, start_date, end_date_with_offset)
    if df_prices is not None and not df_prices.empty:
        fig_hist = px.line(
            df_prices,
            x="ds",
            y="close_price",
            title=f"{crypto_name} Historical Prices",
            labels={"ds": "Date", "close_price": "Price (USD)"}
        )
        fig_hist.update_layout(template="plotly_dark")
        fig_hist.update_xaxes(tickformat="%Y-%m-%d")
        st.plotly_chart(fig_hist, use_container_width=True)
    else:
        st.warning("Historical data could not be loaded for the selected range.")

    tabs = st.tabs([
        "Model Training",
        "Short Term Prediction",
        "Medium/Long Term Prediction",
        "Sentiment Analysis",
        "Recent News"
    ])

    with tabs[0]:
        if st.button("Train Model and Predict"):
            result = train_and_predict_with_sentiment(
                coin_id, horizon,
                start_date, end_date_with_offset,
                training_period_years=training_period
            )
            if result:
                st.success("Training and prediction completed!")
                st.write(f"News Sentiment ({result['symbol']}): {result['crypto_sent']:.2f}")
                st.write(f"Market Sentiment (Fear & Greed): {result['market_sent']:.2f}")
                st.write(f"Combined Gauge: {result['gauge_val']:.2f}")
                st.metric("Ensemble Accuracy (Test)", f"{result['accuracy']:.2f}%")
                st.write(f"LSTM MAPE: {result['lstm_mape']:.2f}%")
                st.write(f"GRU MAPE: {result['gru_mape']:.2f}%")
                st.session_state["result"] = result

    with tabs[1]:
        result = st.session_state.get("result", None)
        if result:
            st.header(f"Short Term Prediction - {result['symbol']}")
            last_date = result["df"]["ds"].iloc[-1].date()
            current_price = result["df"]["close_price"].iloc[-1]
            pred_series = np.concatenate(([current_price], result["future_preds"]))
            future_dates_display = [last_date] + [fd.date() for fd in result["future_dates"]]
            fig_future = go.Figure()
            fig_future.add_trace(go.Scatter(
                x=future_dates_display,
                y=pred_series,
                mode="lines+markers",
                name=f"Prediction – {result['symbol']}",
                line=dict(color="#ff7f0e", width=2, shape="spline")
            ))
            fig_future.update_layout(
                title=f"Future Prediction ({horizon} days) - {result['symbol']}",
                template="plotly_dark",
                xaxis_title="Date",
                yaxis_title="Price (USD)"
            )
            st.plotly_chart(fig_future, use_container_width=True)
            st.header("Numeric Results (Short Term)")
            df_future = pd.DataFrame({"Date": future_dates_display, "Prediction": pred_series})
            st.dataframe(df_future.style.format({"Prediction": "{:.2f}"}))
            st.download_button(
                label="Download short-term predictions in CSV",
                data=df_future.to_csv(index=False).encode("utf-8"),
                file_name="short_term_predictions.csv",
                mime="text/csv"
            )
        else:
            st.info("Train the model first to view the short-term prediction.")

    with tabs[2]:
        result = st.session_state.get("result", None)
        if result:
            st.header(f"Medium/Long Term Prediction - {result['symbol']}")
            current_price = result["df"]["close_price"].iloc[-1]
            _, forecast_long = medium_long_term_prediction(result["df"], days=180, current_price=current_price)
            forecast_long["ds"] = pd.to_datetime(forecast_long["ds"]).dt.date
            forecast_long_part = forecast_long[["ds", "exp_yhat"]].tail(180)
            fig_long = go.Figure()
            df_plot = result["df"].copy()
            df_plot["ds"] = df_plot["ds"].dt.date
            fig_long.add_trace(go.Scatter(
                x=df_plot["ds"],
                y=df_plot["close_price"],
                mode="lines",
                name="Historical",
                line=dict(color="#1f77b4", width=2)
            ))
            fig_long.add_trace(go.Scatter(
                x=forecast_long_part["ds"],
                y=forecast_long_part["exp_yhat"],
                mode="lines",
                name="180-day Prediction",
                line=dict(color="#ff7f0e", width=2, dash="dash")
            ))
            fig_long.update_layout(
                title="180-day Prediction (Medium/Long Term) - Prophet",
                template="plotly_dark",
                xaxis_title="Date",
                yaxis_title="Price (USD)"
            )
            st.plotly_chart(fig_long, use_container_width=True)
            st.header("Numeric Values (180-day Horizon)")
            styled_forecast = forecast_long_part.copy()
            styled_forecast.columns = ["Date", "Prediction (USD)"]
            st.dataframe(styled_forecast.style.format({"Prediction (USD)": "{:.2f}"}))
            st.download_button(
                label="Download medium/long term predictions in CSV",
                data=styled_forecast.to_csv(index=False).encode("utf-8"),
                file_name="medium_long_term_predictions.csv",
                mime="text/csv"
            )
        else:
            st.info("Train the model to view the medium/long term prediction.")

    with tabs[3]:
        st.header("Sentiment Analysis")
        crypto_sent = get_news_sentiment(coin_id)
        market_sent = get_fear_greed_index()
        _, _, gauge_val = get_crypto_sentiment_combined(coin_id)

        if gauge_val < 20:
            gauge_text = "Very Bearish"
        elif gauge_val < 40:
            gauge_text = "Bearish"
        elif gauge_val < 60:
            gauge_text = "Neutral"
        elif gauge_val < 80:
            gauge_text = "Bullish"
        else:
            gauge_text = "Very Bullish"

        fig_sentiment = go.Figure(go.Indicator(
            mode="gauge+number",
            value=gauge_val,
            number={'suffix': "", "font": {"size": 36}},
            gauge={
                "axis": {"range": [0, 100], "tickwidth": 2, "tickcolor": "#fff"},
                "bar": {"color": "LightSkyBlue"},
                "bgcolor": "#2c2c3e",
                "borderwidth": 2,
                "bordercolor": "#4a4a6a",
                "steps": [
                    {"range": [0, 20], "color": "#ff0000"},
                    {"range": [20, 40], "color": "#ff7f0e"},
                    {"range": [40, 60], "color": "#ffff00"},
                    {"range": [60, 80], "color": "#90ee90"},
                    {"range": [80, 100], "color": "#008000"}
                ],
                "threshold": {
                    "line": {"color": "#000", "width": 4},
                    "thickness": 0.8,
                    "value": gauge_val
                }
            },
            domain={"x": [0, 1], "y": [0, 1]}
        ))
        fig_sentiment.update_layout(
            title={
                "text": f"Sentiment - {coinid_to_symbol[coin_id]}",
                "x": 0.5, "xanchor": "center",
                "font": {"size": 24}
            },
            template="plotly_dark",
            height=400,
            margin=dict(l=20, r=20, t=80, b=20)
        )
        st.plotly_chart(fig_sentiment, use_container_width=True)
        st.write(f"**News Sentiment ({coinid_to_symbol[coin_id]}):** {crypto_sent:.2f}")
        st.write(f"**Market Sentiment (Fear & Greed):** {market_sent:.2f}")
        st.write(f"**Gauge Value:** {gauge_val:.2f} → **{gauge_text}**")

    with tabs[4]:
        st.subheader(f"Recent News about {crypto_name} ({coinid_to_symbol[coin_id]})")
        articles = get_newsapi_articles(coin_id, show_warning=True)
        if articles:
            st.markdown(
                """
                <style>
                .news-container {
                    display: flex;
                    flex-direction: row;
                    gap: 1rem;
                    overflow-x: auto;
                    padding: 1rem 0;
                }
                .news-item {
                    flex: 0 0 auto;
                    width: 280px;
                    min-height: 380px;
                    background-color: #2c2c3e;
                    padding: 0.5rem;
                    border-radius: 5px;
                    border: 1px solid #4a4a6a;
                    display: flex;
                    flex-direction: column;
                    justify-content: space-between;
                }
                .news-item img {
                    width: 100%;
                    height: 160px;
                    object-fit: cover;
                    border-radius: 5px;
                    margin-bottom: 0.5rem;
                }
                .news-item h4 {
                    margin: 0 0 0.3rem 0;
                    font-size: 1rem;
                }
                .news-item p {
                    font-size: 0.8rem;
                    margin: 0 0 0.3rem 0;
                }
                .read-more-btn {
                    display: block;
                    margin: 0.5rem auto;
                    padding: 0.4rem 0.8rem;
                    background-color: #fff;
                    color: #000;
                    text-decoration: none;
                    border-radius: 3px;
                    text-align: center;
                    font-size: 0.8rem;
                    font-weight: 600;
                }
                .read-more-btn:hover {
                    background-color: #e6e6e6;
                    color: #000;
                }
                </style>
                """,
                unsafe_allow_html=True
            )
            st.markdown("<div class='news-container'>", unsafe_allow_html=True)
            for article in articles:
                image_tag = ""
                if article['image']:
                    image_tag = f"<img src='{article['image']}' alt='Image'/>"
                link_button = f"<a href='{article['link']}' target='_blank' class='read-more-btn'>Read more</a>"
                st.markdown(
                    f"""
                    <div class='news-item'>
                        {image_tag}
                        <div>
                            <h4>{article['title']}</h4>
                            <p><em>{article['pubDate']}</em></p>
                            <p>{article['description']}</p>
                        </div>
                        <div>
                            {link_button}
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True
                )
            st.markdown("</div>", unsafe_allow_html=True)
        else:
            st.warning("Request rate limit exceeded or no news available.")

if __name__ == "__main__":
    main_app()
