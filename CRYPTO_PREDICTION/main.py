import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.regularizers import l2

# Activar mixed precision si hay GPU (en CPU se ignora)
if tf.config.list_physical_devices('GPU'):
    from tensorflow.keras.mixed_precision import set_global_policy
    set_global_policy('mixed_float16')

import yfinance as yf
import requests
import certifi
import os
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from textblob import TextBlob
from dateutil.parser import parse as date_parse
from urllib3.util.retry import Retry
from requests.adapters import HTTPAdapter
from newsapi import NewsApiClient
from prophet import Prophet
from ta.momentum import RSIIndicator
from ta.trend import MACD, SMAIndicator
from ta.volatility import BollingerBands, AverageTrueRange
from transformers import pipeline
import optuna
from xgboost import XGBRegressor
import time

# Reducir verbosidad de Optuna
optuna.logging.set_verbosity(optuna.logging.WARNING)

os.environ["REQUESTS_CA_BUNDLE"] = certifi.where()
session = requests.Session()
retry = Retry(total=5, backoff_factor=1, status_forcelist=[429, 500, 502, 503, 504])
adapter = HTTPAdapter(max_retries=retry)
session.mount("https://", adapter)

# Diccionario de criptomonedas
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

# =============================================================================
# Funciones para indicadores técnicos
# =============================================================================
def compute_indicators(df):
    df["RSI"] = RSIIndicator(close=df["close_price"], window=14).rsi()
    df["rsi_norm"] = df["RSI"] / 100.0
    df["macd"] = MACD(close=df["close_price"], window_fast=12, window_slow=26, window_sign=9).macd()
    bb = BollingerBands(close=df["close_price"], window=20, window_dev=2)
    df["bollinger_upper"] = bb.bollinger_hband()
    df["bollinger_lower"] = bb.bollinger_lband()
    df["sma50"] = SMAIndicator(close=df["close_price"], window=50).sma_indicator()
    atr = AverageTrueRange(high=df["high"], low=df["low"], close=df["close_price"], window=14)
    df["atr"] = atr.average_true_range()
    df.ffill(inplace=True)
    return df

# =============================================================================
# Análisis de Sentimiento con Transformers y TextBlob
# =============================================================================
@st.cache_resource(show_spinner=False)
def load_sentiment_pipeline():
    return pipeline("sentiment-analysis")

def get_advanced_sentiment(text):
    sentiment_pipe = load_sentiment_pipeline()
    result = sentiment_pipe(text)[0]
    if result["label"].upper() == "POSITIVE":
        return 50 + (result["score"] * 50)
    else:
        return 50 - (result["score"] * 50)

# =============================================================================
# Carga y preprocesamiento de datos históricos (yfinance)
# =============================================================================
@st.cache_data
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
        st.error("Ticker no encontrado.")
        return None
    if start_date is None or end_date is None:
        df = yf.download(ticker, period="max", progress=False)
    else:
        df = yf.download(ticker, start=start_date, end=end_date, progress=False)
    if df.empty:
        st.warning("Datos no disponibles desde yfinance.")
        return None
    df = df.reset_index()
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    df.rename(columns={"Date": "ds", "Close": "close_price", "Volume": "volume",
                       "High": "high", "Low": "low"}, inplace=True)
    df = compute_indicators(df)
    df.dropna(inplace=True)
    return df[["ds", "close_price", "volume", "high", "low", "RSI", "rsi_norm", "macd", "atr"]]

def create_sequences(data, window_size):
    if len(data) <= window_size:
        return None, None
    X, y = [], []
    for i in range(window_size, len(data)):
        X.append(data[i-window_size:i])
        y.append(data[i, 0])
    return np.array(X), np.array(y)

# =============================================================================
# Modelo LSTM y entrenamiento
# =============================================================================
def build_lstm_model(input_shape, learning_rate=0.0005, l2_lambda=0.01,
                     lstm_units1=128, lstm_units2=64, dropout_rate=0.3, dense_units=100):
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

def train_model(X_train, y_train, X_val, y_val, model, epochs=10, batch_size=32):
    tf.keras.backend.clear_session()
    callbacks = [
        EarlyStopping(patience=8, restore_best_weights=True),
        ReduceLROnPlateau(patience=4, factor=0.5, min_lr=1e-6)
    ]
    model.fit(X_train, y_train, validation_data=(X_val, y_val),
              epochs=epochs, batch_size=batch_size, callbacks=callbacks, verbose=0)
    return model

# =============================================================================
# XGBoost para ensamble
# =============================================================================
def flatten_sequences(X_seq):
    return X_seq.reshape((X_seq.shape[0], X_seq.shape[1]*X_seq.shape[2]))

def train_xgboost(X, y):
    model_xgb = XGBRegressor(n_estimators=200, max_depth=5, learning_rate=0.05,
                             subsample=0.8, colsample_bytree=0.8)
    model_xgb.fit(X, y)
    return model_xgb

def ensemble_prediction(lstm_pred, xgb_pred, prophet_pred, w_lstm=0.5, w_xgb=0.3, w_prophet=0.2):
    return w_lstm * lstm_pred + w_xgb * xgb_pred + w_prophet * prophet_pred

def medium_long_term_prediction(df, days=180):
    df_prophet = df[["ds", "close_price"]].copy()
    df_prophet.rename(columns={"close_price": "y"}, inplace=True)
    df_prophet["y"] = np.log1p(df_prophet["y"])
    model = Prophet()
    model.fit(df_prophet)
    future_dates = model.make_future_dataframe(periods=days)
    forecast = model.predict(future_dates)
    forecast["exp_yhat"] = np.expm1(forecast["yhat"])
    return model, forecast

def apply_shock_factor(df, base_sentiment):
    df["pct_change"] = df["close_price"].pct_change().fillna(0)
    sentiment_array = []
    for i in range(len(df)):
        if abs(df["pct_change"].iloc[i]) > 0.05:
            sentiment_array.append(base_sentiment * 1.2 if df["pct_change"].iloc[i] > 0 else base_sentiment * 0.8)
        else:
            sentiment_array.append(base_sentiment)
    df.drop(columns=["pct_change"], inplace=True)
    return np.array(sentiment_array)

# =============================================================================
# Para acelerar el tuning, usamos una muestra pequeña (sample_size=50)
# =============================================================================
def sample_for_tuning(X, y, sample_size=50):
    if len(X) > sample_size:
        return X[-sample_size:], y[-sample_size:]
    return X, y

# =============================================================================
# Optimización de hiperparámetros con Optuna (1 ensayo por tuning)
# Se descartan caminos no viables mediante prunning
# =============================================================================
def objective(trial, X_train_adj, y_train, X_val_adj, y_val, input_shape, progress_text, progress_bar):
    trial_num = trial.number + 1
    batch_size = trial.suggest_int("batch_size", 16, 64, step=16)
    progress_text.text(f"Esto puede tardar un poco, por favor espera...\nEnsayo {trial_num}: Entrenando (1 epoch) con batch_size={batch_size} ...")
    lr = trial.suggest_float("learning_rate", 1e-5, 1e-3, log=True)
    lstm_units1 = trial.suggest_int("lstm_units1", 64, 256, step=32)
    lstm_units2 = trial.suggest_int("lstm_units2", 32, 128, step=16)
    dropout_rate = trial.suggest_float("dropout_rate", 0.2, 0.5, step=0.05)
    dense_units = trial.suggest_int("dense_units", 50, 150, step=10)
    
    model = build_lstm_model(input_shape, lr, 0.01, lstm_units1, lstm_units2, dropout_rate, dense_units)
    model = train_model(X_train_adj, y_train, X_val_adj, y_val, model, epochs=1, batch_size=batch_size)
    preds = model.predict(X_val_adj, verbose=0)
    reconst = np.concatenate([preds, np.zeros((len(preds), 4))], axis=1)
    loss = np.sqrt(mean_squared_error(y_val, reconst[:, 0]))
    progress_bar.progress(min(100, 40 + int(trial_num * 20)))
    progress_text.text(f"Ensayo {trial_num}: Pérdida obtenida = {loss:.4f}")
    return loss

def tune_lstm_params(X_train_adj, y_train, X_val_adj, y_val, input_shape, progress_text, progress_bar):
    X_train_sample, y_train_sample = sample_for_tuning(X_train_adj, y_train, sample_size=50)
    X_val_sample, y_val_sample = sample_for_tuning(X_val_adj, y_val, sample_size=50)
    pruner = optuna.pruners.MedianPruner(n_startup_trials=1, n_warmup_steps=2)
    study = optuna.create_study(direction="minimize", pruner=pruner)
    # Reducimos a 1 ensayo para acelerar el tuning en CPU
    study.optimize(lambda trial: objective(trial, X_train_sample, y_train_sample, X_val_sample, y_val_sample, input_shape, progress_text, progress_bar),
                   n_trials=1)
    progress_text.text(f"Optimización completada. Mejor pérdida: {study.best_value:.4f}")
    return study.best_params

@st.cache_data(ttl=43200)
def get_newsapi_articles(coin_id, show_warning=True):
    newsapi_key = st.secrets.get("newsapi_key", "")
    if not newsapi_key:
        st.error("Clave 'newsapi_key' no encontrada.")
        return []
    try:
        query = f"{coin_id} crypto"
        newsapi = NewsApiClient(api_key=newsapi_key)
        data = newsapi.get_everything(
            q=query,
            language="en",
            sort_by="relevancy",
            page_size=5
        )
        articles = []
        if data.get("articles"):
            for art in data["articles"]:
                image_url = art.get("urlToImage", "")
                title = art.get("title") or "Sin título"
                description = art.get("description") or "Sin descripción"
                pub_date = art.get("publishedAt") or "Fecha no disponible"
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
            st.warning("Oh, vaya, parece que hemos hecho más peticiones de las debidas a la API. Vuelve en 12 horas si quieres ver noticias :)")
        elif show_warning:
            st.error(f"Error al obtener noticias: {e}")
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
        st.warning("Índice Fear & Greed no disponible. Se usará 50.0.")
        return 50.0

def get_crypto_sentiment_combined(coin_id):
    news_sent = get_news_sentiment(coin_id)
    market_sent = get_fear_greed_index()
    gauge_val = 50 + (news_sent - market_sent)
    gauge_val = max(0, min(100, gauge_val))
    return news_sent, market_sent, gauge_val

def adjust_predictions_for_sentiment(future_preds, gauge_val):
    if gauge_val > 80:
        return future_preds * 1.03
    elif gauge_val < 20:
        return future_preds * 0.97
    return future_preds

# =============================================================================
# Función principal de entrenamiento y predicción con ensamble
# =============================================================================
def train_and_predict_with_sentiment(coin_id, horizon_days, start_date=None, end_date=None):
    progress_text = st.empty()
    progress_bar = st.progress(0)

    progress_text.text("Cargando datos históricos...")
    progress_bar.progress(5)
    df = load_crypto_data(coin_id, start_date, end_date)
    if df is None or df.empty:
        st.error("Datos históricos no disponibles.")
        return None

    progress_text.text("Calculando shock factor y sentimiento base...")
    progress_bar.progress(15)
    news_sent, market_sent, gauge_val = get_crypto_sentiment_combined(coin_id)
    base_sentiment = gauge_val / 100.0
    shock_array = apply_shock_factor(df.copy(), base_sentiment)

    progress_text.text("Generando secuencias y escalando datos...")
    progress_bar.progress(25)
    df["log_price"] = np.log1p(df["close_price"])
    df["log_volume"] = np.log1p(df["volume"] + 1)
    data_array = df[["log_price", "log_volume", "rsi_norm", "macd", "atr"]].values
    scaler = MinMaxScaler()
    scaled_data = scaler.fit_transform(data_array)

    window_size = 90
    X, y = create_sequences(scaled_data, window_size)
    if X is None:
        st.error("No hay suficientes datos para crear secuencias.")
        return None

    if len(shock_array) != len(df):
        st.error("Inconsistencia en longitudes para shock factor.")
        return None
    shock_list = [shock_array[i] for i in range(window_size, len(df))]
    shock_list = np.array(shock_list)
    if len(shock_list) != len(X):
        st.error("Error en el tamaño de shock list.")
        return None

    def add_shock_col(X_seq, shock_seq):
        X_adj = []
        for i in range(len(X_seq)):
            shock_window = np.full((window_size, 1), shock_seq[i])
            new_window = np.concatenate([X_seq[i], shock_window], axis=1)
            X_adj.append(new_window)
        return np.array(X_adj)

    X_adj = add_shock_col(X, shock_list)

    split = int(len(X_adj) * 0.8)
    X_train, X_test = X_adj[:split], X_adj[split:]
    y_train, y_test = y[:split], y[split:]
    val_split = int(len(X_train) * 0.9)
    X_val, y_val = X_train[val_split:], y_train[val_split:]
    X_train, y_train = X_train[:val_split], y_train[:val_split]

    progress_text.text("Optimizando hiperparámetros con Optuna...")
    progress_bar.progress(40)
    input_shape = (window_size, 6)
    best_params = tune_lstm_params(X_train, y_train, X_val, y_val, input_shape, progress_text, progress_bar)
    lr = best_params["learning_rate"]
    lstm_units1 = best_params["lstm_units1"]
    lstm_units2 = best_params["lstm_units2"]
    dropout_rate = best_params["dropout_rate"]
    dense_units = best_params["dense_units"]
    batch_size = best_params["batch_size"]

    progress_text.text("Entrenando modelo LSTM final...")
    progress_bar.progress(60)
    lstm_model = build_lstm_model(input_shape, lr, 0.01, lstm_units1, lstm_units2, dropout_rate, dense_units)
    lstm_model = train_model(X_train, y_train, X_val, y_val, lstm_model, epochs=10, batch_size=batch_size)

    progress_text.text("Realizando predicción en test (LSTM)...")
    progress_bar.progress(70)
    preds_test_scaled = lstm_model.predict(X_test, verbose=0)
    reconst_test = np.concatenate([preds_test_scaled, np.zeros((len(preds_test_scaled), 4))], axis=1)
    reconst_test_inv = scaler.inverse_transform(reconst_test)
    preds_test_log = reconst_test_inv[:, 0]
    lstm_test_preds = np.expm1(preds_test_log)

    reconst_y = np.concatenate([y_test.reshape(-1, 1), np.zeros((len(y_test), 4))], axis=1)
    reconst_y_inv = scaler.inverse_transform(reconst_y)
    y_test_log = reconst_y_inv[:, 0]
    y_test_real = np.expm1(y_test_log)

    lstm_rmse = np.sqrt(mean_squared_error(y_test_real, lstm_test_preds))
    lstm_mape = np.mean(np.abs((y_test_real - lstm_test_preds) / np.maximum(np.abs(y_test_real), 1e-9))) * 100

    progress_text.text("Entrenando XGBoost y Prophet para ensamble...")
    progress_bar.progress(80)
    X_train_val = np.concatenate([X_train, X_val], axis=0)
    y_train_val = np.concatenate([y_train, y_val], axis=0)
    X_train_val_flat = flatten_sequences(X_train_val)
    xgb_model = train_xgboost(X_train_val_flat, y_train_val)

    X_test_flat = flatten_sequences(X_test)
    xgb_test_scaled = xgb_model.predict(X_test_flat)
    xgb_reconst = np.concatenate([xgb_test_scaled.reshape(-1, 1), np.zeros((len(xgb_test_scaled), 4))], axis=1)
    xgb_test_inv = scaler.inverse_transform(xgb_reconst)
    xgb_test_log = xgb_test_inv[:, 0]
    xgb_test_preds = np.expm1(xgb_test_log)

    df_prophet = df[["ds", "close_price"]].copy()
    df_prophet.rename(columns={"close_price": "y"}, inplace=True)
    df_prophet["y"] = np.log1p(df_prophet["y"])
    prophet_model = Prophet()
    prophet_model.fit(df_prophet)
    future_prophet = prophet_model.make_future_dataframe(periods=len(X_test))
    forecast = prophet_model.predict(future_prophet)
    prophet_preds_log = forecast["yhat"].tail(len(X_test)).values
    prophet_test_preds = np.expm1(prophet_preds_log)

    test_ens_preds = ensemble_prediction(lstm_test_preds, xgb_test_preds, prophet_test_preds, 0.5, 0.3, 0.2)
    ens_rmse = np.sqrt(mean_squared_error(y_test_real, test_ens_preds))
    ens_mape = np.mean(np.abs((y_test_real - test_ens_preds) / np.maximum(np.abs(y_test_real), 1e-9))) * 100

    progress_text.text("Realizando predicción futura...")
    progress_bar.progress(90)
    last_window = scaled_data[-window_size:]
    last_shock = shock_array[-1]
    current_input = np.concatenate([
        last_window.reshape(1, window_size, 5),
        np.full((1, window_size, 1), last_shock)
    ], axis=-1)
    future_preds_log_lstm = []
    for _ in range(horizon_days):
        p_scaled = lstm_model.predict(current_input, verbose=0)[0][0]
        reconst_f = np.concatenate([[p_scaled], np.zeros(4)]).reshape(1, -1)
        inv_f = scaler.inverse_transform(reconst_f)
        plog = inv_f[0, 0]
        future_preds_log_lstm.append(plog)
        new_feature = np.copy(current_input[:, -1:, :])
        new_feature[0, 0, 0] = p_scaled
        new_feature[0, 0, 5] = last_shock
        current_input = np.append(current_input[:, 1:, :], new_feature, axis=1)
    lstm_future_preds = np.expm1(np.array(future_preds_log_lstm))

    X_last_flat = flatten_sequences(current_input)
    xgb_future_preds_log = []
    for _ in range(horizon_days):
        xgb_p = xgb_model.predict(X_last_flat)[0]
        reconst_xgb = np.concatenate([[xgb_p], np.zeros((4,))]).reshape(1, -1)
        inv_xgb = scaler.inverse_transform(reconst_xgb)
        xgb_log = inv_xgb[0, 0]
        xgb_future_preds_log.append(xgb_log)
    xgb_future_preds = np.expm1(np.array(xgb_future_preds_log))

    future_prophet2 = prophet_model.make_future_dataframe(periods=horizon_days)
    forecast2 = prophet_model.predict(future_prophet2)
    prophet_preds_log2 = forecast2["yhat"].tail(horizon_days).values
    prophet_future_preds = np.expm1(prophet_preds_log2)

    final_future_preds = ensemble_prediction(lstm_future_preds, xgb_future_preds, prophet_future_preds, 0.5, 0.3, 0.2)
    last_date = df["ds"].iloc[-1]
    future_dates = pd.date_range(start=last_date + timedelta(days=1), periods=horizon_days).tolist()

    progress_text.text("¡Predicción completada con éxito!")
    progress_bar.progress(100)

    return {
        "df": df,
        "test_preds": test_ens_preds,
        "rmse": ens_rmse,
        "mape": ens_mape,
        "symbol": coinid_to_symbol[coin_id],
        "crypto_sent": get_news_sentiment(coin_id),
        "market_sent": get_fear_greed_index(),
        "gauge_val": gauge_val,
        "future_preds": final_future_preds,
        "future_dates": future_dates,
        "test_dates": df["ds"].iloc[-len(test_ens_preds):].values,
        "real_prices": y_test_real
    }

# =============================================================================
# Pantalla de bienvenida (splash screen) con transición
# =============================================================================
def splash_screen():
    st.set_page_config(page_title="Crypto Price Predictions", layout="wide")
    st.title("Crypto Price Predictions 🔮")
    st.markdown("""
    <style>
    .fade-out {
        animation: fadeOut 1.5s forwards;
    }
    @keyframes fadeOut {
        from { opacity: 1; }
        to { opacity: 0; }
    }
    </style>
    **Bienvenido a Crypto Price Predictions**

    Este dashboard ha sido desarrollado para predecir el precio futuro de criptomonedas utilizando múltiples fuentes de datos y técnicas avanzadas:

    - **Datos Históricos e Indicadores Técnicos:**  
      Se extraen datos de yfinance y se calculan indicadores (RSI, MACD, ATR, Bollinger Bands, SMA) para analizar el mercado.
      
    - **Análisis de Sentimiento:**  
      Se evalúa el “estado de ánimo” combinando información de noticias (NewsAPI) y el índice Fear & Greed, utilizando Transformers y TextBlob.

    - **Ensamble de Modelos:**  
      Se entrenan un modelo LSTM, uno XGBoost y Prophet. Estos se combinan (50% LSTM, 30% XGBoost, 20% Prophet) para obtener una predicción robusta.

    - **Optimización Automática:**  
      Con Optuna se ajustan automáticamente los hiperparámetros del modelo LSTM en un proceso ligero, descartando rápidamente configuraciones no prometedoras para acelerar el entrenamiento.

    ¡Explora el dashboard y descubre las predicciones para tu criptomoneda favorita!
    """)
    if st.button("Comenzar"):
        # Mostrar una breve animación de transición
        st.markdown("<div class='fade-out'>Cargando dashboard...</div>", unsafe_allow_html=True)
        time.sleep(1.5)
        st.experimental_set_query_params(page="dashboard")

# =============================================================================
# Función principal del dashboard
# =============================================================================
def main_app():
    st.set_page_config(page_title="Crypto Price Predictions 🔮", layout="wide")
    st.title("Crypto Price Predictions 🔮")
    st.markdown("""
    **Descripción del Dashboard:**  
    Este dashboard ha sido desarrollado para predecir el precio futuro de criptomonedas utilizando múltiples fuentes de datos y técnicas avanzadas:
    - **Datos Históricos e Indicadores Técnicos:** Se extraen datos de yfinance y se calculan indicadores (RSI, MACD, ATR, Bollinger Bands, SMA) para analizar el mercado.
    - **Análisis de Sentimiento:** Se evalúa el “estado de ánimo” combinando información de noticias (NewsAPI) y el índice Fear & Greed, utilizando Transformers y TextBlob.
    - **Ensamble de Modelos:** Se entrenan un modelo LSTM, uno XGBoost y Prophet, que se combinan (50% LSTM, 30% XGBoost, 20% Prophet) para obtener una predicción robusta.
    - **Optimización Automática:** Optuna ajusta los hiperparámetros del modelo LSTM de forma ligera para reducir el tiempo de entrenamiento sin sacrificar precisión.
    """)
    st.sidebar.title("Configuración de Predicción")
    crypto_name = st.sidebar.selectbox("Seleccione una criptomoneda:", list(coincap_ids.keys()))
    coin_id = coincap_ids[crypto_name]

    use_custom_range = st.sidebar.checkbox("Habilitar rango de fechas", value=False)
    default_end = datetime.utcnow()
    default_start = default_end - timedelta(days=7)
    if use_custom_range:
        start_date = st.sidebar.date_input("Fecha de inicio", default_start.date())
        end_date = st.sidebar.date_input("Fecha de fin", default_end.date())
        if start_date > end_date:
            st.sidebar.error("La fecha de inicio no puede ser posterior a la de fin.")
            return
        if (end_date - start_date).days > 7:
            st.sidebar.warning("El rango excede 7 días. Se ajustará a 7 días.")
            end_date = start_date + timedelta(days=7)
        if start_date > datetime.utcnow().date():
            start_date = datetime.utcnow().date() - timedelta(days=7)
            st.sidebar.warning("La fecha de inicio no puede ser futura. Ajustado a 7 días atrás.")
        if end_date > datetime.utcnow().date():
            end_date = datetime.utcnow().date()
            st.sidebar.warning("La fecha de fin no puede ser futura. Ajustado a hoy.")
        end_date_with_offset = end_date + timedelta(days=1)
    else:
        start_date = None
        end_date_with_offset = None

    horizon = st.sidebar.slider("Días a predecir (corto plazo):", 1, 60, 5)
    show_stats = st.sidebar.checkbox("Mostrar estadísticas descriptivas", value=False)

    df_prices = load_crypto_data(coin_id, start_date, end_date_with_offset)
    if df_prices is not None and not df_prices.empty:
        fig_hist = px.line(
            df_prices,
            x="ds",
            y="close_price",
            title=f"Histórico de {crypto_name}",
            labels={"ds": "Fecha", "close_price": "Precio (USD)"}
        )
        fig_hist.update_layout(template="plotly_dark")
        fig_hist.update_xaxes(tickformat="%Y-%m-%d")
        st.plotly_chart(fig_hist, use_container_width=True)
        if show_stats:
            st.subheader("Estadísticas Descriptivas")
            st.write(df_prices["close_price"].describe())
    else:
        st.warning("No se pudieron cargar datos históricos para el rango seleccionado.")

    tabs = st.tabs([
        "Entrenamiento y Test",
        "Predicción de Precios (Corto Plazo)",
        "Predicción a Medio/Largo Plazo",
        "Análisis de Sentimientos",
        "Noticias Recientes"
    ])

    with tabs[0]:
        if st.button("Entrenar Modelo y Predecir"):
            result = train_and_predict_with_sentiment(coin_id, horizon, start_date, end_date_with_offset)
            if result:
                st.success("Entrenamiento y predicción completados!")
                st.write(f"Sentimiento Noticias ({result['symbol']}): {result['crypto_sent']:.2f}")
                st.write(f"Sentimiento Mercado (Fear & Greed): {result['market_sent']:.2f}")
                st.write(f"Gauge Combinado: {result['gauge_val']:.2f}")
                col1, col2 = st.columns(2)
                col1.metric("RMSE (Test)", f"{result['rmse']:.2f}", help="Error medio en USD.")
                col2.metric("MAPE (Test)", f"{result['mape']:.2f}%", help="Error porcentual medio.")
                test_dates = result["test_dates"][:min(len(result["test_dates"]), len(result["real_prices"]), len(result["test_preds"]))]
                real_prices = result["real_prices"][:len(test_dates)]
                test_preds = result["test_preds"][:len(test_dates)]
                fig_test = go.Figure()
                fig_test.add_trace(go.Scatter(
                    x=test_dates,
                    y=real_prices,
                    mode="lines",
                    name="Precio Real",
                    line=dict(color="#1f77b4", width=3, shape="spline")
                ))
                fig_test.add_trace(go.Scatter(
                    x=test_dates,
                    y=test_preds,
                    mode="lines",
                    name="Predicción (Ensamble)",
                    line=dict(color="#ff7f0e", width=3, dash="dash", shape="spline")
                ))
                fig_test.update_layout(
                    title=f"Precio Real vs. Predicción (Ensamble) - {result['symbol']}",
                    xaxis=dict(tickformat="%Y-%m-%d"),
                    template="plotly_dark",
                    xaxis_title="Fecha",
                    yaxis_title="Precio (USD)"
                )
                st.plotly_chart(fig_test, use_container_width=True)
                st.session_state["result"] = result

    with tabs[1]:
        if "result" in st.session_state and isinstance(st.session_state["result"], dict):
            result = st.session_state["result"]
        else:
            result = None
        if result:
            st.header(f"Predicción de Precios (Corto Plazo) - {result['symbol']}")
            last_date = result["df"]["ds"].iloc[-1].date()
            current_price = result["df"]["close_price"].iloc[-1]
            pred_series = np.concatenate(([current_price], result["future_preds"]))
            future_dates_display = [last_date] + [fd.date() for fd in result["future_dates"]]
            fig_future = go.Figure()
            fig_future.add_trace(go.Scatter(
                x=future_dates_display,
                y=pred_series,
                mode="lines+markers",
                name="Predicción (Ensamble)",
                line=dict(color="#ff7f0e", width=2, shape="spline")
            ))
            fig_future.update_layout(
                title=f"Predicción Futura ({horizon} días) - {result['symbol']}",
                template="plotly_dark",
                xaxis_title="Fecha",
                yaxis_title="Precio (USD)"
            )
            st.plotly_chart(fig_future, use_container_width=True)
            st.subheader("Resultados Numéricos (Corto Plazo)")
            df_future = pd.DataFrame({"Fecha": future_dates_display, "Predicción": pred_series})
            st.dataframe(df_future.style.format({"Predicción": "{:.2f}"}))
            st.download_button(
                label="Descargar predicciones (corto plazo) en CSV",
                data=df_future.to_csv(index=False).encode("utf-8"),
                file_name="predicciones_corto_plazo.csv",
                mime="text/csv"
            )
        else:
            st.info("Entrene el modelo primero para ver la predicción de corto plazo.")

    with tabs[2]:
        if "result" in st.session_state and isinstance(st.session_state["result"], dict):
            result = st.session_state["result"]
            if result:
                st.header(f"Predicción a Medio/Largo Plazo - {result['symbol']}")
                last_date = result["df"]["ds"].iloc[-1].date()
                _, forecast_long = medium_long_term_prediction(result["df"], days=180)
                forecast_long_part = forecast_long[["ds", "exp_yhat"]].tail(180)
                fig_long = go.Figure()
                fig_long.add_trace(go.Scatter(
                    x=result["df"]["ds"],
                    y=result["df"]["close_price"],
                    mode="lines",
                    name="Histórico",
                    line=dict(color="#1f77b4", width=2)
                ))
                fig_long.add_trace(go.Scatter(
                    x=forecast_long_part["ds"],
                    y=forecast_long_part["exp_yhat"],
                    mode="lines",
                    name="Predicción 180 días",
                    line=dict(color="#ff7f0e", width=2, dash="dash")
                ))
                fig_long.update_layout(
                    title="Predicción a 180 días (Medio/Largo Plazo) - Prophet",
                    template="plotly_dark",
                    xaxis_title="Fecha",
                    yaxis_title="Precio (USD)"
                )
                st.plotly_chart(fig_long, use_container_width=True)
                st.subheader("Valores Numéricos (Horizonte 180 días)")
                st.dataframe(
                    forecast_long_part.rename(columns={"exp_yhat": "Predicción (USD)"})
                    .style.format({"Predicción (USD)": "{:.2f}"})
                )
                st.download_button(
                    label="Descargar predicción (medio/largo plazo) en CSV",
                    data=forecast_long_part.to_csv(index=False).encode("utf-8"),
                    file_name="predicciones_medio_largo_plazo.csv",
                    mime="text/csv"
                )
            else:
                st.info("Entrene el modelo para ver la predicción a medio/largo plazo.")
        else:
            st.info("Entrene el modelo para ver la predicción a medio/largo plazo.")

    with tabs[3]:
        st.header("Análisis de Sentimientos")
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
            title={"text": f"Sentimiento - {coinid_to_symbol[coin_id]}", "x": 0.5, "xanchor": "center", "font": {"size": 24}},
            template="plotly_dark",
            height=400,
            margin=dict(l=20, r=20, t=80, b=20)
        )
        st.plotly_chart(fig_sentiment, use_container_width=True)
        st.write(f"**Sentimiento Noticias ({coinid_to_symbol[coin_id]}):** {crypto_sent:.2f}")
        st.write(f"**Sentimiento Mercado (Fear & Greed):** {market_sent:.2f}")
        st.write(f"**Gauge Value:** {gauge_val:.2f} → **{gauge_text}**")

    with tabs[4]:
        st.subheader(f"Últimas noticias sobre {crypto_name} ({coinid_to_symbol[coin_id]})")
        articles = get_newsapi_articles(coin_id, show_warning=True)
        if articles:
            st.markdown(
                """
                <style>
                .news-container {
                    display: grid;
                    grid-template-columns: repeat(auto-fill, minmax(280px, 1fr));
                    gap: 1rem;
                }
                .news-item {
                    background-color: #2c2c3e;
                    padding: 0.5rem;
                    border-radius: 5px;
                    border: 1px solid #4a4a6a;
                    max-width: 280px;
                }
                .news-item img {
                    width: 100%;
                    height: auto;
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
                @media (max-width: 1024px) {
                    .news-container {
                        grid-template-columns: repeat(auto-fill, minmax(200px, 1fr));
                    }
                }
                @media (max-width: 600px) {
                    .news-container {
                        grid-template-columns: 1fr;
                    }
                }
                </style>
                """,
                unsafe_allow_html=True
            )
            st.markdown("<div class='news-container'>", unsafe_allow_html=True)
            for article in articles:
                image_tag = f"<img src='{article['image']}' alt='Imagen' />" if article['image'] else ""
                st.markdown(
                    f"""
                    <div class='news-item'>
                        {image_tag}
                        <h4>{article['title']}</h4>
                        <p><em>{article['pubDate']}</em></p>
                        <p>{article['description']}</p>
                        <p><a href="{article['link']}" target="_blank">Leer más</a></p>
                    </div>
                    """,
                    unsafe_allow_html=True
                )
            st.markdown("</div>", unsafe_allow_html=True)
        else:
            st.warning("Oh, vaya, parece que hemos hecho más peticiones de las debidas a la API. Vuelve en 12 horas si quieres ver noticias :)")

# =============================================================================
# Pantalla de bienvenida (splash screen) con transición
# =============================================================================
def splash_screen():
    st.set_page_config(page_title="Crypto Price Predictions", layout="wide")
    st.title("Crypto Price Predictions 🔮")
    st.markdown("""
    <style>
    .fade-out {
        animation: fadeOut 1.5s forwards;
    }
    @keyframes fadeOut {
        from { opacity: 1; }
        to { opacity: 0; }
    }
    </style>
    **Bienvenido a Crypto Price Predictions**

    Este dashboard ha sido desarrollado para predecir el precio futuro de criptomonedas utilizando múltiples fuentes de datos y técnicas avanzadas:

    - **Datos Históricos e Indicadores Técnicos:**  
      Se extraen datos de yfinance y se calculan indicadores (RSI, MACD, ATR, Bollinger Bands, SMA) para analizar el mercado.
      
    - **Análisis de Sentimiento:**  
      Se evalúa el “estado de ánimo” combinando información de noticias (NewsAPI) y el índice Fear & Greed, utilizando Transformers y TextBlob.

    - **Ensamble de Modelos:**  
      Se entrenan un modelo LSTM, uno XGBoost y Prophet. Estos se combinan (50% LSTM, 30% XGBoost, 20% Prophet) para obtener una predicción robusta.

    - **Optimización Automática:**  
      Con Optuna se ajustan automáticamente los hiperparámetros del modelo LSTM en un proceso ligero para acelerar el entrenamiento.

    ¡Explora el dashboard y descubre las predicciones para tu criptomoneda favorita!
    """)
    if st.button("Comenzar"):
        st.markdown("<div class='fade-out'>Cargando dashboard...</div>", unsafe_allow_html=True)
        time.sleep(1.5)
        st.experimental_set_query_params(page="dashboard")

# =============================================================================
# Flujo principal
# =============================================================================
def run_dashboard():
    if st.experimental_get_query_params().get("page") != ["dashboard"]:
        splash_screen()
    else:
        main_app()

if __name__ == "__main__":
    run_dashboard()
