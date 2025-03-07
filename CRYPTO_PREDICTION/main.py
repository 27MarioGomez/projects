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
from tensorflow.keras.regularizers import l2

# Activar mixed precision si hay GPU (en CPU se ignora)
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
from joblib import Parallel, delayed
from textblob import TextBlob
from dateutil.parser import parse as date_parse
from urllib3.util.retry import Retry
from requests.adapters import HTTPAdapter
from newsapi import NewsApiClient
from prophet import Prophet
# Indicadores esenciales
from ta.momentum import RSIIndicator
from ta.trend import MACD, SMAIndicator, EMAIndicator
from ta.volatility import BollingerBands, AverageTrueRange
from ta.volume import OnBalanceVolumeIndicator
from transformers.pipelines import pipeline
from xgboost import XGBRegressor
from numba import njit

# Intentamos importar Keras Tuner (se instala como keras-tuner)
try:
    import keras_tuner as kt
except ModuleNotFoundError:
    import kerastuner as kt

# -----------------------------------------------------------------------------
# Configuración de la página (ancho completo)
# -----------------------------------------------------------------------------
st.set_page_config(page_title="Crypto Price Predictions 🔮", layout="wide")

# -----------------------------------------------------------------------------
# Definiciones globales (variables)
# -----------------------------------------------------------------------------
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

# -----------------------------------------------------------------------------
# Transformador para conservar DataFrame en imputación
# -----------------------------------------------------------------------------
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

# -----------------------------------------------------------------------------
# Transformador para selección de features
# -----------------------------------------------------------------------------
class FeatureSelector(BaseEstimator, TransformerMixin):
    """
    Realiza una selección automática de features mediante ElasticNetCV,
    refinando el resultado con la importancia calculada por XGBoost.
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
        enet = ElasticNetCV(cv=5, random_state=42).fit(X_arr, y_arr)
        coefs = enet.coef_
        initial_selected = [self.feature_cols[i] for i in range(len(self.feature_cols)) if abs(coefs[i]) > self.enet_threshold]
        if not initial_selected:
            initial_selected = self.feature_cols
        xgb = XGBRegressor(n_estimators=50, max_depth=3, learning_rate=0.1, random_state=42)
        xgb.fit(df[initial_selected].values, y_arr)
        importances = xgb.feature_importances_
        refined = [initial_selected[i] for i in range(len(initial_selected)) if importances[i] > self.importance_threshold]
        self.selected_features_ = refined if refined else initial_selected
        return self

    def transform(self, X):
        return X[self.selected_features_]

# -----------------------------------------------------------------------------
# Funciones de análisis de sentimiento
# -----------------------------------------------------------------------------
@st.cache_resource(show_spinner=False)
def load_sentiment_pipeline():
    # Especificamos modelo y revisión para evitar warning
    return pipeline("sentiment-analysis", model="distilbert/distilbert-base-uncased-finetuned-sst-2-english", revision="714eb0f")

def get_advanced_sentiment(text):
    pipe = load_sentiment_pipeline()
    result = pipe(text)[0]
    return 50 + (result["score"] * 50) if result["label"].upper() == "POSITIVE" else 50 - (result["score"] * 50)

@st.cache_data(ttl=43200)
def get_newsapi_articles(coin_id, show_warning=True):
    newsapi_key = st.secrets.get("newsapi_key", "")
    if not newsapi_key:
        st.error("Clave 'newsapi_key' no encontrada en Secrets.")
        return []
    try:
        query = f"{coin_id} crypto"
        newsapi = NewsApiClient(api_key=newsapi_key)
        data = newsapi.get_everything(q=query, language="en", sort_by="relevancy", page_size=5)
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
            st.warning("Se ha excedido el límite de peticiones. Vuelve en 12 horas.")
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
    return news_sent, market_sent, max(0, min(100, gauge_val))

def adjust_predictions_for_sentiment(preds_array, gauge_val, current_price):
    offset = (gauge_val - 50) / 50.0
    factor = offset * 0.05
    preds_adj = preds_array * (1 + factor)
    if gauge_val > 60 and preds_adj[0] < current_price:
        extra_factor = min((current_price / preds_adj[0] - 1), 0.10)
        preds_adj = preds_adj * (1 + extra_factor)
    return preds_adj

# -----------------------------------------------------------------------------
# Cálculo de indicadores técnicos (paralelización y vectorización)
# -----------------------------------------------------------------------------
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
        delayed(func)(df) for func in [compute_rsi, compute_macd, compute_bollinger_upper, compute_bollinger_lower, compute_sma50, compute_atr]
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

# -----------------------------------------------------------------------------
# Creación de secuencias (usando Numba)
# -----------------------------------------------------------------------------
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

# -----------------------------------------------------------------------------
# Modelo LSTM con Keras Tuner (espacio de búsqueda acotado)
# -----------------------------------------------------------------------------
def build_lstm_model_tuner(input_shape):
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

# -----------------------------------------------------------------------------
# Funciones para predicción iterativa
# -----------------------------------------------------------------------------
def iterative_lstm_forecast(model, current_input, scaler, feature_cols, horizon_days):
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

def ensemble_prediction(lstm_pred, xgb_pred, prophet_pred, w_lstm=0.6, w_xgb=0.2, w_prophet=0.2):
    return w_lstm * lstm_pred + w_xgb * xgb_pred + w_prophet * prophet_pred

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

# -----------------------------------------------------------------------------
# Función para cargar datos históricos
# -----------------------------------------------------------------------------
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
        st.warning("Datos no disponibles en yfinance.")
        return None
    df = df.reset_index()
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    df.rename(columns={"Date": "ds", "Close": "close_price", "Volume": "volume",
                       "High": "high", "Low": "low"}, inplace=True)
    df = compute_base_indicators(df)
    df = compute_additional_features(df)
    df.dropna(inplace=True)
    current_sent = get_news_sentiment(coin_id)
    df["sentiment"] = current_sent
    return df

# -----------------------------------------------------------------------------
# Función principal de entrenamiento y predicción
# -----------------------------------------------------------------------------
def train_and_predict_with_sentiment(coin_id, horizon_days, start_date=None, end_date=None, training_period_years=1):
    st.info("El proceso de entrenamiento y predicción puede tardar un poco. Por favor, espera...")
    progress_text = st.empty()
    progress_bar = st.progress(0)

    full_df = load_crypto_data(coin_id, start_date, end_date)
    if full_df is None or full_df.empty:
        st.error("No se pudo cargar el histórico.")
        return None

    # Filtrar para entrenamiento/predicción: usar solo los datos del período seleccionado
    last_date = full_df["ds"].max()
    period_start = last_date - pd.DateOffset(years=training_period_years)
    df_pred = full_df[full_df["ds"] >= period_start].copy()
    if df_pred.empty:
        st.error("No hay suficientes datos recientes para entrenamiento.")
        return None

    progress_text.text("Preparando dataset y escalando datos...")
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
        st.error("No hay suficientes datos para crear secuencias.")
        return None

    split = int(len(X) * 0.8)
    X_train, X_test = X[:split], X[split:]
    y_train, y_test = y[:split], y[split:]
    val_split = int(len(X_train) * 0.9)
    X_val, y_val = X_train[val_split:], y_train[val_split:]
    X_train, y_train = X_train[:val_split], y_train[:val_split]

    # Reducir épocas y usar early stopping en el tuner
    epochs = 8
    batch_size = 32
    input_shape = (window_size, len(selected_features))
    
    # Limpiar directorio de checkpoints para evitar incompatibilidades
    if os.path.exists('kt_dir'):
        shutil.rmtree('kt_dir')
    
    # Callbacks de early stopping y reduce LR
    callbacks = [
        tf.keras.callbacks.EarlyStopping(patience=3, restore_best_weights=True),
        tf.keras.callbacks.ReduceLROnPlateau(patience=2, factor=0.5, min_lr=1e-6)
    ]
    
    tuner = kt.Hyperband(
        build_lstm_model_tuner(input_shape),
        objective='val_loss',
        max_epochs=epochs,
        factor=3,
        directory='kt_dir',
        project_name=f'{coin_id}_crypto_lstm'
    )
    tuner.search(X_train, y_train, validation_data=(X_val, y_val), epochs=epochs, batch_size=batch_size, callbacks=callbacks, verbose=0)
    lstm_model = tuner.get_best_models(num_models=1)[0]

    progress_text.text("Realizando predicción en datos de test (LSTM)...")
    progress_bar.progress(40)
    preds_test_scaled = lstm_model.predict(X_test, verbose=0)
    reconst_test = np.concatenate([preds_test_scaled, np.zeros((len(preds_test_scaled), len(selected_features)-1))], axis=1)
    reconst_test_inv = pipe.named_steps['scaler'].inverse_transform(reconst_test)
    preds_test_log = reconst_test_inv[:, 0]
    lstm_test_preds = np.expm1(preds_test_log)

    reconst_y = np.concatenate([y_test.reshape(-1, 1), np.zeros((len(y_test), len(selected_features)-1))], axis=1)
    reconst_y_inv = pipe.named_steps['scaler'].inverse_transform(reconst_y)
    y_test_log = reconst_y_inv[:, 0]
    y_test_real = np.expm1(y_test_log)

    lstm_mape = np.mean(np.abs((y_test_real - lstm_test_preds) / np.maximum(np.abs(y_test_real), 1e-9))) * 100
    accuracy = max(0, 100 - lstm_mape)

    progress_text.text("Entrenando XGBoost y Prophet (ensamble)...")
    progress_bar.progress(55)
    X_train_val = np.concatenate([X_train, X_val], axis=0)
    y_train_val = np.concatenate([y_train, y_val], axis=0)
    X_train_val_flat = flatten_sequences(X_train_val)
    xgb_model = XGBRegressor(n_estimators=150, max_depth=6, learning_rate=0.05,
                             subsample=0.8, colsample_bytree=0.8)
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

    test_ens_preds = ensemble_prediction(lstm_test_preds, xgb_test_preds, prophet_test_preds, 0.6, 0.2, 0.2)
    ens_mape = np.mean(np.abs((y_test_real - test_ens_preds) / np.maximum(np.abs(y_test_real), 1e-9))) * 100
    ens_accuracy = max(0, 100 - ens_mape)

    progress_text.text("Realizando predicción futura...")
    progress_bar.progress(70)
    last_window = pipe.named_steps['scaler'].transform(df_pred[selected_features].values[-window_size:])
    current_input = last_window.reshape(1, window_size, len(selected_features))
    lstm_future_preds = iterative_lstm_forecast(lstm_model, current_input, pipe.named_steps['scaler'], selected_features, horizon_days)
    current_input_xgb = np.copy(last_window).reshape(1, window_size, len(selected_features))
    xgb_future_preds = iterative_xgb_forecast(xgb_model, current_input_xgb, pipe.named_steps['scaler'], selected_features, horizon_days)
    current_price = full_df["close_price"].iloc[-1]
    future_prophet2 = prophet_model.make_future_dataframe(periods=horizon_days)
    forecast2 = prophet_model.predict(future_prophet2)
    prophet_preds_log2 = forecast2["yhat"].tail(horizon_days).values
    prophet_future_preds = np.expm1(prophet_preds_log2)
    prophet_future_preds[0] = current_price
    final_future_preds = ensemble_prediction(lstm_future_preds, xgb_future_preds, prophet_future_preds, 0.6, 0.2, 0.2)
    final_future_preds[0] = current_price
    _, _, gauge_val = get_crypto_sentiment_combined(coin_id)
    final_future_preds = adjust_predictions_for_sentiment(final_future_preds, gauge_val, current_price)
    future_dates = pd.date_range(start=full_df["ds"].iloc[-1] + timedelta(days=1), periods=horizon_days).tolist()

    progress_text.text("¡Predicción completada con éxito!")
    progress_bar.progress(100)

    return {
        "df": full_df,
        "df_train": df_pred,
        "test_preds": test_ens_preds,
        "accuracy": ens_accuracy,
        "symbol": coinid_to_symbol[coin_id],
        "crypto_sent": get_news_sentiment(coin_id),
        "market_sent": get_fear_greed_index(),
        "gauge_val": gauge_val,
        "future_preds": final_future_preds,
        "future_dates": future_dates,
        "test_dates": full_df["ds"].iloc[-len(test_ens_preds):].values,
        "real_prices": y_test_real
    }

def main_app():
    st.title("Crypto Price Predictions 🔮")
    st.markdown("""
    **Descripción del Dashboard:**  
    Este sistema integra datos históricos obtenidos de yfinance, indicadores técnicos y análisis de sentimiento (noticias y Fear & Greed) para predecir el precio futuro de criptomonedas.  
    **Componentes del Modelo:**  
      - **Indicadores Técnicos:** Se calculan RSI, MACD, Bollinger Bands, SMA, ATR, OBV, EMA200, log_return, vol_30d y se utiliza el sentimiento.  
      - **Optimización de Features:** Se utiliza un pipeline que aplica imputación (mediana), selección automática de features (con ElasticNetCV refinado con XGBoost) y escalado, usando solo los datos del último año para entrenamiento sin afectar la visualización completa.  
      - **Análisis de Sentimiento:** Se combina el sentimiento derivado de noticias y el índice Fear & Greed para ajustar las predicciones.  
      - **Modelos de Predicción:** Se emplea un ensamble de:
          - **LSTM:** Hiperparámetros optimizados con Keras Tuner (Hyperband con búsqueda acotada) en 8 épocas.  
          - **XGBoost:** Para predicción iterativa a corto plazo.  
          - **Prophet:** Para predicciones a mediano/largo plazo, anclando el primer valor al precio actual.  
    **NFA:** Not Financial Advice.
    """)
    st.sidebar.title("Configuración de Predicción")
    crypto_name = st.sidebar.selectbox("Seleccione una criptomoneda:", list(coincap_ids.keys()))
    coin_id = coincap_ids[crypto_name]
    
    training_period = st.sidebar.select_slider(
        "Periodo de entrenamiento (años):",
        options=[1, 2, 3],
        value=1,
        help="A mayor período, mayor tiempo tardará en entrenar el modelo."
    )
    
    horizon = st.sidebar.slider("Días a predecir:", 1, 60, 5, help="Más días a predecir implican mayor tiempo de procesamiento.")
    
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
    else:
        st.warning("No se pudieron cargar datos históricos para el rango seleccionado.")

    tabs = st.tabs([
        "Entrenamiento del modelo",
        "Predicción a corto plazo",
        "Predicción a medio/largo plazo",
        "Análisis de sentimiento",
        "Noticias recientes"
    ])

    with tabs[0]:
        if st.button("Entrenar Modelo y Predecir"):
            result = train_and_predict_with_sentiment(coin_id, horizon, start_date, end_date_with_offset, training_period_years=training_period)
            if result:
                st.success("Entrenamiento y predicción completados!")
                st.write(f"Sentimiento Noticias ({result['symbol']}): {result['crypto_sent']:.2f}")
                st.write(f"Sentimiento Mercado (Fear & Greed): {result['market_sent']:.2f}")
                st.write(f"Gauge Combinado: {result['gauge_val']:.2f}")
                st.metric("Precisión (Test)", f"{result['accuracy']:.2f}%")
                st.session_state["result"] = result

    with tabs[1]:
        if "result" in st.session_state and isinstance(st.session_state["result"], dict):
            result = st.session_state["result"]
        else:
            result = None
        if result:
            st.header(f"Predicción a corto plazo - {result['symbol']}")
            last_date = result["df"]["ds"].iloc[-1].date()
            current_price = result["df"]["close_price"].iloc[-1]
            pred_series = np.concatenate(([current_price], result["future_preds"]))
            future_dates_display = [last_date] + [fd.date() for fd in result["future_dates"]]
            fig_future = go.Figure()
            fig_future.add_trace(go.Scatter(
                x=future_dates_display,
                y=pred_series,
                mode="lines+markers",
                name=f"Predicción – {result['symbol']}",
                line=dict(color="#ff7f0e", width=2, shape="spline")
            ))
            fig_future.update_layout(
                title=f"Predicción Futura ({horizon} días) - {result['symbol']}",
                template="plotly_dark",
                xaxis_title="Fecha",
                yaxis_title="Precio (USD)"
            )
            st.plotly_chart(fig_future, use_container_width=True)
            st.header("Resultados Numéricos (Corto Plazo)")
            df_future = pd.DataFrame({"Fecha": future_dates_display, "Predicción": pred_series})
            st.dataframe(df_future.style.format({"Predicción": "{:.2f}"}))
            st.download_button(
                label="Descargar predicciones (corto plazo) en CSV",
                data=df_future.to_csv(index=False).encode("utf-8"),
                file_name="predicciones_corto_plazo.csv",
                mime="text/csv"
            )
        else:
            st.info("Entrene el modelo primero para ver la predicción a corto plazo.")

    with tabs[2]:
        if "result" in st.session_state and isinstance(st.session_state["result"], dict):
            result = st.session_state["result"]
            if result:
                st.header(f"Predicción a medio/largo plazo - {result['symbol']}")
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
                st.header("Valores Numéricos (Horizonte 180 días)")
                styled_forecast = forecast_long_part.copy()
                styled_forecast.columns = ["Fecha", "Predicción (USD)"]
                st.dataframe(styled_forecast.style.format({"Predicción (USD)": "{:.2f}"}))
                st.download_button(
                    label="Descargar predicción (medio/largo plazo) en CSV",
                    data=styled_forecast.to_csv(index=False).encode("utf-8"),
                    file_name="predicciones_medio_largo_plazo.csv",
                    mime="text/csv"
                )
            else:
                st.info("Entrene el modelo para ver la predicción a medio/largo plazo.")
        else:
            st.info("Entrene el modelo para ver la predicción a medio/largo plazo.")

    with tabs[3]:
        st.header("Análisis de sentimiento")
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
        st.subheader(f"Noticias recientes sobre {crypto_name} ({coinid_to_symbol[coin_id]})")
        articles = get_newsapi_articles(coin_id, show_warning=True)
        if articles:
            # Estilos para scroll horizontal y tarjetas de tamaño fijo
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
                    display: block;            /* Para que ocupe toda la línea */
                    margin: 0.5rem auto;      /* Auto para centrar horizontalmente */
                    padding: 0.4rem 0.8rem;
                    background-color: #fff;   /* Fondo blanco */
                    color: #000;              /* Texto negro */
                    text-decoration: none;    /* Sin subrayado */
                    border-radius: 3px;
                    text-align: center;
                    font-size: 0.8rem;
                    font-weight: 600;         /* (Opcional) un poco más de grosor */
                }
                .read-more-btn:hover {
                    background-color: #e6e6e6; /* Un gris suave al pasar el ratón */
                    color: #000;              /* Texto sigue siendo negro */
}

                </style>
                """,
                unsafe_allow_html=True
            )
            st.markdown("<div class='news-container'>", unsafe_allow_html=True)
            for article in articles:
                image_tag = ""
                if article['image']:
                    image_tag = f"<img src='{article['image']}' alt='Imagen'/>"
                link_button = f"<a href='{article['link']}' target='_blank' class='read-more-btn'>Leer más</a>"
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
            st.warning("Se ha superado el límite de peticiones. Vuelve en 12 horas para ver más noticias.")

if __name__ == "__main__":
    main_app()
