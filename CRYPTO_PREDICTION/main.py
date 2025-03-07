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

# Activar mixed precision si existe GPU (en CPU se ignora)
if tf.config.list_physical_devices('GPU'):
    from tensorflow.keras.mixed_precision import set_global_policy
    set_global_policy('mixed_float16')

import yfinance as yf
import requests
import certifi
import os
from sklearn.preprocessing import MinMaxScaler
from textblob import TextBlob
from dateutil.parser import parse as date_parse
from urllib3.util.retry import Retry
from requests.adapters import HTTPAdapter
from newsapi import NewsApiClient
from prophet import Prophet
from ta.momentum import RSIIndicator
from ta.trend import MACD, SMAIndicator, EMAIndicator
from ta.volatility import BollingerBands, AverageTrueRange
from ta.volume import OnBalanceVolumeIndicator
from transformers.pipelines import pipeline
from xgboost import XGBRegressor

# Se intenta importar keras_tuner; si falla, se intenta con kerastuner
try:
    import keras_tuner as kt
except ModuleNotFoundError:
    import kerastuner as kt

# -----------------------------------------------------------------------------
# Configuración inicial de la página y de la sesión HTTP
# -----------------------------------------------------------------------------
st.set_page_config(page_title="Crypto Price Predictions 🔮", layout="wide")
os.environ["REQUESTS_CA_BUNDLE"] = certifi.where()
session = requests.Session()
retry = Retry(total=5, backoff_factor=1, status_forcelist=[429, 500, 502, 503, 504])
adapter = HTTPAdapter(max_retries=retry)
session.mount("https://", adapter)

# -----------------------------------------------------------------------------
# Diccionarios de criptomonedas y mapeo
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
# Funciones de análisis de sentimiento
# -----------------------------------------------------------------------------
@st.cache_resource(show_spinner=False)
def load_sentiment_pipeline():
    """Carga la pipeline de Transformers para análisis de sentimiento."""
    return pipeline("sentiment-analysis")

def get_advanced_sentiment(text):
    """Calcula el sentimiento con Transformers en rango [0,100]."""
    pipe = load_sentiment_pipeline()
    result = pipe(text)[0]
    return 50 + (result["score"] * 50) if result["label"].upper() == "POSITIVE" else 50 - (result["score"] * 50)

@st.cache_data(ttl=43200)
def get_newsapi_articles(coin_id, show_warning=True):
    """
    Solicita 5 artículos a NewsAPI. Controla el rate-limit y actualiza según el límite.
    """
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
            st.warning("Oh, vaya, parece que hemos superado el límite de peticiones. Vuelve en 12 horas si quieres ver noticias :)")
        elif show_warning:
            st.error(f"Error al obtener noticias: {e}")
        return []

def get_news_sentiment(coin_id):
    """Obtiene el sentimiento promedio a partir de artículos (TextBlob + Transformers)."""
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
    """Obtiene el índice Fear & Greed; retorna 50.0 si falla."""
    try:
        data = requests.get("https://api.alternative.me/fng/?format=json", timeout=10).json()
        return float(data["data"][0]["value"])
    except Exception:
        st.warning("Índice Fear & Greed no disponible. Se usará 50.0.")
        return 50.0

def get_crypto_sentiment_combined(coin_id):
    """Combina sentimiento de noticias y Fear & Greed para obtener gauge_val."""
    news_sent = get_news_sentiment(coin_id)
    market_sent = get_fear_greed_index()
    gauge_val = 50 + (news_sent - market_sent)
    gauge_val = max(0, min(100, gauge_val))
    return news_sent, market_sent, gauge_val

def adjust_predictions_for_sentiment(preds_array, gauge_val, current_price):
    """
    Ajuste final: aplica ±5% según gauge_val.
    Además, para corto plazo bullish, si la predicción es inferior al precio actual,
    se ajusta para que no caiga por debajo del valor actual.
    """
    offset = (gauge_val - 50) / 50.0
    factor = offset * 0.05  # factor en [-0.05, +0.05]
    preds_adj = preds_array * (1 + factor)
    if gauge_val > 60 and preds_adj[0] < current_price:
        extra_factor = min((current_price / preds_adj[0] - 1), 0.10)
        preds_adj = preds_adj * (1 + extra_factor)
    return preds_adj

# -----------------------------------------------------------------------------
# Cálculo de indicadores técnicos y features adicionales
# -----------------------------------------------------------------------------
def compute_base_indicators(df):
    """
    Calcula indicadores básicos: RSI(14), MACD, Bollinger Bands, SMA50 y ATR.
    Se realiza forward-fill para evitar huecos.
    """
    df["RSI"] = RSIIndicator(close=df["close_price"], window=14).rsi()
    df["rsi_norm"] = df["RSI"] / 100.0

    macd_calc = MACD(close=df["close_price"], window_fast=12, window_slow=26, window_sign=9)
    df["macd"] = macd_calc.macd()

    bb = BollingerBands(close=df["close_price"], window=20, window_dev=2)
    df["bollinger_upper"] = bb.bollinger_hband()
    df["bollinger_lower"] = bb.bollinger_lband()

    df["sma50"] = SMAIndicator(close=df["close_price"], window=50).sma_indicator()

    atr_calc = AverageTrueRange(high=df["high"], low=df["low"], close=df["close_price"], window=14)
    df["atr"] = atr_calc.average_true_range()

    df.ffill(inplace=True)
    return df

def compute_additional_features(df):
    """
    Añade features adicionales: log_sma50, log_return, volatilidad a 30 días, OBV y EMA(200).
    """
    df["log_sma50"] = np.log1p(df["sma50"])
    df["log_return"] = np.log(df["close_price"] / df["close_price"].shift(1)).fillna(0.0)
    df["vol_30d"] = df["log_return"].rolling(window=30).std().fillna(0.0)
    obv_calc = OnBalanceVolumeIndicator(close=df["close_price"], volume=df["volume"])
    df["obv"] = obv_calc.on_balance_volume()
    ema200_calc = EMAIndicator(close=df["close_price"], window=200)
    df["ema200"] = ema200_calc.ema_indicator()
    return df

@st.cache_data
def load_crypto_data(coin_id, start_date=None, end_date=None):
    """
    Descarga datos desde yfinance, aplica indicadores y añade el sentimiento actual.
    """
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

def create_sequences(data, window_size):
    """
    Genera secuencias de longitud 'window_size' para entrenar la LSTM.
    La primera columna de 'data' (log_price) se usa como target.
    """
    if len(data) <= window_size:
        return None, None
    X, y = [], []
    for i in range(window_size, len(data)):
        X.append(data[i-window_size:i])
        y.append(data[i, 0])
    return np.array(X), np.array(y)

def flatten_sequences(X_seq):
    """
    Aplana secuencias (samples, timesteps, features) a (samples, timesteps*features) para XGBoost.
    """
    return X_seq.reshape((X_seq.shape[0], X_seq.shape[1] * X_seq.shape[2]))

def build_lstm_model_tuner(input_shape):
    """
    Función para construir la LSTM usando Keras Tuner.
    Se define un espacio de búsqueda para hiperparámetros relevantes.
    """
    def model_builder(hp):
        lstm_units1 = hp.Int('lstm_units1', min_value=64, max_value=256, step=32)
        lstm_units2 = hp.Int('lstm_units2', min_value=32, max_value=128, step=16)
        dropout_rate = hp.Float('dropout_rate', 0.1, 0.5, step=0.05)
        dense_units = hp.Int('dense_units', min_value=50, max_value=150, step=25)
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

def ensemble_prediction(lstm_pred, xgb_pred, prophet_pred, w_lstm=0.6, w_xgb=0.2, w_prophet=0.2):
    """
    Combina las predicciones de LSTM, XGBoost y Prophet con pesos fijos.
    """
    return w_lstm * lstm_pred + w_xgb * xgb_pred + w_prophet * prophet_pred

def medium_long_term_prediction(df, days=180, current_price=None):
    """
    Construye un modelo Prophet con log(price) y predice a 'days' días.
    Se revierte la transformación logarítmica con expm1.
    Si se proporciona current_price, se fuerza que la primera predicción sea el precio actual.
    """
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

def train_and_predict_with_sentiment(coin_id, horizon_days, start_date=None, end_date=None):
    """
    Entrena el modelo (LSTM con hiperparámetros optimizados, XGBoost y Prophet) y realiza la predicción.
    Se entrena usando datos históricos y se testea con los datos más recientes (separación cronológica).
    Se fuerza que la primera predicción sea igual al precio actual para coherencia entre horizontes.
    """
    st.info("Esto puede tardar un poco, por favor espera...")

    progress_text = st.empty()
    progress_bar = st.progress(0)

    # Definir ventana según horizonte: corta si ≤30 días, más amplia para horizontes mayores.
    if horizon_days <= 30:
        window_size = 60
    else:
        window_size = 90

    progress_text.text("Cargando datos e indicadores...")
    progress_bar.progress(5)
    df = load_crypto_data(coin_id, start_date, end_date)
    if df is None or df.empty:
        st.error("No se pudo cargar el histórico.")
        return None

    progress_text.text("Preparando dataset y escalando...")
    progress_bar.progress(25)
    df["log_price"] = np.log1p(df["close_price"])

    feature_cols = [
        "log_price",
        "volume",
        "high",
        "low",
        "rsi_norm",
        "macd",
        "bollinger_upper",
        "bollinger_lower",
        "atr",
        "obv",
        "ema200",
        "log_sma50",
        "log_return",
        "vol_30d",
        "sentiment"
    ]
    data_array = df[feature_cols].values
    scaler = MinMaxScaler()
    scaled_data = scaler.fit_transform(data_array)

    X, y = create_sequences(scaled_data, window_size)
    if X is None:
        st.error("No hay suficientes datos para secuencias.")
        return None

    # Separación cronológica: se usa el 80% más antiguo para entrenamiento y el 20% más reciente para test.
    split = int(len(X) * 0.8)
    X_train, X_test = X[:split], X[split:]
    y_train, y_test = y[:split], y[split:]
    val_split = int(len(X_train) * 0.9)
    X_val, y_val = X_train[val_split:], y_train[val_split:]
    X_train, y_train = X_train[:val_split], y_train[:val_split]

    # Configuración de hiperparámetros para Keras Tuner (usando Hyperband y Random Search)
    epochs = 25
    batch_size = 32
    input_shape = (window_size, len(feature_cols))
    tuner = kt.Hyperband(
        build_lstm_model_tuner(input_shape),
        objective='val_loss',
        max_epochs=epochs,
        factor=3,
        directory='kt_dir',
        project_name=f'{coin_id}_crypto_lstm'
    )
    tuner.search(X_train, y_train, validation_data=(X_val, y_val), epochs=epochs, batch_size=batch_size, verbose=0)
    lstm_model = tuner.get_best_models(num_models=1)[0]

    progress_text.text("Realizando predicción en test (LSTM)...")
    progress_bar.progress(40)
    preds_test_scaled = lstm_model.predict(X_test, verbose=0)
    reconst_test = np.concatenate([preds_test_scaled, np.zeros((len(preds_test_scaled), len(feature_cols)-1))], axis=1)
    reconst_test_inv = scaler.inverse_transform(reconst_test)
    preds_test_log = reconst_test_inv[:, 0]
    lstm_test_preds = np.expm1(preds_test_log)

    reconst_y = np.concatenate([y_test.reshape(-1,1), np.zeros((len(y_test), len(feature_cols)-1))], axis=1)
    reconst_y_inv = scaler.inverse_transform(reconst_y)
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
    xgb_reconst = np.concatenate([xgb_test_scaled.reshape(-1,1), np.zeros((len(xgb_test_scaled), len(feature_cols)-1))], axis=1)
    xgb_test_inv = scaler.inverse_transform(xgb_reconst)
    xgb_test_log = xgb_test_inv[:, 0]
    xgb_test_preds = np.expm1(xgb_test_log)

    # Modelo Prophet para test
    df_prophet = df[["ds", "close_price"]].copy()
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

    # Predicción futura LSTM (iterativa)
    last_window = scaler.transform(df[feature_cols].values[-window_size:])
    current_input = last_window.reshape(1, window_size, len(feature_cols))
    future_preds_log_lstm = []
    for _ in range(horizon_days):
        p_scaled = lstm_model.predict(current_input, verbose=0)[0][0]
        reconst_f = np.concatenate([[p_scaled], np.zeros((len(feature_cols)-1,))]).reshape(1, -1)
        inv_f = scaler.inverse_transform(reconst_f)
        plog = inv_f[0, 0]
        future_preds_log_lstm.append(plog)
        new_feature = np.copy(current_input[:, -1:, :])
        new_feature[0, 0, 0] = p_scaled
        current_input = np.append(current_input[:, 1:, :], new_feature, axis=1)
    lstm_future_preds = np.expm1(np.array(future_preds_log_lstm))

    # Predicción futura con XGBoost (iterativa)
    current_input_xgb = np.copy(last_window).reshape(1, window_size, len(feature_cols))
    xgb_future_preds_log = []
    for _ in range(horizon_days):
        X_last_flat = flatten_sequences(current_input_xgb)
        xgb_p_scaled = xgb_model.predict(X_last_flat)[0]
        reconst_xgb = np.concatenate([[xgb_p_scaled], np.zeros((len(feature_cols)-1,))]).reshape(1, -1)
        inv_xgb = scaler.inverse_transform(reconst_xgb)
        xgb_log = inv_xgb[0, 0]
        xgb_future_preds_log.append(xgb_log)
        new_feature_xgb = np.copy(current_input_xgb[:, -1:, :])
        new_feature_xgb[0, 0, 0] = xgb_p_scaled
        current_input_xgb = np.append(current_input_xgb[:, 1:, :], new_feature_xgb, axis=1)
    xgb_future_preds = np.expm1(np.array(xgb_future_preds_log))

    # Predicción futura con Prophet
    current_price = df["close_price"].iloc[-1]
    future_prophet2 = prophet_model.make_future_dataframe(periods=horizon_days)
    forecast2 = prophet_model.predict(future_prophet2)
    prophet_preds_log2 = forecast2["yhat"].tail(horizon_days).values
    prophet_future_preds = np.expm1(prophet_preds_log2)
    prophet_future_preds[0] = current_price  # Forzar coherencia

    # Ensamble final
    final_future_preds = ensemble_prediction(lstm_future_preds, xgb_future_preds, prophet_future_preds, 0.6, 0.2, 0.2)
    final_future_preds[0] = current_price  # Forzar que el primer valor sea igual al precio actual

    # Ajuste final según sentimiento
    _, _, gauge_val = get_crypto_sentiment_combined(coin_id)
    final_future_preds = adjust_predictions_for_sentiment(final_future_preds, gauge_val, current_price)

    last_date = df["ds"].iloc[-1]
    future_dates = pd.date_range(start=last_date + timedelta(days=1), periods=horizon_days).tolist()

    progress_text.text("¡Predicción completada con éxito!")
    progress_bar.progress(100)

    return {
        "df": df,
        "test_preds": test_ens_preds,
        "accuracy": ens_accuracy,
        "symbol": coinid_to_symbol[coin_id],
        "crypto_sent": get_news_sentiment(coin_id),
        "market_sent": get_fear_greed_index(),
        "gauge_val": gauge_val,
        "future_preds": final_future_preds,
        "future_dates": future_dates,
        "test_dates": df["ds"].iloc[-len(test_ens_preds):].values,
        "real_prices": y_test_real
    }

def main_app():
    """
    Interfaz principal del dashboard.
    """
    st.title("Crypto Price Predictions 🔮")
    st.markdown("""
    **Descripción del Dashboard:**  
    Este sistema integra datos históricos de yfinance, indicadores técnicos y análisis de sentimiento para predecir el precio futuro de criptomonedas.  
    - Se emplean dos ventanas de entrenamiento: 60 días para horizontes cortos (≤ 30 días) y 90 días para horizontes mayores.  
    - Se utilizan transformaciones logarítmicas y normalización para estabilizar los datos.  
    - El modelo final es un ensamble de LSTM (60%), XGBoost (20%) y Prophet (20%), con optimización de hiperparámetros vía Keras Tuner (Hyperband + Random Search).  
    - Se fuerza que el precio actual sea el punto de partida en todas las predicciones para mantener coherencia entre horizontes.  
    - **NFA: Not Financial Advice.**
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

    horizon = st.sidebar.slider("Días a predecir:", 1, 60, 5)
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
        "Entrenamiento del modelo",
        "Predicción a corto plazo",
        "Predicción a medio/largo plazo",
        "Análisis de sentimiento",
        "Noticias recientes"
    ])

    with tabs[0]:
        if st.button("Entrenar Modelo y Predecir"):
            result = train_and_predict_with_sentiment(coin_id, horizon, start_date, end_date_with_offset)
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
                st.subheader("Valores Numéricos (Horizonte 180 días)")
                styled_forecast = forecast_long_part.copy()
                styled_forecast.columns = ["Fecha", "Predicción (USD)"]
                st.dataframe(
                    styled_forecast.style.format({"Predicción (USD)": "{:.2f}"})
                )
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
            st.warning("Oh, vaya, parece que hemos superado el límite de peticiones. Vuelve en 12 horas si quieres ver noticias :)")

if __name__ == "__main__":
    main_app()
