#########################
# main.py (actualizado con mejoras de LunarCrush)
#########################

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import requests
from datetime import datetime, date, timedelta
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, Bidirectional, LSTM, Dense, Dropout
from tensorflow.keras.optimizers import Adam
import time
import certifi
import os
import tweepy
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

# Configurar certificados SSL para requests y tweepy, asegurando conexiones seguras
os.environ["REQUESTS_CA_BUNDLE"] = certifi.where()
os.environ["SSL_CERT_FILE"] = certifi.where()

##############################################
# Funciones de apoyo
##############################################

def robust_mape(y_true, y_pred, eps=1e-9):
    """
    Calcula el Mean Absolute Percentage Error (MAPE) evitando divisiones por cero.
    
    Args:
        y_true (np.ndarray): Valores reales.
        y_pred (np.ndarray): Valores predichos.
        eps (float): Pequeño valor para evitar divisiones por cero.
    
    Returns:
        float: MAPE en porcentaje.
    """
    return np.mean(np.abs((y_true - y_pred) / np.maximum(np.abs(y_true), eps))) * 100

# Diccionarios con IDs y símbolos de criptomonedas
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
crypto_hashtags = {
    "Bitcoin (BTC)": "#Bitcoin",
    "Ethereum (ETH)": "#Ethereum",
    "Ripple (XRP)": "#XRP",
    "Binance Coin (BNB)": "#Binance",
    "Cardano (ADA)": "#Cardano",
    "Solana (SOL)": "#Solana",
    "Dogecoin (DOGE)": "#Dogecoin",
    "Polkadot (DOT)": "#Polkadot",
    "Polygon (MATIC)": "#Polygon",
    "Litecoin (LTC)": "#Litecoin",
    "TRON (TRX)": "#TRON",
    "Stellar (XLM)": "#Stellar"
}
lunarcrush_symbols = {
    "Bitcoin (BTC)": "BTC",
    "Ethereum (ETH)": "ETH",
    "Ripple (XRP)": "XRP",
    "Binance Coin (BNB)": "BNB",
    "Cardano (ADA)": "ADA",
    "Solana (SOL)": "SOL",
    "Dogecoin (DOGE)": "DOGE",
    "Polkadot (DOT)": "DOT",
    "Polygon (MATIC)": "MATIC",
    "Litecoin (LTC)": "LTC",
    "TRON (TRX)": "TRX",
    "Stellar (XLM)": "XLM"
}
trusted_accounts = ["@CoinMarketCap", "@CoinDesk", "@Binance", "@Krakenfx"]

##############################################
# Descarga de datos desde CoinCap (intervalo diario)
##############################################
@st.cache_data
def load_coincap_data(coin_id, start_ms=None, end_ms=None, max_retries=3):
    """
    Descarga datos históricos diarios de precios y volumen de una criptomoneda desde CoinCap.
    
    Args:
        coin_id (str): ID de la criptomoneda en CoinCap.
        start_ms (int, optional): Timestamp de inicio en milisegundos.
        end_ms (int, optional): Timestamp de fin en milisegundos.
        max_retries (int): Número máximo de reintentos en caso de fallo.
    
    Returns:
        pd.DataFrame: DataFrame con columnas 'ds', 'close_price', y 'volume', o None si falla.
    """
    url = f"https://api.coincap.io/v2/assets/{coin_id}/history?interval=d1"
    if start_ms is not None and end_ms is not None:
        url += f"&start={start_ms}&end={end_ms}"
    headers = {"User-Agent": "Mozilla/5.0"}
    for attempt in range(max_retries):
        try:
            resp = requests.get(url, headers=headers, verify=certifi.where(), timeout=10)
            if resp.status_code == 200:
                data = resp.json()
                if "data" not in data:
                    st.warning("CoinCap: 'data' faltante.")
                    return None
                df = pd.DataFrame(data["data"])
                if df.empty:
                    st.info("CoinCap devolvió datos vacíos. Reajusta el rango de fechas.")
                    return None
                if "time" not in df.columns or "priceUsd" not in df.columns:
                    st.warning("CoinCap: Faltan las columnas 'time' o 'priceUsd'.")
                    return None
                df["ds"] = pd.to_datetime(df["time"], unit="ms", errors="coerce")
                df["close_price"] = pd.to_numeric(df["priceUsd"], errors="coerce")
                df["volume"] = pd.to_numeric(df.get("volumeUsd", 0), errors="coerce").fillna(0.0)
                df = df[["ds", "close_price", "volume"]].dropna(subset=["ds", "close_price"])
                df.sort_values(by="ds", inplace=True)
                df.reset_index(drop=True, inplace=True)
                df = df[df["close_price"] > 0].copy()
                return df
            elif resp.status_code == 429:
                st.warning(f"CoinCap: Error 429 en intento {attempt+1}. Esperando {15*(attempt+1)}s...")
                time.sleep(15*(attempt+1))
            elif resp.status_code == 400:
                st.info("CoinCap: (400) Parámetros inválidos o rango excesivo.")
                return None
            else:
                st.info(f"CoinCap: status code {resp.status_code}. Revisa parámetros.")
                return None
        except requests.exceptions.SSLError as e:
            st.error(f"Error SSL al conectar con CoinCap: {e}")
            return None
    st.info("CoinCap: Máx reintentos sin éxito.")
    return None

##############################################
# Descarga de datos desde LunarCrush
##############################################
@st.cache_data
def load_lunarcrush_data(symbol, api_key, max_retries=3):
    """
    Obtiene métricas sociales y de mercado de LunarCrush para una criptomoneda específica.
    
    Args:
        symbol (str): Símbolo de la criptomoneda en LunarCrush (ej. "BTC").
        api_key (str): Clave API de LunarCrush desde st.secrets.
        max_retries (int): Número máximo de reintentos en caso de fallo.
    
    Returns:
        dict: Métricas seleccionadas o None si falla.
    """
    url = "https://api.lunarcrush.com/v1/coin"
    params = {
        "key": api_key,
        "symbol": symbol,
        "data_points": 1,  # Minimizar uso en plan gratuito, datos más recientes
        "interval": "day"
    }
    headers = {"User-Agent": "Mozilla/5.0"}
    for attempt in range(max_retries):
        try:
            resp = requests.get(url, params=params, headers=headers, timeout=10)
            if resp.status_code == 200:
                data = resp.json()
                if "data" not in data or not data["data"]:
                    st.warning(f"LunarCrush: No hay datos para {symbol}.")
                    return None
                coin_data = data["data"][0]
                metrics = {
                    "social_volume": coin_data.get("social_volume", 0),
                    "social_score": coin_data.get("social_score", 0),
                    "social_dominance": coin_data.get("social_dominance", 0.0),
                    "average_sentiment": coin_data.get("average_sentiment", 3),  # 1-5, 3 neutral
                    "social_contributors": coin_data.get("social_contributors", 0)
                }
                return metrics
            elif resp.status_code == 429:
                st.warning(f"LunarCrush: Límite alcanzado (429). Esperando {15*(attempt+1)}s...")
                time.sleep(15*(attempt+1))
            else:
                st.info(f"LunarCrush: status code {resp.status_code}.")
                return None
        except requests.exceptions.RequestException as e:
            st.error(f"Error al conectar con LunarCrush: {e}")
            return None
    st.info("LunarCrush: Máx reintentos sin éxito.")
    return None

##############################################
# Creación de secuencias para LSTM
##############################################
def create_sequences(data, window_size=30):
    """
    Crea secuencias temporales de tamaño 'window_size' a partir de datos multidimensionales.
    
    Args:
        data (np.ndarray): Array con datos (precio, sentimiento, etc.).
        window_size (int): Tamaño de la ventana de tiempo.
    
    Returns:
        tuple: (X, y) con secuencias de entrada y valores objetivo, o (None, None) si insuficientes datos.
    """
    if len(data) <= window_size:
        st.warning(f"No hay datos suficientes para una ventana de {window_size} días.")
        return None, None
    X, y = [], []
    for i in range(window_size, len(data)):
        X.append(data[i - window_size : i])
        y.append(data[i, 0])  # Precio como objetivo
    return np.array(X), np.array(y)

##############################################
# Modelo LSTM ajustado para múltiples características
##############################################
def build_lstm_model(input_shape, learning_rate=0.001):
    """
    Construye un modelo LSTM con Conv1D y Bidirectional LSTM para múltiples características.
    
    Args:
        input_shape (tuple): Forma de entrada (window_size, n_features).
        learning_rate (float): Tasa de aprendizaje para Adam.
    
    Returns:
        Sequential: Modelo Keras compilado.
    """
    model = Sequential()
    model.add(Conv1D(filters=32, kernel_size=3, activation="relu", input_shape=input_shape))
    model.add(Bidirectional(LSTM(64, return_sequences=True)))
    model.add(Dropout(0.3, name="dropout_1"))
    model.add(Bidirectional(LSTM(64, return_sequences=True)))
    model.add(Dropout(0.3, name="dropout_2"))
    model.add(Bidirectional(LSTM(64, return_sequences=False)))
    model.add(Dropout(0.3, name="dropout_3"))
    model.add(Dense(1))  # Salida: predicción del precio
    opt = Adam(learning_rate=learning_rate)
    model.compile(optimizer=opt, loss="mean_squared_error")
    return model

##############################################
# Función aislada para entrenar el modelo
##############################################
def train_model(X_train, y_train, X_val, y_val, input_shape, epochs, batch_size, learning_rate):
    """
    Entrena el modelo LSTM, limpiando el estado global para evitar conflictos en Streamlit.
    
    Args:
        X_train (np.ndarray): Datos de entrenamiento (shape: (None, window_size, n_features)).
        y_train (np.ndarray): Valores objetivo (precio, shape: (None, 1)).
        X_val (np.ndarray): Datos de validación.
        y_val (np.ndarray): Valores objetivo de validación.
        input_shape (tuple): Forma de entrada (window_size, n_features).
        epochs (int): Número de épocas.
        batch_size (int): Tamaño del batch.
        learning_rate (float): Tasa de aprendizaje.
    
    Returns:
        model: Modelo LSTM entrenado.
    """
    tf.keras.backend.clear_session()
    model = build_lstm_model(input_shape, learning_rate=learning_rate)
    model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=epochs,
        batch_size=batch_size,
        verbose=1
    )
    return model

##############################################
# Ajuste dinámico de hiperparámetros
##############################################
def get_dynamic_params(df, horizon_days):
    """
    Calcula hiperparámetros dinámicos basados en características del dataset.
    
    Args:
        df (pd.DataFrame): DataFrame con datos históricos de precios.
        horizon_days (int): Días a predecir.
    
    Returns:
        tuple: (window_size, epochs, batch_size, learning_rate) ajustados dinámicamente.
    """
    data_len = len(df)
    volatility = df["close_price"].pct_change().std()
    mean_price = df["close_price"].mean()
    window_size = min(max(10, horizon_days * 2), min(60, data_len // 2))
    epochs = min(50, max(20, int(data_len / 100) + int(volatility * 100)))
    batch_size = 16 if volatility > 0.05 or data_len < 500 else 32
    learning_rate = 0.0005 if mean_price > 1000 or volatility > 0.1 else 0.001
    return window_size, epochs, batch_size, learning_rate

##############################################
# Entrenamiento y predicción con LunarCrush y X
##############################################
def train_and_predict_with_sentiment(
    coin_id, symbol, use_custom_range, start_ms, end_ms, horizon_days=30, test_size=0.2
):
    """
    Entrena un modelo LSTM con datos de CoinCap, sentimiento de X y métricas de LunarCrush.
    
    Args:
        coin_id (str): ID de la criptomoneda en CoinCap.
        symbol (str): Símbolo de la criptomoneda en LunarCrush.
        use_custom_range (bool): Usa rango de fechas personalizado.
        start_ms (int): Timestamp de inicio en milisegundos.
        end_ms (int): Timestamp de fin en milisegundos.
        horizon_days (int): Días a predecir.
        test_size (float): Proporción para el conjunto de test.
    
    Returns:
        tuple: (df, test_preds, y_test_real, future_preds, rmse, mape, sentiment_factor, lunar_metrics)
               o None si falla.
    """
    # Cargar datos de CoinCap
    temp_df = load_coincap_data(coin_id, start_ms, end_ms)
    if temp_df is None or temp_df.empty:
        st.warning("No se pudieron descargar datos suficientes de CoinCap.")
        return None
    df = temp_df.copy()

    if "close_price" not in df.columns:
        st.warning("No se encontró 'close_price' en los datos.")
        return None

    # Obtener sentimientos de X
    crypto_sentiment = get_crypto_sentiment(crypto_hashtags[st.session_state["crypto_name"]])
    market_sentiment = get_market_crypto_sentiment()
    sentiment_factor = (crypto_sentiment + market_sentiment) / 200  # Normalizar 0-1

    # Obtener datos de LunarCrush
    api_key = st.secrets["lunarcrush_api_key"]
    lunar_metrics = load_lunarcrush_data(symbol, api_key)
    if lunar_metrics:
        lunar_sentiment = (lunar_metrics["average_sentiment"] - 1) / 4  # Normalizar 1-5 a 0-1
        social_volume_factor = min(lunar_metrics["social_volume"] / 10000, 1.0)  # Cap a 10000
        social_dominance = min(lunar_metrics["social_dominance"] / 100, 1.0)  # Normalizar 0-100 a 0-1
        combined_sentiment = (sentiment_factor + lunar_sentiment) / 2  # Promedio X y LunarCrush
    else:
        lunar_sentiment, social_volume_factor, social_dominance = 0.5, 0.5, 0.5
        combined_sentiment = sentiment_factor
        lunar_metrics = {"social_volume": 0, "social_score": 0, "social_dominance": 0, "average_sentiment": 3, "social_contributors": 0}

    st.write(f"Sentimiento de la criptomoneda (X): {crypto_sentiment:.2f}")
    st.write(f"Sentimiento del mercado crypto (X): {market_sentiment:.2f}")
    st.write(f"Sentimiento LunarCrush: {lunar_metrics['average_sentiment']:.2f}/5")
    st.write(f"Factor de sentimiento combinado: {combined_sentiment:.2f}")

    # Hiperparámetros dinámicos
    window_size, epochs, batch_size, learning_rate = get_dynamic_params(df, horizon_days)
    st.info(f"Hiperparámetros ajustados: window_size={window_size}, epochs={epochs}, "
            f"batch_size={batch_size}, learning_rate={learning_rate}")

    # Preparar datos con múltiples características
    data_for_model = df[["close_price"]].values
    scaler_target = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler_target.fit_transform(data_for_model)
    features = np.concatenate([
        scaled_data,
        np.full((len(scaled_data), 1), combined_sentiment),
        np.full((len(scaled_data), 1), social_volume_factor),
        np.full((len(scaled_data), 1), social_dominance)
    ], axis=1)

    # Dividir en entrenamiento y test
    split_index = int(len(features) * (1 - test_size))
    if split_index <= window_size:
        st.warning("Datos insuficientes para entrenar. Reajusta parámetros.")
        return None

    train_data = features[:split_index]
    test_data = features[split_index:]
    X_train, y_train = create_sequences(train_data, window_size=window_size)
    if X_train is None:
        return None
    X_test, y_test = create_sequences(test_data, window_size=window_size)
    if X_test is None:
        return None

    # Dividir en entrenamiento y validación
    val_split = int(len(X_train) * 0.9)
    X_val, y_val = X_train[val_split:], y_train[val_split:]
    X_train, y_train = X_train[:val_split], y_train[:val_split]

    input_shape = (X_train.shape[1], X_train.shape[2])  # (window_size, 4)

    # Entrenar el modelo
    lstm_model = train_model(X_train, y_train, X_val, y_val, input_shape, epochs, batch_size, learning_rate)

    # Predicciones en test
    test_preds_scaled = lstm_model.predict(X_test)
    test_preds = scaler_target.inverse_transform(test_preds_scaled)
    y_test_real = scaler_target.inverse_transform(y_test.reshape(-1, 1))

    # Calcular métricas de error
    valid_mask = ~np.isnan(test_preds) & ~np.isnan(y_test_real)
    if np.sum(valid_mask) == 0:
        rmse, mape = np.nan, np.nan
    else:
        rmse = np.sqrt(np.mean((y_test_real[valid_mask] - test_preds[valid_mask]) ** 2))
        mape = robust_mape(y_test_real[valid_mask], test_preds[valid_mask])

    # Predicciones futuras
    last_window = features[-window_size:]
    future_preds_scaled = []
    current_input = last_window.reshape(1, window_size, -1)
    for _ in range(horizon_days):
        future_pred = lstm_model.predict(current_input)[0][0]
        future_preds_scaled.append(future_pred)
        new_feature = np.array([[future_pred, combined_sentiment, social_volume_factor, social_dominance]])
        current_input = np.append(current_input[:, 1:, :], new_feature.reshape(1, 1, -1), axis=1)
    future_preds = scaler_target.inverse_transform(np.array(future_preds_scaled).reshape(-1, 1)).flatten()

    return df, test_preds, y_test_real, future_preds, rmse, mape, combined_sentiment, lunar_metrics

##############################################
# Análisis de sentimiento usando X API (sin cambios)
##############################################
def setup_x_api():
    """Configura la API de X con el bearer_token desde st.secrets."""
    try:
        secrets = st.secrets
        bearer_token = secrets.get("bearer_token", "")
        if not bearer_token or bearer_token.strip() == "":
            raise ValueError("Bearer Token no configurado o vacío en los Secrets de Streamlit")
        client = tweepy.Client(
            bearer_token=bearer_token,
            consumer_key="",
            consumer_secret="",
            access_token="",
            access_token_secret=""
        )
        client.get_me()
        return client
    except tweepy.TweepyException as e:
        st.error(f"Error de autenticación en la API de X: {e} (Código {e.response.status_code if e.response else 'N/A'})")
        return None
    except ValueError as e:
        st.error(f"Error de configuración en los Secrets: {e}")
        return None
    except Exception as e:
        st.error(f"Error inesperado configurando la API de X: {e}")
        return None

def is_fake_news(tweet_text):
    """Detecta posibles fake news con una heurística simple."""
    fake_keywords = ["fake", "scam", "hoax", "misinformation", "rumor", "false"]
    return any(keyword.lower() in tweet_text.lower() for keyword in fake_keywords)

def get_crypto_sentiment(hashtag, max_tweets=10, retries=3, delay=5):
    """Obtiene el sentimiento promedio de tweets para una criptomoneda."""
    client = setup_x_api()
    if client is None:
        return 50.0
    analyzer = SentimentIntensityAnalyzer()
    tweets = []
    for attempt in range(retries):
        try:
            query = f"{hashtag} -is:retweet lang:en"
            response = client.search_recent_tweets(
                query=query,
                max_results=max_tweets,
                tweet_fields=["public_metrics", "created_at", "text", "context_annotations"],
                expansions=["author_id"]
            )
            for tweet in response.data or []:
                if tweet:
                    metrics = tweet.public_metrics
                    interactions = metrics.get("like_count", 0) + metrics.get("retweet_count", 0) + metrics.get("reply_count", 0)
                    if interactions > 5 and not is_fake_news(tweet.text):
                        if any(acc.lower() in tweet.text.lower() for acc in trusted_accounts) or \
                           any("News" in str(ann) or "Verified" in str(ann) for ann in tweet.context_annotations or []):
                            tweets.append(tweet.text)
            if tweets:
                break
            elif attempt < retries - 1:
                time.sleep(delay)
        except tweepy.TweepyException as e:
            if e.response and e.response.status_code == 500 and attempt < retries - 1:
                st.warning(f"Error 500 en API de X para {hashtag}, reintentando en {delay}s...")
                time.sleep(delay)
                continue
            st.error(f"Error de API de X para {hashtag}: {e}")
            return 50.0
    if not tweets:
        return 50.0
    scores = [analyzer.polarity_scores(tweet)["compound"] for tweet in tweets]
    avg_sentiment = np.mean(scores) * 50 + 50
    return max(0, min(100, avg_sentiment))

def get_market_crypto_sentiment(max_tweets=30, retries=3, delay=5):
    """Obtiene el sentimiento promedio del mercado crypto."""
    client = setup_x_api()
    if client is None:
        return 50.0
    analyzer = SentimentIntensityAnalyzer()
    market_hashtags = ["#Crypto", "#Cryptocurrency", "#Blockchain"]
    all_tweets = []
    for attempt in range(retries):
        try:
            for hashtag in market_hashtags:
                query = f"{hashtag} -is:retweet lang:en"
                response = client.search_recent_tweets(
                    query=query,
                    max_results=min(max_tweets // len(market_hashtags), 10),
                    tweet_fields=["public_metrics", "created_at", "text", "context_annotations"],
                    expansions=["author_id"]
                )
                for tweet in response.data or []:
                    if tweet:
                        metrics = tweet.public_metrics
                        interactions = metrics.get("like_count", 0) + metrics.get("retweet_count", 0) + metrics.get("reply_count", 0)
                        if interactions > 5 and not is_fake_news(tweet.text):
                            if any(acc.lower() in tweet.text.lower() for acc in trusted_accounts) or \
                               any("News" in str(ann) or "Verified" in str(ann) for ann in tweet.context_annotations or []):
                                all_tweets.append(tweet.text)
            if all_tweets:
                break
            elif attempt < retries - 1:
                time.sleep(delay)
        except tweepy.TweepyException as e:
            if e.response and e.response.status_code == 500 and attempt < retries - 1:
                st.warning(f"Error 500 en API de X para mercado crypto, reintentando en {delay}s...")
                time.sleep(delay)
                continue
            st.error(f"Error de API de X para mercado crypto: {e}")
            return 50.0
    if not all_tweets:
        return 50.0
    scores = [analyzer.polarity_scores(tweet)["compound"] for tweet in all_tweets]
    avg_sentiment = np.mean(scores) * 50 + 50
    return max(0, min(100, avg_sentiment))

##############################################
# Función principal de la app
##############################################
def main_app():
    """
    Interfaz principal de la aplicación en Streamlit para predicción de precios con análisis mejorado.
    
    Notes:
        Integra CoinCap, X API y LunarCrush sin exponer datos sensibles.
    """
    st.set_page_config(page_title="Crypto Price Predictions with Sentiment 🔮", layout="wide")
    st.title("Crypto Price Predictions with Sentiment 🔮")
    st.markdown("**Fuentes de Datos:** CoinCap, X API, LunarCrush")

    st.session_state["crypto_name"] = st.sidebar.selectbox(
        "Selecciona una criptomoneda:",
        list(coincap_ids.keys()),
        help="Elige la criptomoneda para la predicción y análisis."
    )
    coin_id = coincap_ids[st.session_state["crypto_name"]]
    symbol = lunarcrush_symbols[st.session_state["crypto_name"]]

    st.sidebar.subheader("Rango de Fechas")
    use_custom_range = st.sidebar.checkbox(
        "Habilitar rango de fechas",
        value=True,
        help="Si se desactiva, se usará todo el histórico disponible."
    )
    default_start = datetime(2021, 1, 1)
    default_end = datetime.now()
    if use_custom_range:
        start_date = st.sidebar.date_input("Fecha de inicio", default_start)
        end_date = st.sidebar.date_input("Fecha de fin", default_end)
        start_ms = int(datetime.combine(start_date, datetime.min.time()).timestamp() * 1000)
        end_ms = int(datetime.combine(end_date, datetime.min.time()).timestamp() * 1000)
    else:
        start_ms = None
        end_ms = None

    st.sidebar.subheader("Parámetros de Predicción")
    horizon = st.sidebar.slider("Días a predecir:", 1, 60, 30,
                                help="Número de días a futuro a predecir.")
    st.sidebar.markdown("**Nota:** Los hiperparámetros se ajustan automáticamente.")

    # Cargar y mostrar datos históricos de CoinCap
    df_prices = load_coincap_data(coin_id, start_ms, end_ms)
    if df_prices is not None and len(df_prices) > 0:
        df_chart = df_prices.copy()
        df_chart["ds_str"] = df_chart["ds"].dt.strftime("%d/%m/%Y")
        fig_hist = px.line(
            df_chart, x="ds_str", y="close_price",
            title=f"Histórico de {st.session_state['crypto_name']}",
            labels={"ds_str": "Fecha", "close_price": "Precio en USD"}
        )
        fig_hist.update_yaxes(tickformat=",.2f")
        fig_hist.update_layout(xaxis=dict(type="category", tickangle=45, nticks=10))
        st.plotly_chart(fig_hist, use_container_width=True)
        
        if st.sidebar.checkbox("Ver estadísticas descriptivas", value=False):
            st.subheader("Estadísticas Descriptivas")
            st.write(df_prices["close_price"].describe().rename({
                "count": "Cuenta", "mean": "Media", "std": "Desv. Estándar", "min": "Mínimo",
                "25%": "Percentil 25", "50%": "Mediana", "75%": "Percentil 75", "max": "Máximo"
            }))
    else:
        st.info("No se encontraron datos históricos válidos. Reajusta el rango de fechas.")

    tabs = st.tabs(["🤖 Entrenamiento y Test", f"🔮 Predicción de Precios - {st.session_state['crypto_name']}", "💬 Sentimiento"])

    with tabs[0]:
        st.header("Entrenamiento del Modelo y Evaluación en Test")
        if st.button("Entrenar Modelo y Predecir", key="train_test"):
            with st.spinner("Entrenando el modelo, por favor espera..."):
                result = train_and_predict_with_sentiment(
                    coin_id=coin_id,
                    symbol=symbol,
                    use_custom_range=use_custom_range,
                    start_ms=start_ms,
                    end_ms=end_ms,
                    horizon_days=horizon,
                    test_size=0.2
                )
            if result is not None:
                df_model, test_preds, y_test_real, future_preds, rmse, mape, sentiment_factor, lunar_metrics = result
                st.success("Entrenamiento y predicción completados!")
                col1, col2 = st.columns(2)
                col1.metric("RMSE (Test)", f"{rmse:.2f}")
                col2.metric("MAPE (Test)", f"{mape:.2f}%")
                st.subheader("Comparación en el Set de Test")
                test_dates = df_model["ds"].iloc[-len(y_test_real):]
                fig_test = go.Figure()
                fig_test.add_trace(go.Scatter(
                    x=test_dates, y=y_test_real.flatten(), mode="lines", name="Precio Real (Test)"
                ))
                fig_test.add_trace(go.Scatter(
                    x=test_dates, y=test_preds.flatten(), mode="lines", name="Predicción (Test)"
                ))
                fig_test.update_layout(
                    title=f"Comparación en Test: {st.session_state['crypto_name']}",
                    xaxis_title="Fecha", yaxis_title="Precio en USD"
                )
                fig_test.update_yaxes(tickformat=",.2f")
                st.plotly_chart(fig_test, use_container_width=True)

    with tabs[1]:
        st.header(f"Predicción de Precios - {st.session_state['crypto_name']}")
        if 'result' in locals() and result is not None:
            df_model, test_preds, y_test_real, future_preds, rmse, mape, sentiment_factor, lunar_metrics = result
            last_date = df_model["ds"].iloc[-1]
            current_price = df_model["close_price"].iloc[-1]
            future_dates = pd.date_range(start=last_date, periods=horizon + 1, freq="D")
            pred_series = np.concatenate(([current_price], future_preds))
            fig_future = go.Figure()
            fig_future.add_trace(go.Scatter(
                x=future_dates, y=pred_series, mode="lines+markers", name="Predicción Futura"
            ))
            fig_future.update_layout(
                title=f"Predicción a Futuro ({horizon} días) - {st.session_state['crypto_name']} (Sentimiento: {sentiment_factor:.2f})",
                xaxis_title="Fecha", yaxis_title="Precio en USD"
            )
            fig_future.update_yaxes(tickformat=",.2f")
            st.plotly_chart(fig_future, use_container_width=True)
            st.subheader("Valores Numéricos de la Predicción Futura")
            future_df = pd.DataFrame({"Fecha": future_dates, "Predicción": pred_series})
            st.dataframe(future_df)
        else:
            st.info("Primero entrena el modelo en 'Entrenamiento y Test' para ver las predicciones.")

    with tabs[2]:
        st.header("Sentimiento y Métricas Sociales")
        st.markdown("Analizando datos de X y LunarCrush...")
        
        # Sentimiento de la criptomoneda (X)
        crypto_sentiment = get_crypto_sentiment(crypto_hashtags[st.session_state["crypto_name"]], max_tweets=10)
        st.subheader(f"Sentimiento de {st.session_state['crypto_name']} (X)")
        st.metric("Sentimiento Promedio", f"{crypto_sentiment:.2f} (0-100, Bearish-Bullish)")

        # Tweets relevantes
        client = setup_x_api()
        if client:
            hashtag = crypto_hashtags[st.session_state["crypto_name"]]
            try:
                response = client.search_recent_tweets(
                    query=f"{hashtag} -is:retweet lang:en",
                    max_results=10,
                    tweet_fields=["public_metrics", "created_at", "text", "context_annotations"],
                    expansions=["author_id"]
                )
                relevant_tweets = []
                for tweet in response.data or []:
                    if tweet:
                        metrics = tweet.public_metrics
                        interactions = metrics.get("like_count", 0) + metrics.get("retweet_count", 0) + metrics.get("reply_count", 0)
                        if interactions > 5 and not is_fake_news(tweet.text):
                            if any(acc.lower() in tweet.text.lower() for acc in trusted_accounts) or \
                               any("News" in str(ann) or "Verified" in str(ann) for ann in tweet.context_annotations or []):
                                relevant_tweets.append({
                                    "texto": tweet.text,
                                    "interacciones": interactions,
                                    "fecha": tweet.created_at
                                })
                relevant_tweets = sorted(relevant_tweets, key=lambda x: x["interacciones"], reverse=True)[:5]
                st.write("Tweets más relevantes (por interacciones, sin fake news):")
                for tweet in relevant_tweets:
                    st.write(f"- **{tweet['texto']}** (Interacciones: {tweet['interacciones']}, Fecha: {tweet['fecha']})")
            except tweepy.TweepyException as e:
                st.error(f"Error de API de X para tweets relevantes: {e}")
            except Exception as e:
                st.error(f"Error inesperado al obtener tweets: {e}")
        else:
            st.error("No se pudo configurar la API de X. Verifica las claves en Secrets.")

        # Sentimiento del mercado crypto (X)
        market_sentiment = get_market_crypto_sentiment(max_tweets=30)
        st.subheader("Sentimiento del Mercado Crypto (X)")
        st.metric("Sentimiento Promedio", f"{market_sentiment:.2f} (0-100, Bearish-Bullish)")

        # Métricas de LunarCrush
        lunar_metrics = load_lunarcrush_data(symbol, st.secrets["lunarcrush_api_key"])
        if lunar_metrics:
            st.subheader(f"Métricas Sociales de {st.session_state['crypto_name']} (LunarCrush)")
            st.metric("Sentimiento Promedio", f"{lunar_metrics['average_sentiment']:.2f} (1-5)")
            st.metric("Volumen Social", f"{lunar_metrics['social_volume']}")
            st.metric("Puntuación Social", f"{lunar_metrics['social_score']}")
            st.metric("Dominancia Social", f"{lunar_metrics['social_dominance']:.2f}%")
            st.metric("Contribuidores Sociales", f"{lunar_metrics['social_contributors']}")
        else:
            st.info("No se pudieron obtener métricas de LunarCrush.")

if __name__ == "__main__":
    main_app()