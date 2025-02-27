#########################
# main.py
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

# Diccionarios con IDs de criptomonedas para CoinCap y hashtags para X
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
trusted_accounts = ["@CoinMarketCap", "@CoinDesk", "@Binance", "@Krakenfx"]  # Cuentas confiables en el mercado crypto

##############################################
# Descarga de datos desde CoinCap (intervalo diario)
##############################################
@st.cache_data  # Cache para optimizar rendimiento en Streamlit
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
    
    Notes:
        Maneja casos en los que 'volumeUsd' puede no existir o ser None, asegurando consistencia con pandas.
    """
    url = f"https://api.coincap.io/v2/assets/{coin_id}/history?interval=d1"
    if start_ms is not None and end_ms is not None:
        url += f"&start={start_ms}&end={end_ms}"
    headers = {"User-Agent": "Mozilla/5.0"}  # Usar User-Agent para cumplir con políticas de CoinCap
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
                
                # Procesar 'time' y 'priceUsd' de manera segura
                df["ds"] = pd.to_datetime(df["time"], unit="ms", errors="coerce")
                df["close_price"] = pd.to_numeric(df["priceUsd"], errors="coerce")
                
                # Manejar 'volumeUsd' de manera segura, asegurando un Series pandas
                if "volumeUsd" in df.columns:
                    df["volume"] = pd.to_numeric(df["volumeUsd"], errors="coerce")
                else:
                    df["volume"] = pd.Series(0.0, index=df.index)  # Crear un Series con ceros
                df["volume"] = df["volume"].fillna(0.0)  # Rellenar NaN con 0.0 de manera segura
                
                # Filtrar y limpiar datos
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
# Creación de secuencias para LSTM
##############################################
def create_sequences(data, window_size=30):
    """
    Crea secuencias temporales de tamaño 'window_size' a partir de datos de precios.
    
    Args:
        data (np.ndarray): Array con los datos de precios (una sola característica por defecto).
        window_size (int): Tamaño de la ventana de tiempo para las secuencias.
    
    Returns:
        tuple: (X, y) con secuencias de entrada y valores objetivo, o (None, None) si hay insuficientes datos.
    """
    if len(data) <= window_size:
        st.warning(f"No hay datos suficientes para una ventana de {window_size} días.")
        return None, None
    X, y = [], []
    for i in range(window_size, len(data)):
        X.append(data[i - window_size : i])
        y.append(data[i, 0])
    return np.array(X), np.array(y)

##############################################
# Modelo LSTM: Conv1D + Bidirectional LSTM ajustado para precio + sentimiento
##############################################
def build_lstm_model(input_shape, learning_rate=0.001):
    """
    Construye un modelo secuencial que combina Conv1D y Bidirectional LSTM, ajustado para manejar
    precio y sentimiento como características (2 dimensiones en axis -1).
    
    Args:
        input_shape (tuple): Forma de entrada (window_size, 2) para precio + sentimiento.
        learning_rate (float): Tasa de aprendizaje para el optimizador Adam.
    
    Returns:
        Sequential: Modelo Keras compilado.
    
    Notes:
        Asegura que el modelo procese 2 características (precio + sentimiento) en la última dimensión.
    """
    model = Sequential()
    # Ajustar input_shape para manejar 2 características (precio + sentimiento)
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
    Entrena el modelo LSTM de forma aislada, limpiando el estado global para evitar conflictos en Streamlit.
    
    Args:
        X_train (np.ndarray): Datos de entrenamiento ajustados (precio + sentimiento, shape: (None, window_size, 2)).
        y_train (np.ndarray): Valores objetivo de entrenamiento (solo precio, shape: (None, 1)).
        X_val (np.ndarray): Datos de validación ajustados.
        y_val (np.ndarray): Valores objetivo de validación.
        input_shape (tuple): Forma de entrada esperada por el modelo (window_size, 2).
        epochs (int): Número de épocas de entrenamiento.
        batch_size (int): Tamaño del batch.
        learning_rate (float): Tasa de aprendizaje.
    
    Returns:
        model: Modelo LSTM entrenado.
    """
    # Limpiar la sesión de Keras para evitar conflictos entre ejecuciones en Streamlit
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
    Calcula hiperparámetros dinámicos basados en características del dataset, como volatilidad y tamaño.
    
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
# Entrenamiento y predicción con LSTM ajustado por sentimiento
##############################################
def train_and_predict_with_sentiment(
    coin_id, use_custom_range, start_ms, end_ms, horizon_days=30, test_size=0.2
):
    """
    Descarga datos de CoinCap, entrena un modelo LSTM ajustado por sentimiento y realiza predicciones.
    
    Args:
        coin_id (str): ID de la criptomoneda en CoinCap.
        use_custom_range (bool): Indica si se usa un rango de fechas personalizado.
        start_ms (int): Timestamp de inicio en milisegundos.
        end_ms (int): Timestamp de fin en milisegundos.
        horizon_days (int): Días a predecir en el futuro.
        test_size (float): Proporción de datos para el conjunto de test.
    
    Returns:
        tuple: (df, test_preds, y_test_real, future_preds, rmse, mape, sentiment_factor) o None si falla.
    
    Notes:
        Asegura que los datos de entrada al modelo tengan 2 características (precio + sentimiento).
    """
    # Cargar datos de precios desde CoinCap
    temp_df = load_coincap_data(coin_id, start_ms, end_ms)
    if temp_df is None or temp_df.empty:
        st.warning("No se pudieron descargar datos suficientes de CoinCap.")
        return None
    
    df = temp_df.copy()

    if "close_price" not in df.columns:
        st.warning("No se encontró 'close_price' en los datos.")
        return None

    # Obtener sentimientos para la criptomoneda y el mercado crypto
    crypto_sentiment = get_crypto_sentiment(crypto_hashtags[st.session_state["crypto_name"]])
    market_sentiment = get_market_crypto_sentiment()
    
    # Combinar sentiments en un factor único (0 a 1, donde 0 es muy bajista, 1 muy alcista)
    sentiment_factor = (crypto_sentiment + market_sentiment) / 200  # Normalizar de 0 a 1
    
    st.write(f"Sentimiento de la criptomoneda: {crypto_sentiment:.2f}")
    st.write(f"Sentimiento del mercado crypto: {market_sentiment:.2f}")
    st.write(f"Factor de sentimiento combinado: {sentiment_factor:.2f}")

    # Calcular hiperparámetros dinámicos
    window_size, epochs, batch_size, learning_rate = get_dynamic_params(df, horizon_days)
    st.info(f"Hiperparámetros ajustados: window_size={window_size}, epochs={epochs}, "
            f"batch_size={batch_size}, learning_rate={learning_rate}")

    # Preparar datos para el modelo
    data_for_model = df[["close_price"]].values

    # Escalar los datos al rango [0, 1]
    scaler_target = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler_target.fit_transform(data_for_model)

    # Dividir en conjuntos de entrenamiento y test
    split_index = int(len(scaled_data) * (1 - test_size))
    if split_index <= window_size:
        st.warning("Datos insuficientes para entrenar. Reajusta parámetros.")
        return None

    train_data = scaled_data[:split_index]
    test_data = scaled_data[split_index:]
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

    # Ajustar los datos de entrenamiento con el factor de sentimiento (precio + sentimiento, 2 características)
    # Verificar dimensiones para depuración
    st.write(f"Dimensión de X_train antes de ajuste: {X_train.shape}")  # Debería ser (None, window_size, 1)
    if X_train.shape[-1] != 1:
        st.error(f"Dimensión inesperada de X_train: {X_train.shape}. Se esperaba (None, {window_size}, 1)")
        return None
    X_train_adjusted = np.concatenate([X_train, np.full((X_train.shape[0], X_train.shape[1], 1), sentiment_factor)], axis=-1)
    X_test_adjusted = np.concatenate([X_test, np.full((X_test.shape[0], X_test.shape[1], 1), sentiment_factor)], axis=-1)
    input_shape = (X_train_adjusted.shape[1], X_train_adjusted.shape[2])  # (window_size, 2)

    # Verificar dimensiones después del ajuste para depuración
    st.write(f"Dimensión de X_train_adjusted después de ajuste: {X_train_adjusted.shape}")  # Debería ser (None, window_size, 2)
    if X_train_adjusted.shape[-1] != 2:
        st.error(f"Dimensión inesperada de X_train_adjusted: {X_train_adjusted.shape}. Se esperaba (None, {window_size}, 2)")
        return None

    # Entrenar el modelo ajustado para manejar precio + sentimiento
    lstm_model = train_model(X_train_adjusted, y_train, X_val, y_val, input_shape, epochs, batch_size, learning_rate)

    # Predicciones ajustadas en el conjunto de test
    test_preds_scaled = lstm_model.predict(X_test_adjusted)
    test_preds = scaler_target.inverse_transform(test_preds_scaled)
    y_test_real = scaler_target.inverse_transform(y_test.reshape(-1, 1))

    # Calcular métricas de error
    valid_mask = ~np.isnan(test_preds) & ~np.isnan(y_test_real)
    if np.sum(valid_mask) == 0:
        rmse, mape = np.nan, np.nan
    else:
        rmse = np.sqrt(np.mean((y_test_real[valid_mask] - test_preds[valid_mask]) ** 2))
        mape = robust_mape(y_test_real[valid_mask], test_preds[valid_mask])

    # Predicciones futuras ajustadas por sentimiento
    last_window = scaled_data[-window_size:]
    future_preds_scaled = []
    current_input = np.concatenate([last_window, np.full((1, window_size, 1), sentiment_factor)], axis=-1)
    for _ in range(horizon_days):
        future_pred = lstm_model.predict(current_input)[0][0]
        future_preds_scaled.append(future_pred)
        new_feature = np.copy(current_input[:, -1:, :])
        new_feature[0, 0, 0] = future_pred
        new_feature[0, 0, 1] = sentiment_factor  # Mantener el factor de sentimiento
        current_input = np.append(current_input[:, 1:, :], new_feature, axis=1)
    future_preds = scaler_target.inverse_transform(np.array(future_preds_scaled).reshape(-1, 1)).flatten()

    return df, test_preds, y_test_real, future_preds, rmse, mape, sentiment_factor

##############################################
# Análisis de sentimiento usando X API (plan gratuito)
##############################################
def setup_x_api():
    """
    Configura la API de X usando los Secrets de Streamlit, manejando errores de autenticación.
    Usa solo bearer_token para el plan gratuito, ignorando otras claves si no son necesarias.
    
    Returns:
        tweepy.Client: Cliente de Tweepy configurado, o None si falla.
    
    Notes:
        No imprime Secrets en la interfaz para evitar exposiciones de seguridad.
        Usa el Bearer Token generado en X Developer Portal para el plan Free, accediendo directamente a st.secrets["bearer_token"].
        Asegúrate de que 'bearer_token' esté configurado correctamente en Streamlit Secrets sin [x_api].
    """
    try:
        secrets = st.secrets
        bearer_token = secrets.get("bearer_token", "")
        if not bearer_token or bearer_token.strip() == "":
            raise ValueError("Bearer Token no configurado, vacío, o contiene solo espacios en los Secrets de Streamlit")
        
        # Usar solo bearer_token para el plan gratuito de X API v2
        client = tweepy.Client(bearer_token=bearer_token)
        # Verificar si el cliente funciona con una solicitud mínima
        client.get_me()  # Prueba rápida para validar autenticación
        return client
    except tweepy.TweepyException as e:
        st.error(f"Error de autenticación en la API de X: {e} (Código {e.response.status_code if e.response else 'N/A'})")
        return None
    except ValueError as e:
        st.error(f"Error de configuración en los Secrets de Streamlit: {e}")
        return None
    except Exception as e:
        st.error(f"Error inesperado configurando la API de X: {e}")
        return None

def is_fake_news(tweet_text):
    """
    Heurística simple para detectar posibles fake news en el texto de un tweet.
    
    Args:
        tweet_text (str): Texto del tweet a evaluar.
    
    Returns:
        bool: True si el texto podría ser fake news, False de lo contrario.
    """
    fake_keywords = ["fake", "scam", "hoax", "misinformation", "rumor", "false"]
    return any(keyword.lower() in tweet_text.lower() for keyword in fake_keywords)

def get_crypto_sentiment(hashtag, max_tweets=10, retries=3, delay=5):
    """
    Obtiene el sentimiento promedio de tweets relevantes para una criptomoneda usando X API.
    Filtra por interacciones y excluye posibles fake news, manejando errores 500 con reintentos.
    
    Args:
        hashtag (str): Hashtag de la criptomoneda (ej. "#Bitcoin").
        max_tweets (int): Máximo de tweets a recuperar (límite 10 por request en plan gratuito).
        retries (int): Número de reintentos para errores 500.
        delay (int): Segundos de espera entre reintentos.
    
    Returns:
        float: Sentimiento promedio normalizado a 0-100 (bearish a bullish), o 50.0 si falla.
    """
    client = setup_x_api()
    if client is None:
        return 50.0  # Valor neutral en caso de fallo de autenticación
    analyzer = SentimentIntensityAnalyzer()
    tweets = []
    
    for attempt in range(retries):
        try:
            # Buscar tweets recientes con el hashtag, excluyendo retweets
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
                    # Filtrar por interacciones y excluir fake news
                    if interactions > 5 and not is_fake_news(tweet.text):
                        # Priorizar tweets de cuentas confiables o con contexto "News" o "Verified"
                        if any(acc.lower() in tweet.text.lower() for acc in trusted_accounts) or \
                           any("News" in str(ann) or "Verified" in str(ann) for ann in tweet.context_annotations or []):
                            tweets.append(tweet.text)
            if tweets:
                break  # Salir si se obtuvieron tweets
            elif attempt < retries - 1:
                time.sleep(delay)  # Esperar antes de reintentar
        except tweepy.TweepyException as e:
            if e.response and e.response.status_code == 500:
                if attempt < retries - 1:
                    st.warning(f"Error 500 en API de X para {hashtag}, reintentando en {delay}s...")
                    time.sleep(delay)
                    continue
            st.error(f"Error de API de X para {hashtag}: {e} (Código {e.response.status_code if e.response else 'N/A'})")
            return 50.0
        except Exception as e:
            st.error(f"Error inesperado al obtener tweets para {hashtag}: {e}")
            return 50.0
    
    if not tweets:
        return 50.0  # Valor neutral si no hay tweets válidos
    scores = [analyzer.polarity_scores(tweet)["compound"] for tweet in tweets]
    avg_sentiment = np.mean(scores) * 50 + 50  # Normalizar a 0-100 (bearish a bullish)
    return max(0, min(100, avg_sentiment))  # Asegurar que esté entre 0 y 100

def get_market_crypto_sentiment(max_tweets=30, retries=3, delay=5):
    """
    Obtiene el sentimiento promedio del mercado crypto usando hashtags generales.
    Filtra por interacciones y excluye posibles fake news, manejando errores 500 con reintentos.
    
    Args:
        max_tweets (int): Máximo de tweets a recuperar (distribuido entre hashtags).
        retries (int): Número de reintentos para errores 500.
        delay (int): Segundos de espera entre reintentos.
    
    Returns:
        float: Sentimiento promedio normalizado a 0-100, o 50.0 si falla.
    """
    client = setup_x_api()
    if client is None:
        return 50.0  # Valor neutral en caso de fallo de autenticación
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
                break  # Salir si se obtuvieron tweets
            elif attempt < retries - 1:
                time.sleep(delay)  # Esperar antes de reintentar
        except tweepy.TweepyException as e:
            if e.response and e.response.status_code == 500:
                if attempt < retries - 1:
                    st.warning(f"Error 500 en API de X para mercado crypto, reintentando en {delay}s...")
                    time.sleep(delay)
                    continue
            st.error(f"Error de API de X para mercado crypto: {e} (Código {e.response.status_code if e.response else 'N/A'})")
            return 50.0
        except Exception as e:
            st.error(f"Error inesperado al obtener sentimiento del mercado crypto: {e}")
            return 50.0
    
    if not all_tweets:
        return 50.0  # Valor neutral si no hay tweets válidos
    scores = [analyzer.polarity_scores(tweet)["compound"] for tweet in all_tweets]
    avg_sentiment = np.mean(scores) * 50 + 50  # Normalizar a 0-100
    return max(0, min(100, avg_sentiment))

##############################################
# Función principal de la app
##############################################
def main_app():
    """
    Interfaz principal de la aplicación en Streamlit para predicción de precios de criptomonedas
    con análisis de sentimiento usando la API de X.
    
    Notes:
        No imprime Secrets ni datos sensibles en la interfaz para evitar exposiciones de seguridad.
    """
    st.set_page_config(page_title="Crypto Price Predictions with Sentiment 🔮", layout="wide")
    st.title("Crypto Price Predictions with Sentiment 🔮")
    st.markdown("**Fuente de Datos:** CoinCap y X API")

    st.session_state["crypto_name"] = st.sidebar.selectbox(
        "Selecciona una criptomoneda:",
        list(coincap_ids.keys()),
        help="Elige la criptomoneda para la predicción y análisis de sentimiento."
    )
    coin_id = coincap_ids[st.session_state["crypto_name"]]

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
    st.sidebar.markdown("**Nota:** Los hiperparámetros (ventana, épocas, etc.) se ajustan automáticamente según los datos.")

    # Cargar datos históricos de CoinCap
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

    tabs = st.tabs(["🤖 Entrenamiento y Test", f"🔮 Predicción de Precios - {st.session_state['crypto_name']}", "💬 Sentimiento en X"])

    with tabs[0]:
        st.header("Entrenamiento del Modelo y Evaluación en Test")
        if st.button("Entrenar Modelo y Predecir", key="train_test"):
            with st.spinner("Entrenando el modelo, por favor espera..."):
                result = train_and_predict_with_sentiment(
                    coin_id=coin_id,
                    use_custom_range=use_custom_range,
                    start_ms=start_ms,
                    end_ms=end_ms,
                    horizon_days=horizon,
                    test_size=0.2
                )
            if result is not None:
                df_model, test_preds, y_test_real, future_preds, rmse, mape, sentiment_factor = result
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
            df_model, test_preds, y_test_real, future_preds, rmse, mape, sentiment_factor = result
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
            st.info("Primero entrena el modelo en la pestaña 'Entrenamiento y Test' para generar las predicciones futuras.")

    with tabs[2]:
        st.header("Sentimiento en X")
        st.markdown("Analizando tweets recientes sobre la criptomoneda y el mercado crypto...")
        
        # Sentimiento de la criptomoneda seleccionada
        crypto_sentiment = get_crypto_sentiment(crypto_hashtags[st.session_state["crypto_name"]], max_tweets=10)
        st.subheader(f"Sentimiento de {st.session_state['crypto_name']}")
        st.metric("Sentimiento Promedio", f"{crypto_sentiment:.2f} (0-100, Bearish-Bullish)")
        
        # Mostrar tweets más relevantes (ordenados por interacciones, sin fake news)
        client = setup_x_api()
        if client:
            hashtag = crypto_hashtags[st.session_state["crypto_name"]]
            try:
                response = client.search_recent_tweets(
                    query=f"{hashtag} -is:retweet lang:en",
                    max_results=10,  # Límite del plan gratuito
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
                st.error(f"Error de API de X para tweets relevantes: {e} (Código {e.response.status_code if e.response else 'N/A'})")
            except Exception as e:
                st.error(f"Error inesperado al obtener tweets relevantes: {e}")
        else:
            st.error("No se pudo configurar la API de X. Verifica las claves en los Secrets de Streamlit.")

        # Sentimiento del mercado crypto
        market_sentiment = get_market_crypto_sentiment(max_tweets=30)
        st.subheader("Sentimiento del Mercado Crypto")
        st.metric("Sentimiento Promedio", f"{market_sentiment:.2f} (0-100, Bearish-Bullish)")

if __name__ == "__main__":
    main_app()