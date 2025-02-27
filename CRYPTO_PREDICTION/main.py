import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.regularizers import l2
import requests
import certifi
import os
from sklearn.metrics import mean_squared_error
from textblob import TextBlob
import socket  # Para manejar errores de DNS
from urllib3.util.retry import Retry  # Importación para Retry
from requests.adapters import HTTPAdapter  # Importación para HTTPAdapter

# Configuración inicial de certificados SSL y solicitudes
os.environ["REQUESTS_CA_BUNDLE"] = certifi.where()
session = requests.Session()
retry = Retry(total=5, backoff_factor=1, status_forcelist=[429, 500, 502, 503, 504])
adapter = HTTPAdapter(max_retries=retry)
session.mount("https://", adapter)

# Diccionarios de criptomonedas y características
coincap_ids = {
    "Bitcoin (BTC)": "bitcoin", "Ethereum (ETH)": "ethereum", "Ripple (XRP)": "xrp",
    "Binance Coin (BNB)": "binance-coin", "Cardano (ADA)": "cardano", "Solana (SOL)": "solana",
    "Dogecoin (DOGE)": "dogecoin", "Polkadot (DOT)": "polkadot", "Polygon (MATIC)": "polygon",
    "Litecoin (LTC)": "litecoin", "TRON (TRX)": "tron", "Stellar (XLM)": "stellar"
}
coinid_to_symbol = {v: k.split(" (")[1][:-1] for k, v in coincap_ids.items()}
coinid_to_coingecko = {v: v if v != "xrp" else "ripple" for v in coincap_ids.values()}
crypto_characteristics = {
    "bitcoin": {"volatility": 0.03}, "ethereum": {"volatility": 0.05}, "xrp": {"volatility": 0.08},
    "binance-coin": {"volatility": 0.06}, "cardano": {"volatility": 0.07}, "solana": {"volatility": 0.09},
    "dogecoin": {"volatility": 0.12}, "polkadot": {"volatility": 0.07}, "polygon": {"volatility": 0.06},
    "litecoin": {"volatility": 0.04}, "tron": {"volatility": 0.06}, "stellar": {"volatility": 0.05}
}

# Funciones de apoyo
def robust_mape(y_true, y_pred, eps=1e-9):
    """Calcula el MAPE de manera robusta evitando división por cero."""
    return np.mean(np.abs((y_true - y_pred) / np.maximum(np.abs(y_true), eps))) * 100

# Carga de datos corregida con soporte para rango personalizado
@st.cache_data
def load_coincap_data(coin_id, start_ms=None, end_ms=None):
    """Carga datos históricos de CoinCap para una criptomoneda específica con rango personalizado."""
    if start_ms is None or end_ms is None:
        end_date = datetime.now()
        start_date = end_date - timedelta(days=730)  # Rango por defecto de 2 años
        start_ms = int(start_date.timestamp() * 1000)
        end_ms = int(end_date.timestamp() * 1000)
    
    url = f"https://api.coincap.io/v2/assets/{coin_id}/history?interval=d1&start={start_ms}&end={end_ms}"
    try:
        resp = session.get(url, headers={"User-Agent": "Mozilla/5.0"}, verify=certifi.where(), timeout=10)
        if resp.status_code != 200:
            st.warning(f"CoinCap: Error {resp.status_code}")
            return None
        df = pd.DataFrame(resp.json().get("data", []))
        if df.empty or "time" not in df.columns or "priceUsd" not in df.columns:
            st.warning("CoinCap: Datos inválidos o vacíos")
            return None
        df["ds"] = pd.to_datetime(df["time"], unit="ms", errors="coerce")
        df["close_price"] = pd.to_numeric(df["priceUsd"], errors="coerce")
        # Manejo robusto de "volumeUsd" como serie de pandas
        if "volumeUsd" in df.columns and not df["volumeUsd"].empty:
            df["volume"] = pd.to_numeric(df["volumeUsd"], errors="coerce").fillna(0.0)
        else:
            df["volume"] = pd.Series(0.0, index=df.index)
        return df[["ds", "close_price", "volume"]].dropna().sort_values("ds").reset_index(drop=True)
    except Exception as e:
        st.error(f"Error al cargar datos: {e}")
        return None

# Secuencias y modelo LSTM mejorado
def create_sequences(data, window_size):
    """Crea secuencias para el modelo LSTM con sentimiento integrado."""
    if len(data) <= window_size:
        return None, None
    X, y = [], []
    for i in range(window_size, len(data)):
        X.append(data[i - window_size:i])
        y.append(data[i, 0])
    return np.array(X), np.array(y)

def build_lstm_model(input_shape, learning_rate=0.001, l2_lambda=0.01):
    """Construye un modelo LSTM mejorado con regularización L2."""
    model = Sequential([
        LSTM(100, return_sequences=True, input_shape=input_shape, kernel_regularizer=l2(l2_lambda)),  # Más unidades y regularización
        Dropout(0.3),  # Aumentar Dropout para reducir sobreajuste
        LSTM(80, kernel_regularizer=l2(l2_lambda)),  # Capa adicional para capturar patrones
        Dropout(0.3),
        Dense(50, activation="relu", kernel_regularizer=l2(l2_lambda)),
        Dense(1)
    ])
    model.compile(optimizer=Adam(learning_rate), loss="mse")
    return model

def train_model(X_train, y_train, X_val, y_val, input_shape, epochs, batch_size):
    """Entrena el modelo LSTM mejorado con validación cruzada simple."""
    tf.keras.backend.clear_session()
    model = build_lstm_model(input_shape)
    callbacks = [
        EarlyStopping(patience=10, restore_best_weights=True),  # Más paciencia para convergencia
        ReduceLROnPlateau(patience=5, factor=0.5, min_lr=1e-6)
    ]
    history = model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=epochs, batch_size=batch_size, callbacks=callbacks, verbose=0)
    return model, history

def get_dynamic_params(df, horizon_days, coin_id):
    """Ajusta parámetros dinámicos mejorados según volatilidad y datos."""
    volatility = df["close_price"].pct_change().std()
    base_volatility = crypto_characteristics.get(coin_id, {"volatility": 0.05})["volatility"]
    
    # Ajustes más precisos para criptos volátiles (XRP) y estables (Bitcoin)
    if coin_id == "xrp":  # XRP, más volátil
        window_size = min(max(15, int(horizon_days * 1.0)), len(df) // 4)  # Ventana más corta para capturar volatilidad
        epochs = min(150, max(40, int(len(df) / 60) + int(volatility * 300)))  # Más épocas para datos complejos
        batch_size = 16  # Batch menor para mayor precisión
        learning_rate = 0.0002  # Learning rate más bajo para convergencia
    elif coin_id == "bitcoin":  # Bitcoin, más estable
        window_size = min(max(30, int(horizon_days * 1.5)), len(df) // 3)  # Ventana más larga para patrones estables
        epochs = min(100, max(30, int(len(df) / 80) + int(volatility * 200)))  # Menos épocas para estabilidad
        batch_size = 32  # Batch mayor para eficiencia
        learning_rate = 0.0005  # Learning rate estándar
    else:  # Otras criptos con volatilidad intermedia
        window_size = min(max(20, int(horizon_days * 1.2)), len(df) // 4)
        epochs = min(120, max(35, int(len(df) / 70) + int(volatility * 250)))
        batch_size = 24
        learning_rate = 0.0003

    return window_size, epochs, batch_size, learning_rate

# Sentimiento dinámico
@st.cache_data(ttl=3600)  # Actualiza cada hora
def get_fear_greed_index():
    """Obtiene el índice Fear & Greed."""
    try:
        return float(session.get("https://api.alternative.me/fng/?format=json", timeout=10).json()["data"][0]["value"])
    except Exception:
        st.warning("No se pudo obtener Fear & Greed Index. Usando valor por defecto.")
        return 50.0

@st.cache_data(ttl=3600)
def get_coingecko_community_activity(coin_id):
    """Obtiene actividad comunitaria de CoinGecko."""
    try:
        cg_id = coinid_to_coingecko.get(coin_id, coin_id)
        data = session.get(f"https://api.coingecko.com/api/v3/coins/{cg_id}?community_data=true", timeout=10).json()["community_data"]
        activity = max(data.get("twitter_followers", 0), data.get("reddit_average_posts_48h", 0) * 1000)
        return min(100, (activity / 20000000) * 100) if activity > 0 else 50.0
    except Exception:
        return 50.0  # Eliminado el mensaje de advertencia según tu solicitud

def get_crypto_sentiment_combined(coin_id, news_sentiment=None):
    """Calcula el sentimiento combinado dinámico con noticias específicas de cripto y pesos ajustados por volatilidad."""
    fg = get_fear_greed_index()
    cg = get_coingecko_community_activity(coin_id)
    volatility = crypto_characteristics.get(coin_id, {"volatility": 0.05})["volatility"]

    # Ajustar pesos: más peso a noticias y CoinGecko para volátiles, menos para estables
    if volatility > 0.07:  # Criptos muy volátiles (e.g., XRP, DOGE)
        fg_weight = 0.15  # Menos peso al mercado global
        cg_weight = 0.45  # Más peso a la actividad comunitaria
        news_weight = 0.40  # Más peso a las noticias cripto para capturar volatilidad
    else:  # Criptos más estables (e.g., BTC, ETH)
        fg_weight = 0.50  # Más peso al mercado global
        cg_weight = 0.30  # Menos peso a la actividad comunitaria
        news_weight = 0.20  # Menos peso a las noticias

    # Sentimiento de noticias (si no hay datos o falla, usar 50.0 como valor por defecto)
    news_sent = 50.0 if news_sentiment is None or pd.isna(news_sentiment) else float(news_sentiment)
    combined_sentiment = (fg * fg_weight + cg * cg_weight + news_sent * news_weight)
    return max(0, min(100, combined_sentiment))  # Asegurar rango 0-100

# Nueva función para análisis de noticias de criptomonedas (usando NewsData.io con API key desde Secrets, optimizada según #crypto-news)
@st.cache_data(ttl=86400)  # Cachear datos diarios para minimizar peticiones
def get_news_sentiment(coin_symbol, start_date=None, end_date=None):
    """Obtiene y analiza el sentimiento de noticias específicas de criptomonedas usando NewsData.io, optimizado para el sector crypto."""
    if start_date is None or end_date is None:
        end_date = datetime.now().date()
        start_date = end_date - timedelta(days=7)  # Reducido a 7 días por defecto para mayor eficiencia
    else:
        # Validar que el rango no exceda 7 días para optimizar y evitar errores 422
        if (end_date - start_date).days > 7:
            start_date = end_date - timedelta(days=7)
        # Asegurar que las fechas no sean futuras
        if start_date > datetime.now().date():
            start_date = datetime.now().date() - timedelta(days=7)
        if end_date > datetime.now().date():
            end_date = datetime.now().date()

    # Obtener la API key desde Streamlit Secrets
    api_key = st.secrets.get("news_data_key", "pub_7227626d8277642d9399e67d37a74d463f7cc")
    if not api_key:
        st.error("No se encontró la API key de NewsData.io en Secrets. Usando valor por defecto para sentimiento.")
        return 50.0

    # Construir la URL siguiendo la documentación de NewsData.io (#crypto-news), optimizada para noticias cripto
    query = f"{coin_symbol} AND (price OR market OR regulation)"  # Consulta específica para noticias cripto relevantes
    url = f"https://newsdata.io/api/1/news?apikey={api_key}&q={requests.utils.quote(query)}&language=en&from_date={start_date.strftime('%Y-%m-%d')}&to_date={end_date.strftime('%Y-%m-%d')}&size=5&category=crypto"
    
    try:
        # Verificar resolución DNS antes de la petición
        try:
            socket.getaddrinfo('newsdata.io', 443)
        except socket.gaierror as dns_error:
            st.error(f"Error de resolución DNS para newsdata.io: {dns_error}. Verifica tu conexión de red o DNS.")
            return 50.0

        resp = session.get(url, timeout=10, headers={"User-Agent": "Mozilla/5.0"})
        if resp.status_code == 200:
            data = resp.json()
            articles = data.get("results", [])
            if not articles:
                return 50.0  # Valor por defecto si no hay noticias, sin mensaje visible
            
            sentiments = []
            for article in articles[:5]:  # Limitar a 5 artículos (0.5 créditos por consulta)
                title = article.get("title", "").strip()
                description = article.get("description", "").strip()
                if title or description:
                    text = title if title else description
                    if ("price" in text.lower() or "market" in text.lower() or "regulation" in text.lower()):
                        blob = TextBlob(text)
                        sentiment = blob.sentiment.polarity
                        # Convertir de -1 a 1 a 0 a 100
                        sentiment_score = 50 + (sentiment * 50)  # Normalizar a 0-100
                        sentiments.append(sentiment_score)
            
            return np.mean(sentiments) if sentiments else 50.0
        elif resp.status_code == 422:
            return 50.0  # Valor por defecto si falla, sin mensaje visible
        elif resp.status_code == 429:
            st.error(f"Error 429 al obtener noticias de NewsData.io: Límite de créditos diarios (200) excedido.")
            return 50.0
        elif resp.status_code == 401:
            st.error(f"Error 401: Clave de API inválida o no autorizada. Verifica tu clave en Secrets.")
            return 50.0
        else:
            return 50.0  # Valor por defecto para otros errores, sin mensaje visible
    except requests.exceptions.ConnectionError as conn_error:
        st.error(f"Error de conexión con NewsData.io: {conn_error}. Verifica tu conexión de red o los límites de la API.")
        return 50.0
    except Exception as e:
        return 50.0  # Valor por defecto para cualquier otro error, sin mensaje visible

# Nueva función para obtener noticias recientes (usando NewsData.io, optimizada para crypto según #crypto-news)
@st.cache_data(ttl=3600)  # Cachear por hora para minimizar peticiones
def get_recent_crypto_news(coin_symbol):
    """Obtiene las noticias más recientes y relevantes de criptomonedas usando NewsData.io."""
    end_date = datetime.now().date()
    start_date = end_date - timedelta(days=14)  # Aumentado a 14 días para capturar más noticias recientes

    # Obtener la API key desde Streamlit Secrets
    api_key = st.secrets.get("news_data_key", "pub_7227626d8277642d9399e67d37a74d463f7cc")
    if not api_key:
        st.error("No se encontró la API key de NewsData.io en Secrets. No se pueden mostrar noticias.")
        return []

    # Construir la URL siguiendo la documentación de NewsData.io (#crypto-news), optimizada para noticias cripto recientes
    query = f"crypto AND {coin_symbol}"  # Consulta simplificada para capturar más noticias relevantes
    url = f"https://newsdata.io/api/1/news?apikey={api_key}&q={requests.utils.quote(query)}&language=en&from_date={start_date.strftime('%Y-%m-%d')}&to_date={end_date.strftime('%Y-%m-%d')}&size=10&category=crypto&sort_by=pubDate"
    
    try:
        # Verificar resolución DNS antes de la petición
        try:
            socket.getaddrinfo('newsdata.io', 443)
        except socket.gaierror as dns_error:
            st.error(f"Error de resolución DNS para newsdata.io: {dns_error}. No se pueden mostrar noticias.")
            return []

        resp = session.get(url, timeout=10, headers={"User-Agent": "Mozilla/5.0"})
        if resp.status_code == 200:
            data = resp.json()
            articles = data.get("results", [])
            if not articles:
                # Intentar con una consulta más genérica y rango reducido (7 días)
                query_simple = "crypto"  # Consulta genérica para capturar cualquier noticia cripto
                start_date_simple = end_date - timedelta(days=7)
                url_retry = f"https://newsdata.io/api/1/news?apikey={api_key}&q={requests.utils.quote(query_simple)}&language=en&from_date={start_date_simple.strftime('%Y-%m-%d')}&to_date={end_date.strftime('%Y-%m-%d')}&size=10&category=crypto&sort_by=pubDate"
                resp_retry = session.get(url_retry, timeout=10, headers={"User-Agent": "Mozilla/5.0"})
                if resp_retry.status_code == 200:
                    data_retry = resp_retry.json()
                    articles_retry = data_retry.get("results", [])
                    if articles_retry:
                        articles_retry = sorted(articles_retry, key=lambda x: x.get("pubDate", ""), reverse=True)[:5]
                        return [
                            {
                                "title": article.get("title", "Sin título"),
                                "description": article.get("description", "Sin descripción"),
                                "pubDate": article.get("pubDate", "Fecha no disponible"),
                                "link": article.get("link", "#")
                            }
                            for article in articles_retry
                        ]
                return []  # Sin noticias si no hay resultados, sin mensaje visible
            
            # Ordenar por fecha de publicación (pubDate) y limitar a las 5 más recientes
            articles = sorted(articles, key=lambda x: x.get("pubDate", ""), reverse=True)[:5]
            return [
                {
                    "title": article.get("title", "Sin título"),
                    "description": article.get("description", "Sin descripción"),
                    "pubDate": article.get("pubDate", "Fecha no disponible"),
                    "link": article.get("link", "#")
                }
                for article in articles
            ]
        elif resp.status_code == 422:
            return []  # Sin noticias si falla, sin mensaje visible
        elif resp.status_code == 429:
            st.error(f"Error 429 al obtener noticias de NewsData.io: Límite de créditos diarios (200) excedido.")
            return []
        elif resp.status_code == 401:
            st.error(f"Error 401: Clave de API inválida o no autorizada. Verifica tu clave en Secrets.")
            return []
        else:
            return []  # Sin noticias para otros errores, sin mensaje visible
    except requests.exceptions.ConnectionError as conn_error:
        st.error(f"Error de conexión con NewsData.io: {conn_error}. No se pueden mostrar noticias.")
        return []
    except Exception as e:
        return []  # Sin noticias para cualquier otro error, sin mensaje visible

# Predicción
def train_and_predict_with_sentiment(coin_id, horizon_days, start_ms=None, end_ms=None):
    """Entrena y predice combinando modelos, sentimiento y noticias específicas de cripto."""
    df = load_coincap_data(coin_id, start_ms, end_ms)
    if df is None:
        return None
    symbol = coinid_to_symbol[coin_id]

    # Obtener sentimiento de noticias para el rango de fechas (si aplica), con manejo de errores
    start_date = datetime.fromtimestamp(start_ms / 1000) if start_ms else (datetime.now() - timedelta(days=7))
    end_date = datetime.fromtimestamp(end_ms / 1000) if end_ms else datetime.now()
    news_sent = get_news_sentiment(symbol, start_date.date(), end_date.date())

    # Asegurar que news_sent sea un número válido antes de usarlo
    news_sent = 50.0 if news_sent is None or pd.isna(news_sent) else float(news_sent)
    # No mostrar mensaje visible aquí; el manejo de errores ya está en get_news_sentiment

    crypto_sent = get_crypto_sentiment_combined(coin_id, news_sent)
    market_sent = get_fear_greed_index()
    sentiment_factor = (crypto_sent + market_sent) / 200.0

    scaler = MinMaxScaler()
    scaled_data = scaler.fit_transform(df[["close_price"]])
    window_size, epochs, batch_size, learning_rate = get_dynamic_params(df, horizon_days, coin_id)
    X, y = create_sequences(scaled_data, window_size)
    if X is None:
        return None

    split = int(len(X) * 0.8)
    X_train, X_test = X[:split], X[split:]
    y_train, y_test = y[:split], y[split:]
    val_split = int(len(X_train) * 0.9)
    X_val, y_val = X_train[val_split:], y_train[val_split:]
    X_train, y_train = X_train[:val_split], y_train[:val_split]

    # Ajustar dimensiones para incluir sentimiento
    X_train_adj = np.concatenate([X_train, np.full((X_train.shape[0], window_size, 1), sentiment_factor)], axis=-1)
    X_val_adj = np.concatenate([X_val, np.full((X_val.shape[0], window_size, 1), sentiment_factor)], axis=-1)
    X_test_adj = np.concatenate([X_test, np.full((X_test.shape[0], window_size, 1), sentiment_factor)], axis=-1)

    # Entrenar modelo y obtener historial para depuración
    lstm_model, history = train_model(X_train_adj, y_train, X_val_adj, y_val, (window_size, 2), epochs, batch_size)
    lstm_test_preds_scaled = lstm_model.predict(X_test_adj, verbose=0)
    lstm_test_preds = scaler.inverse_transform(lstm_test_preds_scaled).flatten()  # Asegurar 1D
    y_test_real = scaler.inverse_transform(y_test.reshape(-1, 1)).flatten()  # Asegurar 1D
    lstm_rmse = np.sqrt(mean_squared_error(y_test_real, lstm_test_preds))
    lstm_mape = robust_mape(y_test_real, lstm_test_preds)

    # Predicciones futuras con sentimiento
    last_window = scaled_data[-window_size:]
    future_preds = []
    current_input = np.concatenate([last_window.reshape(1, window_size, 1), np.full((1, window_size, 1), sentiment_factor)], axis=-1)
    for _ in range(horizon_days):
        pred = lstm_model.predict(current_input, verbose=0)[0][0]
        future_preds.append(pred)
        new_feature = np.copy(current_input[:, -1:, :])
        new_feature[0, 0, 0] = pred
        new_feature[0, 0, 1] = sentiment_factor
        current_input = np.append(current_input[:, 1:, :], new_feature, axis=1)
    future_preds = scaler.inverse_transform(np.array(future_preds).reshape(-1, 1)).flatten()

    last_date = df["ds"].iloc[-1]
    future_dates = pd.date_range(start=last_date + timedelta(days=1), periods=horizon_days).tolist()

    test_dates = df["ds"].iloc[-len(lstm_test_preds):].values  # Fechas para el set de test
    real_prices = df["close_price"].iloc[-len(lstm_test_preds):].values  # Precios reales del set de test

    return {
        "df": df,
        "test_preds": lstm_test_preds,
        "future_preds": future_preds,  # Usamos "future_preds" para consistencia
        "rmse": lstm_rmse,
        "mape": lstm_mape,
        "sentiment_factor": sentiment_factor,
        "symbol": symbol,
        "crypto_sent": crypto_sent,
        "market_sent": market_sent,
        "future_dates": future_dates,
        "test_dates": test_dates,
        "real_prices": real_prices
    }

# Aplicación principal
def main_app():
    st.set_page_config(page_title="Crypto Price Predictions 🔮", layout="wide")
    st.title("Crypto Price Predictions 🔮")
    st.markdown("""
    **Descripción del Modelo:**  
    Esta plataforma utiliza un modelo avanzado de aprendizaje automático basado en redes LSTM (Long Short-Term Memory) para predecir precios futuros de criptomonedas como Bitcoin, Ethereum, Ripple y otras. El modelo integra datos históricos de precios y volúmenes de CoinCap, abarcando hasta dos años de información diaria, ajustando dinámicamente sus hiperparámetros (como tamaño de ventana, épocas, tamaño de lote y tasa de aprendizaje) según la volatilidad específica de cada criptomoneda. Además, incorpora un análisis de sentimiento dinámico que combina el índice Fear & Greed para el mercado global, la actividad comunitaria en redes sociales (Twitter y Reddit) de CoinGecko para cada cripto, y noticias específicas de criptomonedas obtenidas a través de NewsData.io, mejorando la precisión al considerar el estado de ánimo del mercado, los inversores y eventos externos. Las predicciones se complementan con métricas clave como RMSE y MAPE para evaluar la precisión, y se presentan en gráficos interactivos y tablas para una experiencia clara y detallada.

    Fuentes de datos: CoinCap, Fear & Greed Index, CoinGecko, NewsData.io
    """)

    # Sidebar
    st.sidebar.title("Configura tu Predicción")
    crypto_name = st.sidebar.selectbox("Selecciona una criptomoneda:", list(coincap_ids.keys()))
    coin_id = coincap_ids[crypto_name]
    use_custom_range = st.sidebar.checkbox("Habilitar rango de fechas", value=False)
    default_end = datetime.now()
    default_start = default_end - timedelta(days=7)  # Reducido a 7 días por defecto para mayor eficiencia
    if use_custom_range:
        start_date = st.sidebar.date_input("Fecha de inicio", default_start.date())
        end_date = st.sidebar.date_input("Fecha de fin", default_end.date())
        # Validar que las fechas sean válidas y no excedan un rango razonable
        if start_date > end_date:
            st.sidebar.error("La fecha de inicio no puede ser posterior a la fecha de fin.")
            return
        if (end_date - start_date).days > 7:
            st.sidebar.warning("El rango de fechas excede 7 días. Ajustando al máximo permitido (7 días).")
            end_date = start_date + timedelta(days=7)
        # Asegurar que las fechas no sean futuras
        if start_date > datetime.now().date():
            start_date = datetime.now().date() - timedelta(days=7)
            st.sidebar.warning("La fecha de inicio no puede ser futura. Ajustando al rango máximo permitido (7 días atrás).")
        if end_date > datetime.now().date():
            end_date = datetime.now().date()
            st.sidebar.warning("La fecha de fin no puede ser futura. Ajustando a hoy.")
        start_ms = int(datetime.combine(start_date, datetime.min.time()).timestamp() * 1000)
        end_ms = int(datetime.combine(end_date, datetime.min.time()).timestamp() * 1000)
    else:
        start_ms = int(default_start.timestamp() * 1000)
        end_ms = int(default_end.timestamp() * 1000)
    horizon = st.sidebar.slider("Días a predecir:", 1, 60, 5)
    st.sidebar.markdown("**Los hiperparámetros se ajustan automáticamente según los datos.**")
    show_stats = st.sidebar.checkbox("Ver estadísticas descriptivas", value=False)

    # Gráfico histórico
    df_prices = load_coincap_data(coin_id, start_ms, end_ms)
    if df_prices is not None:
        fig_hist = px.line(df_prices, x="ds", y="close_price", title=f"Histórico de {crypto_name}", labels={"ds": "Fecha", "close_price": "Precio en USD"})
        fig_hist.update_layout(template="plotly_dark")
        st.plotly_chart(fig_hist, use_container_width=True)
        if show_stats:
            st.subheader("Estadísticas Descriptivas")
            st.write(df_prices["close_price"].describe())

    # Pestañas
    tabs = st.tabs(["🤖 Entrenamiento y Test", "🔮 Predicción de Precios", "📊 Análisis de Sentimientos", "📰 Noticias Recientes"])
    with tabs[0]:
        st.header("Entrenamiento del Modelo y Evaluación en Test")
        if st.button("Entrenar Modelo y Predecir"):
            with st.spinner("Esto puede tardar un poco, por favor espera..."):  # Mensaje cambiado según tu solicitud
                result = train_and_predict_with_sentiment(coin_id, horizon, start_ms, end_ms)
            if result:
                st.success("Entrenamiento y predicción completados!")
                st.write(f"Sentimiento combinado de {result['symbol']}: {result['crypto_sent']:.2f}")
                st.write(f"Sentimiento global del mercado: {result['market_sent']:.2f}")
                st.write(f"Factor combinado: {result['sentiment_factor']:.2f}")
                col1, col2 = st.columns(2)
                col1.metric("RMSE (Test)", f"{result['rmse']:.2f}", help="Error promedio en dólares.")
                col2.metric("MAPE (Test)", f"{result['mape']:.2f}%", help="Error relativo promedio.")

                # Verificación estricta de dimensiones
                if len(result["test_dates"]) != len(result["test_preds"]) or len(result["test_dates"]) != len(result["real_prices"]):
                    st.error(f"Error en las dimensiones de los datos: test_dates ({len(result['test_dates'])}), test_preds ({len(result['test_preds'])}), real_prices ({len(result['real_prices'])}). Ajustando...")
                    min_len = min(len(result["test_dates"]), len(result["test_preds"]), len(result["real_prices"]))
                    result["test_dates"] = result["test_dates"][:min_len]
                    result["test_preds"] = result["test_preds"][:min_len]
                    result["real_prices"] = result["real_prices"][:min_len]

                # Crear el gráfico mejorado para precio real y predicción (solo líneas, sin fondo ni configuraciones adicionales, con días en el eje X)
                if len(result["test_dates"]) > 0 and len(result["real_prices"]) > 0 and len(result["test_preds"]) > 0:
                    fig_test = go.Figure()
                    fig_test.add_trace(go.Scatter(
                        x=result["test_dates"],
                        y=result["real_prices"],
                        mode="lines",
                        name="Precio Real",
                        line=dict(color="#1f77b4", width=3)  # Azul oscuro, línea sólida más gruesa
                    ))
                    fig_test.add_trace(go.Scatter(
                        x=result["test_dates"],
                        y=result["test_preds"],
                        mode="lines",
                        name="Predicción",
                        line=dict(color="#ff7f0e", width=3, dash="dash")  # Naranja, línea discontinua más gruesa
                    ))
                    fig_test.update_layout(
                        title=f"Comparación entre el precio real y la predicción: {result['symbol']}",
                        xaxis=dict(tickformat="%Y-%m-%d")  # Mostrar solo días en el eje X
                    )  # Solo título y formato de días, sin fondo ni otras configuraciones
                    st.plotly_chart(fig_test, use_container_width=True)
                else:
                    st.error("No hay suficientes datos para mostrar el gráfico de entrenamiento y test.")
                st.session_state["result"] = result

    with tabs[1]:
        st.header(f"Predicción de Precios - {crypto_name}")
        if "result" in st.session_state:
            # Verificar si result es un diccionario
            if isinstance(st.session_state["result"], dict):
                result = st.session_state["result"]
                last_date = result["df"]["ds"].iloc[-1]
                current_price = result["df"]["close_price"].iloc[-1]
                pred_series = np.concatenate(([current_price], result["future_preds"]))  # Usamos "future_preds" para consistencia
                fig_future = go.Figure()
                future_dates_display = [last_date] + result["future_dates"]
                fig_future.add_trace(go.Scatter(x=future_dates_display, y=pred_series, mode="lines+markers", name="Predicción", line=dict(color="#ff7f0e", width=2)))
                fig_future.update_layout(title=f"Predicción a Futuro ({horizon} días) - {result['symbol']}", template="plotly_dark", xaxis_title="Fecha", yaxis_title="Precio en USD", plot_bgcolor="#1e1e2f", paper_bgcolor="#1e1e2f")
                st.plotly_chart(fig_future, use_container_width=True)
                st.subheader("Valores Numéricos")
                st.dataframe(pd.DataFrame({"Fecha": future_dates_display, "Predicción": pred_series}).style.format({"Predicción": "{:.2f}"}))
            else:
                st.error("El resultado almacenado no es un diccionario válido. Por favor, entrena el modelo nuevamente.")
        else:
            st.info("Entrena el modelo primero.")

    with tabs[2]:
        st.header("📊 Análisis de Sentimientos")
        if "result" in st.session_state:
            # Verificar si result es un diccionario
            if isinstance(st.session_state["result"], dict):
                result = st.session_state["result"]
                crypto_sent, market_sent = result["crypto_sent"], result["market_sent"]
                level = (crypto_sent - 50) / 5  # Escala -10 a 10 para determinar el estado
                sentiment_label = "Very Bearish" if level <= -5 else "Bearish" if level <= -2 else \
                                 "Neutral" if -2 < level < 2 else "Bullish" if level <= 5 else "Very Bullish"
                color = "#ff7f0e" if level < 0 else "#1f77b4"  # Naranja para bearish, azul para bullish

                # Ajustar el threshold para que esté en el valor exacto de crypto_sent
                threshold_value = crypto_sent  # Colocar la línea blanca en el valor exacto de crypto_sent

                # Mejorar el diseño del gauge para hacerlo más amigable y dinámico
                fig_sentiment = go.Figure(go.Indicator(
                    mode="gauge+number+delta",
                    value=crypto_sent,
                    domain={"x": [0, 1], "y": [0, 1]},
                    title={
                        "text": f"Sentimiento - {result['symbol']}",
                        "font": {"size": 24, "color": "#ffffff", "family": "Arial, sans-serif"}
                    },
                    gauge={
                        "axis": {
                            "range": [0, 100],
                            "tickvals": [0, 25, 40, 50, 60, 75, 100],
                            "ticktext": ["Very Bearish", "Bearish", "", "Neutral", "", "Bullish", "Very Bullish"],
                            "tickcolor": "#ffffff",
                            "tickwidth": 2,
                            "tickfont": {"size": 14, "color": "#ffffff"}
                        },
                        "bar": {"color": color},
                        "bgcolor": "#2c2c3e",
                        "borderwidth": 2,
                        "bordercolor": "#4a4a6a",
                        "steps": [
                            {"range": [0, 25], "color": "#ff7f0e"},  # Very Bearish, naranja oscuro
                            {"range": [25, 40], "color": "#ffaa7f"},  # Bearish, naranja claro
                            {"range": [40, 60], "color": "#666666"},  # Neutral, gris
                            {"range": [60, 75], "color": "#7fb4ff"},  # Bullish, azul claro
                            {"range": [75, 100], "color": "#1f77b4"}  # Very Bullish, azul oscuro
                        ],
                        "threshold": {
                            "line": {"color": "#ffffff", "width": 4},
                            "thickness": 0.75,
                            "value": threshold_value  # Ajustado dinámicamente al valor de crypto_sent
                        }
                    },
                    delta={
                        "reference": market_sent,
                        "increasing": {"color": "#1f77b4"},
                        "decreasing": {"color": "#ff7f0e"}
                    },
                    number={"font": {"size": 48, "color": "#ffffff", "family": "Arial, sans-serif"}}
                ))
                fig_sentiment.update_layout(
                    template="plotly_dark",
                    plot_bgcolor="#1e1e2f",
                    paper_bgcolor="#1e1e2f",
                    height=500,  # Aumentado para mayor visibilidad
                    width=800,  # Aumentado para mayor visibilidad
                    margin=dict(l=20, r=20, t=80, b=20)  # Ajuste de márgenes para mejor presentación
                )
                st.plotly_chart(fig_sentiment, use_container_width=True)
                st.write(f"**Estado:** {sentiment_label} (Mercado: {market_sent:.2f})")
                st.write("**NFA (Not Financial Advice):** Esto es solo información educativa, no un consejo financiero. Consulta a un experto antes de invertir.")
            else:
                st.error("El resultado almacenado no es un diccionario válido. Por favor, entrena el modelo nuevamente.")
        else:
            st.info("Entrena el modelo para ver el análisis.")

    with tabs[3]:
        st.header("📰 Noticias Recientes de Criptomonedas")
        news = get_recent_crypto_news(coinid_to_symbol[coin_id])
        if news:
            st.subheader(f"Últimas 5 noticias sobre {crypto_name}")
            for article in news:
                with st.expander(f"**{article['title']}** - {article['pubDate']}", expanded=False):
                    st.write(article['description'])
                    if article['link']:
                        st.markdown(f"[Leer más]({article['link']})", unsafe_allow_html=True)
            # Mostrar en forma de tabla para mejor UX/UI
            news_df = pd.DataFrame(news)
            st.dataframe(news_df[["title", "pubDate"]].style.format({"pubDate": "{:%Y-%m-%d %H:%M:%S}"}).set_properties(**{'background-color': '#2c2c3e', 'color': 'white', 'border-color': '#4a4a6a'}))
        else:
            st.info("No se encontraron noticias recientes. Verifica tu conexión, los límites de la API, o intenta más tarde.")

if __name__ == "__main__":
    main_app()