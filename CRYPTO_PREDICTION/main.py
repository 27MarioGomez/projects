import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import requests
from datetime import datetime, timedelta
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
import time
import certifi
import os
from sklearn.metrics import mean_squared_error
from textblob import TextBlob
import socket  # Para manejar errores de DNS

# Configuración inicial de certificados SSL y sesión de requests
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

# Secuencias y modelo LSTM
def create_sequences(data, window_size):
    """Crea secuencias para el modelo LSTM."""
    if len(data) <= window_size:
        return None, None
    X, y = [], []
    for i in range(window_size, len(data)):
        X.append(data[i - window_size:i])
        y.append(data[i, 0])
    return np.array(X), np.array(y)

def build_lstm_model(input_shape, learning_rate=0.001):
    """Construye el modelo LSTM."""
    model = Sequential([
        LSTM(50, return_sequences=True, input_shape=input_shape),
        Dropout(0.2),
        LSTM(50),
        Dropout(0.2),
        Dense(20, activation="relu"),
        Dense(1)
    ])
    model.compile(optimizer=Adam(learning_rate), loss="mse")
    return model

def train_model(X_train, y_train, X_val, y_val, input_shape, epochs, batch_size):
    """Entrena el modelo LSTM."""
    tf.keras.backend.clear_session()
    model = build_lstm_model(input_shape)
    callbacks = [
        EarlyStopping(patience=5, restore_best_weights=True),
        ReduceLROnPlateau(patience=3, factor=0.5, min_lr=1e-6)
    ]
    model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=epochs, batch_size=batch_size, callbacks=callbacks, verbose=0)
    return model

def get_dynamic_params(df, horizon_days, coin_id):
    """Ajusta parámetros dinámicos según volatilidad y datos."""
    volatility = df["close_price"].pct_change().std()
    base_volatility = crypto_characteristics.get(coin_id, {"volatility": 0.05})["volatility"]
    # Ajustes más agresivos para criptos volátiles como XRP
    window_size = min(max(10, int(horizon_days * (1.2 if volatility > base_volatility else 1.5))), len(df) // 4)
    epochs = min(100, max(30, int(len(df) / 80) + int(volatility * 200)))  # Más épocas para datos complejos
    batch_size = 16 if volatility > base_volatility else 32  # Batch menor para volátiles
    learning_rate = 0.0003 if volatility > base_volatility else 0.0005  # LR menor para volátiles
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
        st.warning(f"No se pudo obtener actividad de CoinGecko para {coin_id}. Usando valor por defecto.")
        return 50.0

def get_crypto_sentiment_combined(coin_id, news_sentiment=None):
    """Calcula el sentimiento combinado dinámico con noticias políticas y pesos ajustados por volatilidad."""
    fg = get_fear_greed_index()
    cg = get_coingecko_community_activity(coin_id)
    volatility = crypto_characteristics.get(coin_id, {"volatility": 0.05})["volatility"]

    # Ajustar pesos: más peso a CoinGecko y noticias para criptos volátiles, menos a Fear & Greed
    if volatility > 0.07:  # Criptos muy volátiles (e.g., XRP, DOGE)
        fg_weight = 0.2  # Menos peso al mercado global
        cg_weight = 0.5  # Peso moderado a la actividad comunitaria
        news_weight = 0.3  # Más peso a las noticias políticas para capturar volatilidad
    else:  # Criptos más estables (e.g., BTC, ETH)
        fg_weight = 0.5  # Más peso al mercado global
        cg_weight = 0.3  # Menos peso a la actividad comunitaria
        news_weight = 0.2  # Menos peso a las noticias, ya que son menos críticas para estables

    # Sentimiento de noticias (si no hay datos o falla, usar 50.0 como valor por defecto)
    news_sent = 50.0 if news_sentiment is None or pd.isna(news_sentiment) else float(news_sentiment)
    combined_sentiment = (fg * fg_weight + cg * cg_weight + news_sent * news_weight)
    return max(0, min(100, combined_sentiment))  # Asegurar rango 0-100

# Nueva función para análisis de noticias (usando NewsData.io con API key desde Secrets, ajustada según documentación)
@st.cache_data(ttl=86400)  # Cachear datos diarios para minimizar peticiones
def get_news_sentiment(coin_symbol, start_date=None, end_date=None):
    """Obtiene y analiza el sentimiento de noticias políticas y relevantes usando NewsData.io."""
    if start_date is None or end_date is None:
        end_date = datetime.now().date()
        start_date = end_date - timedelta(days=30)  # Rango por defecto de 30 días
    else:
        # Validar que el rango no exceda 30 días para evitar errores 422
        if (end_date - start_date).days > 30:
            start_date = end_date - timedelta(days=30)
        # Asegurar que las fechas no sean futuras
        if start_date > datetime.now().date():
            start_date = datetime.now().date() - timedelta(days=30)
        if end_date > datetime.now().date():
            end_date = datetime.now().date()

    # Obtener la API key desde Streamlit Secrets
    api_key = st.secrets.get("news_data_key", "pub_7227626d8277642d9399e67d37a74d463f7cc")
    if not api_key:
        st.error("No se encontró la API key de NewsData.io en Secrets. Usando valor por defecto para sentimiento.")
        return 50.0

    # Construir la URL siguiendo la documentación de NewsData.io
    query = f"{coin_symbol} AND (crypto OR regulation OR policy)"
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
                if title:
                    blob = TextBlob(title)
                    sentiment = blob.sentiment.polarity
                    # Convertir de -1 a 1 a 0 a 100
                    sentiment_score = 50 + (sentiment * 50)  # Normalizar a 0-100
                    sentiments.append(sentiment_score)
            
            return np.mean(sentiments) if sentiments else 50.0
        elif resp.status_code == 422:
            # Intentar con una consulta simplificada y rango reducido (7 días)
            query_simple = f"{coin_symbol} AND crypto"
            start_date_simple = end_date - timedelta(days=7)
            url_retry = f"https://newsdata.io/api/1/news?apikey={api_key}&q={requests.utils.quote(query_simple)}&language=en&from_date={start_date_simple.strftime('%Y-%m-%d')}&to_date={end_date.strftime('%Y-%m-%d')}&size=5&category=crypto"
            resp_retry = session.get(url_retry, timeout=10, headers={"User-Agent": "Mozilla/5.0"})
            if resp_retry.status_code == 200:
                data_retry = resp_retry.json()
                articles_retry = data_retry.get("results", [])
                if articles_retry:
                    sentiments = []
                    for article in articles_retry[:5]:
                        title = article.get("title", "").strip()
                        if title:
                            blob = TextBlob(title)
                            sentiment = blob.sentiment.polarity
                            sentiment_score = 50 + (sentiment * 50)
                            sentiments.append(sentiment_score)
                    return np.mean(sentiments) if sentiments else 50.0
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

# Predicción
def train_and_predict_with_sentiment(coin_id, horizon_days, start_ms=None, end_ms=None):
    """Entrena y predice combinando modelos, sentimiento y noticias."""
    df = load_coincap_data(coin_id, start_ms, end_ms)
    if df is None:
        return None
    symbol = coinid_to_symbol[coin_id]

    # Obtener sentimiento de noticias para el rango de fechas (si aplica), con manejo de errores
    start_date = datetime.fromtimestamp(start_ms / 1000).date() if start_ms else None
    end_date = datetime.fromtimestamp(end_ms / 1000).date() if end_ms else None
    news_sent = get_news_sentiment(symbol, start_date, end_date)

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

    X_train_adj = np.concatenate([X_train, np.full((X_train.shape[0], window_size, 1), sentiment_factor)], axis=-1)
    X_val_adj = np.concatenate([X_val, np.full((X_val.shape[0], window_size, 1), sentiment_factor)], axis=-1)
    X_test_adj = np.concatenate([X_test, np.full((X_test.shape[0], window_size, 1), sentiment_factor)], axis=-1)

    lstm_model = train_model(X_train_adj, y_train, X_val_adj, y_val, (window_size, 2), epochs, batch_size)
    lstm_test_preds_scaled = lstm_model.predict(X_test_adj, verbose=0)
    lstm_test_preds = scaler.inverse_transform(lstm_test_preds_scaled).flatten()  # Asegurar 1D
    y_test_real = scaler.inverse_transform(y_test.reshape(-1, 1)).flatten()  # Asegurar 1D
    lstm_rmse = np.sqrt(mean_squared_error(y_test_real, lstm_test_preds))
    lstm_mape = robust_mape(y_test_real, lstm_test_preds)

    last_window = scaled_data[-window_size:]
    future_preds_scaled = []
    current_input = np.concatenate([last_window.reshape(1, window_size, 1), np.full((1, window_size, 1), sentiment_factor)], axis=-1)
    for _ in range(horizon_days):
        pred = lstm_model.predict(current_input, verbose=0)[0][0]
        future_preds_scaled.append(pred)
        new_feature = np.copy(current_input[:, -1:, :])
        new_feature[0, 0, 0] = pred
        new_feature[0, 0, 1] = sentiment_factor
        current_input = np.append(current_input[:, 1:, :], new_feature, axis=1)
    lstm_future_preds = scaler.inverse_transform(np.array(future_preds_scaled).reshape(-1, 1)).flatten()

    last_date = df["ds"].iloc[-1]
    future_dates = pd.date_range(start=last_date + timedelta(days=1), periods=horizon_days).tolist()

    test_dates = df["ds"].iloc[-len(lstm_test_preds):].values  # Fechas para el set de test
    real_prices = df["close_price"].iloc[-len(lstm_test_preds):].values  # Precios reales del set de test

    return {
        "df": df,
        "test_preds": lstm_test_preds,
        "future_preds": lstm_future_preds,
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
    Esta plataforma utiliza un modelo avanzado de aprendizaje automático basado en redes LSTM (Long Short-Term Memory) para predecir precios futuros de criptomonedas como Bitcoin, Ethereum, Ripple y otras. El modelo integra datos históricos de precios y volúmenes de CoinCap, abarcando hasta dos años de información diaria, ajustando dinámicamente sus hiperparámetros (como tamaño de ventana, épocas, tamaño de lote y tasa de aprendizaje) según la volatilidad específica de cada criptomoneda. Además, incorpora un análisis de sentimiento dinámico que combina el índice Fear & Greed para el mercado global, la actividad comunitaria en redes sociales (Twitter y Reddit) de CoinGecko para cada cripto, y noticias políticas y económicas relevantes obtenidas a través de NewsData.io, mejorando la precisión al considerar el estado de ánimo del mercado, los inversores y eventos externos. Las predicciones se complementan con métricas clave como RMSE y MAPE para evaluar la precisión, y se presentan en gráficos interactivos y tablas para una experiencia clara y detallada.

    Fuentes de datos: CoinCap, Fear & Greed Index, CoinGecko, NewsData.io
    """)

    # Sidebar
    st.sidebar.title("Configura tu Predicción")
    crypto_name = st.sidebar.selectbox("Selecciona una criptomoneda:", list(coincap_ids.keys()))
    coin_id = coincap_ids[crypto_name]
    use_custom_range = st.sidebar.checkbox("Habilitar rango de fechas", value=False)
    default_end = datetime.now().date()
    default_start = default_end - timedelta(days=30)  # Reducido a 30 días por defecto para evitar errores 422
    if use_custom_range:
        start_date = st.sidebar.date_input("Fecha de inicio", default_start)
        end_date = st.sidebar.date_input("Fecha de fin", default_end)
        # Validar que las fechas sean válidas y no excedan un rango razonable
        if start_date > end_date:
            st.sidebar.error("La fecha de inicio no puede ser posterior a la fecha de fin.")
            return
        if (end_date - start_date).days > 30:
            st.sidebar.warning("El rango de fechas excede 30 días. Ajustando al máximo permitido (30 días).")
            end_date = start_date + timedelta(days=30)
        # Asegurar que las fechas no sean futuras
        if start_date > datetime.now().date():
            start_date = datetime.now().date() - timedelta(days=30)
            st.sidebar.warning("La fecha de inicio no puede ser futura. Ajustando al rango máximo permitido (30 días atrás).")
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
    tabs = st.tabs(["🤖 Entrenamiento y Test", "🔮 Predicción de Precios", "📊 Análisis de Sentimientos"])
    with tabs[0]:
        st.header("Entrenamiento del Modelo y Evaluación en Test")
        if st.button("Entrenar Modelo y Predecir"):
            with st.spinner("Procesando..."):
                result = train_and_predict_with_sentiment(coin_id, horizon, start_ms, end_ms)
            if result:
                st.success("Entrenamiento y predicción completados!")
                st.write(f"Sentimiento combinado de {result['symbol']}: {result['crypto_sent']:.2f}")
                st.write(f"Sentimiento global del mercado: {result['market_sent']:.2f}")
                st.write(f"Factor combinado: {result['sentiment_factor']:.2f}")
                col1, col2 = st.columns(2)
                col1.metric("RMSE (Test)", f"{result['rmse']:.2f}", help="Error promedio en dólares.")
                col2.metric("MAPE (Test)", f"{result['mape']:.2f}%", help="Error relativo promedio.")

                # Verificación de dimensiones
                if len(result["test_dates"]) != len(result["test_preds"]):
                    st.warning(f"Advertencia: Longitud de test_dates ({len(result['test_dates'])}) no coincide con test_preds ({len(result['test_preds'])}). Ajustando...")
                    min_len = min(len(result["test_dates"]), len(result["test_preds"]))
                    result["test_dates"] = result["test_dates"][:min_len]
                    result["test_preds"] = result["test_preds"][:min_len]
                    result["real_prices"] = result["real_prices"][:min_len]

                # Crear el gráfico
                fig_test = go.Figure()
                fig_test.add_trace(go.Scatter(
                    x=result["test_dates"],
                    y=result["real_prices"],
                    mode="lines",
                    name="Precio Real",
                    line=dict(color="blue")
                ))
                fig_test.add_trace(go.Scatter(
                    x=result["test_dates"],
                    y=result["test_preds"],
                    mode="lines",
                    name="Predicción",
                    line=dict(color="orange", dash="dash")
                ))
                fig_test.update_layout(
                    title=f"Comparación entre el precio real y la predicción: {result['symbol']}",
                    template="plotly_dark",
                    xaxis_title="Fecha",
                    yaxis_title="Precio en USD"
                )
                st.plotly_chart(fig_test, use_container_width=True)
                st.session_state["result"] = result

    with tabs[1]:
        st.header(f"Predicción de Precios - {crypto_name}")
        if "result" in st.session_state:
            # Verificar si result es un diccionario
            if isinstance(st.session_state["result"], dict):
                result = st.session_state["result"]
                last_date = result["df"]["ds"].iloc[-1]
                current_price = result["df"]["close_price"].iloc[-1]
                pred_series = np.concatenate(([current_price], result["future_preds"]))
                fig_future = go.Figure()
                future_dates_display = [last_date] + result["future_dates"]
                fig_future.add_trace(go.Scatter(x=future_dates_display, y=pred_series, mode="lines+markers", name="Predicción"))
                fig_future.update_layout(title=f"Predicción a Futuro ({horizon} días) - {result['symbol']}", template="plotly_dark")
                st.plotly_chart(fig_future, use_container_width=True)
                st.subheader("Valores Numéricos")
                st.dataframe(pd.DataFrame({"Fecha": future_dates_display, "Predicción": pred_series}))
            else:
                st.error("El resultado almacenado no es un diccionario válido. Por favor, entrena el modelo nuevamente.")
        else:
            st.info("Entrena el modelo primero.")

    with tabs[2]:
        st.header("Análisis de Sentimientos")
        if "result" in st.session_state:
            # Verificar si result es un diccionario
            if isinstance(st.session_state["result"], dict):
                result = st.session_state["result"]
                sentiment_texts = {
                    "BTC": f"El sentimiento de Bitcoin está en {result['crypto_sent']:.2f}, lo que muestra cierta cautela entre los inversores, aunque su comunidad sigue activa. El mercado en general está en {result['market_sent']:.2f}, indicando miedo. Con un factor combinado de {result['sentiment_factor']:.2f}, parece que Bitcoin podría mantenerse estable, pero no esperes grandes subidas pronto. ¡Ojo con las noticias!",
                    "ETH": f"Ethereum tiene un sentimiento de {result['crypto_sent']:.2f}, reflejando dudas, pero su tecnología sigue siendo un punto fuerte. El mercado está en {result['market_sent']:.2f}, con miedo dominando. El factor combinado de {result['sentiment_factor']:.2f} sugiere que podría haber oportunidades si el ánimo mejora. Estate atento a sus actualizaciones.",
                    "XRP": f"XRP está en {result['crypto_sent']:.2f}, mostrando pesimismo en su comunidad, y el mercado en {result['market_sent']:.2f} no ayuda mucho. Con un factor combinado de {result['sentiment_factor']:.2f}, parece que XRP podría seguir moviéndose poco a menos que haya noticias grandes, como su caso legal. Cuidado con la volatilidad."
                }
                sentiment_text = sentiment_texts.get(result['symbol'], f"El sentimiento de {result['symbol']} está en {result['crypto_sent']:.2f}, lo que indica {'optimismo' if result['crypto_sent'] > 50 else 'pesimismo'} entre sus seguidores. El mercado general está en {result['market_sent']:.2f}. Con un factor combinado de {result['sentiment_factor']:.2f}, hay {'potencial' if result['sentiment_factor'] > 0.5 else 'cautela'} a corto plazo.")
                st.write(sentiment_text)
                fig_sentiment = go.Figure(data=[
                    go.Bar(name="Sentimiento Combinado", x=[result['symbol']], y=[result['crypto_sent']], marker_color="#1f77b4"),
                    go.Bar(name="Sentimiento Global", x=[result['symbol']], y=[result['market_sent']], marker_color="#ff7f0e")
                ])
                fig_sentiment.update_layout(barmode="group", title=f"Análisis de Sentimiento de {result['symbol']}", template="plotly_dark")
                st.plotly_chart(fig_sentiment, use_container_width=True)
                st.write("**NFA (Not Financial Advice):** Esto es solo información educativa, no un consejo financiero. Consulta a un experto antes de invertir.")
            else:
                st.error("El resultado almacenado no es un diccionario válido. Por favor, entrena el modelo nuevamente.")
        else:
            st.info("Entrena el modelo para ver el análisis.")

if __name__ == "__main__":
    main_app()