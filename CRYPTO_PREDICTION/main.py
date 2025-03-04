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
import socket
from urllib3.util.retry import Retry
from requests.adapters import HTTPAdapter

# Configura certificados SSL para solicitudes HTTPS seguras
os.environ["REQUESTS_CA_BUNDLE"] = certifi.where()
session = requests.Session()
retry_strategy = Retry(total=5, backoff_factor=1, status_forcelist=[429, 500, 502, 503, 504])
adapter = HTTPAdapter(max_retries=retry_strategy)
session.mount("https://", adapter)

# Define diccionarios de mapeo para criptomonedas y características de volatilidad
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

# Calcula el MAPE de manera robusta, evitando divisiones por cero
def robust_mape(y_true, y_pred, eps=1e-9):
    return np.mean(np.abs((y_true - y_pred) / np.maximum(np.abs(y_true), eps))) * 100

# Carga datos históricos diarios de CoinCap para una criptomoneda específica
@st.cache_data
def load_coincap_data(coin_id, start_ms=None, end_ms=None):
    if start_ms is None or end_ms is None:
        end_date = datetime.now()
        start_date = end_date - timedelta(days=730)
        start_ms, end_ms = int(start_date.timestamp() * 1000), int(end_date.timestamp() * 1000)
    
    url = f"https://api.coincap.io/v2/assets/{coin_id}/history?interval=d1&start={start_ms}&end={end_ms}"
    try:
        resp = session.get(url, headers={"User-Agent": "Mozilla/5.0"}, verify=certifi.where(), timeout=10)
        if resp.status_code != 200:
            st.warning(f"CoinCap: Error {resp.status_code}")
            return None
        df = pd.DataFrame(resp.json().get("data", []))
        if df.empty or "time" not in df or "priceUsd" not in df:
            st.warning("CoinCap: Datos inválidos o vacíos")
            return None
        df["ds"] = pd.to_datetime(df["time"], unit="ms", errors="coerce")
        df["close_price"] = pd.to_numeric(df["priceUsd"], errors="coerce")
        if "volumeUsd" in df and not df["volumeUsd"].empty:
            df["volume"] = pd.to_numeric(df["volumeUsd"], errors="coerce").fillna(0.0)
        else:
            df["volume"] = pd.Series(0.0, index=df.index)
        return df[["ds", "close_price", "volume"]].dropna().sort_values("ds").reset_index(drop=True)
    except Exception as e:
        st.error(f"Error al cargar datos de CoinCap: {e}")
        return None

# Prepara secuencias de datos para el modelo LSTM
def create_sequences(data, window_size):
    if len(data) <= window_size:
        return None, None
    X, y = [], []
    for i in range(window_size, len(data)):
        X.append(data[i - window_size:i])
        y.append(data[i, 0])
    return np.array(X), np.array(y)

# Construye un modelo LSTM mejorado con regularización L2
def build_lstm_model(input_shape, learning_rate=0.001, l2_lambda=0.01):
    model = Sequential([
        LSTM(100, return_sequences=True, input_shape=input_shape, kernel_regularizer=l2(l2_lambda)),
        Dropout(0.3),
        LSTM(80, kernel_regularizer=l2(l2_lambda)),
        Dropout(0.3),
        Dense(50, activation="relu", kernel_regularizer=l2(l2_lambda)),
        Dense(1)
    ])
    model.compile(optimizer=Adam(learning_rate), loss="mse")
    return model

# Entrena el modelo LSTM con validación y callbacks
def train_model(X_train, y_train, X_val, y_val, input_shape, epochs, batch_size):
    tf.keras.backend.clear_session()
    model = build_lstm_model(input_shape)
    callbacks = [EarlyStopping(patience=10, restore_best_weights=True), ReduceLROnPlateau(patience=5, factor=0.5, min_lr=1e-6)]
    history = model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=epochs, batch_size=batch_size, callbacks=callbacks, verbose=0)
    return model, history

# Ajusta parámetros dinámicos según la volatilidad y el horizonte
def get_dynamic_params(df, horizon_days, coin_id):
    volatility = df["close_price"].pct_change().std()
    base_volatility = crypto_characteristics.get(coin_id, {"volatility": 0.05})["volatility"]
    if coin_id == "xrp":
        window_size = min(max(15, int(horizon_days * 1.0)), len(df) // 4)
        epochs = min(150, max(40, int(len(df) / 60) + int(volatility * 300)))
        batch_size, learning_rate = 16, 0.0002
    elif coin_id == "bitcoin":
        window_size = min(max(30, int(horizon_days * 1.5)), len(df) // 3)
        epochs = min(100, max(30, int(len(df) / 80) + int(volatility * 200)))
        batch_size, learning_rate = 32, 0.0005
    else:
        window_size = min(max(20, int(horizon_days * 1.2)), len(df) // 4)
        epochs = min(120, max(35, int(len(df) / 70) + int(volatility * 250)))
        batch_size, learning_rate = 24, 0.0003
    return window_size, epochs, batch_size, learning_rate

# Obtiene el índice Fear & Greed para el mercado global
@st.cache_data(ttl=3600)
def get_fear_greed_index():
    try:
        return float(session.get("https://api.alternative.me/fng/?format=json", timeout=10).json()["data"][0]["value"])
    except Exception:
        st.warning("No se pudo obtener Fear & Greed Index. Usando valor por defecto 50.0")
        return 50.0

# Obtiene actividad comunitaria desde CoinGecko
@st.cache_data(ttl=3600)
def get_coingecko_community_activity(coin_id):
    try:
        cg_id = coinid_to_coingecko.get(coin_id, coin_id)
        data = session.get(f"https://api.coingecko.com/api/v3/coins/{cg_id}?community_data=true", timeout=10).json()["community_data"]
        activity = max(data.get("twitter_followers", 0), data.get("reddit_average_posts_48h", 0) * 1000)
        return min(100, (activity / 20000000) * 100) if activity > 0 else 50.0
    except Exception:
        return 50.0

# Combina sentimiento de múltiples fuentes ajustado por volatilidad
def get_crypto_sentiment_combined(coin_id, news_sentiment=None):
    fg = get_fear_greed_index()
    cg = get_coingecko_community_activity(coin_id)
    volatility = crypto_characteristics.get(coin_id, {"volatility": 0.05})["volatility"]
    if volatility > 0.07:  # Criptos volátiles (XRP, DOGE)
        fg_weight, cg_weight, news_weight = 0.15, 0.45, 0.40
    else:  # Criptos estables (BTC, ETH)
        fg_weight, cg_weight, news_weight = 0.50, 0.30, 0.20
    news_sent = 50.0 if news_sentiment is None or pd.isna(news_sentiment) else float(news_sentiment)
    combined = (fg * fg_weight + cg * cg_weight + news_sent * news_weight)
    return max(0, min(100, combined))

# Obtiene y analiza sentimiento de noticias usando NewsData.io
@st.cache_data(ttl=86400)
def get_news_sentiment(coin_symbol, start_date=None, end_date=None):
    if start_date is None or end_date is None:
        end_date, start_date = datetime.now().date(), end_date - timedelta(days=7)
    if (end_date - start_date).days > 7:
        start_date = end_date - timedelta(days=7)
    if start_date > datetime.now().date() or end_date > datetime.now().date():
        start_date, end_date = datetime.now().date() - timedelta(days=7), datetime.now().date()
    
    api_key = st.secrets.get("news_data_key", "pub_7227626d8277642d9399e67d37a74d463f7cc")
    if not api_key:
        st.error("API key de NewsData.io no encontrada en Secrets. Usando valor por defecto.")
        return 50.0
    
    query = f"{coin_symbol} AND (price OR market OR regulation)"
    url = f"https://newsdata.io/api/1/news?apikey={api_key}&q={requests.utils.quote(query)}&language=en&from_date={start_date.strftime('%Y-%m-%d')}&to_date={end_date.strftime('%Y-%m-%d')}&size=5&category=crypto"
    
    try:
        try:
            socket.getaddrinfo('newsdata.io', 443)
        except socket.gaierror as dns_error:
            st.error(f"Error DNS para newsdata.io: {dns_error}. Usando valor por defecto.")
            return 50.0
        
        resp = session.get(url, timeout=10, headers={"User-Agent": "Mozilla/5.0"})
        if resp.status_code == 200:
            data = resp.json()
            articles = data.get("results", [])
            if not articles:
                return 50.0
            sentiments = []
            for article in articles[:5]:
                text = (article.get("title", "") + " " + article.get("description", "")).strip()
                if text and any(k in text.lower() for k in ["price", "market", "regulation"]):
                    sentiment = TextBlob(text).sentiment.polarity
                    sentiments.append(50 + (sentiment * 50))
            return np.mean(sentiments) if sentiments else 50.0
        elif resp.status_code in [422, 429, 401]:
            st.error(f"Error NewsData.io {resp.status_code}: {resp.text}. Usando valor por defecto.")
            return 50.0
        else:
            st.warning(f"Error NewsData.io {resp.status_code}. Usando valor por defecto 50.0")
            return 50.0
    except requests.exceptions.ConnectionError as conn_error:
        st.error(f"Error de conexión con NewsData.io: {conn_error}. Usando valor por defecto.")
        return 50.0
    except Exception as e:
        st.error(f"Error inesperado en NewsData.io: {e}. Usando valor por defecto.")
        return 50.0

# Obtiene noticias recientes de criptomonedas
@st.cache_data(ttl=3600)
def get_recent_crypto_news(coin_symbol):
    end_date, start_date = datetime.now().date(), end_date - timedelta(days=14)
    api_key = st.secrets.get("news_data_key", "pub_7227626d8277642d9399e67d37a74d463f7cc")
    if not api_key:
        st.error("API key de NewsData.io no encontrada. No se pueden mostrar noticias.")
        return []
    
    query = f"{coin_symbol} AND crypto"
    url = f"https://newsdata.io/api/1/news?apikey={api_key}&q={requests.utils.quote(query)}&language=en&from_date={start_date.strftime('%Y-%m-%d')}&to_date={end_date.strftime('%Y-%m-%d')}&size=10&category=crypto&sort_by=pubDate"
    
    try:
        try:
            socket.getaddrinfo('newsdata.io', 443)
        except socket.gaierror as dns_error:
            st.error(f"Error DNS para newsdata.io: {dns_error}. No se pueden mostrar noticias.")
            return []
        
        resp = session.get(url, timeout=10, headers={"User-Agent": "Mozilla/5.0"})
        if resp.status_code == 200:
            data = resp.json()
            articles = data.get("results", [])
            if not articles:
                query_simple, start_date = "crypto", end_date - timedelta(days=7)
                url_retry = f"https://newsdata.io/api/1/news?apikey={api_key}&q={requests.utils.quote(query_simple)}&language=en&from_date={start_date.strftime('%Y-%m-%d')}&to_date={end_date.strftime('%Y-%m-%d')}&size=10&category=crypto&sort_by=pubDate"
                resp_retry = session.get(url_retry, timeout=10, headers={"User-Agent": "Mozilla/5.0"})
                if resp_retry.status_code == 200:
                    articles = sorted(resp_retry.json().get("results", []), key=lambda x: x.get("pubDate", ""), reverse=True)[:5]
                    if articles:
                        return [{"title": a.get("title", "Sin título"), "description": a.get("description", "Sin descripción"), 
                                 "pubDate": a.get("pubDate", "Fecha no disponible"), "link": a.get("link", "#")} for a in articles]
                st.warning("No se encontraron noticias específicas. Consulta genérica sin resultados.")
                return []
            return [{"title": a.get("title", "Sin título"), "description": a.get("description", "Sin descripción"), 
                     "pubDate": a.get("pubDate", "Fecha no disponible"), "link": a.get("link", "#")} for a in sorted(articles, key=lambda x: x.get("pubDate", ""), reverse=True)[:5]]
        elif resp.status_code in [422, 429, 401]:
            st.error(f"Error NewsData.io {resp.status_code}: {resp.text}. No se pueden mostrar noticias.")
            return []
        else:
            st.warning(f"Error NewsData.io {resp.status_code}. No se pueden mostrar noticias.")
            return []
    except requests.exceptions.ConnectionError as conn_error:
        st.error(f"Error de conexión con NewsData.io: {conn_error}. No se pueden mostrar noticias.")
        return []
    except Exception as e:
        st.error(f"Error inesperado en NewsData.io: {e}. No se pueden mostrar noticias.")
        return []

# Realiza predicciones usando LSTM con integración de sentimiento
def train_and_predict_with_sentiment(coin_id, horizon_days, start_ms=None, end_ms=None):
    df = load_coincap_data(coin_id, start_ms, end_ms)
    if df is None:
        return None
    symbol = coinid_to_symbol[coin_id]
    start_date = datetime.fromtimestamp(start_ms / 1000) if start_ms else datetime.now() - timedelta(days=7)
    end_date = datetime.fromtimestamp(end_ms / 1000) if end_ms else datetime.now()
    news_sent = get_news_sentiment(symbol, start_date.date(), end_date.date()) or 50.0
    crypto_sent = get_crypto_sentiment_combined(coin_id, news_sent)
    market_sent = get_fear_greed_index()
    sentiment_factor = (crypto_sent + market_sent) / 200.0

    scaler = MinMaxScaler()
    scaled_data = scaler.fit_transform(df[["close_price"]])
    window_size, epochs, batch_size, learning_rate = get_dynamic_params(df, horizon_days, coin_id)
    X, y = create_sequences(scaled_data, window_size)
    if X is None:
        return None

    split, val_split = int(len(X) * 0.8), int(len(X[:split]) * 0.9)
    X_train, X_test = X[:split], X[split:]
    y_train, y_test = y[:split], y[split:]
    X_val, y_val = X_train[val_split:], y_train[val_split:]
    X_train, y_train = X_train[:val_split], y_train[:val_split]

    X_train_adj = np.concatenate([X_train, np.full((len(X_train), window_size, 1), sentiment_factor)], axis=-1)
    X_val_adj = np.concatenate([X_val, np.full((len(X_val), window_size, 1), sentiment_factor)], axis=-1)
    X_test_adj = np.concatenate([X_test, np.full((len(X_test), window_size, 1), sentiment_factor)], axis=-1)

    model, _ = train_model(X_train_adj, y_train, X_val_adj, y_val, (window_size, 2), epochs, batch_size)
    test_preds_scaled = model.predict(X_test_adj, verbose=0).flatten()
    test_preds = scaler.inverse_transform(test_preds_scaled.reshape(-1, 1)).flatten()
    y_test_real = scaler.inverse_transform(y_test.reshape(-1, 1)).flatten()
    rmse, mape = np.sqrt(mean_squared_error(y_test_real, test_preds)), robust_mape(y_test_real, test_preds)

    last_window = scaled_data[-window_size:]
    future_preds = []
    current_input = np.concatenate([last_window.reshape(1, window_size, 1), np.full((1, window_size, 1), sentiment_factor)], axis=-1)
    for _ in range(horizon_days):
        pred = model.predict(current_input, verbose=0)[0][0]
        future_preds.append(pred)
        current_input = np.append(current_input[:, 1:, :], [[[pred, sentiment_factor]]], axis=1)
    future_preds = scaler.inverse_transform(np.array(future_preds).reshape(-1, 1)).flatten()

    last_date = df["ds"].iloc[-1]
    future_dates = pd.date_range(start=last_date + timedelta(days=1), periods=horizon_days).tolist()
    test_dates, real_prices = df["ds"].iloc[-len(test_preds):].values, df["close_price"].iloc[-len(test_preds):].values

    return {"df": df, "test_preds": test_preds, "future_preds": future_preds, "rmse": rmse, "mape": mape,
            "sentiment_factor": sentiment_factor, "symbol": symbol, "crypto_sent": crypto_sent, "market_sent": market_sent,
            "future_dates": future_dates, "test_dates": test_dates, "real_prices": real_prices}

# Configura y ejecuta la aplicación de predicción de precios de criptomonedas
def main_app():
    st.set_page_config(page_title="Crypto Price Predictions 🔮", layout="wide")
    st.title("Crypto Price Predictions 🔮")
    st.markdown("""
    **Descripción del Modelo:**  
    Esta plataforma utiliza un modelo avanzado de aprendizaje automático basado en redes LSTM (Long Short-Term Memory) para predecir precios futuros de criptomonedas como Bitcoin, Ethereum, Ripple y otras. El modelo integra datos históricos de precios y volúmenes de CoinCap, abarcando hasta dos años de información diaria, ajustando dinámicamente sus hiperparámetros según la volatilidad específica de cada criptomoneda. Además, incorpora un análisis de sentimiento dinámico que combina el índice Fear & Greed para el mercado global, la actividad comunitaria en redes sociales (Twitter y Reddit) de CoinGecko, y noticias específicas de criptomonedas obtenidas a través de NewsData.io, mejorando la precisión al considerar el estado de ánimo del mercado, los inversores y eventos externos. Las predicciones se complementan con métricas clave como RMSE y MAPE para evaluar la precisión, y se presentan en gráficos interactivos y tablas para una experiencia clara y detallada.

    Fuentes de datos: CoinCap, Fear & Greed Index, CoinGecko, NewsData.io
    """)

    st.sidebar.title("Configura tu Predicción")
    crypto_name = st.sidebar.selectbox("Selecciona una criptomoneda:", list(coincap_ids.keys()))
    coin_id = coincap_ids[crypto_name]
    use_custom_range = st.sidebar.checkbox("Habilitar rango de fechas", value=False)
    default_end, default_start = datetime.now(), default_end - timedelta(days=7)
    if use_custom_range:
        start_date = st.sidebar.date_input("Fecha de inicio", default_start.date())
        end_date = st.sidebar.date_input("Fecha de fin", default_end.date())
        if start_date > end_date:
            st.sidebar.error("La fecha de inicio no puede ser posterior a la fecha de fin.")
            return
        if (end_date - start_date).days > 7:
            st.sidebar.warning("Rango excede 7 días. Ajustando al máximo permitido.")
            end_date = start_date + timedelta(days=7)
        if start_date > datetime.now().date() or end_date > datetime.now().date():
            start_date, end_date = datetime.now().date() - timedelta(days=7), datetime.now().date()
            st.sidebar.warning("Fechas futuras ajustadas al rango máximo permitido.")
        start_ms, end_ms = int(datetime.combine(start_date, datetime.min.time()).timestamp() * 1000), int(datetime.combine(end_date, datetime.min.time()).timestamp() * 1000)
    else:
        start_ms, end_ms = int(default_start.timestamp() * 1000), int(default_end.timestamp() * 1000)
    horizon = st.sidebar.slider("Días a predecir:", 1, 60, 5)
    st.sidebar.markdown("**Los hiperparámetros se ajustan dinámicamente según datos y volatilidad.**")
    show_stats = st.sidebar.checkbox("Ver estadísticas descriptivas", value=False)

    df_prices = load_coincap_data(coin_id, start_ms, end_ms)
    if df_prices is not None:
        fig_hist = px.line(df_prices, x="ds", y="close_price", title=f"Histórico de Precios - {crypto_name}", labels={"ds": "Fecha", "close_price": "Precio en USD"})
        fig_hist.update_layout(template="plotly_dark")
        st.plotly_chart(fig_hist, use_container_width=True)
        if show_stats:
            st.subheader("Estadísticas Descriptivas de Precios")
            st.write(df_prices["close_price"].describe())

    tabs = st.tabs(["🤖 Entrenamiento y Test", "🔮 Predicción de Precios", "📊 Análisis de Sentimientos", "📰 Noticias Recientes"])
    with tabs[0]:
        st.header("Entrenamiento del Modelo y Evaluación en Test")
        if st.button("Entrenar Modelo y Predecir"):
            with st.spinner("Procesando predicciones, esto puede tardar..."):
                result = train_and_predict_with_sentiment(coin_id, horizon, start_ms, end_ms)
            if result:
                st.success("Entrenamiento y predicción completados con éxito")
                st.write(f"Sentimiento combinado de {result['symbol']}: {result['crypto_sent']:.2f}")
                st.write(f"Sentimiento global del mercado: {result['market_sent']:.2f}")
                st.write(f"Factor de sentimiento combinado: {result['sentiment_factor']:.2f}")
                col1, col2 = st.columns(2)
                col1.metric("RMSE (Test)", f"{result['rmse']:.2f}", help="Error promedio en dólares")
                col2.metric("MAPE (Test)", f"{result['mape']:.2f}%", help="Error relativo promedio en porcentaje")

                if not (len(result["test_dates"]) > 0 and len(result["real_prices"]) > 0 and len(result["test_preds"]) > 0):
                    st.error("Datos insuficientes para mostrar gráfico de comparación.")
                    return
                min_len = min(len(result["test_dates"]), len(result["real_prices"]), len(result["test_preds"]))
                if min_len < 1:
                    st.error("No hay datos suficientes para generar el gráfico.")
                    return
                result["test_dates"], result["real_prices"], result["test_preds"] = result["test_dates"][:min_len], result["real_prices"][:min_len], result["test_preds"][:min_len]

                fig_test = go.Figure()
                fig_test.add_trace(go.Scatter(x=result["test_dates"], y=result["real_prices"], mode="lines", name="Precio Real", line=dict(color="#1f77b4", width=3, shape="spline")))
                fig_test.add_trace(go.Scatter(x=result["test_dates"], y=result["test_preds"], mode="lines", name="Predicción", line=dict(color="#ff7f0e", width=3, dash="dash", shape="spline")))
                fig_test.update_layout(title=f"Comparación Precio Real vs Predicción - {result['symbol']}", xaxis=dict(tickformat="%Y-%m-%d"), template="plotly_dark")
                st.plotly_chart(fig_test, use_container_width=True)
                st.session_state["result"] = result

    with tabs[1]:
        st.header(f"🔮 Predicción de Precios - {crypto_name}")
        if "result" in st.session_state and isinstance(st.session_state["result"], dict):
            result = st.session_state["result"]
            last_date, current_price = result["df"]["ds"].iloc[-1], result["df"]["close_price"].iloc[-1]
            pred_series = np.concatenate([np.array([current_price]), result["future_preds"]])
            future_dates_display = [last_date] + result["future_dates"]
            fig_future = go.Figure(go.Scatter(x=future_dates_display, y=pred_series, mode="lines+markers", name="Predicción", line=dict(color="#ff7f0e", width=2, shape="spline")))
            fig_future.update_layout(title=f"Predicción Futura ({horizon} días) - {result['symbol']}", template="plotly_dark", xaxis_title="Fecha", yaxis_title="Precio en USD", plot_bgcolor="#1e1e2f", paper_bgcolor="#1e1e2f")
            st.plotly_chart(fig_future, use_container_width=True)
            st.subheader("Valores Numéricos de Predicción")
            st.dataframe(pd.DataFrame({"Fecha": future_dates_display, "Predicción": pred_series}).style.format({"Predicción": "{:.2f}"}))
        else:
            st.info("Entrena el modelo primero para ver predicciones.")

    with tabs[2]:
        st.header("📊 Análisis de Sentimientos")
        if "result" in st.session_state and isinstance(st.session_state["result"], dict):
            result = st.session_state["result"]
            crypto_sent, market_sent = result["crypto_sent"], result["market_sent"]
            level = (crypto_sent - 50) / 5
            sentiment_label = "Very Bearish" if level <= -5 else "Bearish" if level <= -2 else "Neutral" if -2 < level < 2 else "Bullish" if level <= 5 else "Very Bullish"
            color = "#ff0000" if level <= -2 else "#ffd700" if -2 < level < 2 else "#00ff00" if level <= 5 else "#008000"

            fig_sentiment = go.Figure(go.Indicator(mode="gauge+number", value=crypto_sent, domain={"x": [0, 1], "y": [0, 1]}, title={"text": f"Sentimiento - {result['symbol']}", "font": {"size": 24, "color": "#ffffff", "family": "Arial, sans-serif"}}, gauge={"axis": {"range": [0, 100], "tickvals": [0, 25, 50, 75, 100], "ticktext": ["Very Bearish", "Bearish", "Neutral", "Bullish", "Very Bullish"], "tickcolor": "#ffffff", "tickwidth": 2, "tickfont": {"size": 16, "color": "#ffffff"}}, "bar": {"color": color}, "bgcolor": "#2c2c3e", "borderwidth": 2, "bordercolor": "#4a4a6a", "steps": [{"range": [0, 25], "color": "#ff0000"}, {"range": [25, 50], "color": "#ffd700"}, {"range": [50, 75], "color": "#00ff00"}, {"range": [75, 100], "color": "#008000"}], "threshold": {"line": {"color": "#ffffff", "width": 4}, "thickness": 1, "value": crypto_sent}}, number={"font": {"size": 48, "color": "#ffffff", "family": "Arial, sans-serif"}}))
            fig_sentiment.update_layout(template="plotly_dark", plot_bgcolor="#1e1e2f", paper_bgcolor="#1e1e2f", height=600, width=900, margin=dict(l=20, r=20, t=80, b=20))
            st.plotly_chart(fig_sentiment, use_container_width=True)
            st.write(f"**Estado:** {sentiment_label} (Mercado: {market_sent:.2f})")
            st.write("**NFA (Not Financial Advice):** Esto es solo información educativa, no un consejo financiero. Consulta a un experto antes de invertir.")
        else:
            st.info("Entrena el modelo para ver el análisis de sentimientos.")

    with tabs[3]:
        st.header("📰 Noticias Recientes de Criptomonedas")
        news = get_recent_crypto_news(coinid_to_symbol[coin_id])
        if news:
            st.subheader(f"Últimas 5 Noticias sobre {crypto_name}")
            for article in news:
                with st.expander(f"**{article['title']}** - {article['pubDate']}", expanded=False):
                    st.write(article['description'])
                    if article['link']:
                        st.markdown(f"[Leer más]({article['link']})", unsafe_allow_html=True)
            news_df = pd.DataFrame(news)
            st.dataframe(news_df[["title", "pubDate"]].style.format({"pubDate": "{:%Y-%m-%d %H:%M:%S}"}).set_properties(**{'background-color': '#2c2c3e', 'color': 'white', 'border-color': '#4a4a6a'}))
        else:
            st.info("No se encontraron noticias recientes. Verifica conexión, límites de API (200 créditos/día), o clave en Secrets.")

if __name__ == "__main__":
    main_app()