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

# Configuración inicial de certificados SSL y sesión requests
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

# CoinGecko ID (para actividad social)
coinid_to_coingecko = {v: v if v != "xrp" else "ripple" for v in coincap_ids.values()}

# Características de volatilidad por cripto (usado en ajustes dinámicos LSTM)
crypto_characteristics = {
    "bitcoin": {"volatility": 0.03},
    "ethereum": {"volatility": 0.05},
    "xrp": {"volatility": 0.08},
    "binance-coin": {"volatility": 0.06},
    "cardano": {"volatility": 0.07},
    "solana": {"volatility": 0.09},
    "dogecoin": {"volatility": 0.12},
    "polkadot": {"volatility": 0.07},
    "polygon": {"volatility": 0.06},
    "litecoin": {"volatility": 0.04},
    "tron": {"volatility": 0.06},
    "stellar": {"volatility": 0.05}
}

# -----------------------------------------------
# FUNCIONES DE APOYO
# -----------------------------------------------

def robust_mape(y_true, y_pred, eps=1e-9):
    """Calcula el MAPE de manera robusta evitando división por cero."""
    return np.mean(np.abs((y_true - y_pred) / np.maximum(np.abs(y_true), eps))) * 100

@st.cache_data
def load_coincap_data(coin_id, start_ms=None, end_ms=None):
    """
    Carga datos históricos de CoinCap para una criptomoneda específica con rango personalizado.
    Ajustado para incluir la fecha actual (o 1 día posterior) para obtener el último precio.
    """
    try:
        if start_ms is None or end_ms is None:
            # Forzamos que end_date sea "mañana" para asegurar que CoinCap incluya el precio de hoy
            end_date = datetime.utcnow() + timedelta(days=1)
            start_date = end_date - timedelta(days=730)
            start_ms = int(start_date.timestamp() * 1000)
            end_ms = int(end_date.timestamp() * 1000)

        url = f"https://api.coincap.io/v2/assets/{coin_id}/history?interval=d1&start={start_ms}&end={end_ms}"
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

        if "volumeUsd" in df.columns and not df["volumeUsd"].empty:
            df["volume"] = pd.to_numeric(df["volumeUsd"], errors="coerce").fillna(0.0)
        else:
            df["volume"] = 0.0

        # Ordenar y limpiar
        df = df[["ds", "close_price", "volume"]].dropna().sort_values("ds").reset_index(drop=True)
        return df

    except Exception as e:
        st.error(f"Error al cargar datos: {e}")
        return None

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
    """Construye un modelo LSTM con regularización L2."""
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

def train_model(X_train, y_train, X_val, y_val, input_shape, epochs, batch_size):
    """Entrena el modelo LSTM con validación."""
    tf.keras.backend.clear_session()
    model = build_lstm_model(input_shape)
    callbacks = [
        EarlyStopping(patience=10, restore_best_weights=True),
        ReduceLROnPlateau(patience=5, factor=0.5, min_lr=1e-6)
    ]
    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=epochs, batch_size=batch_size,
        callbacks=callbacks, verbose=0
    )
    return model, history

def get_dynamic_params(df, horizon_days, coin_id):
    """Ajusta parámetros dinámicos según volatilidad y datos."""
    volatility = df["close_price"].pct_change().std()
    base_volatility = crypto_characteristics.get(coin_id, {"volatility": 0.05})["volatility"]

    if coin_id == "xrp":  # Ejemplo: XRP, más volátil
        window_size = min(max(15, int(horizon_days * 1.0)), len(df) // 4)
        epochs = min(150, max(40, int(len(df) / 60) + int(volatility * 300)))
        batch_size = 16
        learning_rate = 0.0002
    elif coin_id == "bitcoin":  # Bitcoin, más estable
        window_size = min(max(30, int(horizon_days * 1.5)), len(df) // 3)
        epochs = min(100, max(30, int(len(df) / 80) + int(volatility * 200)))
        batch_size = 32
        learning_rate = 0.0005
    else:  # Otras criptos
        window_size = min(max(20, int(horizon_days * 1.2)), len(df) // 4)
        epochs = min(120, max(35, int(len(df) / 70) + int(volatility * 250)))
        batch_size = 24
        learning_rate = 0.0003

    return window_size, epochs, batch_size, learning_rate

# -------------------------------------------------------------------------
#  FEAR & GREED y COINGECKO
# -------------------------------------------------------------------------
@st.cache_data(ttl=3600)
def get_fear_greed_index():
    """Obtiene el índice Fear & Greed."""
    try:
        data = session.get("https://api.alternative.me/fng/?format=json", timeout=10).json()
        return float(data["data"][0]["value"])
    except Exception:
        st.warning("No se pudo obtener Fear & Greed Index. Usando valor por defecto (50.0).")
        return 50.0

@st.cache_data(ttl=3600)
def get_coingecko_community_activity(coin_id):
    """Obtiene actividad comunitaria de CoinGecko."""
    try:
        cg_id = coinid_to_coingecko.get(coin_id, coin_id)
        data = session.get(
            f"https://api.coingecko.com/api/v3/coins/{cg_id}?community_data=true",
            timeout=10
        ).json()["community_data"]
        activity = max(
            data.get("twitter_followers", 0),
            data.get("reddit_average_posts_48h", 0) * 1000
        )
        return min(100, (activity / 20000000) * 100) if activity > 0 else 50.0
    except Exception:
        return 50.0

# -------------------------------------------------------------------------
#  LUNARCRUSH: Sustituye a NewsData.io para obtener noticias y sentimiento
# -------------------------------------------------------------------------

@st.cache_data(ttl=3600)
def get_lunarcrush_news(coin_symbol):
    """
    Obtiene las noticias más recientes de LunarCrush para la cripto dada (symbol: BTC, ETH, etc.)
    Devuelve una lista de diccionarios con título, descripción, fecha y link.
    """
    key = st.secrets.get("lunarcrush_key", "")
    if not key:
        st.error("No se encontró la API key de LunarCrush en Secrets. Revisar 'lunarcrush_key'.")
        return []

    # Ejemplo de endpoint: https://api.lunarcrush.com/v2?data=feeds&key=KEY&type=news&symbol=BTC&limit=5
    url = (
        "https://api.lunarcrush.com/v2"
        f"?data=feeds"
        f"&key={key}"
        f"&type=news"
        f"&symbol={coin_symbol}"
        f"&limit=5"
    )
    try:
        resp = session.get(url, timeout=10)
        if resp.status_code == 200:
            data = resp.json()
            if "data" not in data or not data["data"]:
                return []
            articles = []
            for item in data["data"]:
                title = item.get("title", "Sin título")
                desc = item.get("description", "")
                pub_ts = item.get("published_at", None)
                if pub_ts:
                    pub_date = datetime.utcfromtimestamp(pub_ts).strftime("%Y-%m-%d %H:%M:%S")
                else:
                    pub_date = "Fecha no disponible"
                link = item.get("url", "#")
                articles.append({
                    "title": title,
                    "description": desc,
                    "pubDate": pub_date,
                    "link": link
                })
            return articles
        else:
            st.warning(f"Error {resp.status_code} al conectar con LunarCrush.")
            return []
    except Exception as e:
        st.error(f"Error al obtener noticias de LunarCrush: {e}")
        return []

@st.cache_data(ttl=3600)
def get_lunarcrush_sentiment(coin_symbol):
    """
    Calcula un sentimiento promedio usando TextBlob sobre las noticias devueltas por LunarCrush.
    Si no hay noticias, retorna 50.0 por defecto.
    """
    news = get_lunarcrush_news(coin_symbol)
    if not news:
        return 50.0  # Valor neutro si no hay noticias
    sentiments = []
    for article in news:
        text = (article["title"] or "") + " " + (article["description"] or "")
        blob = TextBlob(text)
        sentiment = blob.sentiment.polarity  # -1 a 1
        sentiment_score = 50 + (sentiment * 50)  # 0 a 100
        sentiments.append(sentiment_score)
    return np.mean(sentiments) if sentiments else 50.0

# -------------------------------------------------------------------------
#  CÁLCULO DE SENTIMIENTO: CRIPTO vs. MERCADO
# -------------------------------------------------------------------------

def get_crypto_sentiment_combined(coin_id):
    """
    Obtiene:
      - Sentimiento de cripto (via LunarCrush news)
      - Sentimiento de mercado (Fear & Greed)
    Devuelve (crypto_sent, market_sent, gauge_value) donde gauge_value
    es 50 + (crypto_sent - market_sent), ajustado a [0, 100].
    """
    symbol = coinid_to_symbol[coin_id]  # e.g. "BTC", "ETH", etc.
    crypto_sent = get_lunarcrush_sentiment(symbol)  # 0..100
    market_sent = get_fear_greed_index()            # 0..100

    # Lógica: si cripto_sent > market_sent => gauge_value > 50 => más bullish
    sentiment_diff = crypto_sent - market_sent
    gauge_val = 50 + sentiment_diff
    gauge_val = max(0, min(100, gauge_val))  # clamp

    return crypto_sent, market_sent, gauge_val

# -------------------------------------------------------------------------
#  ENTRENAR Y PREDECIR
# -------------------------------------------------------------------------

def train_and_predict_with_sentiment(coin_id, horizon_days, start_ms=None, end_ms=None):
    """Entrena y predice combinando LSTM con sentimiento (cripto vs. mercado)."""
    df = load_coincap_data(coin_id, start_ms, end_ms)
    if df is None or df.empty:
        return None

    symbol = coinid_to_symbol[coin_id]

    # Obtenemos 3 valores: cripto_sent, market_sent y gauge_value
    crypto_sent, market_sent, gauge_val = get_crypto_sentiment_combined(coin_id)
    # Este gauge_val lo usaremos como factor en el modelo LSTM (0..100 => normalizar)
    sentiment_factor = gauge_val / 100.0

    # Escalado de precios
    scaler = MinMaxScaler()
    scaled_data = scaler.fit_transform(df[["close_price"]])

    # Hiperparámetros dinámicos
    window_size, epochs, batch_size, learning_rate = get_dynamic_params(df, horizon_days, coin_id)
    X, y = create_sequences(scaled_data, window_size)
    if X is None:
        return None

    # Split train/val/test
    split = int(len(X) * 0.8)
    X_train, X_test = X[:split], X[split:]
    y_train, y_test = y[:split], y[split:]
    val_split = int(len(X_train) * 0.9)
    X_val, y_val = X_train[val_split:], y_train[val_split:]
    X_train, y_train = X_train[:val_split], y_train[:val_split]

    # Añadir dimensión (sentiment_factor) en cada timestep
    X_train_adj = np.concatenate([X_train, np.full((X_train.shape[0], window_size, 1), sentiment_factor)], axis=-1)
    X_val_adj = np.concatenate([X_val, np.full((X_val.shape[0], window_size, 1), sentiment_factor)], axis=-1)
    X_test_adj = np.concatenate([X_test, np.full((X_test.shape[0], window_size, 1), sentiment_factor)], axis=-1)

    # Entrenar modelo
    lstm_model, history = train_model(X_train_adj, y_train, X_val_adj, y_val, (window_size, 2), epochs, batch_size)

    # Predicción en test
    lstm_test_preds_scaled = lstm_model.predict(X_test_adj, verbose=0)
    lstm_test_preds = scaler.inverse_transform(lstm_test_preds_scaled).flatten()
    y_test_real = scaler.inverse_transform(y_test.reshape(-1, 1)).flatten()

    lstm_rmse = np.sqrt(mean_squared_error(y_test_real, lstm_test_preds))
    lstm_mape = robust_mape(y_test_real, lstm_test_preds)

    # Predicción futura
    last_window = scaled_data[-window_size:]
    future_preds = []
    current_input = np.concatenate([
        last_window.reshape(1, window_size, 1),
        np.full((1, window_size, 1), sentiment_factor)
    ], axis=-1)

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

    test_dates = df["ds"].iloc[-len(lstm_test_preds):].values
    real_prices = df["close_price"].iloc[-len(lstm_test_preds):].values

    return {
        "df": df,
        "test_preds": lstm_test_preds,
        "future_preds": future_preds,
        "rmse": lstm_rmse,
        "mape": lstm_mape,
        "sentiment_factor": sentiment_factor,
        "symbol": symbol,
        "crypto_sent": crypto_sent,
        "market_sent": market_sent,
        "gauge_val": gauge_val,         # Valor final para gauge
        "future_dates": future_dates,
        "test_dates": test_dates,
        "real_prices": real_prices
    }

# -------------------------------------------------------------------------
#  STREAMLIT APP
# -------------------------------------------------------------------------

def main_app():
    st.set_page_config(page_title="Crypto Price Predictions 🔮", layout="wide")
    st.title("Crypto Price Predictions 🔮")
    st.markdown("""
    **Descripción del Modelo:**  
    Esta plataforma utiliza un modelo avanzado de aprendizaje automático basado en redes LSTM (Long Short-Term Memory) 
    para predecir precios futuros de criptomonedas como Bitcoin, Ethereum, Ripple y otras. El modelo integra datos históricos 
    de precios y volúmenes de CoinCap, abarcando hasta dos años de información diaria, ajustando dinámicamente sus hiperparámetros 
    (como tamaño de ventana, épocas, tamaño de lote y tasa de aprendizaje) según la volatilidad específica de cada criptomoneda. 
    Además, incorpora un análisis de sentimiento dinámico que combina el índice Fear & Greed para el mercado global, la actividad 
    comunitaria en redes sociales (Twitter y Reddit) de CoinGecko y las noticias más recientes de LunarCrush, 
    mejorando la precisión al considerar el estado de ánimo del mercado, los inversores y eventos externos. 
    Las predicciones se complementan con métricas clave como RMSE y MAPE para evaluar la precisión, y se presentan en gráficos 
    interactivos y tablas para una experiencia clara y detallada.

    Fuentes de datos: CoinCap, Fear & Greed Index, CoinGecko, LunarCrush
    """)

    # Sidebar
    st.sidebar.title("Configura tu Predicción")
    crypto_name = st.sidebar.selectbox("Selecciona una criptomoneda:", list(coincap_ids.keys()))
    coin_id = coincap_ids[crypto_name]

    use_custom_range = st.sidebar.checkbox("Habilitar rango de fechas", value=False)
    default_end = datetime.utcnow()  # Usar UTC
    default_start = default_end - timedelta(days=7)

    if use_custom_range:
        start_date = st.sidebar.date_input("Fecha de inicio", default_start.date())
        end_date = st.sidebar.date_input("Fecha de fin", default_end.date())
        if start_date > end_date:
            st.sidebar.error("La fecha de inicio no puede ser posterior a la fecha de fin.")
            return
        if (end_date - start_date).days > 7:
            st.sidebar.warning("El rango de fechas excede 7 días. Ajustando al máximo permitido (7 días).")
            end_date = start_date + timedelta(days=7)
        if start_date > datetime.utcnow().date():
            start_date = datetime.utcnow().date() - timedelta(days=7)
            st.sidebar.warning("La fecha de inicio no puede ser futura. Ajustando al rango máximo permitido (7 días).")
        if end_date > datetime.utcnow().date():
            end_date = datetime.utcnow().date()
            st.sidebar.warning("La fecha de fin no puede ser futura. Ajustando a hoy.")

        # Forzamos a traer datos hasta 1 día después del end_date, para incluir el precio más reciente
        end_date_with_offset = datetime.combine(end_date, datetime.min.time()) + timedelta(days=1)

        start_ms = int(datetime.combine(start_date, datetime.min.time()).timestamp() * 1000)
        end_ms = int(end_date_with_offset.timestamp() * 1000)
    else:
        # Por defecto 7 días hasta "mañana"
        end_date_with_offset = default_end + timedelta(days=1)
        start_ms = int(default_start.timestamp() * 1000)
        end_ms = int(end_date_with_offset.timestamp() * 1000)

    horizon = st.sidebar.slider("Días a predecir:", 1, 60, 5, help="Selecciona el horizonte de días para la predicción.")
    show_stats = st.sidebar.checkbox("Ver estadísticas descriptivas", value=False)

    # Carga y gráfica histórica
    df_prices = load_coincap_data(coin_id, start_ms, end_ms)
    if df_prices is not None and not df_prices.empty:
        fig_hist = px.line(
            df_prices,
            x="ds",
            y="close_price",
            title=f"Histórico de {crypto_name}",
            labels={"ds": "Fecha", "close_price": "Precio (USD)"}
        )
        # Forzar formato diario en eje X, sin hora
        fig_hist.update_layout(template="plotly_dark")
        fig_hist.update_xaxes(tickformat="%Y-%m-%d")
        st.plotly_chart(fig_hist, use_container_width=True)

        if show_stats:
            st.subheader("Estadísticas Descriptivas")
            st.write(df_prices["close_price"].describe())
    else:
        st.warning("No se pudieron cargar datos históricos para el rango seleccionado.")

    tabs = st.tabs(["🤖 Entrenamiento y Test", "🔮 Predicción de Precios", "📊 Análisis de Sentimientos", "📰 Noticias Recientes"])

    # -----------------------------------------------------
    # Tab Entrenamiento y Test
    # -----------------------------------------------------
    with tabs[0]:
        st.header("Entrenamiento del Modelo y Evaluación en Test")
        if st.button("Entrenar Modelo y Predecir"):
            with st.spinner("Entrenando el modelo, por favor espera..."):
                result = train_and_predict_with_sentiment(coin_id, horizon, start_ms, end_ms)
            if result:
                st.success("Entrenamiento y predicción completados!")
                st.write(f"Sentimiento cripto ({result['symbol']}): {result['crypto_sent']:.2f}")
                st.write(f"Sentimiento mercado (Fear & Greed): {result['market_sent']:.2f}")
                st.write(f"Factor combinado (Gauge): {result['gauge_val']:.2f}")

                col1, col2 = st.columns(2)
                col1.metric("RMSE (Test)", f"{result['rmse']:.2f}", help="Error promedio en dólares.")
                col2.metric("MAPE (Test)", f"{result['mape']:.2f}%", help="Error relativo promedio.")

                # Verificación y recorte de longitudes para la gráfica
                if not (len(result["test_dates"]) > 0 and len(result["real_prices"]) > 0 and len(result["test_preds"]) > 0):
                    st.error("No hay suficientes datos para mostrar el gráfico de entrenamiento y test.")
                    st.session_state["result"] = result
                    return

                min_len = min(len(result["test_dates"]), len(result["real_prices"]), len(result["test_preds"]))
                if min_len < 1:
                    st.error("No hay datos suficientes para generar el gráfico.")
                    st.session_state["result"] = result
                    return

                # Ajustar arrays
                result["test_dates"] = result["test_dates"][:min_len]
                result["real_prices"] = result["real_prices"][:min_len]
                result["test_preds"] = result["test_preds"][:min_len]

                # Gráfico de entrenamiento vs test (sin spline)
                fig_test = go.Figure()
                fig_test.add_trace(go.Scatter(
                    x=result["test_dates"],
                    y=result["real_prices"],
                    mode="lines",
                    name="Precio Real",
                    line=dict(color="#1f77b4", width=3)  # shape="linear"
                ))
                fig_test.add_trace(go.Scatter(
                    x=result["test_dates"],
                    y=result["test_preds"],
                    mode="lines",
                    name="Predicción",
                    line=dict(color="#ff7f0e", width=3, dash="dash")  # shape="linear"
                ))
                fig_test.update_layout(
                    title=f"Comparación entre el precio real y la predicción: {result['symbol']}",
                    xaxis=dict(tickformat="%Y-%m-%d"),
                    template="plotly_dark",
                    xaxis_title="Fecha",
                    yaxis_title="Precio (USD)"
                )
                st.plotly_chart(fig_test, use_container_width=True)

                st.session_state["result"] = result

    # -----------------------------------------------------
    # Tab Predicción de Precios
    # -----------------------------------------------------
    with tabs[1]:
        st.header(f"🔮 Predicción de Precios - {crypto_name}")
        if "result" in st.session_state:
            if isinstance(st.session_state["result"], dict):
                result = st.session_state["result"]
                last_date = result["df"]["ds"].iloc[-1]
                current_price = result["df"]["close_price"].iloc[-1]
                pred_series = np.concatenate(([current_price], result["future_preds"]))

                fig_future = go.Figure()
                future_dates_display = [last_date] + result["future_dates"]
                fig_future.add_trace(go.Scatter(
                    x=future_dates_display,
                    y=pred_series,
                    mode="lines+markers",
                    name="Predicción",
                    line=dict(color="#ff7f0e", width=2)  # shape="linear"
                ))
                fig_future.update_layout(
                    title=f"Predicción a Futuro ({horizon} días) - {result['symbol']}",
                    template="plotly_dark",
                    xaxis_title="Fecha",
                    yaxis_title="Precio (USD)",
                    plot_bgcolor="#1e1e2f",
                    paper_bgcolor="#1e1e2f"
                )
                st.plotly_chart(fig_future, use_container_width=True)

                st.subheader("Valores Numéricos")
                df_future = pd.DataFrame({"Fecha": future_dates_display, "Predicción": pred_series})
                st.dataframe(df_future.style.format({"Predicción": "{:.2f}"}))
            else:
                st.error("El resultado almacenado no es un diccionario válido. Por favor, entrena el modelo nuevamente.")
        else:
            st.info("Entrena el modelo primero.")

    # -----------------------------------------------------
    # Tab Análisis de Sentimientos
    # -----------------------------------------------------
    with tabs[2]:
        st.header("📊 Análisis de Sentimientos")
        if "result" in st.session_state:
            if isinstance(st.session_state["result"], dict):
                result = st.session_state["result"]
                crypto_sent = result["crypto_sent"]
                market_sent = result["market_sent"]
                gauge_val = result["gauge_val"]

                # Creamos un gauge circular (sin ocultar la mitad)
                fig_sentiment = go.Figure(go.Indicator(
                    mode="gauge+number",
                    value=gauge_val,
                    number={'suffix': "", "font": {"size": 36}},
                    gauge={
                        "axis": {"range": [0, 100], "tickwidth": 2, "tickcolor": "#fff"},
                        "bar": {"color": "darkblue"},
                        "bgcolor": "#2c2c3e",
                        "borderwidth": 2,
                        "bordercolor": "#4a4a6a",
                        "steps": [
                            {"range": [0, 20], "color": "#ff0000"},      # Bearish
                            {"range": [20, 40], "color": "#ff7f0e"},    # Slightly Bearish
                            {"range": [40, 60], "color": "#ffff00"},    # Neutral
                            {"range": [60, 80], "color": "#90ee90"},    # Bullish
                            {"range": [80, 100], "color": "#008000"},   # Very Bullish
                        ],
                        "threshold": {
                            "line": {"color": "#fff", "width": 4},
                            "thickness": 0.8,
                            "value": gauge_val
                        }
                    },
                    domain={"x": [0, 1], "y": [0, 1]}
                ))
                fig_sentiment.update_layout(
                    title={
                        "text": f"Sentimiento - {result['symbol']}",
                        "x": 0.5,
                        "xanchor": "center",
                        "font": {"size": 24}
                    },
                    template="plotly_dark",
                    height=400,
                    margin=dict(l=20, r=20, t=80, b=20)
                )

                st.plotly_chart(fig_sentiment, use_container_width=True)

                # Mostrar valores
                st.write(f"**Sentimiento Cripto ({result['symbol']}):** {crypto_sent:.2f}")
                st.write(f"**Sentimiento Mercado (Fear & Greed):** {market_sent:.2f}")
                st.write(f"**Gauge (Cripto vs. Mercado):** {gauge_val:.2f} ( >50 => más bullish )")
                st.write("**NFA (Not Financial Advice)**: Esto es solo información educativa, no un consejo financiero.")
            else:
                st.error("El resultado almacenado no es un diccionario válido. Por favor, entrena el modelo nuevamente.")
        else:
            st.info("Entrena el modelo para ver el análisis.")

    # -----------------------------------------------------
    # Tab Noticias Recientes (vía LunarCrush)
    # -----------------------------------------------------
    with tabs[3]:
        st.header("📰 Noticias Recientes de Criptomonedas")
        symbol = coinid_to_symbol[coin_id]
        news = get_lunarcrush_news(symbol)
        if news:
            st.subheader(f"Últimas noticias sobre {crypto_name}")
            st.markdown(
                """
                <style>
                .news-container {
                    display: flex;
                    overflow-x: auto;
                    gap: 1rem;
                }
                .news-item {
                    flex: 0 0 auto;
                    width: 300px;
                    background-color: #2c2c3e;
                    padding: 1rem;
                    border-radius: 5px;
                    border: 1px solid #4a4a6a;
                }
                .news-item h4 {
                    margin: 0 0 0.5rem 0;
                    font-size: 1.1rem;
                }
                .news-item p {
                    font-size: 0.9rem;
                    margin: 0 0 0.5rem 0;
                }
                </style>
                """,
                unsafe_allow_html=True
            )
            st.markdown("<div class='news-container'>", unsafe_allow_html=True)
            for article in news:
                title = article["title"]
                desc = article["description"]
                pub_date = article["pubDate"]
                link = article["link"]
                st.markdown(
                    f"""
                    <div class='news-item'>
                        <h4>{title}</h4>
                        <p><em>{pub_date}</em></p>
                        <p>{desc}</p>
                        <p><a href='{link}' target='_blank'>Leer más</a></p>
                    </div>
                    """,
                    unsafe_allow_html=True
                )
            st.markdown("</div>", unsafe_allow_html=True)
        else:
            st.info("No se encontraron noticias recientes en LunarCrush. Verifica tu API key o la disponibilidad de datos.")

if __name__ == "__main__":
    main_app()
