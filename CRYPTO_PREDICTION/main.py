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
import keras_tuner as kt  # <-- Asegúrate de tener keras-tuner instalado

# Configuración inicial: certificados SSL y sesión requests
os.environ["REQUESTS_CA_BUNDLE"] = certifi.where()
session = requests.Session()
retry = Retry(total=5, backoff_factor=1, status_forcelist=[429, 500, 502, 503, 504])
adapter = HTTPAdapter(max_retries=retry)
session.mount("https://", adapter)

# Diccionarios: Criptomonedas y sus identificadores
coincap_ids = {
    "Bitcoin (BTC)": "bitcoin", "Ethereum (ETH)": "ethereum", "Ripple (XRP)": "xrp",
    "Binance Coin (BNB)": "binance-coin", "Cardano (ADA)": "cardano", "Solana (SOL)": "solana",
    "Dogecoin (DOGE)": "dogecoin", "Polkadot (DOT)": "polkadot", "Polygon (MATIC)": "polygon",
    "Litecoin (LTC)": "litecoin", "TRON (TRX)": "tron", "Stellar (XLM)": "stellar"
}
coinid_to_symbol = {v: k.split(" (")[1][:-1] for k, v in coincap_ids.items()}
coinid_to_coingecko = {v: v if v != "xrp" else "ripple" for v in coincap_ids.values()}

# Características de volatilidad para cada cripto (para ajustar el modelo LSTM)
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

# -------------------------------
# FUNCIONES DE APOYO
# -------------------------------
def robust_mape(y_true, y_pred, eps=1e-9):
    """Calcula el MAPE evitando división por cero."""
    return np.mean(np.abs((y_true - y_pred) / np.maximum(np.abs(y_true), eps))) * 100

@st.cache_data
def load_coincap_data(coin_id, start_ms=None, end_ms=None):
    """
    Carga datos históricos desde CoinCap para la cripto dada.
    Se fuerza a incluir datos hasta 1 día después para obtener el precio más actual.
    """
    try:
        if start_ms is None or end_ms is None:
            # Para incluir el precio de hoy, forzamos end_date a mañana (UTC)
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
        df["volume"] = pd.to_numeric(df.get("volumeUsd", 0), errors="coerce").fillna(0.0)
        df = df[["ds", "close_price", "volume"]].dropna().sort_values("ds").reset_index(drop=True)
        return df

    except Exception as e:
        st.error(f"Error al cargar datos: {e}")
        return None

def create_sequences(data, window_size):
    """Genera secuencias para el modelo LSTM."""
    if len(data) <= window_size:
        return None, None
    X, y = [], []
    for i in range(window_size, len(data)):
        X.append(data[i - window_size:i])
        y.append(data[i, 0])
    return np.array(X), np.array(y)

def build_lstm_model(input_shape, learning_rate=0.001, l2_lambda=0.01):
    """Construye un modelo LSTM con regularización y dropout."""
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
    """Entrena el modelo LSTM con callbacks para optimización."""
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
    """Ajusta parámetros del modelo en función de la volatilidad y cantidad de datos."""
    volatility = df["close_price"].pct_change().std()
    if coin_id == "xrp":
        window_size = min(max(15, int(horizon_days * 1.0)), len(df) // 4)
        epochs = min(150, max(40, int(len(df) / 60) + int(volatility * 300)))
        batch_size = 16
        learning_rate = 0.0002
    elif coin_id == "bitcoin":
        window_size = min(max(30, int(horizon_days * 1.5)), len(df) // 3)
        epochs = min(100, max(30, int(len(df) / 80) + int(volatility * 200)))
        batch_size = 32
        learning_rate = 0.0005
    else:
        window_size = min(max(20, int(horizon_days * 1.2)), len(df) // 4)
        epochs = min(120, max(35, int(len(df) / 70) + int(volatility * 250)))
        batch_size = 24
        learning_rate = 0.0003

    return window_size, epochs, batch_size, learning_rate

# -------------------------------
# INTEGRACIÓN CON FEAR & GREED y COINGECKO
# -------------------------------
@st.cache_data(ttl=3600)
def get_fear_greed_index():
    """Obtiene el índice Fear & Greed del mercado."""
    try:
        data = session.get("https://api.alternative.me/fng/?format=json", timeout=10).json()
        return float(data["data"][0]["value"])
    except Exception:
        st.warning("No se pudo obtener Fear & Greed Index. Usando 50.0 por defecto.")
        return 50.0

@st.cache_data(ttl=3600)
def get_coingecko_community_activity(coin_id):
    """Obtiene actividad comunitaria desde CoinGecko."""
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

# -------------------------------
# INTEGRACIÓN CON LUNARCRUSH (Noticias y Sentimiento)
# -------------------------------
@st.cache_data(ttl=3600)
def get_lunarcrush_news(coin_symbol):
    """
    Obtiene las noticias más recientes de LunarCrush para la cripto dada.
    Devuelve una lista de artículos con título, descripción, fecha y link.
    """
    key = st.secrets.get("lunarcrush_key", "")
    if not key:
        st.error("No se encontró la API key de LunarCrush en Secrets ('lunarcrush_key').")
        return []

    url = (
        "https://api.lunarcrush.com/v2"
        f"?data=feeds&key={key}&type=news&symbol={coin_symbol}&limit=5"
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
                pub_date = datetime.utcfromtimestamp(pub_ts).strftime("%Y-%m-%d %H:%M:%S") if pub_ts else "Fecha no disponible"
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
    Calcula un sentimiento promedio a partir de las noticias de LunarCrush.
    Retorna 50.0 si no hay noticias.
    """
    news = get_lunarcrush_news(coin_symbol)
    if not news:
        return 50.0
    sentiments = []
    for article in news:
        text = (article["title"] or "") + " " + (article["description"] or "")
        blob = TextBlob(text)
        sentiment = blob.sentiment.polarity  # Valor entre -1 y 1
        sentiment_score = 50 + (sentiment * 50)  # Normalizado a 0-100
        sentiments.append(sentiment_score)
    return np.mean(sentiments) if sentiments else 50.0

def get_crypto_sentiment_combined(coin_id):
    """
    Obtiene el sentimiento propio de la cripto (de LunarCrush) y el sentimiento del mercado
    (Fear & Greed), y calcula un valor de gauge:
        gauge_val = 50 + (crypto_sent - market_sent)
    Se ajusta a [0, 100].
    """
    symbol = coinid_to_symbol[coin_id]
    crypto_sent = get_lunarcrush_sentiment(symbol)
    market_sent = get_fear_greed_index()
    gauge_val = 50 + (crypto_sent - market_sent)
    gauge_val = max(0, min(100, gauge_val))
    return crypto_sent, market_sent, gauge_val

# -------------------------------
# INTEGRACIÓN CON KERAS TUNER
# -------------------------------
def build_model_tuner(input_shape):
    """Función modelo para Keras Tuner. Permite ajustar hiperparámetros del LSTM."""
    def model_builder(hp):
        lstm_units1 = hp.Int('lstm_units1', min_value=50, max_value=150, step=25, default=100)
        lstm_units2 = hp.Int('lstm_units2', min_value=30, max_value=100, step=10, default=80)
        dropout_rate = hp.Float('dropout_rate', min_value=0.2, max_value=0.5, step=0.1, default=0.3)
        dense_units = hp.Int('dense_units', min_value=30, max_value=70, step=10, default=50)
        learning_rate = hp.Float('learning_rate', min_value=1e-4, max_value=1e-2, sampling='log', default=1e-3)
        l2_lambda = hp.Float('l2_lambda', min_value=1e-4, max_value=1e-2, sampling='log', default=1e-2)

        model = Sequential([
            LSTM(lstm_units1, return_sequences=True, input_shape=input_shape, kernel_regularizer=l2(l2_lambda)),
            Dropout(dropout_rate),
            LSTM(lstm_units2, kernel_regularizer=l2(l2_lambda)),
            Dropout(dropout_rate),
            Dense(dense_units, activation='relu', kernel_regularizer=l2(l2_lambda)),
            Dense(1)
        ])
        model.compile(optimizer=Adam(learning_rate), loss='mse')
        return model
    return model_builder

# -------------------------------
# ENTRENAMIENTO Y PREDICCIÓN
# -------------------------------
def train_and_predict_with_sentiment(coin_id, horizon_days, start_ms=None, end_ms=None, tune=False):
    """
    Entrena el modelo LSTM y realiza predicciones futuras, integrando
    el factor de sentimiento (gauge_val/100) en cada timestep.
    Si tune==True, se optimizan los hiperparámetros con Keras Tuner.
    Retorna un diccionario con métricas y datos para gráficas.
    """
    df = load_coincap_data(coin_id, start_ms, end_ms)
    if df is None or df.empty:
        return None

    symbol = coinid_to_symbol[coin_id]
    crypto_sent, market_sent, gauge_val = get_crypto_sentiment_combined(coin_id)
    sentiment_factor = gauge_val / 100.0

    scaler = MinMaxScaler()
    scaled_data = scaler.fit_transform(df[["close_price"]])
    window_size, epochs, batch_size, default_lr = get_dynamic_params(df, horizon_days, coin_id)
    X, y = create_sequences(scaled_data, window_size)
    if X is None:
        return None

    # División de datos: train, validation y test
    split = int(len(X) * 0.8)
    X_train, X_test = X[:split], X[split:]
    y_train, y_test = y[:split], y[split:]
    val_split = int(len(X_train) * 0.9)
    X_val, y_val = X_train[val_split:], y_train[val_split:]
    X_train, y_train = X_train[:val_split], y_train[:val_split]

    # Añadir el factor de sentimiento a cada timestep
    X_train_adj = np.concatenate([X_train, np.full((X_train.shape[0], window_size, 1), sentiment_factor)], axis=-1)
    X_val_adj   = np.concatenate([X_val,   np.full((X_val.shape[0], window_size, 1), sentiment_factor)], axis=-1)
    X_test_adj  = np.concatenate([X_test,  np.full((X_test.shape[0], window_size, 1), sentiment_factor)], axis=-1)

    input_shape = (window_size, 2)
    if tune:
        tuner = kt.RandomSearch(
            build_model_tuner(input_shape),
            objective='val_loss',
            max_trials=5,
            executions_per_trial=1,
            directory='kt_dir',
            project_name='crypto_prediction'
        )
        tuner.search(X_train_adj, y_train, validation_data=(X_val_adj, y_val), epochs=20, batch_size=batch_size, verbose=0)
        lstm_model = tuner.get_best_models(num_models=1)[0]
    else:
        lstm_model, history = train_model(X_train_adj, y_train, X_val_adj, y_val, input_shape, epochs, batch_size)

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
        "gauge_val": gauge_val,
        "future_dates": future_dates,
        "test_dates": test_dates,
        "real_prices": real_prices
    }

# -------------------------------
# STREAMLIT APP
# -------------------------------
def main_app():
    st.set_page_config(page_title="Crypto Price Predictions 🔮", layout="wide")
    st.title("Crypto Price Predictions 🔮")
    st.markdown("""
    **Descripción del Modelo:**  
    Esta plataforma utiliza un modelo avanzado de aprendizaje automático basado en redes LSTM (Long Short-Term Memory) 
    para predecir precios futuros de criptomonedas como Bitcoin, Ethereum, Ripple y otras. Se integran datos históricos 
    de CoinCap (hasta dos años) y se ajustan dinámicamente los hiperparámetros según la volatilidad de cada cripto.  
    Además, se incorpora un análisis de sentimiento que combina el índice Fear & Greed, la actividad comunitaria de CoinGecko  
    y noticias de LunarCrush. Las predicciones se evalúan mediante RMSE y MAPE, y se presentan en gráficos interactivos.
    Fuentes de datos: CoinCap, Fear & Greed Index, CoinGecko, LunarCrush
    """)

    # Sidebar: Configuración de la predicción
    st.sidebar.title("Configura tu Predicción")
    crypto_name = st.sidebar.selectbox("Selecciona una criptomoneda:", list(coincap_ids.keys()))
    coin_id = coincap_ids[crypto_name]
    use_custom_range = st.sidebar.checkbox("Habilitar rango de fechas", value=False)
    default_end = datetime.utcnow()  # Usamos UTC
    default_start = default_end - timedelta(days=7)

    if use_custom_range:
        start_date = st.sidebar.date_input("Fecha de inicio", default_start.date())
        end_date = st.sidebar.date_input("Fecha de fin", default_end.date())
        if start_date > end_date:
            st.sidebar.error("La fecha de inicio no puede ser posterior a la fecha de fin.")
            return
        if (end_date - start_date).days > 7:
            st.sidebar.warning("El rango de fechas excede 7 días. Se ajustará al máximo permitido (7 días).")
            end_date = start_date + timedelta(days=7)
        if start_date > datetime.utcnow().date():
            start_date = datetime.utcnow().date() - timedelta(days=7)
            st.sidebar.warning("La fecha de inicio no puede ser futura. Se ajustará al rango máximo permitido (7 días).")
        if end_date > datetime.utcnow().date():
            end_date = datetime.utcnow().date()
            st.sidebar.warning("La fecha de fin no puede ser futura. Se ajustará a hoy.")
        # Para incluir el precio más actual, se suma 1 día al end_date
        end_date_with_offset = datetime.combine(end_date, datetime.min.time()) + timedelta(days=1)
        start_ms = int(datetime.combine(start_date, datetime.min.time()).timestamp() * 1000)
        end_ms = int(end_date_with_offset.timestamp() * 1000)
    else:
        end_date_with_offset = default_end + timedelta(days=1)
        start_ms = int(default_start.timestamp() * 1000)
        end_ms = int(end_date_with_offset.timestamp() * 1000)

    # Opción para optimizar hiperparámetros con Keras Tuner
    optimize_hp = st.sidebar.checkbox("Optimizar hiperparámetros", value=False)
    horizon = st.sidebar.slider("Días a predecir:", 1, 60, 5, help="Selecciona el horizonte para la predicción.")
    show_stats = st.sidebar.checkbox("Ver estadísticas descriptivas", value=False)

    # Gráfico Histórico de CoinCap
    df_prices = load_coincap_data(coin_id, start_ms, end_ms)
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

    tabs = st.tabs(["🤖 Entrenamiento y Test", "🔮 Predicción de Precios", "📊 Análisis de Sentimientos", "📰 Noticias Recientes"])

    # Tab: Entrenamiento y Test
    with tabs[0]:
        st.header("Entrenamiento del Modelo y Evaluación en Test")
        if st.button("Entrenar Modelo y Predecir"):
            with st.spinner("Entrenando el modelo, por favor espera..."):
                result = train_and_predict_with_sentiment(coin_id, horizon, start_ms, end_ms, tune=optimize_hp)
            if result:
                st.success("Entrenamiento y predicción completados!")
                st.write(f"Sentimiento cripto ({result['symbol']}): {result['crypto_sent']:.2f}")
                st.write(f"Sentimiento mercado (Fear & Greed): {result['market_sent']:.2f}")
                st.write(f"Factor combinado (Gauge): {result['gauge_val']:.2f}")
                col1, col2 = st.columns(2)
                col1.metric("RMSE (Test)", f"{result['rmse']:.2f}", help="Error promedio en USD.")
                col2.metric("MAPE (Test)", f"{result['mape']:.2f}%", help="Error relativo promedio.")
                if not (len(result["test_dates"]) > 0 and len(result["real_prices"]) > 0 and len(result["test_preds"]) > 0):
                    st.error("No hay suficientes datos para mostrar el gráfico de Test.")
                    st.session_state["result"] = result
                    return
                min_len = min(len(result["test_dates"]), len(result["real_prices"]), len(result["test_preds"]))
                result["test_dates"] = result["test_dates"][:min_len]
                result["real_prices"] = result["real_prices"][:min_len]
                result["test_preds"] = result["test_preds"][:min_len]
                fig_test = go.Figure()
                fig_test.add_trace(go.Scatter(
                    x=result["test_dates"],
                    y=result["real_prices"],
                    mode="lines",
                    name="Precio Real",
                    line=dict(color="#1f77b4", width=3)
                ))
                fig_test.add_trace(go.Scatter(
                    x=result["test_dates"],
                    y=result["test_preds"],
                    mode="lines",
                    name="Predicción",
                    line=dict(color="#ff7f0e", width=3, dash="dash")
                ))
                fig_test.update_layout(
                    title=f"Comparación: Precio Real vs Predicción ({result['symbol']})",
                    xaxis=dict(tickformat="%Y-%m-%d"),
                    template="plotly_dark",
                    xaxis_title="Fecha",
                    yaxis_title="Precio (USD)"
                )
                st.plotly_chart(fig_test, use_container_width=True)
                st.session_state["result"] = result

    # Tab: Predicción de Precios
    with tabs[1]:
        st.header(f"🔮 Predicción de Precios - {crypto_name}")
        if "result" in st.session_state and isinstance(st.session_state["result"], dict):
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
                line=dict(color="#ff7f0e", width=2)
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
            st.info("Entrena el modelo primero.")

    # Tab: Análisis de Sentimientos
    with tabs[2]:
        st.header("📊 Análisis de Sentimientos")
        if "result" in st.session_state:
            if isinstance(st.session_state["result"], dict):
                result = st.session_state["result"]
                if result is None:
                    st.info("No se obtuvo resultado. Por favor, entrena el modelo.")
                elif "gauge_val" not in result:
                    st.warning("El resultado no contiene 'gauge_val'. Vuelve a entrenar el modelo con la versión actualizada.")
                else:
                    crypto_sent = result["crypto_sent"]
                    market_sent = result["market_sent"]
                    gauge_val   = result["gauge_val"]
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
                                {"range": [0, 20], "color": "#ff0000"},
                                {"range": [20, 40], "color": "#ff7f0e"},
                                {"range": [40, 60], "color": "#ffff00"},
                                {"range": [60, 80], "color": "#90ee90"},
                                {"range": [80, 100], "color": "#008000"}
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
                    st.write(f"**Sentimiento Cripto ({result['symbol']}):** {crypto_sent:.2f}")
                    st.write(f"**Sentimiento Mercado (Fear & Greed):** {market_sent:.2f}")
                    st.write(f"**Gauge (Cripto vs. Mercado):** {gauge_val:.2f} (valores >50 indican tendencia bullish)")
                    st.write("**NFA (Not Financial Advice):** Esta información es educativa y no constituye consejo financiero.")
            else:
                st.error("El resultado almacenado no es válido. Por favor, entrena el modelo nuevamente.")
        else:
            st.info("Entrena el modelo para ver el análisis de sentimientos.")

    # Tab: Noticias Recientes (vía LunarCrush)
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
