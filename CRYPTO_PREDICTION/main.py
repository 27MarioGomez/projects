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
import keras_tuner as kt  # Asegúrate de tener "keras-tuner" en requirements.txt

# ------------------------------------------------------------------------------
# Configuración inicial: certificados SSL y sesión requests
# ------------------------------------------------------------------------------
os.environ["REQUESTS_CA_BUNDLE"] = certifi.where()
session = requests.Session()
retry = Retry(total=5, backoff_factor=1, status_forcelist=[429, 500, 502, 503, 504])
adapter = HTTPAdapter(max_retries=retry)
session.mount("https://", adapter)

# ------------------------------------------------------------------------------
# Diccionarios: Criptomonedas y sus identificadores
# ------------------------------------------------------------------------------
coincap_ids = {
    "Bitcoin (BTC)": "bitcoin", "Ethereum (ETH)": "ethereum", "Ripple (XRP)": "xrp",
    "Binance Coin (BNB)": "binance-coin", "Cardano (ADA)": "cardano", "Solana (SOL)": "solana",
    "Dogecoin (DOGE)": "dogecoin", "Polkadot (DOT)": "polkadot", "Polygon (MATIC)": "polygon",
    "Litecoin (LTC)": "litecoin", "TRON (TRX)": "tron", "Stellar (XLM)": "stellar"
}
coinid_to_symbol = {v: k.split(" (")[1][:-1] for k, v in coincap_ids.items()}
coinid_to_coingecko = {v: v if v != "xrp" else "ripple" for v in coincap_ids.values()}

# Características de volatilidad para cada cripto
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

# ------------------------------------------------------------------------------
# FUNCIONES DE APOYO
# ------------------------------------------------------------------------------
def robust_mape(y_true, y_pred, eps=1e-9):
    """Calcula el MAPE evitando división por cero."""
    return np.mean(np.abs((y_true - y_pred) / np.maximum(np.abs(y_true), eps))) * 100

@st.cache_data
def load_coincap_data(coin_id, start_ms=None, end_ms=None):
    """
    Carga datos históricos desde CoinCap para la cripto dada.
    Se fuerza a incluir datos hasta 1 día después para obtener el precio más actual.
    Eliminamos el volumen, ya que no se usa en el modelo.
    """
    try:
        if start_ms is None or end_ms is None:
            end_date = datetime.utcnow() + timedelta(days=1)  # Hasta mañana
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
        df = df[["ds", "close_price"]].dropna().sort_values("ds").reset_index(drop=True)
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

# ------------------------------------------------------------------------------
# MODELOS: LSTM y Ajuste dinámico de hiperparámetros
# ------------------------------------------------------------------------------
def build_lstm_model(input_shape, learning_rate=0.001, l2_lambda=0.01, lstm_units1=100, lstm_units2=80, dropout_rate=0.3, dense_units=50):
    """
    Construye un modelo LSTM con regularización y dropout.
    Parametrizamos las capas LSTM para que se ajusten según la volatilidad u otra lógica.
    """
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

def train_model(X_train, y_train, X_val, y_val, model, epochs, batch_size):
    """
    Entrena el modelo LSTM con callbacks para optimización.
    Recibimos 'model' ya construido (posiblemente ajustado dinámicamente).
    """
    tf.keras.backend.clear_session()
    callbacks = [
        EarlyStopping(patience=10, restore_best_weights=True),
        ReduceLROnPlateau(patience=5, factor=0.5, min_lr=1e-6)
    ]
    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=epochs,
        batch_size=batch_size,
        callbacks=callbacks,
        verbose=0
    )
    return model, history

def get_dynamic_params(df, horizon_days, coin_id):
    """
    Ajusta parámetros del modelo en función de la volatilidad y datos.
    Devuelve un diccionario con hyperparams que iremos afinando.
    """
    # Extraemos la volatilidad real de los datos
    real_volatility = df["close_price"].pct_change().std()
    base_volatility = crypto_characteristics.get(coin_id, {"volatility": 0.05})["volatility"]

    # Lógica extra: combinamos la volatilidad "definida" y la "real"
    combined_volatility = (real_volatility + base_volatility) / 2.0

    # Ajustamos la ventana y epochs en base al horizon y la volatilidad
    # De manera más flexible que antes
    window_size = int(max(15, min(60, horizon_days * (1.0 + combined_volatility * 5))))
    epochs = int(max(30, min(200, (len(df) / 70) + combined_volatility * 300)))
    batch_size = int(max(16, min(64, (combined_volatility * 500) + 16)))

    # Ajustamos param LSTM
    # Ejemplo: si hay más volatilidad => más neuronas
    # Nota: Límite 200 para no exagerar
    lstm_units1 = int(max(50, min(200, 100 + (combined_volatility * 400))))
    lstm_units2 = int(max(30, min(150, 80 + (combined_volatility * 200))))
    dropout_rate = 0.3 if combined_volatility < 0.1 else 0.4
    dense_units = int(max(30, min(100, 50 + (combined_volatility * 100))))

    # Learning rate
    # Menor LR si la volatilidad es alta
    learning_rate = 0.0005 if combined_volatility < 0.08 else 0.0002

    # L2 regularization
    l2_lambda = 0.01 if combined_volatility < 0.07 else 0.02

    return {
        "window_size": window_size,
        "epochs": epochs,
        "batch_size": batch_size,
        "lstm_units1": lstm_units1,
        "lstm_units2": lstm_units2,
        "dropout_rate": dropout_rate,
        "dense_units": dense_units,
        "learning_rate": learning_rate,
        "l2_lambda": l2_lambda
    }

# ------------------------------------------------------------------------------
# FEAR & GREED y COINGECKO
# ------------------------------------------------------------------------------
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

# ------------------------------------------------------------------------------
# LUNARCRUSH (Noticias y Sentimiento)
# ------------------------------------------------------------------------------
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
    Retorna 50.0 si no hay noticias o si ocurre un error.
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
    Obtiene el sentimiento de la cripto (LunarCrush) y del mercado (Fear & Greed),
    y calcula un valor de gauge:
        gauge_val = 50 + (crypto_sent - market_sent), forzando a [0, 100].
    """
    symbol = coinid_to_symbol[coin_id]
    crypto_sent = get_lunarcrush_sentiment(symbol)
    market_sent = get_fear_greed_index()
    gauge_val = 50 + (crypto_sent - market_sent)
    gauge_val = max(0, min(100, gauge_val))
    return crypto_sent, market_sent, gauge_val

# ------------------------------------------------------------------------------
# KERAS TUNER
# ------------------------------------------------------------------------------
def build_model_tuner(input_shape):
    """
    Función modelo para Keras Tuner. Ajusta hiperparámetros LSTM.
    """
    def model_builder(hp):
        lstm_units1 = hp.Int('lstm_units1', min_value=50, max_value=200, step=50, default=100)
        lstm_units2 = hp.Int('lstm_units2', min_value=30, max_value=150, step=30, default=60)
        dropout_rate = hp.Float('dropout_rate', min_value=0.2, max_value=0.5, step=0.1, default=0.3)
        dense_units = hp.Int('dense_units', min_value=30, max_value=100, step=10, default=50)
        learning_rate = hp.Float('learning_rate', min_value=1e-4, max_value=1e-3, sampling='log', default=1e-4)
        l2_lambda = hp.Float('l2_lambda', min_value=1e-4, max_value=2e-2, sampling='log', default=1e-2)

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

# ------------------------------------------------------------------------------
# ENTRENAMIENTO Y PREDICCIÓN
# ------------------------------------------------------------------------------
def train_and_predict_with_sentiment(coin_id, horizon_days, start_ms=None, end_ms=None, tune=False):
    """
    Entrena el modelo LSTM y realiza predicciones futuras, integrando
    el factor de sentimiento (gauge_val/100) en cada timestep.
    Si tune==True, se optimizan los hiperparámetros con Keras Tuner.
    """
    df = load_coincap_data(coin_id, start_ms, end_ms)
    if df is None or df.empty:
        return None

    symbol = coinid_to_symbol[coin_id]
    crypto_sent, market_sent, gauge_val = get_crypto_sentiment_combined(coin_id)
    sentiment_factor = gauge_val / 100.0

    # Ajuste dinámico (nuestro "mini-autoML" manual)
    params = get_dynamic_params(df, horizon_days, coin_id)
    window_size = params["window_size"]
    epochs = params["epochs"]
    batch_size = params["batch_size"]
    lstm_units1 = params["lstm_units1"]
    lstm_units2 = params["lstm_units2"]
    dropout_rate = params["dropout_rate"]
    dense_units = params["dense_units"]
    learning_rate = params["learning_rate"]
    l2_lambda = params["l2_lambda"]

    scaler = MinMaxScaler()
    scaled_data = scaler.fit_transform(df[["close_price"]])
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
        # Búsqueda con Keras Tuner
        tuner = kt.RandomSearch(
            build_model_tuner(input_shape),
            objective='val_loss',
            max_trials=3,   # Ajusta si quieres más o menos
            executions_per_trial=1,
            directory='kt_dir',
            project_name='crypto_prediction'
        )
        tuner.search(X_train_adj, y_train, validation_data=(X_val_adj, y_val), epochs=20, batch_size=batch_size, verbose=0)
        lstm_model = tuner.get_best_models(num_models=1)[0]
    else:
        # Creamos el modelo con parámetros dinámicos
        lstm_model = build_lstm_model(
            input_shape=input_shape,
            learning_rate=learning_rate,
            l2_lambda=l2_lambda,
            lstm_units1=lstm_units1,
            lstm_units2=lstm_units2,
            dropout_rate=dropout_rate,
            dense_units=dense_units
        )
        lstm_model, history = train_model(X_train_adj, y_train, X_val_adj, y_val, lstm_model, epochs, batch_size)

    # Predicción en Test
    lstm_test_preds_scaled = lstm_model.predict(X_test_adj, verbose=0)
    lstm_test_preds = scaler.inverse_transform(lstm_test_preds_scaled).flatten()
    y_test_real = scaler.inverse_transform(y_test.reshape(-1, 1)).flatten()
    lstm_rmse = np.sqrt(mean_squared_error(y_test_real, lstm_test_preds))
    lstm_mape = robust_mape(y_test_real, lstm_test_preds)

    # Predicción Futura
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

# ------------------------------------------------------------------------------
# STREAMLIT APP
# ------------------------------------------------------------------------------
def main_app():
    st.set_page_config(page_title="Crypto Price Predictions 🔮", layout="wide")
    st.title("Crypto Price Predictions 🔮")
    st.markdown("""
    **Descripción del Modelo:**  
    Esta plataforma utiliza un modelo avanzado de aprendizaje automático basado en redes LSTM (Long Short-Term Memory) 
    para predecir precios futuros de criptomonedas como Bitcoin, Ethereum, Ripple y otras. Integra datos históricos 
    de CoinCap y ajusta dinámicamente sus hiperparámetros según la volatilidad.  
    Además, combina el índice Fear & Greed, la actividad de CoinGecko y las noticias de LunarCrush para estimar el sentimiento.  
    Las predicciones se evalúan mediante RMSE y MAPE, y se muestran en gráficos interactivos.
    """)

    # Sidebar
    st.sidebar.title("Configura tu Predicción")
    crypto_name = st.sidebar.selectbox("Selecciona una criptomoneda:", list(coincap_ids.keys()))
    coin_id = coincap_ids[crypto_name]
    use_custom_range = st.sidebar.checkbox("Habilitar rango de fechas", value=False)
    default_end = datetime.utcnow()
    default_start = default_end - timedelta(days=7)

    if use_custom_range:
        start_date = st.sidebar.date_input("Fecha de inicio", default_start.date())
        end_date = st.sidebar.date_input("Fecha de fin", default_end.date())
        if start_date > end_date:
            st.sidebar.error("La fecha de inicio no puede ser posterior a la fecha de fin.")
            return
        if (end_date - start_date).days > 7:
            st.sidebar.warning("El rango excede 7 días. Se ajustará al máximo (7).")
            end_date = start_date + timedelta(days=7)
        if start_date > datetime.utcnow().date():
            start_date = datetime.utcnow().date() - timedelta(days=7)
            st.sidebar.warning("La fecha de inicio no puede ser futura. Se ajusta a 7 días atrás.")
        if end_date > datetime.utcnow().date():
            end_date = datetime.utcnow().date()
            st.sidebar.warning("La fecha de fin no puede ser futura. Se ajusta a hoy.")
        end_date_with_offset = datetime.combine(end_date, datetime.min.time()) + timedelta(days=1)
        start_ms = int(datetime.combine(start_date, datetime.min.time()).timestamp() * 1000)
        end_ms = int(end_date_with_offset.timestamp() * 1000)
    else:
        end_date_with_offset = default_end + timedelta(days=1)
        start_ms = int(default_start.timestamp() * 1000)
        end_ms = int(end_date_with_offset.timestamp() * 1000)

    optimize_hp = st.sidebar.checkbox("Optimizar hiperparámetros con Keras Tuner", value=False)
    horizon = st.sidebar.slider("Días a predecir:", 1, 60, 5)
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
        fig_hist.update_layout(template="plotly_dark")
        fig_hist.update_xaxes(tickformat="%Y-%m-%d")
        st.plotly_chart(fig_hist, use_container_width=True)
        if show_stats:
            st.subheader("Estadísticas Descriptivas")
            st.write(df_prices["close_price"].describe())
    else:
        st.warning("No se pudieron cargar datos históricos para el rango seleccionado.")

    # Tabs
    tabs = st.tabs(["🤖 Entrenamiento y Test", "🔮 Predicción de Precios", "📊 Análisis de Sentimientos", "📰 Noticias Recientes"])

    # --------------------------------------------------------------------------
    # Tab 1: Entrenamiento y Test
    # --------------------------------------------------------------------------
    with tabs[0]:
        st.header("Entrenamiento del Modelo y Evaluación en Test")
        if st.button("Entrenar Modelo y Predecir"):
            with st.spinner("Entrenando el modelo..."):
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
                    st.error("No hay datos suficientes para la gráfica de Test.")
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
                    line=dict(color="#1f77b4", width=3, shape="spline")
                ))
                fig_test.add_trace(go.Scatter(
                    x=result["test_dates"],
                    y=result["test_preds"],
                    mode="lines",
                    name="Predicción",
                    line=dict(color="#ff7f0e", width=3, dash="dash", shape="spline")
                ))
                fig_test.update_layout(
                    title=f"Comparación: Precio Real vs. Predicción ({result['symbol']})",
                    xaxis=dict(tickformat="%Y-%m-%d"),
                    template="plotly_dark",
                    xaxis_title="Fecha",
                    yaxis_title="Precio (USD)"
                )
                st.plotly_chart(fig_test, use_container_width=True)
                st.session_state["result"] = result

    # --------------------------------------------------------------------------
    # Tab 2: Predicción de Precios
    # --------------------------------------------------------------------------
    with tabs[1]:
        st.header(f"🔮 Predicción de Precios - {crypto_name}")
        if "result" in st.session_state and isinstance(st.session_state["result"], dict):
            result = st.session_state["result"]
            if result is not None:
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
                    line=dict(color="#ff7f0e", width=2, shape="spline")
                ))
                fig_future.update_layout(
                    title=f"Predicción Futura ({horizon} días) - {result['symbol']}",
                    template="plotly_dark",
                    xaxis_title="Fecha",
                    yaxis_title="Precio (USD)"
                )
                st.plotly_chart(fig_future, use_container_width=True)

                st.subheader("Valores Numéricos")
                df_future = pd.DataFrame({"Fecha": future_dates_display, "Predicción": pred_series})
                st.dataframe(df_future.style.format({"Predicción": "{:.2f}"}))
            else:
                st.info("No se obtuvo resultado. Entrena el modelo primero.")
        else:
            st.info("Entrena el modelo primero.")

    # --------------------------------------------------------------------------
    # Tab 3: Análisis de Sentimientos
    # --------------------------------------------------------------------------
    with tabs[2]:
        st.header("📊 Análisis de Sentimientos")
        if "result" in st.session_state:
            if isinstance(st.session_state["result"], dict):
                result = st.session_state["result"]
                if result is None:
                    st.info("No se obtuvo resultado. Entrena el modelo primero.")
                elif "gauge_val" not in result:
                    st.warning("El resultado no contiene 'gauge_val'. Reentrena con la versión actual.")
                else:
                    crypto_sent = result["crypto_sent"]
                    market_sent = result["market_sent"]
                    gauge_val   = result["gauge_val"]

                    # Texto dinámico para un público no técnico
                    # Dependiendo de gauge_val, clasificamos en:
                    # 0-20 Muy Bearish, 20-40 Bearish, 40-60 Neutral, 60-80 Bullish, 80-100 Muy Bullish
                    if gauge_val < 20:
                        gauge_text = "Muy Bearish"
                    elif gauge_val < 40:
                        gauge_text = "Bearish"
                    elif gauge_val < 60:
                        gauge_text = "Neutral"
                    elif gauge_val < 80:
                        gauge_text = "Bullish"
                    else:
                        gauge_text = "Muy Bullish"

                    # Creamos el gauge
                    fig_sentiment = go.Figure(go.Indicator(
                        mode="gauge+number",
                        value=gauge_val,
                        number={'suffix': "", "font": {"size": 36}},
                        gauge={
                            "axis": {"range": [0, 100], "tickwidth": 2, "tickcolor": "#fff"},
                            # Cambiamos la barra a un color claro para resaltar
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
                    st.write(f"**Gauge Value:** {gauge_val:.2f} → **{gauge_text}**")
                    if gauge_val > 50:
                        st.write("**Tendencia:** El sentimiento de esta cripto está por encima del mercado. Es posible un escenario bullish.")
                    else:
                        st.write("**Tendencia:** El sentimiento de esta cripto está por debajo o igual al mercado. Prudencia recomendada.")
            else:
                st.error("El resultado almacenado no es válido. Entrena el modelo nuevamente.")
        else:
            st.info("Entrena el modelo para ver el análisis de sentimientos.")

    # --------------------------------------------------------------------------
    # Tab 4: Noticias Recientes (vía LunarCrush)
    # --------------------------------------------------------------------------
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
            st.warning("No se encontraron noticias recientes en LunarCrush o hubo un error.")

# ------------------------------------------------------------------------------
# MAIN
# ------------------------------------------------------------------------------
if __name__ == "__main__":
    main_app()
