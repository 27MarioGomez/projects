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

# Verificación de dependencias opcionales
try:
    from statsmodels.tsa.arima.model import ARIMA
    from sklearn.metrics import mean_squared_error
    ARIMA_AVAILABLE = True
except ImportError:
    ARIMA_AVAILABLE = False
    st.warning("Statsmodels no instalado. ARIMA no disponible.")

try:
    from prophet import Prophet
    PROPHET_AVAILABLE = True
except ImportError:
    PROPHET_AVAILABLE = False
    st.warning("Prophet no instalado.")

# Configuración de SSL y sesión de requests
os.environ["REQUESTS_CA_BUNDLE"] = certifi.where()
session = requests.Session()
retry = Retry(total=5, backoff_factor=1, status_forcelist=[429, 500, 502, 503, 504])
adapter = HTTPAdapter(max_retries=retry)
session.mount("https://", adapter)

# Diccionarios
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
    """Calcula MAPE evitando división por cero."""
    return np.mean(np.abs((y_true - y_pred) / np.maximum(np.abs(y_true), eps))) * 100

# Carga de datos
@st.cache_data
def load_coincap_data(coin_id):
    """Carga datos históricos de CoinCap."""
    end_date = datetime.now()
    start_date = end_date - timedelta(days=730)
    url = f"https://api.coincap.io/v2/assets/{coin_id}/history?interval=d1&start={int(start_date.timestamp()*1000)}&end={int(end_date.timestamp()*1000)}"
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
        df["volume"] = pd.to_numeric(df.get("volumeUsd", pd.Series(0.0, index=df.index)), errors="coerce").fillna(0.0)
        return df[["ds", "close_price", "volume"]].dropna().sort_values("ds").reset_index(drop=True)
    except Exception as e:
        st.error(f"Error al cargar datos: {e}")
        return None

# Optimización de hiperparámetros
def get_dynamic_params(df, horizon_days, coin_id):
    """Ajusta parámetros según volatilidad."""
    volatility = df["close_price"].pct_change().std()
    base_volatility = crypto_characteristics.get(coin_id, {"volatility": 0.05})["volatility"]
    window_size = min(max(10, int(horizon_days * 1.5)), len(df) // 4) if volatility > base_volatility else min(max(20, int(horizon_days * 2)), len(df) // 4)
    epochs = min(100, max(50, int(len(df) / 50)))
    batch_size = 16 if volatility > base_volatility else 32
    learning_rate = 0.0003 if volatility > base_volatility else 0.0005
    return window_size, epochs, batch_size, learning_rate

# Creación de secuencias y modelo LSTM
def create_sequences(data, window_size):
    """Crea secuencias para LSTM."""
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
        LSTM(64, return_sequences=True, input_shape=input_shape),
        Dropout(0.2),
        LSTM(64),
        Dropout(0.2),
        Dense(32, activation="relu"),
        Dense(1)
    ])
    model.compile(optimizer=Adam(learning_rate), loss="mse")
    return model

def train_model(X_train, y_train, X_val, y_val, input_shape, epochs, batch_size):
    """Entrena el modelo LSTM."""
    tf.keras.backend.clear_session()
    model = build_lstm_model(input_shape)
    callbacks = [
        EarlyStopping(patience=10, restore_best_weights=True),
        ReduceLROnPlateau(patience=5, factor=0.5, min_lr=1e-6)
    ]
    model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=epochs, batch_size=batch_size, callbacks=callbacks, verbose=0)
    return model

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

def get_crypto_sentiment_combined(coin_id):
    """Calcula el sentimiento combinado con pesos ajustados."""
    fg = get_fear_greed_index()
    cg = get_coingecko_community_activity(coin_id)
    volatility = crypto_characteristics.get(coin_id, {"volatility": 0.05})["volatility"]
    fg_weight = 0.4 if volatility > 0.07 else 0.6
    cg_weight = 1 - fg_weight
    return fg * fg_weight + cg * cg_weight

# Predicción
def train_and_predict_with_sentiment(coin_id, horizon_days):
    """Entrena y predice, devolviendo 10 valores."""
    df = load_coincap_data(coin_id)
    if df is None:
        return None
    symbol = coinid_to_symbol[coin_id]
    crypto_sent = get_crypto_sentiment_combined(coin_id)
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
    lstm_test_preds = scaler.inverse_transform(lstm_test_preds_scaled)
    y_test_real = scaler.inverse_transform(y_test.reshape(-1, 1))
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

    return df, lstm_test_preds, lstm_future_preds, lstm_rmse, lstm_mape, sentiment_factor, symbol, crypto_sent, market_sent, future_dates

# Aplicación principal
def main_app():
    st.set_page_config(page_title="Crypto Price Predictions 🔮", layout="wide")
    st.title("Crypto Price Predictions 🔮")
    st.markdown("Predice precios de criptomonedas con LSTM, datos históricos y análisis de sentimiento.")

    # Sidebar
    st.sidebar.title("Configura tu Predicción")
    crypto_name = st.sidebar.selectbox("Selecciona una criptomoneda:", list(coincap_ids.keys()))
    coin_id = coincap_ids[crypto_name]
    horizon = st.sidebar.slider("Días a predecir:", 1, 60, 5)
    st.sidebar.markdown("**Los hiperparámetros se ajustan automáticamente según los datos.**")
    show_stats = st.sidebar.checkbox("Ver estadísticas descriptivas", value=False)

    # Gráfico histórico
    df_prices = load_coincap_data(coin_id)
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
                result = train_and_predict_with_sentiment(coin_id, horizon)
            if result:
                df, test_preds, future_preds, rmse, mape, sentiment_factor, symbol, crypto_sent, market_sent, future_dates = result
                st.success("Entrenamiento y predicción completados!")
                st.write(f"Sentimiento combinado de {symbol}: {crypto_sent:.2f}")
                st.write(f"Sentimiento global del mercado: {market_sent:.2f}")
                st.write(f"Factor combinado: {sentiment_factor:.2f}")
                col1, col2 = st.columns(2)
                col1.metric("RMSE (Test)", f"{rmse:.2f}", help="Error promedio en dólares.")
                col2.metric("MAPE (Test)", f"{mape:.2f}%", help="Error relativo promedio.")
                fig_test = go.Figure()
                test_dates = df["ds"].iloc[-len(test_preds):]
                fig_test.add_trace(go.Scatter(x=test_dates, y=df["close_price"].iloc[-len(test_preds):], mode="lines", name="Precio (Real)"))
                fig_test.add_trace(go.Scatter(x=test_dates, y=test_preds, mode="lines", name="Predicción (Test)", line=dict(dash="dash")))
                fig_test.update_layout(title=f"Comparación en el set de Test: {symbol}", template="plotly_dark")
                st.plotly_chart(fig_test, use_container_width=True)
                st.session_state["result"] = result

    with tabs[1]:
        st.header(f"Predicción de Precios - {crypto_name}")
        if "result" in st.session_state:
            df, test_preds, future_preds, rmse, mape, sentiment_factor, symbol, crypto_sent, market_sent, future_dates = st.session_state["result"]
            last_date = df["ds"].iloc[-1]
            current_price = df["close_price"].iloc[-1]
            pred_series = np.concatenate(([current_price], future_preds))
            fig_future = go.Figure()
            future_dates_display = [last_date] + future_dates
            fig_future.add_trace(go.Scatter(x=future_dates_display, y=pred_series, mode="lines+markers", name="Predicción"))
            fig_future.update_layout(title=f"Predicción a Futuro ({horizon} días) - {symbol}", template="plotly_dark")
            st.plotly_chart(fig_future, use_container_width=True)
            st.subheader("Valores Numéricos")
            st.dataframe(pd.DataFrame({"Fecha": future_dates_display, "Predicción": pred_series}))
        else:
            st.info("Entrena el modelo primero.")

    with tabs[2]:
        st.header("Análisis de Sentimientos")
        if "result" in st.session_state:
            df, test_preds, future_preds, rmse, mape, sentiment_factor, symbol, crypto_sent, market_sent, future_dates = st.session_state["result"]
            sentiment_texts = {
                "BTC": f"El sentimiento de Bitcoin está en {crypto_sent:.2f}, lo que muestra cierta cautela entre los inversores, aunque su comunidad sigue activa. El mercado en general está en {market_sent:.2f}, indicando miedo. Con un factor combinado de {sentiment_factor:.2f}, parece que Bitcoin podría mantenerse estable, pero no esperes grandes subidas pronto. ¡Ojo con las noticias!",
                "ETH": f"Ethereum tiene un sentimiento de {crypto_sent:.2f}, reflejando dudas, pero su tecnología sigue siendo un punto fuerte. El mercado está en {market_sent:.2f}, con miedo dominando. El factor combinado de {sentiment_factor:.2f} sugiere que podría haber oportunidades si el ánimo mejora.Estate atento a sus actualizaciones.",
                "XRP": f"XRP está en {crypto_sent:.2f}, mostrando pesimismo en su comunidad, y el mercado en {market_sent:.2f} no ayuda mucho. Con un factor combinado de {sentiment_factor:.2f}, parece que XRP podría seguir moviéndose poco a menos que haya noticias grandes, como su caso legal. Cuidado con la volatilidad."
            }
            sentiment_text = sentiment_texts.get(symbol, f"El sentimiento de {symbol} está en {crypto_sent:.2f}, lo que indica {'optimismo' if crypto_sent > 50 else 'pesimismo'} entre sus seguidores. El mercado general está en {market_sent:.2f}. Con un factor combinado de {sentiment_factor:.2f}, hay {'potencial' if sentiment_factor > 0.5 else 'cautela'} a corto plazo.")
            st.write(sentiment_text)
            fig_sentiment = go.Figure(data=[
                go.Bar(name="Sentimiento Combinado", x=[symbol], y=[crypto_sent], marker_color="#1f77b4"),
                go.Bar(name="Sentimiento Global", x=[symbol], y=[market_sent], marker_color="#ff7f0e")
            ])
            fig_sentiment.update_layout(barmode="group", title=f"Análisis de Sentimiento de {symbol}", template="plotly_dark")
            st.plotly_chart(fig_sentiment, use_container_width=True)
            st.write("**NFA (Not Financial Advice):** Esto es solo información educativa, no un consejo financiero. Consulta a un experto antes de invertir.")
        else:
            st.info("Entrena el modelo para ver el análisis.")

if __name__ == "__main__":
    main_app()