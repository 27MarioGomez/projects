import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import requests
from datetime import datetime, timedelta
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import TimeSeriesSplit
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from sklearn.metrics import mean_squared_error
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
import time
import certifi
import os

# Configurar certificados SSL y sesión de requests
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
    """Calcula MAPE de forma robusta evitando división por cero."""
    return np.mean(np.abs((y_true - y_pred) / np.maximum(np.abs(y_true), eps))) * 100

# Carga de datos desde CoinCap
@st.cache_data
def load_coincap_data(coin_id):
    """Carga datos históricos de CoinCap para una criptomoneda."""
    end_date = datetime.now()
    start_date = end_date - timedelta(days=730)  # 2 años de datos
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

# Optimización dinámica de hiperparámetros
def get_dynamic_params(df, horizon_days, coin_id):
    """Ajusta parámetros dinámicamente según volatilidad y datos."""
    volatility = df["close_price"].pct_change().std()
    base_volatility = crypto_characteristics.get(coin_id, {"volatility": 0.05})["volatility"]
    
    window_size = min(max(10, int(horizon_days * (2 if volatility > base_volatility else 1.5))), len(df) // 3)
    epochs = min(100, max(30, int(len(df) / 50) + int(volatility * 200)))
    batch_size = 16 if volatility > base_volatility else 32
    learning_rate = 0.0002 if volatility > base_volatility else 0.0005
    
    return window_size, epochs, batch_size, learning_rate

# Creación de secuencias y modelo LSTM
def create_sequences(data, window_size):
    """Crea secuencias para el modelo LSTM."""
    if len(data) <= window_size:
        return None, None
    X, y = [], []
    for i in range(window_size, len(data)):
        X.append(data[i - window_size:i])
        y.append(data[i, 0])  # Precio de cierre como objetivo
    return np.array(X), np.array(y)

def build_lstm_model(input_shape, learning_rate=0.001):
    """Construye un modelo LSTM simplificado."""
    model = Sequential([
        LSTM(32, return_sequences=True, input_shape=input_shape),
        Dropout(0.3),
        LSTM(32),
        Dropout(0.3),
        Dense(16, activation="relu"),
        Dense(1)
    ])
    model.compile(optimizer=Adam(learning_rate=learning_rate), loss="mse")
    return model

def train_model(X_train, y_train, X_val, y_val, input_shape, epochs, batch_size):
    """Entrena el modelo LSTM con validación."""
    model = build_lstm_model(input_shape)
    callbacks = [
        EarlyStopping(patience=5, restore_best_weights=True),
        ReduceLROnPlateau(patience=3, factor=0.5, min_lr=1e-6)
    ]
    model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=epochs, batch_size=batch_size, callbacks=callbacks, verbose=0)
    return model

# Funciones de sentimiento
@st.cache_data(ttl=3600)
def get_fear_greed_index():
    """Obtiene el índice Fear & Greed."""
    try:
        return float(session.get("https://api.alternative.me/fng/?format=json", timeout=10).json()["data"][0]["value"])
    except Exception:
        st.warning("No se pudo obtener Fear & Greed Index. Usando valor por defecto.")
        return 50.0

@st.cache_data(ttl=3600)
def get_coingecko_community_activity(coin_id):
    """Obtiene actividad comunitaria desde CoinGecko."""
    try:
        cg_id = coinid_to_coingecko.get(coin_id, coin_id)
        data = session.get(f"https://api.coingecko.com/api/v3/coins/{cg_id}?community_data=true", timeout=10).json()["community_data"]
        activity = max(data.get("twitter_followers", 0), data.get("reddit_average_posts_48h", 0) * 1000)
        return min(100, (activity / 20000000) * 100) if activity > 0 else 50.0
    except Exception:
        st.warning(f"No se pudo obtener actividad de CoinGecko para {coin_id}. Usando valor por defecto.")
        return 50.0

def get_crypto_sentiment_combined(coin_id):
    """Combina sentimiento cripto-específico y global."""
    fg = get_fear_greed_index()
    cg = get_coingecko_community_activity(coin_id)
    volatility = crypto_characteristics.get(coin_id, {"volatility": 0.05})["volatility"]
    fg_weight = 0.6 if volatility > 0.07 else 0.5
    cg_weight = 1 - fg_weight
    return fg * fg_weight + cg * cg_weight

# Entrenamiento y predicción
def train_and_predict_with_sentiment(coin_id, horizon_days):
    """Entrena el modelo y realiza predicciones con ajuste de sentimiento."""
    df = load_coincap_data(coin_id)
    if df is None:
        return None
    symbol = coinid_to_symbol[coin_id]
    crypto_sent = get_crypto_sentiment_combined(coin_id)
    market_sent = get_fear_greed_index()
    sentiment_factor = (crypto_sent + market_sent) / 200.0

    # Preparar datos con precio y volumen
    data_for_model = df[["close_price", "volume"]].values
    scaler = MinMaxScaler()
    scaled_data = scaler.fit_transform(data_for_model)

    window_size, epochs, batch_size, learning_rate = get_dynamic_params(df, horizon_days, coin_id)
    X, y = create_sequences(scaled_data, window_size)
    if X is None:
        return None

    # División de datos
    split = int(len(X) * 0.8)
    X_train, X_test = X[:split], X[split:]
    y_train, y_test = y[:split], y[split:]
    val_split = int(len(X_train) * 0.9)
    X_val, y_val = X_train[val_split:], y_train[val_split:]
    X_train, y_train = X_train[:val_split], y_train[:val_split]

    # Entrenar modelo
    model = train_model(X_train, y_train, X_val, y_val, (window_size, 2), epochs, batch_size)

    # Predicciones en test
    test_preds_scaled = model.predict(X_test, verbose=0)
    test_preds = scaler.inverse_transform(np.concatenate([test_preds_scaled, np.zeros((len(test_preds_scaled), 1))], axis=1))[:, 0]
    y_test_real = scaler.inverse_transform(np.concatenate([y_test.reshape(-1, 1), np.zeros((len(y_test), 1))], axis=1))[:, 0]
    rmse = np.sqrt(mean_squared_error(y_test_real, test_preds))
    mape = robust_mape(y_test_real, test_preds)

    # Predicciones futuras
    last_window = scaled_data[-window_size:]
    future_preds_scaled = []
    current_input = last_window.reshape(1, window_size, 2)
    for _ in range(horizon_days):
        pred = model.predict(current_input, verbose=0)[0][0]
        future_preds_scaled.append(pred)
        new_feature = np.array([[pred, 0.0]])  # Sin predicción de volumen
        current_input = np.append(current_input[:, 1:, :], scaler.transform(new_feature).reshape(1, 1, 2), axis=1)
    future_preds = scaler.inverse_transform(np.concatenate([np.array(future_preds_scaled).reshape(-1, 1), np.zeros((horizon_days, 1))], axis=1))[:, 0]

    future_dates = pd.date_range(start=df["ds"].iloc[-1], periods=horizon_days + 1, freq="D")[1:]
    return df, test_preds, future_preds, rmse, mape, future_dates, sentiment_factor, symbol, crypto_sent, market_sent

# Aplicación principal
def main_app():
    """Configura y ejecuta la aplicación Streamlit."""
    st.set_page_config(page_title="Crypto Price Predictions 🔮", layout="wide")
    st.title("Crypto Price Predictions 🔮")
    st.markdown("Predice precios de criptomonedas usando LSTM con datos históricos de CoinCap y análisis de sentimiento.")

    # Sidebar
    st.sidebar.title("Configura tu Predicción")
    crypto_name = st.sidebar.selectbox("Selecciona una criptomoneda:", list(coincap_ids.keys()))
    coin_id = coincap_ids[crypto_name]
    horizon = st.sidebar.slider("Días a predecir:", 1, 60, 5)
    st.sidebar.markdown("**Nota:** Los hiperparámetros se ajustan automáticamente según volatilidad.")
    show_stats = st.sidebar.checkbox("Ver estadísticas descriptivas", value=False)

    # Gráfico histórico
    df_prices = load_coincap_data(coin_id)
    if df_prices is not None:
        fig_hist = px.line(df_prices, x="ds", y="close_price", title=f"Histórico de {crypto_name}", labels={"ds": "Fecha", "close_price": "Precio (USD)"})
        fig_hist.update_layout(template="plotly_dark")
        st.plotly_chart(fig_hist, use_container_width=True)
        if show_stats:
            st.subheader("Estadísticas Descriptivas")
            st.write(df_prices["close_price"].describe())

    # Pestañas
    tabs = st.tabs(["🤖 Entrenamiento y Test", "🔮 Predicción Futura", "📊 Análisis de Sentimiento"])
    
    with tabs[0]:
        st.header("Entrenamiento del Modelo y Evaluación en Test")
        if st.button("Entrenar Modelo y Predecir"):
            with st.spinner("Entrenando modelo..."):
                result = train_and_predict_with_sentiment(coin_id, horizon)
            if result:
                df, test_preds, future_preds, rmse, mape, future_dates, sentiment_factor, symbol, crypto_sent, market_sent = result
                st.success("¡Entrenamiento y predicción completados!")
                st.write(f"Sentimiento combinado de {symbol}: {crypto_sent:.2f}")
                st.write(f"Sentimiento global del mercado: {market_sent:.2f}")
                st.write(f"Factor combinado: {sentiment_factor:.2f}")
                col1, col2 = st.columns(2)
                col1.metric("RMSE (Test)", f"{rmse:.2f}", help="Error promedio en USD")
                col2.metric("MAPE (Test)", f"{mape:.2f}%", help="Error relativo promedio")
                fig_test = go.Figure()
                test_dates = df["ds"].iloc[-len(test_preds):]
                fig_test.add_trace(go.Scatter(x=test_dates, y=df["close_price"].iloc[-len(test_preds):], mode="lines", name="Precio Real"))
                fig_test.add_trace(go.Scatter(x=test_dates, y=test_preds, mode="lines", name="Predicción", line=dict(dash="dash")))
                fig_test.update_layout(title=f"Comparación en Test: {symbol}", template="plotly_dark")
                st.plotly_chart(fig_test, use_container_width=True)
                st.session_state["result"] = result

    with tabs[1]:
        st.header(f"Predicción de Precios - {crypto_name}")
        if "result" in st.session_state:
            df, test_preds, future_preds, rmse, mape, future_dates, sentiment_factor, symbol, crypto_sent, market_sent = st.session_state["result"]
            fig_future = go.Figure()
            fig_future.add_trace(go.Scatter(x=future_dates, y=future_preds, mode="lines+markers", name="Predicción Futura"))
            fig_future.update_layout(title=f"Predicción a {horizon} días - {symbol}", template="plotly_dark")
            st.plotly_chart(fig_future, use_container_width=True)
            st.subheader("Valores Predichos")
            st.dataframe(pd.DataFrame({"Fecha": future_dates, "Predicción (USD)": future_preds}))
        else:
            st.info("Entrena el modelo primero para ver las predicciones.")

    with tabs[2]:
        st.header("Análisis de Sentimiento")
        if "result" in st.session_state:
            df, test_preds, future_preds, rmse, mape, future_dates, sentiment_factor, symbol, crypto_sent, market_sent = st.session_state["result"]
            sentiment_texts = {
                "BTC": f"Sentimiento combinado de {crypto_sent:.2f} refleja cautela con optimismo moderado.",
                "ETH": f"Sentimiento combinado de {crypto_sent:.2f} sugiere neutralidad.",
                "XRP": f"Sentimiento combinado de {crypto_sent:.2f} indica pesimismo leve.",
            }
            st.write(sentiment_texts.get(symbol, f"Sentimiento combinado: {crypto_sent:.2f}, Mercado: {market_sent:.2f}"))
            fig_sentiment = go.Figure(data=[
                go.Bar(name="Sentimiento Combinado", x=[symbol], y=[crypto_sent], marker_color="#1f77b4"),
                go.Bar(name="Sentimiento Global", x=[symbol], y=[market_sent], marker_color="#ff7f0e")
            ])
            fig_sentiment.update_layout(barmode="group", title=f"Análisis de Sentimiento - {symbol}", template="plotly_dark")
            st.plotly_chart(fig_sentiment, use_container_width=True)
        else:
            st.info("Entrena el modelo para ver el análisis de sentimiento.")

if __name__ == "__main__":
    main_app()