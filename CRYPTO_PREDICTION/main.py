import streamlit as st
import pandas as pd
import numpy as np
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

# Configurar certificados SSL y sesión requests con reintentos
os.environ["REQUESTS_CA_BUNDLE"] = certifi.where()
session = requests.Session()
retry = Retry(total=5, backoff_factor=1, status_forcelist=[429, 500, 502, 503, 504])
adapter = HTTPAdapter(max_retries=retry)
session.mount("https://", adapter)

# Diccionarios para CoinCap y mapeo a símbolo para CoinGecko
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

coinid_to_symbol = {
    "bitcoin": "BTC",
    "ethereum": "ETH",
    "xrp": "XRP",
    "binance-coin": "BNB",
    "cardano": "ADA",
    "solana": "SOL",
    "dogecoin": "DOGE",
    "polkadot": "DOT",
    "polygon": "MATIC",
    "litecoin": "LTC",
    "tron": "TRX",
    "stellar": "XLM"
}

coinid_to_coingecko = {
    "bitcoin": "bitcoin",
    "ethereum": "ethereum",
    "xrp": "ripple",
    "binance-coin": "binancecoin",
    "cardano": "cardano",
    "solana": "solana",
    "dogecoin": "dogecoin",
    "polkadot": "polkadot",
    "polygon": "polygon",
    "litecoin": "litecoin",
    "tron": "tron",
    "stellar": "stellar"
}

# Promedios históricos aproximados de sentimiento por criptomoneda
historical_sentiment = {
    "bitcoin": 65.0,
    "ethereum": 60.0,
    "xrp": 21.5,
    "binance-coin": 55.0,
    "cardano": 50.0,
    "solana": 45.0,
    "dogecoin": 40.0,
    "polkadot": 50.0,
    "polygon": 45.0,
    "litecoin": 50.0,
    "tron": 45.0,
    "stellar": 48.0
}

# Descarga de datos desde CoinCap
@st.cache_data
def load_coincap_data(coin_id):
    end_date = datetime.now()
    start_date = end_date - timedelta(days=730)  # 2 años
    start_ms = int(start_date.timestamp() * 1000)
    end_ms = int(end_date.timestamp() * 1000)
    url = f"https://api.coincap.io/v2/assets/{coin_id}/history?interval=d1&start={start_ms}&end={end_ms}"
    headers = {"User-Agent": "Mozilla/5.0"}
    try:
        resp = session.get(url, headers=headers, verify=certifi.where(), timeout=10)
        if resp.status_code == 200:
            data = resp.json()
            df = pd.DataFrame(data["data"])
            df["ds"] = pd.to_datetime(df["time"], unit="ms")
            df["close_price"] = pd.to_numeric(df["priceUsd"], errors="coerce")
            df["volume"] = pd.to_numeric(df.get("volumeUsd", 0), errors="coerce").fillna(0.0)
            df = df[["ds", "close_price", "volume"]].dropna(subset=["ds", "close_price"])
            df.sort_values(by="ds", inplace=True)
            df.reset_index(drop=True, inplace=True)
            return df
        else:
            st.warning(f"Error {resp.status_code} al obtener datos de CoinCap.")
            return None
    except Exception as e:
        st.error(f"Error al conectar con CoinCap: {e}")
        return None

# Funciones de Sentimiento
def get_fear_greed_index():
    url = "https://api.alternative.me/fng/?format=json"
    try:
        resp = session.get(url, timeout=10)
        if resp.status_code == 200:
            return float(resp.json()["data"][0]["value"])
        return 50.0
    except Exception:
        return 50.0

@st.cache_data(ttl=86400)
def get_coingecko_community_activity(coin_id):
    cg_id = coinid_to_coingecko.get(coin_id, coin_id)
    url = f"https://api.coingecko.com/api/v3/coins/{cg_id}?localization=false&tickers=false&market_data=false&community_data=true"
    try:
        resp = session.get(url, timeout=10)
        if resp.status_code == 200:
            data = resp.json()
            followers = data["community_data"].get("twitter_followers", 0)
            posts = data["community_data"].get("reddit_average_posts_48h", 0)
            activity = max(followers, posts * 1000)
            return min(100, (activity / 20000000) * 100) if activity > 0 else 50.0
        return 50.0
    except Exception:
        return 50.0

def get_crypto_sentiment(coin_id):
    fg = get_fear_greed_index()
    cg_activity = get_coingecko_community_activity(coin_id)
    hist_sent = historical_sentiment.get(coin_id, 50.0)
    return 0.5 * fg + 0.3 * cg_activity + 0.2 * hist_sent

def get_market_sentiment():
    return get_fear_greed_index()

# Creación de secuencias para LSTM
def create_sequences(data, window_size=20):
    if len(data) <= window_size:
        return None, None
    X, y = [], []
    for i in range(window_size, len(data)):
        X.append(data[i-window_size:i])
        y.append(data[i, 0])  # Precio como objetivo
    return np.array(X), np.array(y)

# Modelo LSTM
def build_lstm_model(input_shape):
    model = Sequential()
    model.add(LSTM(64, return_sequences=True, input_shape=input_shape))
    model.add(Dropout(0.3))
    model.add(LSTM(64))
    model.add(Dropout(0.3))
    model.add(Dense(16, activation="relu"))
    model.add(Dense(1))
    model.compile(optimizer=Adam(learning_rate=0.0005), loss="mean_squared_error")
    return model

def train_and_predict_with_sentiment(df, horizon_days, coin_id):
    # Agregar columnas de sentimiento al DataFrame
    df["market_sentiment"] = get_market_sentiment()
    df["crypto_sentiment"] = get_crypto_sentiment(coin_id)

    # Escalar datos
    scaler = MinMaxScaler()
    scaled_data = scaler.fit_transform(df[["close_price", "volume", "market_sentiment", "crypto_sentiment"]])

    # Crear secuencias
    window_size = 20
    X, y = create_sequences(scaled_data, window_size)
    if X is None or y is None:
        return None

    # Dividir en entrenamiento, validación y prueba
    train_size = int(len(X) * 0.7)
    val_size = int(len(X) * 0.15)
    X_train, X_val, X_test = X[:train_size], X[train_size:train_size+val_size], X[train_size+val_size:]
    y_train, y_val, y_test = y[:train_size], y[train_size:train_size+val_size], y[train_size+val_size:]

    # Entrenar modelo
    model = build_lstm_model((window_size, X.shape[2]))
    early_stop = EarlyStopping(monitor="val_loss", patience=5, restore_best_weights=True)
    model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=50, batch_size=32, callbacks=[early_stop], verbose=0)

    # Predicciones en el conjunto de prueba
    test_preds = model.predict(X_test, verbose=0)
    test_preds = scaler.inverse_transform(np.concatenate([test_preds, np.zeros((len(test_preds), 3))], axis=1))[:, 0]

    # Predicciones futuras
    last_sequence = scaled_data[-window_size:]
    future_preds = []
    for _ in range(horizon_days):
        pred = model.predict(last_sequence.reshape(1, window_size, X.shape[2]), verbose=0)
        future_preds.append(pred[0, 0])
        last_sequence = np.roll(last_sequence, -1, axis=0)
        last_sequence[-1] = np.concatenate([pred[0], last_sequence[-1, 1:]])

    future_preds = scaler.inverse_transform(np.concatenate([np.array(future_preds).reshape(-1, 1), np.zeros((horizon_days, 3))], axis=1))[:, 0]

    # Devolver resultados como diccionario
    return {
        "df": df,
        "test_preds": test_preds,
        "future_preds": future_preds,
        "market_sentiment": df["market_sentiment"].iloc[-1],
        "crypto_sentiment": df["crypto_sentiment"].iloc[-1]
    }

# Aplicación principal
def main_app():
    st.title("Crypto Price Prediction")

    # Sidebar
    st.sidebar.header("Configura tu Predicción")
    crypto_name = st.sidebar.selectbox("Selecciona la criptomoneda", list(coincap_ids.keys()), index=0)
    horizon_days = st.sidebar.slider("Días a predecir", 1, 60, 30)
    auto_adjust = st.sidebar.checkbox("Los hiperparámetros se ajustan automáticamente", value=True)

    coin_id = coincap_ids[crypto_name]
    symbol = coinid_to_symbol[coin_id]

    # Cargar datos
    df = load_coincap_data(coin_id)
    if df is None:
        st.error("No se pudieron cargar los datos.")
        return

    # Botón para entrenar
    if st.button("Entrena Modelo y Predice"):
        with st.spinner("Entrenando modelo..."):
            result = train_and_predict_with_sentiment(df, horizon_days, coin_id)
            if result is not None:
                st.session_state["result"] = result
                st.success("Entrenamiento predicción completado")

    # Mostrar resultados
    tabs = st.tabs(["Predicción de Precios", "Análisis de Sentimientos"])
    
    with tabs[0]:
        st.header(f"Predicción de Precios - {crypto_name}")
        if "result" in st.session_state:
            result = st.session_state["result"]
            last_date = result["df"]["ds"].iloc[-1]
            current_price = result["df"]["close_price"].iloc[-1]
            pred_series = np.concatenate(([current_price], result["future_preds"]))

            # Gráfico de predicción futura
            future_dates = pd.date_range(start=last_date, periods=horizon_days + 1)
            fig_future = go.Figure()
            fig_future.add_trace(go.Scatter(x=result["df"]["ds"], y=result["df"]["close_price"], mode="lines", name="Precio Real", line=dict(color="blue")))
            fig_future.add_trace(go.Scatter(x=future_dates, y=pred_series, mode="lines", name="Predicción", line=dict(color="red", dash="dash")))
            fig_future.update_layout(title=f"Predicción de Precios para {symbol}", xaxis_title="Fecha", yaxis_title="USD")
            st.plotly_chart(fig_future)

            # Métricas
            st.write(f"Sentimiento combinado de {symbol}: {result['crypto_sentiment']:.2f}")
            st.write(f"Sentimiento global del mercado: {result['market_sentiment']:.2f}")

    with tabs[1]:
        st.header("Análisis de Sentimientos")
        if "result" in st.session_state:
            result = st.session_state["result"]
            sentiment_df = pd.DataFrame({
                "Fecha": result["df"]["ds"],
                "Sentimiento del mercado": result["df"]["market_sentiment"],
                "Sentimiento de la criptomoneda seleccionada": result["df"]["crypto_sentiment"]
            })

            # Gráfico de sentimientos
            fig_sentiment = go.Figure()
            fig_sentiment.add_trace(go.Scatter(x=sentiment_df["Fecha"], y=sentiment_df["Sentimiento del mercado"], mode="lines", name="Sentimiento del mercado", line=dict(color="blue")))
            fig_sentiment.add_trace(go.Scatter(x=sentiment_df["Fecha"], y=sentiment_df["Sentimiento de la criptomoneda seleccionada"], mode="lines", name="Sentimiento de la criptomoneda seleccionada", line=dict(color="green")))
            fig_sentiment.update_layout(title="Análisis de Sentimientos", xaxis_title="Fecha", yaxis_title="Puntuación (0-100)")
            st.plotly_chart(fig_sentiment)
        else:
            st.info("Entrena el modelo para ver el análisis.")

if __name__ == "__main__":
    main_app()