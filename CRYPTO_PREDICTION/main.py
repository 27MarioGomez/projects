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
import socket

# Configurar certificados SSL para requests
os.environ["REQUESTS_CA_BUNDLE"] = certifi.where()
os.environ["SSL_CERT_FILE"] = certifi.where()

##############################################
# Funciones de apoyo
##############################################

def robust_mape(y_true, y_pred, eps=1e-9):
    """Calcula el MAPE evitando divisiones por cero."""
    return np.mean(np.abs((y_true - y_pred) / np.maximum(np.abs(y_true), eps))) * 100

# Diccionarios para mapear criptomonedas
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

##############################################
# Descarga de datos desde CoinCap
##############################################
@st.cache_data
def load_coincap_data(coin_id, start_ms=None, end_ms=None, max_retries=3):
    """Descarga datos históricos diarios de precios y volumen desde CoinCap."""
    url = f"https://api.coincap.io/v2/assets/{coin_id}/history?interval=d1"
    if start_ms and end_ms:
        url += f"&start={start_ms}&end={end_ms}"
    headers = {"User-Agent": "Mozilla/5.0"}
    for attempt in range(max_retries):
        try:
            resp = requests.get(url, headers=headers, verify=certifi.where(), timeout=10)
            if resp.status_code == 200:
                data = resp.json().get("data", [])
                if not data:
                    st.warning("CoinCap: Datos vacíos.")
                    return None
                df = pd.DataFrame(data)
                df["ds"] = pd.to_datetime(df["time"], unit="ms", errors="coerce")
                df["close_price"] = pd.to_numeric(df["priceUsd"], errors="coerce")
                df["volume"] = pd.to_numeric(df.get("volumeUsd", 0), errors="coerce").fillna(0.0)
                df = df[["ds", "close_price", "volume"]].dropna(subset=["ds", "close_price"])
                df.sort_values(by="ds", inplace=True)
                df.reset_index(drop=True, inplace=True)
                return df[df["close_price"] > 0].copy()
            elif resp.status_code == 429:
                st.warning(f"CoinCap: Límite excedido. Reintento {attempt+1}...")
                time.sleep(15 * (attempt + 1))
            else:
                st.warning(f"CoinCap: Error {resp.status_code}.")
                return None
        except requests.RequestException as e:
            st.warning(f"Error al conectar con CoinCap: {e}. Reintento {attempt+1}")
            time.sleep(5 * (attempt + 1))
    st.error("CoinCap: Fallo tras reintentos.")
    return None

##############################################
# Creación de secuencias para LSTM
##############################################
def create_sequences(data, window_size=30):
    """Crea secuencias temporales para el modelo LSTM."""
    if len(data) <= window_size:
        st.warning(f"Datos insuficientes para ventana de {window_size} días.")
        return None, None
    X, y = [], []
    for i in range(window_size, len(data)):
        X.append(data[i - window_size:i])
        y.append(data[i, 0])
    return np.array(X), np.array(y)

##############################################
# Modelo LSTM optimizado
##############################################
def build_lstm_model(input_shape, learning_rate=0.001):
    """Construye un modelo LSTM con Conv1D y Bidirectional LSTM."""
    model = Sequential([
        Conv1D(filters=32, kernel_size=3, activation="relu", input_shape=input_shape),
        Bidirectional(LSTM(64, return_sequences=True)),
        Dropout(0.3),
        Bidirectional(LSTM(64, return_sequences=True)),
        Dropout(0.3),
        Bidirectional(LSTM(64)),
        Dropout(0.3),
        Dense(1)
    ])
    model.compile(optimizer=Adam(learning_rate=learning_rate), loss="mean_squared_error")
    return model

##############################################
# Entrenamiento del modelo
##############################################
def train_model(X_train, y_train, X_val, y_val, input_shape, epochs, batch_size, learning_rate):
    """Entrena el modelo LSTM con datos de entrenamiento y validación."""
    tf.keras.backend.clear_session()
    model = build_lstm_model(input_shape, learning_rate)
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
    """Ajusta hiperparámetros según datos y horizonte."""
    data_len = len(df)
    volatility = df["close_price"].pct_change().std()
    window_size = min(max(10, horizon_days * 2), min(60, data_len // 2))
    epochs = min(50, max(20, int(data_len / 100) + int(volatility * 100)))
    batch_size = 16 if volatility > 0.05 or data_len < 500 else 32
    learning_rate = 0.0005 if df["close_price"].mean() > 1000 or volatility > 0.1 else 0.001
    return window_size, epochs, batch_size, learning_rate

##############################################
# Funciones optimizadas para LunarCrush
##############################################

@st.cache_data(ttl=3600)
def get_crypto_sentiment_lunarcrush(symbol, max_retries=3):
    """Obtiene el sentimiento de una criptomoneda desde LunarCrush."""
    if "lunarcrush_api_key" not in st.secrets:
        st.error("Clave API de LunarCrush no configurada en Secrets.")
        return 50.0
    api_key = st.secrets["lunarcrush_api_key"]
    url = f"https://api.lunarcrush.com/v2?data=assets&symbol={symbol}&key={api_key}"
    for attempt in range(max_retries):
        try:
            resp = requests.get(url, timeout=10, verify=certifi.where())
            if resp.status_code == 200:
                data = resp.json().get("data", [])
                if data and "galaxy_score" in data[0]:
                    return float(data[0]["galaxy_score"])
                return 50.0
            elif resp.status_code == 429:
                st.warning(f"LunarCrush: Límite excedido (assets). Reintento {attempt+1}...")
                time.sleep(15 * (attempt + 1))
            else:
                st.warning(f"LunarCrush: Error {resp.status_code} (assets).")
                return 50.0
        except (requests.RequestException, socket.gaierror) as e:
            st.warning(f"Error de red en LunarCrush (assets): {e}. Reintento {attempt+1}")
            time.sleep(5 * (attempt + 1))
    st.error("LunarCrush (assets): Fallo tras reintentos.")
    return 50.0

@st.cache_data(ttl=3600)
def get_market_crypto_sentiment_lunarcrush(max_retries=3):
    """Obtiene un sentimiento global aproximado del mercado cripto."""
    if "lunarcrush_api_key" not in st.secrets:
        st.error("Clave API de LunarCrush no configurada en Secrets.")
        return 50.0
    api_key = st.secrets["lunarcrush_api_key"]
    url = f"https://api.lunarcrush.com/v2?data=market&key={api_key}"
    for attempt in range(max_retries):
        try:
            resp = requests.get(url, timeout=10, verify=certifi.where())
            if resp.status_code == 200:
                data = resp.json().get("data", [])
                if data:
                    btc_dom = data[0].get("btc_dominance", 45)
                    return max(0, min(100, (btc_dom - 30) / (60 - 30) * 100))
                return 50.0
            elif resp.status_code == 429:
                st.warning(f"LunarCrush: Límite excedido (market). Reintento {attempt+1}...")
                time.sleep(15 * (attempt + 1))
            else:
                st.warning(f"LunarCrush: Error {resp.status_code} (market).")
                return 50.0
        except (requests.RequestException, socket.gaierror) as e:
            st.warning(f"Error de red en LunarCrush (market): {e}. Reintento {attempt+1}")
            time.sleep(5 * (attempt + 1))
    st.error("LunarCrush (market): Fallo tras reintentos.")
    return 50.0

@st.cache_data(ttl=3600)
def get_lunarcrush_news(symbol, limit=5, max_retries=3):
    """Obtiene noticias recientes de LunarCrush."""
    if "lunarcrush_api_key" not in st.secrets:
        st.error("Clave API de LunarCrush no configurada en Secrets.")
        return []
    api_key = st.secrets["lunarcrush_api_key"]
    url = f"https://api.lunarcrush.com/v2?data=news&symbol={symbol}&limit={limit}&key={api_key}"
    for attempt in range(max_retries):
        try:
            resp = requests.get(url, timeout=10, verify=certifi.where())
            if resp.status_code == 200:
                items = resp.json().get("data", [])
                return [{"title": item.get("title"), "url": item.get("url"), 
                         "description": item.get("description"), "published_at": item.get("published_at")} 
                        for item in items]
            elif resp.status_code == 429:
                st.warning(f"LunarCrush: Límite excedido (news). Reintento {attempt+1}...")
                time.sleep(15 * (attempt + 1))
            else:
                st.warning(f"LunarCrush: Error {resp.status_code} (news).")
                return []
        except (requests.RequestException, socket.gaierror) as e:
            st.warning(f"Error de red en LunarCrush (news): {e}. Reintento {attempt+1}")
            time.sleep(5 * (attempt + 1))
    st.error("LunarCrush (news): Fallo tras reintentos.")
    return []

##############################################
# Entrenamiento y predicción con LSTM
##############################################
def train_and_predict_with_sentiment(coin_id, use_custom_range, start_ms, end_ms, horizon_days=30, test_size=0.2):
    """Entrena el modelo LSTM con datos de precios y sentimiento."""
    df = load_coincap_data(coin_id, start_ms, end_ms)
    if df is None or df.empty:
        st.error("No se pudieron cargar datos de CoinCap.")
        return None
    
    symbol = coinid_to_symbol.get(coin_id, "BTC")
    crypto_sent = get_crypto_sentiment_lunarcrush(symbol)
    market_sent = get_market_crypto_sentiment_lunarcrush()
    sentiment_factor = (crypto_sent + market_sent) / 200.0

    st.write(f"Sentimiento de {symbol}: {crypto_sent:.2f}")
    st.write(f"Sentimiento de mercado: {market_sent:.2f}")
    st.write(f"Factor combinado: {sentiment_factor:.2f}")

    window_size, epochs, batch_size, learning_rate = get_dynamic_params(df, horizon_days)
    st.info(f"Parámetros: window={window_size}, epochs={epochs}, batch={batch_size}, lr={learning_rate}")

    scaler = MinMaxScaler()
    scaled_data = scaler.fit_transform(df[["close_price"]].values)
    split_idx = int(len(scaled_data) * (1 - test_size))
    if split_idx <= window_size:
        st.error("Datos insuficientes para entrenar.")
        return None

    train_data, test_data = scaled_data[:split_idx], scaled_data[split_idx:]
    X_train, y_train = create_sequences(train_data, window_size)
    X_test, y_test = create_sequences(test_data, window_size)
    if X_train is None or X_test is None:
        return None

    val_split = int(len(X_train) * 0.9)
    X_val, y_val = X_train[val_split:], y_train[val_split:]
    X_train, y_train = X_train[:val_split], y_train[:val_split]

    X_train_adj = np.concatenate([X_train, np.full((X_train.shape[0], window_size, 1), sentiment_factor)], axis=-1)
    X_val_adj = np.concatenate([X_val, np.full((X_val.shape[0], window_size, 1), sentiment_factor)], axis=-1)
    X_test_adj = np.concatenate([X_test, np.full((X_test.shape[0], window_size, 1), sentiment_factor)], axis=-1)

    model = train_model(X_train_adj, y_train, X_val_adj, y_val, (window_size, 2), epochs, batch_size, learning_rate)
    test_preds_scaled = model.predict(X_test_adj)
    test_preds = scaler.inverse_transform(test_preds_scaled)

    last_window = scaled_data[-window_size:]
    current_input = np.concatenate([last_window.reshape(1, window_size, 1), 
                                    np.full((1, window_size, 1), sentiment_factor)], axis=-1)
    future_preds_scaled = []
    for _ in range(horizon_days):
        pred = model.predict(current_input, verbose=0)[0][0]
        future_preds_scaled.append(pred)
        new_input = np.append(current_input[:, 1:, :], 
                              np.array([[[pred, sentiment_factor]]]), axis=1)
        current_input = new_input
    future_preds = scaler.inverse_transform(np.array(future_preds_scaled).reshape(-1, 1)).flatten()

    y_test_real = scaler.inverse_transform(y_test.reshape(-1, 1))
    rmse = np.sqrt(np.mean((y_test_real - test_preds) ** 2))
    mape = robust_mape(y_test_real, test_preds)

    return df, test_preds, y_test_real, future_preds, rmse, mape, sentiment_factor, symbol

##############################################
# Aplicación principal
##############################################
def main_app():
    st.set_page_config(page_title="Crypto Price Predictions 🔮", layout="wide")
    st.title("Crypto Price Predictions 🔮")
    st.markdown("**Datos:** CoinCap (precios) y LunarCrush (sentimiento/noticias)")

    crypto_name = st.sidebar.selectbox("Criptomoneda:", list(coincap_ids.keys()))
    coin_id = coincap_ids[crypto_name]

    use_custom_range = st.sidebar.checkbox("Rango de fechas", value=True)
    if use_custom_range:
        start_date = st.sidebar.date_input("Inicio", datetime(2021, 1, 1))
        end_date = st.sidebar.date_input("Fin", datetime.now())
        start_ms = int(datetime.combine(start_date, datetime.min.time()).timestamp() * 1000)
        end_ms = int(datetime.combine(end_date, datetime.min.time()).timestamp() * 1000)
    else:
        start_ms, end_ms = None, None

    horizon = st.sidebar.slider("Días a predecir:", 1, 60, 30)

    df_prices = load_coincap_data(coin_id, start_ms, end_ms)
    if df_prices is not None:
        fig = px.line(df_prices, x="ds", y="close_price", title=f"Histórico de {crypto_name}")
        st.plotly_chart(fig, use_container_width=True)

    tabs = st.tabs(["Entrenamiento", "Predicción", "Noticias"])

    with tabs[0]:
        st.header("Entrenamiento y Evaluación")
        if st.button("Entrenar Modelo"):
            with st.spinner("Entrenando..."):
                result = train_and_predict_with_sentiment(coin_id, use_custom_range, start_ms, end_ms, horizon)
            if result:
                df, test_preds, y_test_real, future_preds, rmse, mape, _, symbol = result
                st.success("¡Modelo entrenado!")
                st.write(f"RMSE: {rmse:.2f}, MAPE: {mape:.2f}%")
                fig_test = go.Figure()
                fig_test.add_trace(go.Scatter(x=df["ds"].iloc[-len(y_test_real):], y=y_test_real.flatten(), name="Real"))
                fig_test.add_trace(go.Scatter(x=df["ds"].iloc[-len(test_preds):], y=test_preds.flatten(), name="Predicho"))
                st.plotly_chart(fig_test, use_container_width=True)
                st.session_state["result"] = result

    with tabs[1]:
        st.header("Predicción Futura")
        if "result" in st.session_state:
            df, _, _, future_preds, _, _, sentiment_factor, symbol = st.session_state["result"]
            future_dates = pd.date_range(start=df["ds"].iloc[-1], periods=horizon + 1, freq="D")
            pred_series = np.concatenate(([df["close_price"].iloc[-1]], future_preds))
            fig_future = go.Figure(go.Scatter(x=future_dates, y=pred_series, mode="lines+markers"))
            fig_future.update_layout(title=f"Predicción ({horizon} días) - {symbol}")
            st.plotly_chart(fig_future, use_container_width=True)
            st.dataframe(pd.DataFrame({"Fecha": future_dates, "Predicción": pred_series}))
        else:
            st.info("Entrena el modelo primero.")

    with tabs[2]:
        st.header("Noticias Recientes")
        if "result" in st.session_state:
            symbol = st.session_state["result"][7]
            news = get_lunarcrush_news(symbol)
            for i, item in enumerate(news, 1):
                st.markdown(f"**{i}. {item['title']}**")
                st.markdown(f"[Leer más]({item['url']})")
                st.write(f"Publicado: {item['published_at']}")
                st.write("---")
        else:
            st.info("Entrena el modelo para ver noticias.")

if __name__ == "__main__":
    main_app()