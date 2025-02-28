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
from tensorflow.keras.layers import Conv1D, LSTM, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
import time
import certifi
import os

# Importaciones opcionales
try:
    from statsmodels.tsa.arima.model import ARIMA
    from sklearn.metrics import mean_squared_error
    ARIMA_AVAILABLE = True
except ImportError:
    ARIMA_AVAILABLE = False
try:
    from prophet import Prophet
    PROPHET_AVAILABLE = True
except ImportError:
    PROPHET_AVAILABLE = False

# Configuración SSL y sesión de requests
os.environ["REQUESTS_CA_BUNDLE"] = certifi.where()
session = requests.Session()
retry = Retry(total=5, backoff_factor=1, status_forcelist=[429, 500, 502, 503, 504])
adapter = HTTPAdapter(max_retries=retry)
session.mount("https://", adapter)

# Diccionarios
coincap_ids = {"Bitcoin (BTC)": "bitcoin", "Ethereum (ETH)": "ethereum", "Ripple (XRP)": "xrp",
               "Binance Coin (BNB)": "binance-coin", "Cardano (ADA)": "cardano", "Solana (SOL)": "solana",
               "Dogecoin (DOGE)": "dogecoin", "Polkadot (DOT)": "polkadot", "Polygon (MATIC)": "polygon",
               "Litecoin (LTC)": "litecoin", "TRON (TRX)": "tron", "Stellar (XLM)": "stellar"}
coinid_to_symbol = {v: k.split(" (")[1][:-1] for k, v in coincap_ids.items()}
historical_sentiment = {"bitcoin": 65.0, "ethereum": 60.0, "xrp": 21.5, "binance-coin": 55.0,
                       "cardano": 50.0, "solana": 45.0, "dogecoin": 40.0, "polkadot": 50.0,
                       "polygon": 45.0, "litecoin": 50.0, "tron": 45.0, "stellar": 48.0}

# Funciones de apoyo
def robust_mape(y_true, y_pred, eps=1e-9):
    return np.mean(np.abs((y_true - y_pred) / np.maximum(np.abs(y_true), eps))) * 100

# Carga de datos
@st.cache_data
def load_coincap_data(coin_id):
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

# Secuencias y modelo LSTM
def create_sequences(data, window_size):
    if len(data) <= window_size: return None, None
    X, y = [], []
    for i in range(window_size, len(data)):
        X.append(data[i-window_size:i])
        y.append(data[i, 0])
    return np.array(X), np.array(y)

def build_lstm_model(input_shape):
    model = Sequential([Conv1D(32, 2, activation="relu", input_shape=input_shape),
                        LSTM(64, return_sequences=True), Dropout(0.3),
                        LSTM(64), Dropout(0.3), Dense(16, activation="relu"), Dropout(0.2), Dense(1)])
    model.compile(optimizer=Adam(0.001), loss="mean_squared_error")
    return model

def train_model(X_train, y_train, X_val, y_val, window_size):
    model = build_lstm_model((window_size, X_train.shape[2]))
    callbacks = [EarlyStopping(patience=5, restore_best_weights=True), ReduceLROnPlateau(patience=3, factor=0.5, min_lr=1e-6)]
    model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=50, batch_size=32, callbacks=callbacks, verbose=0)
    return model

# Sentimiento
@st.cache_data(ttl=3600)
def get_sentiment_data(coin_id):
    try:
        fg = float(session.get("https://api.alternative.me/fng/?format=json", timeout=10).json()["data"][0]["value"])
    except Exception:
        fg = 50.0
    try:
        cg_id = coin_id if coin_id != "xrp" else "ripple"
        cg_data = session.get(f"https://api.coingecko.com/api/v3/coins/{cg_id}?community_data=true", timeout=10).json()["community_data"]
        cg = min(100, max(cg_data.get("twitter_followers", 0), cg_data.get("reddit_average_posts_48h", 0) * 1000) / 20000000 * 100)
    except Exception:
        cg = 50.0
    hist = historical_sentiment.get(coin_id, 50.0)
    return max(0, min(100, 0.5 * fg + 0.3 * cg + 0.2 * hist)), fg

# Modelos alternativos
def train_alternative_models(df, horizon_days, test_size=0.2):
    train_size = int(len(df) * (1 - test_size))
    train, test = df["close_price"][:train_size], df["close_price"][train_size:]
    arima_result = prophet_result = (None, None, np.nan, np.nan)
    if ARIMA_AVAILABLE:
        arima_model = ARIMA(train, order=(5, 1, 0)).fit()
        arima_preds = arima_model.forecast(steps=len(test))
        arima_result = (arima_preds, arima_preds, np.sqrt(mean_squared_error(test, arima_preds)), robust_mape(test, arima_preds))
    if PROPHET_AVAILABLE:
        prophet_df = df.rename(columns={"ds": "ds", "close_price": "y"})[:train_size]
        model = Prophet(yearly_seasonality=True, weekly_seasonality=True).fit(prophet_df)
        future = model.make_future_dataframe(periods=horizon_days + len(test))
        forecast = model.predict(future)
        prophet_result = (forecast["yhat"][train_size:train_size+len(test)].values,
                         forecast["yhat"][-horizon_days:].values,
                         np.sqrt(mean_squared_error(test, forecast["yhat"][train_size:train_size+len(test)])),
                         robust_mape(test, forecast["yhat"][train_size:train_size+len(test)]))
    return arima_result, prophet_result

# Predicción
def train_and_predict(coin_id, horizon_days):
    df = load_coincap_data(coin_id)
    if df is None: return None
    symbol = coinid_to_symbol[coin_id]
    sentiment, fg = get_sentiment_data(coin_id)
    sentiment_factor = (sentiment + fg) / 200.0

    scaler = MinMaxScaler()
    scaled_data = scaler.fit_transform(df[["close_price"]])
    window_size = min(max(15, int(horizon_days * (1.5 if df["close_price"].pct_change().std() > 0.05 else 1))), len(df) // 3)
    X, y = create_sequences(scaled_data, window_size)
    if X is None: return None

    split = int(len(X) * 0.8)
    X_train, X_test = X[:split], X[split:]
    y_train, y_test = y[:split], y[split:]
    val_split = int(len(X_train) * 0.9)
    X_val, y_val = X_train[val_split:], y_train[val_split:]
    X_train, y_train = X_train[:val_split], y_train[:val_split]

    X_train_adj = np.concatenate([X_train, np.full((X_train.shape[0], window_size, 1), sentiment_factor)], axis=-1)
    X_val_adj = np.concatenate([X_val, np.full((X_val.shape[0], window_size, 1), sentiment_factor)], axis=-1)
    X_test_adj = np.concatenate([X_test, np.full((X_test.shape[0], window_size, 1), sentiment_factor)], axis=-1)

    lstm_model = train_model(X_train_adj, y_train, X_val_adj, y_val, window_size)
    test_preds_scaled = lstm_model.predict(X_test_adj, verbose=0)
    test_preds = scaler.inverse_transform(test_preds_scaled)
    y_test_real = scaler.inverse_transform(y_test.reshape(-1, 1))
    lstm_rmse, lstm_mape = np.sqrt(mean_squared_error(y_test_real, test_preds)), robust_mape(y_test_real, test_preds)

    arima_result, prophet_result = train_alternative_models(df, horizon_days)
    models = {"LSTM": (test_preds, None, lstm_rmse, lstm_mape)}
    if ARIMA_AVAILABLE: models["ARIMA"] = arima_result
    if PROPHET_AVAILABLE: models["Prophet"] = prophet_result

    best_model = min(models.items(), key=lambda x: x[1][2])[0]
    test_preds = models[best_model][0]
    future_preds = models[best_model][1]
    if best_model == "LSTM":
        last_window = np.concatenate([scaled_data[-window_size:], np.full((window_size, 1), sentiment_factor)], axis=-1).reshape(1, window_size, 2)
        future_preds_scaled = []
        current_input = last_window
        for _ in range(horizon_days):
            pred = lstm_model.predict(current_input, verbose=0)[0][0]
            future_preds_scaled.append(pred)
            current_input = np.append(current_input[:, 1:, :], [[[pred, sentiment_factor]]], axis=1)
        future_preds = scaler.inverse_transform(np.array(future_preds_scaled).reshape(-1, 1)).flatten()

    return df, test_preds, y_test_real, future_preds, models[best_model][2], models[best_model][3], sentiment_factor, symbol, best_model

# App
def main_app():
    st.set_page_config(page_title="Crypto Price Predictions 🔮", layout="wide")
    st.title("Crypto Price Predictions 🔮")
    st.sidebar.title("Configuración")
    crypto_name = st.sidebar.selectbox("Criptomoneda:", list(coincap_ids.keys()))
    horizon = st.sidebar.slider("Días a predecir:", 1, 90, 60)

    df = load_coincap_data(coincap_ids[crypto_name])
    if df is not None:
        st.plotly_chart(px.line(df, x="ds", y="close_price", title=f"Histórico de {crypto_name}", template="plotly_dark"), use_container_width=True)

    tabs = st.tabs(["Entrenamiento", "Predicción", "Sentimiento"])
    with tabs[0]:
        if st.button("Entrenar y Predecir"):
            with st.spinner("Procesando..."):
                result = train_and_predict(coincap_ids[crypto_name], horizon)
            if result:
                df, test_preds, y_test_real, future_preds, rmse, mape, _, symbol, best_model = result
                st.success(f"Mejor modelo: {best_model} (RMSE: {rmse:.2f}, MAPE: {mape:.2f}%)")
                col1, col2 = st.columns(2)
                col1.metric("RMSE", f"{rmse:.2f}")
                col2.metric("MAPE", f"{mape:.2f}%")
                fig = go.Figure()
                test_dates = df["ds"].iloc[-len(test_preds):]
                fig.add_trace(go.Scatter(x=test_dates, y=y_test_real.flatten(), mode="lines", name="Real"))
                fig.add_trace(go.Scatter(x=test_dates, y=test_preds.flatten(), mode="lines", name="Predicción"))
                fig.update_layout(template="plotly_dark", title=f"Test: {symbol}")
                st.plotly_chart(fig, use_container_width=True)

    with tabs[1]:
        if "result" in locals() and result:
            last_date = df["ds"].iloc[-1]
            future_dates = pd.date_range(start=last_date, periods=horizon+1, freq="D")
            pred_series = np.concatenate(([df["close_price"].iloc[-1]], future_preds))
            fig = go.Figure(go.Scatter(x=future_dates, y=pred_series, mode="lines+markers", name="Predicción"))
            fig.update_layout(template="plotly_dark", title=f"Predicción ({horizon} días) - {symbol}")
            st.plotly_chart(fig, use_container_width=True)
            st.dataframe(pd.DataFrame({"Fecha": future_dates, "Predicción": pred_series}))

    with tabs[2]:
        if "result" in locals() and result:
            sentiment, fg = get_sentiment_data(coincap_ids[crypto_name])
            st.write(f"Sentimiento {symbol}: {sentiment:.2f}, Factor: {result[6]:.2f}")
            st.plotly_chart(go.Figure(go.Bar(x=[symbol], y=[sentiment], name="Sentimiento"), layout={"template": "plotly_dark"}))

if __name__ == "__main__":
    main_app()