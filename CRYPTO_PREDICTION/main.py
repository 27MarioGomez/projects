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

try:
    from statsmodels.tsa.arima.model import ARIMA
    from sklearn.metrics import mean_squared_error
    ARIMA_AVAILABLE = True
except ImportError:
    ARIMA_AVAILABLE = False
    st.warning("statsmodels no está instalado. ARIMA no disponible.")

try:
    from prophet import Prophet
    PROPHET_AVAILABLE = True
except ImportError:
    PROPHET_AVAILABLE = False
    st.warning("prophet no está instalado. Prophet no disponible.")

# Configurar SSL
os.environ["REQUESTS_CA_BUNDLE"] = certifi.where()
os.environ["SSL_CERT_FILE"] = certifi.where()

# Sesión de requests con reintentos
def get_requests_session():
    session = requests.Session()
    retry = Retry(total=5, backoff_factor=1, status_forcelist=[429, 500, 502, 503, 504], raise_on_status=False)
    adapter = HTTPAdapter(max_retries=retry)
    session.mount("http://", adapter)
    session.mount("https://", adapter)
    return session

session = get_requests_session()

# Funciones de apoyo
def robust_mape(y_true, y_pred, eps=1e-9):
    return np.mean(np.abs((y_true - y_pred) / np.maximum(np.abs(y_true), eps))) * 100

# Diccionarios
coincap_ids = {
    "Bitcoin (BTC)": "bitcoin", "Ethereum (ETH)": "ethereum", "Ripple (XRP)": "xrp",
    "Binance Coin (BNB)": "binance-coin", "Cardano (ADA)": "cardano", "Solana (SOL)": "solana",
    "Dogecoin (DOGE)": "dogecoin", "Polkadot (DOT)": "polkadot", "Polygon (MATIC)": "polygon",
    "Litecoin (LTC)": "litecoin", "TRON (TRX)": "tron", "Stellar (XLM)": "stellar"
}

coinid_to_symbol = {k: v.split(" (")[1][:-1] for k, v in coincap_ids.items()}
coinid_to_coingecko = {v: v.replace("xrp", "ripple").replace("binance-coin", "binancecoin") for v in coincap_ids.values()}

historical_sentiment = {
    "bitcoin": 65.0, "ethereum": 60.0, "xrp": 21.5, "binance-coin": 55.0, "cardano": 50.0,
    "solana": 45.0, "dogecoin": 40.0, "polkadot": 50.0, "polygon": 45.0, "litecoin": 50.0,
    "tron": 45.0, "stellar": 48.0
}

crypto_characteristics = {
    "bitcoin": {"volatility": 0.03}, "ethereum": {"volatility": 0.05}, "xrp": {"volatility": 0.08},
    "binance-coin": {"volatility": 0.06}, "cardano": {"volatility": 0.07}, "solana": {"volatility": 0.09},
    "dogecoin": {"volatility": 0.12}, "polkadot": {"volatility": 0.07}, "polygon": {"volatility": 0.06},
    "litecoin": {"volatility": 0.04}, "tron": {"volatility": 0.06}, "stellar": {"volatility": 0.05}
}

# Carga de datos
@st.cache_data
def load_coincap_data(coin_id):
    end_date = datetime.now()
    start_date = end_date - timedelta(days=730)
    url = f"https://api.coincap.io/v2/assets/{coin_id}/history?interval=d1&start={int(start_date.timestamp() * 1000)}&end={int(end_date.timestamp() * 1000)}"
    try:
        resp = session.get(url, headers={"User-Agent": "Mozilla/5.0"}, verify=certifi.where(), timeout=10)
        if resp.status_code == 200:
            df = pd.DataFrame(resp.json()["data"])
            df["ds"] = pd.to_datetime(df["time"], unit="ms").dt.normalize()
            df["close_price"] = pd.to_numeric(df["priceUsd"])
            df["volume"] = pd.to_numeric(df.get("volumeUsd", 0.0)).fillna(0.0)
            return df[["ds", "close_price", "volume"]].dropna().sort_values("ds").reset_index(drop=True)
        st.warning(f"CoinCap: Error {resp.status_code}")
        return None
    except Exception as e:
        st.error(f"Error al cargar datos: {e}")
        return None

# Secuencias LSTM
def create_sequences(data, window_size):
    if len(data) <= window_size:
        st.warning(f"Datos insuficientes para ventana de {window_size}.")
        return None, None
    X, y = [], []
    for i in range(window_size, len(data)):
        X.append(data[i-window_size:i])
        y.append(data[i, 0])
    return np.array(X), np.array(y)

# Modelo LSTM
def build_lstm_model(input_shape):
    model = Sequential([
        Conv1D(32, 2, activation="relu", input_shape=input_shape, kernel_regularizer=tf.keras.regularizers.l2(0.005)),
        LSTM(64, return_sequences=True), Dropout(0.3),
        LSTM(64), Dropout(0.3),
        Dense(16, activation="relu"), Dropout(0.2),
        Dense(1)
    ])
    model.compile(optimizer=Adam(0.001), loss="mean_squared_error")
    return model

def train_model(X_train, y_train, X_val, y_val, input_shape, epochs, batch_size):
    model = build_lstm_model(input_shape)
    early_stop = EarlyStopping(monitor="val_loss", patience=5, restore_best_weights=True)
    reduce_lr = ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=3, min_lr=1e-6)
    model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=epochs, batch_size=batch_size, verbose=1, callbacks=[early_stop, reduce_lr])
    return model

def get_dynamic_params(df, horizon_days, coin_id):
    volatility = df["close_price"].pct_change().std()
    char = crypto_characteristics.get(coin_id, {"volatility": 0.05})
    window_size = min(max(15, int(horizon_days * (1.5 if volatility > char["volatility"] else 1))), len(df) // 3)
    epochs = min(50, max(20, int(len(df) / 100) + int(volatility * 150)))
    return window_size, epochs, 32

# Sentimiento
@st.cache_data(ttl=86400)
def get_fear_greed_index():
    try:
        resp = session.get("https://api.alternative.me/fng/?format=json", timeout=10)
        return float(resp.json()["data"][0]["value"]) if resp.status_code == 200 else 50.0
    except:
        return 50.0

@st.cache_data(ttl=86400)
def get_coingecko_community_activity(coin_id):
    try:
        url = f"https://api.coingecko.com/api/v3/coins/{coinid_to_coingecko[coin_id]}?localization=false&tickers=false&market_data=false&community_data=true"
        resp = session.get(url, timeout=10)
        if resp.status_code == 200:
            comm = resp.json().get("community_data", {})
            activity = max(comm.get("twitter_followers", 0), comm.get("reddit_average_posts_48h", 0) * 1000)
            return min(100, (activity / 20000000) * 100) if activity > 0 else 50.0
        return 50.0
    except:
        return 50.0

def get_crypto_sentiment_combined(coin_id):
    fg = get_fear_greed_index()
    cg = get_coingecko_community_activity(coin_id)
    hist = historical_sentiment.get(coin_id, 50.0)
    return max(0, min(100, 0.6 * fg + 0.3 * cg + 0.1 * hist))

# Modelos alternativos
def train_arima_model(df, test_size=0.2):
    if not ARIMA_AVAILABLE:
        return None, np.nan, np.nan
    train_size = int(len(df) * (1 - test_size))
    train, test = df["close_price"][:train_size], df["close_price"][train_size:]
    model = ARIMA(train, order=(5, 1, 0)).fit()
    preds = model.forecast(steps=len(test))
    return preds, np.sqrt(mean_squared_error(test, preds)), robust_mape(test, preds)

def train_prophet_model(df, horizon_days, test_size=0.2):
    if not PROPHET_AVAILABLE:
        return None, None, np.nan, np.nan
    df_prophet = df.rename(columns={"ds": "ds", "close_price": "y"})
    train_size = int(len(df_prophet) * (1 - test_size))
    train, test = df_prophet[:train_size], df_prophet[train_size:]
    model = Prophet(yearly_seasonality=True, weekly_seasonality=True).fit(train)
    future = model.make_future_dataframe(periods=horizon_days)
    forecast = model.predict(future)
    preds, future_preds = forecast["yhat"].tail(len(test)).values, forecast["yhat"].tail(horizon_days).values
    return preds, future_preds, np.sqrt(mean_squared_error(test["y"], preds)), robust_mape(test["y"], preds)

# Predicción
def train_and_predict(coin_id, horizon_days=5, test_size=0.2):
    df = load_coincap_data(coin_id)
    if df is None or df.empty:
        return None

    symbol = coinid_to_symbol[coin_id]
    sentiment = get_crypto_sentiment_combined(coin_id)
    sentiment_factor = (sentiment + get_fear_greed_index()) / 200.0
    st.write(f"Sentimiento {symbol}: {sentiment:.2f}, Factor: {sentiment_factor:.2f}")

    window_size, epochs, batch_size = get_dynamic_params(df, horizon_days, coin_id)
    scaler = MinMaxScaler()
    scaled_data = scaler.fit_transform(df[["close_price"]])

    split = int(len(scaled_data) * (1 - test_size))
    train_data, test_data = scaled_data[:split], scaled_data[split:]
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

    model = train_model(X_train_adj, y_train, X_val_adj, y_val, (window_size, 2), epochs, batch_size)
    test_preds_scaled = model.predict(X_test_adj, verbose=0)
    test_preds = scaler.inverse_transform(test_preds_scaled)
    y_test_real = scaler.inverse_transform(y_test.reshape(-1, 1))

    arima_preds, arima_rmse, arima_mape = train_arima_model(df) if ARIMA_AVAILABLE else (None, np.nan, np.nan)
    prophet_preds, prophet_future, prophet_rmse, prophet_mape = train_prophet_model(df, horizon_days) if PROPHET_AVAILABLE else (None, None, np.nan, np.nan)

    last_window = np.concatenate([scaled_data[-window_size:].reshape(1, window_size, 1), np.full((1, window_size, 1), sentiment_factor)], axis=-1)
    future_preds_scaled = []
    current_input = last_window
    for _ in range(horizon_days):
        pred = model.predict(current_input, verbose=0)[0][0]
        future_preds_scaled.append(pred)
        current_input = np.append(current_input[:, 1:, :], [[[pred, sentiment_factor]]], axis=1)
    future_preds = scaler.inverse_transform(np.array(future_preds_scaled).reshape(-1, 1)).flatten()

    models = {"LSTM": (test_preds, future_preds, np.sqrt(mean_squared_error(y_test_real, test_preds)), robust_mape(y_test_real, test_preds))}
    if arima_preds is not None:
        models["ARIMA"] = (arima_preds, arima_preds[:len(test_preds)], arima_rmse, arima_mape)
    if prophet_preds is not None:
        models["Prophet"] = (prophet_preds, prophet_future, prophet_rmse, prophet_mape)
    best_model = min(models.items(), key=lambda x: x[1][2])[0]
    return df, *models[best_model], sentiment_factor, symbol, best_model

# App
def main_app():
    st.set_page_config(page_title="Crypto Price Predictions 🔮", layout="wide")
    st.title("Crypto Price Predictions 🔮")
    st.markdown("Predice precios combinando datos de CoinCap y sentimiento de Fear & Greed y CoinGecko.")

    st.sidebar.title("Configuración")
    crypto_name = st.sidebar.selectbox("Criptomoneda:", list(coincap_ids.keys()))
    coin_id = coincap_ids[crypto_name]
    horizon = st.sidebar.slider("Días a predecir:", 1, 60, 5)

    df = load_coincap_data(coin_id)
    if df is not None:
        fig = px.line(df, x="ds", y="close_price", title=f"Histórico de {crypto_name}")
        st.plotly_chart(fig, use_container_width=True)

    tabs = st.tabs(["Entrenamiento", "Predicción", "Sentimiento"])
    with tabs[0]:
        if st.button("Entrenar y Predecir"):
            with st.spinner("Procesando..."):
                result = train_and_predict(coin_id, horizon)
            if result:
                df, test_preds, future_preds, rmse, mape, sentiment_factor, symbol, best_model = result
                st.success(f"Mejor modelo: {best_model} (RMSE: {rmse:.2f}, MAPE: {mape:.2f}%)")
                col1, col2 = st.columns(2)
                col1.metric("RMSE", f"{rmse:.2f}")
                col2.metric("MAPE", f"{mape:.2f}%")
                fig_test = go.Figure()
                test_dates = df["ds"].iloc[-len(test_preds):]
                fig_test.add_trace(go.Scatter(x=test_dates, y=df["close_price"].iloc[-len(test_preds):], mode="lines", name="Real"))
                fig_test.add_trace(go.Scatter(x=test_dates, y=test_preds.flatten(), mode="lines", name="Predicción"))
                st.plotly_chart(fig_test, use_container_width=True)

    with tabs[1]:
        if "result" in locals() and result:
            last_date = df["ds"].iloc[-1]
            future_dates = pd.date_range(start=last_date, periods=horizon+1, freq="D")
            pred_series = np.concatenate(([df["close_price"].iloc[-1]], future_preds))
            fig_future = go.Figure(go.Scatter(x=future_dates, y=pred_series, mode="lines+markers", name="Predicción"))
            fig_future.update_layout(title=f"Predicción ({horizon} días) - {symbol}")
            st.plotly_chart(fig_future, use_container_width=True)
            st.dataframe(pd.DataFrame({"Fecha": future_dates, "Predicción": pred_series}))

    with tabs[2]:
        if "result" in locals() and result:
            sentiment_texts = {
                "BTC": "Bitcoin: Cautela con optimismo moderado (Sentimiento: {:.2f}).",
                "ETH": "Ethereum: Neutral con potencial si el mercado mejora (Sentimiento: {:.2f}).",
                "XRP": "Ripple: Pesimismo leve, atento a regulaciones (Sentimiento: {:.2f})."
            }
            sentiment = result[4] * 200 - get_fear_greed_index()
            text = sentiment_texts.get(symbol, "Sentimiento de {}: {:.2f}").format(symbol, sentiment)
            st.write(text)
            fig_sent = go.Figure([
                go.Bar(x=[symbol], y=[sentiment], name="Sentimiento Combinado"),
                go.Bar(x=[symbol], y=[get_fear_greed_index()], name="Sentimiento Global")
            ])
            fig_sent.update_layout(barmode="group", title=f"Sentimiento de {symbol}", template="plotly_dark")
            st.plotly_chart(fig_sent)

if __name__ == "__main__":
    main_app()