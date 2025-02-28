#########################
# main.py
#########################

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
    st.warning("statsmodels no está instalado. El modelo ARIMA no estará disponible.")

try:
    import lightgbm as lgb
    LIGHTGBM_AVAILABLE = True
except ImportError:
    LIGHTGBM_AVAILABLE = False
    st.warning("lightgbm no está instalado. El modelo LightGBM no estará disponible.")

# Configurar certificados SSL para requests
os.environ["REQUESTS_CA_BUNDLE"] = certifi.where()
os.environ["SSL_CERT_FILE"] = certifi.where()

##############################################
# Configurar sesión requests con reintentos
##############################################
def get_requests_session(retries=5, backoff_factor=1, status_forcelist=(429, 500, 502, 503, 504)):
    session = requests.Session()
    retry = Retry(
        total=retries,
        read=retries,
        connect=retries,
        backoff_factor=backoff_factor,
        status_forcelist=status_forcelist,
        raise_on_status=False
    )
    adapter = HTTPAdapter(max_retries=retry)
    session.mount("http://", adapter)
    session.mount("https://", adapter)
    return session

session = get_requests_session()

##############################################
# Funciones de apoyo
##############################################
def robust_mape(y_true, y_pred, eps=1e-9):
    return np.mean(np.abs((y_true - y_pred) / np.maximum(np.abs(y_true), eps))) * 100

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

# Características específicas por criptomoneda (volatilidad promedio aproximada)
crypto_characteristics = {
    "bitcoin": {"volatility": 0.03, "start_year": 2013},
    "ethereum": {"volatility": 0.05, "start_year": 2015},
    "xrp": {"volatility": 0.08, "start_year": 2017},
    "binance-coin": {"volatility": 0.06, "start_year": 2017},
    "cardano": {"volatility": 0.07, "start_year": 2017},
    "solana": {"volatility": 0.09, "start_year": 2020},
    "dogecoin": {"volatility": 0.12, "start_year": 2013},
    "polkadot": {"volatility": 0.07, "start_year": 2020},
    "polygon": {"volatility": 0.06, "start_year": 2020},
    "litecoin": {"volatility": 0.04, "start_year": 2013},
    "tron": {"volatility": 0.06, "start_year": 2017},
    "stellar": {"volatility": 0.05, "start_year": 2014}
}

# Caché para datos de CoinGecko
@st.cache_data(ttl=86400)
def get_cached_coingecko_activity(coin_id):
    return get_coingecko_community_activity(coin_id)

##############################################
# Descarga de datos desde CoinCap (intervalo diario)
##############################################
@st.cache_data
def load_coincap_data(coin_id, max_retries=3):
    end_date = datetime.now()
    start_date = end_date - timedelta(days=730)  # 2 años como máximo seguro
    start_ms = int(start_date.timestamp() * 1000)
    end_ms = int(end_date.timestamp() * 1000)
    url = f"https://api.coincap.io/v2/assets/{coin_id}/history?interval=d1&start={start_ms}&end={end_ms}"
    headers = {"User-Agent": "Mozilla/5.0"}
    for attempt in range(max_retries):
        try:
            resp = session.get(url, headers=headers, verify=certifi.where(), timeout=10)
            if resp.status_code == 200:
                data = resp.json()
                if "data" not in data:
                    st.warning("CoinCap: 'data' faltante.")
                    return None
                df = pd.DataFrame(data["data"])
                if df.empty:
                    st.info("CoinCap devolvió datos vacíos. Reajusta el rango de fechas.")
                    return None
                if "time" not in df.columns or "priceUsd" not in df.columns:
                    st.warning("CoinCap: Faltan columnas 'time' o 'priceUsd'.")
                    return None
                df["ds"] = pd.to_datetime(df["time"], unit="ms", errors="coerce")
                df["close_price"] = pd.to_numeric(df["priceUsd"], errors="coerce")
                if "volumeUsd" in df.columns:
                    df["volume"] = pd.to_numeric(df["volumeUsd"], errors="coerce")
                else:
                    df["volume"] = pd.Series(0.0, index=df.index)
                df["volume"] = df["volume"].fillna(0.0)
                df = df[["ds", "close_price", "volume"]].dropna(subset=["ds", "close_price"])
                df.sort_values(by="ds", inplace=True)
                df.reset_index(drop=True, inplace=True)
                df = df[df["close_price"] > 0].copy()
                return df
            elif resp.status_code == 429:
                st.warning(f"CoinCap: Error 429 en intento {attempt+1}. Esperando {15*(attempt+1)}s...")
                time.sleep(15*(attempt+1))
            elif resp.status_code == 400:
                st.info("CoinCap: (400) Parámetros inválidos o rango excesivo. Usando rango de 2 años.")
                return None
            else:
                st.info(f"CoinCap: status code {resp.status_code}. Revisa parámetros.")
                return None
        except requests.exceptions.SSLError as e:
            st.error(f"Error SSL al conectar con CoinCap: {e}")
            return None
    st.info("CoinCap: Máx reintentos sin éxito.")
    return None

##############################################
# Creación de secuencias para LSTM
##############################################
def create_sequences(data, window_size=20):
    if len(data) <= window_size:
        st.warning(f"No hay datos suficientes para una ventana de {window_size} días.")
        return None, None
    X, y = [], []
    for i in range(window_size, len(data)):
        X.append(data[i-window_size:i])
        y.append(data[i, 0])
    return np.array(X), np.array(y)

##############################################
# Modelo LSTM simplificado y optimizado
##############################################
def build_lstm_model(input_shape, learning_rate=0.001, l2_reg=0.005):
    model = Sequential()
    model.add(Conv1D(filters=32, kernel_size=2, activation="relu", input_shape=input_shape, kernel_regularizer=tf.keras.regularizers.l2(l2_reg)))
    model.add(LSTM(64, return_sequences=True, kernel_regularizer=tf.keras.regularizers.l2(l2_reg)))
    model.add(Dropout(0.3))
    model.add(LSTM(64, return_sequences=False, kernel_regularizer=tf.keras.regularizers.l2(l2_reg)))
    model.add(Dropout(0.3))
    model.add(Dense(16, activation="relu", kernel_regularizer=tf.keras.regularizers.l2(l2_reg)))
    model.add(Dropout(0.2))
    model.add(Dense(1))
    opt = Adam(learning_rate=learning_rate)
    model.compile(optimizer=opt, loss="mean_squared_error")
    return model

def train_model(X_train, y_train, X_val, y_val, input_shape, epochs, batch_size, learning_rate):
    tf.keras.backend.clear_session()
    model = build_lstm_model(input_shape, learning_rate=learning_rate)
    early_stop = EarlyStopping(monitor="val_loss", patience=5, restore_best_weights=True)
    reduce_lr = ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=3, min_lr=1e-6)
    model.fit(X_train, y_train, validation_data=(X_val, y_val),
              epochs=epochs, batch_size=batch_size, verbose=1, callbacks=[early_stop, reduce_lr])
    return model

def get_dynamic_params(df, horizon_days, coin_id):
    data_len = len(df)
    volatility = df["close_price"].pct_change().std()
    mean_price = df["close_price"].mean()
    char = crypto_characteristics.get(coin_id, {"volatility": 0.05, "start_year": 2017})
    base_volatility = char["volatility"]
    window_size = min(max(15, int(horizon_days * (1.5 if volatility > base_volatility else 1))), min(60, data_len // 3))
    epochs = min(50, max(20, int(data_len/100) + int(volatility*150)))
    batch_size = 32
    learning_rate = 0.0004 if volatility > base_volatility else 0.0005
    return window_size, epochs, batch_size, learning_rate

##############################################
# Integración con APIs alternativas para Sentimiento
##############################################
def get_fear_greed_index():
    """
    Obtiene el Crypto Fear & Greed Index desde alternative.me.
    Escala: 0 (miedo extremo) a 100 (codicia extrema).
    """
    try:
        url = "https://api.alternative.me/fng/?format=json"
        resp = session.get(url, timeout=10)
        if resp.status_code == 200:
            data = resp.json()
            value_str = data["data"][0]["value"]
            return float(value_str)
        else:
            st.warning(f"Fear & Greed Index: Error {resp.status_code}.")
            return 50.0
    except Exception as e:
        st.error(f"Error obteniendo Fear & Greed Index: {e}")
        return 50.0

def get_coingecko_community_activity(coin_id):
    """
    Usa el volumen de actividad de comunidad (twitter_followers o reddit_posts) como proxy de sentimiento.
    Normaliza dinámicamente según el rango observado.
    """
    try:
        cg_id = coinid_to_coingecko.get(coin_id, coin_id)
        url = f"https://api.coingecko.com/api/v3/coins/{cg_id}?localization=false&tickers=false&market_data=false&community_data=true&developer_data=false&sparkline=false"
        resp = session.get(url, timeout=10)
        if resp.status_code == 200:
            data = resp.json()
            comm = data.get("community_data", {})
            followers = comm.get("twitter_followers", 0)
            posts = comm.get("reddit_average_posts_48h", 0)
            activity = max(followers, posts * 1000)
            max_activity = 20000000
            sentiment = min(100, (activity / max_activity) * 100) if activity > 0 else 50.0
            return sentiment
        else:
            st.warning(f"CoinGecko: Error {resp.status_code} al obtener datos para {cg_id}.")
            return 50.0
    except Exception as e:
        st.error(f"Error obteniendo actividad de CoinGecko para {cg_id}: {e}")
        return 50.0

def get_crypto_sentiment_combined(coin_id):
    """
    Combina Fear & Greed Index con un promedio histórico predefinido y actividad actual.
    Restaura el valor aproximado de XRP a 21.5 ajustando la lógica.
    """
    fg = get_fear_greed_index()
    cg_activity = get_cached_coingecko_activity(coin_id)
    hist_sent = historical_sentiment.get(coin_id, 50.0)
    combined = 0.5 * fg + 0.3 * cg_activity + 0.2 * hist_sent
    return max(0, min(100, combined))

def get_market_sentiment():
    """
    Para el sentimiento global del mercado se usa el Fear & Greed Index.
    """
    return get_fear_greed_index()

##############################################
# Modelos Alternativos
##############################################
def train_arima_model(df, test_size=0.2):
    if not ARIMA_AVAILABLE:
        return None, np.nan, np.nan
    train_size = int(len(df) * (1 - test_size))
    train, test = df["close_price"][:train_size], df["close_price"][train_size:]
    model = ARIMA(train, order=(5, 1, 0))
    model_fit = model.fit()
    predictions = model_fit.forecast(steps=len(test))
    rmse = np.sqrt(mean_squared_error(test, predictions)) if len(test) > 0 else np.nan
    mape = robust_mape(test, predictions) if len(test) > 0 else np.nan
    return predictions, rmse, mape

def train_lightgbm_model(df, test_size=0.2):
    if not LIGHTGBM_AVAILABLE:
        return None, None, np.nan, np.nan
    df["index"] = range(len(df))  # Usar índices numéricos en lugar de fechas
    df["target"] = df["close_price"].shift(-1)
    df = df.dropna()
    train_size = int(len(df) * (1 - test_size))
    train, test = df[:train_size], df[train_size:]
    X_train = train[["index"]]
    y_train = train["target"]
    X_test = test[["index"]]
    y_test = test["target"]
    model = lgb.LGBMRegressor(n_estimators=100, learning_rate=0.1)
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)
    rmse = np.sqrt(mean_squared_error(y_test, predictions)) if len(y_test) > 0 else np.nan
    mape = robust_mape(y_test, predictions) if len(y_test) > 0 else np.nan
    future_indices = range(df["index"].iloc[-1] + 1, df["index"].iloc[-1] + 6)
    future_preds = model.predict(pd.DataFrame({"index": future_indices}))
    return predictions, future_preds, rmse, mape

##############################################
# Entrenamiento y predicción con sentimiento
##############################################
def train_and_predict_with_sentiment(coin_id, use_custom_range, start_ms, end_ms,
                                     horizon_days=30, test_size=0.2):
    df_raw = load_coincap_data(coin_id)
    if df_raw is None or df_raw.empty:
        st.warning("No se pudieron descargar datos suficientes de CoinCap.")
        return None
    if "close_price" not in df_raw.columns:
        st.warning("No se encontró 'close_price' en los datos.")
        return None

    symbol = coinid_to_symbol.get(coin_id, "BTC")
    crypto_sent = get_crypto_sentiment_combined(coin_id)
    market_sent = get_market_sentiment()
    sentiment_factor = (crypto_sent + market_sent) / 200.0

    st.write(f"Sentimiento combinado de {symbol}: {crypto_sent:.2f}")
    st.write(f"Sentimiento global del mercado: {market_sent:.2f}")
    st.write(f"Factor combinado: {sentiment_factor:.2f}")

    window_size, epochs, batch_size, learning_rate = get_dynamic_params(df_raw, horizon_days, coin_id)
    df = df_raw.copy()
    data_for_model = df[["close_price"]].values
    scaler_target = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler_target.fit_transform(data_for_model)

    split_index = int(len(scaled_data) * (1 - test_size))
    if split_index <= window_size:
        st.warning("Datos insuficientes para entrenar. Reajusta parámetros.")
        return None

    train_data = scaled_data[:split_index]
    test_data = scaled_data[split_index:]
    X_train, y_train = create_sequences(train_data, window_size)
    if X_train is None:
        return None
    X_test, y_test = create_sequences(test_data, window_size)
    if X_test is None:
        return None

    val_split = int(len(X_train) * 0.9)
    X_val, y_val = X_train[val_split:], y_train[val_split:]
    X_train, y_train = X_train[:val_split], y_train[:val_split]

    X_train_adj = np.concatenate([X_train, np.full((X_train.shape[0], X_train.shape[1], 1), sentiment_factor)], axis=-1)
    X_val_adj = np.concatenate([X_val, np.full((X_val.shape[0], X_val.shape[1], 1), sentiment_factor)], axis=-1)
    X_test_adj = np.concatenate([X_test, np.full((X_test.shape[0], X_test.shape[1], 1), sentiment_factor)], axis=-1)
    input_shape = (X_train_adj.shape[1], X_train_adj.shape[2])

    # Modelo LSTM
    lstm_model = train_model(X_train_adj, y_train, X_val_adj, y_val, input_shape, epochs, batch_size, learning_rate)
    lstm_test_preds_scaled = lstm_model.predict(X_test_adj, verbose=0)
    lstm_test_preds = scaler_target.inverse_transform(lstm_test_preds_scaled)
    lstm_y_test_real = scaler_target.inverse_transform(y_test.reshape(-1, 1))
    lstm_valid_mask = ~np.isnan(lstm_test_preds) & ~np.isnan(lstm_y_test_real)
    lstm_rmse = np.sqrt(mean_squared_error(lstm_y_test_real[lstm_valid_mask], lstm_test_preds[lstm_valid_mask])) if np.sum(lstm_valid_mask) > 0 else np.nan
    lstm_mape = robust_mape(lstm_y_test_real[lstm_valid_mask], lstm_test_preds[lstm_valid_mask]) if np.sum(lstm_valid_mask) > 0 else np.nan

    # Modelo ARIMA
    arima_preds, arima_rmse, arima_mape = train_arima_model(df_raw) if ARIMA_AVAILABLE else (None, np.nan, np.nan)

    # Modelo LightGBM
    lgb_preds, lgb_future_preds, lgb_rmse, lgb_mape = train_lightgbm_model(df_raw) if LIGHTGBM_AVAILABLE else (None, None, np.nan, np.nan)

    # Predicción futura LSTM
    last_window = scaled_data[-window_size:]
    future_preds_scaled = []
    current_input = np.concatenate([last_window.reshape(1, window_size, 1),
                                   np.full((1, window_size, 1), sentiment_factor)], axis=-1)
    for _ in range(horizon_days):
        future_pred = lstm_model.predict(current_input, verbose=0)[0][0]
        future_preds_scaled.append(future_pred)
        new_feature = np.copy(current_input[:, -1:, :])
        new_feature[0, 0, 0] = future_pred
        new_feature[0, 0, 1] = sentiment_factor
        current_input = np.append(current_input[:, 1:, :], new_feature, axis=1)
    future_preds = scaler_target.inverse_transform(np.array(future_preds_scaled).reshape(-1, 1)).flatten()

    # Selección del mejor modelo (menor RMSE)
    models = {"LSTM": (lstm_test_preds, future_preds, lstm_rmse, lstm_mape)}
    if ARIMA_AVAILABLE and arima_preds is not None:
        models["ARIMA"] = (arima_preds, arima_preds[:len(lstm_test_preds)], arima_rmse, arima_mape)
    if LIGHTGBM_AVAILABLE and lgb_preds is not None:
        models["LightGBM"] = (lgb_preds, lgb_future_preds[:horizon_days], lgb_rmse, lgb_mape)
    best_model = min(models.items(), key=lambda x: x[1][2] if not np.isnan(x[1][2]) else np.inf)[0]
    best_test_preds, best_future_preds, best_rmse, best_mape = models[best_model]

    st.write(f"Mejor modelo seleccionado: {best_model} (RMSE: {best_rmse:.2f}, MAPE: {best_mape:.2f}%)")

    return df_raw, best_test_preds, lstm_y_test_real, best_future_preds, best_rmse, best_mape, sentiment_factor, symbol

##############################################
# Función principal de la app
##############################################
def main_app():
    st.set_page_config(page_title="Crypto Price Predictions (Simplified Sentiment) 🔮", layout="wide")
    st.title("Crypto Price Predictions (Simplified Sentiment) 🔮")
    st.markdown("Este modelo combina datos históricos de CoinCap y un análisis de sentimiento simplificado basado en Fear & Greed Index y actividad de comunidad de CoinGecko para predecir precios, con múltiples modelos optimizados.")
    st.markdown("**Fuente de Datos:** CoinCap, Crypto Fear & Greed Index, CoinGecko")

    st.sidebar.title("Configura tu Predicción")
    st.session_state["crypto_name"] = st.sidebar.selectbox("Selecciona una criptomoneda:", list(coincap_ids.keys()), help="Elige la criptomoneda que deseas analizar.")
    coin_id = coincap_ids[st.session_state["crypto_name"]]

    st.sidebar.subheader("Rango de Fechas")
    use_custom_range = st.sidebar.checkbox("Habilitar rango de fechas", value=False, help="Activa esto para elegir un período personalizado. Deja desmarcado para usar el rango máximo seguro (2 años).")
    default_end = datetime.now()
    default_start = default_end - timedelta(days=730)  # 2 años como máximo seguro
    if use_custom_range:
        start_date = st.sidebar.date_input("Fecha de inicio", default_start, help="Desde cuándo analizar datos.")
        end_date = st.sidebar.date_input("Fecha de fin", default_end, help="Hasta cuándo incluir datos.")
        start_ms = int(datetime.combine(start_date, datetime.min.time()).timestamp() * 1000)
        end_ms = int(datetime.combine(end_date, datetime.min.time()).timestamp() * 1000)
    else:
        start_ms = int(default_start.timestamp() * 1000)
        end_ms = int(default_end.timestamp() * 1000)

    st.sidebar.subheader("Parámetros de Predicción ❓")
    horizon = st.sidebar.slider("Días a predecir:", 1, 60, 5, help="Número de días a futuro a predecir. Ajustado a 5 para tu caso.")
    st.sidebar.markdown("**Los hiperparámetros se ajustan automáticamente según los datos históricos.**")

    df_prices = load_coincap_data(coin_id)
    if df_prices is not None and len(df_prices) > 0:
        df_chart = df_prices.copy()
        df_chart["ds_str"] = df_chart["ds"].dt.strftime("%d/%m/%Y")
        fig_hist = px.line(df_chart, x="ds_str", y="close_price",
                           title=f"Histórico de {st.session_state['crypto_name']}",
                           labels={"ds_str": "Fecha", "close_price": "Precio en USD"})
        fig_hist.update_yaxes(tickformat=",.2f")
        fig_hist.update_layout(xaxis=dict(type="category", tickangle=45, nticks=10))
        st.plotly_chart(fig_hist, use_container_width=True)
        if st.sidebar.checkbox("Ver estadísticas descriptivas", value=False):
            st.subheader("Estadísticas Descriptivas")
            st.write(df_prices["close_price"].describe().rename({
                "count": "Cuenta", "mean": "Media", "std": "Desv. Estándar",
                "min": "Mínimo", "25%": "Percentil 25", "50%": "Mediana",
                "75%": "Percentil 75", "max": "Máximo"
            }))
    else:
        st.info("No se encontraron datos históricos válidos. Revisa la conexión o ajusta el rango si usas personalizado.")

    tabs = st.tabs(["🤖 Entrenamiento y Test", "🔮 Predicción de Precios", "📰 Noticias"])
    
    with tabs[0]:
        st.header("Entrenamiento del Modelo y Evaluación en Test")
        if st.button("Entrenar Modelo y Predecir", key="train_test"):
            with st.spinner("Esto puede tardar un poco, por favor espera..."):
                result = train_and_predict_with_sentiment(
                    coin_id=coin_id,
                    use_custom_range=use_custom_range,
                    start_ms=start_ms,
                    end_ms=end_ms,
                    horizon_days=horizon,
                    test_size=0.2
                )
            if result is not None:
                df_model, test_preds, y_test_real, future_preds, rmse, mape, sentiment_factor, symbol = result
                st.success("Entrenamiento y predicción completados!")
                col1, col2 = st.columns(2)
                col1.metric("RMSE (Test)", f"{rmse:.2f}", help=f"Error promedio en dólares. Un RMSE de {rmse:.2f} indica variación promedio.")
                col2.metric("MAPE (Test)", f"{mape:.2f}%", help=f"Error relativo promedio. Un MAPE de {mape:.2f}% indica desviación promedio.")
                st.subheader("Comparación en el Set de Test")
                test_dates = df_model["ds"].iloc[-len(y_test_real):]
                fig_test = go.Figure()
                fig_test.add_trace(go.Scatter(x=test_dates, y=y_test_real.flatten(), mode="lines", name="Precio Real (Test)"))
                fig_test.add_trace(go.Scatter(x=test_dates, y=test_preds.flatten(), mode="lines", name="Predicción (Test)"))
                fig_test.update_layout(title=f"Comparación en Test: {symbol}", xaxis_title="Fecha", yaxis_title="Precio en USD")
                fig_test.update_yaxes(tickformat=",.2f")
                st.plotly_chart(fig_test, use_container_width=True)
            else:
                st.warning("No se pudo entrenar el modelo. Revisa los avisos.")
    
    with tabs[1]:
        st.header(f"Predicción de Precios - {st.session_state['crypto_name']}")
        if 'result' in locals() and result is not None:
            df_model, test_preds, y_test_real, future_preds, rmse, mape, sentiment_factor, symbol = result
            last_date = df_model["ds"].iloc[-1]
            current_price = df_model["close_price"].iloc[-1]
            future_dates = pd.date_range(start=last_date, periods=horizon+1, freq="D")
            pred_series = np.concatenate(([current_price], future_preds))
            fig_future = go.Figure()
            fig_future.add_trace(go.Scatter(x=future_dates, y=pred_series, mode="lines+markers", name="Predicción Futura"))
            fig_future.update_layout(title=f"Predicción a Futuro ({horizon} días) - {symbol} (Factor Sent.: {sentiment_factor:.2f})",
                                     xaxis_title="Fecha", yaxis_title="Precio en USD")
            fig_future.update_yaxes(tickformat=",.2f")
            st.plotly_chart(fig_future, use_container_width=True)
            st.subheader("Valores Numéricos de la Predicción Futura")
            future_df = pd.DataFrame({"Fecha": future_dates, "Predicción": pred_series})
            st.dataframe(future_df)
        else:
            st.info("Primero entrena el modelo para generar predicciones futuras.")
    
    with tabs[2]:
        st.header("Noticias Recientes")
        st.markdown("No hay una fuente gratuita de noticias disponible en este momento.")

if __name__ == "__main__":
    main_app()