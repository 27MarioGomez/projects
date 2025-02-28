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
from sklearn.metrics import mean_squared_error

# Configuración inicial de certificados SSL y sesión de requests
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
    """Calcula el MAPE de manera robusta evitando división por cero."""
    return np.mean(np.abs((y_true - y_pred) / np.maximum(np.abs(y_true), eps))) * 100

# Carga de datos corregida
@st.cache_data
def load_coincap_data(coin_id):
    """Carga datos históricos de CoinCap para una criptomoneda específica."""
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
        # Manejo robusto de "volumeUsd" como serie de pandas
        if "volumeUsd" in df.columns and not df["volumeUsd"].empty:
            df["volume"] = pd.to_numeric(df["volumeUsd"], errors="coerce").fillna(0.0)
        else:
            df["volume"] = pd.Series(0.0, index=df.index)
        return df[["ds", "close_price", "volume"]].dropna().sort_values("ds").reset_index(drop=True)
    except Exception as e:
        st.error(f"Error al cargar datos: {e}")
        return None

# Secuencias y modelo LSTM
def create_sequences(data, window_size):
    """Crea secuencias para el modelo LSTM."""
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
        LSTM(50, return_sequences=True, input_shape=input_shape),
        Dropout(0.2),
        LSTM(50),
        Dropout(0.2),
        Dense(20, activation="relu"),
        Dense(1)
    ])
    model.compile(optimizer=Adam(learning_rate), loss="mse")
    return model

def train_model(X_train, y_train, X_val, y_val, input_shape, epochs, batch_size):
    """Entrena el modelo LSTM."""
    tf.keras.backend.clear_session()
    model = build_lstm_model(input_shape)
    callbacks = [
        EarlyStopping(patience=5, restore_best_weights=True),
        ReduceLROnPlateau(patience=3, factor=0.5, min_lr=1e-6)
    ]
    model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=epochs, batch_size=batch_size, callbacks=callbacks, verbose=0)
    return model

def get_dynamic_params(df, horizon_days, coin_id):
    """Ajusta parámetros dinámicos según volatilidad y datos."""
    volatility = df["close_price"].pct_change().std()
    base_volatility = crypto_characteristics.get(coin_id, {"volatility": 0.05})["volatility"]
    window_size = min(max(15, int(horizon_days * (1.5 if volatility > base_volatility else 1))), len(df) // 3)
    epochs = min(50, max(20, int(len(df) / 100) + int(volatility * 150)))
    batch_size = 32
    learning_rate = 0.0004 if volatility > base_volatility else 0.0005
    return window_size, epochs, batch_size, learning_rate

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
    """Calcula el sentimiento combinado dinámico."""
    fg = get_fear_greed_index()
    cg = get_coingecko_community_activity(coin_id)
    volatility = crypto_characteristics.get(coin_id, {"volatility": 0.05})["volatility"]
    fg_weight = 0.6 if volatility > 0.07 else 0.5
    cg_weight = 1 - fg_weight
    return fg * fg_weight + cg * cg_weight

# Predicción
def train_and_predict_with_sentiment(coin_id, horizon_days):
    """Entrena y predice combinando modelos y sentimiento."""
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
    lstm_test_preds = scaler.inverse_transform(lstm_test_preds_scaled).flatten()  # Asegurar 1D
    y_test_real = scaler.inverse_transform(y_test.reshape(-1, 1)).flatten()  # Asegurar 1D
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

    test_dates = df["ds"].iloc[-len(lstm_test_preds):].values  # Fechas para el set de test
    real_prices = df["close_price"].iloc[-len(lstm_test_preds):].values  # Precios reales del set de test

    return {
        "df": df,
        "test_preds": lstm_test_preds,
        "future_preds": lstm_future_preds,
        "rmse": lstm_rmse,
        "mape": lstm_mape,
        "sentiment_factor": sentiment_factor,
        "symbol": symbol,
        "crypto_sent": crypto_sent,
        "market_sent": market_sent,
        "future_dates": future_dates,
        "test_dates": test_dates,
        "real_prices": real_prices
    }

# Aplicación principal
def main_app():
    st.set_page_config(page_title="Crypto Price Predictions 🔮", layout="wide")
    st.title("Crypto Price Predictions 🔮")
    st.markdown("""
    **Descripción del Modelo:**  
    Esta plataforma utiliza un modelo avanzado de aprendizaje automático basado en redes LSTM (Long Short-Term Memory) para predecir precios futuros de criptomonedas como Bitcoin, Ethereum, Ripple y otras. El modelo integra datos históricos de precios y volúmenes de CoinCap, abarcando hasta dos años de información diaria, ajustando dinámicamente sus hiperparámetros (como tamaño de ventana, épocas, tamaño de lote y tasa de aprendizaje) según la volatilidad específica de cada criptomoneda. Además, incorpora un análisis de sentimiento dinámico que combina el índice Fear & Greed para el mercado global con la actividad comunitaria en redes sociales (Twitter y Reddit) de CoinGecko para cada cripto, mejorando la precisión al considerar el estado de ánimo del mercado y los inversores. Las predicciones se complementan con métricas clave como RMSE y MAPE para evaluar la precisión, y se presentan en gráficos interactivos y tablas para una experiencia clara y detallada.

    **Fuentes de Datos:**  
    <small>CoinCap, Fear & Greed Index, CoinGecko</small>
    """)

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
                st.success("Entrenamiento y predicción completados!")
                st.write(f"Sentimiento combinado de {result['symbol']}: {result['crypto_sent']:.2f}")
                st.write(f"Sentimiento global del mercado: {result['market_sent']:.2f}")
                st.write(f"Factor combinado: {result['sentiment_factor']:.2f}")
                col1, col2 = st.columns(2)
                col1.metric("RMSE (Test)", f"{result['rmse']:.2f}", help="Error promedio en dólares.")
                col2.metric("MAPE (Test)", f"{result['mape']:.2f}%", help="Error relativo promedio.")

                # Verificación de dimensiones
                if len(result["test_dates"]) != len(result["test_preds"]):
                    st.warning(f"Advertencia: Longitud de test_dates ({len(result['test_dates'])}) no coincide con test_preds ({len(result['test_preds'])}). Ajustando...")
                    min_len = min(len(result["test_dates"]), len(result["test_preds"]))
                    result["test_dates"] = result["test_dates"][:min_len]
                    result["test_preds"] = result["test_preds"][:min_len]
                    result["real_prices"] = result["real_prices"][:min_len]

                # Crear el gráfico
                fig_test = go.Figure()
                fig_test.add_trace(go.Scatter(
                    x=result["test_dates"],
                    y=result["real_prices"],
                    mode="lines",
                    name="Precio Real",
                    line=dict(color="blue")
                ))
                fig_test.add_trace(go.Scatter(
                    x=result["test_dates"],
                    y=result["test_preds"],
                    mode="lines",
                    name="Predicción",
                    line=dict(color="orange", dash="dash")
                ))
                fig_test.update_layout(
                    title=f"Comparación entre el precio real y la predicción: {result['symbol']}",
                    template="plotly_dark",
                    xaxis_title="Fecha",
                    yaxis_title="Precio en USD"
                )
                st.plotly_chart(fig_test, use_container_width=True)
                st.session_state["result"] = result

    with tabs[1]:
        st.header(f"Predicción de Precios - {crypto_name}")
        if "result" in st.session_state:
            # Verificar si result es un diccionario
            if isinstance(st.session_state["result"], dict):
                result = st.session_state["result"]
                last_date = result["df"]["ds"].iloc[-1]
                current_price = result["df"]["close_price"].iloc[-1]
                pred_series = np.concatenate(([current_price], result["future_preds"]))
                fig_future = go.Figure()
                future_dates_display = [last_date] + result["future_dates"]
                fig_future.add_trace(go.Scatter(x=future_dates_display, y=pred_series, mode="lines+markers", name="Predicción"))
                fig_future.update_layout(title=f"Predicción a Futuro ({horizon} días) - {result['symbol']}", template="plotly_dark")
                st.plotly_chart(fig_future, use_container_width=True)
                st.subheader("Valores Numéricos")
                st.dataframe(pd.DataFrame({"Fecha": future_dates_display, "Predicción": pred_series}))
            else:
                st.error("El resultado almacenado no es un diccionario válido. Por favor, entrena el modelo nuevamente.")
        else:
            st.info("Entrena el modelo primero.")

    with tabs[2]:
        st.header("Análisis de Sentimientos")
        if "result" in st.session_state:
            # Verificar si result es un diccionario
            if isinstance(st.session_state["result"], dict):
                result = st.session_state["result"]
                sentiment_texts = {
                    "BTC": f"El sentimiento de Bitcoin está en {result['crypto_sent']:.2f}, lo que muestra cierta cautela entre los inversores, aunque su comunidad sigue activa. El mercado en general está en {result['market_sent']:.2f}, indicando miedo. Con un factor combinado de {result['sentiment_factor']:.2f}, parece que Bitcoin podría mantenerse estable, pero no esperes grandes subidas pronto. ¡Ojo con las noticias!",
                    "ETH": f"Ethereum tiene un sentimiento de {result['crypto_sent']:.2f}, reflejando dudas, pero su tecnología sigue siendo un punto fuerte. El mercado está en {result['market_sent']:.2f}, con miedo dominando. El factor combinado de {result['sentiment_factor']:.2f} sugiere que podría haber oportunidades si el ánimo mejora. Estate atento a sus actualizaciones.",
                    "XRP": f"XRP está en {result['crypto_sent']:.2f}, mostrando pesimismo en su comunidad, y el mercado en {result['market_sent']:.2f} no ayuda mucho. Con un factor combinado de {result['sentiment_factor']:.2f}, parece que XRP podría seguir moviéndose poco a menos que haya noticias grandes, como su caso legal. Cuidado con la volatilidad."
                }
                sentiment_text = sentiment_texts.get(result['symbol'], f"El sentimiento de {result['symbol']} está en {result['crypto_sent']:.2f}, lo que indica {'optimismo' if result['crypto_sent'] > 50 else 'pesimismo'} entre sus seguidores. El mercado general está en {result['market_sent']:.2f}. Con un factor combinado de {result['sentiment_factor']:.2f}, hay {'potencial' if result['sentiment_factor'] > 0.5 else 'cautela'} a corto plazo.")
                st.write(sentiment_text)
                fig_sentiment = go.Figure(data=[
                    go.Bar(name="Sentimiento Combinado", x=[result['symbol']], y=[result['crypto_sent']], marker_color="#1f77b4"),
                    go.Bar(name="Sentimiento Global", x=[result['symbol']], y=[result['market_sent']], marker_color="#ff7f0e")
                ])
                fig_sentiment.update_layout(barmode="group", title=f"Análisis de Sentimiento de {result['symbol']}", template="plotly_dark")
                st.plotly_chart(fig_sentiment, use_container_width=True)
                st.write("**NFA (Not Financial Advice):** Esto es solo información educativa, no un consejo financiero. Consulta a un experto antes de invertir.")
            else:
                st.error("El resultado almacenado no es un diccionario válido. Por favor, entrena el modelo nuevamente.")
        else:
            st.info("Entrena el modelo para ver el análisis.")

if __name__ == "__main__":
    main_app()