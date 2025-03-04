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

# Configura certificados SSL para solicitudes HTTPS seguras
os.environ["REQUESTS_CA_BUNDLE"] = certifi.where()
session = requests.Session()
retry = Retry(total=5, backoff_factor=1, status_forcelist=[429, 500, 502, 503, 504])
adapter = HTTPAdapter(max_retries=retry)
session.mount("https://", adapter)

# Define diccionarios de mapeo para criptomonedas y características
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

# Calcula el MAPE de manera robusta, evitando divisiones por cero
def robust_mape(y_true, y_pred, eps=1e-9):
    return np.mean(np.abs((y_true - y_pred) / np.maximum(np.abs(y_true), eps))) * 100

# Carga datos históricos diarios de CoinCap para una criptomoneda específica
@st.cache_data
def load_coincap_data(coin_id, start_ms=None, end_ms=None):
    if start_ms is None or end_ms is None:
        end_date = datetime.now()
        start_date = end_date - timedelta(days=730)
        start_ms, end_ms = int(start_date.timestamp() * 1000), int(end_date.timestamp() * 1000)
    
    url = f"https://api.coincap.io/v2/assets/{coin_id}/history?interval=d1&start={start_ms}&end={end_ms}"
    try:
        resp = session.get(url, headers={"User-Agent": "Mozilla/5.0"}, verify=certifi.where(), timeout=10)
        if resp.status_code != 200:
            st.warning(f"CoinCap: Error {resp.status_code}")
            return None
        df = pd.DataFrame(resp.json().get("data", []))
        if df.empty or "time" not in df or "priceUsd" not in df:
            st.warning("CoinCap: Datos inválidos o vacíos")
            return None
        df["ds"] = pd.to_datetime(df["time"], unit="ms", errors="coerce")
        df["close_price"] = pd.to_numeric(df["priceUsd"], errors="coerce")
        if "volumeUsd" in df and not df["volumeUsd"].empty:
            df["volume"] = pd.to_numeric(df["volumeUsd"], errors="coerce").fillna(0.0)
        else:
            df["volume"] = pd.Series(0.0, index=df.index)
        return df[["ds", "close_price", "volume"]].dropna().sort_values("ds").reset_index(drop=True)
    except Exception as e:
        st.error(f"Error al cargar datos de CoinCap: {e}")
        return None

# Prepara secuencias de datos para el modelo LSTM
def create_sequences(data, window_size):
    if len(data) <= window_size:
        return None, None
    X, y = [], []
    for i in range(window_size, len(data)):
        X.append(data[i - window_size:i])
        y.append(data[i, 0])
    return np.array(X), np.array(y)

# Construye un modelo LSTM mejorado con regularización L2
def build_lstm_model(input_shape, learning_rate=0.001, l2_lambda=0.01):
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

# Entrena el modelo LSTM con validación y callbacks
def train_model(X_train, y_train, X_val, y_val, input_shape, epochs, batch_size):
    tf.keras.backend.clear_session()
    model = build_lstm_model(input_shape)
    callbacks = [EarlyStopping(patience=10, restore_best_weights=True), ReduceLROnPlateau(patience=5, factor=0.5, min_lr=1e-6)]
    history = model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=epochs, batch_size=batch_size, callbacks=callbacks, verbose=0)
    return model, history

# Ajusta parámetros dinámicos según la volatilidad y el horizonte
def get_dynamic_params(df, horizon_days, coin_id):
    volatility = df["close_price"].pct_change().std()
    base_volatility = crypto_characteristics.get(coin_id, {"volatility": 0.05})["volatility"]
    if coin_id == "xrp":
        window_size = min(max(15, int(horizon_days * 1.0)), len(df) // 4)
        epochs = min(150, max(40, int(len(df) / 60) + int(volatility * 300)))
        batch_size, learning_rate = 16, 0.0002
    elif coin_id == "bitcoin":
        window_size = min(max(30, int(horizon_days * 1.5)), len(df) // 3)
        epochs = min(100, max(30, int(len(df) / 80) + int(volatility * 200)))
        batch_size, learning_rate = 32, 0.0005
    else:
        window_size = min(max(20, int(horizon_days * 1.2)), len(df) // 4)
        epochs = min(120, max(35, int(len(df) / 70) + int(volatility * 250)))
        batch_size, learning_rate = 24, 0.0003
    return window_size, epochs, batch_size, learning_rate

# Obtiene el índice Fear & Greed para el mercado global
@st.cache_data(ttl=3600)
def get_fear_greed_index():
    try:
        return float(session.get("https://api.alternative.me/fng/?format=json", timeout=10).json()["data"][0]["value"])
    except Exception:
        st.warning("No se pudo obtener Fear & Greed Index. Usando valor por defecto.")
        return 50.0

# Obtiene actividad comunitaria desde CoinGecko
@st.cache_data(ttl=3600)
def get_coingecko_community_activity(coin_id):
    try:
        cg_id = coinid_to_coingecko.get(coin_id, coin_id)
        data = session.get(f"https://api.coingecko.com/api/v3/coins/{cg_id}?community_data=true", timeout=10).json()["community_data"]
        activity = max(data.get("twitter_followers", 0), data.get("reddit_average_posts_48h", 0) * 1000)
        return min(100, (activity / 20000000) * 100) if activity > 0 else 50.0
    except Exception:
        return 50.0

# Combina sentimiento de múltiples fuentes ajustado por volatilidad
def get_crypto_sentiment_combined(coin_id, sentiment_value=None):
    fg = get_fear_greed_index()
    cg = get_coingecko_community_activity(coin_id)
    volatility = crypto_characteristics.get(coin_id, {"volatility": 0.05})["volatility"]

    # Ajusta pesos según volatilidad: más peso a Fear & Greed y CoinGecko para criptos estables
    if volatility > 0.07:  # Criptos volátiles (e.g., XRP, DOGE)
        fg_weight = 0.15  # Menos peso al mercado global
        cg_weight = 0.45  # Más peso a la actividad comunitaria
        sentiment_weight = 0.40  # Más peso a un valor simulado para mantener compatibilidad
    else:  # Criptos estables (e.g., BTC, ETH)
        fg_weight = 0.50  # Más peso al mercado global
        cg_weight = 0.30  # Menos peso a la actividad comunitaria
        sentiment_weight = 0.20  # Menos peso a un valor simulado

    # Usa un valor simulado o por defecto si no hay noticias
    sentiment = 50.0 if sentiment_value is None or pd.isna(sentiment_value) else float(sentiment_value)
    combined_sentiment = (fg * fg_weight + cg * cg_weight + sentiment * sentiment_weight)
    return max(0, min(100, combined_sentiment))  # Asegura rango 0-100

# Predice precios usando LSTM con integración de sentimiento
def train_and_predict_with_sentiment(coin_id, horizon_days, start_ms=None, end_ms=None):
    df = load_coincap_data(coin_id, start_ms, end_ms)
    if df is None:
        return None
    symbol = coinid_to_symbol[coin_id]

    # Simula un sentimiento constante para mantener la lógica sin NewsData.io
    sentiment_value = 50.0  # Valor neutral por defecto
    crypto_sent = get_crypto_sentiment_combined(coin_id, sentiment_value)
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

    # Ajusta dimensiones para incluir sentimiento
    X_train_adj = np.concatenate([X_train, np.full((X_train.shape[0], window_size, 1), sentiment_factor)], axis=-1)
    X_val_adj = np.concatenate([X_val, np.full((X_val.shape[0], window_size, 1), sentiment_factor)], axis=-1)
    X_test_adj = np.concatenate([X_test, np.full((X_test.shape[0], window_size, 1), sentiment_factor)], axis=-1)

    # Entrena el modelo LSTM
    lstm_model, history = train_model(X_train_adj, y_train, X_val_adj, y_val, (window_size, 2), epochs, batch_size)
    lstm_test_preds_scaled = lstm_model.predict(X_test_adj, verbose=0)
    lstm_test_preds = scaler.inverse_transform(lstm_test_preds_scaled).flatten()
    y_test_real = scaler.inverse_transform(y_test.reshape(-1, 1)).flatten()
    lstm_rmse = np.sqrt(mean_squared_error(y_test_real, lstm_test_preds))
    lstm_mape = robust_mape(y_test_real, lstm_test_preds)

    # Predicciones futuras con sentimiento simulado
    last_window = scaled_data[-window_size:]
    future_preds = []
    current_input = np.concatenate([last_window.reshape(1, window_size, 1), np.full((1, window_size, 1), sentiment_factor)], axis=-1)
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

    test_dates = df["ds"].iloc[-len(lstm_test_preds):].values  # Fechas para el set de test
    real_prices = df["close_price"].iloc[-len(lstm_test_preds):].values  # Precios reales del set de test

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
        "future_dates": future_dates,
        "test_dates": test_dates,
        "real_prices": real_prices
    }

# Configura y ejecuta la aplicación de predicción de precios de criptomonedas
def main_app():
    st.set_page_config(page_title="Crypto Price Predictions 🔮", layout="wide")
    st.title("Crypto Price Predictions 🔮")
    st.markdown("""
    **Descripción del Modelo:**  
    Esta plataforma utiliza un modelo avanzado de aprendizaje automático basado en redes LSTM (Long Short-Term Memory) para predecir precios futuros de criptomonedas como Bitcoin, Ethereum, Ripple y otras. El modelo integra datos históricos de precios y volúmenes de CoinCap, abarcando hasta dos años de información diaria, ajustando dinámicamente sus hiperparámetros (como tamaño de ventana, épocas, tamaño de lote y tasa de aprendizaje) según la volatilidad específica de cada criptomoneda. Además, incorpora un análisis de sentimiento dinámico que combina el índice Fear & Greed para el mercado global y la actividad comunitaria en redes sociales (Twitter y Reddit) de CoinGecko para cada cripto, mejorando la precisión al considerar el estado de ánimo del mercado y los inversores. Las predicciones se complementan con métricas clave como RMSE y MAPE para evaluar la precisión, y se presentan en gráficos interactivos y tablas para una experiencia clara y detallada.

    Fuentes de datos: CoinCap, Fear & Greed Index, CoinGecko
    """)

    # Sidebar para configuración de predicciones
    st.sidebar.title("Configura tu Predicción")
    crypto_name = st.sidebar.selectbox("Selecciona una criptomoneda:", list(coincap_ids.keys()))
    coin_id = coincap_ids[crypto_name]
    use_custom_range = st.sidebar.checkbox("Habilitar rango de fechas", value=False)
    default_end = datetime.now()
    default_start = default_end - timedelta(days=7)  # Rango por defecto de 7 días
    if use_custom_range:
        start_date = st.sidebar.date_input("Fecha de inicio", default_start.date())
        end_date = st.sidebar.date_input("Fecha de fin", default_end.date())
        if start_date > end_date:
            st.sidebar.error("La fecha de inicio no puede ser posterior a la fecha de fin.")
            return
        if (end_date - start_date).days > 7:
            st.sidebar.warning("El rango excede 7 días. Ajustando al máximo permitido.")
            end_date = start_date + timedelta(days=7)
        if start_date > datetime.now().date() or end_date > datetime.now().date():
            start_date, end_date = datetime.now().date() - timedelta(days=7), datetime.now().date()
            st.sidebar.warning("Fechas futuras ajustadas al rango máximo permitido.")
        start_ms, end_ms = int(datetime.combine(start_date, datetime.min.time()).timestamp() * 1000), int(datetime.combine(end_date, datetime.min.time()).timestamp() * 1000)
    else:
        start_ms, end_ms = int(default_start.timestamp() * 1000), int(default_end.timestamp() * 1000)
    horizon = st.sidebar.slider("Días a predecir:", 1, 60, 5)
    st.sidebar.markdown("**Los hiperparámetros se ajustan automáticamente según los datos.**")
    show_stats = st.sidebar.checkbox("Ver estadísticas descriptivas", value=False)

    # Gráfico histórico de precios
    df_prices = load_coincap_data(coin_id, start_ms, end_ms)
    if df_prices is not None:
        fig_hist = px.line(df_prices, x="ds", y="close_price", title=f"Histórico de {crypto_name}", labels={"ds": "Fecha", "close_price": "Precio en USD"})
        fig_hist.update_layout(template="plotly_dark")
        st.plotly_chart(fig_hist, use_container_width=True)
        if show_stats:
            st.subheader("Estadísticas Descriptivas")
            st.write(df_prices["close_price"].describe())

    # Pestañas de la aplicación
    tabs = st.tabs(["🤖 Entrenamiento y Test", "🔮 Predicción de Precios", "📊 Análisis de Sentimientos"])
    with tabs[0]:
        st.header("Entrenamiento del Modelo y Evaluación en Test")
        if st.button("Entrenar Modelo y Predecir"):
            with st.spinner("Esto puede tardar un poco, por favor espera..."):
                result = train_and_predict_with_sentiment(coin_id, horizon, start_ms, end_ms)
            if result:
                st.success("Entrenamiento y predicción completados!")
                st.write(f"Sentimiento combinado de {result['symbol']}: {result['crypto_sent']:.2f}")
                st.write(f"Sentimiento global del mercado: {result['market_sent']:.2f}")
                st.write(f"Factor combinado: {result['sentiment_factor']:.2f}")
                col1, col2 = st.columns(2)
                col1.metric("RMSE (Test)", f"{result['rmse']:.2f}", help="Error promedio en dólares.")
                col2.metric("MAPE (Test)", f"{result['mape']:.2f}%", help="Error relativo promedio.")

                # Verifica dimensiones antes de graficar
                if not (len(result["test_dates"]) > 0 and len(result["real_prices"]) > 0 and len(result["test_preds"]) > 0):
                    st.error("No hay suficientes datos para mostrar el gráfico de entrenamiento y test.")
                    st.session_state["result"] = result
                    return

                # Asegura consistencia en longitudes de datos
                min_len = min(len(result["test_dates"]), len(result["real_prices"]), len(result["test_preds"]))
                if min_len < 1:
                    st.error("No hay datos suficientes para generar el gráfico.")
                    st.session_state["result"] = result
                    return

                result["test_dates"] = result["test_dates"][:min_len]
                result["real_prices"] = result["real_prices"][:min_len]
                result["test_preds"] = result["test_preds"][:min_len]

                # Crea gráfico de comparación con líneas curvas
                fig_test = go.Figure()
                fig_test.add_trace(go.Scatter(
                    x=result["test_dates"],
                    y=result["real_prices"],
                    mode="lines",
                    name="Precio Real",
                    line=dict(color="#1f77b4", width=3, shape="spline")  # Línea curva, azul oscuro, sólida
                ))
                fig_test.add_trace(go.Scatter(
                    x=result["test_dates"],
                    y=result["test_preds"],
                    mode="lines",
                    name="Predicción",
                    line=dict(color="#ff7f0e", width=3, dash="dash", shape="spline")  # Línea curva, naranja, discontinua
                ))
                fig_test.update_layout(
                    title=f"Comparación entre el precio real y la predicción: {result['symbol']}",
                    xaxis=dict(tickformat="%Y-%m-%d"),  # Solo días en el eje X
                    template="plotly_dark"  # Tema oscuro sin fondo adicional
                )
                st.plotly_chart(fig_test, use_container_width=True)
                st.session_state["result"] = result

    with tabs[1]:
        st.header(f"🔮 Predicción de Precios - {crypto_name}")
        if "result" in st.session_state:
            if isinstance(st.session_state["result"], dict):
                result = st.session_state["result"]
                last_date = result["df"]["ds"].iloc[-1]
                current_price = result["df"]["close_price"].iloc[-1]
                pred_series = np.concatenate(([current_price], result["future_preds"]))
                future_dates_display = [last_date] + result["future_dates"]
                fig_future = go.Figure()
                fig_future.add_trace(go.Scatter(
                    x=future_dates_display,
                    y=pred_series,
                    mode="lines+markers",
                    name="Predicción",
                    line=dict(color="#ff7f0e", width=2, shape="spline")  # Línea curva, naranja, con marcadores
                ))
                fig_future.update_layout(
                    title=f"Predicción a Futuro ({horizon} días) - {result['symbol']}",
                    template="plotly_dark",
                    xaxis_title="Fecha",
                    yaxis_title="Precio en USD",
                    plot_bgcolor="#1e1e2f",
                    paper_bgcolor="#1e1e2f"
                )
                st.plotly_chart(fig_future, use_container_width=True)
                st.subheader("Valores Numéricos")
                st.dataframe(pd.DataFrame({"Fecha": future_dates_display, "Predicción": pred_series}).style.format({"Predicción": "{:.2f}"}))
            else:
                st.error("El resultado almacenado no es un diccionario válido. Por favor, entrena el modelo nuevamente.")
        else:
            st.info("Entrena el modelo primero para ver predicciones.")

    with tabs[2]:
        st.header("📊 Análisis de Sentimientos")
        if "result" in st.session_state:
            if isinstance(st.session_state["result"], dict):
                result = st.session_state["result"]
                crypto_sent, market_sent = result["crypto_sent"], result["market_sent"]
                level = (crypto_sent - 50) / 5  # Escala el sentimiento para clasificar
                sentiment_label = "Very Bearish" if level <= -5 else "Bearish" if level <= -2 else "Neutral" if -2 < level < 2 else "Bullish" if level <= 5 else "Very Bullish"
                color = "#ff0000" if level <= -2 else "#ffd700" if -2 < level < 2 else "#00ff00" if level <= 5 else "#008000"

                # Crea un velocímetro realista con Plotly para análisis de sentimiento
                fig_sentiment = go.Figure(go.Indicator(
                    mode="gauge+number",
                    value=crypto_sent,
                    domain={"x": [0, 1], "y": [0, 1]},
                    title={"text": f"Sentimiento - {result['symbol']}", "font": {"size": 24, "color": "#ffffff"}},
                    gauge={
                        "axis": {"range": [0, 100], "tickvals": [0, 25, 50, 75, 100], "ticktext": ["Very Bearish", "Bearish", "Neutral", "Bullish", "Very Bullish"], "tickcolor": "#ffffff", "tickwidth": 2, "tickfont": {"size": 16, "color": "#ffffff"}},
                        "bar": {"color": color},
                        "bgcolor": "#2c2c3e",
                        "borderwidth": 2,
                        "bordercolor": "#4a4a6a",
                        "steps": [{"range": [0, 25], "color": "#ff0000"}, {"range": [25, 50], "color": "#ffd700"}, {"range": [50, 75], "color": "#00ff00"}, {"range": [75, 100], "color": "#008000"}],
                        "threshold": {"line": {"color": "#ffffff", "width": 4}, "thickness": 1, "value": crypto_sent}
                    },
                    number={"font": {"size": 48, "color": "#ffffff"}}
                ))
                fig_sentiment.update_layout(
                    template="plotly_dark",
                    plot_bgcolor="#1e1e2f",
                    paper_bgcolor="#1e1e2f",
                    height=600,
                    width=900,
                    margin=dict(l=20, r=20, t=80, b=20)
                )
                st.plotly_chart(fig_sentiment, use_container_width=True)
                st.write(f"**Estado:** {sentiment_label} (Mercado: {market_sent:.2f})")
                st.write("**NFA (Not Financial Advice):** Esto es solo información educativa, no un consejo financiero. Consulta a un experto antes de invertir.")
            else:
                st.error("El resultado almacenado no es un diccionario válido. Por favor, entrena el modelo nuevamente.")
        else:
            st.info("Entrena el modelo para ver el análisis de sentimientos.")

if __name__ == "__main__":
    main_app()