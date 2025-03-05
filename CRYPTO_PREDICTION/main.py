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
import yfinance as yf
import requests
import certifi
import os
from sklearn.metrics import mean_squared_error
from textblob import TextBlob
from dateutil.parser import parse as date_parse
from urllib3.util.retry import Retry
from requests.adapters import HTTPAdapter
from newsapi import NewsApiClient

# ------------------------------------------------------------------------------
# Configuración SSL y sesión HTTP
# ------------------------------------------------------------------------------
os.environ["REQUESTS_CA_BUNDLE"] = certifi.where()
session = requests.Session()
retry = Retry(total=5, backoff_factor=1, status_forcelist=[429, 500, 502, 503, 504])
adapter = HTTPAdapter(max_retries=retry)
session.mount("https://", adapter)

# ------------------------------------------------------------------------------
# Diccionarios de criptomonedas
# ------------------------------------------------------------------------------
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
coinid_to_symbol = {v: k.split(" (")[1][:-1] for k, v in coincap_ids.items()}

# Volatilidad predefinida (no se usa directamente, pero puede ayudar a calibrar)
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
# Funciones de utilidad
# ------------------------------------------------------------------------------
def robust_mape(y_true, y_pred, eps=1e-9):
    """Calcula el MAPE evitando división por cero."""
    return np.mean(np.abs((y_true - y_pred) / np.maximum(np.abs(y_true), eps))) * 100

@st.cache_data
def load_crypto_data(coin_id, start_date=None, end_date=None):
    """
    Descarga datos históricos (precio y volumen) de una criptomoneda usando yfinance.
    - Si no se especifica rango, se descarga todo el histórico (period="max").
    """
    ticker_ids = {
        "bitcoin": "BTC-USD",
        "ethereum": "ETH-USD",
        "xrp": "XRP-USD",
        "binance-coin": "BNB-USD",
        "cardano": "ADA-USD",
        "solana": "SOL-USD",
        "dogecoin": "DOGE-USD",
        "polkadot": "DOT-USD",
        "polygon": "MATIC-USD",
        "litecoin": "LTC-USD",
        "tron": "TRX-USD",
        "stellar": "XLM-USD"
    }
    ticker = ticker_ids.get(coin_id)
    if not ticker:
        st.error("No se encontró el ticker para la criptomoneda.")
        return None

    if start_date is None or end_date is None:
        df = yf.download(ticker, period="max", progress=False)
    else:
        df = yf.download(ticker, start=start_date, end=end_date, progress=False)

    if df.empty:
        st.warning("No se obtuvieron datos de yfinance.")
        return None
    
    df = df.reset_index()
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    df.rename(columns={"Date": "ds", "Close": "close_price", "Volume": "volume"}, inplace=True)
    # Nos quedamos con ds, close_price y volume (si existe)
    if "volume" not in df.columns:
        df["volume"] = 0.0  # Fallback si no hay volumen
    return df[["ds", "close_price", "volume"]]

def create_sequences(data, window_size):
    """
    Genera secuencias (X, y) para el modelo LSTM.
    data: array Nx2 (log_price, log_volume)
    """
    if len(data) <= window_size:
        return None, None
    X, y = [], []
    for i in range(window_size, len(data)):
        X.append(data[i - window_size:i])
        y.append(data[i, 0])  # La variable a predecir es la primera columna (log_price)
    return np.array(X), np.array(y)

def build_lstm_model(input_shape,
                     learning_rate=0.0005,
                     l2_lambda=0.01,
                     lstm_units1=128,
                     lstm_units2=64,
                     dropout_rate=0.3,
                     dense_units=100):
    """
    Modelo LSTM con 2 capas, dropout y densas amplias para mayor precisión.
    """
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import LSTM, Dense, Dropout
    from tensorflow.keras.optimizers import Adam
    from tensorflow.keras.regularizers import l2

    model = Sequential([
        LSTM(lstm_units1, return_sequences=True, input_shape=input_shape, kernel_regularizer=l2(l2_lambda)),
        Dropout(dropout_rate),
        LSTM(lstm_units2, kernel_regularizer=l2(l2_lambda)),
        Dropout(dropout_rate),
        Dense(dense_units, activation="relu", kernel_regularizer=l2(l2_lambda)),
        Dense(1)  # Solo predecimos log_price
    ])
    model.compile(optimizer=Adam(learning_rate), loss="mse")
    return model

def train_model(X_train, y_train, X_val, y_val, model, epochs=25, batch_size=32):
    """Entrena el modelo LSTM con EarlyStopping y ReduceLROnPlateau."""
    from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
    tf.keras.backend.clear_session()
    callbacks = [
        EarlyStopping(patience=10, restore_best_weights=True),
        ReduceLROnPlateau(patience=5, factor=0.5, min_lr=1e-6)
    ]
    model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=epochs,
        batch_size=batch_size,
        callbacks=callbacks,
        verbose=0
    )
    return model

@st.cache_data(ttl=3600)
def get_fear_greed_index():
    """Obtiene el índice Fear & Greed."""
    try:
        data = requests.get("https://api.alternative.me/fng/?format=json", timeout=10).json()
        return float(data["data"][0]["value"])
    except Exception:
        st.warning("No se pudo obtener el índice Fear & Greed. Se usará 50.0 por defecto.")
        return 50.0

@st.cache_data(ttl=300)
def get_newsapi_articles(coin_id):
    """
    Obtiene hasta 10 artículos de noticias usando NewsAPI, ordenadas de más reciente a más antigua.
    """
    newsapi_key = st.secrets.get("newsapi_key", "")
    if not newsapi_key:
        st.error("No se encontró la clave 'newsapi_key' en Secrets.")
        return []
    try:
        query = f"{coin_id} crypto"
        newsapi = NewsApiClient(api_key=newsapi_key)
        data = newsapi.get_everything(q=query, language="en", sort_by="relevancy", page_size=10)
        articles = []
        if data.get("articles"):
            for art in data["articles"]:
                image_url = art.get("urlToImage", "")
                title = art.get("title") or "Sin título"
                description = art.get("description") or "Sin descripción"
                pub_date = art.get("publishedAt") or "Fecha no disponible"
                link = art.get("url") or "#"

                # Parsear fecha para ordenar
                try:
                    parsed_date = date_parse(pub_date)
                except:
                    parsed_date = datetime(1970, 1, 1)
                pub_date_str = parsed_date.strftime("%Y-%m-%d %H:%M:%S")

                articles.append({
                    "title": title,
                    "description": description,
                    "pubDate": pub_date_str,
                    "link": link,
                    "image": image_url,
                    "parsed_date": parsed_date
                })

            articles = sorted(articles, key=lambda x: x["parsed_date"], reverse=True)
        return articles
    except Exception as e:
        st.error(f"Error al obtener noticias: {e}")
        return []

def get_news_sentiment(coin_id):
    """Calcula el sentimiento (0..100) a partir de los artículos de NewsAPI."""
    articles = get_newsapi_articles(coin_id)
    if not articles:
        return 50.0
    sentiments = []
    for article in articles:
        text = (article["title"] or "") + " " + (article["description"] or "")
        blob = TextBlob(text)
        polarity = blob.sentiment.polarity
        sentiments.append(50 + (polarity * 50))
    return np.mean(sentiments) if sentiments else 50.0

def get_crypto_sentiment_combined(coin_id):
    """
    Combina el sentimiento de las noticias y Fear & Greed:
    gauge_val = 50 + (news_sent - market_sent)
    """
    news_sent = get_news_sentiment(coin_id)
    market_sent = get_fear_greed_index()
    gauge_val = 50 + (news_sent - market_sent)
    gauge_val = max(0, min(100, gauge_val))
    return news_sent, market_sent, gauge_val

def adjust_predictions_for_sentiment(future_preds, gauge_val):
    """
    Ajusta ligeramente las predicciones según el gauge:
    - Very Bullish (gauge_val > 80): +3%
    - Very Bearish (gauge_val < 20): -3%
    """
    if gauge_val > 80:
        return future_preds * 1.03
    elif gauge_val < 20:
        return future_preds * 0.97
    return future_preds

def train_and_predict_with_sentiment(coin_id, horizon_days, start_date=None, end_date=None):
    """
    Entrena el modelo LSTM (25 épocas) con precios y volumen en log 
    e integra el sentimiento (noticias + Fear & Greed).
    """
    with st.spinner("Esto puede tardar un poco, enseguida estamos..."):
        # 1) Carga de datos
        if start_date and end_date:
            df = load_crypto_data(coin_id, start_date, end_date)
        else:
            df = load_crypto_data(coin_id, None, None)
        if df is None or df.empty:
            st.error("No se pudieron obtener datos históricos.")
            return None

        # 2) Transformación logarítmica de precio y volumen
        df["volume"] = df["volume"].fillna(0.0)
        df["log_price"] = np.log1p(df["close_price"])
        df["log_volume"] = np.log1p(df["volume"] + 1)

        # 3) Sentimiento
        news_sent, market_sent, gauge_val = get_crypto_sentiment_combined(coin_id)
        sentiment_factor = gauge_val / 100.0

        # 4) Hiperparámetros fijos
        window_size = 60
        epochs = 25
        batch_size = 32
        lstm_units1 = 128
        lstm_units2 = 64
        dropout_rate = 0.3
        dense_units = 100
        learning_rate = 0.0005
        l2_lambda = 0.01

        # 5) Escalado
        # Para entrenar, usaremos [log_price, log_volume], y predecimos log_price
        data_array = df[["log_price", "log_volume"]].values
        scaler = MinMaxScaler()
        scaled_data = scaler.fit_transform(data_array)

        # 6) Creación de secuencias
        X, y = create_sequences(scaled_data, window_size)
        if X is None:
            st.error("No hay suficientes datos para crear secuencias.")
            return None

        # 7) División en train, val, test
        split = int(len(X) * 0.8)
        X_train, X_test = X[:split], X[split:]
        y_train, y_test = y[:split], y[split:]
        val_split = int(len(X_train) * 0.9)
        X_val, y_val = X_train[val_split:], y_train[val_split:]
        X_train, y_train = X_train[:val_split], y_train[:val_split]

        # 8) Incorporar factor de sentimiento como tercera dimensión
        # X_train_adj => shape (num_samples, window_size, 3) [log_price, log_volume, sentiment_factor]
        X_train_adj = np.concatenate([X_train, np.full((X_train.shape[0], window_size, 1), sentiment_factor)], axis=-1)
        X_val_adj   = np.concatenate([X_val,   np.full((X_val.shape[0], window_size, 1), sentiment_factor)], axis=-1)
        X_test_adj  = np.concatenate([X_test,  np.full((X_test.shape[0], window_size, 1), sentiment_factor)], axis=-1)

        input_shape = (window_size, 3)

        # 9) Construir y entrenar modelo
        lstm_model = build_lstm_model(
            input_shape=input_shape,
            learning_rate=learning_rate,
            l2_lambda=l2_lambda,
            lstm_units1=lstm_units1,
            lstm_units2=lstm_units2,
            dropout_rate=dropout_rate,
            dense_units=dense_units
        )
        lstm_model = train_model(X_train_adj, y_train, X_val_adj, y_val, lstm_model, epochs, batch_size)

        # 10) Predicción en Test
        preds_test_scaled = lstm_model.predict(X_test_adj, verbose=0)
        # La primera columna (log_price) es la variable objetivo
        # Revertimos la escala
        # Construimos un array Nx2 para revertir: la segunda columna es dummy
        reconst_log = np.concatenate([preds_test_scaled, np.zeros((len(preds_test_scaled), 1))], axis=1)
        reconst_inv = scaler.inverse_transform(reconst_log)
        preds_test_log = reconst_inv[:, 0]
        lstm_test_preds = np.expm1(preds_test_log)

        # y_test => la primera columna del array, hacemos algo similar
        reconst_y_test = np.concatenate([y_test.reshape(-1, 1), np.zeros((len(y_test), 1))], axis=1)
        reconst_y_test_inv = scaler.inverse_transform(reconst_y_test)
        y_test_log = reconst_y_test_inv[:, 0]
        y_test_real = np.expm1(y_test_log)

        lstm_rmse = np.sqrt(mean_squared_error(y_test_real, lstm_test_preds))
        lstm_mape = robust_mape(y_test_real, lstm_test_preds)

        # 11) Predicción futura
        future_preds_log = []
        last_window = scaled_data[-window_size:]
        # Expand dims => shape (1, window_size, 2)
        current_input = np.concatenate([
            last_window.reshape(1, window_size, 2),
            np.full((1, window_size, 1), sentiment_factor)
        ], axis=-1)  # shape (1, window_size, 3)

        for _ in range(horizon_days):
            pred_scaled = lstm_model.predict(current_input, verbose=0)[0][0]
            # Reconstruimos
            reconst_pred = np.array([[pred_scaled, 0.0]])
            reconst_inv = scaler.inverse_transform(reconst_pred)
            pred_log = reconst_inv[0, 0]
            future_preds_log.append(pred_log)
            # Actualizamos
            new_feature = np.copy(current_input[:, -1:, :])
            new_feature[0, 0, 0] = current_input[0, -1, 0]  # log_price or not?
            new_feature[0, 0, 0] = pred_scaled  # Actual log_price escalado
            new_feature[0, 0, 1] = current_input[0, -1, 1]  # keep volume escalado
            new_feature[0, 0, 2] = sentiment_factor
            current_input = np.append(current_input[:, 1:, :], new_feature, axis=1)

        # Convertimos de log a real
        future_preds = np.expm1(np.array(future_preds_log))

        # Ajuste según gauge
        future_preds = adjust_predictions_for_sentiment(future_preds, gauge_val)

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
            "symbol": coinid_to_symbol[coin_id],
            "crypto_sent": news_sent,
            "market_sent": market_sent,
            "gauge_val": gauge_val,
            "future_dates": future_dates,
            "test_dates": test_dates,
            "real_prices": real_prices
        }

def main_app():
    st.set_page_config(page_title="Crypto Price Predictions 🔮", layout="wide")
    st.title("Crypto Price Predictions 🔮")

    # Descripción más elaborada, sin detalles innecesarios
    st.markdown("""
    **Descripción del Dashboard:**  
    Este panel integra **datos históricos** de criptomonedas desde *yfinance* (con precios y volumen), 
    junto con un **análisis de sentimiento** derivado de las noticias (NewsAPI) y el índice **Fear & Greed**. 
    Utilizamos un modelo LSTM con 25 épocas y una **transformación logarítmica** para mejorar la precisión en mercados volátiles.  
    Además, se realizan ajustes adicionales en la predicción según el nivel de confianza (o temor) detectado en el mercado.  
    El objetivo es **aproximar** los precios futuros lo mejor posible, considerando tanto la **historia** de la criptomoneda 
    como el **sentimiento** que generan las noticias y eventos externos que pueden impactar el sector.
    """)

    st.sidebar.title("Configuración de Predicción")
    crypto_name = st.sidebar.selectbox("Seleccione una criptomoneda:", list(coincap_ids.keys()))
    coin_id = coincap_ids[crypto_name]

    use_custom_range = st.sidebar.checkbox("Habilitar rango de fechas", value=False)
    default_end = datetime.utcnow()
    default_start = default_end - timedelta(days=7)

    if use_custom_range:
        start_date = st.sidebar.date_input("Fecha de inicio", default_start.date())
        end_date = st.sidebar.date_input("Fecha de fin", default_end.date())
        if start_date > end_date:
            st.sidebar.error("La fecha de inicio no puede ser posterior a la de fin.")
            return
        if (end_date - start_date).days > 7:
            st.sidebar.warning("El rango excede 7 días. Se ajustará a 7 días.")
            end_date = start_date + timedelta(days=7)
        if start_date > datetime.utcnow().date():
            start_date = datetime.utcnow().date() - timedelta(days=7)
            st.sidebar.warning("La fecha de inicio no puede ser futura. Se ajusta a 7 días atrás.")
        if end_date > datetime.utcnow().date():
            end_date = datetime.utcnow().date()
            st.sidebar.warning("La fecha de fin no puede ser futura. Se ajusta a hoy.")
        end_date_with_offset = end_date + timedelta(days=1)
    else:
        start_date = None
        end_date_with_offset = None

    horizon = st.sidebar.slider("Días a predecir:", 1, 60, 5)
    show_stats = st.sidebar.checkbox("Mostrar estadísticas descriptivas", value=False)

    # Carga de datos
    if start_date is not None and end_date_with_offset is not None:
        df_prices = load_crypto_data(coin_id, start_date, end_date_with_offset)
    else:
        df_prices = load_crypto_data(coin_id, None, None)

    # Sección de histórico
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
    tabs = st.tabs(["Entrenamiento y Test", "Predicción de Precios", "Análisis de Sentimientos", "Noticias Recientes"])

    # --------------------------------------------------------------------------
    # Tab 1: Entrenamiento y Test (sin header duplicado)
    # --------------------------------------------------------------------------
    with tabs[0]:
        # Se elimina el st.header para evitar duplicación
        if st.button("Entrenar Modelo y Predecir"):
            result = train_and_predict_with_sentiment(coin_id, horizon, start_date, end_date_with_offset)
            if result:
                st.success("Entrenamiento y predicción completados!")
                st.write(f"Sentimiento Noticias ({result['symbol']}): {result['crypto_sent']:.2f}")
                st.write(f"Sentimiento Mercado (Fear & Greed): {result['market_sent']:.2f}")
                st.write(f"Gauge Combinado: {result['gauge_val']:.2f}")

                col1, col2 = st.columns(2)
                col1.metric("RMSE (Test)", f"{result['rmse']:.2f}", help="Error medio en USD.")
                col2.metric("MAPE (Test)", f"{result['mape']:.2f}%", help="Error porcentual medio.")

                if not (len(result["test_dates"]) > 0 and len(result["real_prices"]) > 0 and len(result["test_preds"]) > 0):
                    st.error("Datos insuficientes para la gráfica de Test.")
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
                    title=f"Precio Real vs. Predicción ({result['symbol']})",
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
        if "result" in st.session_state and isinstance(st.session_state["result"], dict):
            result = st.session_state["result"]
            if result is not None:
                st.header(f"Predicción de Precios - {crypto_name}")
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

                st.subheader("Resultados Numéricos")
                df_future = pd.DataFrame({"Fecha": future_dates_display, "Predicción": pred_series})
                st.dataframe(df_future.style.format({"Predicción": "{:.2f}"}))
            else:
                st.info("No se obtuvo resultado. Entrene el modelo primero.")
        else:
            st.info("Entrene el modelo primero.")

    # --------------------------------------------------------------------------
    # Tab 3: Análisis de Sentimientos
    # --------------------------------------------------------------------------
    with tabs[2]:
        if "result" in st.session_state and isinstance(st.session_state["result"], dict):
            result = st.session_state["result"]
            if result is not None and "gauge_val" in result:
                st.header("Análisis de Sentimientos")
                crypto_sent = result["crypto_sent"]
                market_sent = result["market_sent"]
                gauge_val = result["gauge_val"]

                if gauge_val < 20:
                    gauge_text = "Very Bearish"
                elif gauge_val < 40:
                    gauge_text = "Bearish"
                elif gauge_val < 60:
                    gauge_text = "Neutral"
                elif gauge_val < 80:
                    gauge_text = "Bullish"
                else:
                    gauge_text = "Very Bullish"

                fig_sentiment = go.Figure(go.Indicator(
                    mode="gauge+number",
                    value=gauge_val,
                    number={'suffix': "", "font": {"size": 36}},
                    gauge={
                        "axis": {"range": [0, 100], "tickwidth": 2, "tickcolor": "#fff"},
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
                    title={"text": f"Sentimiento - {result['symbol']}", "x": 0.5, "xanchor": "center", "font": {"size": 24}},
                    template="plotly_dark",
                    height=400,
                    margin=dict(l=20, r=20, t=80, b=20)
                )
                st.plotly_chart(fig_sentiment, use_container_width=True)

                st.write(f"**Sentimiento Noticias ({result['symbol']}):** {crypto_sent:.2f}")
                st.write(f"**Sentimiento Mercado (Fear & Greed):** {market_sent:.2f}")
                st.write(f"**Gauge Value:** {gauge_val:.2f} → **{gauge_text}**")
                if gauge_val > 50:
                    st.write("**Tendencia:** El sentimiento de la cripto supera al del mercado. Posible escenario bullish.")
                else:
                    st.write("**Tendencia:** El sentimiento de la cripto es igual o inferior al del mercado. Se recomienda precaución.")
            else:
                st.warning("No se obtuvo un resultado válido. Reentrene el modelo.")
        else:
            st.info("Entrene el modelo primero.")

    # --------------------------------------------------------------------------
    # Tab 4: Noticias Recientes (grid, más pequeñas)
    # --------------------------------------------------------------------------
    with tabs[3]:
        symbol = coinid_to_symbol[coin_id]
        st.subheader(f"Últimas noticias sobre {crypto_name} ({symbol})")
        articles = get_newsapi_articles(coin_id)
        if articles:
            st.markdown(
                """
                <style>
                .news-container {
                    display: grid;
                    grid-template-columns: repeat(auto-fit, minmax(220px, 1fr));
                    gap: 1rem;
                }
                .news-item {
                    background-color: #2c2c3e;
                    padding: 0.5rem;
                    border-radius: 5px;
                    border: 1px solid #4a4a6a;
                    display: inline-block;
                    vertical-align: top;
                }
                .news-item img {
                    width: 100%;
                    height: auto;
                    border-radius: 5px;
                    margin-bottom: 0.5rem;
                }
                .news-item h4 {
                    margin: 0 0 0.3rem 0;
                    font-size: 1rem;
                }
                .news-item p {
                    font-size: 0.8rem;
                    margin: 0 0 0.3rem 0;
                }
                </style>
                """,
                unsafe_allow_html=True
            )
            st.markdown("<div class='news-container'>", unsafe_allow_html=True)
            for article in articles:
                image_tag = f"<img src='{article['image']}' alt='Imagen de la noticia' />" if article['image'] else ""
                st.markdown(
                    f"""
                    <div class='news-item'>
                        {image_tag}
                        <h4>{article['title']}</h4>
                        <p><em>{article['pubDate']}</em></p>
                        <p>{article['description']}</p>
                        <p><a href="{article['link']}" target="_blank">Leer más</a></p>
                    </div>
                    """,
                    unsafe_allow_html=True
                )
            st.markdown("</div>", unsafe_allow_html=True)
        else:
            st.warning("No se encontraron noticias recientes o ocurrió un error.")

if __name__ == "__main__":
    main_app()
