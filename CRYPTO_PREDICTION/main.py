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
from prophet import Prophet
import talib  # Asegúrate de que TA-Lib esté instalado

# ------------------------------------------------------------------------------
# Configuración SSL y sesión HTTP
# ------------------------------------------------------------------------------
os.environ["REQUESTS_CA_BUNDLE"] = certifi.where()
session = requests.Session()
retry = Retry(total=5, backoff_factor=1, status_forcelist=[429, 500, 502, 503, 504])
adapter = HTTPAdapter(max_retries=retry)
session.mount("https://", adapter)

# ------------------------------------------------------------------------------
# Diccionarios de criptomonedas y sus símbolos
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

# ------------------------------------------------------------------------------
# Función para calcular indicadores técnicos con TA‑Lib
# ------------------------------------------------------------------------------
def compute_indicators(df):
    # Asumimos que df contiene 'close_price', 'high' y 'low'
    df["RSI"] = talib.RSI(df["close_price"], timeperiod=14)
    df["rsi_norm"] = df["RSI"] / 100.0
    df["macd"], df["macdsignal"], df["macdhist"] = talib.MACD(df["close_price"], fastperiod=12, slowperiod=26, signalperiod=9)
    upper, middle, lower = talib.BBANDS(df["close_price"], timeperiod=20)
    df["bollinger_upper"] = upper
    df["bollinger_lower"] = lower
    df["sma50"] = talib.SMA(df["close_price"], timeperiod=50)
    df["atr"] = talib.ATR(df["high"], df["low"], df["close_price"], timeperiod=14)
    # Se pueden añadir más indicadores si se desea
    df.fillna(method="bfill", inplace=True)
    return df

# ------------------------------------------------------------------------------
# Funciones de carga y procesamiento de datos
# ------------------------------------------------------------------------------
@st.cache_data
def load_crypto_data(coin_id, start_date=None, end_date=None):
    """
    Descarga datos históricos de una criptomoneda usando yfinance.
    Incluye High y Low para indicadores técnicos.
    Si no se especifica rango, descarga todo el histórico.
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
    df.rename(columns={"Date": "ds", "Close": "close_price", "Volume": "volume", "High": "high", "Low": "low"}, inplace=True)
    df = compute_indicators(df)
    return df[["ds", "close_price", "volume", "high", "low", "RSI", "rsi_norm", "macd", "atr"]]

def create_sequences(data, window_size):
    """
    Genera secuencias para el modelo LSTM.
    data: array con columnas [log_price, log_volume, rsi_norm, macd, atr]
    y: log_price
    """
    if len(data) <= window_size:
        return None, None
    X, y = [], []
    for i in range(window_size, len(data)):
        X.append(data[i - window_size:i])
        y.append(data[i, 0])
    return np.array(X), np.array(y)

# ------------------------------------------------------------------------------
# Modelo LSTM y entrenamiento
# ------------------------------------------------------------------------------
def build_lstm_model(input_shape,
                     learning_rate=0.0005,
                     l2_lambda=0.01,
                     lstm_units1=128,
                     lstm_units2=64,
                     dropout_rate=0.3,
                     dense_units=100):
    model = Sequential([
        LSTM(lstm_units1, return_sequences=True, input_shape=input_shape, kernel_regularizer=l2(l2_lambda)),
        Dropout(dropout_rate),
        LSTM(lstm_units2, kernel_regularizer=l2(l2_lambda)),
        Dropout(dropout_rate),
        Dense(dense_units, activation="relu", kernel_regularizer=l2(l2_lambda)),
        Dense(1)
    ])
    model.compile(optimizer=Adam(learning_rate), loss="mse")
    return model

def train_model(X_train, y_train, X_val, y_val, model, epochs=25, batch_size=32):
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

# ------------------------------------------------------------------------------
# Modelo Prophet para tendencias a largo plazo
# ------------------------------------------------------------------------------
@st.cache_data
def train_prophet_model(df):
    """
    Entrena un modelo Prophet usando el histórico de precios (logarithmizado).
    """
    df_prophet = df[["ds", "close_price"]].copy()
    df_prophet["y"] = np.log1p(df_prophet["close_price"])
    model = Prophet()
    model.fit(df_prophet)
    return model

# ------------------------------------------------------------------------------
# Ensamble de modelos: LSTM y Prophet
# ------------------------------------------------------------------------------
def ensemble_prediction(lstm_pred, prophet_pred, weight_lstm=0.7):
    """Combina la predicción de LSTM y Prophet con un promedio ponderado."""
    return weight_lstm * lstm_pred + (1 - weight_lstm) * prophet_pred

# ------------------------------------------------------------------------------
# NewsAPI y Sentimiento
# ------------------------------------------------------------------------------
@st.cache_data(ttl=300)
def get_newsapi_articles(coin_id):
    """
    Obtiene hasta 10 artículos de noticias usando NewsAPI.
    Ordena los artículos de más reciente a más antiguo.
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
    Combina el sentimiento de las noticias y el índice Fear & Greed.
    gauge_val = 50 + (news_sent - market_sent)
    """
    news_sent = get_news_sentiment(coin_id)
    market_sent = get_fear_greed_index()
    gauge_val = 50 + (news_sent - market_sent)
    gauge_val = max(0, min(100, gauge_val))
    return news_sent, market_sent, gauge_val

def adjust_predictions_for_sentiment(future_preds, gauge_val):
    """Ajusta las predicciones si el gauge es Very Bullish (>80) o Very Bearish (<20)."""
    if gauge_val > 80:
        return future_preds * 1.03
    elif gauge_val < 20:
        return future_preds * 0.97
    return future_preds

# ------------------------------------------------------------------------------
# Ensamble: Entrenamiento y predicción
# ------------------------------------------------------------------------------
def train_and_predict_with_sentiment(coin_id, horizon_days, start_date=None, end_date=None):
    with st.spinner("Esto puede tardar un poco, enseguida estamos..."):
        # 1) Cargar datos históricos
        df = load_crypto_data(coin_id, start_date, end_date)
        if df is None or df.empty:
            st.error("No se pudieron obtener datos históricos.")
            return None

        # 2) Transformación logarítmica de precio y volumen
        df["log_price"] = np.log1p(df["close_price"])
        df["log_volume"] = np.log1p(df["volume"] + 1)

        # Los indicadores técnicos ya se calcularon en compute_indicators y RSI está normalizado en df["rsi_norm"]
        # Usaremos: log_price, log_volume, rsi_norm, macd y atr.
        # Rellenamos NaN si existen
        df.fillna(method="bfill", inplace=True)

        # 3) Entrenar modelo Prophet para largo plazo
        prophet_model = train_prophet_model(df)

        # 4) Sentimiento
        news_sent, market_sent, gauge_val = get_crypto_sentiment_combined(coin_id)
        sentiment_factor = gauge_val / 100.0

        # 5) Construir array de features para LSTM:
        # Usamos columnas: log_price, log_volume, rsi_norm, macd, atr
        data_array = df[["log_price", "log_volume", "rsi_norm", "macd", "atr"]].values
        scaler = MinMaxScaler()
        scaled_data = scaler.fit_transform(data_array)

        # 6) Crear secuencias
        window_size = 60
        X, y = create_sequences(scaled_data, window_size)
        if X is None:
            st.error("No hay suficientes datos para crear secuencias.")
            return None

        # 7) Dividir datos en train/val/test
        split = int(len(X) * 0.8)
        X_train, X_test = X[:split], X[split:]
        y_train, y_test = y[:split], y[split:]
        val_split = int(len(X_train) * 0.9)
        X_val, y_val = X_train[val_split:], y_train[val_split:]
        X_train, y_train = X_train[:val_split], y_train[:val_split]

        # 8) Incorporar el factor de sentimiento (como columna adicional)
        X_train_adj = np.concatenate([X_train, np.full((X_train.shape[0], window_size, 1), sentiment_factor)], axis=-1)
        X_val_adj   = np.concatenate([X_val,   np.full((X_val.shape[0], window_size, 1), sentiment_factor)], axis=-1)
        X_test_adj  = np.concatenate([X_test,  np.full((X_test.shape[0], window_size, 1), sentiment_factor)], axis=-1)
        input_shape = (window_size, 6)  # 5 features + 1 de sentimiento

        # 9) Construir y entrenar modelo LSTM
        lstm_model = build_lstm_model(
            input_shape=input_shape,
            learning_rate=0.0005,
            l2_lambda=0.01,
            lstm_units1=128,
            lstm_units2=64,
            dropout_rate=0.3,
            dense_units=100
        )
        lstm_model = train_model(X_train_adj, y_train, X_val_adj, y_val, lstm_model, epochs=25, batch_size=32)

        # 10) Predicción en Test (LSTM)
        preds_test_scaled = lstm_model.predict(X_test_adj, verbose=0)
        # Reconstruir para revertir escala; rellenamos columnas dummy para las demás features
        reconst_test = np.concatenate([preds_test_scaled, np.zeros((len(preds_test_scaled), 5))], axis=1)
        reconst_test_inv = scaler.inverse_transform(reconst_test)
        preds_test_log = reconst_test_inv[:, 0]
        lstm_test_preds = np.expm1(preds_test_log)

        reconst_y = np.concatenate([y_test.reshape(-1, 1), np.zeros((len(y_test), 5))], axis=1)
        reconst_y_inv = scaler.inverse_transform(reconst_y)
        y_test_log = reconst_y_inv[:, 0]
        y_test_real = np.expm1(y_test_log)

        lstm_rmse = np.sqrt(mean_squared_error(y_test_real, lstm_test_preds))
        lstm_mape = robust_mape(y_test_real, lstm_test_preds)

        # 11) Predicción futura con LSTM
        future_preds_log = []
        last_window = scaled_data[-window_size:]
        current_input = np.concatenate([
            last_window.reshape(1, window_size, 5),
            np.full((1, window_size, 1), sentiment_factor)
        ], axis=-1)  # shape: (1, window_size, 6)
        for _ in range(horizon_days):
            pred_scaled = lstm_model.predict(current_input, verbose=0)[0][0]
            reconst_future = np.array([[pred_scaled, 0, 0, 0, 0, 0]])
            reconst_future_inv = scaler.inverse_transform(reconst_future)
            pred_log = reconst_future_inv[0, 0]
            future_preds_log.append(pred_log)
            new_feature = np.copy(current_input[:, -1:, :])
            new_feature[0, 0, 0] = pred_scaled
            new_feature[0, 0, 1] = current_input[0, -1, 1]  # volumen
            new_feature[0, 0, 2] = current_input[0, -1, 2]  # RSI
            new_feature[0, 0, 3] = current_input[0, -1, 3]  # MACD
            new_feature[0, 0, 4] = current_input[0, -1, 4]  # ATR
            new_feature[0, 0, 5] = sentiment_factor
            current_input = np.append(current_input[:, 1:, :], new_feature, axis=1)
        lstm_future_preds = np.expm1(np.array(future_preds_log))

        # 12) Predicción con Prophet (sobre todo el histórico)
        df_prophet = df[["ds", "close_price"]].copy()
        df_prophet.rename(columns={"close_price": "y"}, inplace=True)
        df_prophet["y"] = np.log1p(df_prophet["y"])
        prophet_model = Prophet()
        prophet_model.fit(df_prophet)
        future_prophet = prophet_model.make_future_dataframe(periods=horizon_days)
        forecast = prophet_model.predict(future_prophet)
        # Usamos la predicción de los últimos 'horizon_days'
        prophet_preds_log = forecast["yhat"].tail(horizon_days).values
        prophet_preds = np.expm1(prophet_preds_log)

        # 13) Ensamble de predicciones: ponderamos 70% LSTM y 30% Prophet
        future_preds = ensemble_prediction(lstm_future_preds, prophet_preds, weight_lstm=0.7)

        # 14) Ajuste final por sentimiento (si gauge es muy extremo)
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

# ------------------------------------------------------------------------------
# Aplicación Streamlit
# ------------------------------------------------------------------------------
def main_app():
    st.set_page_config(page_title="Crypto Price Predictions 🔮", layout="wide")
    st.title("Crypto Price Predictions 🔮")

    st.markdown("""
    **Descripción del Dashboard:**  
    Este panel integra datos históricos de criptomonedas obtenidos desde *yfinance* y complementados con numerosos indicadores técnicos (precio, volumen, RSI, MACD, ATR, entre otros) – calculados en parte con TA‑Lib.  
    Además, se analiza el sentimiento del mercado mediante la agregación de noticias relevantes (NewsAPI) y el índice **Fear & Greed**.  
    Se entrena un modelo LSTM para predecir precios a corto plazo y se combina con un modelo Prophet para captar tendencias a medio-largo plazo, generando un ensamble de modelos cuya predicción final se obtiene mediante un promedio ponderado.  
    La herramienta ofrece además intervalos de predicción y señales de trading simples para ayudar a tomar decisiones informadas.  
    Este dashboard está diseñado para ser intuitivo y responsive, adaptándose tanto a usuarios técnicos como no técnicos.
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

    if start_date and end_date_with_offset:
        df_prices = load_crypto_data(coin_id, start_date, end_date_with_offset)
    else:
        df_prices = load_crypto_data(coin_id, None, None)

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

    # Pestañas principales
    tabs = st.tabs(["Entrenamiento y Test", "Predicción de Precios", "Análisis de Sentimientos", "Noticias Recientes"])

    # Tab 1: Entrenamiento y Test
    with tabs[0]:
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

    # Tab 2: Predicción de Precios
    with tabs[1]:
        if "result" in st.session_state and isinstance(st.session_state["result"], dict):
            result = st.session_state["result"]
            if result is not None:
                st.header(f"Predicción de Precios - {crypto_name}")
                last_date = result["df"]["ds"].iloc[-1].date()  # Solo la fecha, sin hora
                current_price = result["df"]["close_price"].iloc[-1]
                pred_series = np.concatenate(([current_price], result["future_preds"]))
                fig_future = go.Figure()
                future_dates_display = [last_date] + [fd.date() for fd in result["future_dates"]]
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

    # Tab 3: Análisis de Sentimientos
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
                # Señal de trading simple:
                current_price = result["df"]["close_price"].iloc[-1]
                future_mean = np.mean(result["future_preds"])
                if future_mean > current_price * 1.05 and gauge_val > 80:
                    st.write("**Señal:** Buy (la predicción supera el precio actual en más de un 5% y el sentimiento es Very Bullish)")
                elif future_mean < current_price * 0.95 and gauge_val < 20:
                    st.write("**Señal:** Sell (la predicción es inferior al precio actual en más de un 5% y el sentimiento es Very Bearish)")
                else:
                    st.write("**Señal:** Neutral")
            else:
                st.warning("No se obtuvo un resultado válido. Reentrene el modelo.")
        else:
            st.info("Entrene el modelo primero.")

    # Tab 4: Noticias Recientes en formato grid (4 tarjetas por fila)
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
                    grid-template-columns: repeat(4, 1fr);
                    gap: 1rem;
                }
                .news-item {
                    background-color: #2c2c3e;
                    padding: 0.5rem;
                    border-radius: 5px;
                    border: 1px solid #4a4a6a;
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
                @media (max-width: 1024px) {
                    .news-container {
                        grid-template-columns: repeat(2, 1fr);
                    }
                }
                @media (max-width: 600px) {
                    .news-container {
                        grid-template-columns: 1fr;
                    }
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

    # Opción de descarga de resultados (CSV)
    if "result" in st.session_state:
        result = st.session_state["result"]
        if result:
            df_download = pd.DataFrame({
                "Fecha": result["future_dates"],
                "Predicción": result["future_preds"]
            })
            csv = df_download.to_csv(index=False).encode('utf-8')
            st.download_button(
                label="Descargar predicciones en CSV",
                data=csv,
                file_name="predicciones.csv",
                mime="text/csv"
            )

if __name__ == "__main__":
    main_app()
