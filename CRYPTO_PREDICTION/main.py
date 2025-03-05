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
import socket
from urllib3.util.retry import Retry
from requests.adapters import HTTPAdapter
import keras_tuner as kt
import tweepy
from newsapi import NewsApiClient

# ------------------------------------------------------------------------------
# Configuración inicial: Certificados SSL y sesión HTTP
# ------------------------------------------------------------------------------
os.environ["REQUESTS_CA_BUNDLE"] = certifi.where()
session = requests.Session()
retry = Retry(total=5, backoff_factor=1, status_forcelist=[429, 500, 502, 503, 504])
adapter = HTTPAdapter(max_retries=retry)
session.mount("https://", adapter)

# ------------------------------------------------------------------------------
# Identificadores y símbolos de criptomonedas
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
coinid_to_coingecko = {v: v if v != "xrp" else "ripple" for v in coincap_ids.values()}

# ------------------------------------------------------------------------------
# Características predefinidas de volatilidad
# ------------------------------------------------------------------------------
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
    """Calcula el Error Porcentual Absoluto Medio (MAPE) evitando división por cero."""
    return np.mean(np.abs((y_true - y_pred) / np.maximum(np.abs(y_true), eps))) * 100

# Cargar datos históricos usando yfinance
@st.cache_data
def load_crypto_data(coin_id, start_date, end_date):
    """
    Descarga datos históricos de una criptomoneda mediante yfinance.
    Se utiliza el ticker correspondiente (ej. "BTC-USD").
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
        st.error("Ticker no encontrado para la criptomoneda.")
        return None
    df = yf.download(ticker, start=start_date, end=end_date)
    if df.empty:
        st.warning("No se obtuvieron datos de yfinance.")
        return None
    df = df.reset_index()
    df.rename(columns={"Date": "ds", "Close": "close_price"}, inplace=True)
    df = df[["ds", "close_price"]]
    return df

def create_sequences(data, window_size):
    """Genera secuencias de datos para el modelo LSTM."""
    if len(data) <= window_size:
        return None, None
    X, y = [], []
    for i in range(window_size, len(data)):
        X.append(data[i - window_size:i])
        y.append(data[i, 0])
    return np.array(X), np.array(y)

# ------------------------------------------------------------------------------
# Modelado: LSTM y ajuste dinámico de hiperparámetros
# ------------------------------------------------------------------------------
def build_lstm_model(input_shape, learning_rate=0.001, l2_lambda=0.01,
                     lstm_units1=100, lstm_units2=80, dropout_rate=0.3, dense_units=50):
    """Construye un modelo LSTM con regularización L2 y dropout."""
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

def train_model(X_train, y_train, X_val, y_val, model, epochs, batch_size):
    """Entrena el modelo LSTM con early stopping y reducción de la tasa de aprendizaje."""
    tf.keras.backend.clear_session()
    callbacks = [
        EarlyStopping(patience=10, restore_best_weights=True),
        ReduceLROnPlateau(patience=5, factor=0.5, min_lr=1e-6)
    ]
    history = model.fit(X_train, y_train, validation_data=(X_val, y_val),
                        epochs=epochs, batch_size=batch_size,
                        callbacks=callbacks, verbose=0)
    return model, history

def get_dynamic_params(df, horizon_days, coin_id):
    """
    Ajusta dinámicamente los hiperparámetros del modelo en función de la volatilidad y datos históricos.
    Devuelve un diccionario con hiperparámetros.
    """
    real_volatility = df["close_price"].pct_change().std()
    base_volatility = crypto_characteristics.get(coin_id, {"volatility": 0.05})["volatility"]
    combined_volatility = (real_volatility + base_volatility) / 2.0

    window_size = int(max(15, min(60, horizon_days * (1.0 + combined_volatility * 5))))
    epochs = int(max(40, min(250, (len(df) / 70) + combined_volatility * 400)))
    batch_size = int(max(16, min(64, (combined_volatility * 500) + 16)))
    lstm_units1 = int(max(50, min(200, 100 + (combined_volatility * 400))))
    lstm_units2 = int(max(30, min(150, 80 + (combined_volatility * 200))))
    dropout_rate = 0.3 if combined_volatility < 0.1 else 0.4
    dense_units = int(max(30, min(100, 50 + (combined_volatility * 100))))
    learning_rate = 0.0005 if combined_volatility < 0.08 else 0.0002
    l2_lambda = 0.01 if combined_volatility < 0.07 else 0.02

    return {
        "window_size": window_size,
        "epochs": epochs,
        "batch_size": batch_size,
        "lstm_units1": lstm_units1,
        "lstm_units2": lstm_units2,
        "dropout_rate": dropout_rate,
        "dense_units": dense_units,
        "learning_rate": learning_rate,
        "l2_lambda": l2_lambda
    }

# ------------------------------------------------------------------------------
# Datos de sentimiento del mercado
# ------------------------------------------------------------------------------
@st.cache_data(ttl=3600)
def get_fear_greed_index():
    """Obtiene el índice Fear & Greed del mercado."""
    try:
        data = session.get("https://api.alternative.me/fng/?format=json", timeout=10).json()
        return float(data["data"][0]["value"])
    except Exception:
        st.warning("No se pudo obtener el índice Fear & Greed. Se usará 50.0 por defecto.")
        return 50.0

@st.cache_data(ttl=3600)
def get_coingecko_community_activity(coin_id):
    """Obtiene la actividad comunitaria desde CoinGecko."""
    try:
        cg_id = coinid_to_coingecko.get(coin_id, coin_id)
        data = session.get(f"https://api.coingecko.com/api/v3/coins/{cg_id}?community_data=true", timeout=10).json()["community_data"]
        activity = max(data.get("twitter_followers", 0), data.get("reddit_average_posts_48h", 0) * 1000)
        return min(100, (activity / 20000000) * 100) if activity > 0 else 50.0
    except Exception:
        return 50.0

# ------------------------------------------------------------------------------
# Integración con la API de Twitter mediante Tweepy
# ------------------------------------------------------------------------------
@st.cache_data(ttl=300)
def get_twitter_news(coin_symbol):
    """
    Obtiene tweets recientes utilizando Tweepy (Twitter API v2).
    Se buscan tweets que contengan el símbolo de la criptomoneda y "crypto", se excluyen retweets y se filtra por inglés.
    """
    bearer_token = st.secrets.get("twitter_bearer")
    if not bearer_token:
        st.error("No se encontró el token 'twitter_bearer' en Secrets.")
        return []
    try:
        client = tweepy.Client(bearer_token=bearer_token)
        query = f"{coin_symbol} crypto -is:retweet lang:en"
        response = client.search_recent_tweets(query=query, tweet_fields=["created_at", "text", "id"], max_results=10)
        tweets = []
        if response.data:
            for tweet in response.data:
                tweets.append({
                    "text": tweet.text,
                    "pubDate": tweet.created_at.strftime("%Y-%m-%d %H:%M:%S") if tweet.created_at else "",
                    "link": f"https://twitter.com/i/web/status/{tweet.id}"
                })
        return tweets
    except Exception as e:
        st.error(f"Error al obtener tweets: {e}")
        return []

def get_twitter_sentiment(coin_symbol):
    """
    Calcula el sentimiento promedio a partir de los tweets obtenidos mediante Tweepy.
    Retorna un puntaje en el rango [0, 100].
    """
    tweets = get_twitter_news(coin_symbol)
    if not tweets:
        return 50.0
    sentiments = []
    for tweet in tweets:
        blob = TextBlob(tweet["text"])
        sentiment = blob.sentiment.polarity
        sentiment_score = 50 + (sentiment * 50)
        sentiments.append(sentiment_score)
    return np.mean(sentiments) if sentiments else 50.0

def get_crypto_sentiment_combined(coin_id):
    """
    Combina el sentimiento de la criptomoneda (extraído de Twitter) con el índice Fear & Greed
    para calcular un valor gauge:
        gauge_val = 50 + (crypto_sent - market_sent)
    El valor se ajusta al rango [0, 100].
    """
    symbol = coinid_to_symbol[coin_id]
    crypto_sent = get_twitter_sentiment(symbol)
    market_sent = get_fear_greed_index()
    gauge_val = 50 + (crypto_sent - market_sent)
    gauge_val = max(0, min(100, gauge_val))
    return crypto_sent, market_sent, gauge_val

# ------------------------------------------------------------------------------
# Integración con NewsAPI para obtener noticias gratuitas
# ------------------------------------------------------------------------------
@st.cache_data(ttl=300)
def get_newsapi_articles(coin_symbol):
    """
    Obtiene artículos de noticias recientes utilizando NewsAPI.
    Se buscan artículos en inglés ordenados por relevancia.
    """
    newsapi_key = st.secrets.get("newsapi_key")
    if not newsapi_key:
        st.error("No se encontró la clave 'newsapi_key' en Secrets.")
        return []
    try:
        newsapi = NewsApiClient(api_key=newsapi_key)
        response = newsapi.get_everything(q=coin_symbol,
                                          language="en",
                                          sort_by="relevancy",
                                          page_size=10)
        articles = []
        if response.get("articles"):
            for art in response["articles"]:
                articles.append({
                    "title": art.get("title", "Sin título"),
                    "description": art.get("description", ""),
                    "pubDate": art.get("publishedAt", "")[:19].replace("T", " "),
                    "link": art.get("url", "#")
                })
        return articles
    except Exception as e:
        st.error(f"Error al obtener noticias: {e}")
        return []

# ------------------------------------------------------------------------------
# Keras Tuner (Hyperband)
# ------------------------------------------------------------------------------
def build_model_tuner(input_shape):
    """
    Función modelo para Keras Tuner. Ajusta los hiperparámetros del modelo LSTM utilizando Hyperband.
    """
    def model_builder(hp):
        lstm_units1 = hp.Int('lstm_units1', min_value=50, max_value=200, step=50, default=100)
        lstm_units2 = hp.Int('lstm_units2', min_value=30, max_value=150, step=30, default=60)
        dropout_rate = hp.Float('dropout_rate', min_value=0.2, max_value=0.5, step=0.1, default=0.3)
        dense_units = hp.Int('dense_units', min_value=30, max_value=100, step=10, default=50)
        learning_rate = hp.Float('learning_rate', min_value=1e-4, max_value=1e-3, sampling='log', default=1e-4)
        l2_lambda = hp.Float('l2_lambda', min_value=1e-4, max_value=2e-2, sampling='log', default=1e-2)

        model = Sequential([
            LSTM(lstm_units1, return_sequences=True, input_shape=input_shape, kernel_regularizer=l2(l2_lambda)),
            Dropout(dropout_rate),
            LSTM(lstm_units2, kernel_regularizer=l2(l2_lambda)),
            Dropout(dropout_rate),
            Dense(dense_units, activation='relu', kernel_regularizer=l2(l2_lambda)),
            Dense(1)
        ])
        model.compile(optimizer=Adam(learning_rate), loss='mse')
        return model
    return model_builder

# ------------------------------------------------------------------------------
# Pipeline de Entrenamiento y Predicción
# ------------------------------------------------------------------------------
def train_and_predict_with_sentiment(coin_id, horizon_days, start_date, end_date, tune=False):
    """
    Entrena el modelo LSTM y realiza predicciones futuras integrando el factor de sentimiento (gauge_val/100)
    en cada timestep. Si 'tune' es True, se optimizan los hiperparámetros con Keras Tuner (Hyperband).
    """
    df = load_crypto_data(coin_id, start_date, end_date)
    if df is None or df.empty:
        return None

    symbol = coinid_to_symbol[coin_id]
    crypto_sent, market_sent, gauge_val = get_crypto_sentiment_combined(coin_id)
    sentiment_factor = gauge_val / 100.0

    params = get_dynamic_params(df, horizon_days, coin_id)
    window_size = params["window_size"]
    epochs = params["epochs"]
    batch_size = params["batch_size"]
    lstm_units1 = params["lstm_units1"]
    lstm_units2 = params["lstm_units2"]
    dropout_rate = params["dropout_rate"]
    dense_units = params["dense_units"]
    learning_rate = params["learning_rate"]
    l2_lambda = params["l2_lambda"]

    scaler = MinMaxScaler()
    scaled_data = scaler.fit_transform(df[["close_price"]])
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
    X_val_adj   = np.concatenate([X_val,   np.full((X_val.shape[0], window_size, 1), sentiment_factor)], axis=-1)
    X_test_adj  = np.concatenate([X_test,  np.full((X_test.shape[0], window_size, 1), sentiment_factor)], axis=-1)

    input_shape = (window_size, 2)

    if tune:
        tuner = kt.Hyperband(
            build_model_tuner(input_shape),
            objective='val_loss',
            max_epochs=50,
            factor=3,
            directory='kt_dir',
            project_name='crypto_prediction_hb'
        )
        stop_early = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
        tuner.search(X_train_adj, y_train, validation_data=(X_val_adj, y_val), epochs=50,
                     batch_size=batch_size, callbacks=[stop_early], verbose=0)
        lstm_model = tuner.get_best_models(num_models=1)[0]
    else:
        lstm_model = build_lstm_model(
            input_shape=input_shape,
            learning_rate=learning_rate,
            l2_lambda=l2_lambda,
            lstm_units1=lstm_units1,
            lstm_units2=lstm_units2,
            dropout_rate=dropout_rate,
            dense_units=dense_units
        )
        lstm_model, history = train_model(X_train_adj, y_train, X_val_adj, y_val, lstm_model, epochs, batch_size)

    lstm_test_preds_scaled = lstm_model.predict(X_test_adj, verbose=0)
    lstm_test_preds = scaler.inverse_transform(lstm_test_preds_scaled).flatten()
    y_test_real = scaler.inverse_transform(y_test.reshape(-1, 1)).flatten()
    lstm_rmse = np.sqrt(mean_squared_error(y_test_real, lstm_test_preds))
    lstm_mape = robust_mape(y_test_real, lstm_test_preds)

    last_window = scaled_data[-window_size:]
    future_preds = []
    current_input = np.concatenate([last_window.reshape(1, window_size, 1),
                                      np.full((1, window_size, 1), sentiment_factor)], axis=-1)
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
    test_dates = df["ds"].iloc[-len(lstm_test_preds):].values
    real_prices = df["close_price"].iloc[-len(lstm_test_preds):].values

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
        "gauge_val": gauge_val,
        "future_dates": future_dates,
        "test_dates": test_dates,
        "real_prices": real_prices
    }

# ------------------------------------------------------------------------------
# Aplicación Streamlit
# ------------------------------------------------------------------------------
def main_app():
    st.set_page_config(page_title="Predicción de Precios Cripto 🔮", layout="wide")
    st.title("Predicción de Precios Cripto 🔮")
    st.markdown("""
    **Descripción del Modelo:**  
    Esta aplicación utiliza un avanzado modelo LSTM para predecir precios futuros de criptomonedas (por ejemplo, Bitcoin, Ethereum, Ripple).  
    Se ajustan dinámicamente los hiperparámetros (con optimización opcional mediante Keras Tuner Hyperband) según la volatilidad y datos históricos obtenidos con yfinance.  
    Además, se estima el sentimiento del mercado combinando el índice Fear & Greed con el sentimiento extraído de tweets (usando Tweepy)  
    y se obtienen noticias recientes mediante NewsAPI. Las predicciones se evalúan con RMSE y MAPE, y se visualizan en gráficos interactivos.
    """)

    # Configuración en la barra lateral
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
        start_date = default_start
        end_date_with_offset = default_end + timedelta(days=1)

    optimize_hp = st.sidebar.checkbox("Optimizar hiperparámetros (Keras Tuner Hyperband)", value=False)
    horizon = st.sidebar.slider("Días a predecir:", 1, 60, 5)
    show_stats = st.sidebar.checkbox("Mostrar estadísticas descriptivas", value=False)

    # Gráfico histórico de precios (yfinance)
    df_prices = load_crypto_data(coin_id, start_date, end_date_with_offset)
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

    tabs = st.tabs(["🤖 Entrenamiento y Test", "🔮 Predicción de Precios", "📊 Análisis de Sentimientos", "📰 Noticias Recientes"])

    # Tab 1: Entrenamiento y Test
    with tabs[0]:
        st.header("Entrenamiento y Evaluación en Test")
        if st.button("Entrenar Modelo y Predecir"):
            with st.spinner("Entrenando el modelo..."):
                result = train_and_predict_with_sentiment(coin_id, horizon, start_date, end_date_with_offset, tune=optimize_hp)
            if result:
                st.success("¡Entrenamiento y predicción completados!")
                st.write(f"Sentimiento Cripto ({result['symbol']}): {result['crypto_sent']:.2f}")
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

    # Tab 2: Predicción de Precios
    with tabs[1]:
        st.header(f"Predicción de Precios - {crypto_name}")
        if "result" in st.session_state and isinstance(st.session_state["result"], dict):
            result = st.session_state["result"]
            if result is not None:
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

    # Tab 3: Análisis de Sentimientos
    with tabs[2]:
        st.header("Análisis de Sentimientos")
        if "result" in st.session_state:
            if isinstance(st.session_state["result"], dict):
                result = st.session_state["result"]
                if result is None or "gauge_val" not in result:
                    st.warning("No se obtuvo un resultado válido. Reentrene el modelo.")
                else:
                    crypto_sent = result["crypto_sent"]
                    market_sent = result["market_sent"]
                    gauge_val = result["gauge_val"]
                    if gauge_val < 20:
                        gauge_text = "Muy Bearish"
                    elif gauge_val < 40:
                        gauge_text = "Bearish"
                    elif gauge_val < 60:
                        gauge_text = "Neutral"
                    elif gauge_val < 80:
                        gauge_text = "Bullish"
                    else:
                        gauge_text = "Muy Bullish"

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
                    st.write(f"**Sentimiento Cripto ({result['symbol']}):** {crypto_sent:.2f}")
                    st.write(f"**Sentimiento Mercado (Fear & Greed):** {market_sent:.2f}")
                    st.write(f"**Gauge Value:** {gauge_val:.2f} → **{gauge_text}**")
                    if gauge_val > 50:
                        st.write("**Tendencia:** El sentimiento de la cripto supera al del mercado. Posible escenario bullish.")
                    else:
                        st.write("**Tendencia:** El sentimiento de la cripto es igual o inferior al del mercado. Se recomienda precaución.")
            else:
                st.error("Datos de resultado inválidos. Reentrene el modelo.")
        else:
            st.info("Entrene el modelo para ver el análisis de sentimientos.")

    # Tab 4: Noticias Recientes (NewsAPI)
    with tabs[3]:
        st.header("Noticias Recientes")
        symbol = coinid_to_symbol[coin_id]
        articles = get_newsapi_articles(symbol)
        if articles:
            st.subheader(f"Últimas noticias sobre {crypto_name}")
            st.markdown(
                """
                <style>
                .news-container {
                    display: flex;
                    overflow-x: auto;
                    gap: 1rem;
                }
                .news-item {
                    flex: 0 0 auto;
                    width: 300px;
                    background-color: #2c2c3e;
                    padding: 1rem;
                    border-radius: 5px;
                    border: 1px solid #4a4a6a;
                }
                .news-item h4 {
                    margin: 0 0 0.5rem 0;
                    font-size: 1.1rem;
                }
                .news-item p {
                    font-size: 0.9rem;
                    margin: 0 0 0.5rem 0;
                }
                </style>
                """,
                unsafe_allow_html=True
            )
            st.markdown("<div class='news-container'>", unsafe_allow_html=True)
            for article in articles:
                st.markdown(
                    f"""
                    <div class='news-item'>
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
