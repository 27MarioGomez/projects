import streamlit as st
import pandas as pd
import numpy as np
import requests
from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification
import torch
from datetime import datetime, timedelta
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.regularizers import l2
from sklearn.metrics import mean_squared_error
import plotly.express as px
import plotly.graph_objects as go
import yfinance as yf
from newsapi import NewsApiClient

# Configurar las claves API desde Secrets en formato TOML
TWITTER_BEARER_TOKEN = st.secrets["twitter_bearer"]  # Manteniendo por compatibilidad, aunque no se usa
NEWSAPI_API_KEY = st.secrets["newsapi_key"]

# Inicializar el modelo de sentimiento avanzado con transformers
tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased-finetuned-sst-2-english")
model = AutoModelForSequenceClassification.from_pretrained("distilbert-base-uncased-finetuned-sst-2-english")
sentiment_analyzer = pipeline("sentiment-analysis", model=model, tokenizer=tokenizer, device=0 if torch.cuda.is_available() else -1)

# Función para analizar el sentimiento de un texto con transformers
def analyze_sentiment(text):
    if not text or not isinstance(text, str):
        return 0.0
    result = sentiment_analyzer(text[:512])[0]  # Limitar a 512 tokens
    score = 1.0 if result['label'] == 'POSITIVE' else -1.0
    return score * result['score']

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

def robust_mape(y_true, y_pred, eps=1e-9):
    """Calcula el MAPE de manera robusta evitando división por cero."""
    return np.mean(np.abs((y_true - y_pred) / np.maximum(np.abs(y_true), eps))) * 100

# Función para cargar datos históricos usando yfinance
@st.cache_data
def load_historical_data(coin_id, days=3650):  # 10 años para máximo histórico
    crypto_map = {
        "bitcoin": "BTC-USD", "ethereum": "ETH-USD", "xrp": "XRP-USD",
        "binance-coin": "BNB-USD", "cardano": "ADA-USD", "solana": "SOL-USD",
        "dogecoin": "DOGE-USD", "polkadot": "DOT-USD", "polygon": "MATIC-USD",
        "litecoin": "LTC-USD", "tron": "TRX-USD", "stellar": "XLM-USD"
    }
    symbol = crypto_map.get(coin_id, "BTC-USD")
    end_date = datetime.now()
    start_date = end_date - timedelta(days=days)
    df = yf.download(symbol, start=start_date, end=end_date)
    if df.empty:
        st.error(f"No se pudieron cargar datos históricos para {symbol}.")
        return pd.DataFrame()
    df = df.reset_index()
    df = df[["Date", "Close", "Volume"]]
    df.columns = ["ds", "close_price", "volume"]
    return df[df["ds"].dt.date <= datetime.now().date()].sort_values("ds")

def create_sequences(data, window_size, include_sentiment=False):
    """Crea secuencias para el modelo LSTM con sentimiento integrado si aplica."""
    if len(data) <= window_size:
        return None, None
    X, y = [], []
    for i in range(window_size, len(data)):
        X.append(data[i - window_size:i])
        y.append(data[i, 0] if not include_sentiment else data[i])
    return np.array(X), np.array(y)

def build_lstm_model(input_shape, learning_rate=0.001, l2_lambda=0.01):
    """Construye un modelo LSTM mejorado con regularización L2 y más capas."""
    model = Sequential([
        LSTM(128, return_sequences=True, input_shape=input_shape, kernel_regularizer=l2(l2_lambda)),
        Dropout(0.3),
        LSTM(96, return_sequences=True, kernel_regularizer=l2(l2_lambda)),
        Dropout(0.3),
        LSTM(64, kernel_regularizer=l2(l2_lambda)),
        Dropout(0.3),
        Dense(64, activation="relu", kernel_regularizer=l2(l2_lambda)),
        Dense(1)
    ])
    model.compile(optimizer=Adam(learning_rate), loss="mse")
    return model

def train_model(X_train, y_train, X_val, y_val, input_shape, epochs, batch_size):
    """Entrena el modelo LSTM mejorado con validación cruzada y callbacks optimizados."""
    tf.keras.backend.clear_session()
    model = build_lstm_model(input_shape)
    callbacks = [
        EarlyStopping(patience=15, restore_best_weights=True, monitor="val_loss"),
        ReduceLROnPlateau(patience=10, factor=0.5, min_lr=1e-6, monitor="val_loss")
    ]
    history = model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=epochs, batch_size=batch_size, callbacks=callbacks, verbose=0)
    return model, history

def get_dynamic_params(df, horizon_days, coin_id):
    """Ajusta parámetros dinámicos mejorados según volatilidad y datos."""
    volatility = df["close_price"].pct_change().std()
    base_volatility = crypto_characteristics.get(coin_id, {"volatility": 0.05})["volatility"]
    
    if coin_id == "xrp":
        window_size = min(max(15, int(horizon_days * 0.8)), len(df) // 4)
        epochs = min(150, max(40, int(len(df) / 50) + int(volatility * 300)))
        batch_size = 16
        learning_rate = 0.0002
    elif coin_id == "bitcoin":
        window_size = min(max(30, int(horizon_days * 1.2)), len(df) // 3)
        epochs = min(100, max(30, int(len(df) / 70) + int(volatility * 200)))
        batch_size = 32
        learning_rate = 0.0005
    else:
        window_size = min(max(20, int(horizon_days * 1.0)), len(df) // 4)
        epochs = min(120, max(35, int(len(df) / 60) + int(volatility * 250)))
        batch_size = 24
        learning_rate = 0.0003

    return window_size, epochs, batch_size, learning_rate

@st.cache_data(ttl=3600)
def get_fear_greed_index():
    """Obtiene el índice Fear & Greed."""
    try:
        return float(requests.get("https://api.alternative.me/fng/?format=json", timeout=10).json()["data"][0]["value"])
    except Exception:
        st.warning("No se pudo obtener Fear & Greed Index. Usando valor por defecto.")
        return 50.0

@st.cache_data(ttl=3600)
def get_coingecko_community_activity(coin_id):
    """Obtiene actividad comunitaria de CoinGecko."""
    try:
        cg_id = coinid_to_coingecko.get(coin_id, coin_id)
        data = requests.get(f"https://api.coingecko.com/api/v3/coins/{cg_id}?community_data=true", timeout=10).json()["community_data"]
        activity = max(data.get("twitter_followers", 0), data.get("reddit_average_posts_48h", 0) * 1000)
        return min(100, (activity / 20000000) * 100) if activity > 0 else 50.0
    except Exception:
        return 50.0

def get_crypto_sentiment_combined(coin_id, news_sentiment=None):
    """Calcula el sentimiento combinado dinámico con pesos ajustados por volatilidad."""
    fg = get_fear_greed_index()
    cg = get_coingecko_community_activity(coin_id)
    volatility = crypto_characteristics.get(coin_id, {"volatility": 0.05})["volatility"]

    if volatility > 0.07:
        fg_weight = 0.15
        cg_weight = 0.45
        news_weight = 0.40
    else:
        fg_weight = 0.50
        cg_weight = 0.30
        news_weight = 0.20

    news_sent = 0.0 if news_sentiment is None or pd.isna(news_sentiment) else float(news_sentiment)
    combined_sentiment = (fg * fg_weight + cg * cg_weight + news_sent * news_weight)
    return max(0, min(100, combined_sentiment))

# Función para obtener el sentimiento de una criptomoneda con NewsAPI
def get_news_sentiment(crypto_symbol):
    try:
        response = NewsApiClient(api_key=NEWSAPI_API_KEY).get_everything(q=crypto_symbol, language='en', page_size=5)
        articles = response.get("articles", [])
        sentiments = [analyze_sentiment(article.get("title", "") + " " + article.get("description", "")) for article in articles]
        return np.mean(sentiments) if sentiments else 0.0
    except Exception as e:
        st.error(f"Error al obtener sentimiento de NewsAPI: {e}. Usando valor por defecto (0.0).")
        return 0.0

def train_and_predict_with_sentiment(coin_id, horizon_days, start_ms=None, end_ms=None):
    """Entrena y predice combinando modelos, sentimiento y noticias específicas de cripto usando NewsAPI."""
    if start_ms is None or end_ms is None:
        end_date = datetime.now()
        start_date = end_date - timedelta(days=3650)  # Máximo histórico (10 años)
        start_ms = int(start_date.timestamp() * 1000)
        end_ms = int(end_date.timestamp() * 1000)
    
    df = load_historical_data(coin_id, days=3650)
    if df is None or df.empty:
        return None
    symbol = coinid_to_symbol[coin_id]

    # Obtener sentimiento con NewsAPI
    news_sent = get_news_sentiment(symbol)
    crypto_sent = get_crypto_sentiment_combined(coin_id, news_sent)
    market_sent = get_fear_greed_index()
    sentiment_factor = (crypto_sent + market_sent) / 200.0

    # Escalar datos
    scaler = MinMaxScaler()
    scaled_data = scaler.fit_transform(df[["close_price", "volume"]])
    window_size, epochs, batch_size, learning_rate = get_dynamic_params(df, horizon_days, coin_id)
    X, y = create_sequences(scaled_data, window_size, include_sentiment=True)
    if X is None:
        return None

    split = int(len(X) * 0.8)
    X_train, X_test = X[:split], X[split:]
    y_train, y_test = y[:split], y[split:]
    val_split = int(len(X_train) * 0.9)
    X_val, y_val = X_train[val_split:], y_train[val_split:]
    X_train, y_train = X_train[:val_split], y_train[:val_split]

    # Ajustar dimensiones para incluir sentimiento
    X_train_adj = np.concatenate([X_train, np.full((X_train.shape[0], window_size, 1), sentiment_factor)], axis=-1)
    X_val_adj = np.concatenate([X_val, np.full((X_val.shape[0], window_size, 1), sentiment_factor)], axis=-1)
    X_test_adj = np.concatenate([X_test, np.full((X_test.shape[0], window_size, 1), sentiment_factor)], axis=-1)

    # Entrenar modelo
    lstm_model, history = train_model(X_train_adj, y_train, X_val_adj, y_val, (window_size, 3), epochs, batch_size)
    lstm_test_preds_scaled = lstm_model.predict(X_test_adj, verbose=0)
    lstm_test_preds = scaler.inverse_transform(lstm_test_preds_scaled[:, 0].reshape(-1, 1)).flatten()
    y_test_real = scaler.inverse_transform(y_test[:, 0].reshape(-1, 1)).flatten()
    lstm_rmse = np.sqrt(mean_squared_error(y_test_real, lstm_test_preds))
    lstm_mape = robust_mape(y_test_real, lstm_test_preds)

    # Predicciones futuras
    last_window = scaled_data[-window_size:]
    future_preds = []
    current_input = np.concatenate([last_window.reshape(1, window_size, 2), np.full((1, window_size, 1), sentiment_factor)], axis=-1)
    for _ in range(horizon_days):
        pred = lstm_model.predict(current_input, verbose=0)[0][0]
        future_preds.append(pred)
        new_feature = np.copy(current_input[:, -1:, :2])
        new_feature[0, 0, 0] = pred
        current_input = np.append(current_input[:, 1:, :], np.concatenate([new_feature, np.full((1, 1, 1), sentiment_factor)], axis=-1), axis=1)
    future_preds = scaler.inverse_transform(np.array(future_preds).reshape(-1, 1)).flatten()

    last_date = df["ds"].iloc[-1]
    future_dates = pd.date_range(start=last_date + timedelta(days=1), periods=horizon_days, freq="D").tolist()
    test_dates = df["ds"].iloc[-len(lstm_test_preds):].values

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
        "real_prices": y_test_real
    }

# Interfaz de Streamlit
def main_app():
    st.set_page_config(page_title="Crypto Price Predictions 🔮", layout="wide")
    st.title("Crypto Price Predictions 🔮")
    st.markdown("""
    **Descripción del Modelo:**  
    Esta plataforma utiliza un modelo avanzado de aprendizaje automático basado en redes LSTM (Long Short-Term Memory) para predecir precios futuros de criptomonedas como Bitcoin, Ethereum, Ripple y otras. El modelo integra datos históricos de precios y volúmenes de Yahoo Finance (yfinance), abarcando hasta diez años de información diaria, ajustando dinámicamente sus hiperparámetros según la volatilidad específica de cada criptomoneda. Además, incorpora un análisis de sentimiento dinámico que combina el índice Fear & Greed para el mercado global, la actividad comunitaria en redes sociales de CoinGecko, y noticias específicas de criptomonedas obtenidas a través de NewsAPI, mejorando la precisión al considerar el estado de ánimo del mercado, los inversores y eventos externos. Las predicciones se complementan con métricas clave como RMSE y MAPE para evaluar la precisión, y se presentan en gráficos interactivos y tablas para una experiencia clara y detallada.

    Fuentes de datos: Yahoo Finance (yfinance), Fear & Greed Index, CoinGecko, NewsAPI
    """)

    # Sidebar para configuración
    st.sidebar.title("Configura tu Predicción")
    crypto_name = st.sidebar.selectbox("Selecciona una criptomoneda:", list(coincap_ids.keys()))
    coin_id = coincap_ids[crypto_name]
    use_custom_range = st.sidebar.checkbox("Habilitar rango de fechas", value=False, help="Permite seleccionar un rango personalizado de fechas para los datos históricos.")
    default_end = datetime.now()
    default_start = default_end - timedelta(days=7)
    if use_custom_range:
        start_date = st.sidebar.date_input("Fecha de inicio", default_start.date())
        end_date = st.sidebar.date_input("Fecha de fin", default_end.date())
        if start_date > end_date:
            st.sidebar.error("La fecha de inicio no puede ser posterior a la fecha de fin.")
            return
        if (end_date - start_date).days > 3650:  # Limitar a 10 años como máximo
            st.sidebar.warning("El rango de fechas excede 10 años. Ajustando al máximo permitido (10 años).")
            end_date = start_date + timedelta(days=3650)
        if start_date > datetime.now().date():
            start_date = datetime.now().date() - timedelta(days=7)
            st.sidebar.warning("La fecha de inicio no puede ser futura. Ajustando al rango máximo permitido (7 días atrás).")
        if end_date > datetime.now().date():
            end_date = datetime.now().date()
            st.sidebar.warning("La fecha de fin no puede ser futura. Ajustando a hoy.")
        start_ms = int(datetime.combine(start_date, datetime.min.time()).timestamp() * 1000)
        end_ms = int(datetime.combine(end_date, datetime.min.time()).timestamp() * 1000)
    else:
        start_ms = int(default_start.timestamp() * 1000)
        end_ms = int(default_end.timestamp() * 1000)
    horizon = st.sidebar.slider("Días a predecir:", 1, 60, 5, help="Número de días futuros para los que se generará la predicción. RMSE y MAPE se calculan para evaluar la precisión.")
    show_stats = st.sidebar.checkbox("Ver estadísticas descriptivas", value=False, help="Muestra estadísticas básicas del precio histórico, como media, mediana y desviación estándar.")

    # Gráfico histórico
    df_prices = load_historical_data(coin_id, days=3650)  # Usar yfinance con máximo histórico
    if df_prices is not None and not df_prices.empty:
        if use_custom_range:
            df_prices = df_prices[(df_prices["ds"] >= pd.Timestamp(start_date)) & (df_prices["ds"] <= pd.Timestamp(end_date))]
        df_prices = df_prices[df_prices["ds"].dt.date <= datetime.now().date()].sort_values("ds").reset_index(drop=True)
        fig_hist = px.line(df_prices, x="ds", y="close_price", title=f"Histórico de {crypto_name}", labels={"ds": "Fecha", "close_price": "Precio en USD"})
        fig_hist.update_layout(
            template="plotly_dark",
            xaxis=dict(tickformat="%Y-%m-%d")
        )
        st.plotly_chart(fig_hist, use_container_width=True)
        if show_stats:
            st.subheader("Estadísticas Descriptivas")
            st.write(df_prices["close_price"].describe())

    # Pestañas
    tabs = st.tabs(["🤖 Entrenamiento y Test", "🔮 Predicción de Precios", "📊 Análisis de Sentimientos", "📰 Noticias Recientes"])
    with tabs[0]:
        st.header("Entrenamiento del Modelo y Evaluación en Test")
        if st.button("Entrenar Modelo y Predecir"):
            with st.spinner("Esto puede tardar un momento, por favor espera..."):
                result = train_and_predict_with_sentiment(coin_id, horizon, start_ms, end_ms)
            if result:
                st.success("Entrenamiento y predicción completados!")
                st.write(f"Sentimiento combinado de {result['symbol']}: {result['crypto_sent']:.2f}")
                st.write(f"Sentimiento global del mercado: {result['market_sent']:.2f}")
                st.write(f"Factor combinado: {result['sentiment_factor']:.2f}")
                col1, col2 = st.columns(2)
                col1.metric("RMSE (Test)", f"{result['rmse']:.2f}", help="Error promedio en dólares.")
                col2.metric("MAPE (Test)", f"{result['mape']:.2f}%", help="Error relativo promedio.")
                
                if not all(len(arr) > 0 for arr in [result["test_dates"], result["real_prices"], result["test_preds"]]):
                    st.error("No hay suficientes datos para mostrar el gráfico de entrenamiento y test.")
                    st.session_state["result"] = result
                    return

                min_len = min(len(result["test_dates"]), len(result["real_prices"]), len(result["test_preds"]))
                if min_len < 1:
                    st.error("No hay datos suficientes para generar el gráfico.")
                    st.session_state["result"] = result
                    return

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
                    title=f"Comparación entre el precio real y la predicción: {result['symbol']}",
                    template="plotly_dark",
                    xaxis=dict(tickformat="%Y-%m-%d"),
                    plot_bgcolor="#1e1e2f",
                    paper_bgcolor="#1e1e2f"
                )
                st.plotly_chart(fig_test, use_container_width=True)
                st.session_state["result"] = result

    with tabs[1]:
        st.header(f"🔮 Predicción de Precios - {crypto_name}")
        if "result" in st.session_state and isinstance(st.session_state["result"], dict):
            result = st.session_state["result"]
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
            st.info("Entrena el modelo primero.")

    with tabs[2]:
        st.header("📊 Análisis de Sentimientos")
        if "result" in st.session_state and isinstance(st.session_state["result"], dict):
            result = st.session_state["result"]
            crypto_sent, market_sent = result["crypto_sent"], result["market_sent"]
            diff = crypto_sent - market_sent
            if diff <= -15:
                sentiment_label = "Very Bearish"
            elif -15 < diff <= -5:
                sentiment_label = "Bearish"
            elif -5 < diff < 5:
                sentiment_label = "Neutral"
            elif 5 <= diff < 15:
                sentiment_label = "Bullish"
            else:
                sentiment_label = "Very Bullish"

            if diff <= -5:
                color = "#ff0000"
            elif -5 < diff < 5:
                color = "#ffd700"
            else:
                color = "#00ff00"

            fig_sentiment = go.Figure(go.Indicator(
                mode="gauge+number+delta",
                value=crypto_sent,
                domain={"x": [0, 1], "y": [0, 1]},
                title={"text": f"Sentimiento - {result['symbol']}", "font": {"size": 24, "color": "#ffffff"}},
                gauge={
                    "axis": {"range": [0, 100], "tickvals": [0, 25, 50, 75, 100], "ticktext": ["Very Bearish", "Bearish", "Neutral", "Bullish", "Very Bullish"], "tickcolor": "#ffffff", "tickwidth": 2, "tickfont": {"size": 16, "color": "#ffffff"}},
                    "bar": {"color": color},
                    "bgcolor": "#2c2c3e",
                    "borderwidth": 2,
                    "bordercolor": "#4a4a6a",
                    "steps": [
                        {"range": [0, 25], "color": "#ff0000"},
                        {"range": [25, 50], "color": "#ffd700"},
                        {"range": [50, 75], "color": "#00ff00"},
                        {"range": [75, 100], "color": "#008000"}
                    ],
                    "threshold": {"line": {"color": "#ffffff", "width": 4}, "thickness": 0.1, "value": crypto_sent}
                },
                number={"font": {"size": 48, "color": "#ffffff"}},
                delta={"reference": market_sent, "increasing": {"color": "#00ff00"}, "decreasing": {"color": "#ff0000"}}
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
            st.info("Entrena el modelo para ver el análisis.")

    with tabs[3]:
        st.header("📰 Noticias Recientes de Criptomonedas")
        news = get_recent_news(coinid_to_symbol[coin_id])
        if news:
            st.subheader(f"Últimas {len(news)} noticias sobre {crypto_name}")
            cols = st.columns(3)
            for i, article in enumerate(news):
                with cols[i % 3]:
                    with st.container(height=300, border=True):
                        st.write(f"**{article['title']}**")
                        st.write(f"Fecha: {article['pubDate']}")
                        st.write(article['description'])
                        if article['link']:
                            st.markdown(f"[Leer más]({article['link']})", unsafe_allow_html=True)
                        st.write(f"Sentimiento: {article['sentiment']:.2f}")
            news_df = pd.DataFrame(news)
            st.dataframe(news_df[["title", "pubDate", "sentiment"]].style.format({"pubDate": "{:%Y-%m-%d %H:%M:%S}", "sentiment": "{:.2f}"}).set_properties(**{'background-color': '#2c2c3e', 'color': 'white', 'border-color': '#4a4a6a'}))
        else:
            st.info("No se encontraron noticias recientes. Verifica la clave API, los límites de NewsAPI (100 solicitudes/día), o tu conexión.")

if __name__ == "__main__":
    main_app()