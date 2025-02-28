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
from textblob import TextBlob
import socket
from urllib3.util.retry import Retry

# Configuración inicial de certificados SSL y solicitudes
os.environ["REQUESTS_CA_BUNDLE"] = certifi.where()
session = requests.Session()
retry = Retry(total=5, backoff_factor=1, status_forcelist=[429, 500, 502, 503, 504])
adapter = HTTPAdapter(max_retries=retry)
session.mount("https://", adapter)

# Constantes y configuraciones
COINS = {
    "Bitcoin (BTC)": "bitcoin", "Ethereum (ETH)": "ethereum", "Ripple (XRP)": "xrp",
    "Binance Coin (BNB)": "binance-coin", "Cardano (ADA)": "cardano", "Solana (SOL)": "solana",
    "Dogecoin (DOGE)": "dogecoin", "Polkadot (DOT)": "polkadot", "Polygon (MATIC)": "polygon",
    "Litecoin (LTC)": "litecoin", "TRON (TRX)": "tron", "Stellar (XLM)": "stellar"
}
SYMBOLS = {v: k.split(" (")[1][:-1] for k, v in COINS.items()}
COINGECKO_IDS = {v: v if v != "xrp" else "ripple" for v in COINS.values()}
VOLATILITY = {
    "bitcoin": 0.03, "ethereum": 0.05, "xrp": 0.08, "binance-coin": 0.06, "cardano": 0.07,
    "solana": 0.09, "dogecoin": 0.12, "polkadot": 0.07, "polygon": 0.06, "litecoin": 0.04,
    "tron": 0.06, "stellar": 0.05
}
DARK_THEME = {"template": "plotly_dark", "plot_bgcolor": "#1e1e2f", "paper_bgcolor": "#1e1e2f", "legend": dict(orientation="h", y=1.02, x=1)}

# Funciones de apoyo
def mape(y_true, y_pred, eps=1e-9):
    """MAPE robusto."""
    return np.mean(np.abs((y_true - y_pred) / np.maximum(np.abs(y_true), eps))) * 100

@st.cache_data
def load_data(coin_id, start_ms=None, end_ms=None):
    """Carga datos de CoinCap."""
    if start_ms is None or end_ms is None:
        end, start = datetime.now(), datetime.now() - timedelta(days=730)
        start_ms, end_ms = int(start.timestamp() * 1000), int(end.timestamp() * 1000)
    url = f"https://api.coincap.io/v2/assets/{coin_id}/history?interval=d1&start={start_ms}&end={end_ms}"
    try:
        resp = session.get(url, headers={"User-Agent": "Mozilla/5.0"}, verify=certifi.where(), timeout=10)
        if resp.status_code != 200: st.warning(f"CoinCap error {resp.status_code}"); return None
        data = pd.DataFrame(resp.json()["data"])
        if data.empty or "time" not in data or "priceUsd" not in data: st.warning("Datos CoinCap inválidos"); return None
        return data[["time", "priceUsd", "volumeUsd"]].rename(columns={
            "time": "ds", "priceUsd": "close_price", "volumeUsd": "volume"
        }).assign(**{
            "ds": lambda x: pd.to_datetime(x["ds"], unit="ms", errors="coerce"),
            "close_price": lambda x: pd.to_numeric(x["close_price"], errors="coerce"),
            "volume": lambda x: pd.to_numeric(x["volume"], errors="coerce").fillna(0.0)
        }).dropna().sort_values("ds")
    except Exception as e: st.error(f"Error cargando datos: {e}"); return None

def create_sequences(data, window_size):
    """Crea secuencias para LSTM."""
    if len(data) <= window_size: return None, None
    X, y = [], []
    for i in range(window_size, len(data)): X.append(data[i - window_size:i]); y.append(data[i, 0])
    return np.array(X), np.array(y)

def build_lstm(input_shape, lr=0.001, l2=0.01):
    """Construye modelo LSTM mejorado."""
    model = Sequential([
        LSTM(100, return_sequences=True, input_shape=input_shape, kernel_regularizer=l2(l2)),
        Dropout(0.3), LSTM(80, kernel_regularizer=l2(l2)), Dropout(0.3),
        Dense(50, "relu", kernel_regularizer=l2(l2)), Dense(1)
    ], name="LSTM").compile(optimizer=Adam(lr), loss="mse")
    return model

def train_lstm(X_train, y_train, X_val, y_val, shape, epochs, batch_size):
    """Entrena modelo LSTM."""
    tf.keras.backend.clear_session()
    model = build_lstm(shape)
    callbacks = [EarlyStopping(patience=10, restore_best_weights=True), ReduceLROnPlateau(patience=5, factor=0.5, min_lr=1e-6)]
    return model, model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=epochs, batch_size=batch_size, callbacks=callbacks, verbose=0)

def get_params(df, horizon, coin_id):
    """Parámetros dinámicos por volatilidad."""
    vol = df["close_price"].pct_change().std()
    base_vol = VOLATILITY[coin_id]
    if coin_id == "xrp": return min(max(15, int(horizon)), len(df) // 4), min(150, max(40, int(len(df) / 60) + int(vol * 300))), 16, 0.0002
    if coin_id == "bitcoin": return min(max(30, int(horizon * 1.5)), len(df) // 3), min(100, max(30, int(len(df) / 80) + int(vol * 200))), 32, 0.0005
    return min(max(20, int(horizon * 1.2)), len(df) // 4), min(120, max(35, int(len(df) / 70) + int(vol * 250))), 24, 0.0003

@st.cache_data(ttl=3600)
def get_fear_greed():
    """Índice Fear & Greed."""
    try: return float(session.get("https://api.alternative.me/fng/?format=json", timeout=10).json()["data"][0]["value"])
    except: st.warning("Error en Fear & Greed. Usando 50.0"); return 50.0

@st.cache_data(ttl=3600)
def get_coingecko(coin_id):
    """Actividad comunitaria de CoinGecko."""
    try:
        cg_id = COINGECKO_IDS[coin_id]
        data = session.get(f"https://api.coingecko.com/api/v3/coins/{cg_id}?community_data=true", timeout=10).json()["community_data"]
        activity = max(data.get("twitter_followers", 0), data.get("reddit_average_posts_48h", 0) * 1000)
        return min(100, activity / 20000000 * 100) if activity else 50.0
    except: st.warning(f"Error en CoinGecko para {coin_id}. Usando 50.0"); return 50.0

def get_sentiment(coin_id, news_sent=None):
    """Sentimiento combinado dinámico."""
    fg, cg = get_fear_greed(), get_coingecko(coin_id)
    vol = VOLATILITY[coin_id]
    weights = (0.15, 0.45, 0.40) if vol > 0.07 else (0.50, 0.30, 0.20)
    news = 50.0 if news_sent is None or pd.isna(news_sent) else float(news_sent)
    return max(0, min(100, sum(w * v for w, v in zip(weights, [fg, cg, news]))))

@st.cache_data(ttl=86400)
def get_news_sentiment(symbol, start=None, end=None):
    """Sentimiento de noticias cripto desde NewsData.io."""
    if start is None or end is None: end, start = datetime.now().date(), datetime.now().date() - timedelta(days=7)
    else:
        if (end - start).days > 7: start = end - timedelta(days=7)
        if start > datetime.now().date(): start = datetime.now().date() - timedelta(days=7)
        if end > datetime.now().date(): end = datetime.now().date()

    api_key = st.secrets.get("news_data_key", "pub_7227626d8277642d9399e67d37a74d463f7cc")
    if not api_key: st.error("API key de NewsData.io no encontrada. Usando 50.0"); return 50.0

    query = f"{symbol} AND (price OR market OR regulation)"
    url = f"https://newsdata.io/api/1/news?apikey={api_key}&q={requests.utils.quote(query)}&language=en&from_date={start.strftime('%Y-%m-%d')}&to_date={end.strftime('%Y-%m-%d')}&size=5&category=crypto"
    
    try:
        if socket.getaddrinfo('newsdata.io', 443): pass
    except socket.gaierror as e: st.error(f"DNS error para newsdata.io: {e}. Usando 50.0"); return 50.0

    try:
        resp = session.get(url, timeout=10, headers={"User-Agent": "Mozilla/5.0"})
        if resp.status_code == 200:
            articles = resp.json().get("results", [])
            if not articles: return 50.0
            sentiments = [50 + (TextBlob(t or d).sentiment.polarity * 50) for t, d in [(a.get("title"), a.get("description")) for a in articles[:5]] if (t or d) and any(k in (t or d).lower() for k in ["price", "market", "regulation"])]
            return np.mean(sentiments) if sentiments else 50.0
        if resp.status_code in (422, 401, 429): st.error(f"NewsData.io error {resp.status_code}: {'Límite' if 429 else 'Parámetros' if 422 else 'Clave'}. Usando 50.0"); return 50.0
        return 50.0
    except requests.ConnectionError as e: st.error(f"Conexión con NewsData.io falló: {e}. Usando 50.0"); return 50.0
    except: return 50.0

@st.cache_data(ttl=3600)
def get_news(coin_symbol):
    """Noticias recientes de criptomonedas desde NewsData.io."""
    end, start = datetime.now().date(), datetime.now().date() - timedelta(days=14)

    api_key = st.secrets.get("news_data_key", "pub_7227626d8277642d9399e67d37a74d463f7cc")
    if not api_key: st.error("API key de NewsData.io no encontrada. Sin noticias."); return []

    query = f"crypto AND {coin_symbol}"
    url = f"https://newsdata.io/api/1/news?apikey={api_key}&q={requests.utils.quote(query)}&language=en&from_date={start.strftime('%Y-%m-%d')}&to_date={end.strftime('%Y-%m-%d')}&size=10&category=crypto&sort_by=pubDate"
    
    try:
        if socket.getaddrinfo('newsdata.io', 443): pass
    except socket.gaierror as e: st.error(f"DNS error para newsdata.io: {e}. Sin noticias."); return []

    try:
        resp = session.get(url, timeout=10, headers={"User-Agent": "Mozilla/5.0"})
        if resp.status_code == 200:
            articles = resp.json().get("results", [])
            if not articles:
                query_simple = "crypto"
                url_retry = f"https://newsdata.io/api/1/news?apikey={api_key}&q={requests.utils.quote(query_simple)}&language=en&from_date={(end - timedelta(days=7)).strftime('%Y-%m-%d')}&to_date={end.strftime('%Y-%m-%d')}&size=10&category=crypto&sort_by=pubDate"
                resp_retry = session.get(url_retry, timeout=10, headers={"User-Agent": "Mozilla/5.0"})
                if resp_retry.status_code == 200: articles = resp_retry.json().get("results", [])
                if not articles: return []
            return [{"title": a.get("title", "Sin título"), "description": a.get("description", "Sin descripción"), "pubDate": a.get("pubDate", "Sin fecha"), "link": a.get("link", "#")} for a in sorted(articles, key=lambda x: x.get("pubDate", ""), reverse=True)[:5]]
        if resp.status_code in (422, 401, 429): st.error(f"NewsData.io error {resp.status_code}: {'Límite' if 429 else 'Parámetros' if 422 else 'Clave'}. Sin noticias."); return []
        return []
    except requests.ConnectionError as e: st.error(f"Conexión con NewsData.io falló: {e}. Sin noticias."); return []
    except: return []

def predict(coin_id, horizon, start_ms=None, end_ms=None):
    """Predice precios con LSTM y sentimiento."""
    df = load_data(coin_id, start_ms, end_ms)
    if df is None:
        return None
    symbol = SYMBOLS[coin_id]
    news_sent = get_news_sentiment(symbol, datetime.fromtimestamp(start_ms / 1000).date() if start_ms else (datetime.now() - timedelta(days=7)).date(), datetime.fromtimestamp(end_ms / 1000).date() if end_ms else datetime.now().date())
    crypto_sent, market_sent = get_sentiment(coin_id, 50.0 if news_sent is None or pd.isna(news_sent) else float(news_sent)), get_fear_greed()
    sentiment = (crypto_sent + market_sent) / 200.0

    scaler, data = MinMaxScaler(), scaler.fit_transform(df[["close_price"]])
    window, epochs, batch, lr = get_params(df, horizon, coin_id)
    X, y = create_sequences(data, window)
    if X is None:
        return None

    split, val_split = int(len(X) * 0.8), int(len(X) * 0.9)
    X_train, X_test, y_train, y_test = X[:split], X[split:], y[:split], y[split:]
    X_val, y_val = X_train[val_split:], y_train[val_split:]
    X_train, y_train = X_train[:val_split], y_train[:val_split]

    X_adj = lambda x: np.concatenate([x, np.full((len(x), window, 1), sentiment)], axis=-1)
    model, _ = train_lstm(X_adj(X_train), y_train, X_adj(X_val), y_val, (window, 2), epochs, batch)
    preds = scaler.inverse_transform(model.predict(X_adj(X_test), verbose=0)).flatten()
    real = scaler.inverse_transform(y_test.reshape(-1, 1)).flatten()
    rmse, mape = np.sqrt(mean_squared_error(real, preds)), mape(real, preds)

    last, future = data[-window:], []
    current = np.concatenate([last.reshape(1, window, 1), np.full((1, window, 1), sentiment)])
    for _ in range(horizon):
        pred = model.predict(current, verbose=0)[0][0]
        future.append(pred)
        current = np.append(current[:, 1:, :], [[[pred, sentiment]]], axis=1)
    future = scaler.inverse_transform(np.array(future).reshape(-1, 1)).flatten()

    return {
        "df": df, "preds": preds, "future": future, "rmse": rmse, "mape": mape, "sentiment": sentiment,
        "symbol": symbol, "crypto": crypto_sent, "market": market_sent,
        "future_dates": pd.date_range(df["ds"].iloc[-1] + timedelta(days=1), periods=horizon).tolist(),
        "test_dates": df["ds"].iloc[-len(preds):].values, "real": df["close_price"].iloc[-len(preds):].values
    }

def main():
    st.set_page_config(page_title="Crypto Price Predictions 🔮", layout="wide")
    st.title("Crypto Price Predictions 🔮")
    st.markdown("**Modelo LSTM para predecir criptos con datos de CoinCap, sentimiento (Fear & Greed, CoinGecko, NewsData.io) y gráficos interactivos.**")

    st.sidebar.title("Configuración")
    coin = st.sidebar.selectbox("Criptomoneda", list(COINS.keys()))
    use_range = st.sidebar.checkbox("Rango de fechas", False)
    default_end, default_start = datetime.now(), datetime.now() - timedelta(days=7)
    start_ms, end_ms = (int(datetime.combine(st.sidebar.date_input("Inicio", default_start.date()), datetime.min.time()).timestamp() * 1000),
                        int(datetime.combine(st.sidebar.date_input("Fin", default_end.date()), datetime.min.time()).timestamp() * 1000)) if use_range else (int(default_start.timestamp() * 1000), int(default_end.timestamp() * 1000))
    if use_range and (end_ms - start_ms) / (1000 * 60 * 60 * 24) > 7:
        st.sidebar.warning("Rango > 7 días. Ajustando."); end_ms = start_ms + 7 * 24 * 60 * 60 * 1000
    if start_ms > end_ms or start_ms > datetime.now().timestamp() * 1000 or end_ms > datetime.now().timestamp() * 1000:
        st.sidebar.error("Fechas inválidas."); return
    horizon, stats = st.sidebar.slider("Días a predecir", 1, 60, 5), st.sidebar.checkbox("Estadísticas", False)

    df = load_data(COINS[coin], start_ms, end_ms)
    if df is not None:
        fig_hist = px.line(df, "ds", "close_price", title=f"Histórico - {coin}", labels={"ds": "Fecha", "close_price": "Precio (USD)"})
        fig_hist.update_layout(**DARK_THEME)
        st.plotly_chart(fig_hist, use_container_width=True)
        if stats: st.subheader("Estadísticas"); st.write(df["close_price"].describe())

    tabs = st.tabs(["🤖 Entrenamiento", "🔮 Predicciones", "📊 Sentimientos", "📰 Noticias"])
    with tabs[0]:
        st.header("Entrenamiento del Modelo")
        if st.button("Entrenar y Predecir"):
            with st.spinner("Procesando..."):
                result = predict(COINS[coin], horizon, start_ms, end_ms)
            if result:
                st.success("Completado!")
                st.write(f"Sentimientos: Combinado {result['crypto']:.2f}, Mercado {result['market']:.2f}, Factor {result['sentiment']:.2f}")
                c1, c2 = st.columns(2)
                c1.metric("RMSE", f"{result['rmse']:.2f} USD", "Error promedio")
                c2.metric("MAPE", f"{result['mape']:.2f}%", "Error relativo")

                if len(result["test_dates"]) == len(result["real"]) == len(result["preds"]):
                    fig_test = go.Figure([
                        go.Scatter(x=result["test_dates"], y=result["real"], mode="lines", name="Real", line=dict(color="#1f77b4", width=3)),
                        go.Scatter(x=result["test_dates"], y=result["preds"], mode="lines", name="Predicción", line=dict(color="#ff7f0e", width=3, dash="dash"))
                    ])
                    fig_test.update_layout(title=f"Real vs Predicción - {coin}", **DARK_THEME, xaxis_title="Fecha", yaxis_title="Precio (USD)", hovermode="x unified")
                    st.plotly_chart(fig_test, use_container_width=True)
                else: st.error("Datos insuficientes para el gráfico.")
                st.session_state.result = result

    with tabs[1]:
        st.header(f"Predicciones - {coin}")
        if "result" in st.session_state and isinstance(st.session_state.result, dict):
            result = st.session_state.result
            last_date, price = result["df"]["ds"].iloc[-1], result["df"]["close_price"].iloc[-1]
            preds = np.concatenate([np.array([price]), result["future"]])
            fig_pred = go.Figure(go.Scatter(x=[last_date] + result["future_dates"], y=preds, mode="lines+markers", name="Predicción", line=dict(color="#ff7f0e", width=2)))
            fig_pred.update_layout(title=f"Predicción ({horizon} días) - {coin}", **DARK_THEME, xaxis_title="Fecha", yaxis_title="Precio (USD)")
            st.plotly_chart(fig_pred, use_container_width=True)
            st.subheader("Valores"); st.dataframe(pd.DataFrame({"Fecha": [last_date] + result["future_dates"], "Predicción": preds}).style.format({"Predicción": "{:.2f}"}))
        else: st.info("Entrena el modelo primero.")

    with tabs[2]:
        st.header("📊 Sentimientos")
        if "result" in st.session_state and isinstance(st.session_state.result, dict):
            result = st.session_state.result
            crypto, market = result["crypto"], result["market"]
            level = (crypto - 50) / 5  # Escala -10 a 10
            label = "Very Bearish" if level <= -5 else "Bearish" if level <= -2 else "Neutral" if -2 < level < 2 else "Bullish" if level <= 5 else "Very Bullish"
            color = "#ff7f0e" if level < 0 else "#1f77b4"

            fig_sent = go.Figure(go.Indicator(
                mode="gauge+number+delta", value=crypto, domain={"x": [0, 1], "y": [0, 1]},
                title={"text": f"Sentimiento - {result['symbol']}", "font": {"size": 20, "color": "white"}},
                gauge={"axis": {"range": [0, 100], "tickcolor": "white", "tickwidth": 2},
                       "bar": {"color": color}, "bgcolor": "#1e1e2f", "borderwidth": 2, "bordercolor": "#4a4a6a",
                       "steps": [
                           {"range": [0, 25], "color": "#ff7f0e"}, {"range": [25, 40], "color": "#ffaa7f"},
                           {"range": [40, 60], "color": "#666666"}, {"range": [60, 75], "color": "#7fb4ff"},
                           {"range": [75, 100], "color": "#1f77b4"}
                       ], "threshold": {"line": {"color": "white", "width": 4}, "thickness": 0.75, "value": 50}},
                delta={"reference": market, "increasing": {"color": "#1f77b4"}, "decreasing": {"color": "#ff7f0e"}},
                number={"font": {"size": 40, "color": "white"}}
            ))
            fig_sent.update_layout(**DARK_THEME, height=400, width=600, margin=dict(l=20, r=20, t=50, b=20))
            st.plotly_chart(fig_sent, use_container_width=True)
            st.write(f"**Estado:** {label} (Mercado: {market:.2f})")
            st.write("**NFA:** Solo educativo, no consejo financiero. Consulta expertos.")
        else: st.info("Entrena el modelo para ver análisis.")

    with tabs[3]:
        st.header("📰 Noticias Recientes")
        news = get_news(SYMBOLS[COINS[coin]])
        if news:
            st.subheader(f"Últimas 5 noticias - {coin}")
            for n in news:
                with st.expander(f"**{n['title']}** - {n['pubDate']}", False):
                    st.write(n['description'])
                    if n['link']: st.markdown(f"[Leer más]({n['link']})", unsafe_allow_html=True)
            df_news = pd.DataFrame(news)[["title", "pubDate"]].style.format({"pubDate": "{:%Y-%m-%d %H:%M:%S}"})
            st.dataframe(df_news.set_properties(**{'background-color': '#2c2c3e', 'color': 'white', 'border-color': '#4a4a6a'}))
        else: st.info("Sin noticias recientes. Verifica conexión, límites API o intenta más tarde.")

if __name__ == "__main__":
    main()