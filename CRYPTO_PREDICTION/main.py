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
from requests.adapters import HTTPAdapter
import keras_tuner as kt
import tweepy

# --- Configuration for SSL and HTTP session ---
os.environ["REQUESTS_CA_BUNDLE"] = certifi.where()
session = requests.Session()
retry = Retry(total=5, backoff_factor=1, status_forcelist=[429, 500, 502, 503, 504])
adapter = HTTPAdapter(max_retries=retry)
session.mount("https://", adapter)

# --- Cryptocurrency identifiers and symbols ---
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

# --- Predefined volatility characteristics per cryptocurrency ---
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

# --- Utility functions ---
def robust_mape(y_true, y_pred, eps=1e-9):
    """Compute the Mean Absolute Percentage Error."""
    return np.mean(np.abs((y_true - y_pred) / np.maximum(np.abs(y_true), eps))) * 100

@st.cache_data
def load_coincap_data(coin_id, start_ms=None, end_ms=None):
    """
    Load historical data from CoinCap for the given cryptocurrency.
    Data is fetched up to one day ahead to include the latest available price.
    Volume data is not processed.
    """
    try:
        if start_ms is None or end_ms is None:
            end_date = datetime.utcnow() + timedelta(days=1)
            start_date = end_date - timedelta(days=730)
            start_ms = int(start_date.timestamp() * 1000)
            end_ms = int(end_date.timestamp() * 1000)
        url = f"https://api.coincap.io/v2/assets/{coin_id}/history?interval=d1&start={start_ms}&end={end_ms}"
        resp = session.get(url, headers={"User-Agent": "Mozilla/5.0"}, verify=certifi.where(), timeout=10)
        if resp.status_code != 200:
            st.warning(f"CoinCap: Error {resp.status_code}")
            return None
        df = pd.DataFrame(resp.json().get("data", []))
        if df.empty or "time" not in df.columns or "priceUsd" not in df.columns:
            st.warning("CoinCap: No valid data available")
            return None
        df["ds"] = pd.to_datetime(df["time"], unit="ms", errors="coerce")
        df["close_price"] = pd.to_numeric(df["priceUsd"], errors="coerce")
        df = df[["ds", "close_price"]].dropna().sort_values("ds").reset_index(drop=True)
        return df
    except Exception as e:
        st.error(f"Error loading CoinCap data: {e}")
        return None

def create_sequences(data, window_size):
    """Generate sequences for LSTM model."""
    if len(data) <= window_size:
        return None, None
    X, y = [], []
    for i in range(window_size, len(data)):
        X.append(data[i - window_size:i])
        y.append(data[i, 0])
    return np.array(X), np.array(y)

# --- LSTM Model and dynamic hyperparameter adjustment ---
def build_lstm_model(
    input_shape,
    learning_rate=0.001,
    l2_lambda=0.01,
    lstm_units1=100,
    lstm_units2=80,
    dropout_rate=0.3,
    dense_units=50
):
    """Build an LSTM model with regularization and dropout."""
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
    """Train the LSTM model with early stopping and learning rate reduction."""
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
    Dynamically adjust hyperparameters based on historical data volatility and prediction horizon.
    Returns a dictionary with model hyperparameters.
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

# --- External Market Sentiment Data ---
@st.cache_data(ttl=3600)
def get_fear_greed_index():
    """Retrieve the market Fear & Greed index."""
    try:
        data = session.get("https://api.alternative.me/fng/?format=json", timeout=10).json()
        return float(data["data"][0]["value"])
    except Exception:
        st.warning("Could not retrieve Fear & Greed Index. Using default 50.0.")
        return 50.0

@st.cache_data(ttl=3600)
def get_coingecko_community_activity(coin_id):
    """Retrieve community activity from CoinGecko."""
    try:
        cg_id = coinid_to_coingecko.get(coin_id, coin_id)
        data = session.get(f"https://api.coingecko.com/api/v3/coins/{cg_id}?community_data=true", timeout=10).json()["community_data"]
        activity = max(data.get("twitter_followers", 0), data.get("reddit_average_posts_48h", 0) * 1000)
        return min(100, (activity / 20000000) * 100) if activity > 0 else 50.0
    except Exception:
        return 50.0

# --- Twitter API Integration using Tweepy ---
@st.cache_data(ttl=300)
def get_twitter_news(coin_symbol):
    """
    Retrieve recent tweets using Tweepy (Twitter API v2).
    The query searches for tweets containing the cryptocurrency symbol and "crypto",
    excluding retweets and filtering for English tweets.
    """
    bearer_token = st.secrets.get("twitter_bearer")
    if not bearer_token:
        st.error("Twitter bearer token not found in Secrets.")
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
        st.error(f"Error retrieving tweets: {e}")
        return []

def get_twitter_sentiment(coin_symbol):
    """
    Compute the average sentiment from recent tweets using TextBlob.
    Returns a sentiment score in the range [0, 100].
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
    Combine the cryptocurrency sentiment (from Twitter) and the market Fear & Greed index
    to compute a gauge value:
        gauge_val = 50 + (crypto_sent - market_sent), clamped to [0, 100].
    """
    symbol = coinid_to_symbol[coin_id]
    crypto_sent = get_twitter_sentiment(symbol)
    market_sent = get_fear_greed_index()
    gauge_val = 50 + (crypto_sent - market_sent)
    gauge_val = max(0, min(100, gauge_val))
    return crypto_sent, market_sent, gauge_val

# --- Keras Tuner Integration (Hyperband) ---
def build_model_tuner(input_shape):
    """
    Model builder for Keras Tuner using Hyperband.
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

# --- Training and Prediction Pipeline ---
def train_and_predict_with_sentiment(coin_id, horizon_days, start_ms=None, end_ms=None, tune=False):
    """
    Train the LSTM model and perform future predictions, integrating the sentiment factor (gauge_val/100)
    at each timestep. If 'tune' is True, hyperparameters are optimized using Keras Tuner (Hyperband).
    """
    df = load_coincap_data(coin_id, start_ms, end_ms)
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
        tuner.search(X_train_adj, y_train, validation_data=(X_val_adj, y_val), epochs=50, batch_size=batch_size, callbacks=[stop_early], verbose=0)
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
    current_input = np.concatenate([
        last_window.reshape(1, window_size, 1),
        np.full((1, window_size, 1), sentiment_factor)
    ], axis=-1)
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

# --- Streamlit Application ---
def main_app():
    st.set_page_config(page_title="Crypto Price Predictions 🔮", layout="wide")
    st.title("Crypto Price Predictions 🔮")
    st.markdown("""
    **Model Overview:**  
    This application uses an advanced LSTM model to predict future cryptocurrency prices (e.g., Bitcoin, Ethereum, Ripple).  
    It dynamically adjusts hyperparameters (with optional optimization via Keras Tuner Hyperband) based on historical volatility data from CoinCap.  
    Additionally, it computes market sentiment by combining the Fear & Greed index with tweet sentiment (retrieved via Tweepy).  
    Predictions are evaluated using RMSE and MAPE and are visualized through interactive charts.
    """)

    # Sidebar for configuration
    st.sidebar.title("Prediction Configuration")
    crypto_name = st.sidebar.selectbox("Select a cryptocurrency:", list(coincap_ids.keys()))
    coin_id = coincap_ids[crypto_name]
    use_custom_range = st.sidebar.checkbox("Enable custom date range", value=False)
    default_end = datetime.utcnow()
    default_start = default_end - timedelta(days=7)

    if use_custom_range:
        start_date = st.sidebar.date_input("Start Date", default_start.date())
        end_date = st.sidebar.date_input("End Date", default_end.date())
        if start_date > end_date:
            st.sidebar.error("Start date cannot be after end date.")
            return
        if (end_date - start_date).days > 7:
            st.sidebar.warning("Date range exceeds 7 days. It will be capped at 7 days.")
            end_date = start_date + timedelta(days=7)
        if start_date > datetime.utcnow().date():
            start_date = datetime.utcnow().date() - timedelta(days=7)
            st.sidebar.warning("Start date cannot be in the future. Adjusted to 7 days back.")
        if end_date > datetime.utcnow().date():
            end_date = datetime.utcnow().date()
            st.sidebar.warning("End date cannot be in the future. Adjusted to today.")
        end_date_with_offset = datetime.combine(end_date, datetime.min.time()) + timedelta(days=1)
        start_ms = int(datetime.combine(start_date, datetime.min.time()).timestamp() * 1000)
        end_ms = int(end_date_with_offset.timestamp() * 1000)
    else:
        end_date_with_offset = default_end + timedelta(days=1)
        start_ms = int(default_start.timestamp() * 1000)
        end_ms = int(end_date_with_offset.timestamp() * 1000)

    optimize_hp = st.sidebar.checkbox("Optimize hyperparameters (Keras Tuner Hyperband)", value=False)
    horizon = st.sidebar.slider("Days to predict:", 1, 60, 5)
    show_stats = st.sidebar.checkbox("Show descriptive statistics", value=False)

    # Historical Price Chart
    df_prices = load_coincap_data(coin_id, start_ms, end_ms)
    if df_prices is not None and not df_prices.empty:
        fig_hist = px.line(
            df_prices,
            x="ds",
            y="close_price",
            title=f"Historical Prices - {crypto_name}",
            labels={"ds": "Date", "close_price": "Price (USD)"}
        )
        fig_hist.update_layout(template="plotly_dark")
        fig_hist.update_xaxes(tickformat="%Y-%m-%d")
        st.plotly_chart(fig_hist, use_container_width=True)
        if show_stats:
            st.subheader("Descriptive Statistics")
            st.write(df_prices["close_price"].describe())
    else:
        st.warning("No historical data available for the selected range.")

    tabs = st.tabs(["🤖 Training & Test", "🔮 Price Prediction", "📊 Sentiment Analysis", "📰 Recent Tweets"])

    # --- Tab 1: Training & Test ---
    with tabs[0]:
        st.header("Training & Test Evaluation")
        if st.button("Train Model and Predict"):
            with st.spinner("Training model..."):
                result = train_and_predict_with_sentiment(coin_id, horizon, start_ms, end_ms, tune=optimize_hp)
            if result:
                st.success("Training and prediction completed!")
                st.write(f"Crypto Sentiment ({result['symbol']}): {result['crypto_sent']:.2f}")
                st.write(f"Market Sentiment (Fear & Greed): {result['market_sent']:.2f}")
                st.write(f"Combined Gauge: {result['gauge_val']:.2f}")

                col1, col2 = st.columns(2)
                col1.metric("RMSE (Test)", f"{result['rmse']:.2f}", help="Mean error in USD.")
                col2.metric("MAPE (Test)", f"{result['mape']:.2f}%", help="Mean percentage error.")

                if not (len(result["test_dates"]) > 0 and len(result["real_prices"]) > 0 and len(result["test_preds"]) > 0):
                    st.error("Insufficient data for Test chart.")
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
                    name="Actual Price",
                    line=dict(color="#1f77b4", width=3, shape="spline")
                ))
                fig_test.add_trace(go.Scatter(
                    x=result["test_dates"],
                    y=result["test_preds"],
                    mode="lines",
                    name="Prediction",
                    line=dict(color="#ff7f0e", width=3, dash="dash", shape="spline")
                ))
                fig_test.update_layout(
                    title=f"Actual vs. Predicted Prices ({result['symbol']})",
                    xaxis=dict(tickformat="%Y-%m-%d"),
                    template="plotly_dark",
                    xaxis_title="Date",
                    yaxis_title="Price (USD)"
                )
                st.plotly_chart(fig_test, use_container_width=True)
                st.session_state["result"] = result

    # --- Tab 2: Price Prediction ---
    with tabs[1]:
        st.header(f"Price Prediction - {crypto_name}")
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
                    name="Prediction",
                    line=dict(color="#ff7f0e", width=2, shape="spline")
                ))
                fig_future.update_layout(
                    title=f"Future Price Prediction ({horizon} days) - {result['symbol']}",
                    template="plotly_dark",
                    xaxis_title="Date",
                    yaxis_title="Price (USD)"
                )
                st.plotly_chart(fig_future, use_container_width=True)
                st.subheader("Numerical Results")
                df_future = pd.DataFrame({"Date": future_dates_display, "Prediction": pred_series})
                st.dataframe(df_future.style.format({"Prediction": "{:.2f}"}))
            else:
                st.info("No result available. Train the model first.")
        else:
            st.info("Train the model first.")

    # --- Tab 3: Sentiment Analysis ---
    with tabs[2]:
        st.header("Sentiment Analysis")
        if "result" in st.session_state:
            if isinstance(st.session_state["result"], dict):
                result = st.session_state["result"]
                if result is None or "gauge_val" not in result:
                    st.warning("No valid result. Please retrain the model.")
                else:
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
                        title={"text": f"Sentiment - {result['symbol']}", "x": 0.5, "xanchor": "center", "font": {"size": 24}},
                        template="plotly_dark",
                        height=400,
                        margin=dict(l=20, r=20, t=80, b=20)
                    )
                    st.plotly_chart(fig_sentiment, use_container_width=True)
                    st.write(f"**Crypto Sentiment ({result['symbol']}):** {crypto_sent:.2f}")
                    st.write(f"**Market Sentiment (Fear & Greed):** {market_sent:.2f}")
                    st.write(f"**Gauge Value:** {gauge_val:.2f} → **{gauge_text}**")
                    if gauge_val > 50:
                        st.write("**Trend:** Crypto sentiment exceeds market sentiment. Bullish scenario possible.")
                    else:
                        st.write("**Trend:** Crypto sentiment is at or below market sentiment. Caution advised.")
            else:
                st.error("Invalid result data. Please retrain the model.")
        else:
            st.info("Train the model to view sentiment analysis.")

    # --- Tab 4: Recent Tweets ---
    with tabs[3]:
        st.header("Recent Tweets on Cryptocurrency")
        symbol = coinid_to_symbol[coin_id]
        tweets = get_twitter_news(symbol)
        if tweets:
            st.subheader(f"Latest tweets on {crypto_name}")
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
            for tweet in tweets:
                st.markdown(
                    f"""
                    <div class='news-item'>
                        <h4>Tweet</h4>
                        <p><em>{tweet['pubDate']}</em></p>
                        <p>{tweet['text']}</p>
                        <p><a href="{tweet['link']}" target="_blank">View Tweet</a></p>
                    </div>
                    """,
                    unsafe_allow_html=True
                )
            st.markdown("</div>", unsafe_allow_html=True)
        else:
            st.warning("No recent tweets found or an error occurred.")

if __name__ == "__main__":
    main_app()
