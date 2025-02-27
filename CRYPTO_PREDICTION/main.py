#########################
# main.py
#########################

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import requests
from datetime import datetime
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, Bidirectional, LSTM, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
import time
import certifi
import os

# Configurar certificados SSL para requests
os.environ["REQUESTS_CA_BUNDLE"] = certifi.where()
os.environ["SSL_CERT_FILE"] = certifi.where()

##############################################
# Funciones de apoyo
##############################################

def robust_mape(y_true, y_pred, eps=1e-9):
    return np.mean(np.abs((y_true - y_pred) / np.maximum(np.abs(y_true), eps))) * 100

# Diccionarios para CoinCap y mapeo a símbolo para LunarCrush
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

##############################################
# Descarga de datos desde CoinCap (intervalo diario)
##############################################
@st.cache_data
def load_coincap_data(coin_id, start_ms=None, end_ms=None, max_retries=3):
    url = f"https://api.coincap.io/v2/assets/{coin_id}/history?interval=d1"
    if start_ms and end_ms:
        url += f"&start={start_ms}&end={end_ms}"
    headers = {"User-Agent": "Mozilla/5.0"}
    for attempt in range(max_retries):
        try:
            resp = requests.get(url, headers=headers, verify=certifi.where(), timeout=10)
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
                st.info("CoinCap: (400) Parámetros inválidos o rango excesivo.")
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
def create_sequences(data, window_size=30):
    if len(data) <= window_size:
        st.warning(f"No hay datos suficientes para una ventana de {window_size} días.")
        return None, None
    X, y = [], []
    for i in range(window_size, len(data)):
        X.append(data[i-window_size:i])
        y.append(data[i, 0])
    return np.array(X), np.array(y)

##############################################
# Modelo LSTM
##############################################
def build_lstm_model(input_shape, learning_rate=0.001):
    model = Sequential()
    model.add(Conv1D(filters=32, kernel_size=3, activation="relu", input_shape=input_shape))
    model.add(Bidirectional(LSTM(64, return_sequences=True)))
    model.add(Dropout(0.3))
    model.add(Bidirectional(LSTM(64, return_sequences=True)))
    model.add(Dropout(0.3))
    model.add(Bidirectional(LSTM(64, return_sequences=False)))
    model.add(Dropout(0.3))
    model.add(Dense(1))
    opt = Adam(learning_rate=learning_rate)
    model.compile(optimizer=opt, loss="mean_squared_error")
    return model

def train_model(X_train, y_train, X_val, y_val, input_shape, epochs, batch_size, learning_rate):
    tf.keras.backend.clear_session()
    model = build_lstm_model(input_shape, learning_rate=learning_rate)
    early_stop = EarlyStopping(monitor="val_loss", patience=5, restore_best_weights=True)
    model.fit(X_train, y_train, validation_data=(X_val, y_val),
              epochs=epochs, batch_size=batch_size, verbose=1, callbacks=[early_stop])
    return model

def get_dynamic_params(df, horizon_days):
    data_len = len(df)
    volatility = df["close_price"].pct_change().std()
    mean_price = df["close_price"].mean()
    window_size = min(max(10, horizon_days * 2), min(60, data_len // 2))
    epochs = min(50, max(20, int(data_len/100) + int(volatility*100)))
    batch_size = 16 if volatility > 0.05 or data_len < 500 else 32
    learning_rate = 0.0005 if mean_price > 1000 or volatility > 0.1 else 0.001
    return window_size, epochs, batch_size, learning_rate

##############################################
# Llamadas a la API de LunarCrush (versión Free)
##############################################
def get_crypto_sentiment_lunarcrush(symbol):
    api_key = st.secrets["lunarcrush_api_key"]
    base = "https://api.lunarcrush.com/v2"
    url = (f"{base}?data=assets&symbol={symbol}&key={api_key}"
           "&metrics=galaxy_score,alt_rank,average_sentiment,bullish_sentiment,bearish_sentiment")
    try:
        resp = requests.get(url, timeout=10)
        if resp.status_code != 200:
            st.warning(f"LunarCrush (assets): Error {resp.status_code} para {symbol}.")
            return 50.0
        assets = resp.json().get("data", [])
        if not assets:
            return 50.0
        asset = assets[0]
        galaxy = asset.get("galaxy_score", 50)
        alt_rank = asset.get("alt_rank", 2000)
        avg_sent = asset.get("average_sentiment", 0.5) * 100
        bull = asset.get("bullish_sentiment", 0)
        bear = asset.get("bearish_sentiment", 0)
        alt_score = max(0, min(100, (2000 - alt_rank)/2000*100))
        raw = 0.4 * galaxy + 0.1 * alt_score + 0.2 * avg_sent + 0.15 * bull - 0.15 * bear
        final = max(0, min(100, raw))
        return final
    except Exception as e:
        st.warning("Streamlit está experimentando algunos problemas. Usamos un valor neutro para el sentimiento.")
        return 50.0

def get_market_crypto_sentiment_lunarcrush():
    api_key = st.secrets["lunarcrush_api_key"]
    url = f"https://api.lunarcrush.com/v2?data=market&key={api_key}"
    try:
        resp = requests.get(url, timeout=10)
        if resp.status_code == 200:
            data = resp.json().get("data", [])
            if data:
                item = data[0]
                btc_dom = item.get("btc_dominance", 45)
                dom_sent = (btc_dom - 30) / (60 - 30) * 100
                return max(0, min(100, dom_sent))
            return 50.0
        else:
            st.warning(f"LunarCrush (market): Error {resp.status_code}.")
            return 50.0
    except Exception as e:
        st.warning("Streamlit está experimentando algunos problemas. Usamos un valor neutro para el sentimiento.")
        return 50.0

def get_lunarcrush_news(symbol, limit=5):
    api_key = st.secrets["lunarcrush_api_key"]
    url = f"https://api.lunarcrush.com/v2?data=news&symbol={symbol}&limit={limit}&key={api_key}"
    try:
        resp = requests.get(url, timeout=10)
        if resp.status_code != 200:
            return []
        items = resp.json().get("data", [])
        news_list = []
        for it in items:
            news_list.append({
                "title": it.get("title"),
                "url": it.get("url"),
                "description": it.get("description"),
                "published_at": it.get("published_at")
            })
        return news_list
    except Exception as e:
        st.warning("No se pueden mostrar noticias ahora mismo, vuelve luego.")
        return []

##############################################
# Entrenamiento y predicción con sentimiento
##############################################
def train_and_predict_with_sentiment(coin_id, use_custom_range, start_ms, end_ms,
                                     horizon_days=30, test_size=0.2):
    df_raw = load_coincap_data(coin_id, start_ms, end_ms)
    if df_raw is None or df_raw.empty:
        st.warning("No se pudieron descargar datos suficientes de CoinCap.")
        return None
    if "close_price" not in df_raw.columns:
        st.warning("No se encontró 'close_price' en los datos.")
        return None

    symbol = coinid_to_symbol.get(coin_id, "BTC")
    crypto_sent = get_crypto_sentiment_lunarcrush(symbol)
    market_sent = get_market_crypto_sentiment_lunarcrush()
    sentiment_factor = (crypto_sent + market_sent) / 200.0

    st.write(f"Sentimiento de {symbol}: {crypto_sent:.2f}")
    st.write(f"Sentimiento de mercado: {market_sent:.2f}")
    st.write(f"Factor combinado: {sentiment_factor:.2f}")

    window_size, epochs, batch_size, learning_rate = get_dynamic_params(df_raw, horizon_days)
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

    model = train_model(X_train_adj, y_train, X_val_adj, y_val, input_shape, epochs, batch_size, learning_rate)

    test_preds_scaled = model.predict(X_test_adj)
    test_preds = scaler_target.inverse_transform(test_preds_scaled)
    y_test_real = scaler_target.inverse_transform(y_test.reshape(-1, 1))
    valid_mask = ~np.isnan(test_preds) & ~np.isnan(y_test_real)
    if np.sum(valid_mask) == 0:
        rmse, mape = np.nan, np.nan
    else:
        rmse = np.sqrt(np.mean((y_test_real[valid_mask] - test_preds[valid_mask])**2))
        mape = robust_mape(y_test_real[valid_mask], test_preds[valid_mask])

    last_window = scaled_data[-window_size:]
    future_preds_scaled = []
    current_input = np.concatenate([last_window.reshape(1, window_size, 1),
                                      np.full((1, window_size, 1), sentiment_factor)],
                                     axis=-1)
    for _ in range(horizon_days):
        future_pred = model.predict(current_input)[0][0]
        future_preds_scaled.append(future_pred)
        new_feature = np.copy(current_input[:, -1:, :])
        new_feature[0, 0, 0] = future_pred
        new_feature[0, 0, 1] = sentiment_factor
        current_input = np.append(current_input[:, 1:, :], new_feature, axis=1)
    future_preds = scaler_target.inverse_transform(np.array(future_preds_scaled).reshape(-1, 1)).flatten()

    return df_raw, test_preds, y_test_real, future_preds, rmse, mape, sentiment_factor, symbol

##############################################
# Función principal de la app
##############################################
def main_app():
    st.set_page_config(page_title="Crypto Price Prediction 🔮", layout="wide")
    st.title("Crypto Price Prediction 🔮")
    st.markdown("Este modelo combina datos históricos y análisis de sentimiento para predecir precios de criptomonedas. Incluye un LSTM entrenado con datos de CoinCap y LunarCrush, gráficos interactivos y noticias relevantes.")
    st.markdown("**Fuente de Datos:** CoinCap y LunarCrush")

    st.sidebar.title("Configura tu Predicción")
    st.session_state["crypto_name"] = st.sidebar.selectbox(
        "Selecciona una criptomoneda:",
        list(coincap_ids.keys()),
        help="Elige la criptomoneda que deseas analizar."
    )
    coin_id = coincap_ids[st.session_state["crypto_name"]]

    st.sidebar.subheader("Rango de Fechas")
    use_custom_range = st.sidebar.checkbox(
        "Habilitar rango de fechas",
        value=True,
        help="Activa esto para elegir un período específico."
    )
    default_start = datetime(2021, 1, 1)
    default_end = datetime.now()
    if use_custom_range:
        start_date = st.sidebar.date_input(
            "Fecha de inicio",
            default_start,
            help="Desde cuándo analizar datos."
        )
        end_date = st.sidebar.date_input(
            "Fecha de fin",
            default_end,
            help="Hasta cuándo incluir datos."
        )
        start_ms = int(datetime.combine(start_date, datetime.min.time()).timestamp() * 1000)
        end_ms = int(datetime.combine(end_date, datetime.min.time()).timestamp() * 1000)
    else:
        start_ms, end_ms = None, None

    st.sidebar.subheader("Parámetros de Predicción")
    horizon = st.sidebar.slider(
        "Días a predecir:",
        1, 60, 30,
        help="Cuántos días en el futuro predecir."
    )
    st.sidebar.markdown("**Hiperparámetros ajustados automáticamente según los datos.**")

    df_prices = load_coincap_data(coin_id, start_ms, end_ms)
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
        st.info("No se encontraron datos históricos válidos. Reajusta el rango de fechas.")

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
                col1.metric(
                    "RMSE (Test)",
                    f"{rmse:.2f}",
                    help=f"Este valor indica el error promedio en dólares. Un RMSE de {rmse:.2f} significa que la predicción puede variar en promedio {rmse:.2f} dólares arriba o abajo del valor real."
                )
                col2.metric(
                    "MAPE (Test)",
                    f"{mape:.2f}%",
                    help=f"Este porcentaje muestra el error promedio en términos relativos. Un MAPE de {mape:.2f}% indica que, en promedio, la predicción se desvía un {mape:.2f}% del valor real."
                )
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
        st.header(f"Noticias recientes de {st.session_state['crypto_name']}")
        if 'result' in locals() and result is not None:
            symbol = result[-1]
            news_items = get_lunarcrush_news(symbol, limit=5)
            if news_items:
                for i, item in enumerate(news_items, start=1):
                    st.markdown(f"**{i}. {item['title']}**")
                    st.markdown(f"[Ver noticia]({item['url']})")
                    if item["description"]:
                        st.write(item["description"])
                    st.write(f"Publicado: {item['published_at']}")
                    st.write("---")
            else:
                st.write("No hay noticias disponibles en este momento.")
        else:
            st.info("Primero entrena el modelo para mostrar noticias.")

if __name__ == "__main__":
    main_app()