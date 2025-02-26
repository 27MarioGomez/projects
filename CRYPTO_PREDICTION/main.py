#########################
# main.py
#########################

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import requests
from io import StringIO
from datetime import datetime
from sklearn.preprocessing import MinMaxScaler
import pandas_ta as ta
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, Bidirectional, LSTM, Dense, Dropout
from tensorflow.keras.optimizers import Adam
import time

#########################
# 1. Funciones de apoyo y diccionarios
#########################
def robust_mape(y_true, y_pred, eps=1e-9):
    """Calcula el MAPE evitando divisiones por cero."""
    return np.mean(np.abs((y_true - y_pred) / np.maximum(np.abs(y_true), eps))) * 100

# Diccionario para CoinCap (solo se incluyen criptos con histórico completo)
coincap_ids = {
    "Bitcoin (BTC)":      "bitcoin",
    "Ethereum (ETH)":     "ethereum",
    "XRP":                "xrp",       # Usamos "xrp" en lugar de "ripple"
    "Stellar (XLM)":      "stellar",
    "Solana (SOL)":       "solana",
    "Cardano (ADA)":      "cardano",
    "Dogecoin (DOGE)":    "dogecoin",
    "Polkadot (DOT)":     "polkadot",
    "Polygon (MATIC)":    "polygon",
    "Litecoin (LTC)":     "litecoin",
    "TRON (TRX)":         "tron",
    "Binance Coin (BNB)": "binancecoin"
}

#########################
# 2. Función para descargar datos desde CoinCap con reintentos
#########################
@st.cache_data
def load_coincap_data(coin_id, vs_currency="usd", days="max", max_retries=3):
    """
    Descarga el histórico de precios desde CoinCap con reintentos en caso de error 429.
    Devuelve un DataFrame con columnas 'ds' y 'close_price'.
    """
    url = f"https://api.coincap.io/v2/assets/{coin_id}/history?vs_currency={vs_currency}&days={days}"
    headers = {"User-Agent": "Mozilla/5.0"}
    for attempt in range(max_retries):
        resp = requests.get(url, headers=headers)
        if resp.status_code == 200:
            data = resp.json()
            if "data" not in data:
                st.error("CoinCap: Datos no disponibles (falta 'data').")
                return None
            df = pd.DataFrame(data["data"])
            # Se asume que la API devuelve 'time' y 'priceUsd'
            if "time" not in df.columns or "priceUsd" not in df.columns:
                st.error("CoinCap: Las columnas 'time' o 'priceUsd' no se encontraron.")
                st.write(df.head())
                return None
            df["ds"] = pd.to_datetime(df["time"], unit="ms")
            df["close_price"] = pd.to_numeric(df["priceUsd"], errors="coerce")
            df = df[["ds", "close_price"]]
            df.dropna(subset=["ds", "close_price"], inplace=True)
            df.sort_values(by="ds", ascending=True, inplace=True)
            df.reset_index(drop=True, inplace=True)
            df = df[df["close_price"] > 0].copy()
            return df
        elif resp.status_code == 429:
            st.warning(f"CoinCap: Error 429 en intento {attempt+1}. Reintentando en {15*(attempt+1)} segundos...")
            time.sleep(15 * (attempt+1))
        else:
            st.error(f"Error al obtener datos de CoinCap (status code {resp.status_code}).")
            return None
    st.error("CoinCap: Se alcanzó el número máximo de reintentos sin éxito.")
    return None

#########################
# 3. Funciones para indicadores técnicos (opcional)
#########################
def add_indicators(df):
    """
    Calcula RSI, MACD y Bollinger Bands con pandas_ta y aplica forward fill.
    """
    df["rsi"] = ta.rsi(df["close_price"], length=14)
    macd_df = ta.macd(df["close_price"])
    bbands_df = ta.bbands(df["close_price"], length=20, std=2)
    df = pd.concat([df, macd_df, bbands_df], axis=1)
    df.ffill(inplace=True)
    return df

def add_all_indicators(df):
    return add_indicators(df)

#########################
# 4. Creación de secuencias para LSTM
#########################
def create_sequences(data, window_size=30):
    if len(data) <= window_size:
        st.error(f"No hay suficientes datos para una ventana de {window_size} días.")
        return None, None
    X, y = [], []
    for i in range(window_size, len(data)):
        X.append(data[i - window_size : i])
        y.append(data[i, 0])
    return np.array(X), np.array(y)

#########################
# 5. Construcción del modelo LSTM (Conv1D + Bidirectional LSTM)
#########################
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

#########################
# 6. Entrenamiento y predicción con LSTM
#########################
def train_and_predict(coin_id, horizon_days=30, window_size=30, test_size=0.2,
                      use_indicators=False, epochs=10, batch_size=32, learning_rate=0.001):
    """
    Descarga datos desde CoinCap, añade indicadores (opcional),
    entrena un modelo LSTM y realiza la predicción futura de forma iterativa.
    """
    df_prices = load_coincap_data(coin_id)
    if df_prices is None:
        return None

    if use_indicators:
        df_prices = add_all_indicators(df_prices)

    if "close_price" not in df_prices.columns:
        st.error("No se encontró 'close_price' en los datos de CoinCap.")
        st.write(df_prices.head())
        return None

    data_for_model = df_prices[["close_price"]].values
    scaler_target = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler_target.fit_transform(data_for_model)

    split_index = int(len(scaled_data) * (1 - test_size))
    if split_index <= window_size:
        st.error("No hay suficientes datos para el conjunto de entrenamiento.")
        return None

    train_data = scaled_data[:split_index]
    test_data = scaled_data[split_index:]
    X_train, y_train = create_sequences(train_data, window_size=window_size)
    if X_train is None:
        return None
    X_test, y_test = create_sequences(test_data, window_size=window_size)
    if X_test is None:
        return None

    val_split = int(len(X_train) * 0.9)
    X_val, y_val = X_train[val_split:], y_train[val_split:]
    X_train, y_train = X_train[:val_split], y_train[:val_split]

    input_shape = (X_train.shape[1], X_train.shape[2])
    lstm_model = build_lstm_model(input_shape, learning_rate=learning_rate)
    lstm_model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=epochs,
        batch_size=batch_size,
        verbose=1
    )

    lstm_preds_test = lstm_model.predict(X_test)
    y_test_deserialized = scaler_target.inverse_transform(y_test.reshape(-1, 1))
    lstm_preds_descaled = scaler_target.inverse_transform(lstm_preds_test)

    valid_mask = ~np.isnan(lstm_preds_descaled) & ~np.isnan(y_test_deserialized)
    if np.sum(valid_mask) == 0:
        rmse, mape = np.nan, np.nan
    else:
        rmse = np.sqrt(np.mean((y_test_deserialized[valid_mask] - lstm_preds_descaled[valid_mask]) ** 2))
        mape = robust_mape(y_test_deserialized[valid_mask], lstm_preds_descaled[valid_mask])

    # Predicción futura iterativa
    last_window = scaled_data[-window_size:]
    future_preds_scaled = []
    current_input = last_window.reshape(1, window_size, X_train.shape[2])

    @tf.function
    def predict_model(x):
        return lstm_model(x)

    for _ in range(horizon_days):
        future_pred = predict_model(current_input)[0][0]
        future_preds_scaled.append(future_pred)
        new_feature = np.zeros((1, 1, X_train.shape[2]))
        new_feature[0, 0, 0] = future_pred
        current_input = np.append(current_input[:, 1:, :], new_feature, axis=1)

    future_preds = scaler_target.inverse_transform(np.array(future_preds_scaled).reshape(-1, 1)).flatten()

    return df_prices, lstm_preds_descaled, y_test_deserialized, future_preds, rmse, mape

#########################
# 7. Lógica principal de la app (main_app)
#########################
def main_app():
    st.set_page_config(page_title="Crypto Price Prediction Dashboard", layout="wide")
    st.title("Crypto Price Predictions 🔮 - Solo CoinCap")
    st.markdown("**Fuente de Datos:** CoinCap (serie diaria, actualizada cada día)")

    st.sidebar.header("Configuración de la predicción")
    crypto_name = st.sidebar.selectbox("Selecciona una criptomoneda:", list(coincap_ids.keys()),
                                         help="Elige la criptomoneda para la predicción.")
    coin_id = coincap_ids[crypto_name]

    horizon = st.sidebar.slider("Días a predecir:", 1, 60, 30,
                                 help="Número de días a futuro a predecir.")
    auto_window = min(60, max(5, horizon * 2))
    st.sidebar.markdown(f"**Tamaño de ventana (auto): {auto_window} días**")

    use_indicators = st.sidebar.checkbox("Incluir indicadores técnicos (RSI, MACD, BBANDS)", value=True,
                                          help="Calcula indicadores técnicos localmente para enriquecer los datos.")

    scenario = st.sidebar.selectbox("Escenario del Modelo:", ["Pesimista", "Neutro", "Optimista"],
                                    help="Ajusta automáticamente los hiperparámetros del modelo.")
    if scenario == "Pesimista":
        epochs_val = 20
        batch_size_val = 32
        learning_rate_val = 0.001
    elif scenario == "Neutro":
        epochs_val = 30
        batch_size_val = 32
        learning_rate_val = 0.0008
    else:
        epochs_val = 50
        batch_size_val = 16
        learning_rate_val = 0.0005

    # Visualización del histórico
    df_prices = load_coincap_data(coin_id)
    if df_prices is not None and len(df_prices) > 0:
        df_chart = df_prices.copy()
        df_chart["ds"] = df_chart["ds"].dt.strftime("%d-%m-%Y")
        fig_hist = px.line(
            df_chart, x="ds", y="close_price",
            title=f"Histórico de Precio de {crypto_name} (CoinCap)",
            labels={"ds": "Fecha", "close_price": "Precio de Cierre"}
        )
        fig_hist.update_layout(xaxis=dict(type="category", tickangle=45))
        st.plotly_chart(fig_hist, use_container_width=True)
    else:
        st.warning("No se encontraron datos históricos válidos para mostrar el gráfico.")

    # Pestañas
    tabs = st.tabs(["🤖 Entrenamiento y Test", f"🔮 Predicción de Precios - {crypto_name}"])

    with tabs[0]:
        st.header("Entrenamiento del Modelo y Evaluación en Test")
        if st.button("Entrenar Modelo y Predecir"):
            with st.spinner("Entrenando el modelo, por favor espera..."):
                result = train_and_predict(
                    coin_id=coin_id,
                    horizon_days=horizon,
                    window_size=auto_window,
                    test_size=0.2,
                    use_indicators=use_indicators,
                    epochs=epochs_val,
                    batch_size=batch_size_val,
                    learning_rate=learning_rate_val
                )
            if result is not None:
                df_model, test_preds, y_test_real, future_preds, rmse, mape = result
                st.success("Entrenamiento y predicción completados!")
                col1, col2 = st.columns(2)
                col1.metric("RMSE (Test)", f"{rmse:.2f}")
                col2.metric("MAPE (Test)", f"{mape:.2f}%")

                st.subheader("Comparación en el Set de Test")
                test_dates = df_model["ds"].iloc[-len(y_test_real):]
                fig_test = go.Figure()
                fig_test.add_trace(go.Scatter(
                    x=test_dates,
                    y=y_test_real.flatten(),
                    mode="lines",
                    name="Precio Real (Test)"
                ))
                fig_test.add_trace(go.Scatter(
                    x=test_dates,
                    y=test_preds.flatten(),
                    mode="lines",
                    name="Predicción (Test)"
                ))
                fig_test.update_layout(
                    title=f"Comparación en Test: {crypto_name}",
                    xaxis_title="Fecha",
                    yaxis_title="Precio"
                )
                st.plotly_chart(fig_test, use_container_width=True)
            else:
                st.error("No se pudo entrenar el modelo debido a un error en la carga de datos.")

    with tabs[1]:
        st.header(f"Predicción de Precios - {crypto_name}")
        if 'result' in locals() and result is not None:
            df_model, test_preds, y_test_real, future_preds, rmse, mape = result
            last_date = df_model["ds"].iloc[-1]
            current_price = df_model["close_price"].iloc[-1]
            future_dates = pd.date_range(start=last_date, periods=horizon + 1, freq="D")
            pred_series = np.concatenate(([current_price], future_preds))
            fig_future = go.Figure()
            fig_future.add_trace(go.Scatter(
                x=future_dates,
                y=pred_series,
                mode="lines+markers",
                name="Predicción Futura"
            ))
            fig_future.update_layout(
                title=f"Predicción a Futuro ({horizon} días) - {crypto_name}",
                xaxis_title="Fecha",
                yaxis_title="Precio"
            )
            st.plotly_chart(fig_future, use_container_width=True)
            st.subheader("Valores Numéricos de la Predicción Futura")
            future_df = pd.DataFrame({"Fecha": future_dates, "Predicción": pred_series})
            st.dataframe(future_df)
        else:
            st.info("Primero entrena el modelo en la pestaña 'Entrenamiento y Test' para generar las predicciones futuras.")


if __name__ == "__main__":
    main_app()
