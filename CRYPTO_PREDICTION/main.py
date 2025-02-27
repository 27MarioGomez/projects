#########################
# main.py
#########################

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import requests
from datetime import datetime, date
from sklearn.preprocessing import MinMaxScaler
import pandas_ta as ta
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, Bidirectional, LSTM, Dense, Dropout
from tensorflow.keras.optimizers import Adam
import time

###############################################################
# 1. Funciones de apoyo y diccionarios
###############################################################

def robust_mape(y_true, y_pred, eps=1e-9):
    """
    Calcula el MAPE evitando divisiones por cero.
    Se usa max(eps, abs(y_true)) para prevenir divisiones por cero en y_true=0.
    """
    return np.mean(np.abs((y_true - y_pred) / np.maximum(np.abs(y_true), eps))) * 100

# Diccionario para CoinCap con IDs que devuelven datos
coincap_ids = {
    "Bitcoin (BTC)":       "bitcoin",
    "Ethereum (ETH)":      "ethereum",
    "XRP":                 "xrp",
    "Binance Coin (BNB)":  "binance-coin",
    "Cardano (ADA)":       "cardano",
    "Solana (SOL)":        "solana",
    "Dogecoin (DOGE)":     "dogecoin",
    "Polkadot (DOT)":      "polkadot",
    "Polygon (MATIC)":     "polygon",
    "Litecoin (LTC)":      "litecoin",
    "TRON (TRX)":          "tron",
    "Stellar (XLM)":       "stellar"
}
# Si alguna cripto da problemas (status 400 o sin datos), puedes retirarla o revisar su ID en la API de CoinCap.

###############################################################
# 2. Función para descargar datos desde CoinCap (intervalo diario fijo)
###############################################################
@st.cache_data
def load_coincap_data(coin_id, start_ms=None, end_ms=None, max_retries=3):
    """
    Descarga el histórico de precios desde CoinCap a intervalo diario (d1).
    Se usan start_ms y end_ms (en milisegundos) para definir el rango de fechas.
    Devuelve un DataFrame con 'ds' y 'close_price' o None si hay error.
    """
    # Intervalo diario fijo
    url = f"https://api.coincap.io/v2/assets/{coin_id}/history?interval=d1"
    # Añadimos start y end si están definidos
    if start_ms is not None and end_ms is not None:
        url += f"&start={start_ms}&end={end_ms}"

    headers = {"User-Agent": "Mozilla/5.0"}

    for attempt in range(max_retries):
        resp = requests.get(url, headers=headers)
        if resp.status_code == 200:
            data = resp.json()
            if "data" not in data:
                st.warning("CoinCap: Datos no disponibles (falta 'data').")
                return None
            df = pd.DataFrame(data["data"])
            if df.empty:
                st.info("CoinCap devolvió datos vacíos. Reajusta el rango de fechas.")
                return None
            if "time" not in df.columns or "priceUsd" not in df.columns:
                st.warning("CoinCap: Columnas 'time' o 'priceUsd' no encontradas en la respuesta.")
                st.write(df.head())
                return None

            # Convertir 'time' a datetime y 'priceUsd' a float
            df["ds"] = pd.to_datetime(df["time"], unit="ms")
            df["close_price"] = pd.to_numeric(df["priceUsd"], errors="coerce")
            df = df[["ds", "close_price"]]
            df.dropna(subset=["ds", "close_price"], inplace=True)
            df.sort_values(by="ds", ascending=True, inplace=True)
            df.reset_index(drop=True, inplace=True)

            # Filtrar valores <= 0
            df = df[df["close_price"] > 0].copy()
            return df

        elif resp.status_code == 429:
            st.warning(f"CoinCap: Error 429 en intento {attempt+1}. Esperando {15*(attempt+1)}s antes de reintentar...")
            time.sleep(15*(attempt+1))
        elif resp.status_code == 400:
            st.info("CoinCap: (400) Parámetros inválidos o rango excesivo. Reajusta el rango de fechas.")
            return None
        else:
            st.info(f"CoinCap: status code {resp.status_code}. Revisa los parámetros o prueba otro rango.")
            return None

    st.info("CoinCap: Se alcanzó el número máximo de reintentos sin éxito.")
    return None

###############################################################
# 3. Funciones para indicadores técnicos (opcional)
###############################################################
def add_indicators(df):
    """
    Calcula RSI, MACD y Bollinger Bands con pandas_ta.
    Aplica forward fill para rellenar huecos.
    """
    df["rsi"] = ta.rsi(df["close_price"], length=14)
    macd_df = ta.macd(df["close_price"])
    bbands_df = ta.bbands(df["close_price"], length=20, std=2)
    df = pd.concat([df, macd_df, bbands_df], axis=1)
    df.ffill(inplace=True)
    return df

def add_all_indicators(df):
    return add_indicators(df)

###############################################################
# 4. Creación de secuencias para LSTM
###############################################################
def create_sequences(data, window_size=30):
    """
    Genera secuencias de longitud 'window_size' a partir de la serie escalada 'data'.
    Devuelve (X, y) como np.array.
    """
    if len(data) <= window_size:
        st.warning(f"No hay suficientes datos para una ventana de {window_size} días. Prueba otro rango.")
        return None, None
    X, y = [], []
    for i in range(window_size, len(data)):
        X.append(data[i - window_size : i])
        y.append(data[i, 0])
    return np.array(X), np.array(y)

###############################################################
# 5. Construcción del modelo LSTM (Conv1D + Bidirectional LSTM)
###############################################################
def build_lstm_model(input_shape, learning_rate=0.001):
    """
    Construye un modelo secuencial con:
    - 1D Conv
    - 3 capas Bidirectional LSTM con Dropout
    - Dense final para regresión (1 salida).
    """
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

###############################################################
# 6. Entrenamiento y predicción con LSTM
###############################################################
def train_and_predict(
    coin_id,
    start_ms,
    end_ms,
    horizon_days=30,
    window_size=30,
    test_size=0.2,
    use_indicators=False,
    epochs=10,
    batch_size=32,
    learning_rate=0.001
):
    """
    - Descarga datos de CoinCap (intervalo diario) con (start_ms, end_ms).
    - Aplica indicadores si se desea.
    - Prepara secuencias para LSTM, entrena y hace predicción futura iterativa.
    """
    # 1) Carga de datos (intervalo diario fijo)
    df_prices = load_coincap_data(coin_id, start_ms=start_ms, end_ms=end_ms)
    if df_prices is None or len(df_prices) == 0:
        st.warning("No se pudo descargar datos suficientes para entrenar. Reajusta el rango de fechas.")
        return None

    # 2) Indicadores técnicos (opcional)
    if use_indicators:
        df_prices = add_all_indicators(df_prices)

    if "close_price" not in df_prices.columns:
        st.warning("No se encontró la columna 'close_price' tras la descarga o aplicación de indicadores.")
        return None

    # 3) Escalado
    data_for_model = df_prices[["close_price"]].values
    scaler_target = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler_target.fit_transform(data_for_model)

    # 4) Split en train/test
    split_index = int(len(scaled_data) * (1 - test_size))
    if split_index <= window_size:
        st.warning("No hay suficientes datos para entrenar (split_index <= window_size). Reajusta parámetros.")
        return None
    train_data = scaled_data[:split_index]
    test_data = scaled_data[split_index:]
    X_train, y_train = create_sequences(train_data, window_size=window_size)
    if X_train is None:
        return None
    X_test, y_test = create_sequences(test_data, window_size=window_size)
    if X_test is None:
        return None

    # 5) Train/val split
    val_split = int(len(X_train) * 0.9)
    X_val, y_val = X_train[val_split:], y_train[val_split:]
    X_train, y_train = X_train[:val_split], y_train[:val_split]

    # 6) Construir y entrenar modelo LSTM
    input_shape = (X_train.shape[1], X_train.shape[2])
    lstm_model = build_lstm_model(input_shape, learning_rate=learning_rate)
    lstm_model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=epochs,
        batch_size=batch_size,
        verbose=1
    )

    # 7) Predicción en test
    test_preds_scaled = lstm_model.predict(X_test)
    test_preds = scaler_target.inverse_transform(test_preds_scaled)
    y_test_deserialized = scaler_target.inverse_transform(y_test.reshape(-1, 1))

    valid_mask = ~np.isnan(test_preds) & ~np.isnan(y_test_deserialized)
    if np.sum(valid_mask) == 0:
        rmse, mape = np.nan, np.nan
    else:
        rmse = np.sqrt(np.mean((y_test_deserialized[valid_mask] - test_preds[valid_mask]) ** 2))
        mape = robust_mape(y_test_deserialized[valid_mask], test_preds[valid_mask])

    # 8) Predicción futura iterativa
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

    return df_prices, test_preds, y_test_deserialized, future_preds, rmse, mape

###############################################################
# 7. Lógica principal de la app (main_app)
###############################################################
def main_app():
    """
    Interfaz principal de Streamlit para predecir precios de criptomonedas
    usando datos de CoinCap a intervalo diario fijo, con:
    - Selección de criptomoneda
    - Rango de fechas (start/end)
    - Parámetros de LSTM
    - Indicadores técnicos
    """
    st.set_page_config(page_title="Crypto Price Prediction Dashboard", layout="wide")
    st.title("Crypto Price Predictions 🔮 - Solo CoinCap (Diario)")
    st.markdown("**Fuente de Datos:** CoinCap (histórico diario con rango).")

    # Barra lateral: configuración principal
    st.sidebar.header("Configuración de la predicción")

    # 1) Selección de criptomoneda
    crypto_name = st.sidebar.selectbox(
        "Selecciona una criptomoneda:",
        list(coincap_ids.keys()),
        help="Elige la criptomoneda para la predicción."
    )
    coin_id = coincap_ids[crypto_name]

    # 2) Rango de fechas
    st.sidebar.subheader("Rango de Fechas (Diario)")
    default_start = datetime(2021, 1, 1)
    default_end = datetime.now()
    start_date = st.sidebar.date_input("Fecha de inicio", default_start)
    end_date = st.sidebar.date_input("Fecha de fin", default_end)
    start_ms = int(datetime.combine(start_date, datetime.min.time()).timestamp() * 1000)
    end_ms = int(datetime.combine(end_date, datetime.min.time()).timestamp() * 1000)

    # 3) Parámetros de Predicción
    st.sidebar.subheader("Parámetros de Predicción")
    horizon = st.sidebar.slider(
        "Días a predecir:",
        min_value=1, max_value=60, value=30,
        help="Número de días a futuro a predecir."
    )
    auto_window = min(60, max(5, horizon * 2))
    st.sidebar.markdown(f"**Tamaño de ventana (auto): {auto_window} días**")

    use_indicators = st.sidebar.checkbox(
        "Incluir indicadores técnicos (RSI, MACD, BBANDS)",
        value=True,
        help="Calcula indicadores técnicos localmente para enriquecer los datos."
    )

    # 4) Escenario del modelo (hiperparámetros)
    st.sidebar.subheader("Escenario del Modelo")
    scenario = st.sidebar.selectbox(
        "Elige un escenario:",
        ["Pesimista", "Neutro", "Optimista"],
        help="Ajusta automáticamente los hiperparámetros del modelo."
    )
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

    # Descargamos datos para mostrar el histórico (gráfico diario)
    df_prices = load_coincap_data(coin_id, start_ms=start_ms, end_ms=end_ms)
    if df_prices is not None and len(df_prices) > 0:
        df_chart = df_prices.copy()
        # Usamos formato '%Y-%m-%d' para no mostrar horas
        df_chart["ds_str"] = df_chart["ds"].dt.strftime("%Y-%m-%d")

        # Creamos el gráfico con Plotly Express
        fig_hist = px.line(
            df_chart, x="ds_str", y="close_price",
            title=f"Histórico de {crypto_name} (Diario)",
            labels={"ds_str": "Fecha", "close_price": "Precio en USD"}
        )
        # Ajuste del formato del eje Y para no usar notación científica
        fig_hist.update_yaxes(tickformat=",.2f")
        # Reducimos el número de ticks en el eje X
        fig_hist.update_layout(xaxis=dict(type="category", tickangle=45, nticks=10))

        st.plotly_chart(fig_hist, use_container_width=True)
    else:
        st.info("No se encontraron datos históricos válidos con los parámetros seleccionados. Reajusta fechas.")

    # Pestañas para separar Entrenamiento/Test y Predicción
    tabs = st.tabs(["🤖 Entrenamiento y Test", f"🔮 Predicción de Precios - {crypto_name}"])

    with tabs[0]:
        st.header("Entrenamiento del Modelo y Evaluación en Test")
        if st.button("Entrenar Modelo y Predecir", key="train_test"):
            with st.spinner("Entrenando el modelo, por favor espera..."):
                result = train_and_predict(
                    coin_id=coin_id,
                    start_ms=start_ms,
                    end_ms=end_ms,
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

                # Métricas
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
                    yaxis_title="Precio en USD"
                )
                fig_test.update_yaxes(tickformat=",.2f")  # Eje Y sin notación científica
                st.plotly_chart(fig_test, use_container_width=True)
            else:
                st.info("No se pudo entrenar el modelo con los parámetros seleccionados. Revisa los avisos arriba.")

    with tabs[1]:
        st.header(f"Predicción de Precios - {crypto_name}")
        if 'result' in locals() and result is not None:
            df_model, test_preds, y_test_real, future_preds, rmse, mape = result
            last_date = df_model["ds"].iloc[-1]
            current_price = df_model["close_price"].iloc[-1]

            # Generamos fechas diarias a futuro
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
                yaxis_title="Precio en USD"
            )
            fig_future.update_yaxes(tickformat=",.2f")  # Formato decimal
            st.plotly_chart(fig_future, use_container_width=True)

            st.subheader("Valores Numéricos de la Predicción Futura")
            future_df = pd.DataFrame({"Fecha": future_dates, "Predicción": pred_series})
            st.dataframe(future_df)
        else:
            st.info("Primero entrena el modelo en la pestaña 'Entrenamiento y Test' para generar las predicciones futuras.")


if __name__ == "__main__":
    main_app()
