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

def robust_mape(y_true, y_pred, eps=1e-9):
    """
    Cálculo robusto de MAPE evitando divisiones por cero.
    """
    return np.mean(np.abs((y_true - y_pred) / np.maximum(np.abs(y_true), eps))) * 100

# Diccionario con IDs de criptomonedas para CoinCap
coincap_ids = {
    "Bitcoin (BTC)":       "bitcoin",
    "Ethereum (ETH)":      "ethereum",
    "Ripple (XRP)":        "xrp",
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

@st.cache_data
def load_coincap_data(coin_id, start_ms=None, end_ms=None, max_retries=3):
    """
    Descarga datos de CoinCap con intervalo diario (d1).
    Intenta además extraer 'volumeUsd' si está disponible.
    """
    url = f"https://api.coincap.io/v2/assets/{coin_id}/history?interval=d1"
    if start_ms is not None and end_ms is not None:
        url += f"&start={start_ms}&end={end_ms}"
    headers = {"User-Agent": "Mozilla/5.0"}

    for attempt in range(max_retries):
        resp = requests.get(url, headers=headers)
        if resp.status_code == 200:
            data = resp.json()
            if "data" not in data:
                st.warning("CoinCap: 'data' faltante.")
                return None
            df = pd.DataFrame(data["data"])
            if df.empty:
                st.info("CoinCap devolvió datos vacíos. Reajusta el rango de fechas.")
                return None
            # Verificamos columnas disponibles
            if "time" not in df.columns or "priceUsd" not in df.columns:
                st.warning("CoinCap: Columnas 'time' o 'priceUsd' no encontradas.")
                st.write(df.head())
                return None

            # Convertimos a datetime y float
            df["ds"] = pd.to_datetime(df["time"], unit="ms")
            df["close_price"] = pd.to_numeric(df["priceUsd"], errors="coerce")

            # Intentamos extraer volumen (si existe)
            if "volumeUsd" in df.columns:
                df["volume"] = pd.to_numeric(df["volumeUsd"], errors="coerce")
            else:
                # Si no existe, creamos una columna de volumen con NaN o 0
                df["volume"] = np.nan  # o df["volume"] = 0.0

            # Ajustamos el DataFrame final
            df = df[["ds", "close_price", "volume"]]
            df.dropna(subset=["ds", "close_price"], inplace=True)
            df.sort_values(by="ds", ascending=True, inplace=True)
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

    st.info("CoinCap: Máx reintentos alcanzado.")
    return None

def add_indicators(df):
    """
    Añade indicadores técnicos usando precio de cierre y volumen si está disponible.
    """
    # RSI, MACD y Bollinger Bands a partir de 'close_price'
    df["rsi"] = ta.rsi(df["close_price"], length=14)
    macd_df = ta.macd(df["close_price"])
    bbands_df = ta.bbands(df["close_price"], length=20, std=2)

    # Si la columna 'volume' existe y no es NaN, podemos calcular OBV u otros indicadores
    if "volume" in df.columns:
        # Ejemplo: On Balance Volume
        df["obv"] = ta.obv(df["close_price"], df["volume"])

    df = pd.concat([df, macd_df, bbands_df], axis=1)
    df.ffill(inplace=True)
    return df

def add_all_indicators(df):
    return add_indicators(df)

def create_sequences(data, window_size=30):
    """
    Crea secuencias de tamaño 'window_size' para entrenar el modelo LSTM.
    data es un array 2D (n_samples, n_features).
    """
    if len(data) <= window_size:
        st.warning(f"No hay datos suficientes para ventana de {window_size} días.")
        return None, None
    X, y = [], []
    for i in range(window_size, len(data)):
        X.append(data[i - window_size : i])
        # y se basa en la primera columna (close_price)
        y.append(data[i, 0])
    return np.array(X), np.array(y)

def build_lstm_model(input_shape, learning_rate=0.001):
    """
    Construye un modelo secuencial que combina:
    - Conv1D
    - 3 capas Bidirectional LSTM con Dropout
    - Dense final
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

def train_and_predict(
    coin_id,
    use_custom_range,
    start_ms,
    end_ms,
    horizon_days=30,
    window_size=30,
    test_size=0.2,
    use_indicators=False,
    epochs=10,
    batch_size=32,
    learning_rate=0.001,
    use_multivariable=False
):
    """
    Descarga datos de CoinCap, añade indicadores si se desea,
    entrena un modelo LSTM (multivariable si use_multivariable=True),
    y realiza predicciones (en test y a futuro).
    """
    # Descarga datos usando rango o histórico completo
    if use_custom_range:
        df_prices = load_coincap_data(coin_id, start_ms=start_ms, end_ms=end_ms)
    else:
        df_prices = load_coincap_data(coin_id)
    if df_prices is None or len(df_prices) == 0:
        st.warning("No se pudo descargar datos suficientes. Reajusta el rango de fechas.")
        return None

    # Añadir indicadores
    if use_indicators:
        df_prices = add_all_indicators(df_prices)

    # Definir las columnas de features
    # La primera columna debe ser 'close_price' para que sea el target principal
    if use_multivariable:
        # Ejemplo: usaremos close_price y volume
        # (además de indicadores si existen)
        # Verificamos si existe la columna 'volume'
        features = ["close_price"]
        if "volume" in df_prices.columns:
            # si no está vacía, la añadimos
            if df_prices["volume"].notna().sum() > 0:
                features.append("volume")

        # Buscamos también las columnas de indicadores que existan
        for col in ["rsi", "MACD_12_26_9", "MACDs_12_26_9", "MACDh_12_26_9",
                    "BBL_20_2.0", "BBM_20_2.0", "BBU_20_2.0", "obv"]:
            if col in df_prices.columns:
                features.append(col)
        # Eliminamos duplicados en caso de colisiones
        features = list(dict.fromkeys(features))
    else:
        # Solo univariado: close_price
        features = ["close_price"]

    # Nos aseguramos de que exista la columna 'close_price'
    if "close_price" not in features:
        st.warning("No se encontró 'close_price' para el entrenamiento.")
        return None

    # Filtramos el DataFrame con las features
    df_model = df_prices[["ds"] + features].copy()
    data_for_model = df_model[features].values

    # Escalado
    # 1) Escalado de todas las features
    scaler_features = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler_features.fit_transform(data_for_model)

    # 2) Escalado específico de la columna 'close_price' para la predicción final
    scaler_target = MinMaxScaler(feature_range=(0, 1))
    scaler_target.fit(df_model[["close_price"]])

    # Split en train/test
    split_index = int(len(scaled_data) * (1 - test_size))
    if split_index <= window_size:
        st.warning("Datos insuficientes para entrenar. Reajusta parámetros.")
        return None

    train_data = scaled_data[:split_index]
    test_data = scaled_data[split_index:]

    # Creación de secuencias
    X_train, y_train = create_sequences(train_data, window_size=window_size)
    if X_train is None:
        return None
    X_test, y_test = create_sequences(test_data, window_size=window_size)
    if X_test is None:
        return None

    # División en train/val
    val_split = int(len(X_train) * 0.9)
    X_val, y_val = X_train[val_split:], y_train[val_split:]
    X_train, y_train = X_train[:val_split], y_train[:val_split]

    # Limpiar la sesión para evitar errores internos de TF
    tf.keras.backend.clear_session()

    # Construir y entrenar el modelo
    input_shape = (X_train.shape[1], X_train.shape[2])  # (window_size, num_features)
    lstm_model = build_lstm_model(input_shape, learning_rate=learning_rate)
    lstm_model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=epochs,
        batch_size=batch_size,
        verbose=1
    )

    # Predicción en test
    test_preds_scaled = lstm_model.predict(X_test)
    test_preds = scaler_target.inverse_transform(test_preds_scaled)  # inverso de la columna close_price
    y_test_deserialized = scaler_target.inverse_transform(y_test.reshape(-1, 1))

    valid_mask = ~np.isnan(test_preds) & ~np.isnan(y_test_deserialized)
    if np.sum(valid_mask) == 0:
        rmse, mape = np.nan, np.nan
    else:
        rmse = np.sqrt(np.mean((y_test_deserialized[valid_mask] - test_preds[valid_mask]) ** 2))
        mape = robust_mape(y_test_deserialized[valid_mask], test_preds[valid_mask])

    # Predicción futura iterativa
    # Tomamos la última ventana y predecimos iterativamente la primera columna (close_price)
    last_window = scaled_data[-window_size:]
    future_preds_scaled = []
    current_input = last_window.reshape(1, window_size, X_train.shape[2])

    @tf.function
    def predict_model(x):
        return lstm_model(x)

    for _ in range(horizon_days):
        future_pred = predict_model(current_input)[0][0]
        future_preds_scaled.append(future_pred)

        # En multivariable, solo estamos prediciendo la primera columna (close_price).
        # Podemos dejar el resto de features fijos o en 0.
        new_feature = np.copy(current_input[:, -1:, :])  # copiamos la última fila
        # Sustituimos la columna 0 (close_price) con la predicción
        new_feature[0, 0, 0] = future_pred

        # El resto de columnas (volumen, indicadores) se podrían dejar igual, 
        # o estimar, o simplemente poner 0
        for c in range(1, X_train.shape[2]):
            new_feature[0, 0, c] = current_input[0, -1, c]  # dejamos fijo

        current_input = np.append(current_input[:, 1:, :], new_feature, axis=1)

    future_preds = scaler_target.inverse_transform(np.array(future_preds_scaled).reshape(-1, 1)).flatten()

    return df_model, test_preds, y_test_deserialized, future_preds, rmse, mape

def main_app():
    st.set_page_config(page_title="Crypto Price Predictions 🔮", layout="wide")
    st.title("Crypto Price Predictions 🔮")
    st.markdown("**Fuente de Datos:** CoinCap")

    st.sidebar.header("Configuración de la predicción")

    crypto_name = st.sidebar.selectbox(
        "Selecciona una criptomoneda:",
        list(coincap_ids.keys()),
        help="Elige la criptomoneda para la predicción."
    )
    coin_id = coincap_ids[crypto_name]

    st.sidebar.subheader("Rango de Fechas")
    use_custom_range = st.sidebar.checkbox(
        "Habilitar rango de fechas",
        value=True,
        help="Si se desactiva, se usará todo el histórico disponible."
    )
    default_start = datetime(2021, 1, 1)
    default_end = datetime.now()
    if use_custom_range:
        start_date = st.sidebar.date_input("Fecha de inicio", default_start)
        end_date = st.sidebar.date_input("Fecha de fin", default_end)
        start_ms = int(datetime.combine(start_date, datetime.min.time()).timestamp() * 1000)
        end_ms = int(datetime.combine(end_date, datetime.min.time()).timestamp() * 1000)
    else:
        start_ms = None
        end_ms = None

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
        help="Calcula indicadores técnicos para enriquecer los datos."
    )

    # Nuevo: permitir multivariable (close_price + volumen + indicadores)
    use_multivariable = st.sidebar.checkbox(
        "Usar multivariable (precio y volumen)",
        value=False,
        help="Incluye volumen y cualquier indicador adicional como features."
    )

    show_stats = st.sidebar.checkbox(
        "Ver estadísticas descriptivas",
        value=False,
        help="Muestra un resumen estadístico del precio."
    )

    st.sidebar.subheader("Escenario del Modelo")
    scenario = st.sidebar.selectbox(
        "Elige un escenario:",
        ["Pesimista", "Neutro", "Optimista"],
        index=0,
        help=("Pesimista: Predicciones conservadoras. Neutro: Balance. "
              "Optimista: Predicciones agresivas con mayor potencial.")
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

    df_prices = load_coincap_data(coin_id, start_ms=start_ms, end_ms=end_ms)
    if df_prices is not None and len(df_prices) > 0:
        df_chart = df_prices.copy()
        df_chart["ds_str"] = df_chart["ds"].dt.strftime("%d/%m/%Y")
        fig_hist = px.line(
            df_chart, x="ds_str", y="close_price",
            title=f"Histórico de {crypto_name}",
            labels={"ds_str": "Fecha", "close_price": "Precio en USD"}
        )
        fig_hist.update_yaxes(tickformat=",.2f")
        fig_hist.update_layout(xaxis=dict(type="category", tickangle=45, nticks=10))
        st.plotly_chart(fig_hist, use_container_width=True)

        if show_stats:
            st.subheader("Estadísticas Descriptivas")
            st.write(df_prices["close_price"].describe().rename({
                "count": "Cuenta",
                "mean": "Media",
                "std": "Desv. Estándar",
                "min": "Mínimo",
                "25%": "Percentil 25",
                "50%": "Mediana",
                "75%": "Percentil 75",
                "max": "Máximo"
            }))
    else:
        st.info("No se encontraron datos históricos válidos. Reajusta el rango de fechas.")

    tabs = st.tabs(["🤖 Entrenamiento y Test", f"🔮 Predicción de Precios - {crypto_name}"])

    with tabs[0]:
        st.header("Entrenamiento del Modelo y Evaluación en Test")
        if st.button("Entrenar Modelo y Predecir", key="train_test"):
            with st.spinner("Entrenando el modelo, por favor espera..."):
                result = train_and_predict(
                    coin_id=coin_id,
                    use_custom_range=use_custom_range,
                    start_ms=start_ms,
                    end_ms=end_ms,
                    horizon_days=horizon,
                    window_size=auto_window,
                    test_size=0.2,
                    use_indicators=use_indicators,
                    epochs=epochs_val,
                    batch_size=batch_size_val,
                    learning_rate=learning_rate_val,
                    use_multivariable=use_multivariable
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
                    yaxis_title="Precio en USD"
                )
                fig_test.update_yaxes(tickformat=",.2f")
                st.plotly_chart(fig_test, use_container_width=True)
            else:
                st.info("No se pudo entrenar el modelo con los parámetros seleccionados.")

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
                yaxis_title="Precio en USD"
            )
            fig_future.update_yaxes(tickformat=",.2f")
            st.plotly_chart(fig_future, use_container_width=True)

            st.subheader("Valores Numéricos de la Predicción Futura")
            future_df = pd.DataFrame({"Fecha": future_dates, "Predicción": pred_series})
            st.dataframe(future_df)
        else:
            st.info("Primero entrena el modelo en la pestaña 'Entrenamiento y Test' para generar las predicciones futuras.")

if __name__ == "__main__":
    main_app()
