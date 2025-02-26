#########################
# main.py
#########################

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
import requests
from io import StringIO
from datetime import datetime
from sklearn.preprocessing import MinMaxScaler
import pandas_ta as ta
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, Bidirectional, LSTM, Dense, Dropout, Input
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

def main_app():
    # -----------------------------
    # 1. CONFIGURACIÓN DE LA PÁGINA Y ESTILO
    # -----------------------------
    st.set_page_config(page_title="Crypto Price Prediction Dashboard", layout="wide")
    st.markdown(
        """
        <style>
        .reportview-container { background: #F5F5F5; }
        .sidebar .sidebar-content { background-image: linear-gradient(#2E7BCF, #2E7BCF); color: white; }
        .stButton>button { background-color: #2E7BCF; color: white; }
        </style>
        """,
        unsafe_allow_html=True
    )
    st.title("Crypto Price Predictions 🔮")
    st.markdown("**Fuente de Datos:** CoinGecko + Indicadores calculados localmente (pandas_ta)")

    # -----------------------------
    # 2. BARRA LATERAL: CONFIGURACIÓN PRINCIPAL
    # -----------------------------
    st.sidebar.header("Configuración de la predicción")
    # Diccionario de criptomonedas con IDs de CoinGecko
    coin_ids = {
        "Bitcoin (BTC)":      "bitcoin",
        "Ethereum (ETH)":     "ethereum",
        "XRP":                "ripple",
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
    crypto_choice = st.sidebar.selectbox("Selecciona una criptomoneda:", list(coin_ids.keys()))
    coin_id = coin_ids[crypto_choice]

    st.sidebar.subheader("Parámetros de Predicción Básicos")
    horizon = st.sidebar.slider("Días a predecir:", min_value=1, max_value=60, value=30)
    window_size = st.sidebar.slider("Tamaño de ventana (días):", min_value=5, max_value=60, value=30)
    use_multivariate = st.sidebar.checkbox("Usar datos multivariados (OHLCV)", value=False)

    # Opción para incluir indicadores técnicos (RSI, MACD, BBANDS)
    use_indicators = st.sidebar.checkbox("Incluir indicadores técnicos (RSI, MACD, BBANDS)", value=True)

    st.sidebar.subheader("Escenario del Modelo")
    scenario = st.sidebar.selectbox("Elige un escenario:", ["Pesimista", "Neutro", "Optimista"])
    if scenario == "Pesimista":
        epochs_val = 20
        batch_size_val = 32
        learning_rate_val = 0.001
    elif scenario == "Neutro":
        epochs_val = 30
        batch_size_val = 32
        learning_rate_val = 0.0008
    else:  # Optimista
        epochs_val = 50
        batch_size_val = 16
        learning_rate_val = 0.0005

    # -----------------------------
    # 3. DESCARGA DE DATOS HISTÓRICOS DESDE COINGECKO
    # -----------------------------
    @st.cache_data
    def load_data_coingecko(coin_id, vs_currency="usd", days="max"):
        """
        Descarga el histórico de precios desde CoinGecko.
        Devuelve un DataFrame con columnas 'ds' y 'close_price'.
        """
        url = f"https://api.coingecko.com/api/v3/coins/{coin_id}/market_chart?vs_currency={vs_currency}&days={days}"
        resp = requests.get(url)
        if resp.status_code != 200:
            st.error("Error al obtener datos de CoinGecko.")
            return None
        data = resp.json()
        # 'prices' es una lista de [timestamp, price]
        df = pd.DataFrame(data["prices"], columns=["timestamp", "close_price"])
        df["ds"] = pd.to_datetime(df["timestamp"], unit="ms")
        df = df[["ds", "close_price"]]
        df.sort_values(by="ds", ascending=True, inplace=True)
        df.reset_index(drop=True, inplace=True)
        return df

    # -----------------------------
    # 4. CÁLCULO DE INDICADORES TÉCNICOS CON PANDAS_TA
    # -----------------------------
    def add_indicators(df):
        """
        Calcula RSI, MACD y Bollinger Bands a partir de los datos de precios.
        Devuelve el DataFrame con nuevas columnas:
          - 'rsi'
          - 'MACD', 'MACD_Signal', 'MACD_Hist'
          - 'BBL', 'BBM', 'BBU' (Bollinger Bands Lower, Middle, Upper)
        Se realiza ffill para alinear datos.
        """
        df["rsi"] = ta.rsi(df["close_price"], length=14)
        macd_df = ta.macd(df["close_price"])  # Por defecto: MACD_12_26_9, MACDs_12_26_9, MACDh_12_26_9
        bbands_df = ta.bbands(df["close_price"], length=20, std=2)  # BBL_20_2.0, BBM_20_2.0, BBU_20_2.0
        df = pd.concat([df, macd_df, bbands_df], axis=1)
        df.ffill(inplace=True)
        return df

    # -----------------------------
    # 5. CREACIÓN DE SECUENCIAS PARA LA LSTM
    # -----------------------------
    def create_sequences(data, window_size=30):
        """
        Genera secuencias de datos de longitud 'window_size'.
        La primera columna se asume que es el target (close_price).
        """
        if len(data) <= window_size:
            st.error(f"No hay suficientes datos para una ventana de {window_size} días.")
            return None, None
        X, y = [], []
        for i in range(window_size, len(data)):
            X.append(data[i - window_size : i])
            y.append(data[i, 0])
        return np.array(X), np.array(y)

    # -----------------------------
    # 6. CONSTRUCCIÓN DEL MODELO HÍBRIDO (Conv1D + LSTM)
    # -----------------------------
    def build_hybrid_model(input_shape, learning_rate=0.001):
        """
        Construye un modelo híbrido que combina una capa Conv1D para extraer
        características locales seguida de capas Bidirectional LSTM para capturar
        dependencias a largo plazo.
        """
        model = Sequential()
        # Capa Conv1D para extraer patrones locales
        model.add(Conv1D(filters=32, kernel_size=3, activation="relu", input_shape=input_shape))
        # Capa LSTM bidireccional
        model.add(Bidirectional(LSTM(64, return_sequences=True)))
        model.add(Dropout(0.3))
        model.add(Bidirectional(LSTM(64, return_sequences=True)))
        model.add(Dropout(0.3))
        model.add(Bidirectional(LSTM(64, return_sequences=False)))
        model.add(Dropout(0.3))
        model.add(Dense(1))
        optimizer = Adam(learning_rate=learning_rate)
        model.compile(optimizer=optimizer, loss="mean_squared_error")
        return model

    # -----------------------------
    # 7. ENTRENAMIENTO Y PREDICCIÓN CON EL MODELO HÍBRIDO
    # -----------------------------
    def train_and_predict(coin_id, horizon_days=30, window_size=30, test_size=0.2,
                          use_multivariate=False, use_indicators=False,
                          epochs=10, batch_size=32, learning_rate=0.001):
        """
        Descarga datos desde CoinGecko, calcula indicadores técnicos (si se solicita),
        prepara las secuencias, entrena el modelo híbrido y genera predicciones.
        """
        # Cargar datos históricos desde CoinGecko
        df_prices = load_data_coingecko(coin_id, vs_currency="usd", days="max")
        if df_prices is None:
            return None
        # Si se solicitan indicadores técnicos, se calculan y se añaden
        if use_indicators:
            df_prices = add_indicators(df_prices)
        # Si se usan datos multivariados o indicadores, se preparan múltiples features;
        # en caso contrario, solo se usa 'close_price'
        if use_multivariate or use_indicators:
            feature_cols = ["close_price", "open_price", "high_price", "low_price", "volume"]
            for col in ["rsi", "MACD_12_26_9", "MACDs_12_26_9", "MACDh_12_26_9", "BBL_20_2.0", "BBM_20_2.0", "BBU_20_2.0"]:
                if col in df_prices.columns:
                    feature_cols.append(col)
            df_model = df_prices[["ds"] + feature_cols].copy()
            data_for_model = df_model[feature_cols].values
            scaler_features = MinMaxScaler(feature_range=(0, 1))
            scaled_data = scaler_features.fit_transform(data_for_model)
            scaler_target = MinMaxScaler(feature_range=(0, 1))
            scaler_target.fit(df_model[["close_price"]])
        else:
            df_model = df_prices[["ds", "close_price"]].copy()
            data_for_model = df_model[["close_price"]].values
            scaler_target = MinMaxScaler(feature_range=(0, 1))
            scaled_data = scaler_target.fit_transform(data_for_model)

        # División en datos de entrenamiento y test
        split_index = int(len(scaled_data) * (1 - test_size))
        train_data = scaled_data[:split_index]
        test_data = scaled_data[split_index:]

        X_train, y_train = create_sequences(train_data, window_size=window_size)
        if X_train is None:
            return None
        X_test, y_test = create_sequences(test_data, window_size=window_size)
        if X_test is None:
            return None

        # División de validación del entrenamiento (90%/10%)
        val_split = int(len(X_train) * 0.9)
        X_val, y_val = X_train[val_split:], y_train[val_split:]
        X_train, y_train = X_train[:val_split], y_train[:val_split]

        # Construcción del modelo híbrido
        input_shape = (X_train.shape[1], X_train.shape[2])
        model = build_hybrid_model(input_shape, learning_rate=learning_rate)

        # Entrenamiento con callbacks
        early_stop = EarlyStopping(monitor="val_loss", patience=5, restore_best_weights=True)
        lr_reducer = ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=3)
        model.fit(X_train, y_train, validation_data=(X_val, y_val),
                  epochs=epochs, batch_size=batch_size, verbose=1,
                  callbacks=[early_stop, lr_reducer])

        # Predicción en test
        test_predictions = model.predict(X_test)
        test_predictions_descaled = scaler_target.inverse_transform(test_predictions)
        y_test_deserialized = scaler_target.inverse_transform(y_test.reshape(-1, 1))
        rmse = np.sqrt(np.mean((y_test_deserialized - test_predictions_descaled) ** 2))
        mape = np.mean(np.abs((y_test_deserialized - test_predictions_descaled) / y_test_deserialized)) * 100

        # Predicción a futuro (iterativa)
        last_window = scaled_data[-window_size:]
        future_preds_scaled = []
        current_input = last_window.reshape(1, window_size, X_train.shape[2])
        for _ in range(horizon_days):
            future_pred = model.predict(current_input)[0][0]
            future_preds_scaled.append(future_pred)
            new_feature = np.zeros((1, 1, X_train.shape[2]))
            new_feature[0, 0, 0] = future_pred
            current_input = np.append(current_input[:, 1:, :], new_feature, axis=1)
        future_preds = scaler_target.inverse_transform(np.array(future_preds_scaled).reshape(-1, 1)).flatten()

        return df_model, test_predictions_descaled, y_test_deserialized, future_preds, rmse, mape

    # ---------------------------------------------------------
    # 8. VISUALIZACIÓN DEL HISTÓRICO DE PRECIOS (SIN HORAS)
    # ---------------------------------------------------------
    df_prices = load_data_coingecko(coin_id, vs_currency="usd", days="max")
    if df_prices is not None and len(df_prices) > 0:
        df_chart = df_prices.copy()
        df_chart["ds"] = df_chart["ds"].dt.strftime("%d-%m-%Y")
        fig_hist = px.line(
            df_chart, x="ds", y="close_price",
            title=f"Histórico de Precio de {crypto_choice}",
            labels={"ds": "Fecha", "close_price": "Precio de Cierre"}
        )
        fig_hist.update_layout(xaxis=dict(type="category", tickangle=45))
        st.plotly_chart(fig_hist, use_container_width=True)
    else:
        st.warning("No se encontraron datos históricos válidos para mostrar el gráfico.")

    # ---------------------------------------------------------
    # 9. PESTAÑAS: ENTRENAMIENTO/TEST Y PREDICCIÓN
    # ---------------------------------------------------------
    tabs = st.tabs(["🤖 Entrenamiento y Test", f"🔮 Predicción de Precios - {crypto_choice}"])

    with tabs[0]:
        st.header("Entrenamiento del Modelo y Evaluación en Test")
        if st.button("Entrenar Modelo y Predecir", key="train_test"):
            with st.spinner("Entrenando el modelo, por favor espera..."):
                result = train_and_predict(
                    coin_id=coin_id,
                    horizon_days=horizon,
                    window_size=window_size,
                    test_size=0.2,
                    use_multivariate=use_multivariate,
                    use_indicators=use_indicators,
                    epochs=epochs_val,
                    batch_size=batch_size_val,
                    learning_rate=learning_rate_val
                )
            if result is not None:
                df_model, test_preds, y_test_deserialized, future_preds, rmse, mape = result
                st.success("Entrenamiento y predicción completados!")
                col1, col2 = st.columns(2)
                col1.metric("RMSE (Test)", f"{rmse:.2f}")
                col2.metric("MAPE (Test)", f"{mape:.2f}%")
                st.subheader("Comparación en el Set de Test")
                test_dates = df_model["ds"].iloc[-len(y_test_deserialized):]
                fig_test = go.Figure()
                fig_test.add_trace(go.Scatter(
                    x=test_dates,
                    y=y_test_deserialized.flatten(),
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
                    title=f"Comparación en Test: {crypto_choice}",
                    xaxis_title="Fecha",
                    yaxis_title="Precio"
                )
                st.plotly_chart(fig_test, use_container_width=True)
            else:
                st.error("No se pudo entrenar el modelo debido a un error en la carga de datos.")

    with tabs[1]:
        st.header(f"Predicción de Precios - {crypto_choice}")
        if 'result' in locals() and result is not None:
            df_model, test_preds, y_test_deserialized, future_preds, rmse, mape = result
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
                title=f"Predicción a Futuro ({horizon} días) - {crypto_choice}",
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
