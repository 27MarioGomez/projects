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

# Modelo LSTM
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, Bidirectional, LSTM, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

# Random Forest para ensamblado
from sklearn.ensemble import RandomForestRegressor

def main_app():
    """
    App para predecir precios de criptomonedas usando un ensamble de:
      1) Modelo híbrido (Conv1D + LSTM)
      2) RandomForestRegressor
    Y calculando indicadores técnicos (RSI, MACD, BBANDS) localmente con pandas_ta.
    """

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
    st.markdown("**Fuente de Datos:** Alpha Vantage (serie diaria, actualizada cada día)")

    # -----------------------------
    # 2. BARRA LATERAL: CONFIGURACIÓN
    # -----------------------------
    st.sidebar.header("Configuración de la predicción")
    alpha_symbols = {
        "Bitcoin (BTC)":      "BTC",
        "Ethereum (ETH)":     "ETH",
        "XRP":                "XRP",
        "Stellar (XLM)":      "XLM",
        "Solana (SOL)":       "SOL",
        "Cardano (ADA)":      "ADA",
        "Dogecoin (DOGE)":    "DOGE",
        "Polkadot (DOT)":     "DOT",
        "Polygon (MATIC)":    "MATIC",
        "Litecoin (LTC)":     "LTC",
        "TRON (TRX)":         "TRX",
        "Binance Coin (BNB)": "BNB"
    }
    crypto_name = st.sidebar.selectbox(
        "Selecciona una criptomoneda:", 
        list(alpha_symbols.keys()),
        help="Elige la criptomoneda para la cual se realizará la predicción."
    )
    symbol = alpha_symbols[crypto_name]

    st.sidebar.subheader("Parámetros de Predicción Básicos")
    horizon = st.sidebar.slider(
        "Días a predecir:", 
        min_value=1, max_value=60, value=30,
        help="Número de días a futuro que se desea predecir."
    )
    window_size = st.sidebar.slider(
        "Tamaño de ventana (días):", 
        min_value=5, max_value=60, value=30,
        help="Cantidad de días históricos utilizados para entrenar el modelo."
    )
    use_multivariate = st.sidebar.checkbox(
        "Usar datos multivariados (OHLCV)", value=False,
        help="Incluir open, high, low y volumen además del precio de cierre."
    )
    use_indicators = st.sidebar.checkbox(
        "Incluir indicadores técnicos (RSI, MACD, BBANDS)", value=True,
        help="Calcula indicadores técnicos localmente para enriquecer los datos."
    )

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

    # -----------------------------
    # 3. FUNCIÓN PARA DESCARGAR Y LIMPIAR DATOS DE ALPHA VANTAGE
    # -----------------------------
    @st.cache_data
    def load_and_clean_data(symbol):
        api_key = st.secrets["ALPHA_VANTAGE_API_KEY"]
        url = (
            "https://www.alphavantage.co/query"
            "?function=DIGITAL_CURRENCY_DAILY"
            f"&symbol={symbol}"
            "&market=USD"
            f"&apikey={api_key}"
            "&datatype=csv"
            "&outputsize=full"
        )
        resp = requests.get(url)
        if resp.status_code != 200:
            st.error(f"Error al obtener datos de Alpha Vantage (status code {resp.status_code}).")
            return None
        data_io = StringIO(resp.text)
        df = pd.read_csv(data_io)
        if df.empty:
            st.error("El CSV obtenido de Alpha Vantage está vacío.")
            return None
        df.rename(columns={
            "timestamp":   "ds",
            "open":        "open_price",
            "high":        "high_price",
            "low":         "low_price",
            "close":       "close_price",
            "volume":      "volume",
            "market cap":  "market_cap"
        }, inplace=True)
        df["ds"] = pd.to_datetime(df["ds"], errors="coerce")
        df.dropna(subset=["ds"], inplace=True)
        df.sort_values(by="ds", ascending=True, inplace=True)
        df.reset_index(drop=True, inplace=True)
        return df

    # -----------------------------
    # 4. FUNCIÓN PARA AÑADIR INDICADORES TÉCNICOS
    # -----------------------------
    def add_indicators(df):
        df["rsi"] = ta.rsi(df["close_price"], length=14)
        macd_df = ta.macd(df["close_price"])
        bbands_df = ta.bbands(df["close_price"], length=20, std=2)
        df = pd.concat([df, macd_df, bbands_df], axis=1)
        df.ffill(inplace=True)
        return df

    # -----------------------------
    # 5. CREACIÓN DE SECUENCIAS (VENTANAS)
    # -----------------------------
    def create_sequences(data, window_size=30):
        if len(data) <= window_size:
            st.error(f"No hay suficientes datos para una ventana de {window_size} días.")
            return None, None
        X, y = [], []
        for i in range(window_size, len(data)):
            X.append(data[i - window_size : i])
            y.append(data[i, 0])
        return np.array(X), np.array(y)

    # -----------------------------
    # 6. MODELO HÍBRIDO (Conv1D + LSTM)
    # -----------------------------
    def build_hybrid_model(input_shape, learning_rate=0.001):
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

    # -----------------------------
    # 7. ENTRENAMIENTO Y PREDICCIÓN ENSAMBLADO
    # -----------------------------
    def train_and_predict(symbol, horizon_days=30, window_size=30, test_size=0.2,
                          use_multivariate=False, use_indicators=False,
                          epochs=10, batch_size=32, learning_rate=0.001):
        """
        Entrena un ensamble de:
          1) Modelo híbrido (Conv1D + LSTM).
          2) RandomForestRegressor.
        Se hace un promedio de ambas predicciones.
        """
        df_prices = load_and_clean_data(symbol)
        if df_prices is None:
            return None

        # Añadir indicadores si se requiere
        if use_indicators:
            df_prices = add_indicators(df_prices)

        # Selección de features
        if use_multivariate or use_indicators:
            feature_cols = ["close_price", "open_price", "high_price", "low_price", "volume"]
            for col in ["rsi", "MACD_12_26_9", "MACDs_12_26_9", "MACDh_12_26_9", 
                        "BBL_20_2.0", "BBM_20_2.0", "BBU_20_2.0"]:
                if col in df_prices.columns:
                    feature_cols.append(col)
            df_model = df_prices[["ds"] + feature_cols].copy()
            data_for_model = df_model[feature_cols].values
            # Escaladores
            scaler_features = MinMaxScaler(feature_range=(0, 1))
            scaled_data = scaler_features.fit_transform(data_for_model)
            scaler_target = MinMaxScaler(feature_range=(0, 1))
            scaler_target.fit(df_model[["close_price"]])
        else:
            df_model = df_prices[["ds", "close_price"]].copy()
            data_for_model = df_model[["close_price"]].values
            scaler_target = MinMaxScaler(feature_range=(0, 1))
            scaled_data = scaler_target.fit_transform(data_for_model)

        # División train/test
        split_index = int(len(scaled_data) * (1 - test_size))
        train_data = scaled_data[:split_index]
        test_data = scaled_data[split_index:]

        X_train, y_train = create_sequences(train_data, window_size=window_size)
        if X_train is None:
            return None
        X_test, y_test = create_sequences(test_data, window_size=window_size)
        if X_test is None:
            return None

        # Train/val
        val_split = int(len(X_train) * 0.9)
        X_val, y_val = X_train[val_split:], y_train[val_split:]
        X_train, y_train = X_train[:val_split], y_train[:val_split]

        # -----------------------------
        # 7.1 Entrenamiento del modelo LSTM
        # -----------------------------
        input_shape = (X_train.shape[1], X_train.shape[2])
        lstm_model = build_hybrid_model(input_shape, learning_rate=learning_rate)
        early_stop = EarlyStopping(monitor="val_loss", patience=5, restore_best_weights=True)
        lr_reducer = ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=3)
        lstm_model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=epochs, batch_size=batch_size,
            verbose=1, callbacks=[early_stop, lr_reducer]
        )

        # -----------------------------
        # 7.2 Entrenamiento del RandomForestRegressor
        # -----------------------------
        # Aplanamos X_train para el random forest (2D)
        # Estructura: (samples, features * window_size)
        # Sin la parte secuencial, se hace un flatten
        X_train_flat = X_train.reshape(X_train.shape[0], -1)
        X_val_flat = X_val.reshape(X_val.shape[0], -1)
        X_test_flat = X_test.reshape(X_test.shape[0], -1)

        rf_model = RandomForestRegressor(
            n_estimators=100,
            max_depth=10,
            random_state=42,
            n_jobs=-1
        )
        # Entrenamos con (X_train_flat + y_train)
        rf_model.fit(X_train_flat, y_train)

        # -----------------------------
        # 7.3 Predicción en test con ambos modelos
        # -----------------------------
        # LSTM
        lstm_preds_test = lstm_model.predict(X_test)
        # RandomForest
        rf_preds_test = rf_model.predict(X_test_flat).reshape(-1, 1)

        # Ensamble: promedio
        ensemble_test = (lstm_preds_test + rf_preds_test) / 2.0

        # Desescalado
        ensemble_test_descaled = scaler_target.inverse_transform(ensemble_test)
        y_test_deserialized = scaler_target.inverse_transform(y_test.reshape(-1, 1))

        # Evitar nan en MAPE si hay ceros en el test
        # Filtramos filas donde y_test_deserialized no sea 0
        mask_nonzero = (y_test_deserialized != 0).flatten()
        y_test_valid = y_test_deserialized[mask_nonzero]
        ensemble_test_valid = ensemble_test_descaled[mask_nonzero]
        if len(y_test_valid) == 0:
            # Si todo es 0, forzamos un error
            rmse = np.nan
            mape = np.nan
        else:
            rmse = np.sqrt(np.mean((y_test_valid - ensemble_test_valid) ** 2))
            mape = np.mean(np.abs((y_test_valid - ensemble_test_valid) / y_test_valid)) * 100

        # -----------------------------
        # 7.4 Predicción futura iterativa
        # -----------------------------
        # La LSTM se usa de forma iterativa (igual que antes).
        # El random forest no es autoregresivo, así que se combina sólo con la LSTM
        # en cada paso. Para simplificar, usaremos sólo la LSTM en la predicción a futuro
        # o un 50/50. Te muestro un ejemplo con 50/50:
        last_window = scaled_data[-window_size:]
        future_preds_scaled = []
        current_input = last_window.reshape(1, window_size, X_train.shape[2])

        for _ in range(horizon_days):
            # Predicción LSTM
            lstm_future_pred = lstm_model.predict(current_input)[0][0]

            # Para random forest, aplanamos current_input
            # y tomamos la media final
            rf_input = current_input.reshape(1, -1)  # Flatten
            rf_future_pred = rf_model.predict(rf_input)[0]
            ensemble_future = (lstm_future_pred + rf_future_pred) / 2.0

            future_preds_scaled.append(ensemble_future)

            new_feature = np.zeros((1, 1, X_train.shape[2]))
            new_feature[0, 0, 0] = ensemble_future
            current_input = np.append(current_input[:, 1:, :], new_feature, axis=1)

        future_preds = scaler_target.inverse_transform(np.array(future_preds_scaled).reshape(-1, 1)).flatten()

        return df_model, ensemble_test_descaled, y_test_deserialized, future_preds, rmse, mape

    # -----------------------------
    # 8. VISUALIZACIÓN HISTÓRICA
    # -----------------------------
    df_prices = load_and_clean_data(symbol)
    if df_prices is not None and len(df_prices) > 0:
        df_chart = df_prices.copy()
        df_chart["ds"] = df_chart["ds"].dt.strftime("%d-%m-%Y")
        fig_hist = px.line(
            df_chart, x="ds", y="close_price",
            title=f"Histórico de Precio de {crypto_name}",
            labels={"ds": "Fecha", "close_price": "Precio de Cierre"}
        )
        fig_hist.update_layout(xaxis=dict(type="category", tickangle=45))
        st.plotly_chart(fig_hist, use_container_width=True)
    else:
        st.warning("No se encontraron datos históricos válidos para mostrar el gráfico.")

    # -----------------------------
    # 9. PESTAÑAS: ENTRENAMIENTO/TEST Y PREDICCIÓN
    # -----------------------------
    tabs = st.tabs(["🤖 Entrenamiento y Test", f"🔮 Predicción de Precios - {crypto_name}"])

    with tabs[0]:
        st.header("Entrenamiento del Modelo y Evaluación en Test")
        if st.button("Entrenar Modelo y Predecir", key="train_test"):
            with st.spinner("Entrenando el modelo, por favor espera..."):
                result = train_and_predict(
                    symbol=symbol,
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
                df_model, test_preds, y_test_real, future_preds, rmse, mape = result
                st.success("Entrenamiento y predicción completados!")
                col1, col2 = st.columns(2)
                if pd.isna(rmse) or pd.isna(mape):
                    col1.metric("RMSE (Test)", "NaN")
                    col2.metric("MAPE (Test)", "NaN%")
                else:
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
                    name="Ensamble (Test)"
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
                name="Predicción Futura (Ensamble)"
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
