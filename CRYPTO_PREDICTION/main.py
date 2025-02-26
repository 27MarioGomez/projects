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
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, Bidirectional
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

def main_app():
    """
    Aplicación Streamlit para predecir precios de criptomonedas, integrando
    indicadores técnicos de Alpha Vantage siempre que estén disponibles.
    """

    # ---------------------------------------------------------
    # 1. CONFIGURACIÓN DE LA PÁGINA Y ESTILO
    # ---------------------------------------------------------
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

    # ---------------------------------------------------------
    # 2. BARRA LATERAL: CONFIGURACIÓN PRINCIPAL
    # ---------------------------------------------------------
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
    crypto_choice = st.sidebar.selectbox("Selecciona una criptomoneda:", list(alpha_symbols.keys()))
    symbol = alpha_symbols[crypto_choice]

    st.sidebar.subheader("Parámetros de Predicción Básicos")
    horizon = st.sidebar.slider("Días a predecir:", min_value=1, max_value=60, value=30)
    window_size = st.sidebar.slider("Tamaño de ventana (días):", min_value=5, max_value=60, value=30)
    use_multivariate = st.sidebar.checkbox("Usar datos multivariados (Open, High, Low, Volume)", value=False)

    # Checkbox para incluir indicadores técnicos (RSI, MACD, BBANDS) si están disponibles
    use_indicators = st.sidebar.checkbox(
        "Incluir indicadores técnicos (RSI, MACD, BBANDS)",
        value=True,
        help="Descarga y fusiona los indicadores RSI, MACD y Bollinger Bands, siempre que Alpha Vantage los provea."
    )

    # Escenario del modelo
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

    # ---------------------------------------------------------
    # 3. DESCARGA DE DATOS DE PRECIOS
    # ---------------------------------------------------------
    @st.cache_data
    def load_and_clean_data(symbol):
        """
        Descarga el CSV diario de Alpha Vantage para la cripto dada,
        renombra las columnas y ordena el DataFrame por fecha.
        """
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
            st.error("Error al obtener datos de precios en Alpha Vantage.")
            return None
        data_io = StringIO(resp.text)
        df = pd.read_csv(data_io)
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

    # ---------------------------------------------------------
    # 4. DESCARGA DE INDICADORES TÉCNICOS
    # ---------------------------------------------------------
    def load_rsi_data(symbol):
        """Descarga el RSI diario para la criptomoneda dada, si existe."""
        api_key = st.secrets["ALPHA_VANTAGE_API_KEY"]
        url = (
            "https://www.alphavantage.co/query"
            "?function=RSI"
            f"&symbol={symbol}"
            "&market=USD"
            "&interval=daily"
            "&time_period=14"
            "&series_type=close"
            f"&apikey={api_key}"
            "&datatype=csv"
        )
        resp = requests.get(url)
        if resp.status_code != 200:
            return None
        data_io = StringIO(resp.text)
        df = pd.read_csv(data_io)
        if not {"time", "RSI"}.issubset(df.columns):
            return None
        df.rename(columns={"time": "ds", "RSI": "rsi"}, inplace=True)
        df["ds"] = pd.to_datetime(df["ds"], errors="coerce")
        df.dropna(subset=["ds"], inplace=True)
        df.sort_values(by="ds", ascending=True, inplace=True)
        df.reset_index(drop=True, inplace=True)
        return df

    def load_bbands_data(symbol):
        """Descarga las Bollinger Bands diarias para la cripto dada, si existen."""
        api_key = st.secrets["ALPHA_VANTAGE_API_KEY"]
        url = (
            "https://www.alphavantage.co/query"
            "?function=BBANDS"
            f"&symbol={symbol}"
            "&market=USD"
            "&interval=daily"
            "&time_period=20"
            "&series_type=close"
            f"&apikey={api_key}"
            "&datatype=csv"
        )
        resp = requests.get(url)
        if resp.status_code != 200:
            return None
        data_io = StringIO(resp.text)
        df = pd.read_csv(data_io)
        expected_cols = {"time", "Real Upper Band", "Real Middle Band", "Real Lower Band"}
        if not expected_cols.issubset(df.columns):
            return None
        df.rename(columns={
            "time": "ds",
            "Real Upper Band": "upper_band",
            "Real Middle Band": "middle_band",
            "Real Lower Band": "lower_band"
        }, inplace=True)
        df["ds"] = pd.to_datetime(df["ds"], errors="coerce")
        df.dropna(subset=["ds"], inplace=True)
        df.sort_values(by="ds", ascending=True, inplace=True)
        df.reset_index(drop=True, inplace=True)
        return df

    def load_macd_data(symbol):
        """Descarga el MACD diario para la criptomoneda dada, si existe."""
        api_key = st.secrets["ALPHA_VANTAGE_API_KEY"]
        url = (
            "https://www.alphavantage.co/query"
            "?function=MACD"
            f"&symbol={symbol}"
            "&market=USD"
            "&interval=daily"
            "&series_type=close"
            f"&apikey={api_key}"
            "&datatype=csv"
        )
        resp = requests.get(url)
        if resp.status_code != 200:
            return None
        data_io = StringIO(resp.text)
        df = pd.read_csv(data_io)
        if not {"time", "MACD"}.issubset(df.columns):
            return None
        df.rename(columns={"time": "ds", "MACD": "macd"}, inplace=True)
        df["ds"] = pd.to_datetime(df["ds"], errors="coerce")
        df.dropna(subset=["ds"], inplace=True)
        df.sort_values(by="ds", ascending=True, inplace=True)
        df.reset_index(drop=True, inplace=True)
        return df

    def merge_indicators(df_prices, df_rsi, df_bbands, df_macd):
        """
        Combina los datos de precios con RSI, Bollinger Bands y MACD por 'ds'.
        Se hace un merge left y luego se aplica ffill para rellenar huecos.
        """
        merged = df_prices.copy()
        if df_rsi is not None:
            merged = pd.merge(merged, df_rsi, on="ds", how="left")
        if df_bbands is not None:
            merged = pd.merge(merged, df_bbands, on="ds", how="left")
        if df_macd is not None:
            merged = pd.merge(merged, df_macd, on="ds", how="left")
        # Rellenamos huecos con forward fill, evitando problemas de NaN
        merged.ffill(inplace=True)
        return merged

    # ---------------------------------------------------------
    # 5. CREACIÓN DE SECUENCIAS
    # ---------------------------------------------------------
    def create_sequences(data, window_size=30):
        if len(data) <= window_size:
            st.error(f"No hay suficientes datos para una ventana de {window_size} días.")
            return None, None
        X, y = [], []
        for i in range(window_size, len(data)):
            X.append(data[i - window_size : i])
            y.append(data[i, 0])  # La primera columna (close_price)
        return np.array(X), np.array(y)

    # ---------------------------------------------------------
    # 6. ENTRENAMIENTO Y PREDICCIÓN CON LSTM
    # ---------------------------------------------------------
    def train_and_predict_lstm(symbol, horizon_days=30, window_size=30, test_size=0.2,
                               use_multivariate=False, use_indicators=False,
                               epochs=10, batch_size=32, learning_rate=0.001):
        """
        Carga datos de precios y, si se solicita, RSI, MACD y BBANDS, los fusiona
        y entrena un modelo LSTM. Luego predice el set de test y un horizonte futuro.
        """
        # Cargamos datos de precios
        df_prices = load_and_clean_data(symbol)
        if df_prices is None:
            return None

        # Si se pide usar indicadores, descargamos lo que esté disponible
        df_rsi = df_bbands = df_macd = None
        if use_indicators:
            df_rsi = load_rsi_data(symbol)
            df_bbands = load_bbands_data(symbol)
            df_macd = load_macd_data(symbol)
            # Hacemos merge de todo
            df_merged = merge_indicators(df_prices, df_rsi, df_bbands, df_macd)
        else:
            df_merged = df_prices.copy()

        # Preparamos el DataFrame final para entrenar
        if use_multivariate or use_indicators:
            # Empezamos con 'close_price' y otras columnas básicas
            feature_cols = ["close_price", "open_price", "high_price", "low_price", "volume"]
            # Agregamos indicadores si están presentes
            for col in ["rsi", "upper_band", "middle_band", "lower_band", "macd"]:
                if col in df_merged.columns:
                    feature_cols.append(col)
            df_model = df_merged[["ds"] + feature_cols].copy()
            data_for_model = df_model[feature_cols].values
            scaler_features = MinMaxScaler(feature_range=(0, 1))
            scaled_data = scaler_features.fit_transform(data_for_model)
            scaler_target = MinMaxScaler(feature_range=(0, 1))
            scaler_target.fit(df_model[["close_price"]])
        else:
            # Solo precio de cierre
            df_model = df_merged[["ds", "close_price"]].copy()
            data_for_model = df_model[["close_price"]].values
            scaler_target = MinMaxScaler(feature_range=(0, 1))
            scaled_data = scaler_target.fit_transform(data_for_model)

        # División en train y test
        split_index = int(len(scaled_data) * (1 - test_size))
        train_data = scaled_data[:split_index]
        test_data = scaled_data[split_index:]

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

        # Construcción del modelo LSTM
        model = Sequential()
        model.add(Bidirectional(LSTM(64, return_sequences=True), input_shape=(X_train.shape[1], X_train.shape[2])))
        model.add(Dropout(0.3))
        model.add(Bidirectional(LSTM(64, return_sequences=True)))
        model.add(Dropout(0.3))
        model.add(Bidirectional(LSTM(64, return_sequences=False)))
        model.add(Dropout(0.3))
        model.add(Dense(1))
        optimizer = Adam(learning_rate=learning_rate)
        model.compile(optimizer=optimizer, loss="mean_squared_error")

        # Entrenamiento con callbacks
        early_stop = EarlyStopping(monitor="val_loss", patience=5, restore_best_weights=True)
        lr_reducer = ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=3)
        model.fit(X_train, y_train, validation_data=(X_val, y_val),
                  epochs=epochs, batch_size=batch_size, verbose=1,
                  callbacks=[early_stop, lr_reducer])

        # Predicción en el set de test
        test_predictions = model.predict(X_test)
        test_predictions_descaled = scaler_target.inverse_transform(test_predictions)
        y_test_deserialized = scaler_target.inverse_transform(y_test.reshape(-1, 1))

        # Cálculo de métricas
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
    # 7. VISUALIZACIÓN DEL HISTÓRICO DE PRECIO (SIN HORAS)
    # ---------------------------------------------------------
    df = load_and_clean_data(symbol)
    if df is not None and len(df) > 0:
        df_chart = df.copy()
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
    # 8. PESTAÑAS: ENTRENAMIENTO/TEST Y PREDICCIÓN
    # ---------------------------------------------------------
    tabs = st.tabs(["🤖 Entrenamiento y Test", f"🔮 Predicción de Precios - {crypto_choice}"])

    # Pestaña 0: Entrenamiento y Test
    with tabs[0]:
        st.header("Entrenamiento del Modelo y Evaluación en Test")
        if st.button("Entrenar Modelo y Predecir", key="train_test"):
            with st.spinner("Entrenando el modelo, por favor espera..."):
                result = train_and_predict_lstm(
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

    # Pestaña 1: Predicción de Precios
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
