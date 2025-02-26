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
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, Bidirectional, LSTM, Dense, Dropout
from tensorflow.keras.optimizers import Adam

# Función robusta para calcular MAPE sin división por cero
def robust_mape(y_true, y_pred, eps=1e-9):
    return np.mean(np.abs((y_true - y_pred) / np.maximum(np.abs(y_true), eps))) * 100

# Mapeo para CoinGecko (usado para la descarga de datos)
coingecko_ids = {
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

def main_app():
    """
    App para predecir precios de criptomonedas usando solo un modelo LSTM.
    Se combinan datos históricos de dos fuentes (Alpha Vantage y CoinGecko) para maximizar el histórico.
    Se calculan indicadores técnicos (RSI, MACD, BBANDS) y se entrena un modelo LSTM (Conv1D + Bidirectional LSTM).
    Se calcula automáticamente el tamaño de ventana en función del horizonte elegido.
    """

    # -------------------------------------------------------------
    # 1. Configuración de la página y estilo
    # -------------------------------------------------------------
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
    st.markdown("**Fuente de Datos:** Alpha Vantage y CoinGecko")

    # -------------------------------------------------------------
    # 2. Configuración de la barra lateral
    # -------------------------------------------------------------
    st.sidebar.header("Configuración de la predicción")
    # Selección de criptomoneda (se usa el mismo nombre para ambas fuentes)
    crypto_name = st.sidebar.selectbox(
        "Selecciona una criptomoneda:",
        list(coingecko_ids.keys()),
        help="Elige la criptomoneda para la cual se realizará la predicción."
    )
    # Para Alpha Vantage usamos el mismo símbolo que en nuestro diccionario anterior
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
    symbol_alpha = alpha_symbols[crypto_name]
    # Para CoinGecko, obtenemos el ID de su diccionario
    symbol_cg = coingecko_ids[crypto_name]

    st.sidebar.subheader("Parámetros de Predicción Básicos")
    horizon = st.sidebar.slider(
        "Días a predecir:", 1, 60, 30,
        help="Número de días a futuro que se desea predecir."
    )
    # Se determina automáticamente el tamaño de ventana en función del horizonte (por ejemplo, horizon*2, limitado a 60)
    auto_window = min(60, max(5, horizon * 2))
    st.sidebar.markdown(f"**Tamaño de ventana (auto): {auto_window} días**")
    
    use_multivariate = st.sidebar.checkbox(
        "Usar datos multivariados (OHLCV)", value=False,
        help="Incluir datos de apertura, máximo, mínimo y volumen además del precio de cierre."
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

    # -------------------------------------------------------------
    # 3. Funciones de descarga de datos
    # -------------------------------------------------------------
    @st.cache_data
    def load_alpha_data(symbol):
        """Descarga y limpia datos desde Alpha Vantage."""
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
            st.error(f"Error Alpha Vantage (status code {resp.status_code}).")
            return None
        data_io = StringIO(resp.text)
        df = pd.read_csv(data_io)
        if df.empty:
            st.error("Alpha Vantage: CSV vacío.")
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
        df = df[df["close_price"] > 0].copy()
        return df

    @st.cache_data
    def load_coingecko_data(coin_id, vs_currency="usd", days="max"):
        """Descarga y limpia datos desde CoinGecko."""
        url = f"https://api.coingecko.com/api/v3/coins/{coin_id}/market_chart?vs_currency={vs_currency}&days={days}"
        resp = requests.get(url)
        if resp.status_code != 200:
            st.error(f"Error CoinGecko (status code {resp.status_code}).")
            return None
        data = resp.json()
        # CoinGecko devuelve 'prices': [ [timestamp, price], ... ]
        if "prices" not in data:
            st.error("CoinGecko: Datos no disponibles.")
            return None
        df = pd.DataFrame(data["prices"], columns=["timestamp", "close_price"])
        df["ds"] = pd.to_datetime(df["timestamp"], unit="ms")
        df = df[["ds", "close_price"]]
        df.sort_values(by="ds", ascending=True, inplace=True)
        df.reset_index(drop=True, inplace=True)
        # A veces CoinGecko puede tener valores cero o muy bajos, se filtran
        df = df[df["close_price"] > 0].copy()
        return df

    def load_combined_data(symbol_alpha, coin_id):
        """
        Combina datos de Alpha Vantage y CoinGecko.
        Se hace un merge outer por fecha y se promedian los precios disponibles.
        """
        df_alpha = load_alpha_data(symbol_alpha)
        df_cg = load_coingecko_data(coin_id)
        if df_alpha is None and df_cg is None:
            st.error("No se pudieron descargar datos de ninguna fuente.")
            return None
        if df_alpha is None:
            return df_cg
        if df_cg is None:
            return df_alpha
        # Merge outer por fecha (asumiendo que 'ds' es la fecha)
        df_merge = pd.merge(df_alpha, df_cg, on="ds", how="outer", suffixes=("_alpha", "_cg"))
        df_merge.sort_values(by="ds", inplace=True)
        # Promediar los precios disponibles (si alguno falta, se usa el que exista)
        df_merge["close_price"] = df_merge[["close_price_alpha", "close_price_cg"]].mean(axis=1)
        # Opcionalmente, se pueden conservar otras columnas de Alpha si se desea
        df_result = df_merge[["ds", "close_price"]].copy()
        df_result.ffill(inplace=True)
        return df_result

    # -------------------------------------------------------------
    # 4. (Opcional) Añadir indicadores técnicos a los datos combinados
    # -------------------------------------------------------------
    def add_all_indicators(df):
        df = add_indicators(df)
        return df

    # -------------------------------------------------------------
    # 5. Creación de secuencias para LSTM
    # -------------------------------------------------------------
    def create_sequences(data, window_size=30):
        if len(data) <= window_size:
            st.error(f"No hay suficientes datos para una ventana de {window_size} días.")
            return None, None
        X, y = [], []
        for i in range(window_size, len(data)):
            X.append(data[i - window_size : i])
            y.append(data[i, 0])
        return np.array(X), np.array(y)

    # -------------------------------------------------------------
    # 6. Modelo LSTM: Conv1D + Bidirectional LSTM
    # -------------------------------------------------------------
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

    # -------------------------------------------------------------
    # 7. Entrenamiento y predicción con LSTM
    # -------------------------------------------------------------
    def train_and_predict(symbol, horizon_days=30, window_size=30, test_size=0.2,
                          use_multivariate=False, use_indicators=False,
                          epochs=10, batch_size=32, learning_rate=0.001):
        # Cargar datos combinados
        df_prices = load_combined_data(symbol_alpha, symbol_cg)
        if df_prices is None:
            st.error("No se pudieron cargar datos combinados.")
            return None
        # Si se quiere usar indicadores, se añaden
        if use_indicators:
            df_prices = add_all_indicators(df_prices)
        # En este ejemplo, para simplificar, se usan solo el precio de cierre
        df_model = df_prices.copy()

        data_for_model = df_model[["close_price"]].values
        scaler_target = MinMaxScaler(feature_range=(0, 1))
        scaled_data = scaler_target.fit_transform(data_for_model)

        # Verificar que hay suficientes datos
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

        # Predicción futura iterativa usando solo LSTM
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

        return df_model, lstm_preds_descaled, y_test_deserialized, future_preds, rmse, mape

    # -------------------------------------------------------------
    # 8. Visualización del histórico (formato DD-MM-YYYY)
    # -------------------------------------------------------------
    df_prices = load_combined_data(symbol_alpha, symbol_cg)
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

    # -------------------------------------------------------------
    # 9. Pestañas: Entrenamiento/Test y Predicción
    # -------------------------------------------------------------
    tabs = st.tabs(["🤖 Entrenamiento y Test", f"🔮 Predicción de Precios - {crypto_name}"])

    with tabs[0]:
        st.header("Entrenamiento del Modelo y Evaluación en Test")
        if st.button("Entrenar Modelo y Predecir", key="train_test"):
            with st.spinner("Entrenando el modelo, por favor espera..."):
                result = train_and_predict(
                    symbol=symbol_alpha,
                    horizon_days=horizon,
                    window_size=auto_window,
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
