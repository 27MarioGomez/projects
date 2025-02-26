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

def robust_mape(y_true, y_pred, eps=1e-9):
    """
    Evita divisiones por cero usando max(eps, |y_true|).
    """
    return np.mean(np.abs((y_true - y_pred) / np.maximum(np.abs(y_true), eps))) * 100

# Diccionarios
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

@st.cache_data
def load_alpha_data(symbol_alpha):
    """
    Descarga datos desde Alpha Vantage y detecta automáticamente la columna de fecha y las columnas de precio.
    """
    api_key = st.secrets["ALPHA_VANTAGE_API_KEY"]
    url = (
        "https://www.alphavantage.co/query"
        "?function=DIGITAL_CURRENCY_DAILY"
        f"&symbol={symbol_alpha}"
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

    # Muestra columnas originales para debug
    st.write("## Columnas devueltas por Alpha Vantage:", df.columns.tolist())

    # Detectar la columna de fecha
    possible_timestamp_cols = ["timestamp", "time", "Date", "date", "Timestamp"]
    date_col = None
    for c in possible_timestamp_cols:
        if c in df.columns:
            date_col = c
            break
    if not date_col:
        st.error("No se encontró ninguna columna de fecha en los datos de Alpha Vantage.")
        st.write(df.head())
        return None

    # Renombramos la columna de fecha a 'ds'
    df.rename(columns={date_col: "ds"}, inplace=True)

    # Detectar columna de 'close' (puede ser 'close', 'close (USD)', etc.)
    # Haremos algo parecido con 'open', 'high', 'low', 'volume', etc.
    # Nota: la API 'DIGITAL_CURRENCY_DAILY' suele devolver 'open (USD)', 'close (USD)', etc.
    for col in df.columns:
        lower_col = col.lower()
        if "open" in lower_col:
            df.rename(columns={col: "open_price"}, inplace=True)
        elif "high" in lower_col:
            df.rename(columns={col: "high_price"}, inplace=True)
        elif "low" in lower_col:
            df.rename(columns={col: "low_price"}, inplace=True)
        elif "close" in lower_col and "market" not in lower_col:  # para evitar 'market cap'
            df.rename(columns={col: "close_price"}, inplace=True)
        elif "volume" in lower_col:
            df.rename(columns={col: "volume"}, inplace=True)
        elif "market" in lower_col and "cap" in lower_col:
            df.rename(columns={col: "market_cap"}, inplace=True)

    if "ds" not in df.columns:
        st.error("No se logró establecer la columna 'ds' tras el rename.")
        st.write(df.head())
        return None

    # Convertir ds a datetime
    df["ds"] = pd.to_datetime(df["ds"], errors="coerce")
    df.dropna(subset=["ds"], inplace=True)
    df.sort_values(by="ds", ascending=True, inplace=True)
    df.reset_index(drop=True, inplace=True)

    # Si existe 'close_price', filtramos filas <= 0
    if "close_price" in df.columns:
        df = df[df["close_price"] > 0].copy()
    else:
        st.error("No se encontró la columna 'close_price' tras renombrar.")
        st.write("Columnas finales:", df.columns.tolist())
        st.write(df.head())
        return None

    return df

@st.cache_data
def load_coingecko_data(coin_id, vs_currency="usd", days="max"):
    """
    Descarga datos desde CoinGecko.
    """
    url = f"https://api.coingecko.com/api/v3/coins/{coin_id}/market_chart?vs_currency={vs_currency}&days={days}"
    headers = {"User-Agent": "Mozilla/5.0"}
    resp = requests.get(url, headers=headers)
    if resp.status_code != 200:
        st.error(f"Error CoinGecko (status code {resp.status_code}).")
        return None
    data = resp.json()
    if "prices" not in data:
        st.error("CoinGecko: Datos no disponibles (falta 'prices').")
        return None

    df = pd.DataFrame(data["prices"], columns=["timestamp", "close_price"])
    df["ds"] = pd.to_datetime(df["timestamp"], unit="ms")
    df = df[["ds", "close_price"]]
    df.sort_values(by="ds", ascending=True, inplace=True)
    df.reset_index(drop=True, inplace=True)
    df = df[df["close_price"] > 0].copy()
    return df

def load_combined_data(symbol_alpha, symbol_cg):
    """
    Combina datos de Alpha Vantage y CoinGecko (outer join por 'ds').
    Promedia los precios si ambos están disponibles, rellena huecos con ffill.
    """
    df_alpha = load_alpha_data(symbol_alpha)
    df_cg = load_coingecko_data(symbol_cg)

    if df_alpha is None and df_cg is None:
        st.error("No se pudieron descargar datos de ninguna fuente.")
        return None
    if df_alpha is None:
        return df_cg
    if df_cg is None:
        return df_alpha

    df_merge = pd.merge(df_alpha, df_cg, on="ds", how="outer", suffixes=("_alpha", "_cg"))
    df_merge.sort_values(by="ds", inplace=True)

    # Si no existen las columnas esperadas, se notifica
    close_alpha = "close_price_alpha"
    close_cg = "close_price_cg"
    if close_alpha not in df_merge.columns and close_cg not in df_merge.columns:
        st.error("No se encontraron columnas 'close_price_alpha' ni 'close_price_cg' tras el merge.")
        st.write(df_merge.head())
        return None

    df_merge["close_price"] = df_merge[[close_alpha, close_cg]].mean(axis=1, skipna=True)
    df_result = df_merge[["ds", "close_price"]].copy()
    df_result.ffill(inplace=True)
    df_result.dropna(subset=["close_price"], inplace=True)
    return df_result

def add_indicators(df):
    """
    Calcula RSI, MACD y Bollinger Bands con pandas_ta.
    """
    df["rsi"] = ta.rsi(df["close_price"], length=14)
    macd_df = ta.macd(df["close_price"])
    bbands_df = ta.bbands(df["close_price"], length=20, std=2)
    df = pd.concat([df, macd_df, bbands_df], axis=1)
    df.ffill(inplace=True)
    return df

def add_all_indicators(df):
    return add_indicators(df)

def create_sequences(data, window_size=30):
    if len(data) <= window_size:
        st.error(f"No hay suficientes datos para una ventana de {window_size} días.")
        return None, None
    X, y = [], []
    for i in range(window_size, len(data)):
        X.append(data[i - window_size : i])
        y.append(data[i, 0])
    return np.array(X), np.array(y)

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

def train_and_predict(symbol_alpha, symbol_cg, horizon_days=30, window_size=30, test_size=0.2,
                      use_multivariate=False, use_indicators=False,
                      epochs=10, batch_size=32, learning_rate=0.001):
    """
    Descarga datos combinados (Alpha + CoinGecko), enriquece con indicadores (opcional),
    entrena un modelo LSTM y realiza la predicción futura de forma iterativa.
    """
    df_prices = load_combined_data(symbol_alpha, symbol_cg)
    if df_prices is None:
        return None

    if use_indicators:
        df_prices = add_all_indicators(df_prices)

    if "close_price" not in df_prices.columns:
        st.error("No se encontró 'close_price' en el DataFrame final.")
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

def main_app():
    st.set_page_config(page_title="Crypto Price Prediction Dashboard", layout="wide")
    st.title("Crypto Price Predictions 🔮")
    st.markdown("**Fuentes de Datos:** Alpha Vantage y CoinGecko")

    st.sidebar.header("Configuración de la predicción")

    # Selección de la cripto
    cryptos_list = list(alpha_symbols.keys())
    crypto_name = st.sidebar.selectbox("Selecciona una criptomoneda:", cryptos_list)
    symbol_alpha = alpha_symbols[crypto_name]   # Símbolo para Alpha Vantage
    symbol_cg = coingecko_ids[crypto_name]      # ID para CoinGecko

    horizon = st.sidebar.slider("Días a predecir:", 1, 60, 30)
    auto_window = min(60, max(5, horizon * 2))
    st.sidebar.markdown(f"**Tamaño de ventana (auto): {auto_window} días**")

    use_indicators = st.sidebar.checkbox("Incluir indicadores técnicos (RSI, MACD, BBANDS)", value=True)

    scenario = st.sidebar.selectbox("Escenario del Modelo:", ["Pesimista", "Neutro", "Optimista"])
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

    # Mostramos gráfico histórico
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

    tabs = st.tabs(["🤖 Entrenamiento y Test", f"🔮 Predicción de Precios - {crypto_name}"])

    with tabs[0]:
        st.header("Entrenamiento del Modelo y Evaluación en Test")
        if st.button("Entrenar Modelo y Predecir"):
            with st.spinner("Entrenando el modelo, por favor espera..."):
                result = train_and_predict(
                    symbol_alpha=symbol_alpha,
                    symbol_cg=symbol_cg,
                    horizon_days=horizon,
                    window_size=auto_window,
                    test_size=0.2,
                    use_multivariate=False,
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
