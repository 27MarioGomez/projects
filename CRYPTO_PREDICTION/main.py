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
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, Bidirectional, LSTM, Dense, Dropout, BatchNormalization
from tensorflow.keras.optimizers import Adam
import time
import ssl
import certifi  # Añadido para manejar certificados SSL

##############################################
# Funciones de apoyo
##############################################

def robust_mape(y_true, y_pred, eps=1e-9):
    """
    Calcula el MAPE evitando divisiones por cero.
    """
    return np.mean(np.abs((y_true - y_pred) / np.maximum(np.abs(y_true), eps))) * 100

def calculate_technical_indicators(df):
    """
    Calcula indicadores técnicos como medias móviles y RSI usando solo close_price.
    """
    df = df.copy()
    # Media móvil simple de 7 días
    df['MA7'] = df['close_price'].rolling(window=7, min_periods=1).mean()
    # Media móvil simple de 14 días
    df['MA14'] = df['close_price'].rolling(window=14, min_periods=1).mean()
    
    # RSI básico (simplificado, usando solo close_price)
    delta = df['close_price'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    df['RSI'] = 100 - (100 / (1 + rs))
    
    return df[['close_price', 'MA7', 'MA14', 'RSI']].fillna(0)

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

##############################################
# Descarga de datos desde CoinCap (intervalo diario)
##############################################
@st.cache_data
def load_coincap_data(coin_id, start_ms=None, end_ms=None, max_retries=3):
    """
    Descarga datos de CoinCap con intervalo diario.
    Si se definen start_ms y end_ms se descarga ese rango; de lo contrario se descarga todo el histórico.
    Retorna un DataFrame con las columnas 'ds' y 'close_price'.
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
            if "time" not in df.columns or "priceUsd" not in df.columns:
                st.warning("CoinCap: Faltan las columnas 'time' o 'priceUsd'.")
                return None
            df["ds"] = pd.to_datetime(df["time"], unit="ms")
            df["close_price"] = pd.to_numeric(df["priceUsd"], errors="coerce")
            # Excluimos explícitamente el volumen para evitar problemas con RMSE y MAPE
            df = df[["ds", "close_price"]].dropna(subset=["ds", "close_price"])
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
    st.info("CoinCap: Máx reintentos sin éxito.")
    return None

##############################################
# Creación de secuencias para LSTM
##############################################
def create_sequences(data, window_size=30):
    """
    Crea secuencias de tamaño 'window_size' a partir de 'data'.
    Se asume que la primera columna es el target ('close_price').
    'data' es un numpy.ndarray con shape (n_samples, n_features).
    """
    if len(data) <= window_size:
        st.warning(f"No hay datos suficientes para una ventana de {window_size} días.")
        return None, None
    X, y = [], []
    for i in range(window_size, len(data)):
        X.append(data[i - window_size : i])  # Usamos indexación directa en el array de NumPy
        y.append(data[i, 0])  # Usamos 'close_price' como target (primera columna)
    return np.array(X), np.array(y)

##############################################
# Modelo LSTM mejorado: Conv1D + Bidirectional LSTM con regularización
##############################################
def build_improved_lstm_model(input_shape, learning_rate=0.001):
    """
    Construye un modelo mejorado con más regularización y características.
    """
    model = Sequential()
    model.add(Conv1D(filters=64, kernel_size=3, activation="relu", input_shape=input_shape, padding="same"))
    model.add(BatchNormalization())  # Añadir BatchNormalization para estabilizar
    model.add(Bidirectional(LSTM(128, return_sequences=True, kernel_regularizer=tf.keras.regularizers.l2(0.01))))
    model.add(Dropout(0.4))  # Aumentar dropout para prevenir overfitting
    model.add(Bidirectional(LSTM(128, return_sequences=True, kernel_regularizer=tf.keras.regularizers.l2(0.01))))
    model.add(Dropout(0.4))
    model.add(Bidirectional(LSTM(64, return_sequences=False, kernel_regularizer=tf.keras.regularizers.l2(0.01))))
    model.add(Dropout(0.4))
    model.add(Dense(1))
    opt = Adam(learning_rate=learning_rate)
    model.compile(optimizer=opt, loss="mean_squared_error")
    return model

##############################################
# Función aislada para entrenar el modelo mejorado
##############################################
def train_model(X_train, y_train, X_val, y_val, input_shape, epochs, batch_size, learning_rate):
    """
    Entrena el modelo LSTM mejorado de forma aislada para evitar conflictos con el contexto global.
    """
    # Reiniciar completamente el grafo de TensorFlow para evitar conflictos
    tf.keras.backend.clear_session()
    
    # Crear y entrenar el modelo directamente, confiando en TensorFlow predeterminado
    model = build_improved_lstm_model(input_shape, learning_rate=learning_rate)
    model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=epochs,
        batch_size=batch_size,
        verbose=1
    )
    return model

##############################################
# Entrenamiento y predicción con LSTM mejorado
##############################################
def train_and_predict(
    coin_id,
    use_custom_range,
    start_ms,
    end_ms,
    horizon_days=30,
    window_size=60,  # Aumentamos la ventana para capturar más patrones
    test_size=0.2,
    epochs=50,  # Más epochs para un mejor ajuste
    batch_size=32,
    learning_rate=0.0005  # Ajuste del learning rate para estabilidad
):
    """
    Descarga datos de CoinCap, entrena un modelo LSTM mejorado y realiza predicciones.
    """
    temp_df = load_coincap_data(coin_id, start_ms, end_ms)
    if temp_df is None or temp_df.empty:
        st.warning("No se pudieron descargar datos suficientes. Reajusta el rango de fechas.")
        return None
    df = temp_df.copy()

    if "close_price" not in df.columns:
        st.warning("No se encontró 'close_price' en los datos.")
        return None

    # Añadir indicadores técnicos (sin volumen)
    df_features = calculate_technical_indicators(df)
    data_for_model = df_features.values

    scaler_target = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler_target.fit_transform(data_for_model)

    split_index = int(len(scaled_data) * (1 - test_size))
    if split_index <= window_size:
        st.warning("Datos insuficientes para entrenar. Reajusta parámetros.")
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

    # Entrenar el modelo mejorado en una función aislada
    input_shape = (X_train.shape[1], X_train.shape[2])
    lstm_model = train_model(X_train, y_train, X_val, y_val, input_shape, epochs, batch_size, learning_rate)

    # Predicciones del modelo LSTM
    test_preds_scaled = lstm_model.predict(X_test)
    # Desescalamos solo la columna 'close_price' (columna 0)
    test_preds = scaler_target.inverse_transform(np.hstack([test_preds_scaled, np.zeros((test_preds_scaled.shape[0], test_data.shape[1]-1))]))[:, 0]
    y_test_deserialized = scaler_target.inverse_transform(np.hstack([y_test.reshape(-1, 1), np.zeros((y_test.shape[0], test_data.shape[1]-1))]))[:, 0]

    valid_mask = ~np.isnan(test_preds) & ~np.isnan(y_test_deserialized)
    if np.sum(valid_mask) == 0:
        rmse, mape = np.nan, np.nan
    else:
        rmse = np.sqrt(np.mean((y_test_deserialized[valid_mask] - test_preds[valid_mask]) ** 2))
        mape = robust_mape(y_test_deserialized[valid_mask], test_preds[valid_mask])

    # Predicción futura iterativa con LSTM
    last_window = scaled_data[-window_size:]
    future_preds_scaled = []
    current_input = last_window.reshape(1, window_size, X_train.shape[2])
    for _ in range(horizon_days):
        future_pred = lstm_model.predict(current_input)[0][0]
        future_preds_scaled.append(future_pred)
        new_feature = np.copy(current_input[:, -1:, :])
        new_feature[0, 0, 0] = future_pred  # Actualizar solo 'close_price'
        # Mantener las otras características constantes (MA7, MA14, RSI)
        for c in range(1, X_train.shape[2]):
            new_feature[0, 0, c] = current_input[0, -1, c]
        current_input = np.append(current_input[:, 1:, :], new_feature, axis=1)
    future_preds = scaler_target.inverse_transform(np.hstack([np.array(future_preds_scaled).reshape(-1, 1), np.zeros((horizon_days, scaled_data.shape[1]-1))]))[:, 0]

    # Suavizar predicciones futuras con media móvil (suavizado exponencial)
    future_preds_smoothed = pd.Series(future_preds).ewm(span=5).mean().values

    return df, test_preds, y_test_deserialized, future_preds_smoothed, rmse, mape

##############################################
# Análisis de sentimiento en Twitter (ahora en X)
##############################################
def analyze_twitter_sentiment(crypto_name, max_tweets=50):
    """
    Extrae hasta max_tweets tweets (ahora desde X.com) relacionados con la criptomoneda
    (se usa la primera palabra) y calcula el sentimiento promedio usando VaderSentiment.
    """
    try:
        import snscrape.modules.twitter as sntwitter
        from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
        import certifi  # Importar certifi para usar certificados
        sntwitter.TWITTER_BASE_URL = "https://x.com"

        # Intentar con verificación SSL primero (usando certificados de certifi)
        try:
            # Configurar el contexto SSL con certificados de certifi
            ssl_context = ssl.create_default_context(cafile=certifi.where())
        except Exception as e:
            st.warning(f"No se pudieron cargar certificados de certifi, intentando sin verificación: {e}")
            # Workaround temporal: deshabilitar verificación SSL (no recomendado en producción)
            ssl_context = ssl.create_default_context()
            ssl_context.check_hostname = False
            ssl_context.verify_mode = ssl.CERT_NONE

    except Exception as e:
        st.error(f"Error importando snscrape, vaderSentiment o certifi: {e}")
        return None, []

    keyword = crypto_name.split(" ")[0]
    tweets = []
    try:
        for i, tweet in enumerate(sntwitter.TwitterSearchScraper(keyword).get_items()):
            if i >= max_tweets:
                break
            tweets.append(tweet.content)
    except Exception as e:
        st.error(f"Error al obtener tweets: {e}")
        return None, []

    if not tweets:
        return None, []
    analyzer = SentimentIntensityAnalyzer()
    scores = [analyzer.polarity_scores(t)['compound'] for t in tweets if t]
    if scores:
        avg_sentiment = np.mean(scores)
        return avg_sentiment, tweets[:5]
    else:
        return None, []

##############################################
# Función principal de la app
##############################################
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
    horizon = st.sidebar.slider("Días a predecir:", 1, 60, 30,
                                help="Número de días a futuro a predecir.")
    auto_window = min(60, max(5, horizon * 2))
    st.sidebar.markdown(f"**Tamaño de ventana (auto): {auto_window} días**")

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

    df_prices = load_coincap_data(coin_id, start_ms, end_ms)
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
        if st.sidebar.checkbox("Ver estadísticas descriptivas", value=False):
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

    tabs = st.tabs(["🤖 Entrenamiento y Test", f"🔮 Predicción de Precios - {crypto_name}", "💬 Sentimiento en X"])

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

    with tabs[2]:
        st.header("Sentimiento en X")
        st.markdown("Analizando tweets recientes sobre la criptomoneda seleccionada...")
        avg_sentiment, sample_tweets = analyze_twitter_sentiment(crypto_name)
        if avg_sentiment is not None:
            st.metric("Sentimiento Promedio", f"{avg_sentiment:.2f}")
            st.write("Ejemplos de tweets analizados:")
            for t in sample_tweets:
                st.write(f"- {t}")
        else:
            st.info("No se encontraron tweets o hubo un error al analizar el sentimiento.")

if __name__ == "__main__":
    main_app()