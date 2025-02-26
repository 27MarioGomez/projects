#########################
# main.py
#########################

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
import os
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.optimizers import Adam



def main_app():
    # ---------------------------------------------------------------------
    # CONFIGURACIÓN DE LA PÁGINA Y ESTILO (UX/UI PROFESIONAL)
    # ---------------------------------------------------------------------
    st.set_page_config(page_title="Crypto Price Prediction Dashboard", layout="wide")
    st.markdown("""
        <style>
        .reportview-container {
            background: #F5F5F5;
        }
        .sidebar .sidebar-content {
            background-image: linear-gradient(#2E7BCF, #2E7BCF);
            color: white;
        }
        .stButton>button {
            background-color: #2E7BCF;
            color: white;
        }
        </style>
        """, unsafe_allow_html=True)

    st.title("Dashboard de Predicción de Precios de Criptomonedas")
    st.markdown("### Interfaz limpia, profesional y funcional para análisis y predicción usando LSTM")
    st.markdown("Utiliza la barra lateral para configurar la criptomoneda, parámetros del modelo y otras opciones.")

    # ---------------------------------------------------------------------
    # BARRA LATERAL: CONFIGURACIÓN DEL PROYECTO Y PARÁMETROS AVANZADOS
    # ---------------------------------------------------------------------
    st.sidebar.header("Configuración del Proyecto")

    # Diccionario con rutas de datos
    crypto_paths = {
        "Bitcoin":  r"C:\Users\mario\Documents\GitHub\projects\CRYPTO PREDICTION\data\coin_Bitcoin.csv",
        "Ethereum": r"C:\Users\mario\Documents\GitHub\projects\CRYPTO PREDICTION\data\coin_Ethereum.csv",
        "XRP":      r"C:\Users\mario\Documents\GitHub\projects\CRYPTO PREDICTION\data\coin_XRP.csv",
        "Stellar":  r"C:\Users\mario\Documents\GitHub\projects\CRYPTO PREDICTION\data\coin_Stellar.csv",
        "Solana":   r"C:\Users\mario\Documents\GitHub\projects\CRYPTO PREDICTION\data\coin_Solana.csv",
        "Cardano":  r"C:\Users\mario\Documents\GitHub\projects\CRYPTO PREDICTION\data\coin_Cardano.csv"
    }

    crypto_choice = st.sidebar.selectbox("Selecciona la criptomoneda:", list(crypto_paths.keys()))
    data_path = crypto_paths[crypto_choice]

    st.sidebar.subheader("Parámetros de Predicción")
    horizon = st.sidebar.slider("Días a predecir:", min_value=1, max_value=60, value=30)
    window_size = st.sidebar.slider("Tamaño de ventana (días):", min_value=10, max_value=120, value=60)
    use_multivariate = st.sidebar.checkbox("Usar datos multivariados (Open, High, Low, Volume)", value=False)

    st.sidebar.subheader("Ajustes Avanzados del Modelo")
    epochs = st.sidebar.number_input("Número de épocas:", min_value=5, max_value=100, value=10, step=1)
    batch_size = st.sidebar.number_input("Batch size:", min_value=16, max_value=256, value=32, step=16)
    learning_rate = st.sidebar.number_input("Learning rate:", min_value=0.0001, max_value=0.01, value=0.001, step=0.0001, format="%.4f")
    show_raw_data = st.sidebar.checkbox("Mostrar datos históricos", value=True)

    # Consejo: Para mejorar aún más el modelo, podrías incluir indicadores técnicos (RSI, MACD, etc.) con la librería "ta"

    # ---------------------------------------------------------------------
    # 1. FUNCIÓN PARA CARGAR Y LIMPIAR LOS DATOS
    # ---------------------------------------------------------------------
    def load_and_clean_data(csv_path):
        df = pd.read_csv(csv_path)
        # Renombramos columnas relevantes: "Date" -> "ds", "Close" -> "close_price"
        df.rename(columns={"Date": "ds", "Close": "close_price"}, inplace=True)
        # Convertir a datetime; si hay valores inválidos, se convierten a NaT
        df['ds'] = pd.to_datetime(df['ds'], errors='coerce')
        # Truncamos a segundos para evitar problemas de serialización con PyArrow
        df['ds'] = df['ds'].dt.floor('s')
        # Eliminamos filas donde la conversión falló
        df.dropna(subset=['ds'], inplace=True)
        df.sort_values(by='ds', inplace=True)
        df.reset_index(drop=True, inplace=True)
        return df

    # ---------------------------------------------------------------------
    # 2. FUNCIÓN PARA CREAR SECUENCIAS (VENTANAS) PARA LA LSTM
    # ---------------------------------------------------------------------
    def create_sequences(data, window_size=60):
        X, y = [], []
        for i in range(window_size, len(data)):
            X.append(data[i-window_size:i])
            y.append(data[i, 0])  # Se asume que la primera columna es close_price
        X, y = np.array(X), np.array(y)
        return X, y

    # ---------------------------------------------------------------------
    # 3. FUNCIÓN PARA ENTRENAR EL MODELO LSTM Y REALIZAR PREDICCIONES
    # ---------------------------------------------------------------------
    def train_and_predict_lstm(csv_path, horizon_days=30, window_size=60, test_size=0.2,
                               use_multivariate=False, epochs=10, batch_size=32, learning_rate=0.001):
        df = load_and_clean_data(csv_path)

        if use_multivariate:
            df_model = df[['ds', 'close_price', 'Open', 'High', 'Low', 'Volume']].copy()
            features_cols = ["close_price", "Open", "High", "Low", "Volume"]
            data_for_model = df_model[features_cols].values
            scaler_features = MinMaxScaler(feature_range=(0, 1))
            scaled_data = scaler_features.fit_transform(data_for_model)
            scaler_target = MinMaxScaler(feature_range=(0, 1))
            scaler_target.fit(df_model[["close_price"]])
        else:
            df_model = df[['ds', 'close_price']].copy()
            data_for_model = df_model[['close_price']].values
            scaler_target = MinMaxScaler(feature_range=(0, 1))
            scaled_data = scaler_target.fit_transform(data_for_model)

        # Dividir en train y test
        split_index = int(len(scaled_data) * (1 - test_size))
        train_data = scaled_data[:split_index]
        test_data = scaled_data[split_index:]

        X_train, y_train = create_sequences(train_data, window_size=window_size)
        X_test, y_test = create_sequences(test_data, window_size=window_size)

        # Dividir el conjunto de entrenamiento en train/validación (90%/10%)
        val_split = int(len(X_train) * 0.9)
        X_val, y_val = X_train[val_split:], y_train[val_split:]
        X_train, y_train = X_train[:val_split], y_train[:val_split]

        # Construir el modelo LSTM
        model = Sequential()
        model.add(LSTM(64, return_sequences=True, input_shape=(X_train.shape[1], X_train.shape[2])))
        model.add(Dropout(0.2))
        model.add(LSTM(64, return_sequences=False))
        model.add(Dropout(0.2))
        model.add(Dense(1))

        optimizer = Adam(learning_rate=learning_rate)
        model.compile(optimizer=optimizer, loss='mean_squared_error')

        model.fit(X_train, y_train, validation_data=(X_val, y_val),
                  epochs=epochs, batch_size=batch_size, verbose=1)

        # Predicciones en el set de test
        test_predictions = model.predict(X_test)
        test_predictions_descaled = scaler_target.inverse_transform(test_predictions)
        y_test_descaled = scaler_target.inverse_transform(y_test.reshape(-1, 1))

        # Métricas
        rmse = np.sqrt(np.mean((y_test_descaled - test_predictions_descaled)**2))
        mape = np.mean(np.abs((y_test_descaled - test_predictions_descaled) / y_test_descaled)) * 100

        # Predicción futura (iterativa)
        last_window = scaled_data[-window_size:]
        future_preds_scaled = []
        current_input = last_window.reshape(1, window_size, X_train.shape[2])

        for _ in range(horizon_days):
            future_pred = model.predict(current_input)[0][0]
            future_preds_scaled.append(future_pred)
            new_feature = np.zeros((1, 1, X_train.shape[2]))
            new_feature[0, 0, 0] = future_pred  # Se actualiza solo el target
            current_input = np.append(current_input[:, 1:, :], new_feature, axis=1)

        future_preds = scaler_target.inverse_transform(np.array(future_preds_scaled).reshape(-1, 1)).flatten()

        return df_model, test_predictions_descaled, y_test_descaled, future_preds, rmse, mape

    # ---------------------------------------------------------------------
    # MOSTRAR DATOS HISTÓRICOS (SI SE HA SELECCIONADO)
    # ---------------------------------------------------------------------
    df = load_and_clean_data(data_path)
    if show_raw_data:
        st.subheader("Datos Históricos")
        # Convertir la columna 'ds' a string para evitar problemas de serialización en st.dataframe()
        df_show = df.copy()
        df_show['ds'] = df_show['ds'].astype(str)
        st.dataframe(df_show.head(100))
        fig_hist = px.line(
            df, x="ds", y="close_price",
            title=f"Histórico de Precio de {crypto_choice}",
            labels={"ds": "Fecha", "close_price": "Precio de Cierre"}
        )
        st.plotly_chart(fig_hist, use_container_width=True)

    # ---------------------------------------------------------------------
    # USO DE PESTAÑAS PARA ORGANIZAR LA INFORMACIÓN Y MEJORAR LA UX
    # ---------------------------------------------------------------------
    tabs = st.tabs(["📊 Overview", "🤖 Entrenamiento y Test", "🔮 Predicción Futura"])

    with tabs[0]:
        st.header("Overview de Datos")
        st.markdown("Visualización general de los datos históricos y estadísticas descriptivas.")
        fig_overview = px.line(
            df, x="ds", y="close_price",
            title=f"Histórico de {crypto_choice}",
            labels={"ds": "Fecha", "close_price": "Precio de Cierre"}
        )
        st.plotly_chart(fig_overview, use_container_width=True)
        st.markdown("**Estadísticas Descriptivas:**")
        st.write(df.describe())

    with tabs[1]:
        st.header("Entrenamiento del Modelo y Evaluación en Test")
        if st.button("Entrenar Modelo y Predecir", key="train_test"):
            with st.spinner("Entrenando el modelo, por favor espera..."):
                df_model, test_preds, y_test_real, future_preds, rmse, mape = train_and_predict_lstm(
                    csv_path=data_path,
                    horizon_days=horizon,
                    window_size=window_size,
                    test_size=0.2,
                    use_multivariate=use_multivariate,
                    epochs=epochs,
                    batch_size=batch_size,
                    learning_rate=learning_rate
                )
            st.success("Entrenamiento y predicción completados!")
            
            # Mostrar métricas en columnas
            col1, col2 = st.columns(2)
            col1.metric("RMSE (Test)", f"{rmse:.2f}")
            col2.metric("MAPE (Test)", f"{mape:.2f}%")
            
            st.subheader("Comparación en el Set de Test")
            test_dates = df_model['ds'].iloc[-len(y_test_real):]
            fig_test = go.Figure()
            fig_test.add_trace(go.Scatter(x=test_dates, y=y_test_real.flatten(), mode='lines', name='Precio Real (Test)'))
            fig_test.add_trace(go.Scatter(x=test_dates, y=test_preds.flatten(), mode='lines', name='Predicción (Test)'))
            fig_test.update_layout(
                title=f"Comparación en Test: {crypto_choice}",
                xaxis_title="Fecha",
                yaxis_title="Precio"
            )
            st.plotly_chart(fig_test, use_container_width=True)
            
    with tabs[2]:
        st.header("Predicción a Futuro")
        # Para que esta pestaña funcione, se debe haber entrenado el modelo previamente y tener 'future_preds' en memoria
        if 'future_preds' in locals():
            last_date = df_model['ds'].iloc[-1]
            future_dates = pd.date_range(start=last_date + pd.Timedelta(days=1), periods=horizon)
            fig_future = go.Figure()
            fig_future.add_trace(go.Scatter(x=future_dates, y=future_preds, mode='lines+markers', name='Predicción Futura'))
            fig_future.update_layout(
                title=f"Predicción a Futuro ({horizon} días) - {crypto_choice}",
                xaxis_title="Fecha",
                yaxis_title="Precio"
            )
            st.plotly_chart(fig_future, use_container_width=True)
            
            st.subheader("Valores Numéricos de la Predicción Futura")
            future_df = pd.DataFrame({'Fecha': future_dates, 'Predicción': future_preds})
            st.dataframe(future_df)
        else:
            st.info("Primero entrena el modelo en la pestaña 'Entrenamiento y Test' para generar las predicciones futuras.")


# ---------------------------------------------------------------------
#  EJECUCIÓN CON PYTHON vs STREAMLIT
# ---------------------------------------------------------------------
if __name__ == "__main__":
    st.set_page_config(page_title="Crypto Price Prediction Dashboard", layout="wide")
    st.title("Dashboard de Predicción de Precios de Criptomonedas")
    st.write("Cargando datos y modelo...")

    # Verifica si Streamlit está ejecutando correctamente
    st.success("✅ Streamlit está funcionando correctamente")

