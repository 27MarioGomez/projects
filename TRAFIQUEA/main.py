# main.py
"""
Trafiquea: Dashboard interactivo para la optimización de rutas y predicción de demanda de transporte.

Este dashboard utiliza servicios gratuitos (Nominatim, OSRM, Open-Meteo) y un modelo LSTM para:
    - Geolocalización y optimización de rutas con mapas interactivos.
    - Consulta de clima en tiempo real.
    - Predicción de demanda con un modelo optimizado.
    - Visualización clara de métricas y análisis de impacto ambiental.

Se prioriza un UX/UI de alto valor, mostrando mapas, rutas y predicciones sin comprometer el rendimiento.
"""

# =============================================================================
# Importación de librerías
# =============================================================================
from datetime import datetime, timedelta

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import requests

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.regularizers import l2

from sklearn.preprocessing import MinMaxScaler
from sklearn.impute import SimpleImputer
from sklearn.base import BaseEstimator, TransformerMixin

from numba import njit
from xgboost import XGBRegressor
from geopy.geocoders import Nominatim

# =============================================================================
# Configuración inicial de la página
# =============================================================================
st.set_page_config(
    page_title="Trafiquea: Optimización y Predicción de Rutas",
    layout="wide",
    initial_sidebar_state="expanded"
)

# =============================================================================
# Funciones de Utilidad y Acceso a APIs Públicas
# =============================================================================

@st.cache_data(ttl=3600)
def geocode_address(address: str):
    """
    Geocodifica una dirección usando Nominatim (OpenStreetMap).
    Retorna una tupla (latitud, longitud) o None si no se encuentra.
    """
    geolocator = Nominatim(user_agent="trafiquea_dashboard")
    location = geolocator.geocode(address)
    if location:
        return (location.latitude, location.longitude)
    return None

def get_route_osrm(origin_coords, destination_coords):
    """
    Obtiene la ruta optimizada entre dos puntos usando la API pública de OSRM.
    Retorna la geometría de la ruta en formato GeoJSON o None en caso de error.
    """
    base_url = "http://router.project-osrm.org/route/v1/driving"
    # OSRM requiere el formato: lon,lat;lon,lat
    coords = f"{origin_coords[1]},{origin_coords[0]};{destination_coords[1]},{destination_coords[0]}"
    params = {"overview": "full", "geometries": "geojson"}
    url = f"{base_url}/{coords}"
    response = requests.get(url, params=params)
    if response.status_code == 200:
        data = response.json()
        if "routes" in data and len(data["routes"]) > 0:
            return data["routes"][0]["geometry"]
    return None

def get_weather_open_meteo(lat: float, lon: float):
    """
    Consulta la API de Open-Meteo para obtener el clima actual basado en latitud y longitud.
    No requiere API key.
    Retorna un diccionario con temperatura y velocidad del viento.
    """
    url = "https://api.open-meteo.com/v1/forecast"
    params = {
        "latitude": lat,
        "longitude": lon,
        "current_weather": True,
        "timezone": "auto"
    }
    response = requests.get(url, params=params)
    if response.status_code == 200:
        data = response.json()
        current = data.get("current_weather", {})
        return {
            "temperature": current.get("temperature"),
            "windspeed": current.get("windspeed")
        }
    return None

@st.cache_data(ttl=3600)
def get_transit_demand_data():
    """
    Carga datos históricos de demanda de transporte desde una URL pública.
    Para este ejemplo se utiliza un CSV de muestra alojado en GitHub.
    Se espera que el CSV contenga columnas 'Fecha' y 'Demanda'.
    """
    url = "https://raw.githubusercontent.com/plotly/datasets/master/2014_apple_stock.csv"
    try:
        # Se simula el dato de 'Demanda' utilizando la columna 'AAPL' y se renombra la fecha
        df = pd.read_csv(url, parse_dates=["AAPL_x"])
        df.rename(columns={"AAPL_x": "Fecha", "AAPL_y": "Demanda"}, inplace=True)
        # Si no existe la columna 'Demanda', se crea con valores aleatorios
        if "Demanda" not in df.columns:
            df["Demanda"] = np.random.randint(100, 500, len(df))
        return df
    except Exception as e:
        st.error(f"Error al cargar datos de demanda: {e}")
        return pd.DataFrame()

# =============================================================================
# Preparación de Datos y Modelado LSTM para Demanda
# =============================================================================
@njit
def create_sequences_numba(data, window_size):
    """
    Crea secuencias para entrenamiento del modelo LSTM usando Numba para aceleración.
    """
    n = data.shape[0]
    num_features = data.shape[1]
    m = n - window_size
    X = np.empty((m, window_size, num_features), dtype=data.dtype)
    y = np.empty(m, dtype=data.dtype)
    for i in range(m):
        X[i] = data[i:i+window_size]
        y[i] = data[i+window_size, 0]
    return X, y

def create_sequences(data: np.ndarray, window_size: int):
    """
    Interfaz para crear secuencias; retorna X e y o (None, None) si no hay suficientes datos.
    """
    if data.shape[0] <= window_size:
        return None, None
    return create_sequences_numba(data, window_size)

def build_demand_lstm_model(input_shape):
    """
    Construye y compila un modelo LSTM para predecir la demanda.
    """
    model = Sequential([
        LSTM(64, return_sequences=True, input_shape=input_shape, kernel_regularizer=l2(0.001)),
        Dropout(0.2),
        LSTM(32, kernel_regularizer=l2(0.001)),
        Dropout(0.2),
        Dense(50, activation="relu", kernel_regularizer=l2(0.001)),
        Dense(1)
    ])
    model.compile(optimizer=Adam(0.001), loss="mse")
    return model

def train_demand_model(df_demand: pd.DataFrame, window_size=7, epochs=10, batch_size=16):
    """
    Entrena el modelo LSTM con datos históricos de demanda.
    Aplica log-transformación a 'Demanda' y escala los datos con MinMaxScaler.
    Retorna el modelo entrenado y el scaler.
    """
    df_demand = df_demand.sort_values("Fecha")
    df_demand["log_demand"] = np.log1p(df_demand["Demanda"])
    scaler = MinMaxScaler()
    demand_scaled = scaler.fit_transform(df_demand[["log_demand"]])
    X, y = create_sequences(demand_scaled, window_size)
    if X is None:
        st.error("No hay suficientes datos para crear secuencias de demanda.")
        return None, scaler
    model = build_demand_lstm_model((window_size, 1))
    es = EarlyStopping(monitor="loss", patience=3, restore_best_weights=True)
    model.fit(X, y, epochs=epochs, batch_size=batch_size, verbose=0, callbacks=[es])
    return model, scaler

def predict_future_demand(model, scaler, df_demand: pd.DataFrame, window_size: int, forecast_days: int):
    """
    Predice la demanda futura para un número de días usando el modelo LSTM.
    Retorna una lista de fechas futuras y sus predicciones.
    """
    demand_scaled = scaler.transform(df_demand[["log_demand"]])
    X, _ = create_sequences(demand_scaled, window_size)
    if X is None:
        return None, None
    current_input = X[-1:].copy()
    preds = []
    for _ in range(forecast_days):
        pred_scaled = model.predict(current_input, verbose=0)[0][0]
        inv = scaler.inverse_transform(np.array([[pred_scaled]]))
        pred = np.expm1(inv[0, 0])
        preds.append(pred)
        new_seq = np.array([[pred_scaled]])
        current_input = np.concatenate([current_input[:, 1:, :], new_seq.reshape(1, 1, 1)], axis=1)
    future_dates = pd.date_range(start=df_demand["Fecha"].max() + timedelta(days=1), periods=forecast_days).tolist()
    return future_dates, preds

# =============================================================================
# Diseño y Lógica Principal del Dashboard
# =============================================================================
def main_app():
    st.title("Trafiquea: Optimización y Predicción de Rutas")
    st.markdown("""
    **Descripción del Proyecto:**  
    Dashboard para optimizar rutas y predecir la demanda de transporte en tiempo real, con mapas interactivos y análisis de impacto ambiental.
    """)
    
    # -------------------------------------------------------------------------
    # Sección: Optimización de Rutas y Visualización en Mapa
    # -------------------------------------------------------------------------
    st.sidebar.title("Configuración")
    st.sidebar.subheader("Optimización de Rutas")
    origin = st.sidebar.text_input("Dirección de Origen", "Plaza Mayor, Madrid")
    destination = st.sidebar.text_input("Dirección de Destino", "Puerta del Sol, Madrid")
    if st.sidebar.button("Obtener Ruta"):
        origin_coords = geocode_address(origin)
        destination_coords = geocode_address(destination)
        if origin_coords and destination_coords:
            route_geo = get_route_osrm(origin_coords, destination_coords)
            weather = get_weather_open_meteo(origin_coords[0], origin_coords[1])
            st.success("Ruta y datos actualizados.")
            st.write(f"**Clima en Origen:** {weather['temperature']}°C, Viento: {weather['windspeed']} km/h")
            
            # Mapa interactivo con la ruta y marcadores para origen y destino
            fig_map = go.Figure()
            # Dibujar la ruta si se obtuvo la geometría
            if route_geo:
                fig_map.add_trace(go.Scattermapbox(
                    mode="lines",
                    lon=route_geo["coordinates"][:, 0] if isinstance(route_geo["coordinates"], np.ndarray) else [pt[0] for pt in route_geo["coordinates"]],
                    lat=route_geo["coordinates"][:, 1] if isinstance(route_geo["coordinates"], np.ndarray) else [pt[1] for pt in route_geo["coordinates"]],
                    marker={"size": 5},
                    line=dict(width=4, color="blue"),
                    name="Ruta Óptima"
                ))
            # Agregar marcadores de origen y destino
            fig_map.add_trace(go.Scattermapbox(
                mode="markers",
                lon=[origin_coords[1]],
                lat=[origin_coords[0]],
                marker={"size": 12, "color": "green"},
                name="Origen"
            ))
            fig_map.add_trace(go.Scattermapbox(
                mode="markers",
                lon=[destination_coords[1]],
                lat=[destination_coords[0]],
                marker={"size": 12, "color": "red"},
                name="Destino"
            ))
            fig_map.update_layout(
                mapbox_style="open-street-map",
                mapbox_zoom=12,
                mapbox_center={"lat": (origin_coords[0]+destination_coords[0])/2,
                               "lon": (origin_coords[1]+destination_coords[1])/2},
                margin={"r":0,"t":0,"l":0,"b":0}
            )
            st.plotly_chart(fig_map, use_container_width=True)
        else:
            st.error("No se pudieron geolocalizar las direcciones.")
    
    # -------------------------------------------------------------------------
    # Sección: Datos de Demanda y Predicción
    # -------------------------------------------------------------------------
    st.sidebar.markdown("---")
    st.sidebar.subheader("Predicción de Demanda")
    forecast_days = st.sidebar.slider("Días a predecir:", min_value=1, max_value=30, value=5,
                                        help="Número de días futuros a predecir.")
    
    df_demand = get_transit_demand_data()
    if df_demand.empty:
        st.error("No se encontraron datos reales de demanda.")
    else:
        st.subheader("Datos Históricos de Demanda de Transporte")
        st.line_chart(df_demand.set_index("Fecha")["Demanda"])
    
    tabs = st.tabs(["Predicción de Demanda", "Dashboard de Métricas", "Análisis de Impacto"])
    
    # Pestaña 1: Predicción de Demanda
    with tabs[0]:
        st.header("Predicción de Demanda de Transporte")
        if not df_demand.empty:
            model_demand, scaler_demand = train_demand_model(df_demand, window_size=7, epochs=10, batch_size=16)
            if model_demand:
                future_dates, demand_preds = predict_future_demand(model_demand, scaler_demand, df_demand, window_size=7, forecast_days=forecast_days)
                if future_dates and demand_preds:
                    df_future = pd.DataFrame({"Fecha": future_dates, "Demanda Predicha": demand_preds})
                    st.subheader("Demanda Futura Estimada")
                    # Se muestra la predicción en un gráfico interactivo
                    fig_pred = px.line(df_future, x="Fecha", y="Demanda Predicha", title="Predicción de Demanda")
                    st.plotly_chart(fig_pred, use_container_width=True)
                    st.download_button(
                        label="Descargar Predicción en CSV",
                        data=df_future.to_csv(index=False).encode("utf-8"),
                        file_name="prediccion_demanda.csv",
                        mime="text/csv"
                    )
    
    # Pestaña 2: Dashboard de Métricas
    with tabs[1]:
        st.header("Métricas de Movilidad en Tiempo Real")
        # Se consulta el clima en Madrid como referencia
        weather = get_weather_open_meteo(40.4168, -3.7038)
        st.metric("Temperatura Actual (°C)", f"{weather['temperature']}°C")
        st.metric("Velocidad del Viento (km/h)", f"{weather['windspeed']} km/h")
        st.metric("Nivel de Congestión", "No disponible")
    
    # Pestaña 3: Análisis de Impacto y Sostenibilidad
    with tabs[2]:
        st.header("Impacto Ambiental y Sostenibilidad")
        st.markdown("""
        Se presentan análisis basados en la optimización de rutas, los ahorros en combustible y la reducción de emisiones de CO₂.
        Estos cálculos se pueden ampliar con datos reales y fórmulas específicas a medida que se disponga de más información.
        """)
        ahorro_combustible = 5.2  # Ejemplo: litros ahorrados
        reduccion_CO2 = 12.3     # Ejemplo: kg de CO₂ reducidos
        st.metric("Ahorro en Combustible (litros)", f"{ahorro_combustible}")
        st.metric("Reducción de Emisiones (kg CO₂)", f"{reduccion_CO2}")

# =============================================================================
# Ejecución de la aplicación
# =============================================================================
if __name__ == "__main__":
    main_app()
