import os
import streamlit as st
import folium
from streamlit_folium import st_folium
from geopy.geocoders import Nominatim
from datetime import datetime, timedelta
import requests
import numpy as np
import pandas as pd
import joblib
import xmltodict
import plotly.express as px
from prophet import Prophet
import lightgbm as lgb

# --------------------------------------------------------------------
# Configuración: carpeta de datos y creación si no existe
# --------------------------------------------------------------------
DATA_DIR = "data"
os.makedirs(DATA_DIR, exist_ok=True)

# --------------------------------------------------------------------
# Geocodificación con Nominatim
# --------------------------------------------------------------------
def geocode_address(address: str):
    geolocator = Nominatim(user_agent="trafiquea_dashboard")
    location = geolocator.geocode(address)
    if location:
        return (location.latitude, location.longitude), location.address
    return None, None

# --------------------------------------------------------------------
# Descarga/carga de archivos CSV en carpeta data
# --------------------------------------------------------------------
def download_file(url, filename, sep=None, encoding="utf-8"):
    filepath = os.path.join(DATA_DIR, filename)
    if not os.path.exists(filepath):
        resp = requests.get(url)
        if resp.status_code == 200:
            with open(filepath, "wb") as f:
                f.write(resp.content)
    try:
        if sep:
            df = pd.read_csv(filepath, sep=sep, encoding=encoding)
        else:
            df = pd.read_csv(filepath, encoding=encoding)
        return df
    except Exception as e:
        st.error(f"Error al cargar {filename}: {e}")
        return None

# --------------------------------------------------------------------
# API de Open-Meteo (sin claves)
# --------------------------------------------------------------------
def get_weather_open_meteo(lat: float, lon: float):
    url = "https://api.open-meteo.com/v1/forecast"
    params = {
        "latitude": lat,
        "longitude": lon,
        "current_weather": True,
        "hourly": "temperature_2m,windspeed_10m,precipitation,cloudcover",
        "timezone": "auto"
    }
    r = requests.get(url, params=params)
    if r.status_code == 200:
        data = r.json()
        current = data.get("current_weather", {})
        return {
            "temperatura": current.get("temperature"),
            "viento": current.get("windspeed"),
            "precipitacion": data.get("hourly", {}).get("precipitation", [0])[0],
            "nubosidad": data.get("hourly", {}).get("cloudcover", [0])[0]
        }
    return None

# --------------------------------------------------------------------
# OSRM: /route/v1/driving y /trip/v1/driving
# --------------------------------------------------------------------
def get_route_osrm(origin_coords, destination_coords):
    base_url = "http://router.project-osrm.org/route/v1/driving"
    # OSRM requiere lon,lat
    coords = f"{origin_coords[1]},{origin_coords[0]};{destination_coords[1]},{destination_coords[0]}"
    params = {"overview": "full", "geometries": "geojson", "steps": "true"}
    url = f"{base_url}/{coords}"
    resp = requests.get(url, params=params)
    if resp.status_code == 200:
        data = resp.json()
        if data.get("routes"):
            route_info = data["routes"][0]
            return {
                "geometry": route_info["geometry"],
                "steps": route_info["legs"][0]["steps"],
                "distance": route_info["distance"],
                "duration": route_info["duration"]
            }
    return None

def get_route_osrm_multiple(coords_list, profile="driving"):
    base_url = f"http://router.project-osrm.org/trip/v1/{profile}"
    coords_str = ";".join([f"{lon},{lat}" for (lat, lon) in coords_list])
    params = {"roundtrip": "false", "source": "first", "destination": "last", "geometries": "geojson"}
    url = f"{base_url}/{coords_str}"
    resp = requests.get(url, params=params)
    if resp.status_code == 200:
        data = resp.json()
        if "trips" in data and data["trips"]:
            return data["trips"][0]
    return None

# --------------------------------------------------------------------
# Eurostat: ejemplo de consulta
# --------------------------------------------------------------------
@st.cache_data(ttl=3600)
def get_eurostat_transport_data():
    # Ajustar según la doc oficial
    dataset_id = "tsdtr020"
    params = {
        "format": "JSON",
        "geo": "ES"
    }
    base_url = f"https://ec.europa.eu/eurostat/api/dissemination/statistics/1.0/data/{dataset_id}"
    resp = requests.get(base_url, params=params)
    if resp.status_code == 200:
        return resp.json()
    return None

# --------------------------------------------------------------------
# UNFCCC_v27.csv (datos de GEI)
# --------------------------------------------------------------------
@st.cache_data(ttl=3600)
def get_unfccc_data():
    """
    Carga UNFCCC_v27.csv (emisiones/absorciones totales GEI) desde la carpeta data.
    Asegúrate de que el archivo existe en data.
    """
    filename = "UNFCCC_v27.csv"
    filepath = os.path.join(DATA_DIR, filename)
    if not os.path.exists(filepath):
        st.error(f"No se encontró {filename} en la carpeta data.")
        return None
    try:
        df = pd.read_csv(filepath)
        return df
    except Exception as e:
        st.error(f"Error al cargar {filename}: {e}")
        return None

# --------------------------------------------------------------------
# Modelo ML: LightGBM + Prophet
# --------------------------------------------------------------------
@st.cache_resource
def load_demand_model():
    """
    Carga/entrena un modelo LightGBM usando UNFCCC_v27.csv.
    """
    model_path = "model_demand_lgbm.pkl"
    try:
        model = joblib.load(model_path)
        return model
    except:
        st.warning("Modelo no encontrado, entrenando uno nuevo con UNFCCC_v27.csv.")
        df = get_unfccc_data()
        if df is None:
            st.error("No se pudo cargar UNFCCC_v27.csv para entrenar el modelo.")
            return None
        
        # Ajustar columnas para el entrenamiento. Ejemplo: asumiendo que
        # hay una columna 'Year' y otra 'Total_GEIs' para la demanda
        # o un valor que queramos predecir
        if "Year" not in df.columns or "Value" not in df.columns:
            st.error("No se encontraron columnas 'Year' y 'Value' en UNFCCC_v27.csv.")
            return None
        
        df["Year"] = pd.to_numeric(df["Year"], errors="coerce")
        df = df.dropna(subset=["Year", "Value"])
        
        X = df[["Year"]]
        y = df["Value"]
        
        model = lgb.LGBMRegressor()
        model.fit(X, y)
        
        joblib.dump(model, model_path)
        return model

def predict_demand(input_features):
    model = load_demand_model()
    if model:
        return model.predict(np.array([input_features]))[0]
    return None

def forecast_with_prophet(data: pd.DataFrame, periods: int = 24):
    """
    Ejemplo con Prophet, ajusta a tu estructura real de UNFCCC_v27
    con columnas 'ds' (fecha) y 'y' (valor).
    """
    m = Prophet()
    m.fit(data)
    future = m.make_future_dataframe(periods=periods, freq='Y')  # Ajusta freq según tu caso
    forecast = m.predict(future)
    return forecast

# --------------------------------------------------------------------
# Análisis histórico en UNFCCC_v27
# --------------------------------------------------------------------
def load_historical_mobility_data():
    # Reemplazamos la carga anterior con la de UNFCCC_v27
    df = get_unfccc_data()
    if df is not None:
        # Ajustar para Prophet: ds, y
        # Ejemplo: si 'Year' es el tiempo y 'Value' la columna
        # Se crea ds como una fecha ficticia (1-1-Year)
        if "Year" in df.columns and "Value" in df.columns:
            df["ds"] = pd.to_datetime(df["Year"], format="%Y", errors="coerce")
            df.rename(columns={"Value": "y"}, inplace=True)
            return df[["ds", "y"]].dropna()
        else:
            st.error("No se encontraron columnas 'Year' y 'Value' en UNFCCC_v27.csv.")
            return None
    return None

def analyze_historical_data():
    df = load_historical_mobility_data()
    if df is None:
        st.error("No se pudo analizar UNFCCC_v27.csv.")
        return
    st.write("Resumen de datos históricos de UNFCCC (GEI):")
    max_val = df["y"].max()
    min_val = df["y"].min()
    avg_val = df["y"].mean()
    st.write(f"- Máximo: {max_val}")
    st.write(f"- Mínimo: {min_val}")
    st.write(f"- Promedio: {avg_val:.2f}")
    
    fig = px.line(df, x="ds", y="y", title="Evolución de Emisiones/Absorciones (UNFCCC)")
    st.plotly_chart(fig)

# --------------------------------------------------------------------
# Cálculo de CAE
# --------------------------------------------------------------------
def calculate_cae(volume_kg: float):
    factor = 0.001
    return volume_kg * factor

# --------------------------------------------------------------------
# Formulario de Integración
# --------------------------------------------------------------------
def show_integration_form():
    with st.form("integration_form_final", clear_on_submit=True):
        nombre = st.text_input("Nombre", key="api_nombre")
        apellidos = st.text_input("Apellidos", key="api_apellidos")
        institucion = st.text_input("Institución o Empresa", key="api_institucion")
        mensaje = st.text_area("Mensaje (opcional)", key="api_mensaje")
        submitted = st.form_submit_button("Enviar Solicitud")
        if submitted:
            st.success("Solicitud enviada. Nos pondremos en contacto contigo.")

# --------------------------------------------------------------------
# Funciones de OSRM y Mapa
# --------------------------------------------------------------------
def show_map_and_traffic():
    st.subheader("Cálculo de Ruta (OSRM)")
    with st.form("traffic_form_final", clear_on_submit=True):
        col1, col2, col3 = st.columns(3)
        with col1:
            origin = st.text_input("Origen", "Plaza Mayor, Madrid", key="traffic_origin")
        with col2:
            destination = st.text_input("Destino", "Puerta del Sol, Madrid", key="traffic_destination")
        with col3:
            start_time = st.time_input("Hora de Inicio", datetime.now().time(), key="traffic_time")
        submitted = st.form_submit_button("Calcular Ruta")
    
    if submitted:
        origin_coords, origin_full = geocode_address(origin)
        dest_coords, dest_full = geocode_address(destination)
        if not origin_coords or not dest_coords:
            st.error("Direcciones inválidas.")
            return
        start_dt = datetime(datetime.now().year, datetime.now().month, datetime.now().day,
                            start_time.hour, start_time.minute)
        weather = get_weather_open_meteo(origin_coords[0], origin_coords[1])
        zone_factor = simulate_zone_factor(origin_full)
        route_data = get_route_osrm(origin_coords, dest_coords)
        if not route_data:
            st.error("No se pudo obtener la ruta con OSRM.")
            return
        segments, arrival_time = simulate_route_segments(route_data["steps"], start_dt, weather, zone_factor)
        
        m_map = folium.Map(location=[(origin_coords[0] + dest_coords[0]) / 2,
                                     (origin_coords[1] + dest_coords[1]) / 2],
                           zoom_start=12, tiles="OpenStreetMap")
        for seg in segments:
            folium.PolyLine([(pt[1], pt[0]) for pt in seg["coords"]], color=seg["color"], weight=4).add_to(m_map)
        folium.Marker(origin_coords, tooltip="Origen", icon=folium.Icon(color="green")).add_to(m_map)
        folium.Marker(dest_coords, tooltip="Destino", icon=folium.Icon(color="red")).add_to(m_map)
        
        st_folium(m_map, width=700)
        dist_km = route_data["distance"] / 1000.0
        total_time_s = sum(s["time_s"] for s in segments)
        st.write(f"Distancia: {dist_km:.2f} km")
        st.write(f"Hora de llegada: {arrival_time.strftime('%H:%M')} (~{int(total_time_s/60)} min)")
        
        if weather:
            st.write(f"Clima: {weather['temperatura']}°C, Viento: {weather['viento']} km/h, "
                     f"Precipitación: {weather['precipitacion']} mm, Nubosidad: {weather['nubosidad']}%")

# --------------------------------------------------------------------
# Predicción de Demanda y Tráfico
# --------------------------------------------------------------------
def show_forecast():
    st.subheader("Predicción de Demanda y Tráfico (Simulado)")
    hours = list(range(1, 25))
    offset = st.selectbox("Horas en el futuro:", hours, key="forecast_hours")
    
    # Predicción de tráfico (simulado)
    traffic_info = simulate_traffic_forecast(offset)
    st.write(f"Nivel de Saturación: {traffic_info['saturacion']}")
    st.write(f"Tiempo Normal: {traffic_info['tiempo_normal']} min")
    st.write(f"Tiempo Proyectado: {traffic_info['tiempo_proyectado']} min")
    
    # Predicción de demanda (LightGBM y Prophet) con UNFCCC
    df_hist = load_historical_mobility_data()
    if df_hist is not None:
        st.write("**Predicción de Demanda con LightGBM:**")
        current_year = datetime.now().year + offset
        pred = predict_demand([current_year])
        if pred is not None:
            st.write(f"Demanda estimada para el año {current_year}: {pred:.2f}")
        
        st.write("**Predicción de Demanda con Prophet (ejemplo):**")
        forecast = forecast_with_prophet(df_hist, periods=5)  # 5 periodos anuales
        fig = px.line(forecast, x="ds", y="yhat", title="Demanda (Prophet)")
        st.plotly_chart(fig)

# --------------------------------------------------------------------
# Optimización de Rutas con Múltiples Paradas
# --------------------------------------------------------------------
def show_trip_service():
    st.subheader("Optimización de Ruta con Múltiples Paradas (OSRM Trip Service)")
    addresses = st.text_area("Direcciones (una por línea):", "Plaza Mayor, Madrid\nPuerta del Sol, Madrid\nAtocha, Madrid")
    if st.button("Optimizar Ruta", key="btn_trip"):
        lines = [a.strip() for a in addresses.split("\n") if a.strip()]
        if len(lines) < 2:
            st.error("Ingrese al menos 2 direcciones.")
            return
        coords_list = []
        for addr in lines:
            geoc, full_addr = geocode_address(addr)
            if not geoc:
                st.error(f"No se pudo geocodificar: {addr}")
                return
            coords_list.append((geoc[0], geoc[1]))
        
        trip_data = get_route_osrm_multiple(coords_list)
        if not trip_data:
            st.error("No se pudo obtener la ruta con OSRM (Trip Service).")
            return
        
        geometry = trip_data["geometry"]
        distance_m = trip_data["distance"]
        duration_s = trip_data["duration"]
        
        m_trip = folium.Map(location=[coords_list[0][0], coords_list[0][1]], zoom_start=12, tiles="OpenStreetMap")
        folium.PolyLine([(pt[1], pt[0]) for pt in geometry["coordinates"]], color="#2ecc71", weight=4).add_to(m_trip)
        
        for i, (lat, lon) in enumerate(coords_list):
            folium.Marker((lat, lon), tooltip=f"Parada {i+1}", icon=folium.Icon(color="blue")).add_to(m_trip)
        
        st_folium(m_trip, width=700)
        st.write(f"Distancia total: {distance_m/1000:.2f} km")
        st.write(f"Duración total: ~{int(duration_s/60)} min")

# --------------------------------------------------------------------
# Sección para Empresas
# --------------------------------------------------------------------
def empresas_section():
    st.header("Empresas")
    tabs = st.tabs(["Mapa y Tráfico", "Predicción de Demanda", "Optimización Múltiples Paradas", "Calculadora CAE", "Integración API"])
    
    with tabs[0]:
        show_map_and_traffic()
    
    with tabs[1]:
        show_forecast()
    
    with tabs[2]:
        show_trip_service()
    
    with tabs[3]:
        st.write("Calculadora de CAE")
        volume = st.number_input("CO₂ evitado (kg)", min_value=0.0, value=1000.0, step=100.0, key="coe_volume_emp")
        if st.button("Calcular CAE", key="calc_cae_emp"):
            cae_val = calculate_cae(volume)
            st.write(f"CAE estimados: {cae_val:.2f}")
    
    with tabs[4]:
        show_integration_form()

# --------------------------------------------------------------------
# Aplicación Principal
# --------------------------------------------------------------------
def main_app():
    st.title("Trafiquea (Versión Solo Empresas)")
    st.markdown("""
    Este dashboard se centra en funcionalidades para empresas logísticas:
    - **Open‑Meteo** para datos meteorológicos,
    - **OSRM** para rutas (incluyendo múltiples paradas),
    - **UNFCCC_v27.csv** (datos GEI) para entrenar/cargar un modelo LightGBM,
    - **Eurostat** (opcional) para datos de transporte en España,
    - **Predicción** de demanda con LightGBM/Prophet,
    - **Calculadora CAE** y formularios de integración.
    """)
    
    # Directo a la sección de empresas
    empresas_section()

if __name__ == "__main__":
    main_app()
