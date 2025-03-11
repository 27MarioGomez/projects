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
# Configuración de la carpeta de datos
# --------------------------------------------------------------------
DATA_DIR = "data"
os.makedirs(DATA_DIR, exist_ok=True)

# --------------------------------------------------------------------
# Funciones de Geocodificación
# --------------------------------------------------------------------
def geocode_address(address: str):
    geolocator = Nominatim(user_agent="trafiquea_dashboard")
    location = geolocator.geocode(address)
    if location:
        return (location.latitude, location.longitude), location.address
    return None, None

# --------------------------------------------------------------------
# Función para descargar y cargar archivos CSV en carpeta data
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
# Open‑Meteo: Datos Meteorológicos
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
    resp = requests.get(url, params=params)
    if resp.status_code == 200:
        data = resp.json()
        current = data.get("current_weather", {})
        return {
            "temperatura": current.get("temperature"),
            "viento": current.get("windspeed"),
            "precipitacion": data.get("hourly", {}).get("precipitation", [0])[0],
            "nubosidad": data.get("hourly", {}).get("cloudcover", [0])[0]
        }
    return None

# --------------------------------------------------------------------
# OSRM: Cálculo de rutas y Trip Service
# --------------------------------------------------------------------
def get_route_osrm(origin_coords, destination_coords):
    base_url = "http://router.project-osrm.org/route/v1/driving"
    # OSRM requiere coordenadas en formato "lon,lat"
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
    # Formatear coordenadas: "lon,lat;lon,lat;..."
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
# Eurostat: Consulta de Datos de Transporte para España (opcional)
# --------------------------------------------------------------------
@st.cache_data(ttl=3600)
def get_eurostat_transport_data():
    dataset_id = "tsdtr020"  # Ajusta según lo que necesites
    base_url = f"https://ec.europa.eu/eurostat/api/dissemination/statistics/1.0/data/{dataset_id}"
    params = {"format": "JSON", "geo": "ES"}
    resp = requests.get(base_url, params=params)
    if resp.status_code == 200:
        return resp.json()
    return None

# --------------------------------------------------------------------
# EEA Code API: Ejemplo de consulta (opcional)
# --------------------------------------------------------------------
@st.cache_data(ttl=3600)
def get_eea_code_api_content_types():
    url = "https://www.eea.europa.eu/code/api/content-types/@@view"
    r = requests.get(url)
    if r.status_code == 200:
        return r.text  # Retorna HTML
    return None

# --------------------------------------------------------------------
# UNFCCC_v27.csv: Datos de Emisiones/Absorciones GEI
# --------------------------------------------------------------------
@st.cache_data(ttl=3600)
def get_unfccc_data():
    filename = "UNFCCC_v27.csv"
    filepath = os.path.join(DATA_DIR, filename)
    if not os.path.exists(filepath):
        st.error(f"No se encontró {filename} en la carpeta {DATA_DIR}.")
        return None
    try:
        df = pd.read_csv(filepath)
        return df
    except Exception as e:
        st.error(f"Error al cargar {filename}: {e}")
        return None

# --------------------------------------------------------------------
# Filtrado para datos de España y agrupación por año
# --------------------------------------------------------------------
def load_historical_mobility_data():
    df = get_unfccc_data()
    if df is None:
        return None
    # Filtrar solo para España (Country_code == "ES")
    df_es = df[df["Country_code"] == "ES"]
    if df_es.empty:
        st.error("No se encontraron datos para España en UNFCCC_v27.csv.")
        return None
    # Opcional: filtrar por pollutante y formato, por ejemplo:
    # df_es = df_es[(df_es["Format_name"] == "IPCC Common Reporting Format") & 
    #               (df_es["Pollutant_name"] == "All greenhouse gases - (CO2 equivalent)")]
    # Agrupar por año y sumar emisiones
    df_grouped = df_es.groupby("Year")["emissions"].sum().reset_index()
    # Crear columna de fecha (usamos 1 de enero de cada año)
    df_grouped["ds"] = pd.to_datetime(df_grouped["Year"].astype(str) + "-01-01", format="%Y-%m-%d")
    df_grouped.rename(columns={"emissions": "y"}, inplace=True)
    df_grouped = df_grouped[["ds", "y"]].sort_values("ds")
    return df_grouped

# --------------------------------------------------------------------
# Modelo de ML: Entrenamiento/Predicción con LightGBM y Prophet
# --------------------------------------------------------------------
@st.cache_resource
def load_demand_model():
    model_path = "model_demand_lgbm.pkl"
    try:
        model = joblib.load(model_path)
        return model
    except:
        st.warning("Modelo no encontrado. Entrenando nuevo modelo LightGBM con datos de UNFCCC_v27.csv para España.")
        df = load_historical_mobility_data()
        if df is None:
            st.error("No se pudieron cargar datos históricos para entrenar el modelo.")
            return None
        # Usamos el año (como número) para entrenar
        df["year_num"] = df["ds"].dt.year
        X = df[["year_num"]]
        y = df["y"]
        model = lgb.LGBMRegressor()
        model.fit(X, y)
        joblib.dump(model, model_path)
        return model

def predict_demand(input_features):
    model = load_demand_model()
    if model:
        return model.predict(np.array([input_features]))[0]
    return None

def forecast_with_prophet(df: pd.DataFrame, periods: int = 5):
    if "ds" not in df.columns or "y" not in df.columns:
        st.error("No se encontraron columnas 'ds' y 'y' para Prophet.")
        return pd.DataFrame()
    m = Prophet()
    m.fit(df)
    future = m.make_future_dataframe(periods=periods, freq="Y")
    forecast = m.predict(future)
    return forecast

# --------------------------------------------------------------------
# Simulación de Tráfico (por hora)
# --------------------------------------------------------------------
def simulate_traffic_forecast(hour_offset: int):
    if hour_offset <= 2:
        saturacion = "Bajo"
        factor = 1.0
    elif hour_offset <= 4:
        saturacion = "Moderado"
        factor = 1.2
    else:
        saturacion = "Alto"
        factor = 1.5
    tiempo_normal = np.random.randint(10, 15)
    tiempo_proyectado = round(tiempo_normal * factor)
    return {
        "saturacion": saturacion,
        "tiempo_normal": tiempo_normal,
        "tiempo_proyectado": tiempo_proyectado
    }

# --------------------------------------------------------------------
# Mapa y Tráfico (OSRM)
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
        zone_factor = 1.0
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
    st.subheader("Predicción de Demanda y Tráfico")
    hours = list(range(1, 25))
    selected_hour = st.selectbox("Horas en el futuro (tráfico simulado):", hours, key="forecast_hours")
    
    traffic_info = simulate_traffic_forecast(selected_hour)
    st.write(f"Nivel de Saturación: {traffic_info['saturacion']}")
    st.write(f"Tiempo Normal: {traffic_info['tiempo_normal']} min")
    st.write(f"Tiempo Proyectado: {traffic_info['tiempo_proyectado']} min")
    
    df_hist = load_historical_mobility_data()
    if df_hist is not None and not df_hist.empty:
        st.write("**Predicción de Demanda (LightGBM)**")
        # Se utiliza el año como variable para predecir
        current_year = datetime.now().year + selected_hour
        pred = predict_demand([current_year])
        if pred is not None:
            st.write(f"Valor estimado para el año {current_year}: {pred:.2f}")
        
        st.write("**Predicción de Demanda (Prophet)**")
        forecast = forecast_with_prophet(df_hist, periods=5)
        if not forecast.empty:
            fig = px.line(forecast, x="ds", y="yhat", title="Proyección (Prophet)")
            st.plotly_chart(fig)

# --------------------------------------------------------------------
# Optimización de Rutas con Múltiples Paradas (Trip Service)
# --------------------------------------------------------------------
def show_trip_service():
    st.subheader("Optimización de Ruta con Múltiples Paradas (OSRM Trip Service)")
    addresses = st.text_area("Direcciones (una por línea):", 
                             "Plaza Mayor, Madrid\nPuerta del Sol, Madrid\nAtocha, Madrid")
    if st.button("Optimizar Ruta", key="btn_trip"):
        lines = [line.strip() for line in addresses.split("\n") if line.strip()]
        if len(lines) < 2:
            st.error("Ingrese al menos 2 direcciones.")
            return
        coords_list = []
        for addr in lines:
            geoc, _ = geocode_address(addr)
            if not geoc:
                st.error(f"No se pudo geocodificar: {addr}")
                return
            coords_list.append((geoc[0], geoc[1]))
        
        trip_data = get_route_osrm_multiple(coords_list)
        if not trip_data:
            st.error("No se pudo obtener la ruta optimizada con OSRM (Trip Service).")
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
# Calculadora de CAE
# --------------------------------------------------------------------
def calculate_cae(volume_kg: float):
    return volume_kg * 0.001

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
# Carga y procesamiento de UNFCCC_v27.csv para España
# --------------------------------------------------------------------
def load_historical_mobility_data():
    df = get_unfccc_data()
    if df is not None:
        # Filtrar para España (Country_code == "ES")
        df_es = df[df["Country_code"] == "ES"]
        if df_es.empty:
            st.error("No se encontraron datos para España en UNFCCC_v27.csv.")
            return None
        # Opcional: filtrar por pollutante si se desea
        # df_es = df_es[df_es["Pollutant_name"] == "All greenhouse gases - (CO2 equivalent)"]
        # Agrupar por año y sumar emisiones
        df_grouped = df_es.groupby("Year")["emissions"].sum().reset_index()
        df_grouped["ds"] = pd.to_datetime(df_grouped["Year"].astype(str) + "-01-01", format="%Y-%m-%d")
        df_grouped.rename(columns={"emissions": "y"}, inplace=True)
        df_grouped = df_grouped[["ds", "y"]].sort_values("ds")
        return df_grouped
    return None

# --------------------------------------------------------------------
# Sección para Empresas
# --------------------------------------------------------------------
def empresas_section():
    st.header("Empresas")
    tabs = st.tabs([
        "Mapa y Tráfico",
        "Predicción de Demanda",
        "Optimización Múltiples Paradas",
        "Calculadora CAE",
        "Integración API"
    ])
    
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
        st.write("**Datos de Eurostat (opcional):**")
        euro_data = get_eurostat_transport_data()
        if euro_data:
            st.write(euro_data)
        st.write("**EEA Code API (opcional):**")
        eea_code = get_eea_code_api_content_types()
        if eea_code:
            st.write("HTML devuelto (content-types):")
            st.code(eea_code[:500] + "...", language="html")

# --------------------------------------------------------------------
# Aplicación Principal
# --------------------------------------------------------------------
def main_app():
    st.title("Trafiquea: Dashboard para Empresas")
    st.markdown("""
    Este dashboard se centra en soluciones para empresas logísticas, integrando:
    - Datos meteorológicos de **Open‑Meteo**,
    - Rutas y optimización con **OSRM**,
    - Predicción de demanda de movilidad basada en datos de **UNFCCC_v27.csv** (filtrado para España),
    - Información de transporte de **Eurostat** y datos de la **EEA Code API** (opcional),
    - Simulación de tráfico por hora y cálculo de CAE.
    """)
    
    empresas_section()

if __name__ == "__main__":
    main_app()
