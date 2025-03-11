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

# Crear carpeta de datos
DATA_DIR = "data"
os.makedirs(DATA_DIR, exist_ok=True)

# =============================================================================
# Funciones de Geocodificación
# =============================================================================

def geocode_address(address: str):
    geolocator = Nominatim(user_agent="trafiquea_dashboard")
    location = geolocator.geocode(address)
    if location:
        return (location.latitude, location.longitude), location.address
    return None, None

def reverse_geocode(lat: float, lon: float):
    geolocator = Nominatim(user_agent="trafiquea_dashboard")
    location = geolocator.reverse((lat, lon), language="es")
    if location:
        return location.address
    return "Dirección no encontrada"

# =============================================================================
# Función para descargar y cargar archivos CSV localmente
# =============================================================================

def download_file(url, filename, sep=None, encoding="utf-8"):
    filepath = os.path.join(DATA_DIR, filename)
    if not os.path.exists(filepath):
        response = requests.get(url)
        if response.status_code == 200:
            with open(filepath, "wb") as f:
                f.write(response.content)
    try:
        if sep:
            df = pd.read_csv(filepath, sep=sep, encoding=encoding)
        else:
            df = pd.read_csv(filepath, encoding=encoding)
        return df
    except Exception as e:
        st.error(f"Error al cargar {filename}: {e}")
        return None

# =============================================================================
# Integración de APIs y Descargas de Datos
# =============================================================================

def get_weather_open_meteo(lat: float, lon: float):
    url = "https://api.open-meteo.com/v1/forecast"
    params = {
        "latitude": lat,
        "longitude": lon,
        "current_weather": True,
        "hourly": "temperature_2m,windspeed_10m,precipitation,cloudcover",
        "timezone": "auto"
    }
    response = requests.get(url, params=params)
    if response.status_code == 200:
        data = response.json()
        current = data.get("current_weather", {})
        return {
            "temperatura": current.get("temperature"),
            "viento": current.get("windspeed"),
            "precipitacion": data.get("hourly", {}).get("precipitation", [0])[0],
            "nubosidad": data.get("hourly", {}).get("cloudcover", [0])[0]
        }
    return None

def get_route_osrm(origin_coords, destination_coords):
    base_url = "http://router.project-osrm.org/route/v1/driving"
    # OSRM espera coordenadas en formato "lon,lat"
    coords = f"{origin_coords[1]},{origin_coords[0]};{destination_coords[1]},{destination_coords[0]}"
    params = {"overview": "full", "geometries": "geojson", "steps": "true"}
    url = f"{base_url}/{coords}"
    response = requests.get(url, params=params)
    if response.status_code == 200:
        data = response.json()
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
    response = requests.get(url, params=params)
    if response.status_code == 200:
        data = response.json()
        if "trips" in data and data["trips"]:
            return data["trips"][0]
    return None

@st.cache_data(ttl=3600)
def get_eea_air_quality():
    url = "https://discomap.eea.europa.eu/AQER/xml/aqmdata.xml"
    response = requests.get(url)
    if response.status_code == 200:
        data_xml = xmltodict.parse(response.content)
        return data_xml
    return None

@st.cache_data(ttl=3600)
def get_eea_mobility_data():
    csv_url = "https://www.eea.europa.eu/system/files/2022_CO2_emissions_passenger_cars.csv"
    return download_file(csv_url, "CO2_emissions_passenger_cars.csv", sep=";", encoding="latin1")

@st.cache_data(ttl=3600)
def get_eea_noise_data():
    noise_url = "https://www.eea.europa.eu/system/files/2022_road_traffic_noise.csv"
    return download_file(noise_url, "road_traffic_noise.csv", encoding="utf-8")

@st.cache_data(ttl=3600)
def load_municipios():
    url = "https://www.ine.es/daco/daco42/codmun/codmun_en.csv"
    return download_file(url, "municipios_es.csv", sep=",", encoding="utf-8")

@st.cache_data(ttl=3600)
def get_eurostat_transport_data():
    dataset_id = "tsdtr020"
    url = f"https://ec.europa.eu/eurostat/api/dissemination/statistics/1.0/data/{dataset_id}?format=JSON&geo=ES"
    response = requests.get(url)
    if response.status_code == 200:
        return response.json()
    return None

def get_real_traffic_data(ayuntamiento: str):
    # Reemplazar por consulta oficial de tráfico local; se utiliza ejemplo
    return {
        "congestion": np.random.choice(["Bajo", "Moderado", "Alto"]),
        "emisiones_co2": np.random.uniform(100, 300)
    }

# =============================================================================
# Simulación y Modelo de ML
# =============================================================================

def simulate_zone_factor(full_address: str):
    address_lower = full_address.lower()
    if "talavera" in address_lower:
        return 1.3
    elif "toledo" in address_lower:
        return 1.25
    return 1.0

def assign_traffic_level(distance_km, start_time, weather, zone_factor):
    hour = start_time.hour
    base_factor = 1.5 if (7 <= hour <= 9 or 17 <= hour <= 19) else 1.2
    if weather:
        if weather.get("viento") and weather["viento"] > 20:
            base_factor += 0.2
        if weather.get("precipitacion") and weather["precipitacion"] > 0.5:
            base_factor += 0.3
        if weather.get("nubosidad") and weather["nubosidad"] > 70:
            base_factor += 0.1
    if distance_km < 1:
        base_factor -= 0.1
    elif distance_km > 5:
        base_factor += 0.1
    base_factor *= zone_factor
    if base_factor <= 1.3:
        level = "Bajo"
    elif 1.3 < base_factor < 1.6:
        level = "Moderado"
    else:
        level = "Alto"
    return level, max(1.0, base_factor)

def color_for_level(level):
    if level == "Bajo":
        return "#2ecc71"
    elif level == "Moderado":
        return "#f39c12"
    else:
        return "#e74c3c"

def simulate_route_segments(steps, start_time, weather, zone_factor):
    segments = []
    current_time = start_time
    for step in steps:
        distance_m = step["distance"]
        distance_km = distance_m / 1000.0
        level, factor = assign_traffic_level(distance_km, current_time, weather, zone_factor)
        duration_s = step["duration"] * factor
        current_time += timedelta(seconds=duration_s)
        segments.append({
            "coords": step["geometry"]["coordinates"],
            "color": color_for_level(level),
            "distance_km": distance_km,
            "time_s": duration_s,
            "level": level
        })
    return segments, current_time

def simulate_traffic_forecast(hour_offset):
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
    return {"saturacion": saturacion, "tiempo_normal": tiempo_normal, "tiempo_proyectado": tiempo_proyectado}

@st.cache_resource
def load_demand_model():
    try:
        model = joblib.load("model_demand_lgbm.pkl")
        return model
    except Exception:
        st.warning("Modelo no encontrado, entrenando modelo LightGBM básico...")
        df = get_eea_mobility_data()
        if df is None:
            st.error("No se pudieron cargar datos históricos para entrenar el modelo.")
            return None
        df.rename(columns={df.columns[0]: "ds", df.columns[1]: "y"}, inplace=True)
        df["ds"] = pd.to_datetime(df["ds"])
        df["day"] = df["ds"].dt.dayofyear
        X = df[["day"]]
        y = df["y"]
        model = lgb.LGBMRegressor()
        model.fit(X, y)
        joblib.dump(model, "model_demand_lgbm.pkl")
        return model

def predict_demand(input_features):
    model = load_demand_model()
    if model:
        return model.predict(np.array([input_features]))[0]
    return None

def forecast_with_prophet(data: pd.DataFrame, periods: int = 24):
    model = Prophet()
    model.fit(data)
    future = model.make_future_dataframe(periods=periods, freq='H')
    forecast = model.predict(future)
    return forecast

# =============================================================================
# Análisis Histórico y Visualización
# =============================================================================

def load_historical_mobility_data():
    df = get_eea_mobility_data()
    if df is not None:
        df.rename(columns={df.columns[0]: "ds", df.columns[1]: "y"}, inplace=True)
        df["ds"] = pd.to_datetime(df["ds"])
        return df
    return None

def load_noise_data():
    return get_eea_noise_data()

def analyze_historical_data():
    df = load_historical_mobility_data()
    if df is None:
        st.error("No se pudieron cargar datos históricos de movilidad.")
        return
    st.write("Resumen de datos históricos (EEA):")
    max_val = df["y"].max()
    min_val = df["y"].min()
    avg_val = df["y"].mean()
    st.write(f"- Máximo: {max_val}")
    st.write(f"- Mínimo: {min_val}")
    st.write(f"- Promedio: {avg_val:.2f}")
    fig = px.line(df, x="ds", y="y", title="Evolución de la Movilidad/Emisiones (EEA)")
    st.plotly_chart(fig)
    
    noise_df = load_noise_data()
    if noise_df is not None:
        st.write("Datos de Ruido Ambiental (EEA):")
        st.dataframe(noise_df)

# =============================================================================
# Eurostat
# =============================================================================

@st.cache_data(ttl=3600)
def get_eurostat_transport_data():
    dataset_id = "tsdtr020"
    url = f"https://ec.europa.eu/eurostat/api/dissemination/statistics/1.0/data/{dataset_id}?format=JSON&geo=ES"
    response = requests.get(url)
    if response.status_code == 200:
        return response.json()
    return None

# =============================================================================
# Calculadora de CAE
# =============================================================================

def calculate_cae(volume_kg: float):
    return volume_kg * 0.001

# =============================================================================
# Formulario de Integración
# =============================================================================

def show_integration_form():
    with st.form("integration_form_final", clear_on_submit=True):
        nombre = st.text_input("Nombre", key="api_nombre")
        apellidos = st.text_input("Apellidos", key="api_apellidos")
        institucion = st.text_input("Institución o Empresa", key="api_institucion")
        mensaje = st.text_area("Mensaje (opcional)", key="api_mensaje")
        submitted = st.form_submit_button("Enviar Solicitud")
        if submitted:
            st.success("Solicitud enviada. Nos pondremos en contacto contigo.")

# =============================================================================
# Interfaz: Mapas, Predicción, Simulación, Trip Service
# =============================================================================

def show_map_and_traffic():
    st.subheader("Cálculo de Ruta (OSRM)")
    with st.form("traffic_form_final", clear_on_submit=True):
        col1, col2, col3 = st.columns(3)
        with col1:
            origin = st.text_input("Origen", "Plaza Mayor, Madrid, España", key="traffic_origin")
        with col2:
            destination = st.text_input("Destino", "Puerta del Sol, Madrid, España", key="traffic_destination")
        with col3:
            start_time = st.time_input("Hora de Inicio", datetime.now().time(), key="traffic_time")
        submitted = st.form_submit_button("Calcular Ruta")
    
    if submitted:
        origin_coords, origin_full = geocode_address(origin)
        dest_coords, dest_full = geocode_address(destination)
        if not (origin_coords and dest_coords):
            st.error("Ingrese direcciones válidas.")
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
            folium.PolyLine([(pt[1], pt[0]) for pt in seg["coords"]],
                            color=seg["color"], weight=4).add_to(m_map)
        folium.Marker(origin_coords, tooltip="Origen", icon=folium.Icon(color="green")).add_to(m_map)
        folium.Marker(dest_coords, tooltip="Destino", icon=folium.Icon(color="red")).add_to(m_map)
        st_folium(m_map, width=700)
        dist_km = route_data["distance"] / 1000.0
        total_time_s = sum(seg["time_s"] for seg in segments)
        st.write(f"Distancia: {dist_km:.2f} km")
        st.write(f"Hora de llegada: {arrival_time.strftime('%H:%M')} (~{int(total_time_s/60)} min)")
        if weather:
            st.write(f"Clima: {weather['temperatura']}°C, Viento: {weather['viento']} km/h, "
                     f"Precipitación: {weather['precipitacion']} mm, Nubosidad: {weather['nubosidad']}%")

def show_forecast():
    st.subheader("Predicción de Demanda y Tráfico")
    hours = list(range(1,25))
    selected_hour = st.selectbox("Horas en el futuro:", hours, key="forecast_hours")
    col1, col2 = st.columns(2)
    with col1:
        origin = st.text_input("Origen (Pronóstico)", "Plaza Mayor, Madrid, España", key="forecast_origin")
    with col2:
        destination = st.text_input("Destino (Pronóstico)", "Puerta del Sol, Madrid, España", key="forecast_destination")
    if st.button("Mostrar Pronóstico", key="btn_forecast"):
        origin_coords, origin_full = geocode_address(origin)
        dest_coords, dest_full = geocode_address(destination)
        if not (origin_coords and dest_coords):
            st.error("Direcciones inválidas.")
            return
        future_dt = datetime.now() + timedelta(hours=selected_hour)
        weather = get_weather_open_meteo(origin_coords[0], origin_coords[1])
        zone_factor = simulate_zone_factor(origin_full)
        route_data = get_route_osrm(origin_coords, dest_coords)
        if not route_data:
            st.error("No se pudo obtener la ruta.")
            return
        segments, arrival_time = simulate_route_segments(route_data["steps"], future_dt, weather, zone_factor)
        m_fore = folium.Map(location=[(origin_coords[0] + dest_coords[0]) / 2,
                                      (origin_coords[1] + dest_coords[1]) / 2],
                             zoom_start=12, tiles="OpenStreetMap")
        for seg in segments:
            folium.PolyLine([(pt[1], pt[0]) for pt in seg["coords"]],
                            color=seg["color"], weight=4).add_to(m_fore)
        folium.Marker(origin_coords, tooltip="Origen", icon=folium.Icon(color="green")).add_to(m_fore)
        folium.Marker(dest_coords, tooltip="Destino", icon=folium.Icon(color="red")).add_to(m_fore)
        st_folium(m_fore, width=700)
        dist_km = route_data["distance"] / 1000.0
        total_time_s = sum(seg["time_s"] for seg in segments)
        st.write(f"Distancia: {dist_km:.2f} km")
        st.write(f"Salida: {future_dt.strftime('%H:%M')}")
        st.write(f"Llegada: {arrival_time.strftime('%H:%M')} (~{int(total_time_s/60)} min)")
        if weather:
            st.write(f"Clima: {weather['temperatura']}°C, Viento: {weather['viento']} km/h, "
                     f"Precipitación: {weather['precipitacion']} mm, Nubosidad: {weather['nubosidad']}%")

def show_simulation_advanced(key_prefix="default"):
    st.subheader("Simulación Avanzada (Múltiples Paradas)")
    st.write("Ingrese cada dirección en una línea (mínimo 3).")
    addresses_text = st.text_area("Direcciones:", "Plaza Mayor, Madrid\nPuerta del Sol, Madrid\nAtocha, Madrid", key=f"trip_addresses_{key_prefix}")
    if st.button("Optimizar Ruta", key=f"btn_trip_{key_prefix}"):
        addresses = [addr.strip() for addr in addresses_text.split("\n") if addr.strip()]
        if len(addresses) < 3:
            st.error("Ingrese al menos 3 direcciones.")
            return
        coords_list = []
        for addr in addresses:
            geoc, full_addr = geocode_address(addr)
            if not geoc:
                st.error(f"No se pudo geocodificar: {addr}")
                return
            coords_list.append((geoc[0], geoc[1]))
        trip_data = get_route_osrm_multiple(coords_list, profile="driving")
        if not trip_data:
            st.error("No se pudo obtener la ruta optimizada con múltiples paradas.")
            return
        geometry = trip_data["geometry"]
        distance_m = trip_data["distance"]
        duration_s = trip_data["duration"]
        m_trip = folium.Map(location=[coords_list[0][0], coords_list[0][1]], zoom_start=12, tiles="OpenStreetMap")
        folium.PolyLine([(pt[1], pt[0]) for pt in geometry["coordinates"]], color="#2ecc71", weight=4).add_to(m_trip)
        for idx, (lat, lon) in enumerate(coords_list):
            folium.Marker((lat, lon), tooltip=f"Parada {idx+1}", icon=folium.Icon(color="blue")).add_to(m_trip)
        st_folium(m_trip, width=700)
        st.write(f"Distancia total: {distance_m/1000:.2f} km")
        st.write(f"Duración total: ~{int(duration_s/60)} min")

# =============================================================================
# Selección de Municipios (Ayuntamientos)
# =============================================================================

def select_municipio():
    df = load_municipios()
    if df is not None:
        pais = st.selectbox("Seleccione País", sorted(df["PAIS"].unique()), key="pais_selector")
        df_pais = df[df["PAIS"] == pais]
        municipio = st.selectbox("Seleccione Municipio", sorted(df_pais["MUNICIPIO"].unique()), key="municipio_selector")
        return municipio
    return None

# =============================================================================
# Secciones: Ayuntamientos y Empresas
# =============================================================================

def ayuntamientos_section():
    st.header("Ayuntamientos")
    municipio = select_municipio()
    if municipio is None:
        st.error("No se pudo seleccionar un municipio.")
        return
    traffic_data = get_real_traffic_data(municipio)
    st.write(f"**Congestión actual:** {traffic_data['congestion']}")
    st.write(f"**Emisiones CO₂ estimadas:** {traffic_data['emisiones_co2']:.1f} kg/día")
    
    air_data = get_eea_air_quality()
    if air_data:
        st.write("**Datos de Calidad del Aire (EEA):**")
        st.write(air_data)
    
    noise_df = get_eea_noise_data()
    if noise_df is not None:
        st.write("**Datos de Ruido Ambiental (EEA):**")
        st.dataframe(noise_df)
    
    eurostat_data = get_eurostat_transport_data()
    if eurostat_data:
        st.write("**Datos de Transporte (Eurostat) para España:**")
        st.write(eurostat_data)
    
    ayto_tabs = st.tabs(["Mapa y Tráfico", "Predicción de Demanda", "Optimización (Múltiples Paradas)", "Integración API"])
    with ayto_tabs[0]:
        show_map_and_traffic()
    with ayto_tabs[1]:
        df_hist = load_historical_mobility_data()
        if df_hist is not None:
            st.write("**Predicción de Demanda con LightGBM:**")
            df_hist["day"] = df_hist["ds"].dt.dayofyear
            input_day = st.number_input("Ingrese el día del año:", min_value=1, max_value=366,
                                          value=datetime.now().timetuple().tm_yday, key="input_day")
            pred = predict_demand([input_day])
            if pred is not None:
                st.write(f"Predicción de demanda: {pred:.2f}")
            st.write("**Predicción de Demanda con Prophet:**")
            forecast = forecast_with_prophet(df_hist, periods=24)
            fig = px.line(forecast, x="ds", y="yhat", title="Predicción de Demanda (Prophet)")
            st.plotly_chart(fig)
    with ayto_tabs[2]:
        show_simulation_advanced(key_prefix="ayto")
    with ayto_tabs[3]:
        show_integration_form()

def empresas_section():
    st.header("Empresas")
    emp_tabs = st.tabs(["Mapa y Tráfico", "Predicción de Demanda", "Optimización (Múltiples Paradas)", "Calculadora CAE", "Integración API"])
    with emp_tabs[0]:
        show_map_and_traffic()
    with emp_tabs[1]:
        df_hist = load_historical_mobility_data()
        if df_hist is not None:
            st.write("**Predicción de Demanda con LightGBM:**")
            df_hist["day"] = df_hist["ds"].dt.dayofyear
            input_day = st.number_input("Ingrese el día del año:", min_value=1, max_value=366,
                                          value=datetime.now().timetuple().tm_yday, key="input_day_emp")
            pred = predict_demand([input_day])
            if pred is not None:
                st.write(f"Predicción de demanda: {pred:.2f}")
            st.write("**Predicción de Demanda con Prophet:**")
            forecast = forecast_with_prophet(df_hist, periods=24)
            fig = px.line(forecast, x="ds", y="yhat", title="Predicción de Demanda (Prophet)")
            st.plotly_chart(fig)
    with emp_tabs[2]:
        show_simulation_advanced(key_prefix="emp")
    with emp_tabs[3]:
        st.write("Calculadora de CAE")
        volume = st.number_input("CO₂ evitado (kg)", min_value=0.0, value=1000.0, step=100.0, key="coe_volume_emp")
        if st.button("Calcular CAE", key="calc_cae_emp"):
            cae_val = calculate_cae(volume)
            st.write(f"CAE estimados: {cae_val:.2f}")
    with emp_tabs[4]:
        show_integration_form()

# =============================================================================
# Aplicación Principal
# =============================================================================

def main_app():
    st.title("Trafiquea: Dashboard de Movilidad y Sostenibilidad")
    st.markdown("""
    Esta plataforma integra datos reales de:
    - **Open‑Meteo** (clima),
    - **OSRM** (rutas optimizadas sin uso de Mapbox),
    - **EEA** (calidad del aire, emisiones y ruido ambiental),
    - **Eurostat** (datos de transporte para España),
    - un dataset oficial de **municipios de España** (INE),
    - y modelos de ML entrenados con **LightGBM** y **Prophet** para predecir la demanda de movilidad.
    """)
    
    user_type = st.selectbox("Tipo de Usuario", ["Ayuntamiento", "Empresa"], key="user_type_selector")
    st.markdown("---")
    
    if user_type == "Ayuntamiento":
        ayuntamientos_section()
    else:
        empresas_section()

if __name__ == "__main__":
    main_app()
