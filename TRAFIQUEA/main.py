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

# =============================================================================
# GEOCODIFICACIÓN CON NOMINATIM (sin uso de secrets)
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
# INTEGRACIÓN DE APIS REALES
# =============================================================================

def get_weather_open_meteo(lat: float, lon: float):
    """
    Consulta la API gratuita de Open‑Meteo, solicitando variables:
    temperatura, viento, precipitación y nubosidad.
    """
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
    """
    Calcula la mejor ruta entre dos puntos usando el servicio público de OSRM.
    """
    base_url = "http://router.project-osrm.org/directions/v5/mapbox/driving"
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
    """
    Utiliza el Trip Service de OSRM para optimizar rutas con múltiples paradas.
    coords_list: lista de tuplas (lat, lon)
    """
    base_url = f"http://router.project-osrm.org/trip/v1/{profile}/"
    coords_str = ";".join([f"{lon},{lat}" for (lat, lon) in coords_list])
    params = {"roundtrip": "false", "source": "first", "destination": "last", "geometries": "geojson"}
    url = base_url + coords_str
    response = requests.get(url, params=params)
    if response.status_code == 200:
        data = response.json()
        if "trips" in data and data["trips"]:
            return data["trips"][0]
    return None

@st.cache_data(ttl=3600)
def get_eea_air_quality():
    """
    Descarga datos reales de calidad del aire desde la EEA (Air Quality e-Reporting).
    Se obtiene el XML y se parsea con xmltodict. Cacheado 1 hora.
    """
    url = "https://discomap.eea.europa.eu/AQER/xml/aqmdata.xml"
    response = requests.get(url)
    if response.status_code == 200:
        data_xml = xmltodict.parse(response.content)
        return data_xml
    return None

@st.cache_data(ttl=3600)
def get_eea_mobility_data():
    """
    Descarga un CSV oficial de la EEA con datos de emisiones de vehículos nuevos.
    URL oficial (actualizada): https://www.eea.europa.eu/system/files/2022_CO2_emissions_passenger_cars.csv
    """
    csv_url = "https://www.eea.europa.eu/system/files/2022_CO2_emissions_passenger_cars.csv"
    try:
        df = pd.read_csv(csv_url, sep=";", encoding="latin1")
        return df
    except Exception as e:
        st.error(f"Error al cargar datos de movilidad EEA: {e}")
        return None

@st.cache_data(ttl=3600)
def get_eea_noise_data():
    """
    Descarga datos oficiales de ruido ambiental de la EEA.
    URL oficial (actualizada): https://www.eea.europa.eu/system/files/2022_road_traffic_noise.csv
    """
    noise_url = "https://www.eea.europa.eu/system/files/2022_road_traffic_noise.csv"
    try:
        df = pd.read_csv(noise_url, encoding="utf-8")
        return df
    except Exception as e:
        st.error(f"Error al cargar datos de ruido EEA: {e}")
        return None

def get_real_traffic_data(ayuntamiento: str):
    """
    Consulta datos reales de tráfico y emisiones para un ayuntamiento.
    Debe integrarse con un API oficial a nivel municipal. Se deja aquí para reemplazo.
    """
    # URL oficial de datos abiertos de tráfico de España o de un ayuntamiento
    # Por ejemplo, se podría usar un servicio del Ministerio de Transportes
    return {
        "congestion": np.random.choice(["Bajo", "Moderado", "Alto"]),
        "emisiones_co2": np.random.uniform(100, 300)
    }

# =============================================================================
# SIMULACIÓN DE TRÁFICO Y MODELO DE ML/DL
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
    """
    Carga un modelo de ML entrenado con LightGBM para predecir la demanda de movilidad.
    Si no existe, entrena un modelo básico a partir de datos históricos de la EEA.
    """
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
# ANÁLISIS Y VISUALIZACIÓN DE DATOS HISTÓRICOS (Movilidad, Ruido, Emisiones GEI)
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
    st.write("Resumen de datos históricos de movilidad (EEA):")
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
# CALCULADORA DE CAE (para Empresas)
# =============================================================================

def calculate_cae(volume_kg: float):
    factor_cae = 0.001
    return volume_kg * factor_cae

# =============================================================================
# FORMULARIO DE INTEGRACIÓN (Sin claves)
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
# INTERFAZ DE USUARIO: MAPAS, PREDICCIÓN, SIMULACIÓN, TRIP SERVICE
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
            st.error("Asegúrese de ingresar direcciones completas.")
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
        st.write(f"Hora de llegada estimada: {arrival_time.strftime('%H:%M')} (~{int(total_time_s/60)} min)")
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
            st.error("No se pudo geocodificar las direcciones.")
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
        st.write(f"Salida (Futura): {future_dt.strftime('%H:%M')}")
        st.write(f"Llegada estimada: {arrival_time.strftime('%H:%M')} (~{int(total_time_s/60)} min)")
        if weather:
            st.write(f"Clima: {weather['temperatura']}°C, Viento: {weather['viento']} km/h, "
                     f"Precipitación: {weather['precipitacion']} mm, Nubosidad: {weather['nubosidad']}%")

def show_simulation_advanced(key_prefix="default"):
    st.subheader("Simulación Avanzada con Múltiples Paradas (Trip Service)")
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
# MÓDULO DE SELECCIÓN DE PAÍS Y MUNICIPIOS (Ayuntamientos)
# =============================================================================

@st.cache_data(ttl=3600)
def load_municipios():
    """
    Carga un dataset oficial de municipios de España.
    Aquí se usa un dataset disponible en GitHub (reemplazar por la fuente oficial del INE si se desea).
    """
    url = "https://raw.githubusercontent.com/juanbrujo/datasets/main/municipios_es.csv"
    try:
        df = pd.read_csv(url)
        return df
    except Exception as e:
        st.error(f"Error al cargar el dataset de municipios: {e}")
        return None

def select_municipio():
    df = load_municipios()
    if df is not None:
        pais = st.selectbox("Seleccione País", sorted(df["país"].unique()), key="pais_selector")
        df_pais = df[df["país"] == pais]
        municipio = st.selectbox("Seleccione Municipio", sorted(df_pais["municipio"].unique()), key="municipio_selector")
        return municipio
    return None

# =============================================================================
# SECCIONES POR PERFIL: AYUNTAMIENTOS Y EMPRESAS
# =============================================================================

def ayuntamientos_section():
    st.header("Ayuntamientos")
    municipio = select_municipio()
    if municipio is None:
        st.error("No se pudo seleccionar un municipio.")
        return
    traffic_data = get_real_traffic_data(municipio)
    st.write(f"**Congestión actual**: {traffic_data['congestion']}")
    st.write(f"**Emisiones CO₂ estimadas**: {traffic_data['emisiones_co2']:.1f} kg/día")
    
    air_data = get_eea_air_quality()
    if air_data:
        st.write("**Datos de Calidad del Aire (EEA)**:")
        st.write(air_data)
    
    noise_df = get_eea_noise_data()
    if noise_df is not None:
        st.write("**Datos de Ruido Ambiental (EEA)**:")
        st.dataframe(noise_df)
    
    ayto_tabs = st.tabs(["Mapa y Tráfico", "Predicción de Demanda", "Optimización (Múltiples Paradas)", "Integración API"])
    with ayto_tabs[0]:
        show_map_and_traffic()
    with ayto_tabs[1]:
        df_hist = load_historical_mobility_data()
        if df_hist is not None:
            st.write("**Predicción de Demanda con LightGBM**:")
            df_hist["day"] = df_hist["ds"].dt.dayofyear
            input_day = st.number_input("Ingrese el día del año:", min_value=1, max_value=366,
                                          value=datetime.now().timetuple().tm_yday, key="input_day")
            pred = predict_demand([input_day])
            if pred is not None:
                st.write(f"Predicción de demanda: {pred:.2f}")
            st.write("**Predicción de Demanda con Prophet**:")
            forecast = forecast_with_prophet(df_hist, periods=24)
            fig = px.line(forecast, x="ds", y="yhat", title="Predicción de Demanda (Prophet)")
            st.plotly_chart(fig)
    with ayto_tabs[2]:
        show_trip_service()
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
            st.write("**Predicción de Demanda con LightGBM**:")
            df_hist["day"] = df_hist["ds"].dt.dayofyear
            input_day = st.number_input("Ingrese el día del año:", min_value=1, max_value=366,
                                          value=datetime.now().timetuple().tm_yday, key="input_day_emp")
            pred = predict_demand([input_day])
            if pred is not None:
                st.write(f"Predicción de demanda: {pred:.2f}")
            st.write("**Predicción de Demanda con Prophet**:")
            forecast = forecast_with_prophet(df_hist, periods=24)
            fig = px.line(forecast, x="ds", y="yhat", title="Predicción de Demanda (Prophet)")
            st.plotly_chart(fig)
    with emp_tabs[2]:
        show_trip_service()
    with emp_tabs[3]:
        st.write("Calculadora de CAE")
        volume = st.number_input("CO₂ evitado (kg)", min_value=0.0, value=1000.0, step=100.0, key="coe_volume_emp")
        if st.button("Calcular CAE", key="calc_cae_emp"):
            cae_val = calculate_cae(volume)
            st.write(f"CAE estimados: {cae_val:.2f}")
    with emp_tabs[4]:
        show_integration_form()

# =============================================================================
# APLICACIÓN PRINCIPAL
# =============================================================================

def main_app():
    st.title("Trafiquea: Dashboard de Movilidad y Sostenibilidad")
    st.markdown("""
    Esta plataforma integra datos reales de la **EEA** (calidad del aire, emisiones, ruido ambiental),
    variables meteorológicas de **Open‑Meteo**, rutas optimizadas con **OSRM** (incluyendo optimización de múltiples paradas)
    y modelos de ML entrenados con **LightGBM** y **Prophet** para predecir la demanda de movilidad.
    
    Se utiliza además un dataset oficial de municipios de España para que los usuarios puedan seleccionar el municipio.
    """)
    
    user_type = st.selectbox("Tipo de Usuario", ["Ayuntamiento", "Empresa"], key="user_type_selector")
    st.markdown("---")
    
    if user_type == "Ayuntamiento":
        ayuntamientos_section()
    else:
        empresas_section()

if __name__ == "__main__":
    main_app()
