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

# =============================================================================
# Geocodificación y Reverse Geocoding (Nominatim, sin secrets)
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
# Integración de APIs Reales (Open-Meteo, OSRM, EEA) sin claves
# =============================================================================

def get_weather_open_meteo(lat: float, lon: float):
    """
    Consulta a la API gratuita de Open-Meteo, incluyendo variables extra
    (temperatura, viento, precipitaciones, nubosidad) para ilustrar
    el recuento fraccionario de llamadas API.
    """
    url = "https://api.open-meteo.com/v1/forecast"
    # Se incluyen hasta 4 variables extra: temperature_2m, windspeed_10m, precipitation, cloudcover
    # Notar que se asume un uso moderado para no saturar la API gratuita.
    params = {
        "latitude": lat,
        "longitude": lon,
        "hourly": "temperature_2m,windspeed_10m,precipitation,cloudcover",
        "current_weather": True,
        "timezone": "auto"
    }
    response = requests.get(url, params=params)
    if response.status_code == 200:
        data = response.json()
        current = data.get("current_weather", {})
        # Extraemos variables de la hora actual
        return {
            "temperatura": current.get("temperature"),
            "viento": current.get("windspeed"),
            "precipitacion": data["hourly"]["precipitation"][0] if "hourly" in data else 0.0,
            "nubosidad": data["hourly"]["cloudcover"][0] if "hourly" in data else 0.0
        }
    return None

def get_route_osrm_multiple(coords_list, profile="driving"):
    """
    Utiliza el Trip Service de OSRM para reordenar múltiples paradas (TSP básico).
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
    Ejemplo de descarga de datos XML de calidad del aire (EEA).
    Se parsea con xmltodict. Cacheado 1h.
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
    CSV de movilidad de la EEA (por ejemplo, emisiones de vehículos nuevos).
    """
    csv_url = "https://www.eea.europa.eu/system/files/documents/CO2_emissions_passenger_cars.csv"
    try:
        df = pd.read_csv(csv_url, sep=";", encoding="latin1")
        return df
    except Exception as e:
        st.error(f"Error al cargar CSV de movilidad EEA: {e}")
        return None

@st.cache_data(ttl=3600)
def get_eea_noise_data():
    """
    Datos de ruido ambiental (simulado) de la EEA.
    Podría ser un CSV o un endpoint WFS. Aquí se ejemplifica un CSV.
    """
    noise_url = "https://www.eea.europa.eu/data-and-maps/data/noise/"
    # Asumimos que se descarga un CSV; para el ejemplo no es real:
    # Se retorna un DataFrame simulado
    data = {
        "city": ["Madrid", "Barcelona", "Valencia", "Toledo", "Sevilla"],
        "noise_index": [np.random.uniform(55, 75) for _ in range(5)]
    }
    df = pd.DataFrame(data)
    return df

def get_real_traffic_data(ayuntamiento: str):
    """
    Placeholder para datos de tráfico municipal. Se simula o se conecta
    con un endpoint abierto sin token.
    """
    return {
        "congestion": np.random.choice(["Bajo", "Moderado", "Alto"]),
        "emisiones_co2": np.random.uniform(100, 300)
    }

# =============================================================================
# Simulación de Tráfico y ML
# =============================================================================

def simulate_zone_factor(full_address: str):
    address_lower = full_address.lower()
    if "talavera" in address_lower:
        return 1.3
    elif "toledo" in address_lower:
        return 1.25
    return 1.2

def assign_traffic_level(distance_km, start_time, weather, zone_factor):
    hour = start_time.hour
    base_factor = 1.5 if (7 <= hour <= 9 or 17 <= hour <= 19) else 1.2
    # Ajustamos un poco más con precipitacion y nubosidad
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
    Carga un modelo de demanda (por ejemplo, LSTM, XGBoost o Prophet).
    """
    try:
        model = joblib.load("model_demand.pkl")
        return model
    except Exception as e:
        st.error(f"Error al cargar modelo: {e}")
        return None

def predict_demand(input_features):
    model = load_demand_model()
    if model:
        return model.predict(np.array([input_features]))[0]
    return None

# =============================================================================
# Ejemplo de ML con Prophet
# =============================================================================

def forecast_with_prophet(data: pd.DataFrame, periods: int = 24):
    model = Prophet()
    model.fit(data)
    future = model.make_future_dataframe(periods=periods, freq='H')
    forecast = model.predict(future)
    return forecast

# =============================================================================
# Análisis de Datos Históricos y Visualización
# =============================================================================

def load_historical_mobility_data():
    df = get_eea_mobility_data()
    if df is not None:
        # Ajustar para Prophet: ds (fecha), y (valor)
        df.rename(columns={df.columns[0]: "ds", df.columns[1]: "y"}, inplace=True)
        return df
    return None

def analyze_historical_data():
    df = load_historical_mobility_data()
    if df is None:
        st.error("No se pudieron cargar datos históricos de movilidad (EEA).")
        return
    st.write("Resumen de datos históricos de movilidad (EEA):")
    max_val = df["y"].max()
    min_val = df["y"].min()
    avg_val = df["y"].mean()
    st.write(f"- Máximo: {max_val}")
    st.write(f"- Mínimo: {min_val}")
    st.write(f"- Promedio: {avg_val:.2f}")
    fig = px.line(df, x="ds", y="y", title="Evolución de Movilidad/Emisiones (EEA)")
    st.plotly_chart(fig)

# =============================================================================
# Calculadora de CAE
# =============================================================================

def calculate_cae(volume_kg: float):
    factor_cae = 0.001
    return volume_kg * factor_cae

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
# Funciones Comunes de Interfaz
# =============================================================================

def show_map_and_traffic():
    st.subheader("Cálculo de Ruta (OSRM) con Origen y Destino")
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
            st.error("No se pudo geocodificar las direcciones.")
            return
        start_dt = datetime(datetime.now().year, datetime.now().month, datetime.now().day,
                            start_time.hour, start_time.minute)
        weather = get_weather_open_meteo(origin_coords[0], origin_coords[1])
        zone_factor = simulate_zone_factor(origin_full)

        # Llamada simple a OSRM con 2 puntos
        route_data = get_route_osrm((origin_coords[0], origin_coords[1]), (dest_coords[0], dest_coords[1]))
        if not route_data:
            st.error("No se pudo obtener la ruta con OSRM.")
            return
        segments, arrival_time = simulate_route_segments(route_data["steps"], start_dt, weather, zone_factor)
        # Mapa Folium
        m_map = folium.Map(location=[(origin_coords[0]+dest_coords[0])/2,
                                     (origin_coords[1]+dest_coords[1])/2],
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
            st.write(f"Clima (Open-Meteo): {weather['temperatura']}°C, Viento: {weather['viento']} km/h, "
                     f"Precipitación: {weather['precipitacion']} mm, Nubosidad: {weather['nubosidad']}%")

def show_forecast():
    st.subheader("Predicción de Tráfico Futuro (Simulación)")
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
            st.error("No se pudo geocodificar.")
            return
        # Llamada simulada de saturación
        forecast_data = simulate_traffic_forecast(selected_hour)
        st.write(f"Nivel de saturación simulado: {forecast_data['saturacion']}")
        st.write(f"Tiempo normal: {forecast_data['tiempo_normal']} min")
        st.write(f"Tiempo proyectado: {forecast_data['tiempo_proyectado']} min")

def show_simulation_advanced(key_prefix="default"):
    st.subheader("Simulación Avanzada con Múltiples Paradas (OSRM Trip Service)")
    st.write("Agregue múltiples direcciones (una por línea).")
    multi_input = st.text_area("Direcciones:", "Plaza Mayor, Madrid\nPuerta del Sol, Madrid\nAtocha, Madrid", key=f"multi_{key_prefix}")
    if st.button("Optimizar Ruta con Múltiples Paradas", key=f"btn_trip_{key_prefix}"):
        # Parsear direcciones
        addresses = multi_input.strip().split("\n")
        coords_list = []
        for addr in addresses:
            geoc, full_addr = geocode_address(addr)
            if not geoc:
                st.error(f"No se pudo geocodificar: {addr}")
                return
            coords_list.append((geoc[0], geoc[1]))  # (lat, lon)

        # Llamada al Trip Service
        trip_data = get_route_osrm_multiple(coords_list, profile="driving")
        if not trip_data:
            st.error("No se pudo obtener la ruta con paradas (Trip Service).")
            return
        geometry = trip_data["geometry"]
        distance_m = trip_data["distance"]
        duration_s = trip_data["duration"]
        # Mapa
        m_map = folium.Map(location=[coords_list[0][0], coords_list[0][1]], zoom_start=12, tiles="OpenStreetMap")
        folium.PolyLine([(pt[1], pt[0]) for pt in geometry["coordinates"]], color="#2ecc71", weight=4).add_to(m_map)
        for idx, (lat, lon) in enumerate(coords_list):
            folium.Marker((lat, lon), tooltip=f"Parada {idx+1}", icon=folium.Icon(color="blue")).add_to(m_map)
        st_folium(m_map, width=700)
        st.write(f"Distancia total: {distance_m/1000:.2f} km")
        st.write(f"Duración total: ~{int(duration_s/60)} min")

# =============================================================================
# Módulo Especializado: Talavera y Comarca
# =============================================================================

def show_talavera_module():
    st.write("### Talavera y Comarca")
    st.write("Este módulo ejemplifica la aplicación para un área concreta (ayuntamiento).")
    with st.form("talavera_module_form", clear_on_submit=True):
        col1, col2, col3 = st.columns(3)
        with col1:
            origin = st.text_input("Origen (Talavera)", "Plaza de España, Talavera de la Reina, España", key="talavera_origin")
        with col2:
            destination = st.text_input("Destino (Talavera)", "Centro, Talavera de la Reina, España", key="talavera_destination")
        with col3:
            start_time = st.time_input("Hora de Inicio", datetime.now().time(), key="talavera_time")
        submitted = st.form_submit_button("Calcular Ruta Talavera")
    if submitted:
        origin_coords, origin_full = geocode_address(origin)
        dest_coords, dest_full = geocode_address(destination)
        if not (origin_coords and dest_coords):
            st.error("No se pudo geocodificar las direcciones para Talavera.")
            return
        start_dt = datetime(datetime.now().year, datetime.now().month, datetime.now().day,
                            start_time.hour, start_time.minute)
        weather = get_weather_open_meteo(origin_coords[0], origin_coords[1])
        zone_factor = simulate_zone_factor(origin_full)
        route_data = get_route_osrm((origin_coords[0], origin_coords[1]), (dest_coords[0], dest_coords[1]))
        if not route_data:
            st.error("No se pudo obtener la ruta para Talavera.")
            return
        segments, arrival_time = simulate_route_segments(route_data["steps"], start_dt, weather, zone_factor)
        m_tal = folium.Map(location=[(origin_coords[0]+dest_coords[0])/2,
                                     (origin_coords[1]+dest_coords[1])/2],
                           zoom_start=13, tiles="OpenStreetMap")
        for seg in segments:
            folium.PolyLine([(pt[1], pt[0]) for pt in seg["coords"]],
                            color=seg["color"], weight=4).add_to(m_tal)
        folium.Marker(origin_coords, tooltip="Origen", icon=folium.Icon(color="green")).add_to(m_tal)
        folium.Marker(dest_coords, tooltip="Destino", icon=folium.Icon(color="red")).add_to(m_tal)
        st_folium(m_tal, width=700)
        dist_km = route_data["distance"] / 1000.0
        total_time_s = sum(seg["time_s"] for seg in segments)
        st.write(f"Distancia Talavera: {dist_km:.2f} km")
        st.write(f"Llegada estimada: {arrival_time.strftime('%H:%M')} (~{int(total_time_s/60)} min)")
        if weather:
            st.write(f"Clima (Open-Meteo): {weather['temperatura']}°C, Viento: {weather['viento']} km/h, "
                     f"Precipitación: {weather['precipitacion']} mm, Nubosidad: {weather['nubosidad']}%")
    st.markdown("#### Datos Históricos (Talavera)")
    analyze_historical_data()
    st.markdown("#### Simulación Avanzada (Talavera)")
    show_simulation_advanced(key_prefix="talavera")

# =============================================================================
# Secciones por Perfil: Ayuntamientos y Empresas
# =============================================================================

def ayuntamientos_section():
    st.header("Ayuntamientos")
    ayto = st.selectbox("Seleccione un ayuntamiento",
                        ["Talavera de la Reina", "Toledo Capital", "Illescas", "Seseña", "Ocaña"],
                        key="ayto_selector")
    real_data = get_real_traffic_data(ayto)
    st.write(f"**Congestión actual**: {real_data['congestion']}")
    st.write(f"**Emisiones CO₂ estimadas**: {real_data['emisiones_co2']:.1f} kg/día")
    
    air_data = get_eea_air_quality()
    if air_data:
        st.write("Datos de Calidad del Aire (EEA) en formato XML parseado:")
        st.write(air_data)
    
    noise_data = get_eea_noise_data()
    if noise_data is not None:
        st.write("Datos de Ruido Ambiental (EEA):")
        st.dataframe(noise_data)
    
    ayto_tabs = st.tabs(["Mapa y Tráfico", "Predicción", "Talavera y Comarca", "Simulación Múltiples Paradas", "Integración API"])
    with ayto_tabs[0]:
        show_map_and_traffic()
    with ayto_tabs[1]:
        show_forecast()
    with ayto_tabs[2]:
        show_talavera_module()
    with ayto_tabs[3]:
        show_simulation_advanced(key_prefix="ayto_multi")
    with ayto_tabs[4]:
        show_integration_form()

def empresas_section():
    st.header("Empresas")
    st.write("Optimización de rutas, predicción de demanda y cálculo de CAE.")
    emp_tabs = st.tabs(["Mapa y Tráfico", "Predicción", "Simulación Múltiples Paradas", "Calculadora CAE", "Integración API"])
    with emp_tabs[0]:
        show_map_and_traffic()
    with emp_tabs[1]:
        show_forecast()
    with emp_tabs[2]:
        show_simulation_advanced(key_prefix="emp_multi")
    with emp_tabs[3]:
        st.write("Calculadora de CAE")
        volume = st.number_input("CO₂ evitado (kg)", min_value=0.0, value=1000.0, step=100.0, key="coe_volume")
        if st.button("Calcular CAE", key="calc_cae"):
            cae_val = calculate_cae(volume)
            st.write(f"CAE estimados: {cae_val:.2f}")
    with emp_tabs[4]:
        show_integration_form()

# =============================================================================
# Aplicación Principal
# =============================================================================

def main_app():
    st.title("Trafiquea: Versión Avanzada en Producción")
    st.markdown("""
    **Conclusiones y Siguientes Iteraciones**:
    - Se integran más variables de **Open-Meteo** (temperatura, viento, precipitaciones, nubosidad) optimizando las llamadas API.
    - Se aprovechan datos de la **EEA** sobre calidad del aire, ruido ambiental y movilidad (CSV).
    - Se añade la opción de **múltiples paradas** con el Trip Service de OSRM (TSP básico).
    - Se entrena y carga un modelo ML (ej. Prophet) para predicción de demanda.
    - Próximas mejoras:
      1. Incorporar VRP más complejo (capacidades, múltiples vehículos).
      2. Usar datos de ruido nocturno para evitar rutas ruidosas.
      3. Ajustar recuento fraccionario de llamadas a Open-Meteo si se agregan más variables o más días.
      4. Profundizar en la integración de datos de emisiones de GEI y reparto modal de transporte (EEA).
    """)
    
    user_type = st.selectbox("Tipo de Usuario", ["Ayuntamiento", "Empresa"], key="user_type_selector")
    st.write("---")
    
    if user_type == "Ayuntamiento":
        ayuntamientos_section()
    else:
        empresas_section()

if __name__ == "__main__":
    main_app()
