import streamlit as st
import folium
from streamlit_folium import st_folium
from geopy.geocoders import Nominatim
from datetime import datetime, timedelta
import requests
import numpy as np
import pandas as pd
import joblib

# =============================================================================
# Geocodificación y Reverse Geocoding
# =============================================================================

def geocode_address(address: str):
    geolocator = Nominatim(user_agent="trafiquea_dashboard")
    location = geolocator.geocode(address)
    if location:
        return (location.latitude, location.longitude), location.address
    return None, None

def reverse_geocode(lat: float, lon: float):
    geolocator = Nominatim(user_agent="trafiquea_reverse")
    location = geolocator.reverse((lat, lon), language="es")
    if location:
        return location.address
    return "Dirección no encontrada"

# =============================================================================
# Integración de APIs Reales: Clima y Rutas
# =============================================================================

def get_weather_open_meteo(lat: float, lon: float):
    url = "https://api.open-meteo.com/v1/forecast"
    params = {"latitude": lat, "longitude": lon, "current_weather": True, "timezone": "auto"}
    response = requests.get(url, params=params)
    if response.status_code == 200:
        data = response.json()
        current = data.get("current_weather", {})
        return {"temperatura": current.get("temperature"), "viento": current.get("windspeed")}
    return None

def get_route_osrm(origin_coords, destination_coords):
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

def get_real_traffic_data(ayuntamiento: str):
    """
    Función de ejemplo para obtener datos reales de tráfico y sostenibilidad 
    para un ayuntamiento. En producción se integraría una API pública de datos 
    abiertos (por ejemplo, del portal de Castilla-La Mancha).
    """
    # Ejemplo: URL real de la API de datos abiertos (a reemplazar)
    # url = f"https://datos.castillalamancha.es/api/trafico?municipio={ayuntamiento}"
    # response = requests.get(url)
    # data = response.json()
    # return data
    return {"congestion": np.random.choice(["Bajo", "Moderado", "Alto"]),
            "emisiones_co2": np.random.uniform(100, 300)}  # kg CO₂/día

# =============================================================================
# Simulación de Tráfico y Tiempos de Viaje
# =============================================================================

def simulate_zone_factor(full_address: str):
    # Ajuste de factor de congestión según la zona; se puede ampliar con datos reales
    address_lower = full_address.lower()
    if "talavera" in address_lower:
        return 1.3
    elif "toledo" in address_lower:
        return 1.25
    return 1.2

def assign_traffic_level(distance_km, start_time, weather, zone_factor):
    hour = start_time.hour
    base_factor = 1.5 if (7 <= hour <= 9 or 17 <= hour <= 19) else 1.2
    if weather and weather.get("viento") and weather["viento"] > 20:
        base_factor += 0.2
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

# =============================================================================
# Integración de Modelos de Machine Learning
# =============================================================================

@st.cache_resource
def load_demand_model():
    try:
        model = joblib.load("model_demand.pkl")
        return model
    except Exception:
        return None

def predict_demand(input_features):
    model = load_demand_model()
    if model:
        prediction = model.predict(np.array([input_features]))
        return prediction[0]
    return None

# =============================================================================
# Análisis de Datos Históricos (Últimos 6 meses)
# =============================================================================

def analyze_historical_data():
    dates = pd.date_range(end=datetime.now(), periods=180)
    demand = np.random.randint(80, 300, len(dates))
    df = pd.DataFrame({"Fecha": dates, "Demanda Estimada": demand})
    df.sort_values("Fecha", inplace=True)
    max_val = df["Demanda Estimada"].max()
    min_val = df["Demanda Estimada"].min()
    avg_val = df["Demanda Estimada"].mean()
    st.write(f"- Máximo de viajes/día: {max_val}")
    st.write(f"- Mínimo de viajes/día: {min_val}")
    st.write(f"- Promedio de viajes/día: {avg_val:.2f}")

# =============================================================================
# Calculadora de CAE para Empresas
# =============================================================================

def calculate_cae(volume_kg: float):
    factor_cae = 0.001  # 1 CAE por cada 1000 kg de CO₂ evitados
    return volume_kg * factor_cae

# =============================================================================
# Componentes Comunes: Mapas y Funcionalidades
# =============================================================================

def show_map_and_traffic():
    with st.form("traffic_form_final", clear_on_submit=True):
        col1, col2, col3 = st.columns(3)
        with col1:
            origin = st.text_input("Origen", "Plaza Mayor, Madrid, España")
        with col2:
            destination = st.text_input("Destino", "Puerta del Sol, Madrid, España")
        with col3:
            start_time = st.time_input("Hora de Inicio", datetime.now().time())
        submitted = st.form_submit_button("Calcular Ruta")
    
    if submitted:
        origin_coords, origin_full = geocode_address(origin)
        dest_coords, dest_full = geocode_address(destination)
        if not (origin_coords and dest_coords):
            st.error("Asegúrese de ingresar direcciones completas.")
            return
        now_dt = datetime.now()
        start_dt = datetime(now_dt.year, now_dt.month, now_dt.day, start_time.hour, start_time.minute)
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
        total_time_s = sum(seg["time_s"] for seg in segments)
        st.write(f"Distancia: {dist_km:.2f} km")
        st.write(f"Tiempo estimado: {arrival_time.strftime('%H:%M')} (~{int(total_time_s/60)} min)")
        if weather:
            st.write(f"Clima: {weather['temperatura']}°C, Viento: {weather['viento']} km/h")

def show_forecast():
    st.write("Seleccione el número de horas en el futuro:")
    hours = list(range(1,25))
    selected_hour = st.selectbox("Horas en el futuro:", hours)
    col1, col2 = st.columns(2)
    with col1:
        origin = st.text_input("Origen (Pronóstico)", "Plaza Mayor, Madrid, España")
    with col2:
        destination = st.text_input("Destino (Pronóstico)", "Puerta del Sol, Madrid, España")
    if st.button("Mostrar Pronóstico"):
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
            folium.PolyLine([(pt[1], pt[0]) for pt in seg["coords"]], color=seg["color"], weight=4).add_to(m_fore)
        folium.Marker(origin_coords, tooltip="Origen", icon=folium.Icon(color="green")).add_to(m_fore)
        folium.Marker(dest_coords, tooltip="Destino", icon=folium.Icon(color="red")).add_to(m_fore)
        st_folium(m_fore, width=700)
        dist_km = route_data["distance"] / 1000.0
        total_time_s = sum(seg["time_s"] for seg in segments)
        st.write(f"Distancia: {dist_km:.2f} km")
        st.write(f"Salida (Futura): {future_dt.strftime('%H:%M')}")
        st.write(f"Llegada estimada: {arrival_time.strftime('%H:%M')} (~{int(total_time_s/60)} min)")
        if weather:
            st.write(f"Clima: {weather['temperatura']}°C, Viento: {weather['viento']} km/h")

def show_simulation_advanced():
    offset = st.slider("Horas en el Futuro (simulación):", 1, 6, 2)
    result = simulate_traffic_forecast(offset)
    st.write(f"Nivel de Saturación: {result['saturacion']}")
    st.write(f"Tiempo Normal: {result['tiempo_normal']} min")
    st.write(f"Tiempo Ajustado: {result['tiempo_proyectado']} min")

# =============================================================================
# Módulo Especializado: Talavera y Comarca
# =============================================================================

def show_talavera_module():
    st.write("### Talavera y Comarca")
    with st.form("talavera_module_form", clear_on_submit=True):
        col1, col2, col3 = st.columns(3)
        with col1:
            origin = st.text_input("Origen (Talavera)", "Plaza de España, Talavera de la Reina, España")
        with col2:
            destination = st.text_input("Destino (Talavera)", "Centro, Talavera de la Reina, España")
        with col3:
            start_time = st.time_input("Hora de Inicio", datetime.now().time())
        submitted = st.form_submit_button("Calcular Ruta Talavera")
    if submitted:
        origin_coords, origin_full = geocode_address(origin)
        dest_coords, dest_full = geocode_address(destination)
        if not (origin_coords and dest_coords):
            st.error("No se pudo geocodificar las direcciones para Talavera.")
            return
        now_dt = datetime.now()
        start_dt = datetime(now_dt.year, now_dt.month, now_dt.day, start_time.hour, start_time.minute)
        weather = get_weather_open_meteo(origin_coords[0], origin_coords[1])
        zone_factor = simulate_zone_factor(origin_full)
        route_data = get_route_osrm(origin_coords, dest_coords)
        if not route_data:
            st.error("No se pudo obtener la ruta para Talavera.")
            return
        segments, arrival_time = simulate_route_segments(route_data["steps"], start_dt, weather, zone_factor)
        m_tal = folium.Map(location=[(origin_coords[0]+dest_coords[0])/2, (origin_coords[1]+dest_coords[1])/2],
                           zoom_start=13, tiles="OpenStreetMap")
        for seg in segments:
            folium.PolyLine([(pt[1], pt[0]) for pt in seg["coords"]], color=seg["color"], weight=4).add_to(m_tal)
        folium.Marker(origin_coords, tooltip="Origen", icon=folium.Icon(color="green")).add_to(m_tal)
        folium.Marker(dest_coords, tooltip="Destino", icon=folium.Icon(color="red")).add_to(m_tal)
        st_folium(m_tal, width=700)
        dist_km = route_data["distance"] / 1000.0
        total_time_s = sum(seg["time_s"] for seg in segments)
        st.write(f"Distancia Talavera: {dist_km:.2f} km")
        st.write(f"Llegada estimada: {arrival_time.strftime('%H:%M')} (~{int(total_time_s/60)} min)")
        if weather:
            st.write(f"Clima: {weather['temperatura']}°C, Viento: {weather['viento']} km/h")
    st.markdown("#### Datos Históricos (Talavera)")
    analyze_historical_data()
    st.markdown("#### Simulación Avanzada (Talavera)")
    show_simulation_advanced()

# =============================================================================
# Secciones por Perfil
# =============================================================================

def ayuntamientos_section():
    st.header("Ayuntamientos")
    st.write("Soluciones para la gestión urbana y la sostenibilidad.")
    # Selección de ayuntamiento real (ejemplo: datos de Castilla-La Mancha)
    ayto = st.selectbox("Seleccione un ayuntamiento", 
                        ["Talavera de la Reina", "Toledo Capital", "Illescas", "Seseña", "Ocaña"],
                        key="ayto_selector")
    real_data = get_real_traffic_data(ayto)
    st.write(f"Congestión actual: {real_data['congestion']}")
    st.write(f"Emisiones CO₂ estimadas: {real_data['emisiones_co2']:.1f} kg/día")
    
    ayto_tabs = st.tabs(["Mapa y Tráfico", "Predicción", "Talavera y Comarca", "Simulación", "Integración API"])
    with ayto_tabs[0]:
        show_map_and_traffic()
    with ayto_tabs[1]:
        show_forecast()
    with ayto_tabs[2]:
        show_talavera_module()
    with ayto_tabs[3]:
        show_simulation_advanced()
    with ayto_tabs[4]:
        show_integration_form()

def empresas_section():
    st.header("Empresas")
    st.write("Optimice rutas, reduzca costes y aproveche créditos de emisiones (CAE).")
    emp_tabs = st.tabs(["Mapa y Tráfico", "Predicción", "Simulación", "Calculadora CAE", "Integración API"])
    with emp_tabs[0]:
        show_map_and_traffic()
    with emp_tabs[1]:
        show_forecast()
    with emp_tabs[2]:
        show_simulation_advanced()
    with emp_tabs[3]:
        st.write("Calculadora de CAE")
        volume = st.number_input("CO₂ evitado (kg)", min_value=0.0, value=1000.0, step=100.0, key="coe_volume")
        if st.button("Calcular CAE", key="calc_cae"):
            cae = calculate_cae(volume)
            st.write(f"CAE estimados: {cae:.2f}")
    with emp_tabs[4]:
        show_integration_form()

def particulares_section():
    st.header("Particulares")
    st.write("Optimice sus desplazamientos diarios.")
    part_tabs = st.tabs(["Mapa y Tráfico", "Predicción", "Opciones Eco"])
    with part_tabs[0]:
        show_map_and_traffic()
    with part_tabs[1]:
        show_forecast()
    with part_tabs[2]:
        st.write("Opciones Eco: Rutas optimizadas para bicicleta y transporte público (simulado).")
        st.write("En una versión real se integrarían APIs en modo cycling/transit.")

# =============================================================================
# Aplicación Principal
# =============================================================================

def main_app():
    st.title("Trafiquea")
    st.write("Bienvenido. Nuestra plataforma ofrece soluciones integrales de movilidad y sostenibilidad adaptadas a Ayuntamientos, Empresas y Particulares.")
    
    # Selección del tipo de usuario
    user_type = st.selectbox("Tipo de Usuario", ["Ayuntamiento", "Empresa", "Particular"], key="user_type")
    st.write("---")
    
    if user_type == "Ayuntamiento":
        ayuntamientos_section()
    elif user_type == "Empresa":
        empresas_section()
    else:
        particulares_section()

if __name__ == "__main__":
    main_app()
