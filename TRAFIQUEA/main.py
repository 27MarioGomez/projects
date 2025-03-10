"""
Trafiquea: Plataforma Integral para Transporte y Rutas en Tiempo Real

Este dashboard está dividido en tres pestañas principales (Ayuntamientos, Empresas, Particulares),
cada una con un mensaje de bienvenida y un conjunto de funcionalidades específicas.
Se usan APIs públicas (OSRM, Open-Meteo, Nominatim) y datos simulados para ilustrar la lógica.

La estructura:
  1. Selección de Usuario (Ayuntamiento, Empresa o Particular).
  2. Para cada usuario, se muestran tabs con mapas y datos relevantes:
     - Tráfico en tiempo real (mapas OSRM).
     - Pronósticos (usando una función de simulación).
     - Análisis histórico de datos (simulados de los últimos 6 meses).
     - Simulación avanzada de escenarios.
     - Integración API (formulario).

Si se quisiera información actualizada de ayuntamientos reales o de la comarca de Talavera,
se integrarían las correspondientes APIs públicas (o privadas gratuitas) para datos de tráfico,
sostenibilidad, etc. En esta versión, se muestra la arquitectura base para un proyecto de portafolio.
"""

import streamlit as st
import folium
from streamlit_folium import st_folium
from geopy.geocoders import Nominatim
from datetime import datetime, timedelta
import requests
import numpy as np
import pandas as pd

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
    location = geolocator.reverse((lat, lon), language="en")
    if location:
        return location.address
    return "Dirección no encontrada"

# =============================================================================
# APIs: OSRM, Open-Meteo, etc.
# =============================================================================

def simulate_zone_factor(full_address: str):
    address_lower = full_address.lower()
    if "talavera" in address_lower:
        return 1.3
    elif "toledo" in address_lower:
        return 1.25
    return 1.2

def get_weather_open_meteo(lat: float, lon: float):
    url = "https://api.open-meteo.com/v1/forecast"
    params = {"latitude": lat, "longitude": lon, "current_weather": True, "timezone": "auto"}
    response = requests.get(url, params=params)
    if response.status_code == 200:
        data = response.json()
        current = data.get("current_weather", {})
        return {"temperature": current.get("temperature"), "windspeed": current.get("windspeed")}
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

# =============================================================================
# Simulación de Tráfico y Tiempos de Viaje
# =============================================================================

def assign_traffic_level(distance_km, start_time, weather, zone_factor):
    hour = start_time.hour
    if 7 <= hour <= 9 or 17 <= hour <= 19:
        base_factor = 1.5
    else:
        base_factor = 1.2

    if weather and weather.get("windspeed") and weather["windspeed"] > 20:
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
# Análisis de Datos Históricos (Últimos Meses)
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
# Formulario de Integración vía API
# =============================================================================

def show_integration_form():
    with st.form("integration_form_final", clear_on_submit=True):
        nombre = st.text_input("Nombre", key="api_nombre_key")
        apellidos = st.text_input("Apellidos", key="api_apellidos_key")
        institucion = st.text_input("Institución o Empresa", key="api_institucion_key")
        mensaje = st.text_area("Mensaje (opcional)", key="api_mensaje_key")
        submitted = st.form_submit_button("Enviar Solicitud")
        if submitted:
            st.success("Solicitud enviada correctamente. Nos pondremos en contacto contigo.")

# =============================================================================
# Funciones Comunes
# =============================================================================

def show_map_and_traffic(key_prefix=""):
    st.markdown("Ingrese origen, destino y hora de inicio para ver el tráfico en tiempo real.")
    form_key = f"traffic_form_{key_prefix}"
    with st.form(form_key, clear_on_submit=True):
        col1, col2, col3 = st.columns(3)
        with col1:
            origin = st.text_input("Origen", "Plaza Mayor, Madrid, España", key=f"origin_{key_prefix}")
        with col2:
            destination = st.text_input("Destino", "Puerta del Sol, Madrid, España", key=f"dest_{key_prefix}")
        with col3:
            time_val = st.time_input("Hora de Inicio", datetime.now().time(), key=f"time_{key_prefix}")
        submitted = st.form_submit_button("Calcular Ruta")
    
    if submitted:
        origin_coords, origin_full = geocode_address(origin)
        dest_coords, dest_full = geocode_address(destination)
        if not (origin_coords and dest_coords):
            st.error("No se pudo geocodificar. Use direcciones completas.")
            return
        now_dt = datetime.now()
        start_dt = datetime(now_dt.year, now_dt.month, now_dt.day, time_val.hour, time_val.minute)
        weather = get_weather_open_meteo(origin_coords[0], origin_coords[1])
        zone_factor = simulate_zone_factor(origin_full)
        route_data = get_route_osrm(origin_coords, dest_coords)
        if not route_data:
            st.error("No se pudo obtener la ruta con OSRM.")
            return
        segments, arrival_time = simulate_route_segments(route_data["steps"], start_dt, weather, zone_factor)
        m_map = folium.Map(location=[(origin_coords[0] + dest_coords[0])/2,
                                     (origin_coords[1] + dest_coords[1])/2],
                           zoom_start=12, tiles="OpenStreetMap")
        for seg in segments:
            folium.PolyLine([(pt[1], pt[0]) for pt in seg["coords"]], color=seg["color"], weight=4).add_to(m_map)
        folium.Marker(origin_coords, tooltip="Origen", icon=folium.Icon(color="green")).add_to(m_map)
        folium.Marker(dest_coords, tooltip="Destino", icon=folium.Icon(color="red")).add_to(m_map)
        st_folium(m_map, width=700)
        dist_km = route_data["distance"]/1000.0
        total_time_s = sum(seg["time_s"] for seg in segments)
        st.write(f"**Distancia Aproximada:** {dist_km:.2f} km")
        st.write(f"**Hora de Llegada Estimada:** {arrival_time.strftime('%H:%M')} (~{int(total_time_s/60)} min)")
        if weather:
            st.write(f"**Clima en Origen:** {weather['temperature']}°C, viento: {weather['windspeed']} km/h")

def show_forecast(key_prefix=""):
    st.markdown("Seleccione cuántas horas en el futuro para ver la situación del tráfico.")
    hour_options = list(range(1,25))
    selected_hour = st.selectbox("Horas en el futuro:", hour_options, key=f"forecast_hours_{key_prefix}")
    col1, col2 = st.columns(2)
    with col1:
        origin = st.text_input("Origen (Pronóstico)", "Plaza Mayor, Madrid, España", key=f"forecast_origin_{key_prefix}")
    with col2:
        destination = st.text_input("Destino (Pronóstico)", "Puerta del Sol, Madrid, España", key=f"forecast_destination_{key_prefix}")
    if st.button("Mostrar Pronóstico", key=f"btn_forecast_{key_prefix}"):
        origin_coords, origin_full = geocode_address(origin)
        dest_coords, dest_full = geocode_address(destination)
        if not (origin_coords and dest_coords):
            st.error("No se pudo geocodificar.")
            return
        future_dt = datetime.now() + timedelta(hours=selected_hour)
        weather = get_weather_open_meteo(origin_coords[0], origin_coords[1])
        zone_factor = simulate_zone_factor(origin_full)
        route_data = get_route_osrm(origin_coords, dest_coords)
        if not route_data:
            st.error("No se pudo obtener la ruta.")
            return
        segments, arrival_time = simulate_route_segments(route_data["steps"], future_dt, weather, zone_factor)
        m_forecast = folium.Map(location=[(origin_coords[0]+dest_coords[0])/2,
                                          (origin_coords[1]+dest_coords[1])/2],
                                zoom_start=12, tiles="OpenStreetMap")
        for seg in segments:
            folium.PolyLine([(pt[1], pt[0]) for pt in seg["coords"]], color=seg["color"], weight=4).add_to(m_forecast)
        folium.Marker(origin_coords, tooltip="Origen", icon=folium.Icon(color="green")).add_to(m_forecast)
        folium.Marker(dest_coords, tooltip="Destino", icon=folium.Icon(color="red")).add_to(m_forecast)
        st_folium(m_forecast, width=700)
        dist_km = route_data["distance"]/1000.0
        total_time_s = sum(seg["time_s"] for seg in segments)
        st.write(f"**Distancia Total:** {dist_km:.2f} km")
        st.write(f"**Hora de Salida (Futura):** {future_dt.strftime('%H:%M')}")
        st.write(f"**Hora de Llegada Estimada:** {arrival_time.strftime('%H:%M')} (~{int(total_time_s/60)} min)")
        if weather:
            st.write(f"**Clima Estimado:** {weather['temperature']}°C, viento: {weather['windspeed']} km/h")

def show_simulation_advanced(key_prefix=""):
    offset = st.slider("Horas en el Futuro (simulación):", 1, 6, 2, key=f"sim_slider_{key_prefix}")
    forecast_res = simulate_traffic_forecast(offset)
    st.write(f"**Nivel de Saturación:** {forecast_res['saturacion']}")
    st.write(f"**Tiempo de Viaje Normal:** {forecast_res['tiempo_normal']} min")
    st.write(f"**Tiempo de Viaje Ajustado:** {forecast_res['tiempo_proyectado']} min")

# ============================================================================
# Tab Especializado: Talavera
# ============================================================================

def show_talavera_module(key_prefix=""):
    st.markdown("### Talavera de la Reina")
    with st.form(f"talavera_form_{key_prefix}", clear_on_submit=True):
        col1, col2, col3 = st.columns(3)
        with col1:
            origin = st.text_input("Origen", "Plaza de España, Talavera de la Reina, España", key=f"talavera_origin_{key_prefix}")
        with col2:
            destination = st.text_input("Destino", "Ayuntamiento, Talavera de la Reina, España", key=f"talavera_destination_{key_prefix}")
        with col3:
            start_time = st.time_input("Hora de Inicio", datetime.now().time(), key=f"talavera_time_{key_prefix}")
        submitted = st.form_submit_button("Calcular Ruta (Talavera)")
    
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
            st.error("No se pudo obtener la ruta con OSRM para Talavera.")
            return
        segments, arrival_time = simulate_route_segments(route_data["steps"], start_dt, weather, zone_factor)
        m_tal = folium.Map(location=[(origin_coords[0]+dest_coords[0])/2, (origin_coords[1]+dest_coords[1])/2],
                           zoom_start=13, tiles="OpenStreetMap")
        for seg in segments:
            folium.PolyLine([(pt[1], pt[0]) for pt in seg["coords"]], color=seg["color"], weight=4).add_to(m_tal)
        folium.Marker(origin_coords, tooltip="Origen", icon=folium.Icon(color="green")).add_to(m_tal)
        folium.Marker(dest_coords, tooltip="Destino", icon=folium.Icon(color="red")).add_to(m_tal)
        st_folium(m_tal, width=700)
        dist_km = route_data["distance"]/1000.0
        total_time_s = sum(seg["time_s"] for seg in segments)
        st.write(f"**Distancia Talavera:** {dist_km:.2f} km")
        st.write(f"**Hora de Llegada Estimada:** {arrival_time.strftime('%H:%M')} (~{int(total_time_s/60)} min)")
        if weather:
            st.write(f"**Clima (Origen - Talavera):** {weather['temperature']}°C, viento: {weather['windspeed']} km/h")

    st.markdown("#### Análisis de Datos Históricos (Talavera)")
    analyze_historical_data()
    st.markdown("#### Simulación Avanzada de Escenarios (Talavera)")
    show_simulation_advanced(key_prefix=f"talavera_{key_prefix}")

# ============================================================================
# Main
# ============================================================================

def main_app():
    st.title("Trafiquea: Soluciones de Movilidad y Sostenibilidad")
    st.markdown("Bienvenido/a. Elija su tipo de usuario para ver funcionalidades adaptadas:")

    user_type = st.selectbox("Tipo de Usuario", ["Ayuntamiento", "Empresa", "Particular"], key="user_type_box")

    st.markdown("---")

    if user_type == "Ayuntamiento":
        st.markdown("## Ayuntamientos: Gestión de Movilidad")
        # Mapa inicial de la zona (p.ej. Toledo) - si se desea
        map_ayto = folium.Map(location=[39.8628, -4.0273], zoom_start=8, tiles="OpenStreetMap")
        st_folium(map_ayto, width=700)

        ayto_tabs = st.tabs(["Tráfico en Tiempo Real", "Predicción de Tráfico", "Talavera Especializado", "Simulación", "Integración API"])
        with ayto_tabs[0]:
            show_map_and_traffic(key_prefix="ayto_trafico")
        with ayto_tabs[1]:
            show_forecast(key_prefix="ayto_forecast")
        with ayto_tabs[2]:
            show_talavera_module(key_prefix="ayto_talavera")
        with ayto_tabs[3]:
            show_simulation_advanced(key_prefix="ayto_simul")
        with ayto_tabs[4]:
            show_integration_form()

    elif user_type == "Empresa":
        st.markdown("## Empresas: Optimización y Rentabilidad")
        map_emp = folium.Map(location=[39.8628, -4.0273], zoom_start=8, tiles="OpenStreetMap")
        st_folium(map_emp, width=700)

        emp_tabs = st.tabs(["Tráfico en Tiempo Real", "Pronósticos", "Simulación Avanzada", "Calculadora CAE", "Integración API"])
        with emp_tabs[0]:
            show_map_and_traffic(key_prefix="emp_trafico")
        with emp_tabs[1]:
            show_forecast(key_prefix="emp_forecast")
        with emp_tabs[2]:
            show_simulation_advanced(key_prefix="emp_simul")
        with emp_tabs[3]:
            st.markdown("Introduzca la cantidad de CO₂ evitado para estimar CAE.")
            co2_avoided = st.number_input("CO₂ evitado (kg)", min_value=0.0, value=1000.0, step=100.0, key="coe_avoided")
            if st.button("Calcular CAE", key="calc_cae"):
                factor_cae = 0.001
                cae_val = co2_avoided * factor_cae
                st.write(f"Se generarían ~{cae_val:.2f} CAE potenciales.")
        with emp_tabs[4]:
            show_integration_form()

    else:
        st.markdown("## Particulares: Rutas Diarias y Opciones Eco")
        map_part = folium.Map(location=[39.8628, -4.0273], zoom_start=8, tiles="OpenStreetMap")
        st_folium(map_part, width=700)

        part_tabs = st.tabs(["Tráfico en Tiempo Real", "Pronósticos de Ruta", "Opciones Eco"])
        with part_tabs[0]:
            show_map_and_traffic(key_prefix="part_trafico")
        with part_tabs[1]:
            show_forecast(key_prefix="part_forecast")
        with part_tabs[2]:
            st.markdown("Bicicleta y Transporte Público (simulado). Se mostraría un mapa y tiempos de viaje aproximados.")
            st.markdown("En una versión real, se integrarían APIs OSRM en modo cycling/transit.")

if __name__ == "__main__":
    main_app()
