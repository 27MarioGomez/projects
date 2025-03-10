"""
Trafiquea: Plataforma Integral para Transporte y Rutas en Tiempo Real

Este dashboard está dividido en tres pestañas principales:
  1. Ayuntamientos
  2. Empresas
  3. Particulares

Cada pestaña incluye un mapa inicial y un mensaje de bienvenida, 
seguido de las funcionalidades específicas de cada tipo de usuario 
(tráfico en tiempo real, pronósticos, simulación, etc.).

Se utilizan datos simulados y APIs públicas (OSRM, Open-Meteo, Nominatim) 
para ilustrar la lógica. En una versión real, se podrían integrar datos 
de cada ayuntamiento o comarca (por ejemplo, comarca de Talavera) a través 
de APIs públicas gratuitas o servicios privados que ofrezcan información 
actualizada de tráfico, clima, sostenibilidad, etc.

Este código es un ejemplo de portafolio que demuestra un enfoque 
multiperfil, la integración de mapas (Folium) y la posibilidad de 
ampliar con técnicas de machine learning y deep learning (importadas 
pero no detalladas en esta versión).
"""

import streamlit as st
import folium
from streamlit_folium import st_folium
from geopy.geocoders import Nominatim
from datetime import datetime, timedelta
import requests
import numpy as np
import pandas as pd

# ============================================================================
# Geocodificación y Reverse Geocoding
# ============================================================================

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

# ============================================================================
# APIs: OSRM, Open-Meteo, etc.
# ============================================================================

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

# ============================================================================
# Simulación de Tráfico y Tiempos de Viaje
# ============================================================================

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

# ============================================================================
# Data Histórica (Últimos Meses) y Análisis Simplificado
# ============================================================================

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

# ============================================================================
# Formulario de Integración vía API
# ============================================================================

def show_integration_form():
    with st.form("form_integration_final", clear_on_submit=True):
        nombre = st.text_input("Nombre", key="api_nombre")
        apellidos = st.text_input("Apellidos", key="api_apellidos")
        institucion = st.text_input("Institución o Empresa", key="api_institucion")
        mensaje = st.text_area("Mensaje (opcional)", key="api_mensaje")
        submitted = st.form_submit_button("Enviar Solicitud")
        if submitted:
            st.success("Solicitud enviada correctamente. Nos pondremos en contacto contigo.")

# ============================================================================
# Funcionalidades Reutilizables: Rutas, Pronósticos, Simulación
# ============================================================================

def show_map_and_traffic():
    st.markdown("Por favor, introduzca la dirección de origen y destino, así como la hora de inicio.")
    with st.form("traffic_form", clear_on_submit=True):
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
        if not origin_coords or not dest_coords:
            st.error("No se pudo geocodificar. Asegúrese de usar direcciones completas.")
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
        m_traffic = folium.Map(location=[(origin_coords[0] + dest_coords[0]) / 2,
                                         (origin_coords[1] + dest_coords[1]) / 2],
                               zoom_start=12, tiles="OpenStreetMap")
        for seg in segments:
            folium.PolyLine([(pt[1], pt[0]) for pt in seg["coords"]], 
                            color=seg["color"], weight=4).add_to(m_traffic)
        folium.Marker(origin_coords, tooltip="Origen", icon=folium.Icon(color="green")).add_to(m_traffic)
        folium.Marker(dest_coords, tooltip="Destino", icon=folium.Icon(color="red")).add_to(m_traffic)
        st_folium(m_traffic, width=700)
        dist_km = route_data["distance"]/1000.0
        total_time_s = sum(seg["time_s"] for seg in segments)
        st.write(f"**Distancia Aproximada:** {dist_km:.2f} km")
        st.write(f"**Hora de Llegada Estimada:** {arrival_time.strftime('%H:%M')} (~{int(total_time_s/60)} min)")
        if weather:
            st.write(f"**Clima en Origen:** {weather['temperature']}°C, viento: {weather['windspeed']} km/h")

def show_forecast_hours():
    st.markdown("¿En cuántas horas desea conocer la situación del tráfico?")
    hours = list(range(1, 25))
    selected_hour = st.selectbox("Horas en el futuro:", hours, key="forecast_hours")
    col1, col2 = st.columns(2)
    with col1:
        origin = st.text_input("Origen (Pronóstico)", "Plaza Mayor, Madrid, España", key="forecast_origin")
    with col2:
        destination = st.text_input("Destino (Pronóstico)", "Puerta del Sol, Madrid, España", key="forecast_destination")
    if st.button("Mostrar Pronóstico", key="btn_forecast"):
        origin_coords, origin_full = geocode_address(origin)
        dest_coords, dest_full = geocode_address(destination)
        if not origin_coords or not dest_coords:
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
        m_future = folium.Map(location=[(origin_coords[0]+dest_coords[0])/2,
                                        (origin_coords[1]+dest_coords[1])/2],
                              zoom_start=12, tiles="OpenStreetMap")
        for seg in segments:
            folium.PolyLine([(pt[1], pt[0]) for pt in seg["coords"]],
                            color=seg["color"], weight=4).add_to(m_future)
        folium.Marker(origin_coords, tooltip="Origen", icon=folium.Icon(color="green")).add_to(m_future)
        folium.Marker(dest_coords, tooltip="Destino", icon=folium.Icon(color="red")).add_to(m_future)
        st_folium(m_future, width=700)
        dist_km = route_data["distance"]/1000.0
        total_time_s = sum(seg["time_s"] for seg in segments)
        st.write(f"**Distancia Total:** {dist_km:.2f} km")
        st.write(f"**Hora de Salida (Futura):** {future_dt.strftime('%H:%M')}")
        st.write(f"**Hora de Llegada Estimada:** {arrival_time.strftime('%H:%M')} (~{int(total_time_s/60)} min)")
        if weather:
            st.write(f"**Clima Estimado:** {weather['temperature']}°C, viento: {weather['windspeed']} km/h")

def show_simulation_advanced():
    st.markdown("Simulación Avanzada de escenarios de congestión y tiempos de viaje.")
    offset = st.slider("Horas en el Futuro (simulación):", 1, 6, 2, key="slider_simulation")
    sim_result = simulate_traffic_forecast(offset)
    st.write(f"**Nivel de Saturación:** {sim_result['saturacion']}")
    st.write(f"**Tiempo Normal:** {sim_result['tiempo_normal']} min")
    st.write(f"**Tiempo Ajustado:** {sim_result['tiempo_proyectado']} min")

def show_talavera_module():
    st.markdown("### Talavera de la Reina: Caso Especializado")
    st.markdown("Mapa y análisis de rutas en Talavera y su comarca (simulado).")
    # Reutilizar la lógica de show_map_and_traffic si se quiere, 
    # pero con direcciones por defecto centradas en Talavera.

    with st.form("talavera_form"):
        origin = st.text_input("Origen (Talavera)", "Plaza de España, Talavera de la Reina, España", key="talavera_origin_key")
        destination = st.text_input("Destino (Talavera)", "Ayuntamiento, Talavera de la Reina, España", key="talavera_destination_key")
        start_time = st.time_input("Hora de Inicio", datetime.now().time(), key="talavera_time_key")
        submitted = st.form_submit_button("Calcular Ruta Talavera")
    
    if submitted:
        origin_coords, origin_full = geocode_address(origin)
        dest_coords, dest_full = geocode_address(destination)
        if not origin_coords or not dest_coords:
            st.error("No se pudo geocodificar direcciones para Talavera.")
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
    
    st.markdown("#### Análisis Histórico en Talavera")
    analyze_historical_data()
    st.markdown("#### Simulación Avanzada para Talavera")
    show_simulation_advanced()

# ============================================================================
# Estructura Principal: Selección de Usuario
# ============================================================================

def main_app():
    st.title("Trafiquea: Soluciones de Movilidad y Sostenibilidad")
    st.markdown("Bienvenido/a. Seleccione su tipo de usuario para acceder a las funcionalidades:")

    user_type = st.selectbox("Tipo de Usuario", ["Ayuntamiento", "Empresa", "Particular"], key="user_type_selector")

    st.markdown("---")

    if user_type == "Ayuntamiento":
        st.markdown("## Ayuntamientos: Movilidad y Sostenibilidad")
        st.markdown("En esta sección se ofrecen datos y mapas para la gestión urbana, con un módulo especializado para Talavera.")
        sub_tabs = st.tabs(["Mapa y Tráfico", "Talavera Especializado", "Simulación Avanzada", "Integración API"])

        with sub_tabs[0]:
            st.markdown("### Mapa y Tráfico en Tiempo Real (Ayuntamiento)")
            show_map_and_traffic()
            st.markdown("### Predicciones de Tráfico Futuro (Ayuntamiento)")
            show_forecast()
            st.markdown("### Análisis de Datos Históricos")
            analyze_historical_data()

        with sub_tabs[1]:
            show_talavera_module()

        with sub_tabs[2]:
            show_simulation_advanced()

        with sub_tabs[3]:
            show_integration_form()

    elif user_type == "Empresa":
        st.markdown("## Empresas: Optimización y Créditos de Emisiones")
        st.markdown("Herramientas para optimizar rutas, estimar costes, calcular CAE y generar valor.")
        sub_tabs = st.tabs(["Tráfico en Tiempo Real", "Pronósticos", "Simulación Avanzada", "Calculadora CAE", "Integración API"])

        with sub_tabs[0]:
            show_map_and_traffic()
        with sub_tabs[1]:
            show_forecast()
        with sub_tabs[2]:
            show_simulation_advanced()
        with sub_tabs[3]:
            st.markdown("### Calculadora de CAE (Créditos de Actividad de Emisiones)")
            vol_co2 = st.number_input("Volumen de CO₂ evitado (kg)", min_value=0.0, value=1000.0, step=100.0, key="co2_input")
            if st.button("Calcular CAE", key="calc_cae_btn"):
                cae_factor = 0.001
                cae_val = vol_co2 * cae_factor
                st.write(f"Se estima la generación de {cae_val:.2f} CAE potenciales.")
        with sub_tabs[4]:
            show_integration_form()

    else:
        st.markdown("## Particulares: Rutas y Predicciones Diarias")
        st.markdown("Soluciones rápidas para desplazamientos en coche, bicicleta o transporte público.")
        sub_tabs = st.tabs(["Mapa y Tráfico en Tiempo Real", "Pronósticos de Ruta", "Opciones Eco (Bicicleta / Transporte)"])

        with sub_tabs[0]:
            show_map_and_traffic()
        with sub_tabs[1]:
            show_forecast()
        with sub_tabs[2]:
            st.markdown("### Rutas en Bicicleta / Transporte Público (simulado)")
            st.markdown("Podríamos integrar otra API OSRM en modo cycling/transit si estuviera disponible.")
            st.markdown("Se mostrarían mapas y tiempos de viaje aproximados sin gráficas saturadas.")

if __name__ == "__main__":
    main_app()
