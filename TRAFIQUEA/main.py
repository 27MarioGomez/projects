"""
Trafiquea: Plataforma Integral para Transporte y Rutas en Tiempo Real

Este dashboard ofrece funcionalidades para distintos públicos:
  
  - Ayuntamientos: Funcionalidades específicas para la gestión urbana, análisis de datos históricos y simulación avanzada de escenarios en ciudades, con un enfoque en Talavera de la Reina.
  
  - Empresas: Herramientas de optimización de rutas en tiempo real, pronósticos de ruta y un formulario de integración vía API para la mejora de la eficiencia logística.
  
  - Particulares: Funcionalidades sencillas de cálculo de rutas y pronósticos para viajes cotidianos.

Se utilizan geocodificación y reverse geocodificación mediante Nominatim, la API de OSRM para obtener rutas y Folium (a través de streamlit-folium) para visualizar mapas de OpenStreetMap.
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
# Factor de Tráfico por Zona
# =============================================================================

def simulate_zone_factor(full_address: str):
    address_lower = full_address.lower()
    if "talavera" in address_lower:
        return 1.3
    return 1.2

# =============================================================================
# Obtener Clima con Open-Meteo
# =============================================================================

def get_weather_open_meteo(lat: float, lon: float):
    url = "https://api.open-meteo.com/v1/forecast"
    params = {"latitude": lat, "longitude": lon, "current_weather": True, "timezone": "auto"}
    response = requests.get(url, params=params)
    if response.status_code == 200:
        data = response.json()
        current = data.get("current_weather", {})
        return {"temperature": current.get("temperature"), "windspeed": current.get("windspeed")}
    return None

# =============================================================================
# Obtener Ruta con OSRM
# =============================================================================

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
        return "#2ecc71"  # Verde
    elif level == "Moderado":
        return "#f39c12"  # Naranja
    else:
        return "#e74c3c"  # Rojo

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
# Análisis de Datos Históricos
# =============================================================================

def analyze_historical_data():
    url = "https://raw.githubusercontent.com/plotly/datasets/master/2014_apple_stock.csv"
    try:
        df = pd.read_csv(url, parse_dates=["AAPL_x"])
        df.rename(columns={"AAPL_x": "Fecha", "AAPL_y": "Demanda"}, inplace=True)
    except Exception as e:
        st.error("Error al cargar datos históricos.")
        return
    if "Demanda" not in df.columns:
        df["Demanda"] = np.random.randint(100, 500, len(df))
    df.sort_values("Fecha", inplace=True)
    st.line_chart(df.set_index("Fecha")["Demanda"])
    st.write(df.describe())

# =============================================================================
# Formulario de Integración vía API
# =============================================================================

def show_integration_form():
    with st.form("form_integration"):
        st.subheader("Solicita Integración vía API")
        nombre = st.text_input("Nombre")
        apellidos = st.text_input("Apellidos")
        institucion = st.text_input("Institución o Empresa")
        mensaje = st.text_area("Mensaje (opcional)")
        submitted = st.form_submit_button("Enviar Solicitud")
        if submitted:
            st.success("Solicitud enviada correctamente. Nos pondremos en contacto contigo.")

# =============================================================================
# Tabs para Particulares, Empresas y Ayuntamientos
# =============================================================================

def tab_particulares():
    st.header("Particulares")
    st.markdown("""
    Funcionalidades para usuarios individuales:
      - Calcular rutas en tiempo real y obtener pronósticos de viaje.
    """)
    st.subheader("Tráfico en Tiempo Real")
    tab_trafico_en_tiempo_real()
    st.subheader("Pronósticos de Ruta")
    tab_pronosticos_ruta()

def tab_empresas():
    st.header("Empresas")
    st.markdown("""
    Herramientas para mejorar la eficiencia operativa:
      - Optimización de rutas en tiempo real.
      - Pronósticos de ruta para planificación logística.
      - Integración vía API.
      - Simulación avanzada de escenarios.
    """)
    st.subheader("Tráfico en Tiempo Real")
    tab_trafico_en_tiempo_real()
    st.subheader("Pronósticos de Ruta")
    tab_pronosticos_ruta()
    st.subheader("Simulación Avanzada")
    tab_simulacion_avanzada()
    st.subheader("Integrar API")
    tab_integrar_api()

def tab_ayuntamientos():
    st.header("Ayuntamientos")
    st.markdown("""
    Funcionalidades orientadas a la planificación urbana y gestión de movilidad:
      - Modelo especializado para Talavera de la Reina.
      - Análisis de datos históricos y simulación avanzada de escenarios.
      - Pronósticos y optimización de rutas con enfoque local.
    """)
    st.subheader("Modelo Especializado - Talavera de la Reina")
    tab_talavera()
    st.subheader("Análisis de Datos Históricos")
    tab_analisis_datos()
    st.subheader("Simulación Avanzada")
    tab_simulacion_avanzada()
    st.subheader("Integrar API")
    tab_integrar_api()

# =============================================================================
# Tabs: Funcionalidades Comunes
# =============================================================================

def tab_trafico_en_tiempo_real():
    st.markdown("""
    Ingrese las direcciones de origen y destino (incluyendo ciudad, estado, país) y la hora de inicio de su viaje.
    La ruta se visualizará con segmentos coloreados según el nivel de congestión simulado.
    """)
    with st.form("form_trafico"):
        col1, col2, col3 = st.columns(3)
        with col1:
            origin = st.text_input("Dirección de Origen", "Plaza Mayor, Madrid, España")
        with col2:
            destination = st.text_input("Dirección de Destino", "Puerta del Sol, Madrid, España")
        with col3:
            start_time_input = st.time_input("Hora de Inicio", datetime.now().time())
        submitted = st.form_submit_button("Calcular Ruta")
    
    if submitted:
        origin_coords, origin_full = geocode_address(origin)
        dest_coords, dest_full = geocode_address(destination)
        if not (origin_coords and dest_coords):
            st.error("Error en la geocodificación. Verifique que las direcciones estén completas.")
            return
        now = datetime.now()
        start_dt = datetime(now.year, now.month, now.day, start_time_input.hour, start_time_input.minute)
        weather = get_weather_open_meteo(origin_coords[0], origin_coords[1])
        zone_factor = simulate_zone_factor(origin_full)
        route_data = get_route_osrm(origin_coords, dest_coords)
        if not route_data:
            st.error("No se pudo obtener la ruta con OSRM.")
            return
        segments, arrival_time = simulate_route_segments(route_data["steps"], start_dt, weather, zone_factor)
        m = folium.Map(location=[(origin_coords[0]+dest_coords[0])/2, (origin_coords[1]+dest_coords[1])/2],
                       zoom_start=12, tiles="OpenStreetMap")
        for seg in segments:
            folium.PolyLine(locations=[(pt[1], pt[0]) for pt in seg["coords"]],
                            color=seg["color"], weight=4).add_to(m)
        folium.Marker(origin_coords, popup=origin_full, tooltip="Origen",
                      icon=folium.Icon(color="green")).add_to(m)
        folium.Marker(dest_coords, popup=dest_full, tooltip="Destino",
                      icon=folium.Icon(color="red")).add_to(m)
        st_folium(m, width=700)
        total_distance = route_data["distance"] / 1000.0
        total_time = sum(seg["time_s"] for seg in segments)
        st.write(f"**Distancia Total:** {total_distance:.2f} km")
        st.write(f"**Hora de Llegada Estimada:** {arrival_time.strftime('%H:%M')} (~{int(total_time/60)} min de viaje)")
        if weather:
            st.write(f"**Clima en Origen:** {weather['temperature']}°C, Viento: {weather['windspeed']} km/h")
        st.markdown("""
        **Leyenda de Saturación:**
        - Verde: Bajo  
        - Naranja: Moderado  
        - Rojo: Alto  
        """)

def tab_pronosticos_ruta():
    st.markdown("""
    Seleccione cuántas horas en el futuro desea conocer la situación del tráfico y la ruta óptima para ese momento.
    Se mostrarán estimaciones de tiempo de viaje (normal y ajustado) y la ruta correspondiente.
    """)
    hour_options = list(range(1, 25))
    selected_hour = st.selectbox("Horas en el futuro:", hour_options, format_func=lambda x: f"{x} hora{'s' if x > 1 else ''}")
    col1, col2 = st.columns(2)
    with col1:
        origin = st.text_input("Dirección de Origen (Pronóstico)", "Plaza Mayor, Madrid, España")
    with col2:
        destination = st.text_input("Dirección de Destino (Pronóstico)", "Puerta del Sol, Madrid, España")
    
    if st.button("Mostrar Pronóstico"):
        origin_coords, origin_full = geocode_address(origin)
        dest_coords, dest_full = geocode_address(destination)
        if not (origin_coords and dest_coords):
            st.error("Error en la geocodificación de las direcciones.")
            return
        future_dt = datetime.now() + timedelta(hours=selected_hour)
        weather = get_weather_open_meteo(origin_coords[0], origin_coords[1])
        zone_factor = simulate_zone_factor(origin_full)
        route_data = get_route_osrm(origin_coords, dest_coords)
        if not route_data:
            st.error("No se pudo obtener la ruta.")
            return
        segments, arrival_time = simulate_route_segments(route_data["steps"], future_dt, weather, zone_factor)
        m_future = folium.Map(location=[(origin_coords[0]+dest_coords[0])/2, (origin_coords[1]+dest_coords[1])/2],
                              zoom_start=12, tiles="OpenStreetMap")
        for seg in segments:
            folium.PolyLine(locations=[(pt[1], pt[0]) for pt in seg["coords"]],
                            color=seg["color"], weight=4).add_to(m_future)
        folium.Marker(origin_coords, popup=origin_full, tooltip="Origen",
                      icon=folium.Icon(color="green")).add_to(m_future)
        folium.Marker(dest_coords, popup=dest_full, tooltip="Destino",
                      icon=folium.Icon(color="red")).add_to(m_future)
        st_folium(m_future, width=700)
        total_distance = route_data["distance"] / 1000.0
        total_time = sum(seg["time_s"] for seg in segments)
        st.write(f"**Distancia Total:** {total_distance:.2f} km")
        st.write(f"**Hora de Salida (Futura):** {future_dt.strftime('%d/%m %H:%M')}")
        st.write(f"**Hora de Llegada Estimada:** {arrival_time.strftime('%H:%M')} (~{int(total_time/60)} min de viaje)")
        if weather:
            st.write(f"**Clima en Origen (aprox.):** {weather['temperature']}°C, Viento: {weather['windspeed']} km/h")

def tab_analisis_datos():
    st.subheader("Análisis de Datos Históricos")
    analyze_historical_data()

def tab_simulacion_avanzada():
    st.subheader("Simulación Avanzada de Escenarios")
    st.markdown("Ajuste los parámetros para simular cómo varían los tiempos de viaje y la congestión.")
    hour_offset = st.slider("Horas en el Futuro:", min_value=1, max_value=6, value=2)
    forecast = simulate_traffic_forecast(hour_offset)
    st.write(f"**Nivel de Saturación Simulado:** {forecast['saturacion']}")
    st.write(f"**Tiempo de Viaje Normal (simulado):** {forecast['tiempo_normal']} min")
    st.write(f"**Tiempo de Viaje Estimado (simulado):** {forecast['tiempo_proyectado']} min")

def tab_integrar_api():
    show_integration_form()

# =============================================================================
# Tab Especializado para Talavera de la Reina
# =============================================================================

def tab_talavera():
    st.header("Modelo Especializado: Talavera de la Reina")
    st.markdown("""
    Este módulo se centra en Talavera de la Reina utilizando datos y simulaciones específicas para la zona.
    Las direcciones se asumen dentro del área de Talavera para optimizar la geocodificación y el análisis.
    """)
    st.subheader("Tráfico en Tiempo Real - Talavera")
    with st.form("form_trafico_talavera"):
        col1, col2, col3 = st.columns(3)
        with col1:
            origin = st.text_input("Origen", "Plaza de España, Talavera de la Reina, España")
        with col2:
            destination = st.text_input("Destino", "Ayuntamiento, Talavera de la Reina, España")
        with col3:
            start_time_input = st.time_input("Hora de Inicio", datetime.now().time())
        submitted = st.form_submit_button("Calcular Ruta")
    
    if submitted:
        origin_coords, origin_full = geocode_address(origin)
        dest_coords, dest_full = geocode_address(destination)
        if not (origin_coords and dest_coords):
            st.error("Verifique que las direcciones estén completas.")
            return
        now = datetime.now()
        start_dt = datetime(now.year, now.month, now.day, start_time_input.hour, start_time_input.minute)
        weather = get_weather_open_meteo(origin_coords[0], origin_coords[1])
        zone_factor = simulate_zone_factor(origin_full)
        route_data = get_route_osrm(origin_coords, dest_coords)
        if not route_data:
            st.error("No se pudo obtener la ruta con OSRM.")
            return
        segments, arrival_time = simulate_route_segments(route_data["steps"], start_dt, weather, zone_factor)
        m = folium.Map(location=[(origin_coords[0] + dest_coords[0]) / 2, (origin_coords[1] + dest_coords[1]) / 2],
                       zoom_start=13, tiles="OpenStreetMap")
        for seg in segments:
            folium.PolyLine(locations=[(pt[1], pt[0]) for pt in seg["coords"]],
                            color=seg["color"], weight=4).add_to(m)
        folium.Marker(origin_coords, popup=origin_full, tooltip="Origen",
                      icon=folium.Icon(color="green")).add_to(m)
        folium.Marker(dest_coords, popup=dest_full, tooltip="Destino",
                      icon=folium.Icon(color="red")).add_to(m)
        st_folium(m, width=700)
        total_distance = route_data["distance"] / 1000.0
        total_time = sum(seg["time_s"] for seg in segments)
        st.write(f"**Distancia Total:** {total_distance:.2f} km")
        st.write(f"**Hora de Llegada Estimada:** {arrival_time.strftime('%H:%M')} (~{int(total_time/60)} min)")
        if weather:
            st.write(f"**Clima en Origen:** {weather['temperature']}°C, Viento: {weather['windspeed']} km/h")
        st.markdown("""
        **Leyenda de Saturación:**
        - Verde: Bajo  
        - Naranja: Moderado  
        - Rojo: Alto  
        """)
    
    st.subheader("Pronóstico de Ruta - Talavera (hasta 24 horas)")
    selected_hour = st.selectbox("Horas en el futuro:", list(range(1, 25)), format_func=lambda x: f"{x} hora{'s' if x > 1 else ''}")
    col1, col2 = st.columns(2)
    with col1:
        origin_forecast = st.text_input("Origen (Pronóstico)", "Plaza de España, Talavera de la Reina, España")
    with col2:
        destination_forecast = st.text_input("Destino (Pronóstico)", "Ayuntamiento, Talavera de la Reina, España")
    
    if st.button("Mostrar Pronóstico Talavera"):
        origin_coords_f, origin_full_f = geocode_address(origin_forecast)
        dest_coords_f, dest_full_f = geocode_address(destination_forecast)
        if not (origin_coords_f and dest_coords_f):
            st.error("Error en la geocodificación de las direcciones.")
            return
        future_dt = datetime.now() + timedelta(hours=selected_hour)
        weather_f = get_weather_open_meteo(origin_coords_f[0], origin_coords_f[1])
        zone_factor_f = simulate_zone_factor(origin_full_f)
        route_data_f = get_route_osrm(origin_coords_f, dest_coords_f)
        if not route_data_f:
            st.error("No se pudo obtener la ruta.")
            return
        segments_f, arrival_time_f = simulate_route_segments(route_data_f["steps"], future_dt, weather_f, zone_factor_f)
        m_f = folium.Map(location=[(origin_coords_f[0] + dest_coords_f[0]) / 2, (origin_coords_f[1] + dest_coords_f[1]) / 2],
                         zoom_start=13, tiles="OpenStreetMap")
        for seg in segments_f:
            folium.PolyLine(locations=[(pt[1], pt[0]) for pt in seg["coords"]],
                            color=seg["color"], weight=4).add_to(m_f)
        folium.Marker(origin_coords_f, popup=origin_full_f, tooltip="Origen",
                      icon=folium.Icon(color="green")).add_to(m_f)
        folium.Marker(dest_coords_f, popup=dest_full_f, tooltip="Destino",
                      icon=folium.Icon(color="red")).add_to(m_f)
        st_folium(m_f, width=700)
        total_distance_f = route_data_f["distance"] / 1000.0
        total_time_f = sum(seg["time_s"] for seg in segments_f)
        st.write(f"**Distancia Total:** {total_distance_f:.2f} km")
        st.write(f"**Hora de Salida (Futura):** {future_dt.strftime('%d/%m %H:%M')}")
        st.write(f"**Hora de Llegada Estimada:** {arrival_time_f.strftime('%H:%M')} (~{int(total_time_f/60)} min)")
        if weather_f:
            st.write(f"**Clima en Origen (aprox.):** {weather_f['temperature']}°C, Viento: {weather_f['windspeed']} km/h")
    
    st.subheader("Análisis de Datos Históricos para Talavera")
    st.markdown("Tendencias y estadísticas simuladas para Talavera de la Reina.")
    url = "https://raw.githubusercontent.com/plotly/datasets/master/2014_apple_stock.csv"
    try:
        df_talavera = pd.read_csv(url, parse_dates=["AAPL_x"])
        df_talavera.rename(columns={"AAPL_x": "Fecha", "AAPL_y": "Demanda"}, inplace=True)
    except Exception as e:
        st.error("Error al cargar datos históricos.")
        df_talavera = pd.DataFrame()
    if df_talavera.empty:
        st.write("No hay datos disponibles.")
    else:
        df_talavera["Demanda"] = df_talavera["Demanda"] * 0.8
        df_talavera.sort_values("Fecha", inplace=True)
        st.line_chart(df_talavera.set_index("Fecha")["Demanda"])
        st.write(df_talavera.describe())
    
    st.subheader("Simulación Avanzada de Escenarios - Talavera")
    hour_offset = st.slider("Horas en el Futuro:", min_value=1, max_value=6, value=2)
    forecast = simulate_traffic_forecast(hour_offset)
    st.write(f"**Nivel de Saturación Simulado:** {forecast['saturacion']}")
    st.write(f"**Tiempo de Viaje Normal (simulado):** {forecast['tiempo_normal']} min")
    st.write(f"**Tiempo de Viaje Estimado (simulado):** {forecast['tiempo_proyectado']} min")
    
    st.subheader("Integrar API")
    show_integration_form()

# =============================================================================
# Función Principal del Dashboard
# =============================================================================

def main_app():
    st.title("Trafiquea: Plataforma Integral para Transporte y Rutas en Tiempo Real")
    st.markdown("""
    Bienvenido a Trafiquea. Explore las funcionalidades según su perfil:
    
    - **Ayuntamientos:** Herramientas específicas para la planificación urbana y gestión de movilidad.
    - **Empresas:** Funcionalidades para optimizar rutas y planificar operaciones logísticas.
    - **Particulares:** Herramientas para usuarios individuales que buscan calcular rutas y obtener pronósticos de viaje.
    """)
    main_tabs = st.tabs(["Ayuntamientos", "Empresas", "Particulares"])
    
    with main_tabs[0]:
        tab_ayuntamientos()
    with main_tabs[1]:
        tab_empresas()
    with main_tabs[2]:
        tab_particulares()

def tab_ayuntamientos():
    st.header("Ayuntamientos")
    st.markdown("""
    Este módulo está orientado a la gestión urbana. Incluye:
      - Modelo especializado para Talavera de la Reina.
      - Análisis de datos históricos y simulación avanzada de escenarios.
      - Funcionalidades de integración para facilitar la toma de decisiones.
    """)
    tab_talavera()
    st.markdown("---")
    st.subheader("Análisis de Datos Históricos")
    tab_analisis_datos()
    st.markdown("---")
    st.subheader("Simulación Avanzada")
    tab_simulacion_avanzada()
    st.markdown("---")
    st.subheader("Integrar API")
    tab_integrar_api()

def tab_empresas():
    st.header("Empresas")
    st.markdown("""
    Herramientas para la optimización operativa y logística:
      - Cálculo de rutas en tiempo real.
      - Pronósticos de ruta para planificar operaciones.
      - Simulación avanzada para comparar escenarios.
      - Integración vía API para la conexión con sistemas internos.
    """)
    st.subheader("Tráfico en Tiempo Real")
    tab_trafico_en_tiempo_real()
    st.markdown("---")
    st.subheader("Pronósticos de Ruta")
    tab_pronosticos_ruta()
    st.markdown("---")
    st.subheader("Simulación Avanzada")
    tab_simulacion_avanzada()
    st.markdown("---")
    st.subheader("Integrar API")
    tab_integrar_api()

def tab_particulares():
    st.header("Particulares")
    st.markdown("""
    Funcionalidades para usuarios individuales:
      - Cálculo de rutas en tiempo real.
      - Pronósticos de ruta para conocer tiempos de viaje.
    """)
    st.subheader("Tráfico en Tiempo Real")
    tab_trafico_en_tiempo_real()
    st.markdown("---")
    st.subheader("Pronósticos de Ruta")
    tab_pronosticos_ruta()

if __name__ == "__main__":
    main_app()
