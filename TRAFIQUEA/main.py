"""
Trafiquea: Modelo Especializado para Talavera de la Reina

Este dashboard está enfocado en la ciudad de Talavera de la Reina y ofrece las siguientes funcionalidades:
  1. Optimización de Rutas en Tiempo Real:
     - Permite ingresar direcciones de origen y destino (por defecto, se asumen direcciones de Talavera de la Reina).
     - Se solicita la hora de inicio del viaje y se calcula la ruta mediante OSRM.
     - La ruta se divide en segmentos a los que se asigna un nivel de congestión simulado según la zona, la hora y condiciones climáticas.
  2. Pronóstico de Ruta:
     - Permite seleccionar un intervalo de horas (hasta 24) en el futuro para obtener proyecciones de tráfico y tiempos de viaje.
  3. Análisis de Datos Históricos:
     - Se muestran tendencias y estadísticas simuladas de demanda y tráfico en Talavera de la Reina.
  4. Simulación Avanzada:
     - Permite comparar escenarios modificando parámetros (p.ej., cambios en condiciones climáticas y hora del viaje) para ver el impacto en los tiempos de viaje.
  5. Integración API:
     - Se ofrece un formulario para que usuarios interesados soliciten la integración vía API (la información se enviará internamente).

Este modelo aprovecha datos públicos y simulaciones específicas para Talavera de la Reina, lo que facilita la presentación de un caso de uso al ayuntamiento y a otros actores locales.
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
# Funciones de Geocodificación
# =============================================================================

def geocode_address(address: str):
    """
    Geocodifica una dirección usando Nominatim.
    Devuelve ((lat, lon), dirección completa) o (None, None) en caso de fallo.
    """
    geolocator = Nominatim(user_agent="trafiquea_dashboard")
    location = geolocator.geocode(address)
    if location:
        return (location.latitude, location.longitude), location.address
    return None, None

def reverse_geocode(lat: float, lon: float):
    """
    Obtiene la dirección completa a partir de latitud y longitud mediante geocodificación inversa.
    """
    geolocator = Nominatim(user_agent="trafiquea_reverse")
    location = geolocator.reverse((lat, lon), language="en")
    if location:
        return location.address
    return "Dirección no encontrada"

# =============================================================================
# Función para Simular Factor de Tráfico Específico para Talavera de la Reina
# =============================================================================

def simulate_zone_factor(full_address: str):
    """
    Asigna un factor de tráfico basado en la zona geográfica.
    Para Talavera de la Reina se asigna un factor específico.
    """
    address_lower = full_address.lower()
    if "talavera" in address_lower:
        return 1.3
    return 1.2

# =============================================================================
# Función para Obtener Ruta con OSRM
# =============================================================================

def get_route_osrm(origin_coords, destination_coords):
    """
    Calcula la ruta entre dos puntos mediante la API OSRM.
    Devuelve un diccionario con:
      - geometry: GeoJSON de la ruta.
      - steps: Lista de segmentos con distancia, duración y geometría.
      - distance: Distancia total en metros.
      - duration: Duración total en segundos.
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

# =============================================================================
# Funciones para Simulación de Tráfico y Tiempos de Viaje
# =============================================================================

def assign_traffic_level(distance_km, start_time, weather, zone_factor):
    """
    Asigna un nivel de tráfico y un factor multiplicador basado en:
      - Hora de inicio (hora punta o no).
      - Condiciones climáticas (viento).
      - Distancia del segmento.
      - Factor específico de la zona (para Talavera se utiliza simulate_zone_factor).
    Devuelve (nivel, factor) donde nivel es "Bajo", "Moderado" o "Alto".
    """
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
    """
    Devuelve un color hexadecimal según el nivel de congestión.
    Verde para "Bajo", naranja para "Moderado", rojo para "Alto".
    """
    if level == "Bajo":
        return "#2ecc71"
    elif level == "Moderado":
        return "#f39c12"
    else:
        return "#e74c3c"

def simulate_route_segments(steps, start_time, weather, zone_factor):
    """
    Recorre los pasos (steps) de la ruta y asigna un nivel de tráfico simulado a cada segmento.
    Devuelve una lista de segmentos (con coordenadas, color, distancia y tiempo ajustado)
    y la hora estimada de llegada.
    """
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
    """
    Simula el pronóstico de tráfico para un intervalo en horas (hasta 6 horas).
    Devuelve un diccionario con:
      - saturacion: Nivel de congestión.
      - tiempo_normal: Tiempo de viaje normal (min).
      - tiempo_proyectado: Tiempo de viaje ajustado (min).
    """
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
# Función para Análisis de Datos Históricos Simulados
# =============================================================================

def analyze_historical_data():
    """
    Carga un dataset simulado (usando datos de ejemplo) y muestra tendencias y estadísticas.
    """
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
# Función para Mostrar el Formulario de Integración vía API
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
# Tab Especializado para Talavera de la Reina
# =============================================================================

def tab_talavera():
    st.header("Modelo Especializado: Talavera de la Reina")
    st.markdown("""
    Este módulo se enfoca en Talavera de la Reina utilizando datos públicos y simulaciones específicas para la zona.
    Las direcciones se asumen dentro del área de Talavera de la Reina para optimizar la geocodificación y análisis.
    """)

    # Sub-tab: Tráfico en Tiempo Real para Talavera
    st.subheader("Tráfico en Tiempo Real")
    with st.form("form_trafico_talavera"):
        col1, col2, col3 = st.columns(3)
        with col1:
            origin = st.text_input("Dirección de Origen", "Plaza de España, Talavera de la Reina, España")
        with col2:
            destination = st.text_input("Dirección de Destino", "Ayuntamiento, Talavera de la Reina, España")
        with col3:
            start_time_input = st.time_input("Hora de Inicio", datetime.now().time())
        submitted = st.form_submit_button("Calcular Ruta")
    
    if submitted:
        origin_coords, origin_full = geocode_address(origin)
        dest_coords, dest_full = geocode_address(destination)
        if not (origin_coords and dest_coords):
            st.error("Verifique que las direcciones estén completas y sean correctas.")
            return
        now = datetime.now()
        start_dt = datetime(now.year, now.month, now.day, start_time_input.hour, start_time_input.minute)
        weather = get_weather_open_meteo(origin_coords[0], origin_coords[1])
        # Utilizamos el factor específico para Talavera
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

    # Sub-tab: Pronóstico de Ruta para Talavera
    st.subheader("Pronóstico de Ruta (hasta 24 horas)")
    hour_options = list(range(1, 25))
    selected_hour = st.selectbox("Horas en el futuro:", hour_options,
                                 format_func=lambda x: f"{x} hora{'s' if x > 1 else ''}")
    col1, col2 = st.columns(2)
    with col1:
        origin_forecast = st.text_input("Origen (Pronóstico)", "Plaza de España, Talavera de la Reina, España")
    with col2:
        destination_forecast = st.text_input("Destino (Pronóstico)", "Ayuntamiento, Talavera de la Reina, España")
    
    if st.button("Mostrar Pronóstico"):
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
    
    # Sub-tab: Análisis de Datos Históricos
    st.subheader("Análisis de Datos Históricos para Talavera")
    st.markdown("A continuación se muestran tendencias y estadísticas simuladas de demanda en Talavera de la Reina.")
    # Simular dataset para Talavera (usar datos de ejemplo y ajustar)
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
        # Simular que los datos corresponden a Talavera ajustando la demanda
        df_talavera["Demanda"] = df_talavera["Demanda"] * 0.8  # Factor de ajuste
        df_talavera.sort_values("Fecha", inplace=True)
        st.line_chart(df_talavera.set_index("Fecha")["Demanda"])
        st.write(df_talavera.describe())
    
    # Sub-tab: Simulación Avanzada de Escenarios
    st.subheader("Simulación Avanzada de Escenarios")
    st.markdown("Ajuste parámetros para simular cómo varían los tiempos de viaje en Talavera de la Reina.")
    hour_offset = st.slider("Horas en el Futuro:", min_value=1, max_value=6, value=2)
    forecast = simulate_traffic_forecast(hour_offset)
    st.write(f"**Nivel de Saturación Simulado:** {forecast['saturacion']}")
    st.write(f"**Tiempo de Viaje Normal (simulado):** {forecast['tiempo_normal']} min")
    st.write(f"**Tiempo de Viaje Estimado (simulado):** {forecast['tiempo_proyectado']} min")
    
    # Sub-tab: Integración API
    st.subheader("Integrar API")
    show_integration_form()

# =============================================================================
# Función Principal del Dashboard
# =============================================================================

def main_app():
    st.title("Trafiquea: Modelo Especializado para Talavera de la Reina")
    st.markdown("""
    Esta plataforma se ha desarrollado específicamente para Talavera de la Reina utilizando datos públicos y simulaciones adaptadas a la zona.  
    Las funcionalidades incluyen:
      - Optimización de rutas en tiempo real con análisis de tráfico por zona.
      - Pronósticos de ruta para intervalos de tiempo futuros.
      - Análisis de tendencias y simulación avanzada para apoyar la planificación.
      - Un formulario para solicitar la integración vía API.
    """)
    tab_talavera()

if __name__ == "__main__":
    main_app()
