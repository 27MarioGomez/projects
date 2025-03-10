# main.py
"""
Trafiquea: Plataforma Integral para Transporte y Rutas en Tiempo Real

Este dashboard utiliza pestañas (tabs) para organizar las siguientes funcionalidades:

1. Tráfico en Tiempo Real:
   - Ingreso manual de dirección de origen y destino (incluyendo ciudad, estado, país).
   - Selección de la hora de inicio del viaje.
   - Cálculo de ruta mediante OSRM y visualización de segmentos coloreados según el nivel de tráfico simulado.
   - Muestra de distancia total y hora de llegada estimada.
   
2. Pronósticos de Ruta:
   - Permite seleccionar cuántas horas en el futuro se desea conocer la situación del tráfico.
   - Se simulan los tiempos de viaje (normal vs. ajustado por tráfico) y se muestra la ruta óptima para ese escenario.
   
3. Análisis de Datos:
   - Se muestran análisis y visualizaciones (por ejemplo, correlación entre clima y tráfico, tendencias históricas de demanda).
   - Se pueden integrar modelos de clustering o regresión para segmentar zonas o predecir la demanda.
   
4. Simulación Avanzada:
   - Permite comparar escenarios modificando parámetros (por ejemplo, condiciones climáticas, hora de viaje) para ver cómo varían los tiempos y rutas.
   
5. Integrar API:
   - Formulario para que interesados dejen sus datos y soliciten integración vía API.

Nota:  
- Se usa Folium para mostrar mapas de OpenStreetMap con un estilo limpio y enfocado en calles.  
- Las funciones de geocodificación se basan en Nominatim, por lo que se recomienda ingresar direcciones completas para mejores resultados.
- Este prototipo se puede potenciar con modelos avanzados de data science (regresión, clustering, series temporales) conforme se disponga de datos reales.
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
    Retorna ((lat, lon), dirección completa) o (None, None) si falla.
    """
    geolocator = Nominatim(user_agent="trafiquea_dashboard")
    location = geolocator.geocode(address)
    if location:
        return (location.latitude, location.longitude), location.address
    return None, None

def reverse_geocode(lat, lon):
    """
    Geocodificación inversa: de lat/lon obtiene la dirección completa.
    """
    geolocator = Nominatim(user_agent="trafiquea_reverse")
    location = geolocator.reverse((lat, lon), language="en")
    if location:
        return location.address
    return "Dirección no encontrada"

# =============================================================================
# Función para Obtener Ruta con OSRM
# =============================================================================

def get_route_osrm(origin_coords, destination_coords):
    """
    Calcula la ruta entre dos puntos usando OSRM.
    Retorna un dict con geometry (GeoJSON), steps (lista de segmentos), distance (metros) y duration (segundos).
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

def assign_traffic_level(distance_km, start_time, weather):
    """
    Asigna un nivel de tráfico y un factor multiplicador de tiempo según:
      - Hora de inicio (hora punta).
      - Condiciones climáticas (viento fuerte).
      - Distancia (trayectos cortos o largos).
    Retorna (nivel, factor) donde nivel es "Bajo", "Moderado" o "Alto".
    """
    hour = start_time.hour
    if 7 <= hour <= 9 or 17 <= hour <= 19:
        base_level = "Alto"
        base_factor = 1.5
    else:
        base_level = "Moderado"
        base_factor = 1.2

    if weather and weather.get("windspeed") and weather["windspeed"] > 20:
        base_factor += 0.2

    if distance_km < 1:
        base_factor -= 0.1
    elif distance_km > 5:
        base_factor += 0.1

    if base_factor <= 1.2:
        level = "Bajo"
    elif 1.2 < base_factor < 1.5:
        level = "Moderado"
    else:
        level = "Alto"

    return level, max(1.0, base_factor)

def color_for_level(level):
    """
    Devuelve un color hexadecimal según el nivel de tráfico.
    """
    if level == "Bajo":
        return "#2ecc71"  # Verde
    elif level == "Moderado":
        return "#f39c12"  # Naranja
    else:
        return "#e74c3c"  # Rojo

def simulate_route_segments(steps, start_time, weather):
    """
    Recorre los steps de la ruta (de OSRM) y asigna para cada segmento un nivel de tráfico.
    Retorna:
      - Una lista de segmentos (con coordenadas, color, distancia y tiempo ajustado).
      - La hora estimada de llegada.
    """
    segments = []
    current_time = start_time
    for step in steps:
        distance_m = step["distance"]
        distance_km = distance_m / 1000.0
        level, factor = assign_traffic_level(distance_km, current_time, weather)
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
    Simula el pronóstico de tráfico para 'hour_offset' horas en el futuro.
    Retorna un dict con saturación, tiempo de viaje normal y tiempo de viaje ajustado.
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
# Funciones para Análisis de Datos (Ejemplo: Tendencias y Correlaciones)
# =============================================================================

def analyze_historical_data():
    """
    Ejemplo de análisis de datos históricos de demanda.
    Aquí se simula la carga de un dataset y se muestran tendencias y correlaciones.
    """
    # Cargamos datos simulados (por ejemplo, del CSV de OSRM)
    url = "https://raw.githubusercontent.com/plotly/datasets/master/2014_apple_stock.csv"
    try:
        df = pd.read_csv(url, parse_dates=["AAPL_x"])
        df.rename(columns={"AAPL_x": "Fecha", "AAPL_y": "Demanda"}, inplace=True)
    except Exception as e:
        st.error("Error al cargar datos históricos.")
        return

    # Convertir la Demanda en valores simulados si es necesario
    if "Demanda" not in df.columns:
        df["Demanda"] = np.random.randint(100, 500, len(df))
    df.sort_values("Fecha", inplace=True)
    
    st.subheader("Tendencia Histórica de Demanda")
    st.line_chart(df.set_index("Fecha")["Demanda"])
    
    # Simular correlación entre demanda y fecha (p.ej., tendencia ascendente o estacionalidad)
    st.subheader("Análisis Estadístico Simulado")
    st.write("Se podría implementar un análisis de regresión o clustering para segmentar zonas y predecir demanda.")
    # Aquí se muestra un resumen estadístico como ejemplo
    st.write(df.describe())

# =============================================================================
# Función para el Formulario de Integración vía API
# =============================================================================

def show_integration_form():
    st.subheader("Solicita Integración vía API")
    with st.form("form_integration"):
        nombre = st.text_input("Nombre")
        apellidos = st.text_input("Apellidos")
        institucion = st.text_input("Institución o Empresa")
        mensaje = st.text_area("Mensaje (opcional)")
        submitted = st.form_submit_button("Enviar Solicitud")
        if submitted:
            st.success("Solicitud enviada correctamente. Nos pondremos en contacto contigo.")

# =============================================================================
# Diseño Principal del Dashboard con Tabs
# =============================================================================

def main_app():
    st.title("Trafiquea: Plataforma Integral para Transporte y Rutas en Tiempo Real")
    st.markdown("""
    **Bienvenido a Trafiquea.**  
    Esta plataforma ofrece múltiples funcionalidades:
    - **Tráfico en Tiempo Real:** Visualiza rutas y niveles de congestión.
    - **Pronósticos de Ruta:** Obtén proyecciones para hasta 24 horas en el futuro.
    - **Análisis de Datos:** Explora tendencias históricas y correlaciones.
    - **Simulación Avanzada:** Compara escenarios y ajusta parámetros para optimizar rutas.
    - **Integrar API:** Solicita integración mediante un formulario.
    """)
    
    tabs = st.tabs(["Tráfico en Tiempo Real", "Pronósticos de Ruta", "Análisis de Datos", "Simulación Avanzada", "Integrar API"])
    
    # Tab 1: Tráfico en Tiempo Real
    with tabs[0]:
        tab_trafico_en_tiempo_real()
    
    # Tab 2: Pronósticos de Ruta
    with tabs[1]:
        tab_pronosticos_ruta()
    
    # Tab 3: Análisis de Datos
    with tabs[2]:
        st.subheader("Análisis de Datos Históricos")
        analyze_historical_data()
    
    # Tab 4: Simulación Avanzada
    with tabs[3]:
        st.subheader("Simulación Avanzada de Escenarios")
        st.markdown("""
        Modifica los parámetros para ver cómo varían los tiempos de viaje y rutas bajo diferentes condiciones.
        """)
        hour_offset = st.slider("Horas en el Futuro:", min_value=1, max_value=6, value=2)
        forecast = simulate_traffic_forecast(hour_offset)
        st.write(f"**Nivel de Saturación:** {forecast['saturacion']}")
        st.write(f"**Tiempo de Viaje Normal:** {forecast['tiempo_normal']} min")
        st.write(f"**Tiempo de Viaje Estimado (con tráfico):** {forecast['tiempo_proyectado']} min")
    
    # Tab 5: Integrar API
    with tabs[4]:
        show_integration_form()

# =============================================================================
# Pestañas Específicas: Tráfico en Tiempo Real y Pronósticos de Ruta
# =============================================================================

def tab_trafico_en_tiempo_real():
    st.subheader("Tráfico en Tiempo Real")
    st.markdown("""
    Ingresa las direcciones de **origen** y **destino** (con ciudad, estado, país) y selecciona la hora de inicio de tu viaje.
    La ruta se dibujará con segmentos coloreados según la saturación de tráfico simulada.
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
            st.error("No se pudieron geocodificar las direcciones. Verifica que estén completas.")
            return
        now = datetime.now()
        start_dt = datetime(now.year, now.month, now.day, start_time_input.hour, start_time_input.minute)
        weather = get_weather_open_meteo(origin_coords[0], origin_coords[1])
        route_data = get_route_osrm(origin_coords, dest_coords)
        if not route_data:
            st.error("No se pudo obtener la ruta con OSRM.")
            return
        segments, arrival_time = simulate_route_segments(route_data["steps"], start_dt, weather)
        m = folium.Map(location=[(origin_coords[0]+dest_coords[0])/2, (origin_coords[1]+dest_coords[1])/2],
                       zoom_start=12, tiles="OpenStreetMap")
        for seg in segments:
            folium.PolyLine(locations=[(pt[1], pt[0]) for pt in seg["coords"]],
                            color=seg["color"], weight=4).add_to(m)
        folium.Marker(origin_coords, popup=origin_full, tooltip="Origen", icon=folium.Icon(color="green")).add_to(m)
        folium.Marker(dest_coords, popup=dest_full, tooltip="Destino", icon=folium.Icon(color="red")).add_to(m)
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
    st.subheader("Pronósticos de Ruta (hasta 24 horas)")
    st.markdown("""
    Selecciona cuántas horas en el futuro deseas conocer la situación del tráfico y la ruta óptima para ese momento.
    Se muestran estimaciones de tiempo de viaje (normal vs. ajustado por tráfico) y la ruta en el mapa.
    """)
    hour_options = list(range(1,25))
    selected_hour = st.selectbox("Horas en el futuro:", hour_options, format_func=lambda x: f"{x} hora{'s' if x>1 else ''}")
    col1, col2 = st.columns(2)
    with col1:
        origin = st.text_input("Dirección de Origen (Pronóstico)", "Plaza Mayor, Madrid, España")
    with col2:
        destination = st.text_input("Dirección de Destino (Pronóstico)", "Puerta del Sol, Madrid, España")
    
    if st.button("Mostrar Pronóstico"):
        origin_coords, origin_full = geocode_address(origin)
        dest_coords, dest_full = geocode_address(destination)
        if not (origin_coords and dest_coords):
            st.error("No se pudieron geocodificar las direcciones.")
            return
        future_dt = datetime.now() + timedelta(hours=selected_hour)
        weather = get_weather_open_meteo(origin_coords[0], origin_coords[1])
        route_data = get_route_osrm(origin_coords, dest_coords)
        if not route_data:
            st.error("No se pudo obtener la ruta.")
            return
        segments, arrival_time = simulate_route_segments(route_data["steps"], future_dt, weather)
        m_future = folium.Map(location=[(origin_coords[0]+dest_coords[0])/2, (origin_coords[1]+dest_coords[1])/2],
                              zoom_start=12, tiles="OpenStreetMap")
        for seg in segments:
            folium.PolyLine(locations=[(pt[1], pt[0]) for pt in seg["coords"]],
                            color=seg["color"], weight=4).add_to(m_future)
        folium.Marker(origin_coords, popup=origin_full, tooltip="Origen", icon=folium.Icon(color="green")).add_to(m_future)
        folium.Marker(dest_coords, popup=dest_full, tooltip="Destino", icon=folium.Icon(color="red")).add_to(m_future)
        st_folium(m_future, width=700)
        total_distance = route_data["distance"] / 1000.0
        total_time = sum(seg["time_s"] for seg in segments)
        st.write(f"**Distancia Total:** {total_distance:.2f} km")
        st.write(f"**Hora de Salida (Futura):** {future_dt.strftime('%d/%m %H:%M')}")
        st.write(f"**Hora de Llegada Estimada:** {arrival_time.strftime('%H:%M')} (~{int(total_time/60)} min de viaje)")
        if weather:
            st.write(f"**Clima en Origen (aprox.):** {weather['temperature']}°C, Viento: {weather['windspeed']} km/h")

# =============================================================================
# Ejecución del Dashboard
# =============================================================================

def main_app():
    st.title("Trafiquea: Plataforma Integral para Transporte y Rutas en Tiempo Real")
    st.markdown("""
    **Bienvenido a Trafiquea.**  
    Explora nuestras funcionalidades:
      - **Tráfico en Tiempo Real:** Calcula rutas con niveles de congestión simulados.
      - **Pronósticos de Ruta:** Proyecta rutas y tiempos de viaje para hasta 24 horas en el futuro.
      - **Análisis de Datos:** Revisa tendencias y análisis históricos de demanda (funcionalidad ampliable).
      - **Simulación Avanzada:** Compara escenarios modificando parámetros.
      - **Integrar API:** Solicita la integración de esta solución en tu sistema.
    """)
    tabs = st.tabs(["Tráfico en Tiempo Real", "Pronósticos de Ruta", "Análisis de Datos", "Simulación Avanzada", "Integrar API"])
    
    with tabs[0]:
        tab_trafico_en_tiempo_real()
    with tabs[1]:
        tab_pronosticos_ruta()
    with tabs[2]:
        st.subheader("Análisis de Datos Históricos")
        analyze_historical_data()
    with tabs[3]:
        st.subheader("Simulación Avanzada de Escenarios")
        st.markdown("Modifica los parámetros para ver cómo varían los tiempos de viaje y la congestión.")
        hour_offset = st.slider("Horas en el Futuro:", min_value=1, max_value=6, value=2)
        forecast = simulate_traffic_forecast(hour_offset)
        st.write(f"**Nivel de Saturación:** {forecast['saturacion']}")
        st.write(f"**Tiempo de Viaje Normal:** {forecast['tiempo_normal']} min")
        st.write(f"**Tiempo de Viaje Estimado (con tráfico):** {forecast['tiempo_proyectado']} min")
    with tabs[4]:
        show_integration_form()

if __name__ == "__main__":
    main_app()
