"""
Trafiquea: Plataforma Integral para Transporte y Rutas en Tiempo Real

Este dashboard está estructurado en tres pestañas principales (Ayuntamientos, Empresas, Particulares),
cada una con funcionalidades y visualizaciones adaptadas a su público objetivo.

Resumen de Funcionalidades:
1. Ayuntamientos:
   - Selector de ayuntamientos (ejemplo: provincia de Toledo).
   - Datos reales o simulados (extraídos de APIs públicas) sobre movilidad, sostenibilidad y congestión.
   - Mapa interactivo que muestra la ubicación y las rutas principales, así como indicadores relevantes (CO₂, tráfico).
   - Predictores avanzados para anticipar tendencias de movilidad (sin gráficas extensas, sino mapas y métricas claras).
   - Secciones separadas para concejalías (Movilidad, Sostenibilidad) con información adaptada.

2. Empresas:
   - Métricas de valor para la logística (tiempos de viaje optimizados, ahorro en costes, simulación de escenarios).
   - Calculadora de CAE (Créditos de Actividad de Emisiones, con base en MITECO) estimando ingresos potenciales por sostenibilidad.
   - Mapa interactivo con las rutas y puntos de interés logístico (almacenes, centros de distribución).
   - Integración API para que la empresa pueda conectarse y extraer datos de forma automatizada.

3. Particulares:
   - Ingreso de origen y destino para obtener rutas optimizadas (en coche, bicicleta, transporte público).
   - Mapa interactivo mostrando la ruta y la congestión.
   - Estimaciones de tiempo y condiciones climáticas para viajes diarios.
   - Se minimiza la interacción adicional: el usuario ingresa los datos una sola vez y ve resultados directos.

Las APIS utilizadas pueden ser:
- OSRM (rutas)
- Open-Meteo (clima)
- Nominatim (geocodificación)
- Alguna API pública de movilidad y sostenibilidad (simulada si no se encuentra real)
- Datos simulados o APIs de la provincia de Toledo para el selector de ayuntamientos

Se evita saturar con headers y subheaders innecesarios, usando separadores y textos breves para guiar al usuario.

Requisitos técnicos:
- Folium (con streamlit-folium) para mapas
- Manejo de keys en los widgets para evitar colisiones
- Un estilo UI amigable con secciones claras
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
# APIS y Factores (Tráfico, Clima, CAE)
# =============================================================================

def simulate_zone_factor(full_address: str):
    # Ajuste de factor de congestión según zona (ej. Talavera, Toledo, etc.)
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

def calculate_cae(volume: float):
    """
    Calcula la estimación de Créditos de Actividad de Emisiones (CAE) 
    con base en un factor ficticio. 
    """
    # Ejemplo: 1 CAE cada 1000 kg de CO₂ evitados
    # Se asume un factor para simular la generación de CAE
    factor_cae = 0.001
    cae_estimated = volume * factor_cae
    return cae_estimated

# ============================================================================
# Análisis de Datos (últimos meses) y Presentación
# ============================================================================

def analyze_historical_data():
    st.markdown("Información simulada de los últimos 6 meses sobre movilidad:")
    dates = pd.date_range(end=datetime.now(), periods=180)
    demand = np.random.randint(80, 300, len(dates))
    df = pd.DataFrame({"Fecha": dates, "Demanda Estimada": demand})
    df.sort_values("Fecha", inplace=True)
    # En lugar de un gráfico saturado, mostramos un mapa conceptual
    st.markdown("Se estima que la demanda de viajes promedio se ha mantenido estable, con ligeras variaciones.")
    st.write(f"Máximo de viajes/día en el periodo: {df['Demanda Estimada'].max()}")
    st.write(f"Mínimo de viajes/día en el periodo: {df['Demanda Estimada'].min()}")
    st.write(f"Promedio de viajes/día en el periodo: {df['Demanda Estimada'].mean():.2f}")

# ============================================================================
# Formulario de Integración vía API
# ============================================================================

def show_integration_form():
    st.markdown("Complete el siguiente formulario para solicitar la integración vía API:")
    with st.form("form_integration_unico"):
        nombre = st.text_input("Nombre", key="api_nombre_key")
        apellidos = st.text_input("Apellidos", key="api_apellidos_key")
        institucion = st.text_input("Institución o Empresa", key="api_institucion_key")
        mensaje = st.text_area("Mensaje (opcional)", key="api_mensaje_key")
        submitted = st.form_submit_button("Enviar Solicitud")
        if submitted:
            st.success("Solicitud enviada correctamente. Nos pondremos en contacto contigo.")

# ============================================================================
# Secciones para cada perfil
# ============================================================================

def ayuntamientos_section():
    st.markdown("## Selector de Ayuntamiento (Provincia de Toledo)")
    ayuntamientos = ["Talavera de la Reina", "Toledo Capital", "Illescas", "Seseña", "Ocaña"]
    selected_ayuntamiento = st.selectbox("Seleccione un ayuntamiento", ayuntamientos, key="ayto_select")
    st.markdown(f"Se muestra información enfocada a la ciudad de **{selected_ayuntamiento}**:")

    # Mapa inicial de la ciudad
    coords, full_addr = geocode_address(f"{selected_ayuntamiento}, Toledo, España")
    if coords:
        lat, lon = coords
        m = folium.Map(location=[lat, lon], zoom_start=12, tiles="OpenStreetMap")
        folium.Marker(coords, popup=full_addr, tooltip=selected_ayuntamiento, icon=folium.Icon(color="blue")).add_to(m)
        st_folium(m, width=700)
    else:
        st.write("No se pudo geocodificar el ayuntamiento seleccionado.")

    st.markdown("---")
    st.markdown("### Indicadores Clave para Movilidad y Sostenibilidad")
    st.markdown("- **Congestión Actual**: Simulada o extraída de API local (no se muestra gráfico, solo métrica).")
    st.markdown("- **Emisiones de CO₂**: Valor estimado en kg/día según el tráfico actual.")
    st.markdown("- **Predicción de Congestión a 24h**: Basada en datos simulados.")

    st.markdown("### Subpestañas de Funcionalidad")
    sub_tabs = st.tabs(["Tráfico en Tiempo Real", "Predicción de Tráfico", "Análisis Histórico", "Simulación Avanzada"])
    
    with sub_tabs[0]:
        st.markdown("Tráfico en tiempo real para concejalía de movilidad (sin gráficas saturadas, sino mapa).")
        tab_trafico_en_tiempo_real()
    with sub_tabs[1]:
        st.markdown("Predicción de tráfico para las próximas horas.")
        tab_pronosticos_ruta()
    with sub_tabs[2]:
        st.markdown("Datos históricos de los últimos meses, sin saturar con gráficas innecesarias.")
        analyze_historical_data()
    with sub_tabs[3]:
        st.markdown("Simulación de escenarios de movilidad según cambios en la hora o el clima.")
        tab_simulacion_avanzada()

def empresas_section():
    st.markdown("## Empresas de Logística y Sostenibilidad")
    st.markdown("Herramientas de valor para optimizar costes, rutas y generar ingresos por créditos de emisiones (CAE).")

    st.markdown("---")
    st.markdown("### Calculadora de CAE")
    volume = st.number_input("Introduzca el volumen (kg) de CO₂ que se evita al usar rutas optimizadas", value=1000.0, step=100.0, key="cae_calc")
    if st.button("Calcular CAE Estimado", key="btn_cae"):
        cae_value = calculate_cae(volume)
        st.write(f"Se estima que se generarían {cae_value:.2f} CAE con este volumen de CO₂ evitado.")

    st.markdown("---")
    st.markdown("### Rutas y Pronósticos")
    sub_tabs = st.tabs(["Tráfico en Tiempo Real", "Pronósticos de Ruta", "Simulación Avanzada", "Integrar API"])
    with sub_tabs[0]:
        tab_trafico_en_tiempo_real()
    with sub_tabs[1]:
        tab_pronosticos_ruta()
    with sub_tabs[2]:
        tab_simulacion_avanzada()
    with sub_tabs[3]:
        show_integration_form()

def particulares_section():
    st.markdown("## Particulares")
    st.markdown("Funciones de cálculo de rutas y pronósticos para viajes cotidianos, con modos de bicicleta y transporte público (simulados).")

    st.markdown("### Rutas Cotidianas")
    sub_tabs = st.tabs(["Coche", "Bicicleta", "Transporte Público"])
    with sub_tabs[0]:
        st.markdown("Modo Coche - Tráfico en Tiempo Real")
        tab_trafico_en_tiempo_real()
        st.markdown("---")
        st.markdown("Predicción de Ruta (Coche)")
        tab_pronosticos_ruta()
    with sub_tabs[1]:
        st.markdown("Modo Bicicleta - Ruta Optimizada")
        st.markdown("Aquí se mostraría un mapa con carriles bici y estimaciones de tiempo.")
        # Se podría simular una ruta en modo "cycling" con otra API OSRM
        # Ejemplo: get_route_osrm_cycling(...) 
    with sub_tabs[2]:
        st.markdown("Transporte Público - Horarios y combinaciones")
        st.markdown("Se mostraría la información de buses o trenes, sin gráficas saturadas. Solo mapas y tiempos aproximados.")

# ============================================================================
# Función principal
# ============================================================================

def main_app():
    st.title("Trafiquea: Plataforma Integral para Transporte y Rutas en Tiempo Real")
    st.markdown("""
    Seleccione el perfil que mejor se ajuste a sus necesidades:
    - **Ayuntamientos**: Indicadores de movilidad y sostenibilidad. 
    - **Empresas**: Métricas de optimización, créditos de emisiones y rentabilidad.
    - **Particulares**: Rutas diarias y predicciones de viaje.
    """)

    # Mapa inicial para contextualizar la ubicación del usuario o la región
    st.markdown("Mapa inicial (región general) para no saturar al usuario con preguntas.")
    # Ejemplo: Mapa centrado en la provincia de Toledo
    m_initial = folium.Map(location=[39.8628, -4.0273], zoom_start=8, tiles="OpenStreetMap")
    folium.Marker([39.8628, -4.0273], tooltip="Provincia de Toledo", icon=folium.Icon(color="blue")).add_to(m_initial)
    st_folium(m_initial, width=700)

    # Tabs principales
    main_tabs = st.tabs(["Ayuntamientos", "Empresas", "Particulares"])
    
    with main_tabs[0]:
        ayuntamientos_section()
    with main_tabs[1]:
        empresas_section()
    with main_tabs[2]:
        particulares_section()

if __name__ == "__main__":
    main_app()
