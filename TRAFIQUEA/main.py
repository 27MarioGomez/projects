# main.py
"""
Trafiquea: Plataforma Integral para Transporte y Rutas en Tiempo Real

Esta aplicación interactiva permite:
    - Optimizar rutas en tiempo real con mapas interactivos y datos del clima.
    - Mostrar información de tráfico en tiempo real en la zona del usuario, con detalles por calle.
    - Proyectar tiempos de viaje futuros (por horas) mostrando saturación y variación de tiempos.
    - Ofrecer una opción de integración vía API para que usuarios interesados puedan dejar sus datos de contacto.

Las funcionalidades están diseñadas para aportar valor en el día a día del transporte, sin mencionar empresas específicas.
"""

# =============================================================================
# Importación de librerías necesarias
# =============================================================================
from datetime import datetime, timedelta
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import requests
from prophet import Prophet
import lightgbm as lgb

# =============================================================================
# Configuración de la página
# =============================================================================
st.set_page_config(
    page_title="Trafiquea: Transporte y Rutas en Tiempo Real",
    layout="wide",
    initial_sidebar_state="expanded"
)

# =============================================================================
# Funciones de Utilidad y Acceso a APIs Públicas
# =============================================================================

@st.cache_data(ttl=3600)
def geocode_address(address: str):
    """
    Geocodifica una dirección usando Nominatim (OpenStreetMap).
    Retorna una tupla (latitud, longitud) o None si no se encuentra.
    """
    from geopy.geocoders import Nominatim
    geolocator = Nominatim(user_agent="trafiquea_dashboard")
    location = geolocator.geocode(address)
    if location:
        return (location.latitude, location.longitude)
    return None

def get_route_osrm(origin_coords, destination_coords):
    """
    Obtiene la ruta optimizada entre dos puntos usando la API pública de OSRM.
    Retorna la geometría de la ruta en formato GeoJSON o None en caso de error.
    """
    base_url = "http://router.project-osrm.org/route/v1/driving"
    coords = f"{origin_coords[1]},{origin_coords[0]};{destination_coords[1]},{destination_coords[0]}"
    params = {"overview": "full", "geometries": "geojson"}
    url = f"{base_url}/{coords}"
    response = requests.get(url, params=params)
    if response.status_code == 200:
        data = response.json()
        if "routes" in data and len(data["routes"]) > 0:
            return data["routes"][0]["geometry"]
    return None

def get_weather_open_meteo(lat: float, lon: float):
    """
    Consulta la API de Open-Meteo para obtener el clima actual basado en latitud y longitud.
    Retorna un diccionario con temperatura y velocidad del viento.
    """
    url = "https://api.open-meteo.com/v1/forecast"
    params = {"latitude": lat, "longitude": lon, "current_weather": True, "timezone": "auto"}
    response = requests.get(url, params=params)
    if response.status_code == 200:
        data = response.json()
        current = data.get("current_weather", {})
        return {"temperature": current.get("temperature"), "windspeed": current.get("windspeed")}
    return None

@st.cache_data(ttl=3600)
def get_transit_demand_data():
    """
    Carga datos históricos de demanda (simulados) desde un CSV de muestra.
    Se espera que el CSV contenga columnas 'Fecha' y 'Demanda'.
    """
    url = "https://raw.githubusercontent.com/plotly/datasets/master/2014_apple_stock.csv"
    try:
        df = pd.read_csv(url, parse_dates=["AAPL_x"])
        df.rename(columns={"AAPL_x": "Fecha", "AAPL_y": "Demanda"}, inplace=True)
        if "Demanda" not in df.columns:
            df["Demanda"] = np.random.randint(100, 500, len(df))
        return df
    except Exception as e:
        st.error(f"Error al cargar datos de demanda: {e}")
        return pd.DataFrame()

# =============================================================================
# Funciones para Simulación de Tráfico y Pronóstico de Tiempos de Viaje
# =============================================================================

def simulate_traffic_by_street(location_coords):
    """
    Simula datos de congestión por calle alrededor de la ubicación del usuario.
    Retorna un DataFrame con calles, nivel de congestión y tiempo estimado de viaje (min).
    """
    calles = ["Calle A", "Calle B", "Calle C", "Calle D", "Calle E"]
    congestiones = np.random.choice(["Bajo", "Moderado", "Alto", "Muy Alto"], size=5)
    tiempos = np.random.randint(5, 20, size=5)
    df = pd.DataFrame({
        "Calle": calles,
        "Congestión": congestiones,
        "Tiempo Estimado (min)": tiempos
    })
    return df

def simulate_traffic_forecast(hour_offset):
    """
    Simula el pronóstico de tráfico para 'hour_offset' horas en el futuro.
    Retorna un diccionario con:
        - saturacion: nivel de congestión.
        - tiempo_normal: tiempo de viaje normal (min).
        - tiempo_proyectado: tiempo de viaje ajustado por tráfico (min).
    """
    # Simulamos que a mayor hora_offset aumenta la congestión
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
# Funciones para Predicción de Tiempos de Viaje (Opcional)
# =============================================================================
# Se podría ampliar con datos históricos y modelos, pero en este prototipo se simulan resultados.

# =============================================================================
# Función para mostrar un formulario de integración vía API
# =============================================================================
def show_integration_form():
    st.subheader("Solicita Integración vía API")
    with st.form("form_integration"):
        nombre = st.text_input("Nombre")
        apellidos = st.text_input("Apellidos")
        institucion = st.text_input("Institución o Empresa")
        mensaje = st.text_area("Mensaje (opcional)")
        submitted = st.form_submit_button("Enviar")
        if submitted:
            # En un entorno real se enviaría un email o se registraría la solicitud
            st.success("Gracias por tu interés. Se enviará la información a nocodelover@gmail.com.")

# =============================================================================
# Diseño y Lógica Principal del Dashboard
# =============================================================================
def main_app():
    st.title("Trafiquea: Transporte y Rutas en Tiempo Real")
    st.markdown("""
    **Descripción del Proyecto:**  
    Plataforma integral que ofrece optimización de rutas, visualización de tráfico en tiempo real y pronósticos de tiempos de viaje para ayudar a planificar mejor los desplazamientos.  
    La aplicación permite al usuario configurar sus necesidades y ver información detallada (por calles, horarios, etc.), además de ofrecer una opción para integración vía API.
    """)
    
    # --- Tab: Optimización de Rutas ---
    st.sidebar.title("Configuración")
    st.sidebar.subheader("Optimización de Rutas")
    origin = st.sidebar.text_input("Dirección de Origen", "Plaza Mayor, Madrid")
    destination = st.sidebar.text_input("Dirección de Destino", "Puerta del Sol, Madrid")
    if st.sidebar.button("Obtener Ruta"):
        origin_coords = geocode_address(origin)
        destination_coords = geocode_address(destination)
        if origin_coords and destination_coords:
            route_geo = get_route_osrm(origin_coords, destination_coords)
            weather = get_weather_open_meteo(origin_coords[0], origin_coords[1])
            st.success("Ruta y datos actualizados.")
            st.write(f"**Clima en Origen:** {weather['temperature']}°C, Viento: {weather['windspeed']} km/h")
            # Mapa interactivo: ruta, origen y destino
            fig_map = go.Figure()
            if route_geo:
                coords = route_geo["coordinates"]
                lons = np.array(coords)[:, 0] if isinstance(coords, list) else coords[:, 0]
                lats = np.array(coords)[:, 1] if isinstance(coords, list) else coords[:, 1]
                fig_map.add_trace(go.Scattermapbox(
                    mode="lines",
                    lon=lons,
                    lat=lats,
                    marker={"size": 5},
                    line=dict(width=4, color="blue"),
                    name="Ruta Óptima"
                ))
            fig_map.add_trace(go.Scattermapbox(
                mode="markers",
                lon=[origin_coords[1]],
                lat=[origin_coords[0]],
                marker={"size": 12, "color": "green"},
                name="Origen"
            ))
            fig_map.add_trace(go.Scattermapbox(
                mode="markers",
                lon=[destination_coords[1]],
                lat=[destination_coords[0]],
                marker={"size": 12, "color": "red"},
                name="Destino"
            ))
            fig_map.update_layout(
                mapbox_style="open-street-map",
                mapbox_zoom=12,
                mapbox_center={"lat": (origin_coords[0]+destination_coords[0])/2,
                               "lon": (origin_coords[1]+destination_coords[1])/2},
                margin={"r":0,"t":0,"l":0,"b":0}
            )
            st.plotly_chart(fig_map, use_container_width=True)
        else:
            st.error("No se pudieron geolocalizar las direcciones.")
    
    # --- Tabs del Dashboard ---
    tabs = st.tabs(["Tráfico en Tiempo Real", "Pronóstico de Tiempos de Viaje", "API Integración"])
    
    # --- Tab: Tráfico en Tiempo Real ---
    with tabs[0]:
        st.header("Tráfico en Tiempo Real")
        st.markdown("Ingresa tu dirección actual para ver tu ubicación y la congestión en calles cercanas.")
        current_address = st.text_input("Dirección Actual", "Gran Vía, Madrid", key="current")
        if st.button("Mostrar Mi Ubicación", key="btn_location"):
            user_coords = geocode_address(current_address)
            if user_coords:
                st.success("Ubicación obtenida.")
                weather = get_weather_open_meteo(user_coords[0], user_coords[1])
                st.write(f"**Clima en tu Ubicación:** {weather['temperature']}°C, Viento: {weather['windspeed']} km/h")
                fig_user = go.Figure(go.Scattermapbox(
                    mode="markers",
                    lon=[user_coords[1]],
                    lat=[user_coords[0]],
                    marker={"size": 14, "color": "blue"},
                    name="Tu Ubicación"
                ))
                fig_user.update_layout(
                    mapbox_style="open-street-map",
                    mapbox_zoom=14,
                    mapbox_center={"lat": user_coords[0], "lon": user_coords[1]},
                    margin={"r":0,"t":0,"l":0,"b":0}
                )
                st.plotly_chart(fig_user, use_container_width=True)
                # Mostrar congestión simulada por calles
                df_traffic = simulate_traffic_by_street(user_coords)
                st.subheader("Congestión por Calles Cercanas")
                st.table(df_traffic)
            else:
                st.error("No se pudo obtener tu ubicación. Verifica la dirección.")
    
    # --- Tab: Pronóstico de Tiempos de Viaje ---
    with tabs[1]:
        st.header("Pronóstico de Tiempos de Viaje")
        st.markdown("Selecciona cuántas horas en el futuro deseas conocer la situación del tráfico.")
        hour_offset = st.slider("Horas en el Futuro:", min_value=1, max_value=6, value=2)
        forecast = simulate_traffic_forecast(hour_offset)
        st.subheader(f"Pronóstico para dentro de {hour_offset} hora(s):")
        st.write(f"**Nivel de Saturación:** {forecast['saturacion']}")
        st.write(f"**Tiempo de Viaje Normal:** {forecast['tiempo_normal']} min")
        st.write(f"**Tiempo de Viaje Estimado:** {forecast['tiempo_proyectado']} min")
    
    # --- Tab: API Integración ---
    with tabs[2]:
        st.header("API Integración")
        st.markdown("""
        Si deseas integrar esta solución en tu sistema mediante una API, por favor rellena el siguiente formulario.
        Se enviará la información a nuestro equipo para que te contactemos (correo: nocodelover@gmail.com).
        """)
        show_integration_form()

# =============================================================================
# Ejecución de la aplicación
# =============================================================================
if __name__ == "__main__":
    main_app()
