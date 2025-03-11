import os
import streamlit as st
import folium
from streamlit_folium import st_folium
import requests
import numpy as np
import pandas as pd
import plotly.express as px
from datetime import datetime
import joblib
from prophet import Prophet
import lightgbm as lgb

# Intentamos importar el custom component para geolocalización (si está instalado)
try:
    from streamlit_geolocation import st_geolocation
except ImportError:
    st_geolocation = None

# -----------------------------------------------------------------------------
# CONFIGURACIÓN: Obtener API key de TomTom desde secrets.toml
# -----------------------------------------------------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

def get_tomtom_key():
    return st.secrets["tomtom"]["api_key"]

# -----------------------------------------------------------------------------
# (Opcional) Obtener la ubicación actual del usuario mediante streamlit-geolocation
# -----------------------------------------------------------------------------
def get_current_location():
    if st_geolocation is not None:
        loc = st_geolocation()
        if loc and "latitude" in loc and "longitude" in loc:
            return loc["latitude"], loc["longitude"]
    return None, None

# -----------------------------------------------------------------------------
# MODELO PROPHET (SINTÉTICO) CON REGRESSORES: distance, temp, wind, precip, cloud
# -----------------------------------------------------------------------------
@st.cache_resource
def load_prophet_model():
    model_path = os.path.join(BASE_DIR, "prophet_tomtom_model.pkl")
    try:
        model = joblib.load(model_path)
        return model
    except:
        st.warning("Entrenando modelo Prophet sintético con distance, temp, wind, precip y cloud.")
        np.random.seed(42)
        dates = pd.date_range("2023-01-01", "2023-12-31", freq="D")
        n = len(dates)
        distance = np.random.uniform(1, 50, n)
        temp = np.random.uniform(-5, 35, n)
        wind = np.random.uniform(0, 50, n)
        precip = np.random.uniform(0, 20, n)
        cloud = np.random.uniform(0, 100, n)
        # Tiempo base ficticio
        y = 60 + 2*distance + 1.5*temp + 0.3*wind + 0.5*precip + 0.1*cloud + np.random.normal(0, 15, n)
        df = pd.DataFrame({
            "ds": dates,
            "y": y,
            "distance": distance,
            "temp": temp,
            "wind": wind,
            "precip": precip,
            "cloud": cloud
        })
        m = Prophet()
        m.add_regressor("distance")
        m.add_regressor("temp")
        m.add_regressor("wind")
        m.add_regressor("precip")
        m.add_regressor("cloud")
        m.fit(df)
        joblib.dump(m, model_path)
        return m

def predict_time_prophet(distance_km: float, temp: float, wind: float, precip: float, cloud: float):
    m = load_prophet_model()
    if not m:
        return None
    df_future = pd.DataFrame({
        "ds": [pd.Timestamp.now()],
        "distance": [distance_km],
        "temp": [temp],
        "wind": [wind],
        "precip": [precip],
        "cloud": [cloud]
    })
    forecast = m.predict(df_future)
    return forecast["yhat"].iloc[0]

# -----------------------------------------------------------------------------
# OPEN-METEO: Obtener variables meteorológicas en tiempo real
# -----------------------------------------------------------------------------
@st.cache_data(ttl=3600)
def get_weather_open_meteo(lat: float, lon: float):
    url = "https://api.open-meteo.com/v1/forecast"
    params = {
        "latitude": lat,
        "longitude": lon,
        "current_weather": True,
        "hourly": "temperature_2m,windspeed_10m,precipitation,cloudcover",
        "timezone": "auto"
    }
    resp = requests.get(url, params=params)
    if resp.status_code == 200:
        data = resp.json()
        current = data.get("current_weather", {})
        return {
            "temp": current.get("temperature", 20.0),
            "wind": current.get("windspeed", 0.0),
            "precip": data.get("hourly", {}).get("precipitation", [0])[0],
            "cloud": data.get("hourly", {}).get("cloudcover", [0])[0]
        }
    return None

# -----------------------------------------------------------------------------
# TOMTOM SEARCH: Geocoding/Autocomplete
# -----------------------------------------------------------------------------
@st.cache_data(ttl=600)
def tomtom_search(query: str, limit=5):
    if not query:
        return []
    tomtom_key = get_tomtom_key()
    url = f"https://api.tomtom.com/search/2/search/{query}.json"
    params = {
        "key": tomtom_key,
        "limit": limit,
        "language": "es-ES"
    }
    r = requests.get(url, params=params)
    if r.status_code == 200:
        data = r.json()
        suggestions = []
        for item in data.get("results", []):
            address_str = item.get("address", {}).get("freeformAddress", "")
            position = item.get("position", {})
            lat = position.get("lat")
            lon = position.get("lon")
            if lat and lon:
                suggestions.append((address_str, lat, lon))
        return suggestions
    return []

# Funciones para actualizar autocompletado al presionar Enter
def update_origin():
    query = st.session_state.origin_query
    results = tomtom_search(query)
    st.session_state.origin_options = [r[0] for r in results] if results else []
    st.session_state.origin_results = results

def update_dest():
    query = st.session_state.dest_query
    results = tomtom_search(query)
    st.session_state.dest_options = [r[0] for r in results] if results else []
    st.session_state.dest_results = results

def address_input_autocomplete(label: str, key: str):
    # El usuario escribe y, al pulsar Enter (on_change), se actualizan las sugerencias
    addr = st.text_input(f"Escribe la dirección {label}", key=key, on_change=update_origin if key=="origin_query" else update_dest)
    options_key = "origin_options" if key=="origin_query" else "dest_options"
    results_key = "origin_results" if key=="origin_query" else "dest_results"
    suggestions = st.session_state.get(options_key, [])
    if suggestions:
        choice = st.selectbox(f"Sugerencias para {label}", suggestions, key=f"sel_{key}")
        for r in st.session_state.get(results_key, []):
            if r[0] == choice:
                return r[0], r[1], r[2]
    return None, None, None

# -----------------------------------------------------------------------------
# TOMTOM ROUTING API
# -----------------------------------------------------------------------------
@st.cache_data(ttl=600)
def tomtom_routing_api(origin_lat, origin_lon, dest_lat, dest_lon, vehicle_type="car"):
    tomtom_key = get_tomtom_key()
    url = f"https://api.tomtom.com/routing/1/calculateRoute/{origin_lat},{origin_lon}:{dest_lat},{dest_lon}/json"
    params = {
        "key": tomtom_key,
        "traffic": "true",
        "travelMode": vehicle_type
    }
    if vehicle_type == "truck":
        params["vehicleCommercial"] = "true"
        params["vehicleMaxHeight"] = 4.0
        params["vehicleWeight"] = 20000
    r = requests.get(url, params=params)
    if r.status_code == 200:
        return r.json()
    return None

# -----------------------------------------------------------------------------
# TOMTOM WAYPOINT OPTIMIZATION API
# -----------------------------------------------------------------------------
@st.cache_data(ttl=600)
def tomtom_waypoint_optimization(coords_list):
    tomtom_key = get_tomtom_key()
    url = f"https://api.tomtom.com/routing/1/waypointOptimization?key={tomtom_key}"
    locations = [{"point": {"latitude": lat, "longitude": lon}} for lat, lon in coords_list]
    body = {"locations": locations, "options": {"computeBestOrder": True}}
    r = requests.post(url, json=body)
    if r.status_code == 200:
        return r.json()
    return None

# -----------------------------------------------------------------------------
# TOMTOM TRAFFIC API (Flow)
# -----------------------------------------------------------------------------
@st.cache_data(ttl=600)
def tomtom_traffic_flow(lat: float, lon: float):
    tomtom_key = get_tomtom_key()
    url = "https://api.tomtom.com/traffic/services/4/flowSegmentData/relative0/10/json"
    params = {"key": tomtom_key, "point": f"{lat},{lon}"}
    r = requests.get(url, params=params)
    if r.status_code == 200:
        return r.json()
    return None

# -----------------------------------------------------------------------------
# CALCULADORA CAE
# -----------------------------------------------------------------------------
def calculate_cae(kwh: float):
    cost_min = kwh * 0.115
    cost_max = kwh * 0.14
    return kwh, cost_min, cost_max

# -----------------------------------------------------------------------------
# RENDERIZAR MAPA CON FOLIUM
# -----------------------------------------------------------------------------
def render_map(route_points, origin_lat, origin_lon, dest_lat=None, dest_lon=None):
    center_lat = origin_lat if dest_lat is None else (origin_lat + dest_lat) / 2
    center_lon = origin_lon if dest_lon is None else (origin_lon + dest_lon) / 2
    m_map = folium.Map(location=[center_lat, center_lon], zoom_start=12)
    tile_url = f"https://api.tomtom.com/map/1/tile/basic/main/{{z}}/{{x}}/{{y}}.png?key={get_tomtom_key()}"
    folium.TileLayer(tiles=tile_url, attr="TomTom", name="TomTom Map").add_to(m_map)
    folium.PolyLine(route_points, color="blue", weight=5).add_to(m_map)
    folium.Marker((origin_lat, origin_lon), tooltip="Origen", icon=folium.Icon(color="green")).add_to(m_map)
    if dest_lat and dest_lon:
        folium.Marker((dest_lat, dest_lon), tooltip="Destino", icon=folium.Icon(color="red")).add_to(m_map)
    st_folium(m_map, width=700)

# -----------------------------------------------------------------------------
# TAB: Mapa y Tráfico
# -----------------------------------------------------------------------------
def tab_mapa_trafico():
    st.subheader("Mapa y Tráfico")
    
    # Mostrar el mapa centrado en la ubicación actual del usuario (si se obtiene)
    current_lat, current_lon = get_current_location()
    if current_lat and current_lon:
        st.info("Mostrando ubicación actual.")
        m_current = folium.Map(location=[current_lat, current_lon], zoom_start=13)
        tile_url = f"https://api.tomtom.com/map/1/tile/basic/main/{{z}}/{{x}}/{{y}}.png?key={get_tomtom_key()}"
        folium.TileLayer(tiles=tile_url, attr="TomTom", name="TomTom Map").add_to(m_current)
        folium.Marker((current_lat, current_lon), tooltip="Tu ubicación", icon=folium.Icon(color="orange")).add_to(m_current)
        st_folium(m_current, width=700)
    
    st.write("Escribe la dirección de Origen y Destino, luego presiona Enter en cada uno para ver sugerencias.")
    origin_addr, origin_lat, origin_lon = address_input_autocomplete("Origen", "origin_query")
    dest_addr, dest_lat, dest_lon = address_input_autocomplete("Destino", "dest_query")
    
    if origin_lat and origin_lon and dest_lat and dest_lon:
        vehicle_type = st.selectbox("Tipo de Vehículo", ["car", "truck", "van", "taxi"], key="veh_type")
        if st.button("Calcular Ruta TomTom"):
            routing_data = tomtom_routing_api(origin_lat, origin_lon, dest_lat, dest_lon, vehicle_type=vehicle_type)
            if not routing_data or "routes" not in routing_data:
                st.error("No se pudo obtener la ruta con TomTom Routing API.")
                return
            route = routing_data["routes"][0]
            summary = route["summary"]
            distance_m = summary["lengthInMeters"]
            duration_s = summary["travelTimeInSeconds"]
            st.write(f"Distancia: {distance_m/1000:.2f} km")
            st.write(f"Duración (base TomTom): ~{int(duration_s/60)} min")
            st.session_state["origin_lat"] = origin_lat
            st.session_state["origin_lon"] = origin_lon
            st.session_state["dest_lat"] = dest_lat
            st.session_state["dest_lon"] = dest_lon
            st.session_state["distance_km"] = distance_m / 1000.0
            st.session_state["duration_min"] = duration_s / 60.0
            route_points = []
            for leg in route["legs"]:
                for point in leg["points"]:
                    route_points.append((point["latitude"], point["longitude"]))
            render_map(route_points, origin_lat, origin_lon, dest_lat, dest_lon)

# -----------------------------------------------------------------------------
# TAB: Optimización Múltiples Paradas
# -----------------------------------------------------------------------------
def tab_optimizar_paradas():
    st.subheader("Optimización Múltiples Paradas")
    addresses_text = st.text_area("Direcciones (una por línea):", 
                                  "Plaza Mayor, Madrid\nPuerta del Sol, Madrid\nAtocha, Madrid")
    if st.button("Optimizar Orden"):
        lines = [l.strip() for l in addresses_text.split("\n") if l.strip()]
        if len(lines) < 2:
            st.error("Ingresa al menos 2 direcciones.")
            return
        coords_list = []
        for line in lines:
            results = tomtom_search(line)
            if not results:
                st.warning(f"No se encontró: {line}")
                return
            coords_list.append((results[0][1], results[0][2]))
        data_opt = tomtom_waypoint_optimization(coords_list)
        if not data_opt or "routes" not in data_opt:
            st.error("No se pudo optimizar con TomTom Waypoint Optimization API.")
            return
        route = data_opt["routes"][0]
        order = route.get("waypointsOrder", "No definido")
        st.success(f"Orden óptimo de paradas: {order}")
        route_points = []
        for leg in route["legs"]:
            for point in leg["points"]:
                route_points.append((point["latitude"], point["longitude"]))
        render_map(route_points, coords_list[0][0], coords_list[0][1])
        
# -----------------------------------------------------------------------------
# TAB: Tráfico en Ruta
# -----------------------------------------------------------------------------
def tab_trafico_ruta():
    st.subheader("Tráfico en Ruta")
    if "origin_lat" not in st.session_state or "dest_lat" not in st.session_state:
        st.warning("Primero calcula la ruta en 'Mapa y Tráfico'.")
        return
    lat_m = (st.session_state["origin_lat"] + st.session_state["dest_lat"]) / 2
    lon_m = (st.session_state["origin_lon"] + st.session_state["dest_lon"]) / 2
    if st.button("Consultar Tráfico en Punto Medio"):
        flow_data = tomtom_traffic_flow(lat_m, lon_m)
        if not flow_data or "flowSegmentData" not in flow_data:
            st.error("No se pudo obtener datos de tráfico.")
            return
        flow = flow_data["flowSegmentData"]
        current_speed = flow.get("currentSpeed", 0)
        free_flow_speed = flow.get("freeFlowSpeed", 0)
        st.write(f"Velocidad actual: {current_speed} km/h")
        st.write(f"Velocidad libre: {free_flow_speed} km/h")
        st.write(f"Tiempo actual: {flow.get('currentTravelTime')} s")
        st.write(f"Tiempo libre: {flow.get('freeFlowTravelTime')} s")
        st.write(f"Confianza: {flow.get('confidence')}")
        if current_speed < 20:
            st.error("Tráfico muy congestionado. Se recomienda buscar ruta alternativa o salir en otro momento.")
        elif current_speed < 50:
            st.warning("Tráfico moderado. Podrías experimentar retrasos.")
        else:
            st.info("Tráfico fluido.")

# -----------------------------------------------------------------------------
# TAB: Predicción de Retrasos (Prophet)
# -----------------------------------------------------------------------------
def tab_prediccion_demanda():
    st.subheader("Predicción de Retrasos")
    if "distance_km" not in st.session_state or "duration_min" not in st.session_state:
        st.warning("Primero calcula la ruta en 'Mapa y Tráfico'.")
        return
    distance_km = st.session_state["distance_km"]
    duration_min = st.session_state["duration_min"]

    lat_m = (st.session_state["origin_lat"] + st.session_state["dest_lat"]) / 2
    lon_m = (st.session_state["origin_lon"] + st.session_state["dest_lon"]) / 2

    meteo = get_weather_open_meteo(lat_m, lon_m)
    if not meteo:
        st.warning("No se pudo obtener datos meteorológicos. Se usan valores por defecto.")
        temp, wind, precip, cloud = 20.0, 0.0, 0.0, 0.0
    else:
        temp = meteo["temp"]
        wind = meteo["wind"]
        precip = meteo["precip"]
        cloud = meteo["cloud"]
        st.write(f"Clima: Temp: {temp:.1f}°C, Viento: {wind:.1f} km/h, Precipitación: {precip:.1f} mm, Nubosidad: {cloud:.1f}%")
    
    st.write(f"Distancia: {distance_km:.2f} km")
    st.write(f"Tiempo base (TomTom): ~{duration_min:.2f} min")
    
    if st.button("Calcular Predicción"):
        predicted_time = predict_time_prophet(distance_km, temp, wind, precip, cloud)
        if predicted_time is None:
            st.error("Error en la predicción con Prophet.")
            return
        st.success(f"Tiempo estimado (modelo): {predicted_time:.2f} min")
        delay = predicted_time - duration_min
        if delay > 0:
            st.error(f"Posible retraso: +{delay:.2f} min")
        else:
            st.info(f"Podrías llegar {abs(delay):.2f} min antes de lo esperado.")

# -----------------------------------------------------------------------------
# TAB: Calculadora CAE
# -----------------------------------------------------------------------------
def tab_calculadora_cae():
    st.subheader("Calculadora CAE")
    st.write("Introduce los kWh ahorrados para estimar ingresos (0.115–0.14 €/kWh).")
    kwh = st.number_input("kWh ahorrados", min_value=0.0, value=1000.0, step=100.0)
    if st.button("Calcular CAE"):
        cae, cost_min, cost_max = calculate_cae(kwh)
        st.write(f"CAE generados: {cae:.2f} kWh")
        st.write(f"Ingresos estimados: entre {cost_min:.2f} € y {cost_max:.2f} €")

# -----------------------------------------------------------------------------
# APP PRINCIPAL
# -----------------------------------------------------------------------------
def main_app():
    st.title("Trafiquea: Dashboard para Empresas Logísticas")
    st.markdown("""
    **Características Principales**:
    - **Map Display & Routing**: Calcula rutas y muestra mapas con tiles de TomTom.  
    - **Waypoint Optimization**: Ordena múltiples paradas a partir de direcciones ingresadas.  
    - **Traffic API**: Consulta el flujo de tráfico y ofrece consejos en función de la congestión.  
    - **Predicción de Retrasos**: Usa variables meteorológicas y un modelo Prophet para estimar retrasos en la ruta.  
    - **Calculadora CAE**: Estima ingresos por kWh ahorrados.
    """)
    tabs = st.tabs([
        "Mapa y Tráfico",
        "Optimización Múltiples Paradas",
        "Tráfico en Ruta",
        "Predicción de Retrasos",
        "Calculadora CAE"
    ])
    with tabs[0]:
        tab_mapa_trafico()
    with tabs[1]:
        tab_optimizar_paradas()
    with tabs[2]:
        tab_trafico_ruta()
    with tabs[3]:
        tab_prediccion_demanda()
    with tabs[4]:
        tab_calculadora_cae()

if __name__ == "__main__":
    main_app()
