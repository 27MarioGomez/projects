import os
import streamlit as st
import folium
from streamlit_folium import st_folium
import requests
import numpy as np
import pandas as pd
import plotly.express as px
from datetime import datetime
from geopy.geocoders import Nominatim
import joblib
import lightgbm as lgb
from prophet import Prophet

# --------------------------------------------------------------------------------
# CONFIGURACIÓN: 
# Se asume que tu API key de TomTom está en secrets.toml:
# [tomtom]
# api_key = "TU_API_KEY"
# No se usa carpeta data, ni CSV locales.
# --------------------------------------------------------------------------------

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

def get_tomtom_key():
    return st.secrets["tomtom"]["api_key"]

# --------------------------------------------------------------------------------
# ENTRENAR UN MODELO PROPHET REAL CON REGRESORES (distance, temp)
# DATOS SINTÉTICOS PARA EJEMPLO
# --------------------------------------------------------------------------------
@st.cache_resource
def train_prophet_model():
    model_path = os.path.join(BASE_DIR, "prophet_tomtom_model.pkl")
    try:
        m = joblib.load(model_path)
        return m
    except:
        st.warning("Entrenando modelo Prophet (sintético) con distance y temp como regressors.")
        np.random.seed(42)
        # Generar datos diarios en 2023
        dates = pd.date_range("2023-01-01", "2023-12-31", freq="D")
        n = len(dates)
        distance = np.random.uniform(1, 50, n)  # km
        temp = np.random.uniform(-5, 35, n)
        # y = 100 + 2*distance + 3*temp + ruido
        y = 100 + 2*distance + 3*temp + np.random.normal(0, 30, n)
        df = pd.DataFrame({"ds": dates, "y": y, "distance": distance, "temp": temp})
        
        m = Prophet()
        m.add_regressor("distance")
        m.add_regressor("temp")
        m.fit(df)
        joblib.dump(m, model_path)
        return m

def predict_demand_prophet(distance_km: float, temperature: float):
    """
    Carga el modelo Prophet entrenado con 'distance' y 'temp' como regressors.
    Predice la demanda para la fecha actual (hoy).
    """
    m = train_prophet_model()
    if not m:
        return None
    # Crear un df futuro con un solo punto (hoy)
    df_future = pd.DataFrame({
        "ds": [pd.Timestamp.now()],
        "distance": [distance_km],
        "temp": [temperature]
    })
    forecast = m.predict(df_future)
    return forecast["yhat"].iloc[0]

# --------------------------------------------------------------------------------
# OPEN-METEO
# --------------------------------------------------------------------------------
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
            "temperatura": current.get("temperature"),
            "viento": current.get("windspeed"),
            "precipitacion": data.get("hourly", {}).get("precipitation", [0])[0],
            "nubosidad": data.get("hourly", {}).get("cloudcover", [0])[0]
        }
    return None

# --------------------------------------------------------------------------------
# TOMTOM: AUTOCOMPLETAR DIRECCIONES (Search/Geocoding)
# --------------------------------------------------------------------------------
def tomtom_search_autocomplete(query: str):
    tomtom_key = get_tomtom_key()
    url = f"https://api.tomtom.com/search/2/search/{query}.json"
    params = {
        "key": tomtom_key,
        "limit": 5,
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

def address_input_autocomplete(label: str, key: str):
    """
    Muestra un text_input. Si la longitud > 3, llama tomtom_search_autocomplete.
    Muestra un selectbox con sugerencias. Retorna (address, lat, lon).
    """
    st.write(f"**{label}**")
    query = st.text_input(f"Escribe la dirección ({label}):", "", key=key)
    lat = None
    lon = None
    address_selected = None

    if len(query) > 3:
        results = tomtom_search_autocomplete(query)
        if results:
            # Construir opciones
            options = [f"{r[0]} [lat={r[1]:.4f}, lon={r[2]:.4f}]" for r in results]
            choice = st.selectbox(f"Sugerencias ({label})", options, key=f"sel_{key}")
            if choice:
                idx = options.index(choice)
                address_selected = results[idx][0]
                lat = results[idx][1]
                lon = results[idx][2]
    return address_selected, lat, lon

# --------------------------------------------------------------------------------
# TOMTOM ROUTING
# --------------------------------------------------------------------------------
def tomtom_routing_api(origin_lat, origin_lon, dest_lat, dest_lon):
    tomtom_key = get_tomtom_key()
    url = f"https://api.tomtom.com/routing/1/calculateRoute/{origin_lat},{origin_lon}:{dest_lat},{dest_lon}/json"
    params = {
        "key": tomtom_key,
        "traffic": "true",
        "travelMode": "car"
    }
    r = requests.get(url, params=params)
    if r.status_code == 200:
        data = r.json()
        return data
    return None

# --------------------------------------------------------------------------------
# TOMTOM WAYPOINT OPTIMIZATION
# --------------------------------------------------------------------------------
def tomtom_waypoint_optimization(coords_list):
    tomtom_key = get_tomtom_key()
    url = f"https://api.tomtom.com/routing/1/waypointOptimization?key={tomtom_key}"
    # Construir el JSON
    locations = []
    for lat, lon in coords_list:
        locations.append({"point": {"latitude": lat, "longitude": lon}})
    body = {
        "locations": locations,
        "options": {
            "computeBestOrder": True
        }
    }
    r = requests.post(url, json=body)
    if r.status_code == 200:
        return r.json()
    return None

# --------------------------------------------------------------------------------
# CALCULADORA DE CAE
# --------------------------------------------------------------------------------
def calculate_cae(volume_kg: float):
    cae_generated = volume_kg
    cost_min = cae_generated * 0.115
    cost_max = cae_generated * 0.14
    return cae_generated, cost_min, cost_max

# --------------------------------------------------------------------------------
# TAB: Mapa y Tráfico
# --------------------------------------------------------------------------------
def tab_mapa_trafico():
    st.subheader("Mapa y Tráfico")
    st.write("Completa los campos de Origen y Destino (autocompletado).")

    # Autocompletar Origen
    origin_address, origin_lat, origin_lon = address_input_autocomplete("Origen", "origin")
    # Autocompletar Destino
    dest_address, dest_lat, dest_lon = address_input_autocomplete("Destino", "dest")

    if origin_lat and origin_lon and dest_lat and dest_lon:
        if st.button("Calcular Ruta"):
            data = tomtom_routing_api(origin_lat, origin_lon, dest_lat, dest_lon)
            if not data or "routes" not in data:
                st.error("No se pudo obtener la ruta con TomTom Routing API.")
                return
            route = data["routes"][0]
            summary = route["summary"]
            distance_m = summary["lengthInMeters"]
            duration_s = summary["travelTimeInSeconds"]
            st.write(f"Distancia: {distance_m/1000:.2f} km")
            st.write(f"Duración: ~{int(duration_s/60)} min")

            # Construir Folium
            center_lat = (origin_lat + dest_lat)/2
            center_lon = (origin_lon + dest_lon)/2
            m_map = folium.Map(location=[center_lat, center_lon], zoom_start=12)
            tile_url = f"https://api.tomtom.com/map/1/tile/basic/main/{{z}}/{{x}}/{{y}}.png?key={get_tomtom_key()}"
            folium.TileLayer(
                tiles=tile_url,
                attr="TomTom",
                name="TomTom Map"
            ).add_to(m_map)

            # Extraer geometry
            latlons = []
            for leg in route["legs"]:
                for point in leg["points"]:
                    latlons.append((point["latitude"], point["longitude"]))
            folium.PolyLine(latlons, color="blue", weight=5).add_to(m_map)
            folium.Marker((origin_lat, origin_lon), tooltip="Origen", icon=folium.Icon(color="green")).add_to(m_map)
            folium.Marker((dest_lat, dest_lon), tooltip="Destino", icon=folium.Icon(color="red")).add_to(m_map)

            st_folium(m_map, width=700)

# --------------------------------------------------------------------------------
# TAB: Optimización Múltiples Paradas
# --------------------------------------------------------------------------------
def tab_optimizar_paradas():
    st.subheader("Optimización Múltiples Paradas")
    st.write("Ingresa direcciones (una por línea) para optimizar su orden.")

    addresses_text = st.text_area("Direcciones:", "Plaza Mayor, Madrid\nPuerta del Sol, Madrid\nAtocha, Madrid")
    if st.button("Optimizar Orden"):
        lines = [l.strip() for l in addresses_text.split("\n") if l.strip()]
        if len(lines) < 2:
            st.error("Ingresa al menos 2 direcciones.")
            return
        coords_list = []
        for line in lines:
            # Autocompletar la 1ra coincidencia
            results = tomtom_search_autocomplete(line)
            if not results:
                st.warning(f"No se encontró la dirección: {line}")
                return
            lat = results[0][1]
            lon = results[0][2]
            coords_list.append((lat, lon))
        # Llamar a la API
        data_opt = tomtom_waypoint_optimization(coords_list)
        if not data_opt or "routes" not in data_opt:
            st.error("No se pudo optimizar con TomTom Waypoint Optimization API.")
            return
        route = data_opt["routes"][0]
        order = route["waypointsOrder"]
        st.success(f"Orden óptimo de paradas: {order}")

        latlons = []
        for leg in route["legs"]:
            for point in leg["points"]:
                latlons.append((point["latitude"], point["longitude"]))

        center_lat = np.mean([c[0] for c in coords_list])
        center_lon = np.mean([c[1] for c in coords_list])
        m_map = folium.Map(location=[center_lat, center_lon], zoom_start=12)
        tile_url = f"https://api.tomtom.com/map/1/tile/basic/main/{{z}}/{{x}}/{{y}}.png?key={get_tomtom_key()}"
        folium.TileLayer(
            tiles=tile_url,
            attr="TomTom",
            name="TomTom Map"
        ).add_to(m_map)

        folium.PolyLine(latlons, color="blue", weight=5).add_to(m_map)
        for i, (lat, lon) in enumerate(coords_list):
            folium.Marker((lat, lon), tooltip=f"Parada {i+1}", icon=folium.Icon(color="blue")).add_to(m_map)
        st_folium(m_map, width=700)

# --------------------------------------------------------------------------------
# TAB: Predicción de Demanda (usando Prophet con 'distance' y 'temp' como regresores)
# --------------------------------------------------------------------------------
def tab_prediccion_demanda():
    st.subheader("Predicción de Demanda")
    st.write("Calcula la ruta con TomTom, extrae la distancia, obtiene la temperatura con Open‑Meteo, y usa Prophet.")

    # Autocompletar Origen/Destino
    origin_address, origin_lat, origin_lon = address_input_autocomplete("Origen", "demanda_origin")
    dest_address, dest_lat, dest_lon = address_input_autocomplete("Destino", "demanda_dest")

    if origin_lat and origin_lon and dest_lat and dest_lon:
        if st.button("Calcular Ruta y Predecir Demanda"):
            # 1) Calcular ruta
            data = tomtom_routing_api(origin_lat, origin_lon, dest_lat, dest_lon)
            if not data or "routes" not in data:
                st.error("No se pudo obtener la ruta con TomTom.")
                return
            route = data["routes"][0]
            summary = route["summary"]
            distance_m = summary["lengthInMeters"]
            distance_km = distance_m / 1000.0
            st.write(f"Distancia: {distance_km:.2f} km")

            # 2) Obtener temperatura con Open-Meteo en el origen (o punto medio)
            lat_clima = origin_lat  # O (origin_lat + dest_lat)/2
            lon_clima = origin_lon  # O (origin_lon + dest_lon)/2
            meteo = get_weather_open_meteo(lat_clima, lon_clima)
            if not meteo:
                st.warning("No se pudo obtener temperatura con Open‑Meteo. Usando temp=20.0 por defecto.")
                temperature = 20.0
            else:
                temperature = meteo["temperatura"] if meteo["temperatura"] is not None else 20.0
                st.write(f"Temperatura actual: {temperature}°C")

            # 3) Predecir demanda con Prophet
            pred_val = predict_demand_prophet(distance_km, temperature)
            if pred_val is not None:
                st.success(f"Demanda estimada: {pred_val:.2f} (unidades ficticias)")
            else:
                st.error("No se pudo predecir la demanda (modelo no disponible).")

# --------------------------------------------------------------------------------
# TAB: Calculadora CAE
# --------------------------------------------------------------------------------
def tab_calculadora_cae():
    st.subheader("Calculadora CAE")
    st.write("1 CAE = 1 kWh ahorrado. Precio medio: 0.115 - 0.14 €/kWh.")
    volume = st.number_input("CO₂ evitado (kg) / kWh ahorrados", min_value=0.0, value=1000.0, step=100.0)
    if st.button("Calcular CAE"):
        cae_val, cost_min, cost_max = calculate_cae(volume)
        st.write(f"CAE generados: {cae_val:.2f} kWh")
        st.write(f"Ingresos estimados: entre {cost_min:.2f} € y {cost_max:.2f} €")

# --------------------------------------------------------------------------------
# APP PRINCIPAL
# --------------------------------------------------------------------------------
def main_app():
    st.title("Trafiquea: Dashboard para Empresas")
    st.markdown("""
    Funcionalidades principales:
    - **Mapa y Tráfico**: autocompletar direcciones (TomTom Search), mostrar ruta en Folium con tiles de TomTom.
    - **Optimización Múltiples Paradas**: TomTom Waypoint Optimization.
    - **Predicción de Demanda**: se calcula la distancia con TomTom, la temperatura con Open‑Meteo, 
      y se usa un modelo **Prophet** (con regressors) para predecir.
    - **Calculadora CAE**: estima ingresos por kWh ahorrado.
    """)

    tabs = st.tabs(["Mapa y Tráfico", "Optimización Múltiples Paradas", "Predicción de Demanda", "Calculadora CAE"])
    with tabs[0]:
        tab_mapa_trafico()
    with tabs[1]:
        tab_optimizar_paradas()
    with tabs[2]:
        tab_prediccion_demanda()
    with tabs[3]:
        tab_calculadora_cae()

if __name__ == "__main__":
    main_app()
