import os
import streamlit as st
import folium
from streamlit_folium import st_folium
import requests
import numpy as np
import pandas as pd
import plotly.express as px
from datetime import datetime, timedelta
from geopy.geocoders import Nominatim
import joblib
import lightgbm as lgb
from prophet import Prophet

# --------------------------------------------------------------------------------
# CONFIGURACIÓN: Se asume que tu API key de TomTom está en secrets.toml:
# [tomtom]
# api_key = "TU_API_KEY"
# --------------------------------------------------------------------------------

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# --------------------------------------------------------------------------------
# OBTENER CLAVE DE TOMTOM DESDE STREAMLIT SECRETS
# --------------------------------------------------------------------------------
def get_tomtom_key():
    return st.secrets["tomtom"]["api_key"]

# --------------------------------------------------------------------------------
# MAP DISPLAY API: URL DE TILES DE TOMTOM
# --------------------------------------------------------------------------------
def get_tomtom_tile_url():
    tomtom_key = get_tomtom_key()
    # Tiles: "https://api.tomtom.com/map/1/tile/basic/main/{z}/{x}/{y}.png?key=TU_API_KEY"
    return f"https://api.tomtom.com/map/1/tile/basic/main/{{z}}/{{x}}/{{y}}.png?key={tomtom_key}"

# --------------------------------------------------------------------------------
# SEARCH/GEOCODING API: Autocompletar / Búsqueda de Direcciones
# --------------------------------------------------------------------------------
def tomtom_geocode_search(query: str):
    tomtom_key = get_tomtom_key()
    # Endpoint: https://api.tomtom.com/search/2/search/{query}.json
    url = f"https://api.tomtom.com/search/2/search/{query}.json"
    params = {
        "key": tomtom_key,
        "limit": 5,  # max 5 sugerencias
        "language": "es-ES"
    }
    r = requests.get(url, params=params)
    if r.status_code == 200:
        data = r.json()
        # Devolver lista de (displayName, lat, lon)
        results = []
        for item in data.get("results", []):
            address_str = item.get("address", {}).get("freeformAddress", "")
            position = item.get("position", {})
            lat = position.get("lat")
            lon = position.get("lon")
            if lat and lon:
                results.append((address_str, lat, lon))
        return results
    return []

# --------------------------------------------------------------------------------
# ROUTING API: Cálculo de Ruta
# --------------------------------------------------------------------------------
def tomtom_routing_api(origin_lat, origin_lon, dest_lat, dest_lon):
    tomtom_key = get_tomtom_key()
    # Endpoint: https://api.tomtom.com/routing/1/calculateRoute/{latO},{lonO}:{latD},{lonD}/json
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
# WAYPOINT OPTIMIZATION API: Múltiples Paradas
# --------------------------------------------------------------------------------
def tomtom_waypoint_optimization(coords_list):
    tomtom_key = get_tomtom_key()
    # Endpoint: https://api.tomtom.com/routing/1/waypointOptimization
    # Body en JSON con puntos [lat,lon]
    url = f"https://api.tomtom.com/routing/1/waypointOptimization?key={tomtom_key}"
    # coords_list => lista de (lat, lon)
    # Debemos construir un JSON con "locations": [{"point": {"latitude":..., "longitude":...}}]
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
        data = r.json()
        return data
    return None

# --------------------------------------------------------------------------------
# SIMULACIÓN DE MODELO DE DEMANDA (LightGBM con variables aleatorias)
# --------------------------------------------------------------------------------
@st.cache_resource
def load_demand_model():
    model_path = os.path.join(BASE_DIR, "demand_tomtom_lgbm.pkl")
    # Intentar cargar un modelo si existe
    try:
        model = joblib.load(model_path)
        return model
    except:
        # Entrenar un modelo de ejemplo
        st.warning("Entrenando un modelo LightGBM ficticio para la demanda (demostración).")
        # Generar datos aleatorios de entrenamiento
        np.random.seed(42)
        n = 100
        year = np.random.randint(2020, 2030, n)
        traffic = np.random.uniform(0, 1, n)  # saturación
        temp = np.random.uniform(-5, 35, n)
        demand = 1000 + 50*(year-2020) + 200*traffic + 3*temp + np.random.normal(0, 50, n)
        
        df_train = pd.DataFrame({
            "year": year,
            "traffic": traffic,
            "temp": temp,
            "demand": demand
        })
        
        X = df_train[["year", "traffic", "temp"]]
        y = df_train["demand"]
        
        model = lgb.LGBMRegressor()
        model.fit(X, y)
        joblib.dump(model, model_path)
        return model

def predict_demand(year: int, traffic_level: float, temp: float):
    model = load_demand_model()
    if not model:
        return None
    # year, traffic, temp
    X = pd.DataFrame([{
        "year": year,
        "traffic": traffic_level,
        "temp": temp
    }])
    return model.predict(X)[0]

# --------------------------------------------------------------------------------
# CÁLCULO DE CAE
# --------------------------------------------------------------------------------
def calculate_cae(volume_kg: float):
    # 1 CAE = 1 kWh, con precio 0.115 - 0.14 €/kWh
    cae_generated = volume_kg
    cost_min = cae_generated * 0.115
    cost_max = cae_generated * 0.14
    return cae_generated, cost_min, cost_max

# --------------------------------------------------------------------------------
# GEOCODIFICACIÓN DIRECTA (autocomplete)
# --------------------------------------------------------------------------------
def autocomplete_address_input(label: str, key: str):
    """
    Muestra un text_input y un botón 'Buscar'.
    Retorna (address, lat, lon) seleccionado.
    """
    st.write(f"**{label}**")
    address_query = st.text_input(f"Escribe tu dirección ({label})", "", key=key)
    if len(address_query) > 3:
        if st.button(f"Buscar {label}", key=f"btn_{key}"):
            results = tomtom_geocode_search(address_query)
            if results:
                selected = st.selectbox(f"Sugerencias ({label})", [r[0] for r in results], key=f"sel_{key}")
                idx = [r[0] for r in results].index(selected)
                lat = results[idx][1]
                lon = results[idx][2]
                st.success(f"Seleccionado: {selected} (lat={lat}, lon={lon})")
                return selected, lat, lon
            else:
                st.warning("No se encontraron sugerencias.")
    return None, None, None

# --------------------------------------------------------------------------------
# TAB: Mapa y Tráfico
# --------------------------------------------------------------------------------
def tab_mapa_trafico():
    st.subheader("Mapa y Tráfico (TomTom)")

    # Autocompletar origen y destino
    st.write("**Origen**")
    origin_selected, origin_lat, origin_lon = autocomplete_address_input("Origen", "origin")
    st.write("---")
    st.write("**Destino**")
    dest_selected, dest_lat, dest_lon = autocomplete_address_input("Destino", "dest")

    # Mostrar botón de calcular ruta
    if origin_lat and origin_lon and dest_lat and dest_lon:
        if st.button("Calcular Ruta", key="calc_route"):
            route_data = tomtom_routing_api(origin_lat, origin_lon, dest_lat, dest_lon)
            if not route_data or "routes" not in route_data:
                st.error("No se pudo obtener la ruta con TomTom Routing API.")
                return
            route = route_data["routes"][0]
            summary = route["summary"]
            distance_m = summary["lengthInMeters"]
            duration_s = summary["travelTimeInSeconds"]
            st.write(f"Distancia: {distance_m/1000:.2f} km")
            st.write(f"Duración: ~{int(duration_s/60)} min")

            # Construir un mapa con Folium y tiles de TomTom
            center_lat = (origin_lat + dest_lat)/2
            center_lon = (origin_lon + dest_lon)/2
            m_map = folium.Map(location=[center_lat, center_lon], zoom_start=12)
            # Añadir capa de TomTom
            tile_url = f"https://api.tomtom.com/map/1/tile/basic/main/{{z}}/{{x}}/{{y}}.png?key={st.secrets['tomtom']['api_key']}"
            folium.TileLayer(
                tiles=tile_url,
                attr="TomTom",
                name="TomTom Map"
            ).add_to(m_map)

            # Decodificar geometry
            # TomTom Routing geometry se encuentra en "legs[].points", 
            # o "legs[].points[].latitude, longitude" (según la doc).
            # Ejemplo de parse:
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
    st.subheader("Optimización Múltiples Paradas (TomTom Waypoint Optimization)")
    addresses_text = st.text_area("Direcciones (una por línea)", 
                                  "Plaza Mayor, Madrid\nPuerta del Sol, Madrid\nAtocha, Madrid\nRetiro, Madrid")
    if st.button("Optimizar Orden", key="btn_opt"):
        lines = [l.strip() for l in addresses_text.split("\n") if l.strip()]
        if len(lines) < 2:
            st.error("Ingrese al menos 2 direcciones.")
            return
        coords_list = []
        for line in lines:
            # Geocodificar
            results = tomtom_geocode_search(line)
            if not results:
                st.warning(f"No se encontró dirección para: {line}")
                return
            # Tomar la primera sugerencia
            lat = results[0][1]
            lon = results[0][2]
            coords_list.append((lat, lon))
        # Llamar a Waypoint Optimization
        data_opt = tomtom_waypoint_optimization(coords_list)
        if not data_opt or "routes" not in data_opt:
            st.error("No se pudo optimizar con TomTom Waypoint Optimization API.")
            return
        route = data_opt["routes"][0]
        # route["waypointsOrder"] => orden de las paradas
        order = route["waypointsOrder"]
        st.success(f"Orden óptimo de paradas: {order}")

        # Sacar la geometry final (similar a la Routing API)
        latlons = []
        for leg in route["legs"]:
            for point in leg["points"]:
                latlons.append((point["latitude"], point["longitude"]))
        
        center_lat = np.mean([c[0] for c in coords_list])
        center_lon = np.mean([c[1] for c in coords_list])
        m_map = folium.Map(location=[center_lat, center_lon], zoom_start=12)
        # Capa de TomTom
        tile_url = f"https://api.tomtom.com/map/1/tile/basic/main/{{z}}/{{x}}/{{y}}.png?key={st.secrets['tomtom']['api_key']}"
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
# TAB: Predicción de Demanda
# --------------------------------------------------------------------------------
def tab_prediccion_demanda():
    st.subheader("Predicción de Demanda (modelo ficticio)")
    st.write("Aquí integramos factores como saturación de tráfico y temperatura actual (Open-Meteo).")

    # Pedir saturación de tráfico simulado
    sat_option = st.selectbox("Saturación de Tráfico", ["Bajo", "Moderado", "Alto"], key="sat_option")
    if sat_option == "Bajo":
        traffic_level = 0.0
    elif sat_option == "Moderado":
        traffic_level = 0.5
    else:
        traffic_level = 1.0

    # Pedir lat/lon para Open-Meteo
    lat = st.number_input("Latitud", value=40.4167, step=0.0001)
    lon = st.number_input("Longitud", value=-3.7033, step=0.0001)
    if st.button("Obtener Clima", key="btn_clima"):
        meteo = get_weather_open_meteo(lat, lon)
        if meteo:
            st.write(f"Temperatura: {meteo['temperatura']}°C")
            st.write(f"Viento: {meteo['viento']} km/h")
            st.write(f"Precipitación: {meteo['precipitacion']} mm")
            st.write(f"Nubosidad: {meteo['nubosidad']}%")
            # Llamar a predict_demand con year actual y traffic, temp
            current_year = datetime.now().year
            temp_val = meteo["temperatura"] if meteo["temperatura"] else 20.0
            pred_val = predict_demand(current_year, traffic_level, temp_val)
            if pred_val is not None:
                st.success(f"Demanda estimada para {current_year}: {pred_val:.2f} unidades (ficticio)")
        else:
            st.warning("No se pudo obtener datos meteorológicos.")

# --------------------------------------------------------------------------------
# TAB: Calculadora CAE
# --------------------------------------------------------------------------------
def tab_calculadora_cae():
    st.subheader("Calculadora de CAE (kWh ahorrados)")
    volume = st.number_input("CO₂ evitado (kg) / kWh ahorrados", min_value=0.0, value=1000.0, step=100.0)
    if st.button("Calcular CAE", key="calc_cae"):
        cae_val, cost_min, cost_max = calculate_cae(volume)
        st.write(f"CAE generados: {cae_val:.2f} kWh")
        st.write(f"Ingresos estimados: entre {cost_min:.2f} € y {cost_max:.2f} €")

# --------------------------------------------------------------------------------
# APLICACIÓN PRINCIPAL
# --------------------------------------------------------------------------------
def main_app():
    st.title("Trafiquea: Dashboard para Empresas (TomTom API)")
    st.markdown("""
    Este dashboard se basa en las **APIs de TomTom** para:
    - Map Display (tiles en Folium),
    - Routing y Waypoint Optimization,
    - Búsqueda/Geocodificación (para autocompletar direcciones),
    - Open‑Meteo para datos meteorológicos,
    - Un modelo de predicción de demanda (ficticio) que integra saturación de tráfico y temperatura,
    - Calculadora de CAE con rango de 0.115 a 0.14 €/kWh.
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
