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

# -----------------------------------------------------------------------------
# CONFIGURACIÓN DE RUTAS Y SECRETOS
# -----------------------------------------------------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

def get_tomtom_key():
    return st.secrets["tomtom"]["api_key"]

# -----------------------------------------------------------------------------
# ENTRENAMIENTO MODELO PROPHET (SINTÉTICO) con 'distance' y 'temp' como REGRESSORS
# -----------------------------------------------------------------------------
@st.cache_resource
def train_prophet_model():
    model_path = os.path.join(BASE_DIR, "prophet_tomtom_model.pkl")
    try:
        m = joblib.load(model_path)
        return m
    except:
        st.warning("Entrenando modelo Prophet sintético con 'distance' y 'temp' como regressors.")
        np.random.seed(42)
        # Generamos datos diarios de un año
        dates = pd.date_range("2023-01-01", "2023-12-31", freq="D")
        n = len(dates)
        distance = np.random.uniform(1, 50, n)
        temp = np.random.uniform(-5, 35, n)
        # y = 60 + 2*distance + 1.5*temp + ruido
        y = 60 + 2*distance + 1.5*temp + np.random.normal(0, 10, n)

        df = pd.DataFrame({"ds": dates, "y": y, "distance": distance, "temp": temp})
        m = Prophet()
        m.add_regressor("distance")
        m.add_regressor("temp")
        m.fit(df)
        joblib.dump(m, model_path)
        return m

def predict_delay_with_prophet(distance_km: float, temperature: float, base_time: float):
    """
    Predice un 'tiempo de viaje' usando Prophet, y retorna la diferencia (retraso) con el base_time de TomTom.
    """
    m = train_prophet_model()
    if not m:
        return None, None
    df_future = pd.DataFrame({
        "ds": [pd.Timestamp.now()],
        "distance": [distance_km],
        "temp": [temperature]
    })
    forecast = m.predict(df_future)
    predicted_time = forecast["yhat"].iloc[0]  # Valor sintético de 'tiempo'
    delay = predicted_time - base_time
    return predicted_time, delay

# -----------------------------------------------------------------------------
# OPEN-METEO
# -----------------------------------------------------------------------------
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

# -----------------------------------------------------------------------------
# TOMTOM: AUTOCOMPLETAR (Search/Geocoding)
# -----------------------------------------------------------------------------
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
    Muestra un text_input. Si la longitud > 3, se llama tomtom_search_autocomplete.
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

# -----------------------------------------------------------------------------
# TOMTOM ROUTING
# -----------------------------------------------------------------------------
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

# -----------------------------------------------------------------------------
# TOMTOM WAYPOINT OPTIMIZATION
# -----------------------------------------------------------------------------
def tomtom_waypoint_optimization(coords_list):
    tomtom_key = get_tomtom_key()
    url = f"https://api.tomtom.com/routing/1/waypointOptimization?key={tomtom_key}"
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

# -----------------------------------------------------------------------------
# CALCULADORA CAE
# -----------------------------------------------------------------------------
def calculate_cae(kwh: float):
    """
    kWh ahorrados. 1 CAE = 1 kWh. 
    Precio medio: 0.115 - 0.14 €/kWh
    """
    cae_generated = kwh
    cost_min = kwh * 0.115
    cost_max = kwh * 0.14
    return cae_generated, cost_min, cost_max

# -----------------------------------------------------------------------------
# TAB: Mapa y Tráfico
# -----------------------------------------------------------------------------
def tab_mapa_trafico():
    st.subheader("Mapa y Tráfico")
    # Si tenemos en session_state un mapa dibujado, lo mostramos primero
    if "route_map_html" in st.session_state:
        st_folium(st.session_state["route_map_html"], width=700)

    origin_address, origin_lat, origin_lon = address_input_autocomplete("Origen", "origin")
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

            center_lat = (origin_lat + dest_lat)/2
            center_lon = (origin_lon + dest_lon)/2
            m_map = folium.Map(location=[center_lat, center_lon], zoom_start=12)
            tile_url = f"https://api.tomtom.com/map/1/tile/basic/main/{{z}}/{{x}}/{{y}}.png?key={get_tomtom_key()}"
            folium.TileLayer(
                tiles=tile_url,
                attr="TomTom",
                name="TomTom Map"
            ).add_to(m_map)

            latlons = []
            for leg in route["legs"]:
                for point in leg["points"]:
                    latlons.append((point["latitude"], point["longitude"]))

            folium.PolyLine(latlons, color="blue", weight=5).add_to(m_map)
            folium.Marker((origin_lat, origin_lon), tooltip="Origen", icon=folium.Icon(color="green")).add_to(m_map)
            folium.Marker((dest_lat, dest_lon), tooltip="Destino", icon=folium.Icon(color="red")).add_to(m_map)

            # Almacenamos en session_state la info
            st.session_state["distance_km"] = distance_m / 1000.0
            st.session_state["duration_s"] = duration_s
            st.session_state["origin_lat"] = origin_lat
            st.session_state["origin_lon"] = origin_lon
            st.session_state["dest_lat"] = dest_lat
            st.session_state["dest_lon"] = dest_lon

            # Renderizar el mapa y guardar su "objeto" en session_state
            map_html = st_folium(m_map, width=700)
            st.session_state["route_map_html"] = map_html

# -----------------------------------------------------------------------------
# TAB: Optimización Múltiples Paradas
# -----------------------------------------------------------------------------
def tab_optimizar_paradas():
    st.subheader("Optimización Múltiples Paradas")
    addresses_text = st.text_area("Direcciones (una por línea)",
                                  "Plaza Mayor, Madrid\nPuerta del Sol, Madrid\nAtocha, Madrid")
    if st.button("Optimizar Orden"):
        lines = [l.strip() for l in addresses_text.split("\n") if l.strip()]
        if len(lines) < 2:
            st.error("Ingresa al menos 2 direcciones.")
            return
        coords_list = []
        for line in lines:
            results = tomtom_search_autocomplete(line)
            if not results:
                st.warning(f"No se encontró la dirección: {line}")
                return
            lat = results[0][1]
            lon = results[0][2]
            coords_list.append((lat, lon))
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

# -----------------------------------------------------------------------------
# TAB: Predicción de Demanda (Prophet)
# -----------------------------------------------------------------------------
def tab_prediccion_demanda():
    st.subheader("Predicción de Demanda con Ruta TomTom + Open-Meteo")
    # Revisamos si tenemos la distancia y duracion en session_state
    if "distance_km" not in st.session_state or "duration_s" not in st.session_state:
        st.warning("Primero calcula la ruta en la pestaña 'Mapa y Tráfico'.")
        return

    distance_km = st.session_state["distance_km"]
    duration_s = st.session_state["duration_s"]
    # Tomamos el punto medio para la meteorología
    lat_m = (st.session_state["origin_lat"] + st.session_state["dest_lat"]) / 2
    lon_m = (st.session_state["origin_lon"] + st.session_state["dest_lon"]) / 2

    # Obtenemos la temperatura actual con Open-Meteo
    meteo = get_weather_open_meteo(lat_m, lon_m)
    if not meteo:
        st.warning("No se pudo obtener la temperatura con Open-Meteo. Asignando temp=20°C.")
        temperature = 20.0
    else:
        temperature = meteo["temperatura"] if meteo["temperatura"] is not None else 20.0

    st.write(f"Distancia de la ruta: {distance_km:.2f} km")
    st.write(f"Tiempo base (TomTom): ~{int(duration_s/60)} min")
    st.write(f"Temperatura actual: {temperature:.1f} °C")

    if st.button("Calcular 'Retraso'"):
        predicted_time, delay = predict_delay_with_prophet(distance_km, temperature, duration_s/60.0)
        if predicted_time is not None:
            st.success(f"Tiempo estimado por el modelo: {predicted_time:.2f} min")
            if delay >= 0:
                st.error(f"Posible retraso: +{delay:.2f} min respecto a TomTom")
            else:
                st.info(f"Podrías llegar {abs(delay):.2f} min antes de lo esperado.")
        else:
            st.error("No se pudo predecir la demanda con Prophet.")

# -----------------------------------------------------------------------------
# TAB: Calculadora CAE
# -----------------------------------------------------------------------------
def tab_calculadora_cae():
    st.subheader("Calculadora CAE")
    st.write("Introduce los kWh ahorrados y estima los ingresos potenciales.")
    kwh = st.number_input("kWh ahorrados", min_value=0.0, value=1000.0, step=100.0)
    if st.button("Calcular CAE"):
        cae_val, cost_min, cost_max = calculate_cae(kwh)
        st.write(f"CAE generados: {cae_val:.2f} kWh")
        st.write(f"Ingresos estimados: entre {cost_min:.2f} € y {cost_max:.2f} €")

# -----------------------------------------------------------------------------
# APP PRINCIPAL
# -----------------------------------------------------------------------------
def main_app():
    st.title("Trafiquea: Dashboard para Empresas")
    st.markdown("""
    **Mapa y Tráfico:**  
    - Autocompleta direcciones usando TomTom Search.  
    - Calcula ruta con TomTom Routing y pinta un mapa con tiles de TomTom.  
    **Optimización Múltiples Paradas:**  
    - Usa TomTom Waypoint Optimization.  
    **Predicción de Demanda:**  
    - Toma la distancia de la ruta y la temperatura actual (Open-Meteo) para alimentar un modelo **Prophet**.  
    - Compara con la duración base de TomTom y estima un posible "retraso" o adelanto.  
    **Calculadora CAE:**  
    - Introduce kWh ahorrados y estima ingresos (0.115-0.14 €/kWh).
    """)

    tabs = st.tabs([
        "Mapa y Tráfico", 
        "Optimización Múltiples Paradas", 
        "Predicción de Demanda", 
        "Calculadora CAE"
    ])

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
