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
import itertools

# NLP
try:
    import spacy
    nlp_es = spacy.load("es_core_news_sm")
except:
    nlp_es = None

# Ubicación actual (opcional)
try:
    from streamlit_geolocation import st_geolocation
except ImportError:
    st_geolocation = None

# -----------------------------------------------------------------------------
# CONFIGURACIÓN
# -----------------------------------------------------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

def get_tomtom_key():
    return st.secrets["tomtom"]["api_key"]

# -----------------------------------------------------------------------------
# MODELO PROPHET (REALISTA)
# -----------------------------------------------------------------------------
@st.cache_resource
def load_prophet_model():
    model_path = os.path.join(BASE_DIR, "prophet_realistic.pkl")
    try:
        model = joblib.load(model_path)
        return model
    except:
        st.warning("Entrenando modelo Prophet con influencias meteorológicas moderadas.")
        np.random.seed(42)
        dates = pd.date_range("2023-01-01", "2023-12-31", freq="D")
        n = len(dates)
        distance = np.random.uniform(1, 50, n)
        temp = np.random.uniform(-5, 35, n)
        wind = np.random.uniform(0, 50, n)
        precip = np.random.uniform(0, 20, n)
        cloud = np.random.uniform(0, 100, n)

        # Base: 1.2 min/km
        base_time = 1.2 * distance
        wind_factor = (wind / 50) * 3  # max ~3 min extra
        precip_factor = (precip / 20) * 5  # max ~5 min extra
        cloud_factor = (cloud / 100) * 2  # max ~2 min extra
        y = base_time + wind_factor + precip_factor + cloud_factor + np.random.normal(0, 2, n)

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

def predict_time(distance_km, temp, wind, precip, cloud):
    model = load_prophet_model()
    if not model:
        return None
    df_future = pd.DataFrame({
        "ds": [pd.Timestamp.now()],
        "distance": [distance_km],
        "temp": [temp],
        "wind": [wind],
        "precip": [precip],
        "cloud": [cloud]
    })
    forecast = model.predict(df_future)
    return forecast["yhat"].iloc[0]

# -----------------------------------------------------------------------------
# OPEN-METEO
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
    r = requests.get(url, params=params)
    if r.status_code == 200:
        data = r.json()
        current = data.get("current_weather", {})
        return {
            "temp": current.get("temperature", 20.0),
            "wind": current.get("windspeed", 0.0),
            "precip": data.get("hourly", {}).get("precipitation", [0])[0],
            "cloud": data.get("hourly", {}).get("cloudcover", [0])[0]
        }
    return None

# -----------------------------------------------------------------------------
# TOMTOM SEARCH
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
            addr = item.get("address", {}).get("freeformAddress", "")
            lat = item.get("position", {}).get("lat")
            lon = item.get("position", {}).get("lon")
            if lat and lon:
                suggestions.append((addr, lat, lon))
        return suggestions
    return []

# -----------------------------------------------------------------------------
# TOMTOM ROUTING (para distancias pairwise en TSP)
# -----------------------------------------------------------------------------
@st.cache_data
def get_distance_time(lat1, lon1, lat2, lon2):
    tomtom_key = get_tomtom_key()
    url = f"https://api.tomtom.com/routing/1/calculateRoute/{lat1},{lon1}:{lat2},{lon2}/json"
    params = {
        "key": tomtom_key,
        "traffic": "false",
        "travelMode": "car"
    }
    r = requests.get(url, params=params)
    if r.status_code == 200:
        data = r.json()
        if "routes" in data and data["routes"]:
            dist_m = data["routes"][0]["summary"]["lengthInMeters"]
            time_s = data["routes"][0]["summary"]["travelTimeInSeconds"]
            return dist_m, time_s
    return 999999, 999999

def compute_pairwise_distances(coords_list):
    n = len(coords_list)
    dist_matrix = [[0]*n for _ in range(n)]
    time_matrix = [[0]*n for _ in range(n)]
    for i in range(n):
        for j in range(n):
            if i == j:
                dist_matrix[i][j] = 0
                time_matrix[i][j] = 0
            else:
                dist_m, time_s = get_distance_time(coords_list[i][0], coords_list[i][1],
                                                   coords_list[j][0], coords_list[j][1])
                dist_matrix[i][j] = dist_m
                time_matrix[i][j] = time_s
    return dist_matrix, time_matrix

def solve_tsp_bruteforce(dist_matrix):
    n = len(dist_matrix)
    nodes = list(range(n))
    best_order = None
    best_cost = float("inf")
    for perm in itertools.permutations(nodes):
        cost = 0
        for i in range(n-1):
            cost += dist_matrix[perm[i]][perm[i+1]]
        if cost < best_cost:
            best_cost = cost
            best_order = perm
    return best_order, best_cost

@st.cache_data
def final_routing_with_order(ordered_coords):
    """
    Llama a TomTom Routing con multiple waypoints en el orden dado.
    Retorna dist_m, time_s, route_points
    """
    tomtom_key = get_tomtom_key()
    route_str = ""
    for i, (lat, lon) in enumerate(ordered_coords):
        if i == 0:
            route_str += f"{lat},{lon}"
        else:
            route_str += f":{lat},{lon}"
    url = f"https://api.tomtom.com/routing/1/calculateRoute/{route_str}/json"
    params = {
        "key": tomtom_key,
        "traffic": "true",
        "travelMode": "car"
    }
    r = requests.get(url, params=params)
    if r.status_code != 200:
        return None, None, []
    data = r.json()
    if "routes" not in data or not data["routes"]:
        return None, None, []
    route = data["routes"][0]
    dist_m = route["summary"]["lengthInMeters"]
    time_s = route["summary"]["travelTimeInSeconds"]
    route_points = []
    for leg in route["legs"]:
        for point in leg["points"]:
            route_points.append((point["latitude"], point["longitude"]))
    return dist_m, time_s, route_points

# -----------------------------------------------------------------------------
# NLP para direcciones en español
# -----------------------------------------------------------------------------
def extract_addresses_es(text: str):
    if not nlp_es:
        # fallback: line by line
        lines = text.split("\n")
        return [l.strip() for l in lines if l.strip()]
    doc = nlp_es(text)
    # Ejemplo simple: buscar oraciones que contengan tokens como 'calle', 'av', etc.
    addresses = []
    for sent in doc.sents:
        if any(tok.lower_ in ["calle", "av", "av.", "plaza", "carretera", "camino"] for tok in sent):
            addresses.append(sent.text.strip())
    if not addresses:
        # fallback line by line
        lines = text.split("\n")
        addresses = [l.strip() for l in lines if l.strip()]
    return addresses

# -----------------------------------------------------------------------------
# MAPA
# -----------------------------------------------------------------------------
def render_map(route_points, lat_start, lon_start, lat_end=None, lon_end=None):
    center_lat = lat_start if lat_end is None else (lat_start + lat_end)/2
    center_lon = lon_start if lon_end is None else (lon_start + lon_end)/2
    m_map = folium.Map(location=[center_lat, center_lon], zoom_start=12)
    tile_url = f"https://api.tomtom.com/map/1/tile/basic/main/{{z}}/{{x}}/{{y}}.png?key={get_tomtom_key()}"
    folium.TileLayer(tiles=tile_url, attr="TomTom", name="TomTom Map").add_to(m_map)
    folium.PolyLine(route_points, color="blue", weight=5).add_to(m_map)
    folium.Marker((lat_start, lon_start), tooltip="Inicio", icon=folium.Icon(color="green")).add_to(m_map)
    if lat_end and lon_end:
        folium.Marker((lat_end, lon_end), tooltip="Fin", icon=folium.Icon(color="red")).add_to(m_map)
    st_folium(m_map, width=700)

# -----------------------------------------------------------------------------
# FUNCIONES DE TABS
# -----------------------------------------------------------------------------
def tab_mapa_y_trafico():
    st.subheader("Mapa y Tráfico (Vehículo en español)")
    st.info("Selecciona el tipo de vehículo: Coche, Camión o Furgoneta. Escribe Origen y Destino en español y pulsa Calcular.")
    
    # Ubicación actual
    lat_default, lon_default = 40.4167, -3.7033
    st.write("Intentando obtener tu ubicación actual (opcional):")
    loc = st_geolocation() if st_geolocation else None
    if loc and "latitude" in loc and "longitude" in loc:
        lat_default, lon_default = loc["latitude"], loc["longitude"]
        st.success(f"Ubicación actual: lat={lat_default:.4f}, lon={lon_default:.4f}")

    # Mapa base
    m_base = folium.Map(location=[lat_default, lon_default], zoom_start=13)
    tile_url = f"https://api.tomtom.com/map/1/tile/basic/main/{{z}}/{{x}}/{{y}}.png?key={get_tomtom_key()}"
    folium.TileLayer(tiles=tile_url, attr="TomTom", name="TomTom Map").add_to(m_base)
    st_folium(m_base, width=700)

    veh_dict = {"Coche": "car", "Camión": "truck", "Furgoneta": "van"}
    veh_type = st.selectbox("Tipo de Vehículo", list(veh_dict.keys()))
    
    origin_query = st.text_input("Origen (escribe y pulsa Enter):", key="origin_query")
    dest_query = st.text_input("Destino (escribe y pulsa Enter):", key="dest_query")

    if st.button("Calcular Ruta"):
        # geocodificar origen
        origin_results = tomtom_search(origin_query)
        if not origin_results:
            st.error("No se encontraron sugerencias para el origen.")
            return
        if len(origin_results) > 1:
            sel_o = st.selectbox("Sugerencias Origen", [r[0] for r in origin_results], key="sel_origin")
            origin_lat, origin_lon = None, None
            for r in origin_results:
                if r[0] == sel_o:
                    origin_lat, origin_lon = r[1], r[2]
                    break
        else:
            origin_lat, origin_lon = origin_results[0][1], origin_results[0][2]

        # geocodificar destino
        dest_results = tomtom_search(dest_query)
        if not dest_results:
            st.error("No se encontraron sugerencias para el destino.")
            return
        if len(dest_results) > 1:
            sel_d = st.selectbox("Sugerencias Destino", [r[0] for r in dest_results], key="sel_dest")
            dest_lat, dest_lon = None, None
            for r in dest_results:
                if r[0] == sel_d:
                    dest_lat, dest_lon = r[1], r[2]
                    break
        else:
            dest_lat, dest_lon = dest_results[0][1], dest_results[0][2]

        # Llamar Routing
        if origin_lat and origin_lon and dest_lat and dest_lon:
            vehicle_api = veh_dict[veh_type]
            routing_data = tomtom_routing_api(origin_lat, origin_lon, dest_lat, dest_lon, vehicle_type=vehicle_api)
            if not routing_data or "routes" not in routing_data:
                st.error("No se pudo obtener la ruta con TomTom.")
                return
            route = routing_data["routes"][0]
            dist_m = route["summary"]["lengthInMeters"]
            time_s = route["summary"]["travelTimeInSeconds"]
            st.write(f"Distancia: {dist_m/1000:.2f} km")
            st.write(f"Tiempo base: ~{int(time_s/60)} min")

            st.session_state["origin_lat"] = origin_lat
            st.session_state["origin_lon"] = origin_lon
            st.session_state["dest_lat"] = dest_lat
            st.session_state["dest_lon"] = dest_lon
            st.session_state["distance_km"] = dist_m/1000.0
            st.session_state["duration_min"] = time_s/60.0

            route_points = []
            for leg in route["legs"]:
                for point in leg["points"]:
                    route_points.append((point["latitude"], point["longitude"]))
            st.session_state["route_points"] = route_points

            render_map(route_points, origin_lat, origin_lon, dest_lat, dest_lon)

def tab_ruta_alternativa():
    st.subheader("Ruta Alternativa si hay incidencias")
    if "origin_lat" not in st.session_state or "dest_lat" not in st.session_state:
        st.warning("Primero calcula la ruta en 'Mapa y Tráfico'.")
        return
    # Ejemplo: si detectamos una carretera principal a evitar
    road_to_avoid = st.text_input("Carretera a evitar (ej: 'A5')", "A5")
    if st.button("Calcular Ruta Alternativa"):
        tomtom_key = get_tomtom_key()
        url = f"https://api.tomtom.com/routing/1/calculateRoute/{st.session_state['origin_lat']},{st.session_state['origin_lon']}:{st.session_state['dest_lat']},{st.session_state['dest_lon']}/json"
        params = {
            "key": tomtom_key,
            "traffic": "true",
            "travelMode": "car",
            "avoidRoads": road_to_avoid
        }
        r = requests.get(url, params=params)
        if r.status_code != 200:
            st.error("No se pudo obtener la ruta alternativa.")
            return
        data = r.json()
        if "routes" not in data or not data["routes"]:
            st.error("No se encontró ruta alternativa.")
            return
        route = data["routes"][0]
        dist_m = route["summary"]["lengthInMeters"]
        time_s = route["summary"]["travelTimeInSeconds"]
        st.write(f"Distancia alternativa: {dist_m/1000:.2f} km")
        st.write(f"Tiempo alternativo: ~{int(time_s/60)} min")

        alt_points = []
        for leg in route["legs"]:
            for point in leg["points"]:
                alt_points.append((point["latitude"], point["longitude"]))

        render_map(alt_points, st.session_state["origin_lat"], st.session_state["origin_lon"],
                   st.session_state["dest_lat"], st.session_state["dest_lon"])

def tab_tsp_nlp():
    st.subheader("Optimización de Múltiples Paradas (NLP + TSP Avanzado)")
    st.info("Introduce un texto en español indicando origen, paradas y destino. Extraeremos direcciones y calcularemos ruta óptima.")
    text = st.text_area("Ej: 'Salir desde Calle Alcalá 100, luego pasar por Plaza Mayor, terminar en Gran Vía 45'")
    if st.button("Optimizar"):
        # Extraer direcciones con spaCy
        addresses = extract_addresses_es(text)
        if not addresses:
            st.error("No se detectaron direcciones en el texto.")
            return
        # Geocodificar
        coords_list = []
        for addr in addresses:
            results = tomtom_search(addr)
            if not results:
                st.error(f"No se encontró geocodificación para: {addr}")
                return
            coords_list.append((results[0][1], results[0][2]))

        n = len(coords_list)
        if n > 10:
            st.warning("Más de 10 direcciones. El TSP brute force puede ser muy lento.")
            return

        dist_matrix, _ = compute_pairwise_distances(coords_list)
        best_order, best_cost = solve_tsp_bruteforce(dist_matrix)
        if not best_order:
            st.error("No se pudo resolver TSP.")
            return
        st.success(f"Orden óptimo (TSP) calculado. Distancia ~{best_cost/1000:.2f} km (aprox).")

        # Ruta final en TomTom
        ordered_coords = [coords_list[i] for i in best_order]
        dist_m, time_s, route_pts = final_routing_with_order(ordered_coords)
        if dist_m is None:
            st.error("No se pudo obtener ruta final con TomTom.")
            return
        st.write(f"Distancia total: {dist_m/1000:.2f} km")
        st.write(f"Tiempo total: ~{int(time_s/60)} min")
        render_map(route_pts, ordered_coords[0][0], ordered_coords[0][1],
                   ordered_coords[-1][0], ordered_coords[-1][1])

def tab_trafico_ruta():
    st.subheader("Tráfico en Ruta")
    origin_lat = st.session_state.get("origin_lat")
    dest_lat = st.session_state.get("dest_lat")
    origin_lon = st.session_state.get("origin_lon")
    dest_lon = st.session_state.get("dest_lon")
    if origin_lat is None or dest_lat is None or origin_lon is None or dest_lon is None:
        st.warning("Primero calcula la ruta en 'Mapa y Tráfico'.")
        return
    lat_m = (origin_lat + dest_lat) / 2
    lon_m = (origin_lon + dest_lon) / 2
    if st.button("Consultar Tráfico"):
        data_flow = tomtom_traffic_flow(lat_m, lon_m)
        if not data_flow or "flowSegmentData" not in data_flow:
            st.error("No se pudo obtener datos de tráfico.")
            return
        flow = data_flow["flowSegmentData"]
        cur_spd = flow.get("currentSpeed", 0)
        free_spd = flow.get("freeFlowSpeed", 0)
        st.write(f"Velocidad actual: {cur_spd} km/h")
        st.write(f"Velocidad libre: {free_spd} km/h")
        if cur_spd < 20:
            st.error("Tráfico muy congestionado. Considera ruta alternativa.")
        elif cur_spd < 50:
            st.warning("Tráfico moderado. Posibles retrasos.")
        else:
            st.info("Tráfico fluido.")


def tab_prediccion_retrasos():
    st.subheader("Predicción de Retrasos con Modelo Realista")
    if "distance_km" not in st.session_state or "duration_min" not in st.session_state:
        st.warning("Primero calcula la ruta en 'Mapa y Tráfico'.")
        return
    dist_km = st.session_state["distance_km"]
    base_min = st.session_state["duration_min"]
    lat_m = (st.session_state["origin_lat"] + st.session_state["dest_lat"]) / 2
    lon_m = (st.session_state["origin_lon"] + st.session_state["dest_lon"]) / 2

    st.write(f"Distancia (TomTom): {dist_km:.2f} km")
    st.write(f"Tiempo base (TomTom): ~{base_min:.2f} min")

    meteo = get_weather_open_meteo(lat_m, lon_m)
    if not meteo:
        st.warning("No se pudo obtener meteo. Se asume temp=20, wind=0, precip=0, cloud=0.")
        temp, wind, precip, cloud = 20.0, 0.0, 0.0, 0.0
    else:
        temp = meteo["temp"]
        wind = meteo["wind"]
        precip = meteo["precip"]
        cloud = meteo["cloud"]
        st.write(f"Clima actual: Temp={temp:.1f}°C, Viento={wind:.1f} km/h, Precip={precip:.1f} mm, Nubosidad={cloud:.1f}%")

    if st.button("Calcular Retraso"):
        predicted = predict_time(dist_km, temp, wind, precip, cloud)
        st.success(f"Tiempo estimado (modelo): {predicted:.2f} min")
        delay = predicted - base_min
        if delay > 0:
            st.error(f"Retraso: +{delay:.2f} min")
        else:
            st.info(f"Adelanto: ~{abs(delay):.2f} min")

def tab_calculadora_cae():
    st.subheader("Calculadora CAE")
    kwh = st.number_input("kWh ahorrados", min_value=0.0, value=500.0, step=50.0)
    if st.button("Calcular Ingresos"):
        cost_min = kwh * 0.115
        cost_max = kwh * 0.14
        st.write(f"CAE generados: {kwh:.2f} kWh")
        st.write(f"Ingresos estimados: entre {cost_min:.2f} € y {cost_max:.2f} €")

def main_app():
    st.title("Trafiquea: Empresas Logísticas (Versión Completa)")
    st.markdown("""
    **Mapa y Tráfico**:
    - Mapa fijo centrado en tu ubicación (si lo permites).
    - Origen y Destino en español, pulsar Enter para sugerencias.
    - Opción de ruta alternativa.

    **TSP Avanzado**:
    - Input en lenguaje natural (español).
    - NLP (spaCy) para extraer direcciones.
    - TomTom Routing pairwise + TSP brute force para orden óptimo.

    **Tráfico en Ruta**:
    - Consulta la Traffic API en el punto medio.

    **Predicción de Retrasos**:
    - Modelo Prophet realista, con variables meteo que suman unos minutos.

    **Calculadora CAE**:
    - kWh ahorrados => ingresos estimados (0.115–0.14 €/kWh).
    """)

    tabs = st.tabs([
        "Mapa y Tráfico",
        "Ruta Alternativa",
        "TSP Avanzado (NLP)",
        "Tráfico en Ruta",
        "Predicción de Retrasos",
        "Calculadora CAE"
    ])
    with tabs[0]:
        tab_mapa_y_trafico()
    with tabs[1]:
        tab_ruta_alternativa()
    with tabs[2]:
        tab_tsp_nlp()
    with tabs[3]:
        tab_trafico_ruta()
    with tabs[4]:
        tab_prediccion_retrasos()
    with tabs[5]:
        tab_calculadora_cae()

if __name__ == "__main__":
    # Iniciar variables de session si no existen
    for key in ["origin_lat", "origin_lon", "dest_lat", "dest_lon", "distance_km", "duration_min", "route_points"]:
        if key not in st.session_state:
            st.session_state[key] = None

    main_app()
