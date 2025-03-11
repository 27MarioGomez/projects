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

# Intentar cargar spaCy y su modelo en español
try:
    import spacy
    nlp_es = spacy.load("es_core_news_sm")
except Exception as e:
    nlp_es = None
    st.error("No se pudo cargar spaCy o el modelo 'es_core_news_sm'. Ejecuta 'python -m spacy download es_core_news_sm'.")

# -----------------------------------------------------------------------------
# CONFIGURACIÓN: API key de TomTom (definida en secrets.toml)
# -----------------------------------------------------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

def get_tomtom_key():
    return st.secrets["tomtom"]["api_key"]

# -----------------------------------------------------------------------------
# SOLICITAR UBICACIÓN ACTUAL CON JAVASCRIPT (con fallback)
# -----------------------------------------------------------------------------
def request_location():
    # Si no existen lat y lon en query params, pedimos ubicación con JavaScript
    params = st.experimental_get_query_params()
    if "lat" not in params or "lon" not in params:
        loc_script = """
        <script>
        if (navigator.geolocation) {
            navigator.geolocation.getCurrentPosition(function(position) {
                var lat = position.coords.latitude;
                var lon = position.coords.longitude;
                window.location.href = window.location.href.split('?')[0] + '?lat=' + lat + '&lon=' + lon;
            });
        }
        </script>
        """
        st.components.v1.html(loc_script, height=0)
request_location()

query_params = st.experimental_get_query_params()
if "lat" in query_params and "lon" in query_params:
    try:
        st.session_state["current_lat"] = float(query_params["lat"][0])
        st.session_state["current_lon"] = float(query_params["lon"][0])
    except:
        st.session_state["current_lat"] = 40.4167
        st.session_state["current_lon"] = -3.7033
else:
    st.session_state["current_lat"] = 40.4167
    st.session_state["current_lon"] = -3.7033

# -----------------------------------------------------------------------------
# MODELO PROPHET REALISTA (SINTÉTICO) CON REGRESSORES
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
        base_time = 1.2 * distance
        wind_factor = (wind / 50) * 3   # hasta 3 min extra
        precip_factor = (precip / 20) * 5  # hasta 5 min extra
        cloud_factor = (cloud / 100) * 2   # hasta 2 min extra
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
def get_weather_open_meteo(lat, lon):
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
def tomtom_search(query, limit=5):
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
# TOMTOM WAYPOINT OPTIMIZATION: Usaremos TSP local (para ≤ 8 paradas)
# -----------------------------------------------------------------------------
@st.cache_data(ttl=600)
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
            return data["routes"][0]["summary"]["lengthInMeters"], data["routes"][0]["summary"]["travelTimeInSeconds"]
    return 999999, 999999

def compute_pairwise_distances(coords_list):
    n = len(coords_list)
    dist_matrix = [[0]*n for _ in range(n)]
    for i in range(n):
        for j in range(n):
            if i != j:
                d, _ = get_distance_time(coords_list[i][0], coords_list[i][1],
                                         coords_list[j][0], coords_list[j][1])
                dist_matrix[i][j] = d
    return dist_matrix

def solve_tsp_bruteforce(dist_matrix):
    n = len(dist_matrix)
    if n > 8:
        return None, None, "Demasiadas paradas (más de 8)."
    nodes = list(range(n))
    best_order = None
    best_cost = float("inf")
    for perm in itertools.permutations(nodes):
        cost = sum(dist_matrix[perm[i]][perm[i+1]] for i in range(n-1))
        if cost < best_cost:
            best_cost = cost
            best_order = perm
    return best_order, best_cost, None

@st.cache_data(ttl=600)
def final_routing_with_order(ordered_coords):
    tomtom_key = get_tomtom_key()
    route_str = ":".join(f"{lat},{lon}" for lat, lon in ordered_coords)
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
# NLP con spaCy en español para extraer direcciones
# -----------------------------------------------------------------------------
def extract_addresses_es(text: str):
    if nlp_es:
        doc = nlp_es(text)
        addresses = [sent.text.strip() for sent in doc.sents if any(tok.lower_ in ["calle", "av", "av.", "plaza", "carretera", "camino"] for tok in sent)]
        if addresses:
            return addresses
    # Fallback: cada línea es una dirección
    return [line.strip() for line in text.split("\n") if line.strip()]

# -----------------------------------------------------------------------------
# RENDERIZAR MAPA CON FOLIUM
# -----------------------------------------------------------------------------
def render_map(route_points, lat_start, lon_start, lat_end=None, lon_end=None, color="blue"):
    center_lat = lat_start if lat_end is None else (lat_start + lat_end)/2
    center_lon = lon_start if lon_end is None else (lon_start + lon_end)/2
    m_map = folium.Map(location=[center_lat, center_lon], zoom_start=12)
    tile_url = f"https://api.tomtom.com/map/1/tile/basic/main/{{z}}/{{x}}/{{y}}.png?key={get_tomtom_key()}"
    folium.TileLayer(tiles=tile_url, attr="TomTom").add_to(m_map)
    folium.PolyLine(route_points, color=color, weight=5).add_to(m_map)
    folium.Marker((lat_start, lon_start), tooltip="Inicio", icon=folium.Icon(color="green")).add_to(m_map)
    if lat_end and lon_end:
        folium.Marker((lat_end, lon_end), tooltip="Fin", icon=folium.Icon(color="red")).add_to(m_map)
    st_folium(m_map, width=700)

# -----------------------------------------------------------------------------
# TAB: Calcular ruta (Tab 1)
# -----------------------------------------------------------------------------
def tab_calcular_ruta():
    st.subheader("Calcular ruta")
    # Dropdown para hora de salida (formato 24 horas)
    hora_salida = st.selectbox("Hora de salida", [f"{i:02d}:00" for i in range(24)])
    st.write("Ingresa Origen y Destino:")
    origin_addr = st.text_input("Origen", key="origin_query")
    dest_addr = st.text_input("Destino", key="dest_query")
    if st.button("Calcular ruta"):
        origin_results = tomtom_search(origin_addr)
        if not origin_results:
            st.error("No se encontraron sugerencias para el origen.")
            return
        if len(origin_results) > 1:
            sel_origin = st.selectbox("Sugerencias para Origen", [r[0] for r in origin_results], key="sel_origin")
            origin_lat, origin_lon = None, None
            for r in origin_results:
                if r[0] == sel_origin:
                    origin_lat, origin_lon = r[1], r[2]
                    break
        else:
            origin_lat, origin_lon = origin_results[0][1], origin_results[0][2]

        dest_results = tomtom_search(dest_addr)
        if not dest_results:
            st.error("No se encontraron sugerencias para el destino.")
            return
        if len(dest_results) > 1:
            sel_dest = st.selectbox("Sugerencias para Destino", [r[0] for r in dest_results], key="sel_dest")
            dest_lat, dest_lon = None, None
            for r in dest_results:
                if r[0] == sel_dest:
                    dest_lat, dest_lon = r[1], r[2]
                    break
        else:
            dest_lat, dest_lon = dest_results[0][1], dest_results[0][2]

        if origin_lat and origin_lon and dest_lat and dest_lon:
            veh_dict = {"Coche": "car", "Camión": "truck", "Furgoneta": "van"}
            veh_type = st.selectbox("Tipo de Vehículo", list(veh_dict.keys()))
            routing_data = tomtom_routing_api(origin_lat, origin_lon, dest_lat, dest_lon, vehicle_type=veh_dict[veh_type])
            if not routing_data or "routes" not in routing_data:
                st.error("No se pudo obtener la ruta.")
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

            # Consultar eventos de tráfico y agregar marcadores si hay incidencias
            lat_mid = (origin_lat + dest_lat) / 2
            lon_mid = (origin_lon + dest_lon) / 2
            flow_data = tomtom_traffic_flow(lat_mid, lon_mid)
            event_markers = []
            if flow_data and "flowSegmentData" in flow_data:
                flow = flow_data["flowSegmentData"]
                current_speed = flow.get("currentSpeed", 0)
                if current_speed < 30:
                    event_markers.append((lat_mid, lon_mid, "Tráfico muy congestionado"))
            # Renderizar mapa base con ruta en azul y eventos (si hay)
            m = folium.Map(location=[lat_mid, lon_mid], zoom_start=12)
            tile_url = f"https://api.tomtom.com/map/1/tile/basic/main/{{z}}/{{x}}/{{y}}.png?key={get_tomtom_key()}"
            folium.TileLayer(tiles=tile_url, attr="TomTom").add_to(m)
            folium.PolyLine(st.session_state["route_points"], color="blue", weight=5).add_to(m)
            folium.Marker((origin_lat, origin_lon), tooltip="Origen", icon=folium.Icon(color="green")).add_to(m)
            folium.Marker((dest_lat, dest_lon), tooltip="Destino", icon=folium.Icon(color="red")).add_to(m)
            for lat_e, lon_e, msg in event_markers:
                folium.Marker((lat_e, lon_e), tooltip=msg, icon=folium.Icon(color="orange", icon="exclamation-sign")).add_to(m)
            st_folium(m, width=700)

# -----------------------------------------------------------------------------
# TAB: Ruta alternativa (pintar sobre el mapa base existente)
# -----------------------------------------------------------------------------
def tab_ruta_alternativa():
    st.subheader("Ruta alternativa")
    if "origin_lat" not in st.session_state or "dest_lat" not in st.session_state:
        st.warning("Primero calcula la ruta en 'Calcular ruta'.")
        return
    road_to_avoid = st.text_input("Carretera a evitar (ejemplo: A5)", "")
    if st.button("Calcular ruta alternativa"):
        tomtom_key = get_tomtom_key()
        url = f"https://api.tomtom.com/routing/1/calculateRoute/{st.session_state['origin_lat']},{st.session_state['origin_lon']}:{st.session_state['dest_lat']},{st.session_state['dest_lon']}/json"
        params = {
            "key": tomtom_key,
            "traffic": "true",
            "travelMode": "car"
        }
        if road_to_avoid.strip():
            params["avoidRoads"] = road_to_avoid.strip()
        r = requests.get(url, params=params)
        if r.status_code != 200:
            st.error("Error al obtener la ruta alternativa.")
            return
        data = r.json()
        if "routes" not in data or not data["routes"]:
            st.error("No se encontró ruta alternativa.")
            return
        route_alt = data["routes"][0]
        dist_alt = route_alt["summary"]["lengthInMeters"]
        time_alt = route_alt["summary"]["travelTimeInSeconds"]
        st.write(f"Ruta alternativa: {dist_alt/1000:.2f} km, {int(time_alt/60)} min")
        alt_points = []
        for leg in route_alt["legs"]:
            for point in leg["points"]:
                alt_points.append((point["latitude"], point["longitude"]))
        # Reutilizamos el mapa base almacenado en session_state["route_points"] si existe, o generamos uno nuevo
        if "route_points" in st.session_state and st.session_state["route_points"]:
            base_points = st.session_state["route_points"]
            # Se vuelve a renderizar el mapa con la ruta base en azul y la alternativa en rojo
            m = folium.Map(location=[(st.session_state["origin_lat"]+st.session_state["dest_lat"])/2,
                                     (st.session_state["origin_lon"]+st.session_state["dest_lon"])/2], zoom_start=12)
            tile_url = f"https://api.tomtom.com/map/1/tile/basic/main/{{z}}/{{x}}/{{y}}.png?key={get_tomtom_key()}"
            folium.TileLayer(tiles=tile_url, attr="TomTom").add_to(m)
            folium.PolyLine(base_points, color="blue", weight=5).add_to(m)
            folium.PolyLine(alt_points, color="red", weight=5).add_to(m)
            folium.Marker((st.session_state["origin_lat"], st.session_state["origin_lon"]), tooltip="Origen", icon=folium.Icon(color="green")).add_to(m)
            folium.Marker((st.session_state["dest_lat"], st.session_state["dest_lon"]), tooltip="Destino", icon=folium.Icon(color="red")).add_to(m)
            st_folium(m, width=700)
        else:
            st.error("No hay ruta base guardada.")

# -----------------------------------------------------------------------------
# TAB: Calcular ruta completa (TSP avanzado con NLP)
# -----------------------------------------------------------------------------
def tab_calcular_ruta_completa():
    st.subheader("Calcular ruta completa")
    st.info("Introduce un texto en español indicando origen, paradas y destino.")
    text = st.text_area("Ejemplo: 'Salir de Calle Alcalá 100, pasar por Plaza Mayor, luego Gran Vía 45 y terminar en Atocha'")
    if st.button("Calcular"):
        addresses = extract_addresses_es(text)
        if not addresses:
            st.error("No se detectaron direcciones.")
            return
        if len(addresses) > 8:
            st.error("Demasiadas direcciones (más de 8).")
            return
        coords_list = []
        for addr in addresses:
            res = tomtom_search(addr)
            if not res:
                st.error(f"No se encontró geocodificación para: {addr}")
                return
            coords_list.append((res[0][1], res[0][2]))
        dist_matrix = compute_pairwise_distances(coords_list)
        best_order, best_cost, err = solve_tsp_bruteforce(dist_matrix)
        if err:
            st.error(err)
            return
        st.success(f"Orden óptimo calculado. Distancia aproximada: {best_cost/1000:.2f} km")
        ordered_coords = [coords_list[i] for i in best_order]
        dist_total, time_total, route_pts = final_routing_with_order(ordered_coords)
        if dist_total is None:
            st.error("Error al obtener la ruta final.")
            return
        st.write(f"Distancia total: {dist_total/1000:.2f} km")
        st.write(f"Tiempo total: ~{int(time_total/60)} min")
        render_map(route_pts, ordered_coords[0][0], ordered_coords[0][1],
                   ordered_coords[-1][0], ordered_coords[-1][1], color="blue")

# -----------------------------------------------------------------------------
# TAB: Tráfico en Ruta
# -----------------------------------------------------------------------------
def tab_trafico_ruta():
    st.subheader("Tráfico en Ruta")
    if not (st.session_state.get("origin_lat") and st.session_state.get("dest_lat")):
        st.warning("Primero calcula la ruta en 'Calcular ruta'.")
        return
    lat_mid = (st.session_state["origin_lat"] + st.session_state["dest_lat"]) / 2
    lon_mid = (st.session_state["origin_lon"] + st.session_state["dest_lon"]) / 2
    if st.button("Consultar Tráfico"):
        data_flow = tomtom_traffic_flow(lat_mid, lon_mid)
        if not data_flow or "flowSegmentData" not in data_flow:
            st.error("No se pudieron obtener datos de tráfico.")
            return
        flow = data_flow["flowSegmentData"]
        cur_speed = flow.get("currentSpeed", 0)
        free_speed = flow.get("freeFlowSpeed", 0)
        st.write(f"Velocidad actual: {cur_speed} km/h")
        st.write(f"Velocidad libre: {free_speed} km/h")
        if cur_speed < 20:
            st.error("Tráfico muy congestionado.")
        elif cur_speed < 50:
            st.warning("Tráfico moderado. Podrías tener retrasos.")
        else:
            st.info("Tráfico fluido.")

# -----------------------------------------------------------------------------
# TAB: Predicción de Retrasos
# -----------------------------------------------------------------------------
def tab_prediccion_retrasos():
    st.subheader("Predicción de retrasos")
    if not (st.session_state.get("distance_km") and st.session_state.get("duration_min")):
        st.warning("Primero calcula la ruta en 'Calcular ruta'.")
        return
    dist_km = st.session_state["distance_km"]
    base_time = st.session_state["duration_min"]
    lat_mid = (st.session_state["origin_lat"] + st.session_state["dest_lat"]) / 2
    lon_mid = (st.session_state["origin_lon"] + st.session_state["dest_lon"]) / 2
    st.write(f"Distancia: {dist_km:.2f} km")
    st.write(f"Tiempo base: ~{base_time:.2f} min")
    meteo = get_weather_open_meteo(lat_mid, lon_mid)
    if not meteo:
        st.warning("No se pudieron obtener datos meteorológicos. Se usarán valores por defecto.")
        temp, wind, precip, cloud = 20.0, 0.0, 0.0, 0.0
    else:
        temp = meteo["temp"]
        wind = meteo["wind"]
        precip = meteo["precip"]
        cloud = meteo["cloud"]
        st.write(f"Clima: Temp={temp:.1f}°C, Viento={wind:.1f} km/h, Precip={precip:.1f} mm, Nubosidad={cloud:.1f}%")
    if st.button("Calcular predicción"):
        pred_time = predict_time(dist_km, temp, wind, precip, cloud)
        if pred_time is None:
            st.error("Error en la predicción.")
            return
        st.success(f"Tiempo estimado (modelo): {pred_time:.2f} min")
        delay = pred_time - base_time
        if delay > 0:
            st.error(f"Retraso estimado: +{delay:.2f} min")
        else:
            st.info(f"Adelanto estimado: {abs(delay):.2f} min")

# -----------------------------------------------------------------------------
# TAB: Calculadora CAE
# -----------------------------------------------------------------------------
def tab_calculadora_cae():
    st.subheader("Calculadora CAE")
    kwh = st.number_input("kWh ahorrados", min_value=0.0, value=500.0, step=50.0)
    if st.button("Calcular ingresos"):
        kwh_val, cost_min, cost_max = calculate_cae(kwh)
        st.write(f"CAE generados: {kwh_val:.2f} kWh")
        st.write(f"Ingresos estimados: entre {cost_min:.2f} € y {cost_max:.2f} €")

# -----------------------------------------------------------------------------
# APP PRINCIPAL
# -----------------------------------------------------------------------------
def main_app():
    st.title("Trafiquea: Dashboard para Empresas Logísticas")
    st.markdown("""
    **Funciones disponibles:**
    - Calcular ruta (con selección de hora de salida y eventos en ruta)
    - Ruta alternativa (evitar carretera específica)
    - Calcular ruta completa (usando NLP para extraer direcciones y optimizar orden)
    - Tráfico en ruta (consultar datos de tráfico en el trayecto)
    - Predicción de retrasos (modelo realista con influencias meteorológicas)
    - Calculadora CAE (ingresos por kWh ahorrados)
    """)
    tabs = st.tabs([
        "Calcular ruta",
        "Ruta alternativa",
        "Calcular ruta completa",
        "Tráfico en ruta",
        "Predicción de retrasos",
        "Calculadora CAE"
    ])
    with tabs[0]:
        tab_mapa_trafico()
    with tabs[1]:
        tab_ruta_alternativa()
    with tabs[2]:
        tab_tsp_advanced()
    with tabs[3]:
        tab_trafico_ruta()
    with tabs[4]:
        tab_prediccion_retrasos()
    with tabs[5]:
        tab_calculadora_cae()

if __name__ == "__main__":
    # Inicializar algunos valores en session_state si no existen
    for key in ["origin_lat", "origin_lon", "dest_lat", "dest_lon", "distance_km", "duration_min", "route_points"]:
        if key not in st.session_state:
            st.session_state[key] = None
    main_app()
