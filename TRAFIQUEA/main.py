import streamlit as st
st.set_page_config(layout="wide")  # Debe ser la primera instrucción

import os
import requests
import numpy as np
import pandas as pd
import joblib
from prophet import Prophet
import itertools
from datetime import datetime
import folium
from streamlit_folium import st_folium

# -----------------------------------------------------------------------------
# CONFIGURACIÓN: API key de TomTom (almacenada en st.secrets)
# -----------------------------------------------------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

def get_tomtom_key():
    return st.secrets["tomtom"]["api_key"]

# -----------------------------------------------------------------------------
# Solicitar ubicación actual mediante JavaScript (fallback: Madrid)
# -----------------------------------------------------------------------------
def request_user_location():
    params = st.query_params
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

request_user_location()
params = st.query_params
if "lat" in params and "lon" in params:
    try:
        st.session_state["current_lat"] = float(params["lat"])
        st.session_state["current_lon"] = float(params["lon"])
    except:
        st.session_state["current_lat"] = 40.4167
        st.session_state["current_lon"] = -3.7033
else:
    st.session_state["current_lat"] = 40.4167
    st.session_state["current_lon"] = -3.7033

# -----------------------------------------------------------------------------
# MODELO PROPHET (SINTÉTICO) con factores moderados
# -----------------------------------------------------------------------------
@st.cache_resource
def load_prophet_model():
    model_path = os.path.join(BASE_DIR, "prophet_realistic.pkl")
    try:
        m = joblib.load(model_path)
        return m
    except:
        st.warning("Entrenando modelo Prophet con precipitaciones moderadas.")
        np.random.seed(42)
        dates = pd.date_range("2023-01-01", "2023-12-31", freq="D")
        n = len(dates)
        distance = np.random.uniform(1, 50, n)
        temp = np.random.uniform(-5, 35, n)
        wind = np.random.uniform(0, 50, n)
        precip = np.random.uniform(0, 20, n)
        cloud = np.random.uniform(0, 100, n)

        base_time = 1.2 * distance
        wind_factor = (wind / 50) * 3
        precip_factor = np.where(precip <= 10, (precip/20)*5, (precip/20)*8)
        cloud_factor = (cloud / 100) * 2
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

def predict_time(distance_km, temp, wind, precip, cloud, wind_dir=180):
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
    base_pred = forecast["yhat"].iloc[0]
    factor = 1.0
    if precip > 2:
        factor *= 1.05
    if wind > 10:
        factor *= 1.02 if wind_dir >= 180 else 0.98
    return base_pred * factor

def format_minutes(total_minutes: float) -> str:
    h = int(total_minutes // 60)
    m = int(total_minutes % 60)
    return f"{h}h {m}min" if h > 0 else f"{m}min"

# -----------------------------------------------------------------------------
# OPEN-METEO (obtener clima)
# -----------------------------------------------------------------------------
@st.cache_data(ttl=3600)
def get_weather_open_meteo(lat, lon):
    url = "https://api.open-meteo.com/v1/forecast"
    params = {
        "latitude": lat,
        "longitude": lon,
        "current_weather": True,
        "hourly": "temperature_2m,windspeed_10m,precipitation,cloudcover,winddirection_10m",
        "timezone": "auto"
    }
    r = requests.get(url, params=params)
    if r.status_code == 200:
        data = r.json()
        current = data.get("current_weather", {})
        wind_dir = current.get("winddirection", 180.0)
        return {
            "temp": current.get("temperature", 20.0),
            "wind": current.get("windspeed", 0.0),
            "wind_dir": wind_dir,
            "precip": data.get("hourly", {}).get("precipitation", [0])[0],
            "cloud": data.get("hourly", {}).get("cloudcover", [0])[0]
        }
    return None

# -----------------------------------------------------------------------------
# TOMTOM: SEARCH y ROUTING (solo "car")
# -----------------------------------------------------------------------------
@st.cache_data(ttl=600)
def tomtom_search(query, limit=5):
    if not query:
        return []
    tomtom_key = get_tomtom_key()
    url = f"https://api.tomtom.com/search/2/search/{query}.json"
    params = {"key": tomtom_key, "limit": limit, "language": "es-ES"}
    r = requests.get(url, params=params)
    if r.status_code == 200:
        data = r.json()
        results = []
        for item in data.get("results", []):
            addr = item.get("address", {}).get("freeformAddress", "")
            lat = item.get("position", {}).get("lat")
            lon = item.get("position", {}).get("lon")
            if lat and lon:
                results.append((addr, lat, lon))
        return results
    return []

@st.cache_data(ttl=600)
def tomtom_routing_api(origin_lat, origin_lon, dest_lat, dest_lon, depart_at=None):
    tomtom_key = get_tomtom_key()
    url = f"https://api.tomtom.com/routing/1/calculateRoute/{origin_lat},{origin_lon}:{dest_lat},{dest_lon}/json"
    params = {"key": tomtom_key, "traffic": "true", "travelMode": "car"}
    if depart_at:
        params["departAt"] = depart_at
    r = requests.get(url, params=params)
    if r.status_code == 200:
        return r.json()
    return None

@st.cache_data(ttl=600)
def tomtom_traffic_flow(lat, lon):
    tomtom_key = get_tomtom_key()
    url = "https://api.tomtom.com/traffic/services/4/flowSegmentData/relative0/10/json"
    params = {"key": tomtom_key, "point": f"{lat},{lon}"}
    r = requests.get(url, params=params)
    if r.status_code == 200:
        return r.json()
    return None

# -----------------------------------------------------------------------------
# TSP: Cálculo de rutas óptimas
# -----------------------------------------------------------------------------
@st.cache_data(ttl=600)
def get_distance_time(lat1, lon1, lat2, lon2):
    tomtom_key = get_tomtom_key()
    url = f"https://api.tomtom.com/routing/1/calculateRoute/{lat1},{lon1}:{lat2},{lon2}/json"
    params = {"key": tomtom_key, "traffic": "false", "travelMode": "car"}
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
    for i in range(n):
        for j in range(n):
            if i != j:
                d, _ = get_distance_time(coords_list[i][0], coords_list[i][1],
                                         coords_list[j][0], coords_list[j][1])
                dist_matrix[i][j] = d
    return dist_matrix

def solve_tsp_bruteforce(dist_matrix):
    n = len(dist_matrix)
    if n > 20:
        return None, None, "Máximo 20 direcciones permitidas."
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
    params = {"key": tomtom_key, "traffic": "true", "travelMode": "car"}
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
# Extracción de direcciones (FALLBACK por regex en español)
# -----------------------------------------------------------------------------
KEYWORDS_ES = {
    "calle","avenida","av","av.","avda","carretera","autovía","autovia",
    "camino","plaza","pza","paseo","polígono","carrer","bulevar",
    "rotonda","pasaje","peatonal","entrada","salida","nacional","provincial",
    "comarcal","km","kilómetro","autopista","via","vía","sendero","ruta",
    "urbanización","urbanizacion","colonia","sector","zona","valle","cerro",
    "barrio","puerta","bloque","edificio","escuela","instituto","mercado",
    "estación","estacion"
}

def fallback_regex_es(text: str):
    lines = [l.strip() for l in text.split("\n") if l.strip()]
    addresses = []
    for line in lines:
        low = line.lower()
        if any(kw in low for kw in KEYWORDS_ES):
            addresses.append(line)
    return addresses

def extract_addresses(text: str):
    addrs = fallback_regex_es(text)
    return addrs[:20]

# -----------------------------------------------------------------------------
# Funciones de tiempo: lista de horas y conversión a ISO 8601
# -----------------------------------------------------------------------------
def half_hour_list():
    times = []
    for h in range(24):
        times.append(f"{h:02d}:00")
        times.append(f"{h:02d}:30")
    return times

def parse_half_hour_string(s: str):
    hh, mm = s.split(":")
    return int(hh), int(mm)

def get_departure_iso(hh: int, mm: int) -> str:
    now = datetime.now()
    dt = now.replace(minute=mm, second=0, microsecond=0)
    if hh < now.hour or (hh == now.hour and mm <= now.minute):
        dt = dt.replace(day=now.day+1, hour=hh)
    else:
        dt = dt.replace(hour=hh)
    return dt.isoformat()

# -----------------------------------------------------------------------------
# Renderizar mapa con Folium (se mantiene el mapa fijo)
# -----------------------------------------------------------------------------
def render_map(route_points, lat_start, lon_start, lat_end=None, lon_end=None, color="blue"):
    m_map = folium.Map(location=[lat_start, lon_start], zoom_start=12, control_scale=True)
    # Usamos TomTom como tile layer
    tile_url = f"https://api.tomtom.com/map/1/tile/basic/main/{{z}}/{{x}}/{{y}}.png?key={get_tomtom_key()}"
    folium.TileLayer(tiles=tile_url, attr="TomTom").add_to(m_map)
    # Marcador de ubicación actual
    folium.Marker((lat_start, lon_start), tooltip="Tu ubicación actual", icon=folium.Icon(color="blue")).add_to(m_map)
    if route_points:
        folium.PolyLine(route_points, color=color, weight=5).add_to(m_map)
        folium.Marker(route_points[0], tooltip="Origen", icon=folium.Icon(color="green")).add_to(m_map)
        if lat_end is not None and lon_end is not None:
            folium.Marker((lat_end, lon_end), tooltip="Destino", icon=folium.Icon(color="red")).add_to(m_map)
    st_folium(m_map, width=700)

# -----------------------------------------------------------------------------
# TAB 1: Calcular ruta
# -----------------------------------------------------------------------------
def tab_calcular_ruta():
    st.header("Calcular ruta")
    origin_query = st.text_input("Origen")
    dest_query = st.text_input("Destino")
    selected_time = st.selectbox("Hora de salida", half_hour_list(), index=18)
    if st.button("Calcular ruta", key="btn_calcular_ruta"):
        hh, mm = parse_half_hour_string(selected_time)
        depart_at_iso = get_departure_iso(hh, mm)
        origin_res = tomtom_search(origin_query)
        if not origin_res:
            st.error("No se encontraron sugerencias para el origen.")
            return
        o_lat, o_lon = origin_res[0][1], origin_res[0][2]
        dest_res = tomtom_search(dest_query)
        if not dest_res:
            st.error("No se encontraron sugerencias para el destino.")
            return
        d_lat, d_lon = dest_res[0][1], dest_res[0][2]
        routing_data = tomtom_routing_api(o_lat, o_lon, d_lat, d_lon, depart_at=depart_at_iso)
        if not routing_data or "routes" not in routing_data:
            st.error("No se pudo obtener la ruta con TomTom.")
            return

        route = routing_data["routes"][0]
        dist_m = route["summary"]["lengthInMeters"]
        time_s = route["summary"]["travelTimeInSeconds"]
        dist_km = dist_m / 1000.0
        base_minutes = time_s / 60.0
        st.success(f"Ruta calculada. Distancia: {dist_km:.2f} km, Tiempo base: {format_minutes(base_minutes)}")

        st.session_state["origin_lat"] = o_lat
        st.session_state["origin_lon"] = o_lon
        st.session_state["dest_lat"] = d_lat
        st.session_state["dest_lon"] = d_lon
        st.session_state["distance_km"] = dist_km
        st.session_state["duration_min"] = base_minutes

        route_points = []
        for leg in route["legs"]:
            for point in leg["points"]:
                route_points.append((point["latitude"], point["longitude"]))
        st.session_state["route_points"] = route_points

        # Clima y tráfico en el punto medio
        lat_mid = (o_lat + d_lat) / 2
        lon_mid = (o_lon + d_lon) / 2
        weather = get_weather_open_meteo(lat_mid, lon_mid) or {}
        temp = weather.get("temp", 20.0)
        wind = weather.get("wind", 0.0)
        precip = weather.get("precip", 0.0)
        cloud = weather.get("cloud", 0.0)
        traffic_data = tomtom_traffic_flow(lat_mid, lon_mid)
        speed = 50
        if traffic_data and "flowSegmentData" in traffic_data:
            speed = traffic_data["flowSegmentData"].get("currentSpeed", 50)
        st.write(f"Resultado: Distancia: {dist_km:.2f} km, Tiempo base: {format_minutes(base_minutes)}")
        st.write(f"Temperatura: {temp:.1f}ºC | Viento: {wind:.1f} km/h | Precipitaciones: {precip:.1f} mm | Nubosidad: {cloud:.1f}%")
        st.write(f"Velocidad de tráfico aprox: {speed} km/h")

        render_map(st.session_state["route_points"],
                   st.session_state["current_lat"],
                   st.session_state["current_lon"],
                   d_lat, d_lon, color="blue")
    else:
        if "route_points" in st.session_state and st.session_state["route_points"]:
            render_map(st.session_state["route_points"],
                       st.session_state["current_lat"],
                       st.session_state["current_lon"],
                       st.session_state.get("dest_lat"),
                       st.session_state.get("dest_lon"),
                       color="blue")
        else:
            st.info("Introduce Origen y Destino y pulsa 'Calcular ruta'.")

# -----------------------------------------------------------------------------
# TAB 2: Calcular ruta completa (TSP) – Mostrar mapa con ruta final
# -----------------------------------------------------------------------------
def tab_calcular_ruta_completa():
    st.header("Calcular ruta completa (TSP)")
    st.write("Introduce hasta 20 direcciones (una por línea).")
    texto = st.text_area("Direcciones:")
    if st.button("Calcular TSP", key="btn_tsp"):
        addresses = extract_addresses(texto)
        if not addresses:
            st.error("No se detectaron direcciones.")
            return
        if len(addresses) > 20:
            st.error("Máximo 20 direcciones permitidas.")
            return
        coords_list = []
        for addr in addresses:
            res = tomtom_search(addr)
            if not res:
                st.error(f"No se pudo geocodificar: {addr}")
                return
            coords_list.append((res[0][1], res[0][2]))
        dist_matrix = compute_pairwise_distances(coords_list)
        best_order, best_cost, err = solve_tsp_bruteforce(dist_matrix)
        if err:
            st.error(err)
            return
        st.success(f"Orden óptimo calculado. Distancia ~{best_cost/1000:.2f} km")
        ordered_coords = [coords_list[i] for i in best_order]
        dist_m, time_s, route_pts = final_routing_with_order(ordered_coords)
        if dist_m is None:
            st.error("No se pudo obtener la ruta final con TomTom.")
            return
        st.info(f"Distancia total: {dist_m/1000:.2f} km, Tiempo total: {format_minutes(time_s/60.0)}")
        st.session_state["route_points"] = route_pts
        st.session_state["origin_lat"] = ordered_coords[0][0]
        st.session_state["origin_lon"] = ordered_coords[0][1]
        st.session_state["dest_lat"] = ordered_coords[-1][0]
        st.session_state["dest_lon"] = ordered_coords[-1][1]
        render_map(route_pts,
                   st.session_state["current_lat"],
                   st.session_state["current_lon"],
                   st.session_state["dest_lat"],
                   st.session_state["dest_lon"],
                   color="red")
    else:
        if "route_points" in st.session_state and st.session_state["route_points"]:
            st.info("Mostrando la última ruta TSP calculada:")
            render_map(st.session_state["route_points"],
                       st.session_state["current_lat"],
                       st.session_state["current_lon"],
                       st.session_state.get("dest_lat"),
                       st.session_state.get("dest_lon"),
                       color="red")
        else:
            st.info("Introduce direcciones y pulsa 'Calcular TSP'.")

# -----------------------------------------------------------------------------
# TAB 3: Predicción de retrasos (sin mapa)
# -----------------------------------------------------------------------------
def tab_prediccion_retrasos():
    st.header("Predicción de retrasos")
    if "route_points" not in st.session_state or not st.session_state["route_points"]:
        st.warning("Primero calcula la ruta en 'Calcular ruta'.")
        return
    dist_km = st.session_state.get("distance_km", 0.0)
    base_time = st.session_state.get("duration_min", 0.0)
    lat_mid = (st.session_state.get("origin_lat", 0.0) + st.session_state.get("dest_lat", 0.0)) / 2
    lon_mid = (st.session_state.get("origin_lon", 0.0) + st.session_state.get("dest_lon", 0.0)) / 2
    st.write(f"Distancia: {dist_km:.2f} km, Tiempo base: {format_minutes(base_time)}")
    weather = get_weather_open_meteo(lat_mid, lon_mid) or {}
    temp = weather.get("temp", 20.0)
    wind = weather.get("wind", 0.0)
    precip = weather.get("precip", 0.0)
    cloud = weather.get("cloud", 0.0)
    st.write(f"Temperatura: {temp:.1f}ºC | Viento: {wind:.1f} km/h | Precipitaciones: {precip:.1f} mm | Nubosidad: {cloud:.1f}%")
    if st.button("Calcular predicción de retrasos"):
        final_time = predict_time(dist_km, temp, wind, precip, cloud, wind_dir=180)
        if final_time is None:
            st.error("No se pudo calcular la predicción.")
            return
        st.success(f"Tiempo estimado (modelo): {format_minutes(final_time)}")
        delay = final_time - base_time
        if delay > 0:
            st.warning(f"Retraso estimado: +{format_minutes(delay)}")
        else:
            st.info(f"Adelanto estimado: {format_minutes(abs(delay))}")

# -----------------------------------------------------------------------------
# TAB 4: Consumo de Combustible
# -----------------------------------------------------------------------------
def tab_consumo():
    st.header("Consumo de Combustible")
    if "distance_km" not in st.session_state or st.session_state["distance_km"] is None:
        st.warning("Primero calcula la ruta en 'Calcular ruta'.")
        return
    veh_type = st.selectbox("Tipo de vehículo", ["Furgoneta", "Camión"])
    dist_km = st.session_state["distance_km"]
    weather = get_weather_open_meteo(
        (st.session_state.get("origin_lat", 0) + st.session_state.get("dest_lat", 0)) / 2,
        (st.session_state.get("origin_lon", 0) + st.session_state.get("dest_lon", 0)) / 2
    ) or {}
    precip = weather.get("precip", 0.0)
    wind = weather.get("wind", 0.0)
    base_consumo = 0.08 if veh_type == "Furgoneta" else 0.25
    consumo = dist_km * base_consumo
    if precip > 2:
        consumo *= 1.05
    if wind > 10:
        consumo *= 1.02
    price = 1.70
    cost = consumo * price
    st.info(f"Distancia: {dist_km:.2f} km")
    st.write(f"Consumo estimado: {consumo:.2f} L")
    st.write(f"Coste estimado: {cost:.2f} € (a {price:.2f} €/L)")

# -----------------------------------------------------------------------------
# TAB 5: Calculadora CAE
# -----------------------------------------------------------------------------
def tab_calculadora_cae():
    st.header("Calculadora CAE")
    kwh = st.number_input("kWh ahorrados", min_value=0.0, value=500.0, step=50.0)
    if st.button("Calcular ingresos", key="btn_calc_cae"):
        cost_min = kwh * 0.115
        cost_max = kwh * 0.14
        st.info(f"CAE generados: {kwh:.2f} kWh")
        st.write(f"Ingresos estimados: entre {cost_min:.2f} € y {cost_max:.2f} €")

# -----------------------------------------------------------------------------
# APP PRINCIPAL: Navegación lateral en el sidebar
# -----------------------------------------------------------------------------
def main_app():
    st.title("Trafiquea: Dashboard para Empresas Logísticas")
    option = st.sidebar.radio("Navegación", [
        "Calcular ruta",
        "Calcular ruta completa",
        "Predicción de retrasos",
        "Consumo",
        "Calculadora CAE"
    ])
    if option == "Calcular ruta":
        tab_calcular_ruta()
    elif option == "Calcular ruta completa":
        tab_calcular_ruta_completa()
    elif option == "Predicción de retrasos":
        tab_prediccion_retrasos()
    elif option == "Consumo":
        tab_consumo()
    elif option == "Calculadora CAE":
        tab_calculadora_cae()

if __name__ == "__main__":
    for key in ["origin_lat", "origin_lon", "dest_lat", "dest_lon", "distance_km", "duration_min", "route_points", "current_lat", "current_lon"]:
        if key not in st.session_state:
            st.session_state[key] = None
    main_app()
