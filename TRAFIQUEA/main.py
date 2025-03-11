import os
import streamlit as st
import folium
from streamlit_folium import st_folium
import requests
import numpy as np
import pandas as pd
import joblib
from prophet import Prophet
import itertools
from datetime import datetime

# -----------------------------------------------------------------------------
# Cargar spaCy en español (es_core_news_sm). Si falla, no hay fallback a keywords.
# -----------------------------------------------------------------------------
try:
    import spacy
    nlp_es = spacy.load("es_core_news_sm")
    spacy_loaded = True
except Exception:
    nlp_es = None
    spacy_loaded = False
    st.warning("No se pudo cargar spaCy en español. No se podrá extraer direcciones si no está instalado.")

# -----------------------------------------------------------------------------
# CONFIGURACIÓN: API key de TomTom (definida en secrets.toml)
# -----------------------------------------------------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

def get_tomtom_key():
    return st.secrets["tomtom"]["api_key"]

# -----------------------------------------------------------------------------
# Solicitar ubicación actual (JavaScript). Fallback: Madrid
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
# MODELO PROPHET SINTÉTICO (con precipitaciones fuertes)
# -----------------------------------------------------------------------------
@st.cache_resource
def load_prophet_model():
    model_path = os.path.join(BASE_DIR, "prophet_realistic.pkl")
    try:
        m = joblib.load(model_path)
        return m
    except:
        st.warning("Entrenando modelo Prophet con precipitaciones fuertes.")
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
        # Manejar precip >10 con np.where
        precip_factor = np.where(precip <= 10,
                                 (precip / 20) * 5,
                                 (precip / 20) * 15)
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
    """
    Ajuste final:
    - Se calcula un tiempo base con Prophet
    - Luego se ajusta en función de la dirección del viento y precipitaciones
    """
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

    # Ajuste manual:
    final_time = base_pred
    # Si precip > 2 mm => +10% de tiempo
    if precip > 2:
        final_time *= 1.1
    # Si wind_dir < 180 => se asume viento "a favor" => -5% de tiempo
    # si wind_dir >= 180 => +5%
    if wind_dir < 180:
        final_time *= 0.95
    else:
        final_time *= 1.05
    return final_time

# -----------------------------------------------------------------------------
# OPEN-METEO (incluyendo winddirection_10m)
# -----------------------------------------------------------------------------
@st.cache_data(ttl=3600)
def get_weather_open_meteo(lat, lon):
    url = "https://api.open-meteo.com/v1/forecast"
    params = {
        "latitude": lat,
        "longitude": lon,
        "current_weather": True,
        "hourly": "temperature_2m,windspeed_10m,winddirection_10m,precipitation,cloudcover",
        "timezone": "auto"
    }
    r = requests.get(url, params=params)
    if r.status_code == 200:
        data = r.json()
        current = data.get("current_weather", {})
        # Extraer direction si existe
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
# TOMTOM: SEARCH y ROUTING (car)
# -----------------------------------------------------------------------------
@st.cache_data(ttl=600)
def tomtom_search(query, limit=5):
    if not query:
        return []
    tomtom_key = st.secrets["tomtom"]["api_key"]
    url = f"https://api.tomtom.com/search/2/search/{query}.json"
    params = {
        "key": tomtom_key,
        "limit": limit,
        "language": "es-ES"
    }
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
    params = {
        "key": tomtom_key,
        "traffic": "true",
        "travelMode": "car"
    }
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
# TSP
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
# NLP con spaCy en español (sin fallback)
# -----------------------------------------------------------------------------
def extract_addresses_es(text: str):
    if not spacy_loaded:
        return []
    doc = nlp_es(text)
    # Consideramos cada oración "sent" y la tomamos como dirección si no está vacía
    # En un caso más avanzado, se podrían buscar entidades "LOC" o "GPE".
    # Aquí haremos algo simple: si la frase no está vacía, la consideramos dirección.
    addresses = []
    for sent in doc.sents:
        line = sent.text.strip()
        if line:
            addresses.append(line)
    return addresses

# -----------------------------------------------------------------------------
# Generar lista de horas en incrementos de 30 min
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
    # si la hora es menor o igual => día siguiente
    if hh < now.hour or (hh == now.hour and mm <= now.minute):
        dt = dt.replace(day=now.day+1, hour=hh)
    else:
        dt = dt.replace(hour=hh)
    return dt.isoformat()

# -----------------------------------------------------------------------------
# Renderizar mapa (con key distinto para cada pestaña)
# -----------------------------------------------------------------------------
def render_map(route_points, lat_start, lon_start, lat_end=None, lon_end=None, color="blue", map_key="map"):
    center_lat = lat_start if lat_end is None else (lat_start + lat_end)/2
    center_lon = lon_start if lon_end is None else (lon_start + lon_end)/2
    m_map = folium.Map(location=[center_lat, center_lon], zoom_start=12)
    tile_url = f"https://api.tomtom.com/map/1/tile/basic/main/{{z}}/{{x}}/{{y}}.png?key={get_tomtom_key()}"
    folium.TileLayer(tiles=tile_url, attr="TomTom").add_to(m_map)
    folium.PolyLine(route_points, color=color, weight=5).add_to(m_map)
    folium.Marker((lat_start, lon_start), tooltip="Origen", icon=folium.Icon(color="green")).add_to(m_map)
    if lat_end and lon_end:
        folium.Marker((lat_end, lon_end), tooltip="Destino", icon=folium.Icon(color="red")).add_to(m_map)
    st_folium(m_map, width=700, key=map_key)

# -----------------------------------------------------------------------------
# TAB 1: Calcular ruta
# -----------------------------------------------------------------------------
def tab_calcular_ruta():
    st.write("Introduce Origen y Destino (en español). Pulsa Enter para ver sugerencias.")
    origin_query = st.text_input("Origen")
    dest_query = st.text_input("Destino")

    # Dropdown de horas en incrementos de 30 min
    hh_mm_list = half_hour_list()
    selected_time = st.selectbox("Hora de salida", hh_mm_list, index=18)  # 09:00 approx

    if st.button("Calcular ruta", key="btn_calcular_ruta"):
        hh, mm = parse_half_hour_string(selected_time)
        depart_at_iso = get_departure_iso(hh, mm)

        origin_res = tomtom_search(origin_query)
        if not origin_res:
            st.error("No se encontraron sugerencias para el origen.")
            return
        if len(origin_res) > 1:
            sel_o = st.selectbox("Sugerencias Origen", [r[0] for r in origin_res], key="sel_o")
            o_lat, o_lon = None, None
            for r in origin_res:
                if r[0] == sel_o:
                    o_lat, o_lon = r[1], r[2]
                    break
        else:
            o_lat, o_lon = origin_res[0][1], origin_res[0][2]

        dest_res = tomtom_search(dest_query)
        if not dest_res:
            st.error("No se encontraron sugerencias para el destino.")
            return
        if len(dest_res) > 1:
            sel_d = st.selectbox("Sugerencias Destino", [r[0] for r in dest_res], key="sel_d")
            d_lat, d_lon = None, None
            for r in dest_res:
                if r[0] == sel_d:
                    d_lat, d_lon = r[1], r[2]
                    break
        else:
            d_lat, d_lon = dest_res[0][1], dest_res[0][2]

        if o_lat and o_lon and d_lat and d_lon:
            routing_data = tomtom_routing_api(o_lat, o_lon, d_lat, d_lon, depart_at=depart_at_iso)
            if not routing_data or "routes" not in routing_data:
                st.error("No se pudo obtener la ruta con TomTom.")
                return
            route = routing_data["routes"][0]
            dist_m = route["summary"]["lengthInMeters"]
            time_s = route["summary"]["travelTimeInSeconds"]
            st.success(f"Ruta calculada. Distancia: {dist_m/1000:.2f} km, Tiempo base: ~{int(time_s/60)} min")

            st.session_state["origin_lat"] = o_lat
            st.session_state["origin_lon"] = o_lon
            st.session_state["dest_lat"] = d_lat
            st.session_state["dest_lon"] = d_lon
            st.session_state["distance_km"] = dist_m/1000.0
            st.session_state["duration_min"] = time_s/60.0

            route_points = []
            for leg in route["legs"]:
                for point in leg["points"]:
                    route_points.append((point["latitude"], point["longitude"]))
            st.session_state["route_points"] = route_points

            # Clima y tráfico
            lat_mid = (o_lat + d_lat)/2
            lon_mid = (o_lon + d_lon)/2
            weather = get_weather_open_meteo(lat_mid, lon_mid) or {}
            temp = weather.get("temp", 20.0)
            wind = weather.get("wind", 0.0)
            wdir = weather.get("wind_dir", 180.0)
            precip = weather.get("precip", 0.0)
            cloud = weather.get("cloud", 0.0)

            traffic_data = tomtom_traffic_flow(lat_mid, lon_mid)
            speed = 50
            if traffic_data and "flowSegmentData" in traffic_data:
                speed = traffic_data["flowSegmentData"].get("currentSpeed", 50)

            # Mostrar en una sola línea
            clima_line = (f"**Temperatura**: {temp:.1f}ºC | "
                          f"**Viento**: {wind:.1f} km/h | "
                          f"**Precipitaciones**: {precip:.1f} mm | "
                          f"**Nubosidad**: {cloud:.1f}%")
            st.write(clima_line)
            st.write(f"Velocidad de tráfico en la zona: ~{speed} km/h")

            render_map(route_points, o_lat, o_lon, d_lat, d_lon, color="blue", map_key="calc_ruta_map")
    elif "route_points" in st.session_state and st.session_state["route_points"]:
        render_map(st.session_state["route_points"],
                   st.session_state.get("origin_lat", 0),
                   st.session_state.get("origin_lon", 0),
                   st.session_state.get("dest_lat", 0),
                   st.session_state.get("dest_lon", 0),
                   color="blue",
                   map_key="calc_ruta_map")

# -----------------------------------------------------------------------------
# TAB 2: Calcular ruta completa (SIN mostrar mapa)
# -----------------------------------------------------------------------------
def tab_calcular_ruta_completa():
    st.write("Introduce direcciones (en español). Cada línea = 1 dirección.")
    texto = st.text_area("Direcciones:")
    if st.button("Calcular TSP", key="btn_calcular_tsp"):
        if not spacy_loaded:
            st.error("No se pudo cargar spaCy en español, no se extraerán direcciones.")
            return
        addresses = extract_addresses_es(texto)
        if not addresses:
            st.error("No te he entendido, por favor, inténtalo de nuevo.")
            return
        if len(addresses) > 8:
            st.error("Máx 8 direcciones.")
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
        st.info(f"Distancia total: {dist_m/1000:.2f} km | Tiempo total: ~{int(time_s/60)} min")

        st.session_state["route_points"] = route_pts
        st.session_state["origin_lat"] = ordered_coords[0][0]
        st.session_state["origin_lon"] = ordered_coords[0][1]
        st.session_state["dest_lat"] = ordered_coords[-1][0]
        st.session_state["dest_lon"] = ordered_coords[-1][1]

        # NO se muestra mapa en esta pestaña

# -----------------------------------------------------------------------------
# TAB 3: Predicción de retrasos (SIN mostrar mapa)
# -----------------------------------------------------------------------------
def tab_prediccion_retrasos():
    if "route_points" not in st.session_state or not st.session_state["route_points"]:
        st.warning("Primero calcula la ruta en 'Calcular ruta'.")
        return

    dist_km = st.session_state.get("distance_km", 0)
    base_time = st.session_state.get("duration_min", 0)
    lat_mid = (st.session_state.get("origin_lat", 0) + st.session_state.get("dest_lat", 0)) / 2
    lon_mid = (st.session_state.get("origin_lon", 0) + st.session_state.get("dest_lon", 0)) / 2

    st.write(f"Distancia: {dist_km:.2f} km | Tiempo base: ~{base_time:.2f} min")

    weather = get_weather_open_meteo(lat_mid, lon_mid) or {}
    temp = weather.get("temp", 20.0)
    wind = weather.get("wind", 0.0)
    wdir = weather.get("wind_dir", 180.0)
    precip = weather.get("precip", 0.0)
    cloud = weather.get("cloud", 0.0)

    clima_line = (f"**Temperatura**: {temp:.1f}ºC | "
                  f"**Viento**: {wind:.1f} km/h | "
                  f"**Precipitaciones**: {precip:.1f} mm | "
                  f"**Nubosidad**: {cloud:.1f}%")
    st.write(clima_line)

    if st.button("Calcular predicción de retrasos", key="btn_prediccion"):
        final_time = predict_time(dist_km, temp, wind, precip, cloud, wind_dir=wdir)
        if final_time is None:
            st.error("No se pudo calcular la predicción.")
            return
        st.success(f"Tiempo estimado (con ajustes): {final_time:.2f} min")
        delay = final_time - base_time
        if delay > 0:
            st.warning(f"Retraso estimado: +{delay:.2f} min")
        else:
            st.info(f"Adelanto estimado: {abs(delay):.2f} min")

# -----------------------------------------------------------------------------
# TAB EXTRA: Consumo y Costes (Combustible)
# -----------------------------------------------------------------------------
def get_fuel_price():
    # Ejemplo ficticio: 1.70 €/L
    return 1.70

def tab_consumo_costes():
    st.write("Estimación de Consumo de Combustible y Costes")
    if "distance_km" not in st.session_state or st.session_state["distance_km"] is None:
        st.warning("Primero calcula la ruta en 'Calcular ruta'.")
        return
    dist_km = st.session_state["distance_km"]
    lat_mid = (st.session_state.get("origin_lat", 0) + st.session_state.get("dest_lat", 0)) / 2
    lon_mid = (st.session_state.get("origin_lon", 0) + st.session_state.get("dest_lon", 0)) / 2
    weather = get_weather_open_meteo(lat_mid, lon_mid) or {}
    wind_dir = weather.get("wind_dir", 180.0)
    precip = weather.get("precip", 0.0)

    # Consumo base => 0.06 L/km (6 L/100km)
    consumo_base = dist_km * 0.06
    # Ajuste por precip => si > 2 => +10%
    if precip > 2:
        consumo_base *= 1.1
    # Ajuste por viento => si wind_dir < 180 => -5%, else +5%
    if wind_dir < 180:
        consumo_base *= 0.95
    else:
        consumo_base *= 1.05

    fuel_price = get_fuel_price()  # 1.70 €/L
    coste_est = consumo_base * fuel_price
    st.info(f"Distancia: {dist_km:.2f} km")
    st.write(f"Consumo estimado: {consumo_base:.2f} L")
    st.write(f"Precio estimado (a {fuel_price:.2f} €/L): {coste_est:.2f} €")

# -----------------------------------------------------------------------------
# APP PRINCIPAL
# -----------------------------------------------------------------------------
def main_app():
    st.title("Trafiquea: Dashboard en Español (30min increments, spaCy, sin fallback)")

    st.markdown("""
    **Funciones**:
    - **Calcular ruta**: Origen, Destino, Hora (incrementos de 30 min). Muestra mapa, clima y tráfico.
    - **Calcular ruta completa**: TSP sin mostrar mapa ni clima/tráfico.
    - **Predicción de retrasos**: Ajusta tiempo base en función de precipitaciones y dirección del viento (sin mostrar mapa).
    - **Consumo y Costes**: Calcula consumo (L) y precio estimado en base a la distancia y condiciones climáticas.
    """)

    tabs = st.tabs([
        "Calcular ruta",
        "Calcular ruta completa",
        "Predicción de retrasos",
        "Consumo y Costes"
    ])
    with tabs[0]:
        tab_calcular_ruta()
    with tabs[1]:
        tab_calcular_ruta_completa()
    with tabs[2]:
        tab_prediccion_retrasos()
    with tabs[3]:
        tab_consumo_costes()

if __name__ == "__main__":
    for key in ["origin_lat", "origin_lon", "dest_lat", "dest_lon", "distance_km", "duration_min", "route_points"]:
        if key not in st.session_state:
            st.session_state[key] = None
    main_app()
