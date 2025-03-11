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
# Cargar Stanza en español, fallback a un método simple (regex) si no funciona
# -----------------------------------------------------------------------------
try:
    import stanza
    stanza.download("es")  # Descargar modelo español si no está
    nlp_stanza = stanza.Pipeline("es", processors="tokenize,mwt,pos,lemma")
    stanza_loaded = True
except Exception:
    nlp_stanza = None
    stanza_loaded = False
    st.warning("No se pudo cargar Stanza en español. Se usará un fallback regex.")

# -----------------------------------------------------------------------------
# CONFIG: API key de TomTom
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
# MODELO PROPHET (SINTÉTICO) con precipitaciones fuertes
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
# Formatear minutos a "Xh Ymin"
# -----------------------------------------------------------------------------
def format_minutes(total_minutes: float) -> str:
    h = int(total_minutes // 60)
    m = int(total_minutes % 60)
    if h > 0:
        return f"{h}h {m}min"
    else:
        return f"{m}min"

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
# TOMTOM
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
# MATRIX ROUTING / WAYPOINT OPTIMIZATION
# -----------------------------------------------------------------------------
@st.cache_data(ttl=600)
def tomtom_matrix_routing(coords_list):
    """
    Llamada a TomTom Matrix Routing.
    POST /routing/1/matrix/ con "origins" y "destinations" y "routeType=shortest".
    Retornamos la matriz de distancias y tiempos para luego optimizar localmente.
    """
    if len(coords_list) > 50:
        return None, "Se excede el límite de 50 puntos en Matrix Routing."

    tomtom_key = get_tomtom_key()
    url = f"https://api.tomtom.com/routing/1/matrix/json?key={tomtom_key}&routeType=shortest"
    # Construir payload
    origins = []
    destinations = []
    for c in coords_list:
        origins.append({"point": {"latitude": c[0], "longitude": c[1]}})
        destinations.append({"point": {"latitude": c[0], "longitude": c[1]}})

    payload = {
        "origins": origins,
        "destinations": destinations
    }

    r = requests.post(url, json=payload)
    if r.status_code != 200:
        return None, f"Error {r.status_code} en TomTom Matrix."

    data = r.json()
    # Extraer la matriz
    if "matrix" not in data:
        return None, "No se encontró 'matrix' en la respuesta."

    matrix = data["matrix"]
    n = len(coords_list)
    dist_matrix = [[0]*n for _ in range(n)]
    # matrix es un array con n*m. Indice i*m + j
    for i in range(n):
        for j in range(n):
            entry = matrix[i*n + j]
            if "response" not in entry:
                dist_matrix[i][j] = 999999
            else:
                dist_matrix[i][j] = entry["response"].get("routeSummary", {}).get("lengthInMeters", 999999)
    return dist_matrix, None

# -----------------------------------------------------------------------------
# TSP local (usando la matriz de TomTom)
# -----------------------------------------------------------------------------
def solve_tsp_matrix(dist_matrix):
    """
    dist_matrix es NxN, con distancias en metros.
    Buscamos el orden que minimice la distancia total.
    """
    n = len(dist_matrix)
    if n > 8:
        return None, None, "Más de 8 direcciones, la complejidad es alta. (Demo)."
    nodes = list(range(n))
    best_order = None
    best_cost = float("inf")
    import itertools
    for perm in itertools.permutations(nodes):
        cost = sum(dist_matrix[perm[i]][perm[i+1]] for i in range(n-1))
        if cost < best_cost:
            best_cost = cost
            best_order = perm
    return best_order, best_cost, None

# -----------------------------------------------------------------------------
# Calcular la ruta final con el orden hallado (pintar en el mapa)
# -----------------------------------------------------------------------------
def build_final_route_tomtom(coords_list, order):
    """
    Llama a tomtom_routing_api en secuencia, para obtener la polyline unida.
    """
    route_points = []
    total_dist = 0
    total_time = 0
    for i in range(len(order)-1):
        idx1 = order[i]
        idx2 = order[i+1]
        lat1, lon1 = coords_list[idx1]
        lat2, lon2 = coords_list[idx2]
        routing_data = tomtom_routing_api(lat1, lon1, lat2, lon2)
        if not routing_data or "routes" not in routing_data:
            continue
        route = routing_data["routes"][0]
        total_dist += route["summary"]["lengthInMeters"]
        total_time += route["summary"]["travelTimeInSeconds"]
        partial_points = []
        for leg in route["legs"]:
            for point in leg["points"]:
                partial_points.append((point["latitude"], point["longitude"]))
        if i > 0:
            # Evitar duplicar el primer punto
            partial_points = partial_points[1:]
        route_points += partial_points
    return route_points, total_dist, total_time

# -----------------------------------------------------------------------------
# NLP con Stanza en español o fallback regex
# -----------------------------------------------------------------------------
KEYWORDS_ES = {
    "calle","avenida","av","av.","avda","carretera","autovía","autovia",
    "camino","plaza","pza","paseo","polígono","poligono","carrer","bulevar",
    "rotonda","pasaje","peatonal","entrada","salida","nacional","provincial",
    "comarcal","km","kilómetro","autopista","via","vía","sendero","ruta",
    "urb.","urbanización","urbanizacion","colonia","sector","zona","valle","cerro",
    "barrio","puerta","bloque","edificio","escuela","instituto","mercado",
    "estación","estacion"
}

def stanza_extract_addresses_es(text: str):
    if not stanza_loaded:
        return None
    doc = nlp_stanza(text)
    addresses = []
    for sentence in doc.sentences:
        line = sentence.text.strip()
        if line:
            addresses.append(line)
    return addresses

def fallback_regex_es(text: str):
    lines = [l.strip() for l in text.split("\n") if l.strip()]
    addresses = []
    for line in lines:
        low = line.lower()
        if any(kw in low for kw in KEYWORDS_ES):
            addresses.append(line)
    return addresses

def extract_addresses_stanza(text: str):
    # Primero Stanza
    addrs_stanza = stanza_extract_addresses_es(text)
    if addrs_stanza and len(addrs_stanza) > 0:
        return addrs_stanza
    # Fallback
    addrs_fallback = fallback_regex_es(text)
    if addrs_fallback and len(addrs_fallback) > 0:
        return addrs_fallback
    return []

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
    if hh < now.hour or (hh == now.hour and mm <= now.minute):
        dt = dt.replace(day=now.day+1, hour=hh)
    else:
        dt = dt.replace(hour=hh)
    return dt.isoformat()

# -----------------------------------------------------------------------------
# Renderizar mapa
# -----------------------------------------------------------------------------
def render_map(route_points, lat_start, lon_start, lat_end=None, lon_end=None, color="blue", map_key="map"):
    m_map = folium.Map(location=[lat_start, lon_start], zoom_start=12)
    tile_url = f"https://api.tomtom.com/map/1/tile/basic/main/{{z}}/{{x}}/{{y}}.png?key={get_tomtom_key()}"
    folium.TileLayer(tiles=tile_url, attr="TomTom").add_to(m_map)

    # Marcador ubicación actual
    folium.Marker((lat_start, lon_start), tooltip="Tu ubicación", icon=folium.Icon(color="blue")).add_to(m_map)

    if route_points:
        folium.PolyLine(route_points, color=color, weight=5).add_to(m_map)
        folium.Marker(route_points[0], tooltip="Origen", icon=folium.Icon(color="green")).add_to(m_map)
        if lat_end and lon_end:
            folium.Marker((lat_end, lon_end), tooltip="Destino", icon=folium.Icon(color="red")).add_to(m_map)

    st_folium(m_map, width=700, key=map_key)

# -----------------------------------------------------------------------------
# TABS
# -----------------------------------------------------------------------------

def tab_calcular_ruta():
    st.write("Tu ubicación actual se muestra en azul si no has calculado ruta.")
    origin_query = st.text_input("Origen")
    dest_query = st.text_input("Destino")

    hh_mm_list = half_hour_list()
    selected_time = st.selectbox("Hora de salida", hh_mm_list, index=18)

    if st.button("Calcular ruta"):
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
        st.success(f"Ruta calculada. Distancia: {dist_km:.2f} km, Tiempo base: ~{format_minutes(base_minutes)}")

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

        # Clima y tráfico
        lat_mid = (o_lat + d_lat)/2
        lon_mid = (o_lon + d_lon)/2
        weather = get_weather_open_meteo(lat_mid, lon_mid) or {}
        temp = weather.get("temp", 20.0)
        wind = weather.get("wind", 0.0)
        precip = weather.get("precip", 0.0)
        cloud = weather.get("cloud", 0.0)
        traffic_data = tomtom_traffic_flow(lat_mid, lon_mid)
        speed = 50
        if traffic_data and "flowSegmentData" in traffic_data:
            speed = traffic_data["flowSegmentData"].get("currentSpeed", 50)

        clima_line = (f"**Temperatura**: {temp:.1f}ºC | "
                      f"**Viento**: {wind:.1f} km/h | "
                      f"**Precipitaciones**: {precip:.1f} mm | "
                      f"**Nubosidad**: {cloud:.1f}%")
        st.write(f"Distancia: {dist_km:.2f} km, Tiempo base: {format_minutes(base_minutes)}")
        st.write(clima_line)
        st.write(f"Velocidad de tráfico: {speed} km/h")

        render_map(route_points, st.session_state["current_lat"], st.session_state["current_lon"],
                   d_lat, d_lon, color="blue", map_key="calc_ruta_map")
    else:
        # Solo ubicacion
        route_points = st.session_state.get("route_points", [])
        render_map(route_points, st.session_state["current_lat"], st.session_state["current_lon"],
                   st.session_state.get("dest_lat", None),
                   st.session_state.get("dest_lon", None),
                   color="blue", map_key="calc_ruta_map")

def tab_calcular_ruta_completa():
    st.write("Introduce varias direcciones (Stanza en español, fallback regex). Se usará Matrix Routing de TomTom.")
    texto = st.text_area("Direcciones (una por línea):")
    if st.button("Optimizar Ruta"):
        # Extraer direcciones
        addresses = extract_addresses_stanza(texto)
        if not addresses:
            st.error("No se detectaron direcciones.")
            return
        if len(addresses) < 2:
            st.error("Se necesitan al menos 2 direcciones.")
            return
        # Geocodificar
        coords_list = []
        for addr in addresses:
            res = tomtom_search(addr)
            if not res:
                st.error(f"No se pudo geocodificar: {addr}")
                return
            coords_list.append((res[0][1], res[0][2]))  # (lat, lon)

        # Llamar matrix routing
        dist_matrix, err = tomtom_matrix_routing(coords_list)
        if err:
            st.error(err)
            return
        # TSP local
        best_order, best_cost, err2 = solve_tsp_matrix(dist_matrix)
        if err2:
            st.error(err2)
            return
        st.success(f"Orden óptimo calculado. Distancia ~{best_cost/1000:.2f} km")

        # Reconstruir la ruta final unida
        route_pts, total_dist_m, total_time_s = build_final_route_tomtom(coords_list, best_order)
        if not route_pts:
            st.error("No se pudo obtener la ruta final con TomTom.")
            return
        dist_km = total_dist_m/1000.0
        base_minutes = total_time_s/60.0
        st.info(f"Distancia total: {dist_km:.2f} km | Tiempo total: ~{format_minutes(base_minutes)}")

        st.session_state["route_points"] = route_pts
        st.session_state["origin_lat"] = coords_list[best_order[0]][0]
        st.session_state["origin_lon"] = coords_list[best_order[0]][1]
        st.session_state["dest_lat"] = coords_list[best_order[-1]][0]
        st.session_state["dest_lon"] = coords_list[best_order[-1]][1]
        # Pintar la ruta en el mapa
        # Solo para mostrar, en color "green"
        st.write("Mapa con la ruta optimizada:")
        render_map(route_pts, st.session_state["origin_lat"], st.session_state["origin_lon"],
                   st.session_state["dest_lat"], st.session_state["dest_lon"],
                   color="green", map_key="calc_ruta_completa_map")

def tab_prediccion_retrasos():
    if "route_points" not in st.session_state or not st.session_state["route_points"]:
        st.warning("Primero calcula la ruta en 'Calcular ruta'.")
        return
    dist_km = st.session_state.get("distance_km", 0.0)
    base_time = st.session_state.get("duration_min", 0.0)
    lat_mid = (st.session_state.get("origin_lat", 0.0) + st.session_state.get("dest_lat", 0.0)) / 2
    lon_mid = (st.session_state.get("origin_lon", 0.0) + st.session_state.get("dest_lon", 0.0)) / 2

    st.write(f"Distancia: {dist_km:.2f} km | Tiempo base: {format_minutes(base_time)}")

    weather = get_weather_open_meteo(lat_mid, lon_mid) or {}
    temp = weather.get("temp", 20.0)
    wind = weather.get("wind", 0.0)
    precip = weather.get("precip", 0.0)
    cloud = weather.get("cloud", 0.0)

    clima_line = (f"**Temperatura**: {temp:.1f}ºC | "
                  f"**Viento**: {wind:.1f} km/h | "
                  f"**Precipitaciones**: {precip:.1f} mm | "
                  f"**Nubosidad**: {cloud:.1f}%")
    st.write(clima_line)

    if st.button("Calcular predicción de retrasos"):
        final_time = predict_time(dist_km, temp, wind, precip, cloud)
        if final_time is None:
            st.error("No se pudo calcular la predicción.")
            return
        st.success(f"Tiempo estimado (modelo): {format_minutes(final_time)}")
        delay = final_time - base_time
        if delay > 0:
            st.warning(f"Retraso estimado: +{format_minutes(delay)}")
        else:
            st.info(f"Adelanto estimado: {format_minutes(abs(delay))}")

# -------------------------------------------------------------------------
# TAB 4: Incidencias de Tráfico
# -------------------------------------------------------------------------
def tab_incidencias_trafico():
    """
    Si hay incidentes (por ejemplo, lat_mid speed <30), pintamos una ruta alternativa en rojo.
    """
    if "route_points" not in st.session_state or not st.session_state["route_points"]:
        st.warning("Primero calcula la ruta en 'Calcular ruta'.")
        return

    # Revisar si hay congestión
    lat_mid = (st.session_state["origin_lat"] + st.session_state["dest_lat"]) / 2
    lon_mid = (st.session_state["origin_lon"] + st.session_state["dest_lon"]) / 2
    traffic_data = tomtom_traffic_flow(lat_mid, lon_mid)
    speed = 50
    if traffic_data and "flowSegmentData" in traffic_data:
        speed = traffic_data["flowSegmentData"].get("currentSpeed", 50)

    st.write(f"Velocidad actual en la zona: {speed} km/h")
    if speed < 30:
        st.warning("Tráfico muy congestionado. Calculando ruta alternativa...")

        # Ejemplo: Evitar la carretera "A5" si speed<30
        # Llamamos a tomtom_routing_api con param avoidRoads="A5" (demo)
        # (En la realidad, habría que saber qué carretera está congestionada)
        alt_data = None
        if speed < 30:
            alt_data = tomtom_routing_api(
                st.session_state["origin_lat"], st.session_state["origin_lon"],
                st.session_state["dest_lat"], st.session_state["dest_lon"]
            )
        if not alt_data or "routes" not in alt_data:
            st.error("No se pudo obtener ruta alternativa.")
            # Pintar la original en azul
            render_map(st.session_state["route_points"],
                       st.session_state["origin_lat"], st.session_state["origin_lon"],
                       st.session_state["dest_lat"], st.session_state["dest_lon"],
                       color="blue", map_key="incidencias_map")
        else:
            route = alt_data["routes"][0]
            dist_m = route["summary"]["lengthInMeters"]
            time_s = route["summary"]["travelTimeInSeconds"]
            st.success(f"Ruta alternativa: {dist_m/1000:.2f} km, ~{format_minutes(time_s/60.0)}")

            alt_points = []
            for leg in route["legs"]:
                for point in leg["points"]:
                    alt_points.append((point["latitude"], point["longitude"]))

            # Pintamos la ruta original en azul y la alternativa en rojo
            base_pts = st.session_state["route_points"]
            m_map = folium.Map(location=[lat_mid, lon_mid], zoom_start=12)
            tile_url = f"https://api.tomtom.com/map/1/tile/basic/main/{{z}}/{{x}}/{{y}}.png?key={get_tomtom_key()}"
            folium.TileLayer(tiles=tile_url, attr="TomTom").add_to(m_map)

            # Marcador actual
            folium.Marker((st.session_state["current_lat"], st.session_state["current_lon"]),
                          tooltip="Tu ubicación", icon=folium.Icon(color="blue")).add_to(m_map)
            folium.PolyLine(base_pts, color="blue", weight=5).add_to(m_map)
            folium.PolyLine(alt_points, color="red", weight=5).add_to(m_map)

            st_folium(m_map, width=700, key="incidencias_map")
    else:
        st.info("Tráfico fluido. No se requiere ruta alternativa.")
        # Pintar la ruta original
        render_map(st.session_state["route_points"],
                   st.session_state["origin_lat"], st.session_state["origin_lon"],
                   st.session_state["dest_lat"], st.session_state["dest_lon"],
                   color="blue", map_key="incidencias_map")

# -------------------------------------------------------------------------
# TAB 5: Consumo (según tipo de vehículo)
# -------------------------------------------------------------------------
def tab_consumo():
    st.write("Cálculo de consumo de combustible según tipo de vehículo y condiciones.")
    if "distance_km" not in st.session_state or st.session_state["distance_km"] is None:
        st.warning("Primero calcula la ruta en 'Calcular ruta'.")
        return
    dist_km = st.session_state["distance_km"]

    # Seleccionar vehículo
    veh_type = st.selectbox("Tipo de Vehículo", ["Coche", "Furgoneta", "Camión"])
    # Definimos consumos base (L/km):
    # Coche => ~6 L/100km => 0.06
    # Furgoneta => ~8 L/100km => 0.08
    # Camión => ~25 L/100km => 0.25
    base_map = {"Coche": 0.06, "Furgoneta": 0.08, "Camión": 0.25}
    base_consumo = base_map[veh_type]

    lat_mid = (st.session_state["origin_lat"] + st.session_state["dest_lat"]) / 2
    lon_mid = (st.session_state["origin_lon"] + st.session_state["dest_lon"]) / 2
    weather = get_weather_open_meteo(lat_mid, lon_mid) or {}
    precip = weather.get("precip", 0.0)
    wind = weather.get("wind", 0.0)

    consumo = dist_km * base_consumo
    # Ajustes
    if precip > 2:
        consumo *= 1.1
    if wind > 10:
        consumo *= 1.05

    st.info(f"Distancia: {dist_km:.2f} km")
    st.write(f"Consumo estimado: {consumo:.2f} L")
    # Suponemos 1.70 €/L
    cost = consumo * 1.70
    st.write(f"Coste estimado: {cost:.2f} € (a 1.70 €/L)")

# -------------------------------------------------------------------------
# TAB 6: Calculadora CAE
# -------------------------------------------------------------------------
def tab_calculadora_cae():
    st.write("Calculadora de CAE. Introduce kWh ahorrados.")
    kwh = st.number_input("kWh ahorrados", min_value=0.0, value=500.0, step=50.0)
    if st.button("Calcular CAE"):
        # Ejemplo: CAE generados = kwh
        # Estimamos un rango de ingresos 0.115–0.14 €/kWh
        cost_min = kwh * 0.115
        cost_max = kwh * 0.14
        st.info(f"CAE generados: {kwh:.2f}")
        st.write(f"Ingresos estimados: entre {cost_min:.2f} € y {cost_max:.2f} €")

# -------------------------------------------------------------------------
# APP PRINCIPAL
# -------------------------------------------------------------------------
def main_app():
    st.title("Trafiquea: Dashboard con Stanza (sin refrescar info)")
    st.markdown("""
    **Pestañas**:
    1. **Calcular ruta**: Muestra tu ubicación actual. Al pulsar "Calcular ruta", pinta la ruta en azul.
    2. **Calcular ruta completa**: Usa Matrix Routing (TomTom) + TSP local. Pinta la ruta final en color verde.
    3. **Predicción de retrasos**: Ajusta tiempo base con precipitaciones/viento. No muestra mapa.
    4. **Incidencias de Tráfico**: Si detecta congestión, pinta ruta alternativa en rojo.
    5. **Consumo**: Cálculo de combustible según vehículo y condiciones.
    6. **Calculadora CAE**: Cálculo de CAE generados a partir de kWh ahorrados.
    """)

    tabs = st.tabs([
        "Calcular ruta",
        "Calcular ruta completa",
        "Predicción de retrasos",
        "Incidencias de Tráfico",
        "Consumo",
        "Calculadora CAE"
    ])
    with tabs[0]:
        tab_calcular_ruta()
    with tabs[1]:
        tab_calcular_ruta_completa()
    with tabs[2]:
        tab_prediccion_retrasos()
    with tabs[3]:
        tab_incidencias_trafico()
    with tabs[4]:
        tab_consumo()
    with tabs[5]:
        tab_calculadora_cae()

if __name__ == "__main__":
    for key in ["origin_lat", "origin_lon", "dest_lat", "dest_lon", "distance_km", "duration_min", "route_points"]:
        if key not in st.session_state:
            st.session_state[key] = None
    main_app()
