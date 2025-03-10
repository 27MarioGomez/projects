import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.regularizers import l2
import requests
import os, shutil
from sklearn.preprocessing import MinMaxScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from joblib import Parallel, delayed
from numba import njit
from textblob import TextBlob
from dateutil.parser import parse as date_parse
from newsapi import NewsApiClient
from prophet import Prophet
from ta.momentum import RSIIndicator
from ta.trend import MACD, SMAIndicator, EMAIndicator
from ta.volatility import BollingerBands, AverageTrueRange
from ta.volume import OnBalanceVolumeIndicator
from transformers.pipelines import pipeline
from xgboost import XGBRegressor
from geopy.geocoders import Nominatim

# -----------------------------------------------------------------------------
# Configuración de la página
# -----------------------------------------------------------------------------
st.set_page_config(page_title="Trafiquea: Optimización y Predicción de Rutas", layout="wide")

# -----------------------------------------------------------------------------
# Función de geocodificación (Nominatim, OpenStreetMap)
# -----------------------------------------------------------------------------
@st.cache_data(ttl=3600)
def geocode_address(address):
    geolocator = Nominatim(user_agent="trafiquea_dashboard")
    location = geolocator.geocode(address)
    if location:
        return (location.latitude, location.longitude)
    return None

# -----------------------------------------------------------------------------
# Función para obtener direcciones y optimizar ruta con Mapbox Directions
# Requiere un token de Mapbox en los Secrets
# -----------------------------------------------------------------------------
def get_directions(origin_coords, destination_coords):
    token = st.secrets.get("mapbox_token")
    if not token:
        st.error("Token de Mapbox no configurado en los Secrets.")
        return None
    url = f"https://api.mapbox.com/directions/v5/mapbox/driving/{origin_coords[1]},{origin_coords[0]};{destination_coords[1]},{destination_coords[0]}"
    params = {"access_token": token, "geometries": "geojson", "overview": "simplified"}
    response = requests.get(url, params=params)
    data = response.json()
    if "routes" in data and len(data["routes"]) > 0:
        return data["routes"][0]["geometry"]
    return None

# -----------------------------------------------------------------------------
# Función para obtener datos reales de clima desde OpenWeatherMap
# Requiere API key en los Secrets (clave: openweather_api_key)
# -----------------------------------------------------------------------------
@st.cache_data(ttl=600)
def get_weather_data(city):
    api_key = st.secrets.get("openweather_api_key")
    if not api_key:
        st.error("API key de OpenWeatherMap no configurada en los Secrets.")
        return None
    url = "http://api.openweathermap.org/data/2.5/weather"
    params = {"q": city, "appid": api_key, "units": "metric", "lang": "es"}
    response = requests.get(url, params=params)
    data = response.json()
    if data and data.get("main"):
        return {"temperature": data["main"]["temp"], "description": data["weather"][0]["description"]}
    return None

# -----------------------------------------------------------------------------
# Función para obtener noticias reales sobre movilidad usando NewsAPI
# Requiere API key en los Secrets (clave: newsapi_key)
# -----------------------------------------------------------------------------
@st.cache_data(ttl=43200)
def get_newsapi_articles(query="movilidad", show_warning=True):
    newsapi_key = st.secrets.get("newsapi_key", "")
    if not newsapi_key:
        st.error("Clave 'newsapi_key' no encontrada en los Secrets.")
        return []
    try:
        newsapi = NewsApiClient(api_key=newsapi_key)
        data = newsapi.get_everything(q=query, language="es", sort_by="relevancy", page_size=5)
        articles = []
        if data.get("articles"):
            for art in data["articles"]:
                image_url = art.get("urlToImage", "")
                title = art.get("title") or "Sin título"
                description = art.get("description") or "Sin descripción"
                pub_date = art.get("publishedAt") or "Fecha no disponible"
                link = art.get("url") or "#"
                try:
                    parsed_date = date_parse(pub_date)
                except:
                    parsed_date = datetime(1970, 1, 1)
                pub_date_str = parsed_date.strftime("%Y-%m-%d %H:%M:%S")
                articles.append({
                    "title": title,
                    "description": description,
                    "pubDate": pub_date_str,
                    "link": link,
                    "image": image_url,
                    "parsed_date": parsed_date
                })
            articles = sorted(articles, key=lambda x: x["parsed_date"], reverse=True)
        return articles
    except Exception as e:
        if ("rateLimited" in str(e) or "429" in str(e)) and show_warning:
            st.warning("Se ha excedido el límite de peticiones a NewsAPI. Vuelve en 12 horas.")
        elif show_warning:
            st.error(f"Error al obtener noticias: {e}")
        return []

# -----------------------------------------------------------------------------
# Función para obtener datos reales de demanda de transporte
# Se espera que exista una URL pública en los Secrets (clave: transit_demand_url)
# -----------------------------------------------------------------------------
@st.cache_data(ttl=3600)
def get_transit_demand_data():
    url = st.secrets.get("transit_demand_url", None)
    if url:
        try:
            df = pd.read_csv(url, parse_dates=["Fecha"])
            return df
        except Exception as e:
            st.error(f"Error al cargar datos de demanda: {e}")
            return pd.DataFrame()
    else:
        st.warning("No se ha configurado la URL de datos de demanda en los Secrets.")
        return pd.DataFrame()

# -----------------------------------------------------------------------------
# Transformadores y Pipeline para preprocesar datos de demanda
# En este ejemplo se asume que la variable de interés es la "Demanda"
# -----------------------------------------------------------------------------
class DataFrameTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, transformer):
        self.transformer = transformer
    def fit(self, X, y=None):
        self.transformer.fit(X, y)
        return self
    def transform(self, X):
        X_trans = self.transformer.transform(X)
        if isinstance(X, pd.DataFrame):
            return pd.DataFrame(X_trans, columns=X.columns, index=X.index)
        return X_trans

class FeatureSelector(BaseEstimator, TransformerMixin):
    def __init__(self, feature_cols, target_col, enet_threshold=0.01, importance_threshold=0.01):
        self.feature_cols = feature_cols
        self.target_col = target_col
        self.enet_threshold = enet_threshold
        self.importance_threshold = importance_threshold
        self.selected_features_ = None
    def fit(self, X, y=None):
        df = X.copy()
        if y is not None:
            df[self.target_col] = y
        y_arr = df[self.target_col].values
        X_arr = df[self.feature_cols].values
        from sklearn.linear_model import ElasticNetCV
        enet = ElasticNetCV(cv=5, random_state=42).fit(X_arr, y_arr)
        coefs = enet.coef_
        initial_selected = [self.feature_cols[i] for i in range(len(self.feature_cols)) if abs(coefs[i]) > self.enet_threshold]
        if not initial_selected:
            initial_selected = self.feature_cols
        from xgboost import XGBRegressor
        xgb = XGBRegressor(n_estimators=50, max_depth=3, learning_rate=0.1, random_state=42)
        xgb.fit(df[initial_selected].values, y_arr)
        importances = xgb.feature_importances_
        refined = [initial_selected[i] for i in range(len(initial_selected)) if importances[i] > self.importance_threshold]
        self.selected_features_ = refined if refined else initial_selected
        return self
    def transform(self, X):
        return X[self.selected_features_]

@st.cache_resource(show_spinner=False)
def load_sentiment_pipeline():
    return pipeline("sentiment-analysis", model="distilbert/distilbert-base-uncased-finetuned-sst-2-english", revision="714eb0f")

def get_advanced_sentiment(text):
    pipe = load_sentiment_pipeline()
    result = pipe(text)[0]
    return 50 + (result["score"] * 50) if result["label"].upper() == "POSITIVE" else 50 - (result["score"] * 50)

@st.cache_data(ttl=43200)
def get_news_sentiment():
    articles = get_newsapi_articles(query="movilidad", show_warning=False)
    if not articles:
        return 50.0
    sentiments_tb = []
    sentiments_trans = []
    for article in articles:
        text = (article["title"] or "") + " " + (article["description"] or "")
        blob = TextBlob(text)
        polarity_tb = blob.sentiment.polarity
        sentiments_tb.append(50 + (polarity_tb * 50))
        sentiments_trans.append(get_advanced_sentiment(text))
    return (np.mean(sentiments_tb) + np.mean(sentiments_trans)) / 2.0

# -----------------------------------------------------------------------------
# Creación de secuencias para el modelo LSTM (usando Numba)
# -----------------------------------------------------------------------------
@njit
def create_sequences_numba(data, window_size):
    n = data.shape[0]
    num_features = data.shape[1]
    m = n - window_size
    X = np.empty((m, window_size, num_features), dtype=data.dtype)
    y = np.empty(m, dtype=data.dtype)
    for i in range(m):
        X[i] = data[i:i+window_size]
        y[i] = data[i+window_size, 0]
    return X, y

def create_sequences(data, window_size):
    if data.shape[0] <= window_size:
        return None, None
    return create_sequences_numba(data, window_size)

def flatten_sequences(X_seq):
    return X_seq.reshape((X_seq.shape[0], X_seq.shape[1] * X_seq.shape[2]))

# -----------------------------------------------------------------------------
# Modelo LSTM para predicción de demanda
# -----------------------------------------------------------------------------
def build_demand_lstm_model(input_shape):
    model = Sequential([
        LSTM(64, return_sequences=True, input_shape=input_shape, kernel_regularizer=l2(0.001)),
        Dropout(0.2),
        LSTM(32, kernel_regularizer=l2(0.001)),
        Dropout(0.2),
        Dense(50, activation="relu", kernel_regularizer=l2(0.001)),
        Dense(1)
    ])
    model.compile(optimizer=Adam(0.001), loss="mse")
    return model

def train_demand_model(df_demand, window_size=7, epochs=10, batch_size=16):
    df_demand = df_demand.sort_values("Fecha")
    df_demand["log_demand"] = np.log1p(df_demand["Demanda"])
    scaler = MinMaxScaler()
    demand_scaled = scaler.fit_transform(df_demand[["log_demand"]])
    X, y = create_sequences(demand_scaled, window_size)
    if X is None:
        st.error("No hay suficientes datos para crear secuencias de demanda.")
        return None, scaler
    model = build_demand_lstm_model((window_size, 1))
    es = EarlyStopping(monitor="loss", patience=3, restore_best_weights=True)
    model.fit(X, y, epochs=epochs, batch_size=batch_size, verbose=0, callbacks=[es])
    return model, scaler

def predict_future_demand(model, scaler, df_demand, window_size, forecast_days):
    demand_scaled = scaler.transform(df_demand[["log_demand"]])
    X, _ = create_sequences(demand_scaled, window_size)
    if X is None:
        return None
    last_window = X[-1:]
    preds = []
    current_input = last_window.copy()
    for _ in range(forecast_days):
        pred_scaled = model.predict(current_input, verbose=0)[0][0]
        # Invertir la escala
        inv = scaler.inverse_transform(np.array([[pred_scaled]]))
        pred = np.expm1(inv[0, 0])
        preds.append(pred)
        new_seq = np.array([[pred_scaled]])
        current_input = np.concatenate([current_input[:, 1:, :], new_seq.reshape(1, 1, 1)], axis=1)
    future_dates = pd.date_range(start=df_demand["Fecha"].max() + timedelta(days=1), periods=forecast_days).tolist()
    return future_dates, preds

# -----------------------------------------------------------------------------
# Función para obtener datos reales de demanda de transporte
# Se requiere que se configure la URL de datos de demanda en los Secrets (clave: transit_demand_url)
# -----------------------------------------------------------------------------
@st.cache_data(ttl=3600)
def get_transit_demand_data():
    url = st.secrets.get("transit_demand_url", None)
    if url:
        try:
            df = pd.read_csv(url, parse_dates=["Fecha"])
            return df
        except Exception as e:
            st.error(f"Error al cargar datos de demanda: {e}")
            return pd.DataFrame()
    else:
        st.warning("No se ha configurado la URL de datos de demanda en los Secrets.")
        return pd.DataFrame()

# -----------------------------------------------------------------------------
# Diseño del Dashboard
# -----------------------------------------------------------------------------
def main_app():
    st.title("Trafiquea: Optimización y Predicción de Rutas")
    st.markdown("""
    **Descripción del Proyecto:**  
    Este dashboard está diseñado para optimizar rutas y predecir la demanda de transporte en tiempo real, ayudando a ayuntamientos, empresas de movilidad y ciudadanos a tomar decisiones informadas.  
    La herramienta integra:
      - **Geolocalización y Optimización de Rutas:** Ingresando un origen y destino, se obtiene una ruta optimizada usando Mapbox.
      - **Datos en Tiempo Real:** Información sobre clima (OpenWeatherMap) y, de ser posible, datos de tráfico.
      - **Predicción de Demanda de Transporte:** Un modelo LSTM entrenado con datos reales de demanda (obtenidos de una fuente pública configurada) para predecir la demanda futura.
      - **Dashboard de Métricas de Movilidad y Análisis de Impacto:** Visualización de métricas clave (tiempos de viaje, ahorro en combustible, reducción de CO₂, etc.).
      - **Noticias sobre Movilidad:** Se muestran noticias reales sobre movilidad y transporte, con un diseño moderno y funcional.
    **NFA:** Not Financial Advice.
    """)
    
    st.sidebar.title("Configuración")
    st.sidebar.subheader("Optimización de Rutas")
    origin = st.sidebar.text_input("Dirección de Origen", "Plaza Mayor, Madrid")
    destination = st.sidebar.text_input("Dirección de Destino", "Puerta del Sol, Madrid")
    if st.sidebar.button("Obtener Ruta"):
        origin_coords = geocode_address(origin)
        destination_coords = geocode_address(destination)
        if origin_coords and destination_coords:
            route_geo = get_directions(origin_coords, destination_coords)
            weather = get_weather_data("Madrid")
            st.success("Ruta obtenida y datos en tiempo real actualizados.")
            st.write(f"Clima: {weather['temperature']}°C, {weather['description']}")
            fig_map = go.Figure(go.Scattermapbox(
                lat=[origin_coords[0], destination_coords[0]],
                lon=[origin_coords[1], destination_coords[1]],
                mode="markers+lines",
                marker=go.scattermapbox.Marker(size=14),
                text=["Origen", "Destino"]
            ))
            fig_map.update_layout(
                mapbox_style="light",
                mapbox_accesstoken=st.secrets.get("mapbox_token"),
                mapbox_zoom=12,
                mapbox_center={"lat": (origin_coords[0]+destination_coords[0])/2, "lon": (origin_coords[1]+destination_coords[1])/2}
            )
            st.plotly_chart(fig_map, use_container_width=True)
        else:
            st.error("No se pudieron geolocalizar las direcciones.")
    
    st.sidebar.markdown("---")
    st.sidebar.subheader("Predicción de Demanda")
    training_period = st.sidebar.select_slider(
        "Histórico de Entrenamiento (años):",
        options=[1, 2, 3],
        value=1,
        help="A mayor período, mayor precisión (aunque aumente el tiempo de entrenamiento)."
    )
    forecast_days = st.sidebar.slider("Días a predecir:", 1, 30, 5, help="Número de días futuros a predecir.")
    
    # Si se dispone de una URL de demanda real, se usa esa; de lo contrario se muestra un aviso.
    df_demand = get_transit_demand_data()
    if df_demand.empty:
        st.error("No se encontraron datos reales de demanda. Configure 'transit_demand_url' en los Secrets.")
    else:
        st.subheader("Datos Históricos de Demanda de Transporte")
        st.line_chart(df_demand.set_index("Fecha")["Demanda"])
    
    tabs = st.tabs([
        "Predicción de Demanda",
        "Dashboard de Métricas",
        "Análisis de Impacto",
        "Noticias Recientes"
    ])
    
    # Pestaña 1: Predicción de Demanda
    with tabs[0]:
        st.header("Predicción de Demanda de Transporte")
        if not df_demand.empty:
            model_demand, scaler_demand = train_demand_model(df_demand, window_size=7, epochs=10, batch_size=16)
            if model_demand:
                future_dates, demand_preds = predict_future_demand(model_demand, scaler_demand, df_demand, window_size=7, forecast_days=forecast_days)
                if future_dates and demand_preds:
                    df_future = pd.DataFrame({"Fecha": future_dates, "Demanda Predicha": demand_preds})
                    st.subheader("Demanda Futura Estimada")
                    st.line_chart(df_future.set_index("Fecha")["Demanda Predicha"])
                    st.download_button(
                        label="Descargar Predicción en CSV",
                        data=df_future.to_csv(index=False).encode("utf-8"),
                        file_name="prediccion_demanda.csv",
                        mime="text/csv"
                    )
    
    # Pestaña 2: Dashboard de Métricas
    with tabs[1]:
        st.header("Métricas de Movilidad en Tiempo Real")
        # Aquí se podrían integrar datos reales de tráfico y clima, si se dispone de ellos
        weather = get_weather_data("Madrid")
        traffic_data = st.secrets.get("traffic_data", "Datos de tráfico no disponibles")
        st.metric("Temperatura Actual (°C)", f"{weather['temperature']}°C")
        st.metric("Condición Climática", weather['description'])
        st.metric("Nivel de Congestión", traffic_data)
    
    # Pestaña 3: Análisis de Impacto y Sostenibilidad
    with tabs[2]:
        st.header("Impacto Ambiental y Sostenibilidad")
        # Si se disponen de datos reales (por ejemplo, de emisiones), se integrarían aquí.
        # Se puede usar datos abiertos de emisiones o cálculos basados en rutas.
        st.write("Aquí se mostrarían análisis de ahorro de CO₂ y coste por kW/h ahorrado.")
    
    # Pestaña 4: Noticias Recientes sobre Movilidad
    with tabs[3]:
        st.header("Noticias Recientes sobre Movilidad")
        articles = get_newsapi_articles(query="movilidad", show_warning=True)
        if articles:
            st.markdown(
                """
                <style>
                .news-container {
                    display: flex;
                    flex-direction: row;
                    gap: 1rem;
                    overflow-x: auto;
                    padding: 1rem 0;
                }
                .news-item {
                    flex: 0 0 auto;
                    width: 280px;
                    min-height: 380px;
                    background-color: #2c2c3e;
                    padding: 0.5rem;
                    border-radius: 5px;
                    border: 1px solid #4a4a6a;
                    display: flex;
                    flex-direction: column;
                    justify-content: space-between;
                }
                .news-item img {
                    width: 100%;
                    height: 160px;
                    object-fit: cover;
                    border-radius: 5px;
                    margin-bottom: 0.5rem;
                }
                .news-item h4 {
                    margin: 0 0 0.3rem 0;
                    font-size: 1rem;
                }
                .news-item p {
                    font-size: 0.8rem;
                    margin: 0 0 0.3rem 0;
                }
                .read-more-btn {
                    display: block;
                    margin: 0.5rem auto;
                    padding: 0.4rem 0.8rem;
                    background-color: #fff;
                    color: #000;
                    text-decoration: none;
                    border-radius: 3px;
                    text-align: center;
                    font-size: 0.8rem;
                    font-weight: 600;
                }
                .read-more-btn:hover {
                    background-color: #e6e6e6;
                    color: #000;
                }
                </style>
                """,
                unsafe_allow_html=True
            )
            st.markdown("<div class='news-container'>", unsafe_allow_html=True)
            for article in articles:
                image_tag = f"<img src='{article['image']}' alt='Imagen' />" if article['image'] else ""
                link_button = f"<a href='{article['link']}' target='_blank' class='read-more-btn'>Leer más</a>"
                st.markdown(
                    f"""
                    <div class='news-item'>
                        {image_tag}
                        <div>
                            <h4>{article['title']}</h4>
                            <p><em>{article['pubDate']}</em></p>
                            <p>{article['description']}</p>
                        </div>
                        <div>
                            {link_button}
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True
                )
            st.markdown("</div>", unsafe_allow_html=True)
        else:
            st.warning("No se pudieron obtener noticias en este momento.")

if __name__ == "__main__":
    main_app()

