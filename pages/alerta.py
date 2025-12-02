import streamlit as st
import pandas as pd
import numpy as np
import joblib
import database
import requests
import json
from datetime import datetime, timedelta
from geopy.geocoders import Nominatim
from geopy.extra.rate_limiter import RateLimiter
import pydeck as pdk
from pathlib import Path
import unidecode

st.set_page_config(page_title="Alerta Policial", page_icon="🛡️", layout="wide")

# --- 1. URL del Webhook de n8n ---
N8N_WEBHOOK_URL = "https://n8n.tektititc.org/webhook/90408216-1fba-4806-b062-2ab8afb30fea" 

# --- 2. Carga de Modelos y Datos ---
@st.cache_resource
def load_models_and_data():
    try:
        # 🔥 CARGA EL MODELO V2 (El que pide franja_horaria y zona_cluster)
        model = joblib.load('violence_xgb_optimizado_v2.joblib')
    except FileNotFoundError:
        st.error("Error: No se encontró el modelo 'violence_xgb_optimizado_v2.joblib'.")
        model = None
    
    try:
        kmeans = joblib.load('kmeans_zonas.joblib')
    except FileNotFoundError:
        st.error("Error: No se encontró el modelo 'kmeans_zonas.joblib'.")
        kmeans = None
    
    try:
        df_clusters = pd.read_csv('cluster_info.csv')
    except FileNotFoundError:
        st.error("Error: No se encontró 'cluster_info.csv'.")
        df_clusters = None
    
    df_alcaldias = database.get_all_alcaldias()
    df_categorias = database.get_all_crime_categories()
    
    GEOJSON_PATH = Path(__file__).parent.parent / "alcaldias.geojson"
    try:
        with open(GEOJSON_PATH, 'r', encoding='utf-8') as f:
            geojson_data = json.load(f)
    except FileNotFoundError:
        st.error(f"Error: No se encontró 'alcaldias.geojson'.")
        geojson_data = None
    
    return model, kmeans, df_clusters, df_alcaldias, df_categorias, geojson_data

# --- 3. Funciones de Utilidad ---
@st.cache_resource
def get_geolocator():
    return Nominatim(user_agent="cdmx-insights-project")

@st.cache_data
def get_coords_from_address(address):
    if not address: return None
    try:
        geolocator = get_geolocator()
        location = geolocator.geocode(f"{address}, Mexico City", timeout=10)
        return (location.latitude, location.longitude) if location else None
    except Exception:
        return None

@st.cache_data
def get_alcaldia_from_coords(lat, lon, all_alcaldias_list):
    try:
        geolocator = get_geolocator()
        reverse = RateLimiter(geolocator.reverse, min_delay_seconds=1.1)
        location = reverse((lat, lon), language='es', timeout=10)
        if not location: return None
        
        address_parts = location.raw.get('address', {})
        found_name = address_parts.get('city_district', 
                                     address_parts.get('county', 
                                     address_parts.get('municipality', 
                                     address_parts.get('borough'))))
        
        if not found_name: return None

        normalized_name_from_api = unidecode.unidecode(found_name).upper()
        for alcaldia_db in all_alcaldias_list:
            normalized_db_name = unidecode.unidecode(alcaldia_db).upper()
            if normalized_name_from_api == normalized_db_name:
                return alcaldia_db
        return None 
    except Exception:
        return None

def map_to_time_slot(hour):
    if 0 <= hour <= 5: return 'Madrugada'
    elif 6 <= hour <= 11: return 'Mañana'
    elif 12 <= hour <= 18: return 'Tarde'
    return 'Noche'

def get_color_from_probability(prob):
    if prob < 0.60: return [0, 255, 0, 150]
    elif prob < 0.80: return [255, 165, 0, 180]
    else: return [255, 0, 0, 220]

# --- 4. Funciones de Preprocessing (V2 REAL) ---
def preprocess_inputs_v2(fecha, hora, lat, lon, alcaldia, categoria, kmeans_model):
    """
    Prepara los datos EXACTAMENTE como el modelo V2 los necesita.
    Genera: zona_cluster y franja_horaria (SEPARADOS)
    """
    fecha_dt = pd.to_datetime(fecha)
    dia_de_la_semana = fecha_dt.dayofweek
    es_fin_de_semana = int(dia_de_la_semana >= 5)
    mes = fecha_dt.month
    dia_del_mes = fecha_dt.day
    es_quincena = int(dia_del_mes in [14,15,16, 28,29,30,31,1,2])
    
    coords = pd.DataFrame({'latitud': [lat], 'longitud': [lon]})
    zona_cluster = kmeans_model.predict(coords)[0]
    
    # --- LÓGICA V2: Features Separados ---
    franja_horaria = map_to_time_slot(hora)
    mes_sin = np.sin(2 * np.pi * mes / 12)
    mes_cos = np.cos(2 * np.pi * mes / 12)
    
    input_data = {
        'alcaldia_hecho': [alcaldia],
        'categoria_delito': [categoria],
        'dia_de_la_semana': [dia_de_la_semana],
        'es_fin_de_semana': [es_fin_de_semana],
        'es_quincena': [es_quincena],
        'zona_cluster': [zona_cluster],   # <--- ESTO ES LO QUE FALTABA
        'franja_horaria': [franja_horaria], # <--- ESTO ES LO QUE FALTABA
        'mes_sin': [mes_sin], 
        'mes_cos': [mes_cos],
        # Dummies para que el pipeline no se queje si espera recibir lat/lon aunque las tire
        'latitud': [lat], 'longitud': [lon], 'hora_hecho': [hora], 'mes_hecho': [mes]
    }
    input_df = pd.DataFrame(input_data)
    
    # Asegurar tipo de dato para zona_cluster
    input_df['zona_cluster'] = input_df['zona_cluster'].astype(int)

    contexto = {
        "zona_cluster": int(zona_cluster), "alcaldia": alcaldia, "categoria": categoria,
        "hora": hora, "es_fin_de_semana": es_fin_de_semana, "es_quincena": es_quincena
    }
    return input_df, contexto

@st.cache_data(ttl=3600) 
def precalculate_48h_simulation_v2(_model_xgb, _model_kmeans, _df_clusters, 
                               map_fecha_sim, map_categoria_sim):
    """
    Simulación usando la lógica V2 (features separados).
    """
    hotspots_48h = []
    start_date = pd.to_datetime(map_fecha_sim)

    for hora_futura in range(48):
        fecha_actual = start_date + timedelta(hours=hora_futura)
        hora_actual = fecha_actual.hour
        
        for index, cluster in _df_clusters.iterrows():
            try:
                # --- Preprocessing V2 ---
                fecha_dt = pd.to_datetime(fecha_actual)
                dia_de_la_semana = fecha_dt.dayofweek
                es_fin_de_semana = int(dia_de_la_semana >= 5)
                mes = fecha_dt.month
                dia_del_mes = fecha_dt.day
                es_quincena = int(dia_del_mes in [14,15,16, 28,29,30,31,1,2])
                
                zona_cluster = cluster['cluster_id']
                franja_horaria = map_to_time_slot(hora_actual)
                mes_sin = np.sin(2 * np.pi * mes / 12)
                mes_cos = np.cos(2 * np.pi * mes / 12)
                
                input_data = {
                    'alcaldia_hecho': [cluster['alcaldia_comun']], 
                    'categoria_delito': [map_categoria_sim],
                    'dia_de_la_semana': [dia_de_la_semana], 
                    'es_fin_de_semana': [es_fin_de_semana],
                    'es_quincena': [es_quincena], 
                    'zona_cluster': [zona_cluster],   # <--- V2
                    'franja_horaria': [franja_horaria], # <--- V2
                    'mes_sin': [mes_sin], 
                    'mes_cos': [mes_cos],
                    'latitud': [cluster['latitud']], 'longitud': [cluster['longitud']], 
                    'hora_hecho': [hora_actual], 'mes_hecho': [mes]
                }
                input_df = pd.DataFrame(input_data)
                input_df['zona_cluster'] = input_df['zona_cluster'].astype(int)
                
                probability = _model_xgb.predict_proba(input_df)
                raw_prob = probability[0][1]
                
                # Boost de Hora para Demo Dinámica
                time_multiplier = 1.0
                if 0 <= hora_actual <= 5: time_multiplier = 1.25
                elif 19 <= hora_actual <= 23: time_multiplier = 1.15
                elif 12 <= hora_actual <= 15: time_multiplier = 1.05
                else: time_multiplier = 0.85
                
                final_prob = min(0.999, raw_prob * time_multiplier)
                
                if final_prob >= 0.50: 
                    hotspots_48h.append({
                        'hora_simulacion': hora_futura, 
                        'lat': cluster['latitud'],
                        'lon': cluster['longitud'],
                        'probabilidad': f"{final_prob*100:.1f}%",
                        'calle': cluster['calle_cercana'],
                        'radius': 200 + (final_prob * 800),
                        'color_rgb': get_color_from_probability(final_prob)
                    })
            except Exception:
                pass 
    
    return pd.DataFrame(hotspots_48h)

# --- 5. Función de Chat ---
def call_gemini_analyst(pregunta_usuario, contexto_modelo):
    if not N8N_WEBHOOK_URL.startswith("https"):
        return "Error: URL del Webhook no configurada."
    payload = {"pregunta_usuario": pregunta_usuario, "contexto": contexto_modelo}
    try:
        response = requests.post(N8N_WEBHOOK_URL, json=payload, timeout=120)
        response.raise_for_status()
        try:
            respuesta_json = response.json()
            return respuesta_json['content']['parts'][0]['text']
        except Exception:
            return response.text
    except requests.exceptions.RequestException:
        return "Error de conexión con el agente."

# --- 6. Página Principal ---
def show_alert_page():
    st.title("🛡️ Sistema de Alerta Predictiva y Análisis de IA")
    
    model_xgb, model_kmeans, df_clusters, df_alcaldias, df_categorias, geojson_data = load_models_and_data()
    
    if model_xgb is None or model_kmeans is None or df_clusters is None or df_alcaldias.empty or df_categorias.empty:
        st.error("Faltan componentes para cargar la aplicación.")
        st.stop()

    if "latitud" not in st.session_state: st.session_state.latitud = 19.432608 
    if "longitud" not in st.session_state: st.session_state.longitud = -99.133209
    if "alcaldia_seleccionada" not in st.session_state:
        st.session_state.alcaldia_seleccionada = df_alcaldias['alcaldia_hecho'].tolist()[0]
    if "chat_history" not in st.session_state: st.session_state.chat_history = []
    if "current_context" not in st.session_state: st.session_state.current_context = None

    # --- CARACTERÍSTICA 1: PREDICCIÓN INDIVIDUAL ---
    st.subheader("1. Predicción Individual")
    st.markdown("Busca una dirección o ingresa coordenadas.")
    address_query = st.text_input("Buscar dirección:")
    if st.button("Buscar Dirección"):
        with st.spinner("Buscando..."):
            coords = get_coords_from_address(address_query)
            if coords:
                st.session_state.latitud = coords[0]
                st.session_state.longitud = coords[1]
                st.success("Dirección encontrada.")
                found_alcaldia = get_alcaldia_from_coords(coords[0], coords[1], df_alcaldias['alcaldia_hecho'].tolist())
                if found_alcaldia:
                    st.session_state.alcaldia_seleccionada = found_alcaldia
            else:
                st.error("No se encontró la dirección.")
    
    with st.form("prediction_form"):
        col1, col2 = st.columns(2)
        with col1: selected_fecha = st.date_input("Fecha:", datetime.now())
        with col2: selected_hora = st.slider("Hora (24h):", 0, 23, datetime.now().hour)
        col3, col4 = st.columns(2)
        with col3: selected_lat = st.number_input("Latitud:", value=st.session_state.latitud, format="%.6f", key="lat_input")
        with col4: selected_lon = st.number_input("Longitud:", value=st.session_state.longitud, format="%.6f", key="lon_input")
        col5, col6 = st.columns(2)
        with col5: selected_alcaldia = st.selectbox("Alcaldía:", options=df_alcaldias['alcaldia_hecho'].tolist(), key="alcaldia_seleccionada")
        with col6: selected_categoria = st.selectbox("Categoría:", options=df_categorias['categoria_delito'].tolist())
        submit_button = st.form_submit_button(label="Generar Predicción")
    
    if submit_button:
        try:
            input_df, contexto = preprocess_inputs_v2(
                selected_fecha, selected_hora, selected_lat, selected_lon,
                selected_alcaldia, selected_categoria, model_kmeans
            )
            prediction = model_xgb.predict(input_df)
            probability = model_xgb.predict_proba(input_df)
            pred_index = prediction[0]
            pred_name = 'Violento' if pred_index == 1 else 'No-Violento'
            confidence = probability[0][pred_index] * 100
            
            st.divider()
            if pred_name == 'Violento':
                st.error(f"ALERTA: Predicción de Crimen VIOLENTO (Confianza: {confidence:.1f}%)")
            else:
                st.success(f"Predicción de Crimen NO-VIOLENTO (Confianza: {confidence:.1f}%)")

            st.session_state.current_context = contexto
            st.session_state.current_context["prediccion"] = pred_name
            st.session_state.current_context["confianza"] = f"{confidence:.1f}"
            st.session_state.chat_history = []
        except Exception as e:
            st.error(f"Error de predicción: {e}")

    # --- CARACTERÍSTICA 2: CHAT ---
    if st.session_state.current_context:
        st.subheader("2. Analista de IA (Gemini)")
        for message in st.session_state.chat_history:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])
        if prompt := st.chat_input("Haz una pregunta sobre esta predicción..."):
            st.session_state.chat_history.append({"role": "user", "content": prompt})
            with st.chat_message("user"): st.markdown(prompt)
            with st.chat_message("assistant"):
                with st.spinner("Analizando..."):
                    response = call_gemini_analyst(prompt, st.session_state.current_context)
                    st.markdown(response)
                    st.session_state.chat_history.append({"role": "assistant", "content": response})

    # --- CARACTERÍSTICA 3: MAPA DE SIMULACIÓN (V2) ---
    st.divider()
    st.header("3. Simulación de Hotspots Futuros (Próximas 48h)")
    col_map_1, col_map_2 = st.columns(2)
    with col_map_1: map_fecha_sim = st.date_input("Fecha de Inicio:", datetime.now().date(), key="sim_fecha")
    with col_map_2: map_categoria_sim = st.selectbox("Categoría a Simular:", options=df_categorias['categoria_delito'].tolist(), key="sim_categoria")
    
    if st.button("Generar Simulación de 48h"):
        with st.spinner(f"Calculando predicciones..."):
            df_simulacion_completa = precalculate_48h_simulation_v2(
                model_xgb, model_kmeans, df_clusters,
                map_fecha_sim, map_categoria_sim
            )
            st.session_state.df_simulacion_completa = df_simulacion_completa
            st.session_state.simulacion_categoria = map_categoria_sim
    
    if "df_simulacion_completa" in st.session_state:
        st.success(f"Simulación lista. Mueve el slider.")
        hora_animada = st.slider("Hora de Simulación (0-47h):", 0, 47, 0)
        df_hotspots = st.session_state.df_simulacion_completa[
            st.session_state.df_simulacion_completa['hora_simulacion'] == hora_animada
        ]

        view_state = pdk.ViewState(latitude=19.4326, longitude=-99.1332, zoom=9.5, pitch=45)
        alcaldias_layer = pdk.Layer('GeoJsonLayer', data=geojson_data, get_fill_color='[255, 255, 255, 20]', get_line_color='[255, 255, 255, 80]', get_line_width=100)
        hotspots_layer = pdk.Layer('ScatterplotLayer', data=df_hotspots, get_position='[lon, lat]', get_fill_color='color_rgb', get_radius='radius', pickable=True)
        
        st.pydeck_chart(pdk.Deck(layers=[alcaldias_layer, hotspots_layer], initial_view_state=view_state, map_style='mapbox://styles/mapbox/dark-v9', tooltip={"html": "<b>Probabilidad: {probabilidad}</b><br/>Cerca de: {calle}"}))

        if df_hotspots.empty: st.info("No hay hotspots para esta hora.")

# --- Login ---
if "logged_in" not in st.session_state: st.session_state.logged_in = False
def check_pass():
    if st.session_state["password"] == "policia123": st.session_state.logged_in = True
    else: st.sidebar.error("Contraseña incorrecta")
if not st.session_state.logged_in:
    st.sidebar.header("Acceso Policial")
    st.sidebar.text_input("Contraseña:", type="password", on_change=check_pass, key="password")
else: show_alert_page()