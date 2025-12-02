import streamlit as st
import pandas as pd
import numpy as np
import joblib
import database  # Módulo de base de datos
import pydeck as pdk
import json
import time
from datetime import datetime, timedelta
from pathlib import Path

st.set_page_config(page_title="Mapa Interactivo", page_icon="🗺️", layout="wide")
st.title("🗺️ Mapa Interactivo de Incidencia Delictiva")

# --- 1. Carga de Modelos y Datos (V2 ESTABLE) ---
@st.cache_resource
def load_models_and_data():
    try:
        model = joblib.load('violence_xgb_optimizado_v2.joblib')
    except FileNotFoundError:
        st.error("Error: Modelo v2 no encontrado.")
        model = None
    try:
        kmeans = joblib.load('kmeans_zonas.joblib')
    except FileNotFoundError:
        st.error("Error: Modelo KMeans no encontrado.")
        kmeans = None
    try:
        df_clusters = pd.read_csv('cluster_info.csv')
    except FileNotFoundError:
        st.error("Error: CSV de clusters no encontrado.")
        df_clusters = None
    
    GEOJSON_PATH = Path(__file__).parent.parent / "alcaldias.geojson"
    try:
        with open(GEOJSON_PATH, 'r', encoding='utf-8') as f:
            geojson_data = json.load(f)
    except FileNotFoundError:
        st.error("Error: GeoJSON no encontrado.")
        geojson_data = None
            
    df_alcaldias = database.get_all_alcaldias()
    df_categorias = database.get_all_crime_categories()
    
    return model, kmeans, df_clusters, df_alcaldias, df_categorias, geojson_data

# --- 2. Funciones Helpers y Colores ---
def map_to_time_slot(hour):
    if 0 <= hour <= 5: return 'Madrugada'
    elif 6 <= hour <= 11: return 'Mañana'
    elif 12 <= hour <= 18: return 'Tarde'
    return 'Noche'

def get_color_from_probability(prob):
    if prob < 0.60: return [0, 255, 0, 180]
    elif prob < 0.80: return [255, 165, 0, 200]
    else: return [255, 0, 0, 220]

def get_color_by_violence(violence_type):
    if violence_type == 'Violent': return [255, 0, 0, 180]   # Rojo
    return [0, 191, 255, 150]     # Azul Claro

def get_color_by_category(categoria):
    c = str(categoria)
    if 'Patrimony' in c: return [255, 165, 0, 180]
    if 'Family' in c: return [128, 0, 128, 180]
    if 'Sexual' in c: return [255, 105, 180, 180]
    if 'Life' in c: return [255, 0, 0, 180]
    if 'Freedom' in c: return [0, 0, 255, 180]
    if 'Society' in c: return [0, 255, 255, 180]
    return [128, 128, 128, 140]

# --- 3. Pre-cálculo de Simulación (Lógica V2 Correcta) ---
@st.cache_data(ttl=3600) 
def precalculate_48h_simulation_v2(_model_xgb, _model_kmeans, _df_clusters, 
                                   map_fecha_sim, map_categoria_sim, map_alcaldia_sim):
    hotspots_48h = []
    start_date = pd.to_datetime(map_fecha_sim)
    
    if map_alcaldia_sim:
        clusters_filtrados = _df_clusters[_df_clusters['alcaldia_comun'].str.upper() == map_alcaldia_sim.upper()]
    else:
        clusters_filtrados = _df_clusters

    if clusters_filtrados.empty: return pd.DataFrame()

    for hora_futura in range(48):
        fecha_actual = start_date + timedelta(hours=hora_futura)
        hora_actual = fecha_actual.hour
        
        for index, cluster in clusters_filtrados.iterrows():
            try:
                # --- Preprocessing V2 (Features Separados) ---
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
                    'zona_cluster': [zona_cluster],     
                    'franja_horaria': [franja_horaria], 
                    'mes_sin': [mes_sin], 'mes_cos': [mes_cos],
                    'latitud': [cluster['latitud']], 'longitud': [cluster['longitud']], 
                    'hora_hecho': [hora_actual], 'mes_hecho': [mes]
                }
                input_df = pd.DataFrame(input_data)
                input_df['zona_cluster'] = input_df['zona_cluster'].astype(int)
                
                probability = _model_xgb.predict_proba(input_df)
                raw_prob = probability[0][1]
                
                # Boost de Hora
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
            except Exception: pass 
    return pd.DataFrame(hotspots_48h)

# Carga Inicial
model_xgb, model_kmeans, df_clusters, df_alcaldias, df_categorias, geojson_data = load_models_and_data()
if model_xgb is None or geojson_data is None: st.stop()

# Vista Compartida
view_state = pdk.ViewState(latitude=19.4326, longitude=-99.1332, zoom=10, pitch=45)

# Capa Alcaldías (Fondo)
alcaldias_layer = pdk.Layer(
    'GeoJsonLayer',
    data=geojson_data,
    get_fill_color='[0, 0, 0, 0]', 
    get_line_color='[100, 100, 100, 255]',
    get_line_width=50,
    pickable=True,
    auto_highlight=True,
    tooltip={"html": "<b>Alcaldía:</b> {nomgeo}", "style": {"backgroundColor": "steelblue", "color": "white"}}
)

# =================================================================
# --- MAPA 1: Histórico Filtrado (Sin Animación - Estático) ---
# =================================================================
st.header("1. Mapa Histórico de Incidencia")
st.sidebar.header("Filtros Mapa 1")
crime_type = st.sidebar.multiselect("Tipo crimen:", options=df_categorias['categoria_delito'].tolist(), default=df_categorias['categoria_delito'].iloc[0:2].tolist() if not df_categorias.empty else [])
hour_slider = st.sidebar.slider("Hora:", 0, 23, (0, 23))
crime_class = st.sidebar.radio("Clase:", ('Violent', 'Non-Violent', 'Ambos'), index=2)

df_mapa_1 = database.get_filtered_map_data(crime_type, hour_slider, crime_class)
heatmap_layer_1 = pdk.Layer('HeatmapLayer', data=df_mapa_1, get_position='[longitud, latitud]', opacity=0.8, pickable=False, get_weight=1)
st.pydeck_chart(pdk.Deck(layers=[heatmap_layer_1, alcaldias_layer], initial_view_state=view_state, map_style='mapbox://styles/mapbox/dark-v9'))
st.info(f"Mostrando {len(df_mapa_1)} puntos.")


# =================================================================
# --- MAPA 2: Histórico Animado (Mes a Mes) ---
# =================================================================
st.divider()
st.header("2. Explorador Histórico por Fecha")
st.markdown("Selecciona Año y usa el **Slider** o el botón **Animar**.")

col1, col2, col3 = st.columns([1, 1, 2])
with col1:
    lista_anios = database.run_query("SELECT DISTINCT anio_hecho FROM crimes ORDER BY anio_hecho DESC")['anio_hecho'].tolist() or [2024]
    selected_anio = st.selectbox("Año:", options=lista_anios, key="map2_anio")
with col2:
    start_anim_2 = st.button("▶️ Animar Año", key="anim_btn_2")
with col3:
    color_mode = st.radio("Color:", ["Violencia (Rojo/Azul)", "Categoría (Multi)"], key="color_mode")

# Slider Manual
meses_dict = {1:"Ene", 2:"Feb", 3:"Mar", 4:"Abr", 5:"May", 6:"Jun", 7:"Jul", 8:"Ago", 9:"Sep", 10:"Oct", 11:"Nov", 12:"Dic"}
selected_mes_nombre = st.select_slider("Mes:", options=list(meses_dict.values()), key="map2_mes_slider")
inv_meses = {v: k for k, v in meses_dict.items()}
selected_mes_num = inv_meses[selected_mes_nombre]

# Contenedor del Mapa 2
map2_placeholder = st.empty()
info2_placeholder = st.empty()

def render_map2(anio, mes_num):
    df_map = database.get_map_data_by_date(anio, mes_num)
    if not df_map.empty:
        if "Violencia" in color_mode:
            df_map['color'] = df_map['violence_type'].apply(get_color_by_violence)
        else:
            df_map['color'] = df_map['categoria_delito'].apply(get_color_by_category)

        points_layer = pdk.Layer(
            'ScatterplotLayer',
            data=df_map,
            get_position='[longitud, latitud]',
            get_fill_color='color',     
            get_radius=30,             
            radiusMinPixels=3,
            radiusMaxPixels=15,
            pickable=True
        )
        tooltip = {"html": "<b>{categoria_delito}</b><br/>{delito}<br/><i>{violence_type}</i>", "style": {"backgroundColor": "black", "color": "white"}}
        
        deck = pdk.Deck(layers=[alcaldias_layer, points_layer], initial_view_state=view_state, map_style='mapbox://styles/mapbox/dark-v9', tooltip=tooltip)
        map2_placeholder.pydeck_chart(deck)
        info2_placeholder.info(f"📅 {meses_dict[mes_num]} {anio}: {len(df_map)} delitos.")
    else:
        map2_placeholder.pydeck_chart(pdk.Deck(layers=[alcaldias_layer], initial_view_state=view_state, map_style='mapbox://styles/mapbox/dark-v9'))
        info2_placeholder.warning(f"Sin datos para {meses_dict[mes_num]}.")

# Lógica: Animación vs Manual
if start_anim_2:
    for m in range(1, 13):
        render_map2(selected_anio, m)
        time.sleep(1.0) # Pausa de 1 segundo
else:
    render_map2(selected_anio, selected_mes_num)


# =================================================================
# --- MAPA 3: Simulación 48h (Animado) ---
# =================================================================
st.divider()
st.header("3. Simulación de Hotspots (Próximas 48h)")

col_map_1, col_map_2 = st.columns(2)
with col_map_1: map_fecha_sim = st.date_input("Fecha Inicio:", datetime.now().date(), key="sim_fecha")
with col_map_2: map_alcaldia_sim = st.selectbox("Alcaldía (Opcional):", options=["Todas"] + df_alcaldias['alcaldia_hecho'].tolist(), key="sim_alcaldia")
map_categoria_sim = st.selectbox("Categoría:", options=df_categorias['categoria_delito'].tolist(), key="sim_categoria")

if st.button("🔄 Generar Datos de Simulación"):
    alcaldia_filtro = None if map_alcaldia_sim == "Todas" else map_alcaldia_sim
    with st.spinner(f"Calculando..."):
        df_sim = precalculate_48h_simulation_v2(model_xgb, model_kmeans, df_clusters, map_fecha_sim, map_categoria_sim, alcaldia_filtro)
        st.session_state.df_sim_mapa = df_sim

if "df_sim_mapa" in st.session_state:
    st.success("Datos listos. Usa el slider o el botón para animar.")
    
    col_anim_1, col_anim_2 = st.columns([1, 3])
    with col_anim_1:
        start_anim_3 = st.button("▶️ Animar 48h", key="anim_btn_3")
    with col_anim_2:
        hora_manual = st.slider("Hora Manual (0-47h):", 0, 47, 0, key="slider_48h")
    
    map3_placeholder = st.empty()
    info3_placeholder = st.empty()

    def render_map3(hora):
        df_hotspots = st.session_state.df_sim_mapa[st.session_state.df_sim_mapa['hora_simulacion'] == hora]
        hotspots_layer = pdk.Layer(
            'ScatterplotLayer', data=df_hotspots, get_position='[lon, lat]', 
            get_fill_color='color_rgb', get_radius='radius', radiusMinPixels=5, pickable=True
        )
        deck = pdk.Deck(layers=[alcaldias_layer, hotspots_layer], initial_view_state=view_state, map_style='mapbox://styles/mapbox/dark-v9', tooltip={"html": "<b>Prob: {probabilidad}</b><br/>{calle}", "style": {"backgroundColor": "black", "color": "white"}})
        map3_placeholder.pydeck_chart(deck)
        if df_hotspots.empty:
            info3_placeholder.info(f"Hora +{hora}h: Sin hotspots críticos.")
        else:
            info3_placeholder.success(f"Hora +{hora}h: {len(df_hotspots)} hotspots.")

    # Lógica: Animación vs Manual
    if start_anim_3:
        for h in range(48):
            render_map3(h)
            time.sleep(1.0) # Pausa de 1 segundo
    else:
        render_map3(hora_manual)

# =================================================================
# --- MAPA 4: Explorador a 6 Meses ---
# =================================================================
st.divider()
st.header("4. Explorador de Riesgo a Futuro (6 Meses)")
col_fut_1, col_fut_2, col_fut_3 = st.columns(3)
with col_fut_1:
    min_date = datetime.now().date()
    max_date = min_date + timedelta(days=180)
    fecha_futura = st.date_input("Fecha Futura:", min_value=min_date, max_value=max_date, value=min_date, key="fut_fecha")
with col_fut_2: hora_futura = st.slider("Hora:", 0, 23, 20, key="fut_hora")
with col_fut_3: categoria_futura = st.selectbox("Categoría:", options=df_categorias['categoria_delito'].tolist(), key="fut_cat")

if st.button("Proyectar Riesgo"):
    with st.spinner("Analizando..."):
        hotspots_futuros = []
        fecha_dt = pd.to_datetime(fecha_futura)
        dia_semana = fecha_dt.dayofweek
        es_finde = int(dia_semana >= 5)
        mes = fecha_dt.month
        dia_mes = fecha_dt.day
        es_quincena = int(dia_mes in [14,15,16, 28,29,30,31,1,2])
        franja_horaria = map_to_time_slot(hora_futura)
        mes_sin = np.sin(2 * np.pi * mes / 12)
        mes_cos = np.cos(2 * np.pi * mes / 12)

        for index, cluster in df_clusters.iterrows():
            try:
                zona_cluster = cluster['cluster_id']
                input_data = {
                    'alcaldia_hecho': [cluster['alcaldia_comun']], 'categoria_delito': [categoria_futura],
                    'dia_de_la_semana': [dia_semana], 'es_fin_de_semana': [es_finde], 'es_quincena': [es_quincena], 
                    'zona_cluster': [zona_cluster], 'franja_horaria': [franja_horaria],
                    'mes_sin': [mes_sin], 'mes_cos': [mes_cos],
                    'latitud': [cluster['latitud']], 'longitud': [cluster['longitud']], 
                    'hora_hecho': [hora_futura], 'mes_hecho': [mes]
                }
                input_df = pd.DataFrame(input_data)
                input_df['zona_cluster'] = input_df['zona_cluster'].astype(int)
                
                probability = model_xgb.predict_proba(input_df)
                prob_val = probability[0][1]
                
                if prob_val >= 0.50:
                    hotspots_futuros.append({
                        'lat': cluster['latitud'], 'lon': cluster['longitud'],
                        'probabilidad': f"{prob_val*100:.1f}%",
                        'calle': cluster['calle_cercana'], 'alcaldia': cluster['alcaldia_comun'],
                        'radius': 200 + (prob_val * 800),
                        'color_rgb': get_color_from_probability(prob_val)
                    })
            except Exception: pass
        
        df_futuro = pd.DataFrame(hotspots_futuros)
        layer_futuro = pdk.Layer('ScatterplotLayer', data=df_futuro, get_position='[lon, lat]', get_fill_color='color_rgb', get_radius='radius', pickable=True)
        st.pydeck_chart(pdk.Deck(layers=[alcaldias_layer, layer_futuro], initial_view_state=view_state, map_style='mapbox://styles/mapbox/dark-v9', tooltip={"html": "<b>{probabilidad}</b><br/>{calle}", "style": {"backgroundColor": "black", "color": "white"}}))
        
        if not df_futuro.empty:
            st.success(f"Se detectaron {len(df_futuro)} zonas de riesgo.")
            st.dataframe(df_futuro[['alcaldia', 'calle', 'probabilidad']].sort_values(by='probabilidad', ascending=False).reset_index(drop=True))
        else: st.info("No se detectaron zonas de riesgo.")