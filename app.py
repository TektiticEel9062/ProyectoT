import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import requests
import json
import database # Tu módulo de base de datos

# --- Configuración de la Página ---
st.set_page_config(
    page_title="CDMX Security Overview",
    page_icon="🚨",
    layout="wide",
    initial_sidebar_state="expanded"
)

# 🔥 ¡PEGA TU URL DE N8N (WORKFLOW DE INSIGHTS) AQUÍ! 🔥
N8N_WEBHOOK_URL_INSIGHTS = "https://n8n.tektititc.org/webhook/90408216-1fba-4806-b062-2ab8afb30fea"


# --- Estilos CSS Personalizados ---
st.markdown("""
<style>
    .metric-card {
        background-color: #1E1E1E;
        border: 1px solid #333;
        border-radius: 10px;
        padding: 20px;
        text-align: center;
        margin-bottom: 10px;
    }
    .metric-value {
        font-size: 32px;
        font-weight: bold;
        color: #FF4B4B;
    }
    .metric-label {
        font-size: 14px;
        color: #AAA;
    }
</style>
""", unsafe_allow_html=True)

# --- Funciones de Carga de Datos ---
@st.cache_data
def load_dashboard_data():
    # 1. Tendencia Histórica (para calcular KPIs recientes)
    df_hist = database.get_historical_tendency()
    
    # 2. Distribución por Categoría
    df_cat = database.get_distribution_by_category()
    
    # 3. Top Alcaldías
    df_alcaldias = database.get_top_alcaldias()
    
    # 4. Datos de Treemap (para sacar el Top Delitos específicos)
    df_tree = database.get_crime_treemap_data()
    
    return df_hist, df_cat, df_alcaldias, df_tree

# --- Función de Gemini (CORREGIDA) ---
@st.cache_data(ttl=3600)
def call_gemini_insights(contexto_datos):
    if not N8N_WEBHOOK_URL_INSIGHTS.startswith("https"):
        return "⚠️ URL del Webhook no configurada en app.py"

    payload = {"contexto_datos": contexto_datos}
    try:
        response = requests.post(N8N_WEBHOOK_URL_INSIGHTS, json=payload, timeout=120)
        response.raise_for_status()
        try:
            # Parseo correcto (objeto, no lista)
            return response.json()['content']['parts'][0]['text']
        except Exception:
            return response.text
    except requests.exceptions.RequestException as e:
        return f"Error conectando con IA: {e}"

# --- Carga de Datos ---
df_hist, df_cat, df_alcaldias, df_tree = load_dashboard_data()

if df_hist.empty:
    st.error("No se pudieron cargar los datos. Revisa la conexión a la base de datos.")
    st.stop()

# --- Procesamiento de KPIs (Pandas Magic) ---
# Convertir fecha
df_hist['fecha'] = pd.to_datetime(df_hist['fecha'])

# 1. Totales
total_historico = df_hist['total_delitos'].sum()

# 2. Comparativa Mes Actual vs Mes Anterior
# Asumimos que el último dato es "hoy". Tomamos los últimos 30 días vs los 30 anteriores.
last_30_days = df_hist.tail(30)
prev_30_days = df_hist.iloc[-60:-30]

volumen_actual = last_30_days['total_delitos'].sum()
volumen_anterior = prev_30_days['total_delitos'].sum()
delta_volumen = ((volumen_actual - volumen_anterior) / volumen_anterior) * 100

promedio_diario_actual = last_30_days['total_delitos'].mean()

dias_traduccion = {
    "Monday": "Lunes", "Tuesday": "Martes", "Wednesday": "Miércoles",
    "Thursday": "Jueves", "Friday": "Viernes", "Saturday": "Sábado", "Sunday": "Domingo"
}

# 3. Día más peligroso (Histórico)
df_hist['dia_semana_eng'] = df_hist['fecha'].dt.day_name()
df_hist['dia_semana'] = df_hist['dia_semana_eng'].map(dias_traduccion)

dia_peligroso = df_hist.groupby('dia_semana')['total_delitos'].mean().idxmax()

# --- INTERFAZ DEL DASHBOARD ---

st.title("🚨 Panorama de Seguridad CDMX")
st.markdown(f"**Estado Actual:** Análisis de los últimos 30 días comparado con el periodo anterior.")

# --- BLOQUE 1: KPIS PRINCIPALES ---
col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric("Crímenes (Últimos 30 días)", f"{volumen_actual:,}", delta=f"{delta_volumen:.1f}%", delta_color="inverse")
with col2:
    st.metric("Promedio Diario", f"{promedio_diario_actual:.0f}", delta="Casos por día")
with col3:
    st.metric("Día de Mayor Riesgo", dia_peligroso)
with col4:
    top_delito = df_tree.iloc[0]['subtipo'] if not df_tree.empty else "N/A"
    st.metric("Delito #1", top_delito)

st.divider()

# --- BLOQUE 2: GEMINI INSIGHTS ---
# Preparamos el contexto para la IA
contexto_ia = f"""
- Volumen últimos 30 días: {volumen_actual} delitos.
- Tendencia respecto al mes anterior: {delta_volumen:.1f}%.
- Día de la semana con más delitos: {dia_peligroso}.
- Delito más frecuente: {top_delito}.
- Top 3 Alcaldías con más crimen: {', '.join(df_alcaldias['alcaldia_hecho'].head(3).tolist())}.
"""

col_ia_1, col_ia_2 = st.columns([1, 3])
with col_ia_1:
    st.subheader("🤖 Análisis Inteligente")
    st.markdown("Gemini analiza los KPIs actuales para darte un resumen ejecutivo.")
    if st.button("🔄 Actualizar Análisis"):
        call_gemini_insights.clear() # Borrar caché para forzar nueva llamada
        st.rerun()

with col_ia_2:
    with st.spinner("Generando reporte ejecutivo..."):
        insight_text = call_gemini_insights(contexto_ia)
        st.info(insight_text)

st.divider()

# --- BLOQUE 3: GRÁFICAS PRINCIPALES ---

col_g1, col_g2 = st.columns([2, 1])

with col_g1:
    st.subheader("Tendencia Reciente (Últimos 6 Meses)")
    # Filtramos los últimos 180 días para que la gráfica sea legible y relevante
    df_recent = df_hist.tail(180)
    
    fig_trend = px.area(
        df_recent, x='fecha', y='total_delitos',
        title="Volumen Diario de Incidentes",
        color_discrete_sequence=['#FF4B4B']
    )
    fig_trend.update_layout(
        template="plotly_dark",
        xaxis_title=None,
        yaxis_title="Delitos",
        height=350,
        margin=dict(l=20, r=20, t=40, b=20)
    )
    st.plotly_chart(fig_trend, use_container_width=True)

with col_g2:
    st.subheader("Top 5 Delitos Específicos")
    # Usamos df_tree que tiene el detalle (subtipo)
    if not df_tree.empty:
        top_specific = df_tree.head(5).sort_values(by='total', ascending=True) # Ascendente para barh
        
        fig_bar = px.bar(
            top_specific, x='total', y='subtipo',
            orientation='h',
            text='total',
            color='total',
            color_continuous_scale='Reds',
            title="Delitos más frecuentes"
        )
        fig_bar.update_layout(
            template="plotly_dark",
            xaxis_title=None,
            yaxis_title=None,
            showlegend=False,
            height=350,
            margin=dict(l=0, r=0, t=40, b=0)
        )
        st.plotly_chart(fig_bar, use_container_width=True)
    else:
        st.warning("No hay datos de delitos específicos.")

# --- BLOQUE 4: CONTEXTO GEOGRÁFICO ---
st.subheader("Distribución por Alcaldía (Contexto Clave)")
if not df_alcaldias.empty:
    # Calculamos el porcentaje del total para dar contexto
    total_global = df_alcaldias['total'].sum()
    df_alcaldias['porcentaje'] = (df_alcaldias['total'] / total_global) * 100
    
    # Mostramos como métricas visuales (Progress Bars custom)
    # Dividimos en columnas de 4 en 4
    for i in range(0, min(8, len(df_alcaldias)), 4):
        cols = st.columns(4)
        for j, col in enumerate(cols):
            if i + j < len(df_alcaldias):
                row = df_alcaldias.iloc[i + j]
                col.metric(row['alcaldia_hecho'], f"{row['total']:,}", f"{row['porcentaje']:.1f}% del total")
                col.progress(min(int(row['porcentaje']), 100) / 100)

else:
    st.warning("No hay datos de alcaldías.")