import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
import database
import pandas as pd
from statsmodels.tsa.statespace.sarimax import SARIMAXResults

st.set_page_config(page_title="Tendencias y Pronóstico", page_icon="📈", layout="wide")

# --- 1. Carga del Modelo (Con el fix de caché) ---
@st.cache_resource
def load_forecast_model():
    try:
        return SARIMAXResults.load('crime_forecaster.pkl')
    except: return None

@st.cache_data
def get_forecast(_model, steps=150):
    if _model:
        try:
            forecast = _model.get_forecast(steps=steps)
            return forecast.summary_frame(alpha=0.05)
        except: pass
    return pd.DataFrame()

st.title("📈 Análisis de Tendencias: El Futuro de la Seguridad")
st.markdown("""
Esta sección conecta el comportamiento histórico con proyecciones matemáticas para anticipar la incidencia delictiva.
""")

# --- NOTAS METODOLÓGICAS (CRÍTICAS) ---
with st.expander("ℹ️ Notas Metodológicas y Calidad de Datos", expanded=False):
    st.markdown("""
    **1. El 'Artefacto' de las 12:00 PM:**
    Se han excluido los registros marcados a las 12:00 PM exactas. Análisis previos mostraron que este horario se usa administrativamente cuando se desconoce la hora real del delito, creando falsos picos.
    
    **2. El Modelo SARIMA y los 'Ciclos':**
    Notarás que la línea de pronóstico roja tiene un patrón de "dientes" o zig-zag. **Esto es correcto.** El modelo ha aprendido la **estacionalidad semanal**: el crimen sube naturalmente los fines de semana y baja al inicio de la semana. El modelo proyecta que este comportamiento humano continuará.
    """)

# --- Carga de Datos ---
df_tendencia = database.get_historical_tendency() # Ya viene limpio de la BD
model_sarima = load_forecast_model()

# ==============================================================================
# SECCIÓN 1: EL PRONÓSTICO (SARIMA)
# ==============================================================================
st.divider()
st.header("1. Proyección a 5 Meses (Modelo SARIMA)")

if not df_tendencia.empty and model_sarima:
    # Generar pronóstico
    df_forecast = get_forecast(model_sarima, steps=150)
    
    if not df_forecast.empty:
        # --- CÁLCULO DE VELOCIDAD (INSIGHT AUTOMÁTICO) ---
        # Comparamos el promedio de los últimos 30 días vs el promedio de los próximos 30
        last_30_mean = df_tendencia.tail(30)['total_delitos'].mean()
        next_30_mean = df_forecast.head(30)['mean'].mean()
        delta_pct = ((next_30_mean - last_30_mean) / last_30_mean) * 100
        
        col_kpi1, col_kpi2, col_kpi3 = st.columns(3)
        col_kpi1.metric("Promedio Diario (Histórico Reciente)", f"{last_30_mean:.0f} delitos")
        col_kpi2.metric("Promedio Diario (Proyectado)", f"{next_30_mean:.0f} delitos", delta=f"{delta_pct:.1f}%", delta_color="inverse")
        col_kpi3.info("El modelo predice una **estabilidad** con ligera tendencia a la baja en el corto plazo, manteniendo los ciclos semanales.")

        # --- GRÁFICO PRINCIPAL ---
        fig = go.Figure()
        
        # 1. Historia (Últimos 6 meses para claridad)
        df_recent = df_tendencia.tail(180)
        fig.add_trace(go.Scatter(
            x=df_recent['fecha'], y=df_recent['total_delitos'], 
            mode='lines', name='Historia Real', line=dict(color='#1F77B4', width=2)
        ))
        
        # 2. Pronóstico
        fig.add_trace(go.Scatter(
            x=df_forecast.index, y=df_forecast['mean'], 
            mode='lines', name='Pronóstico IA', line=dict(dash='dot', color='#FF4B4B', width=2)
        ))
        
        # 3. Intervalo de Confianza (La "Nube" de incertidumbre)
        fig.add_trace(go.Scatter(
            x=df_forecast.index, y=df_forecast['mean_ci_upper'],
            mode='lines', line=dict(width=0), showlegend=False, hoverinfo='skip'
        ))
        fig.add_trace(go.Scatter(
            x=df_forecast.index, y=df_forecast['mean_ci_lower'],
            mode='lines', line=dict(width=0), fill='tonexty', 
            fillcolor='rgba(255, 75, 75, 0.15)', showlegend=False, hoverinfo='skip', name="Margen de Error (95%)"
        ))
        
        fig.update_layout(
            title="Tendencia Histórica y Proyección Futura",
            xaxis_title="Fecha", yaxis_title="Delitos Diarios",
            template="plotly_dark", height=450, hovermode="x unified"
        )
        st.plotly_chart(fig, use_container_width=True)
        
    else:
        st.error("El modelo no pudo generar el pronóstico.")
else:
    st.warning("No se pudieron cargar los datos históricos o el modelo.")


# ==============================================================================
# SECCIÓN 2: ANATOMÍA TEMPORAL (PATRONES)
# ==============================================================================
st.divider()
st.header("2. Patrones de Comportamiento (¿Cuándo ocurre?)")

col1, col2 = st.columns(2)

with col1:
    st.subheader("Ciclo Diario (Hora x Hora)")
    # Usamos la función que trae datos limpios (sin las 12pm)
    df_hora = database.get_violence_heatmap_data()
    
    if not df_hora.empty:
        # Limpieza visual explícita (aunque la BD ya lo haga, para asegurar gráfica perfecta)
        df_hora = df_hora[df_hora['hora_hecho'] != 12]
        
        fig_line = px.line(
            df_hora, x="hora_hecho", y="total", color="violence_type",
            title="Actividad Delictiva por Hora",
            color_discrete_map={"Violent": "#FF4B4B", "Non-Violent": "#1F77B4"},
            labels={"hora_hecho": "Hora (24h)", "total": "Volumen", "violence_type": "Tipo"}
        )
        fig_line.update_layout(template="plotly_dark", xaxis=dict(tickmode='linear', dtick=4))
        st.plotly_chart(fig_line, use_container_width=True)
        st.caption("Se observa el 'valle' de actividad en la madrugada (3-5 AM) y el repunte sostenido desde la mañana.")

with col2:
    st.subheader("Ciclo Semanal (Mapa de Calor)")
    df_semana = database.get_day_hour_heatmap()
    
    if not df_semana.empty:
        # Mapeo para lectura fácil
        dias = {0: "Dom", 1: "Lun", 2: "Mar", 3: "Mié", 4: "Jue", 5: "Vie", 6: "Sáb"}
        df_semana['dia'] = df_semana['dia_semana'].map(dias)
        
        heatmap_data = df_semana.pivot(index='dia', columns='hora', values='total')
        orden = ["Lun", "Mar", "Mié", "Jue", "Vie", "Sáb", "Dom"]
        heatmap_data = heatmap_data.reindex(orden)
        
        fig_heat = px.imshow(
            heatmap_data,
            labels=dict(x="Hora", y="Día", color="Intensidad"),
            color_continuous_scale="Magma",
            title="Concentración de Delitos (Semana vs Hora)"
        )
        fig_heat.update_layout(template="plotly_dark", xaxis=dict(tickmode='linear', dtick=4))
        st.plotly_chart(fig_heat, use_container_width=True)
        st.caption("Las zonas más brillantes indican los momentos de mayor riesgo en la semana.")

# ==============================================================================
# SECCIÓN 3: COMPOSICIÓN (QUÉ OCURRE)
# ==============================================================================
st.divider()
st.header("3. Composición del Delito")

df_tree = database.get_crime_treemap_data()

if not df_tree.empty:
    fig_tree = px.treemap(
        df_tree, 
        path=[px.Constant("Total CDMX"), 'categoria', 'subtipo'], 
        values='total',
        color='categoria', 
        title="Mapa Jerárquico de Delitos (Tamaño = Frecuencia)",
        color_discrete_sequence=px.colors.qualitative.Bold
    )
    fig_tree.update_traces(root_color="#262730")
    fig_tree.update_layout(template="plotly_dark", margin=dict(t=30, l=10, r=10, b=10))
    st.plotly_chart(fig_tree, use_container_width=True)