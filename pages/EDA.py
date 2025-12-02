import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
import database
import pandas as pd

st.set_page_config(page_title="EDA Profesional", page_icon="📊", layout="wide")

st.title("📊 Análisis Exploratorio: La Anatomía del Crimen")
st.markdown("""
Este análisis profundiza en los patrones de seguridad de la CDMX. Buscamos validar hipótesis sobre horarios, 
identificar zonas de alto riesgo y entender la composición detallada de los delitos.
""")

# --- SECCIÓN 1: VALIDACIÓN DE HIPÓTESIS (DÍA VS NOCHE) ---
st.header("1. La Hipótesis: ¿La noche es más peligrosa?")
st.markdown("> *Hipótesis: Los crímenes violentos predominan en la noche, mientras que los patrimoniales ocurren de día.*")

df_metrics = database.get_violence_time_metrics()

if not df_metrics.empty:
    pct_noche = df_metrics.loc[df_metrics['franja_horaria'].str.contains('Noche'), 'porcentaje'].iloc[0]
    pct_dia = df_metrics.loc[df_metrics['franja_horaria'].str.contains('Día'), 'porcentaje'].iloc[0]
    fig_hyp = px.bar(
        df_metrics, 
        x="porcentaje", 
        y="franja_horaria", 
        orientation='h',
        color="franja_horaria",
        text_auto='.1f',
        title="Porcentaje de Violencia por Franja Horaria",
        color_discrete_sequence=["#FF4B4B", "#1F77B4"]
    )
    
    fig_hyp.update_layout(xaxis_title="% del Total de su Tipo", yaxis_title=None, showlegend=False)
    st.plotly_chart(fig_hyp, use_container_width=True)
    st.divider()
    if pct_noche >= 80:
        st.success(f"✅ **HIPÓTESIS CONFIRMADA:** Los datos respaldan tu teoría. El {pct_noche:.1f}% de la violencia ocurre de noche.")
    else:
        st.error(f"❌ **HIPÓTESIS REFUTADA:** Los datos cuentan otra historia.")
        st.markdown(f"""
        **Insight Clave:** Incluso limpiando los datos, la noche **no concentra el 80%** de la violencia. 
        La realidad es que el **{pct_dia:.1f}%** de los crímenes violentos ocurren en el día (07:00 - 19:00).
          
        Esto sugiere que la violencia en la CDMX es un problema estructural de 24 horas, no solo nocturno.
        """)
else:
    st.warning("No hay datos suficientes para la métrica de día/noche.")


# --- SECCIÓN 2: EL RITMO DEL CRIMEN ---
st.divider()
st.header("2. El Ritmo Cardíaco de la Ciudad")
st.markdown("Comparación hora por hora (excluyendo el registro administrativo de las 12:00 PM).")

df_hora_violencia = database.get_violence_heatmap_data()

if not df_hora_violencia.empty:
    # Filtro de limpieza (12pm suele ser error de captura)
    df_hora_clean = df_hora_violencia[df_hora_violencia['hora_hecho'] != 12]

    fig_ritmo = px.line(
        df_hora_clean, 
        x="hora_hecho", 
        y="total", 
        color="violence_type",
        markers=True,
        title="Evolución Horaria: Violencia vs. No Violencia",
        color_discrete_map={"Violent": "red", "Non-Violent": "cyan"}
    )
    fig_ritmo.update_layout(xaxis_title="Hora del Día", yaxis_title="Volumen", hovermode="x unified", template="plotly_dark")
    st.plotly_chart(fig_ritmo, use_container_width=True)
else:
    st.warning("Faltan datos horarios.")


# --- SECCIÓN 3: PATRONES SEMANALES ---
st.divider()
st.header("3. ¿Cuándo ocurren los delitos? (Semana vs Hora)")

df_semana = database.get_day_hour_heatmap()

if not df_semana.empty:
    dias = {0: "Domingo", 1: "Lunes", 2: "Martes", 3: "Miércoles", 4: "Jueves", 5: "Viernes", 6: "Sábado"}
    df_semana['nombre_dia'] = df_semana['dia_semana'].map(dias)
    
    heatmap_data = df_semana.pivot(index='nombre_dia', columns='hora', values='total')
    orden_dias = ["Lunes", "Martes", "Miércoles", "Jueves", "Viernes", "Sábado", "Domingo"]
    heatmap_data = heatmap_data.reindex(orden_dias)
    
    fig_heat = px.imshow(
        heatmap_data,
        labels=dict(x="Hora", y="Día", color="Delitos"),
        x=heatmap_data.columns,
        y=heatmap_data.index,
        color_continuous_scale="Magma"
    )
    st.plotly_chart(fig_heat, use_container_width=True)


# --- SECCIÓN 4: MATRIZ DE RIESGO POR ALCALDÍA ---
st.divider()
st.header("4. Matriz de Riesgo: Volumen vs. Violencia")

df_risk = database.get_alcaldia_risk_metrics()

if not df_risk.empty:
    fig_scatter = px.scatter(
        df_risk,
        x="total_delitos",
        y="porcentaje_violencia",
        size="total_delitos",
        color="porcentaje_violencia",
        text="alcaldia_hecho",
        title="Perfil de Riesgo por Alcaldía",
        color_continuous_scale="RdYlGn_r"
    )
    fig_scatter.update_traces(textposition='top center')
    st.plotly_chart(fig_scatter, use_container_width=True)


# --- 🔥 SECCIÓN 5: TOP COLONIAS PELIGROSAS (NUEVO) 🔥 ---
st.divider()
st.header("5. Zonas Rojas a Nivel Calle (Top Colonias)")
st.markdown("Las alcaldías son muy grandes. Aquí identificamos las **colonias específicas** con mayor concentración de violencia.")

df_colonias = database.get_top_colonias_violent(limit=15)

if not df_colonias.empty:
    fig_col = px.bar(
        df_colonias,
        x="total_violentos",
        y="colonia_hecho",
        orientation='h',
        color="total_violentos",
        title="Top 15 Colonias con más Crímenes Violentos",
        labels={'total_violentos': 'Crímenes Violentos', 'colonia_hecho': 'Colonia'},
        color_continuous_scale="Reds"
    )
    # Invertir eje Y para que el #1 salga arriba
    fig_col.update_yaxes(categoryorder="total ascending")
    st.plotly_chart(fig_col, use_container_width=True)
    
    top_1 = df_colonias.iloc[0]
    st.error(f"🚨 **Foco Rojo Principal:** La colonia **{top_1['colonia_hecho']}** ({top_1['alcaldia_hecho']}) encabeza la lista.")
else:
    st.warning("No se pudieron cargar los datos de colonias.")


# --- 🔥 SECCIÓN 6: COMPOSICIÓN DEL CRIMEN (NUEVO TREEMAP) 🔥 ---
st.divider()
st.header("6. Jerarquía del Crimen (¿Qué delitos ocurren?)")
st.markdown("Desglose detallado de categorías y subtipos de delitos. El tamaño representa la frecuencia.")

df_tree = database.get_crime_treemap_data()

if not df_tree.empty:
    fig_tree = px.treemap(
        df_tree,
        path=[px.Constant("Todos los Delitos"), 'categoria', 'subtipo'],
        values='total',
        color='categoria',
        title="Mapa de Árbol de Incidencia Delictiva",
        color_discrete_sequence=px.colors.qualitative.Pastel
    )
    fig_tree.update_traces(root_color="lightgrey")
    fig_tree.update_layout(margin = dict(t=50, l=25, r=25, b=25))
    st.plotly_chart(fig_tree, use_container_width=True)
    
    st.info("💡 Este gráfico permite ver rápidamente no solo que 'Patrimonio' es lo más común, sino qué tipos específicos de robo predominan.")
else:
    st.warning("No se pudieron cargar los datos para el Treemap.")