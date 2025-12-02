import streamlit as st
import duckdb
import pandas as pd

# Nombre del archivo de la base de datos
DB_FILE = "cdmx_insights.db"

@st.cache_resource
def get_db_connection():
    """
    Crea y cachea una conexión a la base de datos DuckDB.
    """
    try:
        con = duckdb.connect(DB_FILE, read_only=True)
        return con
    except duckdb.Error as e:
        st.error(f"Error al conectar con la base de datos: {e}")
        return None

@st.cache_data
def run_query(query, params=None):
    """
    Ejecuta una consulta y cachea el resultado.
    """
    con = get_db_connection()
    if con:
        try:
            if params:
                df = con.execute(query, params).fetchdf()
            else:
                df = con.execute(query).fetchdf()
            return df
        except duckdb.Error as e:
            st.error(f"Error al ejecutar la consulta: {e}")
            return pd.DataFrame()
    return pd.DataFrame()

# --- Funciones para Dashboard (app.py) ---

def get_all_alcaldias():
    """
    Obtiene una lista única de todas las 'alcaldia_hecho' para el filtro.
    """
    query = """
    SELECT DISTINCT 
        alcaldia_hecho 
    FROM 
        crimes 
    WHERE 
        alcaldia_hecho IS NOT NULL
    ORDER BY 
        alcaldia_hecho
    """
    return run_query(query)

def get_crime_stats():
    # ...
    query = """
    SELECT
        COUNT(*) AS total_delitos,
        COUNT(*) / COUNT(DISTINCT CAST(fecha_hecho AS DATE)) AS promedio_diario
    FROM
        crimes
    WHERE HOUR(hora_hecho) 
    """
    return run_query(query)

def get_top_alcaldias():
    """
    Obtiene las 5 alcaldías con más delitos.
    """
    query = """
    SELECT
        alcaldia_hecho,
        COUNT(*) AS total
    FROM
        crimes
    WHERE
        alcaldia_hecho IS NOT NULL
        AND HOUR(hora_hecho) != 12
    GROUP BY
        alcaldia_hecho
    ORDER BY
        total DESC
    LIMIT 5
    """
    return run_query(query)

# --- Funciones para Analysis.py ---

def get_historical_tendency():
    """
    Consulta la tendencia histórica de delitos por fecha.
    """
    query = """
    SELECT 
        CAST(fecha_hecho AS DATE) AS fecha,
        COUNT(*) AS total_delitos
    FROM 
        crimes
    WHERE HOUR(hora_hecho) != 12
    GROUP BY 
        fecha
    ORDER BY 
        fecha ASC
    """
    return run_query(query)

def get_distribution_by_category():
    """
    Obtiene la cuenta de delitos por 'categoria_delito'.
    """
    query = """
    SELECT
        categoria_delito,
        COUNT(*) AS total
    FROM
        crimes
    WHERE HOUR(hora_hecho) != 12
    GROUP BY
        categoria_delito
    ORDER BY
        total DESC
    """
    return run_query(query)

def get_distribution_by_hour():
    """
    Obtiene la cuenta de delitos por 'hora_hecho'.
    """
    query = """
    SELECT
        hora_hecho,
        COUNT(*) AS total
    FROM
        crimes
    WHERE HOUR(hora_hecho) != 12
    GROUP BY
        hora_hecho
    ORDER BY
        hora_hecho ASC
    """
    return run_query(query)

def get_violence_heatmap_data():
    """
    Obtiene la cuenta de delitos por hora y tipo de violencia.
    """
    query = """
    SELECT
        hora_hecho,
        violence_type,
        COUNT(*) AS total
    FROM
        crimes
    WHERE
        violence_type IS NOT NULL
        AND HOUR(hora_hecho) != 12
    GROUP BY
        hora_hecho, violence_type
    ORDER BY
        hora_hecho
    """
    return run_query(query)

def get_violence_time_metrics():
    """
    Calcula los porcentajes de crímenes violentos de noche vs día.
    (Mockup: 40% vs 60%)
    """
    query = """
    WITH time_classified AS (
        SELECT
            violence_type,
            CASE 
                WHEN HOUR(hora_hecho) >= 19 OR HOUR(hora_hecho) <= 7 THEN 'Noche (19-07)'
                ELSE 'Día (07-19)'
            END AS franja_horaria
        FROM
            crimes
        WHERE
            violence_type = 'Violent'
            AND HOUR(hora_hecho) != 12
    )
    SELECT
        franja_horaria,
        COUNT(*) AS total,
        (COUNT(*) * 100.0 / SUM(COUNT(*)) OVER ()) AS porcentaje
    FROM
        time_classified
    GROUP BY
        franja_horaria
    """
    return run_query(query)


# --- Funciones para Mapa.py ---

def get_all_crime_categories():
    """
    Obtiene una lista única de todas las 'categoria_delito' para el filtro.
    """
    query = """
    SELECT DISTINCT 
        categoria_delito 
    FROM 
        crimes 
    WHERE 
        categoria_delito IS NOT NULL
    ORDER BY 
        categoria_delito
    """
    return run_query(query)

def get_filtered_map_data(crime_types, hour_range, classification):
    """
    Obtiene datos filtrados de latitud y longitud para el mapa.
    """
    query = """
    SELECT 
        latitud, 
        longitud 
    FROM 
        crimes 
    WHERE 
        latitud IS NOT NULL AND longitud IS NOT NULL
    """
    
    params = []

    # 1. Filtro por tipo de crimen (multiselect)
    if crime_types:
        placeholders = ", ".join(["?"] * len(crime_types))
        query += f" AND categoria_delito IN ({placeholders})"
        params.extend(crime_types)
        
    # 2. Filtro por hora (slider)
    query += " AND HOUR(hora_hecho) BETWEEN ? AND ?"
    params.append(hour_range[0])
    params.append(hour_range[1])

    # 3. Filtro por clasificación (radio)
    if classification != 'Ambos':
        query += " AND violence_type = ?"  # Asumiendo 'violence_type' para 'Violent'/'Non-Violent'
        params.append(classification)
    
    # Para optimizar, limitamos a 50,000 puntos en el mapa
    query += " LIMIT 50000"

    return run_query(query, params)

# --- Funciones para EDA.py ---

def get_category_violence_heatmap():
    """
    Obtiene la cuenta de delitos por categoría y tipo de violencia.
    (Para el heatmap en EDA.py)
    """
    query = """
    SELECT
        crime_classification,
        violence_type,
        COUNT(*) AS total
    FROM
        crimes
    WHERE
        violence_type IS NOT NULL AND crime_classification IS NOT NULL
    GROUP BY
        crime_classification, violence_type
    """
    return run_query(query)

def get_yearly_violence_trend():
    """
    Obtiene la tendencia anual de crímenes violentos vs no violentos.
    (Para el gráfico de línea en EDA.py)
    """
    query = """
    SELECT
        anio_hecho,
        violence_type,
        COUNT(*) AS total
    FROM
        crimes
    WHERE
        violence_type IS NOT NULL
    GROUP BY
        anio_hecho, violence_type
    ORDER BY
        anio_hecho ASC
    """
    return run_query(query)

def get_alcaldia_distribution(limit=10):
    """
    Obtiene el top 10 de alcaldías por número de crímenes.
    (Para el gráfico de barras en EDA.py)
    """
    query = f"""
    SELECT
        alcaldia_hecho,
        COUNT(*) AS total
    FROM
        crimes
    WHERE
        alcaldia_hecho IS NOT NULL
    GROUP BY
        alcaldia_hecho
    ORDER BY
        total DESC
    LIMIT {limit}
    """
    return run_query(query)

@st.cache_data
def get_map_data_by_date(anio, mes):
    query = """
    SELECT 
        latitud, 
        longitud,
        violence_type,     -- Para color Rojo/Azul
        categoria_delito,  -- Para color por Categoría
        delito             -- Para el tooltip
    FROM 
        crimes 
    WHERE 
        anio_hecho = ? AND mes_hecho = ? 
        AND latitud IS NOT NULL AND longitud IS NOT NULL
    LIMIT 50000
    """
    params = [anio, mes]
    return run_query(query, params)


def get_day_hour_heatmap():
    """
    Obtiene la cuenta de delitos agrupada por Día de la Semana y Hora.
    """
    # DAYOFWEEK devuelve 0=Dom, 1=Lun... en DuckDB
    query = """
    SELECT
        DAYOFWEEK(fecha_hecho) as dia_semana,
        HOUR(hora_hecho) as hora,
        COUNT(*) as total
    FROM
        crimes
    WHERE
        fecha_hecho IS NOT NULL AND hora_hecho IS NOT NULL
        AND HOUR(hora_hecho) != 12
    GROUP BY
        dia_semana, hora
    ORDER BY
        dia_semana, hora
    """
    return run_query(query)

def get_alcaldia_risk_metrics():
    """
    Obtiene total de delitos y % de violencia por Alcaldía.
    """
    query = """
    SELECT
        alcaldia_hecho,
        COUNT(*) as total_delitos,
        SUM(CASE WHEN violence_type = 'Violent' THEN 1 ELSE 0 END) * 100.0 / COUNT(*) as porcentaje_violencia
    FROM
        crimes
    WHERE
        alcaldia_hecho IS NOT NULL
        AND HOUR(hora_hecho) != 12
    GROUP BY
        alcaldia_hecho
    HAVING
        COUNT(*) > 100
    """
    return run_query(query)

# --- AÑADE ESTO AL FINAL DE database.py ---

def get_top_colonias_violent(limit=15):
    """
    Obtiene las colonias con mayor número absoluto de crímenes VIOLENTOS.
    """
    query = """
    SELECT
        colonia_hecho,
        alcaldia_hecho,
        COUNT(*) as total_violentos
    FROM
        crimes
    WHERE
        violence_type = 'Violent' 
        AND colonia_hecho IS NOT NULL 
        AND colonia_hecho != 'NA'
        AND HOUR(hora_hecho) != 12
    GROUP BY
        colonia_hecho, alcaldia_hecho
    ORDER BY
        total_violentos DESC
    LIMIT ?
    """
    return run_query(query, [limit])

def get_crime_treemap_data():
    query = """
    SELECT
        crime_classification as categoria,
        delito as subtipo,
        COUNT(*) as total
    FROM
        crimes
    WHERE
        crime_classification IS NOT NULL AND delito IS NOT NULL
        AND HOUR(hora_hecho) != 12 -- Filtro de limpieza
    GROUP BY
        categoria, subtipo
    HAVING 
        COUNT(*) > 500
    ORDER BY
        total DESC
    """
    return run_query(query)


def get_dynamic_analysis_data(alcaldia=None, categoria=None):
    """
    Obtiene los datos filtrados para el análisis interactivo.
    """
    query = "SELECT * FROM crimes WHERE HOUR(hora_hecho) != 12"
    params = []
    
    if alcaldia and alcaldia != "Todas":
        query += " AND alcaldia_hecho = ?"
        params.append(alcaldia)
        
    if categoria and categoria != "Todas":
        query += " AND categoria_delito = ?"
        params.append(categoria)
        
    # Traemos los datos crudos filtrados (DuckDB es rápido, puede manejarlo)
    # Limitamos a las columnas necesarias para ahorrar memoria
    query = "SELECT fecha_hecho, hora_hecho, violence_type, delito FROM (" + query + ")"
    
    return run_query(query, params)

def get_filtered_hourly_stats(alcaldia=None, categoria=None):
    """
    Estadísticas por hora filtradas.
    """
    query = """
    SELECT 
        HOUR(hora_hecho) as hora, 
        COUNT(*) as total 
    FROM crimes 
    WHERE HOUR(hora_hecho) != 12
    """
    params = []
    
    if alcaldia and alcaldia != "Todas":
        query += " AND alcaldia_hecho = ?"
        params.append(alcaldia)
    if categoria and categoria != "Todas":
        query += " AND categoria_delito = ?"
        params.append(categoria)
        
    query += " GROUP BY hora ORDER BY hora"
    return run_query(query, params)

def get_crime_type_risk_metrics():
    """
    Analiza cada TIPO de delito específico para ver su volumen y % de violencia.
    """
    query = """
    SELECT
        delito,
        crime_classification,
        COUNT(*) as total_casos,
        SUM(CASE WHEN violence_type = 'Violent' THEN 1 ELSE 0 END) * 100.0 / COUNT(*) as probabilidad_violencia
    FROM
        crimes
    WHERE
        delito IS NOT NULL
        AND HOUR(hora_hecho) != 12 -- Mantenemos el filtro de limpieza
    GROUP BY
        delito, crime_classification
    HAVING
        COUNT(*) > 200 -- Ignoramos delitos muy raros para limpiar el gráfico
    ORDER BY
        probabilidad_violencia DESC
    """
    return run_query(query)