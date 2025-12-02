import joblib
import pandas as pd
import duckdb
from geopy.geocoders import Nominatim
from geopy.extra.rate_limiter import RateLimiter
import time

print("Iniciando pre-procesamiento de clusters...")

# --- 1. Cargar el modelo KMeans ---
try:
    kmeans = joblib.load('kmeans_zonas.joblib')
    cluster_centers = kmeans.cluster_centers_
    if len(cluster_centers) < 50:
         print(f"ADVERTENCIA: Se cargaron solo {len(cluster_centers)} clusters. ¿El modelo KMeans se entrenó correctamente?")
    else:
        print(f"Éxito: Se cargaron {len(cluster_centers)} centros de cluster desde 'kmeans_zonas.joblib'.")
except Exception as e:
    print(f"Error fatal cargando 'kmeans_zonas.joblib': {e}")
    exit()

# --- 2. Cargar datos de crímenes para contexto ---
try:
    con = duckdb.connect('cdmx_insights.db')
    # Traemos solo las columnas necesarias para optimizar
    df_crimes = con.execute("SELECT alcaldia_hecho, categoria_delito, latitud, longitud FROM crimes WHERE alcaldia_hecho IS NOT NULL AND categoria_delito IS NOT NULL AND latitud IS NOT NULL AND longitud IS NOT NULL").fetchdf()
    con.close()
    if df_crimes.empty:
        print("Error fatal: La base de datos 'cdmx_insights.db' no devolvió datos.")
        exit()
    print(f"Cargados {len(df_crimes)} registros de la base de datos.")
except Exception as e:
    print(f"Error fatal conectando o leyendo 'cdmx_insights.db': {e}")
    exit()

# --- 3. Inicializar Geocoder (Nominatim) ---
geolocator = Nominatim(user_agent="cdmx-insights-project")
# Geocodificación inversa (coords -> dirección)
# Ponemos un límite de 1.1 segundos entre llamadas para cumplir las reglas de la API gratuita
reverse = RateLimiter(geolocator.reverse, min_delay_seconds=1.1, error_wait_seconds=10)
print("Geocoder inicializado. El proceso tardará aprox 1-2 minutos...")

# --- 4. Procesar cada cluster ---
cluster_info_list = []
for i, center in enumerate(cluster_centers):
    lat, lon = center[0], center[1]
    
    # Encontrar la alcaldía y categoría más comunes para este cluster
    df_crimes['distancia'] = ((df_crimes['latitud'] - lat)**2 + (df_crimes['longitud'] - lon)**2)**0.5
    punto_mas_cercano = df_crimes.loc[df_crimes['distancia'].idxmin()]
    
    alcaldia = punto_mas_cercano['alcaldia_hecho']
    categoria = punto_mas_cercano['categoria_delito']
    
    # Obtener nombre de la calle (Reverse Geocoding)
    try:
        location = reverse((lat, lon), language='es', timeout=10)
        calle = location.address if location else "Dirección no encontrada"
        # ¡Línea clave para ver el progreso!
        print(f"Procesando Cluster {i+1}/{len(cluster_centers)}: {calle}") 
    except Exception as e:
        calle = f"Error de geocoding: {e}"
        print(f"Cluster {i+1}/{len(cluster_centers)}: Error de geocoding.")
        
    cluster_info_list.append({
        'cluster_id': i,
        'latitud': lat,
        'longitud': lon,
        'alcaldia_comun': alcaldia,
        'categoria_comun': categoria,
        'calle_cercana': calle
    })

# --- 5. Guardar en CSV ---
df_clusters = pd.DataFrame(cluster_info_list)
df_clusters.to_csv('cluster_info.csv', index=False)

print("\n-------------------------------------------------")
print(f"¡Éxito! Archivo 'cluster_info.csv' creado con {len(df_clusters)} filas.")
print("-------------------------------------------------")
print(df_clusters.head())