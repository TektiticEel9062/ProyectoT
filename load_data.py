import duckdb
import pandas as pd
import os

# --- Configuración ---+
CSV_FILE_NAME = 'cleaned_crime_data.csv' 

# 2. Este es el nombre del archivo de base de datos que se creará
DB_FILE_NAME = 'cdmx_insights.db'

AÑO_MINIMO = 2016

# --- Fin de Configuración ---

def create_database():
    """
    Lee el CSV y crea el archivo de base de datos DuckDB,
    aplicando un filtro de año.
    """
    
    # Verifica si el CSV existe
    if not os.path.exists(CSV_FILE_NAME):
        print(f"Error: No se encontró el archivo '{CSV_FILE_NAME}' en esta carpeta.")
        return

    # Verifica si la base de datos ya existe
    if os.path.exists(DB_FILE_NAME):
        print(f"El archivo '{DB_FILE_NAME}' ya existe.")
        overwrite = input("¿Quieres sobrescribirlo? (s/n): ").lower()
        if overwrite != 's':
            print("Operación cancelada.")
            return
        else:
            os.remove(DB_FILE_NAME)

    print(f"Iniciando carga de '{CSV_FILE_NAME}'... Esto puede tardar unos minutos.")
    print(f"Aplicando filtro: se incluirán datos desde el año {AÑO_MINIMO} en adelante.")

    try:
        con = duckdb.connect(DB_FILE_NAME)

        query = f"""
        CREATE TABLE crimes AS 
        SELECT * FROM read_csv_auto('{CSV_FILE_NAME}')
        WHERE anio_hecho >= {AÑO_MINIMO}
        """
        
        con.execute(query)
        
        print("\n¡Éxito! Tabla 'crimes' creada con el filtro de año.")
        
        count_result = con.execute("SELECT COUNT(*) FROM crimes").fetchdf()
        print(f"Total de filas insertadas (ya filtradas): {count_result.iloc[0,0]}")
        
        con.close()
        print(f"\n¡Base de datos '{DB_FILE_NAME}' creada exitosamente!")

    except Exception as e:
        print(f"Ocurrió un error: {e}")
        print("POSIBLE ERROR: Asegúrate de que la columna 'anio_hecho' exista en tu CSV y sea numérica.")

if __name__ == "__main__":
    create_database()