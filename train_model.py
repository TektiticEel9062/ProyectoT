import duckdb
import pandas as pd
import numpy as np
import joblib
import statsmodels.api as sm
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.cluster import KMeans 
from sklearn.metrics import classification_report
from scipy.stats import randint 
import warnings

warnings.filterwarnings('ignore')

DB_FILE = "cdmx_insights.db"

def load_data_for_classification():
    """
    Carga de datos V2.
    Incluye lat, lon, hora_hecho, etc.
    """
    print("Cargando datos para clasificación (v2)...")
    con = duckdb.connect(DB_FILE, read_only=True)
    query = """
    SELECT 
        HOUR(hora_hecho) AS hora_hecho,
        mes_hecho, 
        CAST(fecha_hecho AS DATE) AS fecha_hecho, 
        alcaldia_hecho, 
        categoria_delito, 
        latitud, 
        longitud, 
        violence_type 
    FROM 
        crimes
    WHERE
        violence_type IS NOT NULL AND
        alcaldia_hecho IS NOT NULL AND
        categoria_delito IS NOT NULL AND
        latitud IS NOT NULL AND
        longitud IS NOT NULL
    LIMIT 300000 -- Un poco más de datos para mejor generalización
    """
    df = con.execute(query).fetchdf()
    con.close()
    df = df.dropna(subset=['latitud', 'longitud', 'alcaldia_hecho', 'categoria_delito'])
    print(f"Datos cargados: {len(df)} filas.")
    return df

def load_data_for_timeseries():
    print("Cargando datos para series de tiempo...")
    con = duckdb.connect(DB_FILE, read_only=True)
    query = "SELECT CAST(fecha_hecho AS DATE) AS fecha, COUNT(*) AS total_delitos FROM crimes GROUP BY fecha ORDER BY fecha ASC"
    df = con.execute(query).fetchdf()
    con.close()
    df['fecha'] = pd.to_datetime(df['fecha'])
    df = df.set_index('fecha')
    df_resampled = df.asfreq('D').fillna(0)
    print(f"Datos de series de tiempo listos: {len(df_resampled)} días.")
    return df_resampled['total_delitos']

def map_to_time_slot(hour):
    """Convierte una hora (0-23) en una franja horaria categórica."""
    if 0 <= hour <= 5: return 'Madrugada'
    elif 6 <= hour <= 11: return 'Mañana'
    elif 12 <= hour <= 18: return 'Tarde'
    return 'Noche'

def train_classification_models():
    """
    Entrena modelos V2 (Features Separados).
    """
    df = load_data_for_classification()
    
    X = df.drop('violence_type', axis=1)
    y = df['violence_type'].map({'Non-Violent': 0, 'Violent': 1})
    target_names = ['Non-Violent', 'Violent']

    # --- Feature Engineering (Fechas) ---
    print("Iniciando Feature Engineering (Fechas)...")
    X['fecha_hecho'] = pd.to_datetime(X['fecha_hecho'])
    X['dia_de_la_semana'] = X['fecha_hecho'].dt.dayofweek
    X['es_fin_de_semana'] = (X['dia_de_la_semana'] >= 5).astype(int)
    X['es_quincena'] = X['fecha_hecho'].dt.day.isin([14,15,16, 28,29,30,31,1,2]).astype(int)

    # --- División de Datos ---
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # --- Feature Engineering (Clustering - V2) ---
    print("Iniciando Feature Engineering (Clustering de Ubicación)...")
    # Usamos 50 clusters como en la V2 original
    kmeans = KMeans(n_clusters=50, random_state=42, n_init=10) 
    coords_train = X_train[['latitud', 'longitud']]
    X_train['zona_cluster'] = kmeans.fit_predict(coords_train)
    coords_test = X_test[['latitud', 'longitud']]
    X_test['zona_cluster'] = kmeans.predict(coords_test)
    
    joblib.dump(kmeans, 'kmeans_zonas.joblib')
    print("Modelo de Clustering guardado.")

    # --- Feature Engineering (Tiempo - V2) ---
    print("Iniciando Feature Engineering (V2 - Separados)...")
    for df_split in [X_train, X_test]:
        # 1. Franja Horaria (SEPARADA)
        df_split['franja_horaria'] = df_split['hora_hecho'].apply(map_to_time_slot)
        
        # 2. Cíclicos Mes
        df_split['mes_sin'] = np.sin(2 * np.pi * df_split['mes_hecho'] / 12)
        df_split['mes_cos'] = np.cos(2 * np.pi * df_split['mes_hecho'] / 12)
    
    # --- Preprocesador V2 ---
    # Aquí 'zona_cluster' y 'franja_horaria' son features independientes
    categorical_features = ['alcaldia_hecho', 'categoria_delito', 'zona_cluster', 'franja_horaria'] 
    
    numeric_features = [
        'dia_de_la_semana', 
        'es_fin_de_semana', 
        'es_quincena',
        'mes_sin', 
        'mes_cos'
    ]
    
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', 'passthrough', numeric_features),
            ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), categorical_features)
        ],
        remainder='drop' 
    )

    # --- Entrenamiento XGBoost (V2) ---
    print("\n===== ENTRENANDO XGBOOST (v2) =====")
    scale_pos_weight = (y_train == 0).sum() / (y_train == 1).sum()
    
    pipeline_xgb = Pipeline(steps=[('preprocessor', preprocessor),
                                   ('classifier', XGBClassifier(random_state=42,
                                                               n_jobs=-1,
                                                               eval_metric='logloss',
                                                               use_label_encoder=False,
                                                               scale_pos_weight=scale_pos_weight))])
    
    # Hyperparameters V2 (Conservadores)
    param_dist_xgb = {
        'classifier__n_estimators': randint(100, 300),
        'classifier__max_depth': [5, 6, 8],
        'classifier__learning_rate': [0.05, 0.1],
        'classifier__subsample': [0.8, 1.0]
    }

    random_search_xgb = RandomizedSearchCV(pipeline_xgb, 
                                           param_distributions=param_dist_xgb, 
                                           n_iter=5, # Pocas iteraciones para ir rápido
                                           cv=3, 
                                           verbose=1, 
                                           random_state=42, 
                                           n_jobs=4, 
                                           scoring='f1_weighted')
    
    random_search_xgb.fit(X_train, y_train)
    print(f"Mejores parámetros XGB v2: {random_search_xgb.best_params_}")
    best_model_xgb = random_search_xgb.best_estimator_
    
    y_pred_xgb = best_model_xgb.predict(X_test)
    print("\n--- Reporte XGBoost (v2) ---")
    print(classification_report(y_test, y_pred_xgb, target_names=target_names))
    
    # 🔥 GUARDAR COMO V2
    joblib.dump(best_model_xgb, 'violence_xgb_optimizado_v2.joblib')
    print("Modelo guardado en 'violence_xgb_optimizado_v2.joblib'")

def train_timeseries_model():
    y = load_data_for_timeseries()
    print("\n===== ENTRENANDO SERIES DE TIEMPO =====")
    model = sm.tsa.statespace.SARIMAX(y, order=(1, 1, 1), seasonal_order=(1, 1, 1, 7))
    model_fit = model.fit(disp=False)
    model_fit.save('crime_forecaster.pkl')
    print("Modelo SARIMA guardado.")

if __name__ == "__main__":
    print("===== INICIANDO ENTRENAMIENTO (Restauración v2) =====")
    train_classification_models()
    train_timeseries_model()
    print("\n===== FINALIZADO =====")