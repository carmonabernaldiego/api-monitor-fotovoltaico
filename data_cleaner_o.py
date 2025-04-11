import pandas as pd
import numpy as np
import logging
from datetime import datetime

def remove_empty_or_zero_columns(df, skip_cols=None):
    """
    Elimina columnas que son completamente NaN o que tienen todos sus valores iguales a 0.
    skip_cols: lista de columnas que NO deben eliminarse (por ejemplo, 'measured_on').
    """
    if skip_cols is None:
        skip_cols = []
        
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    for col in numeric_cols:
        if col in skip_cols:
            continue
        # Elimina la columna si todos los valores son NaN
        if df[col].isna().all():
            df.drop(columns=[col], inplace=True)
            continue
        # Rellena temporalmente NaN con 0 y verifica si todos sus valores son 0
        temp_col = df[col].fillna(0)
        if (temp_col == 0).all():
            df.drop(columns=[col], inplace=True)
    return df

def remove_rows_completely_zero(df):
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    # Calcula la suma de cada fila en las columnas numéricas
    zero_sum = (df[numeric_cols].fillna(0).sum(axis=1) == 0)
    # Conserva filas donde measured_on no sea NaT y además NO tengan suma cero en *todas* las numéricas
    return df.loc[~zero_sum].copy()

class DataCleaner:
    def __init__(self):
        self.environment_data = None
        self.electrical_data = None
        self.irradiance_data = None
        self.quality_metrics = {
            'missing_values': {},
            'outliers': {},
            'processing_time': {}
        }
        self._setup_logging()
    
    def _setup_logging(self):
        logging.basicConfig(
            filename='data_processing.log',
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s'
        )
    
    def load_data(self):
        """
        Carga los datos de los CSV y convierte la columna 'measured_on' a datetime.
        Se asume que los archivos se encuentran en la carpeta "data/".
        """
        self.electrical_data = pd.read_csv('data/2107_electrical_data.csv', engine='python')
        self.environment_data = pd.read_csv('data/2107_environment_data.csv', engine='python')
        self.irradiance_data = pd.read_csv('data/2107_irradiance_data.csv', engine='python')
        
        self.electrical_data['measured_on'] = pd.to_datetime(self.electrical_data['measured_on'], errors='coerce')
        self.environment_data['measured_on'] = pd.to_datetime(self.environment_data['measured_on'], errors='coerce')
        self.irradiance_data['measured_on'] = pd.to_datetime(self.irradiance_data['measured_on'], errors='coerce')
    
    def clean_electrical_data(self):
        start_time = datetime.now()
        initial_rows = len(self.electrical_data)
        logging.info(f"Inicio limpieza eléctrico: {initial_rows} registros")

        # Reemplazo de cadenas inválidas
        self.electrical_data = self.electrical_data.replace(
            ['N/A', '', ' ', 'null', 'NULL'], np.nan
        )

        # Conversión a numérico
        numeric_cols = self.electrical_data.columns.drop('measured_on')
        for col in numeric_cols:
            self.electrical_data[col] = pd.to_numeric(self.electrical_data[col], errors='coerce')

        # Eliminar columnas totalmente vacías o todas ceros
        self.electrical_data = remove_empty_or_zero_columns(
            self.electrical_data, skip_cols=['measured_on']
        )

        # Rellenar NaN numéricos (o interpolar)
        num_cols = self.electrical_data.select_dtypes(include=[np.number]).columns
        self.electrical_data[num_cols] = self.electrical_data[num_cols].fillna(0)
        # — o bien:
        # self.electrical_data[num_cols] = self.electrical_data[num_cols].interpolate()

        # Eliminar sólo filas sin fecha válida
        self.electrical_data.dropna(subset=['measured_on'], inplace=True)

        # Ordenar
        self.electrical_data.sort_values('measured_on', inplace=True)

        # Eliminar filas que queden completamente a cero
        self.electrical_data = remove_rows_completely_zero(self.electrical_data)

        # Guardar CSV limpio
        final_rows = len(self.electrical_data)
        processing_time = (datetime.now() - start_time).total_seconds()
        self.quality_metrics['processing_time']['cleaning'] = processing_time
        self.quality_metrics['rows_removed'] = initial_rows - final_rows
        logging.info(
            f"Limpieza eléctrica completada: se eliminaron {initial_rows - final_rows} registros en {processing_time:.2f}s"
        )

        self.electrical_data.to_csv('data/2107_electrical_data_cleaned.csv', index=False)
        return self.electrical_data

    def clean_environment_data(self):
        """
        Limpia el dataset de ambiente de forma similar al eléctrico.
        Guarda el CSV limpio en 'data/2107_environment_data_cleaned.csv'.
        """
        self.environment_data = self.environment_data.replace(['N/A', '', ' ', 'null', 'NULL'], np.nan)
        numeric_cols = self.environment_data.columns.drop('measured_on')
        for col in numeric_cols:
            self.environment_data[col] = pd.to_numeric(self.environment_data[col], errors='coerce')
        
        self.environment_data = remove_empty_or_zero_columns(self.environment_data, skip_cols=['measured_on'])
        self.environment_data.dropna(axis=0, how='any', inplace=True)
        self.environment_data.sort_values('measured_on', inplace=True)
        self.environment_data.to_csv('data/2107_environment_data_cleaned.csv', index=False)
        return self.environment_data

    def clean_irradiance_data(self):
        """
        Limpia el dataset de irradiancia.
        Guarda el CSV limpio en 'data/2107_irradiance_data_cleaned.csv'.
        """
        self.irradiance_data = self.irradiance_data.replace(['N/A', '', ' ', 'null', 'NULL'], np.nan)
        numeric_cols = self.irradiance_data.columns.drop('measured_on')
        for col in numeric_cols:
            self.irradiance_data[col] = pd.to_numeric(self.irradiance_data[col], errors='coerce')
        
        self.irradiance_data = remove_empty_or_zero_columns(self.irradiance_data, skip_cols=['measured_on'])
        self.irradiance_data.dropna(axis=0, how='any', inplace=True)
        self.irradiance_data.sort_values('measured_on', inplace=True)
        self.irradiance_data.to_csv('data/2107_irradiance_data_cleaned.csv', index=False)
        return self.irradiance_data

    def filter_by_date(self, df, start_date=None, end_date=None):
        """
        Filtra el DataFrame por rango de fechas utilizando 'measured_on'.
        start_date y end_date pueden ser cadenas en formato 'YYYY-MM-DD' o datetime.
        """
        df['measured_on'] = pd.to_datetime(df['measured_on'])
        if start_date is not None:
            df = df[df['measured_on'] >= pd.to_datetime(start_date)]
        if end_date is not None:
            df = df[df['measured_on'] <= pd.to_datetime(end_date)]
        return df

    def aggregate_data(self, df, freq='H'):
        """
        Agrega (resamplea) el DataFrame con la frecuencia indicada (por ejemplo, '15T' para 15 minutos, 'H' para cada hora, 'D' diario, etc.).
        Calcula la media de los valores numéricos.
        """
        df['measured_on'] = pd.to_datetime(df['measured_on'])
        df = df.set_index('measured_on')
        df_agg = df.resample(freq).mean().dropna().reset_index()
        return df_agg

__all__ = ['DataCleaner']
