import pandas as pd
import logging
from sqlalchemy import create_engine
from datetime import datetime

class DataCleaner:
    def __init__(self):
        self.environment_data = None
        self.electrical_data = None
        self.irradiance_data = None
        self.quality_metrics = {}
        self._setup_logging()
        
        # Cambia esto según tu configuración de MySQL
        self.db_url = "mysql+mysqlconnector://admin:3CB0onBwj6dKk5ibLc8l@database-monitor-solar.c388c6akq1gk.us-east-2.rds.amazonaws.com:3306/solar_data"
        self.engine = create_engine(self.db_url)

    def _setup_logging(self):
        logging.basicConfig(
            filename='data_processing.log',
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s'
        )

    def load_data(self):
        try:
            self.electrical_data = pd.read_sql("SELECT * FROM electrical_data", self.engine)
            self.environment_data = pd.read_sql("SELECT * FROM environment_data", self.engine)
            self.irradiance_data = pd.read_sql("SELECT * FROM irradiance_data", self.engine)
            
            # Convertimos a datetime
            self.electrical_data['measured_on'] = pd.to_datetime(self.electrical_data['measured_on'], errors='coerce')
            self.environment_data['measured_on'] = pd.to_datetime(self.environment_data['measured_on'], errors='coerce')
            self.irradiance_data['measured_on'] = pd.to_datetime(self.irradiance_data['measured_on'], errors='coerce')

            logging.info("Datos cargados desde la base de datos")
        except Exception as e:
            logging.error(f"Error al cargar datos desde MySQL: {e}")

    def filter_by_date(self, df, start_date=None, end_date=None):
        df['measured_on'] = pd.to_datetime(df['measured_on'])
        if start_date:
            df = df[df['measured_on'] >= pd.to_datetime(start_date)]
        if end_date:
            df = df[df['measured_on'] <= pd.to_datetime(end_date)]
        return df

    def aggregate_data(self, df, freq='H'):
        df['measured_on'] = pd.to_datetime(df['measured_on'])
        df = df.set_index('measured_on')
        df_agg = df.resample(freq).mean().dropna().reset_index()
        return df_agg
