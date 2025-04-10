import os
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

# ===============================
# Fijar semillas para reproducibilidad
# ===============================
seed_value = 42
os.environ['PYTHONHASHSEED'] = str(seed_value)
random.seed(seed_value)
np.random.seed(seed_value)
tf.random.set_seed(seed_value)

# ===============================
# 1. Carga y limpieza de datos
# ===============================
def load_and_clean_data(filepath):
    """
    Carga el dataset, convierte la columna 'measured_on' a datetime,
    reemplaza celdas vacías por NaN, convierte las demás columnas a numérico
    y elimina filas en que todas las mediciones sean 0 o NaN.
    """
    # Cargar el CSV
    df = pd.read_csv(filepath, header=0)
    
    # Convertir la columna de fecha a datetime
    df['measured_on'] = pd.to_datetime(df['measured_on'], errors='coerce')
    
    # Reemplazar celdas vacías o espacios por NaN
    df.replace(r'^\s*$', np.nan, regex=True, inplace=True)
    
    # Convertir todas las columnas excepto 'measured_on' a numéricas
    cols = df.columns.drop('measured_on')
    df[cols] = df[cols].apply(pd.to_numeric, errors='coerce')
    
    # Eliminar filas con fecha inválida y ordenar por fecha
    df = df.dropna(subset=['measured_on'])
    df.sort_values('measured_on', inplace=True)
    
    # Función para determinar si una fila está "vacía" (sin actividad)
    def fila_sin_actividad(row):
        return (row[cols].isna() | (row[cols] == 0)).all()
    
    # Eliminar filas donde todas las mediciones sean 0 o NaN
    df_clean = df[~df.apply(fila_sin_actividad, axis=1)].copy()
    
    return df_clean

# ===============================
# 2. Análisis exploratorio de datos
# ===============================
def exploratory_data_analysis(df):
    print("Primeras filas del dataset:")
    print(df.head())
    
    print("\nInformación del dataset:")
    print(df.info())
    
    print("\nEstadísticas descriptivas:")
    print(df.describe())
    
    # Matriz de correlación (excluyendo la columna de fecha)
    plt.figure(figsize=(12, 10))
    corr = df.drop(columns=['measured_on']).corr()
    sns.heatmap(corr, cmap='coolwarm', annot=False)
    plt.title("Matriz de correlación de mediciones")
    plt.show()
    
    # Ejemplo: Serie de tiempo del Voltaje DC del Inversor 1
    plt.figure(figsize=(14, 6))
    plt.plot(df['measured_on'], df['inv_01_dc_voltage_inv_149580'], marker='.', linestyle='-')
    plt.title("Serie de tiempo - Voltaje DC (Inversor 1)")
    plt.xlabel("Fecha")
    plt.ylabel("Voltaje DC (V)")
    plt.show()

# ===============================
# 3. Cálculo de eficiencia para el Inversor 1
# ===============================
def compute_efficiency_inv1(df):
    """
    Calcula la eficiencia para el Inversor 1 utilizando:
      - Corriente DC: inv_01_dc_current_inv_149579
      - Voltaje DC: inv_01_dc_voltage_inv_149580
      - Potencia AC: inv_01_ac_power_inv_149583
    Eficiencia (%) = (Potencia AC / (Voltaje DC * Corriente DC)) * 100
    """
    current_col = "inv_01_dc_current_inv_149579"
    voltage_col = "inv_01_dc_voltage_inv_149580"
    ac_power_col = "inv_01_ac_power_inv_149583"
    
    # Calcular la potencia DC
    df['potencia_dc_inv1'] = df[voltage_col] * df[current_col]
    
    # Calcular eficiencia; evitar división por cero
    df['eficiencia_inv1'] = np.where(df['potencia_dc_inv1'] > 0,
                                     (df[ac_power_col] / df['potencia_dc_inv1']) * 100,
                                     np.nan)
    return df

# ===============================
# 4. Modelo de regresión lineal para Inversor 1
# ===============================
def regression_model_inv1(df):
    """
    Se Calcula el area bajo la curpa de la potencia en DC y de la potencia AC y se dividen e
    Entrena un modelo de regresión lineal para predecir la Potencia AC
    del Inversor 1 a partir de la Corriente DC y el Voltaje DC.
    """
    current_col = "inv_01_dc_current_inv_149579"
    voltage_col = "inv_01_dc_voltage_inv_149580"
    ac_power_col = "inv_01_ac_power_inv_149583"
    
    # Seleccionar registros sin nulos en las columnas de interés
    reg_df = df[[current_col, voltage_col, ac_power_col]].dropna()
    X = reg_df[[current_col, voltage_col]].values
    y = reg_df[ac_power_col].values
    
    # División en train y test
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=seed_value)
    
    # Ajuste del modelo
    model = LinearRegression()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    
    mse = mean_squared_error(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    
    print("=== Regresión Lineal - Inversor 1 ===")
    print(f"Coeficientes: {model.coef_}")
    print(f"Intercepto: {model.intercept_}")
    print(f"MSE: {mse:.4f}")
    print(f"MAE: {mae:.4f}")
    print(f"R²: {r2:.4f}")
    
    # Gráfica: Valores reales vs predichos
    plt.figure(figsize=(8, 6))
    plt.scatter(y_test, y_pred, alpha=0.5)
    plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], 'r--')
    plt.xlabel("Valores reales")
    plt.ylabel("Predicciones")
    plt.title("Regresión: Potencia AC (Inversor 1)")
    plt.show()

# ===============================
# 5. Pronóstico de series temporales con LSTM (Inversor 1)
# ===============================
def create_sequences(data, seq_length):
    sequences = []
    targets = []
    for i in range(len(data) - seq_length):
        sequences.append(data[i: i + seq_length])
        targets.append(data[i + seq_length])
    return np.array(sequences), np.array(targets)

def forecast_ac_power_lstm(df, seq_length=50, epochs=50):
    """
    Pronostica la Potencia AC del Inversor 1 utilizando un modelo LSTM.
    Se normaliza la serie de tiempo, se crean secuencias y se entrena el modelo.
    """
    ac_power_col = "inv_01_ac_power_inv_149583"
    # Seleccionar la serie de tiempo y ordenar por fecha
    ts = df[['measured_on', ac_power_col]].dropna()
    ts = ts.sort_values('measured_on')
    ts_values = ts[ac_power_col].values.reshape(-1, 1)
    
    # Normalización
    scaler = MinMaxScaler(feature_range=(0, 1))
    ts_scaled = scaler.fit_transform(ts_values)
    
    # Crear secuencias
    X, y = create_sequences(ts_scaled, seq_length)
    
    # División en train y test
    train_size = int(len(X) * 0.8)
    X_train, X_test = X[:train_size], X[train_size:]
    y_train, y_test = y[:train_size], y[train_size:]
    
    # Definir el modelo LSTM
    model = Sequential()
    model.add(LSTM(50, activation='relu', input_shape=(seq_length, 1)))
    model.add(Dropout(0.2))
    model.add(Dense(1))
    model.compile(optimizer='adam', loss='mse')
    
    # Callbacks para evitar sobreentrenamiento
    early_stop = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=3, min_lr=1e-6)
    
    print("\nEntrenando modelo LSTM para pronosticar Potencia AC (Inversor 1)...")
    history = model.fit(X_train, y_train, epochs=epochs, batch_size=64,
                        validation_split=0.2, callbacks=[early_stop, reduce_lr], verbose=1)
    
    # Predicciones y evaluación
    y_pred = model.predict(X_test)
    y_pred_inv = scaler.inverse_transform(y_pred)
    y_test_inv = scaler.inverse_transform(y_test)
    
    mse = mean_squared_error(y_test_inv, y_pred_inv)
    r2 = r2_score(y_test_inv, y_pred_inv)
    
    print("=== Modelo LSTM para Potencia AC (Inversor 1) ===")
    print(f"MSE: {mse:.4f}")
    print(f"R²: {r2:.4f}")
    
    # Gráfica de la serie real vs predicha
    plt.figure(figsize=(10, 6))
    plt.plot(y_test_inv, label="Real")
    plt.plot(y_pred_inv, label="Predicción")
    plt.title("Pronóstico de Potencia AC - LSTM")
    plt.xlabel("Muestras")
    plt.ylabel("Potencia AC")
    plt.legend()
    plt.show()

# ===============================
# Función principal
# ===============================
if __name__ == "__main__":
    # Ruta al archivo CSV (ajusta la ruta según corresponda)
    filepath = "2107_electrical_data.csv"
    
    # Cargar y limpiar datos
    df = load_and_clean_data(filepath)
    print(f"Dataset limpio: {df.shape[0]} filas, {df.shape[1]} columnas")
    
    # Análisis exploratorio
    exploratory_data_analysis(df)
    
    # Calcular eficiencia para el Inversor 1 y visualizar su distribución
    df = compute_efficiency_inv1(df)
    plt.figure(figsize=(10, 6))
    sns.histplot(df['eficiencia_inv1'].dropna(), bins=30, kde=True)
    plt.title("Distribución de Eficiencia - Inversor 1")
    plt.xlabel("Eficiencia (%)")
    plt.ylabel("Frecuencia")
    plt.show()
    
    # Modelo de regresión lineal para predecir Potencia AC (Inversor 1)
    regression_model_inv1(df)
    
    # Pronóstico de series temporales con LSTM para Potencia AC (Inversor 1)
    forecast_ac_power_lstm(df, seq_length=50, epochs=50)
