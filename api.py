import os
import logging
import shutil
from fastapi import FastAPI, Query, HTTPException
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from datetime import datetime
from typing import Optional
from fastapi.middleware.cors import CORSMiddleware

import pandas as pd
import numpy as np
import matplotlib
# Configurar backend no interactivo ANTES de importar pyplot
matplotlib.use('Agg')  # Esto es crucial - debe ir antes de importar plt
import matplotlib.pyplot as plt
import seaborn as sns

# Importamos el DataCleaner y todas las funciones de análisis que tenías en tu main.py
from data_cleaner import DataCleaner

###################################
# 1. Instanciamos la aplicación
###################################
app = FastAPI(
    title="Sistema de Monitoreo Solar - API",
    description="API que expone datos limpios, métricas y gráficos de un sistema fotovoltaico.",
    version="1.0.0",
)

# Aquí el middleware:
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # o ["http://localhost:5500"] si tienes un servidor web
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Montar carpeta para servir archivos estáticos (imágenes)
if not os.path.exists("static/images"):
    os.makedirs("static/images")
app.mount("/images", StaticFiles(directory="static/images"), name="images")

########################################
# 2. Copiamos las funciones de análisis
########################################

def calcular_performance_ratio(electrical_data, irradiance_data, capacidad_instalada_kw):
    electrical_data['measured_on'] = pd.to_datetime(electrical_data['measured_on'])
    irradiance_data['measured_on'] = pd.to_datetime(irradiance_data['measured_on'])
    merged = pd.merge(electrical_data, irradiance_data, on='measured_on', how='inner')
    if merged.empty:
        return None
    
    ac_power_cols = [col for col in merged.columns if 'ac_power' in col]
    irr_cols = [col for col in merged.columns if 'irradiance' in col or 'poa_irradiance' in col]
    if not ac_power_cols or not irr_cols:
        return None
    
    merged['total_ac_power'] = merged[ac_power_cols].sum(axis=1)
    irr_col = irr_cols[0]
    day_data = merged[merged[irr_col] > 50]
    if day_data.empty:
        return None
    day_data['pr'] = (day_data['total_ac_power'] / capacidad_instalada_kw) / (day_data[irr_col] / 1000)
    day_data['pr'] = day_data['pr'].clip(0, 1.2)
    
    return day_data['pr'].mean(), day_data

def calcular_rendimiento_especifico(electrical_data, capacidad_instalada_kw):
    ac_power_cols = [col for col in electrical_data.columns if 'ac_power' in col]
    if not ac_power_cols:
        return None
    electrical_data['total_ac_power'] = electrical_data[ac_power_cols].sum(axis=1)
    electrical_data.sort_values('measured_on', inplace=True)
    electrical_data['date'] = electrical_data['measured_on'].dt.date
    daily_energy = electrical_data.groupby('date')['total_ac_power'].mean() * 24
    daily_specific_yield = daily_energy / capacidad_instalada_kw
    return daily_specific_yield

def detectar_anomalias(electrical_data, environment_data, irradiance_data, umbral_potencia=0.7):
    electrical_data['measured_on'] = pd.to_datetime(electrical_data['measured_on'])
    irradiance_data['measured_on'] = pd.to_datetime(irradiance_data['measured_on'])
    merged = pd.merge(electrical_data, irradiance_data, on='measured_on', how='inner')
    if merged.empty:
        return ["No hay datos coincidentes para detectar anomalías"]
    
    ac_power_cols = [col for col in merged.columns if 'ac_power' in col]
    irr_cols = [col for col in merged.columns if 'irradiance' in col or 'poa_irradiance' in col]
    if not ac_power_cols or not irr_cols:
        return ["Faltan columnas necesarias para detectar anomalías"]
    
    irr_col = irr_cols[0]
    day_data = merged[merged[irr_col] > 100].copy()
    if day_data.empty:
        return ["No hay suficientes datos diurnos para detectar anomalías"]
    
    alertas = []
    for col in ac_power_cols:
        inv_num = col.split('_')[1]
        day_data[f'ratio_{inv_num}'] = day_data[col] / day_data[irr_col]
        active_data = day_data[day_data[col] > 0.1]
        if not active_data.empty:
            ratio_promedio = active_data[f'ratio_{inv_num}'].median()
            problemas = day_data[(day_data[irr_col] > 200) &
                                 (day_data[f'ratio_{inv_num}'] < ratio_promedio * umbral_potencia) &
                                 (day_data[col] > 0)]
            if not problemas.empty:
                for idx, row in problemas.iterrows():
                    alertas.append(f"Inversor {inv_num} bajo rendimiento el {row['measured_on']}. Potencia: {row[col]:.2f}kW, ratio < {ratio_promedio*umbral_potencia:.2f}")
    
    high_irr_data = day_data[day_data[irr_col] > 300]
    for col in ac_power_cols:
        inv_num = col.split('_')[1]
        inactivos = high_irr_data[high_irr_data[col] < 0.1]
        if not inactivos.empty:
            for idx, row in inactivos.iterrows():
                alertas.append(f"Inversor {inv_num} inactivo con alta irradiancia ({row[irr_col]:.2f}W/m²) el {row['measured_on']}")
    
    return alertas

def calcular_estadisticas_energia(electrical_data):
    electrical_data['measured_on'] = pd.to_datetime(electrical_data['measured_on'])
    ac_power_cols = [col for col in electrical_data.columns if 'ac_power' in col]
    if not ac_power_cols:
        return None
    
    electrical_data['total_ac_power'] = electrical_data[ac_power_cols].sum(axis=1)
    electrical_data['date'] = electrical_data['measured_on'].dt.date
    electrical_data['month'] = electrical_data['measured_on'].dt.to_period('M')
    electrical_data['hour'] = electrical_data['measured_on'].dt.hour
    
    daily_energy = electrical_data.groupby('date')['total_ac_power'].sum()
    monthly_energy = electrical_data.groupby('month')['total_ac_power'].sum()
    hourly_avg = electrical_data.groupby('hour')['total_ac_power'].mean()
    peak_hour = hourly_avg.idxmax()
    peak_power = hourly_avg.max()
    
    return {
        'daily_energy': daily_energy,
        'monthly_energy': monthly_energy,
        'peak_hour': peak_hour,
        'peak_power': peak_power,
        'hourly_averages': hourly_avg
    }

###############################################
# 6. Graficar (convertimos a funciones de API)
###############################################
import uuid

def plot_power_chart(electrical_data, freq=None):
    df = electrical_data.copy()
    df['measured_on'] = pd.to_datetime(df['measured_on'])
    df.sort_values('measured_on', inplace=True)
    
    if freq is not None:
        df = df.set_index('measured_on').resample(freq).mean().dropna().reset_index()
    
    ac_power_cols = [col for col in df.columns if 'ac_power' in col]
    if not ac_power_cols:
        return None
    
    for col in ac_power_cols:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    
    active_inverters = [col for col in ac_power_cols if (df[col] > 0.1).sum() >= 10]
    if not active_inverters:
        return None
    
    active_inverters.sort()
    # Crear figura sin que matplotlib intente usar un backend interactivo
    plt.figure(figsize=(14,8))
    palette = plt.cm.tab20(np.linspace(0,1,len(active_inverters)))
    for i, col in enumerate(active_inverters):
        inv_num = col.split('_')[1]
        plt.plot(df['measured_on'], df[col],
                 label=f'Inverter {inv_num}',
                 linewidth=1.5, alpha=0.8, color=palette[i])
    
    if len(active_inverters) > 1:
        total_power = df[active_inverters].sum(axis=1)
        rolling_avg = total_power.rolling(window=3, center=True).mean()
        plt.plot(df['measured_on'], rolling_avg, label='Total Power (rolling avg)',
                 linewidth=3, color='black', linestyle='--')
    
    plt.title("AC Power by Inverter (kW)")
    plt.xlabel("Time")
    plt.ylabel("AC Power (kW)")
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.xticks(rotation=45)
    plt.tight_layout()
    
    filename = f"power_{uuid.uuid4()}.png"
    output_path = os.path.join("static/images", filename)
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    return filename

def plot_environment_chart(environment_data, freq=None):
    df = environment_data.copy()
    df['measured_on'] = pd.to_datetime(df['measured_on'])
    df.sort_values('measured_on', inplace=True)
    if freq is not None:
        df = df.set_index('measured_on').resample(freq).mean().dropna().reset_index()
    
    temp_cols = [col for col in df.columns if 'ambient_temperature' in col]
    wind_cols = [col for col in df.columns if 'wind_speed' in col]
    if not temp_cols or not wind_cols:
        return None
    
    temp_col = temp_cols[0]
    wind_col = wind_cols[0]
    
    plt.figure(figsize=(12,6))
    ax1 = plt.gca()
    ax1.plot(df['measured_on'], df[temp_col], color='red', marker='o', label='Temperatura (°C)')
    ax1.set_xlabel("Time")
    ax1.set_ylabel("Temperatura (°C)", color='red')
    ax1.tick_params(axis='y', labelcolor='red')
    
    ax2 = ax1.twinx()
    ax2.plot(df['measured_on'], df[wind_col], color='blue', marker='s', label='Vel. Viento (m/s)')
    ax2.set_ylabel("Velocidad del Viento (m/s)", color='blue')
    ax2.tick_params(axis='y', labelcolor='blue')
    
    plt.title("Condiciones Ambientales")
    plt.xticks(rotation=45)
    plt.tight_layout()
    
    filename = f"env_{uuid.uuid4()}.png"
    output_path = os.path.join("static/images", filename)
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    return filename

def plot_correlation_chart(electrical_data, irradiance_data, freq=None):
    df_elec = electrical_data.copy()
    df_irr = irradiance_data.copy()
    df_elec['measured_on'] = pd.to_datetime(df_elec['measured_on'])
    df_irr['measured_on'] = pd.to_datetime(df_irr['measured_on'])
    df_elec.sort_values('measured_on', inplace=True)
    df_irr.sort_values('measured_on', inplace=True)
    
    if freq is not None:
        df_elec = df_elec.set_index('measured_on').resample(freq).mean().dropna().reset_index()
        df_irr = df_irr.set_index('measured_on').resample(freq).mean().dropna().reset_index()
    
    merged = pd.merge(df_elec, df_irr, on='measured_on', how='inner')
    if merged.empty:
        return None
    
    ac_power_cols = [col for col in df_elec.columns if 'ac_power' in col]
    if not ac_power_cols:
        return None
    
    merged['total_ac_power'] = merged[ac_power_cols].sum(axis=1)
    irradiance_cols = [col for col in df_irr.columns if 'poa_irradiance' in col or 'irradiance' in col]
    if not irradiance_cols:
        return None
    irr_col = irradiance_cols[0]
    
    merged = merged[(merged['total_ac_power'] > 0) & (merged[irr_col] > 0)]
    if len(merged) < 5:
        return None
    
    plt.figure(figsize=(12,6))
    ax1 = plt.gca()
    ax1.plot(merged['measured_on'], merged[irr_col], color='#FFC107', marker='o', linewidth=2, label='Irradiance (W/m²)')
    ax1.set_xlabel("Time")
    ax1.set_ylabel("Irradiance (W/m²)", color='#FFC107')
    ax1.tick_params(axis='y', labelcolor='#FFC107')
    
    ax2 = ax1.twinx()
    ax2.plot(merged['measured_on'], merged['total_ac_power'],
             color='#4CAF50', marker='s', linewidth=2, label='Total AC Power (kW)')
    ax2.set_ylabel("Total AC Power (kW)", color='#4CAF50')
    ax2.tick_params(axis='y', labelcolor='#4CAF50')
    
    plt.title("Irradiance vs Total AC Power")
    plt.xticks(rotation=45)
    plt.tight_layout()
    
    filename = f"corr_{uuid.uuid4()}.png"
    output_path = os.path.join("static/images", filename)
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    return filename

# Añadir función de predicción que parece faltar en el código original
def predecir_produccion(irradiance_data, electrical_data):
    # Creamos una función básica ya que no estaba en el código original
    try:
        irr = irradiance_data.copy()
        elec = electrical_data.copy()
        
        irr['measured_on'] = pd.to_datetime(irr['measured_on'])
        elec['measured_on'] = pd.to_datetime(elec['measured_on'])
        
        # Fusionar datos
        merged = pd.merge(elec, irr, on='measured_on', how='inner')
        if merged.empty:
            return None
            
        # Identificar columnas
        ac_power_cols = [col for col in elec.columns if 'ac_power' in col]
        irr_cols = [col for col in irr.columns if 'irradiance' in col or 'poa_irradiance' in col]
        
        if not ac_power_cols or not irr_cols:
            return None
            
        # Preparar datos
        merged['total_ac_power'] = merged[ac_power_cols].sum(axis=1)
        irr_col = irr_cols[0]
        
        # Filtrar para datos diurnos válidos
        day_data = merged[(merged[irr_col] > 50) & (merged['total_ac_power'] > 0)]
        if len(day_data) < 10:
            return None
            
        # Modelo simple de regresión lineal
        from sklearn.linear_model import LinearRegression
        X = day_data[[irr_col]]
        y = day_data['total_ac_power']
        
        model = LinearRegression()
        model.fit(X, y)
        
        # Generar predicción para los próximos días
        # Usando los mismos patrones de irradiancia pero con el modelo
        pred_df = irr.copy()
        pred_df['forecasted_power'] = model.predict(pred_df[[irr_col]])
        pred_df['forecasted_power'] = pred_df['forecasted_power'].clip(0, None)
        
        # Calcular energía diaria predicha
        pred_df['date'] = pred_df['measured_on'].dt.date
        daily_energy = pred_df.groupby('date')['forecasted_power'].mean() * 24
        
        return pred_df[['measured_on', 'forecasted_power']], float(daily_energy.mean())
    except Exception as e:
        logging.error(f"Error en predicción: {str(e)}")
        return None

# Añadir función que falta para generar resumen ejecutivo
def generar_resumen_ejecutivo(cleaner, capacidad_instalada):
    try:
        elec = cleaner.electrical_data.copy()
        env = cleaner.environment_data.copy()
        irr = cleaner.irradiance_data.copy()
        
        # Calcular PR
        pr_result = calcular_performance_ratio(elec, irr, capacidad_instalada)
        pr_val = pr_result[0] if pr_result else None
        
        # Rendimiento específico
        spec_yield = calcular_rendimiento_especifico(elec, capacidad_instalada)
        spec_mean = spec_yield.mean() if spec_yield is not None else None
        
        # Estadísticas
        stats = calcular_estadisticas_energia(elec)
        if stats:
            daily_energy = float(stats['daily_energy'].sum())
            best_day = str(stats['daily_energy'].idxmax())
            best_day_energy = float(stats['daily_energy'].max())
            peak_hour = int(stats['peak_hour'])
            peak_power = float(stats['peak_power'])
        else:
            daily_energy = best_day = best_day_energy = peak_hour = peak_power = None
        
        # Anomalías
        alerts = detectar_anomalias(elec, env, irr)
        
        # Crear resumen
        resumen = {
            "performance_ratio": pr_val,
            "specific_yield": spec_mean,
            "daily_energy_total": daily_energy,
            "best_day": best_day,
            "best_day_energy": best_day_energy,
            "peak_hour": peak_hour,
            "peak_power": peak_power,
            "alerts": alerts,
            "summary_text": f"""
            Resumen Ejecutivo del Sistema Fotovoltaico:
            
            Performance Ratio: {pr_val:.2f if pr_val else 'N/A'}
            Rendimiento Específico: {spec_mean:.2f if spec_mean else 'N/A'} kWh/kWp
            Energía Total Generada: {daily_energy:.2f if daily_energy else 'N/A'} kWh
            Mejor Día: {best_day} con {best_day_energy:.2f if best_day_energy else 'N/A'} kWh
            Hora Pico: {peak_hour}:00 con {peak_power:.2f if peak_power else 'N/A'} kW promedio
            
            Alertas Detectadas: {len(alerts)}
            """
        }
        
        return resumen
    except Exception as e:
        logging.error(f"Error generando resumen: {str(e)}")
        return {"error": "No se pudo generar el resumen ejecutivo"}

##############################################
# 7. API con FastAPI
##############################################

cleaner = DataCleaner()

@app.on_event("startup")
def startup_event():
    """
    Al iniciar el servidor, cargamos y limpiamos datos para que estén disponibles en endpoints.
    """
    cleaner.load_data()
    cleaner.clean_electrical_data()
    cleaner.clean_environment_data()
    cleaner.clean_irradiance_data()
    logging.info("Datos cargados y limpios al iniciar la API")

@app.get("/")
def root():
    return {"message": "Bienvenido a la API del Sistema de Monitoreo Solar"}

@app.get("/metrics")
def get_metrics(
    start_date: Optional[str] = Query(None),
    end_date: Optional[str] = Query(None),
    freq: Optional[str] = Query(None, description="Frecuencia de agregación, e.g. 'D', 'H', '15T'"),
    capacidad_instalada: float = 100.0
):
    """
    Devuelve métricas de rendimiento, rendimiento específico y anomalías en JSON.
    """
    # Copiamos los DataFrames limpios
    elec = cleaner.electrical_data.copy()
    env = cleaner.environment_data.copy()
    irr = cleaner.irradiance_data.copy()
    
    # Filtramos
    if start_date or end_date:
        elec = cleaner.filter_by_date(elec, start_date, end_date)
        env = cleaner.filter_by_date(env, start_date, end_date)
        irr = cleaner.filter_by_date(irr, start_date, end_date)
    # Agregamos
    if freq:
        elec = cleaner.aggregate_data(elec, freq=freq)
        env = cleaner.aggregate_data(env, freq=freq)
        irr = cleaner.aggregate_data(irr, freq=freq)
    
    # Calculo PR
    pr_result = calcular_performance_ratio(elec, irr, capacidad_instalada)
    pr_val = pr_result[0] if pr_result else None
    
    # Rend. específico
    spec_yield = calcular_rendimiento_especifico(elec, capacidad_instalada)
    spec_mean = spec_yield.mean() if spec_yield is not None else None
    
    # Estadísticas
    stats = calcular_estadisticas_energia(elec)
    if stats:
        daily_energy = float(stats['daily_energy'].sum())
        best_day = str(stats['daily_energy'].idxmax())
        best_day_energy = float(stats['daily_energy'].max())
        peak_hour = int(stats['peak_hour'])
        peak_power = float(stats['peak_power'])
    else:
        daily_energy = best_day = best_day_energy = peak_hour = peak_power = None
    
    # Anomalías
    alerts = detectar_anomalias(elec, env, irr)
    
    return JSONResponse(content={
        "performance_ratio": pr_val,
        "specific_yield": spec_mean,
        "daily_energy_total": daily_energy,
        "best_day": best_day,
        "best_day_energy": best_day_energy,
        "peak_hour": peak_hour,
        "peak_power": peak_power,
        "alerts": alerts
    })

@app.get("/graph/power")
def graph_power(
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    freq: Optional[str] = None
):
    """
    Genera y devuelve el gráfico de potencia AC.
    Retorna un JSON con la URL de la imagen generada.
    """
    df = cleaner.electrical_data.copy()
    if start_date or end_date:
        df = cleaner.filter_by_date(df, start_date, end_date)
    if freq:
        df = cleaner.aggregate_data(df, freq=freq)
    
    filename = plot_power_chart(df)
    if filename is None:
        raise HTTPException(status_code=400, detail="No se pudo generar el gráfico (faltan columnas AC power)")
    return {"image_url": f"/images/{filename}"}

@app.get("/graph/environment")
def graph_environment(
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    freq: Optional[str] = None
):
    """
    Genera el gráfico de condiciones ambientales.
    """
    df = cleaner.environment_data.copy()
    if start_date or end_date:
        df = cleaner.filter_by_date(df, start_date, end_date)
    if freq:
        df = cleaner.aggregate_data(df, freq=freq)
    
    filename = plot_environment_chart(df)
    if filename is None:
        raise HTTPException(status_code=400, detail="No se pudo generar el gráfico (faltan columnas de ambiente).")
    return {"image_url": f"/images/{filename}"}

@app.get("/graph/correlation")
def graph_correlation(
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    freq: Optional[str] = None
):
    """
    Genera el gráfico de correlación entre la potencia AC y la irradiancia.
    """
    elec = cleaner.electrical_data.copy()
    irr = cleaner.irradiance_data.copy()
    
    if start_date or end_date:
        elec = cleaner.filter_by_date(elec, start_date, end_date)
        irr = cleaner.filter_by_date(irr, start_date, end_date)
    if freq:
        elec = cleaner.aggregate_data(elec, freq=freq)
        irr = cleaner.aggregate_data(irr, freq=freq)
    
    filename = plot_correlation_chart(elec, irr)
    if filename is None:
        raise HTTPException(status_code=400, detail="No se pudo generar el gráfico de correlación.")
    return {"image_url": f"/images/{filename}"}

@app.get("/forecast")
def forecast_production(
    start_date: Optional[str] = None,
    end_date: Optional[str] = None
):
    """
    Realiza una predicción simple de producción basándose en los datos
    de irradiancia vs potencia AC.
    """
    elec = cleaner.electrical_data.copy()
    irr = cleaner.irradiance_data.copy()
    if start_date or end_date:
        elec = cleaner.filter_by_date(elec, start_date, end_date)
        irr = cleaner.filter_by_date(irr, start_date, end_date)
    
    result = predecir_produccion(irr, elec)
    if result is None:
        raise HTTPException(status_code=400, detail="No se pudo crear el modelo predictivo (datos insuficientes).")
    
    pred_df, daily_energy = result
    return {
        "forecasted_hourly_data": pred_df.to_dict(orient='records'),
        "daily_energy_forecast_kwh": daily_energy
    }

@app.get("/summary")
def summary(capacidad_instalada: float = 100.0):
    """
    Genera y muestra el resumen ejecutivo en texto, y retorna en JSON también.
    """
    resumen = generar_resumen_ejecutivo(cleaner, capacidad_instalada)
    return resumen

@app.get("/inverters/status")
def get_inverters_status():
    """
    Devuelve el estado de los inversores basado en los datos eléctricos más recientes
    """
    elec = cleaner.electrical_data.copy()
    
    # Obtener las últimas mediciones
    elec.sort_values('measured_on', inplace=True)
    latest_data = elec.iloc[-1:].copy() if not elec.empty else None
    
    if latest_data is None or latest_data.empty:
        return []
        
    # Identificar columnas de inversores
    ac_power_cols = [col for col in elec.columns if 'ac_power' in col]
    dc_voltage_cols = [col for col in elec.columns if 'dc_voltage' in col]
    dc_current_cols = [col for col in elec.columns if 'dc_current' in col]
    
    inverters = []
    for col in ac_power_cols:
        inv_num = col.split('_')[1]
        
        # Buscar columnas relacionadas para este inversor
        vdc_col = next((c for c in dc_voltage_cols if f'_{inv_num}_' in c), None)
        idc_col = next((c for c in dc_current_cols if f'_{inv_num}_' in c), None)
        
        power = float(latest_data[col].iloc[0]) if col in latest_data else 0
        voltage = float(latest_data[vdc_col].iloc[0]) if vdc_col and vdc_col in latest_data else 0
        current = float(latest_data[idc_col].iloc[0]) if idc_col and idc_col in latest_data else 0
        
        # Calcular eficiencia si es posible
        dc_power = voltage * current
        efficiency = round((power / dc_power * 100) if dc_power > 0 else 0, 2)
        
        # Determinar estado
        status = "Activo" if power > 0.1 else "Inactivo"
        
        inverters.append({
            "name": f"Inversor {inv_num}",
            "voltage_dc": round(voltage, 1),
            "current_dc": round(current, 2),
            "ac_power": round(power, 2),
            "efficiency": efficiency,
            "status": status
        })
    
    return inverters

##############################################
# Para correr: uvicorn api:app --reload
##############################################