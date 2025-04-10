import os
import json
import logging
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
from data_cleaner import DataCleaner

# Parámetros globales para filtrado y agregación:
START_DATE = '2022-01-01'
END_DATE = '2022-12-31'
RESAMPLE_FREQ = 'D'  # Por ejemplo, datos agregados cada hora

##########################################
# 1. Funciones de Análisis de Rendimiento
##########################################

def calcular_performance_ratio(electrical_data, irradiance_data, capacidad_instalada_kw):
    """
    Calcula el Performance Ratio (PR) = (Potencia AC / Capacidad Instalada) / (Irradiancia / 1000)
    """
    # Asegurar que 'measured_on' sea datetime
    electrical_data['measured_on'] = pd.to_datetime(electrical_data['measured_on'])
    irradiance_data['measured_on'] = pd.to_datetime(irradiance_data['measured_on'])
    
    # Merge de ambos datasets en función de measured_on
    merged = pd.merge(electrical_data, irradiance_data, on='measured_on', how='inner')
    if merged.empty:
        print("No hay datos coincidentes para calcular PR")
        return None
    
    ac_power_cols = [col for col in merged.columns if 'ac_power' in col]
    irr_cols = [col for col in merged.columns if 'irradiance' in col or 'poa_irradiance' in col]
    
    if not ac_power_cols or not irr_cols:
        print("Faltan columnas necesarias para calcular PR")
        return None
    
    merged['total_ac_power'] = merged[ac_power_cols].sum(axis=1)
    irr_col = irr_cols[0]
    day_data = merged[merged[irr_col] > 50]  # considerar solo horas de luz
    
    if day_data.empty:
        print("No hay suficientes datos diurnos para calcular PR")
        return None
    
    # Cálculo del PR
    day_data['pr'] = (day_data['total_ac_power'] / capacidad_instalada_kw) / (day_data[irr_col] / 1000)
    day_data['pr'] = day_data['pr'].clip(0, 1.2)
    
    pr_promedio = day_data['pr'].mean()
    return pr_promedio, day_data

def calcular_rendimiento_especifico(electrical_data, capacidad_instalada_kw):
    """
    Calcula el rendimiento específico (kWh/kWp), es decir,
    la energía producida por cada kW de capacidad instalada.
    """
    ac_power_cols = [col for col in electrical_data.columns if 'ac_power' in col]
    if not ac_power_cols:
        print("No se encontraron columnas de potencia AC")
        return None
    
    electrical_data['total_ac_power'] = electrical_data[ac_power_cols].sum(axis=1)
    electrical_data.sort_values('measured_on', inplace=True)
    electrical_data['date'] = electrical_data['measured_on'].dt.date
    daily_energy = electrical_data.groupby('date')['total_ac_power'].mean() * 24  # Aproximadamente kWh diarios
    daily_specific_yield = daily_energy / capacidad_instalada_kw
    return daily_specific_yield

##########################################
# 2. Detección de Anomalías y Alertas
##########################################

def detectar_anomalias(electrical_data, environment_data, irradiance_data, umbral_potencia=0.7):
    """
    Detecta anomalías comparando la producción real con la esperada en función de la irradiancia.
    Genera alertas si la producción es baja respecto a lo esperado o si el inversor está inactivo.
    """
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
                    alertas.append(f"ALERTA: Inversor {inv_num} bajo rendimiento el {row['measured_on']}. " +
                                   f"Potencia: {row[col]:.2f} kW, Irradiancia: {row[irr_col]:.2f} W/m², " +
                                   f"Ratio: {row[f'ratio_{inv_num}']:.5f} (Normal: {ratio_promedio:.5f})")
    
    high_irr_data = day_data[day_data[irr_col] > 300]
    for col in ac_power_cols:
        inv_num = col.split('_')[1]
        inactivos = high_irr_data[high_irr_data[col] < 0.1]
        if not inactivos.empty:
            for idx, row in inactivos.iterrows():
                alertas.append(f"ALERTA: Inversor {inv_num} inactivo en alta irradiancia ({row[irr_col]:.2f} W/m²) el {row['measured_on']}")
    
    return alertas

##########################################
# 3. Estadísticas Avanzadas de Energía
##########################################

def calcular_estadisticas_energia(electrical_data):
    """
    Calcula estadísticas de energía, agrupadas por día y mes, y extrae
    promedios, picos y patrones de producción.
    """
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

def plot_energy_heatmap(electrical_data, output_dir='static/images'):
    """
    Crea mapas de calor que muestran la producción de energía:
      - Un heatmap diario (día vs hora).
      - Un heatmap mensual (mes vs hora).
    """
    os.makedirs(output_dir, exist_ok=True)
    df = electrical_data.copy()
    df['measured_on'] = pd.to_datetime(df['measured_on'])
    ac_power_cols = [col for col in df.columns if 'ac_power' in col]
    if not ac_power_cols:
        print("No se encontraron columnas de potencia AC.")
        return None
    df['total_ac_power'] = df[ac_power_cols].sum(axis=1)
    df['hour'] = df['measured_on'].dt.hour
    df['day'] = df['measured_on'].dt.day
    df['month'] = df['measured_on'].dt.month

    # Mapa de calor diario: día vs hora
    pivot_day = df.pivot_table(index='day', columns='hour', values='total_ac_power', aggfunc='mean')
    plt.figure(figsize=(12,8))
    sns.heatmap(pivot_day, cmap='YlOrRd', cbar_kws={'label': 'AC Power (kW)'})
    plt.title('Patrón Diario de Producción', fontsize=16)
    plt.xlabel('Hora del día', fontsize=12)
    plt.ylabel('Día del mes', fontsize=12)
    output_file_day = os.path.join(output_dir, 'produccion_diaria_heatmap.png')
    plt.savefig(output_file_day, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Heatmap diario guardado en {output_file_day}")

    # Mapa de calor mensual: mes vs hora
    pivot_month = df.pivot_table(index='month', columns='hour', values='total_ac_power', aggfunc='mean')
    plt.figure(figsize=(12,6))
    sns.heatmap(pivot_month, cmap='YlOrRd', cbar_kws={'label': 'AC Power (kW)'})
    plt.title('Patrón Mensual de Producción', fontsize=16)
    plt.xlabel('Hora del día', fontsize=12)
    plt.ylabel('Mes', fontsize=12)
    output_file_month = os.path.join(output_dir, 'produccion_mensual_heatmap.png')
    plt.savefig(output_file_month, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Heatmap mensual guardado en {output_file_month}")
    return output_file_day, output_file_month

##########################################
# 4. Forecasting Simple
##########################################

def predecir_produccion(irradiance_data, electrical_data, dias_predecir=1):
    """
    Realiza una predicción simple de la producción utilizando una regresión lineal
    sobre la irradiancia y la potencia AC total.
    """
    from sklearn.linear_model import LinearRegression
    
    irradiance_data['measured_on'] = pd.to_datetime(irradiance_data['measured_on'])
    electrical_data['measured_on'] = pd.to_datetime(electrical_data['measured_on'])
    merged = pd.merge(electrical_data, irradiance_data, on='measured_on', how='inner')
    if merged.empty:
        print("No hay datos coincidentes para la predicción")
        return None
    
    ac_power_cols = [col for col in merged.columns if 'ac_power' in col]
    irr_cols = [col for col in merged.columns if 'irradiance' in col or 'poa_irradiance' in col]
    if not ac_power_cols or not irr_cols:
        print("Faltan columnas necesarias para la predicción")
        return None
    
    merged['total_ac_power'] = merged[ac_power_cols].sum(axis=1)
    irr_col = irr_cols[0]
    valid_data = merged[(merged[irr_col] > 50) & (merged['total_ac_power'] > 0)].copy()
    if valid_data.shape[0] < 24:
        print("Datos insuficientes para crear un modelo predictivo")
        return None

    X = valid_data[[irr_col]].values
    y = valid_data['total_ac_power'].values
    model = LinearRegression()
    model.fit(X, y)
    
    valid_data['hour'] = valid_data['measured_on'].dt.hour
    hourly_irradiance = valid_data.groupby('hour')[irr_col].mean()
    prediction = []
    for hour, irr in hourly_irradiance.items():
        if irr > 50:
            pred_power = model.predict([[irr]])[0]
            prediction.append({'hour': hour, 'irradiance': irr, 'predicted_power': pred_power})
    prediction_df = pd.DataFrame(prediction)
    daily_energy = prediction_df['predicted_power'].sum()  # Aproximadamente kWh diarios
    return prediction_df, daily_energy

##########################################
# 5. Dashboard y Reporte Ejecutivo
##########################################

def generar_resumen_ejecutivo(cleaner, capacidad_instalada_kw=100):
    """
    Genera un resumen ejecutivo a partir de los datos limpios.
    Se calculan Performance Ratio, rendimiento específico, estadísticas de energía y se detectan alertas.
    """
    elec_data = cleaner.electrical_data.copy()
    env_data = cleaner.environment_data.copy()
    irr_data = cleaner.irradiance_data.copy()
    
    # Calcular Performance Ratio
    pr_result = calcular_performance_ratio(elec_data, irr_data, capacidad_instalada_kw)
    pr_valor = pr_result[0] if pr_result is not None else "N/A"
    
    # Calcular rendimiento específico
    specific_yield = calcular_rendimiento_especifico(elec_data, capacidad_instalada_kw)
    yield_promedio = specific_yield.mean() if specific_yield is not None else "N/A"
    
    # Estadísticas de energía
    energy_stats = calcular_estadisticas_energia(elec_data)
    if energy_stats:
        total_energy = energy_stats['daily_energy'].sum()
        best_day = energy_stats['daily_energy'].idxmax()
        best_day_energy = energy_stats['daily_energy'].max()
        peak_hour = energy_stats['peak_hour']
        peak_power = energy_stats['peak_power']
    else:
        total_energy = best_day = best_day_energy = peak_hour = peak_power = "N/A"
    
    # Detectar anomalías
    alertas = detectar_anomalias(elec_data, env_data, irr_data)
    
    # Imprimir resumen ejecutivo
    print("\n" + "="*50)
    print("RESUMEN EJECUTIVO - SISTEMA FOTOVOLTAICO")
    print("="*50)
    print(f"Fecha de análisis: {datetime.now().strftime('%Y-%m-%d')}")
    print(f"Período analizado: {elec_data['measured_on'].min().strftime('%Y-%m-%d')} a {elec_data['measured_on'].max().strftime('%Y-%m-%d')}")
    print(f"\nPerformance Ratio (PR): {pr_valor:.3f}" if isinstance(pr_valor, float) else f"\nPerformance Ratio (PR): {pr_valor}")
    print(f"Rendimiento Específico: {yield_promedio:.2f} kWh/kWp" if isinstance(yield_promedio, float) else f"Rendimiento Específico: {yield_promedio}")
    print(f"Energía Total Generada: {total_energy:.2f} kWh" if isinstance(total_energy, float) else f"Energía Total Generada: {total_energy}")
    print(f"Mejor día de producción: {best_day} ({best_day_energy:.2f} kWh)" if isinstance(best_day_energy, float) else f"Mejor día de producción: {best_day}")
    print(f"Hora pico de producción: {peak_hour}:00 ({peak_power:.2f} kW)" if isinstance(peak_hour, int) else f"Hora pico de producción: {peak_hour}")
    print("\nAlertas y Anomalías:")
    if not alertas:
        print("No se detectaron anomalías significativas.")
    else:
        for alerta in alertas:
            print(f"- {alerta}")
    
    return {
        'pr': pr_valor,
        'specific_yield': yield_promedio,
        'total_energy': total_energy,
        'best_day': best_day,
        'best_day_energy': best_day_energy,
        'peak_hour': peak_hour,
        'peak_power': peak_power,
        'alertas': alertas
    }

# Opcional: función para generar reporte PDF (requiere reportlab)
def generar_reporte_pdf(cleaner, output_file='reports/informe_sistema_solar.pdf'):
    import os
    from reportlab.lib.pagesizes import letter
    from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image, Table, TableStyle
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
    from reportlab.lib import colors
    
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    doc = SimpleDocTemplate(output_file, pagesize=letter)
    styles = getSampleStyleSheet()
    elements = []
    
    elements.append(Paragraph("Informe del Sistema Fotovoltaico", styles['Heading1']))
    elements.append(Spacer(1, 12))
    elements.append(Paragraph(f"Fecha de generación: {datetime.now().strftime('%Y-%m-%d %H:%M')}", styles['Normal']))
    elements.append(Spacer(1, 24))
    
    resumen = generar_resumen_ejecutivo(cleaner)
    
    data = [
        ["Métrica", "Valor"],
        ["Performance Ratio (PR)", f"{resumen['pr']:.3f}" if isinstance(resumen['pr'], float) else str(resumen['pr'])],
        ["Rendimiento Específico", f"{resumen['specific_yield']:.2f} kWh/kWp" if isinstance(resumen['specific_yield'], float) else str(resumen['specific_yield'])],
        ["Energía Total Generada", f"{resumen['total_energy']:.2f} kWh" if isinstance(resumen['total_energy'], float) else str(resumen['total_energy'])],
        ["Mejor día de producción", f"{resumen['best_day']} ({resumen['best_day_energy']:.2f} kWh)" if isinstance(resumen['best_day_energy'], float) else str(resumen['best_day'])],
        ["Hora pico", f"{resumen['peak_hour']}:00 ({resumen['peak_power']:.2f} kW)" if isinstance(resumen['peak_hour'], int) else str(resumen['peak_hour'])]
    ]
    t = Table(data, colWidths=[250, 250])
    t.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (1,0), colors.grey),
        ('TEXTCOLOR', (0,0), (1,0), colors.whitesmoke),
        ('ALIGN', (0,0), (-1,0), 'CENTER'),
        ('FONTNAME', (0,0), (-1,0), 'Helvetica-Bold'),
        ('BOTTOMPADDING', (0,0), (-1,0), 12),
        ('BACKGROUND', (0,1), (-1,-1), colors.beige),
        ('GRID', (0,0), (-1,-1), 1, colors.black)
    ]))
    elements.append(t)
    elements.append(Spacer(1, 24))
    
    # Incluir algún gráfico (por ejemplo, el de potencia AC)
    power_chart = plot_power_chart(cleaner.electrical_data, output_dir='static/images', freq=None)
    if power_chart and os.path.exists(power_chart):
        elements.append(Paragraph("Potencia AC por Inversor", styles['Heading3']))
        elements.append(Spacer(1, 12))
        elements.append(Image(power_chart, width=450, height=250))
    
    doc.build(elements)
    print(f"Reporte PDF generado en: {output_file}")
    return output_file

##########################################
# 6. Función Principal (main)
##########################################

def main():
    try:
        start_time = datetime.now()
        cleaner = DataCleaner()
        print("Initializing dashboard...")
        
        # Cargar y limpiar datos
        cleaner.load_data()
        elec_clean = cleaner.clean_electrical_data()
        env_clean = cleaner.clean_environment_data()
        irr_clean = cleaner.clean_irradiance_data()
        
        print(f"Dataset eléctrico limpio: {elec_clean.shape[0]} registros, {len(elec_clean.columns)} columnas")
        print(f"Dataset de ambiente limpio: {env_clean.shape[0]} registros, {len(env_clean.columns)} columnas")
        print(f"Dataset de irradiancia limpio: {irr_clean.shape[0]} registros, {len(irr_clean.columns)} columnas")
        
        # Filtrar datos por rango de fechas
        elec_filtered = cleaner.filter_by_date(elec_clean, START_DATE, END_DATE)
        env_filtered = cleaner.filter_by_date(env_clean, START_DATE, END_DATE)
        irr_filtered = cleaner.filter_by_date(irr_clean, START_DATE, END_DATE)
        
        # Agregar (resamplear) los datos a una granularidad deseada
        elec_agg = cleaner.aggregate_data(elec_filtered, freq=RESAMPLE_FREQ)
        env_agg = cleaner.aggregate_data(env_filtered, freq=RESAMPLE_FREQ)
        irr_agg = cleaner.aggregate_data(irr_filtered, freq=RESAMPLE_FREQ)
        
        # Generar gráficos y análisis avanzados
        plot_power_chart(elec_agg, freq=None)  # Datos ya agregados
        plot_environment_chart(env_agg, freq=None)
        plot_correlation_chart(elec_agg, irr_agg, freq=None)
        plot_correlation_matrix(elec_agg, title="Matriz de Correlación - Datos Eléctricos", 
                                  output_file="static/images/correlation_matrix.png")
        plot_energy_heatmap(elec_clean, output_dir="static/images")
        
        # Calcular métricas avanzadas
        capacidad_instalada_kw = 100  # Ajusta según tu sistema
        pr_result = calcular_performance_ratio(elec_clean, irr_clean, capacidad_instalada_kw)
        if pr_result:
            pr_promedio = pr_result[0]
        else:
            pr_promedio = None
        
        specific_yield = calcular_rendimiento_especifico(elec_clean, capacidad_instalada_kw)
        energy_stats = calcular_estadisticas_energia(elec_clean)
        prediction, daily_energy_forecast = predecir_produccion(irr_clean, elec_clean)
        
        resumen = generar_resumen_ejecutivo(cleaner, capacidad_instalada_kw)
        
        # Opcionalmente, generar reporte PDF (requiere reportlab instalado)
        # generar_reporte_pdf(cleaner, output_file='reports/informe_sistema_solar.pdf')
        
        energia_j, energia_kwh = cleaner.integrate_irradiance()
        print(f"\nEnergía total recibida (irradiancia): {energia_j:.2f} J/m² o {energia_kwh:.4f} kWh/m²")
        
        execution_time = (datetime.now() - start_time).total_seconds()
        print(f"\nTotal execution time: {execution_time:.2f} seconds")
    except Exception as e:
        logging.error(f"Execution error: {e}")
        print(f"Error: {e}")

if __name__ == "__main__":
    main()
