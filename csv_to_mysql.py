import pandas as pd
import mysql.connector
from mysql.connector import errorcode

# Configuración de conexión
config = {
    'user': 'admin',
    'password': '3CB0onBwj6dKk5ibLc8l',
    'host': 'database-monitor-solar.c388c6akq1gk.us-east-2.rds.amazonaws.com',
    'database': 'solar_data'
}

# Archivos CSV
csv_files = {
    'electrical_data': 'data/2107_electrical_data_cleaned.csv',
    'environment_data': 'data/2107_environment_data_cleaned.csv',
    'irradiance_data': 'data/2107_irradiance_data_cleaned.csv'
}

# Conexión
try:
    conn = mysql.connector.connect(**config)
    cursor = conn.cursor()
    print("✅ Conexión exitosa a MySQL")

    for table_name, file_path in csv_files.items():
        print(f"\n▶ Procesando: {table_name}")

        # Cargar CSV
        df = pd.read_csv(file_path)
        df.columns = [col.replace(' ', '_').lower() for col in df.columns]

        # Crear tabla si no existe
        columns_sql = []
        for col in df.columns:
            if col == 'measured_on':
                columns_sql.append(f"`{col}` DATETIME")
            else:
                columns_sql.append(f"`{col}` FLOAT")
        create_sql = f"""
        CREATE TABLE IF NOT EXISTS {table_name} (
            {','.join(columns_sql)}
        );
        """
        cursor.execute(f"DROP TABLE IF EXISTS {table_name}")
        cursor.execute(create_sql)
        print("✅ Tabla creada")

        # Insertar filas
        for _, row in df.iterrows():
            placeholders = ','.join(['%s'] * len(row))
            insert_sql = f"INSERT INTO {table_name} VALUES ({placeholders})"
            cursor.execute(insert_sql, tuple(row.values))
        conn.commit()
        print(f"✅ {len(df)} registros insertados en {table_name}")

except mysql.connector.Error as err:
    if err.errno == errorcode.ER_ACCESS_DENIED_ERROR:
        print("❌ Usuario o contraseña incorrectos")
    elif err.errno == errorcode.ER_BAD_DB_ERROR:
        print("❌ Base de datos no existe")
    else:
        print(err)
finally:
    if conn.is_connected():
        cursor.close()
        conn.close()
        print("🔒 Conexión cerrada")
