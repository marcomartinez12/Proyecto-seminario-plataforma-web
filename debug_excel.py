import pandas as pd
import os

# Ruta del archivo con error
file_path = "uploads/1eacdc8c-1a31-46e6-afaa-547295775d2a.xls"

print("=== DIAGNÓSTICO DEL ARCHIVO EXCEL ===")
print(f"Archivo: {file_path}")
print(f"Existe: {os.path.exists(file_path)}")

try:
    # Leer el archivo Excel
    df = pd.read_excel(file_path)
    
    print(f"\n=== INFORMACIÓN BÁSICA ===")
    print(f"Filas: {len(df)}")
    print(f"Columnas: {len(df.columns)}")
    print(f"Columnas encontradas: {list(df.columns)}")
    
    print(f"\n=== PRIMERAS 5 FILAS ===")
    print(df.head())
    
    print(f"\n=== TIPOS DE DATOS ===")
    print(df.dtypes)
    
    print(f"\n=== VALORES NULOS ===")
    print(df.isnull().sum())
    
    # Verificar columnas requeridas
    required_columns = ['ID', 'Edad', 'Sexo', 'Peso', 'Altura', 'Presion_Arterial', 'Glucosa', 'Colesterol', 'Fumador', 'Diagnostico']
    missing_columns = [col for col in required_columns if col not in df.columns]
    
    print(f"\n=== VALIDACIÓN DE COLUMNAS ===")
    print(f"Columnas requeridas: {required_columns}")
    print(f"Columnas faltantes: {missing_columns}")
    
    if not missing_columns:
        print("\n=== MUESTRA DE DATOS POR COLUMNA ===")
        for col in required_columns:
            print(f"{col}: {df[col].iloc[0] if len(df) > 0 else 'N/A'} (tipo: {df[col].dtype})")
    
except Exception as e:
    print(f"\nERROR al leer el archivo: {str(e)}")
    print(f"Tipo de error: {type(e).__name__}")