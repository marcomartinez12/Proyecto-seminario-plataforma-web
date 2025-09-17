import pandas as pd
import os

# Buscar el archivo más reciente
upload_dir = "uploads"
files = [f for f in os.listdir(upload_dir) if f.endswith(('.xlsx', '.xls'))]
if files:
    latest_file = max([os.path.join(upload_dir, f) for f in files], key=os.path.getctime)
    print(f"Analizando archivo: {latest_file}")
    
    try:
        df = pd.read_excel(latest_file)
        print(f"Total de registros: {len(df)}")
        print(f"Columnas: {list(df.columns)}")
        
        if 'Diagnostico' in df.columns:
            print(f"\n=== DIAGNÓSTICOS ÚNICOS (primeros 20) ===")
            diagnosticos_unicos = df['Diagnostico'].value_counts().head(20)
            print(diagnosticos_unicos)
            
            print(f"\n=== MUESTRA DE DIAGNÓSTICOS ===")
            print(df['Diagnostico'].head(10).tolist())
            
            print(f"\n=== BÚSQUEDA DE HIPERTENSIÓN ===")
            variaciones = ['Hipertensión', 'Hipertension', 'HTA', 'Presión alta', 'Hipertensiva', 'hipertension', 'hipertensión']
            total_hipertension = 0
            for var in variaciones:
                casos = len(df[df['Diagnostico'].str.contains(var, case=False, na=False)])
                if casos > 0:
                    print(f"'{var}': {casos} casos")
                    total_hipertension += casos
            print(f"Total hipertensión: {total_hipertension}")
                
            print(f"\n=== BÚSQUEDA DE DIABETES ===")
            variaciones_diabetes = ['Diabetes', 'Diabética', 'DM', 'Mellitus', 'diabetes', 'diabetica']
            total_diabetes = 0
            for var in variaciones_diabetes:
                casos = len(df[df['Diagnostico'].str.contains(var, case=False, na=False)])
                if casos > 0:
                    print(f"'{var}': {casos} casos")
                    total_diabetes += casos
            print(f"Total diabetes: {total_diabetes}")
            
            # Verificar valores nulos
            print(f"\n=== INFORMACIÓN ADICIONAL ===")
            print(f"Valores nulos en Diagnostico: {df['Diagnostico'].isnull().sum()}")
            print(f"Valores vacíos en Diagnostico: {(df['Diagnostico'] == '').sum()}")
            
        else:
            print("No se encontró la columna 'Diagnostico'")
            print(f"Columnas disponibles: {list(df.columns)}")
            
    except Exception as e:
        print(f"Error al leer el archivo: {e}")
else:
    print("No se encontraron archivos Excel en uploads/")