import pandas as pd
import numpy as np

# Archivo a analizar (usar el más reciente en uploads)
import os
import glob

upload_files = glob.glob("uploads/*.xlsx")
if upload_files:
    file_path = max(upload_files, key=os.path.getctime)  # Más reciente
    print(f"Analizando: {file_path}\n")
else:
    print("ERROR: No se encontraron archivos Excel en uploads/")
    exit(1)

print("="*80)
print("DIAGNOSTICO COMPLETO DEL DATASET")
print("="*80)

# Cargar datos
df = pd.read_excel(file_path)

print(f"\n1. INFORMACIÓN GENERAL")
print(f"   Total de registros: {len(df):,}")
print(f"   Columnas: {list(df.columns)}")

# Análisis de diagnósticos
print(f"\n2. DISTRIBUCIÓN DE DIAGNÓSTICOS (TODOS)")
print("-"*80)
diag_counts = df['Diagnostico'].value_counts()
for diag, count in diag_counts.items():
    pct = (count / len(df)) * 100
    print(f"   {diag:30} {count:>6,} ({pct:>5.2f}%)")

print(f"\n   Total de diagnósticos únicos: {len(diag_counts)}")

# Filtrar diagnósticos válidos
diagnosticos_validos = [
    'Normal', 'Hipertension', 'Hipertensión', 'hipertension',
    'Diabetes', 'diabetes',
    'Prediabetes', 'prediabetes',
    'Obesidad', 'obesidad'
]

# Normalizar
df_copy = df.copy()
df_copy['Diagnostico'] = df_copy['Diagnostico'].str.strip()
df_copy['Diagnostico'] = df_copy['Diagnostico'].replace({
    'Hipertensión': 'Hipertension',
    'hipertension': 'Hipertension',
    'diabetes': 'Diabetes'
})

df_validos = df_copy[df_copy['Diagnostico'].isin(diagnosticos_validos)]

print(f"\n3. DESPUÉS DE FILTRAR DIAGNÓSTICOS VÁLIDOS")
print("-"*80)
print(f"   Registros válidos: {len(df_validos):,} ({len(df_validos)/len(df)*100:.1f}% del total)")
print(f"   Registros descartados: {len(df) - len(df_validos):,} ({(len(df) - len(df_validos))/len(df)*100:.1f}%)")

if len(df_validos) > 0:
    print(f"\n   Distribución de diagnósticos válidos:")
    valid_counts = df_validos['Diagnostico'].value_counts()
    for diag, count in valid_counts.items():
        pct = (count / len(df_validos)) * 100
        print(f"   {diag:30} {count:>6,} ({pct:>5.2f}%)")

    # Balance
    max_class = valid_counts.max()
    min_class = valid_counts.min()
    ratio = max_class / min_class
    print(f"\n   Ratio de desbalance: {ratio:.2f}:1")

    if ratio > 10:
        print(f"   ⚠️  DESBALANCE EXTREMO - Clase mayoritaria tiene {ratio:.0f}x más casos")
    elif ratio > 5:
        print(f"   ⚠️  DESBALANCE ALTO - Necesita SMOTE")
    elif ratio > 2:
        print(f"   ⚠️  Desbalance moderado - SMOTE recomendado")
    else:
        print(f"   ✓ Balance aceptable")

# Análisis de calidad de datos
print(f"\n4. CALIDAD DE DATOS")
print("-"*80)

# Valores nulos
print(f"\n   Valores nulos por columna:")
for col in df.columns:
    nulls = df[col].isnull().sum()
    if nulls > 0:
        print(f"   {col:30} {nulls:>6,} ({nulls/len(df)*100:.2f}%)")

# Valores únicos en columnas categóricas
print(f"\n   Valores únicos en columnas categóricas:")
print(f"   Sexo: {df['Sexo'].unique()}")
print(f"   Fumador: {df['Fumador'].unique()}")

# Estadísticas de columnas numéricas
print(f"\n5. ESTADÍSTICAS DE VARIABLES NUMÉRICAS")
print("-"*80)

numeric_cols = ['Edad', 'Peso', 'Altura', 'Glucosa', 'Colesterol']

for col in numeric_cols:
    if col in df.columns:
        # Convertir a numérico
        df_num = pd.to_numeric(df[col], errors='coerce')

        print(f"\n   {col}:")
        print(f"      Min:     {df_num.min():.2f}")
        print(f"      Max:     {df_num.max():.2f}")
        print(f"      Media:   {df_num.mean():.2f}")
        print(f"      Mediana: {df_num.median():.2f}")
        print(f"      Std Dev: {df_num.std():.2f}")

        # Detectar outliers extremos
        outliers_low = (df_num < 0).sum()

        if col == 'Edad':
            outliers_high = (df_num > 120).sum()
        elif col == 'Peso':
            outliers_high = (df_num > 300).sum()
        elif col == 'Altura':
            outliers_high = (df_num > 250).sum()
        elif col == 'Glucosa':
            outliers_high = (df_num > 600).sum()
        elif col == 'Colesterol':
            outliers_high = (df_num > 500).sum()
        else:
            outliers_high = 0

        total_outliers = outliers_low + outliers_high
        if total_outliers > 0:
            print(f"      ⚠️  Outliers extremos: {total_outliers:,} ({total_outliers/len(df)*100:.2f}%)")

# Análisis de presión arterial
print(f"\n6. ANÁLISIS DE PRESIÓN ARTERIAL")
print("-"*80)

presion_samples = df['Presion_Arterial'].head(20).tolist()
print(f"   Muestra de valores: {presion_samples[:10]}")

# Verificar formato
formatos_validos = 0
formatos_invalidos = 0

for val in df['Presion_Arterial']:
    if '/' in str(val):
        formatos_validos += 1
    else:
        formatos_invalidos += 1

print(f"   Formato válido (XXX/XX): {formatos_validos:,} ({formatos_validos/len(df)*100:.1f}%)")
print(f"   Formato inválido: {formatos_invalidos:,} ({formatos_invalidos/len(df)*100:.1f}%)")

# RESUMEN Y RECOMENDACIONES
print(f"\n")
print("="*80)
print("RESUMEN Y DIAGNÓSTICO")
print("="*80)

problemas = []

# Problema 1: Diagnósticos inválidos
if len(df_validos) < len(df) * 0.5:
    problemas.append(f"⚠️  CRÍTICO: {(len(df) - len(df_validos))/len(df)*100:.1f}% de datos tienen diagnósticos no válidos (Texto1-5)")

# Problema 2: Balance
if len(df_validos) > 0:
    if ratio > 5:
        problemas.append(f"⚠️  ALTO: Desbalance de clases extremo ({ratio:.1f}:1)")

# Problema 3: Datos insuficientes por clase
if len(df_validos) > 0:
    for diag, count in valid_counts.items():
        if count < 500:
            problemas.append(f"⚠️  MODERADO: Clase '{diag}' solo tiene {count} casos (recomendado: >500)")

# Problema 4: Outliers
total_numeric_outliers = 0
for col in numeric_cols:
    df_num = pd.to_numeric(df[col], errors='coerce')
    if col == 'Edad':
        total_numeric_outliers += ((df_num < 0) | (df_num > 120)).sum()
    elif col == 'Peso':
        total_numeric_outliers += ((df_num < 20) | (df_num > 300)).sum()
    elif col == 'Altura':
        total_numeric_outliers += ((df_num < 100) | (df_num > 250)).sum()

if total_numeric_outliers > len(df) * 0.05:
    problemas.append(f"⚠️  MODERADO: {total_numeric_outliers:,} outliers extremos ({total_numeric_outliers/len(df)*100:.1f}%)")

print(f"\nPROBLEMAS DETECTADOS: {len(problemas)}")
for i, problema in enumerate(problemas, 1):
    print(f"{i}. {problema}")

print(f"\n")
print("="*80)
print("CAUSA PROBABLE DE LA BAJA PRECISIÓN (41.06%)")
print("="*80)

if len(df_validos) < len(df) * 0.3:
    print(f"""
⚠️  CAUSA PRINCIPAL: DATOS INVÁLIDOS

El {(len(df) - len(df_validos))/len(df)*100:.1f}% de tus datos tienen diagnósticos como "Texto1", "Texto2", etc.
que NO son diagnósticos médicos reales.

El modelo está intentando aprender patrones de estos datos sin sentido,
lo que resulta en una precisión muy baja (41.06%).

SOLUCIÓN:
- El código actualizado YA filtra estos datos automáticamente
- Solo usará los {len(df_validos):,} registros válidos ({len(df_validos)/len(df)*100:.1f}%)
- Precisión esperada después del filtrado: 85-92%
""")
elif ratio > 5:
    print(f"""
⚠️  CAUSA PRINCIPAL: DESBALANCE EXTREMO

La clase mayoritaria tiene {ratio:.0f}x más casos que la minoritaria.
El modelo aprende a predecir siempre la clase mayoritaria.

SOLUCIÓN:
- El código actualizado YA aplica SMOTE automáticamente
- Balancea las clases durante el entrenamiento
- Precisión esperada: 80-90%
""")
else:
    print(f"""
✓ Los datos parecen estar en buen estado.

Si aún tienes 41.06% de precisión, verifica:
1. ¿Ejecutaste el servidor con el código actualizado?
2. ¿Ves en la consola el mensaje "VERSION MEJORADA"?
3. ¿Se está filtrando los diagnósticos correctamente?
""")

print("="*80)
