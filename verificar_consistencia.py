import pandas as pd
import numpy as np
import glob
import os

# Cargar archivo más reciente
upload_files = glob.glob("uploads/*.xlsx")
file_path = max(upload_files, key=os.path.getctime)

print("="*80)
print("VERIFICACIÓN DE CONSISTENCIA DE DIAGNÓSTICOS")
print("="*80)
print(f"\nArchivo: {file_path}\n")

df = pd.read_excel(file_path)

# Filtrar solo diagnósticos válidos
df = df[df['Diagnostico'].isin(['Normal', 'Hipertension', 'Diabetes'])].copy()

# Convertir a numérico
df['Glucosa'] = pd.to_numeric(df['Glucosa'], errors='coerce')
df['Presion_Arterial_str'] = df['Presion_Arterial'].astype(str)

# Extraer presión sistólica
def get_sistolica(pa):
    try:
        if '/' in pa:
            return float(pa.split('/')[0])
        return float(pa)
    except:
        return np.nan

df['Presion_Sistolica'] = df['Presion_Arterial_str'].apply(get_sistolica)

print("="*80)
print("ANÁLISIS DE CONSISTENCIA CLÍNICA")
print("="*80)

# Criterios médicos reales
print("\nCriterios medicos estandar:")
print("  - Diabetes: Glucosa > 126 mg/dL (en ayunas)")
print("  - Hipertension: Presion Sistolica > 140 mmHg")
print("  - Normal: Glucosa <= 126 Y Presion <= 140")

print("\n" + "="*80)
print("INCONSISTENCIAS DETECTADAS")
print("="*80)

inconsistencias = {
    'diabetes_mal_etiquetados': 0,
    'hipertension_mal_etiquetados': 0,
    'normal_mal_etiquetados': 0
}

# Verificar DIABETES
print("\n1. CASOS ETIQUETADOS COMO 'Diabetes':")
diabetes_df = df[df['Diagnostico'] == 'Diabetes']
print(f"   Total: {len(diabetes_df)} casos")

# ¿Cuántos REALMENTE tienen glucosa alta?
diabetes_con_glucosa_alta = diabetes_df[diabetes_df['Glucosa'] > 126]
diabetes_con_glucosa_normal = diabetes_df[diabetes_df['Glucosa'] <= 126]

print(f"   - Con glucosa > 126 (correcto): {len(diabetes_con_glucosa_alta)} ({len(diabetes_con_glucosa_alta)/len(diabetes_df)*100:.1f}%)")
print(f"   - Con glucosa <= 126 (INCORRECTO): {len(diabetes_con_glucosa_normal)} ({len(diabetes_con_glucosa_normal)/len(diabetes_df)*100:.1f}%)")

inconsistencias['diabetes_mal_etiquetados'] = len(diabetes_con_glucosa_normal)

if len(diabetes_con_glucosa_normal) > 0:
    print(f"\n   Muestra de casos mal etiquetados como Diabetes:")
    sample = diabetes_con_glucosa_normal[['ID', 'Glucosa', 'Presion_Arterial', 'Diagnostico']].head(5)
    print(sample.to_string(index=False))

# Verificar HIPERTENSIÓN
print("\n2. CASOS ETIQUETADOS COMO 'Hipertension':")
hta_df = df[df['Diagnostico'] == 'Hipertension']
print(f"   Total: {len(hta_df)} casos")

hta_con_presion_alta = hta_df[hta_df['Presion_Sistolica'] > 140]
hta_con_presion_normal = hta_df[hta_df['Presion_Sistolica'] <= 140]

print(f"   - Con presion > 140 (correcto): {len(hta_con_presion_alta)} ({len(hta_con_presion_alta)/len(hta_df)*100:.1f}%)")
print(f"   - Con presion <= 140 (INCORRECTO): {len(hta_con_presion_normal)} ({len(hta_con_presion_normal)/len(hta_df)*100:.1f}%)")

inconsistencias['hipertension_mal_etiquetados'] = len(hta_con_presion_normal)

if len(hta_con_presion_normal) > 0:
    print(f"\n   Muestra de casos mal etiquetados como Hipertension:")
    sample = hta_con_presion_normal[['ID', 'Glucosa', 'Presion_Arterial', 'Diagnostico']].head(5)
    print(sample.to_string(index=False))

# Verificar NORMAL
print("\n3. CASOS ETIQUETADOS COMO 'Normal':")
normal_df = df[df['Diagnostico'] == 'Normal']
print(f"   Total: {len(normal_df)} casos")

# Deberían tener glucosa Y presión normales
normal_realmente_normal = normal_df[(normal_df['Glucosa'] <= 126) & (normal_df['Presion_Sistolica'] <= 140)]
normal_con_problemas = normal_df[(normal_df['Glucosa'] > 126) | (normal_df['Presion_Sistolica'] > 140)]

print(f"   - Sin factores de riesgo (correcto): {len(normal_realmente_normal)} ({len(normal_realmente_normal)/len(normal_df)*100:.1f}%)")
print(f"   - Con glucosa o presion alta (INCORRECTO): {len(normal_con_problemas)} ({len(normal_con_problemas)/len(normal_df)*100:.1f}%)")

inconsistencias['normal_mal_etiquetados'] = len(normal_con_problemas)

if len(normal_con_problemas) > 0:
    print(f"\n   Muestra de casos mal etiquetados como Normal:")
    sample = normal_con_problemas[['ID', 'Glucosa', 'Presion_Arterial', 'Diagnostico']].head(5)
    print(sample.to_string(index=False))

# RESUMEN
print("\n" + "="*80)
print("RESUMEN DE INCONSISTENCIAS")
print("="*80)

total_inconsistencias = sum(inconsistencias.values())
porcentaje_inconsistencias = (total_inconsistencias / len(df)) * 100

print(f"\nTotal de casos analizados: {len(df):,}")
print(f"Casos con etiquetas inconsistentes: {total_inconsistencias:,} ({porcentaje_inconsistencias:.1f}%)")
print(f"Casos con etiquetas correctas: {len(df) - total_inconsistencias:,} ({100-porcentaje_inconsistencias:.1f}%)")

print(f"\nDesglose:")
print(f"  - Diabetes mal etiquetados: {inconsistencias['diabetes_mal_etiquetados']:,}")
print(f"  - Hipertensión mal etiquetados: {inconsistencias['hipertension_mal_etiquetados']:,}")
print(f"  - Normal mal etiquetados: {inconsistencias['normal_mal_etiquetados']:,}")

print("\n" + "="*80)
print("CONCLUSIÓN")
print("="*80)

if porcentaje_inconsistencias > 40:
    print(f"""
*** PROBLEMA CRITICO DETECTADO ***

El {porcentaje_inconsistencias:.1f}% de tus datos tienen etiquetas INCORRECTAS.

EXPLICACION:
Las etiquetas de diagnostico en tu Excel NO coinciden con los valores clinicos reales.

Por ejemplo:
- Pacientes etiquetados como "Diabetes" pero con glucosa normal (<126)
- Pacientes etiquetados como "Normal" pero con presion alta (>140)

CONSECUENCIA:
El modelo aprende patrones FALSOS. Por eso tiene solo 41% de precision.
Es como ensenarle a un estudiante con un libro de respuestas equivocado.

SOLUCIONES:

1. RE-ETIQUETAR AUTOMATICAMENTE (basado en criterios clinicos):
   - Crear script que corrija las etiquetas segun glucosa y presion
   - Precision esperada despues: 80-90%

2. USAR ETIQUETAS CLINICAS EN LUGAR DE LAS DEL EXCEL:
   - Ignorar columna "Diagnostico" del Excel
   - Generar diagnosticos basados en valores numericos
   - Precision esperada: 85-95%

3. REVISAR Y CORREGIR DATOS MANUALMENTE:
   - Identificar casos inconsistentes
   - Corregir etiquetas en el Excel
   - Volver a entrenar
""")
elif porcentaje_inconsistencias > 20:
    print(f"""
*** INCONSISTENCIAS MODERADAS ***

El {porcentaje_inconsistencias:.1f}% de tus datos tienen etiquetas sospechosas.

Esto reduce la precision del modelo a ~41%.

SOLUCION: Re-etiquetar casos inconsistentes.
""")
else:
    print(f"""
OK - Los datos parecen tener etiquetas correctas ({100-porcentaje_inconsistencias:.1f}% consistentes).

Si aun tienes 41% de precision, el problema puede ser:
1. Relaciones no lineales complejas entre variables
2. Necesitas mas features
3. Necesitas mas datos
""")

print("="*80)
