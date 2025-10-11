"""
GENERADOR DE DATASET SINTÉTICO MÉDICO
Genera datos clínicamente coherentes con etiquetas correctas
"""

import pandas as pd
import numpy as np
from datetime import datetime

def generar_dataset_medico(n_registros=10000, seed=42, nivel="COMPLETO"):
    """
    Genera dataset médico sintético con datos clínicamente coherentes.

    Parámetros:
    - n_registros: Número de pacientes a generar
    - seed: Semilla para reproducibilidad
    - nivel: "MINIMO", "ESTANDAR" o "COMPLETO"
    """

    np.random.seed(seed)
    print(f"\n{'='*70}")
    print(f"GENERADOR DE DATASET MÉDICO SINTÉTICO")
    print(f"{'='*70}")
    print(f"Generando {n_registros:,} registros...")
    print(f"Nivel de datos: {nivel}")

    # ===== DISTRIBUCIÓN DE DIAGNÓSTICOS =====
    # Distribución realista basada en epidemiología
    diagnosticos_dist = {
        'Normal': 0.45,          # 45% población sana
        'Prediabetes': 0.20,     # 20% prediabetes
        'Hipertension': 0.20,    # 20% hipertensión
        'Diabetes': 0.10,        # 10% diabetes
        'Obesidad': 0.05         # 5% obesidad mórbida
    }

    diagnosticos = np.random.choice(
        list(diagnosticos_dist.keys()),
        size=n_registros,
        p=list(diagnosticos_dist.values())
    )

    print(f"\nDistribución de diagnósticos:")
    for diag, pct in diagnosticos_dist.items():
        count = int(n_registros * pct)
        print(f"  - {diag}: {count:,} ({pct*100:.1f}%)")

    # ===== GENERAR DATOS DEMOGRÁFICOS =====
    print("\nGenerando datos demográficos...")

    # Edad: distribución realista (más común 30-60 años)
    edad_base = np.random.beta(2, 2, n_registros) * 70 + 18  # 18-88 años
    edad = np.round(edad_base).astype(int)

    # Sexo: 50/50
    sexo = np.random.choice(['M', 'F'], size=n_registros)

    # ===== GENERAR DATOS ANTROPOMÉTRICOS =====
    print("Generando medidas antropométricas...")

    # Peso base por sexo
    peso_base_m = np.random.normal(75, 15, n_registros)
    peso_base_f = np.random.normal(65, 12, n_registros)
    peso = np.where(sexo == 'M', peso_base_m, peso_base_f)

    # Altura base por sexo
    altura_base_m = np.random.normal(175, 8, n_registros)
    altura_base_f = np.random.normal(162, 7, n_registros)
    altura = np.where(sexo == 'M', altura_base_m, altura_base_f)

    # Ajustar peso según diagnóstico
    for i, diag in enumerate(diagnosticos):
        if diag == 'Obesidad':
            peso[i] = peso[i] * 1.4 + 20  # Aumentar peso significativamente
        elif diag == 'Diabetes':
            peso[i] = peso[i] * 1.2 + 10  # Diabetes tipo 2 asociada con sobrepeso

    # Asegurar valores razonables
    peso = np.clip(peso, 40, 200)
    altura = np.clip(altura, 140, 210)

    # Calcular IMC
    imc = peso / ((altura / 100) ** 2)

    # ===== GENERAR DATOS METABÓLICOS (CLÍNICAMENTE COHERENTES) =====
    print("Generando datos metabólicos coherentes...")

    # Glucosa base
    glucosa = np.random.normal(95, 10, n_registros)

    # Ajustar glucosa según diagnóstico (CRÍTICO para coherencia)
    for i, diag in enumerate(diagnosticos):
        if diag == 'Diabetes':
            # Diabetes: glucosa > 126
            glucosa[i] = np.random.uniform(130, 250)
        elif diag == 'Prediabetes':
            # Prediabetes: 100-125
            glucosa[i] = np.random.uniform(100, 125)
        elif diag == 'Normal':
            # Normal: < 100
            glucosa[i] = np.random.uniform(70, 99)
        elif diag == 'Obesidad':
            # Obesidad: tiende a hiperglucemia
            glucosa[i] = np.random.uniform(95, 130)
        elif diag == 'Hipertension':
            # Hipertensión: glucosa variable
            glucosa[i] = np.random.uniform(85, 110)

    # Añadir correlación con edad e IMC
    glucosa = glucosa + (edad - 40) * 0.3 + (imc - 25) * 0.8
    glucosa = np.clip(glucosa, 60, 400)

    # ===== GENERAR PRESIÓN ARTERIAL (CLÍNICAMENTE COHERENTE) =====
    print("Generando presión arterial coherente...")

    # Presión sistólica base
    presion_sistolica = np.random.normal(120, 10, n_registros)

    # Ajustar según diagnóstico (CRÍTICO)
    for i, diag in enumerate(diagnosticos):
        if diag == 'Hipertension':
            # Hipertensión: > 140
            presion_sistolica[i] = np.random.uniform(145, 180)
        elif diag == 'Normal':
            # Normal: < 120
            presion_sistolica[i] = np.random.uniform(90, 119)
        elif diag == 'Diabetes':
            # Diabetes: a menudo con presión elevada
            presion_sistolica[i] = np.random.uniform(120, 150)
        elif diag == 'Obesidad':
            # Obesidad: asociada con hipertensión
            presion_sistolica[i] = np.random.uniform(130, 160)

    # Correlación con edad e IMC
    presion_sistolica = presion_sistolica + (edad - 40) * 0.5 + (imc - 25) * 1.2
    presion_sistolica = np.clip(presion_sistolica, 80, 220)

    # Presión diastólica (aproximadamente 60% de sistólica)
    presion_diastolica = presion_sistolica * 0.6 + np.random.normal(0, 5, n_registros)
    presion_diastolica = np.clip(presion_diastolica, 50, 130)

    # ===== COLESTEROL =====
    print("Generando perfil lipídico...")

    colesterol_total = np.random.normal(190, 35, n_registros)
    colesterol_total = colesterol_total + (edad - 40) * 0.4 + (imc - 25) * 1.5
    colesterol_total = np.clip(colesterol_total, 120, 350)

    # ===== FUMADOR =====
    # Probabilidad de fumar aumenta con ciertas condiciones
    prob_fumar = 0.15  # 15% fumadores base
    fumador = np.random.random(n_registros) < prob_fumar
    fumador = fumador.astype(int)

    # ===== DATOS NIVEL 2 (ESTÁNDAR) =====
    if nivel in ["ESTANDAR", "COMPLETO"]:
        print("Generando datos NIVEL 2 (estándar)...")
        # Ya tenemos: Peso, Altura, Presion_Diastolica, Colesterol, Fumador ✓

    # ===== DATOS NIVEL 3 (COMPLETO) =====
    if nivel == "COMPLETO":
        print("Generando datos NIVEL 3 (completo)...")

        # Hemoglobina A1C (correlacionada con glucosa)
        hemoglobina_a1c = 4.5 + (glucosa - 70) * 0.02 + np.random.normal(0, 0.3, n_registros)
        hemoglobina_a1c = np.clip(hemoglobina_a1c, 4.0, 14.0)

        # Colesterol LDL (aproximadamente 60-70% del total)
        colesterol_ldl = colesterol_total * 0.65 + np.random.normal(0, 15, n_registros)
        colesterol_ldl = np.clip(colesterol_ldl, 50, 250)

        # Colesterol HDL (inversamente relacionado con IMC)
        colesterol_hdl = 55 - (imc - 25) * 0.8 + np.random.normal(0, 8, n_registros)
        colesterol_hdl = np.clip(colesterol_hdl, 20, 100)

        # Triglicéridos (correlacionados con glucosa e IMC)
        trigliceridos = 100 + (glucosa - 90) * 0.8 + (imc - 25) * 3 + np.random.normal(0, 30, n_registros)
        trigliceridos = np.clip(trigliceridos, 50, 400)

        # Circunferencia de cintura (correlacionada con IMC)
        circ_cintura_base_m = 85 + (imc - 25) * 2.5
        circ_cintura_base_f = 75 + (imc - 25) * 2.3
        circunferencia_cintura = np.where(
            sexo == 'M',
            circ_cintura_base_m + np.random.normal(0, 8, n_registros),
            circ_cintura_base_f + np.random.normal(0, 7, n_registros)
        )
        circunferencia_cintura = np.clip(circunferencia_cintura, 60, 150)

        # Circunferencia de cadera
        circ_cadera_base = 95 + (imc - 25) * 2
        circunferencia_cadera = circ_cadera_base + np.random.normal(0, 8, n_registros)
        circunferencia_cadera = np.clip(circunferencia_cadera, 70, 160)

        # Antecedentes familiares (30% tiene)
        antecedentes_familiares = (np.random.random(n_registros) < 0.30).astype(int)

        # Creatinina (función renal)
        creatinina = np.random.normal(1.0, 0.2, n_registros)
        # Aumentada en diabetes/hipertensión avanzada
        for i, diag in enumerate(diagnosticos):
            if diag in ['Diabetes', 'Hipertension'] and edad[i] > 55:
                creatinina[i] = creatinina[i] * 1.3
        creatinina = np.clip(creatinina, 0.5, 3.0)

        # Frecuencia cardíaca
        frecuencia_cardiaca = np.random.normal(72, 10, n_registros)
        # Aumentada en obesidad/hipertensión
        for i, diag in enumerate(diagnosticos):
            if diag in ['Obesidad', 'Hipertension']:
                frecuencia_cardiaca[i] = frecuencia_cardiaca[i] * 1.1
        frecuencia_cardiaca = np.clip(frecuencia_cardiaca, 50, 120).astype(int)

        # Actividad física (minutos/semana)
        actividad_fisica = np.random.gamma(2, 40, n_registros)  # Mayoría hace poco ejercicio
        # Personas normales hacen más ejercicio
        for i, diag in enumerate(diagnosticos):
            if diag == 'Normal':
                actividad_fisica[i] = actividad_fisica[i] * 1.8
            elif diag == 'Obesidad':
                actividad_fisica[i] = actividad_fisica[i] * 0.4
        actividad_fisica = np.clip(actividad_fisica, 0, 500).astype(int)

        # Consumo de alcohol (bebidas/semana)
        consumo_alcohol = np.random.poisson(2, n_registros)  # Promedio 2 bebidas/semana
        consumo_alcohol = np.clip(consumo_alcohol, 0, 20)

    # ===== CREAR DATAFRAME =====
    print("\nCreando DataFrame...")

    df = pd.DataFrame({
        'ID': range(1, n_registros + 1),
        'Edad': edad,
        'Sexo': sexo,
        'Peso': np.round(peso, 1),
        'Altura': np.round(altura, 1),
        'IMC': np.round(imc, 2),
        'Presion_Sistolica': np.round(presion_sistolica, 0).astype(int),
        'Presion_Diastolica': np.round(presion_diastolica, 0).astype(int),
        'Glucosa': np.round(glucosa, 0).astype(int),
        'Colesterol': np.round(colesterol_total, 0).astype(int),
        'Fumador': np.where(fumador == 1, 'Si', 'No'),
        'Diagnostico': diagnosticos
    })

    # Agregar columna Presion_Arterial en formato "120/80"
    df['Presion_Arterial'] = df['Presion_Sistolica'].astype(str) + '/' + df['Presion_Diastolica'].astype(str)

    # Agregar datos nivel 3 si aplica
    if nivel == "COMPLETO":
        df['Hemoglobina_A1C'] = np.round(hemoglobina_a1c, 1)
        df['Colesterol_LDL'] = np.round(colesterol_ldl, 0).astype(int)
        df['Colesterol_HDL'] = np.round(colesterol_hdl, 0).astype(int)
        df['Trigliceridos'] = np.round(trigliceridos, 0).astype(int)
        df['Circunferencia_Cintura'] = np.round(circunferencia_cintura, 1)
        df['Circunferencia_Cadera'] = np.round(circunferencia_cadera, 1)
        df['Antecedentes_Familiares'] = np.where(antecedentes_familiares == 1, 'Si', 'No')
        df['Creatinina'] = np.round(creatinina, 2)
        df['Frecuencia_Cardiaca'] = frecuencia_cardiaca
        df['Actividad_Fisica'] = actividad_fisica
        df['Consumo_Alcohol'] = consumo_alcohol

    # ===== VERIFICAR COHERENCIA CLÍNICA =====
    print("\nVerificando coherencia clínica...")

    diabetes_correcto = df[df['Diagnostico'] == 'Diabetes']['Glucosa'].apply(lambda x: x > 126).sum()
    diabetes_total = len(df[df['Diagnostico'] == 'Diabetes'])

    hipertension_correcto = df[df['Diagnostico'] == 'Hipertension']['Presion_Sistolica'].apply(lambda x: x > 140).sum()
    hipertension_total = len(df[df['Diagnostico'] == 'Hipertension'])

    normal_correcto = df[df['Diagnostico'] == 'Normal'].apply(
        lambda row: row['Glucosa'] < 100 and row['Presion_Sistolica'] < 120,
        axis=1
    ).sum()
    normal_total = len(df[df['Diagnostico'] == 'Normal'])

    print(f"\nCoherencia de etiquetas:")
    if diabetes_total > 0:
        print(f"  - Diabetes con glucosa > 126: {diabetes_correcto}/{diabetes_total} ({diabetes_correcto/diabetes_total*100:.1f}%)")
    if hipertension_total > 0:
        print(f"  - Hipertension con presion > 140: {hipertension_correcto}/{hipertension_total} ({hipertension_correcto/hipertension_total*100:.1f}%)")
    if normal_total > 0:
        print(f"  - Normal con valores normales: {normal_correcto}/{normal_total} ({normal_correcto/normal_total*100:.1f}%)")

    # ===== ESTADÍSTICAS FINALES =====
    print(f"\n{'='*70}")
    print("DATASET GENERADO EXITOSAMENTE")
    print(f"{'='*70}")
    print(f"Total de registros: {len(df):,}")
    print(f"Total de columnas: {len(df.columns)}")
    print(f"Nivel de datos: {nivel}")
    print(f"\nColumnas: {', '.join(df.columns.tolist())}")

    return df


if __name__ == "__main__":
    # Generar diferentes versiones del dataset

    print("\n" + "="*70)
    print("GENERADOR DE DATASETS MÉDICOS SINTÉTICOS")
    print("="*70)

    # 1. Dataset COMPLETO (10,000 registros con todas las columnas)
    print("\n\n>>> Generando dataset COMPLETO...")
    df_completo = generar_dataset_medico(n_registros=10000, nivel="COMPLETO")
    filename_completo = f"dataset_medico_completo_10k_{datetime.now().strftime('%Y%m%d')}.xlsx"
    df_completo.to_excel(filename_completo, index=False)
    print(f"\nOK - Guardado: {filename_completo}")

    # 2. Dataset ESTÁNDAR (5,000 registros sin columnas avanzadas)
    print("\n\n>>> Generando dataset ESTÁNDAR...")
    df_estandar = generar_dataset_medico(n_registros=5000, nivel="ESTANDAR")
    filename_estandar = f"dataset_medico_estandar_5k_{datetime.now().strftime('%Y%m%d')}.xlsx"
    df_estandar.to_excel(filename_estandar, index=False)
    print(f"\nOK - Guardado: {filename_estandar}")

    # 3. Dataset MÍNIMO (2,000 registros solo columnas básicas)
    print("\n\n>>> Generando dataset MÍNIMO...")
    df_minimo = generar_dataset_medico(n_registros=2000, nivel="MINIMO")
    # Mantener solo columnas mínimas
    columnas_minimas = ['ID', 'Edad', 'Sexo', 'Glucosa', 'Presion_Sistolica', 'Presion_Arterial', 'Diagnostico']
    df_minimo = df_minimo[columnas_minimas]
    filename_minimo = f"dataset_medico_minimo_2k_{datetime.now().strftime('%Y%m%d')}.xlsx"
    df_minimo.to_excel(filename_minimo, index=False)
    print(f"\nOK - Guardado: {filename_minimo}")

    # 4. Dataset en CSV (para probar múltiples formatos)
    print("\n\n>>> Generando versión CSV...")
    filename_csv = f"dataset_medico_completo_10k_{datetime.now().strftime('%Y%m%d')}.csv"
    df_completo.to_csv(filename_csv, index=False)
    print(f"\nOK - Guardado: {filename_csv}")

    # 5. Muestra de estadísticas
    print("\n\n" + "="*70)
    print("ESTADÍSTICAS DEL DATASET COMPLETO")
    print("="*70)
    print("\nPrimeras 5 filas:")
    print(df_completo.head())

    print("\n\nEstadísticas descriptivas:")
    print(df_completo.describe())

    print("\n\n" + "="*70)
    print("DATASETS GENERADOS EXITOSAMENTE!")
    print("="*70)
    print("\nArchivos creados:")
    print(f"  1. {filename_completo} (10,000 registros, TODAS las columnas)")
    print(f"  2. {filename_estandar} (5,000 registros, columnas estándar)")
    print(f"  3. {filename_minimo} (2,000 registros, columnas mínimas)")
    print(f"  4. {filename_csv} (10,000 registros, formato CSV)")
    print("\nListo para usar en tu sistema!")
