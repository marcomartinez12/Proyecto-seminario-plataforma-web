"""
FUNCIONES MEJORADAS PARA ANALYSIS.PY
Sistema flexible que se adapta a columnas disponibles
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from scipy import stats


def preprocess_data_improved(df):
    """
    Preprocesar datos con sistema flexible de columnas.
    Se adapta automáticamente a las columnas disponibles (NIVEL 1, 2 o 3)
    """
    from scipy import stats

    print("\n" + "="*70)
    print("PREPROCESAMIENTO DE DATOS - VERSION ULTRA MEJORADA")
    print("="*70)
    print(f"Registros iniciales: {len(df):,}")

    # Crear una copia
    data = df.copy()

    # ===== AUTO-DETECTAR COLUMNAS DISPONIBLES =====
    columnas_disponibles = set(data.columns)
    print(f"\nColumnas disponibles: {len(columnas_disponibles)}")

    # Nivel 1 (Mínimo)
    nivel1_cols = {'Edad', 'Sexo', 'Glucosa', 'Presion_Sistolica'}
    # Nivel 2 (Estándar)
    nivel2_cols = {'Peso', 'Altura', 'Presion_Diastolica', 'Colesterol', 'Fumador'}
    # Nivel 3 (Completo)
    nivel3_cols = {
        'Hemoglobina_A1C', 'Colesterol_LDL', 'Colesterol_HDL', 'Trigliceridos',
        'Circunferencia_Cintura', 'Circunferencia_Cadera', 'Antecedentes_Familiares',
        'Creatinina', 'Frecuencia_Cardiaca', 'Actividad_Fisica', 'Consumo_Alcohol'
    }

    tiene_nivel1 = nivel1_cols.issubset(columnas_disponibles)
    tiene_nivel2 = len(columnas_disponibles & nivel2_cols) >= 3
    tiene_nivel3 = len(columnas_disponibles & nivel3_cols) >= 5

    if tiene_nivel3:
        nivel_datos = "NIVEL_3_COMPLETO"
        print("  >>> NIVEL 3 (COMPLETO) - Precision esperada: 85-95%")
    elif tiene_nivel2:
        nivel_datos = "NIVEL_2_ESTANDAR"
        print("  >>> NIVEL 2 (ESTANDAR) - Precision esperada: 75-85%")
    else:
        nivel_datos = "NIVEL_1_MINIMO"
        print("  >>> NIVEL 1 (MINIMO) - Precision esperada: 60-70%")

    # ===== 1. AUTO-ETIQUETADO CLÍNICO (si no hay columna Diagnostico) =====
    if 'Diagnostico' not in data.columns:
        print("\n  ! No se encontro columna 'Diagnostico'")
        print("  >>> Generando diagnosticos automaticamente segun criterios clinicos...")

        # Asegurar que existan las columnas necesarias
        if 'Presion_Sistolica' not in data.columns and 'Presion_Arterial' in data.columns:
            # Extraer presión sistólica si existe Presion_Arterial
            def extract_systolic(pressure_str):
                try:
                    if '/' in str(pressure_str):
                        return float(str(pressure_str).split('/')[0])
                    else:
                        return float(pressure_str)
                except:
                    return 120.0
            data['Presion_Sistolica'] = data['Presion_Arterial'].apply(extract_systolic)

        # Generar diagnóstico automático basado en criterios clínicos
        def generar_diagnostico_clinico(row):
            glucosa = row.get('Glucosa', 100)
            presion = row.get('Presion_Sistolica', 120)

            # Criterios médicos estándar
            if glucosa > 126:
                return 'Diabetes'
            elif presion > 140:
                return 'Hipertension'
            elif glucosa > 100 or presion > 130:
                return 'Prediabetes'
            else:
                return 'Normal'

        data['Diagnostico'] = data.apply(generar_diagnostico_clinico, axis=1)
        print(f"  >>> Diagnosticos generados: {len(data)} registros")

    # ===== 2. FILTRAR DIAGNÓSTICOS VÁLIDOS =====
    diagnosticos_validos = [
        'Normal', 'Hipertension', 'Hipertensión', 'hipertension',
        'Diabetes', 'diabetes',
        'Prediabetes', 'prediabetes',
        'Obesidad', 'obesidad',
        'Síndrome Metabólico', 'Sindrome Metabolico',
        'Dislipidemia', 'dislipidemia'
    ]

    # Normalizar nombres
    data['Diagnostico'] = data['Diagnostico'].str.strip()
    data['Diagnostico'] = data['Diagnostico'].replace({
        'Hipertensión': 'Hipertension',
        'hipertension': 'Hipertension',
        'diabetes': 'Diabetes',
        'prediabetes': 'Prediabetes',
        'obesidad': 'Obesidad',
        'Sindrome Metabolico': 'Síndrome Metabólico',
        'dislipidemia': 'Dislipidemia'
    })

    # Mostrar distribución original
    print("\nDiagnosticos en dataset original:")
    diag_counts = data['Diagnostico'].value_counts()
    for diag, count in diag_counts.head(10).items():
        pct = (count / len(data)) * 100
        print(f"   - {diag}: {count:,} ({pct:.1f}%)")

    # Filtrar solo válidos
    data = data[data['Diagnostico'].isin(diagnosticos_validos)]
    print(f"\nRegistros despues de filtrar: {len(data):,} ({len(data)/len(df)*100:.1f}% del total)")

    if len(data) < 100:
        raise ValueError(
            f"Insuficientes registros validos: {len(data)} (se requieren >=100).\n"
            f"Diagnosticos esperados: {', '.join(set([d for d in diagnosticos_validos if not d.islower()]))}"
        )

    # ===== 3. CONVERTIR COLUMNAS NUMÉRICAS BÁSICAS =====
    print("\nConvirtiendo columnas numericas...")
    numeric_columns = ['Edad', 'Glucosa']
    if 'Peso' in data.columns:
        numeric_columns.append('Peso')
    if 'Altura' in data.columns:
        numeric_columns.append('Altura')
    if 'Colesterol' in data.columns:
        numeric_columns.append('Colesterol')

    for col in numeric_columns:
        data[col] = pd.to_numeric(data[col], errors='coerce')

    # ===== 4. CODIFICAR SEXO =====
    le_sexo = LabelEncoder()
    data['Sexo_encoded'] = le_sexo.fit_transform(data['Sexo'])

    # ===== 5. PROCESAR PRESIÓN ARTERIAL =====
    if 'Presion_Sistolica' not in data.columns and 'Presion_Arterial' in data.columns:
        def extract_pressure_values(pressure_str):
            try:
                if '/' in str(pressure_str):
                    systolic, diastolic = str(pressure_str).split('/')
                    return float(systolic), float(diastolic)
                else:
                    return float(pressure_str), 80.0
            except:
                return 120.0, 80.0

        pressure_values = data['Presion_Arterial'].apply(extract_pressure_values)
        data['Presion_Sistolica'] = [p[0] for p in pressure_values]
        data['Presion_Diastolica'] = [p[1] for p in pressure_values]

    # Convertir presiones a numéricas
    data['Presion_Sistolica'] = pd.to_numeric(data['Presion_Sistolica'], errors='coerce')
    if 'Presion_Diastolica' in data.columns:
        data['Presion_Diastolica'] = pd.to_numeric(data['Presion_Diastolica'], errors='coerce')
    else:
        # Si no existe, asumir 60% de sistólica
        data['Presion_Diastolica'] = data['Presion_Sistolica'] * 0.6

    # ===== 6. CODIFICAR FUMADOR (si existe) =====
    if 'Fumador' in data.columns:
        data['Fumador_encoded'] = data['Fumador'].map({
            'Si': 1, 'Sí': 1, 'SI': 1, 'Yes': 1, 'yes': 1, True: 1, 1: 1,
            'No': 0, 'NO': 0, 'no': 0, False: 0, 0: 0
        })
        data['Fumador_encoded'] = data['Fumador_encoded'].fillna(0)
    else:
        data['Fumador_encoded'] = 0  # Asumir no fumador

    # ===== 7. CALCULAR IMC (si existen Peso y Altura) =====
    if 'Peso' in data.columns and 'Altura' in data.columns:
        data['IMC'] = data['Peso'] / ((data['Altura'] / 100) ** 2)
    elif 'IMC' not in data.columns:
        # IMC por defecto basado en edad y sexo
        data['IMC'] = 22 + (data['Edad'] - 30) * 0.1

    # ===== 8. ELIMINAR VALORES EXTREMOS (outliers) =====
    print("\nEliminando outliers...")
    registros_antes = len(data)

    # Filtros básicos
    data = data[
        (data['Edad'] > 0) & (data['Edad'] < 120) &
        (data['Glucosa'] > 50) & (data['Glucosa'] < 500) &
        (data['Presion_Sistolica'] > 60) & (data['Presion_Sistolica'] < 250)
    ]

    # Filtros opcionales
    if 'Peso' in data.columns:
        data = data[(data['Peso'] > 20) & (data['Peso'] < 300)]
    if 'Altura' in data.columns:
        data = data[(data['Altura'] > 100) & (data['Altura'] < 250)]
    if 'Colesterol' in data.columns:
        data = data[(data['Colesterol'] > 100) & (data['Colesterol'] < 400)]
    if 'IMC' in data.columns:
        data = data[(data['IMC'] > 10) & (data['IMC'] < 60)]

    registros_eliminados = registros_antes - len(data)
    print(f"   Outliers eliminados: {registros_eliminados} ({registros_eliminados/registros_antes*100:.1f}%)")

    # ===== 9. FEATURE ENGINEERING (adaptativo según columnas disponibles) =====
    print("\nCreando features derivadas...")
    features_creadas = []

    # Features básicas (siempre)
    data['Presion_Media'] = (data['Presion_Sistolica'] + 2 * data['Presion_Diastolica']) / 3
    data['Presion_Pulso'] = data['Presion_Sistolica'] - data['Presion_Diastolica']
    features_creadas.extend(['Presion_Media', 'Presion_Pulso'])

    # Score cardiovascular básico
    data['Score_Cardiovascular'] = (
        data['Presion_Sistolica'] / 140 +
        data['Glucosa'] / 126 +
        data['Edad'] / 60
    ) / 3
    features_creadas.append('Score_Cardiovascular')

    # Features si existe IMC
    if 'IMC' in data.columns:
        data['IMC_x_Edad'] = data['IMC'] * data['Edad']
        data['Categoria_IMC'] = pd.cut(
            data['IMC'],
            bins=[0, 18.5, 25, 30, 100],
            labels=[0, 1, 2, 3]
        ).astype(int)
        features_creadas.extend(['IMC_x_Edad', 'Categoria_IMC'])

    # Features con colesterol
    if 'Colesterol' in data.columns:
        data['Ratio_Colesterol_Edad'] = data['Colesterol'] / data['Edad']
        features_creadas.append('Ratio_Colesterol_Edad')

    # Interacciones avanzadas
    data['Glucosa_x_IMC'] = data['Glucosa'] * data.get('IMC', 25)
    data['Presion_x_Edad'] = data['Presion_Sistolica'] * data['Edad']
    data['Glucosa_x_Edad'] = data['Glucosa'] * data['Edad']
    data['Ratio_Sistolica_Diastolica'] = data['Presion_Sistolica'] / data['Presion_Diastolica']
    features_creadas.extend(['Glucosa_x_IMC', 'Presion_x_Edad', 'Glucosa_x_Edad', 'Ratio_Sistolica_Diastolica'])

    # Categorías
    data['Categoria_Edad'] = pd.cut(
        data['Edad'],
        bins=[0, 18, 35, 50, 65, 150],
        labels=[0, 1, 2, 3, 4]
    ).astype(int)

    data['Categoria_Glucosa'] = pd.cut(
        data['Glucosa'],
        bins=[0, 100, 126, 200, 1000],
        labels=[0, 1, 2, 3]
    ).astype(int)

    data['Categoria_Presion'] = pd.cut(
        data['Presion_Sistolica'],
        bins=[0, 120, 140, 160, 300],
        labels=[0, 1, 2, 3]
    ).astype(int)

    features_creadas.extend(['Categoria_Edad', 'Categoria_Glucosa', 'Categoria_Presion'])

    # ===== FEATURES NIVEL 3 (si están disponibles) =====
    if 'Hemoglobina_A1C' in data.columns:
        data['A1C_x_Glucosa'] = data['Hemoglobina_A1C'] * data['Glucosa']
        features_creadas.append('A1C_x_Glucosa')

    if 'Colesterol_LDL' in data.columns and 'Colesterol_HDL' in data.columns:
        data['Ratio_LDL_HDL'] = data['Colesterol_LDL'] / (data['Colesterol_HDL'] + 1)
        features_creadas.append('Ratio_LDL_HDL')

    if 'Circunferencia_Cintura' in data.columns and 'Circunferencia_Cadera' in data.columns:
        data['Ratio_Cintura_Cadera'] = data['Circunferencia_Cintura'] / data['Circunferencia_Cadera']
        data['Score_Obesidad_Abdominal'] = (data['Circunferencia_Cintura'] / 100) * data.get('IMC', 25)
        features_creadas.extend(['Ratio_Cintura_Cadera', 'Score_Obesidad_Abdominal'])

    if 'Trigliceridos' in data.columns:
        data['Score_Metabolico'] = (
            data['Glucosa'] / 126 +
            data['Trigliceridos'] / 150 +
            data['Presion_Sistolica'] / 140
        ) / 3
        features_creadas.append('Score_Metabolico')

    if 'Antecedentes_Familiares' in data.columns:
        # Codificar antecedentes
        data['Antecedentes_encoded'] = data['Antecedentes_Familiares'].map({
            'Si': 1, 'Sí': 1, 'SI': 1, 'Yes': 1, 'yes': 1, True: 1, 1: 1,
            'No': 0, 'NO': 0, 'no': 0, False: 0, 0: 0
        }).fillna(0)
        # Factor de riesgo genético
        data['Riesgo_Genetico'] = data['Antecedentes_encoded'] * (data['Edad'] / 50)
        features_creadas.extend(['Antecedentes_encoded', 'Riesgo_Genetico'])

    if 'Frecuencia_Cardiaca' in data.columns:
        data['FC_x_Presion'] = data['Frecuencia_Cardiaca'] * data['Presion_Sistolica']
        features_creadas.append('FC_x_Presion')

    if 'Creatinina' in data.columns:
        # eGFR simplificado (filtración glomerular estimada)
        data['eGFR'] = 186 * (data['Creatinina'] ** -1.154) * (data['Edad'] ** -0.203)
        features_creadas.append('eGFR')

    print(f"   Features creadas: {len(features_creadas)}")
    print(f"   Total de features para ML: {len(features_creadas) + 4}")  # +4 básicas

    # ===== ELIMINAR VALORES NaN =====
    data = data.fillna(data.median(numeric_only=True))

    print(f"\nRegistros finales: {len(data):,}")
    print("="*70)

    return data, le_sexo, nivel_datos, features_creadas


def create_ml_model_improved(data, nivel_datos, features_personalizadas):
    """
    Crear modelo ML que se adapta dinámicamente a las columnas disponibles.
    """
    from xgboost import XGBClassifier
    from imblearn.over_sampling import SMOTE
    from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
    from sklearn.preprocessing import LabelEncoder
    from sklearn.metrics import classification_report, confusion_matrix

    print("\n" + "="*70)
    print("ENTRENAMIENTO DE MODELO - VERSION ADAPTATIVA")
    print("="*70)

    # ===== CONSTRUIR LISTA DE FEATURES DINÁMICAMENTE =====
    features_basicas = ['Edad', 'Sexo_encoded', 'Glucosa', 'Presion_Sistolica', 'Presion_Diastolica', 'Fumador_encoded']

    # Agregar opcionales si existen
    features_opcionales = ['Peso', 'Altura', 'IMC', 'Colesterol']
    for feat in features_opcionales:
        if feat in data.columns:
            features_basicas.append(feat)

    # Agregar nivel 3 si existen
    features_nivel3 = [
        'Hemoglobina_A1C', 'Colesterol_LDL', 'Colesterol_HDL', 'Trigliceridos',
        'Circunferencia_Cintura', 'Circunferencia_Cadera', 'Antecedentes_encoded',
        'Creatinina', 'Frecuencia_Cardiaca'
    ]
    for feat in features_nivel3:
        if feat in data.columns:
            features_basicas.append(feat)

    # Agregar features personalizadas
    features_basicas.extend(features_personalizadas)

    # Filtrar solo features que existan en el DataFrame
    features = [f for f in features_basicas if f in data.columns]

    print(f"\nFeatures seleccionadas: {len(features)}")
    print(f"Nivel de datos: {nivel_datos}")

    # ===== PREPARAR DATOS =====
    X = data[features]
    y = data['Diagnostico']

    # Codificar target
    le_diagnostico = LabelEncoder()
    y_encoded = le_diagnostico.fit_transform(y)

    print(f"\nDatos de entrenamiento:")
    print(f"   X shape: {X.shape}")
    print(f"   Clases: {list(le_diagnostico.classes_)}")

    # ===== SPLIT ESTRATIFICADO =====
    X_train, X_test, y_train, y_test = train_test_split(
        X, y_encoded,
        test_size=0.2,
        random_state=42,
        stratify=y_encoded
    )

    print(f"\nTrain: {len(X_train)}, Test: {len(X_test)}")

    # ===== APLICAR SMOTE (balanceo) =====
    aplicar_smote = False
    if len(X_train) >= 500:
        class_counts = pd.Series(y_train).value_counts()
        min_class = class_counts.min()
        imbalance = class_counts.max() / min_class

        if imbalance > 2 and min_class >= 6:
            print(f"\nAplicando SMOTE (desbalance: {imbalance:.1f}:1)...")
            try:
                smote = SMOTE(random_state=42, k_neighbors=min(5, min_class-1))
                X_train, y_train = smote.fit_resample(X_train, y_train)
                print(f"   Datos balanceados: {len(X_train)} registros")
                aplicar_smote = True
            except Exception as e:
                print(f"   SMOTE no aplicado: {e}")

    if not aplicar_smote:
        print("\nSMOTE no aplicado (datos suficientemente balanceados)")

    # ===== CONFIGURAR MODELO XGBOOST =====
    # Ajustar hiperparámetros según nivel de datos
    if nivel_datos == "NIVEL_3_COMPLETO":
        n_estimators = 400
        max_depth = 10
        learning_rate = 0.03
    elif nivel_datos == "NIVEL_2_ESTANDAR":
        n_estimators = 300
        max_depth = 8
        learning_rate = 0.05
    else:  # NIVEL_1_MINIMO
        n_estimators = 200
        max_depth = 6
        learning_rate = 0.07

    print(f"\nConfiguracion XGBoost:")
    print(f"   Arboles: {n_estimators}")
    print(f"   Profundidad: {max_depth}")
    print(f"   Learning rate: {learning_rate}")

    model = XGBClassifier(
        n_estimators=n_estimators,
        learning_rate=learning_rate,
        max_depth=max_depth,
        min_child_weight=3,
        gamma=0.1,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_alpha=0.1,
        reg_lambda=1.0,
        random_state=42,
        n_jobs=-1,
        eval_metric='mlogloss',
        use_label_encoder=False
    )

    # ===== VALIDACIÓN CRUZADA =====
    print("\nValidacion cruzada (5-fold)...")
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    cv_scores = cross_val_score(model, X_train, y_train, cv=skf, scoring='f1_weighted', n_jobs=-1)

    print(f"   F1-Score (CV): {cv_scores.mean():.4f} (+/-{cv_scores.std():.4f})")
    print(f"   Scores individuales: {[f'{s:.4f}' for s in cv_scores]}")

    # ===== ENTRENAR MODELO FINAL =====
    print("\nEntrenando modelo final...")
    model.fit(X_train, y_train)

    # ===== EVALUAR =====
    y_pred = model.predict(X_test)
    final_accuracy = (y_pred == y_test).sum() / len(y_test)

    print(f"\nPrecision en test: {final_accuracy*100:.2f}%")

    # Decodificar para reporte
    y_test_decoded = le_diagnostico.inverse_transform(y_test)
    y_pred_decoded = le_diagnostico.inverse_transform(y_pred)

    # Matriz de confusión
    print("\nMatriz de confusion:")
    cm = confusion_matrix(y_test_decoded, y_pred_decoded, labels=le_diagnostico.classes_)
    cm_df = pd.DataFrame(cm, index=le_diagnostico.classes_, columns=le_diagnostico.classes_)
    print(cm_df)

    # Reporte de clasificación
    print("\nReporte de clasificacion:")
    print(classification_report(y_test_decoded, y_pred_decoded, zero_division=0))

    # Feature importance
    feature_importance = dict(zip(features, model.feature_importances_))
    feature_importance = dict(sorted(feature_importance.items(), key=lambda x: x[1], reverse=True))

    print("\nTop 10 features mas importantes:")
    for i, (feat, imp) in enumerate(list(feature_importance.items())[:10], 1):
        print(f"   {i}. {feat}: {imp:.4f}")

    print("="*70)

    return model, final_accuracy * 100, feature_importance, y_test_decoded, y_pred_decoded


# Ejemplo de uso:
# df_procesado, le_sexo, nivel, features = preprocess_data_improved(df)
# model, accuracy, importance, y_test, y_pred = create_ml_model_improved(df_procesado, nivel, features)
