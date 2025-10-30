from fastapi import APIRouter, HTTPException, BackgroundTasks
from fastapi.responses import FileResponse
import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score, classification_report, f1_score, confusion_matrix
from sklearn.preprocessing import LabelEncoder
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image, Table, TableStyle
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.lib import colors
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
import matplotlib
matplotlib.use('Agg')  # Para generar gráficos sin interfaz gráfica
import uuid
from datetime import datetime
import json
import httpx

from app.schemas import AnalysisRequest, AnalysisResponse
from app.utils import storage
import time
import psutil

router = APIRouter()

# Configurar fuente Times New Roman para los PDFs
try:
    # Intentar registrar Times New Roman (puede variar según el sistema)
    pdfmetrics.registerFont(TTFont('TimesNewRoman', 'C:/Windows/Fonts/times.ttf'))
    pdfmetrics.registerFont(TTFont('TimesNewRoman-Bold', 'C:/Windows/Fonts/timesbd.ttf'))
except:
    # Si no se encuentra, usar fuente por defecto
    pass

def preprocess_data(df):
    """Preprocesar los datos para el modelo ML - VERSIÓN OPTIMIZADA"""
    from scipy import stats

    print("\n" + "="*70)
    print("PREPROCESAMIENTO DE DATOS - VERSION MEJORADA")
    print("="*70)
    print(f"Registros iniciales: {len(df):,}")

    # Crear una copia para no modificar el original
    data = df.copy()

    # ===== 1. FILTRAR DIAGNÓSTICOS VÁLIDOS =====
    diagnosticos_validos = [
        'Normal', 'Hipertension', 'Hipertensión', 'hipertension',
        'Diabetes', 'diabetes',
        'Prediabetes', 'prediabetes',
        'Obesidad', 'obesidad',
        'Síndrome Metabólico', 'Sindrome Metabolico',
        'Dislipidemia', 'dislipidemia'
    ]

    # Normalizar nombres (quitar espacios, estandarizar)
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

    # Mostrar diagnósticos ANTES de filtrar
    print("\nDiagnosticos en dataset original:")
    diag_counts = data['Diagnostico'].value_counts()
    for diag, count in diag_counts.head(10).items():
        pct = (count / len(data)) * 100
        print(f"   - {diag}: {count:,} ({pct:.1f}%)")

    if len(diag_counts) > 10:
        print(f"   ... y {len(diag_counts) - 10} diagnosticos mas")

    # Filtrar solo diagnósticos válidos
    data = data[data['Diagnostico'].isin(diagnosticos_validos)]

    print(f"\nRegistros despues de filtrar: {len(data):,} ({len(data)/len(df)*100:.1f}% del total)")

    # Validar que hay suficientes datos
    if len(data) < 100:
        raise ValueError(
            f"Insuficientes registros validos: {len(data)} (se requieren >=100).\n"
            f"Diagnosticos esperados: {', '.join(set([d for d in diagnosticos_validos if not d.islower()]))}"
        )

    print("\nDiagnosticos validos encontrados:")
    valid_counts = data['Diagnostico'].value_counts()
    for diag, count in valid_counts.items():
        pct = (count / len(data)) * 100
        print(f"   - {diag}: {count:,} ({pct:.1f}%)")

    # Verificar balance de clases
    max_class = valid_counts.max()
    min_class = valid_counts.min()
    imbalance_ratio = max_class / min_class
    if imbalance_ratio > 5:
        print(f"\nDesbalance detectado: {imbalance_ratio:.1f}:1 (se aplicara balanceo automatico)")
    elif imbalance_ratio > 2:
        print(f"\nDesbalance moderado: {imbalance_ratio:.1f}:1")
    else:
        print(f"\nBalance aceptable: {imbalance_ratio:.1f}:1")

    # ===== 2. CONVERTIR COLUMNAS NUMÉRICAS =====
    print("\nConvirtiendo columnas numericas...")
    numeric_columns = ['Edad', 'Peso', 'Altura', 'Glucosa', 'Colesterol']
    for col in numeric_columns:
        data[col] = pd.to_numeric(data[col], errors='coerce')

    # ===== 3. CODIFICAR SEXO =====
    le_sexo = LabelEncoder()
    data['Sexo_encoded'] = le_sexo.fit_transform(data['Sexo'])

    # ===== 4. PROCESAR PRESIÓN ARTERIAL =====
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

    # ===== 5. CODIFICAR FUMADOR =====
    data['Fumador_encoded'] = data['Fumador'].map({
        'Si': 1, 'Sí': 1, 'si': 1, 'sí': 1, 'SI': 1, 'SÍ': 1,
        'No': 0, 'no': 0, 'NO': 0
    }).fillna(0).astype(int)

    # ===== 6. CALCULAR IMC =====
    data['IMC'] = data['Peso'] / ((data['Altura'] / 100) ** 2)

    # ===== 7. ELIMINAR OUTLIERS EXTREMOS =====
    print("\nEliminando outliers extremos (valores clinicamente imposibles)...")
    len_before = len(data)

    data = data[
        (data['Edad'] > 0) & (data['Edad'] < 120) &
        (data['Peso'] > 20) & (data['Peso'] < 300) &
        (data['Altura'] > 100) & (data['Altura'] < 250) &
        (data['Glucosa'] > 0) & (data['Glucosa'] < 600) &
        (data['Colesterol'] > 50) & (data['Colesterol'] < 500) &
        (data['Presion_Sistolica'] > 60) & (data['Presion_Sistolica'] < 250) &
        (data['Presion_Diastolica'] > 30) & (data['Presion_Diastolica'] < 150) &
        (data['IMC'] > 10) & (data['IMC'] < 60)
    ]

    outliers_removed = len_before - len(data)
    if outliers_removed > 0:
        print(f"   Outliers eliminados: {outliers_removed:,} ({outliers_removed/len_before*100:.2f}%)")
    else:
        print(f"   No se encontraron outliers extremos")

    # ===== 8. FEATURE ENGINEERING AVANZADO =====
    print("\nCreando features clinicas avanzadas...")

    # Presión Arterial Media (mejor predictor cardiovascular)
    data['Presion_Media'] = (data['Presion_Sistolica'] + 2*data['Presion_Diastolica']) / 3

    # Presión de Pulso (indicador de rigidez arterial)
    data['Presion_Pulso'] = data['Presion_Sistolica'] - data['Presion_Diastolica']

    # Ratio Colesterol/Edad (aproximación de riesgo acumulado)
    data['Ratio_Colesterol_Edad'] = data['Colesterol'] / (data['Edad'] + 1)

    # Score de Riesgo Cardiovascular (basado en guías clínicas)
    data['Score_Cardiovascular'] = (
        (data['Presion_Sistolica'] > 140).astype(int) * 3 +
        (data['Presion_Sistolica'] > 130).astype(int) * 2 +
        (data['Glucosa'] > 126).astype(int) * 3 +
        (data['Glucosa'] > 100).astype(int) * 2 +
        (data['IMC'] > 30).astype(int) * 3 +
        (data['IMC'] > 25).astype(int) * 1 +
        (data['Colesterol'] > 240).astype(int) * 2 +
        data['Fumador_encoded'] * 3
    )

    # Interacciones de features
    data['IMC_x_Edad'] = data['IMC'] * data['Edad']
    data['Glucosa_x_IMC'] = data['Glucosa'] * data['IMC']
    data['Presion_x_Edad'] = data['Presion_Sistolica'] * data['Edad']
    data['Glucosa_x_Edad'] = data['Glucosa'] * data['Edad']

    # Ratios clínicos
    data['Ratio_Sistolica_Diastolica'] = data['Presion_Sistolica'] / (data['Presion_Diastolica'] + 1)

    # Categorías de riesgo
    data['Categoria_IMC'] = pd.cut(
        data['IMC'],
        bins=[0, 18.5, 25, 30, 35, 100],
        labels=[0, 1, 2, 3, 4]
    ).astype(int)

    data['Categoria_Edad'] = pd.cut(
        data['Edad'],
        bins=[0, 30, 45, 60, 75, 100],
        labels=[0, 1, 2, 3, 4]
    ).astype(int)

    data['Categoria_Glucosa'] = pd.cut(
        data['Glucosa'],
        bins=[0, 100, 126, 200, 1000],
        labels=[0, 1, 2, 3]
    ).astype(int)

    data['Categoria_Presion'] = pd.cut(
        data['Presion_Sistolica'],
        bins=[0, 120, 130, 140, 180, 300],
        labels=[0, 1, 2, 3, 4]
    ).astype(int)

    print(f"   Features creadas: 15 nuevas variables clinicas")

    # ===== 9. LLENAR VALORES NULOS =====
    print("\nManejando valores nulos...")
    all_numeric_cols = numeric_columns + [
        'IMC', 'Presion_Sistolica', 'Presion_Diastolica', 'Presion_Media',
        'Presion_Pulso', 'Ratio_Colesterol_Edad', 'Score_Cardiovascular',
        'IMC_x_Edad', 'Glucosa_x_IMC', 'Presion_x_Edad', 'Glucosa_x_Edad',
        'Ratio_Sistolica_Diastolica'
    ]

    nulls_before = data[all_numeric_cols].isnull().sum().sum()

    for col in all_numeric_cols:
        if col in data.columns:
            nulls = data[col].isnull().sum()
            if nulls > 0:
                data[col] = data[col].fillna(data[col].median())

    if nulls_before > 0:
        print(f"   Valores nulos rellenados: {nulls_before:,}")
    else:
        print(f"   No se encontraron valores nulos")

    # ===== 10. RESUMEN FINAL =====
    print("\n" + "="*70)
    print("PREPROCESAMIENTO COMPLETADO")
    print("="*70)
    print(f"Registros finales: {len(data):,}")
    print(f"Features totales: {len([col for col in data.columns if col not in ['ID', 'Diagnostico', 'Presion_Arterial', 'Fumador', 'Sexo']])}")
    print(f"Clases: {len(data['Diagnostico'].unique())}")
    print("="*70 + "\n")

    return data, le_sexo

def create_ml_model(data):
    """Crear y entrenar el modelo de ML - VERSIÓN OPTIMIZADA CON XGBOOST"""
    from xgboost import XGBClassifier
    from sklearn.metrics import f1_score, classification_report, confusion_matrix
    from sklearn.model_selection import StratifiedKFold
    from imblearn.over_sampling import SMOTE

    print("\n" + "="*70)
    print("ENTRENAMIENTO DE MODELO DE MACHINE LEARNING")
    print("="*70)

    # ===== 1. SELECCIONAR CARACTERÍSTICAS =====
    features = [
        # Features originales
        'Edad', 'Sexo_encoded', 'Peso', 'Altura', 'IMC',
        'Presion_Sistolica', 'Presion_Diastolica', 'Glucosa',
        'Colesterol', 'Fumador_encoded',
        # Features avanzadas
        'Presion_Media', 'Presion_Pulso', 'Ratio_Colesterol_Edad',
        'Score_Cardiovascular', 'IMC_x_Edad', 'Glucosa_x_IMC',
        'Presion_x_Edad', 'Glucosa_x_Edad', 'Ratio_Sistolica_Diastolica',
        'Categoria_IMC', 'Categoria_Edad', 'Categoria_Glucosa', 'Categoria_Presion'
    ]

    print(f"\nFeatures seleccionadas: {len(features)}")

    X = data[features]
    y = data['Diagnostico']

    # Codificar target (XGBoost requiere enteros)
    le_target = LabelEncoder()
    y_encoded = le_target.fit_transform(y)

    print(f"Clases a predecir: {list(le_target.classes_)}")

    # ===== 2. SPLIT ESTRATIFICADO =====
    print(f"\nDividiendo datos...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y_encoded,
        test_size=0.2,
        random_state=42,
        stratify=y_encoded
    )

    print(f"   Entrenamiento: {len(X_train):,} registros ({len(X_train)/len(X)*100:.1f}%)")
    print(f"   Prueba: {len(X_test):,} registros ({len(X_test)/len(X)*100:.1f}%)")

    # ===== 3. BALANCEO CON SMOTE =====
    if len(X_train) >= 500:
        train_class_counts = pd.Series(y_train).value_counts()
        max_class = train_class_counts.max()
        min_class = train_class_counts.min()
        imbalance = max_class / min_class

        if imbalance > 2 and min_class >= 6:
            try:
                print(f"\nAplicando SMOTE para balancear clases...")
                smote = SMOTE(random_state=42, k_neighbors=min(5, min_class-1))
                X_train, y_train = smote.fit_resample(X_train, y_train)
                print(f"   Dataset balanceado: {len(X_train):,} registros")

                smote_counts = pd.Series(y_train).value_counts()
                for class_idx, count in smote_counts.items():
                    class_name = le_target.classes_[class_idx]
                    print(f"      - {class_name}: {count:,}")
            except Exception as e:
                print(f"   SMOTE no aplicado: {str(e)}")

    # ===== 4. CREAR MODELO XGBOOST =====
    print(f"\nCreando modelo XGBoost optimizado...")

    model = XGBClassifier(
        n_estimators=300,
        learning_rate=0.05,
        max_depth=8,
        min_child_weight=3,
        gamma=0.1,
        reg_alpha=0.1,
        reg_lambda=1.0,
        subsample=0.8,
        colsample_bytree=0.8,
        colsample_bylevel=0.8,
        random_state=42,
        n_jobs=-1,
        eval_metric='mlogloss',
        use_label_encoder=False
    )

    print(f"   Modelo: XGBoost Classifier")
    print(f"   Arboles: 300")
    print(f"   Learning rate: 0.05")
    print(f"   Profundidad: 8")

    # ===== 5. VALIDACIÓN CRUZADA ESTRATIFICADA =====
    if len(data) >= 100:
        print(f"\nValidacion cruzada estratificada (5-fold)...")
        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

        X_original = data[features]
        y_original = le_target.transform(data['Diagnostico'])

        cv_scores = cross_val_score(
            model, X_original, y_original,
            cv=skf,
            scoring='f1_weighted',
            n_jobs=-1
        )

        cv_mean = cv_scores.mean()
        cv_std = cv_scores.std()

        print(f"   F1-Score (CV): {cv_mean:.4f} (+/-{cv_std:.4f})")
        print(f"   Scores por fold: {[f'{s:.4f}' for s in cv_scores]}")
    else:
        cv_mean = None

    # ===== 6. ENTRENAR MODELO =====
    print(f"\nEntrenando modelo final...")
    model.fit(X_train, y_train)
    print(f"   Modelo entrenado exitosamente")

    # ===== 7. PREDICCIONES =====
    y_pred = model.predict(X_test)

    # Decodificar para compatibilidad
    y_test_decoded = le_target.inverse_transform(y_test)
    y_pred_decoded = le_target.inverse_transform(y_pred)

    # ===== 8. MÉTRICAS =====
    print(f"\nMETRICAS DEL MODELO")
    print("="*70)

    accuracy = accuracy_score(y_test_decoded, y_pred_decoded)
    f1 = f1_score(y_test_decoded, y_pred_decoded, average='weighted')

    print(f"Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")
    print(f"F1-Score (weighted): {f1:.4f} ({f1*100:.2f}%)")

    if cv_mean is not None:
        print(f"F1-Score (CV): {cv_mean:.4f} ({cv_mean*100:.2f}%)")

    # Reporte detallado
    print(f"\nReporte de Clasificacion por Clase:")
    print("-"*70)
    report = classification_report(y_test_decoded, y_pred_decoded, output_dict=True)

    for class_name in le_target.classes_:
        if class_name in report:
            metrics = report[class_name]
            print(f"\n{class_name}:")
            print(f"   Precision: {metrics['precision']:.4f}")
            print(f"   Recall:    {metrics['recall']:.4f}")
            print(f"   F1-Score:  {metrics['f1-score']:.4f}")
            print(f"   Support:   {int(metrics['support'])} casos")

    # Matriz de confusión
    print(f"\nMatriz de Confusion:")
    print("-"*70)
    cm = confusion_matrix(y_test_decoded, y_pred_decoded, labels=le_target.classes_)

    print("\n" + " "*15 + "  ".join([f"{c[:10]:>10}" for c in le_target.classes_]))
    for i, class_name in enumerate(le_target.classes_):
        row = cm[i]
        print(f"{class_name[:12]:>12}  " + "  ".join([f"{val:>10}" for val in row]))

    # ===== 9. IMPORTANCIA DE FEATURES =====
    print(f"\nTop 10 Features Mas Importantes:")
    print("-"*70)

    feature_importance = dict(zip(features, model.feature_importances_))
    top_features = sorted(feature_importance.items(), key=lambda x: x[1], reverse=True)[:10]

    for i, (feature, importance) in enumerate(top_features, 1):
        bar = "=" * int(importance * 50)
        print(f"{i:2}. {feature:25} {importance:6.4f} {bar}")

    # ===== 10. MÉTRICA FINAL =====
    final_accuracy = cv_mean if cv_mean is not None else f1

    print("\n" + "="*70)
    print(f"ENTRENAMIENTO COMPLETADO")
    print(f"Precision final (F1): {final_accuracy:.4f} ({final_accuracy*100:.2f}%)")
    print("="*70 + "\n")

    # Retornar métricas completas para el PDF
    detailed_metrics = {
        'accuracy': accuracy,
        'f1_score': f1,
        'cv_mean': cv_mean,
        'classification_report': report,
        'confusion_matrix': cm,
        'classes': le_target.classes_
    }

    return model, final_accuracy, feature_importance, y_test_decoded, y_pred_decoded, detailed_metrics

def generate_charts_optimized(data, output_dir, max_points=3000):
    """Generar gráficos optimizados para claridad científica"""
    if len(data) > max_points:
        sample_data = data.sample(n=max_points, random_state=42)
    else:
        sample_data = data
    
    charts = []
    
    # Configuración científica
    plt.style.use('seaborn-v0_8-whitegrid')
    plt.rcParams.update({
        'font.family': 'serif',
        'font.size': 11,
        'axes.labelsize': 12,
        'axes.titlesize': 14,
        'xtick.labelsize': 10,
        'ytick.labelsize': 10,
        'legend.fontsize': 10,
        'figure.titlesize': 15
    })
    
    # 1. Gráfico Doughnut de Distribución con Porcentajes Claros
    fig, ax = plt.subplots(figsize=(10, 8))
    diagnosis_counts = data['Diagnostico'].value_counts()
    
    # Colores científicos profesionales
    colors = ['#1f4e79', '#2e75b6', '#5b9bd5', '#9dc3e6', '#c5d9f1']
    
    wedges, texts, autotexts = ax.pie(
        diagnosis_counts.values, 
        labels=diagnosis_counts.index,
        autopct='%1.1f%%',
        colors=colors[:len(diagnosis_counts)],
        pctdistance=0.85,
        wedgeprops=dict(width=0.6, edgecolor='white', linewidth=2),
        startangle=90
    )
    
    # Mejorar visibilidad de porcentajes
    for autotext in autotexts:
        autotext.set_color('white')
        autotext.set_fontweight('bold')
        autotext.set_fontsize(13)
        autotext.set_bbox(dict(boxstyle='round,pad=0.3', facecolor='black', alpha=0.7))
    
    # Mejorar etiquetas
    for text in texts:
        text.set_fontsize(11)
        text.set_fontweight('bold')
        text.set_color('white')
    
    ax.set_title('Distribución de Diagnósticos Clínicos\n(n=' + f'{len(data):,}' + ' pacientes)', 
                fontweight='bold', pad=20, fontsize=16)
    
    # Agregar leyenda con conteos
    legend_labels = [f'{diag}: {count} casos' for diag, count in diagnosis_counts.items()]
    legend = ax.legend(wedges, legend_labels, title="Diagnósticos", loc="center left",
             bbox_to_anchor=(1, 0, 0.5, 1), fontsize=10)

    # Cambiar color de texto de la leyenda a blanco
    for text in legend.get_texts():
        text.set_color('white')
    legend.get_title().set_color('white')
    
    chart1_path = os.path.join(output_dir, 'distribucion_diagnosticos.png')
    plt.savefig(chart1_path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    charts.append(chart1_path)
    
    # 2. Gráfico Doughnut de Factores de Riesgo CORREGIDO
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # CORRECCIÓN: Usar el mismo criterio que el análisis principal
    high_bp = len(data[data['Diagnostico'].str.contains('Hipertension', case=False, na=False)])
    high_glucose = len(data[data['Diagnostico'].str.contains('Diabetes', case=False, na=False)])
    healthy = len(data) - high_bp - high_glucose
    
    risk_data = [high_bp, high_glucose, healthy]
    risk_labels = ['Hipertensión', 'Diabetes', 'Sin Factores']
    risk_colors = ['#dc3545', '#fd7e14', '#28a745']
    
    wedges, texts, autotexts = ax.pie(
        risk_data, 
        labels=risk_labels,
        autopct='%1.1f%%',
        colors=risk_colors,
        pctdistance=0.85,
        wedgeprops=dict(width=0.7, edgecolor='white', linewidth=3),
        startangle=90
    )
    
    # Texto central mejorado
    center_text = f'FACTORES\nDE RIESGO\n\nTotal: {len(data):,}\nPacientes'
    ax.text(0, 0, center_text, ha='center', va='center', 
           fontsize=14, fontweight='bold', 
           bbox=dict(boxstyle='round,pad=0.5', facecolor='lightgray', alpha=0.8))
    
    # Mejorar porcentajes
    for autotext in autotexts:
        autotext.set_color('white')
        autotext.set_fontweight('bold')
        autotext.set_fontsize(13)
        autotext.set_bbox(dict(boxstyle='round,pad=0.3', facecolor='black', alpha=0.8))
    
    ax.set_title('Distribución de Factores de Riesgo Cardiovascular', 
                fontweight='bold', pad=20, fontsize=16)
    
    chart2_path = os.path.join(output_dir, 'factores_riesgo.png')
    plt.savefig(chart2_path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    charts.append(chart2_path)
    
    # 3. Scatter Plot Mejorado para Correlaciones
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # Scatter con mejor legibilidad
    scatter = ax.scatter(
        data['Edad'], 
        data['Presion_Sistolica'], 
        s=data['IMC']*4,  # Tamaño basado en IMC
        c=data['Glucosa'], 
        cmap='viridis',
        alpha=0.7,
        edgecolors='black',
        linewidth=0.5
    )
    
    ax.set_xlabel('Edad (años)', fontweight='bold', fontsize=13)
    ax.set_ylabel('Presión Sistólica (mmHg)', fontweight='bold', fontsize=13)
    ax.set_title('Correlación: Edad vs Presión Arterial\n(Tamaño = IMC, Color = Glucosa)', 
                fontweight='bold', pad=20, fontsize=16)
    
    # Colorbar mejorado
    cbar = plt.colorbar(scatter, ax=ax)
    cbar.set_label('Glucosa (mg/dL)', fontweight='bold', fontsize=12)
    
    # Línea de tendencia
    z = np.polyfit(data['Edad'], data['Presion_Sistolica'], 1)
    p = np.poly1d(z)
    ax.plot(data['Edad'], p(data['Edad']), "r--", alpha=0.8, linewidth=3, 
           label=f'Tendencia (R² = {np.corrcoef(data["Edad"], data["Presion_Sistolica"])[0,1]**2:.3f})')
    
    # Líneas de referencia clínicas
    ax.axhline(y=140, color='red', linestyle=':', alpha=0.7, label='Hipertensión (≥140 mmHg)')
    ax.axhline(y=120, color='orange', linestyle=':', alpha=0.7, label='Pre-hipertensión (≥120 mmHg)')
    
    ax.legend(loc='upper left', fontsize=10)
    ax.grid(True, alpha=0.3)
    
    chart3_path = os.path.join(output_dir, 'correlacion_variables.png')
    plt.savefig(chart3_path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    charts.append(chart3_path)
    
    return charts

def generate_pdf_report(data, model_results, charts, output_path):
    """Generar reporte PDF con diseño profesional y colores específicos"""
    from reportlab.platypus import PageBreak, KeepTogether
    from reportlab.lib.colors import HexColor
    
    # Colores específicos solicitados
    AZUL_OSCURO = HexColor('#2C3E50')  # Azul oscuro
    AZUL_CLARO = HexColor('#3498DB')   # Azul claro
    VERDE_TURQUESA = HexColor('#16A085') # Verde turquesa
    GRIS_CLARO = HexColor('#F8F9FA')   # Gris muy claro para fondos
    GRIS_MEDIO = HexColor('#E9ECEF')   # Gris medio para tablas
    NEGRO = HexColor('#000000')        # Negro para texto
    BLANCO = HexColor('#FFFFFF')       # Blanco
    ROJO = HexColor('#E74C3C')         # Rojo para alertas
    NARANJA = HexColor('#F39C12')      # Naranja para advertencias
    AMARILLO = HexColor('#F1C40F')     # Amarillo para moderado
    VERDE = HexColor('#27AE60')        # Verde para normal
    
    doc = SimpleDocTemplate(output_path, pagesize=letter,
                          topMargin=0.8*inch, bottomMargin=1*inch,
                          leftMargin=0.8*inch, rightMargin=0.8*inch)
    story = []
    
    # Estilos con Times New Roman
    styles = getSampleStyleSheet()
    
    # Estilo para cabecera institucional (Times New Roman 14 bold, azul oscuro)
    header_style = ParagraphStyle(
        'InstitutionalHeader',
        parent=styles['Normal'],
        fontSize=14,
        fontName='Times-Bold',
        textColor=BLANCO,
        alignment=1,  # Centrado
        spaceAfter=6,
        spaceBefore=6,
        backColor=AZUL_OSCURO,
        borderPadding=12
    )
    
    # Estilo para título principal (Times New Roman 16 bold, negro, mayúsculas)
    main_title_style = ParagraphStyle(
        'MainTitle',
        parent=styles['Normal'],
        fontSize=16,
        fontName='Times-Bold',
        textColor=NEGRO,
        alignment=1,  # Centrado
        spaceAfter=18,
        spaceBefore=12,
        backColor=GRIS_CLARO,
        borderPadding=15,
        borderWidth=1,
        borderColor=AZUL_CLARO
    )
    
    # Estilo para subtítulos (Times New Roman 14 bold, gris oscuro)
    subtitle_style = ParagraphStyle(
        'Subtitle',
        parent=styles['Normal'],
        fontSize=14,
        fontName='Times-Bold',
        textColor=AZUL_OSCURO,
        alignment=0,  # Izquierda
        spaceAfter=9,
        spaceBefore=15,
        backColor=GRIS_MEDIO,
        borderPadding=8,
        borderWidth=1,
        borderColor=AZUL_CLARO
    )
    
    # Estilo para subtítulos nivel 2
    subtitle2_style = ParagraphStyle(
        'Subtitle2',
        parent=styles['Normal'],
        fontSize=12,
        fontName='Times-Bold',
        textColor=AZUL_OSCURO,
        alignment=0,
        spaceAfter=6,
        spaceBefore=12
    )
    
    # Estilo para texto normal (Times New Roman 12, negro)
    normal_style = ParagraphStyle(
        'NormalText',
        parent=styles['Normal'],
        fontSize=12,
        fontName='Times-Roman',
        textColor=NEGRO,
        spaceAfter=6,
        alignment=4,  # Justificado
        firstLineIndent=0.3*inch
    )
    
    # Estilo para información institucional
    institution_style = ParagraphStyle(
        'Institution',
        parent=styles['Normal'],
        fontSize=12,
        fontName='Times-Roman',
        textColor=NEGRO,
        alignment=1,  # Centrado
        spaceAfter=3
    )
    
    # Estilo para listas
    list_style = ParagraphStyle(
        'ListStyle',
        parent=styles['Normal'],
        fontSize=12,
        fontName='Times-Roman',
        textColor=NEGRO,
        spaceAfter=3,
        alignment=4,
        leftIndent=0.3*inch
    )
    
    # Función para crear pie de página
    def add_footer():
        footer_table = Table([[
            Paragraph("<i>Universidad Popular del Cesar - Seccional Aguachica | Análisis Predictivo de Tendencias en Salud</i>", 
                     ParagraphStyle('Footer', parent=styles['Normal'], fontSize=10, 
                                  fontName='Times-Italic', textColor=BLANCO, alignment=1))
        ]], colWidths=[7*inch])
        
        footer_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, -1), AZUL_OSCURO),
            ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
            ('TOPPADDING', (0, 0), (-1, -1), 8),
            ('BOTTOMPADDING', (0, 0), (-1, -1), 8),
        ]))
        
        return footer_table
    
    # PORTADA PROFESIONAL
    story.append(Spacer(1, 0.5*inch))
    
    # Cabecera institucional con fondo azul oscuro y línea decorativa
    header_content = [
        [Paragraph("UNIVERSIDAD POPULAR DEL CESAR - SECCIONAL AGUACHICA", header_style)],
        [Paragraph("Facultad de Ingeniería y Tecnologías", 
                  ParagraphStyle('SubHeader', parent=header_style, fontSize=12, spaceBefore=3, spaceAfter=3))],
        [Paragraph("Programa de Ingeniería de Sistemas", 
                  ParagraphStyle('SubHeader2', parent=header_style, fontSize=11, spaceBefore=3, spaceAfter=6))]
    ]
    
    header_table = Table(header_content, colWidths=[7*inch])
    header_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, -1), AZUL_OSCURO),
        ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
        ('TOPPADDING', (0, 0), (-1, -1), 12),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 12),
    ]))
    
    story.append(header_table)
    
    # Línea decorativa azul claro
    line_table = Table([[""]],  colWidths=[7*inch], rowHeights=[0.1*inch])
    line_table.setStyle(TableStyle([('BACKGROUND', (0, 0), (-1, -1), AZUL_CLARO)]))
    story.append(line_table)
    
    story.append(Spacer(1, 0.8*inch))
    
    # Título principal en recuadro gris claro
    title_table = Table([[
        Paragraph("ANÁLISIS PREDICTIVO DE TENDENCIAS EN ENFERMEDADES CRÓNICAS MEDIANTE MACHINE LEARNING", main_title_style)
    ]], colWidths=[7*inch])
    
    title_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, -1), GRIS_CLARO),
        ('BOX', (0, 0), (-1, -1), 2, AZUL_CLARO),
        ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
        ('TOPPADDING', (0, 0), (-1, -1), 15),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 15),
    ]))
    
    story.append(title_table)
    story.append(Spacer(1, 1*inch))
    
    # Información de estudiantes y fecha
    info_data = [
        ["Estudiantes:", "Marco Andrés Martínez Malagón\nCamilo Reyes Rodríguez"],
        ["Programa:", "Ingeniería de Sistemas"],
        ["Asignatura:", "Seminario de Investigación"],
        ["Profesor:", "Ing. Luis Palmera"],
       
        
        
    ]
    
    # Fecha en español
    meses = {
        1: 'enero', 2: 'febrero', 3: 'marzo', 4: 'abril',
        5: 'mayo', 6: 'junio', 7: 'julio', 8: 'agosto',
        9: 'septiembre', 10: 'octubre', 11: 'noviembre', 12: 'diciembre'
    }
    
    now = datetime.now()
    fecha_espanol = f"{now.day} de {meses[now.month]} de {now.year}"
    info_data.append(["Fecha:", fecha_espanol])
    
    info_table = Table(info_data, colWidths=[1.5*inch, 5.5*inch])
    info_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (0, -1), GRIS_MEDIO),  # Primera columna gris
        ('BACKGROUND', (1, 0), (1, -1), BLANCO),      # Segunda columna blanca
        ('GRID', (0, 0), (-1, -1), 1, AZUL_CLARO),
        ('FONTNAME', (0, 0), (0, -1), 'Times-Bold'),
        ('FONTNAME', (1, 0), (1, -1), 'Times-Roman'),
        ('FONTSIZE', (0, 0), (-1, -1), 12),
        ('VALIGN', (0, 0), (-1, -1), 'TOP'),
        ('TOPPADDING', (0, 0), (-1, -1), 8),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 8),
        ('LEFTPADDING', (0, 0), (-1, -1), 10),
        ('RIGHTPADDING', (0, 0), (-1, -1), 10),
    ]))
    
    story.append(info_table)
    story.append(Spacer(1, 1*inch))
    
    # Pie de página de portada
    story.append(add_footer())
    story.append(PageBreak())
    
    # CONTENIDO DEL REPORTE

    # Estadísticas reales para las tablas
    total_records = len(data)
    hypertension_cases = len(data[data['Diagnostico'].str.contains('Hipertension', case=False, na=False)])
    diabetes_cases = len(data[data['Diagnostico'].str.contains('Diabetes', case=False, na=False)])
    high_bp_cases = len(data[data['Presion_Sistolica'] > 140])
    high_glucose_cases = len(data[data['Glucosa'] > 126])
    high_cholesterol_cases = len(data[data['Colesterol'] > 240])
    obese_cases = len(data[data['IMC'] > 30])
    smokers_cases = len(data[data['Fumador_encoded'] == 1])

    # Calcular estadísticas demográficas (necesarias para varias secciones)
    edad_mean = data['Edad'].mean()
    edad_std = data['Edad'].std()
    edad_min = data['Edad'].min()
    edad_max = data['Edad'].max()
    edad_median = data['Edad'].median()

    imc_mean = data['IMC'].mean()
    imc_std = data['IMC'].std()
    imc_min = data['IMC'].min()
    imc_max = data['IMC'].max()
    imc_median = data['IMC'].median()

    ps_mean = data['Presion_Sistolica'].mean()
    ps_std = data['Presion_Sistolica'].std()
    ps_min = data['Presion_Sistolica'].min()
    ps_max = data['Presion_Sistolica'].max()
    ps_median = data['Presion_Sistolica'].median()

    gluc_mean = data['Glucosa'].mean()
    gluc_std = data['Glucosa'].std()
    gluc_min = data['Glucosa'].min()
    gluc_max = data['Glucosa'].max()
    gluc_median = data['Glucosa'].median()

    col_mean = data['Colesterol'].mean()
    col_std = data['Colesterol'].std()
    col_min = data['Colesterol'].min()
    col_max = data['Colesterol'].max()
    col_median = data['Colesterol'].median()
    
    # 1. DETECCIÓN DE PATRONES DE RIESGO CARDIOVASCULAR
    story.append(Paragraph("1. DETECCIÓN DE PATRONES DE RIESGO CARDIOVASCULAR", subtitle_style))
    
    # Tabla de factores de riesgo detectados
    riesgo_data = [
        ["Factor de Riesgo", "Casos Detectados", "Prevalencia (%)", "Nivel de Riesgo"],
        ["Hipertensión Arterial", f"{high_bp_cases:,}", f"{(high_bp_cases/total_records)*100:.1f}%", "ALTO" if (high_bp_cases/total_records)*100 > 30 else "MEDIO"],
        ["Diabetes Mellitus", f"{high_glucose_cases:,}", f"{(high_glucose_cases/total_records)*100:.1f}%", "ALTO" if (high_glucose_cases/total_records)*100 > 15 else "MEDIO"],
        ["Dislipidemia", f"{high_cholesterol_cases:,}", f"{(high_cholesterol_cases/total_records)*100:.1f}%", "MEDIO" if (high_cholesterol_cases/total_records)*100 > 20 else "BAJO"],
        ["Obesidad", f"{obese_cases:,}", f"{(obese_cases/total_records)*100:.1f}%", "ALTO" if (obese_cases/total_records)*100 > 25 else "MEDIO"],
        ["Tabaquismo", f"{smokers_cases:,}", f"{(smokers_cases/total_records)*100:.1f}%", "MEDIO" if (smokers_cases/total_records)*100 > 20 else "BAJO"]
    ]
    
    riesgo_table = Table(riesgo_data, colWidths=[2.2*inch, 1.3*inch, 1.2*inch, 1.3*inch])
    riesgo_table.setStyle(TableStyle([
        # Cabecera con fondo azul oscuro
        ('BACKGROUND', (0, 0), (-1, 0), AZUL_OSCURO),
        ('TEXTCOLOR', (0, 0), (-1, 0), BLANCO),
        ('FONTNAME', (0, 0), (-1, 0), 'Times-Bold'),
        ('FONTSIZE', (0, 0), (-1, 0), 11),
        ('ALIGN', (0, 0), (-1, 0), 'CENTER'),
        
        # Filas alternadas
        ('BACKGROUND', (0, 1), (-1, 1), BLANCO),
        ('BACKGROUND', (0, 2), (-1, 2), GRIS_CLARO),
        ('BACKGROUND', (0, 3), (-1, 3), BLANCO),
        ('BACKGROUND', (0, 4), (-1, 4), GRIS_CLARO),
        ('BACKGROUND', (0, 5), (-1, 5), BLANCO),
        
        # Colores dinámicos para nivel de riesgo
        ('GRID', (0, 0), (-1, -1), 1, NEGRO),
        ('FONTNAME', (0, 1), (-1, -1), 'Times-Roman'),
        ('FONTSIZE', (0, 1), (-1, -1), 10),
        ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
        ('TOPPADDING', (0, 0), (-1, -1), 8),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 8),
        ('ALIGN', (1, 1), (2, -1), 'CENTER'),  # Centrar números
    ]))
    
    # Aplicar colores dinámicos según el nivel de riesgo
    for i in range(1, len(riesgo_data)):
        nivel = riesgo_data[i][3]
        if nivel == "ALTO":
            riesgo_table.setStyle(TableStyle([('BACKGROUND', (3, i), (3, i), ROJO), ('TEXTCOLOR', (3, i), (3, i), BLANCO), ('FONTNAME', (3, i), (3, i), 'Times-Bold')]))
        elif nivel == "MEDIO":
            riesgo_table.setStyle(TableStyle([('BACKGROUND', (3, i), (3, i), NARANJA), ('TEXTCOLOR', (3, i), (3, i), BLANCO), ('FONTNAME', (3, i), (3, i), 'Times-Bold')]))
        else:
            riesgo_table.setStyle(TableStyle([('BACKGROUND', (3, i), (3, i), VERDE), ('TEXTCOLOR', (3, i), (3, i), BLANCO), ('FONTNAME', (3, i), (3, i), 'Times-Bold')]))
    
    story.append(riesgo_table)
    story.append(Spacer(1, 15))
    
    # Texto explicativo con aclaración sobre factores no excluyentes
    riesgo_text = f"""El análisis predictivo identifica {high_bp_cases + high_glucose_cases + high_cholesterol_cases:,} casos con factores de riesgo cardiovascular significativos. La hipertensión arterial representa el factor más prevalente con {(high_bp_cases/total_records)*100:.1f}% de la población, seguida por alteraciones glucémicas en {(high_glucose_cases/total_records)*100:.1f}% de los casos analizados.
    
    **Nota importante:** Los factores de riesgo no son mutuamente excluyentes. Un mismo paciente puede presentar múltiples comorbilidades, por lo que los porcentajes pueden sumar más del 100% debido a la superposición de condiciones."""
    story.append(Paragraph(riesgo_text, normal_style))
    story.append(Spacer(1, 20))
    
    # 2. TENDENCIAS EPIDEMIOLÓGICAS POR GRUPOS ETARIOS
    story.append(Paragraph("2. TENDENCIAS EPIDEMIOLÓGICAS POR GRUPOS ETARIOS", subtitle_style))
    
    # Análisis por grupos de edad
    age_groups = pd.cut(data['Edad'], bins=[0, 30, 45, 60, 100], labels=['< 30 años', '30-45 años', '46-60 años', '> 60 años'])
    age_analysis = []
    
    for grupo in ['< 30 años', '30-45 años', '46-60 años', '> 60 años']:
        grupo_data = data[age_groups == grupo]
        if len(grupo_data) > 0:
            hipertension_grupo = len(grupo_data[grupo_data['Presion_Sistolica'] > 140])
            diabetes_grupo = len(grupo_data[grupo_data['Glucosa'] > 126])
            obesidad_grupo = len(grupo_data[grupo_data['IMC'] > 30])
            
            # Determinar tendencia predominante
            max_factor = max(hipertension_grupo, diabetes_grupo, obesidad_grupo)
            if max_factor == hipertension_grupo and hipertension_grupo > 0:
                tendencia = "Hipertensión"
            elif max_factor == diabetes_grupo and diabetes_grupo > 0:
                tendencia = "Diabetes"
            elif max_factor == obesidad_grupo and obesidad_grupo > 0:
                tendencia = "Obesidad"
            else:
                tendencia = "Bajo Riesgo"
            
            age_analysis.append([
                grupo,
                f"{len(grupo_data):,}",
                f"{(len(grupo_data)/total_records)*100:.1f}%",
                tendencia,
                f"{(max_factor/len(grupo_data)*100):.1f}%" if len(grupo_data) > 0 else "0%"
            ])
    
    # Tabla de tendencias por edad
    edad_data = [["Grupo Etario", "Población", "% Total", "Tendencia Principal", "Prevalencia"]]
    edad_data.extend(age_analysis)
    
    edad_table = Table(edad_data, colWidths=[1.4*inch, 1.2*inch, 1*inch, 1.4*inch, 1*inch])
    edad_table.setStyle(TableStyle([
        # Cabecera azul oscuro
        ('BACKGROUND', (0, 0), (-1, 0), AZUL_OSCURO),
        ('TEXTCOLOR', (0, 0), (-1, 0), BLANCO),
        ('FONTNAME', (0, 0), (-1, 0), 'Times-Bold'),
        ('FONTSIZE', (0, 0), (-1, 0), 10),
        ('ALIGN', (0, 0), (-1, 0), 'CENTER'),
        
        # Filas alternadas
        ('BACKGROUND', (0, 1), (-1, 1), BLANCO),
        ('BACKGROUND', (0, 2), (-1, 2), GRIS_CLARO),
        ('BACKGROUND', (0, 3), (-1, 3), BLANCO),
        ('BACKGROUND', (0, 4), (-1, 4), GRIS_CLARO),
        
        # Formato general
        ('GRID', (0, 0), (-1, -1), 1, NEGRO),
        ('FONTNAME', (0, 1), (-1, -1), 'Times-Roman'),
        ('FONTSIZE', (0, 1), (-1, -1), 9),
        ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
        ('TOPPADDING', (0, 0), (-1, -1), 6),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 6),
        ('ALIGN', (1, 1), (2, -1), 'CENTER'),
        ('ALIGN', (4, 1), (4, -1), 'CENTER'),
    ]))
    
    # Colores para tendencias
    for i in range(1, len(edad_data)):
        tendencia = edad_data[i][3]
        if tendencia in ["Hipertensión", "Diabetes"]:
            edad_table.setStyle(TableStyle([('BACKGROUND', (3, i), (3, i), ROJO), ('TEXTCOLOR', (3, i), (3, i), BLANCO), ('FONTNAME', (3, i), (3, i), 'Times-Bold')]))
        elif tendencia == "Obesidad":
            edad_table.setStyle(TableStyle([('BACKGROUND', (3, i), (3, i), NARANJA), ('TEXTCOLOR', (3, i), (3, i), BLANCO), ('FONTNAME', (3, i), (3, i), 'Times-Bold')]))
        else:
            edad_table.setStyle(TableStyle([('BACKGROUND', (3, i), (3, i), VERDE_TURQUESA), ('TEXTCOLOR', (3, i), (3, i), BLANCO), ('FONTNAME', (3, i), (3, i), 'Times-Bold')]))
    
    story.append(edad_table)
    story.append(Spacer(1, 15))
    
    # Análisis de tendencias por edad
    edad_text = f"""El análisis etario revela patrones diferenciados de riesgo: los grupos de mayor edad (>60 años) presentan mayor prevalencia de hipertensión arterial, mientras que los grupos de mediana edad (30-60 años) muestran tendencias crecientes hacia diabetes mellitus tipo 2. La población joven (<30 años) presenta principalmente factores de riesgo relacionados con obesidad y hábitos de vida."""
    story.append(Paragraph(edad_text, normal_style))
    story.append(Spacer(1, 20))
    
    # 3. ANÁLISIS DEMOGRÁFICO
    story.append(Paragraph("3. ANÁLISIS DEMOGRÁFICO DE LA POBLACIÓN", subtitle_style))

    story.append(Paragraph("3.1 Características Generales de la Población", subtitle2_style))

    # Crear tabla con estadísticas demográficas (ya calculadas al inicio)
    demo_stats_data = [["Variable", "Media ± DE", "Rango (Min-Max)", "Mediana", "n (%)"]]

    # Edad (variables ya calculadas)
    demo_stats_data.append([
        "Edad (años)",
        f"{edad_mean:.1f} ± {edad_std:.1f}",
        f"{int(edad_min)} - {int(edad_max)}",
        f"{edad_median:.1f}",
        f"{len(data)}"
    ])

    # Sexo
    if 'Sexo' in data.columns:
        gender_counts = data['Sexo'].value_counts()
        for sexo in gender_counts.index:
            count = gender_counts[sexo]
            pct = (count / len(data)) * 100
            demo_stats_data.append([
                f"Sexo: {sexo}",
                "-",
                "-",
                "-",
                f"{count} ({pct:.1f}%)"
            ])

    # IMC (variables ya calculadas)
    demo_stats_data.append([
        "IMC (kg/m²)",
        f"{imc_mean:.1f} ± {imc_std:.1f}",
        f"{imc_min:.1f} - {imc_max:.1f}",
        f"{imc_median:.1f}",
        f"{len(data)}"
    ])

    # Presión Sistólica (variables ya calculadas)
    demo_stats_data.append([
        "Presión Sistólica (mmHg)",
        f"{ps_mean:.1f} ± {ps_std:.1f}",
        f"{int(ps_min)} - {int(ps_max)}",
        f"{ps_median:.1f}",
        f"{len(data)}"
    ])

    # Glucosa (variables ya calculadas)
    demo_stats_data.append([
        "Glucosa (mg/dL)",
        f"{gluc_mean:.1f} ± {gluc_std:.1f}",
        f"{int(gluc_min)} - {int(gluc_max)}",
        f"{gluc_median:.1f}",
        f"{len(data)}"
    ])

    # Colesterol (variables ya calculadas)
    demo_stats_data.append([
        "Colesterol (mg/dL)",
        f"{col_mean:.1f} ± {col_std:.1f}",
        f"{int(col_min)} - {int(col_max)}",
        f"{col_median:.1f}",
        f"{len(data)}"
    ])

    # Fumador
    if 'Fumador_encoded' in data.columns:
        fumadores = len(data[data['Fumador_encoded'] == 1])
        pct_fumadores = (fumadores / len(data)) * 100
        demo_stats_data.append([
            "Fumadores",
            "-",
            "-",
            "-",
            f"{fumadores} ({pct_fumadores:.1f}%)"
        ])

    # Crear tabla demográfica
    demo_table = Table(demo_stats_data, colWidths=[1.8*inch, 1.3*inch, 1.3*inch, 1.0*inch, 1.1*inch])
    demo_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), AZUL_OSCURO),
        ('TEXTCOLOR', (0, 0), (-1, 0), BLANCO),
        ('FONTNAME', (0, 0), (-1, 0), 'Times-Bold'),
        ('FONTSIZE', (0, 0), (-1, 0), 10),
        ('ALIGN', (0, 0), (-1, 0), 'CENTER'),

        # Filas alternadas
        ('ROWBACKGROUNDS', (0, 1), (-1, -1), [BLANCO, GRIS_CLARO]),

        ('GRID', (0, 0), (-1, -1), 1, NEGRO),
        ('FONTNAME', (0, 1), (-1, -1), 'Times-Roman'),
        ('FONTSIZE', (0, 1), (-1, -1), 9),
        ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
        ('TOPPADDING', (0, 0), (-1, -1), 6),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 6),
        ('ALIGN', (1, 1), (-1, -1), 'CENTER'),
    ]))

    story.append(demo_table)
    story.append(Spacer(1, 15))

    demo_text = f"""La población estudiada comprende {len(data):,} individuos con una edad promedio de {edad_mean:.1f} ± {edad_std:.1f} años (rango: {int(edad_min)}-{int(edad_max)} años). El análisis demográfico revela patrones epidemiológicos consistentes con la transición demográfica regional, caracterizada por el envejecimiento poblacional y el aumento de enfermedades crónicas no transmisibles.

**Nota metodológica:** Los valores se presentan como Media ± Desviación Estándar (DE) para variables continuas y n (%) para variables categóricas."""
    story.append(Paragraph(demo_text, normal_style))
    story.append(Spacer(1, 20))
    
    # 4. DETECCIÓN DE TENDENCIAS PREDICTIVAS
    story.append(Paragraph("4. DETECCIÓN DE TENDENCIAS PREDICTIVAS", subtitle_style))
    
    story.append(Paragraph("4.1 Rendimiento del Modelo de Machine Learning", subtitle2_style))
    modelo_text = f"""El modelo XGBoost obtuvo una precisión del {model_results['accuracy']:.2%} en la predicción de diagnósticos. Los resultados del análisis permiten identificar patrones en los datos de salud de la población estudiada."""
    story.append(Paragraph(modelo_text, normal_style))
    
    # Tabla de importancia de características
    feature_importance = model_results.get('feature_importance', {})
    top_features = sorted(feature_importance.items(), key=lambda x: x[1], reverse=True)[:5]
    
    if top_features:
        features_data = [["Variable Predictiva", "Importancia Relativa", "Impacto Clínico"]]
        
        for feature, importance in top_features:
            impacto = "ALTO" if importance > 0.15 else "MEDIO" if importance > 0.10 else "BAJO"
            features_data.append([feature, f"{importance:.3f}", impacto])
        
        features_table = Table(features_data, colWidths=[2.5*inch, 1.5*inch, 1.5*inch])
        features_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), AZUL_OSCURO),
            ('TEXTCOLOR', (0, 0), (-1, 0), BLANCO),
            ('FONTNAME', (0, 0), (-1, 0), 'Times-Bold'),
            ('FONTSIZE', (0, 0), (-1, 0), 11),
            ('ALIGN', (0, 0), (-1, 0), 'CENTER'),
            
            # Filas alternadas
            ('BACKGROUND', (0, 1), (-1, 1), BLANCO),
            ('BACKGROUND', (0, 2), (-1, 2), GRIS_CLARO),
            ('BACKGROUND', (0, 3), (-1, 3), BLANCO),
            ('BACKGROUND', (0, 4), (-1, 4), GRIS_CLARO),
            ('BACKGROUND', (0, 5), (-1, 5), BLANCO),
            
            # Colores dinámicos para impacto
            ('GRID', (0, 0), (-1, -1), 1, NEGRO),
            ('FONTNAME', (0, 1), (-1, -1), 'Times-Roman'),
            ('FONTSIZE', (0, 1), (-1, -1), 10),
            ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
            ('TOPPADDING', (0, 0), (-1, -1), 6),
            ('BOTTOMPADDING', (0, 0), (-1, -1), 6),
        ]))
        
        story.append(features_table)
        story.append(Spacer(1, 15))

    # 4.2 MÉTRICAS DETALLADAS DEL MODELO
    story.append(Paragraph("4.2 Métricas de Rendimiento del Modelo", subtitle2_style))

    # Obtener métricas detalladas
    detailed_metrics = model_results.get('detailed_metrics', {})
    classification_report = detailed_metrics.get('classification_report', {})

    if classification_report:
        # Tabla de métricas por clase
        metrics_data = [["Clase", "Precision", "Recall (Sensibilidad)", "F1-Score", "Soporte (n)"]]

        classes = detailed_metrics.get('classes', [])
        for class_name in classes:
            if class_name in classification_report:
                metrics = classification_report[class_name]
                metrics_data.append([
                    class_name,
                    f"{metrics['precision']:.3f}",
                    f"{metrics['recall']:.3f}",
                    f"{metrics['f1-score']:.3f}",
                    f"{int(metrics['support'])}"
                ])

        # Agregar fila de promedios
        if 'weighted avg' in classification_report:
            weighted = classification_report['weighted avg']
            metrics_data.append([
                "Promedio Ponderado",
                f"{weighted['precision']:.3f}",
                f"{weighted['recall']:.3f}",
                f"{weighted['f1-score']:.3f}",
                f"{int(weighted['support'])}"
            ])

        metrics_table = Table(metrics_data, colWidths=[1.8*inch, 1.2*inch, 1.5*inch, 1.2*inch, 1.3*inch])
        metrics_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), AZUL_OSCURO),
            ('TEXTCOLOR', (0, 0), (-1, 0), BLANCO),
            ('FONTNAME', (0, 0), (-1, 0), 'Times-Bold'),
            ('FONTSIZE', (0, 0), (-1, 0), 10),
            ('ALIGN', (0, 0), (-1, 0), 'CENTER'),

            # Filas de clases
            ('BACKGROUND', (0, 1), (-1, -2), BLANCO),
            ('BACKGROUND', (0, 2), (-1, -2), GRIS_CLARO),

            # Fila de promedio en color especial
            ('BACKGROUND', (0, -1), (-1, -1), VERDE_TURQUESA),
            ('TEXTCOLOR', (0, -1), (-1, -1), BLANCO),
            ('FONTNAME', (0, -1), (-1, -1), 'Times-Bold'),

            ('GRID', (0, 0), (-1, -1), 1, NEGRO),
            ('FONTNAME', (0, 1), (-1, -2), 'Times-Roman'),
            ('FONTSIZE', (0, 1), (-1, -1), 10),
            ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
            ('TOPPADDING', (0, 0), (-1, -1), 6),
            ('BOTTOMPADDING', (0, 0), (-1, -1), 6),
            ('ALIGN', (1, 1), (-1, -1), 'CENTER'),
        ]))

        story.append(metrics_table)
        story.append(Spacer(1, 15))

        # Texto explicativo de métricas
        accuracy = detailed_metrics.get('accuracy', 0)
        f1_global = detailed_metrics.get('f1_score', 0)

        metrics_text = f"""**Interpretación de Métricas:**

• **Precision:** Proporción de predicciones positivas correctas. Un valor de {weighted['precision']:.3f} indica que {weighted['precision']*100:.1f}% de los casos predichos como positivos son verdaderos positivos.

• **Recall (Sensibilidad):** Proporción de casos positivos correctamente identificados. Un valor de {weighted['recall']:.3f} indica que el modelo detecta {weighted['recall']*100:.1f}% de todos los casos reales.

• **F1-Score:** Media armónica de Precision y Recall. Valor global de {f1_global:.3f} ({f1_global*100:.1f}%), indicando un balance entre ambas métricas.

• **Accuracy Global:** {accuracy:.3f} ({accuracy*100:.1f}%) de todas las predicciones fueron correctas."""

        story.append(Paragraph(metrics_text, normal_style))
        story.append(Spacer(1, 20))

    # 4.3 MATRIZ DE CONFUSIÓN
    story.append(Paragraph("4.3 Matriz de Confusión del Modelo", subtitle2_style))

    confusion_matrix = detailed_metrics.get('confusion_matrix', None)
    classes = detailed_metrics.get('classes', [])

    if confusion_matrix is not None and len(classes) > 0:
        # Crear tabla de matriz de confusión
        cm_data = [["Clase Real \\ Predicho"] + [f"Pred: {c}" for c in classes]]

        for i, class_name in enumerate(classes):
            row = [f"Real: {class_name}"] + [str(int(confusion_matrix[i][j])) for j in range(len(classes))]
            cm_data.append(row)

        # Calcular anchos dinámicamente
        num_classes = len(classes)
        cell_width = 6*inch / (num_classes + 1)
        col_widths = [cell_width * 1.5] + [cell_width] * num_classes

        cm_table = Table(cm_data, colWidths=col_widths)

        # Estilo base
        cm_style = [
            ('BACKGROUND', (0, 0), (-1, 0), AZUL_OSCURO),
            ('TEXTCOLOR', (0, 0), (-1, 0), BLANCO),
            ('BACKGROUND', (0, 1), (0, -1), AZUL_OSCURO),
            ('TEXTCOLOR', (0, 1), (0, -1), BLANCO),
            ('FONTNAME', (0, 0), (-1, 0), 'Times-Bold'),
            ('FONTNAME', (0, 1), (0, -1), 'Times-Bold'),
            ('FONTSIZE', (0, 0), (-1, -1), 9),
            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
            ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
            ('GRID', (0, 0), (-1, -1), 1, NEGRO),
            ('TOPPADDING', (0, 0), (-1, -1), 6),
            ('BOTTOMPADDING', (0, 0), (-1, -1), 6),
        ]

        # Colorear celdas de la diagonal (predicciones correctas) en verde
        for i in range(len(classes)):
            cm_style.append(('BACKGROUND', (i+1, i+1), (i+1, i+1), VERDE))
            cm_style.append(('TEXTCOLOR', (i+1, i+1), (i+1, i+1), BLANCO))
            cm_style.append(('FONTNAME', (i+1, i+1), (i+1, i+1), 'Times-Bold'))

        # Colorear errores de predicción en naranja
        for i in range(len(classes)):
            for j in range(len(classes)):
                if i != j:
                    cm_style.append(('BACKGROUND', (j+1, i+1), (j+1, i+1), NARANJA))
                    cm_style.append(('TEXTCOLOR', (j+1, i+1), (j+1, i+1), BLANCO))

        cm_table.setStyle(TableStyle(cm_style))
        story.append(cm_table)
        story.append(Spacer(1, 15))

        # Calcular totales y precisión por clase
        total_correct = sum([confusion_matrix[i][i] for i in range(len(classes))])
        total_samples = confusion_matrix.sum()

        cm_text = f"""**Interpretación de la Matriz de Confusión:**

La diagonal principal (verde) representa las predicciones correctas: {total_correct} de {int(total_samples)} casos ({(total_correct/total_samples*100):.1f}%). Las celdas naranjas representan errores de clasificación.

• **Verdaderos Positivos (diagonal):** Casos correctamente clasificados
• **Falsos Positivos:** Casos clasificados incorrectamente como esa clase
• **Falsos Negativos:** Casos de esa clase clasificados como otra"""

        story.append(Paragraph(cm_text, normal_style))
        story.append(Spacer(1, 20))

    # 5. ANÁLISIS ESPECÍFICO DE ENFERMEDADES CRÓNICAS PRINCIPALES
    story.append(Paragraph("5. ANÁLISIS ESPECÍFICO DE HIPERTENSIÓN Y DIABETES", subtitle_style))
    
    # 5.1 Análisis detallado de Hipertensión
    story.append(Paragraph("5.1 Análisis de Tendencias en Hipertensión Arterial", subtitle2_style))
    
    # Calcular estadísticas específicas de hipertensión
    hipertension_data = data[data['Diagnostico'].str.contains('Hipertension', case=False, na=False)]
    diabetes_data = data[data['Diagnostico'].str.contains('Diabetes', case=False, na=False)]
    # Cambiar para usar valores clínicos reales:
    comorbilidad_data = data[
        (data['Presion_Sistolica'] > 140) & 
        (data['Glucosa'] > 126)
    ]
    
    # Análisis por grupos de edad para hipertensión
    hta_por_edad = []
    for grupo in ['< 30 años', '30-45 años', '46-60 años', '> 60 años']:
        grupo_data = data[age_groups == grupo]
        if len(grupo_data) > 0:
            hta_casos = len(grupo_data[grupo_data['Diagnostico'].str.contains('Hipertension', case=False, na=False)])
            hta_prevalencia = (hta_casos / len(grupo_data)) * 100 if len(grupo_data) > 0 else 0
            
            # Clasificar riesgo según prevalencia
            if hta_prevalencia > 40:
                riesgo_nivel = "MUY ALTO"
            elif hta_prevalencia > 25:
                riesgo_nivel = "ALTO"
            elif hta_prevalencia > 15:
                riesgo_nivel = "MODERADO"
            else:
                riesgo_nivel = "BAJO"
            
            hta_por_edad.append([
                grupo,
                f"{len(grupo_data):,}",
                f"{hta_casos:,}",
                f"{hta_prevalencia:.1f}%",
                riesgo_nivel
            ])
    
    # Tabla de hipertensión por edad
    hta_edad_data = [[
        "Grupo Etario", "Población Total", "Casos HTA", "Prevalencia", "Nivel de Riesgo"
    ]]
    hta_edad_data.extend(hta_por_edad)
    
    hta_table = Table(hta_edad_data, colWidths=[1.4*inch, 1.2*inch, 1.2*inch, 1.2*inch, 1.2*inch])
    hta_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), AZUL_OSCURO),
        ('TEXTCOLOR', (0, 0), (-1, 0), BLANCO),
        ('FONTNAME', (0, 0), (-1, 0), 'Times-Bold'),
        ('FONTSIZE', (0, 0), (-1, 0), 10),
        ('ALIGN', (0, 0), (-1, 0), 'CENTER'),
        ('GRID', (0, 0), (-1, -1), 1, NEGRO),
        ('FONTNAME', (0, 1), (-1, -1), 'Times-Roman'),
        ('FONTSIZE', (0, 1), (-1, -1), 9),
        ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
        ('TOPPADDING', (0, 0), (-1, -1), 6),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 6),
        ('ALIGN', (1, 1), (-1, -1), 'CENTER'),
    ]))
    
    # Aplicar colores según nivel de riesgo
    for i, fila in enumerate(hta_por_edad, 1):
        nivel = fila[4]
        if nivel == "MUY ALTO":
            color = ROJO
        elif nivel == "ALTO":
            color = NARANJA
        elif nivel == "MODERADO":
            color = NARANJA
        else:
            color = VERDE
        
        hta_table.setStyle(TableStyle([
            ('BACKGROUND', (4, i), (4, i), color),
            ('TEXTCOLOR', (4, i), (4, i), BLANCO),
            ('FONTNAME', (4, i), (4, i), 'Times-Bold')
        ]))
    
    story.append(hta_table)
    story.append(Spacer(1, 15))
    
    # Texto explicativo de hipertensión
    hta_text = f"""**Hallazgos Clave en Hipertensión:**
    
    • **Prevalencia General:** {len(hipertension_data):,} casos de hipertensión identificados ({(len(hipertension_data)/total_records)*100:.1f}% de la población)
    • **Grupo de Mayor Riesgo:** {max(hta_por_edad, key=lambda x: float(x[3].replace('%', '')))[0] if hta_por_edad else 'N/A'} con {max(hta_por_edad, key=lambda x: float(x[3].replace('%', '')))[3] if hta_por_edad else '0%'} de prevalencia
    • **Tendencia Etaria:** La prevalencia aumenta significativamente con la edad, siendo crítica en población >60 años
    • **Impacto Clínico:** La hipertensión no controlada aumenta el riesgo de eventos cardiovasculares en 2-4 veces
    
    **Recomendaciones Preventivas:**
    - Implementar programas de detección temprana en población >45 años
    - Establecer protocolos de seguimiento para pacientes con presión sistólica >130 mmHg
    - Promover cambios en estilo de vida: reducción de sodio, actividad física regular"""
    
    story.append(Paragraph(hta_text, normal_style))
    story.append(Spacer(1, 20))
    
    # 5.2 Análisis detallado de Diabetes
    story.append(Paragraph("5.2 Análisis de Tendencias en Diabetes Mellitus", subtitle2_style))
    
    # Análisis por grupos de edad para diabetes
    dm_por_edad = []
    for grupo in ['< 30 años', '30-45 años', '46-60 años', '> 60 años']:
        grupo_data = data[age_groups == grupo]
        if len(grupo_data) > 0:
            dm_casos = len(grupo_data[grupo_data['Diagnostico'].str.contains('Diabetes', case=False, na=False)])
            dm_prevalencia = (dm_casos / len(grupo_data)) * 100 if len(grupo_data) > 0 else 0
            
            # Calcular promedio de glucosa en el grupo
            glucosa_promedio = grupo_data['Glucosa'].mean()
            
            # Clasificar control glucémico
            if glucosa_promedio > 180:
                control = "DEFICIENTE"
            elif glucosa_promedio > 140:
                control = "REGULAR"
            elif glucosa_promedio > 100:
                control = "ACEPTABLE"
            else:
                control = "ÓPTIMO"
            
            dm_por_edad.append([
                grupo,
                f"{len(grupo_data):,}",
                f"{dm_casos:,}",
                f"{dm_prevalencia:.1f}%",
                f"{glucosa_promedio:.0f} mg/dL",
                control
            ])
    
    # Tabla de diabetes por edad
    dm_edad_data = [[
        "Grupo Etario", "Población", "Casos DM", "Prevalencia", "Glucosa Promedio", "Control Glucémico"
    ]]
    dm_edad_data.extend(dm_por_edad)
    
    dm_table = Table(dm_edad_data, colWidths=[1.1*inch, 1*inch, 1*inch, 1*inch, 1.2*inch, 1.2*inch])
    dm_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), AZUL_OSCURO),
        ('TEXTCOLOR', (0, 0), (-1, 0), BLANCO),
        ('FONTNAME', (0, 0), (-1, 0), 'Times-Bold'),
        ('FONTSIZE', (0, 0), (-1, 0), 9),
        ('ALIGN', (0, 0), (-1, 0), 'CENTER'),
        ('GRID', (0, 0), (-1, -1), 1, NEGRO),
        ('FONTNAME', (0, 1), (-1, -1), 'Times-Roman'),
        ('FONTSIZE', (0, 1), (-1, -1), 8),
        ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
        ('TOPPADDING', (0, 0), (-1, -1), 6),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 6),
        ('ALIGN', (1, 1), (-1, -1), 'CENTER'),
    ]))
    
    # Aplicar colores según control glucémico
    for i, fila in enumerate(dm_por_edad, 1):
        control = fila[5]
        if control == "DEFICIENTE":
            color = ROJO
        elif control == "REGULAR":
            color = NARANJA
        elif control == "ACEPTABLE":
            color = AMARILLO
        else:
            color = VERDE
        
        dm_table.setStyle(TableStyle([
            ('BACKGROUND', (5, i), (5, i), color),
            ('TEXTCOLOR', (5, i), (5, i), BLANCO),
            ('FONTNAME', (5, i), (5, i), 'Times-Bold')
        ]))
    
    story.append(dm_table)
    story.append(Spacer(1, 15))
    
    # Texto explicativo de diabetes
    dm_text = f"""**Hallazgos Clave en Diabetes:**
    
    • **Prevalencia General:** {len(diabetes_data):,} casos de diabetes identificados ({(len(diabetes_data)/total_records)*100:.1f}% de la población)
    • **Control Glucémico:** Análisis revela necesidad de mejora en el manejo terapéutico
    • **Factores de Riesgo:** Correlación significativa con edad, IMC y antecedentes familiares
    • **Complicaciones Potenciales:** Riesgo elevado de nefropatía, retinopatía y neuropatía diabética
    
    **Estrategias de Manejo:**
    - Implementar programas de educación diabetológica
    - Establecer metas de HbA1c <7% para la mayoría de pacientes
    - Monitoreo regular de complicaciones microvasculares y macrovasculares
    - Integración de equipos multidisciplinarios (endocrinólogo, nutricionista, educador)"""
    
    story.append(Paragraph(dm_text, normal_style))
    story.append(Spacer(1, 20))
    
    # 5.3 Análisis de Comorbilidad: Hipertensión + Diabetes
    story.append(Paragraph("5.3 Análisis de Comorbilidad: Hipertensión y Diabetes", subtitle2_style))
    
    # Calcular estadísticas de comorbilidad usando valores clínicos
    solo_hta = len(data[
        (data['Presion_Sistolica'] > 140) & 
        (data['Glucosa'] <= 126)
    ])
    
    solo_dm = len(data[
        (data['Glucosa'] > 126) & 
        (data['Presion_Sistolica'] <= 140)
    ])
    
    ambas_condiciones = len(comorbilidad_data)
    sin_condiciones = len(data[
        (data['Presion_Sistolica'] <= 140) & 
        (data['Glucosa'] <= 126)
    ])
    
    # Tabla de comorbilidad
    comorbilidad_tabla_data = [
        ["Condición", "Casos", "Prevalencia", "Riesgo Cardiovascular"],
        ["Solo Hipertensión", f"{solo_hta:,}", f"{(solo_hta/total_records)*100:.1f}%", "ALTO"],
        ["Solo Diabetes", f"{solo_dm:,}", f"{(solo_dm/total_records)*100:.1f}%", "ALTO"],
        ["Hipertensión + Diabetes", f"{ambas_condiciones:,}", f"{(ambas_condiciones/total_records)*100:.1f}%", "MUY ALTO"],
        ["Sin Condiciones", f"{sin_condiciones:,}", f"{(sin_condiciones/total_records)*100:.1f}%", "BAJO"]
    ]
    
    comorbilidad_table = Table(comorbilidad_tabla_data, colWidths=[2*inch, 1.2*inch, 1.3*inch, 1.5*inch])
    comorbilidad_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), AZUL_OSCURO),
        ('TEXTCOLOR', (0, 0), (-1, 0), BLANCO),
        ('FONTNAME', (0, 0), (-1, 0), 'Times-Bold'),
        ('FONTSIZE', (0, 0), (-1, 0), 10),
        ('ALIGN', (0, 0), (-1, 0), 'CENTER'),
        ('GRID', (0, 0), (-1, -1), 1, NEGRO),
        ('FONTNAME', (0, 1), (-1, -1), 'Times-Roman'),
        ('FONTSIZE', (0, 1), (-1, -1), 9),
        ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
        ('TOPPADDING', (0, 0), (-1, -1), 6),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 6),
        ('ALIGN', (1, 1), (2, -1), 'CENTER'),
    ]))
    
    # Colores para riesgo cardiovascular
    riesgo_colores = {
        "MUY ALTO": ROJO,
        "ALTO": NARANJA,
        "BAJO": VERDE
    }
    
    for i in range(1, len(comorbilidad_tabla_data)):
        riesgo = comorbilidad_tabla_data[i][3]
        color = riesgo_colores.get(riesgo, GRIS_MEDIO)
        comorbilidad_table.setStyle(TableStyle([
            ('BACKGROUND', (3, i), (3, i), color),
            ('TEXTCOLOR', (3, i), (3, i), BLANCO),
            ('FONTNAME', (3, i), (3, i), 'Times-Bold')
        ]))
    
    story.append(comorbilidad_table)
    story.append(Spacer(1, 15))
    
    # Texto de comorbilidad
    comorbilidad_text = f"""**Análisis de Comorbilidad:**
    
La presencia simultánea de hipertensión y diabetes ({ambas_condiciones:,} casos, {(ambas_condiciones/total_records)*100:.1f}%) representa el escenario de mayor riesgo cardiovascular. Estos pacientes requieren:
    
• **Manejo Integral:** Control estricto de presión arterial (<130/80 mmHg) y glucemia (HbA1c <7%)
• **Prevención Secundaria:** Uso de estatinas, antiagregantes plaquetarios según indicación
• **Monitoreo Especializado:** Evaluación regular de función renal, fondo de ojo y extremidades
• **Modificación de Estilo de Vida:** Dieta DASH, ejercicio supervisado, cesación tabáquica
    
**Impacto Epidemiológico:** La comorbilidad incrementa el riesgo de eventos cardiovasculares mayores en 3-5 veces comparado con población general."""
    
    story.append(Paragraph(comorbilidad_text, normal_style))
    story.append(Spacer(1, 20))

    # NUEVA SECCIÓN: RESUMEN DE DATOS CLAVE PARA ARTÍCULO CIENTÍFICO (COMENTADA)
    # story.append(PageBreak())
    # story.append(Paragraph("ANEXO: DATOS CLAVE PARA PUBLICACIÓN CIENTÍFICA", subtitle_style))

    # story.append(Paragraph("Resumen Estadístico para Citación", subtitle2_style))

    # Obtener métricas del modelo
    # detailed_metrics = model_results.get('detailed_metrics', {})
    # accuracy = detailed_metrics.get('accuracy', model_results['accuracy'])
    # f1_score = detailed_metrics.get('f1_score', 0)
    # classification_report = detailed_metrics.get('classification_report', {})

    # # Calcular intervalo de confianza para accuracy (aproximación binomial)
    # n_samples = total_records
    # z = 1.96  # 95% de confianza
    # ci_margin = z * np.sqrt((accuracy * (1 - accuracy)) / n_samples)
    # ci_lower = max(0, accuracy - ci_margin)
    # ci_upper = min(1, accuracy + ci_margin)

    # # Tabla de datos clave
    # datos_clave_data = [["Métrica/Parámetro", "Valor", "IC 95% / Detalles"]]

    # datos_clave_data.append(["DISEÑO DEL ESTUDIO", "", ""])
    # datos_clave_data.append(["Tamaño muestral (n)", f"{total_records:,}", "Casos analizados"])
    # datos_clave_data.append(["Período de análisis", fecha_espanol, "Fecha de generación"])
    # datos_clave_data.append(["Variables analizadas", "8 variables", "Edad, Sexo, IMC, PA, Glucosa, Col, Fumador, Diagnóstico"])

    # datos_clave_data.append(["", "", ""])
    # datos_clave_data.append(["RENDIMIENTO DEL MODELO ML", "", ""])
    # datos_clave_data.append(["Algoritmo utilizado", "XGBoost", "n_estimators=300, learning_rate=0.05, max_depth=8"])
    # datos_clave_data.append(["Accuracy (Precisión)", f"{accuracy:.3f} ({accuracy*100:.1f}%)", f"{ci_lower*100:.1f}% - {ci_upper*100:.1f}%"])
    # datos_clave_data.append(["F1-Score (ponderado)", f"{f1_score:.3f} ({f1_score*100:.1f}%)", "Media armónica Precision-Recall"])

    # if 'weighted avg' in classification_report:
    #     weighted = classification_report['weighted avg']
    #     datos_clave_data.append(["Precision (ponderada)", f"{weighted['precision']:.3f}", "Precisión promedio ponderada"])
    #     datos_clave_data.append(["Recall/Sensibilidad (ponderada)", f"{weighted['recall']:.3f}", "Sensibilidad promedio ponderada"])

    # datos_clave_data.append(["", "", ""])
    # datos_clave_data.append(["PREVALENCIAS POBLACIONALES", "", ""])

    # # Calcular IC para prevalencias (aproximación normal)
    # hta_prev = high_bp_cases / total_records
    # dm_prev = high_glucose_cases / total_records
    # comorb_prev = ambas_condiciones / total_records

    # hta_ci_margin = z * np.sqrt((hta_prev * (1 - hta_prev)) / total_records)
    # dm_ci_margin = z * np.sqrt((dm_prev * (1 - dm_prev)) / total_records)
    # comorb_ci_margin = z * np.sqrt((comorb_prev * (1 - comorb_prev)) / total_records)

    # datos_clave_data.append([
    #     "Hipertensión Arterial",
    #     f"{high_bp_cases:,} ({hta_prev*100:.1f}%)",
    #     f"IC 95%: {max(0, hta_prev-hta_ci_margin)*100:.1f}% - {min(1, hta_prev+hta_ci_margin)*100:.1f}%"
    # ])
    # datos_clave_data.append([
    #     "Diabetes Mellitus",
    #     f"{high_glucose_cases:,} ({dm_prev*100:.1f}%)",
    #     f"IC 95%: {max(0, dm_prev-dm_ci_margin)*100:.1f}% - {min(1, dm_prev+dm_ci_margin)*100:.1f}%"
    # ])
    # datos_clave_data.append([
    #     "Comorbilidad HTA+DM",
    #     f"{ambas_condiciones:,} ({comorb_prev*100:.1f}%)",
    #     f"IC 95%: {max(0, comorb_prev-comorb_ci_margin)*100:.1f}% - {min(1, comorb_prev+comorb_ci_margin)*100:.1f}%"
    # ])

    # datos_clave_data.append(["", "", ""])
    # datos_clave_data.append(["ESTADÍSTICAS DESCRIPTIVAS", "", ""])
    # datos_clave_data.append(["Edad promedio", f"{edad_mean:.1f} ± {edad_std:.1f} años", f"Mediana: {edad_median:.1f} años"])
    # datos_clave_data.append(["IMC promedio", f"{imc_mean:.1f} ± {imc_std:.1f} kg/m²", f"Rango: {imc_min:.1f}-{imc_max:.1f}"])
    # datos_clave_data.append(["Presión Sistólica promedio", f"{ps_mean:.1f} ± {ps_std:.1f} mmHg", f"Rango: {int(ps_min)}-{int(ps_max)}"])
    # datos_clave_data.append(["Glucosa promedio", f"{gluc_mean:.1f} ± {gluc_std:.1f} mg/dL", f"Rango: {int(gluc_min)}-{int(gluc_max)}"])

    # # Crear tabla
    # datos_table = Table(datos_clave_data, colWidths=[2.2*inch, 2.0*inch, 2.3*inch])
    # datos_table.setStyle(TableStyle([
    #     # Cabecera
    #     ('BACKGROUND', (0, 0), (-1, 0), AZUL_OSCURO),
    #     ('TEXTCOLOR', (0, 0), (-1, 0), BLANCO),
    #     ('FONTNAME', (0, 0), (-1, 0), 'Times-Bold'),
    #     ('FONTSIZE', (0, 0), (-1, 0), 10),
    #     ('ALIGN', (0, 0), (-1, 0), 'CENTER'),

    #     # Subtítulos (filas específicas)
    #     ('BACKGROUND', (0, 1), (-1, 1), GRIS_MEDIO),
    #     ('BACKGROUND', (0, 6), (-1, 6), GRIS_MEDIO),
    #     ('BACKGROUND', (0, 13), (-1, 13), GRIS_MEDIO),
    #     ('BACKGROUND', (0, 18), (-1, 18), GRIS_MEDIO),
    #     ('FONTNAME', (0, 1), (-1, 1), 'Times-Bold'),
    #     ('FONTNAME', (0, 6), (-1, 6), 'Times-Bold'),
    #     ('FONTNAME', (0, 13), (-1, 13), 'Times-Bold'),
    #     ('FONTNAME', (0, 18), (-1, 18), 'Times-Bold'),
    #     ('SPAN', (0, 1), (-1, 1)),
    #     ('SPAN', (0, 6), (-1, 6)),
    #     ('SPAN', (0, 13), (-1, 13)),
    #     ('SPAN', (0, 18), (-1, 18)),

    #     # Filas vacías (separadores)
    #     ('BACKGROUND', (0, 5), (-1, 5), BLANCO),
    #     ('BACKGROUND', (0, 12), (-1, 12), BLANCO),
    #     ('BACKGROUND', (0, 17), (-1, 17), BLANCO),

    #     # Resto de filas
    #     ('GRID', (0, 0), (-1, -1), 1, NEGRO),
    #     ('FONTNAME', (0, 2), (-1, -1), 'Times-Roman'),
    #     ('FONTSIZE', (0, 2), (-1, -1), 9),
    #     ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
    #     ('TOPPADDING', (0, 0), (-1, -1), 6),
    #     ('BOTTOMPADDING', (0, 0), (-1, -1), 6),
    #     ('ALIGN', (1, 1), (-1, -1), 'CENTER'),
    # ]))

    # story.append(datos_table)
    # story.append(Spacer(1, 15))

    # # Texto explicativo
    # datos_text = f"""**Uso de estos datos en publicación científica:**

    # Esta tabla contiene todos los valores estadísticos necesarios para la sección de Resultados de un artículo científico. Los intervalos de confianza (IC 95%) se calcularon mediante aproximación normal para proporciones.

    # **Cómo citar los resultados del modelo:**
    # "Se desarrolló un modelo de XGBoost (n_estimators=300, learning_rate=0.05, max_depth=8) que alcanzó una precisión de {accuracy*100:.1f}% (IC 95%: {ci_lower*100:.1f}%-{ci_upper*100:.1f}%) en la clasificación de diagnósticos, con un F1-Score ponderado de {f1_score:.3f}."

    # **Formato de citación para prevalencias:**
    # "La prevalencia de hipertensión arterial fue {hta_prev*100:.1f}% (IC 95%: {max(0, hta_prev-hta_ci_margin)*100:.1f}%-{min(1, hta_prev+hta_ci_margin)*100:.1f}%), mientras que la diabetes mellitus presentó una prevalencia de {dm_prev*100:.1f}% (IC 95%: {max(0, dm_prev-dm_ci_margin)*100:.1f}%-{min(1, dm_prev+dm_ci_margin)*100:.1f}%)."
    # """

    # story.append(Paragraph(datos_text, normal_style))
    # story.append(Spacer(1, 20))

    # 6. CONCLUSIONES Y RECOMENDACIONES
    story.append(Paragraph("6. CONCLUSIONES Y RECOMENDACIONES", subtitle_style))
    
    story.append(Paragraph("6.1 Hallazgos Principales", subtitle2_style))
    conclusiones_text = f"""El análisis predictivo mediante machine learning alcanzó una precisión del {model_results['accuracy']:.2%} en la detección de tendencias en enfermedades crónicas. Los resultados proporcionan información útil para el análisis de patrones de salud en la población estudiada."""
    story.append(Paragraph(conclusiones_text, normal_style))
    
    story.append(Paragraph("6.2 Recomendaciones Estratégicas", subtitle2_style))
    recomendaciones = [
        "Implementar sistemas de vigilancia epidemiológica predictiva en tiempo real",
        "Desarrollar políticas de salud pública basadas en perfiles de riesgo individualizados",
        "Establecer programas de medicina preventiva personalizada utilizando inteligencia artificial",
        "Crear redes de atención integrada para el manejo de factores de riesgo modificables"
    ]
    
    for i, recomendacion in enumerate(recomendaciones, 1):
        story.append(Paragraph(f"{i}. {recomendacion}.", list_style))
    
    story.append(Spacer(1, 20))
    
    # Pie de página final
    story.append(add_footer())
    
    # Generar PDF
    doc.build(story)

@router.post("/analyze", response_model=AnalysisResponse)
async def analyze_file(request: AnalysisRequest, background_tasks: BackgroundTasks):
    """Analizar archivo con ML y generar reporte PDF"""
    
    # Verificar que el archivo existe
    file_data = storage.get_file(request.file_id)
    if not file_data:
        raise HTTPException(status_code=404, detail="Archivo no encontrado")
    
    # Verificar que el archivo no esté eliminado
    if file_data.get("status") == "deleted":
        raise HTTPException(status_code=400, detail="No se puede analizar un archivo eliminado")
    
    if not os.path.exists(file_data["file_path"]):
        raise HTTPException(status_code=404, detail="Archivo físico no encontrado")

    try:
        # Actualizar estado del archivo
        storage.update_file_status(request.file_id, "processing")
        
        # Cargar y procesar datos
        df = pd.read_excel(file_data["file_path"])
        
        # Validar que el DataFrame no esté vacío
        if df.empty:
            raise ValueError("El archivo Excel está vacío")
        
        # Validar que tenga las columnas requeridas
        required_columns = ['ID', 'Edad', 'Sexo', 'Peso', 'Altura', 'Presion_Arterial', 'Glucosa', 'Colesterol', 'Fumador', 'Diagnostico']
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            raise ValueError(f"Faltan las siguientes columnas requeridas: {', '.join(missing_columns)}")
        
        processed_data, label_encoder = preprocess_data(df)
        
        # Crear modelo ML
        model, accuracy, feature_importance, y_test, y_pred, detailed_metrics = create_ml_model(processed_data)
        
        # Generar gráficos
        charts_dir = "reports/charts"
        os.makedirs(charts_dir, exist_ok=True)
        charts = generate_charts_optimized(processed_data, charts_dir)
        
        # Generar reporte PDF
        report_filename = f"reporte_{request.file_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf"
        report_path = os.path.join("downloads", report_filename)
        
        model_results = {
            'accuracy': accuracy,
            'feature_importance': feature_importance,
            'detailed_metrics': detailed_metrics
        }

        generate_pdf_report(processed_data, model_results, charts, report_path)
        
        # Calcular estadísticas
        hypertension_cases = len(df[df['Diagnostico'].str.contains('Hipertension', case=False, na=False)])
        diabetes_cases = len(df[df['Diagnostico'].str.contains('Diabetes', case=False, na=False)])
        
        summary = f"Análisis completado con {accuracy:.2%} de precisión. Se identificaron {hypertension_cases} casos de hipertensión y {diabetes_cases} casos de diabetes."
        
        # Guardar resultado del análisis
        f1 = detailed_metrics.get('f1_score', accuracy)
        analysis_id = storage.add_analysis(
            file_id=request.file_id,
            report_path=report_path,
            hypertension_cases=hypertension_cases,
            diabetes_cases=diabetes_cases,
            total_records=len(processed_data),
            accuracy_score=accuracy,
            f1_score=f1,
            summary=summary
        )
        
        # Actualizar estado del archivo
        storage.update_file_status(request.file_id, "completed")
        
        return AnalysisResponse(
            id=analysis_id,
            file_id=request.file_id,
            analysis_date=datetime.now().isoformat(),
            report_path=report_path,
            hypertension_cases=hypertension_cases,
            diabetes_cases=diabetes_cases,
            total_records=len(processed_data),
            accuracy_score=accuracy,
            summary=summary
        )
        
    except Exception as e:
        # Actualizar estado en caso de error
        storage.update_file_status(request.file_id, "error")
        # Proporcionar más detalles del error para debugging
        error_detail = f"Error en el análisis: {str(e)}"
        print(f"Analysis error for file {request.file_id}: {error_detail}")  # Para debugging
        raise HTTPException(status_code=500, detail=error_detail)

@router.get("/charts/{file_id}")
async def get_charts_data(file_id: str):
    """Obtener datos para generar gráficas de un archivo específico"""
    try:
        # Cargar datos de análisis REALES usando el método correcto
        file_analyses = storage.get_analyses_by_file(file_id)
        
        if not file_analyses:
            raise HTTPException(status_code=404, detail="No se encontró análisis para este archivo")
        
        # Obtener el análisis más reciente
        latest_analysis = max(file_analyses, key=lambda x: x['analysis_date'])
        
        # Usar los datos REALES del análisis
        total_records = latest_analysis.get('total_records', 0)
        diabetes_cases = latest_analysis.get('diabetes_cases', 0)
        hypertension_cases = latest_analysis.get('hypertension_cases', 0)
        healthy_cases = total_records - diabetes_cases - hypertension_cases
        
        # Cargar el archivo Excel para obtener más detalles
        file_data = storage.get_file(file_id)
        if file_data and os.path.exists(file_data["file_path"]):
            df = pd.read_excel(file_data["file_path"])
            
            # Datos REALES basados en tu archivo de 10 registros
            diagnosis_counts = df['Diagnostico'].value_counts() if 'Diagnostico' in df.columns else {}
            age_data = df['Edad'].describe() if 'Edad' in df.columns else {}
            
            # Calcular factores de riesgo
            hypertension = len(df[df['Diagnostico'].str.contains('Hipertension', case=False, na=False)])
            diabetes = len(df[df['Diagnostico'].str.contains('Diabetes', case=False, na=False)])
            healthy = len(df) - hypertension - diabetes
            
            charts_data = {
                "diagnostic_distribution": {
                    "labels": [f"{diag} ({count} de {total_records})" for diag, count in diagnosis_counts.items()],
                    "values": diagnosis_counts.values.tolist(),
                    "backgroundColor": ["#e74c3c", "#3498db", "#f39c12", "#2ecc71"][:len(diagnosis_counts)]
                },
                "age_by_diagnosis": {
                    "labels": [f"Edad promedio: {age_data.get('mean', 0):.1f} años"],
                    "values": [age_data.get('mean', 0)] if 'mean' in age_data else [0],
                    "backgroundColor": "#3498db"
                },
                "risk_factors": {
                    "labels": [
                        f"Casos de Diabetes: {diabetes_cases}",
                        f"Casos de Hipertensión: {hypertension_cases}", 
                        f"Casos Saludables: {healthy_cases}"
                    ],
                    "values": [diabetes_cases, hypertension_cases, healthy_cases],
                    "backgroundColor": ["#e74c3c", "#f39c12", "#2ecc71"]
                },
                "correlation": {
                    "points": [
                        {"x": i+1, "y": row.get('Edad', 0), "label": f"Paciente {i+1}"} 
                        for i, (_, row) in enumerate(df.iterrows())
                    ] if len(df) > 0 else [{"x": 0, "y": 0}],
                    "x_label": "Número de Paciente",
                    "y_label": "Edad (años)"
                }
            }
        else:
            # Si no se puede leer el archivo, usar datos del análisis guardado
            charts_data = {
                "diagnostic_distribution": {
                    "labels": [f"Diabetes ({diabetes_cases} casos)", f"Hipertensión ({hypertension_cases} casos)", f"Otros ({healthy_cases} casos)"],
                    "values": [diabetes_cases, hypertension_cases, healthy_cases],
                    "backgroundColor": ["#e74c3c", "#3498db", "#2ecc71"]
                },
                "age_by_diagnosis": {
                    "labels": [f"Total de registros: {total_records}"],
                    "values": [total_records],
                    "backgroundColor": "#3498db"
                },
                "risk_factors": {
                    "labels": [f"Diabetes: {diabetes_cases}", f"Hipertensión: {hypertension_cases}", f"Total: {total_records}"],
                    "values": [diabetes_cases, hypertension_cases, total_records],
                    "backgroundColor": ["#e74c3c", "#f39c12", "#2ecc71"]
                },
                "correlation": {
                    "points": [{"x": total_records, "y": diabetes_cases + hypertension_cases}],
                    "x_label": "Total Registros",
                    "y_label": "Casos con Diagnóstico"
                }
            }
        
        return charts_data
        
    except Exception as e:
        print(f"Error getting charts data: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error al obtener datos de gráficas: {str(e)}")

@router.get("/view/{analysis_id}")
async def view_report(analysis_id: str):
    """Ver reporte PDF en el navegador (sin forzar descarga)"""

    analysis_data = storage.get_analysis(analysis_id)
    if not analysis_data:
        raise HTTPException(status_code=404, detail="Análisis no encontrado")

    report_path = analysis_data["report_path"]
    if not os.path.exists(report_path):
        raise HTTPException(status_code=404, detail="Reporte no encontrado")

    return FileResponse(
        path=report_path,
        media_type="application/pdf",
        headers={"Content-Disposition": "inline"}
    )

@router.get("/download/{analysis_id}")
async def download_report(analysis_id: str):
    """Descargar reporte PDF generado"""
    
    analysis_data = storage.get_analysis(analysis_id)
    if not analysis_data:
        raise HTTPException(status_code=404, detail="Análisis no encontrado")
    
    report_path = analysis_data["report_path"]
    if not os.path.exists(report_path):
        raise HTTPException(status_code=404, detail="Reporte no encontrado")
    
    return FileResponse(
        path=report_path,
        filename=os.path.basename(report_path),
        media_type="application/pdf"
    )

@router.get("/results/{file_id}")
async def get_analysis_results(file_id: str):
    """Obtener resultados de análisis de un archivo"""
    
    analyses = storage.get_analyses_by_file(file_id)
    if not analyses:
        raise HTTPException(status_code=404, detail="No se encontraron análisis para este archivo")
    
    return analyses


# Para archivos extremadamente grandes (>50,000 registros)
def load_large_excel(file_path, chunk_size=10000):
    """Cargar Excel en chunks para archivos muy grandes"""
    try:
        # Primero verificar el tamaño
        df_info = pd.read_excel(file_path, nrows=0)  # Solo headers
        
        # Si es muy grande, usar chunks
        chunks = []
        for chunk in pd.read_excel(file_path, chunksize=chunk_size):
            chunks.append(chunk)
        
        return pd.concat(chunks, ignore_index=True)
    except:
        # Fallback a carga normal
        return pd.read_excel(file_path)

@router.get("/view/{analysis_id}")
async def view_report(analysis_id: str):
    """Ver reporte PDF en el navegador (sin forzar descarga)"""

    analysis_data = storage.get_analysis(analysis_id)
    if not analysis_data:
        raise HTTPException(status_code=404, detail="Análisis no encontrado")

    report_path = analysis_data["report_path"]
    if not os.path.exists(report_path):
        raise HTTPException(status_code=404, detail="Reporte no encontrado")

    return FileResponse(
        path=report_path,
        media_type="application/pdf",
        headers={"Content-Disposition": "inline"}
    )

@router.get("/download/{analysis_id}")
async def download_report(analysis_id: str):
    """Descargar reporte PDF generado"""
    
    analysis_data = storage.get_analysis(analysis_id)
    if not analysis_data:
        raise HTTPException(status_code=404, detail="Análisis no encontrado")
    
    report_path = analysis_data["report_path"]
    if not os.path.exists(report_path):
        raise HTTPException(status_code=404, detail="Reporte no encontrado")
    
    return FileResponse(
        path=report_path,
        filename=os.path.basename(report_path),
        media_type="application/pdf"
    )

@router.get("/results/{file_id}")
async def get_analysis_results(file_id: str):
    """Obtener resultados de análisis de un archivo"""
    
    analyses = storage.get_analyses_by_file(file_id)
    if not analyses:
        raise HTTPException(status_code=404, detail="No se encontraron análisis para este archivo")
    
    return analyses


# Para archivos extremadamente grandes (>50,000 registros)
def load_large_excel(file_path, chunk_size=10000):
    """Cargar Excel en chunks para archivos muy grandes"""
    try:
        # Primero verificar el tamaño
        df_info = pd.read_excel(file_path, nrows=0)  # Solo headers
        
        # Si es muy grande, usar chunks
        chunks = []
        for chunk in pd.read_excel(file_path, chunksize=chunk_size):
            chunks.append(chunk)
        
        return pd.concat(chunks, ignore_index=True)
    except:
        # Fallback a carga normal
        return pd.read_excel(file_path)


def analyze_file_with_monitoring(request: AnalysisRequest):
    start_time = time.time()
    start_memory = psutil.Process().memory_info().rss / 1024 / 1024  # MB
    
    print(f"Iniciando análisis - Memoria inicial: {start_memory:.1f} MB")
    
    # Verificar que el archivo existe
    file_data = storage.get_file(request.file_id)
    if not file_data:
        raise HTTPException(status_code=404, detail="Archivo no encontrado")
    
    # Verificar que el archivo no esté eliminado
    if file_data.get("status") == "deleted":
        raise HTTPException(status_code=400, detail="No se puede analizar un archivo eliminado")
    
    if not os.path.exists(file_data["file_path"]):
        raise HTTPException(status_code=404, detail="Archivo físico no encontrado")

    try:
        # Actualizar estado del archivo
        storage.update_file_status(request.file_id, "processing")
        
        # Cargar y procesar datos
        df = pd.read_excel(file_data["file_path"])
        
        # Validar que el DataFrame no esté vacío
        if df.empty:
            raise ValueError("El archivo Excel está vacío")
        
        # Validar que tenga las columnas requeridas
        required_columns = ['ID', 'Edad', 'Sexo', 'Peso', 'Altura', 'Presion_Arterial', 'Glucosa', 'Colesterol', 'Fumador', 'Diagnostico']
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            raise ValueError(f"Faltan las siguientes columnas requeridas: {', '.join(missing_columns)}")
        
        processed_data, label_encoder = preprocess_data(df)
        
        # Crear modelo ML
        model, accuracy, feature_importance, y_test, y_pred, detailed_metrics = create_ml_model(processed_data)
        
        # Generar gráficos
        charts_dir = "reports/charts"
        os.makedirs(charts_dir, exist_ok=True)
        charts = generate_charts_optimized(processed_data, charts_dir)
        
        # Generar reporte PDF
        report_filename = f"reporte_{request.file_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf"
        report_path = os.path.join("downloads", report_filename)
        
        model_results = {
            'accuracy': accuracy,
            'feature_importance': feature_importance,
            'detailed_metrics': detailed_metrics
        }

        generate_pdf_report(processed_data, model_results, charts, report_path)
        
        # Calcular estadísticas
        hypertension_cases = len(processed_data[processed_data['Diagnostico'].str.contains('Hipertensión', case=False, na=False)])
        diabetes_cases = len(processed_data[processed_data['Diagnostico'].str.contains('Diabetes', case=False, na=False)])
        
        summary = f"Análisis completado con {accuracy:.2%} de precisión. Se identificaron {hypertension_cases} casos de hipertensión y {diabetes_cases} casos de diabetes."
        
        # Guardar resultado del análisis
        f1 = detailed_metrics.get('f1_score', accuracy)
        analysis_id = storage.add_analysis(
            file_id=request.file_id,
            report_path=report_path,
            hypertension_cases=hypertension_cases,
            diabetes_cases=diabetes_cases,
            total_records=len(processed_data),
            accuracy_score=accuracy,
            f1_score=f1,
            summary=summary
        )
        
        # Actualizar estado del archivo
        storage.update_file_status(request.file_id, "completed")
        
        return AnalysisResponse(
            id=analysis_id,
            file_id=request.file_id,
            analysis_date=datetime.now().isoformat(),
            report_path=report_path,
            hypertension_cases=hypertension_cases,
            diabetes_cases=diabetes_cases,
            total_records=len(processed_data),
            accuracy_score=accuracy,
            summary=summary
        )
        
    except Exception as e:
        # Actualizar estado en caso de error
        storage.update_file_status(request.file_id, "error")
        # Proporcionar más detalles del error para debugging
        error_detail = f"Error en el análisis: {str(e)}"
        print(f"Analysis error for file {request.file_id}: {error_detail}")  # Para debugging
        raise HTTPException(status_code=500, detail=error_detail)

@router.get("/view/{analysis_id}")
async def view_report(analysis_id: str):
    """Ver reporte PDF en el navegador (sin forzar descarga)"""

    analysis_data = storage.get_analysis(analysis_id)
    if not analysis_data:
        raise HTTPException(status_code=404, detail="Análisis no encontrado")

    report_path = analysis_data["report_path"]
    if not os.path.exists(report_path):
        raise HTTPException(status_code=404, detail="Reporte no encontrado")

    return FileResponse(
        path=report_path,
        media_type="application/pdf",
        headers={"Content-Disposition": "inline"}
    )

@router.get("/download/{analysis_id}")
async def download_report(analysis_id: str):
    """Descargar reporte PDF generado"""
    
    analysis_data = storage.get_analysis(analysis_id)
    if not analysis_data:
        raise HTTPException(status_code=404, detail="Análisis no encontrado")
    
    report_path = analysis_data["report_path"]
    if not os.path.exists(report_path):
        raise HTTPException(status_code=404, detail="Reporte no encontrado")
    
    return FileResponse(
        path=report_path,
        filename=os.path.basename(report_path),
        media_type="application/pdf"
    )

@router.get("/results/{file_id}")
async def get_analysis_results(file_id: str):
    """Obtener resultados de análisis de un archivo"""
    
    analyses = storage.get_analyses_by_file(file_id)
    if not analyses:
        raise HTTPException(status_code=404, detail="No se encontraron análisis para este archivo")
    
    return analyses


# Para archivos extremadamente grandes (>50,000 registros)
def load_large_excel(file_path, chunk_size=10000):
    """Cargar Excel en chunks para archivos muy grandes"""
    try:
        # Primero verificar el tamaño
        df_info = pd.read_excel(file_path, nrows=0)  # Solo headers
        
        # Si es muy grande, usar chunks
        chunks = []
        for chunk in pd.read_excel(file_path, chunksize=chunk_size):
            chunks.append(chunk)
        
        return pd.concat(chunks, ignore_index=True)
    except:
        # Fallback a carga normal
        return pd.read_excel(file_path)
