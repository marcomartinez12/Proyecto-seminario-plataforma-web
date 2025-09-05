from fastapi import APIRouter, HTTPException, BackgroundTasks
from fastapi.responses import FileResponse
import os
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import LabelEncoder
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
    """Preprocesar los datos para el modelo ML"""
    # Crear una copia para no modificar el original
    data = df.copy()
    
    # Convertir columnas numéricas a float (por si vienen como string)
    numeric_columns = ['Edad', 'Peso', 'Altura', 'Glucosa', 'Colesterol']
    for col in numeric_columns:
        data[col] = pd.to_numeric(data[col], errors='coerce')
    
    # Codificar variables categóricas
    le_sexo = LabelEncoder()
    data['Sexo_encoded'] = le_sexo.fit_transform(data['Sexo'])
    
    # Procesar presión arterial (extraer valores numéricos)
    def extract_pressure_values(pressure_str):
        try:
            if '/' in str(pressure_str):
                systolic, diastolic = str(pressure_str).split('/')
                return float(systolic), float(diastolic)
            else:
                # Si no tiene formato sistólica/diastólica, usar como sistólica
                return float(pressure_str), 80.0  # Valor por defecto para diastólica
        except:
            return 120.0, 80.0  # Valores por defecto
    
    pressure_values = data['Presion_Arterial'].apply(extract_pressure_values)
    data['Presion_Sistolica'] = [p[0] for p in pressure_values]
    data['Presion_Diastolica'] = [p[1] for p in pressure_values]
    
    # Convertir Fumador a numérico (Si=1, No=0)
    data['Fumador_encoded'] = data['Fumador'].map({'Si': 1, 'Sí': 1, 'si': 1, 'sí': 1, 'No': 0, 'no': 0}).fillna(0).astype(int)
    
    # Calcular IMC (convertir altura de cm a metros)
    # Asegurar que Peso y Altura son numéricos antes del cálculo
    data['IMC'] = data['Peso'] / ((data['Altura'] / 100) ** 2)
    
    # Llenar valores NaN con la mediana de cada columna
    for col in numeric_columns + ['IMC', 'Presion_Sistolica', 'Presion_Diastolica']:
        if col in data.columns:
            data[col] = data[col].fillna(data[col].median())
    
    return data, le_sexo

def create_ml_model(data):
    """Crear y entrenar el modelo de ML"""
    # Seleccionar características para el modelo
    features = ['Edad', 'Sexo_encoded', 'Peso', 'Altura', 'IMC', 
                'Presion_Sistolica', 'Presion_Diastolica', 'Glucosa', 
                'Colesterol', 'Fumador_encoded']
    
    X = data[features]
    y = data['Diagnostico']
    
    # Dividir datos en entrenamiento y prueba
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Crear y entrenar el modelo
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    
    # Hacer predicciones
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    
    # Obtener importancia de características
    feature_importance = dict(zip(features, model.feature_importances_))
    
    return model, accuracy, feature_importance, y_test, y_pred

def generate_charts_optimized(data, output_dir, max_points=5000):
    """Generar gráficos con muestreo para datasets grandes"""
    # Si el dataset es muy grande, usar una muestra para gráficos
    if len(data) > max_points:
        sample_data = data.sample(n=max_points, random_state=42)
        print(f"Usando muestra de {max_points} registros para gráficos")
    else:
        sample_data = data
    
    charts = []
    
    # Configurar estilo
    plt.style.use('default')
    sns.set_palette("husl")
    
    # Gráfico 1: Distribución de diagnósticos
    plt.figure(figsize=(10, 6))
    diagnosis_counts = data['Diagnostico'].value_counts()
    plt.pie(diagnosis_counts.values, labels=diagnosis_counts.index, autopct='%1.1f%%')
    plt.title('Distribución de Diagnósticos', fontsize=14, fontweight='bold')
    chart1_path = os.path.join(output_dir, 'distribucion_diagnosticos.png')
    plt.savefig(chart1_path, dpi=300, bbox_inches='tight')
    plt.close()
    charts.append(chart1_path)
    
    # Gráfico 2: Distribución por edad y diagnóstico
    plt.figure(figsize=(12, 6))
    sns.boxplot(data=data, x='Diagnostico', y='Edad')
    plt.title('Distribución de Edad por Diagnóstico', fontsize=14, fontweight='bold')
    plt.xticks(rotation=45)
    chart2_path = os.path.join(output_dir, 'edad_por_diagnostico.png')
    plt.savefig(chart2_path, dpi=300, bbox_inches='tight')
    plt.close()
    charts.append(chart2_path)
    
    # Gráfico 3: Correlación entre variables
    plt.figure(figsize=(10, 8))
    numeric_cols = ['Edad', 'Peso', 'Altura', 'Glucosa', 'Colesterol', 'Presion_Sistolica', 'Presion_Diastolica']
    correlation_matrix = data[numeric_cols].corr()
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0)
    plt.title('Matriz de Correlación de Variables Médicas', fontsize=14, fontweight='bold')
    chart3_path = os.path.join(output_dir, 'correlacion_variables.png')
    plt.savefig(chart3_path, dpi=300, bbox_inches='tight')
    plt.close()
    charts.append(chart3_path)
    
    return charts

def generate_pdf_report(data, model_results, charts, output_path):
    """Generar reporte PDF con análisis médico"""
    doc = SimpleDocTemplate(output_path, pagesize=letter)
    story = []
    
    # Estilos
    styles = getSampleStyleSheet()
    title_style = ParagraphStyle(
        'CustomTitle',
        parent=styles['Heading1'],
        fontSize=16,
        fontName='Times-Bold',
        alignment=1,  # Centrado
        spaceAfter=30
    )
    
    heading_style = ParagraphStyle(
        'CustomHeading',
        parent=styles['Heading2'],
        fontSize=12,
        fontName='Times-Bold',
        spaceAfter=12
    )
    
    normal_style = ParagraphStyle(
        'CustomNormal',
        parent=styles['Normal'],
        fontSize=12,
        fontName='Times-Roman',
        spaceAfter=12
    )
    
    # Título del reporte
    story.append(Paragraph("REPORTE DE ANÁLISIS DE ENFERMEDADES CRÓNICAS", title_style))
    story.append(Spacer(1, 20))
    
    # Información general
    story.append(Paragraph("1. RESUMEN EJECUTIVO", heading_style))
    
    total_records = len(data)
    hypertension_cases = len(data[data['Diagnostico'].str.contains('Hipertensión', case=False, na=False)])
    diabetes_cases = len(data[data['Diagnostico'].str.contains('Diabetes', case=False, na=False)])
    
    summary_text = f"""
    Este reporte presenta el análisis de {total_records} registros médicos utilizando técnicas de 
    machine learning para la detección de tendencias en enfermedades crónicas.
    
    <b>Resultados principales:</b><br/>
    • Total de registros analizados: {total_records}<br/>
    • Casos de hipertensión detectados: {hypertension_cases}<br/>
    • Casos de diabetes detectados: {diabetes_cases}<br/>
    • Precisión del modelo: {model_results['accuracy']:.2%}<br/>
    """
    
    story.append(Paragraph(summary_text, normal_style))
    story.append(Spacer(1, 20))
    
    # Análisis demográfico
    story.append(Paragraph("2. ANÁLISIS DEMOGRÁFICO", heading_style))
    
    avg_age = data['Edad'].mean()
    gender_dist = data['Sexo'].value_counts()
    smokers_pct = (data['Fumador_encoded'].sum() / len(data)) * 100
    
    demographic_text = f"""
    <b>Características de la población estudiada:</b><br/>
    • Edad promedio: {avg_age:.1f} años<br/>
    • Distribución por género: {gender_dist.to_dict()}<br/>
    • Porcentaje de fumadores: {smokers_pct:.1f}%<br/>
    • IMC promedio: {data['IMC'].mean():.1f}<br/>
    """
    
    story.append(Paragraph(demographic_text, normal_style))
    story.append(Spacer(1, 20))
    
    # Insertar gráficos
    story.append(Paragraph("3. ANÁLISIS GRÁFICO", heading_style))
    
    for i, chart_path in enumerate(charts):
        if os.path.exists(chart_path):
            story.append(Image(chart_path, width=6*inch, height=3.6*inch))
            story.append(Spacer(1, 20))
    
    # Análisis de factores de riesgo
    story.append(Paragraph("4. FACTORES DE RIESGO IDENTIFICADOS", heading_style))
    
    # Análisis de presión arterial
    high_bp = len(data[data['Presion_Sistolica'] > 140])
    high_glucose = len(data[data['Glucosa'] > 126])
    high_cholesterol = len(data[data['Colesterol'] > 240])
    
    risk_text = f"""
    <b>Factores de riesgo prevalentes:</b><br/>
    • Presión arterial elevada (>140 mmHg): {high_bp} casos ({(high_bp/total_records)*100:.1f}%)<br/>
    • Glucosa elevada (>126 mg/dL): {high_glucose} casos ({(high_glucose/total_records)*100:.1f}%)<br/>
    • Colesterol alto (>240 mg/dL): {high_cholesterol} casos ({(high_cholesterol/total_records)*100:.1f}%)<br/>
    """
    
    story.append(Paragraph(risk_text, normal_style))
    story.append(Spacer(1, 20))
    
    # Recomendaciones
    story.append(Paragraph("5. RECOMENDACIONES MÉDICAS", heading_style))
    
    recommendations = """
    <b>Basado en el análisis realizado, se recomienda:</b><br/><br/>
    
    1. <b>Prevención primaria:</b> Implementar programas de educación sobre estilos de vida 
    saludables, especialmente en población de riesgo.<br/><br/>
    
    2. <b>Detección temprana:</b> Establecer protocolos de screening regular para diabetes 
    e hipertensión en pacientes con factores de riesgo identificados.<br/><br/>
    
    3. <b>Seguimiento personalizado:</b> Desarrollar planes de seguimiento individualizados 
    basados en el perfil de riesgo de cada paciente.<br/><br/>
    
    4. <b>Intervención en factores modificables:</b> Enfocar esfuerzos en la reducción del 
    tabaquismo, control de peso y mejora de hábitos alimentarios.<br/><br/>
    
    <b>Nota importante:</b> Este análisis es de carácter informativo y no sustituye 
    el criterio médico profesional. Todos los casos requieren evaluación clínica individual.
    """
    
    story.append(Paragraph(recommendations, normal_style))
    
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
        model, accuracy, feature_importance, y_test, y_pred = create_ml_model(processed_data)
        
        # Generar gráficos
        charts_dir = "reports/charts"
        os.makedirs(charts_dir, exist_ok=True)
        charts = generate_charts_optimized(processed_data, charts_dir)
        
        # Generar reporte PDF
        report_filename = f"reporte_{request.file_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf"
        report_path = os.path.join("downloads", report_filename)
        
        model_results = {
            'accuracy': accuracy,
            'feature_importance': feature_importance
        }
        
        generate_pdf_report(processed_data, model_results, charts, report_path)
        
        # Calcular estadísticas
        hypertension_cases = len(processed_data[processed_data['Diagnostico'].str.contains('Hipertensión', case=False, na=False)])
        diabetes_cases = len(processed_data[processed_data['Diagnostico'].str.contains('Diabetes', case=False, na=False)])
        
        summary = f"Análisis completado con {accuracy:.2%} de precisión. Se identificaron {hypertension_cases} casos de hipertensión y {diabetes_cases} casos de diabetes."
        
        # Guardar resultado del análisis
        analysis_id = storage.add_analysis(
            file_id=request.file_id,
            report_path=report_path,
            hypertension_cases=hypertension_cases,
            diabetes_cases=diabetes_cases,
            total_records=len(processed_data),
            accuracy_score=accuracy,
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
        model, accuracy, feature_importance, y_test, y_pred = create_ml_model(processed_data)
        
        # Generar gráficos
        charts_dir = "reports/charts"
        os.makedirs(charts_dir, exist_ok=True)
        charts = generate_charts(processed_data, charts_dir)
        
        # Generar reporte PDF
        report_filename = f"reporte_{request.file_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf"
        report_path = os.path.join("downloads", report_filename)
        
        model_results = {
            'accuracy': accuracy,
            'feature_importance': feature_importance
        }
        
        generate_pdf_report(processed_data, model_results, charts, report_path)
        
        # Calcular estadísticas
        hypertension_cases = len(processed_data[processed_data['Diagnostico'].str.contains('Hipertensión', case=False, na=False)])
        diabetes_cases = len(processed_data[processed_data['Diagnostico'].str.contains('Diabetes', case=False, na=False)])
        
        summary = f"Análisis completado con {accuracy:.2%} de precisión. Se identificaron {hypertension_cases} casos de hipertensión y {diabetes_cases} casos de diabetes."
        
        # Guardar resultado del análisis
        analysis_id = storage.add_analysis(
            file_id=request.file_id,
            report_path=report_path,
            hypertension_cases=hypertension_cases,
            diabetes_cases=diabetes_cases,
            total_records=len(processed_data),
            accuracy_score=accuracy,
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
        model, accuracy, feature_importance, y_test, y_pred = create_ml_model(processed_data)
        
        # Generar gráficos
        charts_dir = "reports/charts"
        os.makedirs(charts_dir, exist_ok=True)
        charts = generate_charts(processed_data, charts_dir)
        
        # Generar reporte PDF
        report_filename = f"reporte_{request.file_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf"
        report_path = os.path.join("downloads", report_filename)
        
        model_results = {
            'accuracy': accuracy,
            'feature_importance': feature_importance
        }
        
        generate_pdf_report(processed_data, model_results, charts, report_path)
        
        # Calcular estadísticas
        hypertension_cases = len(processed_data[processed_data['Diagnostico'].str.contains('Hipertensión', case=False, na=False)])
        diabetes_cases = len(processed_data[processed_data['Diagnostico'].str.contains('Diabetes', case=False, na=False)])
        
        summary = f"Análisis completado con {accuracy:.2%} de precisión. Se identificaron {hypertension_cases} casos de hipertensión y {diabetes_cases} casos de diabetes."
        
        # Guardar resultado del análisis
        analysis_id = storage.add_analysis(
            file_id=request.file_id,
            report_path=report_path,
            hypertension_cases=hypertension_cases,
            diabetes_cases=diabetes_cases,
            total_records=len(processed_data),
            accuracy_score=accuracy,
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