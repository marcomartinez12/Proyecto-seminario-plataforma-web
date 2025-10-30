from fastapi import APIRouter, HTTPException
from fastapi.responses import HTMLResponse
import os
import pandas as pd
import numpy as np
import json
from app.utils import storage

router = APIRouter()

@router.get("/detailed-analysis/{file_id}", response_class=HTMLResponse)
async def get_detailed_analysis(file_id: str):
    """
    Genera una página HTML educativa que muestra paso a paso cómo funciona el modelo XGBoost
    Estilo 'Pedro el Ingeniero' - explicaciones visuales y didácticas
    """

    # Buscar el archivo
    file_data = storage.get_file(file_id)
    if not file_data:
        raise HTTPException(status_code=404, detail="Archivo no encontrado")

    # Buscar análisis del archivo
    analyses = storage.get_analyses_by_file(file_id)
    if not analyses:
        raise HTTPException(status_code=404, detail="No hay análisis disponibles para este archivo")

    # Obtener el último análisis
    latest_analysis = analyses[-1]

    # Cargar el dataframe original
    file_path = file_data['file_path']
    if not os.path.exists(file_path):
        raise HTTPException(status_code=404, detail="Archivo de datos no encontrado")

    # Leer el archivo
    if file_path.endswith('.xlsx') or file_path.endswith('.xls'):
        df = pd.read_excel(file_path)
    elif file_path.endswith('.csv'):
        df = pd.read_csv(file_path)
    else:
        raise HTTPException(status_code=400, detail="Formato de archivo no soportado")

    # Obtener estadísticas básicas
    total_records = len(df)
    columns_count = len(df.columns)
    columns_list = list(df.columns)

    # Distribución de diagnósticos
    if 'Diagnostico' in df.columns:
        diagnosis_dist = df['Diagnostico'].value_counts().to_dict()
    else:
        diagnosis_dist = {}

    # Estadísticas numéricas
    numeric_stats = {}
    for col in df.select_dtypes(include=[np.number]).columns:
        numeric_stats[col] = {
            'mean': float(df[col].mean()),
            'min': float(df[col].min()),
            'max': float(df[col].max()),
            'std': float(df[col].std())
        }

    # Generar HTML educativo
    html_content = generate_educational_html(
        file_name=file_data['original_filename'],
        total_records=total_records,
        columns_count=columns_count,
        columns_list=columns_list,
        diagnosis_dist=diagnosis_dist,
        numeric_stats=numeric_stats,
        analysis_results=latest_analysis
    )

    return HTMLResponse(content=html_content)


def generate_educational_html(file_name, total_records, columns_count, columns_list, diagnosis_dist, numeric_stats, analysis_results):
    """
    Genera el HTML educativo paso a paso estilo 'Pedro el Ingeniero'
    """

    # Calcular métricas del análisis
    accuracy = analysis_results.get('accuracy_score', 0) * 100
    f1_score = analysis_results.get('f1_score', analysis_results.get('accuracy_score', 0)) * 100

    html = f"""
<!DOCTYPE html>
<html lang="es">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Análisis Detallado - {file_name}</title>
    <link href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.0.0/css/all.min.css" rel="stylesheet">
    <style>
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

        * {{
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }}

        body {{
            font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
            background: #0a0a0a;
            color: #e5e5e5;
            padding: 40px 20px;
            line-height: 1.7;
        }}

        .container {{
            max-width: 1100px;
            margin: 0 auto;
        }}

        .header {{
            text-align: center;
            margin-bottom: 60px;
            padding: 50px 40px;
            background: #141414;
            border-radius: 12px;
            border: 1px solid #2a2a2a;
        }}

        .header h1 {{
            font-size: 2.5rem;
            margin-bottom: 12px;
            color: #ffffff;
            font-weight: 600;
            letter-spacing: -0.02em;
        }}

        .header p {{
            font-size: 1.1rem;
            color: #a0a0a0;
            font-weight: 400;
        }}

        .step-card {{
            background: #141414;
            color: #e5e5e5;
            padding: 40px;
            margin-bottom: 24px;
            border-radius: 12px;
            border: 1px solid #2a2a2a;
            position: relative;
            overflow: hidden;
            transition: border-color 0.2s ease;
        }}

        .step-card:hover {{
            border-color: #3a3a3a;
        }}

        .step-number {{
            position: absolute;
            top: -12px;
            left: 30px;
            width: 56px;
            height: 56px;
            background: #1e1e1e;
            color: #ffffff;
            border-radius: 8px;
            display: flex;
            align-items: center;
            justify-content: center;
            font-size: 1.5rem;
            font-weight: 600;
            border: 1px solid #2a2a2a;
        }}

        .step-content {{
            margin-top: 30px;
        }}

        .step-title {{
            font-size: 1.75rem;
            margin-bottom: 20px;
            color: #ffffff;
            display: flex;
            align-items: center;
            gap: 12px;
            font-weight: 600;
        }}

        .step-title i {{
            font-size: 1.75rem;
            color: #a0a0a0;
        }}

        .explanation {{
            font-size: 1rem;
            line-height: 1.75;
            margin-bottom: 24px;
            padding: 20px;
            background: #1a1a1a;
            border-left: 3px solid #3a3a3a;
            border-radius: 8px;
            color: #d0d0d0;
        }}

        .analogy {{
            background: #1e1e1e;
            color: #e5e5e5;
            padding: 20px;
            border-radius: 8px;
            margin: 20px 0;
            border-left: 3px solid #3a3a3a;
        }}

        .analogy-title {{
            font-weight: 600;
            font-size: 1rem;
            margin-bottom: 10px;
            display: flex;
            align-items: center;
            gap: 8px;
            color: #ffffff;
        }}

        .stats-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 16px;
            margin: 24px 0;
        }}

        .stat-box {{
            background: #1a1a1a;
            border: 1px solid #2a2a2a;
            color: #e5e5e5;
            padding: 24px;
            border-radius: 10px;
            text-align: center;
        }}

        .stat-value {{
            font-size: 2.25rem;
            font-weight: 600;
            margin-bottom: 8px;
            color: #ffffff;
        }}

        .stat-label {{
            font-size: 0.875rem;
            color: #a0a0a0;
            font-weight: 400;
        }}

        .code-box {{
            background: #0d0d0d;
            color: #d0d0d0;
            padding: 20px;
            border-radius: 8px;
            font-family: 'JetBrains Mono', 'Courier New', monospace;
            font-size: 0.9rem;
            margin: 20px 0;
            overflow-x: auto;
            border: 1px solid #2a2a2a;
        }}

        .process-flow {{
            display: flex;
            align-items: center;
            justify-content: space-between;
            margin: 30px 0;
            flex-wrap: wrap;
            gap: 12px;
        }}

        .flow-item {{
            flex: 1;
            min-width: 140px;
            background: #1e1e1e;
            color: #e5e5e5;
            padding: 18px;
            border-radius: 8px;
            text-align: center;
            font-weight: 500;
            font-size: 0.9rem;
            border: 1px solid #2a2a2a;
        }}

        .flow-arrow {{
            color: #5a5a5a;
            font-size: 1.5rem;
            margin: 0 8px;
        }}

        .feature-list {{
            list-style: none;
            padding: 0;
        }}

        .feature-item {{
            padding: 14px 16px;
            margin: 8px 0;
            background: #1a1a1a;
            border-radius: 8px;
            border: 1px solid #2a2a2a;
            display: flex;
            align-items: center;
            gap: 12px;
            color: #e5e5e5;
        }}

        .feature-item i {{
            color: #a0a0a0;
            font-size: 1.25rem;
        }}

        .highlight {{
            background: #1e1e1e;
            padding: 2px 6px;
            border-radius: 4px;
            color: #ffffff;
            font-weight: 500;
            border: 1px solid #3a3a3a;
        }}

        .warning-box {{
            background: #2a2410;
            color: #ffc107;
            padding: 18px;
            border-radius: 8px;
            border-left: 3px solid #ffc107;
            margin: 20px 0;
        }}

        .success-box {{
            background: #1a2e1a;
            color: #4ade80;
            padding: 18px;
            border-radius: 8px;
            border-left: 3px solid #4ade80;
            margin: 20px 0;
        }}

        .comparison-table {{
            width: 100%;
            border-collapse: collapse;
            margin: 20px 0;
            background: #1a1a1a;
            border-radius: 8px;
            overflow: hidden;
            border: 1px solid #2a2a2a;
        }}

        .comparison-table th {{
            background: #1e1e1e;
            color: #ffffff;
            padding: 14px;
            text-align: left;
            font-weight: 600;
        }}

        .comparison-table td {{
            padding: 14px;
            border-bottom: 1px solid #2a2a2a;
            font-size: 0.9rem;
            color: #e5e5e5;
        }}

        .comparison-table tr:hover {{
            background: #1e1e1e;
        }}

        @media (max-width: 768px) {{
            .header h1 {{
                font-size: 1.75rem;
            }}

            .step-title {{
                font-size: 1.4rem;
            }}

            .process-flow {{
                flex-direction: column;
            }}

            .flow-arrow {{
                transform: rotate(90deg);
            }}

            .stat-box {{
                padding: 18px;
            }}

            .stat-value {{
                font-size: 2rem;
            }}
        }}
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1><i class="fas fa-brain"></i> Análisis Detallado del Modelo XGBoost</h1>
            <p>Entendiendo el proceso paso a paso - Estilo educativo</p>
            <p style="font-size: 1rem; margin-top: 10px; opacity: 0.8;">Archivo: <strong>{file_name}</strong></p>
        </div>

        <!-- PASO 1: CARGA DE DATOS -->
        <div class="step-card">
            <div class="step-number">1</div>
            <div class="step-content">
                <h2 class="step-title">
                    <i class="fas fa-file-upload"></i>
                    Carga de Datos
                </h2>

                <div class="explanation">
                    <strong>¿Qué sucedió aquí?</strong><br>
                    El sistema leyó tu archivo Excel y cargó <span class="highlight">{total_records:,} registros</span>
                    de pacientes con <span class="highlight">{columns_count} características</span> diferentes.
                </div>

                <div class="analogy">
                    <div class="analogy-title">
                        <i class="fas fa-lightbulb"></i>
                        Analogía simple:
                    </div>
                    Imagina que tienes una biblioteca con {total_records:,} fichas médicas. Cada ficha tiene {columns_count} campos
                    (como nombre, edad, presión arterial, etc.). El modelo lee todas estas fichas para aprender patrones.
                </div>

                <div class="stats-grid">
                    <div class="stat-box">
                        <div class="stat-value">{total_records:,}</div>
                        <div class="stat-label">Registros totales</div>
                    </div>
                    <div class="stat-box">
                        <div class="stat-value">{columns_count}</div>
                        <div class="stat-label">Características</div>
                    </div>
                    <div class="stat-box">
                        <div class="stat-value">{len(diagnosis_dist)}</div>
                        <div class="stat-label">Diagnósticos únicos</div>
                    </div>
                </div>

                <h3 style="margin-top: 30px; color: #667eea;">Columnas detectadas:</h3>
                <ul class="feature-list">
                    {generate_column_items(columns_list[:10])}
                    {f'<li class="feature-item"><i class="fas fa-ellipsis-h"></i> Y {len(columns_list) - 10} columnas más...</li>' if len(columns_list) > 10 else ''}
                </ul>
            </div>
        </div>

        <!-- PASO 2: PREPROCESAMIENTO -->
        <div class="step-card">
            <div class="step-number">2</div>
            <div class="step-content">
                <h2 class="step-title">
                    <i class="fas fa-broom"></i>
                    Limpieza y Preprocesamiento
                </h2>

                <div class="explanation">
                    <strong>¿Por qué es necesario esto?</strong><br>
                    Los modelos de Machine Learning no entienden texto como "Masculino" o "Femenino".
                    Necesitamos convertir todo a números y crear nuevas características útiles.
                </div>

                <div class="analogy">
                    <div class="analogy-title">
                        <i class="fas fa-lightbulb"></i>
                        Analogía simple:
                    </div>
                    Es como preparar ingredientes antes de cocinar. No puedes cocinar con huevos sin cáscara,
                    ¿verdad? Aquí "pelamos" los datos: convertimos texto a números, eliminamos valores raros,
                    y creamos nuevas medidas útiles (como calcular el IMC a partir de peso y altura).
                </div>

                <h3 style="margin-top: 30px; color: #667eea;">Transformaciones realizadas:</h3>

                <div class="process-flow">
                    <div class="flow-item">Datos crudos</div>
                    <div class="flow-arrow"><i class="fas fa-arrow-right"></i></div>
                    <div class="flow-item">Convertir texto a números</div>
                    <div class="flow-arrow"><i class="fas fa-arrow-right"></i></div>
                    <div class="flow-item">Crear nuevas features</div>
                    <div class="flow-arrow"><i class="fas fa-arrow-right"></i></div>
                    <div class="flow-item">Datos listos</div>
                </div>

                <div class="code-box">
                    <strong>Ejemplo de transformación:</strong><br><br>
                    Sexo: "Masculino" → 1<br>
                    Sexo: "Femenino" → 0<br><br>
                    IMC = Peso (kg) / (Altura (m))²<br>
                    Edad > 60 → Adulto_Mayor = 1
                </div>

                <div class="warning-box">
                    <strong><i class="fas fa-exclamation-triangle"></i> Importante:</strong><br>
                    También se eliminaron valores imposibles (como edades negativas o presiones de 0 mmHg)
                    para que el modelo aprenda correctamente.
                </div>
            </div>
        </div>

        <!-- PASO 3: DIVISIÓN DE DATOS -->
        <div class="step-card">
            <div class="step-number">3</div>
            <div class="step-content">
                <h2 class="step-title">
                    <i class="fas fa-cut"></i>
                    División: Entrenamiento vs Prueba
                </h2>

                <div class="explanation">
                    <strong>¿Por qué dividir los datos?</strong><br>
                    Usamos el <span class="highlight">80%</span> de los datos para enseñar al modelo,
                    y guardamos el <span class="highlight">20%</span> para probarlo después.
                    Es como hacer un examen: no puedes usar las mismas preguntas que usaste para estudiar.
                </div>

                <div class="analogy">
                    <div class="analogy-title">
                        <i class="fas fa-lightbulb"></i>
                        Analogía simple:
                    </div>
                    Imagina que estás enseñando a un niño a reconocer frutas. Le muestras 80 manzanas para que aprenda,
                    pero guardas 20 manzanas que nunca ha visto. Luego, le muestras esas 20 para ver si realmente aprendió
                    o solo memorizó las primeras 80.
                </div>

                <div class="stats-grid">
                    <div class="stat-box" style="background: linear-gradient(135deg, #11998e, #38ef7d);">
                        <div class="stat-value">{int(total_records * 0.8):,}</div>
                        <div class="stat-label" style="color: white;">Datos de entrenamiento (80%)</div>
                    </div>
                    <div class="stat-box" style="background: linear-gradient(135deg, #eb3349, #f45c43);">
                        <div class="stat-value">{int(total_records * 0.2):,}</div>
                        <div class="stat-label" style="color: white;">Datos de prueba (20%)</div>
                    </div>
                </div>

                <div class="success-box">
                    <strong><i class="fas fa-check-circle"></i> Buena práctica:</strong><br>
                    Esta división se hace de forma <strong>aleatoria</strong> pero <strong>estratificada</strong>,
                    lo que significa que mantenemos la misma proporción de cada diagnóstico en ambos grupos.
                </div>
            </div>
        </div>

        <!-- PASO 4: BALANCEO CON SMOTE -->
        <div class="step-card">
            <div class="step-number">4</div>
            <div class="step-content">
                <h2 class="step-title">
                    <i class="fas fa-balance-scale"></i>
                    Balanceo de Clases (SMOTE)
                </h2>

                <div class="explanation">
                    <strong>¿Qué es SMOTE?</strong><br>
                    SMOTE (Synthetic Minority Over-sampling Technique) es una técnica que <strong>crea ejemplos sintéticos</strong>
                    de las clases minoritarias para balancear el dataset.
                </div>

                <div class="analogy">
                    <div class="analogy-title">
                        <i class="fas fa-lightbulb"></i>
                        Analogía simple:
                    </div>
                    Imagina que tienes 1000 fotos de gatos y solo 100 de perros. Si entrenas un modelo así,
                    aprenderá a decir "gato" casi siempre. SMOTE crea 900 fotos sintéticas de perros (no copias,
                    sino nuevas combinaciones realistas) para que el modelo aprenda ambos por igual.
                </div>

                <div class="process-flow">
                    <div class="flow-item" style="background: #e74c3c;">Clases desbalanceadas</div>
                    <div class="flow-arrow"><i class="fas fa-arrow-right"></i></div>
                    <div class="flow-item" style="background: #f39c12;">SMOTE genera ejemplos sintéticos</div>
                    <div class="flow-arrow"><i class="fas fa-arrow-right"></i></div>
                    <div class="flow-item" style="background: #27ae60;">Clases balanceadas</div>
                </div>

                <div class="code-box">
                    <strong>Cómo funciona SMOTE:</strong><br><br>
                    1. Toma un ejemplo de la clase minoritaria<br>
                    2. Encuentra sus vecinos más cercanos<br>
                    3. Crea un nuevo ejemplo "intermedio" entre ellos<br>
                    4. Repite hasta balancear las clases
                </div>
            </div>
        </div>

        <!-- PASO 5: ENTRENAMIENTO XGBOOST -->
        <div class="step-card">
            <div class="step-number">5</div>
            <div class="step-content">
                <h2 class="step-title">
                    <i class="fas fa-brain"></i>
                    Entrenamiento del Modelo XGBoost
                </h2>

                <div class="explanation">
                    <strong>¿Qué es XGBoost?</strong><br>
                    XGBoost (eXtreme Gradient Boosting) es un algoritmo que crea <span class="highlight">300 árboles de decisión</span>
                    que trabajan en equipo. Cada árbol aprende de los errores del anterior.
                </div>

                <div class="analogy">
                    <div class="analogy-title">
                        <i class="fas fa-lightbulb"></i>
                        Analogía simple:
                    </div>
                    Imagina 300 médicos especializados votando un diagnóstico. El primer médico hace su predicción,
                    el segundo ve dónde se equivocó el primero y corrige, el tercero mejora sobre los dos anteriores,
                    y así sucesivamente. Al final, los 300 votan y la mayoría gana. ¡Por eso es tan preciso!
                </div>

                <h3 style="margin-top: 30px; color: #667eea;">Configuración del modelo:</h3>

                <table class="comparison-table">
                    <thead>
                        <tr>
                            <th>Parámetro</th>
                            <th>Valor</th>
                            <th>¿Qué significa?</th>
                        </tr>
                    </thead>
                    <tbody>
                        <tr>
                            <td><strong>n_estimators</strong></td>
                            <td>300</td>
                            <td>Número de árboles de decisión</td>
                        </tr>
                        <tr>
                            <td><strong>learning_rate</strong></td>
                            <td>0.05</td>
                            <td>Qué tan rápido aprende (bajo = más cuidadoso)</td>
                        </tr>
                        <tr>
                            <td><strong>max_depth</strong></td>
                            <td>8</td>
                            <td>Profundidad máxima de cada árbol</td>
                        </tr>
                        <tr>
                            <td><strong>subsample</strong></td>
                            <td>0.8</td>
                            <td>Usa 80% de datos en cada árbol (evita sobreajuste)</td>
                        </tr>
                        <tr>
                            <td><strong>colsample_bytree</strong></td>
                            <td>0.8</td>
                            <td>Usa 80% de características en cada árbol</td>
                        </tr>
                    </tbody>
                </table>

                <div class="warning-box">
                    <strong><i class="fas fa-exclamation-triangle"></i> Importante:</strong><br>
                    Estos parámetros fueron elegidos cuidadosamente para maximizar la precisión sin sobreajustar.
                    Un modelo "sobreajustado" es como un estudiante que memoriza respuestas en vez de entender conceptos.
                </div>
            </div>
        </div>

        <!-- PASO 6: VALIDACIÓN CRUZADA -->
        <div class="step-card">
            <div class="step-number">6</div>
            <div class="step-content">
                <h2 class="step-title">
                    <i class="fas fa-check-double"></i>
                    Validación Cruzada (5-Fold)
                </h2>

                <div class="explanation">
                    <strong>¿Qué es la validación cruzada?</strong><br>
                    El modelo se entrena y evalúa <span class="highlight">5 veces diferentes</span>
                    usando distintas particiones de datos. Luego se promedian los resultados.
                </div>

                <div class="analogy">
                    <div class="analogy-title">
                        <i class="fas fa-lightbulb"></i>
                        Analogía simple:
                    </div>
                    Es como tomar 5 exámenes diferentes en lugar de solo 1. Divide los datos en 5 partes,
                    usa 4 partes para entrenar y 1 para evaluar. Repite esto 5 veces (cada vez con una parte diferente
                    como evaluación). Así tienes una medida más confiable de qué tan bien funciona el modelo.
                </div>

                <div class="process-flow">
                    <div class="flow-item">Dividir en 5 partes</div>
                    <div class="flow-arrow"><i class="fas fa-arrow-right"></i></div>
                    <div class="flow-item">Entrenar 5 veces</div>
                    <div class="flow-arrow"><i class="fas fa-arrow-right"></i></div>
                    <div class="flow-item">Promediar resultados</div>
                </div>

                <div class="success-box">
                    <strong><i class="fas fa-check-circle"></i> Beneficio:</strong><br>
                    Esto nos da una estimación más realista y confiable del rendimiento del modelo.
                    Si funciona bien en las 5 pruebas, sabemos que es robusto.
                </div>
            </div>
        </div>

        <!-- PASO 7: PREDICCIONES -->
        <div class="step-card">
            <div class="step-number">7</div>
            <div class="step-content">
                <h2 class="step-title">
                    <i class="fas fa-magic"></i>
                    Hacer Predicciones
                </h2>

                <div class="explanation">
                    <strong>¿Cómo predice el modelo?</strong><br>
                    Una vez entrenado, el modelo toma los datos del 20% que guardamos (que nunca ha visto)
                    y predice el diagnóstico de cada paciente.
                </div>

                <div class="analogy">
                    <div class="analogy-title">
                        <i class="fas fa-lightbulb"></i>
                        Analogía simple:
                    </div>
                    El modelo ya "estudió" con el 80% de casos. Ahora le muestras casos nuevos (el 20% restante)
                    y le preguntas: "¿Este paciente tiene diabetes, hipertensión o está normal?".
                    El modelo analiza edad, glucosa, presión, etc., y da su respuesta.
                </div>

                <div class="code-box">
                    <strong>Ejemplo de predicción:</strong><br><br>
                    Paciente: Edad=55, Glucosa=140, Presión=150, IMC=32<br><br>

                    Árbol 1: "Creo que es Hipertensión" (80% confianza)<br>
                    Árbol 2: "Creo que es Hipertensión" (75% confianza)<br>
                    Árbol 3: "Creo que es Diabetes" (60% confianza)<br>
                    ...<br>
                    Árbol 300: "Creo que es Hipertensión" (85% confianza)<br><br>

                    <strong>Votación final: Hipertensión ✓</strong>
                </div>
            </div>
        </div>

        <!-- PASO 8: EVALUACIÓN -->
        <div class="step-card">
            <div class="step-number">8</div>
            <div class="step-content">
                <h2 class="step-title">
                    <i class="fas fa-chart-line"></i>
                    Evaluación del Rendimiento
                </h2>

                <div class="explanation">
                    <strong>¿Cómo sabemos si es bueno?</strong><br>
                    Comparamos las predicciones del modelo con los diagnósticos reales y calculamos métricas de rendimiento.
                </div>

                <div class="stats-grid">
                    <div class="stat-box" style="background: linear-gradient(135deg, #f093fb, #f5576c);">
                        <div class="stat-value">{accuracy:.1f}%</div>
                        <div class="stat-label" style="color: white;">Precisión (Accuracy)</div>
                    </div>
                    <div class="stat-box" style="background: linear-gradient(135deg, #4facfe, #00f2fe);">
                        <div class="stat-value">{f1_score:.1f}%</div>
                        <div class="stat-label" style="color: white;">F1-Score</div>
                    </div>
                </div>

                <div class="analogy">
                    <div class="analogy-title">
                        <i class="fas fa-lightbulb"></i>
                        ¿Qué significan estas métricas?
                    </div>
                    <strong>Precisión (Accuracy):</strong> De 100 predicciones, {accuracy:.0f} fueron correctas.<br><br>
                    <strong>F1-Score:</strong> Es un promedio balanceado que considera tanto aciertos como errores.
                    Un F1 alto significa que el modelo no solo acierta mucho, sino que también evita errores graves.
                </div>

                <h3 style="margin-top: 30px; color: #667eea;">Matriz de Confusión:</h3>
                <div class="explanation">
                    La matriz de confusión muestra exactamente qué predijo el modelo vs qué era la realidad.
                    Los valores en la diagonal (arriba-izquierda a abajo-derecha) son los <strong>aciertos</strong>.
                    Los demás son <strong>errores</strong>.
                </div>

                <div class="success-box">
                    <strong><i class="fas fa-trophy"></i> Resultado:</strong><br>
                    Con una precisión de <strong>{accuracy:.1f}%</strong>, este modelo está listo para ayudar
                    en la detección temprana de diabetes e hipertensión.
                    {get_performance_message(accuracy)}
                </div>
            </div>
        </div>

        <!-- PASO 9: IMPORTANCIA DE CARACTERÍSTICAS -->
        <div class="step-card">
            <div class="step-number">9</div>
            <div class="step-content">
                <h2 class="step-title">
                    <i class="fas fa-star"></i>
                    Características Más Importantes
                </h2>

                <div class="explanation">
                    <strong>¿Qué características influyeron más?</strong><br>
                    XGBoost puede decirnos cuáles variables fueron más importantes para hacer predicciones.
                </div>

                <div class="analogy">
                    <div class="analogy-title">
                        <i class="fas fa-lightbulb"></i>
                        Analogía simple:
                    </div>
                    Es como preguntarle al modelo: "¿En qué te fijaste más para decidir?".
                    El modelo responde: "Me fijé principalmente en la glucosa, luego en la presión arterial,
                    luego en la edad, etc." Esto nos ayuda a entender qué factores son más predictivos.
                </div>

                <div class="warning-box">
                    <strong><i class="fas fa-info-circle"></i> Nota médica:</strong><br>
                    Las características más importantes según el modelo coinciden con el conocimiento médico:
                    niveles altos de glucosa predicen diabetes, presión arterial alta predice hipertensión, etc.
                </div>
            </div>
        </div>

        <!-- PASO 10: REPORTE FINAL -->
        <div class="step-card">
            <div class="step-number">10</div>
            <div class="step-content">
                <h2 class="step-title">
                    <i class="fas fa-file-pdf"></i>
                    Generación del Reporte PDF
                </h2>

                <div class="explanation">
                    <strong>¿Qué incluye el reporte?</strong><br>
                    El sistema genera un PDF profesional con todas las visualizaciones, estadísticas y conclusiones del análisis.
                </div>

                <ul class="feature-list">
                    <li class="feature-item">
                        <i class="fas fa-check-circle"></i>
                        <span>Portada profesional con identificador único</span>
                    </li>
                    <li class="feature-item">
                        <i class="fas fa-check-circle"></i>
                        <span>Resumen ejecutivo con estadísticas clave</span>
                    </li>
                    <li class="feature-item">
                        <i class="fas fa-check-circle"></i>
                        <span>Gráficos de distribución y tendencias</span>
                    </li>
                    <li class="feature-item">
                        <i class="fas fa-check-circle"></i>
                        <span>Matriz de confusión del modelo</span>
                    </li>
                    <li class="feature-item">
                        <i class="fas fa-check-circle"></i>
                        <span>Importancia de características (Top 10)</span>
                    </li>
                    <li class="feature-item">
                        <i class="fas fa-check-circle"></i>
                        <span>Análisis de comorbilidades</span>
                    </li>
                    <li class="feature-item">
                        <i class="fas fa-check-circle"></i>
                        <span>Conclusiones y recomendaciones</span>
                    </li>
                </ul>

                <div class="success-box">
                    <strong><i class="fas fa-check-circle"></i> ¡Proceso completado!</strong><br>
                    Ya tienes un análisis completo con Machine Learning de última generación.
                    El modelo XGBoost ha procesado {total_records:,} registros y está listo para ayudar
                    en la detección de enfermedades crónicas.
                </div>
            </div>
        </div>

        <!-- CONCLUSIÓN -->
        <!-- SIMULADOR DE 3 ESCENARIOS - BOTÓN -->
        <div class="step-card">
            <div class="step-number">8</div>
            <div class="step-content">
                <h2 class="step-title">
                    <i class="fas fa-flask"></i>
                    Simulación de Escenarios
                </h2>

                <div class="explanation">
                    <strong>¿Qué pasaría si los pacientes cambiaran su estilo de vida?</strong><br>
                    Simula 3 escenarios diferentes usando el modelo entrenado para predecir el impacto
                    de cambios en los hábitos de salud.
                </div>

                <div style="text-align: center; margin-top: 30px;">
                    <button id="openScenariosBtn" onclick="openScenariosModal()" style="
                        background: linear-gradient(135deg, #667eea, #764ba2);
                        color: white;
                        border: none;
                        padding: 20px 50px;
                        font-size: 18px;
                        font-weight: 600;
                        border-radius: 12px;
                        cursor: pointer;
                        box-shadow: 0 10px 30px rgba(102, 126, 234, 0.4);
                        transition: all 0.3s ease;
                        display: inline-flex;
                        align-items: center;
                        gap: 12px;
                    ">
                        <i class="fas fa-play-circle" style="font-size: 24px;"></i>
                        <span>Simular Escenarios</span>
                    </button>
                </div>

                <div class="success-box" style="margin-top: 30px;">
                    <strong><i class="fas fa-info-circle"></i> Información:</strong><br>
                    La simulación calcula predicciones basadas en el <strong>perfil promedio de pacientes con diagnósticos de riesgo</strong>
                    (Hipertensión, Diabetes, Prediabetes, Obesidad). Los cambios mostrados son de carácter educativo y
                    <strong>pueden no aplicar igual para todos los pacientes</strong>. Cada persona requiere evaluación médica individualizada.
                </div>
            </div>
        </div>

        <!-- MODAL DE ESCENARIOS -->
        <div id="scenariosModal" style="
            display: none;
            position: fixed;
            top: 0;
            left: 0;
            width: 100%;
            height: 100%;
            background: rgba(0, 0, 0, 0.8);
            z-index: 10000;
            overflow-y: auto;
            animation: fadeIn 0.3s ease;
        ">
            <div style="
                max-width: 1400px;
                margin: 50px auto;
                background: #1a1a2e;
                border-radius: 20px;
                box-shadow: 0 20px 60px rgba(0, 0, 0, 0.5);
                position: relative;
                animation: slideDown 0.4s ease;
            ">
                <!-- Header del Modal -->
                <div style="
                    background: linear-gradient(135deg, #667eea, #764ba2);
                    padding: 30px;
                    border-radius: 20px 20px 0 0;
                    display: flex;
                    justify-content: space-between;
                    align-items: center;
                ">
                    <div>
                        <h2 style="margin: 0; color: white; font-size: 28px; font-weight: 700;">
                            <i class="fas fa-flask"></i>
                            Simulación de Escenarios
                        </h2>
                        <p style="margin: 8px 0 0 0; color: rgba(255,255,255,0.9); font-size: 14px;">
                            ¿Qué pasaría si los pacientes cambiaran su estilo de vida?
                        </p>
                    </div>
                    <button onclick="closeScenariosModal()" style="
                        background: rgba(255, 255, 255, 0.2);
                        border: none;
                        color: white;
                        width: 40px;
                        height: 40px;
                        border-radius: 50%;
                        cursor: pointer;
                        font-size: 20px;
                        transition: all 0.3s ease;
                    " onmouseover="this.style.background='rgba(255,255,255,0.3)'" onmouseout="this.style.background='rgba(255,255,255,0.2)'">
                        <i class="fas fa-times"></i>
                    </button>
                </div>

                <!-- Contenido del Modal -->
                <div style="padding: 40px;">
                    <!-- Loading State -->
                    <div id="modal-scenarios-loading" style="text-align: center; padding: 60px 0;">
                        <div style="display: inline-block; position: relative;">
                            <div style="
                                width: 80px;
                                height: 80px;
                                border: 8px solid #333;
                                border-top: 8px solid #667eea;
                                border-radius: 50%;
                                animation: spin 1s linear infinite;
                            "></div>
                            <div style="
                                position: absolute;
                                top: 50%;
                                left: 50%;
                                transform: translate(-50%, -50%);
                            ">
                                <i class="fas fa-flask" style="font-size: 32px; color: #667eea;"></i>
                            </div>
                        </div>
                        <p style="color: #999; margin-top: 25px; font-size: 18px; font-weight: 500;">Calculando escenarios...</p>
                        <p style="color: #666; margin-top: 10px; font-size: 14px;">Esto puede tomar unos segundos</p>
                    </div>

                    <!-- Scenarios Content -->
                    <div id="modal-scenarios-content" style="display: none;">

                        <!-- Nota informativa -->
                        <div style="
                            background: linear-gradient(135deg, #667eea22, #764ba222);
                            border-left: 4px solid #667eea;
                            padding: 20px;
                            border-radius: 8px;
                            margin-bottom: 25px;
                        ">
                            <div style="display: flex; align-items: start; gap: 15px;">
                                <i class="fas fa-info-circle" style="color: #667eea; font-size: 24px; margin-top: 3px;"></i>
                                <div>
                                    <strong style="color: #667eea; font-size: 16px; display: block; margin-bottom: 8px;">
                                        Nota Importante
                                    </strong>
                                    <p style="color: #bbb; margin: 0; line-height: 1.6; font-size: 14px;">
                                        Esta simulación está basada en el <strong>perfil promedio de pacientes con diagnósticos de riesgo</strong>
                                        (Hipertensión, Diabetes, Prediabetes, Obesidad, Síndrome Metabólico). Los cambios recomendados
                                        pueden <strong>no aplicar de la misma manera para todos los pacientes</strong>, ya que cada persona tiene
                                        necesidades específicas según su condición particular. Esta herramienta es de carácter <strong>educativo
                                        e informativo</strong> y no reemplaza la evaluación médica profesional.
                                    </p>
                                </div>
                            </div>
                        </div>

                        <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(350px, 1fr)); gap: 30px;">

                        <!-- ESCENARIO 1: SIN CAMBIOS -->
                        <div class="scenario-box" style="background: linear-gradient(135deg, #f59e0b, #d97706); padding: 30px; border-radius: 12px; color: white; box-shadow: 0 10px 40px rgba(245, 158, 11, 0.3);">
                            <div style="text-align: center; margin-bottom: 20px;">
                                <i class="fas fa-minus-circle" style="font-size: 48px; opacity: 0.9;"></i>
                            </div>
                            <h3 style="text-align: center; font-size: 22px; margin-bottom: 15px; font-weight: 700;" id="escenario1-titulo">Sin Cambios</h3>
                            <div style="text-align: center; font-size: 48px; font-weight: 700; margin: 20px 0;" id="escenario1-riesgo">--</div>
                            <p style="text-align: center; font-size: 14px; opacity: 0.9; margin-bottom: 20px;" id="escenario1-desc">Manteniendo el estilo de vida actual</p>
                            <div style="background: rgba(0,0,0,0.2); padding: 15px; border-radius: 8px;">
                                <div style="font-size: 13px; opacity: 0.9;" id="escenario1-cambios">
                                    • Sin modificaciones
                                </div>
                            </div>
                        </div>

                        <!-- ESCENARIO 2: CAMBIOS MODERADOS -->
                        <div class="scenario-box" style="background: linear-gradient(135deg, #10b981, #059669); padding: 30px; border-radius: 12px; color: white; box-shadow: 0 10px 40px rgba(16, 185, 129, 0.3);">
                            <div style="text-align: center; margin-bottom: 20px;">
                                <i class="fas fa-check-circle" style="font-size: 48px; opacity: 0.9;"></i>
                            </div>
                            <h3 style="text-align: center; font-size: 22px; margin-bottom: 15px; font-weight: 700;" id="escenario2-titulo">Cambios Moderados</h3>
                            <div style="text-align: center; font-size: 48px; font-weight: 700; margin: 20px 0;" id="escenario2-riesgo">--</div>
                            <p style="text-align: center; font-size: 14px; opacity: 0.9; margin-bottom: 20px;" id="escenario2-desc">Mejoras alcanzables</p>
                            <div style="background: rgba(0,0,0,0.2); padding: 15px; border-radius: 8px;">
                                <div style="font-size: 13px; opacity: 0.9; line-height: 1.8;" id="escenario2-cambios">
                                    Cargando...
                                </div>
                            </div>
                            <div style="text-align: center; margin-top: 15px; padding: 10px; background: rgba(255,255,255,0.2); border-radius: 6px;">
                                <div style="font-size: 12px; opacity: 0.8;">Mejora potencial</div>
                                <div style="font-size: 24px; font-weight: 700;" id="escenario2-mejora">--</div>
                            </div>
                        </div>

                        <!-- ESCENARIO 3: CAMBIOS ÓPTIMOS -->
                        <div class="scenario-box" style="background: linear-gradient(135deg, #667eea, #764ba2); padding: 30px; border-radius: 12px; color: white; box-shadow: 0 10px 40px rgba(102, 126, 234, 0.3);">
                            <div style="text-align: center; margin-bottom: 20px;">
                                <i class="fas fa-star" style="font-size: 48px; opacity: 0.9;"></i>
                            </div>
                            <h3 style="text-align: center; font-size: 22px; margin-bottom: 15px; font-weight: 700;" id="escenario3-titulo">Cambios Óptimos</h3>
                            <div style="text-align: center; font-size: 48px; font-weight: 700; margin: 20px 0;" id="escenario3-riesgo">--</div>
                            <p style="text-align: center; font-size: 14px; opacity: 0.9; margin-bottom: 20px;" id="escenario3-desc">Máxima mejora posible</p>
                            <div style="background: rgba(0,0,0,0.2); padding: 15px; border-radius: 8px;">
                                <div style="font-size: 13px; opacity: 0.9; line-height: 1.8;" id="escenario3-cambios">
                                    Cargando...
                                </div>
                            </div>
                            <div style="text-align: center; margin-top: 15px; padding: 10px; background: rgba(255,255,255,0.2); border-radius: 6px;">
                                <div style="font-size: 12px; opacity: 0.8;">Mejora potencial</div>
                                <div style="font-size: 24px; font-weight: 700;" id="escenario3-mejora">--</div>
                            </div>
                        </div>

                    </div>

                        <div class="success-box" style="margin-top: 30px;">
                            <strong><i class="fas fa-lightbulb"></i> Interpretación:</strong><br>
                            Estos escenarios muestran cómo cambios en el estilo de vida pueden impactar el riesgo de enfermedad.
                            Los cambios moderados son más fáciles de mantener a largo plazo, mientras que los óptimos requieren
                            mayor compromiso pero ofrecen la mejor reducción de riesgo.
                        </div>
                    </div>
                </div>
            </div>
        </div>

        <style>
            @keyframes fadeIn {{
                from {{ opacity: 0; }}
                to {{ opacity: 1; }}
            }}

            @keyframes slideDown {{
                from {{
                    opacity: 0;
                    transform: translateY(-50px);
                }}
                to {{
                    opacity: 1;
                    transform: translateY(0);
                }}
            }}

            @keyframes spin {{
                0% {{ transform: rotate(0deg); }}
                100% {{ transform: rotate(360deg); }}
            }}

            @keyframes pulse {{
                0%, 100% {{ transform: scale(1); }}
                50% {{ transform: scale(1.05); }}
            }}

            #openScenariosBtn:hover {{
                transform: translateY(-2px);
                box-shadow: 0 15px 40px rgba(102, 126, 234, 0.5);
            }}

            #openScenariosBtn:active {{
                transform: translateY(0);
            }}

            .scenario-box {{
                transition: all 0.3s ease;
                animation: slideDown 0.5s ease;
            }}

            .scenario-box:hover {{
                transform: translateY(-5px);
                box-shadow: 0 15px 50px rgba(0, 0, 0, 0.4) !important;
            }}
        </style>

        <script>
            let scenariosDataCache = null;

            function openScenariosModal() {{
                const modal = document.getElementById('scenariosModal');
                modal.style.display = 'block';
                document.body.style.overflow = 'hidden'; // Prevent background scroll

                // Si ya tenemos los datos cargados, mostrarlos inmediatamente
                if (scenariosDataCache) {{
                    displayScenarios(scenariosDataCache);
                }} else {{
                    // Si no, cargar los datos
                    loadScenarios();
                }}
            }}

            function closeScenariosModal() {{
                const modal = document.getElementById('scenariosModal');
                modal.style.display = 'none';
                document.body.style.overflow = ''; // Restore scroll
            }}

            // Cerrar modal al hacer clic fuera del contenido
            document.addEventListener('click', function(event) {{
                const modal = document.getElementById('scenariosModal');
                if (event.target === modal) {{
                    closeScenariosModal();
                }}
            }});

            // Cerrar modal con tecla ESC
            document.addEventListener('keydown', function(event) {{
                if (event.key === 'Escape') {{
                    closeScenariosModal();
                }}
            }});

            async function loadScenarios() {{
                try {{
                    // Obtener file_id de la URL
                    const urlParts = window.location.pathname.split('/');
                    const fileId = urlParts[urlParts.length - 1];

                    const response = await fetch(`/api/scenarios/${{fileId}}`);
                    if (!response.ok) throw new Error('Error al cargar escenarios');

                    const data = await response.json();
                    scenariosDataCache = data; // Guardar en cache

                    displayScenarios(data);

                }} catch (error) {{
                    console.error('Error:', error);
                    document.getElementById('modal-scenarios-loading').innerHTML = `
                        <div style="color: #f5576c; text-align: center; padding: 40px;">
                            <i class="fas fa-exclamation-circle" style="font-size: 64px;"></i>
                            <h3 style="margin-top: 20px; font-size: 22px;">Error al cargar los escenarios</h3>
                            <p style="color: #999; margin-top: 10px;">${{error.message}}</p>
                            <button onclick="loadScenarios()" style="
                                margin-top: 20px;
                                background: #667eea;
                                color: white;
                                border: none;
                                padding: 12px 30px;
                                border-radius: 8px;
                                cursor: pointer;
                                font-size: 14px;
                                font-weight: 600;
                            ">
                                <i class="fas fa-redo"></i> Reintentar
                            </button>
                        </div>
                    `;
                }}
            }}

            function displayScenarios(data) {{
                // Ocultar loading, mostrar content
                document.getElementById('modal-scenarios-loading').style.display = 'none';
                document.getElementById('modal-scenarios-content').style.display = 'block';

                // ESCENARIO 1: Sin cambios
                document.getElementById('escenario1-titulo').textContent = data.escenario_actual.titulo;
                document.getElementById('escenario1-riesgo').textContent = data.escenario_actual.riesgo.toFixed(1) + '%';
                document.getElementById('escenario1-desc').textContent = data.escenario_actual.descripcion;

                // ESCENARIO 2: Cambios moderados
                document.getElementById('escenario2-titulo').textContent = data.escenario_moderado.titulo;
                document.getElementById('escenario2-riesgo').textContent = data.escenario_moderado.riesgo.toFixed(1) + '%';
                document.getElementById('escenario2-desc').textContent = data.escenario_moderado.descripcion;
                document.getElementById('escenario2-cambios').innerHTML = data.escenario_moderado.cambios.map(c => `• ${{c}}`).join('<br>');
                document.getElementById('escenario2-mejora').textContent = '-' + data.mejora_moderada.toFixed(1) + '%';

                // ESCENARIO 3: Cambios óptimos
                document.getElementById('escenario3-titulo').textContent = data.escenario_optimo.titulo;
                document.getElementById('escenario3-riesgo').textContent = data.escenario_optimo.riesgo.toFixed(1) + '%';
                document.getElementById('escenario3-desc').textContent = data.escenario_optimo.descripcion;
                document.getElementById('escenario3-cambios').innerHTML = data.escenario_optimo.cambios.map(c => `• ${{c}}`).join('<br>');
                document.getElementById('escenario3-mejora').textContent = '-' + data.mejora_optima.toFixed(1) + '%';
            }}
        </script>

        <div class="step-card" style="background: #1e1e1e; color: white; border: 1px solid #2a2a2a;">
            <div class="step-content">
                <h2 style="color: white; text-align: center; font-size: 2rem; margin-bottom: 20px; font-weight: 600;">
                    <i class="fas fa-graduation-cap"></i> Resumen del Proceso
                </h2>

                <div style="background: #0d0d0d; padding: 28px; border-radius: 8px; border: 1px solid #2a2a2a;">
                    <p style="font-size: 1.05rem; line-height: 1.8; text-align: center; font-weight: 400; color: #d0d0d0;">
                        El modelo XGBoost funcionó creando <strong style="color: #ffffff;">300 árboles de decisión inteligentes</strong>
                        que aprendieron patrones de {total_records:,} pacientes. Usó técnicas avanzadas como
                        <strong style="color: #ffffff;">SMOTE</strong> para balancear datos y <strong style="color: #ffffff;">validación cruzada</strong> para
                        asegurar resultados confiables. El resultado final tiene una precisión de
                        <strong style="color: #ffffff;">{accuracy:.1f}%</strong>, lo que significa que de cada 100 predicciones,
                        aproximadamente {int(accuracy)} son correctas.
                    </p>
                </div>

                <div style="text-align: center; margin-top: 30px;">
                    <button onclick="window.close()" style="
                        background: #2a2a2a;
                        color: #ffffff;
                        border: 1px solid #3a3a3a;
                        padding: 12px 32px;
                        font-size: 1rem;
                        border-radius: 8px;
                        cursor: pointer;
                        font-weight: 500;
                        transition: all 0.2s ease;
                        font-family: 'Inter', sans-serif;
                    " onmouseover="this.style.background='#3a3a3a'" onmouseout="this.style.background='#2a2a2a'">
                        <i class="fas fa-arrow-left"></i> Cerrar análisis
                    </button>
                </div>
            </div>
        </div>
    </div>

    <style>
        @keyframes spin {{
            0% {{ transform: rotate(0deg); }}
            100% {{ transform: rotate(360deg); }}
        }}
    </style>
</body>
</html>
"""

    return html


def generate_column_items(columns):
    """Genera items HTML para la lista de columnas"""
    items_html = ""
    for col in columns:
        items_html += f'<li class="feature-item"><i class="fas fa-table"></i> <span>{col}</span></li>\n'
    return items_html


def get_performance_message(accuracy):
    """Genera un mensaje personalizado según la precisión"""
    if accuracy >= 90:
        return "¡Excelente! Este es un rendimiento excepcional para un modelo médico predictivo."
    elif accuracy >= 80:
        return "¡Muy bueno! Este modelo tiene un rendimiento sólido y confiable."
    elif accuracy >= 70:
        return "Buen rendimiento. El modelo es útil pero podría mejorarse con más datos o ajustes."
    else:
        return "El rendimiento es aceptable, pero se recomienda revisar la calidad de los datos o ajustar parámetros."
