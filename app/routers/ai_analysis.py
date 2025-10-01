from fastapi import APIRouter, HTTPException
import os
import pandas as pd
import httpx
import sys

from app.utils import storage
from app.routers.analysis import preprocess_data

router = APIRouter()

# Configurar encoding UTF-8 para Windows
if sys.platform == "win32":
    import locale
    locale.setlocale(locale.LC_ALL, 'en_US.UTF-8')

@router.post("/ai-analysis/{file_id}")
async def ai_analysis(file_id: str):
    """Generar explicación con IA usando OpenRouter sobre el análisis del archivo"""

    try:
        # Obtener análisis del archivo
        analyses = storage.get_analyses_by_file(file_id)
        if not analyses:
            raise HTTPException(status_code=404, detail="No se encontró análisis para este archivo")

        # Obtener el análisis más reciente
        latest_analysis = max(analyses, key=lambda x: x['analysis_date'])

        # Obtener datos del archivo
        file_data = storage.get_file(file_id)
        if not file_data or not os.path.exists(file_data["file_path"]):
            raise HTTPException(status_code=404, detail="Archivo no encontrado")

        # Cargar datos del Excel
        df = pd.read_excel(file_data["file_path"])

        # Preprocesar datos para obtener estadísticas
        processed_data, _ = preprocess_data(df)

        # Preparar información para la IA
        total_records = len(df)
        hypertension_cases = latest_analysis.get('hypertension_cases', 0)
        diabetes_cases = latest_analysis.get('diabetes_cases', 0)
        accuracy = latest_analysis.get('accuracy_score', 0)

        # Estadísticas adicionales
        avg_age = processed_data['Edad'].mean()
        avg_bmi = processed_data['IMC'].mean()
        avg_glucose = processed_data['Glucosa'].mean()
        avg_systolic = processed_data['Presion_Sistolica'].mean()
        smokers = len(processed_data[processed_data['Fumador_encoded'] == 1])

        # Distribución por diagnóstico (sin caracteres especiales para evitar problemas de encoding)
        diagnosis_dist = df['Diagnostico'].value_counts().to_dict()
        diagnosis_list = "\n".join([f"- {str(diag)}: {int(count)} casos" for diag, count in diagnosis_dist.items()])

        # Crear prompt para la IA (evitando caracteres especiales)
        prompt = """Eres un medico especialista en enfermedades cronicas y analisis epidemiologico. Analiza los siguientes datos de salud poblacional y proporciona una explicacion profesional, clara y detallada:

DATOS DEL ANALISIS:
- Total de pacientes: {}
- Casos de Hipertension: {} ({}%)
- Casos de Diabetes: {} ({}%)
- Precision del modelo ML: {}%

ESTADISTICAS CLINICAS:
- Edad promedio: {} años
- IMC promedio: {} kg/m2
- Glucosa promedio: {} mg/dL
- Presion sistolica promedio: {} mmHg
- Fumadores: {} ({}%)

DISTRIBUCION DE DIAGNOSTICOS:
{}

Por favor, proporciona:
1. Un analisis epidemiologico de los hallazgos principales
2. Interpretacion de los factores de riesgo detectados
3. Recomendaciones preventivas especificas basadas en los datos
4. Evaluacion de la gravedad de la situacion poblacional
5. Estrategias de intervencion prioritarias

Responde de manera profesional, estructurada y en español, como si fueras un medico epidemiologo presentando resultados a un comite de salud publica.""".format(
            total_records,
            hypertension_cases,
            round((hypertension_cases/total_records*100), 1),
            diabetes_cases,
            round((diabetes_cases/total_records*100), 1),
            round(accuracy * 100, 1),
            round(avg_age, 1),
            round(avg_bmi, 1),
            round(avg_glucose, 1),
            round(avg_systolic, 1),
            smokers,
            round((smokers/total_records*100), 1),
            diagnosis_list
        )

        # Llamar a OpenRouter API
        OPENROUTER_API_KEY = os.environ.get("OPENROUTER_API_KEY", "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")

        async with httpx.AsyncClient(timeout=60.0) as client:
            response = await client.post(
                "https://openrouter.ai/api/v1/chat/completions",
                headers={
                    "Authorization": f"Bearer {OPENROUTER_API_KEY}",
                    "Content-Type": "application/json; charset=utf-8",
                    "HTTP-Referer": "http://localhost:8000",
                    "X-Title": "Plataforma Analisis Medico"
                },
                json={
                    "model": "mistralai/mistral-small-24b-instruct-2501:free",
                    "messages": [
                        {
                            "role": "system",
                            "content": "Eres un medico especialista en epidemiologia y salud publica con amplia experiencia en analisis de datos de enfermedades cronicas."
                        },
                        {
                            "role": "user",
                            "content": prompt
                        }
                    ],
                    "temperature": 0.7,
                    "max_tokens": 2000
                }
            )

        if response.status_code != 200:
            raise HTTPException(
                status_code=response.status_code,
                detail=f"Error en OpenRouter API: {response.text}"
            )

        result = response.json()
        ai_explanation = result['choices'][0]['message']['content']

        return {
            "file_id": file_id,
            "analysis_summary": {
                "total_records": total_records,
                "hypertension_cases": hypertension_cases,
                "diabetes_cases": diabetes_cases,
                "accuracy": accuracy,
                "avg_age": round(avg_age, 1),
                "avg_bmi": round(avg_bmi, 1),
                "avg_glucose": round(avg_glucose, 1),
                "avg_systolic": round(avg_systolic, 1),
                "smokers": smokers
            },
            "ai_explanation": ai_explanation,
            "model_used": "mistralai/mistral-small-24b-instruct-2501:free"
        }

    except Exception as e:
        print(f"Error en análisis IA: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error al generar análisis con IA: {str(e)}")
