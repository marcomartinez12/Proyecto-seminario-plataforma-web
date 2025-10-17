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
        prompt = """Eres un experto en salud que explica informacion medica de manera clara y sencilla para que cualquier persona pueda entenderla. Analiza los siguientes datos de salud y proporciona una explicacion facil de comprender:

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

Por favor, proporciona una explicacion en lenguaje simple y accesible que incluya:

1. **Resumen general**: Que nos dicen estos datos en palabras sencillas
2. **Hallazgos importantes**: Los puntos mas relevantes explicados de forma clara
3. **Que significan los numeros**: Interpreta las estadisticas de manera comprensible (por ejemplo: si el IMC promedio es alto, bajo o normal)
4. **Recomendaciones practicas**: Consejos de prevencion que cualquiera pueda entender y aplicar
5. **Conclusion**: Un mensaje final breve sobre la situacion general de salud

IMPORTANTE: Usa lenguaje cotidiano, evita terminos medicos complicados o explicalo cuando los uses. Escribe como si le estuvieras explicando a un familiar o amigo. Se claro, directo y positivo cuando sea posible.""".format(
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
        OPENROUTER_API_KEY = os.environ.get("OPENROUTER_API_KEY")

        if not OPENROUTER_API_KEY:
            raise HTTPException(
                status_code=500,
                detail="OPENROUTER_API_KEY no configurada. Por favor configura la variable de entorno."
            )

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
                    "model": "openai/gpt-oss-20b:free",
                    "messages": [
                        {
                            "role": "system",
                            "content": "Eres un comunicador de salud experto en explicar informacion medica compleja de forma simple y accesible para todo publico. Tu objetivo es que cualquier persona, sin importar su nivel educativo, pueda entender los conceptos de salud."
                        },
                        {
                            "role": "user",
                            "content": prompt
                        }
                    ],
                    "temperature": 0.7,
                    "max_tokens": 1500
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
            "model_used": "openai/gpt-oss-20b:free"
        }

    except Exception as e:
        print(f"Error en análisis IA: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error al generar análisis con IA: {str(e)}")
