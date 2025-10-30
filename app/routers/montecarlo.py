from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
import numpy as np
import joblib
import os
from typing import List, Dict
import pandas as pd

router = APIRouter()

class MonteCarloRequest(BaseModel):
    file_id: str
    num_simulaciones: int = 10000
    años: int = 5

class MonteCarloResponse(BaseModel):
    probabilidad_diabetes: float
    probabilidad_hipertension: float
    probabilidad_obesidad: float
    probabilidad_normal: float
    percentil_10: float
    percentil_50: float
    percentil_90: float
    mejor_caso: float
    peor_caso: float
    caso_promedio: float
    distribuciones: Dict[str, List[float]]
    timeline_datos: Dict[str, List[float]]

@router.post("/montecarlo/simulate", response_model=MonteCarloResponse)
async def simulate_montecarlo(request: MonteCarloRequest):
    """
    Ejecuta una simulación Monte Carlo para predecir futuros posibles de un paciente
    """

    # Cargar el modelo entrenado
    model_path = f"models/model_{request.file_id}.pkl"

    if not os.path.exists(model_path):
        raise HTTPException(status_code=404, detail="Modelo no encontrado. Debe ejecutar un análisis primero.")

    try:
        model_data = joblib.load(model_path)
        modelo = model_data['model']
        feature_names = model_data['feature_names']
        processed_data = model_data['processed_data']
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error al cargar modelo: {str(e)}")

    # Obtener datos de pacientes con riesgo (excluir "Normal")
    if 'Diagnostico' in processed_data.columns:
        datos_riesgo = processed_data[processed_data['Diagnostico'] != 'Normal'].copy()
    else:
        datos_riesgo = processed_data.copy()

    if len(datos_riesgo) == 0:
        raise HTTPException(status_code=400, detail="No hay pacientes con diagnósticos de riesgo en el dataset")

    # Calcular perfil promedio (mediana) de pacientes con riesgo
    datos_numericos = datos_riesgo[feature_names].select_dtypes(include=[np.number])
    perfil_base = datos_numericos.median()

    # Ejecutar simulaciones Monte Carlo
    resultados_simulaciones = []
    probabilidades_diabetes = []
    probabilidades_hipertension = []
    probabilidades_obesidad = []
    probabilidades_normal = []
    riesgos_totales = []

    np.random.seed(42)  # Para reproducibilidad

    for i in range(request.num_simulaciones):
        # Crear escenario aleatorio con variaciones
        escenario = perfil_base.copy()

        # Variables aleatorias para simular diferentes futuros
        imc_cambio = np.random.uniform(-10, 5)  # Cambio en IMC
        glucosa_cambio = np.random.uniform(-30, 20)  # Cambio en glucosa
        presion_cambio = np.random.uniform(-20, 15)  # Cambio en presión
        colesterol_cambio = np.random.uniform(-30, 20)  # Cambio en colesterol
        ejercicio_factor = np.random.uniform(0, 1)  # Factor de ejercicio (0 = nada, 1 = mucho)
        dieta_factor = np.random.uniform(0, 1)  # Factor de dieta (0 = mala, 1 = excelente)
        adherencia = np.random.uniform(0.3, 1.0)  # Adherencia al tratamiento
        genetica = np.random.uniform(0.85, 1.15)  # Factor genético
        stress = np.random.uniform(0.9, 1.2)  # Factor de stress

        # Aplicar cambios con interacciones
        if 'IMC' in escenario.index:
            cambio_imc = imc_cambio * adherencia * ejercicio_factor
            escenario['IMC'] = max(18, min(45, escenario['IMC'] + cambio_imc))

        if 'Glucosa' in escenario.index:
            cambio_glucosa = glucosa_cambio * adherencia * dieta_factor * genetica
            escenario['Glucosa'] = max(70, min(300, escenario['Glucosa'] + cambio_glucosa))

        if 'Presion_Sistolica' in escenario.index:
            cambio_presion = presion_cambio * adherencia * stress
            escenario['Presion_Sistolica'] = max(90, min(200, escenario['Presion_Sistolica'] + cambio_presion))

        if 'Colesterol' in escenario.index:
            cambio_colesterol = colesterol_cambio * adherencia * dieta_factor
            escenario['Colesterol'] = max(100, min(350, escenario['Colesterol'] + cambio_colesterol))

        # Actualizar features derivadas
        if 'IMC_x_Edad' in escenario.index and 'IMC' in escenario.index and 'Edad' in escenario.index:
            escenario['IMC_x_Edad'] = escenario['IMC'] * escenario['Edad']

        if 'Glucosa_x_IMC' in escenario.index and 'Glucosa' in escenario.index and 'IMC' in escenario.index:
            escenario['Glucosa_x_IMC'] = escenario['Glucosa'] * escenario['IMC']

        if 'Score_Cardiovascular' in escenario.index:
            escenario['Score_Cardiovascular'] = escenario['Score_Cardiovascular'] * stress

        # Actualizar categorías basadas en nuevos valores
        if 'Categoria_IMC' in escenario.index and 'IMC' in escenario.index:
            if escenario['IMC'] < 18.5:
                escenario['Categoria_IMC'] = 0
            elif escenario['IMC'] < 25:
                escenario['Categoria_IMC'] = 1
            elif escenario['IMC'] < 30:
                escenario['Categoria_IMC'] = 2
            else:
                escenario['Categoria_IMC'] = 3

        if 'Categoria_Glucosa' in escenario.index and 'Glucosa' in escenario.index:
            if escenario['Glucosa'] < 100:
                escenario['Categoria_Glucosa'] = 0
            elif escenario['Glucosa'] < 126:
                escenario['Categoria_Glucosa'] = 1
            else:
                escenario['Categoria_Glucosa'] = 2

        # Preparar datos para predicción
        datos_prediccion = pd.DataFrame([escenario[feature_names]])

        # Hacer predicción
        probabilidades = modelo.predict_proba(datos_prediccion)[0]

        # Guardar probabilidades por clase
        # Clases: 0=Diabetes, 1=Hipertension, 2=Normal, 3=Obesidad, 4=Prediabetes
        if len(probabilidades) >= 5:
            prob_diabetes = probabilidades[0]
            prob_hipertension = probabilidades[1]
            prob_normal = probabilidades[2]
            prob_obesidad = probabilidades[3]
            # Riesgo total = suma de todas las clases de riesgo (excluir Normal)
            riesgo_total = prob_diabetes + prob_hipertension + probabilidades[3] + probabilidades[4]
        else:
            prob_diabetes = probabilidades[0] if len(probabilidades) > 0 else 0
            prob_hipertension = probabilidades[1] if len(probabilidades) > 1 else 0
            prob_normal = probabilidades[2] if len(probabilidades) > 2 else 0
            prob_obesidad = probabilidades[3] if len(probabilidades) > 3 else 0
            riesgo_total = 1 - prob_normal if len(probabilidades) > 2 else prob_diabetes + prob_hipertension

        probabilidades_diabetes.append(prob_diabetes)
        probabilidades_hipertension.append(prob_hipertension)
        probabilidades_obesidad.append(prob_obesidad)
        probabilidades_normal.append(prob_normal)
        riesgos_totales.append(riesgo_total)

        resultados_simulaciones.append({
            'imc': float(escenario['IMC']) if 'IMC' in escenario.index else 0,
            'glucosa': float(escenario['Glucosa']) if 'Glucosa' in escenario.index else 0,
            'riesgo': float(riesgo_total),
            'prob_diabetes': float(prob_diabetes),
            'prob_hipertension': float(prob_hipertension),
            'prob_obesidad': float(prob_obesidad),
            'prob_normal': float(prob_normal)
        })

    # Calcular estadísticas
    riesgos_array = np.array(riesgos_totales)

    percentil_10 = float(np.percentile(riesgos_array, 10))
    percentil_50 = float(np.percentile(riesgos_array, 50))
    percentil_90 = float(np.percentile(riesgos_array, 90))
    mejor_caso = float(np.min(riesgos_array))
    peor_caso = float(np.max(riesgos_array))
    caso_promedio = float(np.mean(riesgos_array))

    # Calcular probabilidades promedio por diagnóstico
    prob_diabetes_promedio = float(np.mean(probabilidades_diabetes))
    prob_hipertension_promedio = float(np.mean(probabilidades_hipertension))
    prob_obesidad_promedio = float(np.mean(probabilidades_obesidad))
    prob_normal_promedio = float(np.mean(probabilidades_normal))

    # Generar distribución para el histograma (bins)
    hist, bin_edges = np.histogram(riesgos_array, bins=50, range=(0, 1))
    hist_normalizado = (hist / request.num_simulaciones).tolist()
    bin_centers = ((bin_edges[:-1] + bin_edges[1:]) / 2).tolist()

    # Generar timeline de proyección (años)
    timeline_años = list(range(request.años + 1))
    timeline_mejor = [mejor_caso * (i / request.años) for i in timeline_años]
    timeline_promedio = [caso_promedio * (i / request.años) for i in timeline_años]
    timeline_peor = [peor_caso * (i / request.años) for i in timeline_años]

    return MonteCarloResponse(
        probabilidad_diabetes=prob_diabetes_promedio,
        probabilidad_hipertension=prob_hipertension_promedio,
        probabilidad_obesidad=prob_obesidad_promedio,
        probabilidad_normal=prob_normal_promedio,
        percentil_10=percentil_10,
        percentil_50=percentil_50,
        percentil_90=percentil_90,
        mejor_caso=mejor_caso,
        peor_caso=peor_caso,
        caso_promedio=caso_promedio,
        distribuciones={
            'histogram_heights': hist_normalizado,
            'histogram_bins': bin_centers
        },
        timeline_datos={
            'años': timeline_años,
            'mejor_caso': timeline_mejor,
            'caso_promedio': timeline_promedio,
            'peor_caso': timeline_peor
        }
    )
