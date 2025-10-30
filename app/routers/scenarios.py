from fastapi import APIRouter, HTTPException
import os
import pandas as pd
import joblib
import numpy as np

from app.utils import storage

router = APIRouter()

@router.get("/scenarios/{file_id}")
async def get_scenarios(file_id: str):
    """
    Calcula 3 escenarios para el análisis:
    1. Sin cambios (actual)
    2. Cambios moderados
    3. Cambios óptimos
    """
    try:
        # Obtener análisis del archivo
        analyses = storage.get_analyses_by_file(file_id)
        if not analyses:
            raise HTTPException(status_code=404, detail="No se encontró análisis")

        # Cargar modelo entrenado
        model_path = os.path.join("models", f"model_{file_id}.pkl")
        if not os.path.exists(model_path):
            raise HTTPException(status_code=500, detail="Modelo no encontrado")

        # Cargar todo el modelo (incluye: model, feature_names, label_encoder, processed_data)
        model_data = joblib.load(model_path)
        modelo = model_data['model']
        feature_names = model_data['feature_names']
        processed_data = model_data['processed_data']

        # Tomar un paciente ejemplo con riesgo moderado (no extremo)
        # En lugar del promedio, usamos la MEDIANA de pacientes con riesgo
        diagnosticos_riesgo = ['Hipertension', 'Diabetes', 'Prediabetes', 'Síndrome Metabólico']
        pacientes_riesgo = processed_data[processed_data['Diagnostico'].isin(diagnosticos_riesgo)]

        if len(pacientes_riesgo) == 0:
            # Si no hay pacientes de riesgo, buscar pacientes que NO sean "Normal"
            pacientes_riesgo = processed_data[processed_data['Diagnostico'] != 'Normal']

        if len(pacientes_riesgo) == 0:
            # Como último recurso, tomar todos los datos
            pacientes_riesgo = processed_data

        # Verificar que las features existan en los datos
        available_features = [f for f in feature_names if f in pacientes_riesgo.columns]

        if not available_features:
            raise HTTPException(status_code=500, detail="No se encontraron features válidas")

        # Usar MEDIANA en lugar de promedio para evitar valores extremos
        # Seleccionar solo las columnas numéricas
        numeric_features = pacientes_riesgo[available_features].select_dtypes(include=[np.number]).columns.tolist()

        if not numeric_features:
            raise HTTPException(status_code=500, detail="No se encontraron features numéricas")

        # Usar mediana para obtener un perfil más "típico" y no extremo
        perfil_ejemplo = pacientes_riesgo[numeric_features].median().to_dict()

        # Preparar features usando EXACTAMENTE los nombres guardados del modelo
        # Solo incluir las features que el modelo espera
        perfil_filtrado = {}
        for feature in feature_names:
            if feature in perfil_ejemplo:
                perfil_filtrado[feature] = perfil_ejemplo[feature]
            else:
                # Si falta alguna feature, usar valor por defecto
                perfil_filtrado[feature] = 0

        # Crear DataFrame con EXACTAMENTE las columnas que el modelo espera, en el orden correcto
        datos_actuales = pd.DataFrame([perfil_filtrado], columns=feature_names)

        # ESCENARIO 1: Sin cambios (actual)
        prob_actual = modelo.predict_proba(datos_actuales)[0]
        # Riesgo = probabilidad de NO ser "Normal" (clase 2)
        # Clases: 0=Diabetes, 1=Hipertension, 2=Normal, 3=Obesidad, 4=Prediabetes
        if len(prob_actual) >= 5:
            # Suma de todas las clases de riesgo (todas excepto Normal)
            riesgo_actual = (prob_actual[0] + prob_actual[1] + prob_actual[3] + prob_actual[4]) * 100
        else:
            riesgo_actual = (1 - prob_actual[2]) * 100 if len(prob_actual) > 2 else prob_actual[1] * 100

        # ESCENARIO 2: Cambios moderados
        datos_moderados = datos_actuales.copy()

        # Reducir peso (IMC -20% en lugar de -10% para mayor impacto)
        if 'IMC' in datos_moderados.columns:
            imc_nuevo = datos_moderados['IMC'].values[0] * 0.8
            datos_moderados['IMC'] = imc_nuevo
            # Actualizar features derivadas del IMC
            if 'IMC_x_Edad' in datos_moderados.columns and 'Edad' in datos_moderados.columns:
                datos_moderados['IMC_x_Edad'] = imc_nuevo * datos_moderados['Edad'].values[0]
            if 'Glucosa_x_IMC' in datos_moderados.columns and 'Glucosa' in datos_moderados.columns:
                datos_moderados['Glucosa_x_IMC'] = datos_moderados['Glucosa'].values[0] * imc_nuevo
            # Actualizar categoría IMC
            if 'Categoria_IMC' in datos_moderados.columns:
                if imc_nuevo < 18.5:
                    datos_moderados['Categoria_IMC'] = 0
                elif imc_nuevo < 25:
                    datos_moderados['Categoria_IMC'] = 1
                elif imc_nuevo < 30:
                    datos_moderados['Categoria_IMC'] = 2
                elif imc_nuevo < 35:
                    datos_moderados['Categoria_IMC'] = 3
                else:
                    datos_moderados['Categoria_IMC'] = 4

        # Reducir glucosa (-30% para mayor impacto)
        if 'Glucosa' in datos_moderados.columns:
            glucosa_nueva = datos_moderados['Glucosa'].values[0] * 0.7
            datos_moderados['Glucosa'] = glucosa_nueva
            # Actualizar features derivadas de glucosa
            if 'Glucosa_x_IMC' in datos_moderados.columns and 'IMC' in datos_moderados.columns:
                datos_moderados['Glucosa_x_IMC'] = glucosa_nueva * datos_moderados['IMC'].values[0]
            if 'Glucosa_x_Edad' in datos_moderados.columns and 'Edad' in datos_moderados.columns:
                datos_moderados['Glucosa_x_Edad'] = glucosa_nueva * datos_moderados['Edad'].values[0]
            # Actualizar categoría glucosa
            if 'Categoria_Glucosa' in datos_moderados.columns:
                if glucosa_nueva < 100:
                    datos_moderados['Categoria_Glucosa'] = 0
                elif glucosa_nueva < 126:
                    datos_moderados['Categoria_Glucosa'] = 1
                elif glucosa_nueva < 200:
                    datos_moderados['Categoria_Glucosa'] = 2
                else:
                    datos_moderados['Categoria_Glucosa'] = 3

        # Reducir presión arterial (-15%)
        if 'Presion_Sistolica' in datos_moderados.columns:
            presion_nueva = datos_moderados['Presion_Sistolica'].values[0] * 0.85
            datos_moderados['Presion_Sistolica'] = presion_nueva
            # Actualizar features derivadas
            if 'Presion_Media' in datos_moderados.columns and 'Presion_Diastolica' in datos_moderados.columns:
                diastolica = datos_moderados['Presion_Diastolica'].values[0] * 0.85
                datos_moderados['Presion_Diastolica'] = diastolica
                datos_moderados['Presion_Media'] = (presion_nueva + 2 * diastolica) / 3
            if 'Presion_Pulso' in datos_moderados.columns and 'Presion_Diastolica' in datos_moderados.columns:
                datos_moderados['Presion_Pulso'] = presion_nueva - datos_moderados['Presion_Diastolica'].values[0]
            if 'Categoria_Presion' in datos_moderados.columns:
                if presion_nueva < 120:
                    datos_moderados['Categoria_Presion'] = 0
                elif presion_nueva < 130:
                    datos_moderados['Categoria_Presion'] = 1
                elif presion_nueva < 140:
                    datos_moderados['Categoria_Presion'] = 2
                elif presion_nueva < 180:
                    datos_moderados['Categoria_Presion'] = 3
                else:
                    datos_moderados['Categoria_Presion'] = 4

        # Reducir colesterol (-20%)
        if 'Colesterol' in datos_moderados.columns:
            colesterol_nuevo = datos_moderados['Colesterol'].values[0] * 0.8
            datos_moderados['Colesterol'] = colesterol_nuevo
            # Actualizar features derivadas
            if 'Ratio_Colesterol_Edad' in datos_moderados.columns and 'Edad' in datos_moderados.columns:
                datos_moderados['Ratio_Colesterol_Edad'] = colesterol_nuevo / datos_moderados['Edad'].values[0]

        # Asegurar que solo tiene las columnas del modelo
        datos_moderados = datos_moderados[feature_names]

        prob_moderado = modelo.predict_proba(datos_moderados)[0]
        # Calcular riesgo igual que en escenario actual
        if len(prob_moderado) >= 5:
            riesgo_moderado = (prob_moderado[0] + prob_moderado[1] + prob_moderado[3] + prob_moderado[4]) * 100
        else:
            riesgo_moderado = (1 - prob_moderado[2]) * 100 if len(prob_moderado) > 2 else prob_moderado[1] * 100

        # ESCENARIO 3: Cambios óptimos
        datos_optimos = datos_actuales.copy()

        # Reducir peso significativamente (IMC -35%)
        if 'IMC' in datos_optimos.columns:
            imc_optimo = datos_optimos['IMC'].values[0] * 0.65
            datos_optimos['IMC'] = imc_optimo
            # Actualizar features derivadas del IMC
            if 'IMC_x_Edad' in datos_optimos.columns and 'Edad' in datos_optimos.columns:
                datos_optimos['IMC_x_Edad'] = imc_optimo * datos_optimos['Edad'].values[0]
            if 'Glucosa_x_IMC' in datos_optimos.columns and 'Glucosa' in datos_optimos.columns:
                datos_optimos['Glucosa_x_IMC'] = datos_optimos['Glucosa'].values[0] * imc_optimo
            # Actualizar categoría IMC
            if 'Categoria_IMC' in datos_optimos.columns:
                if imc_optimo < 18.5:
                    datos_optimos['Categoria_IMC'] = 0
                elif imc_optimo < 25:
                    datos_optimos['Categoria_IMC'] = 1
                elif imc_optimo < 30:
                    datos_optimos['Categoria_IMC'] = 2
                elif imc_optimo < 35:
                    datos_optimos['Categoria_IMC'] = 3
                else:
                    datos_optimos['Categoria_IMC'] = 4

        # Reducir glucosa significativamente (-50%)
        if 'Glucosa' in datos_optimos.columns:
            glucosa_optima = datos_optimos['Glucosa'].values[0] * 0.5
            datos_optimos['Glucosa'] = glucosa_optima
            # Actualizar features derivadas de glucosa
            if 'Glucosa_x_IMC' in datos_optimos.columns and 'IMC' in datos_optimos.columns:
                datos_optimos['Glucosa_x_IMC'] = glucosa_optima * datos_optimos['IMC'].values[0]
            if 'Glucosa_x_Edad' in datos_optimos.columns and 'Edad' in datos_optimos.columns:
                datos_optimos['Glucosa_x_Edad'] = glucosa_optima * datos_optimos['Edad'].values[0]
            # Actualizar categoría glucosa
            if 'Categoria_Glucosa' in datos_optimos.columns:
                if glucosa_optima < 100:
                    datos_optimos['Categoria_Glucosa'] = 0
                elif glucosa_optima < 126:
                    datos_optimos['Categoria_Glucosa'] = 1
                elif glucosa_optima < 200:
                    datos_optimos['Categoria_Glucosa'] = 2
                else:
                    datos_optimos['Categoria_Glucosa'] = 3

        # Reducir presión sistólica (-30%)
        if 'Presion_Sistolica' in datos_optimos.columns:
            presion_nueva = datos_optimos['Presion_Sistolica'].values[0] * 0.7
            datos_optimos['Presion_Sistolica'] = presion_nueva
            # Actualizar features derivadas de presión
            if 'Presion_Media' in datos_optimos.columns and 'Presion_Diastolica' in datos_optimos.columns:
                diastolica = datos_optimos['Presion_Diastolica'].values[0]
                datos_optimos['Presion_Media'] = (presion_nueva + 2 * diastolica) / 3
            if 'Presion_Pulso' in datos_optimos.columns and 'Presion_Diastolica' in datos_optimos.columns:
                datos_optimos['Presion_Pulso'] = presion_nueva - datos_optimos['Presion_Diastolica'].values[0]
            if 'Presion_x_Edad' in datos_optimos.columns and 'Edad' in datos_optimos.columns:
                datos_optimos['Presion_x_Edad'] = presion_nueva * datos_optimos['Edad'].values[0]
            if 'Ratio_Sistolica_Diastolica' in datos_optimos.columns and 'Presion_Diastolica' in datos_optimos.columns:
                datos_optimos['Ratio_Sistolica_Diastolica'] = presion_nueva / datos_optimos['Presion_Diastolica'].values[0]
            # Actualizar categoría presión
            if 'Categoria_Presion' in datos_optimos.columns:
                if presion_nueva < 120:
                    datos_optimos['Categoria_Presion'] = 0
                elif presion_nueva < 130:
                    datos_optimos['Categoria_Presion'] = 1
                elif presion_nueva < 140:
                    datos_optimos['Categoria_Presion'] = 2
                elif presion_nueva < 180:
                    datos_optimos['Categoria_Presion'] = 3
                else:
                    datos_optimos['Categoria_Presion'] = 4

        # Reducir colesterol significativamente (-40%)
        if 'Colesterol' in datos_optimos.columns:
            colesterol_optimo = datos_optimos['Colesterol'].values[0] * 0.6
            datos_optimos['Colesterol'] = colesterol_optimo
            if 'Ratio_Colesterol_Edad' in datos_optimos.columns and 'Edad' in datos_optimos.columns:
                datos_optimos['Ratio_Colesterol_Edad'] = colesterol_optimo / datos_optimos['Edad'].values[0]

        # Dejar de fumar
        if 'Fumador_encoded' in datos_optimos.columns:
            datos_optimos['Fumador_encoded'] = 0

        # Reducir score cardiovascular dramáticamente
        if 'Score_Cardiovascular' in datos_optimos.columns:
            datos_optimos['Score_Cardiovascular'] = datos_optimos['Score_Cardiovascular'].values[0] * 0.4

        # Asegurar que solo tiene las columnas del modelo
        datos_optimos = datos_optimos[feature_names]

        prob_optimo = modelo.predict_proba(datos_optimos)[0]
        # Calcular riesgo igual que en escenarios anteriores
        if len(prob_optimo) >= 5:
            riesgo_optimo = (prob_optimo[0] + prob_optimo[1] + prob_optimo[3] + prob_optimo[4]) * 100
        else:
            riesgo_optimo = (1 - prob_optimo[2]) * 100 if len(prob_optimo) > 2 else prob_optimo[1] * 100

        # Obtener valores actuales para mostrar
        valores_actuales = {
            "imc": float(datos_actuales['IMC'].values[0]) if 'IMC' in datos_actuales.columns else 0,
            "glucosa": float(datos_actuales['Glucosa'].values[0]) if 'Glucosa' in datos_actuales.columns else 0,
            "presion": float(datos_actuales['Presion_Sistolica'].values[0]) if 'Presion_Sistolica' in datos_actuales.columns else 0,
            "fumador": int(datos_actuales['Fumador_encoded'].values[0]) if 'Fumador_encoded' in datos_actuales.columns else 0,
        }

        return {
            "escenario_actual": {
                "riesgo": float(riesgo_actual),
                "titulo": "Sin Cambios",
                "descripcion": "Manteniendo el estilo de vida actual",
                "cambios": []
            },
            "escenario_moderado": {
                "riesgo": float(riesgo_moderado),
                "titulo": "Cambios Moderados",
                "descripcion": "Mejoras alcanzables en 6-12 meses",
                "cambios": [
                    "Reducción de peso (IMC -20%)",
                    "Control de glucosa (-30%)",
                    "Reducción de presión arterial (-15%)",
                    "Reducción de colesterol (-20%)"
                ]
            },
            "escenario_optimo": {
                "riesgo": float(riesgo_optimo),
                "titulo": "Cambios Óptimos",
                "descripcion": "Transformación completa del estilo de vida",
                "cambios": [
                    "Pérdida significativa de peso (IMC -35%)",
                    "Control estricto de glucosa (-50%)",
                    "Reducción fuerte de presión arterial (-30%)",
                    "Reducción significativa de colesterol (-40%)",
                    "Dejar de fumar completamente",
                    "Ejercicio regular intenso",
                    "Dieta cardiovascular óptima"
                ]
            },
            "valores_actuales": valores_actuales,
            "mejora_moderada": float(riesgo_actual - riesgo_moderado),
            "mejora_optima": float(riesgo_actual - riesgo_optimo)
        }

    except Exception as e:
        print(f"Error en simulación de escenarios: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error: {str(e)}")
