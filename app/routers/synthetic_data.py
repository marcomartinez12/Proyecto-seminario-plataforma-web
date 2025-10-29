"""
Router para generación de datasets sintéticos
Permite a los usuarios generar datos médicos aleatorios desde la interfaz web
"""

from fastapi import APIRouter, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field
import pandas as pd
import numpy as np
from datetime import datetime
import os
import tempfile
from typing import Literal

router = APIRouter()


class DatasetGenerationRequest(BaseModel):
    """Modelo para solicitud de generación de dataset"""
    num_records: int = Field(default=1000, ge=100, le=50000, description="Número de registros (100-50000)")
    level: Literal["MINIMO", "ESTANDAR", "COMPLETO"] = Field(default="COMPLETO", description="Nivel de columnas")
    format: Literal["excel", "csv", "json"] = Field(default="excel", description="Formato de salida")
    seed: int = Field(default=42, description="Semilla para reproducibilidad")


def generar_dataset_medico(n_registros: int = 10000, seed: int = 42, nivel: str = "COMPLETO"):
    """
    Genera dataset médico sintético con datos clínicamente coherentes.

    Parámetros:
    - n_registros: Número de pacientes a generar
    - seed: Semilla para reproducibilidad
    - nivel: "MINIMO", "ESTANDAR" o "COMPLETO"
    """

    np.random.seed(seed)

    # ===== DISTRIBUCIÓN DE DIAGNÓSTICOS =====
    diagnosticos_dist = {
        'Normal': 0.45,
        'Prediabetes': 0.20,
        'Hipertension': 0.20,
        'Diabetes': 0.10,
        'Obesidad': 0.05
    }

    diagnosticos = np.random.choice(
        list(diagnosticos_dist.keys()),
        size=n_registros,
        p=list(diagnosticos_dist.values())
    )

    # ===== GENERAR DATOS DEMOGRÁFICOS =====
    edad_base = np.random.beta(2, 2, n_registros) * 70 + 18
    edad = np.round(edad_base).astype(int)
    sexo = np.random.choice(['M', 'F'], size=n_registros)

    # ===== GENERAR DATOS ANTROPOMÉTRICOS =====
    peso_base_m = np.random.normal(75, 15, n_registros)
    peso_base_f = np.random.normal(65, 12, n_registros)
    peso = np.where(sexo == 'M', peso_base_m, peso_base_f)

    altura_base_m = np.random.normal(175, 8, n_registros)
    altura_base_f = np.random.normal(162, 7, n_registros)
    altura = np.where(sexo == 'M', altura_base_m, altura_base_f)

    # Ajustar peso según diagnóstico
    for i, diag in enumerate(diagnosticos):
        if diag == 'Obesidad':
            peso[i] = peso[i] * 1.4 + 20
        elif diag == 'Diabetes':
            peso[i] = peso[i] * 1.2 + 10

    peso = np.clip(peso, 40, 200)
    altura = np.clip(altura, 140, 210)
    imc = peso / ((altura / 100) ** 2)

    # ===== GENERAR DATOS METABÓLICOS =====
    glucosa = np.random.normal(95, 10, n_registros)

    for i, diag in enumerate(diagnosticos):
        if diag == 'Diabetes':
            glucosa[i] = np.random.uniform(130, 250)
        elif diag == 'Prediabetes':
            glucosa[i] = np.random.uniform(100, 125)
        elif diag == 'Normal':
            glucosa[i] = np.random.uniform(70, 99)
        elif diag == 'Obesidad':
            glucosa[i] = np.random.uniform(95, 130)
        elif diag == 'Hipertension':
            glucosa[i] = np.random.uniform(85, 110)

    glucosa = glucosa + (edad - 40) * 0.3 + (imc - 25) * 0.8
    glucosa = np.clip(glucosa, 60, 400)

    # ===== GENERAR PRESIÓN ARTERIAL =====
    presion_sistolica = np.random.normal(120, 10, n_registros)

    for i, diag in enumerate(diagnosticos):
        if diag == 'Hipertension':
            presion_sistolica[i] = np.random.uniform(145, 180)
        elif diag == 'Normal':
            presion_sistolica[i] = np.random.uniform(90, 119)
        elif diag == 'Diabetes':
            presion_sistolica[i] = np.random.uniform(120, 150)
        elif diag == 'Obesidad':
            presion_sistolica[i] = np.random.uniform(130, 160)

    presion_sistolica = presion_sistolica + (edad - 40) * 0.5 + (imc - 25) * 1.2
    presion_sistolica = np.clip(presion_sistolica, 80, 220)

    presion_diastolica = presion_sistolica * 0.6 + np.random.normal(0, 5, n_registros)
    presion_diastolica = np.clip(presion_diastolica, 50, 130)

    # ===== COLESTEROL =====
    colesterol_total = np.random.normal(190, 35, n_registros)
    colesterol_total = colesterol_total + (edad - 40) * 0.4 + (imc - 25) * 1.5
    colesterol_total = np.clip(colesterol_total, 120, 350)

    # ===== FUMADOR =====
    prob_fumar = 0.15
    fumador = np.random.random(n_registros) < prob_fumar
    fumador = fumador.astype(int)

    # ===== CREAR DATAFRAME BASE =====
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

    # Presión arterial en formato "120/80"
    df['Presion_Arterial'] = df['Presion_Sistolica'].astype(str) + '/' + df['Presion_Diastolica'].astype(str)

    # ===== AGREGAR DATOS AVANZADOS SI NIVEL COMPLETO =====
    if nivel == "COMPLETO":
        # Hemoglobina A1C
        hemoglobina_a1c = 4.5 + (glucosa - 70) * 0.02 + np.random.normal(0, 0.3, n_registros)
        hemoglobina_a1c = np.clip(hemoglobina_a1c, 4.0, 14.0)

        # Perfil lipídico
        colesterol_ldl = colesterol_total * 0.65 + np.random.normal(0, 15, n_registros)
        colesterol_ldl = np.clip(colesterol_ldl, 50, 250)

        colesterol_hdl = 55 - (imc - 25) * 0.8 + np.random.normal(0, 8, n_registros)
        colesterol_hdl = np.clip(colesterol_hdl, 20, 100)

        trigliceridos = 100 + (glucosa - 90) * 0.8 + (imc - 25) * 3 + np.random.normal(0, 30, n_registros)
        trigliceridos = np.clip(trigliceridos, 50, 400)

        # Circunferencias
        circ_cintura_base_m = 85 + (imc - 25) * 2.5
        circ_cintura_base_f = 75 + (imc - 25) * 2.3
        circunferencia_cintura = np.where(
            sexo == 'M',
            circ_cintura_base_m + np.random.normal(0, 8, n_registros),
            circ_cintura_base_f + np.random.normal(0, 7, n_registros)
        )
        circunferencia_cintura = np.clip(circunferencia_cintura, 60, 150)

        circ_cadera_base = 95 + (imc - 25) * 2
        circunferencia_cadera = circ_cadera_base + np.random.normal(0, 8, n_registros)
        circunferencia_cadera = np.clip(circunferencia_cadera, 70, 160)

        # Otros datos clínicos
        antecedentes_familiares = (np.random.random(n_registros) < 0.30).astype(int)

        creatinina = np.random.normal(1.0, 0.2, n_registros)
        for i, diag in enumerate(diagnosticos):
            if diag in ['Diabetes', 'Hipertension'] and edad[i] > 55:
                creatinina[i] = creatinina[i] * 1.3
        creatinina = np.clip(creatinina, 0.5, 3.0)

        frecuencia_cardiaca = np.random.normal(72, 10, n_registros)
        for i, diag in enumerate(diagnosticos):
            if diag in ['Obesidad', 'Hipertension']:
                frecuencia_cardiaca[i] = frecuencia_cardiaca[i] * 1.1
        frecuencia_cardiaca = np.clip(frecuencia_cardiaca, 50, 120).astype(int)

        actividad_fisica = np.random.gamma(2, 40, n_registros)
        for i, diag in enumerate(diagnosticos):
            if diag == 'Normal':
                actividad_fisica[i] = actividad_fisica[i] * 1.8
            elif diag == 'Obesidad':
                actividad_fisica[i] = actividad_fisica[i] * 0.4
        actividad_fisica = np.clip(actividad_fisica, 0, 500).astype(int)

        consumo_alcohol = np.random.poisson(2, n_registros)
        consumo_alcohol = np.clip(consumo_alcohol, 0, 20)

        # Agregar columnas al dataframe
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

    # ===== FILTRAR COLUMNAS SEGÚN NIVEL =====
    if nivel == "MINIMO":
        columnas_minimas = ['ID', 'Edad', 'Sexo', 'Glucosa', 'Presion_Sistolica', 'Presion_Arterial', 'Diagnostico']
        df = df[columnas_minimas]

    return df


@router.post("/generate")
async def generate_synthetic_data(request: DatasetGenerationRequest):
    """
    Genera un dataset sintético de datos médicos
    """
    try:
        # Generar dataset
        df = generar_dataset_medico(
            n_registros=request.num_records,
            seed=request.seed,
            nivel=request.level
        )

        # Crear directorio temporal si no existe
        downloads_dir = "downloads"
        os.makedirs(downloads_dir, exist_ok=True)

        # Generar nombre de archivo único
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename_base = f"dataset_medico_{request.level.lower()}_{request.num_records}_{timestamp}"

        # Guardar según formato solicitado
        if request.format == "excel":
            filepath = os.path.join(downloads_dir, f"{filename_base}.xlsx")
            df.to_excel(filepath, index=False, engine='openpyxl')
            media_type = "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"

        elif request.format == "csv":
            filepath = os.path.join(downloads_dir, f"{filename_base}.csv")
            df.to_csv(filepath, index=False, encoding='utf-8-sig')
            media_type = "text/csv"

        elif request.format == "json":
            filepath = os.path.join(downloads_dir, f"{filename_base}.json")
            df.to_json(filepath, orient='records', indent=2, force_ascii=False)
            media_type = "application/json"

        # Retornar archivo para descarga
        return FileResponse(
            path=filepath,
            media_type=media_type,
            filename=os.path.basename(filepath),
            headers={
                "Content-Disposition": f"attachment; filename={os.path.basename(filepath)}"
            }
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error al generar dataset: {str(e)}")


@router.get("/info")
async def get_generation_info():
    """
    Retorna información sobre las opciones de generación disponibles
    """
    return {
        "levels": {
            "MINIMO": {
                "description": "Columnas mínimas requeridas",
                "columns": ["ID", "Edad", "Sexo", "Glucosa", "Presion_Sistolica", "Presion_Arterial", "Diagnostico"],
                "count": 7
            },
            "ESTANDAR": {
                "description": "Columnas estándar (incluye antropometría básica)",
                "columns": ["ID", "Edad", "Sexo", "Peso", "Altura", "IMC", "Presion_Sistolica", "Presion_Diastolica", "Presion_Arterial", "Glucosa", "Colesterol", "Fumador", "Diagnostico"],
                "count": 13
            },
            "COMPLETO": {
                "description": "Todas las columnas disponibles",
                "columns": ["ID", "Edad", "Sexo", "Peso", "Altura", "IMC", "Presion_Sistolica", "Presion_Diastolica", "Presion_Arterial", "Glucosa", "Colesterol", "Fumador", "Hemoglobina_A1C", "Colesterol_LDL", "Colesterol_HDL", "Trigliceridos", "Circunferencia_Cintura", "Circunferencia_Cadera", "Antecedentes_Familiares", "Creatinina", "Frecuencia_Cardiaca", "Actividad_Fisica", "Consumo_Alcohol", "Diagnostico"],
                "count": 24
            }
        },
        "formats": ["excel", "csv", "json"],
        "min_records": 100,
        "max_records": 50000,
        "diagnostics": ["Normal", "Prediabetes", "Hipertension", "Diabetes", "Obesidad"]
    }
