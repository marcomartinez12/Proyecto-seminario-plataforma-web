from fastapi import APIRouter, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
import os
import pandas as pd
from typing import List
import uuid
from datetime import datetime

from app.schemas import FileUploadResponse
from app.utils import storage

router = APIRouter()

ALLOWED_EXTENSIONS = {".xlsx", ".xls", ".csv", ".json", ".tsv", ".txt"}
UPLOAD_DIR = "uploads"

# Sistema de columnas flexibles con 3 niveles
COLUMN_LEVELS = {
    "NIVEL_1_MINIMO": {
        "required": ["Edad", "Sexo", "Glucosa", "Presion_Sistolica"],
        "aliases": {
            "Edad": ["Age", "age", "edad", "EDAD"],
            "Sexo": ["Gender", "gender", "sexo", "SEXO", "Sex"],
            "Glucosa": ["Glucose", "glucose", "glucosa", "GLUCOSA", "Blood_Glucose"],
            "Presion_Sistolica": ["Systolic", "systolic", "Presion_Arterial", "BP_Sys", "SBP", "Systolic_BP"]
        }
    },
    "NIVEL_2_ESTANDAR": {
        "optional": ["Peso", "Altura", "Presion_Diastolica", "Colesterol", "Fumador"],
        "aliases": {
            "Peso": ["Weight", "weight", "peso", "PESO"],
            "Altura": ["Height", "height", "altura", "ALTURA"],
            "Presion_Diastolica": ["Diastolic", "diastolic", "DBP", "Diastolic_BP"],
            "Colesterol": ["Cholesterol", "cholesterol", "colesterol", "COLESTEROL"],
            "Fumador": ["Smoker", "smoker", "fumador", "FUMADOR", "Smoking"]
        }
    },
    "NIVEL_3_COMPLETO": {
        "optional": [
            "Hemoglobina_A1C", "Colesterol_LDL", "Colesterol_HDL",
            "Trigliceridos", "Circunferencia_Cintura", "Circunferencia_Cadera",
            "Antecedentes_Familiares", "Creatinina", "Frecuencia_Cardiaca",
            "Actividad_Fisica", "Consumo_Alcohol"
        ],
        "aliases": {
            "Hemoglobina_A1C": ["A1C", "HbA1c", "Hemoglobin_A1C", "Glycated_Hemoglobin"],
            "Colesterol_LDL": ["LDL", "LDL_Cholesterol", "Bad_Cholesterol"],
            "Colesterol_HDL": ["HDL", "HDL_Cholesterol", "Good_Cholesterol"],
            "Trigliceridos": ["Triglycerides", "triglycerides", "TG"],
            "Circunferencia_Cintura": ["Waist", "waist", "Waist_Circumference", "WC"],
            "Circunferencia_Cadera": ["Hip", "hip", "Hip_Circumference", "HC"],
            "Antecedentes_Familiares": ["Family_History", "family_history", "FH"],
            "Creatinina": ["Creatinine", "creatinine", "Cr"],
            "Frecuencia_Cardiaca": ["Heart_Rate", "heart_rate", "HR", "Pulse"],
            "Actividad_Fisica": ["Physical_Activity", "physical_activity", "Exercise"],
            "Consumo_Alcohol": ["Alcohol", "alcohol", "Alcohol_Consumption"]
        }
    }
}

# Columnas mínimas obligatorias
REQUIRED_COLUMNS = COLUMN_LEVELS["NIVEL_1_MINIMO"]["required"]

def load_file(file_path: str) -> pd.DataFrame:
    """Cargar archivo en múltiples formatos"""
    file_extension = os.path.splitext(file_path)[1].lower()

    try:
        if file_extension in ['.xlsx', '.xls']:
            return pd.read_excel(file_path)
        elif file_extension == '.csv':
            return pd.read_csv(file_path)
        elif file_extension == '.json':
            return pd.read_json(file_path)
        elif file_extension in ['.tsv', '.txt']:
            return pd.read_csv(file_path, sep='\t')
        else:
            raise ValueError(f"Formato no soportado: {file_extension}")
    except Exception as e:
        raise Exception(f"Error al leer archivo: {str(e)}")

def auto_map_columns(df: pd.DataFrame) -> tuple[pd.DataFrame, dict, str]:
    """
    Detectar y mapear columnas automáticamente usando aliases.
    Retorna: (DataFrame mapeado, diccionario de mapeo, nivel detectado)
    """
    column_mapping = {}
    df_mapped = df.copy()
    available_columns = set(df.columns)

    # Función para encontrar columna por aliases
    def find_column(standard_name: str, aliases: list) -> str:
        # Primero buscar nombre exacto
        if standard_name in available_columns:
            return standard_name
        # Luego buscar en aliases
        for alias in aliases:
            if alias in available_columns:
                return alias
        return None

    # Mapear columnas de todos los niveles
    all_aliases = {}
    for level_name, level_data in COLUMN_LEVELS.items():
        if "aliases" in level_data:
            all_aliases.update(level_data["aliases"])

    # Encontrar mapeos
    for standard_name, aliases in all_aliases.items():
        found_column = find_column(standard_name, aliases)
        if found_column and found_column != standard_name:
            column_mapping[found_column] = standard_name
            df_mapped.rename(columns={found_column: standard_name}, inplace=True)

    # Determinar nivel de datos disponibles
    nivel_detectado = "NIVEL_1_MINIMO"
    columnas_disponibles = set(df_mapped.columns)

    nivel2_cols = set(COLUMN_LEVELS["NIVEL_2_ESTANDAR"]["optional"])
    nivel3_cols = set(COLUMN_LEVELS["NIVEL_3_COMPLETO"]["optional"])

    if len(columnas_disponibles & nivel3_cols) >= 5:
        nivel_detectado = "NIVEL_3_COMPLETO"
    elif len(columnas_disponibles & nivel2_cols) >= 3:
        nivel_detectado = "NIVEL_2_ESTANDAR"

    return df_mapped, column_mapping, nivel_detectado

def validate_file(file_path: str) -> tuple[bool, str, int, dict, str]:
    """
    Valida archivo con sistema flexible de columnas.
    Retorna: (válido, mensaje, num_registros, mapeo_columnas, nivel_datos)
    """
    try:
        # Cargar archivo
        df = load_file(file_path)

        # Verificar que no esté vacío
        if len(df) == 0:
            return False, "El archivo está vacío", 0, {}, ""

        # Auto-mapear columnas
        df_mapped, column_mapping, nivel_detectado = auto_map_columns(df)

        # Verificar columnas mínimas requeridas
        missing_columns = [col for col in REQUIRED_COLUMNS if col not in df_mapped.columns]

        if missing_columns:
            return False, f"Faltan columnas críticas: {', '.join(missing_columns)}", 0, {}, ""

        # Generar mensaje informativo sobre nivel de datos
        mensaje_nivel = f"Archivo válido - {nivel_detectado}"
        if nivel_detectado == "NIVEL_1_MINIMO":
            mensaje_nivel += " (Solo datos básicos. Precisión estimada: 60-70%)"
        elif nivel_detectado == "NIVEL_2_ESTANDAR":
            mensaje_nivel += " (Datos estándar. Precisión estimada: 75-85%)"
        else:
            mensaje_nivel += " (Datos completos. Precisión estimada: 85-95%)"

        # Guardar DataFrame mapeado temporalmente para análisis
        temp_path = file_path.replace('.xlsx', '_mapped.xlsx').replace('.csv', '_mapped.csv')
        if file_path.endswith(('.xlsx', '.xls')):
            df_mapped.to_excel(temp_path, index=False)
        else:
            df_mapped.to_csv(temp_path, index=False)

        return True, mensaje_nivel, len(df_mapped), column_mapping, nivel_detectado

    except Exception as e:
        return False, f"Error al procesar archivo: {str(e)}", 0, {}, ""

@router.post("/upload", response_model=FileUploadResponse)
async def upload_file(file: UploadFile = File(...)):
    """Subir archivo (Excel, CSV, JSON, etc.) para análisis"""

    # Validar extensión
    file_extension = os.path.splitext(file.filename)[1].lower()
    if file_extension not in ALLOWED_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail=f"Tipo de archivo no permitido. Formatos aceptados: {', '.join(ALLOWED_EXTENSIONS)}"
        )

    # Generar nombre único para el archivo
    unique_filename = f"{uuid.uuid4()}{file_extension}"
    file_path = os.path.join(UPLOAD_DIR, unique_filename)

    try:
        # Guardar archivo
        with open(file_path, "wb") as buffer:
            content = await file.read()
            buffer.write(content)

        # Validar contenido con sistema flexible
        is_valid, message, records_count, column_mapping, nivel_datos = validate_file(file_path)

        if not is_valid:
            # Eliminar archivo si no es válido
            os.remove(file_path)
            raise HTTPException(status_code=400, detail=message)

        # Guardar información en JSON
        file_id = storage.add_file(
            filename=unique_filename,
            original_filename=file.filename,
            file_path=file_path,
            file_size=len(content),
            records_count=records_count
        )

        print(f"\n✓ Archivo cargado: {file.filename}")
        print(f"  - Nivel de datos: {nivel_datos}")
        print(f"  - Registros: {records_count:,}")
        if column_mapping:
            print(f"  - Columnas mapeadas: {len(column_mapping)}")
            for old_col, new_col in column_mapping.items():
                print(f"    {old_col} -> {new_col}")

        return FileUploadResponse(
            id=file_id,
            filename=unique_filename,
            original_filename=file.filename,
            upload_date=datetime.now().isoformat(),
            file_size=len(content),
            status="uploaded",
            records_count=records_count
        )

    except Exception as e:
        # Limpiar archivo si hay error
        if os.path.exists(file_path):
            os.remove(file_path)
        raise HTTPException(status_code=500, detail=f"Error al procesar archivo: {str(e)}")

@router.get("/list", response_model=List[FileUploadResponse])
async def list_files():
    """Listar todos los archivos subidos"""
    files = storage.get_all_files()
    return [
        FileUploadResponse(
            id=file_data["id"],
            filename=file_data["filename"],
            original_filename=file_data["original_filename"],
            upload_date=file_data["upload_date"],
            file_size=file_data["file_size"],
            status=file_data["status"],
            records_count=file_data["records_count"]
        )
        for file_data in files
    ]

@router.get("/{file_id}", response_model=FileUploadResponse)
async def get_file_info(file_id: str):
    """Obtener información de un archivo específico"""
    file_data = storage.get_file(file_id)
    if not file_data:
        raise HTTPException(status_code=404, detail="Archivo no encontrado")
    
    return FileUploadResponse(
        id=file_data["id"],
        filename=file_data["filename"],
        original_filename=file_data["original_filename"],
        upload_date=file_data["upload_date"],
        file_size=file_data["file_size"],
        status=file_data["status"],
        records_count=file_data["records_count"]
    )

@router.delete("/{file_id}")
async def delete_file(file_id: str):
    """Eliminar un archivo completamente"""
    file_data = storage.get_file(file_id)
    if not file_data:
        raise HTTPException(status_code=404, detail="Archivo no encontrado")
    
    # Eliminar archivo físico
    if os.path.exists(file_data["file_path"]):
        os.remove(file_data["file_path"])
    
    # Eliminar análisis asociados
    storage.delete_analyses_by_file(file_id)
    
    # Eliminar completamente del JSON
    if storage.delete_file(file_id):
        return {"message": "Archivo eliminado completamente"}
    else:
        raise HTTPException(status_code=500, detail="Error al eliminar el archivo")