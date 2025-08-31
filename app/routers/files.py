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

ALLOWED_EXTENSIONS = {".xlsx", ".xls"}
UPLOAD_DIR = "uploads"

# Columnas requeridas en el Excel
REQUIRED_COLUMNS = [
    "ID", "Edad", "Sexo", "Peso", "Altura", 
    "Presion_Arterial", "Glucosa", "Colesterol", "Fumador", "Diagnostico"
]

def validate_excel_file(file_path: str) -> tuple[bool, str, int]:
    """Valida que el archivo Excel tenga las columnas correctas"""
    try:
        df = pd.read_excel(file_path)
        
        # Verificar columnas requeridas
        missing_columns = [col for col in REQUIRED_COLUMNS if col not in df.columns]
        if missing_columns:
            return False, f"Faltan las siguientes columnas: {', '.join(missing_columns)}", 0
        
        # Verificar que no esté vacío
        if len(df) == 0:
            return False, "El archivo está vacío", 0
        
        return True, "Archivo válido", len(df)
    
    except Exception as e:
        return False, f"Error al leer el archivo: {str(e)}", 0

@router.post("/upload", response_model=FileUploadResponse)
async def upload_file(file: UploadFile = File(...)):
    """Subir archivo Excel para análisis"""
    
    # Validar extensión
    file_extension = os.path.splitext(file.filename)[1].lower()
    if file_extension not in ALLOWED_EXTENSIONS:
        raise HTTPException(
            status_code=400, 
            detail=f"Tipo de archivo no permitido. Use: {', '.join(ALLOWED_EXTENSIONS)}"
        )
    
    # Generar nombre único para el archivo
    unique_filename = f"{uuid.uuid4()}{file_extension}"
    file_path = os.path.join(UPLOAD_DIR, unique_filename)
    
    try:
        # Guardar archivo
        with open(file_path, "wb") as buffer:
            content = await file.read()
            buffer.write(content)
        
        # Validar contenido del Excel
        is_valid, message, records_count = validate_excel_file(file_path)
        
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