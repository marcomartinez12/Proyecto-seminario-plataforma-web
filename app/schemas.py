from pydantic import BaseModel
from datetime import datetime
from typing import Optional, List

class FileUploadResponse(BaseModel):
    id: str
    filename: str
    original_filename: str
    upload_date: str
    file_size: int
    status: str
    records_count: int

class AnalysisRequest(BaseModel):
    file_id: str

class AnalysisResponse(BaseModel):
    id: str
    file_id: str
    analysis_date: str
    report_path: str
    hypertension_cases: int
    diabetes_cases: int
    total_records: int
    accuracy_score: float
    summary: str

class MedicalRecord(BaseModel):
    ID: int
    Edad: int
    Sexo: str
    Peso: float
    Altura: float
    Presion_Arterial: str
    Glucosa: float
    Colesterol: float
    Fumador: bool
    Diagnostico: str