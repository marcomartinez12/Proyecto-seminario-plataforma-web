import json
import os
from datetime import datetime
from typing import List, Dict, Any
import uuid

class JSONStorage:
    def __init__(self, data_dir: str = "data"):
        self.data_dir = data_dir
        os.makedirs(data_dir, exist_ok=True)
        self.files_db = os.path.join(data_dir, "files.json")
        self.analysis_db = os.path.join(data_dir, "analysis.json")
        
        # Inicializar archivos JSON si no existen
        if not os.path.exists(self.files_db):
            self._save_json(self.files_db, [])
        if not os.path.exists(self.analysis_db):
            self._save_json(self.analysis_db, [])
    
    def _load_json(self, file_path: str) -> List[Dict[str, Any]]:
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        except (FileNotFoundError, json.JSONDecodeError):
            return []
    
    def _save_json(self, file_path: str, data: List[Dict[str, Any]]):
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
    
    def add_file(self, filename: str, original_filename: str, file_path: str, file_size: int, records_count: int = 0) -> str:
        files = self._load_json(self.files_db)
        file_id = str(uuid.uuid4())
        
        file_data = {
            "id": file_id,
            "filename": filename,
            "original_filename": original_filename,
            "file_path": file_path,
            "upload_date": datetime.now().isoformat(),
            "file_size": file_size,
            "status": "uploaded",
            "records_count": records_count
        }
        
        files.append(file_data)
        self._save_json(self.files_db, files)
        return file_id
    
    def get_file(self, file_id: str) -> Dict[str, Any]:
        files = self._load_json(self.files_db)
        for file_data in files:
            if file_data["id"] == file_id:
                return file_data
        return None
    
    def get_all_files(self) -> List[Dict[str, Any]]:
        return self._load_json(self.files_db)
    
    def update_file_status(self, file_id: str, status: str, records_count: int = None):
        files = self._load_json(self.files_db)
        for file_data in files:
            if file_data["id"] == file_id:
                file_data["status"] = status
                if records_count is not None:
                    file_data["records_count"] = records_count
                break
        self._save_json(self.files_db, files)
    
    def add_analysis(self, file_id: str, report_path: str, hypertension_cases: int,
                    diabetes_cases: int, total_records: int, accuracy_score: float, summary: str,
                    f1_score: float = None) -> str:
        analyses = self._load_json(self.analysis_db)
        analysis_id = str(uuid.uuid4())

        analysis_data = {
            "id": analysis_id,
            "file_id": file_id,
            "analysis_date": datetime.now().isoformat(),
            "report_path": report_path,
            "hypertension_cases": hypertension_cases,
            "diabetes_cases": diabetes_cases,
            "total_records": total_records,
            "accuracy_score": accuracy_score,
            "f1_score": f1_score if f1_score is not None else accuracy_score,
            "summary": summary
        }
        
        analyses.append(analysis_data)
        self._save_json(self.analysis_db, analyses)
        return analysis_id
    
    def get_analysis(self, analysis_id: str) -> Dict[str, Any]:
        analyses = self._load_json(self.analysis_db)
        for analysis_data in analyses:
            if analysis_data["id"] == analysis_id:
                return analysis_data
        return None
    
    def get_analyses_by_file(self, file_id: str) -> List[Dict[str, Any]]:
        analyses = self._load_json(self.analysis_db)
        return [analysis for analysis in analyses if analysis["file_id"] == file_id]
    
    def delete_file(self, file_id: str) -> bool:
        """Eliminar completamente un archivo del JSON"""
        files = self._load_json(self.files_db)
        original_length = len(files)
        
        # Filtrar el archivo a eliminar
        files = [file_data for file_data in files if file_data["id"] != file_id]
        
        if len(files) < original_length:
            self._save_json(self.files_db, files)
            return True
        return False
    
    def delete_analyses_by_file(self, file_id: str):
        """Eliminar todos los análisis asociados a un archivo"""
        analyses = self._load_json(self.analysis_db)
        analyses = [analysis for analysis in analyses if analysis["file_id"] != file_id]
        self._save_json(self.analysis_db, analyses)

# Instancia global del almacenamiento
storage = JSONStorage()