from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
import uvicorn
import os
from pathlib import Path
from dotenv import load_dotenv

# Cargar variables de entorno desde .env
load_dotenv()

# Importar módulos locales
from app.routers import files, analysis, ai_analysis, detailed_analysis, synthetic_data, scenarios

app = FastAPI(
    title="Plataforma de Detección de Enfermedades Crónicas",
    description="API para análisis de datos médicos y detección de tendencias",
    version="1.0.0"
)

# Configurar CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # En producción, especificar dominios específicos
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Servir archivos estáticos
app.mount("/static", StaticFiles(directory="static"), name="static")

# Incluir routers
app.include_router(files.router, prefix="/api/files", tags=["files"])
app.include_router(analysis.router, prefix="/api/analysis", tags=["analysis"])
app.include_router(ai_analysis.router, prefix="/api", tags=["ai-analysis"])
app.include_router(detailed_analysis.router, prefix="/api", tags=["detailed-analysis"])
app.include_router(synthetic_data.router, prefix="/api/synthetic", tags=["synthetic-data"])
app.include_router(scenarios.router, prefix="/api", tags=["scenarios"])

# Crear directorios necesarios
os.makedirs("uploads", exist_ok=True)
os.makedirs("reports", exist_ok=True)
os.makedirs("downloads", exist_ok=True)
os.makedirs("data", exist_ok=True)
os.makedirs("static", exist_ok=True)

@app.get("/")
async def root():
    return FileResponse('static/index.html')

@app.get("/health")
async def health_check():
    return {"status": "healthy"}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)