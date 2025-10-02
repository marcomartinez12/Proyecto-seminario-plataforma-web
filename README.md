# Plataforma de Análisis Médico

## Descripción

Aplicación web para análisis de datos médicos con detección de hipertensión y diabetes mediante Machine Learning.

## Funcionalidades

- Carga de archivos Excel con datos médicos
- Análisis predictivo con Random Forest
- Generación de reportes PDF con gráficos
- Explicaciones de resultados con IA

## Tecnologías

### Backend
- **FastAPI** - Framework web
- **Python 3.8+**
- **scikit-learn** - Machine Learning
- **pandas** - Procesamiento de datos
- **matplotlib/seaborn** - Gráficos
- **ReportLab** - Generación de PDFs

### Frontend
- HTML5/CSS3/JavaScript
- Fetch API

## Dependencias

```txt
fastapi
uvicorn[standard]
python-multipart
pandas
numpy
scikit-learn
matplotlib
seaborn
reportlab
openpyxl
httpx
```

## Instalación

### 1. Clonar repositorio
```bash
git clone <url-del-repositorio>
cd Proyecto-seminario-plataforma-web
```

### 2. Crear entorno virtual
```bash
python -m venv venv

# Windows
venv\Scripts\activate

# Linux/Mac
source venv/bin/activate
```

### 3. Instalar dependencias
```bash
pip install -r requirements.txt
```

### 4. Configurar API key de OpenRouter
Editar la clave API en `app/routers/ai_analysis.py` línea 101 o configurar como variable de entorno:
```bash
set OPENROUTER_API_KEY=tu-clave-aqui
```

### 5. Ejecutar aplicación
```bash
python main.py
```

La aplicación estará en: `http://localhost:8000`

## Formato de datos

El archivo Excel debe contener las siguientes columnas:

- ID
- Edad
- Sexo
- Peso
- Altura
- Presion_Arterial (formato: "120/80")
- Glucosa
- Colesterol
- Fumador ("Si" o "No")
- Diagnostico

## Estructura del proyecto

```
Proyecto-seminario-plataforma-web/
├── app/
│   ├── routers/         # Endpoints de la API
│   ├── schemas.py       # Modelos de datos
│   └── utils/           # Utilidades
├── static/              # Archivos estáticos (CSS, JS)
├── templates/           # Templates HTML
├── uploads/             # Archivos cargados
├── downloads/           # Reportes generados
├── main.py             # Punto de entrada
└── requirements.txt    # Dependencias
```
