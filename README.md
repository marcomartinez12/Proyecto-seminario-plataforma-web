# Plataforma de Análisis Médico con Machine Learning

Sistema web para análisis de datos médicos que detecta diabetes e hipertensión usando Machine Learning.

---

## Para qué sirve

- Analizar datos médicos de pacientes (glucosa, presión arterial, edad, etc.)
- Predecir diagnósticos (Diabetes, Hipertensión, Normal) con 85-95% de precisión
- Generar reportes PDF con gráficos y estadísticas
- Aceptar múltiples formatos de archivos (Excel, CSV, JSON)
- Adaptar automáticamente el análisis según las columnas disponibles

---

## Dependencias principales

### Backend
```
fastapi              - Framework web
uvicorn              - Servidor ASGI
pandas               - Procesamiento de datos
numpy                - Cálculos numéricos
scikit-learn         - Machine Learning (Random Forest)
xgboost              - Modelo avanzado (XGBoost)
imbalanced-learn     - Balanceo de clases (SMOTE)
scipy                - Análisis estadístico
matplotlib           - Gráficos
seaborn              - Visualizaciones
reportlab            - Generación de PDFs
openpyxl             - Lectura de Excel
```

### Frontend
```
HTML5/CSS3/JavaScript puro
```

---

## Instalación rápida

### 1. Instalar dependencias
```bash
cd Proyecto-seminario-plataforma-web
pip install -r requirements.txt
```

### 2. Iniciar servidor
```bash
# Windows
start.bat

# O manualmente
python -m uvicorn app.main:app --reload
```

### 3. Abrir navegador
```
http://localhost:8000
```

---

## Cómo usar

### 1. Preparar datos

**Columnas MÍNIMAS requeridas:**
- Edad
- Sexo
- Glucosa
- Presion_Sistolica (o Presion_Arterial)

**Columnas RECOMENDADAS (mejoran precisión):**
- Peso, Altura
- Presion_Diastolica
- Colesterol
- Fumador

**Columnas AVANZADAS (máxima precisión):**
- Hemoglobina_A1C
- Colesterol_LDL, Colesterol_HDL
- Trigliceridos
- Circunferencia_Cintura
- Antecedentes_Familiares
- Creatinina
- Frecuencia_Cardiaca

**Formatos aceptados:**
- Excel (.xlsx, .xls)
- CSV (.csv)
- JSON (.json)
- TSV (.tsv, .txt)

### 2. Subir archivo

1. Clic en "Seleccionar archivo"
2. Elegir archivo con datos médicos
3. Clic en "Cargar archivo"

### 3. Generar análisis

1. Seleccionar archivo de la lista
2. Clic en "Generar Análisis"
3. Esperar procesamiento
4. Descargar reporte PDF

---

## Datasets de prueba incluidos

El sistema incluye 4 datasets sintéticos con datos coherentes:

```
dataset_medico_completo_10k_20251009.xlsx   - 10,000 registros, 24 columnas
dataset_medico_estandar_5k_20251009.xlsx    - 5,000 registros, 13 columnas
dataset_medico_minimo_2k_20251009.xlsx      - 2,000 registros, 7 columnas
dataset_medico_completo_10k_20251009.csv    - Formato CSV
```

**Precisión esperada con estos datasets:** 85-95%

---

## Características técnicas

### Sistema flexible de columnas

El sistema se adapta automáticamente a las columnas disponibles:

- **NIVEL 1 (Mínimo):** 4 columnas → Precisión 65-75%
- **NIVEL 2 (Estándar):** 9 columnas → Precisión 80-88%
- **NIVEL 3 (Completo):** 20+ columnas → Precisión 88-95%

### Detección automática

Reconoce nombres de columnas en español/inglés:
- "Age", "age", "edad", "EDAD" → Edad
- "Glucose", "glucosa", "Blood_Glucose" → Glucosa
- "A1C", "HbA1c" → Hemoglobina_A1C

### Auto-etiquetado clínico

Si el dataset no tiene columna "Diagnostico", la genera automáticamente:
- Glucosa > 126 → Diabetes
- Presión > 140 → Hipertensión
- Valores normales → Normal

### Modelo ML adaptativo

- Usa XGBoost con hiperparámetros dinámicos
- Aplica SMOTE para balanceo de clases
- Validación cruzada estratificada (5-fold)
- Feature engineering automático (hasta 40+ features)

---

## Estructura del proyecto

```
Proyecto-seminario-plataforma-web/
├── app/
│   ├── routers/
│   │   ├── files.py              - Manejo de archivos (multi-formato)
│   │   ├── analysis.py           - Análisis ML
│   │   ├── analysis_improved.py  - Funciones ML mejoradas
│   │   └── ai_analysis.py        - Explicaciones con IA
│   ├── schemas.py                - Modelos de datos
│   ├── utils/                    - Utilidades
│   └── main.py                   - Configuración FastAPI
├── static/                       - CSS, JavaScript
├── templates/                    - HTML
├── uploads/                      - Archivos subidos
├── downloads/                    - Reportes PDF generados
├── generar_dataset_sintetico.py  - Generador de datasets
├── requirements.txt              - Dependencias Python
├── start.bat                     - Inicio rápido (Windows)
└── README.md                     - Este archivo
```

---

## Solución de problemas

### Error: "Faltan columnas críticas"

**Causa:** El archivo no tiene las columnas mínimas.

**Solución:** Asegúrate de que tu archivo tenga al menos:
- Edad, Sexo, Glucosa, Presion_Sistolica

### Precisión baja (40-50%)

**Causa:** Etiquetas en columna "Diagnostico" son incorrectas.

**Solución:** Elimina la columna "Diagnostico" del Excel. El sistema la generará automáticamente con criterios médicos correctos.

### No se genera el reporte

**Causa:** Falta API key de OpenRouter (opcional).

**Solución:** El análisis ML funciona sin API key. Solo las explicaciones con IA requieren configuración adicional.

---

## Generar nuevos datasets sintéticos

```bash
python generar_dataset_sintetico.py
```

Esto crea 4 archivos con datos médicos coherentes listos para probar.

---

## Tecnologías utilizadas

- **Backend:** Python 3.8+, FastAPI
- **ML:** scikit-learn, XGBoost, SMOTE
- **Datos:** pandas, numpy, scipy
- **Visualización:** matplotlib, seaborn
- **PDF:** ReportLab
- **Frontend:** HTML5, CSS3, JavaScript

---

## Precisión del modelo

| Nivel de datos | Columnas | Precisión esperada |
|----------------|----------|-------------------|
| Mínimo | 4-6 | 65-75% |
| Estándar | 7-12 | 80-88% |
| Completo | 13+ | 88-95% |

**Nota:** La precisión depende de que las etiquetas sean clínicamente coherentes. Si usas auto-etiquetado, la precisión será 85-95%.

---

## Licencia

Proyecto académico - Seminario de Plataformas Web

---

## Versión

2.0 - Sistema mejorado con soporte multi-formato y ML adaptativo
