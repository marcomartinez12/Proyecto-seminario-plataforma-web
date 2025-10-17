# Plataforma de Análisis Médico con Machine Learning

Sistema web inteligente para análisis predictivo de datos médicos que detecta diabetes e hipertensión usando XGBoost con explicaciones generadas por IA.

---

## Para qué sirve

- Analizar datos médicos de pacientes (glucosa, presión arterial, edad, IMC, etc.)
- Predecir diagnósticos (Diabetes, Hipertensión, Normal) con 85-95% de precisión usando XGBoost
- Generar reportes PDF profesionales con gráficos, estadísticas y análisis clínico
- Explicaciones médicas en lenguaje simple generadas por IA (GPT-OSS-20B)
- Aceptar múltiples formatos de archivos (Excel, CSV, JSON, TSV)
- Adaptar automáticamente el análisis según las columnas disponibles
- Detección de comorbilidades (HTA + Diabetes)

---

## Dependencias principales

### Backend
```
fastapi              - Framework web
uvicorn              - Servidor ASGI
pandas               - Procesamiento de datos
numpy                - Cálculos numéricos
xgboost              - Modelo de Machine Learning (XGBoost Classifier)
scikit-learn         - Preprocesamiento y métricas ML
imbalanced-learn     - Balanceo de clases (SMOTE)
scipy                - Análisis estadístico
matplotlib           - Gráficos
seaborn              - Visualizaciones
reportlab            - Generación de PDFs profesionales
openpyxl             - Lectura de Excel
httpx                - Cliente HTTP para API de IA
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

### 2. Configurar variables de entorno (opcional)

Para habilitar explicaciones con IA, crea un archivo `.env`:
```bash
OPENROUTER_API_KEY=tu_api_key_aqui
```

**Nota:** El sistema funciona completamente sin API key. Solo las explicaciones con IA requieren esta configuración.

### 3. Iniciar servidor
```bash
# Windows
start.bat

# O manualmente
python main.py
```

### 4. Abrir navegador
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
3. Esperar procesamiento (el modelo XGBoost entrenará con tus datos)
4. Descargar reporte PDF profesional

### 4. Obtener explicaciones con IA (opcional)

1. Después de generar el análisis
2. Clic en "Explicar con IA"
3. Lee la explicación médica en lenguaje simple generada por GPT-OSS-20B

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

### Modelo ML con XGBoost

**Configuración del modelo:**
- Algoritmo: XGBoost Classifier
- Estimadores: 300 árboles
- Learning rate: 0.05
- Profundidad máxima: 8
- Balanceo: SMOTE para clases desbalanceadas
- Validación: Cross-validation estratificada (5-fold)
- Feature engineering: Hasta 40+ características automáticas

**Hiperparámetros avanzados:**
- subsample: 0.8
- colsample_bytree: 0.8
- min_child_weight: 3
- gamma: 0.1
- scale_pos_weight: balance automático

---

## Estructura del proyecto

```
Proyecto-seminario-plataforma-web/
├── app/
│   ├── routers/
│   │   ├── files.py              - Gestión de archivos multi-formato
│   │   ├── analysis.py           - Análisis ML con XGBoost (PRINCIPAL)
│   │   ├── analysis_improved.py  - Funciones auxiliares (no usado)
│   │   └── ai_analysis.py        - Explicaciones con IA (GPT-OSS-20B)
│   ├── schemas.py                - Modelos de datos Pydantic
│   ├── utils/
│   │   └── storage.py            - Almacenamiento en memoria
│   └── main.py                   - Punto de entrada FastAPI
├── static/
│   ├── index.html                - Interfaz web principal
│   ├── script.js                 - Lógica del frontend
│   └── styles.css                - Estilos (si existe)
├── uploads/                      - Archivos subidos temporalmente
├── downloads/                    - Reportes PDF generados
├── data/                         - Datasets de prueba
├── generar_dataset_sintetico.py  - Generador de datasets
├── requirements.txt              - Dependencias Python
├── main.py                       - Launcher del servidor
├── start.bat                     - Inicio rápido (Windows)
├── .env                          - Variables de entorno (crear manualmente)
└── README.md                     - Documentación
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

**Causa:** Error durante el entrenamiento del modelo o datos insuficientes.

**Solución:** Verifica que tengas al menos 50 registros con las columnas mínimas requeridas.

### Explicaciones con IA no funcionan

**Causa:** No está configurada la API key de OpenRouter.

**Solución:**
1. Crea un archivo `.env` en la raíz del proyecto
2. Añade: `OPENROUTER_API_KEY=tu_key_aqui`
3. Reinicia el servidor

**Nota:** El análisis ML funciona perfectamente sin IA. Las explicaciones son una característica adicional opcional.

---

## Generar nuevos datasets sintéticos

```bash
python generar_dataset_sintetico.py
```

Esto crea 4 archivos con datos médicos coherentes listos para probar.

---

## Tecnologías utilizadas

- **Backend:** Python 3.8+, FastAPI, Uvicorn
- **Machine Learning:** XGBoost Classifier (300 estimadores)
- **Balanceo de datos:** SMOTE (imbalanced-learn)
- **Preprocesamiento:** scikit-learn (LabelEncoder, train_test_split)
- **Análisis de datos:** pandas, numpy, scipy
- **Visualización:** matplotlib, seaborn
- **Generación de PDFs:** ReportLab (reportes profesionales)
- **IA:** OpenRouter API con GPT-OSS-20B (explicaciones médicas)
- **Frontend:** HTML5, CSS3, JavaScript vanilla
- **Comunicación:** httpx (cliente HTTP async)

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

## Características del Reporte PDF

El reporte generado incluye:

1. **Portada profesional** con identificador único
2. **Resumen ejecutivo** con estadísticas clave
3. **Análisis descriptivo** con gráficos estadísticos
4. **Matriz de confusión** del modelo XGBoost
5. **Importancia de características** (top 10 features)
6. **Análisis de comorbilidad** HTA + Diabetes
7. **Conclusiones y recomendaciones** estratégicas
8. **Pie de página** con fecha y numeración

**Nota:** La sección "ANEXO: DATOS CLAVE PARA PUBLICACIÓN CIENTÍFICA" está comentada en esta versión.

---

## Endpoints de la API

### Archivos
- `POST /api/files/upload` - Subir archivo
- `GET /api/files/list` - Listar archivos
- `DELETE /api/files/{file_id}` - Eliminar archivo

### Análisis
- `POST /api/analysis/analyze` - Generar análisis ML
- `GET /api/analysis/download/{analysis_id}` - Descargar PDF
- `GET /api/analysis/results/{file_id}` - Obtener resultados

### IA
- `POST /api/ai-analysis/{file_id}` - Generar explicación con IA

---

## Modelo de IA para Explicaciones

- **Proveedor:** OpenRouter
- **Modelo:** `openai/gpt-oss-20b:free`
- **Temperatura:** 0.7
- **Max tokens:** 1500
- **Propósito:** Traducir análisis técnico a lenguaje accesible

El modelo recibe las estadísticas del análisis y genera explicaciones que incluyen:
- Resumen general en palabras simples
- Hallazgos importantes explicados claramente
- Interpretación de los números (ej: si un IMC promedio es alto/bajo/normal)
- Recomendaciones prácticas de prevención
- Conclusión sobre la situación general de salud

---

## Historial de Cambios

### v3.0 (Actual)
- ✅ Modelo cambiado de Random Forest a **XGBoost** (300 estimadores)
- ✅ Integración con **IA (GPT-OSS-20B)** para explicaciones médicas
- ✅ Reportes PDF profesionales con 7 secciones
- ✅ Análisis de comorbilidades HTA + Diabetes
- ✅ Sección de anexo científico comentada (opcional)
- ✅ Interfaz web con modal de IA
- ✅ Soporte multi-formato (Excel, CSV, JSON, TSV)

### v2.0
- Sistema con ML adaptativo
- Soporte multi-formato de archivos
- Auto-detección de columnas

### v1.0
- Sistema básico con Random Forest
- Solo archivos Excel

---

## Contribuciones

Este es un proyecto académico. Si deseas contribuir:

1. Asegúrate de que el código sea para fines educativos/médicos defensivos
2. Documenta tus cambios
3. Prueba con los datasets incluidos

---

## Contacto y Soporte

Para reportar problemas o sugerencias, revisa la documentación del código fuente en los archivos principales:
- [analysis.py](app/routers/analysis.py) - Lógica principal del análisis ML
- [ai_analysis.py](app/routers/ai_analysis.py) - Integración con IA
- [main.py](main.py) - Configuración del servidor

---

## Versión

**3.0** - Sistema profesional con XGBoost, explicaciones IA y reportes médicos completos
