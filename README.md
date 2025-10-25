# Plataforma de Análisis Médico con Machine Learning

Sistema web inteligente para análisis predictivo de datos médicos que detecta diabetes e hipertensión usando XGBoost con explicaciones generadas por IA y visualizaciones interactivas.

---

## Para qué sirve

- Analizar datos médicos de pacientes (glucosa, presión arterial, edad, IMC, etc.)
- Predecir diagnósticos (Diabetes, Hipertensión, Normal) con 85-95% de precisión usando XGBoost
- Generar reportes PDF profesionales con gráficos, estadísticas y análisis clínico
- Visualizar gráficas interactivas del análisis en el navegador
- Análisis detallado paso a paso del funcionamiento del modelo XGBoost
- Explicaciones médicas en lenguaje simple generadas por IA (GPT-OSS-20B)
- Aceptar múltiples formatos de archivos (Excel, CSV, JSON, TSV)
- Adaptar automáticamente el análisis según las columnas disponibles
- Detección de comorbilidades (HTA + Diabetes)

---

## Instalación rápida

### 1. Clonar o descargar el proyecto
```bash
cd "c:\seminario\aplicativo web\Proyecto-seminario-plataforma-web"
```

### 2. Instalar dependencias (automático con start.bat)
Las dependencias se instalan automáticamente al ejecutar `start.bat`

**O manualmente:**
```bash
pip install -r requirements.txt
```

### 3. Iniciar servidor
```bash
# Windows (RECOMENDADO - Abre navegador automáticamente)
start.bat

# O manualmente
python main.py
```

### 4. Navegador
El navegador se abrirá automáticamente en `http://127.0.0.1:8000` después de 3 segundos.

**IMPORTANTE:** Debes usar `http://127.0.0.1:8000` (no Live Server) porque la aplicación requiere el backend FastAPI ejecutándose.

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
4. Verás una animación de carga elegante

### 3. Generar análisis

1. Seleccionar archivo de la lista
2. Clic en "Analizar"
3. Esperar procesamiento con animación simplificada que muestra:
   - Validación del archivo (20%)
   - Entrenamiento del modelo XGBoost (40%)
   - Generación de análisis estadístico (65%)
   - Creación de gráficos (85%)
   - Finalización del análisis (100%)
4. El análisis se completa **SIN descargar automáticamente** el PDF

### 4. Explorar resultados

Una vez completado el análisis, tienes 4 opciones:

**📊 Ver Gráficas**
- Modal con 4 gráficas interactivas de Chart.js:
  1. Distribución de Diagnósticos (Pie Chart)
  2. Matriz de Confusión del Modelo
  3. Importancia de Características
  4. Métricas de Precisión por Clase

**🔬 Análisis Detallado** (NUEVO)
- Se abre en nueva pestaña
- Explicación educativa paso a paso de cómo funciona XGBoost
- 10 pasos con gráficos y explicaciones sencillas
- Diseño dark mode estilo "Grok AI"
- Fuente Inter para mejor legibilidad

**🧠 Análisis IA**
- Explicación en lenguaje simple generada por IA
- Resumen de hallazgos importantes
- Recomendaciones prácticas

**📥 Descargar Reporte**
- Descarga el PDF profesional completo
- Solo se descarga cuando haces clic (no automáticamente)

### 5. Gestionar archivos

**Eliminar archivo:**
- Clic en el botón "Eliminar"
- Confirmar eliminación
- La lista se actualiza **automáticamente sin refrescar la página**
- Si eliminas todos los archivos, verás el mensaje "No hay archivos subidos aún"

---

## Características del proyecto

### Interfaz de Usuario

**Tema Dark Mode**
- Diseño negro mate (#0a0a0a)
- Fuente: Inter (similar a Grok AI)
- Tarjetas con gradientes sutiles
- Animaciones suaves y modernas
- Iconos de Font Awesome 6.0

**Animaciones**
- Carga de archivos con barra de progreso fluida
- Análisis con spinner circular, barra de progreso y paso actual
- Efectos shimmer y glow
- Transiciones suaves en todos los elementos

**Responsive Design**
- Se adapta a diferentes tamaños de pantalla
- Contenedor de 1400px de ancho para mostrar todos los botones

### Backend y Almacenamiento

**Sistema de archivos JSON** (NO base de datos SQL tradicional):
```
data/
├── files.json      - Información de archivos subidos
└── analysis.json   - Resultados de análisis realizados
```

**Carpetas de almacenamiento:**
```
uploads/   - Archivos Excel/CSV subidos
reports/   - PDFs generados
```

### Modelo de Machine Learning

**XGBoost Classifier optimizado:**
- 300 árboles de decisión
- Learning rate: 0.05
- Profundidad máxima: 8
- Balanceo con SMOTE para clases desbalanceadas
- Validación cruzada estratificada (5-fold)
- Feature engineering automático (hasta 40+ características)

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
│   │   ├── analysis.py           - Análisis ML con XGBoost
│   │   ├── ai_analysis.py        - Explicaciones con IA
│   │   └── detailed_analysis.py  - Análisis educativo paso a paso (NUEVO)
│   ├── schemas.py                - Modelos de datos Pydantic
│   └── utils.py                  - JSONStorage para archivos JSON
├── static/
│   ├── index.html                - Interfaz web principal
│   ├── script.js                 - Lógica del frontend
│   └── styles.css                - Estilos dark mode
├── data/
│   ├── files.json                - Base de datos de archivos
│   └── analysis.json             - Base de datos de análisis
├── uploads/                      - Archivos subidos temporalmente
├── reports/                      - Reportes PDF generados
├── requirements.txt              - Dependencias Python
├── main.py                       - Launcher del servidor FastAPI
├── start.bat                     - Inicio rápido (abre navegador automáticamente)
├── .env                          - Variables de entorno (opcional, para IA)
└── README.md                     - Esta documentación
```

---

## Endpoints de la API

### Archivos
- `POST /api/files/upload` - Subir archivo (Excel, CSV, JSON, TSV)
- `GET /api/files/list` - Listar todos los archivos
- `DELETE /api/files/{file_id}` - Eliminar archivo y sus análisis

### Análisis
- `POST /api/analysis/analyze` - Generar análisis ML con XGBoost
- `GET /api/analysis/download/{analysis_id}` - Descargar PDF
- `GET /api/analysis/results/{file_id}` - Obtener resultados JSON
- `GET /api/analysis/charts/{file_id}` - Obtener datos para gráficos

### Análisis Detallado (NUEVO)
- `GET /api/detailed-analysis/{file_id}` - Página educativa paso a paso

### IA
- `POST /api/ai-analysis/{file_id}` - Generar explicación con IA

### Documentación
- `GET /docs` - Swagger UI interactivo
- `GET /redoc` - Documentación alternativa

---

## Dependencias principales

### Backend
```
fastapi              - Framework web moderno
uvicorn              - Servidor ASGI de alto rendimiento
pandas               - Procesamiento de datos tabulares
numpy                - Cálculos numéricos
xgboost              - Modelo de Machine Learning
scikit-learn         - Preprocesamiento y métricas
imbalanced-learn     - Balanceo de clases (SMOTE)
scipy                - Análisis estadístico
matplotlib           - Gráficos estáticos
seaborn              - Visualizaciones estadísticas
reportlab            - Generación de PDFs
openpyxl             - Lectura de archivos Excel
httpx                - Cliente HTTP async para IA
python-dotenv        - Variables de entorno
```

### Frontend
```
HTML5/CSS3/JavaScript vanilla
Chart.js             - Gráficas interactivas
Font Awesome 6.0     - Iconos
```

---

## Configuración de IA (Opcional)

Para habilitar explicaciones con IA, crea un archivo `.env` en la raíz:

```bash
OPENROUTER_API_KEY=tu_api_key_aqui
```

**Modelo usado:**
- Proveedor: OpenRouter
- Modelo: `openai/gpt-oss-20b:free`
- Temperatura: 0.7
- Max tokens: 1500

**Nota:** El sistema funciona completamente sin IA. Las explicaciones son opcionales.

---

## Características del Reporte PDF

El reporte profesional incluye:

1. **Portada** con identificador único y fecha
2. **Resumen ejecutivo** con KPIs principales
3. **Análisis descriptivo** con estadísticas
4. **Matriz de confusión** del modelo
5. **Importancia de características** (top 10)
6. **Análisis de comorbilidad** HTA + Diabetes
7. **Conclusiones y recomendaciones**
8. **Pie de página** con numeración

---

## Solución de problemas

### Error: "Faltan columnas críticas"
**Causa:** El archivo no tiene las columnas mínimas.

**Solución:** Asegúrate de que tu archivo tenga al menos: Edad, Sexo, Glucosa, Presion_Sistolica

### No puedo ver el botón "Eliminar"
**Causa:** Ventana del navegador muy pequeña.

**Solución:** El contenedor tiene 1400px de ancho. Maximiza la ventana o usa zoom menor.

### La lista no se actualiza después de eliminar
**Causa:** Caché del navegador.

**Solución:** Presiona Ctrl+Shift+R para forzar recarga completa.

### El navegador no se abre automáticamente con start.bat
**Causa:** El servidor tarda más de 3 segundos en iniciar.

**Solución:**
1. Abre manualmente `http://127.0.0.1:8000`
2. O edita `start.bat` línea 74 y cambia `timeout /t 3` por `timeout /t 5`

### Live Server no funciona
**Causa:** Live Server solo sirve archivos estáticos, no ejecuta Python.

**Solución:** **SIEMPRE** usa `start.bat` o `python main.py` y abre `http://127.0.0.1:8000`

---

## Precisión del modelo

| Nivel de datos | Columnas | Precisión esperada |
|----------------|----------|-------------------|
| Mínimo | 4-6 | 65-75% |
| Estándar | 7-12 | 80-88% |
| Completo | 13+ | 88-95% |

**Nota:** La precisión depende de la coherencia clínica de los datos.

---

## Historial de Cambios

### v3.5 (Actual)
- ✅ **Análisis Detallado** educativo paso a paso en nueva pestaña
- ✅ **Diseño dark mode** negro mate con fuente Inter
- ✅ **Animaciones simplificadas** con spinner, progress bar y steps
- ✅ **Eliminación sin refrescar** - actualización automática de la lista
- ✅ **Sin descarga automática** - PDF solo se descarga manualmente
- ✅ **Navegador automático** - start.bat abre el navegador solo
- ✅ **Gráficas interactivas** con Chart.js en modal
- ✅ Sistema de almacenamiento en archivos JSON

### v3.0
- ✅ Modelo cambiado de Random Forest a **XGBoost**
- ✅ Integración con **IA (GPT-OSS-20B)**
- ✅ Reportes PDF profesionales
- ✅ Análisis de comorbilidades
- ✅ Soporte multi-formato

### v2.0
- Sistema con ML adaptativo
- Auto-detección de columnas

### v1.0
- Sistema básico con Random Forest

---

## Tecnologías utilizadas

- **Backend:** Python 3.8+, FastAPI, Uvicorn
- **Machine Learning:** XGBoost Classifier (300 estimadores)
- **Balanceo:** SMOTE (imbalanced-learn)
- **Procesamiento:** pandas, numpy, scipy, scikit-learn
- **Visualización:** matplotlib, seaborn, Chart.js
- **PDFs:** ReportLab
- **IA:** OpenRouter API + GPT-OSS-20B
- **Frontend:** HTML5, CSS3, JavaScript vanilla
- **Almacenamiento:** Archivos JSON (files.json, analysis.json)

---

## Licencia

Proyecto académico - Seminario de Plataformas Web

---

## Versión

**3.5** - Sistema profesional con XGBoost, análisis educativo detallado, gráficas interactivas y UX mejorada
