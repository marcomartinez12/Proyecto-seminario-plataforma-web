# Explicación Técnica del Proyecto

## ¿Qué hace el sistema?

Analiza datos médicos de pacientes usando Machine Learning para detectar patrones de hipertensión y diabetes, y genera reportes automáticos con explicaciones en lenguaje simple.

---

## Machine Learning: Random Forest

### ¿Por qué Random Forest?

**Razones de elección:**

1. **Robusto con datos médicos**: No requiere que los datos estén perfectamente balanceados
2. **Maneja variables mixtas**: Combina edad, peso, sexo, presión arterial sin problemas
3. **Menos overfitting**: No se "memoriza" los datos como otros algoritmos
4. **Interpretable**: Muestra qué variables son más importantes (glucosa, edad, etc.)
5. **Buen rendimiento**: Precisión aceptable sin necesidad de ajustes complejos

**Alternativas descartadas:**
- **Regresión Logística**: Muy simple, asume relaciones lineales
- **SVM**: Lento con datasets grandes
- **Redes Neuronales**: Requiere muchos más datos y poder de cómputo

### ¿Por qué 50 árboles?

```python
RandomForestClassifier(n_estimators=50, n_jobs=-1)
```

**Justificación:**
- **100 árboles**: Más precisión (+2-3%) pero el doble de tiempo
- **50 árboles**: Balance perfecto velocidad/precisión
- **25 árboles**: Muy rápido pero menos confiable

**Tiempo de procesamiento:**
- 50 árboles: ~8-10 segundos
- 100 árboles: ~18-20 segundos

**Conclusión**: 50 árboles da buena precisión manteniendo la velocidad

### Validación Cruzada

**¿Por qué se agregó?**

Sin validación cruzada:
- Dataset 1010 registros: 31% precisión
- Dataset 5000 registros: 52% precisión
- **Problema**: Mucha variación, resultados poco confiables

Con validación cruzada:
- Divide datos en 5 grupos
- Entrena 5 veces con combinaciones diferentes
- Promedia resultados → Precisión más estable

**Resultado**: Métricas más confiables, especialmente con datasets pequeños

---

## Frameworks y Tecnologías

### Backend: FastAPI

**¿Por qué FastAPI?**

1. **Velocidad**: Basado en Starlette y Pydantic, muy rápido
2. **Async nativo**: Maneja múltiples solicitudes simultáneas
3. **Validación automática**: Pydantic valida datos sin código extra
4. **Documentación automática**: Genera Swagger UI
5. **Moderno**: Python 3.8+, type hints, async/await

**Alternativas:**
- **Flask**: Más simple pero más lento y sin async nativo
- **Django**: Muy pesado para este proyecto
- **Express (Node.js)**: Cambiar de lenguaje complicaría el ML

### ML: scikit-learn

**¿Por qué scikit-learn?**

1. **Estándar de la industria**: Librería más usada para ML clásico
2. **Bien documentada**: Fácil de aprender y mantener
3. **Random Forest optimizado**: Implementación en C, muy rápida
4. **Validación cruzada integrada**: `cross_val_score` listo para usar
5. **Preprocesamiento incluido**: `LabelEncoder`, `train_test_split`

**Alternativas:**
- **TensorFlow/PyTorch**: Overkill para clasificación simple
- **XGBoost**: Más complejo de configurar
- **Statsmodels**: Enfocado en estadística, no ML

### Visualización: Matplotlib + Seaborn

**¿Por qué estas librerías?**

**Matplotlib:**
- Control total sobre gráficos
- Exporta a PNG para PDFs
- Backend 'Agg' sin interfaz gráfica (servidor)

**Seaborn:**
- Gráficos profesionales con pocas líneas
- Paletas de colores científicas
- Basado en matplotlib, se integra perfecto

**Alternativas:**
- **Plotly**: Interactivo pero pesado para PDF estático
- **Bokeh**: Similar a Plotly
- **Altair**: Menos control sobre diseño

### Reportes: ReportLab

**¿Por qué ReportLab?**

1. **PDF profesional**: Control total sobre diseño (Times New Roman, colores APA)
2. **Integra imágenes**: Inserta gráficos matplotlib directamente
3. **Tablas avanzadas**: TableStyle con colores y bordes
4. **Python puro**: No requiere LaTeX ni dependencias externas

**Alternativas:**
- **FPDF**: Muy básico, diseño limitado
- **WeasyPrint**: Requiere HTML/CSS, menos control
- **LaTeX**: Complejo de configurar y mantener

### IA: OpenRouter API

**¿Por qué OpenRouter?**

1. **Múltiples modelos**: Acceso a Mistral, Claude, GPT, etc.
2. **Capa gratuita**: Mistral Small gratis
3. **API unificada**: Un solo endpoint para todos los modelos
4. **Sin límites estrictos**: Mejor que APIs gratuitas individuales

**Modelo elegido: Mistral Small 24B**
- Gratuito
- Bueno en español
- Respuestas claras y concisas
- Rápido (~2-4 segundos)

**Alternativas:**
- **OpenAI GPT**: De pago, más caro
- **Modelo local (Llama)**: Requiere GPU potente
- **Google Gemini**: Límites de API más restrictivos

---

## Optimizaciones de Rendimiento

### 1. Machine Learning
```python
n_estimators=50          # Balance velocidad/precisión
n_jobs=-1               # Usa todos los CPU cores
random_state=42         # Reproducibilidad
```

### 2. Gráficos
```python
dpi=150                 # No 300 (calidad suficiente, 50% más rápido)
max_points=3000         # Muestrea datasets grandes
backend='Agg'           # Sin GUI, más rápido
```

### 3. IA
```python
max_tokens=1500         # No 2000 (respuesta concisa, 25% más rápido)
temperature=0.7         # Balance creatividad/consistencia
timeout=60              # Previene esperas infinitas
```

**Resultado final**: ~8-10 segundos total (antes era ~20 segundos)

---

## Flujo de Datos Simplificado

```
1. Usuario sube Excel
   ↓
2. Validación de columnas requeridas
   ↓
3. Preprocesamiento (calcular IMC, separar presión arterial)
   ↓
4. Random Forest (entrenamiento + validación cruzada)
   ↓
5. Generación de gráficos (matplotlib/seaborn)
   ↓
6. Creación de PDF (ReportLab)
   ↓
7. Explicación IA (OpenRouter)
   ↓
8. Descarga automática
```

---

## Variables del Modelo

**Entrada (10 features):**
1. Edad
2. Sexo (codificado: 0/1)
3. Peso (kg)
4. Altura (cm)
5. IMC (calculado automáticamente)
6. Presión Sistólica (mmHg)
7. Presión Diastólica (mmHg)
8. Glucosa (mg/dL)
9. Colesterol (mg/dL)
10. Fumador (codificado: 0/1)

**Salida:**
- Diagnóstico predicho (Hipertensión, Diabetes, Normal, etc.)

**Importancia típica:**
- Glucosa: 25%
- Presión Sistólica: 22%
- Edad: 18%
- IMC: 15%
- Resto: 20%

---

## Decisiones Técnicas Clave

### ¿Por qué JSON en lugar de base de datos SQL?

**Ventajas:**
- Despliegue simple (no requiere servidor DB)
- Suficiente para volumen esperado
- Fácil de respaldar (un solo archivo)

**Desventaja aceptada:**
- No escala a millones de registros (no es el caso de uso)

### ¿Por qué validación cruzada solo con ≥100 registros?

Con menos de 100:
- Cada fold tendría muy pocos datos
- No mejora la confiabilidad
- Desperdicia tiempo de cómputo

### ¿Por qué async con httpx para la API de IA?

```python
async with httpx.AsyncClient() as client:
    response = await client.post(...)
```

- FastAPI es async
- No bloquea el servidor mientras espera respuesta de IA
- Puede atender otras peticiones simultáneamente

---

## Preguntas Frecuentes Técnicas

**¿Por qué no Deep Learning?**
- Random Forest da 50-60% precisión
- Deep Learning necesita 10x más datos para superar esto
- Mayor complejidad sin beneficio claro

**¿50 árboles es estándar?**
- Sklearn usa 100 por defecto
- Reducimos a 50 por velocidad
- Precisión baja solo 2-3%

**¿Se puede cambiar a otro modelo de IA?**
Sí, línea 114 de `ai_analysis.py`:
```python
"model": "anthropic/claude-3-haiku"  # Más rápido
"model": "openai/gpt-4"              # Más preciso (de pago)
```

**¿Por qué Times New Roman en PDF?**
- Normas APA para reportes médicos
- Fuente profesional y legible
- Disponible en Windows por defecto
