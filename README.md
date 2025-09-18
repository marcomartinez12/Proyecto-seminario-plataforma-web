# 🏥 Plataforma de Análisis Médico con Machine Learning

## 📋 Descripción

Plataforma web para detectar tendencias de enfermedades crónicas no transmisibles como la hipertensión y diabetes

## ✨ Características Principales

### 🤖 Machine Learning
- **Algoritmo**: Random Forest Classifier con 100 estimadores
- **Análisis predictivo** para detección de diabetes e hipertensión
- **Procesamiento automático** de datos médicos
- **Cálculo de importancia** de características clínicas
- **Precisión del modelo** reportada en tiempo real

### 📊 Análisis de Datos Avanzado
- Procesamiento de **10 variables médicas clave**:
  - Edad, Sexo, Peso, Altura, IMC
  - Presión Arterial (Sistólica/Diastólica)
  - Glucosa, Colesterol, Hábito de fumar
- **Detección automática** de factores de riesgo cardiovascular
- **Análisis demográfico** completo por grupos etarios
- **Correlaciones** entre variables clínicas
- **🆕 Análisis de comorbilidad** hipertensión + diabetes basado en valores clínicos reales
- **🆕 Tendencias epidemiológicas** por grupos de edad (< 30, 30-45, 46-60, > 60 años)
- **🆕 Detección de patrones** de riesgo cardiovascular específicos

### 🩺 Análisis Clínico Especializado
- **Análisis de Hipertensión Arterial**:
  - Prevalencia por grupos etarios
  - Clasificación de riesgo (Bajo, Moderado, Alto, Muy Alto)
  - Recomendaciones preventivas específicas
- **Análisis de Diabetes Mellitus**:
  - Tendencias por edad y factores de riesgo
  - Estrategias de manejo terapéutico
  - Detección de complicaciones potenciales
- **🆕 Comorbilidad Inteligente**:
  - Detección basada en criterios clínicos (Presión Sistólica >140 mmHg + Glucosa >126 mg/dL)
  - Análisis de riesgo cardiovascular combinado
  - Estrategias de manejo integral

### 📈 Visualización
- **Gráficos interactivos** con matplotlib y seaborn
- Distribución de diagnósticos
- Correlación entre variables
- Análisis por edad y diagnóstico
- **🆕 Factores de riesgo** con datos precisos de comorbilidad

### 📄 Reportes PDF Profesionales
- **Formato profesional** con Times New Roman 12pt siguiendo normas APA
- **🆕 Secciones especializadas**:
  - Detección de patrones de riesgo cardiovascular
  - Tendencias epidemiológicas por grupos etarios
  - Análisis específico de hipertensión arterial
  - Análisis detallado de diabetes mellitus
  - **Análisis de comorbilidad hipertensión + diabetes**
- **Gráficos integrados** de alta calidad
- **Recomendaciones médicas** automatizadas y específicas
- **Tablas de riesgo** con codificación por colores
- **Descarga automática** al completar análisis

### 🎨 Interfaz de Usuario
- **Diseño moderno** con tema oscuro
- **Drag & drop** para carga de archivos
- **Barra de progreso** en tiempo real
- **Notificaciones** de estado
- **Responsive design**

## 🔬 Metodología Clínica

### Criterios de Diagnóstico
- **Hipertensión**: Presión Sistólica > 140 mmHg
- **Diabetes**: Glucosa > 126 mg/dL
- **Obesidad**: IMC > 30 kg/m²
- **Dislipidemia**: Colesterol > 240 mg/dL

### Análisis de Comorbilidad
El sistema utiliza valores clínicos reales para detectar comorbilidad, identificando pacientes que presentan simultáneamente:
- Presión arterial elevada (>140 mmHg sistólica)
- Glucosa elevada (>126 mg/dL)

Esta metodología proporciona una evaluación más precisa del riesgo cardiovascular combinado.

## 🛠️ Tecnologías Utilizadas

### Backend
- **FastAPI** - Framework web moderno y rápido
- **Python 3.8+** - Lenguaje principal
- **scikit-learn** - Machine Learning
- **pandas** - Manipulación de datos
- **numpy** - Computación numérica
- **matplotlib/seaborn** - Visualización
- **ReportLab** - Generación de PDFs

### Frontend
- **HTML5/CSS3** - Estructura y estilos
- **JavaScript ES6+** - Interactividad
- **Fetch API** - Comunicación con backend
- **CSS Grid/Flexbox** - Layout responsivo

### Dependencias
```txt
fastapi
uvicorn
python-multipart
pandas
numpy
scikit-learn
matplotlib
seaborn
reportlab
openpyxl
aiofiles
jinja2
requests
psutil
```

## 🚀 Instalación y Configuración

### Prerrequisitos
- Python 3.8 o superior
- Node.js y npm (para desarrollo frontend)
- Git

### 1. Clonar el Repositorio
```bash
git clone <url-del-repositorio>
cd Proyecto-seminario-plataforma-web
```

### 2. Crear Entorno Virtual
```bash
python -m venv venv

# Windows
venv\Scripts\activate

# Linux/Mac
source venv/bin/activate
```

### 3. Instalar Dependencias
```bash
pip install -r requirements.txt
```

### 4. Configurar Directorios
```bash
mkdir uploads downloads data reports\charts
```

### 5. Ejecutar la Aplicación
```bash
python main.py
```

La aplicación estará disponible en: `http://localhost:8000`

## 📁 Estructura del Proyecto