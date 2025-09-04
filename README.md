# 🏥 Plataforma de Análisis Médico con Machine Learning

## 📋 Descripción

Plataforma web avanzada para el análisis automatizado de datos médicos utilizando técnicas de Machine Learning. Permite cargar archivos Excel/CSV con información de pacientes y genera reportes PDF profesionales con análisis predictivo para la detección de enfermedades crónicas como diabetes e hipertensión.

## ✨ Características Principales

### 🤖 Machine Learning
- **Algoritmo**: Random Forest Classifier con 100 estimadores
- **Análisis predictivo** para detección de diabetes e hipertensión
- **Procesamiento automático** de datos médicos
- **Cálculo de importancia** de características clínicas
- **Precisión del modelo** reportada en tiempo real

### 📊 Análisis de Datos
- Procesamiento de **10 variables médicas clave**:
  - Edad, Sexo, Peso, Altura, IMC
  - Presión Arterial (Sistólica/Diastólica)
  - Glucosa, Colesterol, Hábito de fumar
- **Detección automática** de factores de riesgo
- **Análisis demográfico** completo
- **Correlaciones** entre variables

### 📈 Visualización
- **Gráficos interactivos** con matplotlib y seaborn
- Distribución de diagnósticos
- Correlación entre variables
- Análisis por edad y diagnóstico

### 📄 Reportes PDF
- **Formato profesional** con Times New Roman 12pt
- **Normas APA** para documentos médicos
- **Gráficos integrados** de alta calidad
- **Recomendaciones médicas** automatizadas
- **Descarga automática** al completar análisis

### 🎨 Interfaz de Usuario
- **Diseño moderno** con tema oscuro
- **Drag & drop** para carga de archivos
- **Barra de progreso** en tiempo real
- **Notificaciones** de estado
- **Responsive design**

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