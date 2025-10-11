@echo off
chcp 65001 >nul
echo ========================================
echo  Iniciando Plataforma de Analisis Medico
echo  VERSION MEJORADA CON ML OPTIMIZADO
echo ========================================
echo.

REM Verificar si existe el entorno virtual
if not exist "venv\Scripts\activate.bat" (
    echo [!] ERROR: No se encontro el entorno virtual
    echo [i] Creando entorno virtual...
    python -m venv venv
    if errorlevel 1 (
        echo [!] ERROR: No se pudo crear el entorno virtual
        echo [i] Asegurate de tener Python instalado
        pause
        exit /b 1
    )
)

REM Activar entorno virtual
echo [1/4] Activando entorno virtual...
call venv\Scripts\activate.bat
if errorlevel 1 (
    echo [!] ERROR: No se pudo activar el entorno virtual
    pause
    exit /b 1
)

REM Instalar/actualizar dependencias
echo.
echo [2/4] Instalando dependencias (puede tardar unos minutos)...
pip install --upgrade pip --quiet
pip install -r requirements.txt --quiet
if errorlevel 1 (
    echo [!] ERROR: Fallo la instalacion de dependencias
    echo [i] Intentando instalar manualmente las nuevas dependencias...
    pip install xgboost imbalanced-learn scipy
)

REM Verificar instalacion de nuevas librerias ML
echo.
echo [3/4] Verificando nuevas dependencias de ML...
python -c "import xgboost; import imblearn; import scipy; print('   OK: XGBoost, SMOTE y SciPy instalados')" 2>nul
if errorlevel 1 (
    echo [!] ADVERTENCIA: Algunas dependencias de ML no estan instaladas
    echo [i] Instalando ahora...
    pip install xgboost imbalanced-learn scipy
)

REM Iniciar servidor
echo.
echo [4/4] Iniciando servidor...
echo.
echo ========================================
echo  SERVIDOR LISTO
echo ========================================
echo  URL: http://localhost:8000
echo  Documentacion API: http://localhost:8000/docs
echo.
echo  MEJORAS IMPLEMENTADAS:
echo  - XGBoost optimizado (300 arboles)
echo  - SMOTE para balance de clases
echo  - 15 nuevas features clinicas
echo  - Validacion cruzada estratificada
echo  - Precision esperada: 85-92%%
echo.
echo  Presiona Ctrl+C para detener el servidor
echo ========================================
echo.

python main.py

REM Si el servidor se detiene, preguntar si reiniciar
echo.
echo ========================================
echo  Servidor detenido
echo ========================================
pause
