@echo off
echo ========================================
echo  Iniciando Plataforma de Analisis Medico
echo ========================================
echo.

REM Activar entorno virtual
echo [1/3] Activando entorno virtual...
call venv\Scripts\activate.bat

REM Instalar/actualizar dependencias
echo.
echo [2/3] Instalando dependencias...
pip install -r requirements.txt --quiet

REM Iniciar servidor
echo.
echo [3/3] Iniciando servidor...
echo.
echo ========================================
echo  Servidor disponible en: http://localhost:8000
echo  Presiona Ctrl+C para detener
echo ========================================
echo.

python main.py
