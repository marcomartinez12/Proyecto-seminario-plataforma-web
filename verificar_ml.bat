@echo off
chcp 65001 >nul
echo ========================================
echo  VERIFICADOR DE DEPENDENCIAS ML
echo ========================================
echo.

REM Activar entorno virtual
if exist "venv\Scripts\activate.bat" (
    call venv\Scripts\activate.bat
) else (
    echo [!] ERROR: Entorno virtual no encontrado
    echo [i] Ejecuta start.bat primero
    pause
    exit /b 1
)

echo Verificando dependencias de Machine Learning...
echo.

REM Verificar Python
python --version
echo.

REM Verificar dependencias principales
echo [1/10] Verificando NumPy...
python -c "import numpy; print(f'   OK - NumPy {numpy.__version__}')" 2>nul || echo [!] FALTA - NumPy

echo [2/10] Verificando Pandas...
python -c "import pandas; print(f'   OK - Pandas {pandas.__version__}')" 2>nul || echo [!] FALTA - Pandas

echo [3/10] Verificando Scikit-learn...
python -c "import sklearn; print(f'   OK - Scikit-learn {sklearn.__version__}')" 2>nul || echo [!] FALTA - Scikit-learn

echo [4/10] Verificando XGBoost...
python -c "import xgboost; print(f'   OK - XGBoost {xgboost.__version__}')" 2>nul || echo [!] FALTA - XGBoost

echo [5/10] Verificando Imbalanced-learn...
python -c "import imblearn; print(f'   OK - Imbalanced-learn {imblearn.__version__}')" 2>nul || echo [!] FALTA - Imbalanced-learn

echo [6/10] Verificando SciPy...
python -c "import scipy; print(f'   OK - SciPy {scipy.__version__}')" 2>nul || echo [!] FALTA - SciPy

echo [7/10] Verificando Matplotlib...
python -c "import matplotlib; print(f'   OK - Matplotlib {matplotlib.__version__}')" 2>nul || echo [!] FALTA - Matplotlib

echo [8/10] Verificando Seaborn...
python -c "import seaborn; print(f'   OK - Seaborn {seaborn.__version__}')" 2>nul || echo [!] FALTA - Seaborn

echo [9/10] Verificando FastAPI...
python -c "import fastapi; print(f'   OK - FastAPI {fastapi.__version__}')" 2>nul || echo [!] FALTA - FastAPI

echo [10/10] Verificando ReportLab...
python -c "import reportlab; print(f'   OK - ReportLab {reportlab.Version}')" 2>nul || echo [!] FALTA - ReportLab

echo.
echo ========================================
echo  PRUEBA DE FUNCIONALIDAD ML
echo ========================================
echo.

echo Probando XGBoost con datos de prueba...
python -c "from xgboost import XGBClassifier; from sklearn.datasets import make_classification; X, y = make_classification(n_samples=100, n_features=10, random_state=42); model = XGBClassifier(n_estimators=10, random_state=42); model.fit(X, y); print(f'   OK - XGBoost funciona correctamente')" 2>nul || echo [!] ERROR en XGBoost

echo Probando SMOTE con datos de prueba...
python -c "from imblearn.over_sampling import SMOTE; from sklearn.datasets import make_classification; X, y = make_classification(n_samples=100, n_features=10, weights=[0.9, 0.1], random_state=42); smote = SMOTE(random_state=42); X_res, y_res = smote.fit_resample(X, y); print(f'   OK - SMOTE funciona correctamente')" 2>nul || echo [!] ERROR en SMOTE

echo.
echo ========================================
echo  VERIFICACION COMPLETADA
echo ========================================
echo.
echo Si todas las dependencias muestran OK,
echo el sistema esta listo para funcionar.
echo.
pause
