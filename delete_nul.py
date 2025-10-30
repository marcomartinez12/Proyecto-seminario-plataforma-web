#!/usr/bin/env python3
"""Elimina el archivo 'nul' usando UNC path"""

import os

# Ruta UNC para acceder al archivo 'nul' (nombre reservado en Windows)
nul_path = r"\\?\C:\seminario\aplicativo web\nul"

try:
    if os.path.exists(nul_path):
        os.remove(nul_path)
        print(f"OK: Archivo eliminado: {nul_path}")
    else:
        print(f"INFO: El archivo no existe: {nul_path}")
except Exception as e:
    print(f"ERROR: No se pudo eliminar el archivo: {e}")
