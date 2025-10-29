#!/usr/bin/env python3
"""Script para limpiar pasos viejos y agregar CSS/JS para animaciones de procesamiento"""

# Leer el archivo
with open('app/routers/detailed_analysis.py', 'r', encoding='utf-8') as f:
    lines = f.readlines()

# Encontrar y eliminar las líneas entre "<!-- PASO 1: ENTRADA DE DATOS -->" y "</div>\n        </div>\n\n        <!-- PASO 1: CARGA DE DATOS -->"
start_delete = None
end_delete = None

for i, line in enumerate(lines):
    if '<!-- PASO 1: ENTRADA DE DATOS -->' in line and start_delete is None:
        start_delete = i
    if start_delete is not None and '</div>\n' == line and i > start_delete + 170:
        # Verificar que las siguientes líneas sean el cierre
        if i + 2 < len(lines) and '</div>' in lines[i+1] and '<!-- PASO 1: CARGA DE DATOS -->' in lines[i+2]:
            end_delete = i + 2
            break

# Eliminar las líneas si se encontraron
if start_delete and end_delete:
    del lines[start_delete:end_delete]
    print(f"Eliminadas líneas {start_delete} a {end_delete} (pasos antiguos)")

# Guardar temporalmente
with open('app/routers/detailed_analysis.py', 'w', encoding='utf-8') as f:
    f.writelines(lines)

print("Archivo limpiado correctamente!")
