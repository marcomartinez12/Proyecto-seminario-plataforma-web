# Limpieza Automática del Puerto 8000

## ¿Qué hace el start.bat ahora?

El archivo `start.bat` ha sido modificado para **limpiar automáticamente el puerto 8000** antes de iniciar el servidor. Esto evita el error:

```
ERROR: [Errno 10048] error while attempting to bind on address ('0.0.0.0', 8000)
```

## Proceso de Limpieza

### Paso [0/4]: Verificación del Puerto

El script ahora ejecuta estos pasos **antes** de iniciar el servidor:

1. **Busca procesos en el puerto 8000**
   ```batch
   netstat -ano | findstr :8000 | findstr LISTENING
   ```

2. **Identifica el PID (Process ID)** del proceso que está ocupando el puerto

3. **Termina el proceso automáticamente**
   ```batch
   taskkill /F /PID [numero_proceso]
   ```

4. **Espera 2 segundos** para que Windows libere completamente el puerto

5. **Inicia el servidor** limpiamente

## Ejemplo de Salida

Cuando ejecutas `start.bat`, verás:

```
========================================
 Iniciando Plataforma de Analisis Medico
 VERSION MEJORADA CON ML OPTIMIZADO
========================================

[0/4] Verificando puerto 8000...
   - Puerto 8000 ocupado por proceso 17360
   - Liberando puerto...
   - Puerto liberado correctamente
   OK: Puerto 8000 disponible

[1/4] Activando entorno virtual...
[2/4] Instalando dependencias...
[3/4] Verificando nuevas dependencias de ML...
[4/4] Iniciando servidor...
```

## Ventajas

✅ **No más errores de puerto ocupado**
✅ **Reinicio automático limpio**
✅ **No necesitas matar procesos manualmente**
✅ **Inicio rápido y confiable**

## Uso

Simplemente ejecuta:

```batch
start.bat
```

O haz doble clic en el archivo `start.bat`

El script se encargará de:
1. Limpiar el puerto 8000
2. Activar el entorno virtual
3. Instalar/verificar dependencias
4. Iniciar el servidor
5. Abrir el navegador automáticamente

## Nota Técnica

El script solo termina procesos que están **escuchando (LISTENING)** en el puerto 8000. No afecta a:
- Conexiones TIME_WAIT (se limpian solas)
- Conexiones CLOSE_WAIT (se cierran automáticamente)
- Otros puertos o servicios

## Solución Manual (Si es necesario)

Si por alguna razón necesitas limpiar el puerto manualmente:

```batch
# Ver qué proceso usa el puerto 8000
netstat -ano | findstr :8000

# Matar el proceso (reemplaza [PID] con el número real)
taskkill /F /PID [PID]

# Esperar 5 segundos
timeout /t 5

# Iniciar el servidor
python main.py
```

## Preguntas Frecuentes

**P: ¿Puedo usar otro puerto?**
R: Sí, edita `main.py` línea 56:
```python
uvicorn.run(app, host="0.0.0.0", port=8001)  # Cambiar 8000 a 8001
```

**P: ¿El script afecta otros programas?**
R: No, solo termina procesos específicamente en el puerto 8000.

**P: ¿Qué pasa si hay múltiples procesos Python?**
R: El script termina TODOS los procesos que estén escuchando en el puerto 8000.

---

**Versión**: 3.6
**Última actualización**: 2025-10-29
