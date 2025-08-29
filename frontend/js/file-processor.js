// Procesamiento de archivos Excel/CSV

let currentFileData = null;
let columnMappings = {};

// Campos esperados del sistema
const SYSTEM_FIELDS = {
    'patientId': 'ID del Paciente',
    'age': 'Edad',
    'gender': 'Género',
    'weight': 'Peso (kg)',
    'height': 'Altura (cm)',
    'systolicBP': 'Presión Sistólica',
    'diastolicBP': 'Presión Diastólica',
    'glucoseLevel': 'Nivel de Glucosa',
    'hba1c': 'HbA1c (%)',
    'diabetesType': 'Tipo de Diabetes',
    'hypertensionDiagnosis': 'Diagnóstico Hipertensión',
    'diabetesMedication': 'Medicación Diabetes',
    'hypertensionMedication': 'Medicación Hipertensión',
    'smokingStatus': 'Estado Fumador',
    'physicalActivity': 'Actividad Física',
    'familyHistory': 'Antecedentes Familiares',
    'visitDate': 'Fecha de Consulta'
};

// Inicialización
document.addEventListener('DOMContentLoaded', function() {
    initializeFileProcessor();
});

function initializeFileProcessor() {
    setupFileUpload();
    setupEventListeners();
}

// Configurar área de carga de archivos
function setupFileUpload() {
    const fileUploadArea = document.getElementById('fileUploadArea');
    const fileInput = document.getElementById('fileInput');

    // Click para seleccionar archivo
    fileUploadArea.addEventListener('click', () => {
        fileInput.click();
    });

    // Drag and drop
    fileUploadArea.addEventListener('dragover', (e) => {
        e.preventDefault();
        fileUploadArea.classList.add('dragover');
    });

    fileUploadArea.addEventListener('dragleave', () => {
        fileUploadArea.classList.remove('dragover');
    });

    fileUploadArea.addEventListener('drop', (e) => {
        e.preventDefault();
        fileUploadArea.classList.remove('dragover');
        const files = e.dataTransfer.files;
        if (files.length > 0) {
            handleFileSelection(files[0]);
        }
    });

    // Selección de archivo
    fileInput.addEventListener('change', (e) => {
        if (e.target.files.length > 0) {
            handleFileSelection(e.target.files[0]);
        }
    });
}

// Configurar event listeners
function setupEventListeners() {
    document.getElementById('uploadBtn').addEventListener('click', processFile);
    document.getElementById('downloadTemplate').addEventListener('click', downloadTemplate);
    document.getElementById('clearData').addEventListener('click', clearAllData);
    document.getElementById('autoMapping').addEventListener('click', autoMapColumns);
    document.getElementById('confirmMapping').addEventListener('click', confirmMappingAndImport);
    document.getElementById('viewAnalytics').addEventListener('click', () => {
        window.location.href = 'analytics.html';
    });
}

// Manejar selección de archivo
function handleFileSelection(file) {
    // Validar tipo de archivo
    const validTypes = ['application/vnd.openxmlformats-officedocument.spreadsheetml.sheet', 'text/csv', 'application/vnd.ms-excel'];
    if (!validTypes.includes(file.type) && !file.name.match(/\.(xlsx|csv)$/i)) {
        Utils.showAlert('Tipo de archivo no válido. Solo se permiten archivos .xlsx y .csv', 'error');
        return;
    }

    // Validar tamaño (10MB máximo)
    if (file.size > 10 * 1024 * 1024) {
        Utils.showAlert('El archivo es demasiado grande. Tamaño máximo: 10MB', 'error');
        return;
    }

    // Mostrar información del archivo
    displayFileInfo(file);
    
    // Leer archivo
    readFile(file);
}

// Mostrar información del archivo
function displayFileInfo(file) {
    document.getElementById('fileName').textContent = file.name;
    document.getElementById('fileSize').textContent = formatFileSize(file.size);
    document.getElementById('fileType').textContent = file.type || 'Desconocido';
    document.getElementById('fileInfo').style.display = 'block';
}

// Leer archivo
function readFile(file) {
    const reader = new FileReader();
    
    reader.onload = function(e) {
        try {
            if (file.name.toLowerCase().endsWith('.csv')) {
                parseCSV(e.target.result);
            } else {
                parseExcel(e.target.result);
            }
        } catch (error) {
            Utils.showAlert('Error al leer el archivo: ' + error.message, 'error');
        }
    };

    if (file.name.toLowerCase().endsWith('.csv')) {
        reader.readAsText(file);
    } else {
        reader.readAsArrayBuffer(file);
    }
}

// Parsear archivo CSV
function parseCSV(csvText) {
    const delimiter = document.getElementById('delimiter').value;
    const hasHeaders = document.getElementById('hasHeaders').checked;
    
    const lines = csvText.split('\n').filter(line => line.trim());
    const data = [];
    
    lines.forEach(line => {
        const row = parseCSVLine(line, delimiter);
        if (row.length > 0) {
            data.push(row);
        }
    });
    
    processData(data, hasHeaders);
}

// Parsear línea CSV (maneja comillas)
function parseCSVLine(line, delimiter) {
    const result = [];
    let current = '';
    let inQuotes = false;
    
    for (let i = 0; i < line.length; i++) {
        const char = line[i];
        
        if (char === '"') {
            inQuotes = !inQuotes;
        } else if (char === delimiter && !inQuotes) {
            result.push(current.trim());
            current = '';
        } else {
            current += char;
        }
    }
    
    result.push(current.trim());
    return result;
}

// Parsear archivo Excel
function parseExcel(arrayBuffer) {
    const workbook = XLSX.read(arrayBuffer, { type: 'array' });
    const firstSheetName = workbook.SheetNames[0];
    const worksheet = workbook.Sheets[firstSheetName];
    
    const data = XLSX.utils.sheet_to_json(worksheet, { header: 1 });
    const hasHeaders = document.getElementById('hasHeaders').checked;
    
    processData(data, hasHeaders);
}

// Procesar datos leídos
function processData(data, hasHeaders) {
    if (data.length === 0) {
        Utils.showAlert('El archivo está vacío', 'error');
        return;
    }
    
    currentFileData = {
        raw: data,
        headers: hasHeaders ? data[0] : null,
        rows: hasHeaders ? data.slice(1) : data
    };
    
    // Actualizar contador de registros
    document.getElementById('recordCount').textContent = currentFileData.rows.length;
    
    // Mostrar vista previa
    displayDataPreview();
    
    // Mostrar mapeo de columnas
    displayColumnMapping();
    
    // Habilitar botón de procesamiento
    document.getElementById('uploadBtn').disabled = false;
}

// Mostrar vista previa de datos
function displayDataPreview() {
    const previewTable = document.getElementById('previewTable');
    const thead = document.getElementById('previewTableHead');
    const tbody = document.getElementById('previewTableBody');
    
    // Limpiar tabla
    thead.innerHTML = '';
    tbody.innerHTML = '';
    
    // Crear encabezados
    const headerRow = document.createElement('tr');
    const maxCols = Math.max(...currentFileData.raw.map(row => row.length));
    
    for (let i = 0; i < maxCols; i++) {
        const th = document.createElement('th');
        th.textContent = currentFileData.headers ? currentFileData.headers[i] : `Columna ${i + 1}`;
        headerRow.appendChild(th);
    }
    thead.appendChild(headerRow);
    
    // Mostrar primeras 5 filas
    const previewRows = currentFileData.rows.slice(0, 5);
    previewRows.forEach(row => {
        const tr = document.createElement('tr');
        for (let i = 0; i < maxCols; i++) {
            const td = document.createElement('td');
            td.textContent = row[i] || '';
            tr.appendChild(td);
        }
        tbody.appendChild(tr);
    });
    
    document.getElementById('dataPreview').style.display = 'block';
}

// Mostrar mapeo de columnas
function displayColumnMapping() {
    const mappingGrid = document.getElementById('mappingGrid');
    mappingGrid.innerHTML = '';
    
    if (!currentFileData.headers) {
        Utils.showAlert('Para mapear columnas, el archivo debe tener encabezados', 'warning');
        return;
    }
    
    // Crear mapeos para cada campo del sistema
    Object.keys(SYSTEM_FIELDS).forEach(fieldKey => {
        const mappingRow = document.createElement('div');
        mappingRow.className = 'mapping-row';
        
        mappingRow.innerHTML = `
            <div class="mapping-field">
                <label>${SYSTEM_FIELDS[fieldKey]}</label>
                ${fieldKey === 'patientId' || fieldKey === 'age' || fieldKey === 'gender' || 
                  fieldKey === 'systolicBP' || fieldKey === 'diastolicBP' || fieldKey === 'glucoseLevel' || 
                  fieldKey === 'visitDate' ? '<span class="required">*</span>' : ''}
            </div>
            <div class="mapping-select">
                <select id="mapping_${fieldKey}" class="form-select">
                    <option value="">-- No mapear --</option>
                    ${currentFileData.headers.map((header, index) => 
                        `<option value="${index}">${header}</option>`
                    ).join('')}
                </select>
            </div>
        `;
        
        mappingGrid.appendChild(mappingRow);
    });
    
    document.getElementById('columnMapping').style.display = 'block';
}

// Mapeo automático de columnas
function autoMapColumns() {
    if (!currentFileData.headers) return;
    
    const autoMappings = {
        'patientId': ['id_paciente', 'id', 'patient_id', 'paciente'],
        'age': ['edad', 'age', 'años'],
        'gender': ['genero', 'gender', 'sexo', 'sex'],
        'weight': ['peso', 'weight', 'kg'],
        'height': ['altura', 'height', 'cm', 'estatura'],
        'systolicBP': ['sistolica', 'systolic', 'presion_sistolica', 'pas'],
        'diastolicBP': ['diastolica', 'diastolic', 'presion_diastolica', 'pad'],
        'glucoseLevel': ['glucosa', 'glucose', 'glicemia'],
        'hba1c': ['hba1c', 'hemoglobina'],
        'visitDate': ['fecha', 'date', 'fecha_consulta', 'visit_date']
    };
    
    Object.keys(autoMappings).forEach(fieldKey => {
        const select = document.getElementById(`mapping_${fieldKey}`);
        const keywords = autoMappings[fieldKey];
        
        for (let i = 0; i < currentFileData.headers.length; i++) {
            const header = currentFileData.headers[i].toLowerCase();
            if (keywords.some(keyword => header.includes(keyword))) {
                select.value = i;
                break;
            }
        }
    });
    
    Utils.showAlert('Mapeo automático aplicado. Revise y ajuste según sea necesario.', 'info');
}

// Confirmar mapeo e importar
function confirmMappingAndImport() {
    // Recopilar mapeos
    columnMappings = {};
    Object.keys(SYSTEM_FIELDS).forEach(fieldKey => {
        const select = document.getElementById(`mapping_${fieldKey}`);
        if (select.value !== '') {
            columnMappings[fieldKey] = parseInt(select.value);
        }
    });
    
    // Validar mapeos obligatorios
    const requiredFields = ['patientId', 'age', 'gender', 'systolicBP', 'diastolicBP', 'glucoseLevel', 'visitDate'];
    const missingFields = requiredFields.filter(field => !columnMappings[field]);
    
    if (missingFields.length > 0) {
        Utils.showAlert(`Faltan mapeos obligatorios: ${missingFields.map(f => SYSTEM_FIELDS[f]).join(', ')}`, 'error');
        return;
    }
    
    // Procesar e importar datos
    importData();
}

// Importar datos
function importData() {
    const results = {
        success: 0,
        errors: 0,
        duplicates: 0,
        errorDetails: []
    };
    
    const existingData = getPatientData() || [];
    const existingIds = new Set(existingData.map(p => p.patientId));
    
    currentFileData.rows.forEach((row, index) => {
        try {
            const patientData = {};
            
            // Mapear datos según configuración
            Object.keys(columnMappings).forEach(fieldKey => {
                const columnIndex = columnMappings[fieldKey];
                let value = row[columnIndex];
                
                // Procesar y validar valor
                value = processFieldValue(fieldKey, value);
                if (value !== null) {
                    patientData[fieldKey] = value;
                }
            });
            
            // Validar datos obligatorios
            if (!patientData.patientId) {
                throw new Error('ID de paciente requerido');
            }
            
            // Verificar duplicados
            if (existingIds.has(patientData.patientId)) {
                results.duplicates++;
                return;
            }
            
            // Agregar metadatos
            patientData.id = Utils.generateId();
            patientData.timestamp = new Date().toISOString();
            
            // Guardar datos
            existingData.push(patientData);
            existingIds.add(patientData.patientId);
            results.success++;
            
        } catch (error) {
            results.errors++;
            results.errorDetails.push({
                row: index + 1,
                error: error.message,
                data: row
            });
        }
    });
    
    // Guardar datos actualizados
    localStorage.setItem('patientsData', JSON.stringify(existingData));
    
    // Mostrar resultados
    displayImportResults(results);
}

// Procesar valor de campo
function processFieldValue(fieldKey, value) {
    if (value === undefined || value === null || value === '') {
        return null;
    }
    
    value = String(value).trim();
    
    switch (fieldKey) {
        case 'age':
        case 'weight':
        case 'height':
        case 'systolicBP':
        case 'diastolicBP':
        case 'glucoseLevel':
        case 'physicalActivity':
            const num = parseFloat(value);
            return isNaN(num) ? null : num;
            
        case 'hba1c':
            const hba1c = parseFloat(value);
            return isNaN(hba1c) ? null : hba1c;
            
        case 'visitDate':
            return parseDate(value);
            
        case 'gender':
            return normalizeGender(value);
            
        case 'diabetesMedication':
        case 'hypertensionMedication':
            return normalizeBooleanField(value);
            
        default:
            return value;
    }
}

// Normalizar género
function normalizeGender(value) {
    const val = value.toLowerCase();
    if (val.includes('m') || val.includes('masculino') || val.includes('male')) return 'masculino';
    if (val.includes('f') || val.includes('femenino') || val.includes('female')) return 'femenino';
    return 'otro';
}

// Normalizar campos booleanos
function normalizeBooleanField(value) {
    const val = value.toLowerCase();
    if (val === 'si' || val === 'sí' || val === 'yes' || val === '1' || val === 'true') return 'si';
    return 'no';
}

// Parsear fecha
function parseDate(dateStr) {
    // Intentar varios formatos
    const formats = [
        /^(\d{1,2})\/(\d{1,2})\/(\d{4})$/, // DD/MM/YYYY
        /^(\d{4})-(\d{1,2})-(\d{1,2})$/, // YYYY-MM-DD
        /^(\d{1,2})-(\d{1,2})-(\d{4})$/ // DD-MM-YYYY
    ];
    
    for (let format of formats) {
        const match = dateStr.match(format);
        if (match) {
            let day, month, year;
            if (format === formats[1]) { // YYYY-MM-DD
                [, year, month, day] = match;
            } else { // DD/MM/YYYY or DD-MM-YYYY
                [, day, month, year] = match;
            }
            
            const date = new Date(year, month - 1, day);
            if (!isNaN(date.getTime())) {
                return date.toISOString().split('T')[0];
            }
        }
    }
    
    return null;
}

// Mostrar resultados de importación
function displayImportResults(results) {
    document.getElementById('successCount').textContent = results.success;
    document.getElementById('errorCount').textContent = results.errors;
    document.getElementById('duplicateCount').textContent = results.duplicates;
    
    if (results.errors > 0) {
        const errorList = document.getElementById('errorList');
        errorList.innerHTML = results.errorDetails.map(error => 
            `<div class="error-item">Fila ${error.row}: ${error.error}</div>`
        ).join('');
        document.getElementById('errorDetails').style.display = 'block';
        document.getElementById('downloadErrors').style.display = 'inline-block';
    }
    
    document.getElementById('importResults').style.display = 'block';
    
    if (results.success > 0) {
        Utils.showAlert(`Importación completada: ${results.success} registros importados exitosamente`, 'success');
    }
}

// Descargar plantilla
function downloadTemplate() {
    const headers = Object.values(SYSTEM_FIELDS);
    const csvContent = headers.join(',') + '\n';
    
    const blob = new Blob([csvContent], { type: 'text/csv' });
    const url = window.URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = 'plantilla_pacientes.csv';
    a.click();
    window.URL.revokeObjectURL(url);
}

// Formatear tamaño de archivo
function formatFileSize(bytes) {
    if (bytes === 0) return '0 Bytes';
    const k = 1024;
    const sizes = ['Bytes', 'KB', 'MB', 'GB'];
    const i = Math.floor(Math.log(bytes) / Math.log(k));
    return parseFloat((bytes / Math.pow(k, i)).toFixed(2)) + ' ' + sizes[i];
}

// Procesar archivo (función principal)
function processFile() {
    if (!currentFileData) {
        Utils.showAlert('Por favor seleccione un archivo primero', 'error');
        return;
    }
    
    // Si no hay mapeo configurado, mostrar sección de mapeo
    if (Object.keys(columnMappings).length === 0) {
        document.getElementById('columnMapping').scrollIntoView({ behavior: 'smooth' });
        Utils.showAlert('Configure el mapeo de columnas antes de procesar', 'info');
        return;
    }
    
    confirmMappingAndImport();
}