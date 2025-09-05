// Configuración de la API
const API_BASE_URL = 'http://localhost:8000/api';

// Variables globales
let uploadedFiles = [];
let currentAnalysis = null;
let isAnalyzing = false; // Nueva variable para controlar el estado

// Cargar archivos subidos con mejor control de debounce
let loadFilesTimeout;
let lastLoadTime = 0;

async function loadUploadedFiles(force = false) {
    const now = Date.now();
    
    // Si estamos analizando y no es forzado, no recargar
    if (isAnalyzing && !force) {
        return;
    }
    
    // Evitar llamadas muy frecuentes (mínimo 2 segundos entre llamadas)
    if (!force && (now - lastLoadTime) < 2000) { // Aumentado de 1000 a 2000
        return;
    }
    
    // Debounce para evitar llamadas múltiples
    if (loadFilesTimeout) {
        clearTimeout(loadFilesTimeout);
    }
    
    loadFilesTimeout = setTimeout(async () => {
        try {
            lastLoadTime = Date.now();
            const response = await fetch(`${API_BASE_URL}/files/list`);
            if (!response.ok) throw new Error('Error al cargar archivos');
            
            uploadedFiles = await response.json();
            renderFilesList();
            
        } catch (error) {
            console.error('Error loading files:', error);
        }
    }, force ? 0 : 800); // Aumentado el delay
}

// Analizar archivo - versión optimizada
// Función analyzeFile optimizada (líneas 50-100)
async function analyzeFile(fileId) {
    try {
        if (isAnalyzing) {
            showToast('Ya hay un análisis en progreso', 'warning');
            return;
        }
        
        isAnalyzing = true;
        showAnalysisProgress();
        
        const response = await fetch(`${API_BASE_URL}/analysis/analyze`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({ file_id: fileId })
        });
        
        if (!response.ok) {
            const errorData = await response.json();
            throw new Error(errorData.detail || 'Error en el análisis');
        }
        
        const result = await response.json();
        currentAnalysis = result;
        
        // Simular progreso de análisis con animaciones mejoradas
        await simulateAnalysisSteps();
        
        // Esperar para que el usuario vea la animación completa
        await new Promise(resolve => setTimeout(resolve, 2500));
        
        hideAnalysisProgress();
        showToast('Análisis completado. Descargando reporte...', 'success');
        
        // Descargar automáticamente el reporte
        await downloadReport(result.id, true);
        
        // OPTIMIZADO: Solo actualizar el estado del archivo específico
        updateFileStatus(fileId, 'completed');
        
    } catch (error) {
        hideAnalysisProgress();
        showToast(`Error en el análisis: ${error.message}`, 'error');
        console.error('Error analyzing file:', error);
    } finally {
        isAnalyzing = false;
        // ELIMINADO: No más refrescos automáticos
    }
}

// Función deleteFile optimizada (líneas 105-125)
async function deleteFile(fileId) {
    if (!confirm('¿Estás seguro de que quieres eliminar este archivo?')) {
        return;
    }
    
    try {
        const response = await fetch(`${API_BASE_URL}/files/${fileId}`, {
            method: 'DELETE'
        });
        
        if (!response.ok) throw new Error('Error al eliminar archivo');
        
        showToast('Archivo eliminado correctamente', 'success');
        
        // OPTIMIZADO: Actualización local sin refresco
        uploadedFiles = uploadedFiles.filter(file => file.id !== fileId);
        renderFilesList();
        
    } catch (error) {
        showToast(`Error: ${error.message}`, 'error');
        console.error('Error deleting file:', error);
    }
}

// Función uploadFile optimizada (líneas 130-165)
async function uploadFile(file) {
    try {
        showUploadProgress();
        
        const formData = new FormData();
        formData.append('file', file);
        
        const response = await fetch(`${API_BASE_URL}/files/upload`, {
            method: 'POST',
            body: formData
        });
        
        if (!response.ok) {
            const errorData = await response.json();
            throw new Error(errorData.detail || 'Error al subir archivo');
        }
        
        const result = await response.json();
        
        updateProgress(100, 'Archivo subido correctamente');
        setTimeout(() => {
            hideUploadProgress();
            showToast('Archivo subido correctamente', 'success');
            
            // OPTIMIZADO: Agregar solo el nuevo archivo
            uploadedFiles.push(result);
            renderFilesList();
        }, 1000);
        
    } catch (error) {
        hideUploadProgress();
        showToast(`Error: ${error.message}`, 'error');
        console.error('Error uploading file:', error);
    }
}

// Nueva función para actualizar estado específico
function updateFileStatus(fileId, newStatus) {
    const fileIndex = uploadedFiles.findIndex(file => file.id === fileId);
    if (fileIndex !== -1) {
        uploadedFiles[fileIndex].status = newStatus;
        renderFilesList();
    }
}

// Función simulateAnalysisSteps mejorada (líneas 510-540)
async function simulateAnalysisSteps() {
    const steps = ['step1', 'step2', 'step3', 'step4'];
    const stepTimes = [800, 2500, 3000, 2000];
    
    for (let i = 0; i < steps.length; i++) {
        // Esperar antes de activar el siguiente paso
        if (i > 0) {
            await new Promise(resolve => setTimeout(resolve, stepTimes[i]));
        }
        
        // Activar paso actual con animación
        const currentStep = document.getElementById(steps[i]);
        currentStep.classList.add('active');
        
        // Cambiar icono a completado después de la animación
        setTimeout(() => {
            const icon = currentStep.querySelector('i');
            icon.className = 'fas fa-check-circle';
        }, 600);
        
        // Mantener pasos anteriores completados
        if (i > 0) {
            const prevStep = document.getElementById(steps[i-1]);
            const prevIcon = prevStep.querySelector('i');
            prevIcon.className = 'fas fa-check-circle';
        }
    }
    
    // Esperar para mostrar todos los pasos completados
    await new Promise(resolve => setTimeout(resolve, 1500));
}

// ELIMINAR completamente la función duplicada de deleteFile (líneas 590-600)
async function deleteFile(fileId) {
    if (!confirm('¿Estás seguro de que quieres eliminar este archivo?')) {
        return;
    }
    
    try {
        const response = await fetch(`${API_BASE_URL}/files/${fileId}`, {
            method: 'DELETE'
        });
        
        if (!response.ok) throw new Error('Error al eliminar archivo');
        
        showToast('Archivo eliminado correctamente', 'success');
        
        // OPTIMIZADO: Actualización local sin refresco
        uploadedFiles = uploadedFiles.filter(file => file.id !== fileId);
        renderFilesList();
        
    } catch (error) {
        showToast(`Error: ${error.message}`, 'error');
        console.error('Error deleting file:', error);
    }
}

// Subir archivo - versión optimizada
async function uploadFile(file) {
    try {
        showUploadProgress();
        
        const formData = new FormData();
        formData.append('file', file);
        
        const response = await fetch(`${API_BASE_URL}/files/upload`, {
            method: 'POST',
            body: formData
        });
        
        if (!response.ok) {
            const errorData = await response.json();
            throw new Error(errorData.detail || 'Error al subir archivo');
        }
        
        const result = await response.json();
        
        updateProgress(100, 'Archivo subido correctamente');
        setTimeout(() => {
            hideUploadProgress();
            showToast('Archivo subido correctamente', 'success');
            
            // Recargar lista solo una vez
            loadUploadedFiles(true);
        }, 1000);
        
    } catch (error) {
        hideUploadProgress();
        showToast(`Error: ${error.message}`, 'error');
        console.error('Error uploading file:', error);
    }
}

// Elementos del DOM
const uploadArea = document.getElementById('uploadArea');
const fileInput = document.getElementById('fileInput');
const uploadProgress = document.getElementById('uploadProgress');
const progressFill = document.getElementById('progressFill');
const progressText = document.getElementById('progressText');
const filesList = document.getElementById('filesList');
const noFiles = document.getElementById('noFiles');
const analysisSection = document.getElementById('analysisSection');
const analysisStatus = document.getElementById('analysisStatus');
const infoModal = document.getElementById('infoModal');
const modalTitle = document.getElementById('modalTitle');
const modalMessage = document.getElementById('modalMessage');
const closeModal = document.getElementById('closeModal');
const toast = document.getElementById('toast');

// Inicialización
document.addEventListener('DOMContentLoaded', function() {
    initializeEventListeners();
    loadUploadedFiles(); // SOLO esta llamada inicial
});

// Event Listeners - versión corregida
function initializeEventListeners() {
    // Upload area events
    uploadArea.addEventListener('click', () => fileInput.click());
    uploadArea.addEventListener('dragover', handleDragOver);
    uploadArea.addEventListener('dragleave', handleDragLeave);
    uploadArea.addEventListener('drop', handleDrop);
    
    // File input change - SIN preventDefault para permitir diálogo de archivos
    fileInput.addEventListener('change', handleFileSelect);
    
    // Modal events
    closeModal.addEventListener('click', hideModal);
    window.addEventListener('click', (e) => {
        if (e.target === infoModal) hideModal();
    });
}

// Drag and Drop handlers
function handleDragOver(e) {
    e.preventDefault();
    uploadArea.classList.add('dragover');
}

function handleDragLeave(e) {
    e.preventDefault();
    uploadArea.classList.remove('dragover');
}

function handleDrop(e) {
    e.preventDefault();
    uploadArea.classList.remove('dragover');
    
    const files = e.dataTransfer.files;
    if (files.length > 0) {
        handleFileUpload(files[0]);
    }
}

// File selection handler - versión corregida sin preventDefault
function handleFileSelect(e) {
    // NO usar preventDefault() aquí - bloquea el diálogo de archivos
    const file = e.target.files[0];
    if (file) {
        handleFileUpload(file);
    }
    
    // Limpiar el input para permitir seleccionar el mismo archivo nuevamente
    e.target.value = '';
}

// File upload handler
async function handleFileUpload(file) {
    // Validar tipo de archivo
    const allowedTypes = ['.xlsx', '.xls'];
    const fileExtension = '.' + file.name.split('.').pop().toLowerCase();
    
    if (!allowedTypes.includes(fileExtension)) {
        showToast('Error: Solo se permiten archivos Excel (.xlsx, .xls)', 'error');
        return;
    }
    
    // Mostrar progreso
    showUploadProgress();
    
    try {
        const formData = new FormData();
        formData.append('file', file);
        
        const response = await fetch(`${API_BASE_URL}/files/upload`, {
            method: 'POST',
            body: formData
        });
        
        if (!response.ok) {
            const errorData = await response.json();
            throw new Error(errorData.detail || 'Error al subir archivo');
        }
        
        const result = await response.json();
        
        // Actualizar progreso al 100%
        updateProgress(100, 'Archivo subido correctamente');
        
        setTimeout(() => {
            hideUploadProgress();
            showToast('Archivo subido exitosamente', 'success');
            // COMENTAR ESTA LÍNEA COMPLETAMENTE
            // loadUploadedFiles(); 
        }, 1000);
        
    } catch (error) {
        hideUploadProgress();
        showToast(`Error: ${error.message}`, 'error');
        console.error('Error uploading file:', error);
    }
}

// Mostrar/ocultar progreso de subida
function showUploadProgress() {
    uploadProgress.style.display = 'block';
    updateProgress(0, 'Preparando subida...');
    
    // Simular progreso
    let progress = 0;
    const interval = setInterval(() => {
        progress += Math.random() * 30;
        if (progress >= 90) {
            clearInterval(interval);
            updateProgress(90, 'Procesando archivo...');
        } else {
            updateProgress(progress, 'Subiendo archivo...');
        }
    }, 200);
}

function hideUploadProgress() {
    uploadProgress.style.display = 'none';
    progressFill.style.width = '0%';
}

function updateProgress(percent, text) {
    progressFill.style.width = `${percent}%`;
    progressText.textContent = text;
}

// Renderizar lista de archivos
function renderFilesList() {
    if (uploadedFiles.length === 0) {
        noFiles.style.display = 'block';
        return;
    }
    
    noFiles.style.display = 'none';
    
    const filesHTML = uploadedFiles.map(file => {
        const uploadDate = new Date(file.upload_date).toLocaleString('es-ES');
        const fileSize = formatFileSize(file.file_size);
        
        return `
            <div class="file-item" data-file-id="${file.id}">
                <div class="file-info-left">
                    <i class="fas fa-file-excel"></i>
                    <div class="file-details">
                        <h4>${file.original_filename}</h4>
                        <p>${fileSize} • ${file.records_count} registros • ${uploadDate}</p>
                        <span class="status-badge status-${file.status}">${getStatusText(file.status)}</span>
                    </div>
                </div>
                <div class="file-actions">
                    ${file.status === 'uploaded' ? `
                        <button class="btn btn-primary" onclick="analyzeFile('${file.id}')">
                            <i class="fas fa-chart-line"></i> Analizar
                        </button>
                    ` : ''}
                    ${file.status === 'completed' ? `
                        <button class="btn btn-secondary" onclick="viewCharts('${file.id}')">
                            <i class="fas fa-chart-bar"></i> Ver Gráficas
                        </button>
                        <button class="btn btn-primary" onclick="downloadReport('${file.id}')">
                            <i class="fas fa-download"></i> Descargar Reporte
                        </button>
                    ` : ''}
                    <button class="btn btn-danger" onclick="deleteFile('${file.id}')">
                        <i class="fas fa-trash"></i> Eliminar
                    </button>
                </div>
            </div>
        `;
    }).join('');
    
    filesList.innerHTML = filesHTML;
}

// Analizar archivo
async function analyzeFile(fileId) {
    try {
        showAnalysisProgress();
        
        const response = await fetch(`${API_BASE_URL}/analysis/analyze`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({ file_id: fileId })
        });
        
        if (!response.ok) {
            const errorData = await response.json();
            throw new Error(errorData.detail || 'Error en el análisis');
        }
        
        const result = await response.json();
        currentAnalysis = result;
        
        // Simular progreso de análisis
        await simulateAnalysisSteps();
        
        // Esperar más tiempo antes de ocultar para que el usuario vea todos los pasos completados
        await new Promise(resolve => setTimeout(resolve, 2500)); // Aumentado de 1000 a 2500ms
        
        hideAnalysisProgress();
        showToast('Análisis completado. Descargando reporte...', 'success');
        
        // Descargar automáticamente el reporte
        await downloadReport(result.id, true);
        
        // Recargar lista de archivos
        loadUploadedFiles();
        
    } catch (error) {
        hideAnalysisProgress();
        showToast(`Error en el análisis: ${error.message}`, 'error');
        console.error('Error analyzing file:', error);
    }
}

// Mostrar progreso de análisis
// Función mejorada para simular pasos del análisis
async function simulateAnalysisSteps() {
    const steps = ['step1', 'step2', 'step3', 'step4'];
    const stepTimes = [0, 2000, 3000, 2500]; // Primer paso inmediato, luego delays
    
    for (let i = 0; i < steps.length; i++) {
        // Esperar antes de activar el siguiente paso (excepto el primero)
        if (i > 0) {
            await new Promise(resolve => setTimeout(resolve, stepTimes[i]));
        }
        
        // Activar paso actual
        const currentStep = document.getElementById(steps[i]);
        if (currentStep) {
            currentStep.classList.add('active');
            
            // Cambiar icono a completado después de un breve delay
            setTimeout(() => {
                const icon = currentStep.querySelector('i');
                if (icon) {
                    icon.className = 'fas fa-check-circle';
                }
            }, 800);
        }
    }
    
    // Esperar un poco más al final para que el usuario vea todos los pasos completados
    await new Promise(resolve => setTimeout(resolve, 2000));
}

// Función mejorada para reiniciar los pasos del análisis
function resetAnalysisSteps() {
    const steps = ['step1', 'step2', 'step3', 'step4'];
    const originalIcons = ['fas fa-upload', 'fas fa-cog fa-spin', 'fas fa-chart-bar', 'fas fa-file-pdf'];
    
    steps.forEach((stepId, index) => {
        const step = document.getElementById(stepId);
        if (step) {
            const icon = step.querySelector('i');
            
            // Remover clase active de todos los pasos
            step.classList.remove('active');
            
            // Restaurar iconos originales
            if (icon) {
                icon.className = originalIcons[index];
            }
        }
    });
    
    // Activar solo el primer paso inicialmente
    const firstStep = document.getElementById('step1');
    if (firstStep) {
        firstStep.classList.add('active');
    }
}

// Función mejorada para mostrar el progreso del análisis
function showAnalysisProgress() {
    console.log('Mostrando progreso de análisis...'); // Para debug
    
    // Reiniciar todos los pasos antes de mostrar
    resetAnalysisSteps();
    
    const analysisSection = document.getElementById('analysisSection');
    if (analysisSection) {
        analysisSection.style.display = 'block';
        analysisSection.scrollIntoView({ behavior: 'smooth' });
    } else {
        console.error('No se encontró analysisSection');
    }
}

function hideAnalysisProgress() {
    const analysisSection = document.getElementById('analysisSection');
    if (analysisSection) {
        analysisSection.style.display = 'none';
    }
}

// Nueva función para reiniciar los pasos del análisis
function resetAnalysisSteps() {
    const steps = ['step1', 'step2', 'step3', 'step4'];
    
    steps.forEach((stepId, index) => {
        const step = document.getElementById(stepId);
        const icon = step.querySelector('i');
        
        // Remover clase active de todos los pasos
        step.classList.remove('active');
        
        // Restaurar iconos originales
        if (index === 0) {
            icon.className = 'fas fa-check-circle';
            step.classList.add('active'); // Solo el primer paso activo inicialmente
        } else if (index === 1) {
            icon.className = 'fas fa-cog fa-spin';
        } else if (index === 2) {
            icon.className = 'fas fa-chart-bar';
        } else if (index === 3) {
            icon.className = 'fas fa-file-pdf';
        }
    });
}

// Función mejorada para simular pasos del análisis con tiempos más realistas
async function simulateAnalysisSteps() {
    const steps = ['step1', 'step2', 'step3', 'step4'];
    const stepTimes = [1000, 3000, 4000, 2500]; // Tiempos más realistas para cada paso
    
    for (let i = 0; i < steps.length; i++) {
        // Esperar antes de activar el siguiente paso
        if (i > 0) {
            await new Promise(resolve => setTimeout(resolve, stepTimes[i]));
        }
        
        // Activar paso actual
        const currentStep = document.getElementById(steps[i]);
        currentStep.classList.add('active');
        
        // Cambiar icono a completado después de un breve delay
        setTimeout(() => {
            const icon = currentStep.querySelector('i');
            icon.className = 'fas fa-check-circle';
        }, 500);
        
        // Desactivar el paso anterior (excepto el primero)
        if (i > 0) {
            const prevStep = document.getElementById(steps[i-1]);
            // Mantener el icono de check pero quitar el spinning
            const prevIcon = prevStep.querySelector('i');
            prevIcon.className = 'fas fa-check-circle';
        }
    }
    
    // Esperar un poco más al final para que el usuario vea todos los pasos completados
    await new Promise(resolve => setTimeout(resolve, 1500));
}

// Descargar reporte
async function downloadReport(analysisId, isAutoDownload = false) {
    try {
        // Si es descarga automática, buscar el análisis por file_id
        let downloadUrl;
        
        if (isAutoDownload && currentAnalysis) {
            downloadUrl = `${API_BASE_URL}/analysis/download/${currentAnalysis.id}`;
        } else {
            // Buscar análisis por file_id
            const response = await fetch(`${API_BASE_URL}/analysis/results/${analysisId}`);
            if (!response.ok) throw new Error('No se encontró el análisis');
            
            const analyses = await response.json();
            if (analyses.length === 0) throw new Error('No hay análisis disponibles');
            
            const latestAnalysis = analyses[analyses.length - 1];
            downloadUrl = `${API_BASE_URL}/analysis/download/${latestAnalysis.id}`;
        }
        
        // Crear enlace de descarga
        const link = document.createElement('a');
        link.href = downloadUrl;
        link.download = `reporte_medico_${new Date().toISOString().split('T')[0]}.pdf`;
        document.body.appendChild(link);
        link.click();
        document.body.removeChild(link);
        
        if (!isAutoDownload) {
            showToast('Descarga iniciada', 'success');
        }
        
    } catch (error) {
        showToast(`Error al descargar: ${error.message}`, 'error');
        console.error('Error downloading report:', error);
    }
}

// Eliminar archivo
async function deleteFile(fileId) {
    if (!confirm('¿Estás seguro de que quieres eliminar este archivo?')) {
        return;
    }
    
    try {
        const response = await fetch(`${API_BASE_URL}/files/${fileId}`, {
            method: 'DELETE'
        });
        
        if (!response.ok) throw new Error('Error al eliminar archivo');
        
        showToast('Archivo eliminado correctamente', 'success');
        loadUploadedFiles();
        
    } catch (error) {
        showToast(`Error: ${error.message}`, 'error');
        console.error('Error deleting file:', error);
    }
}

// Utilidades
function formatFileSize(bytes) {
    if (bytes === 0) return '0 Bytes';
    const k = 1024;
    const sizes = ['Bytes', 'KB', 'MB', 'GB'];
    const i = Math.floor(Math.log(bytes) / Math.log(k));
    return parseFloat((bytes / Math.pow(k, i)).toFixed(2)) + ' ' + sizes[i];
}

function getStatusText(status) {
    const statusMap = {
        'uploaded': 'Subido',
        'processing': 'Procesando',
        'completed': 'Completado',
        'error': 'Error'
    };
    return statusMap[status] || status;
}

// Modal functions
function showModal(title, message) {
    modalTitle.textContent = title;
    modalMessage.textContent = message;
    infoModal.style.display = 'block';
}

function hideModal() {
    infoModal.style.display = 'none';
}

// Toast notifications
function showToast(message, type = 'info') {
    const toastContent = toast.querySelector('.toast-content');
    const toastIcon = toast.querySelector('.toast-icon');
    const toastMessage = toast.querySelector('.toast-message');
    
    // Configurar icono según tipo
    const icons = {
        'success': 'fas fa-check-circle',
        'error': 'fas fa-exclamation-circle',
        'info': 'fas fa-info-circle'
    };
    
    toastIcon.className = `toast-icon ${icons[type] || icons.info}`;
    toastMessage.textContent = message;
    
    // Aplicar clase de tipo
    toast.className = `toast ${type}`;
    toast.classList.add('show');
    
    // Auto-hide después de 4 segundos
    setTimeout(() => {
        toast.classList.remove('show');
    }, 4000);
}

// Función global para hacer disponibles las funciones en el HTML
window.analyzeFile = analyzeFile;
window.downloadReport = downloadReport;
window.deleteFile = deleteFile;

// Variables globales para las gráficas
let currentCharts = [];

// Función para ver gráficas
async function viewCharts(fileId) {
    try {
        showToast('Cargando gráficas...', 'info');
        
        // Obtener datos de las gráficas para este archivo
        const response = await fetch(`${API_BASE_URL}/analysis/charts/${fileId}`);
        if (!response.ok) {
            throw new Error('No se pudieron cargar las gráficas');
        }
        
        const chartsData = await response.json();
        
        // Mostrar modal
        showChartsModal(chartsData, fileId);
        
    } catch (error) {
        showToast(`Error al cargar gráficas: ${error.message}`, 'error');
        console.error('Error loading charts:', error);
    }
}

// Mostrar modal de gráficas
function showChartsModal(chartsData, fileId) {
    const modal = document.getElementById('chartsModal');
    const title = document.getElementById('chartsModalTitle');
    
    // Encontrar el archivo para mostrar su nombre
    const file = uploadedFiles.find(f => f.id === fileId);
    title.textContent = `Gráficas del Análisis - ${file ? file.original_filename : 'Archivo'}`;
    
    // Limpiar gráficas anteriores
    destroyCurrentCharts();
    
    // Crear nuevas gráficas
    createCharts(chartsData);
    
    // Mostrar modal
    modal.style.display = 'block';
}

// Crear gráficas con Chart.js
function createCharts(data) {
    // Gráfica de distribución de diagnósticos
    const diagnosticCtx = document.getElementById('diagnosticChart').getContext('2d');
    const diagnosticChart = new Chart(diagnosticCtx, {
        type: 'pie',
        data: {
            labels: data.diagnostic_distribution?.labels || ['Sin datos'],
            datasets: [{
                data: data.diagnostic_distribution?.values || [1],
                backgroundColor: [
                    '#FF6384', '#36A2EB', '#FFCE56', '#4BC0C0',
                    '#9966FF', '#FF9F40', '#FF6384', '#C9CBCF'
                ]
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            plugins: {
                legend: {
                    position: 'bottom'
                }
            }
        }
    });
    currentCharts.push(diagnosticChart);

    // Gráfica de edad por diagnóstico
    const ageCtx = document.getElementById('ageChart').getContext('2d');
    const ageChart = new Chart(ageCtx, {
        type: 'bar',
        data: {
            labels: data.age_by_diagnosis?.labels || ['Sin datos'],
            datasets: [{
                label: 'Edad Promedio',
                data: data.age_by_diagnosis?.values || [0],
                backgroundColor: '#36A2EB',
                borderColor: '#36A2EB',
                borderWidth: 1
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            scales: {
                y: {
                    beginAtZero: true,
                    title: {
                        display: true,
                        text: 'Edad (años)'
                    }
                }
            }
        }
    });
    currentCharts.push(ageChart);

    // Gráfica de factores de riesgo
    const riskCtx = document.getElementById('riskChart').getContext('2d');
    const riskChart = new Chart(riskCtx, {
        type: 'doughnut',
        data: {
            labels: data.risk_factors?.labels || ['Sin datos'],
            datasets: [{
                data: data.risk_factors?.values || [1],
                backgroundColor: [
                    '#FF6384', '#36A2EB', '#FFCE56', '#4BC0C0',
                    '#9966FF', '#FF9F40'
                ]
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            plugins: {
                legend: {
                    position: 'bottom'
                }
            }
        }
    });
    currentCharts.push(riskChart);

    // Gráfica de correlación
    const correlationCtx = document.getElementById('correlationChart').getContext('2d');
    const correlationChart = new Chart(correlationCtx, {
        type: 'scatter',
        data: {
            datasets: [{
                label: 'Correlación',
                data: data.correlation?.points || [{x: 0, y: 0}],
                backgroundColor: '#FF6384',
                borderColor: '#FF6384'
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            scales: {
                x: {
                    title: {
                        display: true,
                        text: data.correlation?.x_label || 'Variable X'
                    }
                },
                y: {
                    title: {
                        display: true,
                        text: data.correlation?.y_label || 'Variable Y'
                    }
                }
            }
        }
    });
    currentCharts.push(correlationChart);
}

// Destruir gráficas actuales
function destroyCurrentCharts() {
    currentCharts.forEach(chart => {
        if (chart) {
            chart.destroy();
        }
    });
    currentCharts = [];
}

// Cerrar modal de gráficas
function hideChartsModal() {
    const modal = document.getElementById('chartsModal');
    modal.style.display = 'none';
    destroyCurrentCharts();
}

// Event listeners para el modal de gráficas
document.addEventListener('DOMContentLoaded', function() {
    // ... existing code ...
    
    // Agregar event listener para cerrar modal de gráficas
    const closeChartsModal = document.getElementById('closeChartsModal');
    if (closeChartsModal) {
        closeChartsModal.addEventListener('click', hideChartsModal);
    }
    
    // Cerrar modal al hacer clic fuera
    window.addEventListener('click', (e) => {
        const chartsModal = document.getElementById('chartsModal');
        if (e.target === chartsModal) {
            hideChartsModal();
        }
        // ... existing modal close code ...
    });
});

// Hacer la función disponible globalmente
window.viewCharts = viewCharts;

// Función para refrescar manualmente
function refreshFilesList() {
    loadUploadedFiles(true);
    showToast('Lista actualizada', 'info');
}

// En initializeEventListeners, agregar:
// document.getElementById('refresh-btn')?.addEventListener('click', refreshFilesList);