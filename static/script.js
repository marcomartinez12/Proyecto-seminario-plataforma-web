// Configuración de la API
const API_BASE_URL = 'http://localhost:8000/api';

// Variables globales
let uploadedFiles = [];
let currentAnalysis = null;
let isAnalyzing = false; // Nueva variable para controlar el estado

// Cargar archivos subidos con debounce
let loadFilesTimeout;
async function loadUploadedFiles(force = false) {
    // Si estamos analizando y no es forzado, no recargar
    if (isAnalyzing && !force) {
        return;
    }
    
    // Debounce para evitar llamadas múltiples
    if (loadFilesTimeout) {
        clearTimeout(loadFilesTimeout);
    }
    
    loadFilesTimeout = setTimeout(async () => {
        try {
            const response = await fetch(`${API_BASE_URL}/files/list`);
            if (!response.ok) throw new Error('Error al cargar archivos');
            
            uploadedFiles = await response.json();
            renderFilesList();
            
        } catch (error) {
            console.error('Error loading files:', error);
        }
    }, 300); // Esperar 300ms antes de ejecutar
}

// Analizar archivo - versión optimizada
async function analyzeFile(fileId) {
    try {
        isAnalyzing = true; // Marcar que estamos analizando
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
        
        hideAnalysisProgress();
        showToast('Análisis completado. Descargando reporte...', 'success');
        
        // Descargar automáticamente el reporte
        await downloadReport(result.id, true);
        
        // Solo recargar archivos una vez al final y de forma forzada
        isAnalyzing = false;
        loadUploadedFiles(true);
        
    } catch (error) {
        isAnalyzing = false;
        hideAnalysisProgress();
        showToast(`Error en el análisis: ${error.message}`, 'error');
        console.error('Error analyzing file:', error);
    }
}

// Eliminar archivo - versión optimizada
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
        
        // Actualizar la lista local inmediatamente sin hacer llamada a la API
        uploadedFiles = uploadedFiles.filter(file => file.id !== fileId);
        renderFilesList();
        
        // Recargar desde el servidor después de un delay
        setTimeout(() => loadUploadedFiles(true), 1000);
        
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
    loadUploadedFiles();
});

// Event Listeners
function initializeEventListeners() {
    // Upload area events
    uploadArea.addEventListener('click', () => fileInput.click());
    uploadArea.addEventListener('dragover', handleDragOver);
    uploadArea.addEventListener('dragleave', handleDragLeave);
    uploadArea.addEventListener('drop', handleDrop);
    
    // File input change
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

// File selection handler
function handleFileSelect(e) {
    const file = e.target.files[0];
    if (file) {
        handleFileUpload(file);
    }
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
            loadUploadedFiles(); // Recargar lista
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
function showAnalysisProgress() {
    analysisSection.style.display = 'block';
    analysisSection.scrollIntoView({ behavior: 'smooth' });
}

function hideAnalysisProgress() {
    analysisSection.style.display = 'none';
}

// Simular pasos del análisis
async function simulateAnalysisSteps() {
    const steps = ['step1', 'step2', 'step3', 'step4'];
    
    for (let i = 0; i < steps.length; i++) {
        await new Promise(resolve => setTimeout(resolve, 2000));
        
        // Activar paso actual
        const currentStep = document.getElementById(steps[i]);
        currentStep.classList.add('active');
        
        // Cambiar icono a completado
        const icon = currentStep.querySelector('i');
        icon.className = 'fas fa-check-circle';
    }
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