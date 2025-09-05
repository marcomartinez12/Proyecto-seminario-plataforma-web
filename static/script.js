const API_BASE_URL = 'http://localhost:8000/api';

// Variables globales
let uploadedFiles = [];
let currentAnalysis = null;
let isAnalyzing = false;
let currentCharts = [];

// Control estricto de carga de archivos
let loadFilesTimeout;
let lastLoadTime = 0;
let isLoadingFiles = false;

// FUNCIÓN ÚNICA DE CARGA CON CONTROL ESTRICTO
async function loadUploadedFiles(force = false) {
    const now = Date.now();
    
    // Prevenir múltiples ejecuciones simultáneas
    if (isLoadingFiles && !force) {
        return;
    }
    
    // Si estamos analizando y no es forzado, no recargar
    if (isAnalyzing && !force) {
        return;
    }
    
    // Control de tiempo mínimo entre llamadas
    if (!force && (now - lastLoadTime) < 3000) {
        return;
    }
    
    // Limpiar timeout anterior
    if (loadFilesTimeout) {
        clearTimeout(loadFilesTimeout);
    }
    
    loadFilesTimeout = setTimeout(async () => {
        if (isLoadingFiles) return;
        
        try {
            isLoadingFiles = true;
            lastLoadTime = Date.now();
            
            const response = await fetch(`${API_BASE_URL}/files/list`);
            if (!response.ok) throw new Error('Error al cargar archivos');
            
            uploadedFiles = await response.json();
            renderFilesList();
            
        } catch (error) {
            console.error('Error loading files:', error);
        } finally {
            isLoadingFiles = false;
        }
    }, force ? 0 : 1000);
}

// ACTUALIZACIÓN LOCAL SIN REFRESCOS
function updateFileStatus(fileId, newStatus) {
    const fileIndex = uploadedFiles.findIndex(file => file.id === fileId);
    if (fileIndex !== -1) {
        uploadedFiles[fileIndex].status = newStatus;
        renderFilesList();
    }
}

// ELEMENTOS DOM
const uploadArea = document.getElementById('uploadArea');
const fileInput = document.getElementById('fileInput');
const uploadProgress = document.getElementById('uploadProgress');
const progressFill = document.getElementById('progressFill');
const progressText = document.getElementById('progressText');
const filesList = document.getElementById('filesList');
const noFiles = document.getElementById('noFiles');
const analysisSection = document.getElementById('analysisSection');
const infoModal = document.getElementById('infoModal');
const modalTitle = document.getElementById('modalTitle');
const modalMessage = document.getElementById('modalMessage');
const closeModal = document.getElementById('closeModal');
const toast = document.getElementById('toast');

// INICIALIZACIÓN ÚNICA
document.addEventListener('DOMContentLoaded', function() {
    initializeEventListeners();
    loadUploadedFiles(); // ÚNICA llamada inicial
});

// EVENT LISTENERS ÚNICOS
function initializeEventListeners() {
    // Upload area events
    uploadArea.addEventListener('click', () => fileInput.click());
    uploadArea.addEventListener('dragover', handleDragOver);
    uploadArea.addEventListener('dragleave', handleDragLeave);
    uploadArea.addEventListener('drop', handleDrop);
    
    // File input
    fileInput.addEventListener('change', handleFileSelect);
    
    // Modal events
    closeModal.addEventListener('click', hideModal);
    
    // Charts modal
    const closeChartsModal = document.getElementById('closeChartsModal');
    if (closeChartsModal) {
        closeChartsModal.addEventListener('click', hideChartsModal);
    }
    
    // Window click events
    window.addEventListener('click', (e) => {
        if (e.target === infoModal) hideModal();
        
        const chartsModal = document.getElementById('chartsModal');
        if (e.target === chartsModal) hideChartsModal();
    });
}

// DRAG AND DROP
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

function handleFileSelect(e) {
    const file = e.target.files[0];
    if (file) {
        handleFileUpload(file);
    }
    e.target.value = '';
}

// SUBIDA DE ARCHIVOS SIN REFRESCOS
async function handleFileUpload(file) {
    const allowedTypes = ['.xlsx', '.xls'];
    const fileExtension = '.' + file.name.split('.').pop().toLowerCase();
    
    if (!allowedTypes.includes(fileExtension)) {
        showToast('Error: Solo se permiten archivos Excel (.xlsx, .xls)', 'error');
        return;
    }
    
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
        
        updateProgressFluid(100, 'Archivo subido correctamente');
        
        setTimeout(() => {
            hideUploadProgress();
            showToast('Archivo subido correctamente', 'success');

            // SOLO ACTUALIZACIÓN LOCAL - NO MÁS REFRESCOS
            if (result) {
                const exists = uploadedFiles.some(f => f.id === result.id);
                if (!exists) {
                    uploadedFiles.push(result);
                } else {
                    uploadedFiles = uploadedFiles.map(f => f.id === result.id ? { ...f, ...result } : f);
                }
                renderFilesList();
            }
        }, 1000);
        
    } catch (error) {
        hideUploadProgress();
        showToast(`Error: ${error.message}`, 'error');
        console.error('Error uploading file:', error);
    }
}

// PROGRESO DE SUBIDA
function showUploadProgress() {
    if (!uploadProgress || !progressFill || !progressText) return;
    
    // Mostrar inmediatamente el div de progreso
    uploadProgress.style.display = 'block';
    progressFill.style.width = '0%';
    progressText.textContent = 'Iniciando subida...';
}

function updateProgressFluid(percent, text) {
    if (progressFill) {
        progressFill.style.width = `${percent}%`;
        // Añadir transición suave
        progressFill.style.transition = 'width 0.3s ease';
    }
    if (progressText) progressText.textContent = text;
}

function hideUploadProgress() {
    if (!uploadProgress) return;
    
    setTimeout(() => {
        uploadProgress.style.display = 'none';
        if (progressFill) {
            progressFill.style.width = '0%';
            progressFill.style.transition = 'none'; // Resetear transición
        }
        if (progressText) progressText.textContent = 'Iniciando subida...';
    }, 1000);
}

// ANÁLISIS ÚNICO SIN REFRESCOS
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
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ file_id: fileId })
        });

        if (!response.ok) {
            const errorData = await response.json();
            throw new Error(errorData.detail || 'Error en el análisis');
        }

        const result = await response.json();
        currentAnalysis = result;

        await simulateAnalysisStepsFluid();
        await new Promise(resolve => setTimeout(resolve, 2000));

        hideAnalysisProgress();
        showToast('Análisis completado. Descargando reporte...', 'success');
        await downloadReport(result.id, true);

        // SOLO ACTUALIZACIÓN LOCAL - CERO REFRESCOS
        updateFileStatus(fileId, 'completed');

    } catch (error) {
        hideAnalysisProgress();
        showToast(`Error en el análisis: ${error.message}`, 'error');
    } finally {
        isAnalyzing = false;
    }
}

// ANIMACIONES FLUIDAS
function showAnalysisProgress() {
    const analysisSection = document.getElementById('analysisSection');
    const analysisSteps = analysisSection.querySelector('.analysis-steps');
    
    analysisSection.className = 'analysis-section-fluid';
    analysisSteps.className = 'analysis-steps-fluid';
    
    const steps = analysisSteps.querySelectorAll('.step');
    steps.forEach(step => {
        step.className = 'step-fluid';
        
        const icon = step.querySelector('.step-icon');
        const text = step.querySelector('.step-text');
        
        if (icon) icon.className = 'step-icon-fluid';
        if (text) text.className = 'step-text-fluid';
    });
    
    resetAnalysisStepsFluid();
    analysisSection.style.display = 'block';
    analysisSection.scrollIntoView({ behavior: 'smooth' });
}

async function simulateAnalysisStepsFluid() {
    const steps = ['step1', 'step2', 'step3', 'step4'];
    const stepTimes = [500, 2500, 3500, 3000];
    const stepTexts = [
        'Cargando y validando archivo...',
        'Procesando datos y estadísticas...',
        'Generando gráficas y visualizaciones...',
        'Creando reporte PDF final...'
    ];
    
    for (let i = 0; i < steps.length; i++) {
        if (i > 0) {
            await new Promise(resolve => setTimeout(resolve, stepTimes[i]));
        }
        
        const currentStep = document.getElementById(steps[i]);
        if (currentStep) {
            steps.forEach(stepId => {
                const step = document.getElementById(stepId);
                if (step && step !== currentStep) {
                    step.classList.remove('active');
                    step.classList.add('completed');
                }
            });
            
            currentStep.classList.add('active');
            
            const stepText = currentStep.querySelector('.step-text-fluid');
            if (stepText) {
                stepText.textContent = stepTexts[i];
            }
            
            setTimeout(() => {
                const icon = currentStep.querySelector('i');
                if (icon && i < steps.length - 1) {
                    icon.className = 'fas fa-check-circle';
                }
            }, stepTimes[i] * 0.8);
        }
    }
    
    await new Promise(resolve => setTimeout(resolve, 1500));
    const lastStep = document.getElementById(steps[steps.length - 1]);
    if (lastStep) {
        lastStep.classList.remove('active');
        lastStep.classList.add('completed');
        const icon = lastStep.querySelector('i');
        if (icon) {
            icon.className = 'fas fa-check-circle';
        }
    }
    
    await new Promise(resolve => setTimeout(resolve, 2000));
}

function resetAnalysisStepsFluid() {
    const steps = ['step1', 'step2', 'step3', 'step4'];
    const originalIcons = ['fas fa-upload', 'fas fa-cog fa-spin', 'fas fa-chart-bar', 'fas fa-file-pdf'];
    const originalTexts = [
        'Subiendo archivo',
        'Procesando datos',
        'Generando gráficas',
        'Creando reporte'
    ];
    
    steps.forEach((stepId, index) => {
        const step = document.getElementById(stepId);
        if (step) {
            const icon = step.querySelector('i');
            const text = step.querySelector('.step-text-fluid');
            
            step.classList.remove('active', 'completed');
            
            if (icon) icon.className = originalIcons[index];
            if (text) text.textContent = originalTexts[index];
        }
    });
}

function hideAnalysisProgress() {
    const analysisSection = document.getElementById('analysisSection');
    
    analysisSection.style.opacity = '0';
    analysisSection.style.transform = 'translateY(20px)';
    
    setTimeout(() => {
        analysisSection.style.display = 'none';
        analysisSection.style.opacity = '1';
        analysisSection.style.transform = 'translateY(0)';
        analysisSection.className = 'analysis-section';
    }, 600);
}

// ELIMINAR ARCHIVO SIN REFRESCOS
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

        // SOLO ACTUALIZACIÓN LOCAL - CERO REFRESCOS
        uploadedFiles = uploadedFiles.filter(file => file.id !== fileId);
        renderFilesList();

    } catch (error) {
        showToast(`Error: ${error.message}`, 'error');
        console.error('Error deleting file:', error);
    }
}

// RENDERIZAR LISTA
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

// DESCARGAR REPORTE
async function downloadReport(analysisId, isAutoDownload = false) {
    try {
        let downloadUrl;
        
        if (isAutoDownload && currentAnalysis) {
            downloadUrl = `${API_BASE_URL}/analysis/download/${currentAnalysis.id}`;
        } else {
            const response = await fetch(`${API_BASE_URL}/analysis/results/${analysisId}`);
            if (!response.ok) throw new Error('No se encontró el análisis');
            
            const analyses = await response.json();
            if (analyses.length === 0) throw new Error('No hay análisis disponibles');
            
            const latestAnalysis = analyses[analyses.length - 1];
            downloadUrl = `${API_BASE_URL}/analysis/download/${latestAnalysis.id}`;
        }
        
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

// VER GRÁFICAS
async function viewCharts(fileId) {
    try {
        showToast('Cargando gráficas...', 'info');
        
        const response = await fetch(`${API_BASE_URL}/analysis/charts/${fileId}`);
        if (!response.ok) {
            throw new Error('No se pudieron cargar las gráficas');
        }
        
        const chartsData = await response.json();
        showChartsModal(chartsData, fileId);
        
    } catch (error) {
        showToast(`Error al cargar gráficas: ${error.message}`, 'error');
        console.error('Error loading charts:', error);
    }
}

function showChartsModal(chartsData, fileId) {
    const modal = document.getElementById('chartsModal');
    const title = document.getElementById('chartsModalTitle');
    
    const file = uploadedFiles.find(f => f.id === fileId);
    title.textContent = `Gráficas del Análisis - ${file ? file.original_filename : 'Archivo'}`;
    
    destroyCurrentCharts();
    createCharts(chartsData);
    modal.style.display = 'block';
}

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

function destroyCurrentCharts() {
    currentCharts.forEach(chart => {
        if (chart) {
            chart.destroy();
        }
    });
    currentCharts = [];
}

function hideChartsModal() {
    const modal = document.getElementById('chartsModal');
    modal.style.display = 'none';
    destroyCurrentCharts();
}

// UTILIDADES
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

function showModal(title, message) {
    modalTitle.textContent = title;
    modalMessage.textContent = message;
    infoModal.style.display = 'block';
}

function hideModal() {
    infoModal.style.display = 'none';
}

function showToast(message, type = 'info') {
    const toastContent = toast.querySelector('.toast-content');
    const toastIcon = toast.querySelector('.toast-icon');
    const toastMessage = toast.querySelector('.toast-message');
    
    const icons = {
        'success': 'fas fa-check-circle',
        'error': 'fas fa-exclamation-circle',
        'info': 'fas fa-info-circle'
    };
    
    toastIcon.className = `toast-icon ${icons[type] || icons.info}`;
    toastMessage.textContent = message;
    
    toast.className = `toast ${type}`;
    toast.classList.add('show');
    
    setTimeout(() => {
        toast.classList.remove('show');
    }, 4000);
}

// FUNCIONES GLOBALES
window.analyzeFile = analyzeFile;
window.downloadReport = downloadReport;
window.deleteFile = deleteFile;
window.viewCharts = viewCharts;