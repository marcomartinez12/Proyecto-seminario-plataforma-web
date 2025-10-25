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

    // AI modal
    const closeAIModal = document.getElementById('closeAIModal');
    if (closeAIModal) {
        closeAIModal.addEventListener('click', hideAIModal);
    }

    // Window click events
    window.addEventListener('click', (e) => {
        if (e.target === infoModal) hideModal();

        const chartsModal = document.getElementById('chartsModal');
        if (e.target === chartsModal) hideChartsModal();

        const aiModal = document.getElementById('aiModal');
        if (e.target === aiModal) hideAIModal();
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
                        <button class="btn btn-info" onclick="viewDetailedAnalysis('${file.id}')" style="background: linear-gradient(135deg, #667eea, #764ba2); border: none;">
                            <i class="fas fa-microscope"></i> Análisis Detallado
                        </button>
                        <button class="btn btn-success" onclick="analyzeWithAI('${file.id}')">
                            <i class="fas fa-brain"></i> Análisis IA
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
    // Paleta de colores MEJORADA - Más vibrante y visible
    const colors = {
        primary: '#2563eb',      // Azul brillante
        secondary: '#3b82f6',    // Azul claro
        accent: '#06b6d4',       // Cyan brillante
        success: '#10b981',      // Verde esmeralda
        warning: '#f59e0b',      // Naranja brillante
        danger: '#ef4444',       // Rojo brillante
        info: '#8b5cf6',         // Púrpura vibrante
        light: '#f3f4f6',        // Gris claro
        gradient: ['#8b5cf6', '#3b82f6', '#06b6d4', '#10b981', '#f59e0b', '#ef4444', '#ec4899', '#6366f1']
    };

    // Configuración global de fuentes MEJORADA - Más grandes y legibles
    Chart.defaults.font.family = 'Inter, sans-serif';
    Chart.defaults.font.size = 18;
    Chart.defaults.font.weight = '700';
    Chart.defaults.color = '#ffffff';

    // Plugin para etiquetas de datos MEJORADO - Más grandes y visibles
    const dataLabelsPlugin = {
        id: 'datalabels',
        afterDatasetsDraw(chart, args, options) {
            const { ctx, data } = chart;
            ctx.save();

            data.datasets.forEach((dataset, datasetIndex) => {
                const meta = chart.getDatasetMeta(datasetIndex);
                if (!meta.hidden) {
                    meta.data.forEach((element, index) => {
                        const value = dataset.data[index];
                        if (value > 0) {
                            ctx.fillStyle = '#FFFFFF';
                            ctx.strokeStyle = '#000000';
                            ctx.lineWidth = 5;
                            ctx.font = 'bold 28px Inter';
                            ctx.textAlign = 'center';
                            ctx.textBaseline = 'middle';

                            let label = '';
                            if (chart.config.type === 'pie' || chart.config.type === 'doughnut') {
                                const total = dataset.data.reduce((a, b) => a + b, 0);
                                const percentage = ((value / total) * 100).toFixed(1);
                                label = `${percentage}%`;
                            } else {
                                label = value.toString();
                            }

                            const position = element.tooltipPosition();
                            ctx.strokeText(label, position.x, position.y);
                            ctx.fillText(label, position.x, position.y);
                        }
                    });
                }
            });

            ctx.restore();
        }
    };

    // Gráfica de distribución de diagnósticos (PIE) MEJORADA
    const diagnosticCtx = document.getElementById('diagnosticChart').getContext('2d');
    const diagnosticChart = new Chart(diagnosticCtx, {
        type: 'pie',
        data: {
            labels: data.diagnostic_distribution?.labels || ['Sin datos'],
            datasets: [{
                data: data.diagnostic_distribution?.values || [1],
                backgroundColor: colors.gradient,
                borderColor: '#1a1a1a',
                borderWidth: 4,
                hoverBorderWidth: 6,
                hoverOffset: 15
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            layout: {
                padding: {
                    bottom: 10
                }
            },
            plugins: {
                legend: {
                    position: 'bottom',
                    labels: {
                        padding: 25,
                        usePointStyle: true,
                        pointStyle: 'circle',
                        font: {
                            family: 'Inter',
                            size: 20,
                            weight: '700'
                        },
                        color: '#ffffff',
                        boxWidth: 22,
                        boxHeight: 22,
                        textAlign: 'left',
                        generateLabels: (chart) => {
                            const data = chart.data;
                            if (data.labels.length && data.datasets.length) {
                                return data.labels.map((label, i) => {
                                    const meta = chart.getDatasetMeta(0);
                                    const style = meta.controller.getStyle(i);
                                    // Truncar texto largo
                                    const shortLabel = label.length > 15 ? label.substring(0, 15) + '...' : label;
                                    return {
                                        text: shortLabel,
                                        fillStyle: style.backgroundColor,
                                        strokeStyle: style.borderColor,
                                        lineWidth: style.borderWidth,
                                        fontColor: '#ffffff',
                                        hidden: !chart.getDataVisibility(i),
                                        index: i
                                    };
                                });
                            }
                            return [];
                        }
                    }
                },
                tooltip: {
                    backgroundColor: 'rgba(0, 0, 0, 0.9)',
                    titleColor: '#FFFFFF',
                    bodyColor: '#FFFFFF',
                    borderColor: colors.info,
                    borderWidth: 2,
                    cornerRadius: 12,
                    padding: 20,
                    titleFont: { family: 'Inter', size: 20, weight: 'bold' },
                    bodyFont: { family: 'Inter', size: 18, weight: '600' },
                    displayColors: true,
                    boxPadding: 8,
                    callbacks: {
                        // Mostrar nombre completo en tooltip
                        label: function(context) {
                            const label = context.label || '';
                            const value = context.parsed || 0;
                            const total = context.dataset.data.reduce((a, b) => a + b, 0);
                            const percentage = ((value / total) * 100).toFixed(1);
                            return `${label}: ${value} (${percentage}%)`;
                        }
                    }
                }
            },
            animation: {
                animateRotate: true,
                animateScale: false,
                duration: 600,
                easing: 'easeOutQuart'
            }
        },
        plugins: [dataLabelsPlugin]
    });
    currentCharts.push(diagnosticChart);
    // KPI de edad por diagnóstico (reemplaza el bar chart)
    const ageKpiCard = document.getElementById('ageKpiCard');
    if (ageKpiCard && data.age_by_diagnosis) {
        // Calcular estadísticas de edad
        const ages = data.age_by_diagnosis.values || [];
        const avgAge = ages.length > 0 ? (ages.reduce((a, b) => a + b, 0) / ages.length).toFixed(1) : 0;
        
        // Crear HTML dinámico para la tarjeta KPI simplificada
        ageKpiCard.innerHTML = `
            <div class="kpi-main-value">${avgAge}</div>
            <div class="kpi-label">Edad Promedio (años)</div>
        `;
    }

    // Gráfica de factores de riesgo (DOUGHNUT) MEJORADA
    const riskCtx = document.getElementById('riskChart').getContext('2d');
    const riskChart = new Chart(riskCtx, {
        type: 'doughnut',
        data: {
            labels: data.risk_factors?.labels || ['Sin datos'],
            datasets: [{
                data: data.risk_factors?.values || [1],
                backgroundColor: colors.gradient,
                borderColor: '#1a1a1a',
                borderWidth: 4,
                hoverBorderWidth: 6,
                hoverOffset: 15
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            cutout: '65%',
            plugins: {
                legend: {
                    position: 'bottom',
                    labels: {
                        padding: 30,
                        usePointStyle: true,
                        pointStyle: 'circle',
                        font: {
                            family: 'Inter',
                            size: 20,
                            weight: '700'
                        },
                        color: '#ffffff',
                        boxWidth: 22,
                        boxHeight: 22
                    }
                },
                tooltip: {
                    backgroundColor: 'rgba(0, 0, 0, 0.9)',
                    titleColor: '#FFFFFF',
                    bodyColor: '#FFFFFF',
                    borderColor: colors.info,
                    borderWidth: 2,
                    cornerRadius: 12,
                    padding: 20,
                    titleFont: { family: 'Inter', size: 20, weight: 'bold' },
                    bodyFont: { family: 'Inter', size: 18, weight: '600' },
                    displayColors: true,
                    boxPadding: 8
                }
            },
            animation: {
                animateRotate: true,
                animateScale: true,
                duration: 1200,
                easing: 'easeInOutQuart'
            }
        },
        plugins: [
            dataLabelsPlugin,
            {
                id: 'centerText',
                afterDraw(chart) {
                    const { ctx, chartArea } = chart;
                    ctx.save();

                    const centerX = (chartArea.left + chartArea.right) / 2;
                    const centerY = (chartArea.top + chartArea.bottom) / 2;

                    ctx.fillStyle = '#8b5cf6';
                    ctx.font = 'bold 36px Inter';
                    ctx.textAlign = 'center';
                    ctx.textBaseline = 'middle';
                    ctx.shadowColor = 'rgba(139, 92, 246, 0.5)';
                    ctx.shadowBlur = 15;
                    ctx.fillText('Factores', centerX, centerY - 15);

                    ctx.font = 'bold 30px Inter';
                    ctx.fillStyle = '#a855f7';
                    ctx.fillText('de Riesgo', centerX, centerY + 20);

                    ctx.restore();
                }
            }
        ]
    });
    currentCharts.push(riskChart);
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

// ANÁLISIS CON IA
async function analyzeWithAI(fileId) {
    console.log('=== INICIANDO ANÁLISIS CON IA ===');
    console.log('File ID:', fileId);

    try {
        // Mostrar modal con loading inmediatamente
        console.log('Mostrando modal de loading...');
        showAIModalLoading();
        showToast('Generando análisis con IA...', 'info');

        console.log('Llamando a la API...');
        const response = await fetch(`${API_BASE_URL}/ai-analysis/${fileId}`, {
            method: 'POST'
        });

        console.log('Respuesta recibida. Status:', response.status);

        if (!response.ok) {
            const errorData = await response.json();
            console.error('Error en la API:', errorData);
            hideAIModal();
            throw new Error(errorData.detail || 'Error al generar análisis con IA');
        }

        const result = await response.json();
        console.log('=== RESULTADO DE LA IA ===');
        console.log('Datos completos:', result);
        console.log('Explicación:', result.ai_explanation);

        showAIModal(result);
        showToast('Análisis con IA completado', 'success');

    } catch (error) {
        console.error('=== ERROR EN ANÁLISIS IA ===');
        console.error('Mensaje:', error.message);
        console.error('Error completo:', error);
        showToast(`Error: ${error.message}`, 'error');
    }
}

function showAIModalLoading() {
    console.log('Ejecutando showAIModalLoading...');

    // Eliminar modal anterior si existe
    const oldModal = document.getElementById('aiModalCustom');
    if (oldModal) {
        oldModal.remove();
    }

    // Crear modal completamente desde JavaScript
    const modalHTML = `
        <div id="aiModalCustom" style="
            display: block !important;
            position: fixed !important;
            z-index: 999999 !important;
            left: 0 !important;
            top: 0 !important;
            width: 100vw !important;
            height: 100vh !important;
            background-color: rgba(0, 0, 0, 0.95) !important;
            overflow: auto !important;
        ">
            <div style="
                background: #1a1a1a;
                margin: 10% auto;
                padding: 40px;
                border: 2px solid #8b5cf6;
                border-radius: 15px;
                width: 80%;
                max-width: 800px;
                position: relative;
                box-shadow: 0 0 50px rgba(139, 92, 246, 0.5);
            ">
                <span onclick="hideAIModalCustom()" style="
                    color: #fff;
                    float: right;
                    font-size: 35px;
                    font-weight: bold;
                    cursor: pointer;
                    line-height: 20px;
                ">&times;</span>

                <h2 style="color: #fff; margin-bottom: 20px; text-align: center;">
                    <i class="fas fa-brain"></i> Análisis con Inteligencia Artificial
                </h2>

                <div id="aiModalContentCustom" style="color: #fff; text-align: center; padding: 40px;">
                    <div class="spinner" style="
                        width: 60px;
                        height: 60px;
                        border: 4px solid rgba(255,255,255,0.3);
                        border-top-color: #8b5cf6;
                        border-radius: 50%;
                        animation: spin 1s linear infinite;
                        margin: 0 auto 20px;
                    "></div>
                    <h3 style="color: #8b5cf6; margin-bottom: 10px;">Generando análisis con IA...</h3>
                    <p style="color: #aaa;">El modelo de IA está analizando los datos médicos. Esto puede tomar unos segundos.</p>
                </div>
            </div>
        </div>
    `;

    document.body.insertAdjacentHTML('beforeend', modalHTML);
    console.log('Modal HTML insertado directamente en body');
}

function showAIModal(data) {
    console.log('Mostrando resultado de IA...');

    const modalContent = document.getElementById('aiModalContentCustom');

    if (!modalContent) {
        console.error('No se encontró el contenedor del modal custom');
        return;
    }

    const summary = data.analysis_summary;
    const explanation = data.ai_explanation;

    modalContent.innerHTML = `
        <div style="background: #222; padding: 25px; border-radius: 12px; margin-bottom: 25px;">
            <h4 style="color: #8b5cf6; margin-bottom: 20px; font-size: 1.3rem;">
                <i class="fas fa-chart-pie"></i> Resumen del Análisis
            </h4>
            <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(250px, 1fr)); gap: 15px;">
                <div class="summary-item">
                    <span class="summary-label">Total de Pacientes:</span>
                    <span class="summary-value">${summary.total_records}</span>
                </div>
                <div class="summary-item">
                    <span class="summary-label">Casos de Hipertensión:</span>
                    <span class="summary-value">${summary.hypertension_cases} (${((summary.hypertension_cases/summary.total_records)*100).toFixed(1)}%)</span>
                </div>
                <div class="summary-item">
                    <span class="summary-label">Casos de Diabetes:</span>
                    <span class="summary-value">${summary.diabetes_cases} (${((summary.diabetes_cases/summary.total_records)*100).toFixed(1)}%)</span>
                </div>
                <div class="summary-item">
                    <span class="summary-label">Precisión del Modelo:</span>
                    <span class="summary-value">${(summary.accuracy * 100).toFixed(1)}%</span>
                </div>
                <div class="summary-item">
                    <span class="summary-label">Edad Promedio:</span>
                    <span class="summary-value">${summary.avg_age} años</span>
                </div>
                <div class="summary-item">
                    <span class="summary-label">IMC Promedio:</span>
                    <span class="summary-value">${summary.avg_bmi} kg/m²</span>
                </div>
                <div class="summary-item">
                    <span class="summary-label">Glucosa Promedio:</span>
                    <span class="summary-value">${summary.avg_glucose} mg/dL</span>
                </div>
                <div class="summary-item">
                    <span class="summary-label">Presión Sistólica Promedio:</span>
                    <span class="summary-value">${summary.avg_systolic} mmHg</span>
                </div>
                <div class="summary-item">
                    <span class="summary-label">Fumadores:</span>
                    <span class="summary-value">${summary.smokers} (${((summary.smokers/summary.total_records)*100).toFixed(1)}%)</span>
                </div>
            </div>
        </div>

        <div style="background: #222; padding: 25px; border-radius: 12px;">
            <h4 style="color: #22c55e; margin-bottom: 20px; font-size: 1.3rem;">
                <i class="fas fa-robot"></i> Explicación del Especialista en IA
            </h4>
            <div id="aiExplanationText" style="
                background: #1a1a1a;
                padding: 20px;
                border-radius: 8px;
                line-height: 1.8;
                margin-bottom: 15px;
                max-height: 400px;
                overflow-y: auto;
                color: #fff;
                white-space: pre-wrap;
            "></div>
            <div style="text-align: center; padding-top: 15px; border-top: 1px solid #333;">
                <small style="color: #888;"><i class="fas fa-info-circle"></i> Generado por: ${data.model_used}</small>
            </div>
        </div>
    `;

    console.log('Contenido del modal actualizado');

    // Efecto de escritura para la explicación
    const explanationDiv = document.getElementById('aiExplanationText');
    if (explanationDiv) {
        typeWriterEffect(explanationDiv, explanation, 10);
    }
}

function hideAIModalCustom() {
    const modal = document.getElementById('aiModalCustom');
    if (modal) {
        modal.remove();
    }
}

// Efecto de escritura tipo máquina de escribir
function typeWriterEffect(element, text, speed = 20) {
    element.innerHTML = '<span class="typing-cursor">|</span>';
    let i = 0;

    const type = () => {
        if (i < text.length) {
            const currentChar = text.charAt(i);
            const currentText = element.innerHTML.replace('<span class="typing-cursor">|</span>', '');
            element.innerHTML = currentText + currentChar + '<span class="typing-cursor">|</span>';
            i++;
            setTimeout(type, speed);
        } else {
            // Remover cursor al finalizar
            element.innerHTML = element.innerHTML.replace('<span class="typing-cursor">|</span>', '');

            // Formatear texto con párrafos
            const formattedText = text.split('\n\n').map(para =>
                para.trim() ? `<p>${para.trim().replace(/\n/g, '<br>')}</p>` : ''
            ).join('');
            element.innerHTML = formattedText;
        }
    };

    type();
}

function hideAIModal() {
    // Intentar cerrar ambos modales (el original y el custom)
    const modal = document.getElementById('aiModal');
    if (modal) {
        modal.style.display = 'none';
    }
    hideAIModalCustom();
}

// VER ANÁLISIS DETALLADO
function viewDetailedAnalysis(fileId) {
    // Abrir en nueva pestaña el análisis educativo paso a paso
    const url = `${API_BASE_URL}/detailed-analysis/${fileId}`;
    window.open(url, '_blank');
    showToast('Abriendo análisis detallado en nueva pestaña...', 'info');
}

// FUNCIONES GLOBALES
window.analyzeFile = analyzeFile;
window.downloadReport = downloadReport;
window.deleteFile = deleteFile;
window.viewCharts = viewCharts;
window.analyzeWithAI = analyzeWithAI;
window.hideAIModalCustom = hideAIModalCustom;
window.viewDetailedAnalysis = viewDetailedAnalysis;