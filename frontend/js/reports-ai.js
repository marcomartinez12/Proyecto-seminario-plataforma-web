class IntelligentReportsManager {
    constructor() {
        this.selectedFiles = [];
        this.reports = JSON.parse(localStorage.getItem('aiReports')) || [];
        this.isProcessing = false;
        this.init();
    }

    init() {
        this.setupEventListeners();
        this.loadExistingReports();
    }

    setupEventListeners() {
        // Elementos del DOM
        this.uploadZone = document.getElementById('aiUploadZone');
        this.fileInput = document.getElementById('aiFileInput');
        this.selectFilesBtn = document.getElementById('selectFilesBtn');
        this.startAnalysisBtn = document.getElementById('startAnalysisBtn');
        this.clearFilesBtn = document.getElementById('clearFilesBtn');
        this.selectedFilesCard = document.getElementById('selectedFilesCard');
        this.filesList = document.getElementById('filesList');
        this.progressModal = document.getElementById('progressModal');

        // Event listeners
        this.selectFilesBtn.addEventListener('click', () => this.fileInput.click());
        this.fileInput.addEventListener('change', (e) => this.handleFileSelection(e));
        this.startAnalysisBtn.addEventListener('click', () => this.startIntelligentAnalysis());
        this.clearFilesBtn.addEventListener('click', () => this.clearSelectedFiles());

        // Drag and drop
        this.uploadZone.addEventListener('dragover', (e) => this.handleDragOver(e));
        this.uploadZone.addEventListener('drop', (e) => this.handleDrop(e));
        this.uploadZone.addEventListener('dragleave', (e) => this.handleDragLeave(e));
    }

    handleFileSelection(event) {
        const files = Array.from(event.target.files);
        this.addFiles(files);
    }

    handleDragOver(event) {
        event.preventDefault();
        this.uploadZone.classList.add('drag-over');
    }

    handleDrop(event) {
        event.preventDefault();
        this.uploadZone.classList.remove('drag-over');
        const files = Array.from(event.dataTransfer.files);
        this.addFiles(files);
    }

    handleDragLeave(event) {
        event.preventDefault();
        this.uploadZone.classList.remove('drag-over');
    }

    addFiles(files) {
        const validFiles = files.filter(file => {
            const validTypes = ['application/vnd.openxmlformats-officedocument.spreadsheetml.sheet', 
                              'application/vnd.ms-excel', 'text/csv'];
            return validTypes.includes(file.type) || file.name.endsWith('.csv');
        });

        if (validFiles.length === 0) {
            this.showNotification('Por favor selecciona archivos Excel o CSV válidos', 'error');
            return;
        }

        this.selectedFiles = [...this.selectedFiles, ...validFiles];
        this.updateFilesDisplay();
        this.startAnalysisBtn.disabled = false;
    }

    updateFilesDisplay() {
        if (this.selectedFiles.length === 0) {
            this.selectedFilesCard.style.display = 'none';
            return;
        }

        this.selectedFilesCard.style.display = 'block';
        this.filesList.innerHTML = this.selectedFiles.map((file, index) => `
            <div class="file-item">
                <div class="file-info">
                    <i class="fas fa-file-excel file-icon"></i>
                    <div class="file-details">
                        <span class="file-name">${file.name}</span>
                        <span class="file-size">${this.formatFileSize(file.size)}</span>
                    </div>
                </div>
                <button class="btn btn-sm btn-danger" onclick="reportsManager.removeFile(${index})">
                    <i class="fas fa-times"></i>
                </button>
            </div>
        `).join('');
    }

    removeFile(index) {
        this.selectedFiles.splice(index, 1);
        this.updateFilesDisplay();
        if (this.selectedFiles.length === 0) {
            this.startAnalysisBtn.disabled = true;
        }
    }

    clearSelectedFiles() {
        this.selectedFiles = [];
        this.updateFilesDisplay();
        this.startAnalysisBtn.disabled = true;
        this.fileInput.value = '';
    }

    async startIntelligentAnalysis() {
        if (this.isProcessing || this.selectedFiles.length === 0) return;

        this.isProcessing = true;
        this.showProgressModal();

        try {
            // Simular proceso de análisis IA
            await this.simulateAIAnalysis();
            
            // Generar reporte
            const report = await this.generateIntelligentReport();
            
            // Guardar reporte
            this.saveReport(report);
            
            // Mostrar éxito
            this.hideProgressModal();
            this.showNotification('¡Reporte generado exitosamente!', 'success');
            
            // Limpiar archivos seleccionados
            this.clearSelectedFiles();
            
        } catch (error) {
            this.hideProgressModal();
            this.showNotification('Error al generar el reporte: ' + error.message, 'error');
        } finally {
            this.isProcessing = false;
        }
    }

    async simulateAIAnalysis() {
        const steps = [
            { text: 'Cargando y validando archivos...', progress: 20 },
            { text: 'Analizando datos con IA...', progress: 40 },
            { text: 'Detectando patrones y tendencias...', progress: 60 },
            { text: 'Generando gráficos inteligentes...', progress: 80 },
            { text: 'Creando reporte PDF...', progress: 100 }
        ];

        for (let i = 0; i < steps.length; i++) {
            await this.delay(1500);
            this.updateProgress(steps[i].text, steps[i].progress);
            this.updateStepStatus(i + 1);
        }
    }

    async generateIntelligentReport() {
        const analysisType = document.getElementById('analysisType').value;
        const reportFormat = document.getElementById('reportFormat').value;
        
        // Simular análisis de datos
        const mockData = this.generateMockAnalysis(analysisType);
        
        const report = {
            id: Date.now().toString(),
            timestamp: new Date().toISOString(),
            files: this.selectedFiles.map(f => f.name),
            analysisType: analysisType,
            reportFormat: reportFormat,
            status: 'completed',
            insights: mockData.insights,
            charts: mockData.charts,
            recommendations: mockData.recommendations,
            downloadUrl: this.generateMockPDFUrl()
        };

        return report;
    }

    generateMockAnalysis(type) {
        const analyses = {
            comprehensive: {
                insights: [
                    'Detectados 3 patrones de riesgo cardiovascular en la población analizada',
                    'Tendencia creciente de diabetes tipo 2 en pacientes de 45-60 años',
                    'Correlación significativa entre hipertensión y factores socioeconómicos'
                ],
                charts: ['Distribución por Edad', 'Prevalencia de Enfermedades', 'Tendencias Temporales'],
                recommendations: [
                    'Implementar programa de prevención cardiovascular',
                    'Reforzar controles en población de riesgo medio-alto',
                    'Desarrollar estrategias de intervención temprana'
                ]
            },
            trends: {
                insights: [
                    'Incremento del 15% en casos de hipertensión en los últimos 6 meses',
                    'Patrón estacional en consultas por diabetes',
                    'Mejora en adherencia al tratamiento tras implementación de seguimiento'
                ],
                charts: ['Tendencias Mensuales', 'Comparativa Anual', 'Proyecciones'],
                recommendations: [
                    'Mantener vigilancia epidemiológica activa',
                    'Ajustar recursos según patrones estacionales'
                ]
            },
            risk: {
                insights: [
                    '23% de pacientes en categoría de alto riesgo cardiovascular',
                    'Factores de riesgo modificables identificados en 67% de casos',
                    'Necesidad de intervención inmediata en 12 pacientes'
                ],
                charts: ['Matriz de Riesgo', 'Factores Contribuyentes', 'Priorización'],
                recommendations: [
                    'Priorizar atención en pacientes de alto riesgo',
                    'Implementar plan de modificación de factores de riesgo'
                ]
            },
            demographics: {
                insights: [
                    'Mayor prevalencia en grupo etario 50-65 años',
                    'Distribución equitativa por género con ligero predominio femenino',
                    'Concentración geográfica en zonas urbanas'
                ],
                charts: ['Pirámide Poblacional', 'Distribución Geográfica', 'Características Sociodemográficas'],
                recommendations: [
                    'Adaptar estrategias según perfil demográfico',
                    'Considerar factores socioeconómicos en intervenciones'
                ]
            }
        };

        return analyses[type] || analyses.comprehensive;
    }

    generateMockPDFUrl() {
        // En una implementación real, esto sería la URL del PDF generado
        return `data:application/pdf;base64,${btoa('Mock PDF Content')}`;
    }

    saveReport(report) {
        this.reports.unshift(report);
        localStorage.setItem('aiReports', JSON.stringify(this.reports));
        this.loadExistingReports();
    }

    loadExistingReports() {
        const container = document.getElementById('reportsContainer');
        const emptyState = document.getElementById('emptyReports');
        
        if (this.reports.length === 0) {
            emptyState.style.display = 'block';
            return;
        }
        
        emptyState.style.display = 'none';
        
        container.innerHTML = this.reports.map(report => `
            <div class="report-item">
                <div class="report-header">
                    <div class="report-info">
                        <h4><i class="fas fa-file-pdf"></i> Reporte ${this.getAnalysisTypeName(report.analysisType)}</h4>
                        <p class="report-date">${this.formatDate(report.timestamp)}</p>
                        <p class="report-files">Archivos: ${report.files.join(', ')}</p>
                    </div>
                    <div class="report-status">
                        <span class="status-badge status-${report.status}">
                            <i class="fas fa-check-circle"></i> Completado
                        </span>
                    </div>
                </div>
                
                <div class="report-content">
                    <div class="insights-section">
                        <h5><i class="fas fa-lightbulb"></i> Insights Principales:</h5>
                        <ul>
                            ${report.insights.map(insight => `<li>${insight}</li>`).join('')}
                        </ul>
                    </div>
                    
                    <div class="charts-section">
                        <h5><i class="fas fa-chart-bar"></i> Gráficos Incluidos:</h5>
                        <div class="charts-list">
                            ${report.charts.map(chart => `<span class="chart-tag">${chart}</span>`).join('')}
                        </div>
                    </div>
                    
                    <div class="recommendations-section">
                        <h5><i class="fas fa-recommendations"></i> Recomendaciones:</h5>
                        <ul>
                            ${report.recommendations.map(rec => `<li>${rec}</li>`).join('')}
                        </ul>
                    </div>
                </div>
                
                <div class="report-actions">
                    <button class="btn btn-primary" onclick="reportsManager.downloadReport('${report.id}')">
                        <i class="fas fa-download"></i> Descargar PDF
                    </button>
                    <button class="btn btn-secondary" onclick="reportsManager.viewReport('${report.id}')">
                        <i class="fas fa-eye"></i> Vista Previa
                    </button>
                    <button class="btn btn-danger" onclick="reportsManager.deleteReport('${report.id}')">
                        <i class="fas fa-trash"></i> Eliminar
                    </button>
                </div>
            </div>
        `).join('');
    }

    downloadReport(reportId) {
        const report = this.reports.find(r => r.id === reportId);
        if (!report) return;
        
        // Simular descarga de PDF
        const link = document.createElement('a');
        link.href = report.downloadUrl;
        link.download = `reporte_ia_${report.analysisType}_${report.id}.pdf`;
        link.click();
        
        this.showNotification('Descarga iniciada', 'success');
    }

    viewReport(reportId) {
        const report = this.reports.find(r => r.id === reportId);
        if (!report) return;
        
        // Abrir en nueva ventana (simulado)
        window.open(report.downloadUrl, '_blank');
    }

    deleteReport(reportId) {
        if (confirm('¿Estás seguro de que quieres eliminar este reporte?')) {
            this.reports = this.reports.filter(r => r.id !== reportId);
            localStorage.setItem('aiReports', JSON.stringify(this.reports));
            this.loadExistingReports();
            this.showNotification('Reporte eliminado', 'success');
        }
    }

    showProgressModal() {
        this.progressModal.style.display = 'flex';
        this.updateProgress('Iniciando análisis...', 0);
        this.resetSteps();
    }

    hideProgressModal() {
        this.progressModal.style.display = 'none';
    }

    updateProgress(text, percentage) {
        document.getElementById('progressText').textContent = text;
        document.getElementById('progressFill').style.width = percentage + '%';
        document.getElementById('progressPercentage').textContent = percentage + '%';
    }

    updateStepStatus(stepNumber) {
        const step = document.getElementById(`step${stepNumber}`);
        if (step) {
            step.classList.add('completed');
        }
    }

    resetSteps() {
        for (let i = 1; i <= 4; i++) {
            const step = document.getElementById(`step${i}`);
            if (step) {
                step.classList.remove('completed');
            }
        }
    }

    getAnalysisTypeName(type) {
        const names = {
            comprehensive: 'Completo',
            trends: 'Tendencias',
            risk: 'Riesgos',
            demographics: 'Demográfico'
        };
        return names[type] || 'Personalizado';
    }

    formatDate(timestamp) {
        return new Date(timestamp).toLocaleString('es-ES', {
            year: 'numeric',
            month: 'long',
            day: 'numeric',
            hour: '2-digit',
            minute: '2-digit'
        });
    }

    formatFileSize(bytes) {
        if (bytes === 0) return '0 Bytes';
        const k = 1024;
        const sizes = ['Bytes', 'KB', 'MB', 'GB'];
        const i = Math.floor(Math.log(bytes) / Math.log(k));
        return parseFloat((bytes / Math.pow(k, i)).toFixed(2)) + ' ' + sizes[i];
    }

    showNotification(message, type = 'info') {
        // Crear notificación temporal
        const notification = document.createElement('div');
        notification.className = `notification notification-${type}`;
        notification.innerHTML = `
            <i class="fas fa-${type === 'success' ? 'check-circle' : type === 'error' ? 'exclamation-circle' : 'info-circle'}"></i>
            <span>${message}</span>
        `;
        
        document.body.appendChild(notification);
        
        setTimeout(() => {
            notification.classList.add('show');
        }, 100);
        
        setTimeout(() => {
            notification.classList.remove('show');
            setTimeout(() => {
                document.body.removeChild(notification);
            }, 300);
        }, 3000);
    }

    delay(ms) {
        return new Promise(resolve => setTimeout(resolve, ms));
    }
}

// Inicializar cuando el DOM esté listo
document.addEventListener('DOMContentLoaded', () => {
    window.reportsManager = new IntelligentReportsManager();
});