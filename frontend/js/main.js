// Funcionalidad principal de la aplicación

// Inicialización cuando el DOM está listo
document.addEventListener('DOMContentLoaded', function() {
    initializeApp();
});

function initializeApp() {
    console.log('Plataforma de Tendencias Médicas - Inicializada');
    
    // Configurar navegación activa
    setActiveNavigation();
    
    // Configurar eventos globales
    setupGlobalEvents();
}

// Configurar navegación activa
function setActiveNavigation() {
    const currentPage = window.location.pathname.split('/').pop() || 'index.html';
    const navLinks = document.querySelectorAll('.nav-link');
    
    navLinks.forEach(link => {
        link.classList.remove('active');
        if (link.getAttribute('href').includes(currentPage)) {
            link.classList.add('active');
        }
    });
}

// Configurar eventos globales
function setupGlobalEvents() {
    // Smooth scrolling para enlaces internos
    document.querySelectorAll('a[href^="#"]').forEach(anchor => {
        anchor.addEventListener('click', function (e) {
            e.preventDefault();
            const target = document.querySelector(this.getAttribute('href'));
            if (target) {
                target.scrollIntoView({
                    behavior: 'smooth'
                });
            }
        });
    });
}

// Utilidades generales
const Utils = {
    // Mostrar loading
    showLoading: function(container) {
        const loadingHTML = `
            <div class="loading">
                <div class="spinner"></div>
            </div>
        `;
        container.innerHTML = loadingHTML;
    },
    
    // Mostrar alerta
    showAlert: function(message, type = 'info') {
        const alertHTML = `
            <div class="alert alert-${type}">
                ${message}
            </div>
        `;
        
        const alertContainer = document.createElement('div');
        alertContainer.innerHTML = alertHTML;
        
        document.body.insertBefore(alertContainer.firstElementChild, document.body.firstChild);
        
        // Auto-remove después de 5 segundos
        setTimeout(() => {
            const alert = document.querySelector('.alert');
            if (alert) {
                alert.remove();
            }
        }, 5000);
    },
    
    // Validar email
    validateEmail: function(email) {
        const re = /^[^\s@]+@[^\s@]+\.[^\s@]+$/;
        return re.test(email);
    },
    
    // Formatear fecha
    formatDate: function(date) {
        return new Date(date).toLocaleDateString('es-ES');
    },
    
    // Generar ID único
    generateId: function() {
        return Date.now().toString(36) + Math.random().toString(36).substr(2);
    }
};

// Exportar para uso global
window.Utils = Utils;