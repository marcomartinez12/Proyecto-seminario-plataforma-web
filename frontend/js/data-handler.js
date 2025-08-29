// Manejo de datos de pacientes

// Almacenamiento local de datos
let patientsData = JSON.parse(localStorage.getItem('patientsData')) || [];

// Inicialización cuando el DOM está listo
document.addEventListener('DOMContentLoaded', function() {
    initializeDataHandler();
});

function initializeDataHandler() {
    const form = document.getElementById('patientForm');
    if (form) {
        setupFormEvents();
        displaySavedData();
        calculateBMI();
    }
}

// Configurar eventos del formulario
function setupFormEvents() {
    const form = document.getElementById('patientForm');
    const weightInput = document.getElementById('weight');
    const heightInput = document.getElementById('height');
    
    // Evento de envío del formulario
    form.addEventListener('submit', handleFormSubmit);
    
    // Cálculo automático del IMC
    weightInput.addEventListener('input', calculateBMI);
    heightInput.addEventListener('input', calculateBMI);
    
    // Validación en tiempo real
    setupRealTimeValidation();
}

// Manejar envío del formulario
function handleFormSubmit(event) {
    event.preventDefault();
    
    if (validateForm()) {
        const formData = collectFormData();
        savePatientData(formData);
        displaySavedData();
        form.reset();
        Utils.showAlert('Datos del paciente guardados exitosamente', 'success');
    }
}

// Recopilar datos del formulario
function collectFormData() {
    const form = document.getElementById('patientForm');
    const formData = new FormData(form);
    const data = {};
    
    for (let [key, value] of formData.entries()) {
        data[key] = value;
    }
    
    // Agregar timestamp y ID único
    data.id = Utils.generateId();
    data.timestamp = new Date().toISOString();
    
    return data;
}

// Guardar datos del paciente
function savePatientData(data) {
    patientsData.push(data);
    localStorage.setItem('patientsData', JSON.stringify(patientsData));
}

// Mostrar datos guardados
function displaySavedData() {
    const container = document.getElementById('savedDataContainer');
    const tableBody = document.getElementById('patientsTableBody');
    
    if (patientsData.length > 0) {
        container.style.display = 'block';
        tableBody.innerHTML = '';
        
        patientsData.forEach((patient, index) => {
            const row = createPatientRow(patient, index);
            tableBody.appendChild(row);
        });
    } else {
        container.style.display = 'none';
    }
}

// Crear fila de paciente en la tabla
function createPatientRow(patient, index) {
    const row = document.createElement('tr');
    row.innerHTML = `
        <td>${patient.patientId}</td>
        <td>${patient.age}</td>
        <td>${patient.gender}</td>
        <td>${patient.systolicBP}</td>
        <td>${patient.diastolicBP}</td>
        <td>${patient.glucoseLevel}</td>
        <td>${Utils.formatDate(patient.visitDate)}</td>
        <td>
            <button onclick="editPatient(${index})" class="btn btn-sm">Editar</button>
            <button onclick="deletePatient(${index})" class="btn btn-sm btn-danger">Eliminar</button>
        </td>
    `;
    return row;
}

// Calcular IMC automáticamente
function calculateBMI() {
    const weight = parseFloat(document.getElementById('weight').value);
    const height = parseFloat(document.getElementById('height').value);
    const bmiField = document.getElementById('bmi');
    
    if (weight && height) {
        const heightInMeters = height / 100;
        const bmi = weight / (heightInMeters * heightInMeters);
        bmiField.value = bmi.toFixed(1);
        
        // Clasificación del IMC
        let classification = '';
        if (bmi < 18.5) classification = ' (Bajo peso)';
        else if (bmi < 25) classification = ' (Normal)';
        else if (bmi < 30) classification = ' (Sobrepeso)';
        else classification = ' (Obesidad)';
        
        bmiField.value += classification;
    } else {
        bmiField.value = '';
    }
}

// Validación del formulario
function validateForm() {
    const requiredFields = ['patientId', 'age', 'gender', 'systolicBP', 'diastolicBP', 'glucoseLevel', 'visitDate'];
    let isValid = true;
    
    requiredFields.forEach(fieldName => {
        const field = document.getElementById(fieldName);
        if (!field.value.trim()) {
            field.style.borderColor = '#dc3545';
            isValid = false;
        } else {
            field.style.borderColor = '#e1e5e9';
        }
    });
    
    if (!isValid) {
        Utils.showAlert('Por favor complete todos los campos obligatorios', 'error');
    }
    
    return isValid;
}

// Validación en tiempo real
function setupRealTimeValidation() {
    const inputs = document.querySelectorAll('.form-input, .form-select');
    
    inputs.forEach(input => {
        input.addEventListener('blur', function() {
            if (this.hasAttribute('required') && !this.value.trim()) {
                this.style.borderColor = '#dc3545';
            } else {
                this.style.borderColor = '#e1e5e9';
            }
        });
    });
}

// Editar paciente
function editPatient(index) {
    const patient = patientsData[index];
    
    // Llenar el formulario con los datos del paciente
    Object.keys(patient).forEach(key => {
        const field = document.getElementById(key);
        if (field) {
            field.value = patient[key];
        }
    });
    
    // Eliminar el registro actual para evitar duplicados
    deletePatient(index);
    
    // Scroll al formulario
    document.getElementById('patientForm').scrollIntoView({ behavior: 'smooth' });
}

// Eliminar paciente
function deletePatient(index) {
    if (confirm('¿Está seguro de que desea eliminar este registro?')) {
        patientsData.splice(index, 1);
        localStorage.setItem('patientsData', JSON.stringify(patientsData));
        displaySavedData();
        Utils.showAlert('Registro eliminado exitosamente', 'info');
    }
}

// Exportar datos para uso en otras páginas
function getPatientData() {
    return patientsData;
}

// Limpiar todos los datos
function clearAllData() {
    if (confirm('¿Está seguro de que desea eliminar todos los datos? Esta acción no se puede deshacer.')) {
        patientsData = [];
        localStorage.removeItem('patientsData');
        displaySavedData();
        Utils.showAlert('Todos los datos han sido eliminados', 'info');
    }
}

// Exportar funciones para uso global
window.editPatient = editPatient;
window.deletePatient = deletePatient;
window.getPatientData = getPatientData;
window.clearAllData = clearAllData;