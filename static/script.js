// API Configuration
const API_BASE_URL = 'http://localhost:5000';

// Initialize app
document.addEventListener('DOMContentLoaded', async () => {
    await loadDisasterTypes();
    await loadModelInfo();
    setupEventListeners();
});

// Load disaster types and locations
async function loadDisasterTypes() {
    try {
        const response = await fetch(`${API_BASE_URL}/api/disaster-types`);
        const data = await response.json();
        
        // Populate disaster types
        const disasterSelect = document.getElementById('disasterType');
        data.disaster_types.forEach(type => {
            const option = document.createElement('option');
            option.value = type;
            option.textContent = type;
            disasterSelect.appendChild(option);
        });
        
        // Populate locations
        const locationSelect = document.getElementById('location');
        data.locations.forEach(location => {
            const option = document.createElement('option');
            option.value = location;
            option.textContent = location;
            locationSelect.appendChild(option);
        });
    } catch (error) {
        console.error('Error loading disaster types:', error);
        showError('Failed to load disaster types. Please refresh the page.');
    }
}

// Load model information
async function loadModelInfo() {
    try {
        const response = await fetch(`${API_BASE_URL}/api/model-info`);
        const data = await response.json();
        
        document.getElementById('modelInfo').textContent = 
            `Trained: ${data.training_date} | Accuracy: ${data.model_performance.accuracy} | Dataset: ${data.dataset_size.toLocaleString()} records`;
    } catch (error) {
        console.error('Error loading model info:', error);
    }
}

// Setup event listeners
function setupEventListeners() {
    // Severity level slider
    const severitySlider = document.getElementById('severityLevel');
    const severityValue = document.getElementById('severityValue');
    
    severitySlider.addEventListener('input', (e) => {
        severityValue.textContent = e.target.value;
        severityValue.style.color = getSeverityColor(parseInt(e.target.value));
    });
    
    // Form submission
    const form = document.getElementById('predictionForm');
    form.addEventListener('submit', handleFormSubmit);
}

// Get color based on severity
function getSeverityColor(severity) {
    if (severity >= 8) return '#ef4444';
    if (severity >= 6) return '#f59e0b';
    if (severity >= 4) return '#eab308';
    return '#10b981';
}

// Handle form submission
async function handleFormSubmit(e) {
    e.preventDefault();
    
    // Show loading state
    document.getElementById('loadingSpinner').style.display = 'block';
    document.getElementById('resultsSection').style.display = 'none';
    
    // Collect form data
    const formData = {
        disaster_type: document.getElementById('disasterType').value,
        location: document.getElementById('location').value,
        latitude: parseFloat(document.getElementById('latitude').value),
        longitude: parseFloat(document.getElementById('longitude').value),
        severity_level: parseInt(document.getElementById('severityLevel').value),
        affected_population: parseInt(document.getElementById('affectedPopulation').value),
        economic_loss: parseFloat(document.getElementById('economicLoss').value)
    };
    
    try {
        const response = await fetch(`${API_BASE_URL}/api/predict`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify(formData)
        });
        
        if (!response.ok) {
            throw new Error('Prediction request failed');
        }
        
        const result = await response.json();
        
        // Hide loading, show results
        document.getElementById('loadingSpinner').style.display = 'none';
        displayResults(result);
        
    } catch (error) {
        console.error('Error making prediction:', error);
        document.getElementById('loadingSpinner').style.display = 'none';
        showError('Failed to generate prediction. Please check your inputs and try again.');
    }
}

// Display prediction results
function displayResults(result) {
    const resultsSection = document.getElementById('resultsSection');
    resultsSection.style.display = 'block';
    
    // Scroll to results
    resultsSection.scrollIntoView({ behavior: 'smooth', block: 'start' });
    
    const { predictions, emergency_response } = result;
    
    // Prediction Results
    document.getElementById('isMajor').textContent = 
        predictions.is_major_disaster ? 'YES ⚠️' : 'NO ✓';
    document.getElementById('isMajor').style.color = 
        predictions.is_major_disaster ? '#ef4444' : '#10b981';
    
    document.getElementById('majorProb').textContent = 
        `${predictions.major_probability}%`;
    
    document.getElementById('damageIndex').textContent = 
        predictions.predicted_damage_index.toFixed(3);
    
    document.getElementById('responseTime').textContent = 
        `${predictions.predicted_response_time_hours} hours`;
    
    // Emergency Response
    const alertBanner = document.getElementById('alertBanner');
    alertBanner.className = 'alert-banner';
    
    // Set priority class
    if (emergency_response.priority === 'CRITICAL') {
        alertBanner.classList.add('priority-critical');
    } else if (emergency_response.priority === 'HIGH') {
        alertBanner.classList.add('priority-high');
    } else if (emergency_response.priority === 'MEDIUM') {
        alertBanner.classList.add('priority-medium');
    } else {
        alertBanner.classList.add('priority-low');
    }
    
    document.getElementById('alertPriority').textContent = 
        `${getPriorityEmoji(emergency_response.priority)} ${emergency_response.priority} PRIORITY`;
    document.getElementById('alertLevel').textContent = emergency_response.alert_level;
    
    // Resources
    const { resources } = emergency_response;
    document.getElementById('personnel').textContent = resources.personnel;
    document.getElementById('medicalTeams').textContent = resources.medical_teams;
    document.getElementById('rescueUnits').textContent = resources.rescue_units;
    document.getElementById('shelters').textContent = 
        `${resources.temporary_shelters.toLocaleString()} units`;
    document.getElementById('equipment').textContent = resources.equipment;
    
    // Evacuation
    if (emergency_response.evacuation.recommended) {
        const evacuationSection = document.getElementById('evacuationSection');
        evacuationSection.style.display = 'block';
        
        document.getElementById('evacuationPeople').textContent = 
            emergency_response.evacuation.people_to_evacuate.toLocaleString();
        document.getElementById('evacuationCenters').textContent = 
            emergency_response.evacuation.evacuation_centers;
        document.getElementById('evacuationVehicles').textContent = 
            emergency_response.evacuation.vehicles_needed;
    } else {
        document.getElementById('evacuationSection').style.display = 'none';
    }
    
    // Action Items
    const actionList = document.getElementById('actionItemsList');
    actionList.innerHTML = '';
    emergency_response.action_items.forEach(item => {
        const li = document.createElement('li');
        li.textContent = item;
        actionList.appendChild(li);
    });
}

// Get priority emoji
function getPriorityEmoji(priority) {
    switch (priority) {
        case 'CRITICAL': return '🔴';
        case 'HIGH': return '🟠';
        case 'MEDIUM': return '🟡';
        case 'LOW': return '🟢';
        default: return '⚪';
    }
}

// Show error message
function showError(message) {
    alert(message); // In production, use a better error UI
}

// Format number with commas
function formatNumber(num) {
    return num.toString().replace(/\B(?=(\d{3})+(?!\d))/g, ',');
}
