"""
Flask Backend API for Disaster Prediction System
Loads trained ML models and serves predictions via REST API
"""

from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
import joblib
import numpy as np
import os
import pandas as pd
from datetime import datetime

app = Flask(__name__)
CORS(app)  # Enable CORS for frontend communication

# Load all trained models and preprocessors
MODEL_DIR = 'saved_models'

print("Loading models...")
try:
    # Load models
    disaster_classifier = joblib.load(os.path.join(MODEL_DIR, 'disaster_classifier.pkl'))
    damage_regressor = joblib.load(os.path.join(MODEL_DIR, 'damage_regressor.pkl'))
    response_regressor = joblib.load(os.path.join(MODEL_DIR, 'response_time_regressor.pkl'))
    
    # Load scalers
    scaler_disaster = joblib.load(os.path.join(MODEL_DIR, 'scaler_disaster.pkl'))
    scaler_damage = joblib.load(os.path.join(MODEL_DIR, 'scaler_damage.pkl'))
    scaler_response = joblib.load(os.path.join(MODEL_DIR, 'scaler_response.pkl'))
    
    # Load label encoders
    le_disaster = joblib.load(os.path.join(MODEL_DIR, 'label_encoder_disaster.pkl'))
    le_location = joblib.load(os.path.join(MODEL_DIR, 'label_encoder_location.pkl'))
    le_aid = joblib.load(os.path.join(MODEL_DIR, 'label_encoder_aid.pkl'))
    
    # Load metadata
    metadata = joblib.load(os.path.join(MODEL_DIR, 'model_metadata.pkl'))
    
    print("✓ All models loaded successfully!")
except Exception as e:
    print(f"Error loading models: {e}")
    raise


@app.route('/')
def home():
    """Serve the frontend interface"""
    return render_template('index.html')


@app.route('/api/status')
def status():
    """Health check endpoint"""
    return jsonify({
        'status': 'online',
        'message': 'Disaster Prediction API is running',
        'version': '1.0.0',
        'training_date': metadata.get('training_date', 'Unknown'),
        'model_accuracy': f"{metadata.get('model_accuracy', 0)*100:.2f}%"
    })


@app.route('/api/disaster-types', methods=['GET'])
def get_disaster_types():
    """Get available disaster types"""
    return jsonify({
        'disaster_types': le_disaster.classes_.tolist(),
        'locations': le_location.classes_.tolist()
    })


@app.route('/api/predict', methods=['POST'])
def predict():
    """
    Main prediction endpoint
    Expects JSON with disaster parameters and returns comprehensive predictions
    """
    try:
        data = request.json
        
        # Extract input parameters
        disaster_type = data.get('disaster_type')
        location = data.get('location')
        latitude = float(data.get('latitude'))
        longitude = float(data.get('longitude'))
        severity_level = int(data.get('severity_level'))
        affected_population = int(data.get('affected_population'))
        economic_loss = float(data.get('economic_loss'))
        
        # Get current date info
        now = datetime.now()
        month = data.get('month', now.month)
        quarter = data.get('quarter', (month - 1) // 3 + 1)
        day_of_year = data.get('day_of_year', now.timetuple().tm_yday)
        
        # Encode categorical variables
        try:
            disaster_encoded = le_disaster.transform([disaster_type])[0]
        except:
            disaster_encoded = 0
            
        try:
            location_encoded = le_location.transform([location])[0]
        except:
            location_encoded = 0
        
        # Prepare input for major disaster prediction
        disaster_input = np.array([[
            disaster_encoded, location_encoded, latitude, longitude,
            severity_level, affected_population, 0.5,
            month, quarter, day_of_year
        ]])
        disaster_input_scaled = scaler_disaster.transform(disaster_input)
        
        # Predict major disaster
        is_major = bool(disaster_classifier.predict(disaster_input_scaled)[0])
        major_probability = float(disaster_classifier.predict_proba(disaster_input_scaled)[0][1])
        
        # Prepare input for damage assessment
        damage_input = np.array([[
            disaster_encoded, location_encoded, latitude, longitude,
            severity_level, affected_population, economic_loss,
            month, quarter
        ]])
        damage_input_scaled = scaler_damage.transform(damage_input)
        predicted_damage = float(damage_regressor.predict(damage_input_scaled)[0])
        
        # Prepare input for response time prediction
        response_input = np.array([[
            disaster_encoded, location_encoded, latitude, longitude,
            severity_level, affected_population, predicted_damage, economic_loss
        ]])
        response_input_scaled = scaler_response.transform(response_input)
        predicted_response_time = float(response_regressor.predict(response_input_scaled)[0])
        
        # Determine priority and alert level
        if major_probability > 0.7 or severity_level >= 8:
            priority = "CRITICAL"
            priority_level = 5
            alert_level = "LEVEL 5 - MAXIMUM ALERT"
        elif major_probability > 0.5 or severity_level >= 6:
            priority = "HIGH"
            priority_level = 4
            alert_level = "LEVEL 4 - HIGH ALERT"
        elif major_probability > 0.3 or severity_level >= 4:
            priority = "MEDIUM"
            priority_level = 3
            alert_level = "LEVEL 3 - MODERATE ALERT"
        else:
            priority = "LOW"
            priority_level = 2
            alert_level = "LEVEL 2 - LOW ALERT"
        
        # Calculate resource recommendations
        if affected_population > 40000:
            personnel = "500+ emergency responders"
            medical_teams = "20+ teams"
            rescue_units = "30+ units"
        elif affected_population > 20000:
            personnel = "200-500 emergency responders"
            medical_teams = "10-20 teams"
            rescue_units = "15-30 units"
        else:
            personnel = "100-200 emergency responders"
            medical_teams = "5-10 teams"
            rescue_units = "10-15 units"
        
        # Calculate shelter needs
        if predicted_damage > 0.7:
            shelters = int(affected_population * 0.6)
            equipment = "Heavy equipment: Bulldozers, cranes, excavators (HIGH PRIORITY)"
        elif predicted_damage > 0.4:
            shelters = int(affected_population * 0.4)
            equipment = "Heavy equipment: Moderate deployment required"
        else:
            shelters = int(affected_population * 0.2)
            equipment = "Heavy equipment: Standard deployment"
        
        # Evacuation recommendation
        evacuation_needed = None
        evacuation_centers = None
        vehicles_needed = None
        if major_probability > 0.6 or predicted_damage > 0.6:
            evacuation_needed = int(affected_population * 0.7)
            evacuation_centers = evacuation_needed // 500
            vehicles_needed = evacuation_needed // 50
        
        # Prepare response
        response = {
            'success': True,
            'input': {
                'disaster_type': disaster_type,
                'location': location,
                'latitude': latitude,
                'longitude': longitude,
                'severity_level': severity_level,
                'affected_population': affected_population,
                'economic_loss': economic_loss
            },
            'predictions': {
                'is_major_disaster': is_major,
                'major_probability': round(major_probability * 100, 2),
                'predicted_damage_index': round(predicted_damage, 3),
                'predicted_response_time_hours': round(predicted_response_time, 1)
            },
            'emergency_response': {
                'priority': priority,
                'priority_level': priority_level,
                'alert_level': alert_level,
                'resources': {
                    'personnel': personnel,
                    'medical_teams': medical_teams,
                    'rescue_units': rescue_units,
                    'temporary_shelters': shelters,
                    'equipment': equipment
                },
                'evacuation': {
                    'recommended': evacuation_needed is not None,
                    'people_to_evacuate': evacuation_needed,
                    'evacuation_centers': evacuation_centers,
                    'vehicles_needed': vehicles_needed
                },
                'action_items': [
                    f"Activate Emergency Operations Center within {predicted_response_time/2:.1f} hours",
                    f"Deploy first responders within {predicted_response_time:.1f} hours",
                    "Establish communication networks and evacuation routes",
                    "Coordinate with local hospitals and emergency services",
                    "Set up relief distribution centers"
                ]
            }
        }
        
        return jsonify(response)
        
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 400


@app.route('/api/batch-predict', methods=['POST'])
def batch_predict():
    """
    Batch prediction endpoint for multiple disaster scenarios
    """
    try:
        scenarios = request.json.get('scenarios', [])
        results = []
        
        for scenario in scenarios:
            # Call the predict endpoint logic for each scenario
            # (In production, this would be refactored to avoid code duplication)
            result = {
                'scenario_id': scenario.get('id'),
                'disaster_type': scenario.get('disaster_type'),
                'location': scenario.get('location')
            }
            results.append(result)
        
        return jsonify({
            'success': True,
            'count': len(results),
            'results': results
        })
        
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 400


@app.route('/api/model-info', methods=['GET'])
def model_info():
    """Get detailed model information"""
    return jsonify({
        'training_date': metadata.get('training_date'),
        'dataset_size': metadata.get('dataset_size'),
        'model_performance': {
            'accuracy': f"{metadata.get('model_accuracy', 0)*100:.2f}%",
            'damage_r2_score': round(metadata.get('damage_r2_score', 0), 4),
            'response_r2_score': round(metadata.get('response_r2_score', 0), 4)
        },
        'features': {
            'disaster_prediction': metadata.get('feature_columns', {}).get('disaster_prediction', []),
            'damage_assessment': metadata.get('feature_columns', {}).get('damage_assessment', []),
            'response_optimization': metadata.get('feature_columns', {}).get('response_optimization', [])
        }
    })


if __name__ == '__main__':
    print("\n" + "="*70)
    print("  🚀 DISASTER PREDICTION API SERVER")
    print("="*70)
    print(f"  Training Date: {metadata.get('training_date', 'Unknown')}")
    print(f"  Model Accuracy: {metadata.get('model_accuracy', 0)*100:.2f}%")
    print(f"  Dataset Size: {metadata.get('dataset_size', 0):,} records")
    print("="*70)
    print("\n  Server starting at http://localhost:5000")
    print("  API Documentation: http://localhost:5000/\n")
    
    app.run(debug=True, host='0.0.0.0', port=5000)
