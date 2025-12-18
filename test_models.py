import joblib
import numpy as np
import os

MODEL_DIR = 'saved_models'

print("Testing all models...\n")

# Test loading all models
print("1. Loading parameter prediction models...")
try:
    severity_model = joblib.load(os.path.join(MODEL_DIR, 'severity_model.pkl'))
    population_model = joblib.load(os.path.join(MODEL_DIR, 'population_model.pkl'))
    economic_loss_model = joblib.load(os.path.join(MODEL_DIR, 'economic_loss_model.pkl'))
    scaler_parameters = joblib.load(os.path.join(MODEL_DIR, 'scaler_parameters.pkl'))
    print("   ✓ Parameter models loaded")
except Exception as e:
    print(f"   ✗ Error: {e}")
    exit(1)

print("\n2. Loading main prediction models...")
try:
    disaster_classifier = joblib.load(os.path.join(MODEL_DIR, 'disaster_classifier.pkl'))
    damage_regressor = joblib.load(os.path.join(MODEL_DIR, 'damage_regressor.pkl'))
    response_regressor = joblib.load(os.path.join(MODEL_DIR, 'response_time_regressor.pkl'))
    print("   ✓ Main models loaded")
except Exception as e:
    print(f"   ✗ Error: {e}")
    exit(1)

print("\n3. Loading scalers...")
try:
    scaler_disaster = joblib.load(os.path.join(MODEL_DIR, 'scaler_disaster.pkl'))
    scaler_damage = joblib.load(os.path.join(MODEL_DIR, 'scaler_damage.pkl'))
    scaler_response = joblib.load(os.path.join(MODEL_DIR, 'scaler_response.pkl'))
    print("   ✓ Scalers loaded")
except Exception as e:
    print(f"   ✗ Error: {e}")
    exit(1)

print("\n4. Loading encoders...")
try:
    le_disaster = joblib.load(os.path.join(MODEL_DIR, 'label_encoder_disaster.pkl'))
    le_location = joblib.load(os.path.join(MODEL_DIR, 'label_encoder_location.pkl'))
    print("   ✓ Encoders loaded")
    print(f"   Disaster types: {le_disaster.classes_[:5]}... (total: {len(le_disaster.classes_)})")
    print(f"   Locations: {le_location.classes_[:5]}... (total: {len(le_location.classes_)})")
except Exception as e:
    print(f"   ✗ Error: {e}")
    exit(1)

print("\n5. Testing prediction pipeline...")
try:
    # Simulate form inputs
    disaster_type = "Earthquake"
    location = "Japan"
    latitude = 35.6762
    longitude = 139.6503
    month = 7
    week = 2
    
    # Encode
    disaster_encoded = le_disaster.transform([disaster_type])[0]
    location_encoded = le_location.transform([location])[0]
    
    # Calculate temporal features
    quarter = (month - 1) // 3 + 1
    is_summer = 1 if month in [6, 7, 8] else 0
    is_winter = 1 if month in [12, 1, 2] else 0
    
    print(f"\n   Input: {disaster_type} in {location}")
    print(f"   Coordinates: ({latitude}, {longitude})")
    print(f"   Time: Month {month}, Week {week}")
    
    # Step 1: Predict parameters
    param_features = np.array([[
        disaster_encoded, location_encoded,
        latitude, longitude,
        month, week, quarter, is_summer, is_winter
    ]])
    param_features_scaled = scaler_parameters.transform(param_features)
    
    severity_level = max(1, min(10, int(severity_model.predict(param_features_scaled)[0])))
    affected_population = max(0, int(population_model.predict(param_features_scaled)[0]))
    economic_loss = max(0, float(economic_loss_model.predict(param_features_scaled)[0]))
    
    print(f"\n   Predicted Parameters:")
    print(f"   - Severity: {severity_level}/10")
    print(f"   - Population: {affected_population:,}")
    print(f"   - Economic Loss: ${economic_loss:,.2f}")
    
    # Step 2: Major disaster prediction
    day_of_year = 182  # approximate
    disaster_input = np.array([[
        disaster_encoded, location_encoded, latitude, longitude,
        severity_level, affected_population, 0.5,
        month, quarter, day_of_year
    ]])
    disaster_input_scaled = scaler_disaster.transform(disaster_input)
    
    is_major = bool(disaster_classifier.predict(disaster_input_scaled)[0])
    major_probability = float(disaster_classifier.predict_proba(disaster_input_scaled)[0][1])
    
    print(f"\n   Major Disaster Prediction:")
    print(f"   - Is Major: {is_major}")
    print(f"   - Probability: {major_probability*100:.2f}%")
    
    # Step 3: Damage prediction
    damage_input = np.array([[
        disaster_encoded, location_encoded, latitude, longitude,
        severity_level, affected_population, economic_loss,
        month, quarter
    ]])
    damage_input_scaled = scaler_damage.transform(damage_input)
    predicted_damage = float(damage_regressor.predict(damage_input_scaled)[0])
    
    print(f"   - Damage Index: {predicted_damage:.3f}")
    
    # Step 4: Response time prediction
    response_input = np.array([[
        disaster_encoded, location_encoded, latitude, longitude,
        severity_level, affected_population, predicted_damage, economic_loss
    ]])
    response_input_scaled = scaler_response.transform(response_input)
    predicted_response_time = float(response_regressor.predict(response_input_scaled)[0])
    
    print(f"   - Response Time: {predicted_response_time:.1f} hours")
    
    print("\n✓ All models working correctly!")
    
except Exception as e:
    print(f"\n✗ Error during prediction: {e}")
    import traceback
    traceback.print_exc()
    exit(1)

print("\n" + "="*70)
print("All models are properly connected and working!")
print("="*70)
