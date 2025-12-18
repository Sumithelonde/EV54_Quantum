import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.preprocessing import StandardScaler

print('Training parameter prediction models...')

# Load data
df = pd.read_csv('Preprocessed data ENVISION ROUND 1.csv')

# Create temporal features
df['date'] = pd.to_datetime(df['date'])
df['month'] = df['date'].dt.month
df['week'] = df['date'].dt.isocalendar().week % 4 + 1
df['quarter'] = (df['month'] - 1) // 3 + 1
df['is_summer'] = df['month'].isin([6,7,8]).astype(int)
df['is_winter'] = df['month'].isin([12,1,2]).astype(int)

# Load encoders
le_disaster = joblib.load('saved_models/label_encoder_disaster.pkl')
le_location = joblib.load('saved_models/label_encoder_location.pkl')

# Encode
df['disaster_encoded'] = le_disaster.transform(df['disaster_type'])
df['location_encoded'] = le_location.transform(df['location'])

# Prepare features
X = df[['disaster_encoded','location_encoded','latitude','longitude','month','week','quarter','is_summer','is_winter']].values

# Scale
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
X_train, X_test = train_test_split(X_scaled, test_size=0.2, random_state=42)

# Train severity model
print('Training severity model...')
y_sev = df['severity_level'].values
y_sev_train, y_sev_test = train_test_split(y_sev, test_size=0.2, random_state=42)
sev_model = GradientBoostingRegressor(n_estimators=200, learning_rate=0.1, max_depth=5, random_state=42)
sev_model.fit(X_train, y_sev_train)
joblib.dump(sev_model, 'saved_models/severity_model.pkl')
print('✓ Severity model saved')

# Train population model
print('Training population model...')
y_pop = df['affected_population'].values
y_pop_train, y_pop_test = train_test_split(y_pop, test_size=0.2, random_state=42)
pop_model = RandomForestRegressor(n_estimators=200, max_depth=15, random_state=42, n_jobs=-1)
pop_model.fit(X_train, y_pop_train)
joblib.dump(pop_model, 'saved_models/population_model.pkl')
print('✓ Population model saved')

# Train economic loss model
print('Training economic loss model...')
y_loss = df['estimated_economic_loss_usd'].values
y_loss_train, y_loss_test = train_test_split(y_loss, test_size=0.2, random_state=42)
loss_model = GradientBoostingRegressor(n_estimators=200, learning_rate=0.1, max_depth=7, random_state=42)
loss_model.fit(X_train, y_loss_train)
joblib.dump(loss_model, 'saved_models/economic_loss_model.pkl')
print('✓ Economic loss model saved')

# Save scaler
joblib.dump(scaler, 'saved_models/scaler_parameters.pkl')
print('✓ Scaler saved')

print('\n✓ All parameter models trained and saved successfully!')
