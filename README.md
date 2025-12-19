# EV54_Quantum - AI-Powered Disaster Prediction System

A full-stack machine learning application that predicts disaster impacts and provides emergency response recommendations using trained Random Forest models.

## 🚀 Features

- **Major Disaster Classification**: Predicts whether a disaster will be classified as major
- **Infrastructure Damage Assessment**: Estimates infrastructure damage index
- **Response Time Optimization**: Calculates optimal emergency response time
- **Resource Allocation**: Provides recommendations for personnel, equipment, and shelters
- **Evacuation Planning**: Determines evacuation needs and logistics
- **Interactive Web Interface**: Modern, responsive frontend for easy data input and visualization

## 📋 Prerequisites

- Python 3.8 or higher
- pip (Python package manager)
- Trained model files in `saved_models/` directory

## 🛠️ Installation

1. **Clone or navigate to the project directory**
   ```bash
   cd "d:\Envision 25-26\new ml model\EV54_Quantum"
   ```

2. **Create a virtual environment (recommended)**
   ```bash
   python -m venv venv
   
   # Activate on Windows
   venv\Scripts\activate
   
   # Activate on macOS/Linux
   source venv/bin/activate
   ```

3. **Install required packages**
   ```bash
   pip install -r requirements.txt
   ```

## 🏃 Running the Application

1. **Start the Flask backend server**
   ```bash
   python app.py
   ```

   The server will start at `http://localhost:5000`

2. **Access the web interface**
   - Open your browser and navigate to: `http://localhost:5000`
   - The frontend interface will load automatically

## 📁 Project Structure

```
EV54_Quantum/
├── app.py                                  # Flask backend API
├── requirements.txt                        # Python dependencies
├── README.md                               # This file
├── .gitignore                              # Git ignore file
├── model.ipynb                             # Main ML model training notebook
├── data_preprocessing.ipynb                # Data preprocessing notebook
├── Preprocessed data ENVISION ROUND 1.csv  # Training dataset
├── test_api.py                             # API testing script
├── test_models.py                          # Model testing script
├── train_params_quick.py                   # Quick training parameters script
├── saved_models/                           # Trained ML models
│   ├── disaster_classifier.pkl
│   ├── damage_regressor.pkl
│   ├── response_time_regressor.pkl
│   ├── scaler_disaster.pkl
│   ├── scaler_damage.pkl
│   ├── scaler_response.pkl
│   ├── label_encoder_disaster.pkl
│   ├── label_encoder_location.pkl
│   ├── label_encoder_aid.pkl
│   └── model_metadata.pkl
├── templates/                              # HTML templates
│   └── index.html
└── static/                                 # Static assets
    ├── style.css
    └── script.js
```

## 🎯 Using the Application

1. **Fill in the disaster information:**
   - Select disaster type (Earthquake, Flood, Hurricane, etc.)
   - Choose location
   - Enter latitude and longitude coordinates
   - Set severity level (1-10)
   - Input affected population
   - Enter estimated economic loss

2. **Generate prediction:**
   - Click "Generate Prediction" button
   - View comprehensive results including:
     - Major disaster probability
     - Infrastructure damage index
     - Recommended response time
     - Resource allocation recommendations
     - Evacuation plans (if needed)
     - Immediate action items

## 🔌 API Endpoints

### GET `/`
Serves the web interface

### GET `/api/status`
Health check endpoint

### GET `/api/disaster-types`
Returns available disaster types and locations

### POST `/api/predict`
Main prediction endpoint

**Request body:**
```json
{
  "disaster_type": "Earthquake",
  "location": "Japan",
  "latitude": 35.6762,
  "longitude": 139.6503,
  "severity_level": 9,
  "affected_population": 50000,
  "economic_loss": 15000000
}
```

**Response:**
```json
{
  "success": true,
  "predictions": {
    "is_major_disaster": true,
    "major_probability": 85.5,
    "predicted_damage_index": 0.823,
    "predicted_response_time_hours": 6.2
  },
  "emergency_response": {
    "priority": "CRITICAL",
    "alert_level": "LEVEL 5 - MAXIMUM ALERT",
    "resources": {...},
    "evacuation": {...},
    "action_items": [...]
  }
}
```

### GET `/api/model-info`
Returns model metadata and performance metrics

## 🧪 Testing

You can test the API using curl or any API client:

```bash
curl -X POST http://localhost:5000/api/predict \
  -H "Content-Type: application/json" \
  -d '{
    "disaster_type": "Earthquake",
    "location": "Japan",
    "latitude": 35.6762,
    "longitude": 139.6503,
    "severity_level": 9,
    "affected_population": 50000,
    "economic_loss": 15000000
  }'
```

## 📊 Model Information

- **Algorithm**: Random Forest Classifier/Regressor
- **Training Dataset**: Historical disaster data
- **Features**: 
  - Disaster type, location, geographic coordinates
  - Severity level, affected population
  - Economic impact, temporal features
- **Models**:
  - Major Disaster Classifier (Classification)
  - Infrastructure Damage Assessor (Regression)
  - Response Time Optimizer (Regression)

## 🚨 Troubleshooting

**Error: "No module named 'flask'"**
- Make sure you've installed all requirements: `pip install -r requirements.txt`

**Error: "Model files not found"**
- Ensure all model files are in the `saved_models/` directory
- Run the Jupyter notebook to train and save models if needed

**Error: "Port 5000 already in use"**
- Change the port in `app.py`: `app.run(port=5001)`

## 🔒 Security Notes

- This is a development server. For production:
  - Use a production WSGI server (gunicorn, waitress)
  - Implement proper authentication and authorization
  - Add input validation and rate limiting
  - Use HTTPS
  - Set up proper logging and monitoring

## 📝 License

This project is for educational and research purposes.

## 👥 Contributors

Machine Learning Model & Full-Stack Integration

## 🤝 Support

For issues or questions, please open an issue in the repository or contact the development team.

---

**Built with ❤️ using Flask, Scikit-learn, and Modern Web Technologies**
