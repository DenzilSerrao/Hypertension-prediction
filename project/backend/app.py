import logging  # new import
from flask import Flask, request, jsonify
from flask_cors import CORS
import joblib
import os
import numpy as np
import pandas as pd

# Configure logging to output errors with stack traces
logging.basicConfig(level=logging.ERROR, format='%(asctime)s %(levelname)s %(message)s')

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

# Load the model
MODEL_PATH = os.path.join(os.path.dirname(__file__), 'model/saved_model.pkl')

def load_model():
    try:
        return joblib.load(MODEL_PATH)
    except (FileNotFoundError, IOError) as e:
        logging.exception("Failed to load model")
        return None

model_data = load_model()

# Define helper functions used by the model
def bmi_category(bmi):
    if bmi < 18.5: 
        return 'Underweight'
    elif bmi < 25: 
        return 'Normal'
    elif bmi < 30: 
        return 'Overweight'
    else: 
        return 'Obese'

def age_group(age):
    if age < 30: return 'Young'
    elif age < 60: return 'Middle-aged'
    else: return 'Senior'

@app.route('/api/health', methods=['GET'])
def health_check():
    if model_data is not None:
        return jsonify({
            'status': 'healthy',
            'model_loaded': True,
            'model_type': model_data.get('model_type', 'unknown')
        })
    else:
        return jsonify({
            'status': 'unhealthy',
            'model_loaded': False,
            'error': 'Model not found or could not be loaded'
        }), 500

@app.route('/api/predict', methods=['POST'])
def predict():
    if model_data is None:
        return jsonify({
            'error': 'Model not loaded. Please train the model first.'
        }), 500
    
    try:
        # Get data from request
        data = request.json
        
        # Create a DataFrame with a single row
        input_df = pd.DataFrame([data])
        
        # Add derived features
        if 'BMI' in input_df.columns:
            input_df['BMI_Category'] = input_df['BMI'].apply(bmi_category)
        else:
            # Calculate BMI if not provided
            if 'Weight_kg' in input_df.columns and 'Height_cm' in input_df.columns:
                input_df['BMI'] = input_df['Weight_kg'] / ((input_df['Height_cm']/100) ** 2)
                input_df['BMI_Category'] = input_df['BMI'].apply(bmi_category)
            else:
                return jsonify({'error': 'BMI or Weight and Height must be provided'}), 400
        
        input_df['Age_Group'] = input_df['Age'].apply(age_group)
        
        # Select and order features to match the model's expected input
        feature_names = model_data['feature_names']
        X = input_df[feature_names].copy()
        
        # Encode categorical features
        encoders = model_data['encoders']
        for col, encoder in encoders.items():
            if col in X.columns:
                # Handle unseen categories
                try:
                    X[col] = encoder.transform(X[col].astype(str))
                except ValueError:
                    # If category not seen during training, use a default value (e.g., most common)
                    X[col] = encoder.transform([encoder.classes_[0]])[0]
        
        # Scale features
        scaler = model_data['scaler']
        X_scaled = scaler.transform(X)
        
        # Make prediction
        model = model_data['model']
        prediction_proba = model.predict_proba(X_scaled)[0][1]  # Probability of class 1
        prediction = 1 if prediction_proba >= 0.5 else 0
        
        # Create response
        result = {
            'prediction': int(prediction),
            'probability': float(prediction_proba),
            'risk_level': get_risk_level(prediction_proba),
            'input_data': data
        }
        print("Prediction result:", result)  # Debugging output
        return jsonify(result)
    
    except Exception as e:
        logging.exception("Error during prediction")
        return jsonify({'error': str(e)}), 500

def get_risk_level(probability):
    if probability < 0.2:
        return 'Low'
    elif probability < 0.5:
        return 'Moderate'
    elif probability < 0.8:
        return 'High'
    else:
        return 'Very High'

@app.route('/api/model-info', methods=['GET'])
def model_info():
    try:
        if model_data is None:
            raise ValueError("Model not loaded")
        # Get feature importance or other model info
        model = model_data['model']
        feature_names = model_data['feature_names']
        
        try:
            importances = model.feature_importances_
            feature_importance = [
                {'feature': name, 'importance': float(importance)}
                for name, importance in sorted(zip(feature_names, importances), 
                                             key=lambda x: x[1], 
                                             reverse=True)
            ]
        except AttributeError:
            feature_importance = []
        
        return jsonify({
            'model_type': model_data.get('model_type', 'unknown'),
            'performance': model_data.get('performance', {}),
            'feature_importance': feature_importance,
            'features': feature_names
        })
    except Exception as e:
        logging.exception("Error fetching model info")
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)