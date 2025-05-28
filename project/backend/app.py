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
        logging.error("Model not loaded.")
        return jsonify({'error': 'Model not loaded. Please train the model first.'}), 500

    try:
        # Step 1: Parse incoming JSON
        data = request.json
        logging.info("Received data: %s", data)

        input_df = pd.DataFrame([data])

        # Step 2: Derived features
        if 'BMI' not in input_df.columns:
            if 'Weight_kg' in input_df and 'Height_cm' in input_df:
                input_df['BMI'] = input_df['Weight_kg'] / ((input_df['Height_cm'] / 100) ** 2)
            else:
                return jsonify({'error': 'Provide BMI or (Weight_kg and Height_cm).'}), 400

        input_df['BMI_Category'] = input_df['BMI'].apply(bmi_category)
        input_df['Age_Group'] = input_df['Age'].apply(age_group)
        logging.info("Derived features added: BMI_Category, Age_Group")

        # Step 3: Select & order features
        feature_names = model_data['feature_names']
        X = input_df[feature_names].copy()
        logging.info("Selected features: %s", X.columns.tolist())

        # Step 4: Encode categorical features
        for col, le in model_data['encoders'].items():
            if col in X:
                original_vals = X[col].astype(str).tolist()
                safe_vals = []
                for val in original_vals:
                    if val in le.classes_:
                        safe_vals.append(val)
                    else:
                        logging.warning(f"Unseen category '{val}' in column '{col}', using fallback '{le.classes_[0]}'")
                        safe_vals.append(le.classes_[0])
                X[col] = le.transform(safe_vals)
        logging.info("Categorical encoding complete")

        # Step 5: Scaling
        scaler = model_data['scaler']
        X_scaled = scaler.transform(X)
        logging.info("Feature scaling applied")

        # Step 6: Predict
        model = model_data['model']
        prediction_proba = model.predict_proba(X_scaled)[0][1]
        prediction = 1 if prediction_proba >= 0.5 else 0
        logging.info(f"Prediction: {prediction}, Probability: {prediction_proba}")

        # Optional: Compute full metrics (if ground truth 'Outcome' provided)
        metrics = {}
        if 'Outcome' in input_df.columns:
            from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
            y_true = [input_df['Outcome'].values[0]]
            y_pred = [prediction]
            y_prob = [prediction_proba]
            metrics = {
                'accuracy': accuracy_score(y_true, y_pred),
                'precision': precision_score(y_true, y_pred, zero_division=0),
                'recall': recall_score(y_true, y_pred, zero_division=0),
                'f1_score': f1_score(y_true, y_pred, zero_division=0),
                'roc_auc': roc_auc_score(y_true, y_prob)
            }
            logging.info("Evaluation metrics computed: %s", metrics)

        # Step 7: Response
        result = {
            'prediction': int(prediction),
            'probability': float(prediction_proba),
            'risk_level': get_risk_level(prediction_proba),
            'input_data': input_df.iloc[0].to_dict(),
            'metrics': metrics if metrics else 'N/A'
        }
        print("Final prediction result:", result)
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