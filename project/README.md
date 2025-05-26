# Hypertension Detection Project

This application provides hypertension risk prediction using machine learning. It consists of a Flask backend API serving a trained model and a React frontend for data collection and displaying predictions.

## Project Structure

```
/
├── backend/                  # Flask backend
│   ├── app.py                # Main Flask application
│   ├── requirements.txt      # Python dependencies
│   ├── model/                # Directory for storing the trained model
│   └── training/             # Directory containing model training code
│       └── model_training.py # Script for training and saving the model
└── frontend/                 # React frontend (src directory)
    ├── components/           # React components
    │   ├── PredictionForm.tsx  # Form for collecting user data
    │   ├── ResultDisplay.tsx   # Component to display prediction results
    │   ├── FeatureImportance.tsx # Visualization of feature importance
    │   ├── About.tsx           # Information about hypertension
    │   └── Footer.tsx          # Page footer
    └── App.tsx               # Main React component
```

## Setup Instructions

### Backend Setup

1. Navigate to the backend directory:
   ```
   cd backend
   ```

2. Create and activate a virtual environment (optional but recommended):
   ```
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. Install the required packages:
   ```
   pip install -r requirements.txt
   ```

4. Train the model (if not already trained):
   ```
   cd training
   python model_training.py
   ```

5. Start the Flask server:
   ```
   cd ..
   python app.py
   ```

The backend API will be available at http://localhost:5000.

### Frontend Setup

The frontend is already set up in the repository. To start the development server:

1. Ensure you're in the root directory:
   ```
   cd /path/to/project
   ```

2. Start the development server:
   ```
   npm run dev
   ```

The frontend will be available at the URL shown in the terminal (typically http://localhost:5173).

## API Endpoints

- `GET /api/health`: Check if the server is running and if the model is loaded
- `POST /api/predict`: Submit user data and get a prediction
- `GET /api/model-info`: Get information about the model and feature importance

## Technologies Used

- **Backend**: Flask, scikit-learn, XGBoost, pandas, numpy
- **Frontend**: React, TypeScript, Tailwind CSS, Lucide Icons
- **Machine Learning**: RandomForest, XGBoost, SMOTE for imbalanced data

## Features

- Input form with default values for all required fields
- Automatic BMI calculation
- Risk level determination with appropriate visual indicators
- Feature importance visualization
- Educational information about hypertension
- Responsive design for all device sizes