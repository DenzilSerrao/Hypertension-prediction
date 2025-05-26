import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_auc_score
import xgboost as xgb
from imblearn.over_sampling import SMOTE
import joblib
import os

# Ensure the model directory exists
os.makedirs("../model", exist_ok=True)

# Step 2: Load the data
try:
    df = pd.read_csv('D:/Programs/Python/ML-1/project/backend/training/personalized_medication_dataset.csv')
except FileNotFoundError:
    print("Dataset file not found. Using sample data for demonstration.")

# Step 3: Create target variable (Hypertension as target)
df['HBP'] = df['Chronic_Conditions'].apply(lambda x: 1 if 'Hypertension' in str(x) else 0)

# Step 4: Feature Engineering (Add new features)
def bmi_category(bmi):
    if bmi < 18.5: return 'Underweight'
    elif bmi < 25: return 'Normal'
    elif bmi < 30: return 'Overweight'
    else: return 'Obese'

def age_group(age):
    if age < 30: return 'Young'
    elif age < 60: return 'Middle-aged'
    else: return 'Senior'

df['BMI_Category'] = df['BMI'].apply(bmi_category)
df['Age_Group'] = df['Age'].apply(age_group)

# Step 5: Select features
features = ['Age', 'Gender', 'Weight_kg', 'Height_cm', 'BMI', 'Genetic_Disorders', 'BMI_Category', 'Age_Group']
X = df[features].copy()
y = df['HBP']

# Save feature information for inference
feature_info = {
    'categorical_cols': ['Gender', 'Genetic_Disorders', 'BMI_Category', 'Age_Group'],
    'numerical_cols': ['Age', 'Weight_kg', 'Height_cm', 'BMI'],
    'derived_features': {
        'BMI_Category': {
            'function': 'bmi_category',
            'input': 'BMI'
        },
        'Age_Group': {
            'function': 'age_group',
            'input': 'Age'
        }
    }
}

# Step 6: Encode categorical features
encoders = {}
categorical_cols = feature_info['categorical_cols']
for col in categorical_cols:
    le = LabelEncoder()
    X[col] = le.fit_transform(X[col].astype(str))
    encoders[col] = le

# Step 7: Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Step 8: Feature Scaling
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Step 9: Apply SMOTE to balance the training data
smote = SMOTE(random_state=42)
X_train_resampled, y_train_resampled = smote.fit_resample(X_train_scaled, y_train)

# Step 10: Hyperparameter tuning using GridSearchCV (Random Forest)
param_grid_rf = {
    'n_estimators': [100, 200],
    'max_depth': [5, 10],
    'min_samples_split': [2, 5]
}

grid_search_rf = GridSearchCV(
    RandomForestClassifier(class_weight='balanced', random_state=42),
    param_grid_rf,
    cv=3,
    scoring='f1_macro',
    n_jobs=-1
)

# Step 11: Train Random Forest with GridSearchCV
print("Training Random Forest model...")
grid_search_rf.fit(X_train_resampled, y_train_resampled)
best_rf = grid_search_rf.best_estimator_

# Step 12: Evaluate Random Forest
y_pred_rf_probs = best_rf.predict_proba(X_test_scaled)[:, 1]
roc_auc_rf = roc_auc_score(y_test, y_pred_rf_probs)
accuracy = accuracy_score(y_test, best_rf.predict(X_test_scaled))
print("Random Forest Accuracy:", accuracy)
print("Random Forest ROC AUC:", roc_auc_rf)
print("Random Forest Classification Report:\n", classification_report(y_test, best_rf.predict(X_test_scaled)))
print("Random Forest Confusion Matrix:\n", confusion_matrix(y_test, best_rf.predict(X_test_scaled)))

# Step 13: Train XGBoost Classifier
print("Training XGBoost model...")
scale_pos_weight = (y_train == 0).sum() / (y_train == 1).sum()

xgb_model = xgb.XGBClassifier(
    eval_metric='logloss',
    random_state=42,
    scale_pos_weight=scale_pos_weight
)
xgb_model.fit(X_train_resampled, y_train_resampled)

# Evaluate XGBoost
y_pred_xgb = xgb_model.predict(X_test_scaled)
y_pred_xgb_probs = xgb_model.predict_proba(X_test_scaled)[:, 1]
roc_auc_xgb = roc_auc_score(y_test, y_pred_xgb_probs)

print("XGBoost ROC AUC:", roc_auc_xgb)
print("XGBoost Classification Report:\n", classification_report(y_test, y_pred_xgb))
print("XGBoost Confusion Matrix:\n", confusion_matrix(y_test, y_pred_xgb))

# Step 14: Save the trained model and preprocessing objects
model_path = os.path.join(os.path.dirname(__file__), "../model/saved_model.pkl")
print(f"Saving model to {model_path}")

# Save the model and preprocessing objects
model_data = {
    'model': xgb_model if roc_auc_xgb > roc_auc_rf else best_rf,
    'scaler': scaler,
    'encoders': encoders,
    'feature_info': feature_info,
    'model_type': 'xgboost' if roc_auc_xgb > roc_auc_rf else 'random_forest',
    'feature_names': list(X.columns),
    'performance': {
        'roc_auc': max(roc_auc_xgb, roc_auc_rf)
    }
}

joblib.dump(model_data, model_path)

# Save the model and preprocessing objects
model_data = {
    'model': xgb_model if roc_auc_xgb > roc_auc_rf else best_rf,
    'scaler': scaler,
    'encoders': encoders,
    'feature_info': feature_info,
    'model_type': 'xgboost' if roc_auc_xgb > roc_auc_rf else 'random_forest',
    'feature_names': list(X.columns),
    'performance': {
        'roc_auc': max(roc_auc_xgb, roc_auc_rf)
    }
}

joblib.dump(model_data, model_path)
print("Model saved successfully!")

# Step 15: Feature Importance
if roc_auc_xgb > roc_auc_rf:
    print("\nUsing XGBoost model for deployment")
    importances = xgb_model.feature_importances_
else:
    print("\nUsing Random Forest model for deployment")
    importances = best_rf.feature_importances_

feature_names = X.columns

print("\nFeature Importances:")
for name, importance in sorted(zip(feature_names, importances), key=lambda x: x[1], reverse=True):
    print(f"{name}: {importance:.4f}")