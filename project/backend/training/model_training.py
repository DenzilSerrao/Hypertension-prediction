import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold, cross_val_score
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
df = pd.read_csv('./personalized_medication_dataset.csv')

# Step 3: Create target variable
df['HBP'] = df['Chronic_Conditions'].apply(lambda x: 1 if 'Hypertension' in str(x) else 0)

# Step 4: Feature Engineering
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

# Step 5: Feature selection
features = ['Age', 'Gender', 'Weight_kg', 'Height_cm', 'BMI', 'Genetic_Disorders', 'BMI_Category', 'Age_Group']
X = df[features].copy()
y = df['HBP']

# Save feature info for inference
feature_info = {
    'categorical_cols': ['Gender', 'Genetic_Disorders', 'BMI_Category', 'Age_Group'],
    'numerical_cols': ['Age', 'Weight_kg', 'Height_cm', 'BMI'],
    'derived_features': {
        'BMI_Category': {'function': 'bmi_category', 'input': 'BMI'},
        'Age_Group': {'function': 'age_group', 'input': 'Age'}
    }
}

# Step 6: Encode categorical features
encoders = {}
for col in feature_info['categorical_cols']:
    le = LabelEncoder()
    X[col] = le.fit_transform(X[col].astype(str))
    encoders[col] = {
        'classes': list(le.classes_),
        'encoder': le
    }

# Step 7: Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Step 8: Scaling
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Step 9: SMOTE for imbalance
smote = SMOTE(random_state=42)
X_train_resampled, y_train_resampled = smote.fit_resample(X_train_scaled, y_train)

# Logging class distribution
print("Original class distribution:\n", y_train.value_counts())
print("Resampled class distribution:\n", pd.Series(y_train_resampled).value_counts())

# Step 10: Grid Search for Random Forest
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

print("Training Random Forest model...")
grid_search_rf.fit(X_train_resampled, y_train_resampled)
best_rf = grid_search_rf.best_estimator_

cv_rf_scores = cross_val_score(best_rf, X_train_resampled, y_train_resampled, cv=5, scoring='roc_auc')
mean_cv_rf = np.mean(cv_rf_scores)
print("Random Forest CV ROC AUC:", mean_cv_rf)

# Step 11: XGBoost model
scale_pos_weight = (y_train == 0).sum() / (y_train == 1).sum()
xgb_model = xgb.XGBClassifier(
    eval_metric='logloss',
    use_label_encoder=False,
    random_state=42,
    scale_pos_weight=scale_pos_weight
)

print("Training XGBoost model...")
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
cv_xgb_scores = cross_val_score(xgb_model, X_train_resampled, y_train_resampled, cv=cv, scoring='roc_auc')
mean_cv_xgb = np.mean(cv_xgb_scores)
print("XGBoost CV ROC AUC:", mean_cv_xgb)

xgb_model.fit(X_train_resampled, y_train_resampled)

# Step 12: Select best model
selected_model = xgb_model if mean_cv_xgb > mean_cv_rf else best_rf
selected_model_type = 'xgboost' if mean_cv_xgb > mean_cv_rf else 'random_forest'
selected_cv_score = max(mean_cv_xgb, mean_cv_rf)

model_path = os.path.join(os.path.dirname(__file__), "../model/saved_model.pkl")
print(f"Saving model ({selected_model_type}) with ROC AUC {selected_cv_score:.4f} to {model_path}")

# Step 13: Save model and metadata
model_data = {
    'model': selected_model,
    'scaler': scaler,
    'encoders': {k: {'classes': v['classes']} for k, v in encoders.items()},
    'feature_info': feature_info,
    'model_type': selected_model_type,
    'feature_names': features,
    'performance': {
        'cv_roc_auc': selected_cv_score
    }
}

joblib.dump(model_data, model_path)
print("Model saved successfully!")

# Step 14: Print Feature Importances
print(f"\nUsing {selected_model_type.upper()} model for deployment")
importances = selected_model.feature_importances_
print("Feature Importances:")
for name, score in sorted(zip(features, importances), key=lambda x: x[1], reverse=True):
    print(f"{name}: {score:.4f}")
