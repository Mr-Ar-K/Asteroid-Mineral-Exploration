# model_training.py

import xgboost as xgb
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score
import joblib
from asteroid_mining_dashboard.utils import calculate_final_aps  # <-- IMPORT from your new utils file

# Load your dataset
# Replace 'your_dataset.csv' with the path to your actual data file
df = pd.read_csv('your_dataset.csv')

# --- Define features and target ---
X = df[['mission_accessibility_score', 'resource_confidence_score', 'albedo', 'e', 'i', 'H']]
y = df['spectral_type_encoded']

# Split data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# --- Hyperparameter Tuning Setup ---
param_grid = {
    'max_depth': [3, 5, 7],
    'learning_rate': [0.01, 0.1, 0.2],
    'n_estimators': [100, 200, 300],
    'subsample': [0.8, 1.0]
}

# Initialize the XGBoost model
xgb_model = xgb.XGBClassifier(
    objective='multi:softprob',
    eval_metric='mlogloss',
    use_label_encoder=False
)

# Set up GridSearchCV to find the best parameters
grid_search = GridSearchCV(
    estimator=xgb_model,
    param_grid=param_grid,
    cv=3,
    n_jobs=-1,
    verbose=2
)

# --- Train the model using the best found parameters ---
print("Starting hyperparameter tuning...")
grid_search.fit(X_train, y_train)

print(f"Best parameters found: {grid_search.best_params_}")

# Get the best model from the search
best_model = grid_search.best_estimator_

# --- Evaluate and Save ---
y_pred = best_model.predict(X_test)
print(f"Final Model Accuracy: {accuracy_score(y_test, y_pred) * 100:.2f}%")

joblib.dump(best_model, 'final_asteroid_model.joblib')

# --- Apply the final scoring to your full dataset ---
df['predicted_class_encoded'] = best_model.predict(df[X.columns])

# You will need to map the encoded class back to 'M', 'C', 'S'
# df['predicted_class'] = df['predicted_class_encoded'].map(your_encoding_map)

df['aps'] = df.apply(calculate_final_aps, axis=1)
df.to_csv('final_dashboard_data.csv', index=False)
print("Final data with APS scores saved.")
