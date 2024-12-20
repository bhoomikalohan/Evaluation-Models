import pandas as pd
from preprocess import load_data, preprocess_data
from train_models import load_and_train
from utils import evaluate_regression
import os

# Get the current script directory
script_dir = os.path.dirname(os.path.abspath(__file__))
print("Current working directory:", script_dir)

# File paths (absolute paths based on the script location)
data_path = os.path.join(script_dir, "../data/Fuel_cell_performance_data-Full.csv")
target_column = "Target3"   

# Load and preprocess data
data = load_data(data_path)
X_train, X_test, y_train, y_test = preprocess_data(data, target_column)

# Train models
load_and_train(X_train, y_train, X_test, y_test)

# Evaluate a single model 
from sklearn.linear_model import LinearRegression
model = LinearRegression().fit(X_train, y_train)
y_pred = model.predict(X_test)
results = evaluate_regression(y_test, y_pred)

# Save metrics
results_df = pd.DataFrame([results])
results_df.to_csv(os.path.join(script_dir, "../outputs/evaluation_metrics.csv"), index=False)
print("Evaluation metrics saved to ../outputs/evaluation_metrics.csv")
