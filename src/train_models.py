import os
import pandas as pd
import joblib
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor
from xgboost import XGBRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# Get the absolute path for the root directory (Evaluation models)
root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))  # Going up one level from the current folder (src)
output_dir = os.path.join(root_dir, "outputs")
model_dir = os.path.join(root_dir, "models")

# Ensure the necessary directories exist
os.makedirs(output_dir, exist_ok=True)
os.makedirs(model_dir, exist_ok=True)

def train_model(model, X_train, y_train, save_path):
    """
    Train the model and save it to the specified path.
    """
    model.fit(X_train, y_train)
    joblib.dump(model, save_path)

def evaluate_model(model, X_test, y_test):
    """
    Evaluate the model and return MSE, MAE, and R² scores.
    """
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    return mse, mae, r2

def load_and_train(X_train, y_train, X_test, y_test):
    """
    Train all models, evaluate them, and print their metrics.
    """
    models = {
        "linear_regression": LinearRegression(),
        "decision_tree": DecisionTreeRegressor(),
        "random_forest": RandomForestRegressor(),
        "svm": SVR(),
        "knn": KNeighborsRegressor(),
        "xgboost": XGBRegressor(),
        "neural_network": MLPRegressor(max_iter=500),
    }

    results = []
    best_model = None
    best_r2 = -float("inf")  # Initialize to a very low value to find the best model

    for name, model in models.items():
        print(f"\nTraining {name}...")
        model_save_path = os.path.join(model_dir, f"{name}.pkl")
        train_model(model, X_train, y_train, model_save_path)
        print(f"Model {name} saved to {model_save_path}")

        # Evaluate the model
        mse, mae, r2 = evaluate_model(model, X_test, y_test)
        print(f"Results for {name}:")
        print(f"  - MSE: {mse:.4f}")
        print(f"  - MAE: {mae:.4f}")
        print(f"  - R²: {r2:.4f}")
        
        results.append({"Model": name, "MSE": mse, "MAE": mae, "R2": r2})

        # Track the best model based on R² score
        if r2 > best_r2:
            best_r2 = r2
            best_model = name

    # Print a summary of all results
    print("\nSummary of Results:")
    for result in results:
        print(f"Model: {result['Model']}, MSE: {result['MSE']:.4f}, MAE: {result['MAE']:.4f}, R²: {result['R2']:.4f}")
    
    print(f"\nBest model based on R² score: {best_model} with R²: {best_r2:.4f}")

    # Save results to CSV file
    results_df = pd.DataFrame(results)
    results_df.to_csv(os.path.join(output_dir, "evaluation_metrics.csv"), index=False, mode='a', header=not os.path.exists(os.path.join(output_dir, "evaluation_metrics.csv")))
    print("\nAll results saved to ../outputs/evaluation_metrics.csv")
