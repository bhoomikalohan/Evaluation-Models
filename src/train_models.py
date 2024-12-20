import joblib
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor
from xgboost import XGBRegressor
from sklearn.neural_network import MLPRegressor

def train_model(model, X_train, y_train, save_path):
    model.fit(X_train, y_train)
    joblib.dump(model, save_path)

def load_and_train(X_train, y_train):
    models = {
        "linear_regression": LinearRegression(),
        "decision_tree": DecisionTreeRegressor(),
        "random_forest": RandomForestRegressor(),
        "svm": SVR(),
        "knn": KNeighborsRegressor(),
        "xgboost": XGBRegressor(),
        "neural_network": MLPRegressor(max_iter=500),
    }

    for name, model in models.items():
        print(f"Training {name}...")
        train_model(model, X_train, y_train, f"../models/{name}.pkl")
        print(f"Model {name} saved to ../models/{name}.pkl")
