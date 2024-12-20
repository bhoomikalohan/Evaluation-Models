import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import os

def load_data(file_path):
    if not os.path.exists(file_path):
        print(f"File not found at path: {file_path}")
        print(f"Current working directory: {os.getcwd()}")
        raise FileNotFoundError(f"File not found: {file_path}")
    return pd.read_csv(file_path)

def preprocess_data(data, target_column):
    print("Data shape before preprocessing:", data.shape)
    print("Columns in dataset:", data.columns.tolist())

    
    if target_column not in data.columns:
        raise KeyError(f"Target column '{target_column}' not found in dataset.")

    data = data.dropna()  
    print("Data shape after dropping NaNs:", data.shape)

    features = data.drop(columns=[target_column])
    target = data[target_column]

    
    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(features)

    
    X_train, X_test, y_train, y_test = train_test_split(
        features_scaled, target, test_size=0.3, random_state=42
    )
    return X_train, X_test, y_train, y_test