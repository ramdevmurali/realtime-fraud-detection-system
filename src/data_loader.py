import pandas as pd
import numpy as np
import joblib
import logging
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import pandas as pd
from src.utils import load_config, configure_logger

# Initialize Logger
configure_logger()

def load_and_preprocess_data():
    """
    Loads data, splits it, scales it, and saves the scaler.
    Returns: X_train, X_val, X_test, y_train, y_val, y_test
    """
    config = load_config()
    
    # 1. Load Data
    raw_path = config['data']['raw_file']
    logging.info(f"Loading data from {raw_path}...")
    try:
        df = pd.read_csv(raw_path)
    except FileNotFoundError:
        logging.error(f"File not found at {raw_path}. Did you run setup_data.py?")
        raise

    # 2. Split Features and Target
    X = df.drop('Class', axis=1)
    y = df['Class']

    # 3. Train / Test Split
    # We do this BEFORE scaling to prevent data leakage
    logging.info("Splitting data...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, 
        test_size=config['data']['test_size'], 
        random_state=config['data']['random_state']
    )

    # 4. Train / Validation Split
    X_train, X_val, y_train, y_val = train_test_split(
        X_train, y_train, 
        test_size=config['data']['val_size'], 
        random_state=config['data']['random_state']
    )

    # 5. Scaling
    # Crucial: Fit ONLY on training data, transform validation and test
    logging.info("Scaling features and saving scaler...")
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_val = scaler.transform(X_val)
    X_test = scaler.transform(X_test)

    # 6. Save the Scaler for the API
    scaler_path = config['data']['scaler_path']
    joblib.dump(scaler, scaler_path)
    logging.info(f"Scaler saved to {scaler_path}")

    logging.info(f"Data Shapes: Train={X_train.shape}, Val={X_val.shape}, Test={X_test.shape}")
    
    return X_train, X_val, X_test, y_train, y_val, y_test

if __name__ == "__main__":
    # Test run to verify it works
    load_and_preprocess_data()